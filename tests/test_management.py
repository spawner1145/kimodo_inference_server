"""Management contracts: persistence, safe apply and isolated downloads."""
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from kimodo_server.api import create_app
from kimodo_server.config import load_config
from kimodo_server.downloads import DownloadManager, DownloadRequest, download_worker, weights_ready
from kimodo_server.management import atomic_save
from test_api import FakeRuntime


class CaptureRuntime(FakeRuntime):
    def generate(self,request,*args):
        self.request = request
        return super().generate(request,*args)


class ManagementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root/'config.json'
        atomic_save(self.path,dict(models_dir='models',data_dir='data',
            models={'soma-fp32':dict(label='Test',path='motion',repo_id='test/motion')},
            encoders={'awq':dict(base='base',mntp='mntp',supervised='sup',repo_id='test/base',
                                mntp_repo_id='test/mntp',supervised_repo_id='test/sup')},
            inference=dict(steps=37,seed=17,transition_frames=12)))
        self.config = load_config(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_config_persistence_validation_auth_and_ui(self):
        app = create_app(self.config,FakeRuntime,token='secret',config_path=self.path)
        with TestClient(app) as c:
            self.assertEqual(c.get('/').status_code,200)
            self.assertIn('推理控制台',c.get('/').text)
            self.assertEqual(c.get('/ui/app.js').status_code,200)
            self.assertEqual(c.get('/ui/private.py').status_code,404)
            self.assertEqual(c.get('/v1/admin/config').status_code,401)
            c.headers['Authorization']='Bearer secret'
            data=c.get('/v1/admin/config').json()
            self.assertEqual(data['saved']['models_dir'],'models')
            self.assertEqual(data['active']['port'],6600)
            self.assertFalse(data['pending_restart'])
            saved=data['saved'];saved['port']=6601;saved['inference']['steps']=25
            self.assertEqual(c.put('/v1/admin/config',json=saved).status_code,200)
            self.assertEqual(load_config(self.path).port,6601)
            self.assertEqual(load_config(self.path).models_dir,self.root/'models')
            data=c.get('/v1/admin/config').json()
            self.assertTrue(data['pending_restart']);self.assertEqual(data['active']['port'],6600)
            saved['port']=99999
            self.assertEqual(c.put('/v1/admin/config',json=saved).status_code,422)
            self.assertEqual(load_config(self.path).port,6601)
            self.assertEqual(c.post('/v1/admin/apply',headers={'Origin':'https://evil.invalid'}).status_code,403)
            self.assertEqual(c.post('/v1/admin/apply').status_code,409)
            self.assertEqual(c.get('/v1/admin/status').status_code,200)

    def test_defaults_only_apply_to_omitted_fields_and_are_revalidated(self):
        app=create_app(self.config,CaptureRuntime,token='')
        with TestClient(app) as c:
            for overrides,expected in [({},37),({'steps':5,'seed':42},5)]:
                j=c.post('/v1/jobs',json={'prompts':['ok'],'durations':[1],**overrides}).json()
                for _ in range(200):
                    if c.get('/v1/jobs/'+j['id']).json()['state']=='succeeded':break
                    time.sleep(.01)
                req=app.state.manager.runtime.request
                self.assertEqual(req.steps,expected);self.assertEqual(req.seed,overrides.get('seed',17))
            # Durations were valid with the schema's default transition but
            # become invalid with the configured default. Must be 422, not 500.
            response=c.post('/v1/jobs',json={'prompts':['a','b'],'durations':[.2,.2]})
            self.assertEqual(response.status_code,422)

    def test_apply_rejects_busy_then_requests_restart_and_freezes_queue(self):
        callbacks=[]
        app=create_app(self.config,FakeRuntime,token='',config_path=self.path,restart=lambda:callbacks.append(True))
        with TestClient(app) as c:
            job=c.post('/v1/jobs',json={'prompts':['wait'],'durations':[1]}).json()
            self.assertEqual(c.post('/v1/admin/apply').status_code,409)
            self.assertEqual(c.post('/v1/admin/downloads',json={'kind':'model','profile':'soma-fp32'}).status_code,409)
            c.post('/v1/jobs/'+job['id']+'/cancel')
            for _ in range(200):
                if c.get('/v1/jobs/'+job['id']).json()['state']=='cancelled':break
                time.sleep(.01)
            with socket.socket() as occupied:
                occupied.bind(('127.0.0.1',0));occupied.listen()
                doc=json.loads(self.path.read_text());doc['port']=occupied.getsockname()[1]
                c.put('/v1/admin/config',json=doc)
                self.assertEqual(c.post('/v1/admin/apply').status_code,409)
            doc['port']=6600;c.put('/v1/admin/config',json=doc)
            self.assertEqual(c.post('/v1/admin/apply').status_code,200)
            self.assertEqual(c.post('/v1/jobs',json={'prompts':['ok'],'durations':[1]}).status_code,503)
            time.sleep(.9);self.assertEqual(callbacks,[True])

    def test_download_process_online_environment_and_secret_redaction(self):
        class Process:
            stdout=io.StringIO(json.dumps({'message':'下载失败 hf_secret'})+'\n')
            def wait(self):return 1
        manager=DownloadManager()
        with patch.dict(os.environ,{'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}),patch(
            'kimodo_server.downloads.subprocess.Popen',return_value=Process()) as popen:
            request=DownloadRequest(kind='model',profile='soma-fp32',token='hf_secret')
            manager.start(self.config,request);manager.thread.join(2)
            env=popen.call_args.kwargs['env']
            self.assertNotIn('HF_HUB_OFFLINE',env);self.assertNotIn('TRANSFORMERS_OFFLINE',env)
            self.assertEqual(os.environ['HF_HUB_OFFLINE'],'1')
            self.assertEqual(env['HF_TOKEN'],'hf_secret')
            self.assertNotIn('hf_secret',json.dumps(manager.snapshots()))
            self.assertNotIn('hf_secret',str(popen.call_args.args))
            self.assertEqual(manager.snapshots()[0]['state'],'failed')
        manager.close()

    def test_download_api_validation_and_cancellation(self):
        from fastapi import HTTPException
        from kimodo_server.downloads import download_items
        profile=self.config.models['soma-fp32'];profile.path='../outside'
        with self.assertRaises(HTTPException):download_items(self.config,'model','soma-fp32')
        profile.path='motion'
        manager=DownloadManager()
        gate=__import__('threading').Event()
        class Output:
            def __iter__(self):gate.wait(2);return iter(())
            def close(self):pass
        class Process:
            stdout=Output()
            def wait(self):return -15
            def terminate(self):gate.set()
        with patch('kimodo_server.downloads.subprocess.Popen',return_value=Process()):
            j=manager.start(self.config,DownloadRequest(kind='model',profile='soma-fp32'))
            with self.assertRaises(HTTPException):manager.start(self.config,DownloadRequest(kind='model',profile='soma-fp32'))
            manager.cancel(j['id']);manager.close()
            self.assertEqual(manager.snapshots()[0]['state'],'cancelled')
        with self.assertRaises(HTTPException):
            manager.start(self.config,DownloadRequest(kind='model',profile='soma-fp32',endpoint='http://example.com',token='secret'))

    def test_worker_completion_marker_and_missing_shards(self):
        dest=self.root/'weights';dest.mkdir()
        (dest/'config.json').write_text('{}')
        (dest/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'a.safetensors','b':'b.safetensors'}}))
        (dest/'a.safetensors').write_bytes(b'weights')
        self.assertFalse(weights_ready(dest))

        def complete(**kwargs):
            self.assertTrue((dest/'.kimodo-downloading').exists())
            (dest/'b.safetensors').write_bytes(b'weights')
        items=[{'repo':'test/repo','path':str(dest)}]
        with patch('huggingface_hub.snapshot_download',side_effect=complete):download_worker(items,'https://example.com')
        self.assertTrue(weights_ready(dest))
        with patch('huggingface_hub.snapshot_download',side_effect=RuntimeError('network failed')):
            with self.assertRaises(SystemExit):download_worker(items,'https://example.com')
        self.assertFalse(weights_ready(dest))

    def test_example_config_bootstrap_preserves_local_settings(self):
        example=self.root/'config.example.json'
        example.write_bytes(self.path.read_bytes())
        self.path.unlink()
        self.assertEqual(load_config(self.path).port,6600)
        doc=json.loads(self.path.read_text());doc['port']=6602
        atomic_save(self.path,doc)
        self.assertEqual(load_config(self.path).port,6602)
        with self.assertRaises(FileNotFoundError):load_config(self.root/'custom.json')

    def test_modelscope_routing_and_failed_download_marker(self):
        from kimodo_server.downloads import download_items
        profile=self.config.models['soma-fp32']
        self.assertEqual(download_items(self.config,'model','soma-fp32')[0]['source'],'huggingface')
        profile.download_source='modelscope'
        profile.repo_id='spawner/Kimodo-SOMA-RP-v1.2-int8-convrot'
        items=download_items(self.config,'model','soma-fp32')
        dest=Path(items[0]['path'])
        with patch.dict(os.environ,{'HF_TOKEN':'hf_wrong','MODELSCOPE_API_TOKEN':'ms_right'}),patch(
                'modelscope.hub.snapshot_download.snapshot_download') as ms,patch(
                'huggingface_hub.snapshot_download') as hf:
            download_worker(items,'https://hf.example.com')
            hf.assert_not_called()
            self.assertEqual(ms.call_args.kwargs['model_id'],profile.repo_id)
            self.assertEqual(ms.call_args.kwargs['endpoint'],'https://www.modelscope.cn')
            self.assertEqual(ms.call_args.kwargs['token'],'ms_right')
            self.assertFalse((dest/'.kimodo-downloading').exists())
            ms.side_effect=RuntimeError('network failed')
            with self.assertRaises(SystemExit):download_worker(items,'https://hf.example.com')
            self.assertTrue((dest/'.kimodo-downloading').exists())

    def test_modelscope_credentials_are_isolated_in_subprocess(self):
        self.config.models['soma-fp32'].download_source='modelscope'
        class Process:
            stdout=io.StringIO(json.dumps({'message':'下载失败 ms_secret'})+'\n')
            def wait(self):return 1
        manager=DownloadManager()
        with patch.dict(os.environ,{'HF_TOKEN':'hf_wrong','MODELSCOPE_API_TOKEN':'ms_fallback'}),patch(
                'kimodo_server.downloads.subprocess.Popen',return_value=Process()) as popen:
            manager.start(self.config,DownloadRequest(kind='model',profile='soma-fp32',
                          token='hf_unused',modelscope_token='ms_secret'))
            manager.thread.join(2)
            env=popen.call_args.kwargs['env']
            self.assertNotIn('HF_TOKEN',env)
            self.assertEqual(env['MODELSCOPE_API_TOKEN'],'ms_secret')
            self.assertNotIn('ms_secret',str(popen.call_args.args))
            self.assertNotIn('ms_secret',json.dumps(manager.snapshots()))
        manager.close()



if __name__=='__main__':unittest.main()
