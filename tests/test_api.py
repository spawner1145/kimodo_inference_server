"""HTTP contracts and queue lifecycle; no model weights or GPU required."""
import json
from pathlib import Path
import tempfile
import time
import unittest

from fastapi.testclient import TestClient

from kimodo_server.api import create_app
from kimodo_server.config import ServerConfig
from kimodo_server.runtime import Cancelled


class FakeRuntime:
    def __init__(self,config):
        self.loaded = None
        self.closed = False

    def profiles(self):
        return [{'id':'soma-fp32','label':'test','available':True}]

    def encoders(self):
        return [{'id':'awq','label':'test','available':True}]

    def load(self,name,progress):
        self.loaded = name

    def generate(self,request,event,progress,path):
        self.loaded = request.model
        if request.prompts[0]=='fail':
            raise ValueError('deliberate test error')
        if request.prompts[0]=='wait':
            for _ in range(500):
                if event.wait(0.01):
                    raise Cancelled()
                progress('test running',0.2)
        path.write_bytes(b'HIERARCHY\nMOTION\nFrames: 30\n')
        return {'frames':30,'model':request.model}

    def close(self):
        self.closed = True


class APIContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.config = ServerConfig(models_dir=root,data_dir=root/'data',queue_size=1,
            encoders={'awq':{'kind':'awq_llm2vec','base':'base','mntp':'mntp','supervised':'sup'}},
            models={'soma-fp32':{'label':'test','path':'model'}})

    def tearDown(self):
        self.temp.cleanup()

    def body(self,prompt='ok',**kwargs):
        return {'prompts':[prompt],'durations':[1],**kwargs}

    def wait(self,client,job_id,states):
        deadline = time.monotonic()+5
        while time.monotonic()<deadline:
            job = client.get('/v1/jobs/'+job_id).json()
            if job['state'] in states:
                return job
            time.sleep(0.01)
        self.fail('Job never reached '+str(states))

    def test_authentication_validation_and_artifact_download(self):
        app = create_app(self.config,FakeRuntime,token='test-secret')
        with TestClient(app) as client:
            self.assertEqual(client.get('/v1/health').status_code,401)
            client.headers['Authorization'] = 'Bearer test-secret'
            self.assertEqual(client.get('/v1/health').json()['api_version'],1)
            self.assertEqual(client.post('/v1/jobs',json=self.body(durations=[0])).status_code,422)
            response = client.post('/v1/jobs',content=json.dumps(self.body(durations=[float('inf')])).encode(),
                                  headers={'Content-Type':'application/json'})
            self.assertEqual(response.status_code,422)
            self.assertEqual(client.post('/v1/jobs',json=self.body(prompts=['one','two'])).status_code,422)
            self.assertEqual(client.post('/v1/jobs',json=self.body(model='missing')).status_code,422)
            self.assertEqual(client.post('/v1/jobs',json=self.body(encoder='missing')).status_code,422)
            self.assertEqual(client.get('/v1/encoders').json()['default_encoder'],'awq')
            self.assertEqual(client.get('/v1/jobs/../secret').status_code,404)
            job = client.post('/v1/jobs',json=self.body()).json()
            result = self.wait(client,job['id'],{'succeeded'})
            self.assertEqual(result['result']['frames'],30)
            self.assertEqual(client.get('/v1/jobs/'+job['id']+'/artifact').content,b'HIERARCHY\nMOTION\nFrames: 30\n')
        self.assertTrue(app.state.manager.runtime.closed)

    def test_queue_cancel_failure_and_recovery(self):
        app = create_app(self.config,FakeRuntime,token='')
        with TestClient(app) as client:
            first = client.post('/v1/jobs',json=self.body('wait')).json()
            self.wait(client,first['id'],{'running'})
            second = client.post('/v1/jobs',json=self.body()).json()
            self.assertEqual(client.post('/v1/jobs',json=self.body()).status_code,429)
            self.assertEqual(client.post('/v1/jobs/'+second['id']+'/cancel').json()['state'],'cancelled')
            client.post('/v1/jobs/'+first['id']+'/cancel')
            self.wait(client,first['id'],{'cancelled'})
            self.assertEqual(client.get('/v1/jobs/'+first['id']+'/artifact').status_code,409)
            self.assertFalse(any(app.state.manager.directory.glob('*.bvh')))
            time.sleep(0.03)  # let the worker consume the already-cancelled queued item
            failed = client.post('/v1/jobs',json=self.body('fail')).json()
            self.assertEqual(self.wait(client,failed['id'],{'failed'})['error'],'deliberate test error')
            good = client.post('/v1/jobs',json=self.body()).json()
            self.assertEqual(self.wait(client,good['id'],{'succeeded'})['result']['frames'],30)


if __name__=='__main__':
    unittest.main()
