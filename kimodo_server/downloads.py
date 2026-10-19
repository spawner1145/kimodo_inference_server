"""Explicit, cancellable downloads in an online subprocess, isolated from inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


def weights_ready(path, adapter=False):
    path = Path(path)
    if (path/'.kimodo-downloading').exists():
        return False
    if not (path/('adapter_config.json' if adapter else 'config.json')).is_file():
        return False
    index = path/'model.safetensors.index.json'
    try:
        files = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {
            'adapter_model.safetensors' if adapter else 'model.safetensors'}
        return bool(files) and all((path/f).is_file() and (path/f).stat().st_size > 0 for f in files)
    except (OSError, ValueError, KeyError):
        return False


def download_items(config, kind, profile_id):
    profiles = config.models if kind=='model' else config.encoders
    if profile_id not in profiles:
        raise HTTPException(422, 'Unknown profile')
    p = profiles[profile_id]
    items = [(p.repo_id, p.path)] if kind=='model' else [(p.repo_id,p.base)] + [
        (getattr(p,n+'_repo_id'),getattr(p,n)) for n in ('mntp','supervised') if getattr(p,n)]
    if any(not repo for repo,_ in items):
        raise HTTPException(422, '此配置没有下载仓库，请先填写下载源和仓库 ID')
    root = config.models_dir.resolve()
    result = []
    for repo,relative in items:
        dest = (root/relative).resolve()
        if not dest.is_relative_to(root) or dest==root:
            raise HTTPException(422, '下载路径必须位于 models_dir 内的子目录')
        result.append({'repo':repo,'path':str(dest),'source':p.download_source})
    return result


class DownloadRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: str = Field(pattern='^(model|encoder)$')
    profile: str = Field(min_length=1, max_length=128)
    endpoint: HttpUrl = 'https://huggingface.co'
    token: str | None = Field(default=None, max_length=4096, repr=False)
    modelscope_token: str | None = Field(default=None, max_length=4096, repr=False)


class DownloadManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.jobs = {}
        self.process = None
        self.thread = None

    def active(self):
        return any(j['state'] in {'running','cancelling'} for j in self.jobs.values())

    def start(self, config, request):
        items = download_items(config,request.kind,request.profile)
        with self.lock:
            if self.active():
                raise HTTPException(409,'已有下载任务，请等待完成或取消')
            job = {'id':uuid.uuid4().hex,'kind':request.kind,'profile':request.profile,
                   'state':'running','message':'正在连接下载源','progress':None,
                   'created_at':time.time(),'finished_at':None,'completed':0,'total':None}
            self.jobs[job['id']] = job
            # Bound the in-memory history; no access tokens are persisted.
            for key in list(self.jobs)[:-32]:
                del self.jobs[key]
            env = os.environ.copy()
            for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):
                env.pop(name,None)
            env['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
            env['HF_ENDPOINT'] = str(request.endpoint).rstrip('/')
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            # Plain HTTP endpoints must never receive a credential.
            source = items[0]['source']
            secret = ((request.modelscope_token or env.get('MODELSCOPE_API_TOKEN')) if source=='modelscope'
                      else (request.token or env.get('HF_TOKEN'))) or ''
            if source=='huggingface' and secret and not str(request.endpoint).startswith('https://'):
                del self.jobs[job['id']]
                raise HTTPException(422,'带令牌的下载源必须使用 HTTPS')
            if source=='modelscope':
                env.pop('HF_TOKEN',None)
                env.pop('HUGGING_FACE_HUB_TOKEN',None)
                env['MODELSCOPE_API_TOKEN'] = secret
            else:
                env.pop('MODELSCOPE_API_TOKEN',None)
                env['HF_TOKEN'] = secret
            try:
                self.process = subprocess.Popen([sys.executable,'-u','-m','kimodo_server.downloads',
                    json.dumps(items),str(request.endpoint).rstrip('/')],stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            except OSError as exc:
                job.update(state='failed',message=str(exc),finished_at=time.time())
                return dict(job)
            self.thread = threading.Thread(target=self._read,args=(job,self.process,secret),daemon=True)
            self.thread.start()
            return dict(job)

    def _read(self,job,process,secret):
        try:
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except (ValueError,TypeError):
                    continue
                with self.lock:
                    for name in ('message','progress','completed','total'):
                        if name in event:
                            value = event[name]
                            job[name] = value.replace(secret,'[redacted]') if secret and isinstance(value,str) else value
            code = process.wait()
            with self.lock:
                if job['state']=='cancelling':
                    job.update(state='cancelled',message='已取消；再次下载可续传')
                elif code==0:
                    job.update(state='succeeded',message='下载完成',progress=1)
                else:
                    job.update(state='failed',message=job['message'] if job['message'].startswith('下载失败')
                               else '下载进程退出，请检查网络、仓库权限或磁盘空间')
                job['finished_at'] = time.time()
        finally:
            process.stdout.close()

    def snapshots(self):
        with self.lock:
            return [dict(j) for j in reversed(list(self.jobs.values()))]

    def cancel(self,job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise HTTPException(404,'Unknown download')
            if job['state']=='running':
                job.update(state='cancelling',message='正在取消')
                self.process.terminate()
            return dict(job)

    def close(self):
        with self.lock:
            for j in self.jobs.values():
                if j['state']=='running':
                    self.cancel(j['id'])
        if self.thread:
            self.thread.join(timeout=10)


def download_worker(items,endpoint):
    from tqdm.auto import tqdm

    def emit(**event):
        print(json.dumps(event,ensure_ascii=False),flush=True)

    class Progress(tqdm):
        def display(self,*args,**kwargs):
            total = self.total or 0
            # HF uses this class for the repository file counter. Individual
            # network transfers may use their own progress bars on older hubs.
            emit(message=f'{current[0]} · {self.desc or "文件"} {self.n}/{total or "?"}',
                 completed=self.n,total=total or None,
                 progress=(index+self.n/total)/len(items) if total else None)

    current = ['']
    try:
        for index,item in enumerate(items):
            current[0] = item['repo']
            dest = Path(item['path'])
            dest.mkdir(parents=True,exist_ok=True)
            marker = dest/'.kimodo-downloading'
            marker.write_text(item['repo'],encoding='utf-8')
            emit(message='正在下载 '+item['repo'],progress=index/len(items))
            if item.get('source','huggingface')=='modelscope':
                from modelscope.hub.snapshot_download import snapshot_download
                # The HF endpoint/token must never be forwarded to ModelScope.
                # ModelScope reports transfer bars on stdout; our WebUI shows
                # repository completion, not a fabricated file/byte percentage.
                snapshot_download(model_id=item['repo'],local_dir=str(dest),
                                  endpoint='https://www.modelscope.cn',
                                  token=os.environ.get('MODELSCOPE_API_TOKEN') or None,
                                  max_workers=4)
            else:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=item['repo'],local_dir=dest,endpoint=endpoint,
                                  token=os.environ.get('HF_TOKEN') or False,tqdm_class=Progress,
                                  max_workers=4)
            marker.unlink(missing_ok=True)
            emit(message='已完成 '+item['repo'],progress=(index+1)/len(items))
        emit(message='下载完成',progress=1)
    except Exception as exc:
        emit(message='下载失败：'+str(exc))
        raise SystemExit(1)


if __name__=='__main__':
    download_worker(json.loads(sys.argv[1]),sys.argv[2])
