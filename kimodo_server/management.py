"""Persisted configuration and a small, bundled management UI."""
import asyncio
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import ValidationError

from .config import ServerConfig, resolve_config
from .downloads import DownloadManager, DownloadRequest


def atomic_save(path, document):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(document,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class ConfigController:
    def __init__(self, config, path=None, restart=None, token=''):
        self.config = config
        self.path = Path(path).resolve() if path else None
        self.restart = restart
        self.token = token
        self.lock = threading.RLock()
        self.restarting = False
        self.downloads = DownloadManager()

    def document(self):
        if self.path:
            return json.loads(self.path.read_text(encoding='utf-8-sig'))
        return self.config.model_dump(mode='json')

    def validate(self,document):
        try:
            config = resolve_config(document,self.path or Path.cwd()/'config.json')
        except (ValidationError, ValueError) as exc:
            raise HTTPException(422,str(exc)) from None
        if config.host not in {'127.0.0.1','localhost','::1'} and not self.token:
            raise HTTPException(422,'对外监听需先在服务进程环境设置 KIMODO_API_TOKEN')
        return config

    def ensure_idle(self,m):
        if self.restarting or m.stopping:
            raise HTTPException(409,'服务正在重启')
        if any(j.state not in {'succeeded','failed','cancelled'} for j in m.jobs.values()):
            raise HTTPException(409,'有排队或运行中的推理任务，请等待完成或取消后重试')
        if self.downloads.active():
            raise HTTPException(409,'有下载任务，请等待完成或取消后重试')


def register_management(app,config,auth,controller):
    assets = Path(__file__).parent/'web'

    @app.get('/',include_in_schema=False)
    def index():
        return FileResponse(assets/'index.html',headers={'Cache-Control':'no-cache'})

    @app.get('/ui/{name}',include_in_schema=False)
    def asset(name:str):
        if name not in {'app.js','style.css'}:
            raise HTTPException(404)
        return FileResponse(assets/name,headers={'Cache-Control':'no-cache'})

    def guarded(request:Request):
        auth(request)
        origin = request.headers.get('origin')
        if origin and urlsplit(origin).netloc != request.headers.get('host'):
            raise HTTPException(403,'请从服务端同源面板操作')
        if config.host in {'127.0.0.1','localhost','::1'} and request.url.hostname not in {'127.0.0.1','localhost','::1','testserver'}:
            raise HTTPException(403,'Invalid local host')

    deps = [Depends(guarded)]

    @app.get('/v1/admin/config',dependencies=deps)
    def read_config():
        with controller.lock:
            saved = controller.document()
            pending = controller.validate(saved).model_dump(mode='json') != config.model_dump(mode='json')
            return {'saved':saved,'active':config.model_dump(mode='json'),'pending_restart':pending,
                    'config_path':str(controller.path) if controller.path else None,
                    'can_restart':bool(controller.restart),'restarting':controller.restarting}

    @app.put('/v1/admin/config',dependencies=deps)
    def save_config(body:ServerConfig):
        with controller.lock:
            if controller.restarting:
                raise HTTPException(409,'服务正在重启')
            if controller.downloads.active():
                raise HTTPException(409,'下载时不能修改配置，请先完成或取消下载')
            if controller.path is None:
                raise HTTPException(409,'服务未绑定配置文件，请用 serve --config 启动')
            document = body.model_dump(mode='json')
            controller.validate(document)
            atomic_save(controller.path,document)
            return {'saved':True,'pending_restart':controller.validate(document).model_dump(mode='json') != config.model_dump(mode='json')}

    @app.post('/v1/admin/apply',dependencies=deps)
    async def apply_config(request:Request):
        m = request.app.state.manager
        with controller.lock, m.lock:
            controller.ensure_idle(m)
            if not controller.restart:
                raise HTTPException(409,'当前启动方式不支持面板重启；请使用 kimodo-server serve')
            target = controller.validate(controller.document())
            if (target.host,target.port)!=(config.host,config.port):
                family = socket.AF_INET6 if ':' in target.host else socket.AF_INET
                try:
                    with socket.socket(family,socket.SOCK_STREAM) as probe:
                        probe.bind((target.host,target.port))
                except OSError as exc:
                    raise HTTPException(409,'新监听地址不可用：'+str(exc)) from None
            controller.restarting = True
            m.stopping = True
            # Flush the HTTP response before graceful shutdown/reload.
            asyncio.get_running_loop().call_later(0.8,controller.restart)
            return {'restarting':True,'host':target.host,'port':target.port}

    @app.get('/v1/admin/status',dependencies=deps)
    def status(request:Request):
        m = request.app.state.manager
        with m.lock:
            jobs = [j.snapshot() for j in m.jobs.values()]
        return {'jobs':list(reversed(jobs))[:32],'downloads':controller.downloads.snapshots(),
                'restarting':controller.restarting}

    @app.post('/v1/admin/downloads',status_code=202,dependencies=deps)
    def download(body:DownloadRequest,request:Request):
        m = request.app.state.manager
        with controller.lock,m.lock:
            controller.ensure_idle(m)
            target = controller.validate(controller.document())
            if target.model_dump(mode='json') != config.model_dump(mode='json'):
                raise HTTPException(409,'请先应用配置并重启，再下载模型')
            return controller.downloads.start(config,body)

    @app.post('/v1/admin/downloads/{job_id}/cancel',dependencies=deps)
    def cancel_download(job_id:str):
        return controller.downloads.cancel(job_id)
