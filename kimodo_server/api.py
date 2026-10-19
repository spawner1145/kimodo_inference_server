"""Versioned HTTP jobs API; one worker owns the model and CUDA context."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hmac
import importlib.util
import json
import logging
from pathlib import Path
import queue
import re
import threading
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from . import __version__
from .config import api_token
from .schemas import GenerateRequest, LoadRequest

log = logging.getLogger(__name__)
TERMINAL = {'succeeded','failed','cancelled'}


@dataclass
class Job:
    id: str
    kind: str
    payload: object
    created_at: float = field(default_factory=time.time)
    state: str = 'queued'
    message: str = '正在排队'
    progress: float = 0
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None
    path: Path | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def snapshot(self):
        return {'id':self.id,'kind':self.kind,'state':self.state,'message':self.message,
                'progress':self.progress,'created_at':self.created_at,'finished_at':self.finished_at,
                'result':self.result,'error':self.error,
                'artifact':bool(self.state=='succeeded' and self.path)}


class JobManager:
    def __init__(self,config,runtime):
        self.config,self.runtime = config,runtime
        self.jobs = {}
        self.lock = threading.RLock()
        self.queue = queue.Queue(maxsize=config.queue_size)
        self.stopping = False
        self.directory = config.data_dir/'jobs'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.worker = threading.Thread(target=self._work,daemon=True,name='kimodo-gpu')
        self.worker.start()

    def _prune(self):
        cutoff = time.time()-self.config.retention_hours*3600
        for path in self.directory.iterdir():
            if re.fullmatch(r'[0-9a-f]{32}\.(bvh|npz|json)',path.name) and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        with self.lock:
            finished = [j for j in self.jobs.values() if j.state in TERMINAL]
            for j in sorted(finished,key=lambda j:j.created_at)[:-256]:
                self.jobs.pop(j.id,None)

    def submit(self,kind,payload):
        if kind=='generate':
            defaults = {key:value for key,value in self.config.inference.model_dump().items()
                        if key not in payload.model_fields_set}
            try:
                payload = GenerateRequest.model_validate({**payload.model_dump(),**defaults})
            except ValidationError as exc:
                raise HTTPException(422,str(exc)) from None
        model = payload.model or self.config.default_model
        if model not in self.config.models:
            raise HTTPException(422,'Unknown model profile: '+model)
        encoder = payload.encoder or self.config.default_encoder
        if encoder not in self.config.encoders:
            raise HTTPException(422,'Unknown encoder profile: '+encoder)
        available = next((p.get('available') for p in self.runtime.profiles() if p['id']==model),False)
        if not available:
            raise HTTPException(422,'Model files are not available; download/configure profile '+model)
        available = next((p.get('available') for p in self.runtime.encoders() if p['id']==encoder),False)
        if not available:
            raise HTTPException(422,'Encoder files are not available; download/configure profile '+encoder)
        payload = payload.model_copy(update={'model':model,'encoder':encoder})
        job = Job(uuid.uuid4().hex,kind,payload)
        with self.lock:
            if self.stopping:
                raise HTTPException(503,'Server is stopping')
            self.jobs[job.id] = job
            try:
                self.queue.put_nowait(job)
            except queue.Full:
                del self.jobs[job.id]
                raise HTTPException(429,'Inference queue is full') from None
        return job.snapshot()

    def get(self,job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404,'Unknown or expired job')
            return self.jobs[job_id]

    def cancel_job(self,job_id):
        with self.lock:
            job = self.get(job_id)
            if job.state not in TERMINAL:
                job.cancel.set()
                if job.state=='queued':
                    job.state,job.finished_at = 'cancelled',time.time()
                else:
                    job.state = 'cancelling'
                job.message = '已请求取消'
            return job.snapshot()

    def _work(self):
        from .runtime import Cancelled,check_cancel
        try:
            self._prune()
            while True:
                job = self.queue.get()
                if job is None:
                    break
                with self.lock:
                    if job.cancel.is_set():
                        continue
                    job.state = 'running'
                def progress(message,value):
                    with self.lock:
                        job.message = message
                        job.progress = round(max(0,min(1,value)),4)
                try:
                    if job.kind=='load':
                        self.runtime.load(job.payload.model,progress)
                        check_cancel(job.cancel)
                        result = {'model':job.payload.model,'fps':30}
                    else:
                        job.path = self.directory/(job.id+'.'+job.payload.output_format)
                        result = self.runtime.generate(job.payload,job.cancel,progress,job.path)
                        check_cancel(job.cancel)
                    with self.lock:
                        # Cancellation and committing success are atomic.
                        check_cancel(job.cancel)
                        job.result,job.state,job.progress,job.message = result,'succeeded',1.0,'已完成'
                except Cancelled:
                    with self.lock:
                        job.state,job.message = 'cancelled','已取消'
                except Exception as exc:
                    log.exception('Job %s failed',job.id)
                    with self.lock:
                        job.state,job.error,job.message = 'failed',str(exc),'任务失败'
                finally:
                    with self.lock:
                        job.finished_at = time.time()
                        if job.state!='succeeded' and job.path:
                            job.path.unlink(missing_ok=True)
                        snapshot = job.snapshot()
                    try:
                        (self.directory/(job.id+'.json')).write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
                        self._prune()
                    except OSError:
                        log.exception('Could not persist/prune job %s metadata',job.id)
        finally:
            self.runtime.close()

    def close(self):
        with self.lock:
            self.stopping = True
            for job in self.jobs.values():
                if job.state not in TERMINAL:
                    job.cancel.set()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put(None)
        self.worker.join(timeout=30)


def create_app(config,runtime_factory=None,token=None,config_path=None,restart=None):
    if runtime_factory is None:
        from .runtime import MotionRuntime
        runtime_factory = MotionRuntime
    token = api_token() if token is None else token
    from .management import ConfigController, register_management
    controller = ConfigController(config,config_path,restart,token)

    @asynccontextmanager
    async def lifespan(app):
        manager = JobManager(config,runtime_factory(config))
        app.state.manager = manager
        app.state.controller = controller
        try:
            yield
        finally:
            await asyncio.to_thread(controller.downloads.close)
            await asyncio.to_thread(manager.close)

    app = FastAPI(title='Kimodo Motion API',version=__version__,lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request,exc):
        # Non-finite JSON numbers must be rejected without echoing Infinity/NaN
        # into JSONResponse (which would itself fail serialization).
        detail = [{k:error[k] for k in ('type','loc','msg')} for error in exc.errors()]
        return JSONResponse(status_code=422,content={'detail':detail})

    def auth(request:Request):
        if token and not hmac.compare_digest(request.headers.get('Authorization','').encode(),('Bearer '+token).encode()):
            raise HTTPException(401,'Invalid API token',headers={'WWW-Authenticate':'Bearer'})

    def manager(request:Request):
        return request.app.state.manager

    dependencies = [Depends(auth)]

    @app.get('/v1/health',dependencies=dependencies)
    def health(m=Depends(manager)):
        return {'status':'ok','api_version':1,'service_version':__version__,
                'default_model':config.default_model,'loaded_model':m.runtime.loaded,
                'device':config.device,'offload':config.offload,'default_encoder':config.default_encoder,
                'text_encoder':config.encoders[config.default_encoder].kind,
                'approximate_text_conditioner':config.encoders[config.default_encoder].kind=='awq_llm2vec',
                'capabilities':{'bvh':True,'npz':True,'multi_prompt':True,'segment_seeds':True,
                    'motion_precisions':['fp32','fp16','bf16','int8_convrot'],
                    'block_swap':True,'text_block_swap':True,'default_block_swap':config.block_swap,
                    'cancellation':'between_denoising_steps','post_processing':importlib.util.find_spec('motion_correction') is not None}}

    @app.get('/v1/models',dependencies=dependencies)
    def models(m=Depends(manager)):
        return {'models':m.runtime.profiles(),'default_model':config.default_model}

    @app.get('/v1/encoders',dependencies=dependencies)
    def encoders(m=Depends(manager)):
        return {'encoders':m.runtime.encoders(),'default_encoder':config.default_encoder}

    @app.post('/v1/models/load',status_code=202,dependencies=dependencies)
    def load(body:LoadRequest,m=Depends(manager)):
        with controller.lock,m.lock:
            if controller.downloads.active():
                raise HTTPException(409,'下载中，请稍后加载模型')
            return m.submit('load',body)

    @app.post('/v1/jobs',status_code=202,dependencies=dependencies)
    def generate(body:GenerateRequest,m=Depends(manager)):
        with controller.lock,m.lock:
            if controller.downloads.active():
                raise HTTPException(409,'下载中，请稍后生成动作')
            return m.submit('generate',body)

    @app.get('/v1/jobs/{job_id}',dependencies=dependencies)
    def status(job_id:str,m=Depends(manager)):
        with m.lock:
            return m.get(job_id).snapshot()

    @app.post('/v1/jobs/{job_id}/cancel',dependencies=dependencies)
    def cancel(job_id:str,m=Depends(manager)):
        return m.cancel_job(job_id)

    @app.get('/v1/jobs/{job_id}/artifact',dependencies=dependencies)
    def artifact(job_id:str,m=Depends(manager)):
        with m.lock:
            job = m.get(job_id)
            if job.state!='succeeded' or not job.path or not job.path.is_file():
                raise HTTPException(409,'Artifact is not available')
            return FileResponse(job.path,filename=job.path.name,media_type='application/octet-stream')

    register_management(app,config,auth,controller)
    return app
