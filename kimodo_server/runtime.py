"""Persistent motion runtime; only the serialized worker accesses GPU state."""
import gc
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch

from .encoder import AWQEncoder, NF4Encoder, OfficialEncoder, EmbeddingStore
from .block_swap import BlockSwap, motion_blocks
from .quantization import adapt, build_denoiser, load_quantized
from .downloads import weights_ready
from .precision import apply_precision, validate_precision


class Cancelled(RuntimeError):
    pass


def check_cancel(event):
    if event.is_set():
        raise Cancelled('Cancelled by client')


class MotionRuntime:
    def __init__(self,config):
        self.config = config
        torch.set_num_threads(config.cpu_threads)
        torch.backends.cuda.matmul.allow_tf32 = config.tf32
        torch.backends.cudnn.allow_tf32 = config.tf32
        # Explicit QKV uses SDPA for both precision paths, keeping comparison fair.
        torch.backends.mha.set_fastpath_enabled(False)
        self.encoder_id = config.default_encoder
        self.encoder_config = config.encoders[self.encoder_id]
        self.store = EmbeddingStore(config,self.encoder_config,config.data_dir/'embeddings')
        self.encoder = None
        self.motion = None
        self.loaded = None
        self.swapper = None

    def profiles(self):
        return [{'id':key,'label':p.label,'precision':p.precision,'fps':30,'skeleton':p.skeleton,
                 'repo_id':p.repo_id,'constraints':p.skeleton=='SOMA',
                 'available':not (self.config.models_dir/p.path/'.kimodo-downloading').exists()
                    and all((self.config.models_dir/p.path/n).exists() for n in ('config.yaml','model.safetensors')),
                 'loaded':self.loaded==key}
                for key,p in self.config.models.items()]

    def encoders(self):
        return [{'id':key,'label':e.label,'kind':e.kind,'block_swap':e.block_swap,
                 'available':weights_ready(self.config.models_dir/e.base)
                    and all(weights_ready(self.config.models_dir/p,adapter=True)
                            for p in (e.mntp,e.supervised) if p),
                 'selected':self.encoder_id==key,'approximate':e.kind=='awq_llm2vec'}
                for key,e in self.config.encoders.items()]

    def _select_encoder(self,name):
        if name==self.encoder_id:
            return
        if self.encoder is not None:
            self.encoder.close()
            self.encoder = None
        self.encoder_id = name
        self.encoder_config = self.config.encoders[name]
        self.store = EmbeddingStore(self.config,self.encoder_config,self.config.data_dir/'embeddings')
        if self.motion is not None:
            self.motion.text_encoder = self.store
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self,name,progress=lambda *a:None):
        if self.loaded==name and self.motion is not None:
            return
        profile = self.config.models[name]
        path = self.config.models_dir/profile.path
        if profile.precision=='int8_convrot' and not self.config.device.startswith('cuda'):
            raise ValueError('ConvRot INT8 requires a CUDA device')
        validate_precision(profile.precision,self.config.device)
        if self.motion is not None:
            if self.swapper:
                self.swapper.close()
            self.motion = None
            self.swapper = None
            self.loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress('正在加载 '+profile.label,0.02)
        denoiser = (load_quantized(path,device='cpu') if profile.precision=='int8_convrot'
                    else adapt(build_denoiser(path),mode='fp32'))
        if profile.precision in {'fp16','bf16'}:
            denoiser = apply_precision(denoiser,profile.precision)
        from kimodo.model.kimodo_model import Kimodo
        class ProgressKimodo(Kimodo):
            # Upstream _multiprompt does not forward progress_bar to _generate.
            # Supply it here so remote progress and step cancellation also work
            # for stitched clips, without modifying the installed Kimodo library.
            service_progress = None

            def _generate(self,*args,**kwargs):
                if self.service_progress is not None:
                    kwargs['progress_bar'] = self.service_progress
                return super()._generate(*args,**kwargs)
        from omegaconf import OmegaConf
        model_config = OmegaConf.load(path/'config.yaml')
        self.motion = ProgressKimodo(denoiser,self.store,num_base_steps=int(model_config.get('num_base_steps',1000)),
                            device='cpu',cfg_type=model_config.get('cfg_type','separated')).eval()
        self.loaded = name
        progress('模型已就绪',0.08)

    def _move_motion(self,device):
        if self.motion is not None:
            if self.swapper:
                self.swapper.move(device)
            else:
                self.motion.to(device)
            self.motion.device = device

    def prewarm(self,texts,event,progress,text_block_swap=None):
        misses = [text for text in dict.fromkeys(texts) if self.store.get(text) is None]
        if not misses:
            return 0
        if self.config.offload:
            self._move_motion('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        check_cancel(event)
        if self.encoder is None:
            progress('正在加载本地文本编码器（首次请求）',0.09)
            cls = {'awq_llm2vec':AWQEncoder,'nf4_llm2vec':NF4Encoder,'llm2vec':OfficialEncoder}[self.encoder_config.kind]
            self.encoder = cls(*self.store.paths,block_swap=self.encoder_config.block_swap if text_block_swap is None else text_block_swap)
        self.encoder.set_block_swap(self.encoder_config.block_swap if text_block_swap is None else text_block_swap)
        try:
            if self.config.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats()
            self.encoder.to(self.config.device)
            for i,text in enumerate(misses):
                check_cancel(event)
                progress(f'正在编码提示词 {i+1}/{len(misses)}',0.10+0.10*i/len(misses))
                self.store.put(text,self.encoder.encode(text))
            peak = torch.cuda.max_memory_allocated()/1024**2 if self.config.device.startswith('cuda') else 0
        finally:
            if self.config.offload:
                self.encoder.to('cpu')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        check_cancel(event)
        return round(peak,1)

    def generate(self,request,event,progress,output_path):
        overall_started = time.perf_counter()
        from kimodo.sanitize import sanitize_texts
        from kimodo.constraints import load_constraints_lst
        from kimodo.tools import seed_everything
        from kimodo.exports.bvh import save_motion_bvh
        from kimodo.exports.motion_io import save_kimodo_npz
        if request.post_processing and importlib.util.find_spec('motion_correction') is None:
            raise RuntimeError('未安装 motion_correction。请关闭“动作后处理”，或在服务端安装兼容的 MotionCorrection。')
        self.load(request.model or self.config.default_model,progress)
        self._select_encoder(request.encoder or self.config.default_encoder)
        count = self.config.block_swap if request.block_swap is None else request.block_swap
        if self.swapper is None or len(self.swapper.blocks)!=count:
            if self.swapper:
                self.swapper.close()
            self.swapper = BlockSwap(self.motion,motion_blocks(self.motion),count)
        check_cancel(event)
        texts = sanitize_texts(request.prompts)
        frames = [max(3,round(d*self.motion.fps)) for d in request.durations]
        started = time.perf_counter()
        text_peak = self.prewarm(texts,event,progress,request.text_block_swap)
        text_seconds = time.perf_counter()-started
        self._move_motion(self.config.device)
        model = self.motion
        constraints = load_constraints_lst(request.constraints,model.skeleton,
                        device=self.config.device,dtype=torch.float32)
        seed_everything(request.seed)
        self.store.seed_iterator = iter(request.seeds) if request.seeds is not None else None
        segment = 0
        total_steps = request.steps*len(texts)
        completed = 0

        def progress_bar(indices):
            nonlocal segment,completed
            segment += 1
            for i in indices:
                check_cancel(event)
                progress(f'片段 {segment}/{len(texts)} · 扩散 {request.steps-i}/{request.steps}',
                         0.22+0.70*completed/total_steps)
                yield i
                completed += 1
            check_cancel(event)

        if self.config.device.startswith('cuda'):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model.service_progress = progress_bar
        try:
            with torch.inference_mode():
                output = model(texts,frames,num_denoising_steps=request.steps,
                    multi_prompt=len(texts)>1,num_samples=1,
                    constraint_lst=constraints,cfg_weight=[request.cfg_text,request.cfg_constraints],
                    num_transition_frames=request.transition_frames,
                    post_processing=request.post_processing,return_numpy=False,
                    progress_bar=progress_bar)
        finally:
            self.store.seed_iterator = None
            model.service_progress = None
        if self.config.device.startswith('cuda'):
            torch.cuda.synchronize()
        motion_seconds = time.perf_counter()-start
        check_cancel(event)
        for key,value in output.items():
            if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
                raise RuntimeError('Non-finite motion output: '+key)
        progress('正在导出动作',0.96)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        single = {key:(value[0] if isinstance(value,torch.Tensor) and value.ndim>0 and value.shape[0]==1 else value)
                  for key,value in output.items()}
        if request.output_format=='bvh':
            save_motion_bvh(str(output_path),single['local_rot_mats'],single['root_positions'],
                skeleton=model.output_skeleton,fps=model.fps,
                standard_tpose=request.standard_tpose or self.config.models[self.loaded].skeleton!='SOMA')
        else:
            save_kimodo_npz(str(output_path),single)
        metadata = {'model':self.loaded,'precision':self.config.models[self.loaded].precision,
                    'neural_weight_dtype':str(next(model.denoiser.parameters()).dtype).removeprefix('torch.'),
                    'sampling_dtype':'float32',
                    'fps':model.fps,'frames':int(single['root_positions'].shape[0]),
                    'prompts':texts,'seed':request.seed,'seeds':request.seeds,
                    'steps':request.steps,'text_seconds':round(text_seconds,3),
                    'motion_seconds':round(motion_seconds,3),'cache':self.store.stats(),
                    'text_encoder':self.encoder_config.kind,'encoder_id':self.encoder_id,
                    'approximate_text_conditioner':self.encoder_config.kind=='awq_llm2vec',
                    'block_swap':self.swapper.stats(),
                    'text_block_swap':self.encoder_config.block_swap if request.text_block_swap is None else request.text_block_swap,
                    'text_swap_stats':self.encoder.swapper.stats() if self.encoder else None,
                    'text_peak_cuda_mb':text_peak,
                    'motion_peak_cuda_mb':round(torch.cuda.max_memory_allocated()/1024**2,1) if self.config.device.startswith('cuda') else 0,
                    'total_seconds':round(time.perf_counter()-overall_started,3)}
        return metadata

    def close(self):
        if self.swapper:
            self.swapper.close()
        if self.encoder:
            self.encoder.close()
        self.motion = self.encoder = None
        self.swapper = None
        self.loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
