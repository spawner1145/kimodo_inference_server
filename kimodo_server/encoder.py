"""Local, disk-cached LLM2Vec conditioning with phased GPU residency.

AWQ is a portable reference path: packed weights remain compressed and each
linear is temporarily dequantized. Original LoRAs on this AWQ base are an
approximate conditioner, not an equivalent replacement for official BF16 LLM2Vec.
"""
from collections import OrderedDict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from .block_swap import BlockSwap


def unpack(packed):
    shifts = torch.tensor([4*i for i in [0,4,1,5,2,6,3,7]], device=packed.device)
    return ((packed[...,None] >> shifts) & 15).to(torch.int8).flatten(-2)


def dequantize(qweight, qzeros, scales):
    w = unpack(qweight).reshape(scales.shape[0],128,scales.shape[1])
    z = unpack(qzeros)[:,None,:]
    return ((w-z).to(scales.dtype)*scales[:,None,:]).reshape(qweight.shape[0],-1)


class AWQLoRALinear(nn.Module):
    def __init__(self, name, tensors, mntp, supervised):
        super().__init__()
        for field, tensor in tensors.items():
            self.register_buffer(field, tensor)
        for prefix,state in (('mntp',mntp),('sup',supervised)):
            for part in ('A','B'):
                suffix = name+'.lora_'+part+'.weight'
                matches = [v for k,v in state.items() if k.replace('.default.','.').endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError('Missing or ambiguous adapter tensor: '+suffix)
                self.register_buffer(prefix+'_'+part,matches[0].float())

    def forward(self,x):
        weight = dequantize(self.qweight,self.qzeros,self.scales).t().contiguous()
        delta = self.mntp_B @ self.mntp_A
        weight = (weight.float()+delta*2.0).half()
        del delta
        result = F.linear(x.half(),weight)
        del weight
        update = F.linear(F.linear(x.float(),self.sup_A),self.sup_B)*2.0
        return (result.float()+update).to(x.dtype)


class EncoderResidency:
    def setup_residency(self,count):
        decoder = next(m for m in self.llm.model.modules() if isinstance(getattr(m,'layers',None),nn.ModuleList))
        self.decoder_blocks = list(decoder.layers)
        self.swapper = BlockSwap(self.llm.model,self.decoder_blocks,count)

    def set_block_swap(self,count):
        if len(self.swapper.blocks)!=count:
            self.swapper.close()
            self.swapper = BlockSwap(self.llm.model,self.decoder_blocks,count)

    def to(self,device):
        self.swapper.move(device)
        self.device = str(device)
        return self

    def close(self):
        self.swapper.close()

    def encode(self,text):
        inp = self.llm.prepare_for_tokenization(self.llm._convert_to_str('',text))
        batch = {k:v.to(self.device) for k,v in self.llm.tokenize([inp]).items()}
        with torch.inference_mode():
            return self.llm(batch).float().cpu().reshape(1,-1)


class AWQEncoder(EncoderResidency):
    def __init__(self, base, mntp, supervised, block_swap=0):
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from transformers import AutoTokenizer, LlamaConfig
        from kimodo.model.llm2vec.llm2vec import LLM2Vec
        from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
        base,mntp,supervised = map(Path,(base,mntp,supervised))
        conf = json.loads((base/'config.json').read_text(encoding='utf-8'))
        quant = conf.pop('quantization_config')
        if not (quant['bits']==4 and quant['group_size']==128 and quant['version'].lower()=='gemm'):
            raise ValueError('AWQ reference encoder requires GEMM INT4 group128')
        for path in (mntp,supervised):
            c = json.loads((path/'adapter_config.json').read_text(encoding='utf-8'))
            if not (c['r']==16 and c['lora_alpha']==32 and c['bias']=='none'
                    and not c.get('use_rslora',False) and not c.get('use_dora',False)):
                raise ValueError('Encoder requires the original LLM2Vec rank16 alpha32 adapters')
        mntp_state = load_file(str(mntp/'adapter_model.safetensors'))
        sup_state = load_file(str(supervised/'adapter_model.safetensors'))
        cfg = LlamaConfig.from_dict(conf)
        cfg._attn_implementation = 'sdpa'
        cfg._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
        cfg.use_cache = False
        with init_empty_weights():
            model = LlamaBiModel(cfg)
        with ExitStack() as stack:
            handles = [stack.enter_context(safe_open(str(p),framework='pt',device='cpu'))
                       for p in sorted(base.glob('*.safetensors'))]
            lookup = {key:h for h in handles for key in h.keys()}
            def tensor(key):
                return lookup[key].get_tensor(key)
            for name,module in list(model.named_modules()):
                if isinstance(module,nn.Linear):
                    parent,child = name.rsplit('.',1)
                    values = {f:tensor('model.'+name+'.'+f) for f in ('qweight','qzeros','scales')}
                    setattr(model.get_submodule(parent),child,AWQLoRALinear(name,values,mntp_state,sup_state))
            for name,param in list(model.named_parameters()):
                set_module_tensor_to_device(model,name,'cpu',value=tensor('model.'+name),dtype=torch.float16)
        if any(p.is_meta for p in model.parameters()):
            raise RuntimeError('Encoder still contains meta tensors')
        tok = AutoTokenizer.from_pretrained(str(mntp),local_files_only=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = 'left'
        self.llm = LLM2Vec(model.eval().requires_grad_(False),tok).eval()
        self.device = 'cpu'
        options_path = supervised/'llm2vec_config.json'
        if options_path.exists():
            options = json.loads(options_path.read_text(encoding='utf-8'))
            self.llm.pooling_mode = options.get('pooling_mode',self.llm.pooling_mode)
            self.llm.max_length = options.get('max_length',self.llm.max_length)
        self.setup_residency(block_swap)


class OfficialEncoder(EncoderResidency):
    def __init__(self,base,mntp,supervised,block_swap=0):
        from peft import PeftModel
        from transformers import AutoTokenizer
        from kimodo.model.llm2vec.llm2vec import LLM2Vec
        from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
        model = LlamaBiModel.from_pretrained(str(base),torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa',local_files_only=True)
        model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
        model = PeftModel.from_pretrained(model,str(mntp),local_files_only=True).merge_and_unload()
        model = PeftModel.from_pretrained(model,str(supervised),local_files_only=True)
        tok = AutoTokenizer.from_pretrained(str(mntp),local_files_only=True)
        tok.pad_token,tok.padding_side = tok.eos_token,'left'
        options_path = Path(supervised)/'llm2vec_config.json'
        options = json.loads(options_path.read_text(encoding='utf-8')) if options_path.exists() else {}
        self.llm = LLM2Vec(model.eval().requires_grad_(False),tok,**options).eval()
        self.device = 'cpu'
        # PEFT wraps the model; locate the actual bidirectional decoder stack.
        self.setup_residency(block_swap)


class NF4Encoder(EncoderResidency):
    """Aero-Ex's prequantized/merged NF4 LLM2Vec, without adding LoRAs twice."""
    def __init__(self,base,mntp=None,supervised=None,block_swap=0):
        from kimodo.model.llm2vec.llm2vec import LLM2Vec
        base = Path(base)
        conf = json.loads((base/'config.json').read_text(encoding='utf-8'))
        quant = conf.get('quantization_config',{})
        if quant.get('quant_method')!='bitsandbytes' or quant.get('bnb_4bit_quant_type')!='nf4':
            raise ValueError('Expected a prequantized bitsandbytes NF4 LLM2Vec directory')
        # Match the original plugin: the public NF4 checkpoint already contains
        # the merged LLM2Vec adaptation and its own pooling configuration.
        self.llm = LLM2Vec.from_pretrained(str(base),peft_model_name_or_path=None,
            torch_dtype=torch.bfloat16,device_map='cpu',local_files_only=True,
            attn_implementation='sdpa').eval()
        self.llm.model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
        self.llm.model.config.use_cache = False
        self.llm.requires_grad_(False)
        self.device = 'cpu'
        self.setup_residency(block_swap)


class EmbeddingStore:
    """Small CPU LRU plus atomic .npy files namespaced by encoder fingerprint."""
    def __init__(self,config,encoder_config,cache_root,max_memory=128):
        self.config = config
        self.encoder_config = encoder_config
        self.paths = [config.models_dir / getattr(encoder_config,n) if getattr(encoder_config,n) else None
                      for n in ('base','mntp','supervised')]
        identity = {'implementation':'llm2vec-v2', 'encoder':encoder_config.model_dump(exclude={'block_swap'})}
        files = []
        for root in self.paths:
            if root is None:
                continue
            for p in sorted(root.glob('*')):
                if p.is_file() and p.suffix in {'.json','.safetensors'}:
                    stat = p.stat()
                    files.append((str(p.resolve()),stat.st_size,stat.st_mtime_ns))
        identity['files'] = files
        self.fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.root = Path(cache_root)/self.fingerprint
        self.root.mkdir(parents=True,exist_ok=True)
        self.memory = OrderedDict()
        self.max_memory = max_memory
        self.hits = self.disk_hits = self.misses = 0
        self.seed_iterator = None

    def _remember(self,key,value):
        self.memory[key] = value
        self.memory.move_to_end(key)
        while len(self.memory)>self.max_memory:
            self.memory.popitem(last=False)

    def key(self,text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self,text):
        if not text.strip():
            return torch.zeros(1,4096)
        key = self.key(text)
        if key in self.memory:
            self.hits += 1
            self.memory.move_to_end(key)
            return self.memory[key]
        path = self.root/(key+'.npy')
        if path.exists():
            try:
                array = np.load(path,allow_pickle=False)
                if array.shape != (1,4096) or array.dtype!=np.float32 or not np.isfinite(array).all():
                    raise ValueError('Invalid cached embedding')
                value = torch.from_numpy(array)
                self._remember(key,value)
                self.disk_hits += 1
                return value
            except (ValueError,OSError,EOFError):
                pass
        self.misses += 1
        return None

    def put(self,text,value):
        value = value.detach().float().cpu().contiguous().clone()
        if value.shape!=(1,4096) or not torch.isfinite(value).all():
            raise ValueError('Text encoder must return a finite (1,4096) feature')
        key = self.key(text)
        temp = self.root/(key+'.tmp')
        with temp.open('wb') as f:
            np.save(f,value.numpy(),allow_pickle=False)
        temp.replace(self.root/(key+'.npy'))
        self._remember(key,value)
        entries = list(self.root.glob('*.npy'))
        if len(entries)>4096:
            for p in sorted(entries,key=lambda p:p.stat().st_mtime)[:len(entries)-4096]:
                p.unlink(missing_ok=True)

    def __call__(self,texts):
        if self.seed_iterator is not None:
            # Kimodo calls the encoder before drawing each segment's initial
            # noise. This preserves segment seeds without patching its sampler.
            from kimodo.tools import seed_everything
            seed_everything(next(self.seed_iterator))
        values = [self.get(text) for text in texts]
        if any(v is None for v in values):
            raise RuntimeError('Text cache was not prewarmed before motion inference')
        return torch.stack(values),[1 if text.strip() else 0 for text in texts]

    def stats(self):
        return {'memory_hits':self.hits,'disk_hits':self.disk_hits,'misses':self.misses,
                'memory_entries':len(self.memory),'encoder_fingerprint':self.fingerprint}
