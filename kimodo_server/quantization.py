"""Local ConvRot INT8 adaptation, not an official author INT8 release.

Reference: feice-huang/ConvRot c80281c radix-4 butterfly, block size 256.
INT8 x INT8 -> INT32 GEMM, with FP32 retained projections and dequantization.
"""
import hashlib
import json
from pathlib import Path
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROT_SIZE = 256
_MATRICES = {}


def build_denoiser(model_dir, checkpoint=True, device='cpu'):
    from omegaconf import OmegaConf
    from kimodo.model.loading import instantiate_from_dict
    model_dir = Path(model_dir).resolve()
    cfg = OmegaConf.load(model_dir / 'config.yaml')
    cfg.checkpoint_dir = str(model_dir)
    node = OmegaConf.to_container(cfg.denoiser, resolve=True)
    if not checkpoint:
        node['ckpt_path'] = None
    return instantiate_from_dict(node).to(device).eval().requires_grad_(False)


def load_quantized(model_dir, device='cuda'):
    model_dir = Path(model_dir)
    from safetensors import safe_open
    with safe_open(str(model_dir/'model.safetensors'), framework='pt') as f:
        if (f.metadata() or {}).get('format') != 'kimodo_local_convrot_w8a8':
            raise ValueError('Expected a Kimodo ConvRot W8A8 checkpoint')
    denoiser = adapt(build_denoiser(model_dir, checkpoint=False), 'int8_convrot', empty=True)
    denoiser.load_state_dict(load_file(str(model_dir/'model.safetensors')), strict=True)
    return denoiser.to(device).eval()


def quantize_model(source, destination, device='cuda'):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new output directory; existing models are never overwritten')
    if not str(device).startswith('cuda'):
        raise ValueError('This INT8 CUDA implementation requires a CUDA device')
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_denoiser(source, device=device)
    total = sum(p.numel() for p in model.parameters())
    adapt(model, 'int8_convrot')
    state = {k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    # Finish quantization before creating the destination, so failures leave no half model.
    destination.mkdir(parents=True)
    save_file(state, str(destination/'model.safetensors'), metadata={
        'format':'kimodo_local_convrot_w8a8', 'convrot_commit':'c80281c25ad99eb4837abb4317978e12c05d05e1'})
    for name in ('config.yaml','LICENSE'):
        if (source/name).exists():
            shutil.copy2(source/name, destination/name)
    shutil.copytree(source/'stats', destination/'stats')
    modules = [m for m in model.modules() if isinstance(m,ConvRotLinear)]
    hasher = hashlib.sha256()
    with (destination/'model.safetensors').open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    info = {'method':'local ConvRot INT8 W8A8 adaptation', 'official_int8_release':False,
            'rotation_size':ROT_SIZE, 'quantized_modules':len(modules),
            'quantized_parameters':sum(m.weight_int8_t.numel() for m in modules),
            'total_parameters':total, 'sha256':digest,
            'source':str(source), 'bytes':(destination/'model.safetensors').stat().st_size}
    (destination/'quantization.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return info


def rotation_matrix(device, dtype=torch.float32):
    key = (str(device), dtype)
    if key not in _MATRICES:
        # Exact sign pattern of the author's radix-4 CUDA butterfly.
        b = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                          [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device) / 2
        h = b
        while h.shape[0] < ROT_SIZE:
            h = torch.kron(h, b)
        _MATRICES[key] = h.contiguous()
    return _MATRICES[key]


def rotate(x):
    assert x.shape[-1] % ROT_SIZE == 0
    return (x.reshape(-1, ROT_SIZE) @ rotation_matrix(x.device, x.dtype)).reshape(x.shape)


def quantize_tensor(x):
    scale = (x.abs().amax() / 127).clamp_min(torch.finfo(torch.float32).tiny)
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


class ConvRotLinear(nn.Module):
    def __init__(self, source, mode='int8_convrot', empty=False):
        super().__init__()
        self.in_features, self.out_features = source.in_features, source.out_features
        self.mode = mode
        self.use_rotation = mode in ('int8_convrot', 'rotation_only', 'int8_convrot_weight_only')
        self.use_quant = mode != 'rotation_only'
        self.calls = 0
        if source.bias is not None:
            self.register_buffer('bias', source.bias.detach().clone())
        else:
            self.register_buffer('bias', None)
        weight = source.weight.detach()
        if self.use_quant:
            if empty:
                q, scale = torch.empty_like(weight, dtype=torch.int8), torch.ones((), device=weight.device)
            else:
                q, scale = quantize_tensor(rotate(weight) if self.use_rotation else weight)
            # Keep the GEMM RHS contiguous; no floating weight cache.
            self.register_buffer('weight_int8_t', q.t().contiguous())
            self.register_buffer('weight_scale', scale)
        else:
            self.register_buffer('rotated_weight', rotate(weight))

    def forward(self, x):
        self.calls += 1
        if self.use_rotation:
            x = rotate(x)
        if not self.use_quant:
            return F.linear(x, self.rotated_weight, self.bias)
        if self.mode == 'int8_convrot_weight_only':
            return F.linear(x, self.weight_int8_t.t().float() * self.weight_scale, self.bias)
        shape = x.shape
        q, scale = quantize_tensor(x.reshape(-1, self.in_features))
        # torch._int_mm CUDA requires M > 16 and suitably aligned dimensions.
        rows = q.shape[0]
        padded = max(32, ((rows + 7) // 8) * 8)
        if rows != padded:
            q = F.pad(q, (0, 0, 0, padded - rows))
        accum = torch._int_mm(q.contiguous(), self.weight_int8_t)
        out = accum[:rows].float() * (scale * self.weight_scale)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)


class ExplicitSelfAttention(nn.Module):
    """Expose PyTorch's packed QKV so it cannot bypass quantized Linear.forward."""
    def __init__(self, source):
        super().__init__()
        assert source.batch_first and source._qkv_same_embed_dim
        assert source.bias_k is None and source.bias_v is None and not source.add_zero_attn
        self.embed_dim, self.num_heads = source.embed_dim, source.num_heads
        self.batch_first, self._qkv_same_embed_dim = True, True
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, device=source.in_proj_weight.device)
        with torch.no_grad():
            self.qkv.weight.copy_(source.in_proj_weight)
            self.qkv.bias.copy_(source.in_proj_bias)
        self.out_proj = source.out_proj

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                attn_mask=None, is_causal=False, **kwargs):
        assert query is key and key is value, 'Only self-attention is supported'
        assert not need_weights
        batch, length, embed = query.shape
        heads, dim = self.num_heads, embed // self.num_heads
        projected = self.qkv(query.transpose(0, 1))
        projected = projected.unflatten(-1, (3, embed)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = [p.view(length, batch * heads, dim).transpose(0, 1).view(batch, heads, length, dim)
                   for p in projected]
        mask = attn_mask
        if mask is not None and mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=query.dtype).masked_fill(mask, -float('inf'))
        if key_padding_mask is not None:
            pad = key_padding_mask
            if pad.dtype == torch.bool:
                pad = torch.zeros_like(pad, dtype=query.dtype).masked_fill(pad, -float('inf'))
            pad = pad[:, None, None, :]
            mask = pad if mask is None else mask + pad
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal)
        out = out.permute(2, 0, 1, 3).contiguous().view(batch * length, embed)
        return self.out_proj(out).view(length, batch, embed).transpose(0, 1), None


def adapt(denoiser, mode='fp32', empty=False):
    torch.backends.mha.set_fastpath_enabled(False)
    stacks = [m for m in denoiser.modules() if isinstance(m,nn.TransformerEncoder)]
    if not stacks:
        raise ValueError('No supported TransformerEncoder stack found in this denoiser')
    for stack in stacks:
        for layer in stack.layers:
            if not isinstance(layer.self_attn, ExplicitSelfAttention):
                layer.self_attn = ExplicitSelfAttention(layer.self_attn)
            if mode != 'fp32':
                for module, name in [(layer.self_attn, 'qkv'), (layer.self_attn, 'out_proj'),
                                     (layer, 'linear1'), (layer, 'linear2')]:
                    setattr(module, name, ConvRotLinear(getattr(module, name), mode=mode, empty=empty))
    denoiser.requires_grad_(False).eval()
    return denoiser


def self_test():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(17)
    device = 'cuda'
    h = rotation_matrix(device)
    assert torch.equal(h @ h.t(), torch.eye(ROT_SIZE, device=device))
    # Independent, line-for-line arithmetic reference to kernels.cu's butterfly.
    x = torch.randn(11, 1024, device=device)
    ref = x.reshape(-1, ROT_SIZE).clone()
    stride = 1
    while stride < ROT_SIZE:
        z = ref.reshape(-1, ROT_SIZE // (4 * stride), 4, stride)
        a, b, c, d = [z[:, :, i].clone() for i in range(4)]
        z[:, :, 0], z[:, :, 1] = a+b+c-d, a+b-c+d
        z[:, :, 2], z[:, :, 3] = a-b+c+d, -a+b+c+d
        stride *= 4
    ref = (ref / ROT_SIZE**0.5).reshape_as(x)
    torch.testing.assert_close(rotate(x), ref, rtol=2e-6, atol=2e-6)
    layer = nn.Linear(1024, 2048, device=device).eval()
    torch.testing.assert_close(ConvRotLinear(layer, 'rotation_only')(x), layer(x), atol=3e-6, rtol=2e-5)
    quant = ConvRotLinear(layer)
    got = quant(x)
    qa, sa = quantize_tensor(rotate(x))
    expected = (qa.float() @ quant.weight_int8_t.float()) * sa * quant.weight_scale + quant.bias
    torch.testing.assert_close(got, expected, atol=2e-6, rtol=1e-5)
    mha = nn.MultiheadAttention(1024, 8, batch_first=True, device=device).eval()
    inp = torch.randn(2, 31, 1024, device=device)
    pad = torch.zeros(2, 31, dtype=torch.bool, device=device)
    pad[1, 25:] = True
    with torch.no_grad():
        expected = mha(inp, inp, inp, key_padding_mask=pad, need_weights=False)[0]
        actual = ExplicitSelfAttention(mha)(inp, inp, inp, key_padding_mask=pad)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    print('PASS: official butterfly parity, orthogonality, linear invariance, INT8 GEMM, padded attention parity', flush=True)
