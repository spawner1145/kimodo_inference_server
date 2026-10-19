"""Half-precision neural stages with FP32 diffusion and motion geometry."""
import torch
from torch import nn


DTYPES = {'fp16':torch.float16,'bf16':torch.bfloat16}


def validate_precision(precision, device):
    if precision not in DTYPES:
        return
    device = torch.device(device)
    if device.type not in {'cpu','cuda'}:
        raise ValueError('FP16/BF16 stages currently support CPU or CUDA devices')
    if device.type=='cuda' and precision=='bf16':
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError('This GPU does not support native BF16; select FP16 or FP32')


class PrecisionStage(nn.Module):
    """Autocast stays inside the neural stage; root conversion remains FP32."""
    def __init__(self,stage,dtype):
        super().__init__()
        self.stage = stage
        self.dtype = dtype

    def forward(self,*args,**kwargs):
        x = args[0] if args else kwargs['x']
        # Disable autocast's weight cache across per-block CPU/CUDA transfers.
        with torch.autocast(device_type=x.device.type,dtype=self.dtype,cache_enabled=False):
            output = self.stage(*args,**kwargs)
        return output.float()


def apply_precision(denoiser,precision):
    if precision=='fp32':
        return denoiser
    dtype = DTYPES[precision]
    # Skeletons are shared by the two neural stages and motion representation.
    # Never quantize skeleton constants or normalization statistics via .half().
    protected = {id(module) for module in denoiser.motion_rep.modules()}
    for name in ('root_model','body_model'):
        stage = getattr(denoiser,name,None)
        if stage is None:
            raise ValueError('Half precision requires Kimodo root_model and body_model stages')
        with torch.no_grad():
            for module in stage.modules():
                if id(module) in protected:
                    continue
                for parameter in module.parameters(recurse=False):
                    if parameter.is_floating_point():
                        parameter.data = parameter.data.to(dtype=dtype)
                for key,buffer in module._buffers.items():
                    if buffer is not None and buffer.is_floating_point():
                        module._buffers[key] = buffer.to(dtype=dtype)
        setattr(denoiser,name,PrecisionStage(stage,dtype))
    return denoiser
