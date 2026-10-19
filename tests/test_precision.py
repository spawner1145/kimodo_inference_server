"""Precision boundaries, protected geometry, masks and CPU/CUDA swapping."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from kimodo_server.block_swap import BlockSwap,motion_blocks
from kimodo_server.config import ModelProfile
from kimodo_server.precision import apply_precision,validate_precision
from kimodo_server.quantization import adapt


class Stage(nn.Module):
    def __init__(self,skeleton):
        super().__init__()
        self.skeleton = skeleton
        self.input = nn.Linear(32,32)
        self.register_buffer('position',torch.randn(1,7,32))
        self.seqTransEncoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(32,4,64,dropout=0,batch_first=True),1,
            enable_nested_tensor=False)
        self.output = nn.Linear(32,32)

    def forward(self,x,mask,timesteps):
        self.seen_dtypes = (x.dtype,mask.dtype,timesteps.dtype)
        output = self.seqTransEncoder(self.input(x)+self.position,src_key_padding_mask=mask)
        output = self.output(output)
        self.activation_dtype = output.dtype
        return output


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_rep = nn.Module()
        self.motion_rep.register_buffer('statistics',torch.randn(4))
        self.motion_rep.skeleton = nn.Module()
        self.motion_rep.skeleton.register_buffer('joints',torch.randn(5,3))
        self.root_model = Stage(self.motion_rep.skeleton)
        self.body_model = Stage(self.motion_rep.skeleton)

    def forward(self,x,mask,timesteps):
        root = self.root_model(x,mask,timesteps)
        self.geometry_dtype = root.dtype
        return self.body_model(root,mask,timesteps)


class PrecisionTests(unittest.TestCase):
    def check_path(self,precision,device):
        torch.manual_seed(17)
        model=adapt(Denoiser().eval(),'fp32')
        original_stats=model.motion_rep.statistics.clone()
        original_joints=model.motion_rep.skeleton.joints.clone()
        model=apply_precision(model,precision).to(device).eval()
        dtype=torch.float16 if precision=='fp16' else torch.bfloat16
        self.assertTrue(all(p.dtype==dtype for p in model.parameters()))
        self.assertEqual(model.motion_rep.statistics.dtype,torch.float32)
        self.assertEqual(model.motion_rep.skeleton.joints.dtype,torch.float32)
        torch.testing.assert_close(model.motion_rep.statistics.cpu(),original_stats,atol=0,rtol=0)
        torch.testing.assert_close(model.motion_rep.skeleton.joints.cpu(),original_joints,atol=0,rtol=0)
        x=torch.randn(2,7,32,device=device)
        mask=torch.zeros(2,7,dtype=torch.bool,device=device);mask[1,-2:]=True
        timesteps=torch.tensor([1,2],device=device)
        with torch.inference_mode():expected=model(x,mask,timesteps)
        self.assertEqual(expected.dtype,torch.float32)
        self.assertTrue(torch.isfinite(expected).all())
        self.assertEqual(model.geometry_dtype,torch.float32)
        self.assertEqual(model.root_model.stage.activation_dtype,dtype)
        self.assertEqual(model.root_model.stage.seen_dtypes,(torch.float32,torch.bool,torch.int64))
        swap=BlockSwap(model,motion_blocks(model),2)
        try:
            swap.move(device)
            with torch.inference_mode():actual=model(x,mask,timesteps)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            self.assertTrue(all(p.dtype==dtype for p in model.parameters()))
            if device=='cuda':
                self.assertEqual(model.root_model.stage.seqTransEncoder.layers[0].linear1.weight.device.type,'cpu')
        finally:swap.close()

    def test_cpu_half_stages_keep_geometry_float32(self):
        for precision in ('fp16','bf16'):
            with self.subTest(precision=precision):self.check_path(precision,'cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_half_stages_and_swap(self):
        for precision in ('fp16','bf16'):
            if precision=='bf16' and not torch.cuda.is_bf16_supported():continue
            with self.subTest(precision=precision):self.check_path(precision,'cuda')

    def test_profile_validation_and_unsupported_bf16(self):
        for precision in ('fp32','fp16','bf16','int8_convrot'):
            self.assertEqual(ModelProfile(label='test',path='same-directory',precision=precision).precision,precision)
        with patch('torch.cuda.device'),patch('torch.cuda.is_bf16_supported',return_value=False):
            with self.assertRaisesRegex(ValueError,'does not support native BF16'):
                validate_precision('bf16','cuda')


if __name__=='__main__':unittest.main()
