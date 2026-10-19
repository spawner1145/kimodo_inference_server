"""Verify identical outputs and residency across block-swap on CUDA."""
import unittest
import torch
from torch import nn

from kimodo_server.block_swap import BlockSwap


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class BlockSwapTests(unittest.TestCase):
    def test_same_output_and_block_residency(self):
        torch.manual_seed(23)
        model = nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,64),nn.Linear(64,64)).eval().cuda()
        x = torch.randn(3,64,device='cuda')
        with torch.inference_mode():
            expected = model(x)
        swap = BlockSwap(model,[model[0],model[2],model[3]],2)
        swap.move('cuda')
        self.assertEqual(model[0].weight.device.type,'cpu')
        self.assertEqual(model[3].weight.device.type,'cuda')
        with torch.inference_mode():
            actual = model(x)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        self.assertEqual(model[0].weight.device.type,'cpu')
        self.assertEqual(swap.transfers,2)
        swap.move('cpu')
        self.assertFalse(swap.handles)
        self.assertTrue(all(p.device.type=='cpu' for p in model.parameters()))
        swap.close()

    def test_forward_exception_still_offloads(self):
        class Broken(nn.Linear):
            def forward(self,x):
                raise RuntimeError('test failure')
        model = nn.Sequential(Broken(32,32))
        swap = BlockSwap(model,list(model),1)
        swap.move('cuda')
        with self.assertRaises(RuntimeError):
            model(torch.zeros(1,32,device='cuda'))
        self.assertEqual(model[0].weight.device.type,'cpu')
        swap.close()


if __name__=='__main__':
    unittest.main()
