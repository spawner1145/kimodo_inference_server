"""Synchronous inference block swapping with deterministic residency.

Selected transformer blocks reside in CPU RAM, enter CUDA immediately before
forward, and return to CPU afterwards. No dtype/weight quantization is changed.
This trades PCIe transfers and latency for lower weight residency, not speed.
"""
from contextlib import contextmanager

import torch
from torch import nn


def motion_blocks(model):
    stacks = [list(module.layers) for module in model.modules() if isinstance(module,nn.TransformerEncoder)]
    # Interleave root/body stacks so a count is distributed across both stages.
    return [stack[i] for i in range(max(map(len,stacks),default=0)) for stack in stacks if i<len(stack)]


@contextmanager
def hide_blocks(model,blocks):
    selected = {id(b) for b in blocks}
    hidden = []
    try:
        for parent in list(model.modules()):
            for key,child in list(parent._modules.items()):
                if id(child) in selected:
                    hidden.append((parent,key,child))
                    parent._modules[key] = nn.Identity()
        yield
    finally:
        for parent,key,child in hidden:
            parent._modules[key] = child


class BlockSwap:
    def __init__(self,model,blocks,count):
        if count<0 or count>len(blocks):
            raise ValueError(f'block_swap={count} exceeds {len(blocks)} available transformer blocks')
        self.model = model
        self.blocks = list(blocks[:count])
        self.total = len(blocks)
        self.handles = []
        self.device = torch.device('cpu')
        self.transfers = 0

    def _remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def move(self,device):
        self._remove_hooks()
        self.device = torch.device(device)
        if self.device.type!='cuda' or not self.blocks:
            self.model.to(device)
            return
        for block in self.blocks:
            block.to('cpu')
        # Never stage all weights in VRAM before offloading; omit swapped blocks
        # during the parent .to() to keep the initial memory peak bounded too.
        with hide_blocks(self.model,self.blocks):
            self.model.to(device)
        for block in self.blocks:
            self.handles.append(block.register_forward_pre_hook(self._before))
            self.handles.append(block.register_forward_hook(self._after,always_call=True))

    def _before(self,module,args):
        module.to(self.device)
        self.transfers += 1

    def _after(self,module,args,result):
        # .to(cpu) is blocking. CUDA work using these weights completes before
        # storage can be reclaimed/reused, including when forward raises.
        module.to('cpu')
        return result

    def close(self):
        self._remove_hooks()
        for block in self.blocks:
            block.to('cpu')

    def stats(self):
        return {'swapped_blocks':len(self.blocks),'total_blocks':self.total,
                'block_loads':self.transfers,'implementation':'synchronous_cpu_cuda'}
