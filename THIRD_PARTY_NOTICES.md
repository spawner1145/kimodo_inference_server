# Sources and licenses

This service's integration code is distributed under Apache-2.0; see LICENSE.
NVIDIA Kimodo inference source and bind skeleton assets are bundled in
`kimodo/`; see `ATTRIBUTIONS.md`, `kimodo/LICENSE` and `kimodo/ATTRIBUTIONS.MD`.
No separate Kimodo repository/package is required. Model weights remain
separate downloads and retain their own model licenses.

The persistent-process, embedding-cache and staged-offload design was studied
from Lewdineer's Kimodo Blender Bridge and Aero-Ex/kimodo. The service HTTP
runtime and cache/residency management are newly implemented, independent of
Blender and the GPL Blender addon.

The local ConvRot INT8 implementation uses the radix-4 rotation's mathematical
sign pattern (block size 256) described in feice-huang/ConvRot, revision
c80281c25ad99eb4837abb4317978e12c05d05e1. It is not an author-released or
author-validated Kimodo INT8 implementation. No CUDA extension from that
repository is bundled. The implementation uses PyTorch tensor operations and
torch._int_mm. Original input/output projections remain FP32.

The portable AWQ executor follows the AutoAWQ GEMM INT4 packing convention.
LLM2Vec, PEFT, Transformers, bitsandbytes, Accelerate, PyTorch, FastAPI and
bvhio remain separately installed dependencies with their own licenses.
Prequantized Aero-Ex NF4 and original Llama weights are downloaded only by an
explicit command, subject to their respective access requirements and licenses.
