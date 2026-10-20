# Bundled upstream inference source

This server includes the inference portion of NVIDIA Kimodo, revision
`e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954`, from
https://github.com/nv-tlabs/kimodo (Apache-2.0).

`kimodo/` contains the model, patched LLM2Vec encoder, motion representation,
constraints, skeleton definitions/bind assets and BVH/NPZ exporters used by this
service. Demo UI, training/metrics, viewers and G1 rendering meshes/XML are omitted.
`kimodo/LICENSE` and `kimodo/ATTRIBUTIONS.MD` retain the upstream notices,
including MIT LLM2Vec components. No separate Kimodo installation is needed.
The bundled LLM2Vec loader forwards local-only loading to its tokenizer and
configuration loader, with remote modeling code disabled.

The optional C++ postprocessor source is in `native/MotionCorrection/` under the
same upstream Apache-2.0 license. Its CMake configuration requires already
installed pybind11/Eigen libraries instead of fetching Git repositories.
Building this optional extension is not required for normal inference.

ConvRot INT8 code in `kimodo_server/quantization.py` is the existing local
adaptation referencing feice-huang/ConvRot revision
`c80281c25ad99eb4837abb4317978e12c05d05e1`. It does not clone or import that repository.

Python libraries and model weights remain separately installed/downloaded
dependencies under their own licenses. Model downloads do not install executable
remote modeling code; inference uses the source included here.
