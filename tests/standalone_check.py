"""Copy the server, block original source/network access, optionally run real inference."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    infer = '--infer' in sys.argv
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='standalone-check-', dir=data) as temp:
        moved = Path(temp) / 'server'
        moved.mkdir()
        for name in ('kimodo', 'kimodo_server'):
            shutil.copytree(ROOT / name, moved / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(ROOT / 'config.example.json', moved / 'config.example.json')
        if infer:
            for name in ('Kimodo-SOMA-RP-v1.2', 'text_encoders/KIMODO-Meta3_llm2vec_NF4'):
                folder = ROOT / 'models' / name
                if not folder.is_dir():
                    raise FileNotFoundError('Validation requires local weights: ' + str(folder))
                for source in folder.rglob('*'):
                    if not source.is_file() or '.cache' in source.parts:
                        continue
                    destination = moved / 'models' / source.relative_to(ROOT / 'models')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source, destination)
        script = r'''
import sys, pathlib, os, json, threading
root, original = map(lambda x: pathlib.Path(x).resolve(), sys.argv[1:3])
sys.path.insert(0, str(root))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
forbidden = [original / n for n in ('kimodo', 'kimodo_server', 'models')]
forbidden += [original.parent / n for n in ('kimodo', 'MotionCorrection', 'models', 'Kimodo_Blender_Bridge')]
def guard(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError('Standalone check disallows network')
    if event in ('open', 'os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = pathlib.Path(os.fsdecode(args[0])).resolve()
        if any(path == p or p in path.parents for p in forbidden):
            raise RuntimeError('External source/model access rejected: ' + str(path))
sys.addaudithook(guard)
import kimodo_server
kimodo_server.check_bundled_core()
import kimodo
from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
from kimodo.skeleton import build_skeleton
from kimodo.exports.bvh import motion_to_bvh
import torch
for module in (kimodo, sys.modules[LlamaBiModel.__module__]):
    assert root in pathlib.Path(module.__file__).resolve().parents
    print('BUNDLED_MODULE', module.__file__, flush=True)
for count in (77, 30, 22, 34):
    skeleton = build_skeleton(count)
    assert skeleton.nbjoints == count
    text = motion_to_bvh(torch.eye(3).expand(3, count, 3, 3).clone(),
                         torch.zeros(3, 3), skeleton=skeleton, fps=30, standard_tpose=True)
    assert 'Frames: 3' in text
print('RELOCATED_SKELETON_EXPORT_OK', flush=True)
from transformers import LlamaConfig
config = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)
encoder = LlamaBiModel(config).eval()
with torch.inference_mode():
    output = encoder(input_ids=torch.tensor([[1, 2, 3]]))
assert torch.isfinite(output.last_hidden_state).all()
del encoder, output
print('RELOCATED_TEXT_CORE_OK', flush=True)
if (root / 'models').exists():
    from kimodo_server.config import load_config
    from kimodo_server.runtime import MotionRuntime
    from kimodo_server.schemas import GenerateRequest
    config = load_config(root / 'config.example.json')
    config.default_encoder = 'nf4'
    config.default_model = 'soma-bf16'
    runtime = MotionRuntime(config)
    try:
        req = GenerateRequest(model='soma-bf16', encoder='nf4',
                              prompts=['A person waves with the right hand.'], durations=[0.4],
                              steps=5, output_format='bvh', block_swap=4)
        path = root / 'data' / 'sample.bvh'
        result = runtime.generate(req, threading.Event(), lambda *a: print('STAGE', a[0], flush=True), path)
        assert result['frames'] == 12 and 'Frames: 12' in path.read_text('utf-8')
        print('RELOCATED_NF4_BF16_MOTION_OK', json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        runtime.close()
'''
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['PYTHONUTF8'] = '1'
        subprocess.run([sys.executable, '-I', '-u', '-X', 'utf8', '-c', script, str(moved), str(ROOT)],
                       cwd=moved, env=env, check=True)
        print('KIMODO_STANDALONE_CHECK_OK', flush=True)


if __name__ == '__main__':
    main()
