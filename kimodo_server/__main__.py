import argparse
import json
import os
from pathlib import Path
import tempfile
import threading


def comparison(config,args):
    import numpy as np
    from .runtime import MotionRuntime
    from .schemas import GenerateRequest
    runtime = MotionRuntime(config)
    event = threading.Event()
    rows = []
    if config.models[args.reference].skeleton != config.models[args.candidate].skeleton:
        raise ValueError('Comparison requires the same skeleton family on both sides')
    try:
        config.data_dir.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='compare-',dir=config.data_dir) as folder:
            files = {}
            for model_id in (args.reference,args.candidate):
                for seed in args.seeds:
                    for i,prompt in enumerate(args.prompts):
                        path = Path(folder)/f'{model_id}-{i}-{seed}.npz'
                        req = GenerateRequest(model=model_id,prompts=[prompt],durations=[args.duration],
                            seed=seed,steps=args.steps,block_swap=args.block_swap,post_processing=False,output_format='npz')
                        metadata = runtime.generate(req,event,lambda *a:None,path)
                        files[model_id,i,seed] = path,metadata
            indices = [runtime.motion.output_skeleton.bone_index[name]
                       for name in runtime.motion.skeleton.bone_order_names]
            count = len(indices)
            for seed in args.seeds:
                for i,prompt in enumerate(args.prompts):
                    path_a,info_a = files[args.reference,i,seed]
                    path_b,info_b = files[args.candidate,i,seed]
                    with np.load(path_a,allow_pickle=False) as a,np.load(path_b,allow_pickle=False) as b:
                        positions = [m['posed_joints'][:,indices].astype(np.float64) for m in (a,b)]
                        roots = [m['root_positions'].astype(np.float64) for m in (a,b)]
                        rel = np.swapaxes(a['local_rot_mats'][:,indices].astype(np.float64),-1,-2) @ b['local_rot_mats'][:,indices]
                        cos = np.clip((np.trace(rel,axis1=-2,axis2=-1)-1)/2,-1,1)
                        skew = np.stack([rel[...,2,1]-rel[...,1,2],rel[...,0,2]-rel[...,2,0],rel[...,1,0]-rel[...,0,1]],-1)
                        rotation = np.degrees(np.arctan2(np.linalg.norm(skew,axis=-1)/2,cos))
                        rows.append({'prompt':prompt,'seed':seed,'frames':len(positions[0]),
                            f'world_mpjpe_{count}_mm':float(np.linalg.norm(positions[1]-positions[0],axis=-1).mean()*1000),
                            f'root_relative_mpjpe_{count}_mm':float(np.linalg.norm((positions[1]-roots[1][:,None])-(positions[0]-roots[0][:,None]),axis=-1).mean()*1000),
                            'root_ade_mm':float(np.linalg.norm(roots[1]-roots[0],axis=-1).mean()*1000),
                            f'local_rotation_{count}_deg':float(rotation.mean()),
                            'reference_seconds':info_a['motion_seconds'],'candidate_seconds':info_b['motion_seconds']})
        report = {'reference':args.reference,'candidate':args.candidate,'steps':args.steps,
            'predicted_joints':count,
            'text_encoder':config.encoders[config.default_encoder].kind,
            'approximate_text_conditioner':config.encoders[config.default_encoder].kind=='awq_llm2vec',
            'scope':'Paired generated-motion drift with shared text embeddings, seed, sampler and settings; not a naturalness score.',
            'block_swap':args.block_swap,
            'samples':rows,'mean':{k:sum(row[k] for row in rows)/len(rows) for k in rows[0] if isinstance(rows[0][k],float)}}
        target = Path(args.output)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        runtime.close()


def main():
    from . import check_bundled_core
    check_bundled_core()
    parser = argparse.ArgumentParser(description='Kimodo motion inference service and quantization tools')
    commands = parser.add_subparsers(dest='command',required=True)
    serve = commands.add_parser('serve',help='Start the HTTP inference API')
    serve.add_argument('--config',default='config.json')
    serve.add_argument('--host')
    serve.add_argument('--port',type=int)
    quant = commands.add_parser('quantize',help='Create a new local ConvRot INT8 W8A8 checkpoint')
    quant.add_argument('--source',required=True)
    quant.add_argument('--output',required=True)
    quant.add_argument('--device',default='cuda')
    commands.add_parser('self-test',help='Verify rotation, INT8 GEMM and attention mask parity on CUDA')
    download = commands.add_parser('download',help='Explicitly download a configured model or encoder')
    download.add_argument('--config',default='config.json')
    target = download.add_mutually_exclusive_group(required=True)
    target.add_argument('--model')
    target.add_argument('--encoder')
    download.add_argument('--endpoint',default='https://huggingface.co')
    compare = commands.add_parser('compare',help='Measure paired generated-motion drift between model profiles')
    compare.add_argument('--config',default='config.json')
    compare.add_argument('--reference',default='soma-fp32')
    compare.add_argument('--candidate',default='soma-int8')
    compare.add_argument('--prompts',nargs='+',required=True)
    compare.add_argument('--seeds',nargs='+',type=int,default=[42])
    compare.add_argument('--duration',type=float,default=4)
    compare.add_argument('--steps',type=int,default=100)
    compare.add_argument('--block-swap',type=int,default=0)
    compare.add_argument('--output',default='data/comparison.json')
    args = parser.parse_args()
    if args.command=='download':
        from .config import load_config
        from .downloads import download_items, download_worker
        config = load_config(args.config)
        from fastapi import HTTPException
        try:
            items = download_items(config,'model' if args.model else 'encoder',args.model or args.encoder)
        except HTTPException as exc:
            parser.error(str(exc.detail))
        if any(item['source']=='huggingface' for item in items) and os.environ.get('HF_TOKEN') and not args.endpoint.startswith('https://'):
            parser.error('带令牌的下载源必须使用 HTTPS')
        for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):
            os.environ.pop(name,None)
        os.environ['HF_ENDPOINT'] = args.endpoint
        download_worker(items,args.endpoint)
        return
    # Inference is strictly local; no fallback model downloads are triggered.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if args.command in {'quantize','self-test'}:
        import torch
        from .quantization import quantize_model,self_test
        with torch.inference_mode():
            if args.command=='self-test':
                self_test()
            else:
                print(json.dumps(quantize_model(args.source,args.output,args.device),indent=2))
        return
    from .config import api_token,load_config
    config = load_config(args.config)
    if args.command=='compare':
        comparison(config,args)
        return
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if config.host not in {'127.0.0.1','localhost','::1'} and not api_token():
        parser.error('Set KIMODO_API_TOKEN before exposing the server outside localhost')
    import uvicorn
    from .api import create_app
    while True:
        reload_requested = False
        server = None
        def restart():
            nonlocal reload_requested
            reload_requested = True
            server.should_exit = True
        app = create_app(config,config_path=args.config,restart=restart)
        server = uvicorn.Server(uvicorn.Config(app,host=config.host,port=config.port,workers=1))
        server.run()
        if not reload_requested:
            break
        config = load_config(args.config)


if __name__=='__main__':
    main()
