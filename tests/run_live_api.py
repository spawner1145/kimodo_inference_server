"""Opt-in real model integration checks against an already running server."""
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np

URL = 'http://127.0.0.1:6600'
OUT = Path(__file__).resolve().parents[1]/'data'/'verification'
OUT.mkdir(parents=True,exist_ok=True)


def call(path,body=None):
    data = json.dumps(body).encode() if body is not None else None
    return json.load(urllib.request.urlopen(urllib.request.Request(URL+path,data=data,
        headers={'Content-Type':'application/json'}),timeout=20))


def wait(job):
    end = time.monotonic()+240
    last = ''
    while job['state'] not in {'succeeded','failed','cancelled'}:
        if job['message']!=last:
            print(job['message'],flush=True)
            last = job['message']
        if time.monotonic()>end:
            raise TimeoutError(job)
        time.sleep(0.2)
        job = call('/v1/jobs/'+job['id'])
    return job


def generate(label,**options):
    request = dict(model='soma-fp32',encoder='awq',prompts=['A person walks forward at a steady pace with natural arm swings.'],
        durations=[2],seed=42,steps=20,output_format='npz',block_swap=0)
    request.update(options)
    result = wait(call('/v1/jobs',request))
    if result['state']!='succeeded':
        raise AssertionError(result)
    artifact = urllib.request.urlopen(URL+'/v1/jobs/'+result['id']+'/artifact').read()
    (OUT/(label+'.'+request['output_format'])).write_bytes(artifact)
    reports[label] = result
    print('PASS',label,result['result'],flush=True)
    if request['output_format']=='npz':
        with np.load(io.BytesIO(artifact),allow_pickle=False) as data:
            arrays = {k:v.copy() for k,v in data.items()}
        assert all(np.isfinite(v).all() for v in arrays.values())
        return arrays
    assert b'Frames: 60' in artifact


reports = {'health':call('/v1/health'),'models':call('/v1/models'),'encoders':call('/v1/encoders')}
try:
    ref = generate('fp32')
    swapped = generate('fp32_swap16',block_swap=16)
    assert all(np.array_equal(ref[k],swapped[k]) for k in ref),'Block swap changed generated motion'
    print('PASS motion block swap preserves every output element',flush=True)
    text_swap = generate('text_swap8',prompts=['A person takes a careful step backward.'],text_block_swap=8,steps=5)
    text_full = generate('text_swap8_cached',prompts=['A person takes a careful step backward.'],text_block_swap=0,steps=5)
    assert all(np.array_equal(text_swap[k],text_full[k]) for k in text_swap),'Cached text feature changed motion'
    quant = generate('int8',model='soma-int8')
    quant_swap = generate('int8_swap16',model='soma-int8',block_swap=16)
    assert all(np.array_equal(quant[k],quant_swap[k]) for k in quant),'INT8 swap changed generated motion'
    generate('int8_bvh',model='soma-int8',output_format='bvh')
    multi = generate('multi',prompts=['A person walks forward.','A person waves with the right hand.'],durations=[1,1],seeds=[17,42])
    assert multi['root_positions'].shape==(60,3)
    repeat = generate('multi_repeat',prompts=['A person walks forward.','A person waves with the right hand.'],durations=[1,1],seeds=[17,42])
    assert all(np.array_equal(multi[k],repeat[k]) for k in multi),'Segment seeds are not repeatable'
    job = call('/v1/jobs',dict(prompts=['A person walks forward.'],durations=[30],seed=42,steps=1000,model='soma-fp32'))
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        job=call('/v1/jobs/'+job['id'])
        if job['state']=='running' and job['progress']>0.22:
            break
        time.sleep(0.1)
    call('/v1/jobs/'+job['id']+'/cancel',{})
    reports['cancel'] = wait(job)
    assert reports['cancel']['state']=='cancelled'
    try:
        urllib.request.urlopen(URL+'/v1/jobs/'+job['id']+'/artifact')
        raise AssertionError('Cancelled job had an artifact')
    except urllib.error.HTTPError as exc:
        assert exc.code==409
    generate('after_cancel',steps=5)
    job = call('/v1/jobs',dict(prompts=['A person walks forward.','A person waves with the right hand.'],
            durations=[5,5],seed=42,seeds=[17,42],steps=1000,model='soma-fp32'))
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        job=call('/v1/jobs/'+job['id'])
        if job['state']=='running' and job['progress']>0.22:
            break
        time.sleep(0.1)
    call('/v1/jobs/'+job['id']+'/cancel',{})
    reports['cancel_multi']=wait(job)
    assert reports['cancel_multi']['state']=='cancelled'
    reports['success'] = True
finally:
    (OUT/'live_api.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
