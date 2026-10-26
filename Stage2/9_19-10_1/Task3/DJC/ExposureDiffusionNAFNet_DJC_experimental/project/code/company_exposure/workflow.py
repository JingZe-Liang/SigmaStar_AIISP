"""Explicit preparation, readiness, teacher, training and evaluation commands.

No import, preparation, CPU check or teacher command starts an optimizer.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from company_exposure.core import STUDENT_SHA, TEACHER_SHA, sha256, load_student, exposure_steps, mixed_loss, signed_signal
from company_exposure.data import write_json, prepare, centers, context, read_packed, make_masks

CODE = Path(__file__).resolve().parents[1]


def paths(root):
    root=Path(root).resolve()
    return root,root/'manifest.json',root/'weights/sid_Pg_naf2.pt',root/'weights/RViDeformer-L_CRVD.pth'


def parameter_errors(s):
    errors=[]
    if s['cfa'] not in ('RGGB','GRBG','GBRG','BGGR'):errors.append('CFA missing/unsupported')
    for key in ('black_dn12','shot_gain_dn12_per_e','read_variance_dn12_sq'):
        value=s.get(key)
        if value is None or np.asarray(value).shape!=(4,) or not np.isfinite(value).all():
            errors.append(key+' missing/invalid')
    if s.get('black_dn12') is not None and any(not 0<=v<4095 for v in s['black_dn12']):errors.append('black outside DN12')
    if s.get('shot_gain_dn12_per_e') is not None and min(s['shot_gain_dn12_per_e'])<=0:errors.append('nonpositive shot gain')
    if s.get('read_variance_dn12_sq') is not None and min(s['read_variance_dn12_sq'])<0:errors.append('negative read variance')
    return errors


def data_signature(manifest):
    fields=['id','sha256','role','splits','raw_spec','black_dn12','shot_gain_dn12_per_e','read_variance_dn12_sq','pooling_variance_factor']
    import hashlib
    payload=[{k:s.get(k) for k in fields} for s in manifest['streams']]
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def code_signature():
    import hashlib
    files=sorted((CODE/'company_exposure').glob('*.py'))+[CODE/'noise_model/raw_io.py',
        CODE/'vendor/ExposureDiffusion/models/arch/NAFNet.py',
        CODE/'vendor/RViDeformer/models/RViDeformer_arch.py',CODE/'scripts/probe_rvideformer_motion_teacher.py']
    return hashlib.sha256(''.join(sha256(p) for p in files).encode()).hexdigest()


def load_teacher(checkpoint, device):
    if sha256(checkpoint)!=TEACHER_SHA:raise ValueError('Teacher checkpoint hash mismatch')
    from scripts.probe_rvideformer_motion_teacher import load_teacher as load
    return load(checkpoint,device)


def cpu_check(root):
    root,manifest_path,student,teacher=paths(root)
    torch.set_num_threads(2);torch.manual_seed(20261018)
    hashes=[sha256(student),sha256(teacher)]
    model=load_student(CODE,student).eval()
    with torch.no_grad():
        out=exposure_steps(model,torch.full((1,4,32,32),150.),[24.],[3838.5])
    assert len(out)==3 and all(x.shape==(1,4,32,32) and torch.isfinite(x).all() for x in out)
    import hashlib
    def parameter_digest():
        h=hashlib.sha256()
        for p in model.parameters():h.update(p.detach().numpy().tobytes())
        return h.hexdigest()
    before=parameter_digest()
    model.train()
    signal=torch.full((2,4,32,32),150.)
    outputs=exposure_steps(model,signal,[24.,24.],[3838.5,3838.5],training=True)
    static=torch.ones(2,1,32,32,dtype=torch.bool);static[:,:,:,16:]=False
    loss,_=mixed_loss(outputs,signal-1,signal,static,~static,[3838.5]*2)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    assert before==parameter_digest()
    parameters=sum(p.numel() for p in model.parameters());del model
    model=load_teacher(teacher,'cpu')
    teacher_parameters=sum(p.numel() for p in model.parameters());del model
    assert hashes==[sha256(student),sha256(teacher)]
    report=dict(passed=True,student_parameters=parameters,teacher_parameters=teacher_parameters,
                student_three_steps=True,teacher_strict_load=True,teacher_forward_executed=False,
                weights_unchanged=True,actual_NAFNet_synthetic_gradient_passed=True,
                optimizer_updates=0,training_started=False,
                code_signature=code_signature(),student_sha256=hashes[0],teacher_sha256=hashes[1])
    write_json(root/'reports/cpu_selfcheck.json',report)
    return report


def cuda_check(root):
    root,_,student,teacher=paths(root)
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable: do not start teacher or training')
    with torch.no_grad():
        model=load_student(CODE,student,'cuda').eval()
        exposure_steps(model,torch.full((1,4,32,32),150.,device='cuda'),[24.],[3838.5]);del model
        model=load_teacher(teacher,'cuda')
        out=model(torch.full((1,8,4,64,64),.1,device='cuda'))
        assert out.shape==(1,8,4,64,64) and torch.isfinite(out).all()
        del model,out
    torch.cuda.empty_cache()
    report=dict(passed=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                code_signature=code_signature(),student_sha256=STUDENT_SHA,teacher_sha256=TEACHER_SHA,
                optimizer_updates=0,teacher_batch_started=False)
    write_json(root/'reports/cuda_selfcheck.json',report)
    return report


def readiness(root, include_teacher=True):
    root,mp,student,teacher=paths(root)
    manifest=json.loads(mp.read_text());errors=[]
    config=json.loads((root/'config.json').read_text())
    for key,value in [('precision','float32'),('virtual_exposure_ratio',100),('static_weight',1.),('motion_weight',.5)]:
        if config.get(key)!=value:errors.append('Unsupported fixed first-run setting: '+key)
    for p,expected in [(student,STUDENT_SHA),(teacher,TEACHER_SHA)]:
        if not p.exists() or sha256(p)!=expected:errors.append(str(p.name)+': missing/hash mismatch')
    development=[s for s in manifest['streams'] if s['role']=='development']
    counts={r:sum(s['role']==r for s in manifest['streams']) for r in ('development','test','calibration')}
    if counts!={'development':20,'test':7,'calibration':18}:errors.append('Full collection roles/counts changed')
    if any('/shdarkroom/' in s['source_relative'] and s['role']!='test' for s in manifest['streams']):
        errors.append('MIS20S1 indoor holdout role was changed')
    test_hashes={s['sha256'] for s in manifest['streams'] if s['role']=='test'}
    if test_hashes.intersection(s['sha256'] for s in development):errors.append('test duplicate leakage')
    for s in manifest['streams']:
        if s['role']=='test':continue  # held-out RAW intentionally not deployed yet
        p=root/s['path']
        if not p.exists() or p.stat().st_size!=s['bytes'] or sha256(p)!=s['sha256']:
            errors.append(s['id']+': source missing/hash mismatch')
        if s['role']=='development':
            errors.extend(s['id']+': '+e for e in parameter_errors(s))
            if not all(centers(s,r) for r in ('train','validation')):errors.append(s['id']+': no valid contexts')
    cpu=root/'reports/cpu_selfcheck.json'
    if not cpu.exists() or json.loads(cpu.read_text()).get('code_signature')!=code_signature():
        errors.append('current CPU selfcheck missing')
    gpu=root/'reports/cuda_selfcheck.json'
    if not torch.cuda.is_available():errors.append('CUDA unavailable')
    if not gpu.exists() or json.loads(gpu.read_text()).get('code_signature')!=code_signature():
        errors.append('current CUDA compatibility check pending')
    if shutil.disk_usage(root).free<2*1024**3:errors.append('less than 2 GiB free for training outputs')
    if include_teacher:
        index=root/'teacher/index.json'
        if not index.exists():errors.append('teacher labels pending; no static-only fallback')
        else:
            cache=json.loads(index.read_text())
            if cache.get('signature')!=data_signature(manifest) or cache.get('teacher_sha256')!=TEACHER_SHA:
                errors.append('teacher labels are stale')
            expected={(s['id'],r,t) for s in development for r in ('train','validation') for t in centers(s,r)}
            actual={(x['stream'],x['role'],x['frame']) for x in cache['items']}
            if expected!=actual:errors.append('teacher labels incomplete or unexpected')
            if len(actual)!=len(cache['items']):errors.append('duplicate teacher records')
            for item in cache['items']:
                p=root/item['path']
                if not p.exists() or sha256(p)!=item['sha256']:errors.append('teacher file checksum mismatch: '+item['path'])
            if cache.get('code_signature')!=code_signature():errors.append('teacher masks/code changed')
            if cache.get('teacher_amp',False)!=config.get('teacher_amp',False):errors.append('teacher inference precision changed')
            for s in development:
                for role in ('train','validation'):
                    if not any(x['stream']==s['id'] and x['role']==role and
                               x.get('static_fraction',0)+x.get('motion_fraction',0)>0 for x in cache['items']):
                        errors.append(s['id']+': '+role+' has no supervised teacher-cache contexts')
    report=dict(ready=not errors,errors=errors,training_started=False,optimizer_updates=0,
                signature=data_signature(manifest),code_signature=code_signature(),
                teacher_required=include_teacher)
    write_json(root/'reports/readiness.json',report)
    return report


@torch.no_grad()
def teacher_frame(model, signal, span, *, motion=None, tile_batch=1, device='cuda', amp=False):
    from scripts.probe_rvideformer_motion_teacher import to_teacher,from_teacher,positions
    official=to_teacher(np.clip(signal,0,span))/span
    _,_,h,w=official.shape
    result=torch.zeros(1,4,h,w);norm=torch.zeros(1,1,h,w)
    if tile_batch<1:raise ValueError('Positive teacher tile batch required')
    # Every overlapping tile touching a supervised pixel is retained, so sparse
    # inference has the same weighted prediction there as full-frame tiling.
    needed=np.ones((h,w),bool) if motion is None else motion[0,::-1,:]
    coords=[(y,x) for y in positions(h) for x in positions(w) if needed[y:y+64,x:x+64].any()]
    outside,total=0,0
    for start in range(0,len(coords),tile_batch):
        group=coords[start:start+tile_batch]
        patch=torch.from_numpy(np.stack([official[:,:,y:y+64,x:x+64] for y,x in group])).to(device)
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp and str(device).startswith('cuda')):
            predictions=model(patch)[:,3].float().cpu()
        if not torch.isfinite(predictions).all():raise ValueError('Teacher nonfinite output')
        outside+=int(((predictions<0)|(predictions>1)).sum());total+=predictions.numel()
        for (y,x),prediction in zip(group,predictions):
            prediction=prediction[None]
            weight=torch.ones(1,1,64,64);ramp=torch.linspace(0,1,9)[1:]
            if y>0:weight[:,:,:8]*=ramp[None,None,:,None]
            if y+64<h:weight[:,:,-8:]*=ramp.flip(0)[None,None,:,None]
            if x>0:weight[:,:,:,:8]*=ramp[None,None,None,:]
            if x+64<w:weight[:,:,:,-8:]*=ramp.flip(0)[None,None,None,:]
            result[:,:,y:y+64,x:x+64]+=prediction.clamp(0,1)*weight
            norm[:,:,y:y+64,x:x+64]+=weight
    if needed.any() and not (norm[0,0].numpy()[needed]>0).all():raise ValueError('Uncovered teacher motion pixels')
    return from_teacher((result/norm.clamp_min(1e-12)).numpy())[0]*span, outside/max(total,1)


def prefetch_map(function, items):
    """One CPU job ahead, bounded RAM; exceptions propagate before consumption."""
    iterator=iter(items)
    with ThreadPoolExecutor(max_workers=1) as pool:
        item=next(iterator,None)
        if item is None:return
        future=pool.submit(function,item)
        while True:
            value=future.result();item=next(iterator,None)
            future=pool.submit(function,item) if item is not None else None
            yield value
            if future is None:return


def generate_teacher(root):
    root,mp,_,teacher=paths(root)
    gate=readiness(root,include_teacher=False)
    if not gate['ready']:raise ValueError('Teacher prerequisites failed: '+ '; '.join(gate['errors']))
    manifest=json.loads(mp.read_text());development=[s for s in manifest['streams'] if s['role']=='development']
    cache=root/'teacher'
    if (cache/'index.json').exists():raise ValueError('Completed teacher cache exists; do not overwrite')
    config=json.loads((root/'config.json').read_text())
    budget=int(config['teacher_cache_gib']*1024**3)
    if shutil.disk_usage(root).free<2*1024**3:
        raise ValueError('Teacher cache free-space failure')
    model=load_teacher(teacher,'cuda');items=[];used=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file()) if cache.exists() else 0
    jobs=[(s,role,t) for s in development for role in ('train','validation') for t in centers(s,role)]
    def prepare_job(job):
        s,role,t=job
        _,indices=context(s,role,t)
        static,motion,stats=make_masks(root,s,role,t)
        signal=signed_signal(read_packed(root,s,indices),s['black_dn12']) if motion.any() else None
        return s,role,t,indices,static,motion,stats,signal
    for s,role,t,indices,static,motion,stats,signal in prefetch_map(prepare_job,jobs):
        span=np.asarray(4095-np.asarray(s['black_dn12'],np.float32)).reshape(1,4,1,1)
        # to_teacher permutes channels spatially, so normalize in canonical space first.
        dest=cache/s['id']/role/f'{t:04d}.pt'
        if dest.exists():raise ValueError('Partial teacher output exists: '+str(dest))
        estimated=static.nbytes+motion.nbytes+int(motion.sum())*4*4+16384
        if used+estimated>budget or shutil.disk_usage(root).free<estimated+2*1024**3:
            raise ValueError('Teacher cache budget/free space exhausted before inference')
        if motion.any():
            # Scalar 1 here because values have already been normalized per canonical channel.
            pred,clipped=teacher_frame(model,signal/span,1.,motion=motion,tile_batch=config.get('teacher_tile_batch',1),amp=config.get('teacher_amp',False))
            pred=pred*span[0]
        else:pred=np.zeros((4,*static.shape[1:]),np.float32);clipped=0.
        if not np.isfinite(pred).all() or pred.min()<0 or pred.max()>4095:
            raise ValueError('Invalid teacher DN12')
        # Lossless sparse supervision: static teacher pixels are unused.
        # Store only valid motion-site vectors, retaining float32 DN12.
        payload=dict(motion_signal_dn12=torch.from_numpy(pred[:,motion[0]].T.copy().astype(np.float32)),
                     shape=list(pred.shape),
                     static=torch.from_numpy(static),motion=torch.from_numpy(motion),
                     stream=s['id'],frame=t,role=role,input_frames=indices,
                     source_sha256=s['sha256'],teacher_sha256=TEACHER_SHA,
                     is_clean_gt=False,stats=stats,preclip_fraction=clipped,
                     label_mode='teacher' if motion.any() else 'no_motion_teacher_not_needed')
        dest.parent.mkdir(parents=True,exist_ok=True)
        torch.save(payload,dest.with_suffix('.pt.part'));dest.with_suffix('.pt.part').replace(dest)
        items.append(dict(stream=s['id'],role=role,frame=t,path=str(dest.relative_to(root)),sha256=sha256(dest),
                          static_fraction=stats['static_fraction'],motion_fraction=stats['motion_fraction']))
        used+=dest.stat().st_size
        if used>budget:raise ValueError('Teacher cache exceeds budget')
        write_json(root/'reports/teacher_progress.json',dict(completed=len(items),latest=items[-1],training_started=False))
        print(json.dumps({'teacher_completed':len(items),'stream':s['id'],'frame':t}),flush=True)
    write_json(cache/'index.json',dict(items=items,signature=data_signature(manifest),code_signature=code_signature(),
               teacher_sha256=TEACHER_SHA,teacher_amp=config.get('teacher_amp',False),tile_batch=config.get('teacher_tile_batch',1),
               is_clean_gt=False,training_started=False))


def sample(root,stream,item,rng,size):
    label=torch.load(root/item['path'],map_location='cpu',weights_only=True)
    static,motion=label['static'].numpy(),label['motion'].numpy()
    h,w=static.shape[1:];desired=motion if rng.random()<.5 else static
    if not desired.any():desired=static|motion
    if not desired.any():raise ValueError('No supervised pixels: '+item['path'])
    for _ in range(64):
        y=int(rng.integers(0,h-size+1));x=int(rng.integers(0,w-size+1))
        if desired[:,y:y+size,x:x+size].mean()>=.1:break
    else:
        sites=np.argwhere(desired[0]);y0,x0=sites[int(rng.integers(len(sites)))]
        y,x=int(np.clip(y0-size//2,0,h-size)),int(np.clip(x0-size//2,0,w-size))
    obs=read_packed(root,stream,[item['frame'],item['frame']+6],(y,x,size))
    signal=signed_signal(obs,stream['black_dn12'])
    teacher=torch.zeros(label['shape'],dtype=torch.float32)
    teacher[:,label['motion'][0]]=label['motion_signal_dn12'].T
    return dict(signal=torch.from_numpy(signal[0]),target=torch.from_numpy(signal[1]),
                teacher=teacher[:,y:y+size,x:x+size],
                static=label['static'][:,y:y+size,x:x+size],motion=label['motion'][:,y:y+size,x:x+size],
                span=torch.tensor(4095-np.asarray(stream['black_dn12']),dtype=torch.float32),
                shot=torch.tensor(stream['shot_gain_dn12_per_e'],dtype=torch.float32))


def batch_samples(root,streams,items,rng,size,batch, *, device='cuda', workers=1, pin_memory=False):
    sensors=sorted({s['sensor'] for s in streams.values()});jobs=[]
    eligible={k:v for k,v in items.items() if v}
    for _ in range(batch):
        sensor=rng.choice(sensors)
        keys=[k for k in eligible if streams[k]['sensor']==sensor]
        key=rng.choice(keys);item=eligible[key][int(rng.integers(len(eligible[key])))]
        jobs.append((streams[key],item,int(rng.integers(0,2**63-1))))
    def build(job):
        stream,item,seed=job
        return sample(root,stream,item,np.random.default_rng(seed),size)
    if workers>1:
        with ThreadPoolExecutor(max_workers=workers) as pool:samples=list(pool.map(build,jobs))
    else:samples=[build(job) for job in jobs]
    values={k:torch.stack([s[k] for s in samples]) for k in samples[0]}
    if pin_memory:values={k:v.pin_memory() for k,v in values.items()}
    return {k:v.to(device,non_blocking=pin_memory) for k,v in values.items()}


def validate(model,root,streams,items,size):
    # Same selected contexts/crops and random Poisson draws at every checkpoint.
    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all();np_state=np.random.get_state()
    rng=np.random.default_rng(20261018);torch.manual_seed(20261018);np.random.seed(20261018)
    values={};model.eval()
    with torch.no_grad():
        for key in sorted(items):
            candidates=items[key]
            for item in [candidates[0],candidates[-1]]:
                s=sample(root,streams[key],item,rng,size)
                b={k:v[None].cuda() for k,v in s.items()}
                out=exposure_steps(model,b['signal'],b['shot'],b['span'])
                loss,_=mixed_loss(out,b['target'],b['teacher'],b['static'],b['motion'],b['span'])
                values.setdefault(streams[key]['sensor'],{}).setdefault(key,[]).append(float(loss))
    torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(gpu);np.random.set_state(np_state)
    model.train()
    return float(np.mean([np.mean([np.mean(v) for v in sensor.values()]) for sensor in values.values()]))


def train(root,run_id):
    root,mp,student,_=paths(root)
    gate=readiness(root)
    if not gate['ready']:raise ValueError('Training blocked before optimizer creation: '+'; '.join(gate['errors']))
    if not run_id or not all(c.isalnum() or c in '-_' for c in run_id):raise ValueError('Neutral alphanumeric run-id required')
    dest=root/'runs'/run_id
    if dest.exists():raise ValueError('Run already exists; refusing overwrite')
    config=json.loads((root/'config.json').read_text());manifest=json.loads(mp.read_text())
    streams={s['id']:s for s in manifest['streams'] if s['role']=='development'}
    index=json.loads((root/'teacher/index.json').read_text())
    groups={r:{k:[] for k in streams} for r in ('train','validation')}
    for item in index['items']:
        label=torch.load(root/item['path'],map_location='cpu',weights_only=True)
        if (label['static']|label['motion']).any():groups[item['role']][item['stream']].append(item)
    if any(not v for g in groups.values() for v in g.values()):raise ValueError('Every stream needs valid train and validation supervision')
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    rng=np.random.default_rng(config['seed']);model=load_student(CODE,student,'cuda').train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    dest.mkdir(parents=True);best=float('inf')
    batches=prefetch_map(lambda _:batch_samples(root,streams,groups['train'],rng,
        config['patch_packed'],config['batch_size'],device='cpu',pin_memory=True,
        workers=config.get('sample_workers',1)),range(config['optimizer_steps']*config['accumulate']))
    started=time.perf_counter()
    for step in range(1,config['optimizer_steps']+1):
        optimizer.zero_grad(set_to_none=True);records=[]
        for _ in range(config['accumulate']):
            b={k:v.to('cuda',non_blocking=True) for k,v in next(batches).items()}
            out=exposure_steps(model,b['signal'],b['shot'],b['span'],training=True)
            loss,metrics=mixed_loss(out,b['target'],b['teacher'],b['static'],b['motion'],b['span'])
            (loss/config['accumulate']).backward();records.append(metrics)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
        optimizer.step()
        write_json(dest/'status.json',dict(optimizer_updates=step,training_started=True,grad_norm=float(norm),metrics=records,
                   elapsed_seconds=time.perf_counter()-started,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   batch_size=config['batch_size'],accumulate=config['accumulate']))
        if step==1 or step%config['validate_every']==0 or step==config['optimizer_steps']:
            score=validate(model,root,streams,groups['validation'],config['patch_packed'])
            state=dict(netG=model.state_dict(),opt_g=optimizer.state_dict(),step=step,validation_loss=score,
                       initialization_sha256=STUDENT_SHA,data_signature=gate['signature'],config=config,
                       method='static N2N + motion teacher; target-free physical recurrence',company_clean_gt=False)
            torch.save(state,dest/f'step_{step:04d}.pt')
            if score<best:best=score;torch.save(state,dest/'best.pt')
            write_json(dest/'validation.json',dict(step=step,score=score,best=best,clean_psnr=None))
        print(json.dumps({'optimizer_updates':step,'metrics':records}),flush=True)


@torch.no_grad()
def evaluate(root,checkpoint,stream_id,frames,allow_heldout=False,references=None,display_report=None):
    from utils.rgb_metrics import packed_to_linear_rgb
    root,mp,student,_=paths(root);manifest=json.loads(mp.read_text())
    s=next(x for x in manifest['streams'] if x['id']==stream_id)
    if s['role']=='test' and not allow_heldout:raise ValueError('Held-out evaluation requires explicit --allow-heldout-evaluation')
    errors=parameter_errors(s)
    if errors:raise ValueError('Evaluation parameters missing: '+','.join(errors))
    if not torch.cuda.is_available():raise ValueError('Evaluation requires CUDA')
    frozen_sha=sha256(checkpoint);before=load_student(CODE,student,'cuda').eval()
    after=load_student(CODE,checkpoint,'cuda',pretrained=False).eval()
    import hashlib
    frame_key=hashlib.sha256(json.dumps(frames).encode()).hexdigest()[:6]
    out=root/'evaluation'/frozen_sha[:12]/stream_id/f'{frames[0]:04d}_{frames[-1]:04d}_{frame_key}'
    out.mkdir(parents=True,exist_ok=False)
    refs=json.loads(Path(references).read_text()) if references else {}
    reports=[];video=None;video_path=None;clip_wb=None;clip_scale=None
    if display_report:
        display=json.loads(Path(display_report).read_text())
        clip_wb=tuple(display['wb_red_blue']);clip_scale=float(display['shared_display_scale'])
        if len(clip_wb)!=2 or not all(np.isfinite(v) and v>0 for v in (*clip_wb,clip_scale)):
            raise ValueError('Invalid shared display report')
    contiguous=len(frames)>1 and all(b==a+1 for a,b in zip(frames,frames[1:]))
    for t in frames:
        observed=read_packed(root,s,[t]);signal=signed_signal(observed,s['black_dn12'])
        span=4095-np.asarray(s['black_dn12'],np.float32)
        tensor=torch.from_numpy(signal).cuda()
        torch.manual_seed(20261018+t)
        a=exposure_steps(before,tensor,[s['shot_gain_dn12_per_e']],[span])[-1].cpu()*torch.tensor(span)[None,:,None,None]
        torch.manual_seed(20261018+t)
        b=exposure_steps(after,tensor,[s['shot_gain_dn12_per_e']],[span])[-1].cpu()*torch.tensor(span)[None,:,None,None]
        if clip_wb is None:
            noisy,clip_wb=packed_to_linear_rgb(torch.from_numpy(signal).clamp_min(0)/4095,'GBRG',return_wb=True)
            clip_scale=max(float(np.quantile(noisy.numpy(),.995)),1e-6)
        wb,scale=clip_wb,clip_scale
        images=[]
        for label,value in [('Noisy',torch.from_numpy(signal)),('Before',a),('Finetuned',b)]:
            rgb=packed_to_linear_rgb(value/4095,'GBRG',wb_gains=wb)
            im=Image.fromarray((rgb[0].permute(1,2,0).numpy().clip(0)/scale).clip(0,1).__mul__(255).astype(np.uint8))
            im.save(out/f'{t:04d}_{label}.png');images.append((label,im))
        for label,path in refs.get(str(t),{}).items():
            im=Image.open(path).convert('RGB')
            if im.size!=images[0][1].size:raise ValueError('Reference is not full resolution')
            images.append((label,im))
        w,h=images[0][1].size;canvas=Image.new('RGB',(w*len(images),h+32))
        for i,(label,im) in enumerate(images):canvas.paste(im,(i*w,32));ImageDraw.Draw(canvas).text((i*w+8,8),label)
        canvas.save(out/f'{t:04d}_comparison.png')
        if contiguous:
            import cv2
            if video is None:
                video_path=out/'finetuned_preview_25fps.mp4'
                video=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),25.,(w,h))
                if not video.isOpened():raise ValueError('Video writer unavailable; full frame PNGs retained')
            video.write(np.array(images[2][1])[:,:,::-1].copy())
        torch.save({'signal_dn12':b,'is_clean_gt':False},out/f'{t:04d}_signal.pt')
        reports.append(dict(frame=t,wb=list(wb),display_scale=scale,panels=[k for k,_ in images]))
    if video is not None:video.release()
    write_json(out/'report.json',dict(checkpoint_sha256=frozen_sha,stream=stream_id,frames=reports,
               video=str(video_path) if video_path else None,preview_fps=25 if video_path else None,
               display_policy='one fixed noisy-derived or explicitly supplied WB/scale for entire clip; 99.5 percentile',
               clean_psnr=None,clean_ssim=None,heldout=s['role']=='test',evaluation_only=True,
               note='References must use the recorded shared ISP/WB/scale; absent methods are not fabricated. Ordered frame PNGs retain temporal evidence.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    sub=parser.add_subparsers(dest='command')
    p=sub.add_parser('prepare');p.add_argument('--fit-noise',action='store_true');p.add_argument('--overrides',type=Path)
    sub.add_parser('cpu-check')
    p=sub.add_parser('check-ready');p.add_argument('--cuda-check',action='store_true')
    sub.add_parser('generate-teacher')
    p=sub.add_parser('train');p.add_argument('--run-id',required=True)
    p=sub.add_parser('evaluate');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--stream',required=True)
    p.add_argument('--frames',nargs='+',type=int,required=True);p.add_argument('--allow-heldout-evaluation',action='store_true');p.add_argument('--references',type=Path);p.add_argument('--display-report',type=Path)
    args=parser.parse_args()
    if args.command is None:parser.print_help();return
    root=args.root.resolve()
    for name in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','HF_HOME','PIP_CACHE_DIR']:
        directory=root/'runtime'/name.lower();directory.mkdir(parents=True,exist_ok=True);os.environ[name]=str(directory)
    torch.set_num_threads(2)
    if args.command=='prepare':result=prepare(root,root/'manifest.json',args.fit_noise,args.overrides)
    elif args.command=='cpu-check':result=cpu_check(root)
    elif args.command=='check-ready':
        if args.cuda_check:cuda_check(root)
        result=readiness(root)
    elif args.command=='generate-teacher':result=generate_teacher(root)
    elif args.command=='train':result=train(root,args.run_id)
    else:result=evaluate(root,args.checkpoint,args.stream,args.frames,args.allow_heldout_evaluation,args.references,args.display_report)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if args.command=='check-ready' and not result['ready']:sys.exit(2)


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError) as e:
        print(json.dumps({'error':str(e),'automatic_training_started':False}),file=sys.stderr);sys.exit(2)
