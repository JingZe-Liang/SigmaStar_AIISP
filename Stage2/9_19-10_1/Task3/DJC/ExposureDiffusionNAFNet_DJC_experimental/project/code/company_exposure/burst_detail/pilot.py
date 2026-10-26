"""Bounded real-observation pilot. No synthetic noise or automatic continuation."""
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from company_exposure.core import (TEACHER_SHA, guidance_masks,
                                    sha256, signed_signal)
from company_exposure.data import read_packed, write_json
from company_exposure.temporal.data import temporal_context
from company_exposure.workflow import code_signature, data_signature, parameter_errors, prefetch_map
from company_adaptation_v2 import BudgetExpired, infer_tiles
from .model import BASE_SHA, ARCHITECTURE, load_model
from .workflow import signature, cpu_check, config as architecture_config, final_mixed_loss


def context(stream, role, frame):
    inputs, guides, target = temporal_context(stream, role, frame)
    lo, hi = stream['splits'][role]
    candidates = [i for i in range(lo, hi+1) if i % 2 and i not in [*inputs,*guides,target]]
    extra = sorted(candidates, key=lambda i: abs(i-frame), reverse=True)[:2]
    if len(extra) != 2:
        raise ValueError('Two independent long-baseline guides required')
    guides = sorted([*guides,*extra])
    assert len(guides)==8 and set(guides).isdisjoint([*inputs,target])
    return inputs,guides,target


def conservative_masks(guides, stream):
    """Intersect old pooled test with ALL-channel, two-scale consistency.

    Guidance is disjoint from the source and target. Finite guide evidence
    cannot prove stationarity between frames; rejected pixels stay ignored.
    """
    static,motion,stats = guidance_masks(guides,stream['shot_gain_dn12_per_e'],
        stream['read_variance_dn12_sq'],stream['black_dn12'],stream.get('pooling_variance_factor'))
    legacy = torch.from_numpy(static.copy())[None]
    safe = torch.ones_like(legacy)
    old_count = int(legacy.sum())
    h,w = guides.shape[-2:]
    for c in range(4):
        x = torch.from_numpy((guides[:,c:c+1]-stream['black_dn12'][c]).copy())
        for kernel,stride in ((5,2),(15,4)):
            low = F.avg_pool2d(x,kernel,stride,kernel//2,count_include_pad=False)
            mean = low.mean(0,keepdim=True)
            factor = stream.get('pooling_variance_factor',[1/25]*4)[c]*25/(kernel*kernel)
            var = (mean*stream['shot_gain_dn12_per_e'][c]+stream['read_variance_dn12_sq'][c]).clamp_min(1)*factor
            # Pool the statistic BEFORE thresholding: independent noisy binary
            # decisions followed by erosion otherwise erase genuinely static areas.
            ratio=low.var(0,unbiased=True,keepdim=True)/var
            peak=(low-mean).square().amax(0,keepdim=True)/var
            consistent = F.avg_pool2d(ratio,3,1,1,count_include_pad=False)<2.
            consistent &= F.avg_pool2d(peak,3,1,1,count_include_pad=False)<12.
            safe &= F.interpolate(consistent.float(),size=(h,w),mode='nearest')>.5
    # Explicit padding rejects crop borders; valid areas erode by four packed pixels.
    safe = F.avg_pool2d(F.pad(safe.float(),(4,4,4,4)),9,1)>.9999
    safe &= legacy  # Do not erode the already-eroded legacy mask a second time.
    stats.update(previous_static_pixels=old_count,static_pixels=int(safe.sum()),
                 new_static_fraction=float(safe.float().mean()),guide_count=8)
    return safe[0].numpy(), motion & ~safe[0].numpy(), stats


def sample(root,stream,item,rng,size=192):
    inputs,guides,target = context(stream,item['role'],item['frame'])
    h,w = stream['raw_spec']['height']//2,stream['raw_spec']['width']//2
    if min(h,w)<size:
        raise ValueError('Crop exceeds RAW frame')
    label = torch.load(Path(root)/item['path'],map_location='cpu',weights_only=True)
    if (label['source_sha256']!=stream['sha256'] or label['teacher_sha256']!=TEACHER_SHA
            or label['frame']!=item['frame'] or label['role']!=item['role']
            or label['input_frames']!=list(range(item['frame']-3,item['frame']+5))):
        raise ValueError('Center teacher payload provenance changed')
    for _ in range(12):
        y,x = int(rng.integers(h-size+1)),int(rng.integers(w-size+1))
        crop=(y,x,size)
        observed=read_packed(root,stream,guides,crop)
        static,moving,stats=conservative_masks(observed,stream)
        motion = moving & label['motion'][:,y:y+size,x:x+size].numpy()
        if int((static|motion).sum())<max(16,size*size//200):
            continue
        observed = read_packed(root,stream,[*inputs,target],crop)
        signal = signed_signal(observed,stream['black_dn12'])
        dense=torch.zeros(label['shape'],dtype=torch.float32)
        dense[:,label['motion'][0]]=label['motion_signal_dn12'].T
        result=dict(signal=torch.from_numpy(signal[:5]),target=torch.from_numpy(signal[5]),
            teacher=dense[:,y:y+size,x:x+size].clone(),static=torch.from_numpy(static),
            motion=torch.from_numpy(motion),span=torch.tensor(4095-np.asarray(stream['black_dn12']),dtype=torch.float32),
            shot=torch.tensor(stream['shot_gain_dn12_per_e'],dtype=torch.float32),
            read=torch.tensor(stream['read_variance_dn12_sq'],dtype=torch.float32))
        return result,dict(stream=stream['id'],role=item['role'],frame=item['frame'],crop_yxs=[y,x,size],
            inputs=inputs,guides=guides,target_frame=target,teacher_center=item['frame'],
            motion_pixels=int(motion.sum()),**stats)
    raise ValueError('No supervised crop after 12 independent-guide trials')


def fingerprint(path):
    path=Path(path);s=path.stat()
    return dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)


def settings(path):
    cfg=json.loads(Path(path).read_text())
    if (cfg.get('architecture')!=ARCHITECTURE or cfg.get('base_sha256')!=BASE_SHA
            or cfg.get('supervision')!='real_n2n_conservative_static_center_rvi_motion_v1'
            or cfg.get('patch_packed')!=192 or cfg.get('precision')!='float32'
            or cfg.get('automatic_training') is not False
            or cfg.get('injection') != 'raw_output_residual_only' or cfg.get('loss_steps') != [3]):
        raise ValueError('Unsupported real-observation pilot settings')
    for k in ('maximum_seconds','maximum_updates','batch_size','accumulate','review_every'):
        if type(cfg[k]) is not int or cfg[k]<=0:
            raise ValueError('Explicit positive integer required: '+k)
    if cfg['maximum_updates']>50 or cfg['review_every']>10 or cfg['maximum_seconds']>1800:
        raise ValueError('First pilot is bounded to <=50 updates, <=10 before review, <=1800 seconds')
    if cfg['learning_rate']!=1e-4:
        raise ValueError('First branch pilot learning rate is 1e-4')
    if Path(cfg['base_checkpoint']).is_absolute() or '..' in Path(cfg['base_checkpoint']).parts:
        raise ValueError('Project-relative frozen base required')
    return cfg


def inventory(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    streams={s['id']:s for s in manifest['streams'] if s['role']=='development'}
    if len(streams)!=20 or sum(s['role']=='test' for s in manifest['streams'])!=7:
        raise ValueError('Full development/holdout collection required')
    heldout={s['sha256'] for s in manifest['streams'] if s['role']=='test'}
    if any(s['sha256'] in heldout or '/shdarkroom/' in s['source_relative'] or parameter_errors(s) for s in streams.values()):
        raise ValueError('Heldout contamination or missing operational RAW parameters')
    teacher=json.loads((root/'teacher/index.json').read_text())
    if (teacher['signature']!=data_signature(manifest) or teacher['code_signature']!=code_signature()
            or teacher['teacher_sha256']!=TEACHER_SHA):
        raise ValueError('Frozen center-teacher cache is stale')
    groups={role:{k:[] for k in streams} for role in ('train','validation')}
    for item in teacher['items']:
        try:context(streams[item['stream']],item['role'],item['frame'])
        except ValueError:continue  # Tighter context, with explicit per-stream coverage below.
        groups[item['role']][item['stream']].append(item)
    if any(not values for split in groups.values() for values in split.values()):
        raise ValueError('At least one stream lost train/validation context coverage')
    return manifest,streams,teacher,groups


def prepare(root,deploy,cfg):
    """CPU-only: hash original assets and try real 192 crops in all 40 groups."""
    root,deploy=Path(root),Path(deploy);torch.set_num_threads(1)
    manifest,streams,teacher,groups=inventory(root)
    stamps=[];records={role:{} for role in groups};coverage=[]
    def verify(path,expected):
        if sha256(path)!=expected:raise ValueError('Asset checksum mismatch: '+str(path))
        stamps.append(fingerprint(path))
    verify(root/cfg['base_checkpoint'],cfg['base_sha256'])
    stamps.extend(fingerprint(root/p) for p in ['manifest.json','teacher/index.json'])
    for s in streams.values():
        verify(root/s['path'],s['sha256'])
        print(json.dumps(dict(checked_source=s['id'])),flush=True)
    for item in teacher['items']:verify(root/item['path'],item['sha256'])
    for role,split in groups.items():
        for key,items in split.items():
            chosen=[]
            # Probe up to four times across the complete partition; future crops remain random.
            for index in sorted(set(np.linspace(0,len(items)-1,min(4,len(items)),dtype=int).tolist())):
                try:
                    _,meta=sample(root,streams[key],items[index],np.random.default_rng(cfg['seed']+index),192)
                except ValueError as e:
                    if not str(e).startswith('No supervised crop'):raise
                    continue
                chosen.append(items[index]);coverage.append(meta)
            if not chosen:raise ValueError('No real supervised crop in '+key+'/'+role)
            # Pilot uses up to four CPU-proven times per partition; spatial
            # crops remain newly sampled over the entire frame on every draw.
            records[role][key]=chosen
            print(json.dumps(dict(checked_group=key+'/'+role,static=meta['static_pixels'],motion=meta['motion_pixels'])),flush=True)
    arch=architecture_config(Path(__file__).resolve().parents[2]/'configs/exposure_paper_burst_20261022.json')
    cpu=cpu_check(root/cfg['base_checkpoint'],arch,deploy/'reports/architecture_cpu.json')
    report=dict(cpu_ready=True,gpu_required_to_start=True,training_started=False,optimizer_updates=0,
        project_root=str(root.resolve()),
        signature=signature(),settings=cfg,base_sha256=BASE_SHA,stamps=stamps,groups=records,
        coverage=coverage,checked_streams=20,checked_partitions=40,teacher_files=len(teacher['items']),
        retained_contexts=sum(len(v) for split in records.values() for v in split.values()),
        synthesis_noise_used=False,noise_parameters='existing approximate working parameters for conditioning/masks only',
        limitation='Conservative finite-guide masks reduce but cannot prove absence of motion contamination; RVi labels are soft targets.')
    write_json(deploy/'reports/ready.json',report)
    return report


def ready(root,deploy,cfg):
    report=json.loads((Path(deploy)/'reports/ready.json').read_text())
    if (not report['cpu_ready'] or report['signature']!=signature() or report['settings']!=cfg
            or report['project_root']!=str(Path(root).resolve())):
        raise ValueError('Current deployment CPU proof required')
    for stamp in report['stamps']:
        if fingerprint(stamp['path'])!=stamp:raise ValueError('Asset changed; repeat CPU prepare before enabling GPU')
    if sha256(Path(root)/cfg['base_checkpoint'])!=BASE_SHA:
        raise ValueError('Original frozen base changed')
    if shutil.disk_usage(deploy).free<2*2**30:raise ValueError('2 GiB reserve required')
    return report


def run(root,deploy,cfg,run_id):
    """Manual one-shot run: CUDA proof -> at most 10 updates -> full-image review."""
    start=time.monotonic();root,deploy=Path(root),Path(deploy)
    report=ready(root,deploy,cfg)
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Neutral run id required')
    dest=deploy/'runs'/run_id
    if dest.exists():raise ValueError('Existing run protected; use a new id')
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable; zero optimizer updates, no CPU training fallback')
    def guard():
        if time.monotonic()-start>=cfg['maximum_seconds']:raise BudgetExpired('Pilot time budget exhausted')
    torch.set_num_threads(min(4,max(1,os.cpu_count() or 1)))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    dest.mkdir(parents=True,exist_ok=False)  # Claim this run before any model allocation.
    model=load_model(root/cfg['base_checkpoint'],device='cuda').train()
    guard();probe=torch.full((1,5,4,32,32),150.,device='cuda')
    sum(o.mean() for o in model(probe,[24],[3838.5],[4])).backward()
    if (any(p.grad is not None for p in model.backbone.parameters())
            or not all(h.weight.grad is not None and torch.isfinite(h.weight.grad).all()
                       and h.weight.grad.abs().sum()>0 for h in (model.to_residual,))):
        raise ValueError('CUDA gradient check failed before optimizer creation')
    model.zero_grad(set_to_none=True)
    write_json(dest/'cuda_check.json',dict(passed=True,device=torch.cuda.get_device_name(),optimizer_updates=0))
    optimizer=torch.optim.AdamW(model.adapter_parameters(),lr=cfg['learning_rate'],weight_decay=0.)
    manifest,streams,_,_=inventory(root);groups=report['groups'];rng=np.random.default_rng(cfg['seed'])
    domains=sorted({s['sensor'] for s in streams.values()});visited=[];updates=0
    def get_batch(n):
        jobs=[]
        domain=domains[n%len(domains)]
        keys=[k for k,s in streams.items() if s['sensor']==domain]
        for _ in range(cfg['batch_size']):
            key=str(rng.choice(keys));items=groups['train'][key]
            for trial in range(8):
                item=items[int(rng.integers(len(items)))]
                try:job,meta=sample(root,streams[key],item,rng)
                except ValueError as e:
                    if not str(e).startswith('No supervised crop'):raise
                    continue
                visited.append(meta);jobs.append(job);break
            else:raise ValueError('No supervised sample for '+key)
        return {k:torch.stack([j[k] for j in jobs]).pin_memory() for k in jobs[0]}
    batches=prefetch_map(get_batch,range(min(cfg['maximum_updates'],cfg['review_every'])*cfg['accumulate']))
    try:
        for step in range(1,min(cfg['maximum_updates'],cfg['review_every'])+1):
            guard();optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg['accumulate']):
                b={k:v.cuda(non_blocking=True) for k,v in next(batches).items()}
                guard();outputs=model(b['signal'],b['shot'],b['span'],b['read'],training=True)
                loss,metrics=final_mixed_loss(outputs,b['target'],b['teacher'],b['static'],b['motion'],b['span'])
                (loss/cfg['accumulate']).backward()
            norm=torch.nn.utils.clip_grad_norm_(list(model.adapter_parameters()),1.,error_if_nonfinite=True)
            guard();optimizer.step();updates=step
            write_json(dest/'status.json',dict(step=updates,loss=float(loss.detach()),metrics=metrics,
                grad_norm=float(norm),gpu_peak_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start))
            print(json.dumps(dict(step=step,loss=float(loss.detach()))),flush=True)
        guard()
        torch.save(dict(architecture=ARCHITECTURE,adapter=model.adapter_state(),base_sha256=BASE_SHA,
            signature=signature(),config=cfg,step=updates,optimizer=optimizer.state_dict(),quality_proven=False),dest/'candidate.pt')
        # Same held-out development partition, RAW window, tile size, RNG and ISP.
        key=next((k for k,s in streams.items() if s['sensor']=='MIS20S1'),sorted(streams)[0])
        item=groups['validation'][key][len(groups['validation'][key])//2];s=streams[key];frame=item['frame']
        observed=torch.from_numpy(signed_signal(read_packed(root,s,list(range(frame-2,frame+3))),s['black_dn12'])).cuda()
        span=torch.tensor(4095-np.asarray(s['black_dn12']),dtype=torch.float32,device='cuda')
        shot=[s['shot_gain_dn12_per_e']];read=[s['read_variance_dn12_sq']]
        from utils.rgb_metrics import packed_to_linear_rgb
        from PIL import Image
        model.eval();noisy=observed[2].cpu()
        rgb,wb=packed_to_linear_rgb(((noisy+16)/4095)[None],'GBRG',return_wb=True)
        scale=max(float(torch.quantile(rgb,.995)),1e-6)
        def render(value,name):
            image=packed_to_linear_rgb(((value.cpu()+16)/4095)[None],'GBRG',wb_gains=wb)
            pixels=np.clip(image[0].permute(1,2,0).numpy()/scale,0,1)
            Image.fromarray((pixels*255).astype('uint8')).save(dest/(name+'.png'))
        render(noisy,'noisy')
        for name,enabled in [('frozen_base',False),('candidate',True)]:
            def predict(patch):
                guard();torch.manual_seed(cfg['seed'])
                return model(patch[None],shot,span[None],read,enabled=enabled)[-1][0]*span[:,None,None]
            restored=infer_tiles(observed,predict,tile=192,stride=128)
            guard();render(restored,name)
        write_json(dest/'visual_review.json',dict(stream=key,frame=frame,role='validation',wb=wb,scale=scale,
            clean_gt=False,quality_proven=False,automatic_continuation=False))
        reason='whole_image_review_required'
    except BudgetExpired:
        reason='time_limit'
    finally:
        if updates and not (dest/'candidate.pt').exists():
            # Preserve paid-for updates even when the time budget expires.
            torch.save(dict(architecture=ARCHITECTURE,adapter=model.adapter_state(),base_sha256=BASE_SHA,
                signature=signature(),config=cfg,step=updates,optimizer=optimizer.state_dict(),
                quality_proven=False,incomplete_review=True),dest/'candidate.pt')
        if sha256(root/cfg['base_checkpoint'])!=BASE_SHA:raise ValueError('Frozen source checkpoint changed')
        write_json(dest/'result.json',dict(updates=updates,elapsed_seconds=time.monotonic()-start,
            stop_reason=locals().get('reason','failure'),visited=visited,automatic_retry=False,automatic_resume=False))
