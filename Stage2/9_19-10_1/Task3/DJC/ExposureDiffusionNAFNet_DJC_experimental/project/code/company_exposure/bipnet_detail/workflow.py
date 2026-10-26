"""Reuse audited independent-frame data protocol; train only the BIPNet branch."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from company_exposure.core import load_student, sha256, mixed_loss
from company_exposure.data import write_json
from company_exposure.workflow import CODE, code_signature, data_signature
from company_exposure.temporal import workflow as temporal
from company_exposure.temporal.data import sample, temporal_context
from .model import BIPNetDetail, VENDOR_SHA

def signature():
    return hashlib.sha256((code_signature()+temporal.signature()+VENDOR_SHA+
        ''.join(sha256(p) for p in sorted(Path(__file__).parent.glob('*.py')))).encode()).hexdigest()

def config(path):
    cfg=json.loads(Path(path).read_text())
    if (cfg['input_offsets']!=[-2,-1,0,1,2] or cfg['guide_offsets']!=[-5,-3,3,5,7,9]
        or cfg['target_offset']!=6 or cfg['num_features']!=64 or not cfg['freeze_backbone']
        or cfg['precision']!='float32' or cfg['automatic_training']):
        raise ValueError('Unsupported BIPNet protocol')
    for key in ('base_checkpoint','cache_directory','cpu_report'):
        p=Path(cfg[key])
        if p.is_absolute() or '..' in p.parts:raise ValueError('Path escapes project root')
    for key in ('batch_size','accumulate','optimizer_steps','validate_every','learning_rate','cache_gib'):
        if not np.isfinite(cfg[key]) or cfg[key]<=0:raise ValueError('Invalid '+key)
    if cfg['patch_packed']<16 or cfg['patch_packed']%16:raise ValueError('Invalid packed patch size')
    return cfg

def prepare(root,cfg):
    # This function only prepares independent masks, never instantiates either
    # temporal model or starts an optimizer. Original teacher support is checked.
    result=temporal.prepare(root,cfg)
    result['bipnet_signature']=signature()
    cache=Path(root)/cfg['cache_directory']
    write_json(cache/'index.json',result)
    (cache/'index.sha256').write_text(sha256(cache/'index.json'))
    return dict(complete=True,contexts=len(result['records']),coverage=result['coverage'])

def ready(root,cfg):
    root=Path(root);cache=root/cfg['cache_directory']
    if not (cache/'index.json').exists():raise ValueError('New independent five-frame masks missing; run prepare explicitly')
    report=json.loads((root/cfg['cpu_report']).read_text())
    if not report['passed'] or report['base_sha256']!=cfg['base_sha256'] or report.get('signature')!=signature():
        raise ValueError('Current company-base CPU check required')
    manifest,streams,teacher=temporal.inputs(root,cfg)
    if sha256(cache/'index.json')!=(cache/'index.sha256').read_text():raise ValueError('Mask index changed')
    index=json.loads((cache/'index.json').read_text())
    if (not index['complete'] or index.get('bipnet_signature')!=signature()
        or index['signature']!=temporal.signature() or index['data_signature']!=data_signature(manifest)
        or index['base_sha256']!=cfg['base_sha256']):raise ValueError('Stale mask cache')
    old={(i['stream'],i['role'],i['frame']):i for i in teacher['items']}
    groups={role:{key:[] for key in streams} for role in ('train','validation')}
    seen=set()
    for item in index['records']:
        key=(item['stream'],item['role'],item['frame'])
        if key in seen:raise ValueError('Duplicate temporal context')
        seen.add(key);parent=old[key]
        if item['teacher_path']!=parent['path'] or item['teacher_sha256']!=parent['sha256']:
            raise ValueError('Teacher reference changed')
        temporal_context(streams[item['stream']],item['role'],item['frame'])
        if sha256(cache/item['mask_path'])!=item['mask_sha256']:raise ValueError('Mask checksum mismatch')
        with np.load(cache/item['mask_path'],allow_pickle=False) as masks:
            if masks['static'].dtype!=bool or masks['motion'].dtype!=bool or (masks['static'] & masks['motion']).any():
                raise ValueError('Invalid supervision masks')
        groups[item['role']][item['stream']].append(item)
    if any(not items for g in groups.values() for items in g.values()):raise ValueError('All 40 stream partitions required')
    return cache,streams,groups

def one_sample(root,cache,stream,item,rng,size):
    job=sample(root,cache,stream,item,rng,size)
    job['read']=torch.tensor(stream['read_variance_dn12_sq'],dtype=torch.float32)
    return job

def batch(root,cache,streams,groups,rng,cfg):
    jobs=[];sensors=sorted({s['sensor'] for s in streams.values()})
    for _ in range(cfg['batch_size']):
        sensor=rng.choice(sensors);key=rng.choice([k for k,s in streams.items() if s['sensor']==sensor])
        items=groups[key];item=items[int(rng.integers(len(items)))]
        jobs.append(one_sample(root,cache,streams[key],item,rng,cfg['patch_packed']))
    return {k:torch.stack([j[k] for j in jobs]).cuda() for k in jobs[0]}

def loss(model,b):
    output,_=model(b['signal'],b['shot'],b['span'],b['read'])
    # Same region-normalized static1/motion0.5 objective, on the single final
    # corrected result. Repeated list preserves existing helper normalization.
    return mixed_loss([output]*3,b['target'],b['teacher'],b['static'],b['motion'],b['span'])[0]

@torch.no_grad()
def validate(model,root,cache,streams,groups,cfg):
    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all()
    torch.manual_seed(20261019);rng=np.random.default_rng(20261019);scores={};model.eval()
    try:
        for key,items in sorted(groups.items()):
            b={k:v[None].cuda() for k,v in one_sample(root,cache,streams[key],items[0],rng,cfg['patch_packed']).items()}
            scores.setdefault(streams[key]['sensor'],[]).append(float(loss(model,b)))
        return float(np.mean([np.mean(v) for v in scores.values()]))
    finally:
        torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(gpu);model.train()

def train(root,cfg,run_id):
    if not torch.cuda.is_available():raise ValueError('CUDA missing; no optimizer or training run created')
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Neutral run ID required')
    root=Path(root);cache,streams,groups=ready(root,cfg)
    dest=root/'runs'/run_id
    if dest.exists():raise ValueError('Refusing to overwrite an existing run')
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=BIPNetDetail(load_student(CODE,root/cfg['base_checkpoint'],'cuda',pretrained=False),
                      CODE/'vendor/BIPNet/Burst De-noising/Network.py').cuda().train()
    probe=torch.full((1,5,4,16,16),150.,device='cuda')
    model(probe,[24.],[[3838.5]*4],[[100.]*4])[0].square().mean().backward()
    grads=[p.grad for p in model.detail.parameters() if p.grad is not None]
    if not grads or not any(g.abs().sum()>0 for g in grads) or not all(torch.isfinite(g).all() for g in grads):
        raise ValueError('CUDA forward/backward probe failed before optimizer creation')
    model.zero_grad(set_to_none=True);dest.mkdir(parents=True)
    optimizer=torch.optim.AdamW(model.detail.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
    initial=validate(model,root,cache,streams,groups['validation'],cfg)
    write_json(dest/'initial_validation.json',dict(loss=initial,clean_psnr=None))
    best=float('inf')
    for step in range(1,cfg['optimizer_steps']+1):
        optimizer.zero_grad(set_to_none=True);value=0.
        for _ in range(cfg['accumulate']):
            term=loss(model,batch(root,cache,streams,groups['train'],rng,cfg))
            (term/cfg['accumulate']).backward();value+=float(term.detach())/cfg['accumulate']
        norm=torch.nn.utils.clip_grad_norm_(model.detail.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        write_json(dest/'status.json',dict(step=step,loss=value,grad_norm=float(norm),base_frozen=True,
                   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        if step==1 or step%cfg['validate_every']==0 or step==cfg['optimizer_steps']:
            score=validate(model,root,cache,streams,groups['validation'],cfg)
            state=dict(detail=model.detail.state_dict(),optimizer=optimizer.state_dict(),step=step,
                       config=cfg,signature=signature(),base_sha256=cfg['base_sha256'],validation_loss=score)
            torch.save(state,dest/f'step_{step:04d}.pt')
            if score<best:best=score;torch.save(state,dest/'best.pt')
            write_json(dest/'validation.json',dict(step=step,loss=score,best=best,clean_psnr=None))
        print(json.dumps(dict(step=step,loss=value)),flush=True)
    if sha256(root/cfg['base_checkpoint'])!=cfg['base_sha256']:raise ValueError('Frozen checkpoint file changed')
