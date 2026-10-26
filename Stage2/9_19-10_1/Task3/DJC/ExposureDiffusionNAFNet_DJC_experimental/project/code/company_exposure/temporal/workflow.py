"""Explicit CPU check, preparation, adapter-only training and ablation inference."""
import copy
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from company_exposure.core import load_student, sha256, mixed_loss, signed_signal, TEACHER_SHA
from company_exposure.data import read_packed, write_json, centers
from company_exposure.workflow import CODE, code_signature, data_signature, parameter_errors, prefetch_map
from .model import TemporalDetailNAF
from .data import temporal_context, independent_masks, sample


def signature():
    files = sorted(Path(__file__).parent.glob('*.py'))
    return hashlib.sha256((code_signature()+''.join(sha256(p) for p in files)).encode()).hexdigest()


def config(path):
    return config_dict(json.loads(Path(path).read_text()))


def inputs(root, cfg):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    counts = {role: sum(s['role'] == role for s in manifest['streams']) for role in ('development','calibration','test')}
    if counts != dict(development=20, calibration=18, test=7):
        raise ValueError('Full collection roles/counts changed')
    heldout = {s['sha256'] for s in manifest['streams'] if s['role'] == 'test'}
    streams = {s['id']: s for s in manifest['streams'] if s['role'] == 'development'}
    if heldout.intersection(s['sha256'] for s in streams.values()):
        raise ValueError('Held-out duplicate in development')
    for s in manifest['streams']:
        if '/shdarkroom/' in s['source_relative'] and s['role'] != 'test':
            raise ValueError('Indoor holdout was reassigned')
        if s['role'] == 'test':
            continue
        p = root/s['path']
        if not p.is_file() or p.stat().st_size != s['bytes'] or sha256(p) != s['sha256']:
            raise ValueError('Missing/changed RAW: '+s['id'])
        if s['role'] == 'development' and parameter_errors(s):
            raise ValueError(s['id']+': '+str(parameter_errors(s)))
    if sha256(root/cfg['base_checkpoint']) != cfg['base_sha256']:
        raise ValueError('Frozen base checkpoint hash mismatch')
    index = json.loads((root/'teacher/index.json').read_text())
    if (index['signature'] != data_signature(manifest) or index['code_signature'] != code_signature()
            or index['teacher_sha256'] != TEACHER_SHA):
        raise ValueError('Original sparse teacher cache is stale')
    expected = {(s['id'], role, t) for s in streams.values() for role in ('train','validation')
                for t in centers(s, role)}
    actual = {(i['stream'], i['role'], i['frame']) for i in index['items']}
    if actual != expected or len(actual) != len(index['items']):
        raise ValueError('Original teacher collection is incomplete/duplicated')
    for item in index['items']:
        if sha256(root/item['path']) != item['sha256']:
            raise ValueError('Teacher file changed: '+item['path'])
    return manifest, streams, index


def prepare(root, cfg):
    root = Path(root); manifest, streams, old = inputs(root, cfg)
    cache = root/cfg['cache_directory']; cache.mkdir(parents=True, exist_ok=False)
    records = []; coverage = {}; used = 0
    for item in old['items']:
        s = streams[item['stream']]
        try:
            _, guides, _ = temporal_context(s, item['role'], item['frame'])
        except ValueError:
            continue  # Explicitly tighter left boundary; every stream is checked below.
        observed = read_packed(root, s, guides)
        static, moving, detail, stats = independent_masks(observed, s)
        label = torch.load(root/item['path'], map_location='cpu', weights_only=True)
        if label['source_sha256'] != s['sha256'] or label['teacher_sha256'] != TEACHER_SHA:
            raise ValueError('Teacher payload provenance mismatch')
        old_motion = label['motion'].numpy()
        motion = moving & old_motion
        # Do not invent teacher values outside the old sparse support. Static
        # masks never depend on old labels, whose guides overlap the new inputs.
        if not (static | motion).any():
            continue
        stats.update(motion_fraction=float(motion.mean()),
                     teacher_support_retained=float(motion.sum()/max(int(moving.sum()),1)),
                     original_teacher_pixels=int(old_motion.sum()),guidance_frames=guides)
        path = Path('masks')/s['id']/item['role']/f"{item['frame']:04d}.npz"
        dest = cache/path; dest.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(cache).free < 2*1024**3 or used+static.nbytes*3 > cfg['cache_gib']*1024**3:
            raise ValueError('Temporal mask cache budget/free space exhausted')
        np.savez_compressed(dest, static=static, motion=motion, detail=detail)
        used += dest.stat().st_size
        records.append(dict(stream=s['id'],role=item['role'],frame=item['frame'],
                            mask_path=str(path),mask_sha256=sha256(dest),
                            teacher_path=item['path'],teacher_sha256=item['sha256'],stats=stats))
        key=s['id']+'/'+item['role']; coverage[key]=coverage.get(key,0)+1
        print(json.dumps(dict(prepared=len(records),stream=s['id'],role=item['role'])),flush=True)
    if len(coverage) != 40:
        raise ValueError('Every one of 20 streams needs nonempty train and validation supervision')
    result=dict(records=records,coverage=coverage,bytes=used,signature=signature(),
                data_signature=data_signature(manifest),base_sha256=cfg['base_sha256'],
                config=cfg,complete=True,original_contexts=len(old['items']),
                note='New independent static masks; dynamic support is intersection with existing sparse teacher. No new teacher inference.')
    write_json(cache/'index.json', result)
    (cache/'index.sha256').write_text(sha256(cache/'index.json'))
    return result


def cpu_check(checkpoint, expected_sha, output):
    torch.set_num_threads(2)
    if sha256(checkpoint) != expected_sha:
        raise ValueError('CPU check checkpoint hash mismatch')
    torch.manual_seed(20261019)
    base=load_student(CODE,checkpoint,pretrained=False)
    model=TemporalDetailNAF(base)
    before={k:v.clone() for k,v in base.state_dict().items()}
    signal=torch.randn(1,5,4,32,48)*8+150
    # Compare inference to inference: enabling gradients in only the adapted
    # path may select different CPU convolution kernels. Tiny roundoff can
    # then change a subsequent Poisson draw; it is not an inference mismatch.
    with torch.no_grad():
        torch.manual_seed(31); baseline=model(signal,[24.],[3838.5],enabled=False)
        torch.manual_seed(31); outputs=model(signal,[24.],[3838.5])
    assert all(torch.equal(a,b) for a,b in zip(baseline,outputs))
    torch.manual_seed(31);outputs=model(signal,[24.],[3838.5])
    loss=sum(x.square().mean() for x in outputs);loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.adapter_parameters())
    assert all(p.grad is None for p in base.parameters())
    assert all(torch.equal(before[k],v) for k,v in base.state_dict().items())
    assert not base.intro._forward_hooks and not base.ending._forward_pre_hooks
    result=dict(passed=True,base_sha256=expected_sha,signature=signature(),zero_branch_bitwise_equal=True,
                three_step_gradient=True,backbone_unchanged=True,optimizer_updates=0,
                backbone_parameters=sum(p.numel() for p in base.parameters()),
                trainable_parameters=sum(p.numel() for p in model.adapter_parameters()),
                cuda_checked=False,real_quality_validated=False)
    write_json(output,result)
    return result


def ready(root, cfg):
    root=Path(root);manifest,streams,original=inputs(root,cfg)
    cache=root/cfg['cache_directory']
    if sha256(cache/'index.json')!=(cache/'index.sha256').read_text():
        raise ValueError('Temporal cache index checksum mismatch')
    index=json.loads((cache/'index.json').read_text())
    structural=('version','base_checkpoint','base_sha256','freeze_backbone','width',
                'input_offsets','guide_offsets','target_offset','precision')
    if (not index['complete'] or index['signature']!=signature()
            or any(index['config'][k]!=cfg[k] for k in structural)
            or index['data_signature']!=data_signature(manifest) or index['base_sha256']!=cfg['base_sha256']):
        raise ValueError('Temporal cache/config/source signature changed')
    old={(i['stream'],i['role'],i['frame']):i for i in original['items']}
    for item in index['records']:
        parent=old[(item['stream'],item['role'],item['frame'])]
        if item['teacher_path']!=parent['path'] or item['teacher_sha256']!=parent['sha256']:
            raise ValueError('Temporal teacher reference changed')
        temporal_context(streams[item['stream']],item['role'],item['frame'])
        if sha256(cache/item['mask_path'])!=item['mask_sha256']:
            raise ValueError('Temporal mask checksum mismatch')
    check=json.loads((cache/'cpu_check.json').read_text())
    if not check['passed'] or check['signature']!=signature() or check['base_sha256']!=cfg['base_sha256']:
        raise ValueError('Current base-specific CPU check required')
    groups={role:{key:[] for key in streams} for role in ('train','validation')}
    for item in index['records']:groups[item['role']][item['stream']].append(item)
    if any(not group for split in groups.values() for group in split.values()):
        raise ValueError('Full stream coverage lost')
    return cache,streams,index,groups


def batch(root,cache,streams,groups,rng,cfg):
    jobs=[];sensors=sorted({s['sensor'] for s in streams.values()})
    for _ in range(cfg['batch_size']):
        sensor=rng.choice(sensors);key=rng.choice([k for k,s in streams.items() if s['sensor']==sensor])
        item=groups[key][int(rng.integers(len(groups[key])))]
        jobs.append(sample(root,cache,streams[key],item,rng,cfg['patch_packed']))
    return {key:torch.stack([j[key] for j in jobs]).pin_memory() for key in jobs[0]}


@torch.no_grad()
def validate(model,root,cache,streams,groups,cfg):
    cpu,gpu,numpy=torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state()
    torch.manual_seed(20261019);np.random.seed(20261019);rng=np.random.default_rng(20261019)
    scores={};model.eval()
    try:
        for key in sorted(groups):
            values=[]
            for item in (groups[key][0],groups[key][-1]):
                b={k:v[None].cuda() for k,v in sample(root,cache,streams[key],item,rng,cfg['patch_packed']).items()}
                outputs=model(b['signal'],b['shot'],b['span'])
                loss,_=mixed_loss(outputs,b['target'],b['teacher'],b['static'],b['motion'],b['span'])
                values.append(float(loss))
            scores.setdefault(streams[key]['sensor'],[]).append(float(np.mean(values)))
        return float(np.mean([np.mean(v) for v in scores.values()]))
    finally:
        torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(gpu);np.random.set_state(numpy);model.train()


def train(root,cfg,run_id):
    if not torch.cuda.is_available():
        raise ValueError('CUDA required; refusing before optimizer creation')
    root=Path(root);cache,streams,index,groups=ready(root,cfg)
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Neutral run id required')
    dest=root/'runs'/run_id
    if dest.exists():raise ValueError('Refusing existing run overwrite')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    model=TemporalDetailNAF(load_student(CODE,root/cfg['base_checkpoint'],'cuda',pretrained=False),cfg['width']).train().cuda()
    # Actual device/gradient gate with no optimizer update.
    probe=torch.full((1,5,4,32,32),150.,device='cuda')
    sum(x.mean() for x in model(probe,[24.],[3838.5])).backward()
    grads=[p.grad for p in model.adapter_parameters() if p.grad is not None]
    if not grads or not any(g.abs().sum()>0 for g in grads) or not all(torch.isfinite(g).all() for g in grads):
        raise ValueError('CUDA gradient probe failed')
    model.zero_grad(set_to_none=True);dest.mkdir(parents=True)
    optimizer=torch.optim.AdamW(model.adapter_parameters(),lr=cfg['learning_rate'],weight_decay=0.)
    best=validate(model,root,cache,streams,groups['validation'],cfg)
    write_json(dest/'initial_validation.json',dict(score=best,clean_psnr=None))
    best=float('inf')  # Select among trained checkpoints, retaining initialization score separately.
    batches=prefetch_map(lambda _:batch(root,cache,streams,groups['train'],rng,cfg),
                         range(cfg['optimizer_steps']*cfg['accumulate']))
    for step in range(1,cfg['optimizer_steps']+1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg['accumulate']):
            b={k:v.cuda(non_blocking=True) for k,v in next(batches).items()}
            outputs=model(b['signal'],b['shot'],b['span'],training=True)
            loss,metrics=mixed_loss(outputs,b['target'],b['teacher'],b['static'],b['motion'],b['span'])
            (loss/cfg['accumulate']).backward()
        norm=torch.nn.utils.clip_grad_norm_(list(model.adapter_parameters()),1.,error_if_nonfinite=True)
        optimizer.step()
        write_json(dest/'status.json',dict(optimizer_updates=step,metrics=metrics,grad_norm=float(norm),
                   backbone_frozen=True,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        if step==1 or step%cfg['validate_every']==0 or step==cfg['optimizer_steps']:
            score=validate(model,root,cache,streams,groups['validation'],cfg)
            saved=dict(adapter=model.adapter_state(),config=cfg,base_sha256=cfg['base_sha256'],
                       signature=signature(),step=step,validation_loss=score,optimizer=optimizer.state_dict())
            torch.save(saved,dest/f'step_{step:04d}.pt')
            if score<best:best=score;torch.save(saved,dest/'best.pt')
            write_json(dest/'validation.json',dict(step=step,score=score,best=best,clean_psnr=None))
        print(json.dumps(dict(step=step,metrics=metrics)),flush=True)
    assert sha256(root/cfg['base_checkpoint'])==cfg['base_sha256']


@torch.inference_mode()
def infer(root,checkpoint,stream_id,frames,output,device='cpu',allow_heldout=False,source=None,calibration=None):
    from PIL import Image,ImageDraw
    from utils.rgb_metrics import packed_to_linear_rgb
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(root);output=Path(output);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=saved['config'];cfg=config_dict(cfg)
    if saved['signature']!=signature() or saved['base_sha256']!=cfg['base_sha256']:
        raise ValueError('Adapter architecture/source signature mismatch')
    if sha256(root/cfg['base_checkpoint'])!=cfg['base_sha256']:raise ValueError('Base checkpoint mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    s=copy.deepcopy(next(s for s in manifest['streams'] if s['id']==stream_id))
    if s['role']=='test' and not allow_heldout:raise ValueError('Explicit heldout evaluation flag required')
    if calibration:
        c=json.loads(Path(calibration).read_text())
        if c['stream']!=stream_id or c['source_sha256']!=s['sha256'] or not c.get('parameter_source'):
            raise ValueError('Independent calibration provenance/stream mismatch')
        for key in ('black_dn12','shot_gain_dn12_per_e'):s[key]=c[key]
    if parameter_errors(s):
        # Read variance is not used by inference; all other fields remain mandatory.
        errors=[e for e in parameter_errors(s) if not e.startswith('read_variance')]
        if errors:raise ValueError(str(errors))
    if source:s['path']=str(Path(source).resolve())
    if sha256(root/s['path'])!=s['sha256']:raise ValueError('Inference RAW hash mismatch')
    if any(t<2 or t+2>=s['frame_count'] for t in frames):raise ValueError('Inference window out of range')
    output.mkdir(parents=True,exist_ok=False)
    model=TemporalDetailNAF(load_student(CODE,root/cfg['base_checkpoint'],device,pretrained=False),cfg['width']).to(device).eval()
    model.load_adapter(saved['adapter']);records=[];wb=None;scale=None;span=4095-np.asarray(s['black_dn12'],np.float32)
    for t in frames:
        obs=read_packed(root,s,list(range(t-2,t+3)));sig=signed_signal(obs,s['black_dn12'])
        signal=torch.from_numpy(sig)[None].to(device);shot=[s['shot_gain_dn12_per_e']];sp=[span.tolist()]
        results={}
        for name,value,enabled in [('SingleFrame',signal,False),('Temporal',signal,True),
                                   ('RepeatedCenter',signal[:,2:3].expand(-1,5,-1,-1,-1),True)]:
            torch.manual_seed(20261019+t)
            results[name]=model(value,shot,sp,enabled=enabled)[-1].cpu()*torch.tensor(span)[None,:,None,None]
        values={'Noisy':torch.from_numpy(sig[2:3]),**results};images=[]
        if wb is None:
            rgb,wb=packed_to_linear_rgb((values['Noisy']+16)/4095,'GBRG',return_wb=True);scale=max(float(np.quantile(rgb.numpy(),.995)),1e-6)
        for name,value in values.items():
            rgb=packed_to_linear_rgb((value+16)/4095,'GBRG',wb_gains=wb)
            image=Image.fromarray((np.clip(rgb[0].permute(1,2,0).numpy()/scale,0,1)*255).astype('uint8'))
            image.save(output/f'{t:04d}_{name}.png');images.append((name,image))
        w,h=images[0][1].size;canvas=Image.new('RGB',(2*w,2*(h+32)))
        for j,(name,image) in enumerate(images):
            x,y=j%2*w,j//2*(h+32);canvas.paste(image,(x,y+32));ImageDraw.Draw(canvas).text((x+8,y+8),name,fill='white')
        canvas.save(output/f'{t:04d}_comparison.png')
        torch.save(results,output/f'{t:04d}_signals_dn12.pt')
        records.append(dict(frame=t,inputs=list(range(t-2,t+3))))
    write_json(output/'report.json',dict(records=records,checkpoint_sha256=sha256(checkpoint),
        base_sha256=cfg['base_sha256'],wb=wb,scale=scale,clean_psnr=None,training=False,
        stream=stream_id,source_sha256=s['sha256'],black_dn12=s['black_dn12'],
        shot_gain_dn12_per_e=s['shot_gain_dn12_per_e'],
        calibration_sha256=sha256(calibration) if calibration else None,
        note='SingleFrame / real five frames / repeated center ablation; fixed display; no test fitting'))


def config_dict(cfg):
    if cfg.get('version')!=1 or not cfg.get('freeze_backbone') or cfg.get('precision')!='float32':
        raise ValueError('Unsupported adapter configuration')
    if cfg.get('input_offsets')!=[-2,-1,0,1,2] or cfg.get('guide_offsets')!=[-5,-3,3,5,7,9] or cfg.get('target_offset')!=6:
        raise ValueError('This version requires the independent five-frame protocol')
    if cfg.get('width')!=24 or any(cfg[k]<=0 for k in ('learning_rate','batch_size','accumulate','optimizer_steps','validate_every','cache_gib')):
        raise ValueError('Invalid pilot width/optimization settings')
    if cfg['patch_packed']<16 or cfg['patch_packed']%16:
        raise ValueError('Packed crop must be a positive multiple of 16')
    if Path(cfg['cache_directory']).is_absolute() or '..' in Path(cfg['cache_directory']).parts:
        raise ValueError('Mask cache must stay under the explicit project root')
    return cfg
