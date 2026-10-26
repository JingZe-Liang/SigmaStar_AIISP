"""Portable V3 runner. Explicit CUDA train; preparation never starts training."""
from pathlib import Path
import argparse,json,sys,time,shutil,random
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from company_exposure.burst_detail import pilot
from company_exposure.burst_detail.model import ARCHITECTURE,BASE_SHA,load_model
from company_exposure.burst_detail.workflow import signature,final_mixed_loss
from company_exposure.core import sha256,signed_signal
from company_exposure.data import write_json,read_packed
from company_exposure.workflow import prefetch_map
from company_adaptation_v2 import infer_tiles


def random_state():
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],python=random.getstate())


def restore_random(state):
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
    n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    random.setstate(state['python'])


def save_checkpoint(path,model,optimizer,step,best,contract):
    path=Path(path)
    if shutil.disk_usage(path.parent).free<2*2**30:raise ValueError('Need 2 GiB disk reserve before saving')
    temporary=path.with_suffix('.part')
    torch.save(dict(architecture=ARCHITECTURE,signature=signature(),base_sha256=BASE_SHA,
        adapter=model.adapter_state(),optimizer=optimizer.state_dict(),step=step,best=best,
        contract=contract,random=random_state(),quality_proven=False),temporary)
    temporary.replace(path)


def fixed_validation(root,streams,groups,seed):
    samples=[]
    for key,items in sorted(groups['validation'].items()):
        for i in np.linspace(0,len(items)-1,min(4,len(items)),dtype=int):
            try:b,meta=pilot.sample(root,streams[key],items[i],np.random.default_rng(seed+int(i)))
            except ValueError as e:
                if not str(e).startswith('No supervised crop'):raise
                continue
            samples.append((b,meta));break
        else:raise ValueError('No fixed validation crop for '+key)
    return samples


def validate(model,samples,streams,device,center_only=False):
    state=random_state();was_training=model.training;domains={};rows=[]
    try:
        model.eval()
        with torch.no_grad():
            for i,(batch,meta) in enumerate(samples):
                b={k:v[None].to(device) for k,v in batch.items()};values={}
                for name,enabled in [('base',False),('candidate',True)]:
                    signal=b['signal']
                    if center_only and enabled:signal=signal[:,2:3].expand_as(signal)
                    torch.manual_seed(5300+i);out=model(signal,b['shot'],b['span'],b['read'],enabled=enabled)
                    loss,parts=final_mixed_loss(out,b['target'],b['teacher'],b['static'],b['motion'],b['span'])
                    error=out[-1]*b['span'][:,:,None,None]-b['target'];edge_sum=0.;count=0
                    for axis in [-1,-2]:
                        mask=b['static']
                        sites=(mask[...,1:]&mask[...,:-1]) if axis==-1 else (mask[...,1:,:]&mask[...,:-1,:])
                        delta=torch.diff(error,dim=axis).square();valid=sites.expand_as(delta)
                        edge_sum+=float(delta[valid].sum());count+=int(valid.sum())
                    values[name]=dict(loss=float(loss),static_edge_mse_dn12=edge_sum/count if count else None,**parts)
                sensor=streams[meta['stream']]['sensor'];domains.setdefault(sensor,[]).append(values)
                rows.append(dict(meta,metrics=values))
        score=float(np.mean([np.mean([v['candidate']['loss'] for v in values]) for values in domains.values()]))
        return dict(equal_domain_proxy_loss=score,rows=rows,clean_gt=False,quality_proven=False)
    finally:model.train(was_training);restore_random(state)


def preview(model,root,streams,groups,dest,device,seed,alpha=1.,center_only=False):
    from utils.rgb_metrics import packed_to_linear_rgb
    from PIL import Image
    state=random_state();model.eval()
    try:
        key=next((k for k,s in sorted(streams.items()) if s['sensor']=='MIS20S1'),sorted(streams)[0])
        items=groups['validation'][key];item=items[len(items)//2];s=streams[key];t=item['frame']
        observed=torch.from_numpy(signed_signal(read_packed(root,s,list(range(t-2,t+3))),s['black_dn12'])).to(device)
        span=torch.tensor(4095-np.asarray(s['black_dn12']),dtype=torch.float32,device=device)
        shot=[s['shot_gain_dn12_per_e']];read=[s['read_variance_dn12_sq']]
        rgb,wb=packed_to_linear_rgb((observed[2:3].cpu()+16)/4095,'GBRG',return_wb=True)
        scale=max(float(torch.quantile(rgb,.995)),1e-6)
        dest.mkdir(parents=True,exist_ok=False)
        def render(v,name):
            rgb=packed_to_linear_rgb((v[None].cpu()+16)/4095,'GBRG',wb_gains=wb)
            pixels=(rgb[0].permute(1,2,0).numpy()/scale).clip(0,1)
            Image.fromarray((pixels*255).astype('uint8')).save(dest/(name+'.png'))
        render(observed[2],'noisy')
        for name,enabled in [('frozen_base',False),('residual_v3',True)]:
            def predict(patch):
                if center_only and enabled:patch=patch[2:3].expand_as(patch)
                torch.manual_seed(seed)
                return model(patch[None],shot,span[None],read,enabled=enabled,residual_scale=alpha)[-1][0]*span[:,None,None]
            render(infer_tiles(observed,predict,tile=192,stride=128),name)
        write_json(dest/'provenance.json',dict(stream=key,frame=t,role='validation',alpha=alpha,wb=wb,scale=scale,clean_gt=False))
    finally:restore_random(state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--work',required=True,type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'configs/exposure_paper_burst_server_20261022.json')
    sub=p.add_subparsers(dest='action',required=True);sub.add_parser('prepare');sub.add_parser('check-ready')
    q=sub.add_parser('train');q.add_argument('--run-id',required=True);q.add_argument('--updates',type=int,default=1000)
    q.add_argument('--validate-every',type=int,default=100);q.add_argument('--batch',type=int,default=2)
    q.add_argument('--accumulate',type=int,default=4);q.add_argument('--lr',type=float,default=1e-4)
    q.add_argument('--resume',type=Path);q.add_argument('--center-only',action='store_true')
    q=sub.add_parser('evaluate');q.add_argument('--checkpoint',required=True,type=Path)
    q.add_argument('--output',required=True,type=Path);q.add_argument('--alpha',type=float,default=1.)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();cfg=pilot.settings(a.config)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.action=='prepare':
        work.mkdir(parents=True,exist_ok=True);pilot.prepare(root,work,cfg);return
    report=pilot.ready(root,work,cfg)
    if a.action=='check-ready':print('CPU ready; CUDA and real GPU backward not yet checked');return
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable; no optimizer created or CPU training fallback')
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);device=torch.device('cuda:0')
    manifest,streams,_,groups=pilot.inventory(root);model=load_model(root/cfg['base_checkpoint'],device=device)
    if a.action=='evaluate':
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        if saved['signature']!=signature() or saved['base_sha256']!=BASE_SHA or saved['architecture']!=ARCHITECTURE:raise ValueError('Checkpoint/source mismatch')
        model.load_adapter(saved['adapter']);preview(model,root,streams,groups,a.output,device,cfg['seed'],a.alpha,saved['contract']['center_only']);return
    if min(a.updates,a.validate_every,a.batch,a.accumulate)<=0 or not 0<a.lr<=1e-3:raise ValueError('Invalid optimization budget')
    if not a.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.run_id):raise ValueError('Invalid run id')
    dest=work/'runs'/a.run_id;start_step=0;best=float('inf');model.train()
    contract=dict(runner_sha256=sha256(__file__),base_sha256=BASE_SHA,manifest_sha256=sha256(root/'manifest.json'),teacher_index_sha256=sha256(root/'teacher/index.json'),
                  batch=a.batch,accumulate=a.accumulate,lr=a.lr,seed=cfg['seed'],center_only=a.center_only,validation_seed=20261023)
    if a.resume:
        if not dest.is_dir() or a.resume.resolve().parent!=dest:raise ValueError('Resume must be a checkpoint in the same run directory')
        saved=torch.load(a.resume,map_location='cpu',weights_only=True)
        if saved['contract']!=contract or saved['signature']!=signature() or saved['architecture']!=ARCHITECTURE:raise ValueError('Resume provenance mismatch')
        model.load_adapter(saved['adapter']);start_step=saved['step'];best=saved['best']
        if start_step>=a.updates:raise ValueError('Requested updates already reached; use evaluate or explicitly raise total budget')
    elif dest.exists():raise ValueError('Existing run protected; use --resume or a new id')
    # Actual torchvision CUDA deformable-convolution backward before any optimizer.
    probe=torch.full((1,5,4,32,32),150.,device=device)
    state=random_state();model(probe,[24],[3838.5],[4])[-1].square().mean().backward()
    if model.to_residual.weight.grad is None or not torch.isfinite(model.to_residual.weight.grad).all():raise ValueError('CUDA residual backward failed')
    assert not any(x.grad is not None for x in model.backbone.parameters());model.zero_grad(set_to_none=True);restore_random(state)
    dest.mkdir(parents=True,exist_ok=bool(a.resume));write_json(dest/'contract.json',contract)
    optimizer=torch.optim.AdamW(model.adapter_parameters(),lr=a.lr,weight_decay=0.)
    if a.resume:optimizer.load_state_dict(saved['optimizer']);restore_random(saved['random'])
    samples=fixed_validation(root,streams,groups,20261023);write_json(dest/'fixed_validation.json',[m for _,m in samples])
    monitor_by_domain={}
    for sample in fixed_validation(root,streams,{'validation':groups['train']},20261024):
        monitor_by_domain.setdefault(streams[sample[1]['stream']]['sensor'],sample)
    monitor=list(monitor_by_domain.values());write_json(dest/'fixed_train_monitor.json',[m for _,m in monitor])
    if not a.resume:
        write_json(dest/'baseline_validation.json',validate(model,samples,streams,device,a.center_only))
        write_json(dest/'baseline_train_monitor.json',validate(model,monitor,streams,device,a.center_only))
    domains=sorted({s['sensor'] for s in streams.values()});began=time.monotonic();step=start_step
    def batch_for(job):
        update,part=job;rng=np.random.default_rng(cfg['seed']+update*a.accumulate+part)
        domain=domains[((update-1)*a.accumulate+part)%len(domains)];keys=sorted(k for k,s in streams.items() if s['sensor']==domain);batch=[]
        for _ in range(a.batch):
            key=str(rng.choice(keys));items=groups['train'][key]
            for attempt in range(8):
                try:b,_=pilot.sample(root,streams[key],items[int(rng.integers(len(items)))],rng)
                except ValueError as e:
                    if not str(e).startswith('No supervised crop'):raise
                    continue
                batch.append(b);break
            else:raise ValueError('No supervised training crop: '+key)
        return {k:torch.stack([b[k] for b in batch]).pin_memory() for k in batch[0]}
    jobs=((n,k) for n in range(start_step+1,a.updates+1) for k in range(a.accumulate))
    batches=prefetch_map(batch_for,jobs)
    try:
        for n in range(start_step+1,a.updates+1):
            if shutil.disk_usage(dest).free<2*2**30:raise ValueError('Disk reserve below 2 GiB')
            optimizer.zero_grad(set_to_none=True);total=0.
            for part in range(a.accumulate):
                b={k:v.to(device,non_blocking=True) for k,v in next(batches).items()}
                signal=b['signal']
                if a.center_only:signal=signal[:,2:3].expand_as(signal)
                out=model(signal,b['shot'],b['span'],b['read'],training=True)
                loss,_=final_mixed_loss(out,b['target'],b['teacher'],b['static'],b['motion'],b['span'])
                (loss/a.accumulate).backward();total+=float(loss.detach())/a.accumulate
            norm=torch.nn.utils.clip_grad_norm_(list(model.adapter_parameters()),1.,error_if_nonfinite=True)
            assert not any(x.grad is not None for x in model.backbone.parameters())
            optimizer.step();step=n
            row=dict(step=step,train_loss=total,gradient_norm=float(norm),elapsed_seconds=time.monotonic()-began,gpu_peak_bytes=torch.cuda.max_memory_allocated())
            if step%a.validate_every==0 or step==a.updates:
                val=validate(model,samples,streams,device,a.center_only);write_json(dest/f'validation_{step:04d}.json',val)
                tracked=validate(model,monitor,streams,device,a.center_only);write_json(dest/f'train_monitor_{step:04d}.json',tracked)
                row['fixed_training_proxy_loss']=tracked['equal_domain_proxy_loss']
                score=val['equal_domain_proxy_loss']
                if score<best:best=score;save_checkpoint(dest/'best.pt',model,optimizer,step,best,contract)
                save_checkpoint(dest/'last.pt',model,optimizer,step,best,contract);row['validation_proxy_loss']=score
            with (dest/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write_json(dest/'status.json',row);print(json.dumps(row),flush=True)
    finally:
        batches.close()
        if step>start_step:save_checkpoint(dest/'last.pt',model,optimizer,step,best,contract)
        if sha256(root/cfg['base_checkpoint'])!=BASE_SHA:raise ValueError('Frozen checkpoint changed')
    save_checkpoint(dest/'final.pt',model,optimizer,step,best,contract)
    write_json(dest/'complete.json',dict(updates=step,quality_proven=False,base_sha256=BASE_SHA))
    # Whole development frame, never the held-out test sequence.
    preview(model,root,streams,groups,dest/'final_preview',device,cfg['seed'],center_only=a.center_only)


if __name__=='__main__':main()
