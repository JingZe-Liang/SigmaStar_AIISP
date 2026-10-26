"""Cost-bounded adaptation primitives; old experiments and signatures stay intact.

CLI defaults to metadata only. Explicit --execute requires reviewed noise
profiles, center-time target review and an explicit compute budget.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import sys

import numpy as np
import torch
from torch.nn import functional as F

from company_exposure.core import signed_signal
from company_exposure.data import read_packed, write_json

OFFICIAL_TAP_SHA = '21a8ed3ca6dd8d760fce9da52b797c5c3d02dd805ce0894cecdadc249c0942a6'
TILE = 192
TAP_WRAPPER_SHA = 'c56295b8a1a1bde21bb7b44c68899c86a394201d5d427ebe9ce0cb28b8f2837a'
TAP_SOURCE_HASHES = dict(
    module_dcn='411ac5074d25d8f41aebf95e9abe1f2eaa202a5436a3e25ec841499a59ad18af',
    module_pcd_alignment='53196efb4ff9f4cc2c5b21afe416c09ec16844c5a169ad8cd2d695aa3d91a7fe',
    network_nafnet='97bed3d4a97e2c3341e248fb678b37b690f5632ca90ef4cc7a0f5c11f2912e74')


class BudgetExpired(RuntimeError):
    pass


def prepare_protocol(manifest):
    streams = [s for s in manifest['streams'] if s['role'] == 'development']
    heldout = {s['sha256'] for s in manifest['streams'] if s['role'] == 'test'}
    if not streams or any('/shdarkroom/' in s['source_relative'] or s['sha256'] in heldout for s in streams):
        raise ValueError('Held-out source assigned to development')
    if len({s['sha256'] for s in streams}) != len(streams):
        raise ValueError('Duplicate development source')
    domains = sorted({s['sensor'] for s in streams})
    reserved = []
    for domain in domains:
        group = sorted([s for s in streams if s['sensor'] == domain], key=lambda s: s['id'])
        if len(group) < 2:
            raise ValueError('Sequence validation requires two streams per domain: '+domain)
        reserved.append(group[-1]['id'])
    records = []
    for s in streams:
        role = 'validation' if s['id'] in reserved else 'train'
        lo, hi = (0, s['frame_count']-1) if role == 'validation' else s['splits']['train']
        centers = [t for t in range(lo+4, hi-3) if t % 4 == 0]
        if not centers:
            raise ValueError('No complete nine-frame contexts: '+s['id'])
        records.append(dict(stream=s['id'], sensor=s['sensor'], role=role, centers=centers,
                            source_sha256=s['sha256'], raw_spec=s['raw_spec']))
    return dict(version=2, initialization_sha256=OFFICIAL_TAP_SHA,
        manifest_content_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        target_protocol='frozen_center_pseudo_independent_recorruption',
        teacher_window_offsets=list(range(-4,5)), student_offsets=list(range(-2,3)),
        target_offset=0, train_tile_packed=TILE, validation_tile_packed=TILE,
        inference_tile_packed=TILE, detail_weight=1., records=records,
        noise_profiles={s['id']:dict(verified=False, reason='Independent calibration reconciliation required') for s in streams},
        pseudo_target_review=dict(accepted=False,teacher_sha256=OFFICIAL_TAP_SHA,
                                  reviewed_streams=[],evidence_file=None,evidence_sha256=None),
        validation_scope='New candidate sequence split; previously viewed development scenes, not blind',
        maximum_seconds=None, maximum_updates=None, training_ready=False,
        blockers=['Noise profiles not reconciled', 'Pseudo-target visual quality not reviewed',
                  'No explicit compute budget'], automatic_training=False)


def check_readiness(protocol, checkpoint):
    """Read-only CPU checks. Flags alone cannot certify a checkpoint or noise fit."""
    blockers=[]
    if protocol.get('initialization_sha256')!=OFFICIAL_TAP_SHA:
        blockers.append('Original official initialization required for the new sequence split')
    if protocol.get('target_protocol')!='frozen_center_pseudo_independent_recorruption':
        blockers.append('Legacy or unknown target protocol')
    if protocol.get('target_offset')!=0 or protocol.get('student_offsets')!=list(range(-2,3)):
        blockers.append('Center-time geometry changed')
    if any(protocol.get(k)!=TILE for k in ['train_tile_packed','validation_tile_packed','inference_tile_packed']):
        blockers.append('Training/validation/inference context mismatch')
    if protocol.get('detail_weight')!=1.:
        blockers.append('Unreviewed detail weighting')
    for k in ['maximum_seconds','maximum_updates']:
        value=protocol.get(k)
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<=0:
            blockers.append('Missing explicit budget: '+k)
    if type(protocol.get('maximum_updates')) is not int:
        blockers.append('Update budget must be an integer')
    train={r['stream'] for r in protocol['records'] if r['role']=='train'}
    validation={r['stream'] for r in protocol['records'] if r['role']=='validation'}
    if not train or not validation or train&validation:
        blockers.append('Sequence split missing or overlapping')
    train_sources={r['source_sha256'] for r in protocol['records'] if r['role']=='train'}
    val_sources={r['source_sha256'] for r in protocol['records'] if r['role']=='validation'}
    if train_sources&val_sources:
        blockers.append('Duplicate source crosses sequence split')
    for key in sorted(train):
        try:
            profile=protocol['noise_profiles'].get(key,{})
            checked_noise(profile)
            record=next(r for r in protocol['records'] if r['stream']==key)
            if profile['sensor']!=record['sensor'] or profile.get('applies_to_source_sha256')!=record['source_sha256']:
                raise ValueError('Noise evidence belongs to another sensor/source')
        except (ValueError,OSError,KeyError) as error:blockers.append(key+': '+str(error))
    review=protocol.get('pseudo_target_review',{})
    if review.get('accepted') is not True or set(review.get('reviewed_streams',[]))!=train or review.get('teacher_sha256')!=OFFICIAL_TAP_SHA:
        blockers.append('Center pseudo-target quality review missing/incomplete')
    else:
        path=Path(review.get('evidence_file') or '')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=review.get('evidence_sha256'):
            blockers.append('Pseudo-target review evidence missing/changed')
        else:
            document=json.loads(path.read_text())
            if document.get('accepted') is not True or set(document.get('reviewed_streams',[]))!=train or document.get('teacher_sha256')!=OFFICIAL_TAP_SHA:
                blockers.append('Pseudo-target review evidence does not accept this candidate')
    if checkpoint is None or not Path(checkpoint).is_file():
        blockers.append('Initialization checkpoint missing')
    else:
        with Path(checkpoint).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=OFFICIAL_TAP_SHA:
                blockers.append('Wrong initialization checkpoint')
    return dict(training_ready=not blockers,blockers=blockers,gpu_calls=0,training_started=False)


class OnlineSchedule:
    """Cycle through every stream/context, choosing fresh spatial crops each draw."""
    def __init__(self, protocol, seed):
        self.rng = np.random.default_rng(seed)
        self.records = {r['stream']: r for r in protocol['records'] if r['role'] == 'train'}
        if not self.records:
            raise ValueError('No training streams')
        self.domains = sorted({r['sensor'] for r in self.records.values()})
        self.streams = {s: deque(sorted(k for k,r in self.records.items() if r['sensor'] == s)) for s in self.domains}
        self.contexts = {k: deque() for k in self.records}
        self.draws = 0

    def next(self):
        domain = self.domains[self.draws % len(self.domains)]
        queue = self.streams[domain]; key = queue[0]; queue.rotate(-1)
        if not self.contexts[key]:
            self.contexts[key].extend(self.rng.permutation(self.records[key]['centers']).tolist())
        frame = self.contexts[key].popleft(); self.draws += 1
        return dict(stream=key, frame=frame, crop_seed=int(self.rng.integers(2**63)), sensor=domain)


def source_stream(dataset_root,stream,layout):
    if layout not in ('collection','deployed'):
        raise ValueError('Explicit source layout required')
    root=Path(dataset_root).resolve()
    path=(root/stream['source_relative' if layout=='collection' else 'path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Source escapes dataset root')
    return dict(stream,path=str(path))


def read_context(dataset_root, stream, frame, crop_seed, size=TILE, role='train',layout='collection'):
    if stream['role'] != 'development' or '/shdarkroom/' in stream['source_relative']:
        raise ValueError('Only development RAW may enter adaptation')
    if role not in ('train','validation'):
        raise ValueError('Unknown role')
    lo, hi = stream['splits']['train'] if role == 'train' else (0, stream['frame_count']-1)
    indices = list(range(frame-4, frame+5))
    if not all(lo <= t <= hi for t in indices):
        raise ValueError('Nine-frame teacher window crosses partition')
    h,w = stream['raw_spec']['height']//2, stream['raw_spec']['width']//2
    if size != TILE or size > min(h,w):
        raise ValueError('Training, validation and inference require the same 192 packed tile')
    rng = np.random.default_rng(crop_seed)
    y,x = int(rng.integers(h-size+1)), int(rng.integers(w-size+1))
    local = source_stream(dataset_root,stream,layout)
    value = read_packed(dataset_root,local,indices,(y,x,size))
    return torch.from_numpy(signed_signal(value,stream['black_dn12'])), dict(frames=indices,crop_yxsize=[y,x,size])


def checked_noise(profile):
    if profile.get('verified') is not True or not profile.get('evidence_sha256') or not profile.get('evidence_file'):
        raise ValueError('Unverified noise profile; no pseudo-label inference or training allowed')
    evidence_path=Path(profile['evidence_file'])
    if not evidence_path.is_file() or hashlib.sha256(evidence_path.read_bytes()).hexdigest()!=profile['evidence_sha256']:
        raise ValueError('Noise evidence missing or changed')
    evidence=json.loads(evidence_path.read_text())
    fields=('shot_dn12_per_e','read_variance_dn12_sq','sensor','analog_gain','digital_gain','applies_to_source_sha256')
    if evidence.get('accepted_for_noise_synthesis') is not True or any(k not in profile or evidence.get(k)!=profile[k] for k in fields):
        raise ValueError('Noise evidence does not accept this sensor/gain/profile')
    shot = torch.as_tensor(profile['shot_dn12_per_e'], dtype=torch.float32)
    read = torch.as_tensor(profile['read_variance_dn12_sq'], dtype=torch.float32)
    if shot.shape != (4,) or read.shape != (4,) or not torch.isfinite(shot).all() or not torch.isfinite(read).all() or (shot<=0).any() or (read<0).any():
        raise ValueError('Noise requires four finite DN12 channel coefficients')
    return shot,read


@torch.no_grad()
def center_pseudo_sample(observed, predict_dn12, span, profile, generator=None):
    """Five targets at their own time, never a mean of future scene positions.

    predict_dn12 receives a 5x4xHxW signed DN12 window and must return its
    center 4xHxW signal. Caller supplies a frozen, hash-verified model.
    """
    shot,read = checked_noise(profile)  # Fail before any potentially costly model call.
    if observed.ndim != 4 or observed.shape[:2] != (9,4) or not torch.isfinite(observed).all():
        raise ValueError('Nine finite canonical signed DN12 frames required')
    span = torch.as_tensor(span, dtype=torch.float32, device=observed.device).reshape(4,1,1)
    if not torch.isfinite(span).all() or (span<=0).any() or (span>4095).any():
        raise ValueError('Invalid DN12 normalization span')
    predictions = [predict_dn12(observed[i:i+5]) for i in range(5)]
    if any(p.shape != observed.shape[1:] or not torch.isfinite(p).all() for p in predictions):
        raise ValueError('Invalid center prediction')
    pseudo = torch.stack(predictions).detach()
    # Physical noise synthesis requires nonnegative signals. Pseudo labels are
    # approximate targets, not independent real clean observations.
    clean = torch.minimum(pseudo.clamp_min(0),span)
    shot = shot.to(observed.device)[:,None,None]; read = read.to(observed.device)[:,None,None]
    noisy = torch.poisson(clean/shot,generator=generator)*shot
    noisy += torch.randn(clean.shape,device=clean.device,generator=generator)*read.sqrt()
    valid = (pseudo[2]>=0).all(0,keepdim=True) & (pseudo[2]<span-8).all(0,keepdim=True)
    if not valid.any():
        raise ValueError('No reliable unsaturated pseudo-target sites')
    return dict(signal=noisy,target=clean[2],valid=valid,
                target_kind='center_pseudo_not_clean_GT',target_offset=0,
                span_dn12=span,signal_unit='signed_dn12',
                noise_evidence_sha256=profile['evidence_sha256'])


def restoration_loss(output,target,valid):
    if output.shape != target.shape or not torch.isfinite(output).all() or not torch.isfinite(target).all():
        raise ValueError('Invalid loss tensors')
    mask = valid.bool().expand_as(output)
    if not mask.any():
        raise ValueError('No supervised sites')
    # One denominator and no overlapping detail MSE: no implicit 5x weighting.
    return (output-target).square()[mask].mean()


def candidate_loss(normalized_output,sample):
    """Training-facing loss rejects legacy future-frame labels and unit mixups."""
    if sample.get('target_kind')!='center_pseudo_not_clean_GT' or sample.get('target_offset')!=0:
        raise ValueError('Only reviewed center-time pseudo targets are supported; legacy temporal N2N is excluded')
    if sample.get('signal_unit')!='signed_dn12' or not sample.get('noise_evidence_sha256'):
        raise ValueError('Missing DN12 units or noise evidence')
    return restoration_loss(normalized_output,sample['target']/sample['span_dn12'],sample['valid'])


@torch.no_grad()
def infer_tiles(signal,predict_dn12,tile=TILE,stride=128):
    if tile != TILE or not 0 < stride <= tile or signal.ndim != 4:
        raise ValueError('Use the same 192 packed inference contract')
    h,w=signal.shape[-2:]
    if min(h,w)<tile:
        raise ValueError('Image smaller than inference tile')
    def coordinates(n):
        return sorted(set([*range(0,n-tile+1,stride),n-tile]))
    ramp=torch.hann_window(tile,periodic=False,device=signal.device).clamp_min(.01)
    weight=(ramp[:,None]*ramp[None,:])[None]
    result=torch.zeros_like(signal[0]);den=torch.zeros_like(result[:1])
    for y in coordinates(h):
        for x in coordinates(w):
            prediction=predict_dn12(signal[...,y:y+tile,x:x+tile])
            if prediction.shape != result[:,:tile,:tile].shape or not torch.isfinite(prediction).all():
                raise ValueError('Invalid tile prediction')
            result[:,y:y+tile,x:x+tile]+=prediction*weight
            den[:,y:y+tile,x:x+tile]+=weight
    if not (den>0).all():
        raise ValueError('Incomplete tile coverage')
    return result/den


def run_bounded(prepare,step,evaluate,save,*,maximum_seconds,maximum_updates,review_every,clock=time.monotonic,started_at=None):
    """One short candidate, then stop for visual review; never auto-resume.

    Deadline includes baseline evaluation, batch/label preparation and saves.
    Checks are between callbacks: one in-flight operation can overrun.
    """
    if not isinstance(maximum_seconds,(int,float)) or isinstance(maximum_seconds,bool) or not math.isfinite(maximum_seconds) or maximum_seconds<=0:
        raise ValueError('Explicit positive time budget required')
    if type(maximum_updates) is not int or maximum_updates<=0 or type(review_every) is not int or review_every<=0:
        raise ValueError('Explicit positive update/review limits required')
    start=clock() if started_at is None else started_at
    updates=0;reason='update_limit';baseline=None;result=None
    def expired():return clock()-start>=maximum_seconds
    try:
        if not expired():baseline=evaluate()
        while updates<maximum_updates:
            if expired():reason='time_limit';break
            batch=prepare()
            if expired():reason='time_limit';break
            value=float(step(batch));updates+=1
            if not math.isfinite(value):raise ValueError('Nonfinite training loss; stop without retry')
            if expired():reason='time_limit';break
            if updates%review_every==0 or updates==maximum_updates:
                result=evaluate()
                if expired():reason='time_limit';break
                save(dict(updates=updates,baseline=baseline,evaluation=result))
                reason='time_limit' if expired() else 'visual_review_required'
                break  # No proxy-loss-driven continuation without whole-image review.
    except BudgetExpired:
        reason='time_limit'
    return dict(updates=updates,elapsed_seconds=clock()-start,stop_reason=reason,
                automatic_retry=False,automatic_resume=False,evaluation=result)


def train_candidate(protocol,manifest,dataset_root,checkpoint,model_code,output,*,layout='collection',review_every=10):
    """Explicit one-candidate runner. No CUDA until CPU readiness checks pass."""
    start=time.monotonic()
    if type(review_every) is not int or review_every<=0:
        raise ValueError('Positive review interval required before CUDA initialization')
    ready=check_readiness(protocol,checkpoint)
    if not ready['training_ready']:
        raise ValueError(json.dumps(ready))
    if protocol.get('manifest_content_sha256')!=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest():
        raise ValueError('Manifest or calibration changed after protocol preparation')
    for record in protocol['records']:
        source=next((s for s in manifest['streams'] if s['id']==record['stream']),None)
        if source is None or source['role']!='development' or source['sha256']!=record['source_sha256']:
            raise ValueError('Protocol/source role or identity changed')
        path=Path(source_stream(dataset_root,source,layout)['path'])
        with path.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=source['sha256']:
                raise ValueError('RAW source hash changed before CUDA initialization')
    model_code=Path(model_code)
    if hashlib.sha256(model_code.read_bytes()).hexdigest()!=TAP_WRAPPER_SHA:
        raise ValueError('Use the pinned official TAP wrapper, not a historical experimental architecture')
    for name,expected in TAP_SOURCE_HASHES.items():
        if hashlib.sha256((model_code.parent/'tap_official'/(name+'.py')).read_bytes()).hexdigest()!=expected:
            raise ValueError('Official TAP source changed: '+name)
    seconds=protocol['maximum_seconds']
    def guard():
        if time.monotonic()-start>=seconds:raise BudgetExpired('Candidate time budget exhausted')
    guard()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    runtime=output/'runtime';runtime.mkdir()
    for key in ['TMPDIR','TORCH_HOME','TORCH_EXTENSIONS_DIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR']:
        path=runtime/key.lower();path.mkdir();os.environ[key]=str(path.resolve())
    tempfile.tempdir=os.environ['TMPDIR']
    if shutil.disk_usage(output).free<3*2**30:
        raise ValueError('3GiB free-space reserve required before CUDA initialization')
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable; no CPU fallback training')
    # All runtime files stay inside the explicitly selected new run directory.
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('_company_v2_pinned_tap',model_code)
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    try:
        guard();teacher=wrapper.OfficialTAP(checkpoint).eval().requires_grad_(False).to('cuda')
        guard();student=wrapper.OfficialTAP(checkpoint).eval().requires_grad_(False).to('cuda')
    except Exception as error:
        write_json(output/'failure.json',dict(error=str(error),updates=0,automatic_retry=False))
        raise
    student.network.pcd_align.requires_grad_(True)
    parameters=[p for p in student.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=0.)
    generator=torch.Generator(device='cuda').manual_seed(20261022)
    schedule=OnlineSchedule(protocol,20261022);streams={s['id']:s for s in manifest['streams']}
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    visited=[];actual_updates=0;baseline_created=False
    def predict(model,signal,shot,span):
        guard()
        return model.restore(signal[None],shot[None],span[None])[0]
    def prepare():
        item=schedule.next();s=streams[item['stream']]
        observed,meta=read_context(dataset_root,s,item['frame'],item['crop_seed'],layout=layout)
        guard();observed=observed.to('cuda');span=torch.tensor(4095-np.asarray(s['black_dn12']),dtype=torch.float32,device='cuda')
        profile=protocol['noise_profiles'][s['id']];shot=torch.tensor(profile['shot_dn12_per_e'],device='cuda')
        sample=center_pseudo_sample(observed,lambda x:predict(teacher,x,shot,span)*span[:,None,None],span,profile,generator)
        visited.append(dict(item,**meta));return sample,shot,span
    def step(batch):
        nonlocal actual_updates
        sample,shot,span=batch;guard();optimizer.zero_grad(set_to_none=True)
        loss=candidate_loss(predict(student,sample['signal'],shot,span),sample)
        if not torch.isfinite(loss):raise ValueError('Nonfinite candidate loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        guard();optimizer.step();actual_updates+=1
        return float(loss.detach())
    def evaluate():
        nonlocal baseline_created
        # Whole-image review with the same tiled forward as training; no
        # teacher-consistency score is presented as independent quality.
        from utils.rgb_metrics import packed_to_linear_rgb
        from PIL import Image
        record=next(r for r in protocol['records'] if r['role']=='validation')
        s=streams[record['stream']];frame=record['centers'][len(record['centers'])//2]
        local=source_stream(dataset_root,s,layout);guard()
        observed=torch.from_numpy(signed_signal(read_packed(dataset_root,local,list(range(frame-2,frame+3))),s['black_dn12'])).to('cuda')
        span=torch.tensor(4095-np.asarray(s['black_dn12']),dtype=torch.float32,device='cuda')
        # TAP forward ignores shot; no validation noise profile is synthesized.
        shot=torch.ones(4,device='cuda')
        model=student if baseline_created else teacher
        with torch.no_grad():
            restored=infer_tiles(observed,lambda x:predict(model,x,shot,span)*span[:,None,None])
        noisy=(observed[2].cpu()+16)/4095
        _,wb=packed_to_linear_rgb(noisy[None],'GBRG',return_wb=True)
        def render(value,name):
            guard();rgb=packed_to_linear_rgb(((value.cpu()+16)/4095)[None],'GBRG',wb_gains=wb)
            Image.fromarray((rgb[0].permute(1,2,0).numpy().clip(0,1)*255).astype('uint8')).save(output/name)
        label='after' if baseline_created else 'before'
        render(restored,label+'.png')
        if not baseline_created:render(observed[2],'noisy.png')
        baseline_created=True
        return dict(stream=s['id'],frame=frame,preview=label+'.png',clean_gt=False,
                    quality_proven=False,additional_sequence_video_review_required=True)
    def save(record):
        guard();torch.save(dict(model=student.state_dict(),optimizer=optimizer.state_dict(),
            protocol=protocol,updates=actual_updates,review=record),output/'candidate.pt')
    try:
        result=run_bounded(prepare,step,evaluate,save,maximum_seconds=seconds,
            maximum_updates=protocol['maximum_updates'],review_every=review_every,started_at=start)
        result.update(actual_optimizer_updates=actual_updates,visited=visited,training_ready_for_automatic_continuation=False)
        write_json(output/'result.json',result)
        return result
    except Exception as error:
        write_json(output/'failure.json',dict(error=str(error),updates=actual_updates,automatic_retry=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--execute',action='store_true',help='Explicit guarded CUDA pilot; never enabled by prepare')
    modes.add_argument('--check-ready',action='store_true',help='Read-only CPU checks; does not initialize CUDA')
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--dataset-root',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--model-code',type=Path)
    parser.add_argument('--layout',choices=['collection','deployed'],default='collection')
    parser.add_argument('--maximum-seconds',type=float)
    parser.add_argument('--maximum-updates',type=int)
    parser.add_argument('--review-every',type=int,default=10)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    if args.check_ready:
        if args.protocol is None:parser.error('--check-ready requires --protocol')
        protocol=json.loads(args.protocol.read_text())
        result=check_readiness(protocol,args.checkpoint)
        if protocol.get('manifest_content_sha256')!=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest():
            result['blockers'].append('Manifest changed after protocol preparation');result['training_ready']=False
        if args.output.exists():raise ValueError('Readiness output exists; choose a new report path')
        write_json(args.output,result);print(json.dumps(result))
        raise SystemExit(0 if result['training_ready'] else 2)
    if args.execute:
        if any(x is None for x in [args.protocol,args.dataset_root,args.checkpoint,args.model_code,args.maximum_seconds,args.maximum_updates]):
            parser.error('Execution requires protocol, data, pinned model files and explicit time/update budgets')
        protocol=json.loads(args.protocol.read_text())
        protocol.update(maximum_seconds=args.maximum_seconds,maximum_updates=args.maximum_updates)
        result=train_candidate(protocol,manifest,args.dataset_root,args.checkpoint,args.model_code,args.output,
            layout=args.layout,review_every=args.review_every)
        print(json.dumps(result));return
    result=prepare_protocol(manifest)
    result['manifest_sha256']=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if args.output.exists():raise ValueError('Output exists; old protocols must be preserved')
    write_json(args.output,result)
    print(json.dumps(dict(protocol_written=str(args.output),training_ready=False,training_started=False)))


if __name__=='__main__':main()
