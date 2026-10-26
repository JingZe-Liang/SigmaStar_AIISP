"""Explicit multi-sensor inventory, independent masks, and bounded RAW loading."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from noise_model.raw_io import RawSpec, read_frames, count_frames, pack_bayer
from company_exposure.core import sha256, signed_signal, guidance_masks


def write_json(path, data):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def split_roles(n):
    cut = int(n * .7)
    return {'train': [0, cut - 1], 'validation': [cut + 10, n - 1]}


def centers(stream, role):
    lo, hi = stream['splits'][role]
    # Source t, N2N target t+6, odd guides t-3,...,t+9, teacher t-3,...,t+4.
    return [t for t in range(lo + 3, hi - 9 + 1) if t % 4 == 0]


def context(stream, role, t):
    lo, hi = stream['splits'][role]
    guides = list(range(t - 3, t + 10, 2))
    teacher = list(range(t - 3, t + 5))
    if t % 4 or not all(lo <= i <= hi for i in [t, t + 6, *guides, *teacher]):
        raise ValueError('Context crosses split or violates independent frame partition')
    assert set(guides).isdisjoint({t, t + 6})
    return guides, teacher


def build_inventory(dataset, output):
    """Full byte hash + all-frame container scan. No model fitting."""
    dataset, output = Path(dataset), Path(output)
    streams = []
    for folder in ['01_fist_company', '02_SC465SL_darkroom18', '04_Sigmastar_7_30', '05_2DNR3DNR']:
        for source in sorted((dataset / folder).rglob('*.raw')):
            rel = str(source.relative_to(dataset));name = source.name
            if folder.startswith('01'):
                sensor = 'SC450AI' if 'sc450ai' in rel else 'SC635HAI'
                width, height, shift = (2688, 1520, 0) if sensor == 'SC450AI' else (3200, 1800, 4)
                cfa, black = 'RGGB', [258.] * 4 if sensor == 'SC450AI' else [257.5] * 4
                basis = 'manufacturer readMe / supplied collection table; SC450AI measured low12 supersedes conflicting filename'
            elif folder.startswith('02'):
                sensor, width, height, shift, cfa, black = 'SC465SL', 2560, 1440, 0, None, [252.] * 4
                basis = 'manufacturer low_12_bits metadata; CFA unprovided'
            elif folder.startswith('05'):
                sensor, width, height, shift, cfa, black = '465_unspecified', 2560, 1440, 4, 'RGGB', None
                basis = 'filename RG; high12 from full storage scan; black level unprovided'
            else:
                sensor, width, height, shift = 'MIS20S1', 1920, 1080, 4
                cfa = 'GRBG' if 'calibrationdata' in rel else 'RGGB'
                black, basis = [256.5] * 4, 'independent MIS20S1 dark calibration reference'
            role = ('calibration' if 'calibrationdata' in rel else
                    'test' if '/shdarkroom/' in rel else 'development')
            stream_id = sensor.lower() + '_' + __import__('hashlib').sha256(rel.encode()).hexdigest()[:12]
            spec = RawSpec(width, height, bit_shift=shift, cfa=cfa or 'RGGB')
            n = count_frames(source, spec)
            entry = dict(id=stream_id, sensor=sensor, source_relative=rel,
                         path=f'data/raw/{stream_id}.raw', role=role, bytes=source.stat().st_size,
                         sha256=sha256(source), frame_count=n, cfa=cfa,
                         raw_spec=dict(width=width, height=height, bit_shift=shift, cfa=cfa,
                                       valid_bits=12, dtype='<u2', strict_alignment=True),
                         black_dn12=black, white_dn12=4095., parameter_basis=basis,
                         shot_gain_dn12_per_e=None, read_variance_dn12_sq=None,
                         splits=split_roles(n) if role == 'development' else {},
                         metadata={k: int(m.group(1)) if (m := re.search(pattern, name)) else None
                                   for k, pattern in [('exposure_us', r'(?:Shutter=|US=)(\d+)'),
                                                      ('sensor_gain_code', r'(?:SenserG=|AG=|_AG)(\d+)'),
                                                      ('isp_gain_code', r'(?:IspG=|DG=)(\d+)')]})
            # Every content/calibration frame is decoded by the sole RAW reader.
            minimum, maximum = 4095., 0.
            for i in range(n):
                frame = read_frames(source, spec, 1, i)
                minimum, maximum = min(minimum, float(frame.min())), max(maximum, float(frame.max()))
            entry['container_audit'] = dict(passed=True, scanned_frames=n, min_dn12=minimum, max_dn12=maximum,
                                            cfa_verified=cfa is not None)
            streams.append(entry)
            print(json.dumps({'inventory_stream':len(streams),'sensor':sensor,'role':role,'frames':n}),flush=True)
    payload = dict(format_version=1, seed=20261018, streams=streams,
                   units='effective DN12, canonical R,G1,G2,B after packing',
                   holdout='MIS20S1 shdarkroom all gains, including same-source 03 denoised/fused outputs',
                   excluded='03 derived streams and frame duplicates; no company 23DNR training targets',
                   scope='cross-sequence holdout; other-sensor laboratory views allowed; previously visually inspected test scene',
                   source_collection='company dataset, 01/02/04/05',
                   state='inventoried_not_trained')
    ids = [s['id'] for s in streams]
    assert len(ids) == len(set(ids))
    tests = {s['sha256'] for s in streams if s['role'] == 'test'}
    assert not tests.intersection(s['sha256'] for s in streams if s['role'] == 'development')
    write_json(output, payload)
    return payload


def spec_for(s):
    if s['cfa'] is None:
        raise ValueError(f"{s['id']}: CFA missing")
    return RawSpec(**s['raw_spec'])


def read_packed(root, stream, indices, crop=None):
    spec = spec_for(stream)
    values = []
    for i in indices:
        raw = read_frames(Path(root) / stream['path'], spec, 1, i)
        if crop is not None:
            y, x, size = crop
            raw = raw[:, 2*y:2*(y+size), 2*x:2*(x+size)]
        values.append(pack_bayer(raw, spec.cfa)[0])
    return np.stack(values)


def fit_noise_approx(root, s):
    """Training-only pair-difference slope; explicitly not dark/FPN calibration."""
    if s['role'] != 'development' or s['cfa'] is None or s['black_dn12'] is None:
        raise ValueError('Cannot fit missing CFA/black or non-training stream')
    hi = s['splits']['train'][1]
    indices = list(range(1, min(hi, 40), 2))
    # Sparse 16x16 tiles across a central downsampled full field are preferable
    # to a single ROI. Re-read small batches, avoiding a full-video RAM cache.
    samples = [[] for _ in range(4)]
    h, w = s['raw_spec']['height']//2, s['raw_spec']['width']//2
    for y in np.linspace(0, h-128, 3, dtype=int):
        for x in np.linspace(0, w-128, 3, dtype=int):
            obs = read_packed(root, s, indices, crop=(int(y), int(x), 128))
            sig = signed_signal(obs, s['black_dn12'])
            for c in range(4):
                for yy in range(0,128,16):
                    for xx in range(0,128,16):
                        v=sig[:,c,yy:yy+16,xx:xx+16]
                        if v.max() > 3800: continue
                        means=v.mean((1,2))
                        dif=np.diff(v,axis=0)
                        noise=float(np.median(dif.var((1,2)))/2)
                        if noise>0 and np.ptp(means)<max(8.,math.sqrt(noise)*2):
                            pooled=np.lib.stride_tricks.sliding_window_view(dif,(5,5),axis=(-2,-1)).mean((-1,-2))
                            factor=float(np.median(pooled.var((1,2)))/2/noise)
                            samples[c].append((float(v.mean()),noise,factor))
    slopes=[];intercepts=[];counts=[];factors=[]
    for points in samples:
        points=np.asarray(points)
        if len(points)<24 or np.ptp(points[:,0])<32:
            raise ValueError('Insufficient stable brightness coverage for noise slope')
        # Robustly suppress moving-texture outliers before a final fit.
        keep=np.ones(len(points),bool)
        for _ in range(3):
            a,b=np.linalg.lstsq(np.c_[points[keep,0],np.ones(keep.sum())],points[keep,1],rcond=None)[0]
            error=points[:,1]-(a*points[:,0]+b)
            mad=max(float(np.median(np.abs(error-np.median(error)))),1.)
            keep=np.abs(error-np.median(error))<3*mad
            if keep.sum()<24:raise ValueError('Unstable noise fit')
        if not np.isfinite(a+b) or a<=0:raise ValueError('Nonpositive shot slope')
        slopes.append(float(a));intercepts.append(max(float(b),1.));counts.append(int(keep.sum()))
        factors.append(float(np.clip(np.median(points[keep,2]),1/25,1)))
    return dict(shot_gain_dn12_per_e=slopes,read_variance_dn12_sq=intercepts,
                noise_basis='approximate training-only odd-frame tile difference fit; not full sensor likelihood or FPN calibration',
                noise_fit_points=counts,noise_fit_source_frames=indices,pooling_variance_factor=factors,
                noise_fit_version=2)


def make_masks(root, s, role, t):
    guides,_=context(s,role,t)
    obs=read_packed(root,s,guides)
    static,motion,stats=guidance_masks(obs,s['shot_gain_dn12_per_e'],s['read_variance_dn12_sq'],s['black_dn12'],s.get('pooling_variance_factor'))
    stats.update(guidance_frames=guides,source=t,target=t+6,role=role)
    return static,motion,stats


def prepare(root, manifest_path, fit_noise=False, overrides=None):
    root=Path(root);manifest=json.loads(Path(manifest_path).read_text());rows=[]
    if overrides:
        updates=json.loads(Path(overrides).read_text())
        by_id={s['id']:s for s in manifest['streams']}
        for key,change in updates.items():
            if key not in by_id:raise ValueError('Unknown override stream '+key)
            if not change.get('source'):raise ValueError('Override requires a metadata/calibration source')
            allowed={'cfa','black_dn12','shot_gain_dn12_per_e','read_variance_dn12_sq','source'}
            if set(change)-allowed:raise ValueError('Override may not change data role, hashes, storage or splits')
            s=by_id[key]
            noise_keys={'shot_gain_dn12_per_e','read_variance_dn12_sq'}
            if ('cfa' in change or 'black_dn12' in change) and not noise_keys.issubset(change):
                # A fit from the previous channel layout/black level is stale.
                s['shot_gain_dn12_per_e']=None;s['read_variance_dn12_sq']=None
                s.pop('noise_fit_version',None)
            if noise_keys.issubset(change):s.pop('noise_fit_error',None)
            for k,v in change.items():
                if k!='source':s[k]=v
            s['parameter_override_source']=change['source']
            if 'cfa' in change:s['raw_spec']['cfa']=change['cfa']
    for s in manifest['streams']:
        if s['role']!='development':continue
        errors=[];p=root/s['path']
        if not p.exists() or p.stat().st_size!=s['bytes'] or sha256(p)!=s['sha256']:
            errors.append('source missing or checksum mismatch')
        if s['cfa'] is None:errors.append('CFA missing')
        if s['black_dn12'] is None:errors.append('black_dn12 missing')
        if not errors and fit_noise and (s['shot_gain_dn12_per_e'] is None or s.get('noise_fit_version')!=2):
            try:
                s.update(fit_noise_approx(root,s));s.pop('noise_fit_error',None)
            except ValueError as e:s['noise_fit_error']=str(e)
        if s.get('noise_fit_error'):errors.append(s['noise_fit_error'])
        if s['shot_gain_dn12_per_e'] is None:errors.append('shot/read noise fit pending')
        row=dict(id=s['id'],sensor=s['sensor'],errors=errors,previews=[])
        if not errors:
            for role in ('train','validation'):
                cs=centers(s,role)
                if not cs:row['errors'].append(f'{role}: no valid context');continue
                t=cs[len(cs)//2]
                static,motion,stats=make_masks(root,s,role,t)
                view=np.zeros((*static.shape[1:],3),np.uint8)
                view[static[0]]=[50,200,80];view[motion[0]]=[230,70,50]
                dst=root/'reports/masks'/f"{s['id']}_{role}_{t}.png"
                dst.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(view).save(dst)
                row['previews'].append(stats)
            if not any(x['static_fraction']+x['motion_fraction']>0 for x in row['previews']):
                row['errors'].append('no supervised pixels in representative contexts')
        rows.append(row)
        print(json.dumps({'prepared':s['id'],'errors':row['errors']}),flush=True)
    manifest['preparation']=rows
    write_json(manifest_path,manifest)
    write_json(root/'reports/data_readiness.json',dict(rows=rows,ready=all(not r['errors'] for r in rows),
               training_started=False,teacher_batch_started=False,optimizer_updates=0))
    return rows
