import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noise_model.raw_io import RawSpec, decode_raw16, encode_raw16, pack_bayer, unpack_bayer
from company_exposure.core import exposure_steps, mixed_loss, signed_signal, guidance_masks


@pytest.mark.parametrize('shift', [0, 4])
@pytest.mark.parametrize('cfa', ['RGGB', 'GRBG', 'GBRG', 'BGGR'])
def test_units_and_bayer_roundtrip(shift, cfa):
    spec = RawSpec(16, 16, bit_shift=shift, cfa=cfa)
    dn = np.arange(256, dtype=np.float32).reshape(16, 16) * 16
    assert np.array_equal(decode_raw16(encode_raw16(dn, spec), spec), dn)
    assert np.array_equal(unpack_bayer(pack_bayer(dn, cfa), cfa)[0], dn)
    bad = np.ones((2, 2), np.uint16) if shift else np.full((2, 2), 4096, np.uint16)
    with pytest.raises(ValueError): decode_raw16(bad, spec)
    with pytest.raises(ValueError): encode_raw16(np.array([[4096]], np.float32), spec)


def test_signed_black_and_photon_units():
    obs = np.full((1, 4, 4, 4), 250, np.float32)
    assert signed_signal(obs, [256] * 4).min() == -6
    assert signed_signal(obs, [256] * 4, clip_min=0).min() == 0
    dn, span, gain, ratio = 150., 3838.5, 24., 100.
    k = gain * 15583 / (span * ratio)
    assert np.isclose(dn / span / ratio / k * 15583, dn / gain)


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 4, 1)

    def forward(self, x): return torch.sigmoid(self.conv(x)), {}


def test_three_step_gradients_and_no_target_in_recurrence():
    model = TinyHead()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    signal = torch.full((2, 4, 16, 16), 150.)
    outputs = exposure_steps(model, signal, [24., 24.], [3838.5, 3838.5], training=True)
    for x in outputs: x.retain_grad()
    static = torch.ones(2, 1, 16, 16, dtype=torch.bool);static[:, :, 8:] = False
    motion = ~static
    target = torch.full_like(signal, -1.)  # Signed noisy target is deliberately not clipped.
    loss, _ = mixed_loss(outputs, target, signal, static, motion, [3838.5] * 2)
    loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in outputs)
    assert model.conv.weight.grad.abs().sum() > 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match='no supervised'):
        mixed_loss(outputs, target, signal, static & False, motion & False, [3838.5] * 2)
    with pytest.raises(ValueError, match='overlap'):
        mixed_loss(outputs, target, signal, static, static, [3838.5] * 2)


def test_mask_rejects_global_illumination_and_detects_local_motion():
    g = np.full((7, 4, 32, 32), 456, np.float32)
    s, m, _ = guidance_masks(g, [2] * 4, [4] * 4, [256] * 4)
    assert s.mean() > .8 and not m.any()
    g[3, :, 12:20, 12:20] += 150
    _, m, _ = guidance_masks(g, [2] * 4, [4] * 4, [256] * 4)
    assert m.any()
    g[3] += 200
    s, m, stats = guidance_masks(g, [2] * 4, [4] * 4, [256] * 4)
    assert stats['global_illumination_rejected'] and not s.any() and not m.any()


def test_context_independence_and_split_guards():
    from company_exposure.data import split_roles,centers,context
    s={'splits':split_roles(200)}
    used={}
    for role in ['train','validation']:
        used[role]=set()
        for t in centers(s,role):
            guides,teacher=context(s,role,t)
            assert all(i%2 for i in guides) and t%2==(t+6)%2==0
            assert set(guides).isdisjoint({t,t+6})
            assert t+6 not in teacher
            used[role].update([t,t+6,*guides,*teacher])
    assert used['train'].isdisjoint(used['validation'])
    with pytest.raises(ValueError):context(s,'train',0)


def test_training_gate_precedes_optimizer(monkeypatch,tmp_path):
    from company_exposure import workflow
    monkeypatch.setattr(workflow,'readiness',lambda root:{'ready':False,'errors':['teacher labels pending']})
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('Optimizer must not be constructed'))
    with pytest.raises(ValueError,match='before optimizer'):
        workflow.train(tmp_path,'must_not_start')
    assert not (tmp_path/'runs').exists()


def test_sparse_teacher_preserves_valid_dn12():
    shape=(4,8,8);mask=torch.zeros(1,8,8,dtype=torch.bool);mask[:,2:6,2:6]=True
    prediction=torch.arange(256,dtype=torch.float32).reshape(shape)/7
    stored=prediction[:,mask[0]].T.clone()
    restored=torch.zeros(shape);restored[:,mask[0]]=stored.T
    assert torch.equal(restored[:,mask[0]],prediction[:,mask[0]])


def test_noise_aware_mask_keeps_static_noisy_background():
    rng=np.random.default_rng(17)
    guides=(456+rng.normal(0,10,(7,4,96,96))).astype(np.float32)
    static,moving,stats=guidance_masks(guides,[.001]*4,[100]*4,[256]*4)
    assert static.mean()>.2
    assert moving.mean()<.01


def test_matches_official_three_step_recurrence():
    import ast
    from types import SimpleNamespace
    root=Path(__file__).resolve().parents[1]
    path=root/'vendor/ExposureDiffusion/models/ELD_model_iter.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ELDModelIter')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    ns={'torch':torch,'np':np}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    model=TinyHead().eval()
    signal=torch.arange(4,dtype=torch.float32).reshape(1,4,1,1).expand(1,4,16,16)*100+10
    holder=SimpleNamespace(input=signal[:,[0,1,3,2]]/3838.5,ratio=100,
        K=24*15583/(3838.5*100),netG=model,
        opt=SimpleNamespace(with_photon=True,concat_origin=False,chop=False),
        target=None,corrector=lambda prediction,target:prediction)
    torch.manual_seed(33)
    ns['forward'](holder,iter_num=2)
    torch.manual_seed(33)
    actual=exposure_steps(model,signal,[24.],[3838.5])
    for value,(expected,_) in zip(actual,holder.output_list):
        assert torch.allclose(value,expected[:,[0,1,3,2]],atol=1e-5,rtol=1e-5)


def test_signed_observation_option_keeps_legacy_default(tmp_path):
    from noise_model.calibrated_observation import CalibratedObservationModel
    model=CalibratedObservationModel(gain=1,gain_dn_per_electron=1.,
        fixed_map_dn=torch.full((4,4,4),256.),mixture_weights=torch.ones(4,1),
        mixture_means_dn=torch.zeros(4,1),mixture_sigmas_dn=torch.ones(4,1),
        row_cov=torch.zeros(4,4),col_cov=torch.zeros(4,4),frame_offsets_dn=torch.zeros(2,4))
    observed=torch.full((1,4,4,4),250.)
    assert model.observation_to_signal(observed,0,0).min()==0
    assert model.observation_to_signal(observed,0,0,clip_min=None).min()==-6


def test_masked_loss_region_normalization_and_weights():
    outputs=[torch.full((1,4,2,2),2.,requires_grad=True) for _ in range(3)]
    static=torch.zeros(1,1,2,2,dtype=torch.bool);static[:,:,0,0]=True
    target=torch.zeros_like(outputs[0]);teacher=torch.full_like(target,4.)
    loss,metrics=mixed_loss(outputs,target,teacher,static,~static,[1.])
    assert torch.allclose(loss,torch.tensor(6.))
    assert metrics=={'static_mse':4.,'motion_mse':4.}


def test_parameter_override_clears_failed_fit_and_invalidates_old_black_fit(tmp_path):
    import json
    from dataclasses import asdict
    from company_exposure.data import prepare,split_roles
    from company_exposure.core import sha256
    raw=tmp_path/'source.raw';spec=RawSpec(16,16,bit_shift=0,cfa='RGGB')
    encode_raw16(np.full((100,16,16),300,np.float32),spec).tofile(raw)
    stream=dict(id='example',sensor='test',role='development',path='source.raw',bytes=raw.stat().st_size,
                sha256=sha256(raw),cfa='RGGB',black_dn12=[256.]*4,raw_spec=asdict(spec),
                splits=split_roles(100),shot_gain_dn12_per_e=None,read_variance_dn12_sq=None,
                noise_fit_error='old fit failed')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'streams':[stream]}))
    overrides=tmp_path/'overrides.json';overrides.write_text(json.dumps({'example':{
        'shot_gain_dn12_per_e':[2.]*4,'read_variance_dn12_sq':[4.]*4,'source':'independent calibration'}}))
    rows=prepare(tmp_path,manifest,overrides=overrides)
    assert not rows[0]['errors']
    overrides.write_text(json.dumps({'example':{'black_dn12':[257.]*4,'source':'updated calibration'}}))
    rows=prepare(tmp_path,manifest,overrides=overrides)
    updated=json.loads(manifest.read_text())['streams'][0]
    assert updated['shot_gain_dn12_per_e'] is None and updated['read_variance_dn12_sq'] is None
    assert 'shot/read noise fit pending' in rows[0]['errors']


@pytest.mark.parametrize('illumination_change',[0.,200.])
def test_mask_memory_refactor_preserves_full_tensor_result(illumination_change):
    import torch.nn.functional as F
    rng=np.random.default_rng(33)
    guides=(456+rng.normal(0,15,(7,4,48,64))).astype(np.float32)
    guides[3,:,12:20,12:20]+=150;guides[0]+=illumination_change
    guides[:,:,4:6,4:6]=4095
    shot,read,black,factor=[2.,3.,4.,5.],[4.,7.,8.,9.],[256.5]*4,[.08,.11,.15,.18]
    low=F.avg_pool2d(torch.from_numpy(signed_signal(guides,black)),5,1,2)
    variance=(torch.tensor(shot)[None,:,None,None]*low.mean(0,keepdim=True).clamp_min(0)+
              torch.tensor(read)[None,:,None,None]).clamp_min(1)*torch.tensor(factor)[None,:,None,None]
    score=(low.var(0,unbiased=True,keepdim=True)/variance).mean(1,keepdim=True)
    shift=float((low-low.median(0,keepdim=True).values).flatten(2).median(2).values.abs().amax())
    rejected=shift>max(4.,float(variance.sqrt().mean())*3)
    static,motion=score<1.5,score>3.
    if rejected:static.zero_();motion.zero_()
    static=-F.max_pool2d(-static.float(),5,1,2)>.5
    safe=(torch.from_numpy(guides).amax((0,1))<4087)[None,None]
    motion&=~static;static&=safe;motion&=safe
    actual_s,actual_m,stats=guidance_masks(guides,shot,read,black,factor)
    assert np.array_equal(actual_s,static[0].numpy())
    assert np.array_equal(actual_m,motion[0].numpy())
    assert stats['global_illumination_rejected']==rejected
    assert np.isclose(stats['global_shift_dn12'],shift)
    assert np.isclose(stats['median_normalized_temporal_variance'],float(score.median()),rtol=1e-5)


def test_real_readiness_rejects_missing_parameters_labels_and_low_space(monkeypatch,tmp_path):
    import json
    from types import SimpleNamespace
    from company_exposure import workflow
    from company_exposure.data import split_roles
    (tmp_path/'weights').mkdir();(tmp_path/'reports').mkdir();(tmp_path/'data').mkdir()
    expected={'sid_Pg_naf2.pt':workflow.STUDENT_SHA,'RViDeformer-L_CRVD.pth':workflow.TEACHER_SHA}
    for name in expected:(tmp_path/'weights'/name).write_bytes(b'fixture')
    streams=[]
    for i in range(45):
        role='development' if i<20 else 'calibration' if i<38 else 'test'
        key=f'stream_{i}';path=f'data/{key}.raw'
        if role!='test':(tmp_path/path).write_bytes(b'x')
        streams.append(dict(id=key,role=role,path=path,bytes=1,sha256=key,source_relative=f'synthetic/{key}',
            cfa='RGGB',black_dn12=None if i==0 else [256.]*4,
            shot_gain_dn12_per_e=[2.]*4,read_variance_dn12_sq=[4.]*4,splits=split_roles(100)))
    (tmp_path/'manifest.json').write_text(json.dumps({'streams':streams}))
    (tmp_path/'config.json').write_text(json.dumps(dict(precision='float32',virtual_exposure_ratio=100,
                                                       static_weight=1.,motion_weight=.5)))
    (tmp_path/'reports/cpu_selfcheck.json').write_text(json.dumps({'passed':True,'code_signature':'fixture'}))
    monkeypatch.setattr(workflow,'sha256',lambda p:expected.get(Path(p).name,Path(p).stem))
    monkeypatch.setattr(workflow,'code_signature',lambda:'fixture')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(workflow.shutil,'disk_usage',lambda p:SimpleNamespace(free=1))
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('Blocked run created optimizer'))
    with pytest.raises(ValueError,match='Training blocked before optimizer') as error:
        workflow.train(tmp_path,'blocked_fixture')
    message=str(error.value)
    assert 'black_dn12 missing/invalid' in message
    assert 'teacher labels pending' in message
    assert 'less than 2 GiB free' in message
    assert not (tmp_path/'runs').exists()


def test_mask_noise_scale_invariant_to_black_reparameterization():
    rng=np.random.default_rng(20261018)
    guides=(256+rng.normal(0,18,(7,4,96,96))).astype(np.float32)
    # Same observed-DN variance line, expressed at different working pedestals.
    static,motion,_=guidance_masks(guides,[25]*4,[400]*4,[256]*4)
    for delta in (-8,8):
        ss,mm,_=guidance_masks(guides,[25]*4,[400+25*delta]*4,[256+delta]*4)
        # Zero-padded pooling has incomplete support at the boundary.
        assert np.mean(ss[:,4:-4,4:-4]!=static[:,4:-4,4:-4])<.001
        assert np.mean(mm[:,4:-4,4:-4]!=motion[:,4:-4,4:-4])<.001


def test_batched_sparse_teacher_preserves_supervised_pixels():
    from company_exposure.workflow import teacher_frame
    class LocalMean(torch.nn.Module):
        def forward(self,x):
            b,t,c,h,w=x.shape
            return torch.nn.functional.avg_pool2d(x.reshape(b*t,c,h,w),3,1,1).reshape(b,t,c,h,w)
    rng=np.random.default_rng(5);signal=rng.uniform(0,.8,(8,4,192,192)).astype(np.float32)
    motion=np.zeros((1,192,192),bool);motion[:,11:34,140:169]=True;motion[:,82:95,95:119]=True
    full,_=teacher_frame(LocalMean(),signal,1.,tile_batch=1,device='cpu')
    sparse,_=teacher_frame(LocalMean(),signal,1.,motion=motion,tile_batch=3,device='cpu')
    assert np.allclose(full[:,motion[0]],sparse[:,motion[0]],atol=1e-7)
    empty,_=teacher_frame(LocalMean(),signal,1.,motion=motion&False,tile_batch=3,device='cpu')
    assert not empty.any()


def test_prefetch_preserves_order_and_propagates_errors():
    from company_exposure.workflow import prefetch_map
    assert list(prefetch_map(lambda x:x*x,range(5)))==[0,1,4,9,16]
    assert list(prefetch_map(lambda x:x,[]))==[]
    def fail(x):
        if x==2:raise ValueError('bad prepared batch')
        return x
    it=prefetch_map(fail,range(4));assert next(it)==0;assert next(it)==1
    with pytest.raises(ValueError,match='bad prepared batch'):next(it)


def test_parallel_cpu_sampling_is_seed_deterministic(monkeypatch):
    from company_exposure import workflow
    def tiny(root,stream,item,rng,size):return {'signal':torch.tensor(rng.random(4),dtype=torch.float32)}
    monkeypatch.setattr(workflow,'sample',tiny)
    streams={'a':{'sensor':'sensor_a'},'b':{'sensor':'sensor_b'}};items={'a':[{}],'b':[{}]}
    a=workflow.batch_samples(None,streams,items,np.random.default_rng(7),128,8,device='cpu',workers=1)
    b=workflow.batch_samples(None,streams,items,np.random.default_rng(7),128,8,device='cpu',workers=4)
    assert torch.equal(a['signal'],b['signal'])
