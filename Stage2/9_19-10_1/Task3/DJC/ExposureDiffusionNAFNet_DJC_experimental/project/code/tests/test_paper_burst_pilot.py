import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from company_exposure.burst_detail import pilot


def test_context_is_independent_and_every_frame_stays_in_partition():
    stream=dict(role='development',splits={'train':[0,139],'validation':[150,199]})
    used={r:set() for r in stream['splits']}
    for role,(lo,hi) in stream['splits'].items():
        for frame in range(lo,hi+1):
            try:inputs,guides,target=pilot.context(stream,role,frame)
            except ValueError:continue
            assert len(guides)==8 and set(guides).isdisjoint([*inputs,target])
            assert all(lo<=i<=hi for i in [*inputs,*guides,target])
            used[role].update([*inputs,*guides,target])
    assert all(used.values()) and used['train'].isdisjoint(used['validation'])


def test_more_conservative_mask_removes_single_channel_temporal_change():
    torch.set_num_threads(2)
    s=dict(black_dn12=[256]*4,shot_gain_dn12_per_e=[2]*4,read_variance_dn12_sq=[4]*4)
    rng=np.random.default_rng(2)
    obs=np.full((8,4,64,64),450.,np.float32)+rng.normal(0,.1,(8,4,64,64)).astype(np.float32)
    static,_,_=pilot.conservative_masks(obs,s)
    assert static[:,8:-8,8:-8].mean()>.95
    changed=obs.copy();changed[0,0,20:44,20:44]+=50
    new,_,_=pilot.conservative_masks(changed,s)
    assert not new[:,24:40,24:40].any()
    assert not new[:,:4].any() and not new[:,:,:4].any()


def test_high_noise_static_area_is_not_destroyed_by_binary_erosion():
    torch.set_num_threads(2)
    s=dict(black_dn12=[256]*4,shot_gain_dn12_per_e=[2]*4,read_variance_dn12_sq=[4]*4)
    rng=np.random.default_rng(44)
    obs=450+rng.normal(0,np.sqrt(2*(450-256)+4),(8,4,192,192)).astype(np.float32)
    static,_,stats=pilot.conservative_masks(obs,s)
    assert stats['previous_static_pixels']>1000
    assert static.mean()>.05


def test_cpu_train_refuses_before_model_optimizer_or_run_creation(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'ready',lambda *a:{})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(pilot,'load_model',lambda *a,**k:pytest.fail('Loaded GPU model'))
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('Created optimizer'))
    with pytest.raises(ValueError,match='zero optimizer updates'):
        pilot.run(tmp_path,tmp_path,dict(maximum_seconds=900),'trial')
    assert not list(tmp_path.iterdir())


def test_revision_rejects_old_architecture_and_injection_contract(tmp_path):
    import json
    root=Path(__file__).resolve().parents[1]
    original=json.loads((root/'configs/exposure_paper_burst_server_20261022.json').read_text())
    path=tmp_path/'config.json';path.write_text(json.dumps(original))
    assert pilot.settings(path)['loss_steps']==[3]
    for changed in [{'architecture':'exposure_paper_burst_v1'}, {'loss_steps':[1,2,3]}, {'injection':'all_steps'}]:
        path.write_text(json.dumps({**original,**changed}))
        with pytest.raises(ValueError,match='Unsupported'):
            pilot.settings(path)
