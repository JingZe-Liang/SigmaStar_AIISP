"""Mechanics and evidence-transfer checks, not an image-quality benchmark."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from company_exposure.burst_detail.model import PaperBurstNAF
from company_exposure.burst_detail.references import VariableAFPM, official_classes
from company_exposure.core import exposure_steps, signed_signal
from company_exposure.data import read_packed
from noise_model.raw_io import RawSpec, encode_raw16, read_frames


def small_model():
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('_burst_test_naf',
        ROOT/'vendor/ExposureDiffusion/models/arch/NAFNet.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    base = source.NAFNet(4, 4, nf=8, enc_blk_nums=[1]*4, dec_blk_nums=[1]*4,
                        middle_blk_num=1, with_photon=True, adaptive_res_and_x0=True)
    return PaperBurstNAF(base, width=8)


def open_branch(model):
    # Open residual gates to test all paths, without an optimizer/training step.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.startswith('backbone.') and name.endswith(('.beta', '.gamma')):
                param.fill_(.1)
        model.to_residual.weight.normal_(std=.01)


def test_zero_injection_matches_all_three_steps_and_freezes_backbone():
    torch.manual_seed(9)
    model = small_model().train()
    x = torch.randn(2, 5, 4, 31, 47)*5+200
    state = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    shot, span, read = [[2., 3., 4., 5.]]*2, [[3800., 3801., 3802., 3803.]]*2, [4.]*2
    with torch.no_grad():
        torch.manual_seed(77)
        expected = exposure_steps(model.backbone, x[:, 2], shot, span)
        torch.manual_seed(77)
        actual = model(x, shot, span, read)
    assert len(actual) == 3 and all(a.shape == (2, 4, 31, 47) for a in actual)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert all(torch.equal(state[k], v) for k, v in model.backbone.state_dict().items())
    assert not model.backbone.training
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert not model.backbone.ending._forward_pre_hooks
    assert not model.backbone.decoders[2]._forward_hooks
    sum(y.square().mean() for y in model(x, shot, span, read)).backward()
    assert model.to_residual.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())


def test_neighbors_noise_alignment_gradients_and_center_ablation_are_real():
    torch.manual_seed(10)
    model = small_model()
    open_branch(model)
    x = (torch.randn(1, 5, 4, 32, 48)*15+100).requires_grad_()
    fine = model.features(x, torch.full((1,1,4,1,1), 20.),
                          torch.full((1,1,4,1,1), 3800.),
                          torch.full((1,1,4,1,1), 40.))
    fine.square().mean().backward()
    assert all(x.grad[:, i].abs().sum() > 0 for i in range(5))
    assert model.align[0].dcn_pack.conv_offset_mask.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.adapter_parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    x.grad = None
    args = (x, torch.full((1,1,4,1,1), 20.), torch.full((1,1,4,1,1), 3800.),
            torch.full((1,1,4,1,1), 40.))
    center = model.features(*args, temporal=False)
    center.sum().backward()
    assert all(x.grad[:, i].abs().sum() == 0 for i in (0, 1, 3, 4))
    assert not torch.equal(center, fine)
    altered = (*args[:3], args[3]*100)
    assert not torch.equal(model.features(*altered), fine)


@pytest.mark.parametrize('shape', [(32, 48), (48, 64)])
def test_dynamic_afpm_matches_official_forward_and_gradient(shape):
    torch.set_num_threads(2)
    _, _, afpm, _ = official_classes()
    official = afpm(4, 16, H=shape[0], W=shape[1])
    adaptive = VariableAFPM(official)
    x = torch.randn(2, 4, *shape, requires_grad=True)
    a, b = official(x), adaptive(x)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    ga = torch.autograd.grad(a.square().mean(), x)[0]
    gb = torch.autograd.grad(b.square().mean(), x)[0]
    torch.testing.assert_close(ga, gb, rtol=2e-5, atol=2e-6)


def test_final_only_reference_loss_and_old_targets_are_refused():
    from company_exposure.burst_detail import workflow
    sample=dict(target=torch.full((4,32,48),100.),span_dn12=torch.full((4,1,1),3800.),
                valid=torch.ones(1,32,48,dtype=torch.bool),target_kind='center_pseudo_not_clean_GT',
                target_offset=0,signal_unit='signed_dn12',noise_evidence_sha256='fixture-evidence')
    predictions=[torch.full((1,4,32,48),.1,requires_grad=True) for _ in range(3)]
    loss=workflow.training_loss(predictions,sample)
    loss.backward()
    assert all(v.grad is None for v in predictions[:2])
    assert predictions[-1].grad.abs().sum()>0
    with pytest.raises(ValueError,match='legacy'):
        workflow.training_loss(predictions,{**sample,'target_offset':6})
    with pytest.raises(ValueError,match='noise evidence'):
        workflow.training_loss(predictions,{**sample,'noise_evidence_sha256':None})
    clean={**sample,'target_kind':'independent_clean_center','reference_evidence_sha256':'fixture-clean-source'}
    torch.testing.assert_close(workflow.training_loss(predictions,clean),loss)
    with pytest.raises(ValueError,match='center geometry'):
        workflow.training_loss(predictions,{**clean,'target_offset':6})


def test_finite_rejection_checkpoint_transaction_and_hook_cleanup():
    model = small_model()
    x = torch.full((1, 5, 4, 32, 32), 150.)
    state = model.adapter_state()
    model.load_adapter(state)
    assert not any(k.startswith('backbone.') for k in state)
    for bad in ({}, {**state, 'stem.weight': state['stem.weight']*float('nan')}):
        with pytest.raises(ValueError):
            model.load_adapter(bad)
        assert all(torch.equal(model.adapter_state()[k], v) for k, v in state.items())
    for xx, shot, span, read in [(x, [0], [3800], [4]), (x, [2], [65535], [4]),
                                (x, [2], [3800], [-1]), (x*100, [2], [3800], [4]),
                                (x.double(), [2], [3800], [4])]:
        with pytest.raises(ValueError):
            model(xx, shot, span, read)
    original = model.backbone.forward
    def failure(*args):
        raise RuntimeError('deliberate forward failure')
    model.backbone.forward = failure
    with pytest.raises(RuntimeError, match='deliberate'):
        model(x, [2], [3800], [4])
    assert not model.backbone.ending._forward_pre_hooks
    assert not model.backbone.decoders[2]._forward_hooks
    model.backbone.forward = original


@pytest.mark.parametrize('shift', [0, 4])
def test_raw_units_through_real_reader_and_signed_branch(tmp_path, shift):
    torch.set_num_threads(2)
    spec = RawSpec(64, 96, cfa='RGGB', bit_shift=shift)
    dn = (np.arange(5*96*64).reshape(5,96,64) % 4096).astype(np.float32)
    raw = encode_raw16(dn, spec)
    path = tmp_path/'fixture.raw'
    path.write_bytes(raw.astype('<u2').tobytes())
    assert np.array_equal(read_frames(path, spec, 5), dn)
    stream = dict(path=path.name, cfa='RGGB', raw_spec=dict(width=64, height=96, cfa='RGGB', bit_shift=shift))
    packed = read_packed(tmp_path, stream, list(range(5)))
    assert packed.max() == 4095
    signal = signed_signal(packed, [256]*4)
    assert signal.min() == -256
    model = small_model()
    with torch.no_grad():
        outputs = model(torch.from_numpy(signal)[None], [2], [3839], [4])
    assert all(o.shape == (1,4,48,32) and torch.isfinite(o).all() for o in outputs)
    if shift == 4:
        corrupt = raw.copy(); corrupt[0,0,0] |= 1
        path.write_bytes(corrupt.astype('<u2').tobytes())
        with pytest.raises(ValueError):
            read_packed(tmp_path, stream, [0])


def test_final_step_isolation_and_real_second_update_reaches_alignment():
    torch.manual_seed(22)
    model=small_model().train()
    x=torch.randn(1,5,4,32,32)*8+140
    optimizer=torch.optim.AdamW(model.adapter_parameters(),lr=1e-3,weight_decay=0)
    original={k:v.clone() for k,v in model.backbone.state_dict().items()}
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(19);out=model(x,[20],[3800],[4])
        out[-1].square().mean().backward()
        if update==0:
            assert model.stem.weight.grad.abs().sum()==0
        else:
            assert model.stem.weight.grad.abs().sum()>0
            assert model.align[0].dcn_pack.conv_offset_mask.weight.grad.abs().sum()>0
            assert model.align[2].conv_fusion.weight.grad.abs().sum()>0
        assert all(torch.isfinite(p.grad).all() for p in model.adapter_parameters() if p.grad is not None)
        optimizer.step()
    with torch.no_grad():
        torch.manual_seed(19);expected=model(x,[20],[3800],[4],enabled=False)
        torch.manual_seed(19);actual=model(x,[20],[3800],[4])
    assert all(torch.equal(a,b) for a,b in zip(actual[:2],expected[:2]))
    assert not torch.equal(actual[-1],expected[-1])
    assert all(torch.equal(original[k],v) for k,v in model.backbone.state_dict().items())
    from company_exposure.burst_detail.workflow import final_mixed_loss
    values=[torch.full((1,4,4,4),.2,requires_grad=True) for _ in range(3)]
    loss,_=final_mixed_loss(values,torch.ones(1,4,4,4),torch.zeros(1,4,4,4),
                            torch.ones(1,1,4,4,dtype=torch.bool),torch.zeros(1,1,4,4,dtype=torch.bool),torch.full((1,4),100.))
    loss.backward()
    assert values[0].grad is None and values[1].grad is None
    torch.testing.assert_close(loss,torch.tensor((.2-.01)**2))


def test_residual_is_added_to_image_and_zero_strength_recovers_original(monkeypatch):
    model=small_model().train();open_branch(model)
    x=torch.randn(1,5,4,31,47)*8+150
    # Any attempt to enter the original model using a hook must fail this test.
    def forbidden(*a,**kw):
        pytest.fail('Internal backbone hook used')
    for module in model.backbone.modules():
        monkeypatch.setattr(module,'register_forward_hook',forbidden)
        monkeypatch.setattr(module,'register_forward_pre_hook',forbidden)
    torch.manual_seed(17);base=model(x,[20],[3800],[4],enabled=False)
    torch.manual_seed(17);full=model(x,[20],[3800],[4])
    torch.manual_seed(17);zero=model(x,[20],[3800],[4],residual_scale=0)
    torch.manual_seed(17);half=model(x,[20],[3800],[4],residual_scale=.5)
    residual=model.features(x,torch.full((1,1,4,1,1),20.),torch.full((1,1,4,1,1),3800.),torch.full((1,1,4,1,1),4.))
    assert all(torch.equal(a,b) for a,b in zip(zero,base))
    assert all(not a.requires_grad for a in base)
    torch.testing.assert_close(full[-1],base[-1]+residual)
    torch.testing.assert_close(half[-1],base[-1]+.5*residual)
    for value in [-1,2,float('nan')]:
        with pytest.raises(ValueError,match='Residual scale'):
            model(x,[20],[3800],[4],residual_scale=value)
