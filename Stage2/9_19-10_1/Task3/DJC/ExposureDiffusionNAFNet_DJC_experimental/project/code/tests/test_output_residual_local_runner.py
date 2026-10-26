import importlib.util,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('portable_runner',ROOT/'scripts/exposure_output_residual_local.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_validation_keeps_training_rng_and_honors_independent_control():
    class Example(torch.nn.Module):
        def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(.1))
        def forward(self,signal,shot,span,read,enabled=True):
            torch.rand(3)
            base=signal[:,2]/10
            return [base,base,base+self.p*signal[:,0] if enabled else base]
    model=Example().train();signal=torch.arange(5.).reshape(5,1,1,1).expand(5,4,4,4)
    b=dict(signal=signal,target=torch.zeros(4,4,4),teacher=torch.zeros(4,4,4),static=torch.ones(1,4,4,dtype=torch.bool),motion=torch.zeros(1,4,4,dtype=torch.bool),span=torch.full((4,),10.),shot=torch.ones(4),read=torch.ones(4))
    sample=[(b,dict(stream='s'))];streams={'s':{'sensor':'sensor'}}
    before=torch.get_rng_state().clone()
    real=runner.validate(model,sample,streams,'cpu',False)
    center=runner.validate(model,sample,streams,'cpu',True)
    assert torch.equal(before,torch.get_rng_state()) and model.training
    assert model.p.grad is None
    assert center['equal_domain_proxy_loss']>real['equal_domain_proxy_loss']
    assert real['rows'][0]['metrics']['candidate']['static_edge_mse_dn12']==0


def test_checkpoint_roundtrip_preserves_adamw_and_all_cpu_rng(tmp_path):
    class Adapter(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(4))
        def adapter_state(self):return {'weight':self.weight.detach().clone()}
    m=Adapter();opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    m.weight.square().sum().backward();opt.step()
    runner.save_checkpoint(tmp_path/'last.pt',m,opt,17,.4,{'fixture':'test'})
    state=torch.load(tmp_path/'last.pt',map_location='cpu',weights_only=True)
    a=torch.rand(4);b=np.random.rand(4)
    runner.restore_random(state['random'])
    torch.testing.assert_close(torch.rand(4),a);np.testing.assert_array_equal(np.random.rand(4),b)
    other=torch.optim.AdamW(m.parameters(),lr=.5);other.load_state_dict(state['optimizer'])
    assert other.param_groups[0]['lr']==1e-4
    assert state['step']==17 and state['contract']=={'fixture':'test'}
    assert not (tmp_path/'last.part').exists()


def test_cuda_absence_refuses_before_model_optimizer_or_run_creation(tmp_path,monkeypatch):
    import pytest
    monkeypatch.setattr(runner.pilot,'ready',lambda *a:{})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(runner,'load_model',lambda *a,**k:pytest.fail('Model created'))
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('Optimizer created'))
    monkeypatch.setattr(sys,'argv',['runner','--root',str(tmp_path),'--work',str(tmp_path/'work'),'train','--run-id','v3'])
    with pytest.raises(ValueError,match='CUDA unavailable'):runner.main()
    assert not (tmp_path/'work').exists()
