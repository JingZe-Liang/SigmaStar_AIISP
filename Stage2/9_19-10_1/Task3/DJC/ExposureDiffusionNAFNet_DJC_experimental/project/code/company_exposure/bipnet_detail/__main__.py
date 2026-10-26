"""CPU checks, explicit mask preparation/readiness and CUDA-only branch training."""
import argparse
import json
from pathlib import Path
import torch
from company_exposure.core import load_student, sha256
from .model import BIPNetDetail, VENDOR_SHA
from . import workflow

def cpu_check(root, checkpoint, expected_sha):
    torch.set_num_threads(2)
    root, checkpoint = Path(root), Path(checkpoint)
    if sha256(checkpoint) != expected_sha:
        raise ValueError('Frozen checkpoint checksum mismatch')
    base = load_student(root, checkpoint, pretrained=False)
    model = BIPNetDetail(base, root/'vendor/BIPNet/Burst De-noising/Network.py')
    before = {k:v.clone() for k,v in base.state_dict().items()}
    x = torch.randn(2,5,4,16,16)*5+150
    args = ([24.,24.], torch.full((2,4),3838.5), torch.full((2,4),100.))
    model.eval()
    with torch.no_grad():
        torch.manual_seed(13); y, reference = model(x,*args)
    assert torch.equal(y,reference), 'Zero residual changed frozen inference'
    model.train()
    y, _ = model(x,*args)
    y.square().mean().backward()
    assert model.detail.conv3[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in base.parameters())
    assert all(torch.equal(before[k],v) for k,v in base.state_dict().items())
    assert torch.isfinite(y).all() and y.shape==(2,4,16,16)
    # Open the output head only to test the *potential* temporal gradient path;
    # no optimizer update or checkpoint is made.
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.detail.conv3[0].weight.normal_(0,.001)
    probe=x[:1].clone().requires_grad_(True)
    model(probe,[24.],[[3838.5]*4],[[100.]*4])[0].square().mean().backward()
    assert probe.grad[:,[0,1,3,4]].abs().sum()>0, 'Neighbor path inactive'
    assert torch.isfinite(probe.grad).all()
    for name in ['deform1','deform2','deform3','deform4']:
        grad=getattr(model.detail,name).weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    result=dict(passed=True,architecture='Official BIPNet core, RAW input/residual output adapted',
        official_network_sha256=VENDOR_SHA,base_sha256=expected_sha,
        trainable_parameters=sum(p.numel() for p in model.detail.parameters()),
        zero_residual_bitwise_equal=True,batch_size_two_preserved=True,
        neighbor_gradient=True,deformable_cpu_forward_backward=True,
        frozen_backbone_unchanged=True,optimizer_updates=0,
        cuda_checked=False,learned_detail_gain_proven=False,signature=workflow.signature())
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command')
    check=sub.add_parser('cpu-check')
    check.add_argument('--code-root',required=True)
    check.add_argument('--checkpoint',required=True)
    check.add_argument('--expected-sha',required=True)
    check.add_argument('--output',required=True)
    for name in ('prepare','check-ready','train'):
        command=sub.add_parser(name)
        command.add_argument('--root',required=True)
        command.add_argument('--config',required=True)
        if name=='train':command.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if not args.command:
        parser.print_help();return
    if args.command!='cpu-check':
        cfg=workflow.config(args.config)
        if args.command=='prepare':result=workflow.prepare(args.root,cfg)
        elif args.command=='check-ready':
            _,streams,groups=workflow.ready(args.root,cfg)
            result=dict(data_ready=True,streams=len(streams),cuda_available=torch.cuda.is_available(),
                        cuda_forward_backward_checked=False,training_started=False)
        else:
            workflow.train(args.root,cfg,args.run_id);result=dict(training_complete=True)
        print(json.dumps(result,indent=2));return
    output=Path(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite existing CPU report')
    result=cpu_check(args.code_root,args.checkpoint,args.expected_sha)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
