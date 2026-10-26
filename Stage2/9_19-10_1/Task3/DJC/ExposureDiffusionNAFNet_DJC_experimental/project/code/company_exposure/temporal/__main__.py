"""All actions are explicit; no arguments prints help and does nothing."""
import argparse
import json
import sys
import torch
from company_exposure.workflow import CODE
from . import workflow as w


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(CODE/'configs/exposure_temporal_detail_20261019.json'))
    commands=parser.add_subparsers(dest='command')
    check=commands.add_parser('cpu-check')
    check.add_argument('--checkpoint',required=True);check.add_argument('--expected-sha',required=True);check.add_argument('--output',required=True)
    for name in ('prepare','check-ready','train','infer'):
        sub=commands.add_parser(name);sub.add_argument('--root',required=True)
        if name=='train':sub.add_argument('--run-id',required=True)
        if name=='infer':
            sub.add_argument('--checkpoint',required=True);sub.add_argument('--stream',required=True)
            sub.add_argument('--frames',type=int,nargs='+',required=True);sub.add_argument('--output',required=True)
            sub.add_argument('--device',choices=['cpu','cuda'],default='cpu')
            sub.add_argument('--allow-heldout',action='store_true');sub.add_argument('--source');sub.add_argument('--calibration')
    args=parser.parse_args()
    if args.command is None:
        parser.print_help();return
    try:
        if args.command=='cpu-check':
            result=w.cpu_check(args.checkpoint,args.expected_sha,args.output)
        elif args.command=='infer':
            w.infer(args.root,args.checkpoint,args.stream,args.frames,args.output,args.device,args.allow_heldout,args.source,args.calibration)
            result=dict(completed=True,training=False)
        else:
            cfg=w.config(args.config)
            if args.command=='prepare':
                r=w.prepare(args.root,cfg);result=dict(prepared=True,contexts=len(r['records']),coverage=r['coverage'],bytes=r['bytes'])
            elif args.command=='check-ready':
                _,_,index,_=w.ready(args.root,cfg)
                result=dict(data_ready=True,contexts=len(index['records']),cuda_available=torch.cuda.is_available(),
                            cuda_gradient_probe='required at explicit train entry',training_started=False)
            else:
                w.train(args.root,cfg,args.run_id);result=dict(completed=True,optimizer_updates=cfg['optimizer_steps'])
        print(json.dumps(result),flush=True)
    except (ValueError,OSError,KeyError,RuntimeError) as error:
        print(json.dumps(dict(ready=False,error=str(error),automatic_retry=False)),file=sys.stderr)
        raise SystemExit(2)


if __name__=='__main__':
    main()
