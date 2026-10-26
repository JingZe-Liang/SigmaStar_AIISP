#!/usr/bin/env python3
"""Default: help. CPU architecture check only; no training/teacher launch."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from company_exposure.burst_detail import workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/exposure_paper_burst_20261022.json'))
    sub = parser.add_subparsers(dest='action')
    check = sub.add_parser('check',help='CPU proof only; zero optimizer updates')
    check.add_argument('--base',required=True)
    check.add_argument('--output',required=True)
    args = parser.parse_args()
    if not args.action:
        parser.print_help()
        return
    cfg = workflow.config(args.config)
    report = workflow.cpu_check(args.base,cfg,args.output)
    print({k:v for k,v in report.items() if k!='config'})


if __name__=='__main__':
    main()
