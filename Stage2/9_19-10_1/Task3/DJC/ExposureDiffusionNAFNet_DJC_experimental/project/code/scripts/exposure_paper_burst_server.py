#!/usr/bin/env python3
"""No arguments: help. prepare/check-ready are CPU-only; train is explicit."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from company_exposure.burst_detail import pilot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,help='Existing company project/data root')
    p.add_argument('--deploy',required=True,help='New isolated branch deployment directory')
    p.add_argument('--config',default=str(ROOT/'configs/exposure_paper_burst_server_20261022.json'))
    sub=p.add_subparsers(dest='action')
    sub.add_parser('prepare',help='Hash data/weights/labels, test all 40 stream partitions; CPU only')
    sub.add_parser('check-ready',help='Read-only fast stamp/source/code proof; CPU only')
    run=sub.add_parser('train',help='Manual bounded pilot; stops after whole-image review')
    run.add_argument('--run-id',required=True)
    a=p.parse_args()
    if not a.action:p.print_help();return
    cfg=pilot.settings(a.config)
    if a.action=='prepare':
        r=pilot.prepare(a.root,a.deploy,cfg)
    elif a.action=='check-ready':
        r=pilot.ready(a.root,a.deploy,cfg)
    else:
        pilot.run(a.root,a.deploy,cfg,a.run_id);return
    print({k:r.get(k) for k in ['cpu_ready','gpu_required_to_start','checked_streams','checked_partitions','training_started','optimizer_updates']})


if __name__=='__main__':main()
