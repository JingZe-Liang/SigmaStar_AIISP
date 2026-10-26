import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Portable delivery entry; never starts training implicitly.')
    parser.add_argument('action', choices=['test', 'prepare', 'check-ready', 'evaluate', 'train', 'infer'])
    parser.add_argument('--run-id', default='reproduction_v3_001')
    parser.add_argument('--updates', type=int, default=5000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stream')
    parser.add_argument('--frame', type=int)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.output:
        args.output = args.output.resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                      MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONIOENCODING='utf-8')
    work = ROOT / 'work_delivery'
    for name in ['TMP', 'TEMP', 'TMPDIR', 'MPLCONFIGDIR', 'TORCH_HOME', 'XDG_CACHE_HOME']:
        target = work / 'cache' / name.lower()
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    runner = ROOT / 'code/scripts/exposure_output_residual_local.py'
    base = [sys.executable, str(runner), '--root', str(ROOT), '--work', str(work)]
    if args.action == 'test':
        command = [sys.executable, '-m', 'pytest', '-q', str(ROOT / 'code/tests')]
    elif args.action in ['prepare', 'check-ready']:
        command = base + [args.action]
    elif args.action == 'evaluate':
        output = args.output or work / 'preview_best'
        checkpoint = args.checkpoint or ROOT / 'work_v3/runs/residual_v3_001/best.pt'
        command = base + ['evaluate', '--checkpoint', str(checkpoint), '--output', str(output), '--alpha', '1']
    elif args.action == 'train':
        if (work / 'runs' / args.run_id).exists():
            raise ValueError('Existing training run protected; choose a new run-id')
        command = base + ['train', '--run-id', args.run_id, '--updates', str(args.updates),
                          '--validate-every', '100', '--batch', '2', '--accumulate', '4', '--lr', '0.0001']
    else:
        if not args.stream or args.frame is None:
            parser.error('infer requires --stream and --frame from the packaged selection')
        command = [sys.executable, str(ROOT / 'portable_infer.py'), '--stream', args.stream,
                   '--frame', str(args.frame), '--output', str(args.output or work / 'inference')]
        if args.checkpoint:
            command += ['--checkpoint', str(args.checkpoint)]
    print(json.dumps({'command': command, 'new_work': str(work)}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
