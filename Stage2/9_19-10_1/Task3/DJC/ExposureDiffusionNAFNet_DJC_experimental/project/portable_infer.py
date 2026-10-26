import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from compare_company_rvi import assemble
from evaluate_final_review import paired_inference
from company_exposure.burst_detail.model import load_model
from company_exposure.core import sha256, signed_signal
from company_exposure.data import read_packed
from company_exposure.workflow import load_teacher, teacher_frame
from utils.rgb_metrics import packed_to_linear_rgb


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description='Reproduce one packaged four-column comparison without fitting test parameters.')
    parser.add_argument('--stream', required=True)
    parser.add_argument('--frame', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'work_v3/runs/residual_v3_001/best.pt')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Existing inference output protected')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError('CUDA required')
    review = ROOT.parent / 'results/all165'
    plan = json.loads((review / 'selection.json').read_text(encoding='utf-8'))
    scene = next(scene for scene in plan['scenes'] if scene['stream']['id'] == args.stream)
    if args.frame not in scene['frames']:
        raise ValueError('Use an existing documented frame; this tool does not invent new test noise conditions')
    streams = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))['streams']
    stream = next(stream for stream in streams if stream['id'] == args.stream)
    if sha256(ROOT / stream['path']) != stream['sha256']:
        raise ValueError('RAW hash mismatch')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['step'] not in [2300, 5000]:
        raise ValueError('Expected packaged best2300 or final5000')
    model = load_model(ROOT / 'runs/company_mixed_gpu_001/best.pt', device='cuda').eval()
    model.load_adapter(checkpoint['adapter'])
    teacher = load_teacher(ROOT / 'weights/RViDeformer-L_CRVD.pth', 'cuda')
    original = json.loads((review / args.stream / f'frame_{args.frame:04d}/provenance.json').read_text(encoding='utf-8'))
    conditions = original['conditions']
    span = (4095 - np.asarray(stream['black_dn12'], dtype=np.float32)).reshape(1, 4, 1, 1)
    signal = signed_signal(read_packed(ROOT, stream, list(range(args.frame - 3, args.frame + 5))), stream['black_dn12'])
    observed = torch.from_numpy(signal[1:6]).cuda()
    pair = paired_inference(model, observed, conditions['shot_gain_dn12_per_e'],
                            torch.from_numpy(span[0, :, 0, 0]).cuda(),
                            conditions['read_variance_dn12_sq'], batch_size=original['student_tile_batch'])
    prediction, _ = teacher_frame(teacher, signal / span, 1., tile_batch=original['teacher_tile_batch'], device='cuda', amp=False)
    args.output.mkdir(parents=True)
    values = [('noisy', observed[2]), ('rvideformer', torch.from_numpy(prediction * span[0])),
              ('frozen_base', pair[0]), ('branch', pair[1])]
    for name, value in values:
        rgb = packed_to_linear_rgb((value[None].cpu() + 16) / 4095, 'GBRG', wb_gains=original['wb'])
        pixels = ((rgb[0].permute(1, 2, 0).numpy() / original['scale']).clip(0, 1) * 255).astype('uint8')
        Image.fromarray(pixels).save(args.output / f'{name}.png')
    labels = ['Noisy', 'RViDeformer / 8 frames', 'Frozen company / 1 frame', f'Branch step{checkpoint["step"]} / 5 frames']
    panels = [(label, args.output / f'{name}.png') for label, (name, _) in zip(labels, values)]
    assemble(panels, f'{args.stream} / frame {args.frame}', args.output / 'comparison.png')
    (args.output / 'provenance.json').write_text(json.dumps(dict(
        stream=args.stream, frame=args.frame, branch_step=checkpoint['step'],
        checkpoint_sha256=sha256(args.checkpoint), optimizer_updates=0,
        note='Same packaged RAW, display, conditions and tile batch; numerical identity can differ across GPU/software versions.',
        output_sha256={name + '.png': sha256(args.output / f'{name}.png') for name, _ in values}), indent=2), encoding='utf-8')
    print(f'Completed: {args.output}')


if __name__ == '__main__':
    main()
