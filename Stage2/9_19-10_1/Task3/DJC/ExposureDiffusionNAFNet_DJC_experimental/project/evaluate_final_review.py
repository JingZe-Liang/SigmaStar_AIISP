import gc
import json
import math
import time

import numpy as np
import torch
from PIL import Image

from compare_company_rvi import ROOT, assemble, write_json
from company_exposure.burst_detail.model import ARCHITECTURE, BASE_SHA, load_model, parameter
from company_exposure.burst_detail.workflow import signature
from company_exposure.core import TEACHER_SHA, sha256, signed_signal
from company_exposure.data import read_packed
from company_exposure.workflow import load_teacher, teacher_frame
from prepare_final_review import OUTPUT
from scripts.probe_rvideformer_motion_teacher import positions, self_check
from utils.rgb_metrics import packed_to_linear_rgb

PANELS = [('Noisy center', 'noisy'), ('RViDeformer-L / 8 frames', 'rvideformer'),
          ('Frozen company model / 1 frame', 'frozen_base'), ('V3 final5000 / 5 frames', 'trained_v3')]


def paired_inference(model, signal, shot, span, read, batch_size=4):
    height, width = signal.shape[-2:]
    coordinate = lambda length: sorted(set([*range(0, length - 192 + 1, 128), length - 192]))
    tiles = [(top, left) for top in coordinate(height) for left in coordinate(width)]
    ramp = torch.hann_window(192, periodic=False, device=signal.device).clamp_min(.01)
    weight = (ramp[:, None] * ramp[None, :])[None]
    result = torch.zeros((2, 4, height, width), device=signal.device)
    denominator = torch.zeros((1, height, width), device=signal.device)
    for start in range(0, len(tiles), batch_size):
        group = tiles[start:start + batch_size]
        patches = torch.stack([signal[..., top:top + 192, left:left + 192] for top, left in group])
        count = len(group)
        shots = [shot] * count
        reads = [read] * count
        spans = span[None].expand(count, -1)
        torch.manual_seed(20261022 + start)
        base = model(patches, shots, spans, reads, enabled=False)[-1]
        residual = model.features(patches, parameter(shots, count, signal.device, 'shot'),
                                  parameter(spans, count, signal.device, 'span'),
                                  parameter(reads, count, signal.device, 'read', zero=True))
        candidate = base + residual
        if start == 0:
            torch.manual_seed(20261022)
            reference = model(patches, shots, spans, reads)[-1]
            torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
        prediction = torch.stack([base, candidate]) * span[None, None, :, None, None]
        if not torch.isfinite(prediction).all():
            raise RuntimeError('Non-finite paired prediction')
        for index, (top, left) in enumerate(group):
            result[:, :, top:top + 192, left:left + 192] += prediction[:, index] * weight
            denominator[:, top:top + 192, left:left + 192] += weight
    if not (denominator > 0).all():
        raise RuntimeError('Incomplete paired tile coverage')
    return result / denominator


def gallery(records):
    rows = [dict(stream=record['stream'], frame=record['frame'], role=record['role'],
                 sensor=record['sensor'], source=record['source_relative'], directory=record['directory']) for record in records]
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>5000步最终对比：27条流165帧</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#182230}header{position:sticky;top:0;background:#f5f6f8;padding:12px 0}button,select{font:inherit;padding:8px;max-width:60vw}img{width:100%}h1{font-size:24px}a{color:#175b9e}#source{word-break:break-all}</style>
<h1>5000步最终对比：20条开发流各3帧、7条留出流各15帧</h1>
<p>左至右：Noisy / RViDeformer-L（8帧）/ 不加新分支的公司模型（1帧）/ 加分支的V3 final5000（5帧）。两种公司模型共享完全相同的冻结主干和本次Poisson推理轨迹。</p>
<p>开发流从未用于优化的验证片段取帧；留出流不参与训练、验证选权重或噪声拟合。留出噪声条件从独立标定录像近似迁移，并非厂家标定。不同帧数和训练数据不是控制变量基准；没有干净GT，不作PSNR/SSIM排名。</p>
<p>每条流全部帧及模型共用首个选定Noisy帧的白平衡、ISP和亮度尺度。帧号从0开始，抽帧非连续，不代表连续视频闪烁验收。</p>
<p><a href="通俗训练报告.html">通俗训练报告</a> · <a href="selection.json">固定抽帧计划</a> · <a href="independent_noise_profiles.json">独立噪声标定及局限</a></p>
<header><button onclick="show(selected-1)">上一帧</button> <select id="menu" onchange="show(this.selectedIndex)"></select> <button onclick="show(selected+1)">下一帧</button> <span id="count"></span></header>
<h2 id="title"></h2><p id="source"></p><p id="links"></p><a id="full"><img id="image"></a><h2>相同区域细节</h2><a id="detail"><img id="crop"></a>
<script>const rows=__ROWS__;let selected=0;const menu=document.getElementById('menu');rows.forEach(row=>{const item=document.createElement('option');item.textContent=`[${row.role==='heldout_test'?'留出':'开发/验证'}] ${row.sensor} / ${row.stream} / frame ${row.frame}`;menu.append(item)});
function show(index){if(!rows.length)return;selected=(index+rows.length)%rows.length;menu.selectedIndex=selected;const row=rows[selected];document.getElementById('title').textContent=`${row.sensor} / ${row.stream} / frame ${row.frame}`;document.getElementById('source').textContent=row.source;document.getElementById('count').textContent=`${selected+1} / ${rows.length}（目标165帧）`;document.getElementById('links').replaceChildren();for(const name of ['noisy','rvideformer','frozen_base','trained_v3']){const link=document.createElement('a');link.href=`${row.directory}/${name}.png`;link.textContent=`${name} 原尺寸　`;link.target='_blank';document.getElementById('links').append(link)}for(const [linkId,imageId,file] of [['full','image','comparison.png'],['detail','crop','details.png']]){const url=`${row.directory}/${file}`;document.getElementById(linkId).href=url;document.getElementById(imageId).src=url}}show(0);</script></html>'''
    (OUTPUT / 'gallery.html').write_text(document.replace('__ROWS__', json.dumps(rows, ensure_ascii=True)), encoding='utf-8')


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    self_check()
    plan = json.loads((OUTPUT / 'selection.json').read_text(encoding='utf-8'))
    if plan['cases'] != 165 or sha256(OUTPUT / 'independent_noise_profiles.json') != plan['noise_profiles_sha256']:
        raise RuntimeError('Final review selection/calibration contract mismatch')
    checkpoint = ROOT / 'work_v3/runs/residual_v3_001/final.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['step'] != 5000 or saved['signature'] != signature() or saved['architecture'] != ARCHITECTURE or saved['base_sha256'] != BASE_SHA:
        raise RuntimeError('Final5000 checkpoint unavailable or invalid')
    benchmark_path = ROOT / 'work_v3/rvi_batch_benchmark.json'
    teacher_default = json.loads(benchmark_path.read_text(encoding='utf-8'))['selected_batch']
    if teacher_default not in [2, 4, 8]:
        raise RuntimeError('Invalid measured RVi batch choice')
    teacher_fallback = max(1, teacher_default // 2)
    contract = dict(checkpoint_sha256=sha256(checkpoint), step=5000, base_sha256=BASE_SHA,
                    teacher_sha256=TEACHER_SHA, source_sha256=sha256(__file__),
                    selection_sha256=sha256(OUTPUT / 'selection.json'), seed=20261022,
                    precision='float32', tf32=False, student_tile_batch=4, teacher_tile_batch=teacher_default,
                    batch_benchmark_sha256=sha256(benchmark_path),
                    bounded_oom_fallback=dict(student=1, teacher=teacher_fallback), alpha=1)
    contract_path = OUTPUT / 'inference_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text(encoding='utf-8')) != contract:
        raise RuntimeError('Existing final review contract changed')
    write_json(contract_path, contract)
    student = load_model(ROOT / 'runs/company_mixed_gpu_001/best.pt', device='cuda').eval()
    student.load_adapter(saved['adapter'])
    teacher = load_teacher(ROOT / 'weights/RViDeformer-L_CRVD.pth', 'cuda')
    records = []
    for scene in plan['scenes']:
        stream = scene['stream']
        source = ROOT / stream['path']
        if sha256(source) != stream['sha256']:
            raise RuntimeError('Final review RAW source changed')
        span = (4095 - np.asarray(stream['black_dn12'], dtype=np.float32)).reshape(1, 4, 1, 1)
        span_gpu = torch.from_numpy(span[0, :, 0, 0]).cuda()
        first = signed_signal(read_packed(ROOT, stream, [scene['frames'][0]]), stream['black_dn12'])
        rgb, wb = packed_to_linear_rgb((torch.from_numpy(first) + 16) / 4095, 'GBRG', return_wb=True)
        scale = max(float(np.quantile(rgb.numpy(), .995)), 1e-6)
        scene_records = []
        for frame in scene['frames']:
            relative = f"{stream['id']}/frame_{frame:04d}"
            destination = OUTPUT / relative
            destination.mkdir(parents=True, exist_ok=True)
            provenance = destination / 'provenance.json'
            if provenance.exists():
                record = json.loads(provenance.read_text(encoding='utf-8'))
                if record['contract'] != contract or record['frame'] != frame:
                    raise RuntimeError('Existing frame identity mismatch')
                for name, digest in record['output_sha256'].items():
                    if sha256(destination / name) != digest:
                        raise RuntimeError('Existing final-review PNG corrupt')
                records.append(record)
                scene_records.append(record)
                continue
            started = time.monotonic()
            indices = list(range(frame - 3, frame + 5))
            if scene['role'] == 'development_validation':
                lower, upper = stream['splits']['validation']
                if indices[0] < lower or indices[-1] > upper:
                    raise RuntimeError('Development inference leaves validation interval')
            signal = signed_signal(read_packed(ROOT, stream, indices), stream['black_dn12'])
            observed = torch.from_numpy(signal[1:6]).cuda()
            state = dict(stream=stream['id'], frame=frame, completed=len(records), total=165, phase='paired_student')
            write_json(OUTPUT / 'status.json', state)

            def render(value, name):
                rgb = packed_to_linear_rgb((value[None].cpu() + 16) / 4095, 'GBRG', wb_gains=wb)
                pixels = (rgb[0].permute(1, 2, 0).numpy() / scale).clip(0, 1)
                Image.fromarray((pixels * 255).astype('uint8')).save(destination / f'{name}.png')

            render(observed[2], 'noisy')
            conditions = scene['conditions']
            student_batch = 4
            try:
                pair = paired_inference(student, observed, conditions['shot_gain_dn12_per_e'], span_gpu, conditions['read_variance_dn12_sq'])
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                student_batch = 1
                pair = paired_inference(student, observed, conditions['shot_gain_dn12_per_e'], span_gpu, conditions['read_variance_dn12_sq'], batch_size=1)
            render(pair[0], 'frozen_base')
            render(pair[1], 'trained_v3')
            branch_rms = float((pair[1] - pair[0]).square().mean().sqrt())
            del pair
            counter = 0

            def tracked(patch):
                nonlocal counter
                counter += 1
                if counter == 1 or counter % 8 == 0:
                    state.update(phase='rvideformer', tile_batch=counter)
                    write_json(OUTPUT / 'status.json', state)
                return teacher(patch)

            teacher_batch = teacher_default
            try:
                prediction, preclip = teacher_frame(tracked, signal / span, 1., tile_batch=teacher_default, device='cuda', amp=False)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                teacher_batch = teacher_fallback
                prediction, preclip = teacher_frame(tracked, signal / span, 1., tile_batch=teacher_fallback, device='cuda', amp=False)
            render(torch.from_numpy(prediction * span[0]), 'rvideformer')
            title = f"{stream['sensor']} | {stream['id']} | frame {frame} | {scene['role']}"
            panels = [(label, destination / f'{name}.png') for label, name in PANELS]
            assemble(panels, title, destination / 'comparison.png')
            width, height = stream['raw_spec']['width'], stream['raw_spec']['height']
            crop = (width // 2 - 160, height // 2 - 160, width // 2 + 160, height // 2 + 160)
            assemble(panels, title + ' | same center / 2x pixels', destination / 'details.png', crop=crop)
            files = [name + '.png' for label, name in PANELS] + ['comparison.png', 'details.png']
            record = dict(stream=stream['id'], sensor=stream['sensor'], source_relative=stream['source_relative'],
                          source_sha256=stream['sha256'], role=scene['role'], frame=frame, directory=relative,
                          contract=contract, conditions=conditions, wb=wb, scale=scale,
                          display_reference_frame=scene['frames'][0], input_frames=dict(company=[frame],
                          v3=list(range(frame - 2, frame + 3)), rvi=indices), crop_xyxy=crop,
                          student_tile_batch=student_batch, teacher_tile_batch=teacher_batch,
                          paired_backbone_shared=True, paired_first_tile_exact_self_check=True,
                          branch_change_rms_dn12=branch_rms, rvi_preclip_fraction=preclip,
                          seconds=time.monotonic() - started, optimizer_updates=0, clean_gt=False,
                          output_sha256={name: sha256(destination / name) for name in files})
            write_json(provenance, record)
            records.append(record)
            scene_records.append(record)
            gallery(records)
            print(json.dumps(dict(completed=len(records), total=165, stream=stream['id'], frame=frame,
                                  seconds=record['seconds'])), flush=True)
        images = []
        for record in scene_records:
            with Image.open(OUTPUT / record['directory'] / 'comparison.png') as image:
                images.append(image.resize((1920, round(image.height * 1920 / image.width)), Image.Resampling.LANCZOS))
        board = Image.new('RGB', (1920, sum(image.height for image in images)), 'white')
        top = 0
        for image in images:
            board.paste(image, (0, top))
            top += image.height
        board.save(OUTPUT / stream['id'] / 'overview.png')
    if len(records) != 165 or sum(record['role'] == 'heldout_test' for record in records) != 105:
        raise RuntimeError('Final review coverage mismatch')
    for record in records:
        for name, digest in record['output_sha256'].items():
            if sha256(OUTPUT / record['directory'] / name) != digest:
                raise RuntimeError('Final full PNG audit failed')
    gallery(records)
    write_json(OUTPUT / 'complete.json', dict(scenes=27, cases=165, development=60, heldout=105,
               primary_pngs=990, all_hashes_verified=True, optimizer_updates=0,
               final_step=5000, contract=contract, quality_proven=False))
    write_json(OUTPUT / 'status.json', dict(phase='complete', cases=165, final_step=5000))


if __name__ == '__main__':
    main()
