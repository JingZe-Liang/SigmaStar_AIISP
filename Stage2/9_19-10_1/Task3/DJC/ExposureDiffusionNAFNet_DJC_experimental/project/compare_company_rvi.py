import argparse
import html
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from company_adaptation_v2 import infer_tiles
from company_exposure.burst_detail import pilot
from company_exposure.burst_detail.model import ARCHITECTURE, BASE_SHA, load_model
from company_exposure.burst_detail.workflow import signature
from company_exposure.core import signed_signal, sha256, TEACHER_SHA
from company_exposure.data import read_packed
from company_exposure.workflow import load_teacher, teacher_frame
from scripts.probe_rvideformer_motion_teacher import positions, self_check
from utils.rgb_metrics import packed_to_linear_rgb

PANELS = [('Noisy center', 'noisy'), ('Company Exposure / 1 frame', 'frozen_base'),
          ('V3 best 400 / 5 frames', 'residual_v3'), ('RViDeformer-L / 8 frames', 'rvideformer')]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def assemble(panels, title, destination, crop=None, panel_width=640):
    resized = []
    for label, path in panels:
        with Image.open(path) as source:
            image = source.convert('RGB')
            if crop is not None:
                image = image.crop(crop)
            height = round(image.height * panel_width / image.width)
            image = image.resize((panel_width, height),
                                 Image.Resampling.NEAREST if crop else Image.Resampling.LANCZOS)
            resized.append((label, image))
    panel_height = max(image.height for label, image in resized)
    sheet = Image.new('RGB', (4 * panel_width, panel_height + 96), 'white')
    drawing = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=23)
    drawing.text((12, 8), title, fill='black', font=font)
    for index, (label, image) in enumerate(resized):
        drawing.text((index * panel_width + 12, 52), label, fill='black', font=font)
        sheet.paste(image, (index * panel_width, 96))
    sheet.save(destination)


def gallery(output, records, expected):
    rows = []
    for record in records:
        relative = record['stream']
        rows.append(dict(id=record['stream'], sensor=record['sensor'], frame=record['frame'],
                         source=record['source_relative'], directory=relative))
    payload = json.dumps(rows, ensure_ascii=True).replace('</', '<\\/')
    labels = json.dumps(PANELS)
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>公司 RAW / RViDeformer 对比</title><style>
body{margin:24px;font:16px system-ui;background:#f5f6f8;color:#17202a}h1{font-size:25px}
select,button{font:inherit;padding:9px;margin-right:8px}header{position:sticky;top:0;background:#f5f6f8;padding:8px 0;z-index:2}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{background:white;padding:10px;border-radius:8px}
img{width:100%;display:block}#source{word-break:break-all;font-size:13px;color:#566573}a{color:#175b9e}
@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style><h1>公司 RAW 场景：同帧四列对比</h1>
<p>每条 development 流选择验证区间中部一个合法中心帧。各模型共用该帧的白平衡、ISP 和亮度尺度。点击图像查看原尺寸。</p>
<p>输入帧数不同：公司原模型 1 帧、V3 5 帧、官方 RViDeformer-L 8 帧。RVi 为完整帧推理；没有干净 GT，不给出 PSNR 排名。</p>
<header><button onclick="move(-1)">上一场景</button><select id="scenes" onchange="show(this.selectedIndex)"></select><button onclick="move(1)">下一场景</button><span id="count"></span></header>
<h2 id="title"></h2><p id="source"></p><div class="grid" id="grid"></div>
<p><a id="full">四列全图</a> · <a id="detail">相同区域细节放大</a> · <a id="provenance">来源与参数</a></p>
<script>const rows=__ROWS__;const panels=__PANELS__;let selected=0;
const menu=document.getElementById('scenes');rows.forEach((row,index)=>{const option=document.createElement('option');option.textContent=`${index+1}. ${row.sensor} / ${row.id}`;menu.appendChild(option)});
function show(index){if(!rows.length)return;selected=(index+rows.length)%rows.length;menu.selectedIndex=selected;const row=rows[selected];document.getElementById('title').textContent=`${row.sensor} / ${row.id} / frame ${row.frame}`;document.getElementById('source').textContent=row.source;document.getElementById('count').textContent=`已完成 ${rows.length}/__EXPECTED__ 场景`;
const grid=document.getElementById('grid');grid.replaceChildren();panels.forEach(([label,name])=>{const card=document.createElement('div');card.className='card';const heading=document.createElement('h3');heading.textContent=label;const link=document.createElement('a');link.href=`${row.directory}/${name}.png`;link.target='_blank';const image=document.createElement('img');image.src=link.href;image.alt=label;link.appendChild(image);card.append(heading,link);grid.appendChild(card)});for(const [name,file] of [['full','comparison.png'],['detail','details.png'],['provenance','provenance.json']])document.getElementById(name).href=`${row.directory}/${file}`;}
function move(delta){show(selected+delta)}show(0);</script></html>'''
    document = document.replace('__ROWS__', payload).replace('__PANELS__', labels).replace('__EXPECTED__', str(expected))
    (output / 'gallery.html').write_text(document, encoding='utf-8')
    report = ['# 公司 RAW 同帧四列对比', '', f'已完成 {len(records)}/{expected} 条 development 流。', '',
              '四列依次为 Noisy、公司冻结 ExposureDiffusion、V3 best（第 400 次更新）、官方 RViDeformer-L CRVD。',
              '统一中心帧、ISP、白平衡、显示尺度；输入分别为中心 1 帧、5 帧、8 帧。',
              'RVi 输出来自完整帧推理，不使用稀疏运动标签代替整图。没有干净 GT，不提供 PSNR/SSIM 排名。', '',
              '[打开交互图集](gallery.html)', '']
    for record in records:
        relative = record['stream']
        report.extend([f"## {record['sensor']} / {relative} / frame {record['frame']}", '',
                       f"来源：`{record['source_relative']}`", '',
                       f'![四列全图]({relative}/comparison.png)', '',
                       f'[原尺寸图与参数]({relative}/provenance.json) · [细节放大]({relative}/details.png)', ''])
    (output / 'COMPARISON_REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    for sensor in sorted({record['sensor'] for record in records}):
        images = []
        for record in records:
            if record['sensor'] == sensor:
                with Image.open(output / record['stream'] / 'comparison.png') as image:
                    images.append(image.resize((1920, round(image.height * 1920 / image.width)), Image.Resampling.LANCZOS))
        board = Image.new('RGB', (1920, sum(image.height for image in images)), 'white')
        row_top = 0
        for image in images:
            board.paste(image, (0, row_top))
            row_top += image.height
        board.save(output / f'overview_{sensor}.png')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'work_v3/comparison_rvi_20261023')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise RuntimeError('Existing comparison protected; use --resume after inspecting its state')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    self_check()
    configuration = pilot.settings(ROOT / 'code/configs/exposure_paper_burst_server_20261022.json')
    pilot.ready(ROOT, ROOT / 'work_v3', configuration)
    manifest, streams, labels, groups = pilot.inventory(ROOT)
    branch_path = ROOT / 'work_v3/runs/residual_v3_001/best.pt'
    saved = torch.load(branch_path, map_location='cpu', weights_only=True)
    if saved['signature'] != signature() or saved['architecture'] != ARCHITECTURE or saved['base_sha256'] != BASE_SHA:
        raise RuntimeError('V3 checkpoint/source mismatch')
    if saved['step'] != 400 or saved['contract']['center_only']:
        raise RuntimeError('Expected frozen best-400 multiframe branch')
    plan = [dict(stream=key, sensor=stream['sensor'], frame=sorted(groups['validation'][key], key=lambda item: item['frame'])[len(groups['validation'][key]) // 2]['frame'],
                 role='validation', source_sha256=stream['sha256'], source_relative=stream['source_relative'])
            for key, stream in sorted(streams.items())]
    contract = dict(architecture=ARCHITECTURE, branch_sha256=sha256(branch_path), base_sha256=BASE_SHA,
                    teacher_sha256=TEACHER_SHA, seed=20261022, precision='float32', tf32=False,
                    teacher_tile_batch=4, selection=plan)
    contract_path = output / 'selection.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError('Comparison contract changed; existing results protected')
    write_json(contract_path, contract)
    student = load_model(ROOT / configuration['base_checkpoint'], device='cuda').eval()
    student.load_adapter(saved['adapter'])
    teacher = load_teacher(ROOT / 'weights/RViDeformer-L_CRVD.pth', 'cuda')
    completed = []
    for index, item in enumerate(plan):
        destination = output / item['stream']
        provenance_path = destination / 'provenance.json'
        if provenance_path.exists():
            record = json.loads(provenance_path.read_text())
            if record['contract'] != contract or record['frame'] != item['frame']:
                raise RuntimeError('Existing scene provenance mismatch')
            for filename, expected in record['output_sha256'].items():
                if sha256(destination / filename) != expected:
                    raise RuntimeError('Existing scene output checksum mismatch')
            completed.append(record)
            continue
        destination.mkdir(exist_ok=True)
        started = time.monotonic()
        state = dict(scene=index + 1, total=len(plan), stream=item['stream'], frame=item['frame'], phase='company_models')
        write_json(output / 'status.json', state)
        stream = streams[item['stream']]
        frame = item['frame']
        indices = list(range(frame - 3, frame + 5))
        lower, upper = stream['splits']['validation']
        if indices[0] < lower or indices[-1] > upper:
            raise RuntimeError('RVi window leaves validation partition')
        signal = signed_signal(read_packed(ROOT, stream, indices), stream['black_dn12'])
        span = (4095 - np.asarray(stream['black_dn12'], dtype=np.float32)).reshape(1, 4, 1, 1)
        observed = torch.from_numpy(signal[1:6]).to('cuda')
        span_gpu = torch.from_numpy(span[0, :, 0, 0]).to('cuda')
        rgb, white_balance = packed_to_linear_rgb((observed[2:3].cpu() + 16) / 4095, 'GBRG', return_wb=True)
        scale = max(float(np.quantile(rgb.numpy(), .995)), 1e-6)

        def render(value, name):
            rgb = packed_to_linear_rgb((value[None].cpu() + 16) / 4095, 'GBRG', wb_gains=white_balance)
            pixels = (rgb[0].permute(1, 2, 0).numpy() / scale).clip(0, 1)
            Image.fromarray((pixels * 255).astype('uint8')).save(destination / (name + '.png'))

        render(observed[2], 'noisy')
        for name, enabled in [('frozen_base', False), ('residual_v3', True)]:
            def predict(patch):
                torch.manual_seed(contract['seed'])
                return student(patch[None], [stream['shot_gain_dn12_per_e']], span_gpu[None],
                               [stream['read_variance_dn12_sq']], enabled=enabled)[-1][0] * span_gpu[:, None, None]
            render(infer_tiles(observed, predict, tile=192, stride=128), name)
        height, width = signal.shape[-2:]
        batches = math.ceil(len(positions(height)) * len(positions(width)) / 4)
        tile_counter = 0

        def tracked_teacher(patch):
            nonlocal tile_counter
            tile_counter += 1
            if tile_counter == 1 or tile_counter % 8 == 0 or tile_counter == batches:
                state.update(phase='rvideformer', tile_batch=tile_counter, tile_batches=batches,
                             elapsed_seconds=round(time.monotonic() - started, 2))
                write_json(output / 'status.json', state)
            return teacher(patch)

        prediction, preclip = teacher_frame(tracked_teacher, signal / span, 1., tile_batch=4, device='cuda', amp=False)
        render(torch.from_numpy(prediction * span[0]), 'rvideformer')
        panels = [(label, destination / (name + '.png')) for label, name in PANELS]
        title = f"{stream['sensor']} | {item['stream']} | frame {frame} | validation"
        assemble(panels, title, destination / 'comparison.png')
        raw_height, raw_width = stream['raw_spec']['height'], stream['raw_spec']['width']
        crop = (raw_width // 2 - 160, raw_height // 2 - 160, raw_width // 2 + 160, raw_height // 2 + 160)
        assemble(panels, title + ' | same center crop / 2x pixels', destination / 'details.png', crop=crop)
        files = [name + '.png' for label, name in PANELS] + ['comparison.png', 'details.png']
        record = dict(item, contract=contract, source=str(ROOT / stream['path']),
                      input_frames=dict(company=[frame], v3=list(range(frame - 2, frame + 3)), rvi=indices),
                      black_dn12=stream['black_dn12'], span_dn12=span.flatten().tolist(),
                      wb=white_balance, scale=scale, detail_crop_xyxy=crop,
                      rvi_normalization='per canonical channel span, clip to [0,1], lossless channel adapter',
                      rvi_tiling=dict(tile=64, stride=56, batch=4, full_frame=True, sparse_mask=False),
                      rvi_preclip_fraction=preclip, seconds=time.monotonic() - started,
                      output_sha256={name: sha256(destination / name) for name in files},
                      clean_ground_truth=False, training_started=False, optimizer_updates=0)
        write_json(provenance_path, record)
        completed.append(record)
        gallery(output, completed, len(plan))
        state.update(phase='scene_complete', completed=len(completed), seconds=record['seconds'])
        write_json(output / 'status.json', state)
        print(json.dumps(state), flush=True)
    gallery(output, completed, len(plan))
    write_json(output / 'complete.json', dict(scenes=len(completed), streams=[record['stream'] for record in completed],
                                            training_started=False, optimizer_updates=0, clean_ground_truth=False))
    write_json(output / 'status.json', dict(phase='complete', scenes=len(completed)))


if __name__ == '__main__':
    main()
