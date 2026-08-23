import os
import re
import sys
import numpy as np

WIDTH = 1920
HEIGHT = 1080
FRAME_BYTES_16 = WIDTH * HEIGHT * 2  # 16bit raw stream

def parse_filename_params(filename):
    """
    从文件名中解析拍摄参数，支持两种格式：
    1. Shutter=..., SenserG=..., IspG=..., R=..., G=..., B=...
    2. FN=..., US=..., AG=..., DG=..., BV=..., R=..., G=..., B=...
    """
    pattern1 = r'Shutter=(\d+),SenserG=(\d+),IspG=(\d+),R=(\d+),G=(\d+),B=(\d+)'
    m = re.search(pattern1, filename)
    if m:
        return {k: int(v) for k, v in zip(
            ['Shutter', 'SenserG', 'IspG', 'R', 'G', 'B'], m.groups())}
    pattern2 = r'FN=(\d+),US=(\d+),AG=(\d+),DG=(\d+),BV=(-?\d+),R=(\d+),G=(\d+),B=(\d+)'
    m = re.search(pattern2, filename)
    if m:
        return {k: int(v) for k, v in zip(
            ['FN', 'US', 'AG', 'DG', 'BV', 'R', 'G', 'B'], m.groups())}
    return None

def get_file_kind(path):
    """
    根据路径判断文件类型：
    - source: 原始 raw stream (路径中不含 denoised/fused)
    - denoised: 2DNR 输出 (路径中包含 'denoised')
    - fused: 3DNR 输出 (路径中包含 'fused')
    - png: 简易 ISP 结果图片
    """
    p = path.lower()
    if p.endswith('.png'):
        return 'png'
    if 'denoised' in p:
        return 'denoised'
    if 'fused' in p:
        return 'fused'
    return 'source'

def analyze_raw_file(filepath, kind):
    """读取并分析 raw 文件，根据 kind 处理不同位深"""
    file_size = os.path.getsize(filepath)
    # 帧数按 16bit 每像素 2 字节计算（所有 raw 文件都按 uint16 存储）
    num_frames = file_size // FRAME_BYTES_16
    if num_frames <= 0:
        return None

    # 读取前 5 帧
    read_frames = min(5, num_frames)
    total_pixels = read_frames * WIDTH * HEIGHT
    try:
        raw = np.fromfile(filepath, dtype=np.uint16, count=total_pixels)
    except Exception as e:
        print(f"  读取失败: {e}")
        return None
    raw = raw.reshape(read_frames, HEIGHT, WIDTH)

    # 对于 denoised/fused，有效位宽 12bit，直接统计，无需缩放
    # 但要注意它们的值域是 0~4095
    info = {
        'num_frames': num_frames,
        'kind': kind,
    }

    # 全局统计
    info['min'] = raw.min()
    info['max'] = raw.max()
    info['mean'] = raw.mean()
    info['std'] = raw.std()
    info['1pct'] = np.percentile(raw, 1)
    info['5pct'] = np.percentile(raw, 5)
    info['25pct'] = np.percentile(raw, 25)
    info['50pct'] = np.percentile(raw, 50)

    # 相邻帧差分（前两帧）
    if read_frames >= 2:
        diff = raw[0].astype(float) - raw[1].astype(float)
        info['diff_var_over2'] = diff.var() / 2.0
        info['diff_std_over_sqrt2'] = diff.std() / np.sqrt(2)
    else:
        info['diff_var_over2'] = None
        info['diff_std_over_sqrt2'] = None

    # 分通道统计（第一帧）
    frame0 = raw[0]
    R = frame0[0::2, 0::2]
    Gr = frame0[0::2, 1::2]
    Gb = frame0[1::2, 0::2]
    B = frame0[1::2, 1::2]
    ch_stats = {}
    for name, ch in [('R', R), ('Gr', Gr), ('Gb', Gb), ('B', B)]:
        ch_stats[name] = {
            'min': ch.min(),
            'max': ch.max(),
            'mean': ch.mean(),
            'std': ch.std(),
            '1pct': np.percentile(ch, 1),
            '5pct': np.percentile(ch, 5),
        }
    info['channel_stats'] = ch_stats
    return info

def main(root_dir):
    root_dir = os.path.abspath(root_dir)
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.raw') or fname.lower().endswith('.png'):
                all_files.append(os.path.join(dirpath, fname))

    print(f"总共找到 {len(all_files)} 个文件\n")

    # 分类计数
    kinds_count = {'source': 0, 'denoised': 0, 'fused': 0, 'png': 0}
    for fp in all_files:
        kind = get_file_kind(fp)
        kinds_count[kind] += 1
    print("文件类型统计:")
    for k, v in kinds_count.items():
        print(f"  {k}: {v} 个文件")
    print("\n" + "="*100)

    # 逐个分析 raw 文件，png 仅列出
    for i, fp in enumerate(all_files, 1):
        kind = get_file_kind(fp)
        fname = os.path.basename(fp)
        print(f"[{i}/{len(all_files)}] {fp}")
        print(f"  类型: {kind}")

        if kind == 'png':
            print("  (跳过统计 PNG)")
            print("-"*100)
            continue

        params = parse_filename_params(fname)
        if params:
            print(f"  参数: {params}")

        info = analyze_raw_file(fp, kind)
        if info is None:
            print("  无法分析，跳过")
            print("-"*100)
            continue

        print(f"  帧数: {info['num_frames']}")
        print(f"  min={info['min']}, max={info['max']}, mean={info['mean']:.2f}, std={info['std']:.2f}")
        print(f"  1%={info['1pct']:.2f}, 5%={info['5pct']:.2f}, 中位数={info['50pct']:.2f}")
        if info['diff_var_over2'] is not None:
            print(f"  相邻帧差方差/2: {info['diff_var_over2']:.4f}, 等效标准差: {info['diff_std_over_sqrt2']:.4f}")
        else:
            print("  相邻帧差：文件只有1帧，无法计算")

        print("  分通道统计（第一帧）:")
        for ch, st in info['channel_stats'].items():
            print(f"    {ch}: min={st['min']}, max={st['max']}, mean={st['mean']:.2f}, "
                  f"std={st['std']:.2f}, 1%={st['1pct']:.2f}, 5%={st['5pct']:.2f}")
        print("-"*100)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python scan_dataset_v2.py <数据集根目录>")
        sys.exit(1)
    main(sys.argv[1])