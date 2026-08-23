import os
import re
import sys
import numpy as np

# 固定参数
WIDTH = 1920
HEIGHT = 1080
FRAME_BYTES = WIDTH * HEIGHT * 2  # 16bit

def parse_filename(filename):
    pattern = r'Shutter=(\d+),SenserG=(\d+),IspG=(\d+),R=(\d+),G=(\d+),B=(\d+)'
    m = re.search(pattern, filename)
    if m:
        return {k: int(v) for k, v in zip(
            ['Shutter', 'SenserG', 'IspG', 'R', 'G', 'B'], m.groups())}
    # 兼容校准数据文件名
    pattern2 = r'FN=(\d+),US=(\d+),AG=(\d+),DG=(\d+),BV=(-?\d+),R=(\d+),G=(\d+),B=(\d+)'
    m2 = re.search(pattern2, filename)
    if m2:
        return {k: int(v) for k, v in zip(
            ['FN', 'US', 'AG', 'DG', 'BV', 'R', 'G', 'B'], m2.groups())}
    return None

def get_scene_type(path):
    p = path.lower()
    if 'shdarkroom' in p:
        return 'darkroom'
    elif 'calibrationdata' in p and 'black' in p:
        return 'black_calib'
    else:
        return 'normal'

def analyze_raw(filepath, scene_type):
    file_size = os.path.getsize(filepath)
    n_frames = file_size // FRAME_BYTES
    if n_frames <= 0:
        return None
    # 读取前5帧
    n_read = min(5, n_frames)
    raw = np.fromfile(filepath, dtype=np.uint16, count=n_read*WIDTH*HEIGHT)
    raw = raw.reshape(n_read, HEIGHT, WIDTH)

    info = {'n_frames': n_frames, 'scene': scene_type}
    info['min'] = raw.min()
    info['max'] = raw.max()
    info['mean'] = raw.mean()
    info['std'] = raw.std()
    info['1pct'] = np.percentile(raw, 1)
    info['5pct'] = np.percentile(raw, 5)

    # 相邻帧差（仅对暗室有意义）
    if n_read >= 2:
        diff = raw[0].astype(float) - raw[1].astype(float)
        info['diff_var_over2'] = diff.var() / 2.0
    else:
        info['diff_var_over2'] = None

    # 暗室和黑帧：逐像素最小值估计黑电平
    if scene_type in ['darkroom', 'black_calib']:
        # 读取所有帧，但暗室数据帧数可能多，限制最多20帧
        n_use = min(20, n_frames)
        raw_all = np.fromfile(filepath, dtype=np.uint16, count=n_use*WIDTH*HEIGHT)
        raw_all = raw_all.reshape(n_use, HEIGHT, WIDTH)
        min_map = raw_all.min(axis=0)
        # 分通道黑电平
        R = min_map[0::2, 0::2]
        Gr = min_map[0::2, 1::2]
        Gb = min_map[1::2, 0::2]
        B = min_map[1::2, 1::2]
        info['black_level_channels'] = {
            'R': np.percentile(R, 5),
            'Gr': np.percentile(Gr, 5),
            'Gb': np.percentile(Gb, 5),
            'B': np.percentile(B, 5),
        }
        # 简单全局黑电平
        info['black_level_global'] = np.percentile(min_map, 5)
    return info

def main(root_dir):
    root_dir = os.path.abspath(root_dir)
    raw_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.raw'):
                raw_files.append(os.path.join(dirpath, fname))

    print(f"总共找到 {len(raw_files)} 个 raw 文件\n")
    for i, fp in enumerate(raw_files, 1):
        fname = os.path.basename(fp)
        scene = get_scene_type(fp)
        print(f"[{i}/{len(raw_files)}] {fp}")
        print(f"  场景类型: {scene}")
        params = parse_filename(fname)
        if params:
            print(f"  参数: {params}")
        info = analyze_raw(fp, scene)
        if info:
            print(f"  帧数: {info['n_frames']}")
            print(f"  min={info['min']}, max={info['max']}, mean={info['mean']:.2f}, std={info['std']:.2f}")
            print(f"  1%={info['1pct']:.2f}, 5%={info['5pct']:.2f}")
            if info['diff_var_over2'] is not None:
                print(f"  相邻帧差方差/2: {info['diff_var_over2']:.4f}")
            if 'black_level_global' in info:
                print(f"  黑电平估计(全局5%): {info['black_level_global']:.2f}")
                print(f"  分通道黑电平: {info['black_level_channels']}")
        print("-"*80)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python scan_all_dataset.py <数据集根目录>")
        sys.exit(1)
    main(sys.argv[1])