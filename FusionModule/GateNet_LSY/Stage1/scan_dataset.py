import os
import re
import sys
import numpy as np

# 固定参数
WIDTH = 1920
HEIGHT = 1080
BIT_DEPTH = 16
BYTES_PER_PIXEL = BIT_DEPTH // 8
FRAME_BYTES = WIDTH * HEIGHT * BYTES_PER_PIXEL

def parse_filename(filename):
    """
    从文件名中提取参数：
    Shutter, SenserG, IspG, R, G, B
    """
    pattern = r'Shutter=(\d+),SenserG=(\d+),IspG=(\d+),R=(\d+),G=(\d+),B=(\d+)'
    m = re.search(pattern, filename)
    if not m:
        return None
    return {
        'Shutter': int(m.group(1)),
        'SenserG': int(m.group(2)),
        'IspG': int(m.group(3)),
        'R': int(m.group(4)),
        'G': int(m.group(5)),
        'B': int(m.group(6)),
    }

def analyze_raw_file(filepath):
    """读取前3帧并返回统计信息，若失败返回None"""
    file_size = os.path.getsize(filepath)
    num_frames = file_size // FRAME_BYTES
    if num_frames <= 0:
        return None

    # 最多读取3帧
    frames_to_read = min(3, num_frames)
    total_pixels = frames_to_read * WIDTH * HEIGHT
    try:
        raw_data = np.fromfile(filepath, dtype=np.uint16, count=total_pixels)
    except Exception as e:
        print(f"  读取失败: {e}")
        return None

    raw_data = raw_data.reshape(frames_to_read, HEIGHT, WIDTH)
    info = {}

    # 全局统计
    info['num_frames'] = num_frames
    info['global_min'] = raw_data.min()
    info['global_max'] = raw_data.max()
    info['global_mean'] = raw_data.mean()
    info['global_std'] = raw_data.std()
    info['global_1pct'] = np.percentile(raw_data, 1)
    info['global_5pct'] = np.percentile(raw_data, 5)

    # 相邻帧差分（如果有≥2帧）
    if frames_to_read >= 2:
        diff = raw_data[0].astype(np.float64) - raw_data[1].astype(np.float64)
        info['diff_var'] = diff.var()
        info['noise_var_approx'] = diff.var() / 2.0
    else:
        info['diff_var'] = None
        info['noise_var_approx'] = None

    # Bayer 四通道统计（仅使用第一帧）
    frame0 = raw_data[0]
    R = frame0[0::2, 0::2]
    Gr = frame0[0::2, 1::2]
    Gb = frame0[1::2, 0::2]
    B = frame0[1::2, 1::2]
    ch_stats = {
        'R': {'mean': R.mean(), 'std': R.std(), 'min': R.min(), 'max': R.max()},
        'Gr': {'mean': Gr.mean(), 'std': Gr.std(), 'min': Gr.min(), 'max': Gr.max()},
        'Gb': {'mean': Gb.mean(), 'std': Gb.std(), 'min': Gb.min(), 'max': Gb.max()},
        'B': {'mean': B.mean(), 'std': B.std(), 'min': B.min(), 'max': B.max()},
    }
    info['channel_stats'] = ch_stats

    return info

def main(root_dir):
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        print(f"目录不存在: {root_dir}")
        return

    raw_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.raw'):
                raw_files.append(os.path.join(dirpath, fname))

    print(f"找到 {len(raw_files)} 个 .raw 文件\n")
    print("=" * 100)

    for idx, filepath in enumerate(raw_files, start=1):
        filename = os.path.basename(filepath)
        print(f"[{idx}/{len(raw_files)}] {filepath}")

        params = parse_filename(filename)
        if params:
            print(f"  参数: Shutter={params['Shutter']}, SenserG={params['SenserG']}, "
                  f"IspG={params['IspG']}, R={params['R']}, G={params['G']}, B={params['B']}")
            # 总增益参考：需要知道基准增益，先打印原始值
            sensor_gain_ratio = params['SenserG'] / 131072.0  # 假设基准为131072
            isp_gain_ratio = params['IspG'] / 8192.0          # 假设基准为8192
            total_gain = sensor_gain_ratio * isp_gain_ratio
            print(f"  增益比(参考): SensorG比={sensor_gain_ratio:.4f}, IspG比={isp_gain_ratio:.4f}, "
                  f"总增益={total_gain:.4f}")
        else:
            print("  未能解析参数")

        info = analyze_raw_file(filepath)
        if info is None:
            print("  无法分析，跳过\n")
            continue

        print(f"  帧数: {info['num_frames']}")
        print(f"  全局统计: min={info['global_min']}, max={info['global_max']}, "
              f"mean={info['global_mean']:.2f}, std={info['global_std']:.2f}")
        print(f"  黑电平参考: 1%分位数={info['global_1pct']:.2f}, 5%分位数={info['global_5pct']:.2f}")
        if info['noise_var_approx'] is not None:
            print(f"  相邻帧差方差/2（粗略噪声方差）: {info['noise_var_approx']:.4f}")
        else:
            print("  相邻帧差方差: 文件只有1帧，无法计算")
        print("  分通道统计（第一帧）:")
        for ch, st in info['channel_stats'].items():
            print(f"    {ch}: mean={st['mean']:.2f}, std={st['std']:.2f}, "
                  f"min={st['min']}, max={st['max']}")
        print("-" * 100)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python scan_dataset.py <数据集目录>")
        sys.exit(1)
    main(sys.argv[1])