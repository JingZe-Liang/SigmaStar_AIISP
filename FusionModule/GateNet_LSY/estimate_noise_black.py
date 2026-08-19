import os
import sys
import numpy as np

# 固定参数
WIDTH = 1920
HEIGHT = 1080
BIT_DEPTH = 16
FRAME_BYTES = WIDTH * HEIGHT * 2
NUM_FRAMES = 198  # 根据扫描结果已知
ROOT = "D:/University/Fusion/Phase Final/Phase2/DATASET/Sigmastar_7_30/mis20s1_outdoor/512x"

def load_raw_frames(filepath, num_frames=NUM_FRAMES):
    """读取所有帧到内存，形状 (N,H,W)"""
    file_size = os.path.getsize(filepath)
    n = file_size // FRAME_BYTES
    n = min(n, num_frames)
    raw = np.fromfile(filepath, dtype=np.uint16, count=n*WIDTH*HEIGHT)
    raw = raw.reshape(n, HEIGHT, WIDTH)
    return raw

def estimate_black_level(frames):
    """逐像素取所有帧的最小值，然后取低百分位数估计黑电平"""
    min_map = frames.min(axis=0)  # (H,W)
    # 分通道统计
    R = min_map[0::2, 0::2]
    Gr = min_map[0::2, 1::2]
    Gb = min_map[1::2, 0::2]
    B = min_map[1::2, 1::2]
    print("黑电平估计（逐像素最小值分布）:")
    for name, ch in [("R", R), ("Gr", Gr), ("Gb", Gb), ("B", B)]:
        print(f"  {name}: min={ch.min()}, 1%={np.percentile(ch, 1):.2f}, "
              f"5%={np.percentile(ch, 5):.2f}, median={np.median(ch):.2f}, mean={ch.mean():.2f}")
    # 全局低百分位
    print(f"全局最小值分位: 1%={np.percentile(min_map,1):.2f}, 5%={np.percentile(min_map,5):.2f}")
    # 建议黑电平取各通道5%分位数的中位数？这里仅报告
    return min_map

def estimate_noise_curve(frames, sample_step=4):
    """
    基于空间平坦区域估计噪声方差。
    对每帧计算局部3x3梯度和局部均值，选择低梯度像素作为平坦区域。
    按均值分桶，计算每桶内的方差，最后拟合 Var = a*mean + b。
    """
    from scipy.ndimage import uniform_filter
    import matplotlib.pyplot as plt

    # 由于数据量大，采样处理
    H, W = frames.shape[1:]
    R_all = frames[:, 0::2, 0::2]
    Gr_all = frames[:, 0::2, 1::2]
    Gb_all = frames[:, 1::2, 0::2]
    B_all = frames[:, 1::2, 1::2]

    # 每帧取部分像素，比如每4步取1个
    # 为简化，只取每帧的中心区域或随机采样？
    # 这里我们用步长采样
    sample_idx_h = np.arange(0, H//2, sample_step)
    sample_idx_w = np.arange(0, W//2, sample_step)
    # 构造网格
    hs, ws = np.meshgrid(sample_idx_h, sample_idx_w, indexing='ij')

    channels = {
        'R': R_all,
        'Gr': Gr_all,
        'Gb': Gb_all,
        'B': B_all,
    }

    for ch_name, ch_data in channels.items():
        print(f"\n通道 {ch_name} 噪声估计:")
        means_list = []
        vars_list = []
        for f in range(ch_data.shape[0]):
            frame = ch_data[f].astype(np.float64)  # (H/2, W/2)
            # 局部均值
            local_mean = uniform_filter(frame, size=3, mode='reflect')
            # 局部梯度近似：中心与上下左右差
            grad_x = np.abs(np.diff(frame, axis=1, prepend=frame[:, :1]))
            grad_y = np.abs(np.diff(frame, axis=0, prepend=frame[:1, :]))
            grad = grad_x + grad_y
            # 平坦区域：梯度小于某一阈值，例如小于全局梯度的10%分位
            grad_thresh = np.percentile(grad, 10)
            flat_mask = grad < grad_thresh
            # 排除饱和
            flat_mask &= (frame < 60000) & (frame > 1000)
            if flat_mask.sum() < 100:
                continue
            flat_vals = frame[flat_mask]
            # 用局部均值作为亮度近似
            flat_means = local_mean[flat_mask]
            # 计算这些平坦区域的局部方差（用3x3窗口）
            local_var = uniform_filter((frame - local_mean)**2, size=3, mode='reflect')
            flat_vars = local_var[flat_mask]
            means_list.append(flat_means)
            vars_list.append(flat_vars)

        if not means_list:
            print("  平坦区域不足")
            continue

        means = np.concatenate(means_list)
        vars_ = np.concatenate(vars_list)

        # 按均值分桶，例如每500一个桶
        bins = np.arange(0, 60001, 500)
        bin_centers = []
        bin_vars = []
        for i in range(len(bins)-1):
            mask = (means >= bins[i]) & (means < bins[i+1])
            if mask.sum() > 50:
                bin_centers.append(means[mask].mean())
                bin_vars.append(vars_[mask].mean())

        if len(bin_centers) < 3:
            print("  分桶不足，无法拟合")
            continue

        bin_centers = np.array(bin_centers)
        bin_vars = np.array(bin_vars)
        # 拟合线性：vars = a * centers + b
        A = np.vstack([bin_centers, np.ones_like(bin_centers)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, bin_vars, rcond=None)
        a, b = coeffs
        print(f"  拟合结果: Var = {a:.6f} * mean + {b:.6f}")
        print(f"  分桶数量: {len(bin_centers)}")

        # 可选保存散点图
        # plt.scatter(bin_centers, bin_vars, s=10)
        # plt.plot(bin_centers, a*bin_centers+b, 'r')
        # plt.title(f'{ch_name} noise curve')
        # plt.savefig(f'noise_{ch_name}.png')
        # plt.close()

def main():
    filepath = os.path.join(ROOT, "raw_stream_1920x1080_16bit@RG_[Shutter=79999,SenserG=131072,IspG=3957,R=1837,G=1024,B=2283].raw")
    print("正在读取所有帧...")
    frames = load_raw_frames(filepath)
    print(f"读取完成，形状: {frames.shape}")
    print("="*60)
    print("估计黑电平...")
    bl_map = estimate_black_level(frames)
    print("="*60)
    print("估计空间噪声曲线...")
    estimate_noise_curve(frames, sample_step=4)

if __name__ == "__main__":
    main()