import os
import subprocess
import cv2
import numpy as np
from pathlib import Path
from Liang.src.data.CRVD.SequenceWise_Load import CRVDDataset


def run_vbm3d_and_recover():
    # --- 1. 路径设置 ---
    bin_dir = Path(r"E:/Reasearch/Code/vbm3d_1/build/bin")
    exe_path = str(bin_dir / "VBM3Ddenoising.exe")
    input_temp_dir = Path(r"E:/vbm3d_input_test")
    input_temp_dir.mkdir(exist_ok=True)

    # 清理 bin 目录下的旧产物，防止“幽灵帧”干扰
    for f in bin_dir.glob("deno_*.tiff"): os.remove(f)
    for f in bin_dir.glob("bsic_*.tiff"): os.remove(f)

    # --- 2. 加载数据 ---
    dataset = CRVDDataset("E:/CRVD_dataset/indoor_raw_noisy", "E:/CRVD_dataset/indoor_raw_gt",
                          scenes=[1], iso_levels=[1600], sequence_mode=True)
    noisy_seq, gt_seq = dataset[0]
    num_frames = 7

    # --- 3. 导出输入 ---
    print(f"导出 {num_frames} 帧输入...")
    for i in range(num_frames):
        # 保持 16位 导出
        cv2.imwrite(str(input_temp_dir / f"test_{i + 1:04d}.tif"), noisy_seq[i].astype(np.uint16))

    # --- 4. 执行算子 ---
    input_pattern = str(input_temp_dir / "test_%04d.tif").replace("\\", "/")
    args = [exe_path, "-i", input_pattern, "-f", "1", "-l", str(num_frames), "-sigma", "20", "-add", "false"]

    print("算子正在计算 (32线程全开)...")
    subprocess.run(args, cwd=str(bin_dir), capture_output=True, check=True)

    # --- 5. 精准回收 ---
    # 根据日志：产物是 deno_001.tiff (3位数字，双f后缀)
    denoised_frames = []
    print("正在回收 float32 结果...")
    for i in range(1, num_frames + 1):
        # 构造产物路径
        product_path = bin_dir / f"deno_{i:03d}.tiff"

        if product_path.exists():
            # 使用 cv2.IMREAD_UNCHANGED 读取 float32 TIFF
            img = cv2.imread(str(product_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                # 算子输出是 float32，我们将其限制在 12bit 合理范围 [0, 4095]
                img_clipped = np.clip(img, 0, 4095)
                denoised_frames.append(img_clipped)
                print(f"成功读回: {product_path.name} | Max: {img_clipped.max():.2f}")
        else:
            print(f"未找到产物: {product_path}")

    # --- 6. 验证结果 ---
    if len(denoised_frames) == num_frames:
        mse = np.mean((denoised_frames[0] - gt_seq[0]) ** 2)
        psnr = 10 * np.log10(4095 ** 2 / mse)
        print(f"\n✅ 数据流闭环完成！")
        print(f"第一帧去噪后 PSNR: {psnr:.2f} dB")
    else:
        print(f"\n❌ 回收失败，只得到 {len(denoised_frames)} 帧。")


if __name__ == "__main__":
    run_vbm3d_and_recover()