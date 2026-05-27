from pathlib import Path

import h5py
import numpy as np


def inspect_h5_dataset(file_path: str | Path) -> None:
    """检查 H5 数据集结构，并推断 noisy 的时序排列方式。"""
    file_path = Path(file_path)
    print(f"正在分析文件：{file_path}")
    print("-" * 50)

    with h5py.File(file_path, "r") as h5_file:
        keys = list(h5_file.keys())
        print(f"包含的数据集 Keys：{keys}")

        for key in keys:
            dataset = h5_file[key]
            chunk_status = "[OK]" if dataset.chunks and dataset.chunks[0] == 1 else "[警告: 首维 chunk 不是 1，随机访问性能可能偏差]"
            print(f"\n数据集 '{key}':")
            print(f"  - Shape: {dataset.shape}")
            print(f"  - Dtype: {dataset.dtype}")
            print(f"  - Chunks: {dataset.chunks} {chunk_status}")

        if "clean" in keys and "noisy" in keys:
            print("\n" + "=" * 20 + " 核心对齐检查 " + "=" * 20)
            idx = min(10, len(h5_file["clean"]) - 1)
            gt = h5_file["clean"][idx]
            noisy_group = h5_file["noisy"][idx]

            dist_0 = float(np.mean(np.abs(noisy_group[0] - gt)))
            dist_1 = float(np.mean(np.abs(noisy_group[1] - gt)))

            if dist_0 < dist_1:
                print(">>> 时序推断：Channel 0 更接近当前帧 t，Channel 1 更像上一帧 t-1。")
            else:
                print(">>> 时序推断：Channel 1 更接近当前帧 t，Channel 0 更像上一帧 t-1。")

            print(f"    平均绝对误差：dist_0={dist_0:.4f}, dist_1={dist_1:.4f}")


if __name__ == "__main__":
    target_file = Path(r"D:\DeepLearning\VideoDenoising\CheckDataset\DarkData\scene1\shard_0.h5")
    if target_file.exists():
        inspect_h5_dataset(target_file)
    else:
        print("未检测到目标文件。")
