"""
将 scene1_gt 文件夹下的 .safetensors 文件转换为 .tiff 格式
以适配 realistic_raw_noise_synthesis.py 脚本的输入要求
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from safetensors.torch import load_file
import tifffile as tiff


def convert_safetensors_to_tiff(
    input_dir: str,
    output_dir: str,
    iso: int,
    dtype: str = "uint16"
):
    """
    将 safetensors 文件转换为 TIFF 格式

    Args:
        input_dir: 输入目录 (scene1_gt)
        output_dir: 输出根目录
        iso: ISO 值 (用于组织输出目录)
        dtype: 输出数据类型 ("uint16" 或 "uint12" 映射到 uint16)
    """
    # input_dir="D:/University/Fusion/Phase Final/Transform/dataset/scene1_gt"
    # output_dir="D:/University/Fusion/Phase Final/Transform/dataset/scene1_gt_tiff"
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 按 realistic_raw_noise_synthesis.py 要求的目录结构组织
    # output_root/ISOxxxx/frame_xxx.tiff
    iso_dir = output_path / f"ISO{iso}"
    iso_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有 safetensors 文件
    safetensors_files = sorted(input_path.glob("*.safetensors"))

    if not safetensors_files:
        print(f"错误: 在 {input_dir} 中没有找到 .safetensors 文件")
        return

    print(f"找到 {len(safetensors_files)} 个文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {iso_dir}")
    print(f"ISO: {iso}")
    print(f"数据类型: {dtype}")
    print("-" * 50)

    for sft_path in tqdm(safetensors_files, desc="转换中"):
        try:
            # 1. 加载 safetensors
            tensor_dict = load_file(str(sft_path))

            # key 通常是 "raw"
            if "raw" in tensor_dict:
                tensor = tensor_dict["raw"]
            else:
                # 取第一个 key
                tensor = list(tensor_dict.values())[0]

            # 2. 转换为 numpy
            data = tensor.cpu().numpy().astype(np.float32)

            # 3. 数据范围检查和转换
            # 原始数据是 12-bit (0-4095)，需要决定如何保存
            if dtype == "uint16":
                # 12-bit 数据保存为 16-bit (保持原值)
                # 如果需要恢复到原始 16-bit 范围，可以乘以 16
                # data = data * 16  # 取消注释以恢复到 16-bit

                data = np.clip(np.round(data), 0, 65535).astype(np.uint16)
            else:
                data = np.clip(np.round(data), 0, 4095).astype(np.uint16)

            # 4. 构造输出文件名
            # 从 frame0.safetensors 提取编号
            stem = sft_path.stem  # frame0, frame1, ...
            frame_idx = int(stem.replace("frame",""))
            out_filename = f"frame{frame_idx:03d}_noisy.tiff"
            out_filepath = iso_dir / out_filename

            # 5. 保存为 TIFF
            tiff.imwrite(str(out_filepath), data)

        except Exception as e:
            print(f"转换失败 {sft_path.name}: {e}")

    print("-" * 50)
    print(f"转换完成! 共处理 {len(safetensors_files)} 个文件")
    print(f"输出目录: {iso_dir}")

    # 打印 realistic_raw_noise_synthesis.py 需要的参数提示
    print("\n" + "=" * 50)
    print("realistic_raw_noise_synthesis.py 参数提示:")
    print("=" * 50)
    print(f"  --clean_root {output_path}")
    print(f"  --dark_root <你的暗帧目录>")
    print(f"  --output_root <输出目录>")
    print(f"  --black_level {256 // 16}")  # 12-bit 下的黑电平
    print(f"  --white_level 4095")  # 12-bit 白电平


def main():
    parser = argparse.ArgumentParser(
        description="将 safetensors 转换为 TIFF 格式，适配 realistic_raw_noise_synthesis.py"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"\PipeLine\scene1_gt",
        help="输入目录 (包含 .safetensors 文件)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"\PipeLine\clean_raw_for_synthesis",
        help="输出根目录"
    )
    parser.add_argument(
        "--iso",
        type=int,
        required=True,
        help="ISO 值 (必需，例如 1600, 3200, 6400)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16"],
        help="输出数据类型"
    )

    args = parser.parse_args()

    convert_safetensors_to_tiff(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        iso=args.iso,
        dtype=args.dtype
    )


if __name__ == "__main__":
    main()
