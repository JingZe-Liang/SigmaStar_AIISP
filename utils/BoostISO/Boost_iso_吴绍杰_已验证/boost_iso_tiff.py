#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAW ISO Boost 工具
功能：
1. 读取 1080p、12bit、GBRG、Bayer RAW 视频帧（TIFF 格式，float32）
2. 减去 black_level
3. 剩余信号乘以 scale_factor（线性手动提高 ISO）
4. 加回 black_level
5. 输出完全相同格式的 TIFF 文件（float32、1080x1920、12bit 范围）
6. 支持批量处理整个文件夹（按文件名排序，保证视频帧顺序）

使用方法：
    python raw_iso_boost.py \
        --input_dir ./original_raw_tiffs \
        --output_dir ./boosted_gt_tiffs \
        --metadata metadata.json \
        --scale 2.5
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tifffile  # pip install tifffile numpy


def load_metadata(json_path: str):
    """加载 metadata.json"""
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    black_level = meta.get("black_level")
    if black_level is None:
        raise ValueError("metadata.json 中必须包含 'black_level' 字段")

    # 支持标量或 list（GBRG 四通道），当前脚本优先使用标量（最常用）
    if isinstance(black_level, list):
        print("⚠️  检测到 per-channel black_level，本脚本当前使用第一个值（标量模式）")
        black_level = float(black_level[0])  # 简化处理，实际可扩展为像素级
    else:
        black_level = float(black_level)

    return {
        "black_level": black_level,
        "bayer_pattern": meta.get("bayer_pattern", "GBRG"),
        "bit_depth": int(meta.get("bit_depth", 12)),
        "scale_factor": meta.get("scale_factor")  # json 中可预设，默认被命令行覆盖
    }


def boost_raw_frame(
        raw: np.ndarray,
        black_level: float,
        scale_factor: float
) -> np.ndarray:

    if raw.dtype != np.float32:
        raw = raw.astype(np.float32)

    # 确保是 2D（单通道 Bayer）
    if raw.ndim == 3 and raw.shape[2] == 1:
        raw = raw.squeeze(axis=2)
    if raw.shape != (1080, 1920):
        raise ValueError(f"输入帧尺寸应为 1080x1920，当前是 {raw.shape}")

    signal = raw - black_level

    signal_boosted = signal * scale_factor

    boosted = signal_boosted + black_level

    boosted = np.clip(boosted, 0.0, 4095.0)

    print(f"   [诊断] 处理前 → min:{raw.min():.1f}  max:{raw.max():.1f}  mean:{raw.mean():.1f}")
    print(f"   [诊断] signal 均值: {signal.mean():.2f}   (有效信号强度)")
    print(
        f"   [诊断] 处理后 → min:{boosted.min():.1f}  max:{boosted.max():.1f}  mean:{boosted.mean():.1f}   ← 提升 {boosted.mean() / raw.mean():.2f}x")

    return boosted.astype(np.float32)

    # 保持 float32
    return boosted.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="RAW 视频帧手动提高 ISO 工具（师兄方法）")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="原始 TIFF 帧文件夹路径（1080p GBRG float32）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出 boosted GT TIFF 文件夹路径")
    parser.add_argument("--metadata", type=str, required=True,
                        help="metadata.json 路径（包含 black_level 等）")
    parser.add_argument("--scale", type=float, default=None,
                        help="手动提高 ISO 的 scale_factor（默认使用 json 中的值）")
    parser.add_argument("--file_suffix", type=str, default=".tiff",
                        help="TIFF 文件后缀（默认 .tiff）")

    args = parser.parse_args()

    # 1. 加载 metadata
    meta = load_metadata(args.metadata)
    black_level = meta["black_level"]
    scale_factor = args.scale if args.scale is not None else meta.get("scale_factor", 2.5)

    print(f"✅ 参数加载完成")
    print(f"   black_level = {black_level}")
    print(f"   scale_factor = {scale_factor}  (模拟 ISO ≈ 5184 × {scale_factor:.1f})")
    print(f"   输入目录: {args.input_dir}")
    print(f"   输出目录: {args.output_dir}")

    # 2. 准备输入输出文件夹
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 3. 获取所有 TIFF 文件并按文件名排序（保证视频帧顺序）
    tiff_files = sorted(
        [f for f in input_path.glob(f"*{args.file_suffix}") if f.is_file()],
        key=lambda x: x.name
    )

    if not tiff_files:
        raise FileNotFoundError(f"在 {input_path} 中未找到任何 {args.file_suffix} 文件")

    print(f"📊 发现 {len(tiff_files)} 帧 RAW TIFF，开始处理...\n")

    # 4. 逐帧处理
    for idx, tiff_file in enumerate(tiff_files):
        # 读取 float32 RAW
        raw = tifffile.imread(str(tiff_file))

        boosted = boost_raw_frame(raw, black_level, scale_factor)

        # 输出路径（保持原文件名）
        out_file = output_path / tiff_file.name

        # 保存为完全相同的格式（float32 TIFF）
        # photometric='minisblack' + compression=None 适合 RAW 数据保留精确数值
        tifffile.imwrite(
            str(out_file),
            boosted,
            photometric='minisblack',  # 单通道 RAW 推荐
            compression=None,  # 不压缩，保持原始精度
            metadata={'description': f'ISO_boosted_scale_{scale_factor}'}
        )

        if (idx + 1) % 20 == 0 or idx == 0 or idx == len(tiff_files) - 1:
            print(f"   [{idx + 1:04d}/{len(tiff_files)}] 已保存 → {out_file.name}")

    print("\n🎉 全部处理完成！")
    print(f"   输出目录：{output_path}")
    print(f"   共处理 {len(tiff_files)} 帧")
    print(f"   建议下一步：使用这些 boosted GT 继续加噪（darkframe + Noise Modeling in One Hour）")


if __name__ == "__main__":
    main()