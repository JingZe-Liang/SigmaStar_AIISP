#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建不含 clean 通道的 HDF5 数据集（支持自定义文件后缀）。

增强：参考 build_h5_dataset.py 完善校验与验证能力。
- 支持 --verify-only 仅校验模式
- 支持 --verify-samples 自定义抽查数量
- 结构校验、样本对齐、首帧策略验证
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import tifffile as tiff

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


def build_frame_paths(index: int, base_2dnr: Path, base_3dnr: Path, base_noisy: Path,
                      suffix_2dnr: str, suffix_3dnr: str, suffix_noisy: str) -> Dict[str, Path]:
    """根据全局帧索引构造文件路径"""
    frame_str = f"frame{index+1:03d}"
    prev_frame_str = f"frame{index:03d}" if index > 0 else frame_str

    return {
        "2dnr": base_2dnr / f"{frame_str}{suffix_2dnr}.tiff",
        "3dnr": base_3dnr / f"{frame_str}{suffix_3dnr}.tiff",
        "noisy_curr": base_noisy / f"{frame_str}{suffix_noisy}.tiff",
        "noisy_prev": base_noisy / f"{prev_frame_str}{suffix_noisy}.tiff",
    }


def read_tiff_u16(path: Path, expected_shape: Sequence[int]) -> np.ndarray:
    arr = tiff.imread(str(path))
    if arr.ndim != 2:
        raise ValueError(f"{path} 不是单通道图像，shape={arr.shape}")
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"{path} 尺寸不匹配：期望 {expected_shape}，实际 {arr.shape}")
    if arr.dtype != np.uint16:
        raise ValueError(f"{path} 数据类型不是 uint16")
    return arr


def load_sample(index: int, config) -> Dict[str, np.ndarray]:
    paths = build_frame_paths(index, config.input_2dnr, config.input_3dnr, config.input_noisy,
                              config.suffix_2dnr, config.suffix_3dnr, config.suffix_noisy)

    dnr2 = read_tiff_u16(paths["2dnr"], config.frame_shape)
    dnr3 = read_tiff_u16(paths["3dnr"], config.frame_shape)
    noisy_curr = read_tiff_u16(paths["noisy_curr"], config.frame_shape)

    if index == 0:
        noisy_prev = noisy_curr.copy()
    else:
        noisy_prev = read_tiff_u16(paths["noisy_prev"], config.frame_shape)

    noisy = np.stack([noisy_prev, noisy_curr], axis=0)
    return {"2dnr": dnr2, "3dnr": dnr3, "noisy": noisy}


def create_shard_datasets(handle: h5py.File, shard_size: int, config) -> Dict[str, h5py.Dataset]:
    H, W = config.frame_shape
    common = {
        "dtype": "uint16",
        "compression": config.compression,
        "compression_opts": config.compression_opts,
        "shuffle": config.shuffle,
    }
    return {
        "2dnr": handle.create_dataset("2dnr", shape=(shard_size, H, W), chunks=(1, H, W), **common),
        "3dnr": handle.create_dataset("3dnr", shape=(shard_size, H, W), chunks=(1, H, W), **common),
        "noisy": handle.create_dataset("noisy", shape=(shard_size, 2, H, W), chunks=(1, 2, H, W), **common),
    }


def write_shard(shard_id: int, indices: Sequence[int], config, overwrite: bool, quiet: bool) -> dict:
    shard_path = config.output_dir / f"shard_{shard_id}.h5"
    tmp_path = shard_path.with_suffix(".h5.tmp")

    if shard_path.exists() and not overwrite:
        raise FileExistsError(f"{shard_path} 已存在，使用 --overwrite 覆盖")
    if tmp_path.exists():
        tmp_path.unlink()

    if not quiet:
        print(f"写入 {shard_path.name}，帧索引 {indices[0]}~{indices[-1]}")

    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.attrs["shard_id"] = shard_id
            h5.attrs["global_start_idx"] = indices[0]
            h5.attrs["global_end_idx"] = indices[-1]
            h5.attrs["num_frames"] = len(indices)

            dsets = create_shard_datasets(h5, len(indices), config)
            for local_i, global_i in enumerate(tqdm(indices, desc=f"shard_{shard_id}", disable=quiet)):
                sample = load_sample(global_i, config)
                dsets["2dnr"][local_i] = sample["2dnr"]
                dsets["3dnr"][local_i] = sample["3dnr"]
                dsets["noisy"][local_i] = sample["noisy"]

        tmp_path.replace(shard_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return {
        "shard_id": shard_id,
        "file": shard_path.name,
        "num_frames": len(indices),
        "global_start_idx": indices[0],
        "global_end_idx": indices[-1],
    }


def write_metadata(config, shard_infos: List[dict]) -> Path:
    H, W = config.frame_shape
    meta = {
        "num_shards": config.num_shards,
        "frames_per_shard": config.frames_per_shard,
        "total_frames": config.num_frames,
        "frame_shape": [H, W],
        "dtype": "uint16",
        "datasets": {
            "2dnr": {"shape": [config.frames_per_shard, H, W], "chunks": [1, H, W]},
            "3dnr": {"shape": [config.frames_per_shard, H, W], "chunks": [1, H, W]},
            "noisy": {"shape": [config.frames_per_shard, 2, H, W], "chunks": [1, 2, H, W]},
        },
        "compression": config.compression,
        "compression_opts": config.compression_opts,
        "shuffle": config.shuffle,
        "first_frame_strategy": "duplicate",
        "shards": shard_infos,
        "source_dirs": {
            "2dnr": str(config.input_2dnr),
            "3dnr": str(config.input_3dnr),
            "noisy": str(config.input_noisy),
        },
        "file_suffixes": {
            "2dnr": config.suffix_2dnr,
            "3dnr": config.suffix_3dnr,
            "noisy": config.suffix_noisy,
        },
    }
    meta_path = config.output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta_path


def validate_inputs(config) -> None:
    for idx in range(min(5, config.num_frames)):
        paths = build_frame_paths(idx, config.input_2dnr, config.input_3dnr, config.input_noisy,
                                  config.suffix_2dnr, config.suffix_3dnr, config.suffix_noisy)
        for key, p in paths.items():
            if key.endswith("_prev") and idx == 0:
                continue
            if not p.is_file():
                raise FileNotFoundError(f"缺失文件: {p}")


# ----------------------------------------------------------------------
# 增强的校验功能（参考 build_h5_dataset.py）
# ----------------------------------------------------------------------

def build_verify_indices(total_frames: int, sample_count: int) -> List[int]:
    """生成要抽查的全局索引列表，确保包含首尾帧。"""
    sample_count = max(1, min(sample_count, total_frames))
    if sample_count == 1:
        return [0]
    raw_indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=int).tolist()
    unique_indices = sorted(set(int(idx) for idx in raw_indices))
    if 0 not in unique_indices:
        unique_indices.insert(0, 0)
    if total_frames - 1 not in unique_indices:
        unique_indices.append(total_frames - 1)
    return unique_indices


def verify_metadata(config) -> dict:
    """校验 metadata.json 的关键字段是否与 config 一致。"""
    meta_path = config.output_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"缺少 metadata.json：{meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    expected = {
        "num_shards": config.num_shards,
        "frames_per_shard": config.frames_per_shard,
        "total_frames": config.num_frames,
        "frame_shape": list(config.frame_shape),
        "compression": config.compression,
        "compression_opts": config.compression_opts,
        "shuffle": config.shuffle,
        "first_frame_strategy": "duplicate",
    }
    for key, exp_val in expected.items():
        act_val = meta.get(key)
        if act_val != exp_val:
            raise ValueError(f"metadata.json 字段 {key} 不匹配，期望 {exp_val}，实际 {act_val}")

    # 检查必要的数据集定义
    if "datasets" not in meta:
        raise ValueError("metadata.json 缺少 'datasets' 字段")
    for ds in ["2dnr", "3dnr", "noisy"]:
        if ds not in meta["datasets"]:
            raise ValueError(f"metadata.json 缺少数据集定义: {ds}")

    return meta


def verify_shard_structure(config, metadata: dict) -> None:
    """检查每个分片 HDF5 文件的结构、shape、dtype、chunks、压缩参数等。"""
    expected_names = {"2dnr", "3dnr", "noisy"}
    shard_entries = metadata.get("shards", [])
    if len(shard_entries) != config.num_shards:
        raise ValueError(
            f"metadata.json 中 shards 数量不匹配，期望 {config.num_shards}，实际 {len(shard_entries)}"
        )

    H, W = config.frame_shape
    for entry in shard_entries:
        shard_id = entry["shard_id"]
        shard_path = config.output_dir / f"shard_{shard_id}.h5"
        if not shard_path.is_file():
            raise FileNotFoundError(f"缺少分片文件：{shard_path}")

        expected_shard_size = entry["num_frames"]
        with h5py.File(shard_path, "r") as f:
            actual_names = set(f.keys())
            if actual_names != expected_names:
                raise ValueError(
                    f"{shard_path} 的 dataset 集合不匹配，期望 {expected_names}，实际 {actual_names}"
                )

            # 检查 2dnr / 3dnr
            for ds_name in ("2dnr", "3dnr"):
                ds = f[ds_name]
                if ds.shape != (expected_shard_size, H, W):
                    raise ValueError(f"{shard_path}:{ds_name} shape 不匹配，实际 {ds.shape}")
                if ds.dtype != np.uint16:
                    raise ValueError(f"{shard_path}:{ds_name} dtype 不匹配，实际 {ds.dtype}")
                if ds.chunks != (1, H, W):
                    raise ValueError(f"{shard_path}:{ds_name} chunks 不匹配，实际 {ds.chunks}")
                if ds.compression != config.compression:
                    raise ValueError(f"{shard_path}:{ds_name} compression 不匹配，实际 {ds.compression}")
                if ds.compression_opts != config.compression_opts:
                    raise ValueError(f"{shard_path}:{ds_name} compression_opts 不匹配，实际 {ds.compression_opts}")
                if bool(ds.shuffle) != config.shuffle:
                    raise ValueError(f"{shard_path}:{ds_name} shuffle 不匹配，实际 {ds.shuffle}")

            # 检查 noisy
            ds_noisy = f["noisy"]
            if ds_noisy.shape != (expected_shard_size, 2, H, W):
                raise ValueError(f"{shard_path}:noisy shape 不匹配，实际 {ds_noisy.shape}")
            if ds_noisy.dtype != np.uint16:
                raise ValueError(f"{shard_path}:noisy dtype 不匹配，实际 {ds_noisy.dtype}")
            if ds_noisy.chunks != (1, 2, H, W):
                raise ValueError(f"{shard_path}:noisy chunks 不匹配，实际 {ds_noisy.chunks}")
            if ds_noisy.compression != config.compression:
                raise ValueError(f"{shard_path}:noisy compression 不匹配，实际 {ds_noisy.compression}")
            if ds_noisy.compression_opts != config.compression_opts:
                raise ValueError(f"{shard_path}:noisy compression_opts 不匹配，实际 {ds_noisy.compression_opts}")
            if bool(ds_noisy.shuffle) != config.shuffle:
                raise ValueError(f"{shard_path}:noisy shuffle 不匹配，实际 {ds_noisy.shuffle}")


def verify_sample_alignment(config, verify_samples: int) -> List[int]:
    """随机抽查若干全局索引，对比 HDF5 内容与原始 TIFF 是否一致。"""
    checked = build_verify_indices(config.num_frames, verify_samples)
    for global_idx in checked:
        shard_id = global_idx // config.frames_per_shard
        local_idx = global_idx % config.frames_per_shard
        shard_path = config.output_dir / f"shard_{shard_id}.h5"

        expected = load_sample(global_idx, config)
        with h5py.File(shard_path, "r") as f:
            actual_2dnr = f["2dnr"][local_idx]
            actual_3dnr = f["3dnr"][local_idx]
            actual_noisy = f["noisy"][local_idx]

        if not np.array_equal(actual_2dnr, expected["2dnr"]):
            raise ValueError(f"索引 {global_idx} 的 2dnr 内容不匹配")
        if not np.array_equal(actual_3dnr, expected["3dnr"]):
            raise ValueError(f"索引 {global_idx} 的 3dnr 内容不匹配")
        if not np.array_equal(actual_noisy, expected["noisy"]):
            raise ValueError(f"索引 {global_idx} 的 noisy 内容不匹配")

        # 第0帧 duplicate 检查
        if global_idx == 0 and not np.array_equal(actual_noisy[0], actual_noisy[1]):
            raise ValueError("索引 0 的 noisy 未满足 duplicate 策略")

    return checked


def verify_dataset_full(config, verify_samples: int, quiet: bool = False) -> None:
    """执行完整校验流程：元数据、分片结构、样本对齐。"""
    if not quiet:
        print("开始校验...")
    validate_inputs(config)  # 确保源文件仍在（仅校验模式可能已移动，但这里保持原样）
    metadata = verify_metadata(config)
    verify_shard_structure(config, metadata)
    checked_indices = verify_sample_alignment(config, verify_samples)
    if not quiet:
        print(f"校验通过！抽查索引：{checked_indices}")


# ----------------------------------------------------------------------
# 命令行参数解析
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="构建不含 clean 的 H5 数据集（支持完整校验）")
    parser.add_argument("--input-2dnr", type=Path, required=True, help="2DNR TIFF 目录")
    parser.add_argument("--input-3dnr", type=Path, required=True, help="3DNR TIFF 目录")
    parser.add_argument("--input-noisy", type=Path, required=True, help="Noisy TIFF 目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--num-frames", type=int, default=150, help="总帧数")
    parser.add_argument("--frames-per-shard", type=int, default=30, help="每个分片的帧数")
    parser.add_argument("--height", type=int, default=1080, help="图像高度")
    parser.add_argument("--width", type=int, default=1920, help="图像宽度")
    parser.add_argument("--compression", default="gzip", help="H5 压缩算法")
    parser.add_argument("--compression-opts", type=int, default=4, help="压缩等级")
    parser.add_argument("--shuffle", action="store_true", default=True, help="启用 shuffle 过滤")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的文件")
    parser.add_argument("--verify", action="store_true", help="写入后执行完整校验")
    parser.add_argument("--verify-only", action="store_true", help="仅校验已有数据集，不重新写入")
    parser.add_argument("--verify-samples", type=int, default=5, help="校验时抽查的全局索引数量")
    parser.add_argument("--quiet", action="store_true", help="减少输出")
    # 自定义文件后缀
    parser.add_argument("--suffix-2dnr", type=str, default="_2dnr", help="2DNR 文件名后缀（不含扩展名），如 _noisy_2dnr")
    parser.add_argument("--suffix-3dnr", type=str, default="_3dnr", help="3DNR 文件名后缀（不含扩展名），如 _noisy_3dnr")
    parser.add_argument("--suffix-noisy", type=str, default="_noisy", help="Noisy 文件名后缀（不含扩展名），如 _noisy")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only and args.overwrite:
        raise ValueError("--verify-only 与 --overwrite 不能同时使用")

    class Config:
        pass
    config = Config()
    for k, v in vars(args).items():
        setattr(config, k, v)
    config.frame_shape = (args.height, args.width)
    config.num_shards = math.ceil(args.num_frames / args.frames_per_shard)

    if args.verify_only:
        # 仅校验模式
        verify_dataset_full(config, args.verify_samples, quiet=args.quiet)
        return

    # 正常构建流程
    validate_inputs(config)

    shard_infos = []
    for shard_id in range(config.num_shards):
        start = shard_id * config.frames_per_shard
        end = min(start + config.frames_per_shard, config.num_frames)
        indices = list(range(start, end))
        info = write_shard(shard_id, indices, config, args.overwrite, args.quiet)
        shard_infos.append(info)

    write_metadata(config, shard_infos)

    if args.verify:
        verify_dataset_full(config, args.verify_samples, quiet=args.quiet)

    print(f"完成！输出目录：{config.output_dir}")


if __name__ == "__main__":
    main()