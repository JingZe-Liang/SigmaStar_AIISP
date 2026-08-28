from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from data import CODE_MAX, HEIGHT, WIDTH, discover_sequences, stage1_h5_split
from train import config_path, read_config


ROOT = Path(__file__).resolve().parent
H5_KEYS = {"2dnr", "3dnr", "clean", "noisy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the NAFBPNNet cloud training environment")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cloud.json")
    parser.add_argument("--environment-only", action="store_true")
    return parser.parse_args()


def check_environment(config: dict) -> dict[str, object]:
    modules = {name: importlib.import_module(name) for name in ("torch", "cv2", "h5py", "tqdm")}
    torch = modules["torch"]
    cuda_available = bool(torch.cuda.is_available())
    if str(config["device"]).startswith("cuda") and not cuda_available:
        raise RuntimeError("config 要求 CUDA，但当前 PyTorch 未检测到可用 GPU")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "opencv": modules["cv2"].__version__,
        "h5py": modules["h5py"].__version__,
        "numpy": np.__version__,
    }


def inspect_h5(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        missing = H5_KEYS - set(handle.keys())
        if missing:
            raise KeyError(f"{path} 缺少 H5 键: {sorted(missing)}")
        clean = handle["clean"]
        expected_image_shape = (int(clean.shape[0]), HEIGHT, WIDTH)
        for key in ("2dnr", "3dnr", "clean"):
            dataset = handle[key]
            if dataset.shape != expected_image_shape or dataset.dtype != np.dtype("uint16"):
                raise ValueError(f"{path}:{key} 应为 uint16 {expected_image_shape}，实际 {dataset.dtype} {dataset.shape}")
        noisy = handle["noisy"]
        if noisy.shape != (expected_image_shape[0], 2, HEIGHT, WIDTH) or noisy.dtype != np.dtype("uint16"):
            raise ValueError(f"{path}:noisy 应为 uint16 [N,2,{HEIGHT},{WIDTH}]，实际 {noisy.dtype} {noisy.shape}")
        return expected_image_shape[0]


def check_stage1(root: Path) -> dict[str, object]:
    train_files, validation_files = stage1_h5_split(root)
    train_samples = sum(inspect_h5(path) for path in train_files)
    validation_samples = sum(inspect_h5(path) for path in validation_files)
    overlap = set(train_files) & set(validation_files)
    if overlap:
        raise AssertionError(f"Stage 1 train/validation H5 重叠: {next(iter(overlap))}")
    return {
        "root": str(root),
        "train_h5_files": len(train_files),
        "validation_h5_files": len(validation_files),
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "validation_policy": "naturally_last_h5_per_scene",
    }


def check_stage2(config: dict) -> dict[str, object]:
    sequences = discover_sequences(
        config_path(config, "data_root"),
        tuple(config["sequence_names"]),
        None,
        str(config["cfa_pattern"]),
        int(config["source_black_level"]),
        int(config["dnr_black_level"]),
        int(config["white_level"]),
    )
    return {
        "root": str(config_path(config, "data_root")),
        "sequences": [item.name for item in sequences],
        "frame_count": len(sequences[0].dnr2_paths),
        "raw_shape": [HEIGHT, WIDTH],
        "raw_code_max": CODE_MAX,
    }


def main() -> int:
    args = parse_args()
    config = read_config(args.config)
    report: dict[str, object] = {"environment": check_environment(config)}
    if not args.environment_only:
        report["stage1"] = check_stage1(config_path(config, "stage1_data_root"))
        report["stage2"] = check_stage2(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
