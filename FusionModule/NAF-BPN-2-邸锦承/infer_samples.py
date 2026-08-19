from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from data import (
    CODE_MAX,
    FRAME_COUNT,
    HEIGHT,
    NR_BLACK_LEVEL,
    SOURCE_BLACK_LEVEL,
    WIDTH,
    discover_sequences,
    raw_to_linear,
    read_candidate,
    read_motion,
    read_source,
)
from model import forward_padded, load_pretrained_model


ROOT = Path(__file__).resolve().parent
SAMPLE_INDICES = {
    "128x": (14, 33, 55, 62, 78, 84, 91, 103, 107, 150),
    "645x": (48, 51, 59, 79, 89, 97, 109, 116, 123, 167),
}
GAIN_PATTERN = re.compile(r"R=(\d+),G=(\d+),B=(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(description="导出 strict v3 checkpoint 的固定十帧四宫格")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def gains_from_path(path: Path) -> tuple[int, int, int]:
    match = GAIN_PATTERN.search(str(path))
    if match is None:
        raise ValueError(f"无法从 source 路径解析 R/G/B 增益: {path}")
    return tuple(int(value) for value in match.groups())


def to_bgr(frame: np.ndarray, gains: tuple[int, int, int], black_level: int, exposure: float) -> np.ndarray:
    normalized = np.clip((frame.astype(np.float32) - black_level) / (CODE_MAX - black_level), 0.0, 1.0)
    red_gain, green_gain, blue_gain = gains
    normalized[0::2, 0::2] *= blue_gain / green_gain
    normalized[1::2, 1::2] *= red_gain / green_gain
    mosaic = np.rint(np.clip(normalized, 0.0, 1.0) * 65535.0).astype(np.uint16)
    bgr = cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2BGR).astype(np.float32) / 65535.0
    return np.rint(np.power(np.clip(bgr * exposure, 0.0, 1.0), 1.0 / 2.2) * 255.0).astype(np.uint8)


def estimate_exposure(sequence, gains: tuple[int, int, int]) -> float:
    reference = cv2.imread(str(sequence.dnr2_paths[100].with_suffix(".png")), cv2.IMREAD_COLOR)
    if reference is None:
        raise FileNotFoundError(f"缺少 ISP 标定 PNG: {sequence.dnr2_paths[100].with_suffix('.png')}")
    preview = to_bgr(read_candidate(sequence.dnr2_paths[100]), gains, NR_BLACK_LEVEL, 1.0)
    preview_luma = np.percentile(cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)[::8, ::8], 90) / 255.0
    reference_luma = np.percentile(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)[::8, ::8], 90) / 255.0
    return float(np.clip((reference_luma**2.2) / max(preview_luma, 1e-6), 0.25, 64.0))


def label(frame: np.ndarray, title: str, sequence: str, index: int) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (440, 64), (0, 0, 0), -1)
    cv2.putText(output, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, f"{sequence}  frame {index:03d}", (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def prediction_raw(model, sequence, frame_index: int, device: torch.device) -> np.ndarray:
    inputs = (
        raw_to_linear(read_candidate(sequence.dnr2_paths[frame_index]), NR_BLACK_LEVEL),
        raw_to_linear(read_candidate(sequence.dnr3_paths[frame_index]), NR_BLACK_LEVEL),
        raw_to_linear(read_source(sequence, frame_index), SOURCE_BLACK_LEVEL),
        read_motion(sequence, frame_index),
    )
    tensors = [torch.from_numpy(np.ascontiguousarray(item)).unsqueeze(0).unsqueeze(0).to(device) for item in inputs]
    linear = forward_padded(model, *tensors).squeeze().float().cpu().numpy()
    return np.rint(NR_BLACK_LEVEL + np.clip(linear, 0.0, 1.0) * (CODE_MAX - NR_BLACK_LEVEL)).astype("<u2")


def export_checkpoint_samples(checkpoint: Path, config: dict, model=None, output_root: Path | None = None) -> Path:
    checkpoint = Path(checkpoint)
    if output_root is None:
        output_root = ROOT / "runs_v3" / "strict_blcfa_md_grad" / "samples" / checkpoint.stem
    device = torch.device(config["device"])
    owns_model = model is None
    if owns_model:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("variant") != "strict_blcfa_md_grad":
            raise ValueError("checkpoint 不是 strict_blcfa_md_grad 权重")
        model = load_pretrained_model(str(config["init_checkpoint"]), device)
        model.load_state_dict(payload["model"], strict=True)
    was_training = model.training
    model.eval()
    sequences = discover_sequences(
        Path(config["data_root"]), tuple(config["sequence_names"]), Path(config["motion_cache_root"])
    )
    manifest = {"checkpoint": str(checkpoint), "variant": "strict_blcfa_md_grad", "frames_per_sequence": 10, "sequences": {}}
    with torch.inference_mode():
        for sequence in sequences:
            indices = SAMPLE_INDICES[sequence.name]
            if len(indices) != 10 or min(indices) < 0 or max(indices) >= FRAME_COUNT:
                raise ValueError(f"{sequence.name} 的固定帧列表错误")
            destination = output_root / sequence.name
            destination.mkdir(parents=True, exist_ok=True)
            gains = gains_from_path(sequence.source_path)
            exposure = estimate_exposure(sequence, gains)
            for frame_index in indices:
                noisy = read_source(sequence, frame_index)
                dnr2 = read_candidate(sequence.dnr2_paths[frame_index])
                dnr3 = read_candidate(sequence.dnr3_paths[frame_index])
                predicted = prediction_raw(model, sequence, frame_index, device)
                panels = [
                    label(to_bgr(noisy, gains, SOURCE_BLACK_LEVEL, exposure), "Noisy", sequence.name, frame_index),
                    label(to_bgr(dnr2, gains, NR_BLACK_LEVEL, exposure), "2DNR", sequence.name, frame_index),
                    label(to_bgr(dnr3, gains, NR_BLACK_LEVEL, exposure), "3DNR", sequence.name, frame_index),
                    label(to_bgr(predicted, gains, NR_BLACK_LEVEL, exposure), "AI Fused", sequence.name, frame_index),
                ]
                panels = [cv2.resize(panel, (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA) for panel in panels]
                grid = np.concatenate((np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)), axis=0)
                cv2.imwrite(str(destination / f"four_grid_{frame_index:04d}.png"), grid)
            manifest["sequences"][sequence.name] = {"indices": list(indices), "display_exposure": exposure}
            print(f"{checkpoint.stem} {sequence.name}: 已输出固定 10 帧四宫格", flush=True)
    if was_training:
        model.train()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_root


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    destination = export_checkpoint_samples(args.checkpoint, config, output_root=args.output_root)
    print(f"四宫格输出目录: {destination}", flush=True)


if __name__ == "__main__":
    main()
