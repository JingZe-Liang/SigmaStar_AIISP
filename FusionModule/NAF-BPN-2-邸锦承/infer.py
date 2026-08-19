from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import FRAME_COUNT, HEIGHT, WIDTH, discover_sequences, linear_to_nr_raw, raw_to_linear, read_candidate, read_motion, read_source, NR_BLACK_LEVEL, SOURCE_BLACK_LEVEL
from model import forward_padded, load_pretrained_model


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="v3 严格 BLCFA + MD 模型的完整 RAW 推理")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence", choices=("128x", "645x", "all"), default="all")
    parser.add_argument("--frame-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "strict_blcfa_md_grad":
        raise ValueError("checkpoint 不是 v3 strict_blcfa_md_grad 权重")
    device = torch.device(config["device"])
    model = load_pretrained_model(str(config["init_checkpoint"]), device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    sequences = discover_sequences(Path(config["data_root"]), tuple(config["sequence_names"]), Path(config["motion_cache_root"]))
    if args.sequence != "all":
        sequences = tuple(item for item in sequences if item.name == args.sequence)
    count = FRAME_COUNT if args.frame_limit is None else min(FRAME_COUNT, args.frame_limit)
    output_root = ROOT / "outputs_v3" / args.checkpoint.stem
    with torch.inference_mode():
        for sequence in sequences:
            destination = output_root / sequence.name
            destination.mkdir(parents=True, exist_ok=True)
            for frame_index in range(count):
                images = (
                    raw_to_linear(read_candidate(sequence.dnr2_paths[frame_index]), NR_BLACK_LEVEL),
                    raw_to_linear(read_candidate(sequence.dnr3_paths[frame_index]), NR_BLACK_LEVEL),
                    raw_to_linear(read_source(sequence, frame_index), SOURCE_BLACK_LEVEL),
                    read_motion(sequence, frame_index),
                )
                tensors = [torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).unsqueeze(0).to(device) for image in images]
                prediction = forward_padded(model, *tensors).squeeze().float().cpu().numpy()
                output = linear_to_nr_raw(prediction)
                target = destination / f"out_{frame_index:04d}.raw"
                output.tofile(target)
                if target.stat().st_size != WIDTH * HEIGHT * 2:
                    raise ValueError(f"输出大小错误: {target}")
                if (frame_index + 1) % 25 == 0 or frame_index + 1 == count:
                    print(f"{sequence.name}: {frame_index + 1}/{count}", flush=True)
            manifest = {"variant": "strict_blcfa_md_grad", "checkpoint": str(args.checkpoint), "frames": count, "input_domain": "black-level-corrected linear", "output_domain": {"black_level": NR_BLACK_LEVEL, "white_level": 4095}, "motion": "precomputed robust MD"}
            (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RAW 输出目录: {output_root}")


if __name__ == "__main__":
    main()
