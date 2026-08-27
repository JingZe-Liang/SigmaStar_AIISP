from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

STAGE3_ROOT = Path(__file__).resolve().parent
STAGE2_ROOT = STAGE3_ROOT.parent / "Stage2"
WORKSPACE_ROOT = STAGE3_ROOT.parents[2]
DATA_ROOT = STAGE2_ROOT / "data"
for path in (STAGE2_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_io import RawStreamReader, discover_dataset, pack_bayer  # noqa: E402
from motionnet_stage3 import TemporalMotionNet, build_motion_features  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Infer Stage3 independent temporal motion maps")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Existing fusion inference root")
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT / "DATASET")
    parser.add_argument("--sequences", nargs="+", default=["128x", "645x"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--halo", type=int, default=3)
    return parser.parse_args()


def predict_tiled(model, arrays, sigma, device, tile_size, halo, amp_enabled):
    height, width = arrays[0].shape[1:]
    output = np.empty((height, width), dtype=np.float32)
    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            et, eb = max(0, top - halo), min(height, bottom + halo)
            el, er = max(0, left - halo), min(width, right + halo)
            tensors = [torch.from_numpy(np.ascontiguousarray(x[:, et:eb, el:er], dtype=np.float32)).unsqueeze(0).to(device) for x in arrays]
            sigma_tensor = torch.from_numpy(sigma).view(1, 4, 1, 1).to(device)
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with torch.inference_mode(), context:
                probability = torch.sigmoid(model(build_motion_features(*tensors, sigma_tensor)))[0, 0]
            core = (slice(top - et, bottom - et), slice(left - el, right - el))
            output[top:bottom, left:right] = probability[core].float().cpu().numpy()
    return output


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "motion_model" not in checkpoint:
        raise KeyError("Checkpoint has no independent motion_model; retrain with the current Stage3 code")
    model = TemporalMotionNet(**checkpoint.get("motion_model_config", {}))
    model.load_state_dict(checkpoint["motion_model"])
    model.to(device).eval()
    config = checkpoint["config"]
    statistics = {**config["val_statistics"], **config["train_statistics"]}
    supported_start = int(config["warmup_frames"]) + 3
    catalog = discover_dataset(args.dataset_root)
    selected = [s for s in catalog.fusion_sequences if s.sequence_id in set(args.sequences)]
    if len(selected) != len(set(args.sequences)):
        raise ValueError("Unknown sequence requested")
    for sequence in selected:
        destination = args.output / sequence.sequence_id / "predicted_motion_u8.raw"
        if not (args.output / sequence.sequence_id / "fusion.raw").is_file():
            raise FileNotFoundError(f"Fusion inference is required first: {destination.parent}")
        reader = RawStreamReader(sequence.source)
        stats = statistics[sequence.sequence_id]
        offset = np.asarray(stats["source_to_dnr_offset"], dtype=np.float32).reshape(4, 1, 1)
        sigma = np.asarray(stats["noise_sigma"], dtype=np.float32)
        temporary = destination.with_suffix(".partial.raw")
        with temporary.open("wb") as handle:
            for index in range(sequence.frame_count):
                if supported_start <= index < sequence.frame_count - 3:
                    def frame_at(frame_index):
                        return pack_bayer(reader.read_frame(frame_index), sequence.source.cfa_pattern).astype(np.float32) - float(config["black_source"]) + offset
                    probability = predict_tiled(model, (frame_at(index), frame_at(index - 1), frame_at(index + 1)), sigma, device, args.tile_size, args.halo, device.type == "cuda")
                    motion = np.clip(np.rint(probability * 255.0), 0, 255).astype(np.uint8)
                else:
                    motion = np.zeros((540, 960), dtype=np.uint8)
                handle.write(motion.tobytes())
        temporary.replace(destination)
        print(f"{sequence.sequence_id}: wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
