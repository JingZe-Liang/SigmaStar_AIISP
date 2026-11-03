#!/usr/bin/env python3
"""Render the A10 sc450ai supervision-audit evidence figure."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ("R", "Gr", "Gb", "B")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=REPO_ROOT / "artifacts/a10_sc450ai_supervision_audit_20261016/audit.json",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=REPO_ROOT
        / "artifacts/a3_cross_sensor_extension_20261012/sequences/sc450ai",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/a10_sc450ai_supervision_audit_20261016",
    )
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    sources = audit["inventory"]["sources"]
    noise = sources["sc450ai_804x_lightchanging"]["spatial_noise_correlation"]
    light_means = np.load(
        args.profile_root
        / "sc450ai_804x_lightchanging/profile/frame_means.npy"
    )
    motion_means = np.load(
        args.profile_root
        / "sc450ai_1435x_motion_black_object/profile/frame_means.npy"
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    frame_axis = axes[0, 0]
    frame_axis.plot(light_means, color="#C43E35", label="804x light-changing")
    frame_axis.plot(motion_means, color="#227C8E", label="1435x black-object")
    frame_axis.axvspan(120, 149, color="#E3B341", alpha=0.20, label="804x validation")
    frame_axis.set_title("Frame-mean drift")
    frame_axis.set_xlabel("Frame")
    frame_axis.set_ylabel("Mean native DN")
    frame_axis.legend(loc="best")
    frame_axis.grid(alpha=0.2)

    window_axis = axes[0, 1]
    for source_name, color, label in (
        ("sc450ai_804x_lightchanging", "#C43E35", "804x"),
        ("sc450ai_1435x_motion_black_object", "#227C8E", "1435x"),
    ):
        windows = sources[source_name]["windows"]["all_windows"]
        window_axis.scatter(
            [100.0 * item["frame_mean_cv"] for item in windows],
            [100.0 * item["flat_stable_pixel_fraction"] for item in windows],
            s=28,
            alpha=0.75,
            color=color,
            label=label,
        )
    window_axis.set_title("Eight-frame window stability")
    window_axis.set_xlabel("Frame-mean CV (%)")
    window_axis.set_ylabel("Flat stable pixels (%)")
    window_axis.legend(loc="best")
    window_axis.grid(alpha=0.2)

    correlation_axis = axes[1, 0]
    positions = np.arange(len(CHANNELS), dtype=np.float64)
    width = 0.18
    series = (
        ("Train horizontal", "train", "horizontal_neighbor_correlation", "#C43E35"),
        ("Validation horizontal", "validation", "horizontal_neighbor_correlation", "#E3B341"),
        ("Train vertical", "train", "vertical_neighbor_correlation", "#227C8E"),
        ("Validation vertical", "validation", "vertical_neighbor_correlation", "#4F7D3B"),
    )
    for index, (label, split, metric, color) in enumerate(series):
        values = [noise[split]["channels"][name][metric] for name in CHANNELS]
        correlation_axis.bar(
            positions + (index - 1.5) * width,
            values,
            width,
            label=label,
            color=color,
        )
    correlation_axis.axhline(0.0, color="#303030", linewidth=0.8)
    correlation_axis.set_xticks(positions, CHANNELS)
    correlation_axis.set_title("Per-channel nearest-neighbor correlation")
    correlation_axis.set_ylabel("Correlation")
    correlation_axis.legend(loc="upper right", fontsize=8)
    correlation_axis.grid(axis="y", alpha=0.2)

    matrix_axis = axes[1, 1]
    train_matrix = np.asarray(noise["train"]["cross_channel_correlation"])
    validation_matrix = np.asarray(noise["validation"]["cross_channel_correlation"])
    difference = validation_matrix - train_matrix
    image = matrix_axis.imshow(difference, cmap="coolwarm", vmin=-0.08, vmax=0.08)
    matrix_axis.set_xticks(range(4), CHANNELS)
    matrix_axis.set_yticks(range(4), CHANNELS)
    matrix_axis.set_title("Cross-channel correlation: validation - train")
    for row in range(4):
        for column in range(4):
            matrix_axis.text(
                column,
                row,
                f"{difference[row, column]:+.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="#111111",
            )
    figure.colorbar(image, ax=matrix_axis, shrink=0.8)
    figure.suptitle(
        "A10 sc450ai existing-data supervision audit",
        fontsize=16,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "audit_evidence.png"
    figure.savefig(output, dpi=140)
    plt.close(figure)
    manifest = {
        "figure": str(output.resolve()),
        "sha256": _sha256(output),
        "shape": [1400, 2240, 4],
        "sources": [
            str(args.audit.resolve()),
            str(
                (
                    args.profile_root
                    / "sc450ai_804x_lightchanging/profile/frame_means.npy"
                ).resolve()
            ),
            str(
                (
                    args.profile_root
                    / "sc450ai_1435x_motion_black_object/profile/frame_means.npy"
                ).resolve()
            ),
        ],
    }
    manifest_path = args.output_dir / "audit_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
