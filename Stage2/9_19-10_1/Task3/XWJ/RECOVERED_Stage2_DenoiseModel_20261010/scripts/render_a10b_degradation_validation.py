#!/usr/bin/env python3
"""Render fixed-ISP and statistical evidence for A10b validation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a10_sc450_audit import load_a10_audit_config  # noqa: E402
from raw_fusion.finetune.cross_sensor import render_fixed_isp  # noqa: E402
from raw_fusion.finetune.stream_io import CHANNEL_NAMES  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_by_name(config_path: Path, name: str):
    config = load_a10_audit_config(config_path)
    matches = [source for source in config.sources if source.name == name]
    if len(matches) != 1:
        raise ValueError(f"A10b render source not uniquely defined: {name}")
    return matches[0]


def _fixed_isp_row(
    samples: np.lib.npyio.NpzFile,
    prefix: str,
    source,
    black: np.ndarray,
    row_label: str,
    cell_width: int,
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    entries = (
        ("real", "Real RAW"),
        ("conditioning_proxy_not_gt", "Proxy (not GT)"),
        ("quantized_iid_q4", "Q4 IID"),
        ("a10b_spectral_q4", "Shared spectral"),
    )
    cells: list[np.ndarray] = []
    for suffix, label in entries:
        rendered = render_fixed_isp(
            samples[f"{prefix}_{suffix}"],
            black,
            source.source.fixed_isp_wb,
            source.source.fixed_isp_exposure,
        )
        if crop is not None:
            x0, y0, x1, y1 = crop
            rendered = rendered[y0:y1, x0:x1]
        scale = cell_width / rendered.shape[1]
        resized = cv2.resize(
            rendered,
            (cell_width, max(1, int(round(rendered.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        canvas = cv2.copyMakeBorder(
            resized, 52, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)
        )
        cv2.putText(
            canvas,
            f"{row_label} | {label}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cells.append(canvas)
    minimum_height = min(cell.shape[0] for cell in cells)
    return np.concatenate([cell[:minimum_height] for cell in cells], axis=1)


def _mean_acf(metrics: dict[str, object], direction: str) -> tuple[np.ndarray, np.ndarray]:
    lags = np.asarray(
        sorted(int(value) for value in metrics["channels"]["R"][direction]),
        dtype=np.int64,
    )
    values = np.asarray(
        [
            [float(metrics["channels"][name][direction][str(lag)]) for lag in lags]
            for name in CHANNEL_NAMES
        ],
        dtype=np.float64,
    )
    return lags, np.mean(values, axis=0)


def _render_statistics(result: dict[str, object], output: Path) -> None:
    validation = result["validation"]
    real = validation["real_metrics"]
    models = validation["models"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    colors = {
        "Real validation": "#222222",
        "Q4 IID": "#227C8E",
        "A10b spectral": "#C43E35",
    }
    for axis, direction, title in (
        (axes[0, 0], "horizontal_correlation", "Horizontal packed-pixel ACF"),
        (axes[0, 1], "vertical_correlation", "Vertical packed-pixel ACF"),
    ):
        for label, metrics in (
            ("Real validation", real),
            ("Q4 IID", models["quantized_iid_q4"]["metrics"]),
            ("A10b spectral", models["a10b_spectral_q4"]["metrics"]),
        ):
            lags, values = _mean_acf(metrics, direction)
            axis.plot(lags, values, marker="o", label=label, color=colors[label])
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Packed-pixel lag")
        axis.set_ylabel("Mean correlation across Bayer channels")
        axis.set_xticks(range(1, 9))
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    sigma_axis = axes[1, 0]
    labels = ("Legacy Q1 IID", "Q4 IID", "A10b spectral")
    keys = ("legacy_a9_iid_q1", "quantized_iid_q4", "a10b_spectral_q4")
    comparisons = [models[key]["comparison_to_real_validation"] for key in keys]
    positions = np.arange(len(labels))
    sigma_axis.bar(
        positions - 0.18,
        [item["aggregate_median_relative_sigma_error"] for item in comparisons],
        0.36,
        label="Median",
        color="#227C8E",
    )
    sigma_axis.bar(
        positions + 0.18,
        [item["aggregate_p90_relative_sigma_error"] for item in comparisons],
        0.36,
        label="P90",
        color="#E3B341",
    )
    sigma_axis.axhline(0.075, color="#227C8E", linestyle="--", linewidth=1.0)
    sigma_axis.axhline(0.20, color="#E3B341", linestyle="--", linewidth=1.0)
    sigma_axis.set_xticks(positions, labels, rotation=12)
    sigma_axis.set_title("Signal-bin sigma relative error")
    sigma_axis.set_ylabel("Relative error")
    sigma_axis.grid(axis="y", alpha=0.25)
    sigma_axis.legend(loc="best")

    pair_axis = axes[1, 1]
    pairwise = validation["pairwise_spatial_comparison"]
    frames = [item["first_frame"] for item in pairwise["pairs"]]
    iid = [
        item["weighted_correlation_mae"]["quantized_iid_q4"]
        for item in pairwise["pairs"]
    ]
    spectral = [
        item["weighted_correlation_mae"]["a10b_spectral_q4"]
        for item in pairwise["pairs"]
    ]
    pair_axis.plot(frames, iid, marker="o", label="Q4 IID", color="#227C8E")
    pair_axis.plot(
        frames, spectral, marker="o", label="A10b spectral", color="#C43E35"
    )
    pair_axis.axhline(0.01, color="#333333", linestyle="--", linewidth=1.0)
    pair_axis.set_title(
        f"Per-pair weighted ACF MAE: spectral wins "
        f"{pairwise['spectral_wins']}/{len(frames)}"
    )
    pair_axis.set_xlabel("First frame of validation pair")
    pair_axis.set_ylabel("Weighted ACF MAE")
    pair_axis.grid(alpha=0.25)
    pair_axis.legend(loc="best")
    figure.suptitle(
        f"A10b sc450ai degradation validation | {result['status']}", fontsize=15
    )
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=REPO_ROOT
        / "artifacts/a10b_sc450ai_degradation_validation_20261016/validation.json",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=REPO_ROOT
        / "artifacts/a10b_sc450ai_degradation_validation_20261016/visual_samples.npz",
    )
    parser.add_argument(
        "--a10-config",
        type=Path,
        default=REPO_ROOT / "configs/a10_sc450ai_supervision_audit_20261016.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "artifacts/a10b_sc450ai_degradation_validation_20261016",
    )
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    samples = np.load(args.samples)
    black = samples["black_levels"].astype(np.float32)
    validation_source = _source_by_name(
        args.a10_config, result["validation"]["source"]
    )
    holdout_source = _source_by_name(
        args.a10_config, result["hard_holdout"]["source"]
    )
    overview_rows = [
        _fixed_isp_row(
            samples, "validation", validation_source, black, "804x val", 480
        ),
        _fixed_isp_row(
            samples, "hard_holdout", holdout_source, black, "1435x holdout", 480
        ),
    ]
    width = max(row.shape[1] for row in overview_rows)
    padded = [
        cv2.copyMakeBorder(
            row,
            0,
            0,
            0,
            width - row.shape[1],
            cv2.BORDER_CONSTANT,
            value=(20, 20, 20),
        )
        for row in overview_rows
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "fixed_isp_evidence.png"
    panel = np.concatenate(padded, axis=0)
    if not cv2.imwrite(str(panel_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write A10b panel: {panel_path}")
    crop_rows = [
        _fixed_isp_row(
            samples,
            "validation",
            validation_source,
            black,
            "804x val crop",
            480,
            crop=(40, 40, 640, 840),
        ),
        _fixed_isp_row(
            samples,
            "hard_holdout",
            holdout_source,
            black,
            "1435x wall crop",
            480,
            crop=(650, 50, 1250, 850),
        ),
    ]
    crop_width = max(row.shape[1] for row in crop_rows)
    crop_padded = [
        cv2.copyMakeBorder(
            row,
            0,
            0,
            0,
            crop_width - row.shape[1],
            cv2.BORDER_CONSTANT,
            value=(20, 20, 20),
        )
        for row in crop_rows
    ]
    crop_panel = np.concatenate(crop_padded, axis=0)
    crop_panel_path = args.output_dir / "fixed_isp_noise_crops.png"
    if not cv2.imwrite(
        str(crop_panel_path), cv2.cvtColor(crop_panel, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError(f"failed to write A10b crop panel: {crop_panel_path}")
    statistics_path = args.output_dir / "statistical_evidence.png"
    _render_statistics(result, statistics_path)
    manifest = {
        "fixed_isp_evidence": {
            "path": str(panel_path.resolve()),
            "sha256": _sha256(panel_path),
            "shape": list(panel.shape),
        },
        "fixed_isp_noise_crops": {
            "path": str(crop_panel_path.resolve()),
            "sha256": _sha256(crop_panel_path),
            "shape": list(crop_panel.shape),
        },
        "statistical_evidence": {
            "path": str(statistics_path.resolve()),
            "sha256": _sha256(statistics_path),
        },
        "conditioning_proxy_is_clean_ground_truth": False,
        "sources": [str(args.result.resolve()), str(args.samples.resolve())],
    }
    manifest_path = args.output_dir / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
