"""Render auditable CPU-only A14 static metrics and fixed native RAW crop comparisons."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "artifacts/.matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np


RESIDUAL_LIMIT_DN = 30.0
SOURCES = (
    ("mis20s1_1000x", "MIS20S1 1000x", "gain_holdout", 92),
    ("sc465sl_2000x", "SC465SL 2000x", "gain_holdout", 144),
    ("sc465sl_darkroom18_256x", "SC465SL 256x", "train_distribution", 144),
    ("sc635hai_131x_low_cct", "sc635hai 131x", "train_distribution", 66),
)
ROLE_LABELS = {"gain_holdout": "Gain holdout", "train_distribution": "Training distribution"}
ROLE_COLORS = {"gain_holdout": "#3572a5", "train_distribution": "#c67525"}


def _path(value: Path | str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict:
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _one(items: list[dict], key: str, value: str) -> dict:
    matches = [item for item in items if item[key] == value]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {key}={value!r}, found {len(matches)}")
    return matches[0]


def _acceptance_label(summary: dict, candidate: dict) -> str:
    if not candidate["passed"]:
        return "Diagnostic only: ROI screen failed; candidate not accepted"
    elif summary["selected"] is not None and summary["selected"]["label"] == candidate["label"]:
        return "Passed ROI screen; larger-field validation remains required"
    else:
        return "Diagnostic view: ROI screen passed, but this candidate was not selected"


def _load_inputs(
    prepared_dir: Path, baseline_dir: Path, candidate_dir: Path, summary_path: Path,
) -> tuple[dict, dict, dict, dict, dict]:
    prepared = _read_json(prepared_dir / "manifest.json")
    baseline_path, candidate_path = baseline_dir / "evaluation.json", candidate_dir / "evaluation.json"
    baseline, candidate, summary = _read_json(baseline_path), _read_json(candidate_path), _read_json(summary_path)
    if summary["schema"] != "a14_static_comparison_v1":
        raise ValueError("summary must be the final A14 comparison, including all candidate states")
    if _sha256(baseline_path) != summary["baseline"]["evaluation_sha256"]:
        raise ValueError("baseline evaluation differs from the comparison summary")
    record = _one(summary["candidates"], "label", candidate["identity"]["label"])
    if _sha256(candidate_path) != record["evaluation"]["sha256"]:
        raise ValueError("candidate evaluation differs from the comparison summary")
    if record["weights"] != candidate["identity"]["weights"] or record["checkpoint"] != candidate["identity"]["checkpoint"]:
        raise ValueError("candidate weights or checkpoint identity differs from the summary")
    manifest_hash = _sha256(prepared_dir / "manifest.json")
    for evaluation in (baseline, candidate):
        if evaluation["identity"]["prepared_manifest"]["sha256"] != manifest_hash:
            raise ValueError("prepared manifest differs from the evaluated cache")
    if baseline["identity"] != summary["baseline"]["identity"]:
        raise ValueError("baseline state identity differs from the summary")
    for key in ("gpu_uuid", "precision", "tf32", "output_clip", "code", "source_roles"):
        if baseline["identity"][key] != candidate["identity"][key]:
            raise ValueError(f"candidate and baseline differ in {key}")
    return prepared, baseline, candidate, summary, record


def _quantitative_figure(record: dict, status: str) -> tuple[Figure, list[dict]]:
    rows = [_one(record["source_results"], "source", source) for source, _, _, _ in SOURCES]
    for row, (_, _, role, _) in zip(rows, SOURCES, strict=True):
        if row["role"] != role:
            raise ValueError("summary source role differs from the frozen plot protocol")
    fields = (
        ("psnr_db_delta", "PSNR change relative to A3 [dB]", "Higher is better"),
        ("highpass_rmse_dn_ratio", "Highpass error / A3 highpass error", "Lower is better"),
        ("gradient_rmse_dn_ratio", "Gradient error / A3 gradient error", "Lower is better"),
        ("flat_residual_sigma_dn_ratio", "Flat residual sigma / A3 flat residual sigma", "Lower is better"),
    )
    labels = [label.replace(" ", "\n", 1) for _, label, _, _ in SOURCES]
    colors = [ROLE_COLORS[role] for _, _, role, _ in SOURCES]
    positions = np.arange(len(SOURCES))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    for index, (axis, (field, title, direction)) in enumerate(zip(axes.flat, fields, strict=True)):
        values = [row[field] for row in rows]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError(f"cannot plot missing or non-finite summary metric {field}")
        values = np.asarray(values, dtype=float)
        if index == 0:
            reference = 0.0
            thresholds = np.asarray([-0.03 if role == "gain_holdout" else 0.05 for _, _, role, _ in SOURCES])
            axis.bar(positions, values, width=0.5, color=colors, alpha=0.85)
            axis.hlines(thresholds, positions - 0.35, positions + 0.35, color="#333333", linestyle="--", linewidth=1.2)
            lower, upper = min(float(values.min()), -0.03, 0), max(float(values.max()), 0.05, 0)
            text_format = "+.3f"
        else:
            reference = 1.0
            axis.vlines(positions, 1.0, values, colors=colors, linewidth=2.5)
            axis.scatter(positions, values, color=colors, s=65, zorder=3)
            axis.axhline(1.01, color="#333333", linestyle="--", linewidth=1.2)
            lower, upper = min(float(values.min()), 1.0), max(float(values.max()), 1.01)
            text_format = ".3f"
        padding = max((upper - lower) * 0.2, 0.015 if index == 0 else 0.003)
        axis.set_ylim(lower - padding, upper + 2 * padding)
        for x, value in zip(positions, values, strict=True):
            axis.annotate(format(float(value), text_format), (x, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
        axis.axhline(reference, color="#8d8d8d", linewidth=0.9)
        axis.set_xticks(positions, labels)
        axis.set_title(f"{title}\n{direction}", fontsize=11, loc="left")
        axis.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=ROLE_COLORS[role], label=ROLE_LABELS[role]) for role in ROLE_LABELS]
    handles.append(Line2D([0], [0], color="#333333", linestyle="--", label="Frozen ROI screening threshold"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=3, frameon=False)
    fig.suptitle(f"A14 static ROI comparison: {record['label']} ({record['weights']}, step {record['step']})\n{status}", fontsize=14, y=0.985)
    fig.text(0.5, 0.005, "Pooled native RAW metrics, 18 clips per source. Approximate split-half references; no unseen-scene or unseen-sensor claim.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.085, 0.98, 0.9), h_pad=2.2, w_pad=2.5)
    return fig, rows


def _prediction(case: dict, directory: Path) -> tuple[np.ndarray, dict]:
    path = _path(case["prediction"]["path"])
    if path.parent != directory or _sha256(path) != case["prediction"]["sha256"]:
        raise ValueError("prediction path or file hash differs from its evaluation record")
    values = np.load(path, allow_pickle=False)
    if values.shape != (8, 4, 64, 64) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("prediction must be finite native DN with shape [8,4,64,64] and FP32 dtype")
    return values, _identity(path)


def _green(frame: np.ndarray) -> np.ndarray:
    return (frame[1].astype(np.float64) + frame[2].astype(np.float64)) * 0.5


def _qualitative_figure(
    prepared: dict, baseline: dict, candidate: dict, record: dict, status: str,
    prepared_dir: Path, baseline_dir: Path, candidate_dir: Path,
) -> tuple[Figure, list[dict]]:
    gray_map = plt.get_cmap("gray").copy()
    gray_map.set_bad("#c7c7c7")
    residual_map = plt.get_cmap("RdBu_r").copy()
    residual_map.set_bad("#c7c7c7")
    fig, axes = plt.subplots(4, 6, figsize=(16.8, 12))
    column_titles = ("Native RAW input", "Approx. static reference", "A3 EMA", f"Candidate {record['weights']}", "A3 minus reference", "Candidate minus reference")
    selected = []
    residual_artist = None
    for row_index, (source_name, display_name, role, start) in enumerate(SOURCES):
        case_name = f"{source_name}_roi2_f{start:06d}_p0"
        case = _one(prepared["cases"], "name", case_name)
        if case["frame_indices"] != list(range(start, start + 16, 2)) or case["input_parity"] != 0 or case["target_parity"] != 1:
            raise ValueError("representative case does not match the frozen middle-window parity policy")
        source = _one(prepared["sources"], "name", source_name)
        base_case, candidate_case = _one(baseline["cases"], "name", case_name), _one(candidate["cases"], "name", case_name)
        if Path(case["file"]).name != case["file"]:
            raise ValueError("prepared case filename must not escape its cache")
        case_path = prepared_dir / case["file"]
        if _sha256(case_path) != case["sha256"]:
            raise ValueError("prepared case has changed since evaluation")
        for evaluation_case in (base_case, candidate_case):
            if evaluation_case["prepared_case_sha256"] != case["sha256"]:
                raise ValueError("prediction was evaluated against a different prepared case")
        base_dn, base_identity = _prediction(base_case, baseline_dir)
        candidate_dn, candidate_identity = _prediction(candidate_case, candidate_dir)
        with np.load(case_path, allow_pickle=False) as data:
            raw, reference = _green(data["noisy_dn"][0]), _green(data["target_dn"][0])
            valid = np.array(data["valid"][0], dtype=bool)
            black = data["black_dn"].tolist()
        if raw.shape != (64, 64) or reference.shape != raw.shape or valid.shape != raw.shape or not valid.any():
            raise ValueError("representative raw/reference/valid arrays do not match the 64-pixel ROI")
        if not np.isfinite(raw).all() or not np.isfinite(reference).all():
            raise ValueError("raw or reference visualization contains non-finite values")
        lower, upper = np.percentile(reference[valid], [2.0, 98.0])
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError("reference cannot define a finite, nonconstant display contrast")
        base, current = _green(base_dn[0]), _green(candidate_dn[0])
        base_residual, current_residual = base - reference, current - reference
        images = (raw, reference, base, current, base_residual, current_residual)
        clipped = {}
        for column, (axis, image) in enumerate(zip(axes[row_index], images, strict=True)):
            masked = np.ma.array(image, mask=~valid)
            if column < 4:
                axis.imshow(masked, cmap=gray_map, vmin=lower, vmax=upper, interpolation="nearest", origin="upper", resample=False)
                fraction = float(((image[valid] < lower) | (image[valid] > upper)).mean())
            else:
                residual_artist = axis.imshow(masked, cmap=residual_map, vmin=-RESIDUAL_LIMIT_DN, vmax=RESIDUAL_LIMIT_DN,
                                               interpolation="nearest", origin="upper", resample=False)
                fraction = float((np.abs(image[valid]) > RESIDUAL_LIMIT_DN).mean())
            clipped[column_titles[column]] = fraction
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel(f"Display clipped: {100 * fraction:.1f}%", fontsize=8)
            if row_index == 0:
                axis.set_title(column_titles[column], fontsize=10, pad=8)
            for spine in axis.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#777777")
        axes[row_index, 0].set_ylabel(f"{display_name}\n{ROLE_LABELS[role]}\nROI ({case['roi']['y']}, {case['roi']['x']}), frame {start}\nTarget q2–q98: {lower:.1f}–{upper:.1f} DN", fontsize=9, labelpad=13)
        selected.append({
            "source": source_name, "role": role, "case": case_name, "prepared_case": _identity(case_path),
            "roi": case["roi"], "frame_index_in_clip": 0, "original_frame_index": start,
            "input_frame_indices": case["frame_indices"], "target_frame_pool": case["target_frame_pool"],
            "raw_source": source["raw"], "target_reference": case["target_reference"], "black_dn": black,
            "baseline_prediction": base_identity, "candidate_prediction": candidate_identity,
            "display_contrast_dn": [float(lower), float(upper)], "display_contrast_population": "Only target green-mean values inside the frozen valid mask; q2 and q98.",
            "residual_display_limits_dn": [-RESIDUAL_LIMIT_DN, RESIDUAL_LIMIT_DN],
            "valid_pixels": int(valid.sum()), "display_clipped_fractions": clipped,
            "reference_limitation": source["brightness_limitation"],
        })
    fig.suptitle(f"Fixed native RAW static detail views: {record['label']}\n{status}", fontsize=14, y=0.99)
    fig.text(0.5, 0.945, "Green-plane mean (Gr + Gb) / 2, linear display. Each row shares target-only contrast. Gray pixels are outside the frozen valid mask.", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.11, top=0.90, wspace=0.08, hspace=0.26)
    colorbar_axis = fig.add_axes((0.62, 0.065, 0.30, 0.012))
    colorbar = fig.colorbar(residual_artist, cax=colorbar_axis, orientation="horizontal", ticks=[-30, -15, 0, 15, 30], extend="both")
    colorbar.set_label("Prediction minus approximate reference [DN]; display clipped at ±30 DN", fontsize=9)
    fig.text(0.13, 0.025, "All rows: preselected highest-texture ROI (roi2), middle temporal window, even input frames, first frame.\nApproximate opposite-parity static references retain noise and possible drift. No demosaicing, gamma, denoising, or interpolation smoothing.", fontsize=9)
    return fig, selected


def render_results(
    prepared_dir: Path, baseline_dir: Path, candidate_dir: Path, summary_path: Path, output_dir: Path,
) -> dict:
    prepared_dir, baseline_dir, candidate_dir, summary_path, output_dir = map(_path, (prepared_dir, baseline_dir, candidate_dir, summary_path, output_dir))
    relative = output_dir.relative_to(REPO_ROOT)
    if not (relative.parts[:2] == ("docs", "assets") or relative.parts[:1] == ("artifacts",)):
        raise ValueError("visualization output must be inside repository docs/assets or artifacts")
    if output_dir.exists():
        raise FileExistsError("visualization directory already exists; previous outputs cannot be overwritten")
    prepared, baseline, candidate, summary, record = _load_inputs(prepared_dir, baseline_dir, candidate_dir, summary_path)
    status = _acceptance_label(summary, record)
    quantitative, rows = _quantitative_figure(record, status)
    qualitative, selected = _qualitative_figure(prepared, baseline, candidate, record, status,
                                                prepared_dir, baseline_dir, candidate_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for name, figure in (("a14_source_metrics", quantitative), ("a14_fixed_static_detail", qualitative)):
        for suffix in ("png", "pdf", "svg"):
            path = output_dir / f"{name}.{suffix}"
            figure.savefig(path, dpi=220, facecolor="white")
            outputs.append(_identity(path))
        plt.close(figure)
    manifest = {
        "schema": "a14_static_visualization_v1", "script": _identity(Path(__file__).resolve()),
        "summary": _identity(summary_path), "prepared_manifest": _identity(prepared_dir / "manifest.json"),
        "baseline_evaluation": _identity(baseline_dir / "evaluation.json"),
        "candidate_evaluation": _identity(candidate_dir / "evaluation.json"),
        "baseline_identity": baseline["identity"], "candidate_identity": candidate["identity"],
        "candidate_step": record["step"], "candidate_passed_roi_screen": record["passed"],
        "comparison_status": summary["status"], "displayed_acceptance_label": status,
        "selected_candidate": summary["selected"], "quantitative_source_rows": rows,
        "qualitative_selection_policy": "Frozen before candidate results: source roi2, middle temporal window (MIS 92, SC465 144, sc635 66), parity0, clip frame0.",
        "display_policy": {"channels": "Arithmetic mean of packed Gr and Gb in native DN", "transfer_function": "Linear",
                           "contrast_quantiles": [2.0, 98.0], "contrast_source": "Approximate target only, inside frozen valid mask",
                           "residual_display_limits_dn": [-RESIDUAL_LIMIT_DN, RESIDUAL_LIMIT_DN],
                           "invalid_pixels": "Neutral gray in all six columns", "interpolation": "nearest", "resample": False},
        "selected_cases": selected, "outputs": outputs,
        "limitations": ["Approximate static references are not independent clean truth.",
                        "Training-distribution results and same-scene gain holdouts are shown separately.",
                        "No new scene, dynamic-scene, or unseen-sensor generalization is established by these views.",
                        "Display clipping percentages describe the visualization scale, not sensor saturation.",
                        "ROI screening or visualization does not replace default inference weights."],
    }
    with (output_dir / "visualization_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = render_results(args.prepared_dir, args.baseline_dir, args.candidate_dir, args.summary, args.output_dir)
    print(json.dumps({"manifest": str(_path(args.output_dir) / "visualization_manifest.json"),
                      "candidate": result["candidate_identity"]["label"],
                      "status": result["displayed_acceptance_label"], "figures": len(result["outputs"])}))


if __name__ == "__main__":
    main()
