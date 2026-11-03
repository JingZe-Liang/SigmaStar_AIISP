"""Validate and summarize the seven paired A12 evaluations without model inference."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_SHAPE = (8, 4, 540, 960)
WINDOWS = [list(range(start, start + 8)) for start in (0, 96, 192)]
PAIRS = {("baseline", "ema")} | {(phase, weights) for phase in ("shared", "fixed", "refresh") for weights in ("model", "ema")}
PROBE_ERRORS = ("rmse_dn", "highpass_rmse_dn", "gradient_rmse_dn", "temporal_difference_error_rmse_dn")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(value: dict, prefix: str = "") -> dict[str, float | int]:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, list):
            result.update(flatten(dict(enumerate(item)), name))
        else:
            require(isinstance(item, (int, float)) and math.isfinite(item), f"non-finite/non-numeric metric: {name}")
            result[name] = item
    return result


def summarize(manifest_path: Path, output_dir: Path) -> dict:
    manifest_path, output_dir = manifest_path.resolve(), output_dir.resolve()
    require(output_dir.parent == (REPO_ROOT / "artifacts").resolve() and output_dir.name.startswith("a12_"), "output must be artifacts/a12_*")
    require(not output_dir.exists(), "output already exists")
    specs = json.loads(manifest_path.read_text(encoding="utf-8"))["specs"]
    require(len(specs) == 7 and {(spec["phase"], spec["weights"]) for spec in specs} == PAIRS, "exactly seven baseline/shared/fixed/refresh states are required")
    require(len({spec["label"] for spec in specs}) == 7, "labels must be unique")
    loaded, file_hashes = {}, {}
    for spec in specs:
        directory = (REPO_ROOT / spec["output_dir"]).resolve()
        reports = {}
        for kind, relative in (("identity", "identity.json"), ("probes", "probes/evaluation.json"), ("native", "native/evaluation.json")):
            path = directory / relative
            reports[kind] = json.loads(path.read_text(encoding="utf-8"))
            file_hashes[str(path)] = sha256(path)
        identity, probes, native = (reports[key] for key in ("identity", "probes", "native"))
        require(identity["label"] == native["label"] == probes["baseline"]["name"] == spec["label"], "evaluation label mismatch")
        require(identity["weights"] == native["weights"] == probes["weights"] == probes["baseline"]["format"] == spec["weights"], "checkpoint state mismatch")
        require(len({identity["checkpoint_sha256"], native["checkpoint_sha256"], probes["checkpoint_sha256"], probes["baseline"]["checkpoint_sha256"]}) == 1, "checkpoint hash mismatch")
        checkpoint = Path(identity["checkpoint"])
        require(Path(native["checkpoint"]) == Path(probes["checkpoint"]) == Path(probes["baseline"]["checkpoint"]) == checkpoint, "checkpoint path mismatch")
        if str(checkpoint) not in file_hashes:
            file_hashes[str(checkpoint)] = sha256(checkpoint)
        require(file_hashes[str(checkpoint)] == identity["checkpoint_sha256"], "checkpoint file changed")
        require(identity["gpu_name"] == "NVIDIA GeForce RTX 3090" and str(identity["cuda_visible_devices"]).startswith("GPU-"), "3090 UUID identity is required")
        require(identity["device"] == probes["device"] == native["device"], "device identity mismatch")
        require(identity["prepared_manifest_sha256"] == probes["prepared_manifest_sha256"] == sha256(Path(probes["prepared_manifest"])), "prepared probe identity mismatch")
        require(native["prediction_dtype"] == "float32" and identity["tile_batch"] == native["tile_batch"] == 1, "FP32/tile_batch protocol mismatch")
        require((native["spatial_tile"], native["spatial_stride"], native["temporal_frames"]) == (64, 56, 8), "native tiling protocol mismatch")
        require(identity["native_starts"] == [0, 96, 192] and identity["native_frames"] == 8, "identity windows mismatch")
        require([window["frame_indices"] for window in native["windows"]] == WINDOWS, "three independent native windows are required")
        require(len(probes["cases"]) == len({case["name"] for case in probes["cases"]}) == 12, "twelve unique probes are required")
        for kind, items in (("probes", probes["cases"]), ("native", native["windows"])):
            for item in items:
                path = directory / kind / item["prediction"]["file"]
                require(sha256(path) == item["prediction"]["sha256"], f"prediction changed: {path}")
                prediction = np.load(path, mmap_mode="r", allow_pickle=False)
                require(prediction.dtype == np.float32 and bool(np.isfinite(prediction).all()), "prediction must be finite FP32")
                if kind == "native":
                    require(tuple(prediction.shape) == tuple(item["input_shape"]) == NATIVE_SHAPE, "native prediction/input shape mismatch")
        loaded[(spec["phase"], spec["weights"])] = {"spec": spec, **reports}
    baseline = loaded[("baseline", "ema")]
    base_cases = {case["name"]: case for case in baseline["probes"]["cases"]}
    native_keys = ("protocol", "prediction_dtype", "tile_batch", "spatial_tile", "spatial_stride", "temporal_frames", "temporal_policy", "manifest", "raw", "motion_mask", "reference", "valid_region")
    runtime_keys = ("device", "gpu_name", "cuda_visible_devices", "torch_version", "cuda_version")
    for (phase, weights), report in loaded.items():
        require(all(report["identity"][key] == baseline["identity"][key] for key in runtime_keys), "evaluation runtime differs from A3")
        require(all(report["probes"][key] == baseline["probes"][key] for key in ("prepared_manifest_sha256", "metric_protocol")), "probe scoring protocol differs from A3")
        require(all(report["native"][key] == baseline["native"][key] for key in native_keys), "native reference/input protocol differs from A3")
        require({case["name"] for case in report["probes"]["cases"]} == set(base_cases), "probe case identities differ from A3")
        if phase != "baseline":
            require(report["identity"]["checkpoint_sha256"] == loaded[(phase, "ema")]["identity"]["checkpoint_sha256"], "model/EMA must come from the same phase checkpoint")
        for case in report["probes"]["cases"]:
            old = base_cases[case["name"]]
            require((case["mode"], case["gain"]) == (old["mode"], old["gain"]), "probe region selection differs")
            require(case["model_input"] == old["model_input"], "probe input metrics differ")
        for window, old in zip(report["native"]["windows"], baseline["native"]["windows"], strict=True):
            require(all(window[key] == old[key] for key in ("frame_indices", "input_shape", "input_sha256", "motion_window_sha256", "black_dn")), "native window input/mask identity differs")
            require(all(window["metrics"][key] == old["metrics"][key] for key in ("input", "static_pixels", "static_fraction")), "native scoring population differs")
            require(window["metrics"]["finetuned"]["pixels"] == old["metrics"]["finetuned"]["pixels"] == window["metrics"]["static_pixels"], "native reconstruction population differs")
    rows, counts = {"probe": [], "native": []}, {}
    for (phase, weights), report in loaded.items():
        label = report["spec"]["label"]
        counts[label] = {}
        for kind in rows:
            items = report["probes"]["cases"] if kind == "probe" else report["native"]["windows"]
            for index, item in enumerate(items):
                unit = item["name"] if kind == "probe" else f"{item['frame_indices'][0]}-{item['frame_indices'][-1]}"
                region = ("all" if item["mode"] == "static" else "motion_swept") if kind == "probe" else "native_static"
                old = base_cases[unit] if kind == "probe" else baseline["native"]["windows"][index]
                values = flatten(item[label][region] if kind == "probe" else item["metrics"])
                old_values = flatten(old[baseline["spec"]["label"]][region] if kind == "probe" else old["metrics"])
                require(values.keys() == old_values.keys(), "metric keys differ from A3")
                if kind == "probe":
                    require(all(values[key] == old_values[key] for key in ("pixels", "highpass_pixels")), "probe scoring population differs")
                fixed_values = {}
                if phase == "refresh":
                    fixed = loaded[("fixed", weights)]
                    fixed_item = next(case for case in fixed["probes"]["cases"] if case["name"] == unit) if kind == "probe" else fixed["native"]["windows"][index]
                    fixed_values = flatten(fixed_item[fixed["spec"]["label"]][region] if kind == "probe" else fixed_item["metrics"])
                for metric, value in values.items():
                    delta = value - old_values[metric]
                    rows[kind].append({"label": label, "phase": phase, "weights": weights, "unit": unit, "region": region, "metric": metric, "value": value, "a3_value": old_values[metric], "delta_from_a3": delta, "fixed_value": fixed_values.get(metric), "delta_from_fixed": value - fixed_values[metric] if fixed_values else None})
                    if kind == "probe" and metric in PROBE_ERRORS:
                        key = f"{item['mode']}.{metric}"
                        counter = counts[label].setdefault(key, {"lower": 0, "equal": 0, "higher": 0})
                        counter["lower" if delta < 0 else "higher" if delta > 0 else "equal"] += 1
    summary = {"schema": "a12_paired_summary_v1", "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "source_file_sha256": file_hashes, "acceptance_status": "diagnostic_only_no_quality_pass_claim", "primary_weights": "ema", "model_weights_role": "diagnostic_only", "direction_counts_vs_a3": counts, "evaluations": [report["identity"] | {"phase": phase} for (phase, _), report in loaded.items()], "probe_metrics": rows["probe"], "native_metrics": rows["native"]}
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for kind, records in rows.items():
        with (output_dir / f"{kind}_metrics.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.manifest, args.output_dir)
    print(json.dumps({"status": "summarized", "evaluations": len(result["evaluations"]), "output_dir": str(args.output_dir)}))
