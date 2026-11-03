"""Measure frozen A12 ROIs using A3 caches without training or inference."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_fusion.finetune.a12_acceptance import file_sha256, measure_sequence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/a12_joint_acceptance_20261029.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/a12_joint_acceptance_20261029"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    baseline = config["baseline"]
    actual_hash = file_sha256(Path(baseline["checkpoint"]))
    if actual_hash != baseline["checkpoint_sha256"]:
        raise ValueError("frozen A3 checkpoint SHA-256 mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for entry in config["sequences"]:
        if entry["assessment_kind"] == "historical_only":
            result = {"id": entry["id"], "sensor": entry["sensor"], "assessment_kind": "historical_only", "data_identity": entry["data_identity"], "historical_evidence": entry["historical_evidence"], "measurement_origin": "historical_only_not_recomputed", "acceptance": "not_assessed_in_this_run"}
        else:
            result = measure_sequence(entry, actual_hash)
        (args.output_dir / f"{entry['id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        results.append(result)
        print(f"[a12] {entry['id']}: {result['measurement_origin']}", flush=True)
    report = {
        "schema_version": 1,
        "date": "2026-10-29",
        "config_sha256": file_sha256(args.config),
        "checkpoint_sha256": actual_hash,
        "status": "baseline_recorded_joint_quality_not_yet_accepted",
        "metric_contract": "Native DN; inputs normalized and clipped as in historical inference; 3x3 support eroded before static scoring; signed gradient error; temporal differences only within contiguous clips; no dynamic pseudo-ground-truth.",
        "limitations": ["Existing scenes have been inspected during development; no new independent final test set.", "ROI classes are reference-derived heuristics, not manual labels.", "Shared same-stream fixed-pattern bias is not independently removed by odd/even references.", "No true dynamic target, no full-sequence temporal acceptance, no calibrated perceptual pass threshold.", "Prediction/reference hashes checked; original RAW source size checked, full RAW hashes not recomputed."],
        "sequences": results,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    rows = ["# A12 联合验收基线", "", "本轮重新计算已有 A3 预测缓存的固定 ROI 指标，不代表新模型效果，也不构成最终画质通过。", "", "| 序列 | 本轮处理 |", "| --- | --- |"]
    rows.extend(f"| {row['id']} | {'历史证据，仅登记' if row['assessment_kind'] == 'historical_only' else 'CPU 重算固定 ROI'} |" for row in results)
    rows.extend(["", "静态结果按奇偶互斥参考分别记录重建、高通、低通、带符号梯度误差和偏差；动态指标仅作诊断。具体数值见同目录各序列 JSON。", "", "尚待补齐：真实动态逐帧参考、独立新场景、人工纹理/运动 ROI 核验、完整连续视频及候选模型比较。"])
    (args.output_dir / "report.md").write_text("\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
