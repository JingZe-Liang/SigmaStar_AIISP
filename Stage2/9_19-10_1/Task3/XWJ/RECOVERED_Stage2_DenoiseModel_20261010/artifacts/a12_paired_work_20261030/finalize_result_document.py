"""Publish compact, versioned evidence from the fully validated evaluation summary."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
ASSETS = ROOT / "docs/assets/a12_paired_20261030"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summary_path = ROOT / "artifacts/a12_paired_summary_20261030/summary.json"
    summary = json.loads(summary_path.read_text())
    training_path = WORK / "final_training_verification.json"
    training = json.loads(training_path.read_text())
    completion = json.loads((WORK / "evaluation_completion_status.json").read_text())
    assert completion["status"] == "completed" and len(completion["completed"]) == 5
    assert training["all_passed"] and len(summary["evaluations"]) == 7
    metrics = {
        "probe": {"rmse_dn", "highpass_rmse_dn", "gradient_rmse_dn", "temporal_difference_error_rmse_dn"},
        "native": {"finetuned.rmse_dn", "finetuned.psnr_db", "flat_residual_sigma_dn", "temporal_mad_dn", "high_frequency_ratio_mean", "static_fraction"},
    }
    for kind, keys in metrics.items():
        rows = [row for row in summary[f"{kind}_metrics"] if row["metric"] in keys]
        with (ASSETS / f"{kind}_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    evidence = {
        "schema": "a12_paired_versioned_evidence_v1", "date": "2026-10-30",
        "scope": "Single-seed development diagnostic; A3 retained; no quality acceptance claim.",
        "training_verification_sha256": digest(training_path),
        "summary_sha256": digest(summary_path),
        "total_actual_updates": training["total_actual_updates"],
        "total_training_elapsed_seconds": training["total_training_elapsed_seconds"],
        "peak_allocated_bytes": training["peak_allocated_bytes"],
        "peak_reserved_bytes": training["peak_reserved_bytes"],
        "both_branch_initial_states_identical": training["both_branch_initial_states_identical"],
        "shared_state_sha256": training["shared_state_sha256"],
        "audit_sha256": training["audit_sha256"],
        "sample_overrides_sha256": training["sample_overrides_sha256"],
        "strict_gpu_trajectory_tolerance_passed": False,
        "gpu_repeatability_report_sha256": digest(WORK / "gpu_resume_repeatability_comparison.json"),
        "interruption_recovery": json.loads((WORK / "evaluation_interruption_recovery.json").read_text()),
        "evaluations": summary["evaluations"],
        "source_file_sha256": summary["source_file_sha256"],
        "exported_metrics_sha256": {kind: digest(ASSETS / f"{kind}_metrics.csv") for kind in metrics},
    }
    with (ASSETS / "result_verification.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")
    psnr = {(row["label"], row["unit"]): row["delta_from_a3"] for row in summary["native_metrics"] if row["metric"] == "finetuned.psnr_db"}
    rows = [
        "EMA 为预先指定的主候选，model 只用于诊断。以下为各权重相对同协议 A3 EMA 的原生 PSNR 变化（dB）。",
        "", "| 权重 | 0–7 | 96–103 | 192–199 |", "| --- | ---: | ---: | ---: |",
    ]
    for label in ("shared512_ema", "shared512_model", "fixed1024_ema", "fixed1024_model", "refresh1024_ema", "refresh1024_model"):
        rows.append(f"| {label} | " + " | ".join(f"{psnr[label, unit]:+.5f}" for unit in ("0-7", "96-103", "192-199")) + " |")
    rows.extend([
        "", "公共阶段 EMA 已有 native 回退；shared 与 fixed 的 model 也低于 A3 和各自 EMA。但刷新组 model 的三个窗口 PSNR 比 A3 高 0.05729–0.08772 dB，比固定组 model 高 0.56761–0.65579 dB。model 与 EMA 的结果确有差异，不能将 EMA 的静态重建回退外推为所有权重均无改善。",
        "", "刷新组 model 的静态收益仍伴随代价：相对 A3，三个窗口平坦区残差 σ 增加 0.18357–0.21145 DN，时序 MAD 增加 0.09275–0.09985 DN，高频幅度比下降约 0.01008–0.01050。四个局部运动案例的重建、高通、梯度和时序误差全部更高，重建增加 9.089–13.839 DN。高频幅度或 MAD 不能单独证明模糊，但总体重建、平坦区与运动指标并未同时改善。",
        "", "两组最终 model 的运动重建误差均有 6/8 高于 A3。预先指定的 EMA 主候选保持不变，model 的 native PSNR 收益完整记录为诊断结果，不能以更换评分权重的方式认定联合验收通过。",
        "", "刷新 model 相对固定 model 的重建误差有 7/12 降低，但高通与梯度误差均有 8/12 升高，不能用单项重建收益替代联合验收。",
    ])
    native_rows = [row for row in summary["native_metrics"] if row["metric"] == "finetuned.psnr_db"]
    for row in native_rows:
        if row["label"] in {"shared512_model", "fixed1024_model"}:
            ema_label = row["label"].removesuffix("_model") + "_ema"
            assert row["delta_from_a3"] < psnr[ema_label, row["unit"]] < 0
        elif row["label"] == "refresh1024_model":
            assert row["delta_from_a3"] > 0 > psnr["refresh1024_ema", row["unit"]]
        else:
            pass
    doc_path = ROOT / "docs/A12_PAIRED_SAMPLING_RESULT_2026-10-30.md"
    document = doc_path.read_text()
    old = "补充 native 评估进行中；完成后写入本节。EMA 始终为预先指定的主候选，model 只诊断短程 EMA 滞后。"
    assert old in document
    document = document.replace(old, "\n".join(rows))
    review = """七套权重均完成 12 个探针和三个原生窗口，共 105 个预测。汇总器核对检查点与预测哈希、精确 model/EMA 身份、3090 UUID 与运行时、有限 FP32 数组、窗口输入及掩码、参考协议和评分区域；全部通过。相关采样、训练、评估测试 66 项通过，汇总器测试另 6 项通过。

最终评估曾因会话中断留下部分 refresh EMA native 输出。确认原进程已退出且无 OOM 记录后，将部分输出和原 status 原样保存在 `native_interrupted_20261030/`、`status_native_interrupted_20261030.json`，按同协议重新评完三个窗口；没有重启训练。完成批次的状态为 `evaluation_completion_status.json`，原中断批次的状态文件保留作历史记录。

- [版本化指标 CSV：探针](assets/a12_paired_20261030/probe_metrics.csv)、[原生窗口](assets/a12_paired_20261030/native_metrics.csv)：全部七套权重的关键指标及双对照差值，未跨案例平均。
- [版本化复核与身份摘要](assets/a12_paired_20261030/result_verification.json)：检查点、审计、运行时、源文件和导出指标的哈希及中断保留记录。
- [完整汇总](../artifacts/a12_paired_summary_20261030/summary.json)、[训练核验](../artifacts/a12_paired_work_20261030/final_training_verification.json)、[GPU 重复性诊断](../artifacts/a12_paired_work_20261030/gpu_resume_repeatability_comparison.json)。
- [独立主结果审查](../artifacts/a12_paired_work_20261030/review_20261030.json)、[原生主结果审查](../artifacts/a12_paired_work_20261030/native_primary_analysis.json)、[最终 model 权重审查](../artifacts/a12_paired_work_20261030/final_model_analysis.json)。
- 正式检查点位于 `artifacts/a12_paired_shared_20261030/checkpoint_step_000512.pth`、`artifacts/a12_paired_fixed_20261030/checkpoint_step_001024.pth`、`artifacts/a12_paired_refresh_20261030/checkpoint_step_001024.pth`；完整 SHA-256 见版本化身份摘要。

汇总复现：

```bash
/home/jingzeliang/miniconda3/envs/aaa_312/bin/python scripts/summarize_a12_paired.py \\
  --manifest configs/a12_paired_evaluation_20261030.json \\
  --output-dir artifacts/a12_paired_summary_recheck_20261030
```

输出目录必须尚不存在；已有评估与结果不会覆盖。图形复现入口为 `docs/assets/a12_paired_20261030/make_probe_figure.py`。"""
    assert "最终检查清单及补充指标待评估完成后填入。" in document
    document = document.replace("最终检查清单及补充指标待评估完成后填入。", review)
    doc_path.write_text(document)
    print(json.dumps({"status": "document_finalized", "probe_rows": 336, "native_rows": 126}))


if __name__ == "__main__":
    main()
