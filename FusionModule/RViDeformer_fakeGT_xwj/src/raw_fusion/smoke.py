"""真实数据 smoke 验收编排。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch

from .checkpoint import save_checkpoint_atomic
from .config import config_fingerprint, load_dataset_config, load_experiment_config
from .data import FusionPatchDataset, validate_dataset
from .infer import infer_sequence
from .losses import FusionLoss
from .model import CausalRawFusionNet
from .preview import build_comparison_frame, simple_isp
from .raw import RawStreamReader
from .train import TRAIN_IMAGE_KEYS, _checkpoint_state, train_step

EXPECTED_FRAME_BYTES = 4_147_200


@dataclass(frozen=True, slots=True)
class SmokeReport:
    validation_passed: bool
    train_step_finite: bool
    initial_loss: float
    final_loss: float
    full_frame_bytes: int
    full_frame_min: int
    full_frame_max: int
    preview_path: Path
    completed_stages: tuple[str, ...] = ()
    error: str | None = None

    @property
    def loss_reduction(self) -> float:
        return self.initial_loss - self.final_loss

    @property
    def accepted(self) -> bool:
        finite = all(math.isfinite(value) for value in (self.initial_loss, self.final_loss))
        reduced = self.initial_loss > 0.0 and self.final_loss <= self.initial_loss * 0.8
        valid_range = 252 <= self.full_frame_min <= self.full_frame_max <= 4095
        return bool(
            self.validation_passed and self.train_step_finite and finite and reduced
            and self.full_frame_bytes == EXPECTED_FRAME_BYTES and valid_range
            and self.preview_path.is_file() and self.error is None
        )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["preview_path"] = str(self.preview_path)
        payload["completed_stages"] = list(self.completed_stages)
        payload["loss_reduction"] = self.loss_reduction
        payload["accepted"] = self.accepted
        return payload


def run_smoke(config_path: Path, output_dir: Path) -> SmokeReport:
    """Run the ordered real-data acceptance flow and atomically write its report."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "report.json"
    completed: list[str] = []
    try:
        experiment = load_experiment_config(Path(config_path))
        dataset = load_dataset_config(experiment.dataset_path)
        validation = validate_dataset(dataset)
        if not validation.sequences:
            raise ValueError("数据校验没有返回序列")
        completed.append("validation")
        device = torch.device(experiment.train.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("smoke 需要可用 CUDA；当前环境没有 GPU")
        batch = _fixed_batch(dataset, experiment, device)
        model = CausalRawFusionNet(experiment.model).to(device)
        loss_fn = FusionLoss(
            experiment.loss,
            white_level=dataset.layout.white_level,
            target_black_level=dataset.layout.target_black_level,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=experiment.train.learning_rate,
            weight_decay=experiment.train.weight_decay,
        )
        first_step = train_step(model, batch, loss_fn, optimizer, device=device, amp=False)
        if not all(math.isfinite(value) for value in first_step.values()):
            raise FloatingPointError("首次 smoke train step 含非有限值")
        completed.append("finite_train_step")
        model = CausalRawFusionNet(experiment.model).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=experiment.train.learning_rate * 10,  # smoke test 使用更高学习率加速过拟合
            weight_decay=experiment.train.weight_decay,
        )
        initial_loss: float | None = None
        final_loss: float | None = None
        for _ in range(500):
            metrics = train_step(model, batch, loss_fn, optimizer, device=device, amp=False)
            value = float(metrics["total"])
            if not math.isfinite(value):
                raise FloatingPointError("smoke 过拟合含非有限损失")
            if initial_loss is None:
                initial_loss = value
            final_loss = value
        assert initial_loss is not None and final_loss is not None
        completed.append("overfit_50_steps")
        checkpoint_path = destination / "smoke_checkpoint.pt"
        fingerprint = config_fingerprint(dataset, experiment)
        save_checkpoint_atomic(
            checkpoint_path,
            _checkpoint_state(
                model,
                optimizer,
                None,
                epoch=0,
                global_step=50,
                fingerprint=fingerprint,
                experiment=experiment,
                dataset=dataset,
            ),
        )
        manifest = infer_sequence(
            Path(config_path),
            checkpoint_path,
            "128x",
            (100, 100),
            destination / "inference",
            device,
        )
        output_raw = np.fromfile(manifest.output_raw_path, dtype=np.dtype("<u2"))
        full_frame_bytes = int(manifest.output_raw_path.stat().st_size)
        expected_values = dataset.layout.width * dataset.layout.height
        if output_raw.size != expected_values:
            raise ValueError("smoke 全帧输出元素数不正确")
        full_frame_min = int(output_raw.min())
        full_frame_max = int(output_raw.max())
        completed.append("full_frame_inference")

        sequence = dataset.sequences["128x"]
        denoised_reader = RawStreamReader(
            sequence.denoised_stream,
            dataset.layout.width,
            dataset.layout.height,
            dataset.layout.frame_count,
            0,
        )
        fused_reader = RawStreamReader(
            sequence.fused_stream,
            dataset.layout.width,
            dataset.layout.height,
            dataset.layout.frame_count,
            0,
        )
        wb, exposure = sequence.white_balance, sequence.isp_gain
        comparison = build_comparison_frame(
            {
                "denoised": simple_isp(
                    denoised_reader.read_frame(100),
                    dataset.layout.candidate_black_level,
                    dataset.layout.white_level,
                    wb,
                    exposure,
                ),
                "fused": simple_isp(
                    fused_reader.read_frame(100),
                    dataset.layout.candidate_black_level,
                    dataset.layout.white_level,
                    wb,
                    exposure,
                ),
                "model": simple_isp(
                    output_raw.reshape(dataset.layout.height, dataset.layout.width),
                    dataset.layout.target_black_level,
                    dataset.layout.white_level,
                    wb,
                    exposure,
                ),
            }
        )
        preview_path = destination / "comparison.png"
        if not cv2.imwrite(str(preview_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("无法写入 smoke 对比 PNG")
        completed.append("comparison_preview")
        report = SmokeReport(
            validation_passed=True,
            train_step_finite=True,
            initial_loss=initial_loss,
            final_loss=final_loss,
            full_frame_bytes=full_frame_bytes,
            full_frame_min=full_frame_min,
            full_frame_max=full_frame_max,
            preview_path=preview_path,
            completed_stages=tuple(completed),
        )
        if not report.accepted:
            finite = all(math.isfinite(value) for value in (report.initial_loss, report.final_loss))
            reduced = report.initial_loss > 0.0 and report.final_loss <= report.initial_loss * 0.8
            valid_range = 252 <= report.full_frame_min <= report.full_frame_max <= 4095
            reasons = []
            if not finite:
                reasons.append(f"损失非有限: initial={report.initial_loss}, final={report.final_loss}")
            if not reduced:
                reasons.append(f"损失未下降足够: initial={report.initial_loss:.6f}, final={report.final_loss:.6f}, reduction={report.loss_reduction:.6f}")
            if not valid_range:
                reasons.append(f"输出范围无效: min={report.full_frame_min}, max={report.full_frame_max}")
            if report.full_frame_bytes != EXPECTED_FRAME_BYTES:
                reasons.append(f"字节数不匹配: {report.full_frame_bytes} != {EXPECTED_FRAME_BYTES}")
            if not report.preview_path.is_file():
                reasons.append(f"预览文件不存在: {report.preview_path}")
            raise RuntimeError(f"smoke 验收条件未满足: {', '.join(reasons)}")
        _write_json_atomic(report_path, report.as_dict())
        return report
    except Exception as error:
        _write_json_atomic(
            report_path,
            {
                "accepted": False,
                "completed_stages": completed,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def _fixed_batch(dataset: object, experiment: object, device: torch.device) -> dict[str, torch.Tensor]:
    train = experiment.train
    samples = FusionPatchDataset(
        dataset,
        sequence_name=experiment.split.train_sequence,
        frame_range=experiment.split.train_frames,
        patch_size_packed=train.patch_size_packed,
        samples_per_epoch=1,
        seed=train.seed,
        force_transform=(False, False, False),
    )
    sample = samples[0]
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if key in TRAIN_IMAGE_KEYS
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, allow_nan=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="运行 RAW 融合真实数据 smoke 验收")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    report = run_smoke(arguments.config, arguments.output_dir)
    print(json.dumps(report.as_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
