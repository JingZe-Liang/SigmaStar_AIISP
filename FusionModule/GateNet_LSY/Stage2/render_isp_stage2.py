from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


STAGE2_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = STAGE2_ROOT.parents[2]
PHASE2_ROOT = WORKSPACE_ROOT / "Phase2"
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

import render_isp_video as phase2_isp  # noqa: E402


def _shoulder(linear: np.ndarray, exposure: float, white_point: float) -> np.ndarray:
    scaled = np.maximum(linear * exposure, 0.0)
    mapped = scaled * (1.0 + scaled / (white_point * white_point)) / (1.0 + scaled)
    return np.clip(mapped, 0.0, 1.0)


class HighlightSafeISP(phase2_isp.FixedSequenceISP):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.white_point = 2.0

    def process_12bit(self, raw: np.ndarray) -> np.ndarray:
        display = np.power(
            _shoulder(self.linear_bgr(raw), self.exposure, self.white_point),
            1.0 / self.gamma,
        )
        return np.clip(np.rint(display * 4095.0), 0, 4095).astype("<u2")


def calibrate_highlight_safe(
    sequence,
    isp: HighlightSafeISP,
    frame_indices: list[int],
    *,
    spatial_step: int = 12,
) -> tuple[float, float]:
    reader = phase2_isp.RawStreamReader(sequence.denoised)
    samples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for frame_index in frame_indices:
        linear = isp.linear_bgr(reader.read_frame(frame_index))[::spatial_step, ::spatial_step]
        reference = cv2.imread(str(sequence.denoised_pngs[frame_index]), cv2.IMREAD_COLOR)
        if reference is None:
            raise FileNotFoundError(sequence.denoised_pngs[frame_index])
        reference = reference[::spatial_step, ::spatial_step].astype(np.float32) / 255.0
        samples.append((linear, reference, float((reference >= 254.0 / 255.0).mean())))

    best_score = float("inf")
    best_error = float("inf")
    best_exposure = 1.0
    best_white_point = 2.0
    for exposure in np.geomspace(0.25, 16.0, 73):
        for white_point in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
            errors: list[float] = []
            saturation_penalties: list[float] = []
            for linear, reference, reference_saturation in samples:
                rendered = np.power(
                    _shoulder(linear, float(exposure), white_point),
                    1.0 / isp.gamma,
                )
                errors.append(float(np.mean(np.abs(rendered - reference))))
                rendered_saturation = float((rendered >= 254.0 / 255.0).mean())
                saturation_penalties.append(
                    max(rendered_saturation - reference_saturation - 0.002, 0.0)
                )
            error = float(np.mean(errors))
            score = error + 2.0 * float(np.mean(saturation_penalties))
            if score < best_score:
                best_score = score
                best_error = error
                best_exposure = float(exposure)
                best_white_point = float(white_point)
    isp.white_point = best_white_point
    return best_exposure, best_error


def main() -> int:
    phase2_isp.FixedSequenceISP = HighlightSafeISP
    phase2_isp.calibrate_exposure = calibrate_highlight_safe
    result = phase2_isp.main()

    # The Phase2 renderer owns video verification; append Stage2 tone metadata.
    input_index = sys.argv.index("--input") + 1 if "--input" in sys.argv else None
    input_root = (
        Path(sys.argv[input_index])
        if input_index is not None
        else PHASE2_ROOT / "DERIVED" / "inference_final_all"
    )
    sequence_ids = ["128x", "645x"]
    if "--sequences" in sys.argv:
        start = sys.argv.index("--sequences") + 1
        sequence_ids = []
        for value in sys.argv[start:]:
            if value.startswith("--"):
                break
            sequence_ids.append(value)
    for sequence_id in sequence_ids:
        summary_path = input_root / sequence_id / "isp_summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["tone_mapping"] = "extended Reinhard highlight shoulder"
        summary["highlight_safe_calibration"] = True
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
