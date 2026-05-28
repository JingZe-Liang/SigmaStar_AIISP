# -*- coding: utf-8 -*-
"""RAW motion detection.

The default mode keeps the original MOG2-based detector.  The robust mode is
intended for noisy high-ISO RAW TIFF sequences and uses a small set of classical
image-processing steps:

1. Normalize RAW and use the Bayer green plane as the detection image.
2. Build a median background from sampled frames.
3. Compensate global brightness flicker before background differencing.
4. Normalize differences by robust local noise, then use seed/grow hysteresis.
5. Reject unsupported small components and smooth masks over one or two frames.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass
class DetectionResult:
    frame_idx: int
    mask: np.ndarray
    bboxes: List[BBox]
    display: np.ndarray
    noise_sigma: Optional[float] = None
    effective_min_area: Optional[int] = None
    raw_bbox_count: int = 0
    confirmed_bbox_count: int = 0
    global_shift: Optional[Tuple[float, float]] = None
    global_motion_response: Optional[float] = None


@dataclass
class MDConfig:
    mode: str = "robust"
    cfa_pattern: str = "GBRG"

    # Original MOG2 path.
    history: int = 200
    var_threshold: float = 25.0
    detect_shadows: bool = False
    learning_rate: float = 0.01
    min_area: int = 250
    blur_kernel: Tuple[int, int] = (5, 5)
    morph_kernel: Tuple[int, int] = (3, 3)

    # RAW range.
    black_level: float = 16.0
    white_level: float = 4095.0
    detection_white_level: Optional[float] = None

    # Robust path.
    noise_sigma: Optional[float] = None
    warmup_frames: int = 0
    temporal_consistency: int = 1
    robust_low_threshold: float = 2.0
    robust_high_threshold: float = 3.5
    robust_min_seed_area: int = 45
    robust_noise_kernel: Tuple[int, int] = (15, 15)
    robust_close_kernel: Tuple[int, int] = (13, 13)
    robust_support_kernel: Tuple[int, int] = (41, 41)
    robust_short_threshold: float = 1.45
    robust_max_motion_fraction: float = 0.55
    robust_fill_holes: bool = True
    robust_background_init_samples: int = 25
    robust_background_lr: float = 0.025
    robust_motion_lr: float = 0.003
    robust_hold_frames: int = 1
    robust_flicker_compensation: bool = True
    robust_flicker_scale_clip: float = 0.25
    robust_flicker_offset_clip: float = 0.06
    robust_static_history: int = 60
    robust_static_min_matches: int = 2
    robust_static_max_area: int = 9000
    robust_static_center_distance: float = 24.0
    robust_static_iou_threshold: float = 0.10

    compensate_global_motion: bool = True
    global_motion_max_shift: float = 5.0
    global_motion_min_response: float = 0.35

    min_bbox_width: int = 3
    min_bbox_height: int = 3
    max_aspect_ratio: float = 12.0


class MotionDetector:
    def __init__(self, config: Optional[MDConfig] = None):
        self.config = config or MDConfig()
        self._validate_config()
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows,
        )
        self._kernel = np.ones(self.config.morph_kernel, np.uint8)
        self._close_kernel = np.ones(self.config.robust_close_kernel, np.uint8)
        self._support_kernel = np.ones(self.config.robust_support_kernel, np.uint8)
        self._frame_count = 0
        self._background: Optional[np.ndarray] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._sigma_history: Deque[float] = deque(maxlen=5)
        self._bbox_history: Deque[List[BBox]] = deque(maxlen=max(1, self.config.temporal_consistency - 1))
        self._stationary_history: Deque[List[BBox]] = deque(maxlen=max(1, self.config.robust_static_history))
        self._mask_history: Deque[np.ndarray] = deque(maxlen=max(1, self.config.robust_hold_frames))
        self._last_shift = (0.0, 0.0)
        self._last_response = 0.0

    def _validate_config(self) -> None:
        if self.config.mode not in {"default", "robust"}:
            raise ValueError(f"Unsupported MD mode: {self.config.mode}")
        if self.config.white_level <= self.config.black_level:
            raise ValueError("white_level must be greater than black_level")
        if self.config.detection_white_level is not None and self.config.detection_white_level <= self.config.black_level:
            raise ValueError("detection_white_level must be greater than black_level")
        if self.config.robust_high_threshold < self.config.robust_low_threshold:
            raise ValueError("robust_high_threshold must be >= robust_low_threshold")
        for name, kernel in {
            "blur_kernel": self.config.blur_kernel,
            "morph_kernel": self.config.morph_kernel,
            "robust_noise_kernel": self.config.robust_noise_kernel,
            "robust_close_kernel": self.config.robust_close_kernel,
            "robust_support_kernel": self.config.robust_support_kernel,
        }.items():
            if len(kernel) != 2 or kernel[0] <= 0 or kernel[1] <= 0:
                raise ValueError(f"{name} must contain two positive integers")
        if self.config.robust_noise_kernel[0] % 2 == 0 or self.config.robust_noise_kernel[1] % 2 == 0:
            raise ValueError("robust_noise_kernel must have odd dimensions")

    def initialize(self, frames: Optional[np.ndarray]) -> None:
        """Initialize robust background from a median of sampled input frames."""
        if self.config.mode != "robust" or frames is None or len(frames) == 0:
            return
        sample_count = min(int(self.config.robust_background_init_samples), int(frames.shape[0]))
        if sample_count <= 1:
            return
        indices = np.linspace(0, frames.shape[0] - 1, sample_count).round().astype(int)
        stack = [self._preprocess(frames[i]) for i in indices]
        self._background = np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)
        self._prev_frame = self._preprocess(frames[0]).copy()

    def detect(self, raw: np.ndarray, frame_idx: Optional[int] = None) -> DetectionResult:
        seq_idx = self._frame_count
        self._frame_count += 1
        if frame_idx is None:
            frame_idx = seq_idx
        if self.config.mode == "default":
            return self._detect_default(raw, frame_idx)
        return self._detect_robust(raw, frame_idx, seq_idx)

    def reset(self) -> None:
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows,
        )
        self._frame_count = 0
        self._background = None
        self._prev_frame = None
        self._sigma_history.clear()
        self._bbox_history.clear()
        self._stationary_history.clear()
        self._mask_history.clear()
        self._last_shift = (0.0, 0.0)
        self._last_response = 0.0

    def _detect_default(self, raw: np.ndarray, frame_idx: int) -> DetectionResult:
        g = self._extract_g_u8(raw)
        fgmask = self._mog2.apply(cv2.GaussianBlur(g, self.config.blur_kernel, 0), learningRate=self.config.learning_rate)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self._kernel)
        fgmask = cv2.dilate(fgmask, self._kernel)
        bboxes = self._extract_bboxes(fgmask, self.config.min_area)
        display = self._draw_bboxes(g, bboxes)
        return DetectionResult(
            frame_idx=frame_idx,
            mask=fgmask,
            bboxes=bboxes,
            display=display,
            effective_min_area=self.config.min_area,
            raw_bbox_count=len(bboxes),
            confirmed_bbox_count=len(bboxes),
        )

    def _detect_robust(self, raw: np.ndarray, frame_idx: int, seq_idx: int) -> DetectionResult:
        g = self._raw_green(raw)
        sigma = self._estimate_sigma(g)
        current = self._denoise(g, sigma)
        display = self._display_image(current)

        if self._background is None:
            self._background = current.copy()
            self._prev_frame = current.copy()
            empty = np.zeros(display.shape, np.uint8)
            return self._result(frame_idx, display, empty, [], sigma, self.config.min_area)

        dx, dy, response = self._estimate_shift(self._prev_frame, current)
        self._last_shift, self._last_response = (dx, dy), response
        if self.config.compensate_global_motion:
            self._background = self._warp(self._background, dx, dy)
            prev = self._warp(self._prev_frame, dx, dy) if self._prev_frame is not None else None
        else:
            prev = self._prev_frame

        local_sigma = self._local_sigma(current, sigma)
        bg_diff = self._brightness_compensated_diff(current, self._background)
        score = np.abs(bg_diff) / (local_sigma + 1e-6)
        mask = self._hysteresis(score)

        rejected = np.zeros_like(mask)
        if prev is not None and np.any(mask):
            short_diff = self._brightness_compensated_diff(current, prev)
            short_score = np.abs(short_diff) / (local_sigma + 1e-6)
            gate, support = self._motion_support(short_score)
            mask, rejected = self._filter_by_support(mask, gate, support)

        if np.count_nonzero(mask) / float(mask.size) > self.config.robust_max_motion_fraction:
            rejected = cv2.bitwise_or(rejected, mask)
            mask = np.zeros_like(mask)

        mask = self._postprocess(mask)
        min_area = self.config.min_area
        raw_bboxes = self._extract_bboxes(mask, min_area)
        raw_bboxes, static_rejected = self._suppress_stationary_bboxes(raw_bboxes)
        if static_rejected:
            rejected = cv2.bitwise_or(rejected, self._mask_from_bboxes(mask, static_rejected))
            mask = self._mask_from_bboxes(mask, raw_bboxes)
        confirmed = self._confirm_bboxes(raw_bboxes, seq_idx < self.config.warmup_frames)
        out_mask = self._mask_from_bboxes(mask, confirmed)
        out_mask = self._temporal_hold(out_mask)
        output_bboxes = self._extract_bboxes(out_mask, min_area)

        update_mask = cv2.bitwise_or(mask, out_mask)
        self._update_background(current, update_mask, rejected)
        self._prev_frame = current.copy()
        self._mask_history.append(out_mask.copy())
        return self._result(
            frame_idx,
            self._draw_bboxes(display, output_bboxes),
            out_mask,
            output_bboxes,
            sigma,
            min_area,
            raw_count=len(raw_bboxes),
        )

    def _result(
        self,
        frame_idx: int,
        display: np.ndarray,
        mask: np.ndarray,
        bboxes: List[BBox],
        sigma: Optional[float],
        min_area: int,
        raw_count: int = 0,
    ) -> DetectionResult:
        return DetectionResult(
            frame_idx=frame_idx,
            mask=mask,
            bboxes=bboxes,
            display=display,
            noise_sigma=sigma,
            effective_min_area=min_area,
            raw_bbox_count=raw_count,
            confirmed_bbox_count=len(bboxes),
            global_shift=self._last_shift,
            global_motion_response=self._last_response,
        )

    def _normalize_raw(self, raw: np.ndarray) -> np.ndarray:
        white = float(self.config.detection_white_level or self.config.white_level)
        raw32 = raw.astype(np.float32, copy=False)
        out = (raw32 - float(self.config.black_level)) / (white - float(self.config.black_level))
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    def _preprocess(self, raw: np.ndarray) -> np.ndarray:
        g = self._raw_green(raw)
        return self._denoise(g, self._quick_sigma(g))

    def _raw_green(self, raw: np.ndarray) -> np.ndarray:
        return self._green(self._normalize_raw(raw)).astype(np.float32, copy=False)

    @staticmethod
    def _denoise(img: np.ndarray, sigma: float) -> np.ndarray:
        sigma_color = float(np.clip(max(sigma * 4.0, 0.012), 0.012, 0.08))
        return cv2.bilateralFilter(img.astype(np.float32, copy=False), d=5, sigmaColor=sigma_color, sigmaSpace=3.0)

    def _extract_g_u8(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(self._green(raw.astype(np.float32, copy=False)) / 16.0, 0, 255).astype(np.uint8)

    def _green(self, img: np.ndarray) -> np.ndarray:
        pattern = self.config.cfa_pattern.upper()
        if pattern in {"GBRG", "GRBG"}:
            g1, g2 = img[0::2, 0::2], img[1::2, 1::2]
        elif pattern in {"RGGB", "BGGR"}:
            g1, g2 = img[0::2, 1::2], img[1::2, 0::2]
        else:
            raise ValueError(f"Unsupported CFA pattern: {pattern}")
        return ((g1.astype(np.float32) + g2.astype(np.float32)) * 0.5).astype(img.dtype, copy=False)

    def _estimate_sigma(self, img: np.ndarray) -> float:
        if self.config.noise_sigma is not None:
            sigma = float(self.config.noise_sigma)
        else:
            sigma = self._quick_sigma(img)
        sigma = float(np.clip(sigma, 1e-5, 0.20))
        self._sigma_history.append(sigma)
        return float(np.median(np.asarray(self._sigma_history, dtype=np.float32)))

    @staticmethod
    def _quick_sigma(img: np.ndarray) -> float:
        residual = img - cv2.GaussianBlur(img, (3, 3), 0)
        med = float(np.median(residual))
        return 1.4826 * float(np.median(np.abs(residual - med)))

    def _local_sigma(self, img: np.ndarray, sigma: float) -> np.ndarray:
        residual = np.abs(img - cv2.GaussianBlur(img, (3, 3), 0))
        local = 1.2533 * cv2.GaussianBlur(residual, self.config.robust_noise_kernel, 0)
        return np.maximum(local, max(sigma, 1e-5)).astype(np.float32, copy=False)

    def _brightness_compensated_diff(self, current: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if not self.config.robust_flicker_compensation:
            return current - reference
        cur = current[::4, ::4].reshape(-1)
        ref = reference[::4, ::4].reshape(-1)
        cur_p10, cur_med, cur_p90 = np.percentile(cur, (10, 50, 90))
        ref_p10, ref_med, ref_p90 = np.percentile(ref, (10, 50, 90))
        ref_span = float(ref_p90 - ref_p10)
        scale = 1.0 if ref_span <= 1e-6 else float(cur_p90 - cur_p10) / ref_span
        scale_clip = float(self.config.robust_flicker_scale_clip)
        scale = float(np.clip(scale, 1.0 - scale_clip, 1.0 + scale_clip))
        offset = float(cur_med - scale * ref_med)
        offset = float(np.clip(offset, -self.config.robust_flicker_offset_clip, self.config.robust_flicker_offset_clip))
        return current - (reference * scale + offset)

    def _hysteresis(self, score: np.ndarray) -> np.ndarray:
        seed = score >= float(self.config.robust_high_threshold)
        candidate = score >= float(self.config.robust_low_threshold)
        if not np.any(seed) or not np.any(candidate):
            return np.zeros(score.shape, np.uint8)
        count, labels, _, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
        seed_count = np.bincount(labels[seed].ravel(), minlength=count)
        keep = seed_count >= int(self.config.robust_min_seed_area)
        keep[0] = False
        return np.where(keep[labels], 255, 0).astype(np.uint8)

    def _motion_support(self, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gate = np.where(score >= float(self.config.robust_short_threshold), 255, 0).astype(np.uint8)
        gate = cv2.morphologyEx(gate, cv2.MORPH_OPEN, self._kernel)
        gate = cv2.morphologyEx(gate, cv2.MORPH_CLOSE, self._kernel)
        return gate, cv2.dilate(gate, self._support_kernel)

    def _filter_by_support(self, mask: np.ndarray, gate: np.ndarray, support: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        if count <= 1:
            return mask, np.zeros_like(mask)
        gate_overlap = np.bincount(labels[gate > 0].ravel(), minlength=count)
        support_overlap = np.bincount(labels[support > 0].ravel(), minlength=count)
        area = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
        keep = (
            (gate_overlap >= np.maximum(35.0, area * 0.012))
            & (support_overlap >= np.maximum(25.0, area * 0.02))
        )
        keep[0] = False
        kept = np.where(keep[labels], 255, 0).astype(np.uint8)
        rejected = cv2.bitwise_and(mask, cv2.bitwise_not(kept))
        return kept, rejected

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        if self.config.robust_fill_holes:
            mask = self._fill_holes(mask)
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        padded = np.zeros((h + 2, w + 2), np.uint8)
        padded[1:h + 1, 1:w + 1] = mask
        flood = padded.copy()
        cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
        return cv2.bitwise_or(padded, cv2.bitwise_not(flood))[1:h + 1, 1:w + 1]

    def _extract_bboxes(self, mask: np.ndarray, min_area: int) -> List[BBox]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes: List[BBox] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w / max(h, 1), h / max(w, 1))
            if w >= self.config.min_bbox_width and h >= self.config.min_bbox_height and (
                aspect <= self.config.max_aspect_ratio or area >= min_area * 2
            ):
                bboxes.append((x, y, w, h))
        return bboxes

    def _confirm_bboxes(self, bboxes: List[BBox], is_warmup: bool) -> List[BBox]:
        if is_warmup:
            self._bbox_history.clear()
            return []
        if self.config.temporal_consistency <= 1:
            self._bbox_history.append(bboxes)
            return bboxes
        confirmed: List[BBox] = []
        for bbox in bboxes:
            consecutive = 1
            for previous in reversed(self._bbox_history):
                if any(self._bbox_match(bbox, item) for item in previous):
                    consecutive += 1
                    if consecutive >= self.config.temporal_consistency:
                        confirmed.append(bbox)
                        break
                else:
                    break
        self._bbox_history.append(bboxes)
        return confirmed

    def _suppress_stationary_bboxes(self, bboxes: List[BBox]) -> Tuple[List[BBox], List[BBox]]:
        if not bboxes:
            self._stationary_history.append([])
            return [], []
        kept: List[BBox] = []
        rejected: List[BBox] = []
        for bbox in bboxes:
            x, y, w, h = bbox
            if w * h > int(self.config.robust_static_max_area):
                kept.append(bbox)
                continue
            matches = 0
            for previous in self._stationary_history:
                if any(self._stationary_match(bbox, item) for item in previous):
                    matches += 1
            if matches >= int(self.config.robust_static_min_matches):
                rejected.append(bbox)
            else:
                kept.append(bbox)
        self._stationary_history.append(bboxes)
        return kept, rejected

    def _temporal_hold(self, mask: np.ndarray) -> np.ndarray:
        if self.config.robust_hold_frames <= 0 or not self._mask_history:
            return mask
        out = mask.copy()
        if np.any(mask):
            support = cv2.dilate(mask, self._support_kernel)
            for old in self._mask_history:
                out = cv2.bitwise_or(out, cv2.bitwise_and(old, support))
        return self._postprocess(out)

    def _update_background(self, current: np.ndarray, motion_mask: np.ndarray, rejected_mask: np.ndarray) -> None:
        if self._background is None:
            self._background = current.copy()
            return
        stable_lr = float(np.clip(self.config.robust_background_lr, 0.0, 1.0))
        motion_lr = float(np.clip(self.config.robust_motion_lr, 0.0, 1.0))
        motion = motion_mask > 0
        rejected = rejected_mask > 0
        stable = ~motion
        self._background[stable] = (1.0 - stable_lr) * self._background[stable] + stable_lr * current[stable]
        if np.any(rejected):
            self._background[rejected] = (1.0 - stable_lr) * self._background[rejected] + stable_lr * current[rejected]
        moving = motion & ~rejected
        if np.any(moving) and motion_lr > 0:
            self._background[moving] = (1.0 - motion_lr) * self._background[moving] + motion_lr * current[moving]

    def _estimate_shift(self, previous: Optional[np.ndarray], current: np.ndarray) -> Tuple[float, float, float]:
        if not self.config.compensate_global_motion or previous is None:
            return 0.0, 0.0, 0.0
        prev = cv2.GaussianBlur(previous.astype(np.float32, copy=False), (5, 5), 0)
        curr = cv2.GaussianBlur(current.astype(np.float32, copy=False), (5, 5), 0)
        try:
            (dx, dy), response = cv2.phaseCorrelate(prev, curr)
        except cv2.error:
            return 0.0, 0.0, 0.0
        valid = (
            np.isfinite(dx)
            and np.isfinite(dy)
            and np.isfinite(response)
            and response >= float(self.config.global_motion_min_response)
            and abs(dx) <= float(self.config.global_motion_max_shift)
            and abs(dy) <= float(self.config.global_motion_max_shift)
        )
        return (float(dx), float(dy), float(response)) if valid else (0.0, 0.0, float(response) if np.isfinite(response) else 0.0)

    @staticmethod
    def _warp(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return img.copy()
        h, w = img.shape
        matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    @staticmethod
    def _display_image(img: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(img, (0.5, 99.7))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
        return np.clip((img - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    @staticmethod
    def _draw_bboxes(gray: np.ndarray, bboxes: Iterable[BBox]) -> np.ndarray:
        display = gray.copy()
        for x, y, w, h in bboxes:
            cv2.rectangle(display, (x, y), (x + w, y + h), 255, 2)
        return display

    @staticmethod
    def _bbox_match(a: BBox, b: BBox) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        inter_x1, inter_y1 = max(ax, bx), max(ay, by)
        inter_x2, inter_y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union = aw * ah + bw * bh - inter
        iou = inter / union if union > 0 else 0.0
        acx, acy = ax + aw * 0.5, ay + ah * 0.5
        bcx, bcy = bx + bw * 0.5, by + bh * 0.5
        center_dist = float(np.hypot(acx - bcx, acy - bcy))
        return iou >= 0.10 or center_dist <= max(20.0, 0.25 * max(aw, ah, bw, bh))

    def _stationary_match(self, a: BBox, b: BBox) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        inter_x1, inter_y1 = max(ax, bx), max(ay, by)
        inter_x2, inter_y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union = aw * ah + bw * bh - inter
        iou = inter / union if union > 0 else 0.0
        acx, acy = ax + aw * 0.5, ay + ah * 0.5
        bcx, bcy = bx + bw * 0.5, by + bh * 0.5
        center_dist = float(np.hypot(acx - bcx, acy - bcy))
        return (
            iou >= float(self.config.robust_static_iou_threshold)
            or center_dist <= float(self.config.robust_static_center_distance)
        )

    @staticmethod
    def _mask_from_bboxes(mask: np.ndarray, bboxes: Iterable[BBox]) -> np.ndarray:
        out = np.zeros_like(mask)
        for x, y, w, h in bboxes:
            out[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
        return out


def run_motion_detection(
    frame_stack: np.ndarray,
    output_dir: str,
    config: Optional[MDConfig] = None,
    fps: int = 24,
    save_masks: bool = True,
    save_bboxes: bool = True,
    save_video: bool = True,
    verbose: bool = False,
) -> List[DetectionResult]:
    config = config or MDConfig()
    detector = MotionDetector(config)
    detector.initialize(frame_stack)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mask_dir = output_path / "masks"
    bbox_dir = output_path / "bboxes"
    if save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if save_bboxes:
        bbox_dir.mkdir(parents=True, exist_ok=True)

    out_h, out_w = frame_stack.shape[1] // 2, frame_stack.shape[2] // 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    mask_writer = None
    bbox_writer = None
    overlay_writer = None
    if save_video:
        mask_writer = cv2.VideoWriter(str(output_path / "md_video.mp4"), fourcc, fps, (out_w, out_h), False)
        bbox_writer = cv2.VideoWriter(str(output_path / "bbox_video.mp4"), fourcc, fps, (out_w, out_h), False)
        overlay_writer = cv2.VideoWriter(str(output_path / "overlay_video.mp4"), fourcc, fps, (out_w, out_h), True)

    diag_path = output_path / "diagnostics.jsonl"
    results: List[DetectionResult] = []
    with diag_path.open("w", encoding="utf-8") as diag_file:
        for i in range(frame_stack.shape[0]):
            result = detector.detect(frame_stack[i], frame_idx=i)
            results.append(result)

            if save_masks:
                cv2.imwrite(str(mask_dir / f"{i:04d}.png"), result.mask)
            if save_bboxes:
                cv2.imwrite(str(bbox_dir / f"{i:04d}.png"), result.display)
            if mask_writer is not None:
                mask_writer.write(result.mask)
            if bbox_writer is not None:
                bbox_writer.write(result.display)
            if overlay_writer is not None:
                overlay_writer.write(_make_overlay(result.display, result.mask, result.bboxes))

            diag_file.write(json.dumps({
                "frame_idx": result.frame_idx,
                "noise_sigma": result.noise_sigma,
                "effective_min_area": result.effective_min_area,
                "raw_bbox_count": result.raw_bbox_count,
                "confirmed_bbox_count": result.confirmed_bbox_count,
                "global_shift": result.global_shift,
                "global_motion_response": result.global_motion_response,
                "bboxes": result.bboxes,
            }) + "\n")

            if verbose and (i + 1) % 10 == 0:
                print(f"[MD] processed {i + 1}/{frame_stack.shape[0]} frames")

    for writer in (mask_writer, bbox_writer, overlay_writer):
        if writer is not None:
            writer.release()
    if verbose:
        print(f"[MD] output: {output_dir}")
    return results


def _make_overlay(gray: np.ndarray, mask: np.ndarray, bboxes: Iterable[BBox]) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(base)
    red[..., 2] = 255
    overlay = np.where(mask[..., None] > 0, (0.55 * base + 0.45 * red).astype(np.uint8), base)
    for x, y, w, h in bboxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return overlay


def _natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _find_tiffs(input_dir: Path) -> List[Path]:
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}],
        key=_natural_key,
    )


def _load_tiff_stack(input_dir: Path) -> np.ndarray:
    try:
        import tifffile as tiff
    except ImportError as exc:
        raise RuntimeError("tifffile is required to load TIFF sequences") from exc
    files = _find_tiffs(input_dir)
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {input_dir}")
    frames = [tiff.imread(str(path)) for path in files]
    return np.stack(frames, axis=0)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run RAW motion detection on a TIFF sequence.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mode", choices=["default", "robust"], default="robust")
    parser.add_argument("--cfa_pattern", choices=["RGGB", "BGGR", "GBRG", "GRBG"], default="GBRG")
    parser.add_argument("--black_level", type=float, default=16.0)
    parser.add_argument("--white_level", type=float, default=4095.0)
    parser.add_argument("--detection_white_level", type=float, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--min_area", type=int, default=250)
    parser.add_argument("--low_threshold", type=float, default=2.0)
    parser.add_argument("--high_threshold", type=float, default=3.5)
    parser.add_argument("--temporal_consistency", type=int, default=1)
    parser.add_argument("--warmup_frames", type=int, default=0)
    parser.add_argument("--disable_global_motion", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    stack = _load_tiff_stack(args.input_dir)
    config = MDConfig(
        mode=args.mode,
        cfa_pattern=args.cfa_pattern,
        black_level=args.black_level,
        white_level=args.white_level,
        detection_white_level=args.detection_white_level,
        min_area=args.min_area,
        robust_low_threshold=args.low_threshold,
        robust_high_threshold=args.high_threshold,
        temporal_consistency=args.temporal_consistency,
        warmup_frames=args.warmup_frames,
        compensate_global_motion=not args.disable_global_motion,
    )
    run_motion_detection(
        frame_stack=stack,
        output_dir=str(args.output_dir),
        config=config,
        fps=args.fps,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
