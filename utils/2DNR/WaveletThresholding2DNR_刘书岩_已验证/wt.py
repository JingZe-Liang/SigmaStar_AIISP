import numpy as np
import pywt
from numba import njit, prange
from typing import Tuple, Optional
import warnings


@njit(parallel=True, cache=True, fastmath=True)
def _soft_threshold(coeff: np.ndarray, threshold: float) -> np.ndarray:
    out = np.empty_like(coeff, dtype=np.float32)
    for i in prange(coeff.shape[0]):
        for j in range(coeff.shape[1]):
            val = coeff[i, j]
            if val > threshold:
                out[i, j] = val - threshold
            elif val < -threshold:
                out[i, j] = val + threshold
            else:
                out[i, j] = 0.0
    return out


@njit(parallel=True, cache=True, fastmath=True)
def _hard_threshold(coeff: np.ndarray, threshold: float) -> np.ndarray:
    out = np.empty_like(coeff, dtype=np.float32)
    for i in prange(coeff.shape[0]):
        for j in range(coeff.shape[1]):
            val = coeff[i, j]
            out[i, j] = val if abs(val) > threshold else 0.0
    return out


def calculate_threshold(coeff: np.ndarray, method: str = "visu", factor: float = 1.0) -> float:
    coeff_flat = coeff.flatten()
    n = len(coeff_flat)
    sigma = np.median(np.abs(coeff_flat)) / 0.6745
    if method == "visu":
        threshold = sigma * np.sqrt(2 * np.log(n))
    elif method == "stein":
        threshold = sigma * np.sqrt(np.log(n))
    elif method == "sqtwolog":
        threshold = sigma * np.sqrt(np.log(n) / np.log(2))
    return threshold * factor


def _bayes_shrink_threshold(coeff: np.ndarray) -> float:
    sigma = np.median(np.abs(coeff)) / 0.6745
    if sigma == 0:
        return 0.0
    var_coeff = np.var(coeff)
    sigma_x = np.sqrt(max(var_coeff - sigma**2, 0.0))
    threshold = sigma**2 / sigma_x if sigma_x > 0 else np.inf
    return threshold


class WaveletThresholding:
    def __init__(
        self,
        img: np.ndarray,
        wavelet: str = "db4",
        level: int = 3,
        threshold_method: str = "visu",
        threshold_factor: float = 1.0,
        threshold_type: str = "soft",
        clip: int = 4095,
        cfa_pattern: str = "GBRG",
        use_swt: bool = True
    ):
        self.img = img.astype(np.float32)
        self.wavelet = wavelet
        self.level = level
        self.threshold_method = threshold_method
        self.threshold_factor = threshold_factor
        self.threshold_type = threshold_type.lower()
        self.clip = clip
        self.cfa_pattern = cfa_pattern.upper()
        self.use_swt = use_swt
        
        self._cfa_pos = self._get_cfa_positions()

    def _get_cfa_positions(self) -> dict:
        mapping = {
            "RGGB": {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)},
            "BGGR": {"B": (0, 0), "G1": (0, 1), "G2": (1, 0), "R": (1, 1)},
            "GBRG": {"G1": (0, 0), "B": (0, 1), "R": (1, 0), "G2": (1, 1)},
            "GRBG": {"G1": (0, 0), "R": (0, 1), "B": (1, 0), "G2": (1, 1)},
        }
        return mapping[self.cfa_pattern]

    def _bayer_to_4planes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h, w = self.img.shape
        planes = {}
        for name, (ry, rx) in self._cfa_pos.items():
            planes[name] = self.img[ry::2, rx::2].copy()
        return planes["R"], planes["G1"], planes["G2"], planes["B"]

    def _4planes_to_bayer(self, r_plane, g1_plane, g2_plane, b_plane) -> np.ndarray:
        h, w = self.img.shape
        out = np.empty((h, w), dtype=np.float32)
        for name, (ry, rx) in self._cfa_pos.items():
            if name == "R":
                out[ry::2, rx::2] = r_plane
            elif name == "G1":
                out[ry::2, rx::2] = g1_plane
            elif name == "G2":
                out[ry::2, rx::2] = g2_plane
            elif name == "B":
                out[ry::2, rx::2] = b_plane
        return out

    # ---------- 修改点：_denoise_plane_swt 增加尺寸安全检查 ----------
    def _denoise_plane_swt(self, plane: np.ndarray) -> np.ndarray:
        h, w = plane.shape
        # 计算 SWT 最大允许分解层数（基于最小边长）
        max_swt_level = pywt.swt_max_level(min(h, w))
        
        actual_level = min(self.level, max_swt_level)

        # 执行 SWT 分解
        coeffs = pywt.swt2(plane, self.wavelet, level=actual_level)

        denoised_coeffs = []
        for cA, (cH, cV, cD) in coeffs:
            thr_H = _bayes_shrink_threshold(cH) * self.threshold_factor
            thr_V = _bayes_shrink_threshold(cV) * self.threshold_factor
            thr_D = _bayes_shrink_threshold(cD) * self.threshold_factor

            if self.threshold_type == "soft":
                cH = _soft_threshold(cH.astype(np.float32), thr_H)
                cV = _soft_threshold(cV.astype(np.float32), thr_V)
                cD = _soft_threshold(cD.astype(np.float32), thr_D)
            else:
                cH = _hard_threshold(cH.astype(np.float32), thr_H)
                cV = _hard_threshold(cV.astype(np.float32), thr_V)
                cD = _hard_threshold(cD.astype(np.float32), thr_D)

            denoised_coeffs.append((cA, (cH, cV, cD)))

        denoised = pywt.iswt2(denoised_coeffs, self.wavelet)
        return denoised

    # ---------- DWT 去噪 ----------
    def _denoise_plane_dwt(self, plane: np.ndarray) -> np.ndarray:
        coeffs = pywt.wavedec2(plane, self.wavelet, level=self.level)
        cA = coeffs[0]
        cD_list = coeffs[1:]

        denoised_cD = []
        for cD in cD_list:
            cH, cV, cD = cD
            threshold = calculate_threshold(
                np.concatenate([cH.flatten(), cV.flatten(), cD.flatten()]),
                method=self.threshold_method,
                factor=self.threshold_factor
            )
            if self.threshold_type == "soft":
                cH = _soft_threshold(cH.astype(np.float32), threshold)
                cV = _soft_threshold(cV.astype(np.float32), threshold)
                cD = _soft_threshold(cD.astype(np.float32), threshold)
            else:
                cH = _hard_threshold(cH.astype(np.float32), threshold)
                cV = _hard_threshold(cV.astype(np.float32), threshold)
                cD = _hard_threshold(cD.astype(np.float32), threshold)
            denoised_cD.append((cH, cV, cD))

        denoised_plane = pywt.waverec2([cA] + denoised_cD, self.wavelet)
        denoised_plane = denoised_plane[:plane.shape[0], :plane.shape[1]]
        return denoised_plane

    def _denoise_plane(self, plane: np.ndarray) -> np.ndarray:
        if self.use_swt:
            return self._denoise_plane_swt(plane)
        else:
            return self._denoise_plane_dwt(plane)

    def _clipping(self, img: np.ndarray) -> np.ndarray:
        img = np.clip(np.rint(img), 0, self.clip)
        return img.astype(np.uint16)

    def execute(self) -> np.ndarray:
        r_plane, g1_plane, g2_plane, b_plane = self._bayer_to_4planes()
        r_denoised = self._denoise_plane(r_plane)
        g1_denoised = self._denoise_plane(g1_plane)
        g2_denoised = self._denoise_plane(g2_plane)
        b_denoised = self._denoise_plane(b_plane)
        denoised_bayer = self._4planes_to_bayer(r_denoised, g1_denoised, g2_denoised, b_denoised)
        return self._clipping(denoised_bayer)