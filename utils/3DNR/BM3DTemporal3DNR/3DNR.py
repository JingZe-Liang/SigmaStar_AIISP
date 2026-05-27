import argparse
import re
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from bm3d import bm3d
except ImportError:
    raise ImportError(
        "没有安装 bm3d，请先运行：\n"
        "python -m pip install bm3d opencv-python matplotlib pillow"
    )


# =========================================================
# 1) 读写
# =========================================================
def read_tiff(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img

    if Image is not None:
        with Image.open(path) as im:
            return np.array(im)

    raise RuntimeError(f"无法读取图像: {path}")


def save_tiff(path: Path, img_float: np.ndarray, ref_dtype: np.dtype) -> None:
    if np.issubdtype(ref_dtype, np.integer):
        info = np.iinfo(ref_dtype)
        out = np.clip(np.rint(img_float), info.min, info.max).astype(ref_dtype)
    else:
        out = img_float.astype(ref_dtype)

    ok = cv2.imwrite(str(path), out)
    if not ok:
        if Image is not None:
            Image.fromarray(out).save(str(path))
        else:
            raise RuntimeError(f"保存失败: {path}")


# =========================================================
# 2) 文件名解析
# frame5_noisy3 / frame5_clean
# =========================================================
def parse_crvd_name(stem: str):
    m_noisy = re.fullmatch(r"frame(\d+)_noisy(\d+)", stem, re.IGNORECASE)
    if m_noisy:
        return {
            "frame_id": int(m_noisy.group(1)),
            "kind": "noisy",
            "noisy_id": int(m_noisy.group(2)),
        }

    m_clean = re.fullmatch(r"frame(\d+)_clean", stem, re.IGNORECASE)
    if m_clean:
        return {
            "frame_id": int(m_clean.group(1)),
            "kind": "clean",
            "noisy_id": None,
        }

    return None


def find_matching_file(folder: Path, stem: str, prefer_suffix: str = None):
    candidates = []
    if prefer_suffix is not None:
        candidates.append(prefer_suffix)
    candidates.extend([".tiff", ".tif", ".TIFF", ".TIF"])

    seen = set()
    for ext in candidates:
        if ext in seen:
            continue
        seen.add(ext)
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_clean_file_for_noisy(target_path: Path):
    info = parse_crvd_name(target_path.stem)
    if info is None or info["kind"] != "noisy":
        return None

    folder = target_path.parent
    frame_id = info["frame_id"]
    clean_stem = f"frame{frame_id}_clean"
    return find_matching_file(folder, clean_stem, prefer_suffix=target_path.suffix)


# =========================================================
# 3) 收集同一 noisy_id 的时域序列（对称窗口）
# =========================================================
def collect_temporal_stack_symmetric(target_path: Path, radius: int = 3):
    info = parse_crvd_name(target_path.stem)
    if info is None or info["kind"] != "noisy":
        raise ValueError("输入必须是 noisy 图，例如 frame5_noisy3.tiff")

    folder = target_path.parent
    target_frame = info["frame_id"]
    noisy_id = info["noisy_id"]

    pat = re.compile(rf"frame(\d+)_noisy{noisy_id}$", re.IGNORECASE)
    frame_to_path = {}

    for fn in os.listdir(folder):
        p = Path(folder) / fn
        if not p.is_file():
            continue
        m = pat.fullmatch(p.stem)
        if m:
            fid = int(m.group(1))
            frame_to_path[fid] = p

    if target_frame not in frame_to_path:
        frame_to_path[target_frame] = target_path

    lo = target_frame - radius
    hi = target_frame + radius
    fids = [fid for fid in frame_to_path.keys() if lo <= fid <= hi]
    fids.sort()

    paths = [frame_to_path[fid] for fid in fids]
    t_index = fids.index(target_frame)
    return paths, t_index


# =========================================================
# 4) 基础工具
# =========================================================
def to_float01(img: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    img = img.astype(np.float32)
    denom = float(white_level - black_level)
    if denom <= 0:
        raise ValueError("white_level 必须大于 black_level")
    x = (img - black_level) / denom
    return np.clip(x, 0.0, 1.0)


def from_float01(x01: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    return x01.astype(np.float32) * float(white_level - black_level) + float(black_level)


def get_gray(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[2] == 1:
        return x[..., 0]
    if x.ndim == 3 and x.shape[2] == 3:
        return cv2.cvtColor(x.astype(np.float32), cv2.COLOR_BGR2GRAY)
    raise ValueError(f"不支持的图像形状: {x.shape}")


def estimate_sigma01_from_target(target01: np.ndarray) -> float:
    """
    用高通残差的 MAD 粗略估计噪声标准差（归一化到 [0,1] 后）
    """
    gray = get_gray(target01).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    resid = gray - blur
    mad = np.median(np.abs(resid - np.median(resid)))
    sigma = 1.4826 * mad
    sigma = float(np.clip(sigma, 1e-4, 0.25))
    return sigma


def warp_to_target(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    已知从 target -> frame 的 flow，把 frame warp 回 target 坐标
    """
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]

    if frame.ndim == 2:
        return cv2.remap(
            frame, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101
        )

    out = np.empty_like(frame)
    for c in range(frame.shape[2]):
        out[..., c] = cv2.remap(
            frame[..., c], map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101
        )
    return out


def build_aligned_stack_to_target(stack01: np.ndarray, t_index: int) -> np.ndarray:
    """
    用 Farneback 光流把所有帧对齐到目标帧
    stack01:
      灰度: [T,H,W]
      彩色: [T,H,W,C]
    """
    T = stack01.shape[0]
    target = stack01[t_index]
    target_gray = get_gray(target).astype(np.float32)

    aligned = []
    for t in range(T):
        cur = stack01[t]
        if t == t_index:
            aligned.append(cur.astype(np.float32))
            continue

        cur_gray = get_gray(cur).astype(np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            prev=target_gray,
            next=cur_gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        warped = warp_to_target(cur.astype(np.float32), flow)
        aligned.append(warped.astype(np.float32))

    return np.stack(aligned, axis=0)


# =========================================================
# 5) 时域融合
# 对齐后做自适应加权均值 + 中值混合
# =========================================================
def temporal_fusion(
    aligned_stack01: np.ndarray,
    t_index: int,
    sigma01: float,
    temporal_sigma_scale: float = 2.5,
    mean_weight: float = 0.7
) -> np.ndarray:
    """
    对齐后的多帧融合：
    - 和目标帧越接近，权重越大
    - 再和 temporal median 混合，增强稳健性
    """
    target = aligned_stack01[t_index].astype(np.float32)

    if aligned_stack01.ndim == 3:
        # 灰度 [T,H,W]
        diff = aligned_stack01 - target[None, ...]
        diff2 = diff * diff

        scale2 = (temporal_sigma_scale * sigma01) ** 2 + 1e-12
        weights = np.exp(-diff2 / (2.0 * scale2)).astype(np.float32)

        # 给目标帧更高权重，防止过度拖影
        weights[t_index] *= 2.0

        fused_mean = np.sum(weights * aligned_stack01, axis=0) / (np.sum(weights, axis=0) + 1e-8)
        fused_med = np.median(aligned_stack01, axis=0).astype(np.float32)

        fused = mean_weight * fused_mean + (1.0 - mean_weight) * fused_med
        return np.clip(fused, 0.0, 1.0)

    elif aligned_stack01.ndim == 4:
        # 彩色 [T,H,W,C]
        target_gray = get_gray(target)
        diff_gray = np.stack([get_gray(aligned_stack01[t]) for t in range(aligned_stack01.shape[0])], axis=0) - target_gray[None, ...]
        diff2 = diff_gray * diff_gray

        scale2 = (temporal_sigma_scale * sigma01) ** 2 + 1e-12
        weights = np.exp(-diff2 / (2.0 * scale2)).astype(np.float32)
        weights[t_index] *= 2.0

        weights_exp = weights[..., None]
        fused_mean = np.sum(weights_exp * aligned_stack01, axis=0) / (np.sum(weights_exp, axis=0) + 1e-8)
        fused_med = np.median(aligned_stack01, axis=0).astype(np.float32)

        fused = mean_weight * fused_mean + (1.0 - mean_weight) * fused_med
        return np.clip(fused, 0.0, 1.0)

    else:
        raise ValueError(f"不支持的 aligned_stack01 形状: {aligned_stack01.shape}")


# =========================================================
# 6) BM3D 去噪
# =========================================================
def bm3d_denoise_image(img01: np.ndarray, sigma01: float) -> np.ndarray:
    """
    img01 应该是 [0,1] float
    """
    den = bm3d(img01, sigma_psd=sigma01)
    den = np.asarray(den).astype(np.float32)
    return np.clip(den, 0.0, 1.0)


# =========================================================
# 7) 外圈评价区域
# =========================================================
def build_metric_outer_mask(shape, hole_area_ratio: float = 0.5) -> np.ndarray:
    if not (0.0 < hole_area_ratio < 1.0):
        raise ValueError("hole_area_ratio 必须在 (0, 1) 之间")

    h, w = shape[:2]

    side_ratio = np.sqrt(hole_area_ratio)
    inner_h = max(1, int(round(h * side_ratio)))
    inner_w = max(1, int(round(w * side_ratio)))

    y0 = (h - inner_h) // 2
    y1 = y0 + inner_h
    x0 = (w - inner_w) // 2
    x1 = x0 + inner_w

    mask = np.ones((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = False
    return mask


# =========================================================
# 8) 指标
# =========================================================
def normalize_raw_for_metric(img: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    img = img.astype(np.float32)
    denom = float(white_level - black_level)
    if denom <= 0:
        raise ValueError("white_level 必须大于 black_level")
    img = (img - black_level) / denom
    return np.clip(img, 0.0, 1.0)


def compute_mse_on_mask(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not np.any(mask):
        return float("nan")

    if img1.ndim == 2:
        a = img1[mask]
        b = img2[mask]
    else:
        a = img1[mask, :]
        b = img2[mask, :]

    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def compute_psnr_on_mask(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    mse = compute_mse_on_mask(img1, img2, mask)
    if not np.isfinite(mse):
        return float("nan")
    if mse <= 1e-15:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def _ssim_map_single_channel(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

    sigma_x2 = np.maximum(sigma_x2, 0.0)
    sigma_y2 = np.maximum(sigma_y2, 0.0)

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12
    )
    return ssim_map


def compute_ssim_on_mask(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    mask = mask.astype(bool)
    if not np.any(mask):
        return float("nan")

    if img1.ndim == 2:
        smap = _ssim_map_single_channel(img1, img2, data_range)
        return float(np.mean(smap[mask]))

    if img1.ndim == 3 and img1.shape[2] == 1:
        smap = _ssim_map_single_channel(img1[..., 0], img2[..., 0], data_range)
        return float(np.mean(smap[mask]))

    if img1.ndim == 3 and img1.shape[2] == 3:
        vals = []
        for c in range(3):
            smap = _ssim_map_single_channel(img1[..., c], img2[..., c], data_range)
            vals.append(float(np.mean(smap[mask])))
        return float(np.mean(vals))

    raise ValueError("不支持的图像维度")


# =========================================================
# 9) 显示
# =========================================================
def normalize_for_display(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)

    low = np.percentile(arr, 1)
    high = np.percentile(arr, 99)
    if high <= low:
        low = float(arr.min())
        high = float(arr.max())
    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)

    out = (arr - low) / (high - low)
    out = np.clip(out, 0.0, 1.0)

    if out.ndim == 3 and out.shape[2] == 3:
        out = out[..., ::-1]
    return out


def make_error_map(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    err = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    return normalize_for_display(err)


def show_comparison(noisy, denoised, clean=None, noisy_psnr=None, noisy_ssim=None, den_psnr=None, den_ssim=None):
    if clean is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(normalize_for_display(noisy), cmap="gray" if noisy.ndim == 2 else None)
        t0 = "Noisy"
        if noisy_psnr is not None and noisy_ssim is not None:
            t0 += f"\nOuter-PSNR={noisy_psnr:.4f}  Outer-SSIM={noisy_ssim:.4f}"
        axes[0].set_title(t0)
        axes[0].axis("off")

        axes[1].imshow(normalize_for_display(denoised), cmap="gray" if denoised.ndim == 2 else None)
        t1 = "Denoised"
        if den_psnr is not None and den_ssim is not None:
            t1 += f"\nOuter-PSNR={den_psnr:.4f}  Outer-SSIM={den_ssim:.4f}"
        axes[1].set_title(t1)
        axes[1].axis("off")

        axes[2].imshow(normalize_for_display(clean), cmap="gray" if clean.ndim == 2 else None)
        axes[2].set_title("Clean (GT)")
        axes[2].axis("off")

        axes[3].imshow(make_error_map(denoised, clean), cmap="gray")
        axes[3].set_title("|Denoised - Clean|")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(normalize_for_display(noisy), cmap="gray" if noisy.ndim == 2 else None)
        axes[0].set_title("Noisy")
        axes[0].axis("off")

        axes[1].imshow(normalize_for_display(denoised), cmap="gray" if denoised.ndim == 2 else None)
        axes[1].set_title("Denoised")
        axes[1].axis("off")

        axes[2].imshow(make_error_map(denoised, noisy), cmap="gray")
        axes[2].set_title("|Denoised - Noisy|")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


# =========================================================
# 10) 主流程
# 真数据：不使用任何 GT 保底 / 回拉
# =========================================================
def run_bm3d_temporal_3dnr(
    target_tiff: str,
    radius: int = 3,
    black_level: float = 0,
    white_level: float = 4095,
    metric_hole_area_ratio: float = 0.5,
    sigma01: float = None,
    temporal_sigma_scale: float = 2.5,
    mean_weight: float = 0.7,
    show_fig: bool = True
):
    target_path = Path(target_tiff)
    if not target_path.exists():
        raise FileNotFoundError(f"文件不存在: {target_path}")

    paths, t_index = collect_temporal_stack_symmetric(target_path, radius=radius)
    raw_frames = [read_tiff(p) for p in paths]
    ref_dtype = raw_frames[t_index].dtype
    noisy_raw = raw_frames[t_index].astype(np.float32)

    stack01 = np.stack([to_float01(f, black_level, white_level) for f in raw_frames], axis=0)

    if sigma01 is None:
        sigma01 = estimate_sigma01_from_target(stack01[t_index])

    aligned_stack01 = build_aligned_stack_to_target(stack01, t_index=t_index)

    fused01 = temporal_fusion(
        aligned_stack01=aligned_stack01,
        t_index=t_index,
        sigma01=sigma01,
        temporal_sigma_scale=temporal_sigma_scale,
        mean_weight=mean_weight
    )

    den01 = bm3d_denoise_image(fused01, sigma01=sigma01)
    den_raw = from_float01(den01, black_level, white_level).astype(np.float32)

    metric_mask = build_metric_outer_mask(noisy_raw.shape, hole_area_ratio=metric_hole_area_ratio)

    clean_path = find_clean_file_for_noisy(target_path)
    clean_raw = None
    noisy_psnr = noisy_ssim = den_psnr = den_ssim = None
    noisy_mse = den_mse = None

    if clean_path is not None and clean_path.exists():
        clean_raw = read_tiff(clean_path).astype(np.float32)
        if clean_raw.shape != noisy_raw.shape:
            raise ValueError(f"clean 与 noisy 尺寸不一致: {clean_raw.shape} vs {noisy_raw.shape}")

        noisy_m = normalize_raw_for_metric(noisy_raw, black_level, white_level)
        den_m = normalize_raw_for_metric(den_raw, black_level, white_level)
        clean_m = normalize_raw_for_metric(clean_raw, black_level, white_level)

        noisy_mse = compute_mse_on_mask(noisy_m, clean_m, metric_mask)
        den_mse = compute_mse_on_mask(den_m, clean_m, metric_mask)

        noisy_psnr = compute_psnr_on_mask(noisy_m, clean_m, metric_mask)
        noisy_ssim = compute_ssim_on_mask(noisy_m, clean_m, metric_mask)

        den_psnr = compute_psnr_on_mask(den_m, clean_m, metric_mask)
        den_ssim = compute_ssim_on_mask(den_m, clean_m, metric_mask)

    out_path = target_path.with_name(target_path.stem + "_bm3d_temporal_3dnr_out" + target_path.suffix)
    save_tiff(out_path, den_raw, ref_dtype)

    print("BM3D-temporal-3DNR 用到的时域帧：")
    for p in paths:
        print("  ", p.name)

    print(f"\nsigma01 = {sigma01:.10f}")
    print(f"temporal_sigma_scale = {temporal_sigma_scale}")
    print(f"mean_weight = {mean_weight}")
    print("输出文件:", out_path)

    if clean_raw is not None:
        print("\n========== 外圈指标（真实结果，不做GT保底） ==========")
        print(f"black_level = {black_level}")
        print(f"white_level = {white_level}")
        print(f"metric_hole_area_ratio = {metric_hole_area_ratio}")
        print(f"Noisy    -> MSE: {noisy_mse:.12e}, PSNR: {noisy_psnr:.10f}, SSIM: {noisy_ssim:.10f}")
        print(f"Denoised -> MSE: {den_mse:.12e}, PSNR: {den_psnr:.10f}, SSIM: {den_ssim:.10f}")
        print(f"PSNR 变化: {(den_psnr - noisy_psnr):.10f}")
        print(f"SSIM 变化: {(den_ssim - noisy_ssim):.10f}")
    else:
        print("\n未找到 clean 图，跳过 PSNR/SSIM。")

    if show_fig:
        show_comparison(
            noisy_raw,
            den_raw,
            clean=clean_raw,
            noisy_psnr=noisy_psnr,
            noisy_ssim=noisy_ssim,
            den_psnr=den_psnr,
            den_ssim=den_ssim
        )

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM3D-temporal 3DNR on one CRVD-style RAW TIFF frame.")
    parser.add_argument("target_tiff", help="当前帧路径，例如 scene1/ISO25600/frame5_noisy3.tiff")
    parser.add_argument("--radius", type=int, default=3, help="时域窗口半径，3 表示最多使用 7 帧")
    parser.add_argument("--black-level", type=float, default=0.0)
    parser.add_argument("--white-level", type=float, default=4095.0)
    parser.add_argument("--metric-hole-area-ratio", type=float, default=0.5, help="指标只在外圈区域计算")
    parser.add_argument("--sigma01", type=float, default=None, help="归一化噪声标准差；不填则自动估计")
    parser.add_argument("--temporal-sigma-scale", type=float, default=2.5)
    parser.add_argument("--mean-weight", type=float, default=0.7, help="加权均值与中值融合中的均值权重")
    parser.add_argument("--show-fig", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_bm3d_temporal_3dnr(
        target_tiff=args.target_tiff,
        radius=args.radius,
        black_level=args.black_level,
        white_level=args.white_level,
        metric_hole_area_ratio=args.metric_hole_area_ratio,
        sigma01=args.sigma01,
        temporal_sigma_scale=args.temporal_sigma_scale,
        mean_weight=args.mean_weight,
        show_fig=args.show_fig,
    )
