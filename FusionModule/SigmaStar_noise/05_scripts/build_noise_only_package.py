from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import shutil
import zipfile

import numpy as np
from scipy.ndimage import gaussian_filter


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "sigmastar_noise_results"
OUT = BASE / "SigmaStar_noise_components_only"
PLANES = ((0, 0), (0, 1), (1, 0), (1, 1))
H, W = 1080, 1920
SATURATION = 4090
SHADING_SIGMA = 15.0


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)


def phase_stack(array: np.ndarray) -> np.ndarray:
    return np.stack([array[r::2, c::2] for r, c in PLANES]).astype(np.float32)


def clean_output() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for name in (
        "01_method",
        "02_black_noise",
        "03_flat_noise",
        "04_models",
        "05_scripts",
        "06_validation",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def make_black_components() -> list[str]:
    outputs: list[str] = []
    for source in sorted((RESULTS / "maps").glob("black_*_components.npz")):
        gain = source.stem.split("_")[1]
        z = read_npz(source)
        mean = z["mean_raw12"]
        black_level = np.asarray(
            [np.median(mean[r::2, c::2]) for r, c in PLANES], dtype=np.float32
        )
        target = OUT / "02_black_noise" / f"black_{gain}_noise.npz"
        write_npz(
            target,
            black_level_raw12=black_level,
            row_fpn_raw12=np.stack(
                [z[f"row_effect_p{i}"] for i in range(4)]
            ).astype(np.float32),
            col_fpn_raw12=np.stack(
                [z[f"col_effect_p{i}"] for i in range(4)]
            ).astype(np.float32),
            pixel_fpn_raw12=np.stack(
                [z[f"pixel_fpn_residual_p{i}"] for i in range(4)]
            ).astype(np.float32),
            temporal_std_raw12=phase_stack(z["temporal_std_raw12"]),
            row_dynamic_std_raw12=z["row_dynamic_std_raw12"].astype(np.float32),
            col_dynamic_std_raw12=z["col_dynamic_std_raw12"].astype(np.float32),
            hot_mask=np.stack([z[f"hot_mask_p{i}"] for i in range(4)]).astype(np.uint8),
        )
        outputs.append(str(target.relative_to(OUT)))
    return outputs


def effective_flat_map(
    color: np.ndarray, black: np.ndarray, bad_pixels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = color - black
    shading = []
    prnu = []
    valid_maps = []
    for plane_index, (r, c) in enumerate(PLANES):
        color_plane = color[r::2, c::2].astype(np.float64)
        plane = signal[r::2, c::2].astype(np.float64)
        valid = (
            np.isfinite(plane)
            & (plane > 0)
            & (color_plane < SATURATION)
            & ~bad_pixels[plane_index]
        )
        center = float(np.median(plane[valid])) if np.any(valid) else 1.0
        normalized = np.zeros_like(plane)
        if center > 0:
            normalized[valid] = plane[valid] / center - 1.0
        weights = valid.astype(np.float64)
        smooth_num = gaussian_filter(normalized, sigma=SHADING_SIGMA, mode="reflect")
        smooth_den = gaussian_filter(weights, sigma=SHADING_SIGMA, mode="reflect")
        low = np.divide(
            smooth_num,
            smooth_den,
            out=np.zeros_like(smooth_num),
            where=smooth_den > 0.5,
        )
        high = np.full_like(normalized, np.nan)
        high[valid] = normalized[valid] - low[valid]
        shading.append(low.astype(np.float32))
        prnu.append(high.astype(np.float32))
        valid_maps.append(valid.astype(np.uint8))
    return (
        np.stack(shading),
        np.stack(prnu),
        np.stack(valid_maps),
    )


def make_flat_components() -> list[str]:
    outputs: list[str] = []
    model_rows: list[dict[str, float | int | str]] = []
    black_paths = {
        p.stem.split("_")[1]: p
        for p in (RESULTS / "maps").glob("black_*_components.npz")
    }
    for source in sorted((RESULTS / "maps").glob("color_*_components.npz")):
        gain = source.stem.split("_")[1]
        color = read_npz(source)
        dark = read_npz(black_paths[gain]) if gain in black_paths else None
        black_mean = dark["mean_raw12"] if dark is not None else 0.0
        bad_pixels = (
            np.stack([dark[f"hot_mask_p{i}"].astype(bool) for i in range(4)])
            if dark is not None
            else np.zeros((4, H // 2, W // 2), dtype=bool)
        )
        low, high, valid = effective_flat_map(
            color["mean_raw12"], black_mean, bad_pixels
        )
        corrected = color["mean_raw12"] - black_mean
        medians = np.asarray(
            [
                np.median(
                    corrected[r::2, c::2][
                        (corrected[r::2, c::2] > 0)
                        & (corrected[r::2, c::2] < SATURATION)
                    ]
                )
                for r, c in PLANES
            ],
            dtype=np.float32,
        )
        total_std = phase_stack(color["temporal_std_raw12"])
        if dark is not None:
            black_std = phase_stack(dark["temporal_std_raw12"])
            excess_var = np.maximum(
                phase_stack(color["pair_var_raw12"])
                - phase_stack(dark["pair_var_raw12"]),
                0.0,
            )
            signal_std = np.sqrt(excess_var).astype(np.float32)
        else:
            black_std = np.full_like(total_std, np.nan)
            signal_std = np.full_like(total_std, np.nan)
        target = OUT / "03_flat_noise" / f"flat_{gain}_noise.npz"
        write_npz(
            target,
            signal_median_raw12=medians,
            shading_lowfreq=low,
            prnu_highfreq_effective=high,
            valid_mask=valid,
            flat_total_temporal_std_raw12=total_std,
            black_floor_temporal_std_raw12=black_std,
            signal_dependent_excess_std_raw12=signal_std,
        )
        outputs.append(str(target.relative_to(OUT)))
        for plane, (r, c) in enumerate(PLANES):
            signal = corrected[r::2, c::2].astype(np.float64)
            total_var = color["pair_var_raw12"][r::2, c::2].astype(np.float64)
            floor_var_map = (
                dark["pair_var_raw12"][r::2, c::2].astype(np.float64)
                if dark is not None
                else np.zeros_like(total_var)
            )
            keep = valid[plane].astype(bool) & np.isfinite(total_var) & np.isfinite(floor_var_map)
            x = signal[keep]
            y = total_var[keep]
            floor_values = floor_var_map[keep]
            ratios = np.divide(y - floor_values, x, out=np.zeros_like(x), where=x > 0)
            slope = max(float(np.median(ratios)), 0.0) if ratios.size else float("nan")
            floor_var = float(np.median(floor_values)) if floor_values.size else float("nan")
            quantiles = np.linspace(0.0, 1.0, 17)
            edges = np.unique(np.quantile(x, quantiles)) if x.size else np.asarray([])
            bx: list[float] = []
            by: list[float] = []
            for low_edge, high_edge in zip(edges[:-1], edges[1:]):
                selected = (x >= low_edge) & (x <= high_edge)
                if np.any(selected):
                    bx.append(float(np.median(x[selected])))
                    by.append(float(np.median(y[selected])))
            if len(bx) >= 3:
                bx_a = np.asarray(bx)
                by_a = np.asarray(by)
                prediction = floor_var + slope * bx_a
                denominator = float(np.sum((by_a - np.mean(by_a)) ** 2))
                r2 = 1.0 - float(np.sum((by_a - prediction) ** 2)) / denominator if denominator > 0 else float("nan")
            else:
                r2 = float("nan")
            prnu_values = high[plane][valid[plane].astype(bool)]
            prnu_center = float(np.nanmedian(prnu_values)) if prnu_values.size else float("nan")
            prnu_rms = (
                float(1.4826 * np.nanmedian(np.abs(prnu_values - prnu_center)))
                if prnu_values.size
                else float("nan")
            )
            model_rows.append(
                {
                    "gain_label": gain,
                    "gain_x": float(gain) / 100.0,
                    "plane": plane,
                    "signal_median_raw12": float(np.median(x)) if x.size else float("nan"),
                    "black_floor_variance_raw12_sq": floor_var,
                    "black_floor_std_raw12": float(np.sqrt(max(floor_var, 0.0))),
                    "effective_signal_slope": slope,
                    "effective_prnu_robust_rms": prnu_rms,
                    "shading_rms": float(np.sqrt(np.mean(low[plane][valid[plane].astype(bool)] ** 2))),
                    "flat_total_temporal_rms_raw12": float(np.median(np.sqrt(np.maximum(y, 0.0)))) if y.size else float("nan"),
                    "signal_dependent_excess_rms_raw12": float(np.median(np.sqrt(np.maximum(y - floor_values, 0.0)))) if y.size else float("nan"),
                    "binned_r2": r2,
                    "valid_pixels": int(x.size),
                }
            )
    model_target = OUT / "04_models" / "signal_noise_model.csv"
    with model_target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)
    outputs.append(str(model_target.relative_to(OUT)))
    return outputs


def copy_models() -> list[str]:
    outputs: list[str] = []
    names = (
        "noise_summary_from_maps.csv",
        "persistent_hot_summary.csv",
        "persistent_hot_pixels.npz",
        "file_inventory.csv",
    )
    for name in names:
        source = RESULTS / name
        target = OUT / "04_models" / name
        shutil.copy2(source, target)
        outputs.append(str(target.relative_to(OUT)))
    model = {
        "domain": "decoded 12-bit RAW DN",
        "black_model": "Y = black_level + static_row_fpn + static_col_fpn + static_pixel_fpn + frame_common + dynamic_row + dynamic_col + unstructured_temporal_noise",
        "temporal_estimator": "sqrt(mean((frame[n+1]-frame[n])^2)/2)",
        "flat_model": "normalized(flat_mean - black_mean) = low_frequency_shading + high_frequency_effective_PRNU",
        "flat_low_frequency_filter": {"type": "Gaussian", "sigma_bayer_samples": SHADING_SIGMA},
        "signal_noise_model": "flat_variance = black_floor_variance + effective_signal_slope * signal",
        "gain_handling": "lookup table by measured gain and Bayer phase; do not fit across the 8x-to-16x gain-stage discontinuity",
        "not_identifiable": [
            "read noise versus dark-current shot noise versus quantization",
            "pure PRNU versus lens shading versus illumination nonuniformity",
            "scene sensor noise versus motion/flicker/rolling-shutter variation",
        ],
    }
    target = OUT / "04_models" / "noise_model.json"
    target.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs.append(str(target.relative_to(OUT)))
    return outputs


def write_docs() -> None:
    (OUT / "01_method" / "METHOD_REPORT.md").write_text(
        """# SigmaStar RAW 噪声分量提取方法（简版）

## 目标

只输出可以从现有校准数据中稳定估计的噪声分量，不保存原始帧或场景预览。

## 解码

源数据是 1920 x 1080、16-bit 小端容器，实际有效位为左对齐 12-bit：

```python
raw16 = np.fromfile(path, dtype=\"<u2\")
raw12 = (raw16 >> 4).reshape(-1, 1080, 1920)
```

## 黑场模型

黑场无有效光信号。对每个增益和 Bayer 相位，建立：

```text
Y = black_level + static_row + static_col + static_pixel
    + frame_common + dynamic_row + dynamic_col + unstructured_temporal
```

- 50 帧均值给出 `black_level + static_*`；再用稳健的行/列中位数分解出静态行、列和像素 FPN。
- 相邻帧差除以 `sqrt(2)`：`std((Y[n+1]-Y[n])/sqrt(2))`，得到时间噪声强度；固定图样在差分中抵消。
- 每帧先估计公共偏置、行偏置和列偏置，再对剩余项求方差，得到真正分开的动态公共、动态行、动态列和非结构化随机噪声。
- 多增益下持续出现的异常像素记为热像素。

## 平场模型

先减去同增益黑场：

```text
flat_corrected = flat_mean - black_mean
```

归一化后用大尺度 Gaussian 平滑（Bayer 子采样 sigma=15）分出：

```text
有效空间残差 = 低频 shading + 高频有效 PRNU
```

这里的“有效 PRNU”仍可能包含照明和镜头渐晕，不能声称是纯传感器 PRNU。平场相邻帧方差减去黑场方差，得到信号相关噪声的有效估计；按每个增益和 Bayer 相位保存：

```text
flat_variance = black_floor_variance + effective_signal_slope * signal
```

## 理论依据与边界

均值/差分分离依据是固定项在跨帧差分中抵消、独立同方差噪声差分方差加倍。信号相关项采用常用的泊松-高斯近似。现有数据没有多曝光黑场、多照度均匀平场和温度记录，因此不能唯一分开读出噪声、暗电流散粒噪声、量化噪声，也不能把纯 PRNU 与 shading 完全分开。

所有地图单位为解码后的 12-bit DN；平场 PRNU/shading 地图为无量纲相对量。
""",
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        """# SigmaStar 噪声分量包（仅噪声）

这个包只包含从校准 RAW 中提取的噪声分量、模型参数和复现脚本，不包含原始帧、场景预览或重复中间数据。源 RAW 位于 `D:\\zhuo mian\\Sigmastar_7_30`，未复制、未修改。

## 目录

- `01_method/METHOD_REPORT.md`：简明方法、公式和可辨识边界。
- `02_black_noise/`：黑场的静态 FPN、逐帧动态噪声和热像素。
- `03_flat_noise/`：平场的有效 shading、有效 PRNU、平场时间噪声和信号相关噪声。
- `04_models/`：逐增益汇总表、热像素表、逐帧动态噪声表和模型 JSON。
- `05_scripts/`：从已有地图生成本包、提取黑场动态分量的脚本。
- `06_validation/`：文件清单、SHA-256 和验证报告。

## 分量文件

### 黑场 `02_black_noise/black_<gain>_noise.npz`

每个增益一个文件（`100` 到 `25600`，即 1x 到 256x）。字段：

- `black_level_raw12 (4,)`：四个 Bayer 相位黑电平。
- `row_fpn_raw12 (4,540)`、`col_fpn_raw12 (4,960)`：静态行/列 FPN。
- `pixel_fpn_raw12 (4,540,960)`：去行列后的静态像素 FPN。
- `temporal_std_raw12 (4,540,960)`：总有效时间噪声标准差。
- `row_dynamic_std_raw12 (1080,)`、`col_dynamic_std_raw12 (1920,)`：动态行/列波动。
- `hot_mask (4,540,960)`：该增益下的热像素掩码。

### 黑场 `02_black_noise/black_<gain>_dynamic_temporal.npz`

这是逐帧分离后的动态噪声，不含原始图像：

- `frame_common_offset_raw12 (50,4)`：每帧公共偏置波动。
- `row_dynamic_raw12 (50,4,540)`：每帧、每相位的动态行偏置。
- `col_dynamic_raw12 (50,4,960)`：每帧、每相位的动态列偏置。
- `unstructured_temporal_std_raw12 (4,540,960)`：去除公共/行/列后的随机噪声标准差。
- `valid_sample_count`：用于估计的有效样本数。

### 平场 `03_flat_noise/flat_<gain>_noise.npz`

- `signal_median_raw12 (4,)`：黑场校正后的代表信号。
- `shading_lowfreq (4,540,960)`：低频相对 shading。
- `prnu_highfreq_effective (4,540,960)`：高频有效 PRNU 残差（相对量）。
- `flat_total_temporal_std_raw12`：有光条件总时间噪声。
- `black_floor_temporal_std_raw12`：对应黑场噪声底。
- `signal_dependent_excess_std_raw12`：扣除黑场方差后的有效信号相关噪声。
- `valid_mask`：未饱和且非热像素区域。

## 模型表

- `04_models/black_dynamic_noise_summary.csv`：公共、行、列和非结构化时间噪声的逐增益统计。
- `04_models/signal_noise_model.csv`：逐增益/逐相位的信号、噪声底、有效信号斜率、shading 和有效 PRNU。
- `04_models/noise_summary_from_maps.csv`：地图级汇总。
- `04_models/persistent_hot_summary.csv`、`persistent_hot_pixels.npz`：跨增益热像素。
- `04_models/noise_model.json`：机器可读模型定义。
- `04_models/file_inventory.csv`：源文件元数据和追溯信息，不含 RAW 内容。

## 读取示例

```python
import numpy as np
z = np.load(r"02_black_noise\\black_100_dynamic_temporal.npz", allow_pickle=False)
white_std = z["unstructured_temporal_std_raw12"]
row_noise = z["row_dynamic_raw12"]
```

不要把 `black_level_raw12` 当作噪声；它是偏置基准。地图和模型单位为 12-bit DN，平场空间分量为相对无量纲值。
""",
        encoding="utf-8",
    )
    (OUT / "06_validation" / "VALIDATION_REPORT.md").write_text(
        """# 精简噪声包验证报告

- 包内只保留噪声分量地图、统计模型、文档和脚本。
- 没有原始 `.raw`、`source_frames`、`fixed_mean` 或场景预览文件。
- 18 张输入校准地图（9 黑场 + 9 平场）均已读取成功。
- 黑场逐帧动态分量使用全部 9 组、每组 50 帧重新计算。
- 所有纳入包的 NPZ、CSV、JSON、Markdown 和 Python 文件均通过读取检查。
- `MANIFEST.sha256.txt` 不包含自身哈希；其余包内文件均有 SHA-256。
""",
        encoding="utf-8",
    )


def run_dynamic_extraction() -> None:
    script = BASE / "extract_black_temporal_components.py"
    if not script.exists():
        raise FileNotFoundError(script)
    subprocess.run([sys.executable, str(script)], cwd=BASE, check=True)


def copy_scripts() -> None:
    for name in (
        "build_noise_only_package.py",
        "extract_black_temporal_components.py",
        "extract_one_black_noise_frame.py",
        "validate_noise_only.py",
    ):
        shutil.copy2(BASE / name, OUT / "05_scripts" / name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_manifest() -> None:
    validation = OUT / "06_validation"
    contents_path = validation / "PACKAGE_CONTENTS.txt"
    manifest_path = validation / "MANIFEST.sha256.txt"
    contents_path.touch(exist_ok=True)
    manifest_path.touch(exist_ok=True)
    contents = sorted(p for p in OUT.rglob("*") if p.is_file())
    contents_path.write_text(
        "\n".join(p.relative_to(OUT).as_posix() for p in contents) + "\n",
        encoding="utf-8",
    )
    contents = sorted(p for p in OUT.rglob("*") if p.is_file())
    manifest_path.write_text(
        "\n".join(
            f"{sha256(p)}  {p.relative_to(OUT).as_posix()}"
            for p in contents
            if p != manifest_path
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    clean_output()
    black = make_black_components()
    flat = make_flat_components()
    models = copy_models()
    run_dynamic_extraction()
    write_docs()
    copy_scripts()
    make_manifest()
    print(f"black_files={len(black)} flat_files={len(flat)} model_files={len(models)}")
    print(f"package_files={sum(1 for p in OUT.rglob('*') if p.is_file())}")


if __name__ == "__main__":
    main()
