# SigmaStar 噪声分量包（仅噪声）

这个包只包含从校准 RAW 中提取的噪声分量、模型参数和复现脚本，不包含原始帧、场景预览或重复中间数据。源 RAW 位于 `D:\zhuo mian\Sigmastar_7_30`，未复制、未修改。

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
z = np.load(r"02_black_noise\black_100_dynamic_temporal.npz", allow_pickle=False)
white_std = z["unstructured_temporal_std_raw12"]
row_noise = z["row_dynamic_raw12"]
```

不要把 `black_level_raw12` 当作噪声；它是偏置基准。地图和模型单位为 12-bit DN，平场空间分量为相对无量纲值。
