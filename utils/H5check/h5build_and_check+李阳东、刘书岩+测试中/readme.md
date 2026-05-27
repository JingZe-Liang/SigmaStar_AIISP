# RAW 视频数据处理流水线

本仓库包含两个脚本，用于将原始 Bayer RAW 视频流转换为 `.safetensors` 中间格式，再进一步转换为 `.tiff` 格式，以适配下游噪声合成与去噪训练流程。

---

## 1. 文件说明

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `raw_to_safetensors.py` | 将二进制 RAW 视频拆帧、中心裁剪并保存为 Safetensors | `.raw` 二进制视频流（16-bit LE，单通道 Bayer） | 每帧一个 `.safetensors` 文件（`uint16`，单通道） |
| `safetensors_to_tiff.py` | 将 Safetensors 转换为 TIFF，组织为 `ISOxxxx/` 目录结构 | `.safetensors` 文件目录 | `.tiff` 文件（`uint16`，单通道） |

---

## 2. 输入输出格式

### `raw_to_safetensors.py`

**输入**
- **格式**：二进制 RAW 视频流（`.raw`），无文件头，纯像素数据
- **像素深度**：16-bit，小端序（`&lt;u2`）
- **排列**：单通道 Bayer（GRBG CFA），逐行扫描
- **分辨率**：原始分辨率（如 `3200×1800`），由脚本参数指定
- **总帧数**：由脚本参数指定

**输出**
- **格式**：`.safetensors`（每帧独立文件）
- **Shape**：`(TAR_HEIGHT, TAR_WIDTH)`，单通道
- **Dtype**：`uint16`
- **命名**：`frame0001.safetensors` ~ `frame{TOTAL_FRAMES:04d}.safetensors`
- **内容**：中心裁剪后的 Bayer RAW 帧，**未做去马赛克、未归一化**

### `safetensors_to_tiff.py`

**输入**
- **格式**：`.safetensors` 文件目录（如 `scene1_gt/`）
- **内容**：单通道 `uint16` 或 `float32` Bayer RAW 帧
- **键名**：优先读取 `"raw"`，否则取第一个键

**输出**
- **格式**：`.tiff`（TIFF 单通道灰度）
- **Shape**：`(H, W)`
- **Dtype**：`uint16`
- **目录结构**：`{output_dir}/ISO{iso}/frame{idx:03d}_noisy.tiff`
- **内容**：保持原始数值范围（默认 12-bit 映射到 `uint16`，范围 `0~4095`）

---

## 3. 方法简述

### `raw_to_safetensors.py`

逐帧从 `.raw` 二进制流中定位偏移量，读取 `ORI_HEIGHT × ORI_WIDTH` 个 `uint16` 像素，重塑为 2D 数组后执行**中心裁剪**。裁剪偏移量和目标分辨率均强制为偶数，确保 GRBG Bayer 图案的 2×2 CFA 单元不被破坏。最终每帧保存为独立的 `.safetensors` 文件。

$$\text{cropped} = \text{frame}\left[\frac{H_{\text{ori}}-H_{\text{tar}}}{2} : \frac{H_{\text{ori}}+H_{\text{tar}}}{2},\ \frac{W_{\text{ori}}-W_{\text{tar}}}{2} : \frac{W_{\text{ori}}+W_{\text{tar}}}{2}\right]$$

### `safetensors_to_tiff.py`

遍历输入目录下的 `.safetensors` 文件，加载张量后转为 NumPy 数组。将数据裁剪到 `[0, 65535]` 并转为 `uint16`，按 `ISOxxxx/frame{idx:03d}_noisy.tiff` 的命名规则写入 TIFF，以兼容 `realistic_raw_noise_synthesis.py` 的目录结构要求。

---

## 4. 运行方式

### Step 1：RAW → Safetensors

编辑脚本头部的配置常量，然后直接运行：

```bash
python raw_to_safetensors.py
```
---
# H5数据集构建流水线
## 1.功能定位
构建H5数据集并校验
## 2. 输入输出

**输入**
- **格式**：TIFF 图片（单通道灰度图）
- **Shape**：`(H, W)`，由 `--height` 和 `--width` 指定，默认 `1080×1920`
- **Dtype**：`uint16`（代码强制检查 `arr.dtype != np.uint16`）
- **归一化**：**不需要**，代码直接保存原始 uint16 值
- **metadata.json**：**需要输出**，构建完成后自动生成，记录分片信息、数据集结构、压缩参数等
- **简短说明**：已配准的 2DNR/3DNR/Noisy 单通道 RAW 帧序列，用于训练不含 clean 目标的去噪网络

**输出**
- **格式**：分片 HDF5 文件（`shard_*.h5`）+ `metadata.json`
- **Shape**：
  - `2dnr` / `3dnr`：`(frames_per_shard, H, W)`
  - `noisy`：`(frames_per_shard, 2, H, W)`，其中 `2` 表示 `[prev_frame, curr_frame]`
- **Dtype**：`uint16`
- **归一化**：**未归一化**，保持原始 uint16 动态范围
- **指标**：**无指标产生**。本脚本为数据集构建工具，不涉及 PSNR/SSIM 等质量评估。

---

## 3. 方法简述

将三个目录（`2dnr`、`3dnr`、`noisy`）中按帧索引对齐的 TIFF 图像读取为 `uint16` 数组，按 `frames_per_shard` 分片写入 HDF5。每个样本包含：当前帧的 2DNR、3DNR 结果，以及 noisy 的 `[前一帧, 当前帧]` 堆叠。首帧（index=0）的前一帧采用 `duplicate` 策略复制当前帧，避免越界。

$$\text{noisy}[i] = \begin{cases} [\text{noisy}_0, \text{noisy}_0] & i=0 \\ [\text{noisy}_{i-1}, \text{noisy}_i] & i>0 \end{cases}$$

---

## 4. 运行方式

```bash
python build_h5_no_clean.py `
  --input-2dnr D:\project\data\scene1\2dnr `
  --input-3dnr D:\project\data\scene1\3dnr `
  --input-noisy D:\project\data\scene1\noisy `
  --output-dir D:\project\h5_output\scene1 `
  --num-frames 150 `
  --frames-per-shard 30 `
  --height 1080 `
  --width 1920 `
  --suffix-2dnr "_2dnr" `
  --suffix-3dnr "_3dnr" `
  --suffix-noisy "_noisy" `
  --compression gzip `
  --compression-opts 4 `
  --verify
```

---

## 5. 参数解析

| 参数 | 作用 |
|------|------|
| `--num-frames` | 要打包的总帧数，决定读取 `frame001.tiff` 到 `frame{num_frames}.tiff` |
| `--frames-per-shard` | 每个 HDF5 分片包含的帧数，影响单个 `.h5` 文件大小和后续读取并行度 |
| `--suffix-*` | 文件名后缀，用于匹配 TIFF 文件。例如 `frame001_2dnr.tiff` 对应 `--suffix-2dnr "_2dnr"` |
| `--compression` / `--compression-opts` | HDF5 压缩算法及等级（默认 `gzip` + `4`），平衡存储体积与解压速度 |
| `--verify` | 写入完成后执行完整校验（元数据、分片结构、像素级内容对齐） |
| `--verify-only` | **仅校验模式**，不重新写入，用于检查已存在的数据集完整性 |
| `--overwrite` | 覆盖输出目录中已存在的同名分片文件 |

**输入 TIFF 的具体来源**：TIFF为经过full_pipeline.py的数据集