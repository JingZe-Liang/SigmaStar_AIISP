# 算子名称：RAWProcessing

负责人：邸锦承
最后修改日期：2026-05-27

## 1. 功能定位

该模块负责把外部 RAW 数据统一成团队内部可直接复用的标准格式，并提供两类辅助能力：

1. 数据标准化：把 `.npy` / `.dng` 对齐成统一的 `12-bit GBRG / fp32 / safetensors`。
2. 数据检查与可视化：快速判断输入是否像真实 Bayer RAW，并通过简化 ISP 把 RAW 渲染成肉眼可检查的视频。

## 2. 输入输出

### `npy_to_safetensors.py`

#### 输入

- 输入文件格式：`.npy` 或 `.dng`
- 输入语义：单帧 Bayer RAW
- 输入 shape：`H x W`
- 输入 dtype：
  - 常见为 `uint16`
  - 也支持 `float32`
- 输入数值范围：
  - 若最大值大于 `4095`，脚本会默认认为当前还是高于 12-bit 的原始范围，并执行 `data / 16.0` 压缩到 12-bit 语义
- 归一化要求：不要求预先归一化
- metadata 需求：不需要

#### 输出

- 输出文件格式：`.safetensors`
- Tensor key：`raw`
- 输出 shape：`1080 x 1920`，默认由 `TARGET_HEIGHT / TARGET_WIDTH` 控制
- 输出 dtype：`float32`
- 输出数值范围：`[0, 4095]`
- 输出归一化状态：不归一化，保留 12-bit RAW 数值语义
- 输出 CFA 约定：GBRG

### `OpenCV_ISP.py`

#### 输入

- 输入文件格式：`.safetensors`
- 输入目录：`INPUT_DIR`
- Tensor key：`raw`
- 输入 shape：`H x W`
- 输入 dtype：`float32`
- 输入数值范围：默认按 `12-bit RAW` 处理，即近似 `[0, 4095]`
- 归一化要求：不要求预先归一化
- metadata 需求：读取 `JSON_PATH` 指向的 `metadata.json`，当前脚本主要保留接口，核心 ISP 参数仍以内置常量为主

#### 输出

- 输出文件格式：`.avi`
- 输出路径：`OUTPUT_VIDEO`
- 输出帧格式：`8-bit BGR`
- 输出分辨率：与输入 RAW 一致
- 输出用途：人工目检 RAW 时序和亮度、颜色、噪声状态

### `testRAW.py`

#### 输入

- 输入文件格式：`.npy`
- 输入语义：单帧待检测 RAW
- 输入 shape：
  - 主要支持 `H x W`
  - 若为 `4 x H/2 x W/2` 或 `H/2 x W/2 x 4`，会被视为 packed RAW
- 输入 dtype：`uint16` 或 `float32`

#### 输出

- 输出形式：终端文字结论 + 局部放大可视化窗口
- 输出指标：
  - 数据维度
  - 数据类型
  - 相邻像素跳变比例 `mosaic_ratio`
- 输出用途：粗判输入是不是 Bayer RAW，而不是已经 demosaic 后的 RGB

## 3. 方法简述

### `npy_to_safetensors.py`

核心逻辑分 4 步：

1. 读取原始 Bayer RAW。
2. 若输入仍保留高位深范围，则压缩到 12-bit：
   - 公式：`raw_12bit = raw_input / 16`
3. 执行中心裁剪，并通过控制裁剪起点奇偶性把 CFA 从原始排布对齐到团队统一的 `GBRG`。
4. 保存为 `fp32 safetensors`。

核心公式：

- 位深压缩：
  - `x_12bit = x_raw / 16`
- 合法裁剪：
  - `crop = raw[start_h:start_h+H_t, start_w:start_w+W_t]`
- 裁剪后数值裁切：
  - `x_out = clip(crop, 0, 4095)`

这一步的核心不是“去噪”，而是“把不同来源 RAW 变成统一可训练输入”。

### `OpenCV_ISP.py`

核心逻辑分 5 步：

1. 黑电平校正与白电平归一化。
2. 灰世界假设下的自动白平衡。
3. 曝光补偿。
4. Bayer demosaic。
5. Gamma 校正并写入视频。

核心公式：

- 线性归一化：
  - `x_norm = clip((x - black_level) / (white_level - black_level), 0, 1)`
- 灰世界白平衡：
  - `r_gain = mean(G) / (mean(R) + eps)`
  - `b_gain = mean(G) / (mean(B) + eps)`
- Gamma：
  - `x_srgb = x^(1/2.2)`

这里的 ISP 是“可视化 ISP”，目标是让人能快速看清数据，不是严格复现相机厂 ISP。

### `testRAW.py`

核心逻辑是利用 Bayer RAW 的像素交替采样特征，检查相邻像素是否存在明显跳变。

核心公式：

- 相邻像素差均值：
  - `diff_h = mean(|patch[:, 0::2] - patch[:, 1::2]|)`
- 马赛克跳变比例：
  - `mosaic_ratio = diff_h / (mean(patch) + eps)`

经验解释：

- 若 `mosaic_ratio` 明显偏大，说明图像里存在 Bayer 交替采样结构，更像真实 RAW。
- 若 `mosaic_ratio` 很小，说明数据更像灰度图、平滑图，或者已经过 ISP / 去马赛克处理。

## 4. 运行方式

```bash
python RAWProcessing\npy_to_safetensors.py
python RAWProcessing\testRAW.py
python RAWProcessing\OpenCV_ISP.py
```

运行前请先修改脚本顶部的输入目录、输出目录和 `metadata.json` 路径。

## 5. 参数解析（按脚本）

### `npy_to_safetensors.py`

- `INPUT_DIR`：原始 `.npy` / `.dng` 输入目录。
- `OUTPUT_DIR`：标准化后的 `.safetensors` 输出目录。
- `TARGET_HEIGHT / TARGET_WIDTH`：目标裁剪尺寸。
- `DEBUG_PRINT`：是否打印首帧 patch 对比信息。
- `PATCH_Y / PATCH_X / PATCH_SIZE`：调试 patch 的位置和大小。

### `OpenCV_ISP.py`

- `INPUT_DIR`：`.safetensors` RAW 帧目录。
- `JSON_PATH`：`metadata.json` 路径。
- `OUTPUT_VIDEO`：ISP 可视化视频输出路径。
- `FPS`：输出视频帧率。
- `EXPOSURE_COMPENSATION`：亮度补偿倍率。

### `testRAW.py`

- `FILE_PATH`：待检查的 `.npy` 文件路径。

## 6. 算子特点

- 能把不同来源的 RAW 数据统一成后续训练可直接使用的格式。
- 明确保留了位深、CFA 排布和数值语义，避免“看起来对了、实际喂错了”的情况。
- 适合做数据入库前的标准化和快速视觉检查。
