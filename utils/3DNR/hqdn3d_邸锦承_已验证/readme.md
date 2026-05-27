# 算子名称：3DNR

负责人：邸锦承
最后修改日期：2026-05-27

## 1. 功能定位

该模块用于复现和对比 FFmpeg `hqdn3d` 3DNR 效果，并输出 noisy 与传统时域降噪结果的对比视频，方便观察拖影和时域平滑的影响。

它的核心不是训练模型，而是构造一个“可控的传统时域降噪基线”，用于：

1. 观察时域滤波对拖影的影响。
2. 为后续 `2DNR + 3DNR + AI` 融合提供传统算法参照。

## 2. 输入输出

### `hqdn3d.py`

#### 输入

- 输入目录：`INPUT_DIR`
- 输入文件格式：`.h5` / `.hdf5`
- 实际使用字段：
  - `2dnr`：作为 FFmpeg 3DNR 的原始输入
- 输入 shape：
  - 常见为 `T x H x W`
- 输入 dtype：
  - H5 中通常为整数 RAW
  - 脚本内部会转成 `float32`
- 数值范围：
  - 默认按 12-bit RAW 解释，即近似 `[0, 4095]`
- metadata：
  - `JSON_PATH` 当前仅作为接口保留，现有实现并未真正读取其中参数参与 ISP 计算

#### 输出

- 最终输出文件：`OUTPUT_VIDEO`
- 输出格式：`.avi`
- 中间产物：
  - `TEMP_IN_RAW`
  - `TEMP_OUT_RAW`
- 输出帧语义：经过 `hqdn3d` 时域滤波后，再经 ISP 渲染得到的 `8-bit BGR` 视频

### `compare_noisy_and_3dnr.py`

#### 输入

- 输入根目录：`COMPANY_DATA_ROOT`
- 输入文件格式：各个 `scene` 子目录中的 `.h5`
- 实际使用字段：
  - `noisy`
  - `2dnr`
- 不直接读取的字段：
  - 虽然有些数据集可能也包含 `3dnr`，但当前脚本不会读取已有 `3dnr` 字段，而是对 `2dnr` 重新运行 FFmpeg 生成传统 3DNR 基线
- 输入 shape：
  - `noisy` 常见为 `T x 2 x H x W`
  - `2dnr` 常见为 `T x H x W`

#### 输出

- 每个 scene 输出一个 `.avi`
- 输出内容：上半部分为 noisy 可视化，下半部分为脚本现场生成的传统 3DNR 可视化
- 输出用途：肉眼对比时域降噪前后差异，尤其是拖影、轮廓残留和过平滑

## 3. 方法简述

### `hqdn3d.py`

核心逻辑分 4 步：

1. 将 Bayer RAW 拆成四象限，避免不同颜色子像素在时域滤波时彼此污染。
2. 把拆分后的单通道序列写成 `gray16le rawvideo`。
3. 调用 FFmpeg `hqdn3d` 执行时域降噪。
4. 再把结果解包回 Bayer，并经过 ISP 渲染成可视化视频。

核心公式：

- Bayer 四象限打包：
  - `Q1 = raw[0::2, 0::2]`
  - `Q2 = raw[0::2, 1::2]`
  - `Q3 = raw[1::2, 0::2]`
  - `Q4 = raw[1::2, 1::2]`
- 12-bit 到 16-bit 容器映射：
  - `x_16 = clip((x_12 / 4095) * 65535, 0, 65535)`
- FFmpeg 滤波参数形式：
  - `hqdn3d = luma_spatial : chroma_spatial : luma_tmp : chroma_tmp`

当前脚本里：

- `HQDN3D_FILTER = "hqdn3d=0.001:0.001:8:3"`

其含义是：

- 几乎关闭空间模糊
- 主要放大时域平滑作用

这样更容易把拖影问题暴露出来。

### `compare_noisy_and_3dnr.py`

核心逻辑是把 noisy 与脚本现场生成的传统 3DNR 基线做上下拼接可视化：

1. 遍历每个 scene。
2. 用 `2dnr` 序列重新生成传统 3DNR 结果。
3. 对 noisy 当前帧和传统 3DNR 结果分别做 ISP 渲染。
4. 上下拼接后写成对比视频。

核心公式：

- 若 noisy 是双帧结构：
  - `noisy_t = noisy[:, 1, :, :]`
- 对比帧拼接：
  - `frame_compare = vstack(noisy_bgr, denoised_bgr)`

最终关注的是视觉现象，而不是单一数值指标，尤其看：

- 移动物体边缘是否留下残影
- 背景是否被过度抹平
- 高频纹理是否被时域滤波吞掉

## 4. 运行方式

```bash
python 3dnr\hqdn3d.py
python 3dnr\compare_noisy_and_3dnr.py
```

运行前请先修改脚本顶部的输入目录、输出目录和 `metadata.json` 路径。

## 5. 参数解析（按脚本）

### `hqdn3d.py`

- `INPUT_DIR`：输入 H5 文件所在目录。
- `JSON_PATH`：保留的 `metadata.json` 路径接口，当前实现未实际使用。
- `OUTPUT_VIDEO`：最终生成的 3DNR AVI 路径。
- `TEMP_IN_RAW / TEMP_OUT_RAW`：FFmpeg 使用的临时 RAW 文件。
- `FPS`：输出视频帧率。
- `EXPOSURE_COMPENSATION`：ISP 渲染时的亮度补偿。
- `HQDN3D_FILTER`：FFmpeg 3DNR 的核心参数字符串。

### `compare_noisy_and_3dnr.py`

- `COMPANY_DATA_ROOT`：包含多个 scene 的数据根目录。
- `OUTPUT_DIR`：对比视频输出目录。
- `FPS`：输出视频帧率。
- `EXPOSURE_COMPENSATION`：ISP 渲染时的亮度补偿。
- `HQDN3D_FILTER`：FFmpeg 3DNR 的核心参数字符串。

## 6. 算子特点

- 适合快速构造一个可解释的传统 3DNR 基线。
- 可以把拖影问题放大出来，便于后续 AI 融合时做针对性补偿。
- 输出对比视频比只看 PSNR 更适合分析时域视觉伪影。
