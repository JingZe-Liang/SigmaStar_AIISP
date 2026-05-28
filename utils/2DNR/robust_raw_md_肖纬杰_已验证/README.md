# 算子名称

Noise-Adaptive Flicker-Robust Motion Detection  
中文名：噪声自适应抗闪烁 RAW 运动检测

## 1. 功能定位

该算子用于对高 ISO、强噪声、亮度闪烁场景下的 RAW TIFF 图像序列做运动检测，输出二值 motion mask、bbox 可视化图和检测视频。

## 2. 输入输出格式

### 输入格式

元素类型：单通道 RAW TIFF 图像序列  
Shape：每帧为 `H * W`，脚本不限制固定分辨率；当前测试数据为 1080p RAW Bayer 图像  
Dtype：通常为 `uint16`  
是否需要归一化：不需要，脚本内部根据 `black_level` 和 `white_level` 归一化  
是否需要 metadata.json：不需要  
说明：输入应为按帧编号排序的 `.tif` / `.tiff` 文件夹，例如 `frame0000.tiff`、`frame0001.tiff`

### 输出格式

输出目录结构：

```text
output_dir/
  masks/              # 每帧二值运动 mask，uint8 PNG，0/255
  bboxes/             # 每帧 bbox 可视化图，uint8 PNG
  md_video.mp4        # mask 视频
  bbox_video.mp4      # bbox 可视化视频
  overlay_video.mp4   # 原图叠加 mask 和 bbox 的视频
  diagnostics.jsonl   # 每帧诊断信息
```

mask 输出：  
Shape：`H/2 * W/2`，因为算法在 Bayer G 通道下检测  
Dtype：`uint8`  
是否已归一化：否，二值图像，背景为 0，运动区域为 255  
Effective bits：8 bit  
Container bits：8 bit  

诊断信息包括每帧噪声估计、bbox 数量、全局位移估计和 bbox 坐标，用于分析误检、漏检和时间稳定性。

## 3. 方法简述

1. 将 RAW 图像按 `black_level / white_level` 归一化到 `[0, 1]`。
2. 从 Bayer RAW 中提取 G 通道作为检测图，降低 CFA 噪声影响。
3. 从采样帧中取中值建立初始背景，避免单帧噪声或偶发运动影响背景。
4. 对当前帧和背景做亮度尺度与偏置补偿，抑制整帧曝光波动和亮度闪烁。
5. 估计全局噪声和局部噪声，计算噪声归一化差分：

```text
score = abs(current - compensated_background) / local_sigma
```

6. 使用双阈值 hysteresis：高阈值生成可靠 seed，低阈值只在 seed 周围扩展。
7. 使用相邻帧短时运动支持过滤，静态亮斑或缓慢闪烁区域不会轻易保留。
8. 经过形态学闭运算、开运算、填洞、面积过滤和重复静态 bbox 抑制，得到最终运动 mask。

## 4. 运行示例

```powershell
python robust_raw_md_肖纬杰/noise_adaptive_flicker_robust_md.py `
  output_0/sc450ai2_noisy_current_tiff `
  output_0/MD_optimized_sc450ai2 `
  --mode robust `
  --cfa_pattern GBRG `
  --black_level 16 `
  --white_level 4095 `
  --min_area 250 `
  --low_threshold 2.0 `
  --high_threshold 3.5 `
  --fps 24
```

另一个测试序列：

```powershell
python robust_raw_md_肖纬杰/noise_adaptive_flicker_robust_md.py `
  output_0/sc450zou_noisy_current_tiff `
  output_0/MD_optimized_sc450zou `
  --mode robust `
  --cfa_pattern GBRG `
  --black_level 16 `
  --white_level 4095 `
  --min_area 250 `
  --low_threshold 2.0 `
  --high_threshold 3.5 `
  --fps 24
```

## 5. 参数解析

`input_dir`：输入 TIFF 图像序列文件夹。  
`output_dir`：输出结果文件夹。  
`--mode`：检测模式，`robust` 为噪声自适应抗闪烁模式，`default` 为原始 MOG2 模式。  
`--cfa_pattern`：RAW Bayer 排列方式，支持 `RGGB / BGGR / GBRG / GRBG`。  
`--black_level`：RAW 黑电平，默认 16。  
`--white_level`：RAW 白电平，默认 4095。  
`--detection_white_level`：检测归一化使用的白电平；不设置时使用 `white_level`。  
`--fps`：输出视频帧率。  
`--min_area`：最小连通域面积，小于该面积的运动区域会被过滤。  
`--low_threshold`：双阈值中的低阈值，用于扩展运动区域。  
`--high_threshold`：双阈值中的高阈值，用于生成可靠运动 seed。  
`--temporal_consistency`：bbox 连续确认帧数，值越大越保守。  
`--warmup_frames`：预热帧数，预热阶段不输出 bbox。  
`--disable_global_motion`：关闭全局平移补偿。  
`--quiet`：减少命令行日志输出。

## 6. 算子特点

1. 不使用深度学习，完全基于传统图像处理方法。
2. 对高 ISO 随机噪声有自适应抑制能力，不容易把孤立噪点判为运动。
3. 对整帧亮度变化、曝光波动和画面闪烁有补偿机制。
4. 通过双阈值和形态学处理，运动区域比单纯帧差更完整。
5. 通过短时运动支持和静态重复区域抑制，减少亮斑、测试卡、固定纹理的误检。
6. 输出 mask、bbox、overlay 视频和诊断日志，便于人工检查和后续接入其他模块。
