# 算子名称：GhostTrail3DNR

## 1. 功能定位：

构造一版带明显运动拖影的 RAW 域 3DNR baseline，用于模拟传统时域降噪在运动区域产生 ghosting artifact 的退化样本。

## 2. 输入输出格式：

输入格式：

- 元素类型：单通道 Bayer RAW TIFF 序列，或 H5 中的 RAW 帧序列。
- Shape：TIFF 模式为单帧 `(H, W)`；H5 批处理模式读取 `[T, H, W]` 或包含 noisy/2dnr/3dnr/clean 的分片。
- Dtype：通常为 `uint16`，脚本内部转为 `float32` 处理。
- 是否需要归一化：不需要，内部按 `black_level` / `white_level` 归一化。
- 是否需要 metadata.json：TIFF 模式不需要；H5 数据集模式可读取场景目录下已有元信息，但不强依赖。

输出格式：

- TIFF 模式：每帧输出 `{stem}_ghost3dnr.tiff`。
- H5 批处理模式：输出 `metrics_summary_all.csv`、各 scene 的指标 CSV，以及少量 RAW / sRGB GIF 预览。
- Shape：与输入帧一致，或按命令参数裁剪生成预览。
- Dtype：TIFF 输出恢复为输入 RAW dtype。
- 是否已归一化：否。
- Effective bits：默认 12-bit `0~4095`。
- Container bits：通常为 16-bit TIFF。
- 指标：PSNR / SSIM 越高表示与 clean 更接近；moving MAE 用于观察运动区域拖影残差。

## 3. 方法简述：

核心是 motion-adaptive recursive temporal filter：

```text
Y_t = a_t * Y_{t-1} + (1 - a_t) * X_t
a_t = (1 - m_t) * a_static + m_t * a_motion
```

其中 `X_t` 是当前 noisy RAW，`Y_t` 是当前输出，`m_t` 是运动分数。为了主动制造拖影，代码额外叠加 multi-tap delayed echo：从过去若干帧提取运动残差并按衰减系数叠加到当前帧。

## 4. 运行示例：

处理 TIFF 序列：

```powershell
python ghost_trail_3dnr.py `
  --input-dir "D:\data\CRVD\scene1\ISO25600" `
  --output-dir "D:\data\ghost3dnr\scene1_ISO25600" `
  --glob "frame*_noisy3.tiff" `
  --black-level 0 `
  --white-level 4095
```

批量处理 H5 数据集：

```powershell
python run_h5_dataset.py `
  --dataset-root "D:\data\H5" `
  --output-root dataset_results `
  --crop-size 320 `
  --video-stride 1 `
  --ssim-stride 8 `
  --video-fps 12
```

## 5. 参数解析：

- `--static-history-weight`：静止区历史权重，越大静止区越干净但响应越慢。
- `--motion-history-weight`：运动区历史权重，越大拖影越明显。
- `--motion-current-floor`：运动目标当前帧保留比例，避免目标被完全抹掉。
- `--trail-strength`：延迟残影叠加强度。
- `--trail-decay`：多段残影的衰减系数。
- `--echo-delay` / `--echo-taps`：残影延迟帧距和叠加段数。
- `--motion-threshold-scale`：运动判定阈值相对噪声强度的倍数。
- `--motion-map-blur-radius`：运动图平滑半径。

## 6. 算子特点：

该 3DNR 算子不是追求最佳去噪，而是稳定地产生可控拖影，适合给后续 AI 融合模块提供“3DNR 运动伪影”对照样本。`dataset_results/` 和 `experiments/` 中保留了小规模实验结果，便于查看拖影强度和指标变化。
