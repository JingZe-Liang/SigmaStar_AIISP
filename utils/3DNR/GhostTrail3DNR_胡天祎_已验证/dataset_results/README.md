# H5 全数据集批处理结果

输入数据：

```text
D:\zhuo mian\H5
```

输出内容：

- 每个 scene 一个 `metrics_frame.csv`
- 每个 scene 一个 `metrics_summary.csv`
- 每个 scene 一个 RAW 亮度对比视频
- 每个 scene 一个简易 sRGB 对比视频
- 全数据汇总：`metrics_summary_all.csv`

视频列顺序：

```text
noisy | 2dnr | dataset_3dnr | ghost3dnr | clean
```

## 指标口径

- `mse / psnr`：全帧 RAW 域归一化指标，和 clean 比。
- `ssim`：RAW 2x2 CFA 亮度代理图上计算，使用下采样加速。
- `static_mse`：整段 scene 中长期稳定区域的 RAW 域 MSE，越低表示静止区越干净。
- `moving_mae`：相邻 clean 帧变化较大的运动区域 MAE，越高表示运动区域偏离 clean 越明显，通常对应更强拖影/残影。

注意：这个算法的目标不是整体 PSNR 最优，而是“静止区强去噪 + 运动区可见拖影”。因此整体 MSE/PSNR 下降是预期现象。

## 全数据汇总

| method | frames | mse | psnr | ssim | static_mse | moving_mae |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| noisy | 2100 | 0.0000876247 | 41.261711 | 0.989617 | 532.2549 | 39.0638 |
| 2dnr | 2100 | 0.0000875621 | 41.266806 | 0.989632 | 530.6646 | 39.0565 |
| dataset_3dnr | 2100 | 0.0000346773 | 46.304514 | 0.996093 | 86.4886 | 35.0007 |
| ghost3dnr | 2100 | 0.0034972733 | 32.706464 | 0.951632 | 63.3369 | 195.1924 |

结论：

- 静止区：`ghost3dnr static_mse=63.3369`，优于 noisy、2dnr，也优于数据集已有 `dataset_3dnr=86.4886`。
- 运动区：`ghost3dnr moving_mae=195.1924`，显著高于 noisy/dataset_3dnr，说明运动区域被引入了强拖影伪影。
- 视觉展示：`scene_1`、`scene_6`、`scene_8` 的运动更明显，适合展示拖影；`scene_5` 更适合展示慢速场景下的静止区去噪。

## 视频路径

```text
scene_1/scene_1_raw_video.gif
scene_1/scene_1_srgb_video.gif
scene_2/scene_2_raw_video.gif
scene_2/scene_2_srgb_video.gif
scene_3/scene_3_raw_video.gif
scene_3/scene_3_srgb_video.gif
scene_4/scene_4_raw_video.gif
scene_4/scene_4_srgb_video.gif
scene_5/scene_5_raw_video.gif
scene_5/scene_5_srgb_video.gif
scene_6/scene_6_raw_video.gif
scene_6/scene_6_srgb_video.gif
scene_7/scene_7_raw_video.gif
scene_7/scene_7_srgb_video.gif
scene_8/scene_8_raw_video.gif
scene_8/scene_8_srgb_video.gif
scene_9/scene_9_raw_video.gif
scene_9/scene_9_srgb_video.gif
```
