# H5 数据集验证记录

数据：

```text
D:\zhuo mian\H5\scene_5\shard_0.h5
```

`scene_5` 在数据说明中属于拖影较少、运动较平缓的视频，因此适合验证“原本拖影较少，经过 3DNR 后制造运动拖影”的目标。

## noisy 双帧维度

`noisy` 的形状是 `[T, 2, H, W]`。实测：

- `noisy[t, 0]` 是上一帧。
- `noisy[t, 1]` 是当前帧。
- `noisy[t, 1] == noisy[t + 1, 0]`。

所以 H5 demo 默认使用 `--noisy-channel 1`。

## 当前算法

最初的纯递归 IIR 版本会把当前运动目标平均到历史背景里，视觉上容易出现“人没了”的问题。

当前版本改为 multi-tap delayed echo 3DNR：

1. 先生成当前帧保底的时域去噪结果。
2. 再把过去多帧的运动残差叠加回当前帧。
3. 当前运动位置用 `motion_current_floor` 保留当前目标，拖影主要出现在过去目标位置。

## 推荐参数

```text
static_history_weight = 0.88
motion_history_weight = 0.94
motion_current_floor = 0.85
trail_strength = 1.35
trail_decay = 0.72
echo_delay = 2
echo_taps = 4
motion_threshold_scale = 1.0
motion_softness = 0.7
sigma01 = 0.012
```

这个参数组合的设计是：

- 静止区仍做明显时间平均，降低随机噪声。
- 当前运动目标不会被历史背景直接抹掉。
- 过去 2/4/6/8 帧的运动残差会叠加到当前结果中，视频里能看到拖尾。

## frame20 指标

评估帧：`scene_5/shard_0/frame20`

| method | static_mse | moving_mae | departed_mae |
| --- | ---: | ---: | ---: |
| noisy | 1162.2251 | 62.5484 | 55.3817 |
| 2dnr | 1161.1771 | 62.5485 | 55.3815 |
| dataset_3dnr | 277.4446 | 55.3736 | 39.6639 |
| ghost3dnr | 167.5753 | 89.8850 | 66.3302 |

解释：

- `static_mse` 越低越好。`ghost3dnr` 静止区比 noisy/2dnr 明显更干净，也优于数据集已有 3DNR。
- `moving_mae` 和 `departed_mae` 在这里用于观察运动伪影/残影，越高表示和 clean 的运动区域偏离越大。`ghost3dnr` 明显高于 noisy/2dnr/dataset_3dnr，说明运动区域被制造出拖影。

## 多帧平均指标

评估范围：`scene_5/shard_0/frame5..frame29`

| method | static_mse_mean | moving_mae_mean | departed_mae_mean |
| --- | ---: | ---: | ---: |
| noisy | 1136.6279 | 62.0151 | 54.6701 |
| 2dnr | 1135.5709 | 62.0150 | 54.6696 |
| dataset_3dnr | 278.4201 | 54.9441 | 39.3804 |
| ghost3dnr | 211.7509 | 94.3118 | 68.4116 |

## 可视化输出

```text
hty/ghost_trail_3dnr/experiments/scene5_shard0_multi_echo/scene_5_shard_0_frame0020_panel.png
hty/ghost_trail_3dnr/experiments/scene5_shard0_multi_echo/scene_5_shard_0_frame0020_ghost_residual.png
hty/ghost_trail_3dnr/experiments/scene5_shard0_multi_echo/scene_5_shard_0_frames0000_0029_raw_video.gif
hty/ghost_trail_3dnr/experiments/scene5_shard0_multi_echo/scene_5_shard_0_frames0000_0029_srgb_video.gif
hty/ghost_trail_3dnr/experiments/scene1_shard0_multi_echo/scene_1_shard_0_frames0000_0029_raw_video.gif
hty/ghost_trail_3dnr/experiments/scene1_shard0_multi_echo/scene_1_shard_0_frames0000_0029_srgb_video.gif
```

panel 顺序：

```text
noisy | 2dnr | dataset_3dnr | ghost3dnr | clean
```

在 `ghost3dnr` 列可以看到运动目标附近的历史残留；残差图中红/蓝错位块集中在运动行人/骑车区域。
