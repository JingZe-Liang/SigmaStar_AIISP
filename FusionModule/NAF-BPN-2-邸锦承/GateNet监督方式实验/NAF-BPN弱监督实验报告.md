# NAF-BPN 无监督融合实验报告

## 1. 实验结果

- 保留原 NAF-BPN/BPN 主干和 Stage 1 预训练权重。
- 在公司数据集上完成 Stage 2 无监督微调
- 推理只使用 `2DNR、3DNR、source_t、source_{t-1}`，不读取 MD cache。
- 已完成 `128x` 和 `645x` 各 200 帧推理；本报告重点整理 `645x` 结果。

## 2. 代码改动

- `model.py`：第四路输入由外部 `motion_mask` 改为 `noisy_previous`；模型内部使用简单帧差；保留原 BPN basis/coeff 和参数 shape。
- `data.py`：增加公司数据弱监督数据集、七帧 trimmed-mean proxy、静止/运动标签、序列 offset/noise sigma 估计；优化为先裁剪再读取转换。
- `losses.py`：增加静止区 proxy 重建、静止区梯度、运动区软 2DNR anchor、候选一致性和低权重 masked noisy loss。
- `train.py`：恢复 Stage 1/Stage 2 两阶段入口，Stage 2 支持严格加载旧权重和跨序列 fold。
- `infer.py`：移除推理阶段 MD 依赖，加入 Bayer tiled inference、halo、AMP 和性能记录。
- `render_videos.py`：增加 ISP 曝光倍率和随机帧保存功能。
- `render_difference_maps.py`：增加 AI-2DNR 差分图、红色单独标注和绿色单独标注。

## 3. 生成视频

`645x` 经过 ISP 后使用自动曝光标定结果的 `5×` 显示曝光：

- AI 融合视频：[645x_ai_fused.mp4](videos/645x_ai_fused.mp4)
- 四宫格对比视频（Noisy / 2DNR / 3DNR / AI）：[645x_comparison.mp4](videos/645x_comparison.mp4)
- 视频信息：[video_manifest.json](videos/video_manifest.json)

视频规格：`1920×1080`、`25 FPS`、`200` 帧、H.264；视频校验通过。

## 5. 随机帧图片

随机帧号：`21、23、26、66、90、111、133、135、152、157`。

- AI 单帧图片目录：`random_frames/ai_fused`
- 四宫格对比图片目录：`random_frames/comparison`

## 6. AI-2DNR 残差图

差分定义为：

```text
residual = AI_fused - 2DNR
```

计算域为扣除黑电平后的线性 RAW，死区阈值为 `0.001`，约 `4` 个 RAW code。

- 红绿差分图： [difference_maps_ai_minus_2dnr](residual_maps/red_green)
- 仅红色图（AI 高于 2DNR）： [difference_maps_red_only](residual_maps/red_only)
- 仅绿色图（AI 低于 2DNR）： [difference_maps_green_only](residual_maps/green_only)

颜色说明：

- 黑色：差异在死区内，或当前版本未选择该方向标注。
- 红色：AI 输出高于 2DNR。
- 绿色：AI 输出低于 2DNR。
- 差分图使用统一 `99%` 分位色阶，超出色阶的差异会饱和显示。

示例：[frame_0090_ai_minus_2dnr.png](residual_maps/red_only/frame_0090_ai_minus_2dnr.png)


