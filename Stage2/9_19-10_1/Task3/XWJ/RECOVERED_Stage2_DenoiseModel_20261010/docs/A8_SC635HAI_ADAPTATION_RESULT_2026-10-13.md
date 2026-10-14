# A8 sc635hai 传感器适配结果

日期：2026-10-13（Asia/Shanghai）

状态：**有条件通过，仅登记为 sc635hai 专用可选检查点；不得替换 A3。**

## 1. 最终裁决

sc635hai 的 500 步保守微调通过全部自动安全检查，训练域静态质量小幅改善，
动态保留集上的合法 RAW、裁剪、高频、运动和静态时序指标均未出现实质退化。
固定 ISP 首、中、尾代表帧未观察到相对 A3 新增的重影、断边、明显色偏或静态背景污染。

因此保留以下 EMA 权重，供 sc635hai 数据上的可选推理和人工视频复核：

- checkpoint：`artifacts/a8_sensor_adaptation/sc635hai/training_20261013/checkpoint_step_000500.pth`
- SHA-256：`8e4980a43767b5b83ecc064e8cdce2f685ea452e94af48a1f4ca252004765b3f`
- 使用权重：`ema`
- 训练范围：仅 `norm`、`conv_after_body`、`conv_last`
- 可训练参数：6 个张量，共 12,400 个参数

该结果不是通用模型替代方案。A3 step-5000 仍是默认基线，其文件未被覆盖或修改：

- checkpoint：`artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`
- SHA-256：`97c4332f685f98959b243e4b100434469f714aed6329e100d22b6d8552592e0b`

## 2. 数据划分与监督约束

本实验按用户于 2026-10-13 的明确授权，重新划分原 A3 跨传感器审计数据：

| 用途 | 序列 | 使用方式 |
|---|---|---|
| sc635hai 适配 | `sc635hai_131x_low_cct`，150 帧 | 奇偶帧分半；输入与目标来自互斥帧池；运动、低置信度和饱和区域不计损失 |
| 动态保留集 | `sc635hai_16x_motion_color`，200 帧 | 不参加反向传播，不作为监督目标，仅做无参考压力测试和 fixed-ISP 视频复核 |
| 防遗忘回放 | MIS20S1 A3 现有监督产物 | 与 sc635hai 样本按 50% 概率混合 |

训练没有使用旧教师输出作为干净真值，也没有把动态序列的时域平均当作干净真值。
数据集目录始终只读。

500 个确定性训练步实际抽样为：sc635hai 238 步，MIS20S1 回放 262 步。
优化器学习率为 `1e-5`，patch 为 `64x64`，BF16 autocast，EMA decay 为
`0.999`。模型主干保持冻结并处于 eval 模式。

## 3. 权重边界审计

以 A3 为参照，普通模型权重和 EMA 权重均只有授权的 6 个输出头张量发生变化，
没有未授权参数变化：

| 权重集合 | 变化张量 | 未授权变化 | 最大绝对变化 |
|---|---:|---:|---:|
| live | 6 | 0 | 约 0.002201 |
| EMA | 6 | 0 | 约 0.0004765 |

## 4. 定量结果

### 4.1 sc635hai 训练域静态尾段

参考来自与输入帧池互斥的另一奇偶帧池，只在静态、有效、未饱和位置统计。

| 输入/参考 | 指标 | A3 | A8 | A8 - A3 |
|---|---|---:|---:|---:|
| 偶数输入 / 奇数参考 | PSNR (dB) | 52.3535 | 52.3875 | +0.0340 |
| 偶数输入 / 奇数参考 | 高频保持率 | 0.919860 | 0.918145 | -0.001715 |
| 偶数输入 / 奇数参考 | temporal MAD (DN) | 2.1628 | 2.1247 | -0.0381 |
| 偶数输入 / 奇数参考 | normalized clip fraction | 0.007408 | 0.006444 | -0.000964 |
| 奇数输入 / 偶数参考 | PSNR (dB) | 52.3552 | 52.3884 | +0.0331 |
| 奇数输入 / 偶数参考 | 高频保持率 | 0.919991 | 0.918279 | -0.001712 |
| 奇数输入 / 偶数参考 | temporal MAD (DN) | 2.1624 | 2.1242 | -0.0383 |
| 奇数输入 / 偶数参考 | normalized clip fraction | 0.007428 | 0.006460 | -0.000968 |

两组 legal RAW fraction 均为 `1.0`。高频变化很小，且仍接近参考值 1；PSNR、
时序稳定性和裁剪比例均略有改善。

### 4.2 sc635hai 动态保留集

动态序列没有独立干净真值，以下指标只能用于退化检查，不能单独证明画质提升。

| 指标 | A3 | A8 | A8 - A3 |
|---|---:|---:|---:|
| temporal MAD (DN) | 14.6745 | 14.6218 | -0.0528 |
| temporal MAD / input | 0.284194 | 0.283172 | -0.001022 |
| motion temporal ratio | 0.599602 | 0.596438 | -0.003165 |
| static temporal ratio | 0.101329 | 0.101169 | -0.000160 |
| spatial high-frequency ratio | 0.763605 | 0.763078 | -0.000527 |
| legal RAW fraction | 1.000000 | 1.000000 | 0 |
| normalized clip fraction | 0.021994 | 0.019344 | -0.002650 |
| mean output bias (DN) | -2.3691 | -2.9808 | -0.6117 |
| mean absolute output change (DN) | 29.7100 | 29.8552 | +0.1452 |

变化总体很小。A8 没有靠显著牺牲运动响应换取静态平滑；但其平均输出相对输入
比 A3 多约 `0.61 DN` 的负偏置，因此仍应限定为 sc635hai 专用实验权重。

### 4.3 MIS20S1 防遗忘回放

| 指标 | A3 | A8 | A8 - A3 |
|---|---:|---:|---:|
| PSNR (dB) | 16.3696 | 18.8586 | +2.4890 |
| 高频保持率 | 0.135638 | 0.341334 | +0.205696 |
| temporal MAD (DN) | 1.7069 | 3.2495 | +1.5426 |
| flat residual sigma (DN) | 260.0292 | 225.7331 | -34.2961 |
| legal RAW fraction | 1.0 | 1.0 | 0 |

回放集的重建误差与高频保持显著改善，但 temporal MAD 增加。其绝对值仍远低于
输入的 `65.2292 DN`，且这不是 sc635hai 的传感器内测试；因此记录为残余风险，
不据此把 A8 提升为通用默认模型。

## 5. Fixed-ISP 视觉证据

统一 fixed-ISP 视频：

`artifacts/a8_sensor_adaptation/sc635hai/evaluation_step500_20261013/fixed_isp_video/sc635hai_16x_input_a3_a8_fixed_isp.mp4`

- 布局：`Input | A3 baseline | A8 sc635hai`
- 内容：3 个互不重叠的动态 clip，共 24 帧
- 编码：HEVC，1920x964，6 fps，4 秒
- SHA-256：`f2b08af63a4570d5840683c06a7766d56c2f8ca421f797f3de62ba9076155cd2`
- 每列上半部为全图，下半部为相同中心细节区域

预览图：

`artifacts/a8_sensor_adaptation/sc635hai/evaluation_step500_20261013/fixed_isp_video/sc635hai_16x_input_a3_a8_fixed_isp_frame0000.png`

首帧、第二段中间帧和末帧的逐帧检查结果：

- 人体轮廓和移动边缘与 A3 基本一致；
- 未见新增双影、拖影、边缘断裂或局部突变；
- 色卡、棋盘、静态人像和细小装饰未见相对 A3 的明显额外平滑；
- 静态背景未见新增闪烁或区域性颜色污染；
- A3 与 A8 的视觉差异很小，符合仅更新输出头的预期。

完整视频仍应由用户进行最终肉眼复核，尤其关注运动人体边缘和暗部背景随时间的变化。

## 6. 自动安全门槛

`evaluation.json` 中 9 项检查全部通过：

1. 静态 PSNR 不低于 A3 - 0.10 dB；
2. 静态高频距离不比 A3 恶化 0.02；
3. 动态 legal RAW fraction 合法；
4. 动态裁剪比例受控；
5. 动态高频保持受控；
6. 动态运动响应受控；
7. 动态静态区域闪烁受控；
8. MIS20S1 PSNR 不遗忘；
9. MIS20S1 高频保持不遗忘。

权威评估文件：

`artifacts/a8_sensor_adaptation/sc635hai/evaluation_step500_20261013/evaluation.json`

## 7. 复现命令

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/run_a8_sensor_adaptation.py \
  --config configs/a8_sc635hai_adaptation_20261013.json

CUDA_VISIBLE_DEVICES=0 \
  /home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/evaluate_a8_sc635hai.py \
  --device cuda:0
```

本机 CUDA runtime 中 `CUDA_VISIBLE_DEVICES=0` 对应 RTX 4090。

## 8. 后续状态

- 用户仍需观看上述 fixed-ISP 视频并给出主观画质意见；
- sc450ai 已于 2026-10-14 完成 blind-spot 适配、全 checkpoint 筛选和 fixed-ISP
  复核，全部 checkpoint 被否决；详见
  `docs/A8_SC450AI_ADAPTATION_RESULT_2026-10-14.md`；
- A8 两传感器总裁决见
  `docs/A8_MULTI_SENSOR_ADAPTATION_FINAL_2026-10-14.md`。
