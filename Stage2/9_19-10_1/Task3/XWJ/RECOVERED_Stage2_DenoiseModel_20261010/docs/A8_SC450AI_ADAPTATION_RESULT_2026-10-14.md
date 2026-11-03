# A8 sc450ai 传感器适配结果

日期：2026-10-14（Asia/Shanghai）

状态：**全部否决。step-100/200/300/400/500 均不得登记为可用 checkpoint。**

## 1. 最终裁决

本轮 sc450ai blind-spot 自监督微调没有修复 A3 在黑物体运动序列上的高频塌缩。
最保守的 step-100 与 A3 几乎相同，但保留集高频仍略降；随着训练步数增加，
输出负偏置和端点裁剪持续增大。step-500 的 fixed-ISP 视频中，黑衣内部结构仍然
空洞、过软，没有出现可确认的细节恢复。

因此：

- 不保留任何 sc450ai A8 checkpoint 作为推理候选；
- 不用任何 A8 sc450ai checkpoint 替换 A3；
- A3 继续作为不可变比较基线，但其原有 sc450ai 部署结论仍是“不支持”；
- 已生成的 checkpoint 仅作为可复现实验产物，不得用于发布或部署。

## 2. 数据划分与监督语义

两个现有 sc450ai 序列都没有独立干净真值。本实验没有使用旧教师输出，也没有把
动态时域平均当作干净目标。

原 A3 跨传感器审计冻结协议经用户于 2026-10-13 明确授权“数据集可自行合理划分”
后修订。`804x_lightchanging` 因参与 A8 训练，不再被声称为独立 A8 测试；完整
`1435x_motion_black_object` 按序列保留，始终不参与训练或目标构造。

| 用途 | 序列与帧范围 | 监督/评估方式 |
|---|---|---|
| blind-spot 训练 | `sc450ai_804x_lightchanging` 帧 0-119 | 单帧自监督；同一帧复制为 8 帧输入；5% packed 空间位置在全部时刻与全部通道同时遮挡；只在遮挡位置对原始 RAW 计算 Charbonnier 损失 |
| 同域保留 | `sc450ai_804x_lightchanging` 帧 120-149 | 完全排除在训练外；固定使用帧 142-149 做无参考检查 |
| 跨序列保留 | 完整 `sc450ai_1435x_motion_black_object` | 不参与训练或调参；固定 3 个 clip、24 帧做无参考压力测试和 fixed-ISP 视频 |
| 防遗忘回放 | MIS20S1 现有 A3 监督产物 | 与 sc450ai 样本按 50% 概率混合 |

遮挡位置从半径 2 的同 CFA packed 通道邻域替换，且 8 个时刻使用相同遮挡位置，
从而避免模型从相邻时刻直接读取目标像素。数据集目录始终只读。

## 3. 训练边界

- 初始化：A3 step-5000 EMA；
- A3 SHA-256：`97c4332f685f98959b243e4b100434469f714aed6329e100d22b6d8552592e0b`；
- 训练步数：500；
- 实际样本：sc450ai blind-spot 238 步，MIS20S1 回放 262 步；
- 学习率：`1e-5`；
- patch：`64x64` packed RAW；
- 精度：BF16 autocast；
- EMA decay：`0.999`；
- 可训练范围：仅 `norm`、`conv_after_body`、`conv_last`；
- 可训练参数：6 个张量，共 12,400 个参数。

step-500 权重审计结果：

| 权重集合 | 变化张量 | 未授权变化 | 最大绝对变化 |
|---|---:|---:|---:|
| live | 6 | 0 | 0.002682 |
| EMA | 6 | 0 | 0.000588 |

A3 checkpoint 在训练后哈希不变。

## 4. 全 checkpoint 筛选

以下均为完整 `1435x_motion_black_object` 保留集的无参考指标。高频与裁剪的基线是
A3；所有输出的 legal RAW fraction 均为 `1.0`。

| 权重 | 输出偏置 (DN) | temporal ratio | motion ratio | static ratio | 高频保持率 | clip fraction | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| A3 | -4.8287 | 0.063235 | 0.056887 | 0.087908 | 0.126621 | 0.032798 | 不支持 sc450ai，但作为基线 |
| step-100 | -4.8553 | 0.063170 | 0.056844 | 0.087790 | 0.126548 | 0.032780 | 高频未恢复，近似无效 |
| step-200 | -4.9354 | 0.063033 | 0.056817 | 0.087416 | 0.126429 | 0.034266 | 高频下降，裁剪增加 |
| step-300 | -5.0538 | 0.062856 | 0.056810 | 0.086863 | 0.126329 | 0.037160 | 高频下降，裁剪继续增加 |
| step-400 | -5.2364 | 0.062634 | 0.056849 | 0.086045 | 0.126227 | 0.042748 | 明显负偏置和裁剪恶化 |
| step-500 | -5.3414 | 0.062595 | 0.056962 | 0.085638 | 0.126079 | 0.046436 | 自动门槛失败，否决 |

筛选要求保留集高频至少比 A3 增加 `0.02`，同时裁剪不超过 A3 + `0.005`。
实际没有任何 checkpoint 恢复高频，`passing_steps` 为空。

同域未训练尾段也呈相同趋势：

| 权重 | 高频保持率 | clip fraction |
|---|---:|---:|
| A3 | 0.684561 | 0.015772 |
| step-100 | 0.684235 | 0.015589 |
| step-200 | 0.683612 | 0.016542 |
| step-300 | 0.682960 | 0.018740 |
| step-400 | 0.682244 | 0.023488 |
| step-500 | 0.681220 | 0.026944 |

权威筛选文件：

`artifacts/a8_sensor_adaptation/sc450ai/checkpoint_screen_20261013.json`

SHA-256：`5da0dbc6dabc400acea84a5380a5d5419cf89b426630f3d7d431401023dc1205`

## 5. step-500 完整安全门槛

step-500 通过了 legal RAW、运动响应、静态闪烁和 MIS20S1 防遗忘检查，但以下三项
失败：

1. `804x` 未训练尾段裁剪由 `0.015772` 增至 `0.026944`；
2. `1435x` 保留集裁剪由 `0.032798` 增至 `0.046436`；
3. `1435x` 高频由 `0.126621` 降至 `0.126079`，未达到任何细节恢复。

MIS20S1 回放的 PSNR 与高频没有遗忘，但 temporal MAD 从 `1.7069 DN` 增至
`3.2940 DN`。这进一步说明该权重不应提升为通用模型。

完整评估：

`artifacts/a8_sensor_adaptation/sc450ai/evaluation_step500_20261013/evaluation.json`

## 6. Fixed-ISP 视频证据

视频：

`artifacts/a8_sensor_adaptation/sc450ai/evaluation_step500_20261013/fixed_isp_video/sc450ai_1435x_input_a3_a8_fixed_isp.mp4`

- 布局：`Input | A3 baseline | A8 sc450ai step-500`；
- 内容：完整保留集的 3 个固定 clip，共 24 帧；
- 分辨率：1920x964；
- 帧率与时长：6 fps，4 秒；
- 编码：HEVC lossless yuv444p；
- SHA-256：`772dc29318c93112e74d7ad8e914c90323cb9e461c9078d1f567f5db04ddf5bc`。

首帧、第二段中间帧和末帧检查结果：

- A8 没有引入明显新双影或边缘断裂；
- A8 与 A3 的黑衣轮廓和内部结构几乎相同，仍表现为空洞、过软；
- 没有观察到足以支持“细节恢复”的衣物纹理或边界改善；
- A8 整体略向更暗方向偏移，与 RAW 负偏置和裁剪增加一致；
- 因此视频不能推翻自动门槛的否决结论。

## 7. checkpoint 哈希

| 步数 | SHA-256 |
|---:|---|
| 100 | `6c5a629c6377287f71320f7e249c854747674b7cf986d3e97c63968220d7ce2a` |
| 200 | `df697926e2e6d7a792a018dbfd7106f45c4b769180c1f3e6250abd978281b44a` |
| 300 | `f49ed7653b57e0ec6722a59c913fb2d6b130c448068d32c24b49f0792b6060ca` |
| 400 | `1aeed37d2c736ff6789d6562a8ce0702d26fd9a2a0878c17a4f42d3ee1073db5` |
| 500 | `44230ba8d64ef262f87ca9c2342c921761c6ded4df20405ed7e8d9bb63f8607e` |

## 8. 复现命令

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/run_a8_sc450ai_blindspot.py \
  --device cuda:0

CUDA_VISIBLE_DEVICES=0 \
  /home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/evaluate_a8_sc450ai.py \
  --device cuda:0

CUDA_VISIBLE_DEVICES=0 \
  /home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/screen_a8_sc450ai_checkpoints.py \
  --device cuda:0
```

本机 CUDA runtime 中 `CUDA_VISIBLE_DEVICES=0` 对应 RTX 4090。

## 9. 未完成项与风险

- 用户仍需观看 fixed-ISP 视频进行主观确认，但视频确认只能加强或维持否决，不能在
  无干净真值且三项硬门槛失败时把 checkpoint 改判为通过；
- 当前已有 sc450ai 数据不足以给出可靠的监督式适配目标；
- 在“不重新拍摄、不得用旧教师或动态时域均值作真值”的约束下，本轮没有剩余的
  可接受 sc450ai checkpoint。若未来存在独立静态多帧、跨增益配对或真实干净 RAW，
  才能重新开启更强的 sc450ai 适配。
