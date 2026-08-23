# GateNet Stage2：去外部 MD 的 2DNR/3DNR 融合网络

本目录实现 GateNet 的第二版网络。Stage2 的核心目标是：在保持 2DNR/3DNR 融合效果接近旧版 Phase2 的同时，将外部 MOG2/MD 从推理流程中移除。

当前版本只训练融合门控网络，不重新训练 2DNR、3DNR 或 MOG2。MOG2/MD 只在训练阶段作为伪标签监督 motion head，推理阶段不再读取任何 MD 文件。

## 0. 当前版本状态

Stage2 相比旧版 Phase2 的主要变化：

- 去掉推理阶段的外部 MD/MOG2 输入；
- 将 MD 保留为训练阶段的辅助监督；
- 主网络输入由 `15` 通道减少到 `9` 通道；
- 在共享主干后接一个轻量 `motion_head`，由网络内部预测 motion map；
- 推理输入只包含 `2DNR`、`3DNR`、`noisy_t`、`noisy_t-1`、`noisy_t+1`；
- 推理输出额外保存 `predicted_motion_u8.raw`、FPS 和 CUDA 峰值显存统计。

当前模型参数量约为 `1702`，旧 Phase2 GateNet 约为 `9602`。

## 1. 网络结构

Stage2 网络仍然是像素级门控网络，不直接生成完整图像。它预测融合权重 `alpha`：

```text
Y = D2 + alpha * (D3 - D2)
```

其中：

- `alpha=0` 表示选择 2DNR；
- `alpha=1` 表示选择 3DNR；
- 输出始终是 2DNR 和 3DNR 的凸融合。

### 输入特征

Stage2 使用 `9` 个 packed-RGGB 特征通道：

```text
signed(D3-D2)                4 通道
abs(D3-D2) mean              1 通道
D3/D2 局部标准差差异         1 通道
noisy 前后帧时域变化         1 通道
D2 亮度                      1 通道
noisy 梯度                   1 通道
```

特征构造见：

```text
gatenet_stage2.py::build_gate_features
```

### 网络主体

网络结构为轻量共享主干 + 双 head：

```text
features(9)
  -> Conv3x3(base=12)
  -> depthwise-pointwise block, dilation=2
  -> depthwise-pointwise block
  -> fusion_head -> alpha
  -> motion_head -> predicted motion logit
```

`fusion_head` 输出融合权重，`motion_head` 只在训练阶段接受 MD 伪标签监督；推理时 motion head 的输出只作为可视化和诊断结果，不作为外部输入。

## 2. MD 使用方式

需要明确区分训练和推理：

```text
训练阶段：读取 Phase2/DERIVED/md_mog2 作为 motion 伪标签
推理阶段：不读取 md_mog2，不需要 --md-root，不依赖外部 MOG2
```

训练脚本中仍然有：

```powershell
--md-root
```

这是为了给 `motion_head` 提供监督。推理脚本 `infer_stage2.py` 没有 `--md-root` 参数，输出 `summary.json` 中也会记录：

```json
"external_md_used": false
```

## 3. 文件说明

| 文件 | 作用 |
|---|---|
| `gatenet_stage2.py` | Stage2 网络结构和 9 通道特征构造 |
| `fusion_loss_stage2.py` | 融合损失 + motion 辅助监督 |
| `train_stage2.py` | Stage2 训练入口 |
| `infer_stage2.py` | 无外部 MD 的整帧推理入口 |
| `render_isp_stage2.py` | 面向显示的 highlight-safe ISP 渲染 |
| `make_four_view_stage2.py` | motion / 2DNR / 3DNR / fusion 四视图视频导出 |
| `test_stage2.py` | 基础回归测试 |

## 4. 环境

建议使用 Python 3.10+，依赖沿用 Phase2：

```powershell
pip install numpy torch opencv-python pillow tqdm imageio-ffmpeg pytest
```

所有命令默认在 Stage2 目录下运行：

```powershell
cd "D:\University\Fusion\Phase Final\FusionModule\GateNet_LSY\Stage2"
```

## 5. 训练

### 5.1 两折跨序列训练

先跑两折跨序列实验，观察泛化情况：

```powershell
python train_stage2.py `
  --fold 128_to_645 `
  --output runs\fold_128_to_645 `
  --epochs 50 `
  --batch-size 8 `
  --crop-size 256 `
  --train-samples 2048 `
  --val-samples 512 `
  --device cuda

python train_stage2.py `
  --fold 645_to_128 `
  --output runs\fold_645_to_128 `
  --epochs 50 `
  --batch-size 8 `
  --crop-size 256 `
  --train-samples 2048 `
  --val-samples 512 `
  --device cuda
```

### 5.2 全量最终训练

两折结果没有明显退化后，再使用全部序列训练最终模型：

```powershell
python train_stage2.py `
  --fold all `
  --output runs\final_all `
  --epochs 50 `
  --batch-size 8 `
  --crop-size 256 `
  --train-samples 2048 `
  --val-samples 512 `
  --device cuda
```

`fold=all` 的训练和验证都来自 `128x`、`645x`，只能作为最终拟合参考，不能当作跨序列泛化结果。

### 5.3 训练输出

每个训练目录包含：

- `config.json`：训练配置、数据统计、模型参数和 MD 使用说明；
- `history.csv`：逐 epoch 指标；
- `last.pt`：最后一个 checkpoint；
- `best.pt`：验证指标最优 checkpoint。

当前实现里 `best.pt` 默认按 `val_total` 选择。Stage2 的 `val_total` 包含 motion 辅助损失，因此看最终融合效果时不要只看 `val_total`。

## 6. 推理

使用最终模型对两个序列整帧推理：

```powershell
python infer_stage2.py `
  --checkpoint runs\final_all\best.pt `
  --output outputs\inference_final_all `
  --sequences 128x 645x `
  --device cuda
```

每个序列输出：

- `fusion.raw`：完整 200 帧融合 RAW，little-endian uint16，右对齐 12-bit；
- `alpha_u8.raw`：融合权重，`0=2DNR`、`255=3DNR`；
- `predicted_motion_u8.raw`：网络内部预测的 motion map；
- `summary.json`：帧数、SHA-256、alpha/motion 均值、FPS、显存和模型参数。

推理默认有效建模区间为中间帧；不满足时间窗的起始和末尾帧直接输出 2DNR。

## 7. ISP 和视频

### 7.1 Highlight-safe ISP

Stage2 提供了显示用 ISP 渲染脚本，主要用于缓解 `645x` 显示时的过曝问题：

```powershell
python render_isp_stage2.py `
  --dataset-root "D:\University\Fusion\Phase Final\Phase2\DATASET" `
  --input outputs\inference_final_all `
  --sequences 128x 645x
```

该步骤只影响 RAW 到视频的显示渲染，不改变 `fusion.raw`，也不影响训练。

### 7.2 四视图视频

生成 motion / 2DNR / 3DNR / fusion 四视图：

```powershell
python make_four_view_stage2.py `
  --dataset-root "D:\University\Fusion\Phase Final\Phase2\DATASET" `
  --inference-root outputs\inference_final_all `
  --output-root outputs\four_view_stage2 `
  --sequences 128x 645x
```

布局为：

```text
┌────────────────┬────────────────┐
│ predicted MD   │ 2DNR           │
├────────────────┼────────────────┤
│ 3DNR           │ fusion         │
└────────────────┴────────────────┘
```

这里的 predicted MD 是 Stage2 内部 motion head 的预测结果，不是外部 MOG2。

## 8. 指标说明

当前没有真实干净 RAW GT，因此训练中使用 proxy 指标作为替代评价。

常用指标：

| 指标 | 含义 |
|---|---|
| `val_output_proxy` | 融合输出的 proxy，越低越好 |
| `val_d2_proxy` | 只使用 2DNR 的 proxy |
| `val_d3_proxy` | 只使用 3DNR 的 proxy |
| `val_fusion_total` | 不含 motion 辅助损失的融合损失 |
| `val_total` | 融合损失 + motion 辅助损失 |
| `val_motion_precision` | motion head 相对 MOG2 伪标签的 precision |
| `val_motion_recall` | motion head 相对 MOG2 伪标签的 recall |

看最终融合效果时，优先看：

1. `val_output_proxy` 是否低于 `val_d2_proxy` 和 `val_d3_proxy`；
2. `val_fusion_total` 的趋势；
3. 导出的四视图和 `fusion.raw` 结果。

`val_total` 可以观察整体训练是否收敛，但不能直接等价为最终融合画质。

## 9. 当前实验结果摘要

当前已经完成的 Stage2 训练结果：

| 训练 | 最佳融合 epoch | `val_output_proxy` | 对 2DNR | 对 3DNR |
|---|---:|---:|---:|---:|
| `fold_128_to_645` | 3 | `0.277268` | 低约 `14.09%` | 低约 `5.79%` |
| `fold_645_to_128` | 6 | `0.240626` | 低约 `16.73%` | 低约 `5.22%` |
| `final_all` | 48 | `0.256339` | 低约 `15.72%` | 低约 `6.04%` |

和旧 Phase2 `final_all` 相比，Stage2 的最终融合 proxy 基本持平：

```text
Phase2 final_all: 0.256020
Stage2 final_all: 0.256339
```

因此当前结论是：Stage2 没有从 proxy 上证明最终融合画质明显超过旧版，但它在推理阶段去掉了外部 MD，且模型参数明显减少，工程部署收益明确。

## 10. 测试

运行基础回归测试：

```powershell
python -m pytest test_stage2.py -q
```

如果需要同时验证旧 Phase2 和 Stage2，可在仓库根目录运行对应测试。

