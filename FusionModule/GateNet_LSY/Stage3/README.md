# GateNet Stage3

Stage3 是基于本目录 `comment.txt` 的分阶段实验实现，目标是验证融合和运动检测能否解耦：先训练稳定的 fusion，再训练不修改 fusion 的独立时序 motion 分支。

## 训练策略

### Phase 1: fusion-only

- 只优化融合损失，关闭原有 motion auxiliary loss；
- 默认使用很弱的静止区域 alpha 偏好：`static_alpha_weight=0.1`；
- 默认不使用静止区域 D2 惩罚：`static_d2_weight=0`；
- 输出 `phase1_best.pt`，作为融合 RAW 的来源。

### Phase 2: independent temporal motion

- 从 Phase 1 最优 checkpoint 开始并冻结融合网络；
- 新建 `TemporalMotionNet`，输入当前帧及前后帧的 noisy temporal features；
- 使用 focal BCE，默认正样本权重 `4`、gamma `2`；
- motion 指标阈值为 probability `>=0.7`。

Phase 2 不会改变 `fusion.raw`。`phase2_best.pt` 同时保存冻结的 fusion state 和独立 `motion_model`，供 `infer_motion_stage3.py` 使用。

## 训练

从本目录运行：

```powershell
cd "D:\University\Fusion\Phase Final\FusionModule\GateNet_LSY\Stage3"
python train_stage3.py `
  --fold 128_to_645 `
  --output runs\fold_128_to_645_temporal `
  --phase1-epochs 50 `
  --phase2-epochs 20 `
  --batch-size 8 `
  --crop-size 256 `
  --train-samples 2048 `
  --val-samples 512 `
  --device cuda
```

反向交叉序列使用相同配置并改为 `--fold 645_to_128 --output runs\fold_645_to_128_temporal`。`--fold all` 仅适合最终拟合，不适合作为泛化结论。

## 产物和指标

- `phase1_best.pt` / `phase1_last.pt`: fusion checkpoint；
- `phase2_best.pt` / `phase2_last.pt`: frozen fusion + independent motion；
- `history.csv`: 两阶段逐 epoch 指标；
- `config.json`: fold、采样、损失权重和推理统计量。

重点查看 `val_output_proxy`、`val_motion_precision`、`val_motion_recall` 和 `val_motion_loss`。`val_fusion_total` 包含 Stage3 新增的 static alpha 项，不能直接和 Stage2 总 loss 横比。

## 推理流程

1. 用 Stage3 入口和 `phase1_best.pt` 生成 fusion RAW。旧 checkpoint 若缺少推理统计量，先准备副本：

```powershell
python prepare_checkpoint.py `
  --checkpoint runs\fold_128_to_645_temporal\phase1_best.pt `
  --reference ..\Stage2\runs\fold_128_to_645\best.pt `
  --output runs\fold_128_to_645_temporal\phase1_best_infer.pt
```

```powershell
python infer_stage3.py `
  --checkpoint runs\fold_128_to_645_temporal\phase1_best.pt `
  --output outputs\fold_128_to_645_temporal `
  --sequences 645x
```

2. 用独立 Phase 2 分支覆盖 diagnostic motion stream，然后使用 Stage3 的 ISP 和视频入口完成后处理：

```powershell
python infer_motion_stage3.py `
  --checkpoint runs\fold_128_to_645_temporal\phase2_best.pt `
  --output outputs\fold_128_to_645_temporal `
  --sequences 645x `
  --device cuda

python render_isp_stage3.py `
  --input outputs\fold_128_to_645_temporal `
  --sequences 645x

python make_compatible_video_stage3.py `
  --root outputs\fold_128_to_645_temporal `
  --sequences 645x

python make_four_view_stage3.py `
  --inference-root outputs\fold_128_to_645_temporal `
  --output-root outputs\four_view_128_to_645_temporal `
  --sequences 645x
```

`infer_motion_stage3.py` 只更新 `predicted_motion_u8.raw`，不重写 `fusion.raw`。其余三个 Stage3 入口依次生成 ISP master、compatible-video 和 four-view。

当前最终四视图：

`outputs\four_view_128_to_645_temporal\645x\stage3_motion_2dnr_3dnr_fusion.mp4`

## 当前结果：双向 temporal runs

两个正式 run 均使用 Phase 1 50 epoch、Phase 2 20 epoch。最佳验证结果如下：

| Fold | Fusion best epoch | `val_output_proxy` | `val_fusion_total` | Motion best epoch | Motion loss | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `128_to_645` | 3 | `0.281995` | `0.092899` | 13 | `0.056311` | `0.689705` | `0.656709` |
| `645_to_128` | 12 | `0.241840` | `0.072258` | 16 | `0.065950` | `0.948707` | `0.255600` |

`128_to_645` 的 `645x` 独立推理在阈值 `0.7` 下：预测运动像素 `0.9959%`，目标 MD `0.9882%`，precision `0.6548`，recall `0.6599`。因此此前 `pred motion` 全红的问题已消失，运动区域主要集中在运动目标附近。反向 fold 的验证 precision 很高但 recall 偏低，说明该方向的 motion 分支偏保守，后续应优先检查阈值和跨序列分布差异。

## 指标评估与限制

- 新 temporal run 的 fusion proxy 为 `0.281995`，Stage2 `128_to_645` 为 `0.277268`，约高 `1.7%`。数值上尚未证明融合主任务提升，应结合四视图观察静止区域噪声和运动区域保留情况。
- `val_fusion_total` 不适合直接排序 Stage2 与 Stage3；优先比较同定义的 `val_output_proxy`。
- 当前两个 fold 都已用新版 independent temporal 分支完成训练，但只有 `128_to_645` 已生成最终 ISP/four-view 视频；反向 fold 目前先以 `history.csv` 指标为准。
- motion 指标方向性差异较大，暂时不建议继续赌单一 loss 权重；应先补齐反向 fold 的推理可视化，再决定是否调整 threshold、static alpha 或 focal 参数。

## 代码检查

```powershell
python -m py_compile motionnet_stage3.py train_stage3.py infer_motion_stage3.py
```
