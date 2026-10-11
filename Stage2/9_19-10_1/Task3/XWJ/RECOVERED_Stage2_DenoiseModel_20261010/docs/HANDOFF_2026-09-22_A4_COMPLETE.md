# Stage2 DenoiseModel Handoff: A4 Complete, 1000x Diagnosis Next

更新时间：2026-09-22 15:45 (Asia/Shanghai)

交接编号：`A4-20260922-01`

交接状态：材料已核对，等待新会话只读核验。本文档是当前项目状态记录，不代替用户的新指令或授权。

## 1. 当前目标

本项目的总体目标是为 MIS20S1 RAW 数据训练一个能在高增益低照度场景下抑制噪声、同时保留真实高频细节的时空 RAW 去噪模型，重点解决旧 teacher 的过平滑问题。

当前阶段已经完成 A1、A2、A3，并完成 A4 的完整 200 帧评估。下一阶段不是立即宣称模型通过，而是先解释 `1000x` 评估失败的原因：确认它是输入预处理/注册问题，还是模型真实的跨增益泛化问题。

## 2. 工作目录与权威资料

- 项目根目录：`/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel`
- 主要计划：`docs/RVIDEFORMER_FINETUNE_PLAN_2026-09-20.md`
- 上一份交接背景：`docs/HANDOFF_2026-09-20.md`
- 数据集（只读）：`/HardDisk/jingzeliang/Data/Company_Dataset/dataset`
- 官方 RViDeformer（只读）：`/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/RViDeformer`
- Python：`/home/jingzeliang/miniconda3/envs/aaa_312/bin/python`
- 源码导入：命令需带 `PYTHONPATH=src`

A1 的权威参考文件位于：

`artifacts/finetune_mis20s1/supervision/references/`

其中关键文件是 `04_shdarkroom_10x_clean_reference.npy`、`04_shdarkroom_10x_valid_region.npy`、`a1_reference_report.json` 和 `a1_reference_manifest.json`。A1 报告的质量门已通过。

## 3. 已完成且已验证

### A1：监督参考

- 生成了 `04/shdarkroom/10x` 的 10x clean reference，形状为 `(4, 540, 960)`，单位为 12-bit DN。
- 生成静态 valid region、confidence、FPN/DSNU/PRNU 文件。
- A1 report 的 split-half、cross-gain 和 645x 高频检查均通过。
- A1 参考的黑电平为：`R=266.125, Gr=275.625, Gb=275.375, B=267.0`。

### A2：噪声模型

- 噪声模型、scene PTC、真实性报告和 synthetic pair manifest 已写入：
  `artifacts/finetune_mis20s1/noise_model/`
- 已生成 100x 到 25600x 的 synthetic noisy pairs。
- 对应代码提交：`3c26523 feat(finetune): add A2 noise model and synthetic pairs`。

### A3：Phase A RViDeformer fine-tuning

- 使用 8 帧时空窗口、packed `4x64x64` patch、真实 cross-gain 窗口和 A2 synthetic pairs。
- `1000x` 是训练 holdout；真实训练增益包含 `128x/313x/632x/645x`，另混合 synthetic pairs。
- 4090 正式训练已从 step 200 恢复到 step 5000，正常结束。
- 最终 checkpoint：
  `artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`
- checkpoint 已验证包含 step 5000、model/EMA 各 1265 个参数键、optimizer 状态。
- 相关提交：
  - `f018191 feat(finetune): run Phase A RViDeformer training`
  - `10beb2e feat(finetune): add temporal synthetic windows and resume`

### A4：完整评估

A4 评估代码：

- `src/raw_fusion/finetune/a4.py`
- `scripts/run_a4_eval.py`
- `src/tests/test_finetune_a4.py`

A4 使用 leave-one-gain-out 评估，完整处理 `1000x` 和 `645x` 各 200 帧，并比较：A3 EMA、旧 teacher、输入基线。空间 tile 为 64，tile batch 为 8，时间窗口为 8 帧。

最终输出目录：

`artifacts/finetune_mis20s1/evaluation/`

已确认存在：

- `a4_metrics.json`
- `metrics_1000x.json`、`metrics_645x.json`
- `prediction_1000x.npy`、`prediction_645x.npy`
- `old_teacher_1000x.npy`、`old_teacher_645x.npy`
- `four_panel_1000x.png`、`four_panel_645x.png`

关键定量结果：

| 增益 | A3 PSNR | 输入 PSNR | 旧 teacher PSNR | A3 高频比 | A3 平坦区残差 sigma | A3 时序 MAD |
|---|---:|---:|---:|---:|---:|---:|
| `1000x` | 18.84 dB | 18.73 dB | 18.78 dB | 0.339 | 226.19 DN | 3.50 DN |
| `645x` | 29.57 dB | 24.35 dB | 28.02 dB | 0.971 | 83.19 DN | 10.96 DN |

两个增益的 legal fraction 都是 `1.0`。`645x` 的结果符合“去噪同时保留细节”的方向；`1000x` 的输入、旧 teacher 和 A3 都明显偏暗，且 A3 高频比严重低于 A4 目标区间 `0.85–1.15`。

A4 不是“全部通过”：当前结论是 `645x` 基本有效，`1000x` 不合格且需要先诊断。

A4 代码提交：

- `65c44e8 feat(evaluation): add A4 leave-one-gain-out metrics`
- `00d78b7 fix(evaluation): convert Bayer panels to uint16`
- `3971228 feat(evaluation): checkpoint each gain for resume`

针对指标辅助测试：`src/tests/test_finetune_a4.py` 为 `2 passed`；A3 相关测试为 `4 passed`。完整 `pytest src/tests` 在此前运行较慢且被中断，不能记录为全套通过。

## 4. 当前运行与 GPU 状态

- A4 已结束，没有正在运行的 A4 进程。
- 4090 当前被外部 TOP_OUT/Dual_branch 训练进程 PID `3633828` 占用约 4.5 GB；不要终止该进程。
- 3090 被另一个外部进程 PID `3460781` 占用约 21.7 GB；不要终止该进程。
- 此环境中已验证 `CUDA_VISIBLE_DEVICES=0` 对应 RTX 4090；启动 GPU 工作前仍须用 `nvidia-smi` 再确认，不要盲目使用 UUID 映射。
- 用户已明确授权使用 4090；如果 GPU 被外部任务占用，不要杀进程，应先反馈或等待。

## 5. 已知问题与判断

### 5.1 `1000x` 的异常首先像输入管线问题

`1000x` 四宫格中 input、旧 teacher、A3 都比 A1 reference 暗很多；旧 teacher 也只有 `18.78 dB`。因此不能直接把问题归因于 A3 训练。

`1000x` manifest 中的关键字段：

- exposure fit slopes：`R=2.4439, Gr=2.1153, Gb=2.1232, B=3.0897`
- exposure fit intercepts：`R=-198.95, Gr=76.92, Gb=71.40, B=-385.82`
- integer registration shift：`shift_x=-31, shift_y=0`
- manifest 中的浮点 registration shift 约为 `(-61.94, -0.60)`，需要确认整数化/裁剪语义是否正确。

这使得下面三项必须先核对：曝光映射是否把 1000x 映到 10x reference DN；黑电平是否使用正确域；`shift_x=-31` 的 moving/base crop 是否与注册定义一致。

### 5.2 A4 的可恢复机制已实现

`run_a4_eval.py` 会在每个 gain 完成后立即保存 prediction、teacher output、metrics 和 panel；再次启动时会跳过完整的已完成 gain。若修改了输入预处理，必须使用新的输出目录或显式使对应旧文件失效，不能把旧预测当作新管线结果。

## 6. 新会话的第一步

新会话先只读核对本文件和以下事实：

1. 读取 `artifacts/finetune_mis20s1/evaluation/a4_metrics.json`。
2. 查看 `four_panel_1000x.png` 与 `four_panel_645x.png`。
3. 检查 `1000x` manifest、A1 reference 的 DN 统计和注册字段。
4. 写一个 CPU 诊断脚本，至少输出每通道的：raw packed DN、曝光映射 DN、归一化均值/分位数、reference DN、静态区域残差，以及注册重叠比例。
5. 先判断是预处理错误还是模型泛化错误，再决定是否修改训练。

推荐优先检查的文件：

- `src/raw_fusion/finetune/a3.py` 中 `_real_sample` 的曝光/黑电平/注册处理
- `src/raw_fusion/finetune/a4.py` 中 `_read_aligned_video` 和 motion mask 对齐
- `artifacts/finetune_mis20s1/supervision/stream_manifests/04_shdarkroom_1000x.json`
- `artifacts/finetune_mis20s1/supervision/references/a1_reference_report.json`

## 7. 后续决策树

### 若确认 `1000x` 预处理或注册有错

修正 A3/A4 共用的输入变换，增加对应单元测试，使用新的输出目录重跑 `1000x` A4。不能复用旧 `prediction_1000x.npy`，因为它来自旧管线。

### 若确认预处理正确

进入针对性 A5 训练：

- 将 `1000x` 真实窗口纳入训练，避免当前完全 holdout 造成域外失败。
- 保留另一个增益（例如 `645x`）作为留出验证，防止只记忆 1000x。
- 继续保留高频/梯度约束，并增加 flat residual 监控。
- 训练后重新跑 A4 全指标和四宫格；最终是否“过平滑已解决”仍需要 dad 进行视觉确认。

## 8. 可恢复评估命令

确认 4090 空闲或获得用户允许等待后，命令为：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
/home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
scripts/run_a4_eval.py \
  --max-frames 200 \
  --tile-batch 8 \
  --output-dir artifacts/finetune_mis20s1/evaluation
```

修改预处理后请使用新的输出目录，例如 `evaluation_1000x_fixed_v1`，避免混用旧缓存。

## 9. 交接边界

- 已确认：A1/A2/A3 产物、A3 step 5000 checkpoint、A4 两增益 200 帧结果、A4 输出文件和上述代码提交。
- 未确认：`1000x` 暗部失败的根因；完整回归测试是否全部通过；A5 是否应立即开始。
- 新会话可以读文件、做 CPU 诊断和修改项目代码；任何新的 GPU 长训仍须先检查 GPU，并遵守“不终止外部进程”的限制。
- 本文档不授权删除旧输出、覆盖用户未提交文件或终止外部训练。
