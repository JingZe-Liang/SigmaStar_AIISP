# Stage2 DenoiseModel Handoff: Native 1000x Evaluation Ready

更新时间：2026-09-22 22:29 (Asia/Shanghai)

交接编号：`A5-NATIVE-20260922-01`

源交接编号：`A4-20260922-01`

交接状态：材料已核对，等待新会话只读核验。本文档不代替用户的新指令或授权。

## 1. 项目与当前目标

- 项目根目录：`/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel`
- 总体目标：为 MIS20S1 RAW 高增益低照度场景训练一个抑制噪声、保留真实高频细节的时空 RAW 去噪模型。
- 当前用户已确认的路线：`1000x` 使用同增益原生 temporal reference 做独立评估，`645x` 保留现有 10x reference 的 cross-gain 留出验证。
- 重要边界：不要再把 `1000x -> 10x` 的 PSNR 直接当作模型泛化门槛；该比较受跨捕获重新打灯和动态范围差异污染。

## 2. 权威资料与环境

- 计划文档：`docs/RVIDEFORMER_FINETUNE_PLAN_2026-09-20.md`
- 上一份 A4 交接：`docs/HANDOFF_2026-09-22_A4.md`
- 数据集（只读）：`/HardDisk/jingzeliang/Data/Company_Dataset/dataset`
- 官方 RViDeformer（只读）：`/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/RViDeformer`
- Python：`/home/jingzeliang/miniconda3/envs/aaa_312/bin/python`
- 源码命令需要：`PYTHONPATH=src`
- A1 主 reference：`artifacts/finetune_mis20s1/supervision/references/04_shdarkroom_10x_clean_reference.npy`
- A3 checkpoint：`artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`

## 3. 已完成的历史里程碑

### A1/A2/A3

- A1 已生成并通过质量门：10x clean reference、valid/confidence、FPN/DSNU/PRNU。
- A2 已生成噪声模型和 100x 到 25600x synthetic pairs。
- A3 已完成 step 5000，checkpoint 含 model/EMA 各 1265 个参数键和 optimizer 状态。
- A3 原训练配置中 `1000x` 是 holdout；真实训练增益为 `128x/313x/632x/645x`，另混合 synthetic pairs。

### 原 A4 结果

原 A4 仍保留在 `artifacts/finetune_mis20s1/evaluation/`，没有被修改：

| 增益 | A3 PSNR | 输入 PSNR | 旧 teacher PSNR | A3 高频比 | A3 flat residual sigma |
|---|---:|---:|---:|---:|---:|
| `1000x` 对 10x reference | 18.84 dB | 18.73 dB | 18.78 dB | 0.339 | 226.19 DN |
| `645x` 对 10x reference | 29.57 dB | 24.35 dB | 28.02 dB | 0.971 | 83.19 DN |

原 A4 代码和测试：

- `src/raw_fusion/finetune/a4.py`
- `scripts/run_a4_eval.py`
- `src/tests/test_finetune_a4.py`
- `src/tests/test_finetune_a4.py`：2 个测试通过

## 4. 已完成的 1000x 诊断

### 4.1 输入、注册和曝光统计

脚本：`scripts/diagnose_a4_1000x.py`

结果：`artifacts/finetune_mis20s1/evaluation/a4_1000x_input_diagnosis.json`

- 200 帧 CPU 诊断已完成。
- `1000x` manifest 的浮点注册偏移 `(-61.94, -0.60)` 是 sensor 像素；除以 2 后 packed 偏移约 `(-30.97, -0.30)`，所以 `integer_shift=(-31, 0)` 正确。
- packed overlap ratio 为 `0.9677`；注册不是当前异常的根因。
- 静态区 residual sigma 约为 `R=218、Gr=391、Gb=390、B=228 DN`。

### 4.2 按 reference DN 分桶的 1000x 与 645x 对比

脚本：`scripts/diagnose_exposure_fit.py`

结果：`artifacts/finetune_mis20s1/evaluation/exposure_fit_diagnosis.json`

- `1000x` 高亮端严重欠映射：reference `1024–2048 DN` 时 residual 约 `-570 ~ -705 DN`；`2048–3072 DN` 时约 `-1.4k ~ -1.8k DN`；更高区间约 `-2.7k ~ -3.1k DN`。
- `645x` 在 `256–2048 DN` 的 residual 约为 `-70 ~ +45 DN`，明显稳定得多。

### 4.3 直接 `1000x -> 645x` 配对检查

脚本：`scripts/diagnose_1000x_vs_645x.py`

结果：`artifacts/finetune_mis20s1/evaluation/1000x_vs_645x_exposure_diagnosis.json`

- 200 帧 CPU 配对已完成，注册仍为 `(-31, 0)`，overlap `0.9677`。
- 低亮度 `256–512 DN` 区域 residual 约 `+3 ~ +10 DN`。
- `512–1024 DN` 区域 residual 约 `-211 ~ -247 DN`，更高区间几乎没有 1000x 对应样本。
- robust fit 的中位数 residual 约 `-6 ~ -7 DN`，但这是在低亮度可配对样本上得到的，不能证明高亮端有效。

### 4.4 1000x 原生 temporal reference 探针

探针目录：`artifacts/finetune_mis20s1/evaluation/1000x_self_reference_probe/`

包含：

- `temporal_mean_native_dn.npy`
- `even_mean_native_dn.npy`
- `odd_mean_native_dn.npy`
- `confidence.npy`
- `valid_region.npy`
- `report.json`

探针报告确认：

- 200 帧、motion-aware valid fraction：`0.9321`
- 逐帧曝光归一化尺度约为 `0.98–1.02`
- 全体有效静态信号区分半 residual sigma：`R=2.09、Gr=2.35、Gb=2.34、B=2.02 DN`
- 原生 temporal mean 足够稳定，可作为 1000x 同增益候选 reference。
- 10x A1 的固定图卡窗口硬校验不能直接复用于 1000x，因为该窗口在 1000x 信号不足；这不是 temporal mean 整体不可用的证据。

## 5. 当前已实现的新代码

### Native evaluator

- `src/raw_fusion/finetune/a5_native.py`
- `scripts/run_native_1000x_eval.py`
- `src/tests/test_finetune_a5_native.py`

行为约定：

- 只读取 `1000x` 原始 packed DN，不使用 1000x 到 10x 的 exposure fit。
- 归一化使用 1000x 自身 black level。
- reference 使用 `1000x_self_reference_probe/temporal_mean_native_dn.npy`。
- 指标输出 native-domain RMSE/PSNR、输入 baseline、temporal MAD、high-frequency ratio、flat residual sigma、legal fraction。
- 输出汇总会显式链接现有 `645x` A4 留出指标：`artifacts/finetune_mis20s1/evaluation/metrics_645x.json`。
- 现有 `a4.py` 和原 A4 产物未修改。

验证状态：

- `src/tests/test_finetune_a5_native.py`：2 个测试通过。
- `src/tests/test_finetune_a4.py`：2 个测试通过。
- A5 native + A4 测试合计：`4 passed`。
- 新脚本和模块已通过 `py_compile`。
- 尚未运行完整 native GPU 推理，因此 native A3 指标和 panel 尚未生成。

## 6. 当前结论

1. `1000x` 的 `shift_x=-31` 是正确的 packed-resolution 注册，不是注册 bug。
2. `1000x -> 10x` 线性曝光映射在高亮端失效，主要原因是各增益重新打灯/捕获动态范围不一致；不能把原 A4 的 1000x 低 PSNR 直接归因于 A3 模型泛化。
3. 1000x 原始流自身的 200 帧 temporal mean 稳定，适合做 native-domain 评估 reference。
4. 当前还没有证据支持立即修改 A3 训练，把真实 1000x 窗口直接配给 10x target；这样可能把照明差异灌入监督目标。
5. `645x` 仍是当前可用的真实 cross-gain 留出验证，原指标为 `29.57 dB`、高频比 `0.971`。

## 7. GPU 与运行状态

最近一次成功的 GPU 查询显示：

- 3090：约 21.7 GiB 已用，仅约 2.4 GiB 空闲。
- 4090：约 16.6 GiB 已用，约 7.5 GiB 空闲，利用率约 78%。

随后一次 `nvidia-smi` 查询出现 driver communication failure，因此新会话必须重新运行并确认 GPU 状态。不要终止外部训练进程，不要根据旧的 GPU index/UUID 映射盲目启动。磁盘空间当前约 289 GiB 可用。

## 8. 新会话的第一步

新会话只读执行以下核验，不要先修改代码或启动 GPU：

1. 读取本文件，确认项目根目录、交接编号 `A5-NATIVE-20260922-01` 和用户已确认路线。
2. 检查以下文件存在且形状/内容可读：
   - `artifacts/finetune_mis20s1/evaluation/1000x_self_reference_probe/report.json`
   - `artifacts/finetune_mis20s1/evaluation/1000x_self_reference_probe/temporal_mean_native_dn.npy`
   - `src/raw_fusion/finetune/a5_native.py`
   - `scripts/run_native_1000x_eval.py`
   - `artifacts/finetune_mis20s1/evaluation/metrics_645x.json`
3. 运行 `nvidia-smi`，确认 RTX 4090 有足够显存且没有破坏性外部任务；如果仍被占用，只反馈并等待，不杀进程。
4. 在 4090 空闲后运行完整 native 评估：

```bash
CUDA_VISIBLE_DEVICES=<经 nvidia-smi 确认的 RTX 4090 索引> PYTHONPATH=src \
/home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
scripts/run_native_1000x_eval.py \
  --max-frames 200 \
  --tile-batch 8 \
  --output-dir artifacts/finetune_mis20s1/evaluation/native_1000x
```

5. 验收以下输出：
   - `native_1000x_metrics.json`
   - `native_metrics_1000x.json`
   - `native_prediction_1000x.npy`
   - `native_four_panel_1000x.png`
   - native `static_pixels > 0`、`legal_fraction` 合理，且比较 native A3 与 native input。
   - 同时回读原有 `metrics_645x.json`，不要覆盖或重解释原 A4 文件。

## 9. 后续决策树

### Native 1000x 评估优于 input

- 说明当前 A3 在 1000x 原生域有实际去噪收益。
- 先以 native 1000x 指标 + 645x cross-gain 指标作为双重验收，再决定是否做 A5 训练。
- 若需要训练改动，必须先向 dad 说明新的 target/domain 设计，再修改 A3 数据集。

### Native 1000x 评估不优于 input

- 先检查 native input/reference 的 black、motion mask、valid region 和 temporal reference 一致性。
- 不要直接把真实 1000x 窗口加入 10x-target 训练。
- 只有完成上述复核后，才讨论针对性 A5 训练。

## 10. 交接边界与限制

- 当前所有新增代码和诊断脚本尚未提交，git 工作区存在未跟踪文件；新会话必须保留这些文件，不要重置或删除。
- 不删除任何数据、旧 prediction、旧 metrics 或 probe 文件。
- 不修改数据集和官方 RViDeformer 目录。
- 修改代码前需向 dad 说明准备修改的文件和行为；启动 GPU 实验前必须先 `nvidia-smi`。
- 不把 native 1000x probe 宣称为已通过 A1 主 reference 质量门；它是同增益评估参考探针。
- 完整 `pytest src/tests` 仍未验证；当前只确认 A4/A5 相关测试通过。
