# A9 多传感器统一泛化实验最终结论

日期：2026-10-16（Asia/Shanghai）

## 最终裁决

**否决 A9 step-100 作为统一多传感器泛化 checkpoint，不继续相同方案到 500 step。**

SC465SL 与 sc635hai 的自动门槛通过，MIS20S1 未出现实质遗忘，但 sc450ai 在完整保留的
`1435x_motion_black_object` 序列上未恢复 A3 缺失的高频细节：空间高频保持率从 A3 的
`0.12662124` 降至 A9 的 `0.12658907`。固定 ISP 视频同时确认黑衣主体和运动轮廓仍然
中空、过平滑，A9 与 A3 肉眼几乎无差异。因此该候选不具备统一部署资格，只保留为诊断
产物。

A3 step-5000 继续作为唯一基线，未被覆盖或替换：

`artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`

SHA-256：`97c4332f685f98959b243e4b100434469f714aed6329e100d22b6d8552592e0b`

A9 step-100 诊断 checkpoint：

`artifacts/a9_multi_sensor_generalization/training_smoke_4090_retry2_20261016/checkpoint_step_000100.pth`

SHA-256：`e30da4169a782679dde2c72013bdba360d24ac80ca0ba2526f9422959ef9e28e`

机器可读最终裁决：

`artifacts/a9_multi_sensor_generalization/evaluation_step100_20261016/summary.json`

## 实验边界

- 单一 checkpoint，从 A3 step-5000 初始化；
- 仅训练 `conv_first`、`norm`、`conv_after_body`、`conv_last`，共 8 个张量、15,508 个参数；
- 100 step，batch size 1，64×64 patch，BF16，学习率 `5e-6`，EMA `0.999`；
- 采样比例：MIS20S1 回放 40%，SC465SL 静态奇偶分半 20%，sc635hai 静态奇偶分半 20%，
  sc450ai 已验证噪声合成 20%；
- 未使用旧教师输出、动态时域均值或 sc450ai 黑衣运动序列作为干净训练真值；
- 数据集目录保持只读，没有重新采集数据；
- MIS20S1 1000x、SC465SL 2000x、sc635hai 16x、sc450ai 1435x 均未进入对应训练监督。

sc450ai 噪声模型只使用 `804x_lightchanging` 的 0-119 帧拟合统计量，120-149 帧用于独立
真实性验证。验证共 146 个观测点，聚合 sigma 相对误差中位数 `4.9821%`，P90
`16.1700%`，四个 Bayer 通道全部通过。该验证只证明噪声幅度统计可复现，不证明缺失的
空间细节能够由合成监督恢复。

## 训练与 checkpoint 审计

实际 100 step 样本计数为：MIS20S1 39、SC465SL 19、sc635hai 24、sc450ai 合成噪声
18。训练日志全部有限，最大梯度范数 `0.61051`。

checkpoint 审计确认 live 和 EMA 均恰好只有授权的 8 个张量发生变化，未授权变化数均为
0。EMA 相对 A3 的最大绝对权重变化仅 `1.5920e-05`，说明该实验是保守微调，也解释了
固定 ISP 画面中 A9 与 A3 的差异非常小。

GPU 执行记录：第一次使用数字编号时因 CUDA 枚举顺序实际落到 RTX 3090，在 step 1 前
OOM；第二次已使用 4090 UUID，但当时有其他进程占用约 14.75 GiB，同样在 step 1 前 OOM。
这两个目录均无 checkpoint。显存释放后，使用物理 RTX 4090 UUID
`GPU-9ee78cda-5391-4e9e-15d2-eeca142f4d7b` 完成 100 step，未再次 OOM。

## 定量结果

静态奇偶分半参考结果如下。`Δ` 为 A9 减 A3；PSNR 越高越好，高频保持率越接近 1 越好，
temporal MAD、偏置和裁剪率需结合参考与视觉结果解释。

| 传感器与集合 | 指标 | A3 | A9 | Δ |
|---|---:|---:|---:|---:|
| SC465SL 2000x 留出 | PSNR / dB | 56.60490 | 56.68683 | +0.08192 |
|  | 高频保持率 | 0.9112685 | 0.9111807 | -0.0000878 |
|  | temporal MAD / DN | 1.306238 | 1.305175 | -0.001063 |
|  | 最大绝对偏置 / DN | 3.703262 | 3.612493 | -0.090769 |
|  | 最大归一化裁剪率 | 0.0008331 | 0.0006945 | -0.0001386 |
| sc635hai 131x 静态尾段 | PSNR / dB | 52.35436 | 52.36445 | +0.01009 |
|  | 高频保持率 | 0.9199258 | 0.9198020 | -0.0001238 |
|  | temporal MAD / DN | 2.162601 | 2.161312 | -0.001289 |
|  | 最大绝对偏置 / DN | 0.962602 | 0.898940 | -0.063662 |
|  | 最大归一化裁剪率 | 0.0074280 | 0.0069591 | -0.0004689 |

无干净动态真值的留出序列只作为诊断，不把 motion/input ratio 单独作为通过依据：

| 留出序列 | 指标 | A3 | A9 | Δ |
|---|---:|---:|---:|---:|
| sc635hai 16x | motion/input ratio | 0.5996022 | 0.5994923 | -0.0001099 |
|  | static/input ratio | 0.1013293 | 0.1013622 | +0.0000329 |
|  | 高频保持率 | 0.7636051 | 0.7635894 | -0.0000157 |
|  | temporal MAD / DN | 14.674537 | 14.673216 | -0.001321 |
|  | 归一化裁剪率 | 0.0219941 | 0.0211083 | -0.0008859 |
| sc450ai 1435x | motion/input ratio | 0.0568871 | 0.0568476 | -0.0000395 |
|  | static/input ratio | 0.0879080 | 0.0882128 | +0.0003048 |
|  | 高频保持率 | 0.1266212 | 0.1265891 | **-0.0000322** |
|  | temporal MAD / DN | 1.346608 | 1.348032 | +0.001424 |
|  | 归一化裁剪率 | 0.0327983 | 0.0300637 | -0.0027346 |

两类动态留出结果的 legal RAW fraction 均为 `1.0`。sc450ai 虽然裁剪率和平均偏置略有
改善，但高频恢复硬门槛失败，且视觉上没有恢复黑衣主体细节，所以不能据此判为通过。

MIS20S1 回放保护结果：PSNR `-0.000346 dB`，高频保持率 `+0.001016`，temporal MAD
`+0.150857 DN`。原域没有实质性质量回退，但这也不能弥补 sc450ai 的失败。

## 固定 ISP 视觉证据

四段视频均为 1920×964、6 fps、HEVC，布局为 `Input | A3 | A9 step-100`：

| 传感器 | 帧数 | 视频 | SHA-256 | 视觉结论 |
|---|---:|---|---|---|
| MIS20S1 | 8 | `artifacts/a9_multi_sensor_generalization/evaluation_step100_20261016/mis20s1/fixed_isp_video/mis20s1_1000x_input_a3_a9_fixed_isp.mp4` | `79b2cea27b42cc28aea91a5a1847f3d68b04fdd8e65e8c173600053c3481a2a2` | A9 与 A3 基本不可区分，无新增伪影 |
| SC465SL | 40 | `artifacts/a9_multi_sensor_generalization/evaluation_step100_20261016/sc465sl/fixed_isp_video/sc465sl_2000x_input_a3_a9_fixed_isp.mp4` | `7d47d7f3a6c5e2766a682b7547a45ea5c9bc78eb19c634adec9fd0591597ff9c` | 无新增闪烁、断边或高光污染 |
| sc635hai | 24 | `artifacts/a9_multi_sensor_generalization/evaluation_step100_20261016/sc635hai/fixed_isp_video/sc635hai_16x_input_a3_a9_fixed_isp.mp4` | `7a76f2b9ba9655bdd0ba916713012cdbfd90c427fbdee0359e893ed6bf65efce` | 运动人物与静态细节保持 A3 外观，无新增重影 |
| sc450ai | 24 | `artifacts/a9_multi_sensor_generalization/evaluation_step100_20261016/sc450ai/fixed_isp_video/sc450ai_1435x_input_a3_a9_fixed_isp.mp4` | `f50ac38e59913a50edd5325286839ca7d4ec3a3e746e3f3cd9f00bbf1202b1d9` | 黑衣和运动轮廓仍中空、过平滑，缺失细节未恢复 |

已复核每段首帧及中后段：MIS20S1 0.0/1.0 秒、SC465SL 0.0/5.8 秒、sc635hai
0.0/2.0 秒、sc450ai 0.0/3.5 秒。完整人工记录位于
`artifacts/a9_multi_sensor_generalization/evaluation_step100_20261016/visual_review.json`。

## 失败原因与下一步边界

当前 sc450ai 分支只学习了经过真实性门槛验证的信号相关噪声幅度。它没有获得可证明的
sc450ai 干净空间细节监督，也没有建模该传感器的空间相关噪声、行列固定模式、PRNU/DSNU、
黑电平漂移、光学模糊与 CFA 响应差异。因此继续相同数据与损失到更多 step，最多可能改变
去噪强度和裁剪行为，不能合理期待凭空恢复当前监督中不存在的黑衣纹理与运动边缘。

下一阶段应先设计新的监督/退化协议，再讨论训练：优先审计现有 sc450ai 数据中是否存在
可用的静态配对、平场、暗场或不同增益序列；若只能合成，则需要把空间相关项、固定模式、
响应非均匀性、模糊/CFA 和黑电平漂移加入退化模型，并继续保留 `1435x_motion_black_object`
作为完整硬留出。没有新的、可证明能提供空间细节约束的监督之前，不批准延长本 A9 方案。

## 复现入口

以下命令均从项目根目录执行，Python 为
`/home/jingzeliang/miniconda3/envs/aaa_312/bin/python`：

```bash
PY=/home/jingzeliang/miniconda3/envs/aaa_312/bin/python
$PY scripts/build_a9_sensor_noise_profile.py
$PY scripts/audit_a9_training_inputs.py
CUDA_VISIBLE_DEVICES=GPU-9ee78cda-5391-4e9e-15d2-eeca142f4d7b \
  $PY scripts/run_a9_generalization.py \
  --output-dir artifacts/a9_multi_sensor_generalization/training_smoke_4090_retry2_20261016 \
  --device cuda:0
$PY scripts/audit_a9_checkpoint.py \
  --candidate-checkpoint artifacts/a9_multi_sensor_generalization/training_smoke_4090_retry2_20261016/checkpoint_step_000100.pth \
  --training-log artifacts/a9_multi_sensor_generalization/training_smoke_4090_retry2_20261016/train.jsonl \
  --output artifacts/a9_multi_sensor_generalization/training_smoke_4090_retry2_20261016/checkpoint_audit.json
$PY scripts/summarize_a9_generalization.py
```

评估入口分别为 `scripts/evaluate_a9_sc465sl.py`、`scripts/evaluate_a8_sc635hai.py`、
`scripts/evaluate_a8_sc450ai.py` 和 `scripts/render_a9_mis20s1_video.py`。后两个脚本是复用的
A8 评估实现，现已支持显式 A9 协议标签；数值、视频和最终语义由 A9 `summary.json` 统一核对。

## 结构与回归检查

- 相关 A8/A9/跨传感器 pytest：`37 passed`，仅有一个既有 `torch.meshgrid` 弃用预警；
- A9 相关 Python 文件 `py_compile`：通过；
- A9 目录 21 个 JSON：全部可解析；
- 四段视频容器：分辨率、帧率、帧数和 HEVC 编码均符合清单；
- `git diff --check`：通过；
- 冲突标记扫描：未发现冲突标记；
- 汇总脚本重新计算并核对 A3、A9 checkpoint 及四段视频 SHA-256：通过。

## 当前未完成项

A9 当前实验的训练、评估、视频导出、视觉复核、结构检查和最终裁决均已完成。整个项目的
最终目标“同一去噪模型在不同传感器上稳定泛化”仍未完成，明确阻塞点是 sc450ai 空间细节
恢复。后续工作不是继续本 checkpoint 的 step 数，而是建立新的 sc450ai 空间监督或更完整的
传感器退化模型，并在相同硬留出上重新验证。
