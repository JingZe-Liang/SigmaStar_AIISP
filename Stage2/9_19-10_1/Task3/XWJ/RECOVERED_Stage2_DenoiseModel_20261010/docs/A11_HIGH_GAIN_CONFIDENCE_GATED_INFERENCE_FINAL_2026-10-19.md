# A11 高增益运动/错位置信门控双分支推理最终报告（2026-10-19）

## 1. 最终裁决

**A11a 明确否决。** 本轮结果不能登记为“可选高增益后处理”，不能进入 A11b，也不能替换 A3。

完整置信规则在两条代表序列上均保持了 legal RAW、静态高置信区域和门控边界连续性，也没有把纯 frame-repeat 的大面积彩噪带回输出；但它只覆盖约 `0.49%/0.71%` 的 packed RAW 像素，主要命中人物四肢或少数移动边缘，没有覆盖造成主体朦胧和重影的大部分区域。固定 ISP 逐帧检查中，A11 与 A3 weighted 的差异极小，不能证明两条序列均有可重复、可见的运动边缘或重影改善。

因此，A11 未满足预注册通过条件中的第一项和第七项。不能用“局部 temporal ratio 提升”替代视觉改善，也不能通过继续降低阈值或扩大 mask 来把本轮变成新的标量扫描。

## 2. 不变量与只读核对

- 项目：`/Data4/Xiao/RECOVERED_Stage2_DenoiseModel_20261010`
- 交接编号：`STAGE2-HIGH-GAIN-GATE-READY-20261019`
- 上游交接编号：`STAGE2-RECOVERY-READY-20261011`
- A3 checkpoint：`artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`
- A3 SHA256：`97c4332f685f98959b243e4b100434469f714aed6329e100d22b6d8552592e0b`
- Git HEAD：`3971228ce6a2fe643cc5c1ec378c73d782199e32`
- 恢复 tree：`a16ffa4f5a66e20a33ca3c7e9ec0f7c66a9496ee`
- `.git/config` 无 remote；恢复期 modified/untracked 工作树保持不变，没有 commit 或 push。
- `RECOVERY_REBUILD_20261010.md` 与交接记录一致；现有 A3 weighted、frame-repeat、uniform baseline 和四路 fixed-ISP 视频均完整。
- 两条源 RAW 仅以只读方式读取；新产物只写项目 `artifacts/` 和 `docs/`。
- 未启动训练、未加载模型、未使用旧教师、未使用动态 temporal mean、未接触 sc450ai 1435x hard holdout。
- 本轮后处理和 ISP 渲染均为 CPU-only。核对时 4090 和 3090 均被外部进程高负载占用，未终止或干预任何进程。

## 3. 实现与冻结协议

新增文件：

- `src/raw_fusion/finetune/a11_confidence_gate.py`
- `scripts/run_a11_high_gain_confidence_gate.py`
- `configs/a11_high_gain_confidence_gate_20261019.json`
- `src/tests/test_finetune_a11_confidence_gate.py`

最终实现指纹：

- `src/raw_fusion/finetune/a11_confidence_gate.py`：`0fdee49c4fcca5b90ab360beeb05a1c4b30397363193969023be227c9d3c28be`
- `scripts/run_a11_high_gain_confidence_gate.py`：`24eb17986477554759b14616e50ab9b37800bd09587a886e50bfdb58c1ac2629`
- `configs/a11_high_gain_confidence_gate_20261019.json`：`61002688d15c88ada0f4927d267bcb071ae5a4361e6c1b38fe083806a4c6a046`

产物根目录：

`artifacts/a11_high_gain_confidence_gate_20261019`

实现只依赖 noisy input 证据：

1. 对 Bayer 绿通道亮度做轻度低通，在原始高频局部噪声标尺下计算鲁棒时域残差。
2. 同时计算梯度幅值和方向不一致。
3. 使用前后 `gap=2` 的双向 DIS flow、forward-backward error 和 warp photometric error，聚合可靠运动与遮挡/错位证据。
4. 对 seed 做小连通域过滤、3 像素椭圆膨胀和 Gaussian feather。
5. 高置信静态 core 强制权重为零。
6. 完整配置最大权重为 `0.65`，按下式直接在 normalized RAW 域凸组合：

```text
A11 = (1 - w) * A3_weighted + w * A3_frame_repeat
```

没有进行 alpha/clip/阈值网格扫描。三种结构消融仅用于验证证据链：`residual_only`、`residual_gradient` 和 `full`；最终候选固定为 `full`。

初始预检发现未平滑高增益像素残差会被局部噪声标尺压成稀疏孤点，全部被连通域过滤。最终实现只做了一次结构性修正：在低通亮度上计算结构残差、梯度和匹配光度误差，同时保留原始高频估计作为噪声标尺。配置阈值和最大融合权重没有因此扫描或逐序列调整。

## 4. 开发序列与 fixed-ISP

统一评估 frames `0–7`：

1. `04_mis20s1_outdoor_1500x`
2. `04_shdarkroom_1000x`

五路 RAW 分支：Input、A3 uniform、A3 weighted、A3 frame-repeat、A11 gated。

ISP 与 2026-10-18 高增益诊断一致：

- highlight recovery：启用
- tone map：Reinhard
- Group03 2DNR/3DNR：禁用
- 每条序列固定曝光、黑电平、WB 和 CFA，五路完全共享
- 视频：1920x1080、HEVC、`yuv444p`、6 fps、8 帧

2x3 画面包含五路 fixed-ISP 结果和第六格 A11 权重热图，分别输出 overview 和中心 detail；另保存每一帧的无损 PNG 供逐帧检查。

## 5. RAW 与结构指标

下表均以 Input 为 `1.0`；`seam amp` 是原 `64/56` 网格边界相对 Input 的放大倍数。

| 序列 | 分支 | 梯度保持 | 低频结构 | 高频残留 | temporal MAD | seam amp | legal RAW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| outdoor 1500x | A3 weighted | 0.3810 | 0.7917 | 0.7926 | 0.0779 | 0.9236 | 1.0000 |
| outdoor 1500x | frame-repeat | 0.7579 | 0.9515 | 0.9139 | 0.5513 | 0.9831 | 1.0000 |
| outdoor 1500x | A11 full | 0.3820 | 0.7919 | 0.7927 | 0.0787 | 0.9239 | 1.0000 |
| shdarkroom 1000x | A3 weighted | 0.3788 | 0.8254 | 0.8022 | 0.0722 | 1.0655 | 1.0000 |
| shdarkroom 1000x | frame-repeat | 0.8551 | 0.9732 | 0.8943 | 0.6936 | 1.0008 | 1.0000 |
| shdarkroom 1000x | A11 full | 0.3800 | 0.8256 | 0.8024 | 0.0732 | 1.0638 | 1.0000 |

A11 的全局结构指标几乎等于 A3 weighted。这证明噪声和静态背景保护良好，但也说明门控覆盖不足以改变主要朦胧失效模式。

## 6. Mask、静态污染与边界

| 序列 | 配置 | `w>0` | `w>=0.5` | weight mean | motion temporal/input | static temporal/input |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| outdoor | residual only | 0.4870% | 0.0000% | 0.000762 | 0.6215 | 0.0878 |
| outdoor | residual + gradient | 0.4527% | 0.0000% | 0.000889 | 0.6346 | 0.0878 |
| outdoor | full | 0.4869% | 0.0951% | 0.001271 | 0.6780 | 0.0878 |
| shdarkroom | residual only | 0.6760% | 0.0000% | 0.000969 | 0.6180 | 0.0642 |
| shdarkroom | residual + gradient | 0.5587% | 0.0000% | 0.001003 | 0.6263 | 0.0642 |
| shdarkroom | full | 0.7126% | 0.1325% | 0.001751 | 0.6850 | 0.0642 |

完整配置的独立输入证据区域：

| 序列 | motion core | high-confidence static core | static nonzero weight | static shift mean/P95/P99/max |
| --- | ---: | ---: | ---: | --- |
| outdoor | 0.1475% | 20.5559% | 0.0000% | 0 / 0 / 0 / 0 DN |
| shdarkroom | 0.1945% | 39.0139% | 0.0000% | 0 / 0 / 0 / 0 DN |

门控边界：

| 序列 | boundary fraction | A11/A3 boundary gradient | shift-gradient mean | shift-gradient P99 |
| --- | ---: | ---: | ---: | ---: |
| outdoor | 0.2249% | 1.0081 | 0.000366 | 0.002720 |
| shdarkroom | 0.3521% | 1.0046 | 0.000370 | 0.002905 |

边界指标与逐帧视频均未发现 halo、断裂、明显锐度台阶或 tile seam 回归。静态污染为零是由高置信静态 core 的输入证据强制保护得到；该 core 只覆盖全图的一部分，因此不能把该结果解释为所有静态像素都有外部 GT 级证明。

## 7. 固定 ISP 逐帧视觉结论

### outdoor 1500x

- frames 0–7 的 mask 持续命中人物腿、脚、手或少数局部边缘。
- 主体躯干、头部及大面积重影/朦胧区域多数未被覆盖。
- A11 在少数四肢像素上略向 frame-repeat 靠近，但 A3 weighted 与 A11 在 overview/detail 中没有稳定、明确的轮廓改善。
- 没有观察到大面积彩噪回流、闪烁、halo 或静态背景污染；这是因为覆盖率非常低，不构成通过依据。

### shdarkroom 1000x

- frames 0–7 的 mask 命中人物局部轮廓、风扇/小物体边缘等稀疏区域。
- 人物主体、标定板、钟表和复杂小物体的整体朦胧没有得到可信修复。
- A11 与 A3 weighted 的画面差异非常小，无法在每帧或连续视频中稳定辨认出重影减少。
- 同样没有明显噪声回流或门控边界伪影，但这不能补偿主要目标未改善。

完整 mask 的 motion temporal/input ratio 达到 `0.678/0.685`，明显高于 A3 全局 temporal ratio，但该值只统计约 `0.15%–0.19%` 的 motion core。它证明局部融合按公式生效，不证明主体运动质量通过。

## 8. 通过条件逐项裁决

| 条件 | 结果 | 说明 |
| --- | --- | --- |
| 两条序列均有可重复运动边缘/重影改善 | **失败** | 局部极小，主体与主要重影未改善 |
| 噪声明显低于纯 frame-repeat | 通过 | 高频残留接近 A3 weighted，未大面积回流 |
| 高置信静态区与 A3 基本一致 | 通过 | static nonzero 与 shift 均为零 |
| mask 边界无突变/halo/锐度台阶 | 通过 | 指标和逐帧检查均无异常 |
| legal RAW fraction 为 1.0 | 通过 | 两条序列均为 1.0 |
| 无旧教师纹理、伪细节或新闪烁 | 通过 | 未使用旧教师，未见新增闪烁 |
| 指标和 fixed-ISP 肉眼结论一致 | **未达到通过要求** | 指标表明全局变化近零，视觉也无可信改善 |

由于关键改善条件失败，A11a 必须整体否决。

## 9. 产物

机器可读指标：

- `artifacts/a11_high_gain_confidence_gate_20261019/04_mis20s1_outdoor_1500x/evaluation.json`
- `artifacts/a11_high_gain_confidence_gate_20261019/04_shdarkroom_1000x/evaluation.json`

每条序列的 `inference/` 包含：

- 三种 ablation 的连续 `weights_*.npy`
- 三种融合结果 `a11_*.npy`
- `evidence_masks.npz`
- 包含源预测哈希和约束声明的 `manifest.json`

Fixed-ISP 视频：

- `artifacts/a11_high_gain_confidence_gate_20261019/04_mis20s1_outdoor_1500x/traditional_isp_video/04_mis20s1_outdoor_1500x_a11_5way_overview.mp4`
- `artifacts/a11_high_gain_confidence_gate_20261019/04_mis20s1_outdoor_1500x/traditional_isp_video/04_mis20s1_outdoor_1500x_a11_5way_detail.mp4`
- `artifacts/a11_high_gain_confidence_gate_20261019/04_shdarkroom_1000x/traditional_isp_video/04_shdarkroom_1000x_a11_5way_overview.mp4`
- `artifacts/a11_high_gain_confidence_gate_20261019/04_shdarkroom_1000x/traditional_isp_video/04_shdarkroom_1000x_a11_5way_detail.mp4`

每条序列另有 8 张 overview、8 张 detail 无损 inspection PNG 和 frames 0–7 detail contact sheet。

## 10. 验证与后续边界

- 完整测试套件：`122 passed, 1 warning`。唯一 warning 是既有 PyTorch `torch.meshgrid` 后续需显式指定 `indexing` 的提示。
- 所有新增 JSON 可解析。
- 所有 NPY shape/dtype/range/finite 检查通过，A11 selected output 为 `(8,4,540,960)`、`float16`、legal RAW fraction `1.0`。
- 四段视频均已检查编码、分辨率、像素格式、帧率、帧数和完整解码。
- A3 checkpoint SHA256 最终复核仍为 `97c4332f685f98959b243e4b100434469f714aed6329e100d22b6d8552592e0b`。
- 未执行 Git commit/push，未覆盖或移动 A3 checkpoint。

A11b 不应启动。当前高增益动态问题仍未解决；下一步若考虑训练，必须另行提交高增益条件化微调方案并由用户明确授权，不能把旧教师或动态 temporal mean 当作干净真值。
