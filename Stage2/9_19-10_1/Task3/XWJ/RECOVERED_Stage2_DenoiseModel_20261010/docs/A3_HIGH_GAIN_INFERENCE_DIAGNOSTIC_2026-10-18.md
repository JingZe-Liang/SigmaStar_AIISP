# A3 高增益推理诊断与无训练优化结论（2026-10-18）

## 1. 最终裁决

本轮没有修改或替换 A3 step-5000 checkpoint，也没有启动训练。结论如下：

1. 高增益视频的主要问题不是传统 ISP、高光重建或 Reinhard，而是 A3 在超出主要训练增益范围后发生跨帧对齐/融合失败。运动物体被过度时域平均，形成朦胧、拖影和半透明残影。
2. `64x64 tile / stride 48 / Hann / floor 0.05` 可保留为可选的空间拼接改进。它小幅降低瓦片边界异常，且与原 A3 的归一化 RAW 平均绝对差仅为 `8.4e-5` 和 `1.08e-4`，但它不能修复高增益朦胧。
3. “单帧重复 8 次后取第 4 个时序输出”只能保留为诊断/保守分支，不能直接作为默认高增益输出。它显著减轻运动拖影并恢复结构，但残余彩噪和高频颗粒明显增加。
4. 当前没有可直接全局上线的高增益模式。下一步应使用同一 A3 checkpoint 构建“普通时域 A3 + 单帧保守 A3”的运动/错配置信度门控，不允许仅依据 motion/input ratio 决定通过；若门控结果仍不能兼顾运动清晰度和噪声，再进入高增益条件化微调。

## 2. 不变量与协议

- A3 checkpoint：`artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`
- SHA-256：`97c4332f685f98959b243e4b100434469f714aed6329e100d22b6d8552592e0b`
- 权重：EMA
- 数据集：只读；本轮未写入 `/HardDisk/jingzeliang/Data/Company_Dataset/dataset`
- 训练：未执行
- 旧教师：未作为目标或残差来源
- ISP：`ISP_traditional`，启用高光重建与 Reinhard，不启用 2DNR/3DNR
- 曝光：逐序列复用原 A3 全数据预览中冻结的曝光值，四个分支完全一致
- 样本：每个序列第一个连续 8 帧片段
- GPU：RTX 3090（PyTorch `cuda:1`）；4090 上的无关任务未被触碰

代表性高增益序列：

| 序列 | 场景 | 总增益标签 | 源帧 |
| --- | --- | ---: | ---: |
| `04_mis20s1_outdoor_1500x` | 室外人物运动 | 1500x | 0-7 |
| `04_shdarkroom_1000x` | 暗室综合测试景 | 1000x | 0-7 |

四路分支：

| 名称 | 时域输入 | 空间拼接 |
| --- | --- | --- |
| Input | 原始单帧 | 无 |
| A3 uniform | 原始连续 8 帧 | 64/56，均匀平均（既有基线） |
| A3 weighted | 原始连续 8 帧 | 64/48，Hann，floor 0.05 |
| A3 frame-repeat | 每个目标帧重复 8 次，取输出索引 4 | 64/48，Hann，floor 0.05 |

## 3. 量化结果

下表中的结构、高频和时序指标均以输入为 `1.0`。`seam amp` 是在原 64/56 网格处的边界/普通邻接差异相对输入的放大倍数；小于 `1.0` 表示没有放大输入自身在该位置的差异。

| 序列 | 分支 | 梯度保持 | 时序均值梯度 | 低频结构 | 高频残留 | temporal MAD | seam amp | RAW 合法率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| outdoor 1500x | A3 uniform | 0.3832 | 0.6111 | 0.7931 | 0.7927 | 0.0782 | 1.0147 | 1.0000 |
| outdoor 1500x | A3 weighted | 0.3810 | 0.6076 | 0.7917 | 0.7926 | 0.0779 | 0.9236 | 1.0000 |
| outdoor 1500x | A3 frame-repeat | 0.7579 | 0.8474 | 0.9515 | 0.9139 | 0.5513 | 0.9831 | 1.0000 |
| shdarkroom 1000x | A3 uniform | 0.3800 | 0.6303 | 0.8257 | 0.8038 | 0.0725 | 1.1415 | 1.0000 |
| shdarkroom 1000x | A3 weighted | 0.3788 | 0.6285 | 0.8254 | 0.8022 | 0.0722 | 1.0655 | 1.0000 |
| shdarkroom 1000x | A3 frame-repeat | 0.8551 | 0.9074 | 0.9732 | 0.8943 | 0.6936 | 1.0008 | 1.0000 |

解释：

- A3 uniform 与 A3 weighted 的结构和时序指标几乎相同，证明扩大重叠并加权不能解决朦胧主因。
- A3 weighted 在两个场景的原 64/56 网格 seam amp 分别从 `1.0147` 降到 `0.9236`、从 `1.1415` 降到 `1.0655`，说明它对空间拼接有小幅、稳定的改善。
- A3 uniform 的 temporal MAD 仅保留输入的 `7.2%-7.8%`，与运动姿态被平均、视频看起来像蒙雾一致。
- frame-repeat 将低频结构恢复到 `95.2%-97.3%`，并将 temporal MAD 恢复到 `55.1%-69.4%`，证明跨帧融合是清晰度损失的主要来源。
- frame-repeat 的高频残留升至输入的 `89.4%-91.4%`，且固定 ISP 面板可见明显彩噪，因此不能把结构提升直接解释为净细节提升，也不能直接上线。

完整机器可读指标：

- `artifacts/a3_high_gain_diagnostic_20261018/04_mis20s1_outdoor_1500x/evaluation.json`
- `artifacts/a3_high_gain_diagnostic_20261018/04_shdarkroom_1000x/evaluation.json`

## 4. 固定 ISP 视觉证据

### outdoor 1500x

- 全景视频：`artifacts/a3_high_gain_diagnostic_20261018/04_mis20s1_outdoor_1500x/traditional_isp_video/04_mis20s1_outdoor_1500x_high_gain_4way_overview.mp4`
- 中心细节视频：`artifacts/a3_high_gain_diagnostic_20261018/04_mis20s1_outdoor_1500x/traditional_isp_video/04_mis20s1_outdoor_1500x_high_gain_4way_detail.mp4`
- 首帧全景：`artifacts/a3_high_gain_diagnostic_20261018/04_mis20s1_outdoor_1500x/traditional_isp_video/04_mis20s1_outdoor_1500x_high_gain_4way_overview_frame0000.png`
- 首帧细节：`artifacts/a3_high_gain_diagnostic_20261018/04_mis20s1_outdoor_1500x/traditional_isp_video/04_mis20s1_outdoor_1500x_high_gain_4way_detail_frame0000.png`

视觉结论：原 A3 和加权 A3 均出现人体轮廓模糊、不同姿态叠加、树木区域蜡化和横向明暗带；frame-repeat 的人体姿态和树枝更接近当前帧，但暗部彩噪明显。

### shdarkroom 1000x

- 全景视频：`artifacts/a3_high_gain_diagnostic_20261018/04_shdarkroom_1000x/traditional_isp_video/04_shdarkroom_1000x_high_gain_4way_overview.mp4`
- 中心细节视频：`artifacts/a3_high_gain_diagnostic_20261018/04_shdarkroom_1000x/traditional_isp_video/04_shdarkroom_1000x_high_gain_4way_detail.mp4`
- 首帧全景：`artifacts/a3_high_gain_diagnostic_20261018/04_shdarkroom_1000x/traditional_isp_video/04_shdarkroom_1000x_high_gain_4way_overview_frame0000.png`
- 首帧细节：`artifacts/a3_high_gain_diagnostic_20261018/04_shdarkroom_1000x/traditional_isp_video/04_shdarkroom_1000x_high_gain_4way_detail_frame0000.png`

视觉结论：原 A3 和加权 A3 的标定板、钟表、人物和小物体边缘均明显变软；frame-repeat 恢复了文字块、棋盘格和轮廓，但墙面与暗部噪声过强。

所有视频均为 HEVC、1920x1080、`yuv444p`、6 fps、8 帧。视频清单及 SHA-256 位于各自的 `video_manifest.json`。

## 5. 下一步实施建议

下一阶段应先做无训练的置信度门控双分支，而不是直接微调或再次做旧教师标量扫描：

1. 同时生成普通 8 帧 A3 weighted 和逐帧 repeat A3 weighted。
2. 从输入连续帧构建运动/错配置信度，必须同时考虑亮度残差、梯度一致性、局部块匹配可靠性和邻域连通性。
3. 静态高置信区域保持普通时域 A3；运动或错配区域只在边界膨胀、羽化后引入 frame-repeat 分支。
4. 评价必须同时检查运动轮廓、静态一致性、低频结构、高频残留、temporal MAD、RAW 合法率和 fixed-ISP 视频；不能以 motion/input ratio 单项通过。
5. 若门控后运动区仍噪声过强，再启动高增益条件化微调。训练应加入增益/噪声条件、真实高增益噪声统计、动态/遮挡约束和静态锚定，并保持 A3 step-5000 为不可覆盖基线。

## 6. 可复现命令

```bash
/home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/run_a3_high_gain_diagnostic.py \
  --phase all --device cuda:1 \
  --sequence 04_mis20s1_outdoor_1500x

/home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/run_a3_high_gain_diagnostic.py \
  --phase all --device cuda:1 \
  --sequence 04_shdarkroom_1000x
```

代码与配置：

- `src/raw_fusion/finetune/high_gain_diagnostic.py`
- `scripts/run_a3_high_gain_diagnostic.py`
- `configs/a3_high_gain_diagnostic_20261018.json`
- `src/tests/test_finetune_high_gain_diagnostic.py`

验证：

- 新增与关联测试：`9 passed`
- Python 语法检查：通过
- `git diff --check`：通过
- 两个 overview 和两个 detail 视频：均已由 `ffprobe` 核对为 1920x1080、HEVC、`yuv444p`、6 fps、8 帧
- A3 checkpoint SHA-256：复核一致

## 7. 未完成项与风险

- 尚未实现运动/错配置信度门控双分支，这是下一项明确工作。
- 本轮只评估两个代表性高增益序列的首个 8 帧片段，没有宣称覆盖全部高增益流。
- 没有干净动态真值，因此视觉清晰度、结构代理和噪声代理不能替代有监督质量判定。
- `64/48 Hann` 仅登记为可选推理拼接策略，尚未替换全数据预览中的 `64/56 uniform` 默认协议。
- frame-repeat 计算量约为普通单个 8 帧片段的 8 倍，直接全分辨率逐帧使用不适合当前部署目标。
