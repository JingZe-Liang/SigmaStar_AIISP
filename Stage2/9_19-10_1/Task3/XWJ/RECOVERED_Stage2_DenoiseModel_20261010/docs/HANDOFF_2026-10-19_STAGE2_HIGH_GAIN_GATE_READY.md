# Stage2 去噪模型跨传感器与高增益问题交接

## 0. 交接记录

- 交接编号：`STAGE2-HIGH-GAIN-GATE-READY-20261019`
- 上游交接编号：`STAGE2-RECOVERY-READY-20261011`
- 记录日期：`2026-10-19`
- 项目目录：`/Data4/Xiao/RECOVERED_Stage2_DenoiseModel_20261010`
- 数据集目录：`/HardDisk/jingzeliang/Data/Company_Dataset/dataset`
- Python：`/home/jingzeliang/miniconda3/envs/aaa_312/bin/python`
- 当前分支：`master`
- 本地 HEAD：`3971228ce6a2fe643cc5c1ec378c73d782199e32`
- 本地恢复树：`a16ffa4f5a66e20a33ca3c7e9ec0f7c66a9496ee`
- GitHub 部分快照：`9964f776dd7ebd015a61f4fd416989b3cf4aba85`
- 当前状态：`materials verified; waiting for receiving-session verification`
- 接收会话编号：留空，由新会话核验后填写
- 本交接允许的下一阶段：只允许执行本文第 12 节所定义的 A11 确定性、无训练后处理实验；不自动授权训练、提交、推送、删除或覆盖文件。

## 1. 一句话结论

项目已经得到一个可靠的 MIS20S1/A3 静态去噪基线，并系统排除了旧教师残差、冻结运动专家、简单标量融合、sc450ai 盲点适配和当前噪声退化生成器等失败路线；当前最明确的未解决问题是高增益动态场景中时域对齐/融合失效导致的朦胧、重影和过度平滑。下一步应保持 A3 checkpoint 不变，使用同一模型的“正常时域分支”和“单帧重复分支”进行输入证据驱动的置信门控融合，先在已有两组高增益诊断数据上完成 A11 无训练裁决。

## 2. 项目最终目标

最终目标不是只在单一 MIS20S1 低增益静态数据上取得高 PSNR，而是形成一个可部署、可复现的 RAW 视频去噪方案，使模型在多种公司传感器、增益档位、照度和运动条件下具有可信泛化能力：

1. 支持至少 MIS20S1、SC465SL、sc635hai、sc450ai 等现有传感器数据。
2. 在静态区域显著去噪，同时不破坏纹理、高频、亮度和颜色关系。
3. 在人体、物体和相机运动区域避免重影、拖影、空洞、边缘断裂和过度平滑。
4. 在高总增益下不出现明显朦胧感，也不因恢复细节而重新引入不可接受的彩噪和颗粒。
5. 输出保持 legal RAW，经过统一传统 ISP 后可进行稳定、可重复的肉眼比较。
6. 每个结论同时依赖定量指标和固定 ISP 视觉证据，不能只依赖 motion/input ratio、PSNR 或单个局部样本。
7. A3 step-5000 始终保留为基线；任何传感器专用模型或后处理只能作为明确标记的可选方案，除非以后经过完整回归验证再另行决策。

当前尚未达到“所有传感器和高增益动态条件均可靠”的最终目标。当前最稳妥的默认模型仍是 A3；sc635hai 有一个条件保留的专用 checkpoint；sc450ai 和高增益动态场景仍未解决。

## 3. 不可违反的约束

1. 不能重新拍摄，只能使用现有数据集、恢复树和现有产物。
2. 数据集目录只读，不得修改、重排、生成旁路文件或写缓存：
   `/HardDisk/jingzeliang/Data/Company_Dataset/dataset`
3. 不覆盖、不移动、不替换 A3 step-5000 checkpoint。
4. 不执行破坏性 Git 操作，不清理恢复树，不回退用户现有修改。
5. 不把旧教师输出当作干净动态真值。
6. 不把动态场景的时域均值当作干净动态真值。
7. 不以 motion/input ratio 单项提升作为通过依据。
8. 不重新进行已经否决的 A5/A6/A7 简单标量扫描。
9. sc450ai 1435x 黑色运动物体序列是 hard holdout，只能评估，不能训练、拟合阈值或选择参数。
10. 未向用户先说明训练数据、监督来源、损失、参数更新范围和验收标准前，不启动任何新训练。
11. 本交接推荐的 A11 阶段是确定性、无训练的推理后处理；即使通过，也只能登记为可选高增益后处理，不能替换 A3 checkpoint。
12. 若需要 GPU，优先考虑 4090，但执行前必须重新检查占用；若可能 OOM，立即停止并报告，不得终止其他用户进程。

## 4. 权威基线与恢复状态

### 4.1 A3 基线

- checkpoint：
  `artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`
- SHA256：
  `97c4332f685f98959b243e4b100434469f714aed6329e100d22b6d8552592e0b`
- 接受报告：
  `docs/A3_BASELINE_ACCEPTANCE_2026-09-23.md`
- 微调总结：
  `docs/STaR-RAW_FINETUNE_SUMMARY_2026-09-24.md`

A3 来源是 CRVD 预训练 RViDeformer 在 MIS20S1 公司数据上的谨慎微调。公司真实数据量有限，因此采用了静态时域均值参考、实测噪声模型、真实/合成混合数据，并排除运动和饱和区域的静态监督。A3 在 MIS20S1 静态域显著优于官方预训练模型：native 1000x 输入 `44.6204 dB`，A3 为 `53.2675 dB`；RMSE 从 `24.0565 DN` 降到 `8.8895 DN`。645x 跨增益 holdout 从 `24.3519 dB` 提升到 `29.5729 dB`。这证明 A3 是当前最可靠的静态基线，但不等于它已经解决动态和跨传感器问题。

### 4.2 磁盘恢复与 GitHub 快照

- 恢复说明：`RECOVERY_REBUILD_20261010.md`
- 上游恢复交接：`docs/HANDOFF_2026-10-11_STAGE2_RECOVERY_READY.md`
- GitHub 超大文件排除表：`GITHUB_UPLOAD_EXCLUDED_LARGE_FILES_20261011.tsv`
- 本地恢复标记：`.recovered_commit`

恢复树可以运行 A3 推理、native 1000x 验证和 A5/A6/A7 检查。GitHub 上仅保存了部分快照，53 个超过 100 MiB 的文件未上传，总计约 `20.842 GiB`，因此本地恢复树仍是运行时权威来源。当前 `.git/config` 没有 remote，且存在一个历史 Git tree object 缺失问题。不要把 GitHub 部分快照误认为完整运行树，也不要尝试破坏性修复。

当前工作树存在大量恢复期的 modified/untracked 文件，这是预期状态。A8、A9、A10、A10b、全数据预览和高增益诊断等后续工作是否已经同步到 GitHub，本会话没有得到可靠证明，必须登记为未核实，不能声称已上传。

## 5. 完整研究与决策历程

### 5.1 从官方模型到 A3：先解决静态去噪

最初直接使用 CRVD/RViDeformer 官方模型时，公司 RAW 上出现明显过度平滑。根本原因是传感器噪声、黑电平、增益和场景分布与预训练域不同。由于无法获得大量成对干净 RAW，项目选择在 MIS20S1 上构造可信的静态监督：对齐静态帧、构造时域均值参考、排除运动和饱和像素、拟合实际噪声模型，并混合真实与合成样本进行小范围微调。

这条路线得到 A3 step-5000。A3 的成功说明：只要监督目标在静态区域可信，RViDeformer 可以适应公司传感器并取得强静态去噪。但这也埋下后续问题：被排除的运动区域没有得到干净动态细节监督，合成数据又主要重复同一干净图像，模型更擅长消除时域变化，而不是区分真实运动与噪声。

### 5.2 A5：确认运动细节损失不是单纯阈值问题

权威设计报告：`docs/A5_MOTION_SUPERVISION_DESIGN_2026-09-25.md`

审计发现，A3 在运动区保留的帧差通常只有输入的约 `27%–32%`，个别转场约 `0.10`。这说明 A3 的强时域融合把真实运动也当成了噪声。随后构建严格 flow Noise2Noise 探针和 8 帧动态窗口，尝试动态监督、教师锚定、不同冻结范围和损失权重。

结论是明确的：增加动态损失可以让运动指标改善，但会导致静态 PSNR、高频保持或 native 表现回退；加强锚定又会把运动增益压回去。问题不是缺少一个更好的标量，而是动态监督本身缺乏可信干净目标。A5 的简单标量/冻结路线被关闭，不应重复扫描。

### 5.3 A6：冻结 A3 加运动残差专家仍无法恢复真细节

权威报告：`docs/A6_FROZEN_MOTION_EXPERT_RESULT_2026-10-09.md`

A6 保留 A3，增加受门控的残差运动专家，希望只修改运动区域。v1 到 v4 均被否决。实验表明门控可以定位运动，但残差专家没有足够监督去判断“应该恢复什么干净动态结构”。旧教师携带的运动纹理和噪声与目标冲突，合成局部平移又过于狭窄。最终没有可推广的 A6 checkpoint。

### 5.4 A7：已有数据上的旧教师局部融合全部否决

权威报告：`docs/A7_EXISTING_DATA_FUSION_FINAL_2026-10-11.md`

A7 对 r3/r7/r11 区域扩张、多个 alpha 和 residual clip 共 36 个配置进行了确定性评估。最强的 r11/alpha0.4/clip96 相对 A3 的 motion/input 仅提升约 `0.9162%`，固定 ISP 下没有可信的人体或边缘清晰度提升，反而把旧教师纹理和噪声带回输出。全部配置被否决，没有保留任何 A7 可选后处理。

这一步确认了一个关键原则：旧教师不是干净动态真值；扩大融合区域或增加残差强度不能产生不存在的干净信息。后续不得原样重复旧教师残差融合。

### 5.5 跨传感器零样本审计：A3 的支持范围并不一致

权威报告：`docs/A3_CROSS_SENSOR_GENERALIZATION_AUDIT_2026-10-12.md`

- SC465SL：在已评估的 256x/2000x 静态域中支持。
- sc635hai：低色温静态域支持；动态运动和颜色表现为条件支持。
- sc450ai：不支持。存在高频塌陷、黑色运动物体发空/变软、伪影和裁剪问题。
- difficultpattern：只作为条件压力测试，不能据此宣称传感器全面支持。

这一结果推动项目从“单一 MIS20S1 微调”转向“验证各传感器监督是否足够，再决定是否适配”。

### 5.6 A8：逐传感器适配只有 sc635hai 条件保留

总报告：`docs/A8_MULTI_SENSOR_ADAPTATION_FINAL_2026-10-14.md`

sc635hai 报告：`docs/A8_SC635HAI_ADAPTATION_RESULT_2026-10-13.md`

- 条件保留 checkpoint：
  `artifacts/a8_sensor_adaptation/sc635hai/training_20261013/checkpoint_step_000500.pth`
- SHA256：
  `8e4980a43767b5b83ecc064e8cdce2f685ea452e94af48a1f4ca252004765b3f`
- 只更新输出头的 6 个 tensor，共 12,400 个参数。
- 只能作为 sc635hai 专用可选模型，不能替换 A3。

sc450ai 报告：`docs/A8_SC450AI_ADAPTATION_RESULT_2026-10-14.md`

sc450ai 的 blind-spot step100–500 全部否决。高频塌陷没有修复，裁剪和偏差反而加重。该结果说明仅依赖 noisy-only blind-spot 目标无法补回被模型抹除的空间结构。

### 5.7 A9：统一多传感器小步微调未解决最难传感器

权威报告：`docs/A9_MULTI_SENSOR_GENERALIZATION_RESULT_2026-10-16.md`

- step100 checkpoint：
  `artifacts/a9_multi_sensor_generalization/training_smoke_4090_retry2_20261016/checkpoint_step_000100.pth`
- SHA256：
  `e30da4169a782679dde2c72013bdba360d24ac80ca0ba2526f9422959ef9e28e`

A9 在 SC465SL/sc635hai 的门槛上通过，MIS20S1 没有明显遗忘，但 sc450ai 1435x 黑色运动物体高频没有改善：A3 为 `0.126621`，A9 为 `0.126589`，视频肉眼也无改善。因此 A9 step100 被否决，不继续同方案到 500 step。

这一步进一步确认：统一噪声幅度建模只能适应部分噪声统计，不能自动恢复传感器特有的模糊、CFA 响应、固定图样噪声或缺失的干净空间细节。

### 5.8 A10/A10b：sc450ai 当前数据不具备训练级监督

监督审计：`docs/A10_SC450AI_SUPERVISION_AUDIT_2026-10-16.md`

sc450ai 只有两条流，没有 dark/flat、跨增益配对或干净 GT。现有数据最多支持噪声和退化统计，不能支持可信监督微调，也不应重复 blind-spot 训练。1435x 全序列被固定为 hard holdout。

退化生成器验证：`docs/A10B_SC450AI_DEGRADATION_VALIDATION_2026-10-16.md`

A10b 验证了现有噪声退化生成器。shared spectral Q4 在 804x 上改善了总体 ACF，但没有通过预注册阈值，并且在 1435x hard holdout 上失败。最终状态为 `rejected_all_generators`，`selected_generator = null`。因此当前不能以“已经有可信 sc450ai 退化生成器”为前提启动训练。

### 5.9 全数据集 A3 + 传统 ISP 预览：发现高增益共同失效模式

权威报告：`docs/A3_ALL_DATASET_TRADITIONAL_ISP_PREVIEW_2026-10-17.md`

- 输出根目录：`artifacts/a3_all_dataset_traditional_isp_preview_20261017`
- 已完成全部 45/45 条原始 RAW 序列。
- 每条均有 A3 prediction 和传统 ISP 视频。
- ISP 启用 highlight recovery 和 Reinhard，不启用 Group03 2DNR/3DNR。
- 每条取 3 个片段，每个 8 帧，共 24 帧。
- 视频为 1920x1080、HEVC、yuv444p、6 fps。
- 总产物约 `6.6 GiB`。

用户在 `04_mis20s1_outdoor_1500x` 及其他高总增益序列中发现共同的朦胧、过平滑问题。这不是单一视频编码或 ISP 2DNR/3DNR 导致，因为两者未启用；它触发了高增益专项诊断。

### 5.10 高增益专项诊断：确认主因是时域错位和域外输入

权威报告：`docs/A3_HIGH_GAIN_INFERENCE_DIAGNOSTIC_2026-10-18.md`

新增实现：

- `configs/a3_high_gain_diagnostic_20261018.json`
- `src/raw_fusion/finetune/high_gain_diagnostic.py`
- `scripts/run_a3_high_gain_diagnostic.py`
- `src/tests/test_finetune_high_gain_diagnostic.py`

产物根目录：`artifacts/a3_high_gain_diagnostic_20261018`

代表序列：

- `04_mis20s1_outdoor_1500x`，frames 0–7
- `04_shdarkroom_1000x`，frames 0–7

比较四个分支：

1. Input。
2. 原 A3 uniform tiling，64/56。
3. A3 weighted tiling，64/48，Hann floor 0.05。
4. 对每个目标帧重复 8 次作为模型输入，取 temporal output index 4，并使用 weighted tiling。

核心结果：

- weighted tiling 只能轻微降低拼接边界异常，无法解决主体朦胧。
- 原 A3 的梯度保持率约 `0.38`，temporal MAD ratio 约 `0.072–0.078`，说明真实时域变化被强烈压制。
- frame-repeat 梯度保持率为 `0.758/0.855`。
- temporal-mean 梯度保持率为 `0.847/0.907`。
- frame-repeat 的低频结构保持为 `0.952/0.973`。
- frame-repeat temporal MAD ratio 为 `0.551/0.694`。

固定 ISP 下，frame-repeat 明显减少了重影和过度时域平均，证明同一个 A3 checkpoint 在不依赖邻帧时仍能保留更多边缘；但它重新暴露了高增益彩噪和高频颗粒，不能作为默认输出。由此可得：

1. 朦胧的主因不是简单 tiling seam，也不是传统 ISP 里的 2DNR/3DNR。
2. 高增益输入超出 A3 主要训练分布，邻帧对齐/融合在低信噪比和运动下失效。
3. 正常 A3 分支去噪强但会把错位运动平均掉；frame-repeat 分支细节更强但噪声更大。
4. 两个分支互补，且都来自同一 A3 checkpoint，不需要引入旧教师或新训练。
5. 这直接构成下一阶段置信门控双分支融合的依据。

固定 ISP 视频：

- `artifacts/a3_high_gain_diagnostic_20261018/04_mis20s1_outdoor_1500x/traditional_isp_video/04_mis20s1_outdoor_1500x_high_gain_4way_overview.mp4`
- `artifacts/a3_high_gain_diagnostic_20261018/04_mis20s1_outdoor_1500x/traditional_isp_video/04_mis20s1_outdoor_1500x_high_gain_4way_detail.mp4`
- `artifacts/a3_high_gain_diagnostic_20261018/04_shdarkroom_1000x/traditional_isp_video/04_shdarkroom_1000x_high_gain_4way_overview.mp4`
- `artifacts/a3_high_gain_diagnostic_20261018/04_shdarkroom_1000x/traditional_isp_video/04_shdarkroom_1000x_high_gain_4way_detail.mp4`

验证结果：相关 9 个测试通过；11 个 JSON 均可解析；四段视频均为 HEVC 1920x1080 yuv444p、6 fps、8 帧；A3 哈希未变化。

## 6. 当前科学判断，也就是“为什么缺少高增益 GT 很关键”

可以简化为：我们缺少高增益动态条件下可信的干净真值或等价监督，但问题不只是“没有一张 GT 图”。

当前数据能够告诉我们噪声多大、帧之间哪里变化、某个输出是否更平滑，却不能可靠告诉模型：运动物体在该时刻的无噪声纹理到底是什么。静态时域均值在静止区域有效，在动态区域会产生重影；旧教师本身带噪和错误纹理；blind-spot 可以学习噪声统计，却不能凭空恢复被抹掉的确定性边缘；合成平移和现有退化模型又未覆盖真实传感器的模糊、CFA、FPN、色噪和运动错位组合。

因此训练级方案目前受监督质量限制。继续增加训练步数、损失权重或融合 alpha，只会在“静态更干净”和“动态更清晰”之间移动，不会自然产生缺失的真细节。这也是为什么下一步先做同一 A3 的确定性门控：它不声称恢复未知 GT，只尝试在模型已经表现出的两种推理行为之间，根据输入可观测的错位置信度做保守选择。

## 7. 当前可用模型与支持矩阵

| 对象 | 状态 | 允许用途 | 禁止事项 |
|---|---|---|---|
| A3 step-5000 | 当前默认基线 | MIS20S1 静态、已支持域、所有对比基准 | 不得覆盖或替换 |
| A8 sc635hai step-500 | 条件保留 | sc635hai 专用可选推理 | 不得宣传为统一模型，不得替换 A3 |
| A8 sc450ai checkpoints | 全部否决 | 仅历史审计 | 不得部署或继续同类盲点扫描 |
| A9 step-100 | 统一模型否决 | 仅历史审计 | 不得继续同方案到 500 step |
| A7 旧教师融合 | 全部否决 | 仅历史审计 | 不得原样复用 |
| 高增益 frame-repeat | 诊断分支 | A11 的保守候选分支 | 不得单独作为默认输出 |

传感器支持状态：

| 传感器/条件 | 当前状态 | 主要风险 |
|---|---|---|
| MIS20S1 低/中增益静态 | 支持，A3 基线 | 动态仍需检查 |
| MIS20S1 高增益动态 | 未解决 | 朦胧、重影、过度平滑 |
| SC465SL 已测静态域 | 支持 | 不能外推到所有动态高增益场景 |
| sc635hai 静态低色温 | 支持/条件支持 | 动态和颜色域仍有风险 |
| sc450ai | 不支持 | 高频塌陷、黑物体发空、裁剪、监督不足 |
| difficultpattern | 条件压力测试 | 不能代表完整传感器支持 |

## 8. 当前文件、进程和资源状态

截至 `2026-10-19 14:54 Asia/Shanghai`：

- 未发现本 Stage2 项目的 A3 高增益诊断、全数据预览、A8/A9/A10b 或训练进程仍在运行。
- 物理 GPU 0 为 RTX 3090，约 `5016 MiB / 24576 MiB`，有其他进程使用。
- 物理 GPU 1 为 RTX 4090，约 `21390 MiB / 24564 MiB`，利用率约 `98%`，由其他训练/推理进程占用。
- 不得终止这些外部进程。
- 历史上 PyTorch 枚举通常为 `cuda:0 = 4090`、`cuda:1 = 3090`，但 PCI 顺序和 `CUDA_VISIBLE_DEVICES` 可改变映射，每次运行必须同时核对 `nvidia-smi` 和 PyTorch device name。
- A11 初始工作应尽量直接复用现有 `.npy` prediction 和权重，在 CPU 上完成 mask、融合、指标和 ISP 渲染；仅在确实缺少某个分支预测时才考虑 GPU。

## 9. 当前 Git 状态和版本控制风险

- 分支：`master`
- HEAD：`3971228ce6a2fe643cc5c1ec378c73d782199e32`
- 工作树存在大量 modified/untracked 文件，包括恢复文件和 A5–A10b、全数据预览、高增益诊断的新增代码与报告。
- `.git/config` 当前无 remote。
- GitHub 已知部分快照提交为 `9964f776dd7ebd015a61f4fd416989b3cf4aba85`。
- 不能推断 2026-10-11 之后的新增文件已被提交或上传。
- 后续如需提交，必须先独立做文件大小审计、远端审计和精确 staging；不得把大 checkpoint、预测数组或视频误提交，也不得清理用户现有修改。
- 本交接不授权 commit/push。

## 10. 已关闭、不要重复的路线

1. A5 动态监督的简单 loss-weight、anchor-weight、冻结范围标量扫描。
2. A6 冻结 A3 加运动残差专家 v1–v4 的同类扫描。
3. A7 旧教师 residual 的 r3/r7/r11、alpha、clip 扫描。
4. 把旧教师输出当作 clean dynamic GT。
5. 把动态序列 temporal mean 当作 clean GT。
6. sc450ai noisy-only blind-spot step100–500 的同类训练。
7. A9 当前统一噪声幅度方案直接延长到 500 step。
8. 在 1435x hard holdout 上选阈值、拟合退化模型或训练。
9. 仅凭 motion/input ratio、单帧锐度或单个 crop 宣布通过。
10. 仅调整传统 ISP 锐化来掩盖 RAW 时域融合产生的重影/朦胧。ISP 可以帮助观察，但不能恢复已经在 RAW 输出中丢失的结构。

## 11. 尚未完成的工作

1. 最终目标“一个方案在多传感器、多增益、静态和动态条件下都可靠”尚未完成。
2. MIS20S1 和其他传感器的高增益动态朦胧仍没有可部署修复。
3. sc450ai 仍不支持，且现有数据不具备可信训练监督。
4. sc635hai 专用 checkpoint 仍只是条件保留，不是统一泛化解。
5. A8/A9/A10/A10b、全数据预览和高增益诊断的 GitHub 同步状态未核实。
6. 尚未建立可信的高增益动态 GT、等价监督或经过验证的完整退化生成器。
7. 用户尚未对最新高增益四路视频给出最终主观评分；A11 仍应使用相同 ISP 生成直观证据。

## 12. 推荐的下一阶段：A11 高增益置信门控双分支推理

### 12.1 目标

在不训练、不改变 A3 checkpoint、不使用旧教师的前提下，使用输入帧可观测的运动/错位证据，在两个 A3 推理分支之间进行保守、连续、边界平滑的 RAW 融合：

- 静态或高时域一致性区域：保留正常 A3 weighted temporal 分支，以获得强去噪。
- 运动、遮挡或明显错位区域：有限度引入 A3 frame-repeat 分支，以减少重影和过度平均。
- 不确定边界：通过连通域过滤、适度膨胀和 feathering 平滑过渡。

建议阶段编号：`A11_HIGH_GAIN_CONFIDENCE_GATED_INFERENCE`。

建议产物根目录：
`artifacts/a11_high_gain_confidence_gate_20261019`

建议最终报告：
`docs/A11_HIGH_GAIN_CONFIDENCE_GATED_INFERENCE_FINAL_2026-10-19.md`

### 12.2 第一阶段只读核对

新会话必须先完成以下核对，然后可直接继续，不必等待用户再次确认：

1. 校验 A3 checkpoint SHA256 与第 4.1 节一致。
2. 读取 `RECOVERY_REBUILD_20261010.md`、本交接和高增益诊断报告。
3. 确认两条高增益序列已有 input、A3 uniform、A3 weighted、frame-repeat 的 RAW/Numpy 产物和四路视频。
4. 检查是否存在本项目运行进程；不得重复启动相同任务。
5. 检查 GPU，但 A11a 优先复用现有 prediction，不应因 4090 被占用而阻塞。
6. 确认数据集目录没有被写入，所有新产物只写项目 `artifacts/`、`docs/`、`configs/`、`scripts/`、`src/`。

### 12.3 建议实现

先阅读现有 `high_gain_diagnostic.py`、A7 mask/feather 工具和 ISP 渲染脚本，复用本项目已有格式。不要重新发明 RAW 读写和 ISP 管线。

为每个目标帧构造只依赖 noisy input 的连续置信权重 `w`：

1. 计算相邻/参考帧的鲁棒亮度残差，先按 Bayer channel 或局部强度做归一化，避免把高增益噪声本身全部误判为运动。
2. 计算局部梯度方向/幅值不一致，识别结构性边缘错位，而不是单纯像素噪声。
3. 若现有依赖允许，加入局部 block/patch matching 的最优代价和次优代价差，作为匹配可靠度；不需要训练光流网络。
4. 将前后帧不一致、遮挡或 temporal outlier 作为低时域置信区域。
5. 对 mask 做小连通域过滤，避免彩噪点触发；对可信运动区域做保守膨胀，再使用距离变换或高斯 feather 形成连续边界。
6. 高置信静态区强制 `w = 0`；只有有充分输入证据的区域允许 `w > 0`。

融合定义应保持简单、可审计：

```text
output = (1 - w) * a3_weighted_temporal + w * a3_frame_repeat
```

所有值在与现有预测一致的 normalized RAW 域中计算，最后裁剪到合法范围并统计 legal RAW fraction。

不要把 A11 变成新的 alpha/clip 网格扫描。应先预注册一套保守 mask 规则，最多只做少量、结构上不同的 mask ablation，例如：

- residual only，用于证明噪声误触发问题；
- residual + gradient agreement；
- residual + gradient + matching confidence，作为推荐完整版。

ablation 的目的应是验证置信证据，而不是从几十个阈值中追求最好数字。所有阈值只能用 `04_mis20s1_outdoor_1500x` 和 `04_shdarkroom_1000x` 的预定开发帧制定；不能使用 sc450ai 1435x hard holdout 调参。

### 12.4 A11a 最小验证范围

先只对以下现有诊断序列的 frames 0–7 进行开发和裁决：

1. `04_mis20s1_outdoor_1500x`
2. `04_shdarkroom_1000x`

优先复用已有 A3 weighted 和 frame-repeat arrays，不重新运行网络。若现有数组确实缺失，才在 GPU 空闲且确认不会 OOM 后补推理。

统一输出五路固定 ISP 视频：

1. Input
2. A3 uniform baseline
3. A3 weighted temporal
4. A3 frame-repeat
5. A11 confidence-gated output

同时生成 overview 和 detail crop，ISP 配置必须与 2026-10-18 高增益诊断一致：启用 highlight recovery 和 Reinhard，不启用 2DNR/3DNR，固定同序列曝光和色彩参数。

### 12.5 必须记录的指标

每条序列、每个分支、每个 mask ablation 至少记录：

1. legal RAW fraction。
2. 融合区域覆盖率，以及 `w > 0`、`w >= 0.5` 的覆盖率。
3. 静态区 nonzero weights 覆盖率。
4. 相对 A3 weighted 的静态 shift：mean、P95、P99、max。
5. 梯度/高频保持率，但必须同时给出噪声敏感解释。
6. 低频结构保持率。
7. temporal MAD 及相对 input、A3、frame-repeat 的比例。
8. motion 区和 static 区分别的 temporal ratio。
9. tile seam/boundary anomaly，确认门控没有重新引入拼接边界。
10. mask boundary 两侧的输出跳变或边界梯度异常。

motion/input ratio 只能作为诊断项，不能单独决定通过。

### 12.6 固定 ISP 视觉验收

必须逐帧检查：

1. 人体、树枝、运动物体边缘是否比正常 A3 更清楚。
2. 是否减少朦胧和重影，而不是仅通过增加噪声造成“看起来更锐”。
3. 是否出现 frame-repeat 的彩噪、高频颗粒或闪烁。
4. 门控边界是否突变、断裂、出现光晕或局部锐度跳变。
5. 静态背景是否与 A3 weighted 保持一致。
6. mask 膨胀是否污染非运动区域。
7. 两条代表序列是否都成立，不能只在一个 crop 上成立。

### 12.7 明确通过条件

A11 只有同时满足以下条件才可登记为“可选高增益后处理”：

1. 两条代表序列均有可重复的运动边缘/重影改善。
2. 输出噪声明显低于纯 frame-repeat，不能只是把纯 frame-repeat 局部复制回来。
3. 高置信静态区与 A3 weighted 基本一致，且静态 nonzero weight 接近零。
4. mask 边界无可见突变、断裂、halo 或锐度台阶。
5. legal RAW fraction 为 `1.0`。
6. 没有引入旧教师纹理、伪细节或新的时域闪烁。
7. 指标和固定 ISP 肉眼结论一致。

即使通过，也只能保留为可选 deterministic postprocess，不能覆盖或替换 A3 checkpoint。

若不能同时满足这些条件，必须明确否决 A11，不允许用“略有改善、建议继续观察”收尾。

### 12.8 A11a 通过后的扩展

只有 A11a 通过，才进入 A11b，并冻结 mask 规则，不再针对每个序列调参。扩展到：

- MIS20S1 其他高增益档位，包括 outdoor 1000x/1500x 附近序列。
- SC465SL 的高增益或 difficultpattern 压力样本。
- sc635hai 的动态样本。
- sc450ai 1435x 仅作为 hard holdout 评估，不允许调整阈值。

A11b 仍然不训练模型。若冻结规则不能跨传感器工作，则结论应是“门控后处理不具备统一泛化”，而不是为每个传感器继续堆积手工阈值。

## 13. 如果 A11 失败，后续才考虑的训练路线

若 A11 明确失败，训练级下一候选应是“高增益条件化微调”，但它不是本交接直接授权的任务。开始前必须先向用户提交单独方案，明确：

1. 哪些现有数据能提供静态可信 target，哪些只能提供 noisy observation。
2. 动态区域如何避免 temporal-mean/旧教师伪真值。
3. 是否只训练输出头、adapter 或极少量参数，以保护 A3 静态能力。
4. 如何编码传感器和总增益条件。
5. 怎样构造更可信的退化，包括空间相关噪声、色通道耦合、FPN、模糊/CFA 响应，而不使用 hard holdout 拟合。
6. MIS20S1、SC465SL、sc635hai、sc450ai 各自的 train/dev/holdout 划分。
7. 静态、动态、跨传感器和固定 ISP 的预注册验收门槛。

在没有可信高增益动态监督时，不应承诺训练能直接解决所有朦胧问题。更现实的路线可能是：静态区域使用可信自监督/均值监督，动态区域使用保守一致性约束和冻结 A3 主干，同时把无法监督的场景保留为 holdout。任何此类训练必须由用户另行明确授权。

## 14. A11 完成标准

1. 代码、配置和报告均可复现，随机性明确固定或完全确定性。
2. 不修改数据集，不修改 A3 checkpoint。
3. 两条 A11a 序列的指标表完整。
4. 每条序列都有统一 fixed-ISP overview 和 detail 视频。
5. 有 mask、权重分布、静态污染和边界连续性的证据。
6. 明确给出一个可保留的 A11 配置，或者明确否决全部配置。
7. 所有新增 JSON 可解析，数组 shape/dtype/range 合法，视频编码和帧数经过检查。
8. 相关单元测试通过。
9. 报告记录当前仍未完成项。
10. 在最终报告中再次记录 A3 SHA256，证明基线未被替换。

## 15. 新会话的执行顺序

1. 读取本交接和第 4、5、10 节列出的权威报告。
2. 只读校验 A3、现有高增益数组/视频、进程、GPU、Git 状态和数据集只读约束。
3. 阅读高增益诊断、A7 mask/feather 和传统 ISP 实现，确定可复用组件。
4. 先用已有数组实现 A11 的 mask、连续 feather、融合、指标和测试。
5. 在两条 A11a 序列上生成五路 fixed-ISP 视频和结构检查结果。
6. 按第 12.7 节作出明确通过/否决裁决并写最终报告。
7. 若 A11a 通过，再按冻结参数进入 A11b；若失败，停止后处理扩展，提交训练级方案供用户另行授权。

不要在只读核对通过后停留在计划阶段。当前用户已授权直接开始 A11 无训练阶段，但没有授权新训练、Git 提交/推送或删除操作。

