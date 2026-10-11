# RViDeformer 微调计划:真实噪声域的高保真去噪模型

日期:2026-09-20
状态:计划稿(待 dad 评审后启动)
关联文档:`/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/raw_fusion_distill/README.md`(现有伪 GT 主线)、`/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/raw_fusion_distill/docs/superpowers/specs/2026-09-02-reliable-detail-supervision-design.md`、`/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/raw_fusion_distill/docs/superpowers/specs/2026-09-05-unreliable-region-reconstruction-design.md`
数据来源:`/HardDisk/jingzeliang/Data/Company_Dataset/dataset`(**只读,永不写入**)
权利边界:所有新增代码、配置、产物只写 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel`(下称"仓库")内;数据集侧不做任何增删改

本计划中的全部数据数字来自 2026-09-20 的像素级实测勘察(6 个并行测量组 + 4 个独立对抗复核;原始勘察 JSON 存于 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/raw_fusion_distill/docs/finetune_plan_20260920_survey_result.json`,过程脚本与预览 PNG 在 `/tmp/rvd_ft_analysis/`,易失,仅仓库内 JSON 为准)。

---

## 1. 目标与定位

**最终目标:训练一个能同时抑制噪声、保留高频细节的 RAW 视频去噪模型。**

- 基座:官方 RViDeformer-L_CRVD 预训练权重(`/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/RViDeformer/checkpoints/RViDeformer-L_CRVD.pth`,CRVD indoor 监督训练,RVRT 风格 3D U 形 Transformer)。已知缺陷:跨域零样本推理时把高频细节抹平——这正是现有伪 GT 主线(reliable-detail)要打补丁的根源,本计划从教师端直接解决。
- 核心思想:**监督信号不依赖任何去噪教师的输出**。用同场景低增益 captures 的时域平均当"真干净参考"(实测残差噪声 <1 DN、高频真实存在),直接惩罚过平滑;用标定组拟合的噪声模型合成数据,放大监督覆盖面。
- 与现有主线的关系:本计划是新设的微调支线,不改现有伪 GT 主线的代码路径;新增代码放独立子包,配置指纹机制照常适用。
- 最终形态(已与 dad 确认):先得到直接可用的去噪模型;收尾阶段(可选)用它为 MIS20S1 重新生成更好的伪 GT,替换现有主线的教师,反哺学生蒸馏。
- 范围策略(已确认):**Phase A 只做 MIS20S1 单传感器闭环;Phase C 最后混训全部传感器保泛化。**

## 2. 数据集实测摘要

### 2.1 总表

| 组 | 传感器 / 场景 | 流数 × 帧数 | 分辨率 | 位对齐 | 黑电平(12bit) | CFA | 静态区单帧 σ(12bit DN) | 参考价值 |
|---|---|---|---|---|---|---|---|---|
| 04/shdarkroom | MIS20S1 暗房,同场景 7 增益点(10x/64x/128x/313x/632x/645x/1000x) | 200 帧 ×6 + 300 帧 ×1(313x) | 1920×1080 | 高 12 位(需 >>4) | 传感器 256,逐流实测 | RGGB | 10x≈6 / 64x≈16 / 128x≈34 / 313x≈20 / 632x、645x、1000x 静尾估计 10–12、典型像素校正 20–28 | **最高**:跨增益真值 |
| 04/mis20s1_outdoor | MIS20S1 公园黄昏,4 增益点(128x/512x/1000x/1500x) | 200 ×2 + 198(512x)+ 195(1500x) | 1920×1080 | 高 12 位 | 传感器 256 | RGGB | 24 / 12 / 7 / 7(后三者同样偏低估) | 次要:运动丰富,无跨增益参考 |
| 04/标定 | MIS20S1 ColorChecker 平场 + 盖镜头黑帧,9 增益(100→25600) | 50 帧 ×18(文件名 FN=10 是假的) | 1920×1080 | 高 12 位 | 256.7(全增益恒定,逐通道差 ≤1.2 DN) | **GRBG**(与场景流相位相反) | 噪声梯子 σ_t(Gr) 9.5 → 166.7 | 噪声模型 + 合成数据 |
| 02 | SC465SL 暗房("2000x"/"256x" 两条) | 300 帧 ×2 | 2560×1440 | 低 12 位 | 实测底座 261 附近(标签 252,漂移 ±9) | RGGB | 17.8 / 24.5(标签与噪声排序相反) | 真实性好;来源待查 |
| 05 | SC465SL 超市布景,6 级"增益梯子" + 2 条 difficultpattern | 200 帧 ×9 + 100 帧 ×1(IspG=1024) | 2560×1440 | 高 12 位 | 底座代理 253–263,逐 capture 漂移 | RGGB | 26.5 → 17.2(随标称增益反常下降) | 存疑:IspG=1024 流低于 shot-noise 下限,疑被时域处理 |
| 01 | sc635hai(3200×1800)+ sc450ai(2688×1520) | 150 / 200 / 210 / 150 帧 | 两种 | sc635hai 高 12 位;sc450ai 低对齐 | 257.5–260 | RGGB | 38.7 / 20.2 / 29.7 / 27.6 | 第三期泛化用 |
| 03 | MIS20S1 的 2DNR(denoised)/ 3DNR(fused)ISP 输出 | 200 帧 ×4(128x/645x 各二) | 1920×1080 | 低 12 位 | ≈300(readme 说 ~300,实测 295–310) | RGGB | 3DNR 高通能量 > 2DNR(645x:28.3 vs 23.2 DN) | 仅这两条流有 reliable-detail 辅助监督的前提 |

### 2.2 Phase A 主战场:MIS20S1(04 组)

**shdarkroom 增益梯子是全数据集的核心资产:**

- 同一个静态暗房布景(分辨率图卡、棋盘、色轮、泰迪熊、假人、风扇)在 7 个增益点重复采集;10x/64x/128x/313x/645x 五点亚像素互锁(相位相关 <0.3 px,静态区相关 0.93–0.95)。
- **10x 流 100 帧时域平均的残差噪声仅 0.62 DN(分半法实测,独立复核 0.68–0.73),且高频细节真实存在**:归一化后图卡区细节能量 58.5 DN vs 645x 单帧 54.8 DN——高增益帧的高频只是被噪声掩埋,不是不存在。
- 645x 时域平均残差 1.25 DN、128x 3.37 DN:中增益平均也可作次级参考,但 128x 有 5.3% 曝光漂移。
- 坑:632x 与 1000x 相对其他点整体横移 +63 px(采集间碰了相机),必须配准或初期弃用;每条流背景都有行人/风扇/反光金属丝运动区(时域平均在该处重影,须用时序 std mask 剔除);每档增益重新打过灯,**标称增益完全不预测噪声量级**(σ 峰值在 128x)。
- 曝光关系:10x 图卡信号 1091 DN vs 645x 194 DN(约 5.6 倍),跨增益配对须做逐通道曝光归一化(线性标定,静态区稳健回归)。

**mis20s1_outdoor(次要):** 同传感器 4 个增益点,但黄昏光照在采集间隔内漂移(18:38–18:55),跨增益像素配对不可靠;行人 + 摇动树叶使 23–76% 面积运动污染。只做单流监督(静态区时域平均)与合成噪声的载体,不承担跨增益真值。

**标定组(噪声模型):**

- 每增益 50 帧(文件名 FN=10 错误)、ColorChecker 平场 + 盖镜头黑帧,黑电平 256.7 全增益恒定。
- 噪声拟合(逐通道):σ² = a·μ·G + b·G²,a ≈ 0.136–0.157 DN/e-,b ≈ 0.1 DN²(读噪声可忽略,shot 主导),梯子残差 ≤6.7%;DSNU 0.18 → 7.89 DN(0.07%→3.1% 底座)。
- 已知异常:AG=800 黑帧有 1-LSB 偏置翻转(帧均值交替 257.04/256.04),AG=1600 黑帧异常干净——两点在拟合时降权或剔除。
- **CFA 相位与场景流相反(GRBG)**:用标定数据前先翻相位(或等价地按 GRBG pack);绿通道两相位类均等 0.5–0.7%(两次独立测量:勘察组 53 窗扫描 0.72%、单文件复核 0.5%)证实。
- 局限:平场拟合的 σ(μ) 还没和场景流实测对齐(场景级 PTC、色度噪声结构未测,见 A2)。

### 2.3 混训组(Phase C)与 03

- **02(SC465SL 暗房)**:全数据集噪声真实性最好的 raw(读噪声主导、时域白噪声、无处理痕迹),300 帧、静态布景;但两条流的"2000x/256x"标签与实测相反(256x 反而噪 1.35 倍,亮点高 3.4 倍,疑为 WDR/色调模式标签),作为"同场景双曝光"使用,标签仅当名字。
- **05(SC465SL 超市布景)**:IspG=1024 流时序噪声 4.4 DN、低于 photon shot-noise 下限(≥5.1 DN),证明经过时域处理;6 级"增益梯子"实测是色调映射梯子(信号恒定、高增益端高光压缩、噪声反降)。噪声统计不可当真 raw;但 B..H 七条流象素互锁(0.91–0.99),difficultpattern(细线图卡、棋盘、暗屏)是很好的细节压力测试内容。数据背景待与 dad 深入确认后定性。
- **01(公司双传感器)**:sc635hai(131x 低色温 + 16x 运动)与 sc450ai(1435x 黑物运动 + 804x 灯光变化),极暗(SNR 1–2)、色度噪声重;sc635hai/131x 时域平均"文字可辨",监督质量好;16x 流无 ≥16 帧无行人窗口。四条流实际都是 10-bit 量化(x%64==0 或 x%4==0 占 100%)。sc450ai 1435x 的 readme 元数据(高 12 位、黑电平 4128)是**错的**,实测低对齐、底座 ~258——按 readme 处理会毁数据。
- **03(2DNR/3DNR)**:与主线同源的 128x/645x ISP 输出(位于 `/HardDisk/jingzeliang/Data/Company_Dataset/dataset/03_MIS20S1_2D3D/`);200 帧、黑电平≈300、低 12 位、逐帧文件与流字节一致。3DNR 比 2DNR 保留更多高频(645x 高通能量 28.3 vs 23.2 DN、配对帧噪声 24.9 vs 17.5 DN)——reliable-detail 的前提在此成立,但**只有 128x/645x 有**,新增益点没有 2DNR/3DNR 可用。

### 2.4 元数据陷阱清单(处理规则)

1. 位对齐逐源判定(04 场景与 05 高 12 位 >>4;02、03 低 12 位;01 组内分裂),不信文件名。
2. 黑电平逐流实测(传感器名义 256,但场景底座 250–273 漂移;禁止沿用主线惯例的 252 常数)。
3. CFA:场景流一律 RGGB(已实测 128x/645x 绿均等 0.3%);标定流 GRBG,用前翻相位。
4. 帧数按文件字节数整除验证(标定 50 非 10;1500x=195、512x=198、313x=300、02=300、05A=100)。
5. 增益/快门元数据一律不进模型,噪声条件只用实测 σ。
6. 01 组 sc450ai 流:低对齐 + 有效 10-bit,readme 黑电平 4128 作废。

## 3. 总体路线

```text
Phase A  MIS20S1 单传感器闭环(本周起步)
  A0 数据管线与适配层 → A1 干净参考构建 → A2 噪声模型补测与合成器
  → A3 RViDeformer 微调 → A4 评估与验收(leave-one-gain-out)
Phase B  验证加固(与 A4 融合进行):消融、失败案例分析、门槛校准
Phase C  混训泛化:01 + 02 + 05 全量混入,防遗忘,泛化评估
收尾(可选)  新模型重生成 MIS20S1 伪 GT → 主线教师替换 → 学生重训对比
```

里程碑:M1 = 参考/mask/曝光标定产物落地;M2 = 噪声模型报告 + 合成器通过真实性检验;M3 = Phase A 模型 + 评估报告(过平滑是否治住的结论);M4 = 混训模型 + 泛化报告;M5 = 新伪 GT + 主线学生对比。

粗略工作量估计(单卡 4090):A0+A1 约 1–2 天,A2 约 0.5–1 天,A3 首跑 1 天、迭代若干,A4 约 0.5 天;Phase A 合计约 1 周。Phase C 约 1 周。收尾约 2–3 天。每次启动 CUDA 前照例 `nvidia-smi` 选空闲卡。

## 4. Phase A 详设(MIS20S1 闭环)

### A0 数据管线与适配层

- 新增子包 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/src/raw_fusion/finetune/`(独立于现有主线模块;代码风格遵循项目规范:>3.12 type hint、英文注释、无 fallback、pytest 放 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/src/tests/`)。
- 流读取层:逐源适配表(位对齐 shift、黑电平、CFA pack、曝光元数据)驱动,memmap 分块;每条流落一份 `stream_manifest.json`(帧数、黑电平实测、σ 实测、运动区统计、曝光系数)。
- 跨增益配准:632x/1000x 相对基准(645x)做相位相关配准(+63 px 平移);配准精度写入 manifest,不合格则该流标记 `unpaired`(初期直接弃用,不阻塞)。
- 曝光归一化:每对 (noisy 流, 参考流) 在静态区拟合逐通道线性系数(斜率+截距),残差分布写入 manifest。
- 时序运动 mask:逐像素时序 std + 自适应阈值(参照勘察配方),输出 [T,1,H,W] float mask;运动区同时从输入监督与参考中剔除。
- 产物全部落 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/artifacts/finetune_mis20s1/`,带 ETA 的进度输出(沿用 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/raw_fusion_distill/src/raw_fusion/reliable_detail_progress.py` 的契约风格)。

### A1 干净参考构建

- 以 shdarkroom/10x 为主参考:100–200 帧时域平均(chunked 累加),产出 [1, 4, 540, 960] packed RGGB 参考 + 逐像素置信图。
- 质量验证(硬性):分半(even/odd)残差 σ ≤ 1 DN;与 64x 平均(曝光归一化后)逐通道一致(偏差中位数 ≤ 2 DN);图卡区高通能量 ≥ 645x 单帧归一化细节的 90%。
- FPN 处理:10x 平均仍含 DSNU(~0.2–1 DN 量级)+ PRNU 残留;用长平均估计逐像素 FPN 图并从参考中扣除(FPN 图本身作为副产物存档,供 A2 合成器把 FPN 加回)。
- 运动区:参考在运动区无效,统一用 A0 的运动 mask 控制;mask 覆盖率(预计 69–84% 静态)写入 manifest。
- 副参考:645x/128x 自身时域平均(残差 1.25/3.37 DN)用于"中增益→中增益"课程与交叉验证,不做主 GT。

### A2 噪声模型补测与合成器

- 场景级 PTC:在 shdarkroom 各流(以及 outdoor)上测 σ(μ) 曲线(分亮度带统计),与标定拟合 σ²=a·μG+bG² 对齐;不一致则以场景实测为准重标 a、b。
- 色度噪声结构:逐 CFA 通道 σ 分解与通道间相关;决定合成时是否需要通道相关项。
- 合成噪声生成器:clean 参考 + 实测 FPN 图 → 按目标增益 G 加 shot+read 噪声、10-bit/12-bit 量化与黑电平,生成 (synthetic noisy, clean) 对;覆盖标定梯子的 9 个增益点。
- **真实性检验(硬门槛)**:合成对的配对帧噪声(3×3 高通差分法)与同增益真实流实测 σ 偏差 ≤15%,高通能量谱形状目检一致;不达标则修模型,不带病进入 A3;连续两轮仍不达标,则回退为仅用真实跨增益对训练(合成对降级为数据增广),并在 M2 报告中显式记录。

### A3 RViDeformer 微调

- 架构与加载:按 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/RViDeformer/run_mis20s1_raw_inference.py` 的方式把官方 checkpoint `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/RViDeformer/checkpoints/RViDeformer-L_CRVD.pth` 的 state_dict strict 加载进训练版 `RViDeformer`(img_size=[8,64,64] 约定);加载时机在全部模块构建完成之后(项目规范)。
- 训练约定:起点以官方监督脚本为基准(sensor 域 patch 128 → packed 64、6 帧时序窗、batch 1–2、lr 1e-4),冒烟 200 步验证显存与收敛后,可试 8 帧窗与 batch 4;输入归一化改为本域约定(实测黑电平、(x−black)/(4095−black))。
- 训练数据(Phase A):
  1. **真实跨增益对**:输入 = 高增益流(313x/645x/1000x、可含 64x/128x)的连续 6–8 帧窗;目标 = 10x(或 64x)时域平均经曝光归一化后的 packed 参考;静态 mask 内计损失,饱和区剔除(阈值按本域黑电平)。**10x 与 64x 流只作监督目标、绝不作为模型输入**,防止"输入≈目标"的恒等映射让训练损失和验证指标双双失真。
  2. **合成对**:A2 合成器产物,目标 = clean 参考,输入按 9 个增益点均匀采样。
  3. 配比起始 1:1,A4 消融调整;outdoor 流以"静态区自平均"配对方式少量混入,防场景单一。
  4. (可选辅助)128x/645x 两条流叠加 reliable-detail 式辅助项(3DNR−2DNR 的 B2 频带 × 可靠性 mask),复用现有 oracle 代码——仅这两条流有 2DNR/3DNR。
- 损失(起始值,冒烟后校准):Charbonnier(pred − target) 主项 + 0.1×梯度损失(高通区按参考细节能量图加权,直接抑制过平滑)+ 0.1×频域幅度 L1(可选消融项);运动 mask 区与饱和区不计损失。
- 训练制度:全参数小 lr 微调(不做从零训练);前 1k 步可冻结光流与浅层(消融决定);EMA 权重备份;按验证增益点(见 A4)early stop;checkpoint 与 train.jsonl 落 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/artifacts/finetune_mis20s1/training/`。
- 长训预期:数据量小(≈2 个场景 × 11 增益点),参照官方"小数据 × 长训"经验,数千步至万步级,以验证指标平台为准,不盲追步数。

### A4 评估与验收

- **验证协议:leave-one-gain-out**。例如:训练含 313x/645x/64x/128x,验证在 1000x(配准修复后)与 645x 的时间段 holdout;最终汇报一次 645x 全序列可视化。禁用验证增益点的任何监督。
- 指标(全部相对 10x/64x 干净参考、静态 mask 内,12bit DN):
  - 重建:PSNR / RMSE(vs 参考);
  - **高频保持(本计划的靶心)**:图卡/棋盘区高通能量比 out/ref ∈ [0.85, 1.15],Sobel/Laplacian 增益相对参考恢复 ≥90%;无过冲(halo)目检;
  - 平坦区:残差 σ ≤ 2 DN 且不劣于 3DNR;
  - 时域:输出帧间平均绝对差 ≤ 旧教师伪 GT 的同指标;
  - 量化合法性:输出经 round(x·(4095−black)+black) 后无越界。
- **对比基线(同一评估协议)**:2DNR、3DNR、旧教师 RViDeformer-L_CRVD 零样本(= 现伪 GT)、现有伪 GT 主线学生;四宫格固定 ISP 视频(`/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/raw_fusion_distill/scripts/render_fusion_video.py` 风格)目检。
- 门槛数值标注为**初值**:A1 量完基线后用"旧教师 vs 参考"的实际差距校准,要求新版在高频保持项上显著优于旧教师且重建项不劣。
- **最终裁决**:量化门槛全过后,"过平滑是否治住"以四宫格目检为准,由 dad 确认后才算 M3 完成;自动指标只作初筛。

## 5. Phase C 详设(混训泛化)

- 混入源与角色:01 组(两颗新传感器,跨传感器泛化主力)、02(真 raw 双曝光,SC465SL)、05(仅 difficultpattern 内容作难样本,噪声当"已处理流"对待,不进噪声模型)、MIS20S1 全量保留。
- 监督来源按可用性分层:跨增益真值(仅 shdarkroom)> 静态区时域平均(02/01 的 131x 流)> 合成噪声对(标定模型仅适用 MIS20S1;其他传感器无标定,用流内实测 σ 的异方差自平均近似或仅用自平均对)。
- 防遗忘:保留 MIS20S1 与(必要时)CRVD 验片子集做 replay;lr 降至 Phase A 终值的 1/5–1/10 再热。
- 架构变更备选(仅当 Phase C 出现域混淆):加增益/噪声条件 embedding(σ 归一后进 patch embedding)——这是唯一预设的架构变更点,需 dad 批准后实施。
- 泛化评估:每传感器留一条流做 held-out;汇报跨传感器指标分解,确认"治过平滑"没有以牺牲泛化为代价。

## 6. 收尾(可选):伪 GT 替换

1. 用新模型按现有推理协议(时域 8 帧窗 overlap-4 线性融合 × 空间 64 tile overlap-8)为 128x/645x 重生成伪 GT;推理导出脚本落在 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/src/`(参考 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/RViDeformer_distill/RViDeformer/run_mis20s1_raw_inference.py` 的切窗逻辑,该仓库只读)。
2. 产出沿用现有契约:逐帧 raw + pixel_sha256 manifest + `label_semantics: pseudo_ground_truth`;伪 GT 落 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/artifacts/finetune_mis20s1/pseudo_gt/`;登记新 artifact manifest → 配置指纹必然变化 → 旧 checkpoint 不可沿用(机制照旧,防误用)。
3. 主线学生(reliable-detail mainline)用新伪 GT 重训,四宫格 + 量化指标对比旧伪 GT 版本,结论回报 dad 后再决定是否切换主线。

## 7. 风险与未决问题

1. **场景多样性不足(Phase A 内最大风险)**:MIS20S1 本质只有暗房 + 公园两个场景,模型可能学到"暗房先验"。缓解:强增广(翻转/转置/亮度-增益抖动)、合成对覆盖全增益、Phase C 尽早启动。
2. **05/02 数据来源未明**:05 疑被处理、02 标签颠倒,背景待与 dad 深入探究(已排期);定性前它们不进噪声模型、不作真值。
3. **参考的系统性偏差**:时域平均 = 时间平均而非真实单帧干净像,拖影/闪烁残差可能以低频形式进入 GT;A1 的双参考交叉验证 + FPN 扣除缓解,残余写入报告。
4. **场景级噪声模型缺口**:平场标定 ≠ 场景实测(A2 补测);合成对若失真会教坏模型,故 A2 设硬门槛。
5. **运动区无监督**:15–47% 面积被 mask 掉,模型在这些区域的行为靠先验与合成对兜底;评估时单独汇报运动区指标(仅参考可用处)。
6. **算力与磁盘**:单卡 4090 足够(训练 patch 小),但启动前必须 `nvidia-smi`;`/HardDisk` 已 99% 满(222G 空闲),所有产物只写本仓库所在盘,启动前检查工作盘余量。
7. **域约定切换**:本支线黑电平用实测 256 而非主线的 252,两套约定并存;所有 manifest 显式记录,防止后续误配。

## 8. 里程碑产出物一览

| 里程碑 | 内容 | 位置 |
|---|---|---|
| M1 | stream/参考/mask/曝光标定产物 + manifest | `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/artifacts/finetune_mis20s1/supervision/` |
| M2 | 噪声模型报告 + 合成器 + 真实性检验记录 | `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/artifacts/finetune_mis20s1/noise_model/` + `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/docs/` |
| M3 | Phase A checkpoint + 评估报告 + 对比四宫格 | `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/artifacts/finetune_mis20s1/training/`、`.../inference/` + `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/docs/` |
| M4 | 混训 checkpoint + 泛化报告 | `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/artifacts/finetune_mixed/` + `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/docs/` |
| M5 | 新伪 GT + 主线学生对比结论 | `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/artifacts/finetune_mis20s1/pseudo_gt/` + `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/docs/` |

每阶段实验结论按项目惯例写入 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/docs/`(报告文档)与 `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/doc/codex-changes/`(代码改动,需 dad 明确要求总结时)。
