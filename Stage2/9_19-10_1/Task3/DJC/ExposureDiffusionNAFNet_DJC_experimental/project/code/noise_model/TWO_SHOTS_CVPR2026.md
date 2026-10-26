# MIS20S1 噪声建模：CVPR 2026 2-Shots in the Dark

主论文：Liying Lu, Raphaël Achddou, Sabine Süsstrunk, [2-Shots in the Dark: Low-Light Denoising with Minimal Data Acquisition](https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_2-Shots_in_the_Dark_Low-Light_Denoising_with_Minimal_Data_Acquisition_CVPR_2026_paper.pdf), CVPR 2026。作者[公开演示代码](https://github.com/IVRL/2-Shots-in-the-Dark)提供了暗帧频谱采样和散粒噪声估计的 notebook。本项目的实现按论文公式独立编写，未复制代码。

本地下载论文：`references/lu_2shots_cvpr2026_arxiv.pdf`（SHA256 `daad0ed1269cf9bf66883ffe9e3de4e429b216a957828dd8683101af0e25efd2`）。同时阅读了 Feng 等人的 [PNNP](https://arxiv.org/abs/2310.09126)（TPAMI 2026），本地 `references/feng_pnnp_tpami2026_arxivv3.pdf`（SHA256 `44b71081d05996eac1c96b91380f399ad7db0813c1be47ca4581b8724972494b`）；[作者仓库](https://github.com/fenghansen/PNNP)没有发布噪声代理的训练代码，因此本阶段以可复核的 CVPR 方法为主。

## 实现与参数

`two_shots.py` 实现论文 Sec. 3.1、Eq. (1)–(14)：按 Bayer 四通道对一张暗帧做 `σ=50` 像素高斯平滑，取剩余噪声的二维 Fourier 幅值，用四通道共享的随机相位扰动生成新噪声，再做 `K=10` 轮逐通道秩直方图匹配与频谱幅值校正，最后加回固定低频图案和通道均值。`σ` 是**打包后的** 540×960 网格上的像素尺度，`K` 与 `σ` 均直接采用论文/作者演示值，没有用留出集调参。SciPy FFT 与 IFFT 使用配对归一化，因而不另乘论文采用酉变换记法中的归一化因子。

`calibrate_two_shots.py` 对九档增益各取第一张 1920×1080 MIS20S1 暗帧作为参考，RAW 解码为小端 16 位容器右移 4 位后的 12 位 DN，并以 GRBG 打包成四通道。输出 `runs/noise/two_shots_cvpr2026/mis20s1_gain*_2shots.npz` 存储参考帧和频谱先验，`calibration.json` 存储每通道暗电平、残差标准差、曝光、散粒噪声系数和拟合诊断。散粒斜率现在用前 15 对彩色帧拟合，在第 15–19 对上选择“暗截距／非负截距／零截距”策略，最后 20–24 对只做检查；这避免把 83 ms 暗帧方差强制当成所有彩色曝光的截距。第 40–49 暗帧仍不参与暗先验拟合。

论文 Eq. (17) 的散粒噪声系数满足 `Var(y | x) = g x + Var(dark)`，暗帧只有 `x=0`，**不能单独识别 g**。这里用每档现有 50 张同一静态彩色场景帧，每对同位置像素之差消除静态纹理，经 32 DN 信号分箱后拟合斜率。前 15 对用于拟合，第 15–19 对选择截距策略，第 20–24 对只做最终检查。非负截距约束用于避免把负的线性回归截距当成物理方差；12800/25600 档最终选择零截距策略，因此仍标记为 provisional。这个重复帧估计是对论文 Eq. (17) 的数据更充分的改写，**不是**论文附录中的单张 3×3 patch 估计。模型合成公式为 `g·Poisson(x/g) + sampled_dark`，其中 `x` 必须是同相机黑电平扣除后的线性 DN。

在项目目录执行：

```bash
/opt/miniconda3/envs/dl/bin/python -m noise_model.calibrate_two_shots \
  --calibration-root ../Data/Company_Dataset/dataset/04_Sigmastar_7_30/mis20s1_calibrationdata \
  --output runs/noise/two_shots_cvpr2026
/opt/miniconda3/envs/dl/bin/python -m noise_model.validate_two_shots \
  --calibration runs/noise/two_shots_cvpr2026/calibration.json
```

程序接口：

```python
from noise_model.two_shots import load_calibrated_model, synthesize_observation

model = load_calibrated_model("runs/noise/two_shots_cvpr2026/mis20s1_gain1600_2shots.npz")
dark_dn = model.sample(seed=2026)
noisy_dn = synthesize_observation(signal_dn, model, shot_gain_dn=2.64307, seed=2026)
```

## 当前实测结果与边界

九档当前选用的 `g`（DN/电子的等效系数，按 100→25600 增益顺序）为 `0.1586, 0.3066, 0.6082, 1.2081, 2.6431, 5.2421, 10.7377, 24.1267, 51.1979`。100–6400 选择非负截距策略，12800/25600 选择零截距策略；后两档仍标记为 `provisional`。最终 5 对彩色留出误差约为 2.1%–4.0%（100–6400），12800 为 4.7%，25600 为 9.9%。这些是同一静态色卡的分箱方差误差，不是极暗近黑区的确认性误差。

**极暗信号范围的补充核查：**彩色留出检验参与计算的最小黑电平扣除信号为 **48 DN**；低于该值的近黑区域没有被这项斜率检验覆盖。彩色帧曝光从 83,317 µs 缩短到 326 µs，所以高增益不再把长曝光暗方差固定为截距。12800/25600 的自由线性回归截距仍为负，故最终使用零截距作为有约束的外推策略；这不是物理证明读出方差为零，而是当前数据下非负截距约束的最小偏差选择。`g` 仍不能视为近黑区精确标定。

| 增益 | 旧固定截距误差 | 当前选中策略误差 | 彩色曝光 (µs) |
|---:|---:|---:|---:|
| 100 | 5.53% | 4.01% | 83,317 |
| 1600 | 6.89% | 3.29% | 5,186 |
| 3200 | 7.65% | 3.73% | 2,578 |
| 6400 | 6.62% | 2.11% | 1,275 |
| 12800 | 13.60% | 4.69% | 623 |
| 25600 | 19.74% | 9.85% | 326 |

这些误差来自同一静态彩色场景的最后 5 对帧，不是独立照度序列，也没有证明低于 48 DN 的外推。因此 `g` 应作为**初步斜率估计**；进入极暗合成前，应检查低信号分箱与量化，并保留非负截距、零截距和旧暗截距三种参数敏感性范围。新的候选值和每种策略的内部／最终误差均保存在 `calibration.json`。

暗帧留出验证记录在 `validation.json`；它比较独立的第 45 帧与合成帧的每通道均值、标准差、残差 Wasserstein 距离、径向功率谱和通道相关系数。九档各四通道的合成/真实残差标准差比值落在 `0.997–1.003`，各档径向频谱比值的中位数落在 `0.996–1.004`，下采样残差 Wasserstein 距离平均为 `0.18–0.29 DN`。这些是单帧分布诊断，只覆盖**同曝光暗帧分布**，不是最终的低照度去噪性能。

更严格的[一帧建模、其余 49 帧验证报告](ONE_FRAME_DARK_VALIDATION.md)显示：高增益固定图案和部分增益的整帧偏置没有被原论文的一帧采样器完整表达。判断适用性时应以这个多帧检验为准。

**后续更正：**原验证报告中的 1 DN 量化分布距离不符合本批 RAW 的 2 DN 码格。已在[40/10 同条件多帧分解对照报告](PNNP_DECOMPOSITION_COMPARISON.md)中更正，并将单帧和多帧模型按相同码格重新比较。多帧方案在高增益改善固定图案、时间方差和频谱，但合并直方图 W1 略差；低增益继续保留本单帧基线。

现有暗帧文件名标注曝光均为 **83317 µs**，彩色校准帧随增益变化，最高增益为 **326 µs**。论文方法的暗帧应匹配目标相机、增益、曝光及温度；当前 83 ms 的暗帧在高增益短曝光图像上直接相加尚未得到验证。建议补拍每档至少与目标曝光匹配的暗帧（最好多个曝光、稳定记录传感器温度），以及每档数级均匀照明的重复平场帧，以确认 12800/25600 的散粒系数和暗电流随曝光的变化。Apple clean RAW 与 MIS20S1 的 CFA、黑/白电平、响应和色谱不同，不能直接视为同一相机的干净真值；若用于合成训练，还需验证跨传感器的线性化/响应映射。

本模块仍是**噪声合成器**，不把频谱采样器直接当作 R1 似然。现在的第一版可求导观测模型单独实现于 [`quantized_likelihood.py`](quantized_likelihood.py)，由泊松计数、暗残差高斯混合和 DN 量化区间概率组成；拟合与 40/10 暗帧验证见 [`fit_quantized_likelihood.py`](fit_quantized_likelihood.py) 和 `runs/noise/quantized_likelihood/calibration.json`。该似然将帧级及行列分量边缘化为每通道方差，属于 R1 的可解释近似，尚未证明能完整表达共享带状相关性。
