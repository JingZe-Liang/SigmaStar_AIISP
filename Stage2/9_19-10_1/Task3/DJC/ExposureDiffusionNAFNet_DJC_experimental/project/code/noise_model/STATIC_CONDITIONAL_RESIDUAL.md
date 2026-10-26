# 静止背景条件残差候选模型

此模型是对 `runs/noise/structure_exploration_20260929/FINDINGS.md` 的直接改进：它不再把高增益有光残差强制为单一条件高斯，而是按 `profile x Bayer channel x signal DN` 保存残差分位数表；采样使用弱水平 Gaussian copula，DPS 似然使用相同分位数样本的 1-DN 量化区间 KDE。

产物位于 `runs/noise/static_conditional_residual_candidate_20260929/`。拟合使用帧 `0--39` 估计独立亮度轴，帧 `40--99` 拟合，帧 `120--179` 评价，并按位置奇偶拆分拟合与评价。ROI 是预先固定的静止背景几何区域；不按评价残差大小删除样本。

七个高增益 profile 的最差逐通道结果为：条件方差相对误差 `3.12%`，偏斜误差 `0.0532`，条件经验 CDF `W1/sigma` 为 `1.26%`。实测相邻同色 packed 水平相关为弱相关，约 `0.025--0.069`；模型将其限制在 `[-0.2, 0.2]` 的 copula 参数中。

这证明候选模型在同一记录的静止 ROI、时间/位置留出下优于探索报告中的单高斯边缘模型；它**不**证明跨场景、跨曝光、跨温度的物理传感器标定，也不分离 FPN 与场景纹理。训练前加载器会检查验证 JSON 与每个模型文件的 SHA-256 和 `accepted` 标记。

复现：

```bash
conda run -n dl python scripts/fit_static_conditional_residual.py \
  --company-root ../Data/Company_Dataset/dataset/04_Sigmastar_7_30 \
  --output runs/noise/static_conditional_residual_candidate_20260929 \
  --points 12000
```
