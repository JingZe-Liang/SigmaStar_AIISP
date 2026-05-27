# rawfusion

这是 RAW 2DNR / 3DNR AI 融合控制器的 Python 包。

主要模块：

- `data/`：读取 H5 数据、构造训练样本和输入特征。
- `models/`：定义 `MotionAwareFusionUNet` 和 `fuse_alpha3d` 融合公式。
- `losses/`：训练损失、PSNR / SSIM 等指标。
- `utils/`：H5 检查、随机种子和可视化导出工具。
- `engine.py`：训练一个 epoch、验证模型和汇总指标的循环逻辑。

包内核心约定是 `alpha_3d=1` 表示完全偏向 3DNR，`alpha_3d=0` 表示完全偏向 2DNR。这个方向和早期 baseline 中的 `w * 2DNR + (1-w) * 3DNR` 不同，阅读旧代码或汇报材料时要特别注意。
