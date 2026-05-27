# losses

这里保存融合模型训练用的损失函数和评估指标。

主要文件：

- `fusion_losses.py`：定义 Charbonnier 重建损失、梯度损失、alpha TV 约束、运动先验约束和 oracle alpha 软目标约束。
- `metrics.py`：定义 SSE 累计、PSNR、SSIM 和指标汇总工具。

训练总损失大致由以下部分组成：

```text
rec + lam_grad * grad + lam_tv * tv + lam_motion * motion + lam_oracle * oracle
```

其中 `oracle_alpha3d_target` 用 clean GT 判断 2DNR 和 3DNR 哪个更接近，`motion_alpha3d_target` 则鼓励静止区域更多用 3DNR、运动区域降低 3DNR 权重。
