# models

这里保存 AI 融合控制器的模型结构。

主要文件：

- `fusion_net.py`：定义 `MotionAwareFusionUNet`、`fuse_alpha3d` 和参数量统计函数。

`MotionAwareFusionUNet` 是轻量 U-Net 结构，包含 residual block、GroupNorm 和 SEGate。模型输出单通道 `alpha_3d`，范围 `[0, 1]`。

融合公式：

```text
fused = alpha_3d * 3DNR + (1 - alpha_3d) * 2DNR
```

初始化时 `init_alpha3d` 默认是 `0.80`，表示模型一开始略偏向 3DNR，后续由数据学习逐像素调整。
