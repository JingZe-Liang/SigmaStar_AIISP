# checkpoints

训练后的权重、配置快照和日志会保存到这里，例如：

```text
checkpoints/tiny_debug/best.pt
checkpoints/full_motion_aware/best.pt
```

当前子目录：

- `tiny_debug/`：`train_tiny.yaml` 冒烟训练结果。
- `full_motion_aware/`：`train_full.yaml` 正式训练结果。

常见文件含义：

- `best.pt`：验证指标最优 checkpoint。
- `last.pt`：最后一个 epoch checkpoint。
- `log.csv`：逐 epoch 指标。
- `summary.json`：训练摘要。

`.pt` 是二进制权重文件，不建议手动编辑。
