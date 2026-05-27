# tiny_debug

这里保存 `configs/train_tiny.yaml` 冒烟训练生成的模型权重和日志。

当前内容：

- `best.pt`：tiny 验证集上最优的 checkpoint。
- `last.pt`：tiny 训练最后一个 epoch 的 checkpoint。
- `config_used.yaml`：冒烟训练实际配置。
- `log.csv`：3 epoch 的训练和验证记录。
- `summary.json`：训练摘要，当前记录的 `best_val_ai_psnr` 为 `41.935950382167135`。

这个目录主要用于确认环境、数据读取、loss 和指标流水线正常，不代表最终模型性能。正式结果优先看 `checkpoints/full_motion_aware/`。
