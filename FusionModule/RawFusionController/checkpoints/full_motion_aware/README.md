# full_motion_aware

这里保存正式训练配置 `configs/train_full.yaml` 生成的模型权重和训练日志。

当前内容：

- `best.pt`：验证集 AI 融合 PSNR 最优时保存的 checkpoint。
- `last.pt`：最后一个 epoch 的 checkpoint。
- `config_used.yaml`：本次训练实际使用的配置快照。
- `log.csv`：逐 epoch 训练与验证指标。
- `summary.json`：训练摘要，当前记录的 `best_val_ai_psnr` 为 `42.66389428735681`。

使用示例：

```bash
python scripts/eval.py --ckpt checkpoints/full_motion_aware/best.pt --list data_catalog/val.txt --out_json outputs/metrics/val_eval.json
python scripts/infer_h5.py --ckpt checkpoints/full_motion_aware/best.pt --h5 D:\scene_5\shard_0.h5 --frames 0,1,2
```

这些 `.pt` 文件是二进制模型权重，通常不要手动编辑。
