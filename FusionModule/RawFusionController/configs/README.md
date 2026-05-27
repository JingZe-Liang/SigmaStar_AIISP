# configs

这里保存 RAW 融合控制器的训练配置文件，供 `scripts/train.py` 读取。

配置文件：

- `train_tiny.yaml`：3 epoch 冒烟训练，用 `train_tiny.txt` / `val_tiny.txt`，主要检查环境、H5 读取、训练和验证流程是否能跑通。
- `train_full.yaml`：正式训练配置，默认保存到 `checkpoints/full_motion_aware`，`model_base=24`，`patch_size=512`。
- `train_stronger.yaml`：更强、更吃显存的配置，`model_base=32`，训练轮数和 oracle 约束更高。

常用命令：

```bash
python scripts/train.py --config configs/train_tiny.yaml
python scripts/train.py --config configs/train_full.yaml
```

显存不足时优先降低 `patch_size`，再考虑降低 `model_base`。修改 bit depth 或数据范围时，要同步检查 `data_max_value` 和 `psnr_max_value`。
