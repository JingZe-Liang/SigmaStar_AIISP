# scripts

这里放工程的主要命令行入口。

脚本说明：

- `train.py`：按 YAML 配置训练 `MotionAwareFusionUNet`，输出 `best.pt`、`last.pt`、`log.csv`、`summary.json`。
- `eval.py`：加载训练好的 checkpoint，对 H5 清单做验证或测试，输出指标 JSON。
- `infer_h5.py`：对单个 H5 shard 推理，导出 fused PNG、alpha 权重图、对比图和 `manifest.json`。

常用命令：

```bash
python scripts/train.py --config configs/train_tiny.yaml
python scripts/eval.py --ckpt checkpoints/full_motion_aware/best.pt --list data_catalog/test.txt --out_json outputs/metrics/test_eval.json
python scripts/infer_h5.py --ckpt checkpoints/full_motion_aware/best.pt --h5 D:\scene_5\shard_0.h5 --frames 0,1,2 --out_dir outputs/infer_scene5_shard0
```

Windows 下也可以使用工程根目录中的 `run_train_tiny.bat`、`run_eval_test.bat` 和 `run_make_splits.bat`。
