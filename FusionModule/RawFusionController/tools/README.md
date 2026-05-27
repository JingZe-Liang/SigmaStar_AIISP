# tools

这里放数据准备、格式检查和传统 baseline 评估工具，不直接训练模型。

脚本说明：

- `check_h5.py`：检查一个 H5 文件是否包含 `noisy`、`2dnr`、`3dnr`、`clean` 等必要数据集。
- `make_splits.py`：按 scene 生成 `train.txt`、`val.txt`、`test.txt` 以及 tiny 版本清单。
- `baseline_eval.py`：不加载 AI 模型，直接评估 2DNR、3DNR、50/50 平均融合和 pixel-wise oracle 上限。

推荐流程：

```bash
python tools/check_h5.py --h5 D:\scene_1\shard_0.h5
python tools/make_splits.py --data_root D:\ --out_dir data_catalog
python tools/baseline_eval.py --list data_catalog/val.txt --out_json outputs/metrics/baseline_val.json
```

默认划分避免把同背景 scene 简单随机混合，以减少验证指标虚高。
