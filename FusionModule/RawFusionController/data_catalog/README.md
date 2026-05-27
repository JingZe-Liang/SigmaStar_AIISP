# data_catalog

这里存放由 `tools/make_splits.py` 生成的训练/验证/测试 H5 清单。

推荐先运行：

```bash
python tools/make_splits.py --data_root D:\ --out_dir data_catalog
```

生成：

- `train_tiny.txt`, `val_tiny.txt`：快速冒烟。
- `train.txt`, `val.txt`, `test.txt`：正式训练/验证/测试。

默认划分：

- train: scene_1, scene_2, scene_4, scene_6, scene_7
- val: scene_3, scene_8
- test: scene_5, scene_9
