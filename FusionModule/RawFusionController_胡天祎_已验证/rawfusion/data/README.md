# data

这里实现 H5 数据读取、样本索引和模型输入特征构造。

主要文件：

- `dataset.py`：定义 `H5FusionDataset`、H5 清单读取、样本索引、数据归一化和 feature 构造。

期望的 H5 数据格式：

```text
noisy : [N, 2, H, W], uint16
2dnr  : [N, H, W], uint16
3dnr  : [N, H, W], uint16
clean : [N, H, W], uint16
```

`feature_mode: strong` 时会构造 7 个输入通道：`prev`、`curr`、`abs(curr-prev)`、`2dnr`、`3dnr`、`abs(2dnr-3dnr)`、`edge(curr)`。

默认按 12-bit RAW 使用，`data_max_value=4095`。若输入数据码值范围不同，需要同步改配置文件。
