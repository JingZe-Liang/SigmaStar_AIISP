# utils

这里放工程运行时的辅助工具。

主要文件：

- `h5_utils.py`：检查 H5 文件结构、shape 和 dtype，供 `tools/check_h5.py` 调用。
- `seed.py`：统一设置 Python、NumPy、PyTorch 的随机种子。
- `visualize.py`：把 RAW / alpha / 对比结果保存为 PNG、PGM 或网格图。

推理脚本 `scripts/infer_h5.py` 会调用这里的可视化函数，生成 `frame_xxxx_fused.png`、`frame_xxxx_alpha3d.png` 和 `frame_xxxx_compare.png`。
