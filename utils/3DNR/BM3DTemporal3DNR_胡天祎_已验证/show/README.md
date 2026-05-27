# show

这里存放 3DNR 实验的展示产物，主要用于快速查看不同 ISO 和不同显示域下的去噪效果。

当前内容：

- `scene1_ISO12800_frame5.png`、`scene1_ISO1600_frame5.png`：RAW 或灰度显示域的样例结果。
- `scene1_ISO1600_frame5_RGB.png`：RGB 显示域的样例结果。
- `show(new)/`：更新后的结果整理目录，包含量化指标和多 scene RGB 展示。

这些文件是实验结果快照，不建议作为训练数据入口。若重新运行 `3DNR.py` 或 `3DNR_to_rgb.py`，新的输出通常会写在原始 TIFF 所在目录，需要手动挑选后放到这里归档。
