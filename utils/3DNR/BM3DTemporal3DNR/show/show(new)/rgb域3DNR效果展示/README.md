# rgb域3DNR效果展示

这个目录保存多个 scene / ISO 的 3DNR RGB 域效果图，用于肉眼观察去噪、拖影、颜色和细节保留情况。

文件名形如：

- `scene1_ISO12800_frame4_rgb.png`
- `scene2_ISO25600_frame4_rgb.png`
- `scene6_ISO25600_frame4_rgb.png`

这些图片通常由 `3DNR_to_rgb.py` 的 RAW 显示流程生成。若出现明显偏色，优先检查脚本里的 `display_bayer_pattern`，可在 `BGGR`、`GBRG`、`GRBG`、`RGGB` 之间切换。
