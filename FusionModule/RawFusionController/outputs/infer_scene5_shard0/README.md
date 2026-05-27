# infer_scene5_shard0

这里保存对 `scene5` 的 `shard_0.h5` 做推理后的可视化样例。

每个帧通常有三类输出：

- `frame_0000_fused.png`：AI 融合后的 RAW 灰度显示图。
- `frame_0000_alpha3d.png`：逐像素 `alpha_3d` 权重图，越亮越偏向 3DNR，越暗越偏向 2DNR。
- `frame_0000_compare.png`：noisy、2DNR、3DNR、AI fused、alpha 和 clean 的横向对比图。

`manifest.json` 记录了导出的帧号、文件名和 alpha 统计量。当前样例包含 frame 0、1、2，`alpha_mean` 大约在 `0.65` 左右。

重新生成示例：

```bash
python scripts/infer_h5.py --ckpt checkpoints/full_motion_aware/best.pt --h5 D:\scene_5\shard_0.h5 --frames 0,1,2 --out_dir outputs/infer_scene5_shard0
```
