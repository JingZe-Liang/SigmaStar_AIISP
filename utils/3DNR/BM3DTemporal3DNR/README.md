# 算子名称：BM3DTemporal3DNR

## 1. 功能定位：

基于相邻 RAW 帧的时域加权融合和 BM3D 空间去噪，生成单帧 RAW 域 3DNR 结果，并在存在 clean GT 时计算外圈 PSNR / SSIM。

## 2. 输入输出格式：

输入格式：

- 元素类型：单通道 Bayer RAW TIFF 图片序列。
- Shape：单帧 `(H, W)`，要求同一序列分辨率一致。
- Dtype：`uint16` 或 OpenCV/Pillow 可读取的 RAW 灰度格式。
- 是否需要归一化：不需要，脚本内部按 `black_level` / `white_level` 归一化。
- 是否需要 metadata.json：不需要。
- 命名约定：CRVD 风格，如 `frame5_noisy3.tiff`；脚本会在同目录搜索同一 noisy id 的相邻帧。

输出格式：

- 文件：`{input_stem}_bm3d_temporal_3dnr_out.tiff`。
- Shape：与输入帧一致。
- Dtype：与输入帧一致。
- 是否已归一化：否，输出会还原到 RAW 码值范围。
- Effective bits：由 `white_level` 决定，默认 12-bit `0~4095`。
- Container bits：通常为 16-bit TIFF。
- 指标：若同目录存在匹配的 `frame*_clean.tiff`，输出外圈 MSE、PSNR、SSIM；PSNR / SSIM 越高表示与 clean 更接近。

## 3. 方法简述：

脚本先从目标帧向前后收集半径为 `radius` 的 RAW 帧序列，按黑白电平归一化到 `[0, 1]`。随后根据相邻帧差异估计噪声强度和时域权重，对当前帧邻域做加权均值 / 中值融合，再调用 BM3D 做空间域精修。指标计算时只使用外圈区域，避免中心区域的特定物体或遮挡影响整体判断。

## 4. 运行示例：

```powershell
python 3DNR.py `
  "D:\data\CRVD\scene1\ISO25600\frame5_noisy3.tiff" `
  --radius 3 `
  --black-level 0 `
  --white-level 4095 `
  --temporal-sigma-scale 2.5 `
  --mean-weight 0.7 `
  --no-show-fig
```

如需把 RAW 结果转成简化 RGB 预览：

```powershell
python 3DNR_to_rgb.py `
  "D:\data\CRVD\scene1\ISO1600\frame5_noisy3.tiff" `
  --display-bayer-pattern GBRG `
  --black-level 0 `
  --white-level 4095
```

## 5. 参数解析：

- `--radius`：时域搜索半径，`3` 表示最多使用 7 帧。
- `--black-level` / `--white-level`：RAW 归一化使用的黑白电平。
- `--sigma01`：归一化噪声标准差；不填则自动估计。
- `--temporal-sigma-scale`：时域融合容忍度，越大越愿意融合差异较大的邻帧。
- `--mean-weight`：加权均值和中值融合时的均值占比。
- `--display-bayer-pattern`：RGB 预览时使用的 Bayer pattern，可选 `RGGB/BGGR/GBRG/GRBG`。

## 6. 算子特点：

该算子能在静止区域明显降低随机噪声；运动或错位区域依赖邻帧差异权重抑制错误融合，整体更适合作为传统 3DNR baseline。`show/` 中保留了少量展示图和 2DNR / 3DNR 对比材料。
