# 算子名称

Green-Guided Joint Bilateral Bayer 2DNR  
中文名：绿通道引导的 Bayer RAW 联合双边滤波 2D 降噪

## 1. 功能定位

该算子在 Bayer RAW 域直接进行 2D 空域降噪，输入和输出都保持为单通道 Bayer RAW，不做 demosaic，也不转换到 RGB/YUV。

## 2. 输入输出格式

### 输入格式

元素类型：`.raw` 文件或 `.safetensors` 张量  
Shape：单帧为 `H * W` 的 2D Bayer RAW，宽高必须为偶数  
Dtype：支持 `uint8`、`uint16`、`float32`  
是否需要归一化：不需要，脚本内部会根据 `bit_depth` 归一化到 `[0, 1]`  
是否需要 metadata.json：不需要  
说明：输入必须是未 demosaic 的 Bayer mosaic 数据，支持 `rggb / bggr / grbg / gbrg` 四种 Bayer pattern。

目录输入说明：

```text
目录输入只支持 .safetensors 文件。
脚本会递归查找输入目录下的 .safetensors，并保持相对路径写入输出目录。
```

### 输出格式

元素类型：`.raw` 文件或 `.safetensors` 张量  
Shape：与输入完全一致，仍为 `H * W`  
Dtype：与输入 dtype 保持一致  
是否已归一化：否，输出会转换回输入 RAW 的数值范围  
Effective bits：由 `--bit-depth` 指定，默认 12 bit  
Container bits：由输入 dtype 决定，例如 12 bit RAW 常存放在 `uint16` 容器中  
说明：输出仍是单通道 Bayer RAW，可继续送入后续 ISP、3DNR 或其他 RAW 域处理模块。

## 3. 方法简述

1. 将输入 RAW 按 `bit_depth` 归一化到 `[0, 1]`。
2. 根据 Bayer pattern 找到 R、G、B 在 mosaic 中的位置。
3. 使用 5x5 插值核估计完整的 green guide 图。
4. 对 G 通道使用自身作为 guide 做 joint bilateral filtering。
5. 对 R/B 通道，使用对应位置的 green estimate 作为 guide 做 joint bilateral filtering。
6. 双边滤波权重由空间距离和 guide 强度差共同决定：

```text
w = exp(-d_space^2 / (2 * sigma_s^2)) * exp(-d_guide^2 / (2 * sigma_r^2))
```

7. 将滤波后的 R、G、B 重新写回 Bayer mosaic，保持 RAW 单通道格式。
8. 将结果裁剪到 `[0, 1]` 后恢复到原始 bit depth 和 dtype。

## 4. 运行示例

处理单个 `.raw` 文件：

```powershell
python joint_bilateral_肖纬杰/green_guided_joint_bilateral_bayer_2dnr.py `
  --input Infinite-ISP/in_frames/normal/ColorChecker_2592x1536_12bits_RGGB.raw `
  --output output/ColorChecker_2dnr.raw `
  --input-format raw `
  --width 2592 `
  --height 1536 `
  --dtype uint16 `
  --bit-depth 12 `
  --bayer-pattern RGGB `
  --filter-window 9 `
  --overwrite
```

处理 `.safetensors` 文件夹：

```powershell
python joint_bilateral_肖纬杰/green_guided_joint_bilateral_bayer_2dnr.py `
  --input "black scene_iso5184" `
  --output "black_scene_iso5184_2dnr" `
  --input-format safetensors `
  --tensor-key raw `
  --bit-depth 12 `
  --bayer-pattern GBRG `
  --filter-window 9 `
  --overwrite
```

只测试前 N 个 safetensors 文件：

```powershell
python joint_bilateral_肖纬杰/green_guided_joint_bilateral_bayer_2dnr.py `
  --input "black scene_iso5184" `
  --output "black_scene_iso5184_2dnr_test" `
  --input-format safetensors `
  --tensor-key raw `
  --bit-depth 12 `
  --bayer-pattern GBRG `
  --limit 1 `
  --overwrite
```

## 5. 参数解析

`--input`：输入 `.raw` / `.safetensors` 文件，或 `.safetensors` 文件夹。  
`--output`：输出文件或输出目录。  
`--input-format`：输入格式，支持 `auto / raw / safetensors`。目录输入只支持 `safetensors`。  
`--tensor-key`：读取和写入 safetensors 时使用的 tensor key，默认 `raw`。  
`--width`：RAW 宽度，处理 `.raw` 文件时必须提供。  
`--height`：RAW 高度，处理 `.raw` 文件时必须提供。  
`--bit-depth`：RAW 有效位深，默认 12。  
`--dtype`：`.raw` 文件的存储 dtype，支持 `uint8 / uint16 / float32`。  
`--bayer-pattern`：Bayer 排列方式，支持 `rggb / bggr / grbg / gbrg`。  
`--filter-window`：G 通道联合双边滤波窗口大小；R/B 使用约一半窗口在同色子采样平面滤波。  
`--r-std-dev-s`：R 通道空间权重 sigma。  
`--r-std-dev-r`：R 通道 range/guide 权重 sigma。  
`--g-std-dev-s`：G 通道空间权重 sigma。  
`--g-std-dev-r`：G 通道 range/guide 权重 sigma。  
`--b-std-dev-s`：B 通道空间权重 sigma。  
`--b-std-dev-r`：B 通道 range/guide 权重 sigma。  
`--overwrite`：允许覆盖已存在的输出文件。  
`--limit`：目录输入时只处理前 N 个 `.safetensors` 文件，便于快速测试。  
`--self-test`：运行脚本内置自检，验证输出 shape、dtype 和数值范围。

## 6. 算子特点

1. 直接在 Bayer RAW 域降噪，不引入 demosaic 伪影。
2. 使用 green guide 引导 R/B 降噪，尽量保留边缘结构。
3. R、G、B 分通道设置空间 sigma 和 range sigma，便于控制不同颜色平面的平滑强度。
4. 输出 shape、dtype、Bayer pattern 不变，方便接入后续 RAW pipeline。
5. 适合做轻量 2DNR，但只利用单帧空间信息，不能解决时域随机噪声和运动拖影问题。
