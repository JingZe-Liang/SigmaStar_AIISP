# OpenCV source-match 传统 ISP 工具 v1.0

这个压缩包是从 DatasetMEF 分析工程中独立整理出的 **OpenCV source-match** 方法。它只使用传统图像处理和 DNG 元数据，不包含学习模型，也不依赖 LibRaw/rawpy。

该版本的目标是尽量匹配 DatasetMEF 自带的 RGB/JPEG 色彩与亮度。它复用了项目中已经修正粉色偏色并完成对比验证的处理链。

## 1. 输入数据约定

工具读取 DatasetMEF 已解包的 RAW TIFF 和数值元数据 JSON，不直接解析任意相机的压缩 DNG 文件。

一个场景应保持下面的目录结构：

```text
scene_0245/
  dng_components/
    under/
      scene_0245_metadata_numeric.json
      raw/raw_unprocessed/scene_0245_raw_unprocessed_raw_full_frame.tiff
    normal/
      scene_0245_metadata_numeric.json
      raw/raw_unprocessed/scene_0245_raw_unprocessed_raw_full_frame.tiff
    over/
      scene_0245_metadata_numeric.json
      raw/raw_unprocessed/scene_0245_raw_unprocessed_raw_full_frame.tiff
```

## 2. 环境安装

建议使用 Python 3.10 或更高版本。在工具目录中执行：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. 使用方法

### 处理一个曝光帧

```powershell
.\.venv\Scripts\python.exe run.py frame `
  --scene-dir "D:\zhuo mian\DatasetMEF\scene_0245" `
  --exposure normal `
  --output "output\scene_0245_normal.png"
```

程序会同时生成同名 JSON 报告，记录 CFA、黑白电平、曝光补偿、白点和矩阵插值权重。

### 处理一个场景的欠曝、正常、过曝三帧

```powershell
.\.venv\Scripts\python.exe run.py scene `
  --scene-dir "D:\zhuo mian\DatasetMEF\scene_0245" `
  --output-dir "output\scene_0245"
```

也可以使用：

```powershell
examples\run_scene.bat "D:\zhuo mian\DatasetMEF\scene_0245" "output\scene_0245"
```

### 批量处理整个数据集

```powershell
.\.venv\Scripts\python.exe run.py dataset `
  --dataset "D:\zhuo mian\DatasetMEF" `
  --output-dir "output\all_scenes"
```

只处理指定场景：

```powershell
.\.venv\Scripts\python.exe run.py dataset `
  --dataset "D:\zhuo mian\DatasetMEF" `
  --output-dir "output\selected" `
  --scenes scene_0001 scene_0245 scene_0300
```

快速预览时，可添加 `--max-long-edge 1600`。缩放只发生在 ISP 完成之后，不改变 RAW 处理流程。默认值为 `0`，即输出完整分辨率。

## 固定公平对比模式

原始 frame / scene / dataset 命令用于复现 DatasetMEF 的 source-match RGB，
保留了按 under / normal / over 标签增加不同 EV 的历史策略。它们不应用于
under / fused / over 的公平比较。

请使用 fair-compare。它要求 under、over 的数值元数据在传感器和色彩字段
上一致，并只采用 under 的这一份共享记录处理三张图。fused 不需要自己的 DNG
元数据，输入为 U2Fusion 的 canonical packed NPY [R, Gr, B, Gb]。

示例：

    python run.py fair-compare --under-raw "F:\RAW_MEF\data_RAW\iPhone17Pro\core_data_mef\scene_0431\under\scene_0431_unprocessed_BGGR_16bit.tiff" --over-raw "F:\RAW_MEF\data_RAW\iPhone17Pro\core_data_mef\scene_0431\over\scene_0431_unprocessed_BGGR_16bit.tiff" --fused-packed "F:\RAW_MEF\scripts\U2Fusion-master\u2fusion_raw_emvd_bf16\outputs\raw_mef_fused_bf16_test\scene_0431\fused_packed_float32.npy" --metadata-under "F:\RAW_MEF\data_RAW\iPhone17Pro\core_matadata_mef\scene_0431\under\scene_0431_metadata_numeric.json" --metadata-over "F:\RAW_MEF\data_RAW\iPhone17Pro\core_matadata_mef\scene_0431\over\scene_0431_metadata_numeric.json" --output-dir "output\scene_0431_fair"

默认行为是公平性约束的一部分：

1. 三图共用同一 Bayer、黑白电平、AsShotNeutral、ColorMatrix、
   CameraCalibration、AnalogBalance 和裁剪参数。
2. 三图均使用 0 EV；忽略 BaselineExposure、ExposureTime、ISO 以及原工具
   按曝光标签添加的 EV。
3. 默认关闭高光重建和 iPhone 机型色偏修正，避免每张图依据自己的内容改写
   高光或色彩。
4. 默认不使用伪色抑制；需要时可加 --enable-false-color-suppression，
   该固定滤波器会对三图使用相同参数。
5. 可加 --enable-highlight-recovery 以使用同一套固定通道重建缓解高亮
   偏色；严格比较裁剪差异时不要启用它。
6. 可加 --tone-map reinhard 避免线性 sRGB 在输出前逐通道硬裁剪。该固定
   shoulder 会应用于三图，不依赖曝光或图像统计，但无法恢复传感器已裁剪的
   通道色彩。

输出目录中的 fairness_report.json 会记录全部约束和输入路径。

## 4. ISP 处理链

1. 按 DNG `BlackLevel` 和 `WhiteLevel` 线性归一化 RAW。
2. 应用 `DefaultCropOrigin/DefaultCropSize`，缺失时使用 `ActiveArea`。
3. 根据 `CFAPattern2` 使用 OpenCV edge-aware Bayer 去马赛克。
4. 对 R-G、B-G 色差执行中值滤波，抑制高频伪色。
5. 在相机 RGB 域进行高光通道重建。
6. 使用 DNG 双光源 `ColorMatrix`、`CameraCalibration`、`AnalogBalance` 和 `AsShotNeutral` 求相机 RGB 到线性 sRGB 的变换，并执行 Bradford 白点适配。
7. 对已校准的 iPhone18,1 主摄/长焦应用小幅固定色偏修正；未知机型会跳过这一步。
8. 应用 DNG `BaselineExposure`，并添加 DatasetMEF source-match 曝光补偿：欠曝 `0 EV`、正常 `+1 EV`、过曝 `+2 EV`。
9. 执行标准 sRGB gamma，输出 8-bit RGB PNG/JPEG/TIFF。

## 5. 重要说明

- 这套 source-match 参数是针对 DatasetMEF 的 RGB 目标标定的，不应直接视为所有相机的通用 ISP 参数。
- 默认 PNG 输出没有 JPEG 二次压缩误差，适合后续评价和训练前处理。
- 全分辨率处理需要较多内存；只做人工检查时建议设置 `--max-long-edge`。
- 批处理遇到不完整场景时会记录错误并继续，最终写入 `batch_processing_report.json`；加 `--fail-fast` 可在首个错误处停止。
- 工具不读取源 RGB/JPEG，输出结果只由 RAW TIFF 和 DNG 数值元数据生成。

## 6. 文件说明

```text
run.py                         命令行入口
opencv_source_match/           ISP 实现
requirements.txt               Python 依赖
examples/run_scene.bat         Windows 单场景示例
NOTICE.txt                     使用和第三方依赖说明
VERIFICATION.json              与原工程实现的复现校验结果
```
