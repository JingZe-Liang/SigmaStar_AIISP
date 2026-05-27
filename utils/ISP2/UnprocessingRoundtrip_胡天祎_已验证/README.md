# 算子名称：UnprocessingRoundtrip

## 1. 功能定位：

验证 sRGB 反处理到 RAW Bayer 4 通道，再经过简化 ISP 回到 sRGB 的往复流程，用于检查 unprocessing 与基础 ISP 显示链路是否能跑通。

## 2. 输入输出格式：

输入格式：

- 元素类型：sRGB 图片。
- Shape：`(H, W, 3)`。
- Dtype：PIL 可读取的 8-bit RGB 图片。
- 是否需要归一化：脚本内部转为张量并归一化。
- 是否需要 metadata.json：不需要；metadata 由 `third_party.unprocessing_torch` 的 unprocess 函数生成。
- 当前示例路径：`images/uestc.jpg`，运行前需要替换为本机存在的图片或补充同名样例。

输出格式：

- `unprocessing_to_raw.py`：打印 metadata、噪声参数和 Bayer 通道统计。
- `test_unprocess_isp_roundtrip.py`：输出 MSE / PSNR，并保存 `/tmp/unprocess_isp_roundtrip.png` 预览图。
- RAW 中间结果：Bayer 4 通道张量，形状近似 `(4, H/2, W/2)`。
- RGB 回转结果：`(H, W, 3)`，`float32` 范围 `[0, 1]`。

## 3. 方法简述：

脚本调用 `third_party.unprocessing_torch.dataloader.unprocess` 将 sRGB 反处理为 RAW-like Bayer 表示，同时得到白平衡、颜色矩阵、gamma 等 metadata。随后用简化 demosaic、白平衡、颜色校正和 gamma 流程把 RAW 回转到 sRGB，并计算原图与回转图之间的 MSE / PSNR。

## 4. 运行示例：

```powershell
python unprocessing_to_raw.py
python test_unprocess_isp_roundtrip.py
```

## 5. 参数解析：

当前脚本没有命令行参数，主要需要修改或准备：

- `img_path`：输入 sRGB 图片路径，默认 `images/uestc.jpg`。
- `third_party.unprocessing_torch`：需要在当前 Python 环境的 import path 中可用。
- `/tmp/unprocess_isp_roundtrip.png`：往复流程可视化输出路径。

## 6. 算子特点：

该验证脚本适合检查 sRGB-to-RAW 的数据准备逻辑，但不是严格可逆 ISP。误差主要来自 demosaic、gamma、颜色矩阵近似和简化 ISP 流程。
