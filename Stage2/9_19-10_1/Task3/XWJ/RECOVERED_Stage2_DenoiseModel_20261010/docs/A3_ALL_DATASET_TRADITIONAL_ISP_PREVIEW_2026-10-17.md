# A3 全数据集 Traditional ISP 预览报告（2026-10-17）

## 结论

本轮已使用 A3 EMA step-5000 对公司数据集中的 45 条原始采集 RAW 流完成推理，并为每条流生成 Input 与 A3 的横向 ISP 对比视频。Group03 的 2DNR/3DNR 派生数据未纳入本轮任务。

本轮证明了现有 A3 能在所有已登记原始流上完成推理，但不能据此判定跨传感器泛化已经通过。代表性视觉抽查显示：MIS20S1 和 sc635hai 场景通常有明显降噪效果；SC465SL difficult-pattern 高增益场景仍可见明显时域拖影、重影和过度平滑，需要作为后续跨传感器微调与验证重点。

## 冻结协议

- checkpoint：`artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`
- checkpoint SHA-256：`97c4332f685f98959b243e4b100434469f714aed6329e100d22b6d8552592e0b`
- 权重：EMA
- 配置：`configs/a3_all_dataset_traditional_isp_preview_20261017.json`
- 配置 SHA-256：`dee8a5575ac8a166c9eb8ed682a7fffa5a08adaf458c467d364bf4e6fbdd518a`
- GPU：RTX 4090，对应本机 PyTorch `cuda:0`
- tile / stride / tile batch：`64 / 56 / 12`
- 每条流采样：首、中、尾各一个连续八帧片段，共 24 帧
- 数据集目录保持只读；未启动训练；未修改或替换 A3 checkpoint

## ISP 设置

- ISP 根目录：`ISP_traditional`
- Bayer 重建：按项目 canonical `[R, Gr, Gb, B]` 与实际 CFA 显式映射
- 去马赛克：OpenCV edge-aware demosaic
- 高光重建：开启
- tone map：Reinhard
- 2DNR / 3DNR：不使用
- 曝光：根据每条序列 Input 的 99.5 百分位估计一次，并在 Input/A3 间冻结共享
- 白平衡：使用对应流的固定 WB，Input/A3 完全一致
- 色彩模式：`uncalibrated_camera_rgb`

传感器 ColorMatrix 不可用，因此视频用于比较降噪、纹理和时序表现，不构成色彩准确性结论。不同序列使用各自固定曝光，序列之间的绝对亮度不可直接比较。

## 数据覆盖

| 分组 | 类型 | 数量 |
|---|---|---:|
| Group01 | 原始场景 | 4 |
| Group02 | 原始场景 | 2 |
| Group04 | calibration | 18 |
| Group04 | 暗室/户外原始场景 | 11 |
| Group05 | 原始场景 | 10 |
| 合计 |  | 45 |

Group03 的 `denoised.raw`、`fused.raw` 和逐帧派生产物全部排除，因为用户要求忽略 2DNR/3DNR，且这些文件不是原始采集 RAW。

## 输出位置

输出根目录：

`artifacts/a3_all_dataset_traditional_isp_preview_20261017`

关键索引：

- `summary.json`：45 条序列状态及视频绝对路径
- `inventory.json`：45 条源流、传感器、尺寸、CFA、位对齐和 WB 清单
- `protocol.json`：冻结配置、checkpoint 与执行范围
- `sequences/<group>/<category>/<name>/runtime.json`：逐流黑电平、曝光、源文件哈希和采样帧
- `sequences/<group>/<category>/<name>/inference/manifest.json`：预测 provenance 与 SHA-256
- `sequences/<group>/<category>/<name>/traditional_isp_video/video_manifest.json`：ISP、编码命令、视频和预览图 SHA-256

每个视频均为左右对比：左侧 Input，右侧 A3 EMA；上半部分为全景，下半部分为固定中心裁剪。

## 结构核验

- 预测文件：45/45
- 视频 manifest：45/45
- MP4 视频：45/45
- 视频规格：全部 1920x1080、24 帧、6 fps、HEVC、`yuv444p`
- 视频总帧数：1080
- 视频总体积：918,042,365 bytes
- ffprobe 解码/结构异常：0
- manifest 协议异常：0
- 单元测试：`4 passed`
- `git diff --check`：通过
- A3 checkpoint SHA-256：复核一致
- 任务结束后 RTX 4090 显存已释放

当前 FFmpeg 不提供 `libx264` 编码器，因此按既定可记录回退规则使用 `libx265`，视频 manifest 中保存了实际编码器和完整命令。

## 代表性视觉抽查

### MIS20S1 outdoor 1500x

A3 明显压低高增益彩噪，使人物、道路和标定板重新可辨。Reinhard 保留了天空区域的高光层次。仍需通过完整视频检查运动人物周围的拖影和纹理稳定性。

### sc635hai 16x motion color

A3 对静态背景和人物区域有明显降噪，主要物体边缘仍可辨，说明模型在该跨传感器场景上具有一定零样本有效性。细纹理存在平滑，不能仅凭首帧判定动态质量通过。

### SC465SL difficult-pattern 8880

A3 虽显著降低随机彩噪，但画面出现明显时域拖影、重影和纹理涂抹。该场景是当前跨传感器泛化的明确失败样例，后续微调与验证必须单独覆盖，不能以降噪强度作为通过依据。

## 未完成项

本轮“对全部原始数据生成 A3 + Traditional ISP 视频”的任务已经完成，无缺失序列。

项目总体目标仍有以下未完成项：

1. 由人工逐条观看 45 个视频，按传感器和增益记录噪声抑制、细节、拖影、闪烁和色彩异常。
2. 基于本轮失败模式确定跨传感器训练/验证划分，尤其保留 SC465SL difficult-pattern 作为严格 holdout 或专项验证集。
3. 在不把旧教师输出当作干净动态真值的前提下，完成其他传感器微调方案及独立验证。
4. 每完成一个传感器微调，继续输出同一 ISP 协议的视频，才能与本轮 A3 零样本基线公平比较。
