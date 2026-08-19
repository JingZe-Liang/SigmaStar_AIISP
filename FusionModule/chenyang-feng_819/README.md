# 局部对齐 RAW 时域降噪

本目录是 SigmaStar 645x/128x 原型的最终复现交付。模型以当前 2DNR 为安全基线，在 Bayer `RGGB` 四平面内对齐历史 source RAW 与 2DNR，只在高置信区域预测有界时域残差。3DNR 只用于视频对照，不进入模型输入、监督或选择指标。

## 当前结论

645x 上已完成一次 `128x128` Bayer-plane patch、1000 步正式训练，使用默认 `2e-5` 选择门槛。结果保存在 `results/645x_formal_1000_patch128.json`：`selection_proxy_improvement=0.0`、`accepted_against_2dnr=false`，因此正式输出严格回退到当前 2DNR。融合 RAW 与公司 `2dnr.raw` 的 SHA-256 完全一致，不能宣称画质优于 2DNR。

`results/645x_formal_1000_patch128_ai.mp4` 是唯一保留的正式 local-align AI 视频：200 帧、30 fps、1920x1080、固定 RGGB ISP。AI 输出逐值等于 2DNR，因此该视频用于验证完整训练、推理回退和固定 ISP 渲染链路，不是画质提升证明。

## 数据契约

公司数据根目录：

```text
/HardDisk/jingzeliang/Data/SigmaStar_project/given_dataset/
```

正式训练仅使用以下两条有配套 2DNR/3DNR 的 200 帧流：

| source | 2DNR/3DNR | 说明 |
| --- | --- | --- |
| `Sigmastar_7_30/shdarkroom/645x/...R=2120,G=1024,B=1956].raw` | `mis20s1_2D&3D/...R=2120,G=1024,B=1956_645x/{2dnr,3dnr}.raw` | 645x 配对流 |
| `Sigmastar_7_30/shdarkroom/128x/...R=2159,G=1024,B=1849].raw` | `mis20s1_2D&3D/...R=2159,G=1024,B=1849_128x/{2dnr,3dnr}.raw` | 128x 配对流 |

source 是左对齐 12-bit 的 16-bit container，读取时除以 `16`，black level 为 `252`。2DNR/3DNR 是直接存储的 12-bit code，black level 为 `300`。CFA 固定为 `RGGB`。

数据目录另外有 9 条未配对 source 序列和多 ISO calibration 流。它们可用于 source-only 预训练或泛化评估，但没有同序公司 2DNR 时不能直接接入本目录的残差训练。

## 网络与保护

输入为 26 通道：当前 source/2DNR 8 通道，局部对齐的 `t-1/t-2` source/2DNR 16 通道，两个对齐置信度图 2 通道。双分支 U-Net 分别编码当前与历史特征；历史特征在每一尺度都由置信度抑制。

```text
output = clamp(2DNR + confidence * sigmoid(gate) * 0.25 * tanh(residual), 0, 1)
```

残差头和 gate 头零初始化。训练、选择和报告的中心帧分别是 `2..119`、`135..155`、`169..198`，完整上下文互不重叠。若选择集改善未严格超过 `2e-5`，推理逐 uint16 值复制 2DNR，避免黑电平以下的合法码值被改写。

## 文件

| 文件 | 用途 |
| --- | --- |
| `raw_fusion_local_align.py` | 当前局部对齐 U-Net 的训练和推理入口；使用 `--local_*` 参数。 |
| `render_local_align_comparison.py` | 固定 RGGB ISP 渲染；`--layout ai` 只输出 local-align AI。 |
| `tests/test_local_align_unet.py` | 26 通道、时间切分、未来帧隔离和精确回退测试。 |
| `results/` | 真实 645x 正式 JSON、全帧原尺寸视频及历史诊断对照。 |

## 运行

从本目录运行。Python 环境使用 `/home/jingzeliang/miniconda3/envs/aaa_312/bin/python`。CUDA 训练前先通过 `nvidia-smi` 选择空闲 GPU。

```bash
DATA_ROOT=/HardDisk/jingzeliang/Data/SigmaStar_project/given_dataset
SOURCE="$DATA_ROOT/Sigmastar_7_30/shdarkroom/645x/raw_stream_1920x1080_16bit@RG_[Shutter=79999,SenserG=131072,IspG=5167,R=2120,G=1024,B=1956].raw"
TWO_DNR="$DATA_ROOT/mis20s1_2D&3D/raw_stream_1920x1080_16bit@RG_R=2120,G=1024,B=1956_645x/2dnr.raw"

CUDA_VISIBLE_DEVICES=1 /home/jingzeliang/miniconda3/envs/aaa_312/bin/python raw_fusion_local_align.py \
  --local_align_source "$SOURCE" --local_align_2dnr "$TWO_DNR" \
  --local_align_out runs/645x_formal_1000_patch128 \
  --local_frames 200 --local_height 1080 --local_width 1920 \
  --local_source_black 252 --local_2dnr_black 300 \
  --local_container_scale 16 --local_clip 4095 --local_cfa RGGB \
  --local_stride 2 --local_patch 128 --local_steps 1000 --local_base_ch 16 \
  --local_device cuda --local_eval_frames 8 --local_eval_patches 8 \
  --local_validate_every 20
```

运行测试：

```bash
/home/jingzeliang/miniconda3/envs/aaa_312/bin/python -m pytest -q tests/test_local_align_unet.py
```

视频脚本依赖当前工程根目录已有的 `opencv_source_match/` 和 `sigmastar_inference/` 固定 ISP 模块；上传到独立 GitHub 仓库时，需要一并带入这两个依赖，或改为项目自己的 ISP 实现。
