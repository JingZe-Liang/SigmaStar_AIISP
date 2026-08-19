# 2DNR-prior Lite U-Net Fusion

本工程用于对对齐的 Bayer RAW 视频进行轻量 U-Net 融合。它不让网络直接生成 RAW，而是以 `denoised.raw`（2DNR）为不可替换的基线，只学习每个 Bayer 采样位置可使用多少 3DNR 修正。

## 设计约束

输入按 RGGB physical CFA 位置打包为 `4 x 540 x 960`。网络输入共 14 通道：2DNR 4 通道、3DNR 4 通道、原始 RAW 4 通道、运动置信度 1 通道、平坦度 1 通道。

网络是三次下采样的轻量 U-Net：`14 -> 24 -> 48 -> 64 -> 80 -> 64 -> 48 -> 24 -> 4`。它输出 `beta in [0, 1]`，最终 RAW 严格使用：

```text
output = denoised + beta_eff * (fused - denoised)
beta_eff = 0.35 * beta * (1 - motion) * flatness
```

因此，3DNR 在任何位置最多只贡献 35%；运动越明显，修正自动趋近 0；原始未降噪 RAW 仅参与运动/平坦度估计，不会回灌进输出。

## 伪标签与损失

当前数据没有 clean ground truth。训练目标是传统规则产生的高置信 3DNR 修正：

```text
teacher = 0.35 * (1 - motion) * flatness * agreement(2DNR, 3DNR)
loss = SmoothL1(beta_eff, teacher) + 0.10 * mean(beta * motion) + 0.01 * TV(beta)
```

这会训练网络学习更连续的局部上下文门控，但不构成“无真值必然优于传统法”的证明。要评价真实提升，必须用未参与训练的视频和 clean/长曝光参考进行测试。

## 数据格式

- 原始 RAW：little-endian `uint16` 容器，12-bit 有效数据左对齐，使用 `value / 16` 解码；black level 为约 252。
- `denoised.raw` 与 `fused.raw`：little-endian `uint16` 容器，数值直接为 12-bit code，范围 0 到 4095；black level 为约 300。
- 三段视频必须是同样的 `1920 x 1080`、相同帧数、相同 Bayer 对齐。

## 训练

前提：Windows CMD 中已可使用 Conda `base` 环境，且当前目录为项目根目录。

```cmd
cd /d C:\Users\HP\Desktop\lite_unet_fusion
set KMP_DUPLICATE_LIB_OK=TRUE
conda run -n base --no-capture-output python train.py ^
  --source "D:\BaiduNetdiskDownload\company\shdarkroom\645x\raw_stream_1920x1080_16bit@RG_[Shutter=79999,SenserG=131072,IspG=5167,R=2120,G=1024,B=1956].raw" ^
  --two-d "D:\BaiduNetdiskDownload\company\denoise\mis20s1_2D&3D\raw_stream_1920x1080_16bit@RG_R=2120,G=1024,B=1956_645x\denoised.raw" ^
  --three-d "D:\BaiduNetdiskDownload\company\denoise\mis20s1_2D&3D\raw_stream_1920x1080_16bit@RG_R=2120,G=1024,B=1956_645x\fused.raw" ^
  --output-dir output\645x_unet ^
  --epochs 40 --samples-per-epoch 1200 --patch-size 128 --batch-size 4
```

CPU 可以运行但会较慢。安装 CUDA 版 PyTorch 后，添加 `--device cuda`。

### AutoDL / Linux GPU 推荐配置

前提：Conda `base` 已有可用的 CUDA PyTorch，且目录为 `/root/autodl-tmp/lyd-8.17`。4090 与 16 核 CPU 推荐使用 AMP、TF32、32 的 batch 和 12 个数据加载 worker；这些 CUDA 优化由训练脚本自动启用。

```bash
cd /root/autodl-tmp/lyd-8.17
conda run -n base --no-capture-output python train.py \
  --source "/root/autodl-tmp/lyd-8.17/data/source.raw" \
  --two-d "/root/autodl-tmp/lyd-8.17/data/denoised.raw" \
  --three-d "/root/autodl-tmp/lyd-8.17/data/fused.raw" \
  --output-dir "/root/autodl-tmp/lyd-8.17/output/645x_unet" \
  --epochs 40 --samples-per-epoch 1200 --patch-size 128 \
  --batch-size 32 --num-workers 12 --device cuda
```

每轮完成后会写入 `last.pt`，其中包含模型、AdamW、学习率调度器、AMP 和训练历史。进程中断后，用相同的训练参数加上 `--resume` 从下一轮恢复：

```bash
conda run -n base --no-capture-output python train.py \
  --source "/root/autodl-tmp/lyd-8.17/data/source.raw" \
  --two-d "/root/autodl-tmp/lyd-8.17/data/denoised.raw" \
  --three-d "/root/autodl-tmp/lyd-8.17/data/fused.raw" \
  --output-dir "/root/autodl-tmp/lyd-8.17/output/645x_unet" \
  --epochs 40 --samples-per-epoch 1200 --patch-size 128 \
  --batch-size 32 --num-workers 12 --device cuda \
  --resume "/root/autodl-tmp/lyd-8.17/output/645x_unet/last.pt"
```

`best.pt` 始终是验证损失最低的模型，可直接交给推理脚本；`last.pt` 只用于恢复训练。若 32 的 batch 显存不足，先改为 16；若 worker 报错或内存不足，改为 8。

旧版训练脚本仅保存模型权重的 `best.pt`，不能精确恢复优化器状态。要在新版中利用该权重，以 `--init-from /path/to/best.pt` 开始新的训练过程；新版的 `last.pt` 才使用 `--resume`。

## 推理

前提：训练已生成 `output\645x_unet\best.pt`。

```cmd
cd /d C:\Users\HP\Desktop\lite_unet_fusion
set KMP_DUPLICATE_LIB_OK=TRUE
conda run -n base --no-capture-output python infer.py ^
  --source "D:\BaiduNetdiskDownload\company\shdarkroom\645x\raw_stream_1920x1080_16bit@RG_[Shutter=79999,SenserG=131072,IspG=5167,R=2120,G=1024,B=1956].raw" ^
  --two-d "D:\BaiduNetdiskDownload\company\denoise\mis20s1_2D&3D\raw_stream_1920x1080_16bit@RG_R=2120,G=1024,B=1956_645x\denoised.raw" ^
  --three-d "D:\BaiduNetdiskDownload\company\denoise\mis20s1_2D&3D\raw_stream_1920x1080_16bit@RG_R=2120,G=1024,B=1956_645x\fused.raw" ^
  --checkpoint output\645x_unet\best.pt ^
  --output output\645x_unet\unet_fused.raw
```

推理会生成 12-bit 直接码值的 `unet_fused.raw` 和同名 JSON。可将 RAW 用现有 OpenCV ISP 的 `shdarkroom-frame`/`shdarkroom-sequence` 流程预览。所有命令均要求 Windows CMD 和 Conda `base` 环境。
