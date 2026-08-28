# 2DNR-Prior ST-Mamba Fusion

这个分支用于当前的 645x RAW 序列，不依赖 H5 或 clean GT。输入为同一帧序列的三路结果：

- 源 RAW：例如 128x 数据的 `raw_stream_1920x1080_16bit@RG_[Shutter=54992,SenserG=131072,IspG=1024,R=2159,G=1024,B=1849].raw`，16-bit 容器中的左对齐 12-bit 码值；黑电平为 252，读取时先除以 `source_container_scale=16`。
- 2DNR：`denoised/out_XXXX.raw` 或一个拼接 RAW 流；直接 12-bit 码值，黑电平为 300。
- 3DNR：`fused/out_XXXX.raw` 或一个拼接 RAW 流；直接 12-bit 码值，黑电平为 300。

默认 CFA 为 RGGB，packed 平面顺序固定为 `[R, G1, G2, B]`。三路输入必须是 1920x1080、帧数相同且逐帧对齐。

当前 128x 数据已经核对为三路各 200 帧：source 数值为左对齐 12-bit，`denoised.raw` 是 2DNR 基线，`fused.raw` 是 3DNR 候选。网络在静态平坦区域从 fused 中有限恢复 2DNR 过度平滑掉的局部变化，在运动区域则通过时域可靠度门控和最终硬约束回退至 denoised。

## 融合约束

网络不直接生成像素，输出始终满足：

```text
out = dnr2 + w3 * (dnr3 - dnr2)
w3  = Wmax * sigmoid(beta) * (1 - motion) * flatness * agreement
```

其中 `Wmax=0.35`，`beta` 是 `[R, G1, G2, B]` 四个独立平面的预测图，末层四个 bias 均初始化为 `-4`。因此训练刚开始时输出几乎就是 2DNR；3DNR 只能在低运动、平坦且两路结果相近的位置小幅介入。运动掩码、`flatness` 和 `agreement` 也分别在四个物理 Bayer plane 上计算，不会在计算先验时混合不同颜色采样。

Mamba 主干能看到 `prev source`、`curr source`、`2DNR`、`3DNR`、残差 `3DNR-2DNR` 与三种四平面显式先验。每个 ST-Mamba block 都以 `1-max(motion[R,G1,G2,B])` 作为显式时域可靠度：高运动位置会以当前帧局部特征替代前帧扫描输入，并在 block 输出处回退到当前帧特征。与此同时，motion feature 经全局池化和小型 MLP 生成每个 crop 的 scan-direction 偏置，用于动态调整当前 `scan_path_mode` 下各路径的融合权重；它不新增 Mamba scan。训练没有伪造 clean 标签，而是将上述保守规则作为教师，并同时使用：输出到教师的 Smooth-L1、运动区 3DNR 泄漏惩罚、边缘感知 TV 与微小均值校正。训练日志和推理 JSON 都会分别记录 R、G1、G2、B 的平均 3DNR 修正权重。它不能替代带 clean GT 的监督训练，但不会将有时域拖影风险的 3DNR 误学成主输出。

四平面先验版本的 `motion_stem` 输入通道数与早期单 mask 版本不同，因此旧 checkpoint 不能直接加载；请使用本版本重新训练产生新的 `best.pt`。

`--disable-block-motion-gate` 与 `--disable-dynamic-direction-gate` 可分别关闭两项新机制，方便进行消融对比。默认均启用。

## Mamba-1 And Mamba-2 Backends

默认 `mamba_variant=mamba1` 是项目内可移植的 selective-scan 实现，适合 CPU 调试与兼容已有 checkpoint。旧 8 路 2DNR-prior 模型约有 109,416 个可训练参数；第一阶段 `temporal4` 模式约有 67,640 个参数，第二阶段 `temporal4_grouped` 与第三阶段默认的 `multiscale_grouped` 模式约有 31,016 个参数。第三阶段将全分辨率局部深度可分离卷积与 1/2 空间分辨率的四路分组 Mamba 结合，Mamba 只处理约 1/4 的空间 token，同时保留局部细节残差。

`mamba_variant=mamba2` 将每条 1D 路径替换成官方 Mamba-2 SSD mixer，并按照 `scan_path_mode` 保留四路或八路扫描、motion-gate 和 2DNR 硬融合约束。它需要 Linux、NVIDIA CUDA 和包含 `Mamba2` 的官方 `mamba-ssm` 安装；没有该依赖或在 CPU 上会直接报错，不会产生不可控的回退。官方实现将 Mamba-2 的 SSD 核心定位为比 Mamba-1 更高效的长序列计算，但这不构成对本 RAW 数据画质提升的保证，必须以同一验证区间进行消融比较。

### Scan path stages

当前 2DNR-prior 训练入口默认使用 `scan_path_mode=multiscale_grouped`。第三阶段在每个 ST-Mamba block 中保留全分辨率局部 DWConv，并将分组 Mamba 全局扫描放到 1/2 空间分辨率；上采样后的全局结果通过局部细节残差回注，减少墙面和细纹理的奶油质感。若要复现实验第二阶段，可显式使用 `scan_path_mode=temporal4_grouped`。每个 ST-Mamba block 的时间优先路径为
`T-H-W` 和 `T-W-H` 两种路径的正向/反向扫描，共 4 条路径。原来的
`H-W-T`、`W-H-T` 路径被去掉，避免在 `T=2` 的双帧输入上重复执行完整长序列扫描。
该模式保持双向和横/纵空间覆盖；在分组模式下，四个方向分别处理一个 1/4 通道组，再通过末端 `1×1 Conv3d` 做跨方向通道混合。参数量与长序列扫描计算量都低于 8 路模式。

训练时可显式选择模式：

```text
--scan-path-mode temporal4          # 第一阶段：4 路完整通道扫描
--scan-path-mode temporal4_grouped  # 第二阶段：4 路、每路 1/4 通道分组扫描
--scan-path-mode multiscale_grouped # 第三阶段：局部全分辨率 + 1/2 分辨率全局分组扫描（默认）
--scan-path-mode 8path              # 原始 8 路基线
```

旧 checkpoint 没有 `scan_path_mode` 字段时，`infer_2dprior.py` 会自动按 `8path`
构建模型以保持兼容；新 checkpoint 会把实际模式写入 `config.json`。

建议在 CUDA 云服务器上从头训练 Mamba-2 配置时，附加以下参数：

```text
--mamba-variant mamba2 --channels 32 --num-blocks 2 --mamba2-state-dim 64 --mamba2-headdim 32 --mamba2-groups 1 --mamba2-chunk-size 256 --device cuda
```

`channels=32`、两个 block 仍属于轻量模型，且 `head_dim=32` 可整除 `expand * channels=64`。Mamba-2 checkpoint 与 Mamba-1 checkpoint 不兼容。建议至少比较四组：Mamba-1、仅 Mamba-2、Mamba-2 加时域可靠度门控、Mamba-2 加两种 motion 机制；验收重点是验证段的运动边缘残影、墙面纹理保留与每帧耗时，而不是只比较训练损失。

## Linux CUDA Preparation

Mamba-2 正式训练需要 Linux、NVIDIA GPU、与 PyTorch 匹配的 CUDA 驱动，以及 Conda `base` 环境中的 Python 3.10+。请完整上传 `Mamba` 目录及 source、`denoised.raw`、`fused.raw` 三个原始文件；不要转码 RAW，三路文件必须保留相同的 200 帧顺序。建议使用至少 16 GB 显存的 GPU，从 `batch_size=1, patch_size=128` 开始；若显存不足，先将 patch 降到 96 或 64（保持为 8 的倍数）。

前提：在 Linux NVIDIA CUDA 服务器的 Bash 中，已进入上传后的 `Mamba` 目录，且 Conda `base` 环境中的 PyTorch 能识别 GPU。

```bash
nvidia-smi
conda run -n base --no-capture-output python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
conda run -n base --no-capture-output python -m pip install -r requirements.txt
conda run -n base --no-capture-output python -m pip install "mamba-ssm[causal-conv1d]" --no-build-isolation
conda run -n base --no-capture-output python -c "from mamba_ssm import Mamba2; print('Mamba2 ready')"
```

前提：上面的预检已全部成功，且 Bash 当前仍在项目目录。以下是从头训练 128x 数据的 Mamba-2 命令；请将三条输入路径替换成服务器上的实际绝对路径。

```bash
conda run -n base --no-capture-output python train_raw_fusion.py \
  --source /data/128x/source.raw \
  --two-d /data/128x/denoised.raw \
  --three-d /data/128x/fused.raw \
  --output-dir results/128x_mamba2 \
  --epochs 50 \
  --samples-per-epoch 1600 \
  --validation-samples 256 \
  --test-samples 256 \
  --patch-size 128 \
  --batch-size 1 \
  --scan-path-mode temporal4 \
  --mamba-variant mamba2 \
  --channels 32 \
  --num-blocks 2 \
  --mamba2-state-dim 64 \
  --mamba2-headdim 32 \
  --mamba2-chunk-size 256 \
  --device cuda \
  --amp auto
```

训练目录会保存 `config.json`、`last.pt`、`best.pt`、`history.json` 和 `test_metrics.json`。不要根据 test 指标继续调参；调参只看 validation 段，确认配置后再用最终 `best.pt` 推理整段 RAW。

## 时序划分

`split_sequence` 按时间连续划分，不随机打散帧。对于当前 200 帧数据，默认得到：

```text
train: 0..149       (150 帧，用于反向传播)
guard: 150..159     (10 帧，不训练、不验证)
val:   160..179     (20 帧，仅用于选择 best.pt)
test:  180..199     (20 帧，训练结束后仅评测一次)
```

guard 区隔离相邻帧依赖，避免训练帧和验证帧共享紧邻时序内容。`test_metrics.json` 只记录无 clean GT 条件下的伪标签损失和运动安全指标；它不能替代 PSNR/SSIM，但能避免用测试段参与模型挑选。

## 文件

```text
mamba_2d_prior/
  model.py          # 2DNR-prior ST-Mamba 网络与硬约束
  raw_dataset.py    # RAW 流/目录读取、特征与时序划分
  losses.py         # 无 clean GT 的保守训练损失
train_raw_fusion.py # RAW 融合训练入口
infer_2dprior.py    # 分块推理，写出 12-bit little-endian RAW 流
```

## 训练

先安装本项目依赖。CUDA 训练建议额外安装与 PyTorch/CUDA 版本匹配的 `mamba-ssm`；未安装时 `auto` 会回退到 PyTorch reference scan，能运行但速度明显较慢。

前提：已打开 Windows `cmd.exe`，且 Conda `base` 环境已安装本项目依赖。目录名含 `&` 时，下面采用其 Windows 8.3 短路径 `MIS20S~1`，避免被 Conda 的 Windows 批处理包装器截断；可先用 `dir /x` 确认短路径。

```cmd
cd /d C:\Users\HP\Desktop\Mamba
set KMP_DUPLICATE_LIB_OK=TRUE
conda run -n base --no-capture-output python train_raw_fusion.py ^
  --source D:\BaiduNetdiskDownload\company\shdarkroom\128x\raw_stream_1920x1080_16bit@RG_[Shutter=54992,SenserG=131072,IspG=1024,R=2159,G=1024,B=1849].raw ^
  --two-d D:\BaiduNetdiskDownload\company\denoise\MIS20S~1\raw_stream_1920x1080_16bit@RG_R=2159,G=1024,B=1849_128x\denoised.raw ^
  --three-d D:\BaiduNetdiskDownload\company\denoise\MIS20S~1\raw_stream_1920x1080_16bit@RG_R=2159,G=1024,B=1849_128x\fused.raw ^
  --output-dir results\128x_2dprior ^
  --epochs 50 ^
  --patch-size 128 ^
  --batch-size 4 ^
  --mamba-scan-backend auto ^
  --device auto
```

首次训练会在 `results\128x_2dprior\prior_cache` 建立四平面全帧先验 memmap；后续训练直接复用，不再为每个随机 patch 重复 Gaussian/Sobel。缓存约为 `4 个先验 × 4 个 Bayer plane × packed 帧尺寸 × 4 bytes`，例如 200 帧 1920×1080 数据约占 6.2 GiB，请预留磁盘空间。缓存会校验输入文件、尺寸、黑电平和版本；需要强制重建时加 `--rebuild-prior-cache`，需要回退旧流程时加 `--disable-prior-cache`，也可用 `--prior-cache <目录>` 指定位置。该命令会创建 `config.json`、每轮 `history.json`、最新权重 `last.pt` 与验证损失最优的 `best.pt`。如显存不足，先将 `--batch-size` 改为 `1`，再将 `--patch-size` 改为 `96` 或 `64`（必须为 8 的倍数）。

## 推理

前提：已完成训练，且 Windows `cmd.exe` 当前在项目目录。推理输出是 200 帧连续、12-bit 有效值的 little-endian `uint16` RAW，不做 +EV，也不调用 ISP。

```cmd
cd /d C:\Users\HP\Desktop\Mamba
set KMP_DUPLICATE_LIB_OK=TRUE
conda run -n base --no-capture-output python infer_2dprior.py ^
  --source D:\BaiduNetdiskDownload\company\shdarkroom\128x\raw_stream_1920x1080_16bit@RG_[Shutter=54992,SenserG=131072,IspG=1024,R=2159,G=1024,B=1849].raw ^
  --two-d D:\BaiduNetdiskDownload\company\denoise\MIS20S~1\raw_stream_1920x1080_16bit@RG_R=2159,G=1024,B=1849_128x\denoised.raw ^
  --three-d D:\BaiduNetdiskDownload\company\denoise\MIS20S~1\raw_stream_1920x1080_16bit@RG_R=2159,G=1024,B=1849_128x\fused.raw ^
  --checkpoint results\128x_2dprior\best.pt ^
  --output results\128x_2dprior\fused_mamba_2dprior.raw ^
  --tile 512 ^
  --overlap 64 ^
  --tile-batch-size 4 ^
  --device auto
```

相邻 tile 使用 Hann 加权重叠，避免空间拼接痕迹。`--tile-batch-size 4` 会把同一帧的 4 个 tile 合并为一次模型调用，并在 GPU 上完成重叠累积，最后只将整帧结果复制回 CPU；它不会消除 overlap 区域本身的重复计算。如显存不足依次改为 `2` 或 `1`，输出结果保持一致。输出旁会生成同名 JSON，记录平均 3DNR 修正权重，便于确认模型仍以 2DNR 为主。

## 建议的验收顺序

先查看 JSON 中的 `mean_3dnr_correction_weight`，正常初期应明显低于 0.35；再将输出与 `denoised`、`fused` 在运动主体边缘、暗部平坦区和高反差边缘逐段对比。若出现运动拖影，优先降低 `--max-3dnr-weight` 至 `0.20` 或提高运动教师强度；若静态暗部残噪仍明显，保留 0.35 并增加训练轮数。没有 clean GT 时，应以稳定性和运动区域不劣于 2DNR 为第一验收条件。
