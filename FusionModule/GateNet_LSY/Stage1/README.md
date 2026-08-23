# Phase2: 2DNR/3DNR Motion-Aware Fusion

本目录实现从严格 RAW 读取、MOG2 运动检测、弱监督 GateNet 训练，到整帧推理、ISP 和四视图视频输出的完整流程。

当前代码只训练融合门控 `GateNet`，不会重新训练 2DNR、3DNR 或 MOG2。

## 0. 网络结构与自监督方法状态

### GateNet 结构

GateNet 是一个约 `9602` 个参数的轻量像素级门控网络。它不直接生成 RGB/RAW 图像，只预测 2DNR 到 3DNR 的融合权重 `alpha`。

输入共有 `15` 个 packed-RGGB 特征通道：

```text
质量特征 13 通道：
  signed(D3-D2)、abs(D3-D2)、D2 局部标准差、D3 局部标准差、
  source 前后帧时域变化、D2 亮度、source 梯度

MD 特征 2 通道：
  MOG2 mask、3x3 MD boundary
```

网络分为两个分支：

```text
quality: 13 -> Conv3x3(16) -> Conv3x3 dilation=2(24)
              -> Conv3x3(16) -> Conv1x1(1)

md:      2  -> Conv3x3(8) -> Conv3x3(8) -> Conv1x1(1)
```

两个分支的最后一层均零初始化。输出为：

```text
alpha = sigmoid(quality_logit + 0.5 * tanh(md_logit))
Y = D2 + alpha * (D3 - D2)
```

因此 `alpha=0` 表示选择 2DNR，`alpha=1` 表示选择 3DNR，输出始终是凸融合。训练时对 MD 输入做默认 `20%` dropout，降低网络对单一 MOG2 掩膜的依赖；部署时不做 dropout。完整实现见 [gatenet.py](D:/University/Fusion/Phase%20Final/Phase2/gatenet.py)。

### SURE 与 J-invariant：当前版本没有启用

需要明确区分设计讨论和当前可执行代码：当前 `train.py` / `fusion_loss.py` **没有实现 SURE，也没有实现 J-invariant**。当前实际使用的是七帧 source 的 trimmed-mean proxy 加静止/运动区域门控监督。因此当前模型不能宣称为 SURE-trained 或 J-invariant-trained。

相关设计笔记保存在 [lsy_0817.md](D:/University/Fusion/Phase%20Final/Phase2/lsy_0817.md)，其中的 SURE/J-invariant 是后续可替换的无 GT 监督方案，不是当前 checkpoint 的训练过程。

SURE 的正确接入方式是：

1. 在静止、平坦、未饱和区域估计 Bayer 四通道的异方差噪声模型 `sigma^2(mu)=a*mu+b`；
2. 对完整流程 `source -> 2DNR -> 3DNR -> MD -> GateNet` 使用 MC-SURE 估计输入散度；
3. 将 SURE 项加入门控训练，而不是只对 GateNet 的局部输入计算散度。

J-invariant 的正确接入方式是：

1. 在固定 Bayer 通道上遮挡中心 RAW 像素；
2. 用不包含中心像素的同通道邻居替换它；
3. 将遮挡后的 RAW 重新运行 2DNR、3DNR 和 MD，防止 target noise 泄漏；
4. 只在被遮挡位置用原始 noisy RAW 计算损失。

如果直接让当前已经看过中心像素的 D2/D3 输出拟合原始 noisy RAW，就不是 J-invariant，而且会鼓励网络复制噪声。SURE 需要可靠的噪声强度估计；J-invariant 需要固定 D2/D3/MD 支持遮挡输入重新运行。两者都不能只在现有 GateNet 外面简单加一项 MSE 代替。

## 1. 数据概况

`DATASET` 当前包含：

- 29 个 source stream，共 3193 帧；
- 两组配对序列：`128x` 和 `645x`；
- 每组 200 帧 source、2DNR、3DNR，共 400 个配对帧；
- 分辨率 `1920x1080`，场景 CFA 为 `RGGB`。

RAW 编码：

| 数据 | 存储 | 有效值 | 读取方式 |
|---|---|---|---|
| source | little-endian uint16，左对齐 12-bit | 0..4095 | 右移 4 位 |
| 2DNR/3DNR | little-endian uint16，右对齐 12-bit | 0..4095 | 不右移 |

训练默认黑电平为 source `252`、DNR `300`。底层 reader 不隐式扣黑电平，扣除只在训练/ISP阶段显式完成。

详细数据说明见 [DATASET_README.md](D:/University/Fusion/Phase%20Final/Phase2/DATASET_README.md)，机器清单见 `dataset_manifest.json`，校验报告见 `dataset_validation.json`。

## 2. 环境

建议使用 Python 3.10+，安装以下依赖：

```powershell
pip install numpy torch opencv-python pillow tqdm imageio-ffmpeg
```

CUDA 可选。训练默认使用 CUDA；没有 CUDA 时显式指定 `--device cpu`。

所有命令均假设当前目录是 `Phase2`：

```powershell
cd "D:\University\Fusion\Phase Final\Phase2"
```

## 3. 数据校验

快速发现数据、检查尺寸、帧数、编号和编码：

```powershell
python dataset_io.py DATASET `
  --manifest dataset_manifest.json `
  --report dataset_validation.json
```

完整校验会读取并哈希全部 source，比较拼接 RAW 与逐帧 RAW，并验证 PNG：

```powershell
python dataset_io.py DATASET `
  --deep `
  --manifest dataset_manifest.json `
  --report dataset_validation.json
```

## 4. 生成 MOG2 运动掩膜

默认对两个配对序列生成二值 MD：背景为 `0`，运动为 `255`。输出尺寸为每帧 `540x960`，对应 Bayer 2x2 packed 像素。

```powershell
python run_mog2_md.py DATASET `
  --output DERIVED/md_mog2 `
  --history 5 `
  --var-threshold 64 `
  --warmup-frames 20 `
  --median-kernel 3
```

已有输出需要明确覆盖：

```powershell
python run_mog2_md.py DATASET --output DERIVED/md_mog2 --overwrite
```

每个序列目录包含：

- `md_mog2.raw`：uint8 二值掩膜流；
- `masks/out_XXXX.png`：逐帧掩膜预览；
- `summary.json`：参数、尺寸和统计信息。

## 5. 训练 GateNet

训练样本使用当前、前一、后一帧 source，以及 `t-3..t+3` 的七帧 source。七帧 source 经黑电平/偏置对齐后构造 trimmed-mean proxy，只在高置信静止区域提供重建监督。

融合输出始终为：

```text
Y = D2 + alpha * (D3 - D2)
```

当前损失由静止区门控、运动区保守门控、静止区 proxy 重建和 edge-aware alpha 平滑组成。运动区目前使用 `alpha -> 0` 的保守策略，能减少拖影，但可能保留动态区域噪声，这是当前已知限制。

建议先做两折跨序列实验：

```powershell
python train.py --fold 128_to_645 --output runs/fold_128_to_645
python train.py --fold 645_to_128 --output runs/fold_645_to_128
```

确认两折没有退化后，再用全部数据拟合最终模型：

```powershell
python train.py --fold all --output runs/final_all
```

当前默认训练参数：`50` epochs、batch `8`、Bayer crop `256`（packed crop `128`）、每 epoch 2048 个训练 patch、512 个验证 patch、AdamW、学习率 `2e-4`、CUDA AMP、MD dropout `0.2`。

`fold=all` 的验证样本来自同一批序列，只能作为最终拟合的早停参考，不能当作跨场景泛化指标。

训练输出包括：

- `config.json`：完整路径、参数和序列统计量；
- `history.csv`：逐 epoch 训练/验证指标；
- `best.pt`：验证损失最低 checkpoint；
- `last.pt`：最后一个 checkpoint。

恢复训练示例：

```powershell
python train.py `
  --fold 128_to_645 `
  --output runs/fold_128_to_645 `
  --resume runs/fold_128_to_645/last.pt
```

## 6. 全帧推理

使用 GateNet 对两个序列的全部 200 帧推理：

```powershell
python infer.py `
  --checkpoint runs/final_all/best.pt `
  --output DERIVED/inference_final_all `
  --sequences 128x 645x
```

推理使用带 halo 的 packed 分块，避免 tile 边界产生卷积接缝。默认模型支持帧区间为 `23..196`；前 23 帧和末 3 帧直接输出 D2，并将 alpha 置零，因为它们处于 MOG2/3DNR 启动区或缺少完整训练时间窗。

每个序列输出：

- `fusion.raw`：完整 200 帧，little-endian uint16、右对齐 12-bit、RGGB；
- `alpha_u8.raw`：`200x540x960`，`0=D2`、`255=D3`；
- `summary.json`：帧数、SHA-256、alpha 统计和直通帧数量。

## 7. ISP 与视频

### 7.1 12-bit 无损 master

从融合 RAW 经过固定 ISP：黑电平、采集白平衡、固定序列曝光、RGGB 去马赛克、Gamma 2.2，然后使用 `libx265 lossless` 编码为 `gbrp12le` RGB 4:4:4 MP4：

```powershell
python render_isp_video.py `
  --input DERIVED/inference_final_all `
  --sequences 128x 645x
```

输出名为 `fusion_isp_master_12bit.mp4`。该格式保留 12-bit RGB，编码前后已经逐字节 SHA-256 验证，但部分普通播放器不支持 HEVC Range Extensions。

### 7.2 通用兼容 MP4

将 master 转为更容易播放的 H.264 `avc1`、High profile、`yuv420p`、CRF 6：

```powershell
python make_compatible_video.py `
  --root DERIVED/inference_final_all `
  --sequences 128x 645x
```

输出名为 `fusion_isp_compatible.mp4`。该版本已用 FFmpeg 和 Windows Media Foundation 完整解码验证。由于播放器兼容性要求，它会发生 12-bit RGB 到 8-bit YUV420 的显示域转换；12-bit master 应作为归档版本保留。

## 8. MD/2DNR/3DNR/融合四视图

生成同步 2x2 四视图视频：

```powershell
python make_four_view_video.py `
  --dataset-root DATASET `
  --inference-root DERIVED/inference_final_all `
  --output-root DERIVED/four_view `
  --sequences 128x 645x
```

布局固定为：

```text
┌──────────────┬──────────────┐
│ MD overlay   │ 2DNR         │
├──────────────┼──────────────┤
│ 3DNR         │ fusion       │
└──────────────┴──────────────┘
```

输出目录为 `DERIVED/four_view/<sequence_id>/`，包含四视图 MP4、`frame_0100.png` 和 `summary.json`。四视图视频为 1920x1080、24 fps、H.264 yuv420p，并已验证 Windows Media Foundation 可读取全部帧。

## 9. 测试

运行数据读取和训练回归测试：

```powershell
python -m pytest test_dataset_io.py test_training.py -q
```

当前基线为 `10 passed`。新增脚本还应至少执行语法检查：

```powershell
python -m py_compile `
  infer.py `
  render_isp_video.py `
  make_compatible_video.py `
  make_four_view_video.py
```

## 10. 重要限制

- 目前只有两个 200 帧配对序列，`fold=all` 不能证明泛化能力。
- 当前运动损失把 alpha 保守地推向 0，动态区域可能残留 D2 噪声；直接提高动态区 D3 权重又可能引入拖影。
- MOG2 是现成的二值先验，不是人工标注的真实运动分割。
- ISP 从 Bayer RAW 到 RGB 本身不可逆；12-bit master 只保证编码阶段无损，不等于 RAW 到显示 RGB 可逆。
- 兼容 MP4 为播放器可用性牺牲了位深和色度采样，需保留 12-bit master 作为高质量归档。
