# NAF-BPN RAW 融合

当前效果：相比于2dnr，有细节纹理补充（如左下角的红袋子上面的字），但是还是有类似于3dnr的噪点闪烁

这是当前版本的 NAF-BPN/BPN RAW 融合代码。模型输入为当前帧的 `2DNR`、`3DNR`、`source`，以及上一帧 `source`，输出一帧融合后的 Bayer RAW。

仓库包含代码、配置和本次训练的 checkpoint，不包含数据、日志和视频。拿到对应数据后，可以直接做推理，也可以从保存的阶段权重继续训练。

## 这版做了什么

### 1. 先把 H5 数据改成更接近公司数据

原来的 H5 数据有 clean 图，但其中的 2DNR、3DNR 与公司数据的观感不完全一致。现在先从 clean 重新生成三路数据：

```text
clean
  -> RAW 噪声
  -> 单帧 Bayer 2DNR
  -> 带简单运动判断的 3DNR
```

仿真的 3DNR 保留了公司数据的主要特点：静止处接近 2DNR，运动处保留更多噪声，避免训练出来的模型只适应理想的 3DNR。仿真脚本在 `tools/company_style_3dnr/`。它只用于准备训练数据，不会进入模型推理。

### 2. 训练分两步

**第一阶段：有 clean 的监督训练**

在重新仿真的 H5 上训练原来的 NAF-BPN。输入是 2DNR、3DNR、当前 source 和上一帧 source，目标是 clean。这个阶段主要让模型学会基本的 RAW 融合能力，得到 `stage1/best.pth`。

**第二阶段：公司数据的无 clean 微调**

公司数据没有 clean GT，所以不再硬套 clean 损失。训练时用当前帧前后共 7 帧 source 做一个去掉极值后的平均图，把它当作静止区域的参考；运动区域只用较弱的约束，防止 3DNR 拖影。离线生成的运动 mask 只用于训练损失，推理时不读取它。

这一阶段仍然训练原来的 NAF-BPN/BPN，网络结构和第一阶段权重形状不变。损失主要看四件事：静止处接近 7 帧参考、边缘不要被抹掉、运动处不要盲目跟随 3DNR、2DNR 与 3DNR 很接近时不要乱改。原有 masked noisy loss 只保留为很小的稳定项。

## 第二阶段借鉴 GateNet_LSY 的部分

借鉴的是它的**训练思路**，不是把 GateNet 网络搬过来：

1. 没有 clean GT 时，用 7 帧 source 的 trimmed-mean 做静止区域参考。
2. 把运动检测结果当作离线伪标签，只在训练时参与损失；部署不依赖 MOG2、Robust MD 或 motion cache。
3. 根据 2DNR 和 3DNR 的差异决定监督强弱。差异小时要求输出稳定，差异大时给网络更多选择空间。
4. 对运动区域采用保守约束，目标是减少拖影，而不是简单地把所有运动像素都切回 2DNR。

没有借鉴的部分：主线没有改成 GateNet 的 alpha 网络，没有改成 packed Bayer，也没有把 GateNet 的 13 个输入通道直接堆进来。这样可以继续使用第一阶段训练好的 NAF-BPN 权重。

## 数据目录

```text
/workspace/data/H5
/workspace/data/SigmaStar
```

H5 每个分片需要包含：`2dnr`、`3dnr`、`clean`、`noisy`。SigmaStar 目录需要包含 `128x`、`645x` 两组 200 帧的 source、2DNR、3DNR。具体目录规则由 `data.py` 和 `preflight.py` 检查。

## 两阶段训练

后台启动完整流程：

```bash
bash scripts/start_training.sh
bash scripts/status_training.sh
```

流程顺序为：

```text
Stage 1 监督预训练
-> 生成训练用运动标签
-> 128x 训练、645x 验证
-> 645x 训练、128x 验证
-> 两条序列联合弱监督拟合
```

关闭 SSH 或终端后训练仍会继续。再次执行 `start_training.sh` 会从各阶段的 `last.pth` 继续。日志在 `logs/`，停止命令为：

```bash
bash scripts/stop_training.sh
```

训练产物默认在：

```text
runs/stage1/best.pth
runs/stage2_128_to_645/best.pth
runs/stage2_645_to_128/best.pth
runs/stage2_final_all/final.pth
```

## 推理

推理只需要四个输入：当前 2DNR、当前 3DNR、当前 source、上一帧 source。示例：

```bash
python infer.py \
  --config configs/cloud.json \
  --checkpoint weights/final.pth \
  --sequence 645x \
  --tile-size 512 \
  --halo 32 \
  --amp \
  --output visuals/final_645x
```

推理不会读取运动 mask，也不会运行 Robust MD 或 MOG2。`render_videos.py` 可将 RAW 输出转成 AI 视频和四宫格对比视频。

`weights/final.pth` 是两条公司序列联合弱监督训练完成后的最终模型。权重已经按当前 `model.py` 做过严格加载检查。

## 645x 阶段对比结果

仓库中已保存四个阶段权重在同一条公司 `645x` 序列上的 200 帧测试结果。所有结果都使用 `tile-size=512`、`halo=32`、AMP 和 5x 显示曝光。

```text
evaluations/645x/stage1/              Stage 1 有 GT 训练后的权重
evaluations/645x/stage2_128_to_645/  128x 训练、645x 验证权重
evaluations/645x/stage2_645_to_128/  645x 训练、128x 验证权重
evaluations/645x/final/               两条序列联合弱监督最终权重
```

每个目录包含：

```text
645x_ai_fused.mp4                    AI 融合视频
645x_comparison.mp4                  Noisy / 2DNR / 3DNR / AI Fused 四宫格
random_frames/ai_fused/               10 张 AI 融合抽帧
random_frames/comparison/             10 张四宫格抽帧
manifest.json                         推理参数和耗时
```

## 权重和继续训练

`weights/` 中保存了本次运行的关键 checkpoint：

```text
final.pth                         最终联合拟合模型
stage1_best.pth                   第一阶段最佳模型
stage1_last.pth                   第一阶段最后模型
stage2_128_to_645_best.pth        128x 训练、645x 验证
stage2_128_to_645_last.pth
stage2_645_to_128_best.pth        645x 训练、128x 验证
stage2_645_to_128_last.pth
```

只想复现当前效果时，准备好 SigmaStar 数据后直接使用 `weights/final.pth` 推理即可。

需要继续第二阶段训练时，先把第一阶段权重放到训练输出目录，再启动对应 fold：

```bash
mkdir -p runs/stage1
cp weights/stage1_best.pth runs/stage1/best.pth
python -u train.py --stage 2 --fold 128_to_645 \
  --config configs/cloud.json \
  --output runs/stage2_128_to_645 \
  --init-checkpoint runs/stage1/best.pth
```

完整重跑仍可直接执行 `bash scripts/start_training.sh`；它会从头完成 Stage 1，再依次完成两个 fold 和最终拟合。若目录中已有 `last.pth`，则自动断点续训。

## 重新生成公司风格 H5

先检查，不覆盖原文件：

```bash
python tools/company_style_3dnr/replace_h5_keys_with_company_style.py \
  --h5-root /workspace/data/H5
```

确认参数后再加 `--apply`。脚本会保留 clean，并重新生成 noisy、2DNR、3DNR；替换前会做临时副本和完整性检查。

## 运行测试

```bash
python -m py_compile model.py data.py losses.py train.py pipeline.py preflight.py cache_motion.py
python pipeline.py --config configs/cloud.json --dry-run
bash scripts/run_smoke_test.sh
```
