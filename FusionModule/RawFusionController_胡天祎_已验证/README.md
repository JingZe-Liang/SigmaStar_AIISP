# 模块名称：RawFusionController

## 1. 功能定位：

训练一个轻量 motion-aware U-Net 控制器，逐像素预测 `alpha_3d`，用于自适应融合 2DNR 与 3DNR 的 RAW 域结果。

## 2. 输入输出格式：

输入格式：

- 元素类型：HDF5 分片数据。
- Shape：`/noisy: (T, 2, H, W)`，`/2dnr: (T, H, W)`，`/3dnr: (T, H, W)`，`/clean: (T, H, W)`。
- Dtype：默认 `uint16`。
- 是否需要归一化：不需要，dataset 内部按 `data_max_value` 归一化。
- 是否需要 metadata.json：不强制；训练/验证主要依赖 `data_catalog/*.txt` 中的 H5 路径列表。

输出格式：

- 训练：`checkpoints/*/best.pt`、`last.pt`、`log.csv`、`summary.json`。
- 验证：`outputs/metrics/*.json`。
- 推理：`frame_XXXX_fused.png`、`frame_XXXX_alpha3d.png`、`frame_XXXX_compare.png`，可选 `.npy` / `.pgm`。
- Shape：融合结果与 H5 中单帧 RAW 一致；`alpha_3d` 为 `(1, H, W)`。
- Dtype：训练内部为 `float32`；可视化输出为 PNG，RAW 导出可选 `uint16` PGM。

## 3. 方法简述：

融合公式统一为：

```text
fused = alpha_3d * 3DNR + (1 - alpha_3d) * 2DNR
```

默认 `feature_mode: strong` 使用 7 个输入通道：`noisy_prev`、`noisy_curr`、帧差 motion、2DNR、3DNR、2DNR/3DNR disagreement 和当前帧边缘。网络结构为轻量 U-Net + Residual Block + GroupNorm + SEGate，训练目标同时约束 clean 重建误差、梯度、权重图平滑性、运动区域倾向和 oracle 上限。

## 4. 运行示例：

生成训练列表：

```powershell
python tools/make_splits.py --data_root "D:\data\H5" --out_dir data_catalog
```

冒烟训练：

```powershell
python scripts/train.py --config configs/train_tiny.yaml
```

验证：

```powershell
python scripts/eval.py `
  --ckpt checkpoints/full_motion_aware/best.pt `
  --list data_catalog/val.txt `
  --out_json outputs/metrics/val_eval.json
```

单个 H5 推理：

```powershell
python scripts/infer_h5.py `
  --ckpt checkpoints/full_motion_aware/best.pt `
  --h5 "D:\data\H5\scene_5\shard_0.h5" `
  --frames 0,1,2 `
  --tile 768 `
  --overlap 32 `
  --out_dir outputs/infer_scene5_shard0
```

## 5. 参数解析：

- `configs/train_*.yaml`：控制 batch size、patch size、学习率、loss 权重、模型宽度和数据码值范围。
- `--train_list` / `--val_list`：覆盖配置文件中的训练/验证 H5 列表。
- `--patch_size`：训练裁剪尺寸；显存不足时优先从 512 降到 384 或 256。
- `--ckpt`：验证或推理使用的 checkpoint。
- `--tile` / `--overlap`：大图推理时的分块大小和重叠像素。
- `--data_root`：`tools/make_splits.py` 扫描 `scene_1` 到 `scene_9` 的根目录。

## 6. 模块特点：

该模块不是重新训练端到端去噪网络，而是把已有 2DNR 和 3DNR 当作两个专家分支，由 AI 控制器学习逐像素融合权重。静止区域更偏向 3DNR，运动或拖影风险区域降低 3DNR 权重，更适合解释和调试。
