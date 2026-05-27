# NAF-BPN-UNet

负责人：邸锦承
最后修改日期：2026-05-27

## 1. 功能定位

该模块用于训练和测试一个融合 `2DNR`、`3DNR` 和 `noisy` 时序信息的 RAW 去噪网络，目标是在保持细节的同时提升最终 sRGB 质量。

它的核心任务不是单纯“比 2DNR 更平滑”，而是学习：

1. 在静止区域更多继承 2DNR 的稳定性。
2. 在运动或时域不一致区域抑制 3DNR 的拖影副作用。
3. 在 RAW 域完成融合，再在 sRGB 域用指标验证质量。

## 2. 输入输出

### `dataset.py`

#### 输入

- 输入文件格式：`.h5`
- 输入目录：`root_dir`
- 关键字段：
  - `2dnr`
  - `3dnr`
  - `noisy`
  - `clean`
- 字段 shape 约定：
  - `2dnr / 3dnr / clean`：`T x H x W`
  - `noisy`：`T x 2 x H x W`
- dtype：通常为整数 RAW，读取后会转成 `float32`
- 归一化方式：
  - `x = raw / 4095.0`
  - 即映射到 `[0, 1]` 线性 RAW

#### 输出

- 输出对象：PyTorch Dataset 样本
- 每个样本返回：
  - `img_2dnr`：`1 x H x W`
  - `img_3dnr`：`1 x H x W`
  - `noisy_t`：`1 x H x W`
  - `noisy_tm1`：`1 x H x W`
  - `img_clean`：`1 x H x W`
- 输出 dtype：`torch.float32`
- 输出归一化状态：已归一化到 `[0, 1]`

### `train.py`

#### 输入

- 输入数据：来自 `dataset.py` 的 batch
- 输入模型：
  - `img_2dnr`
  - `img_3dnr`
  - `noisy_t`
  - `noisy_tm1`
- 输入 shape：
  - batch 后为 `B x 1 x H x W`
- 验证集输入：
  - `val_files` 指定的固定 H5 文件列表

#### 输出

- `checkpoint_dir` 下的模型权重：
  - 常规 epoch checkpoint
  - best checkpoint
- `log_dir` 下的 TensorBoard 日志：
  - loss
  - patch PSNR / SSIM
  - 可视化拼图

### `test.py`

#### 输入

- 模型权重：`model_path`
- 测试数据根目录：`data_root`
- 输入字段：
  - `2dnr`
  - `3dnr`
  - `noisy`
  - `clean`

#### 输出

- 输出文件：`save_csv`
- 输出内容：
  - 每个 H5 文件的 RAW PSNR / SSIM
  - 每个 H5 文件的 sRGB PSNR / SSIM
  - 2DNR、3DNR、AI 三组结果对比

## 3. 方法简述

### `dataset.py`

核心逻辑：

1. 从 H5 中读取 `2dnr / 3dnr / noisy / clean`。
2. 统一按 `4095` 做线性归一化。
3. 若训练模式开启，则优先采样时序差异较大的区域，提高模型对运动与拖影问题的关注。

核心公式：

- 归一化：
  - `x_norm = x_raw / 4095`
- 运动差异图：
  - `diff_map = |noisy_t - noisy_{t-1}|`
- patch 运动能量：
  - `E(i, j) = mean(diff_map[i:i+h, j:j+w])`

训练时优先选取 `E(i, j)` 较大的 patch，本质上是一种简化的 hard example mining。

### `net.py`

核心逻辑：

1. 从 noisy 时序差构造 motion 相关提示。
2. 从 `2DNR / 3DNR` 的差异构造算法分歧提示。
3. 用 U-Net 风格主干预测像素级融合系数。
4. 用 Basis Prediction Network 预测一组动态卷积核基底。
5. 对 `2DNR` 和 `3DNR` 做动态融合。

核心公式：

- noisy 时序差：
  - `diff_noise = |noisy_t - noisy_{t-1}|`
- 算法差异图：
  - `diff_algo = |img_3d - img_2d|`
- 平滑后的提示图：
  - `md_noise = AvgPool(diff_noise)`
  - `md_algo = AvgPool(diff_algo)`
- 像素级融合输出：
  - `fusion(x) = sum_n coeff_n(x) * (kernel_n * [img_2d, img_3d])`

其中：

- `coeff_n(x)` 是位置相关的融合系数
- `kernel_n` 是由 basis head 预测的动态核

这等价于“根据当前局部内容，自适应决定更信任 2DNR 还是 3DNR”。

### `train.py`

核心逻辑：

1. 把模型输出和 GT 都映射到 sRGB 域。
2. 在 sRGB 域计算基础重建损失。
3. 用 motion mask 加权梯度损失，强化边缘与运动区域的一致性。
4. 用 anchor loss 约束输出不要在静止区域无意义偏离 2DNR。

核心公式：

- Charbonnier loss：
  - `L_charb = mean(sqrt((y_hat - y)^2 + eps^2))`
- Gradient loss：
  - `L_grad = mean(|grad_x(y_hat) - grad_x(y)| * mask) + mean(|grad_y(y_hat) - grad_y(y)| * mask)`
- Anchor loss：
  - `L_anchor = mean(mask * sqrt((y_hat - y_2d)^2 + eps))`
- 总损失：
  - `L = L_charb + lambda_grad * L_grad + lambda_anchor * L_anchor`
- 日志展示损失：
  - `display_loss = L + beta * alpha^step * (loss_2d + loss_3d)`

这里的设计重点是：

- 不只追求“整体更干净”
- 还要尽量减少 3DNR 风格的时域拖影和边缘错融
- `display_loss` 只用于日志观察和训练过程对比，不参与反向传播

### `test.py`

核心逻辑：

1. 逐帧读取测试 H5。
2. 执行模型推理得到 AI 融合结果。
3. 分别在 RAW 域和 sRGB 域统计指标。
4. 输出与 `2DNR / 3DNR` 的对比表。

核心公式：

- PSNR：
  - `PSNR = 10 * log10(MAX^2 / MSE)`
- SSIM：
  - 调用 `skimage.metrics.structural_similarity`

之所以同时评估 RAW 和 sRGB，是因为：

- RAW 域更反映底层数值保真
- sRGB 域更接近最终人眼感受

## 4. 运行方式

```bash
python NAF-BPN-UNet\train.py
python NAF-BPN-UNet\test.py
```

运行前请先修改脚本里的数据目录、权重路径和输出目录。

## 5. 参数解析（按脚本）

### `dataset.py`

- `root_dir`：训练或验证时读取 H5 数据的根目录。
- `patch_size`：每次裁剪的 patch 尺寸。
- `is_training`：是否启用训练态随机裁剪。
- `motion_prob`：优先抽运动区域的概率。
- `motion_trials`：搜索高运动 patch 的最大尝试次数。

### `train.py`

- `data_path`：训练数据根目录。
- `checkpoint_dir`：模型权重保存目录。
- `log_dir`：TensorBoard 日志目录。
- `val_files`：固定验证集文件列表。
- `batch_size`：训练批大小。
- `num_workers`：DataLoader 线程数。
- `save_every`：每隔多少轮保存 checkpoint。
- `preview_every`：每隔多少轮记录一次预览图。
- `patch_size`：训练时裁剪的 patch 尺寸。
- `lambda_grad`：梯度损失权重，用于强调边缘和运动区域的一致性。
- `lambda_anchor`：锚点损失权重，用于限制输出在静止区域无意义偏离 `2DNR`。
- `beta`：展示损失里的初始基线权重，只影响日志显示。
- `alpha`：展示损失的指数衰减系数，只影响日志显示。

### `test.py`

- `model_path`：测试时加载的模型权重路径。
- `data_root`：测试数据根目录。
- `save_csv`：测试结果保存路径。
- `num_files_to_test`：本次随机抽测多少个 H5 文件。
- `num_basis / kernel_size / width`：模型结构配置，必须与训练时保持一致。

## 6. 算子特点

- 同时利用了传统 2DNR、传统 3DNR 和时序 noisy 信息。
- 融合策略不是固定加权，而是内容自适应动态融合。
- 训练目标明确聚焦在“保细节、抑拖影、稳指标”这三件事上。

## 7. 验证结果

实验设置与测试结果表格请见 [验证.md](./验证.md)。
