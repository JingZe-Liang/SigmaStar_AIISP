# RAW 视频 2DNR/3DNR 融合项目进展

## 1. 项目目标

公司数据只有同一时刻的 `source/noisy`、传统 `2DNR` 和传统 `3DNR` 结果，没有逐帧 clean GT。目标是保留 2DNR 的降噪能力，同时减少它的过度平滑；在运动区域尽量避免 3DNR 的残影和拖影。

数据序列：

- `128x`：200 帧，分辨率 `1920×1080`。
- `645x`：200 帧，分辨率 `1920×1080`。
- source/noisy
- 2DNR/3DNR

## 2. 原版 NAF-BPN

网络结构：

- 主干是 NAFNet 风格的 U-Net，宽度 `32`。
- U-Net 产生融合系数和 BPN basis；BPN 使用 `15` 个 basis、`7×7` 局部核。
- 接口为四路输入：`2DNR`、`3DNR`、当前 `noisy_t`、上一帧 `noisy_tm1`。
- 原版内部用当前帧与上一帧的差异，以及 2DNR/3DNR 的差异作为提示，最后在候选图上做 BPN 融合。

原版效果：

- 偏向3dnr，有一定去除运动物体环绕噪点的改善

## 3. 当前版本：BLCFA + Robust MD + 梯度约束 自监督微调

### 输入和数据域

- 四路接口仍保留，但第四路从上一帧 RAW 改为缓存的 `motion_mask`：`2DNR`、`3DNR`、`noisy_t`、`motion_mask`。
- 三通道 U-Net 实际读取：当前线性域 `noisy_t`、运动 mask 的平滑结果、`abs(3DNR-2DNR)` 的平滑结果。

### 运动检测

使用已验证的 Robust MD

- 模式固定为 `robust`，CFA 为 `BGGR`。
- black level `252`，white level `4095`。
- 两段视频各预计算 200 张 mask；训练和推理只读取缓存，不在随机 patch 内重新计算。
- 原始 mask 是 `960×540` 绿通道，加载时最近邻放大到完整 Bayer 尺寸，使一个 `2×2` Bayer 单元共享同一运动状态。
- 缓存位置：`D:\DeepLearning\VideoDenoising\BPNPipeline\pipeline3_mask_md_finetune\motion_cache\128x` 和 `645x`。
- 可视化抽检：`motion_contact_sheet.png`；当前记录的平均覆盖率约为 `128x=10.0%`、`645x=5.8%`。

### CFA 和 BPN 修改

- BPN `7×7` 核只允许采样同 CFA 相位，即相对中心的 `dy`、`dx` 都为偶数。
- 中心位置永久屏蔽；训练和推理保持一致，避免训练时遮挡、推理时恢复中心像素造成分布不一致。（这导致很偏向2dnr）
- 非法位置和中心权重固定为 `0`，有效位置经过 softmax 后归一化。

### 监督和 loss

当前只保留两项：

- masked noisy loss，占 `50%`：中心遮挡位置与遮挡前 `noisy_t` 比较，只用于限制亮度和颜色漂移，不能当作 clean 质量指标。
- candidate gradient loss，占 `50%`：只比较同 CFA、间隔 2 像素的横纵梯度；静止区偏向 3DNR，运动区和 2DNR/3DNR 差异较大的区域偏向 2DNR。不使用 noisy 梯度作为目标，避免把噪点重新优化回输出。

日志还记录 motion 覆盖率和 2DNR 梯度目标占比，便于检查 mask 是否异常。
