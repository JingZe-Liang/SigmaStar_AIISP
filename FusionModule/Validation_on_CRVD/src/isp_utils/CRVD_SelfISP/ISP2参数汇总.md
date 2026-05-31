# ISP Module 参数配置汇总

## 概述
本文档汇总了 ISP (Image Signal Processing) 模块的所有参数配置和实现细节。

---

## 1. 传感器参数

### Sony IMX385 (STARVIS)
- **型号**: IMX385LQR-C
- **分辨率**: 2MP (1920×1080)
- **传感器尺寸**: 1/2.8"
- **像素大小**: 3.75 μm × 3.75 μm
- **技术**: STARVIS (低光性能)
- **动态范围**: >120dB (WDR)

---

## 2. Raw 数据参数

### Bayer Pattern: GBRG
```
行0: G B G B G B ...
行1: R G R G R G ...
行2: G B G B G B ...
行3: R G R G R G ...
```

### 像素值范围
- **Black Level**: 240
- **White Level**: 4095 (2^12 - 1, 12-bit)
- **有效范围**: [240, 4095]
- **归一化**: (pixel - 240) / (4095 - 240) → [0, 1]

---

## 3. Pack 操作 (GBRG → 4 通道)

### 通道映射
根据 `utils.py` 的 `pack_gbrg_raw` 函数:

```python
# 原始 Bayer (H, W):
# G B G B ...  (行0)
# R G R G ...  (行1)

# Pack 后 (4, H/2, W/2):
packed[0] = raw[1::2, 0::2]  # R 通道
packed[1] = raw[1::2, 1::2]  # G1 通道  
packed[2] = raw[0::2, 1::2]  # B 通道
packed[3] = raw[0::2, 0::2]  # G2 通道
```

### 示例
```
原始 4×4 Bayer:           Pack 后 4×2×2:
G₀ B₀ G₁ B₁               R: [R₀ R₁]  G1: [G₂ G₃]
R₀ G₂ R₁ G₃                  [R₂ R₃]      [G₄ G₅]
G₄ B₂ G₅ B₃               
R₂ G₆ R₃ G₇               B: [B₀ B₁]  G2: [G₀ G₁]
                             [B₂ B₃]      [G₄ G₅]
```

---

## 4. ISP Pipeline 流程

### 完整流程
```
Raw Bayer (H, W)
    ↓ [1] Pack GBRG
Packed (4, H/2, W/2)
    ↓ [2] Demosaic
RGB (3, H, W) - Linear RGB
    ↓ [3] White Balance
RGB_WB (3, H, W)
    ↓ [4] Color Correction Matrix
RGB_CCM (3, H, W)
    ↓ [5] Gamma Correction
sRGB (3, H, W) - Final Output
```

### 各步骤详解

#### [1] Pack GBRG
- **输入**: (B, T, H, W) 或 (B, H, W)
- **输出**: (B, T, 4, H/2, W/2) 或 (B, 4, H/2, W/2)
- **操作**: 归一化 + 重排列

#### [2] Demosaic (去马赛克)
- **方法**: 双线性插值 (Bilinear Interpolation)
- **输入**: (4, H/2, W/2) - [R, G1, B, G2]
- **输出**: (3, H, W) - [R, G, B]
- **G 通道**: 平均 G1 和 G2
- **插值**: 使用 PyTorch `F.interpolate(..., mode='bilinear')`

#### [3] White Balance (白平衡)
**方法 A: Gray World 算法 (默认)**
```python
# 假设: 场景平均为灰色
R_mean = mean(R_channel)
G_mean = mean(G_channel)  
B_mean = mean(B_channel)

# 增益 (以 G 为参考)
R_gain = G_mean / R_mean
B_gain = G_mean / B_mean

# 应用
R_out = R_in × R_gain
G_out = G_in × 1.0
B_out = B_in × B_gain
```

**方法 B: 固定增益 (可选)**
```python
# 用户提供 wb_gains=[R_gain, G_gain, B_gain]
RGB_out = RGB_in × wb_gains
```

#### [4] Color Correction Matrix (CCM)
**目的**: 将相机 RGB 转换到标准 sRGB 色彩空间

**当前使用的 CCM (通用 D65 光源)**:
```python
CCM = [
    [1.6,  -0.4, -0.2],  # R_out
    [-0.3,  1.5, -0.2],  # G_out
    [-0.1, -0.5,  1.6],  # B_out
]

RGB_out = CCM @ RGB_in
```

**说明**:
- 这是一个基于典型 D65 光源优化的通用矩阵
- 增强对比度和色彩饱和度
- **限制**: 未针对 IMX385 具体标定,可能需要后续优化

**如何优化 CCM**:
1. 使用 Macbeth ColorChecker 拍摄标准色卡
2. 提取 24 个色块的 RGB 值
3. 使用最小二乘法计算最优 CCM:
   ```
   CCM_optimal = (S^T × S)^(-1) × S^T × T
   其中 S = 相机 RGB, T = 标准 sRGB
   ```

#### [5] Gamma Correction (伽马校正)
**sRGB 标准**:
```python
gamma = 2.2
sRGB = RGB_linear^(1/2.2)
```

**作用**:
- 线性 RGB → 非线性 sRGB
- 符合人眼感知特性
- 标准显示器需要 gamma=2.2

---

## 5. PyTorch 实现细节

### 输入/输出形状
```python
# 单帧模式
输入: (B, H, W)          # B=batch, H=1080, W=1920
输出: (B, 3, H, W)       # 3=RGB channels

# 序列模式 (配合 CRVDDataset)
输入: (B, T, H, W)       # T=7 帧
输出: (B, T, 3, H, W)
```

### 与 DataLoader 配合使用
```python
from torch.utils.data import DataLoader

# 序列模式 dataset
dataset = CRVDDataset(sequence_mode=True, ...)
loader = DataLoader(dataset, batch_size=4)

# ISP 模块
isp = ISPModule()

for noisy_seq, gt_seq in loader:
    # noisy_seq: (4, 7, 1080, 1920)
    
    # 直接喂给 ISP
    srgb_seq = isp(noisy_seq)  # → (4, 7, 3, 1080, 1920)
    
    # 后续处理...
```

---

## 6. 参数可调项

### ISPModule 初始化参数
```python
isp = ISPModule(
    black_level=240,              # 黑电平
    white_level=4095,             # 白电平  
    gamma=2.2,                    # 伽马值
    wb_gains=[1.2, 1.0, 1.5],     # 白平衡增益 (可选)
    ccm=[[...], [...], [...]],    # 自定义 CCM (可选)
)
```

### 可优化方向

#### 1. 白平衡优化
**当前**: Gray World (场景假设)
**可改进**:
- 使用相机 metadata 的 WB 增益
- 基于色温的自适应 WB
- 学习式 WB (需要训练)

#### 2. CCM 优化
**当前**: 通用 D65 矩阵
**可改进**:
- 使用 ColorChecker 标定 IMX385 的专用 CCM
- 根据 ISO 级别使用不同 CCM
- 多光源 CCM (荧光灯/日光/白炽灯)

#### 3. Demosaic 优化
**当前**: 双线性插值
**可改进**:
- Malvar-He-Cutler 算法 (边缘感知)
- 基于学习的 demosaic (如 DeepISP)
- 频域 demosaic

#### 4. Gamma 优化
**当前**: 固定 2.2
**可改进**:
- 分段 gamma (sRGB 完整曲线)
- 自适应 gamma (根据亮度分布)

---

## 7. 当前已知限制

1. **CCM 未专门标定**
   - 使用通用矩阵,非 IMX385 专用
   - 可能导致色彩偏差

2. **White Balance 依赖场景**
   - Gray World 假设场景平均灰色
   - 单色调场景 (如全红) 会失效

3. **Demosaic 较简单**
   - 双线性插值可能产生伪影
   - 边缘/纹理区域可能模糊

4. **未考虑 Lens Shading**
   - 镜头边缘暗角未校正
   - 需要 Lens Shading Correction (LSC)

5. **未实现降噪**
   - ISP 通常包含降噪步骤
   - 当前跳过,由去噪网络处理

---

## 8. 完整代码示例

### 基础用法
```python
import torch
from isp_module import ISPModule

# 创建 ISP
isp = ISPModule()

# 加载 raw 数据 (假设从 CRVDDataset)
raw_bayer = torch.rand(4, 7, 1080, 1920) * (4095 - 240) + 240

# 转换为 sRGB
srgb = isp(raw_bayer)  # (4, 7, 3, 1080, 1920)

print(f"输出形状: {srgb.shape}")
print(f"输出范围: [{srgb.min():.4f}, {srgb.max():.4f}]")
```

### 与训练流程结合
```python
from torch.utils.data import DataLoader
from SequenceWise_Load import CRVDDataset

# 数据集
dataset = CRVDDataset(
    noisy_root="path/to/noisy",
    gt_root="path/to/gt",
    scenes=[1, 2, 3, 4, 5, 6],
    iso_levels=[1600, 3200, 6400],
    sequence_mode=True,
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

# ISP 模块
isp = ISPModule().cuda()

# 训练循环
for noisy_seq, gt_seq in loader:
    noisy_seq = noisy_seq.cuda()  # (2, 7, 1080, 1920)
    
    # 方案 1: 直接在 raw 域训练
    # output_raw = denoising_model(noisy_seq)
    # loss = criterion(output_raw, gt_seq)
    
    # 方案 2: 训练后转 sRGB 可视化
    # output_raw = denoising_model(noisy_seq)
    # output_srgb = isp(output_raw)
    # gt_srgb = isp(gt_seq)
    # visualization or additional loss
    
    pass
```

---

## 9. 参数速查表

| 参数类别 | 参数名 | 默认值 | 说明 |
|---------|--------|--------|------|
| **传感器** | Sensor | IMX385 | Sony STARVIS |
| | Bayer Pattern | GBRG | 从左上角开始 |
| | Pixel Size | 3.75 μm | 单像素尺寸 |
| **Raw 数据** | Black Level | 240 | 最小像素值 |
| | White Level | 4095 | 最大像素值 (12-bit) |
| **白平衡** | Method | Gray World | 默认算法 |
| | WB Gains | None | 可自定义 [R, G, B] |
| **CCM** | Matrix | 通用 D65 | 3×3 矩阵 |
| | 优化方式 | 未标定 | 可用 ColorChecker 优化 |
| **Gamma** | Value | 2.2 | sRGB 标准 |
| **Demosaic** | Method | Bilinear | 双线性插值 |

---

## 10. 下一步优化建议

### 短期 (立即可做)
1. ✅ 实现基础 ISP pipeline
2. 测试与 CRVDDataset 配合
3. 可视化输出效果

### 中期 (需要数据)
1. 使用 ColorChecker 标定 CCM
2. 测试不同 ISO 下的色彩表现
3. 优化白平衡策略

### 长期 (需要研究)
1. 实现学习式 ISP (参考 PyNET)
2. 端到端联合训练 (去噪 + ISP)
3. 适配不同传感器

---

## 附录: 相关资源

### 学术论文
- RViDeNet (CVPR 2020): 本项目参考的去噪网络
- PyNET (CVPR 2020): 端到端学习式 ISP
- DeepISP: 深度学习 ISP pipeline

### 工具库
- `colour-demosaicing`: Python demosaic 实现
- `rawpy`: RAW 文件处理
- `kornia`: PyTorch 图像处理库

### 标准文档
- sRGB IEC 61966-2-1: sRGB 色彩空间标准
- ISO 12233: 图像分辨率测试标准
- Macbeth ColorChecker: 色彩标定标准