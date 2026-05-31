## 2026-01-22 (周四) 

### ✅ 完成功能

- **CRVD数据集加载器**
    - 实现 `CRVDDataset` PyTorch接口
    - 支持场景/ISO/帧级别索引
    - 自动noisy-GT配对，随机选择noisy版本
    - 测试通过：scene 1-2, ISO 1600
    
- **新增序列模式支持** ✅
    - 增加 `sequence_mode=True` 参数（默认启用）
    - 单帧模式：`sequence_mode=False` → 返回 `(H, W)`（原版行为）
    - 序列模式：`sequence_mode=True` → 返回 `(7, H, W)` 完整帧序列
    - 时序一致性：整个序列使用同一个noisy版本
    - DataLoader：序列模式下 `shuffle=True` 安全，仅打乱序列顺序

### 📝 技术细节

- 环境：Python 3.12.12 + PyTorch 2.10.0+cpu
- 代码位置：`Data_CRVD/Load_Data.py`
- 数据格式：Raw Bayer (GBRG), 范围[240, 4095]
- 序列模式：支持视频时序处理，保持帧间依赖关系
    
---
## 2026-01-23

### 文件清单

**0. SequenceWise_Load.py**
- CRVD数据集PyTorch接口，支持单帧/序列双模式
- 验证：序列模式返回 `(T, H, W)`，DataLoader batch后 `(B, T, H, W)`

**1. ISP2.py & ISP2_Without_BLc.py**
- 基础ISP pipeline实现（有/无黑电平矫正版本）
- 问题：重影、色彩空间矩阵未知导致颜色偏差严重

**2. OpenISP2Transform.py**
- 数据格式转换：`(B, T, H, W) tensor` → `(BT, H, W) numpy`
- 作用：适配OpenISP输入要求

**3. OpenISP2.py**
- 经过调整的OpenISP2（disable多个滤波模块）
- 问题：速度极慢（即使禁用多个模块）



### 方向选择

**当前问题**：ISP pipeline处理速度慢 + 输出质量不稳定（重影/色偏）

**下一步方向**：
换用rawpy
---
