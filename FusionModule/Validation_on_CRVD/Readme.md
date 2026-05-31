# Liang 早期验证工程

这个文件夹记录了 AI-ISP 融合降噪工程的起点。从 2026 年 1 月开始，项目围绕“用 AI 控制传统 ISP/降噪链路”展开：目标不是直接用端到端大模型吃掉整条 RAW 到 sRGB 流程，而是在 ISP 控制不精准、端到端 AI 降噪算力开销过高之间寻找折中方案。一路推进过程中，这里先后验证了 CRVD 数据集加载、闭源 ISP 绕行、自建 ISP、加噪与拍摄数据适配、运动检测、2D/3D 降噪融合、AI 合成与指标验证等问题。

`Liang` 目录是最早期的个人实验区，代码形态保留了探索阶段的痕迹：有可复用模块，也有一次性验证脚本。当前整理后的结构尽量把“可复用代码”和“实验脚本”分开，方便后续迁移到正式工程。

## 目录结构

```text
Liang/
├── configs/                  # 早期训练与实验配置
├── docs/                     # 开发记录与阶段性笔记
├── experiments/              # 探索性实验脚本
│   ├── model_training/       # RAW fusion / AI 合成训练验证
│   ├── motion_detection/     # 运动检测对比实验
│   └── two_three_dnr/        # 2DNR、3DNR 与 RAW 可视化实验
├── scripts/                  # 可直接运行的整体流程脚本
├── src/                      # 早期可复用源码模块
│   ├── data/CRVD/            # CRVD 数据集加载
│   ├── denoise/              # 模块化降噪链路
│   └── isp_utils/            # OpenCV ISP、OpenISP、自建 ISP 工具
├── tools/                    # 数据集分析、可视化、检查工具
└── utils/                    # 通用评估辅助函数
```

## 核心代码

### `src/data/CRVD/`

CRVD 数据集加载与测试代码。这里是早期工作的基础，主要解决 noisy/GT 配对、scene/ISO/frame 索引、单帧与序列两种读取模式等问题。

- `Load.py`：早期 CRVD 单帧加载实现。
- `SequenceWise_Load.py`：更完整的 CRVD PyTorch Dataset，支持按序列返回连续帧，用于时序降噪和 3DNR 验证。
- `Test_Load.py`、`Test_Single_Image.py`：数据加载与单图读取的测试脚本。

### `src/denoise/`

模块化降噪链路，体现了“传统降噪 + 运动检测 + 融合控制”的早期思路。

- `Data_Adjust/`：降噪前后的数据整理、归一化和反归一化。
- `Pre_Denoise/`：均值滤波等预降噪实验。
- `Two_DNR/`：基于双边滤波等空间域 2D 降噪。
- `Three_DNR/`：基于时序平均的 3D 降噪。
- `MD/`：运动检测与运动自适应 gating。
- `Fusion/`：根据运动权重融合 2DNR 与 3DNR 输出。
- `Denoising_Pipeline/`：串联上述模块的早期整体降噪 pipeline。

### `src/isp_utils/`

RAW 到可视化图像的 ISP 辅助模块，用来绕开 ISP 闭源和输出不可控的问题。

- `CRVD_OpenCV_ISP/`：基于 OpenCV 的快速 ISP 与 tensor/numpy 格式转换。
- `CRVD_OpenISP/`：接入 OpenISP 组件的尝试，包含配置、模型模块和转换入口。
- `CRVD_SelfISP/`：自建 ISP2 实验，包括有无黑电平校正的版本与参数笔记。

## 实验脚本

### `scripts/overall_pipeline.py`

早期端到端验证入口：加载 CRVD 序列，执行降噪，再通过 OpenCV ISP 转换到 sRGB，最后计算快速指标。适合回看“数据加载 - 降噪 - ISP - 指标验证”这一条最早的完整链路。

### `experiments/model_training/raw_fusion_training_main.py`

RAW fusion / AI 合成方向的大型训练验证脚本，包含 Bayer 拆分、数据读取、训练模型与损失计算等内容。它更接近一次阶段性原型，而不是稳定库代码。

### `experiments/motion_detection/motion_detection_comparison.py`

运动检测实验脚本，对比了直接检测、分通道检测、4 通道打包检测等方案，用于判断运动区域中 3DNR 是否应该减弱。

### `experiments/two_three_dnr/`

2DNR、3DNR 与 RAW 可视化相关的早期实验。

- `cnn_2dnr.py`：基于 CNN/传统方法的 2DNR 实验。
- `denoise_opencv_baselines.py`：OpenCV、BM3D 等传统降噪 baseline。
- `sigmastar_raw_reconstruction.py`：针对 SigmaStar RAW 的读取、白平衡、demosaic 与可视化重建实验，原文件名为 `111.py`。
- `smart_raw_viewer.py`：RAW 图像快速查看工具。

## 工具与配置

- `configs/raw_fusion_training.yaml`：早期 RAW fusion 训练配置。
- `tools/dataset_structure_inspector.py`：数据集目录结构检查工具。
- `tools/crvd_analysis/`：CRVD 数据集分析、PSNR/SSIM 场景检测、TIFF 可视化、VBM3D 测试、通道拆分消融等脚本。
- `utils/quick_video_eval_gt.py`：快速计算 noisy/denoise 与 GT 之间的 PSNR、MAE、MSE 等指标。
- `docs/development_log.md`：早期开发日志，记录了 CRVD 加载、序列模式、ISP 尝试等阶段性进展。

## 备注

这部分代码以“探索验证”为主，部分文件仍保留了硬编码路径、实验性依赖和历史命名风格。后续若要进入正式主干，建议按功能逐步迁移到仓库根目录的 `src/`、`scripts/` 和 `src/test/`，并补齐可重复运行的 pytest 用例。
