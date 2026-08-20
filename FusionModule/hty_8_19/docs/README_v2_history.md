# RAW 2DNR/3DNR 安全融合交付包

本目录是 2026-08-18 对公司两组暗光 RAW 序列的可复现交付。最终输出采用 `v2` 管线：联合训练一个门控网络，使用显式安全置信度限制 3DNR 的注入，在运动风险区硬回退到 2DNR，在非风险区对门控做因果迟滞，并在原生候选 DN 域完成量化。

## 交付结论

- `645x` 和 `128x` 各 200 帧，三路输入帧数、分辨率和帧顺序已核对一致。
- 最终融合在两组序列的硬运动区都做到 100% 精确回退到 2DNR，融合输出与 2DNR 的偏差为 `0 DN`。
- 经过迟滞后的静态持久区域门控变化均值约 `0.007`，P99 为 `0.059`（8-bit gate 归一化值），比逐帧独立门控稳定。
- 当前数据没有 clean ground truth，因此报告的是安全性、候选偏差和时序诊断，不虚报 PSNR/SSIM 或“优于真实 GT”的结论。

## 目录

```text
configs/company.yaml                 数据路径、黑白电平、阈值和训练参数
src/dnr_fusion/                      RAW IO、置信度、模型、训练、推理、评估、视频
tests/                               单元测试
scripts/                             PowerShell 一键命令
docs/hty8.17_original.md             用户附件原文副本
docs/方案说明_修订版_20260818.md      对原方案的逐条修订和调研依据
reports/公司数据融合结果_20260818.md  本次训练、融合、视频和指标结果
outputs/audit/                       数据审计 JSON
outputs/checkpoints/joint_v2/        最终联合模型和训练历史
outputs/raw/                         最终连续 uint16 RAW 融合流
outputs/gates/                       每帧 packed gate uint8 流
outputs/metrics/                     推理、评估、稳定性 JSON
outputs/videos/                      2DNR/3DNR/融合对比 MP4
outputs/images/                      视频第 50 帧接触图
```

## 用户请求与附件边界

用户请求是：检查现有方案、调研、构建代码、在公司数据上训练和融合，并把代码、说明和效果视频整理在一起。`hty8.17.md` 只作为待审查的设计输入保存；其中的文献判断、伪 GT 假设、Noise2Noise 假设和网络公式都经过了数据核验，不能自动当作执行指令或实验事实。修订后的决定记录在 `docs/方案说明_修订版_20260818.md`。

## 输入约定

输入路径写在 `configs/company.yaml`，默认是：

- 原始源：`D:\zhuo mian\Sigmastar_7_30\shdarkroom\{scene}\...raw`
- 2DNR：`D:\zhuo mian\mis20s1_2D&3D\...\denoised.raw`
- 3DNR：`D:\zhuo mian\mis20s1_2D&3D\...\fused.raw`

源 RAW 是左移 4 位的 16-bit 存储，读取时右移 4 位；候选流直接是有效 12-bit DN。当前配置使用源黑电平 `252`、候选黑电平 `300`、白电平 `4095`、RGGB 和小端 `uint16`。

## 复现

在本目录执行（Python 解释器为公司机器上的 `D:\AI\python.exe`）：

```powershell
$env:PYTHONPATH = 'src'

# 数据审计
D:\AI\python.exe -m dnr_fusion.audit --config configs/company.yaml

# 联合训练（时间留出验证）
D:\AI\python.exe -m dnr_fusion.train_joint --config configs/company.yaml --device cuda

# 全序列融合
D:\AI\python.exe -m dnr_fusion.infer_v2 --config configs/company.yaml --scene 645x --checkpoint outputs/checkpoints/joint_v2/best.pt --device cuda --rise-alpha 0.08 --fall-alpha 1.0 --overwrite
D:\AI\python.exe -m dnr_fusion.infer_v2 --config configs/company.yaml --scene 128x --checkpoint outputs/checkpoints/joint_v2/best.pt --device cuda --rise-alpha 0.08 --fall-alpha 1.0 --overwrite

# 安全、候选偏差和门控时序指标
D:\AI\python.exe -m dnr_fusion.evaluate --config configs/company.yaml --scene 645x --device cuda
D:\AI\python.exe -m dnr_fusion.evaluate --config configs/company.yaml --scene 128x --device cuda
D:\AI\python.exe -m dnr_fusion.stability --config configs/company.yaml --scene 645x --device cuda
D:\AI\python.exe -m dnr_fusion.stability --config configs/company.yaml --scene 128x --device cuda

# 生成对比视频（中文路径下建议先输出到 ASCII 临时目录，再复制）
D:\AI\python.exe -m dnr_fusion.video --config configs/company.yaml --scene 645x --overwrite
D:\AI\python.exe -m dnr_fusion.video --config configs/company.yaml --scene 128x --overwrite

# 测试
D:\AI\python.exe -m unittest discover -s tests -v
```

视频脚本使用 OpenCV；若系统 OpenCV 无法直接写中文绝对路径，使用 `--output` 和 `--contact-sheet` 指向 ASCII 临时路径，再用 PowerShell `Copy-Item` 放回 `outputs/videos` 和 `outputs/images`。

## 最终结果摘要

| 场景 | 静态时序 MAE：2DNR / 3DNR / 融合 (DN) | 硬运动区精确回退 | 静态高置信区平均 gate |
|---|---:|---:|---:|
| 645x | 0.575 / 0.694 / **0.536** | 1.000 | 0.819 |
| 128x | 1.894 / 1.660 / **1.334** | 1.000 | 0.811 |

这些是无 GT 条件下的诊断指标，不是画质真值指标。详细数值、视频哈希、训练摘要和限制见 `reports/公司数据融合结果_20260818.md`。

## 调研入口

实现取舍参考了 [CRVD/RViDeNet](https://openaccess.thecvf.com/content_CVPR_2020/html/Yue_Supervised_Raw_Video_Denoising_With_a_Benchmark_Dataset_on_Dynamic_CVPR_2020_paper.html)、[RViDeformer](https://arxiv.org/abs/2305.00767)、[UDVD](https://openaccess.thecvf.com/content/ICCV2021/html/Sheth_Unsupervised_Deep_Video_Denoising_ICCV_2021_paper.html)、[Neighbor2Neighbor](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.html) 和 [AIM 2025 RAW 视频降噪挑战](https://openaccess.thecvf.com/content/ICCV2025W/AIM/html/Yakovenko_AIM_2025_Low-light_RAW_Video_Denoising_Challenge_Dataset_Methods_and_ICCVW_2025_paper.html)。公开数据集方法用于设计参考，不被当作本公司序列的 ground truth。
