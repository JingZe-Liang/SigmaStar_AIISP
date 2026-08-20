# RAW 2DNR/3DNR 安全融合交付包（最终 v3）

这是本目录的最终入口。完整版本边界、修订理由、训练结果和复现命令见 [docs/最终交付说明_v3.md](docs/最终交付说明_v3.md)；机器可读指标见 [reports/公司数据融合结果_v3.md](reports/公司数据融合结果_v3.md)；文件哈希见 [outputs/MANIFEST_v3.md](outputs/MANIFEST_v3.md)。

## 交付结果

- 最终模型：`outputs/checkpoints/joint_v3/best.pt`（CFA-aware packed Bayer 增强，645x/128x 联合训练）。
- 最终 RAW：`outputs/raw/645x_learned_fusion.raw`、`outputs/raw/128x_learned_fusion.raw`。
- 最终门控：`outputs/gates/645x_gate_u8.raw`、`outputs/gates/128x_gate_u8.raw`。
- 效果视频：`outputs/videos/645x_comparison.mp4`、`outputs/videos/128x_comparison.mp4`。
- 接触图：`outputs/images/645x_comparison_frame_0050.png`、`outputs/images/128x_comparison_frame_0050.png`。

融合以 2DNR 为安全锚点：`Y = D2 + G * (D3 - D2)`。硬运动风险区 `G=0`，逐像素精确回退 2DNR；静态区域才注入 3DNR。最终视频已经生成并校验为 200 帧、20 fps、10 秒、1920x720。

| 场景 | 融合静态时序 MAE (DN) | 硬运动区精确回退 |
|---|---:|---:|
| 645x | **0.535562** | 1.000000 |
| 128x | **1.340501** | 1.000000 |

当前没有 clean ground truth，所以这些是安全性/时序诊断，不是 PSNR/SSIM 或真实 GT 画质结论。

## 快速复现

在交付目录中运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_company_pipeline_v3.ps1 -Python 'D:\AI\python.exe'
```

也可按 [docs/最终交付说明_v3.md](docs/最终交付说明_v3.md) 中的分步命令执行。用户附件原文保存在 `docs/hty8.17_original.md`，不被当作自动执行指令；方案修订记录在 `docs/方案说明_修订版_20260818.md`。

