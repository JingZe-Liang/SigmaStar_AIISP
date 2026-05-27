# 算子名称：DarkFrameRawNoise

## 1. 功能定位：

基于同传感器 dark frame 和 Poisson shot noise，为 clean RAW 合成更接近真实传感器的 noisy RAW。

## 2. 输入输出格式：

输入格式：

- 元素类型：clean RAW TIFF 与 dark frame TIFF。
- Shape：单帧 `(H, W)`，clean 与 dark frame 尺寸必须一致。
- Dtype：建议 `uint16`。
- 是否需要归一化：不需要，脚本内部按 RAW 码值处理。
- 是否需要 metadata.json：不需要。
- 目录要求：clean 与 dark frame 路径中应包含 `ISO1600`、`ISO3200` 这类 ISO 字段，脚本会自动解析。

输出格式：

- 文件：镜像 clean 目录结构输出 `{clean_stem}_noisy.tiff`。
- Shape：与输入 clean RAW 一致。
- Dtype：`uint16`。
- 是否已归一化：否。
- Effective bits：由 `white_level` 决定，默认 12-bit `0~4095`。
- Container bits：16-bit TIFF。
- 附加输出：`noise_synthesis_log.csv` 记录每张图的 ISO、K 值、dark frame 来源和统计值；`run_metadata.json` 记录运行参数。

## 3. 方法简述：

每个 ISO 下先对 dark frame 求平均得到 dark shading，再随机采样一张 dark frame 并减去 dark shading 得到 dark residual。clean RAW 减 black level 后按 Poisson 分布生成 signal-dependent shot noise，最后叠加 black level 与 dark residual：

```text
DS_iso = mean(D_1, ..., D_N)
R_k = D_k - DS_iso
K = (ISO / base_iso) * qe
I_noisy = clip(K * Poisson(max(I_clean - BL, 0) / K) + BL + R_k, 0, WL)
```

## 4. 运行示例：

```powershell
python realistic_raw_noise_synthesis.py `
  --clean_root "D:\data\clean_raw" `
  --dark_root "D:\data\dark_frames" `
  --output_root "D:\data\synthetic_noisy" `
  --black_level 240 `
  --white_level 4095 `
  --base_iso 400 `
  --qe 0.4 `
  --save_dark_shading
```

## 5. 参数解析：

- `--clean_root`：clean RAW 根目录。
- `--dark_root`：dark frame 根目录。
- `--output_root`：noisy RAW 输出根目录。
- `--black_level` / `--white_level`：RAW 黑白电平。
- `--base_iso`：计算系统增益 K 的基础 ISO。
- `--qe`：近似量子效率，参与 `K = (ISO/base_iso) * qe`。
- `--seed`：随机采样 dark frame 的种子。
- `--save_dark_shading`：是否额外保存每个 ISO 的 dark shading。
- `--max_dark_frames_per_iso`：每个 ISO 最多使用多少张 dark frame。

## 6. 算子特点：

相比只加高斯噪声，该算子同时包含 signal-dependent shot noise 和 signal-independent dark residual，更适合为 RAW 域 2DNR / 3DNR 训练或测试构造 noisy-clean paired 数据。详细流程见 `README_加噪pipeline使用说明.md`。
