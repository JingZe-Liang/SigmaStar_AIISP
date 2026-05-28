# 算子名称：WaveletThresholding2DNR

## 1. 功能定位：

基于小波变换（SWT/DWT）的 Bayer RAW 域单帧 2DNR 处理，针对 Bayer 格式 RAW 图像的四个色彩平面（R/G1/G2/B）分别进行小波阈值去噪，生成去噪后的 RAW 图像；支持在存在 clean GT 时计算全图 PSNR / SSIM 指标，量化去噪效果。

## 2. 输入输出格式：

输入格式：
- 元素类型：单通道 Bayer RAW 图像（H5 格式数据集或二维数组）。
- Shape：单帧 `(H, W)`，要求 H/W 为偶数（适配 Bayer 四平面拆分），同一批次分辨率一致。
- Dtype：`uint16`（默认码值范围 0~4095）。
- 是否需要归一化：不需要，脚本内部自动转换为 float32 处理，最终还原到原始码值范围。
- 是否需要 metadata.json：不需要。
- 命名约定：H5 文件遵循 CRVD 风格目录结构，如 `scene1/shard_0.h5`，内部包含 `noisy`（噪声图）和 `clean`（干净参考图）数据集。

输出格式：
- 文件：去噪后 H5 文件（默认路径 `./denoised_h5_wt/`），内部包含 `2dnr`（去噪结果）、`noisy`（原始噪声图）、`clean`（干净参考图）；指标结果保存为 CSV 文件（如 `wt_results_scene1.csv`）。
- Shape：与输入帧一致。
- Dtype：`uint16`（与输入一致）。
- 是否已归一化：否，输出还原到 RAW 码值范围（0~clip，默认 4095）。
- Effective bits：由 `clip` 参数决定，默认 12-bit（0~4095）。
- Container bits：16-bit（H5 数据集存储）。
- 指标：全图计算 PSNR（峰值信噪比）、SSIM（结构相似性）；PSNR / SSIM 越高表示去噪结果与 clean 越接近，PSNR 单位为 dB。

## 3. 方法简述：

脚本首先将 Bayer RAW 图像拆分为 R、G1、G2、B 四个色彩平面（按指定 CFA 模式），对每个平面独立执行小波变换去噪：
- 若使用 SWT（Stationary Wavelet Transform，默认开启）：计算最大可分解层数，执行多尺度 SWT 分解，对高频系数（cH/cV/cD）采用 BayesShrink 自适应阈值计算 + 软/硬阈值处理，再逆变换重构平面；
- 若使用 DWT（Discrete Wavelet Transform）：执行多尺度 DWT 分解，对高频系数采用 visu/stein/sqtwolog 等方法计算全局阈值 + 软/硬阈值处理，逆变换后裁剪至原尺寸；
最后将四个去噪后的平面合并为完整 Bayer 图像，裁剪到指定码值范围后输出，并对比 clean 图计算 PSNR/SSIM 指标。


通过将图像进行多分辨率分析，分为：LL,LH,HL,HH 四个部分，通过行列低通滤波和高通滤波下采样后组合得到，分别表示：整体轮廓，水平细节，垂直细节，对角细节。对细节高频子带进行阈值处理（硬阈值，和 ReLU 类似，和硬阈值，加减 T），其中还可以进行多层优化，将 LL 进行细分。之后将四个部分重新组合，进行逆小波变换，逐层上采样。

## 4. 运行示例：

### 基础批量测试（处理 scene1 所有 shard 文件）
```powershell
python valid_wt.py
```

### 自定义参数运行

```python
# 直接在 valid_wt.py 中修改 wt_kwargs 后运行
if __name__ == "__main__":
    wt_kwargs = {
        "wavelet": "db8",               # 更换小波基为 db8
        "level": 4,                     # 分解层数调整为 4
        "threshold_method": "stein",    # 阈值计算方法改为 stein
        "threshold_factor": 1.0,        # 阈值缩放因子
        "threshold_type": "hard",       # 改用硬阈值（保留更多细节）
        "cfa_pattern": "RGGB",          # CFA 模式改为 RGGB
        "clip": 4095,                   # 码值裁剪上限
        "use_swt": True                 # 启用 SWT（关闭则用 DWT）
    }

    batch_test(
        root_dir="./dataset",
        scene_ids=[1, 2],               # 处理 scene1 和 scene2
        shard_pattern="shard_*.h5",
        output_csv="wt_results_scene1_2.csv",
        save_denoised_h5=True,
        output_denoised_dir="./denoised_h5_wt_custom",
        noisy_axis=0,
        **wt_kwargs
    )
```

## 5. 参数解析：

| 参数               | 类型  | 说明                                                                                                                         | 默认值   |
| ------------------ | ----- | ---------------------------------------------------------------------------------------------------------------------------- | -------- |
| `wavelet`          | str   | 小波基函数（支持 pywt 所有小波类型，如 db4/db8/haar 等）                                                                     | `"db4"`  |
| `level`            | int   | 小波分解层数（SWT 模式下自动限制为最大可分解层数）                                                                           | 3        |
| `threshold_method` | str   | 阈值计算方法（仅 DWT 生效）：<br>- visu: VisuShrink (σ√(2lnN))<br>- stein: SteinShrink (σ√(lnN))<br>- sqtwolog: σ√(lnN/ln2) | `"visu"` |
| `threshold_factor` | float | 阈值缩放因子（BayesShrink/SWT 或全局阈值 / DWT 均生效）                                                                      | 1.0      |
| `threshold_type`   | str   | 阈值类型：<br>- soft: 软阈值（更平滑，损失少量细节）<br>- hard: 硬阈值（保留细节，可能残留噪声）                             | `"soft"` |
| `cfa_pattern`      | str   | Bayer 模式，可选 RGGB/BGGR/GBRG/GRBG                                                                                          | `"GBRG"` |
| `clip`             | int   | RAW 码值裁剪上限（输出时限制 0~clip）                                                                                        | 4095     |
| `use_swt`          | bool  | 是否使用 SWT（True）或 DWT（False）                                                                                          | True     |
| `noisy_axis`       | int   | H5 文件中 noisy 数据集的通道轴（仅批量测试时生效）                                                                           | 0        |

---

## 6. 算子特点：

该算子在参数上比 BNF 效果略优，但在视觉效果上并无明显改变，仍然存在边缘模糊的问题。可以在一定基础上与 BNF 混用提高 2dnr 效果。

---

### 补充说明：
- SWT 模式下分解层数会自动限制为 `pywt.swt_max_level(min(H, W))`，避免因分辨率不足导致的分解失败；
- DWT 模式逆变换后会裁剪至原平面尺寸，保证输出与输入分辨率一致；
- 批量处理时自动跳过 H/W 为奇数的图像，避免 Bayer 四平面拆分出错；
- 输出的 H5 文件采用 gzip 压缩，节省存储空间，可直接用于后续 RAW 图像处理流程。