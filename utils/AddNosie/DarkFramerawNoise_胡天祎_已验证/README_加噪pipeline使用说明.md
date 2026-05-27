# 基于 dark frame 的 RAW 加噪 pipeline 使用说明

## 1. 这份代码在做什么

这份脚本实现的是一个更贴近真实传感器的 RAW 加噪流程：

- 从同一传感器、同一 ISO 下采集多张 **dark frame**；
- 先计算每个 ISO 的 **dark shading**；
- 对每张 clean RAW：
  - 生成 **Poisson shot noise**；
  - 随机采样一张 dark frame，并减去 dark shading 得到 **dark residual**；
  - 将两者叠加到 clean RAW 上，得到 noisy RAW。

这条流程对应的核心思想是：

> **Noisy RAW = shot noise + dark residual + black level**

更完整地写成：

\[
DS_{iso} = \frac{1}{N}\sum_{i=1}^{N} D_i
\]

\[
R_k = D_k - DS_{iso}
\]

\[
K \approx \left(\frac{ISO}{base\_iso}\right) \times QE
\]

\[
I_{shot} = K \cdot \text{Poisson}\left(\frac{\max(I_{clean} - BL, 0)}{K}\right)
\]

\[
I_{noisy} = \text{clip}(I_{shot} + BL + R_k,\ 0,\ WL)
\]

其中：

- \(D_i\)：同一 ISO 下的第 \(i\) 张 dark frame；
- \(DS_{iso}\)：该 ISO 下的 dark shading；
- \(R_k\)：随机采样得到的 dark residual；
- \(BL\)：black level；
- \(WL\)：white level；
- \(K\)：假设的 system gain。

---

## 2. pipeline 详细流程

### Step 1：采集 dark frame

对每个 ISO 分别采多张 dark frame，要求：

- 完全不进光；
- 同一传感器；
- 同一 ISO；
- 尽量保持相同曝光时间、bit depth、black level 设置。

建议每个 ISO 至少准备 **10 张**，更稳妥一点可以准备 **20~50 张**。

### Step 2：计算 dark shading

对每个 ISO 下的所有 dark frame 逐像素求平均：

\[
DS_{iso} = \text{mean}(D_1, D_2, ..., D_N)
\]

这一步得到的是这个 ISO 下相对固定的底纹和偏置。

### Step 3：随机采样 dark residual

从当前 ISO 的 dark frame 中随机抽一张 \(D_k\)，然后减去 dark shading：

\[
R_k = D_k - DS_{iso}
\]

这样得到的 \(R_k\) 更像是该 ISO 下的一次随机 signal-independent noise 样本。

### Step 4：生成 shot noise

对 clean RAW 先减 black level，再在信号域里生成 Poisson shot noise：

\[
I_{shot} = K \cdot \text{Poisson}\left(\frac{\max(I_{clean} - BL, 0)}{K}\right)
\]

这里默认：

\[
K \approx \left(\frac{ISO}{base\_iso}\right) \times QE
\]

脚本默认参数是：

- `base_iso = 400`
- `qe = 0.4`

于是：

- ISO1600 时，\(K = (1600/400) \times 0.4 = 1.6\)
- ISO3200 时，\(K = (3200/400) \times 0.4 = 3.2\)

### Step 5：合成 noisy RAW

最终：

\[
I_{noisy} = \text{clip}(I_{shot} + BL + R_k,\ 0,\ WL)
\]

如果你的 clean RAW 本身已经做过 black-level subtraction，那么运行脚本时把 `--black_level` 设为 `0` 即可。

---

## 3. 代码文件说明

主文件：

- `realistic_raw_noise_synthesis.py`

它会完成以下工作：

1. 扫描 `clean_root` 下所有 tif / tiff；
2. 扫描 `dark_root` 下所有 tif / tiff，并按 ISO 分组；
3. 计算每个 ISO 的 dark shading；
4. 对每张 clean RAW 进行加噪；
5. 把 noisy 结果保存到 `output_root`；
6. 生成日志文件 `noise_synthesis_log.csv`；
7. 生成一次运行的配置记录 `run_metadata.json`。

---

## 4. 目录组织建议

### clean 数据目录示例

```text
clean_root/
├─scene1/
│  ├─ISO1600/
│  │  ├─frame1_clean.tiff
│  │  ├─frame2_clean.tiff
│  ├─ISO3200/
│  │  ├─frame1_clean.tiff
├─scene2/
│  ├─ISO1600/
│  │  ├─frame1_clean.tiff
```

### dark frame 目录示例

```text
dark_root/
├─ISO1600/
│  ├─dark_001.tiff
│  ├─dark_002.tiff
│  ├─dark_003.tiff
├─ISO3200/
│  ├─dark_001.tiff
│  ├─dark_002.tiff
```

脚本会自动从路径里解析 `ISO1600`、`ISO3200` 这种信息。

---

## 5. 环境依赖

建议 Python 3.10 及以上。

需要安装：

```bash
pip install numpy tifffile
```

---

## 6. 命令行用法

### 基本用法

```bash
python realistic_raw_noise_synthesis.py \
  --clean_root D:\\CRVD_clean \
  --dark_root D:\\dark_frames \
  --output_root D:\\CRVD_noisy \
  --black_level 240 \
  --white_level 4095
```

### 保存 dark shading

```bash
python realistic_raw_noise_synthesis.py \
  --clean_root D:\\CRVD_clean \
  --dark_root D:\\dark_frames \
  --output_root D:\\CRVD_noisy \
  --black_level 240 \
  --white_level 4095 \
  --save_dark_shading
```

这样会额外输出：

```text
output_root/
├─_dark_shading/
│  ├─ISO1600_dark_shading.tiff
│  ├─ISO3200_dark_shading.tiff
```

### 限制每个 ISO 只用部分 dark frame

```bash
python realistic_raw_noise_synthesis.py \
  --clean_root D:\\CRVD_clean \
  --dark_root D:\\dark_frames \
  --output_root D:\\CRVD_noisy \
  --max_dark_frames_per_iso 10
```

这适合先做快速实验。

### 修改 K 的假设方式

默认：

\[
K = (ISO / base\_iso) \times qe
\]

例如把 `base_iso` 设为 400，`qe` 设为 0.5：

```bash
python realistic_raw_noise_synthesis.py \
  --clean_root D:\\CRVD_clean \
  --dark_root D:\\dark_frames \
  --output_root D:\\CRVD_noisy \
  --base_iso 400 \
  --qe 0.5
```

---

## 7. 输出结果说明

### 1）noisy 图像

输出目录会镜像 clean 数据的目录结构，文件名会在原文件名后加 `_noisy`：

```text
output_root/
├─scene1/
│  ├─ISO1600/
│  │  ├─frame1_clean_noisy.tiff
│  │  ├─frame2_clean_noisy.tiff
```

### 2）日志文件 `noise_synthesis_log.csv`

每一行会记录：

- 原 clean 文件路径；
- 输出 noisy 文件路径；
- ISO；
- 使用的 K 值；
- 被随机采样到的 dark frame 路径；
- noisy 图像的最小值、最大值、均值。

### 3）运行配置 `run_metadata.json`

会保存：

- 本次运行的参数；
- 使用的公式；
- 每个 ISO 实际使用了多少张 dark frame。

---

## 8. 一个具体例子

假设：

- `clean_root = D:\CRVD_clean`
- `dark_root = D:\dark_frames`
- clean 图像是 `scene1/ISO1600/frame1_clean.tiff`
- dark frame 有 `ISO1600/dark_001.tiff ~ dark_050.tiff`

运行：

```bash
python realistic_raw_noise_synthesis.py \
  --clean_root D:\\CRVD_clean \
  --dark_root D:\\dark_frames \
  --output_root D:\\CRVD_noisy \
  --black_level 240 \
  --white_level 4095 \
  --save_dark_shading
```

脚本会做：

1. 读取 `ISO1600` 下所有 dark frame；
2. 计算 `ISO1600_dark_shading.tiff`；
3. 随机抽一张，例如 `dark_017.tiff`；
4. 计算 `dark_residual = dark_017 - dark_shading`；
5. 对 `frame1_clean.tiff` 生成 shot noise；
6. 最后得到 `frame1_clean_noisy.tiff`。

---

## 9. 使用时要注意的几点

### 1）black level 要搞清楚

如果 clean RAW 是原始传感器输出，通常带有 black level，需要设成真实值，比如 240。  
如果 clean RAW 已经做过 black-level subtraction，就把 `--black_level 0`。

### 2）dark frame 和 clean 图最好匹配

尽量保证：

- 同一传感器；
- 同一 ISO；
- 尽量相同曝光时间；
- 相同 bit depth；
- 相同 black level / ISP 前处理设置。

### 3）尺寸必须一致

当前脚本要求：

- clean RAW
- dark frame
- dark shading

三者尺寸一致。

### 4）当前版本默认处理 2D 单通道 RAW

如果你的数据是：

- packed Bayer RAW
- 单通道 `.tiff`

那可以直接用。  
如果是 4 通道分离后的 Bayer，或者带额外维度的数据，需要再改一版读写逻辑。

---

## 10. 这份脚本适合什么场景

这份脚本适合：

- 用近似干净 RAW 构造 paired noisy/clean 数据；
- 在 CRVD 风格的数据结构上批量生成 noisy 图；
- 做 2DNR / 3DNR / ISP 链路前的数据准备；
- 做不同 ISO 下的加噪实验。

---

## 11. 后续可以继续扩展什么

如果后面你想继续往“更真实”做，可以在这份代码上继续加：

- 行噪声 / 列噪声开关；
- hot pixel 注入；
- 不同曝光时间下的 dark frame 匹配；
- 视频序列里“静态固定噪声 + 动态随机噪声”两层实现；
- 保存 shot noise、dark residual 的中间结果，便于可视化分析。

---

## 12. 一句话总结

这份代码实现的是：

> **用 Poisson shot noise 表达 signal-dependent noise，用 dark frame 采样表达 signal-independent noise，从而生成更接近真实传感器的 noisy RAW。**

