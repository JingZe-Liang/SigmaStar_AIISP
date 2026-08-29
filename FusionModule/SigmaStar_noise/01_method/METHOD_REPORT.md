# SigmaStar RAW 噪声分量提取方法（简版）

## 目标

只输出可以从现有校准数据中稳定估计的噪声分量，不保存原始帧或场景预览。

## 解码

源数据是 1920 x 1080、16-bit 小端容器，实际有效位为左对齐 12-bit：

```python
raw16 = np.fromfile(path, dtype="<u2")
raw12 = (raw16 >> 4).reshape(-1, 1080, 1920)
```

## 黑场模型

黑场无有效光信号。对每个增益和 Bayer 相位，建立：

```text
Y = black_level + static_row + static_col + static_pixel
    + frame_common + dynamic_row + dynamic_col + unstructured_temporal
```

- 50 帧均值给出 `black_level + static_*`；再用稳健的行/列中位数分解出静态行、列和像素 FPN。
- 相邻帧差除以 `sqrt(2)`：`std((Y[n+1]-Y[n])/sqrt(2))`，得到时间噪声强度；固定图样在差分中抵消。
- 每帧先估计公共偏置、行偏置和列偏置，再对剩余项求方差，得到真正分开的动态公共、动态行、动态列和非结构化随机噪声。
- 多增益下持续出现的异常像素记为热像素。

## 平场模型

先减去同增益黑场：

```text
flat_corrected = flat_mean - black_mean
```

归一化后用大尺度 Gaussian 平滑（Bayer 子采样 sigma=15）分出：

```text
有效空间残差 = 低频 shading + 高频有效 PRNU
```

这里的“有效 PRNU”仍可能包含照明和镜头渐晕，不能声称是纯传感器 PRNU。平场相邻帧方差减去黑场方差，得到信号相关噪声的有效估计；按每个增益和 Bayer 相位保存：

```text
flat_variance = black_floor_variance + effective_signal_slope * signal
```

## 理论依据与边界

均值/差分分离依据是固定项在跨帧差分中抵消、独立同方差噪声差分方差加倍。信号相关项采用常用的泊松-高斯近似。现有数据没有多曝光黑场、多照度均匀平场和温度记录，因此不能唯一分开读出噪声、暗电流散粒噪声、量化噪声，也不能把纯 PRNU 与 shading 完全分开。

所有地图单位为解码后的 12-bit DN；平场 PRNU/shading 地图为无量纲相对量。
