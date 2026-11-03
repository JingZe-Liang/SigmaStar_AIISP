# A10b sc450ai 噪声退化生成器验证

日期：2026-10-16（Asia/Shanghai）

## 最终裁决

**否决本轮全部 sc450ai 传感器级噪声退化生成器，不准入模型训练。**

本轮比较了三个确定性候选：

1. A9 legacy 独立高斯噪声，1 DN 量化；
2. 修正为 sc450ai 实测 4 DN 格点的独立高斯噪声；
3. 在 4 DN 量化基础上增加四个 Bayer 通道共享的最小水平/垂直频谱整形。

共享频谱模型在 `804x_lightchanging` 的独立验证区间将聚合空间相关 RMSE 相对 Q4 IID
降低 `36.58%`，但仍同时失败于：

- lag-1 最大相关误差 `0.015381`，超过 `0.015` 门槛；
- 8 个独立验证帧对只胜出 6 个，低于 7 个门槛；
- Gr/Gb/B 的逐通道 sigma 门槛没有全部通过；
- 在完整冻结后的 `1435x_motion_black_object` 硬留出上，weighted ACF MAE 为
  `0.015773`，lag-1 最大误差为 `0.049863`，均明显失败。

因此不能将该模型登记为 sc450ai 传感器级退化，也不能把它加入后续训练混合。A3 step-5000
保持不可变，本轮没有启动模型训练或生成 checkpoint。

机器可读结果：

`artifacts/a10b_sc450ai_degradation_validation_20261016/validation.json`

SHA-256：`893450bb0484c9a4164342e2c23ed5221adb481b105b684ec2b5f1cca3bcd6ae`

## 对 A10 空间结论的修正

A10 在 Bayer 打包后先执行 `spatial_sample_stride=4`，再在抽样网格上计算“相邻”相关。
因此原指标实际对应 packed lag 4，即传感器坐标 lag 8，不是最近同色像素相关。

A10 中大量精确等于 `0.25` 的行列 sigma 比值也不能解释成物理行列噪声。两条真实 RAW
均为 4 DN 格点；在 full packed resolution 重算后，训练区间和验证区间的行列比值降为
`0.0`，而仅改变抽样步长即可产生原来的 `0.25` 平台值。该现象主要来自量化和中位数
估计器，不足以支持增加 row/column 公共噪声分量。

因此 A10 的“每通道空间结构通过”只保留为发现弱相关的初步信号，原建议平均参数不得直接
进入生成器。A10b 只使用 `[0,120)` 拟合，未使用 A10 将训练与验证统计平均后的参数。

## 冻结协议

| 项目 | 设置 |
|---|---|
| 参数拟合 | `804x_lightchanging` 帧 `[0,120)` |
| 独立验证 | `804x_lightchanging` 帧 `[120,150)` |
| 传感器级硬留出 | `1435x_motion_black_object` 帧 `[172,180)` |
| 相邻帧对步长 | 4 帧，共 30/8/2 个真实帧对 |
| 空间测量 | full packed resolution lag；每第四个起点抽样，报告 lag 1-8 |
| 最终相关门槛 | packed lag 1/2/4/8 |
| 信号条件 | 相邻帧均值仅用于信号分桶和合成条件，不是干净真值 |
| 量化 | `4 * round(x / 4)`，合法范围 0-4092 DN |
| 跨通道 | identity，不拟合通道耦合 |
| 随机性 | 固定 seed；IID 与频谱候选使用相同随机序列 |
| 禁止组件 | PRNU、DSNU、row/column 公共项、MTF/blur、CFA 响应 |

条件图经过空间平滑，只用于 fixed-ISP 噪声外观展示。它没有进入 sigma 或 ACF 的真实指标，
也没有被当作训练目标、动态时域均值真值或干净空间真值。

## 独立验证结果

| 候选 | Q4 合法 | sigma median | sigma P90 | weighted ACF MAE | lag-1 最大误差 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| Legacy A9 IID Q1 | 否，24.999% | 0.08333 | 0.11905 | 0.01319 | 0.02522 | 否决 |
| Q4 IID | 是，100% | 0.00000 | 0.15429 | 0.01341 | 0.02510 | 否决 |
| A10b shared spectral Q4 | 是，100% | 0.00000 | 0.15429 | **0.00719** | 0.01538 | 否决 |

总量 sigma 不能掩盖逐通道失败：Q4 IID 和频谱模型的 Gr median 相对误差为 `0.11111`、
Gr P90 为 `0.25714`，B median 为 `0.12698`；对应门槛分别为 `0.10`、`0.25` 和
`0.10`。

频谱模型的 ACF 聚合值有改善，但稳定性不足：

| 验证帧对起始帧 | Q4 IID weighted MAE | Shared spectral weighted MAE | 频谱胜出 |
|---:|---:|---:|---|
| 120 | 0.02544 | 0.01809 | 是 |
| 124 | 0.02632 | 0.02177 | 是 |
| 128 | 0.02557 | 0.01924 | 是 |
| 132 | 0.03005 | 0.02437 | 是 |
| 136 | 0.01906 | 0.01783 | 是 |
| 140 | 0.01799 | 0.01667 | 是 |
| 144 | 0.01742 | 0.01859 | 否 |
| 148 | 0.01704 | 0.01921 | 否 |

结果为 6/8，未达到预注册的 7/8。不能因为聚合平均改善就忽略后两个帧对的反转。

## 1435x 硬留出

最终严格协议的参数和阈值在生成本轮 1435x 结果前已经冻结。早期宽门槛试跑已废弃，不构成
正式结果，也没有用于拟合频谱参数。硬留出只参与最终传感器级准入，不参与拟合或 804x
模型选择。

| 候选 | sigma median | sigma P90 | weighted ACF MAE | lag-1 最大误差 | 结果 |
|---|---:|---:|---:|---:|---|
| Q4 IID | 0.00000 | 0.15952 | 0.01491 | 0.03386 | 否决 |
| Shared spectral Q4 | 0.00000 | 0.16667 | 0.01577 | 0.04986 | 否决 |

共享频谱在 1435x 上不但没有保持 804x 的聚合改善，weighted ACF MAE 还略差于 Q4 IID。
这说明现有 804x 序列不足以识别可跨拍摄条件迁移的 sc450ai 空间噪声结构。

## Fixed-ISP 视觉证据

统一 fixed-ISP 总览：

`artifacts/a10b_sc450ai_degradation_validation_20261016/fixed_isp_evidence.png`

SHA-256：`5401bd8d6976a4451606da0b8faa830ce5f6e945bf105a776bf9c3cbc57baa11`

固定倍率噪声裁剪：

`artifacts/a10b_sc450ai_degradation_validation_20261016/fixed_isp_noise_crops.png`

SHA-256：`f889766946e754a79c6287daac96cde1c4715383191f8a2fd1ba88b24dc8a71b`

统计证据：

`artifacts/a10b_sc450ai_degradation_validation_20261016/statistical_evidence.png`

SHA-256：`50d49ece83928e6cf296896268b36093f6f2333346250a894c3baa7adf6c4075`

人工复核结论：

- Q4 IID 与共享频谱在两条序列上肉眼近似，频谱候选没有稳定、明确的视觉优势；
- 两个候选均未出现非法 RAW、明显条带、块边界或突发裁剪伪影；
- 1435x 平墙区域的真实噪声纹理没有被共享频谱可靠复现；
- 条件代理的模糊不属于生成器输出质量，它明确标记为 `not GT`；
- 视觉近似不能覆盖独立统计和硬留出的失败。

## 可复现实现

新增核心文件：

- `configs/a10b_sc450ai_degradation_validation_20261016.json`
- `src/raw_fusion/finetune/a10b_degradation.py`
- `scripts/run_a10b_degradation_validation.py`
- `scripts/render_a10b_degradation_validation.py`
- `src/tests/test_finetune_a10b_degradation.py`

复现命令：

```bash
PY=/home/jingzeliang/miniconda3/envs/aaa_312/bin/python
$PY scripts/run_a10b_degradation_validation.py
MPLCONFIGDIR=/tmp/matplotlib-a10b \
  $PY scripts/render_a10b_degradation_validation.py
$PY -m pytest -q \
  src/tests/test_finetune_a10b_degradation.py \
  src/tests/test_finetune_a10_sc450_audit.py \
  src/tests/test_finetune_a9_generalization.py
```

## 训练准入与未完成项

`model_training_allowed=false`，原因不只是本轮生成器失败：现有数据仍无法识别 sc450ai 的
干净空间细节、sensor blur/MTF、CFA 光谱响应、PRNU/DSNU 和动态运动真值。即使未来噪声
生成器通过，也不能自动获得训练授权。

本轮 A10b 的实现、独立验证、硬留出、fixed-ISP 证据和明确裁决已经完成。项目总目标仍未
完成；下一步不应继续调 A10b 门槛或用 1435x 反向拟合，而应寻找现有的独立 sc450ai 标定/
配对空间数据。如果现有数据确实不存在，则必须保持 A3 和当前传感器支持边界，不进行
sc450ai 合成微调。
