# A12 联合验收基线

本轮重新计算已有 A3 预测缓存的固定 ROI 指标，不代表新模型效果，也不构成最终画质通过。

| 序列 | 本轮处理 |
| --- | --- |
| sc465sl_darkroom18_256x | CPU 重算固定 ROI |
| sc465sl_darkroom18_2000x | CPU 重算固定 ROI |
| sc635hai_131x_low_cct | CPU 重算固定 ROI |
| sc635hai_16x_motion_color | CPU 重算固定 ROI |
| sc450ai_1435x_motion_black_object | CPU 重算固定 ROI |
| sc450ai_804x_lightchanging | CPU 重算固定 ROI |
| sc465sl_difficultpattern_4717 | CPU 重算固定 ROI |
| sc465sl_difficultpattern_8880 | CPU 重算固定 ROI |
| 04_shdarkroom_645x | 历史证据，仅登记 |
| 04_shdarkroom_1000x | 历史证据，仅登记 |
| 04_mis20s1_outdoor_1500x | 历史证据，仅登记 |

静态结果按奇偶互斥参考分别记录重建、高通、低通、带符号梯度误差和偏差；动态指标仅作诊断。具体数值见同目录各序列 JSON。

尚待补齐：真实动态逐帧参考、独立新场景、人工纹理/运动 ROI 核验、完整连续视频及候选模型比较。
