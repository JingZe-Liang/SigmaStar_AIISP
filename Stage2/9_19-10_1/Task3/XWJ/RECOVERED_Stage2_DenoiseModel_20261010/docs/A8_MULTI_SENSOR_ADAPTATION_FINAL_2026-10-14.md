# A8 多传感器域适配最终结论

日期：2026-10-14（Asia/Shanghai）

## 最终状态

| 传感器 | 训练方案 | 最终裁决 | 可用产物 |
|---|---|---|---|
| sc635hai | `131x_low_cct` 奇偶分半静态监督 + MIS20S1 回放；`16x_motion_color` 完整保留 | **有条件通过** | 仅 step-500 EMA 可作为 sc635hai 专用可选权重；不得替换 A3 |
| sc450ai | `804x_lightchanging` 单帧 blind-spot + MIS20S1 回放；`1435x_motion_black_object` 完整保留 | **全部否决** | step-100/200/300/400/500 均不可用 |

A3 step-5000 始终未被覆盖或修改，SHA-256 保持：

`97c4332f685f98959b243e4b100434469f714aed6329e100d22b6d8552592e0b`

数据集目录保持只读，没有使用旧教师输出或不可靠动态时域平均作为干净真值。

本阶段的数据重分类来自用户于 2026-10-13 的明确授权。参与适配的
`sc635hai_131x_low_cct` 与 `sc450ai_804x_lightchanging` 不再被当作独立 A8 测试；
独立保留结论分别来自完整的 `sc635hai_16x_motion_color` 和
`sc450ai_1435x_motion_black_object` 序列。

## sc635hai

保留 checkpoint：

`artifacts/a8_sensor_adaptation/sc635hai/training_20261013/checkpoint_step_000500.pth`

SHA-256：`8e4980a43767b5b83ecc064e8cdce2f685ea452e94af48a1f4ca252004765b3f`

固定 ISP 视频：

`artifacts/a8_sensor_adaptation/sc635hai/evaluation_step500_20261013/fixed_isp_video/sc635hai_16x_input_a3_a8_fixed_isp.mp4`

完整报告：`docs/A8_SC635HAI_ADAPTATION_RESULT_2026-10-13.md`

## sc450ai

没有可保留 checkpoint。step-500 代表性失败视频：

`artifacts/a8_sensor_adaptation/sc450ai/evaluation_step500_20261013/fixed_isp_video/sc450ai_1435x_input_a3_a8_fixed_isp.mp4`

完整报告：`docs/A8_SC450AI_ADAPTATION_RESULT_2026-10-14.md`

## 验证状态

- A8、A3、A4、跨传感器和恢复相关回归：41 passed；
- Python `compileall`：通过；
- `git diff --check`：通过；
- 冲突标记扫描：通过；
- 两段视频均为 1920x964、24 帧、6 fps、4 秒 HEVC；
- 两个传感器的训练均只改动 6 个授权输出头张量，未授权变化为 0。

## 当前未完成项

计算、筛选、报告和视频导出均已完成。剩余项只有用户对两段 fixed-ISP 视频的主观
观看反馈。该反馈可以决定 sc635hai 可选权重是否进入后续部署验收；sc450ai 因硬门槛
失败，不因主观反馈而自动改判。
