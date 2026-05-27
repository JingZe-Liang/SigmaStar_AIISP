# outputs

推理图、权重图和指标 JSON 会保存到这里。

当前子目录：

- `metrics/`：验证集、测试集和 baseline 的 JSON 指标。
- `infer_scene5_shard0/`：对 scene5 shard0 的推理可视化样例。

常见输出：

- `*_fused.png`：AI 融合后的 RAW 灰度显示。
- `*_alpha3d.png`：3DNR 权重图，越亮越偏向 3DNR。
- `*_compare.png`：noisy、2DNR、3DNR、AI fused、alpha、clean 的对比图。
- `manifest.json`：推理输出索引和 alpha 统计。

这些文件是实验产物，可以删除后由 `scripts/eval.py` 或 `scripts/infer_h5.py` 重新生成。
