# metrics

这里存放评估脚本输出的 JSON 指标。

当前文件：

- `baseline_val.json`：验证集传统 baseline 指标，包含 2DNR、3DNR、50/50 平均融合和 pixel-wise oracle。
- `val_eval.json`：加载 AI 融合模型后的验证集指标。
- `test_eval.json`：加载 AI 融合模型后的测试集指标。

当前记录中，`val_eval.json` 的 `ai_psnr` 为 `45.16563353708121`，高于验证集 3DNR baseline 的 `44.945990290795635`。`test_eval.json` 的 `ai_psnr` 为 `46.18523691304436`，略低于测试集 3DNR 的 `46.50843507101618`，说明测试集上还需要继续看 alpha 分布、运动区域和训练配置。

生成命令示例：

```bash
python scripts/eval.py --ckpt checkpoints/full_motion_aware/best.pt --list data_catalog/val.txt --out_json outputs/metrics/val_eval.json
python scripts/eval.py --ckpt checkpoints/full_motion_aware/best.pt --list data_catalog/test.txt --out_json outputs/metrics/test_eval.json
```
