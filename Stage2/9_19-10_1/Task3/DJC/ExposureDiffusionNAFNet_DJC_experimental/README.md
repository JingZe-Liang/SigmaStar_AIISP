# ExposureDiffusion NAFNet五帧RAW去噪

负责人：DJC（邸锦承）  
状态：experimental，已完成训练与抽帧测试。

## 1. 功能定位

直接从公司原始noisy RAW恢复中心帧RAW：公司适配的ExposureDiffusion NAFNet完成三步恢复，再由五帧对齐补充信息。推理不依赖公司2DNR/3DNR或RVi教师。

[实验报告与四组对比图](../Report.MD)。轻量提交不包含全量165组输出、原始数据、教师标签张量和权重。

## 2. 输入输出格式

| 项目 | 格式 |
|---|---|
| 原始输入 | 连续5帧Bayer RAW，little-endian uint16容器；分辨率按各视频记录 |
| 模型输入 | `float32 [B,5,4,H,W]`，H/W为packed尺寸，四通道R、G1、G2、B |
| 数值单位 | 有符号减黑DN12；内部按每通道有效范围归一化 |
| 必要参数 | CFA、位位置、黑电平、shot gain和读出方差，记录在`project/manifest.json` |
| 模型输出 | 三步`float32 [B,4,H,W]`归一化RAW；最后一步包含五帧修正 |
| 演示输出 | 同帧四列RGB PNG，用于观察Noisy、RVi、单帧基准和最终模型 |

MIS20S1使用`raw16 >> 4`，有效12位、容器16位；其他视频按各自`bit_shift`解码。模型API返回RAW张量，演示命令保存RGB图，不直接导出传感器RAW容器。

## 3. 方法简述

1. 官方SID NAFNet初始化，在公司数据上更新主干1000步。
2. 静态区域用t+6独立有噪帧监督；运动区域用冻结RVi八帧中心结果作为软目标。
3. 区域由独立引导帧的噪声归一化时间变化判断，不使用光流；不确定位置忽略。
4. 固定公司主干，五帧TAP对齐模块学习最终RAW修正：`output = base + residual`。
5. 五帧部分训练5000步，按固定开发验证采用2300步权重。

## 4. 运行示例

以下命令在本方法目录执行。训练环境记录为Python3.12、PyTorch2.12.1、torchvision0.27.1；先安装与CUDA兼容的PyTorch/torchvision，再安装依赖：

```bash
python -m pip install -r project/code/requirements-local.txt
```

下载下方权重并解压到本方法目录；对应公司RAW另行按`project/manifest.json`放到`project/data/raw/`。重放一个已记录帧：

```bash
python project/deliver.py infer --stream mis20s1_95abebb62dd9 --frame 8 --output project/work_delivery/replay_10x
```

其他代表帧：`mis20s1_ac9e58a75df1`/99、`sc450ai_57878d7435b2`/132、`sc635hai_fff725f9cdc8`/132。四列对比额外运行RVi，最终模型自身不需要RVi推理。没有数据或显卡可直接查看报告和图片。

## 5. 参数解析

| 参数 | 作用 |
|---|---|
| `infer` | 重放代表对比，不更新参数 |
| `--stream`、`--frame` | 选择视频ID和从0开始的中心帧号 |
| `--output` | 新输出目录，已有目录不覆盖 |
| `--checkpoint` | 可选多帧检查点，默认2300步best，与公司主干配套 |
| `--run-id` | 重新训练使用的新实验名 |
| `--updates` | 重新训练的总参数更新数，默认5000 |

复现训练还需补齐完整开发RAW及828份教师标签，本目录仅有标签索引。资产齐全后运行：

```bash
python project/deliver.py prepare
python project/deliver.py check-ready
python project/deliver.py train --run-id reproduction_001 --updates 5000
```

准备步骤不启动训练；缺少资产会退出。对齐需要torchvision可变形卷积的CUDA支持。

## 6. 方法特点与结果

- 整体模型约759.77万参数；RAW恢复主干与五帧部分配套使用。
- 已在未参与训练的传感器及场景上测试；完整实验165组，这里仅保留4组代表图。
- 公司主干原固定验证误差下降11.118%；最终模型在另一套固定验证下相对单帧基准下降1.224%，两者不累加。
- 以上为有噪参考/教师误差，不是干净GT画质提升比例。可见纹理收益仍较小，连续视频拖影和闪烁尚未评价。

## 7. 权重下载

[百度网盘下载权重](https://pan.baidu.com/s/1SqCAm9zZLwOd6IVKdygOBg?pwd=dgxm)，提取码：`dgxm`。

权重单独打包为`task3_raw_denoising_weights_20261026.zip`，不进入GitHub仓库。下载后解压到本方法目录，保持包内`project/...`路径。

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `project/runs/company_mixed_gpu_001/best.pt` | 79.5 MiB | `564d94f3031b9eb6682b13b317860beb4c55a2bed7ed09b34c668a0bc018e7ac` |
| `project/work_v3/runs/residual_v3_001/best.pt` | 8.1 MiB | `5abf0450505cf1771af6e150df45471e1406bf4b0d01b04fb6e8016c414180ef` |
| `project/weights/sid_Pg_naf2.pt` | 79.5 MiB | `5dbb17ff0edd4bfe58e59644a105f1801f7663ddda3fa90ef03d07111846c3f9` |
| `project/weights/RViDeformer-L_CRVD.pth` | 49.1 MiB | `0870c25a4eede7fd77a8da2131c5d5d357c5e9a94e9bbc7d24651aa07d7f2e37` |

前两个权重共同组成最终模型；SID权重用于复现主干初始化，RVi权重用于教师或四列参考对比。权重保留原始检查点，没有重编码或裁剪优化器状态。


## 8. 代码位置

- `project/code/company_exposure/core.py`：主干递推、基础动静划分及区域损失。
- `project/code/company_exposure/burst_detail/model.py`：五帧对齐与最终输出补偿。
- `project/code/company_exposure/burst_detail/pilot.py`：独立引导、保守动静划分与采样。
- `project/code/scripts/exposure_output_residual_local.py`：训练与固定验证。
- `project/deliver.py`：推理和训练入口。

保留原始完整源码以维持检查点签名；第三方源码、许可及来源记录位于`project/code/vendor`等对应目录。
