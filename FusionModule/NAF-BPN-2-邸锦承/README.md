# Strict BLCFA + Robust MD RAW Video Fusion

这是 `pipeline3_mask_md_finetune` 的可上传代码版本。工程用于在没有 clean GT 的公司 RAW 视频上，对原版 NAF-BPN 做严格版自监督微调。

## 方法概览

- source/noisy 与 2DNR/3DNR 先按各自 black level 转到统一线性域。
- 使用预计算的 Robust MD mask 作为运动提示；训练时不在随机 patch 内重新计算运动检测。
- 网络接口为 `2DNR`、`3DNR`、`noisy_t`、`motion_mask`。
- BPN `7×7` 核只采样同 CFA 相位位置，中心位置永久禁用。
- loss 由两部分组成：遮挡中心的 noisy Charbonnier（限制亮度/颜色漂移）和同 CFA 候选梯度约束（静止区偏 3DNR，运动/候选差异区偏 2DNR）。
- 训练输出重新编码为 2DNR/3DNR 使用的 black level 域。

这不是有 GT 的监督训练。masked noisy loss 只用于检查训练是否稳定，不能单独用来选择最终权重。

## 文件说明

- `model.py`：NAF-BPN 主体、三通道 U-Net、严格 CFA BPN 融合。
- `data.py`：公司数据发现、RAW 读取、黑电平转换和完整 Bayer patch 采样。
- `masking.py`：同 CFA 中心遮挡。
- `losses.py`：masked noisy loss 和候选梯度 loss。
- `cache_motion.py`：调用外部 Robust MD，预计算并缓存运动 mask。
- `train.py`：独立微调训练入口。
- `infer.py`：对完整 200 帧序列推理并输出 RAW。
- `infer_samples.py`：导出固定 10 帧四宫格，便于 checkpoint 间比较。
- `render_videos.py`：将完整 RAW 输出转成 AI 视频和四宫格视频。
- `verify.py`：检查配置、数据尺寸、CFA 掩码和 smoke test。
- `config.example.json`：不含本机绝对路径的配置模板。

## 依赖

建议使用 Python 3.10+、PyTorch（需要 CUDA）、NumPy、OpenCV。运动检测依赖 SigmaStar 工程中的：

`utils/MD/robust_raw_md_肖纬杰_已验证`

该模块没有复制到本目录，需在本机配置 `cache_motion.py` 中的 `MD_DIR`，或将其改为可导入的包路径。

## 数据要求

配置中的 `data_root` 下需要有：

```text
Sigmastar_7_30/shdarkroom/<sequence>/<one source RAW>
mis20s1_2D&3D/<matching sequence>/denoised/out_0000.raw ... out_0199.raw
mis20s1_2D&3D/<matching sequence>/fused/out_0000.raw ... out_0199.raw
```

每帧是 `1920×1080`、小端 `uint16` RAW；source 是 12-bit 左移 4 位格式。训练前需要为每段序列生成 200 张 MD mask。

## 使用流程

1. 复制配置：`Copy-Item config.example.json config.json`，填写数据根目录、MD 缓存目录和原始 NAF-BPN 权重路径。
2. 运行 `python cache_motion.py --sequence all` 生成 mask，并人工查看 contact sheet。
3. 运行 `python verify.py`，再运行 `python train.py --smoke-test`。
4. 运行 `python train.py` 开始正式训练。训练结果写入本目录的 `runs_v3/strict_blcfa_md_grad`。
5. 用固定 checkpoint 推理：

```powershell
python infer.py --checkpoint runs_v3\strict_blcfa_md_grad\checkpoints\checkpoint_step_005000.pth
python infer_samples.py --checkpoint runs_v3\strict_blcfa_md_grad\checkpoints\checkpoint_step_005000.pth
```

6. 将推理得到的 `outputs_v3/<checkpoint>/<sequence>` 下的 200 张 RAW 转成视频：

```powershell
python render_videos.py --config config.json --output-root outputs_v3/checkpoint_step_005000 --sequence 645x
```

该脚本输出 AI 视频和 `Noisy / 2DNR / 3DNR / AI` 四宫格视频，并用 `ffprobe` 检查分辨率、帧率和 200 帧是否完整。

## 选择模型

比较 `2500 / 5000 / 7500 / 10000 step` 的四宫格：先排除发紫、Bayer 色块、像素块和噪点回升，再看静止细节与运动拖影。没有 GT 的公司视频不要按最低训练 loss 选权重。

## 不上传的内容

训练数据、MD 缓存、checkpoint、RAW 输出、MP4、PNG 和日志均应留在本地，不要提交到 Git。仓库根目录的 `.gitignore` 已覆盖这些文件类型和目录。
