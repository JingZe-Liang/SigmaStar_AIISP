# FGRF-Net v2.0 Base16 + Separable Convolution

This is an isolated lightweight package. It changes `base_channels` from 32
to 16 and replaces every 3x3 convolution with depthwise 3x3 followed by
pointwise 1x1. Depthwise bias is disabled because GroupNorm follows each
separable convolution pair.

Expected model size:

```text
parameters: 17,501
full-resolution packed inference: about 2.36 GMAC/frame
```

The model still takes only current noisy RAW, 2DNR, and 3DNR as inputs. RAFT
flows are used only to construct training supervision. The two scene configs
use the requested 128x and 645x flow directories.

Train on an RTX 3090:

```bash
cd /HardDisk/jingzeliang/projects/SigmaStar_project/FGRF_Net_v2_0/FGRF_Net_v2_base16_sepconv
GPU=0 EPOCHS=50 BATCH_SIZE=24 WORKERS=8 LEARNING_RATE=3e-4 bash train_3090.sh
```

Weights are written to `checkpoints_base16_sepconv_3090/`. The package does
not read or overwrite the parent v2.0 checkpoint directory.

Run inference and fixed-scale quad rendering after training:

```bash
GPU=0 bash run_infer_render_3090.sh all
```

The package includes its own model, data, losses, training, inference,
rendering, validation, and test files.
