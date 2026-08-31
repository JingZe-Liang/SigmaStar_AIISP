# FGRF-Net v2.0 Separable Convolution

This is an isolated lightweight package. It keeps `base_channels=32` and
replaces every 3x3 convolution with a depthwise 3x3 convolution followed by a
pointwise 1x1 convolution. Depthwise bias is disabled because GroupNorm
follows each separable convolution pair.

Expected model size:

```text
parameters: 61,005
full-resolution packed inference: about 7.85 GMAC/frame
```

The model still takes only current noisy RAW, 2DNR, and 3DNR as inputs. RAFT
flows are used only to construct training supervision. The two scene configs
use the requested 128x and 645x flow directories.

Train on an RTX 3090:

```bash
cd /HardDisk/jingzeliang/projects/SigmaStar_project/FGRF_Net_v2_0/FGRF_Net_v2_sepconv
GPU=0 EPOCHS=50 BATCH_SIZE=16 WORKERS=8 LEARNING_RATE=3e-4 bash train_3090.sh
```

Weights are written to `checkpoints_sepconv_3090/`. The package does not read
or overwrite the parent v2.0 checkpoint directory.

Run inference and fixed-scale quad rendering after training:

```bash
GPU=0 NO_LABELS=1 CQ=10 bash run_infer_render_3090.sh all
```

The renderer keeps each panel at `1920x1080` and writes a `3840x2160` MP4.
Panel labels are disabled by default; each MP4 is accompanied by a same-name
`.txt` file describing the four panel contents. The package includes its own
model, data, losses, training, inference, rendering, validation, and test files.
