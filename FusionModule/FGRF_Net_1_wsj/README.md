# FGRF-Net 1

This directory contains the first runnable implementation of the Flow-Guided
Residual Fusion Network for low-light RAW video.

## What the network predicts

The network does not predict a clean RAW frame. It predicts gate maps
`alpha[s, direction]` for 3 scales and 4 directional responses. The responses
come from the residual

```text
D = 3DNR - 2DNR
```

through a learnable, Sobel-initialized depthwise directional transform bank.
The final frame is always

```text
Y = clamp(2DNR + R_injected, 0, 1)
```

The current noisy RAW is used as a weak consistency term. An external denoised
RAW is the pseudo ground truth. RAFT is not a network input: its forward and
backward results create a binary static mask, and every residual path is hard
masked to zero outside that static mask.

The input is packed RGGB `[R, G1, G2, B]`. Flow files are expected to be
`[H, W, 2]` float32 `.npy`, with flow components `[u, v]` in pixels. The
forward directory contains `t -> t+1`; the backward directory contains
`t+1 -> t`. Full-resolution flow is automatically resized to packed RAW size.

## Losses

The current training objective is:

1. `pseudo_gt_mse_loss`: MSE against the external denoised pseudo GT.
2. `raw_data_consistency_loss`: weak robust consistency with the noisy RAW.
3. `base_constraint_loss`: low-pass consistency with the 2DNR base frame.

The total is

```text
L = 1.0 * L_pseudo_gt + 0.25 * L_raw + 1.0 * L_base
```

The temporal consistency loss and the old correlation prior are removed. A
pixel is static only when packed-resolution flow magnitude is at most 0.25
pixels, forward/backward confidence is at least 0.30, and a 3x3 binary erosion
keeps a safety ring around moving boundaries. Soft-shrink thresholds have a
hard floor of 0.008 in normalized RAW units.

## Important data pairing note

`config_train_128.json` and `config_train_645.json` use the two Linux data
scenes supplied for this experiment. Each scene has a matching noisy RAW,
2DNR RAW, 3DNR RAW, forward RAFT directory, and backward RAFT directory. All
streams are 200 frames at 1920x1080; training uses the 198 non-boundary frames
from each scene.

## Training

Validate all files before training:

```bash
python verify_data.py \
  --config config_train_128.json \
  --config config_train_645.json
```

Train both scenes on the RTX 4090:

```bash
bash train_all_4090.sh
```

The launcher defaults to 50 epochs, batch size 64, four persistent data-loader
workers, prefetch factor 2, learning rate `1e-3`, and mixed precision. The
training loader caches the four packed-resolution flow sequences in shared CPU
memory to avoid repeatedly decoding full-resolution `.npy` files. The learning
rate is scaled from the original `2e-4` for the larger batch. Override
without editing files, for example:

```bash
EPOCHS=100 BATCH_SIZE=64 WORKERS=4 LEARNING_RATE=1e-3 bash train_all_4090.sh
```

The packed RAW crop is 256x256 (`crop_size_packed`); this is intentional for
the 24 GB 4090. Add `--max-steps 10` after the launcher command for a smoke
run. The launcher refuses to start if either scene has fewer than 199 forward
or backward flow files.

## Inference and residual statistics

```powershell
python infer.py --config config_example_645.json `
  --checkpoint checkpoints/fgrf_epoch_001.pt `
  --output-dir inference_output --device auto --save-raw
```

Each frame produces a packed `.npy` result. With `--save-raw`, an unpacked RGGB
12-bit-in-uint16 `.raw` frame is also written. `injection_stats.csv` and
`injection_stats.json` contain:

```text
injected_l1_ratio = sum(abs(Y - 2DNR)) / sum(abs(2DNR))
injected_l2_ratio = ||Y - 2DNR||_2 / ||2DNR||_2
active_pixel_ratio = fraction of pixels whose mean absolute injection exceeds the threshold
```

`active_pixel_ratio` is the direct estimate of how much of the image receives
a meaningful injected residual. The threshold defaults to `0.002` in the
normalized 12-bit domain and can be changed with `--active-threshold`.

## Inference RAW and ISP MP4

`run_infer_isp_videos_4090.sh` runs both configured scenes on the 4090, saves
the fused 16-bit RAW frames, and renders an MP4 with the supplied OpenCV ISP.
The adapter enables ISP highlight reconstruction and a per-frame adaptive
Reinhard mapping. It parses the SigmaStar `R=...,G=...,B=...` coefficients
from each scene path as white-balance gains relative to `G=1024`. The current
training script writes epoch checkpoints only; when no
`checkpoints_pseudogt_static_4090/*best*.pt` exists, the launcher uses
`checkpoints_pseudogt_static_4090/fgrf_epoch_050.pt` and prints that fallback.

```bash
cd /HardDisk/jingzeliang/projects/SigmaStar_project/FGRF_Net_1
bash run_infer_isp_videos_4090.sh checkpoints_pseudogt_static_4090/fgrf_epoch_050.pt
```

Outputs are written to `inference_128x_best/` and `inference_645x_best/`.
Each directory contains `fused_raw_frames/out_XXXX.raw`, packed NPY files,
and `<scene>_fused_isp_highlight_adaptive.mp4`. Use `FPS=25` (or another
value) before the command to change the video frame rate.
