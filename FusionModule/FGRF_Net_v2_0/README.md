# FGRF-Net v2.0

FGRF-Net v2.0 is an isolated replacement for `FGRF_Net_1`. It predicts a
convex 2DNR/3DNR gate from the current frame only:

```text
alpha = GateNet(noisy_RAW, 2DNR, 3DNR)
Y = 2DNR + alpha * (3DNR - 2DNR)
```

The model has exactly three packed-RGGB inputs (12 channels). RAFT flow is
never concatenated, passed to `forward`, or needed by inference. It is used
only while training to build a warped multi-frame noisy-Raw proxy and static
or motion loss weights.

RAFT alignment is performed on each crop plus a 16-packed-pixel context
margin. Only the center crop reaches the model and loss, so crop borders do
not turn valid static pixels into false motion due to local warping bounds.

## Training objective

The initial v2 objective deliberately has three terms:

```text
L = 1.0 * L_gate + 0.35 * L_texture + 0.25 * L_motion
```

- `L_gate`: static, flow-supervised oracle gate from the proxy projection on
  `3DNR - 2DNR`.
- `L_texture`: static, multi-scale high-pass agreement with the proxy.
- `L_motion`: motion or uncertain regions are trained to select 2DNR.

There is no duplicate proxy-reconstruction term and no low-pass term in the
first version. Add those only after an ablation demonstrates a specific need.

## Data and flow configuration

`config_train_645.json` intentionally uses:

```text
.../denoised_raft_things_645x/flow_npy
.../denoised_raft_things_645x_backward/flow_npy
```

These are the flow directories requested for v2, not the older `_3dnr`
variant. The dataset validates 199 forward and 199 backward pairs for each
200-frame sequence.

## Run on an RTX 3090

```bash
cd /HardDisk/jingzeliang/projects/SigmaStar_project/FGRF_Net_v2_0
GPU=0 bash train_3090.sh
```

Set `GPU` to the physical `nvidia-smi` index of the RTX 3090. The launcher
uses that one visible device as `cuda:0`, validates both scenes first, then
trains into `checkpoints_v2_3090/`.

Useful overrides:

```bash
EPOCHS=100 BATCH_SIZE=8 WORKERS=4 GPU=0 bash train_3090.sh
MAX_STEPS=10 GPU=0 bash train_3090.sh
```

## Inference and Quad MP4

The v2 launcher first saves 16-bit fused RAW frames, then renders a native
1920x1080-per-panel, 3840x2160 fixed-scale comparison video. All three RAW
panels use the same ISP path and the scale is estimated once from the complete
video, so it cannot introduce per-frame exposure pumping. H.264 NVENC with
`yuv420p` is the default for broad MP4 decoder compatibility.

```bash
GPU=0 bash run_infer_render_3090.sh all
```

The outputs are `inference_v2_128x/` and `inference_v2_645x/`, each with
`fused_raw_frames/` and `<scene>_quad_fixed_scale.mp4` plus its JSON report.

## Files

- `model.py`: three-input gate network; no motion or flow argument.
- `dataset.py`: training-only warped proxy and motion/static loss masks.
- `flow_supervision.py`: flow loading, composition, warping, and confidence.
- `losses.py`: gate, texture, and motion losses.
- `train.py`, `infer.py`, `verify_data.py`: runnable entry points.
