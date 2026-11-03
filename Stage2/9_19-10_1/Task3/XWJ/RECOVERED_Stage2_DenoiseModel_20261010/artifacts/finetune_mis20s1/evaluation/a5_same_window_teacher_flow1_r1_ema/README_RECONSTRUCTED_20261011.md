# A5 Final EMA Evaluation Reconstruction

- Reconstructed: 2026-10-11 (Asia/Shanghai)
- Status: regenerated evaluation, not recovered historical GPU output
- Checkpoint weights: `ema`
- Checkpoint step: `20`
- Device: NVIDIA GeForce RTX 4090, UUID
  `GPU-9ee78cda-5391-4e9e-15d2-eeca142f4d7b`
- Runtime: PyTorch `2.5.1+cu124`, CUDA runtime `12.4`, cuDNN `90100`
- Tile batch: `1`

The original final same-window A5 EMA evaluation JSON and prediction arrays
were not present in the journal recovery, the old `/Data2` path, or the GitHub
snapshot. This directory was regenerated from the surviving final checkpoint
and the rebuilt canonical A3 inputs. It must not be described as a byte-for-byte
recovery of the historical prediction arrays.

## Inputs

- checkpoint SHA-256:
  `cff20b1a84438a67ae30d889ed448e3589b46518f0c08ba82070dcd563b8d7d6`
- evaluator SHA-256:
  `18c3dbfeb3aee4a0935336b96f5c549adacbeb212f08313cdc2519af1385ebc9`
- canonical 645x A3 prediction SHA-256:
  `c16421ac2371174e2739285f6e79cbb4d186f86564882a43531c374708a79f76`
- canonical native 1000x A3 prediction SHA-256:
  `2d98e6c8e7cc48d462cbc9cc8e8b477cb83c20d0eb2dd4d9d1a8652b69d17ed0`
- native temporal mean SHA-256:
  `9d8be5d2f7a6676041d332bac21c87c0cbb3b0212bcb000d6a42e81fcfb103dd`
- native valid region SHA-256:
  `377208b21f37abdd392534191138becd239b5759451f70c6772b5cdae6f2acd7`

## Command

```bash
CUDA_VISIBLE_DEVICES=GPU-9ee78cda-5391-4e9e-15d2-eeca142f4d7b \
  /home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/evaluate_a5_motion_smoke.py \
  --checkpoint artifacts/finetune_mis20s1/training_a5_same_window_teacher_flow1_r1/checkpoint_step_000020.pth \
  --baseline-645-prediction artifacts/finetune_mis20s1/evaluation/prediction_645x.npy \
  --baseline-native-prediction artifacts/finetune_mis20s1/evaluation/native_1000x/native_prediction_1000x.npy \
  --native-probe-dir artifacts/finetune_mis20s1/evaluation/1000x_self_reference_probe \
  --output-dir artifacts/finetune_mis20s1/evaluation/a5_same_window_teacher_flow1_r1_ema \
  --device cuda:0 --tile-batch 1 --weights ema
```

## Metric Check

| Metric | Historical report | Reconstructed | Delta |
| --- | ---: | ---: | ---: |
| 645x motion/input ratio | `0.48808357` | `0.4880833625793457` | `-2.0742e-07` |
| 645x PSNR | `29.513075` | `29.513074046372353` | `-9.5363e-07` |
| 645x HF ratio | `0.97253010` | `0.9725297623468112` | `-3.3765e-07` |
| 645x temporal MAD | `11.699882` | `11.699892044067383` | `+1.0044e-05` |
| native PSNR | `52.345669` | `52.34566930890153` | `+3.0890e-07` |
| native RMSE | `9.884858` | `9.884858131408691` | `+1.3141e-07` |

Both reconstructed arrays have shape `(40, 4, 540, 960)`, dtype `float16`,
finite values, and range `[0.0, 1.0]`. The historical compatibility filenames
are exact copies of the explicit frame-range outputs:

- `prediction_645x_holdout.npy` copies
  `prediction_645x_holdout_0160_0199.npy`;
- `prediction_native_1000x.npy` copies
  `prediction_1000x_native_frames_0000_0039.npy`.

The reconstruction reproduces the documented rejection metrics within
`1.1e-5` absolute error and does not change the A5 decision.
