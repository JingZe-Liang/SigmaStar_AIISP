# Curated Results

`results/2026-08-26-safe-q/` is a reproducibility baseline from a 1,000-step
GPU run using the cached MOG2 supervision policy current at that time. It is
not a production-quality model.

The bundle contains the final checkpoint, training metric log, training and
inference manifests, per-condition inference traces, a denoised comparison
summary, and a 128x triptych video rendered through one fixed ISP path.

## Comparison summary

Across 36 frames per condition, the output completed with zero fallback:

| Condition | MAE | RMSE | PSNR | SSIM | q behavior |
| --- | ---: | ---: | ---: | ---: | --- |
| 128x | 0.016290 | 0.210311 | 85.7878 | 0.99999992 | 12 of 64 cells select q=0.125; q is nonzero on about 21.33% of pixels |
| 645x | 0.000000 | 0.000000 | 99.0000 | 1.00000000 | all 64 cells select q=0 |
| Aggregate | 0.008145 | 0.105155 | 92.3939 | 0.99999996 | baseline comparison only |

The zero delta at 645x is a known limitation, not evidence of successful
fusion. Under the current fixed texture admission threshold, only about three
645x texture cells are available per frame, compared with about 54 at 128x.
The next supervision revision should select static cells using condition-aware
fused-versus-denoised high-frequency utility.

The video is an inspection artifact, not a quality metric. It compares fusion,
2DNR, and 3DNR through the same fixed ISP implementation for frames 58-93 of
the 128x sequence.
