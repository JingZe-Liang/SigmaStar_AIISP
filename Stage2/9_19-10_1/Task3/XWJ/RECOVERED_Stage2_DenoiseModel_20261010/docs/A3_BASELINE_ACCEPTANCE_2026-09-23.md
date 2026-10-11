# A3 Baseline Acceptance: Native 1000x and Cross-Gain 645x

Date: 2026-09-23 (Asia/Shanghai)

## Decision

The step-5000 A3 EMA checkpoint is accepted as the current candidate baseline.
No A5 retraining is started. The target/domain decision is recorded in
`docs/A5_TARGET_DOMAIN_DECISION_2026-09-23.md`. In particular, a real 1000x
window must not be paired directly with the 10x target through the invalid
cross-capture exposure fit.

## Results

| Protocol | Input PSNR | A3 PSNR | Input RMSE | A3 RMSE |
|---|---:|---:|---:|---:|
| 1000x native reference | 44.6204 dB | 53.2675 dB | 24.0565 DN | 8.8895 DN |
| 645x cross-gain holdout | 24.3519 dB | 29.5729 dB | 248.1197 DN | 136.0222 DN |

Native 1000x additional checks:

- legal fraction: `1.0`
- static pixels: `382477404`
- static fraction: `0.9222545`
- high-frequency ratio mean: `0.8561584`
- flat residual sigma: `7.1652 DN`
- prediction shape: `(200, 4, 540, 960)`

Mutually exclusive split-half reference check:

| Evaluation split | Input PSNR | A3 PSNR | A3 RMSE |
|---|---:|---:|---:|
| Odd input vs even reference | 44.5501 dB | 53.1041 dB | 9.0583 DN |
| Even input vs odd reference | 44.5526 dB | 53.0711 dB | 9.0928 DN |

This confirms that the native-domain gain is stable when the reference half
does not contain the evaluated frames.

## Verification

- Full test suite: `34 passed`, one existing `torch.meshgrid` warning.
- Native outputs: `artifacts/finetune_mis20s1/evaluation/native_1000x/`
- Existing 645x metrics were read without modification.
- The native reference panel was regenerated after fixing DN-vs-normalized
  rendering in `src/raw_fusion/finetune/a5_native.py`.

## Next Gate

No A5 training gate is open at present. Reconsider only if a concrete native
performance failure or a new disjoint supervision source appears.
