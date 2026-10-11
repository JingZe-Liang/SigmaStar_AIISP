# Stage2 DenoiseModel Recovery Rebuild

- Rebuild period: 2026-10-10 through 2026-10-11 (Asia/Shanghai)
- Recovery root: `/Data4/Xiao/RECOVERED_Stage2_DenoiseModel_20261010`
- Data root: `/HardDisk/jingzeliang/Data/Company_Dataset/dataset`
- Journal sources: `/Data4/restore_journal/MAGIC-1` and `MAGIC-2`
- Original handoff: `STAGE2-A5-FINAL-20261007`
- Runtime configuration: `configs/recovery_runtime_20261010.json`

This report supersedes the temporary-location warning in
`RECOVERY_20261010.md`. That file remains unchanged as provenance for the
initial journal recovery in `/tmp`; the durable operational copy is the
`/Data4/Xiao` tree above.

## Current Status

The recovered project is operational for A3 inference, native 1000x
evaluation, A5/A6 result inspection, and A7 deterministic motion-region
fusion. The missing supervision and canonical baseline outputs needed by the
next Stage2 investigation have been rebuilt from the surviving RAW dataset.

No model training was started during this rebuild. GPU work was limited to
deterministic reconstruction inference. The remaining inference, completed on
2026-10-11, used the RTX 4090 selected by UUID
`GPU-9ee78cda-5391-4e9e-15d2-eeca142f4d7b`, then addressed as `cuda:0`.

## Rebuilt Supervision

All 15 missing motion-mask arrays were regenerated under
`artifacts/finetune_mis20s1/supervision/motion_masks/`.

- Status: `complete`
- Black-level error against recovered manifests: `0.0`
- Threshold-mean error: `0.0 DN`
- Maximum per-frame static-fraction error: `0.0`
- Verification report:
  `artifacts/finetune_mis20s1/supervision/motion_masks/rebuild_report_20261010.json`
- Critical shdarkroom 645x mask SHA-256:
  `0b96da771805703a5e89cd34c9ea5a842a23c0b755530903937b88d4c0594c97`

The A5 dynamic-window data was rebuilt from the exact recovered manifests:

- train windows: 19
- holdout windows: 4
- original journal metadata preserved under
  `.recovery/evidence/original_metadata/`
- operational manifests were path-normalized and then verified against those
  originals

## Native 1000x Reference

The motion-aware 200-frame native reference was rebuilt under
`artifacts/finetune_mis20s1/evaluation/1000x_self_reference_probe/`.

- valid-region fraction: `0.9320698302469136`
- split-half residual sigma DN, R/Gr/Gb/B:
  `2.0895131 / 2.3458307 / 2.3444002 / 2.0223231`
- `temporal_mean_native_dn.npy` SHA-256:
  `9d8be5d2f7a6676041d332bac21c87c0cbb3b0212bcb000d6a42e81fcfb103dd`
- the temporal mean is byte-identical to the anonymous recovered journal
  candidate

The missing `scripts/run_native_1000x_eval.py` entrypoint was restored from
the journal. Its SHA-256 is
`bfff627059effaf7e0c57cc252995850c668906feff82177dca1a1c209b2e409`.

## Canonical A3 Outputs

The canonical full 200-frame 645x outputs were regenerated:

- `prediction_645x.npy` SHA-256:
  `c16421ac2371174e2739285f6e79cbb4d186f86564882a43531c374708a79f76`
- `old_teacher_645x.npy` SHA-256:
  `cd1279bae52cdd85be55a77f7cf58ebd3e1bea24f03c73eb25e7a583ff0a7e9a`
- full A3 PSNR: `29.572845102228044 dB`
- full input PSNR: `24.351854594563612 dB`
- full old-teacher PSNR: `28.019591307211353 dB`
- legal RAW fraction: `1.0`

The regenerated 645x holdout fingerprint is tolerance-equivalent, not
byte-identical, to the historical run:

- PSNR delta: `3.872233241608569e-05 dB`
- motion/input-ratio delta: `9.000301361083984e-06`
- audit: `artifacts/finetune_mis20s1/evaluation/recovery_a3_full_prediction_audit.json`

The canonical native 1000x full 200-frame output was regenerated:

- prediction shape/dtype: `(200, 4, 540, 960)`, `float16`
- prediction SHA-256:
  `2d98e6c8e7cc48d462cbc9cc8e8b477cb83c20d0eb2dd4d9d1a8652b69d17ed0`
- full A3: `53.26749821864931 dB`, `8.889533996582031 DN RMSE`
- full input: `44.620441181907225 dB`, `24.056467056274414 DN RMSE`
- legal RAW fraction: `1.0`
- static fraction: `0.922254542824074`
- historical first-40-frame fingerprint: exact match at
  `52.36373272976205 dB`, `9.864322662353516 DN RMSE`
- audit:
  `artifacts/finetune_mis20s1/evaluation/recovery_native_full_prediction_audit.json`

## A7 Recovery

The missing r7 and r11 evaluation JSON files, prediction previews, and six
fixed-ISP panels per radius were regenerated. Their recomputed fusion weights
are byte-identical to the recovered pre-damage arrays.

- r7 weight SHA-256:
  `d67685094cf4634f62c64ed0ddcd1512d21c9ed187ccf76f3b58d0a1f4ab626f`
- r7 fusion-weight fraction mean: `0.024376639660493825`
- r11 weight SHA-256:
  `7ca6f1ddd42ca0d5d1203e06a4ac07256f843aadd942cb45c271b73117d4310b`
- r11 fusion-weight fraction mean: `0.02808439429012346`
- static nonzero weights: `0` for both variants

These are post-processing research results. They do not replace the A3 model
checkpoint.

## Historical A4 1000x Output

The missing legacy `1000x -> 10x` cross-gain A4 files were also rebuilt for
historical completeness. Their metrics are recorded in
`artifacts/finetune_mis20s1/evaluation/metrics_1000x.json`.

- A3: `18.837191398880137 dB`, `468.1600341796875 DN RMSE`
- input: `18.734944733107064 dB`, `473.7035827636719 DN RMSE`
- old teacher: `18.775610590126767 dB`, `471.490966796875 DN RMSE`
- legal RAW fraction: `1.0`
- `prediction_1000x.npy` SHA-256:
  `54fefa80a9bd5a80e5e2a775ba32043bec8366c087b4a39f26b8d527afbcc32d`
- `old_teacher_1000x.npy` SHA-256:
  `e82821acbf0e48247012185a748559aa540f61f3770676b21c2f4f97a44e1db9`

This cross-capture comparison is not a valid native-domain promotion gate. The
1000x and 10x captures used different lighting, and the historical result is
retained only for diagnosis and reproducibility. Native 1000x results above
remain authoritative.

## Compatibility Changes

The initial journal recovery already reconstructed two required compatibility
changes:

1. `src/raw_fusion/finetune/noise.py` accepts packed `(4,H,W)` inputs required
   by the recovered A6 path.
2. `src/raw_fusion/finetune/a3.py` falls back to the recovered local official
   checkpoint and loads the original architecture module name safely.

This rebuild added the recovered post-HEAD frame-range behavior to
`src/raw_fusion/finetune/a4.py`:

- `_read_aligned_video(..., start_frame=...)`
- explicit range validation
- sequential A3/teacher model loading with CUDA-cache release between stages
- focused regression tests in `src/tests/test_recovery_a4_start_frame.py`

The recovered HEAD version is preserved at
`.recovery/evidence/original_source/a4.py.recovered_head` with SHA-256
`3b4612c417e1c1ab5a6b88cfcc545a60a3314e2bfe8265d555c31f68269e986d`.

## Verification

- complete test suite: `61 passed, 1 warning`
- warning: pre-existing PyTorch `torch.meshgrid` indexing warning
- `git diff --check`: passed
- rebuilt JSON files parsed successfully
- rebuilt NPY arrays opened with expected shapes and finite values
- rebuilt PNG panels decoded successfully
- A7 r7/r11 regenerated weights matched the surviving arrays byte for byte

Post-rebuild checksums are stored separately in
`.recovery/SHA256SUMS_POST_REBUILD`; the original
`.recovery/SHA256SUMS` was not modified.

## Remaining Irrecoverable Gaps

- Historical prediction arrays cannot generally be reproduced byte for byte
  across GPU runs; the 645x baseline is therefore verified by explicit metric
  tolerance.
- Some early A5 checkpoints and logs were already absent from the journal
  recovery, so those exact training runs cannot be resumed.
- One old Git tree object remains missing:
  `0cf8327baca79abf6e0bcd2d8f44c8c856536a04`. The recovered HEAD is complete.
- Anonymous damaged panel-sized objects remain quarantined under
  `.recovery/evidence/` and are not presented as valid images.

## Preserved Decision

- A3 step 5000 remains the Stage2 baseline.
- A5 and A6 candidates remain rejected by the disjoint static, motion, and
  native-domain gates.
- A7 remains a deterministic post-process investigation.
- No rebuilt artifact changes the decision in handoff
  `STAGE2-A5-FINAL-20261007`.
