# A7 Existing-Data Deterministic Fusion Final Decision

- Date: 2026-10-11 (Asia/Shanghai)
- Handoff: `STAGE2-RECOVERY-READY-20261011`
- Project root: `/Data4/Xiao/RECOVERED_Stage2_DenoiseModel_20261010`
- Scope: recovered A7 deterministic restricted motion-region fusion only
- Baseline retained: `artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`

## Final Decision

**Reject all 36 recovered A7 configurations. No r3, r7, or r11 setting is
retained as a deployment-time optional post-process. A3 step-5000 remains the
only accepted Stage2 denoiser baseline.**

All A7 settings preserve the measured static region, but none provides a
stable, visually credible improvement to the moving person's body or motion
edges under the common fixed-ISP review. The strongest diagnostic result,
`r11 / alpha=0.4 / residual_clip=96 DN`, raises motion/input ratio from
`0.488790452` to `0.493268609` (`+0.004478157`, or `+0.9162%` relative), while
moving the A3 output by `9.116125 DN` on average and `38.385632 DN` at P95 in
the motion region. The visible result mainly moves local texture toward the
old teacher; it does not recover a consistently clearer silhouette, limb
edge, or body detail. Because the old teacher is not clean dynamic truth, the
motion/input-ratio increase cannot justify retaining this post-process.

## Recovery And Safety Check

- The A3 checkpoint exists, is unchanged, and has SHA-256
  `97c4332f685f98959b243e4b100434469f714aed6329e100d22b6d8552592e0b`,
  matching `.recovery/SHA256SUMS`.
- `RECOVERY_REBUILD_20261010.md` and
  `.recovery/SHA256SUMS_POST_REBUILD` are present and were reviewed.
- The r3, r7, and r11 `evaluation.json` files, fusion weights, prediction
  previews, and recovered fixed-ISP panels are present.
- The r7 and r11 weight hashes match the post-rebuild checksum record:
  `d67685094cf4634f62c64ed0ddcd1512d21c9ed187ccf76f3b58d0a1f4ab626f`
  and `7ca6f1ddd42ca0d5d1203e06a4ac07256f843aadd942cb45c271b73117d4310b`.
  The surviving r3 weight hash is
  `0a9a41db6afa3687aaf75e1d12e83d3748135a8729b7c9ffffeea3c8da0f3462`.
- All three weight arrays have shape `(40, 540, 960)`, dtype `float16`, finite
  values, and range `[0, 1]`. `static_nonzero_weights` is zero in all three
  evaluation reports.
- No process belonging to this recovered Stage2 project, A7 evaluator, or
  `finetune_mis20s1` was active during the initial or final process checks.
  Other unrelated GPU training processes were present and were not touched.
- No model training was started. The dataset under
  `/HardDisk/jingzeliang/Data/Company_Dataset/dataset` was only read by the
  pre-existing recovered workflow and was not modified in this final review.

### GitHub Partial Snapshot Check

GitHub `main` resolves to the documented commit
`9964f776dd7ebd015a61f4fd416989b3cf4aba85`. The exclusion TSV downloaded from
that exact commit is byte-identical to the local
`GITHUB_UPLOAD_EXCLUDED_LARGE_FILES_20261011.tsv` and has SHA-256
`cf958d3ebc54bc080b68c7909655da6d7b8d41382307c990db7a3781bdefdc81`.

The list contains 53 files totaling `22,379,264,218` bytes
(`20.842313970 GiB`). Every listed large file exists in the local recovery
tree and matches the documented byte size; none is available from the GitHub
snapshot by design. The local recovery tree therefore remains the complete
runtime artifact source.

The recovered local Git database is based on older commit `3971228c` and does
not contain the GitHub backup commit object. It also has many expected
recovery-era modified and untracked files. This is a version-control recovery
difference, not an A7 artifact gap, and no destructive Git operation was used.

### Non-Blocking Recovered-Artifact Differences

- The historical r3 JSON still embeds stale `/Data2/...` paths. The actual
  files used here are under the authoritative `/Data4/Xiao` recovery root.
- The historical r3 A3 fingerprint is `29.516440308 dB`; the rebuilt r7/r11
  reports use the tolerance-equivalent baseline at `29.516401586 dB`. The
  `0.000038722 dB` difference is already covered by the recovery audit and
  does not affect the decision.
- JSON fusion coverage is computed before saving the weights as `float16`.
  Very small feather values can quantize to zero in the saved arrays, so the
  loaded-array nonzero fractions are slightly lower than the JSON coverage.

## Evaluation Protocol

The recovered reports evaluate frames `[160, 200)` from the 645x holdout.
Fusion injects a clipped old-teacher residual into A3 only where the recovered
flow/motion weight is nonzero:

`candidate = A3 + alpha * weight * clip(old_teacher - A3, +/- residual_clip_dn)`

The sweep contains `alpha in {0.1, 0.2, 0.3, 0.4}` and
`residual_clip_dn in {32, 64, 96}` for each dilation radius. Static PSNR,
high-frequency retention, temporal MAD, legal RAW fraction, static nonzero
weights, and fusion coverage are invariant across alpha/clip within a radius,
so they are reported once per radius below. Motion shifts are mean/P95;
static shifts are mean/maximum.

## Radius-Level Static And Coverage Metrics

| Radius | A3 motion/input | Static PSNR dB | HF ratio | Temporal MAD DN | Legal RAW | Static nonzero | Fusion coverage | Weight mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r3 | 0.488799453 | 29.516440308 | 0.973895702 | 11.758078575 | 1.000000 | 0 | 1.969551% | 0.011087152 |
| r7 | 0.488790452 | 29.516401586 | 0.973880264 | 11.759798050 | 1.000000 | 0 | 2.437664% | 0.015677957 |
| r11 | 0.488790452 | 29.516401586 | 0.973880264 | 11.759798050 | 1.000000 | 0 | 2.808439% | 0.019232320 |

Relative to r3, r7 expands fusion coverage by `23.77%` and r11 by `42.59%`.
The expansion remains inside the recovered A0 motion mask according to the
static guard, but it exposes a larger motion-labeled area to old-teacher
residuals.

## Complete Alpha And Residual-Clip Sweep

| Radius | Alpha | Clip DN | Motion/input | Gain vs A3 | Relative gain | Motion shift DN mean/P95 | Static shift DN mean/max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r3 | 0.1 | 32 | 0.488970518 | +0.000171065 | +0.0350% | 0.568106 / 3.135155 | 0.000011641 / 0.000228219 |
| r3 | 0.1 | 64 | 0.489161462 | +0.000362009 | +0.0741% | 0.992041 / 6.177131 | 0.000011641 / 0.000228219 |
| r3 | 0.1 | 96 | 0.489341408 | +0.000541955 | +0.1109% | 1.331835 / 9.009785 | 0.000011641 / 0.000228219 |
| r3 | 0.2 | 32 | 0.489198804 | +0.000399351 | +0.0817% | 1.136196 / 6.270310 | 0.000011641 / 0.000228219 |
| r3 | 0.2 | 64 | 0.489664435 | +0.000864983 | +0.1770% | 1.984064 / 12.354368 | 0.000011641 / 0.000228219 |
| r3 | 0.2 | 96 | 0.490089536 | +0.001290083 | +0.2639% | 2.663650 / 18.019512 | 0.000011641 / 0.000228219 |
| r3 | 0.3 | 32 | 0.489478052 | +0.000678599 | +0.1388% | 1.704286 / 9.405509 | 0.000011641 / 0.000228219 |
| r3 | 0.3 | 64 | 0.490269870 | +0.001470417 | +0.3008% | 2.976089 / 18.531507 | 0.000011641 / 0.000228219 |
| r3 | 0.3 | 96 | 0.490972370 | +0.002172917 | +0.4445% | 3.995469 / 27.029297 | 0.000011641 / 0.000228219 |
| r3 | 0.4 | 32 | 0.489800751 | +0.001001298 | +0.2048% | 2.272374 / 12.540676 | 0.000011641 / 0.000228219 |
| r3 | 0.4 | 64 | 0.490947396 | +0.002147943 | +0.4394% | 3.968112 / 24.708729 | 0.000011641 / 0.000228219 |
| r3 | 0.4 | 96 | 0.491947800 | +0.003148347 | +0.6441% | 5.327285 / 36.039066 | 0.000011641 / 0.000228219 |
| r7 | 0.1 | 32 | 0.489015967 | +0.000225514 | +0.0461% | 0.799542 / 3.196386 | 0.000011639 / 0.000228219 |
| r7 | 0.1 | 64 | 0.489261538 | +0.000471085 | +0.0964% | 1.392292 / 6.383930 | 0.000011639 / 0.000228219 |
| r7 | 0.1 | 96 | 0.489494354 | +0.000703901 | +0.1440% | 1.867628 / 9.553806 | 0.000011639 / 0.000228219 |
| r7 | 0.2 | 32 | 0.489307135 | +0.000516683 | +0.1057% | 1.599069 / 6.392806 | 0.000011639 / 0.000228219 |
| r7 | 0.2 | 64 | 0.489895701 | +0.001105249 | +0.2261% | 2.784568 / 12.767815 | 0.000011639 / 0.000228219 |
| r7 | 0.2 | 96 | 0.490434676 | +0.001644224 | +0.3364% | 3.735237 / 19.107668 | 0.000011639 / 0.000228219 |
| r7 | 0.3 | 32 | 0.489658386 | +0.000867933 | +0.1776% | 2.398599 / 9.589207 | 0.000011639 / 0.000228219 |
| r7 | 0.3 | 64 | 0.490650237 | +0.001859784 | +0.3805% | 4.176845 / 19.151731 | 0.000011639 / 0.000228219 |
| r7 | 0.3 | 96 | 0.491535187 | +0.002744734 | +0.5615% | 5.602845 / 28.661449 | 0.000011639 / 0.000228219 |
| r7 | 0.4 | 32 | 0.490060657 | +0.001270205 | +0.2599% | 3.198127 / 12.785600 | 0.000011639 / 0.000228219 |
| r7 | 0.4 | 64 | 0.491491199 | +0.002700746 | +0.5525% | 5.569118 / 25.535629 | 0.000011639 / 0.000228219 |
| r7 | 0.4 | 96 | 0.492749751 | +0.003959298 | +0.8100% | 7.470461 / 38.215336 | 0.000011639 / 0.000228219 |
| r11 | 0.1 | 32 | 0.489053339 | +0.000262886 | +0.0538% | 0.977904 / 3.199865 | 0.000011639 / 0.000228219 |
| r11 | 0.1 | 64 | 0.489337534 | +0.000547081 | +0.1119% | 1.700245 / 6.399312 | 0.000011639 / 0.000228219 |
| r11 | 0.1 | 96 | 0.489605159 | +0.000814706 | +0.1667% | 2.279044 / 9.596408 | 0.000011639 / 0.000228219 |
| r11 | 0.2 | 32 | 0.489384919 | +0.000594467 | +0.1216% | 1.955793 / 6.399730 | 0.000011639 / 0.000228219 |
| r11 | 0.2 | 64 | 0.490054488 | +0.001264036 | +0.2586% | 3.400481 / 12.798621 | 0.000011639 / 0.000228219 |
| r11 | 0.2 | 96 | 0.490668863 | +0.001878411 | +0.3843% | 4.558071 / 19.192854 | 0.000011639 / 0.000228219 |
| r11 | 0.3 | 32 | 0.489779413 | +0.000988960 | +0.2023% | 2.933687 / 9.599679 | 0.000011639 / 0.000228219 |
| r11 | 0.3 | 64 | 0.490900815 | +0.002110362 | +0.4318% | 5.100713 / 19.197931 | 0.000011639 / 0.000228219 |
| r11 | 0.3 | 96 | 0.491904885 | +0.003114432 | +0.6372% | 6.837107 / 28.789261 | 0.000011639 / 0.000228219 |
| r11 | 0.4 | 32 | 0.490228057 | +0.001437604 | +0.2941% | 3.911577 / 12.799575 | 0.000011639 / 0.000228219 |
| r11 | 0.4 | 64 | 0.491841197 | +0.003050745 | +0.6241% | 6.800945 / 25.597242 | 0.000011639 / 0.000228219 |
| r11 | 0.4 | 96 | 0.493268609 | +0.004478157 | +0.9162% | 9.116125 / 38.385632 | 0.000011639 / 0.000228219 |

## Fixed-ISP Visual Review

The pre-existing panels were structurally valid but compared one radius at a
time and only rendered `alpha=0.3 / clip=64 DN`. A unified deterministic
renderer was therefore added at:

`scripts/render_a7_final_comparison.py`

It reads the existing A3 prediction, old-teacher prediction, black levels,
and recovered r3/r7/r11 weight arrays. It does not run a model. The fixed ISP
is the same deterministic Bayer RG demosaic plus gamma path used by the
recovered A7 evaluator.

The generated evidence is under:

`artifacts/finetune_mis20s1/evaluation/a7_motion_fusion_final_review_20261011/`

`review_manifest.json` records all source paths, weight hashes, settings,
frames, crop coordinates, and output paths. Review coverage is:

- frames 167 through 173: labeled person/motion crops and r3/r7/r11 weight
  panels;
- frames 168, 170, and 172: full-frame horizontal panels;
- settings `alpha=0.3 / clip=64 DN` and `alpha=0.4 / clip=96 DN`;
- columns: A3, old teacher, r3, r7, r11.

Representative files:

- `frame_0168_preview_full.png`
- `frame_0170_strongest_full.png`
- `frame_0172_strongest_person_crop.png`
- `frame_0168_weights.png`
- `frame_0172_weights.png`

### Visual Findings

1. **Body and motion edges:** neither the preview setting nor the strongest
   setting produces a stable visible improvement in face, arm, torso, shorts,
   leg, or shoe boundaries over A3 across frames 167-173. Differences among
   r3/r7/r11 are close to imperceptible at normal fixed-ISP viewing scale.
2. **Teacher noise and false detail:** the old teacher has visibly different
   local texture on the white shirt, shorts, limbs, and moving edges. Stronger
   A7 settings move those regions toward that texture, but there is no clean
   dynamic target proving it is recovered detail rather than teacher noise or
   temporal residue.
3. **Ghosting and edge fracture:** no large new double image or gross edge
   break is visible in the reviewed frames. This is a static-safety result,
   not a positive motion-fidelity result.
4. **Motion-boundary behavior:** no conspicuous boundary step is visible at
   the reviewed display scale. However, the final weight is hard-constrained
   back to the binary motion support after feathering, and the strongest rows
   reach approximately `38 DN` P95 motion shift. There is no visible benefit
   that justifies accepting this boundary risk.
5. **Static consistency:** background regions classified static remain
   visually consistent with A3, matching `static_nonzero_weights=0`, unchanged
   static PSNR/HF/temporal metrics, and sub-millidigital-number numerical
   round-trip shifts.
6. **r7/r11 expansion:** r7 and r11 visibly broaden the weight support over
   the moving person and other motion-labeled regions. They improve only the
   diagnostic ratio and do not reveal additional credible edge detail. The
   larger support therefore increases old-teacher exposure without a
   compensating fixed-ISP benefit.

## Reproduction

From the project root, using the documented Python 3.12 environment:

```bash
/home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/render_a7_final_comparison.py
```

This regenerates the unified panels and `review_manifest.json` from existing
predictions and weights. It does not access a checkpoint or start CUDA work.

## Verification

- `scripts/render_a7_final_comparison.py` passed `py_compile` and completed
  successfully.
- A second render from the same sources produced byte-identical SHA-256 lists
  for all 27 PNG files and a byte-identical `review_manifest.json`.
- The manifest parsed successfully and every listed PNG decoded as a
  non-empty three-channel image.
- An independent structured check matched all 36 Markdown sweep rows against
  the three source `evaluation.json` files within their printed precision.
- The focused A7 tests passed: `4 passed`.
- The complete repository test suite passed: `61 passed, 1 warning` in
  `353.20 s`. The warning is the pre-existing PyTorch `torch.meshgrid`
  indexing warning documented by the recovery report.
- Whitespace/error checks passed for the new script and report.

## Completion And Remaining Items

The A7 existing-data deterministic fusion task is complete and has no
remaining execution item within the authorized scope. All recovered sweep
rows were compared, fixed-ISP evidence was reviewed, the result is
reproducible, and the decision is explicit.

The unresolved scientific limitation is the absence of clean dynamic truth
and the inability to recapture data. That is not a pending A7 action: under
the current constraints it is the reason a small diagnostic ratio increase
cannot be promoted. A3 step-5000 remains unchanged and must not be replaced by
any A7 output.
