# Stage2 DenoiseModel Handoff: Denoising-Only Focus

Handoff ID: `STAGE2-DENOISE-ONLY-20260924`
Date: 2026-09-24 (Asia/Shanghai)
Project root: `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel`
Handoff status: materials verified; no new handoff task created

## Highest-Priority Scope Rule

**Forget Stage1 FusionNet for the current project phase.** Stage1 FusionNet is
out of scope and must not be trained, modified, compared, promoted, or used as
the next-step objective. The current and only primary objective is training,
evaluating, and improving the Stage2 RAW denoising model.

The `downstream_a3_step005000*` artifacts were produced as a diagnostic check
of a previously discussed integration path. They are historical evidence of a
domain-contract issue, not the current workstream. Do not resume that
downstream experiment unless the user explicitly re-authorizes it in a new
phase.

## Current Model

The current denoising model is the A3 EMA checkpoint at step 5000:

`artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`

Training contract:

- Real training gains: `128x`, `313x`, `632x`, and `645x`.
- Synthetic calibrated pairs are mixed into training.
- `1000x` is excluded from training and retained as an independent holdout.
- Model input/output is packed RGGB RAW with the existing A3 normalization and
  temporal/spatial inference contract.

## Verified Denoising Results

Native 1000x evaluation:

- Input: `44.6204 dB`, `24.0565 DN RMSE`
- A3: `53.2675 dB`, `8.8895 DN RMSE`
- Legal fraction: `1.0`
- Static fraction: `0.9222545`
- Flat residual sigma: `7.1652 DN`

Mutually exclusive native split-half validation:

- Odd input vs even reference: A3 `53.1041 dB`
- Even input vs odd reference: A3 `53.0711 dB`

Existing 645x cross-gain evaluation:

- Input: `24.3519 dB`
- A3: `29.5729 dB`

These metrics are model-evaluation evidence. They are not downstream student
metrics and should not be replaced by the historical FusionNet artifacts.

## Where To See The Current Denoiser

For direct fixed-ISP application visualization of the Stage2 denoiser:

`artifacts/finetune_mis20s1/application_validation/input_vs_a3_step005000_fixed/`

Relevant files:

- `input_vs_a3_pseudogt_128x.mp4`
- `input_vs_a3_pseudogt_645x.mp4`
- `input_vs_a3_pseudogt_128x_frame0000.png`
- `input_vs_a3_pseudogt_645x_frame0000.png`
- `manifest.json`

For the independent 1000x native-domain result:

`artifacts/finetune_mis20s1/evaluation/native_1000x/native_four_panel_1000x.png`

The `pseudo_gt/a3_step005000/` files are the exported A3 denoiser outputs in
the downstream RAW file contract. They are still useful as denoiser artifacts,
but the direct visual judgment should use the fixed-ISP application videos
above.

## Important Domain Finding

The A3 pseudo-GT used for direct denoiser visualization is in the A1/10x
evaluation domain. It must not be silently interpreted as a native-domain
target for another model. A later diagnostic experiment demonstrated that
feeding this domain directly into a different pipeline caused a strong purple
cast at 645x. That finding is recorded only as a boundary condition; it does
not reopen Stage1 work.

Do not alter the A3 target/domain contract merely to accommodate Stage1. Any
future change to the denoiser training domain requires a new supervision design,
disjoint evaluation split, and explicit user approval.

## Completed Work

- A3 step-5000 training completed.
- Full native 1000x evaluation completed using a same-gain temporal reference.
- Native split-half validation completed.
- A3 pseudo-GT export completed for `128x` and `645x`.
- Fixed-ISP direct input-versus-A3 visual validation completed.
- The MP4 encoding path was corrected to `libx265`, `yuv444p`, `lossless=1`;
  the earlier `yuv420p` videos with magenta row artifacts are invalid.
- Full Stage2 test suite: `34 passed`, one pre-existing torch.meshgrid warning.

## Explicitly Out Of Scope

- Stage1 FusionNet training or retraining.
- Stage1 pseudo-GT replacement or downstream student promotion.
- Any comparison whose purpose is to select a FusionNet checkpoint.
- Adding real `1000x` frames to A3 training without a new disjoint native-domain
  supervision design.
- Treating the invalid cross-capture `1000x -> 10x` fit as a denoiser quality
  gate.

## Next Work, Denoising Only

The next task must start from the A3 denoiser, not from any downstream model:

1. Review A3 denoiser failure cases in native 1000x and existing 645x metrics,
   especially flat-region residuals, highlight behavior, temporal stability,
   and detail retention.
2. If a concrete denoising failure is confirmed, design an A5/A3-iteration
   supervision change with a new disjoint holdout before editing training data.
3. Otherwise, keep A3 step-5000 as the candidate denoiser and expand only the
   denoiser evaluation set or application visual checks.
4. Re-run the Stage2 tests and regenerate only Stage2 denoiser artifacts after
   any model or preprocessing change.

Acceptance remains based on native-domain improvement, legal RAW output,
temporal/detail checks, and fixed-ISP visual sanity. Stage1 FusionNet is not an
acceptance gate for this phase.

## Authoritative References

- `docs/A3_BASELINE_2026-09-23.md`
- `docs/A5_TARGET_DOMAIN_DECISION_2026-09-23.md`
- `docs/A3_INPUT_APPLICATION_VALIDATION_2026-09-23.md`
- `docs/A3_DOWNSTREAM_INFERENCE_2026-09-23.md`
- `artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`
- `artifacts/finetune_mis20s1/evaluation/native_1000x/native_1000x_metrics.json`

The downstream validation record
`docs/A3_DOWNSTREAM_STUDENT_VALIDATION_2026-09-24.md` is retained for historical
diagnosis only and does not define the current project scope.
