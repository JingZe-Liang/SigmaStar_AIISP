# Stage2 DenoiseModel Handoff: A5 Motion Final Decision

Handoff ID: `STAGE2-A5-FINAL-20261007`

- Updated: 2026-10-07 (Asia/Shanghai)
- Project root: `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel`
- Git branch / HEAD: `master` / `3971228`
- Scope: Stage2 RAW denoising only
- Status: A5 motion experiments are complete and rejected; A3 step-5000 remains the final Stage2 baseline
- Running Stage2 jobs at handoff time: none
- Handoff state: materials verified; waiting for the next window to read and confirm

## Read This First

The Stage2 motion-fidelity investigation is finished. The final decision is:

> Keep A3 step-5000 as the Stage2 denoiser baseline. Do not promote any A5
> checkpoint produced by the motion-supervision experiments.

The authoritative baseline checkpoint is:

`artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`

Do not overwrite or replace this checkpoint. The latest A5 same-window
checkpoint is an experiment artifact only:

`artifacts/finetune_mis20s1/training_a5_same_window_teacher_flow1_r1/checkpoint_step_000020.pth`

The A5 candidate improved one motion diagnostic in the live weights, but it
did not satisfy the combined motion, static-quality, detail, temporal, and
native-domain promotion gates. EMA did not preserve the motion gain.

No new experiment has been authorized after this result. The next window
should first verify this handoff and ask the user for the next Stage2 objective
before changing code or starting CUDA work.

## Original Problem

The A3 denoiser works well in static regions, but the fixed-ISP 645x preview
shows a moving person being temporally averaged into the background. The A3
training objective excludes detected motion pixels from real static
supervision, while its synthetic temporal data repeats one clean image with
independent noise. This creates strong pressure to average static content but
provides little positive supervision for preserving genuine motion.

The motion diagnostic is implemented in:

`scripts/audit_motion_temporal_fidelity.py`

The relevant A3 application preview is under:

`artifacts/finetune_mis20s1/application_validation/input_vs_a3_step005000_fixed/`

The motion/input difference ratio is a diagnostic, not a clean-reference
quality metric. A larger value means the prediction follows input motion more
closely, but it is never sufficient by itself for promotion.

## Approved Evaluation Protocol

The approved motion confidence rule is fixed at:

`flow_magnitude_px >= 1.0`

Do not substitute the older low-flow masks.

- Dynamic training manifest: `artifacts/finetune_mis20s1/evaluation/a5_dynamic_train_flow1/manifest.json`
- Dynamic holdout manifest: `artifacts/finetune_mis20s1/evaluation/a5_dynamic_holdout_flow1/manifest.json`
- Training frame range: `[0, 160)`
- Disjoint 645x holdout range: `[160, 200)`
- Independent native holdout: 1000x frames `[0, 40)`

The A3 promotion-gate baselines are:

- 645x motion/input ratio: `0.48879945278167725`
- 645x static PSNR: `29.51644030829074 dB`
- 645x high-frequency ratio: `0.9738957024988305`
- 645x temporal MAD: `11.758078575134277 DN`
- native 1000x PSNR: `52.36373272976205 dB`
- native 1000x RMSE: `9.864322662353516 DN`
- legal RAW fraction: `1.0`

A candidate is promotable only if motion improves without regressing 645x
static PSNR, native PSNR/RMSE, legal RAW range, detail retention, static
temporal stability, or fixed-ISP visual quality.

## Final A5 Experiment

The final targeted experiment applied both objectives to the same 645x
dynamic window:

1. The strict flow-1 confidence mask supervised moving pixels with a
   Noise2Noise Charbonnier loss.
2. The motion mask was spatially dilated by one packed-resolution pixel to
   exclude uncertain motion boundaries.
3. Pixels outside the dilated mask were supervised against a frozen A3 EMA
   teacher output from the same dynamic input window.
4. Dynamic and static losses were logged separately.

Configuration:

- `static_target_mode: same_window_a3_teacher`
- `motion_boundary_radius: 1`
- `dynamic_weight: 1.0`
- `static_anchor_weight: 1.0`
- `learning_rate: 1e-5`
- `steps: 20`
- `trainable_scope: all`
- initialization: A3 step-5000 EMA

Authoritative configuration and smoke report:

- `artifacts/finetune_mis20s1/training_a5_same_window_teacher_flow1_r1/config.json`
- `artifacts/finetune_mis20s1/training_a5_same_window_teacher_flow1_r1/smoke_report.json`

The dynamic holdout loss changed as follows:

- before: `0.02995390968862921`
- live model: `0.029861595365218818`
- EMA: `0.029937291867099702`

This small training-objective improvement did not imply promotion. The
disjoint 645x and native 1000x evaluations below are authoritative.

## Final Metrics And Decision

| Weights | 645x motion ratio | 645x PSNR | 645x HF ratio | 645x temporal MAD | Native PSNR | Native RMSE | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A3 baseline | `0.48879945` | `29.516440` | `0.97389570` | `11.758079` | `52.363733` | `9.864323` | final baseline |
| A5 live | `0.49388218` | `29.485918` | `0.96100717` | `12.057498` | `52.385133` | `9.840049` | rejected |
| A5 EMA | `0.48808357` | `29.513075` | `0.97253010` | `11.699882` | `52.345669` | `9.884858` | rejected |

Interpretation:

- Live weights improved the motion ratio by about `0.00508`, but regressed
  645x static PSNR by about `0.03052 dB`, reduced high-frequency retention,
  and increased temporal MAD.
- EMA stayed closer to A3 on 645x static quality, but its motion ratio was
  worse than A3 and its native PSNR/RMSE regressed.
- Legal RAW fraction remained `1.0`.
- Neither weight set passed all numerical gates, so no fixed-ISP promotion
  review was prepared.

Final decision: **A5 same-window supervision failed promotion. Retain A3
step-5000 as the final Stage2 denoiser baseline.**

Authoritative evaluations:

- Live: `artifacts/finetune_mis20s1/evaluation/a5_same_window_teacher_flow1_r1_model/evaluation.json`
- EMA: `artifacts/finetune_mis20s1/evaluation/a5_same_window_teacher_flow1_r1_ema/evaluation.json`

Both directories also contain the generated 645x holdout and native 1000x
prediction arrays.

## Implemented Code

The final A5 implementation is in:

- `src/raw_fusion/finetune/a5_motion.py`
  - adds `static_target_mode="same_window_a3_teacher"`;
  - adds explicit `motion_boundary_radius` configuration;
  - implements frame-local spatial motion-mask dilation;
  - implements same-window frozen-A3 static anchoring;
  - handles an empty static mask without NaN;
  - keeps static and dynamic metrics separate.
- `scripts/run_a5_motion_smoke.py`
  - exposes the same-window mode and boundary radius.
- `scripts/evaluate_a5_motion_smoke.py`
  - evaluates live or EMA weights on disjoint 645x and native 1000x data;
  - now initializes the repository `src` path so it can run directly.
- `src/tests/test_finetune_a5_motion.py`
  - covers dilation, frame isolation, exclusion, empty-mask behavior, and
    configuration.
- `src/raw_fusion/finetune/a4.py`
  - `_read_aligned_video()` accepts `start_frame` so `[160, 200)` can be read
    without reusing training frames.

The focused regression suite passed:

```text
17 passed, 1 warning
```

The warning is the pre-existing `torch.meshgrid` indexing warning.

Reproduction command:

```bash
/home/jingzeliang/miniconda3/envs/aaa_312/bin/python -m pytest -q \
  src/tests/test_finetune_a3.py \
  src/tests/test_finetune_a4.py \
  src/tests/test_finetune_a5_native.py \
  src/tests/test_finetune_a5_motion.py
```

## Repository And Artifact State

The repository is on `master` at `3971228`, but the Stage2 work is not
committed.

Important worktree facts:

- `src/raw_fusion/finetune/a4.py` is a tracked modified file.
- The A5 scripts, A5 modules, tests, reports, configs, and handoff documents
  currently appear as untracked files.
- There are many pre-existing untracked Stage2 files beyond the final A5
  change. Do not delete, reset, or mass-stage them without explicit user
  approval.
- `artifacts/` is ignored by Git. The checkpoints and evaluation arrays exist
  locally but will not appear in normal `git status` output.
- No Stage2 training or evaluation process was running when this handoff was
  written.

Do not use destructive Git commands to clean the worktree. A future commit
must first identify the intended file set and exclude unrelated user work.

## Closed Routes

Do not repeat the following unless the user explicitly reopens the research
question with a materially new hypothesis:

- Do not promote any A5 smoke checkpoint.
- Do not scan more scalar values of `static_anchor_weight`.
- Do not rerun the same-window radius-1 experiment unchanged.
- Do not use old low-flow motion masks.
- Do not add native 1000x frames to A5 training.
- Do not overwrite the A3 step-5000 checkpoint.
- Do not describe a motion-ratio-only gain as an A5 improvement.
- Do not resume Stage1 FusionNet or downstream student work under this
  Stage2-only handoff.

The earlier design history remains available in:

- `docs/HANDOFF_2026-09-30_STAGE2_A5_MOTION.md`
- `docs/A5_MOTION_SUPERVISION_DESIGN_2026-09-25.md`
- `artifacts/finetune_mis20s1/evaluation/a5_motion_experiment_summary.json`

This 2026-10-07 document supersedes the earlier handoff's `Exact Next Task`:
that task was completed, evaluated, and rejected.

## First Action In The New Window

The new window should perform a read-only recovery check:

1. Confirm the project root and handoff ID.
2. Confirm that the A3 baseline checkpoint and both final evaluation JSON
   files exist.
3. Confirm there is no active Stage2 training/evaluation process.
4. Report that A5 is closed and A3 is the final baseline.
5. Ask the user which new objective to pursue. Do not infer a new experiment
   from the old A5 plan.

Suggested opening message for the new window:

```text
$project-handoff
请读取并核对：
/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel/docs/HANDOFF_2026-10-07_STAGE2_A5_FINAL.md

交接编号：STAGE2-A5-FINAL-20261007。
项目目录：/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel。
先只读核对 A3 最终基线、A5 最终失败结论、live/EMA 评估产物、未提交工作区状态和当前运行进程。本轮不要修改代码、不要启动实验。核对后告诉我差异，并等待我指定新的目标。
```

