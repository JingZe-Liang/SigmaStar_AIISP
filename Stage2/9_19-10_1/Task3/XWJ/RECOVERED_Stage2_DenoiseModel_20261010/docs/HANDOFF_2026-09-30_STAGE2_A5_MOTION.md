# Stage2 Denoise Handoff: A5 Motion Fidelity

Handoff ID: `STAGE2-A5-MOTION-20260930`
Date: 2026-09-30 (Asia/Shanghai)
Project root: `/Data2/Bad_Example_Do_Not_Follow/Jaime/Projects/Stage2_DenoiseModel`
Scope: Stage2 RAW denoising only
Status: handoff materials saved; A3 remains the production baseline

## Read This First

The active problem is motion ghosting in the Stage2 A3 denoiser. The fixed-ISP
645x application preview shows a moving person being temporally averaged into
the background. Static-region denoising is useful, but motion content is not
preserved strongly enough.

The current production/reference checkpoint is:

`artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`

This is the A3 step-5000 checkpoint. Do not replace it, overwrite it, or treat
an A5 smoke checkpoint as production until the gates below are passed.

Stage1 FusionNet and downstream student work are out of scope. Do not resume
those experiments in the new session.

## Problem And Evidence

The original A3 training objective excludes detected motion pixels from real
static supervision. The synthetic temporal cache repeats one clean image with
independent noise, so it provides no positive moving-content target. The model
therefore receives strong temporal-averaging pressure in static regions and
little supervision for preserving moving objects.

The motion audit is implemented in:

`scripts/audit_motion_temporal_fidelity.py`

The A3 failure is visible in the fixed-ISP application artifacts under:

`artifacts/finetune_mis20s1/application_validation/input_vs_a3_step005000_fixed/`

The existing 645x holdout diagnostic uses frames `[160, 200)` and reports an
A3 motion/input difference ratio of:

`0.48879945278167725`

This ratio is diagnostic rather than a clean-reference metric. A higher value
means the prediction follows input motion more closely. It must be read
together with static PSNR, temporal stability, detail retention, and native
1000x quality.

## Approved Dynamic Target

The first motion-mask probes admitted too many low-flow pixels. The audit found
that as much as `77%` of confidence pixels in one window had flow below `1 px`.
The approved diagnostic rule is therefore:

`flow_magnitude_px >= 1.0`

The strict probes are stored separately and must not be confused with the old
probes:

- Training windows: `artifacts/finetune_mis20s1/evaluation/a5_dynamic_train_flow1/manifest.json`
- Dynamic holdout windows: `artifacts/finetune_mis20s1/evaluation/a5_dynamic_holdout_flow1/manifest.json`
- Training frame range: `[0, 160)`
- Dynamic holdout frame range: `[160, 200)`
- Native holdout: `1000x`, frames `[0, 40)`

The dynamic data construction and validation scripts are:

- `scripts/build_dynamic_n2n_probe.py`
- `scripts/build_dynamic_window_probe.py`
- `scripts/validate_dynamic_n2n_probe.py`
- `scripts/validate_dynamic_window_probe.py`
- `scripts/diagnose_motion_mask_noise.py`
- `scripts/render_motion_mask_variants.py`

The strict flow-1 probe remains a target-construction diagnostic. It is not a
claim that the warped target is a clean reference.

## Implemented A5 Smoke Code

The bounded smoke implementation is in:

- `src/raw_fusion/finetune/a5_motion.py`
- `scripts/run_a5_motion_smoke.py`
- `scripts/evaluate_a5_motion_smoke.py`
- `src/tests/test_finetune_a5_motion.py`

`src/raw_fusion/finetune/a4.py` was extended so `_read_aligned_video()` can
read a contiguous segment with `start_frame`. This is required to evaluate the
disjoint `[160, 200)` holdout without reusing training frames.

The A5 smoke supports:

- `static_target_mode="a1"`: original A1 static target;
- `static_target_mode="a3_teacher"`: frozen copy of the A3 EMA output for the
  static batch;
- `trainable_scope="all"` or `"mutual_attention"`;
- `static_anchor_weight`, default `1.0`.

The teacher-anchor implementation still trains one separate A1/static batch
and one dynamic window batch per step. It does not yet anchor static pixels of
the same dynamic window. That distinction is the reason for the next design.

## Experiment Results

Baseline values on the disjoint evaluation protocol:

- 645x motion/input ratio: `0.48879945`
- 645x static PSNR: `29.516440 dB`
- native 1000x PSNR: `52.363733 dB`
- legal RAW fraction: `1.0`

All smoke runs start from the A3 step-5000 EMA weights and run 20 steps.

| Candidate | Motion ratio | 645x PSNR | Native PSNR | Decision |
| --- | ---: | ---: | ---: | --- |
| A1 static control, lr `1e-5` | `0.48589` | `29.55408` | `51.01608` | control only; native regression |
| Full model, lr `1e-5`, dynamic weight `0.1` | `0.49424` | `29.52520` | `51.05544` | rejected; native regression |
| Full model, lr `1e-6`, dynamic weight `1.0` | `0.48927` | `29.51066` | `52.28196` | rejected; small motion gain and static/native regression |
| Mutual attention only, lr `1e-5`, dynamic weight `1.0` | `0.48794` | `29.51175` | `52.33929` | rejected; no motion gain |
| A3 teacher, anchor weight `1.0`, live model | `0.49462` | `29.48915` | `52.39362` | promising, not promotable |
| A3 teacher, anchor weight `1.0`, EMA | `0.48845` | `29.51155` | `52.36988` | rejected; no motion gain |
| A3 teacher, anchor weight `2.0`, live model | `0.49159` | `29.50114` | `52.40357` | rejected; static/detail regression |
| A3 teacher, anchor weight `2.0`, EMA | `0.48816` | `29.51281` | `52.37002` | rejected; no motion gain |

Interpretation:

1. Continuing the original A1 static reconstruction objective at lr `1e-5`
   causes most of the unsafe native drift, even without dynamic supervision.
2. Mutual-attention-only updates do not produce the required motion gain.
3. The teacher-anchor direction is the first one with a clear live-model
   motion gain and native-domain gain.
4. Increasing the separate static anchor weight from `1.0` to `2.0` reduces
   the 645x PSNR regression from `-0.027 dB` to `-0.015 dB`, but halves the
   motion gain and still lowers the 645x high-frequency ratio.
5. Scalar anchor-weight scanning is therefore closed. A3 remains the baseline.

The complete machine-readable summary is:

`artifacts/finetune_mis20s1/evaluation/a5_motion_experiment_summary.json`

The detailed design record is:

`docs/A5_MOTION_SUPERVISION_DESIGN_2026-09-25.md`

The latest checkpoints and evaluations are:

- `artifacts/finetune_mis20s1/training_a5_teacher_anchor_flow1_staticw2/`
- `artifacts/finetune_mis20s1/evaluation/a5_teacher_anchor_flow1_staticw2_model_weights/`
- `artifacts/finetune_mis20s1/evaluation/a5_teacher_anchor_flow1_staticw2_ema/`

## Promotion Gates

An A5 candidate is promotable only if the same disjoint protocol shows all of
the following:

- motion/input ratio improves over `0.48879945`;
- 645x static PSNR does not regress from `29.516440 dB`;
- native 1000x PSNR and RMSE do not regress;
- legal RAW fraction remains `1.0`;
- high-frequency/detail retention does not regress materially;
- static temporal stability does not regress materially;
- fixed-ISP visual review does not show new ghosting, color, or highlight
  failures.

The motion ratio alone is never sufficient for promotion.

## Exact Next Task

Implement one targeted follow-up that uses the same 645x dynamic window for
both populations:

1. Use the strict-flow-1 dynamic confidence mask for the moving-content
   Noise2Noise loss.
2. Dilate that mask by a small, explicit spatial radius to form an uncertain
   motion-boundary exclusion band.
3. On the remaining static pixels of the same dynamic window, supervise the
   prediction against the frozen A3 teacher output.
4. Keep the dynamic and static losses reported separately.
5. Do not increase the scalar `static_anchor_weight` again.
6. Keep the training/holdout split unchanged: `[0,160)` versus `[160,200)`.
7. Run one bounded smoke first, then evaluate live and EMA with the existing
   `scripts/evaluate_a5_motion_smoke.py` protocol.

The purpose is to test whether same-domain static anchoring preserves 645x
background detail while leaving the confirmed moving region free to learn.
The boundary dilation radius must be recorded in the config and report. Start
with the smallest defensible radius supported by the packed-resolution mask;
do not silently change the motion rule or holdout.

Before changing code, inspect the current `compute_loss()` mask semantics in
`src/raw_fusion/finetune/a3.py` and the saved dynamic sample shapes. Preserve
the existing `(T, C, H, W)` source/target layout and `(T, 1, H, W)` confidence
layout.

Suggested first implementation files:

- modify `src/raw_fusion/finetune/a5_motion.py`;
- extend `scripts/run_a5_motion_smoke.py` only for explicit new parameters;
- add focused tests in `src/tests/test_finetune_a5_motion.py` for mask dilation,
  exclusion, and empty-mask behavior.

Do not start a long run before the focused test passes.

## Evaluation Commands

Use the project Python environment:

`/home/jingzeliang/miniconda3/envs/aaa_312/bin/python`

Check GPU state from the host environment before starting CUDA work. The
sandbox previously reported no CUDA while host-side PyTorch had CUDA. The
4090 was also occupied during the latest run; the successful static-weight-2
run used PyTorch `cuda:1`, which mapped to the RTX 3090.

Run the focused regression suite after code changes:

```bash
/home/jingzeliang/miniconda3/envs/aaa_312/bin/python -m pytest -q \
  src/tests/test_finetune_a3.py \
  src/tests/test_finetune_a4.py \
  src/tests/test_finetune_a5_native.py \
  src/tests/test_finetune_a5_motion.py
```

The previous implementation passed `13 tests` with one pre-existing
`torch.meshgrid` warning.

## Do Not Repeat

- Do not promote either teacher-anchor checkpoint.
- Do not resume Stage1 FusionNet or downstream student experiments.
- Do not add native 1000x frames to A5 training.
- Do not use the old low-flow motion masks as training targets.
- Do not scan more scalar values of `static_anchor_weight` before testing the
  same-window static-mask design.
- Do not overwrite the A3 step-5000 checkpoint.

## Completion Condition For The Next Session

The next session should end with one of two explicit outcomes:

1. The same-window static-mask candidate passes the numerical gates and a
   short fixed-ISP visual review is prepared for dad; or
2. The candidate fails, its exact metrics and failure mode are recorded here,
   and A3 is retained as the final Stage2 denoiser baseline.

No result should be described as an A5 improvement without the disjoint 645x
motion/static evaluation and the independent native 1000x evaluation.

## Result Of Next Session (2026-10-01)

The same-window static-mask smoke was implemented and run from the A3 step-5000
EMA checkpoint with:

- `static_target_mode: same_window_a3_teacher`
- `motion_boundary_radius: 1` packed-resolution pixel
- `dynamic_weight: 1.0`
- `static_anchor_weight: 1.0`
- learning rate `1e-5`, `20` steps, full-model training
- training/holdout split unchanged: `[0,160)` versus `[160,200)`

Artifacts:

- Checkpoint: `artifacts/finetune_mis20s1/training_a5_same_window_teacher_flow1_r1/checkpoint_step_000020.pth`
- Live evaluation: `artifacts/finetune_mis20s1/evaluation/a5_same_window_teacher_flow1_r1_model/evaluation.json`
- EMA evaluation: `artifacts/finetune_mis20s1/evaluation/a5_same_window_teacher_flow1_r1_ema/evaluation.json`

| Weights | 645x motion ratio | 645x PSNR | 645x high-frequency ratio | Native PSNR | Native RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| A3 baseline | `0.48879945` | `29.516440 dB` | `0.97389570` | `52.363733 dB` | `9.864323` |
| A5 live | `0.49388218` | `29.485918 dB` | `0.96100717` | `52.385133 dB` | `9.840049` |
| A5 EMA | `0.48808357` | `29.513075 dB` | `0.97253010` | `52.345669 dB` | `9.884858` |

Decision: **failed promotion gates; A3 remains the final Stage2 denoiser
baseline.** The live model improved motion fidelity but regressed 645x static
PSNR, high-frequency retention, and temporal stability. The EMA model preserved
645x quality more closely but did not improve motion fidelity and regressed
native PSNR/RMSE. Legal RAW fraction remained `1.0` for all evaluations.

No fixed-ISP visual promotion review was prepared because neither weight set
passed the numerical gates.
