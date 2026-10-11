# A5 Motion Supervision Design

Date: 2026-09-25 (Asia/Shanghai)

Status: stronger teacher-anchor smoke completed on 2026-09-29; candidate rejected, A3 remains baseline.

## Confirmed A3 Failure

The A3 static-region metrics are strong, but the fixed-ISP 645x application
preview shows motion ghosting around the moving person. The saved A3 prediction
was audited with `scripts/audit_motion_temporal_fidelity.py`.

The audit found:

- motion mask coverage of about `3.91%` at packed resolution;
- A3 motion-region frame differences at only `27%` to `32%` of the noisy input;
- several transitions with a prediction/input change ratio near `0.10`;
- static-region metrics remain improved, so the failure is motion fidelity,
  not a general failure to denoise.

This is a model behavior failure, not an MP4 encoding artifact.

## Root Cause

The current real-data sampler excludes every pixel marked as motion from the
training mask. The loss then applies the reconstruction, gradient, and high-pass
terms only to the remaining static pixels. The synthetic temporal cache repeats
one clean image with independent noise, so it contains no moving content.

The temporal model therefore has no positive supervision telling it to preserve
moving content. Temporal averaging is rewarded in static regions and is
unconstrained on motion regions.

## Supervision Contract For A5

The existing static A1 target/domain remains unchanged. `1000x` remains an
independent native holdout and must not be added to training.

A5 should use two explicitly separated supervision populations:

1. Static population: retain the current A1 temporal-mean target and current
   static/saturation mask.
2. Dynamic population: add a target that follows per-frame content. A candidate
   implementation is a same-stream Noise2Noise pair with robust motion-aware
   alignment and a confidence mask. A synthetic dynamic sequence with known
   foreground motion can be used as a controlled auxiliary population, but a
   temporal mean must never be used as its moving-object target.

The dynamic loss must be reported separately from the static loss. It must not
silently turn motion pixels back into A1 temporal-mean supervision.

## Dynamic Target Probe

The first probe was built with:

`scripts/build_dynamic_n2n_probe.py`

It uses DIS optical flow to warp a later 645x frame into the current-frame
coordinates. The probe is restricted to the candidate training range and does
not read the `[160, 200)` holdout. The current samples are:

- `34 -> 38`
- `50 -> 54`
- `148 -> 152`

The saved samples and previews are under
`artifacts/finetune_mis20s1/evaluation/dynamic_n2n_probe_645x/`.

The probe is technically usable for a smoke experiment:

- confidence covers `37%` to `44%` of the detected dynamic pixels;
- forward/backward flow error P95 is `1.16` to `1.28` packed pixels;
- warped green-channel photometric error P95 is about `0.072` to `0.073` in
  normalized A1 units;
- flow magnitudes are non-zero and remain below the configured 30-pixel cap.

This is not yet a production target. It is a later noisy observation, not a
clean reference, and the confidence mask intentionally discards uncertain
motion edges. The next validation must measure noise independence and confirm
that the dynamic loss has a small enough weight not to reintroduce noise into
static regions.

The follow-up validation reports high-pass source/target correlations with
per-pair means of approximately `0.088`, `0.110`, and `0.023`. R/B channels are
near zero and the green-channel correlations remain below `0.21`. This supports
the Noise2Noise independence assumption as a diagnostic, but does not prove it.
The saved validation report is
`artifacts/finetune_mis20s1/evaluation/dynamic_n2n_probe_645x/validation.json`.

The three saved pairs are therefore approved only as target-construction
probes. They are not yet an A5 training dataset: the training loader must first
construct eight-frame windows with per-frame warped targets and per-frame
confidence masks.

An eight-frame window probe was then generated with
`scripts/build_dynamic_window_probe.py` using temporal gap `4` and source
windows starting at `30`, `46`, and `140`. Its samples and previews are under
`artifacts/finetune_mis20s1/evaluation/dynamic_window_probe_645x/`.

All three windows stayed inside the `[0, 160)` training candidate range. Mean
confidence covered about `31%` to `41%` of detected dynamic pixels per window.
Across the eight-frame samples, mean high-pass source/target correlation was
`0.039` to `0.105` (green channels about `0.08` to `0.20`), which is consistent
with a usable Noise2Noise diagnostic but not proof of clean supervision.

The window probe is the first artifact that matches the model's eight-frame
input contract. It still must remain a probe until a small training smoke test
demonstrates that the dynamic loss improves motion fidelity without increasing
static-region noise.

## Motion-Mask Noise Audit

The green overlays were audited with
`scripts/diagnose_motion_mask_noise.py`. This audit does not change the saved
masks or training data; it measures whether pixels that pass the current
dynamic confidence rule also have meaningful optical flow. The report is saved
at `artifacts/finetune_mis20s1/evaluation/motion_mask_noise_diagnostic_645x.json`.

The current confidence rule has a substantial low-flow component:

- `30 -> 34`: `39.2%` of confidence pixels have flow below `1 px`;
- `46 -> 50`: `77.0%` have flow below `1 px`, and `51.6%` are below `0.5 px`;
- `140 -> 144`: `65.8%` have flow below `1 px`.

This does not prove every low-flow pixel is noise, because a slowly moving or
low-texture object can also have small flow. It does show that the current
green mask is not a reliable motion-only region. Sensor noise, fixed-pattern
residuals, static texture, and registration residuals can enter through the
raw-frame deviation mask and then survive the flow/photometric gates. The
existing window probes are therefore diagnostic artifacts only and must not be
promoted to A5 training data yet.

Before any dynamic-loss smoke training, rebuild or refine the confidence rule
with a minimum-flow or multi-frame temporal-consistency check, then regenerate
the three windows and repeat both the noise-independence and visual audits.

For the required visual gate, side-by-side current/strict mask variants were
rendered by `scripts/render_motion_mask_variants.py` under
`artifacts/finetune_mis20s1/evaluation/motion_mask_variants_645x/`. The strict
variants apply flow magnitude cutoffs of `1 px` and `2 px`; they are review
artifacts only and have not replaced the current masks.

The approved `1 px` candidate was then regenerated as a separate probe under
`artifacts/finetune_mis20s1/evaluation/dynamic_window_probe_645x_flow1/` and
`artifacts/finetune_mis20s1/evaluation/dynamic_n2n_probe_645x_flow1/`. It does
not overwrite the original probe. Mean eight-frame confidence fractions are
`1.81%`, `2.27%`, and `3.82%` for starts `140`, `46`, and `30`, respectively.
The window high-pass correlation means are `0.027`, `0.099`, and `0.096`; green
channel correlations remain below `0.19`. These results preserve the earlier
Noise2Noise diagnostic behavior while removing the low-flow confidence pixels.
The strict candidate is still a target-construction probe, not a training
dataset; a dynamic-loss smoke test remains gated on the full motion audit and
static/native regression checks.

## First Dynamic-Loss Smoke Result

A bounded 20-step smoke was run from the A3 step-5000 EMA weights with a fresh
Adam optimizer, learning rate `1e-5`, dynamic weight `0.1`, and EMA decay
`0.99`. Every update contained one original A3 static-supervision sample and
one strict-1px dynamic sample. Dynamic training frames were restricted to
`[0, 160)`; evaluation frames were restricted to `[160, 200)`.

The dynamic holdout Charbonnier objective changed from `0.0299539` to
`0.0298652` for the live model and `0.0299436` for EMA. End-to-end regression
evaluation rejected both candidate weight sets:

- EMA: 645x motion/input ratio changed from `0.48880` to `0.48831`, while
  1000x native PSNR changed from `52.364` to `52.230 dB`;
- live model: the motion/input ratio improved to `0.49424`, but native 1000x
  PSNR fell to `51.055 dB` (`-1.308 dB`).

The dynamic direction is therefore measurable before EMA smoothing, but the
current update is not domain-safe. This checkpoint must not replace A3. The
next controlled experiment is a matched 20-step static-only continuation from
the same A3 EMA initialization. That control is required before changing the
dynamic weight or freezing model blocks, because it separates ordinary
fine-tuning drift from drift introduced by dynamic supervision.

The matched static-only control was subsequently completed. Its live model
dropped native 1000x PSNR from `52.364` to `51.016 dB`, versus `51.055 dB` for
the dynamic candidate. Its 645x motion/input ratio fell to `0.48589`, versus
`0.49424` for the dynamic candidate. Relative to this matched control, dynamic
supervision contributes `+0.00834` motion ratio and `+0.039 dB` native PSNR.
Most of the unsafe native drift therefore comes from ordinary continuation of
the A3 static objective at learning rate `1e-5`, not from the strict-1px target
alone.

The next bounded candidate uses learning rate `1e-6` and dynamic weight `1.0`.
This reduces the static continuation step while retaining the previous nominal
`learning_rate * dynamic_weight` scale. It remains a smoke experiment and must
pass the same disjoint 645x motion and native 1000x gates.

That low-learning-rate candidate was also rejected. Its live model improved the
motion/input ratio only from `0.48880` to `0.48927`, while 645x static PSNR fell
by `0.0058 dB` and native 1000x PSNR fell by `0.082 dB`. Its EMA reduced the
native loss to `0.022 dB`, but the motion ratio fell to `0.48786`.

A final bounded parameter-scope probe trained only the `hi_qkv_mut`, `g_q_mut`,
and `g_kv_mut` mutual-attention parameters (`674,730` elements). It preserved
native quality more closely (`-0.024 dB`) but reduced the motion ratio to
`0.48794`; it was rejected as well.

These results close the simple hyperparameter/freeze sweep. The next design
should keep the A3 output as an explicit teacher anchor on static pixels and
apply strict-1px Noise2Noise loss only on dynamic pixels. Continuing the A1
static reconstruction objective changes the accepted A3 solution even without
the dynamic branch, while mutual-attention-only updates do not produce the
required motion gain.

## Teacher-Anchor Smoke Result

The teacher-anchor implementation was then tested for 20 steps from the same
A3 step-5000 EMA initialization. Static pixels were supervised against a
frozen copy of the A3 EMA output, while strict-1px dynamic Noise2Noise targets
were used only on the dynamic population. The configuration used learning rate
`1e-5`, dynamic weight `0.1`, and all model parameters trainable.

The live model is the first A5 candidate with a clear motion and native-domain
gain: the disjoint 645x motion/input ratio increased from `0.48880` to
`0.49462` (`+0.00582`), and native 1000x PSNR increased from `52.364` to
`52.394 dB` (`+0.030 dB`). Legal RAW fraction stayed at `1.0`.

It still regressed the disjoint 645x static PSNR from `29.516` to `29.489 dB`
(`-0.027 dB`) and lowered the 645x high-frequency ratio. Under the current
promotion gate, which requires no static/native regression, this is a
promising diagnostic checkpoint but not a production candidate. The EMA output
did not improve motion fidelity (`0.48845`) and is rejected for promotion.

Artifacts:

- `artifacts/finetune_mis20s1/training_a5_teacher_anchor_flow1/`
- `artifacts/finetune_mis20s1/evaluation/a5_teacher_anchor_flow1_model_weights/`
- `artifacts/finetune_mis20s1/evaluation/a5_teacher_anchor_flow1_ema/`

The next bounded experiment should retain the teacher-anchor direction and
reduce the 645x static tradeoff, either by strengthening the A3 consistency
term on static pixels or by reducing the dynamic update magnitude. A3 remains
the production baseline until that regression is removed.

## Stronger Teacher-Anchor Result

On 2026-09-29, a second 20-step teacher-anchor smoke doubled the separate
static anchor weight from `1.0` to `2.0`. The dynamic loss weight, learning
rate, strict-1px targets, training range, and holdout ranges were unchanged.

The live model retained a smaller motion gain: the 645x motion/input ratio
increased from `0.48880` to `0.49159` (`+0.00279`). Native 1000x PSNR improved
from `52.364` to `52.404 dB` (`+0.040 dB`), and legal RAW fraction remained
`1.0`. The 645x static PSNR still fell from `29.516` to `29.501 dB`
(`-0.015 dB`), while the high-frequency ratio fell from `0.97390` to
`0.95911`. EMA reduced the motion ratio to `0.48816` and was rejected.

Doubling the anchor reduced the static PSNR loss by about `44%` relative to
the weight-1 teacher smoke, but it also reduced the motion gain by about `52%`.
This establishes a scalar tradeoff rather than a passing operating point, so
further scalar anchor-weight sweeps are not justified.

The next design should apply the frozen A3 teacher directly to static pixels
of the same 645x dynamic training window. A spatial exclusion band around the
strict dynamic-confidence mask is required so uncertain motion boundaries are
not forced back toward the ghosted A3 output. This tests whether same-domain,
same-window anchoring can preserve 645x background detail while leaving the
confirmed moving region free to learn the Noise2Noise target.

Artifacts:

- `artifacts/finetune_mis20s1/training_a5_teacher_anchor_flow1_staticw2/`
- `artifacts/finetune_mis20s1/evaluation/a5_teacher_anchor_flow1_staticw2_model_weights/`
- `artifacts/finetune_mis20s1/evaluation/a5_teacher_anchor_flow1_staticw2_ema/`

Artifacts:

- `artifacts/finetune_mis20s1/training_a5_motion_smoke_flow1/`
- `artifacts/finetune_mis20s1/evaluation/a5_motion_smoke_flow1/`
- `artifacts/finetune_mis20s1/evaluation/a5_motion_smoke_flow1_model_weights/`

## Disjoint Holdout

For the first A5 design experiment, reserve a contiguous dynamic holdout from
the 645x stream. The current motion audit shows the strongest motion coverage in
the late sequence; use frames `[160, 200)` as the initial candidate holdout.
Before training, verify this range with the saved masks and record the exact
range in the experiment manifest.

- dynamic training candidate: frames `[0, 160)`;
- dynamic holdout: frames `[160, 200)`;
- native `1000x`: remains a separate gain holdout and stays out of training;
- no checkpoint or evaluation frame may be reused as a dynamic target.

The holdout must be evaluated with both static metrics and motion-temporal
metrics. A5 is promotable only if it improves motion fidelity without regressing
native 1000x PSNR/RMSE, legal RAW fraction, detail retention, or static temporal
stability.

## Required Evaluation Additions

The motion audit must be run for every candidate checkpoint and must report:

- motion fraction;
- prediction/input frame-difference ratio in motion regions;
- the lowest and highest motion transition ratios with frame indices;
- static-region temporal difference for context;
- legal RAW fraction and existing native-domain metrics.

The motion-region ratio is diagnostic, not a clean-reference quality metric.
Visual review remains required for the shortlisted holdout clips.

## Gate Before Implementation

Do not expand this probe into training data or start A5 until the target noise
independence, confidence-mask behavior, and dynamic-loss weight are verified on
the saved samples. The existing A3 checkpoint remains frozen as the
static-domain baseline during this design stage.
