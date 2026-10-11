# A6 Frozen Motion Expert Result

- Date: 2026-10-09 (Asia/Shanghai)
- Scope: Stage2 RAW denoising only
- Baseline retained: `artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`
- Decision: no A6 checkpoint is promotable

## Motivation

A3 has strong static denoising quality but suppresses real motion. A same-protocol
audit of the official pretrained teacher confirmed that A3 fine-tuning made this
worse rather than exposing a hard architectural limit.

On the 645x sequence, the official teacher motion/input ratios were
`0.334/0.371/0.406/0.399` for R/Gr/Gb/B. A3 reduced them to
`0.270/0.323/0.308/0.286`. The lowest transition ratios fell from roughly
`0.176-0.224` to `0.097-0.127`.

Audit artifact:

`artifacts/finetune_mis20s1/evaluation/motion_audit_old_teacher_645x.json`

## Implemented Route

A6 freezes the complete A3 EMA backbone and trains only a small gated residual
expert:

`candidate = A3 + gate * residual`

The implementation includes:

- exact same-frame 10x dynamic targets with independent A2 calibrated noise;
- disjoint train `[0,160)` and holdout `[160,200)` frame ranges;
- spatial-only expert convolutions so the new branch cannot average across time;
- gate supervision from motion masks;
- same-window and independent static A3 anchors;
- optional same-frame official-teacher motion distillation on real 645x input;
- optional exact procedural local motion from the clean A1 reference;
- exact dynamic, teacher-distillation, and procedural holdout reports.

Code:

- `src/raw_fusion/finetune/a6_motion_expert.py`
- `scripts/run_a6_motion_expert.py`
- `scripts/evaluate_a6_motion_expert.py`
- `src/tests/test_finetune_a6_motion_expert.py`

## Experiments

### v1: noisy-input temporal features, 100 steps

Artifact:

`artifacts/finetune_mis20s1/training_a6_motion_expert_4090/`

- live exact holdout RMSE: `79.89 -> 82.94 DN`
- EMA exact holdout RMSE: `79.89 -> 79.87 DN`
- EMA static shift from A3: `0.11 DN`
- EMA temporal ratio: no improvement

Result: rejected. The gate and residual used high-noise temporal differences and
did not generalize.

### v2: A3 temporal features and spatial-only expert, 200 steps

Artifact:

`artifacts/finetune_mis20s1/training_a6_motion_expert_v2_4090/`

- EMA exact holdout RMSE: `79.89 -> 79.68 DN`
- EMA static shift from A3: `1.10 DN`
- gate IoU: `0.533`
- temporal ratio: `0.292258 -> 0.292305`

Result: rejected. The gate learned to locate motion, but the residual did not
restore measurable temporal content.

### v3: mixed 10x exact targets and old-teacher 645x distillation, 200 steps

Artifact:

`artifacts/finetune_mis20s1/training_a6_motion_expert_v3_teacher_4090/`

- old-teacher holdout RMSE: `166.54 -> 153.55 DN` with EMA
- exact 10x holdout RMSE: `79.89 -> 84.60 DN` with EMA
- exact 10x static shift from A3: `4.78 DN`

Result: rejected. The expert can recover part of the pretrained teacher's motion
behavior, but that behavior conflicts with the low-noise exact target domain.

### v4: mixed real 10x and procedural exact local motion, 300 steps

Artifact:

`artifacts/finetune_mis20s1/training_a6_motion_expert_v4_procedural_4090/`

- procedural EMA holdout RMSE: `35.57 -> 35.60 DN`
- real 10x EMA holdout RMSE: `79.89 -> 79.93 DN`
- real 10x temporal ratio decreased slightly

Result: rejected. Single-reference procedural motion does not transfer to the
real dynamic holdout.

## Final Interpretation

The frozen-backbone design solved the global-parameter drift problem and showed
that a gate can identify motion. The remaining blocker is the supervision:

1. The available exact 10x dynamic source contains only one scene and 200 frames.
2. Old-teacher outputs carry useful motion behavior but are not clean targets in
   the desired A1 domain.
3. Synthetic local translations are too narrow to represent real person motion.
4. The residual branch cannot infer missing clean dynamic detail from the current
   high-noise observation without broader paired motion data.

## Required Next Data

Do not continue scalar A6 sweeps with the current data. The next justified step
is to collect or obtain multi-scene paired dynamic data with exact same-time
targets:

- several people, speeds, directions, occlusions, and textures;
- low-gain clean and high-gain noisy captures with synchronized motion;
- fixed train/validation scene separation;
- enough clips to prevent one-trajectory memorization;
- target-domain dynamic pretrained weights as an alternative if available.

Until that data exists, A3 step-5000 remains the Stage2 baseline and all A6
checkpoints are research artifacts only.
