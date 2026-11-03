# Stage2 DenoiseModel Recovery Handoff

Handoff ID: `STAGE2-RECOVERY-READY-20261011`

- Updated: 2026-10-11 (Asia/Shanghai)
- Source handoff: `STAGE2-A5-FINAL-20261007`
- Authoritative project root: `/Data4/Xiao/RECOVERED_Stage2_DenoiseModel_20261010`
- Dataset root: `/HardDisk/jingzeliang/Data/Company_Dataset/dataset`
- Scope: Stage2 RAW denoising only
- State: materials verified; recoverable omissions replenished; waiting for the user's next objective
- Active Stage2 training/evaluation jobs at handoff time: none found

## Read This First

The disk-recovery and reconstruction work is complete enough to continue the
Stage2 investigation. The authoritative working copy is the local recovery
tree under `/Data4/Xiao`; the GitHub copy is only a size-limited backup and is
not a complete runtime artifact set.

The current model decision has not changed:

> Keep A3 step-5000 as the only accepted Stage2 denoiser baseline. A5 and A6
> are rejected. A7 is a deterministic post-processing investigation and is
> not a promoted model.

The user has confirmed a hard constraint:

> No new data can be captured. Any next route must use the datasets and
> artifacts that already exist.

No new training or evaluation experiment is authorized by this handoff. The
new session must first verify the recovery state, report any differences, and
then follow the user's newly specified objective.

## Authoritative Locations

### Complete local recovery

Project root:

`/Data4/Xiao/RECOVERED_Stage2_DenoiseModel_20261010`

This is the authoritative copy for future execution because it retains the
large checkpoints, predictions, motion masks, and arrays that cannot be stored
in ordinary GitHub Git objects.

Primary recovery report:

`RECOVERY_REBUILD_20261010.md`

Post-rebuild checksums:

`.recovery/SHA256SUMS_POST_REBUILD`

Recovered-source provenance:

- `.recovery/SOURCE_MAP.tsv`
- `.recovery/SHA256SUMS`
- `.recovery/evidence/`

### GitHub backup

Repository: `JingZe-Liang/SigmaStar_AIISP`

Branch and commit:

- branch: `main`
- commit: `9964f776dd7ebd015a61f4fd416989b3cf4aba85`

Snapshot path in that repository:

`Stage2/9_19-10_1/Task3/XWJ/RECOVERED_Stage2_DenoiseModel_20261010`

The uploaded snapshot contains 307 files totaling approximately 933.5 MiB.
Files larger than 100 MiB were intentionally skipped. The exact 53-file,
20.842-GiB exclusion list is:

`GITHUB_UPLOAD_EXCLUDED_LARGE_FILES_20261011.tsv`

Do not use the GitHub snapshot alone for model execution. It is suitable for
code, reports, configurations, small evaluations, and disaster-recovery
reference, but not as a complete artifact store.

## Preserved Model Decision

The accepted baseline checkpoint is:

`artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`

Do not overwrite or replace it.

The original A5 final decision remains authoritative:

`docs/HANDOFF_2026-10-07_STAGE2_A5_FINAL.md`

Key conclusions:

- A3 produces strong static denoising but suppresses genuine motion.
- A5 same-window motion supervision produced only a small diagnostic motion
  gain while regressing one or more static, detail, temporal, or native-domain
  gates.
- No A5 checkpoint is promotable.
- A6 frozen motion-expert variants v1 through v4 did not recover useful motion
  without unacceptable target-domain or static regressions.
- No A6 checkpoint is promotable.
- The limitation is primarily supervision quality and diversity, not an
  inability to detect motion regions.

Relevant result documents:

- `docs/A6_FROZEN_MOTION_EXPERT_RESULT_2026-10-09.md`
- `docs/A7_REAL_MOTION_CAPTURE_PILOT_2026-10-09.md`
- `docs/RVIDEFORMER_FINETUNE_PLAN_2026-09-20.md`

The A7 capture document remains useful as a statement of the missing ideal
supervision, but its capture route is currently unavailable because the user
cannot collect new data.

## A7 Deterministic Fusion Status

A7 deterministic restricted motion-region fusion was recovered and rebuilt.
It mixes the A3 output with the old official teacher only inside strict motion
regions while applying a static guard.

Recovered evaluation directories include:

- `artifacts/finetune_mis20s1/evaluation/a7_motion_fusion/`
- `artifacts/finetune_mis20s1/evaluation/a7_motion_fusion_r7/`
- `artifacts/finetune_mis20s1/evaluation/a7_motion_fusion_r11/`

The r7 and r11 fusion weights regenerated after recovery are byte-identical to
the surviving pre-damage arrays. Static nonzero fusion weights are zero for
both variants.

The strongest reported diagnostic setting increases the 645x motion/input
ratio only modestly, from approximately `0.48879` to `0.49327`. This is not a
clean-reference motion-quality result and is not sufficient for promotion.
A7 therefore remains a post-processing research result, not a replacement for
the A3 model.

## Recovery Work Completed

The following capabilities are operational in the `/Data4/Xiao` tree:

- A3 inference and inspection;
- native 1000x evaluation;
- A5 and A6 result inspection;
- A7 deterministic motion-region fusion;
- all 15 recovered motion-mask arrays;
- A5 dynamic train and disjoint holdout windows;
- the native 1000x motion-aware reference;
- canonical full 200-frame A3 outputs for 645x and native 1000x;
- rebuilt A7 r7/r11 JSON reports and fixed-ISP panels.

Important verification results:

- complete test suite: `61 passed, 1 warning`;
- warning: pre-existing PyTorch `torch.meshgrid` indexing warning;
- rebuilt JSON, NPY, and PNG files passed structural checks;
- A7 r7/r11 regenerated weight arrays matched the surviving arrays byte for
  byte;
- no model training was started during recovery;
- no active Stage2 process was found when this handoff was written.

Canonical checks and metrics are recorded in:

`RECOVERY_REBUILD_20261010.md`

Do not replace that report with summaries from chat history.

### Replenishment after handoff verification

On 2026-10-11, the locally missing GitHub exclusion TSV was restored exactly
from backup commit `9964f776`, and the missing final A5 EMA evaluation was
regenerated from the surviving step-20 checkpoint. The regenerated metrics
match the historical handoff values within `1.1e-5` absolute error.

The A5 EMA prediction arrays are reconstructed outputs, not byte-identical
historical artifacts. Provenance is recorded in:

`artifacts/finetune_mis20s1/evaluation/a5_same_window_teacher_flow1_r1_ema/README_RECONSTRUCTED_20261011.md`

Replenishment checksums are in:

`.recovery/SHA256SUMS_REPLENISH_20261011`

## Remaining Gaps

The following losses are real and must not be represented as recovered:

- some early A5 checkpoints and logs were absent from the journal sources;
- the original final A5 EMA prediction arrays remain unavailable byte for
  byte; explicitly marked reconstructed replacements now exist;
- historical GPU-generated arrays cannot always be recreated byte for byte,
  although the rebuilt 645x baseline passed explicit metric-tolerance checks;
- one old Git tree object is still missing:
  `0cf8327baca79abf6e0bcd2d8f44c8c856536a04`;
- anonymous damaged objects remain quarantined under `.recovery/evidence/` and
  are not valid experiment outputs;
- the GitHub snapshot omits 53 large files by design.

These gaps do not block A3 inference, the rebuilt evaluations, or a new
existing-data investigation. They do block exact resumption of some historical
A5 training runs.

## Git And Workspace Rules

The recovered local `.git` database is not fully trustworthy because of the
missing historical tree object. Do not run destructive Git cleanup commands in
the recovery tree.

In particular, do not use:

- `git reset --hard`;
- `git clean -fdx`;
- mass deletion of untracked or recovered artifacts;
- an operation that replaces the A3 step-5000 checkpoint.

The clean GitHub backup commit is the recoverable version-control snapshot for
the 307 included files. The `/Data4/Xiao` tree remains the operational artifact
source.

## Data And Authorization Constraints

- Treat `/HardDisk/jingzeliang/Data/Company_Dataset/dataset` as read-only.
- Write new code, reports, and experiment outputs only inside the recovered
  Stage2 project unless the user explicitly chooses another destination.
- No recapture is available.
- Do not claim that existing unpaired motion footage supplies exact clean
  dynamic targets.
- Do not start CUDA work before checking GPU occupancy and receiving a concrete
  next objective.
- Do not resume Stage1 FusionNet or downstream student work under this Stage2
  handoff unless the user explicitly changes scope.

## Closed Routes

Do not repeat these routes unchanged unless the user explicitly reopens them
with a materially new hypothesis:

- additional scalar A5 same-window sweeps;
- promotion of any A5 smoke checkpoint;
- additional scalar A6 v1-v4 sweeps using the same supervision;
- treating old-teacher output as clean dynamic ground truth;
- treating single-reference procedural translation as sufficient real-motion
  supervision;
- using the motion/input ratio alone as proof of motion fidelity;
- presenting A7 restricted fusion as a trained denoiser;
- training on the original A5 train frames and reporting the same frames as an
  independent validation result.

## Next Objective Boundary

The next objective is intentionally not inferred here. Under the existing-data
constraint, a defensible next task should begin with one of these bounded
questions, selected by the user in the new session:

1. Evaluate whether any unused existing stream can support a genuinely
   independent multi-scene motion benchmark without pretending it has clean
   targets.
2. Decide whether to continue A7 as a conservative deployment-time
   post-process, with fixed-ISP visual review and strict static invariance.
3. Freeze the motion-fidelity research at A3 and package the recovered A3
   baseline for downstream use.
4. Investigate a new self-supervised or test-time route, but only after stating
   how it avoids the supervision failures already demonstrated by A5 and A6.

These are decision options, not authorization to execute an experiment.

## First Action In The New Session

The new session should perform a read-only recovery check before modifying
files or starting a GPU job:

1. Confirm the handoff ID and `/Data4/Xiao` project root.
2. Confirm the A3 step-5000 checkpoint exists.
3. Read `RECOVERY_REBUILD_20261010.md` and the original A5 final handoff.
4. Confirm `.recovery/SHA256SUMS_POST_REBUILD` exists.
5. Confirm no Stage2 training or evaluation process is active.
6. Confirm GitHub `main` contains commit `9964f77` at the documented path.
7. Report any differences, then ask for or follow the user's concrete next
   objective.

The read-only check must not start an experiment or alter the recovered tree.

Suggested opening message:

```text
$project-handoff
请读取并核对：
/Data4/Xiao/RECOVERED_Stage2_DenoiseModel_20261010/docs/HANDOFF_2026-10-11_STAGE2_RECOVERY_READY.md

交接编号：STAGE2-RECOVERY-READY-20261011。
项目目录：/Data4/Xiao/RECOVERED_Stage2_DenoiseModel_20261010。
硬约束：不能重新拍摄，只能使用现有数据集。

先只读核对恢复报告、A3 step-5000 基线、A5/A6 否决结论、A7 后处理状态、GitHub 部分快照与大文件缺失清单，并检查当前是否有 Stage2 进程。本轮不要修改代码、不要启动实验。核对后告诉我当前结论和发现的差异，然后等待我指定下一步目标。
```
