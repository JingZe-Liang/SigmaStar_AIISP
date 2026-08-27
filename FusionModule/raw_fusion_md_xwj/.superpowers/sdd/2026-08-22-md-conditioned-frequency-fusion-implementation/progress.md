# SDD ledger — plan: docs/superpowers/plans/2026-08-22-md-conditioned-frequency-fusion-implementation.md

## Workspace

- Worktree: `/data1/wangzepu/Jaime/Projects/RViDeformer_distill/raw_fusion_distill/.worktrees/v2-fusion-implementation`
- Branch: `v2-fusion-implementation`
- Plan commit: `c10ff1f` (with worktree-ignore setup commit `17653af`)
- Baseline: `117 passed` (`env OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 /data1/wangzepu/.conda/envs/aaa_312/bin/python -m pytest src/tests -q`)
- The original repository's dirty V1 files are intentionally outside this worktree and remain untouched.

## Preflight Interface Scan

| Boundary | Shared surface | Ruling |
|---|---|---|
| P0 -> all | Exact schemas, hashes, `OwnedArtifactRef`, atomic publication | P0 owns recursive validation and path/SHA binding; later tasks fill declared values only and add no optional/unknown keys. |
| P1 -> P3/P7/P8/P11/P12/P13/P14 | RAW/B2/morphology/stats/bootstrap kernels | P1 is the sole numerical definition owner; consumers call the kernels and do not reimplement rounding, CFA order, or bootstrap. |
| P2 -> P4/P5/P6/P7/P9 | Dataset and split ancestry | P2 owns train/validation frame sets and source hashes; no consumer may infer a split from filenames. |
| P3 -> P8/P9/P10/P11/P12/P13/P14 | Protection, selector, composer, quantization | Runtime and oracle use the same selector/composer; hard protection and `denoised_bypass` are fail-closed. |
| P4/P5 -> P7/P8/P9/P12/P13/P14 | MOG2 seed and causal estimator/replay | MOG2 h50/h5 manifests are versioned inputs; reset/replay state is explicit and source-frame causal. |
| P6 -> P8b/P11/P14/P15 | Manual motion masks and evaluation windows | Manual masks are immutable, SHA-bound artifacts; active-frame metrics use the locked windows only. |
| P7 -> P8/P8b/P9/P11/P14 | Alignment and evidence fields | Evidence masks/statistics are serialized, split-local, and never silently recomputed from validation data. |
| P8 -> P8b/P9/P11/P12/P14/P15 | Label bundle and feasible reference | Label rows/shards have exact condition/split/frame mapping; missing-label evaluation uses the frozen `missing_policy_label_bypass`. |
| P8b -> P9/P10/P11/P12/P14 | Diagnostics, stability, pretraining gate | P8b publishes the gate before model work; P9 consumes it as orchestration prerequisite, not as an extra sampler key. |
| P9 -> P10/P12 | Deterministic sampler and training rows | Sampler is train-only and rank depends only on train pool/contract ancestry, never validation bytes. |
| P10 -> P11/P12/P13 | Core, tiling, dependency radius | Core output is pixel logits; selector reduction is full-frame/cell-aware; tile radius <=32 is checked before model mutation. |
| P11 -> P12/P14/P15 | Metric series and hard-stop decisions | P11 returns pure decisions with ordered per-frame series; P12 applies checkpoint-safety rules, P14 publishes final report. |
| P12 -> P13/P14/P15 | Experiment, checkpoints, train runs | Every checkpoint is bound to one experiment/contracts/noise state and loaded only after validation. |
| P13 -> P14/P15 | Runtime/inference outputs and traces | Evaluation records actual effective inputs, `core_called`, fallback stage/reason; startup/reset previous input is unused/null. |
| P14 -> P15 | Evaluation, audit, delivery | P14 owns the 17-run result and full 0..199 raw index; P15 only consumes validated artifacts and never cherry-picks metrics. |

## Frozen Decisions

- V1 contracts/loaders remain unchanged; V2 lives under `src/raw_fusion/v2/` and rejects V1/unknown keys before model or optimizer factories are invoked.
- `output_source` uses the runtime literal `denoised_bypass`. Pre-core bypass clears unused model inputs; post-core bypass preserves consumed inputs and records `core_called` plus `fallback_stage`.
- Evaluation raw bundles carry all frames 0..199 (with explicit scored/context partitions), allowing temporal and any-frame metrics to be recomputed from bytes. Feasible-label rows without labels use the one declared fallback.
- No-MD control still replays nominal h50 estimator/P_high/q; only `P_hard` is overridden and W_edge follows the declared no-MD rule.
- P0 is the only owner of recursive exact validators and artifact ownership context. P11 modifies P8b metric/hard-stop kernels rather than creating a duplicate diagnostics stack.

## Task Status

| Task | Status | Implementer | Review | Commit |
|---|---|---|---|---|
| P0 | completed | p0_implementer + root review | root review recorded | `b6c6544`, `c138e82`, `b6cc4a8`, `d864f3d`, `7d8af24`; base `17653af2ec7f2541996310982ebcd728227183db` |
| P1 | completed | root (TDD) | root review recorded | `bc8b986` |
| P2 | completed | root (TDD) | root review recorded | `e7b14ee` |
| P3 | completed | root (TDD) | independent review complete, no Blocking/High | `a4df9a0`, `40ef886` |
| P4 | completed | root (TDD) | independent review complete, no Blocking/High | `19ebca7` |
| P5 | completed | p5_estimator + root contract integration (TDD) | independent review complete, Ready | `275bf09` |
| P6 | completed | p6_manual_motion + root integration (TDD) | independent review complete, no Critical/Important | `8d0773a` |
| P7 | completed | root alignment + p7_evidence (TDD) | independent review complete; support-warp contract fixes verified | `32a7f5d`, `47a5174`, `5f87448`, `8c98e5a`, `58519e7` |
| P8 | completed | p8_labels + root integration (TDD) | formal generator, audit-redacted runtime closure, and canonical texture-admission reader verified | `9a67460`, `3333905`, `3f3f62e` |
| P8b | completed | root (TDD) | independent review fixes incorporated; recursive diagnostic ancestry verified | `2fba134`, `9d34a09`, `1481652`, `56fcf4d` |
| P9 | completed | p9_sampler_resume (TDD) | deterministic dual-condition sampler/data commit recorded | `55a9bdb` |
| P10 | completed | root (TDD) | shared noise-conditioned core and tiling commit recorded | `ff60c12` |
| P11 | completed | root (TDD) | fixed schedules, immutable bootstrap evidence, and label-derived texture admission verified | `faaf482` |
| P12 | in progress | root (TDD) | fixed-loss review findings resolved; checkpoint/train pending | `8bc2fb7` |
| P13 | pending |  |  |  |
| P14 | pending |  |  |  |
| P15 | pending |  |  |  |

## Review Log

### P0 implementation

- Implementer report: `.superpowers/sdd/2026-08-22-md-conditioned-frequency-fusion-implementation/reports/p0-implementer.md`
- Focused V2 artifacts/schemas: 8 passed; selected V2+V1: 22 passed; full `src/tests`: 125 passed.
- Self-review found a real contract bug: generic plan-owned validators reapplied the parent key set to every nested mapping and silently accepted unregistered names. Fixed with an explicit registry key map, a structural-only nested JSON walker, and a closed MetricId enum; added regression tests for unknown keys and nested mappings. V2 suite: 11 passed; full suite: 128 passed.
- Independent reviewer slot was unavailable after context compaction; this review was performed against the generated diff and focused tests, with the finding recorded here for later broad review.

### P1 dispatch

- Task brief: `p1-brief.md` (generated from the plan's P1 section with normalized task heading).
- RED: focused kernel collection failed with 16 `ModuleNotFoundError` failures.
- GREEN: focused P1 tests 18 passed; V2 suite 29 passed; V1 RAW suite 10 passed; kernel smoke/gradient checks passed.
- Review notes: pack order is explicitly `[R,Gr,B,Gb]`; morphology uses square/Chebyshev with zero border and 4-connected hole filling; bootstrap has an independent test-local SHA/struct golden recomputation.

### P2 dispatch

- Task brief: `p2-brief.md`.
- RED: dataset/split tests failed on missing DatasetV2 value objects and missing fixed packed-order/range checks.
- GREEN: P2 tests 7 passed; real source config load resolved all nine source assets and verified their SHA; V2 suite 36 passed; full suite 153 passed.
- Review notes: source paths are workspace-root-relative (`Data/...`) and resolved only through the declared allowed root; pseudo-GT is absent from DatasetV2; split ranges are expanded from inclusive endpoints before ancestry checks.

### P3 dispatch

- RED: focused protection/selector/composer/quantization collection failed with 19 missing-module failures.
- GREEN: focused P3 tests 19 passed; C++17 half-even quantization golden passed; full suite 172 passed.
- Review notes: B2 is computed before any tile/crop, candidate validity is per CFA, selector forward values are exactly `{0,.125,.25,.5}`, and every quantization/LP/protection failure returns the original denoised object with literal `denoised_bypass`.

### P4 dispatch

- Initial RED/GREEN slice covers widened Gr/Gb averaging, fresh split-safe label background replay, exact 18-key common parameters, fixed histories/shapes, OpenCV getter identity, binary masks, and warmup readiness.
- GREEN: focused P4/schema tests 18 passed. A transactional 8x8, 200-frame fixture verifies both h50/h5 runtime streams and split-local label rows before atomic publication.
- Review fixes bind each input stream to the matching DatasetV2 denoised asset, recompute label input and ancestry digests, reject query-time background mutation, and recursively validate the protocol hash closure before opening source frames.
- Independent P3/P4 review: 48 focused tests passed, protocol hash closure passed, and no Blocking/High remained. A real full-resolution MOG2 bundle is intentionally deferred until the remaining replay integration is committed.

### P6 dispatch

- RED/GREEN covered the locked 52-frame-per-condition annotation windows, center-priority deduplication, physical PNG validation, packed 2x2 OR masks, pause/resume, frozen ISP and DatasetV2 2DNR ancestry, exact CLI behavior, and atomic final publication.
- Root integration replaced the stale shallow registry entry with a lazy delegation to the authoritative recursive manifest validator. The legal-manifest and nested-extra registry regression passes.
- GREEN: all 33 manual-motion tests passed in 179.68 seconds. Independent review found no remaining Critical/Important; the documented single-annotator flow retains a Minor concurrent-writer no-clobber race.

### P5 dispatch

- Two-pass calibration now fits support only from frame 58-93 seed candidates, replays the full coupled h50/h5 MD and estimator state per condition, and derives model normalization only from pass-two ACTIVE EMA values.
- Review fixes require HOLD recovery through ACQUIRE streak 1, Euclidean translation limits, recursive Dataset/Split/MD cross-links, and content-level recomputation of seed, frozen stats, timelines, reset schedules, and all four 200-frame coupled replays.
- The versioned noise-estimator contract now records the complete five-state machine and numerical policy; its SHA is bound through the protocol contract and EstimatorBundle generator. Focused P5: 51 passed; P4/P5 cross-layer: 69 passed; V1 full suite: 117 passed. Independent review found no Blocking/High.
- Real full-resolution Milestone A generation remains a later data-run checkpoint for runtime, artifact-size, and OpenCV replay evidence.

### P7 dispatch

- Alignment RED began with an absent module. GREEN now covers source-to-target phase direction, the fixed float32 Hann, 0.5 packed-pixel validation, ten-iteration through-origin exposure IRLS, train-only residual P99, nearest binary-mask warp, and out-of-bounds invalid evidence.
- Focused alignment tests: 7 passed. Evidence/state-label implementation is running independently against the frozen P1/P2/P4/P5 interfaces.

### P7 completion

- Production alignment now accepts strict normalized packed `[R,Gr,B,Gb]` 2DNR/noisy tensors, derives the low-pass green plane internally with the shared A2 kernel, and applies the estimated translation/exposure only to the corresponding pre-NR noisy frame.
- Support evidence is filled in source coordinates before support-to-canonical-to-target warping; bilinear and nearest-neighbor out-of-bounds validity remain explicit and fail closed.
- Bootstrap resamples each raster position from its actual valid block set. It emits replicate-level cell stability as `p_state_cell`, using the exact cell core and 90% texture rule; P7 exposes only the G/support `texture_candidate`, while P8 remains the sole texture-admission authority.
- Review fixes keep motion independent of G bootstrap validity and remove E/V from the P7 classifier. Focused P7: 54 passed; full V2 snapshot: 258 passed, 1 CUDA-unavailable skip. Independent re-review found no Blocking/High and marked the slice Ready.

### P7 support-warp closure

- The noisy CFA evidence is now warped per source frame before the shared B2 stage, preserving canonical `[R, Gr, B, Gb]` order.
- Bilinear correlation validity requires the complete footprint; out-of-bounds evidence is unknown rather than a Dilate12 seed. Focused alignment/evidence/bootstrap verification: 61 passed.

### P8/P8b progress

- P8 bounded label kernels freeze selector/composer/quantization reuse, exact cell state recomputation, local feasibility, E/V bootstrap actual-n, and audit redaction before the P7 kernel. The formal source-recursive bundle writer is still pending.
- P8 adds a fail-closed label input resolver that validates RAW packing/offsets, split roots, h50-only MD/estimator use, B2 dependency radius, and the audit firewall before generation.
- P8b adds typed metric series, paired moving-block bootstrap, label stability, projected-reference checks, seven pretraining rules, and atomic invalid diagnostics. The source-recursive ancestry and bootstrap cardinality checks were tightened in follow-up fixes; focused diagnostics/hard-stop tests: 25 passed.

### P8/P5b/P11 completion

- P5b now emits and recursively replays an audit-redacted h50 estimator view.  Its closed domain, redacted inputs, seed candidates, frozen stats, normalization, runtime masks, state timelines, reset schedule, and view digest are all rebound before labels can use it.
- P8 now has the formal four-shard real-data adapter.  It binds only the audit-redacted runtime view for label/protection replay and seed material, freezes G/support texture admission before E/V policy evidence, and exposes immutable canonical admission rows with per-cell H_E/H_V-core validity.
- P11 adds the fixed 0..199/train/validation/manual schedule, all twelve promotion rules, checkpoint-safety projection, exact paired bootstrap reductions, and immutable bootstrap material.  Pretraining projected-reference admission is now recomputed from a fully validated LabelBundleV2; marked synthetic test sources get a zero denominator and cannot pass the gate.
- Regression found one stale schema smoke fixture whose empty MetricFrameV2 values contradicted the pre-existing non-empty schema contract.  The fixture now carries one valid invalid-metric row; the contract was not weakened.
- Verification: focused replay/estimator `56 passed`; label pipeline/bundle/protection `69 passed`; hard-stop/metric `69 passed`; full V2 suite `440 passed, 2 skipped` (CUDA unavailable).
- Independent agent review attempts during this completion window were rate-limited; root performed the final diff/contract review and full-suite verification.

### P12 fixed-loss slice

- Added the eight fixed region losses with independent per-condition regions, equal 128x/645x macro reduction, graph-safe empty masks, finite/positive epoch-denominator checks, and no pseudo-GT/H_V or `labels_tm1` input surface.
- P9 batches now carry the canonical 192x192 cell-core raster derived from geometry; an integration test constructs `FusionLossInputsV2` directly from a collated dual-condition batch.
- Review-driven corrections constrain non-admitted texture cells with invalid policy alpha to abstain, preserve the specified motion-or-MD-boundary definition, and reject non-finite denominator records.
- Verification: targeted data/loss tests `34 passed, 1 skipped`; associated model/data/loss/selector/composer tests `63 passed, 2 skipped`; complete V2 suite `466 passed, 3 skipped` (all skips CUDA unavailable).
