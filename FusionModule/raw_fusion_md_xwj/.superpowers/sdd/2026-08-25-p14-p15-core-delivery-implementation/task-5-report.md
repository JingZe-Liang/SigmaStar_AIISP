# Task 5 Report: Preview, Delivery, and V2 CLI Entries

## Changed Files

- `src/raw_fusion/v2/preview.py`
  - Added fixed 30 fps preview metadata and a deterministic synthetic encoder
    boundary.  The default production-shaped frame count is 200; injected
    synthetic manifests may use a smaller positive count without changing the
    fps contract.
- `src/raw_fusion/v2/delivery.py`
  - Added recursive artifact reopening for P14 evaluation, audit, smoke, and
    both inference manifests.
  - Enforced one plan, conditioned experiment/checkpoint, shared root identity,
    distinct 128x/645x stream IDs, and a passed smoke gate before publication.
  - Publishes a complete two-condition `DeliveryManifestV2` atomically with
    deterministic marker artifacts in place of real video encoding.
- `src/raw_fusion/v2/cli.py`
  - Added twelve exact V2 wrapper functions with `allow_abbrev=False` and
    stable exit codes: parser/preflight `2`, semantic smoke failure `1`, and
    successful publication/validation `0`.
  - The V2 train/infer wrappers fail closed at their deferred real-data/P13
    boundary and do not start CUDA or inference.
- `scripts/render_v2_delivery.py`
  - Added the exact six-flag delivery script.
- `pyproject.toml`
  - Added the twelve `raw-fusion-v2-*` entry points without changing V1 names.
- `src/tests/v2/test_delivery.py`, `test_cli.py`, `test_pipeline.py`
  - Added small owned-artifact fixtures for matching delivery, checkpoint
    mismatch/no-output behavior, exact argument rejection, and entry-point
    presence.

## Verification

Focused synthetic checks:

```text
6 passed in 2.05s
```

Compilation of the four changed Python modules and `git diff --check` both
exited successfully.  No real RAW, MD, CUDA, P13 inference, ffmpeg, or video
render was started.

## Commit

Commit: `0e05975 feat(v2): add delivery and command workflow` (amended below
with post-publication provenance re-open checks).
