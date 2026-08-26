# Task 7 Report

Status: DONE

Commit: `8d6fefb feat(v2): build atomic seed artifact roots`

Implemented `SeedArtifactsV2` and `build_v2_seed_artifacts()` in
`src/raw_fusion/v2/seed_artifacts.py`.  The producer rebinds all inputs,
requires a shared allowed root, stages MD h50/h5 then estimator replay under
one outer directory, recursively validates both, and atomically publishes
only the complete root.  `scripts/build_v2_seed_artifacts.py` is a thin
four-argument CLI with JSON output.

TDD evidence:

- Initial module tests: RED because `raw_fusion.v2.seed_artifacts` did not
  exist; GREEN `3 passed` after implementation.
- Initial CLI test: RED because the script did not exist; GREEN `4 passed`
  after implementation.
- Joint regression:
  `env PYTHONPATH=src OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
  /data1/wangzepu/.conda/envs/aaa_312/bin/python -m pytest
  src/tests/v2/test_seed_artifacts.py src/tests/v2/test_md.py
  src/tests/v2/test_md_cli.py src/tests/v2/test_replay.py -q`
  -> `42 passed in 1.82s`.

No real data run has occurred yet.  It will be a new, source-only output
directory and should occur only after this task review is clean.

## Fix Round 1

Addressed final-path seed artifact graph verification.  After atomic
publication, both manifests are rebound through the final output root and
recursively validated again.  If this relocation check fails, only the newly
created output root is removed before the original error is surfaced.

Added realistic nested MD-to-estimator reference coverage using
`rebase_artifact_ref`, direct MD-generation failure cleanup coverage, and
final-path validation cleanup coverage.

TDD evidence:

- RED: `test_seed_artifact_builder_discards_published_root_after_final_validation_failure`
  failed because the producer returned without final-path validation (`DID NOT RAISE`).
- GREEN focused:
  `env PYTHONPATH=src OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
  /data1/wangzepu/.conda/envs/aaa_312/bin/python -m pytest
  src/tests/v2/test_seed_artifacts.py -q` -> `7 passed`.
- GREEN joint regression:
  `env PYTHONPATH=src OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
  /data1/wangzepu/.conda/envs/aaa_312/bin/python -m pytest
  src/tests/v2/test_seed_artifacts.py src/tests/v2/test_md.py
  src/tests/v2/test_md_cli.py src/tests/v2/test_replay.py -q`
  -> `45 passed in 1.91s`.
