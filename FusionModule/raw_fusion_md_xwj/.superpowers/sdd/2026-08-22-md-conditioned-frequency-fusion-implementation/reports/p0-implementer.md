# P0 Implementer Report

Baseline: `17653af2ec7f2541996310982ebcd728227183db`

Implemented the V2 dependency root under `src/raw_fusion/v2/`: immutable owner-aware `ArtifactRef`/`OwnedArtifactRef`/`ArrayRef`, lowercase SHA-256 verification, realpath containment and symlink escape rejection, nested child ownership, rebasing, JSON object loading, and validated atomic directory publication. Added recursive exact-schema primitives, V1 `CheckpointV2` rejection before model/optimizer factories, schema validator registry and P0 validator entry points, typed config consumers, protocol contract hash closure, canonical contract artifacts, and the locked V2 requirements file. V1 modules were left unchanged.

Tests:

- `PYTHONPATH=src env OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 /data1/wangzepu/.conda/envs/aaa_312/bin/python -m pytest src/tests/v2/test_artifacts.py src/tests/v2/test_schemas.py -q` -> 8 passed.
- `PYTHONPATH=src env OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 /data1/wangzepu/.conda/envs/aaa_312/bin/python -m pytest src/tests -q` -> 125 passed.
- `git diff --check` -> clean.

Deviation: plan-owned diagnostic schemas are registered and recursively validated with the literal key sets available in the P0 brief/plan; producer-specific semantic checks remain owned by their later tasks.
