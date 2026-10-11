# GitHub Recovery Export

This repository is a clean export of:

`/Data4/Xiao/RECOVERED_Stage2_DenoiseModel_20261010`

Export date: 2026-10-11 (Asia/Shanghai)

## Export Policy

- Files larger than 100 MiB were not copied into this Git repository.
- Python caches, pytest caches, egg-info directories, and the damaged recovered
  `.git` directory were not copied.
- Existing files under `artifacts/` that fit the size limit are included in
  this snapshot even though `.gitignore` excludes future artifact additions.
- The recovered source directory remains the authoritative local copy.

The excluded-file manifest is:

`GITHUB_UPLOAD_EXCLUDED_LARGE_FILES_20261011.tsv`

It lists 53 skipped files totaling approximately 20.84 GiB. These include the
large checkpoints, full-frame predictions, motion masks, and other arrays that
exceed GitHub's regular Git file limit.

## Recovery Records

- `RECOVERY_REBUILD_20261010.md` describes the completed reconstruction.
- `.recovery/SHA256SUMS_POST_REBUILD` records checksums for the key rebuilt
  artifacts, including files that are too large for this export.
- No model training was started as part of the GitHub export.
