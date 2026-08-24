# v1 Results

All reported values use RViDeformer pseudo ground truth, not real ground truth.

## Videos

- `videos/v1_pseudogt_fusionnet_645x_frames_0008_0191_1080p.mp4` is the optimized v1 FusionNet output for 645x frames 8 through 191.
- `videos/v1_pseudogt_fusionnet_645x_comparison_0008_0191_1080p.mp4` compares 2DNR, 3DNR, FusionNet, and the RViDeformer pseudo-GT reference over the same frame range.

The MOG2 overlays, single-frame probe loops, RAW streams, checkpoints, and the duplicate 4K comparison are intentionally excluded.

## Metrics and Curves

- `metrics/eval_fold_a_645x.json` and `metrics/eval_residual_vs_full_645x.json` contain per-frame and aggregate pseudo-GT metrics.
- `metrics/model_diagnosis.json` contains model diagnostics.
- `training/` contains retained JSONL curves. `fold_b_candidate_gate_incomplete.jsonl` stopped after epoch 6; no Fold B candidate-residual run is available.
