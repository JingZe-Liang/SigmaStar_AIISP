# RViDeformer Fake-GT Fusion v1

This package contains the v1 causal RAW fusion model trained with RViDeformer pseudo ground truth. It combines previous/current noisy RAW, a 2DNR candidate, and a 3DNR candidate into a packed-RGGB RAW output.

## Method

The model receives four aligned packed RAW inputs:

- previous noisy RAW;
- current noisy RAW;
- current 2DNR output;
- current 3DNR output.

It predicts a shared gate and a four-channel residual:

```text
base = gate * denoised + (1 - gate) * fused
prediction = base + residual_scale * tanh(residual_logits)
```

RViDeformer output is used only as training supervision. It is teacher pseudo-GT, not real ground truth.

## Layout

```text
configs/       Experiment configurations and a local-data template
src/raw_fusion/ v1 data, model, loss, train, infer, evaluation, and preview code
src/tests/     v1 synthetic tests
results/       Lightweight metrics, curves, and final 1080p videos
```

## Setup

```bash
python -m pip install -e ".[dev]"
cp configs/dataset_mis20s1.json.example configs/dataset_mis20s1.json
```

Edit every path in `configs/dataset_mis20s1.json` to point to local noisy, denoised, fused, and pseudo-GT RAW assets. The local file is ignored by Git.

The bundled configurations retain the v1 Fold A/Fold B protocol. `fold_a_full.json` is the baseline; `fold_a_full_optimized.json` contains the tuned gate-loss settings used for the published videos.

## Commands

```bash
raw-fusion-validate --config configs/dataset_mis20s1.json
raw-fusion-train --config configs/fold_a_full.json --output-dir outputs/fold_a_full
raw-fusion-evaluate --model candidate=configs/fold_a_candidate_gate.json,outputs/candidate/best.pt --model full=configs/fold_a_full.json,outputs/full/best.pt --sequence 645x --frames 8:191 --output outputs/eval.json
raw-fusion-infer --config configs/fold_a_full_optimized.json --checkpoint /path/to/best.pt --sequence 645x --start 8 --end 191 --output-dir outputs/infer --device cuda
```

## Results

See [results/README.md](results/README.md). The two final 1080p videos are included; source RAW, checkpoints, logs, MOG2 artifacts, probe loops, and the duplicate 4K comparison are intentionally not published.

## Verification

The included v1 test suite passed with Python 3.10, PyTorch 2.5.1+cu124, and NumPy 2.2.6:

```text
117 passed
```

## Scope

Only two scenes, 128x and 645x, are represented. This v1 package is a feasibility study and does not establish production generalization.
