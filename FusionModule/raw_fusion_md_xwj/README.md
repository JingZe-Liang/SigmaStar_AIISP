# RAW Fusion Distill V2

Data-first RAW fusion training for aligned `noisy`, `denoised`, and `fused`
streams. One conditioned model serves the 128x and 645x conditions.

MOG2 is used only to produce training supervision. It is not a network input,
is not required during inference, and no `label_bundle` or `sampler_manifest`
is needed.

## Repository layout

```text
configs/v2/       Dataset and split configuration
src/raw_fusion/   Training, inference, comparison, and contracts
src/tests/v2/     Unit and contract tests
docs/             Protocol and baseline-result documentation
results/          Curated safe-q reproducibility baseline
```

Large generated data is intentionally outside the repository. The default
local experiment directory is `../DERIVED/v2_data_first/` relative to this
repository.

## Install and validate

```bash
conda run -n aaa_312 python -m pip install --no-deps -e .
PYTHONPATH=src conda run -n aaa_312 python -m pytest src/tests/v2 -q
```

```bash
raw-fusion-v2-dataset-validate \
  --config configs/v2/dataset_mis20s1.json \
  --allowed-root /data1/wangzepu/Jaime
raw-fusion-v2-split-validate --config configs/v2/split.json
```

## Offline MOG2 supervision

Generate the restartable MOG2 cache once. Its masks are supervision data only;
do not pass them to the model or package them for deployment.

```bash
raw-fusion-v2-mog2-cache-generate \
  --dataset configs/v2/dataset_mis20s1.json \
  --split configs/v2/split.json \
  --output-dir ../DERIVED/v2_data_first/mog2_cache_train \
  --workers 32
```

## Train and resume

```bash
raw-fusion-v2-train --data-first \
  --dataset configs/v2/dataset_mis20s1.json \
  --split configs/v2/split.json \
  --mog2-cache ../DERIVED/v2_data_first/mog2_cache_train \
  --output-dir ../DERIVED/v2_data_first/train_run \
  --device cuda --batch-size 2 --max-steps 1000 \
  --log-interval 10 --checkpoint-interval 25
```

Monitor loss with `tail -f ../DERIVED/v2_data_first/train_run/metrics.jsonl`.
To resume, keep the dataset, split, seed, and batch size unchanged:

```bash
raw-fusion-v2-train --data-first \
  --dataset configs/v2/dataset_mis20s1.json \
  --split configs/v2/split.json \
  --mog2-cache ../DERIVED/v2_data_first/mog2_cache_train \
  --output-dir ../DERIVED/v2_data_first/train_run \
  --device cuda --batch-size 2 --max-steps 1000 \
  --resume ../DERIVED/v2_data_first/train_run/checkpoint_step_000500.pt
```

## Inference and comparison

Inference needs aligned noisy(t-1), noisy(t), denoised, and fused RAW frames.
It does not read MD or MOG2 data.

```bash
raw-fusion-v2-infer --data-first \
  --checkpoint ../DERIVED/v2_data_first/train_run/data_first_v2.pt \
  --dataset configs/v2/dataset_mis20s1.json \
  --split configs/v2/split.json \
  --output-dir ../DERIVED/v2_data_first/inference_run
```

```bash
raw-fusion-v2-compare \
  --prediction ../DERIVED/v2_data_first/inference_run \
  --denoised ../DERIVED/v2_data_first/denoised_baseline \
  --output ../DERIVED/v2_data_first/comparison.json
```

See [the protocol](docs/protocol.md) for the data boundary and
[the baseline results](docs/results.md) for the checked-in evidence and its
limitations.
