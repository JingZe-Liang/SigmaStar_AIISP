# A8 Sensor Adaptation Data Gate

Audit date: 2026-10-13 (Asia/Shanghai)

Status: blocked by the absence of a compliant, non-audit training source for
the requested target sensors.

## Decision

No A8 sensor fine-tuning was started. Under the currently authorized protocol,
starting backpropagation would require at least one source that is both outside
the frozen eight-sequence cross-sensor audit and backed by an approved target
semantics. The existing data satisfies neither condition for sc635hai or
sc450ai.

This is a data-eligibility failure, not a GPU or implementation failure. The
RTX 4090 has enough remaining memory for a conservative stage8-only experiment,
but compute availability does not make an invalid target trustworthy.

## Machine-checked result

The eligibility gate inspected file existence, byte-derived frame counts,
source location, current inode identity, current full-file SHA-256, and target
semantics. On every run it recomputes the content hash of all eight frozen RAW
files. It locks the authoritative dataset root and the final A3 validation
file, whose expected SHA-256 is
`14538b3495af329252c075c65b71d0c2e638c9eda28bde632543d10d40cb8f64`.
It also cross-checks each validation sequence, source manifest, and data-report
provenance against a code-level table containing the exact eight frozen
sequence names, source paths, byte sizes, and source SHA-256 values. It did not
modify or hash-rewrite any dataset file.

For an otherwise eligible future source, the gate computes its content hash to
reject a copied frozen RAW, then requires a hash-bound supervision manifest,
independent target artifacts, and disjoint input/target frame pools where the
declared semantics require them. Required sensor identities and every known
candidate path are also code-level constants. The authoritative approved
training-source table is currently empty, so editing the JSON configuration
alone cannot authorize backpropagation.

| Sensor | Candidates | Trainable | Frozen-audit blocks | Target-semantics blocks |
|---|---:|---:|---:|---:|
| sc635hai | 2 | 0 | 2 | 0 |
| sc450ai | 2 | 0 | 2 | 0 |
| SC465SL non-audit normal streams | 8 | 0 | 0 | 8 |

The authoritative result is:

- `artifacts/a8_sensor_adaptation/data_audit_20261013.json`
- status: `blocked_no_compliant_training_source`
- frozen A3 audit sources loaded from the hash-locked final validator: 8
- required sensors without a trainable source: `sc635hai`, `sc450ai`

## Per-sensor evidence

### sc635hai

The configured inventory contains two RAW streams. A separate read-only path
inventory across `/HardDisk/jingzeliang/Data`, `/Data2`, `/Data4`, and
`/home/jingzeliang` found no additional sc635hai stream; the gate itself does
not perform filesystem-wide discovery.

- `131x_low_cct`, 150 frames: it has the only usable static reference evidence,
  but it is part of the frozen A3 cross-sensor audit;
- `16x_motion_color`, 200 frames: it is also frozen and has no independent
  clean dynamic target.

Consequently, zero sc635hai frames are authorized for A8 backpropagation.

### sc450ai

The configured inventory contains two RAW streams, and the same separate
read-only path inventory found no additional sc450ai stream:

- `1435x_motion_black_object`, 210 frames;
- `804x_lightchanging`, 150 frames.

Both streams are frozen audit sources and both are dynamic or nonstationary.
Neither an old-teacher output nor a temporal average is an acceptable clean
target. Consequently, zero sc450ai frames are authorized for A8
backpropagation.

### SC465SL auxiliary data

Eight non-audit `465_sh_normal` streams are present, totaling 1500 frames.
Their available evidence identifies them as processed display-oriented streams,
not independent clean RAW references. They can remain no-reference stress
material, but the gate rejects them as supervision.

## Required protocol choice

There are only two scientifically distinct next states:

1. Keep the eight audit sequences frozen. No target-sensor fine-tuning can be
   performed with the current files. A new independent existing dataset would
   be required; no new capture is assumed or requested here.
2. Explicitly revise the protocol and reclassify one sequence per sensor as
   adaptation data. A possible experimental split is sc635hai `131x` for
   static adaptation with `16x` held out, and sc450ai `804x` for explicitly
   exploratory self-supervision with `1435x` held out. This would invalidate
   the reclassified streams as independent A8 tests and requires direct user
   authorization before implementation or training.

The second option has not been assumed. The immutable A3 step-5000 checkpoint
and all existing A3 audit artifacts remain unchanged.

## Reproduction

```bash
/home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/audit_a8_sensor_training_data.py \
  --config configs/a8_sensor_training_data_20261013.json \
  --output artifacts/a8_sensor_adaptation/data_audit_20261013.json

/home/jingzeliang/miniconda3/envs/aaa_312/bin/python -m pytest -q \
  src/tests/test_finetune_a8_data.py
```

The audit command writes the complete JSON result first and then returns exit
code 2 when the gate is blocked. Exit code 0 is reserved for a configuration
where every required sensor has at least one compliant trainable source.

The focused eligibility tests pass: 15 passed. They include hard-link and
same-content aliases, frozen-validator and source-manifest replacement,
authoritative-root substitution, JSON type confusion, overlapping supervision
frames, duplicated noisy-source targets, candidate sensor reassignment, and
configuration-only source approval. The combined A8, cross-sensor, and A4
regression selection passes: 24 passed. No sensor adaptation checkpoint
or ISP comparison video exists yet because no sensor has passed the training
data gate.
