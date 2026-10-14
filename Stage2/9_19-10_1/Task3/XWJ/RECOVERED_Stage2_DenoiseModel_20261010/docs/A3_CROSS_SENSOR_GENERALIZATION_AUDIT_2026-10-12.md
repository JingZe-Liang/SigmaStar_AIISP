# A3 Cross-Sensor Generalization Audit

Audit date: 2026-10-12 (Asia/Shanghai)

Finalized: 2026-10-13 (Asia/Shanghai)

Status: complete. The accepted A3 step-5000 checkpoint was evaluated without
training or checkpoint modification on SC465SL, sc635hai, and sc450ai data.

## Final decision

The cross-sensor result is not a blanket pass:

- **SC465SL darkroom18 is supported** at both evaluated gain settings. Both
  parity directions improve native static PSNR by 10.52-13.21 dB, preserve
  about 91-96% of reference high-frequency RMS, and remain fully legal in RAW.
- **sc635hai is conditionally supported.** The static low-CCT sequence passes
  quantitatively in both directions with a 13.59 dB gain. The motion/color
  sequence has coherent motion and no observed ghost duplicate, but it has no
  clean target and therefore cannot establish unrestricted dynamic support.
- **sc450ai is unsupported for deployment with this checkpoint.** The black
  moving-object sequence shows severe high-frequency collapse, softened/hollow
  subject structure, persistent low-frequency artifacts, and 3.28% normalized
  endpoint clipping. The light-changing sequence is less severe but does not
  rescue the sensor-level verdict and has only diagnostic evidence.
- **SC465SL difficultpattern is conditional stress evidence only.** Foreground
  labels, flowers, cables, and major edges survive, but dark regions retain
  low-frequency color mottling and the measured high-frequency ratio falls to
  0.64 and 0.53. These streams are not clean RAW truth.

This audit expands the evidence for using the existing A3 checkpoint on the
explicitly supported domains. It does not replace, overwrite, or mutate A3.

## Scope and invariants

- checkpoint: `artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`
- checkpoint SHA-256:
  `97c4332f685f98959b243e4b100434469f714aed6329e100d22b6d8552592e0b`
- inference weights: EMA from the checkpoint;
- dataset root remained read-only;
- no model training or parameter adaptation was performed;
- no old-teacher output was used as a clean target;
- no existing checkpoint or A3 artifact was replaced;
- dynamic and difficult-pattern streams were evaluated without pseudo-PSNR;
- motion/input and temporal ratios are diagnostic evidence, never a sole pass
  gate.

## Protocol

Every source has an explicit sequence manifest containing width, height, bit
alignment, CFA, nominal black, evaluated frame count, clip starts, fixed-ISP
white balance, exposure, and panel crops. Filename metadata is not trusted when
it conflicts with measured pixels.

Quantitative static protocol:

1. Measure per-channel native black from 16 evenly sampled frames.
2. Build a preliminary temporal mean.
3. Reject per-frame motion using a `median + 3 * 1.4826 * MAD` deviation rule.
4. Keep pixels static in at least 90% of all, even, and odd frames.
5. Evaluate even-only 8-frame clips against the odd-only reference, then the
   reverse. Inputs and targets are mutually exclusive.
6. Report native DN PSNR/RMSE, temporal MAD, bias, legal RAW fraction,
   normalized clipping, high-frequency ratio, flat residual sigma, and usable
   coverage.

Diagnostic stress protocol:

- evaluate contiguous 8-frame clips at early, middle, and late positions;
- report output change, temporal MAD, motion/static temporal ratios, spatial
  high-frequency ratio, legal fraction, and endpoint clipping;
- inspect deterministic `Input t / A3 t / A3 t+1` fixed-ISP panels;
- never treat the temporal mean or an old model output as dynamic ground truth.

A3 retains its original 8-frame temporal input, 64x64 packed tile, 56-pixel
spatial stride, and EMA weights. The inference implementation was generalized
from the original 540x960 packed-size assumption to arbitrary native shapes of
at least 64x64 without changing model arithmetic.

## Source classification

| Sensor / sequence | Native shape | Format | Evaluation role |
|---|---:|---|---|
| SC465SL darkroom18 `256x` | 2560x1440, 300 frames | low-12 RGGB | motion-aware static quantitative |
| SC465SL darkroom18 `2000x` | 2560x1440, 300 frames | low-12 RGGB | motion-aware static quantitative |
| sc635hai `131x_low_cct` | 3200x1800, 150 frames | high-12, effective 10-bit, RGGB | motion-aware static quantitative |
| sc635hai `16x_motion_color` | 3200x1800, 200 frames | high-12, effective 10-bit, RGGB | motion/color stress only |
| sc450ai `1435x_motion_black_object` | 2688x1520, 210 frames | low-aligned, effective 10-bit, RGGB | dark-object motion stress only |
| sc450ai `804x_lightchanging` | 2688x1520, 150 frames | low-aligned, effective 10-bit, RGGB | illumination transition stress only |
| SC465SL difficultpattern `4717` | 2560x1440, 200 frames | high-12 RGGB | detail stress only |
| SC465SL difficultpattern `8880` | 2560x1440, 200 frames | high-12 RGGB | detail stress only |

The sc450ai filename/readme claims of high-12 alignment and black=4128 are
inconsistent with the actual range and quantization and were rejected. The
normal 2DNR/3DNR gain ladder was not used as clean truth.

## Reference quality

| Sequence | Black R/Gr/Gb/B (DN) | Frame-mean CV | Static valid | Split RMSE range | Input PSNR |
|---|---|---:|---:|---:|---:|
| SC465SL `256x` | 254.81 / 258.00 / 258.00 / 254.25 | 0.357% | 85.38% | 3.66-4.49 DN | 41.840 dB |
| SC465SL `2000x` | 251.19 / 252.44 / 252.44 / 250.94 | 0.114% | 95.26% | 2.54-2.92 DN | 46.037 dB |
| sc635hai `131x` | 261.75 / 263.00 / 263.50 / 254.75 | 0.081% | 77.68% | 6.35-8.84 DN | 38.737 dB |

The SC465SL references support a high-confidence ranking. The sc635hai split
reference is noisier, but the A3 improvement is much larger than the split-half
uncertainty and agrees in both parity directions.

## Quantitative static results

`HF input/A3` is RMS high-frequency energy relative to the opposite-parity
reference. Input values above 1 include sensor noise; the A3 value is the useful
comparison to reference detail. `Clip` is the normalized fraction exactly at 0
or 1 after inference, not an out-of-range RAW fraction.

| Sequence / input parity | Input PSNR | A3 PSNR | Gain | Temporal MAD input -> A3 | Bias | HF input / A3 | Flat sigma | Legal | Clip | Usable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SC465SL `256x` even | 41.840 | 55.052 | +13.211 dB | 31.889 -> 1.567 DN | -2.184 DN | 1.672 / 0.957 | 5.830 DN | 100% | 0.603% | 83.67% |
| SC465SL `256x` odd | 41.840 | 54.999 | +13.158 dB | 31.886 -> 1.578 DN | -2.214 DN | 1.675 / 0.958 | 5.854 DN | 100% | 0.592% | 83.66% |
| SC465SL `2000x` even | 46.039 | 56.563 | +10.523 dB | 20.238 -> 1.312 DN | -3.703 DN | 2.560 / 0.911 | 5.189 DN | 100% | 0.083% | 94.28% |
| SC465SL `2000x` odd | 46.034 | 56.647 | +10.613 dB | 20.248 -> 1.301 DN | -3.631 DN | 2.561 / 0.912 | 5.203 DN | 100% | 0.079% | 94.28% |
| sc635hai `131x` even | 38.737 | 52.325 | +13.588 dB | 47.481 -> 2.165 DN | -0.882 DN | 2.108 / 0.919 | 7.473 DN | 100% | 0.743% | 76.54% |
| sc635hai `131x` odd | 38.736 | 52.324 | +13.587 dB | 47.481 -> 2.163 DN | -0.837 DN | 2.109 / 0.921 | 7.481 DN | 100% | 0.725% | 76.54% |

All three static cases pass in both directions. The parity agreement rules out a
single split artifact, and the legal RAW fraction is 1.0 for every output.

## Diagnostic stress results

These values have no clean-target interpretation. In particular, reducing
high-frequency energy can mean both noise removal and detail loss; the fixed-ISP
review is required for the verdict.

| Sequence | Bias | Mean abs change | Temporal MAD input -> A3 | Temporal ratio | Motion / static ratio | Spatial HF | Legal | Clip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sc635hai `16x_motion_color` | -2.369 DN | 29.710 DN | 51.636 -> 14.675 DN | 0.284 | 0.600 / 0.101 | 0.764 | 100% | 2.199% |
| sc450ai `1435x_motion_black_object` | -4.829 DN | 15.367 DN | 21.295 -> 1.347 DN | 0.063 | 0.057 / 0.088 | 0.127 | 100% | 3.280% |
| sc450ai `804x_lightchanging` | -4.646 DN | 15.571 DN | 21.624 -> 1.274 DN | 0.059 | 0.050 / 0.084 | 0.690 | 100% | 1.855% |
| difficultpattern `4717` | -5.459 DN | 24.200 DN | 33.882 -> 1.979 DN | 0.058 | 0.056 / 0.076 | 0.635 | 100% | 0.257% |
| difficultpattern `8880` | -6.376 DN | 18.043 DN | 24.884 -> 1.393 DN | 0.056 | 0.044 / 0.084 | 0.527 | 100% | 0.114% |

## Fixed-ISP review

All panels use the per-sequence black, one fixed white balance, one exposure,
the same Bayer demosaic, and the same sRGB transfer for every column. Panel
hashes and crop coordinates are recorded in each `panel_manifest.json`.

### SC465SL darkroom18

- The `motion_center` panels show a single moving-person silhouette. A3 retains
  the source exposure blur but does not add a second contour, color trail, or
  edge break in either parity direction.
- The fan grille, mannequin boundary, shelf uprights, color boards, floor
  cables, and boxes remain aligned with the opposite-parity static reference.
- The `dark_left` crop reveals room structure after denoising without a tile
  boundary jump. Some low-frequency dark-region residual remains, but no
  teacher-like noise or temporal duplicate appears.

Evidence directories:

- `artifacts/a3_cross_sensor_zero_shot_20261012/sequences/SC465SL/sc465sl_darkroom18_256x/panels`
- `artifacts/a3_cross_sensor_zero_shot_20261012/sequences/SC465SL/sc465sl_darkroom18_2000x/panels`

### sc635hai

- On `131x_low_cct`, color-board boundaries, the checker target, fan grille,
  mannequin, and shelf details are close to the opposite-parity reference in
  both directions. No color discontinuity or warped edge is visible.
- On `16x_motion_color`, the moving person remains one coherent silhouette;
  `A3 t` to `A3 t+1` follows the actual motion while static props remain stable.
  Motion is mildly softened, so this sequence is supporting stress evidence,
  not a clean-target pass.

Evidence directories:

- `artifacts/a3_cross_sensor_extension_20261012/sequences/sc635hai/sc635hai_131x_low_cct/panels`
- `artifacts/a3_cross_sensor_extension_20261012/sequences/sc635hai/sc635hai_16x_motion_color/panels`

### sc450ai

- On `1435x_motion_black_object`, the black garment loses internal structure
  and becomes soft/hollow around its boundary. Large-scale wall/subject
  artifacts remain, the spatial HF ratio is only 0.127, and endpoint clipping
  reaches 3.28%. This is a deployment-blocking failure.
- On `804x_lightchanging`, the illumination change is not frozen and major
  geometry is stable, but dark drape/wall regions show mottled low-frequency
  residual and the moving person is strongly softened. With no clean target,
  this sequence is only conditional and cannot reverse the sensor-level fail.

Evidence directories:

- `artifacts/a3_cross_sensor_extension_20261012/sequences/sc450ai/sc450ai_1435x_motion_black_object/panels`
- `artifacts/a3_cross_sensor_extension_20261012/sequences/sc450ai/sc450ai_804x_lightchanging/panels`

### SC465SL difficultpattern

- Bright foreground flowers, labels, the resolution chart, cable edges, and
  major shelf geometry remain recognizable and stable between `A3 t` and
  `A3 t+1`.
- Dark shelves and walls retain broad color mottling. The `8880` stream is more
  aggressive, with a 0.527 HF ratio. No new double contour is obvious, but the
  data cannot establish clean-detail fidelity.

Evidence directories:

- `artifacts/a3_cross_sensor_extension_20261012/sequences/SC465SL/sc465sl_difficultpattern_4717/panels`
- `artifacts/a3_cross_sensor_extension_20261012/sequences/SC465SL/sc465sl_difficultpattern_8880/panels`

## Sequence verdict matrix

| Sensor / sequence | Evidence | Verdict | Reason |
|---|---|---|---|
| SC465SL darkroom18 `256x` | quantitative + fixed ISP | **supported** | +13.16 dB minimum gain, legal output, 0.957-0.958 HF, acceptable motion/static visuals |
| SC465SL darkroom18 `2000x` | quantitative + fixed ISP | **supported** | +10.52 dB minimum gain, legal output, 0.911-0.912 HF, no critical visual failure |
| sc635hai `131x_low_cct` | quantitative + fixed ISP | **supported** | +13.587 dB minimum gain in both directions, legal output, 0.919-0.921 HF |
| sc635hai `16x_motion_color` | diagnostic + fixed ISP | **conditional** | coherent motion and stable background, but no clean dynamic target |
| sc450ai `1435x_motion_black_object` | diagnostic + fixed ISP | **unsupported** | severe HF collapse, hollow/soft moving object, persistent artifacts, 3.28% clipping |
| sc450ai `804x_lightchanging` | diagnostic + fixed ISP | **conditional** | illumination transition survives, but dark-region mottling and motion softness remain |
| difficultpattern `4717` | diagnostic + fixed ISP | **conditional** | major detail survives, but no clean target and HF is 0.635 |
| difficultpattern `8880` | diagnostic + fixed ISP | **conditional** | no gross edge break, but stronger dark mottling and HF is 0.527 |

## Sensor support matrix

| Sensor | Aggregate status | Authorized claim |
|---|---|---|
| SC465SL | **supported, scope-limited** | A3 step-5000 is supported for the evaluated darkroom18 `256x` and `2000x` domains. Difficult-pattern behavior remains conditional. |
| sc635hai | **conditional** | Static low-CCT zero-shot use is supported. Motion/color use requires a deployment-specific acceptance test because only no-reference stress evidence exists. |
| sc450ai | **unsupported** | Do not deploy A3 step-5000 as a general sc450ai denoiser based on current data; the black-object stress failure blocks a sensor-level claim. |

## Reproduction

```bash
/home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/evaluate_a3_cross_sensor.py \
  --config configs/a3_cross_sensor_zero_shot_20261012.json \
  --output-dir artifacts/a3_cross_sensor_zero_shot_20261012 \
  --phase all --device cuda:0 --tile-batch 2

/home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/evaluate_a3_cross_sensor.py \
  --config configs/a3_cross_sensor_extension_20261012.json \
  --output-dir artifacts/a3_cross_sensor_extension_20261012 \
  --phase all --device cuda:0 --tile-batch 2

/home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/validate_a3_cross_sensor_audit.py \
  --root artifacts/a3_cross_sensor_zero_shot_20261012 \
  --root artifacts/a3_cross_sensor_extension_20261012 \
  --require-complete --verify-sources \
  --output artifacts/a3_cross_sensor_validation_final_20261012.json
```

On this host, PyTorch logical `cuda:0` mapped to the physical RTX 4090. The
inference arrays use resumable per-clip progress files and provenance keys; a
restart skips only fully completed clips with an identical source, checkpoint,
protocol, and tile batch.

## Verification and unfinished work

- complete-mode structural validation: passed for all 8 sequences;
- source SHA-256 verification: passed for all 8 read-only source streams;
- focused cross-sensor/A4 tests: 9 passed;
- full repository suite: 68 passed, with one pre-existing `torch.meshgrid`
  warning;
- bytecode compilation: passed for the cross-sensor evaluator, validator,
  helpers, and tests;
- panel manifests: 44 fixed-ISP PNG panels, all with recorded SHA-256;
- incomplete model evaluation work: none.

A3 remains the accepted checkpoint. This audit changes deployment evidence and
support boundaries only.
