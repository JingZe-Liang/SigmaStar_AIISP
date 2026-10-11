# A7 Real-Motion Capture Pilot

- Date: 2026-10-09 (Asia/Shanghai)
- Scope: Stage2 RAW denoising only
- Baseline retained: `artifacts/finetune_mis20s1/training/checkpoint_step_005000.pth`
- Status: capture protocol and read-only acceptance tooling prepared; no new capture exists yet

## Decision

A5 and A6 showed that the current blocker is dynamic supervision, not an
inability to locate motion. No existing unused company stream provides both
real motion and an exact clean target. The next justified action is a small
MIS20S1 low-gain dynamic capture pilot. Model training stays paused until the
pilot passes the gates in this document.

## Capture Size And Split

- Four training scenes and two validation scenes. A physical scene may appear
  in only one split.
- Five clips per scene, one for each required motion type.
- Each clip contains at least 160 consecutive frames; 200 frames is preferred.
- Minimum total: 30 clips and 4,800 frames. At 1920x1080 uint16 this is about
  20 GB for 160-frame clips or 25 GB for 200-frame clips.
- Validation scenes, actors' paths, foreground layouts, and textures must not
  duplicate the training scenes.

## Required Motion Types

1. `lateral_walk`: left-to-right and right-to-left person motion.
2. `depth_walk`: approach and departure with meaningful scale change.
3. `fast_slow_gesture`: hands, arms, or objects at mixed speeds.
4. `occlusion_reentry`: disappear behind a foreground object and reappear.
5. `fine_texture_motion`: striped/checkered clothing, printed text, hair, cloth,
   or another high-frequency moving surface.

Keep the camera fixed. Motion should occupy useful portions of the frame but
must not turn into full-frame camera motion. Include different speeds,
directions, body sizes, occlusion boundaries, and dark/light clothing across
the six scenes.

## RAW Capture Contract

- Sensor: MIS20S1 scene path, 1920x1080, little-endian uint16.
- Bayer: RGGB, high-12 storage (`dn12 << 4`), black level near 256 DN, white
  level 4095 DN after right shift.
- Capture low-gain clean targets only. Add calibrated A2 noise later to create
  exact same-frame noisy inputs.
- Disable 2DNR, 3DNR, temporal filtering, spatial denoising, sharpening, tone
  mapping, WDR, automatic exposure, automatic gain, and automatic white balance.
- Fix exposure, sensor gain, ISP gain, WB, FPS, illumination, focus, and aperture
  within each clip. Record their numerical values in the manifest.
- Use one low sensor/ISP gain setting and one FPS across the entire pilot. WB and
  exposure may be set per scene but must remain fixed within each clip.
- Use additional illumination and a short enough exposure to keep moving edges
  inspectable. Do not solve brightness by increasing gain.
- Record the native stream directly. Do not decode and re-encode the RAW.

Before the first dynamic clip, capture a static chart and a lens-cap sequence
at the same low-gain pipeline settings. These controls are retained for manual
provenance and black-level checks; they are not training clips.

## Automatic Acceptance Gates

The validator rejects a pilot when any of these conditions fails:

- fewer than four train scenes or two validation scenes;
- train/validation scene overlap;
- any scene missing one of the five motion types;
- multiple gain/FPS settings across clips;
- NR enabled, auto exposure/gain/WB enabled, or tripod not fixed in the manifest;
- fewer than 160 complete frames;
- sampled values are not exactly high-12 aligned;
- robust quiet-region adjacent-frame noise exceeds 8 DN; 5 DN or below is the
  preferred target;
- background-level CV exceeds 1.5% or maximum deviation exceeds 4%;
- saturated samples exceed 1%;
- motion is active in fewer than 25% of sampled adjacent pairs;
- mean changed area is below 1% or above 60% of the frame.

The noise and motion estimators are screening metrics. Passing them does not
replace visual inspection of motion blur, clipping, Bayer phase, real detail,
or confirmation that the hardware path truly bypassed NR.

## Directory And Manifest

Place the manifest at the capture root and use relative RAW paths:

```text
a7_motion_pilot/
  pilot_manifest.json
  train/s01/lateral_walk.raw
  train/s01/depth_walk.raw
  ...
  validation/s06/fine_texture_motion.raw
```

Start from:

`configs/a7_motion_pilot_manifest.example.json`

The example contains one clip to document the fields. A real pilot must contain
all 30 entries and will fail inventory validation until it does.

## Read-Only Validation

```bash
/home/jingzeliang/miniconda3/envs/aaa_312/bin/python \
  scripts/validate_a7_motion_capture.py \
  --manifest /path/to/a7_motion_pilot/pilot_manifest.json \
  --dataset-root /path/to/a7_motion_pilot \
  --out artifacts/finetune_mis20s1/a7_motion_pilot/capture_validation.json
```

The command only reads the capture files. It writes one JSON report inside the
Stage2 repository and exits with code 0 only when every inventory and clip gate
passes.

## Promotion To Training

Do not start A7 training after an automatic pass alone. First review fixed-ISP
frames from every validation clip and confirm:

- moving edges are real and not dominated by exposure blur;
- fine textures remain visible in the low-gain target;
- no temporal denoising trail or spatial denoising texture loss is present;
- black level, CFA, and stored bit alignment match the manifest;
- validation scenes are genuinely independent.

After that review, build exact pairs as:

`input = clean dynamic frame + A2 calibrated noise`

`target = the same clean dynamic frame`

The first A7 experiment must retain A3 step-5000 as the frozen comparison,
train on the four training scenes, and evaluate only on the two held-out scenes
plus the existing 645x, outdoor, and sc635hai/16x motion diagnostics.
