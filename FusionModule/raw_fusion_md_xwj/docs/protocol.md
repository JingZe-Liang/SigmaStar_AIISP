# Data-First V2 Protocol

## Boundary

Each deployment sample supplies four aligned packed-RGGB RAW tensors:
`noisy(t-1)`, `noisy(t)`, `denoised(t)`, and `fused(t)`.

The runtime normalizes noisy inputs with black level 252 and denoised/fused
inputs with black level 300, using a 3795 DN range. The model receives those
four tensors and the derived continuous noise condition `c_tilde`.

MOG2 is not an input at deployment. During training it produces a static/motion
mask used to construct supervision. The optional offline cache stores only
those masks and is validated against the configured dataset before use.

## Selection and composition

The model predicts four q-class logits per pixel. Inference averages logits in
32x32 cells, applies softmax, and accepts a nonzero class only when the top
probability is at least 0.80 and exceeds the runner-up by at least 0.30.

The selected class maps to q in `{0, 0.125, 0.25, 0.5}`. The q map is smoothed
with A1 and composed only with `B2(fused - denoised)`.

`limit_q_to_raw_range` then limits q per location so the normalized candidate
remains in `[0, 1]` for every packed RAW channel. It protects RAW range but
does not create nonzero q values.

## Training

The sampler builds paired positive and zero-q crops from cached or online MOG2
supervision. Loss combines selected-cell cross entropy and q regression with
high-frequency, low-frequency preservation, zero-q, range, and smoothness
terms. Metrics are appended to `metrics.jsonl`; periodic checkpoints and the
final `data_first_v2.pt` are written atomically.

Current supervision is conservative. The 645x condition has too few admitted
texture cells under the fixed threshold, so future work should use
condition-aware, cell-level fusion-utility supervision rather than lowering
the inference confidence threshold.
