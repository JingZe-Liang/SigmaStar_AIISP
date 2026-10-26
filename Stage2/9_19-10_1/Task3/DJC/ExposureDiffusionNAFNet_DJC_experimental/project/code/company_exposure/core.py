"""Target-free ExposureDiffusion recurrence and region-normalized company loss."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

STUDENT_SHA = '5dbb17ff0edd4bfe58e59644a105f1801f7663ddda3fa90ef03d07111846c3f9'
TEACHER_SHA = '0870c25a4eede7fd77a8da2131c5d5d357c5e9a94e9bbc7d24651aa07d7f2e37'


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_student(code_root, checkpoint, device='cpu', pretrained=True):
    if pretrained and sha256(checkpoint) != STUDENT_SHA:
        raise ValueError('Official student checkpoint hash mismatch')
    path = Path(code_root) / 'vendor/ExposureDiffusion/models/arch/NAFNet.py'
    spec = importlib.util.spec_from_file_location('_company_official_naf', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.NAFNet(4, 4, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 4],
                         dec_blk_nums=[2, 2, 2, 2], resid=False, with_photon=True,
                         concat_origin=False, adaptive_res_and_x0=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['netG'], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite student parameters')
    return model.to(device)


def exposure_steps(model, signal, shot_gain, span, *, training=False, ratio=100):
    """Canonical signed DN12 in; three canonical normalized outputs.

    Same official photon recurrence and Poisson straight-through estimator.
    No target-dependent correction. Batch-size-one also samples two distinct
    interior times, avoiding the original single-sample duplicate-time case.
    """
    if signal.ndim != 4 or signal.shape[1] != 4 or not torch.isfinite(signal).all():
        raise ValueError('Expected finite canonical [B,4,H,W] signal DN12')
    b, _, h, w = signal.shape
    span = torch.as_tensor(span, device=signal.device, dtype=torch.float32).reshape(b, -1, 1, 1)
    shot_gain = torch.as_tensor(shot_gain, device=signal.device, dtype=torch.float32).reshape(b, -1, 1, 1)
    if (span <= 0).any() or (shot_gain <= 0).any():
        raise ValueError('Positive normalization span and DN12/e gain required')
    if span.shape[1] == 4:
        span = span[:, [0, 1, 3, 2]]
    if shot_gain.shape[1] == 4:
        shot_gain = shot_gain[:, [0, 1, 3, 2]]
    origin = (signal[:, [0, 1, 3, 2]].float() / span).clamp(0, 1)
    origin = F.pad(origin, (0, (-w) % 16, 0, (-h) % 16), mode='reflect')
    k = shot_gain * 15583 / (span * ratio)
    if training:
        times = [torch.tensor([1, *sorted(np.random.choice(np.arange(2, ratio), 2, replace=False)), ratio],
                              device=signal.device) for _ in range(b)]
        times = torch.stack(times).float()
    else:
        times = torch.linspace(1, ratio, 4, device=signal.device).expand(b, -1)
    current = origin
    outputs = []
    for i in range(3):
        now = times[:, i].reshape(b, 1, 1, 1)
        nxt = times[:, i + 1].reshape(b, 1, 1, 1)
        photon = current / (ratio / now) / k * 15583
        output, _ = model(torch.cat([current, photon / 15583], dim=1))
        if not torch.isfinite(output).all():
            raise ValueError('Nonfinite ExposureDiffusion output')
        outputs.append(output[:, [0, 1, 3, 2], :h, :w])
        # Preserve the official sample and its straight-through derivative.
        mean = output.clamp(0, 1) / (ratio / (nxt - 1)) / k * 15583
        increment = (torch.poisson(mean) - mean).detach() + mean
        current = (origin / ratio / k * 15583 + increment) / nxt * ratio * k / 15583
    return outputs


def mixed_loss(outputs, target_dn, teacher_dn, static, motion, span):
    if len(outputs) != 3:
        raise ValueError('Exactly three outputs required')
    span = torch.as_tensor(span, device=outputs[0].device).reshape(len(target_dn), -1, 1, 1)
    static, motion = static.bool(), motion.bool()
    if (static & motion).any():
        raise ValueError('Static and motion masks overlap')
    valid_samples = (static | motion).flatten(1).any(1)
    if not valid_samples.all():
        raise ValueError('Sample has no supervised pixels')
    terms = []
    metrics = {'static_mse': 0., 'motion_mse': 0.}
    for output in outputs:
        loss = output.sum() * 0
        for name, mask, target, weight in [('static', static, target_dn, 1.),
                                            ('motion', motion, teacher_dn, .5)]:
            expanded = mask.expand_as(output)
            if expanded.any():
                selected = (output - target / span).square()[expanded]
                term = selected.mean()
                loss = loss + weight * term
                metrics[name + '_mse'] += float(term.detach()) / 3
        terms.append(loss)
    loss = torch.stack(terms).mean()
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite mixed loss')
    return loss, metrics


def signed_signal(observed, black, clip_min=None):
    """Shared explicit DN12 subtraction; never feed container values here."""
    black = np.asarray(black, np.float32).reshape(1, 4, 1, 1)
    value = np.asarray(observed, np.float32) - black
    return value if clip_min is None else np.maximum(value, clip_min)


def guidance_masks(guides, shot_gain, read_variance, black, pooling_factor=None):
    """Odd guidance frames only; masks have [1,H,W] packed-site support."""
    # Four-channel averaged reduced temporal chi-square. A max-range test
    # followed by erosion rejects almost all genuinely static noisy pixels.
    # Process one Bayer plane at a time: a full SC635 tensor plus temporal
    # median workspaces can exhaust the CPU-only server's 2 GiB container.
    score=torch.zeros((1,1,*guides.shape[-2:]),dtype=torch.float32)
    factor=pooling_factor if pooling_factor is not None else [1/25]*4
    global_shift=0.;sigma_mean=0.
    for channel in range(4):
        signal=np.asarray(guides[:,channel:channel+1],np.float32)-np.float32(black[channel])
        low=F.avg_pool2d(torch.from_numpy(signal),5,1,2)
        # Keep the fitted affine variance in raw-DN coordinates: clipping the
        # signed mean makes an uncertain black pedestal change static masks.
        # Only the resulting variance needs a positive numerical floor.
        mean=low.mean(0,keepdim=True)
        variance=(float(shot_gain[channel])*mean+float(read_variance[channel])).clamp_min(1)*float(factor[channel])
        score.add_(low.var(0,unbiased=True,keepdim=True)/variance/4)
        sigma_mean+=float(variance.sqrt().mean())/4
        # Global illumination drift is not evidence of a stationary signal.
        reference=low.median(0,keepdim=True).values
        global_shift=max(global_shift,float((low-reference).flatten(2).median(2).values.abs().amax()))
        del signal,low,mean,variance,reference
    global_limit = max(4., sigma_mean * 3)
    stable = score < 1.5
    moving = score > 3.
    safe = (torch.from_numpy(guides).amax((0, 1)) < 4087)[None, None]
    if global_shift > global_limit:
        stable.zero_()
        moving.zero_()
    stable = -F.max_pool2d(-stable.float(), 5, 1, 2) > .5
    moving &= ~stable
    stable &= safe
    moving &= safe
    return stable[0].numpy(), moving[0].numpy(), {
        'static_fraction': float(stable.float().mean()), 'motion_fraction': float(moving.float().mean()),
        'global_illumination_rejected': global_shift > global_limit,
        'global_shift_dn12': global_shift,
        'median_normalized_temporal_variance':float(score.median()),
    }
