"""Five observed RAW frames -> TAP residual added AFTER frozen ExposureDiffusion.

Only architecture components are transferred; the new branch is not pretrained.
The original three-step photon recurrence and adaptive RAW head are reused.
"""
from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F

from company_exposure.core import exposure_steps, load_student, sha256
from .references import ROOT, official_classes

BASE_SHA = '564d94f3031b9eb6682b13b317860beb4c55a2bed7ed09b34c668a0bc018e7ac'
ARCHITECTURE = 'exposure_output_residual_v3'


def parameter(value, batch, device, name, *, zero=False):
    value = torch.as_tensor(value, dtype=torch.float32, device=device)
    if value.numel() == 1:
        value = value.reshape(1, 1).expand(batch, 1)
    elif value.ndim == 1 and value.numel() == batch:
        value = value.reshape(batch, 1)
    elif batch == 1 and value.shape == (4,):
        value = value[None]
    if value.shape not in ((batch, 1), (batch, 4)):
        raise ValueError(name + ': expected scalar, [B,1] or [B,4]')
    if not torch.isfinite(value).all() or (value < 0 if zero else value <= 0).any():
        raise ValueError(name + ': invalid DN12 calibration')
    return value.expand(batch, 4).reshape(batch, 1, 4, 1, 1)


class PaperBurstNAF(nn.Module):
    def __init__(self, backbone, width=16):
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError('Width must be divisible by 8')
        if (backbone.intro.in_channels != 8 or len(backbone.decoders) != 4
                or backbone.ending.in_channels != backbone.intro.out_channels):
            raise ValueError('Expected four-scale photon-conditioned ExposureDiffusion NAFNet')
        _, _, _, pcd = official_classes()
        self.backbone = backbone.eval().requires_grad_(False)
        self.width = width
        self.stem = nn.Conv2d(8, width, 3, padding=1)
        sizes = [width, width*2, width*4]
        self.downs = nn.ModuleList(nn.Conv2d(c, c*2, 2, stride=2) for c in sizes[:2])
        self.align = nn.ModuleList(pcd(c, deformable_groups=8, n_frames=5,
                                      total_levels=3, current_level=i+1)
                                   for i, c in enumerate(sizes))
        self.up_coarse = nn.Conv2d(sizes[2], sizes[1], 1)
        self.up_fine = nn.Conv2d(sizes[1], sizes[0], 1)
        # Four CANONICAL R,G1,G2,B normalized RAW residual channels.
        self.to_residual = nn.Conv2d(width, 4, 1)
        nn.init.zeros_(self.to_residual.weight)
        nn.init.zeros_(self.to_residual.bias)
        # The output projection alone protects the frozen backbone. Keep TAP's
        # internal residual path open so alignment can learn after that head opens.
        for block in self.align:
            nn.init.constant_(block.beta, .1)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def adapter_parameters(self):
        return (p for name, p in self.named_parameters() if not name.startswith('backbone.'))

    def adapter_state(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items()
                if not k.startswith('backbone.')}

    def load_adapter(self, state):
        expected = self.adapter_state()
        if (set(state) != set(expected) or any(not isinstance(v, torch.Tensor)
                or v.shape != expected[k].shape or v.dtype != expected[k].dtype
                or not torch.isfinite(v).all() for k, v in state.items())):
            raise ValueError('Adapter architecture, dtype, shape or finite check failed')
        combined = dict(self.state_dict())
        combined.update(state)
        self.load_state_dict(combined, strict=True)

    def features(self, signal, shot, span, read_variance, *, temporal=True):
        b, _, _, h, w = signal.shape
        # RAW signal and variance are both signed-mean aware. No clipped noisy
        # high-pass is ever added to the image; this is learned feature input.
        variance = (shot * signal + read_variance).clamp_min(1e-6)
        obs = (signal / span)[:, :, [0, 1, 3, 2]]
        sigma = (variance.sqrt() / span)[:, :, [0, 1, 3, 2]]
        x = torch.cat([obs, sigma], 2).reshape(b*5, 8, h, w)
        x = F.pad(x, (0, (-w) % 16, 0, (-h) % 16), mode='reflect')
        x = self.stem(x)
        scales = []
        for i in range(3):
            scales.append(x.reshape(b, 5, x.shape[1], *x.shape[-2:]))
            if i < 2:
                x = self.downs[i](x)
        fused, transferred, offsets = [None]*3, None, None
        for i in (2, 1, 0):
            value = scales[i]
            if temporal:
                fused[i], transferred, offsets = self.align[i](value.flatten(1, 2), transferred, offsets)
            else:
                fused[i] = value[:, 2]  # Separate center-only structural ablation.
        half = fused[1] + F.interpolate(self.up_coarse(fused[2]), size=fused[1].shape[-2:],
                                       mode='bilinear', align_corners=False)
        fine = fused[0] + F.interpolate(self.up_fine(half), size=fused[0].shape[-2:],
                                        mode='bilinear', align_corners=False)
        return self.to_residual(fine)[..., :h, :w]

    def forward(self, signal, shot, span, read_variance, *, training=False,
                enabled=True, temporal=True, residual_scale=1.):
        """[B,5,4,H,W] signed canonical DN12 -> three normalized canonical frames.

        Read variance is DN12², shot gain DN12/e, span DN12. Targets/teacher/
        masks are deliberately not accepted. The residual is added in normalized canonical RAW, before display ISP.
        Five frames must share exposure,
        gain and black-level calibration. RAW container decoding stays upstream.
        """
        if (signal.ndim != 5 or signal.shape[1:3] != (5, 4) or signal.dtype != torch.float32
                or signal.shape[0] < 1 or min(signal.shape[-2:]) < 16
                or not torch.isfinite(signal).all() or signal.abs().max() > 4095):
            raise ValueError('Expected finite float32 [B,5,4,H,W] signed DN12, H/W >=16')
        batch, device = len(signal), signal.device
        gain = parameter(shot, batch, device, 'shot DN12/e')
        scale = parameter(span, batch, device, 'span DN12')
        read = parameter(read_variance, batch, device, 'read variance DN12²', zero=True)
        if (scale > 4095).any():
            raise ValueError('DN12 span cannot exceed 4095')
        if not isinstance(residual_scale, (int, float)) or not math.isfinite(residual_scale) or not 0 <= residual_scale <= 1:
            raise ValueError('Residual scale must be a finite scalar in [0,1]')
        # No hooks, no internal injection and no gradient through the original
        # model. Its complete three-step trajectory runs unchanged.
        with torch.no_grad():
            outputs = exposure_steps(self.backbone, signal[:, 2], gain[:, 0, :, 0, 0],
                                     scale[:, 0, :, 0, 0], training=training)
        if not enabled or residual_scale == 0:
            return outputs
        residual = self.features(signal, gain, scale, read, temporal=temporal)
        return [*outputs[:2], outputs[-1] + residual_scale * residual]


def load_model(checkpoint, expected_sha=BASE_SHA, *, device='cpu', width=16):
    checkpoint = Path(checkpoint)
    if sha256(checkpoint) != expected_sha:
        raise ValueError('Company NAFNet checkpoint hash mismatch')
    model = PaperBurstNAF(load_student(ROOT, checkpoint, pretrained=False), width)
    return model.to(device)
