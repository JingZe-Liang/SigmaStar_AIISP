"""Small two-scale RAW conditioner around an unchanged, frozen official NAFNet."""
import torch
from torch import nn
from torch.nn import functional as F
from company_exposure.core import exposure_steps


class TemporalDetailNAF(nn.Module):
    def __init__(self, backbone, width=24):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        channels = backbone.intro.out_channels
        self.width = width
        self.fine = nn.Sequential(nn.Conv2d(20, width, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        self.coarse = nn.Sequential(nn.Conv2d(width, 2*width, 3, stride=2, padding=1), nn.GELU(),
                                    nn.Conv2d(2*width, 2*width, 3, padding=1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(3*width, width, 3, padding=1), nn.GELU())
        self.to_intro = nn.Conv2d(width, channels, 1)
        self.to_decoder = nn.Conv2d(width, channels, 1)
        for layer in (self.to_intro, self.to_decoder):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def adapter_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def adapter_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('backbone.')}

    def load_adapter(self, state):
        expected = set(self.adapter_state())
        if set(state) != expected:
            raise ValueError('Adapter keys do not match this architecture')
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Nonfinite adapter checkpoint')
        combined = dict(self.state_dict())
        combined.update(state)
        self.load_state_dict(combined, strict=True)

    def forward(self, signal, shot, span, *, training=False, enabled=True):
        """Signed canonical DN12 [B,5,4,H,W] -> three canonical normalized outputs.

        Only observed RAW frames condition the adapter. Targets/masks/teacher
        predictions are never arguments. The official exposure recurrence is reused.
        """
        if signal.ndim != 5 or signal.shape[1:3] != (5, 4) or not torch.isfinite(signal).all():
            raise ValueError('Expected finite [B,5,4,H,W] signed DN12')
        if not enabled:
            return exposure_steps(self.backbone, signal[:, 2], shot, span, training=training)
        batch, _, _, height, width = signal.shape
        scale = torch.as_tensor(span, dtype=signal.dtype, device=signal.device).reshape(batch, 1, -1, 1, 1)
        if scale.shape[2] not in (1, 4) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Positive finite scalar/four-channel DN12 span required')
        # Preserve signed observations in this new branch; only the existing
        # physical recurrence decides where nonnegative values are necessary.
        condition = (signal / scale)[:, :, [0, 1, 3, 2]].reshape(batch, 20, height, width)
        condition = F.pad(condition, (0, (-width) % 16, 0, (-height) % 16), mode='reflect')
        fine = self.fine(condition)
        coarse = F.interpolate(self.coarse(fine), size=fine.shape[-2:], mode='bilinear', align_corners=False)
        features = self.fuse(torch.cat([fine, coarse], 1))
        early, late = self.to_intro(features), self.to_decoder(features)
        # Scoped hooks reuse the vendor forward/head exactly. This wrapper is
        # for sequential eager calls; shared-instance concurrent calls/compile
        # are not supported in this first implementation.
        intro = self.backbone.intro.register_forward_hook(lambda module, args, out: out + early)
        ending = self.backbone.ending.register_forward_pre_hook(lambda module, args: (args[0] + late,))
        try:
            return exposure_steps(self.backbone, signal[:, 2], shot, span, training=training)
        finally:
            intro.remove()
            ending.remove()

