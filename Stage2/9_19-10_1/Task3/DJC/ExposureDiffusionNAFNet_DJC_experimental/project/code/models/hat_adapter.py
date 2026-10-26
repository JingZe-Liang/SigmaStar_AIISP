"""Incremental RAW adapter using the HAT authors' actual HAB and OCAB blocks.

Block source/license: models/vendor_hat/. This wrapper adds shape handling only;
the zero gate and insertion into our CNN are project adaptations, not HAT claims.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .vendor_hat.hab import HAB, OCAB


class AuthorHATHybridAdapter(nn.Module):
    def __init__(self, channels, heads=8, window=8):
        super().__init__()
        if channels % heads or window < 2 or window % 2:
            raise ValueError("HAT requires channels divisible by heads and an even window >=2")
        self.window = window
        self.hab = HAB(channels, (window, window), heads, window_size=window, shift_size=0, squeeze_factor=min(30, channels))
        self.oca = OCAB(channels, (window, window), window, 0.5, heads)
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing='ij')).flatten(1)
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        relative += window - 1
        relative[..., 0] *= 2 * window - 1
        self.register_buffer('rpi_sa', relative.sum(-1), persistent=False)
        extended = window + window // 2
        ext_coords = torch.stack(torch.meshgrid(torch.arange(extended), torch.arange(extended), indexing='ij')).flatten(1)
        relative = (ext_coords[:, None, :] - coords[:, :, None]).permute(1, 2, 0).contiguous()
        relative += window - extended + 1
        relative[..., 0] *= window + extended - 1
        self.register_buffer('rpi_oca', relative.sum(-1), persistent=False)

    def forward(self, video):
        b, t, c, h, w = video.shape
        ph, pw = (-h) % self.window, (-w) % self.window
        flat = video.reshape(b * t, c, h, w)
        flat = F.pad(flat, (0, pw, 0, ph), mode='reflect' if ph < h and pw < w else 'replicate')
        height, width = flat.shape[-2:]
        tokens = flat.flatten(2).transpose(1, 2)
        tokens = self.hab(tokens, (height, width), self.rpi_sa, None)
        tokens = self.oca(tokens, (height, width), self.rpi_oca)
        return tokens.transpose(1, 2).reshape(b, t, c, height, width)[..., :h, :w]
