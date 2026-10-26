"""Frozen ExposureDiffusion + official BIPNet burst-denoising residual.

The vendor file stays byte-for-byte unchanged. Only Lightning/metric scaffolding
is stripped on import; all official forward methods remain identical.
"""
import ast
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import DeformConv2d
from company_exposure.core import exposure_steps

VENDOR_SHA = '5ddf490a974139e1099c5c70dece28661acff8d88150581506d8f3aead447e66'

def official_class(vendor):
    source = Path(vendor).read_bytes()
    if hashlib.sha256(source).hexdigest() != VENDOR_SHA:
        raise ValueError('Pinned official BIPNet source changed')
    tree = ast.parse(source)
    # Drop framework imports, import-time global seeding and metric construction.
    # No neural computation in any forward method is changed.
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    for node in tree.body:
        if node.name == 'BIPNet':
            node.bases = [ast.Attribute(ast.Name('nn', ast.Load()), 'Module', ast.Load())]
            init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
            init.body = [n for n in init.body if not (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == 'valid_psnr' for t in n.targets))]
    namespace = dict(torch=torch, nn=nn, F=F, DeformConv2d=DeformConv2d)
    exec(compile(ast.fix_missing_locations(tree), str(vendor), 'exec'), namespace)
    return namespace['BIPNet']

class BIPNetDetail(nn.Module):
    def __init__(self, backbone, vendor):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.detail = official_class(vendor)(mode='color', num_features=64, burst_size=5)
        # The only network-layer substitutions: RGB+noise -> packed RAW+noise,
        # and RGB restoration -> zero-initialized packed RAW residual.
        self.detail.conv1[0] = nn.Conv2d(8, 64, 3, padding=1, bias=False)
        self.detail.conv3[0] = nn.Conv2d(64, 4, 3, padding=1, bias=False)
        nn.init.zeros_(self.detail.conv3[0].weight)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, signal, shot, span, read_variance, *, repeated_center=False):
        if signal.ndim != 5 or signal.shape[1:3] != (5,4) or not torch.isfinite(signal).all():
            raise ValueError('Expected finite canonical signed DN12 [B,5,4,H,W]')
        b, _, _, h, w = signal.shape
        def parameter(value, name, positive=False):
            value = torch.as_tensor(value, device=signal.device, dtype=signal.dtype).reshape(b, -1)
            if value.shape[1] not in (1,4) or not torch.isfinite(value).all() or (value < 0).any() or (positive and (value <= 0).any()):
                raise ValueError('Invalid '+name+' in DN12 units')
            return value[:,None,:,None,None]
        scale = parameter(span, 'span', True)
        gain = parameter(shot, 'shot gain', True)
        read = parameter(read_variance, 'read variance')
        with torch.no_grad():
            baseline = exposure_steps(self.backbone, signal[:,2], shot, span, training=False)[-1]
        observed = signal[:,2:3].expand_as(signal) if repeated_center else signal
        normalized = observed / scale
        # Approximate per-pixel standard deviation, not a clean detail target.
        sigma = (gain*observed + read).clamp_min(1e-8).sqrt()/scale
        # BIPNet reference is frame 0; chronological inputs have center at 2.
        order = [2,0,1,3,4]
        normalized, sigma = normalized[:,order], sigma[:,order]
        pad = (0, (-w)%4, 0, (-h)%4)
        normalized = F.pad(normalized.flatten(0,1), pad, mode='replicate').reshape(b,5,4,h+pad[3],w+pad[1])
        sigma = F.pad(sigma.flatten(0,1), pad, mode='replicate').reshape_as(normalized)
        # Official forward explicitly selects burst[0]. Never silently discard
        # other batch samples: call its unchanged graph once per sample.
        residual = torch.cat([self.detail(normalized[i:i+1], sigma[i:i+1]) for i in range(b)])[:,:,:h,:w]
        if not torch.isfinite(residual).all():
            raise ValueError('Nonfinite BIPNet residual')
        return baseline + residual, baseline
