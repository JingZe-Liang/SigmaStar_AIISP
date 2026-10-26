"""Load selected, hash-pinned official classes, without their training imports.

AST selection leaves class bodies unchanged; no timm, Mamba, custom CUDA build
or vendor training entry point is imported. Source/commit attribution is in
vendor/detail_reference_20261022/provenance.json.
"""
import ast
from functools import lru_cache
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d

from company_exposure.core import sha256

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'dark_util': ('vendor/detail_reference_20261022/DarkIR/archs/arch_util.py',
                  'db96fc60cdbc166aacdf15c9c445524a3918e661614ff5275816ca0a089eb1da'),
    'dark_block': ('vendor/detail_reference_20261022/DarkIR/archs/arch_model.py',
                   '3d80012a31cbb1f5a0c07b5efe92f82d2e71668b905ba875f30445ec6b1f4f12'),
    'fre_util': ('vendor/detail_reference_20261022/FrENet/utils/model_util.py',
                 '3e7a1c66e753d51f7e412201d526016339c7fbff6946e350b78cb398a8f3eda9'),
    'fre_block': ('vendor/detail_reference_20261022/FrENet/models/frenet.py',
                  'dfc9bb1093a74bd196eb5a11822456bc974a12ec204fb56ccc4eb4bd9b425c9c'),
    'tap_dcn': ('vendor/detail_reference_20261022/TAP/module_dcn.py',
                '411ac5074d25d8f41aebf95e9abe1f2eaa202a5436a3e25ec841499a59ad18af'),
    'tap_pcd': ('vendor/detail_reference_20261022/TAP/module_pcd_alignment.py',
                '53196efb4ff9f4cc2c5b21afe416c09ec16844c5a169ad8cd2d695aa3d91a7fe'),
}


def select(key, names, namespace):
    path, expected = SOURCES[key]
    path = ROOT / path
    if sha256(path) != expected:
        raise ValueError('Official module source changed: ' + str(path))
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Missing official class: ' + key)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


@lru_cache(maxsize=1)
def official_classes():
    dark = dict(torch=torch, nn=nn, __name__=__name__)
    select('dark_util', ['LayerNormFunction', 'LayerNorm2d'], dark)
    select('dark_block', ['SimpleGate', 'Branch', 'DBlock'], dark)
    fre = dict(torch=torch, nn=nn, math=math, __name__=__name__)
    select('fre_util', ['SimpleGate'], fre)
    select('fre_block', ['LayerNormFunction', 'LayerNorm2d', 'SCA', 'AFPM',
                          'FeedForward', 'FrEBlock'], fre)
    tap = dict(torch=torch, nn=nn, deform_conv2d=deform_conv2d, __name__=__name__)
    select('tap_dcn', ['ModulatedDeformableConv2d'], tap)
    select('tap_pcd', ['PCDAlignment'], tap)
    return dark['DBlock'], fre['FrEBlock'], fre['AFPM'], tap['PCDAlignment']


class VariableAFPM(nn.Module):
    """Official AFPM with its position grid computed for runtime H/W.

    The original forward fixes the number of frequency patches at construction.
    Here the same radius formula/unfold/modulate/fold graph works for rectangular
    full frames too. Kernel size and all learned weights stay fixed. The exact
    official graph is tested against this adapter at matching dimensions.
    """
    def __init__(self, original):
        super().__init__()
        self.K = original.K
        self.kernel_generator_mlp = original.kernel_generator_mlp
        self.pos_bias_mlp = original.pos_bias_mlp
        self.global_conv = original.global_conv

    def forward(self, x):
        b, c, h, w = x.shape
        k = self.K
        if h % k or w % k:
            raise ValueError('AFPM dimensions must be divisible by its frequency kernel')
        y = (torch.arange(h // k, device=x.device, dtype=x.dtype) + .5) * k - h / 2
        xx = (torch.arange(w // k, device=x.device, dtype=x.dtype) + .5) * k - w / 2
        dist = (y[:, None].square() + xx[None, :].square()).sqrt()
        pos = (dist / math.hypot(h / 2, w / 2)).reshape(-1, 1)
        patches = F.unfold(x, k, stride=k).reshape(b, c, k*k, -1)
        kernels = self.kernel_generator_mlp(pos)
        summary = torch.einsum('bckl,lk->bcl', patches, kernels)
        summary = summary + self.pos_bias_mlp(pos).reshape(1, 1, -1)
        count = pos.shape[0]
        summary = summary.transpose(1, 2).reshape(b*count, c, 1, 1)
        modulation = self.global_conv(summary).reshape(b, count, c).transpose(1, 2)
        return F.fold((patches * modulation[:, :, None]).reshape(b, c*k*k, count),
                      (h, w), k, stride=k)
