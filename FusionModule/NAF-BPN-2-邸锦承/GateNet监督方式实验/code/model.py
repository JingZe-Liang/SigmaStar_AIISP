from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(1, keepdim=True)
        variance = (inputs - mean).pow(2).mean(1, keepdim=True)
        return (inputs - mean) / torch.sqrt(variance + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first, second = inputs.chunk(2, dim=1)
        return first * second


class SCA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.conv(self.pool(inputs))


class NAFBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.gate1 = SimpleGate()
        self.sca = SCA(channels)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels * 2, 1)
        self.gate2 = SimpleGate()
        self.conv5 = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.conv3(self.sca(self.gate1(self.conv2(self.conv1(self.norm1(inputs))))))
        outputs = inputs + features * self.beta
        return outputs + self.conv5(self.gate2(self.conv4(self.norm2(outputs)))) * self.gamma


class NAFBPNMotionFusionNet(nn.Module):
    """NAF-BPN fusion model with deployment-safe temporal input.

    The fourth input is the previous noisy/source frame.  Offline motion
    masks are deliberately kept out of this model and are used only by the
    weak-supervision loss during training.
    """

    def __init__(self, num_basis: int = 15, kernel_size: int = 7, width: int = 32):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("BPN kernel_size 必须为奇数")
        self.num_basis, self.kernel_size = num_basis, kernel_size
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.encoders, self.downs, self.decoders, self.ups = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        channels = width
        for _ in range(3):
            self.encoders.append(nn.Sequential(NAFBlock(channels)))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle_blks = nn.Sequential(NAFBlock(channels))
        for _ in range(3):
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(nn.Sequential(NAFBlock(channels)))
        self.coeff_head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.Conv2d(width, num_basis, 1), nn.Softmax(dim=1))
        self.basis_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(width * 8, width * 4, 1), nn.Flatten(), nn.Linear(width * 4, num_basis * 2 * kernel_size * kernel_size))
        center = kernel_size // 2
        allowed = torch.zeros(2, kernel_size, kernel_size, dtype=torch.bool)
        for row in range(kernel_size):
            for column in range(kernel_size):
                allowed[:, row, column] = (row - center) % 2 == 0 and (column - center) % 2 == 0 and not (row == center and column == center)
        self.register_buffer("basis_allowed", allowed, persistent=False)

    def _basis(self, bottleneck: torch.Tensor) -> torch.Tensor:
        logits = self.basis_head(bottleneck).reshape(-1, self.num_basis, 2, self.kernel_size, self.kernel_size)
        illegal = ~self.basis_allowed.unsqueeze(0).unsqueeze(0)
        logits = logits.masked_fill(illegal, torch.finfo(logits.dtype).min)
        return F.softmax(logits.flatten(2), dim=-1).reshape_as(logits)

    def apply_basis_fusion(self, image_2dnr: torch.Tensor, image_3dnr: torch.Tensor, basis: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        batch_size, basis_count, _, kernel_size, _ = basis.shape
        sources = torch.cat([image_2dnr, image_3dnr], dim=1)
        kernels = basis.reshape(batch_size * basis_count, 2, kernel_size, kernel_size)
        stacked = sources.unsqueeze(1).expand(-1, basis_count, -1, -1, -1).reshape(1, batch_size * basis_count * 2, *sources.shape[-2:])
        filtered = F.conv2d(stacked, kernels, padding=kernel_size // 2, groups=batch_size * basis_count)
        return (filtered.reshape(batch_size, basis_count, *sources.shape[-2:]) * coefficients).sum(dim=1, keepdim=True)

    def forward(
        self,
        image_2dnr: torch.Tensor,
        image_3dnr: torch.Tensor,
        noisy_current: torch.Tensor,
        noisy_previous: torch.Tensor,
    ) -> torch.Tensor:
        inputs = (image_2dnr, image_3dnr, noisy_current, noisy_previous)
        if any(item.ndim != 4 for item in inputs):
            raise ValueError("NAF-BPN 输入必须为 BCHW")
        if any(item.shape[1] != 1 for item in inputs):
            raise ValueError("NAF-BPN 主线使用单通道 Bayer 输入")
        if len({tuple(item.shape[-2:]) for item in inputs}) != 1:
            raise ValueError("NAF-BPN 输入的空间尺寸必须一致")

        motion_feature = F.avg_pool2d(
            torch.abs(noisy_current - noisy_previous), 5, stride=1, padding=2
        )
        algorithm_difference = F.avg_pool2d(torch.abs(image_3dnr - image_2dnr), 5, stride=1, padding=2)
        features = self.intro(torch.cat([noisy_current, motion_feature, algorithm_difference], dim=1))
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features)
            skips.append(features)
            features = down(features)
        bottleneck = self.middle_blks(features)
        features = bottleneck
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            features = decoder(up(features) + skip)
        return self.apply_basis_fusion(image_2dnr, image_3dnr, self._basis(bottleneck), self.coeff_head(features))


class PackedAlphaFusionNet(nn.Module):
    """轻量 packed Bayer 凸融合网络。

    The fusion path never consumes an externally generated motion mask.  A
    motion head is exposed only as an optional training auxiliary output.
    """

    def __init__(self, width: int = 24, depth: int = 2):
        super().__init__()
        if width < 8 or depth < 1:
            raise ValueError("width 必须至少为 8，depth 必须为正数")
        self.width = width
        self.depth = depth
        channels = [width, width * 2, width * 4]
        self.stem = nn.Conv2d(16, channels[0], 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for level in range(2):
            self.encoders.append(nn.Sequential(*[NAFBlock(channels[level]) for _ in range(depth)]))
            self.downs.append(nn.Conv2d(channels[level], channels[level + 1], 2, stride=2))
        self.middle = nn.Sequential(*[NAFBlock(channels[2]) for _ in range(depth)])
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in (1, 0):
            self.ups.append(nn.Sequential(
                nn.Conv2d(channels[level + 1], channels[level] * 4, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            self.decoders.append(nn.Sequential(*[NAFBlock(channels[level]) for _ in range(depth)]))
        self.alpha_head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.Conv2d(width, 4, 1))
        self.motion_head = nn.Sequential(nn.Conv2d(width, width // 2, 3, padding=1), nn.GELU(), nn.Conv2d(width // 2, 1, 1))
        nn.init.zeros_(self.alpha_head[-1].weight)
        nn.init.constant_(self.alpha_head[-1].bias, -2.0)
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)

    def encode(self, image_2dnr: torch.Tensor, image_3dnr: torch.Tensor, noisy_current: torch.Tensor, noisy_previous: torch.Tensor) -> torch.Tensor:
        if any(item.ndim != 4 for item in (image_2dnr, image_3dnr, noisy_current, noisy_previous)):
            raise ValueError("packed 输入必须为 BCHW")
        if any(item.shape[1] != 4 for item in (image_2dnr, image_3dnr, noisy_current, noisy_previous)):
            raise ValueError("packed Bayer 输入必须包含 4 个 CFA 通道")
        if len({tuple(item.shape[-2:]) for item in (image_2dnr, image_3dnr, noisy_current, noisy_previous)}) != 1:
            raise ValueError("packed 输入的空间尺寸必须一致")
        features = torch.cat(
            [image_2dnr, image_3dnr - image_2dnr, noisy_current - image_2dnr, noisy_current - noisy_previous],
            dim=1,
        )
        features = self.stem(features)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features)
            skips.append(features)
            features = down(features)
        features = self.middle(features)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            features = decoder(up(features) + skip)
        return features

    def forward(
        self,
        image_2dnr: torch.Tensor,
        image_3dnr: torch.Tensor,
        noisy_current: torch.Tensor,
        noisy_previous: torch.Tensor,
        return_aux: bool = False,
    ):
        features = self.encode(image_2dnr, image_3dnr, noisy_current, noisy_previous)
        alpha = torch.sigmoid(self.alpha_head(features))
        output = image_2dnr + alpha * (image_3dnr - image_2dnr)
        if return_aux:
            return output, alpha, self.motion_head(features)
        return output

    @torch.no_grad()
    def predict_alpha(self, image_2dnr: torch.Tensor, image_3dnr: torch.Tensor, noisy_current: torch.Tensor, noisy_previous: torch.Tensor) -> torch.Tensor:
        features = self.encode(image_2dnr, image_3dnr, noisy_current, noisy_previous)
        return torch.sigmoid(self.alpha_head(features))


def extract_model_state(payload):
    """Extract a state dict from legacy raw or structured checkpoints."""
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError("checkpoint 必须是 state_dict 或包含 model/state_dict 的字典")
    if any(key.startswith("module.") for key in payload):
        payload = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in payload.items()
        }
    return payload


def load_pretrained_model(checkpoint_path: str, device: torch.device) -> NAFBPNMotionFusionNet:
    model = NAFBPNMotionFusionNet().to(device)
    state = extract_model_state(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    model.load_state_dict(state, strict=True)
    return model


def forward_padded(model: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    height, width = inputs[0].shape[-2:]
    pad_height, pad_width = (-height) % 8, (-width) % 8
    if not pad_height and not pad_width:
        return model(*inputs)
    padded = [F.pad(item, (0, pad_width, 0, pad_height), mode="reflect") for item in inputs]
    return model(*padded)[..., :height, :width]
