"""Flow-Guided Residual Fusion Network (FGRF-Net)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DirectionalTransformBank(nn.Module):
    """Learnable multi-scale directional analysis initialized by Sobel filters."""

    def __init__(self, channels: int = 4, scales: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.channels = channels
        self.scales = scales
        self.directions = 4
        self.analysis = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels * self.directions,
                    kernel_size=3,
                    padding=1,
                    groups=channels,
                    bias=False,
                )
                for _ in scales
            ]
        )
        self._initialize_directional_filters()

    def _initialize_directional_filters(self) -> None:
        kernels = torch.tensor(
            [
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
                [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
            ],
            dtype=torch.float32,
        ) / 8.0
        for layer in self.analysis:
            weight = kernels[:, None].repeat(self.channels, 1, 1, 1)
            layer.weight.data.copy_(weight)

    def forward(self, residual: torch.Tensor) -> list[torch.Tensor]:
        output: list[torch.Tensor] = []
        height, width = residual.shape[-2:]
        for scale, layer in zip(self.scales, self.analysis):
            source = residual if scale == 1 else F.avg_pool2d(residual, scale, scale)
            response = layer(source)
            response = response.view(
                residual.shape[0], self.channels, self.directions, response.shape[-2], response.shape[-1]
            )
            if scale != 1:
                response = F.interpolate(
                    response.flatten(1, 2), size=(height, width), mode="bilinear", align_corners=False
                ).view(residual.shape[0], self.channels, self.directions, height, width)
            output.append(response)
        return output


class FGRFNet(nn.Module):
    """Predict gates for injecting trusted texture residuals into 2DNR."""

    def __init__(
        self,
        input_channels: int = 12,
        base_channels: int = 24,
        scales: tuple[int, ...] = (1, 2, 4),
        gate_bias: float = -2.0,
        threshold_floor: float = 0.008,
        initial_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.directions = 4
        self.encoder1 = ConvBlock(input_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.decoder2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.decoder1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.gate_head = nn.Conv2d(base_channels, len(scales) * self.directions, 1)
        self.transform = DirectionalTransformBank(channels=4, scales=scales)
        if initial_threshold <= threshold_floor:
            raise ValueError("initial_threshold must be greater than threshold_floor")
        self.threshold_floor = float(threshold_floor)
        inverse_softplus = torch.log(
            torch.expm1(torch.tensor(initial_threshold - threshold_floor))
        )
        self.threshold = nn.Parameter(
            torch.full((len(scales), self.directions), float(inverse_softplus))
        )
        nn.init.constant_(self.gate_head.bias, gate_bias)
        nn.init.zeros_(self.gate_head.weight)

    @staticmethod
    def _soft_shrink(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return x.sign() * F.relu(x.abs() - threshold)

    def forward(
        self,
        base: torch.Tensor,
        temporal_residual: torch.Tensor,
        noisy_residual: torch.Tensor,
        static_mask: torch.Tensor,
    ) -> dict[str, Any]:
        network_input = torch.cat(
            (
                base,
                temporal_residual,
                noisy_residual,
            ),
            dim=1,
        )
        f1 = self.encoder1(network_input)
        f2 = self.encoder2(f1)
        f3 = self.encoder3(f2)
        d2 = self.decoder2(torch.cat((F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2), dim=1))
        d1 = self.decoder1(torch.cat((F.interpolate(d2, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1), dim=1))
        logits = self.gate_head(d1)
        gates = torch.sigmoid(logits).view(base.shape[0], len(self.scales), self.directions, *base.shape[-2:])
        # The binary mask is computed from RAFT only. Motion pixels have no
        # residual path, including when the learned gate predicts a high value.
        gates = gates * static_mask.unsqueeze(1)

        transformed = self.transform(temporal_residual)
        injected = torch.zeros_like(base)
        gate_index = 0
        for scale_index, response in enumerate(transformed):
            threshold = (
                self.threshold_floor + F.softplus(self.threshold[scale_index])
            ).view(1, 1, self.directions, 1, 1)
            response = self._soft_shrink(response, threshold)
            for direction_index in range(self.directions):
                injected = injected + 0.25 * gates[:, scale_index, direction_index].unsqueeze(1) * response[:, :, direction_index]
                gate_index += 1
        output = (base + injected * static_mask).clamp(0.0, 1.0)
        return {
            "output": output,
            "injected_residual": output - base,
            "gates": gates,
            "transform_responses": transformed,
            "network_input": network_input,
        }
