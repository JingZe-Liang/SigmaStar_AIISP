"""Four-class cell selector with fail-closed confidence thresholds."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .bands import a1


ALPHA_VALUES = torch.tensor([0.0, 0.125, 0.25, 0.5], dtype=torch.float32)
PROBABILITY_THRESHOLD = 0.80
MARGIN_THRESHOLD = 0.30


@dataclass(frozen=True, slots=True)
class SelectorOutput:
    q_logits_cell: Tensor
    p_cell: Tensor
    q_cell: Tensor
    q: Tensor
    class_index: Tensor
    confidence: Tensor
    valid: Tensor

    @property
    def probabilities(self) -> Tensor:
        return self.p_cell


CELL_SIZE = 32


def _reduce_logits(logits: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 4 or logits.shape[1] != 4:
        raise ValueError("q_logits_pixel must have shape [B,4,H,W]")
    mask = valid_mask.to(device=logits.device, dtype=logits.dtype)
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape[0] != logits.shape[0] or mask.shape[-2:] != logits.shape[-2:]:
        raise ValueError("valid_mask must match logits batch/spatial dimensions")
    if mask.shape[1] != 1:
        raise ValueError("valid_mask must have one shared spatial channel")
    height, width = logits.shape[-2:]
    pad_bottom = (-height) % CELL_SIZE
    pad_right = (-width) % CELL_SIZE
    padded_mask = F.pad(mask, (0, pad_right, 0, pad_bottom), value=0.0)
    padded_logits = F.pad(logits * mask, (0, pad_right, 0, pad_bottom), value=0.0)
    count = F.avg_pool2d(padded_mask, CELL_SIZE, stride=CELL_SIZE) * float(CELL_SIZE * CELL_SIZE)
    summed = F.avg_pool2d(padded_logits, CELL_SIZE, stride=CELL_SIZE) * float(CELL_SIZE * CELL_SIZE)
    if not bool(torch.all(count > 0)):
        raise ValueError("CellMean32 encountered a cell with zero valid pixels")
    return summed / count, count > 0


def select_q(q_logits_pixel: Tensor, valid_mask: Tensor) -> SelectorOutput:
    logits = q_logits_pixel.to(dtype=torch.float32)
    pooled, cell_has_valid = _reduce_logits(logits, valid_mask)
    probabilities = F.softmax(pooled, dim=1)
    top_probability, winner = probabilities.max(dim=1, keepdim=True)
    without_winner = probabilities.scatter(1, winner, float("-inf"))
    second_probability = without_winner.max(dim=1, keepdim=True).values
    confident = (
        (top_probability >= PROBABILITY_THRESHOLD)
        & ((top_probability - second_probability) >= MARGIN_THRESHOLD)
        & cell_has_valid
    )
    class_index = torch.where(confident, winner, torch.zeros_like(winner))
    alpha_values = ALPHA_VALUES.to(device=logits.device)
    alpha = alpha_values[class_index]
    # Straight-through hard forward: gradients follow the pooled soft alpha,
    # while the deployed value is exactly one of the four contract classes.
    soft_alpha = (probabilities * alpha_values[None, :, None, None]).sum(dim=1, keepdim=True)
    q_cell = alpha + (soft_alpha - soft_alpha.detach())
    expanded = F.interpolate(q_cell, size=(q_cell.shape[-2] * CELL_SIZE, q_cell.shape[-1] * CELL_SIZE), mode="nearest")
    expanded = expanded[..., : logits.shape[-2], : logits.shape[-1]]
    q = a1(expanded) if min(expanded.shape[-2:]) > 2 else expanded
    confidence = top_probability
    return SelectorOutput(
        q_logits_cell=pooled,
        p_cell=probabilities,
        q_cell=q_cell,
        q=q,
        class_index=class_index,
        confidence=confidence,
        valid=confident,
    )
