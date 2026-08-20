from __future__ import annotations

import torch

from .confidence import SafetyParams


def build_threshold_normalized_features(
    current_source: torch.Tensor,
    current_2dnr: torch.Tensor,
    current_3dnr: torch.Tensor,
    previous_2dnr: torch.Tensor,
    safety: SafetyParams,
) -> torch.Tensor:
    """Build candidate-aware features whose deltas transfer across noise levels."""

    disagreement_scale = max(4.0 * safety.disagreement_threshold, 1e-8)
    motion_scale = max(4.0 * safety.motion_threshold, 1e-8)
    candidate_delta = torch.clamp(
        (current_3dnr - current_2dnr) / disagreement_scale, -1.0, 1.0
    )
    temporal_delta = torch.clamp(
        (current_2dnr - previous_2dnr) / motion_scale, -1.0, 1.0
    )
    return torch.cat(
        (
            current_source,
            current_2dnr,
            current_3dnr,
            previous_2dnr,
            candidate_delta,
            temporal_delta,
        ),
        dim=1,
    )
