"""Per-company-profile observation adapter.

Company gain is ``SenserG * IspG / 1024**2``.  The calibrated likelihoods
only cover a finite ISO/gain range, so a profile is eligible only when its
total company gain is covered by that range.  ``IspG`` must not be applied as
a scalar to a calibration RAW: that would also amplify the black map and has
already been falsified by the company stream's measured black-subtracted DN.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from data.signal_mapping import CompanySceneMapping

from .calibrated_observation import CalibratedObservationModel
from .conditional_residual_observation import ConditionalResidualObservationModel
from .temporal_observation import TemporalObservationModel
from .spatial_temporal_observation import SpatialTemporalObservationModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProfileNoiseSelection:
    profile_name: str
    requested_iso: float
    selected_calibration_gain: int
    relative_gain_error: float


class ProfileObservationBank:
    """Select and apply calibrated observation models per company profile."""

    def __init__(
        self,
        calibration_json: str | Path,
        mapping: CompanySceneMapping,
        temporal_measurement_json: str | Path | None = None,
        temporal_validation_json: str | Path | None = None,
        spatial_model_manifest_json: str | Path | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.calibration_json = str(Path(calibration_json).resolve())
        self.mapping = mapping
        self.device = torch.device(device)
        report = json.loads(Path(calibration_json).read_text())
        available = sorted(int(value) for value in report["gains"])
        if not available:
            raise ValueError("calibration report contains no gains")
        self.available_gains = tuple(available)
        self.models: dict[int, CalibratedObservationModel] = {}
        self.temporal_models: dict[str, TemporalObservationModel] = {}
        self.spatial_models: dict[
            str, SpatialTemporalObservationModel | ConditionalResidualObservationModel
        ] = {}
        self._selections: dict[str, ProfileNoiseSelection] = {}

        for name in mapping.profiles:
            requested_iso = mapping.total_gain_for(name)
            if requested_iso < self.available_gains[0] or requested_iso > self.available_gains[-1]:
                continue
            selected = min(
                self.available_gains,
                key=lambda gain: abs(math.log(float(gain) / requested_iso)),
            )
            self._selections[name] = ProfileNoiseSelection(
                profile_name=name,
                requested_iso=requested_iso,
                selected_calibration_gain=selected,
                relative_gain_error=abs(selected - requested_iso) / requested_iso,
            )
        validated_temporal_profiles: set[str] = set()
        if temporal_validation_json is not None:
            validation = json.loads(Path(temporal_validation_json).read_text())
            if not validation.get("passed"):
                validated_temporal_profiles = {
                    item["name"] for item in validation.get("profiles", []) if item.get("passed")
                }
            else:
                validated_temporal_profiles = set(validation.get("covered_profiles", []))
        if temporal_measurement_json is not None:
            temporal_path = Path(temporal_measurement_json)
            report = json.loads(temporal_path.read_text())
            measurements = {item["name"]: item for item in report.get("profiles", [])}
            manifest = json.loads(Path(mapping.manifest_path).read_text())
            for name, measurement in measurements.items():
                if (
                    name not in mapping.profiles
                    or name in self._selections
                    or name not in validated_temporal_profiles
                ):
                    continue
                self.temporal_models[name] = TemporalObservationModel.from_measurement(
                    name, manifest["profiles"][name], measurement, device=self.device
                )
        if spatial_model_manifest_json is not None:
            spatial_manifest_path = Path(spatial_model_manifest_json)
            spatial_manifest = json.loads(spatial_manifest_path.read_text())
            validation_name = spatial_manifest.get("validation_file")
            if not spatial_manifest.get("passed") or not validation_name:
                raise ValueError("spatial noise manifest is not acceptance-approved")
            validation_path = spatial_manifest_path.parent / str(validation_name)
            expected_validation_sha = spatial_manifest.get("validation_sha256")
            if not expected_validation_sha or _sha256(validation_path) != expected_validation_sha:
                raise ValueError("spatial noise validation artifact does not match its manifest")
            validation = json.loads(validation_path.read_text())
            if not validation.get("passed"):
                raise ValueError("spatial noise validation did not pass")
            validated = {item["name"]: item for item in validation.get("profiles", [])}
            for name, item in spatial_manifest.get("profiles", {}).items():
                if name not in mapping.profiles:
                    continue
                if not item.get("accepted") or not validated.get(name, {}).get("accepted"):
                    raise ValueError(f"spatial noise model for {name} is not accepted")
                model_path = Path(item["model_file"])
                if not model_path.is_absolute():
                    model_path = spatial_manifest_path.parent / model_path
                if _sha256(model_path) != item.get("model_sha256"):
                    raise ValueError(f"spatial noise model for {name} does not match its manifest")
                model_type = item.get("model_type", "spatial_temporal_gaussian_v1")
                if model_type == "conditional_residual_v1":
                    self.spatial_models[name] = ConditionalResidualObservationModel.load(
                        model_path, device=self.device
                    )
                elif model_type == "spatial_temporal_gaussian_v1":
                    self.spatial_models[name] = SpatialTemporalObservationModel.load(
                        model_path, device=self.device
                    )
                else:
                    raise ValueError(f"unsupported spatial noise model type {model_type!r}")

    @property
    def white_level(self) -> float:
        return 4095.0

    @property
    def covered_profiles(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._selections) | set(self.temporal_models) | set(self.spatial_models)))

    def selection_for(self, profile_name: str) -> ProfileNoiseSelection:
        try:
            return self._selections[profile_name]
        except KeyError as error:
            if profile_name not in self.mapping.profiles:
                raise KeyError(f"unknown profile {profile_name!r}") from error
            value = self.mapping.total_gain_for(profile_name)
            raise ValueError(
                f"profile {profile_name} has company ISO-equivalent {value:.3f}, "
                f"outside calibrated range {self.available_gains[0]}--{self.available_gains[-1]}"
            ) from error

    def _observation_for(
        self, profile_name: str
    ) -> (
        CalibratedObservationModel
        | TemporalObservationModel
        | SpatialTemporalObservationModel
        | ConditionalResidualObservationModel
    ):
        if profile_name in self.spatial_models:
            return self.spatial_models[profile_name]
        if profile_name in self.temporal_models:
            return self.temporal_models[profile_name]
        return self._model_for(self.selection_for(profile_name).selected_calibration_gain)

    def _model_for(self, gain: int) -> CalibratedObservationModel:
        model = self.models.get(gain)
        if model is None:
            model = CalibratedObservationModel.load(
                self.calibration_json, gain, device=self.device
            )
            self.models[gain] = model
        return model

    @staticmethod
    def _indices(profile_names: Sequence[str], name: str, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [index for index, value in enumerate(profile_names) if value == name],
            device=device,
            dtype=torch.long,
        )

    def sample(
        self,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        profile_names: Sequence[str],
        saturation_margin_dn: float = 8.0,
        saturation_safety_sigma: float = 3.0,
        noise_strength: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample final company DN using one calibrated model per profile."""
        if signal_dn.ndim != 4 or signal_dn.shape[1] != 4:
            raise ValueError("signal_dn must have shape [B,4,H,W]")
        if len(profile_names) != signal_dn.shape[0]:
            raise ValueError("one profile name is required per batch element")
        if not torch.isfinite(signal_dn).all():
            raise ValueError("signal_dn contains non-finite values")
        output = torch.empty_like(signal_dn)
        saturated = torch.empty(
            signal_dn.shape, device=signal_dn.device, dtype=torch.bool
        )
        for name in dict.fromkeys(profile_names):
            indices = self._indices(profile_names, name, signal_dn.device)
            model = self._observation_for(name)
            group_top = torch.as_tensor(top, device=signal_dn.device).reshape(-1)
            group_left = torch.as_tensor(left, device=signal_dn.device).reshape(-1)
            group_top = group_top.index_select(0, indices) if group_top.numel() > 1 else group_top
            group_left = group_left.index_select(0, indices) if group_left.numel() > 1 else group_left
            if noise_strength != 1.0 and not isinstance(model, ConditionalResidualObservationModel):
                raise ValueError("noise strength augmentation requires the empirical residual model")
            sampler = model.sample if noise_strength == 1.0 else model.sample_strength
            extra = {} if noise_strength == 1.0 else {"noise_strength": noise_strength}
            observed, group_saturated = sampler(
                signal_dn.index_select(0, indices),
                group_top,
                group_left,
                saturation_margin_dn=saturation_margin_dn,
                saturation_safety_sigma=saturation_safety_sigma,
                **extra,
            )
            output.index_copy_(0, indices, observed)
            saturated.index_copy_(0, indices, group_saturated)
        return output, saturated

    def observation_to_signal(
        self,
        observation_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        profile_names: Sequence[str],
        clip_min: float | None = 0.0,
    ) -> torch.Tensor:
        """Subtract the selected calibration's fixed dark map per sample."""
        if len(profile_names) != observation_dn.shape[0]:
            raise ValueError("one profile name is required per batch element")
        output = torch.empty_like(observation_dn)
        for name in dict.fromkeys(profile_names):
            indices = self._indices(profile_names, name, observation_dn.device)
            model = self._observation_for(name)
            group_top = torch.as_tensor(top, device=observation_dn.device).reshape(-1)
            group_left = torch.as_tensor(left, device=observation_dn.device).reshape(-1)
            group_top = group_top.index_select(0, indices) if group_top.numel() > 1 else group_top
            group_left = group_left.index_select(0, indices) if group_left.numel() > 1 else group_left
            output.index_copy_(
                0,
                indices,
                model.observation_to_signal(
                    observation_dn.index_select(0, indices), group_top, group_left,
                    clip_min=clip_min,
                ),
            )
        return output

    def limit_signal(
        self,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        profile_names: Sequence[str],
        saturation_margin_dn: float = 8.0,
        saturation_safety_sigma: float = 3.0,
    ) -> torch.Tensor:
        if len(profile_names) != signal_dn.shape[0]:
            raise ValueError("one profile name is required per batch element")
        output = torch.empty_like(signal_dn)
        for name in dict.fromkeys(profile_names):
            indices = self._indices(profile_names, name, signal_dn.device)
            group_top = torch.as_tensor(top, device=signal_dn.device).reshape(-1)
            group_left = torch.as_tensor(left, device=signal_dn.device).reshape(-1)
            group_top = group_top.index_select(0, indices) if group_top.numel() > 1 else group_top
            group_left = group_left.index_select(0, indices) if group_left.numel() > 1 else group_left
            output.index_copy_(0, indices, self._observation_for(name).limit_signal(
                signal_dn.index_select(0, indices), group_top, group_left,
                saturation_margin_dn, saturation_safety_sigma,
            ))
        return output

    def log_prob(
        self,
        observation_dn: torch.Tensor,
        signal_dn: torch.Tensor,
        top: int | torch.Tensor,
        left: int | torch.Tensor,
        profile_names: Sequence[str],
    ) -> torch.Tensor:
        if len(profile_names) != signal_dn.shape[0]:
            raise ValueError("one profile name is required per batch element")
        output = torch.empty_like(signal_dn)
        for name in dict.fromkeys(profile_names):
            indices = self._indices(profile_names, name, signal_dn.device)
            group_top = torch.as_tensor(top, device=signal_dn.device).reshape(-1)
            group_left = torch.as_tensor(left, device=signal_dn.device).reshape(-1)
            group_top = group_top.index_select(0, indices) if group_top.numel() > 1 else group_top
            group_left = group_left.index_select(0, indices) if group_left.numel() > 1 else group_left
            output.index_copy_(0, indices, self._observation_for(name).log_prob(
                observation_dn.index_select(0, indices), signal_dn.index_select(0, indices),
                group_top, group_left,
            ))
        return output

    def noise_std(self, signal_dn: torch.Tensor, profile_names: Sequence[str]) -> torch.Tensor:
        if len(profile_names) != signal_dn.shape[0]:
            raise ValueError("one profile name is required per batch element")
        values = []
        for index, name in enumerate(profile_names):
            values.append(self._observation_for(name).noise_std(signal_dn[index:index + 1]))
        return torch.stack(values).median()

    def metadata(self) -> dict[str, object]:
        return {
            "calibration_json": self.calibration_json,
            "available_calibration_gains": list(self.available_gains),
            "selection_rule": "nearest calibration gain in log total ISO-equivalent",
            "eligible_profiles": sorted(self._selections),
            "temporal_likelihood_profiles": sorted(self.temporal_models),
            "spatial_temporal_likelihood_profiles": sorted(self.spatial_models),
            "temporal_validation_required": False,
            "ineligible_profiles": sorted(
                set(self.mapping.profiles) - set(self._selections) - set(self.temporal_models) - set(self.spatial_models)
            ),
            "profiles": {
                name: {
                    "requested_iso": selection.requested_iso,
                    "selected_calibration_gain": selection.selected_calibration_gain,
                    "relative_gain_error": selection.relative_gain_error,
                }
                for name, selection in self._selections.items()
            },
        }
