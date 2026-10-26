"""Map H5 clean content into measured company-scene brightness conditions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class CompanySceneProfile:
    name: str
    scene_kind: str
    luma_median_dn: float
    black_level_dn: float
    exposure_us: int | None
    sensor_gain_code: int | None
    isp_gain_code: int | None
    calibration_gain_estimate: float | None
    analog_gain_factor: float | None
    isp_gain_factor: float | None
    total_gain_factor: float | None
    analog_iso_equivalent: float | None
    iso_equivalent: float | None


@dataclass(frozen=True)
class CompanySceneMapping:
    """Relative exposure mapping anchored to measured company RAW profiles."""

    source_luma_median_dn: float
    source_black_level_dn: float
    profiles: Mapping[str, CompanySceneProfile]
    training_profiles: tuple[str, ...]
    validation_profile: str
    test_profile: str
    manifest_path: str
    target_calibration_gain: int | None = None

    def __post_init__(self) -> None:
        if self.source_luma_median_dn <= 0:
            raise ValueError("H5 source luma median must be positive")
        if self.source_black_level_dn < 0:
            raise ValueError("H5 source black level must be nonnegative")
        if self.target_calibration_gain is not None and self.target_calibration_gain <= 0:
            raise ValueError("target calibration gain must be positive when provided")
        names = set(self.profiles)
        requested = set(self.training_profiles) | {
            self.validation_profile,
            self.test_profile,
        }
        missing = requested - names
        if missing:
            raise KeyError(f"company scene profiles missing from manifest: {sorted(missing)}")
        if not self.training_profiles:
            raise ValueError("at least one training company profile is required")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CompanySceneMapping":
        if not isinstance(config, dict):
            raise ValueError("data.signal_mapping must be a mapping")
        manifest_path = Path(config["profiles_manifest"])
        manifest = json.loads(manifest_path.read_text())
        h5_source = manifest["h5_source"]
        profiles = {
            name: CompanySceneProfile(
                name=name,
                scene_kind=str(item["scene_kind"]),
                luma_median_dn=float(item["luma_median_dn"]),
                black_level_dn=float(item["black_level_dn"]),
                exposure_us=(
                    int(item["exposure_us"])
                    if item.get("exposure_us") is not None else None
                ),
                sensor_gain_code=(
                    int(item["sensor_gain_code"])
                    if item.get("sensor_gain_code") is not None else None
                ),
                isp_gain_code=(
                    int(item["isp_gain_code"])
                    if item.get("isp_gain_code") is not None else None
                ),
                calibration_gain_estimate=(
                    float(item["calibration_gain_estimate"])
                    if item.get("calibration_gain_estimate") is not None else None
                ),
                analog_gain_factor=(
                    float(item["analog_gain_factor"])
                    if item.get("analog_gain_factor") is not None else None
                ),
                isp_gain_factor=(
                    float(item["isp_gain_factor"])
                    if item.get("isp_gain_factor") is not None else None
                ),
                total_gain_factor=(
                    float(item["total_gain_factor"])
                    if item.get("total_gain_factor") is not None else None
                ),
                analog_iso_equivalent=(
                    float(item["analog_iso_equivalent"])
                    if item.get("analog_iso_equivalent") is not None else None
                ),
                iso_equivalent=(
                    float(item["iso_equivalent"])
                    if item.get("iso_equivalent") is not None else None
                ),
            )
            for name, item in manifest["profiles"].items()
        }
        training_profiles = tuple(str(name) for name in config["training_profiles"])
        return cls(
            source_luma_median_dn=float(h5_source["luma_median_dn"]),
            source_black_level_dn=float(config.get("source_black_level_dn", 0.0)),
            profiles=profiles,
            training_profiles=training_profiles,
            validation_profile=str(config["validation_profile"]),
            test_profile=str(config["test_profile"]),
            manifest_path=str(manifest_path),
            target_calibration_gain=(
                int(config["target_calibration_gain"])
                if config.get("target_calibration_gain") is not None else None
            ),
        )

    def _profile(self, name: str) -> CompanySceneProfile:
        try:
            return self.profiles[name]
        except KeyError as error:
            raise KeyError(f"unknown company scene profile {name!r}") from error

    def scale_for(self, profile_name: str) -> float:
        profile = self._profile(profile_name)
        if profile.luma_median_dn <= 0:
            raise ValueError(f"profile {profile_name} has nonpositive luma median")
        # The company profile already contains the brightness produced by its
        # actual gain and exposure.  Do not remove that gain and reapply an
        # arbitrary global calibration gain: doing so changes the measured
        # scene condition.  Noise selection uses analog_iso_for() separately.
        return profile.luma_median_dn / self.source_luma_median_dn

    def analog_iso_for(self, profile_name: str) -> float:
        """Return the analog ISO-equivalent used for sensor-noise lookup."""
        profile = self._profile(profile_name)
        value = profile.analog_iso_equivalent
        if value is None or value <= 0:
            raise ValueError(f"profile {profile_name} lacks a positive analog ISO-equivalent")
        return value

    def total_gain_for(self, profile_name: str) -> float:
        """Return the measured total company gain, including ISP gain."""
        profile = self._profile(profile_name)
        value = profile.iso_equivalent
        if value is None or value <= 0:
            raise ValueError(f"profile {profile_name} lacks a positive total gain")
        return value

    def profile_names_for_locations(
        self,
        locations: Mapping[str, torch.Tensor],
        mode: str,
    ) -> tuple[str, ...]:
        """Choose one measured condition per sample without global RNG state."""
        if mode == "validation":
            count = int(locations["frame"].reshape(-1).shape[0])
            return (self.validation_profile,) * count
        if mode == "test":
            count = int(locations["frame"].reshape(-1).shape[0])
            return (self.test_profile,) * count
        if mode != "training":
            raise ValueError("mapping mode must be training, validation, or test")
        frames = locations["frame"].reshape(-1).to(device="cpu", dtype=torch.int64)
        shards = locations["shard"].reshape(-1).to(device="cpu", dtype=torch.int64)
        tops = locations["top"].reshape(-1).to(device="cpu", dtype=torch.int64)
        lefts = locations["left"].reshape(-1).to(device="cpu", dtype=torch.int64)
        count = len(self.training_profiles)
        names = []
        for frame, shard, top, left in zip(
            frames.tolist(), shards.tolist(), tops.tolist(), lefts.tolist()
        ):
            digest = hashlib.sha256(f"{frame}:{shard}:{top}:{left}".encode()).digest()
            index = int.from_bytes(digest[:8], "little") % count
            names.append(self.training_profiles[index])
        return tuple(names)

    def apply(
        self,
        patch: torch.Tensor,
        white_level: float,
        black_level_dn: float,
        profile_names: Sequence[str],
    ) -> torch.Tensor:
        """Convert H5 normalized DN to profile-matched normalized signal."""
        if patch.ndim != 4:
            raise ValueError("patch must have shape [B,4,H,W]")
        if white_level <= 0:
            raise ValueError("white_level must be positive")
        if black_level_dn < 0 or black_level_dn >= white_level:
            raise ValueError("H5 clean black level must lie in [0, white_level)")
        if len(profile_names) != patch.shape[0]:
            raise ValueError("one company profile is required per patch")
        if not torch.isfinite(patch).all():
            raise ValueError("H5 patch contains non-finite values")
        scales = torch.tensor(
            [self.scale_for(name) for name in profile_names],
            device=patch.device,
            dtype=patch.dtype,
        ).reshape(-1, 1, 1, 1)
        signal_dn = torch.clamp(patch * white_level - black_level_dn, min=0.0)
        # The source contains clipped white codes; exposure transfer cannot
        # create values outside the target sensor's representable range.
        return torch.clamp(signal_dn * scales, min=0.0, max=white_level) / white_level

    def metadata(self) -> dict[str, Any]:
        return {
            "manifest_path": str(Path(self.manifest_path).resolve()),
            "source_luma_median_dn": self.source_luma_median_dn,
            "source_black_level_dn": self.source_black_level_dn,
            "target_calibration_gain": self.target_calibration_gain,
            "training_profiles": list(self.training_profiles),
            "validation_profile": self.validation_profile,
            "test_profile": self.test_profile,
            "absolute_lux_calibrated": False,
            "gain_mapping_to_calibration_established": True,
            "mapping_rule": (
                "linear signal scale = company luma median / H5 source luma median; "
                "analog ISO selects noise and total gain remains in final DN"
            ),
            "profile_scales": {
                name: self.scale_for(name) for name in self.profiles
            },
        }
