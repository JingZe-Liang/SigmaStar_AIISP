"""Model-side package for RAW 2DNR/3DNR fusion research.

Imports stay lazy so model-only development does not eagerly import dataset
dependencies such as h5py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "CurrentFrameRefineHead",
    "DEFAULT_RAW_RANGE",
    "GatedFFN",
    "H5FusionDataset",
    "H5FusionDatasetConfig",
    "LocalContextMixer",
    "MambaFusionWeightNetLite",
    "MambaFusionWeightNetLiteConfig",
    "MambaFusionWeightNetLiteOutput",
    "MambaSelectiveScan1D",
    "MotionPriorStem",
    "PackedRawStemEncoder",
    "RawPairFeatures",
    "RawRange",
    "SharedRawStem",
    "SpatioTemporalBidirectionalSSM3D",
    "StemConfig",
    "StemFeatures",
    "STMambaLiteBlock",
    "STMambaLiteConfig",
    "STMambaLiteStack",
    "STMambaLiteStackOutput",
    "WeightHead",
    "bayer_pack",
    "bayer_unpack",
    "build_motion_prior",
    "get_cfa_positions",
    "normalize_raw",
    "prepare_noisy_pair_features",
    "selective_scan_reference",
]


if TYPE_CHECKING:
    from .dataset_h5 import H5FusionDataset, H5FusionDatasetConfig
    from .models import (
        CurrentFrameRefineHead,
        GatedFFN,
        LocalContextMixer,
        MambaFusionWeightNetLite,
        MambaFusionWeightNetLiteConfig,
        MambaFusionWeightNetLiteOutput,
        MambaSelectiveScan1D,
        MotionPriorStem,
        PackedRawStemEncoder,
        SharedRawStem,
        SpatioTemporalBidirectionalSSM3D,
        StemConfig,
        StemFeatures,
        STMambaLiteBlock,
        STMambaLiteConfig,
        STMambaLiteStack,
        STMambaLiteStackOutput,
        WeightHead,
        selective_scan_reference,
    )
    from .raw_utils import (
        DEFAULT_RAW_RANGE,
        RawPairFeatures,
        RawRange,
        bayer_pack,
        bayer_unpack,
        build_motion_prior,
        get_cfa_positions,
        normalize_raw,
        prepare_noisy_pair_features,
    )


def __getattr__(name: str):
    if name in {"H5FusionDataset", "H5FusionDatasetConfig"}:
        from .dataset_h5 import H5FusionDataset, H5FusionDatasetConfig

        return {
            "H5FusionDataset": H5FusionDataset,
            "H5FusionDatasetConfig": H5FusionDatasetConfig,
        }[name]

    model_names = {
        "CurrentFrameRefineHead",
        "GatedFFN",
        "LocalContextMixer",
        "MambaFusionWeightNetLite",
        "MambaFusionWeightNetLiteConfig",
        "MambaFusionWeightNetLiteOutput",
        "MambaSelectiveScan1D",
        "MotionPriorStem",
        "PackedRawStemEncoder",
        "SharedRawStem",
        "SpatioTemporalBidirectionalSSM3D",
        "StemConfig",
        "StemFeatures",
        "STMambaLiteBlock",
        "STMambaLiteConfig",
        "STMambaLiteStack",
        "STMambaLiteStackOutput",
        "WeightHead",
        "selective_scan_reference",
    }
    if name in model_names:
        from .models import (
            CurrentFrameRefineHead,
            GatedFFN,
            LocalContextMixer,
            MambaFusionWeightNetLite,
            MambaFusionWeightNetLiteConfig,
            MambaFusionWeightNetLiteOutput,
            MambaSelectiveScan1D,
            MotionPriorStem,
            PackedRawStemEncoder,
            SharedRawStem,
            SpatioTemporalBidirectionalSSM3D,
            StemConfig,
            StemFeatures,
            STMambaLiteBlock,
            STMambaLiteConfig,
            STMambaLiteStack,
            STMambaLiteStackOutput,
            WeightHead,
            selective_scan_reference,
        )

        return {
            "CurrentFrameRefineHead": CurrentFrameRefineHead,
            "GatedFFN": GatedFFN,
            "LocalContextMixer": LocalContextMixer,
            "MambaFusionWeightNetLite": MambaFusionWeightNetLite,
            "MambaFusionWeightNetLiteConfig": MambaFusionWeightNetLiteConfig,
            "MambaFusionWeightNetLiteOutput": MambaFusionWeightNetLiteOutput,
            "MambaSelectiveScan1D": MambaSelectiveScan1D,
            "MotionPriorStem": MotionPriorStem,
            "PackedRawStemEncoder": PackedRawStemEncoder,
            "SharedRawStem": SharedRawStem,
            "SpatioTemporalBidirectionalSSM3D": SpatioTemporalBidirectionalSSM3D,
            "StemConfig": StemConfig,
            "StemFeatures": StemFeatures,
            "STMambaLiteBlock": STMambaLiteBlock,
            "STMambaLiteConfig": STMambaLiteConfig,
            "STMambaLiteStack": STMambaLiteStack,
            "STMambaLiteStackOutput": STMambaLiteStackOutput,
            "WeightHead": WeightHead,
            "selective_scan_reference": selective_scan_reference,
        }[name]

    if name in {
        "DEFAULT_RAW_RANGE",
        "RawPairFeatures",
        "RawRange",
        "bayer_pack",
        "bayer_unpack",
        "build_motion_prior",
        "get_cfa_positions",
        "normalize_raw",
        "prepare_noisy_pair_features",
    }:
        from .raw_utils import (
            DEFAULT_RAW_RANGE,
            RawPairFeatures,
            RawRange,
            bayer_pack,
            bayer_unpack,
            build_motion_prior,
            get_cfa_positions,
            normalize_raw,
            prepare_noisy_pair_features,
        )

        return {
            "DEFAULT_RAW_RANGE": DEFAULT_RAW_RANGE,
            "RawPairFeatures": RawPairFeatures,
            "RawRange": RawRange,
            "bayer_pack": bayer_pack,
            "bayer_unpack": bayer_unpack,
            "build_motion_prior": build_motion_prior,
            "get_cfa_positions": get_cfa_positions,
            "normalize_raw": normalize_raw,
            "prepare_noisy_pair_features": prepare_noisy_pair_features,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
