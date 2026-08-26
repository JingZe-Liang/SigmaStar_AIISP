"""Atomic checkpoint contract for the data-first V2 protocol."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import torch

from .data_first_contracts import DATA_FIRST_PROTOCOL
from .schemas.common import ContractError


@dataclass(frozen=True, slots=True)
class DataFirstCheckpoint:
    path: Path
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DataFirstCheckpointRef:
    path: Path
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_data_first_checkpoint(path: Path, model, optimizer, state: Mapping[str, object], provenance: Mapping[str, object]) -> DataFirstCheckpointRef:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": DATA_FIRST_PROTOCOL,
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": dict(state),
        "provenance": dict(provenance),
        "md_used_for_supervision": True,
        "md_used_as_model_input": False,
        "formal_v2_compatible": False,
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return DataFirstCheckpointRef(destination, _sha256(destination))


def load_data_first_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> DataFirstCheckpoint:
    destination = Path(path).resolve()
    try:
        payload = torch.load(destination, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ContractError("cannot load data-first checkpoint") from error
    if not isinstance(payload, Mapping) or payload.get("protocol") != DATA_FIRST_PROTOCOL:
        raise ContractError("checkpoint is not a data-first V2 payload")
    if payload.get("md_used_as_model_input") is not False or payload.get("formal_v2_compatible") is not False:
        raise ContractError("data-first checkpoint provenance flags are invalid")
    return DataFirstCheckpoint(destination, payload)


def write_data_first_manifest(output_dir: Path, checkpoint_ref: DataFirstCheckpointRef, training_summary: Mapping[str, object]) -> Path:
    import json

    destination = Path(output_dir) / "manifest.json"
    manifest = {"protocol": DATA_FIRST_PROTOCOL, "checkpoint": {"path": checkpoint_ref.path.name, "sha256": checkpoint_ref.sha256}, **dict(training_summary)}
    destination.write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=True, indent=2) + "\n", encoding="ascii")
    return destination


__all__ = ["DataFirstCheckpoint", "DataFirstCheckpointRef", "load_data_first_checkpoint", "save_data_first_checkpoint", "write_data_first_manifest"]
