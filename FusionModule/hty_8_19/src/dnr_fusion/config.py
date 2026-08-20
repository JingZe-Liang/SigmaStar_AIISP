from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def validate_scene(config: dict[str, Any], scene_name: str) -> dict[str, Any]:
    try:
        scene = config["scenes"][scene_name]
    except KeyError as exc:
        available = ", ".join(sorted(config.get("scenes", {})))
        raise KeyError(f"Unknown scene {scene_name!r}; available: {available}") from exc

    missing = []
    for key in ("source", "denoised", "fused"):
        path = Path(scene[key])
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing scene inputs:\n" + "\n".join(missing))
    return scene


def project_root(config: dict[str, Any]) -> Path:
    return Path(config["_config_path"]).parent.parent

