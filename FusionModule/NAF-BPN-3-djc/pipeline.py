from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train import ROOT, config_path, read_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume the complete NAFBPNNet cloud training pipeline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cloud.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_digest(config_path_value: Path) -> str:
    return hashlib.sha256(config_path_value.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_completed(marker: Path, digest: str) -> bool:
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("config_sha256") != digest:
        raise RuntimeError(f"配置已变更，不能复用旧阶段: {marker.parent}")
    return True


def checkpoint_or_resume(output: Path) -> list[str]:
    last = output / "last.pth"
    return ["--resume", str(last)] if last.is_file() else []


def run_command(command: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def run_stage(
    state_path: Path,
    config_digest_value: str,
    name: str,
    output: Path,
    expected_checkpoint: Path,
    command: list[str],
    dry_run: bool,
) -> None:
    marker = output / "completed.json"
    if read_completed(marker, config_digest_value):
        if not expected_checkpoint.is_file():
            raise FileNotFoundError(f"{name} 标记为完成，但缺少 checkpoint: {expected_checkpoint}")
        print(f"skip completed stage: {name}", flush=True)
        return
    if not dry_run:
        write_json(state_path, {"status": "running", "current_stage": name, "updated_at": utc_now()})
    run_command(command, dry_run)
    if dry_run:
        return
    if not expected_checkpoint.is_file():
        raise FileNotFoundError(f"{name} 未产生预期 checkpoint: {expected_checkpoint}")
    write_json(marker, {"stage": name, "checkpoint": str(expected_checkpoint.relative_to(ROOT)), "config_sha256": config_digest_value, "completed_at": utc_now()})


def main() -> int:
    args = parse_args()
    config_file = args.config if args.config.is_absolute() else ROOT / args.config
    config = read_config(config_file)
    digest = config_digest(config_file)
    runs_root = resolve_path("runs")
    state_path = runs_root / "pipeline" / "pipeline_state.json"
    python = sys.executable
    stage1_output = resolve_path(config.get("stage1_output", "runs/stage1"))
    stage1_best = stage1_output / "best.pth"
    eval_128_output = resolve_path("runs/stage2_128_to_645")
    eval_645_output = resolve_path("runs/stage2_645_to_128")
    final_output = resolve_path("runs/stage2_final_all")
    try:
        if not args.dry_run:
            write_json(state_path, {"status": "running", "current_stage": "preflight", "updated_at": utc_now()})
        run_command([python, "-u", "preflight.py", "--config", str(config_file)], args.dry_run)
        run_stage(
            state_path, digest, "stage1", stage1_output, stage1_best,
            [python, "-u", "train.py", "--stage", "1", "--config", str(config_file), "--output", str(stage1_output), *checkpoint_or_resume(stage1_output)],
            args.dry_run,
        )
        md_marker = runs_root / "pipeline" / "motion_cache_completed.json"
        if read_completed(md_marker, digest):
            print("skip completed stage: motion_cache", flush=True)
        else:
            if not args.dry_run:
                write_json(state_path, {"status": "running", "current_stage": "motion_cache", "updated_at": utc_now()})
            run_command([python, "-u", "cache_motion.py", "--config", str(config_file), "--sequence", "all"], args.dry_run)
            if not args.dry_run:
                motion_root = config_path(config, "motion_cache_root")
                missing = [motion_root / name / "masks" / "0199.png" for name in config["sequence_names"] if not (motion_root / name / "masks" / "0199.png").is_file()]
                if missing:
                    raise FileNotFoundError(f"MD cache 未完整生成: {missing[0]}")
                write_json(md_marker, {"stage": "motion_cache", "config_sha256": digest, "completed_at": utc_now()})
        run_stage(
            state_path, digest, "stage2_128_to_645", eval_128_output, eval_128_output / "best.pth",
            [python, "-u", "train.py", "--stage", "2", "--fold", "128_to_645", "--config", str(config_file), "--output", str(eval_128_output), "--init-checkpoint", str(stage1_best), *checkpoint_or_resume(eval_128_output)],
            args.dry_run,
        )
        run_stage(
            state_path, digest, "stage2_645_to_128", eval_645_output, eval_645_output / "best.pth",
            [python, "-u", "train.py", "--stage", "2", "--fold", "645_to_128", "--config", str(config_file), "--output", str(eval_645_output), "--init-checkpoint", str(stage1_best), *checkpoint_or_resume(eval_645_output)],
            args.dry_run,
        )
        run_stage(
            state_path, digest, "stage2_final_all", final_output, final_output / "final.pth",
            [python, "-u", "train.py", "--stage", "2", "--fold", "all", "--config", str(config_file), "--output", str(final_output), "--init-checkpoint", str(stage1_best), *checkpoint_or_resume(final_output)],
            args.dry_run,
        )
    except Exception as error:
        if not args.dry_run:
            write_json(state_path, {"status": "failed", "error": str(error), "updated_at": utc_now()})
        traceback.print_exc()
        return 1
    if not args.dry_run:
        write_json(state_path, {"status": "completed", "updated_at": utc_now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
