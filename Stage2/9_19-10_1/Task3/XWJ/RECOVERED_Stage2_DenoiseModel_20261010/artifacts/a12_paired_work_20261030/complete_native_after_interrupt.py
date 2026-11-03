"""Complete the five remaining native evaluations without changing their protocol."""

import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from raw_fusion.finetune.a12_paired_eval import evaluate_paired_checkpoint


def main() -> None:
    work = Path(__file__).resolve().parent
    status_path = work / "evaluation_completion_status.json"
    status = {"status": "running", "oom_policy": "stop_without_retry", "completed": []}
    with status_path.open("x", encoding="utf-8") as stream:
        json.dump(status, stream, indent=2)
    labels = ["refresh1024_ema", "shared512_ema", "shared512_model", "fixed1024_model", "refresh1024_model"]
    try:
        for label in labels:
            output = ROOT / f"artifacts/a12_paired_eval_{label}_20261030"
            identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
            status["active"] = {"label": label, "phase": "native"}
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            print(f"BEGIN native {label}", flush=True)
            result = evaluate_paired_checkpoint(
                checkpoint=Path(identity["checkpoint"]), weights=identity["weights"],
                label=label, output_dir=output, device_name="cuda:0", phase="native",
                expected_checkpoint_sha256=identity["checkpoint_sha256"],
            )
            status["completed"].append({"label": label, "result": result})
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            print(f"END native {label}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        status["status"] = "completed"
        status["active"] = None
    except BaseException as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        status.update(status="stopped_oom" if is_oom else "stopped_error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
