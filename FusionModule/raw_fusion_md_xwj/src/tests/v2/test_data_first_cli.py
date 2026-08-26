from __future__ import annotations

import importlib.util


def test_formal_only_runtime_modules_are_not_publicly_importable() -> None:
    assert importlib.util.find_spec("raw_fusion.v2.composer") is None
    assert importlib.util.find_spec("raw_fusion.v2.tiling") is None


def test_data_first_train_dispatch_rejects_missing_dataset() -> None:
    from raw_fusion.v2.cli import train

    assert train(["--data-first", "--output-dir", "/tmp/data-first-test"]) == 2


def test_data_first_train_accepts_monitoring_and_resume_options(capsys) -> None:
    from raw_fusion.v2.cli import train

    assert train([
        "--data-first",
        "--dataset", "/missing/dataset.json",
        "--split", "/missing/split.json",
        "--output-dir", "/tmp/data-first-test",
        "--log-interval", "5",
        "--checkpoint-interval", "25",
        "--resume", "/tmp/checkpoint_step_000025.pt",
    ]) == 2
    assert "unrecognized arguments" not in capsys.readouterr().err


def test_mog2_cache_command_accepts_parallel_worker_option(capsys) -> None:
    from raw_fusion.v2.cli import mog2_cache_generate

    assert mog2_cache_generate([
        "--dataset", "/missing/dataset.json",
        "--split", "/missing/split.json",
        "--output-dir", "/tmp/mog2-cache",
        "--workers", "16",
    ]) == 2
    assert "unrecognized arguments" not in capsys.readouterr().err


def test_formal_infer_remains_deferred_but_data_first_flags_are_parsed() -> None:
    from raw_fusion.v2.cli import infer

    assert infer(["--checkpoint", "x.pt", "--output-dir", "/tmp/out"]) == 2
