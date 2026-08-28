from __future__ import annotations

"""CLI entry point for AI fusion training.

Run from the repository root:

    python -m MyNet.ai_fusion.train --data-root H5 --output-dir results/ai_fusion_stage0
"""

from MyNet.ai_fusion.training.cli import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
