"""CLI for losslessly re-phasing headerless Bayer RAW video streams."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cfa_conversion import convert_rggb_video_to_gbrg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a packed uint16 RGGB RAW video to GBRG.")
    parser.add_argument("source", type=Path, help="Headerless input RAW video in RGGB CFA phase.")
    parser.add_argument("destination", type=Path, help="New output RAW video in GBRG CFA phase.")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = convert_rggb_video_to_gbrg(
        args.source,
        args.destination,
        width=args.width,
        height=args.height,
    )
    print(f"Converted {frames} frames: {args.source} -> {args.destination}")


if __name__ == "__main__":
    main()
