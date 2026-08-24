#!/usr/bin/env python3
"""Prepare microscopy images, instance masks, and CytoGen HSV conditions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cytogen.data.preprocess import prepare_manifest, prepare_tissuenet


def _positive_size(value: str) -> int:
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare diffusion images and HSV condition maps for CytoGen."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tissuenet = subparsers.add_parser(
        "tissuenet", help="Prepare official TissueNet NPZ archives."
    )
    tissuenet.add_argument("--input-root", type=Path, required=True)
    tissuenet.add_argument("--output-root", type=Path, required=True)
    tissuenet.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    tissuenet.add_argument("--size", type=_positive_size)
    tissuenet.add_argument("--limit", type=int)

    manifest = subparsers.add_parser(
        "manifest", help="Prepare paired images and masks listed in JSONL."
    )
    manifest.add_argument("--manifest", type=Path, required=True)
    manifest.add_argument("--output-root", type=Path, required=True)
    manifest.add_argument("--size", type=_positive_size)
    manifest.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.command == "tissuenet":
        prepare_tissuenet(
            input_root=args.input_root,
            output_root=args.output_root,
            splits=args.splits,
            size=args.size,
            limit=args.limit,
        )
    else:
        prepare_manifest(
            manifest_path=args.manifest,
            output_root=args.output_root,
            size=args.size,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
