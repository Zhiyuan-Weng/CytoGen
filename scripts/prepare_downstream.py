#!/usr/bin/env python
"""Combine real and CytoGen pairs for a downstream segmenter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cytogen.downstream import (
    build_training_records,
    export_training_dataset,
    load_pair_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_dataset", type=Path)
    parser.add_argument("--synthetic_dataset", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("manifest", "cellpose", "omnipose", "cellotype"),
        required=True,
    )
    parser.add_argument(
        "--segmentation_task",
        choices=("whole_cell", "nuclear"),
        default="whole_cell",
    )
    parser.add_argument("--real_split", default="train")
    parser.add_argument("--real_metadata", type=Path)
    parser.add_argument("--synthetic_metadata", type=Path)
    parser.add_argument("--real_fraction", type=float, default=1.0)
    synthetic_group = parser.add_mutually_exclusive_group()
    synthetic_group.add_argument("--synthetic_ratio", type=float)
    synthetic_group.add_argument("--synthetic_count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.real_dataset is None and args.synthetic_dataset is None:
        raise ValueError("At least one real or synthetic dataset is required")
    real_records = []
    if args.real_dataset is not None:
        real_records = load_pair_records(
            args.real_dataset,
            source="real",
            split=args.real_split,
            metadata_path=args.real_metadata,
        )
    synthetic_records = []
    if args.synthetic_dataset is not None:
        synthetic_records = load_pair_records(
            args.synthetic_dataset,
            source="synthetic",
            metadata_path=args.synthetic_metadata,
        )
    synthetic_ratio = args.synthetic_ratio
    synthetic_count = args.synthetic_count
    if synthetic_ratio is None and synthetic_count is None:
        if real_records and synthetic_records:
            synthetic_ratio = 1.0
        elif synthetic_records:
            synthetic_count = len(synthetic_records)
        else:
            synthetic_ratio = 0.0
    records = build_training_records(
        real_records,
        synthetic_records,
        real_fraction=args.real_fraction,
        synthetic_ratio=synthetic_ratio,
        synthetic_count=synthetic_count,
        seed=args.seed,
    )
    selection = {
        "real_dataset": (
            str(args.real_dataset.expanduser())
            if args.real_dataset is not None
            else None
        ),
        "synthetic_dataset": (
            str(args.synthetic_dataset.expanduser())
            if args.synthetic_dataset is not None
            else None
        ),
        "real_fraction": args.real_fraction,
        "synthetic_ratio": synthetic_ratio,
        "synthetic_count": synthetic_count,
        "seed": args.seed,
    }
    export_training_dataset(
        records,
        args.output_dir,
        args.format,
        segmentation_task=args.segmentation_task,
        overwrite=args.overwrite,
        selection_config=selection,
    )
    print(f"Prepared {len(records)} samples in {args.output_dir.expanduser()}")


if __name__ == "__main__":
    main()
