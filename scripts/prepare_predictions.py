#!/usr/bin/env python
"""Convert downstream segmenter outputs to a CytoGen prediction manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cytogen.predictions import adapt_coco_predictions, adapt_mask_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        required=True,
        choices=["cellpose", "omnipose", "cellotype"],
    )
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Mask directory for Cellpose/Omnipose or COCO result JSON for CelloType",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--ground_truth_column", default="mask_name")
    parser.add_argument(
        "--filename_suffix",
        default=None,
        help="Additional suffix removed when matching mask files to sample_id",
    )
    parser.add_argument(
        "--coco_ground_truth",
        type=Path,
        default=None,
        help="COCO ground-truth JSON used to map CelloType image IDs",
    )
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument(
        "--max_predictions",
        type=int,
        default=0,
        help="Maximum CelloType predictions per image; 0 keeps all",
    )
    parser.add_argument(
        "--mapping_mode",
        choices=["auto", "filename", "index", "order"],
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.format in {"cellpose", "omnipose"}:
        summary = adapt_mask_predictions(
            dataset_path=args.dataset_path,
            prediction_dir=args.predictions,
            output_dir=args.output_dir,
            model_family=args.format,
            split=args.split,
            ground_truth_column=args.ground_truth_column,
            filename_suffix=args.filename_suffix,
            overwrite=args.overwrite,
        )
    else:
        if args.coco_ground_truth is None:
            raise ValueError("--coco_ground_truth is required for CelloType")
        summary = adapt_coco_predictions(
            dataset_path=args.dataset_path,
            prediction_json=args.predictions,
            coco_ground_truth=args.coco_ground_truth,
            output_dir=args.output_dir,
            split=args.split,
            ground_truth_column=args.ground_truth_column,
            score_threshold=args.score_threshold,
            max_predictions=args.max_predictions,
            mapping_mode=args.mapping_mode,
            overwrite=args.overwrite,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
