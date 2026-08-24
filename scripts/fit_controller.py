#!/usr/bin/env python
"""Fit one CytoGen failure-aware controller round from predicted masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cytogen.controller import FailureModelConfig, fit_controller_round


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--prediction_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--ground_truth_column", default="mask_name")
    parser.add_argument("--prediction_column", default="prediction_mask")
    parser.add_argument("--minimum_instance_area", type=int, default=15)
    parser.add_argument("--density_radius", type=float, default=64.0)
    parser.add_argument("--contact_tolerance", type=int, default=1)
    parser.add_argument("--density_bins", type=int, default=8)
    parser.add_argument("--contact_bins", type=int, default=6)
    parser.add_argument("--elongation_bins", type=int, default=8)
    parser.add_argument("--support_exponent", type=float, default=0.5)
    parser.add_argument("--failure_exponent", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--huber_delta", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = FailureModelConfig(
        hidden_units=args.hidden_units,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        huber_delta=args.huber_delta,
        seed=args.seed,
        device=args.device,
    )
    config = fit_controller_round(
        dataset_path=args.dataset_path,
        prediction_manifest=args.prediction_manifest,
        output_dir=args.output_dir,
        model_config=model_config,
        split=args.split,
        ground_truth_column=args.ground_truth_column,
        prediction_column=args.prediction_column,
        minimum_instance_area=args.minimum_instance_area,
        density_radius=args.density_radius,
        contact_tolerance=args.contact_tolerance,
        descriptor_bins=(
            args.density_bins,
            args.contact_bins,
            args.elongation_bins,
        ),
        support_exponent=args.support_exponent,
        failure_exponent=args.failure_exponent,
        epsilon=args.epsilon,
        round_index=args.round_index,
        overwrite=args.overwrite,
    )
    print(
        f"Fitted round {config['round_index']} from "
        f"{config['number_of_observed_instances']} instances in "
        f"{args.output_dir.expanduser()}"
    )


if __name__ == "__main__":
    main()
