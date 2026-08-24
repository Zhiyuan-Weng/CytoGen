#!/usr/bin/env python
"""Generate synthetic cellular layouts from a prepared real dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cytogen.layout import LayoutGenerator, LayoutGeneratorConfig, extract_layout_priors
from cytogen.layout.cpm import CPMConfig
from cytogen.layout.priors import load_failure_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    count_group = parser.add_mutually_exclusive_group(required=True)
    count_group.add_argument("--num_layouts", type=int)
    count_group.add_argument("--num_layouts_per_category", type=int)
    parser.add_argument(
        "--categories",
        help="Comma-separated categories; by default all observed categories are used.",
    )
    parser.add_argument("--failure_scores", type=Path)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--minimum_instance_area", type=int, default=15)
    parser.add_argument("--density_radius", type=float, default=64.0)
    parser.add_argument("--contact_tolerance", type=int, default=1)
    parser.add_argument("--density_octaves", type=int, default=4)
    parser.add_argument("--density_persistence", type=float, default=0.5)
    parser.add_argument("--density_base_scale", type=float, default=64.0)
    parser.add_argument("--density_heterogeneity", type=float, default=1.0)
    parser.add_argument("--spacing_scale", type=float, default=0.9)
    parser.add_argument("--minimum_spacing_scale", type=float, default=0.45)
    parser.add_argument("--contact_spacing_factor", type=float, default=0.7)
    parser.add_argument("--isolated_spacing_factor", type=float, default=1.1)
    parser.add_argument("--density_bins", type=int, default=8)
    parser.add_argument("--contact_bins", type=int, default=6)
    parser.add_argument("--elongation_bins", type=int, default=8)
    parser.add_argument("--support_exponent", type=float, default=0.5)
    parser.add_argument("--failure_exponent", type=float, default=1.0)
    parser.add_argument("--sampling_epsilon", type=float, default=1e-3)
    parser.add_argument("--minimum_survival_fraction", type=float, default=0.8)
    parser.add_argument("--maximum_layout_attempts", type=int, default=3)
    parser.add_argument("--mcs_steps", type=int, default=200)
    parser.add_argument("--mcs_attempt_fraction", type=float, default=1.0)
    parser.add_argument("--lambda_area", type=float, default=1.0)
    parser.add_argument("--lambda_perimeter", type=float, default=0.05)
    parser.add_argument("--cell_medium_energy", type=float, default=12.0)
    parser.add_argument("--cell_cell_energy", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--initial_area_fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def allocate_layouts(
    priors,
    total: int | None,
    per_category: int | None,
) -> dict[str, int]:
    if per_category is not None:
        if per_category < 1:
            raise ValueError("--num_layouts_per_category must be positive")
        return {category: per_category for category in priors}
    if total is None or total < 1:
        raise ValueError("--num_layouts must be positive")
    categories = sorted(priors)
    weights = np.asarray(
        [len(priors[category].field_counts) for category in categories], dtype=float
    )
    weights /= weights.sum()
    expected = weights * total
    allocated = np.floor(expected).astype(int)
    remainder_order = np.argsort(-(expected - allocated), kind="stable")
    for index in remainder_order[: total - int(allocated.sum())]:
        allocated[index] += 1
    return {
        category: int(count)
        for category, count in zip(categories, allocated)
        if count > 0
    }


def main() -> None:
    args = parse_args()
    priors = extract_layout_priors(
        args.dataset_path,
        split=args.split,
        minimum_instance_area=args.minimum_instance_area,
        density_radius=args.density_radius,
        contact_tolerance=args.contact_tolerance,
    )
    if args.categories:
        requested = [item.strip() for item in args.categories.split(",") if item.strip()]
        unknown = sorted(set(requested) - set(priors))
        if unknown:
            raise ValueError(f"Unknown categories: {unknown}")
        priors = {category: priors[category] for category in requested}
    category_counts = allocate_layouts(
        priors, args.num_layouts, args.num_layouts_per_category
    )
    layout_config = LayoutGeneratorConfig(
        height=args.height,
        width=args.width,
        density_octaves=args.density_octaves,
        density_persistence=args.density_persistence,
        density_base_scale=args.density_base_scale,
        density_heterogeneity=args.density_heterogeneity,
        spacing_scale=args.spacing_scale,
        minimum_spacing_scale=args.minimum_spacing_scale,
        contact_spacing_factor=args.contact_spacing_factor,
        isolated_spacing_factor=args.isolated_spacing_factor,
        density_bins=args.density_bins,
        contact_bins=args.contact_bins,
        elongation_bins=args.elongation_bins,
        support_exponent=args.support_exponent,
        failure_exponent=args.failure_exponent,
        sampling_epsilon=args.sampling_epsilon,
        minimum_survival_fraction=args.minimum_survival_fraction,
        maximum_layout_attempts=args.maximum_layout_attempts,
        seed=args.seed,
    )
    cpm_config = CPMConfig(
        mcs_steps=args.mcs_steps,
        attempt_fraction=args.mcs_attempt_fraction,
        lambda_area=args.lambda_area,
        lambda_perimeter=args.lambda_perimeter,
        cell_medium_energy=args.cell_medium_energy,
        cell_cell_energy=args.cell_cell_energy,
        temperature=args.temperature,
        initial_area_fraction=args.initial_area_fraction,
    )
    generator = LayoutGenerator(
        priors,
        layout_config,
        cpm_config,
        failure_scores=load_failure_scores(args.failure_scores),
    )
    generator.generate_dataset(args.output_dir, category_counts)
    generated = sum(category_counts.values())
    print(f"Generated {generated} layouts in {args.output_dir.expanduser()}")


if __name__ == "__main__":
    main()
