"""Failure-aware sampling over biologically observed layout states."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from .priors import CategoryPrior


DEFAULT_DESCRIPTOR_BINS = (8, 6, 8)


def descriptor_matrix(prior: CategoryPrior) -> np.ndarray:
    return np.asarray(
        [
            (item.local_density, item.contact_number, item.elongation)
            for item in prior.instances
        ],
        dtype=float,
    )


def _bin_counts(number_of_bins: int | Sequence[int]) -> np.ndarray:
    if isinstance(number_of_bins, int):
        counts = np.repeat(number_of_bins, 3)
    else:
        counts = np.asarray(tuple(number_of_bins), dtype=int)
    if counts.shape != (3,) or np.any(counts < 1):
        raise ValueError("number_of_bins must contain three positive values")
    return counts


def configuration_policy(
    descriptors: np.ndarray,
    predicted_failures: np.ndarray,
    number_of_bins: int | Sequence[int] = DEFAULT_DESCRIPTOR_BINS,
    support_exponent: float = 0.5,
    failure_exponent: float = 1.0,
    epsilon: float = 1e-3,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, list[float]]]:
    """Compute bin and instance probabilities within the observed feasible region."""
    descriptors = np.asarray(descriptors, dtype=float)
    predicted_failures = np.asarray(predicted_failures, dtype=float)
    if descriptors.ndim != 2 or descriptors.shape[1] != 3:
        raise ValueError(f"Expected Nx3 descriptors, got {descriptors.shape}")
    if predicted_failures.shape != (len(descriptors),):
        raise ValueError("predicted_failures must contain one value per descriptor")
    if not len(descriptors):
        raise ValueError("At least one descriptor is required")
    if support_exponent < 0 or failure_exponent < 0 or epsilon <= 0:
        raise ValueError("sampling exponents must be non-negative and epsilon positive")
    counts = _bin_counts(number_of_bins)
    lower, upper = np.quantile(descriptors, (0.01, 0.99), axis=0)
    upper = np.maximum(upper, lower + 1e-8)
    scaled = (descriptors - lower) / (upper - lower)
    coordinates = np.floor(scaled * counts).astype(int)
    coordinates = np.clip(coordinates, 0, counts - 1)
    members: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, coordinate in enumerate(coordinates):
        members[tuple(int(value) for value in coordinate)].append(index)

    number_of_feasible_bins = len(members)
    support_denominator = len(descriptors) + epsilon * number_of_feasible_bins
    bin_rows = []
    weights = []
    for key in sorted(members):
        indices = members[key]
        support = (len(indices) + epsilon) / support_denominator
        failure = float(np.mean(predicted_failures[indices]))
        weight = (support + epsilon) ** (-support_exponent) * (
            failure + epsilon
        ) ** failure_exponent
        weights.append(weight)
        bin_rows.append(
            {
                "density_bin": key[0],
                "contact_bin": key[1],
                "elongation_bin": key[2],
                "instance_count": len(indices),
                "p_real": float(support),
                "predicted_failure": failure,
                "unnormalized_weight": float(weight),
                "feasible": True,
            }
        )
    weights = np.asarray(weights, dtype=float)
    bin_probabilities = weights / weights.sum()
    instance_probabilities = np.zeros(len(descriptors), dtype=float)
    for row, probability in zip(bin_rows, bin_probabilities):
        key = (
            int(row["density_bin"]),
            int(row["contact_bin"]),
            int(row["elongation_bin"]),
        )
        indices = members[key]
        row["sampling_probability"] = float(probability)
        instance_probabilities[indices] = probability / len(indices)
    edges = {
        "lower": lower.astype(float).tolist(),
        "upper": upper.astype(float).tolist(),
        "number_of_bins": counts.astype(int).tolist(),
    }
    return instance_probabilities, bin_rows, edges


def instance_sampling_probabilities(
    prior: CategoryPrior,
    failure_scores: dict[tuple[str, int], float] | None = None,
    number_of_bins: int | Sequence[int] = DEFAULT_DESCRIPTOR_BINS,
    support_exponent: float = 0.5,
    failure_exponent: float = 1.0,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Compute the paper's support- and failure-aware instance policy."""
    failure_scores = failure_scores or {}
    observed_failures = [
        failure_scores[(item.sample_id, item.instance_label)]
        for item in prior.instances
        if (item.sample_id, item.instance_label) in failure_scores
    ]
    default_failure = float(np.mean(observed_failures)) if observed_failures else 1.0
    predicted_failures = np.asarray(
        [
            failure_scores.get(
                (item.sample_id, item.instance_label), default_failure
            )
            for item in prior.instances
        ],
        dtype=float,
    )
    probabilities, _, _ = configuration_policy(
        descriptor_matrix(prior),
        predicted_failures,
        number_of_bins=number_of_bins,
        support_exponent=support_exponent,
        failure_exponent=failure_exponent,
        epsilon=epsilon,
    )
    return probabilities
