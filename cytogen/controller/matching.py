"""One-to-one instance matching and IoU-threshold failure scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


IOU_THRESHOLDS = tuple(float(value) for value in np.arange(0.50, 1.00, 0.05))


@dataclass(frozen=True)
class InstanceMatch:
    ground_truth_label: int
    prediction_label: int | None
    iou: float
    success_score: float
    failure_score: float


def _as_mask(mask: np.ndarray, name: str) -> np.ndarray:
    mask = np.squeeze(np.asarray(mask))
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D instance map, got {mask.shape}")
    if np.any(mask < 0):
        raise ValueError(f"{name} contains negative labels")
    return mask.astype(np.int64, copy=False)


def _iou_matrix(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    ground_truth_labels: np.ndarray,
    prediction_labels: np.ndarray,
) -> np.ndarray:
    if not len(ground_truth_labels) or not len(prediction_labels):
        return np.zeros((len(ground_truth_labels), len(prediction_labels)))
    all_ground_truth, ground_truth_inverse = np.unique(
        ground_truth, return_inverse=True
    )
    all_prediction, prediction_inverse = np.unique(prediction, return_inverse=True)
    intersections = np.bincount(
        (
            ground_truth_inverse * len(all_prediction) + prediction_inverse
        ).ravel(),
        minlength=len(all_ground_truth) * len(all_prediction),
    ).reshape(len(all_ground_truth), len(all_prediction))
    ground_truth_indices = np.searchsorted(all_ground_truth, ground_truth_labels)
    prediction_indices = np.searchsorted(all_prediction, prediction_labels)
    intersections = intersections[np.ix_(ground_truth_indices, prediction_indices)]
    ground_truth_areas = np.bincount(
        ground_truth_inverse.ravel(), minlength=len(all_ground_truth)
    )[ground_truth_indices]
    prediction_areas = np.bincount(
        prediction_inverse.ravel(), minlength=len(all_prediction)
    )[prediction_indices]
    unions = (
        ground_truth_areas[:, None]
        + prediction_areas[None, :]
        - intersections
    )
    return intersections / np.maximum(unions, 1)


def match_instances(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    valid_ground_truth_labels: set[int] | None = None,
    thresholds: tuple[float, ...] = IOU_THRESHOLDS,
) -> list[InstanceMatch]:
    """Match instances globally with Hungarian assignment and score each GT cell."""
    ground_truth = _as_mask(ground_truth, "ground_truth")
    prediction = _as_mask(prediction, "prediction")
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Mask shapes differ: {ground_truth.shape} versus {prediction.shape}"
        )
    ground_truth_labels = np.unique(ground_truth)
    ground_truth_labels = ground_truth_labels[ground_truth_labels != 0]
    if valid_ground_truth_labels is not None:
        ground_truth_labels = np.asarray(
            [
                int(label)
                for label in ground_truth_labels
                if int(label) in valid_ground_truth_labels
            ],
            dtype=np.int64,
        )
    prediction_labels = np.unique(prediction)
    prediction_labels = prediction_labels[prediction_labels != 0]
    ious = _iou_matrix(
        ground_truth,
        prediction,
        ground_truth_labels,
        prediction_labels,
    )
    assignments: dict[int, tuple[int, float]] = {}
    if ious.size:
        ground_truth_indices, prediction_indices = linear_sum_assignment(1.0 - ious)
        for ground_truth_index, prediction_index in zip(
            ground_truth_indices, prediction_indices
        ):
            iou = float(ious[ground_truth_index, prediction_index])
            if iou > 0:
                assignments[int(ground_truth_index)] = (
                    int(prediction_labels[prediction_index]),
                    iou,
                )
    threshold_array = np.asarray(thresholds, dtype=float)
    if threshold_array.size == 0:
        raise ValueError("At least one IoU threshold is required")
    matches = []
    for ground_truth_index, ground_truth_label in enumerate(ground_truth_labels):
        prediction_label = None
        iou = 0.0
        if ground_truth_index in assignments:
            prediction_label, iou = assignments[ground_truth_index]
        success = float(np.mean(iou >= threshold_array)) if prediction_label else 0.0
        matches.append(
            InstanceMatch(
                ground_truth_label=int(ground_truth_label),
                prediction_label=prediction_label,
                iou=iou,
                success_score=success,
                failure_score=1.0 - success,
            )
        )
    return matches
