"""End-to-end fitting of one CytoGen failure-controller round."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from cytogen.layout.controller import configuration_policy, descriptor_matrix
from cytogen.layout.priors import CategoryPrior, extract_layout_priors

from .matching import IOU_THRESHOLDS, match_instances
from .model import FailureModelConfig, fit_failure_model


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "sample_id" not in row:
                raise KeyError(f"{path}:{line_number} has no sample_id")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def _resolve_path(root: Path, metadata_path: Path, split: str, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        resolved = path
    else:
        candidates = (root / split / path, root / path, metadata_path.parent / path)
        resolved = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0],
        )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved.resolve()


def _read_mask(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        mask = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        key = next(
            (name for name in ("prediction", "masks", "mask", "arr_0") if name in archive),
            None,
        )
        if key is None:
            raise KeyError(f"Cannot identify a mask array in {path}")
        mask = archive[key]
    elif suffix in {".tif", ".tiff"}:
        mask = tifffile.imread(path)
    else:
        mask = np.asarray(Image.open(path))
    mask = np.squeeze(np.asarray(mask))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D instance mask at {path}, got {mask.shape}")
    return mask


def _prediction_paths(
    prediction_manifest: Path,
    prediction_column: str,
) -> dict[str, Path]:
    paths = {}
    for row in _read_jsonl(prediction_manifest):
        sample_id = str(row["sample_id"])
        if sample_id in paths:
            raise ValueError(f"Duplicate prediction for sample_id={sample_id!r}")
        if prediction_column not in row:
            raise KeyError(
                f"Prediction {sample_id!r} has no {prediction_column!r} field"
            )
        path = Path(str(row[prediction_column])).expanduser()
        if not path.is_absolute():
            path = prediction_manifest.parent / path
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[sample_id] = path.resolve()
    return paths


def _ground_truth_paths(
    dataset_path: Path,
    split: str,
    ground_truth_column: str,
) -> dict[str, Path]:
    metadata_path = dataset_path / split / "metadata.jsonl"
    if not metadata_path.is_file():
        metadata_path = dataset_path / "metadata.jsonl"
    paths = {}
    for row in _read_jsonl(metadata_path):
        sample_id = str(row["sample_id"])
        if ground_truth_column not in row:
            raise KeyError(
                f"Ground-truth record {sample_id!r} has no {ground_truth_column!r}"
            )
        paths[sample_id] = _resolve_path(
            dataset_path,
            metadata_path,
            split,
            row[ground_truth_column],
        )
    return paths


def _collect_observations(
    priors: dict[str, CategoryPrior],
    ground_truth_paths: dict[str, Path],
    prediction_paths: dict[str, Path],
) -> list[dict[str, object]]:
    instances_by_sample = {}
    category_by_sample = {}
    for category, prior in priors.items():
        for instance in prior.instances:
            instances_by_sample.setdefault(instance.sample_id, {})[
                instance.instance_label
            ] = instance
            category_by_sample[instance.sample_id] = category
    missing_predictions = sorted(set(instances_by_sample) - set(prediction_paths))
    if missing_predictions:
        preview = ", ".join(missing_predictions[:5])
        raise ValueError(
            f"Missing predictions for {len(missing_predictions)} samples: {preview}"
        )
    observations = []
    for sample_id in sorted(instances_by_sample):
        if sample_id not in ground_truth_paths:
            raise KeyError(f"No ground-truth mask for sample_id={sample_id!r}")
        ground_truth = _read_mask(ground_truth_paths[sample_id])
        prediction = _read_mask(prediction_paths[sample_id])
        instance_lookup = instances_by_sample[sample_id]
        matches = match_instances(
            ground_truth,
            prediction,
            valid_ground_truth_labels=set(instance_lookup),
        )
        for match in matches:
            instance = instance_lookup[match.ground_truth_label]
            observations.append(
                {
                    "sample_id": sample_id,
                    "category": category_by_sample[sample_id],
                    "instance_label": match.ground_truth_label,
                    "prediction_label": match.prediction_label,
                    "matched_iou": match.iou,
                    "success_score": match.success_score,
                    "failure_score": match.failure_score,
                    "local_density": instance.local_density,
                    "contact_number": instance.contact_number,
                    "elongation": instance.elongation,
                }
            )
    if not observations:
        raise ValueError("No valid matched ground-truth instances were found")
    return observations


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def fit_controller_round(
    dataset_path: str | Path,
    prediction_manifest: str | Path,
    output_dir: str | Path,
    model_config: FailureModelConfig,
    split: str = "train",
    ground_truth_column: str = "mask_name",
    prediction_column: str = "prediction_mask",
    minimum_instance_area: int = 15,
    density_radius: float = 64.0,
    contact_tolerance: int = 1,
    descriptor_bins: tuple[int, int, int] = (8, 6, 8),
    support_exponent: float = 0.5,
    failure_exponent: float = 1.0,
    epsilon: float = 1e-3,
    round_index: int = 0,
    overwrite: bool = False,
) -> dict[str, object]:
    """Fit one failure landscape and write the next-round controller policy."""
    dataset_root = Path(dataset_path).expanduser().resolve()
    prediction_path = Path(prediction_manifest).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    _prepare_output(output_path, overwrite)
    priors = extract_layout_priors(
        dataset_root,
        split=split,
        minimum_instance_area=minimum_instance_area,
        density_radius=density_radius,
        contact_tolerance=contact_tolerance,
    )
    observations = _collect_observations(
        priors,
        _ground_truth_paths(dataset_root, split, ground_truth_column),
        _prediction_paths(prediction_path, prediction_column),
    )
    descriptors = np.asarray(
        [
            (
                row["local_density"],
                row["contact_number"],
                row["elongation"],
            )
            for row in observations
        ],
        dtype=float,
    )
    targets = np.asarray([row["failure_score"] for row in observations], dtype=float)
    fitted_model = fit_failure_model(descriptors, targets, model_config)
    fitted_model.save(output_path / "failure_model.pt")
    observed_lookup = {
        (str(row["sample_id"]), int(row["instance_label"])): float(
            row["failure_score"]
        )
        for row in observations
    }

    failure_rows = []
    bin_rows = []
    bin_edges = {}
    for category in sorted(priors):
        prior = priors[category]
        category_descriptors = descriptor_matrix(prior)
        predicted = fitted_model.predict(category_descriptors)
        _, category_bin_rows, edges = configuration_policy(
            category_descriptors,
            predicted,
            number_of_bins=descriptor_bins,
            support_exponent=support_exponent,
            failure_exponent=failure_exponent,
            epsilon=epsilon,
        )
        bin_edges[category] = edges
        for row in category_bin_rows:
            bin_rows.append({"category": category, **row})
        for instance, descriptor, prediction in zip(
            prior.instances, category_descriptors, predicted
        ):
            key = (instance.sample_id, instance.instance_label)
            failure_rows.append(
                {
                    "sample_id": instance.sample_id,
                    "instance_label": instance.instance_label,
                    "category": category,
                    "local_density": float(descriptor[0]),
                    "contact_number": int(descriptor[1]),
                    "elongation": float(descriptor[2]),
                    "observed_failure_score": observed_lookup.get(key),
                    "failure_score": float(prediction),
                }
            )

    _write_csv(output_path / "observed_instance_failures.csv", observations)
    _write_csv(output_path / "failure_scores.csv", failure_rows)
    _write_csv(output_path / "controller_bins.csv", bin_rows)
    _write_csv(
        output_path / "training_loss.csv",
        [
            {"epoch": epoch + 1, "huber_loss": loss}
            for epoch, loss in enumerate(fitted_model.losses)
        ],
    )
    config = {
        "round_index": round_index,
        "dataset_path": str(dataset_root),
        "split": split,
        "prediction_manifest": str(prediction_path),
        "ground_truth_column": ground_truth_column,
        "prediction_column": prediction_column,
        "iou_thresholds": list(IOU_THRESHOLDS),
        "number_of_observed_instances": len(observations),
        "number_of_prior_instances": len(failure_rows),
        "observed_failure_mean": float(targets.mean()),
        "predicted_failure_mean": float(
            np.mean([row["failure_score"] for row in failure_rows])
        ),
        "descriptor_mean": fitted_model.descriptor_mean.tolist(),
        "descriptor_scale": fitted_model.descriptor_scale.tolist(),
        "failure_model": asdict(model_config),
        "controller": {
            "descriptor_bins": list(descriptor_bins),
            "support_exponent": support_exponent,
            "failure_exponent": failure_exponent,
            "epsilon": epsilon,
            "bin_edges_by_category": bin_edges,
        },
    }
    with (output_path / "controller_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return config
