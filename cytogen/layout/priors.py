"""Biological prior estimation for the CytoGen layout generator."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree
from skimage.measure import regionprops


TISSUENET_CATEGORIES = sorted(
    [
        ("breast", "imc"),
        ("breast", "mibi"),
        ("breast", "vectra"),
        ("gi", "codex"),
        ("gi", "mibi"),
        ("gi", "mxif"),
        ("immune", "cycif"),
        ("immune", "mibi"),
        ("immune", "vectra"),
        ("lung", "cycif"),
        ("lung", "mibi"),
        ("pancreas", "codex"),
        ("pancreas", "vectra"),
        ("skin", "mibi"),
    ]
)
TISSUENET_CATEGORY_TO_INDEX = {
    f"{tissue}_{platform}": index
    for index, (tissue, platform) in enumerate(TISSUENET_CATEGORIES)
}


@dataclass
class InstancePrior:
    sample_id: str
    instance_label: int
    area: float
    elongation: float
    orientation: float
    local_density: float
    contact_number: int
    nuclear_ratio: float | None = None


@dataclass
class CategoryPrior:
    name: str
    category_index: int
    number_of_categories: int
    field_counts: list[int] = field(default_factory=list)
    field_coverages: list[float] = field(default_factory=list)
    field_areas: list[int] = field(default_factory=list)
    instances: list[InstancePrior] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    @property
    def dual_compartment(self) -> bool:
        return any(instance.nuclear_ratio is not None for instance in self.instances)

    def summary(self) -> dict[str, object]:
        instance_areas = np.asarray([item.area for item in self.instances], dtype=float)
        elongations = np.asarray([item.elongation for item in self.instances], dtype=float)
        contacts = np.asarray([item.contact_number for item in self.instances], dtype=float)
        nuclear_ratios = np.asarray(
            [item.nuclear_ratio for item in self.instances if item.nuclear_ratio is not None],
            dtype=float,
        )
        return {
            "category": self.name,
            "category_index": self.category_index,
            "number_of_categories": self.number_of_categories,
            "number_of_real_fields": len(self.field_counts),
            "number_of_real_instances": len(self.instances),
            "dual_compartment": self.dual_compartment,
            "field_count_quantiles": _quantiles(self.field_counts),
            "field_coverage_quantiles": _quantiles(self.field_coverages),
            "instance_area_quantiles": _quantiles(instance_areas),
            "elongation_quantiles": _quantiles(elongations),
            "contact_quantiles": _quantiles(contacts),
            "nuclear_ratio_quantiles": _quantiles(nuclear_ratios),
        }


def _quantiles(values) -> dict[str, float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    levels = (0.05, 0.25, 0.5, 0.75, 0.95)
    result = np.quantile(values, levels)
    return {f"q{int(level * 100):02d}": float(value) for level, value in zip(levels, result)}


def _resolve_record_path(dataset_path: Path, split: str, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    split_path = dataset_path / split / path
    return split_path if split_path.exists() else dataset_path / path


def _read_metadata(dataset_path: Path, split: str) -> list[dict[str, object]]:
    metadata_path = dataset_path / split / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    records = []
    with metadata_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "mask_name" not in record:
                raise KeyError(f"{metadata_path}:{line_number} has no mask_name")
            records.append(record)
    return records


def _contact_counts(
    mask: np.ndarray,
    valid_labels: set[int],
    tolerance: int,
) -> dict[int, int]:
    radius = max(1, tolerance + 1)
    edges: set[tuple[int, int]] = set()
    height, width = mask.shape
    for row_offset in range(0, radius + 1):
        for column_offset in range(-radius, radius + 1):
            if row_offset == 0 and column_offset <= 0:
                continue
            if row_offset * row_offset + column_offset * column_offset > radius * radius:
                continue
            row_start_a = max(0, -row_offset)
            row_end_a = min(height, height - row_offset)
            column_start_a = max(0, -column_offset)
            column_end_a = min(width, width - column_offset)
            first = mask[
                row_start_a:row_end_a,
                column_start_a:column_end_a,
            ]
            second = mask[
                row_start_a + row_offset : row_end_a + row_offset,
                column_start_a + column_offset : column_end_a + column_offset,
            ]
            different = (first != second) & (first != 0) & (second != 0)
            if not np.any(different):
                continue
            pairs = np.column_stack((first[different], second[different]))
            pairs.sort(axis=1)
            for left, right in np.unique(pairs, axis=0):
                left = int(left)
                right = int(right)
                if left in valid_labels and right in valid_labels:
                    edges.add((left, right))
    counts = {label: 0 for label in valid_labels}
    for left, right in edges:
        counts[left] += 1
        counts[right] += 1
    return counts


def _local_densities(
    centroids: np.ndarray,
    shape: tuple[int, int],
    radius: float,
) -> np.ndarray:
    if centroids.size == 0:
        return np.empty(0, dtype=float)
    tree = cKDTree(centroids)
    counts = np.asarray(
        [len(neighbors) for neighbors in tree.query_ball_point(centroids, radius, p=np.inf)],
        dtype=float,
    )
    height, width = shape
    row_extent = np.minimum(centroids[:, 0] + radius, height) - np.maximum(
        centroids[:, 0] - radius, 0
    )
    column_extent = np.minimum(centroids[:, 1] + radius, width) - np.maximum(
        centroids[:, 1] - radius, 0
    )
    return counts / np.maximum(row_extent * column_extent, 1.0)


def _nuclear_ratios(
    whole_cell_mask: np.ndarray,
    nuclear_mask: np.ndarray,
    valid_areas: dict[int, float],
) -> dict[int, float]:
    assigned_pixels = {label: 0 for label in valid_areas}
    for nuclear_label in np.unique(nuclear_mask):
        if nuclear_label == 0:
            continue
        nucleus = nuclear_mask == nuclear_label
        labels, counts = np.unique(whole_cell_mask[nucleus], return_counts=True)
        foreground = labels != 0
        if not np.any(foreground):
            continue
        labels = labels[foreground]
        counts = counts[foreground]
        whole_label = int(labels[np.argmax(counts)])
        if whole_label in assigned_pixels:
            assigned_pixels[whole_label] += int(counts.max())
    ratios = {}
    for label, overlap_area in assigned_pixels.items():
        ratio = overlap_area / valid_areas[label]
        if overlap_area > 0 and 0.0 < ratio <= 1.0:
            ratios[label] = float(ratio)
    return ratios


def extract_layout_priors(
    dataset_path: Path,
    split: str = "train",
    minimum_instance_area: int = 15,
    density_radius: float = 64.0,
    contact_tolerance: int = 1,
) -> dict[str, CategoryPrior]:
    """Extract only the real-mask measurements required for layout synthesis."""
    dataset_path = Path(dataset_path).expanduser()
    records = _read_metadata(dataset_path, split)
    category_names = sorted({str(record.get("category", "default")) for record in records})
    fallback_indices = {name: index for index, name in enumerate(category_names)}
    observed_category_count = max(
        [int(record.get("number_of_categories", 0)) for record in records] + [0]
    )
    fallback_count = max(observed_category_count, len(category_names), 1)
    priors: dict[str, CategoryPrior] = {}

    for record in records:
        category = str(record.get("category", "default"))
        category_index = record.get("category_index")
        number_of_categories = record.get("number_of_categories")
        if category_index is None and category in TISSUENET_CATEGORY_TO_INDEX:
            category_index = TISSUENET_CATEGORY_TO_INDEX[category]
            number_of_categories = len(TISSUENET_CATEGORIES)
        if category_index is None:
            category_index = fallback_indices[category]
        if number_of_categories is None:
            number_of_categories = fallback_count
        prior = priors.setdefault(
            category,
            CategoryPrior(
                name=category,
                category_index=int(category_index),
                number_of_categories=int(number_of_categories),
            ),
        )

        mask_path = _resolve_record_path(dataset_path, split, str(record["mask_name"]))
        mask = np.squeeze(tifffile.imread(mask_path)).astype(np.int64, copy=False)
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask at {mask_path}, got {mask.shape}")
        properties = [
            prop for prop in regionprops(mask) if prop.area > minimum_instance_area
        ]
        if not properties:
            continue
        valid_labels = {int(prop.label) for prop in properties}
        valid_areas = {int(prop.label): float(prop.area) for prop in properties}
        retained_mask = mask.copy()
        retained_mask[~np.isin(retained_mask, list(valid_labels))] = 0
        contacts = _contact_counts(retained_mask, valid_labels, contact_tolerance)
        centroids = np.asarray([prop.centroid for prop in properties], dtype=float)
        densities = _local_densities(centroids, mask.shape, density_radius)

        nuclear_ratios: dict[int, float] = {}
        nuclear_name = record.get("nuclear_mask_name")
        if nuclear_name:
            nuclear_path = _resolve_record_path(dataset_path, split, str(nuclear_name))
            nuclear_mask = np.squeeze(tifffile.imread(nuclear_path)).astype(
                np.int64, copy=False
            )
            nuclear_ratios = _nuclear_ratios(mask, nuclear_mask, valid_areas)

        sample_id = str(record.get("sample_id", mask_path.stem))
        for prop, density in zip(properties, densities):
            minor_axis = max(float(prop.axis_minor_length), 1e-6)
            elongation = max(float(prop.axis_major_length) / minor_axis, 1.0)
            label = int(prop.label)
            prior.instances.append(
                InstancePrior(
                    sample_id=sample_id,
                    instance_label=label,
                    area=float(prop.area),
                    elongation=elongation,
                    orientation=float(prop.orientation),
                    local_density=float(density),
                    contact_number=contacts.get(label, 0),
                    nuclear_ratio=nuclear_ratios.get(label),
                )
            )
        prior.field_counts.append(len(properties))
        prior.field_coverages.append(float(np.count_nonzero(retained_mask) / mask.size))
        prior.field_areas.append(int(mask.size))
        text = str(record.get("text", f"Microscopy image of {category} cells"))
        if text not in prior.prompts:
            prior.prompts.append(text)

    empty_categories = [name for name, prior in priors.items() if not prior.instances]
    if empty_categories:
        raise ValueError(f"No valid instances found for categories: {empty_categories}")
    return priors


def load_failure_scores(path: Path | None) -> dict[tuple[str, int], float]:
    """Load optional instance failure scores produced by a downstream segmenter."""
    if path is None:
        return {}
    scores = {}
    with Path(path).expanduser().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "instance_label", "failure_score"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise KeyError(f"Failure CSV must contain {sorted(required)}")
        for row in reader:
            score = float(row["failure_score"])
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"failure_score must be in [0, 1], got {score}")
            scores[(str(row["sample_id"]), int(row["instance_label"]))] = score
    return scores
