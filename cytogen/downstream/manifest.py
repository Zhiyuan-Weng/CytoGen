"""Canonical real and synthetic pair manifests used by downstream adapters."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    source: str
    category: str
    image_path: Path
    mask_path: Path
    nuclear_mask_path: Path | None
    text: str | None = None

    def selected_mask(self, segmentation_task: str) -> Path:
        if segmentation_task == "whole_cell":
            return self.mask_path
        if segmentation_task == "nuclear" and self.nuclear_mask_path is not None:
            return self.nuclear_mask_path
        raise ValueError(
            f"Sample {self.sample_id!r} has no mask for task {segmentation_task!r}"
        )

    def to_dict(self, segmentation_task: str) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "category": self.category,
            "image_path": str(self.image_path),
            "mask_path": str(self.selected_mask(segmentation_task)),
            "whole_cell_mask_path": str(self.mask_path),
            "nuclear_mask_path": (
                str(self.nuclear_mask_path)
                if self.nuclear_mask_path is not None
                else None
            ),
            "segmentation_task": segmentation_task,
            "text": self.text,
        }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "sample_id" not in record:
                raise KeyError(f"{path}:{line_number} has no sample_id")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _resolve_asset(
    dataset_root: Path,
    metadata_path: Path,
    split: str | None,
    value: object,
) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        resolved = path
    else:
        candidates = []
        if split:
            candidates.append(dataset_root / split / path)
        candidates.extend((dataset_root / path, metadata_path.parent / path))
        resolved = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0],
        )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved.resolve()


def load_pair_records(
    dataset_path: str | Path,
    source: str,
    split: str | None = None,
    metadata_path: str | Path | None = None,
) -> list[PairRecord]:
    """Load rendered image-mask pairs from a prepared or synthetic dataset."""
    dataset_root = Path(dataset_path).expanduser().resolve()
    if metadata_path is not None:
        metadata_file = Path(metadata_path).expanduser().resolve()
    elif split and (dataset_root / split / "metadata.jsonl").is_file():
        metadata_file = dataset_root / split / "metadata.jsonl"
    else:
        metadata_file = dataset_root / "metadata.jsonl"
    rows = _read_jsonl(metadata_file)
    records = []
    for row in rows:
        if "file_name" not in row or "mask_name" not in row:
            raise KeyError(
                f"Record {row['sample_id']!r} must contain file_name and mask_name"
            )
        nuclear_path = None
        if row.get("nuclear_mask_name"):
            nuclear_path = _resolve_asset(
                dataset_root, metadata_file, split, row["nuclear_mask_name"]
            )
        records.append(
            PairRecord(
                sample_id=str(row["sample_id"]),
                source=source,
                category=str(row.get("category", "default")),
                image_path=_resolve_asset(
                    dataset_root, metadata_file, split, row["file_name"]
                ),
                mask_path=_resolve_asset(
                    dataset_root, metadata_file, split, row["mask_name"]
                ),
                nuclear_mask_path=nuclear_path,
                text=str(row["text"]) if row.get("text") is not None else None,
            )
        )
    return records


def _rank(seed: int, namespace: str, record: PairRecord) -> bytes:
    payload = f"{seed}:{namespace}:{record.source}:{record.sample_id}".encode()
    return hashlib.blake2b(payload, digest_size=16).digest()


def _stratified_subset(
    records: list[PairRecord],
    target_count: int,
    seed: int,
    namespace: str,
) -> list[PairRecord]:
    if target_count < 0 or target_count > len(records):
        raise ValueError(
            f"Requested {target_count} records from an available {len(records)}"
        )
    if target_count == len(records):
        return sorted(
            records,
            key=lambda record: (record.category, _rank(seed, namespace, record)),
        )
    groups: dict[str, list[PairRecord]] = defaultdict(list)
    for record in records:
        groups[record.category].append(record)
    categories = sorted(groups)
    expected = {
        category: target_count * len(groups[category]) / len(records)
        for category in categories
    }
    quotas = {category: int(math.floor(expected[category])) for category in categories}
    remaining = target_count - sum(quotas.values())
    remainder_order = sorted(
        categories,
        key=lambda category: (
            -(expected[category] - quotas[category]),
            category,
        ),
    )
    for category in remainder_order[:remaining]:
        quotas[category] += 1
    selected = []
    for category in categories:
        ordered = sorted(
            groups[category], key=lambda record: _rank(seed, namespace, record)
        )
        selected.extend(ordered[: quotas[category]])
    return selected


def build_training_records(
    real_records: list[PairRecord],
    synthetic_records: list[PairRecord],
    real_fraction: float = 1.0,
    synthetic_ratio: float | None = None,
    synthetic_count: int | None = None,
    seed: int = 42,
) -> list[PairRecord]:
    """Select a category-stratified real/synthetic training mixture."""
    if not 0.0 <= real_fraction <= 1.0:
        raise ValueError("real_fraction must be in [0, 1]")
    if synthetic_count is not None and synthetic_ratio is not None:
        raise ValueError("Specify synthetic_count or synthetic_ratio, not both")
    real_count = int(round(len(real_records) * real_fraction))
    selected_real = _stratified_subset(real_records, real_count, seed, "real")
    if synthetic_count is None:
        if synthetic_ratio is None:
            synthetic_count = len(synthetic_records)
        else:
            if synthetic_ratio < 0:
                raise ValueError("synthetic_ratio must be non-negative")
            if not selected_real and synthetic_ratio > 0:
                raise ValueError(
                    "synthetic_ratio requires selected real samples; use synthetic_count"
                )
            synthetic_count = int(round(len(selected_real) * synthetic_ratio))
    selected_synthetic = _stratified_subset(
        synthetic_records, synthetic_count, seed, "synthetic"
    )
    return selected_real + selected_synthetic
