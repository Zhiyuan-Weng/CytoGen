"""Materialize canonical CytoGen pairs for supported downstream segmenters."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from skimage.exposure import equalize_adapthist, rescale_intensity
from tqdm.auto import tqdm

from .manifest import PairRecord


def _safe_name(record: PairRecord) -> str:
    value = f"{record.source}_{record.sample_id}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if not safe:
        raise ValueError(f"Invalid sample name: {value!r}")
    return safe


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return np.asarray(tifffile.imread(path))
    with Image.open(path) as image:
        return np.asarray(image)


def _read_mask(path: Path) -> np.ndarray:
    mask = np.squeeze(np.asarray(tifffile.imread(path)))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D instance mask at {path}, got {mask.shape}")
    if np.any(mask < 0):
        raise ValueError(f"Negative labels found in {path}")
    return mask.astype(np.int64, copy=False)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    dtype = (
        np.uint16
        if int(mask.max(initial=0)) <= np.iinfo(np.uint16).max
        else np.uint32
    )
    tifffile.imwrite(path, mask.astype(dtype))


def _write_manifest(
    records: list[PairRecord],
    segmentation_task: str,
    output_path: Path,
    exported_paths: dict[tuple[str, str], tuple[Path, Path]] | None = None,
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = record.to_dict(segmentation_task)
            key = (record.source, record.sample_id)
            if exported_paths and key in exported_paths:
                image_path, mask_path = exported_paths[key]
                row["exported_image_path"] = str(image_path)
                row["exported_mask_path"] = str(mask_path)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _export_cellpose_family(
    records: list[PairRecord],
    output_dir: Path,
    segmentation_task: str,
) -> dict[tuple[str, str], tuple[Path, Path]]:
    train_dir = output_dir / "train"
    train_dir.mkdir()
    exported = {}
    for record in tqdm(records, desc="Cellpose/Omnipose pairs"):
        name = _safe_name(record)
        image_path = train_dir / f"{name}.tif"
        mask_path = train_dir / f"{name}_masks.tif"
        image = _read_image(record.image_path)
        mask = _read_mask(record.selected_mask(segmentation_task))
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Spatial mismatch for {record.sample_id}: {image.shape} versus {mask.shape}"
            )
        tifffile.imwrite(image_path, image)
        _write_mask(mask_path, mask)
        exported[(record.source, record.sample_id)] = (
            image_path.resolve(),
            mask_path.resolve(),
        )
    return exported


def _mask_annotations(mask: np.ndarray) -> list[dict[str, object]]:
    annotations = []
    for instance_label in np.unique(mask):
        if instance_label == 0:
            continue
        binary = (mask == instance_label).astype(np.uint8)
        rows, columns = np.where(binary)
        if not len(rows):
            continue
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = [
            contour.flatten().astype(float).tolist()
            for contour in contours
            if len(contour) >= 3
        ]
        if not polygons:
            continue
        annotations.append(
            {
                "bbox": [
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max()),
                    int(rows.max()),
                ],
                "bbox_mode": 0,
                "category_id": 0,
                "segmentation": polygons,
            }
        )
    return annotations


def _cellotype_image(path: Path) -> np.ndarray:
    image = np.asarray(_read_image(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[-1] not in {3, 4}:
        raise ValueError(f"Unsupported CelloType image shape at {path}: {image.shape}")
    image = image[..., :3]
    image = rescale_intensity(image, out_range=(0.0, 1.0))
    image = equalize_adapthist(image, kernel_size=None)
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def _export_cellotype(
    records: list[PairRecord],
    output_dir: Path,
    segmentation_task: str,
) -> dict[tuple[str, str], tuple[Path, Path]]:
    image_dir = output_dir / "train"
    mask_dir = output_dir / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    dataset_dicts = []
    exported = {}
    for image_id, record in enumerate(tqdm(records, desc="CelloType records")):
        name = _safe_name(record)
        image_path = image_dir / f"{name}.png"
        Image.fromarray(_cellotype_image(record.image_path), mode="RGB").save(
            image_path
        )
        mask = _read_mask(record.selected_mask(segmentation_task))
        mask_path = mask_dir / f"{name}.tif"
        _write_mask(mask_path, mask)
        image = _read_image(image_path)
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Spatial mismatch for {record.sample_id}: {image.shape} versus {mask.shape}"
            )
        dataset_dicts.append(
            {
                "file_name": str(image_path.resolve()),
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
                "image_id": image_id,
                "annotations": _mask_annotations(mask),
                "source": record.source,
                "sample_id": record.sample_id,
                "category": record.category,
            }
        )
        exported[(record.source, record.sample_id)] = (
            image_path.resolve(),
            mask_path.resolve(),
        )
    np.save(
        output_dir / "dataset_dicts_cell_train.npy",
        np.asarray(dataset_dicts, dtype=object),
        allow_pickle=True,
    )
    return exported


def export_training_dataset(
    records: list[PairRecord],
    output_dir: str | Path,
    output_format: str,
    segmentation_task: str = "whole_cell",
    overwrite: bool = False,
    selection_config: dict[str, object] | None = None,
) -> None:
    """Export selected pairs and their canonical manifest."""
    if output_format not in {"manifest", "cellpose", "omnipose", "cellotype"}:
        raise ValueError(f"Unsupported output format: {output_format}")
    if segmentation_task not in {"whole_cell", "nuclear"}:
        raise ValueError(f"Unsupported segmentation task: {segmentation_task}")
    if not records:
        raise ValueError("No training records were selected")
    exported_names = [_safe_name(record) for record in records]
    if len(exported_names) != len(set(exported_names)):
        raise ValueError("Selected samples collide after filename sanitization")
    output_path = Path(output_dir).expanduser().resolve()
    _prepare_output(output_path, overwrite)
    exported = None
    if output_format in {"cellpose", "omnipose"}:
        exported = _export_cellpose_family(records, output_path, segmentation_task)
    elif output_format == "cellotype":
        exported = _export_cellotype(records, output_path, segmentation_task)
    _write_manifest(
        records,
        segmentation_task,
        output_path / "training_manifest.jsonl",
        exported,
    )
    source_counts = Counter(record.source for record in records)
    category_counts = Counter(record.category for record in records)
    summary = {
        "format": output_format,
        "segmentation_task": segmentation_task,
        "number_of_samples": len(records),
        "source_counts": dict(sorted(source_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "selection": selection_config or {},
    }
    with (output_path / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
