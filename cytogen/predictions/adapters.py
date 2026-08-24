"""Convert Cellpose2, Omnipose, and CelloType outputs to one mask manifest."""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


MASK_SUFFIXES = ("_pred_masks", "_masks", "_mask", "_seg")
MASK_EXTENSIONS = {".npy", ".npz", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    metadata_index: int
    source_index: int | None
    aliases: tuple[str, ...]
    ground_truth_path: Path


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


def _metadata_path(dataset_path: Path, split: str) -> Path:
    candidates = (
        dataset_path / split / "metadata.jsonl",
        dataset_path / "metadata.jsonl",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(candidates[0])
    return path


def _resolve_asset(
    dataset_path: Path,
    metadata_path: Path,
    split: str,
    value: object,
) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        resolved = path
    else:
        candidates = (
            dataset_path / split / path,
            dataset_path / path,
            metadata_path.parent / path,
        )
        resolved = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0],
        )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved.resolve()


def _strip_mask_suffix(stem: str, extra_suffix: str | None = None) -> str:
    suffixes = ((extra_suffix,) if extra_suffix else ()) + MASK_SUFFIXES
    value = stem
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if suffix and value.endswith(suffix):
                value = value[: -len(suffix)]
                changed = True
                break
    return value


def _mask_file_priority(stem: str, extra_suffix: str | None) -> int:
    ordered_suffixes = tuple(
        suffix
        for suffix in (extra_suffix, "_pred_masks", "_seg", "_masks", "_mask")
        if suffix
    )
    for priority, suffix in enumerate(ordered_suffixes):
        if stem.endswith(suffix):
            return priority
    return len(ordered_suffixes)


def _path_aliases(value: object) -> set[str]:
    stem = Path(str(value)).stem
    return {stem, _strip_mask_suffix(stem)}


def _load_samples(
    dataset_path: str | Path,
    split: str,
    ground_truth_column: str,
) -> tuple[Path, list[DatasetSample]]:
    dataset_root = Path(dataset_path).expanduser().resolve()
    metadata_file = _metadata_path(dataset_root, split)
    samples = []
    sample_ids = set()
    for metadata_index, row in enumerate(_read_jsonl(metadata_file)):
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id={sample_id!r}")
        sample_ids.add(sample_id)
        if ground_truth_column not in row:
            raise KeyError(
                f"Dataset record {sample_id!r} has no {ground_truth_column!r}"
            )
        aliases = {sample_id, _strip_mask_suffix(sample_id)}
        for key in ("file_name", "source_image", "image", "mask_name"):
            if row.get(key) is not None:
                aliases.update(_path_aliases(row[key]))
        source_index = None
        if row.get("source_index") is not None:
            source_index = int(row["source_index"])
            aliases.update({str(source_index), f"X_{source_index}"})
        samples.append(
            DatasetSample(
                sample_id=sample_id,
                metadata_index=metadata_index,
                source_index=source_index,
                aliases=tuple(sorted(aliases)),
                ground_truth_path=_resolve_asset(
                    dataset_root,
                    metadata_file,
                    split,
                    row[ground_truth_column],
                ),
            )
        )
    return metadata_file, samples


def _extract_mask_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=True)
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        if isinstance(value, dict):
            key = next(
                (name for name in ("masks", "mask", "prediction") if name in value),
                None,
            )
            if key is None:
                raise KeyError(f"Cannot identify a mask array in {path}")
            value = value[key]
        mask = value
    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            key = next(
                (
                    name
                    for name in ("masks", "mask", "prediction", "arr_0")
                    if name in archive
                ),
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
    if not np.issubdtype(mask.dtype, np.integer):
        if not np.all(np.isfinite(mask)) or not np.all(mask == np.rint(mask)):
            raise ValueError(f"Instance mask contains non-integer values: {path}")
    mask = np.asarray(mask, dtype=np.int64)
    if np.any(mask < 0):
        raise ValueError(f"Instance mask contains negative labels: {path}")
    return mask


def _relabel_mask(mask: np.ndarray) -> np.ndarray:
    labels = np.unique(mask)
    labels = labels[labels > 0]
    if labels.size == 0:
        return np.zeros(mask.shape, dtype=np.uint16)
    relabeled = np.zeros(mask.shape, dtype=np.uint32)
    foreground = mask > 0
    relabeled[foreground] = np.searchsorted(labels, mask[foreground]) + 1
    if labels.size <= np.iinfo(np.uint16).max:
        return relabeled.astype(np.uint16)
    return relabeled


def _validate_shape(mask: np.ndarray, sample: DatasetSample) -> None:
    ground_truth = _extract_mask_array(sample.ground_truth_path)
    if mask.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch for {sample.sample_id!r}: prediction {mask.shape}, "
            f"ground truth {ground_truth.shape}"
        )


def _prepare_output(output_dir: Path, overwrite: bool) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    return masks_dir


def _output_mask_path(
    masks_dir: Path,
    sample: DatasetSample,
) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", sample.sample_id).strip("._")
    if not safe_id:
        safe_id = "sample"
    return masks_dir / f"{sample.metadata_index:06d}_{safe_id}.tif"


def _write_outputs(
    output_dir: Path,
    samples: list[DatasetSample],
    masks: dict[str, np.ndarray],
    summary: dict[str, object],
) -> dict[str, object]:
    manifest_path = output_dir / "predictions.jsonl"
    number_of_instances = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            mask = _relabel_mask(masks[sample.sample_id])
            _validate_shape(mask, sample)
            output_path = _output_mask_path(output_dir / "masks", sample)
            tifffile.imwrite(output_path, mask)
            instances = int(mask.max())
            number_of_instances += instances
            record = {
                "sample_id": sample.sample_id,
                "prediction_mask": str(output_path.relative_to(output_dir)),
                "number_of_instances": instances,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary.update(
        {
            "number_of_samples": len(samples),
            "number_of_instances": number_of_instances,
            "prediction_manifest": str(manifest_path.resolve()),
            "mask_directory": str((output_dir / "masks").resolve()),
        }
    )
    summary_path = output_dir / "adapter_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _unique_alias_map(samples: list[DatasetSample]) -> dict[str, str]:
    owners: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        for alias in sample.aliases:
            owners[alias].add(sample.sample_id)
    return {
        alias: next(iter(sample_ids))
        for alias, sample_ids in owners.items()
        if len(sample_ids) == 1
    }


def adapt_mask_predictions(
    dataset_path: str | Path,
    prediction_dir: str | Path,
    output_dir: str | Path,
    model_family: str,
    split: str = "train",
    ground_truth_column: str = "mask_name",
    filename_suffix: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Adapt Cellpose2 or Omnipose instance-label files."""
    if model_family not in {"cellpose", "omnipose"}:
        raise ValueError("model_family must be 'cellpose' or 'omnipose'")
    metadata_path, samples = _load_samples(dataset_path, split, ground_truth_column)
    prediction_root = Path(prediction_dir).expanduser().resolve()
    if not prediction_root.is_dir():
        raise NotADirectoryError(prediction_root)
    output_path = Path(output_dir).expanduser().resolve()
    alias_map = _unique_alias_map(samples)
    matched_paths: dict[str, tuple[int, Path]] = {}
    ignored_files = 0
    for path in sorted(prediction_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MASK_EXTENSIONS:
            continue
        if path.name.endswith("_flows.tif") or path.name.endswith("_flows.tiff"):
            ignored_files += 1
            continue
        stem = _strip_mask_suffix(path.stem, filename_suffix)
        sample_id = alias_map.get(stem)
        if sample_id is None:
            ignored_files += 1
            continue
        priority = _mask_file_priority(path.stem, filename_suffix)
        existing = matched_paths.get(sample_id)
        if existing is not None and priority == existing[0]:
            raise ValueError(
                f"Multiple prediction files match {sample_id!r}: "
                f"{existing[1]} and {path}"
            )
        if existing is None or priority < existing[0]:
            if existing is not None:
                ignored_files += 1
            matched_paths[sample_id] = (priority, path.resolve())
        else:
            ignored_files += 1
    missing = [
        sample.sample_id for sample in samples if sample.sample_id not in matched_paths
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing predictions for {len(missing)} samples: {preview}")
    masks = {
        sample.sample_id: _extract_mask_array(matched_paths[sample.sample_id][1])
        for sample in samples
    }
    _prepare_output(output_path, overwrite)
    return _write_outputs(
        output_path,
        samples,
        masks,
        {
            "adapter": "instance_mask_directory",
            "model_family": model_family,
            "dataset_path": str(Path(dataset_path).expanduser().resolve()),
            "split": split,
            "dataset_metadata": str(metadata_path.resolve()),
            "ground_truth_column": ground_truth_column,
            "prediction_source": str(prediction_root),
            "filename_suffix": filename_suffix,
            "ignored_files": ignored_files,
        },
    )


def _coco_mask(segmentation: object, height: int, width: int) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError(
            "CelloType COCO adaptation requires pycocotools"
        ) from exc
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        decoded = mask_utils.decode(rle)
    elif isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(mask_utils.merge(rles))
    else:
        raise TypeError(
            f"Unsupported COCO segmentation type: {type(segmentation).__name__}"
        )
    decoded = np.asarray(decoded, dtype=bool)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    if decoded.shape != (height, width):
        raise ValueError(
            f"Decoded COCO mask has shape {decoded.shape}, expected {(height, width)}"
        )
    return decoded


def _match_coco_images(
    images: list[dict[str, object]],
    samples: list[DatasetSample],
    mapping_mode: str,
) -> dict[int, DatasetSample]:
    if mapping_mode not in {"auto", "filename", "index", "order"}:
        raise ValueError("mapping_mode must be auto, filename, index, or order")
    alias_map = _unique_alias_map(samples)
    samples_by_id = {sample.sample_id: sample for sample in samples}
    source_indices: dict[int, DatasetSample] = {}
    for sample in samples:
        if sample.source_index is not None:
            if sample.source_index in source_indices:
                raise ValueError(f"Duplicate source_index={sample.source_index}")
            source_indices[sample.source_index] = sample
    mapping = {}
    used_samples = set()
    for position, image in enumerate(images):
        image_id = int(image["id"])
        sample = None
        file_stem = Path(str(image.get("file_name", ""))).stem
        if mapping_mode in {"auto", "filename"}:
            sample_id = alias_map.get(_strip_mask_suffix(file_stem))
            if sample_id is not None:
                sample = samples_by_id[sample_id]
        if sample is None and mapping_mode in {"auto", "index"}:
            match = re.search(r"(?:^|_)X?_(\d+)$", file_stem)
            source_index = int(match.group(1)) if match else None
            if source_index is None and image_id in source_indices:
                source_index = image_id
            if source_index is not None:
                sample = source_indices.get(source_index)
        if sample is None and mapping_mode in {"auto", "order"}:
            if len(images) == len(samples):
                sample = samples[position]
        if sample is None:
            raise ValueError(
                f"Cannot map COCO image_id={image_id}, file_name={file_stem!r} "
                f"with mapping_mode={mapping_mode!r}"
            )
        if sample.sample_id in used_samples:
            raise ValueError(f"Multiple COCO images map to {sample.sample_id!r}")
        used_samples.add(sample.sample_id)
        mapping[image_id] = sample
    missing = [
        sample.sample_id for sample in samples if sample.sample_id not in used_samples
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"COCO ground truth omits {len(missing)} samples: {preview}")
    return mapping


def adapt_coco_predictions(
    dataset_path: str | Path,
    prediction_json: str | Path,
    coco_ground_truth: str | Path,
    output_dir: str | Path,
    split: str = "train",
    ground_truth_column: str = "mask_name",
    score_threshold: float = 0.0,
    max_predictions: int = 0,
    mapping_mode: str = "auto",
    overwrite: bool = False,
) -> dict[str, object]:
    """Rasterize CelloType COCO predictions into canonical instance masks."""
    if max_predictions < 0:
        raise ValueError("max_predictions must be non-negative")
    metadata_path, samples = _load_samples(dataset_path, split, ground_truth_column)
    prediction_path = Path(prediction_json).expanduser().resolve()
    ground_truth_path = Path(coco_ground_truth).expanduser().resolve()
    with prediction_path.open(encoding="utf-8") as handle:
        predictions = json.load(handle)
    with ground_truth_path.open(encoding="utf-8") as handle:
        coco_ground_truth_data = json.load(handle)
    if not isinstance(predictions, list):
        raise TypeError("CelloType prediction JSON must contain a COCO result list")
    images = coco_ground_truth_data.get("images")
    if not isinstance(images, list):
        raise KeyError("COCO ground truth has no images list")
    image_mapping = _match_coco_images(images, samples, mapping_mode)
    image_metadata = {int(image["id"]): image for image in images}
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    number_after_threshold = 0
    for json_index, prediction in enumerate(predictions):
        image_id = int(prediction["image_id"])
        if image_id not in image_mapping:
            raise KeyError(f"Prediction refers to unknown COCO image_id={image_id}")
        score = float(prediction.get("score", 0.0))
        if score < score_threshold:
            continue
        item = dict(prediction)
        item["_score"] = score
        item["_json_index"] = json_index
        grouped[image_id].append(item)
        number_after_threshold += 1
    masks = {}
    suppressed_predictions = 0
    written_predictions = 0
    for image_id, sample in image_mapping.items():
        image = image_metadata[image_id]
        height = int(image["height"])
        width = int(image["width"])
        canvas = np.zeros((height, width), dtype=np.uint32)
        ordered = sorted(
            grouped.get(image_id, []),
            key=lambda item: (-float(item["_score"]), int(item["_json_index"])),
        )
        if max_predictions:
            ordered = ordered[:max_predictions]
        next_label = 1
        for prediction in ordered:
            instance = _coco_mask(prediction["segmentation"], height, width)
            visible = instance & (canvas == 0)
            if not np.any(visible):
                suppressed_predictions += 1
                continue
            canvas[visible] = next_label
            next_label += 1
            written_predictions += 1
        masks[sample.sample_id] = canvas
    output_path = Path(output_dir).expanduser().resolve()
    _prepare_output(output_path, overwrite)
    return _write_outputs(
        output_path,
        samples,
        masks,
        {
            "adapter": "cellotype_coco",
            "model_family": "cellotype",
            "dataset_path": str(Path(dataset_path).expanduser().resolve()),
            "split": split,
            "dataset_metadata": str(metadata_path.resolve()),
            "ground_truth_column": ground_truth_column,
            "prediction_source": str(prediction_path),
            "coco_ground_truth": str(ground_truth_path),
            "mapping_mode": mapping_mode,
            "score_threshold": float(score_threshold),
            "max_predictions_per_image": int(max_predictions),
            "overlap_policy": "score_descending_then_json_order_claims_free_pixels",
            "number_of_predictions_in_json": len(predictions),
            "number_of_predictions_after_threshold": number_after_threshold,
            "number_of_predictions_written": written_predictions,
            "number_of_predictions_fully_suppressed": suppressed_predictions,
        },
    )
