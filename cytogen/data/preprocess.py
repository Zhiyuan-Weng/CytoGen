"""Core dataset preprocessing for CytoGen image/condition pairs."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

from .hsv import encode_dual_compartment, encode_single_compartment, hsv_to_rgb


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
    category: index for index, category in enumerate(TISSUENET_CATEGORIES)
}


def _decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_channel(channel: np.ndarray, lower: float, upper: float) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float32)
    lower_value, upper_value = np.percentile(channel, (lower, upper))
    channel = np.clip(channel, lower_value, upper_value) - lower_value
    denominator = float(channel.max())
    if denominator > 1e-8:
        channel /= denominator
    else:
        channel.fill(0)
    return channel


def normalize_tissuenet_image(image: np.ndarray) -> np.ndarray:
    """Convert TissueNet [nuclear, whole-cell] channels to RGB [0, whole, nuclear]."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] < 2:
        raise ValueError(f"Expected TissueNet HxWx2 image, got {image.shape}")
    nuclear = _normalize_channel(image[..., 0], 0.1, 99.9)
    whole_cell = _normalize_channel(image[..., 1], 0.1, 99.9)
    rgb = np.stack((np.zeros_like(nuclear), whole_cell, nuclear), axis=-1)
    return np.rint(rgb * 255).astype(np.uint8)


def normalize_microscopy_image(image: np.ndarray) -> np.ndarray:
    """Percentile-normalize a grayscale or multichannel microscopy image to RGB."""
    image = np.squeeze(np.asarray(image))
    if image.ndim == 2:
        channel = _normalize_channel(image, 0.01, 99.99)
        rgb = np.stack((channel, channel, channel), axis=-1)
    elif image.ndim == 3:
        if image.shape[0] <= 4 and image.shape[-1] > 4:
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] == 1:
            channel = _normalize_channel(image[..., 0], 0.01, 99.99)
            rgb = np.stack((channel, channel, channel), axis=-1)
        elif image.shape[-1] >= 3:
            rgb = np.stack(
                [_normalize_channel(image[..., index], 0.01, 99.99) for index in range(3)],
                axis=-1,
            )
        else:
            raise ValueError(f"Cannot convert image with shape {image.shape} to RGB")
    else:
        raise ValueError(f"Expected a 2D or 3D microscopy image, got {image.shape}")
    return np.rint(rgb * 255).astype(np.uint8)


def read_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return tifffile.imread(path)
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        archive = np.load(path)
        if len(archive.files) != 1:
            raise ValueError(f"{path} contains multiple arrays; use a TIFF, PNG, or NPY file")
        return archive[archive.files[0]]
    return np.asarray(Image.open(path))


def _resize_rgb(image: np.ndarray, size: int | None) -> np.ndarray:
    if size is None:
        return image
    return np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR))


def _resize_mask(mask: np.ndarray, size: int | None) -> np.ndarray:
    mask = np.squeeze(np.asarray(mask))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D instance mask, got {mask.shape}")
    if size is None:
        return mask
    resized = Image.fromarray(mask.astype(np.int32), mode="I").resize(
        (size, size), Image.Resampling.NEAREST
    )
    return np.asarray(resized).astype(mask.dtype, copy=False)


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError(f"Invalid empty sample id derived from {value!r}")
    return safe


def _output_dirs(output_root: Path, split: str) -> dict[str, Path]:
    split_root = output_root / split
    directories = {
        "root": split_root,
        "images": split_root / "images",
        "conditions": split_root / "conditions",
        "masks": split_root / "masks",
        "nuclear_masks": split_root / "nuclear_masks",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _write_record(
    directories: dict[str, Path],
    sample_id: str,
    image: np.ndarray,
    mask: np.ndarray,
    condition_hsv: np.ndarray,
    nuclear_mask: np.ndarray | None,
    metadata: dict[str, object],
) -> dict[str, object]:
    image_name = f"{sample_id}.png"
    condition_name = f"{sample_id}.png"
    mask_name = f"{sample_id}.tif"
    Image.fromarray(image).save(directories["images"] / image_name)
    Image.fromarray(hsv_to_rgb(condition_hsv)).save(
        directories["conditions"] / condition_name
    )
    tifffile.imwrite(directories["masks"] / mask_name, mask)

    record = {
        "file_name": f"images/{image_name}",
        "layout_name": f"conditions/{condition_name}",
        "conditioning_image": f"conditions/{condition_name}",
        "mask_name": f"masks/{mask_name}",
        **metadata,
    }
    if nuclear_mask is not None:
        tifffile.imwrite(directories["nuclear_masks"] / mask_name, nuclear_mask)
        record["nuclear_mask_name"] = f"nuclear_masks/{mask_name}"
    return record


def _write_metadata(output_root: Path, records_by_split: dict[str, list[dict]]) -> None:
    for split, records in records_by_split.items():
        metadata_path = output_root / split / "metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_tissuenet(
    input_root: Path,
    output_root: Path,
    splits: list[str],
    size: int | None = None,
    limit: int | None = None,
) -> None:
    """Prepare TissueNet NPZ archives and their dual-compartment HSV maps."""
    records_by_split: dict[str, list[dict]] = defaultdict(list)
    for split in splits:
        archive_path = input_root / f"tissuenet_v1.0_{split}.npz"
        if not archive_path.exists():
            raise FileNotFoundError(archive_path)
        directories = _output_dirs(output_root, split)
        with np.load(archive_path, allow_pickle=True) as archive:
            images = archive["X"]
            masks = archive["y"]
            tissues = archive["tissue_list"]
            platforms = archive["platform_list"]
            number_of_samples = len(images) if limit is None else min(len(images), limit)
            iterator = tqdm(range(number_of_samples), desc=f"TissueNet {split}")
            for index in iterator:
                tissue = _decode_text(tissues[index])
                platform = _decode_text(platforms[index])
                category = (tissue, platform)
                if category not in TISSUENET_CATEGORY_TO_INDEX:
                    raise ValueError(f"Unknown TissueNet category: {category}")

                image = _resize_rgb(normalize_tissuenet_image(images[index]), size)
                whole_cell_mask = _resize_mask(masks[index, ..., 0], size)
                nuclear_mask = _resize_mask(masks[index, ..., 1], size)
                condition = encode_dual_compartment(
                    whole_cell_mask,
                    nuclear_mask,
                    TISSUENET_CATEGORY_TO_INDEX[category],
                    len(TISSUENET_CATEGORIES),
                )
                sample_id = f"tissuenet_{split}_{index:05d}"
                prompt = (
                    f"Microscopy image of {tissue} tissue cells, stained for nuclear "
                    f"(blue) and whole-cell (green) markers, captured on {platform} platform"
                )
                record = _write_record(
                    directories,
                    sample_id,
                    image,
                    whole_cell_mask,
                    condition,
                    nuclear_mask,
                    {
                        "text": prompt,
                        "sample_id": sample_id,
                        "split": split,
                        "tissue": tissue,
                        "platform": platform,
                        "category": f"{tissue}_{platform}",
                        "category_index": TISSUENET_CATEGORY_TO_INDEX[
                            f"{tissue}_{platform}"
                        ],
                        "number_of_categories": len(TISSUENET_CATEGORIES),
                        "source_file": archive_path.name,
                        "source_index": index,
                    },
                )
                records_by_split[split].append(record)
    _write_metadata(output_root, records_by_split)


def _read_manifest(manifest_path: Path) -> list[dict[str, object]]:
    records = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"image", "mask", "category"} - set(record)
            if missing:
                raise ValueError(f"Line {line_number} is missing fields: {sorted(missing)}")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {manifest_path}")
    return records


def _resolve_path(value: object, manifest_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else manifest_root / path


def prepare_manifest(
    manifest_path: Path,
    output_root: Path,
    size: int | None = None,
    limit: int | None = None,
) -> None:
    """Prepare arbitrary paired microscopy images from a JSONL manifest."""
    records = _read_manifest(manifest_path)
    if limit is not None:
        records = records[:limit]
    categories = sorted({str(record["category"]) for record in records})
    category_to_index = {category: index for index, category in enumerate(categories)}
    manifest_root = manifest_path.parent
    records_by_split: dict[str, list[dict]] = defaultdict(list)
    used_ids: dict[str, set[str]] = defaultdict(set)

    for source_record in tqdm(records, desc="Manifest samples"):
        image_path = _resolve_path(source_record["image"], manifest_root)
        mask_path = _resolve_path(source_record["mask"], manifest_root)
        split = str(source_record.get("split", "train"))
        category = str(source_record["category"])
        sample_id = _safe_id(str(source_record.get("sample_id", image_path.stem)))
        if sample_id in used_ids[split]:
            raise ValueError(f"Duplicate sample_id={sample_id!r} in split={split!r}")
        used_ids[split].add(sample_id)

        image_raw = read_array(image_path)
        image = _resize_rgb(normalize_microscopy_image(image_raw), size)
        mask = _resize_mask(read_array(mask_path), size)
        nuclear_mask = None
        if source_record.get("nuclear_mask") is not None:
            nuclear_path = _resolve_path(source_record["nuclear_mask"], manifest_root)
            nuclear_mask = _resize_mask(read_array(nuclear_path), size)
            condition = encode_dual_compartment(
                mask,
                nuclear_mask,
                category_to_index[category],
                len(categories),
            )
        else:
            condition = encode_single_compartment(
                mask,
                category_to_index[category],
                len(categories),
            )

        prompt = str(source_record.get("text", f"Microscopy image of {category} cells"))
        directories = _output_dirs(output_root, split)
        record = _write_record(
            directories,
            sample_id,
            image,
            mask,
            condition,
            nuclear_mask,
            {
                "text": prompt,
                "sample_id": sample_id,
                "split": split,
                "category": category,
                "category_index": category_to_index[category],
                "number_of_categories": len(categories),
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "original_shape": list(np.asarray(image_raw).shape),
            },
        )
        records_by_split[split].append(record)
    _write_metadata(output_root, records_by_split)
