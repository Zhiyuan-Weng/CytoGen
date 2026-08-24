"""ControlNet renderer for CytoGen HSV cellular layouts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image
from scipy.ndimage import binary_dilation
from tqdm.auto import tqdm


ASSET_COLUMNS = (
    "layout_name",
    "conditioning_image",
    "mask_name",
    "nuclear_mask_name",
)


@dataclass
class RenderConfig:
    resolution: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    batch_size: int = 4
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    postprocess: str = "none"
    negative_prompt: str | None = None
    blackout_background: bool = False
    background_dilation: int = 5
    enable_xformers: bool = True

    def validate(self) -> None:
        if self.resolution <= 0 or self.resolution % 8:
            raise ValueError("resolution must be a positive multiple of 8")
        if self.num_inference_steps <= 0 or self.batch_size <= 0:
            raise ValueError("inference steps and batch size must be positive")
        if self.guidance_scale < 0 or self.controlnet_conditioning_scale < 0:
            raise ValueError("guidance scales must be non-negative")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.postprocess not in {"none", "grayscale", "grayscale_normalize"}:
            raise ValueError(f"Unsupported postprocess mode: {self.postprocess}")
        if self.background_dilation < 0:
            raise ValueError("background_dilation must be non-negative")


class BatchRenderer(Protocol):
    def render(
        self,
        prompts: list[str],
        conditions: list[Image.Image],
        seeds: list[int],
    ) -> list[Image.Image]: ...


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    """Derive a deterministic seed independent of batching and process order."""
    payload = f"{int(base_seed)}:{sample_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)


def _torch_dtype(name: str, device: str) -> torch.dtype:
    if device.startswith("cpu"):
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


class ControlNetRenderer:
    def __init__(
        self,
        appearance_model_path: str | Path,
        controlnet_path: str | Path,
        config: RenderConfig,
    ) -> None:
        self.config = config
        dtype = _torch_dtype(config.dtype, config.device)
        controlnet = ControlNetModel.from_pretrained(
            str(Path(controlnet_path).expanduser()), torch_dtype=dtype
        )
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            str(Path(appearance_model_path).expanduser()),
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline.to(config.device)
        self.pipeline.set_progress_bar_config(disable=True)
        if config.enable_xformers:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception as error:
                print(f"Warning: xformers attention was not enabled: {error}")

    def render(
        self,
        prompts: list[str],
        conditions: list[Image.Image],
        seeds: list[int],
    ) -> list[Image.Image]:
        generators = [
            torch.Generator(device=self.config.device).manual_seed(seed)
            for seed in seeds
        ]
        negative_prompts = None
        if self.config.negative_prompt:
            negative_prompts = [self.config.negative_prompt] * len(prompts)
        with torch.inference_mode():
            result = self.pipeline(
                prompt=prompts,
                negative_prompt=negative_prompts,
                image=conditions,
                height=self.config.resolution,
                width=self.config.resolution,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                controlnet_conditioning_scale=self.config.controlnet_conditioning_scale,
                generator=generators,
            )
        return result.images


def _read_metadata(path: Path) -> list[dict[str, object]]:
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
        raise ValueError(f"No layout records found in {path}")
    return records


def _resolve_asset(layout_root: Path, metadata_path: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    candidates = (layout_root / path, metadata_path.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(candidates[0])


def _safe_sample_id(value: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    if not safe:
        raise ValueError(f"Invalid sample_id: {value!r}")
    return safe


def _normalize_grayscale(array: np.ndarray) -> np.ndarray:
    lower, upper = np.percentile(array, (0.01, 99.99))
    if upper <= lower:
        return np.clip(array, 0, 255).astype(np.uint8)
    normalized = np.clip((array.astype(np.float32) - lower) / (upper - lower), 0, 1)
    return np.rint(normalized * 255).astype(np.uint8)


def postprocess_image(
    image: Image.Image,
    condition: Image.Image,
    config: RenderConfig,
) -> Image.Image:
    image = image.convert("RGB")
    if config.postprocess != "none":
        grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
        if config.postprocess == "grayscale_normalize":
            grayscale = _normalize_grayscale(grayscale)
        image = Image.fromarray(np.repeat(grayscale[..., None], 3, axis=2), mode="RGB")
    if config.blackout_background:
        foreground = np.any(np.asarray(condition.convert("RGB")) > 0, axis=2)
        if config.background_dilation:
            foreground = binary_dilation(
                foreground, iterations=config.background_dilation
            )
        output = np.asarray(image).copy()
        output[~foreground] = 0
        image = Image.fromarray(output, mode="RGB")
    return image


def _output_record(
    record: dict[str, object],
    layout_root: Path,
    metadata_path: Path,
    output_root: Path,
    image_name: str,
    render_seed: int,
    appearance_model_path: str | Path,
    controlnet_path: str | Path,
    config: RenderConfig,
) -> dict[str, object]:
    output = dict(record)
    if output_root.resolve() != layout_root.resolve():
        for column in ASSET_COLUMNS:
            if column in output:
                output[column] = str(
                    _resolve_asset(layout_root, metadata_path, output[column])
                )
    output.update(
        {
            "file_name": f"images/{image_name}",
            "render_seed": render_seed,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "controlnet_conditioning_scale": config.controlnet_conditioning_scale,
            "appearance_model_path": str(Path(appearance_model_path).expanduser()),
            "controlnet_path": str(Path(controlnet_path).expanduser()),
            "postprocess": config.postprocess,
        }
    )
    return output


def _write_jsonl_atomic(records: list[dict[str, object]], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def render_layout_dataset(
    layout_path: str | Path,
    output_dir: str | Path,
    appearance_model_path: str | Path,
    controlnet_path: str | Path,
    config: RenderConfig,
    metadata_path: str | Path | None = None,
    conditioning_column: str = "layout_name",
    overwrite: bool = False,
    start_index: int = 0,
    limit: int | None = None,
    renderer: BatchRenderer | None = None,
) -> list[dict[str, object]]:
    """Render a layout manifest and write paired-image metadata."""
    config.validate()
    if start_index < 0 or (limit is not None and limit <= 0):
        raise ValueError("start_index must be non-negative and limit must be positive")
    layout_root = Path(layout_path).expanduser().resolve()
    metadata_file = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else layout_root / "metadata.jsonl"
    )
    output_root = Path(output_dir).expanduser().resolve()
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    all_records = _read_metadata(metadata_file)
    selected_records = all_records[
        start_index : None if limit is None else start_index + limit
    ]
    if not selected_records:
        raise ValueError("No records selected for rendering")

    sample_ids = [str(record["sample_id"]) for record in all_records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")
    safe_sample_ids = [_safe_sample_id(sample_id) for sample_id in sample_ids]
    if len(safe_sample_ids) != len(set(safe_sample_ids)):
        raise ValueError("sample_id values collide after filename sanitization")

    tasks = []
    selected_output_records = []
    for record in selected_records:
        if conditioning_column not in record:
            raise KeyError(
                f"Record {record['sample_id']} has no {conditioning_column!r} field"
            )
        condition_path = _resolve_asset(
            layout_root, metadata_file, record[conditioning_column]
        )
        sample_id = _safe_sample_id(record["sample_id"])
        image_name = f"{sample_id}.png"
        image_path = image_dir / image_name
        render_seed = stable_sample_seed(config.seed, str(record["sample_id"]))
        selected_output_records.append(
            _output_record(
                record,
                layout_root,
                metadata_file,
                output_root,
                image_name,
                render_seed,
                appearance_model_path,
                controlnet_path,
                config,
            )
        )
        if overwrite or not image_path.is_file():
            tasks.append(
                {
                    "record": record,
                    "condition_path": condition_path,
                    "image_path": image_path,
                    "render_seed": render_seed,
                }
            )

    if tasks and renderer is None:
        renderer = ControlNetRenderer(
            appearance_model_path, controlnet_path, config
        )
    progress = tqdm(total=len(tasks), desc="Rendered images")
    for start in range(0, len(tasks), config.batch_size):
        batch = tasks[start : start + config.batch_size]
        conditions = [
            Image.open(task["condition_path"])
            .convert("RGB")
            .resize((config.resolution, config.resolution), Image.Resampling.NEAREST)
            for task in batch
        ]
        prompts = [
            str(
                task["record"].get(
                    "text",
                    f"Microscopy image of {task['record'].get('category', 'cells')}",
                )
            )
            for task in batch
        ]
        seeds = [int(task["render_seed"]) for task in batch]
        if renderer is None:
            raise RuntimeError("Renderer initialization failed")
        images = renderer.render(prompts, conditions, seeds)
        if len(images) != len(batch):
            raise RuntimeError("Renderer returned a different number of images")
        for task, condition, image in zip(batch, conditions, images):
            processed = postprocess_image(image, condition, config)
            processed.save(task["image_path"])
            progress.update(1)
    progress.close()

    if output_root == layout_root and len(selected_records) != len(all_records):
        selected_by_id = {
            str(record["sample_id"]): record for record in selected_output_records
        }
        metadata_records = [
            selected_by_id.get(str(record["sample_id"]), record)
            for record in all_records
        ]
    else:
        metadata_records = selected_output_records
    _write_jsonl_atomic(metadata_records, output_root / "metadata.jsonl")
    run_config = {
        "layout_path": str(layout_root),
        "metadata_path": str(metadata_file),
        "appearance_model_path": str(Path(appearance_model_path).expanduser()),
        "controlnet_path": str(Path(controlnet_path).expanduser()),
        "conditioning_column": conditioning_column,
        "number_of_selected_records": len(selected_output_records),
        "number_of_metadata_records": len(metadata_records),
        "render_config": asdict(config),
    }
    with (output_root / "generation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    return selected_output_records
