#!/usr/bin/env python3
"""Adapt the CytoGen microscopy appearance prior to a target dataset."""

from __future__ import annotations

import argparse
import logging
import math
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as functional
import torchvision.transforms.functional as vision_functional
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the UNet of a microscopy Stable Diffusion checkpoint."
    )
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--spatial_mode",
        choices=("none", "resize", "random_crop", "pad_random_crop"),
        default="none",
    )
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.3)
    parser.add_argument("--flip_prob", type=float, default=0.5)
    parser.add_argument("--rotation_prob", type=float, default=0.8)
    parser.add_argument("--max_rotation_angle", type=float, default=90.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_start_epoch", type=int, default=0)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    args = parser.parse_args()

    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    if args.train_batch_size <= 0 or args.num_train_epochs <= 0:
        parser.error("batch size and epoch count must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient_accumulation_steps must be positive")
    if args.save_start_epoch < 0 or args.save_every_n_epochs <= 0:
        parser.error("checkpoint epochs must be non-negative and non-zero")
    for name in ("cfg_dropout_prob", "flip_prob", "rotation_prob"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name} must be in [0, 1]")
    return args


def transform_spatially(image, resolution: int, mode: str):
    if mode == "none":
        if image.size != (resolution, resolution):
            raise ValueError(
                f"Expected a {resolution}x{resolution} image, got {image.size}; "
                "select --spatial_mode resize, random_crop, or pad_random_crop"
            )
        return image
    if mode == "resize":
        return vision_functional.resize(
            image,
            [resolution, resolution],
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
    if mode == "random_crop":
        if image.width < resolution or image.height < resolution:
            raise ValueError(
                f"Cannot crop {resolution}x{resolution} from image size {image.size}"
            )
    elif mode == "pad_random_crop":
        pad_width = max(resolution - image.width, 0)
        pad_height = max(resolution - image.height, 0)
        image = vision_functional.pad(
            image,
            [
                pad_width // 2,
                pad_height // 2,
                pad_width - pad_width // 2,
                pad_height - pad_height // 2,
            ],
            fill=0,
        )
    top, left, height, width = transforms.RandomCrop.get_params(
        image, output_size=(resolution, resolution)
    )
    return vision_functional.crop(image, top, left, height, width)


def resolve_resume_checkpoint(output_dir: Path, requested: str) -> Path:
    if requested == "latest":
        candidates = []
        for path in output_dir.glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found under {output_dir}")
        return max(candidates)[1]
    path = Path(requested).expanduser()
    if not path.is_absolute() and not path.exists():
        path = output_dir / path
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if not match:
        raise ValueError(f"Checkpoint directory must be named checkpoint-EPOCH: {path}")
    return int(match.group(1))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    logging_dir = output_dir / "logs"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=ProjectConfiguration(
            project_dir=str(output_dir), logging_dir=str(logging_dir)
        ),
    )
    logging.basicConfig(
        level=logging.INFO if accelerator.is_local_main_process else logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    model_path = args.pretrained_model_name_or_path
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if not args.disable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise RuntimeError("xFormers is not installed")
        unet.enable_xformers_memory_efficient_attention()

    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    dataset = load_dataset("imagefolder", data_dir=args.dataset_path)
    if args.train_split not in dataset:
        raise KeyError(
            f"Split {args.train_split!r} not found; available splits: {list(dataset)}"
        )
    train_dataset = dataset[args.train_split]
    if "image" not in train_dataset.column_names or "text" not in train_dataset.column_names:
        raise KeyError("The prepared dataset must contain image and text columns")

    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    def preprocess_train(examples):
        pixel_values = []
        prompts = []
        for image, text in zip(examples["image"], examples["text"], strict=True):
            image = transform_spatially(
                image.convert("RGB"), args.resolution, args.spatial_mode
            )
            if random.random() < args.flip_prob:
                image = vision_functional.hflip(image)
            if random.random() < args.flip_prob:
                image = vision_functional.vflip(image)
            if random.random() < args.rotation_prob:
                angle = random.uniform(-args.max_rotation_angle, args.max_rotation_angle)
                image = vision_functional.rotate(
                    image,
                    angle,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0,
                )
            pixel_values.append(normalize(image))
            prompts.append("" if random.random() < args.cfg_dropout_prob else str(text))
        input_ids = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    with accelerator.main_process_first():
        train_dataset.set_transform(preprocess_train)

    def collate_fn(examples):
        return {
            "pixel_values": torch.stack(
                [example["pixel_values"] for example in examples]
            ).contiguous(memory_format=torch.contiguous_format),
            "input_ids": torch.stack([example["input_ids"] for example in examples]),
        }

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    updates_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    total_updates = args.num_train_epochs * updates_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_updates,
    )
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    first_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint:
        resume_path = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
        accelerator.load_state(str(resume_path))
        first_epoch = checkpoint_epoch(resume_path) + 1
        global_step = first_epoch * updates_per_epoch
        accelerator.print(f"Resumed from {resume_path}")

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.init_trackers("cytogen-appearance", config=vars(args))

    last_saved_epoch = None

    def save_checkpoint(epoch: int) -> None:
        nonlocal last_saved_epoch
        save_path = output_dir / f"checkpoint-{epoch}"
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                unet=accelerator.unwrap_model(unet),
            )
            pipeline.save_pretrained(save_path)
            del pipeline
        accelerator.wait_for_everyone()
        accelerator.save_state(str(save_path))
        last_saved_epoch = epoch

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        progress = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_local_main_process,
        )
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                prediction = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {noise_scheduler.config.prediction_type}"
                    )
                loss = functional.mse_loss(
                    prediction.float(), target.float(), reduction="mean"
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.update(1)
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log(
                    {"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]},
                    step=global_step,
                )
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")
        progress.close()

        should_save = (
            epoch >= args.save_start_epoch
            and (epoch - args.save_start_epoch) % args.save_every_n_epochs == 0
        )
        if should_save:
            save_checkpoint(epoch)

    final_epoch = args.num_train_epochs - 1
    if last_saved_epoch != final_epoch:
        save_checkpoint(final_epoch)
    accelerator.end_training()


if __name__ == "__main__":
    main()
