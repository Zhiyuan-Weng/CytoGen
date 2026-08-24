#!/usr/bin/env python
"""Render CytoGen layouts with a target-adapted model and ControlNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cytogen.rendering import RenderConfig, render_layout_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--controlnet_path", required=True)
    parser.add_argument("--metadata_path", type=Path)
    parser.add_argument("--conditioning_column", default="layout_name")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--postprocess",
        choices=("none", "grayscale", "grayscale_normalize"),
        default="none",
    )
    parser.add_argument("--negative_prompt")
    parser.add_argument("--blackout_background", action="store_true")
    parser.add_argument("--background_dilation", type=int, default=5)
    parser.add_argument("--disable_xformers", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.layout_path
    config = RenderConfig(
        resolution=args.resolution,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        postprocess=args.postprocess,
        negative_prompt=args.negative_prompt,
        blackout_background=args.blackout_background,
        background_dilation=args.background_dilation,
        enable_xformers=not args.disable_xformers,
    )
    records = render_layout_dataset(
        layout_path=args.layout_path,
        output_dir=output_dir,
        appearance_model_path=args.pretrained_model_name_or_path,
        controlnet_path=args.controlnet_path,
        config=config,
        metadata_path=args.metadata_path,
        conditioning_column=args.conditioning_column,
        overwrite=args.overwrite,
        start_index=args.start_index,
        limit=args.limit,
    )
    print(f"Rendered {len(records)} paired samples in {output_dir.expanduser()}")


if __name__ == "__main__":
    main()
