#!/usr/bin/env python
"""Run or inspect one configuration-driven CytoGen iteration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cytogen.workflow import build_iteration_plan, run_iteration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--stop_after",
        choices=(
            "inference",
            "predictions",
            "controller",
            "layouts",
            "rendering",
            "downstream",
            "training",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_iteration_plan(args.config)
    if args.dry_run:
        print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
        return
    state = run_iteration(plan, resume=args.resume, stop_after=args.stop_after)
    print(
        f"Iteration {plan.round_index} finished with status {state['status']} in "
        f"{plan.round_dir}"
    )


if __name__ == "__main__":
    main()
