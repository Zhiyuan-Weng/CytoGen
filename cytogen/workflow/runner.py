"""Run one resumable CytoGen generation and downstream-training iteration."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template

import yaml


STAGE_ORDER = (
    "inference",
    "predictions",
    "controller",
    "layouts",
    "rendering",
    "downstream",
    "training",
)

BUILTIN_SCRIPTS = {
    "predictions": "prepare_predictions.py",
    "controller": "fit_controller.py",
    "layouts": "generate_layouts.py",
    "rendering": "render_images.py",
    "downstream": "prepare_downstream.py",
}


@dataclass(frozen=True)
class StagePlan:
    name: str
    command: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    expected_outputs: tuple[Path, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "command": list(self.command),
            "cwd": str(self.cwd),
            "environment": self.environment,
            "expected_outputs": [str(path) for path in self.expected_outputs],
        }


@dataclass(frozen=True)
class IterationPlan:
    config_path: Path
    round_index: int
    round_dir: Path
    state_path: Path
    context: dict[str, str]
    stages: tuple[StagePlan, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config_path": str(self.config_path),
            "round_index": self.round_index,
            "round_dir": str(self.round_dir),
            "state_path": str(self.state_path),
            "context": self.context,
            "stages": [stage.to_dict() for stage in self.stages],
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute_path(value: object, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _expand(value: object, context: dict[str, str]) -> object:
    if isinstance(value, str):
        return Template(value).substitute(context)
    if isinstance(value, list):
        return [_expand(item, context) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand(item, context) for item in value)
    if isinstance(value, dict):
        return {str(key): _expand(item, context) for key, item in value.items()}
    return value


def _command_tokens(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = shlex.split(value)
    elif isinstance(value, (list, tuple)):
        tokens = [str(item) for item in value]
    else:
        raise TypeError("Stage command must be a string or a list of arguments")
    if not tokens:
        raise ValueError("Stage command cannot be empty")
    return tuple(tokens)


def _args_to_cli(arguments: dict[str, object]) -> tuple[str, ...]:
    tokens = []
    for name, value in arguments.items():
        if value is None or value is False:
            continue
        flag = f"--{name}"
        if value is True:
            tokens.append(flag)
            continue
        tokens.append(flag)
        if isinstance(value, (list, tuple)):
            tokens.extend(str(item) for item in value)
        elif isinstance(value, dict):
            raise TypeError(f"Nested mapping is not valid for command argument {name!r}")
        else:
            tokens.append(str(value))
    return tuple(tokens)


def _stage_defaults(
    stage_name: str,
    context: dict[str, str],
    round_index: int,
    split: str,
    enabled_stages: set[str],
) -> dict[str, object]:
    if stage_name == "predictions":
        return {
            "dataset_path": context["real_dataset"],
            "output_dir": context["predictions_dir"],
            "split": split,
            "overwrite": True,
        }
    if stage_name == "controller":
        return {
            "dataset_path": context["real_dataset"],
            "prediction_manifest": context["prediction_manifest"],
            "output_dir": context["controller_dir"],
            "split": split,
            "round_index": round_index,
            "overwrite": True,
        }
    if stage_name == "layouts":
        defaults = {
            "dataset_path": context["real_dataset"],
            "output_dir": context["layouts_dir"],
            "split": split,
        }
        if "controller" in enabled_stages:
            defaults["failure_scores"] = context["failure_scores"]
        return defaults
    if stage_name == "rendering":
        return {
            "layout_path": context["layouts_dir"],
            "output_dir": context["rendered_dir"],
        }
    if stage_name == "downstream":
        return {
            "real_dataset": context["real_dataset"],
            "synthetic_dataset": context["rendered_dir"],
            "output_dir": context["downstream_dir"],
            "real_split": split,
            "overwrite": True,
        }
    return {}


def _default_expected_outputs(
    stage_name: str,
    context: dict[str, str],
) -> tuple[Path, ...]:
    values = {
        "predictions": (context["prediction_manifest"],),
        "controller": (context["failure_scores"],),
        "layouts": (
            f"{context['layouts_dir']}/metadata.jsonl",
            f"{context['layouts_dir']}/layout_prior.json",
        ),
        "rendering": (
            f"{context['rendered_dir']}/metadata.jsonl",
            f"{context['rendered_dir']}/generation_config.json",
            f"{context['rendered_dir']}/images",
        ),
        "downstream": (
            f"{context['downstream_dir']}/training_manifest.jsonl",
            f"{context['downstream_dir']}/dataset_summary.json",
        ),
    }
    return tuple(Path(value) for value in values.get(stage_name, ()))


def _stage_cwd(
    value: object | None,
    default: Path,
    config_dir: Path,
) -> Path:
    if value is None:
        return default
    return _absolute_path(value, config_dir)


def _expected_paths(
    configured: object,
    defaults: tuple[Path, ...],
    context: dict[str, str],
    round_dir: Path,
) -> tuple[Path, ...]:
    if configured is None:
        return defaults
    expanded = _expand(configured, context)
    if not isinstance(expanded, list):
        raise TypeError("expected_outputs must be a list")
    paths = list(defaults)
    for value in expanded:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = round_dir / path
        paths.append(path.resolve())
    return tuple(dict.fromkeys(paths))


def build_iteration_plan(config_path: str | Path) -> IterationPlan:
    """Resolve one YAML iteration configuration into executable stage commands."""
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Iteration configuration must be a YAML mapping")
    config_dir = path.parent
    repo_root = Path(__file__).resolve().parents[2]
    round_index = int(config.get("round_index", 0))
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    if "real_dataset" not in config:
        raise KeyError("Iteration configuration requires real_dataset")
    real_dataset = _absolute_path(config["real_dataset"], config_dir)
    output_root = _absolute_path(config.get("output_root", "outputs/iterations"), config_dir)
    if config.get("round_dir") is None:
        round_dir = output_root / f"round_{round_index:03d}"
    else:
        round_dir = _absolute_path(config["round_dir"], config_dir)
    context = {
        "config_dir": str(config_dir),
        "repo_root": str(repo_root),
        "round_index": str(round_index),
        "round_dir": str(round_dir),
        "real_dataset": str(real_dataset),
        "raw_predictions_dir": str(round_dir / "raw_predictions"),
        "predictions_dir": str(round_dir / "predictions"),
        "prediction_manifest": str(round_dir / "predictions" / "predictions.jsonl"),
        "controller_dir": str(round_dir / "controller"),
        "failure_scores": str(round_dir / "controller" / "failure_scores.csv"),
        "layouts_dir": str(round_dir / "layouts"),
        "rendered_dir": str(round_dir / "rendered"),
        "downstream_dir": str(round_dir / "downstream"),
    }
    configured_stages = config.get("stages", {})
    if not isinstance(configured_stages, dict):
        raise TypeError("stages must be a YAML mapping")
    enabled_stages = {
        name
        for name in STAGE_ORDER
        if isinstance(configured_stages.get(name), dict)
        and bool(configured_stages[name].get("enabled", True))
    }
    split = str(config.get("split", "train"))
    stages = []
    for stage_name in STAGE_ORDER:
        stage_config = configured_stages.get(stage_name)
        if stage_name not in enabled_stages:
            continue
        if not isinstance(stage_config, dict):
            raise TypeError(f"Stage {stage_name!r} must be a YAML mapping")
        expanded_config = _expand(stage_config, context)
        environment_value = expanded_config.get("env", {})
        if not isinstance(environment_value, dict):
            raise TypeError(f"Stage {stage_name!r} env must be a mapping")
        environment = {str(key): str(value) for key, value in environment_value.items()}
        if expanded_config.get("command") is not None:
            command = _command_tokens(expanded_config["command"])
            default_cwd = config_dir
        else:
            if stage_name not in BUILTIN_SCRIPTS:
                raise KeyError(f"Stage {stage_name!r} requires command")
            arguments = expanded_config.get("args", {})
            if not isinstance(arguments, dict):
                raise TypeError(f"Stage {stage_name!r} args must be a mapping")
            arguments = dict(arguments)
            arguments.update(
                _stage_defaults(
                    stage_name,
                    context,
                    round_index,
                    split,
                    enabled_stages,
                )
            )
            script_path = repo_root / "scripts" / BUILTIN_SCRIPTS[stage_name]
            command = (sys.executable, str(script_path), *_args_to_cli(arguments))
            default_cwd = repo_root
        stages.append(
            StagePlan(
                name=stage_name,
                command=command,
                cwd=_stage_cwd(expanded_config.get("cwd"), default_cwd, config_dir),
                environment=environment,
                expected_outputs=_expected_paths(
                    expanded_config.get("expected_outputs"),
                    _default_expected_outputs(stage_name, context),
                    context,
                    round_dir,
                ),
            )
        )
    if not stages:
        raise ValueError("No enabled stages were configured")
    return IterationPlan(
        config_path=path,
        round_index=round_index,
        round_dir=round_dir,
        state_path=round_dir / "workflow_state.json",
        context=context,
        stages=tuple(stages),
    )


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_state(plan: IterationPlan) -> dict[str, object]:
    if plan.state_path.is_file():
        with plan.state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, dict) or not isinstance(state.get("stages"), dict):
            raise TypeError(f"Invalid workflow state: {plan.state_path}")
        return state
    return {
        "config_path": str(plan.config_path),
        "round_index": plan.round_index,
        "round_dir": str(plan.round_dir),
        "created_at": _now(),
        "stages": {},
    }


def _outputs_exist(stage: StagePlan) -> bool:
    return all(path.exists() for path in stage.expected_outputs)


def _selected_stages(
    plan: IterationPlan,
    stop_after: str | None,
) -> tuple[StagePlan, ...]:
    if stop_after is None:
        return plan.stages
    if stop_after not in STAGE_ORDER:
        raise ValueError(f"Unknown stage: {stop_after}")
    cutoff = STAGE_ORDER.index(stop_after)
    return tuple(
        stage for stage in plan.stages if STAGE_ORDER.index(stage.name) <= cutoff
    )


def run_iteration(
    plan: IterationPlan,
    resume: bool = False,
    stop_after: str | None = None,
) -> dict[str, object]:
    """Execute an iteration plan and persist resumable stage state and logs."""
    if plan.round_dir.exists() and any(plan.round_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Round directory is not empty: {plan.round_dir}; use --resume"
        )
    plan.round_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = plan.round_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    state = _load_state(plan)
    stage_state = state["stages"]
    state["status"] = "running"
    state["updated_at"] = _now()
    _write_json_atomic(plan.state_path, state)
    for stage in _selected_stages(plan, stop_after):
        previous = stage_state.get(stage.name, {})
        if (
            resume
            and previous.get("status") == "completed"
            and _outputs_exist(stage)
        ):
            print(f"[{stage.name}] already completed; skipping")
            continue
        log_path = logs_dir / f"{stage.name}.log"
        stage_state[stage.name] = {
            "status": "running",
            "started_at": _now(),
            "command": list(stage.command),
            "cwd": str(stage.cwd),
            "log": str(log_path),
            "expected_outputs": [str(path) for path in stage.expected_outputs],
        }
        state["updated_at"] = _now()
        _write_json_atomic(plan.state_path, state)
        environment = os.environ.copy()
        environment.update(stage.environment)
        print(f"[{stage.name}] {shlex.join(stage.command)}")
        process = None
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    stage.command,
                    cwd=stage.cwd,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                if process.stdout is None:
                    raise RuntimeError("Failed to capture stage output")
                for line in process.stdout:
                    print(f"[{stage.name}] {line}", end="")
                    log_handle.write(line)
                    log_handle.flush()
                return_code = process.wait()
        except BaseException as error:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            stage_state[stage.name].update(
                {
                    "status": "interrupted",
                    "finished_at": _now(),
                    "error": repr(error),
                }
            )
            state["status"] = "interrupted"
            state["updated_at"] = _now()
            _write_json_atomic(plan.state_path, state)
            raise
        if return_code != 0:
            stage_state[stage.name].update(
                {
                    "status": "failed",
                    "finished_at": _now(),
                    "return_code": return_code,
                }
            )
            state["status"] = "failed"
            state["updated_at"] = _now()
            _write_json_atomic(plan.state_path, state)
            raise RuntimeError(
                f"Stage {stage.name!r} failed with exit code {return_code}; "
                f"see {log_path}"
            )
        missing = [path for path in stage.expected_outputs if not path.exists()]
        if missing:
            stage_state[stage.name].update(
                {
                    "status": "failed",
                    "finished_at": _now(),
                    "return_code": return_code,
                    "missing_outputs": [str(path) for path in missing],
                }
            )
            state["status"] = "failed"
            state["updated_at"] = _now()
            _write_json_atomic(plan.state_path, state)
            raise FileNotFoundError(
                f"Stage {stage.name!r} did not produce: "
                + ", ".join(str(path) for path in missing)
            )
        stage_state[stage.name].update(
            {
                "status": "completed",
                "finished_at": _now(),
                "return_code": return_code,
            }
        )
        state["updated_at"] = _now()
        _write_json_atomic(plan.state_path, state)
    state["status"] = "completed" if stop_after is None else f"stopped_after_{stop_after}"
    state["updated_at"] = _now()
    _write_json_atomic(plan.state_path, state)
    return state
