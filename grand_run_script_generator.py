"""Unified Hyrax run script/config generator for Delta SLURM runs.

This file replaces the notebook-only run generators with one reusable script.
Edit the "Configured run" section near the bottom for your usual workflow, or
call this file from the command line.

Common commands:
    HYRAX_PROFILE=delta python grand_run_script_generator.py list-models
    HYRAX_PROFILE=delta python grand_run_script_generator.py print-plan
    HYRAX_PROFILE=delta python grand_run_script_generator.py write-training
    HYRAX_PROFILE=delta python grand_run_script_generator.py submit-training
    HYRAX_PROFILE=delta python grand_run_script_generator.py write-infer --experiments 1-8
    HYRAX_PROFILE=delta python grand_run_script_generator.py submit-infer --experiments 1-8

Experiment numbers are per run. If run1 is HyraxAutoencoderV2 and run2 is
SimCLR, both can use experiment suffixes _1, _2, ... inside their own run
directories. The generated jobs submit with sbatch; SLURM then runs them in
parallel.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

try:
    import tomlkit
except ModuleNotFoundError:  # Keep the generator usable outside the Hyrax env.
    tomlkit = None
    import tomllib

try:
    from research_paths import load_paths
except Exception:  # pragma: no cover - lets the script fail later with a clearer path fallback.
    load_paths = None


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent
DEFAULT_HYRAX_CONFIG_PATH = WORKSPACE_ROOT / "hyrax" / "src" / "hyrax" / "hyrax_default_config.toml"
DEFAULT_MODEL_SOURCE_DIR = WORKSPACE_ROOT / "hyrax" / "src" / "hyrax" / "models"
DEFAULT_FILTERS = ["g", "r", "i", "z", "y"]
AUTOENCODER_LIKE_MODELS = {
    "HyraxAutoencoder",
    "HyraxAutoencoderV2",
    "ImageDCAE",
    "HSCAutoencoder",
    "HSCDCAE",
}
MODEL_CONFIG_FALLBACKS = {
    # HSCDCAE currently reads this key directly from config["model"].
    "HSCDCAE": {"HSCDCAE_final_layer": "tanh"},
}


def _load_toml(path: str | Path):
    path = Path(path)
    if tomlkit is not None:
        with path.open("r", encoding="utf-8") as handle:
            return tomlkit.load(handle)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _write_toml(path: str | Path, document: Mapping[str, Any]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        if tomlkit is not None:
            handle.write(tomlkit.dumps(document))
        else:
            handle.write(_dump_toml(document))


def _toml_key(key: str) -> str:
    if re.match(r"^[A-Za-z0-9_-]+$", key):
        return key
    return _toml_value(key)


def _toml_table_path(parts: Sequence[str]) -> str:
    return ".".join(_toml_key(part) for part in parts)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Cannot render TOML value {value!r} ({type(value).__name__})")


def _dump_toml(document: Mapping[str, Any]) -> str:
    """Small TOML writer for the dicts this generator emits when tomlkit is absent."""
    lines: list[str] = []

    def write_table(table: Mapping[str, Any], prefix: list[str]) -> None:
        scalar_items = [(key, value) for key, value in table.items() if not isinstance(value, Mapping)]
        child_items = [(key, value) for key, value in table.items() if isinstance(value, Mapping)]

        if prefix:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"[{_toml_table_path(prefix)}]")
        for key, value in scalar_items:
            if value is None:
                continue
            lines.append(f"{_toml_key(str(key))} = {_toml_value(value)}")
        for key, value in child_items:
            write_table(value, [*prefix, str(key)])

    write_table(document, [])
    return "\n".join(lines).rstrip() + "\n"


def _to_plain(value: Any) -> Any:
    """Convert tomlkit containers to plain Python containers."""
    if hasattr(value, "unwrap"):
        value = value.unwrap()
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recursively merge updates into base and return base."""
    for key, value in (updates or {}).items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _with_trailing_slash(path: str | Path) -> str:
    text = str(path)
    return text if text.endswith("/") else f"{text}/"


def _iter_dataset_wrappers(data_request_section: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(data_request_section, Mapping):
        return
    for dataset_wrapper in data_request_section.values():
        if isinstance(dataset_wrapper, Mapping) and "data" in dataset_wrapper:
            yield dataset_wrapper


def _normalize_number_sequence(numbers: Iterable[int | str]) -> list[int]:
    return [int(number) for number in numbers]


def parse_number_spec(spec: str | Sequence[int] | None, default: Sequence[int] | None = None) -> list[int]:
    """Parse strings like '1-8,11,101-108' into a sorted unique list."""
    if spec is None:
        return list(default or [])
    if not isinstance(spec, str):
        return sorted(set(_normalize_number_sequence(spec)))

    values: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.update(range(int(start), int(end) + 1))
        else:
            values.add(int(part))
    return sorted(values)


def optimizer_config(optimizer: str, lr: float, momentum: bool | float = False, **params: Any) -> dict[str, Any]:
    """Create one optimizer spec in the format expected by the TOML writer."""
    config = {"optimizer": optimizer, "lr": lr}
    if momentum is not False:
        config["momentum"] = 0.9 if momentum is True else momentum
    config.update(params)
    return config


OPTIMIZER_GRID_V2 = {
    1: optimizer_config("torch.optim.SGD", 0.01, momentum=True),
    2: optimizer_config("torch.optim.SGD", 0.1, momentum=True),
    3: optimizer_config("torch.optim.SGD", 0.001, momentum=True),
    4: optimizer_config("torch.optim.SGD", 0.0001, momentum=True),
    5: optimizer_config("torch.optim.Adam", 0.01),
    6: optimizer_config("torch.optim.Adam", 0.1),
    7: optimizer_config("torch.optim.Adam", 0.001),
    8: optimizer_config("torch.optim.Adam", 0.0001),
}


def experiment_spec(
    description: str,
    optimizer_config: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    baseline_experiment: int | None = None,
) -> dict[str, Any]:
    return {
        "description": description,
        "optimizer_config": deepcopy(dict(optimizer_config or {})),
        "overrides": deepcopy(dict(overrides or {})),
        "baseline_experiment": baseline_experiment,
    }


def model_group(
    model_name: str,
    start_number: int,
    optimizer_grid: Mapping[int, Mapping[str, Any]] | None = None,
    filters: Sequence[str] | None = None,
    model_config_overrides: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    include_autoencoder_variants: bool = False,
    baseline_experiment_number: int = 7,
) -> dict[str, Any]:
    """Define a set of experiments for one model.

    Each group creates one optimizer sweep. Set include_autoencoder_variants for
    the old HyraxAutoencoderV2-style one-variable variants.
    """
    return {
        "model_name": model_name,
        "start_number": int(start_number),
        "optimizer_grid": deepcopy(dict(optimizer_grid or OPTIMIZER_GRID_V2)),
        "filters": list(filters) if filters is not None else None,
        "model_config_overrides": deepcopy(dict(model_config_overrides or {})),
        "overrides": deepcopy(dict(overrides or {})),
        "include_autoencoder_variants": bool(include_autoencoder_variants),
        "baseline_experiment_number": int(baseline_experiment_number),
    }


def get_research_paths(profile: str | None = None):
    if load_paths is None:
        raise RuntimeError("Could not import research_paths.load_paths. Run this from Hyrax-Research.")
    return load_paths(profile=profile)


def default_path_config(profile: str | None = None) -> dict[str, Any]:
    """Resolve local or Delta paths through research_paths.py."""
    paths = get_research_paths(profile)
    return {
        "profile": paths.profile,
        "base_directory": paths.hyrax_runs,
        "data_dir": paths.split_images_120,
        "results_dir": _with_trailing_slash(paths.hyrax_results),
        "filter_catalog": paths.catalog("raw_merger_flags"),
    }


def load_default_hyrax_config(default_config_path: str | Path = DEFAULT_HYRAX_CONFIG_PATH) -> dict[str, Any]:
    return _to_plain(_load_toml(default_config_path))


def discover_builtin_model_names(
    default_config_path: str | Path = DEFAULT_HYRAX_CONFIG_PATH,
    model_source_dir: str | Path = DEFAULT_MODEL_SOURCE_DIR,
) -> list[str]:
    """Find model names from the default config and @hyrax_model source classes."""
    names: set[str] = set()
    default_config_path = Path(default_config_path)
    if default_config_path.exists():
        model_section = load_default_hyrax_config(default_config_path).get("model", {})
        names.update(key for key in model_section if key != "name")

    model_source_dir = Path(model_source_dir)
    if model_source_dir.exists():
        class_pattern = re.compile(r"@hyrax_model\s+class\s+([A-Za-z_][A-Za-z0-9_]*)\(", re.MULTILINE)
        for source_path in sorted(model_source_dir.glob("*.py")):
            names.update(class_pattern.findall(source_path.read_text(encoding="utf-8")))

    return sorted(names)


def default_model_config(
    model_name: str,
    default_config_path: str | Path = DEFAULT_HYRAX_CONFIG_PATH,
) -> dict[str, Any]:
    model_section = load_default_hyrax_config(default_config_path).get("model", {})
    if model_name in model_section:
        return {model_name: deepcopy(model_section[model_name])}
    return deepcopy(MODEL_CONFIG_FALLBACKS.get(model_name, {}))


def model_default_keys(model_name: str) -> set[str]:
    config = default_model_config(model_name).get(model_name, {})
    return set(config) if isinstance(config, Mapping) else set()


def default_criterion_config(model_name: str) -> dict[str, Any]:
    if model_name == "SimCLR" or model_name == "HyraxLoopback":
        return {
            # SimCLR defines NTXentLoss internally from model.SimCLR.temperature.
            "name": "",
            "band_loss_reduction": "mean",
        }
    if model_name in AUTOENCODER_LIKE_MODELS or "Autoencoder" in model_name:
        return {"name": "torch.nn.MSELoss", "band_loss_reduction": "mean"}
    return {"name": "torch.nn.CrossEntropyLoss", "band_loss_reduction": "mean"}


def build_optimizer_sweep_specs(
    model_name: str,
    start_number: int = 1,
    optimizer_grid: Mapping[int, Mapping[str, Any]] | None = None,
    filters: Sequence[str] | None = None,
    model_config_overrides: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    """Build a model-specific optimizer sweep with arbitrary numbering."""
    optimizer_grid = optimizer_grid or OPTIMIZER_GRID_V2
    base_overrides = deepcopy(dict(overrides or {}))
    base_overrides["model_name"] = model_name
    if filters is not None:
        base_overrides["filters"] = list(filters)
    if model_config_overrides:
        base_overrides["model_config_overrides"] = {model_name: deepcopy(dict(model_config_overrides))}

    specs: dict[int, dict[str, Any]] = {}
    for offset, (optimizer_idx, cfg) in enumerate(sorted(optimizer_grid.items())):
        experiment_number = int(start_number) + offset
        optimizer_label = str(cfg["optimizer"]).split(".")[-1]
        specs[experiment_number] = experiment_spec(
            description=f"{model_name} {optimizer_label} lr={cfg['lr']} (optimizer grid _{optimizer_idx})",
            optimizer_config=cfg,
            overrides=base_overrides,
        )
    return specs


def build_autoencoder_variant_specs(
    model_name: str = "HyraxAutoencoderV2",
    start_number: int = 1,
    baseline_experiment_number: int = 7,
    optimizer_grid: Mapping[int, Mapping[str, Any]] | None = None,
    filters: Sequence[str] | None = None,
    model_config_overrides: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    """Build the old 1-18 style optimizer grid plus one-variable variants."""
    optimizer_grid = optimizer_grid or OPTIMIZER_GRID_V2
    specs = build_optimizer_sweep_specs(
        model_name=model_name,
        start_number=start_number,
        optimizer_grid=optimizer_grid,
        filters=filters,
        model_config_overrides=model_config_overrides,
        overrides=overrides,
    )
    baseline_number = int(start_number) + int(baseline_experiment_number) - 1
    if baseline_number not in specs:
        raise KeyError(f"Baseline experiment _{baseline_number} is not defined")

    baseline_optimizer = deepcopy(specs[baseline_number]["optimizer_config"])
    baseline_label = f"baseline _{baseline_number}"
    config_keys = model_default_keys(model_name)

    next_number = int(start_number) + len(optimizer_grid) - 1
    variant_specs: list[tuple[str, dict[str, Any]]] = [
        ("batch_size=512", {"batch_size": 512}),
        ("batch_size=128", {"batch_size": 128}),
    ]
    if "final_layer" in config_keys:
        variant_specs.append(
            (
                "final_layer=arcsinh and transform=arcsinh",
                {"model_config_overrides": {model_name: {"final_layer": "arcsinh"}}, "transform": "arcsinh"},
            )
        )
    variant_specs.append(("filters=[g,r,i,z]", {"filters": ["g", "r", "i", "z"]}))
    if "latent_dim" in config_keys:
        for latent_dim in (128, 256, 1024):
            variant_specs.append(
                (f"latent_dim={latent_dim}", {"model_config_overrides": {model_name: {"latent_dim": latent_dim}}})
            )
    if "base_channel_size" in config_keys:
        for base_channel_size in (16, 64):
            variant_specs.append(
                (
                    f"base_channel_size={base_channel_size}",
                    {"model_config_overrides": {model_name: {"base_channel_size": base_channel_size}}},
                )
            )
    variant_specs.append(("crop_to=[100,100]", {"crop_to": [100, 100]}))

    for label, variant_overrides in variant_specs:
        next_number += 1
        merged_overrides = deepcopy(dict(overrides or {}))
        merged_overrides["model_name"] = model_name
        if filters is not None:
            merged_overrides["filters"] = list(filters)
        if model_config_overrides:
            merged_overrides["model_config_overrides"] = {model_name: deepcopy(dict(model_config_overrides))}
        _deep_update(merged_overrides, variant_overrides)
        specs[next_number] = experiment_spec(
            description=f"{baseline_label} with {label}",
            optimizer_config=baseline_optimizer,
            overrides=merged_overrides,
            baseline_experiment=baseline_number,
        )

    return dict(sorted(specs.items()))


def build_extra_experiment_specs(
    extra_experiments: Mapping[int, Mapping[str, Any]] | None,
    existing_specs: Mapping[int, Mapping[str, Any]],
    default_baseline_experiment: int,
) -> dict[int, dict[str, Any]]:
    """Build custom experiments, inheriting optimizer and overrides from a baseline."""
    if not extra_experiments:
        return {}

    if default_baseline_experiment not in existing_specs:
        raise KeyError(f"Default baseline experiment _{default_baseline_experiment} is not defined")

    default_optimizer = existing_specs[default_baseline_experiment]["optimizer_config"]
    built: dict[int, dict[str, Any]] = {}
    for idx, spec in extra_experiments.items():
        idx = int(idx)
        spec = deepcopy(dict(spec))
        baseline_idx = spec.get("baseline_experiment")
        baseline_spec = None
        if baseline_idx is not None:
            baseline_idx = int(baseline_idx)
            if baseline_idx not in existing_specs:
                raise KeyError(f"Baseline experiment _{baseline_idx} is not defined")
            baseline_spec = existing_specs[baseline_idx]

        optimizer = spec.get("optimizer_config")
        if optimizer is None and baseline_spec is not None:
            optimizer = baseline_spec.get("optimizer_config")
        if optimizer is None:
            optimizer = default_optimizer

        merged_overrides = deepcopy(baseline_spec.get("overrides", {})) if baseline_spec else {}
        _deep_update(merged_overrides, spec.get("overrides", {}))

        built[idx] = experiment_spec(
            description=spec.get("description", f"custom experiment {idx}"),
            optimizer_config=optimizer,
            overrides=merged_overrides,
            baseline_experiment=baseline_idx,
        )
    return built


def build_specs_from_model_groups(groups: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    specs: dict[int, dict[str, Any]] = {}
    for group in groups:
        builder = build_autoencoder_variant_specs if group.get("include_autoencoder_variants") else build_optimizer_sweep_specs
        group_specs = builder(
            model_name=group["model_name"],
            start_number=group.get("start_number", 1),
            optimizer_grid=group.get("optimizer_grid"),
            filters=group.get("filters"),
            model_config_overrides=group.get("model_config_overrides"),
            overrides=group.get("overrides"),
            **(
                {"baseline_experiment_number": group.get("baseline_experiment_number", 7)}
                if builder is build_autoencoder_variant_specs
                else {}
            ),
        )
        overlap = set(specs).intersection(group_specs)
        if overlap:
            raise ValueError(f"Duplicate experiment numbers from model groups: {sorted(overlap)}")
        specs.update(group_specs)
    return dict(sorted(specs.items()))


@dataclass
class RunPlan:
    run_number: int
    base_directory: Path
    data_dir: Path
    results_dir: str
    filter_catalog: Path
    model_name: str = "HyraxAutoencoderV2"
    model_config: dict[str, Any] | None = None
    batch_size: int = 256
    epochs: int = 20
    crop_to: list[int] = field(default_factory=lambda: [120, 120])
    dataset_class: str = "FitsImageDataSet"
    object_id_column_name: str = "object_id"
    filters: list[str] = field(default_factory=lambda: list(DEFAULT_FILTERS))
    transform: str = "tanh"
    data_fields: list[str] = field(default_factory=lambda: ["image"])
    primary_id_field: str = "object_id"
    data_set: dict[str, Any] = field(
        default_factory=lambda: {
            "use_cache": True,
            "preload_cache": True,
            "seed": 1,
            "train_size": 0.8,
            "validate_size": 0.1,
            "test_size": 0.1,
        }
    )
    slurm: dict[str, Any] = field(
        default_factory=lambda: {
            "account": "bemi-delta-gpu",
            "partition": "gpuA40x4",
            "nodes": 1,
            "cpus-per-gpu": 5,
            "mem": "50G",
            "gpus": 1,
            "time": "1:00:00",
        }
    )
    shell_preamble: list[str] = field(default_factory=lambda: ["set -euo pipefail"])
    setup_lines: list[str] = field(
        default_factory=lambda: [
            "# source /mmfs1/home/aritrag/.bashrc",
            "# conda activate hyrax",
        ]
    )

    @property
    def run_dir(self) -> Path:
        return Path(self.base_directory) / f"run{self.run_number}"


def print_experiment_specs(experiment_specs: Mapping[int, Mapping[str, Any]]) -> None:
    for idx, spec in sorted(experiment_specs.items()):
        baseline = spec.get("baseline_experiment")
        baseline_text = f" baseline=_{baseline}" if baseline else ""
        model_name = spec.get("overrides", {}).get("model_name")
        model_text = f" [{model_name}]" if model_name else ""
        print(f"_{idx}{model_text}: {spec['description']}{baseline_text}")


def _resolve_model_config(
    model_name: str,
    base_model_config: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    model_config = deepcopy(dict(base_model_config or default_model_config(model_name)))
    return _deep_update(model_config, overrides or {})


def _resolve_optimizer_table(optimizer_config_value: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    optimizer_config_value = dict(optimizer_config_value)
    optimizer_name = str(optimizer_config_value.pop("optimizer"))
    momentum = optimizer_config_value.pop("momentum", None)
    params = dict(optimizer_config_value)
    if momentum is not None and "momentum" not in params:
        params["momentum"] = 0.9 if momentum is True else momentum
    return optimizer_name, params


def _build_data_request(plan: RunPlan, data_dir: str | Path, dataset_class: str, fields: Sequence[str], primary_id: str):
    return {
        "train": {
            "data": {
                "dataset_class": dataset_class,
                "data_location": str(data_dir),
                "fields": list(fields),
                "primary_id_field": primary_id,
            }
        },
        "infer": {
            "data": {
                "dataset_class": dataset_class,
                "data_location": str(data_dir),
                "fields": list(fields),
                "primary_id_field": primary_id,
            }
        },
    }


def create_training_toml(
    plan: RunPlan,
    file_number: int,
    spec: Mapping[str, Any],
) -> Path:
    """Create one Hyrax runtime config for training."""
    overrides = deepcopy(dict(spec.get("overrides", {})))
    model_name = overrides.get("model_name", plan.model_name)
    model_config = _resolve_model_config(
        model_name=model_name,
        base_model_config=plan.model_config if model_name == plan.model_name else None,
        overrides=overrides.get("model_config_overrides"),
    )
    optimizer_name, optimizer_params = _resolve_optimizer_table(spec["optimizer_config"])

    data_dir = overrides.get("data_dir", plan.data_dir)
    results_dir = overrides.get("results_dir", plan.results_dir)
    filter_catalog = overrides.get("filter_catalog", plan.filter_catalog)
    dataset_class = overrides.get("dataset_class", plan.dataset_class)
    object_id_column_name = overrides.get("object_id_column_name", plan.object_id_column_name)
    data_fields = overrides.get("data_fields", plan.data_fields)
    primary_id_field = overrides.get("primary_id_field", plan.primary_id_field)
    filters = overrides.get("filters", plan.filters)
    crop_to = overrides.get("crop_to", plan.crop_to)
    transform = overrides.get("transform", plan.transform)

    document: dict[str, Any] = {}
    document["general"] = {
        "dev_mode": False,
        "log_level": "debug",
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
    }
    _deep_update(document["general"], overrides.get("general"))

    document["model"] = {"name": model_name}
    _deep_update(document["model"], model_config)

    document["criterion"] = deepcopy(overrides.get("criterion", default_criterion_config(model_name)))
    document["optimizer"] = {"name": optimizer_name}
    document[optimizer_name] = optimizer_params

    if "scheduler" in overrides:
        document["scheduler"] = deepcopy(overrides["scheduler"])
    if "scheduler_config" in overrides:
        scheduler_name = document.get("scheduler", {}).get("name")
        if scheduler_name:
            document[scheduler_name] = deepcopy(overrides["scheduler_config"])

    document["train"] = {
        "weights_filename": overrides.get("weights_filename", "example_model.pth"),
        "epochs": overrides.get("epochs", plan.epochs),
        "resume": overrides.get("resume", False),
        "split": overrides.get("split", "train"),
        "experiment_name": overrides.get("experiment_name", f"run{plan.run_number}"),
        "run_name": overrides.get("run_name", f"run{plan.run_number}_{file_number}"),
    }
    _deep_update(document["train"], overrides.get("train"))

    document["data_request"] = _build_data_request(
        plan=plan,
        data_dir=data_dir,
        dataset_class=dataset_class,
        fields=data_fields,
        primary_id=primary_id_field,
    )
    _deep_update(document["data_request"], overrides.get("data_request"))

    document["data_set"] = {
        "name": dataset_class,
        "object_id_column_name": object_id_column_name,
        "filter_catalog": str(filter_catalog),
        "filters": list(filters),
        "transform": transform,
        "crop_to": list(crop_to),
    }
    _deep_update(document["data_set"], deepcopy(plan.data_set))
    _deep_update(document["data_set"], overrides.get("data_set"))

    document["data_loader"] = {"batch_size": overrides.get("batch_size", plan.batch_size)}
    _deep_update(document["data_loader"], overrides.get("data_loader"))

    _deep_update(document, overrides.get("toml_overrides"))

    output_path = plan.run_dir / f"train{plan.run_number}_{file_number}.toml"
    _write_toml(output_path, document)
    print(f"Created {output_path}")
    return output_path


def _slurm_flag(key: str) -> str:
    return key if key.startswith("--") else f"--{key.replace('_', '-')}"


def render_slurm_script(
    job_name: str,
    output_file: str,
    commands: Sequence[str],
    slurm: Mapping[str, Any],
    shell_preamble: Sequence[str] | None = None,
    setup_lines: Sequence[str] | None = None,
) -> str:
    directives = deepcopy(dict(slurm))
    directives["job-name"] = job_name
    directives["output"] = output_file

    lines = ["#!/bin/bash", "#"]
    for key, value in directives.items():
        if value is None or value is False:
            continue
        flag = _slurm_flag(str(key))
        if value is True:
            lines.append(f"#SBATCH {flag}")
        else:
            lines.append(f"#SBATCH {flag}={value}")
    lines.append("")
    for line in shell_preamble or []:
        lines.append(str(line))
    for line in setup_lines or []:
        lines.append(str(line))
    if shell_preamble or setup_lines:
        lines.append("")
    lines.extend(str(command) for command in commands)
    return "\n".join(lines).rstrip() + "\n"


def create_job_file(
    plan: RunPlan,
    prefix: str,
    job_name: str,
    file_number: int,
    commands: Sequence[str],
    slurm_overrides: Mapping[str, Any] | None = None,
) -> Path:
    slurm = deepcopy(plan.slurm)
    _deep_update(slurm, slurm_overrides)
    output_file = f"{prefix}{plan.run_number}_{file_number}.txt"
    content = render_slurm_script(
        job_name=job_name,
        output_file=output_file,
        commands=commands,
        slurm=slurm,
        shell_preamble=plan.shell_preamble,
        setup_lines=plan.setup_lines,
    )
    output_path = plan.run_dir / f"{prefix}{plan.run_number}_{file_number}.sh"
    output_path.write_text(content, encoding="utf-8")
    print(f"Created {output_path}")
    return output_path


def create_training_files(
    plan: RunPlan,
    experiment_specs: Mapping[int, Mapping[str, Any]],
    experiment_numbers: Sequence[int] | None = None,
) -> list[tuple[Path, Path]]:
    """Create training TOMLs and SLURM job files for selected experiments."""
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    numbers = list(experiment_numbers or sorted(experiment_specs))
    created: list[tuple[Path, Path]] = []
    for file_number in numbers:
        if file_number not in experiment_specs:
            raise KeyError(f"Experiment _{file_number} is not defined in experiment_specs")
        spec = experiment_specs[file_number]
        toml_path = create_training_toml(plan, file_number, spec)
        job_path = create_job_file(
            plan=plan,
            prefix="train",
            job_name=f"t{plan.run_number}_{file_number}",
            file_number=file_number,
            commands=[f"hyrax train --runtime-config={toml_path}"],
            slurm_overrides=spec.get("overrides", {}).get("slurm"),
        )
        created.append((toml_path, job_path))
    return created


def submit_jobs(
    run_number: int,
    job_numbers: Sequence[int],
    prefix: str,
    base_directory: str | Path,
    pause_after_first_seconds: int = 10,
    dry_run: bool = False,
) -> None:
    """Submit selected jobs with sbatch."""
    run_dir = Path(base_directory) / f"run{run_number}"
    for offset, number in enumerate(job_numbers):
        job_file = f"{prefix}{run_number}_{number}.sh"
        cmd = ["sbatch", job_file]
        print(f"Submitting: {' '.join(cmd)}")
        if not dry_run:
            subprocess.run(cmd, cwd=run_dir, check=True)
        if offset == 0 and pause_after_first_seconds:
            time.sleep(pause_after_first_seconds)


def _find_results_dir_by_run_name(results_root: str | Path, run_name: str, expected_suffix: str | None = None) -> Path | None:
    results_root = Path(results_root)
    if not results_root.exists():
        return None

    for candidate in sorted(results_root.iterdir(), reverse=True):
        if not candidate.is_dir():
            continue
        if expected_suffix and expected_suffix not in candidate.name:
            continue
        runtime_config = candidate / "runtime_config.toml"
        if not runtime_config.exists():
            continue
        try:
            config = _load_toml(runtime_config)
        except Exception:
            continue
        if config.get("train", {}).get("run_name") == run_name:
            return candidate
    return None


def extract_model_directory(train_output_file: str | Path) -> str:
    """Return the trained weights path, using log parsing first and run-name matching as fallback."""
    train_output_file = Path(train_output_file)
    if not train_output_file.exists():
        raise FileNotFoundError(f"Training output file not found: {train_output_file}")

    content = train_output_file.read_text(encoding="utf-8", errors="replace")
    patterns = [
        r"Latest checkpoint saved as: (.+)/checkpoint_epoch_\d+\.pt",
        r"Best metric checkpoint saved as: (.+)/checkpoint_[^/\s]+\.pt",
        r"Exported model to ONNX format: (.+)/[^/\s]+\.onnx",
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return str(Path(match.group(1)) / "example_model.pth")

    sibling_toml = train_output_file.with_suffix(".toml")
    if sibling_toml.exists():
        config = _load_toml(sibling_toml)
        results_root = config.get("general", {}).get("results_dir")
        run_name = config.get("train", {}).get("run_name")
        weights_filename = config.get("train", {}).get("weights_filename", "example_model.pth")
        if results_root and run_name:
            results_dir = _find_results_dir_by_run_name(results_root, run_name, expected_suffix="-train-")
            if results_dir is not None:
                candidate = results_dir / weights_filename
                if candidate.exists():
                    return str(candidate)

    raise ValueError(f"Could not determine model weights path from {train_output_file}")


def create_infer_scripts_batch(
    plan: RunPlan,
    experiment_numbers: Sequence[int],
) -> list[tuple[Path, Path]]:
    """Generate infer TOMLs/jobs from sibling train files and train logs."""
    run_dir = plan.run_dir
    created: list[tuple[Path, Path]] = []
    for number in experiment_numbers:
        train_name = f"train{plan.run_number}_{number}"
        infer_name = f"infer{plan.run_number}_{number}"
        train_toml = run_dir / f"{train_name}.toml"
        train_output = run_dir / f"{train_name}.txt"
        if not train_toml.exists():
            print(f"Warning: {train_toml} not found, skipping...")
            continue

        try:
            model_weights_path = extract_model_directory(train_output)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error processing {train_name}: {exc}")
            continue

        infer_toml = run_dir / f"{infer_name}.toml"
        config = _load_toml(train_toml)
        config["infer"] = {"model_weights_file": str(model_weights_path), "split": False}
        _write_toml(infer_toml, config)

        infer_job = create_job_file(
            plan=plan,
            prefix="infer",
            job_name=f"i{plan.run_number}_{number}",
            file_number=number,
            commands=[f"hyrax infer --runtime-config={infer_toml}"],
        )
        created.append((infer_toml, infer_job))
        print(f"Created infer scripts for {train_name}")
    return created


def extract_inference_directory(infer_output_file: str | Path) -> str:
    """Extract the inference results directory from infer output logs."""
    infer_output_file = Path(infer_output_file)
    if not infer_output_file.exists():
        raise FileNotFoundError(f"Inference output file not found: {infer_output_file}")

    content = infer_output_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Saving inference results at: (.+)", content)
    if match:
        return _with_trailing_slash(match.group(1).strip().rstrip("/"))
    raise ValueError(f"Could not find inference directory in {infer_output_file}")


def create_udb_scripts_batch(
    plan: RunPlan,
    experiment_numbers: Sequence[int],
    umap_config: Mapping[str, Any] | None = None,
    vector_db_config: Mapping[str, Any] | None = None,
) -> list[tuple[Path, Path]]:
    """Generate UDB TOMLs/jobs from infer artifacts."""
    run_dir = plan.run_dir
    created: list[tuple[Path, Path]] = []
    for number in experiment_numbers:
        infer_name = f"infer{plan.run_number}_{number}"
        udb_name = f"udb{plan.run_number}_{number}"
        infer_toml = run_dir / f"{infer_name}.toml"
        infer_output = run_dir / f"{infer_name}.txt"
        if not infer_toml.exists():
            print(f"Warning: {infer_toml} not found, skipping...")
            continue

        try:
            inference_dir = extract_inference_directory(infer_output)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error processing {infer_name}: {exc}")
            continue

        udb_toml = run_dir / f"{udb_name}.toml"
        config = _load_toml(infer_toml)
        config["results"] = {"inference_dir": inference_dir}
        config["vector_db"] = {"name": "chromadb", "infer_results_dir": inference_dir}
        _deep_update(config["vector_db"], vector_db_config)
        config["umap"] = {
            "fit_sample_size": 5000,
            "save_fit_umap": False,
            "parallel": True,
            "name": "umap.UMAP",
            "UMAP": {"n_components": 2, "n_neighbors": 15},
        }
        _deep_update(config["umap"], umap_config)
        _write_toml(udb_toml, config)

        input_dir = inference_dir.rstrip("/")
        udb_job = create_job_file(
            plan=plan,
            prefix="udb",
            job_name=f"u{plan.run_number}_{number}",
            file_number=number,
            commands=[
                f"hyrax umap --runtime-config={udb_toml} --input-dir={input_dir}/",
                f"hyrax save_to_database --runtime-config={udb_toml} --input-dir={input_dir}/",
            ],
        )
        created.append((udb_toml, udb_job))
        print(f"Created UDB scripts for {infer_name}")
    return created


def create_3dumap_scripts_batch(plan: RunPlan, experiment_numbers: Sequence[int]) -> list[tuple[Path, Path]]:
    """Generate 3D UMAP TOMLs/jobs from UDB artifacts."""
    run_dir = plan.run_dir
    created: list[tuple[Path, Path]] = []
    for number in experiment_numbers:
        udb_name = f"udb{plan.run_number}_{number}"
        dumap_name = f"3dumap{plan.run_number}_{number}"
        udb_toml = run_dir / f"{udb_name}.toml"
        if not udb_toml.exists():
            print(f"Warning: {udb_toml} not found, skipping...")
            continue

        dumap_toml = run_dir / f"{dumap_name}.toml"
        config = _load_toml(udb_toml)
        config.setdefault("umap", {})
        config["umap"].setdefault("UMAP", {})
        config["umap"]["UMAP"]["n_components"] = 3
        _write_toml(dumap_toml, config)

        inference_dir = config.get("vector_db", {}).get("infer_results_dir") or config.get("results", {}).get("inference_dir")
        if not inference_dir:
            print(f"Warning: could not find inference_dir in {udb_toml}, skipping job file...")
            continue

        input_dir = str(inference_dir).rstrip("/")
        dumap_job = create_job_file(
            plan=plan,
            prefix="3dumap",
            job_name=f"3u{plan.run_number}_{number}",
            file_number=number,
            commands=[f"hyrax umap --runtime-config={dumap_toml} --input-dir={input_dir}/"],
        )
        created.append((dumap_toml, dumap_job))
        print(f"Created 3D UMAP scripts for {udb_name}")
    return created


def extract_umap_directory(dumap_output_file: str | Path) -> str:
    """Extract the UMAP results directory from 3D UMAP output logs."""
    dumap_output_file = Path(dumap_output_file)
    if not dumap_output_file.exists():
        raise FileNotFoundError(f"3D UMAP output file not found: {dumap_output_file}")

    content = dumap_output_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Saving UMAP results to (.+)", content)
    if match:
        return match.group(1).strip()
    raise ValueError(f"Could not find UMAP directory in {dumap_output_file}")


def extract_catalog_settings(dumap_config_file: str | Path) -> tuple[str, bool]:
    """Extract the catalog path and whether duplicate object IDs should be collapsed."""
    dumap_config_file = Path(dumap_config_file)
    if not dumap_config_file.exists():
        raise FileNotFoundError(f"3D UMAP config file not found: {dumap_config_file}")

    config = _load_toml(dumap_config_file)
    filter_catalog = config.get("data_set", {}).get("filter_catalog")
    if filter_catalog:
        return str(filter_catalog), True

    legacy_path = config.get("data_set", {}).get("astropy_table")
    if legacy_path:
        return str(legacy_path), False

    for split_name in ("train", "infer", "validate"):
        split_config = config.get("data_request", {}).get(split_name)
        for dataset_wrapper in _iter_dataset_wrappers(split_config):
            dataset_config = dataset_wrapper["data"].get("dataset_config", {})
            if isinstance(dataset_config, Mapping):
                filter_catalog = dataset_config.get("filter_catalog")
                if filter_catalog:
                    return str(filter_catalog), True
                astropy_table = dataset_config.get("astropy_table")
                if astropy_table:
                    return str(astropy_table), False

    raise ValueError(f"Could not find filter_catalog or astropy_table in {dumap_config_file}")


def create_3d_viz_json(plan: RunPlan, experiment_number: int, id_column: str = "object_id") -> str:
    """Create a 3D visualization JSON file from 3D UMAP results."""
    try:
        save_umap_json = import_module("hyrax.3d_viz.save_umap_to_json").save_umap_json
    except Exception as exc:
        raise ImportError("Could not import save_umap_json from hyrax. Activate a Hyrax environment.") from exc

    run_dir = plan.run_dir
    dumap_name = f"3dumap{plan.run_number}_{experiment_number}"
    dumap_output = run_dir / f"{dumap_name}.txt"
    dumap_config = run_dir / f"{dumap_name}.toml"

    umap_results_dir = extract_umap_directory(dumap_output)
    fits_table_path, keep_first_match_only = extract_catalog_settings(dumap_config)
    viz_dir = Path(plan.base_directory) / "3d_viz_files"
    viz_dir.mkdir(exist_ok=True)
    output_json = viz_dir / f"umap{plan.run_number}_{experiment_number}.json"

    save_umap_json(
        results_dir=umap_results_dir,
        output_json=str(output_json),
        fits_table_path=fits_table_path,
        id_column=id_column,
        keep_first_match_only=keep_first_match_only,
    )
    print(f"Successfully created {output_json}")
    return str(output_json)


def create_3d_viz_json_batch(plan: RunPlan, experiment_numbers: Sequence[int], id_column: str = "object_id") -> list[str]:
    results = []
    failed = []
    for number in experiment_numbers:
        try:
            results.append(create_3d_viz_json(plan, number, id_column=id_column))
        except Exception as exc:
            print(f"Failed to process 3dumap{plan.run_number}_{number}: {exc}")
            failed.append(number)

    print("\n3D Visualization JSON Generation Summary:")
    print(f"  Successfully created: {len(results)} JSON files")
    print(f"  Failed: {len(failed)} files")
    if failed:
        print(f"  Failed job numbers: {failed}")
    return results


def copy_training_configs(
    source_dir: str | Path,
    target_dir: str | Path,
    old_run_num: int,
    new_run_num: int,
    new_data_dir: str | Path | None = None,
    new_filter_catalog_root: str | Path | None = None,
) -> None:
    """Copy generated training configs/jobs and update run identifiers and key paths."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    copied = 0
    for toml_file in sorted(source_path.glob(f"train{old_run_num}_*.toml")):
        suffix = toml_file.stem.split("_")[-1]
        new_toml_name = f"train{new_run_num}_{suffix}.toml"
        new_job_name = f"train{new_run_num}_{suffix}.sh"

        config = _load_toml(toml_file)
        config.setdefault("train", {})
        config["train"]["experiment_name"] = f"run{new_run_num}"
        config["train"]["run_name"] = f"run{new_run_num}_{suffix}"
        config["train"]["weights_filename"] = config["train"].get("weights_filename", "example_model.pth")
        config["general"]["log_level"] = "debug"

        if new_data_dir is not None:
            config["general"]["data_dir"] = str(new_data_dir)
            for split_name in ("train", "infer", "validate"):
                split_config = config.get("data_request", {}).get(split_name)
                for dataset_wrapper in _iter_dataset_wrappers(split_config):
                    dataset_wrapper["data"]["data_location"] = str(new_data_dir)

        if new_filter_catalog_root is not None and config.get("data_set", {}).get("filter_catalog"):
            current_name = Path(str(config["data_set"]["filter_catalog"])).name
            config["data_set"]["filter_catalog"] = str(Path(new_filter_catalog_root) / current_name)

        _write_toml(target_path / new_toml_name, config)

        job_file = source_path / f"train{old_run_num}_{suffix}.sh"
        if job_file.exists():
            job_content = job_file.read_text(encoding="utf-8")
            job_content = job_content.replace(f"--job-name=t{old_run_num}_{suffix}", f"--job-name=t{new_run_num}_{suffix}")
            job_content = job_content.replace(f"--output=train{old_run_num}_{suffix}.txt", f"--output=train{new_run_num}_{suffix}.txt")
            job_content = job_content.replace(f"run{old_run_num}/train{old_run_num}_{suffix}.toml", f"run{new_run_num}/train{new_run_num}_{suffix}.toml")
            job_content = job_content.replace(f"train{old_run_num}_{suffix}.toml", f"train{new_run_num}_{suffix}.toml")
            (target_path / new_job_name).write_text(job_content, encoding="utf-8")

        copied += 1

    print(f"Copied {copied} training configurations from run{old_run_num} to run{new_run_num}")


# ---------------------------------------------------------------------------
# Configured run
#
# Edit this section for the usual "one file where I can do everything" flow.
# The CLI uses these defaults unless flags override them.
# ---------------------------------------------------------------------------

RUN_NUMBER = 10
DEFAULT_MODEL_NAME = "HyraxAutoencoderV2"
BASELINE_EXPERIMENT_NUMBER = 7
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 20
DEFAULT_CROP_TO = [120, 120]

MODEL_GROUPS = [
    model_group(
        DEFAULT_MODEL_NAME,
        start_number=1,
        include_autoencoder_variants=DEFAULT_MODEL_NAME == "HyraxAutoencoderV2",
        baseline_experiment_number=BASELINE_EXPERIMENT_NUMBER,
    ),
]

EXTRA_EXPERIMENTS: dict[int, dict[str, Any]] = {
    # Example:
    # 109: {
    #     "description": "SimCLR _101 with epochs=40",
    #     "baseline_experiment": 101,
    #     "overrides": {"epochs": 40},
    # },
    # 201: {
    #     "description": "ImageDCAE Adam lr=0.001 with latent_dim=256",
    #     "optimizer_config": optimizer_config("torch.optim.Adam", 0.001),
    #     "overrides": {
    #         "model_name": "ImageDCAE",
    #         "model_config_overrides": {"ImageDCAE": {"latent_dim": 256}},
    #     },
    # },
}


def configured_experiment_specs(
    groups: Sequence[Mapping[str, Any]] | None = None,
    extra_experiments: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    specs = build_specs_from_model_groups(groups or MODEL_GROUPS)
    extras = build_extra_experiment_specs(
        extra_experiments if extra_experiments is not None else EXTRA_EXPERIMENTS,
        specs,
        default_baseline_experiment=BASELINE_EXPERIMENT_NUMBER,
    )
    specs.update(extras)
    return dict(sorted(specs.items()))


def configured_run_plan(
    profile: str | None = None,
    run_number: int | None = None,
    model_name: str | None = None,
    base_directory: str | Path | None = None,
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    filter_catalog: str | Path | None = None,
) -> RunPlan:
    paths = default_path_config(profile)
    resolved_model_name = model_name or DEFAULT_MODEL_NAME
    return RunPlan(
        run_number=int(run_number if run_number is not None else RUN_NUMBER),
        base_directory=Path(base_directory or paths["base_directory"]),
        data_dir=Path(data_dir or paths["data_dir"]),
        results_dir=_with_trailing_slash(results_dir or paths["results_dir"]),
        filter_catalog=Path(filter_catalog or paths["filter_catalog"]),
        model_name=resolved_model_name,
        model_config=default_model_config(resolved_model_name),
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
        crop_to=list(DEFAULT_CROP_TO),
    )


def _groups_from_models(models: Sequence[str] | None) -> list[dict[str, Any]] | None:
    if not models:
        return None
    if len(models) > 1:
        raise ValueError(
            "Experiment numbers are per run. Pass one model here, or use the notebook RUNS mapping "
            "to assign different models to different run numbers."
        )
    model_name = models[0]
    filters = ["g", "r", "i"] if model_name == "SimCLR" else None
    return [
        model_group(
            model_name,
            start_number=1,
            filters=filters,
            include_autoencoder_variants=model_name == "HyraxAutoencoderV2",
            baseline_experiment_number=BASELINE_EXPERIMENT_NUMBER,
        )
    ]


def _parse_models(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and submit Hyrax Delta SLURM scripts/configs.")
    parser.add_argument(
        "action",
        choices=[
            "list-models",
            "print-plan",
            "write-training",
            "submit-training",
            "write-infer",
            "submit-infer",
            "write-udb",
            "submit-udb",
            "write-3dumap",
            "submit-3dumap",
            "write-3d-viz-json",
        ],
    )
    parser.add_argument("--profile", choices=["local", "delta"], help="Path profile. Defaults to research_paths detection.")
    parser.add_argument("--run-number", type=int, help="Run number, e.g. 10.")
    parser.add_argument("--models", help="Model name for this run. Use the notebook RUNS mapping for many models/runs.")
    parser.add_argument("--experiments", help="Experiment numbers, e.g. 1-8,101-108. Defaults to all configured specs.")
    parser.add_argument("--base-dir", help="Base hyrax_runs directory.")
    parser.add_argument("--data-dir", help="Image data directory.")
    parser.add_argument("--results-dir", help="Hyrax results directory.")
    parser.add_argument("--filter-catalog", help="Filter catalog FITS path.")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting.")
    parser.add_argument("--no-pause", action="store_true", help="Do not pause after the first sbatch submit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.action == "list-models":
        for name in discover_builtin_model_names():
            print(name)
        return 0

    try:
        groups = _groups_from_models(_parse_models(args.models))
        specs = configured_experiment_specs(groups=groups)
    except ValueError as exc:
        parser.error(str(exc))
    experiment_numbers = parse_number_spec(args.experiments, default=sorted(specs))
    plan = configured_run_plan(
        profile=args.profile,
        run_number=args.run_number,
        base_directory=args.base_dir,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        filter_catalog=args.filter_catalog,
    )

    if args.action == "print-plan":
        print(f"Run directory: {plan.run_dir}")
        print_experiment_specs({number: specs[number] for number in experiment_numbers})
        return 0

    if args.action == "write-training":
        create_training_files(plan, specs, experiment_numbers)
    elif args.action == "submit-training":
        submit_jobs(
            run_number=plan.run_number,
            job_numbers=experiment_numbers,
            prefix="train",
            base_directory=plan.base_directory,
            pause_after_first_seconds=0 if args.no_pause else 10,
            dry_run=args.dry_run,
        )
    elif args.action == "write-infer":
        create_infer_scripts_batch(plan, experiment_numbers)
    elif args.action == "submit-infer":
        submit_jobs(plan.run_number, experiment_numbers, "infer", plan.base_directory, dry_run=args.dry_run)
    elif args.action == "write-udb":
        create_udb_scripts_batch(plan, experiment_numbers)
    elif args.action == "submit-udb":
        submit_jobs(plan.run_number, experiment_numbers, "udb", plan.base_directory, dry_run=args.dry_run)
    elif args.action == "write-3dumap":
        create_3dumap_scripts_batch(plan, experiment_numbers)
    elif args.action == "submit-3dumap":
        submit_jobs(plan.run_number, experiment_numbers, "3dumap", plan.base_directory, dry_run=args.dry_run)
    elif args.action == "write-3d-viz-json":
        create_3d_viz_json_batch(plan, experiment_numbers)
    else:
        parser.error(f"Unknown action: {args.action}")
    return 0


# Short alias matching the optimizer helper name used in the V2 notebooks.
optimizer_config_v2 = optimizer_config


if __name__ == "__main__":
    raise SystemExit(main())
