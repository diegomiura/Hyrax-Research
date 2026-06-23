"""Shared local/HPC path selection for Hyrax-Research.

Usage in notebooks/scripts:

    from research_paths import paths

    data_dir = paths.split_images_120
    catalog = paths.catalog("all")
    runs_dir = paths.hyrax_runs

Set HYRAX_PROFILE=local or HYRAX_PROFILE=delta to override auto-detection.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import socket
from string import Formatter
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE_FILE = REPO_ROOT / "path_profiles.toml"
PROFILE_ENV_VARS = ("HYRAX_PROFILE", "RESEARCH_PATH_PROFILE")


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Python 3.11+ is required to read path_profiles.toml via the "
            "standard library. On older Python, install tomli or set up the "
            "profile paths directly before importing research_paths."
        ) from exc

    with path.open("rb") as handle:
        return tomllib.load(handle)


def _requested_profile() -> str | None:
    for env_var in PROFILE_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            return value.strip()
    return None


def detect_profile() -> str:
    """Detect the runtime profile unless HYRAX_PROFILE overrides it."""
    requested = _requested_profile()
    if requested:
        return requested

    hostname = socket.gethostname().lower()
    slurm_cluster = (
        os.environ.get("SLURM_CLUSTER_NAME")
        or os.environ.get("SLURM_JOB_CLUSTER")
        or ""
    ).lower()

    if (
        "delta" in hostname
        or hostname.startswith("dt-")
        or "delta" in slurm_cluster
        or Path("/work/hdd/bemi/dmiura").exists()
    ):
        return "delta"

    if platform.system() == "Darwin" or Path("/Users/diegomiura").exists():
        return "local"

    raise RuntimeError(
        "Could not detect a path profile. Set HYRAX_PROFILE=local or "
        "HYRAX_PROFILE=delta before running this code."
    )


def _format_fields(template: str) -> set[str]:
    return {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }


def _expand_path_templates(raw_paths: dict[str, str]) -> dict[str, Path]:
    values: dict[str, str] = {"repo_root": str(REPO_ROOT)}
    pending = {key: str(value) for key, value in raw_paths.items()}

    while pending:
        progressed = False
        for key, template in list(pending.items()):
            fields = _format_fields(template)
            if fields <= values.keys():
                values[key] = template.format(**values)
                pending.pop(key)
                progressed = True

        if not progressed:
            unresolved = {
                key: sorted(_format_fields(template) - values.keys())
                for key, template in pending.items()
            }
            raise ValueError(
                "Could not resolve path profile placeholders: "
                f"{unresolved}"
            )

    return {key: Path(value).expanduser() for key, value in values.items()}


@dataclass(frozen=True)
class ResearchPaths:
    """Resolved path profile with convenience helpers."""

    profile: str
    _paths: dict[str, Path]

    def __getattr__(self, name: str) -> Path:
        try:
            return self._paths[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._paths))
            raise AttributeError(
                f"No path named '{name}'. Available paths: {available}"
            ) from exc

    def as_dict(self) -> dict[str, Path]:
        return dict(self._paths)

    def catalog(self, name: str = "all") -> Path:
        aliases = {
            "all": "catalog_all",
            "catalog2": "catalog_raw_merger_flags",
            "raw_flags": "catalog_raw_merger_flags",
            "raw_merger_flags": "catalog_raw_merger_flags",
            "merger_flags": "catalog_raw_merger_flags",
            "le": "catalog_le_120x120",
            "le_120": "catalog_le_120x120",
            "le_120x120": "catalog_le_120x120",
            "gt": "catalog_gt_120x120",
            "gt_120": "catalog_gt_120x120",
            "gt_120x120": "catalog_gt_120x120",
        }
        key = aliases.get(name, name)
        return self._paths[key]

    def run_dir(self, run_number: int | str) -> Path:
        return self.hyrax_runs / f"run{run_number}"

    def run_config(self, prefix: str, run_number: int | str, experiment: int | str) -> Path:
        return self.run_dir(run_number) / f"{prefix}{run_number}_{experiment}.toml"

    def validate(self, required: tuple[str, ...] | None = None) -> dict[str, bool]:
        if required is None:
            required = (
                "data_root",
                "split_images",
                "split_images_120",
                "hyrax_runs",
                "hyrax_results",
                "catalog_all",
            )
        return {name: self._paths[name].exists() for name in required}


def load_paths(
    profile: str | None = None,
    profile_file: str | Path = DEFAULT_PROFILE_FILE,
) -> ResearchPaths:
    profile_file = Path(profile_file)
    config = _load_toml(profile_file)
    profiles = config.get("profiles", {})

    selected_profile = profile or detect_profile()
    if selected_profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(
            f"Unknown path profile '{selected_profile}'. "
            f"Available profiles: {available}"
        )

    raw_paths = profiles[selected_profile].get("paths", {})
    if not raw_paths:
        raise ValueError(f"Profile '{selected_profile}' has no paths configured.")

    return ResearchPaths(
        profile=selected_profile,
        _paths=_expand_path_templates(raw_paths),
    )


paths = load_paths()
