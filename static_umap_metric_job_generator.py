#!/usr/bin/env python3
"""Generate Slurm jobs for static_umap_metrics.py.

The defaults mirror the notebook:
  - run10 experiments 7,10,12,13,14,15,18 with time-since and future flags
  - run11 experiments 7,10,12,13,14 with time-since flags

Explicit --run-expts values switch to the requested runs and use the overlay
groups supplied with --overlay-group, or time_since_merger when omitted.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_SCRIPT = THIS_DIR / "static_umap_metrics.py"
DEFAULT_OUTPUT_DIR = THIS_DIR / "results" / "static_umap_metrics"

DEFAULT_NOTEBOOK_PLAN = [
    {
        "run": 10,
        "experiments": [7, 10, 12, 13, 14, 15, 18],
        "overlay_groups": ["time_since_merger", "future_merger_flags"],
    },
    {
        "run": 11,
        "experiments": [7, 10, 12, 13, 14],
        "overlay_groups": ["time_since_merger"],
    },
]

DEFAULT_SLURM = {
    "account": "bemi-delta-gpu",
    "partition": "gpuA40x4",
    "nodes": 1,
    "ntasks": 1,
    "cpus-per-task": 4,
    "mem": "32G",
    "time": "2:00:00",
}

DEFAULT_SHELL_PREAMBLE = ["set -euo pipefail"]
DEFAULT_SETUP_LINES = [
    'export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-manual}"',
    'export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba-${SLURM_JOB_ID:-manual}"',
    'mkdir -p "$MPLCONFIGDIR"',
    'mkdir -p "$NUMBA_CACHE_DIR"',
    "# source /mmfs1/home/aritrag/.bashrc",
    "# conda activate hyrax",
]


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any] | None) -> dict[str, Any]:
    for key, value in (updates or {}).items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


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


def parse_number_spec(spec: str | Sequence[int] | None, default: Sequence[int] | None = None) -> list[int]:
    """Parse strings like '1-8,11,101-108' into a sorted unique list."""
    if spec is None:
        return list(default or [])
    if not isinstance(spec, str):
        return sorted({int(number) for number in spec})

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


def parse_run_expts(specs: Sequence[str] | None) -> dict[int, list[int]]:
    """Parse --run-expts values like '10:7,10,12' or '11=7-14'."""
    if not specs:
        return {}

    parsed: dict[int, list[int]] = {}
    for spec in specs:
        if ":" in spec:
            run_text, expts_text = spec.split(":", 1)
        elif "=" in spec:
            run_text, expts_text = spec.split("=", 1)
        else:
            raise ValueError(f"Run spec '{spec}' must look like RUN:EXPTS, for example 10:7,10,12")
        run = int(run_text.strip())
        parsed[run] = parse_number_spec(expts_text)
    return parsed


def parse_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(text)
    except ValueError:
        return text


def parse_key_value_overrides(items: Sequence[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be KEY=VALUE")
        key, value = item.split("=", 1)
        overrides[key.strip()] = parse_scalar(value)
    return overrides


def load_default_base_directory(profile: str | None = None) -> Path:
    if profile:
        os.environ["HYRAX_PROFILE"] = profile
    from research_paths import load_paths

    paths = load_paths(profile=profile)
    return Path(paths.hyrax_runs)


def resolve_base_directory(profile: str | None, base_directory: str | Path | None) -> Path:
    if base_directory:
        return Path(base_directory).expanduser()
    return load_default_base_directory(profile)


def resolve_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    explicit_runs = parse_run_expts(args.run_expts)
    if explicit_runs:
        overlay_groups = list(args.overlay_group or ["time_since_merger"])
        return [
            {
                "run": run,
                "experiments": experiments,
                "overlay_groups": overlay_groups,
            }
            for run, experiments in sorted(explicit_runs.items())
        ]

    if args.overlay_group:
        overlay_groups = list(args.overlay_group)
        return [
            {
                "run": item["run"],
                "experiments": list(item["experiments"]),
                "overlay_groups": overlay_groups,
            }
            for item in DEFAULT_NOTEBOOK_PLAN
        ]

    return deepcopy(DEFAULT_NOTEBOOK_PLAN)


def build_analysis_command(
    args: argparse.Namespace,
    run: int,
    expt: int,
    overlay_groups: Sequence[str],
    base_directory: Path,
) -> str:
    command = [
        args.python_executable,
        str(Path(args.analysis_script).expanduser().resolve()),
        "--run",
        str(run),
        "--expt",
        str(expt),
        "--base-directory",
        str(base_directory),
        "--output-dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--catalog-key",
        args.catalog_key,
        "--n-permutations",
        str(args.n_permutations),
        "--min-cluster-size",
        str(args.min_cluster_size),
        "--seed",
        str(args.seed),
        "--dpi",
        str(args.dpi),
        "--alpha-background",
        str(args.alpha_background),
        "--s-background",
        str(args.s_background),
        "--density-cmap",
        args.density_cmap,
    ]

    if args.profile:
        command.extend(["--profile", args.profile])
    if args.catalog_path:
        command.extend(["--catalog-path", str(Path(args.catalog_path).expanduser())])
    if args.catalog_id_column:
        command.extend(["--catalog-id-column", args.catalog_id_column])
    if args.id_field:
        command.extend(["--id-field", args.id_field])
    if args.time_since_merger_max_gyr is not None:
        command.extend(["--time-since-merger-max-gyr", str(args.time_since_merger_max_gyr)])
    if args.include_highdim:
        command.append("--include-highdim")
    if args.all_overlay_groups:
        command.append("--all-overlay-groups")
    else:
        for group in overlay_groups:
            command.extend(["--overlay-group", group])
    if args.density:
        command.append("--density")
    if args.log_colorbar:
        command.append("--log-colorbar")
    if not args.show_legend:
        command.append("--no-show-legend")
    if not args.suppress_logs:
        command.append("--no-suppress-logs")

    return shlex.join(command)


def create_job_file(
    run_dir: Path,
    run: int,
    expt: int,
    command: str,
    slurm: Mapping[str, Any],
    shell_preamble: Sequence[str],
    setup_lines: Sequence[str],
    job_prefix: str,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    script_name = f"{job_prefix}{run}_{expt}.sh"
    log_name = f"{job_prefix}{run}_{expt}.txt"
    job_name = f"pm{run}_{expt}"
    content = render_slurm_script(
        job_name=job_name,
        output_file=log_name,
        commands=[command],
        slurm=slurm,
        shell_preamble=shell_preamble,
        setup_lines=setup_lines,
    )
    script_path = run_dir / script_name
    script_path.write_text(content, encoding="utf-8")
    print(f"Created {script_path}")
    return script_path


def create_jobs(args: argparse.Namespace) -> list[Path]:
    base_directory = resolve_base_directory(args.profile, args.base_directory)
    plan = resolve_plan(args)
    slurm = deepcopy(DEFAULT_SLURM)
    _deep_update(slurm, parse_key_value_overrides(args.slurm))

    setup_lines = list(args.setup_line if args.setup_line is not None else DEFAULT_SETUP_LINES)
    if args.no_default_setup and args.setup_line is None:
        setup_lines = []
    shell_preamble = list(args.shell_preamble if args.shell_preamble is not None else DEFAULT_SHELL_PREAMBLE)

    created: list[Path] = []
    for item in plan:
        run = int(item["run"])
        run_dir = base_directory / f"run{run}"
        overlay_groups = list(item["overlay_groups"])
        for expt in item["experiments"]:
            command = build_analysis_command(args, run, int(expt), overlay_groups, base_directory)
            script_path = create_job_file(
                run_dir=run_dir,
                run=run,
                expt=int(expt),
                command=command,
                slurm=slurm,
                shell_preamble=shell_preamble,
                setup_lines=setup_lines,
                job_prefix=args.job_prefix,
            )
            created.append(script_path)

    return created


def print_plan(args: argparse.Namespace) -> None:
    base_directory = resolve_base_directory(args.profile, args.base_directory)
    plan = resolve_plan(args)
    slurm = deepcopy(DEFAULT_SLURM)
    _deep_update(slurm, parse_key_value_overrides(args.slurm))

    print(f"Analysis script: {Path(args.analysis_script).expanduser().resolve()}")
    print(f"Base directory: {base_directory}")
    print(f"Output directory: {Path(args.output_dir).expanduser().resolve()}")
    print(f"Catalog key: {args.catalog_key}")
    print(f"Slurm: {slurm}")
    for item in plan:
        print(
            f"run{item['run']}: experiments {item['experiments']} | "
            f"overlay groups {item['overlay_groups']}"
        )


def submit_jobs(job_paths: Sequence[Path], pause_after_first_seconds: int = 10, dry_run: bool = False) -> None:
    for index, job_path in enumerate(job_paths):
        command = ["sbatch", job_path.name]
        print(f"Submitting from {job_path.parent}: {' '.join(command)}")
        if not dry_run:
            subprocess.run(command, cwd=job_path.parent, check=True)
        if index == 0 and pause_after_first_seconds:
            time.sleep(pause_after_first_seconds)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Slurm scripts for batch static UMAP metric jobs.")
    parser.add_argument("action", nargs="?", choices=["print-plan", "write", "submit"], default="write")
    parser.add_argument(
        "--run-expts",
        action="append",
        help="Run and experiments to process, e.g. 10:7,10,12 or 11:7-14. Repeat for multiple runs.",
    )
    parser.add_argument("--profile", default="delta", help="research_paths profile to use in generated commands.")
    parser.add_argument("--base-directory", type=Path, help="Hyrax runs base directory. Defaults to research_paths.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-script", type=Path, default=DEFAULT_ANALYSIS_SCRIPT)
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--job-prefix", default="plot_metrics")

    parser.add_argument("--catalog-key", default="raw_merger_flags")
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--catalog-id-column")
    parser.add_argument("--id-field", default="objectId_data")
    parser.add_argument("--overlay-group", action="append", help="Overlay group to process. Repeat for multiple groups.")
    parser.add_argument("--all-overlay-groups", action="store_true", help="Pass --all-overlay-groups to analysis jobs.")
    parser.add_argument("--time-since-merger-max-gyr", type=float)

    parser.add_argument("--n-permutations", type=positive_int, default=500)
    parser.add_argument("--min-cluster-size", type=positive_int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-highdim", action="store_true")

    parser.add_argument("--dpi", type=positive_int, default=150)
    parser.add_argument("--alpha-background", type=float, default=0.5)
    parser.add_argument("--s-background", type=float, default=1.0)
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--log-colorbar", action="store_true")
    parser.add_argument("--density-cmap", default="viridis")
    parser.add_argument("--show-legend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suppress-logs", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--slurm", action="append", help="Override a Slurm directive, e.g. partition=cpu. Repeat as needed.")
    parser.add_argument("--shell-preamble", action="append", help="Shell line before setup. Defaults to set -euo pipefail.")
    parser.add_argument("--setup-line", action="append", help="Setup line in the Slurm script. Repeat for multiple lines.")
    parser.add_argument("--no-default-setup", action="store_true", help="Do not include commented default setup lines.")
    parser.add_argument("--pause-after-first-seconds", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="For submit action, print sbatch commands without running them.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.action == "print-plan":
        print_plan(args)
        return 0

    job_paths = create_jobs(args)
    if args.action == "submit":
        submit_jobs(
            job_paths,
            pause_after_first_seconds=args.pause_after_first_seconds,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
