# Hyrax-Research Agent Guide

## Scope and purpose

This file applies to the entire `Hyrax-Research` repository. The repository is a
research workspace for IllustrisTNG/HSC image preparation, Hyrax model runs,
UMAP/clustering analysis, merger studies, and Slurm orchestration. It is not the
Hyrax library itself and is not an installable Python package.

The Hyrax source checkout normally lives in the sibling `../hyrax` repository.
Treat that as a separate project: do not edit it unless the task explicitly
includes Hyrax library changes. The local checkout may contain unrelated user
work, so never reset, clean, or reformat it as part of a change here.

## Environment and paths

- Run commands from this repository root.
- Use Python 3.11 or newer. `research_paths.py` uses the standard-library
  `tomllib` module.
- This repository has no `pyproject.toml`, requirements file, or lockfile. Use
  an environment in which `hyrax` and the required scientific packages are
  already installed. For local Hyrax development, the sibling checkout can be
  installed with `python -m pip install -e ../hyrax`.
- Select paths with `HYRAX_PROFILE=local` or `HYRAX_PROFILE=delta`. Auto-detection
  exists, but explicit profiles make runs reproducible.
- `static_umap_metric_job_generator.py` defaults its own `--profile` to `delta`;
  pass `--profile local` explicitly for local planning. Its default Slurm account
  and GPU partition remain Delta-specific even with the local path profile.
- `path_profiles.toml` and `research_paths.py` are the source of truth for data,
  catalog, run, and result locations. Add reusable paths there rather than
  introducing new machine-specific absolute paths.
- The local profile relies on ignored data below `test_dir_100images`, so validate
  required paths before assuming that a fresh clone can run local workflows.
- A quick profile check is:

  ```bash
  HYRAX_PROFILE=local python -c "from research_paths import paths; print(paths.profile, paths.validate())"
  ```

- Keep credentials out of code and command output. The IllustrisTNG downloader
  reads `TNG_API_KEY`; it does not automatically load `.env`. The ignored `.env`
  must never be committed. Some historical files may contain embedded
  credentials—do not copy or expose them.

## Repository map

- `grand_run_script_generator.py`: canonical reusable generator for Hyrax TOML
  configs and training, inference, UDB, and 3D UMAP Slurm jobs.
- `static_umap_metrics.py` and `static_umap_metric_job_generator.py`: batchable
  UMAP overlay/metric analysis and its Slurm job generator.
- `STATIC_UMAP_METRICS_SLURM.md`: authoritative usage notes for the static UMAP
  metrics workflow.
- `research_paths.py` and `path_profiles.toml`: local/Delta path resolution.
- `download_hsc_pipeline.py`: IllustrisTNG URL discovery, FITS download/split,
  catalog generation, and cutout-size analysis.
- `cutout_size_histogram.py`, `split_catalog_by_cutout_size.py`, and
  `standardize_fits_cutouts.py`: FITS preprocessing utilities.
- Root notebooks: exploratory and presentation-oriented research workflows.
  When a notebook wraps a root Python script, keep the reusable logic in the
  script and leave the notebook as a thin control/visualization layer.
- `illustris_mergers/`: legacy merger-history programs with site-specific paths
  and an `illustris_python` dependency. Do not assume they are portable.
- `random/`, `juneau/`, and older generator notebooks: prototypes and historical
  experiments, not the default source of truth for current workflows.
- `data/`: small tracked catalogs/URL lists plus ignored local data products.
  `test_dir*`, `results/`, and profile run directories are local fixtures or
  generated artifacts, not an automated test suite.

## Change conventions

- Prefer small reusable functions and CLI options in root Python scripts over
  duplicating logic across notebooks.
- Follow the style of the newer scripts: four-space indentation, `pathlib.Path`,
  type hints where useful, clear CLI help, and a `main()` entry point.
- Preserve deterministic analysis defaults and pass seeds explicitly when
  adding randomized metrics, clustering, or sampling.
- Preserve FITS column names, object-ID semantics, filter ordering, and catalog
  to-image alignment. Treat object IDs as identifiers rather than measurements;
  avoid lossy float conversions.
- Keep expensive scientific imports lazy where the existing script does so.
  This lets `--help`, plan generation, and lightweight checks work outside the
  full GPU/science environment.
- Do not broadly reformat notebooks or regenerate all notebook outputs. Make
  targeted cell changes, avoid JSON churn, and execute only the cells or small
  fixtures needed for the task.
- Do not commit generated models, plots, caches, Slurm logs, downloaded FITS
  images, local configs, or secrets. Respect `.gitignore`; do not force-add
  ignored artifacts unless the user explicitly requests it.
- Write experimental outputs to profile-managed result directories or `/tmp`,
  not over tracked catalogs or reference artifacts.

## External side effects

- `print-plan`, `list-models`, `--help`, and submission commands with `--dry-run`
  are safe inspection tools.
- Commands named `submit*` call `sbatch` and create external cluster jobs. Do not
  run them without explicit user authorization. Prefer generating scripts or
  using `--dry-run` for validation.
- Downloader `fetch` and `split` operations make network requests and may create
  large data sets. Use a deliberately small batch for validation and only when
  the task authorizes downloads.
- Avoid destructive or in-place transformations of FITS data. Use a separate
  output directory and an explicit overwrite option when replacement is truly
  intended.

## Validation

There is no repository-level automated test suite. Match validation to the
changed workflow and state clearly when data, Hyrax, Slurm, or GPU access makes
an end-to-end run unavailable.

Start with lightweight checks:

```bash
python -m py_compile *.py illustris_mergers/*.py random/*.py
python grand_run_script_generator.py --help
python static_umap_metric_job_generator.py --help
HYRAX_PROFILE=local python grand_run_script_generator.py print-plan
python static_umap_metric_job_generator.py print-plan --profile local
```

For a changed CLI, also run its `--help` path and the smallest non-destructive
operation available. For FITS/data changes, use a copy or a small ignored fixture
and verify row counts, column names, object IDs, shapes, and output paths. For
notebook changes, execute only a minimal representative path in the intended
kernel and inspect the resulting tables/figures for scientific plausibility.
