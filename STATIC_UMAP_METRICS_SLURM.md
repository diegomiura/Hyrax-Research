# Static UMAP Metrics Slurm Workflow

This workflow converts `static_umap_plotting_w_xmatched_samples_and_metric.ipynb`
into batchable scripts:

- `static_umap_metrics.py` processes one run/experiment pair and writes plots,
  metrics CSV, overlay-summary CSV, and a JSON record.
- `static_umap_metric_job_generator.py` writes Slurm job scripts that call the
  analysis script for many run/experiment pairs.

## Quick Start On Delta

Preview the default notebook-derived plan:

```bash
HYRAX_PROFILE=delta python static_umap_metric_job_generator.py print-plan
```

Write Slurm scripts:

```bash
HYRAX_PROFILE=delta python static_umap_metric_job_generator.py write
```

Submit the generated scripts:

```bash
HYRAX_PROFILE=delta python static_umap_metric_job_generator.py submit
```

The generator writes scripts into each run directory, for example
`run10/plot_metrics10_7.sh`. Each script writes its Slurm log beside the script,
for example `run10/plot_metrics10_7.txt`.

## Defaults

When no `--run-expts` is supplied, the generator mirrors the notebook without
duplicating the repeated Run 10 time-since section:

- Run 10: experiments `7,10,12,13,14,15,18`, overlay groups
  `time_since_merger` and `future_merger_flags`.
- Run 11: experiments `7,10,12,13,14`, overlay group `time_since_merger`.

The default catalog is `raw_merger_flags`, which resolves through
`research_paths.py` to `catalog2.fits` on the selected profile.

## Common Customizations

Run only selected experiments:

```bash
python static_umap_metric_job_generator.py write \
  --run-expts 10:7,10,12 \
  --run-expts 11:7-14
```

Use explicit overlay groups:

```bash
python static_umap_metric_job_generator.py write \
  --run-expts 10:7,10,12 \
  --overlay-group time_since_merger \
  --overlay-group future_merger_flags
```

Override Slurm resources:

```bash
python static_umap_metric_job_generator.py write \
  --slurm partition=cpu \
  --slurm account=YOUR_ACCOUNT \
  --slurm cpus-per-task=8 \
  --slurm mem=64G \
  --slurm time=4:00:00
```

Use a specific Python executable:

```bash
python static_umap_metric_job_generator.py write \
  --python-executable /path/to/env/bin/python
```

Or activate an environment inside each generated job:

```bash
python static_umap_metric_job_generator.py write \
  --setup-line 'source /path/to/.bashrc' \
  --setup-line 'conda activate hyrax'
```

## Running One Job Directly

For debugging without Slurm:

```bash
python static_umap_metrics.py \
  --run 10 \
  --expt 12 \
  --profile delta \
  --catalog-key raw_merger_flags \
  --overlay-group time_since_merger
```

List detected overlay groups for a catalog:

```bash
python static_umap_metrics.py \
  --profile delta \
  --catalog-key raw_merger_flags \
  --list-overlays
```

## Outputs

For run `10`, experiment `12`, outputs are written under:

```text
results/static_umap_metrics/run10/expt12/
```

Files include:

- `run10_expt12_<overlay_group>.png`: static UMAP with overlay points and metric annotation.
- `run10_expt12_metrics.csv`: flattened per-overlay metric table.
- `run10_expt12_overlay_summary.csv`: selected catalog-row counts and value ranges.
- `run10_expt12_metrics.json`: full nested metric payload with paths and settings.

## Metrics

MNLN is the median nearest labeled-neighbor distance ratio. Values below 1 mean
the labeled points are closer together than random same-sized samples. Its
p-value is lower-tail: the fraction of random samples with median nearest-neighbor
distance less than or equal to the observed value.

CMC-Gini fits HDBSCAN once to the coordinate space and measures how concentrated
the labeled points are across HDBSCAN clusters, including noise as a bin. Higher
Gini means stronger concentration. Its p-value is upper-tail: the fraction of
random label assignments with Gini greater than or equal to the observed Gini.

`--include-highdim` computes both metrics in the original latent space as well
as in the 2D UMAP space, when `[results].inference_dir` is available in the UMAP
config.

## Notebook Logic Corrections

The script keeps the notebook behavior but fixes or hardens several parts:

- UMAP path parsing no longer catches every `ValueError` and silently retries;
  it parses the UMAP log first, then reads optional `[results].inference_dir`
  from the TOML.
- Object IDs are normalized without lossy float parsing, while still matching
  integer-like values such as `123`, `123.0`, and byte strings.
- MNLN uses exact nearest-neighbor search instead of materializing dense
  pairwise distance matrices for every permutation.
- HDBSCAN labels are fit once per coordinate space per job and reused across
  overlays.
- `MPLCONFIGDIR` and `NUMBA_CACHE_DIR` are placed under temporary storage in
  generated Slurm jobs to avoid cache-write failures on cluster filesystems.
