# Multi-latent merger pilot

The pilot compares Run 11 experiments 9, 13, and 14 on the same 6,079 galaxies.
The target is `0 <= Major_TimeSinceMerger <= 0.75` Gyr. Scientific conclusions
and exported figures live in the generated result bundle, not in this guide.

Open `multi_latent_analysis.ipynb` to view the completed bundle or run a new one.
Reusable logic is in `multi_latent_analysis.py`; existing analysis notebooks and
Hyrax result directories are unchanged. The current bundle is
`results/multi_latent_run11/full/`, and a small check is in `smoke/`.
All staged data and generated outputs are ignored by Git through `results/`.

## Reproduce

Run from the Hyrax-Research repository root. For this meeting the environment is
`/private/tmp/multi_latent_venv/bin/python`. To create a persistent environment:

```bash
python3.12 -m venv results/multi_latent_run11/venv
results/multi_latent_run11/venv/bin/python -m pip install -r multi_latent_requirements.txt
```

With that environment active:

```bash
python -m pytest tests/test_multi_latent_analysis.py -q
python multi_latent_analysis.py --help
python multi_latent_analysis.py \
  --manifest results/multi_latent_run11/inputs_manifest.json \
  --output-dir results/multi_latent_run11/reproduction \
  --n-splits 100
```

Use `--n-splits 2 --no-figures` for a smoke run. An existing completed bundle
is never overwritten: choose a new directory, or load it from the notebook.
The seed, neighbor ranks, target column/cutoff, reference, and split fractions
are CLI options. The full installed dependency lock is saved in
`results/multi_latent_run11/environment.lock.txt`.

## Input manifest

Paths resolve relative to the manifest. Each experiment supplies existing Hyrax
inference and two-dimensional UMAP result directories. Only files matching
`batch_<integer>.npy` are read; metadata files are excluded. Each numeric batch
must contain `id` and `tensor` fields. IDs are mandatory; row-order fallback is
never used. Here is a schematic manifest (the actual pilot contains three members):

```json
{
  "catalog": "inputs/split_images/catalog2.fits",
  "catalog_id_column": "object_id",
  "experiments": [
    {
      "key": "run11_expt9",
      "inference_dir": "inputs/hyrax_runs/results/EXISTING_INFER_DIRECTORY",
      "umap_dir": "inputs/hyrax_runs/results/EXISTING_UMAP_DIRECTORY",
      "source_configs": ["inputs/hyrax_runs/run11/udb11_9.toml"]
    }
  ]
}
```

At least two members, including `--reference`, are required. The actual pilot
manifest is already populated from the selected Delta artifacts. For Delta use,
absolute paths can point directly to original result directories and catalog;
the module itself does not submit jobs or download data.

All representations must expose exactly the same unique object IDs. Catalog
filter rows are collapsed only if knownness and target membership agree.
Unknown labels are excluded from evaluation and counted separately. Finite
negative sentinels retain the existing workflow's known-negative semantics.
Missing catalog IDs, duplicate embedding IDs, unsafe float IDs, nonfinite
vectors, and zero-scale spaces fail explicitly.

## Evaluation

- Every method shares the same object partitions per repetition. Class counts
  use the existing largest-remainder 50/25/25 allocation. Only training positives
  act as merger anchors; no self-anchor can enter validation or test scoring.
- Each latent block is centered and divided by its training RMS vector norm.
  Concatenating blocks with weight `1/sqrt(number_of_experiments)` makes squared
  distance the mean normalized squared distance. Query-to-anchor distances are
  computed directly, avoiding full all-object distance matrices.
- UMAP scores use the right-continuous validation distance CDF for each member
  and k, then average member percentiles. The CDF is frozen for test objects.
- Validation AP selects k from 3/5/7/9; ties choose smaller k. Validation F1
  selects a fixed threshold using the existing deterministic tie rules. The
  best-single comparator also selects its member on validation AP; member ties
  break lexicographically. Test labels do not enter any of these choices.
- AP is the step area of the tied-score precision–recall curve. Reported purity,
  completeness, and F1 use the frozen validation threshold. Curve bands are
  pointwise split quantiles at the first threshold reaching each recall.
- The shuffled control permutes labels at unchanged prevalence and reruns the
  entire anchor/validation/test procedure. Methods share partitions within each
  control repetition. It is a negative control, not a formal permutation p-value.

CKA uses a fixed unlabeled sample of up to 2,000 common galaxies. Candidate
agreement is median Jaccard across observed splits between exact top-10% test
sets (`ceil(0.1 * n_test)`, object-ID tie breaking). The new combined-space UMAP
uses all-object normalization and is **only descriptive**; no test metric is
computed from that projection.

## Result bundle

- `meeting_brief.md`, `01_recovery`, `02_agreement`, and
  `03_spatial_interpretation` (each figure in PNG and SVG).
- `summary.csv`, `split_metrics.csv`, `paired_summary.csv`,
  `paired_differences.csv`, `curve_samples.csv`, `candidate_overlap.csv`,
  `cka.csv`, `validation_selections.csv`, and `normalization.csv`.
- `split_audit.npz`: exact string IDs, method/mode order, partitions, working
  labels, held-out scores, and CKA sample IDs. Partition codes are -1 unknown,
  0 training, 1 validation, 2 test. Scores are NaN outside each test set. Audit
  scores are stored as float32 for inspection; reported metrics use float64.
- `cohort.csv` and `descriptive_projection.csv`. Read the `object_id` column as
  string when loading CSVs; use the NPZ as the exact-ID audit source.
- `analysis_manifest.json`: settings, SHA-256 input/source fingerprints,
  dependency versions, cohort counts, and a completion marker. It is written
  last, after requested outputs are complete.

These are exploratory results from previously trained representations and a
target informed by earlier analysis. Overlapping repeated splits are correlated:
the 16th–84th bands describe split variability, not confidence intervals or
untouched-galaxy generalization.
