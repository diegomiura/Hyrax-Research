# Which galaxies are consistently classified as mergers?

Use `merger_consensus.ipynb` to inspect the results and change the analysis
settings. Reusable logic lives in `merger_consensus.py`. This extends the
research workflow alongside the earlier `multi_latent_analysis.py` pilot;
neither that pilot nor the Hyrax training code is changed.

The question is now **agreement on individual galaxies**, with classifier
performance reported alongside it. Agreement alone does not establish that a
galaxy is a merger.

## Defaults

| Setting | Default |
|---|---|
| Target | Major merger in the past 0.75 Gyr, including both endpoints |
| Experiments retained | Top 3 per run, representation, and classifier |
| Selection metric | Validation average precision (AP); key breaks ties |
| Classifiers | Logistic regression, random forest, kth-nearest merger anchor |
| Representations | Original latent vectors and existing UMAP coordinates, separately |
| Repetitions | 100, seed 42 |
| Split | 25% training / 25% validation / **50% held out**, stratified |
| Binary decision | Each classifier's threshold maximizes validation F1 |
| Neighbor rank | Validation AP chooses among 3, 5, 7, 9; smaller k breaks ties |
| Consensus weighting | Equal experiments within algorithms, equal algorithms within runs, equal runs |
| Cross-run stability | At least 2 runs, at least 5 holdouts, vote >= 0.5 in every run |

“Top 3” is implemented in both useful senses: select the top three experiments
for each classifier and export the top three **galaxies** for each classifier.
If a run supplies fewer than three experiments, all available experiments are
used, with the actual counts in the selection audit.

## Why the split differs from the previous pilot

The previous combined-space pilot used 50/25/25 train/validation/test. Here the
**test half is 50%**, as requested in the meeting notes. The other half is split
equally between fitting and validation. The models are not refitted after
threshold selection. This keeps test labels out of experiment, k, and threshold
selection and keeps validation thresholds on the same fitted model used for test
predictions. It also means this experiment trains on fewer labeled objects than
the previous pilot: its performance numbers are not a paired improvement claim.

All runs, experiments, classifiers, and representations share exactly the same
partitions. Scaling is fit on training vectors only. Logistic regression uses
per-feature standardization and balanced class weights. The forest uses 128
trees, balanced subsampling, and a minimum leaf size of 5. The anchor baseline
preserves Euclidean geometry using a single training RMS scale and uses only
training mergers as anchors. Hyperparameters other than anchor k are fixed.

## Run on the locally staged science data

The local science manifest contains **Run 11, experiments 9/13/14** and 6,079
galaxies, including 231 targets. This is a useful single-run pilot; it cannot
answer the cross-run part yet. Since exactly three experiments are staged,
top-three selection retains all three. It does not discover the best three
among the complete Run 11 experiment family.

The completed local bundle is `results/merger_consensus_run11/full/`, with 100
observed-label repetitions and 100 shuffled-label control repetitions.
`pilot_findings.md` interprets the results, `candidate_agreement.png/.svg` shows
the leading shared candidates by classifier, and
`cross_representation_candidates.csv` compares the latent and UMAP votes on
each galaxy. Rebuild those descriptive pilot summaries using the bundle's
`make_readout.py`. The notebook also computes the shared-candidate comparison
directly from any selected bundle.

Use Python 3.11+ with `merger_consensus_requirements.txt`. The installed local
Anaconda Python was used for this analysis; the old temporary pilot environment
no longer has its scientific dependencies. Exact versions and source/input
hashes are saved in each result bundle.

```bash
python -m pytest tests/test_merger_consensus.py -q
python merger_consensus.py run --help

MPLCONFIGDIR=/tmp/merger_consensus_mpl LOKY_MAX_CPU_COUNT=2 \
python merger_consensus.py run \
  --manifest results/multi_latent_run11/inputs_manifest.json \
  --output-dir results/merger_consensus_run11/reproduction \
  --n-splits 100 --shuffled-control
```

For a small check, use `--n-splits 2 --forest-trees 16 --no-figures` and a new
output directory. Such a check is not enough for stable candidate frequencies.
Existing nonempty output directories are never overwritten.

To change the scientific target, rerun with, for example:

```bash
python merger_consensus.py run \
  --manifest results/multi_latent_run11/inputs_manifest.json \
  --output-dir results/merger_consensus_run11/minor_1gyr \
  --merger-type minor --cutoff-gyr 1.0
```

`--target-column` supports another time-since-merger column. Finite negative
sentinels remain known negatives; NaN and infinite values are unknown. Mini and
minor mergers should be analyzed separately, not mixed into the major-merger
target or averaged as if they were identical labels.

## Discover experiments across runs on Delta

After authenticating to Delta, run the following **on Delta** from the repository
with the required Python environment:

```bash
python merger_consensus.py discover --profile delta \
  --output results/merger_consensus_inputs.json

python merger_consensus.py run \
  --manifest results/merger_consensus_inputs.json \
  --output-dir results/merger_consensus_all_runs \
  --n-splits 100 --top-experiments 3 --shuffled-control
```

Discovery reads existing `run*/udb*.toml` configs and their UMAP logs through
the repository's artifact resolver. It records missing/unusable artifacts in
`discovery_failures`; review these before interpreting run coverage. Optional
`--runs 10 11 12` restricts discovery. `--run-base` and `--catalog` override the
profile paths. Discovery does not submit Slurm jobs, retrain Hyrax, or download
data. The classification command runs in the foreground; use the cluster's
approved compute environment for a large experiment family.

Each manifest experiment requires `key`, `run`, `experiment`, `inference_dir`,
and `umap_dir`; the pilot manifest already has those fields. Paths resolve
relative to the manifest. `source_configs` and `umap_params` preserve provenance.
Different runs can have different UMAP `n_neighbors`, metrics, and other
parameters; consensus compares decisions on shared galaxy IDs, not coordinates
between runs. UMAP `n_neighbors` is distinct from the merger-anchor k above.

The default requires identical object IDs in every supplied space. If the
historical experiments intentionally sampled different populations, rerun with
`--cohort-mode intersection`. The manifest then records the source/common/dropped
object counts for every space. No row-order fallback or missing-as-negative
fill is used. Compare runs within a common simulation/snapshot ID namespace;
do not combine unrelated catalogs that happen to reuse numeric IDs.

## Reading the result bundle

- `meeting_brief.md` and `consensus_distribution.png/.svg`: quick inspection.
- `consensus_candidates.csv`: complete per-representation ranking;
  `vote_fraction` is mean held-out agreement. `minimum_run_vote_fraction` makes
  disagreement between runs visible. `unanimous_split_fraction` is the fraction
  of test appearances where **all selected classifiers in all runs** voted merger.
- `algorithm_consensus.csv` and `top_candidates_per_algorithm.csv`: individual
  classifier rankings, averaged equally across runs, and each top-three list.
- `run_candidates.csv` and `algorithm_candidates.csv`: the components of the
  consensus. `n_holdouts` is a galaxy's evaluation count, not the number of
  models. Small counts are flagged, not silently treated as reliable.
- `experiment_selections.csv` and `experiment_selection_frequency.csv`: every
  validation decision and how often each experiment enters its group.
- `run_agreement.csv`: pairwise Jaccard of candidate sets and Spearman correlation
  of vote fractions. Empty when only one run is present. Empty-set Jaccard and
  correlation of constant scores are undefined (blank), not perfect agreement.
- `split_metrics.csv` and `performance_summary.csv`: held-out AP, purity,
  completeness, and F1 at the algorithm, run, and combined levels. AP uses mean
  validation-percentile rank scores; binary metrics use the vote threshold.
- `shuffled_split_metrics.csv`, `shuffled_experiment_selections.csv`, and
  `shuffled_prediction_audit.npz`: optional control rerunning the entire pipeline
  after shuffling labels at unchanged prevalence. This is a negative control,
  not a permutation p-value. The control makes its own validation selections.
- `prediction_audit.npz`: exact string IDs, shared partitions, working labels,
  selected model masks, thresholds, scores, and votes. Scores are NaN and votes
  are -1 outside test sets. Scores are float32 for compactness; votes were
  computed in float64 and are authoritative near thresholds.
- `cohort.csv`: includes unknown and zero-holdout objects; their absence from
  a ranking is explicit. `analysis_manifest.json` is written last, on success.

Read CSV IDs as strings (`pd.read_csv(path, dtype={"object_id": str})`). Training
normalization and model fits use float64; object IDs never pass through floats.

Vote fractions, classifier outputs, and percentile ranks are **not calibrated
merger probabilities**. The existing representations were trained previously;
this is an exploratory label-holdout analysis. Repeated splits overlap, and
experiment choices made before this workflow may have used the full sample.
Cross-run agreement is reproducibility on the same galaxies, not validation on
independent observations or another simulation.
