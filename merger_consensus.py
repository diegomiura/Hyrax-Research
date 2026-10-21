"""Find galaxies repeatedly classified as mergers across experiments and runs.

Run ``python merger_consensus.py --help``. Existing representations are read,
never retrained. Every experimental choice uses a validation quarter; the test
half contributes only held-out predictions and descriptive performance metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib import metadata
from itertools import combinations
import json
import math
from pathlib import Path
import re
import sys

import multi_latent_analysis as mla


ALGORITHMS = ("logistic", "random_forest", "anchor")
REPRESENTATIONS = ("latent", "umap")
np = pd = sm = None


def init_stack():
    global np, pd, sm
    mla.init_stack()
    np, pd, sm = mla.np, mla.pd, mla.sm


@dataclass(frozen=True)
class ConsensusConfig:
    n_splits: int = 100
    seed: int = 42
    top_experiments: int = 3
    algorithms: tuple[str, ...] = ALGORITHMS
    representations: tuple[str, ...] = REPRESENTATIONS
    neighbor_ranks: tuple[int, ...] = (3, 5, 7, 9)
    target_column: str = "Major_TimeSinceMerger"
    cutoff_gyr: float = 0.75
    consensus_threshold: float = 0.5
    min_holdouts: int = 5
    cohort_mode: str = "strict"
    forest_trees: int = 128

    def validate(self):
        for name in ("n_splits", "top_experiments", "min_holdouts", "forest_trees"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name, allowed in (("algorithms", ALGORITHMS), ("representations", REPRESENTATIONS)):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
                raise ValueError(f"{name} must be distinct entries from {allowed}")
        if (not self.neighbor_ranks or len(set(self.neighbor_ranks)) != len(self.neighbor_ranks)
                or any(isinstance(k, bool) or not isinstance(k, int) or k < 1
                       for k in self.neighbor_ranks)):
            raise ValueError("neighbor_ranks must be distinct positive integers")
        if not math.isfinite(self.cutoff_gyr) or self.cutoff_gyr < 0:
            raise ValueError("cutoff_gyr must be finite and nonnegative")
        if not math.isfinite(self.consensus_threshold) or not 0 < self.consensus_threshold <= 1:
            raise ValueError("consensus_threshold must be in (0, 1]")
        if self.cohort_mode not in ("strict", "intersection"):
            raise ValueError("cohort_mode must be strict or intersection")


@dataclass
class ConsensusCohort:
    ids: object
    spaces: dict
    members: dict
    known: object
    targets: object
    coverage: dict


def build_cohort(embeddings, members, catalog, config, id_column="object_id"):
    """Align by exact IDs; optional intersection is explicit and audited."""
    init_stack()
    config.validate()
    if not embeddings or set(embeddings) != set(members):
        raise ValueError("Need embeddings and metadata for exactly the same experiments")
    prepared, id_sets = {}, []
    for key in sorted(embeddings):
        if not {"run", "experiment"} <= members[key].keys():
            raise ValueError(f"Missing run/experiment metadata for {key}")
        for representation in config.representations:
            ids, x = embeddings[key][representation]
            ids = mla.exact_ids(ids)
            x = np.asarray(x, dtype=float)
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"Duplicate embedding IDs in {key}/{representation}")
            if (x.ndim != 2 or len(x) != len(ids) or x.shape[1] < 1
                    or not np.isfinite(x).all()):
                raise ValueError(f"Invalid or nonfinite vectors in {key}/{representation}")
            prepared[key, representation] = (ids, x)
            id_sets.append(set(ids))
    common = set.intersection(*id_sets)
    if config.cohort_mode == "strict" and any(s != common for s in id_sets):
        raise ValueError("Embedding ID sets differ; use --cohort-mode intersection to audit a common subset")
    if not common:
        raise ValueError("No common object IDs across experiments and representations")
    ids = np.array(sorted(common))
    spaces, coverage_rows = {}, []
    for (key, representation), (source_ids, x) in prepared.items():
        lookup = {oid: i for i, oid in enumerate(source_ids)}
        aligned = np.ascontiguousarray(x[[lookup[oid] for oid in ids]])
        if not np.any(aligned != aligned[0]):
            raise ValueError(f"Zero-scale common embedding in {key}/{representation}")
        spaces[key, representation] = aligned
        coverage_rows.append({"key": key, "representation": representation,
                              "source_objects": len(source_ids), "common_objects": len(ids),
                              "dropped_objects": len(source_ids) - len(ids), "dimensions": x.shape[1]})
    catalog = catalog.copy()
    catalog[id_column] = mla.exact_ids(catalog[id_column])
    if not common <= set(catalog[id_column]):
        raise ValueError("Missing IDs in catalog")
    values = pd.to_numeric(catalog[config.target_column], errors="coerce")
    known_rows = pd.Series(np.isfinite(values), index=catalog.index)
    if known_rows.groupby(catalog[id_column]).nunique().gt(1).any():
        raise ValueError("Repeated catalog rows disagree on whether the label is known")
    overlay = {"key": config.target_column, "min_value": 0., "include_min": True,
               "max_value": config.cutoff_gyr, "include_max": True}
    known = sm.overlay_evaluation_mask({"rubin_ids": ids}, catalog, overlay, id_column)
    targets = sm.overlay_labeled_mask({"rubin_ids": ids}, catalog, overlay, id_column) & known
    coverage = {"objects": len(ids), "known_objects": int(known.sum()),
                "unknown_objects": int((~known).sum()), "targets": int(targets.sum()),
                "prevalence": float(targets.sum() / known.sum()) if known.any() else None,
                "runs": sorted({str(m["run"]) for m in members.values()}),
                "experiments": len(members), "spaces": coverage_rows}
    return ConsensusCohort(ids, spaces, members, known, targets, coverage)


def load_manifest(path, config):
    init_stack()
    path = Path(path).expanduser().resolve()
    manifest = json.loads(path.read_text())
    resolve = lambda value: (path.parent / Path(value).expanduser()).resolve()
    catalog_path = resolve(manifest["catalog"])
    id_column = manifest.get("catalog_id_column", "object_id")
    if catalog_path.suffix.lower() == ".csv":
        catalog = pd.read_csv(catalog_path, dtype={id_column: str})
    else:
        from astropy.table import Table
        catalog = Table.read(catalog_path)[[id_column, config.target_column]].to_pandas()
    embeddings, members, inputs = {}, {}, {path, catalog_path}
    identities = set()
    for item in manifest["experiments"]:
        key = str(item["key"])
        identity = (str(item["run"]), str(item["experiment"]))
        if key in members or identity in identities:
            raise ValueError(f"Duplicate experiment identity: {key}")
        identities.add(identity)
        members[key] = {"run": identity[0], "experiment": identity[1],
                        "umap_params": item.get("umap_params", {})}
        embeddings[key] = {}
        for representation in config.representations:
            directory = resolve(item["inference_dir" if representation == "latent" else "umap_dir"])
            embeddings[key][representation] = mla.load_batches(directory)
            inputs.update(mla.batch_files(directory))
        inputs.update(resolve(p) for p in item.get("source_configs", []))
    cohort = build_cohort(embeddings, members, catalog, config, id_column)
    fingerprints = [{"path": str(p), "sha256": mla.sha256_file(p)} for p in sorted(inputs)]
    return cohort, fingerprints


def make_partition(cohort, config, split, shuffled=False):
    """Paired 25/25/50 partitions; the last half is never used for selection."""
    init_stack()
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, int(shuffled), split]))
    y = cohort.targets.copy()
    if shuffled:
        y[cohort.known] = rng.permutation(y[cohort.known])
    partition = np.full(len(y), -1, dtype=np.int8)
    for label in (False, True):
        indices = np.flatnonzero(cohort.known & (y == label))
        counts = sm._largest_remainder_counts(len(indices), [.25, .25, .5])
        if min(counts) < 1:
            raise ValueError("Each class needs enough objects for train, validation, and test partitions")
        for code, part in enumerate(sm._partition_shuffled_indices(indices, counts, rng)):
            partition[part] = code
    if "anchor" in config.algorithms and np.sum(y & (partition == 0)) < max(config.neighbor_ranks):
        raise ValueError("Too few training mergers for requested neighbor_ranks")
    return partition, y


def threshold_from_validation(scores, targets):
    """F1 threshold with deterministic purity/recall/threshold tie breaking."""
    # The existing curve routine handles ties together; translate high scores to distances.
    maximum = float(np.max(scores))
    curve, summary = sm.distance_completeness_purity_curve(maximum - scores, targets)
    point = sm.select_distance_operating_point(curve)
    threshold = maximum - point["distance_threshold"]
    # The round trip through distances can shift by an ulp. Use an observed score.
    threshold = float(np.sort(np.unique(scores))[np.argmin(np.abs(np.sort(np.unique(scores)) - threshold))])
    return threshold, float(summary["average_precision"])


def fit_candidate(x, algorithm, train_y, config, seed):
    """Fit only on training vectors and labels; return a query scorer."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    if algorithm == "anchor":
        # One scalar scale preserves the original Euclidean latent geometry.
        center = x.mean(axis=0)
        scale = float(np.sqrt(np.mean(np.sum((x - center)**2, axis=1))))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Zero-scale training embedding")
        model = NearestNeighbors(n_neighbors=max(config.neighbor_ranks), n_jobs=1)
        model.fit((x[train_y] - center) / scale)

        def score(query):
            distances = model.kneighbors((query - center) / scale, return_distance=True)[0]
            return {str(k): -distances[:, k - 1] for k in sorted(config.neighbor_ranks)}
        return score
    if algorithm == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1., class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=seed))
    elif algorithm == "random_forest":
        model = RandomForestClassifier(n_estimators=config.forest_trees, min_samples_leaf=5,
                                       class_weight="balanced_subsample", random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    model.fit(x, train_y)
    return lambda query: {"fixed": model.predict_proba(query)[:, 1]}


def rank_experiments(records, top_experiments):
    """Only validation AP enters selection, independently per run/space/algorithm."""
    frame = pd.DataFrame(records)
    columns = ["run", "representation", "algorithm"]
    frame = frame.sort_values(columns + ["validation_ap", "key"],
                              ascending=[True, True, True, False, True], kind="stable")
    frame["validation_rank"] = frame.groupby(columns).cumcount() + 1
    frame["selected"] = frame.validation_rank.le(top_experiments)
    return frame


def summarize_performance(y, score, selected):
    from sklearn.metrics import average_precision_score
    tp = int(np.sum(y & selected))
    n_selected = int(selected.sum())
    purity = tp / n_selected if n_selected else 0.
    completeness = tp / int(y.sum())
    return {"n_test": len(y), "n_targets": int(y.sum()), "n_selected": n_selected,
            "average_precision": float(average_precision_score(y, score)),
            "purity": purity, "completeness": completeness,
            "f1": 2 * tp / (n_selected + int(y.sum())), "prevalence": float(y.mean())}


def aggregate_split(selected, test_scores, config):
    """Average experiments, then algorithms, then runs; never pool unequal runs."""
    algorithm_groups, run_groups, total_groups = {}, {}, {}
    for group, frame in selected.groupby(["representation", "run", "algorithm"], sort=True):
        votes, ranks = [], []
        for row in frame.itertuples():
            score, percentile = test_scores[row.key, row.representation, row.algorithm]
            votes.append(score >= row.threshold)
            ranks.append(percentile)
        algorithm_groups[group] = (np.mean(votes, axis=0), np.mean(ranks, axis=0),
                                    np.all(votes, axis=0))
    for representation, run in sorted({key[:2] for key in algorithm_groups}):
        values = [value for key, value in algorithm_groups.items() if key[:2] == (representation, run)]
        run_groups[representation, run] = (np.mean([v[0] for v in values], axis=0),
                                           np.mean([v[1] for v in values], axis=0),
                                           np.all([v[2] for v in values], axis=0))
    for representation in config.representations:
        values = [value for key, value in run_groups.items() if key[0] == representation]
        total_groups[representation] = (np.mean([v[0] for v in values], axis=0),
                                        np.mean([v[1] for v in values], axis=0),
                                        np.all([v[2] for v in values], axis=0),
                                        np.all([v[0] >= config.consensus_threshold for v in values], axis=0))
    return algorithm_groups, run_groups, total_groups


def evaluate(cohort, config, shuffled=False, progress=True):
    """Keep a compact, exact-ID audit of every selected held-out decision."""
    init_stack()
    config.validate()
    n = len(cohort.ids)
    candidates = [(key, representation, algorithm) for key, representation in sorted(cohort.spaces)
                  for algorithm in config.algorithms]
    audit = {"object_ids": cohort.ids, "candidate_keys": np.array([c[0] for c in candidates]),
             "representations": np.array([c[1] for c in candidates]),
             "algorithms": np.array([c[2] for c in candidates]),
             "partitions": np.full((config.n_splits, n), -1, dtype=np.int8),
             "working_targets": np.zeros((config.n_splits, n), dtype=bool),
             "selected_models": np.zeros((config.n_splits, len(candidates)), dtype=bool),
             "thresholds": np.full((config.n_splits, len(candidates)), np.nan),
             "test_scores": np.full((config.n_splits, len(candidates), n), np.nan, dtype=np.float32),
             "test_votes": np.full((config.n_splits, len(candidates), n), -1, dtype=np.int8)}
    selections, metrics = [], []
    accumulators = {"algorithm": {}, "run": {}, "consensus": {}}

    def accumulate(level, group, indices, values):
        store = accumulators[level].setdefault(group, np.zeros((5, n)))
        store[0, indices] += 1
        for i, value in enumerate(values, start=1):
            store[i, indices] += value

    for split in range(config.n_splits):
        p, y = make_partition(cohort, config, split, shuffled)
        train, validation, test = [p == code for code in range(3)]
        test_indices = np.flatnonzero(test)
        audit["partitions"][split], audit["working_targets"][split] = p, y
        records, query_scores = [], {}
        for ci, (key, representation, algorithm) in enumerate(candidates):
            x = cohort.spaces[key, representation]
            seed = int(np.random.SeedSequence([config.seed, int(shuffled), split]).generate_state(1)[0])
            scorer = fit_candidate(x[train], algorithm, y[train], config, seed)
            val_scores = scorer(x[validation])
            choices = []
            for parameter, scores in val_scores.items():
                threshold, ap = threshold_from_validation(scores, y[validation])
                choices.append((ap, parameter, threshold))
            # Fixed models have one choice; anchor ties prefer the smaller k.
            choices.sort(key=lambda c: (-c[0], int(c[1]) if c[1] != "fixed" else 0))
            ap, parameter, threshold = choices[0]
            scores = scorer(x[test])[parameter]
            # Percentiles align score scales; they are rankings, never probabilities.
            percentile = np.searchsorted(np.sort(val_scores[parameter]), scores, side="right") / validation.sum()
            query_scores[key, representation, algorithm] = (scores, percentile)
            record = {"split": split, "key": key, "run": str(cohort.members[key]["run"]),
                      "experiment": str(cohort.members[key]["experiment"]),
                      "representation": representation, "algorithm": algorithm,
                      "parameter": parameter, "validation_ap": ap, "threshold": threshold,
                      "n_train": int(train.sum()), "n_validation": int(validation.sum())}
            records.append(record)
            audit["thresholds"][split, ci] = threshold
            audit["test_scores"][split, ci, test] = scores
            audit["test_votes"][split, ci, test] = (scores >= threshold).astype(np.int8)
        ranked = rank_experiments(records, config.top_experiments)
        selections.append(ranked)
        chosen = ranked[ranked.selected]
        selected_keys = set(zip(chosen.key, chosen.representation, chosen.algorithm))
        audit["selected_models"][split] = [c in selected_keys for c in candidates]
        algorithm_groups, run_groups, total_groups = aggregate_split(chosen, query_scores, config)
        for level, groups in (("algorithm", algorithm_groups), ("run", run_groups), ("consensus", total_groups)):
            for group, values in groups.items():
                if isinstance(group, str):
                    group = (group,)
                accumulate(level, group, test_indices, values)
                row = {"split": split, "level": level, "representation": group[0],
                       "run": group[1] if len(group) > 1 else "all",
                       "algorithm": group[2] if len(group) > 2 else "all"}
                row.update(summarize_performance(y[test], values[1], values[0] >= config.consensus_threshold))
                metrics.append(row)
        if progress and (split == 0 or (split + 1) % 5 == 0 or split + 1 == config.n_splits):
            print(f"{'shuffled' if shuffled else 'observed'}: {split + 1}/{config.n_splits} splits", flush=True)

    tables = {"experiment_selections": pd.concat(selections, ignore_index=True),
              "split_metrics": pd.DataFrame(metrics)}
    for level, groups in accumulators.items():
        frames = []
        for group, store in groups.items():
            evaluated = store[0] > 0
            counts = store[0, evaluated]
            frame = pd.DataFrame({"object_id": cohort.ids[evaluated], "representation": group[0],
                                  "target": cohort.targets[evaluated], "n_holdouts": counts.astype(int),
                                  "vote_fraction": store[1, evaluated] / counts,
                                  "mean_rank_score": store[2, evaluated] / counts,
                                  "unanimous_split_fraction": store[3, evaluated] / counts,
                                  "sufficient_holdouts": counts >= config.min_holdouts})
            if level != "consensus":
                frame["run"] = group[1]
            if level == "algorithm":
                frame["algorithm"] = group[2]
            if level == "consensus":
                frame["all_runs_majority_fraction"] = store[4, evaluated] / counts
            frames.append(frame)
        tables[f"{level}_candidates"] = pd.concat(frames, ignore_index=True)
    return tables, audit


def finish_tables(tables, cohort, config):
    """Coverage, cross-run stability, algorithm shortlists, and run overlap."""
    init_stack()
    runs = tables["run_candidates"]
    consensus = tables["consensus_candidates"]
    stats = runs.groupby(["representation", "object_id"], sort=True).agg(
        n_runs=("run", "nunique"), minimum_run_vote_fraction=("vote_fraction", "min"),
        maximum_run_vote_fraction=("vote_fraction", "max"))
    consensus = consensus.merge(stats, on=["representation", "object_id"], validate="one_to_one")
    consensus["stable_across_runs"] = (consensus.n_runs.ge(2) & consensus.sufficient_holdouts
        & consensus.minimum_run_vote_fraction.ge(config.consensus_threshold))
    consensus["cross_run_tested"] = consensus.n_runs.ge(2)
    tables["consensus_candidates"] = consensus
    algo = tables["algorithm_candidates"]
    combined = algo.groupby(["representation", "algorithm", "object_id"], sort=True).agg(
        target=("target", "first"), n_runs=("run", "nunique"), n_holdouts=("n_holdouts", "min"),
        vote_fraction=("vote_fraction", "mean"), mean_rank_score=("mean_rank_score", "mean"))
    combined = combined.reset_index()
    combined["sufficient_holdouts"] = combined.n_holdouts.ge(config.min_holdouts)
    tables["algorithm_consensus"] = combined
    for name in ("algorithm_candidates", "run_candidates", "consensus_candidates", "algorithm_consensus"):
        tables[name] = tables[name].sort_values(
            ["representation", "sufficient_holdouts", "vote_fraction", "mean_rank_score", "object_id"],
            ascending=[True, False, False, False, True], kind="stable").reset_index(drop=True)
    tables["top_candidates_per_algorithm"] = tables["algorithm_consensus"].groupby(
        ["representation", "algorithm"], sort=False).head(3).reset_index(drop=True)
    selections = tables["experiment_selections"]
    tables["experiment_selection_frequency"] = selections.groupby(
        ["run", "representation", "algorithm", "key"], sort=True).agg(
            splits=("selected", "size"), times_selected=("selected", "sum"),
            median_validation_ap=("validation_ap", "median")).reset_index()
    frequency = tables["experiment_selection_frequency"]
    frequency["selection_fraction"] = frequency.times_selected / frequency.splits
    pairs = []
    for representation, frame in runs.groupby("representation"):
        pivot = frame.pivot(index="object_id", columns="run", values="vote_fraction")
        for a, b in combinations(pivot.columns, 2):
            common = pivot[[a, b]].dropna()
            left, right = [set(common.index[common[column] >= config.consensus_threshold]) for column in (a, b)]
            union = left | right
            varying = common[a].nunique() > 1 and common[b].nunique() > 1
            pairs.append({"representation": representation, "run_a": a, "run_b": b,
                          "n_common": len(common), "n_selected_a": len(left), "n_selected_b": len(right),
                          "n_intersection": len(left & right), "n_union": len(union),
                          "jaccard": len(left & right) / len(union) if union else np.nan,
                          "spearman": common[a].corr(common[b], method="spearman") if varying else np.nan})
    tables["run_agreement"] = pd.DataFrame(pairs, columns=[
        "representation", "run_a", "run_b", "n_common", "n_selected_a", "n_selected_b",
        "n_intersection", "n_union", "jaccard", "spearman"])
    tables["performance_summary"] = mla.summarize_rows(tables["split_metrics"],
        ["level", "representation", "run", "algorithm"],
        ["average_precision", "purity", "completeness", "f1", "prevalence"])
    return tables


def write_report(output_dir, cohort, config, tables):
    rows = ["# Merger consensus", "",
            f"Target: `0 <= {config.target_column} <= {config.cutoff_gyr:g}` Gyr. "
            f"{cohort.coverage['known_objects']:,} known objects; {cohort.coverage['targets']:,} targets.", "",
            f"{config.n_splits} paired repetitions: 25% training, 25% validation, 50% held out. "
            f"Select up to {config.top_experiments} experiments **per run, representation, and algorithm** using validation AP. "
            "Validation F1 fixes each classifier's threshold. No classifier is refitted on validation or test objects.", "",
            "Votes are averaged over selected experiments, then equally over algorithms, then equally over runs. "
            "A galaxy's vote fraction averages only repetitions where it was held out. It is agreement, not a merger probability. "
            "Mean rank scores use each classifier's validation score CDF and also are not probabilities.", ""]
    if len(cohort.coverage["runs"]) < 2:
        rows += ["**Cross-run consistency has not been tested: this bundle contains only one run.**", ""]
    for representation in config.representations:
        frame = tables["consensus_candidates"].query("representation == @representation")
        rows += [f"## {representation.capitalize()}", "",
                 f"{int(frame.stable_across_runs.sum())} galaxies meet the cross-run stability rule; "
                 f"{int((~frame.sufficient_holdouts).sum())} scored galaxies have fewer than {config.min_holdouts} holdouts.", "",
                 "| Object ID | Vote fraction | Lowest run vote | Unanimous splits | Holdouts | Catalog target |",
                 "|---|---:|---:|---:|---:|---|"]
        for row in frame.head(10).itertuples():
            rows.append(f"| {row.object_id} | {row.vote_fraction:.3f} | {row.minimum_run_vote_fraction:.3f} "
                        f"| {row.unanimous_split_fraction:.3f} | {row.n_holdouts} | {row.target} |")
        metrics = tables["split_metrics"].query("level == 'consensus' and representation == @representation")
        rows += ["", f"Median held-out AP: {metrics.average_precision.median():.4f}; "
                 f"purity: {metrics.purity.median():.1%}; completeness: {metrics.completeness.median():.1%}.", ""]
    rows += ["## Reading the outputs", "",
             "- `top_candidates_per_algorithm.csv`: top three galaxies for each algorithm and representation, averaged equally across runs.",
             "- `consensus_candidates.csv`: complete consensus ranking with run coverage and stability flags.",
             "- `run_candidates.csv`, `algorithm_candidates.csv`: the contributions behind that ranking.",
             "- `experiment_selections.csv`: every validation rank, selected neighbor k, threshold, and top-experiment decision.",
             "- `prediction_audit.npz`: exact IDs, partitions, selected model masks, thresholds, scores, and exact binary votes. "
             "Votes are -1 outside test sets. Stored float32 scores are approximate; exact votes are authoritative.", "",
             f"Stability requires at least two runs, at least {config.min_holdouts} holdout appearances, "
             f"and average vote >= {config.consensus_threshold:g} in **every** included run. "
             "This is distinct from unanimous classifier agreement, reported separately.", "",
             "## Limits", "",
             "Repeated holdouts overlap. Vote frequencies and split quantiles are descriptive, not independent probabilities or confidence intervals. "
             "Representations were previously trained, and existing UMAPs may have been fit using held-out objects without labels; "
             "this is a transductive, exploratory label-holdout analysis. It does not establish transfer to a new simulation or snapshot. "
             "Experiments preselected using earlier full-sample results retain selection optimism. "
             "Changing the target requires rerunning experiment selection and thresholds. "
             "Unknown labels are excluded and zero-holdout objects remain visible in cohort.csv. "
             "An empty selected set has purity reported as zero in this workflow.", ""]
    (output_dir / "meeting_brief.md").write_text("\n".join(rows))


def render_figures(output_dir, cohort, config, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(config.representations), figsize=(7 * len(config.representations), 5), squeeze=False)
    for ax, representation in zip(axes[0], config.representations):
        frame = tables["consensus_candidates"].query("representation == @representation")
        for target, label, color in ((False, "Other known galaxies", "#778899"), (True, "Catalog mergers", "#cd6538")):
            values = frame.loc[frame.target.eq(target), "vote_fraction"]
            ax.hist(values, bins=np.linspace(0, 1, 21), density=True, histtype="step", linewidth=2, color=color, label=label)
        ax.set(title=f"{representation.capitalize()} consensus ({len(cohort.coverage['runs'])} run(s))",
               xlabel="Held-out merger vote fraction (not probability)", ylabel="Density")
        ax.legend()
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"consensus_distribution.{suffix}", dpi=160)
    plt.close(fig)


def run_analysis(manifest, output_dir, config=None, figures=True, shuffled_control=False):
    init_stack()
    from threadpoolctl import threadpool_limits

    config = config or ConsensusConfig()
    config.validate()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty; previous results will not be overwritten")
    cohort, fingerprints = load_manifest(manifest, config)
    make_partition(cohort, config, 0)  # Fail before creating an incomplete result directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(cohort.coverage, indent=2), flush=True)
    with threadpool_limits(limits=2):
        tables, audit = evaluate(cohort, config)
        finish_tables(tables, cohort, config)
        np.savez_compressed(output_dir / "prediction_audit.npz", **audit)
        holdouts = np.sum(audit["partitions"] == 2, axis=0)
        pd.DataFrame({"object_id": cohort.ids, "known_label": cohort.known, "target": cohort.targets,
                      "n_holdouts": holdouts, "sufficient_holdouts": holdouts >= config.min_holdouts}).to_csv(
                          output_dir / "cohort.csv", index=False)
        for name, frame in tables.items():
            frame.to_csv(output_dir / f"{name}.csv", index=False)
        if shuffled_control:
            control, control_audit = evaluate(cohort, config, shuffled=True)
            control["split_metrics"].to_csv(output_dir / "shuffled_split_metrics.csv", index=False)
            control["experiment_selections"].to_csv(output_dir / "shuffled_experiment_selections.csv", index=False)
            np.savez_compressed(output_dir / "shuffled_prediction_audit.npz", **control_audit)
        if figures:
            render_figures(output_dir, cohort, config, tables)
    write_report(output_dir, cohort, config, tables)
    record = {"status": "complete", "config": asdict(config), "coverage": cohort.coverage,
              "inputs": fingerprints, "source_sha256": {Path(p).name: mla.sha256_file(p)
                  for p in (__file__, mla.__file__, sm.__file__)},
              "python": sys.version, "versions": {p: metadata.version(p) for p in
                  ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")},
              "partition_codes": {"unknown": -1, "train": 0, "validation": 1, "test": 2},
              "fractions": {"train": .25, "validation": .25, "test": .5},
              "shuffled_control": shuffled_control, "cross_run_tested": len(cohort.coverage["runs"]) >= 2,
              "score_direction": "higher is more merger-like", "weighting": "experiments -> algorithms -> runs",
              "zero_holdout_objects": int(np.sum(cohort.known & (holdouts == 0)))}
    # Written last, so a directory with failed fits cannot look complete.
    (output_dir / "analysis_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return {"cohort": cohort, "tables": tables, "output_dir": output_dir}


def discover_manifest(run_base, catalog, output, runs=None):
    """Read existing configs/logs only. Record failures instead of inventing paths."""
    from umap_parameter_explorer import ExperimentRef, discover_experiment
    import tomllib

    run_base, output = Path(run_base).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    experiments, failures = [], []
    for path in sorted(run_base.glob("run*/udb*.toml")):
        match = re.fullmatch(r"udb(\d+)_(\d+)\.toml", path.name)
        if not match:
            continue
        run, experiment = map(int, match.groups())
        if runs is not None and run not in runs:
            continue
        try:
            artifact = discover_experiment(ExperimentRef(run, experiment), run_base)
            config = tomllib.loads(path.read_text())
            params = config.get("umap.UMAP", config.get("umap", {}).get("UMAP", {}))
            experiments.append({"key": artifact.ref.key, "run": run, "experiment": experiment,
                                "inference_dir": str(artifact.inference_dir), "umap_dir": str(artifact.historical_umap_dir),
                                "umap_params": params, "source_configs": [str(path), str(artifact.log_path)]})
        except (RuntimeError, ValueError, OSError, KeyError) as exc:
            failures.append({"key": f"run{run}_expt{experiment}", "reason": str(exc)})
    payload = {"catalog": str(Path(catalog).expanduser().resolve()), "catalog_id_column": "object_id",
               "experiments": experiments, "discovery_failures": failures}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Discovered {len(experiments)} experiments; {len(failures)} failures. Manifest: {output}")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover", help="Inventory existing Hyrax configs/logs without submitting jobs")
    discover.add_argument("--profile", choices=("local", "delta"), default="local")
    discover.add_argument("--run-base", type=Path)
    discover.add_argument("--catalog", type=Path)
    discover.add_argument("--runs", type=int, nargs="+")
    discover.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="Select experiments and build held-out galaxy consensus")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--n-splits", type=int, default=100)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--top-experiments", type=int, default=3)
    run.add_argument("--algorithms", choices=ALGORITHMS, nargs="+", default=list(ALGORITHMS))
    run.add_argument("--representations", choices=REPRESENTATIONS, nargs="+", default=list(REPRESENTATIONS))
    run.add_argument("--neighbor-ranks", type=int, nargs="+", default=[3, 5, 7, 9])
    run.add_argument("--target-column", default="Major_TimeSinceMerger")
    run.add_argument("--merger-type", choices=("major", "minor", "mini"),
                     help="Convenience option for <Type>_TimeSinceMerger; overrides --target-column")
    run.add_argument("--cutoff-gyr", type=float, default=.75)
    run.add_argument("--consensus-threshold", type=float, default=.5)
    run.add_argument("--min-holdouts", type=int, default=5)
    run.add_argument("--cohort-mode", choices=("strict", "intersection"), default="strict")
    run.add_argument("--forest-trees", type=int, default=128)
    run.add_argument("--shuffled-control", action="store_true", help="Rerun the entire selection with shuffled labels")
    run.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "discover":
        from research_paths import load_paths
        paths = load_paths(args.profile)
        discover_manifest(args.run_base or paths.hyrax_runs, args.catalog or paths.catalog("raw_merger_flags"),
                          args.output, args.runs)
        return 0
    names = ConsensusConfig.__dataclass_fields__
    values = {name: getattr(args, name) for name in names}
    for name in ("algorithms", "representations", "neighbor_ranks"):
        values[name] = tuple(values[name])
    if args.merger_type:
        values["target_column"] = args.merger_type.title() + "_TimeSinceMerger"
    run_analysis(args.manifest, args.output_dir, ConsensusConfig(**values),
                 figures=not args.no_figures, shuffled_control=args.shuffled_control)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
