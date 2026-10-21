"""Paired multi-experiment merger retrieval; run --help for the manifest CLI.

All statistical evaluation uses existing representations. The optional new UMAP
is descriptive only. Scientific imports stay lazy for lightweight CLI discovery.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping


np = pd = sm = None


def init_stack():
    global np, pd, sm
    if np is None:
        import numpy as np
        import pandas as pd
        import static_umap_metrics as sm

        sm.init_tabular_stack()


@dataclass(frozen=True)
class AnalysisConfig:
    n_splits: int = 100
    seed: int = 42
    neighbor_ranks: tuple[int, ...] = (3, 5, 7, 9)
    train_fraction: float = 0.5
    validation_fraction: float = 0.25
    target_column: str = "Major_TimeSinceMerger"
    cutoff_gyr: float = 0.75
    reference: str = "run11_expt9"
    cka_sample_size: int = 2000

    def validate(self):
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, int) or self.n_splits < 1:
            raise ValueError("n_splits must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (not self.neighbor_ranks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1
                                           for k in self.neighbor_ranks)):
            raise ValueError("neighbor_ranks must contain positive integers")
        if len(set(self.neighbor_ranks)) != len(self.neighbor_ranks):
            raise ValueError("neighbor_ranks must be distinct")
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1
                and self.train_fraction + self.validation_fraction < 1):
            raise ValueError("train/validation/test fractions must each be positive")
        if not math.isfinite(self.cutoff_gyr) or self.cutoff_gyr < 0:
            raise ValueError("cutoff_gyr must be finite and nonnegative")
        if self.cka_sample_size < 2:
            raise ValueError("cka_sample_size must be at least two")


@dataclass
class Cohort:
    ids: Any
    latents: dict[str, Any]
    umaps: dict[str, Any]
    known: Any
    targets: Any
    coverage: dict[str, Any]


def exact_ids(values):
    """Canonical identifiers without coercing integers through floating point."""
    init_stack()
    raw = np.asarray(values)
    if raw.ndim != 1 or not len(raw):
        raise ValueError("IDs must be a nonempty one-dimensional array")
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Boolean object IDs are invalid")
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value) or abs(value) >= 2**53 or value != int(value):
                raise ValueError("Unsafe floating-point object ID; supply exact strings or integers")
    normalized = sm.normalize_object_ids(raw)
    if normalized.isna().any():
        raise ValueError("Missing or blank object IDs")
    return normalized.to_numpy(dtype=str)


def build_cohort(embeddings, catalog, config, catalog_id_column="object_id"):
    """Strict common-cohort join; each member contains latent/umap (IDs, X)."""
    init_stack()
    config.validate()
    if len(embeddings) < 2 or config.reference not in embeddings:
        raise ValueError("Need at least two experiments including the configured reference")
    reference_ids = exact_ids(embeddings[config.reference]["latent"][0])
    ids = np.sort(reference_ids)
    id_set = set(ids)
    latents, umaps, dimensions = {}, {}, {}
    for name, member in embeddings.items():
        for kind, destination in [("latent", latents), ("umap", umaps)]:
            member_ids = exact_ids(member[kind][0])
            if len(np.unique(member_ids)) != len(member_ids):
                raise ValueError(f"Duplicate embedding IDs: {name}/{kind}")
            if set(member_ids) != id_set:
                raise ValueError(f"Missing or extra IDs in {name}/{kind}; strict common cohort required")
            x = np.asarray(member[kind][1], dtype=float)
            if x.ndim != 2 or len(x) != len(member_ids) or x.shape[1] < 1:
                raise ValueError(f"Invalid embedding shape in {name}/{kind}: {x.shape}")
            if kind == "umap" and x.shape[1] != 2:
                raise ValueError(f"UMAP must have exactly two columns: {name}")
            if not np.isfinite(x).all():
                raise ValueError(f"Nonfinite embedding values: {name}/{kind}")
            lookup = {oid: i for i, oid in enumerate(member_ids)}
            x = np.ascontiguousarray(x[[lookup[oid] for oid in ids]])
            if not np.any(x != x[0]):
                raise ValueError(f"Zero-scale embedding: {name}/{kind}")
            destination[name] = x
        dimensions[name] = latents[name].shape[1]

    catalog = catalog.copy()
    catalog[catalog_id_column] = exact_ids(catalog[catalog_id_column])
    if not id_set <= set(catalog[catalog_id_column]):
        raise ValueError("Missing IDs in catalog")
    values = pd.to_numeric(catalog[config.target_column], errors="coerce")
    # A mixture of known and unknown duplicate labels must not be resolved silently.
    known_rows = pd.Series(np.isfinite(values), index=catalog.index)
    if known_rows.groupby(catalog[catalog_id_column]).nunique().gt(1).any():
        raise ValueError("Repeated catalog rows disagree on whether the label is known")
    overlay = {"key": config.target_column, "min_value": 0.0, "include_min": True,
               "max_value": config.cutoff_gyr, "include_max": True}
    umap_data = {"rubin_ids": ids}
    known = sm.overlay_evaluation_mask(umap_data, catalog, overlay, catalog_id_column)
    targets = sm.overlay_labeled_mask(umap_data, catalog, overlay, catalog_id_column) & known
    coverage = {"objects": len(ids), "catalog_rows": len(catalog),
                "catalog_unique_ids": int(catalog[catalog_id_column].nunique()),
                "known_objects": int(known.sum()), "unknown_objects": int((~known).sum()),
                "target_objects": int(targets.sum()), "latent_dimensions": dimensions,
                "label_rule": f"0 <= {config.target_column} <= {config.cutoff_gyr}",
                "prevalence": float(targets.sum() / known.sum()) if known.any() else None}
    return Cohort(ids, latents, umaps, known, targets, coverage)


def batch_files(directory):
    files = [p for p in Path(directory).glob("batch_*.npy")
             if re.fullmatch(r"batch_\d+\.npy", p.name)]
    if not files:
        raise FileNotFoundError(f"No numeric Hyrax batches in {directory}")
    return sorted(files, key=lambda p: int(p.stem.split("_")[-1]))


def load_batches(directory):
    init_stack()
    arrays = [np.load(p, allow_pickle=False) for p in batch_files(directory)]
    if any(not {"id", "tensor"} <= set(a.dtype.names or ()) for a in arrays):
        raise ValueError(f"Every batch must expose id and tensor fields: {directory}")
    ids = np.concatenate([a["id"] for a in arrays])
    x = np.concatenate([a["tensor"] for a in arrays])
    return ids, x.reshape(len(ids), -1)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path, config):
    init_stack()
    from astropy.table import Table

    path = Path(path).resolve()
    manifest = json.loads(path.read_text())

    def resolve(value):
        return (path.parent / Path(value).expanduser()).resolve()

    catalog_path = resolve(manifest["catalog"])
    id_column = manifest.get("catalog_id_column", "object_id")
    # Read only the two required columns; preserve the identifier dtype.
    table = Table.read(catalog_path)
    catalog = table[[id_column, config.target_column]].to_pandas()
    embeddings, inputs = {}, [path, catalog_path]
    for member in manifest["experiments"]:
        key = member["key"]
        if key in embeddings:
            raise ValueError(f"Duplicate experiment key: {key}")
        embeddings[key] = {}
        for kind, field in [("latent", "inference_dir"), ("umap", "umap_dir")]:
            directory = resolve(member[field])
            embeddings[key][kind] = load_batches(directory)
            inputs.extend(batch_files(directory))
        inputs.extend(resolve(v) for v in member.get("source_configs", []))
    fingerprints = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                    for p in inputs]
    return build_cohort(embeddings, catalog, config, id_column), fingerprints


def fit_normalization(latents, train_mask):
    init_stack()
    if not np.asarray(train_mask, dtype=bool).any():
        raise ValueError("Empty normalization training set")
    stats = {}
    for name, x in latents.items():
        training = x[train_mask]
        mean = training.mean(axis=0)
        scale = float(np.sqrt(np.mean(np.sum((training - mean) ** 2, axis=1))))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Zero-scale training embedding: {name}")
        stats[name] = (mean, scale)
    return stats


def combine_latents(latents, stats):
    init_stack()
    return np.concatenate([(x - stats[name][0]) / (stats[name][1] * np.sqrt(len(latents)))
                           for name, x in latents.items()], axis=1)


def make_partition(targets, known, config, split_index, shuffled=False):
    """One partition shared by every method; unknown labels have partition -1."""
    init_stack()
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, int(shuffled), split_index]))
    working = np.array(targets, copy=True)
    if shuffled:
        working[known] = rng.permutation(working[known])
    fractions = [config.train_fraction, config.validation_fraction,
                 1 - config.train_fraction - config.validation_fraction]
    partitions = np.full(len(targets), -1, dtype=np.int8)
    for label in [False, True]:
        indices = np.flatnonzero(known & (working == label))
        counts = sm._largest_remainder_counts(len(indices), fractions)
        parts = sm._partition_shuffled_indices(indices, counts, rng)
        for code, part in enumerate(parts):
            partitions[part] = code
    if np.sum((partitions == 0) & working) < max(config.neighbor_ranks):
        raise ValueError("Too few training mergers for the maximum requested neighbor rank")
    return partitions, working


def percentile_transform(validation_distances, distances):
    """Right-continuous empirical CDF fitted on validation scores only."""
    init_stack()
    reference = np.sort(np.asarray(validation_distances))
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("Percentile calibration requires finite validation scores")
    return np.searchsorted(reference, distances, side="right") / len(reference)


def score_candidates(cohort, partition, training_targets, config):
    """Scores need training labels only; validation/test labels are unavailable."""
    init_stack()
    from scipy.spatial.distance import cdist

    train = partition == 0
    validation = partition == 1
    queries = np.flatnonzero((partition == 1) | (partition == 2))
    anchors = np.flatnonzero(train & training_targets)
    ranks = np.array(sorted(config.neighbor_ranks)) - 1
    if len(anchors) < max(config.neighbor_ranks):
        raise ValueError("Too few training merger anchors")
    stats = fit_normalization(cohort.latents, train)
    scores = {}
    fused_squared = np.zeros((len(queries), len(anchors)))

    def ranked(distance):
        result = np.full((len(cohort.ids), len(ranks)), np.nan)
        result[queries] = np.partition(distance, ranks, axis=1)[:, ranks]
        return result

    for name, x in cohort.latents.items():
        squared = cdist(x[queries], x[anchors], metric="sqeuclidean") / stats[name][1] ** 2
        fused_squared += squared / len(cohort.latents)
        scores[f"latent:{name}"] = ranked(np.sqrt(squared))
    scores["latent:ensemble"] = ranked(np.sqrt(fused_squared))
    percentile_scores = []
    for name, x in cohort.umaps.items():
        distance = cdist(x[queries], x[anchors], metric="euclidean")
        member_scores = ranked(distance)
        scores[f"umap:{name}"] = member_scores
        transformed = np.full_like(member_scores, np.nan)
        for j in range(len(ranks)):
            transformed[queries, j] = percentile_transform(member_scores[validation, j], member_scores[queries, j])
        percentile_scores.append(transformed)
    scores["umap:ensemble"] = np.mean(percentile_scores, axis=0)
    return scores, stats


def select_candidate(scores, validation_targets, validation_mask, ranks):
    """Only validation labels may enter this selection function."""
    init_stack()
    candidates = []
    for j, rank in enumerate(sorted(ranks)):
        curve, summary = sm.distance_completeness_purity_curve(
            scores[:, j], validation_targets, validation_mask)
        point = sm.select_distance_operating_point(curve, objective="max_f1")
        candidates.append({"rank": rank, "column": j, "validation_ap": summary["average_precision"],
                           "threshold": point["distance_threshold"]})
    return sorted(candidates, key=lambda c: (-c["validation_ap"], c["rank"]))[0]


def summarize_rows(rows, group_columns, value_columns):
    init_stack()
    records = []
    for group, frame in rows.groupby(group_columns, sort=False):
        group = group if isinstance(group, tuple) else (group,)
        record = dict(zip(group_columns, group))
        for value in value_columns:
            finite = frame[value].dropna().to_numpy(dtype=float)
            for suffix, q in [("", 0.5), ("_p16", 0.16), ("_p84", 0.84)]:
                record[value + suffix] = float(np.quantile(finite, q)) if len(finite) else np.nan
        record["n_splits"] = len(frame)
        records.append(record)
    return pd.DataFrame(records)


def evaluate(cohort, config, progress=True):
    init_stack()
    config.validate()
    members = list(cohort.latents)
    methods = [f"{kind}:{name}" for kind in ["latent", "umap"]
               for name in members + ["ensemble", "best_single"]]
    n, count = len(cohort.ids), config.n_splits
    partitions = np.full((2, count, n), -1, dtype=np.int8)
    labels = np.zeros((2, count, n), dtype=bool)
    test_scores = np.full((2, count, len(methods), n), np.nan, dtype=np.float32)
    split_rows, curve_rows, jaccard_rows, selection_rows, normalization_rows = [], [], [], [], []
    grid = np.linspace(0, 1, 101)
    start = time.monotonic()
    for mode_index, mode in enumerate(["observed", "shuffled"]):
        for split in range(count):
            partition, working = make_partition(cohort.targets, cohort.known, config, split, mode_index == 1)
            partitions[mode_index, split] = partition
            labels[mode_index, split] = working
            train, validation, test = [partition == i for i in range(3)]
            # Passing masked labels enforces the information boundary in code.
            scores, stats = score_candidates(cohort, partition, working & train, config)
            selected = {method: select_candidate(value, working & validation, validation, config.neighbor_ranks)
                        for method, value in scores.items()}
            for name, (_, scale) in stats.items():
                normalization_rows.append({"mode": mode, "split": split, "experiment": name, "rms_scale": scale})
            for method, choice in selected.items():
                selection_rows.append({"mode": mode, "split": split, "method": method, **choice})
            for kind in ["latent", "umap"]:
                eligible = [f"{kind}:{name}" for name in members]
                winner = sorted(eligible, key=lambda m: (-selected[m]["validation_ap"], m))[0]
                scores[f"{kind}:best_single"] = scores[winner]
                selected[f"{kind}:best_single"] = {**selected[winner], "selected_member": winner}
            top_sets = {}
            for mi, method in enumerate(methods):
                choice = selected[method]
                distance = scores[method][:, choice["column"]]
                curve, ranking = sm.distance_completeness_purity_curve(distance, working & test, test)
                fixed = sm.fixed_distance_selection_summary(distance, working & test, test, choice["threshold"])
                row = {"mode": mode, "split": split, "method": method,
                       "selected_member": choice.get("selected_member", method),
                       "neighbor_rank": choice["rank"], "validation_ap": choice["validation_ap"],
                       "average_precision": ranking["average_precision"], "ap_lift": ranking["ap_lift"],
                       **fixed, "n_train": int(train.sum()), "n_validation": int(validation.sum()),
                       "n_anchors": int((train & working).sum())}
                split_rows.append(row)
                test_scores[mode_index, split, mi, test] = distance[test]
                # First threshold achieving each requested recall; no precision envelope/maxima.
                positions = np.searchsorted(curve["completeness"].to_numpy(), grid, side="left")
                points = curve.iloc[np.minimum(positions, len(curve) - 1)]
                for recall, purity in zip(grid, points["purity"], strict=True):
                    curve_rows.append({"mode": mode, "split": split, "method": method,
                                       "completeness": recall, "purity": float(purity)})
                indices = np.flatnonzero(test)
                # IDs are canonically sorted; stable ties therefore do not depend on file order.
                order = np.argsort(distance[indices], kind="stable")
                top_sets[method] = set(indices[order[:math.ceil(0.1 * len(indices))]])
            if mode == "observed":
                for a in methods:
                    for b in methods:
                        jaccard_rows.append({"split": split, "method_a": a, "method_b": b,
                                             "jaccard": len(top_sets[a] & top_sets[b]) / len(top_sets[a] | top_sets[b])})
            if progress and ((split + 1) % 10 == 0 or split == 0 or split + 1 == count):
                print(f"{mode}: {split + 1}/{count} splits; elapsed {time.monotonic() - start:.1f}s", flush=True)
    results = pd.DataFrame(split_rows)
    values = ["average_precision", "ap_lift", "purity", "completeness", "f1", "prevalence", "n_selected"]
    summary = summarize_rows(results, ["mode", "method"], values)
    paired = []
    for mode in ["observed", "shuffled"]:
        data = results[results["mode"] == mode].set_index(["method", "split"])
        for kind in ["latent", "umap"]:
            ensemble = f"{kind}:ensemble"
            baselines = [f"{kind}:{config.reference}", f"{kind}:best_single"]
            if kind == "latent":
                baselines.append(f"umap:{config.reference}")
            for baseline in baselines:
                for split in range(count):
                    row = {"mode": mode, "split": split, "ensemble": ensemble, "baseline": baseline}
                    for value in ["average_precision", "purity", "completeness", "f1"]:
                        row["delta_" + value] = data.loc[(ensemble, split), value] - data.loc[(baseline, split), value]
                    paired.append(row)
    paired = pd.DataFrame(paired)
    paired_summary = summarize_rows(paired, ["mode", "ensemble", "baseline"],
                                    ["delta_average_precision", "delta_purity", "delta_completeness", "delta_f1"])
    arrays = {"ids": cohort.ids, "methods": np.array(methods), "modes": np.array(["observed", "shuffled"]),
              "partitions": partitions, "target_labels": labels, "test_scores": test_scores}
    tables = {"summary": summary, "split_metrics": results, "curve_samples": pd.DataFrame(curve_rows),
              "paired_differences": paired, "paired_summary": paired_summary,
              "candidate_overlap": pd.DataFrame(jaccard_rows), "validation_selections": pd.DataFrame(selection_rows),
              "normalization": pd.DataFrame(normalization_rows)}
    return tables, arrays


def linear_cka(x, y):
    init_stack()
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    cross = np.linalg.norm(x.T @ y, "fro") ** 2
    denominator = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    if denominator <= 0:
        raise ValueError("CKA is undefined for a zero-variance representation")
    return float(cross / denominator)


def agreement_diagnostics(cohort, config):
    init_stack()
    indices = np.sort(np.random.default_rng(config.seed).choice(
        len(cohort.ids), size=min(config.cka_sample_size, len(cohort.ids)), replace=False))
    records = [{"experiment_a": a, "experiment_b": b, "cka": linear_cka(x[indices], y[indices])}
               for a, x in cohort.latents.items() for b, y in cohort.latents.items()]
    return pd.DataFrame(records), cohort.ids[indices]


def method_label(method):
    kind, member = method.split(":", 1)
    prefix = "Latent" if kind == "latent" else "UMAP"
    if member == "ensemble":
        return f"{prefix} ensemble"
    if member == "best_single":
        return f"{prefix} best single (validation)"
    return f"{prefix} E{member.split('expt')[-1]}"


def render_figures(cohort, config, tables, output_dir):
    init_stack()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/multi_latent_matplotlib")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/multi_latent_numba")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from umap import UMAP

    output_dir = Path(output_dir)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.titleweight": "bold", "svg.fonttype": "none"})
    chosen = [f"umap:{config.reference}", "umap:best_single", "umap:ensemble",
              f"latent:{config.reference}", "latent:best_single", "latent:ensemble"]
    colors = dict(zip(chosen, ["#687383", "#98b2c6", "#087e8b", "#d6a57c", "#ae7654", "#bc4b2a"]))
    paths = []

    def save(fig, stem):
        for extension in ["png", "svg"]:
            path = output_dir / f"{stem}.{extension}"
            fig.savefig(path, dpi=180, facecolor="white")
            paths.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), layout="constrained")
    fig.suptitle("Does combining experiments improve merger recovery?", fontsize=17)
    curves = tables["curve_samples"]
    for method in chosen:
        f = curves[(curves["mode"] == "observed") & (curves["method"] == method)]
        bands = f.groupby("completeness")["purity"].quantile([0.16, 0.5, 0.84]).unstack()
        axes[0].plot(bands.index, bands[0.5], label=method_label(method), color=colors[method], lw=1.8)
        if method.endswith("ensemble"):
            axes[0].fill_between(bands.index, bands[0.16], bands[0.84], color=colors[method], alpha=0.12)
    axes[0].axhline(cohort.coverage["prevalence"], ls=":", color="#343c48", label="Target prevalence")
    axes[0].set(xlim=(0.02, 0.6), ylim=(0, 0.45), xlabel="Completeness", ylabel="Purity", title="Test ranking curves")
    axes[0].legend(fontsize=7.7, loc="upper right")
    pair = tables["paired_summary"]
    pair = pair[(pair["mode"] == "observed") & pair["baseline"].str.endswith("best_single")]
    for y, (_, row) in enumerate(pair.iterrows()):
        m = row["delta_average_precision"]
        axes[1].errorbar(m, y, xerr=[[m-row["delta_average_precision_p16"]],
                                   [row["delta_average_precision_p84"]-m]],
                         fmt="o", color=colors[row["ensemble"]], capsize=5)
    axes[1].axvline(0, color="#687383", ls=":")
    axes[1].set(yticks=range(len(pair)), yticklabels=[method_label(m) for m in pair["ensemble"]],
                ylim=(-0.7, 1.7), xlabel="Paired change in average precision", title="Against validation-selected single")
    summary = tables["summary"]
    for mode, offset, marker in [("observed", -0.08, "o"), ("shuffled", 0.08, "s")]:
        for y, method in enumerate(chosen):
            r = summary[(summary["mode"] == mode) & (summary["method"] == method)].iloc[0]
            m = r["average_precision"]
            axes[2].errorbar(m, y+offset, xerr=[[m-r["average_precision_p16"]],
                                              [r["average_precision_p84"]-m]], fmt=marker,
                             color=colors[method] if mode == "observed" else "#a8adb5", capsize=2,
                             label=mode.capitalize() if y == 0 else None)
    axes[2].set(yticks=range(len(chosen)), yticklabels=[method_label(m) for m in chosen],
                xlabel="Test average precision", title="Observed and shuffled labels")
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=9)
    fig.supxlabel(f"Run 11 · major mergers ≤ {config.cutoff_gyr:g} Gyr · {config.n_splits} paired splits · bands/bars: 16th–84th split variability", fontsize=10)
    save(fig, "01_recovery")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), layout="constrained")
    fig.suptitle("Shared representations and candidate agreement", fontsize=17)
    cka = tables["cka"].pivot(index="experiment_a", columns="experiment_b", values="cka")
    overlap = tables["candidate_overlap"].groupby(["method_a", "method_b"])["jaccard"].median().unstack()
    overlap = overlap.loc[chosen, chosen]
    for ax, matrix, title, labels in [
        (axes[0], cka, f"Linear CKA · {min(config.cka_sample_size, len(cohort.ids)):,} shared objects", ["E"+n.split("expt")[-1] for n in cka.index]),
        (axes[1], overlap, "Top 10% of test objects · median Jaccard", [method_label(m).replace(" (validation)", "") for m in chosen]),
    ]:
        im = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu")
        ax.set(xticks=range(len(matrix)), yticks=range(len(matrix)), xticklabels=labels, yticklabels=labels, title=title)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color="white" if value > .65 else "#172b42")
        fig.colorbar(im, ax=ax, shrink=.65)
    fig.supxlabel("Agreement describes redundancy; it does not by itself establish better merger recovery.", fontsize=10)
    save(fig, "02_agreement")

    print("Fitting descriptive combined-space UMAP (not used in scoring)...", flush=True)
    stats = fit_normalization(cohort.latents, np.ones(len(cohort.ids), dtype=bool))
    combined = combine_latents(cohort.latents, stats)
    projection = UMAP(n_neighbors=15, min_dist=.1, metric="euclidean", random_state=config.seed,
                      transform_seed=config.seed, n_jobs=1).fit_transform(combined)
    pd.DataFrame({"object_id": cohort.ids, "x": projection[:, 0], "y": projection[:, 1]}).to_csv(output_dir / "descriptive_projection.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    fig.suptitle("Where do known recent major mergers appear?", fontsize=17)
    for ax, points, title in [(axes[0], cohort.umaps[config.reference], "Existing experiment 9 UMAP"),
                              (axes[1], projection, "Combined latent geometry → new UMAP")]:
        ax.scatter(points[:, 0], points[:, 1], s=4, c="#85949e", alpha=.25, rasterized=True, linewidths=0)
        ax.scatter(points[cohort.targets, 0], points[cohort.targets, 1], s=12, c="#bd492a", alpha=.85,
                   label=f"Major mergers ({cohort.targets.sum():,})", linewidths=.2, edgecolors="white")
        ax.set(title=title, xticks=[], yticks=[], xlabel="UMAP 1", ylabel="UMAP 2")
        ax.legend(loc="upper right", fontsize=9)
    fig.supxlabel("Descriptive projections only; independent axes. Combined-space retrieval is evaluated before UMAP.", fontsize=10)
    save(fig, "03_spatial_interpretation")
    return paths


def write_meeting_notes(cohort, config, tables, output_dir):
    init_stack()
    summary = tables["summary"]
    observed = summary[summary["mode"] == "observed"].set_index("method")
    shuffled = summary[summary["mode"] == "shuffled"].set_index("method")
    lines = ["# Multi-latent merger recovery: meeting brief", "",
             "**Question:** Do existing experiments provide complementary information that improves merger recovery?", "",
             f"Run 11, experiments 9/13/14; {len(cohort.ids):,} objects, "
             f"{cohort.targets.sum():,} major mergers within {config.cutoff_gyr:g} Gyr "
             f"({cohort.coverage['prevalence']:.2%} prevalence).", "",
             "Equal-weight ensembles combine either normalized latent distances or validation-calibrated UMAP retrieval scores. "
             "Only training mergers act as anchors. Validation chooses k and the threshold; test galaxies provide the reported metrics.", "",
             "| Method | Test AP | Purity | Completeness | F1 | Shuffled AP |",
             "|---|---:|---:|---:|---:|---:|"]
    chosen = [f"umap:{config.reference}", "umap:best_single", "umap:ensemble",
              f"latent:{config.reference}", "latent:best_single", "latent:ensemble"]
    for method in chosen:
        r = observed.loc[method]
        lines.append(f"| {method_label(method)} | {r.average_precision:.4f} | {r.purity:.1%} | "
                     f"{r.completeness:.1%} | {r.f1:.4f} | {shuffled.loc[method, 'average_precision']:.4f} |")
    lines += ["", "All table entries are separate medians across splits; a row is not one operating point. "
              "Purity/completeness/F1 use validation-fixed thresholds. AP summarizes the whole test ranking.", "", "## Paired ensemble comparison", ""]
    pair = tables["paired_summary"]
    for _, row in pair[(pair["mode"] == "observed") & pair["baseline"].str.endswith("best_single")].iterrows():
        lines.append(f"- {method_label(row['ensemble'])} minus its validation-selected single: "
                     f"median ΔAP = **{row['delta_average_precision']:+.4f}**; "
                     f"16th–84th split range [{row['delta_average_precision_p16']:+.4f}, {row['delta_average_precision_p84']:+.4f}].")
    lines += ["", "## Interpretation and next experiment", "",
              "Compare each ensemble to the best single selected on validation, not to whichever model scores highest on the test set. "
              "A positive median difference suggests useful complementarity; a range crossing zero indicates inconsistent gains across these splits. "
              "The shuffled control measures how much retrieval survives after the merger association is removed.", "",
              "Next, freeze the target, membership, and scoring rule and test on an untouched galaxy sample or snapshot; "
              "use leave-one-experiment-out ablations to identify which member contributes. "
              "If the ensemble loses, inspect dilution by weaker members before expanding the ensemble.", "",
              "## Limits", "",
              f"- {config.n_splits} repeated 50/25/25 splits, seed {config.seed}; k ∈ {list(sorted(config.neighbor_ranks))}.",
              "- Percentile bands are split variability, not confidence intervals; overlapping splits are correlated.",
              "- The target was informed by earlier results and the representations were already trained; this is exploratory, not an untouched-galaxy generalization test.",
              "- Shuffled-label comparisons are negative controls, not a formal permutation significance test.",
              "- CKA and visual UMAP structure are descriptive; retrieval results determine usefulness.", "",
              "Linear CKA reference: [Kornblith et al. (2019)](https://proceedings.mlr.press/v97/kornblith19a.html).", ""]
    Path(output_dir, "meeting_brief.md").write_text("\n".join(lines))


def run_analysis(manifest, output_dir, config=None, figures=True):
    init_stack()
    from threadpoolctl import threadpool_limits

    config = config or AnalysisConfig()
    config.validate()
    output_dir = Path(output_dir).resolve()
    if (output_dir / "analysis_manifest.json").exists():
        raise FileExistsError("Result bundle already exists; load it or choose a new output directory")
    cohort, fingerprints = load_manifest(manifest, config)
    print(json.dumps(cohort.coverage, indent=2), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        tables, arrays = evaluate(cohort, config)
        tables["cka"], cka_ids = agreement_diagnostics(cohort, config)
        arrays["cka_sample_ids"] = cka_ids
        np.savez_compressed(output_dir / "split_audit.npz", **arrays)
        pd.DataFrame({"object_id": cohort.ids, "known_label": cohort.known,
                      "target": cohort.targets}).to_csv(output_dir / "cohort.csv", index=False)
        figure_paths = render_figures(cohort, config, tables, output_dir) if figures else []
    write_meeting_notes(cohort, config, tables, output_dir)
    versions = {package: metadata.version(package) for package in
                ["numpy", "pandas", "scipy", "scikit-learn", "astropy", "matplotlib", "umap-learn"]}
    sm.save_analysis_bundle(output_dir, tables, asdict(config), metadata={
        "coverage": cohort.coverage, "inputs": fingerprints, "versions": versions,
        "python": sys.version, "source_sha256": sha256_file(__file__),
        "metric_source_sha256": sha256_file(sm.__file__),
        "split_codes": {"unknown": -1, "train": 0, "validation": 1, "test": 2},
        "figures": [p.name for p in figure_paths], "status": "complete",
        "score_direction": "lower is more merger-like", "cdf_ties": "right",
        "candidate_overlap": "exact ceil(10% of test set), stable object-ID tie break",
        "cka": "biased linear CKA on fixed unlabeled sample; descriptive",
        "projection": "all-object RMS normalization, UMAP(15, 0.1, euclidean, seed); descriptive only"})
    return {"cohort": cohort, "tables": tables, "output_dir": output_dir}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbor-ranks", type=int, nargs="+", default=[3, 5, 7, 9])
    parser.add_argument("--train-fraction", type=float, default=.5)
    parser.add_argument("--validation-fraction", type=float, default=.25)
    parser.add_argument("--reference", default="run11_expt9")
    parser.add_argument("--target-column", default="Major_TimeSinceMerger")
    parser.add_argument("--cutoff-gyr", type=float, default=.75)
    parser.add_argument("--no-figures", action="store_true", help="Skip descriptive plots for a smoke run")
    args = parser.parse_args(argv)
    config = AnalysisConfig(n_splits=args.n_splits, seed=args.seed, neighbor_ranks=tuple(args.neighbor_ranks),
                            train_fraction=args.train_fraction, validation_fraction=args.validation_fraction,
                            reference=args.reference, target_column=args.target_column, cutoff_gyr=args.cutoff_gyr)
    run_analysis(args.manifest, args.output_dir, config, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
