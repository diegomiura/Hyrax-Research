#!/usr/bin/env python3
"""Batchable static UMAP overlay plots and clustering metrics.

This is the script form of
static_umap_plotting_w_xmatched_samples_and_metric.ipynb.  It processes one
run/experiment pair at a time so Slurm can parallelize the expensive metric
work across jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


np = None
pd = None
plt = None
LogNorm = None
HDBSCAN = None


DEFAULT_N_PERMUTATIONS = 500
DEFAULT_MIN_CLUSTER_SIZE = 15
DEFAULT_CATALOG_KEY = "all"


@dataclass(frozen=True)
class PathContext:
    profile: str
    paths: Any
    project_root: Path
    hyrax_run_base: Path
    sample_catalog_paths: dict[str, Path]


def init_science_stack() -> None:
    """Import scientific dependencies only when the analysis actually runs."""
    global np, pd, plt, LogNorm, HDBSCAN
    if np is not None:
        return

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(os.environ.get("TMPDIR", "/tmp")) / f"matplotlib-{os.environ.get('SLURM_JOB_ID', 'manual')}"),
    )
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(Path(os.environ.get("TMPDIR", "/tmp")) / f"numba-{os.environ.get('SLURM_JOB_ID', 'manual')}"),
    )

    import numpy as _np
    import pandas as _pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.colors import LogNorm as _LogNorm

    try:
        from sklearn.cluster import HDBSCAN as _HDBSCAN
    except ImportError:
        from hdbscan import HDBSCAN as _HDBSCAN

    np = _np
    pd = _pd
    plt = _plt
    LogNorm = _LogNorm
    HDBSCAN = _HDBSCAN


def _load_toml(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    try:
        import tomllib

        with path.open("rb") as handle:
            return tomllib.load(handle)
    except ModuleNotFoundError:
        import tomli

        with path.open("rb") as handle:
            return tomli.load(handle)


def first_existing_path(candidates: Iterable[str | Path]) -> Path:
    """Return the first existing path, or the first candidate for readable errors."""
    resolved = [Path(candidate).expanduser() for candidate in candidates]
    for path in resolved:
        if path.exists():
            return path
    return resolved[0]


def profile_path(paths: Any, name: str, fallback: str | Path) -> Path:
    """Read a path from research_paths.py, with a fallback for older profiles."""
    return Path(paths.as_dict().get(name, fallback)).expanduser()


def load_path_context(
    profile: str | None = None,
    hyrax_run_base: str | Path | None = None,
) -> PathContext:
    """Resolve profile-managed paths after applying an optional profile override."""
    if profile:
        os.environ["HYRAX_PROFILE"] = profile

    from research_paths import load_paths

    paths = load_paths(profile=profile)
    project_root = Path(__file__).resolve().parent
    run_base = Path(hyrax_run_base or paths.hyrax_runs).expanduser()

    time_since_all_catalog = first_existing_path(
        [
            profile_path(
                paths,
                "catalog_time_since_merger_all",
                "/work/hdd/bemi/dmiura/data_downloads/tng100_snap72/split_images/catalog2.fits",
            ),
            paths.catalog("raw_merger_flags"),
            paths.catalog("all"),
        ]
    )

    sample_catalog_paths = {
        # Match static_umap_plotting_time_since_merger_by_type.ipynb:
        # "all" means the full sample catalog with appended merger fields.
        "all": time_since_all_catalog,
        "raw_merger_flags": paths.catalog("raw_merger_flags"),
        "le_120x120": paths.catalog("le_120x120"),
        "gt_120x120": paths.catalog("gt_120x120"),
    }

    return PathContext(
        profile=paths.profile,
        paths=paths,
        project_root=project_root,
        hyrax_run_base=run_base,
        sample_catalog_paths=sample_catalog_paths,
    )


def catalog_path_status(ctx: PathContext):
    """Show profile-managed sample catalogs used by this workflow."""
    rows = []
    for name, path in ctx.sample_catalog_paths.items():
        rows.append({"source": "sample", "name": name, "path": str(path), "exists": Path(path).exists()})
    return pd.DataFrame(rows)


def load_external_catalog(catalog_path: str | Path):
    """Load an overlay catalog from parquet, FITS, or CSV."""
    catalog_path = Path(catalog_path).expanduser()
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    suffix = catalog_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(catalog_path)
    if suffix in {".fits", ".fit", ".fts"}:
        from astropy.table import Table

        return Table.read(catalog_path).to_pandas()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(catalog_path)

    raise ValueError(f"Unsupported catalog format '{suffix}'. Use parquet, FITS, or CSV.")


def load_sample_catalog(ctx: PathContext, name: str = DEFAULT_CATALOG_KEY, required: bool = True):
    """Load one of the profile-managed Hyrax/TNG sample catalogs."""
    if name not in ctx.sample_catalog_paths:
        valid = ", ".join(sorted(ctx.sample_catalog_paths))
        raise KeyError(f"Unknown sample catalog '{name}'. Choose one of: {valid}")

    path = Path(ctx.sample_catalog_paths[name])
    if not path.exists():
        message = f"Sample catalog '{name}' is not available at {path}. Check path_profiles.toml or HYRAX_PROFILE."
        if required:
            raise FileNotFoundError(message)
        print(f"Skipping sample catalog: {message}")
        return None

    catalog = load_external_catalog(path)
    print(f"Loaded sample catalog '{name}': {len(catalog):,} rows from {path}")
    return catalog


def _is_missing_scalar(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8").strip()
    return value


def _integer_like_text(text: str) -> str | None:
    """Normalize integer-like strings such as '123.0' without using floats."""
    if not re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if value == value.to_integral_value():
        return str(value.to_integral_value())
    return None


def _normalize_object_id(value: Any) -> Any:
    value = _decode_scalar(value)
    if _is_missing_scalar(value):
        return pd.NA

    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return format(float(value), ".15g")

    text = str(value).strip()
    if not text:
        return pd.NA
    integer_text = _integer_like_text(text)
    return integer_text if integer_text is not None else text


def normalize_object_ids(values: Any):
    """Normalize object IDs to comparable nullable strings."""
    return pd.Series(values, copy=False).map(_normalize_object_id).astype("string")


def resolve_catalog_id_column(catalog, catalog_id_column: str | None = None) -> str:
    """Find the catalog object-ID column used to match rows back to UMAP metadata."""
    candidates = [
        catalog_id_column,
        "object_id",
        "rubin_object_id",
        "objectId",
        "objectId_data",
        "id",
    ]
    for candidate in candidates:
        if candidate is not None and candidate in catalog.columns:
            return candidate

    raise KeyError("Could not find an object ID column in the catalog. Pass catalog_id_column explicitly.")


def first_present_column(catalog, candidates: Iterable[str]) -> str | None:
    """Return the first candidate column present in a catalog."""
    for candidate in candidates:
        if candidate in catalog.columns:
            return candidate
    return None


def build_default_overlay_groups(catalog, time_since_merger_max_gyr: float | None = None) -> dict[str, list[dict[str, Any]]]:
    """Build useful overlay groups from the columns present in the loaded catalog."""
    groups: dict[str, list[dict[str, Any]]] = {}

    recent_specs = [
        ("has_major_past_1gyr", "tab:red", "x", "Major past 1 Gyr"),
        ("has_minor_past_1gyr", "tab:green", "s", "Minor past 1 Gyr"),
        ("has_mini_past_1gyr", "tab:blue", ".", "Mini past 1 Gyr"),
    ]
    recent = [
        {"key": key, "threshold": 0.5, "color": color, "marker": marker, "label": label, "s": 10}
        for key, color, marker, label in recent_specs
        if key in catalog.columns
    ]
    if recent:
        groups["recent_merger_flags"] = recent

    future_specs = [
        ("has_major_future_1gyr", "tab:red", "x", "Major future 1 Gyr"),
        ("has_minor_future_1gyr", "tab:green", "s", "Minor future 1 Gyr"),
        ("has_mini_future_1gyr", "tab:blue", ".", "Mini future 1 Gyr"),
    ]
    future = [
        {"key": key, "threshold": 0.5, "color": color, "marker": marker, "label": label, "s": 10}
        for key, color, marker, label in future_specs
        if key in catalog.columns
    ]
    if future:
        groups["future_merger_flags"] = future

    time_since = []
    for merger_type, color, marker in [
        ("Mini", "tab:blue", "."),
        ("Minor", "tab:green", "s"),
        ("Major", "tab:red", "x"),
    ]:
        key = first_present_column(
            catalog,
            [
                f"{merger_type}_TimeSinceMerger",
                f"{merger_type.lower()}_time_since_merger",
            ],
        )
        if key is not None:
            overlay = {
                "key": key,
                "min_value": 0.0,
                "include_min": True,
                "color": color,
                "marker": marker,
                "label": f"{merger_type} time since merger",
                "s": 10,
            }
            if time_since_merger_max_gyr is not None:
                overlay["max_value"] = float(time_since_merger_max_gyr)
                overlay["include_max"] = True
                overlay["label"] = f"{merger_type} time since <= {float(time_since_merger_max_gyr):g} Gyr"
            time_since.append(overlay)
    if time_since:
        groups["time_since_merger"] = time_since

    count_since = []
    for merger_type, color, marker in [
        ("Major", "tab:red", "x"),
        ("Minor", "tab:green", "s"),
        ("Mini", "tab:blue", "."),
    ]:
        key = first_present_column(
            catalog,
            [
                f"{merger_type}_CountSince1Gyr",
                f"{merger_type.lower()}_count_since_1gyr",
            ],
        )
        if key is not None:
            count_since.append(
                {
                    "key": key,
                    "threshold": 1,
                    "color": color,
                    "marker": marker,
                    "label": f"{merger_type} count since 1 Gyr >= 1",
                    "s": 10,
                }
            )
    if count_since:
        groups["count_since_merger_1gyr"] = count_since

    if "merging_merger_fraction" in catalog.columns:
        groups["euclid_vissyn_merger_fraction"] = [
            {
                "key": "merging_merger_fraction",
                "threshold": 0.7,
                "color": "red",
                "marker": "x",
                "label": "Euclid VisSyn mergers",
                "s": 10,
            },
        ]

    return groups


def choose_overlay_group(overlay_groups: Mapping[str, list[dict[str, Any]]], preferred: str = "recent_merger_flags") -> str:
    """Choose the preferred overlay group, or the first available group."""
    if preferred in overlay_groups:
        return preferred
    if overlay_groups:
        return next(iter(overlay_groups))
    raise ValueError(
        "No usable overlay columns were found in the loaded catalog. "
        "Expected columns like has_major_past_1gyr or Major_TimeSinceMerger."
    )


def _overlay_row_mask(catalog, overlay: Mapping[str, Any]):
    """Return the catalog-row mask requested by one overlay specification."""
    key = overlay.get("key")
    if key is None:
        return pd.Series(True, index=catalog.index)

    if key not in catalog.columns:
        raise KeyError(f"Catalog column '{key}' not found. Available columns include: {list(catalog.columns[:20])}")

    values = pd.to_numeric(catalog[key], errors="coerce")
    if "min_value" in overlay or "max_value" in overlay:
        mask = values.notna() & np.isfinite(values)
        if overlay.get("min_value") is not None:
            mask &= values >= overlay["min_value"] if overlay.get("include_min", True) else values > overlay["min_value"]
        if overlay.get("max_value") is not None:
            mask &= values <= overlay["max_value"] if overlay.get("include_max", True) else values < overlay["max_value"]
        return mask

    threshold = overlay.get("threshold", 0.0)
    comparator = overlay.get("comparator", ">=")
    if comparator == ">=":
        return values >= threshold
    if comparator == ">":
        return values > threshold
    if comparator == "<=":
        return values <= threshold
    if comparator == "<":
        return values < threshold
    if comparator == "==":
        return values == threshold
    raise ValueError(f"Unsupported comparator '{comparator}'")


def summarize_overlays(catalog, overlays: Sequence[Mapping[str, Any]]):
    """Return catalog-row counts for each overlay selection."""
    rows = []
    for overlay in overlays:
        mask = _overlay_row_mask(catalog, overlay).fillna(False)
        key = overlay.get("key")
        row = {
            "label": overlay.get("label", overlay.get("key")),
            "key": key,
            "selected_catalog_rows": int(mask.sum()),
        }
        if key in catalog.columns:
            values = pd.to_numeric(catalog.loc[mask, key], errors="coerce")
            values = values[np.isfinite(values)]
            if len(values) > 0:
                row["selected_min"] = float(values.min())
                row["selected_median"] = float(values.median())
                row["selected_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def extract_umap_info(run: int, expt: int, base_directory: str | Path) -> tuple[Path, Path | None, Path]:
    """Resolve UMAP output, optional inference output, and config paths."""
    import static_umap_plotting as sup

    umap_dir, config_file = sup.extract_umap_info(run, expt, base_directory=base_directory)
    config_file = Path(config_file)
    config = _load_toml(config_file)
    inference_dir = config.get("results", {}).get("inference_dir")

    return (
        Path(umap_dir),
        Path(inference_dir) if inference_dir else None,
        config_file,
    )


def _extract_metadata_column(metadata_obj: Any, field_name: str):
    """Extract one metadata field from dict, DataFrame, structured array, or array payloads."""
    if isinstance(metadata_obj, dict):
        if field_name in metadata_obj:
            return np.asarray(metadata_obj[field_name])
        if len(metadata_obj) == 1:
            return np.asarray(next(iter(metadata_obj.values())))
        return None

    if hasattr(metadata_obj, "columns"):
        columns = list(metadata_obj.columns)
        if field_name in columns:
            return metadata_obj[field_name].to_numpy()
        if len(columns) == 1:
            return metadata_obj[columns[0]].to_numpy()
        return None

    dtype = getattr(metadata_obj, "dtype", None)
    names = getattr(dtype, "names", None)
    if names:
        if field_name in names:
            return np.asarray(metadata_obj[field_name])
        if len(names) == 1:
            return np.asarray(metadata_obj[names[0]])
        return None

    array = np.asarray(metadata_obj)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    return None


def get_umap_with_ids(
    config: Any = None,
    input_dir: str | Path | None = None,
    suppress_logs: bool = True,
    id_field: str = "objectId_data",
) -> dict[str, Any]:
    """Load UMAP coordinates and the object IDs used for catalog matching."""
    from hyrax.data_sets.inference_dataset import InferenceDataSet

    if suppress_logs:
        logging.disable(logging.CRITICAL)

    try:
        umap_results = InferenceDataSet(config, results_dir=input_dir, verb="umap")
        points = np.array([point.numpy() for point in umap_results])
        all_indices = list(range(len(umap_results)))
        available_fields = list(umap_results.metadata_fields())
    finally:
        if suppress_logs:
            logging.disable(logging.NOTSET)

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"UMAP results must be a 2D array with at least two columns; got {points.shape}")

    preferred_fields = [
        id_field,
        "objectId_data",
        "object_id_data",
        "objectId",
        "object_id",
        "rubin_object_id",
        "id",
    ]
    candidate_fields = []
    for field in preferred_fields:
        if field is not None and field not in candidate_fields:
            candidate_fields.append(field)
    candidate_fields = [field for field in candidate_fields if field in available_fields] + [
        field for field in candidate_fields if field not in available_fields
    ]

    attempts = []
    rubin_ids = None
    resolved_field = None
    for candidate in candidate_fields:
        try:
            metadata = umap_results.metadata(all_indices, [candidate])
            extracted = _extract_metadata_column(metadata, candidate)
            if extracted is None:
                attempts.append(f"{candidate}: not found in metadata payload")
                continue
            if len(extracted) != len(umap_results):
                attempts.append(f"{candidate}: length mismatch ({len(extracted)} vs {len(umap_results)})")
                continue

            rubin_ids = np.asarray(extracted)
            resolved_field = candidate
            break
        except Exception as exc:
            attempts.append(f"{candidate}: {exc}")

    if rubin_ids is None:
        raise KeyError(
            "Could not extract object IDs from UMAP metadata. "
            f"Available metadata fields: {available_fields}. "
            f"Attempts: {'; '.join(attempts)}"
        )

    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "rubin_ids": rubin_ids,
        "id_field": resolved_field,
        "umap_results": umap_results,
    }


def get_highdim_data(
    config: Any = None,
    inference_dir: str | Path | None = None,
    umap_ids: Any = None,
    suppress_logs: bool = True,
):
    """Load high-dimensional latent vectors and align them to UMAP ID order."""
    if inference_dir is None:
        return None

    import static_umap_plotting as sup

    inference_dir = Path(inference_dir)
    result_ids = None

    try:
        points, result_ids = sup.load_result_tensors(inference_dir, flatten=True)
    except Exception as batch_error:
        from hyrax.data_sets.inference_dataset import InferenceDataSet

        if suppress_logs:
            logging.disable(logging.CRITICAL)
        try:
            inference_results = InferenceDataSet(config, results_dir=inference_dir)
            points = np.array([point.numpy() for point in inference_results])
        finally:
            if suppress_logs:
                logging.disable(logging.NOTSET)

        points = points.reshape(points.shape[0], -1)
        print(f"Loaded high-dimensional data through Hyrax fallback after batch read failed: {batch_error}")

    if umap_ids is None:
        return points

    if result_ids is None:
        if len(points) != len(umap_ids):
            raise ValueError(
                "High-dimensional results did not expose IDs and length does not match UMAP data: "
                f"{len(points)} vs {len(umap_ids)}"
            )
        return points

    highdim_lookup = pd.DataFrame(
        {
            "_match_id": normalize_object_ids(result_ids),
            "_hd_index": np.arange(len(points)),
        }
    ).dropna(subset=["_match_id"])

    if highdim_lookup["_match_id"].duplicated().any():
        duplicates = highdim_lookup.loc[highdim_lookup["_match_id"].duplicated(), "_match_id"].head().tolist()
        raise ValueError(f"High-dimensional result IDs contain duplicates; first duplicates: {duplicates}")

    umap_lookup = pd.DataFrame(
        {
            "_match_id": normalize_object_ids(umap_ids),
            "_umap_index": np.arange(len(umap_ids)),
        }
    )

    aligned = umap_lookup.merge(highdim_lookup, on="_match_id", how="left", validate="many_to_one")
    missing = aligned["_hd_index"].isna()
    if missing.any():
        preview = aligned.loc[missing, "_match_id"].head().tolist()
        raise ValueError(
            f"{int(missing.sum())} UMAP IDs are missing from the high-dimensional inference results. "
            f"First missing IDs: {preview}"
        )

    return points[aligned["_hd_index"].to_numpy(dtype=int)]


def _as_2d_numeric_array(coords: Any, name: str = "coords"):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    elif coords.ndim > 2:
        coords = coords.reshape(coords.shape[0], -1)
    if coords.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return coords


def mnln_ratio(coords: Any, labeled_mask: Any, n_permutations: int = 500, seed: int = 42) -> dict[str, Any]:
    """Median Nearest Labeled Neighbor distance ratio with a lower-tail permutation test.

    The ratio is observed_median / expected_median. Values below 1 indicate
    that labeled points are closer together than random same-sized samples.
    The p-value is the fraction of random samples with a median nearest-neighbor
    distance less than or equal to the observed distance.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = _as_2d_numeric_array(coords)
    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    rng = np.random.default_rng(seed)
    n_points = len(coords)
    n_labeled = int(labeled_mask.sum())

    if n_labeled < 2:
        return {
            "ratio": np.nan,
            "observed": np.nan,
            "expected": np.nan,
            "std_null": np.nan,
            "z_score": np.nan,
            "p_value": np.nan,
            "n_labeled": n_labeled,
        }

    def _median_nln(indices: Any) -> float:
        pts = coords[indices]
        if len(pts) < 2:
            return np.nan
        neighbors = NearestNeighbors(n_neighbors=2, metric="euclidean")
        neighbors.fit(pts)
        distances, _ = neighbors.kneighbors(pts, return_distance=True)
        return float(np.median(distances[:, 1]))

    labeled_indices = np.where(labeled_mask)[0]
    observed = _median_nln(labeled_indices)

    null_medians = np.empty(n_permutations)
    for p in range(n_permutations):
        rand_indices = rng.choice(n_points, size=n_labeled, replace=False)
        null_medians[p] = _median_nln(rand_indices)

    expected = float(null_medians.mean())
    std_null = float(null_medians.std())
    ratio = float(observed / expected) if expected > 0 else np.nan
    z_score = float((observed - expected) / std_null) if std_null > 0 else np.nan
    p_value = float((np.sum(null_medians <= observed) + 1) / (n_permutations + 1))

    return {
        "ratio": ratio,
        "observed": float(observed),
        "expected": expected,
        "std_null": std_null,
        "z_score": z_score,
        "p_value": p_value,
        "n_labeled": n_labeled,
    }


def fit_hdbscan_labels(coords: Any, min_cluster_size: int = 15):
    """Fit HDBSCAN once per coordinate space for all CMC overlay metrics."""
    coords = _as_2d_numeric_array(coords)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    return clusterer.fit_predict(coords)


def cmc_gini_from_labels(cluster_labels: Any, labeled_mask: Any, n_permutations: int = 500, seed: int = 42) -> dict[str, Any]:
    """Cluster Membership Concentration via Gini coefficient."""
    cluster_labels = np.asarray(cluster_labels)
    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    rng = np.random.default_rng(seed)
    n_points = len(cluster_labels)
    n_labeled = int(labeled_mask.sum())

    if n_labeled < 2:
        return {
            "gini": np.nan,
            "expected_gini": np.nan,
            "std_null": np.nan,
            "z_score": np.nan,
            "p_value": np.nan,
            "n_labeled": n_labeled,
            "n_clusters": 0,
            "cluster_counts": {},
        }

    unique_bins = np.unique(cluster_labels)
    n_hdbscan_clusters = int(len(unique_bins[unique_bins != -1]))
    if len(unique_bins) < 2:
        return {
            "gini": np.nan,
            "expected_gini": np.nan,
            "std_null": np.nan,
            "z_score": np.nan,
            "p_value": np.nan,
            "n_labeled": n_labeled,
            "n_clusters": n_hdbscan_clusters,
            "cluster_counts": {},
        }

    def _gini(mask: Any) -> float:
        counts = np.array([np.sum(mask & (cluster_labels == cluster)) for cluster in unique_bins], dtype=float)
        total = counts.sum()
        if total == 0:
            return 0.0
        counts_sorted = np.sort(counts)
        n_bins = len(counts_sorted)
        index = np.arange(1, n_bins + 1)
        return float((2 * np.sum(index * counts_sorted) - (n_bins + 1) * total) / (n_bins * total))

    observed_gini = _gini(labeled_mask)
    cluster_counts = {}
    for cluster in unique_bins:
        count = int(np.sum(labeled_mask & (cluster_labels == cluster)))
        if count > 0:
            cluster_counts[int(cluster)] = count

    null_ginis = np.empty(n_permutations)
    for p in range(n_permutations):
        perm_mask = np.zeros(n_points, dtype=bool)
        perm_mask[rng.choice(n_points, size=n_labeled, replace=False)] = True
        null_ginis[p] = _gini(perm_mask)

    expected_gini = float(null_ginis.mean())
    std_null = float(null_ginis.std())
    z_score = float((observed_gini - expected_gini) / std_null) if std_null > 0 else np.nan
    p_value = float((np.sum(null_ginis >= observed_gini) + 1) / (n_permutations + 1))

    return {
        "gini": observed_gini,
        "expected_gini": expected_gini,
        "std_null": std_null,
        "z_score": z_score,
        "p_value": p_value,
        "n_labeled": n_labeled,
        "n_clusters": n_hdbscan_clusters,
        "cluster_counts": cluster_counts,
    }


def overlay_labeled_mask(umap_data: Mapping[str, Any], catalog: Any, overlay: Mapping[str, Any], catalog_id_column: str | None = None):
    """Build a boolean mask over UMAP points for one overlay selection."""
    catalog_id_column = resolve_catalog_id_column(catalog, catalog_id_column)
    selected = catalog.loc[_overlay_row_mask(catalog, overlay).fillna(False), [catalog_id_column]].copy()

    if selected.empty:
        return np.zeros(len(umap_data["rubin_ids"]), dtype=bool)

    selected_ids = set(normalize_object_ids(selected[catalog_id_column]).dropna().drop_duplicates())
    umap_ids = normalize_object_ids(umap_data["rubin_ids"])
    return umap_ids.isin(selected_ids).to_numpy(dtype=bool)


def compute_overlay_metrics(
    umap_data: Mapping[str, Any],
    catalog: Any,
    overlays: Sequence[Mapping[str, Any]],
    n_permutations: int = 500,
    min_cluster_size: int = 15,
    highdim_coords: Any = None,
    catalog_id_column: str | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Compute MNLN ratio and CMC-Gini for each overlay."""
    if catalog is None:
        raise ValueError("A catalog DataFrame is required to compute overlay metrics.")

    umap_coords = np.column_stack([umap_data["x"], umap_data["y"]])
    n_points = len(umap_coords)
    if highdim_coords is not None and len(highdim_coords) != n_points:
        raise ValueError(
            "highdim_coords must be aligned to UMAP order and have the same length: "
            f"{len(highdim_coords)} vs {n_points}"
        )

    cluster_labels_2d = fit_hdbscan_labels(umap_coords, min_cluster_size=min_cluster_size)
    cluster_labels_hd = None
    if highdim_coords is not None:
        cluster_labels_hd = fit_hdbscan_labels(highdim_coords, min_cluster_size=min_cluster_size)

    results = []
    for overlay in overlays:
        label = overlay.get("label", str(overlay.get("key")))
        labeled_mask = overlay_labeled_mask(umap_data, catalog, overlay, catalog_id_column=catalog_id_column)
        n_matched = int(labeled_mask.sum())

        entry = {"label": label, "key": overlay.get("key"), "n_matched": n_matched}
        entry["mnln_2d"] = mnln_ratio(umap_coords, labeled_mask, n_permutations=n_permutations, seed=seed)
        entry["cmc_2d"] = cmc_gini_from_labels(cluster_labels_2d, labeled_mask, n_permutations=n_permutations, seed=seed)

        if highdim_coords is not None and cluster_labels_hd is not None:
            entry["mnln_hd"] = mnln_ratio(highdim_coords, labeled_mask, n_permutations=n_permutations, seed=seed)
            entry["cmc_hd"] = cmc_gini_from_labels(cluster_labels_hd, labeled_mask, n_permutations=n_permutations, seed=seed)

        results.append(entry)

    return results


def _format_p(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if value < 0.001:
        return "p<.001"
    if value < 0.01:
        return f"p={value:.3f}"
    return f"p={value:.2f}"


def format_metric_text(metric_results: Sequence[Mapping[str, Any]], include_hd: bool = False) -> str:
    """Format metric results into annotation strings for plot subtitles."""
    lines = []
    for metric in metric_results:
        mnln = metric["mnln_2d"]
        cmc = metric["cmc_2d"]
        line = (
            f"{metric['label']} (n={metric['n_matched']}): "
            f"MNLN={mnln['ratio']:.2f} ({_format_p(mnln['p_value'])}), "
            f"Gini={cmc['gini']:.2f} ({_format_p(cmc['p_value'])})"
        )

        if include_hd and "mnln_hd" in metric:
            mnln_hd = metric["mnln_hd"]
            cmc_hd = metric["cmc_hd"]
            line += (
                f"\n  HD: MNLN={mnln_hd['ratio']:.2f} ({_format_p(mnln_hd['p_value'])}), "
                f"Gini={cmc_hd['gini']:.2f} ({_format_p(cmc_hd['p_value'])})"
            )

        lines.append(line)
    return "\n".join(lines)


def matched_overlay_points(umap_data: Mapping[str, Any], catalog: Any, overlay: Mapping[str, Any], catalog_id_column: str | None = None):
    """Return UMAP rows that match one catalog overlay."""
    catalog_id_column = resolve_catalog_id_column(catalog, catalog_id_column)
    key = overlay.get("key")
    selected_columns = [catalog_id_column]
    if key is not None and key in catalog.columns:
        selected_columns.append(key)

    selected = catalog.loc[_overlay_row_mask(catalog, overlay).fillna(False), selected_columns].copy()
    if selected.empty:
        return pd.DataFrame(columns=["x", "y", "_match_id"])

    selected["_match_id"] = normalize_object_ids(selected[catalog_id_column])
    selected = selected.dropna(subset=["_match_id"]).drop_duplicates("_match_id")

    umap_lookup = pd.DataFrame(
        {
            "x": umap_data["x"],
            "y": umap_data["y"],
            "_match_id": normalize_object_ids(umap_data["rubin_ids"]),
        }
    ).dropna(subset=["_match_id"])

    return selected.merge(umap_lookup, on="_match_id", how="inner")


def plot_umap_with_multi_overlay(
    ax: Any,
    umap_data: Mapping[str, Any],
    catalog: Any,
    overlays: Sequence[Mapping[str, Any]],
    catalog_id_column: str | None = None,
    alpha_background: float = 0.1,
    s_background: float = 1.0,
    title: str | None = None,
    show_legend: bool = True,
    density: bool = False,
    log_colorbar: bool = False,
    density_cmap: str = "viridis",
):
    """Plot UMAP with multiple catalog overlays."""
    if catalog is None:
        raise ValueError("A catalog DataFrame is required for overlay plotting.")

    x = umap_data["x"]
    y = umap_data["y"]

    if density:
        norm = LogNorm() if log_colorbar else None
        hb = ax.hexbin(x, y, gridsize=50, cmap=density_cmap, norm=norm)
        plt.colorbar(hb, ax=ax, label="Count")
    else:
        ax.scatter(x, y, alpha=alpha_background, s=s_background, c="gray", label="All", linewidths=0)

    for overlay in overlays:
        matched = matched_overlay_points(umap_data, catalog, overlay, catalog_id_column=catalog_id_column)
        if matched.empty:
            continue

        marker = overlay["marker"]
        color = overlay["color"]
        scatter_kwargs = {
            "alpha": overlay.get("alpha", 1.0),
            "s": overlay.get("s", 20),
            "c": color,
            "marker": marker,
            "label": f"{overlay['label']} (n={len(matched)})",
            "linewidths": overlay.get("linewidths", 1.0),
        }
        if marker not in {"x", "+", "1", "2", "3", "4", "|", "_"}:
            scatter_kwargs["edgecolors"] = overlay.get("edgecolors", "none")

        ax.scatter(matched["x"].to_numpy(), matched["y"].to_numpy(), **scatter_kwargs)

    if show_legend:
        ax.legend(loc="best", fontsize="small")
    if title:
        ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    return ax


def save_group_plot(
    save_path: Path,
    run: int,
    expt: int,
    overlay_group_name: str,
    catalog_label: str,
    umap_data: Mapping[str, Any],
    catalog: Any,
    overlays: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    catalog_id_column: str | None = None,
    include_hd: bool = False,
    dpi: int = 150,
    alpha_background: float = 0.5,
    s_background: float = 1.0,
    show_legend: bool = True,
    density: bool = False,
    log_colorbar: bool = False,
    density_cmap: str = "viridis",
) -> Path:
    """Save one run/experiment/group plot to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.4), dpi=dpi)
    plot_umap_with_multi_overlay(
        ax,
        umap_data,
        catalog,
        overlays,
        catalog_id_column=catalog_id_column,
        alpha_background=alpha_background,
        s_background=s_background,
        title=f"Run {run}, Expt {expt} - {overlay_group_name}",
        show_legend=show_legend,
        density=density,
        log_colorbar=log_colorbar,
        density_cmap=density_cmap,
    )
    annotation = format_metric_text(metrics, include_hd=include_hd)
    ax.text(
        0.02,
        0.02,
        annotation,
        transform=ax.transAxes,
        fontsize=6,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    fig.suptitle(f"{catalog_label} catalog", fontsize=10, y=0.99)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {save_path}")
    return save_path


def print_metrics_summary(rows: Sequence[Mapping[str, Any]], include_hd: bool = False) -> None:
    """Print a summary table of flattened metric records."""
    print(
        f"{'Run':>4} {'Expt':>5}  {'Group':<22} {'Overlay':<28} {'n':>5}  "
        f"{'MNLN':>7} {'p_mnln':>8} {'Gini':>7} {'p_gini':>8} {'nClus':>5}",
        end="",
    )
    if include_hd:
        print(f"  {'MNLN_HD':>7} {'p_HD':>8} {'Gini_HD':>7} {'p_gHD':>8}", end="")
    print()
    print("-" * (100 + (36 if include_hd else 0)))

    for row in rows:
        print(
            f"{row['run']:>4} {row['expt']:>5}  {row['overlay_group']:<22} "
            f"{row['overlay_label']:<28} {row['n_matched']:>5}  "
            f"{row.get('mnln_2d_ratio', math.nan):>7.3f} {row.get('mnln_2d_p_value', math.nan):>8.4f} "
            f"{row.get('cmc_2d_gini', math.nan):>7.3f} {row.get('cmc_2d_p_value', math.nan):>8.4f} "
            f"{row.get('cmc_2d_n_clusters', 0):>5}",
            end="",
        )
        if include_hd and "mnln_hd_ratio" in row:
            print(
                f"  {row.get('mnln_hd_ratio', math.nan):>7.3f} {row.get('mnln_hd_p_value', math.nan):>8.4f} "
                f"{row.get('cmc_hd_gini', math.nan):>7.3f} {row.get('cmc_hd_p_value', math.nan):>8.4f}",
                end="",
            )
        print()


def flatten_metric_record(run: int, expt: int, overlay_group: str, metric: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested metric dictionaries for CSV output."""
    record = {
        "run": int(run),
        "expt": int(expt),
        "overlay_group": overlay_group,
        "overlay_label": metric.get("label"),
        "overlay_key": metric.get("key"),
        "n_matched": metric.get("n_matched"),
    }
    for block_name in ("mnln_2d", "cmc_2d", "mnln_hd", "cmc_hd"):
        if block_name not in metric:
            continue
        for key, value in metric[block_name].items():
            record[f"{block_name}_{key}"] = json.dumps(value, sort_keys=True) if key == "cluster_counts" else value
    return record


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if np is not None:
        if isinstance(value, np.ndarray):
            return [_jsonable(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {path}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote JSON: {path}")


def resolve_selected_overlay_groups(
    overlay_groups: Mapping[str, list[dict[str, Any]]],
    requested: Sequence[str] | None = None,
    preferred: str = "time_since_merger",
    all_overlay_groups: bool = False,
) -> list[str]:
    """Resolve requested overlay group names with helpful validation."""
    if all_overlay_groups:
        return list(overlay_groups)
    if requested:
        missing = [name for name in requested if name not in overlay_groups]
        if missing:
            available = ", ".join(sorted(overlay_groups))
            raise KeyError(f"Unknown overlay group(s) {missing}. Available groups: {available}")
        return list(requested)
    return [choose_overlay_group(overlay_groups, preferred=preferred)]


def analyze_run_experiment(args: argparse.Namespace) -> int:
    init_science_stack()

    ctx = load_path_context(profile=args.profile, hyrax_run_base=args.base_directory)
    print(f"Path profile: {ctx.profile}")
    print(f"Hyrax run base: {ctx.hyrax_run_base}")
    print(catalog_path_status(ctx).to_string(index=False))

    if args.catalog_path:
        catalog_path = Path(args.catalog_path).expanduser()
        catalog = load_external_catalog(catalog_path)
        catalog_label = catalog_path.stem
    else:
        catalog = load_sample_catalog(ctx, args.catalog_key)
        catalog_label = args.catalog_key
        catalog_path = ctx.sample_catalog_paths[args.catalog_key]

    catalog_id_column = resolve_catalog_id_column(catalog, args.catalog_id_column)
    overlay_groups = build_default_overlay_groups(catalog, time_since_merger_max_gyr=args.time_since_merger_max_gyr)

    if args.list_overlays:
        print(f"Catalog ID column: {catalog_id_column}")
        print(f"Available overlay groups: {list(overlay_groups)}")
        for name, overlays in overlay_groups.items():
            print(f"\n[{name}]")
            print(summarize_overlays(catalog, overlays).to_string(index=False))
        return 0

    selected_group_names = resolve_selected_overlay_groups(
        overlay_groups,
        requested=args.overlay_group,
        preferred=args.preferred_overlay_group,
        all_overlay_groups=args.all_overlay_groups,
    )

    if args.run is None or args.expt is None:
        raise ValueError("--run and --expt are required unless --list-overlays is used")

    print(f"Active catalog: {catalog_label}")
    print(f"Catalog path: {catalog_path}")
    print(f"Catalog ID column: {catalog_id_column}")
    print(f"Selected overlay groups: {selected_group_names}")

    import hyrax

    umap_dir, inference_dir, config_file = extract_umap_info(args.run, args.expt, base_directory=ctx.hyrax_run_base)
    print(f"UMAP dir: {umap_dir}")
    print(f"Inference dir: {inference_dir}")
    print(f"Config file: {config_file}")

    if args.suppress_logs:
        logging.disable(logging.CRITICAL)
    try:
        h = hyrax.Hyrax(config_file=config_file)
    finally:
        if args.suppress_logs:
            logging.disable(logging.NOTSET)

    umap_data = get_umap_with_ids(
        config=h.config,
        input_dir=umap_dir,
        suppress_logs=args.suppress_logs,
        id_field=args.id_field,
    )

    highdim_coords = None
    if args.include_highdim:
        if inference_dir is None:
            message = f"Run {args.run}, Expt {args.expt}: no inference_dir in config; skipping high-dimensional metrics."
            if args.require_highdim:
                raise ValueError(message.replace("skipping high-dimensional metrics", "cannot compute required high-dimensional metrics"))
            print(message)
        else:
            highdim_coords = get_highdim_data(
                config=h.config,
                inference_dir=inference_dir,
                umap_ids=umap_data["rubin_ids"],
                suppress_logs=args.suppress_logs,
            )
            if args.require_highdim and highdim_coords is None:
                raise ValueError(f"Run {args.run}, Expt {args.expt}: required high-dimensional data loaded as None")

    output_root = Path(args.output_dir).expanduser() if args.output_dir else ctx.hyrax_run_base / "static_umap_metrics"
    output_dir = output_root / f"run{args.run}" / f"expt{args.expt}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metric_rows = []
    overlay_summary_rows = []
    json_groups: dict[str, Any] = {}

    for group_name in selected_group_names:
        overlays = overlay_groups[group_name]
        overlay_summary = summarize_overlays(catalog, overlays)
        overlay_summary["overlay_group"] = group_name
        overlay_summary_rows.extend(overlay_summary.to_dict(orient="records"))

        metrics = compute_overlay_metrics(
            umap_data,
            catalog,
            overlays,
            n_permutations=args.n_permutations,
            min_cluster_size=args.min_cluster_size,
            highdim_coords=highdim_coords,
            catalog_id_column=catalog_id_column,
            seed=args.seed,
        )

        all_metric_rows.extend(flatten_metric_record(args.run, args.expt, group_name, metric) for metric in metrics)
        json_groups[group_name] = {
            "overlay_summary": overlay_summary.to_dict(orient="records"),
            "metrics": metrics,
        }

        plot_path = output_dir / f"run{args.run}_expt{args.expt}_{group_name}.png"
        save_group_plot(
            plot_path,
            run=args.run,
            expt=args.expt,
            overlay_group_name=group_name,
            catalog_label=catalog_label,
            umap_data=umap_data,
            catalog=catalog,
            overlays=overlays,
            metrics=metrics,
            catalog_id_column=catalog_id_column,
            include_hd=args.include_highdim and highdim_coords is not None,
            dpi=args.dpi,
            alpha_background=args.alpha_background,
            s_background=args.s_background,
            show_legend=args.show_legend,
            density=args.density,
            log_colorbar=args.log_colorbar,
            density_cmap=args.density_cmap,
        )

    if args.require_highdim:
        if highdim_coords is None:
            raise ValueError(f"Run {args.run}, Expt {args.expt}: required high-dimensional metrics were not computed")
        missing_hd = [
            (row.get("overlay_group"), row.get("overlay_label"))
            for row in all_metric_rows
            if "mnln_hd_ratio" not in row or "cmc_hd_gini" not in row
        ]
        if missing_hd:
            raise ValueError(
                f"Run {args.run}, Expt {args.expt}: required high-dimensional metric columns are missing "
                f"for overlays {missing_hd[:5]}"
            )

    metrics_csv = output_dir / f"run{args.run}_expt{args.expt}_metrics.csv"
    metrics_json = output_dir / f"run{args.run}_expt{args.expt}_metrics.json"
    overlay_summary_csv = output_dir / f"run{args.run}_expt{args.expt}_overlay_summary.csv"

    write_csv(metrics_csv, all_metric_rows)
    write_csv(overlay_summary_csv, overlay_summary_rows)
    write_json(
        metrics_json,
        {
            "run": args.run,
            "expt": args.expt,
            "profile": ctx.profile,
            "catalog_label": catalog_label,
            "catalog_path": catalog_path,
            "catalog_id_column": catalog_id_column,
            "umap_dir": umap_dir,
            "inference_dir": inference_dir,
            "config_file": config_file,
            "n_permutations": args.n_permutations,
            "min_cluster_size": args.min_cluster_size,
            "include_highdim": args.include_highdim and highdim_coords is not None,
            "groups": json_groups,
        },
    )

    print_metrics_summary(all_metric_rows, include_hd=args.include_highdim and highdim_coords is not None)
    return 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create static UMAP overlay plots and clustering metrics for one Hyrax run/experiment."
    )
    parser.add_argument("--run", type=int, help="Hyrax run number, e.g. 10.")
    parser.add_argument("--expt", type=int, help="Experiment number inside the run, e.g. 12.")
    parser.add_argument("--profile", help="Path profile to use, e.g. local or delta.")
    parser.add_argument("--base-directory", type=Path, help="Base Hyrax runs directory. Defaults to research_paths.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for plots and metric tables. Defaults to <hyrax_run_base>/static_umap_metrics.",
    )
    parser.add_argument(
        "--catalog-key",
        default=DEFAULT_CATALOG_KEY,
        help="Profile catalog key when --catalog-path is not set. Defaults to the time-since notebook's all/catalog2.fits catalog.",
    )
    parser.add_argument("--catalog-path", type=Path, help="Explicit catalog path. Supports parquet, FITS, CSV.")
    parser.add_argument("--catalog-id-column", help="Catalog object ID column. Auto-detected by default.")
    parser.add_argument("--id-field", default="objectId_data", help="Preferred UMAP metadata object ID field.")
    parser.add_argument("--overlay-group", action="append", help="Overlay group to process. Repeat for multiple groups.")
    parser.add_argument("--preferred-overlay-group", default="time_since_merger", help="Default group when --overlay-group is omitted.")
    parser.add_argument("--all-overlay-groups", action="store_true", help="Process every detected overlay group.")
    parser.add_argument("--time-since-merger-max-gyr", type=float, help="Optional max Gyr cutoff for time-since-merger overlays.")
    parser.add_argument("--n-permutations", type=positive_int, default=DEFAULT_N_PERMUTATIONS)
    parser.add_argument("--min-cluster-size", type=positive_int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--seed", type=int, default=42, help="Permutation RNG seed.")
    parser.add_argument("--include-highdim", action="store_true", help="Also compute metrics in high-dimensional latent space.")
    parser.add_argument(
        "--require-highdim",
        action="store_true",
        help="Fail instead of silently skipping when requested high-dimensional metrics cannot be computed.",
    )
    parser.add_argument("--dpi", type=positive_int, default=150)
    parser.add_argument("--alpha-background", type=float, default=0.5)
    parser.add_argument("--s-background", type=float, default=1.0)
    parser.add_argument("--density", action="store_true", help="Use a hexbin density background instead of scatter.")
    parser.add_argument("--log-colorbar", action="store_true", help="Use log scaling for density colorbars.")
    parser.add_argument("--density-cmap", default="viridis")
    parser.add_argument("--show-legend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suppress-logs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--list-overlays", action="store_true", help="Load catalog, print available overlay groups, and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.require_highdim:
        args.include_highdim = True

    if not args.list_overlays and (args.run is None or args.expt is None):
        parser.error("--run and --expt are required unless --list-overlays is used")

    try:
        return analyze_run_experiment(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
