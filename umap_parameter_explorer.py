"""Deterministic UMAP parameter exploration for Hyrax inference results.

This module keeps expensive/reusable logic out of the companion notebook.  It
discovers the existing ``runN/udbN_E`` artifacts, creates deterministic UMAP
embeddings in an isolated exploration tree, writes Hyrax-compatible result
datasets, and provides comparison/diagnostic helpers.

The historical Hyrax result directories are always treated as read-only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import csv
import datetime as dt
import hashlib
from importlib import metadata as importlib_metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


CACHE_SCHEMA_VERSION = 1
DEFAULT_RANDOM_SEED = 42
DEFAULT_EVALUATION_SIZE = 2_000
DEFAULT_DIAGNOSTIC_K = 15
_UMAP_RESULT_PATTERN = re.compile(r"Saving UMAP results to\s+(.+?)\s*$", re.MULTILINE)
_SAFE_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class ExplorationError(RuntimeError):
    """Base error for UMAP exploration failures."""


class ArtifactDiscoveryError(ExplorationError):
    """Raised when an experiment's source artifacts cannot be resolved."""


class ParameterValidationError(ExplorationError, ValueError):
    """Raised when UMAP parameters are not valid for the fit sample."""


@dataclass(frozen=True, order=True)
class ExperimentRef:
    """A Hyrax experiment identified by its run and experiment numbers."""

    run: int
    expt: int

    def __post_init__(self) -> None:
        if int(self.run) < 0 or int(self.expt) < 0:
            raise ValueError("run and expt must be non-negative integers")
        object.__setattr__(self, "run", int(self.run))
        object.__setattr__(self, "expt", int(self.expt))

    @property
    def key(self) -> str:
        return f"run{self.run}_expt{self.expt}"

    @property
    def udb_stem(self) -> str:
        return f"udb{self.run}_{self.expt}"


@dataclass(frozen=True)
class UmapParams:
    """The three adjustable UMAP parameters used by this workflow."""

    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    n_components: int = field(default=2, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_neighbors", int(self.n_neighbors))
        object.__setattr__(self, "min_dist", float(self.min_dist))
        metric = str(self.metric).strip().lower()
        if not metric:
            raise ParameterValidationError("metric must be a non-empty UMAP metric name")
        object.__setattr__(self, "metric", metric)

    def validate(self, fit_sample_size: int) -> None:
        if fit_sample_size < 3:
            raise ParameterValidationError(
                f"UMAP requires at least 3 fit points; got {fit_sample_size}"
            )
        if not 2 <= self.n_neighbors < fit_sample_size:
            raise ParameterValidationError(
                "n_neighbors must satisfy 2 <= n_neighbors < fit_sample_size; "
                f"got n_neighbors={self.n_neighbors}, fit_sample_size={fit_sample_size}"
            )
        if not 0.0 <= self.min_dist <= 1.0:
            raise ParameterValidationError(
                f"min_dist must be between 0 and 1 inclusive; got {self.min_dist}"
            )

    def as_umap_kwargs(self, seed: int) -> dict[str, Any]:
        return {
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "n_components": self.n_components,
            "random_state": int(seed),
            "transform_seed": int(seed),
            "n_jobs": 1,
        }


@dataclass(frozen=True)
class ExperimentArtifacts:
    """Resolved source artifacts for one historical Hyrax experiment."""

    ref: ExperimentRef
    config_path: Path
    log_path: Path
    inference_dir: Path
    historical_umap_dir: Path
    original_params: UmapParams
    configured_fit_sample_size: int | None
    batch_size: int

    def as_record(self) -> dict[str, Any]:
        return {
            "run": self.ref.run,
            "expt": self.ref.expt,
            "config": str(self.config_path),
            "log": str(self.log_path),
            "inference_dir": str(self.inference_dir),
            "historical_umap_dir": str(self.historical_umap_dir),
            "n_neighbors": self.original_params.n_neighbors,
            "min_dist": self.original_params.min_dist,
            "metric": self.original_params.metric,
            "n_components": self.original_params.n_components,
            "fit_sample_size": self.configured_fit_sample_size or "all",
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True)
class UmapRun:
    """A historical or deterministic UMAP result ready for plotting."""

    artifact: ExperimentArtifacts
    result_dir: Path
    params: UmapParams
    label: str
    seed: int | None = None
    reused: bool = False
    historical: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "run": self.artifact.ref.run,
            "expt": self.artifact.ref.expt,
            "label": self.label,
            "result_dir": str(self.result_dir),
            "n_neighbors": self.params.n_neighbors,
            "min_dist": self.params.min_dist,
            "metric": self.params.metric,
            "n_components": self.params.n_components,
            "seed": self.seed,
            "historical": self.historical,
            "reused": self.reused,
            **dict(self.diagnostics),
        }


@dataclass
class SelectionEvaluation:
    """In-memory completeness-purity result for one named UMAP selection."""

    selection_name: str
    run: UmapRun
    summary: Any
    selected_objects: Any


@dataclass(frozen=True)
class MergerTimeOverlay:
    """One catalog merger-time rule prepared for display and scoring."""

    merger_type: str
    cutoff_gyr: float
    catalog_key: str
    label: str
    target_rule: Mapping[str, Any]
    visualizer_overlay: Mapping[str, Any]
    n_catalog_targets: int
    n_umap_targets: int


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - Python 3.11+ is required here.
        raise RuntimeError("Python 3.11+ is required to read Hyrax TOML files") from exc

    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolved_existing_dir(value: Any, *, field_name: str, ref: ExperimentRef) -> Path:
    if not value:
        raise ArtifactDiscoveryError(f"{ref.key}: missing {field_name}")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_dir():
        raise ArtifactDiscoveryError(f"{ref.key}: {field_name} is not a directory: {path}")
    if not (path / "batch_index.npy").is_file():
        raise ArtifactDiscoveryError(
            f"{ref.key}: {field_name} lacks batch_index.npy: {path}"
        )
    return path


def discover_experiment(ref: ExperimentRef, run_base: str | Path) -> ExperimentArtifacts:
    """Resolve one ``runN/udbN_E`` config, log, inference run, and historical UMAP."""

    run_base = Path(run_base).expanduser().resolve()
    run_dir = run_base / f"run{ref.run}"
    config_path = run_dir / f"{ref.udb_stem}.toml"
    log_path = run_dir / f"{ref.udb_stem}.txt"

    if not config_path.is_file():
        raise ArtifactDiscoveryError(f"{ref.key}: UDB config not found: {config_path}")
    if not log_path.is_file():
        raise ArtifactDiscoveryError(f"{ref.key}: UDB log not found: {log_path}")

    config = _read_toml(config_path)
    inference_value = config.get("results", {}).get("inference_dir")
    if not inference_value:
        inference_value = config.get("vector_db", {}).get("infer_results_dir")
    inference_dir = _resolved_existing_dir(
        inference_value,
        field_name="inference directory",
        ref=ref,
    )

    matches = _UMAP_RESULT_PATTERN.findall(log_path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ArtifactDiscoveryError(
            f"{ref.key}: no 'Saving UMAP results to ...' entry in {log_path}"
        )
    historical_umap_dir = _resolved_existing_dir(
        matches[-1].strip(),
        field_name="historical UMAP directory",
        ref=ref,
    )

    umap_section = config.get("umap", {})
    umap_params = umap_section.get("UMAP", {})
    n_components = int(umap_params.get("n_components", 2))
    if n_components != 2:
        raise ArtifactDiscoveryError(
            f"{ref.key}: historical config has n_components={n_components}; this notebook requires 2"
        )
    original_params = UmapParams(
        n_neighbors=umap_params.get("n_neighbors", 15),
        min_dist=umap_params.get("min_dist", 0.1),
        metric=umap_params.get("metric", "euclidean"),
    )

    configured_fit_sample_size = umap_section.get("fit_sample_size", 1_024)
    if configured_fit_sample_size is False or configured_fit_sample_size is None:
        configured_fit_sample_size = None
    else:
        configured_fit_sample_size = int(configured_fit_sample_size)
        if configured_fit_sample_size <= 0:
            raise ArtifactDiscoveryError(
                f"{ref.key}: fit_sample_size must be positive or false; got {configured_fit_sample_size}"
            )

    batch_size = int(config.get("data_loader", {}).get("batch_size", 256))
    if batch_size <= 0:
        raise ArtifactDiscoveryError(f"{ref.key}: data_loader.batch_size must be positive")

    return ExperimentArtifacts(
        ref=ref,
        config_path=config_path.resolve(),
        log_path=log_path.resolve(),
        inference_dir=inference_dir,
        historical_umap_dir=historical_umap_dir,
        original_params=original_params,
        configured_fit_sample_size=configured_fit_sample_size,
        batch_size=batch_size,
    )


def preflight(
    experiments: Iterable[ExperimentRef],
    run_base: str | Path,
) -> dict[ExperimentRef, ExperimentArtifacts]:
    """Resolve all experiments before any exploratory result is written."""

    refs = list(experiments)
    if not refs:
        raise ValueError("At least one experiment is required")
    if len(set(refs)) != len(refs):
        raise ValueError("Experiment references must be unique")

    resolved: dict[ExperimentRef, ExperimentArtifacts] = {}
    failures: list[str] = []
    for ref in refs:
        try:
            resolved[ref] = discover_experiment(ref, run_base)
        except ArtifactDiscoveryError as exc:
            failures.append(str(exc))
    if failures:
        raise ArtifactDiscoveryError("Preflight failed:\n- " + "\n- ".join(failures))
    return resolved


def artifacts_table(artifacts: Mapping[ExperimentRef, ExperimentArtifacts]):
    """Return a display-friendly pandas table of resolved experiment artifacts."""

    import pandas as pd

    return pd.DataFrame([artifact.as_record() for artifact in artifacts.values()])


def historical_runs(
    artifacts: Mapping[ExperimentRef, ExperimentArtifacts],
) -> dict[ExperimentRef, UmapRun]:
    """Wrap historical results without copying or modifying them."""

    return {
        ref: UmapRun(
            artifact=artifact,
            result_dir=artifact.historical_umap_dir,
            params=artifact.original_params,
            label="historical (seed unknown)",
            historical=True,
        )
        for ref, artifact in artifacts.items()
    }


def fixed_sample_indices(total_size: int, sample_size: int, seed: int) -> "Any":
    """Return a sorted deterministic sample of row indexes."""

    import numpy as np

    total_size = int(total_size)
    sample_size = int(sample_size)
    if total_size <= 0:
        raise ValueError("total_size must be positive")
    if not 0 < sample_size <= total_size:
        raise ValueError(
            f"sample_size must satisfy 0 < sample_size <= total_size; got {sample_size}, {total_size}"
        )
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(total_size, size=sample_size, replace=False).astype(np.int64))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_fingerprint(artifact: ExperimentArtifacts) -> str:
    index_path = artifact.inference_dir / "batch_index.npy"
    return _hash_json(
        {
            "inference_dir": str(artifact.inference_dir),
            "batch_index_sha256": _sha256_file(index_path),
        }
    )


def _safe_component(value: str, *, name: str) -> str:
    text = _SAFE_COMPONENT_PATTERN.sub("-", str(value).strip()).strip(".-")
    if not text:
        raise ValueError(f"{name} must contain at least one filename-safe character")
    return text


def _package_versions() -> dict[str, str]:
    packages = {
        "hyrax": "hyrax",
        "numpy": "numpy",
        "scikit_learn": "scikit-learn",
        "umap_learn": "umap-learn",
    }
    versions: dict[str, str] = {}
    for key, package in packages.items():
        try:
            versions[key] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = "unknown"
    return versions


def _resolve_sample_size(artifact: ExperimentArtifacts, total_size: int) -> int:
    configured = artifact.configured_fit_sample_size
    return total_size if configured is None else min(configured, total_size)


def _hyrax_from_config(config_path: Path):
    from hyrax import Hyrax

    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        return Hyrax(config_file=config_path, setup_logging=False)
    finally:
        logging.disable(previous_disable_level)


def _open_inference_dataset(artifact: ExperimentArtifacts):
    from hyrax.data_sets.inference_dataset import InferenceDataSet

    hyrax_instance = _hyrax_from_config(artifact.config_path)
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        dataset = InferenceDataSet(hyrax_instance.config, results_dir=artifact.inference_dir)
    finally:
        logging.disable(previous_disable_level)
    return hyrax_instance, dataset


def _sample_file(
    experiment_dir: Path,
    *,
    kind: str,
    source_fingerprint: str,
    sample_size: int,
    seed: int,
) -> Path:
    return (
        experiment_dir
        / "_samples"
        / f"{kind}__n{sample_size}__seed{seed:04d}__{source_fingerprint[:8]}.npz"
    )


def _load_or_create_sample(
    dataset,
    sample_path: Path,
    *,
    sample_size: int,
    seed: int,
) -> tuple[Any, Any]:
    import numpy as np

    all_ids = np.asarray(list(dataset.ids()), dtype=str)
    if sample_path.is_file():
        with np.load(sample_path, allow_pickle=False) as stored:
            indices = np.asarray(stored["indices"], dtype=np.int64)
            ids = np.asarray(stored["ids"], dtype=str)
        if len(indices) != sample_size or len(ids) != sample_size:
            raise ExplorationError(f"Stored sample has the wrong length: {sample_path}")
        if np.any(indices < 0) or np.any(indices >= len(all_ids)):
            raise ExplorationError(f"Stored sample has out-of-range indexes: {sample_path}")
        if not np.array_equal(all_ids[indices], ids):
            raise ExplorationError(
                f"Stored sample IDs no longer match the source inference dataset: {sample_path}"
            )
        return indices, ids

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    indices = fixed_sample_indices(len(all_ids), sample_size, seed)
    ids = all_ids[indices]
    temp_path = sample_path.with_name(f".{sample_path.name}.{os.getpid()}.tmp.npz")
    np.savez(temp_path, indices=indices, ids=ids)
    os.replace(temp_path, sample_path)
    return indices, ids


def _cache_identity(
    artifact: ExperimentArtifacts,
    params: UmapParams,
    *,
    seed: int,
    fit_sample_size: int,
    fit_sample_ids_hash: str,
    source_fingerprint: str,
    versions: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "experiment": asdict(artifact.ref),
        "source_fingerprint": source_fingerprint,
        "params": asdict(params),
        "seed": int(seed),
        "transform_seed": int(seed),
        "fit_sample_size": int(fit_sample_size),
        "fit_sample_ids_hash": fit_sample_ids_hash,
        "versions": dict(versions),
    }


def _float_slug(value: float) -> str:
    return f"{float(value):.3f}".replace("-", "m").replace(".", "p")


def _variant_slug(params: UmapParams, seed: int, cache_key: str) -> str:
    metric = _safe_component(params.metric, name="metric")
    return (
        f"nn{params.n_neighbors:03d}__md{_float_slug(params.min_dist)}__"
        f"metric-{metric}__seed{seed:04d}__{cache_key[:8]}"
    )


def session_directory(output_root: str | Path, session: str) -> Path:
    """Resolve the isolated directory for a named exploration session."""

    return Path(output_root).expanduser().resolve() / _safe_component(session, name="session")


def _read_provenance(result_dir: Path) -> dict[str, Any]:
    provenance_path = result_dir / "provenance.json"
    if not provenance_path.is_file():
        raise ExplorationError(f"Missing provenance file: {provenance_path}")
    return json.loads(provenance_path.read_text(encoding="utf-8"))


def _cached_run(
    target_dir: Path,
    artifact: ExperimentArtifacts,
    params: UmapParams,
    label: str,
    seed: int,
    cache_key: str,
) -> UmapRun | None:
    if not target_dir.exists():
        return None
    if not target_dir.is_dir() or not (target_dir / "_SUCCESS").is_file():
        raise ExplorationError(
            f"Cache target exists but is incomplete; move it aside before retrying: {target_dir}"
        )
    provenance = _read_provenance(target_dir)
    if provenance.get("cache_key") != cache_key:
        raise ExplorationError(f"Cache-key mismatch in completed result: {target_dir}")
    diagnostics = provenance.get("diagnostics") or {}
    return UmapRun(
        artifact=artifact,
        result_dir=target_dir,
        params=params,
        label=label,
        seed=seed,
        reused=True,
        diagnostics=diagnostics,
    )


def _batch_files(result_dir: Path) -> list[Path]:
    matches: list[tuple[int, Path]] = []
    for path in result_dir.glob("batch_*.npy"):
        match = re.fullmatch(r"batch_(\d+)\.npy", path.name)
        if match:
            matches.append((int(match.group(1)), path))
    return [path for _, path in sorted(matches)]


def load_embedding_arrays(result_dir: str | Path) -> tuple[Any, Any]:
    """Load object IDs and 2D coordinates from a Hyrax result directory."""

    import numpy as np

    result_dir = Path(result_dir)
    batch_files = _batch_files(result_dir)
    if not batch_files:
        raise ExplorationError(f"No batch_N.npy files found in {result_dir}")
    batches = [np.load(path, allow_pickle=False) for path in batch_files]
    ids = np.concatenate([batch["id"].astype(str) for batch in batches])
    points = np.concatenate([np.asarray(batch["tensor"]) for batch in batches], axis=0)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ExplorationError(f"Expected a (N, 2) embedding in {result_dir}; got {points.shape}")
    if len(ids) != len(points):
        raise ExplorationError(f"ID/coordinate length mismatch in {result_dir}")
    if not np.all(np.isfinite(points)):
        raise ExplorationError(f"Embedding contains non-finite coordinates: {result_dir}")
    return ids, points


def _neighbor_rows(indices: Any, *, k: int) -> Any:
    """Drop each row's self-index and retain the first k neighbors."""

    import numpy as np

    rows: list[Any] = []
    for row_number, row in enumerate(indices):
        without_self = row[row != row_number]
        rows.append(without_self[:k])
    return np.asarray(rows, dtype=np.int64)


def projection_diagnostics(
    artifact: ExperimentArtifacts,
    result_dir: str | Path,
    *,
    experiment_dir: str | Path,
    seed: int = DEFAULT_RANDOM_SEED,
    evaluation_size: int = DEFAULT_EVALUATION_SIZE,
    k: int = DEFAULT_DIAGNOSTIC_K,
    dataset: Any | None = None,
) -> dict[str, Any]:
    """Measure local-neighborhood preservation on a fixed evaluation sample.

    These values assess the 2D projection relative to the original inference
    vectors.  They do not establish whether the learned representation is
    scientifically meaningful.
    """

    import numpy as np
    from sklearn.manifold import trustworthiness
    from sklearn.neighbors import NearestNeighbors

    if dataset is None:
        _, dataset = _open_inference_dataset(artifact)

    total_size = len(dataset)
    eval_size = min(int(evaluation_size), total_size)
    if eval_size < 4:
        return {
            "diagnostic_metric": artifact.original_params.metric,
            "evaluation_size": eval_size,
            "diagnostic_k": None,
            "trustworthiness": math.nan,
            "knn_retention": math.nan,
            "diagnostic_warning": "At least four points are required for diagnostics.",
        }

    source_fingerprint = _source_fingerprint(artifact)
    eval_path = _sample_file(
        Path(experiment_dir),
        kind="evaluation",
        source_fingerprint=source_fingerprint,
        sample_size=eval_size,
        seed=int(seed) + 1,
    )
    eval_indices, eval_ids = _load_or_create_sample(
        dataset,
        eval_path,
        sample_size=eval_size,
        seed=int(seed) + 1,
    )

    high_dim = dataset[eval_indices].numpy().reshape((eval_size, -1))
    result_ids, all_low_dim = load_embedding_arrays(result_dir)
    id_to_low_index = {str(object_id): idx for idx, object_id in enumerate(result_ids)}
    missing = [object_id for object_id in eval_ids if str(object_id) not in id_to_low_index]
    if missing:
        raise ExplorationError(
            f"{len(missing)} diagnostic IDs are missing from {result_dir}; examples: {missing[:5]}"
        )
    low_indices = np.asarray([id_to_low_index[str(object_id)] for object_id in eval_ids])
    low_dim = all_low_dim[low_indices]

    k_effective = min(int(k), eval_size - 1)
    trust_k = min(k_effective, max(1, (eval_size - 1) // 2))
    metric = artifact.original_params.metric

    try:
        trust = float(
            trustworthiness(
                high_dim,
                low_dim,
                n_neighbors=trust_k,
                metric=metric,
            )
        )
        high_raw = NearestNeighbors(n_neighbors=k_effective + 1, metric=metric).fit(
            high_dim
        ).kneighbors(high_dim, return_distance=False)
        low_raw = NearestNeighbors(n_neighbors=k_effective + 1, metric="euclidean").fit(
            low_dim
        ).kneighbors(low_dim, return_distance=False)
        high_neighbors = _neighbor_rows(high_raw, k=k_effective)
        low_neighbors = _neighbor_rows(low_raw, k=k_effective)
        retention = float(
            np.mean(
                [
                    len(set(high_row.tolist()).intersection(low_row.tolist())) / k_effective
                    for high_row, low_row in zip(high_neighbors, low_neighbors, strict=True)
                ]
            )
        )
        warning = None
    except Exception as exc:
        trust = math.nan
        retention = math.nan
        warning = f"Diagnostics could not use metric {metric!r}: {exc}"

    result = {
        "diagnostic_metric": metric,
        "evaluation_size": eval_size,
        "diagnostic_k": k_effective,
        "trustworthiness": trust,
        "knn_retention": retention,
    }
    if warning:
        result["diagnostic_warning"] = warning
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_json(temp_path, value)
    os.replace(temp_path, path)


def _validate_written_result(result_dir: Path, expected_ids: Sequence[str]) -> None:
    import numpy as np

    output_ids, points = load_embedding_arrays(result_dir)
    if points.shape != (len(expected_ids), 2):
        raise ExplorationError(
            f"Written embedding has shape {points.shape}; expected ({len(expected_ids)}, 2)"
        )
    index = np.load(result_dir / "batch_index.npy", allow_pickle=False)
    if len(index) != len(expected_ids):
        raise ExplorationError("Written batch index length does not match the source inference data")
    if sorted(output_ids.tolist()) != sorted(str(value) for value in expected_ids):
        raise ExplorationError("Written embedding object IDs do not match the source inference data")


def _compute_embedding(
    artifact: ExperimentArtifacts,
    params: UmapParams,
    *,
    seed: int,
    fit_indices: Any,
    output_dir: Path,
    dataset: Any,
) -> Any:
    import numpy as np
    import umap
    from hyrax.data_sets.inference_dataset import InferenceDataSetWriter
    from tqdm.auto import tqdm

    fit_sample_size = len(fit_indices)
    params.validate(fit_sample_size)
    fit_vectors = dataset[fit_indices].numpy().reshape((fit_sample_size, -1))
    if not np.all(np.isfinite(fit_vectors)):
        raise ExplorationError(f"{artifact.ref.key}: fit sample contains non-finite values")

    try:
        reducer = umap.UMAP(**params.as_umap_kwargs(seed))
        reducer.fit(fit_vectors)
    except Exception as exc:
        raise ExplorationError(
            f"{artifact.ref.key}: UMAP fit failed for {params}: {exc}"
        ) from exc

    del fit_vectors
    all_ids = np.asarray(list(dataset.ids()))
    writer = InferenceDataSetWriter(dataset, output_dir)
    transformed_batches: list[Any] = []
    batch_starts = range(0, len(dataset), artifact.batch_size)

    try:
        for start in tqdm(
            batch_starts,
            total=math.ceil(len(dataset) / artifact.batch_size),
            desc=f"{artifact.ref.key}: transform",
        ):
            stop = min(start + artifact.batch_size, len(dataset))
            batch_indices = np.arange(start, stop, dtype=np.int64)
            vectors = dataset[batch_indices].numpy().reshape((len(batch_indices), -1))
            try:
                transformed = np.asarray(reducer.transform(vectors))
            except Exception as exc:
                raise ExplorationError(
                    f"{artifact.ref.key}: UMAP transform failed for rows {start}:{stop}: {exc}"
                ) from exc
            if transformed.shape != (len(batch_indices), 2):
                raise ExplorationError(
                    f"{artifact.ref.key}: UMAP returned {transformed.shape}; "
                    f"expected ({len(batch_indices)}, 2)"
                )
            if not np.all(np.isfinite(transformed)):
                raise ExplorationError(
                    f"{artifact.ref.key}: UMAP returned non-finite coordinates "
                    f"for rows {start}:{stop}"
                )
            writer.write_batch(all_ids[batch_indices], transformed)
            transformed_batches.append(transformed)

        writer.write_index()
    except Exception:
        # InferenceDataSetWriter owns a multiprocessing pool. Avoid leaving its
        # workers behind when a transform fails before write_index closes it.
        pool = getattr(writer, "writer_pool", None)
        if pool is not None:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
        raise
    _validate_written_result(output_dir, all_ids.astype(str).tolist())
    return np.concatenate(transformed_batches, axis=0)


def _provenance_record(
    *,
    artifact: ExperimentArtifacts,
    params: UmapParams,
    label: str,
    session: str,
    result_dir: Path,
    cache_key: str,
    cache_identity: Mapping[str, Any],
    fit_sample_file: Path,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "session": session,
        "label": label,
        "experiment": asdict(artifact.ref),
        "params": asdict(params),
        "seed": cache_identity["seed"],
        "transform_seed": cache_identity["transform_seed"],
        "fit_sample_size": cache_identity["fit_sample_size"],
        "fit_sample_ids_hash": cache_identity["fit_sample_ids_hash"],
        "fit_sample_file": os.path.relpath(fit_sample_file, result_dir.parent),
        "source": {
            "config_path": str(artifact.config_path),
            "log_path": str(artifact.log_path),
            "inference_dir": str(artifact.inference_dir),
            "historical_umap_dir": str(artifact.historical_umap_dir),
            "source_fingerprint": cache_identity["source_fingerprint"],
        },
        "versions": cache_identity["versions"],
        "result_dir": str(result_dir),
        "diagnostics": dict(diagnostics),
        "diagnostic_scope": (
            "Projection fidelity relative to the inference vectors; these metrics do not assess "
            "the scientific or semantic quality of the learned representation."
        ),
    }


def _flatten_manifest_record(provenance: Mapping[str, Any]) -> dict[str, Any]:
    experiment = provenance.get("experiment", {})
    params = provenance.get("params", {})
    source = provenance.get("source", {})
    versions = provenance.get("versions", {})
    diagnostics = provenance.get("diagnostics", {})
    return {
        "session": provenance.get("session"),
        "run": experiment.get("run"),
        "expt": experiment.get("expt"),
        "label": provenance.get("label"),
        "n_neighbors": params.get("n_neighbors"),
        "min_dist": params.get("min_dist"),
        "metric": params.get("metric"),
        "n_components": params.get("n_components"),
        "seed": provenance.get("seed"),
        "fit_sample_size": provenance.get("fit_sample_size"),
        "diagnostic_metric": diagnostics.get("diagnostic_metric"),
        "trustworthiness": diagnostics.get("trustworthiness"),
        "knn_retention": diagnostics.get("knn_retention"),
        "evaluation_size": diagnostics.get("evaluation_size"),
        "result_dir": provenance.get("result_dir"),
        "source_inference_dir": source.get("inference_dir"),
        "source_config": source.get("config_path"),
        "created_at_utc": provenance.get("created_at_utc"),
        "umap_learn_version": versions.get("umap_learn"),
        "cache_key": provenance.get("cache_key"),
    }


def manifest_records(session_dir: str | Path) -> list[dict[str, Any]]:
    """Rebuild manifest records from completed per-run provenance files."""

    session_dir = Path(session_dir)
    records: list[dict[str, Any]] = []
    if not session_dir.exists():
        return records
    for provenance_path in sorted(session_dir.glob("run*_expt*/*/provenance.json")):
        result_dir = provenance_path.parent
        if not (result_dir / "_SUCCESS").is_file():
            continue
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        records.append(_flatten_manifest_record(provenance))
    return records


def rebuild_manifest(session_dir: str | Path) -> Path:
    """Atomically rebuild ``manifest.csv`` from completed provenance records."""

    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    records = manifest_records(session_dir)
    manifest_path = session_dir / "manifest.csv"
    fieldnames = list(_flatten_manifest_record({}).keys())
    temp_path = session_dir / f".manifest.{os.getpid()}.tmp.csv"
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temp_path, manifest_path)
    return manifest_path


def manifest_table(session_dir: str | Path):
    """Return the rebuilt session manifest as a pandas DataFrame."""

    import pandas as pd

    return pd.read_csv(rebuild_manifest(session_dir))


def run_selected(
    artifact: ExperimentArtifacts,
    params: UmapParams,
    *,
    output_root: str | Path,
    session: str,
    seed: int = DEFAULT_RANDOM_SEED,
    label: str = "parameter variant",
    evaluation_size: int = DEFAULT_EVALUATION_SIZE,
    diagnostic_k: int = DEFAULT_DIAGNOSTIC_K,
) -> UmapRun:
    """Create or reuse one deterministic UMAP result."""

    import numpy as np

    seed = int(seed)
    session_name = _safe_component(session, name="session")
    session_dir = session_directory(output_root, session_name)
    experiment_dir = session_dir / artifact.ref.key
    experiment_dir.mkdir(parents=True, exist_ok=True)

    _, dataset = _open_inference_dataset(artifact)
    total_size = len(dataset)
    fit_sample_size = _resolve_sample_size(artifact, total_size)
    params.validate(fit_sample_size)
    source_fingerprint = _source_fingerprint(artifact)
    fit_sample_path = _sample_file(
        experiment_dir,
        kind="fit",
        source_fingerprint=source_fingerprint,
        sample_size=fit_sample_size,
        seed=seed,
    )
    fit_indices, fit_ids = _load_or_create_sample(
        dataset,
        fit_sample_path,
        sample_size=fit_sample_size,
        seed=seed,
    )
    fit_sample_ids_hash = _hash_json(np.asarray(fit_ids, dtype=str).tolist())
    versions = _package_versions()
    identity = _cache_identity(
        artifact,
        params,
        seed=seed,
        fit_sample_size=fit_sample_size,
        fit_sample_ids_hash=fit_sample_ids_hash,
        source_fingerprint=source_fingerprint,
        versions=versions,
    )
    cache_key = _hash_json(identity)
    target_dir = experiment_dir / _variant_slug(params, seed, cache_key)
    cached = _cached_run(target_dir, artifact, params, label, seed, cache_key)
    if cached is not None:
        expected_evaluation_size = min(int(evaluation_size), total_size)
        expected_k = (
            min(int(diagnostic_k), expected_evaluation_size - 1)
            if expected_evaluation_size >= 4
            else None
        )
        cached_diagnostics = dict(cached.diagnostics)
        if (
            cached_diagnostics.get("evaluation_size") != expected_evaluation_size
            or cached_diagnostics.get("diagnostic_k") != expected_k
        ):
            cached_diagnostics = projection_diagnostics(
                artifact,
                target_dir,
                experiment_dir=experiment_dir,
                seed=seed,
                evaluation_size=evaluation_size,
                k=diagnostic_k,
                dataset=dataset,
            )
            provenance = _read_provenance(target_dir)
            provenance["diagnostics"] = cached_diagnostics
            provenance["diagnostics_updated_at_utc"] = dt.datetime.now(
                dt.timezone.utc
            ).isoformat()
            _write_json_atomic(target_dir / "provenance.json", provenance)
            cached = UmapRun(
                artifact=artifact,
                result_dir=target_dir,
                params=params,
                label=label,
                seed=seed,
                reused=True,
                diagnostics=cached_diagnostics,
            )
        rebuild_manifest(session_dir)
        return cached

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_dir.name}.incomplete-",
            dir=experiment_dir,
        )
    )
    try:
        _compute_embedding(
            artifact,
            params,
            seed=seed,
            fit_indices=fit_indices,
            output_dir=temp_dir,
            dataset=dataset,
        )
        diagnostics = projection_diagnostics(
            artifact,
            temp_dir,
            experiment_dir=experiment_dir,
            seed=seed,
            evaluation_size=evaluation_size,
            k=diagnostic_k,
            dataset=dataset,
        )
        provenance = _provenance_record(
            artifact=artifact,
            params=params,
            label=label,
            session=session_name,
            result_dir=target_dir,
            cache_key=cache_key,
            cache_identity=identity,
            fit_sample_file=fit_sample_path,
            diagnostics=diagnostics,
        )
        _write_json(temp_dir / "provenance.json", provenance)
        (temp_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
        temp_dir.rename(target_dir)
    except Exception as exc:
        _write_json(
            temp_dir / "failure.json",
            {
                "failed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "experiment": asdict(artifact.ref),
                "params": asdict(params),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    rebuild_manifest(session_dir)
    return UmapRun(
        artifact=artifact,
        result_dir=target_dir,
        params=params,
        label=label,
        seed=seed,
        diagnostics=diagnostics,
    )


def run_all(
    artifacts: Mapping[ExperimentRef, ExperimentArtifacts],
    params: UmapParams,
    *,
    output_root: str | Path,
    session: str,
    seed: int = DEFAULT_RANDOM_SEED,
    label: str = "parameter variant",
    evaluation_size: int = DEFAULT_EVALUATION_SIZE,
    diagnostic_k: int = DEFAULT_DIAGNOSTIC_K,
) -> dict[ExperimentRef, UmapRun]:
    """Sequentially create or reuse the same parameter variant for all experiments."""

    runs: dict[ExperimentRef, UmapRun] = {}
    for ref, artifact in artifacts.items():
        runs[ref] = run_selected(
            artifact,
            params,
            output_root=output_root,
            session=session,
            seed=seed,
            label=label,
            evaluation_size=evaluation_size,
            diagnostic_k=diagnostic_k,
        )
    return runs


def run_parameter_sweep(
    artifact: ExperimentArtifacts,
    parameter: str,
    values: Sequence[Any],
    *,
    output_root: str | Path,
    session: str,
    seed: int = DEFAULT_RANDOM_SEED,
    baseline_params: UmapParams | None = None,
    evaluation_size: int = DEFAULT_EVALUATION_SIZE,
    diagnostic_k: int = DEFAULT_DIAGNOSTIC_K,
) -> dict[Any, UmapRun]:
    """Run one-at-a-time variants of a single UMAP parameter.

    Parameters not named by ``parameter`` are copied from ``baseline_params``;
    by default, the experiment's historical parameters are the baseline.  The
    returned mapping preserves the requested value order and uses normalized
    parameter values as keys, making notebook selections such as
    ``n_neighbors_sweep[30]`` straightforward.  Existing completed variants
    are reused through :func:`run_selected`.
    """

    allowed_parameters = {"n_neighbors", "min_dist", "metric"}
    if parameter not in allowed_parameters:
        raise ValueError(
            f"parameter must be one of {sorted(allowed_parameters)}; got {parameter!r}"
        )

    requested_values = list(values)
    if not requested_values:
        raise ValueError("values must contain at least one sweep value")

    baseline = baseline_params or artifact.original_params
    baseline_record = {
        "n_neighbors": baseline.n_neighbors,
        "min_dist": baseline.min_dist,
        "metric": baseline.metric,
    }
    sweep_specs: list[tuple[Any, UmapParams]] = []
    normalized_values: set[Any] = set()
    for requested_value in requested_values:
        parameter_record = dict(baseline_record)
        parameter_record[parameter] = requested_value
        params = UmapParams(**parameter_record)
        normalized_value = getattr(params, parameter)
        if normalized_value in normalized_values:
            raise ValueError(
                f"Duplicate normalized {parameter} sweep value: {normalized_value!r}"
            )
        normalized_values.add(normalized_value)
        sweep_specs.append((normalized_value, params))

    runs: dict[Any, UmapRun] = {}
    for normalized_value, params in sweep_specs:
        runs[normalized_value] = run_selected(
            artifact,
            params,
            output_root=output_root,
            session=session,
            seed=seed,
            label=f"{parameter}={normalized_value}",
            evaluation_size=evaluation_size,
            diagnostic_k=diagnostic_k,
        )
    return runs


def ensure_controls(
    artifacts: Mapping[ExperimentRef, ExperimentArtifacts],
    *,
    output_root: str | Path,
    session: str,
    seed: int = DEFAULT_RANDOM_SEED,
    evaluation_size: int = DEFAULT_EVALUATION_SIZE,
    diagnostic_k: int = DEFAULT_DIAGNOSTIC_K,
) -> dict[ExperimentRef, UmapRun]:
    """Create deterministic controls using each experiment's historical UMAP settings."""

    controls: dict[ExperimentRef, UmapRun] = {}
    for ref, artifact in artifacts.items():
        controls[ref] = run_selected(
            artifact,
            artifact.original_params,
            output_root=output_root,
            session=session,
            seed=seed,
            label="deterministic control",
            evaluation_size=evaluation_size,
            diagnostic_k=diagnostic_k,
        )
    return controls


def diagnose_runs(
    runs: Mapping[ExperimentRef, UmapRun],
    *,
    output_root: str | Path,
    session: str,
    seed: int = DEFAULT_RANDOM_SEED,
    evaluation_size: int = DEFAULT_EVALUATION_SIZE,
    diagnostic_k: int = DEFAULT_DIAGNOSTIC_K,
) -> dict[ExperimentRef, UmapRun]:
    """Attach diagnostics to historical or previously loaded run wrappers."""

    diagnosed: dict[ExperimentRef, UmapRun] = {}
    session_dir = session_directory(output_root, session)
    for ref, run in runs.items():
        diagnostics = projection_diagnostics(
            run.artifact,
            run.result_dir,
            experiment_dir=session_dir / ref.key,
            seed=seed,
            evaluation_size=evaluation_size,
            k=diagnostic_k,
        )
        diagnosed[ref] = UmapRun(
            artifact=run.artifact,
            result_dir=run.result_dir,
            params=run.params,
            label=run.label,
            seed=run.seed,
            reused=run.reused,
            historical=run.historical,
            diagnostics=diagnostics,
        )
    return diagnosed


def runs_table(runs: Mapping[Any, UmapRun]):
    """Return one display row per historical, control, variant, or sweep run."""

    import pandas as pd

    return pd.DataFrame([run.as_record() for run in runs.values()])


def plot_grid(
    runs: Mapping[ExperimentRef, UmapRun] | Sequence[UmapRun],
    *,
    suptitle: str | None = None,
    ncols: int = 2,
    point_size: float = 1.0,
    alpha: float = 0.45,
    figsize: tuple[float, float] | None = None,
):
    """Plot a lightweight rasterized comparison grid for UMAP runs."""

    import matplotlib.pyplot as plt

    run_list = list(runs.values()) if isinstance(runs, Mapping) else list(runs)
    if not run_list:
        raise ValueError("At least one UMAP run is required")
    ncols = max(1, int(ncols))
    nrows = math.ceil(len(run_list) / ncols)
    if figsize is None:
        figsize = (5.0 * ncols, 4.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for ax, run in zip(axes.flat, run_list, strict=False):
        _, points = load_embedding_arrays(run.result_dir)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=point_size,
            alpha=alpha,
            rasterized=True,
            linewidths=0,
        )
        title = (
            f"Run {run.artifact.ref.run}, Expt {run.artifact.ref.expt} — {run.label}\n"
            f"nn={run.params.n_neighbors}, min_dist={run.params.min_dist:g}, "
            f"metric={run.params.metric}"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_aspect("equal", adjustable="datalim")

    for ax in list(axes.flat)[len(run_list) :]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    else:
        fig.tight_layout()
    return fig, axes


def build_merger_time_overlay(
    run: UmapRun,
    catalog: Any,
    merger_type: str,
    cutoff_gyr: float,
    *,
    catalog_id_column: str | None = None,
) -> MergerTimeOverlay:
    """Prepare one mini/minor/major time-since-merger overlay.

    The returned catalog contains only IDs that both satisfy the requested
    time window and occur in ``run``.  IDs are copied from the UMAP result so
    Hyrax's overlay cross-match uses exactly the same representation as the
    plotted points.  ``target_rule`` uses the same inclusive time window for
    completeness-purity scoring.
    """

    import numpy as np
    import pandas as pd

    import static_umap_metrics as metrics

    if not isinstance(run, UmapRun):
        raise TypeError("run must be a UmapRun")
    if catalog is None or not hasattr(catalog, "columns"):
        raise TypeError("catalog must be a pandas-like table with columns")

    normalized_type = str(merger_type).strip().lower()
    styles = {
        "major": {
            "color": "#d62728",
            "static_marker": "x",
            "visualizer_marker": "x",
        },
        "minor": {
            "color": "#2ca02c",
            "static_marker": "s",
            "visualizer_marker": "square",
        },
        "mini": {
            "color": "#1f77b4",
            "static_marker": ".",
            "visualizer_marker": "circle",
        },
    }
    if normalized_type not in styles:
        raise ValueError(
            "merger_type must be 'mini', 'minor', or 'major'; "
            f"got {merger_type!r}"
        )

    if isinstance(cutoff_gyr, bool):
        raise ValueError("cutoff_gyr must be a finite non-negative number")
    try:
        cutoff = float(cutoff_gyr)
    except (TypeError, ValueError) as exc:
        raise ValueError("cutoff_gyr must be a finite non-negative number") from exc
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError("cutoff_gyr must be a finite non-negative number")

    title_type = normalized_type.title()
    catalog_key = metrics.first_present_column(
        catalog,
        [
            f"{title_type}_TimeSinceMerger",
            f"{normalized_type}_time_since_merger",
        ],
    )
    if catalog_key is None:
        raise KeyError(
            f"No time-since-merger column found for {normalized_type!r}. Expected "
            f"'{title_type}_TimeSinceMerger' or "
            f"'{normalized_type}_time_since_merger'."
        )

    resolved_id_column = metrics.resolve_catalog_id_column(
        catalog,
        catalog_id_column,
    )
    values = pd.to_numeric(catalog[catalog_key], errors="coerce")
    known_mask = values.notna() & np.isfinite(values)
    labels = pd.DataFrame(
        {
            "_match_id": metrics.normalize_object_ids(catalog[resolved_id_column]),
            "_is_target": (known_mask & (values >= 0.0) & (values <= cutoff)),
            "_is_known": known_mask,
        }
    )
    known = labels.loc[
        labels["_match_id"].notna() & labels["_is_known"],
        ["_match_id", "_is_target"],
    ].drop_duplicates()

    if not known.empty:
        membership_counts = known.groupby("_match_id", sort=False)["_is_target"].nunique()
        conflicts = membership_counts[membership_counts > 1].index.tolist()
        if conflicts:
            raise ValueError(
                "Repeated catalog rows disagree on merger-time target membership for "
                f"{len(conflicts)} object IDs. First conflicts: {conflicts[:5]}"
            )

    target_ids = known.loc[known["_is_target"], "_match_id"].drop_duplicates()
    n_catalog_targets = len(target_ids)
    if n_catalog_targets == 0:
        raise ValueError(
            f"No catalog objects satisfy the {title_type} merger cutoff of {cutoff:g} Gyr"
        )

    umap_ids, _ = load_embedding_arrays(run.result_dir)
    normalized_umap_ids = metrics.normalize_object_ids(umap_ids)
    if normalized_umap_ids.isna().any():
        raise ValueError("UMAP contains missing object IDs")
    duplicated_umap_ids = normalized_umap_ids.duplicated(keep=False)
    if duplicated_umap_ids.any():
        duplicates = (
            normalized_umap_ids[duplicated_umap_ids]
            .drop_duplicates()
            .head()
            .tolist()
        )
        raise ValueError(
            "UMAP object IDs must be unique for merger overlays. "
            f"First duplicate IDs: {duplicates}"
        )

    target_id_set = set(target_ids.tolist())
    matched_mask = normalized_umap_ids.isin(target_id_set).to_numpy(dtype=bool)
    matched_umap_ids = np.asarray(umap_ids)[matched_mask].astype(str)
    n_umap_targets = len(matched_umap_ids)
    if n_umap_targets == 0:
        raise ValueError(
            f"No {title_type} merger targets at <= {cutoff:g} Gyr occur in {run.artifact.ref.key}"
        )

    label = f"{title_type} merger <= {cutoff:g} Gyr ago"
    style = styles[normalized_type]
    target_rule = {
        "key": catalog_key,
        "min_value": 0.0,
        "max_value": cutoff,
        "include_min": True,
        "include_max": True,
        "color": style["color"],
        "marker": style["static_marker"],
        "label": label,
        "s": 12,
    }
    visualizer_overlay = {
        "catalog": pd.DataFrame({"object_id": matched_umap_ids}),
        "id_column": "object_id",
        "color": style["color"],
        "marker": style["visualizer_marker"],
        "size": 10,
        "label": label,
    }
    return MergerTimeOverlay(
        merger_type=normalized_type,
        cutoff_gyr=cutoff,
        catalog_key=catalog_key,
        label=label,
        target_rule=target_rule,
        visualizer_overlay=visualizer_overlay,
        n_catalog_targets=n_catalog_targets,
        n_umap_targets=n_umap_targets,
    )


def open_visualizer(
    run: UmapRun,
    *,
    display_images: bool = False,
    overlays: Sequence[Mapping[str, Any]] | None = None,
    **plot_options: Any,
):
    """Open a historical/control/variant result through Hyrax's visualize verb."""

    hyrax_instance = _hyrax_from_config(run.artifact.config_path)
    hyrax_instance.config["visualize"]["display_images"] = bool(display_images)
    pane, visualizer = hyrax_instance.visualize(
        input_dir=run.result_dir,
        return_verb=True,
        overlays=list(overlays) if overlays is not None else None,
        **plot_options,
    )
    return pane, visualizer


def _selection_metadata(run: UmapRun, selection_name: str) -> dict[str, Any]:
    return {
        "selection_name": selection_name,
        "run": run.artifact.ref.run,
        "expt": run.artifact.ref.expt,
        "umap_label": run.label,
        "result_dir": str(run.result_dir),
        "n_neighbors": run.params.n_neighbors,
        "min_dist": run.params.min_dist,
        "metric": run.params.metric,
        "seed": run.seed,
        "historical": run.historical,
    }


def _prepend_metadata(frame: Any, metadata: Mapping[str, Any]):
    result = frame.copy()
    for column, value in reversed(tuple(metadata.items())):
        result.insert(0, column, value)
    return result


def _visualizer_selected_dataframe(visualizer: Any):
    """Return selected rows, including a fallback for metadata-free Hyrax runs.

    Hyrax ``Visualize.get_selected_df`` currently constructs an invalid pandas
    frame when ``data_fields`` is empty. The visualizer selection streams have
    already populated ``points_id`` and ``points`` in that case, so build the
    equivalent three-column frame directly without changing the Hyrax checkout.
    """
    import numpy as np
    import pandas as pd

    data_fields = getattr(visualizer, "data_fields", None)
    if data_fields != []:
        selected_df = visualizer.get_selected_df()
        if not isinstance(selected_df, pd.DataFrame):
            raise TypeError("visualizer.get_selected_df() must return a pandas DataFrame")
        return selected_df

    object_id_column = getattr(visualizer, "object_id_column_name", "object_id")
    points_id = np.asarray(getattr(visualizer, "points_id", np.array([])))
    points = np.asarray(getattr(visualizer, "points", np.array([])))
    if points_id.ndim != 1:
        raise ValueError(
            f"visualizer.points_id must be one-dimensional; got shape {points_id.shape}"
        )
    if len(points_id) == 0:
        return pd.DataFrame(columns=[object_id_column, "x", "y"])
    if points.ndim != 2 or points.shape[1] < 2 or len(points) != len(points_id):
        raise ValueError(
            "visualizer selection coordinates must have shape (N, 2) and align "
            f"with points_id; got points={points.shape}, points_id={points_id.shape}"
        )
    return pd.DataFrame(
        {
            object_id_column: points_id,
            "x": points[:, 0],
            "y": points[:, 1],
        }
    )


def evaluate_visualizer_selection(
    run: UmapRun,
    visualizer: Any,
    catalog: Any,
    overlays: Sequence[Mapping[str, Any]],
    *,
    selection_name: str,
    catalog_id_column: str | None = None,
) -> SelectionEvaluation:
    """Score the current ``get_selected_df`` result for one UMAP run.

    The returned summary has one row per overlay target. Selected objects are
    represented in long form so their known-label and target membership can be
    inspected for every target rule without mutating or saving notebook state.
    """
    import static_umap_metrics as metrics

    selection_name = str(selection_name).strip()
    if not selection_name:
        raise ValueError("selection_name must be a non-empty string")
    if not isinstance(run, UmapRun):
        raise TypeError("run must be a UmapRun")
    if not hasattr(visualizer, "get_selected_df"):
        raise TypeError("visualizer must provide get_selected_df()")

    selected_df = _visualizer_selected_dataframe(visualizer)

    selected_id_column = getattr(visualizer, "object_id_column_name", None)
    id_candidates = [
        selected_id_column,
        "object_id",
        "rubin_object_id",
        "objectId",
        "objectId_data",
        "id",
    ]
    selected_id_column = next(
        (candidate for candidate in id_candidates if candidate and candidate in selected_df.columns),
        None,
    )
    if selected_id_column is None:
        raise KeyError(
            "Could not find the visualizer object-ID column in get_selected_df() output"
        )

    umap_ids, _ = load_embedding_arrays(run.result_dir)
    score = metrics.compute_selection_completeness_purity(
        umap_ids,
        selected_df[selected_id_column].to_numpy(),
        catalog,
        overlays,
        catalog_id_column=catalog_id_column,
    )

    selected_rows = selected_df.reset_index(drop=True).copy()
    selected_rows["_match_id"] = metrics.normalize_object_ids(
        selected_rows[selected_id_column]
    ).to_numpy()
    duplicated_selection_ids = selected_rows["_match_id"].duplicated(keep=False)
    if duplicated_selection_ids.any():
        duplicates = (
            selected_rows.loc[duplicated_selection_ids, "_match_id"]
            .drop_duplicates()
            .head()
            .tolist()
        )
        raise ValueError(
            "get_selected_df() returned duplicate object IDs. "
            f"First duplicates: {duplicates}"
        )

    membership = score["selected_membership"].drop(columns=["object_id"])
    selected_objects = selected_rows.merge(
        membership,
        on="_match_id",
        how="inner",
        validate="one_to_many",
    ).drop(columns=["_match_id"])

    metadata = _selection_metadata(run, selection_name)
    summary = _prepend_metadata(score["summary"], metadata)
    selected_objects = _prepend_metadata(selected_objects, metadata)
    return SelectionEvaluation(
        selection_name=selection_name,
        run=run,
        summary=summary,
        selected_objects=selected_objects,
    )


def selection_results_table(evaluations: Iterable[SelectionEvaluation]):
    """Concatenate named selection summaries into one comparison table."""
    import pandas as pd

    evaluations = list(evaluations)
    if any(not isinstance(evaluation, SelectionEvaluation) for evaluation in evaluations):
        raise TypeError("evaluations must contain only SelectionEvaluation instances")
    if not evaluations:
        return pd.DataFrame()
    return pd.concat(
        [evaluation.summary for evaluation in evaluations],
        ignore_index=True,
        sort=False,
    )


def plot_selection_results(
    evaluations: Iterable[SelectionEvaluation],
    *,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
):
    """Plot named manual selections in faceted completeness-purity space."""
    import matplotlib.pyplot as plt
    import numpy as np

    table = selection_results_table(evaluations)
    if table.empty:
        raise ValueError("At least one SelectionEvaluation is required for plotting")
    if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1:
        raise ValueError("ncols must be a positive integer")

    target_labels = list(dict.fromkeys(table["target_label"].tolist()))
    ncols = min(ncols, len(target_labels))
    nrows = math.ceil(len(target_labels) / ncols)
    if figsize is None:
        figsize = (5.0 * ncols, 4.2 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    selection_names = list(dict.fromkeys(table["selection_name"].tolist()))
    cmap = plt.get_cmap("tab10")
    selection_colors = {
        name: cmap(index % 10) for index, name in enumerate(selection_names)
    }

    for ax, target_label in zip(axes.flat, target_labels, strict=False):
        target_rows = table.loc[table["target_label"] == target_label]
        plotted = 0
        for row in target_rows.itertuples(index=False):
            if not np.isfinite(row.completeness) or not np.isfinite(row.purity):
                continue
            ax.scatter(
                row.completeness,
                row.purity,
                color=selection_colors[row.selection_name],
                s=55,
                zorder=3,
            )
            ax.annotate(
                row.selection_name,
                (row.completeness, row.purity),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
            plotted += 1

        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No selection with known labels",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title(str(target_label))
        ax.set_xlabel("Completeness")
        ax.set_ylabel("Purity")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)

    for ax in list(axes.flat)[len(target_labels) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig, axes


__all__ = [
    "ArtifactDiscoveryError",
    "ExperimentArtifacts",
    "ExperimentRef",
    "ExplorationError",
    "MergerTimeOverlay",
    "ParameterValidationError",
    "SelectionEvaluation",
    "UmapParams",
    "UmapRun",
    "artifacts_table",
    "build_merger_time_overlay",
    "diagnose_runs",
    "discover_experiment",
    "ensure_controls",
    "evaluate_visualizer_selection",
    "fixed_sample_indices",
    "historical_runs",
    "load_embedding_arrays",
    "manifest_records",
    "manifest_table",
    "open_visualizer",
    "plot_grid",
    "plot_selection_results",
    "preflight",
    "projection_diagnostics",
    "rebuild_manifest",
    "run_all",
    "run_parameter_sweep",
    "run_selected",
    "runs_table",
    "selection_results_table",
    "session_directory",
]
