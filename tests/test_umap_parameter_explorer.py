from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import umap_parameter_explorer as explorer


def _write_discovery_fixture(root: Path, *, run: int = 3, expt: int = 7) -> tuple[Path, Path, Path]:
    run_base = root / "runs"
    run_dir = run_base / f"run{run}"
    inference_dir = root / "results" / "source-infer"
    umap_dir = root / "results" / "source-umap"
    run_dir.mkdir(parents=True)
    inference_dir.mkdir(parents=True)
    umap_dir.mkdir(parents=True)
    (inference_dir / "batch_index.npy").touch()
    (umap_dir / "batch_index.npy").touch()

    config = f"""
[results]
inference_dir = "{inference_dir}"

[data_loader]
batch_size = 64

[umap]
fit_sample_size = 50

[umap.UMAP]
n_components = 2
n_neighbors = 23
""".strip()
    (run_dir / f"udb{run}_{expt}.toml").write_text(config + "\n", encoding="utf-8")
    (run_dir / f"udb{run}_{expt}.txt").write_text(
        f"INFO Saving UMAP results to {umap_dir}\n",
        encoding="utf-8",
    )
    return run_base, inference_dir, umap_dir


def test_discover_experiment_resolves_artifacts_and_defaults(tmp_path: Path) -> None:
    run_base, inference_dir, umap_dir = _write_discovery_fixture(tmp_path)
    ref = explorer.ExperimentRef(3, 7)

    artifact = explorer.discover_experiment(ref, run_base)

    assert artifact.ref == ref
    assert artifact.inference_dir == inference_dir.resolve()
    assert artifact.historical_umap_dir == umap_dir.resolve()
    assert artifact.original_params == explorer.UmapParams(
        n_neighbors=23,
        min_dist=0.1,
        metric="euclidean",
    )
    assert artifact.configured_fit_sample_size == 50
    assert artifact.batch_size == 64


def test_preflight_reports_all_missing_experiments(tmp_path: Path) -> None:
    refs = [explorer.ExperimentRef(1, 1), explorer.ExperimentRef(2, 2)]
    with pytest.raises(explorer.ArtifactDiscoveryError) as exc_info:
        explorer.preflight(refs, tmp_path)
    message = str(exc_info.value)
    assert "run1_expt1" in message
    assert "run2_expt2" in message


def test_fixed_sample_is_stable_sorted_and_seeded() -> None:
    first = explorer.fixed_sample_indices(100, 20, 42)
    second = explorer.fixed_sample_indices(100, 20, 42)
    changed = explorer.fixed_sample_indices(100, 20, 43)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert np.all(first[:-1] < first[1:])
    assert len(np.unique(first)) == 20


@pytest.mark.parametrize(
    ("params", "sample_size"),
    [
        (explorer.UmapParams(n_neighbors=1), 10),
        (explorer.UmapParams(n_neighbors=10), 10),
        (explorer.UmapParams(min_dist=-0.1), 10),
        (explorer.UmapParams(min_dist=1.1), 10),
    ],
)
def test_parameter_validation_rejects_invalid_values(
    params: explorer.UmapParams,
    sample_size: int,
) -> None:
    with pytest.raises(explorer.ParameterValidationError):
        params.validate(sample_size)


def test_cache_identity_and_slug_are_stable(tmp_path: Path) -> None:
    run_base, _, _ = _write_discovery_fixture(tmp_path)
    artifact = explorer.discover_experiment(explorer.ExperimentRef(3, 7), run_base)
    params = explorer.UmapParams(n_neighbors=30, min_dist=0.05, metric="cosine")
    versions = {"umap_learn": "0.5.test"}
    identity = explorer._cache_identity(
        artifact,
        params,
        seed=42,
        fit_sample_size=50,
        fit_sample_ids_hash="sample-hash",
        source_fingerprint="source-hash",
        versions=versions,
    )
    key = explorer._hash_json(identity)

    assert key == explorer._hash_json(identity)
    assert explorer._variant_slug(params, 42, key).startswith(
        "nn030__md0p050__metric-cosine__seed0042__"
    )
    changed = dict(identity)
    changed["seed"] = 43
    assert explorer._hash_json(changed) != key


def test_manifest_ignores_incomplete_results(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    complete = session_dir / "run1_expt1" / "variant-complete"
    incomplete = session_dir / "run1_expt1" / "variant-incomplete"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    provenance = {
        "session": "session",
        "label": "control",
        "experiment": {"run": 1, "expt": 1},
        "params": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
            "n_components": 2,
        },
        "seed": 42,
        "fit_sample_size": 50,
        "source": {"inference_dir": "/source", "config_path": "/config"},
        "versions": {"umap_learn": "0.5.test"},
        "diagnostics": {"trustworthiness": 0.9, "knn_retention": 0.5},
        "result_dir": str(complete),
        "cache_key": "abc",
    }
    for directory in (complete, incomplete):
        (directory / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (complete / "_SUCCESS").write_text("complete\n", encoding="utf-8")

    records = explorer.manifest_records(session_dir)
    manifest_path = explorer.rebuild_manifest(session_dir)

    assert len(records) == 1
    assert records[0]["result_dir"] == str(complete)
    assert manifest_path.is_file()
    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.skipif(
    os.environ.get("HYRAX_RUN_INTEGRATION") != "1",
    reason="Set HYRAX_RUN_INTEGRATION=1 to run the local Hyrax fixture workflow.",
)
def test_local_fixture_controls_variant_cache_and_visualizer(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_base = repo_root / "test_dir_100images" / "hyrax_runs"
    refs = [explorer.ExperimentRef(1, expt) for expt in range(1, 5)]
    artifacts = explorer.preflight(refs, run_base)

    controls = explorer.ensure_controls(
        artifacts,
        output_root=tmp_path,
        session="integration",
        evaluation_size=50,
        diagnostic_k=5,
    )
    assert len(controls) == 4

    variant = explorer.run_selected(
        artifacts[refs[0]],
        explorer.UmapParams(n_neighbors=10, min_dist=0.05, metric="cosine"),
        output_root=tmp_path,
        session="integration",
        evaluation_size=50,
        diagnostic_k=5,
    )
    reused = explorer.run_selected(
        artifacts[refs[0]],
        explorer.UmapParams(n_neighbors=10, min_dist=0.05, metric="cosine"),
        output_root=tmp_path,
        session="integration",
        evaluation_size=50,
        diagnostic_k=5,
    )
    assert reused.reused is True
    assert reused.result_dir == variant.result_dir

    re_diagnosed = explorer.run_selected(
        artifacts[refs[0]],
        explorer.UmapParams(n_neighbors=10, min_dist=0.05, metric="cosine"),
        output_root=tmp_path,
        session="integration",
        evaluation_size=40,
        diagnostic_k=4,
    )
    assert re_diagnosed.reused is True
    assert re_diagnosed.result_dir == variant.result_dir
    assert re_diagnosed.diagnostics["evaluation_size"] == 40
    assert re_diagnosed.diagnostics["diagnostic_k"] == 4

    for ref, control in controls.items():
        source_index = np.load(artifacts[ref].inference_dir / "batch_index.npy", allow_pickle=False)
        source_ids = source_index["id"].astype(str)
        output_ids, output_points = explorer.load_embedding_arrays(control.result_dir)
        assert output_points.shape == (len(source_ids), 2)
        assert sorted(output_ids.tolist()) == sorted(source_ids.tolist())
        assert (control.result_dir / "provenance.json").is_file()
        assert (control.result_dir / "_SUCCESS").is_file()
        assert np.isfinite(control.diagnostics["trustworthiness"])
        assert np.isfinite(control.diagnostics["knn_retention"])

    fig, _ = explorer.plot_grid(controls, suptitle="Integration controls")
    assert fig is not None

    pane, visualizer = explorer.open_visualizer(
        variant,
        display_images=False,
        width=350,
        height=350,
    )
    assert pane is not None
    assert visualizer.umap_results.results_dir == variant.result_dir
