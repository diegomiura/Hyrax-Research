from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import static_umap_metrics as metrics
from umap_parameter_explorer import (
    ExperimentArtifacts,
    ExperimentRef,
    UmapParams,
    UmapRun,
    build_merger_time_overlay,
    evaluate_visualizer_selection,
    open_visualizer,
    plot_selection_results,
    selection_results_table,
)


UMAP_IDS = np.array(["1", "2", "3", "4", "5", "6", "7"])
OVERLAYS = [
    {
        "key": "major",
        "threshold": 0.5,
        "label": "Major past 1 Gyr",
    },
    {
        "key": "minor",
        "threshold": 0.5,
        "label": "Minor past 1 Gyr",
    },
]


@pytest.fixture(scope="module", autouse=True)
def _tabular_stack() -> None:
    metrics.init_tabular_stack()


@pytest.fixture
def catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            # ID 1 is repeated consistently, ID 5 has an unknown major label,
            # and UMAP ID 7 is deliberately absent from the catalog.
            "object_id": [1, 1, 2, 3, 4, 5, 6],
            "major": [1.0, 1.0, 0.0, 1.0, 0.0, np.nan, 0.0],
            "minor": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )


def test_selection_scores_multiple_rules_with_known_label_denominators(
    catalog: pd.DataFrame,
) -> None:
    result = metrics.compute_selection_completeness_purity(
        UMAP_IDS,
        np.array([1.0, 2.0, 5.0, 7.0]),
        catalog,
        OVERLAYS,
    )
    summary = result["summary"].set_index("target_label")

    major = summary.loc["Major past 1 Gyr"]
    assert major["n_umap"] == 7
    assert major["n_evaluated"] == 5
    assert major["n_targets"] == 2
    assert major["n_selected"] == 4
    assert major["n_selected_evaluated"] == 2
    assert major["n_selected_unknown"] == 2
    assert major["true_positives"] == 1
    assert major["false_positives"] == 1
    assert major["false_negatives"] == 1
    assert major["true_negatives"] == 2
    assert major["completeness"] == pytest.approx(0.5)
    assert major["purity"] == pytest.approx(0.5)
    assert major["f1"] == pytest.approx(0.5)
    assert major["prevalence"] == pytest.approx(0.4)
    assert major["lift"] == pytest.approx(1.25)
    assert major["selected_label_coverage"] == pytest.approx(0.5)

    minor = summary.loc["Minor past 1 Gyr"]
    assert minor["n_evaluated"] == 6
    assert minor["n_targets"] == 3
    assert minor["n_selected_evaluated"] == 3
    assert minor["n_selected_unknown"] == 1
    assert minor["true_positives"] == 2
    assert minor["false_positives"] == 1
    assert minor["false_negatives"] == 1
    assert minor["true_negatives"] == 2
    assert minor["completeness"] == pytest.approx(2 / 3)
    assert minor["purity"] == pytest.approx(2 / 3)
    assert minor["f1"] == pytest.approx(2 / 3)
    assert minor["lift"] == pytest.approx(4 / 3)
    assert minor["selected_label_coverage"] == pytest.approx(0.75)

    membership = result["selected_membership"]
    assert len(membership) == 8
    major_membership = membership.loc[
        membership["target_label"] == "Major past 1 Gyr"
    ].set_index("_match_id")
    assert bool(major_membership.loc["1", "is_target"])
    assert not bool(major_membership.loc["5", "is_known_label"])
    assert not bool(major_membership.loc["7", "is_known_label"])


def test_empty_and_unknown_only_selections_have_explicit_semantics(
    catalog: pd.DataFrame,
) -> None:
    empty = metrics.compute_selection_completeness_purity(
        UMAP_IDS,
        [],
        catalog,
        [OVERLAYS[0]],
    )["summary"].iloc[0]
    assert empty["completeness"] == 0.0
    assert empty["purity"] == 1.0
    assert empty["f1"] == 0.0
    assert np.isnan(empty["lift"])
    assert np.isnan(empty["selected_label_coverage"])

    unknown_only = metrics.compute_selection_completeness_purity(
        UMAP_IDS,
        [5, 7],
        catalog,
        [OVERLAYS[0]],
    )["summary"].iloc[0]
    assert unknown_only["n_selected"] == 2
    assert unknown_only["n_selected_evaluated"] == 0
    assert unknown_only["n_selected_unknown"] == 2
    assert unknown_only["completeness"] == 0.0
    assert np.isnan(unknown_only["purity"])
    assert np.isnan(unknown_only["f1"])
    assert np.isnan(unknown_only["lift"])


def test_selection_scoring_rejects_invalid_object_membership(
    catalog: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="absent from the UMAP"):
        metrics.compute_selection_completeness_purity(
            UMAP_IDS,
            ["missing"],
            catalog,
            [OVERLAYS[0]],
        )

    with pytest.raises(ValueError, match="must be unique"):
        metrics.compute_selection_completeness_purity(
            ["1", "1", "2"],
            ["1"],
            catalog,
            [OVERLAYS[0]],
        )


def test_selection_scoring_rejects_conflicting_catalog_rows() -> None:
    conflicting_catalog = pd.DataFrame(
        {
            "object_id": [1, 1, 2],
            "major": [1.0, 0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="disagree on overlay target membership"):
        metrics.compute_selection_completeness_purity(
            ["1", "2"],
            ["1"],
            conflicting_catalog,
            [OVERLAYS[0]],
        )


def test_selection_scoring_rejects_rule_without_targets(catalog: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="has no evaluated target objects"):
        metrics.compute_selection_completeness_purity(
            UMAP_IDS,
            ["1"],
            catalog,
            [{"key": "major", "threshold": 2.0, "label": "Impossible"}],
        )


def test_selection_scoring_rejects_empty_evaluation_population() -> None:
    unknown_catalog = pd.DataFrame(
        {
            "object_id": [1, 2],
            "major": [np.nan, np.nan],
        }
    )
    with pytest.raises(ValueError, match="has no evaluated UMAP objects"):
        metrics.compute_selection_completeness_purity(
            ["1", "2"],
            ["1"],
            unknown_catalog,
            [OVERLAYS[0]],
        )


def _write_umap_result(result_dir: Path) -> None:
    result_dir.mkdir()
    batch = np.zeros(
        len(UMAP_IDS),
        dtype=[("id", "U8"), ("tensor", np.float64, (2,))],
    )
    batch["id"] = UMAP_IDS
    batch["tensor"] = np.column_stack(
        [np.arange(len(UMAP_IDS), dtype=float), np.zeros(len(UMAP_IDS))]
    )
    np.save(result_dir / "batch_0.npy", batch, allow_pickle=False)


def _test_run(result_dir: Path) -> UmapRun:
    ref = ExperimentRef(4, 2)
    artifact = ExperimentArtifacts(
        ref=ref,
        config_path=result_dir / "config.toml",
        log_path=result_dir / "run.txt",
        inference_dir=result_dir,
        historical_umap_dir=result_dir,
        original_params=UmapParams(),
        configured_fit_sample_size=7,
        batch_size=7,
    )
    return UmapRun(
        artifact=artifact,
        result_dir=result_dir,
        params=UmapParams(n_neighbors=5, min_dist=0.05, metric="cosine"),
        label="candidate A",
        seed=42,
    )


def test_build_merger_time_overlay_aligns_display_and_scoring_rules(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "umap"
    _write_umap_result(result_dir)
    catalog = pd.DataFrame(
        {
            # ID 1 is repeated consistently; ID 8 is a target absent from the UMAP.
            "object_id": [1, 1, 2, 3, 4, 5, 8],
            "Major_TimeSinceMerger": [0.2, 0.2, 1.0, 1.1, -1.0, np.nan, 0.5],
        }
    )

    overlay = build_merger_time_overlay(
        _test_run(result_dir),
        catalog,
        "MAJOR",
        1.0,
    )

    assert overlay.merger_type == "major"
    assert overlay.cutoff_gyr == 1.0
    assert overlay.catalog_key == "Major_TimeSinceMerger"
    assert overlay.label == "Major merger <= 1 Gyr ago"
    assert overlay.n_catalog_targets == 3
    assert overlay.n_umap_targets == 2
    assert overlay.target_rule["min_value"] == 0.0
    assert overlay.target_rule["max_value"] == 1.0
    assert overlay.target_rule["include_max"] is True
    displayed_ids = overlay.visualizer_overlay["catalog"]["object_id"].tolist()
    assert displayed_ids == ["1", "2"]

    score = metrics.compute_selection_completeness_purity(
        UMAP_IDS,
        ["1"],
        catalog,
        [overlay.target_rule],
    )["summary"].iloc[0]
    assert score["n_targets"] == 2
    assert score["true_positives"] == 1
    assert score["completeness"] == pytest.approx(0.5)
    assert score["purity"] == pytest.approx(1.0)


def test_build_merger_time_overlay_validates_controls_and_catalog_conflicts(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "umap"
    _write_umap_result(result_dir)
    run = _test_run(result_dir)
    valid_catalog = pd.DataFrame(
        {"object_id": [1], "Mini_TimeSinceMerger": [0.5]}
    )

    with pytest.raises(ValueError, match="mini.*minor.*major"):
        build_merger_time_overlay(run, valid_catalog, "micro", 1.0)
    with pytest.raises(ValueError, match="finite non-negative"):
        build_merger_time_overlay(run, valid_catalog, "mini", -1.0)
    with pytest.raises(KeyError, match="No time-since-merger column"):
        build_merger_time_overlay(run, valid_catalog, "major", 1.0)

    conflicting_catalog = pd.DataFrame(
        {
            "object_id": [1, 1],
            "Minor_TimeSinceMerger": [0.5, 2.0],
        }
    )
    with pytest.raises(ValueError, match="disagree on merger-time target membership"):
        build_merger_time_overlay(run, conflicting_catalog, "minor", 1.0)


def test_open_visualizer_passes_external_overlays_to_hyrax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_dir = tmp_path / "umap"
    _write_umap_result(result_dir)
    run = _test_run(result_dir)
    received: dict[str, object] = {}

    class FakeHyrax:
        config = {"visualize": {"display_images": False}}

        def visualize(self, **kwargs: object):
            received.update(kwargs)
            return "pane", "visualizer"

    monkeypatch.setattr(
        "umap_parameter_explorer._hyrax_from_config",
        lambda _: FakeHyrax(),
    )
    overlay = {"catalog": pd.DataFrame({"object_id": ["1"]}), "id_column": "object_id"}

    pane, visualizer = open_visualizer(
        run,
        display_images=True,
        overlays=[overlay],
        width=640,
    )

    assert pane == "pane"
    assert visualizer == "visualizer"
    assert received["input_dir"] == result_dir
    assert received["return_verb"] is True
    assert received["overlays"] == [overlay]
    assert received["width"] == 640


class _MockVisualizer:
    object_id_column_name = "object_id"

    def __init__(self, object_ids: list[str] | None = None) -> None:
        self.object_ids = object_ids or ["1", "2", "5", "7"]

    def get_selected_df(self) -> pd.DataFrame:
        x_values = [float(int(object_id) - 1) for object_id in self.object_ids]
        return pd.DataFrame(
            {
                "object_id": self.object_ids,
                "x": x_values,
                "y": np.zeros(len(self.object_ids)),
            }
        )


class _MetadataFreeVisualizer:
    object_id_column_name = "object_id"
    data_fields: list[str] = []
    points_id = np.array(["1", "2", "5", "7"])
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
        ]
    )

    def get_selected_df(self) -> pd.DataFrame:
        raise AssertionError("metadata-free selections must use visualizer state")


def test_visualizer_wrapper_preserves_run_selection_and_object_metadata(
    tmp_path: Path,
    catalog: pd.DataFrame,
) -> None:
    result_dir = tmp_path / "umap"
    _write_umap_result(result_dir)
    evaluation = evaluate_visualizer_selection(
        _test_run(result_dir),
        _MockVisualizer(),
        catalog,
        OVERLAYS,
        selection_name="upper-left lasso",
    )

    summary = selection_results_table([evaluation])
    assert len(summary) == 2
    assert set(summary["selection_name"]) == {"upper-left lasso"}
    assert set(summary["run"]) == {4}
    assert set(summary["expt"]) == {2}
    assert set(summary["umap_label"]) == {"candidate A"}
    assert set(summary["n_neighbors"]) == {5}
    assert set(summary["metric"]) == {"cosine"}

    selected_objects = evaluation.selected_objects
    assert len(selected_objects) == 8
    assert {"x", "y", "target_label", "is_known_label", "is_target"} <= set(
        selected_objects.columns
    )
    assert set(selected_objects["selection_name"]) == {"upper-left lasso"}


def test_visualizer_wrapper_handles_hyrax_with_no_metadata_fields(
    tmp_path: Path,
    catalog: pd.DataFrame,
) -> None:
    result_dir = tmp_path / "umap"
    _write_umap_result(result_dir)
    evaluation = evaluate_visualizer_selection(
        _test_run(result_dir),
        _MetadataFreeVisualizer(),
        catalog,
        OVERLAYS,
        selection_name="metadata-free lasso",
    )

    assert len(evaluation.summary) == 2
    assert set(evaluation.summary["n_selected"]) == {4}
    assert len(evaluation.selected_objects) == 8
    assert set(evaluation.selected_objects["object_id"]) == {"1", "2", "5", "7"}


def test_plot_selection_results_facets_targets_at_metric_coordinates(
    tmp_path: Path,
    catalog: pd.DataFrame,
) -> None:
    result_dir = tmp_path / "umap"
    _write_umap_result(result_dir)
    evaluation = evaluate_visualizer_selection(
        _test_run(result_dir),
        _MockVisualizer(),
        catalog,
        OVERLAYS,
        selection_name="candidate region",
    )
    second_evaluation = evaluate_visualizer_selection(
        _test_run(result_dir),
        _MockVisualizer(["3", "4", "6"]),
        catalog,
        OVERLAYS,
        selection_name="second region",
    )

    fig, axes = plot_selection_results([evaluation, second_evaluation])
    assert axes.shape == (1, 2)
    expected = selection_results_table([evaluation, second_evaluation])
    for ax in axes.flat:
        target_label = ax.get_title()
        expected_points = expected.loc[
            expected["target_label"] == target_label,
            ["completeness", "purity"],
        ].to_numpy()
        actual_points = np.vstack(
            [collection.get_offsets()[0] for collection in ax.collections]
        )
        np.testing.assert_allclose(actual_points, expected_points)
        assert ax.get_xlim() == pytest.approx((0.0, 1.0))
        assert ax.get_ylim() == pytest.approx((0.0, 1.0))
    plt.close(fig)


def test_selection_results_table_empty_input() -> None:
    assert selection_results_table([]).empty


def _holdout_inputs(n_targets: int = 8, n_non_targets: int = 8):
    n_objects = n_targets + n_non_targets
    object_ids = np.arange(n_objects)
    umap_data = {
        "rubin_ids": object_ids.astype(str),
        "x": np.concatenate(
            [
                np.linspace(0.0, 0.7, n_targets),
                np.linspace(10.0, 10.7, n_non_targets),
            ]
        ),
        "y": np.zeros(n_objects),
    }
    holdout_catalog = pd.DataFrame(
        {
            "object_id": object_ids,
            "merger": np.concatenate(
                [np.ones(n_targets), np.zeros(n_non_targets)]
            ),
        }
    )
    overlay = {"key": "merger", "threshold": 0.5, "label": "Merger"}
    return umap_data, holdout_catalog, overlay


def test_holdout_analysis_uses_only_stratified_test_halves() -> None:
    umap_data, holdout_catalog, overlay = _holdout_inputs()
    result = metrics.compute_overlay_holdout_completeness_purity(
        umap_data,
        holdout_catalog,
        overlay,
        n_neighbors=3,
        test_fraction=0.5,
        n_splits=5,
        seed=17,
    )

    summary = result["summary"]
    assert summary["evaluation_scheme"] == "stratified_holdout_target_anchor"
    assert summary["target_neighbor_rank"] == 3
    assert summary["n_splits"] == 5
    assert summary["n_known_total"] == 16
    assert summary["n_targets_total"] == 8
    assert summary["n_train"] == 8
    assert summary["n_train_targets"] == 4
    assert summary["n_train_non_targets"] == 4
    assert summary["n_evaluated"] == summary["n_test"] == 8
    assert summary["n_targets"] == summary["n_test_targets"] == 4
    assert summary["n_test_non_targets"] == 4
    assert summary["prevalence"] == pytest.approx(0.5)
    assert summary["average_precision"] == pytest.approx(1.0)
    assert summary["best_f1"] == pytest.approx(1.0)

    target_mask = result["target_mask"]
    for split_index in range(5):
        test_mask = result["split_test_masks"][split_index]
        train_mask = result["split_train_masks"][split_index]
        anchor_mask = result["split_anchor_masks"][split_index]
        distances = result["split_distances"][split_index]

        assert not np.any(test_mask & train_mask)
        np.testing.assert_array_equal(
            test_mask | train_mask,
            result["evaluation_mask"],
        )
        assert not np.any(test_mask & anchor_mask)
        assert np.all(train_mask[anchor_mask])
        assert np.all(target_mask[anchor_mask])
        assert int(anchor_mask.sum()) == 4
        assert int(np.sum(test_mask & target_mask)) == 4
        assert int(np.sum(test_mask & ~target_mask)) == 4
        assert np.all(np.isfinite(distances[test_mask]))
        assert np.all(np.isnan(distances[~test_mask]))

    split_summaries = result["split_summaries"]
    assert set(split_summaries["n_evaluated"]) == {8}
    assert set(split_summaries["n_targets"]) == {4}
    assert set(split_summaries["average_precision"]) == {1.0}
    assert np.all(result["curve"]["purity"] == 1.0)


def test_holdout_analysis_is_deterministic_for_a_fixed_seed() -> None:
    umap_data, holdout_catalog, overlay = _holdout_inputs()
    kwargs = {
        "n_neighbors": 3,
        "test_fraction": 0.5,
        "n_splits": 4,
        "seed": 91,
    }
    first = metrics.compute_overlay_holdout_completeness_purity(
        umap_data,
        holdout_catalog,
        overlay,
        **kwargs,
    )
    second = metrics.compute_overlay_holdout_completeness_purity(
        umap_data,
        holdout_catalog,
        overlay,
        **kwargs,
    )

    np.testing.assert_array_equal(
        first["split_test_masks"],
        second["split_test_masks"],
    )
    np.testing.assert_array_equal(
        first["split_anchor_masks"],
        second["split_anchor_masks"],
    )
    np.testing.assert_array_equal(
        first["split_train_masks"],
        second["split_train_masks"],
    )
    np.testing.assert_allclose(
        first["split_distances"],
        second["split_distances"],
        equal_nan=True,
    )
    pd.testing.assert_frame_equal(first["split_curves"], second["split_curves"])


def test_holdout_analysis_requires_three_training_anchors() -> None:
    umap_data, holdout_catalog, overlay = _holdout_inputs(
        n_targets=5,
        n_non_targets=5,
    )
    with pytest.raises(ValueError, match="training split needs at least n_neighbors"):
        metrics.compute_overlay_holdout_completeness_purity(
            umap_data,
            holdout_catalog,
            overlay,
            n_neighbors=3,
            test_fraction=0.5,
            n_splits=2,
        )


def test_neighbor_rank_sweep_returns_paired_comparisons_and_fixed_points() -> None:
    umap_data, holdout_catalog, overlay = _holdout_inputs()
    result = metrics.compute_overlay_neighbor_rank_sweep(
        umap_data,
        holdout_catalog,
        overlay,
        neighbor_ranks=[1, 3],
        test_fraction=0.5,
        n_splits=4,
        seed=23,
    )

    assert result["neighbor_ranks"] == [1, 3]
    assert result["summary"]["target_neighbor_rank"].tolist() == [1, 3]
    assert len(result["split_summaries"]) == 8
    assert set(result["split_summaries"]["split_index"]) == {0, 1, 2, 3}

    paired = metrics.paired_neighbor_rank_differences(result["split_summaries"])
    assert paired["comparison"].tolist() == ["3 minus 1"]
    assert paired.iloc[0]["n_paired_splits"] == 4
    assert paired.iloc[0]["median_change"] == pytest.approx(0.0)

    fixed = metrics.purity_at_fixed_completeness(
        result["curves"],
        completeness_values=[0.1, 0.2],
    )
    assert len(fixed) == 4
    assert set(fixed["target_neighbor_rank"]) == {1, 3}
    assert set(fixed["requested_completeness"]) == {0.1, 0.2}
    assert np.all(fixed["purity"] == 1.0)


def test_neighbor_rank_sweep_rejects_duplicate_ranks() -> None:
    umap_data, holdout_catalog, overlay = _holdout_inputs()
    with pytest.raises(ValueError, match="duplicate value 3"):
        metrics.compute_overlay_neighbor_rank_sweep(
            umap_data,
            holdout_catalog,
            overlay,
            neighbor_ranks=[1, 3, 3],
            n_splits=2,
        )


def _nested_inputs():
    n_targets = 16
    n_non_targets = 16
    n_objects = n_targets + n_non_targets
    object_ids = np.arange(n_objects)
    return (
        {
            "rubin_ids": object_ids.astype(str),
            "x": np.concatenate([np.zeros(n_targets), np.full(n_non_targets, 10.0)]),
            "y": np.zeros(n_objects),
        },
        pd.DataFrame(
            {
                "object_id": object_ids,
                "merger": np.concatenate(
                    [np.ones(n_targets), np.zeros(n_non_targets)]
                ),
            }
        ),
        {"key": "merger", "threshold": 0.5, "label": "Merger"},
    )


def test_nested_analysis_selects_on_validation_and_scores_only_test() -> None:
    umap_data, nested_catalog, overlay = _nested_inputs()
    result = metrics.compute_overlay_nested_completeness_purity(
        umap_data,
        nested_catalog,
        overlay,
        neighbor_ranks=[1, 3, 5],
        train_fraction=0.5,
        validation_fraction=0.25,
        n_splits=5,
        seed=31,
    )

    summary = result["summary"]
    assert summary["evaluation_scheme"] == "nested_stratified_train_validation_test"
    assert summary["selected_neighbor_rank_mode"] == 1
    assert summary["n_train"] == 16
    assert summary["n_train_targets"] == 8
    assert summary["n_validation"] == 8
    assert summary["n_validation_targets"] == 4
    assert summary["n_test"] == 8
    assert summary["n_test_targets"] == 4
    assert summary["test_average_precision"] == pytest.approx(1.0)
    assert summary["test_completeness"] == pytest.approx(1.0)
    assert summary["test_purity"] == pytest.approx(1.0)

    evaluation_mask = result["evaluation_mask"]
    target_mask = result["target_mask"]
    for split_index in range(5):
        train = result["split_train_masks"][split_index]
        validation = result["split_validation_masks"][split_index]
        test = result["split_test_masks"][split_index]
        anchors = result["split_anchor_masks"][split_index]
        assert not np.any(train & validation)
        assert not np.any(train & test)
        assert not np.any(validation & test)
        np.testing.assert_array_equal(train | validation | test, evaluation_mask)
        assert np.all(train[anchors])
        assert np.all(target_mask[anchors])


def test_nested_permutation_control_and_comparison() -> None:
    umap_data, nested_catalog, overlay = _nested_inputs()
    observed = metrics.compute_overlay_nested_completeness_purity(
        umap_data,
        nested_catalog,
        overlay,
        neighbor_ranks=[1, 3],
        n_splits=8,
        seed=47,
    )
    permuted = metrics.compute_overlay_nested_completeness_purity(
        umap_data,
        nested_catalog,
        overlay,
        neighbor_ranks=[1, 3],
        n_splits=8,
        seed=47,
        permute_labels=True,
    )
    comparison = metrics.compare_nested_permutation_baseline(
        observed["split_results"],
        permuted["split_results"],
        metrics=["test_average_precision"],
    )

    assert bool(permuted["summary"]["permuted_labels"])
    assert comparison.iloc[0]["metric"] == "test_average_precision"
    assert comparison.iloc[0]["observed_median"] == pytest.approx(1.0)
    assert 0.0 < comparison.iloc[0]["null_exceedance_fraction"] <= 1.0


def test_analysis_bundle_round_trip(tmp_path: Path) -> None:
    tables = {
        "summary": pd.DataFrame({"rank": [1, 3], "ap": [0.1, 0.2]}),
        "curves": pd.DataFrame({"completeness": [0.0, 1.0]}),
    }
    config = {"neighbor_ranks": [1, 3], "seed": 42}
    paths = metrics.save_analysis_bundle(
        tmp_path / "bundle",
        tables,
        config,
        metadata={"label": "test"},
    )
    assert paths["manifest"].exists()

    loaded = metrics.load_analysis_bundle(
        tmp_path / "bundle",
        expected_config=config,
    )
    assert loaded["metadata"] == {"label": "test"}
    pd.testing.assert_frame_equal(loaded["tables"]["summary"], tables["summary"])
    pd.testing.assert_frame_equal(loaded["tables"]["curves"], tables["curves"])

    with pytest.raises(ValueError, match="does not match"):
        metrics.load_analysis_bundle(
            tmp_path / "bundle",
            expected_config={"neighbor_ranks": [7]},
        )
