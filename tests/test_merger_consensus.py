"""Protect held-out selection, exact IDs, and equal weighting across runs."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score

import merger_consensus as mc


@pytest.fixture
def inputs():
    rng = np.random.default_rng(37)
    ids = np.array([str(2**60 + i) for i in range(96)])
    targets = np.arange(96) < 32
    embeddings, members = {}, {}
    for run, expt in [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1)]:
        key = f"run{run}_expt{expt}"
        x = rng.normal(size=(96, 5))
        x[:, 0] += 2 * targets
        order = rng.permutation(96)
        embeddings[key] = {"latent": (ids[order], x[order]),
                           "umap": (ids[::-1], x[::-1, :2])}
        members[key] = {"run": str(run), "experiment": str(expt)}
    catalog = pd.DataFrame({"object_id": np.repeat(ids, 2),
                            "Major_TimeSinceMerger": np.repeat(np.where(targets, .3, -1.), 2)})
    config = mc.ConsensusConfig(n_splits=3, min_holdouts=1, neighbor_ranks=(1, 3), forest_trees=8)
    return embeddings, members, catalog, config


def test_exact_alignment_row_order_and_large_ids(inputs):
    embeddings, members, catalog, config = inputs
    first = mc.build_cohort(embeddings, members, catalog, config)
    shuffled = {key: {kind: (ids[::-1], x[::-1]) for kind, (ids, x) in value.items()}
                for key, value in embeddings.items()}
    second = mc.build_cohort(shuffled, members, catalog.sample(frac=1, random_state=9), config)
    np.testing.assert_array_equal(first.ids, second.ids)
    assert str(2**60 + 1) in first.ids
    assert first.targets.sum() == 32
    for key in first.spaces:
        np.testing.assert_array_equal(first.spaces[key], second.spaces[key])


def test_intersection_is_explicit_and_audits_drops(inputs):
    embeddings, members, catalog, config = inputs
    ids, x = embeddings["run2_expt1"]["umap"]
    embeddings["run2_expt1"]["umap"] = ids[1:], x[1:]
    with pytest.raises(ValueError, match="ID sets differ"):
        mc.build_cohort(embeddings, members, catalog, config)
    c = mc.build_cohort(embeddings, members, catalog, replace(config, cohort_mode="intersection"))
    assert len(c.ids) == 95
    assert sum(row["dropped_objects"] for row in c.coverage["spaces"]) == 9


def test_unknowns_and_conflicting_catalog_rows(inputs):
    embeddings, members, catalog, config = inputs
    catalog.loc[:1, config.target_column] = np.nan
    c = mc.build_cohort(embeddings, members, catalog, config)
    assert c.known.sum() == 95
    p, y = mc.make_partition(c, config, 0)
    assert p[~c.known].tolist() == [-1]
    catalog.loc[0, config.target_column] = .2
    with pytest.raises(ValueError, match="whether the label is known"):
        mc.build_cohort(embeddings, members, catalog, config)
    catalog.loc[1, config.target_column] = -1
    with pytest.raises(ValueError, match="disagree"):
        mc.build_cohort(embeddings, members, catalog, config)


@pytest.mark.parametrize("failure", ["duplicate", "nonfinite", "constant", "unsafe_id"])
def test_invalid_embeddings_fail(inputs, failure):
    embeddings, members, catalog, config = inputs
    ids, x = embeddings["run1_expt1"]["latent"]
    if failure == "duplicate":
        ids[0] = ids[1]
    elif failure == "nonfinite":
        x[0, 0] = np.nan
    elif failure == "constant":
        x[:] = 0
    else:
        embeddings["run1_expt1"]["latent"] = ids.astype(float), x
    with pytest.raises(ValueError):
        mc.build_cohort(embeddings, members, catalog, config)


def test_partitions_are_paired_half_heldout_and_deterministic(inputs):
    c = mc.build_cohort(*inputs)
    config = inputs[-1]
    p, y = mc.make_partition(c, config, 0)
    other, other_y = mc.make_partition(c, config, 0)
    np.testing.assert_array_equal(p, other)
    np.testing.assert_array_equal(y, other_y)
    assert [(p == code).sum() for code in range(3)] == [24, 24, 48]
    assert [(y & (p == code)).sum() for code in range(3)] == [8, 8, 16]
    shuffled_p, shuffled_y = mc.make_partition(c, config, 0, shuffled=True)
    assert shuffled_y.sum() == y.sum()
    assert not np.array_equal(shuffled_y, y)


def test_threshold_ties_match_f1_optimum_and_ap():
    mc.init_stack()
    score = np.array([.9, .9, .7, .5, .5, .1])
    y = np.array([True, False, True, False, True, False])
    threshold, ap = mc.threshold_from_validation(score, y)
    assert ap == pytest.approx(average_precision_score(y, score))
    expected = max(f1_score(y, score >= value) for value in np.unique(score))
    assert f1_score(y, score >= threshold) == pytest.approx(expected)


def test_top_experiments_select_within_run_and_algo_using_validation_only():
    mc.init_stack()
    rows = [{"run": run, "representation": "latent", "algorithm": algo,
             "key": f"{run}_{i}", "validation_ap": float(i), "test_ap": -float(i)}
            for run, count in [("1", 5), ("2", 2)] for algo in mc.ALGORITHMS for i in range(count)]
    first = mc.rank_experiments(rows, 3)
    assert first[first.selected].groupby(["run", "algorithm"]).size().tolist() == [3, 3, 3, 2, 2, 2]
    for row in rows:
        row["test_ap"] = 1000
    pd.testing.assert_frame_equal(first.drop(columns="test_ap"),
                                  mc.rank_experiments(rows, 3).drop(columns="test_ap"))
    rows[0]["validation_ap"] = rows[1]["validation_ap"]
    assert mc.rank_experiments(rows, 3).query("run == '1' and algorithm == 'logistic'").key.tolist()[-2:] == ["1_0", "1_1"]


def test_equal_weighting_for_unequal_experiment_and_algorithm_groups():
    mc.init_stack()
    config = mc.ConsensusConfig()
    rows, scores = [], {}
    # Run 1: logistic has three positive models, forest and anchor one negative each.
    # Run 2: one positive model per algorithm. Equal run vote = (1/3 + 1)/2 = 2/3.
    for run in ("1", "2"):
        for algorithm in config.algorithms:
            count = 3 if run == "1" and algorithm == "logistic" else 1
            for index in range(count):
                key = f"{run}_{algorithm}_{index}"
                positive = run == "2" or algorithm == "logistic"
                rows.append({"run": run, "representation": "latent", "algorithm": algorithm,
                             "key": key, "threshold": .5})
                scores[key, "latent", algorithm] = (np.array([float(positive)]), np.array([float(positive)]))
    config = replace(config, representations=("latent",))
    _, runs, total = mc.aggregate_split(pd.DataFrame(rows), scores, config)
    assert runs["latent", "1"][0][0] == pytest.approx(1/3)
    assert total["latent"][0][0] == pytest.approx(2/3)
    assert not total["latent"][2][0]  # Not unanimous.
    assert not total["latent"][3][0]  # Run 1 is below the majority threshold.


def test_test_vectors_and_labels_cannot_affect_experiment_or_threshold_selection(inputs, monkeypatch):
    c = mc.build_cohort(*inputs)
    config = replace(inputs[-1], n_splits=1, representations=("latent",), algorithms=("logistic", "anchor"))
    c.spaces = {key: x for key, x in c.spaces.items() if key[1] == "latent"}
    p, y = mc.make_partition(c, config, 0)
    monkeypatch.setattr(mc, "make_partition", lambda *args, **kwargs: (p, y))
    first, first_audit = mc.evaluate(c, config, progress=False)
    changed_y = y.copy()
    changed_y[p == 2] = ~changed_y[p == 2]
    changed = replace(c, targets=changed_y,
                      spaces={key: x + (p == 2)[:, None] * 1000 for key, x in c.spaces.items()})
    monkeypatch.setattr(mc, "make_partition", lambda *args, **kwargs: (p, changed_y))
    second, second_audit = mc.evaluate(changed, config, progress=False)
    pd.testing.assert_frame_equal(first["experiment_selections"], second["experiment_selections"])
    np.testing.assert_array_equal(first_audit["selected_models"], second_audit["selected_models"])
    assert not np.allclose(first_audit["test_scores"], second_audit["test_scores"], equal_nan=True)


def test_consensus_reproducibility_denominators_and_audit_reconstruction(inputs):
    c = mc.build_cohort(*inputs)
    config = inputs[-1]
    tables, audit = mc.evaluate(c, config, progress=False)
    repeated, again = mc.evaluate(c, config, progress=False)
    pd.testing.assert_frame_equal(tables["experiment_selections"], repeated["experiment_selections"])
    np.testing.assert_array_equal(audit["test_votes"], again["test_votes"])
    counts = (audit["partitions"] == 2).sum(axis=0)
    for row in tables["consensus_candidates"].itertuples():
        oi = np.flatnonzero(c.ids == row.object_id)[0]
        assert row.n_holdouts == counts[oi]
        frequencies = []
        for split in np.flatnonzero(audit["partitions"][:, oi] == 2):
            run_values = []
            for run in ("1", "2"):
                algo_values = []
                for algorithm in config.algorithms:
                    mask = ((audit["representations"] == row.representation)
                            & (audit["algorithms"] == algorithm)
                            & np.array([c.members[key]["run"] == run for key in audit["candidate_keys"]])
                            & audit["selected_models"][split])
                    algo_values.append(audit["test_votes"][split, mask, oi].mean())
                run_values.append(np.mean(algo_values))
            frequencies.append(np.mean(run_values))
        assert row.vote_fraction == pytest.approx(np.mean(frequencies))
    mc.finish_tables(tables, c, config)
    assert tables["consensus_candidates"].cross_run_tested.all()
    assert tables["run_agreement"].shape[0] == 2
    assert tables["top_candidates_per_algorithm"].groupby(["representation", "algorithm"]).size().eq(3).all()
    for split, partitions in enumerate(audit["partitions"]):
        assert np.all(audit["test_votes"][split, :, partitions != 2] == -1)
        assert np.all(np.isnan(audit["test_scores"][split, :, partitions != 2]))


def test_single_run_never_claims_cross_run_stability(inputs):
    c = mc.build_cohort(*inputs)
    c.spaces = {key: x for key, x in c.spaces.items() if key[0].startswith("run2")}
    c.members = {key: m for key, m in c.members.items() if key.startswith("run2")}
    config = replace(inputs[-1], n_splits=1)
    tables, _ = mc.evaluate(c, config, progress=False)
    mc.finish_tables(tables, c, config)
    assert not tables["consensus_candidates"].stable_across_runs.any()
    assert not tables["consensus_candidates"].cross_run_tested.any()
    assert tables["run_agreement"].empty


def test_manifest_cli_bundle_control_and_no_overwrite(inputs, tmp_path):
    embeddings, members, catalog, config = inputs
    catalog.to_csv(tmp_path / "catalog.csv", index=False)
    experiments = []
    for key in ("run1_expt1", "run2_expt1"):
        item = {"key": key, **members[key]}
        for representation, field in (("latent", "inference_dir"), ("umap", "umap_dir")):
            ids, x = embeddings[key][representation]
            directory = tmp_path / f"{key}_{representation}"
            directory.mkdir()
            batch = np.zeros(len(ids), dtype=[("id", "U30"), ("tensor", "f8", (x.shape[1],))])
            batch["id"], batch["tensor"] = ids, x
            np.save(directory / "batch_0.npy", batch)
            np.save(directory / "batch_0_metadata.npy", np.zeros(3))
            item[field] = directory.name
        experiments.append(item)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"catalog": "catalog.csv", "experiments": experiments}))
    output = tmp_path / "bundle"
    assert mc.main(["run", "--manifest", str(manifest), "--output-dir", str(output),
                    "--n-splits", "1", "--min-holdouts", "1", "--neighbor-ranks", "1", "3",
                    "--forest-trees", "8", "--no-figures", "--shuffled-control"]) == 0
    record = json.loads((output / "analysis_manifest.json").read_text())
    assert record["status"] == "complete" and record["cross_run_tested"]
    assert record["shuffled_control"]
    assert (output / "shuffled_prediction_audit.npz").exists()
    exact = np.load(output / "prediction_audit.npz", allow_pickle=False)["object_ids"]
    assert str(2**60 + 1) in exact
    cohort = pd.read_csv(output / "cohort.csv", dtype={"object_id": str})
    assert len(cohort) == 96 and cohort.object_id.nunique() == 96
    with pytest.raises(FileExistsError):
        mc.run_analysis(manifest, output, config)


@pytest.mark.parametrize("setting", [{"n_splits": 0}, {"top_experiments": 0}, {"seed": -1},
                                      {"algorithms": ()}, {"representations": ("bad",)},
                                      {"consensus_threshold": 0}, {"cutoff_gyr": float("nan")}])
def test_invalid_configuration(setting):
    with pytest.raises(ValueError):
        replace(mc.ConsensusConfig(), **setting).validate()
