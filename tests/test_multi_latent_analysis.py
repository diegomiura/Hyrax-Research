"""Scientific invariants for the multi-experiment evaluation."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import average_precision_score

import multi_latent_analysis as mla


@pytest.fixture
def fixture_data():
    rng = np.random.default_rng(100)
    # Above the float64 exact integer range, and close enough to expose collisions.
    ids = np.array([str(2**60 + i) for i in range(80)])
    targets = np.arange(80) < 24
    members = {}
    for expt, dim in [(9, 6), (13, 4), (14, 8)]:
        x = rng.normal(size=(80, dim))
        x[:, 0] += targets * 2
        u = x[:, :2] + rng.normal(size=(80, 2)) * .2
        order = rng.permutation(80)
        members[f"run11_expt{expt}"] = {
            "latent": (ids[order], x[order]),
            "umap": (ids[::-1], u[::-1]),
        }
    catalog = pd.DataFrame({"object_id": np.repeat(ids, 2),
                            "Major_TimeSinceMerger": np.repeat(np.where(targets, .3, -1), 2)})
    config = mla.AnalysisConfig(n_splits=2, neighbor_ranks=(1, 3), cka_sample_size=50)
    return members, catalog, config


def test_exact_join_and_row_order_invariance(fixture_data):
    members, catalog, config = fixture_data
    first = mla.build_cohort(members, catalog, config)
    shuffled = {key: {kind: (ids[::-1], x[::-1]) for kind, (ids, x) in member.items()}
                for key, member in members.items()}
    second = mla.build_cohort(shuffled, catalog.sample(frac=1, random_state=17), config)
    np.testing.assert_array_equal(first.ids, second.ids)
    assert len(set(first.ids)) == 80
    assert str(2**60 + 1) in first.ids
    for name in members:
        np.testing.assert_array_equal(first.latents[name], second.latents[name])
        np.testing.assert_array_equal(first.umaps[name], second.umaps[name])
    np.testing.assert_array_equal(first.targets, second.targets)
    assert first.targets.sum() == 24
    assert first.coverage["catalog_rows"] == 160
    assert first.coverage["catalog_unique_ids"] == 80


def test_integer_ids_never_round_trip_through_float():
    ids = np.array([2**60, 2**60 + 1, 2**60 + 2], dtype=np.int64)
    assert mla.exact_ids(ids).tolist() == [str(int(i)) for i in ids]
    with pytest.raises(ValueError, match="Unsafe floating"):
        mla.exact_ids(ids.astype(float))
    with pytest.raises(ValueError, match="Missing"):
        mla.exact_ids(["1", ""])


def test_catalog_duplicate_semantics_and_unknowns(fixture_data):
    members, catalog, config = fixture_data
    conflicting = catalog.copy()
    conflicting.loc[0, config.target_column] = -1
    with pytest.raises(ValueError, match="disagree"):
        mla.build_cohort(members, conflicting, config)
    unknown = catalog.copy()
    unknown.loc[:1, config.target_column] = np.nan
    c = mla.build_cohort(members, unknown, config)
    assert c.known.sum() == 79
    assert c.targets.sum() == 23
    unknown.loc[0, config.target_column] = .2
    with pytest.raises(ValueError, match="whether the label is known"):
        mla.build_cohort(members, unknown, config)
    with pytest.raises(ValueError, match="Missing IDs in catalog"):
        mla.build_cohort(members, catalog.iloc[2:], config)


@pytest.mark.parametrize("failure", ["duplicate", "missing", "nonfinite", "constant"])
def test_invalid_embeddings_fail(fixture_data, failure):
    members, catalog, config = fixture_data
    ids, x = members["run11_expt13"]["latent"]
    ids, x = ids.copy(), x.copy()
    if failure == "duplicate":
        ids[0] = ids[1]
    elif failure == "missing":
        ids[0] = "different"
    elif failure == "nonfinite":
        x[0, 0] = np.nan
    else:
        x[:] = 1
    members["run11_expt13"]["latent"] = ids, x
    with pytest.raises(ValueError):
        mla.build_cohort(members, catalog, config)


def test_fused_distance_identity_rotation_and_scale(fixture_data):
    members, catalog, config = fixture_data
    c = mla.build_cohort(members, catalog, config)
    train = np.arange(len(c.ids)) % 2 == 0
    stats = mla.fit_normalization(c.latents, train)
    combined = mla.combine_latents(c.latents, stats)
    expected = sum(pdist(x, "sqeuclidean") / stats[name][1]**2
                   for name, x in c.latents.items()) / len(c.latents)
    np.testing.assert_allclose(pdist(combined, "sqeuclidean"), expected, atol=1e-12)
    rng = np.random.default_rng(3)
    transformed = {}
    for index, (name, x) in enumerate(c.latents.items()):
        rotation, _ = np.linalg.qr(rng.normal(size=(x.shape[1], x.shape[1])))
        transformed[name] = (x @ rotation) * (index + 2) + 10
    rotated = mla.combine_latents(transformed, mla.fit_normalization(transformed, train))
    np.testing.assert_allclose(pdist(rotated, "sqeuclidean"), expected, atol=1e-12)


def test_scoring_matches_fused_embedding_geometry(fixture_data):
    members, catalog, config = fixture_data
    c = mla.build_cohort(members, catalog, config)
    p, y = mla.make_partition(c.targets, c.known, config, 0)
    scores, stats = mla.score_candidates(c, p, y & (p == 0), config)
    full = squareform(pdist(mla.combine_latents(c.latents, stats)))
    query = (p == 1) | (p == 2)
    anchor = (p == 0) & y
    expected = np.sort(full[query][:, anchor], axis=1)[:, np.array(config.neighbor_ranks)-1]
    np.testing.assert_allclose(scores["latent:ensemble"][query], expected)


def test_test_labels_and_test_vectors_cannot_change_fitted_choices(fixture_data):
    members, catalog, config = fixture_data
    c = mla.build_cohort(members, catalog, config)
    p, y = mla.make_partition(c.targets, c.known, config, 0)
    train, val, test = [p == i for i in range(3)]
    scores, stats = mla.score_candidates(c, p, y & train, config)
    changed_labels = y.copy()
    changed_labels[test] = ~changed_labels[test]
    changed = replace(c, targets=changed_labels,
                      latents={name: x + test[:, None] * 1000 for name, x in c.latents.items()},
                      umaps={name: x + test[:, None] * 1000 for name, x in c.umaps.items()})
    other_scores, other_stats = mla.score_candidates(changed, p, changed_labels & train, config)
    for name in stats:
        np.testing.assert_array_equal(stats[name][0], other_stats[name][0])
        assert stats[name][1] == other_stats[name][1]
    for method in scores:
        np.testing.assert_allclose(scores[method][val], other_scores[method][val])
        before = mla.select_candidate(scores[method], y & val, val, config.neighbor_ranks)
        after = mla.select_candidate(other_scores[method], changed_labels & val, val, config.neighbor_ranks)
        assert before == after


def test_cdf_ties_and_frozen_test_mapping():
    np.testing.assert_allclose(mla.percentile_transform([1, 2, 2, 4], [0, 1, 2, 3, 5]),
                               [0, .25, .75, .75, 1])


def test_paired_evaluation_reproducible_and_denominators(fixture_data):
    members, catalog, config = fixture_data
    c = mla.build_cohort(members, catalog, config)
    tables, audit = mla.evaluate(c, config, progress=False)
    again, repeated = mla.evaluate(c, config, progress=False)
    pd.testing.assert_frame_equal(tables["split_metrics"], again["split_metrics"])
    np.testing.assert_array_equal(audit["partitions"], repeated["partitions"])
    for mode, mode_name in enumerate(audit["modes"]):
        for split in range(config.n_splits):
            partition = audit["partitions"][mode, split]
            y = audit["target_labels"][mode, split]
            assert y.sum() == c.targets.sum()
            assert set(partition) == {0, 1, 2}
            frame = tables["split_metrics"].query("mode == @mode_name and split == @split")
            assert frame["n_evaluated"].eq(np.sum(partition == 2)).all()
            assert frame["n_targets"].eq(np.sum(y & (partition == 2))).all()
            assert frame["n_scored"].eq(frame["n_evaluated"]).all()
            for mi, method in enumerate(audit["methods"]):
                scores = audit["test_scores"][mode, split, mi]
                np.testing.assert_array_equal(np.isfinite(scores), partition == 2)
            for kind in ["latent", "umap"]:
                winner = frame[frame.method == f"{kind}:best_single"].iloc[0]
                single = frame[frame.method == winner.selected_member].iloc[0]
                assert winner.validation_ap == single.validation_ap
                assert winner.average_precision == single.average_precision
                eligible = frame[frame.method.str.startswith(kind+":run")]
                assert winner.validation_ap == eligible.validation_ap.max()


def test_curve_ap_agrees_with_sklearn_for_ties():
    mla.init_stack()
    y = np.array([True, False, False, True, True, False])
    distances = np.array([1., 1., 2., 3., 3., 4.])
    _, metrics = mla.sm.distance_completeness_purity_curve(distances, y)
    assert metrics["average_precision"] == pytest.approx(average_precision_score(y, -distances))


def test_cka_invariances():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(30, 5))
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    assert mla.linear_cka(x, 3*x @ rotation + 8) == pytest.approx(1)


def test_load_batches_excludes_metadata_and_refuses_missing_ids(tmp_path):
    batch = np.zeros(2, dtype=[("id", "U30"), ("tensor", "f4", (3,))])
    batch["id"] = [str(2**60), str(2**60+1)]
    np.save(tmp_path / "batch_0.npy", batch)
    np.save(tmp_path / "batch_0_metadata.npy", np.zeros(5))
    ids, x = mla.load_batches(tmp_path)
    assert len(ids) == 2 and x.shape == (2, 3)
    np.save(tmp_path / "batch_1.npy", np.zeros(2, dtype=[("tensor", "f4", (3,))]))
    with pytest.raises(ValueError, match="id and tensor"):
        mla.load_batches(tmp_path)
