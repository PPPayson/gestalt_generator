from pipeline import pairs as P

METAS = [{"stem": "s1"}, {"stem": "s2"}]


def test_catalog_contrast_order(cfg, use_fixture_catalog):
    c = cfg(pairs={"source": "catalog", "select": "contrast", "n_per_silhouette": 3})
    prs = P.resolve(c, METAS)
    got = [it["contrast"] for it in prs["s1"]]
    assert got == sorted(got, reverse=True)        # highest contrast first
    assert len(prs["s1"]) == 3 and set(prs) == {"s1", "s2"}


def test_random_select_seeded(cfg, use_fixture_catalog):
    base = {"source": "catalog", "select": "random", "n_per_silhouette": 3}
    a = P.resolve(cfg(pairs={**base, "seed": 1}), METAS)["s1"]
    b = P.resolve(cfg(pairs={**base, "seed": 1}), METAS)["s1"]
    assert [x["object"] for x in a] == [x["object"] for x in b]     # same seed -> same


def test_random_select_seed_varies(cfg, use_fixture_catalog):
    base = {"source": "catalog", "select": "random", "n_per_silhouette": 3}
    picks = {tuple((x["object"], x["background"]) for x in
                   P.resolve(cfg(pairs={**base, "seed": s}), METAS)["s1"]) for s in range(6)}
    assert len(picks) >= 2


def test_random_inherits_global_seed(cfg, use_fixture_catalog):
    base = {"source": "catalog", "select": "random", "n_per_silhouette": 3, "seed": None}
    a = P.resolve(cfg(seed=5, pairs=base), METAS)["s1"]
    b = P.resolve(cfg(seed=5, pairs=base), METAS)["s1"]
    assert [x["object"] for x in a] == [x["object"] for x in b]


def test_filter_min_contrast(cfg, use_fixture_catalog):
    c = cfg(pairs={"source": "catalog", "n_per_silhouette": 10,
                   "filter": {"min_contrast": 0.55}})
    prs = P.resolve(c, METAS)
    assert all(it["contrast"] >= 0.55 for it in prs["s1"])


def test_explicit_source(cfg):
    c = cfg(pairs={"source": "explicit", "n_per_silhouette": 5,
                   "explicit": [{"object": "white disc", "background": "black bg"}]})
    prs = P.resolve(c, METAS)
    assert prs["s1"][0]["object"] == "white disc"
    assert prs["s1"][0]["natural"] is False        # default filled in


def test_needed(cfg, use_fixture_catalog):
    prs = P.resolve(cfg(pairs={"source": "catalog", "n_per_silhouette": 2}), METAS)
    objs, bgs = P.needed(prs)
    assert objs and bgs and isinstance(objs, set)
