"""`type: generate`: knob space, exhaustive sweep (rng fresh), aliases and config validation."""
import json
import pytest
from pipeline import silhouettes as S, config
from pipeline.utils import SIL, SIL_CTRL, SIL_NEG

SMALL = {"polygon_randomness": [0], "n_vertices": [5], "n_inducers": [4], "corner_sharpness": [0.1], "side_bulge": [0.1]}


def test_space_names_and_defaults():
    g = S._space({"n_inducers": [3, 4], "corner_sharpness": {"min": 0.1, "max": 1}})
    assert g["n_points"] == [3, 4] and g["edgy"] == {"min": 0.1, "max": 1.0}
    assert g["randomness"] == S.GRID_DEFAULT["randomness"]
    with pytest.raises(ValueError):
        S._space({"bogus": [1]})


def test_sweep_combos_radius_rule():
    g = S._space(None)
    assert len(S._sweep_combos(g, "derived", 125)) == 4 * 6 * 4 * 3 * 3
    listed = S._sweep_combos(g, [100, 75, 60, 50], 100)
    per_count = {k: sum(1 for c in listed if c[2] == k) for k in (3, 4, 5, 6)}
    assert [per_count[k] // (4 * 6 * 9) for k in (3, 4, 5, 6)] == [4, 3, 2, 1]


def test_sweep_fresh_writes_three_stores():
    stems = S.use_generate(space=SMALL, n_figures="all", figures_per_combination=2, seed=7, rng="fresh", jobs=1)
    assert len(stems) == 2 and stems[0] != stems[1]
    for st in stems:
        assert (SIL / f"{st}.png").exists() and (SIL_CTRL / f"{st}.png").exists() and (SIL_NEG / f"{st}.png").exists()
        meta = json.loads((SIL / f"{st}.json").read_text())
        assert meta["n_inducers"] == 4 and meta["polygon_vertices"] == 5 and meta["rng"] == "fresh"
        assert meta["inducer_radius"] == 3 * 125 / 4


def test_sweep_is_reproducible():
    a = S.use_generate(space=SMALL, n_figures="all", seed=3, rng="shared", jobs=1)
    pa = (SIL / f"{a[0]}.png").read_bytes()
    b = S.use_generate(space=SMALL, n_figures="all", seed=3, rng="shared", jobs=1)
    assert a == b and (SIL / f"{b[0]}.png").read_bytes() == pa


def test_old_names_still_work():
    stems = S.use_generate(figures=[[0, 6, 3, 0.1, 0.1, 2]], closeness=0.5, seed=1)
    assert stems and stems[0].startswith("img_")


def test_config_rejects_unknown_condition(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\ndress: {conditions: [illusory, bogus]}\n")
    with pytest.raises(ValueError):
        config.load(p)


def test_space_rejects_bad_values():
    for bad in ({"n_inducers": []}, {"n_inducers": [1]}, {"n_vertices": [2]}, {"side_bulge": [1.5]},
                {"polygon_randomness": {"min": 1, "max": 0}}, {"n_inducers": ["4"]}):
        with pytest.raises(ValueError):
            S._space(bad)


def test_inducer_radius_validation():
    with pytest.raises(ValueError):                       # every listed radius above the derived one
        S._sweep_combos(S._space({"n_inducers": [6]}), [200], 100)
    with pytest.raises(ValueError):
        S.use_generate(space=SMALL, n_figures="all", inducer_radius="Derived", jobs=1)
    with pytest.raises(ValueError):                       # index outside the sweep
        S.use_generate(space=SMALL, n_figures="all", indices=[5], jobs=1)


def test_quota_counts():
    assert S._quota_counts(10, "uniform", (3, 4)) == {3: 5, 4: 5}
    assert S._quota_counts(10, {4: 0.75, 5: 0.25}) == {4: 8, 5: 2}
    assert S._quota_counts(3, {3: 1, 4: 1, 5: 1}) == {3: 1, 4: 1, 5: 1}   # ones are counts, not shares
    with pytest.raises(ValueError):
        S._quota_counts(10, {3: 2, 4: 3})


def test_sweep_same_across_worker_counts():
    kw = dict(space=SMALL, n_figures="all", figures_per_combination=4, seed=11, rng="shared")
    one = S.use_generate(jobs=1, **kw)
    b1 = [(SIL / f"{s}.png").read_bytes() + (SIL_NEG / f"{s}.png").read_bytes() for s in one]
    two = S.use_generate(jobs=2, **kw)
    b2 = [(SIL / f"{s}.png").read_bytes() + (SIL_NEG / f"{s}.png").read_bytes() for s in two]
    assert one == two and b1 == b2
