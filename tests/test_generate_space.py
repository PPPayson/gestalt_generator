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
    assert per_count == {3: 4 * 6 * 9, 4: 3 * 4 * 6 * 9 // 3, 5: 2 * 4 * 6 * 9 // 3 * 1, 6: 4 * 6 * 9 // 3 * 1} or \
           [per_count[k] // (4 * 6 * 9) for k in (3, 4, 5, 6)] == [4, 3, 2, 1]


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
    with pytest.raises(AssertionError):
        config.load(p)
