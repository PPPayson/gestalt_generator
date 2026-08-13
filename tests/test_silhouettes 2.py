import json
from pipeline import silhouettes as S
from pipeline.geometry import inducers
from pipeline.utils import SIL


def test_produce_kanizsa_counts():
    S.produce_kanizsa(n_sides=3)
    S.produce_kanizsa(n_sides=4)
    assert len(inducers(SIL / "kanizsa_triangle.png")) == 3
    assert len(inducers(SIL / "kanizsa_square.png")) == 4
    meta = json.loads((SIL / "kanizsa_triangle.json").read_text())
    assert meta["illusory_shape"] == "triangle" and meta["n_inducers"] == 3


def test_parse_gestalt_name():
    p = S.parse_gestalt_name("img_0.4_4_4_0.5_0.1")
    assert p == {"randomness": 0.4, "n_vertices": 4, "n_points": 4, "edgy": 0.5, "rad": 0.1}
    assert S.parse_gestalt_name("kanizsa_triangle") is None


def test_gestalt_select_seeded_reproducible(cfg):
    a = S.use_gestalt(mode="existing", n=2, seed=1)
    b = S.use_gestalt(mode="existing", n=2, seed=1)
    assert a == b                                  # same seed -> same stems


def test_gestalt_select_seed_varies():
    # with only 3 in the pool, seed changes ordering/selection of n=2 at least sometimes
    picks = {tuple(S.use_gestalt(mode="existing", n=2, seed=s)) for s in range(6)}
    assert len(picks) >= 2


def test_gestalt_clamps_when_n_exceeds_pool():
    got = S.use_gestalt(mode="existing", n=999, seed=0)
    assert 1 <= len(got) <= 3                       # only 3 gestalt silhouettes in the world


def test_gestalt_sidecar_has_parsed_meta():
    stem = S.use_gestalt(mode="existing", n=1, seed=0)[0]
    meta = json.loads((SIL / f"{stem}.json").read_text())
    assert meta["illusory_shape"] == "irregular"
    assert meta["primitive"] == "circle"
    assert "polygon_vertices" in meta and "n_inducers" in meta


def test_build_inherits_global_seed(cfg):
    c1 = cfg(seed=3, silhouettes=[{"type": "gestalt", "params": {"mode": "existing", "n": 2}}])
    c2 = cfg(seed=3, silhouettes=[{"type": "gestalt", "params": {"mode": "existing", "n": 2}}])
    s1 = [m["stem"] for m in S.build(c1)]
    s2 = [m["stem"] for m in S.build(c2)]
    assert s1 == s2


def test_dropin_infers_sidecar():
    stem = sorted(p.stem for p in SIL.glob("img_*.png"))[0]
    (SIL / f"{stem}.json").unlink(missing_ok=True)
    got = S.use_dropin(files=[f"{stem}.png"])
    assert got == [stem]
    assert (SIL / f"{stem}.json").exists()
