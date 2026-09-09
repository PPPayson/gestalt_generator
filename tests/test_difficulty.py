"""Difficulty knobs: closeness (support ratio), texture level, background-recolor contrast."""
import json
import numpy as np
import pytest
from PIL import Image
import main
from pipeline import dress_composite as DC, silhouettes as S
from pipeline.utils import SIL, SIL_CTRL, BG, OBJ, OUT


@pytest.fixture(autouse=True)
def _cleanup_generated():
    """Remove silhouettes/assets these tests generate so the shared world stays fixed for others."""
    before = {p.name for d in (SIL, SIL_CTRL, BG, OBJ) for p in d.glob("*")}
    yield
    for d in (SIL, SIL_CTRL, BG, OBJ):
        for p in d.glob("*"):
            if p.name not in before:
                p.unlink()


def _gen(closeness, seed=3, n_points=4):
    return main.generate_one(0, 4, n_points, 0.1, 0.1, closeness=closeness,
                             sil_dir=str(SIL), ctrl_dir=str(SIL_CTRL), seed=seed)


def _support(stem):
    return json.loads((SIL / f"{stem}.json").read_text())["support_ratio"]


def test_closeness_monotone_support_ratio():
    stems = {c: _gen(c) for c in (0.1, 0.5, 1.0)}
    assert all(stems.values()), "disc placement failed at some closeness"
    s = {c: _support(st) for c, st in stems.items()}
    assert s[0.1] < s[1.0]                       # bigger discs trace more of the contour
    assert s[0.1] <= s[0.5] <= s[1.0]


def test_closeness_none_keeps_legacy_name_and_writes_sidecar():
    stem = main.generate_one(0, 4, 4, 0.1, 0.1, closeness=None,
                             sil_dir=str(SIL), ctrl_dir=str(SIL_CTRL), seed=3)
    assert stem == "img_0_4_4_0.1_0.1"           # no suffix -> legacy filename
    assert (SIL / f"{stem}.png").exists() and (SIL_CTRL / f"{stem}.png").exists()
    side = json.loads((SIL / f"{stem}.json").read_text())
    assert 0 < side["support_ratio"] < 1 and "closeness" not in side


def test_closeness_measured_recorded_and_monotone():
    """The realised mean nearest-point closeness (1 - gap/d) is recorded and rises with the knob."""
    ms = []
    for c in (0.1, 0.5, 1.0):
        stem = _gen(c)
        ms.append(json.loads((SIL / f"{stem}.json").read_text())["closeness_measured"])
    assert all(0 < m < 1 for m in ms) and ms[0] < ms[1] < ms[2]


def test_closeness_always_feasible():
    """Combos that were structurally impossible at max closeness under the fixed radius range
    (triangle w/ 4 discs, square/pentagon w/ 3 huge discs) must now generate: closeness maps to
    the figure's OWN feasible radius range."""
    for nv, np_ in [(3, 4), (4, 3), (5, 3)]:
        stems = [main.generate_one(0, nv, np_, 0.1, 0.1, closeness=1.0,
                                   sil_dir=str(SIL), ctrl_dir=str(SIL_CTRL), seed=s)
                 for s in range(1, 6)]
        assert any(stems), f"nv={nv} np={np_} still infeasible at closeness=1.0"


def test_closeness_arbitrary_values_roundtrip():
    """Any float knob value must generate AND round-trip through the filename (no 1e-05 stems)."""
    for c in (1 / 3, 0.9999999, 1e-05, 1e-12, -0.2, 1.7):
        stem = main.generate_one(0, 4, 5, 0.1, 0.1, closeness=c,
                                 sil_dir=str(SIL), ctrl_dir=str(SIL_CTRL), seed=3)
        assert stem is not None
        p = S.parse_gestalt_name(stem)
        expect = round(min(1.0, max(0.0, float(c))), 4)
        assert p is not None and p["closeness"] == expect


def test_parse_gestalt_name_closeness_roundtrip():
    p = S.parse_gestalt_name("img_0_4_4_0.1_0.1_0.75")
    assert p["closeness"] == 0.75 and p["n_points"] == 4
    assert "closeness" not in S.parse_gestalt_name("img_0_4_4_0.1_0.1")


def test_dropin_merges_generation_sidecar():
    stem = _gen(0.5, seed=4)
    assert S.use_dropin(files=[f"{stem}.png"]) == [stem]
    meta = json.loads((SIL / f"{stem}.json").read_text())
    assert meta["n_inducers"] == 4 and meta["closeness"] == 0.5
    assert 0 < meta["support_ratio"] < 1         # generation sidecar survived the meta write


def _kanizsa():
    S.produce_kanizsa(n_sides=3)
    return SIL / "kanizsa_triangle.png"


def test_texture0_flat_black_on_white():
    img, _ = DC.render(_kanizsa(), "white disc", "black bg", shadow=0, crop=False, texture=0)
    a = np.asarray(img)
    from pipeline.geometry import inducer_labels
    lab, n, _ = inducer_labels(_kanizsa())
    ys, xs = np.where(lab == 1)
    assert a[int(ys.mean()), int(xs.mean())].mean() < 40    # inducer painted black
    assert a[5, 5].mean() > 240                             # flat white fill, no bg image


def test_texture0_camouflage_darkens_fill():
    info = {}
    img, _ = DC.render(_kanizsa(), "white disc", "black bg", shadow=0, crop=False,
                       texture=0, contrast=0.6, info=info)
    assert abs(np.asarray(img)[5, 5].mean() - 0.4 * 255) < 3   # fill slides toward the black inducers
    assert abs(info["lum_gap"] - 0.4) < 0.02


def test_texture1_solid_canvas_with_camouflage():
    info = {}
    img, _ = DC.render(_kanizsa(), "white disc", "black bg", shadow=0, crop=False,
                       texture=1, contrast=0.6, info=info)
    a = np.asarray(img)
    corner = a[5, 5].astype(float).mean() / 255
    assert abs(corner - 0.6) < 0.05              # bg moved 60% of the way to the white object
    assert abs(info["lum_gap"] - 0.4) < 0.05
    assert a[5, 5].std() < 2                     # solid color, not the bg image


def test_texture3_camouflage_recolors_background():
    info = {}
    base, _ = DC.render(_kanizsa(), "white disc", "black bg", shadow=0, crop=False, texture=3)
    shifted, _ = DC.render(_kanizsa(), "white disc", "black bg", shadow=0, crop=False,
                           texture=3, contrast=0.4, info=info)
    b, s = np.asarray(base).astype(float), np.asarray(shifted).astype(float)
    assert s[5, 5].mean() > b[5, 5].mean() + 20  # black bg pulled toward the white object
    assert abs(info["lum_gap"] - 0.6) < 0.05


def test_camouflage_converges_in_hue_not_just_luminance():
    # red object on white bg: at high camouflage the bg must move TOWARD RED, not just darken
    info = {}
    img, _ = DC.render(_kanizsa(), "red disc", "white bg", shadow=0, crop=False,
                       texture=1, contrast=0.9, info=info)
    r, g, b = np.asarray(img)[5, 5].astype(float)
    assert r > 200 and g < 60 and b < 60          # corner is nearly the object's red
    assert info["lum_gap"] < 0.1


def test_camouflage_preserves_background_detail():
    # textured bg: after a big recolor shift the texture must survive (re-centered, not clipped flat)
    from pipeline.utils import BG
    rng = np.random.default_rng(0)
    noise = rng.integers(40, 90, (512, 512, 3), dtype=np.uint8)   # dark noisy surface
    Image.fromarray(noise, "RGB").save(BG / "noise_bg.png")
    shifted, _ = DC.render(_kanizsa(), "white disc", "noise bg", shadow=0, crop=False,
                           texture=3, contrast=0.8)               # big shift toward white
    patch = np.asarray(shifted)[:40, :40].astype(float)
    assert patch.std() > 4                        # detail survived; naive clipping would flatten it


def _noise_disc(path, size=512, seed=7):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size, 4), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    arr[..., 3] = (((xx - size / 2) ** 2 + (yy - size / 2) ** 2) <= (size / 3) ** 2) * 255
    Image.fromarray(arr, "RGBA").save(path)


def test_texture_levels_discrete():
    from pipeline.utils import BG, OBJ
    from pipeline.geometry import inducer_labels
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8), "RGB").save(BG / "noise_bg.png")
    _noise_disc(OBJ / "noise_disc.png")
    lab, _, _ = inducer_labels(_kanizsa())
    ys, xs = np.where(lab == 1)
    cy, cx = int(ys.mean()), int(xs.mean())
    imgs = {t: np.asarray(DC.render(_kanizsa(), "noise disc", "noise bg", shadow=0, crop=False,
                                    texture=t)[0]).astype(float) for t in (1, 2, 3)}
    def std_at(a, y, x):
        return a[y - 12:y + 12, x - 12:x + 12].std()
    assert std_at(imgs[1], cy, cx) < 5 < std_at(imgs[2], cy, cx)   # 1: flat chip, 2: textured object
    assert imgs[2][:40, :40].std() < 5 < imgs[3][:40, :40].std()   # 2: solid bg, 3: textured bg


def test_gt_carries_difficulty_block(cfg):
    stem = _gen(0.5, seed=5)
    metas = [json.loads((SIL / f"{stem}.json").read_text()) | {"stem": stem}]
    S.use_dropin(files=[f"{stem}.png"])          # upgrade sidecar to full meta
    metas = [json.loads((SIL / f"{stem}.json").read_text()) | {"stem": stem}]
    pairs = {stem: [{"object": "white disc", "background": "black bg", "natural": False}]}
    c = cfg(dress={"backends": ["composite"], "modes": ["perfect"], "conditions": ["illusory"],
                   "contour": "auto", "shadow": 0, "crop": True, "feather": 1.5,
                   "texture": 1, "contrast": 0.5})
    DC.run(c, metas, pairs)
    gt = json.loads(next((OUT / "gt").glob("*.json")).read_text())
    d = gt["difficulty"]
    assert d["texture"] == 1 and d["contrast"] == 0.5 and d["closeness"] == 0.5
    assert 0 < d["support_ratio"] < 1 and abs(d["lum_gap"] - 0.5) < 0.05
