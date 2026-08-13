"""Test harness: redirect all I/O to a tmp world (NO network / modal / gpu / token).

GESTALT_DATA / GESTALT_OUT are set BEFORE importing pipeline so utils' module-level paths resolve
into the tmp dir. The world is seeded with 2 kanizsa + 3 real gestalt silhouettes and FAKE assets
(a filled-disc object + solid-color backgrounds) so nothing depends on the shipped pack or a model.
"""
import os, sys, tempfile, pathlib, shutil, json
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
_TMP = pathlib.Path(tempfile.mkdtemp(prefix="gestalt_test_"))
os.environ["GESTALT_DATA"] = str(_TMP / "data")
os.environ["GESTALT_OUT"] = str(_TMP / "out")

# import pipeline only AFTER env is set
from pipeline import utils, silhouettes, config          # noqa: E402
from PIL import Image, ImageDraw                          # noqa: E402

FIXTURE_CATALOG = _TMP / "fixture_catalog.yaml"


def _disc(color, size=512):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    m = size // 6
    d.ellipse([m, m, size - m, size - m], fill=color)
    return im


def _solid(color, size=512):
    return Image.new("RGB", (size, size), color)


def _build_world():
    utils.SIL.mkdir(parents=True, exist_ok=True)
    utils.OBJ.mkdir(parents=True, exist_ok=True)
    utils.BG.mkdir(parents=True, exist_ok=True)
    utils.OUT.mkdir(parents=True, exist_ok=True)
    # silhouettes: 2 registry kanizsa + 3 real CLEAN gestalt drop-ins (all inducers separate)
    silhouettes.produce_kanizsa(n_sides=3)
    silhouettes.produce_kanizsa(n_sides=4)
    from pipeline.geometry import separate_inducer_count
    utils.SIL_CTRL.mkdir(parents=True, exist_ok=True)
    clean = []
    for f in sorted((REPO / "data" / "silhouettes").glob("img_*.png")):
        p = silhouettes.parse_gestalt_name(f.stem)
        if p and separate_inducer_count(f) == p["n_points"]:
            shutil.copy(f, utils.SIL / f.name); clean.append(f.name)
            ctrl = REPO / "data" / "control" / f.name           # copy the paired control too
            if ctrl.exists():
                shutil.copy(ctrl, utils.SIL_CTRL / f.name)
        if len(clean) >= 3:
            break
    # fake assets (slug names): white/gray discs, black/white/gray backgrounds
    _disc((255, 255, 255, 255)).save(utils.OBJ / "white_disc.png")
    _disc((128, 128, 128, 255)).save(utils.OBJ / "gray_disc.png")
    _disc((255, 0, 0, 255)).save(utils.OBJ / "red_disc.png")
    _solid((0, 0, 0)).save(utils.BG / "black_bg.png")
    _solid((255, 255, 255)).save(utils.BG / "white_bg.png")
    _solid((128, 128, 128)).save(utils.BG / "gray_bg.png")
    # fixture catalog: high-contrast + low-contrast pairs (enough for random-select tests)
    pairs = [
        {"object": "white disc", "background": "black bg", "natural": False, "contrast": 1.0, "roundness": 1.0},
        {"object": "red disc",   "background": "black bg", "natural": False, "contrast": 0.6, "roundness": 1.0},
        {"object": "white disc", "background": "gray bg",  "natural": False, "contrast": 0.5, "roundness": 1.0},
        {"object": "red disc",   "background": "white bg", "natural": True,  "contrast": 0.4, "roundness": 1.0},
        {"object": "gray disc",  "background": "gray bg",  "natural": True,  "contrast": 0.0, "roundness": 1.0},
    ]
    FIXTURE_CATALOG.write_text("pairs:\n" + "\n".join(
        f"  - {{object: \"{p['object']}\", background: \"{p['background']}\", "
        f"natural: {str(p['natural']).lower()}, contrast: {p['contrast']}, roundness: {p['roundness']}}}"
        for p in pairs) + "\n")


_build_world()


@pytest.fixture(autouse=True)
def clean_out():
    """Fresh out/ + manifest per test so counts/ids don't bleed across tests."""
    if utils.OUT.exists():
        shutil.rmtree(utils.OUT)
    utils.OUT.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture
def gestalt_stem():
    return sorted(p.stem for p in utils.SIL.glob("img_*.png"))[0]


@pytest.fixture
def cfg():
    """Factory: merged config dict with overrides."""
    def _make(**over):
        return config._merge(config.DEFAULTS, over)
    return _make


@pytest.fixture
def use_fixture_catalog(monkeypatch):
    from pipeline import pairs
    monkeypatch.setattr(pairs, "CATALOG", FIXTURE_CATALOG)
    return FIXTURE_CATALOG
