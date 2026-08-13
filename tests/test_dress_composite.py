import numpy as np
from pipeline import dress_composite as DC, silhouettes as S
from pipeline.utils import SIL, OUT
from pipeline.geometry import inducer_labels


def _sil():
    S.produce_kanizsa(n_sides=3)
    return SIL / "kanizsa_triangle.png"


def test_render_fills_inducer_regions():
    # crop=False so image coords match inducer_labels(uncropped silhouette)
    img, (w, h) = DC.render(_sil(), "white disc", "black bg", shadow=0, mode="perfect", crop=False)
    assert img.size == (w, h)
    lab, n, _ = inducer_labels(_sil())
    a = np.asarray(img)
    # inside an inducer region -> bright (white disc); outside -> black bg
    ys, xs = np.where(lab == 1)
    cy, cx = int(ys.mean()), int(xs.mean())
    assert a[cy, cx].mean() > 150
    assert a[5, 5].mean() < 60                       # corner is background


def test_render_deterministic():
    a, _ = DC.render(_sil(), "white disc", "black bg", shadow=55, mode="perfect")
    b, _ = DC.render(_sil(), "white disc", "black bg", shadow=55, mode="perfect")
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_render_aspect_preserved_for_gestalt():
    stem = sorted(p.stem for p in SIL.glob("img_*.png"))[0]
    img, (w, h) = DC.render(SIL / f"{stem}.png", "white disc", "black bg", shadow=0)
    assert max(w, h) == 1024 and img.size == (w, h)


def test_run_writes_stimuli_and_manifest(cfg):
    _sil()
    metas = [{"stem": "kanizsa_triangle", "illusion": "kanizsa_triangle",
              "illusory_shape": "triangle"}]
    pairs = {"kanizsa_triangle": [{"object": "white disc", "background": "black bg", "natural": False}]}
    DC.run(cfg(), metas, pairs)
    assert len(list((OUT / "comp").glob("*.png"))) == 1
    assert (OUT / "manifest.csv").exists()
    assert len(list((OUT / "gt").glob("*.json"))) == 1
