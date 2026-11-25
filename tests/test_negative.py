"""Negatives: same inducer blobs, every one moved, seeded, for stored silhouettes of any origin."""
import numpy as np
from PIL import Image
from scipy import ndimage
from pipeline import silhouettes as S
from pipeline.negative import make_negative, ensure_negatives, _blobs
from pipeline.utils import SIL, SIL_NEG


def _arr(p):
    return np.array(Image.open(p).convert("L"))


def _blob_sizes(a):
    lab, ids = _blobs(a)
    return sorted(int((lab == i).sum()) for i in ids)


def _centroids(a):
    lab, ids = _blobs(a)
    return np.array(ndimage.center_of_mass(np.ones_like(a), lab, ids))


def test_negative_keeps_parts_moves_all(gestalt_stem, tmp_path):
    src = SIL / f"{gestalt_stem}.png"; dst = tmp_path / "neg.png"
    assert make_negative(src, dst, seed=1)
    a, b = _arr(src), _arr(dst)
    assert _blob_sizes(a) == _blob_sizes(b)                       # same blobs, same pixel counts
    assert not np.array_equal(a, b)
    # match blobs across the two images by pixel count (unique here): each one has moved
    la, ia = _blobs(a); lb, ib = _blobs(b)
    pos_a = {int((la == i).sum()): ndimage.center_of_mass(la == i) for i in ia}
    pos_b = {int((lb == i).sum()): ndimage.center_of_mass(lb == i) for i in ib}
    if len(pos_a) == len(ia):
        for k in pos_a:
            assert np.hypot(*(np.subtract(pos_a[k], pos_b[k]))) > 20


def test_negative_is_seeded(gestalt_stem, tmp_path):
    src = SIL / f"{gestalt_stem}.png"
    make_negative(src, tmp_path / "a.png", seed=1); make_negative(src, tmp_path / "b.png", seed=1)
    assert np.array_equal(_arr(tmp_path / "a.png"), _arr(tmp_path / "b.png"))


def test_negative_for_kanizsa(tmp_path):
    S.produce_kanizsa(n_sides=4)
    assert make_negative(SIL / "kanizsa_square.png", tmp_path / "k.png", seed=0)
    assert _blob_sizes(_arr(SIL / "kanizsa_square.png")) == _blob_sizes(_arr(tmp_path / "k.png"))


def test_ensure_negatives_fills_store(gestalt_stem):
    (SIL_NEG / f"{gestalt_stem}.png").unlink(missing_ok=True)
    assert ensure_negatives([gestalt_stem], seed=1) == []
    assert (SIL_NEG / f"{gestalt_stem}.png").exists()
