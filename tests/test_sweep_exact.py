"""`n_figures: all` with rng: shared reproduces the in-memory closure-sweep triplets bit for bit.
Oracle hashes (md5, first 12 hex, over uint8 0/1 arrays of ring, illusory, negative) were recorded on
2026-11-24 from the original in-memory generator (IllusionDataset, seed 0) for figure indices spanning
the whole 21,600-item sweep: first items, a middle cell, the last combination and the first repeat.
The hashes depend on numpy's Generator stream (integers, permutation) and on the polygon and mask
rasterisation (scipy, OpenCV). A deliberate upgrade of those can change them with no bug in this
repo: then re-record from a run checked by eye, never loosen the comparison."""
import hashlib
import numpy as np
from PIL import Image
from pipeline import silhouettes
from pipeline.utils import SIL, SIL_CTRL, SIL_NEG

SWEEP = dict(space={"polygon_randomness": [0, 0.4, 0.8, 1], "n_vertices": [3, 4, 5, 6, 7, 8],
                    "n_inducers": [3, 4, 5, 6], "corner_sharpness": [0.1, 0.5, 1], "side_bulge": [0.1, 0.2, 0.4],
                    "inducer_radius": [100, 75, 60, 50]},
             n_figures="all", figures_per_combination=10, seed=0, polygon_size=[600, 600],
             inducer_scale=100, quality_filter=False, rng="shared")
ORACLE = {0: ("672ea90686cd", "ea8c787f2b0f", "0fa48a26ceaa"),
          1: ("54bfe1d73e33", "c9c793afd8c5", "ad8f83329609"),
          2: ("e22d14adecae", "7b5347610396", "c9625a2587f1"),
          300: ("2506c7c37433", "892c6267ea3a", "1e147d7b5252"),
          2159: ("7accc566fa1c", "7c8aab361312", "e2de5e7c1286"),
          2160: ("4605ce6e12d0", "d7331efd31c0", "bff21a39f770"),
          21599: ("39e4df3adb0b", "5f458af95957", "949ee4160a82")}


def _bin(path):
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def _h(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def test_sweep_matches_oracle():
    stems = silhouettes.use_generate(indices=sorted(ORACLE), jobs=1, **SWEEP)
    assert len(stems) == len(ORACLE)
    for idx, stem in zip(sorted(ORACLE), stems):
        pos = _bin(SIL / f"{stem}.png")
        ctrl = _bin(SIL_CTRL / f"{stem}.png")
        neg = _bin(SIL_NEG / f"{stem}.png")
        ring = (ctrl & ~pos.astype(bool)).astype(np.uint8)
        assert (_h(ring), _h(pos), _h(neg)) == ORACLE[idx], f"figure {idx} ({stem}) differs"
