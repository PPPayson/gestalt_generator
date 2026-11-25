"""IllusionDataset: triplets read from a pipeline output folder."""
import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
from src.data.dataset import IllusionDataset  # noqa: E402


def _png(path, arr):
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def _world(root, stems=("a", "b"), partial=("c",)):
    for d in ("silhouettes", "control", "negative"):
        (root / d).mkdir(parents=True)
    pos = np.zeros((8, 8), np.uint8); pos[2:4, 2:4] = 1
    ring = np.zeros((8, 8), np.uint8); ring[1, 1:6] = 1
    ctrl = pos | ring
    neg = np.zeros((8, 8), np.uint8); neg[5:7, 5:7] = 1
    for s in stems:
        _png(root / "silhouettes" / f"{s}.png", pos); _png(root / "control" / f"{s}.png", ctrl)
        _png(root / "negative" / f"{s}.png", neg)
    for s in partial:                                      # silhouette + control, no negative
        _png(root / "silhouettes" / f"{s}.png", pos); _png(root / "control" / f"{s}.png", ctrl)
    return pos, ring, neg


def test_dataset_reads_complete_triplets(tmp_path):
    pos, ring, neg = _world(tmp_path)
    ds = IllusionDataset(str(tmp_path))
    assert ds.stems == ["a", "b"]                          # the stem with no negative is dropped
    r, p, n = ds[0]
    assert all(a.dtype == np.float32 and set(np.unique(a)) <= {0.0, 1.0} for a in (r, p, n))
    assert np.array_equal(r, ring) and np.array_equal(p, pos) and np.array_equal(n, neg)


def test_dataset_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        IllusionDataset(str(tmp_path / "nowhere"))
    _world(tmp_path, stems=(), partial=("c",))
    with pytest.raises(RuntimeError):
        IllusionDataset(str(tmp_path))
