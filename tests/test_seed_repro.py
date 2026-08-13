"""Reproducibility: one global seed controls selection (gestalt + random textures) and generation.
Same seed -> identical stimuli; different seed -> different selection."""
import shutil
from pipeline import orchestrate
from pipeline.utils import OUT


def _cfg(cfg, seed):
    return cfg(seed=seed,
               silhouettes=[{"type": "gestalt", "params": {"mode": "existing", "n": 2}}],
               pairs={"source": "catalog", "select": "random", "n_per_silhouette": 2},
               models={"image": {"provider": "tests.fakeprovider:FakeProvider"}},
               dress={"backends": ["composite"]}, output={"contact_sheet": False})


def _run_collect(cfg, seed):
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    orchestrate.run(_cfg(cfg, seed))
    return {f.name: f.read_bytes() for f in (OUT / "comp").glob("*.png")}


def test_same_seed_identical(cfg, use_fixture_catalog):
    a = _run_collect(cfg, 1)
    b = _run_collect(cfg, 1)
    assert a.keys() == b.keys() and a                 # same ids
    assert all(a[k] == b[k] for k in a)               # identical bytes


def test_different_seed_changes_selection(cfg, use_fixture_catalog):
    id_sets = {frozenset(_run_collect(cfg, s)) for s in range(6)}
    assert len(id_sets) >= 2                           # selection varies with seed
