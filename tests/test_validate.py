import json
from pipeline import validate
from pipeline.utils import DATA


def _cand(monkeypatch, items):
    from pipeline import validate as V
    monkeypatch.setattr(V, "load_all_candidates", lambda: {"s1": items})


def test_high_contrast_disc_kept(cfg, monkeypatch):
    _cand(monkeypatch, [{"object": "white disc", "background": "black bg", "natural": False}])
    prs = validate.run(cfg(), [{"stem": "s1"}])
    assert len(prs["s1"]) == 1
    assert (DATA / "pairs.json").exists()


def test_low_contrast_non_natural_dropped(cfg, monkeypatch):
    _cand(monkeypatch, [{"object": "gray disc", "background": "gray bg", "natural": False}])
    prs = validate.run(cfg(), [{"stem": "s1"}])
    assert prs["s1"] == []                          # contrast 0 < 0.22 and not natural


def test_missing_asset_dropped(cfg, monkeypatch):
    _cand(monkeypatch, [{"object": "nonexistent", "background": "black bg", "natural": False}])
    prs = validate.run(cfg(), [{"stem": "s1"}])
    assert prs["s1"] == []
