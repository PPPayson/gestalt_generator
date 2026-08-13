import pytest
from pipeline import config


def test_defaults_present():
    d = config.DEFAULTS
    assert d["seed"] == 0
    assert d["models"]["image"]["provider"] == "hf_api"
    assert d["models"]["proposer"]["provider"] == "hf_api"
    assert d["pairs"]["seed"] is None            # inherits global seed


def test_merge_is_deep():
    merged = config._merge(config.DEFAULTS, {"pairs": {"select": "random"}})
    assert merged["pairs"]["select"] == "random"
    assert merged["pairs"]["source"] == "catalog"   # untouched sibling kept


def test_load_valid(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: t\npairs: {source: explicit}\ndress: {backends: [composite]}\n")
    cfg = config.load(p)
    assert cfg["name"] == "t" and cfg["pairs"]["source"] == "explicit"


def test_load_rejects_bad_source(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text("pairs: {source: nope}\n")
    with pytest.raises(AssertionError):
        config.load(p)


def test_load_rejects_bad_backend(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text("dress: {backends: [wat]}\n")
    with pytest.raises(AssertionError):
        config.load(p)
