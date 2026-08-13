import pytest
from pipeline import providers


def test_load_custom_dotted_path(cfg):
    """Someone plugs in their own model via 'module:Class'."""
    c = cfg(models={"image": {"provider": "tests.fakeprovider:FakeProvider", "model": "m", "params": {"k": 1}}})
    p = providers.load_image(c)
    assert p.__class__.__name__ == "FakeProvider"
    assert p.model == "m" and p.params == {"k": 1}       # params passthrough
    assert isinstance(p.gen_object("x", 0), bytes)


def test_load_unknown_provider_errors(cfg):
    c = cfg(models={"image": {"provider": "nope_not_real"}})
    with pytest.raises(ValueError):
        providers.load_image(c)


def test_builtin_names_registered():
    assert set(providers.BUILTINS) == {"hf_api", "modal_hf"}


def test_load_proposer_custom(cfg):
    c = cfg(models={"proposer": {"provider": "tests.fakeprovider:FakeProvider"}})
    p = providers.load_proposer(c)
    assert "white disc" in p.propose("kanizsa", "disc", True, 1)
