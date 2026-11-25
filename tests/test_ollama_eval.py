"""ollama_eval runs every condition folder that exists and fails loudly when none does."""
import argparse
import sys
import types
import pytest
from PIL import Image

if "ollama" not in sys.modules:                            # the client library is not needed here
    try:
        import ollama  # noqa: F401
    except ImportError:
        sys.modules["ollama"] = types.SimpleNamespace(chat=None, show=None)
import ollama_eval as OE  # noqa: E402


def _args(data_dir, variant=None):
    return argparse.Namespace(data_dir=str(data_dir), variant=variant, max_images=None, reps=1,
                              provider="ollama", model="fake", max_tokens=16, max_context=16, thinking=False, temperature=0.0)


def test_runs_present_condition_folders(tmp_path, monkeypatch):
    for d in ("illusory", "negative"):                     # no control/ folder
        (tmp_path / d).mkdir()
        Image.new("RGB", (8, 8)).save(tmp_path / d / "stim_1.png")
    monkeypatch.setattr(OE, "get_model_response", lambda **kw: {"has_illusion": True})
    out = OE.run_model(_args(tmp_path))
    assert set(out) == {"illusory", "negative"}
    assert out["negative"]["stim_1.png"]["0"] == {"has_illusion": True}


def test_no_folder_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(OE, "get_model_response", lambda **kw: {})
    with pytest.raises(FileNotFoundError):
        OE.run_model(_args(tmp_path, variant="texture"))
