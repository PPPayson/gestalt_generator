"""hf_api tested fully offline: fake huggingface_hub.InferenceClient + fake rembg injected into
sys.modules; no real network. (Real end-to-end is verified separately with a live token.)"""
import io, sys, types
import pytest
from PIL import Image


class _FakeClient:
    def __init__(self, token=None):
        self.token = token; self.calls = []
    def text_to_image(self, prompt, model=None, seed=None):
        self.calls.append(("t2i", prompt, model, seed))
        return Image.new("RGB", (64, 64), (10, 120, 60))
    def chat_completion(self, messages=None, model=None, max_tokens=None, temperature=None):
        self.calls.append(("chat", model))
        msg = types.SimpleNamespace(content='[{"object":"a","background":"b"}]')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


@pytest.fixture
def hf(monkeypatch):
    hub = types.ModuleType("huggingface_hub")
    hub.InferenceClient = _FakeClient
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    rembg = types.ModuleType("rembg")
    rembg.remove = lambda im: im                     # passthrough cutout
    monkeypatch.setitem(sys.modules, "rembg", rembg)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    import pipeline.providers.hf_api as H
    return H


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    from pipeline.providers.hf_api import HFApi
    with pytest.raises(RuntimeError):
        HFApi(model="m")


def test_gen_bg_returns_png_with_model(hf):
    p = hf.HFApi(model="stabilityai/stable-diffusion-xl-base-1.0")
    out = p.gen_bg("green felt", 7)
    Image.open(io.BytesIO(out))                       # valid PNG
    kind, prompt, model, seed = p.client.calls[-1]
    assert kind == "t2i" and seed == 7
    assert model == "stabilityai/stable-diffusion-xl-base-1.0"
    assert "green felt" in prompt                     # background prompt templated from the name


def test_gen_object_uses_rembg(hf):
    p = hf.HFApi(model="m")
    Image.open(io.BytesIO(p.gen_object("poker chip", 0)))


def test_propose_parses_choices(hf):
    import json
    p = hf.HFApi(model="Qwen/Qwen2.5-7B-Instruct")
    assert json.loads(p.propose("kanizsa_triangle", "disc", True, 3))[0]["object"] == "a"


def test_hf_api_has_no_dress(hf):
    assert not hasattr(hf.HFApi(model="m"), "dress")
