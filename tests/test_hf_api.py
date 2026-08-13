"""hf_api tested fully offline: fake `requests` + `rembg` injected into sys.modules; no token needed
except where we assert the missing-token error."""
import io, sys, types, json
import pytest
from PIL import Image


def _png_bytes(color=(200, 100, 50)):
    b = io.BytesIO(); Image.new("RGB", (64, 64), color).save(b, format="PNG"); return b.getvalue()


class _Resp:
    def __init__(self, status=200, content=b"", payload=None):
        self.status_code = status; self.content = content; self._payload = payload; self.text = "err"
    def json(self):
        return self._payload


class _FakeRequests:
    """Scriptable: `statuses` consumed per call; records calls."""
    def __init__(self):
        self.statuses = [200]; self.calls = []; self.payload = None
    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        st = self.statuses.pop(0) if self.statuses else 200
        if "chat/completions" in url:
            return _Resp(st, payload={"choices": [{"message": {"content": '[{"object":"a","background":"b"}]'}}]})
        return _Resp(st, content=_png_bytes())


@pytest.fixture
def hf(monkeypatch):
    fake_req = _FakeRequests()
    monkeypatch.setitem(sys.modules, "requests", fake_req)
    rembg = types.ModuleType("rembg")
    rembg.remove = lambda im: im                     # passthrough cutout
    monkeypatch.setitem(sys.modules, "rembg", rembg)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    import pipeline.providers.hf_api as H
    monkeypatch.setattr(H.time, "sleep", lambda *_: None)   # no real waiting on 503 retry
    return fake_req, H


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    from pipeline.providers.hf_api import HFApi
    with pytest.raises(RuntimeError):
        HFApi(model="m")


def test_gen_bg_returns_png_and_sends_model(hf):
    fake_req, H = hf
    p = H.HFApi(model="stabilityai/stable-diffusion-xl-base-1.0")
    out = p.gen_bg("green felt", 0)
    Image.open(io.BytesIO(out))                       # valid PNG
    assert "stable-diffusion-xl-base-1.0" in fake_req.calls[-1]["url"]
    assert fake_req.calls[-1]["headers"]["Authorization"] == "Bearer hf_test"


def test_gen_object_uses_rembg(hf):
    _, H = hf
    p = H.HFApi(model="m")
    out = p.gen_object("poker chip", 0)
    Image.open(io.BytesIO(out))


def test_503_then_success(hf):
    fake_req, H = hf
    fake_req.statuses = [503, 200]
    p = H.HFApi(model="m")
    assert p.gen_bg("x", 0)                            # retried and succeeded
    assert len(fake_req.calls) == 2


def test_propose_parses_choices(hf):
    _, H = hf
    p = H.HFApi(model="Qwen/Qwen2.5-7B-Instruct")
    txt = p.propose("kanizsa_triangle", "disc", True, 3)
    assert json.loads(txt)[0]["object"] == "a"


def test_hf_api_has_no_dress(hf):
    _, H = hf
    assert not hasattr(H.HFApi(model="m"), "dress")
