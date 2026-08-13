"""modal_hf tested with a FAKE `modal` module — no real Modal, no deploy, no network."""
import sys, types
import pytest


def _fake_modal():
    m = types.ModuleType("modal")
    m.REC = []

    class _Method:
        def __init__(self, name):
            self.name = name
        def remote(self, *a, **k):
            m.REC.append(("remote", self.name, a)); return b"PNG_" + self.name.encode()

    class _Instance:
        def __init__(self, params):
            m.REC.append(("construct", params))
            for name in ("gen_object", "gen_bg", "dress_controlnet", "propose"):
                setattr(self, name, _Method(name))

    class _Handle:
        def __init__(self, app, name):
            self.app = app; self.name = name
        def with_options(self, **opts):
            m.REC.append(("with_options", self.name, opts)); return self
        def __call__(self, **params):
            return _Instance(params)

    class _Cls:
        @staticmethod
        def from_name(app, name, **kw):
            m.REC.append(("from_name", app, name)); return _Handle(app, name)

    class _Secret:
        @staticmethod
        def from_name(name, **kw):
            m.REC.append(("secret", name)); return f"secret:{name}"

    m.Cls = _Cls; m.Secret = _Secret
    return m


@pytest.fixture
def modal_hf(monkeypatch):
    fake = _fake_modal()
    monkeypatch.setitem(sys.modules, "modal", fake)
    import importlib, pipeline.providers.modal_hf as M
    importlib.reload(M)
    return fake, M


def test_gen_object_delegates_to_remote(modal_hf):
    fake, M = modal_hf
    p = M.ModalHF(gpu="L4", secret=None)
    out = p.gen_object("poker chip", 0)
    assert out == b"PNG_gen_object"
    assert ("from_name", "perceptual-grouping", "SDXL") in fake.REC
    assert any(r[0] == "with_options" and r[2].get("gpu") == "L4" for r in fake.REC)


def test_gen_bg_and_dress(modal_hf):
    fake, M = modal_hf
    p = M.ModalHF(gpu="L4")
    assert p.gen_bg("felt", 0) == b"PNG_gen_bg"
    assert p.dress(b"sil", "prompt", 0.7, 0) == b"PNG_dress_controlnet"


def test_secret_attached(modal_hf):
    fake, M = modal_hf
    p = M.ModalHF(gpu="L4", secret="gestalt-generation")
    p.gen_object("x", 0)
    assert ("secret", "gestalt-generation") in fake.REC
    assert any(r[0] == "with_options" and "secrets" in r[2] for r in fake.REC)


def test_propose_uses_llm(modal_hf):
    fake, M = modal_hf
    p = M.ModalHF()
    assert p.propose("kanizsa_triangle", "disc", True, 3) == b"PNG_propose"
    assert ("from_name", "perceptual-grouping", "LLM") in fake.REC
