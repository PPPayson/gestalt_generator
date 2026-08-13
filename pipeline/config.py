"""Load + validate a benchmark YAML config, filling defaults."""
import pathlib, yaml

DEFAULTS = {
    "name": "benchmark",
    "seed": 0,                              # global seed: selection + generation are reproducible
    "silhouettes": [{"type": "kanizsa", "params": {"n_sides": 3}},
                    {"type": "kanizsa", "params": {"n_sides": 4}}],
    # Generation models. provider = builtin name (hf_api | modal_hf) OR "module:Class" (bring your own).
    "models": {
        "image": {"provider": "hf_api",
                  "model": "stabilityai/stable-diffusion-xl-base-1.0",
                  "controlnet": "diffusers/controlnet-canny-sdxl-1.0",
                  "gpu": "L4", "secret": None, "params": {}},
        "proposer": {"provider": "hf_api",
                     "model": "Qwen/Qwen2.5-7B-Instruct",
                     "gpu": "L4", "secret": None, "params": {}},
    },
    "credentials": {"secret": None},        # Modal secret name (modal_hf, gated models); HF via env HF_TOKEN
    "pairs": {"source": "catalog",          # catalog | propose | explicit
              "select": "contrast",         # contrast | random
              "seed": None,                 # None -> inherit top-level seed
              "n_per_silhouette": 8,
              "filter": {"natural": None, "min_contrast": 0.0, "objects": None},
              "explicit": []},              # [{object, background, natural}] when source=explicit
    "propose": {"n_per": 12, "natural": [True, False]},
    "assets": {"best_of": 3},
    "dress": {"backends": ["composite"], "modes": ["perfect"], "shadow": 55},
    "select": {"round_min": 0.55, "contrast_high": 0.22, "contrast_low_natural": 0.10},
    "output": {"contact_sheet": True},
}


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load(path):
    raw = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    cfg = _merge(DEFAULTS, raw)
    assert cfg["pairs"]["source"] in ("catalog", "propose", "explicit"), "bad pairs.source"
    for b in cfg["dress"]["backends"]:
        assert b in ("composite", "controlnet"), f"bad backend {b}"
    return cfg
