"""Provider loading: builtin short name OR "module:Class" dotted path -> constructed instance.

Shipped (both free): hf_api (DEFAULT, pure HuggingFace) and modal_hf (maintainer, deployed Modal app).
Bring your own for testing: set provider to "yourmodule:YourClass" (see base.py / template.py).
"""
from importlib import import_module

BUILTINS = {
    "hf_api": "pipeline.providers.hf_api:HFApi",
    "modal_hf": "pipeline.providers.modal_hf:ModalHF",
}


def _construct(spec, params):
    path = BUILTINS.get(spec, spec)
    if ":" not in path:
        raise ValueError(f"unknown provider '{spec}'. Use a builtin {list(BUILTINS)} "
                         f"or a 'module:Class' dotted path.")
    mod, cls = path.split(":", 1)
    return getattr(import_module(mod), cls)(**params)


def _load(model_cfg):
    cfg = dict(model_cfg or {})
    spec = cfg.pop("provider", "hf_api")
    return _construct(spec, cfg)


def load_image(cfg):
    return _load(cfg["models"]["image"])


def load_proposer(cfg):
    return _load(cfg["models"]["proposer"])
