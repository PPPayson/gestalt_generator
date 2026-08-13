"""modal_hf — the maintainer's access point. Calls a DEPLOYED Modal app (`modal deploy app.py`)
so the rest of the repo never imports modal. Does images + proposer + dress (ControlNet).

Setup once:  modal deploy app.py
Then:        set models.image.provider / models.proposer.provider to "modal_hf".
Optional secret (for gated models): create it and set `secret: <name>` in the config.
"""
from . import base

APP = "perceptual-grouping"


class ModalHF:
    def __init__(self, model=None, controlnet=None, gpu="L4", secret=None, params=None, **_ignored):
        import modal
        self.params = params or {}

        def _cls(name):
            c = modal.Cls.from_name(APP, name)
            opts = {}
            if gpu:
                opts["gpu"] = gpu
            if secret:
                opts["secrets"] = [modal.Secret.from_name(secret)]
            return c.with_options(**opts) if opts else c

        # SDXL image class (object/bg/dress); LLM class (propose). Instantiate lazily on first use.
        self._Sdxl = _cls("SDXL")
        self._Llm = _cls("LLM")
        self._model = model or base.SDXL_MODEL
        self._cn = controlnet or base.CN_MODEL
        self._sdxl = None
        self._llm = None

    def _sdxl_h(self):
        if self._sdxl is None:
            self._sdxl = self._Sdxl(model_id=self._model, cn_id=self._cn)
        return self._sdxl

    def _llm_h(self):
        if self._llm is None:
            self._llm = self._Llm(model_id=self._model)
        return self._llm

    # --- image ---
    def gen_object(self, name, seed=0):
        return self._sdxl_h().gen_object.remote(name, seed)

    def gen_bg(self, name, seed=0):
        return self._sdxl_h().gen_bg.remote(name, seed)

    def dress(self, sil_png, prompt, scale=0.7, seed=0):
        return self._sdxl_h().dress_controlnet.remote(sil_png, prompt, scale, seed)

    # --- proposer ---
    def propose(self, illusion, inducer_shape, natural=True, n=12, seed=0):
        return self._llm_h().propose.remote(illusion, inducer_shape, natural, n, seed)
