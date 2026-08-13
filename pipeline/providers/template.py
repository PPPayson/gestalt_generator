"""COPY ME to plug in your own model.

1. Copy this file (e.g. to myprovider.py, anywhere importable).
2. Implement gen_object + gen_bg to return PNG bytes. Add dress() only if your model can do the
   one-pass ControlNet render (otherwise the ControlNet backend is warned + skipped).
3. In your config:  models: {image: {provider: "myprovider:MyProvider", model: "...", params: {...}}}
   Any keys under the model config (except `provider`) are passed to __init__ as kwargs.

Return values are plain bytes so nothing here depends on the pipeline.
"""
from . import base   # object_prompt / background_prompt / propose_prompt helpers (optional to use)


class MyProvider:
    def __init__(self, model=None, params=None, **kwargs):
        self.model = model
        self.params = params or {}
        # set up your client / load your model here

    def gen_object(self, name: str, seed: int = 0) -> bytes:
        """Return PNG bytes of ONE object as an RGBA cutout on transparency."""
        raise NotImplementedError("implement gen_object -> PNG bytes (RGBA cutout)")

    def gen_bg(self, name: str, seed: int = 0) -> bytes:
        """Return PNG bytes of a plain background surface."""
        raise NotImplementedError("implement gen_bg -> PNG bytes")

    # Optional — only if your model supports guided/one-pass generation (ControlNet path):
    # def dress(self, sil_png: bytes, prompt: str, scale: float = 0.7, seed: int = 0) -> bytes:
    #     raise NotImplementedError

    # Optional — only if you also want to use this as the pair proposer:
    # def propose(self, illusion, inducer_shape, natural=True, n=12) -> str:
    #     raise NotImplementedError  # return JSON-array text
