"""hf_api — pure HuggingFace, no Modal, no local GPU (DEFAULT provider).

Uses huggingface_hub.InferenceClient, which routes to HF Inference Providers (the legacy
api-inference.huggingface.co host is gone). Images: text_to_image (gen_bg) + local rembg cutout
(gen_object). Proposer: chat_completion. Token: env HF_TOKEN (or .env).

Needs: pip install -r requirements-hf.txt  (huggingface_hub, rembg, onnxruntime).
No dress() -> composite works; ControlNet is warned + skipped.
Caveat: image generation routes through an inference provider and may consume HF credits; the model
must be served by a provider (defaults SDXL / FLUX.1-schnell both work)."""
import io, os
from . import base


class HFApi:
    def __init__(self, model=None, params=None, token=None, **_ignored):
        self.model = model or base.SDXL_MODEL           # image model, or chat model when used as proposer
        self.params = params or {}
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not self.token:
            raise RuntimeError("hf_api needs a token: `export HF_TOKEN=...` (or put it in .env).")
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=self.token)

    # --- image ---
    def _t2i(self, prompt, seed):
        try:
            return self.client.text_to_image(prompt, model=self.model, seed=int(seed))
        except TypeError:                               # provider that doesn't accept seed
            return self.client.text_to_image(prompt, model=self.model)

    def gen_bg(self, name, seed=0):
        im = self._t2i(base.background_prompt(name), seed).convert("RGB").resize((1024, 1024))
        b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()

    def gen_object(self, name, seed=0):
        from rembg import remove
        im = self._t2i(base.object_prompt(name), seed).convert("RGB").resize((1024, 1024))
        b = io.BytesIO(); remove(im).save(b, format="PNG"); return b.getvalue()

    # --- proposer ---
    def propose(self, illusion, inducer_shape, natural=True, n=12):
        msgs = [{"role": "user", "content": base.propose_prompt(illusion, inducer_shape, natural, n)}]
        out = self.client.chat_completion(
            messages=msgs, model=self.model,
            max_tokens=self.params.get("max_new_tokens", 900),
            temperature=self.params.get("temperature", 0.9))
        return out.choices[0].message.content
