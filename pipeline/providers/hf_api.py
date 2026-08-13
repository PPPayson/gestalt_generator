"""hf_api — pure HuggingFace, no Modal, no local GPU (DEFAULT provider).

Images  : HF Inference API text->image (gen_bg) + local rembg cutout (gen_object).
Proposer: HF router chat-completions (OpenAI-compatible).
Token   : env HF_TOKEN (or .env). No dress() -> composite works; ControlNet is warned+skipped.

Needs: pip install -r requirements-hf.txt  (requests, rembg, onnxruntime, Pillow, numpy).
Caveat: HF serverless can cold-start (503) or gate some models (PRO) — we retry then error clearly.
"""
import io, os, time, json
from . import base

T2I_URL = "https://api-inference.huggingface.co/models/{model}"
CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


class HFApi:
    def __init__(self, model=None, params=None, token=None, **_ignored):
        self.model = model or base.SDXL_MODEL           # image model, or chat model when used as proposer
        self.params = params or {}
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not self.token:
            raise RuntimeError("hf_api needs a token: `export HF_TOKEN=...` (or put it in .env).")
        self._headers = {"Authorization": f"Bearer {self.token}"}

    # --- image ---
    def _t2i(self, prompt, negative, seed):
        import requests
        payload = {"inputs": prompt,
                   "parameters": {"negative_prompt": negative, "seed": int(seed)}}
        url = T2I_URL.format(model=self.model)
        for attempt in range(6):
            r = requests.post(url, headers=self._headers, json=payload, timeout=180)
            if r.status_code == 200:
                return r.content
            if r.status_code == 503:                    # model loading -> wait and retry
                wait = min(30, 2 ** attempt)
                print(f"    hf_api: {self.model} loading (503), retry in {wait}s")
                time.sleep(wait); continue
            raise RuntimeError(f"hf_api t2i {r.status_code}: {r.text[:200]}")
        raise RuntimeError(f"hf_api t2i: {self.model} unavailable after retries")

    def gen_bg(self, name, seed=0):
        img = self._t2i(base.background_prompt(name), base.background_negative(), seed)
        from PIL import Image
        im = Image.open(io.BytesIO(img)).convert("RGB").resize((1024, 1024))
        b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()

    def gen_object(self, name, seed=0):
        raw = self._t2i(base.object_prompt(name), base.object_negative(), seed)
        from PIL import Image
        from rembg import remove
        im = Image.open(io.BytesIO(raw)).convert("RGB").resize((1024, 1024))
        b = io.BytesIO(); remove(im).save(b, format="PNG"); return b.getvalue()

    # --- proposer ---
    def propose(self, illusion, inducer_shape, natural=True, n=12):
        import requests
        prompt = base.propose_prompt(illusion, inducer_shape, natural, n)
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": self.params.get("max_new_tokens", 900),
                   "temperature": self.params.get("temperature", 0.9)}
        r = requests.post(CHAT_URL, headers=self._headers, json=payload, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"hf_api chat {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]
