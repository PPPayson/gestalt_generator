"""A fake provider simulating someone plugging in their own model via a dotted path.
Deterministic: output is a pure function of (name, seed) so seed-reproducibility is testable.
NO dress() -> exercises the ControlNet-without-dress warning path."""
import io, hashlib
from PIL import Image, ImageDraw


def _png(name, seed, mode, color):
    h = int(hashlib.md5(f"{name}|{seed}".encode()).hexdigest()[:6], 16)
    im = Image.new(mode, (512, 512), (0, 0, 0, 0) if mode == "RGBA" else (h % 256, 0, 0))
    if mode == "RGBA":
        ImageDraw.Draw(im).ellipse([80, 80, 432, 432], fill=color)
    b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()


class FakeProvider:
    """Image + proposer; no dress()."""
    def __init__(self, model=None, params=None, **kwargs):
        self.model = model
        self.params = params or {}
        self.calls = []

    def gen_object(self, name, seed=0):
        self.calls.append(("gen_object", name, seed))
        return _png(name, seed, "RGBA", (255, 255, 255, 255))

    def gen_bg(self, name, seed=0):
        self.calls.append(("gen_bg", name, seed))
        return _png(name, seed, "RGB", None)

    def propose(self, illusion, inducer_shape, natural=True, n=12, seed=0):
        self.calls.append(("propose", illusion, natural, n, seed))
        items = [{"object": "white disc", "background": "black bg", "tone": "light"}]
        import json
        return json.dumps(items)
