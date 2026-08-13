"""Fake provider that ALSO implements dress() -> the ControlNet-capable case."""
import io
from PIL import Image
from tests.fakeprovider import FakeProvider


class FakeProviderCN(FakeProvider):
    def dress(self, sil_png, prompt, scale=0.7, seed=0):
        self.calls.append(("dress", prompt, seed))
        b = io.BytesIO(); Image.new("RGB", (1024, 1024), (10, 20, 30)).save(b, format="PNG")
        return b.getvalue()
