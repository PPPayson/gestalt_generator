"""ControlNet backend (Modal): canny edges of the silhouette guide SDXL to render the scene.
Note: SDXL renders at 1024x1024, so GT geometry for this backend is approximate for non-square
silhouettes; the composite backend is pixel-exact."""
import io
from .utils import SIL, OUT, sid, append_manifest
from .geometry import crop_to_content
from . import gt


def run(provider, cfg, metas, pairs):
    out = OUT / "cn"; out.mkdir(parents=True, exist_ok=True); k = 0
    crop = cfg["dress"].get("crop", True)
    for meta in metas:
        sil_img = SIL / f"{meta['stem']}.png"
        if crop:                                          # trim dead margin so the figure fills the frame
            buf = io.BytesIO(); crop_to_content(sil_img).save(buf, format="PNG"); sil_png = buf.getvalue()
        else:
            sil_png = sil_img.read_bytes()
        for it in pairs.get(meta["stem"], []):
            o, b, nat = it["object"], it["background"], it.get("natural", True)
            prompt = (f"top-down realistic photograph of {o}s arranged on {b}, "
                      f"natural light, sharp focus, high contrast")
            img = provider.dress(sil_png, prompt, 0.7, seed=k)
            i = sid(meta["stem"], o, "controlnet", "perfect")
            (out / f"{i}.png").write_bytes(img)
            append_manifest([i, meta["stem"], o, b, "controlnet", "perfect", nat, "illusory"])
            gt.write(i, meta, it, "controlnet", "perfect", condition="illusory")
            k += 1
    print(f"  controlnet: {k} stimuli")
