"""ControlNet backend (Modal): canny edges of the silhouette guide SDXL to render the scene.
Note: SDXL renders at 1024x1024, so GT geometry for this backend is approximate for non-square
silhouettes; the composite backend is pixel-exact."""
from PIL import Image
from .utils import SIL, OUT, sid, append_manifest
from . import gt


def run(sdxl, cfg, metas, pairs):
    out = OUT / "cn"; out.mkdir(parents=True, exist_ok=True); k = 0
    for meta in metas:
        sil_png = (SIL / f"{meta['stem']}.png").read_bytes()
        for it in pairs.get(meta["stem"], []):
            o, b, nat = it["object"], it["background"], it.get("natural", True)
            prompt = (f"top-down realistic photograph of {o}s arranged on {b}, "
                      f"natural light, sharp focus, high contrast")
            img = sdxl.dress_controlnet.remote(sil_png, prompt, 0.7, seed=k)
            i = sid(meta["stem"], o, "controlnet", "perfect")
            (out / f"{i}.png").write_bytes(img)
            append_manifest([i, meta["stem"], o, b, "controlnet", "perfect", nat])
            gt.write(i, meta, it, "controlnet", "perfect")
            k += 1
    print(f"  controlnet: {k} stimuli")
