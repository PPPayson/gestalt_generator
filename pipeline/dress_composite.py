"""Composite backend (local): fill each silhouette inducer region with a validated object cutout.
Polarity-agnostic, aspect-preserving, object clipped EXACTLY to the region, inward-cast shadow."""
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from .utils import OBJ, BG, OUT, SIL, slug, sid, append_manifest
from .geometry import inducer_labels, crop_to_content
from . import gt


def render(sil_path, o, b, shadow=55, mode="perfect", crop=True, pad_frac=0.10):
    """Return (RGB image, canvas_size). `crop` trims the silhouette to its inducer region so the
    objects fill the frame (bigger); proportions between inducers and the inward shadow are kept."""
    sil = crop_to_content(sil_path, pad_frac) if crop else sil_path
    lab, n, (TW, TH) = inducer_labels(sil)
    obj = Image.open(OBJ / f"{slug(o)}.png").convert("RGBA")
    obj = obj.crop(obj.getbbox())
    canvas = ImageOps.fit(Image.open(BG / f"{slug(b)}.png").convert("RGBA"), (TW, TH))
    cys, cxs = np.where(lab > 0)
    Cx, Cy = cxs.mean(), cys.mean()
    for lb in range(1, n + 1):
        ys, xs = np.where(lab == lb)
        if len(xs) < 200:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        w, h = x1 - x0 + 1, y1 - y0 + 1
        region = ((lab == lb)[y0:y1 + 1, x0:x1 + 1] * 255).astype(np.uint8)
        if mode == "imperfect":
            region = np.array(Image.fromarray(region).filter(ImageFilter.GaussianBlur(2.0)))
        d = int(np.hypot(w, h) * 1.06) + 1              # cover-scale disc to span the box
        big = obj.resize((d, d))
        l, t = (d - w) // 2, (d - h) // 2
        a = np.array(big.crop((l, t, l + w, t + h)))
        a[..., 3] = region                              # clip EXACTLY to the silhouette region
        if shadow > 0:                                  # drop shadow cast INWARD (toward center)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            vx, vy = Cx - cx, Cy - cy
            nrm = (vx * vx + vy * vy) ** 0.5 or 1.0
            dx, dy = int(vx / nrm * 16), int(vy / nrm * 16)
            sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            sh.paste((0, 0, 0, shadow), (x0 + dx, y0 + dy), Image.fromarray(region, "L"))
            canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(7)))
        canvas.alpha_composite(Image.fromarray(a, "RGBA"), (x0, y0))
    return canvas.convert("RGB"), (TW, TH)


def run(cfg, metas, pairs):
    shadow = cfg["dress"].get("shadow", 55)
    crop = cfg["dress"].get("crop", True)
    out = OUT / "comp"; out.mkdir(parents=True, exist_ok=True); k = 0
    for meta in metas:
        sp = SIL / f"{meta['stem']}.png"
        for it in pairs.get(meta["stem"], []):
            for mode in cfg["dress"]["modes"]:
                img, canvas = render(sp, it["object"], it["background"], shadow, mode, crop)
                i = sid(meta["stem"], it["object"], "composite", mode)
                img.save(out / f"{i}.png")
                append_manifest([i, meta["stem"], it["object"], it["background"],
                                 "composite", mode, it.get("natural", True)])
                gt.write(i, meta, it, "composite", mode)
                k += 1
    print(f"  composite: {k} stimuli")
