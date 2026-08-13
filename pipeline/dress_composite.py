"""Composite backend (local): fill each silhouette inducer region with a validated object cutout.
Polarity-agnostic, aspect-preserving, object clipped EXACTLY to the region, inward-cast shadow."""
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage
from .utils import OBJ, BG, OUT, SIL, slug, sid, append_manifest
from .geometry import inducer_labels, crop_to_content
from . import gt


import math, random


def _c2(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, math.dist(a, b) / 2)


def _c3(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy, math.dist((ux, uy), a))


def _inside(c, p, eps=1e-7):
    return c is not None and math.dist((c[0], c[1]), p) <= c[2] + eps


def min_enclosing_circle(xs, ys):
    """Minimum enclosing circle (Welzl) of the region. Because the region is a disc with a bite,
    its MEC IS the inducer's disc (center, radius) exactly -> the object's full rim lands on the arc."""
    pts = np.c_[xs, ys].astype(float)
    if len(pts) > 3:
        try:
            from scipy.spatial import ConvexHull
            pts = pts[ConvexHull(pts).vertices]         # MEC is determined by hull points only
        except Exception:
            pass
    P = [tuple(p) for p in pts]
    random.Random(0).shuffle(P)                         # deterministic (reproducible) + fast
    c = None
    for i, p in enumerate(P):
        if _inside(c, p):
            continue
        c = (p[0], p[1], 0.0)
        for j in range(i):
            if _inside(c, P[j]):
                continue
            c = _c2(p, P[j])
            for k in range(j):
                if _inside(c, P[k]):
                    continue
                c = _c3(p, P[j], P[k]) or c
    return c


def render(sil_path, o, b, shadow=55, mode="perfect", crop=True, pad_frac=0.10, feather=1.5):
    """Return (RGB image, canvas_size). `crop` trims the silhouette to its inducer region so the
    objects fill the frame (bigger); proportions between inducers and the inward shadow are kept.
    `feather` softens the cutout edge (px) so it blends into the scene instead of looking pasted."""
    sil = crop_to_content(sil_path, pad_frac) if crop else sil_path
    lab, n, (TW, TH) = inducer_labels(sil)
    obj = Image.open(OBJ / f"{slug(o)}.png").convert("RGBA")
    obj = obj.crop(obj.getbbox())
    canvas = ImageOps.fit(Image.open(BG / f"{slug(b)}.png").convert("RGBA"), (TW, TH))
    cys, cxs = np.where(lab > 0)
    Cx, Cy = cxs.mean(), cys.mean()
    for lb in range(1, n + 1):
        mask = lab == lb
        if mask.sum() < 200:
            continue
        # the inducer's disc = the minimum enclosing circle of the region (disc-minus-bite)
        ys, xs = np.where(mask)
        cx, cy, R = min_enclosing_circle(xs, ys)
        D = 2 * int(np.ceil(R)) + 2                     # object disc == inducer disc: full rim on the arc
        disc = np.array(obj.resize((D, D)))             # chip fills the disc -> its rim is at radius ~R
        layer = np.zeros((TH, TW, 4), np.uint8)
        x0, y0 = int(round(cx - D / 2)), int(round(cy - D / 2))
        lx0, ly0, lx1, ly1 = max(0, x0), max(0, y0), min(TW, x0 + D), min(TH, y0 + D)
        if lx1 > lx0 and ly1 > ly0:
            layer[ly0:ly1, lx0:lx1] = disc[ly0 - y0:ly1 - y0, lx0 - x0:lx1 - x0]
        # alpha = region (feathered) AND the object's own disc -> keeps the whole chip (rim included,
        # since the disc contains the region), removes only the straight bite, and never shows black.
        reg = Image.fromarray((mask * 255).astype(np.uint8), "L")
        if feather > 0:
            reg = reg.filter(ImageFilter.GaussianBlur(feather))
        layer[..., 3] = (layer[..., 3].astype(int) * np.array(reg) // 255).astype(np.uint8)
        if shadow > 0:                                  # drop shadow cast INWARD (toward center)
            vx, vy = Cx - cx, Cy - cy
            nrm = (vx * vx + vy * vy) ** 0.5 or 1.0
            dx, dy = int(vx / nrm * 16), int(vy / nrm * 16)
            sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            sh.paste((0, 0, 0, shadow), (dx, dy), Image.fromarray((mask * 255).astype(np.uint8), "L"))
            canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(7)))
        canvas.alpha_composite(Image.fromarray(layer, "RGBA"))
    return canvas.convert("RGB"), (TW, TH)


def run(cfg, metas, pairs):
    shadow = cfg["dress"].get("shadow", 55)
    crop = cfg["dress"].get("crop", True)
    feather = cfg["dress"].get("feather", 1.5)
    out = OUT / "comp"; out.mkdir(parents=True, exist_ok=True); k = 0
    for meta in metas:
        sp = SIL / f"{meta['stem']}.png"
        for it in pairs.get(meta["stem"], []):
            for mode in cfg["dress"]["modes"]:
                img, canvas = render(sp, it["object"], it["background"], shadow, mode, crop, feather=feather)
                i = sid(meta["stem"], it["object"], "composite", mode)
                img.save(out / f"{i}.png")
                append_manifest([i, meta["stem"], it["object"], it["background"],
                                 "composite", mode, it.get("natural", True)])
                gt.write(i, meta, it, "composite", mode)
                k += 1
    print(f"  composite: {k} stimuli")
