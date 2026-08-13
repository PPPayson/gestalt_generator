"""Composite backend (local): fill each silhouette inducer region with a validated object cutout.
Polarity-agnostic, aspect-preserving, object clipped EXACTLY to the region, inward-cast shadow."""
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage
from .utils import OBJ, BG, OUT, SIL, SIL_CTRL, slug, sid, append_manifest
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


def _draw_contour(canvas, ring, style="auto", feather=1.6):
    """Draw the illusory shape's outline `ring` onto the composite (positive control) as a soft,
    anti-aliased line. `auto` picks white or black so the line contrasts the background."""
    if not ring.any():
        return canvas
    arr = np.array(canvas)
    if style == "white":
        rgb = (255, 255, 255)
    elif style == "black":
        rgb = (0, 0, 0)
    else:                                              # auto: contrast the local background
        lum = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]) / 255.0
        rgb = (255, 255, 255) if lum[ring].mean() < 0.5 else (0, 0, 0)
    alpha = Image.fromarray((ring * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(feather))
    line = Image.new("RGBA", canvas.size, rgb + (0,))
    line.putalpha(alpha)                               # soft edges instead of hard pixels
    out = canvas.copy()
    out.alpha_composite(line)
    return out


def render(sil_path, o, b, shadow=55, mode="perfect", crop=True, pad_frac=0.10, feather=1.5,
           control=None, contour="auto"):
    """Return (RGB image, canvas_size). `crop` trims the silhouette margin so objects fill the frame;
    `feather` softens the cutout edge. If `control` (a paired control silhouette) is given, the
    illusory shape's drawn outline is added as a positive control."""
    from .geometry import crop_box
    sil_img = sil_path if isinstance(sil_path, Image.Image) else Image.open(sil_path)
    sil_img = sil_img.convert("RGBA")
    box = crop_box(sil_img, pad_frac) if crop else (0, 0, sil_img.width, sil_img.height)
    sil = sil_img.crop(box)
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
    if control is not None:                             # draw the illusory shape's outline (control)
        ctrl = (control if isinstance(control, Image.Image) else Image.open(control)).convert("RGBA")
        ctrl = ctrl.crop(box).resize((TW, TH))          # SAME box -> aligned to the composite
        cl = np.array(ctrl.convert("L"))
        sl = np.array(sil.resize((TW, TH)).convert("L"))
        ring = (cl > 127) & (sl <= 127)                 # pixels drawn in control but not the silhouette
        canvas = _draw_contour(canvas, ring, contour)
    return canvas.convert("RGB"), (TW, TH)


def run(cfg, metas, pairs):
    shadow = cfg["dress"].get("shadow", 55)
    crop = cfg["dress"].get("crop", True)
    feather = cfg["dress"].get("feather", 1.5)
    conditions = cfg["dress"].get("conditions", ["illusory"])
    contour = cfg["dress"].get("contour", "auto")
    out = OUT / "comp"
    for cond in conditions:
        (out / cond).mkdir(parents=True, exist_ok=True)   # out/comp/illusory, out/comp/control
    k = 0
    for meta in metas:
        sp = SIL / f"{meta['stem']}.png"
        cp = SIL_CTRL / f"{meta['stem']}.png"
        for it in pairs.get(meta["stem"], []):
            for mode in cfg["dress"]["modes"]:
                for cond in conditions:                 # illusory (no contour) + control (drawn contour)
                    if cond == "control" and not cp.exists():
                        print(f"  ! no control silhouette for {meta['stem']}; skipping control")
                        continue                        # never emit a 'control' without a drawn contour
                    ctrl = cp if cond == "control" else None
                    img, _ = render(sp, it["object"], it["background"], shadow, mode, crop,
                                    feather=feather, control=ctrl, contour=contour)
                    i = sid(meta["stem"], it["object"], "composite", f"{mode}_{cond}")
                    img.save(out / cond / f"{i}.png")
                    append_manifest([i, meta["stem"], it["object"], it["background"],
                                     "composite", mode, it.get("natural", True), cond])
                    gt.write(i, meta, it, "composite", mode, condition=cond)
                    k += 1
    print(f"  composite: {k} stimuli")
