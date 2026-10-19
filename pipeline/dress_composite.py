"""Composite backend (local): fill each silhouette inducer region with a validated object cutout.
Polarity-agnostic, aspect-preserving, object clipped EXACTLY to the region, inward-cast shadow."""
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage
from .utils import OBJ, BG, OUT, SIL, SIL_CTRL, slug, sid, append_manifest, stim_path
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


_LUM = np.array([0.2126, 0.7152, 0.0722])


def _contrast_shift(canvas, obj_rgb, contrast, info=None):
    """Camouflage recolor. `contrast` 0 = leave the background's native colors untouched;
    1 = move the background's mean RGB all the way to the OBJECT'S mean color, so figure-ground
    separation (hue AND lightness) vanishes and only the object's own detail/shadow remain.
    Texture detail is preserved by re-centering it around the new mean (compressed only as far as
    0..255 headroom requires, never clipped flat). `info` gets `lum_gap`: the REMAINING
    mean-luminance gap between background and object, in [0,1]."""
    a = np.array(canvas).astype(float)
    rgb = a[..., :3]
    bg_mean = rgb.reshape(-1, 3).mean(0)
    obj_rgb = np.asarray(obj_rgb, dtype=float)
    target = np.clip(bg_mean + contrast * (obj_rgb - bg_mean), 0, 255)
    detail = rgb - bg_mean
    hi = np.percentile(detail, 99, axis=(0, 1))     # robust headroom: ignore extreme outliers
    lo = np.percentile(detail, 1, axis=(0, 1))
    k = 1.0
    for c in range(3):
        if hi[c] > 1e-6:
            k = min(k, (255.0 - target[c]) / hi[c])
        if lo[c] < -1e-6:
            k = min(k, (0.0 - target[c]) / lo[c])
    k = max(0.0, min(1.0, k))
    a[..., :3] = np.clip(target + detail * k, 0, 255)
    got = float(a[..., :3].reshape(-1, 3).mean(0) @ _LUM / 255.0)
    if info is not None:
        info["lum_gap"] = round(abs(got - float(_LUM @ obj_rgb) / 255.0), 4)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def render(sil_path, o, b, shadow=0, mode="perfect", crop=True, pad_frac=0.10, feather=1.5,
           control=None, contour="auto", texture=2, contrast=None, info=None):
    """Return (RGB image, canvas_size). `crop` trims the silhouette margin so objects fill the frame;
    `feather` softens the cutout edge. If `control` (a paired control silhouette) is given, the
    illusory shape's drawn outline is added as a positive control.
    Difficulty knobs: `texture` has 4 discrete levels: 0 = flat black inducers on a flat fill (no
    objects/backgrounds); 1 = solid-color chips (the object's mean color, no texture) on a solid
    canvas (the BG image's mean color); 2 = real object cutouts on the solid canvas; 3 = real
    objects on the real background image (the legacy look).
    `contrast` in [0,1] is a CAMOUFLAGE knob: 0/None = the pair's native colors, 1 = background
    recolored to the object's mean color (figure-ground separation gone); `info` (a dict)
    receives `lum_gap`, the remaining background-object mean-luminance gap."""
    from .geometry import crop_box
    sil_img = sil_path if isinstance(sil_path, Image.Image) else Image.open(sil_path)
    sil_img = sil_img.convert("RGBA")
    box = crop_box(sil_img, pad_frac) if crop else (0, 0, sil_img.width, sil_img.height)
    sil = sil_img.crop(box)
    lab, n, (TW, TH) = inducer_labels(sil)
    texture = int(texture)
    if texture >= 1:
        obj = Image.open(OBJ / f"{slug(o)}.png").convert("RGBA")
        obj = obj.crop(obj.getbbox())
        oa = np.array(obj).astype(float)
        om = oa[..., 3] > 128
        obj_rgb = oa[..., :3][om].mean(0) if om.any() else np.full(3, 255.0)
        if texture == 1:                            # solid-color chip: object's mean color, no texture
            oa[..., :3] = obj_rgb
            obj = Image.fromarray(np.clip(oa, 0, 255).astype(np.uint8), "RGBA")
        bg_img = Image.open(BG / f"{slug(b)}.png").convert("RGBA")
        mean = np.array(bg_img.convert("RGB")).reshape(-1, 3).mean(0)
        if texture <= 2:                            # solid canvas: BG image's mean color
            canvas = Image.new("RGBA", (TW, TH), tuple(int(v) for v in mean) + (255,))
        else:                                       # texture 3: the real background image
            canvas = ImageOps.fit(bg_img, (TW, TH))
        if contrast is not None:
            canvas = _contrast_shift(canvas, obj_rgb, contrast, info)
    else:                                           # texture 0: black inducers on a flat gray fill
        g = 255 if contrast is None else int(round((1.0 - contrast) * 255))   # camouflage: 1 -> black
        canvas = Image.new("RGBA", (TW, TH), (g, g, g, 255))
        if info is not None and contrast is not None:
            info["lum_gap"] = round(g / 255.0, 4)
    cys, cxs = np.where(lab > 0)
    Cx, Cy = cxs.mean(), cys.mean()
    for lb in range(1, n + 1):
        mask = lab == lb
        if mask.sum() < 200:
            continue
        if texture == 0:                            # flat: paint the region itself, feathered black
            reg = Image.fromarray((mask * 255).astype(np.uint8), "L")
            if feather > 0:
                reg = reg.filter(ImageFilter.GaussianBlur(feather))
            ink = Image.new("RGBA", (TW, TH), (0, 0, 0, 255))
            ink.putalpha(reg)
            canvas.alpha_composite(ink)
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
    shadow = cfg["dress"].get("shadow", 0)
    crop = cfg["dress"].get("crop", True)
    feather = cfg["dress"].get("feather", 1.5)
    conditions = cfg["dress"].get("conditions", ["illusory"])
    contour = cfg["dress"].get("contour", "auto")
    texture = cfg["dress"].get("texture", 3)
    multi = isinstance(texture, (list, tuple))      # several texture levels in one run
    textures = list(texture) if multi else [texture]
    contrast = cfg["dress"].get("contrast")
    for cond in conditions:                           # out/comp/[<texture>/]illusory|control
        for tex in textures:
            stim_path(OUT, tex, cond, multi).mkdir(parents=True, exist_ok=True)
    from .utils import progress
    k = 0
    for meta in progress(metas, desc="composite"):
        sp = SIL / f"{meta['stem']}.png"
        cp = SIL_CTRL / f"{meta['stem']}.png"
        for it in pairs.get(meta["stem"], []):
            for mode in cfg["dress"]["modes"]:
                for cond in conditions:                 # illusory (no contour) + control (drawn contour)
                    if cond == "control" and not cp.exists():
                        print(f"  ! no control silhouette for {meta['stem']}; skipping control")
                        continue                        # never emit a 'control' without a drawn contour
                    ctrl = cp if cond == "control" else None
                    for tex in textures:
                        info = {}
                        img, _ = render(sp, it["object"], it["background"], shadow, mode, crop,
                                        feather=feather, control=ctrl, contour=contour,
                                        texture=tex, contrast=contrast, info=info)
                        tag = f"{mode}_{cond}" + (f"_t{tex}" if multi else "")   # scalar: legacy ids
                        i = sid(meta["stem"], it["object"], "composite", tag)
                        img.save(stim_path(OUT, tex, cond, multi) / f"{i}.png")
                        append_manifest([i, meta["stem"], it["object"], it["background"],
                                         "composite", mode, it.get("natural", True), cond, tex])
                        gt.write(i, meta, it, "composite", mode, condition=cond,
                                 texture=tex, contrast=contrast,
                                 lum_gap=info.get("lum_gap"))
                        k += 1
    print(f"  composite: {k} stimuli")
