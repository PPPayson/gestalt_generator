"""Silhouette geometry shared by the dresser and the ground-truth writer.
Aspect-preserving; inducers = compact interior blobs (polarity-agnostic)."""
import numpy as np
from PIL import Image
from scipy import ndimage

LONG = 1024                                          # long-side target, aspect preserved


def target_size(size):
    W, H = size
    s = LONG / max(W, H)
    return round(W * s), round(H * s)


def _as_rgba(sil):
    return sil.convert("RGBA") if isinstance(sil, Image.Image) else Image.open(sil).convert("RGBA")


def crop_to_content(sil, pad_frac=0.10):
    """Crop a silhouette to the bounding box of its inducers (+ padding) so the arrangement fills
    the frame instead of floating in dead margin. Returns a PIL RGBA image. Proportions between
    inducers are preserved; only empty border is removed."""
    rgba = _as_rgba(sil)
    lab, n, (TW, TH) = inducer_labels(rgba)
    ys, xs = np.where(lab > 0)
    if len(xs) == 0:
        return rgba
    W, H = rgba.size
    sx, sy = W / TW, H / TH
    x0, x1 = xs.min() * sx, (xs.max() + 1) * sx
    y0, y1 = ys.min() * sy, (ys.max() + 1) * sy
    px, py = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    box = (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))
    return rgba.crop(tuple(int(v) for v in box))


def inducer_labels(sil):
    """Return (labeled array, count, (canvas_w, canvas_h)). Inducers are the compact
    interior blobs that don't touch the border, so black-on-white and white-on-black both work.
    `sil` may be a path or a PIL image."""
    rgba = _as_rgba(sil)
    TW, TH = target_size(rgba.size)
    base = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    base.alpha_composite(rgba)                       # transparent -> black field
    a = np.array(base.convert("L").resize((TW, TH)))
    best = None
    for mask in (a < 128, a >= 128):
        lab, n = ndimage.label(mask)
        keep = []
        for lb in range(1, n + 1):
            ys, xs = np.where(lab == lb)
            border = xs.min() == 0 or ys.min() == 0 or xs.max() == TW - 1 or ys.max() == TH - 1
            if border or len(xs) > 0.25 * TW * TH or len(xs) < 200:
                continue
            keep.append(lb)
        if best is None or len(keep) > len(best[1]):
            best = (lab, keep)
    lab, keep = best
    out = np.zeros_like(lab)
    for i, lb in enumerate(keep, 1):
        out[lab == lb] = i
    return out, len(keep), (TW, TH)


def inducers(sil):
    """List of inducer descriptors {cx, cy, r, area_px} at canvas coords."""
    lab, n, _ = inducer_labels(sil)
    out = []
    for lb in range(1, n + 1):
        ys, xs = np.where(lab == lb)
        area = int(len(xs))
        r = float(np.sqrt(area / np.pi))            # equivalent-disc radius
        out.append({"cx": round(float(xs.mean()), 1), "cy": round(float(ys.mean()), 1),
                    "r": round(r, 1), "area_px": area})
    return out
