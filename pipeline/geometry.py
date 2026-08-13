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


def inducer_labels(sil_path):
    """Return (labeled array, count, (canvas_w, canvas_h)). Inducers are the compact
    interior blobs that don't touch the border, so black-on-white and white-on-black both work."""
    rgba = Image.open(sil_path).convert("RGBA")
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


def inducers(sil_path):
    """List of inducer descriptors {cx, cy, r, area_px} at canvas coords."""
    lab, n, _ = inducer_labels(sil_path)
    out = []
    for lb in range(1, n + 1):
        ys, xs = np.where(lab == lb)
        area = int(len(xs))
        r = float(np.sqrt(area / np.pi))            # equivalent-disc radius
        out.append({"cx": round(float(xs.mean()), 1), "cy": round(float(ys.mean()), 1),
                    "r": round(r, 1), "area_px": area})
    return out
