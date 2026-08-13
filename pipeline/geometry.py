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


def crop_box(sil, pad_frac=0.02):
    """The crop box (left, top, right, bottom) that trims the empty border MARGIN, never the
    silhouette — the bbox of everything differing from the corner colour, plus a little padding.
    Returned so the control silhouette can be cropped with the SAME box (keeps the contour aligned)."""
    rgba = _as_rgba(sil)
    arr = np.array(rgba).astype(int)
    corner = arr[0, 0]                              # border/margin colour (+alpha)
    content = np.any(np.abs(arr - corner) > 12, axis=-1)   # anything that isn't the margin
    ys, xs = np.where(content)
    W, H = rgba.size
    if len(xs) == 0:
        return (0, 0, W, H)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    return (int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(W, x1 + px)), int(min(H, y1 + py)))


def crop_to_content(sil, pad_frac=0.02):
    """Crop a silhouette to `crop_box` (removes the empty margin, keeps all inducers)."""
    rgba = _as_rgba(sil)
    return rgba.crop(crop_box(rgba, pad_frac))


def _label_keep(mask, TW, TH, use_border=True):
    """Label `mask` and keep compact blobs (optionally dropping border-touching ones)."""
    lab, n = ndimage.label(mask)
    out, k = np.zeros((TH, TW), int), 0
    for lb in range(1, n + 1):
        ys, xs = np.where(lab == lb)
        border = xs.min() == 0 or ys.min() == 0 or xs.max() == TW - 1 or ys.max() == TH - 1
        if (use_border and border) or len(xs) > 0.25 * TW * TH or len(xs) < 120:
            continue
        k += 1; out[lab == lb] = k
    return out, k


def inducer_labels(sil):
    """Return (labeled array, count, (canvas_w, canvas_h)).
    Friend's silhouettes are inducers on a dark PLOT SQUARE inside a white margin: detect the square
    and take the inducer blobs INSIDE it (keeps inducers that touch the plot edge; excludes the
    margin). Falls back to interior-blob detection for clean silhouettes (e.g. Kanizsa)."""
    rgba = _as_rgba(sil)
    TW, TH = target_size(rgba.size)
    base = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    base.alpha_composite(rgba)                       # transparent -> black field
    a = np.array(base.convert("L").resize((TW, TH)))

    # find a large dark "plot square" that does NOT touch the frame (i.e. sits inside a margin)
    lab_d, nd = ndimage.label(a < 128)
    field = None
    for i in range(1, nd + 1):
        ys, xs = np.where(lab_d == i)
        border = xs.min() == 0 or ys.min() == 0 or xs.max() == TW - 1 or ys.max() == TH - 1
        if not border and len(xs) > 0.12 * TW * TH:
            field = (xs.min(), xs.max(), ys.min(), ys.max()); break
    if field:                                        # inducers = white blobs inside the square bbox
        x0, x1, y0, y1 = field
        reg = np.zeros((TH, TW), bool); reg[y0:y1 + 1, x0:x1 + 1] = True
        out, k = _label_keep((a >= 128) & reg, TW, TH, use_border=False)
        if k:
            return out, k, (TW, TH)

    best = (np.zeros((TH, TW), int), 0)               # fallback: interior blobs, either polarity
    for mask in (a < 128, a >= 128):
        out, k = _label_keep(mask, TW, TH, use_border=True)
        if k > best[1]:
            best = (out, k)
    return best[0], best[1], (TW, TH)


def separate_inducer_count(sil):
    """Number of inducers that are clean, separate shapes NOT touching the border (i.e. not merged
    into the plot margin). A silhouette is 'clean' when this equals its declared inducer count."""
    rgba = _as_rgba(sil)
    TW, TH = target_size(rgba.size)
    base = Image.new("RGBA", rgba.size, (0, 0, 0, 255)); base.alpha_composite(rgba)
    a = np.array(base.convert("L").resize((TW, TH)))
    best = 0
    for mask in (a < 128, a >= 128):
        _, k = _label_keep(mask, TW, TH, use_border=True)
        best = max(best, k)
    return best


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
