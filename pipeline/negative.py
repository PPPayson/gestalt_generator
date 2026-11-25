"""NEGATIVE silhouettes: the same inducers, moved so they no longer share a contour.

A negative keeps every inducer blob of a silhouette (same pixels, same bite orientation) and places
each one at another inducer's centre, by a seeded permutation that moves every blob. The parts are
unchanged, only their arrangement is, so no illusory figure is seen. Sweep figures (`rng: shared`)
write their negative in-stream; this module covers every other stored silhouette (legacy pool,
kanizsa, drop-ins, closeness figures)."""
import zlib
from pathlib import Path
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from scipy import ndimage
from .utils import SIL, SIL_NEG

RETRIES = 100


def _blobs(a):
    """Inducer blobs of a binary silhouette: (labels, ids, n_lost). Polarity-agnostic (light or dark
    inducers on the opposite field). `ids` keeps the compact interior components; `n_lost` counts
    components dropped because they touch the border or cover more than a quarter of the canvas
    (a clipped or merged inducer). Slivers under 120 px are dropped without counting."""
    best = (None, [], 0)
    for fg in (a > 127, a <= 127):
        lab, n = ndimage.label(fg)
        ids = []
        lost = 0
        for lb in range(1, n + 1):
            ys, xs = np.where(lab == lb)
            if len(xs) < 120:
                continue
            border = xs.min() == 0 or ys.min() == 0 or xs.max() == a.shape[1] - 1 or ys.max() == a.shape[0] - 1
            if border or len(xs) > 0.25 * a.size:
                lost += 1
                continue
            ids.append(lb)
        if len(ids) > len(best[1]):
            best = (lab, ids, lost)
    return best


def _tag(png):
    """The `negative` text chunk of a stored negative: "sweep", "derangement seed=<n>" or ""."""
    try:
        return Image.open(png).text.get("negative", "")
    except (OSError, AttributeError):
        return ""


def make_negative(sil_png, dst_png, seed=0):
    """Write the negative of `sil_png` to `dst_png`. Blobs are translated (centroid to centroid) by
    a derangement drawn from default_rng([seed, crc32(stem)]); a draw is rejected when two moved
    blobs overlap, touch, or one leaves the canvas, up to RETRIES times (later draws nudge targets
    slightly). Fails (returns False) when fewer than two blobs are usable or when an inducer was
    clipped at the border or merged, so a negative never has fewer inducers than its silhouette."""
    img = Image.open(sil_png).convert("L")
    a = np.array(img)
    lab, ids, n_lost = _blobs(a)
    if len(ids) < 2 or n_lost:
        return False
    fg_val = int(a[lab == ids[0]][0] > 127) * 255           # inducer colour
    bg_val = 255 - fg_val
    cents = np.array(ndimage.center_of_mass(np.ones_like(a), lab, ids))
    rng = np.random.default_rng([int(seed), zlib.crc32(Path(sil_png).name.encode())])
    H, W = a.shape
    r_eq = float(np.sqrt(ndimage.sum(np.ones_like(a), lab, ids).mean() / np.pi))   # typical blob radius
    for t in range(RETRIES):
        perm = rng.permutation(len(ids))
        if np.any(perm == np.arange(len(ids))):            # every blob must move
            continue
        # blobs differ in size (deeper bites), so a pure centroid swap can overlap; after the first
        # failures each target is nudged by a random offset that grows with the retry count
        jitter = rng.uniform(-1, 1, size=(len(ids), 2)) * (0.6 * r_eq * t / RETRIES)
        out = np.full_like(a, bg_val)
        taken = np.zeros_like(a, dtype=bool)
        ok = True
        for i, lb in enumerate(ids):
            ys, xs = np.where(lab == lb)
            dy, dx = np.round(cents[perm[i]] + jitter[i] - cents[i]).astype(int)
            dy = int(np.clip(dy, -ys.min(), H - 1 - ys.max()))     # keep the blob on the canvas
            dx = int(np.clip(dx, -xs.min(), W - 1 - xs.max()))
            ny, nx = ys + dy, xs + dx
            if (dy == 0 and dx == 0) or taken[ny, nx].any():
                ok = False
                break
            out[ny, nx] = fg_val
            placed = np.zeros_like(taken)
            placed[ny, nx] = True
            taken |= ndimage.binary_dilation(placed, iterations=2)   # a 2 px moat: blobs may not touch
        if ok:
            info = PngInfo()
            info.add_text("negative", f"derangement seed={int(seed)}")
            Image.fromarray(out).save(dst_png, pnginfo=info)
            return True
    return False


def ensure_negatives(stems, seed=0, log=print):
    """Write SIL_NEG/<stem>.png for every stem whose negative is missing, built under another seed,
    or older than its silhouette (the store keeps negatives across runs). Negatives tagged "sweep"
    were written in-stream by the sweep and are kept as they are. Returns the stems that have none."""
    SIL_NEG.mkdir(parents=True, exist_ok=True)
    missing = []
    for stem in stems:
        src, dst = SIL / f"{stem}.png", SIL_NEG / f"{stem}.png"
        if dst.exists():
            tag = _tag(dst)
            if tag == "sweep":
                continue
            if tag == f"derangement seed={int(seed)}" and dst.stat().st_mtime >= src.stat().st_mtime:
                continue
        if not make_negative(src, dst, seed):
            missing.append(stem)
    if missing:
        log(f"  ! negative: could not build a negative for {len(missing)} silhouette(s): {missing[:5]}")
    return missing
