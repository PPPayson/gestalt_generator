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
from scipy import ndimage
from .utils import SIL, SIL_NEG

RETRIES = 100


def _blobs(a):
    """Inducer blobs of a binary silhouette: (labels, ids). Polarity-agnostic (light or dark
    inducers on the opposite field); keeps compact interior components."""
    best = (None, [])
    for fg in (a > 127, a <= 127):
        lab, n = ndimage.label(fg)
        ids = []
        for lb in range(1, n + 1):
            ys, xs = np.where(lab == lb)
            border = xs.min() == 0 or ys.min() == 0 or xs.max() == a.shape[1] - 1 or ys.max() == a.shape[0] - 1
            if border or len(xs) > 0.25 * a.size or len(xs) < 120:
                continue
            ids.append(lb)
        if len(ids) > len(best[1]):
            best = (lab, ids)
    return best


def make_negative(sil_png, dst_png, seed=0):
    """Write the negative of `sil_png` to `dst_png`. Blobs are translated (centroid to centroid) by
    a derangement drawn from default_rng([seed, crc32(stem)]); a draw is rejected when two moved
    blobs overlap or one leaves the canvas, up to RETRIES times (later draws nudge targets slightly). Returns True on success."""
    img = Image.open(sil_png).convert("L")
    a = np.array(img)
    lab, ids = _blobs(a)
    if len(ids) < 2:
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
                ok = False; break
            out[ny, nx] = fg_val; taken[ny, nx] = True
        if ok:
            Image.fromarray(out).save(dst_png)
            return True
    return False


def ensure_negatives(stems, seed=0, log=print):
    """Write SIL_NEG/<stem>.png for every stem that lacks one or whose silhouette is newer than its
    negative (the store keeps negatives across runs). Returns the stems that have none."""
    SIL_NEG.mkdir(parents=True, exist_ok=True)
    missing = []
    for stem in stems:
        src, dst = SIL / f"{stem}.png", SIL_NEG / f"{stem}.png"
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:   # up to date
            continue
        if not make_negative(src, dst, seed):
            missing.append(stem)
    if missing:
        log(f"  ! negative: could not build a negative for {len(missing)} silhouette(s): {missing[:5]}")
    return missing
