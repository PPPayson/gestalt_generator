"""Shared helpers: paths, slug, robust metrics, opaque IDs, manifest, candidate loading.

Paths honor GESTALT_DATA / GESTALT_OUT env vars so tests (and users) can redirect I/O."""
import os, re, hashlib, csv, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("GESTALT_DATA", ROOT / "data"))
SIL = DATA / "silhouettes"           # silhouette store (his generator output lives here too)
OBJ = DATA / "assets" / "objects"
BG = DATA / "assets" / "bg"
OUT = pathlib.Path(os.environ.get("GESTALT_OUT", ROOT / "out"))   # benchmark output (stimuli, gt, manifest)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def sid(illusion, obj, backend, mode):
    h = hashlib.md5(f"{illusion}|{obj}|{backend}|{mode}".encode()).hexdigest()[:10]
    return f"stim_{h}"


def _mask(img):
    a = np.array(img.convert("RGBA"))
    return a, a[..., 3] > 128            # solid pixels only (ignore faint rembg specks)


def luminance(img):
    a, m = _mask(img)
    if m.sum() < 50:
        return -1.0                      # empty cutout -> fails contrast on purpose
    rgb = a[..., :3][m].astype(float)
    return float((0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]).mean() / 255)


def color_contrast(obj, bg):
    """Perceptual color contrast (CIELAB delta-E) between the object and the background.
    Captures colour difference (red chip on green felt) that plain luminance misses."""
    from skimage.color import rgb2lab
    a, m = _mask(obj)
    if m.sum() < 50:
        return 0.0
    oc = a[..., :3][m].mean(0) / 255.0
    bc = np.array(bg.convert("RGB")).reshape(-1, 3).mean(0) / 255.0
    lab_o = rgb2lab(oc.reshape(1, 1, 3))[0, 0]
    lab_b = rgb2lab(bc.reshape(1, 1, 3))[0, 0]
    return float(np.linalg.norm(lab_o - lab_b))


def roundness(img):
    """1.0 = clean disc; square ~0.64; thin/elongated/empty -> low. Robust to stray specks."""
    _, m = _mask(img)
    ys, xs = np.where(m)
    if len(xs) < 200:
        return 0.0
    cx, cy = xs.mean(), ys.mean()
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    r = np.percentile(d, 98)             # ignore a few far outliers
    return float(min(1.5, m.sum() / (np.pi * r * r + 1e-6)))


def load_all_candidates(data=DATA):
    """Merge natural + non-natural proposal files; tag each item with its `natural` flag."""
    out = {}
    for fn, nat in (("candidates_nat.json", True), ("candidates_any.json", False)):
        p = data / fn
        if not p.exists():
            continue
        for ill, items in json.loads(p.read_text()).items():
            for it in items:
                it = dict(it); it["natural"] = nat
                out.setdefault(ill, []).append(it)
    return out


def append_manifest(row, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    p = out / "manifest.csv"; new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "silhouette", "object", "background", "backend", "mode", "natural"])
        w.writerow(row)
