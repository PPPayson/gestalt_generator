"""Shared helpers: paths, slug, robust metrics, opaque IDs, manifest, candidate loading."""
import re, hashlib, csv, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SIL = DATA / "silhouettes"           # silhouette store (his generator output lives here too)
OBJ = DATA / "assets" / "objects"
BG = DATA / "assets" / "bg"
OUT = ROOT / "out"                   # our benchmark output (stimuli, gt, manifest)


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
