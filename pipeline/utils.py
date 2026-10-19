"""Shared helpers: paths, slug, robust metrics, opaque IDs, manifest, candidate loading.

Paths honor GESTALT_DATA / GESTALT_OUT env vars so tests (and users) can redirect I/O."""
import os, re, hashlib, csv, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("GESTALT_DATA", ROOT / "data"))
SIL = DATA / "silhouettes"           # silhouette store (his generator output lives here too)
SIL_CTRL = DATA / "control"          # paired control silhouettes (illusory shape drawn as an outline)
OBJ = DATA / "assets" / "objects"
BG = DATA / "assets" / "bg"
OUT = pathlib.Path(os.environ.get("GESTALT_OUT", ROOT / "out"))   # benchmark output (stimuli, gt, manifest)


# --- progress bars -----------------------------------------------------------------------------
# Nested tqdm bars on a terminal; one log line per 5% when output is not a TTY (log files, Slurm);
# GESTALT_PROGRESS=0 disables them. Nesting is automatic: each open bar sits one row below its
# parent. GESTALT_PROGRESS_DEPTH lets a driver (tools/run_difficulty_set.py) push child-process
# bars below its own.
import sys
_PROGRESS_ON = os.environ.get("GESTALT_PROGRESS", "1") != "0"
_DEPTH = [int(os.environ.get("GESTALT_PROGRESS_DEPTH", "0"))]


class _LogBar:
    """Non-TTY / no-tqdm bar: prints `desc: 25% (100/400) postfix` at every 5% step."""
    def __init__(self, total, desc, step=0.05):
        self.total, self.desc, self.n, self.post = total, desc, 0, ""
        self.step = max(1, round((total or 0) * step)) if total else 1   # integer item threshold
        self._next = 0
        self._emit()

    def update(self, k=1):
        if k <= 0:
            return
        self.n += k
        if self.n >= self._next or (self.total and self.n >= self.total):
            self._emit()

    def _emit(self):
        if self.total:
            print(f"  {self.desc}: {int(100 * self.n / self.total):3d}% ({self.n}/{self.total}) "
                  f"{self.post}".rstrip(), flush=True)
            self._next = (self.n // self.step + 1) * self.step
        else:                                        # unknown total: a line every 50 items
            print(f"  {self.desc}: {self.n} {self.post}".rstrip(), flush=True)
            self._next = self.n + 50

    def set_postfix_str(self, s): self.post = s
    def set_description(self, d): self.desc = d
    def close(self): pass


class _NoBar:
    def update(self, k=1): pass
    def set_postfix_str(self, s): pass
    def set_description(self, d): pass
    def close(self): pass


def bar(total, desc="", unit="fig", keep=None):
    """A manual progress bar (`update(k)`, `set_postfix_str`, `set_description`, `close`).
    Outermost bar stays on screen; nested bars vanish when closed (`keep` overrides)."""
    if not _PROGRESS_ON:
        return _NoBar()
    depth = _DEPTH[0]
    _DEPTH[0] += 1
    leave = keep if keep is not None else depth == 0
    try:
        from tqdm import tqdm
        if not sys.stderr.isatty():
            raise ImportError
        b = tqdm(total=total, desc=f"  {desc}", unit=unit, position=depth, leave=leave, ncols=100)
    except ImportError:
        b = _LogBar(total, desc)
    close = b.close

    def _close():
        _DEPTH[0] -= 1
        close()
    b.close = _close
    return b


def progress(it, desc="", total=None, unit="fig"):
    """Wrap an iterable in a bar (nested automatically). Drop-in for `for x in it`."""
    if not _PROGRESS_ON:
        yield from it
        return
    total = total if total is not None else (len(it) if hasattr(it, "__len__") else None)
    b = bar(total, desc, unit)
    try:
        for x in it:
            yield x
            b.update(1)
    finally:
        b.close()


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


def busyness(img):
    """Background 'busyness': how many bounded, distractor-like shapes the surface contains.
    Local deviation from a smoothed copy isolates speckles/chips from smooth material texture;
    the score combines the count of speckle patches with the deviation's overall strength.
    Returns (score, n_patches, dev_std) — raw parts kept so a cutoff can be revisited."""
    from PIL import Image
    from scipy import ndimage
    im = img if isinstance(img, Image.Image) else Image.open(img)
    a = np.array(im.convert("L").resize((256, 256))).astype(float)
    dev = np.abs(a - ndimage.uniform_filter(a, 31))
    lab, n = ndimage.label(dev > 25)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        n = int((sizes >= 4).sum())                  # bounded speckles, not lone pixels
    dev_std = float(dev.std())
    return float(n + 20 * dev_std), n, dev_std


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
            cols = ["id", "silhouette", "object", "background", "backend", "mode", "natural", "condition"]
            if len(row) > len(cols):                 # texture column only for multi-texture runs
                cols.append("texture")               # (single-texture manifests stay byte-identical)
            w.writerow(cols)
        w.writerow(row)


TEXTURE_DIRS = {0: "plain", 1: "chips", 2: "objects", 3: "texture"}   # folder per texture level


def stim_path(out, texture, cond, multi):
    """Where a composite stimulus lives: out/comp/<cond>/ for a single-texture run, or
    out/comp/<plain|chips|objects|texture>/<cond>/ when several texture levels are rendered."""
    d = out / "comp"
    if multi:
        d = d / TEXTURE_DIRS[int(texture)]
    return d / cond
