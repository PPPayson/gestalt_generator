"""Silhouettes come from config specs only. Three spec types:
  - kanizsa : built-in registry producer (clean illusory n-gons)
  - gestalt : select N (seeded) from PPPayson/gestalt_generator output, or regenerate via main.generate
  - dropin  : explicit PNG file(s) already in data/silhouettes/
The friend's generator (main.py) writes into data/silhouettes/, which is also our store."""
import json, math, re, random
import numpy as np
from PIL import Image, ImageDraw
from .utils import SIL

POLY = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon"}
GESTALT_RE = re.compile(r"img_([0-9.]+)_(\d+)_(\d+)_([0-9.]+)_([0-9.]+)")


# --- kanizsa registry -------------------------------------------------------
def produce_kanizsa(n_sides=3, size=1024, radius_frac=0.30, inducer_frac=0.13):
    """Black pac-man inducers on white; each wedge = the polygon's interior angle, pointing
    inward so the aligned bite-edges induce an illusory n-gon."""
    name = POLY.get(n_sides, f"{n_sides}gon")
    stem = f"kanizsa_{name}"
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    cx = cy = size / 2
    R = size * radius_frac
    r = size * inducer_frac
    interior = (n_sides - 2) * 180.0 / n_sides          # wedge angle
    for k in range(n_sides):
        phi = -90 + k * 360.0 / n_sides                 # vertex position angle
        ix = cx + R * math.cos(math.radians(phi))
        iy = cy + R * math.sin(math.radians(phi))
        d.ellipse([ix - r, iy - r, ix + r, iy + r], fill=0)          # black disc
        toward = phi + 180                              # direction to centroid
        a0, a1 = toward - interior / 2, toward + interior / 2
        d.pieslice([ix - r, iy - r, ix + r, iy + r], a0, a1, fill=255)  # cut the wedge
    SIL.mkdir(parents=True, exist_ok=True)
    img.save(SIL / f"{stem}.png")
    (SIL / f"{stem}.json").write_text(json.dumps({
        "illusion": stem,
        "inducer_shape": "a round disc with a wedge bitten out (pac-man)",
        "illusory_shape": name,                     # exact GT for registry silhouettes
        "primitive": "circle",
        "n_inducers": n_sides,
    }))
    return [stem]


REGISTRY = {"kanizsa": produce_kanizsa}


# --- gestalt source (friend's silhouettes) ----------------------------------
def parse_gestalt_name(stem):
    """img_{randomness}_{n_vertices}_{n_points}_{edgy}_{rad} -> params dict, or None."""
    m = GESTALT_RE.match(stem)
    if not m:
        return None
    randomness, n_vertices, n_points, edgy, rad = m.groups()
    return {"randomness": float(randomness), "n_vertices": int(n_vertices),
            "n_points": int(n_points), "edgy": float(edgy), "rad": float(rad)}


def _gestalt_sidecar(stem, p):
    """Write meta parsed from the filename so GT/propose have real values."""
    meta = {
        "illusion": stem,
        "inducer_shape": "a disc with a bezier-polygon bite (gestalt inducer)",
        "illusory_shape": "irregular",              # bezier figure has no canonical name
        "primitive": "circle",                      # his draw_shapes uses circles
        "n_inducers": p["n_points"],
        "polygon_vertices": p["n_vertices"],
        "randomness": p["randomness"], "edgy": p["edgy"], "rad": p["rad"],
    }
    (SIL / f"{stem}.json").write_text(json.dumps(meta))


def _blob_centroids(img):
    """Inducer-blob centroids as an (n, 2) array of (row, col), or None if detection is unclean.
    Inducers may be light-on-dark or dark-on-light; keeps only inducer-sized components."""
    from scipy import ndimage
    for fg in (img > 127, img < 128):
        lab, k = ndimage.label(fg)
        if k < 2:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, k + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s > 0.003 * img.size]
        if 3 <= len(keep) <= 8:
            return np.array(ndimage.center_of_mass(fg, lab, keep))
    return None


def rank_score(path, meta):
    """Higher = crisper illusory contour with more regular full-frame coverage. Combines a geometric
    coverage term (inducers evenly spaced in angle, at a uniform radius, centred, spanning all sides)
    with a sharpness weight favouring full-disc inducers (low edgy/rad) over crescents/pinwheels.
    Returns -1.0 if the inducer blobs cannot be cleanly detected (count != n_points)."""
    img = np.array(Image.open(path).convert("L"))
    H, W = img.shape
    c = _blob_centroids(img)
    if c is None or (meta and len(c) != meta["n_points"]):
        return -1.0
    y, x = c[:, 0], c[:, 1]
    Gy, Gx = y.mean(), x.mean()
    centered = np.hypot(Gy - H / 2, Gx - W / 2) / (W / 2)              # 0 = centred in frame
    ang = np.sort(np.arctan2(y - Gy, x - Gx))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    angle_reg = max(0.0, 1 - np.std(gaps) / (2 * np.pi / len(c)))      # 1 = even angular spacing
    r = np.hypot(y - Gy, x - Gx)
    radius_unif = max(0.0, 1 - np.std(r) / (r.mean() + 1e-6))          # 1 = equidistant from centre
    coverage = min((x.max() - x.min()) / W, (y.max() - y.min()) / H)   # both axes spanned -> all sides
    geom = angle_reg * radius_unif * coverage * max(0.0, 1 - centered)
    sharp = 1.0 / (1.0 + 3.0 * meta["edgy"]) * (1.0 if meta["rad"] <= 0.2 else 0.5) if meta else 1.0
    return geom * sharp


def use_gestalt(mode="random", n=20, seed=0, src_dir=None, regenerate=None, inducers=None):
    """Return N gestalt silhouettes, keeping only CLEAN ones (all inducers separate). `mode`:
      "random" (seeded random sample; alias "existing"),
      "rank"   (top-N by rank_score: crispest contour + most regular full-frame coverage),
      "regenerate" (run main.generate first, then random-select).
    `inducers` optionally restricts the inducer count from the filename, e.g. [3, 4]."""
    from pathlib import Path
    from .geometry import separate_inducer_count
    src = Path(src_dir) if src_dir else SIL
    if mode == "regenerate":
        import main                                  # his generator (loop extracted into generate())
        print(f"  gestalt: regenerating into {src} ...")
        main.generate(output_folder=str(src), seed=seed, **(regenerate or {}))
        mode = "random"                             # then select randomly from the fresh grid
    pool = [p.stem for p in src.glob("img_*.png") if parse_gestalt_name(p.stem)]
    if inducers:                                     # restrict inducer count from the filename
        lo, hi = (inducers if isinstance(inducers, (list, tuple)) else (inducers, inducers))
        pool = [s for s in pool if lo <= parse_gestalt_name(s)["n_points"] <= hi]
    pool.sort()
    if not pool:
        print(f"  ! gestalt: no matching img_*.png in {src}"); return []
    SIL.mkdir(parents=True, exist_ok=True)

    def clean(stem):                                 # all inducers separate (none merged with margin)
        return separate_inducer_count(src / f"{stem}.png") == parse_gestalt_name(stem)["n_points"]

    if mode == "rank":                               # top-N by contour sharpness + regular coverage
        scored = []
        for stem in pool:
            if not clean(stem):
                continue
            sc = rank_score(src / f"{stem}.png", parse_gestalt_name(stem))
            if sc >= 0:
                scored.append((sc, stem))
        scored.sort(key=lambda t: (-t[0], t[1]))     # score desc, name tiebreak -> reproducible
        chosen, checked = [s for _, s in scored[:n]], len(pool)
    else:                                            # "random" / "existing": seeded random clean sample
        random.Random(seed).shuffle(pool)
        chosen, checked = [], 0
        for stem in pool:
            if len(chosen) >= n:
                break
            checked += 1
            if clean(stem):
                chosen.append(stem)

    for stem in chosen:                              # materialise: copy into SIL + write sidecar meta
        if src != SIL:
            Image.open(src / f"{stem}.png").save(SIL / f"{stem}.png")
        _gestalt_sidecar(stem, parse_gestalt_name(stem))
    print(f"  gestalt: selected {len(chosen)} clean silhouettes "
          f"(mode={mode}, {checked} checked, seed={seed})")
    return chosen


# --- explicit drop-in -------------------------------------------------------
def use_dropin(files=None, glob=None):
    stems = []
    src = list(files or [])
    if glob:
        src += [p.stem for p in SIL.glob(glob)]
    for f in src:
        stem = f[:-4] if f.endswith(".png") else f
        if not (SIL / f"{stem}.png").exists():
            print(f"  ! dropin: {stem}.png not found"); continue
        sc = SIL / f"{stem}.json"
        if not sc.exists():
            p = parse_gestalt_name(stem)
            if p:
                _gestalt_sidecar(stem, p)
            else:
                sc.write_text(json.dumps(
                    {"illusion": stem, "inducer_shape": "a round disc-shaped object",
                     "illusory_shape": "irregular", "primitive": "circle"}))
        stems.append(stem)
    return stems


# --- dispatch ---------------------------------------------------------------
def build(cfg):
    """Silhouette set = exactly what the config specs declare (no blanket include)."""
    stems = []
    for spec in cfg["silhouettes"]:
        t, params = spec["type"], dict(spec.get("params") or {})
        if t in REGISTRY:
            stems += REGISTRY[t](**params)
        elif t == "gestalt":
            params.setdefault("seed", cfg.get("seed", 0))   # inherit global seed
            stems += use_gestalt(**params)
        elif t == "dropin":
            stems += use_dropin(**params)
        else:
            print(f"  ! unknown silhouette type {t}, skipping")
    seen, uniq = set(), []
    for s in stems:                                 # dedupe, preserve order
        if s not in seen:
            seen.add(s); uniq.append(s)
    metas = [json.loads((SIL / f"{s}.json").read_text()) | {"stem": s} for s in uniq]
    print(f"  silhouettes: {len(metas)} ({', '.join(uniq[:6])}{' ...' if len(uniq) > 6 else ''})")
    return metas
