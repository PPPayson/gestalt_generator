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
GESTALT_RE = re.compile(r"img_([0-9.]+)_(\d+)_(\d+)_([0-9.]+)_([0-9.]+)(?:_([0-9.]+))?$")


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
    """img_{randomness}_{n_vertices}_{n_points}_{edgy}_{rad}[_{closeness}] -> params dict, or None.
    `closeness` (the support-ratio knob) is included only when the suffix is present."""
    m = GESTALT_RE.match(stem)
    if not m:
        return None
    randomness, n_vertices, n_points, edgy, rad, closeness = m.groups()
    p = {"randomness": float(randomness), "n_vertices": int(n_vertices),
         "n_points": int(n_points), "edgy": float(edgy), "rad": float(rad)}
    if closeness is not None:
        p["closeness"] = float(closeness)
    return p


def _gestalt_sidecar(stem, p, src=None):
    """Write meta parsed from the filename so GT/propose have real values. If the generator left a
    sidecar next to the source PNG (support_ratio, closeness), merge it in instead of losing it."""
    meta = {
        "illusion": stem,
        "inducer_shape": "a disc with a bezier-polygon bite (gestalt inducer)",
        "illusory_shape": "irregular",              # bezier figure has no canonical name
        "primitive": "circle",                      # his draw_shapes uses circles
        "n_inducers": p["n_points"],
        "polygon_vertices": p["n_vertices"],
        "randomness": p["randomness"], "edgy": p["edgy"], "rad": p["rad"],
    }
    if p.get("closeness") is not None:
        meta["closeness"] = p["closeness"]
    gen = (src or SIL) / f"{stem}.json"
    if gen.exists():
        try:
            meta = {**json.loads(gen.read_text()), **meta}
        except (json.JSONDecodeError, OSError):
            pass
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


def _inducer_solidity(img):
    """Per-inducer solidity (area / convex-hull area). A proper pac-man has an inward wedge bite
    (concave -> solidity well below 1); a FLAT inducer is a convex half-disc/cap with no mouth
    (solidity ~1). Returns a list aligned with the detected inducer blobs, or [] if none clean."""
    import cv2
    from scipy import ndimage
    for fg in (img > 127, img < 128):
        lab, k = ndimage.label(fg)
        if k < 2:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, k + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s > 0.003 * img.size]
        if 3 <= len(keep) <= 8:
            out = []
            for lb in keep:
                m = (lab == lb).astype(np.uint8)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(cnts, key=cv2.contourArea)
                a = cv2.contourArea(c); h = cv2.contourArea(cv2.convexHull(c))
                out.append(a / h if h > 0 else 1.0)
            return out
    return []


FLAT_SOLIDITY = 0.90          # >= this => inducer is a flat convex cap (no clear inward bite)
MIN_INDUCERS = 4              # hero-like figures wrap the contour with 4+ inducers


def rank_score(path, meta):
    """Higher = a hero-like Kanizsa figure. Encodes the "find more like the README hero" criterion:
      (1) 4+ inducers wrapping the whole contour (fewer reads as a sparse/degenerate figure);
      (2) EVERY inducer is a proper pac-man with a clear inward bite -- no flat convex half-disc caps
          (measured by solidity = area / convex-hull area; a bite makes it concave, solidity < 1);
      (3) inducers spread on ALL sides with no big empty gap between neighbours (even angular spacing,
          bounded max gap), at a uniform radius, centred and filling the frame;
      (4) crisp contour params (low edgy/rad). Deeper bites score higher.
    Returns -1.0 if blobs are unclean, there are <4 inducers, or any inducer is a flat cap."""
    img = np.array(Image.open(path).convert("L"))
    H, W = img.shape
    c = _blob_centroids(img)
    if c is None or (meta and len(c) != meta["n_points"]):
        return -1.0
    n = len(c)
    if n < MIN_INDUCERS:                                               # (1) need enough inducers
        return -1.0
    sol = _inducer_solidity(img)
    if len(sol) == n and max(sol) >= FLAT_SOLIDITY:                    # (2) reject any flat cap inducer
        return -1.0
    bite = float(np.mean([np.clip((0.90 - s) / 0.25, 0.0, 1.0) for s in sol])) if len(sol) == n else 0.5
    y, x = c[:, 0], c[:, 1]
    Gy, Gx = y.mean(), x.mean()
    centered = np.hypot(Gy - H / 2, Gx - W / 2) / (W / 2)              # 0 = centred in frame
    ang = np.arctan2(y - Gy, x - Gx)
    quads = {int((a % (2 * np.pi)) // (np.pi / 2)) for a in ang}       # (3) which 90-deg sides are occupied
    if len(quads) < 4:                                                 # a whole side (top/right/bottom/left) is empty
        return -1.0
    sang = np.sort(ang)
    gaps = np.diff(np.concatenate([sang, [sang[0] + 2 * np.pi]]))
    ideal = 2 * np.pi / n
    angle_reg = max(0.0, 1 - np.std(gaps) / ideal)                    # 1 = even angular spacing
    gap_term = max(0.0, 1 - (gaps.max() / ideal - 1.0))               # 0 if neighbours leave a ~2x gap
    r = np.hypot(y - Gy, x - Gx)
    radius_unif = max(0.0, 1 - np.std(r) / (r.mean() + 1e-6))          # 1 = equidistant from centre
    coverage = min((x.max() - x.min()) / W, (y.max() - y.min()) / H)   # both axes spanned -> all sides
    geom = angle_reg * gap_term * radius_unif * coverage * max(0.0, 1 - centered)
    sharp = 1.0 / (1.0 + 3.0 * meta["edgy"]) * (1.0 if meta["rad"] <= 0.2 else 0.5) if meta else 1.0
    return geom * sharp * (0.4 + 0.6 * bite)                           # reward clear inward bites


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
        rng = random.Random(seed)                    # seeded random tiebreak (reproducible variety)
        scored.sort(key=lambda t: (-round(t[0], 3), rng.random()))
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
        _gestalt_sidecar(stem, parse_gestalt_name(stem), src=src)
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
        p = parse_gestalt_name(stem)
        if p:                                    # (re)write full meta, merging any generation sidecar
            _gestalt_sidecar(stem, p)
        elif not sc.exists():
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
