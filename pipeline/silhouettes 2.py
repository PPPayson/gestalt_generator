"""Silhouettes come from config specs only. Three spec types:
  - kanizsa : built-in registry producer (clean illusory n-gons)
  - gestalt : select N (seeded) from PPPayson/gestalt_generator output, or regenerate via main.generate
  - dropin  : explicit PNG file(s) already in data/silhouettes/
The friend's generator (main.py) writes into data/silhouettes/, which is also our store."""
import json, math, re, random
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


def use_gestalt(mode="existing", n=20, seed=0, src_dir=None, regenerate=None, inducers=None):
    """Return N gestalt silhouettes (seeded random sample). `inducers` optionally restricts the
    number of inducers from the filename, e.g. [3, 4]. mode=regenerate runs main.generate."""
    from pathlib import Path
    src = Path(src_dir) if src_dir else SIL
    if mode == "regenerate":
        import main                                  # his generator (loop extracted into generate())
        grid = regenerate or {}
        print(f"  gestalt: regenerating into {src} ...")
        main.generate(output_folder=str(src), seed=seed, **grid)
    from .geometry import separate_inducer_count
    pool = [p.stem for p in src.glob("img_*.png") if parse_gestalt_name(p.stem)]
    if inducers:                                     # restrict inducer count from the filename
        lo, hi = (inducers if isinstance(inducers, (list, tuple)) else (inducers, inducers))
        pool = [s for s in pool if lo <= parse_gestalt_name(s)["n_points"] <= hi]
    pool.sort()
    random.Random(seed).shuffle(pool)               # reproducible scan order
    if not pool:
        print(f"  ! gestalt: no matching img_*.png in {src}"); return []
    SIL.mkdir(parents=True, exist_ok=True)
    chosen, checked = [], 0
    for stem in pool:                               # keep only CLEAN silhouettes (all inducers separate)
        if len(chosen) >= n:
            break
        checked += 1
        want = parse_gestalt_name(stem)["n_points"]
        if separate_inducer_count(src / f"{stem}.png") != want:
            continue                                # some inducer merged with the margin -> skip
        if src != SIL:
            Image.open(src / f"{stem}.png").save(SIL / f"{stem}.png")
        _gestalt_sidecar(stem, parse_gestalt_name(stem))
        chosen.append(stem)
    print(f"  gestalt: selected {len(chosen)} clean silhouettes ({checked} checked, seed={seed})")
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
