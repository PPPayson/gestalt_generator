"""Silhouettes come from config specs only. Four spec types:
  - kanizsa : built-in registry producer (clean illusory n-gons)
  - gestalt : select N (seeded) from the stored pool in data/silhouettes/
  - dropin  : explicit PNG file(s) already in data/silhouettes/
  - generate: render figures fresh (pinned rows, sample until N pass the filter, or an exhaustive
              sweep of every parameter combination); see use_generate for every parameter
Every silhouette in the store has a paired control (outline drawn, data/control/) and a negative
(inducers shuffled so they share no contour, data/negative/)."""
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import main
from .geometry import separate_inducer_count
from .negative import ensure_negatives
from .utils import SIL, SIL_CTRL, SIL_NEG, bar, progress

POLY = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon"}
GESTALT_RE = re.compile(r"img_([0-9.]+)_(\d+)_(\d+)_([0-9.]+)_([0-9.]+)(?:_([0-9.]+))?(?:_s(\d+))?$")


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
    """img_{randomness}_{n_vertices}_{n_points}_{edgy}_{rad}[_{closeness}][_s{seed}] -> params dict, or None.
    `closeness` (the support-ratio knob) is included only when the suffix is present."""
    m = GESTALT_RE.match(stem)
    if not m:
        return None
    randomness, n_vertices, n_points, edgy, rad, closeness, seed = m.groups()
    p = {"randomness": float(randomness), "n_vertices": int(n_vertices),
         "n_points": int(n_points), "edgy": float(edgy), "rad": float(rad)}
    if closeness is not None:
        p["closeness"] = float(closeness)
    if seed is not None:
        p["seed"] = int(seed)
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
    if p.get("seed") is not None:
        meta["seed"] = p["seed"]
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

# What "passes" the silhouette filter. Every key can be overridden from a config
# (`filter: {...}` on a gestalt/generate spec) or from tools/pick_difficulty_figures.py.
FILTER_DEFAULTS = {
    "min_inducers": MIN_INDUCERS,   # reject figures with fewer inducers
    "all_sides": True,              # inducers must occupy all 4 quadrants (False: bounded max gap instead)
    "flat_solidity": FLAT_SOLIDITY, # any inducer with solidity >= this is a flat cap -> reject (None: off)
    "max_gap": 1.6,                 # all_sides=False: reject if an angular gap > max_gap x even spacing
    "max_flat_frac": 0.0,           # fraction of inducers allowed to be flat caps (0 = none; 0.2 lets
                                    # one of 5-6 sit on a straight contour stretch)
}


def rank_score(path, meta, **overrides):
    """Higher = a hero-like Kanizsa figure. Thresholds default to FILTER_DEFAULTS; pass them as
    kwargs (or a `filter` dict in a config) to loosen or tighten what passes. With all_sides=False
    the 4-quadrant test is replaced by a bounded max angular gap (3-inducer figures can never
    occupy all four sides). Encodes the "find more like the README hero" criterion:
      (1) 4+ inducers wrapping the whole contour (fewer reads as a sparse/degenerate figure);
      (2) EVERY inducer is a proper pac-man with a clear inward bite -- no flat convex half-disc caps
          (measured by solidity = area / convex-hull area; a bite makes it concave, solidity < 1);
      (3) inducers spread on ALL sides with no big empty gap between neighbours (even angular spacing,
          bounded max gap), at a uniform radius, centred and filling the frame;
      (4) crisp contour params (low edgy/rad). Deeper bites score higher.
    Returns -1.0 if blobs are unclean, there are <4 inducers, or any inducer is a flat cap."""
    unknown = set(overrides) - set(FILTER_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown filter keys {sorted(unknown)}; known: {sorted(FILTER_DEFAULTS)}")
    f = {**FILTER_DEFAULTS, **overrides}
    img = np.array(Image.open(path).convert("L"))
    H, W = img.shape
    c = _blob_centroids(img)
    if c is None or (meta and len(c) != meta["n_points"]):
        return -1.0
    n = len(c)
    if n < f["min_inducers"]:                                          # (1) need enough inducers
        return -1.0
    sol = _inducer_solidity(img)
    if f["flat_solidity"] is not None and len(sol) == n:                # (2) flat-cap inducers
        n_flat = sum(s_ >= f["flat_solidity"] for s_ in sol)
        if n_flat > f["max_flat_frac"] * n:                            # default 0: any flat cap rejects
            return -1.0
    bite = float(np.mean([np.clip((0.90 - s) / 0.25, 0.0, 1.0) for s in sol])) if len(sol) == n else 0.5
    y, x = c[:, 0], c[:, 1]
    Gy, Gx = y.mean(), x.mean()
    centered = np.hypot(Gy - H / 2, Gx - W / 2) / (W / 2)              # 0 = centred in frame
    ang = np.arctan2(y - Gy, x - Gx)
    quads = {int((a % (2 * np.pi)) // (np.pi / 2)) for a in ang}       # (3) which 90-deg sides are occupied
    sang = np.sort(ang)
    gaps = np.diff(np.concatenate([sang, [sang[0] + 2 * np.pi]]))
    ideal = 2 * np.pi / n
    if f["all_sides"]:
        if len(quads) < 4:                                             # a whole side (top/right/bottom/left) is empty
            return -1.0
    elif gaps.max() > f["max_gap"] * ideal:                            # loosened: no oversized gap between neighbours
        return -1.0
    angle_reg = max(0.0, 1 - np.std(gaps) / ideal)                    # 1 = even angular spacing
    gap_term = max(0.0, 1 - (gaps.max() / ideal - 1.0))               # 0 if neighbours leave a ~2x gap
    r = np.hypot(y - Gy, x - Gx)
    radius_unif = max(0.0, 1 - np.std(r) / (r.mean() + 1e-6))          # 1 = equidistant from centre
    coverage = min((x.max() - x.min()) / W, (y.max() - y.min()) / H)   # both axes spanned -> all sides
    geom = angle_reg * gap_term * radius_unif * coverage * max(0.0, 1 - centered)
    sharp = 1.0 / (1.0 + 3.0 * meta["edgy"]) * (1.0 if meta["rad"] <= 0.2 else 0.5) if meta else 1.0
    return geom * sharp * (0.4 + 0.6 * bite)                           # reward clear inward bites


def use_gestalt(mode="random", n=20, seed=0, src_dir=None, inducers=None, filter=None):
    """Return N gestalt silhouettes, keeping only CLEAN ones (all inducers separate). `mode`:
      "random" (seeded random sample; alias "existing"),
      "rank"   (top-N by rank_score: crispest contour + most regular full-frame coverage).
    `inducers` optionally restricts the inducer count from the filename, e.g. [3, 4]; `filter`
    overrides FILTER_DEFAULTS for mode rank."""
    src = Path(src_dir) if src_dir else SIL
    # legacy pool only: closeness-suffixed figures (type: generate) would otherwise shift the
    # seeded draw of every existing config
    pool = [p.stem for p in src.glob("img_*.png")
            if parse_gestalt_name(p.stem) and "closeness" not in parse_gestalt_name(p.stem)]
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
            sc = rank_score(src / f"{stem}.png", parse_gestalt_name(stem), **(filter or {}))
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


# --- generate: render figures fresh ------------------------------------------------------
# Config names for the five figure knobs -> the internal names main.py / filenames use.
KNOBS = {"polygon_randomness": "randomness", "n_vertices": "n_vertices", "n_inducers": "n_points",
         "corner_sharpness": "edgy", "side_bulge": "rad"}
# The SHARED value grid: the legacy silhouette pool, the closure sweep and `type: generate` all draw
# from these lists unless a config narrows them (`space`).
GRID_DEFAULT = {"randomness": [0, 0.4, 0.8, 1], "n_vertices": [3, 4, 5, 6, 7, 8],
                "n_points": [3, 4, 5, 6], "edgy": [0.1, 0.5, 1], "rad": [0.1, 0.2, 0.4]}


KNOB_MIN = {"n_vertices": 3, "n_points": 2}      # a polygon needs 3 corners; a shuffle needs 2 discs


def _space(space):
    """Normalise a config `space` (new or internal knob names; list or {min, max} per knob) to
    internal names. Missing knobs -> GRID_DEFAULT. Ranges stay dicts for the sampler."""
    out = dict(GRID_DEFAULT)
    for k, v in (space or {}).items():
        key = KNOBS.get(k, k)
        if key not in GRID_DEFAULT:
            raise ValueError(f"unknown space knob {k!r}; known: {sorted(KNOBS)}")
        if isinstance(v, dict):
            if set(v) != {"min", "max"}:
                raise ValueError(f"space.{k}: a range is {{min, max}}, got {v}")
            out[key] = {"min": float(v["min"]), "max": float(v["max"])}
            if out[key]["min"] > out[key]["max"]:
                raise ValueError(f"space.{k}: min {v['min']} is above max {v['max']}")
            lo, hi = out[key]["min"], out[key]["max"]
        else:
            vals = list(v) if isinstance(v, (list, tuple)) else [v]
            if not vals or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vals):
                raise ValueError(f"space.{k}: needs a non-empty list of numbers, got {v!r}")
            out[key] = vals
            lo, hi = min(vals), max(vals)
        floor = KNOB_MIN.get(key)
        if floor is not None and lo < floor:
            raise ValueError(f"space.{k}: values must be at least {floor}, got {lo}")
        if key in ("randomness", "edgy", "rad") and (lo < 0 or hi > 1):
            raise ValueError(f"space.{k}: values must lie in [0, 1], got {lo}..{hi}")
    return out


def _lists_only(g, what):
    bad = [k for k, v in g.items() if isinstance(v, dict)]
    if bad:
        raise ValueError(f"{what} needs lists in space, not ranges: {bad}")


def _draw_knob(v, rng, integer):
    """One value for a knob: pick from a list or sample a {min, max} range."""
    if isinstance(v, dict):
        x = rng.uniform(v["min"], v["max"])
        return int(round(x)) if integer else round(float(x), 3)
    return v[rng.integers(len(v))]


def _quota_counts(n, quota, n_points_list=(3, 4, 5, 6)):
    """{n_points: count}. quota None/"uniform" -> equal shares over `n_points_list`. Non-integer values
    (or values summing to 1) are SHARES of n, renormalised, largest remainder gets the rounding;
    whole numbers are absolute counts and must sum to n."""
    if quota is None or quota == "uniform":
        quota = {int(k): 1.0 / len(n_points_list) for k in n_points_list}
    q = {int(k): float(v) for k, v in quota.items()}
    if not q or sum(q.values()) <= 0:
        raise ValueError(f"inducer_count_mix must request at least one figure: {quota}")
    fractions = any(v != int(v) for v in q.values()) or abs(sum(q.values()) - 1.0) < 1e-6
    if fractions:                                                   # shares -> counts
        tot = sum(q.values())
        exact = {k: n * v / tot for k, v in q.items()}
        counts = {k: int(x) for k, x in exact.items()}
        for k in sorted(q, key=lambda k: -(exact[k] - counts[k]))[: n - sum(counts.values())]:
            counts[k] += 1                                              # largest-remainder rounding
    else:
        counts = {k: int(v) for k, v in q.items()}
        if n and sum(counts.values()) != n:
            raise ValueError(f"inducer_count_mix counts {counts} sum to {sum(counts.values())}, not {n}")
    if sum(counts.values()) == 0:
        raise ValueError(f"inducer_count_mix {quota} yields no figures for n={n}")
    return counts


def _filter(quality_filter):
    """quality_filter: True -> FILTER_DEFAULTS, False -> None (keep every clean figure), dict -> overrides."""
    if quality_filter is True or quality_filter is None:
        return dict(FILTER_DEFAULTS)
    if quality_filter is False:
        return None
    unknown = set(quality_filter) - set(FILTER_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown filter keys {sorted(unknown)}; known: {sorted(FILTER_DEFAULTS)}")
    return {**FILTER_DEFAULTS, **quality_filter}


def _eval_candidate(job):
    """Worker: generate one candidate at every closeness in `levels`; the FIRST level is judged
    with the filter (None = no filter), the rest only need to be feasible + clean. -> (score, fig) or None."""
    fig, levels, filt, geom = job
    r, nv, np_, e, rad, seed = fig
    work = tempfile.mkdtemp(prefix="gestalt_sample_")
    try:
        score = 0.0
        for i, c in enumerate(levels):
            sd, cd = f"{work}/s{i}", f"{work}/c{i}"
            os.makedirs(sd); os.makedirs(cd)
            stem = main.generate_one(r, nv, np_, e, rad, closeness=c, sil_dir=sd, ctrl_dir=cd,
                                     seed=seed, tag_seed=True, **geom)
            if stem is None or separate_inducer_count(f"{sd}/{stem}.png") != np_:
                return None
            if i == 0 and filt is not None:
                score = rank_score(f"{sd}/{stem}.png", parse_gestalt_name(stem), **filt)
                if score < 0:
                    return None
        return (score, fig)
    finally:
        shutil.rmtree(work, ignore_errors=True)


class _Serial:
    """Pool stand-in for jobs == 1: runs in-process (no multiprocessing at all)."""
    def imap_unordered(self, fn, items, chunksize=1):
        return map(fn, items)
    def __enter__(self): return self
    def __exit__(self, *a): return False


def sample_figures(n, closeness=0.5, quota=None, grid=None, filter=None, seed=0,
                   also_closeness=(), max_seeds=200, jobs=None, log=print, geom=None):
    """Keep drawing figures until `n` PASS the filter, in the requested inducer-count mix.
    `grid` is a normalised space (lists or {min,max} ranges per knob, internal names); each round
    enumerates the list knobs under seed, seed+1, ... and samples the range knobs once per cell.
    Each candidate is rendered at `closeness` (judged by rank_score with `filter`; filter None keeps
    every clean figure) and, if `also_closeness` is given, at those levels too (must be feasible +
    clean there: matched multi-level sets). Stops when every quota is met or `max_seeds` rounds are
    exhausted (then returns what it has, with a warning). Returns
    [[randomness, n_vertices, n_points, edgy, rad, seed], ...] best-first per inducer count.
    Parallel over `jobs` workers (default: all cores but one)."""
    g = _space(grid)                                   # partial / old-name grids are fine
    filt = filter
    if isinstance(g["n_points"], dict):
        raise ValueError("space.n_inducers must be a list (the inducer-count mix needs discrete counts)")
    counts = g["n_points"]
    if quota in (None, "uniform") and filt is not None and filt.get("min_inducers"):
        # the default mix only covers counts the filter can pass (3-inducer figures never pass
        # min_inducers 4 / all_sides, and would stall the draw for max_seeds rounds)
        counts = [k for k in counts if k >= filt["min_inducers"]] or counts
    need = _quota_counts(n, quota, counts)
    levels = [closeness] + list(also_closeness)
    have = {k: [] for k in need}
    jobs = jobs or max(1, mp.cpu_count() - 1)
    geom = geom or {}
    done = lambda: all(len(have[k]) >= need[k] for k in need)
    log(f"  sample: need {need} at closeness={closeness} (+{list(also_closeness)}), filter={filt}")
    passed = lambda: sum(min(len(have[k]), need[k]) for k in need)
    short_str = lambda: " ".join(f"{k}:{min(len(have[k]), need[k])}/{need[k]}" for k in need)
    top = bar(n, f"figures passed (closeness={closeness})")     # persists across seed rounds
    top.set_postfix_str(short_str())
    lists = {k: (v if isinstance(v, list) else [None]) for k, v in g.items()}   # None = range, drawn per cell
    # NOTE: the default start method (spawn on macOS/Windows) re-imports the caller's __main__ in
    # every worker, so entry scripts must be guarded (run.py is); fork is not used because forking
    # a process with live BLAS/tqdm threads deadlocks on macOS. jobs == 1 never forks or spawns.
    with (mp.Pool(jobs) if jobs > 1 else _Serial()) as pool:
        for rnd in range(max_seeds):
            s = seed + rnd
            short = [k for k in need if len(have[k]) < need[k]]
            combos = []
            for j, (r, nv, np_, e, rad) in enumerate(itertools.product(
                    lists["randomness"], lists["n_vertices"], short, lists["edgy"], lists["rad"])):
                cell_rng = np.random.default_rng([s, j])
                r = _draw_knob(g["randomness"], cell_rng, False) if r is None else r
                nv = _draw_knob(g["n_vertices"], cell_rng, True) if nv is None else nv
                e = _draw_knob(g["edgy"], cell_rng, False) if e is None else e
                rad = _draw_knob(g["rad"], cell_rng, False) if rad is None else rad
                combos.append([r, nv, np_, e, rad, s])
            before = passed()
            for res in progress(pool.imap_unordered(_eval_candidate, [(f, levels, filt, geom) for f in combos]),
                                desc=f"round {rnd + 1} (seed {s})", total=len(combos)):
                if res:
                    have[res[1][2]].append(res)
            # one figure per polygon: edgy/rad cells of the same (randomness, n_vertices, n_points,
            # seed) are near-identical figures, keep only the best-scoring one
            for k in have:
                best = {}
                for sc, fig in have[k]:
                    key = (fig[0], fig[1], fig[2], fig[5])
                    if key not in best or (sc, fig) > best[key]:     # ties broken by the row itself,
                        best[key] = (sc, fig)                        # not by worker arrival order
                have[k] = list(best.values())
            top.set_postfix_str(short_str()); top.update(passed() - before)
            if done():
                break
    top.close()
    out = []
    for k in need:
        got = sorted(have[k], key=lambda t: (-t[0], t[1]))[: need[k]]
        if len(got) < need[k]:
            log(f"  ! sample: only {len(got)}/{need[k]} figures with {k} inducers after {max_seeds} seeds")
        out += [fig for _, fig in got]
    log(f"  sample: {len(out)}/{n} figures passed")
    return out


def _sweep_combos(g, inducer_radius, inducer_scale):
    """Every combination of the space lists, loop order randomness > n_vertices > n_inducers > edgy
    > rad > radius. `inducer_radius`: "derived" -> one radius per cell, 3 * inducer_scale /
    n_inducers; a list -> every listed radius not larger than the derived one (bigger discs would
    overlap at that count). Returns [(randomness, n_vertices, n_points, edgy, rad, radius), ...]."""
    _lists_only(g, "n_figures: all")
    out = []
    for r in g["randomness"]:
        for nv in g["n_vertices"]:
            for np_ in g["n_points"]:
                derived = inducer_scale * 3.0 / np_
                radii = [derived] if inducer_radius == "derived" else \
                        [float(x) for x in inducer_radius if float(x) <= derived + 1e-9]
                if not radii:
                    raise ValueError(f"inducer_radius {inducer_radius}: every radius is above the derived "
                                     f"{derived:g} px for {np_} inducers (discs would overlap)")
                for e in g["edgy"]:
                    for rad in g["rad"]:
                        for radius in radii:
                            out.append((r, int(nv), int(np_), e, rad, radius))
    if not out:
        raise ValueError("n_figures: all has no combinations to render (an empty space list?)")
    return out


def _sweep_cell(job):
    """Worker: render one sweep figure (illusory, control, negative) at an explicit disc radius.
    rng "shared": ONE stream `default_rng(seed + idx)` feeds polygon, placement and negative, with up
    to `attempts` re-draws of the whole figure when placement or the negative fails.
    rng "fresh": polygon and placement each restart from `seed + idx` (main.py's convention), the
    negative draws from a fresh stream of the same seed. Returns (idx, stem) or (idx, None)."""
    idx, combo, seed, rng_mode, polygon_shape, output_shape, contour_px, attempts, dirs = job
    r, nv, np_, e, rad, radius = combo
    sil_dir, ctrl_dir, neg_dir = dirs
    figure_seed = seed + idx
    rng = np.random.default_rng(figure_seed) if rng_mode == "shared" else None
    pos = neg = ring = None
    for _ in range(attempts if rng_mode == "shared" else 1):
        kw = {"rng": rng} if rng is not None else {"seed": figure_seed}
        a = main.get_random_points(n=nv, scale=0.8, randomness=r, **kw)
        x, y, _ = main.get_bezier_curve(a, rad=rad, edgy=e)
        x = np.int16(polygon_shape[0] * x)
        y = np.int16(polygon_shape[1] * y)
        coords = np.stack([x, y], axis=1)
        mask, coords = main.padded_polygon_mask(coords, polygon_shape, output_shape)
        sampled = main.sample_points(coords, n=np_, mindist=radius * 2.5, **kw)
        if sampled is None:
            continue
        out = main.draw_shapes(sampled, mindist=radius, shape=output_shape, shape_type="circle")
        out[mask] = 0                                               # carve the illusory polygon
        nrng = rng if rng is not None else np.random.default_rng(figure_seed)
        neg = main.shuffle_discs(out, sampled, mindist=radius, shape=output_shape, shape_type="circle", rng=nrng)
        if neg is None:
            continue
        ring = mask - main.erode_mask(mask, contour_px)                # polygon outline
        pos = out
        break
    if pos is None:
        return idx, None
    stem = f"gen_s{seed}_{idx:05d}"
    save = lambda d, arr: Image.fromarray((np.asarray(arr, dtype=np.uint8) * 255)).save(Path(d) / f"{stem}.png")
    save(sil_dir, pos)
    save(ctrl_dir, np.clip(pos + ring, 0, 1))
    save(neg_dir, neg)
    meta = {"illusion": stem, "inducer_shape": "a disc with a bezier-polygon bite (gestalt inducer)",
            "illusory_shape": "irregular", "primitive": "circle", "n_inducers": np_,
            "polygon_vertices": nv, "randomness": r, "edgy": e, "rad": rad,
            "inducer_radius": radius, "seed": figure_seed, "sweep_index": idx, "rng": rng_mode,
            "support_ratio": round(main.support_ratio(main.draw_shapes(sampled, mindist=radius, shape=output_shape), mask, contour_px), 4)}
    (Path(sil_dir) / f"{stem}.json").write_text(json.dumps(meta))
    return idx, stem


def render_sweep(seed=0, space=None, figures_per_combination=1, inducer_radius="derived",
                 inducer_scale=125, polygon_size=(700, 700), output_shape=(1024, 1024), contour_px=5,
                 rng="fresh", attempts=100, indices=None, jobs=None, dirs=None, log=print):
    """`n_figures: all`: render EVERY combination of the space lists (x `figures_per_combination`
    random draws each) at an explicit disc radius, with illusory + control + negative written side by
    side. Figure idx -> combination idx % len(combos), seed `seed + idx`. `indices` restricts to a
    subset of idx (partial sweeps, tests). Returns the stems in idx order."""
    g = _space(space)
    combos = _sweep_combos(g, inducer_radius, inducer_scale)
    total = len(combos) * int(figures_per_combination)
    idxs = list(range(total)) if indices is None else [int(i) for i in indices]
    bad = [i for i in idxs if not 0 <= i < total]
    if bad:
        raise ValueError(f"indices {bad[:5]} outside the sweep (0 .. {total - 1})")
    dirs = dirs or (SIL, SIL_CTRL, SIL_NEG)
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    jobs = jobs or max(1, mp.cpu_count() - 1)
    polygon_shape = list(polygon_size)
    output_shape = list(output_shape)
    jobs_list = [(i, combos[i % len(combos)], int(seed), rng, polygon_shape, output_shape, int(contour_px),
                  int(attempts), tuple(str(d) for d in dirs)) for i in idxs]
    log(f"  sweep: {len(combos)} combinations x {figures_per_combination} = {total} figures"
        + (f" (rendering {len(idxs)})" if indices is not None else "") + f", rng={rng}")
    got = {}
    with (mp.Pool(jobs) if jobs > 1 and len(jobs_list) > 1 else _Serial()) as pool:
        for i, stem in progress(pool.imap_unordered(_sweep_cell, jobs_list, chunksize=4),
                                desc="sweep", total=len(jobs_list)):
            if stem is None:
                log(f"  ! sweep: figure {i} failed placement after {attempts} attempts; skipping")
            else:
                got[i] = stem
    return [got[i] for i in idxs if i in got]


def use_generate(figures=None, seed=None, space=None, n_figures=None, figures_per_combination=1,
                 inducer_count_mix="uniform", inducer_closeness=0.5, also_closeness=(),
                 quality_filter=True, max_seeds=200, polygon_size=(700, 700), inducer_scale=125,
                 inducer_radius="derived", contour_px=5, rng="fresh", jobs=None, indices=None):
    """`type: generate`: render figures fresh into the store. Three ways to say WHICH figures:
      figures            pinned rows [[randomness, n_vertices, n_points, edgy, rad, seed], ...]
      n_figures: <int>   draw from `space` until that many pass `quality_filter`, in the
                         `inducer_count_mix` (default equal shares over space.n_inducers)
      n_figures: all     every combination of the space lists, `figures_per_combination` draws each,
                         no filter (an exhaustive sweep)
    HOW they are drawn: `polygon_size` (px, inside a 1024 canvas), `inducer_scale` (disc radius =
    3 * inducer_scale / n_inducers when `inducer_radius` is "derived"; a list of px radii sweeps disc
    size as its own factor and is honoured for `n_figures: all`), `inducer_closeness` (bite depth,
    0 far .. 1 nearly touching; used when the radius is derived), `contour_px` (control outline),
    `rng` ("fresh": restart from the seed per step, so the same seed gives the same polygon at every
    closeness; "shared": one stream per figure for polygon, placement and negative).
    `indices` (with n_figures: all) renders only those sweep positions, e.g. a few figures to dress;
    `max_seeds` caps how many seeds a numeric n_figures tries per row before giving up."""
    space = dict(space or {})
    if "inducer_radius" in space:                      # the radius knob may be written inside `space`
        inducer_radius = space.pop("inducer_radius")
    if rng not in ("fresh", "shared"):
        raise ValueError(f"rng must be 'fresh' or 'shared', got {rng!r}")
    if isinstance(inducer_radius, (int, float)) and not isinstance(inducer_radius, bool):
        inducer_radius = [inducer_radius]
    if inducer_radius != "derived" and not (isinstance(inducer_radius, (list, tuple)) and inducer_radius
                                            and all(isinstance(x, (int, float)) for x in inducer_radius)):
        raise ValueError(f"inducer_radius must be 'derived' or a list of px radii, got {inducer_radius!r}")
    ignored = (["figures", "inducer_closeness", "also_closeness", "inducer_count_mix", "max_seeds"]
               if n_figures == "all" else ["figures_per_combination", "rng", "indices"])
    given = {"figures": figures is not None, "inducer_closeness": inducer_closeness != 0.5,
             "also_closeness": bool(also_closeness), "inducer_count_mix": inducer_count_mix != "uniform",
             "max_seeds": max_seeds != 200, "figures_per_combination": figures_per_combination != 1,
             "rng": rng != "fresh", "indices": indices is not None}
    for name in ignored:
        if given[name]:
            print(f"  ! generate: `{name}` has no effect with n_figures: {n_figures}")
    seed = 0 if seed is None else int(seed)
    geom = {"polygon_shape": tuple(polygon_size), "basedist": inducer_scale, "contour_px": contour_px}
    if n_figures == "all":
        if quality_filter is not False:
            print("  ! generate: n_figures: all renders every combination; quality_filter is not applied")
        return render_sweep(seed=seed, space=space, figures_per_combination=figures_per_combination,
                            inducer_radius=inducer_radius, inducer_scale=inducer_scale,
                            polygon_size=polygon_size, contour_px=contour_px, rng=rng,
                            indices=indices, jobs=jobs)
    if inducer_radius != "derived":
        raise ValueError("a list of inducer_radius is only used with n_figures: all; "
                         "numeric n_figures and pinned figures size discs by inducer_closeness")
    if figures is None:
        assert n_figures, "type: generate needs `figures` or `n_figures`"
        figures = sample_figures(int(n_figures), closeness=inducer_closeness, quota=inducer_count_mix,
                                 grid=_space(space), filter=_filter(quality_filter), seed=seed,
                                 also_closeness=also_closeness, max_seeds=max_seeds, jobs=jobs, geom=geom)
    return _render_figures(figures, inducer_closeness, seed, geom=geom)


def _render_figures(figures, closeness, seed=None, geom=None):
    """Render figures fresh via main.generate_one at ONE `closeness` into the store (SIL + SIL_CTRL).
    `figures` = [[randomness, n_vertices, n_points, edgy, rad, (seed)], ...]; a per-figure seed
    overrides `seed`. The same figures + seeds at another closeness give a matched figure (same
    polygon rotation, resized bites). Unclean or infeasible figures are skipped with a warning."""
    SIL.mkdir(parents=True, exist_ok=True); SIL_CTRL.mkdir(parents=True, exist_ok=True)
    stems = []
    for fig in progress(figures, desc=f"generate closeness={closeness}"):
        randomness, n_vertices, n_points, edgy, rad = fig[:5]
        s = fig[5] if len(fig) > 5 else seed
        stem = main.generate_one(randomness, int(n_vertices), int(n_points), edgy, rad,
                                 closeness=closeness, sil_dir=str(SIL), ctrl_dir=str(SIL_CTRL),
                                 seed=s, tag_seed=True, **(geom or {}))
        if stem is None:
            print(f"  ! generate: placement failed for {fig} at closeness={closeness}; skipping"); continue
        if separate_inducer_count(SIL / f"{stem}.png") != int(n_points):
            print(f"  ! generate: {stem} not clean; skipping"); continue
        _gestalt_sidecar(stem, parse_gestalt_name(stem))
        stems.append(stem)
    print(f"  generate: {len(stems)}/{len(figures)} figures at closeness={closeness}")
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
        elif t == "generate":
            params.setdefault("seed", cfg.get("seed", 0))   # inherit global seed
            stems += use_generate(**params)
        else:
            print(f"  ! unknown silhouette type {t}, skipping")
    seen, uniq = set(), []
    for s in stems:                                 # dedupe, preserve order
        if s not in seen:
            seen.add(s); uniq.append(s)
    ensure_negatives(uniq, seed=cfg.get("seed", 0))     # every silhouette gets a seeded negative
    metas = [json.loads((SIL / f"{s}.json").read_text()) | {"stem": s} for s in uniq]
    print(f"  silhouettes: {len(metas)} ({', '.join(uniq[:6])}{' ...' if len(uniq) > 6 else ''})")
    return metas
