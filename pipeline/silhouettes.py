"""Silhouettes come from config specs only. Three spec types:
  - kanizsa : built-in registry producer (clean illusory n-gons)
  - gestalt : select N (seeded) from PPPayson/gestalt_generator output, or regenerate via main.generate
  - dropin  : explicit PNG file(s) already in data/silhouettes/
The friend's generator (main.py) writes into data/silhouettes/, which is also our store."""
import json, math, re, random
import numpy as np
from PIL import Image, ImageDraw
from .utils import SIL, SIL_CTRL

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


_UNSET = object()


def rank_score(path, meta, min_inducers=_UNSET, all_sides=_UNSET, flat_solidity=_UNSET,
               max_gap=_UNSET, max_flat_frac=_UNSET, **filt):
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
    unknown = set(filt) - set(FILTER_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown filter keys {sorted(unknown)}; known: {sorted(FILTER_DEFAULTS)}")
    f = {**FILTER_DEFAULTS, **{k: v for k, v in dict(min_inducers=min_inducers, all_sides=all_sides,
                                                   flat_solidity=flat_solidity, max_gap=max_gap,
                                                   max_flat_frac=max_flat_frac,
                                                   **filt).items() if v is not _UNSET}}
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


def use_gestalt(mode="random", n=20, seed=0, src_dir=None, regenerate=None, inducers=None,
                filter=None):
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


# --- generate at a chosen closeness ----------------------------------------
QUOTA_DEFAULT = {3: 0.15, 4: 0.35, 5: 0.35, 6: 0.15}      # inducer-count mix: peaked, not uniform
GRID_DEFAULT = {"randomness": [0, 0.4, 0.8, 1], "n_vertices": [3, 4, 5, 6, 7, 8],
                "edgy": [0.1, 0.5], "rad": [0.1, 0.2]}       # main.generate's grid minus n_points


def _quota_counts(n, quota):
    """{n_points: count}. `quota` values all <= 1 are FRACTIONS (renormalised to sum 1, remainder
    to the largest); otherwise they are absolute counts and must sum to `n`. None -> default mix."""
    q = {int(k): float(v) for k, v in (QUOTA_DEFAULT if quota is None else quota).items()}
    if not q or sum(q.values()) <= 0:
        raise ValueError(f"quota must request at least one figure: {quota}")
    if all(v <= 1.0 for v in q.values()):                           # fractions -> counts
        tot = sum(q.values())
        exact = {k: n * v / tot for k, v in q.items()}
        counts = {k: int(x) for k, x in exact.items()}
        for k in sorted(q, key=lambda k: -(exact[k] - counts[k]))[: n - sum(counts.values())]:
            counts[k] += 1                                              # largest-remainder rounding
    else:
        counts = {k: int(v) for k, v in q.items()}
        if n and sum(counts.values()) != n:
            raise ValueError(f"quota counts {counts} sum to {sum(counts.values())}, not n={n}")
    if sum(counts.values()) == 0:
        raise ValueError(f"quota {quota} yields no figures for n={n}")
    return counts


def _eval_candidate(job):
    """Worker: generate one candidate at every closeness in `levels`; the FIRST level is judged
    with the filter, the rest only need to be feasible + clean. -> (score, fig) or None."""
    import main, tempfile, shutil
    from .geometry import separate_inducer_count
    fig, levels, filt = job
    r, nv, np_, e, rad, seed = fig
    work = tempfile.mkdtemp(prefix="gestalt_sample_")
    try:
        score = None
        for i, c in enumerate(levels):
            sd, cd = f"{work}/s{i}", f"{work}/c{i}"
            import os; os.makedirs(sd); os.makedirs(cd)
            stem = main.generate_one(r, nv, np_, e, rad, closeness=c, sil_dir=sd, ctrl_dir=cd,
                                     seed=seed, tag_seed=True)
            if stem is None or separate_inducer_count(f"{sd}/{stem}.png") != np_:
                return None
            if i == 0:
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
                   also_closeness=(), max_seeds=200, jobs=None, log=print):
    """Keep drawing figures until `n` PASS the filter, in the requested inducer-count mix.
    Draws the parameter `grid` (x n_points from the quota) under seed, seed+1, ... ; each candidate
    is rendered at `closeness` (judged by rank_score with `filter`) and, if `also_closeness` is
    given, at those levels too (must be feasible + clean there: matched multi-level sets). Stops
    when every quota is met or `max_seeds` rounds are exhausted (then returns what it has, with a
    warning). Returns [[randomness, n_vertices, n_points, edgy, rad, seed], ...] best-first per
    inducer count. Parallel over `jobs` workers (default: all cores but one)."""
    import itertools, multiprocessing as mp
    from .utils import progress, bar
    g = {**GRID_DEFAULT, **(grid or {})}
    filt = {**FILTER_DEFAULTS, **(filter or {})}
    need = _quota_counts(n, quota)
    levels = [closeness] + list(also_closeness)
    have = {k: [] for k in need}
    jobs = jobs or max(1, mp.cpu_count() - 1)
    unknown = set(filt) - set(FILTER_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown filter keys {sorted(unknown)}; known: {sorted(FILTER_DEFAULTS)}")
    done = lambda: all(len(have[k]) >= need[k] for k in need)
    log(f"  sample: need {need} at closeness={closeness} (+{list(also_closeness)}), filter={filt}")
    passed = lambda: sum(min(len(have[k]), need[k]) for k in need)
    short_str = lambda: " ".join(f"{k}:{min(len(have[k]), need[k])}/{need[k]}" for k in need)
    top = bar(n, f"figures passed (closeness={closeness})")     # persists across seed rounds
    top.set_postfix_str(short_str())
    # NOTE: the default start method (spawn on macOS/Windows) re-imports the caller's __main__ in
    # every worker, so entry scripts must be guarded (run.py is); fork is not used because forking
    # a process with live BLAS/tqdm threads deadlocks on macOS. jobs == 1 never forks or spawns.
    with (mp.Pool(jobs) if jobs > 1 else _Serial()) as pool:
        for rnd in range(max_seeds):
            s = seed + rnd
            short = [k for k in need if len(have[k]) < need[k]]
            combos = [[r, nv, np_, e, rad, s] for r, nv, np_, e, rad in itertools.product(
                g["randomness"], g["n_vertices"], short, g["edgy"], g["rad"])]
            before = passed()
            for res in progress(pool.imap_unordered(_eval_candidate, [(f, levels, filt) for f in combos]),
                                desc=f"round {rnd + 1} (seed {s})", total=len(combos)):
                if res:
                    have[res[1][2]].append(res)
            # one figure per polygon: edgy/rad cells of the same (randomness, n_vertices, n_points,
            # seed) are near-identical figures, keep only the best-scoring one
            for k in have:
                best = {}
                for sc, fig in have[k]:
                    key = (fig[0], fig[1], fig[2], fig[5])
                    if key not in best or sc > best[key][0]:
                        best[key] = (sc, fig)
                have[k] = list(best.values())
            top.set_postfix_str(short_str()); top.update(passed() - before)
            if done():
                break
    top.close()
    out = []
    for k in need:
        got = sorted(have[k], key=lambda t: -t[0])[: need[k]]
        if len(got) < need[k]:
            log(f"  ! sample: only {len(got)}/{need[k]} figures with {k} inducers after {max_seeds} seeds")
        out += [fig for _, fig in got]
    log(f"  sample: {len(out)}/{n} figures passed")
    return out


def use_generate(figures=None, closeness=0.5, seed=None, n=None, quota=None, grid=None,
                 filter=None, also_closeness=(), max_seeds=200):
    """`type: generate`. Either pin `figures` (explicit [[randomness, n_vertices, n_points, edgy,
    rad, seed], ...]) or ask for `n` figures that pass the filter (sample_figures: quota / grid /
    filter / also_closeness as above). Figures are then rendered at `closeness` into the store."""
    if figures is None:
        assert n, "type: generate needs `figures` or `n`"
        figures = sample_figures(n, closeness=closeness, quota=quota, grid=grid, filter=filter,
                                 seed=seed or 0, also_closeness=also_closeness, max_seeds=max_seeds)
    return _render_figures(figures, closeness, seed)


def _render_figures(figures, closeness, seed=None):
    """Render figures fresh via main.generate_one at ONE `closeness` into the store (SIL + SIL_CTRL).
    `figures` = [[randomness, n_vertices, n_points, edgy, rad, (seed)], ...]; a per-figure seed
    overrides `seed`. The same figures + seeds at another closeness give a matched figure (same
    polygon rotation, resized bites). Unclean or infeasible figures are skipped with a warning."""
    import main
    from .geometry import separate_inducer_count
    SIL.mkdir(parents=True, exist_ok=True); SIL_CTRL.mkdir(parents=True, exist_ok=True)
    from .utils import progress
    stems = []
    for fig in progress(figures, desc=f"generate closeness={closeness}"):
        randomness, n_vertices, n_points, edgy, rad = fig[:5]
        s = fig[5] if len(fig) > 5 else seed
        stem = main.generate_one(randomness, int(n_vertices), int(n_points), edgy, rad,
                                 closeness=closeness, sil_dir=str(SIL), ctrl_dir=str(SIL_CTRL),
                                 seed=s, tag_seed=True)
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
    metas = [json.loads((SIL / f"{s}.json").read_text()) | {"stem": s} for s in uniq]
    print(f"  silhouettes: {len(metas)} ({', '.join(uniq[:6])}{' ...' if len(uniq) > 6 else ''})")
    return metas
