"""Generate object cutouts (best-of-N, keyed by roundness) and background flat-lays via a provider.
Provider-agnostic (threadpool over provider.gen_object/gen_bg). Idempotent: skips what's on disk."""
import io
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from .utils import OBJ, BG, slug, roundness

MAX_WORKERS = 8


def ensure(provider, objects, bgs, best_of=3):
    OBJ.mkdir(parents=True, exist_ok=True); BG.mkdir(parents=True, exist_ok=True)
    objects = sorted({o for o in objects if not (OBJ / f"{slug(o)}.png").exists()})
    bgs = sorted({b for b in bgs if not (BG / f"{slug(b)}.png").exists()})
    print(f"  assets: {len(objects)} new objects, {len(bgs)} new bgs")
    if objects:
        jobs = [(o, s) for o in objects for s in range(best_of)]     # best-of-N jobs
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            results = list(ex.map(lambda j: (j[0], provider.gen_object(j[0], j[1])), jobs))
        best = {}
        for o, b in results:
            sc = roundness(Image.open(io.BytesIO(b)))
            if o not in best or sc > best[o][0]:
                best[o] = (sc, b)
        for o, (sc, b) in best.items():
            (OBJ / f"{slug(o)}.png").write_bytes(b); print(f"    obj {o}: roundness {sc:.2f}")
    if bgs:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            results = list(ex.map(lambda b: (b, provider.gen_bg(b, 0)), bgs))
        for bg, b in results:
            (BG / f"{slug(bg)}.png").write_bytes(b); print(f"    bg {bg}")
