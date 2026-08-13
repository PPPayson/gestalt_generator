"""Generate object cutouts (best-of-N, keyed by roundness) and background flat-lays on Modal.
Idempotent: skips anything already on disk."""
import io
from PIL import Image
from .utils import OBJ, BG, slug, roundness


def ensure(sdxl, objects, bgs, best_of=3):
    OBJ.mkdir(parents=True, exist_ok=True); BG.mkdir(parents=True, exist_ok=True)
    objects = sorted({o for o in objects if not (OBJ / f"{slug(o)}.png").exists()})
    bgs = sorted({b for b in bgs if not (BG / f"{slug(b)}.png").exists()})
    print(f"  assets: {len(objects)} new objects, {len(bgs)} new bgs")
    if objects:
        args = [(o, s) for o in objects for s in range(best_of)]     # best-of-N jobs
        best = {}
        for (o, _), b in zip(args, sdxl.gen_object.starmap(args)):   # fan out across containers
            sc = roundness(Image.open(io.BytesIO(b)))
            if o not in best or sc > best[o][0]:
                best[o] = (sc, b)
        for o, (sc, b) in best.items():
            (OBJ / f"{slug(o)}.png").write_bytes(b); print(f"    obj {o}: roundness {sc:.2f}")
    if bgs:
        for bg, b in zip(bgs, sdxl.gen_bg.starmap([(x, 0) for x in bgs])):
            (BG / f"{slug(bg)}.png").write_bytes(b); print(f"    bg {bg}")
