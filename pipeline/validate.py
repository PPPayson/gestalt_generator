"""Shape + COLOR-contrast filter over proposed candidates -> pairs.json (source=propose).
A pair only passes if the object is disc-shaped AND visibly stands out from the background by
colour (CIELAB delta-E) -- so camouflage (apple on red apples) does not pass, regardless of how
'fitting' it is."""
import json
from PIL import Image
from .utils import DATA, OBJ, BG, slug, roundness, color_contrast, load_all_candidates


def run(cfg, metas=None):
    s = cfg["select"]
    de_min = s.get("delta_e_min", 30.0)                 # min perceptual colour contrast
    cand = load_all_candidates()
    pairs, dropped = {}, []
    for illusion, items in cand.items():
        kept = []
        for it in items:
            o, b, nat = it["object"], it["background"], it["natural"]
            of, bf = OBJ / f"{slug(o)}.png", BG / f"{slug(b)}.png"
            if not of.exists() or not bf.exists():
                dropped.append((o, b, "missing")); continue
            rnd = roundness(Image.open(of))
            if rnd < s["round_min"]:
                dropped.append((o, b, f"not-disc {rnd:.2f}")); continue
            de = color_contrast(Image.open(of), Image.open(bf))
            if de < de_min:
                dropped.append((o, b, f"camouflage dE={de:.0f} nat={nat}")); continue
            kept.append({"object": o, "background": b, "natural": nat,
                         "roundness": round(rnd, 2), "contrast": round(de, 1)})
        pairs[illusion] = kept
    (DATA / "pairs.json").write_text(json.dumps(pairs, indent=2))
    tot = sum(len(v) for v in pairs.values())
    print(f"  validate: kept {tot}, dropped {len(dropped)}")
    return pairs
