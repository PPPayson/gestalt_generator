"""Shape + contrast filter over proposed candidates -> pairs.json (source=propose)."""
import json
from PIL import Image
from .utils import DATA, OBJ, BG, slug, roundness, luminance, load_all_candidates


def run(cfg, metas=None):
    s = cfg["select"]
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
            c = abs(luminance(Image.open(of)) - luminance(Image.open(bf)))
            if not (c >= s["contrast_high"] or (nat and c >= s["contrast_low_natural"])):
                dropped.append((o, b, f"low-contrast {c:.2f} nat={nat}")); continue
            kept.append({"object": o, "background": b, "natural": nat,
                         "roundness": round(rnd, 2), "contrast": round(c, 2)})
        pairs[illusion] = kept
    (DATA / "pairs.json").write_text(json.dumps(pairs, indent=2))
    tot = sum(len(v) for v in pairs.values())
    print(f"  validate: kept {tot}, dropped {len(dropped)}")
    return pairs
