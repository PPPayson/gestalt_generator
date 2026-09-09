"""Score every shipped background's intrinsic busyness (bounded distractor-like speckles) and
render a labeled line-up sorted busiest-first to out/bg_busyness.png, so cluttered surfaces can
be picked out and their pairs dropped from the catalog.

Run from the repo root:  python3 tools/measure_busyness.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import yaml
from PIL import Image, ImageDraw
from pipeline.utils import BG, OUT, slug, busyness

CELL, PAD, TEXT_H = 260, 18, 46


def main():
    cat = yaml.safe_load((pathlib.Path("pipeline") / "pairs_catalog.yaml").read_text())["pairs"]
    used_by = {}
    for p in cat:
        used_by.setdefault(slug(p["background"]), []).append(p["object"])
    scored = []
    for f in sorted(BG.glob("*.png")):
        score, n, dev = busyness(f)
        scored.append((score, n, dev, f))
    scored.sort(reverse=True)                        # busiest first

    print(f"{'background':32s} {'score':>8s} {'patches':>8s} {'dev_std':>8s}  pairs using it")
    for score, n, dev, f in scored:
        print(f"{f.stem:32s} {score:8.0f} {n:8d} {dev:8.1f}  {', '.join(used_by.get(f.stem, ['-']))}")

    cols = 4
    rows = (len(scored) + cols - 1) // cols
    W = cols * (CELL + PAD) + PAD
    H = rows * (CELL + TEXT_H + PAD) + PAD
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for i, (score, n, dev, f) in enumerate(scored):
        x = PAD + (i % cols) * (CELL + PAD)
        y = PAD + (i // cols) * (CELL + TEXT_H + PAD)
        im = Image.open(f).convert("RGB")
        im.thumbnail((CELL, CELL))
        sheet.paste(im, (x, y))
        d.text((x, y + CELL + 4), f"{f.stem}  score={score:.0f}", fill="black")
        d.text((x, y + CELL + 18), f"patches={n} dev_std={dev:.1f}", fill="#666")
        d.text((x, y + CELL + 32), ("pairs: " + ", ".join(used_by.get(f.stem, ["none"])))[:48], fill="#666")
    out = OUT / "bg_busyness.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
