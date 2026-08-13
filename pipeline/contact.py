"""Contact sheet from the manifest: backends side by side per (silhouette, object). Labels are
for the maintainer only -- the stimulus files themselves keep opaque names."""
import csv
from collections import defaultdict
from PIL import Image, ImageDraw
from .utils import OUT

CELL, PAD, LIMIT = 300, 26, 80


def run(cfg):
    man = OUT / "manifest.csv"
    if not man.exists():
        print("  contact: no manifest"); return
    rows = list(csv.DictReader(open(man)))
    conds = [c for c in ("illusory", "control") if any(r.get("condition") == c for r in rows)]
    if not conds:
        conds = ["illusory"]
    g = defaultdict(dict)                                # (silhouette, object) -> condition -> id
    for r in rows:
        g[(r["silhouette"], r["object"])][r.get("condition", "illusory")] = r["id"]
    keys = sorted(g)[:LIMIT]
    W = len(conds) * CELL + (len(conds) + 1) * PAD
    H = len(keys) * (CELL + PAD) + PAD
    sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
    for i, key in enumerate(keys):
        y = PAD + i * (CELL + PAD)
        for j, cond in enumerate(conds):
            x = PAD + j * (CELL + PAD)
            sid_ = g[key].get(cond)
            f = OUT / "comp" / cond / f"{sid_}.png" if sid_ else None
            if f and not f.exists():
                f = OUT / "cn" / f"{sid_}.png"          # controlnet writes flat to out/cn/
            if f and f.exists():
                im = Image.open(f).convert("RGB"); im.thumbnail((CELL, CELL))
                sheet.paste(im, (x, y))
            else:
                d.rectangle([x, y, x + CELL, y + CELL], outline="#ccc")
            d.text((x + 2, y + CELL + 4), f"{cond} {key[1]}"[:44], fill="#333")
    o = OUT / "contact_sheet.png"; sheet.save(o)
    print(f"  contact: wrote {o} ({len(keys)} rows)")
