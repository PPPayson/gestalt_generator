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
    backends = [b for b in ("controlnet", "composite") if any(r["backend"] == b for r in rows)]
    g = defaultdict(dict)
    for r in rows:
        g[(r["silhouette"], r["object"], r.get("natural", ""))][r["backend"]] = (r["id"], r["mode"])
    keys = sorted(g)[:LIMIT]
    folders = {"controlnet": OUT / "cn", "composite": OUT / "comp"}
    W = len(backends) * CELL + (len(backends) + 1) * PAD
    H = len(keys) * (CELL + PAD) + PAD
    sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
    for i, key in enumerate(keys):
        y = PAD + i * (CELL + PAD)
        for j, backend in enumerate(backends):
            x = PAD + j * (CELL + PAD)
            hit = g[key].get(backend)
            f = folders[backend] / f"{hit[0]}.png" if hit else None
            if f and f.exists():
                im = Image.open(f).convert("RGB"); im.thumbnail((CELL, CELL))
                sheet.paste(im, (x, y))
            else:
                d.rectangle([x, y, x + CELL, y + CELL], outline="#ccc")
            tag = "nat" if key[2] in ("True", "true") else "any"
            d.text((x + 2, y + CELL + 4), f"{backend} {key[1]} [{tag}]"[:44], fill="#333")
    o = OUT / "contact_sheet.png"; sheet.save(o)
    print(f"  contact: wrote {o} ({len(keys)} rows)")
