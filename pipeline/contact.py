"""Contact sheet from the manifest: backends side by side per (silhouette, object). Labels are
for the maintainer only -- the stimulus files themselves keep opaque names."""
import csv
from collections import defaultdict
from PIL import Image, ImageDraw
from .utils import OUT, stim_path, TEXTURE_DIRS

CELL, PAD, LIMIT = 300, 26, 80


def _find(sid_, cond, tex):
    """Probe both layouts: comp/<texture>/<cond>/ (texture list) then comp/<cond>/, then cn/."""
    for f in (stim_path(OUT, tex, cond, True) / f"{sid_}.png",
              stim_path(OUT, tex, cond, False) / f"{sid_}.png",
              OUT / "cn" / f"{sid_}.png"):                 # controlnet writes flat to out/cn/
        if f.exists():
            return f
    return None


def run(cfg):
    man = OUT / "manifest.csv"
    if not man.exists():
        print("  contact: no manifest"); return
    rows = list(csv.DictReader(open(man)))
    tex = lambda r: int(r["texture"]) if r.get("texture") not in (None, "") else 3
    cond = lambda r: r.get("condition") or "illusory"
    cols = sorted({(tex(r), cond(r)) for r in rows},        # one column per (texture, condition)
                  key=lambda t: (t[0], ["illusory", "control"].index(t[1])))
    multi = len({t for t, _ in cols}) > 1               # labels only; file lookup probes both layouts
    g = defaultdict(dict)                                # (silhouette, object) -> (texture, cond) -> id
    for r in rows:
        g[(r["silhouette"], r["object"])][(tex(r), cond(r))] = r["id"]
    keys = sorted(g)[:LIMIT]
    W = len(cols) * CELL + (len(cols) + 1) * PAD
    H = len(keys) * (CELL + PAD) + PAD
    sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
    for i, key in enumerate(keys):
        y = PAD + i * (CELL + PAD)
        for j, (t, c) in enumerate(cols):
            x = PAD + j * (CELL + PAD)
            sid_ = g[key].get((t, c))
            f = _find(sid_, c, t) if sid_ else None
            if f:
                im = Image.open(f).convert("RGB"); im.thumbnail((CELL, CELL))
                sheet.paste(im, (x, y))
            else:
                d.rectangle([x, y, x + CELL, y + CELL], outline="#ccc")
            label = f"{TEXTURE_DIRS[t]} {c}" if multi else c
            d.text((x + 2, y + CELL + 4), f"{label} {key[1]}"[:44], fill="#333")
    o = OUT / "contact_sheet.png"; sheet.save(o)
    print(f"  contact: wrote {o} ({len(keys)} rows)")
