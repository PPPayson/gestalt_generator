"""Render per-lever sweep examples: one fixed figure, three rows, each row moving ONE difficulty
lever (texture / closeness / contrast-camouflage) easy -> hard while the other two sit at the
baseline (texture=3, closeness=0.5, contrast=0). Output: docs/difficulty_examples.png, the figure
the README embeds. Silhouettes are generated fresh into out/difficulty/ so the shipped store in
data/ is untouched.

Run from the repo root:  python3 tools/make_difficulty_examples.py
"""
import sys, pathlib, random, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import yaml
from PIL import Image, ImageDraw, ImageFont
import main
from pipeline import dress_composite as DC
from pipeline.utils import ROOT, OUT, OBJ, slug


def _font(size):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()

# the one figure shown in every cell
NV, NP, EDGY, RAD, RANDOMNESS, SEED = 4, 5, 0.1, 0.1, 0, 2
BASE = {"texture": 3, "closeness": 0.5, "contrast": 0.0}
ROWS = [  # (label, lever, swept values) -- each row goes easy -> hard, others fixed at BASE
    ("TEXTURE",   "texture",   [0, 1, 2, 3]),
    ("CLOSENESS", "closeness", [1.0, 0.75, 0.5, 0.25, 0.0]),
    ("CONTRAST",  "contrast",  [0.0, 0.25, 0.5, 0.75, 1.0]),
]
CELL, CAP_H, LABEL_W, LINE = 300, 52, 190, "#999"


def catalog_pair(seed=1, prefer="orange sand"):
    """One catalog pair whose assets are shipped; prefer a strongly TEXTURED background so the
    fractional texture levels are visible on the sheet."""
    from pipeline.utils import BG
    cat = yaml.safe_load((pathlib.Path("pipeline") / "pairs_catalog.yaml").read_text())["pairs"]
    ok = [p for p in cat if (OBJ / f"{slug(p['object'])}.png").exists()
          and (BG / f"{slug(p['background'])}.png").exists()]
    hits = [p for p in ok if p["background"] == prefer]
    return hits[0] if hits else random.Random(seed).choice(ok)


def gen_silhouette(closeness, sil_dir, ctrl_dir):
    stem = main.generate_one(RANDOMNESS, NV, NP, EDGY, RAD, closeness=closeness,
                             sil_dir=str(sil_dir), ctrl_dir=str(ctrl_dir), seed=SEED)
    if stem is None:                                 # cannot happen post feasibility fix, but be loud
        raise RuntimeError(f"disc placement failed at closeness={closeness}")
    return stem


def build_sheet():
    work = OUT / "difficulty"
    sil_dir, ctrl_dir = work / "silhouettes", work / "control"
    sil_dir.mkdir(parents=True, exist_ok=True); ctrl_dir.mkdir(parents=True, exist_ok=True)
    pair = catalog_pair()
    ncols = max(len(vals) for _, _, vals in ROWS)
    ROW_H = CELL + CAP_H
    W = LABEL_W + ncols * CELL + 1
    H = len(ROWS) * ROW_H + 1
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    f_big, f_small = _font(30), _font(20)
    for row, (label, lever, values) in enumerate(ROWS):
        y = row * ROW_H
        fixed = {k: v for k, v in BASE.items() if k != lever}
        d.rectangle([0, y, LABEL_W, y + ROW_H], outline=LINE)
        d.text((LABEL_W // 2, y + ROW_H // 2 - 20), label, fill="black", font=f_big, anchor="mm")
        for j, (k, v) in enumerate(fixed.items()):
            d.text((LABEL_W // 2, y + ROW_H // 2 + 16 + j * 24),
                   f"{k}={v}", fill="#666", font=f_small, anchor="mm")
        for col, val in enumerate(values):
            p = dict(BASE); p[lever] = val
            stem = gen_silhouette(p["closeness"], sil_dir, ctrl_dir)
            support = json.loads((sil_dir / f"{stem}.json").read_text())["support_ratio"]
            info = {}
            img, _ = DC.render(sil_dir / f"{stem}.png", pair["object"], pair["background"],
                               shadow=0, crop=True,
                               texture=p["texture"], contrast=p["contrast"], info=info)
            img.thumbnail((CELL - 12, CELL - 12))
            x = LABEL_W + col * CELL
            d.rectangle([x, y, x + CELL, y + ROW_H], outline=LINE)     # table cell
            d.line([x, y + CELL, x + CELL, y + CELL], fill=LINE)       # image | caption divider
            sheet.paste(img, (x + (CELL - img.width) // 2, y + (CELL - img.height) // 2))
            d.text((x + CELL // 2, y + CELL + CAP_H // 2), f"{lever} = {val}",
                   fill="black", font=f_big, anchor="mm")
            print(f"  {label}: {lever}={val} stem={stem} support={support:.2f} "
                  f"lum_gap={info.get('lum_gap')}")
    out = ROOT / "docs" / "difficulty_examples.png"
    sheet.save(out)
    print(f"wrote {out}  (figure: {NV}-gon, {NP} inducers, seed {SEED}; "
          f"pair: {pair['object']} / {pair['background']})")
    return out


if __name__ == "__main__":
    build_sheet()
