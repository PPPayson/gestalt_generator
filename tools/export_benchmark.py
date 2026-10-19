"""Export a finished pipeline output into the committed benchmark layout under benchmarks/<name>/:

    benchmarks/<name>/
      README.md                               what the set is + how to read a filename
      preview_<level>.jpg                     the level's contact sheet (half size)
      <level>/
        plain_illusory/    f001_plain_illusory.jpg ...     black inducers on a flat fill, no contour
        plain_contour/     f001_plain_contour.jpg ...      same, contour drawn
        texture_illusory/  f001_texture_illusory.jpg ...   real object on real background
        texture_contour/   f001_texture_contour.jpg ...
        answers.csv                           one row per image (figure, variant, condition, keys ...)
        answers/           f001_plain_illusory.json ...   per-image answer key (evaluate.py reads these)

`fNNN` is the FIGURE number: the same figure keeps its number in every level, variant and condition,
so matched comparisons are a filename away. Figures are numbered in a seeded shuffled order, so the
number carries no answer information; the pipeline's opaque stimulus id is kept in answers.csv.
A source dir holding level sub-dirs (each with a manifest.csv) is exported level by level; a
single-level output (out/) is exported as one level named "all".

    python3 tools/export_benchmark.py out/varying_difficulty_benchmark_100 varying_difficulty_benchmark_100
"""
import sys, csv, json, shutil, pathlib, argparse, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pipeline.utils import TEXTURE_DIRS, progress
from pipeline.silhouettes import GESTALT_RE
from PIL import Image

COND = {"illusory": "illusory", "control": "contour"}
ANSWER_COLS = ["figure", "file", "variant", "condition", "n_inducers", "has_illusion",
               "has_visible_contour", "illusory_shape", "polygon_vertices", "object", "background",
               "texture", "closeness", "closeness_measured", "support_ratio", "contrast", "lum_gap",
               "silhouette", "stim_id"]


def _source(src, r, multi):
    tex = int(r["texture"])
    for f in (src / "comp" / TEXTURE_DIRS[tex] / r["condition"] / f"{r['id']}.png",
              src / "comp" / r["condition"] / f"{r['id']}.png", src / "cn" / f"{r['id']}.png"):
        if f.exists():
            return f
    return None


def save_preview(sheet_png, out_jpg, scale=0.5, quality=80):
    """Contact sheets are 1330 x 26000 px PNGs (10-14 MB); the committed preview is a half-size JPEG."""
    im = Image.open(sheet_png).convert("RGB")
    im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    im.save(out_jpg, quality=quality, optimize=True)


def export_level(src, dst, numbers, quality):
    rows = list(csv.DictReader(open(src / "manifest.csv")))
    for r in rows:
        if r.get("texture") in (None, ""):
            r["texture"] = json.loads((src / "gt" / f"{r['id']}.json").read_text())["difficulty"]["texture"]
    multi = len({int(r["texture"]) for r in rows}) > 1
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "answers").mkdir(parents=True)
    out_rows = []
    for r in progress(rows, desc=f"export {dst.name}"):
        f = _source(src, r, multi)
        if f is None:
            print(f"  ! missing stimulus {r['id']}"); continue
        variant, cond = TEXTURE_DIRS[int(r["texture"])], COND[r["condition"]]
        stem = f"f{numbers[r['silhouette']]:03d}_{variant}_{cond}"
        d = dst / f"{variant}_{cond}"; d.mkdir(exist_ok=True)
        Image.open(f).convert("RGB").save(d / f"{stem}.jpg", quality=quality)
        gt = json.loads((src / "gt" / f"{r['id']}.json").read_text())
        diff = gt.get("difficulty", {})
        row = {"figure": f"f{numbers[r['silhouette']]:03d}", "file": f"{variant}_{cond}/{stem}.jpg",
               "variant": variant, "condition": cond, "n_inducers": gt["n_inducers"],
               "has_illusion": gt["has_illusion"], "has_visible_contour": gt["has_visible_contour"],
               "illusory_shape": gt["illusory_shape"], "polygon_vertices": gt.get("polygon_vertices"),
               "object": r["object"], "background": r["background"], "texture": int(r["texture"]),
               "closeness": diff.get("closeness"), "closeness_measured": diff.get("closeness_measured"),
               "support_ratio": diff.get("support_ratio"), "contrast": diff.get("contrast"),
               "lum_gap": diff.get("lum_gap"), "silhouette": r["silhouette"], "stim_id": r["id"]}
        out_rows.append(row)
        (dst / "answers" / f"{stem}.json").write_text(json.dumps({**gt, **row}, indent=2))
    out_rows.sort(key=lambda x: (x["figure"], x["variant"], x["condition"]))
    with open(dst / "answers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ANSWER_COLS); w.writeheader(); w.writerows(out_rows)
    if (src / "contact_sheet.png").exists():
        save_preview(src / "contact_sheet.png", dst.parent / f"preview_{dst.name}.jpg")
    print(f"  exported {len(out_rows)} images -> {dst}")
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="pipeline output dir, or a dir of per-level output dirs")
    ap.add_argument("name", help="benchmark name -> benchmarks/<name>/")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--seed", type=int, default=0, help="figure numbering shuffle")
    a = ap.parse_args()
    src, dst = pathlib.Path(a.src), pathlib.Path("benchmarks") / a.name
    order = {"easy": 0, "normal": 1, "hard": 2}                     # readable level order
    levels = [src] if (src / "manifest.csv").exists() else \
        sorted((p for p in src.iterdir() if p.is_dir() and (p / "manifest.csv").exists()),
               key=lambda p: (order.get(p.name, 9), p.name))
    # figure numbers: shared across levels, keyed on the silhouette with its closeness suffix removed
    def base(stem):                                   # figure identity = params + seed, closeness dropped
        m = GESTALT_RE.match(stem)
        if not m:
            return stem
        params, seed = m.groups()[:5], m.group(7)
        return "img_" + "_".join(params) + (f"_s{seed}" if seed else "")
    figs = sorted({base(r["silhouette"]) for lv in levels
                   for r in csv.DictReader(open(lv / "manifest.csv"))})
    random.Random(a.seed).shuffle(figs)
    number = {f: i + 1 for i, f in enumerate(figs)}
    summary = {}
    for lv in levels:
        rows = list(csv.DictReader(open(lv / "manifest.csv")))
        numbers = {r["silhouette"]: number[base(r["silhouette"])] for r in rows}
        name = lv.name if lv != src else "all"
        out = export_level(lv, dst / name, numbers, a.quality)
        summary[name] = out
    _write_readme(dst, a.name, summary)
    print(f"wrote benchmarks/{a.name}/README.md")


def _write_readme(dst, name, summary):
    lines = [f"# {name}", "",
             f"{len(summary)} level(s), {sum(len(v) for v in summary.values())} images. "
             "Every image is `<level>/<variant>_<condition>/f<figure>_<variant>_<condition>.jpg`; "
             "`f<figure>` is the same figure in every level, variant and condition.", "",
             "| level | images | closeness | contrast | support ratio (mean) |", "|---|---|---|---|---|"]
    for lv, rows in summary.items():
        sr = [float(r["support_ratio"]) for r in rows if r["support_ratio"] not in (None, "")]
        lines.append(f"| {lv} | {len(rows)} | {rows[0]['closeness']} | {rows[0]['contrast']} | "
                     f"{sum(sr) / len(sr):.2f} |" if sr else f"| {lv} | {len(rows)} | | | |")
    lines += ["",
              "- `plain_*`: black inducers on a flat fill (texture 0). `texture_*`: real object on a real background (texture 3).",
              "- `*_illusory`: no contour drawn (`has_illusion: true`). `*_contour`: the shape's outline drawn (positive control).",
              "- `answers.csv`: one row per image with the answer key and the difficulty parameters; `answers/<stem>.json` is the same per image.",
              "- `preview_<level>.jpg`: contact sheet of the level (half size).", "",
              "Regenerate: `python3 tools/run_difficulty_set.py --name " + name + "` then "
              "`python3 tools/export_benchmark.py out/" + name + " " + name + "`."]
    (dst / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
