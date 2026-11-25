"""tools/export_benchmark lays out a three-condition run with a negative folder and answer rows."""
import csv
import json
from pipeline import dress_composite as DC
from pipeline.negative import ensure_negatives
from pipeline.utils import OUT, SIL
from tools import export_benchmark as EB


def test_export_level_writes_negative_folder(cfg, tmp_path):
    stem = sorted(p.stem for p in SIL.glob("img_*.png"))[0]
    assert ensure_negatives([stem], seed=1) == []
    metas = [{"stem": stem, "illusion": stem, "illusory_shape": "irregular"}]
    pairs = {stem: [{"object": "white disc", "background": "black bg", "natural": False}]}
    c = cfg(dress={"backends": ["composite"], "modes": ["perfect"],
                   "conditions": ["illusory", "control", "negative"], "contour": "auto",
                   "shadow": 0, "crop": True, "feather": 1.5})
    DC.run(c, metas, pairs)
    rows = EB.export_level(OUT, tmp_path / "all", {stem: 7}, 92)
    assert {r["condition"] for r in rows} == {"illusory", "contour", "negative"}
    assert all(r["figure"] == "f007" for r in rows)
    for r in rows:
        assert (tmp_path / "all" / r["file"]).exists()
        key = json.loads((tmp_path / "all" / "answers" / (r["file"].split("/")[1][:-4] + ".json")).read_text())
        assert key["has_illusion"] == (r["condition"] == "illusory")
    neg = [r for r in rows if r["condition"] == "negative"][0]
    assert neg["file"].startswith("texture_negative/") and neg["has_illusion"] is False   # default texture 3
    assert len(list(csv.DictReader(open(tmp_path / "all" / "answers.csv")))) == 3
