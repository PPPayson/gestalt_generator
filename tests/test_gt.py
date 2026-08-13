from pipeline import gt, silhouettes as S
from pipeline.utils import SIL, OUT


def _meta_kanizsa():
    S.produce_kanizsa(n_sides=3)
    return {"stem": "kanizsa_triangle", "illusion": "kanizsa_triangle", "illusory_shape": "triangle"}


def test_gt_schema_and_values():
    meta = _meta_kanizsa()
    item = {"object": "moon", "background": "sky", "natural": True}
    g = gt.write("stim_x", meta, item, "composite", "perfect")
    assert set(g) == {"id", "n_inducers", "has_illusion", "illusory_shape", "nonillusory_shapes"}
    assert g["id"] == "stim_x" and g["has_illusion"] is True
    assert g["illusory_shape"] == "triangle" and g["n_inducers"] == 3
    assert g["nonillusory_shapes"]["real_life"] == ["moon", "moon", "moon"]
    assert g["nonillusory_shapes"]["primitive"] == ["circle", "circle", "circle"]
    assert (OUT / "gt" / "stim_x.json").exists()


def test_gt_gestalt_is_irregular():
    stem = S.use_gestalt(mode="existing", n=1, seed=0)[0]
    import json
    meta = json.loads((SIL / f"{stem}.json").read_text()) | {"stem": stem}
    g = gt.write("stim_g", meta, {"object": "coin", "background": "felt"}, "composite", "perfect")
    assert g["illusory_shape"] == "irregular"
    assert g["n_inducers"] == len(g["nonillusory_shapes"]["real_life"])
