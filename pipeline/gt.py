"""Private ground-truth writer: the graded answers only, one JSON per stimulus, keyed by
opaque id (kept in out/gt/, separate from the images so filenames stay uninformative).

Task = the model names the illusory shape and counts inducers, so GT holds exactly:
  n_inducers, has_illusion, illusory_shape, nonillusory_shapes{real_life, primitive}.
Provenance (object/background/backend/mode) lives in out/manifest.csv, joined by id.
"""
import json
from .utils import OUT, SIL
from .geometry import inducers


def _illusory_shape(meta):
    if meta.get("illusory_shape"):
        return meta["illusory_shape"]                       # registry / gestalt sidecar
    if meta["illusion"].startswith("kanizsa_"):
        return meta["illusion"].split("_", 1)[1]            # kanizsa_triangle -> triangle
    return "irregular"                                      # arbitrary drop-in


def write(stim_id, meta, item, backend, mode, condition="illusory"):
    n = len(inducers(SIL / f"{meta['stem']}.png"))
    obj = item["object"]
    primitive = meta.get("primitive", "circle")             # inducer geometry (his + our discs = circle)
    control = condition == "control"                        # the illusory shape's contour is DRAWN
    gt = {
        "id": stim_id,
        "condition": condition,                             # illusory | control
        "n_inducers": n,
        "has_illusion": not control,                        # control shows a real (drawn) contour, not illusory
        "has_visible_contour": control,
        "illusory_shape": _illusory_shape(meta),
        "polygon_vertices": meta.get("polygon_vertices"),   # deterministic shape descriptor (gestalt)
        "nonillusory_shapes": {
            "real_life": [obj] * n,                         # the dressed object, one per inducer
            "primitive": [primitive] * n,                   # geometric primitive, one per inducer
        },
    }
    d = OUT / "gt"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{stim_id}.json").write_text(json.dumps(gt, indent=2))
    return gt
