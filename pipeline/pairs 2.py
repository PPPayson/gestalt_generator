"""Resolve object/background pairs from the shipped catalog or an explicit list (no LLM run)."""
import json, random, yaml
from .utils import DATA, ROOT

CATALOG = ROOT / "pipeline" / "pairs_catalog.yaml"


def resolve(cfg, metas):
    src = cfg["pairs"]["source"]
    if src == "explicit":
        items = [dict(i, natural=i.get("natural", False)) for i in cfg["pairs"]["explicit"]]
    else:                                                # catalog
        cat = (yaml.safe_load(CATALOG.read_text()) or {}).get("pairs", [])
        f = cfg["pairs"]["filter"]
        def ok(it):
            if f.get("natural") is not None and it.get("natural") != f["natural"]:
                return False
            if it.get("contrast", 1.0) < f.get("min_contrast", 0.0):
                return False
            if f.get("objects") and it["object"] not in f["objects"]:
                return False
            return True
        items = [i for i in cat if ok(i)]
        if cfg["pairs"].get("select") == "random":       # random N textures (seeded)
            seed = cfg["pairs"].get("seed")
            if seed is None:
                seed = cfg.get("seed", 0)                 # inherit global seed
            items = random.Random(seed).sample(
                items, min(cfg["pairs"]["n_per_silhouette"], len(items)))
        else:                                             # highest contrast first
            items = sorted(items, key=lambda x: -x.get("contrast", 0))
    items = items[: cfg["pairs"]["n_per_silhouette"]]
    pairs = {m["stem"]: [dict(i) for i in items] for m in metas}
    (DATA / "pairs.json").write_text(json.dumps(pairs, indent=2))
    print(f"  pairs[{src}]: {len(items)} per silhouette")
    return pairs


def needed(pairs):
    objs = {it["object"] for v in pairs.values() for it in v}
    bgs = {it["background"] for v in pairs.values() for it in v}
    return objs, bgs
