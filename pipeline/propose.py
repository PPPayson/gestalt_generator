"""Generative pair proposal via a Proposer provider. Merges into candidate files (no clobber)."""
import json, re
from .utils import DATA


def run(proposer, cfg, metas):
    n = cfg["propose"]["n_per"]
    seed = cfg.get("seed", 0)                            # reproducible proposer sampling
    for natural in cfg["propose"]["natural"]:
        fn = DATA / ("candidates_nat.json" if natural else "candidates_any.json")
        cand = json.loads(fn.read_text()) if fn.exists() else {}
        for meta in metas:
            resp = proposer.propose(meta["illusion"], meta["inducer_shape"], natural, n, seed)
            mt = re.search(r"\[.*\]", resp, re.S)
            try:
                items = json.loads(mt.group(0)) if mt else []
            except Exception:
                items = []
            prev = cand.get(meta["stem"], [])
            seen = {(i["object"].lower(), i["background"].lower()) for i in prev}
            new = [i for i in items if (i["object"].lower(), i["background"].lower()) not in seen]
            cand[meta["stem"]] = prev + new
            print(f"  propose[{'nat' if natural else 'any'}] {meta['stem']}: +{len(new)} "
                  f"({len(cand[meta['stem']])} total)")
        fn.write_text(json.dumps(cand, indent=2))
