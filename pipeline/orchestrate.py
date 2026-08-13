"""Pure orchestration — NO `import modal`. Runs every stage from a loaded config.
A generation provider is loaded ONLY when needed (missing assets / propose / controlnet); if all
assets are already present (offline), no provider is loaded at all."""
from . import (silhouettes, pairs as P, assets, validate, dress_composite,
               dress_controlnet, contact, providers)
from .utils import OBJ, BG, slug, load_all_candidates


def _missing(objs, bgs):
    mo = [o for o in objs if not (OBJ / f"{slug(o)}.png").exists()]
    mb = [b for b in bgs if not (BG / f"{slug(b)}.png").exists()]
    return mo, mb


def run(cfg):
    import shutil
    from .utils import OUT
    # start from a clean output so a re-run reproduces EXACTLY (manifest is not append-idempotent)
    for sub in ("comp", "cn", "gt", "silhouettes"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
    (OUT / "manifest.csv").unlink(missing_ok=True)

    print(f"== {cfg['name']} : pairs.source={cfg['pairs']['source']} ==")
    metas = silhouettes.build(cfg)

    if cfg["pairs"]["source"] == "propose":
        proposer = providers.load_proposer(cfg)
        from . import propose
        propose.run(proposer, cfg, metas)
        cand = load_all_candidates()
        objs = {it["object"] for v in cand.values() for it in v}
        bgs = {it["background"] for v in cand.values() for it in v}
        assets.ensure(providers.load_image(cfg), objs, bgs, cfg["assets"]["best_of"], cfg.get("seed", 0))
        prs = validate.run(cfg, metas)
    else:
        prs = P.resolve(cfg, metas)
        objs, bgs = P.needed(prs)
        mo, mb = _missing(objs, bgs)
        if mo or mb:                                      # generation needed -> load provider
            assets.ensure(providers.load_image(cfg), objs, bgs, cfg["assets"]["best_of"], cfg.get("seed", 0))
        else:
            print("  assets: all present (offline, no provider loaded)")

    backends = cfg["dress"]["backends"]
    if "composite" in backends:
        dress_composite.run(cfg, metas, prs)
    if "controlnet" in backends:
        image = providers.load_image(cfg)
        if hasattr(image, "dress"):
            dress_controlnet.run(image, cfg, metas, prs)
        else:
            prov = cfg["models"]["image"]["provider"]
            print(f"  ! WARNING: provider '{prov}' can't do ControlNet (no dress()). "
                  f"Skipping controlnet — use provider modal_hf, or drop it from dress.backends.")
    _copy_silhouettes(metas)
    if cfg["output"]["contact_sheet"]:
        contact.run(cfg)
    from .utils import OUT
    print(f"DONE -> {OUT} (stimuli + gt + manifest + silhouettes)")


def _copy_silhouettes(metas):
    """Copy the silhouettes actually used into out/silhouettes/ for completeness."""
    import shutil
    from .utils import SIL, OUT
    d = OUT / "silhouettes"; d.mkdir(parents=True, exist_ok=True)
    for m in metas:
        for ext in (".png", ".json"):
            src = SIL / f"{m['stem']}{ext}"
            if src.exists():
                shutil.copy(src, d / src.name)
    print(f"  silhouettes: copied {len(metas)} used silhouettes -> {d}")
