"""Pick the figures for a MATCHED multi-level set (the same figures at several closeness values).

Thin CLI over pipeline.silhouettes.sample_figures: keeps drawing seeds over the parameter grid
until `--n` figures pass the filter at the judged closeness (`--judge`, the "normal" level) AND
are feasible + clean at every other level (`--levels`). Small inducers (closeness 0.25) sit on
near-straight contour stretches and become half discs, which the pac-man "flat cap" test would
reject, but that IS what the hard level looks like -- hence judge once, require feasibility elsewhere.
Writes a YAML `figures:` list for tools/make_difficulty_configs.py.

    python3 tools/pick_difficulty_figures.py --n 100 --judge 0.5 --levels 0.25 0.75 \\
        --quota 3=0.15 4=0.35 5=0.35 6=0.15 --filter min_inducers=3 all_sides=false

A config can do the same for ONE level with no tool at all:
    silhouettes: [{type: generate, params: {n: 100, closeness: 0.5, filter: {min_inducers: 3, all_sides: false}}}]
"""
import sys, pathlib, argparse, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pipeline.silhouettes import sample_figures  # noqa: E402


def _kv(items, cast):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[cast(k)] = v
    return out


def _val(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    return float(v) if "." in v else int(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--judge", type=float, default=0.5, help="closeness at which figures are scored")
    ap.add_argument("--levels", type=float, nargs="*", default=[0.25, 0.75],
                    help="other closeness values the figure must be feasible + clean at")
    ap.add_argument("--quota", nargs="*", default=["3=0.15", "4=0.35", "5=0.35", "6=0.15"],
                    help="inducer-count mix n_points=fraction (or absolute counts)")
    ap.add_argument("--filter", nargs="*",
                    default=["min_inducers=3", "all_sides=false", "max_flat_frac=0.2"],
                    help="rank_score thresholds, e.g. flat_solidity=0.95 max_gap=1.8")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-seeds", type=int, default=200)
    ap.add_argument("--out", default="out/difficulty_figures.yaml")
    args = ap.parse_args()
    quota = {k: float(v) for k, v in _kv(args.quota, int).items()}
    filt = {k: _val(v) for k, v in _kv(args.filter, str).items()}
    figs = sample_figures(args.n, closeness=args.judge, quota=quota, filter=filt, seed=args.seed,
                          also_closeness=args.levels, max_seeds=args.max_seeds)
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# {len(figs)} figures [randomness, n_vertices, n_points, edgy, rad, seed]; judged at "
                f"closeness {args.judge}, feasible at {args.levels}; filter {filt}; quota {quota}\n")
        f.write("figures:\n")
        for fig in figs:
            f.write(f"  - {json.dumps(fig)}\n")
    print(f"wrote {out} ({len(figs)} figures)")


if __name__ == "__main__":
    main()
