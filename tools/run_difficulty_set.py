"""Generate every level of a multi-level set with one command and one overall progress bar.

    python3 tools/run_difficulty_set.py                       # configs/<name>_{easy,normal,hard}.yaml -> out/<name>/<level>
    python3 tools/run_difficulty_set.py --levels easy hard    # a subset
    python3 tools/run_difficulty_set.py --name my_set

Each level runs as its own `run.py` process with GESTALT_OUT=<out>/<level>, so the pipeline's own
stage bars nest under this script's level bar (GESTALT_PROGRESS_DEPTH). Set GESTALT_PROGRESS=0 to
silence all bars (e.g. in a batch job log).
"""
import sys, os, pathlib, argparse, subprocess
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=["easy", "normal", "hard"])
    ap.add_argument("--name", default="varying_difficulty_benchmark_100",
                    help="set name: runs configs/<name>_<level>.yaml into out/<name>/<level>")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--out", default=None, help="output root (default out/<name>)")
    args = ap.parse_args()
    from pipeline.utils import bar
    args.out = args.out or f"out/{args.name}"
    top = bar(len(args.levels), "levels", unit="level")
    failed = []
    for lvl in args.levels:
        top.set_postfix_str(lvl)
        cfg = (ROOT / args.configs / f"{args.name}_{lvl}.yaml").resolve()
        env = {**os.environ, "GESTALT_OUT": str((ROOT / args.out / lvl).resolve()),
               "GESTALT_PROGRESS_DEPTH": str(int(os.environ.get("GESTALT_PROGRESS_DEPTH", "0")) + 1)}
        rc = subprocess.call([sys.executable, str(ROOT / "run.py"), str(cfg)], env=env, cwd=ROOT)
        if rc != 0:
            failed.append(lvl)
            print(f"  ! level {lvl} failed (exit {rc})")
        top.update(1)
    top.set_postfix_str("done" if not failed else f"FAILED: {failed}"); top.close()
    print(f"levels: {len(args.levels) - len(failed)}/{len(args.levels)} ok -> {args.out}/<level>/")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
