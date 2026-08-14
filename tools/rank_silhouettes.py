"""Rank gestalt silhouettes by illusory-contour quality, for curating a benchmark set.

Scoring lives in the pipeline (pipeline.silhouettes.rank_score) so this CLI and the
`mode: rank` silhouette selector share one implementation. rank_score combines:
  (1) Sharp contour  -- a weight favouring full-disc inducers (low edgy/rad) over crescents/pinwheels.
  (2) Regular full-frame coverage -- inducers evenly spaced in angle around the centre, at a uniform
      radius, spanning all sides, with the whole arrangement centred (rejects lopsided/clustered
      configurations such as img_0.8_8_3_0.1_0.1).
Geometry is measured directly from the mask, so the ranking does not depend on any model.

    python3 tools/rank_silhouettes.py [N]        # prints the top-N labels (default 30)

The same ranking runs inside the pipeline via `{type: gestalt, params: {mode: rank, n: N}}`.
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.silhouettes import parse_gestalt_name, rank_score  # noqa: E402
from pipeline.utils import SIL                                    # noqa: E402


def main():
    topn = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rows = []
    for p in glob.glob(os.path.join(str(SIL), "*.png")):
        meta = parse_gestalt_name(os.path.basename(p)[:-4])
        if not meta:
            continue
        sc = rank_score(p, meta)
        if sc >= 0:
            rows.append((sc, os.path.basename(p)[4:-4], meta["n_points"]))
    rows.sort(key=lambda t: (-t[0], t[1]))
    print(f"ranked {len(rows)} silhouettes; top {topn}:\n{'label':22} n  score")
    for sc, lab, n in rows[:topn]:
        print(f"{lab:22} {n}  {sc:.3f}")


if __name__ == "__main__":
    main()
