# varying_difficulty_benchmark_100

3 level(s), 1200 images. Every image is `<level>/<variant>_<condition>/f<figure>_<variant>_<condition>.jpg`; `f<figure>` is the same figure in every level, variant and condition.

| level | images | closeness | contrast | support ratio (mean) |
|---|---|---|---|---|
| easy | 400 | 0.75 | 0.0 | 0.69 |
| normal | 400 | 0.5 | 0.0 | 0.52 |
| hard | 400 | 0.25 | 0.5 | 0.37 |

- `plain_*`: black inducers on a flat fill (texture 0). `texture_*`: real object on a real background (texture 3).
- `*_illusory`: no contour drawn (`has_illusion: true`). `*_contour`: the shape's outline drawn (positive control).
- `answers.csv`: one row per image with the answer key and the difficulty parameters; `answers/<stem>.json` is the same per image.
- `preview_<level>.jpg`: contact sheet of the level.

Regenerate: `python3 tools/run_difficulty_set.py --name varying_difficulty_benchmark_100` then `python3 tools/export_benchmark.py out/varying_difficulty_benchmark_100 varying_difficulty_benchmark_100`.
