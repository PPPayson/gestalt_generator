# Gestalt Generator + Perceptual-Grouping Dressing Pipeline

Two halves in one repo:

1. **`main.py`** — the silhouette generator (bezier-polygon Gestalt figures). Produces
   `img_{randomness}_{n_vertices}_{n_points}_{edgy}_{rad}.png` into `data/silhouettes/`.
2. **The dressing pipeline** (`app.py` + `pipeline/`) — dresses those silhouettes with real
   objects (moons, poker chips, satellite dishes) so aligned inducers form an **illusory contour**,
   producing benchmark stimuli that test **perceptual grouping** in vision models. Two backends
   (pixel-exact **composite**, generative **ControlNet**), opaque filenames, and a private
   **ground-truth** answer key per stimulus.

```
python main.py                       # (re)generate the silhouette grid into data/silhouettes/
python run.py configs/gestalt.yaml   # dress N of them into a benchmark
python run.py configs/example.yaml   # or use the built-in Kanizsa silhouettes
```

## Install

```
pip install -r requirements.txt
modal token new        # one-time Modal auth (GPU stages)
```

## Layout

```
main.py                    silhouette generator (generate() importable; python main.py unchanged)
run.py                     thin CLI -> modal run app.py::run --config <yaml>
app.py                     ONE Modal app: LLM cls (Qwen) + SDXL cls (SDXL+ControlNet+rembg) + run()
configs/
  gestalt.yaml             dress the friend's gestalt silhouettes
  example.yaml             dress built-in Kanizsa silhouettes
pipeline/
  config.py   silhouettes.py   propose.py   assets.py   validate.py   pairs.py
  pairs_catalog.yaml   geometry.py   dress_composite.py   dress_controlnet.py   gt.py
  contact.py   utils.py
data/
  silhouettes/             TRACKED: generated silhouettes (main.py output + registry) + sidecars
  assets/                  gitignored: generated object cutouts + backgrounds
out/                       gitignored: our benchmark output
  comp/  cn/               stimuli (opaque names)
  gt/<id>.json             ground truth (answer key)
  manifest.csv             provenance
  contact_sheet.png        labeled overview (maintainer only)
```

## Silhouette sources (config `silhouettes:` specs)

- **`gestalt`** — the friend's silhouettes. `{type: gestalt, params: {mode, n, seed}}`.
  - `mode: existing` — seeded random sample of **N** from `data/silhouettes/*.png`.
  - `mode: regenerate` — run `main.generate(...)` first (optional subset grid via `regenerate:`), then sample N.
- **`kanizsa`** — built-in registry, clean illusory n-gons: `{type: kanizsa, params: {n_sides: 3}}`.
- **`dropin`** — explicit files already in `data/silhouettes/`: `{type: dropin, params: {files: [...]}}`.

Inducers are detected whether black-on-white or white-on-black. `main.py` behaviour is unchanged
apart from its default output folder (`data/silhouettes/`); the generation grid and results are identical.

## Pair source (`pairs.source`)

- `catalog` (default) — pick from `pipeline/pairs_catalog.yaml`; filter by `natural`, `min_contrast`,
  `objects`; take `n_per_silhouette`. **No LLM run.**
- `propose` — Qwen invents pairs -> assets generated -> shape+contrast validated.
- `explicit` — use your own `pairs.explicit` list.

Assets are generated on first use and cached, so catalog pairs render fast.

## Output

- `out/comp/<id>.png`, `out/cn/<id>.png` — stimuli with **opaque** names.
- `out/manifest.csv` — `id -> silhouette/object/background/backend/mode` (provenance).
- `out/gt/<id>.json` — **ground truth** (the graded answers), joined to the manifest by `id`:

  ```json
  {"id": "stim_...", "n_inducers": 3, "has_illusion": true,
   "illusory_shape": "triangle",
   "nonillusory_shapes": {"real_life": ["moon","moon","moon"],
                           "primitive": ["circle","circle","circle"]}}
  ```

  `illusory_shape` is a clean name for Kanizsa (`triangle`, `square`, …) and `"irregular"` for the
  bezier gestalt figures. `has_illusion` is `true` for everything generated here; it's reserved for
  future non-illusory control stimuli (which would set it `false` and `illusory_shape: "none"`).
