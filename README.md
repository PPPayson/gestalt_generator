# Perceptual-Grouping Benchmark — stimulus generation

Code to **generate** a benchmark that probes **perceptual grouping** — illusory contours in the style
of the *Kanizsa* illusion, built from **real-world objects**. This repo covers **stimulus generation
only**; an evaluation/scoring harness will ship separately.

You can **reproduce our exact stimulus set** with one command — no GPU, no API key, no internet.

> Commands below use `python3` (works on a stock macOS/Linux Python 3.10+). If your `python` already
> points at Python 3, `python` works too.

---

## 1. The task (what the stimuli are, and what a correct answer is)

**What it measures.** Human vision groups separate elements into a whole. In a Kanizsa figure, several
"pac-man" **inducers** (a disc with a wedge bitten out) are arranged so their aligned bite-edges make
you see an **illusory shape** — a contour that isn't actually drawn. We build the inducers from **real
objects** (poker chips, coins, grapefruit slices, camera lenses, …) so the test uses natural imagery.

**What a stimulus looks like.** N real objects, each rendered as an inducer (the full round object with
a wedge removed), arranged so the wedges point inward. See `examples/gen100/images/` for 100 rendered
examples and `examples/gen100/contact_sheet.png` for an overview.

**What a correct answer is.** Every stimulus ships a machine-readable answer key at `out/gt/<id>.json`
describing the ground truth about that image:

| Field | Meaning (what a model shown the image should report) |
|---|---|
| `n_inducers` | how many inducers / objects are present |
| `has_illusion` | whether an illusory figure is present |
| `illusory_shape` | the illusory figure's shape |
| `nonillusory_shapes.real_life` | the real objects, one per inducer (e.g. `["poker chip", …]`) |
| `nonillusory_shapes.primitive` | their geometric primitive (e.g. `["circle", …]`) |

Example from the shipped `gen100` set:

```json
{
  "id": "stim_ab12cd34ef",
  "n_inducers": 5,
  "has_illusion": true,
  "illusory_shape": "irregular",
  "nonillusory_shapes": {
    "real_life": ["poker chip","poker chip","poker chip","poker chip","poker chip"],
    "primitive": ["circle","circle","circle","circle","circle"]
  }
}
```

Filenames are opaque (`stim_<hash>`) so the answer isn't in the name; join image ↔ ground truth ↔
provenance by `id`.

**Read this before using the set (current scope & limitations):**
- The `gen100` set uses the **bezier "gestalt" silhouettes**, so `illusory_shape` is always
  `"irregular"` and `has_illusion` is always `true` — there are **no named-shape variety and no
  no-illusion control stimuli yet**. For clean named shapes (`triangle`, `square`, …) use a
  **Kanizsa** config (§5); the `has_illusion:false` control condition is reserved for a later release.
- Backgrounds are real photographs of the "fitting" surface. A few background assets were generated as
  a *distinct object* (e.g. a marble **tile** rather than a marble **surface**), which can read as an
  occluder and weaken the illusory contour; the illusory shape is clearest on uniform backgrounds
  (again, the Kanizsa configs). Improving those backgrounds is on the roadmap.

---

## 2. Install

```bash
pip install -r requirements.txt
```

Pure Python (numpy/scipy/Pillow/matplotlib/scikit-image/pyyaml). **No GPU, no Modal, no API key, no
internet** are needed to reproduce our stimuli. (Optional providers for generating *new* textures are
in §6.)

---

## 3. Reproduce our exact stimulus set

```bash
python3 run.py configs/gen100.yaml
```

Fully deterministic — `configs/gen100.yaml` pins `seed: 1`, so you get **the identical set** we did.
It uses the **shipped, pre-validated assets** in `data/assets/`, so no image generation runs. Each run
starts from a clean `out/`, so re-running reproduces exactly (identical files and `id`s).

Output:

```
out/
├── comp/<id>.png        100 stimulus images (the benchmark)
├── gt/<id>.json         100 ground-truth answer keys (§1)
├── manifest.csv         id -> silhouette, object, background, backend, mode
├── silhouettes/         the exact silhouettes used (for completeness)
└── contact_sheet.png    labeled overview (for humans; do NOT show a model)
```

Check: `ls out/comp | wc -l` → `100`.

`out/` is gitignored (regenerable). A rendered preview of our set is committed under `examples/gen100/`
for browsing without running anything — note those previews are **`.jpg`** (compressed for size) while
`out/comp/` are full-resolution **`.png`**; the per-stimulus `gt/*.json` are identical.

### How this reproduces exactly

The stimuli are built from an **LLM proposer** (invents candidate object/background pairs) and a
**diffusion model** (renders each object/background). Those steps are stochastic and run on a GPU, so
they are *not* bit-reproducible on someone else's hardware. To make our exact set reproducible we
**freeze the proposer's output** and ship it:

- `pipeline/pairs_catalog.yaml` — the proposer's candidate pairs **that passed validation** (disc-shape
  + colour contrast). This *is* the proposer output; the reproduce command samples from it.
- `data/assets/` — the diffusion-generated object cutouts and backgrounds for those pairs.

Given those frozen artifacts, the only remaining randomness is **selection** (which silhouettes and
which pairs) and it's fully seeded, and **compositing** is deterministic — so `seed: 1` yields the
identical set every time, offline. To generate a *new* set from scratch (re-run the proposer + the
diffusion model), see §5; it's seeded too, but only reproducible on the same model+GPU.

---

## 4. Generate your own set (edit one config)

Copy `configs/gen100.yaml`, edit it, and run `python3 run.py configs/your.yaml`. Everything is in
that one file:

```yaml
seed: 1                       # change for a different random draw (fully reproducible)

silhouettes:
  - type: gestalt             # our silhouettes (data/silhouettes/*.png from main.py)
    params: {mode: existing, n: 10}   # pick N clean silhouettes (all inducers complete)
  # or a clean named-shape Kanizsa:
  # - {type: kanizsa, params: {n_sides: 3}}   # triangle;  n_sides: 4 -> square, etc.

pairs:
  source: catalog             # object/background pairs from pipeline/pairs_catalog.yaml (pre-validated)
  select: random              # random | contrast
  n_per_silhouette: 10        # textures per silhouette

dress:
  backends: [composite]       # composite = paste real objects (default; offline; pixel-exact)
  shadow: 55                  # inward drop shadow
  feather: 1.5                # soft cutout edge
  crop: true                  # trim empty silhouette margin (never cuts an inducer)
```

**Total stimuli `N = (#silhouettes) × (n_per_silhouette)`.** The shipped catalog has 16 visible pairs
(see `pipeline/pairs_catalog.yaml`), so up to 16 distinct textures per silhouette.

---

## 5. Generate *new* objects/backgrounds (optional — needs a model)

The catalog is a fixed set of textures. To invent new ones, set a **provider** in the config
(`models.image.provider`) and give it credentials; the rest of the pipeline is unchanged. New pairs
are validated (disc-shape + perceptual **colour** contrast, CIELAB ΔE) before use, so camouflage pairs
(e.g. a red apple on red apples) are rejected.

- **`hf_api`** — HuggingFace Inference API. `pip install -r requirements-hf.txt`, `export HF_TOKEN=...`
- **`modal_hf`** — SDXL on Modal GPU. `pip install -r requirements-modal.txt`, then
  `modal deploy app.py` once. Also the only backend for the generative **ControlNet** path
  (`dress.backends: [controlnet]`).
- **bring your own model** — copy `pipeline/providers/template.py`, implement `gen_object`/`gen_bg` to
  return PNG bytes, set `provider: "yourmodule:YourClass"`.

See `configs/example.yaml` (HF) and `pipeline/providers/`.

---

## 6. Regenerate the silhouettes from scratch (optional)

The silhouettes in `data/silhouettes/` come from `main.py` (random bezier-polygon arrangements of
`n_points` inducer circles):

```bash
python3 main.py    # writes img_<randomness>_<nverts>_<npoints>_<edgy>_<rad>.png into data/silhouettes/
```

or from a config: `silhouettes: [{type: gestalt, params: {mode: regenerate}}]`. The filename encodes
the parameters; `n_points` = number of inducers.

---

## 7. How a stimulus is built (pipeline)

1. **Silhouettes** (`main.py` / `pipeline/silhouettes.py`) — inducer arrangements. For our set only
   *clean* silhouettes are used, where every inducer is a complete, separate shape.
2. **Assets** (`pipeline/assets.py`) — one transparent cutout per object and one background, generated
   once via a provider and cached in `data/assets/` (shipped for the catalog).
3. **Validation** (`pipeline/validate.py`) — a pair enters `pipeline/pairs_catalog.yaml` only if the
   object is disc-shaped and stands out from the background by colour.
4. **Compositing** (`pipeline/dress_composite.py`) — each inducer region is filled with the object,
   placed at the region's minimum-enclosing circle so the object's **full rim** lands on the arc and
   only the wedge is removed; soft edge + inward shadow.
5. **Ground truth** (`pipeline/gt.py`) — the answer key per stimulus.

```
main.py                       silhouette generator
run.py                        entrypoint: python3 run.py configs/<name>.yaml   (pure Python)
app.py                        deployable Modal app (only for the modal_hf provider)
configs/
  gen100.yaml                 OUR set (reproduce with this)
  example.yaml                HF-provider example (generate new textures)
  offline.yaml                built-in Kanizsa + shipped assets, zero network
pipeline/
  orchestrate.py  config.py  silhouettes.py  pairs.py  pairs_catalog.yaml  validate.py  assets.py
  geometry.py  dress_composite.py  dress_controlnet.py  gt.py  contact.py  utils.py
  providers/                  base.py + hf_api.py + modal_hf.py + template.py (bring-your-own)
data/
  silhouettes/*.png           860 silhouettes (tracked)
  assets/                     object/background PNGs (catalog set shipped)
examples/gen100/              rendered preview of our set (jpg previews + full GT + manifest)
```

---

## 8. Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

Covers every module and runs with **no network, no Modal, no GPU, no token** (all externals mocked),
including a seed-reproducibility test (same seed → identical bytes).
