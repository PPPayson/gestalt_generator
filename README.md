# Gestalt Generator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

<p align="center">
  <img src="examples/gen100/images/illusory/stim_2350676d38.jpg" width="300">
  <img src="examples/gen100/images/control/stim_bec2bca3aa.jpg" width="300">
</p>
<p align="center"><i>Left: illusory condition (the triangle is only implied). Right: matched control (the same figure with its contour drawn).</i></p>

**Gestalt Generator** produces benchmark stimuli that test perceptual grouping in vision models.
Perceptual grouping — binding separate elements into a whole — underlies how we perceive objects and
contours, and the *Kanizsa* figure is its classic probe: several "pac-man" inducers, arranged so their
bitten edges align, evoke a shape whose contour is never drawn. We build these figures from **real
objects** rather than abstract discs, and ship each figure in two matched conditions — **illusory** (no
contour) and a positive-**control** with the shape's contour drawn — so a model that fails on the
illusory version can be distinguished from one that simply cannot name or count the shape. Every
stimulus carries a machine-readable answer key, and the set reproduces from a single command, offline,
without a GPU or an API key.

*This repository covers stimulus generation. An evaluation harness will follow separately.*

## Installation

```bash
pip install -r requirements.txt
```

The reproduction path is pure Python and needs no GPU, API key, or internet connection.

## Quick start

We ship a fixed set of 200 stimuli — 100 figures, each in the illusory and control conditions. To
regenerate it exactly — the configuration pins `seed: 1` — run:

```bash
python3 run.py configs/gen100.yaml
```

This writes to `out/`:

```
out/comp/illusory/<id>.png    100 illusory stimuli
out/comp/control/<id>.png     100 control stimuli (contour drawn)
out/gt/<id>.json              one answer key per image (carries `condition`)
out/manifest.csv              id -> silhouette / object / background / condition
out/silhouettes/  out/control/   the source silhouettes used, split like the input
```

Filenames are opaque hashes, so the answer is not encoded in the name; each image is joined to its
answer key by `id`. A rendered preview of the set is committed under `examples/gen100/`.

## The task

We consider a task where a model, shown a stimulus, must report what is recorded in its answer key
`out/gt/<id>.json`:

```json
{
  "condition": "illusory",
  "n_inducers": 6,
  "has_illusion": true,
  "has_visible_contour": false,
  "illusory_shape": "irregular",
  "nonillusory_shapes": {
    "real_life": ["satellite dish", "satellite dish", "..."],
    "primitive": ["circle", "circle", "..."]
  }
}
```

Each field states one question the model should answer:

| Field | Question |
|---|---|
| `n_inducers` | how many objects are present |
| `has_illusion` | whether an illusory figure appears |
| `illusory_shape` | what the illusory figure is |
| `nonillusory_shapes.real_life` | what the real objects are |
| `nonillusory_shapes.primitive` | their geometric primitive |

Each figure appears in two `condition`s that share the same objects, count, and shape, differing only
in whether the contour is drawn:

- **`illusory`** — the shape is only induced (`has_illusion: true`, `has_visible_contour: false`).
- **`control`** — the shape's contour is drawn (`has_illusion: false`, `has_visible_contour: true`).

The control is a positive control: it isolates perception of the illusory contour from the ability to
name and count the shape. The shipped set uses bezier figures, so `illusory_shape` is always
`"irregular"`; for named shapes (`triangle`, `square`, …), use a Kanizsa configuration below.

## Make your own set

Each benchmark is defined by one configuration file. To create your own, copy `configs/gen100.yaml`,
edit the fields below, and run `python3 run.py configs/your.yaml`.

```yaml
seed: 1                                              # any seed gives a reproducible draw
silhouettes:
  - {type: gestalt, params: {mode: existing, n: 10}} # or  {type: kanizsa, params: {n_sides: 3}}
pairs:
  source: catalog                                    # object/background pairs (pipeline/pairs_catalog.yaml)
  select: random
  n_per_silhouette: 10
dress:
  backends: [composite]                              # composite real objects into the figure
  conditions: [illusory, control]                    # matched pair per figure (drop control for illusory-only)
  contour: auto                                      # control-outline colour: auto | white | black
  shadow: 55
  feather: 1.5
```

The number of stimuli is `#silhouettes × n_per_silhouette × #conditions`. Textures are drawn from a
catalog of 16 pre-validated object/background pairs.

<details>
<summary><b>Generate new objects and backgrounds (requires a model)</b></summary>

The catalog is fixed. To introduce new textures, name a provider in the configuration and supply its
credentials; the rest of the pipeline is unchanged. New pairs are validated for round shape and colour
contrast before use.

- **`hf_api`** — HuggingFace Inference API. `pip install -r requirements-hf.txt`, `export HF_TOKEN=...`
- **`modal_hf`** — SDXL on a Modal GPU. `pip install -r requirements-modal.txt`, `modal deploy app.py`
- **your own** — copy `pipeline/providers/template.py` and set `provider: "yourmodule:YourClass"`

</details>

## Reproducibility

Stimuli are assembled from two stochastic, GPU-bound models — a language model that proposes
object/background pairs, and a diffusion model that renders them. To make the set reproducible we ship
their frozen output: the proposed pairs that passed validation (`pipeline/pairs_catalog.yaml`) and the
rendered cutouts (`data/assets/`). Every remaining step — which silhouettes and pairs are selected, and
how objects are composited — is seeded and deterministic, so `seed: 1` yields the identical set on any
machine. The silhouettes and their matched controls are regenerated with `python3 main.py`.
