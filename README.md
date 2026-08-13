# Gestalt Generator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

<p align="center"><img src="examples/gen100/images/stim_67efcc909b.jpg" width="380"></p>

**Gestalt Generator** produces benchmark stimuli that test perceptual grouping in vision models.
Perceptual grouping — binding separate elements into a whole — underlies how we perceive objects and
contours, and the *Kanizsa* figure is its classic probe: several "pac-man" inducers, arranged so their
bitten edges align, evoke a shape whose contour is never drawn. We build these figures from **real
objects** rather than abstract discs (above: satellite dishes evoking an illusory hexagon), and pair
each stimulus with a machine-readable answer key. The stimulus set reproduces from a single command,
offline, without a GPU or an API key.

*This repository covers stimulus generation. An evaluation harness will follow separately.*

## Installation

```bash
pip install -r requirements.txt
```

The reproduction path is pure Python and needs no GPU, API key, or internet connection.

## Quick start

We ship a fixed set of 100 stimuli. To regenerate it exactly — the configuration pins `seed: 1` —
run:

```bash
python3 run.py configs/gen100.yaml
```

This writes to `out/`:

```
out/comp/<id>.png      100 stimulus images (the benchmark)
out/gt/<id>.json       one answer key per image
out/manifest.csv       id -> silhouette / object / background
```

Filenames are opaque hashes, so the answer is not encoded in the name; each image is joined to its
answer key by `id`. A rendered preview of the set is committed under `examples/gen100/`.

## The task

We consider a task where a model, shown a stimulus, must report what is recorded in its answer key
`out/gt/<id>.json`:

```json
{
  "n_inducers": 6,
  "has_illusion": true,
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

The shipped set uses bezier figures, so `illusory_shape` is always `"irregular"` and includes no
no-illusion controls. For named shapes (`triangle`, `square`, …), use a Kanizsa configuration below.

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
  shadow: 55
  feather: 1.5
```

The number of stimuli is `#silhouettes × n_per_silhouette`. Textures are drawn from a catalog of 16
pre-validated object/background pairs.

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
machine. The silhouettes themselves are regenerated with `python3 main.py`.

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest        # no network, GPU, or token — all external calls are mocked
```
