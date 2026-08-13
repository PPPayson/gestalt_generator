# Gestalt Generator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

<p align="center"><img src="examples/gen100/images/stim_67efcc909b.jpg" width="380"></p>

**Gestalt Generator** builds benchmark stimuli that test **perceptual grouping** in vision models:
Kanizsa-style **illusory contours** made from real objects. Several objects are cut into "pac-man"
inducers and arranged so their aligned bites make you see a shape that isn't actually drawn (above:
satellite dishes forming an illusory hexagon). Every stimulus ships a ground-truth answer key, and the
exact set reproduces from **one command — offline, no GPU or API key.**

*This repo covers stimulus generation only; an evaluation harness will ship separately.*

## Installation

```bash
pip install -r requirements.txt      # pure Python; no GPU, API key, or internet
```

## Quick start

Reproduce our 100-stimulus benchmark (deterministic — the config pins `seed: 1`):

```bash
python3 run.py configs/gen100.yaml
```

This writes to `out/`:

```
out/comp/<id>.png      100 stimulus images  (the benchmark)
out/gt/<id>.json       ground-truth answer key, one per image
out/manifest.csv       id -> silhouette / object / background
```

Filenames are opaque hashes so the answer isn't in the name — join image ↔ answer key by `id`. A
rendered preview of the set is committed in `examples/gen100/`.

## The task

Shown a stimulus, a model should report what's in its `out/gt/<id>.json`:

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

| Field | The question it answers |
|---|---|
| `n_inducers` | how many objects/inducers are there |
| `has_illusion` | is there an illusory figure |
| `illusory_shape` | what shape the illusory figure is |
| `nonillusory_shapes.real_life` | what the real objects are |
| `nonillusory_shapes.primitive` | their geometric primitive |

> The `gen100` set uses bezier silhouettes, so `illusory_shape` is always `"irregular"` and there are
> no no-illusion controls yet. For clean named shapes (`triangle`, `square`, …) use a **Kanizsa** config.

## Make your own set

Copy `configs/gen100.yaml`, edit it, and run `python3 run.py configs/your.yaml`. Everything is in the
one file:

```yaml
seed: 1                                              # any seed -> a reproducible draw
silhouettes:
  - {type: gestalt, params: {mode: existing, n: 10}} # or  {type: kanizsa, params: {n_sides: 3}}
pairs:
  source: catalog                                    # object/background pairs (pipeline/pairs_catalog.yaml)
  select: random
  n_per_silhouette: 10
dress:
  backends: [composite]                              # paste real objects (offline, pixel-exact)
  shadow: 55
  feather: 1.5
```

Number of stimuli = `#silhouettes × n_per_silhouette`. The shipped catalog has 16 pre-validated pairs.

<details>
<summary><b>Generate new objects &amp; backgrounds (needs a model)</b></summary>

The catalog is fixed. To invent new textures, set a provider in the config and give it credentials —
the rest of the pipeline is unchanged. New pairs are auto-validated (round shape + colour contrast).

- **`hf_api`** — HuggingFace Inference API. `pip install -r requirements-hf.txt`, `export HF_TOKEN=...`
- **`modal_hf`** — SDXL on Modal GPU. `pip install -r requirements-modal.txt`, `modal deploy app.py`
- **your own** — copy `pipeline/providers/template.py`, set `provider: "yourmodule:YourClass"`

</details>

## How it reproduces

Generation uses an LLM (invents object/background pairs) and a diffusion model (renders them) — both
stochastic and GPU-bound. We ship their **frozen output** (`pipeline/pairs_catalog.yaml` + the cutouts
in `data/assets/`), so the only remaining randomness is seeded selection and the compositing is exact.
`seed: 1` therefore yields the identical set every time, offline. Silhouettes are regenerable with
`python3 main.py`.

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest        # no network / GPU / token — all externals mocked
```
