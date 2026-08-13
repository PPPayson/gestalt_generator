# Example benchmark — 100 stimuli (gen100)

100 perceptual-grouping stimuli = **10 gestalt silhouettes × 10 random textures**, produced by:

```
python run.py configs/gen100.yaml
```

Reproducible: the config uses `seed: 0`, so this command regenerates the identical set at full
resolution. What's committed here:

- `images/` — the 100 stimuli as **512px JPEG previews** (viewable on GitHub; full-res 1024px PNGs
  are regenerable via the command above).
- `gt/<id>.json` — the **ground truth** answer key per stimulus (n_inducers, has_illusion,
  illusory_shape, nonillusory_shapes). This is the exact, full-fidelity data.
- `manifest.csv` — `id -> silhouette / object / background / backend / mode` (provenance).
- `contact_sheet.png` — labeled overview of the set.

The stimulus filenames are opaque `stim_<hash>` so a tested model can't read the answer off the name;
join `images/` ↔ `gt/` ↔ `manifest.csv` by `id`.

Textures come from `pipeline/pairs_catalog.yaml` (pre-validated for disc-shape + contrast). The object
cutouts were generated on Modal (SDXL + rembg); the silhouettes are from `main.py`.
