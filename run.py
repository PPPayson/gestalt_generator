#!/usr/bin/env python
"""Run a benchmark from a config. Pure Python — never imports modal.

    python run.py configs/example.yaml

The generation provider is chosen in the config (models.image.provider / models.proposer.provider).
Default is hf_api (pure HuggingFace). modal_hf is the maintainer's option and self-contains Modal
(run `modal deploy app.py` once first)."""
import sys
from pipeline import config as C, orchestrate

if __name__ == "__main__":          # guard: multiprocessing (spawn) re-imports __main__ in workers
    cfg = C.load(sys.argv[1] if len(sys.argv) > 1 else "configs/example.yaml")
    orchestrate.run(cfg)
