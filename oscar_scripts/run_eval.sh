#!/bin/sh
export HUGGINGFACE_HUB_CACHE=/oscar/data/tserre/Users/pzhou10/CVM/onevision/pretrained_ckpts

conda run -p ../CVM/onevision_venv python -m src.eval.run_eval