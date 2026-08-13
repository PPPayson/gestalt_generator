#!/usr/bin/env python
"""Thin CLI: `python run.py configs/example.yaml` -> `modal run app.py::run --config ...`"""
import sys, subprocess

cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/example.yaml"
sys.exit(subprocess.run(["modal", "run", "app.py::run", "--config", cfg]).returncode)
