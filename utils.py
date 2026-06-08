"""Small shared helpers used across the pipeline (device + config loading)."""

from __future__ import annotations

import torch
import yaml


def pick_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    """Load a YAML config file into a dict."""
    with open(path) as fh:
        return yaml.safe_load(fh)
