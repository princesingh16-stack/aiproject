"""Utility functions for Legal Document LoRA Tagger."""

import os
import logging
import random

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_device():
    """Return best available torch device."""
    if HAS_TORCH:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise RuntimeError("torch is not installed")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    if HAS_NUMPY:
        np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def format_clause(text: str, label: str) -> str:
    """Return text formatted with label token appended."""
    return f"{text.strip()} <{label}>"


def clean_label(label: str) -> str:
    """Strip angle brackets, strip whitespace, replace spaces with underscores."""
    label = label.strip()
    label = label.strip("<>")
    label = label.strip()
    label = label.replace(" ", "_")
    return label


def ensure_dir(path: str):
    """Create directory (and parents) if it does not exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def count_parameters(model) -> dict:
    """
    Count model parameters.
    Returns dict with keys: total, trainable, frozen, trainable_pct.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is not installed")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    trainable_pct = (trainable / total * 100) if total > 0 else 0.0
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": trainable_pct,
    }


def get_logger(name: str) -> logging.Logger:
    """Return a configured standard Python logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
