"""
Lightweight package init with lazy imports.

Avoid importing heavy training dependencies (e.g., scikit-learn, matplotlib)
at import time so inference can run with only torch/pandas/numpy installed.

All training-related symbols remain available via lazy attribute access.
"""

from importlib import import_module
from typing import Any

# Eagerly expose common inference symbols
from .model import DraftNet  # noqa: F401
from .draftassistant import DraftModel, parse_cardnames, list_sets  # noqa: F401
from .onnx import create_onnx_model, create_all_onnx_models  # noqa: F401

__all__ = [
    # Inference
    "DraftNet",
    "DraftModel",
    "parse_cardnames",
    "list_sets",
    # ONNX
    "create_onnx_model",
    "create_all_onnx_models",
    # Training (lazy)
    "train_model",
    "default_training_pipeline",
    "create_dataset",
    "PickDataset",
    "remove_basics",
]


def __getattr__(name: str) -> Any:
    # Lazy-load training utilities to avoid importing scikit-learn/matplotlib
    if name in {"train_model", "default_training_pipeline"}:
        mod = import_module(".train", __name__)
        return getattr(mod, name)
    if name in {"create_dataset", "PickDataset", "remove_basics"}:
        mod = import_module(".trainingset", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module 'statisticaldrafting' has no attribute {name!r}")


def __dir__():  # help() and dir() friendliness
    return sorted(__all__)
