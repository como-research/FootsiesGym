"""The array types environments return (config key ``tensor_type``).

``"numpy"`` (default), ``"torch"`` or ``"jax"``. Conversions from numpy are
zero-copy for torch (``torch.from_numpy``); JAX arrays are always copies.
torch and JAX are imported only when asked for.
"""

from typing import Any

import numpy as np

TENSOR_TYPES = ("numpy", "torch", "jax")


def check_tensor_type(tensor_type: str) -> str:
    if tensor_type not in TENSOR_TYPES:
        raise ValueError(
            f"tensor_type must be one of {TENSOR_TYPES}, not {tensor_type!r}"
        )
    if tensor_type == "torch":
        _import("torch")
    elif tensor_type == "jax":
        _import("jax")
    return tensor_type


def _import(name: str):
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise ImportError(
            f"tensor_type={name!r} needs {name} installed (pip install {name})"
        ) from e


def from_numpy(x: np.ndarray, tensor_type: str):
    """A host numpy array as ``tensor_type`` (on the CPU)."""
    if tensor_type == "numpy":
        return x
    if tensor_type == "torch":
        import torch

        return torch.from_numpy(x)
    import jax

    return jax.device_put(x, jax.devices("cpu")[0])


def to_numpy(x: Any) -> np.ndarray:
    """numpy, torch, JAX, lists and scalars as a host numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch
        return x.detach().cpu().numpy()
    return np.asarray(x)


def convert(values: dict, tensor_type: str) -> dict:
    """Every numpy array in a ``{agent: value}`` dict as ``tensor_type``;
    other values (python scalars, info dicts) are left as they are."""
    if tensor_type == "numpy":
        return values
    return {
        k: from_numpy(v, tensor_type) if isinstance(v, np.ndarray) else v
        for k, v in values.items()
    }
