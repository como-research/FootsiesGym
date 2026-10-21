"""Byte-identical C port of FOOTSIES (``csrc/``), for CPU (and CUDA).

The game and env logic live in ``csrc/footsies.h`` as plain C functions that
compile for both CPU and CUDA; this package builds the CPU library on first
use and exposes it to Python:

  * ``CGames``: batches of ``BattleSimulation``s stepped frame by frame
    (the interface of ``footsiesgym.jax.validation.JaxReplay``).
  * ``FootsiesCEnv``: ``FootsiesJaxEnv``'s environment, batched, on numpy
    arrays, multi-threaded with OpenMP.
  * ``FootsiesFFIEnv``: the same environment as an XLA FFI call (CPU or
    CUDA), with ``FootsiesJaxEnv``'s functional API, for JAX training loops.

Both match the JAX port (and so the Unity game) bit for bit; see
``testing/test_c_backend.py``.
"""

__all__ = ["CGames", "FootsiesCEnv", "FootsiesFFIEnv"]


def __getattr__(name):  # lazy, so `python -m footsiesgym.c.build` imports nothing
    if name == "CGames":
        from .game import CGames

        return CGames
    if name == "FootsiesCEnv":
        from .env import FootsiesCEnv

        return FootsiesCEnv
    if name == "FootsiesFFIEnv":
        from .jax_env import FootsiesFFIEnv

        return FootsiesFFIEnv
    raise AttributeError(name)
