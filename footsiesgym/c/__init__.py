"""Byte-identical C port of FOOTSIES (``csrc/``), for CPU (and CUDA).

The game and env logic live in ``csrc/footsies.h`` as plain C functions that
compile for both CPU and CUDA; this package builds the CPU library on first
use and exposes it to Python:

  * ``footsiesgym.make(backend="c", device=..., tensor_type=...)`` is the
    way in: ``runtime.CRuntime`` on ``FootsiesCEnv`` (CPU, numpy buffers,
    SIMD + OpenMP) or ``cuda.CudaEnv`` (one kernel per step).
  * ``FootsiesFFIEnv`` (needs JAX): the same step as an XLA FFI call, with
    Sardine's functional API. ``tensor_type="jax"`` steps through it, and
    ``make(functional=True)`` returns it for jitted JAX training loops.
  * ``CGames``: batches of ``BattleSimulation``s stepped frame by frame.

The simulation matches the Unity game bit for bit, and the environment
matches ``FootsiesEnv`` on the Unity game server; see
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
