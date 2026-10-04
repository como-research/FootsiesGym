"""Bit-identical JAX port of the FOOTSIES game (Unity ``BattleSimulation``).

Modules, from the game outwards:

  * ``frame_data``: fighter F00's frame data from the Unity assets, as
    read-only lookup tables.
  * ``fighter``: ``Fighter``, port of ``Fighter.cs`` (inputs, action
    requests, movement, boxes, damage).
  * ``game``: ``FootsiesGame``, port of ``BattleSimulation.cs``
    (``new``, ``reset``, ``step``, ``step_n``, ``raw_state``).
  * ``encoder``: ``ObservationEncoder``, port of ``VectorizedEncoder``.
  * ``rewards``: ``RewardFunction``, the ``FootsiesEnv`` reward arithmetic.
  * ``env``: ``FootsiesJaxEnv``, the functional environment (pure
    ``reset``/``step``, Sardine-compatible; ``vectorize(num_envs)`` batches).
  * ``validation``: frame-by-frame comparison against the Unity binary, and
    a JAX stand-in for the vectorized server used in tests.

State classes (``Fighter``, ``FootsiesGame``, ``EnvState``) are immutable
JAX pytrees (``_struct``): methods return new instances, and work under
``jax.jit`` and ``jax.vmap``.

Use ``FootsiesJaxEnv`` directly for JAX training loops (for example
``Experiment().environment(env=FootsiesJaxEnv, ...)`` in Sardine), or
``footsiesgym.make(backend="jax")`` for a single-game PettingZoo env.
"""

try:
    import jax  # noqa: F401
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "footsiesgym.jax needs JAX >= 0.11.2. "
        'Install it with: pip install "footsies-gym[jax]"',
        name=e.name,
    ) from e

from .env import FootsiesJaxEnv  # noqa: E402
from .game import FootsiesGame  # noqa: E402

__all__ = ["FootsiesGame", "FootsiesJaxEnv"]
