"""The C environment as an XLA FFI call, for JAX training loops.

``FootsiesFFIEnv(config).vectorize(num_envs)`` has Sardine's functional
multi-agent API and returns the same values as ``FootsiesCEnv``, bit for
bit; each ``step`` is a single custom call into ``csrc/`` (the CUDA kernel
on GPU, OpenMP on CPU)::

    env = FootsiesFFIEnv(frame_skip=4).vectorize(4096)
    obs, state, infos = env.reset(key)
    obs, state, rewards, terminateds, truncateds, infos = env.step(key, state, actions)

The state is an opaque ``uint8[num_envs, sizeof(EnvState)]`` buffer (updated
in place under ``jit`` with donation), wrapped in ``CEnvState`` so that
``state.t`` and ``state.episode_id`` still read as ``(num_envs,)`` arrays.
"""

import ctypes
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from . import build
from ._struct import Struct, struct
from .env import env_config, make_config

AGENTS = ("p1", "p2")
OBS_DIM = 88


@struct
class CEnvState(Struct):
    data: jax.Array  # uint8[num_envs, sizeof(EnvState)]

    # Both players act every step and every action is always legal.
    active_agents = None
    action_mask = None

    def _int32(self, offset: int) -> jax.Array:
        return jax.lax.bitcast_convert_type(
            self.data[..., offset : offset + 4], jnp.int32
        )

    @property
    def t(self) -> jax.Array:
        return self._int32(_layout()["t"])

    @property
    def episode_id(self) -> jax.Array:
        return self._int32(_layout()["episode_id"])


@functools.cache
def _ffi_cpu():
    lib = ctypes.CDLL(str(build.build_ffi_cpu()))
    for name in ("FootsiesEnvStepCpu", "FootsiesEnvResetCpu"):
        getattr(lib, name).restype = ctypes.c_void_p
    return lib


@functools.cache
def _layout() -> dict[str, int]:
    lib = _ffi_cpu()
    return dict(
        size=lib.footsies_ffi_sizeof_env_state(),
        t=lib.footsies_ffi_offsetof_t(),
        episode_id=lib.footsies_ffi_offsetof_episode_id(),
    )


@functools.cache
def register(cuda: bool | None = None) -> tuple[str, ...]:
    """Register the FFI targets; returns the platforms registered.

    The CPU handlers are always built. The CUDA handlers are built when JAX
    has a GPU backend (``cuda=None``) or when ``cuda=True``.
    """
    lib = _ffi_cpu()
    jax.ffi.register_ffi_target(
        "footsies_env_step", jax.ffi.pycapsule(lib.FootsiesEnvStepCpu), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "footsies_env_reset", jax.ffi.pycapsule(lib.FootsiesEnvResetCpu), platform="cpu"
    )
    platforms = ["cpu"]
    if cuda is None:
        cuda = _has_gpu()
    if cuda:
        cu = ctypes.CDLL(str(build.build_cuda()))
        for name in ("FootsiesEnvStepCuda", "FootsiesEnvResetCuda"):
            getattr(cu, name).restype = ctypes.c_void_p
        if cu.footsies_cuda_sizeof_env_state() != _layout()["size"]:
            raise RuntimeError(
                "EnvState layout differs between the CPU and CUDA builds"
            )
        jax.ffi.register_ffi_target(
            "footsies_env_step",
            jax.ffi.pycapsule(cu.FootsiesEnvStepCuda),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(
            "footsies_env_reset",
            jax.ffi.pycapsule(cu.FootsiesEnvResetCuda),
            platform="CUDA",
        )
        platforms.append("cuda")
    return tuple(platforms)


def _has_gpu() -> bool:
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


class FootsiesFFIEnv:
    """The env config and spaces (Sardine's functional multi-agent API);
    ``vectorize`` gives the env."""

    possible_agents = list(AGENTS)

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any):
        self.config = make_config(config, **kwargs)
        self.max_t: int = self.config["max_t"]
        self.possible_agents = list(AGENTS)
        num_actions = 9 if self.config["use_special_charge_action"] else 6
        self.observation_spaces = {
            a: spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,)) for a in AGENTS
        }
        self.action_spaces = {a: spaces.Discrete(num_actions) for a in AGENTS}

    def obs_dim(self, agent: str) -> int:
        return OBS_DIM

    def action_dim(self, agent: str) -> int:
        return int(self.action_spaces[agent].n)

    def vectorize(self, num_envs: int) -> "VectorizedFootsiesFFIEnv":
        return VectorizedFootsiesFFIEnv(self, num_envs)


class VectorizedFootsiesFFIEnv:
    """``num_envs`` games, stepped by one FFI call."""

    possible_agents = list(AGENTS)

    def __init__(self, env: FootsiesFFIEnv, num_envs: int):
        register()
        self.env = env
        self.num_envs = num_envs
        self.possible_agents = env.possible_agents
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        self.max_t = env.max_t
        size = _layout()["size"]
        max_delay = ctypes.CDLL(str(build.build_cpu())).footsies_max_delay_steps()
        self._config = np.frombuffer(
            env_config(env.config, max_delay).tobytes(), np.uint8
        )
        n = num_envs
        self._state_shape = jax.ShapeDtypeStruct((n, size), jnp.uint8)
        self._step_shapes = (
            self._state_shape,
            jax.ShapeDtypeStruct((n, 2, OBS_DIM), jnp.float32),
            jax.ShapeDtypeStruct((n, 2), jnp.float32),
            jax.ShapeDtypeStruct((n,), jnp.bool_),
            jax.ShapeDtypeStruct((n,), jnp.bool_),
        )

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def action_dim(self) -> int:
        return self.env.action_dim(self.possible_agents[0])

    def reset(self, key=None, initial_state: CEnvState | None = None):
        """Fresh games: ``(obs, state, infos)``. ``key`` is unused (deterministic)."""
        if initial_state is not None:
            raise NotImplementedError("initial_state is not supported by the C env")
        data, obs = jax.ffi.ffi_call(
            "footsies_env_reset",
            (self._state_shape, self._step_shapes[1]),
        )()
        return self._split(obs), CEnvState(data=data), {a: {} for a in AGENTS}

    def step(self, key, state: CEnvState, actions: dict[str, jax.Array]):
        """One step, resetting finished games (``FootsiesCEnv.step``)."""
        return self._step(state, actions, autoreset=True)

    def step_env(self, key, state: CEnvState, actions: dict[str, jax.Array]):
        """One step without the automatic reset
        (``FootsiesCEnv.step(autoreset=False)``)."""
        return self._step(state, actions, autoreset=False)

    def _step(self, state: CEnvState, actions, autoreset: bool):
        acts = jnp.stack(
            [jnp.asarray(actions["p1"]), jnp.asarray(actions["p2"])], axis=-1
        ).astype(jnp.int32)
        data, obs, rew, term, trunc = jax.ffi.ffi_call(
            "footsies_env_step",
            self._step_shapes,
            input_output_aliases={0: 0},
        )(state.data, acts, config=self._config, autoreset=np.int64(autoreset))
        return (
            self._split(obs),
            CEnvState(data=data),
            {"p1": rew[:, 0], "p2": rew[:, 1]},
            {a: term for a in AGENTS},
            {a: trunc for a in AGENTS},
            {a: {} for a in AGENTS},
        )

    @staticmethod
    def _split(obs):
        return {"p1": obs[:, 0], "p2": obs[:, 1]}
