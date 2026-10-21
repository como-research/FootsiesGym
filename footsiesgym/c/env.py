"""The FOOTSIES environment on the C backend, batched over ``num_envs`` games.

``FootsiesCEnv`` takes ``FootsiesEnv``'s env config and returns the same
values, bit for bit, as the vectorized ``FootsiesEnv`` on the Unity game
server, as numpy arrays::

    env = FootsiesCEnv(num_envs=1024, frame_skip=4, action_delay=8)
    obs = env.reset()                                  # {"p1": (N, 88), "p2": ...}
    obs, rewards, terminateds, truncateds = env.step({"p1": a1, "p2": a2})

``step`` resets a finished game itself: the returned observation is the
fresh game's, the rewards and done flags are the episode's that ended. Pass
``autoreset=False`` to ``step`` to keep the terminal observation instead.

The returned arrays are views of buffers reused by the next ``step``; copy
them to keep them.
"""

import itertools
from typing import Any

import numpy as np

from .lib import ENV_CONFIG, RAW_STATE, load, ptr, raw_to_fields

AGENTS = ("p1", "p2")
# At most three guard-health decreases per player per episode (3 -> 0).
MAX_BUDGET_DEDUCTIONS = 3


def make_config(config: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """``FootsiesEnv``'s env config keys, with its defaults."""
    config = {**(config or {}), **kwargs}
    return dict(
        frame_skip=config.get("frame_skip", 4),
        action_delay=config.get("action_delay", 8),
        max_t=config.get("max_t", 4000),
        win_reward_scaling_coeff=config.get("win_reward_scaling_coeff", 1.0),
        guard_break_reward=config.get("guard_break_reward", 0.0),
        use_reward_budget=config.get("use_reward_budget", False),
        use_special_charge_action=config.get("use_special_charge_action", False),
    )


def reward_table(config: dict[str, Any]) -> np.ndarray:
    """Rewards matching ``FootsiesEnv._step_vectorized`` bit for bit.

    ``FootsiesEnv`` accumulates rewards in float64. A step's rewards depend
    only on whose guard broke, who died and how many guard-break deductions
    each reward budget has had, so the exact float64 operation sequence is
    evaluated once for every combination (then rounded once to float32).

    Indexed [p2_guard_broken, p1_guard_broken, p2_dead, p1_dead,
    p1_deductions, p2_deductions] -> (p1, p2) rewards.
    """
    win = float(config["win_reward_scaling_coeff"])
    gb = float(config["guard_break_reward"])
    assert gb * 3 < win, (
        "Guard break reward total must be less than the win "
        "reward (guard break reward * 3 < win reward)"
    )
    n = MAX_BUDGET_DEDUCTIONS + 1
    table = np.zeros((2, 2, 2, 2, n, n, 2), dtype=np.float64)
    for p2_gb, p1_gb, p2_dead, p1_dead, k1, k2 in itertools.product(
        (0, 1), (0, 1), (0, 1), (0, 1), range(n), range(n)
    ):
        budget = np.full(2, win, dtype=np.float64)
        for _ in range(k1):
            budget[0] -= gb
        for _ in range(k2):
            budget[1] -= gb
        p1, p2 = np.float64(0.0), np.float64(0.0)
        if gb != 0:
            if p2_gb:
                p1 += gb
                p2 -= gb
            if p1_gb:
                p2 += gb
                p1 -= gb
        p1 += budget[0] * np.float64(p2_dead)
        p2 -= budget[0] * np.float64(p2_dead)
        p2 += budget[1] * np.float64(p1_dead)
        p1 -= budget[1] * np.float64(p1_dead)
        table[p2_gb, p1_gb, p2_dead, p1_dead, k1, k2] = (p1, p2)
    return table.astype(np.float32)


def env_config(config: dict[str, Any], max_delay_steps: int) -> np.ndarray:
    """The C ``EnvConfig`` struct for a ``make_config`` dict."""
    frame_skip, delay = int(config["frame_skip"]), int(config["action_delay"])
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    if delay < 0 or delay % frame_skip != 0:
        raise ValueError("action_delay must be divisible by frame_skip")
    if delay // frame_skip > max_delay_steps:
        raise ValueError(
            f"action_delay / frame_skip = {delay // frame_skip} exceeds "
            f"MAX_DELAY_STEPS = {max_delay_steps} (rebuild with -DMAX_DELAY_STEPS=...)"
        )
    c = np.zeros((), ENV_CONFIG)
    c["frame_skip"] = frame_skip
    c["action_delay_frames"] = delay
    c["max_t"] = int(config["max_t"])
    c["guard_break_reward_nonzero"] = float(config["guard_break_reward"]) != 0
    c["use_reward_budget"] = bool(config["use_reward_budget"])
    c["reward_table"] = reward_table(config).ravel()
    return c


def aligned_zeros(shape, dtype, align: int = 64) -> np.ndarray:
    """Zeros whose data starts on an ``align``-byte boundary (full-width
    vector stores never straddle cache lines)."""
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    raw = np.zeros(nbytes + align, np.uint8)
    offset = (-raw.ctypes.data) % align
    return raw[offset : offset + nbytes].view(dtype).reshape(shape)


class FootsiesCEnv:
    """``num_envs`` FOOTSIES games stepped together on the CPU.

    ``impl="batch"`` steps blocks of 16 games with SIMD
    (``csrc/footsies_batch.h``, AVX-512 when the CPU has it) in
    struct-of-arrays state allocated once; ``impl="scalar"`` steps one game
    at a time (``csrc/footsies.h``). ``impl="auto"`` (default) picks "batch"
    when the library was built with AVX-512 and "scalar" otherwise: without
    AVX-512 the portable batches are about 2.5x slower than scalar. Both are
    byte-identical and use OpenMP across cores. Every buffer is preallocated,
    so ``step`` allocates nothing.
    """

    def __init__(
        self,
        num_envs: int,
        config: dict[str, Any] | None = None,
        num_threads: int | None = None,
        lib=None,
        impl: str = "auto",
        **kwargs: Any,
    ):
        if impl not in ("auto", "batch", "scalar"):
            raise ValueError(f"impl must be 'auto', 'batch' or 'scalar', not {impl!r}")
        self.lib = lib or load()
        if impl == "auto":
            impl = "batch" if self.lib.batch_avx512() else "scalar"
        if num_threads is not None:
            self.lib.set_num_threads(int(num_threads))
        self.impl = impl
        self.num_envs = num_envs
        self.config = make_config(config, **kwargs)
        self.max_t = self.config["max_t"]
        self.num_actions = 9 if self.config["use_special_charge_action"] else 6
        self.obs_dim = self.lib.obs_dim
        self.possible_agents = list(AGENTS)
        self._config = env_config(self.config, self.lib.max_delay_steps)

        n = num_envs
        self._state = np.zeros((n, self.lib.env_state_size), np.uint8)  # scalar impl
        self._batch = None
        if impl == "batch":
            self._batch = self.lib.batch_new(ptr(self._config), n)
            if not self._batch:
                raise MemoryError(f"could not allocate {n} envs")
        self._actions = aligned_zeros((n, 2), np.int32)
        self._obs = aligned_zeros((n, 2, self.obs_dim), np.float32)
        self._rew = aligned_zeros((n, 2), np.float32)
        self._term = aligned_zeros(n, np.uint8)
        self._trunc = aligned_zeros(n, np.uint8)

    def __del__(self):
        if getattr(self, "_batch", None):
            self.lib.batch_free(self._batch)
            self._batch = None

    @property
    def state(self) -> np.ndarray:
        """Every env as a footsies.h ``EnvState``: ``uint8[num_envs, sizeof]``
        (a copy for the batch impl)."""
        if self._batch is None:
            return self._state
        out = np.zeros_like(self._state)
        self.lib.batch_get(self._batch, ptr(out))
        return out

    @state.setter
    def state(self, value: np.ndarray):
        value = np.ascontiguousarray(value, np.uint8).reshape(self._state.shape)
        if self._batch is None:
            self._state[:] = value
        else:
            self.lib.batch_set(self._batch, ptr(value))

    def reset(self) -> dict[str, np.ndarray]:
        """Fresh games everywhere; returns their observations."""
        if self._batch is None:
            self.lib.env_reset(ptr(self._state), ptr(self._obs), self.num_envs)
        else:
            self.lib.batch_reset(self._batch, ptr(self._obs))
        return self._split(self._obs)

    def step(self, actions, autoreset: bool = True):
        """One step of every env.

        :param actions: ``{"p1": (N,), "p2": (N,)}`` or an ``(N, 2)`` array.
        :return: ``(obs, rewards, terminateds, truncateds)``, dicts keyed by agent.
        """
        if isinstance(actions, dict):
            self._actions[:, 0] = actions["p1"]
            self._actions[:, 1] = actions["p2"]
        else:
            self._actions[:] = actions
        if self._batch is None:
            self.lib.env_step(
                ptr(self._config),
                ptr(self._state),
                ptr(self._actions),
                ptr(self._obs),
                ptr(self._rew),
                ptr(self._term),
                ptr(self._trunc),
                int(autoreset),
                self.num_envs,
            )
        else:
            self.lib.batch_step(
                self._batch,
                ptr(self._actions),
                ptr(self._obs),
                ptr(self._rew),
                ptr(self._term),
                ptr(self._trunc),
                int(autoreset),
            )
        term = self._term.view(bool)
        trunc = self._trunc.view(bool)
        return (
            self._split(self._obs),
            {"p1": self._rew[:, 0], "p2": self._rew[:, 1]},
            {a: term for a in AGENTS},
            {a: trunc for a in AGENTS},
        )

    def observe(self) -> dict[str, np.ndarray]:
        """The current observations (without stepping)."""
        obs = np.zeros_like(self._obs)
        state = self.state  # keep the copy alive through the call
        self.lib.env_observe(ptr(state), ptr(obs), self.num_envs)
        return self._split(obs)

    def info(self) -> dict[str, np.ndarray]:
        """Per-env ``t`` and ``episode_id``."""
        t = np.zeros(self.num_envs, np.int32)
        episode_id = np.zeros(self.num_envs, np.int32)
        state = self.state
        self.lib.env_info(ptr(state), ptr(t), ptr(episode_id), self.num_envs)
        return {"t": t, "episode_id": episode_id}

    def raw_fields(self) -> dict[str, np.ndarray]:
        """BatchRawState fields of every env's game."""
        games = np.zeros((self.num_envs, self.lib.game_size), np.uint8)
        state = self.state
        self.lib.env_games(ptr(state), ptr(games), self.num_envs)
        raw = np.zeros(self.num_envs, RAW_STATE)
        self.lib.game_raw(ptr(games), ptr(raw), self.num_envs)
        return raw_to_fields(raw)

    @staticmethod
    def _split(obs: np.ndarray) -> dict[str, np.ndarray]:
        return {"p1": obs[:, 0], "p2": obs[:, 1]}
