"""Batches of FOOTSIES games on the C backend (``FootsiesGame`` frame stepping)."""

import numpy as np

from .lib import RAW_STATE, load, ptr, raw_to_fields


class CGames:
    """``n`` ``BattleSimulation``s stepped together, one frame (or StepN) at a time.

    The same interface as ``footsiesgym.jax.validation.JaxReplay``: ``step``
    takes ``(n, 2)`` input bits, and ``fields`` returns the BatchRawState
    fields of every game.
    """

    def __init__(self, num_games: int, lib=None):
        self.lib = lib or load()
        self.num_games = num_games
        self.state = np.zeros((num_games, self.lib.game_size), np.uint8)
        self._raw = np.zeros(num_games, RAW_STATE)
        self.lib.game_new(ptr(self.state), num_games)

    def step(self, inputs: np.ndarray, n_frames: int = 1):
        """StepN with per-game ``(p1, p2)`` input bits, shape ``(n, 2)``."""
        inputs = np.asarray(inputs, np.int32)
        p1 = np.ascontiguousarray(inputs[:, 0])
        p2 = np.ascontiguousarray(inputs[:, 1])
        self.lib.game_step(
            ptr(self.state), ptr(p1), ptr(p2), int(n_frames), self.num_games
        )

    def reset(self, mask: np.ndarray | None = None):
        """Reset() the games where ``mask`` holds (all if None)."""
        if mask is None:
            self.lib.game_reset(ptr(self.state), None, self.num_games)
        else:
            mask = np.ascontiguousarray(mask, np.uint8)
            self.lib.game_reset(ptr(self.state), ptr(mask), self.num_games)

    def raw(self) -> np.ndarray:
        """The ``RAW_STATE`` record of every game."""
        self.lib.game_raw(ptr(self.state), ptr(self._raw), self.num_games)
        return self._raw.copy()

    def fields(self) -> dict[str, np.ndarray]:
        return raw_to_fields(self.raw())


class BatchGames:
    """``CGames`` on the SIMD batches (``csrc/footsies_batch.h``)."""

    def __init__(self, num_games: int, lib=None):
        from .env import env_config, make_config

        self.lib = lib or load()
        self.num_games = num_games
        self._config = env_config(make_config(), self.lib.max_delay_steps)
        self._batch = self.lib.batch_new(ptr(self._config), num_games)
        obs = np.zeros((num_games, 2, self.lib.obs_dim), np.float32)
        self.lib.batch_reset(self._batch, ptr(obs))  # fresh games (game_new)
        self._raw = np.zeros(num_games, RAW_STATE)

    def __del__(self):
        if getattr(self, "_batch", None):
            self.lib.batch_free(self._batch)
            self._batch = None

    def step(self, inputs: np.ndarray, n_frames: int = 1):
        bits = np.ascontiguousarray(inputs, np.int32)
        self.lib.batch_game_step(self._batch, ptr(bits), int(n_frames))

    @property
    def state(self) -> np.ndarray:
        """The games as footsies.h ``Game`` structs."""
        envs = np.zeros((self.num_games, self.lib.env_state_size), np.uint8)
        self.lib.batch_get(self._batch, ptr(envs))
        games = np.zeros((self.num_games, self.lib.game_size), np.uint8)
        self.lib.env_games(ptr(envs), ptr(games), self.num_games)
        return games

    def raw(self) -> np.ndarray:
        games = self.state
        self.lib.game_raw(ptr(games), ptr(self._raw), self.num_games)
        return self._raw.copy()

    def fields(self) -> dict[str, np.ndarray]:
        return raw_to_fields(self.raw())
