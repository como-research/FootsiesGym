"""The FOOTSIES environment in pure JAX.

``FootsiesJaxEnv`` follows the functional multi-agent API used by Sardine
(and JaxMARL / CoGrid): pure functions of an explicit state, with
per-agent dicts keyed by ``"p1"`` and ``"p2"``::

    env = FootsiesJaxEnv(frame_skip=4, action_delay=8)
    obs, state, infos = env.reset(key)
    obs, state, rewards, terminateds, truncateds, infos = env.step(key, state, actions)

For one game, obs are ``(88,)`` float32 and rewards, terminateds and
truncateds are scalars. ``env.vectorize(num_envs)`` (or ``jax.vmap``) runs
many games and adds a leading ``(num_envs,)`` axis to every value. The game
is deterministic, so ``key`` is accepted for API compatibility only.

``step`` resets a finished game itself: when an episode ends, the returned
obs and state come from a fresh game, and the rewards and done flags from
the episode that ended. ``step_env`` is the same step without the reset.

Within an episode, observations, rewards and done flags match the Unity
game bit for bit, as does a fresh reset. (Unity's vectorized server keeps
hit stun, frame advantage and two proximity-guard flags from the previous
episode when it resets a game; a fresh reset does not.)

Rewards are computed in float64 by ``FootsiesEnv``; here they are float32
(the float64 value rounded once) unless ``jax_enable_x64`` is on, in which
case they are float64 and identical.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from footsiesgym.footsies.game import constants

from ._struct import Struct, struct
from .encoder import ObservationEncoder
from .game import FootsiesGame
from .rewards import RewardFunction

AGENTS = ("p1", "p2")
E = constants.EnvActions


@struct
class EnvState(Struct):
    game: FootsiesGame
    t: jax.Array  # steps taken in this episode
    episode_id: jax.Array  # episodes finished in this env so far
    action_queue: jax.Array  # (2, K) delayed actions
    queue_head: jax.Array
    prev_selected: jax.Array  # (2,)
    holding_special: jax.Array  # (2,) bool
    budget_deductions: jax.Array  # (2,) guard-break deductions this episode
    last_guard_health: jax.Array  # (2,) guard health at the previous step

    # Both players act every step and every action is always legal.
    active_agents = None
    action_mask = None


class FootsiesJaxEnv:
    """One FOOTSIES game as pure ``reset``/``step`` functions.

    Takes the ``FootsiesEnv`` config keys that affect the game, as a dict or
    as keyword arguments: ``frame_skip``, ``action_delay``, ``max_t``,
    ``win_reward_scaling_coeff``, ``guard_break_reward``,
    ``use_reward_budget`` and ``use_special_charge_action``.
    """

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any):
        config = {**(config or {}), **kwargs}
        self.frame_skip: int = config.get("frame_skip", 4)
        self.action_delay_frames: int = config.get("action_delay", 8)
        assert (
            self.action_delay_frames % self.frame_skip == 0
        ), "action_delay must be divisible by frame_skip"
        self.action_delay_steps = self.action_delay_frames // self.frame_skip
        self.max_t: int = config.get("max_t", 4000)
        self.encoder = ObservationEncoder(num_actions=9)  # all 9 env actions, always
        self.reward_fn = RewardFunction(
            win_reward=config.get("win_reward_scaling_coeff", 1.0),
            guard_break_reward=config.get("guard_break_reward", 0.0),
            use_reward_budget=config.get("use_reward_budget", False),
        )

        self.possible_agents = list(AGENTS)
        num_actions = 9 if config.get("use_special_charge_action", False) else 6
        obs_shape = (self.encoder.observation_size,)
        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(num_actions) for agent in self.possible_agents
        }

    def obs_dim(self, agent: str) -> int:
        return int(np.prod(self.observation_spaces[agent].shape))

    def action_dim(self, agent: str) -> int:
        return int(self.action_spaces[agent].n)

    def vectorize(self, num_envs: int) -> "VectorizedFootsiesJaxEnv":
        """``num_envs`` games stepped together (``jax.vmap``)."""
        return VectorizedFootsiesJaxEnv(self, num_envs)

    # ── Episode ──────────────────────────────────────────────────────────

    def reset(self, key=None, initial_state: EnvState | None = None):
        """A fresh game, or ``initial_state`` if given: ``(obs, state, infos)``."""
        state = self._fresh_state() if initial_state is None else initial_state
        return self._observe(state), state, {agent: {} for agent in AGENTS}

    def step(self, key, state: EnvState, actions: dict[str, jax.Array]):
        """One step, resetting the game if the episode ends.

        Returns ``(obs, state, rewards, terminateds, truncateds, infos)``.
        """
        obs, state, rewards, terminateds, truncateds, infos = self.step_env(
            key, state, actions
        )
        episode_over = terminateds["p1"] | truncateds["p1"]
        fresh = self._fresh_state().replace(episode_id=state.episode_id + 1)
        state = fresh.where(episode_over, state)
        obs = jax.tree.map(
            lambda new, old: jnp.where(episode_over, new, old),
            self._observe(fresh),
            obs,
        )
        return obs, state, rewards, terminateds, truncateds, infos

    def step_env(self, key, state: EnvState, actions: dict[str, jax.Array]):
        """One step without the automatic reset (same outputs as ``step``)."""
        selected = jnp.stack([actions["p1"], actions["p2"]]).astype(jnp.int32)
        to_execute, queue, head = self._delay(state, selected)
        to_execute, holding = self._resolve_special_charge(
            to_execute, state.holding_special
        )
        game = state.game.step_n(*self._input_bits(to_execute), self.frame_skip)

        t = state.t + 1
        guard = game.fighters.guard_health
        dead = game.fighters.vital_health <= 0
        rewards, deductions = self.reward_fn(
            state.last_guard_health, guard, dead, state.budget_deductions
        )
        state = state.replace(
            game=game,
            t=t,
            action_queue=queue,
            queue_head=head,
            prev_selected=selected,
            holding_special=holding,
            budget_deductions=deductions,
            last_guard_health=guard,
        )
        terminated = dead[0] | dead[1]
        truncated = t >= self.max_t
        return (
            self._observe(state),
            state,
            {agent: rewards[i] for i, agent in enumerate(AGENTS)},
            {agent: terminated for agent in AGENTS},
            {agent: truncated for agent in AGENTS},
            {agent: {} for agent in AGENTS},
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _fresh_state(self) -> EnvState:
        game = FootsiesGame.new()
        return EnvState(
            game=game,
            t=jnp.int32(0),
            episode_id=jnp.int32(0),
            action_queue=jnp.zeros((2, max(self.action_delay_steps, 1)), jnp.int32),
            queue_head=jnp.int32(0),
            prev_selected=jnp.zeros(2, jnp.int32),
            holding_special=jnp.zeros(2, bool),
            budget_deductions=jnp.zeros(2, jnp.int32),
            last_guard_health=game.fighters.guard_health,
        )

    def _delay(self, state: EnvState, selected):
        """Action delay: a FIFO of ``action_delay_steps`` steps.

        Returns (actions to execute now, new queue, new head).
        """
        if self.action_delay_frames == 0:
            return selected, state.action_queue, state.queue_head
        head = state.queue_head
        to_execute = state.action_queue[:, head]
        queue = state.action_queue.at[:, head].set(selected)
        return to_execute, queue, (head + 1) % self.action_delay_steps

    @staticmethod
    def _resolve_special_charge(actions, holding):
        """Toggle special charge and map env actions to the base actions 0-5.

        Returns (base actions to execute, new holding state).
        """
        is_special = (
            (actions == E.SPECIAL_CHARGE)
            | (actions == E.FORWARD_SPECIAL_CHARGE)
            | (actions == E.BACK_SPECIAL_CHARGE)
        )
        holding = jnp.where(is_special, ~holding, holding)
        base = jnp.where(
            actions == E.SPECIAL_CHARGE,
            E.NONE,
            jnp.where(
                actions == E.FORWARD_SPECIAL_CHARGE,
                E.FORWARD,
                jnp.where(actions == E.BACK_SPECIAL_CHARGE, E.BACK, actions),
            ),
        )
        charge = jnp.asarray(constants.CHARGE_ACTION_LUT, jnp.int32)
        return jnp.where(holding, charge[jnp.clip(base, 0, 5)], base), holding

    @staticmethod
    def _input_bits(actions):
        """Base env actions (P1, P2) -> game input bits (P2 faces left)."""
        return (
            jnp.asarray(constants.P1_ENV_TO_BITS, jnp.int32)[actions[0]],
            jnp.asarray(constants.P2_ENV_TO_BITS, jnp.int32)[actions[1]],
        )

    def _observe(self, state: EnvState) -> dict[str, jax.Array]:
        raw = state.game.raw_state()
        return self.encoder.encode(raw, state.prev_selected, state.holding_special)


class VectorizedFootsiesJaxEnv:
    """``num_envs`` FOOTSIES games with one key and batched states and actions.

    Same API as ``FootsiesJaxEnv``; every value has a leading ``(num_envs,)``
    axis. Each game gets its own key, split from the one passed in.
    """

    def __init__(self, env: FootsiesJaxEnv, num_envs: int):
        self.env = env
        self.num_envs = num_envs
        self.possible_agents = env.possible_agents
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        self.max_t = env.max_t

    @property
    def obs_dim(self) -> int:
        return self.env.obs_dim(self.possible_agents[0])

    @property
    def action_dim(self) -> int:
        return self.env.action_dim(self.possible_agents[0])

    def reset(self, key, initial_state: EnvState | None = None):
        if initial_state is not None:
            return jax.vmap(lambda state: self.env.reset(None, state))(initial_state)
        return jax.vmap(self.env.reset)(jax.random.split(key, self.num_envs))

    def step(self, key, state: EnvState, actions: dict[str, jax.Array]):
        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self.env.step)(keys, state, actions)

    def step_env(self, key, state: EnvState, actions: dict[str, jax.Array]):
        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self.env.step_env)(keys, state, actions)
