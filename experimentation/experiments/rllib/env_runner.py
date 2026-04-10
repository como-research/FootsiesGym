"""Custom EnvRunner for server-side vectorized FootsiesEnv.

RLlib's default MultiAgentEnvRunner creates N separate env instances via
SyncVectorMultiAgentEnv, each launching its own Unity server. FootsiesEnv
already supports running N games inside a SINGLE Unity server via gRPC
batch APIs. This module provides a custom env runner that uses one
server-side vectorized env but satisfies RLlib's VectorMultiAgentEnv
interface.
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from ray.rllib.callbacks.utils import make_callback
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.env.vector.vector_multi_agent_env import (
    VectorMultiAgentEnv,
)
from ray.rllib.utils.typing import AgentID

from footsiesgym.footsies.footsies_env import FootsiesEnv
from footsiesgym.footsies.game import constants


class _SubEnvProxy:
    """Lightweight proxy satisfying the interface that
    MultiAgentEnvRunner._new_episode() and random-action sampling
    expect from ``self.env.envs[i].unwrapped``.

    All N proxies share the same space objects since all games have
    identical spaces.
    """

    def __init__(
        self,
        possible_agents: List[str],
        observation_spaces: Dict[str, gym.Space],
        action_spaces: Dict[str, gym.Space],
    ):
        self.possible_agents = possible_agents
        self._observation_spaces = observation_spaces
        self._action_spaces = action_spaces
        # MultiAgentEnv base uses these attributes in
        # get_observation_space / get_action_space.
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

    def get_observation_space(self, agent_id: AgentID) -> gym.Space:
        return self._observation_spaces[agent_id]

    def get_action_space(self, agent_id: AgentID) -> gym.Space:
        return self._action_spaces[agent_id]

    @property
    def unwrapped(self):
        return self


class FootsiesVectorEnv(VectorMultiAgentEnv):
    """Wraps a single ``FootsiesEnv(num_envs=N)`` and presents
    the ``VectorMultiAgentEnv`` interface that RLlib expects.

    Converts between:
    - **FootsiesEnv**: batched dicts
      ``{"p1": ndarray(N,86), "p2": ndarray(N,86)}``
    - **VectorMultiAgentEnv**: ``List[Dict[AgentID, value]]``
      of length N
    """

    def __init__(self, footsies_env: FootsiesEnv):
        super().__init__()
        self._footsies = footsies_env
        self.num_envs = footsies_env.num_envs

        # Spaces — shared across all sub-envs.
        obs_space = footsies_env.observation_space("p1")
        act_space = footsies_env.action_space("p1")
        possible = list(footsies_env.possible_agents)  # ["p1", "p2"]

        self.single_observation_spaces = {a: obs_space for a in possible}
        self.single_action_spaces = {a: act_space for a in possible}
        self.single_observation_space = gym.spaces.Dict(
            self.single_observation_spaces
        )
        self.single_action_space = gym.spaces.Dict(self.single_action_spaces)
        # Old API compat attributes.
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

        self.metadata = dict(footsies_env.metadata)
        self.metadata["autoreset_mode"] = "next_step"
        self.render_mode = None

        # Build proxy list so self.envs[i].unwrapped works.
        obs_spaces = {a: obs_space for a in possible}
        act_spaces = {a: act_space for a in possible}
        self.envs = [
            _SubEnvProxy(possible, obs_spaces, act_spaces)
            for _ in range(self.num_envs)
        ]

    # ── reset ─────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        obs_dict, _ = self._footsies.reset(seed=seed, options=options)

        observations = []
        infos = []
        for i in range(self.num_envs):
            observations.append({a: obs_dict[a][i] for a in ("p1", "p2")})
            infos.append({a: {} for a in ("p1", "p2")})
        return observations, infos

    # ── step ──────────────────────────────────────────────────

    def step(
        self,
        actions: List[Dict[AgentID, Any]],
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
        N = self.num_envs

        # Build batched action arrays for FootsiesEnv.
        p1_actions = np.zeros(N, dtype=np.int64)
        p2_actions = np.zeros(N, dtype=np.int64)

        for i in range(N):
            act = actions[i]
            p1_actions[i] = act.get("p1", constants.EnvActions.NONE)
            p2_actions[i] = act.get("p2", constants.EnvActions.NONE)

        # Single batched gRPC call. FootsiesEnv auto-resets
        # done envs and returns post-reset obs for them.
        (
            obs_dict,
            rew_dict,
            term_dict,
            trunc_dict,
            _,
        ) = self._footsies.step({"p1": p1_actions, "p2": p2_actions})

        # Convert batched results to per-env dicts.
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i in range(N):
            observations.append(
                {a: obs_dict[a][i] for a in ("p1", "p2")}
            )
            rewards.append(
                {a: float(rew_dict[a][i]) for a in ("p1", "p2")}
            )
            p1_term = bool(term_dict["p1"][i])
            p2_term = bool(term_dict["p2"][i])
            p1_trunc = bool(trunc_dict["p1"][i])
            p2_trunc = bool(trunc_dict["p2"][i])
            terminateds.append({
                "p1": p1_term,
                "p2": p2_term,
                "__all__": p1_term or p2_term,
            })
            truncateds.append({
                "p1": p1_trunc,
                "p2": p2_trunc,
                "__all__": p1_trunc or p2_trunc,
            })
            infos.append({"p1": {}, "p2": {}})

        return observations, rewards, terminateds, truncateds, infos

    # ── cleanup ───────────────────────────────────────────────

    def close_extras(self, **kwargs):
        self._footsies.close()


class FootsiesEnvRunner(MultiAgentEnvRunner):
    """Custom env runner that uses a single server-side vectorized
    FootsiesEnv instead of N separate env instances.

    Only overrides ``make_env()`` — all sampling/episode logic is
    inherited from ``MultiAgentEnvRunner``.
    """

    def make_env(self):
        # Close existing env if present.
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            del self.env

        # Build EnvContext (same as parent).
        env_ctx = self.config.env_config
        if not isinstance(env_ctx, EnvContext):
            env_ctx = EnvContext(
                env_ctx,
                worker_index=self.worker_index,
                num_workers=self.config.num_env_runners,
                remote=self.config.remote_worker_envs,
            )

        # Merge num_envs into config so FootsiesEnv runs N games
        # in one server.
        footsies_config = dict(env_ctx)
        footsies_config["num_envs"] = self.config.num_envs_per_env_runner

        # Create a single FootsiesEnv with server-side
        # vectorization, then wrap it.
        footsies_env = FootsiesEnv(config=footsies_config)
        self.env = FootsiesVectorEnv(footsies_env)
        self.num_envs = self.env.num_envs

        # Set the flag to reset all envs upon the next sample().
        self._needs_initial_reset = True

        # Fire the on_environment_created callback.
        make_callback(
            "on_environment_created",
            callbacks_objects=self._callbacks,
            callbacks_functions=(self.config.callbacks_on_environment_created),
            kwargs=dict(
                env_runner=self,
                metrics_logger=self.metrics,
                env=self.env.unwrapped,
                env_context=env_ctx,
            ),
        )
