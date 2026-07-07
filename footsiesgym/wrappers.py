"""Wrappers for integrating FootsiesEnv with external frameworks."""

import numpy as np


def wrap_rllib(env):
    """Wrap a FootsiesEnv (PettingZoo ParallelEnv) for use with RLlib.

    Args:
        env: A FootsiesEnv instance (pettingzoo.ParallelEnv).

    Returns:
        An RLlib MultiAgentEnv wrapping the given environment.

    Raises:
        ImportError: If ray[rllib] is not installed.
    """
    try:
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    except ImportError:
        raise ImportError(
            "ray[rllib] is required to use the RLlib wrapper. "
            "Install it with: pip install 'footsies-gym[rllib]'"
        )

    return ParallelPettingZooEnv(env)


def _get_vectorized_env_class():
    """Build and cache the VectorizedFootsiesRLlibEnv class."""
    from ray.rllib.env.multi_agent_env import MultiAgentEnv

    from footsiesgym.footsies.footsies_env import FootsiesEnv

    class VectorizedFootsiesRLlibEnv(MultiAgentEnv):
        """Wraps a vectorized FootsiesEnv as an RLlib MultiAgentEnv.

        N game instances are multiplexed into a single env with
        agents "p1_0", "p2_0", ..., "p1_{N-1}", "p2_{N-1}".

        Episode lifecycle:
          - When game i finishes, terminateds for p1_i and p2_i
            are True. The vectorized env auto-resets server-side.
          - On the next step, RLlib won't send actions for
            terminated agents. We fill default (NONE) actions and
            return fresh obs as a new episode start.
        """

        def __init__(self, config):
            super().__init__()
            self.num_games = config.get("num_envs", 1)
            self.env = FootsiesEnv(config=config)

            obs_space = self.env.observation_space("p1")
            act_space = self.env.action_space("p1")

            self._agent_ids = set()
            self.observation_spaces = {}
            self.action_spaces = {}
            for i in range(self.num_games):
                for prefix in ["p1", "p2"]:
                    aid = f"{prefix}_{i}"
                    self._agent_ids.add(aid)
                    self.observation_spaces[aid] = obs_space
                    self.action_spaces[aid] = act_space

            # Single-space versions for RLlib internals
            self.observation_space = obs_space
            self.action_space = act_space

        def reset(self, *, seed=None, options=None):
            obs_dict, _ = self.env.reset(seed=seed, options=options)

            obs = {}
            infos = {}
            for i in range(self.num_games):
                obs[f"p1_{i}"] = obs_dict["p1"][i]
                obs[f"p2_{i}"] = obs_dict["p2"][i]
                infos[f"p1_{i}"] = {}
                infos[f"p2_{i}"] = {}
            return obs, infos

        def step(self, action_dict):
            N = self.num_games

            p1_actions = np.zeros(N, dtype=np.int64)
            p2_actions = np.zeros(N, dtype=np.int64)
            for i in range(N):
                if f"p1_{i}" in action_dict:
                    p1_actions[i] = action_dict[f"p1_{i}"]
                if f"p2_{i}" in action_dict:
                    p2_actions[i] = action_dict[f"p2_{i}"]

            obs_dict, rew_dict, term_dict, trunc_dict, _ = self.env.step(
                {"p1": p1_actions, "p2": p2_actions}
            )

            obs = {}
            rewards = {}
            terminateds = {}
            truncateds = {}
            infos = {}

            for i in range(N):
                for prefix in ["p1", "p2"]:
                    aid = f"{prefix}_{i}"
                    obs[aid] = obs_dict[prefix][i]
                    rewards[aid] = float(rew_dict[prefix][i])
                    terminateds[aid] = bool(term_dict[prefix][i])
                    truncateds[aid] = bool(trunc_dict[prefix][i])
                    infos[aid] = {}

            terminateds["__all__"] = False
            truncateds["__all__"] = False

            return obs, rewards, terminateds, truncateds, infos

        def close(self):
            self.env.close()

    return VectorizedFootsiesRLlibEnv


# Lazy singleton so ray import is deferred
_VectorizedClass = None


def VectorizedFootsiesRLlibEnv(config):
    """Factory that returns a VectorizedFootsiesRLlibEnv instance."""
    global _VectorizedClass
    if _VectorizedClass is None:
        _VectorizedClass = _get_vectorized_env_class()
    return _VectorizedClass(config)
