import collections
from typing import TYPE_CHECKING, Dict, Optional, Union

import gymnasium as gym
from ray.rllib.algorithms import callbacks
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID

from footsiesgym.footsies.game import constants

if TYPE_CHECKING:
    from ray.rllib.env.env_runner import EnvRunner


class ScriptMetrics(callbacks.DefaultCallbacks):
    def __init__(self) -> None:
        self.prev_p1_action_id = None
        self.prev_p2_action_id = None

        # Create action ID to str mapping
        self.action_id_to_str = {
            action_id: action_str
            for action_str, action_id in constants.FOOTSIES_ACTION_IDS.items()
        }

    def on_episode_start(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:

        if worker.env.evaluation:
            return

        episode.user_data["script_metrics"] = collections.defaultdict(
            lambda: 0
        )

    def on_episode_step(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:

        env = worker.env
        if env.evaluation:
            return

        p1_action_id = env.last_game_state.player1.current_action_id
        p2_action_id = env.last_game_state.player2.current_action_id

        # map to string

        # Add 1/2 because we'll track for both agents
        if p1_action_id != self.prev_p1_action_id:
            p1_action_str = self.action_id_to_str[p1_action_id]
            episode.user_data["script_metrics"][p1_action_str] += 1 / 2
        if p2_action_id != self.prev_p2_action_id:
            p2_action_str = self.action_id_to_str[p2_action_id]
            episode.user_data["script_metrics"][p2_action_str] += 1 / 2

        self.prev_p1_action_id = p1_action_id
        self.prev_p2_action_id = p2_action_id

    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:

        if worker.env.evaluation:
            return
        for k, v in episode.user_data["script_metrics"].items():
            episode.custom_metrics[f"script_metrics/{k}"] = [v]
