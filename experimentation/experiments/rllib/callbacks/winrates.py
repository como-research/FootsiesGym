from typing import TYPE_CHECKING, Dict, Optional

import gymnasium as gym
import numpy as np

# Import psutil after ray so the packaged version is used.
from ray.rllib.algorithms import callbacks
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import PolicyID

if TYPE_CHECKING:
    from ray.rllib.env.env_runner import EnvRunner


class Winrates(callbacks.DefaultCallbacks):
    def __init__(self, focal_policy_id: str = "focal_policy") -> None:
        super().__init__()
        self.focal_policy_id = focal_policy_id

    def on_episode_end(
        self,
        *,
        episode,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:

        if not base_env.get_sub_environments()[env_index].evaluation:
            return

        env = base_env.get_sub_environments()[env_index]

        last_game_state = env.last_game_state

        p1_policy = episode.policy_for("p1")
        p2_policy = episode.policy_for("p2")

        if self.focal_policy_id == p1_policy:
            opponent_id = p2_policy
            focal_win = last_game_state.player2.is_dead
            opponent_win = last_game_state.player1.is_dead
        elif self.focal_policy_id == p2_policy:
            opponent_id = p1_policy
            focal_win = last_game_state.player1.is_dead
            opponent_win = last_game_state.player2.is_dead
        else:
            return  # focal agent not in this episode

        episode.custom_metrics[
            f"winrates/{self.focal_policy_id}/vs_{p2_policy}_ties"
        ] = [0]

        if focal_win or opponent_win:
            episode.custom_metrics[
                f"winrates/{self.focal_policy_id}/vs_{opponent_id}"
            ] = [focal_win]
        else:
            episode.custom_metrics[
                f"winrates/{self.focal_policy_id}/vs_{opponent_id}"
            ] = [0.5]

    def on_train_result(
        self,
        *,
        algorithm,
        metrics_logger: MetricsLogger | None = None,
        result: Dict,
        **kwargs,
    ) -> None:
        custom_metrics = result[ENV_RUNNER_RESULTS]["custom_metrics"]

        for k, v in custom_metrics.items():
            if "winrates" in k:
                custom_metrics[k] = np.mean(v)
