"""Winrate tracking callback for the new RLlib API stack (v2).

Compatible with ``enable_env_runner_and_connector_v2=True`` and
``MultiAgentEnvRunner``.  Uses ``MetricsLogger`` instead of
``episode.custom_metrics``.
"""

from typing import TYPE_CHECKING, Optional

import gymnasium as gym
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

if TYPE_CHECKING:
    from ray.rllib.env.env_runner import EnvRunner


class WinratesV2(RLlibCallback):
    """Logs per-opponent winrates for the focal policy during
    evaluation episodes.

    Metrics are logged as::

        winrates/<focal_id>/vs_<opponent_id>

    Values are 1.0 (win), 0.0 (loss), or 0.5 (draw/timeout),
    reduced to a mean over a sliding window.
    """

    def __init__(
        self,
        focal_policy_id: str = "focal_policy",
        window: int = 100,
    ) -> None:
        super().__init__()
        self.focal_policy_id = focal_policy_id
        self.window = window

    def on_episode_end(
        self,
        *,
        episode,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        **kwargs,
    ) -> None:
        # Only track during evaluation.
        if not env_runner.config.in_evaluation:
            return

        # Determine which agent the focal policy controls.
        p1_module = episode.module_for("p1")
        p2_module = episode.module_for("p2")

        if p1_module == self.focal_policy_id:
            focal_agent = "p1"
            opponent_id = p2_module
        elif p2_module == self.focal_policy_id:
            focal_agent = "p2"
            opponent_id = p1_module
        else:
            return  # Focal policy not in this episode.

        # Determine outcome from the focal agent's return.
        # Positive return → win, negative → loss, zero → draw.
        focal_return = episode.agent_episodes[focal_agent].get_return()

        if focal_return > 0:
            result = 1.0
        elif focal_return < 0:
            result = 0.0
        else:
            result = 0.5

        metrics_logger.log_value(
            f"winrates/{self.focal_policy_id}/vs_{opponent_id}",
            result,
            reduce="mean",
            window=self.window,
        )
