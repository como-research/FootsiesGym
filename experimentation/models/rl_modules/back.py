import pathlib
from typing import Any, Mapping

import gymnasium as gym
import tree  # pip install dm_tree
from ray.rllib.core import rl_module
from ray.rllib.policy import sample_batch
from ray.rllib.utils.spaces.space_utils import batch as batch_func

from footsies.game import constants as footsies_constants


class BackRLModule(rl_module.RLModule):
    def _forward_inference(self, batch, **kwargs):
        return self._noop_forward(batch, **kwargs)

    def _forward_exploration(self, batch, **kwargs):
        return self._noop_forward(batch, **kwargs)

    def _forward_train(self, *args, **kwargs):
        raise NotImplementedError("Back RLModule: Should not be trained!")

    def output_specs_inference(self):
        return [sample_batch.SampleBatch.ACTIONS]

    def output_specs_exploration(self):
        return [sample_batch.SampleBatch.ACTIONS]

    def _noop_forward(self, batch, **kwargs):
        obs_batch_size = len(
            tree.flatten(batch[sample_batch.SampleBatch.OBS])[0]
        )
        actions = batch_func(
            [footsies_constants.EnvActions.BACK for _ in range(obs_batch_size)]
        )
        return {sample_batch.SampleBatch.ACTIONS: actions}

    def _module_state_file_name(self) -> pathlib.Path:
        return pathlib.Path("noop_rl_module_dummy_state")

    def compile(self, *args, **kwargs):
        """Dummy method for compatibility with TorchRLModule.

        This is hit when RolloutWorker tries to compile TorchRLModule."""

    @classmethod
    def from_model_config(
        cls,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        model_config_dict: Mapping[str, Any],
    ) -> "rl_module.RLModule":
        return cls(action_space)
