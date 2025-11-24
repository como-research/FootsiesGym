# @OldAPIStack
import random
from typing import List, Optional, Union

import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Box
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights, TensorStructType, TensorType
from ray.rllib.connectors.connector import AgentConnector
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions

from ray.rllib.utils.typing import (
    AgentConnectorDataType,
    TensorType,
)

from footsiesgym.footsies.game import constants, footsies_bot

class FootsiesBot(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.footsies_bot = footsies_bot.FootsiesBot(frame_skip=kwargs.get("frame_skip", 4))

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        if self.config.get("ignore_action_bounds", False) and isinstance(
            self.action_space, Box
        ):
            self.action_space_for_sampling = Box(
                -float("inf"),
                float("inf"),
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_for_sampling = self.action_space

    @override(Policy)
    def init_view_requirements(self):
        super().init_view_requirements()
        # Disable for_training and action attributes for SampleBatch.INFOS column
        # since it can not be properly batched.
        vr = self.view_requirements[SampleBatch.INFOS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False

    @override(Policy)
    def compute_actions_from_input_dict(self, input_dict: Union[SampleBatch, dict[str, TensorStructType]], explore: Optional[bool] = None, timestep: Optional[int] = None, episodes=None, **kwargs) -> tuple[TensorType, List[TensorType], dict[str, TensorType]]:
        """Instead of passing the observation to the FootsiesBot, we pass the infos that are returned from the environment to establish the bots FightState."""
        state_batches = [s for k, s in input_dict.items() if k.startswith("state_in")]
        obs_batch_size = len(tree.flatten(input_dict[SampleBatch.OBS])[0])
        return self.compute_actions(
            input_dict.get(SampleBatch.OBS),
            state_batches,
            prev_action_batch=input_dict.get(SampleBatch.PREV_ACTIONS),
            prev_reward_batch=input_dict.get(SampleBatch.PREV_REWARDS),
            info_batch=input_dict.get(SampleBatch.INFOS),
            explore=explore,
            timestep=timestep,
            episodes=episodes,
            obs_batch_size=obs_batch_size,
            agent_index=input_dict.get(SampleBatch.AGENT_INDEX),
            **kwargs,
        )

    @override(Policy)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[
            List[TensorStructType], TensorStructType
        ] = None,
        prev_reward_batch: Union[
            List[TensorStructType], TensorStructType
        ] = None,
        info_batch: Union[List[TensorStructType], TensorStructType] = None,
        **kwargs,
    ):
        episodes = kwargs.get("episodes", [])        

        action_list = []
        for episode in episodes:
            env_id = episode.env_id
            agent_id = ["p1", "p2"][kwargs.get("agent_index")[0]]
            assert episode.policy_for(agent_id) == "footsies_bot"
            infos = episode.last_info_for(agent_id)

            if infos is None:
                action_list.append(constants.EnvActions.NONE)
            else:
                action_list.append(self.footsies_bot.get_next_input(fight_state_dict=infos, agent_id=agent_id, env_id=env_id))
        return (
            action_list,
            [],
            {},
        )

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        **kwargs,
    ):
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(
                    lambda s: s[None], self.observation_space.sample()
                ),
            }
        )
