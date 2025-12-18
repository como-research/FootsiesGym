import collections
import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray.rllib.connectors import connector as rllib_connector
from ray.rllib.connectors import registry
from ray.rllib.policy import sample_batch
from ray.rllib.utils import typing as rllib_typing
from scipy import special




class ActionMaskConnector(rllib_connector.ActionConnector):
    """
    Apply action masking to the policy output.
    """
    @staticmethod
    def action_from_logits(logits: np.ndarray) -> int:
        action_probs = special.softmax(logits.reshape(-1))
        return np.random.choice(len(action_probs), p=action_probs)
       
    def transform(
        self, ac_data: rllib_typing.ActionConnectorDataType
    ) -> rllib_typing.ActionConnectorDataType:
        """
        Apply action masking to the policy output if it's in the INFOS dict.
        """
        # ac_data.output is a tuple of (action, states, fetches)
        # If infos are available, apply mask:
        infos = ac_data.input_dict.get("infos")
        if not (infos is not None and "action_mask" in infos):
            return ac_data
        
        _, states, fetches = ac_data.output
        logits = fetches["action_dist_inputs"]
        # Apply action mask to logits
        action_mask = infos["action_mask"]
        masked_logits = logits.copy()
        masked_logits[action_mask == 0] = -np.inf  # Set invalid actions to negative infinity
        
        # Update logits with masked values
        fetches["action_dist_inputs"] = masked_logits

        print(f"Action mask: {action_mask}")
        print(f"Original logits: {logits}")
        print(f"Masked logits: {masked_logits}")


        action = self.action_from_logits(
            logits=masked_logits
        )

        return rllib_typing.ActionConnectorDataType(
            ac_data.env_id,
            ac_data.agent_id,
            ac_data.input_dict,
            (action, states, fetches),
        )
