import collections
import copy
from typing import Any

import numpy as np

from footsiesgym.footsies.game import constants
from footsiesgym.footsies.game.proto import footsies_service_pb2 as footsies_pb2
from footsiesgym.footsies.typing import AgentID

import dataclasses

@dataclasses.dataclass
class NormalizationConstants:
    stage_width: float = 8.0
    max_x_value: float = 4.0
    meaningful_velocity_x: float = 5.0
    meaningful_frame_count: float = 25.0
    meaningful_sprite_shake_frame: float = 10.0
    meaningful_hit_stun_frame: float = 10.0
    meaningful_frame_advantage: float = 10.0
    meaningful_special_attack_progress: float = 1.0
    meaningful_guard_health: float = 3.0
    meaningful_vital_health: float = 1.0
    


class EncoderMethods:
    @staticmethod
    def one_hot(
        value: int | float | str, collection: list[int | float | str]
    ) -> np.ndarray:
        vector = np.zeros(len(collection), dtype=np.float32)
        vector[collection.index(value)] = 1
        return vector


class FootsiesEncoder:
    """Encoder class to generate observations from the game state"""

    observation_size: int = 85
    privileged_feature_names: list[str] = ["special_attack_progress", "would_next_forward_input_dash", "would_next_backward_input_dash", "previous_action", "is_holding_special_charge"]

    def __init__(self):
        self._last_common_state: np.ndarray | None = None

    def reset(self):
        self._last_common_state = None

    def encode(
        self,
        game_state: footsies_pb2.GameState,
        prev_actions: dict[AgentID, int],
        is_charging_special: dict[AgentID, bool],
        **kwargs,
    ) -> dict[str, Any]:
        """Encodes the game state into observations for all agents.

        kwargs can be used to pass in additional features that
        are added directly to the observation, keyed by the agent
        IDs, e.g.,
            kwargs = {
                "p1": {"p1_feature": 1},
                "p2": {"p2_feature": 2},
            }

        :param game_state: The game state to encode
        :type game_state: footsies_pb2.GameState
        :return: The encoded observations for all agents.
        :rtype: dict[str, Any]
        """
        common_state = self.encode_common_state(game_state)
        p1_encoding = self.encode_player_state(
            game_state.player1, prev_actions["p1"], is_charging_special["p1"], **kwargs.get("p1", {})
        )
        p2_encoding = self.encode_player_state(
            game_state.player2, prev_actions["p2"], is_charging_special["p2"], **kwargs.get("p2", {})
        )

        self._last_common_state = common_state

        # Concatenate the observations for the undelayed encoding
        p1_encoding_concat = np.hstack(list(p1_encoding.values()), dtype=np.float32)
        p2_encoding_concat = np.hstack(list(p2_encoding.values()), dtype=np.float32)

        # Opponent states that remove privileged features
        # Remove privileged features from the opponent's state dict then concatenate
        p1_well_known_state = np.hstack(
            [p1_encoding[key] for key in p1_encoding if key not in self.privileged_feature_names],
            dtype=np.float32,
        )
        p2_well_known_state = np.hstack(
            [p2_encoding[key] for key in p2_encoding if key not in self.privileged_feature_names],
            dtype=np.float32,
        )


        p1_centric_observation = np.hstack(
            [common_state, p1_encoding_concat, p2_well_known_state]
        )

        p2_centric_observation = np.hstack(
            [common_state, p2_encoding_concat, p1_well_known_state]
        )

        
        return {"p1": p1_centric_observation, "p2": p2_centric_observation}

    def encode_common_state(
        self, game_state: footsies_pb2.GameState
    ) -> np.ndarray:
        """
        Encode features that are always the same for both agents. These
        should be features that are a function of both players' states.

        Currently only encodes the distance between players. 

        :param game_state: The game state to encode
        :type game_state: footsies_pb2.GameState
        :return: The encoded common state
        :rtype: np.ndarray
        """
        p1_state, p2_state = game_state.player1, game_state.player2

        dist_x = (
            np.abs(p1_state.player_position_x - p2_state.player_position_x)
            / NormalizationConstants.stage_width
        )

        return np.array(
            [
                dist_x,
            ],
            dtype=np.float32,
        )

    def encode_player_state(
        self,
        player_state: footsies_pb2.PlayerState,
        prev_action: int,
        holding_special_charge: bool,
        **kwargs,
    ) -> dict[str, int | float | list]:
        """Encodes the player state into observations.

        :param player_state: The player state to encode
        :type player_state: footsies_pb2.PlayerState
        :return: The encoded observations for the player
        :rtype: dict[str, Any]

        TODO(chase): Test mirroring the positions so
            the agent always thinks it's LHS
        """
        feature_dict = {
            "player_position_x": player_state.player_position_x / NormalizationConstants.max_x_value,
            "velocity_x": player_state.velocity_x / NormalizationConstants.meaningful_velocity_x,
            "is_dead": int(player_state.is_dead),
            "vital_health": player_state.vital_health,
            "guard_health": EncoderMethods.one_hot(
                player_state.guard_health, [0, 1, 2, 3]
            ),
            "current_action_id": self._encode_action_id(
                player_state.current_action_id
            ),
            "current_action_frame": player_state.current_action_frame / NormalizationConstants.meaningful_frame_count,
            "current_action_frame_count": player_state.current_action_frame_count
            / NormalizationConstants.meaningful_frame_count,
            "current_action_remaining_frames": (
                player_state.current_action_frame_count
                - player_state.current_action_frame
            )
            / NormalizationConstants.meaningful_frame_count,
            "is_action_end": int(player_state.is_action_end),
            "is_always_cancelable": int(player_state.is_always_cancelable),
            "current_action_hit_count": player_state.current_action_hit_count,
            "current_hit_stun_frame": player_state.current_hit_stun_frame / NormalizationConstants.meaningful_hit_stun_frame,
            "is_in_hit_stun": int(player_state.is_in_hit_stun),
            "sprite_shake_position": player_state.sprite_shake_position,
            "max_sprite_shake_frame": player_state.max_sprite_shake_frame / NormalizationConstants.meaningful_sprite_shake_frame,
            "is_face_right": int(player_state.is_face_right),
            "current_frame_advantage": player_state.current_frame_advantage
            / NormalizationConstants.meaningful_frame_advantage,

            # Begin privileged features
            "would_next_forward_input_dash": int(
                player_state.would_next_forward_input_dash
            ),
            "would_next_backward_input_dash": int(
                player_state.would_next_backward_input_dash
            ),
            "special_attack_progress": min(
                player_state.special_attack_progress, 1.0
            ),
            "previous_action": EncoderMethods.one_hot(prev_action, [i for i in range(len(constants.ACTION_TO_BITS))]),
            "is_holding_special_charge": int(holding_special_charge),
        }

        if kwargs:
            feature_dict.update(kwargs)

        return feature_dict

    def _encode_action_id(self, action_id: int) -> np.ndarray:
        """Encodes the action id into a one-hot vector.

        :param action_id: The action id to encode
        :type action_id: int
        :return: The encoded one-hot vector
        :rtype: np.ndarray
        """

        action_id_values = list(constants.FOOTSIES_ACTION_IDS.values())
        action_vector = np.zeros(len(action_id_values), dtype=np.float32)

        # Get the index of the action id in constants.ActionID
        action_index = action_id_values.index(action_id)
        action_vector[action_index] = 1

        assert action_vector.max() == 1 and action_vector.min() == 0

        return action_vector

    def _encode_input_buffer(
        self, input_buffer: list[int], last_n: int | None = None
    ) -> np.ndarray:
        """Encodes the input buffer into a one-hot vector.

        :param input_buffer: The input buffer to encode
        :type input_buffer: list[int]
        :return: The encoded one-hot vector
        :rtype: np.ndarray
        """

        if last_n is not None:
            input_buffer = input_buffer[last_n:]

        ib_encoding = []
        for action_id in input_buffer:
            arr = [0] * (len(constants.ACTION_TO_BITS) + 1)
            arr[action_id] = 1
            ib_encoding.extend(arr)

        input_buffer_vector = np.asarray(ib_encoding, dtype=np.float32)

        return input_buffer_vector
