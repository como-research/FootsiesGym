import dataclasses
from typing import Any

import numpy as np

from footsiesgym.footsies.game import constants
from footsiesgym.footsies.game.proto import footsies_service_pb2 as footsies_pb2
from footsiesgym.footsies.typing import ActionType, AgentID


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

    observation_size: int = 88
    privileged_feature_names: list[str] = [
        "special_attack_progress",
        "would_next_forward_input_dash",
        "would_next_backward_input_dash",
        "previous_action",
        "is_holding_special_charge",
    ]

    def __init__(self):
        self._last_common_state: np.ndarray | None = None

    def reset(self):
        self._last_common_state = None

    def encode(
        self,
        game_state: footsies_pb2.GameState,
        prev_actions: dict[AgentID, ActionType],
        is_charging_special: dict[AgentID, bool],
        num_actions: int,
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
            game_state.player1,
            prev_actions["p1"],
            is_charging_special["p1"],
            num_actions,
            **kwargs.get("p1", {}),
        )
        p2_encoding = self.encode_player_state(
            game_state.player2,
            prev_actions["p2"],
            is_charging_special["p2"],
            num_actions,
            **kwargs.get("p2", {}),
        )

        self._last_common_state = common_state

        # Concatenate the observations for the undelayed encoding
        p1_encoding_concat = np.hstack(list(p1_encoding.values()), dtype=np.float32)
        p2_encoding_concat = np.hstack(list(p2_encoding.values()), dtype=np.float32)

        # Opponent states that remove privileged features
        # Remove privileged features from the opponent's state dict then concatenate
        p1_well_known_state = np.hstack(
            [
                p1_encoding[key]
                for key in p1_encoding
                if key not in self.privileged_feature_names
            ],
            dtype=np.float32,
        )
        p2_well_known_state = np.hstack(
            [
                p2_encoding[key]
                for key in p2_encoding
                if key not in self.privileged_feature_names
            ],
            dtype=np.float32,
        )

        p1_centric_observation = np.hstack(
            [common_state, p1_encoding_concat, p2_well_known_state]
        )

        p2_centric_observation = np.hstack(
            [common_state, p2_encoding_concat, p1_well_known_state]
        )

        return {"p1": p1_centric_observation, "p2": p2_centric_observation}

    def encode_common_state(self, game_state: footsies_pb2.GameState) -> np.ndarray:
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
        prev_action: ActionType,
        holding_special_charge: bool,
        num_actions: int,
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
            "player_position_x": player_state.player_position_x
            / NormalizationConstants.max_x_value,
            "velocity_x": player_state.velocity_x
            / NormalizationConstants.meaningful_velocity_x,
            "is_dead": int(player_state.is_dead),
            "vital_health": player_state.vital_health,
            "guard_health": EncoderMethods.one_hot(
                player_state.guard_health, [0, 1, 2, 3]
            ),
            "current_action_id": self._encode_action_id(player_state.current_action_id),
            "current_action_frame": player_state.current_action_frame
            / NormalizationConstants.meaningful_frame_count,
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
            "current_hit_stun_frame": player_state.current_hit_stun_frame
            / NormalizationConstants.meaningful_hit_stun_frame,
            "is_in_hit_stun": int(player_state.is_in_hit_stun),
            "sprite_shake_position": player_state.sprite_shake_position,
            "max_sprite_shake_frame": player_state.max_sprite_shake_frame
            / NormalizationConstants.meaningful_sprite_shake_frame,
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
            "special_attack_progress": min(player_state.special_attack_progress, 1.0),
            "previous_action": EncoderMethods.one_hot(
                prev_action, [i for i in range(num_actions)]
            ),
            "is_holding_special_charge": int(holding_special_charge),
        }

        if kwargs:
            feature_dict.update(kwargs)

        return feature_dict

    def _encode_action_id(self, action_id: int) -> np.ndarray:
        """Encodes the action id into a one-hot vector.
        Note that the action ID is _not_ the action the agent selects,
        but rather an integer that corresponds to the action (script) being
        executed in the game.

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


# Pre-computed lookup table: action_id (sparse int) -> one-hot index
_ACTION_ID_VALUES = list(constants.FOOTSIES_ACTION_IDS.values())
_NUM_ACTION_IDS = len(_ACTION_ID_VALUES)  # 17
_ACTION_ID_LUT = np.full(max(_ACTION_ID_VALUES) + 1, -1, dtype=np.int32)
for _idx, _val in enumerate(_ACTION_ID_VALUES):
    _ACTION_ID_LUT[_val] = _idx

# Well-known feature width (shared between self and opponent views)
_WELL_KNOWN_WIDTH = 37
# Scalar well-known features before and after the one-hot blocks
_N_GUARD_CLASSES = 4


class VectorizedEncoder:
    """Batch encoder for BatchRawState -> (num_envs, obs_size) arrays.

    Uses vectorized NumPy operations with no per-env Python loops.
    Produces output identical to FootsiesEncoder.encode() applied
    to each environment independently.
    """

    def __init__(self, num_actions: int = 7):
        self.num_actions = num_actions
        self.privileged_width = (
            4 + num_actions
        )  # dash(2) + special(1) + prev_action(num_actions) + holding(1)
        self.self_width = _WELL_KNOWN_WIDTH + self.privileged_width
        self.obs_size = 1 + self.self_width + _WELL_KNOWN_WIDTH

    @staticmethod
    def observation_size(num_actions: int = 7) -> int:
        return 79 + num_actions

    def encode(
        self,
        batch_raw_state: footsies_pb2.BatchRawState,
        prev_p1_actions: np.ndarray,
        prev_p2_actions: np.ndarray,
        p1_holding_special: np.ndarray,
        p2_holding_special: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Encode a BatchRawState into per-player observation arrays.

        :param batch_raw_state: Raw per-field state from the server.
        :param prev_p1_actions: (num_envs,) int array of previous
            selected actions for p1.
        :param prev_p2_actions: (num_envs,) int array for p2.
        :param p1_holding_special: (num_envs,) bool array.
        :param p2_holding_special: (num_envs,) bool array.
        :return: {"p1": (N, obs_size), "p2": (N, obs_size)} float32.
        """
        s = batch_raw_state
        num_envs = len(s.p1_position_x)

        # --- Extract proto fields to numpy (one-time copy) ---
        p1_pos = np.array(s.p1_position_x, dtype=np.float32)
        p2_pos = np.array(s.p2_position_x, dtype=np.float32)

        p1_fields = self._extract_player_fields(s, "p1")
        p2_fields = self._extract_player_fields(s, "p2")

        # --- Common feature ---
        common = np.abs(p1_pos - p2_pos) / NormalizationConstants.stage_width

        # --- Encode player blocks ---
        p1_full, p1_wk = self._encode_player_block(
            p1_fields, num_envs, prev_p1_actions, p1_holding_special
        )
        p2_full, p2_wk = self._encode_player_block(
            p2_fields, num_envs, prev_p2_actions, p2_holding_special
        )

        # --- Assemble observations ---
        # P1-centric: [common, p1_full, p2_well_known]
        out_p1 = np.empty((num_envs, self.obs_size), dtype=np.float32)
        out_p1[:, 0] = common
        out_p1[:, 1 : 1 + self.self_width] = p1_full
        out_p1[:, 1 + self.self_width :] = p2_wk

        # P2-centric: [common, p2_full, p1_well_known]
        out_p2 = np.empty((num_envs, self.obs_size), dtype=np.float32)
        out_p2[:, 0] = common
        out_p2[:, 1 : 1 + self.self_width] = p2_full
        out_p2[:, 1 + self.self_width :] = p1_wk

        return {"p1": out_p1, "p2": out_p2}

    def _extract_player_fields(
        self, s: footsies_pb2.BatchRawState, prefix: str
    ) -> dict[str, np.ndarray]:
        """Extract all per-player fields from the proto into numpy."""
        g = lambda field: np.array(getattr(s, f"{prefix}_{field}"), dtype=np.float32)
        gi = lambda field: np.array(getattr(s, f"{prefix}_{field}"), dtype=np.int64)
        return {
            "position_x": g("position_x"),
            "velocity_x": g("velocity_x"),
            "is_dead": g("is_dead"),
            "vital_health": gi("vital_health").astype(np.float32),
            "guard_health": gi("guard_health"),
            "current_action_id": gi("current_action_id"),
            "current_action_frame": gi("current_action_frame").astype(np.float32),
            "current_action_frame_count": gi("current_action_frame_count").astype(
                np.float32
            ),
            "is_action_end": g("is_action_end"),
            "is_always_cancelable": g("is_always_cancelable"),
            "current_action_hit_count": gi("current_action_hit_count").astype(
                np.float32
            ),
            "current_hit_stun_frame": gi("current_hit_stun_frame").astype(np.float32),
            "is_in_hit_stun": g("is_in_hit_stun"),
            "sprite_shake_position": gi("sprite_shake_position").astype(np.float32),
            "max_sprite_shake_frame": gi("max_sprite_shake_frame").astype(np.float32),
            "is_face_right": g("is_face_right"),
            "current_frame_advantage": gi("current_frame_advantage").astype(np.float32),
            "would_next_forward_input_dash": g("would_next_forward_input_dash"),
            "would_next_backward_input_dash": g("would_next_backward_input_dash"),
            "special_attack_progress": g("special_attack_progress"),
        }

    def _encode_player_block(
        self,
        f: dict[str, np.ndarray],
        num_envs: int,
        prev_actions: np.ndarray,
        holding_special: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode one player's features into full and well-known blocks.

        Returns:
            full: (num_envs, self_width) - all features including
                privileged
            well_known: (num_envs, 37) - view of first 37 columns
        """
        full = np.zeros((num_envs, self.self_width), dtype=np.float32)
        c = 0  # column cursor

        # --- Well-known features (37 total) ---
        full[:, c] = f["position_x"] / NormalizationConstants.max_x_value
        c += 1
        full[:, c] = f["velocity_x"] / NormalizationConstants.meaningful_velocity_x
        c += 1
        full[:, c] = f["is_dead"]
        c += 1
        full[:, c] = f["vital_health"]
        c += 1

        # guard_health one-hot (4)
        gh_indices = f["guard_health"].astype(np.intp)
        full[np.arange(num_envs), c + gh_indices] = 1.0
        c += _N_GUARD_CLASSES

        # current_action_id one-hot (17)
        aid_raw = f["current_action_id"]
        aid_indices = _ACTION_ID_LUT[aid_raw.astype(np.intp)]
        full[np.arange(num_envs), c + aid_indices] = 1.0
        c += _NUM_ACTION_IDS

        full[:, c] = (
            f["current_action_frame"] / NormalizationConstants.meaningful_frame_count
        )
        c += 1
        full[:, c] = (
            f["current_action_frame_count"]
            / NormalizationConstants.meaningful_frame_count
        )
        c += 1
        full[:, c] = (
            f["current_action_frame_count"] - f["current_action_frame"]
        ) / NormalizationConstants.meaningful_frame_count
        c += 1
        full[:, c] = f["is_action_end"]
        c += 1
        full[:, c] = f["is_always_cancelable"]
        c += 1
        full[:, c] = f["current_action_hit_count"]
        c += 1
        full[:, c] = (
            f["current_hit_stun_frame"]
            / NormalizationConstants.meaningful_hit_stun_frame
        )
        c += 1
        full[:, c] = f["is_in_hit_stun"]
        c += 1
        full[:, c] = f["sprite_shake_position"]
        c += 1
        full[:, c] = (
            f["max_sprite_shake_frame"]
            / NormalizationConstants.meaningful_sprite_shake_frame
        )
        c += 1
        full[:, c] = f["is_face_right"]
        c += 1
        full[:, c] = (
            f["current_frame_advantage"]
            / NormalizationConstants.meaningful_frame_advantage
        )
        c += 1

        assert c == _WELL_KNOWN_WIDTH

        # --- Privileged features ---
        full[:, c] = f["would_next_forward_input_dash"]
        c += 1
        full[:, c] = f["would_next_backward_input_dash"]
        c += 1
        full[:, c] = np.minimum(f["special_attack_progress"], 1.0)
        c += 1

        # previous_action one-hot (num_actions)
        prev_idx = np.asarray(prev_actions, dtype=np.intp)
        full[np.arange(num_envs), c + prev_idx] = 1.0
        c += self.num_actions

        full[:, c] = np.asarray(holding_special, dtype=np.float32)
        c += 1

        assert c == self.self_width

        well_known = full[:, :_WELL_KNOWN_WIDTH]
        return full, well_known
