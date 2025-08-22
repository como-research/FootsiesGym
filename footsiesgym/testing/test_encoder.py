import numpy as np
import pytest

from footsies import footsies_env
from footsies.encoder import EncoderMethods, FootsiesEncoder
from footsies.game import constants
from footsies.game.proto import footsies_service_pb2 as footsies_pb2


@pytest.fixture
def env():
    return footsies_env.FootsiesEnv(
        config={
            "frame_skip": 1,
            "observation_delay": 0,
            "max_t": 4000,
            "port": 80051,
        }
    )


@pytest.fixture
def encoder():
    return FootsiesEncoder(observation_delay=0)


@pytest.fixture
def game_state():
    """Create a test game state with known values"""
    state = footsies_pb2.GameState()

    # Set up player 1
    state.player1.player_position_x = 100
    state.player1.player_position_y = 0
    state.player1.velocity_x = 5
    state.player1.is_dead = False
    state.player1.vital_health = 2
    state.player1.guard_health = 3
    state.player1.is_face_right = True
    state.player1.current_action_id = constants.FOOTSIES_ACTION_IDS["STAND"]

    # Set up player 2
    state.player2.player_position_x = 300
    state.player2.player_position_y = 0
    state.player2.velocity_x = -5
    state.player2.vital_health = 100
    state.player2.guard_health = 2
    state.player2.is_face_right = False
    state.player2.current_action_id = constants.FOOTSIES_ACTION_IDS["FORWARD"]

    return state


def test_encoder_normalization(encoder, game_state):
    """Test that encoded values are properly normalized"""
    encoded = encoder.encode(game_state)

    # Position should be normalized by dividing by 2.0
    p1_pos_x = game_state.player1.player_position_x / 2.0
    p2_pos_x = game_state.player2.player_position_x / 2.0

    # Velocity should be normalized by dividing by 5.0
    p1_vel_x = game_state.player1.velocity_x / 5.0
    p2_vel_x = game_state.player2.velocity_x / 5.0

    # Extract values from the encoded observations using feature indices
    # You'll need to use the indices from your feature_indices.json
    # This is a simplified example - you'll need to adjust indices based on your actual encoding
    assert np.isclose(encoded["p1"][1], p1_pos_x)  # Adjust index as needed
    assert np.isclose(encoded["p2"][1], p2_pos_x)  # Adjust index as needed
    assert np.isclose(encoded["p1"][3], p1_vel_x)  # Adjust index as needed
    assert np.isclose(encoded["p2"][3], p2_vel_x)  # Adjust index as needed


def test_observation_delay():
    """Test that observation delay works correctly"""
    delay = 2
    delayed_encoder = FootsiesEncoder(observation_delay=delay)

    # Create sequence of different game states
    states = []
    for i in range(3):
        state = footsies_pb2.GameState()
        state.player1.player_position_x = 100 + (i * 50)
        state.player2.player_position_x = 300 - (i * 50)
        states.append(state)

    # Get encodings for each state
    encodings = [delayed_encoder.encode(state) for state in states]

    # First two observations should use immediate values
    assert np.isclose(
        encodings[0]["p1"][1], states[0].player1.player_position_x / 2.0
    )
    assert np.isclose(
        encodings[1]["p1"][1], states[1].player1.player_position_x / 2.0
    )

    # Third observation should show delayed values for opponent
    # The opponent's position should be from state[1] not state[2]
    opponent_pos_index = (
        None  # Set this to the correct index for opponent position
    )
    assert np.isclose(
        encodings[2]["p1"][opponent_pos_index],
        states[1].player2.player_position_x / 2.0,
    )


def test_one_hot_encoding():
    """Test one-hot encoding method"""
    collection = [0, 1, 2, 3]

    # Test each value in collection
    for value in collection:
        encoded = EncoderMethods.one_hot(value, collection)
        assert len(encoded) == len(collection)
        assert encoded[value] == 1
        assert sum(encoded) == 1
        assert all(v >= 0 and v <= 1 for v in encoded)


def test_action_id_encoding(encoder):
    """Test action ID encoding"""
    action_ids = constants.FOOTSIES_ACTION_IDS

    for action_name, action_id in action_ids.items():
        encoded = encoder._encode_action_id(action_id)
        assert len(encoded) == len(action_ids)
        assert encoded.sum() == 1
        assert all(v >= 0 and v <= 1 for v in encoded)


def test_input_buffer_encoding(encoder):
    """Test input buffer encoding"""
    test_inputs = [0, 1, 2, 3]  # Example input sequence
    encoded = encoder._encode_input_buffer(test_inputs)

    # Check that the encoding has the correct shape
    expected_length = len(test_inputs) * (len(constants.ACTION_TO_BITS) + 1)
    assert len(encoded) == expected_length

    # Check that each input is one-hot encoded
    for i, input_value in enumerate(test_inputs):
        start_idx = i * (len(constants.ACTION_TO_BITS) + 1)
        end_idx = start_idx + (len(constants.ACTION_TO_BITS) + 1)
        section = encoded[start_idx:end_idx]
        assert section[input_value] == 1
        assert sum(section) == 1


def test_edge_cases(encoder):
    """Test edge cases and potential error conditions"""
    state = footsies_pb2.GameState()

    # Test with zero health
    state.player1.vital_health = 0
    encoded = encoder.encode(state)
    assert encoded["p1"] is not None

    # Test with extreme positions
    state.player1.player_position_x = 1000000
    encoded = encoder.encode(state)
    assert encoded["p1"] is not None

    # Test with empty input buffer
    encoded = encoder.encode(state)
    assert encoded["p1"] is not None


def test_encode_player_state(env, encoder):

    for _ in range(10):
        terminateds = truncateds = {"__all__": False}
        obs = env.reset()
        while not terminateds["__all__"] and not truncateds["__all__"]:
            obs, rew, terminateds, truncateds, _ = env.step(
                env.action_space.sample()
            )
            game_state = env.last_game_state

            for player_state in [game_state.player1, game_state.player2]:
                encoded_player_state = encoder.encode_player_state(
                    player_state
                )
                print(
                    "PlayerPositionX",
                    encoded_player_state["player_position_x"],
                    player_state.player_position_x,
                )
                print(
                    "PlayerPositionY",
                    encoded_player_state["player_position_y"],
                    player_state.player_position_y,
                )
                print(
                    "VelocityX",
                    encoded_player_state["velocity_x"],
                    player_state.velocity_x,
                )

                for one_hot_feature in [
                    "current_action_id",
                    "guard_health",
                ]:
                    assert np.max(encoded_player_state[one_hot_feature]) == 1.0
                    assert np.min(encoded_player_state[one_hot_feature]) == 0.0
                    assert np.sum(encoded_player_state[one_hot_feature]) == 1.0

                assert np.max(encoded_player_state["input_buffer"]) == 1.0
                assert np.min(encoded_player_state["input_buffer"]) == 0.0
                assert np.sum(encoded_player_state["input_buffer"]) == 180


def test_encode_player_state(env, encoder):

    for _ in range(10):
        terminateds = truncateds = {"__all__": False}
        obs = env.reset()
        while not terminateds["__all__"] and not truncateds["__all__"]:
            obs, rew, terminateds, truncateds, _ = env.step(
                env.action_space.sample()
            )
            game_state = env.last_game_state

            for i, player_state in enumerate(
                [game_state.player1, game_state.player2]
            ):
                encoded_player_state = encoder.encode_player_state(
                    player_state
                )

                for one_hot_feature in [
                    "current_action_id",
                    "guard_health",
                ]:
                    assert np.max(encoded_player_state[one_hot_feature]) == 1.0
                    assert np.min(encoded_player_state[one_hot_feature]) == 0.0
                    assert np.sum(encoded_player_state[one_hot_feature]) == 1.0

                assert np.max(encoded_player_state["input_buffer"]) == 1.0
                assert np.min(encoded_player_state["input_buffer"]) == 0.0
                assert np.sum(encoded_player_state["input_buffer"]) == 180


if __name__ == "__main__":
    # This allows you to run the tests with python -m
    pytest.main([__file__, "-v"])
