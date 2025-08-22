import json
import os
from typing import Any

import numpy as np

from footsies import footsies_env

MAX_FPS = 600


def check_feature_mismatch(
    proto_obs: np.ndarray,
    obs: np.ndarray,
    feature_name: str,
    start_idx: int,
    length: int,
):
    """Check if a specific feature matches between proto and env observations."""
    feature_slice = slice(start_idx, start_idx + length)
    if not np.isclose(proto_obs[feature_slice], obs[feature_slice]).all():
        print(
            f"Mismatch in {feature_name}, index {start_idx} to {start_idx + length}:"
        )
        print(f"  Proto: {proto_obs[feature_slice]}")
        print(f"    Env: {obs[feature_slice]}")


def play_local_episode(env: footsies_env.FootsiesEnv) -> dict[str, Any]:
    obs, _ = env.reset()
    result = {"p1_reward": 0, "p2_reward": 0}

    terminateds = {"__all__": False}
    truncateds = {"__all__": False}

    # Load feature indices from JSON
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "feature_indices.json"
    )
    try:
        with open(json_path, "r") as f:
            features = json.load(f)
    except FileNotFoundError:
        print(
            "Feature indices file not found. Please run the encoder first to generate it."
        )
        return result

    frame = 0
    while not terminateds["__all__"] and not truncateds["__all__"]:
        actions = {}
        encoded_state = env.game.get_encoded_state()
        encoded_state_dict = {
            "p1": encoded_state.player1_encoding,
            "p2": encoded_state.player2_encoding,
        }

        for agent_id, obs in obs.items():
            actions[agent_id] = env.action_space[agent_id].sample()
            proto_obs = encoded_state_dict[agent_id]

            if agent_id == "p1":
                # Check each feature individually
                if not np.isclose(proto_obs, obs).all():
                    print(
                        f"\nMismatches found for {agent_id} at frame {frame}."
                    )
                for feature_name, feature_info in features.items():
                    check_feature_mismatch(
                        proto_obs,
                        obs.tolist(),
                        feature_name,
                        feature_info["start"],
                        feature_info["length"],
                    )

        frame += 1

        obs, reward, terminateds, truncateds, _ = env.step(actions)
        result["p1_reward"] += reward["p1"]
        result["p2_reward"] += reward["p2"]
        result["p1_win"] = reward["p1"] == 1
        result["p2_win"] = reward["p2"] == 1

    return result


def main():

    env = footsies_env.FootsiesEnv(
        config={
            "frame_skip": 1,
            "observation_delay": 16,
            "max_t": 1000,
            "port": 50051,
        }
    )

    while True:
        episode_results = play_local_episode(env)


if __name__ == "__main__":
    main()
