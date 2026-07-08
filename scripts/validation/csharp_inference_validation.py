"""
This script is used to validate the csharp inference code. We export the encoded game state
and the logits from running the csharp inference code, then run the python inference
and compare the logits.
"""

import json
import os

import numpy as np
from ray.rllib.utils import policy as rllib_policy_utils
from scipy import special

from experimentation.experiments.rllib.components import module_repository

MODULE = "4fs-16od-13c7f7b-0.05to0.01-sp-02"


def action_from_logits(logits: np.ndarray) -> int:
    action_probs = special.softmax(logits.reshape(-1))
    return np.random.choice(len(action_probs), p=action_probs)


def validate_game(game_data: list[dict[str, list[float]]], module, index: int):
    """
    Game data is a list of frames. Each frame is a dictionary with the following keys:
    - "frame": int
    - "encoding": list[float]
    - "logits": list[float]
    - "isPlayer1": bool
    """
    # sort by frame to ensure hidden states propagate correctly.
    game_data.sort(key=lambda x: x["frame"])

    for i, frame in enumerate(game_data):
        action, [state_out_0, state_out_1], fetch = (
            rllib_policy_utils.local_policy_inference(
                module,
                env_id=f"local_env_{index}",
                agent_id="p2",
                obs=np.asarray(frame["encoding"], dtype=np.float32),
            )[0]
        )

        if i > 0:
            assert np.allclose(client_cxs, frame["cell_state_in"])

            assert np.allclose(client_hxs, frame["hidden_state_in"])

        torch_logits = fetch["action_dist_inputs"]
        client_logits = frame["logits"]
        assert np.allclose(
            torch_logits, client_logits, atol=1e-4
        ), f"Logits don't match, max diff: {np.max(np.abs(torch_logits - client_logits))}"

        client_cxs = frame["cell_state_out"]
        assert np.allclose(
            state_out_0, client_cxs, atol=1e-4
        ), f"Cell state out doesn't match, max diff: {np.max(np.abs(state_out_0 - client_cxs))}"

        client_hxs = frame["hidden_state_out"]
        assert np.allclose(
            state_out_1, client_hxs, atol=1e-4
        ), f"Hidden state out doesn't match, max diff: {np.max(np.abs(state_out_1 - client_hxs))}"


def main():

    module = module_repository.ModuleRepository.get(MODULE)

    validation_jsons = [
        f for f in os.listdir(f"scripts/validation/{MODULE}") if f.endswith(".json")
    ]
    for i, validation_json in enumerate(validation_jsons):
        with open(f"scripts/validation/{MODULE}/{validation_json}", "r") as f:
            validation_data = json.load(f)
        validate_game(validation_data, module, i)

    print(f"All {len(validation_jsons)} games validated successfully")


if __name__ == "__main__":

    main()
