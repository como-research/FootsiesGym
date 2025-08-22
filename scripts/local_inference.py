import collections
import time
from typing import Any

import numpy as np
from ray.rllib import policy as rllib_policy
from ray.rllib.utils import policy as rllib_policy_utils
from ray.rllib.utils import typing as rllib_typing
from scipy import special

from experimentation.components import module_repository
from footsiesgym.footsies import footsies_env
from footsiesgym.footsies.game.constants import EnvActions

try:
    import pygame
except ImportError:
    pygame = None

MODEL_FRAME_SKIP = 4

MODULES = {
    "p1": "human",  # human must be p1 for correct control mapping
    "p2": "noop",  # Add the name of a policy in the ModuleRepository here
}

if "human" in MODULES.values():
    assert (
        pygame is not None
    ), "PyGame is required for human control. Install pygame with `pip install pygame`."
    pygame.init()
    screen = pygame.display.set_mode((1, 1), pygame.NOFRAME)


def get_human_action() -> int:
    """Get the current pressed key using PyGame."""
    pygame.event.pump()
    keys = pygame.key.get_pressed()

    if keys[pygame.K_a] and keys[pygame.K_SPACE]:
        return EnvActions.BACK_ATTACK
    elif keys[pygame.K_d] and keys[pygame.K_SPACE]:
        return EnvActions.FORWARD_ATTACK
    elif keys[pygame.K_a]:
        return EnvActions.BACK
    elif keys[pygame.K_d]:
        return EnvActions.FORWARD
    elif keys[pygame.K_SPACE]:
        return EnvActions.ATTACK
    else:
        return EnvActions.NONE


MAX_FPS = 30


def action_from_logits(logits: np.ndarray) -> int:
    print(logits)
    action_probs = special.softmax(logits.reshape(-1))
    return np.random.choice(len(action_probs), p=action_probs)


def play_local_episode(
    env: footsies_env.FootsiesEnv,
    modules: dict[rllib_typing.AgentID, rllib_policy.Policy],
) -> dict[str, Any]:

    obs, _ = env.reset()
    result = {"p1_reward": 0, "p2_reward": 0}

    terminateds = {"__all__": False}
    truncateds = {"__all__": False}

    # Store last actions for non-human agents
    last_actions = {agent_id: None for agent_id in MODULES.keys()}
    frame = 0
    while not terminateds["__all__"] and not truncateds["__all__"]:
        actions = {}
        for agent_id, obs in obs.items():
            # For human agents, get action every frame
            if MODULES[agent_id] == "human":
                actions[agent_id] = get_human_action()

            else:
                if frame % MODEL_FRAME_SKIP == 0:
                    if MODULES[agent_id] == "random":
                        last_actions[agent_id] = env.action_space[
                            agent_id
                        ].sample()
                    elif MODULES[agent_id] == "noop":
                        last_actions[agent_id] = EnvActions.NONE
                    else:
                        action, _, fetch = (
                            rllib_policy_utils.local_policy_inference(
                                modules[agent_id],
                                env_id="local_env",
                                agent_id=agent_id,
                                obs=obs,
                            )[0]
                        )
                        last_actions[agent_id] = action
                actions[agent_id] = last_actions[agent_id]
        frame += 1

        obs, reward, terminateds, truncateds, _ = env.step(actions)
        result["p1_reward"] += reward["p1"]
        result["p2_reward"] += reward["p2"]
        result["p1_win"] = reward["p1"] >= 1
        result["p2_win"] = reward["p2"] >= 1

        if MAX_FPS is not None:
            time.sleep(1 / MAX_FPS)

    if terminateds["__all__"] or truncateds["__all__"]:
        time.sleep(3)

    return result


def main():

    modules = {}
    for agent_id, policy_id in MODULES.items():
        modules[agent_id] = (
            module_repository.ModuleRepository.get(policy_id)
            if policy_id != "human"
            else "human"
        )

    env = footsies_env.FootsiesEnv(
        config={
            "frame_skip": 4,
            "observation_delay": 16,
            "max_t": 4000,
            "port": 80051,
            "reward_guard_break": False,
        }
    )

    cumulative_results = collections.defaultdict(lambda: 0)
    num_games = 0
    while True:
        num_games += 1
        episode_results = play_local_episode(env, modules)
        for k, v in episode_results.items():
            cumulative_results[k] += v

        print(
            f"{num_games} games played. {MODULES['p1']} winrate: {np.round(cumulative_results['p1_win'] / num_games, 2)}"
        )


if __name__ == "__main__":
    main()
