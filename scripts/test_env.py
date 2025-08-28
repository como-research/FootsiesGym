import time

import footsiesgym
from footsiesgym.footsies.game import constants

if __name__ == "__main__":

    env = footsiesgym.make(config={"max_t": 1000, "frame_skip": 4, "action_delay": 8, "port": 50051, "headless": False}, launch_binaries=True, platform="linux")

    # Define the action cycle: 20 NOOP actions followed by 1 SPECIAL_CHARGE
    action_cycle = [constants.EnvActions.NONE] * 5 + [constants.EnvActions.SPECIAL_CHARGE] # + [constants.EnvActions.NONE] * 20 + [constants.EnvActions.SPECIAL_CHARGE]
    cycle_step = 0

    obs, _ = env.reset()
    while True:
        # Get the current action from the cycle
        current_action = action_cycle[cycle_step % len(action_cycle)]
        
        actions = {
            agent: current_action for agent in env.agents
        }
        observations, rewards, terminateds, truncateds, _ = env.step(actions)

        # Advance to next step in the cycle
        cycle_step += 1

        if truncateds["__all__"] or terminateds["__all__"]:
            obs, _ = env.reset()

        time.sleep(1 / 60)
