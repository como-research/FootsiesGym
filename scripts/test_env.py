import time

import footsiesgym

env = footsiesgym.make(config={"max_t": 4000, "frame_skip": 4, "observation_delay": 16, "port": 50051, "headless": True}, launch_binaries=True, platform="linux")


obs, _ = env.reset()
while True:
    actions = {
        agent: env.action_space[agent].sample() for agent in env.agents
    }
    observations, rewards, terminateds, truncateds, _ = env.step(actions)

    if truncateds["__all__"] or terminateds["__all__"]:
        print()
        obs, _ = env.reset()

    time.sleep(1 / 60)
