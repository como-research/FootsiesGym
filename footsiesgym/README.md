# Environment Reference

## Configuration

Each option is a config key. You can give the options in a config dictionary, as keyword arguments, or both. For example, use `footsiesgym.make(config={...})` or `footsiesgym.make(num_envs=64)`. FootsiesGym merges the two. If you set one key to two different values, FootsiesGym shows an error.

| Key | Default | Description |
|-----|---------|-------------|
| `backend` | `"grpc"` | `"grpc"` (the Unity game server) or `"c"` (the [C port](c/README.md), in the Python process) |
| `num_envs` | `1` | Number of games that run at the same time (vectorized mode when the value is more than 1) |
| `tensor_type` | `"numpy"` | Type of the returned arrays: `"numpy"`, `"torch"` or `"jax"`. Actions can be any of the three types. |
| `device` | `"cpu"` | `"cpu"`, `"cuda"` or `"cuda:N"` (only for `backend="c"`) |
| `num_threads` | all cores | Number of CPU threads for `backend="c"` on the CPU. One thread for each physical core gives the best speed. |
| `max_t` | `4000` | Number of steps in an episode before truncation |
| `frame_skip` | `4` | Number of game frames in each step |
| `action_delay` | `8` | Number of frames before a selected action starts. The value must be a multiple of `frame_skip`. |
| `use_special_charge_action` | `False` | Adds the special-charge toggle actions (6 to 8) |
| `win_reward_scaling_coeff` | `1.0` | Size of the win and loss reward |
| `guard_break_reward` | `0.0` | Reward for each guard break |
| `use_reward_budget` | `False` | Subtracts guard-break rewards from the win reward |
| `return_fight_state_in_infos` | `False` | Adds fight-state data to `infos` (single-game mode only) |
| `headless` | `True` | Runs the Unity server headless or in a window (only for `backend="grpc"`) |
| `port`, `host` | automatic, `"localhost"` | gRPC address of the Unity server |
| `platform` | `"linux"` | `"linux"` or `"mac"` (Unity server) |
| `launch_binaries` | `True` | Starts the Unity server automatically |
| `functional` | `False` | Returns pure JAX `reset` and `step` functions for jitted loops (`backend="c"`, refer to [Jitted JAX training loops](c/README.md#jitted-jax-training-loops)) |

## Framework classes

Some frameworks make the environment themselves. These frameworks need a class, not an environment object. `footsiesgym.make_class(...)` accepts the same settings as `make()`. It returns a class, and each instance of the class is the object that `make()` returns. When the framework makes an instance, FootsiesGym merges the framework configuration with the settings of the class. If one key has two different values, FootsiesGym shows an error.

The `functional` setting selects the type of class. Each class has the class attribute `possible_agents` (`["p1", "p2"]`).

```python
Cls = footsiesgym.make_class(backend="c", num_envs=64)            # FootsiesEnv subclass: Cls(**more_settings)

FootsiesEnv = footsiesgym.make_class(backend="c", functional=True)  # Sardine: Cls(**env_config).vectorize(n)
Experiment().environment(env=FootsiesEnv, env_config={"max_t": 4000}, num_envs=4096)
```

`FootsiesEnv` uses the PettingZoo `ParallelEnv` API. RLlib uses its own `MultiAgentEnv` class, which has an API that is almost the same. Use `footsiesgym.wrappers.wrap_rllib` to make the environment usable in RLlib. The wrapper accepts a single-game environment or a vectorized environment. For a vectorized environment, the agents of game `i` are `p1_i` and `p2_i`. The wrapper needs `tensor_type="numpy"` on the CPU.

```python
from ray import tune
from footsiesgym.wrappers import wrap_rllib

tune.register_env("footsies", lambda cfg: wrap_rllib(footsiesgym.make(cfg)))
config.environment("footsies", env_config={"backend": "c", "num_envs": 64})
```

## Actions

Each agent selects one action from `Discrete(6)`. With `use_special_charge_action=True`, the agent selects one action from `Discrete(9)`.

| ID | Action | ID | Action |
|----|--------|----|--------|
| 0 | `NONE` | 5 | `FORWARD_ATTACK` |
| 1 | `BACK` | 6 | `SPECIAL_CHARGE` |
| 2 | `FORWARD` | 7 | `FORWARD_SPECIAL_CHARGE` |
| 3 | `ATTACK` | 8 | `BACK_SPECIAL_CHARGE` |
| 4 | `BACK_ATTACK` | | |

- **Special charge.** For a special attack, the agent must hold the attack button for 60 frames. At `frame_skip=4`, this is 15 steps. A `SPECIAL_CHARGE` action holds the attack button. While the button is held, the movement actions change to their attack versions (`FORWARD` changes to `FORWARD_ATTACK`). A second `SPECIAL_CHARGE` action releases the button.
- **Action delay.** An action starts `action_delay // frame_skip` steps after the agent selects it. This delay simulates reaction time.

## Observations

Each agent gets a `Box` with the shape `(88,)`:

| Part | Size | Contents |
|------|------|----------|
| Common | 1 | Distance between the players |
| Self | 50 | 37 public features (position, velocity, health, guard, action state) and 13 private features: dash readiness (2), special-attack progress (1), previous action (9), charge state (1) |
| Opponent | 37 | Public features only |

## Rewards

The rewards are zero-sum (`rewards["p1"] + rewards["p2"] == 0`):

| Event | Reward |
|-------|--------|
| The opponent dies | `± win_reward_scaling_coeff` |
| The guard of the opponent breaks | `± guard_break_reward` (a maximum of 3 times in each episode) |

With `use_reward_budget=True`, FootsiesGym subtracts the guard-break rewards from the win reward. Thus, the total reward of an episode is never more than `win_reward_scaling_coeff`.
