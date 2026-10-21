# FootsiesGym

<p align="left">
  <a href="https://arxiv.org/abs/2607.06514"><img src="https://img.shields.io/badge/arXiv-2607.06514-b31b1b.svg" alt="arXiv:2607.06514"></a>
</p>

<p align="center">
  <img src="assets/footsies_trim.gif" alt="Footsies gameplay" />
</p>

A two-player zero-sum reinforcement learning benchmark built on HiFight's [Footsies](https://hifight.github.io/footsies/) fighting game. See the paper: [FootsiesGym: A Fighting Game Benchmark for Two-Player Zero-Sum Imperfect-Information Games](https://arxiv.org/abs/2607.06514).

The environment is a PettingZoo `ParallelEnv` that runs the game in one of two backends:

- **Unity** (default): the original game, driven over gRPC. Binaries download automatically.
- **C/CUDA**: a bit-identical port of the game in C that runs in-process on the CPU (SIMD + OpenMP) or a GPU, at millions to over a hundred million steps per second.

## Installation

FootsiesGym requires Python 3.12 or newer.

```bash
pip install footsies-gym           # or: uv add footsies-gym
pip install "footsies-gym[jax]"    # with JAX
```

`tensor_type="torch"` or `"jax"` needs PyTorch or JAX installed; FootsiesGym itself doesn't depend on either.

From source:

```bash
git clone https://github.com/como-research/FootsiesGym.git
cd FootsiesGym
uv sync --all-extras               # or: pip install -e ".[all]"
```

## Quick Start

```python
import footsiesgym

env = footsiesgym.make()           # downloads and launches the game server

obs, infos = env.reset()
while True:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    if terminateds["p1"] or truncateds["p1"]:
        obs, infos = env.reset()
```

With `num_envs > 1` the games are stepped together, and every value becomes an array with a leading `num_envs` axis. Finished games reset automatically inside `step()`:

```python
env = footsiesgym.make(num_envs=64)
```

The C backend runs the same game in-process, with no binaries, on the CPU or a CUDA GPU (see [C/CUDA Backend](#ccuda-backend)). `tensor_type` picks the arrays every backend returns:

```python
env = footsiesgym.make(backend="c", num_envs=65536, device="cuda", tensor_type="torch")
obs, infos = env.reset()                   # obs["p1"]: torch.float32 (65536, 88) on cuda:0
obs, rewards, terminateds, truncateds, infos = env.step({"p1": a1, "p2": a2})
```

## Architecture

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/architecture-dark.svg">
    <img alt="FootsiesGym architecture: a policy exchanges actions and observations with FootsiesEnv, which drives the Unity game server over gRPC. On first use, BinaryManager downloads, verifies, and launches the server automatically." src="assets/architecture-light.svg">
  </picture>
</p>

Solid arrows are the per-step data flow; dotted arrows run once, on first use. Each `step()` advances the game `frame_skip` frames. With `num_envs > 1`, all games are stepped in one batched RPC.

## Configuration

Every option is a config key. Pass them as `footsiesgym.make(config={...})`, as keyword arguments (`make(num_envs=64)`), or both: the two are merged, and setting a key both ways to different values is an error.

| Key | Default | Description |
|-----|---------|-------------|
| `backend` | `"grpc"` | `"grpc"` (the Unity game server) or `"c"` (the [C port](#ccuda-backend), in-process) |
| `num_envs` | `1` | Number of games stepped together (vectorized mode when > 1) |
| `tensor_type` | `"numpy"` | Arrays returned: `"numpy"`, `"torch"` or `"jax"` (actions may be any of the three) |
| `device` | `"cpu"` | `"cpu"`, `"cuda"` or `"cuda:N"` (`backend="c"`) |
| `num_threads` | all cores | CPU threads for `backend="c"` on the CPU (one per physical core is fastest) |
| `max_t` | `4000` | Steps per episode before truncation |
| `frame_skip` | `4` | Game frames per step |
| `action_delay` | `8` | Frames before a selected action executes (multiple of `frame_skip`) |
| `use_special_charge_action` | `False` | Add the special-charge toggle actions (6-8) |
| `win_reward_scaling_coeff` | `1.0` | Win/loss reward magnitude |
| `guard_break_reward` | `0.0` | Reward per guard break |
| `use_reward_budget` | `False` | Deduct guard-break rewards from the win reward |
| `return_fight_state_in_infos` | `False` | Add fight-state details to `infos` (single-env mode) |
| `headless` | `True` | Run the Unity server headless or windowed |
| `port`, `host` | auto, `"localhost"` | gRPC address of the Unity server |
| `platform` | `"linux"` | `"linux"` or `"mac"` (Unity server) |
| `launch_binaries` | `True` | Start the Unity server automatically |
| `functional` | `False` | Return pure JAX `reset`/`step` functions for jitted loops (`backend="c"`, [below](#jitted-jax-training-loops)) |

### Framework classes

Frameworks that build the env themselves take a class, not an env. `footsiesgym.make_class(...)` takes the same settings as `make()` and returns a class whose instances are what `make()` returns; the framework's own config is merged in when it builds one (setting a key twice to different values is an error). `functional` chooses the kind of class, and every class has `possible_agents` (`["p1", "p2"]`) as a class attribute:

```python
Cls = footsiesgym.make_class(backend="c", num_envs=64)            # FootsiesEnv subclass: Cls(**more_settings)

FootsiesEnv = footsiesgym.make_class(backend="c", functional=True)  # Sardine: Cls(**env_config).vectorize(n)
Experiment().environment(env=FootsiesEnv, env_config={"max_t": 4000}, num_envs=4096)
```

`FootsiesEnv` follows the PettingZoo `ParallelEnv` API. RLlib takes its own `MultiAgentEnv`, which mirrors that API; `footsiesgym.wrappers.wrap_rllib` adapts the env, single or vectorized (`num_envs > 1` gives agents `p1_i` / `p2_i`), numpy on the CPU:

```python
from ray import tune
from footsiesgym.wrappers import wrap_rllib

tune.register_env("footsies", lambda cfg: wrap_rllib(footsiesgym.make(cfg)))
config.environment("footsies", env_config={"backend": "c", "num_envs": 64})
```

## Actions

Each agent picks from `Discrete(6)`, or `Discrete(9)` with `use_special_charge_action=True`:

| ID | Action | ID | Action |
|----|--------|----|--------|
| 0 | `NONE` | 5 | `FORWARD_ATTACK` |
| 1 | `BACK` | 6 | `SPECIAL_CHARGE` |
| 2 | `FORWARD` | 7 | `FORWARD_SPECIAL_CHARGE` |
| 3 | `ATTACK` | 8 | `BACK_SPECIAL_CHARGE` |
| 4 | `BACK_ATTACK` | | |

- **Special charge.** A special attack requires holding attack for 60 frames (15 steps at `frame_skip=4`). The `SPECIAL_CHARGE` actions toggle a held attack button: while it's held, movement actions become their attack variants (`FORWARD` → `FORWARD_ATTACK`). Toggling again releases the attack.
- **Action delay.** Actions execute `action_delay // frame_skip` steps after they are selected, which simulates reaction time.

## Observations

Each agent gets a `Box` of shape `(88,)`:

| Part | Size | Contents |
|------|------|----------|
| Common | 1 | Distance between the players |
| Self | 50 | 37 public features (position, velocity, health, guard, action state) and 13 privileged ones: dash readiness (2), special-attack progress (1), previous action (9), charge state (1) |
| Opponent | 37 | Public features only |

## Rewards

Rewards are zero-sum (`rewards["p1"] + rewards["p2"] == 0`):

| Event | Reward |
|-------|--------|
| Opponent dies | `± win_reward_scaling_coeff` |
| Opponent's guard breaks | `± guard_break_reward` (at most 3 times per episode) |

With `use_reward_budget=True`, guard-break rewards are deducted from the win reward, so an episode's total reward never exceeds `win_reward_scaling_coeff`.

## C/CUDA Backend

`backend="c"` runs a port of the game's simulation (Unity's `BattleSimulation` and `Fighter`, with frame data extracted from the Unity assets) and of the environment, as plain C functions ([`csrc/`](csrc/)) that compile for the CPU (SIMD + OpenMP) and CUDA. Its observations, rewards and done flags are bit-identical to the Unity game's, so a policy trained on it plays the same in the real game. The library is compiled on first use, so it needs a C compiler (and `nvcc` for the GPU, found from `$CUDA_HOME`, `PATH` or `/usr/local/cuda`).

```python
# CPU, numpy
env = footsiesgym.make(backend="c", num_envs=16384, num_threads=24)

# GPU, PyTorch: observations and rewards never leave the GPU
env = footsiesgym.make(backend="c", num_envs=65536, device="cuda", tensor_type="torch")
obs, infos = env.reset()
actions = {"p1": policy(obs["p1"]), "p2": opponent(obs["p2"])}   # torch tensors on cuda:0
obs, rewards, terminateds, truncateds, infos = env.step(actions)
```

- The API is the same as the Unity backend's: the single-game PettingZoo env with `num_envs=1`, and arrays with a leading `num_envs` axis otherwise. Finished games reset inside `step`: the returned obs come from a fresh game, and the rewards and done flags from the episode that ended.
- `step` returns new arrays every time, so they can be kept across steps. On the GPU, torch outputs are device-to-device copies, JAX outputs are the step's own (it runs as an XLA call), and `tensor_type="numpy"` copies them to the host. Actions already on the GPU (torch, JAX, CuPy) are read in place.
- Rewards are float32 (the float64 value rounded once).
- Parity with the Unity binary is tested frame by frame (`testing/test_c_backend.py`; the live-server tests are marked `slow`). For a larger check, run `python -m footsiesgym.c.validation unity`. Unlike a fresh reset, Unity's vectorized server keeps a few fighter fields (hit stun, frame advantage, proximity-guard flags) from the previous episode when it resets a game.
- Unity's Mono runtime does float math in double precision; the port reproduces it exactly. The compiler flags are part of that guarantee: see [`csrc/README.md`](csrc/README.md).
- After changing the Unity assets, regenerate the frame data with `python scripts/extract_unity_frame_data.py <path-to-FootsiesV2>`, then `python scripts/gen_c_frame_data.py`.

### Jitted JAX training loops

With `tensor_type="jax"`, each step runs the C/CUDA step as an XLA custom call (FFI). To compile the whole rollout into one jitted program (Sardine's Anakin setup), pass `functional=True`: `make()` then returns the same step as pure functions of an explicit state, which run on JAX's device (`jax.default_device`, or the jit's):

```python
f = footsiesgym.make(backend="c", num_envs=4096, functional=True)   # Sardine's functional API

@jax.jit
def rollout(key):
    obs, state, infos = f.reset(key)
    def body(state, key):
        actions = ...                     # from your policy
        obs, state, rewards, terminateds, truncateds, infos = f.step(key, state, actions)
        return state, rewards
    return jax.lax.scan(body, state, jax.random.split(key, 128))
```

`step` resets finished games (`step_env` doesn't), and `state.t` / `state.episode_id` read as `(num_envs,)` arrays. Sardine builds the env itself from a class (`Cls(**env_config).vectorize(num_envs)`); `make_class` returns one (see [Framework classes](#framework-classes)):

```python
from sardine.experiment import Experiment

FootsiesEnv = footsiesgym.make_class(backend="c", functional=True, frame_skip=4)
Experiment().environment(env=FootsiesEnv, env_config={"max_t": 4000}, num_envs=4096)
```

The JAX port that preceded the C backend is archived on the `archive/jax-replica` branch.

## Game Server

Binaries are hosted on a CDN (`footsiesgym.chasemcd.com`) with [GitHub Releases](https://github.com/como-research/FootsiesGym/releases) as a fallback. They are downloaded on first use, verified with SHA256 checksums and cached, so you need to be online the first time you run.

The server runs on Linux and macOS; Windows is not supported. On macOS it is re-signed automatically and runs under Rosetta, because Grpc.Core is x86_64-only.

To run the server yourself, download a build from the CDN or the [`binaries-v1` release](https://github.com/como-research/FootsiesGym/releases/tag/binaries-v1), start it, and point the environment at its port:

```bash
curl -LO https://footsiesgym.chasemcd.com/v0.7.0/footsies_mac_headless_bbdb506.zip
unzip footsies_mac_headless_bbdb506.zip
codesign --force --deep --sign - footsies_mac_headless_bbdb506/FOOTSIES   # macOS, if it refuses to start
arch -x86_64 footsies_mac_headless_bbdb506/FOOTSIES --port 50051 -batchmode --grpc
```

```python
env = footsiesgym.make(config={"port": 50051}, launch_binaries=False)
```

## Training

> [!NOTE]
> The code for the paper's experiments is coming soon. Those experiments were not run with the examples below.

- **RLlib** ([APPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#appo)), with the RLModule stack (recommended) or the legacy ModelV2 stack:

  ```bash
  python -m experimentation.experiments.rllib.train_rlmodule --experiment-name <name> [--debug]
  python -m experimentation.experiments.rllib.train --experiment-name <name>
  ```

- **CleanRL**: a self-contained [PPO example](experimentation/experiments/cleanrl/).

## Throughput

<table align="center">
  <tr>
    <td align="center" width="50%"><img src="assets/throughput_scaling_linux.png" alt="Throughput scaling on Linux: environment steps per second vs. number of parallel environments, for 1-4 game-server processes. Peaks near 50,000 steps per second."></td>
    <td align="center" width="50%"><img src="assets/throughput_scaling_mac.png" alt="Throughput scaling on macOS: environment steps per second vs. number of parallel environments, for 1-4 game-server processes. Peaks above 70,000 steps per second."></td>
  </tr>
  <tr>
    <td align="center"><b>Linux</b></td>
    <td align="center"><b>macOS</b></td>
  </tr>
</table>

Unity backend: environment steps per second against `num_envs`, with $P$ game-server processes running in parallel. The C backend reaches about 165M steps per second with 1M games on one RTX 3090, and about 43M on the 24 cores of a Threadripper 3960X. See [`benchmarking/`](benchmarking/) to reproduce both.

## Development

```bash
uv sync --all-extras
uv run pytest                # add -m "not slow" to skip tests that launch the game server
uv build
```

## Citation

```bibtex
@article{mcdonald2026footsiesgym,
  title={FootsiesGym: A Fighting Game Benchmark for Two-Player Zero-Sum Imperfect-Information Games},
  author={McDonald, Chase and Tsang, Nathan and Kerr, Wesley N},
  journal={arXiv preprint arXiv:2607.06514},
  year={2026}
}
```

## License

[GNU General Public License v3.0](LICENSE.txt). FootsiesGym is based on the open-source [Footsies](https://github.com/hifight/Footsies) game by HiFight.
