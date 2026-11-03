# C/CUDA backend

## Use

`backend="c"` runs a port of the game in C, in the Python process. It runs on the CPU or on a CUDA GPU. Its observations, rewards and done flags are bit-identical to the Unity game. Thus, a policy that you train with the C backend plays the same in the real game.

The C backend does not need the game server. It needs a C compiler, because FootsiesGym compiles the library on first use. For the GPU, it also needs `nvcc`. FootsiesGym finds `nvcc` through `$CUDA_HOME`, `PATH` or `/usr/local/cuda`.

```python
# CPU, numpy
env = footsiesgym.make(backend="c", num_envs=16384, num_threads=24)

# GPU, PyTorch: the observations and rewards stay on the GPU
env = footsiesgym.make(backend="c", num_envs=65536, device="cuda", tensor_type="torch")
obs, infos = env.reset()                   # obs["p1"]: torch.float32 (65536, 88) on cuda:0
actions = {"p1": policy(obs["p1"]), "p2": opponent(obs["p2"])}   # torch tensors on cuda:0
obs, rewards, terminateds, truncateds, infos = env.step(actions)
```

- The API is the same as the API of the Unity backend. With `num_envs=1`, the environment is a single-game PettingZoo environment. With more games, each value is an array, and its first axis has the length `num_envs`.
- When a game ends, `step` starts a new game in its position. The returned observations come from the new game. The rewards and done flags come from the episode that ended.
- Each `step` returns new arrays. Thus, you can keep the arrays from one step after the next step.
- The actions can be numpy, torch or JAX arrays. If the actions are on the GPU of the environment, they stay on the GPU.
- The rewards are float32 values.
- On the CPU, one thread for each physical core gives the best speed. Set this number with `num_threads`.

### Jitted JAX training loops

With `functional=True`, `make()` returns the environment as pure functions of an explicit state. JAX can compile all of the rollout into one jitted program, as in Sardine's Anakin setup. The functions run on the JAX device: `jax.default_device`, or the device of the jitted function.

```python
import jax
import footsiesgym

env = footsiesgym.make(backend="c", num_envs=4096, functional=True)

@jax.jit
def rollout(key):
    obs, state, infos = env.reset(key)

    def step(carry, key):
        state, obs = carry
        k1, k2 = jax.random.split(key)
        actions = {                                   # replace with your policy
            "p1": jax.random.randint(k1, (env.num_envs,), 0, env.action_dim),
            "p2": jax.random.randint(k2, (env.num_envs,), 0, env.action_dim),
        }
        obs, state, rewards, terminateds, truncateds, infos = env.step(key, state, actions)
        return (state, obs), rewards["p1"]

    _, rewards = jax.lax.scan(step, (state, obs), jax.random.split(key, 1000))
    return rewards                                    # (1000, 4096)

rewards = rollout(jax.random.key(0))
```

`step` starts new games when games end. `step_env` does not start new games. `state.t` and `state.episode_id` are arrays with the shape `(num_envs,)`.

Sardine makes the environment from a class (`Cls(**env_config).vectorize(num_envs)`). `make_class` returns this class (refer to [Framework classes](../README.md#framework-classes)):

```python
from sardine.experiment import Experiment

FootsiesEnv = footsiesgym.make_class(backend="c", functional=True, frame_skip=4)
Experiment().environment(env=FootsiesEnv, env_config={"max_t": 4000}, num_envs=4096)
```

## Design

The sections that follow are for developers of FootsiesGym.

The C backend is a port of the Footsies game simulation (Unity's `BattleSimulation` and `Fighter`) and of the environment. The C code is in [`csrc/`](../../csrc/). It compiles for the CPU and for CUDA from one source. For the C design (SIMD batches, GPU kernels, compiler flags), refer to [`csrc/README.md`](../../csrc/README.md).

## Modules

| Module | Contents |
|--------|----------|
| `runtime.py` | `CRuntime`: the runtime that `footsiesgym.make(backend="c")` uses. It selects one of the three runtimes below and returns the outputs as new arrays of `tensor_type`. |
| `env.py` | `FootsiesCEnv`: the CPU runtime. It uses preallocated numpy buffers, SIMD and OpenMP. |
| `cuda.py` | `CudaEnv`: the GPU runtime. It uses device buffers and one kernel for each step. It uses ctypes, not torch or JAX. |
| `jax_env.py` | `FootsiesFFIEnv`: the step as an XLA FFI call. `tensor_type="jax"` and `functional=True` use it. |
| `dlpack.py` | `DLPackBuffer`: gives device buffers to torch and CuPy through DLPack, without a copy. |
| `game.py` | `CGames`, `BatchGames`: batches of games, frame by frame (for validation). |
| `validation.py` | Comparison with the recorded Unity trace and with a live Unity server. |
| `build.py`, `lib.py` | Compilation of `csrc/` and the ctypes bindings. |
| `frame_data.json`, `frame_data.py` | The frame data from the Unity assets, and its loader. |

## How `make()` selects the runtime

| `tensor_type` | `device="cpu"` | `device="cuda"` |
|---------------|----------------|-----------------|
| `"numpy"` | `FootsiesCEnv` | `CudaEnv`, outputs copied to the host |
| `"torch"` | `FootsiesCEnv`, `torch.from_numpy` | `CudaEnv`, outputs copied on the device |
| `"jax"` | `FootsiesFFIEnv` on the CPU | `FootsiesFFIEnv` on the GPU |

`functional=True` returns the vectorized `FootsiesFFIEnv` directly.

## Outputs and actions

Each `step` returns new arrays. The runtime never gives its own buffers to the user.

- **CPU:** a parallel C function (`footsies_split_outputs`) copies the outputs into a set of arrays for each agent. The runtime uses a set again only when no other object has a reference to it. A pool keeps a maximum of 4 sets.
- **GPU, torch:** the runtime makes torch views of its device buffers one time, through DLPack. For each step, it clones the views on the legacy default stream. If the current torch stream is a different stream, that stream waits for the clones.
- **GPU, numpy:** the runtime copies the outputs to the host.
- **JAX:** the outputs come from the XLA call. They are new JAX arrays.
- **Actions on the GPU:** a small kernel copies int32 or int64 action tensors into the action buffer of the environment. If the actions are on a different stream, the stream of the environment waits for that stream.

## Compilation

FootsiesGym compiles the libraries on first use and keeps them in `~/.cache/footsiesgym` (or `$FOOTSIES_C_CACHE`). The name of each library contains a hash of the sources, the compiler flags, the compiler and the CPU features. Thus, a change to one of these causes a new compilation.

| Variable | Effect |
|----------|--------|
| `FOOTSIES_C_CACHE` | Directory for the compiled libraries |
| `FOOTSIES_CSRC` | Directory of the C sources |
| `FOOTSIES_MARCH` | CPU target (default `native`) |
| `FOOTSIES_CUDA_ARCH` | GPU architecture for `nvcc` (default `native`) |
| `NVCC`, `CUDA_HOME`, `CUDA_PATH` | Location of `nvcc` (`/usr/local/cuda/bin/nvcc` if not set) |
| `CC`, `CXX` | C and C++ compilers |

`impl="auto"` (the default for `FootsiesCEnv`) uses the SIMD batches only when the library has AVX-512. Without AVX-512, the scalar code is faster.

## Exactness

- The Mono runtime of Unity calculates float values in double precision. The C code gives the same results exactly.
- The compiler flags are part of this match. Do not use `-ffast-math` or FMA contraction. For the full rules, refer to [`csrc/README.md`](../../csrc/README.md#byte-identity).
- On reset, the vectorized Unity server keeps some fighter fields from the previous episode: hit stun, frame advantage and the proximity-guard flags. A reset of the C backend starts a fully new game. Thus, the tests compare each game with Unity through its first episode only.

## Tests

```bash
uv run pytest testing/test_c_backend.py                    # fast tests (CPU, and GPU if available)
uv run pytest testing/test_c_backend.py -m slow            # compare with a live Unity server
FOOTSIES_LONG_TESTS=1 uv run pytest testing/test_c_backend.py   # exhaustive tests (minutes)
```

The GPU tests need a CUDA GPU and `nvcc`. The torch and JAX tests need torch and JAX. If they are not available, pytest skips these tests.

For a larger comparison with Unity:

```bash
uv run python -m footsiesgym.c.validation trace                                  # the recorded Unity trace
uv run python -m footsiesgym.c.validation unity --num-envs 256 --frames 20000    # a live Unity server
uv run python -m footsiesgym.c.validation unity --record testing/data/unity_trace.npz   # record a new trace
```

## Change the frame data

If the Unity assets change, make the frame data again:

1. Run `uv run python scripts/extract_unity_frame_data.py <path-to-FootsiesV2>`. This step writes `footsiesgym/c/frame_data.json`.
2. Run `uv run python scripts/gen_c_frame_data.py`. This step writes `csrc/frame_data.gen.h`.
3. Run `uv run pytest testing/test_c_backend.py`.

`uv run python scripts/gen_c_frame_data.py --check` shows an error if `csrc/frame_data.gen.h` is not up to date.

## History

A JAX port of the game was the version before the C backend. The C code uses the order of the float operations from that port. The JAX port is on the `archive/jax-replica` branch.
