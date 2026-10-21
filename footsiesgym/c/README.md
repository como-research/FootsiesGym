# C/CUDA backend: design and tests

This document is for developers of FootsiesGym. For the use of the C backend in RL training, refer to the [main README](../../README.md#ccuda-backend).

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
pytest testing/test_c_backend.py                    # fast tests (CPU, and GPU if available)
pytest testing/test_c_backend.py -m slow            # compare with a live Unity server
FOOTSIES_LONG_TESTS=1 pytest testing/test_c_backend.py   # exhaustive tests (minutes)
```

The GPU tests need a CUDA GPU and `nvcc`. The torch and JAX tests need torch and JAX. If they are not available, pytest skips these tests.

For a larger comparison with Unity:

```bash
python -m footsiesgym.c.validation trace                                  # the recorded Unity trace
python -m footsiesgym.c.validation unity --num-envs 256 --frames 20000    # a live Unity server
python -m footsiesgym.c.validation unity --record testing/data/unity_trace.npz   # record a new trace
```

## Change the frame data

If the Unity assets change, make the frame data again:

1. Run `python scripts/extract_unity_frame_data.py <path-to-FootsiesV2>`. This step writes `footsiesgym/c/frame_data.json`.
2. Run `python scripts/gen_c_frame_data.py`. This step writes `csrc/frame_data.gen.h`.
3. Run `pytest testing/test_c_backend.py`.

`python scripts/gen_c_frame_data.py --check` shows an error if `csrc/frame_data.gen.h` is not up to date.

## History

A JAX port of the game was the version before the C backend. The C code uses the order of the float operations from that port. The JAX port is on the `archive/jax-replica` branch.
