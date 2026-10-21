// CUDA build of footsies.h: kernels, a plain C API on device pointers (for
// tests and benchmarks through ctypes), and XLA FFI handlers for JAX.
//
// Build (see footsiesgym/c/build.py, which adds the same flags):
//   nvcc -O3 -fmad=false -prec-div=true -ftz=false -prec-sqrt=true
//        -Xcompiler -fPIC -shared -I<jax.ffi.include_dir()> footsies_cuda.cu
// Never --use_fast_math. Check `-Xptxas -v` for "0 bytes stack frame".

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

#include "footsies.h"

#ifndef FOOTSIES_NO_FFI
#include "xla/ffi/api/ffi.h"
#endif

#define EXPORT extern "C" __attribute__((visibility("default")))
#define BLOCK 256

// The frame data lives in global memory once; each block copies it to shared
// memory. (Not __constant__: warps hold different actions, and constant
// memory serializes reads of different addresses.)
__device__ const FrameTables DEVICE_TABLES = FRAME_TABLES_INIT;

static_assert(sizeof(FrameTables) % 4 == 0, "tables are copied as 32-bit words");

__device__ __forceinline__ void load_tables(FrameTables* dst) {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(&DEVICE_TABLES);
    uint32_t* out = reinterpret_cast<uint32_t*>(dst);
    for (int i = threadIdx.x; i < (int)(sizeof(FrameTables) / 4); i += blockDim.x)
        out[i] = src[i];
    __syncthreads();
}

// ── Kernels ──────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(BLOCK)
footsies_env_step_kernel(EnvConfig cfg, EnvState* __restrict__ states,
                         const int32_t* __restrict__ actions, float* __restrict__ obs,
                         float* __restrict__ rew, uint8_t* __restrict__ term,
                         uint8_t* __restrict__ trunc, int autoreset, int n) {
    __shared__ FrameTables T;
    load_tables(&T);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    EnvState st = states[i];  // into registers; the whole frameskip runs there
    env_step(&st, &T, &cfg, actions[2 * i], actions[2 * i + 1],
             obs + (size_t)i * 2 * OBS_DIM, rew + 2 * i, term + i, trunc + i, autoreset);
    states[i] = st;
}

__global__ void __launch_bounds__(BLOCK)
footsies_env_reset_kernel(EnvState* __restrict__ states, float* __restrict__ obs, int n) {
    __shared__ FrameTables T;
    load_tables(&T);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    EnvState st;
    env_fresh(&st);
    env_encode(&st, &T, obs + (size_t)i * 2 * OBS_DIM);
    states[i] = st;
}

__global__ void __launch_bounds__(BLOCK)
footsies_game_step_kernel(Game* __restrict__ games, const int32_t* __restrict__ p1_bits,
                          const int32_t* __restrict__ p2_bits, int n_frames, int n) {
    __shared__ FrameTables T;
    load_tables(&T);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Game g = games[i];
    game_step_n(&g, &T, p1_bits[i], p2_bits[i], n_frames);
    games[i] = g;
}

__global__ void footsies_game_new_kernel(Game* games, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Game g;
    game_new(&g);
    games[i] = g;
}

static int blocks(int n) { return (n + BLOCK - 1) / BLOCK; }

static int launch_env_step(const EnvConfig* cfg, EnvState* states, const int32_t* actions,
                           float* obs, float* rew, uint8_t* term, uint8_t* trunc,
                           int autoreset, int n, cudaStream_t stream) {
    if (n <= 0) return 0;
    footsies_env_step_kernel<<<blocks(n), BLOCK, 0, stream>>>(
        *cfg, states, actions, obs, rew, term, trunc, autoreset, n);
    return (int)cudaGetLastError();
}

static int launch_env_reset(EnvState* states, float* obs, int n, cudaStream_t stream) {
    if (n <= 0) return 0;
    footsies_env_reset_kernel<<<blocks(n), BLOCK, 0, stream>>>(states, obs, n);
    return (int)cudaGetLastError();
}

// ── Plain C API on device pointers (ctypes: tests and benchmarks) ────────

EXPORT int footsies_cuda_sizeof_env_state(void) { return (int)sizeof(EnvState); }
EXPORT int footsies_cuda_sizeof_game(void) { return (int)sizeof(Game); }
EXPORT int footsies_cuda_sizeof_env_config(void) { return (int)sizeof(EnvConfig); }
EXPORT int footsies_cuda_max_delay_steps(void) { return MAX_DELAY_STEPS; }

EXPORT int footsies_cuda_device_count(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
}

EXPORT void* footsies_cuda_malloc(size_t bytes) {
    void* p = nullptr;
    return cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
}
EXPORT int footsies_cuda_free(void* p) { return (int)cudaFree(p); }
EXPORT int footsies_cuda_h2d(void* dst, const void* src, size_t bytes) {
    return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}
EXPORT int footsies_cuda_d2h(void* dst, const void* src, size_t bytes) {
    return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}
EXPORT int footsies_cuda_synchronize(void) { return (int)cudaDeviceSynchronize(); }

EXPORT int footsies_cuda_env_reset(EnvState* states, float* obs, int n) {
    return launch_env_reset(states, obs, n, 0);
}

// cfg is a host pointer; everything else is on the device.
EXPORT int footsies_cuda_env_step(const EnvConfig* cfg, EnvState* states,
                                  const int32_t* actions, float* obs, float* rew,
                                  uint8_t* term, uint8_t* trunc, int autoreset, int n) {
    return launch_env_step(cfg, states, actions, obs, rew, term, trunc, autoreset, n, 0);
}

EXPORT int footsies_cuda_game_new(Game* games, int n) {
    if (n <= 0) return 0;
    footsies_game_new_kernel<<<blocks(n), BLOCK>>>(games, n);
    return (int)cudaGetLastError();
}

EXPORT int footsies_cuda_game_step(Game* games, const int32_t* p1_bits,
                                   const int32_t* p2_bits, int n_frames, int n) {
    if (n <= 0) return 0;
    footsies_game_step_kernel<<<blocks(n), BLOCK>>>(games, p1_bits, p2_bits, n_frames, n);
    return (int)cudaGetLastError();
}

// ── XLA FFI handlers ─────────────────────────────────────────────────────

#ifndef FOOTSIES_NO_FFI
namespace ffi = xla::ffi;

static ffi::Error cuda_error(cudaError_t err) {
    if (err == cudaSuccess) return ffi::Error::Success();
    return ffi::Error::Internal(cudaGetErrorString(err));
}

static ffi::Error read_config(ffi::Span<const uint8_t> bytes, EnvConfig* cfg) {
    if (bytes.size() != sizeof(EnvConfig))
        return ffi::Error::InvalidArgument("config has the wrong size");
    std::memcpy(cfg, bytes.begin(), sizeof(EnvConfig));
    return ffi::Error::Success();
}

// (state uint8[N, S], actions int32[N, 2]) ->
//   (state uint8[N, S] aliased, obs f32[N, 2, 88], rew f32[N, 2], term, trunc bool[N])
static ffi::Error EnvStepCuda(cudaStream_t stream, ffi::Span<const uint8_t> config,
                              int64_t autoreset, ffi::AnyBuffer state_in,
                              ffi::Buffer<ffi::S32> actions,
                              ffi::Result<ffi::AnyBuffer> state_out,
                              ffi::ResultBuffer<ffi::F32> obs, ffi::ResultBuffer<ffi::F32> rew,
                              ffi::ResultBuffer<ffi::PRED> term,
                              ffi::ResultBuffer<ffi::PRED> trunc) {
    EnvConfig cfg;
    ffi::Error err = read_config(config, &cfg);
    if (err.failure()) return err;
    int n = (int)(state_in.size_bytes() / sizeof(EnvState));
    if (state_in.size_bytes() != (size_t)n * sizeof(EnvState))
        return ffi::Error::InvalidArgument("state buffer is not [N, sizeof(EnvState)]");
    void* states = state_out->untyped_data();
    if (states != state_in.untyped_data()) {  // not aliased: copy, then step in place
        cudaError_t e = cudaMemcpyAsync(states, state_in.untyped_data(), state_in.size_bytes(),
                                        cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return cuda_error(e);
    }
    return cuda_error((cudaError_t)launch_env_step(
        &cfg, (EnvState*)states, actions.typed_data(), obs->typed_data(), rew->typed_data(),
        (uint8_t*)term->typed_data(), (uint8_t*)trunc->typed_data(), (int)autoreset, n,
        stream));
}

// () -> (state uint8[N, S], obs f32[N, 2, 88])
static ffi::Error EnvResetCuda(cudaStream_t stream, ffi::Result<ffi::AnyBuffer> state,
                               ffi::ResultBuffer<ffi::F32> obs) {
    int n = (int)(state->size_bytes() / sizeof(EnvState));
    return cuda_error(
        (cudaError_t)launch_env_reset((EnvState*)state->untyped_data(), obs->typed_data(), n,
                                      stream));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FootsiesEnvStepCuda, EnvStepCuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Span<const uint8_t>>("config")
                                  .Attr<int64_t>("autoreset")
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::PRED>>()
                                  .Ret<ffi::Buffer<ffi::PRED>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(FootsiesEnvResetCuda, EnvResetCuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::Buffer<ffi::F32>>());
#endif  // FOOTSIES_NO_FFI
