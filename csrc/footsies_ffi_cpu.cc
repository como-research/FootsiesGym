// XLA FFI handlers for the CPU build of footsies.h, so that JAX programs can
// call the C environment on the host with the same signature as on the GPU
// (footsies_cuda.cu).
//
// Build (see footsiesgym/c/build.py, which adds the same flags):
//   c++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared
//     -I<jax.ffi.include_dir()> footsies_ffi_cpu.cc

#include <cstdint>
#include <cstring>

#include "footsies.h"

// Below this many envs, waking the OpenMP threads costs more than it saves.
#define OMP_MIN_ENVS 64
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static const FrameTables HOST_TABLES = FRAME_TABLES_INIT;

static ffi::Error read_config(ffi::Span<const uint8_t> bytes, EnvConfig* cfg) {
    if (bytes.size() != sizeof(EnvConfig))
        return ffi::Error::InvalidArgument("config has the wrong size");
    std::memcpy(cfg, bytes.begin(), sizeof(EnvConfig));
    return ffi::Error::Success();
}

// (state uint8[N, S], actions int32[N, 2]) ->
//   (state uint8[N, S] aliased, obs f32[N, 2, 88], rew f32[N, 2], term, trunc bool[N])
static ffi::Error EnvStepCpu(ffi::Span<const uint8_t> config, int64_t autoreset,
                             ffi::AnyBuffer state_in, ffi::Buffer<ffi::S32> actions,
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
    EnvState* states = (EnvState*)state_out->untyped_data();
    if ((void*)states != state_in.untyped_data())
        std::memcpy(states, state_in.untyped_data(), state_in.size_bytes());
    const int32_t* a = actions.typed_data();
    float* o = obs->typed_data();
    float* r = rew->typed_data();
    uint8_t* te = (uint8_t*)term->typed_data();
    uint8_t* tr = (uint8_t*)trunc->typed_data();
#pragma omp parallel for schedule(static) if (n >= OMP_MIN_ENVS)
    for (int i = 0; i < n; i++)
        env_step(&states[i], &HOST_TABLES, &cfg, a[2 * i], a[2 * i + 1],
                 o + (size_t)i * 2 * OBS_DIM, r + 2 * i, te + i, tr + i, (int)autoreset);
    return ffi::Error::Success();
}

// () -> (state uint8[N, S], obs f32[N, 2, 88])
static ffi::Error EnvResetCpu(ffi::Result<ffi::AnyBuffer> state,
                              ffi::ResultBuffer<ffi::F32> obs) {
    int n = (int)(state->size_bytes() / sizeof(EnvState));
    EnvState* states = (EnvState*)state->untyped_data();
    float* o = obs->typed_data();
#pragma omp parallel for schedule(static) if (n >= OMP_MIN_ENVS)
    for (int i = 0; i < n; i++) {
        env_fresh(&states[i]);
        env_encode(&states[i], &HOST_TABLES, o + (size_t)i * 2 * OBS_DIM);
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FootsiesEnvStepCpu, EnvStepCpu,
                              ffi::Ffi::Bind()
                                  .Attr<ffi::Span<const uint8_t>>("config")
                                  .Attr<int64_t>("autoreset")
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::PRED>>()
                                  .Ret<ffi::Buffer<ffi::PRED>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(FootsiesEnvResetCpu, EnvResetCpu,
                              ffi::Ffi::Bind()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::Buffer<ffi::F32>>());

extern "C" int footsies_ffi_sizeof_env_state(void) { return (int)sizeof(EnvState); }
extern "C" int footsies_ffi_offsetof_t(void) { return (int)offsetof(EnvState, t); }
extern "C" int footsies_ffi_offsetof_episode_id(void) {
    return (int)offsetof(EnvState, episode_id);
}
