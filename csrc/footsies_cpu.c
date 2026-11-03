// Batched CPU entry points for footsies.h (OpenMP), called through ctypes
// by footsiesgym/c. Every function takes contiguous arrays of n games/envs.
//
// Build (see footsiesgym/c/build.py, which adds the same flags):
//   cc -O3 -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared
//      footsies_cpu.c -o libfootsies.so

#include "footsies.h"
#include "footsies_batch.h"

#include <string.h>

// Below this many envs, waking the OpenMP threads costs more than it saves.
#define OMP_MIN_ENVS 64

#ifdef _OPENMP
#include <omp.h>
#endif

#define EXPORT __attribute__((visibility("default")))

static const FrameTables HOST_TABLES = FRAME_TABLES_INIT;
static const int32_t ACTION_IDS[NUM_ACTIONS] = ACTION_IDS_INIT;

__attribute__((constructor)) static void init_batch_tables(void) {
    batch_init(&HOST_TABLES);
}

// ── Layout queries (the Python side checks its mirrors against these) ──

EXPORT int footsies_sizeof_game(void) { return (int)sizeof(Game); }
EXPORT int footsies_sizeof_env_state(void) { return (int)sizeof(EnvState); }
EXPORT int footsies_sizeof_env_config(void) { return (int)sizeof(EnvConfig); }
EXPORT int footsies_sizeof_raw_state(void) { return (int)sizeof(RawState); }
EXPORT int footsies_max_delay_steps(void) { return MAX_DELAY_STEPS; }
EXPORT int footsies_obs_dim(void) { return OBS_DIM; }

EXPORT void footsies_set_num_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

EXPORT int footsies_get_num_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// ── Single-movement check (exhaustive mono_move test) ──

EXPORT void footsies_mono_move(const float* pos, float* out, float speed, float sign,
                               int64_t n) {
#pragma omp parallel for schedule(static) if (n >= OMP_MIN_ENVS)
    for (int64_t i = 0; i < n; i++) out[i] = mono_move(pos[i], speed, sign);
}

// ── Games (FootsiesGame) ──

EXPORT void footsies_game_new(Game* g, int n) {
#pragma omp parallel for schedule(static) if (n >= OMP_MIN_ENVS)
    for (int i = 0; i < n; i++) game_new(&g[i]);
}

// mask may be NULL (reset every game)
EXPORT void footsies_game_reset(Game* g, const uint8_t* mask, int n) {
#pragma omp parallel for schedule(static) if (n >= OMP_MIN_ENVS)
    for (int i = 0; i < n; i++)
        if (!mask || mask[i]) game_reset(&g[i]);
}

// StepN for every game; n_frames = 1 is Step.
EXPORT void footsies_game_step(Game* g, const int32_t* p1_bits, const int32_t* p2_bits,
                               int n_frames, int n) {
#pragma omp parallel for schedule(static) if (n >= OMP_MIN_ENVS)
    for (int i = 0; i < n; i++)
        game_step_n(&g[i], &HOST_TABLES, p1_bits[i], p2_bits[i], n_frames);
}

EXPORT void footsies_game_raw(const Game* g, RawState* out, int n) {
#pragma omp parallel for schedule(static) if (n >= OMP_MIN_ENVS)
    for (int i = 0; i < n; i++) game_raw(&g[i], &HOST_TABLES, ACTION_IDS, &out[i]);
}

// ── Environments (FootsiesJaxEnv) ──

// Fresh episodes and their observations (obs: [n, 2, OBS_DIM]).
EXPORT void footsies_env_reset(EnvState* s, float* obs, int n) {
#pragma omp parallel for schedule(static) if (n >= OMP_MIN_ENVS)
    for (int i = 0; i < n; i++) {
        env_fresh(&s[i]);
        env_encode(&s[i], &HOST_TABLES, obs + (size_t)i * 2 * OBS_DIM);
    }
}

EXPORT void footsies_env_observe(const EnvState* s, float* obs, int n) {
#pragma omp parallel for schedule(static) if (n >= OMP_MIN_ENVS)
    for (int i = 0; i < n; i++) env_encode(&s[i], &HOST_TABLES, obs + (size_t)i * 2 * OBS_DIM);
}

// actions: [n, 2]; obs: [n, 2, OBS_DIM]; rew: [n, 2]; term, trunc: [n].
EXPORT void footsies_env_step(const EnvConfig* cfg, EnvState* s, const int32_t* actions,
                              float* obs, float* rew, uint8_t* term, uint8_t* trunc,
                              int autoreset, int n) {
    const EnvConfig c = *cfg;
#pragma omp parallel for schedule(static) if (n >= OMP_MIN_ENVS)
    for (int i = 0; i < n; i++)
        env_step(&s[i], &HOST_TABLES, &c, actions[2 * i], actions[2 * i + 1],
                 obs + (size_t)i * 2 * OBS_DIM, rew + 2 * i, term + i, trunc + i,
                 autoreset);
}

// Per-env bookkeeping fields, for tests: t, episode_id (each [n]).
EXPORT void footsies_env_info(const EnvState* s, int32_t* t, int32_t* episode_id, int n) {
    for (int i = 0; i < n; i++) {
        t[i] = s[i].t;
        episode_id[i] = s[i].episode_id;
    }
}

// The Game inside each EnvState (for raw-state comparisons).
EXPORT void footsies_env_games(const EnvState* s, Game* out, int n) {
    for (int i = 0; i < n; i++) out[i] = s[i].game;
}

// ── SIMD batches (footsies_batch.h): preallocated SoA blocks ──

EXPORT int footsies_batch_width(void) { return BATCH_W; }
EXPORT int footsies_batch_avx512(void) { return FOOTSIES_AVX512; }

EXPORT BatchEnv* footsies_batch_new(const EnvConfig* cfg, int n) { return batch_env_new(cfg, n); }

EXPORT void footsies_batch_free(BatchEnv* b) { batch_env_free(b); }

// Fresh episodes everywhere and their observations ([n, 2, OBS_DIM]).
EXPORT void footsies_batch_reset(BatchEnv* b, float* obs) {
#pragma omp parallel for schedule(static) if (b->nblocks >= 4)
    for (int i = 0; i < b->nblocks; i++) {
        EnvBlock* e = &b->blocks[i];
        int count = b->n - i * BATCH_W < BATCH_W ? b->n - i * BATCH_W : BATCH_W;
        BlockScratch s;
        LANES lane_env_fresh(e, l, 1, 0);
        block_encode(e, &s);
        block_write_obs(&s, obs + (size_t)i * BATCH_W * 2 * OBS_DIM, count);
    }
}

// Same arguments and outputs as footsies_env_step, on the batch's own state.
EXPORT void footsies_batch_step(BatchEnv* b, const int32_t* actions, float* obs, float* rew,
                                uint8_t* term, uint8_t* trunc, int autoreset) {
    const EnvConfig* cfg = &b->cfg;
#pragma omp parallel for schedule(static) if (b->nblocks >= 4)
    for (int i = 0; i < b->nblocks; i++) {
        size_t o = (size_t)i * BATCH_W;
        int count = b->n - (int)o < BATCH_W ? b->n - (int)o : BATCH_W;
        BLOCK_ENV_STEP(&b->blocks[i], cfg, actions + 2 * o, count, obs + o * 2 * OBS_DIM,
                       rew + 2 * o, term + o, trunc + o, autoreset);
    }
}

// StepN on the games only (input bits [n, 2]), for frame-level validation.
EXPORT void footsies_batch_game_step(BatchEnv* b, const int32_t* bits, int n_frames) {
#pragma omp parallel for schedule(static) if (b->nblocks >= 4)
    for (int i = 0; i < b->nblocks; i++) {
        size_t o = (size_t)i * BATCH_W;
        BlockScratch s;
        for (int l = 0; l < BATCH_W; l++) {
            int in = (int)o + l < b->n;
            s.bits[0][l] = in ? bits[2 * (o + l)] : 0;
            s.bits[1][l] = in ? bits[2 * (o + l) + 1] : 0;
        }
        BLOCK_GAME_STEP_N(&b->blocks[i].g, &s, n_frames);
    }
}

// The batch as footsies.h EnvStates ([n]), and back.
EXPORT void footsies_batch_get(const BatchEnv* b, EnvState* out) {
    for (int i = 0; i < b->n; i++) lane_to_env_state(&b->blocks[i / BATCH_W], i % BATCH_W, &out[i]);
}

EXPORT void footsies_batch_set(BatchEnv* b, const EnvState* in) {
    for (int i = 0; i < b->n; i++) lane_from_env_state(&b->blocks[i / BATCH_W], i % BATCH_W, &in[i]);
}

// ── Outputs for the caller ──

// Per-agent copies of a step's outputs (obs [n][2][OBS_DIM] -> p1 / p2
// [n][OBS_DIM], rew [n][2] -> p1 / p2 [n], done flags as is), in parallel:
// first-touching new arrays from many threads is what makes this fast.
EXPORT void footsies_split_outputs(const float* obs, const float* rew, const uint8_t* term,
                                   const uint8_t* trunc, float* obs1, float* obs2,
                                   float* rew1, float* rew2, uint8_t* term_out,
                                   uint8_t* trunc_out, int n) {
#pragma omp parallel for schedule(static) if (n >= OMP_MIN_ENVS)
    for (int i = 0; i < n; i++) {
        memcpy(obs1 + (size_t)i * OBS_DIM, obs + (size_t)i * 2 * OBS_DIM, OBS_DIM * sizeof(float));
        memcpy(obs2 + (size_t)i * OBS_DIM, obs + ((size_t)i * 2 + 1) * OBS_DIM,
               OBS_DIM * sizeof(float));
        if (rew1) {
            rew1[i] = rew[2 * i];
            rew2[i] = rew[2 * i + 1];
            term_out[i] = term[i];
            trunc_out[i] = trunc[i];
        }
    }
}
