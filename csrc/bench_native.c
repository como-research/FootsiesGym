// Single-thread throughput of footsies.h without Python: env steps (game +
// rewards + observations), and the game frames alone, with random actions.
//
//   cc -O3 -ffp-contract=off -march=native csrc/bench_native.c -o bench_native
//   taskset -c 0 ./bench_native [num_envs] [seconds]

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "footsies.h"
#include "footsies_batch.h"

static const FrameTables T = FRAME_TABLES_INIT;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define POOL 64

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 4096;
    double seconds = argc > 2 ? atof(argv[2]) : 5.0;

    EnvConfig cfg = {0};
    cfg.frame_skip = 4;
    cfg.max_t = 10000;
    for (int i = 0; i < REWARD_TABLE_SIZE; i++) cfg.reward_table[i] = 0.0f;

    EnvState* s = calloc((size_t)n, sizeof(EnvState));
    float* obs = malloc((size_t)n * 2 * OBS_DIM * sizeof(float));
    float* rew = malloc((size_t)n * 2 * sizeof(float));
    uint8_t* term = malloc((size_t)n);
    uint8_t* trunc = malloc((size_t)n);
    int32_t* actions = malloc((size_t)POOL * n * 2 * sizeof(int32_t));
    srand(0);
    for (long i = 0; i < (long)POOL * n * 2; i++) actions[i] = rand() % 6;
    for (int i = 0; i < n; i++) env_fresh(&s[i]);

    // Env steps
    long steps = 0;
    double t0 = now(), t;
    do {
        const int32_t* a = actions + (size_t)(steps % POOL) * n * 2;
        for (int i = 0; i < n; i++)
            env_step(&s[i], &T, &cfg, a[2 * i], a[2 * i + 1], obs + (size_t)i * 2 * OBS_DIM,
                     rew + 2 * i, term + i, trunc + i, 1);
        steps++;
        t = now() - t0;
    } while (t < seconds);
    double env_sps = (double)steps * n / t;

    // Game frames only (no rewards, observations or delay)
    Game* g = calloc((size_t)n, sizeof(Game));
    for (int i = 0; i < n; i++) game_new(&g[i]);
    long frames = 0;
    t0 = now();
    do {
        const int32_t* a = actions + (size_t)(frames % POOL) * n * 2;
        for (int i = 0; i < n; i++) game_step(&g[i], &T, a[2 * i] + 1, a[2 * i + 1]);
        frames++;
        t = now() - t0;
    } while (t < seconds);
    double fps = (double)frames * n / t;

    // Observation encoding only
    long encodes = 0;
    t0 = now();
    do {
        for (int i = 0; i < n; i++) env_encode(&s[i], &T, obs + (size_t)i * 2 * OBS_DIM);
        encodes++;
        t = now() - t0;
    } while (t < seconds);
    double eps = (double)encodes * n / t;

    // SIMD batches (footsies_batch.h), same actions
    batch_init(&T);
    BatchEnv* b = batch_env_new(&cfg, n);
    float* bobs = aligned_alloc(64, (size_t)n * 2 * OBS_DIM * sizeof(float));
    for (int i = 0; i < b->nblocks; i++)
        for (int l = 0; l < BATCH_W; l++) lane_env_fresh(&b->blocks[i], l, 1, 0);
    long bsteps = 0;
    t0 = now();
    do {
        const int32_t* a = actions + (size_t)(bsteps % POOL) * n * 2;
        for (int i = 0; i < b->nblocks; i++) {
            size_t o = (size_t)i * BATCH_W;
            int count = n - (int)o < BATCH_W ? n - (int)o : BATCH_W;
            BLOCK_ENV_STEP(&b->blocks[i], &cfg, a + 2 * o, count, bobs + o * 2 * OBS_DIM,
                           rew + 2 * o, term + o, trunc + o, 1);
        }
        bsteps++;
        t = now() - t0;
    } while (t < seconds);
    double batch_sps = (double)bsteps * n / t;

    // Batched frames only
    for (int i = 0; i < b->nblocks; i++)
        for (int l = 0; l < BATCH_W; l++) lane_env_fresh(&b->blocks[i], l, 1, 0);
    long bframes = 0;
    BlockScratch scratch;
    t0 = now();
    do {
        const int32_t* a = actions + (size_t)(bframes % POOL) * n * 2;
        for (int i = 0; i < b->nblocks; i++) {
            for (int l = 0; l < BATCH_W; l++) {
                scratch.bits[0][l] = a[2 * (i * BATCH_W + l) % (2 * n)] + 1;
                scratch.bits[1][l] = a[(2 * (i * BATCH_W + l) + 1) % (2 * n)];
            }
            BLOCK_GAME_STEP_N(&b->blocks[i].g, &scratch, 1);
        }
        bframes++;
        t = now() - t0;
    } while (t < seconds);
    double bfps = (double)bframes * b->nblocks * BATCH_W / t;

#if FOOTSIES_AVX512
    // Breakdown: 4 frames on registers, and encode + transpose
    long b4 = 0;
    t0 = now();
    do {
        for (int i = 0; i < b->nblocks; i++) vblock_game_step_n(&b->blocks[i].g, &scratch, 4);
        b4++;
        t = now() - t0;
    } while (t < seconds);
    double step4 = t / ((double)b4 * b->nblocks * BATCH_W) * 1e9;
    long be = 0;
    t0 = now();
    do {
        for (int i = 0; i < b->nblocks; i++) {
            vblock_encode(&b->blocks[i], &scratch);
            vblock_write_obs(&scratch, bobs + (size_t)i * BATCH_W * 2 * OBS_DIM, BATCH_W);
        }
        be++;
        t = now() - t0;
    } while (t < seconds);
    double enc = t / ((double)be * b->nblocks * BATCH_W) * 1e9;
    printf("      per env: 4 frames %.1f ns, encode+write %.1f ns, whole step %.1f ns\n", step4, enc,
           1e9 / batch_sps);
#endif
    printf("N=%d  batched env-steps/s %.3gM, game frames/s %.3gM (%.0f ns/frame)\n", n,
           batch_sps / 1e6, bfps / 1e6, 1e9 / bfps);
    printf("N=%d  env-steps/s %.3gM  (frame_skip 4: %.3gM frames/s incl. env)\n", n,
           env_sps / 1e6, 4 * env_sps / 1e6);
    printf("      game frames/s %.3gM  (%.0f ns/frame)\n", fps / 1e6, 1e9 / fps);
    printf("      encodes/s %.3gM  (%.0f ns/encode, both players)\n", eps / 1e6, 1e9 / eps);
    return (int)(obs[0] * 0);  // keep the outputs live
}
