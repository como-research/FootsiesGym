// SIMD-batched CPU implementation of footsies.h: blocks of BATCH_W games in
// struct-of-arrays layout, stepped together.
//
// Every function here is the branch-free form of its namesake in footsies.h
// (early returns become per-lane masks, "first active entry" scans become
// fixed-length loops), written as per-lane scalar code inside `omp simd`
// loops so the compiler vectorizes it across the BATCH_W lanes. The float
// operations are the same IEEE operations in the same order, so the result
// is byte-identical to footsies.h (testing/test_c_backend.py compares them
// after every step).
//
// State blocks are allocated once, 64-byte aligned (BatchEnv), and nothing
// is allocated while stepping.
//
// Compile with the footsies.h flags plus -fopenmp-simd (or -fopenmp) and a
// vector ISA, e.g. -march=native: -O3 -ffp-contract=off -march=native.

#ifndef FOOTSIES_BATCH_H
#define FOOTSIES_BATCH_H

#include <stdlib.h>
#include <string.h>

#include "footsies.h"

#define BATCH_W 16  // lanes per block: one AVX-512 register of 32-bit values

#define AI static inline __attribute__((always_inline))
#define LANES _Pragma("omp simd") for (int l = 0; l < BATCH_W; l++)
#define FIXED _Pragma("GCC unroll 64")

// ── Frame data as flat arrays (gathers) ──────────────────────────────────

typedef struct {
    int32_t fc[NUM_ACTIONS], loop_from[NUM_ACTIONS], is_loop[NUM_ACTIONS],
        canc[NUM_ACTIONS], type[NUM_ACTIONS], hit0[NUM_ACTIONS], nhit[NUM_ACTIONS],
        hurt0[NUM_ACTIONS], nhurt[NUM_ACTIONS], push0[NUM_ACTIONS], npush[NUM_ACTIONS],
        move0[NUM_ACTIONS], nmove[NUM_ACTIONS], cancel0[NUM_ACTIONS], ncancel[NUM_ACTIONS];
    // Entry arrays are padded so that start + k is always in bounds.
    int32_t hit_lo[N_HIT_ENTRIES + MAX_HITBOXES], hit_hi[N_HIT_ENTRIES + MAX_HITBOXES],
        hit_atk[N_HIT_ENTRIES + MAX_HITBOXES], hit_prox[N_HIT_ENTRIES + MAX_HITBOXES];
    float hit_x[N_HIT_ENTRIES + MAX_HITBOXES], hit_y[N_HIT_ENTRIES + MAX_HITBOXES],
        hit_w[N_HIT_ENTRIES + MAX_HITBOXES], hit_h[N_HIT_ENTRIES + MAX_HITBOXES];
    int32_t hurt_lo[N_HURT_ENTRIES + MAX_HURTBOXES], hurt_hi[N_HURT_ENTRIES + MAX_HURTBOXES];
    float hurt_x[N_HURT_ENTRIES + MAX_HURTBOXES], hurt_y[N_HURT_ENTRIES + MAX_HURTBOXES],
        hurt_w[N_HURT_ENTRIES + MAX_HURTBOXES], hurt_h[N_HURT_ENTRIES + MAX_HURTBOXES];
    int32_t push_lo[N_PUSH_ENTRIES + MAX_PUSHBOXES], push_hi[N_PUSH_ENTRIES + MAX_PUSHBOXES];
    float push_x[N_PUSH_ENTRIES + MAX_PUSHBOXES], push_y[N_PUSH_ENTRIES + MAX_PUSHBOXES],
        push_w[N_PUSH_ENTRIES + MAX_PUSHBOXES], push_h[N_PUSH_ENTRIES + MAX_PUSHBOXES];
    int32_t move_lo[N_MOVE_ENTRIES + MAX_MOVES], move_hi[N_MOVE_ENTRIES + MAX_MOVES];
    float move_v[N_MOVE_ENTRIES + MAX_MOVES];
    int32_t cancel_lo[N_CANCEL_ENTRIES + MAX_CANCELS], cancel_hi[N_CANCEL_ENTRIES + MAX_CANCELS],
        cancel_targets[N_CANCEL_ENTRIES + MAX_CANCELS];
    int32_t damage_action[NUM_ATTACKS], guard_action[NUM_ATTACKS], noh[NUM_ATTACKS],
        vd[NUM_ATTACKS], gd[NUM_ATTACKS], hs[NUM_ATTACKS], gs[NUM_ATTACKS], gbs[NUM_ATTACKS];
} BatchTables;

static BatchTables BT;

static void batch_tables_init(const FrameTables* T) {
    memset(&BT, 0, sizeof BT);
    for (int a = 0; a < NUM_ACTIONS; a++) {
        const ActionData* d = &T->actions[a];
        BT.fc[a] = d->frame_count;
        BT.loop_from[a] = d->loop_from;
        BT.is_loop[a] = d->is_loop;
        BT.canc[a] = d->always_cancelable;
        BT.type[a] = d->type;
        BT.hit0[a] = d->hit0;
        BT.nhit[a] = d->n_hit;
        BT.hurt0[a] = d->hurt0;
        BT.nhurt[a] = d->n_hurt;
        BT.push0[a] = d->push0;
        BT.npush[a] = d->n_push;
        BT.move0[a] = d->move0;
        BT.nmove[a] = d->n_move;
        BT.cancel0[a] = d->cancel0;
        BT.ncancel[a] = d->n_cancel;
    }
    for (int i = 0; i < N_HIT_ENTRIES; i++) {
        const HitboxEntry* e = &T->hit[i];
        BT.hit_lo[i] = e->lo, BT.hit_hi[i] = e->hi, BT.hit_atk[i] = e->attack;
        BT.hit_prox[i] = e->proximity;
        BT.hit_x[i] = e->x, BT.hit_y[i] = e->y, BT.hit_w[i] = e->w, BT.hit_h[i] = e->h;
    }
    for (int i = 0; i < N_HURT_ENTRIES; i++) {
        const RectEntry* e = &T->hurt[i];
        BT.hurt_lo[i] = e->lo, BT.hurt_hi[i] = e->hi;
        BT.hurt_x[i] = e->x, BT.hurt_y[i] = e->y, BT.hurt_w[i] = e->w, BT.hurt_h[i] = e->h;
    }
    for (int i = 0; i < N_PUSH_ENTRIES; i++) {
        const RectEntry* e = &T->push[i];
        BT.push_lo[i] = e->lo, BT.push_hi[i] = e->hi;
        BT.push_x[i] = e->x, BT.push_y[i] = e->y, BT.push_w[i] = e->w, BT.push_h[i] = e->h;
    }
    for (int i = 0; i < N_MOVE_ENTRIES; i++) {
        BT.move_lo[i] = T->move[i].lo, BT.move_hi[i] = T->move[i].hi;
        BT.move_v[i] = T->move[i].velocity;
    }
    for (int i = 0; i < N_CANCEL_ENTRIES; i++) {
        BT.cancel_lo[i] = T->cancel[i].lo, BT.cancel_hi[i] = T->cancel[i].hi;
        BT.cancel_targets[i] = (int32_t)T->cancel[i].targets;
    }
    for (int i = 0; i < NUM_ATTACKS; i++) {
        const AttackData* a = &T->attacks[i];
        BT.damage_action[i] = a->damage_action, BT.guard_action[i] = a->guard_action;
        BT.noh[i] = a->number_of_hit, BT.vd[i] = a->vital_damage, BT.gd[i] = a->guard_damage;
        BT.hs[i] = a->hit_stun, BT.gs[i] = a->guard_stun, BT.gbs[i] = a->guard_break_stun;
    }
}

// ── Blocks ───────────────────────────────────────────────────────────────

// Every row is 32-bit: the 64-bit input histories are split in lo/hi halves.
typedef struct {
    float pos_x[2][BATCH_W], vel_x[2][BATCH_W];
    int32_t vital[2][BATCH_W], guard[2][BATCH_W], action[2][BATCH_W],
        action_frame[2][BATCH_W], hit_count[2][BATCH_W], frame_adv[2][BATCH_W],
        hit_stun[2][BATCH_W], sprite_shake[2][BATCH_W], buffer_action[2][BATCH_W],
        reserve_damage_action[2][BATCH_W], face_right[2][BATCH_W], has_won[2][BATCH_W],
        input_backward[2][BATCH_W], reserve_prox_guard[2][BATCH_W],
        attack_run[2][BATCH_W];  // consecutive frames attack held (== ctz(~in_attack))
    int32_t round_state[BATCH_W], frame_count[BATCH_W], done[BATCH_W], reward[BATCH_W];
    uint32_t in_left_lo[2][BATCH_W], in_left_hi[2][BATCH_W], in_right_lo[2][BATCH_W],
        in_right_hi[2][BATCH_W], in_attack_lo[2][BATCH_W], in_attack_hi[2][BATCH_W];
} __attribute__((aligned(64))) GameBlock;

#define B_ALIGNED __attribute__((aligned(64)))

#define B_JOIN(g, name, p, l) (((uint64_t)(g)->name##_hi[p][l] << 32) | (g)->name##_lo[p][l])
#define B_SPLIT(g, name, p, l, v) \
    ((g)->name##_lo[p][l] = (uint32_t)(v), (g)->name##_hi[p][l] = (uint32_t)((v) >> 32))

typedef struct {
    GameBlock g;
    int32_t t[BATCH_W], episode_id[BATCH_W], queue_head[BATCH_W], budget[2][BATCH_W],
        last_guard[2][BATCH_W], queue[2][MAX_DELAY_STEPS][BATCH_W], prev_selected[2][BATCH_W],
        holding[2][BATCH_W];
} __attribute__((aligned(64))) EnvBlock;

// Per-step scratch of one block (lives on the stack of the stepping thread).
typedef struct {
    int32_t bits[2][BATCH_W], running[BATCH_W];
    float obs[2 * OBS_DIM][BATCH_W];
} __attribute__((aligned(64))) BlockScratch;

#if defined(__AVX512F__) && !defined(FOOTSIES_NO_AVX512)
#define FOOTSIES_AVX512 1
static void vblock_game_step_n(GameBlock* g, const BlockScratch* s, int n_frames);
#define BLOCK_GAME_STEP_N vblock_game_step_n
#define BLOCK_ENV_STEP vblock_env_step
#else
#define FOOTSIES_AVX512 0
#define BLOCK_GAME_STEP_N block_game_step_n
#define BLOCK_ENV_STEP block_env_step
#endif

// ── One fighter in one lane ──────────────────────────────────────────────

typedef struct {
    float pos, vel;
    int32_t vital, guard, action, af, hit, adv, hs, shake, buf, rdmg, face, won, inback,
        rprox, run;
    uint64_t left, right, attack;
} LF;

AI LF lf_load(const GameBlock* g, int p, int l) {
    LF f;
    f.pos = g->pos_x[p][l], f.vel = g->vel_x[p][l];
    f.vital = g->vital[p][l], f.guard = g->guard[p][l], f.action = g->action[p][l];
    f.af = g->action_frame[p][l], f.hit = g->hit_count[p][l], f.adv = g->frame_adv[p][l];
    f.hs = g->hit_stun[p][l], f.shake = g->sprite_shake[p][l], f.buf = g->buffer_action[p][l];
    f.rdmg = g->reserve_damage_action[p][l], f.face = g->face_right[p][l];
    f.won = g->has_won[p][l], f.inback = g->input_backward[p][l];
    f.rprox = g->reserve_prox_guard[p][l], f.run = g->attack_run[p][l];
    f.left = B_JOIN(g, in_left, p, l), f.right = B_JOIN(g, in_right, p, l);
    f.attack = B_JOIN(g, in_attack, p, l);
    return f;
}

AI void lf_store(GameBlock* g, int p, int l, const LF* f) {
    g->pos_x[p][l] = f->pos, g->vel_x[p][l] = f->vel;
    g->vital[p][l] = f->vital, g->guard[p][l] = f->guard, g->action[p][l] = f->action;
    g->action_frame[p][l] = f->af, g->hit_count[p][l] = f->hit, g->frame_adv[p][l] = f->adv;
    g->hit_stun[p][l] = f->hs, g->sprite_shake[p][l] = f->shake, g->buffer_action[p][l] = f->buf;
    g->reserve_damage_action[p][l] = f->rdmg, g->face_right[p][l] = f->face;
    g->has_won[p][l] = f->won, g->input_backward[p][l] = f->inback;
    g->reserve_prox_guard[p][l] = f->rprox, g->attack_run[p][l] = f->run;
    B_SPLIT(g, in_left, p, l, f->left), B_SPLIT(g, in_right, p, l, f->right);
    B_SPLIT(g, in_attack, p, l, f->attack);
}

AI int b_clamp(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
AI int b_frame(const LF* f) { return b_clamp(f->af, 0, NUM_FRAMES - 1); }
AI int b_in(int lo, int hi, int fr) { return (lo <= fr) & (fr <= hi); }

// set_current_action where en
AI void lf_set_action(LF* f, int en, int a) {
    f->action = en ? a : f->action;
    f->af = en ? 0 : f->af;
    f->hit = en ? 0 : f->hit;
    f->buf = en ? -1 : f->buf;
    f->rdmg = en ? -1 : f->rdmg;
    f->shake = en ? 0 : f->shake;
}

AI int b_cancel_buffer(int a, int fr, int target) {
    int r = 0;
    FIXED for (int k = 0; k < MAX_CANCELS; k++) {
        int i = BT.cancel0[a] + k;
        r |= (k < BT.ncancel[a]) & b_in(BT.cancel_lo[i], BT.cancel_hi[i], fr) &
             ((BT.cancel_targets[i] >> target) & 1);
    }
    return r;
}

// request_action where en
AI void lf_request(LF* f, int en, int a) {
    int end = f->af >= BT.fc[f->action];
    int same = f->action == a;
    int canc = BT.canc[f->action];
    int start = en & (end | (!same & canc));
    int buffer = en & !end & !same & !canc & b_cancel_buffer(f->action, b_frame(f), a);
    lf_set_action(f, start, a);
    f->buf = buffer ? a : f->buf;
}

AI void lf_update_input(LF* f, int bits) {
    f->left = (f->left << 1) | (uint64_t)((bits & INPUT_LEFT) != 0);
    f->right = (f->right << 1) | (uint64_t)((bits & INPUT_RIGHT) != 0);
    f->attack = (f->attack << 1) | (uint64_t)((bits & INPUT_ATTACK) != 0);
    int held = (bits & INPUT_ATTACK) != 0;
    f->run = held ? (f->run < 64 ? f->run + 1 : 64) : 0;
}

// Table loads below are unconditional (indices are always in bounds) and
// selected afterwards: a conditional load would stop if-conversion.
AI int lf_frames_left(const LF* f) {
    int canc = BT.canc[f->action], fc = BT.fc[f->action];
    return canc ? 0 : fc - f->af;
}

AI void lf_increment_action_frame(LF* f) {
    int neg = -f->shake;
    int shook = neg + (neg > 0 ? -1 : 1);
    f->shake = f->shake != 0 ? shook : f->shake;
    int stun = f->hs > 0;
    int frame = f->af + 1;
    int loop_from = BT.loop_from[f->action];
    int loop = (frame >= BT.fc[f->action]) & BT.is_loop[f->action];
    frame = loop ? loop_from : frame;
    f->hs = stun ? f->hs - 1 : f->hs;
    f->af = stun ? f->af : frame;
}

// dash_check without early exits
AI int b_dash(uint64_t toward, uint64_t away, int first, int last) {
    uint64_t neutral = ~(toward | away);
    int result = 0, decided = 0;
    FIXED for (int i = first; i < last; i++) {
        int a = (int)((away >> i) & 1u), t = (int)((toward >> i) & 1u);
        int window = (neutral & (((1ULL << (DASH_ALLOW_FRAME - 1)) - 1) << (i + 1))) != 0;
        result = decided ? result : (a ? 0 : (t ? window : 0));
        decided |= a | t;
    }
    return result;
}

AI int b_pressed(uint64_t h) { return (int)(h & 1u) & (int)(((h >> 1) & 1u) ^ 1u); }

AI void lf_update_action_request(LF* f) {
    int won = f->won != 0;
    int use_res = !won & (f->rdmg != -1) & (f->hs <= 0);
    int use_buf = !won & !use_res & (f->buf != -1) & (CAN_CANCEL_ON_WHIFF | (f->hit > 0)) &
                  (f->hs <= 0);
    int normal = !won & !use_res & !use_buf;
    int res_a = f->rdmg, buf_a = f->buf;

    // Read before any request (as in footsies.h)
    uint64_t fwd = f->face ? f->right : f->left;
    uint64_t bwd = f->face ? f->left : f->right;
    int is_f = (int)(fwd & 1u), is_b = (int)(bwd & 1u);
    int directional = is_f | is_b;
    int special = (int)((f->attack & 1u) ^ 1u) & ((f->attack & HELD_SINCE_1) == HELD_SINCE_1);
    int attack = b_pressed(f->attack);
    int in_normal = ((f->action == ACT_N_ATTACK) | (f->action == ACT_B_ATTACK)) &
                    !(f->af >= BT.fc[f->action]);
    int normal_a = in_normal ? ACT_N_SPECIAL : (directional ? ACT_B_ATTACK : ACT_N_ATTACK);
    int fdash = b_pressed(fwd) & b_dash(fwd, bwd, 1, DASH_ALLOW_FRAME);
    int bdash = b_pressed(bwd) & b_dash(bwd, fwd, 1, DASH_ALLOW_FRAME);
    int move = (is_f & !is_b) ? ACT_FORWARD
               : (is_b & !is_f) ? (f->rprox ? ACT_GUARD_PROXIMITY : ACT_BACKWARD)
                                : ACT_STAND;

    // The early-return paths (each touches only its own lanes)
    lf_request(f, won, ACT_WIN);
    lf_set_action(f, use_res, res_a);
    lf_set_action(f, use_buf, buf_a);

    lf_request(f, normal & special, directional ? ACT_B_SPECIAL : ACT_N_SPECIAL);
    lf_request(f, normal & !special & attack, normal_a);
    lf_request(f, normal & fdash, ACT_DASH_FORWARD);
    lf_request(f, normal & !fdash & bdash, ACT_DASH_BACKWARD);
    f->inback = normal ? is_b : f->inback;
    lf_request(f, normal, move);
    f->rprox = normal ? 0 : f->rprox;
}

AI void lf_update_movement(LF* f) {
    int a = f->action, fr = b_frame(f);
    int has = 0;
    float v = 0.0f;
    FIXED for (int k = 0; k < MAX_MOVES; k++) {  // the first active entry
        int i = BT.move0[a] + k;
        int ok = (k < BT.nmove[a]) & b_in(BT.move_lo[i], BT.move_hi[i], fr) & !has;
        float mv = BT.move_v[i];
        v = ok ? mv : v;
        has |= ok;
    }
    int fwd = a == ACT_FORWARD, bwd = a == ACT_BACKWARD;
    int by_data = !fwd & !bwd & has;
    int moves = fwd | bwd | (by_data & (v != 0.0f));
    float speed = fwd ? FORWARD_MOVE_SPEED : (bwd ? -BACKWARD_MOVE_SPEED : v);
    float sign = f->face ? 1.0f : -1.0f;
    float moved = mono_move(f->pos, speed, sign);
    int active = f->hs <= 0;
    f->pos = (active & moves) ? moved : f->pos;
    f->vel = (active & by_data) ? v : f->vel;
}

typedef struct {
    float hx[MAX_HITBOXES], hy[MAX_HITBOXES], hw[MAX_HITBOXES], hh[MAX_HITBOXES];
    int32_t hvalid[MAX_HITBOXES], hatk[MAX_HITBOXES], hprox[MAX_HITBOXES];
    float ux[MAX_HURTBOXES], uy[MAX_HURTBOXES], uw[MAX_HURTBOXES], uh[MAX_HURTBOXES];
    int32_t uvalid[MAX_HURTBOXES];
    float px, py, pw, ph;
} LB;

AI void lf_boxes(const LF* f, LB* b) {
    int a = f->action, fr = b_frame(f);
    float sign = f->face ? 1.0f : -1.0f;
    FIXED for (int k = 0; k < MAX_HITBOXES; k++) {
        int i = BT.hit0[a] + k;
        int valid = (k < BT.nhit[a]) & b_in(BT.hit_lo[i], BT.hit_hi[i], fr);
        float x = BT.hit_x[i], y = BT.hit_y[i], w = BT.hit_w[i], h = BT.hit_h[i];
        int atk = BT.hit_atk[i], prox = BT.hit_prox[i];
        b->hvalid[k] = valid;
        b->hx[k] = f->pos + (valid ? x : 0.0f) * sign;
        b->hy[k] = valid ? y : 0.0f;
        b->hw[k] = valid ? w : 0.0f;
        b->hh[k] = valid ? h : 0.0f;
        b->hatk[k] = valid ? atk : 0;
        b->hprox[k] = valid ? prox : 0;
    }
    FIXED for (int k = 0; k < MAX_HURTBOXES; k++) {
        int i = BT.hurt0[a] + k;
        int valid = (k < BT.nhurt[a]) & b_in(BT.hurt_lo[i], BT.hurt_hi[i], fr);
        float x = BT.hurt_x[i], y = BT.hurt_y[i], w = BT.hurt_w[i], h = BT.hurt_h[i];
        b->uvalid[k] = valid;
        b->ux[k] = f->pos + (valid ? x : 0.0f) * sign;
        b->uy[k] = valid ? y : 0.0f;
        b->uw[k] = valid ? w : 0.0f;
        b->uh[k] = valid ? h : 0.0f;
    }
    int has = 0;
    float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;
    FIXED for (int k = 0; k < MAX_PUSHBOXES; k++) {  // the first active entry
        int i = BT.push0[a] + k;
        int ok = (k < BT.npush[a]) & b_in(BT.push_lo[i], BT.push_hi[i], fr) & !has;
        float px = BT.push_x[i], py = BT.push_y[i], pw = BT.push_w[i], ph = BT.push_h[i];
        x = ok ? px : x, y = ok ? py : y;
        w = ok ? pw : w, h = ok ? ph : h;
        has |= ok;
    }
    b->px = f->pos + x * sign, b->py = y, b->pw = w, b->ph = h;
}

AI void lf_shift(LF* f, LB* b, int en, float dx) {
    f->pos = en ? f->pos + dx : f->pos;
    FIXED for (int k = 0; k < MAX_HITBOXES; k++) b->hx[k] = en ? b->hx[k] + dx : b->hx[k];
    FIXED for (int k = 0; k < MAX_HURTBOXES; k++) b->ux[k] = en ? b->ux[k] + dx : b->ux[k];
    b->px = en ? b->px + dx : b->px;
}

AI void b_push_character_vs_character(LF* f0, LF* f1, LB* b0, LB* b1) {
    float x1 = b0->px, y1 = b0->py, w1 = b0->pw, h1 = b0->ph;
    float x2 = b1->px, y2 = b1->py, w2 = b1->pw, h2 = b1->ph;
    int overlaps = (x2 + w2 > x1) & (x2 < x1 + w1) & (y2 + h2 > y1) & (y2 < y1 + h1);
    int p1_left = overlaps & (f0->pos < f1->pos);
    int p1_right = overlaps & (f0->pos > f1->pos);
    float left_overlap = (x1 + w1) - x2;
    float right_overlap = (x2 + w2) - x1;
    float dx0 = p1_left ? -left_overlap * 0.5f : right_overlap * 0.5f;
    float dx1 = p1_left ? left_overlap * 0.5f : -right_overlap * 0.5f;
    int apply = p1_left | p1_right;
    lf_shift(f0, b0, apply, dx0);
    lf_shift(f1, b1, apply, dx1);
}

AI void b_push_vs_background(LF* f, LB* b) {
    float x_min = b->px - b->pw * 0.5f, x_max = b->px + b->pw * 0.5f;
    int left = x_min < STAGE_MIN_X;
    int right = !left & (x_max > STAGE_MAX_X);
    float dx = left ? STAGE_MIN_X - x_min : STAGE_MAX_X - x_max;
    lf_shift(f, b, left | right, dx);
}

AI int b_overlap(float ax, float ay, float aw, float ah, float bx, float by, float bw,
                 float bh) {
    float a_x_min = ax - aw * 0.5f, a_x_max = ax + aw * 0.5f;
    float b_x_min = bx - bw * 0.5f, b_x_max = bx + bw * 0.5f;
    float a_y_min = ay, a_y_max = ay + ah;
    float b_y_min = by, b_y_max = by + bh;
    return (b_x_max >= a_x_min) & (b_x_min <= a_x_max) & (b_y_max >= a_y_min) &
           (b_y_min <= a_y_max);
}

AI void b_check_attack(LF* att, LF* dmg, const LB* ab, const LB* db) {
    int is_hit = 0, attack = 0, prox = 0;
    FIXED for (int k = 0; k < MAX_HITBOXES; k++) {
        int valid = ab->hvalid[k] & (att->hit < BT.noh[ab->hatk[k]]);
        int touching = 0;
        FIXED for (int j = 0; j < MAX_HURTBOXES; j++)
            touching |= db->uvalid[j] & b_overlap(ab->hx[k], ab->hy[k], ab->hw[k], ab->hh[k],
                                                  db->ux[j], db->uy[j], db->uw[j], db->uh[j]);
        int t = valid & touching;
        prox |= t & ab->hprox[k];
        int first = t & !ab->hprox[k] & !is_hit;  // the first damaging box in asset order
        int atk = ab->hatk[k];
        attack = first ? atk : attack;
        is_hit |= first;
    }

    // NotifyDamaged where is_hit
    int gd = BT.gd[attack];
    int guard1 = gd > 0 ? dmg->guard - gd : dmg->guard;
    int guard_break = (gd > 0) & (guard1 < 0);
    int guard2 = guard_break ? 0 : guard1;
    int guarding = (dmg->action == ACT_BACKWARD) | (BT.type[dmg->action] == ACTION_TYPE_GUARD);
    int vd = BT.vd[attack];
    int vital1 = dmg->vital - vd;
    int vital2 = vd > 0 ? (vital1 <= 0 ? 0 : vital1) : dmg->vital;
    int guard_action = BT.guard_action[attack], damage_action = BT.damage_action[attack];
    int gs = BT.gs[attack], gbs = BT.gbs[attack], hs = BT.hs[attack];
    int new_action = guarding ? guard_action : damage_action;
    int result = guarding ? (guard_break ? GUARD_BREAK_RESULT : GUARD_RESULT) : DAMAGE_RESULT;
    int stun = result == GUARD_RESULT ? gs : (result == GUARD_BREAK_RESULT ? gbs : hs);

    dmg->guard = is_hit ? guard2 : dmg->guard;
    lf_set_action(dmg, is_hit, new_action);
    dmg->rdmg = (is_hit & guarding & guard_break) ? ACT_GUARD_BREAK : dmg->rdmg;
    dmg->vital = (is_hit & !guarding) ? vital2 : dmg->vital;
    dmg->hs = is_hit ? stun : dmg->hs;
    int shake = stun / 3;
    shake = shake > MAX_SPRITE_SHAKE_FRAME ? MAX_SPRITE_SHAKE_FRAME : shake;
    shake = shake * (dmg->face ? -1 : 1);
    dmg->shake = is_hit ? shake : dmg->shake;
    att->hit = is_hit ? att->hit + 1 : att->hit;
    att->hs = is_hit ? stun : att->hs;
    // NotifyInProximityGuardRange
    dmg->rprox = (!is_hit & prox) ? (dmg->rprox | dmg->inback) : dmg->rprox;
}

AI void lf_setup_battle_start(LF* f, int en, float start_x, int face_right) {
    f->pos = en ? start_x : f->pos;
    f->face = en ? face_right : f->face;
    f->vital = en ? 1 : f->vital;
    f->guard = en ? START_GUARD_HEALTH : f->guard;
    f->won = en ? 0 : f->won;
    f->vel = en ? 0.0f : f->vel;
    f->left = en ? 0 : f->left;
    f->right = en ? 0 : f->right;
    f->attack = en ? 0 : f->attack;
    f->run = en ? 0 : f->run;
    lf_set_action(f, en, ACT_STAND);
}

// game_step for lane l of a block, where `running` (else the lane is unchanged)
AI void lane_game_step(GameBlock* g, int l, int bits0, int bits1, int running) {
    LF f0 = lf_load(g, 0, l), f1 = lf_load(g, 1, l);
    LF o0 = f0, o1 = f1;
    int round_state = g->round_state[l], frame_count = g->frame_count[l];
    int done = g->done[l], reward = g->reward[l];

    int reset = round_state != ROUND_FIGHT;
    lf_setup_battle_start(&f0, reset, -2.0f, 1);
    lf_setup_battle_start(&f1, reset, 2.0f, 0);
    frame_count = reset ? -1 : frame_count;

    lf_update_input(&f0, bits0);
    lf_update_input(&f1, bits1);
    int l0 = lf_frames_left(&f0), l1 = lf_frames_left(&f1);
    f0.adv = l1 - l0;
    f1.adv = l0 - l1;
    lf_increment_action_frame(&f0);
    lf_increment_action_frame(&f1);
    lf_update_action_request(&f0);
    lf_update_action_request(&f1);
    lf_update_movement(&f0);
    lf_update_movement(&f1);

    LB b0, b1;
    lf_boxes(&f0, &b0);
    lf_boxes(&f1, &b1);
    b_push_character_vs_character(&f0, &f1, &b0, &b1);
    b_push_vs_background(&f0, &b0);
    b_push_vs_background(&f1, &b1);
    b_check_attack(&f0, &f1, &b0, &b1);
    b_check_attack(&f1, &f0, &b1, &b0);

    int d0 = f0.vital <= 0, d1 = f1.vital <= 0;
    int new_done = d0 | d1;
    int new_reward = (d1 & !d0) - (d0 & !d1);

#define KEEP(field) f0.field = running ? f0.field : o0.field, f1.field = running ? f1.field : o1.field
    KEEP(pos), KEEP(vel), KEEP(vital), KEEP(guard), KEEP(action), KEEP(af), KEEP(hit);
    KEEP(adv), KEEP(hs), KEEP(shake), KEEP(buf), KEEP(rdmg), KEEP(face), KEEP(won);
    KEEP(inback), KEEP(rprox), KEEP(run), KEEP(left), KEEP(right), KEEP(attack);
#undef KEEP
    lf_store(g, 0, l, &f0);
    lf_store(g, 1, l, &f1);
    g->done[l] = running ? new_done : done;
    g->reward[l] = running ? new_reward : reward;
    g->round_state[l] = running ? (new_done ? ROUND_KO : ROUND_FIGHT) : round_state;
    g->frame_count[l] = running ? frame_count + 1 : g->frame_count[l];
}

// ── Blocks: frames ───────────────────────────────────────────────────────

// StepN on one block: inputs bits[p][l], stopping each lane at its KO.
static inline void block_game_step_n(GameBlock* g, BlockScratch* s, int n_frames) {
    LANES s->running[l] = 1;
    for (int k = 0; k < n_frames; k++) {
        LANES {
            lane_game_step(g, l, s->bits[0][l], s->bits[1][l], s->running[l]);
            s->running[l] = s->running[l] & !g->done[l];
        }
    }
}

// ── Blocks: env ──────────────────────────────────────────────────────────

AI void lane_env_fresh(EnvBlock* e, int l, int en, int episode_id) {
    GameBlock* g = &e->g;
    FIXED for (int p = 0; p < 2; p++) {
        LF f = lf_load(g, p, l);
        // fighter_new, then setup_battle_start (game_new)
        f.adv = en ? 0 : f.adv;
        f.hs = en ? 0 : f.hs;
        f.inback = en ? 0 : f.inback;
        f.rprox = en ? 0 : f.rprox;
        lf_setup_battle_start(&f, en, p == 0 ? -2.0f : 2.0f, p == 0);
        lf_store(g, p, l, &f);
        e->budget[p][l] = en ? 0 : e->budget[p][l];
        e->last_guard[p][l] = en ? START_GUARD_HEALTH : e->last_guard[p][l];
        FIXED for (int k = 0; k < MAX_DELAY_STEPS; k++)
            e->queue[p][k][l] = en ? 0 : e->queue[p][k][l];
        e->prev_selected[p][l] = en ? 0 : e->prev_selected[p][l];
        e->holding[p][l] = en ? 0 : e->holding[p][l];
    }
    g->round_state[l] = en ? ROUND_FIGHT : g->round_state[l];
    g->frame_count[l] = en ? -1 : g->frame_count[l];
    g->done[l] = en ? 0 : g->done[l];
    g->reward[l] = en ? 0 : g->reward[l];
    e->t[l] = en ? 0 : e->t[l];
    e->episode_id[l] = en ? episode_id : e->episode_id[l];
    e->queue_head[l] = en ? 0 : e->queue_head[l];
}

AI int lane_resolve_special_charge(EnvBlock* e, int p, int l, int action) {
    int is_special = (action == ENV_SPECIAL_CHARGE) | (action == ENV_FORWARD_SPECIAL_CHARGE) |
                     (action == ENV_BACK_SPECIAL_CHARGE);
    int holding = e->holding[p][l] ^ is_special;
    e->holding[p][l] = holding;
    int base = action == ENV_SPECIAL_CHARGE           ? ENV_NONE
               : action == ENV_FORWARD_SPECIAL_CHARGE ? ENV_FORWARD
               : action == ENV_BACK_SPECIAL_CHARGE    ? ENV_BACK
                                                      : action;
    return holding ? lut4(CHARGE_ACTION_LUT, b_clamp(base, 0, 5)) : base;
}

AI float b_int_feature(int k, float d) { return (float)b_clamp(k, -1024, 1024) / d; }

// 37 public features of fighter p of lane l into o[f][l]
AI void lane_encode_public(const GameBlock* g, int p, int l, float (*o)[BATCH_W]) {
    int action = g->action[p][l], frame = g->action_frame[p][l], count = BT.fc[action];
    float vel = g->vel_x[p][l];
    int vital = g->vital[p][l], guard = g->guard[p][l], hs = g->hit_stun[p][l];
    o[0][l] = g->pos_x[p][l] * 0.25f;
    o[1][l] = vel == 0.0f ? 0.0f : vel / 5.0f;
    o[2][l] = vital <= 0 ? 1.0f : 0.0f;
    o[3][l] = (float)vital;
    FIXED for (int k = 0; k < 4; k++) o[4 + k][l] = guard == k ? 1.0f : 0.0f;
    FIXED for (int k = 0; k < NUM_ACTIONS; k++) o[8 + k][l] = action == k ? 1.0f : 0.0f;
    o[25][l] = b_int_feature(frame, 25.0f);
    o[26][l] = b_int_feature(count, 25.0f);
    o[27][l] = b_int_feature(count - frame, 25.0f);
    o[28][l] = frame >= count ? 1.0f : 0.0f;
    o[29][l] = BT.canc[action] ? 1.0f : 0.0f;
    o[30][l] = (float)g->hit_count[p][l];
    o[31][l] = b_int_feature(hs, 10.0f);
    o[32][l] = hs > 0 ? 1.0f : 0.0f;
    o[33][l] = (float)g->sprite_shake[p][l];
    o[34][l] = b_int_feature(MAX_SPRITE_SHAKE_FRAME, 10.0f);
    o[35][l] = g->face_right[p][l] ? 1.0f : 0.0f;
    o[36][l] = b_int_feature(g->frame_adv[p][l], 10.0f);
}

AI void lane_encode_private(const EnvBlock* e, int p, int l, float (*o)[BATCH_W]) {
    const GameBlock* g = &e->g;
    uint64_t left = B_JOIN(g, in_left, p, l), right = B_JOIN(g, in_right, p, l);
    uint64_t fwd = g->face_right[p][l] ? right : left;
    uint64_t bwd = g->face_right[p][l] ? left : right;
    o[0][l] = b_dash(fwd, bwd, 0, DASH_ALLOW_FRAME - 1) ? 1.0f : 0.0f;
    o[1][l] = b_dash(bwd, fwd, 0, DASH_ALLOW_FRAME - 1) ? 1.0f : 0.0f;
    int run = g->attack_run[p][l];
    run = run > SPECIAL_ATTACK_HOLD_FRAME ? SPECIAL_ATTACK_HOLD_FRAME : run;
    float progress = (float)run / (float)SPECIAL_ATTACK_HOLD_FRAME;
    o[2][l] = progress < 1.0f ? progress : 1.0f;
    int prev = e->prev_selected[p][l];
    FIXED for (int k = 0; k < NUM_ENV_ACTIONS; k++) o[3 + k][l] = prev == k ? 1.0f : 0.0f;
    o[12][l] = e->holding[p][l] ? 1.0f : 0.0f;
}

static inline void block_encode(const EnvBlock* e, BlockScratch* s) {
    const GameBlock* g = &e->g;
    LANES {
        float distance = fabsf(g->pos_x[0][l] - g->pos_x[1][l]) * 0.125f;
        s->obs[0][l] = distance;
        s->obs[OBS_DIM][l] = distance;
    }
    LANES {
        lane_encode_public(g, 0, l, &s->obs[1]);
        lane_encode_public(g, 1, l, &s->obs[OBS_DIM + 1]);
    }
    LANES {
        lane_encode_private(e, 0, l, &s->obs[1 + PUBLIC_DIM]);
        lane_encode_private(e, 1, l, &s->obs[OBS_DIM + 1 + PUBLIC_DIM]);
    }
    // Opponent public features are the other player's own
    for (int f = 0; f < PUBLIC_DIM; f++) {
        LANES {
            s->obs[1 + PUBLIC_DIM + PRIVATE_DIM + f][l] = s->obs[OBS_DIM + 1 + f][l];
            s->obs[OBS_DIM + 1 + PUBLIC_DIM + PRIVATE_DIM + f][l] = s->obs[1 + f][l];
        }
    }
}

// Transpose the block's [feature][lane] observations into obs[lane][feature].
static inline void block_write_obs(const BlockScratch* s, float* obs, int count) {
    for (int l = 0; l < count; l++) {
        float* o = obs + (size_t)l * 2 * OBS_DIM;
        for (int f = 0; f < 2 * OBS_DIM; f++) o[f] = s->obs[f][l];
    }
}

// env_step on one block. actions: [count, 2]; outputs for `count` lanes.
static inline void block_env_step(EnvBlock* e, const EnvConfig* cfg, const int32_t* actions,
                                  int count, float* obs, float* rew, uint8_t* term,
                                  uint8_t* trunc, int autoreset) {
    BlockScratch s;
    int32_t a[2][BATCH_W];
    for (int l = 0; l < BATCH_W; l++) {  // lanes past `count` idle on action 0
        a[0][l] = l < count ? actions[2 * l] : 0;
        a[1][l] = l < count ? actions[2 * l + 1] : 0;
    }
    int delay = cfg->action_delay_frames != 0, steps = delay ? delay_steps(cfg) : 1;
    LANES {
        int x0 = a[0][l], x1 = a[1][l];
        int head = e->queue_head[l];
        FIXED for (int k = 0; k < MAX_DELAY_STEPS; k++) {
            int here = delay & (k == head);
            int q0 = e->queue[0][k][l], q1 = e->queue[1][k][l];
            e->queue[0][k][l] = here ? (int32_t)(int8_t)a[0][l] : q0;
            e->queue[1][k][l] = here ? (int32_t)(int8_t)a[1][l] : q1;
            x0 = here ? q0 : x0;
            x1 = here ? q1 : x1;
        }
        e->queue_head[l] = delay ? (head + 1) % steps : head;
        int base0 = lane_resolve_special_charge(e, 0, l, x0);
        int base1 = lane_resolve_special_charge(e, 1, l, x1);
        s.bits[0][l] = lut4(P1_ENV_TO_BITS, b_clamp(base0, 0, 5));
        s.bits[1][l] = lut4(P2_ENV_TO_BITS, b_clamp(base1, 0, 5));
    }
    BLOCK_GAME_STEP_N(&e->g, &s, cfg->frame_skip);

    float r0[BATCH_W], r1[BATCH_W];
    int32_t te[BATCH_W], tr[BATCH_W];
    LANES {
        GameBlock* g = &e->g;
        int t = e->t[l] + 1;
        e->t[l] = t;
        int dead0 = g->vital[0][l] <= 0, dead1 = g->vital[1][l] <= 0;
        int gb_on = cfg->guard_break_reward_nonzero != 0;
        int broken0 = gb_on & (g->guard[0][l] < e->last_guard[0][l]);
        int broken1 = gb_on & (g->guard[1][l] < e->last_guard[1][l]);
        int budget_on = cfg->use_reward_budget != 0;
        e->budget[0][l] += budget_on ? broken1 : 0;
        e->budget[1][l] += budget_on ? broken0 : 0;
        int k0 = b_clamp(e->budget[0][l], 0, MAX_BUDGET_DEDUCTIONS);
        int k1 = b_clamp(e->budget[1][l], 0, MAX_BUDGET_DEDUCTIONS);
        int idx = ((((((broken1 * 2 + broken0) * 2 + dead1) * 2 + dead0) * 4 + k0) * 4 + k1) * 2);
        r0[l] = cfg->reward_table[idx];
        r1[l] = cfg->reward_table[idx + 1];
        e->prev_selected[0][l] = (int32_t)(int8_t)a[0][l];
        e->prev_selected[1][l] = (int32_t)(int8_t)a[1][l];
        e->last_guard[0][l] = g->guard[0][l];
        e->last_guard[1][l] = g->guard[1][l];
        te[l] = dead0 | dead1;
        tr[l] = t >= cfg->max_t;
    }
    if (autoreset) {
        LANES lane_env_fresh(e, l, te[l] | tr[l], e->episode_id[l] + 1);
    }
    block_encode(e, &s);
    block_write_obs(&s, obs, count);
    for (int l = 0; l < count; l++) {
        rew[2 * l] = r0[l];
        rew[2 * l + 1] = r1[l];
        term[l] = (uint8_t)te[l];
        trunc[l] = (uint8_t)tr[l];
    }
}

// ── Conversion to and from footsies.h's EnvState ─────────────────────────

static inline void lane_from_env_state(EnvBlock* e, int l, const EnvState* st) {
    GameBlock* g = &e->g;
    for (int p = 0; p < 2; p++) {
        const Fighter* f = &st->game.f[p];
        g->pos_x[p][l] = f->pos_x, g->vel_x[p][l] = f->vel_x;
        g->vital[p][l] = f->vital, g->guard[p][l] = f->guard, g->action[p][l] = f->action;
        g->action_frame[p][l] = f->action_frame, g->hit_count[p][l] = f->hit_count;
        g->frame_adv[p][l] = f->frame_adv, g->hit_stun[p][l] = f->hit_stun;
        g->sprite_shake[p][l] = f->sprite_shake, g->buffer_action[p][l] = f->buffer_action;
        g->reserve_damage_action[p][l] = f->reserve_damage_action;
        g->face_right[p][l] = f->face_right, g->has_won[p][l] = f->has_won;
        g->input_backward[p][l] = f->input_backward;
        g->reserve_prox_guard[p][l] = f->reserve_prox_guard;
        uint64_t released = ~f->in_attack;
        g->attack_run[p][l] = released ? ctz64(released) : 64;
        B_SPLIT(g, in_left, p, l, f->in_left), B_SPLIT(g, in_right, p, l, f->in_right);
        B_SPLIT(g, in_attack, p, l, f->in_attack);
        e->budget[p][l] = st->budget_deductions[p];
        e->last_guard[p][l] = st->last_guard[p];
        for (int k = 0; k < MAX_DELAY_STEPS; k++) e->queue[p][k][l] = st->action_queue[p][k];
        e->prev_selected[p][l] = st->prev_selected[p];
        e->holding[p][l] = st->holding_special[p];
    }
    g->round_state[l] = st->game.round_state, g->frame_count[l] = st->game.frame_count;
    g->done[l] = st->game.done, g->reward[l] = st->game.reward;
    e->t[l] = st->t, e->episode_id[l] = st->episode_id, e->queue_head[l] = st->queue_head;
}

static inline void lane_to_env_state(const EnvBlock* e, int l, EnvState* st) {
    const GameBlock* g = &e->g;
    memset(st, 0, sizeof *st);
    for (int p = 0; p < 2; p++) {
        Fighter* f = &st->game.f[p];
        f->pos_x = g->pos_x[p][l], f->vel_x = g->vel_x[p][l];
        f->vital = g->vital[p][l], f->guard = g->guard[p][l], f->action = g->action[p][l];
        f->action_frame = g->action_frame[p][l], f->hit_count = g->hit_count[p][l];
        f->frame_adv = g->frame_adv[p][l], f->hit_stun = g->hit_stun[p][l];
        f->sprite_shake = g->sprite_shake[p][l], f->buffer_action = g->buffer_action[p][l];
        f->reserve_damage_action = g->reserve_damage_action[p][l];
        f->face_right = (uint8_t)g->face_right[p][l], f->has_won = (uint8_t)g->has_won[p][l];
        f->input_backward = (uint8_t)g->input_backward[p][l];
        f->reserve_prox_guard = (uint8_t)g->reserve_prox_guard[p][l];
        f->in_left = B_JOIN(g, in_left, p, l), f->in_right = B_JOIN(g, in_right, p, l);
        f->in_attack = B_JOIN(g, in_attack, p, l);
        st->budget_deductions[p] = e->budget[p][l];
        st->last_guard[p] = e->last_guard[p][l];
        for (int k = 0; k < MAX_DELAY_STEPS; k++)
            st->action_queue[p][k] = (int8_t)e->queue[p][k][l];
        st->prev_selected[p] = (int8_t)e->prev_selected[p][l];
        st->holding_special[p] = (uint8_t)e->holding[p][l];
    }
    st->game.round_state = g->round_state[l], st->game.frame_count = g->frame_count[l];
    st->game.done = (uint8_t)g->done[l], st->game.reward = (int8_t)g->reward[l];
    st->t = e->t[l], st->episode_id = e->episode_id[l], st->queue_head = e->queue_head[l];
}

// ── A preallocated batch of envs ─────────────────────────────────────────

typedef struct {
    int n, nblocks;
    EnvConfig cfg;
    EnvBlock* blocks;  // 64-byte aligned, allocated once
} BatchEnv;

static BatchEnv* batch_env_new(const EnvConfig* cfg, int n) {
    BatchEnv* b = (BatchEnv*)calloc(1, sizeof(BatchEnv));
    if (!b) return NULL;
    b->n = n;
    b->nblocks = (n + BATCH_W - 1) / BATCH_W;
    b->cfg = *cfg;
    size_t bytes = (size_t)(b->nblocks > 0 ? b->nblocks : 1) * sizeof(EnvBlock);
    bytes = (bytes + 63) / 64 * 64;
    b->blocks = (EnvBlock*)aligned_alloc(64, bytes);
    if (!b->blocks) {
        free(b);
        return NULL;
    }
    memset(b->blocks, 0, bytes);
    return b;
}

static void batch_env_free(BatchEnv* b) {
    if (!b) return;
    free(b->blocks);
    free(b);
}

#if FOOTSIES_AVX512
#include "footsies_avx512.h"
#endif

static void batch_init(const FrameTables* T) {
    batch_tables_init(T);
#if FOOTSIES_AVX512
    vec_tables_init();
#endif
}

#endif  // FOOTSIES_BATCH_H
