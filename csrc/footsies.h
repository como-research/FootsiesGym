// FOOTSIES game and environment as plain C functions for CPU and CUDA.
//
// Byte-identical port of footsiesgym.jax (itself validated against the Unity
// game): the operation order of every float computation comes from the JAX
// port, the control flow (early returns, in-place mutation) from the C#.
//
// Byte identity depends on the compiler flags, which are part of the spec:
//   gcc/clang: -O3 -ffp-contract=off       never -ffast-math or -Ofast
//   nvcc:      -fmad=false -prec-div=true -ftz=false   never --use_fast_math
// Without them the compiler may fuse a*b+c into an FMA, which rounds once
// instead of twice.
//
// Layout:
//   Fighter, Game      game state (FootsiesGame / Fighter in footsiesgym.jax)
//   game_*             one game: new, reset, step, step_n, raw state
//   EnvState, env_*    FootsiesJaxEnv: action delay, special charge, rewards,
//                      truncation, autoreset, the 88-float observations
//
// Every function takes the frame data as `const FrameTables* T`: a static
// const on the host, a shared-memory copy on the GPU.

#ifndef FOOTSIES_H
#define FOOTSIES_H

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "frame_data.gen.h"

#if defined(__CUDACC__)
#define HD static __host__ __device__ __forceinline__
#define UNROLL _Pragma("unroll")
#else
#define HD static inline
#define UNROLL
#endif

// Floats must be evaluated in their own precision (SSE2 / NEON / CUDA, not x87).
// 16 (C23) only widens _Float16, which is unused; float and double stay as is.
#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0 && FLT_EVAL_METHOD != 16 && \
    !defined(__CUDA_ARCH__)
#error "footsies.h needs FLT_EVAL_METHOD == 0 (e.g. x86-64 SSE2 or aarch64)"
#endif
#if defined(__FAST_MATH__)
#error "footsies.h must not be compiled with -ffast-math"
#endif

#ifndef MAX_DELAY_STEPS
#define MAX_DELAY_STEPS 8  // action_delay / frame_skip, at most
#endif

#define OBS_DIM 88
#define PUBLIC_DIM 37
#define PRIVATE_DIM 13

// BattleSimulation.RoundStateType
#define ROUND_FIGHT 2
#define ROUND_KO 3

// Fighter.NotifyDamaged results (DamageResult)
#define DAMAGE_RESULT 1
#define GUARD_RESULT 2
#define GUARD_BREAK_RESULT 3

// Reward table: [p2 guard broken][p1 guard broken][p2 dead][p1 dead]
//               [p1 deductions][p2 deductions][player]
#define MAX_BUDGET_DEDUCTIONS 3
#define REWARD_TABLE_SIZE (2 * 2 * 2 * 2 * 4 * 4 * 2)

// ── State ────────────────────────────────────────────────────────────────

typedef struct {
    float pos_x, vel_x;  // position.y is always 0
    int32_t vital, guard, action, action_frame, hit_count, frame_adv, hit_stun,
        sprite_shake, buffer_action, reserve_damage_action;  // actions: index or -1
    // Input history, one bit per frame: bit k = input held k frames ago.
    // JAX keeps INPUT_HISTORY (60) frames; the bits above 59 are never read.
    uint64_t in_left, in_right, in_attack;
    uint8_t face_right, has_won, input_backward, reserve_prox_guard;
    uint8_t pad_[4];  // explicit, so states compare byte for byte (always 0)
} Fighter;

typedef struct {
    Fighter f[2];  // P1, P2
    int32_t round_state, frame_count;
    uint8_t done;   // the episode ended on the last step
    int8_t reward;  // +1 P1 won, -1 P2 won, 0 otherwise
    uint8_t pad_[6];
} Game;

typedef struct {
    Game game;
    int32_t t, episode_id, queue_head, budget_deductions[2], last_guard[2];
    int8_t action_queue[2][MAX_DELAY_STEPS], prev_selected[2];
    uint8_t holding_special[2];
} EnvState;

typedef struct {
    int32_t frame_skip, action_delay_frames, max_t;
    uint8_t guard_break_reward_nonzero, use_reward_budget;
    // RewardFunction._build_table() as float32 (the float64 values rounded once)
    float reward_table[REWARD_TABLE_SIZE];
} EnvConfig;

// No implicit padding: CPU and CUDA states must compare equal with memcmp.
#ifdef __cplusplus
#define FOOTSIES_STATIC_ASSERT static_assert
#else
#define FOOTSIES_STATIC_ASSERT _Static_assert
#endif
FOOTSIES_STATIC_ASSERT(sizeof(Fighter) == 2 * 4 + 10 * 4 + 3 * 8 + 4 + 4, "Fighter padding");
FOOTSIES_STATIC_ASSERT(sizeof(Game) == 2 * sizeof(Fighter) + 2 * 4 + 2 + 6, "Game padding");
FOOTSIES_STATIC_ASSERT(sizeof(EnvState) == sizeof(Game) + 7 * 4 + 2 * MAX_DELAY_STEPS + 4,
                       "EnvState padding: MAX_DELAY_STEPS must be a multiple of 4");

// BatchRawState fields of one game (the proto's p1_*/p2_* pairs as [2])
typedef struct {
    float position_x[2], velocity_x[2], special_attack_progress[2];
    int32_t vital_health[2], guard_health[2], current_action_id[2],
        current_action_frame[2], current_action_frame_count[2],
        current_action_hit_count[2], current_hit_stun_frame[2],
        sprite_shake_position[2], max_sprite_shake_frame[2],
        current_frame_advantage[2];
    uint8_t is_dead[2], is_action_end[2], is_always_cancelable[2], is_in_hit_stun[2],
        is_face_right[2], would_next_forward_input_dash[2],
        would_next_backward_input_dash[2];
    int32_t round_states, frame_counts;
    uint8_t dones;
    int8_t rewards;
} RawState;

typedef struct { float x, y, w, h; } Rect;

// A fighter's boxes this frame, in fixed slots (asset order), like the JAX port.
typedef struct {
    Rect hit[MAX_HITBOXES];
    uint8_t hit_valid[MAX_HITBOXES], hit_attack[MAX_HITBOXES], hit_prox[MAX_HITBOXES];
    Rect hurt[MAX_HURTBOXES];
    uint8_t hurt_valid[MAX_HURTBOXES];
    Rect push;
} Boxes;

// ── Small helpers ────────────────────────────────────────────────────────

HD int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }

HD int ctz64(uint64_t x) {  // x != 0
#ifdef __CUDA_ARCH__
    return __ffsll((long long)x) - 1;
#else
    return __builtin_ctzll(x);
#endif
}

HD int lut4(uint32_t lut, int i) { return (int)((lut >> (4 * i)) & 0xFu); }

// position.x += speed * sign * FIXED_DELTA_TIME, as Mono evaluates it: in
// double, rounded to float on the store. speed * sign is exact (sign is +-1),
// and its product with dt is exact in double (24 + 24 significant bits).
HD float mono_move(float pos, float speed, float sign) {
    return (float)((double)pos + (double)(speed * sign) * (double)FIXED_DELTA_TIME);
}

// ── Fighter: current action ─────────────────────────────────────────────

HD int frame_count_of(const Fighter* f, const FrameTables* T) {
    return T->actions[f->action].frame_count;
}

HD int is_action_end(const Fighter* f, const FrameTables* T) {
    return f->action_frame >= frame_count_of(f, T);
}

HD int is_always_cancelable(const Fighter* f, const FrameTables* T) {
    return T->actions[f->action].always_cancelable;
}

// Frames left in the current action, as GetFrameAdvantage counts them.
HD int frames_left(const Fighter* f, const FrameTables* T) {
    return is_always_cancelable(f, T) ? 0 : frame_count_of(f, T) - f->action_frame;
}

// The action frame as looked up in the frame data (clamped, as in JAX).
HD int table_frame(const Fighter* f) { return clampi(f->action_frame, 0, NUM_FRAMES - 1); }

HD int in_range(int lo, int hi, int frame) { return lo <= frame && frame <= hi; }

HD void set_current_action(Fighter* f, int action) {
    f->action = action;
    f->action_frame = 0;
    f->hit_count = 0;
    f->buffer_action = -1;
    f->reserve_damage_action = -1;
    f->sprite_shake = 0;
}

// Whether requesting `target` during (action, frame) buffers it.
HD int cancel_buffer(const FrameTables* T, int action, int frame, int target) {
    const ActionData* ad = &T->actions[action];
    for (int k = 0; k < ad->n_cancel; k++) {
        const CancelEntry* e = &T->cancel[ad->cancel0 + k];
        if (in_range(e->lo, e->hi, frame) && ((e->targets >> target) & 1u)) return 1;
    }
    return 0;
}

// RequestAction (callers never use its return value).
HD void request_action(Fighter* f, const FrameTables* T, int action) {
    int end = is_action_end(f, T);
    int same = f->action == action;
    int cancelable = is_always_cancelable(f, T);
    if (end || (!same && cancelable)) {
        set_current_action(f, action);
    } else if (!same && !cancelable &&
               cancel_buffer(T, f->action, table_frame(f), action)) {
        f->buffer_action = action;
    }
}

HD void increment_action_frame(Fighter* f, const FrameTables* T) {
    if (f->sprite_shake != 0) {
        int shake = -f->sprite_shake;
        f->sprite_shake = shake + (shake > 0 ? -1 : 1);
    }
    if (f->hit_stun > 0) {
        f->hit_stun -= 1;
        return;
    }
    const ActionData* ad = &T->actions[f->action];
    int frame = f->action_frame + 1;
    if (frame >= ad->frame_count && ad->is_loop) frame = ad->loop_from;
    f->action_frame = frame;
}

// ── Fighter: input ───────────────────────────────────────────────────────

HD void update_input(Fighter* f, int bits) {
    f->in_left = (f->in_left << 1) | (uint64_t)((bits & INPUT_LEFT) != 0);
    f->in_right = (f->in_right << 1) | (uint64_t)((bits & INPUT_RIGHT) != 0);
    f->in_attack = (f->in_attack << 1) | (uint64_t)((bits & INPUT_ATTACK) != 0);
}

HD uint64_t forward_history(const Fighter* f) { return f->face_right ? f->in_right : f->in_left; }
HD uint64_t backward_history(const Fighter* f) { return f->face_right ? f->in_left : f->in_right; }

HD int pressed_now(uint64_t h) { return (int)((h & 1u) && !(h & 2u)); }   // inputDown[0]
HD int released_now(uint64_t h) { return (int)(!(h & 1u) && (h & 2u)); }  // inputUp[0]

// Shared loop of Check*DashInput (first=1, last=n) and WouldNext*InputDash
// (first=0, last=n-1): the first input pointing `away` fails, the first
// pointing `toward` succeeds iff one of the next dashAllowFrame-1 is neutral.
HD int dash_check(uint64_t toward, uint64_t away, int first, int last) {
    uint64_t neutral = ~(toward | away);
    for (int i = first; i < last; i++) {
        if ((away >> i) & 1u) return 0;
        if ((toward >> i) & 1u) {
            uint64_t window = ((1ULL << (DASH_ALLOW_FRAME - 1)) - 1) << (i + 1);
            return (neutral & window) != 0;
        }
    }
    return 0;
}

HD int would_next_forward_dash(const Fighter* f) {
    return dash_check(forward_history(f), backward_history(f), 0, DASH_ALLOW_FRAME - 1);
}

HD int would_next_backward_dash(const Fighter* f) {
    return dash_check(backward_history(f), forward_history(f), 0, DASH_ALLOW_FRAME - 1);
}

// Attack held on frames 1 .. hold-1 (JAX: all(attack_held[1:hold]))
#define HELD_SINCE_1 ((((uint64_t)1 << (SPECIAL_ATTACK_HOLD_FRAME - 1)) - 1) << 1)

// GetSpecialAttackProgress: consecutive held frames from now, over hold.
HD float special_progress(const Fighter* f) {
    uint64_t released = ~f->in_attack;
    int n = released ? ctz64(released) : 64;
    if (n > SPECIAL_ATTACK_HOLD_FRAME) n = SPECIAL_ATTACK_HOLD_FRAME;
    return (float)n / (float)SPECIAL_ATTACK_HOLD_FRAME;
}

HD void update_action_request(Fighter* f, const FrameTables* T) {
    // Early returns, in priority order
    if (f->has_won) {
        request_action(f, T, ACT_WIN);
        return;
    }
    if (f->reserve_damage_action != -1 && f->hit_stun <= 0) {
        set_current_action(f, f->reserve_damage_action);
        return;
    }
    if (f->buffer_action != -1 && (CAN_CANCEL_ON_WHIFF || f->hit_count > 0) &&
        f->hit_stun <= 0) {
        set_current_action(f, f->buffer_action);
        return;
    }

    // Everything below is read from the fighter before any request.
    uint64_t fwd = forward_history(f), bwd = backward_history(f);
    int is_forward = (int)(fwd & 1u), is_backward = (int)(bwd & 1u);
    int directional = is_forward | is_backward;
    int special = released_now(f->in_attack) &&
                  (f->in_attack & HELD_SINCE_1) == HELD_SINCE_1;
    int attack = pressed_now(f->in_attack);
    int in_normal = (f->action == ACT_N_ATTACK || f->action == ACT_B_ATTACK) &&
                    !is_action_end(f, T);
    int normal = in_normal ? ACT_N_SPECIAL : (directional ? ACT_B_ATTACK : ACT_N_ATTACK);
    int forward_dash = pressed_now(fwd) && dash_check(fwd, bwd, 1, DASH_ALLOW_FRAME);
    int backward_dash = pressed_now(bwd) && dash_check(bwd, fwd, 1, DASH_ALLOW_FRAME);

    // Attacks
    if (special) request_action(f, T, directional ? ACT_B_SPECIAL : ACT_N_SPECIAL);
    else if (attack) request_action(f, T, normal);

    // Dashes
    if (forward_dash) request_action(f, T, ACT_DASH_FORWARD);
    else if (backward_dash) request_action(f, T, ACT_DASH_BACKWARD);

    // Walking and guarding
    f->input_backward = (uint8_t)is_backward;
    int move = ACT_STAND;
    if (is_forward && !is_backward) move = ACT_FORWARD;
    else if (is_backward && !is_forward)
        move = f->reserve_prox_guard ? ACT_GUARD_PROXIMITY : ACT_BACKWARD;
    request_action(f, T, move);
    f->reserve_prox_guard = 0;
}

// ── Fighter: movement and boxes ──────────────────────────────────────────

HD float facing_sign(const Fighter* f) { return f->face_right ? 1.0f : -1.0f; }

HD void update_movement(Fighter* f, const FrameTables* T) {
    if (f->hit_stun > 0) return;
    float sign = facing_sign(f);
    if (f->action == ACT_FORWARD) {
        f->pos_x = mono_move(f->pos_x, FORWARD_MOVE_SPEED, sign);
    } else if (f->action == ACT_BACKWARD) {
        f->pos_x = mono_move(f->pos_x, -BACKWARD_MOVE_SPEED, sign);
    } else {
        const ActionData* ad = &T->actions[f->action];
        int frame = table_frame(f);
        for (int k = 0; k < ad->n_move; k++) {  // the first active entry
            const MoveEntry* e = &T->move[ad->move0 + k];
            if (in_range(e->lo, e->hi, frame)) {
                f->vel_x = e->velocity;
                if (e->velocity != 0.0f) f->pos_x = mono_move(f->pos_x, e->velocity, sign);
                break;
            }
        }
    }
}

// TransformToFightRect (position.y is always 0)
HD Rect place(const Fighter* f, float x, float y, float w, float h) {
    Rect r;
    r.x = f->pos_x + x * facing_sign(f);
    r.y = y;
    r.w = w;
    r.h = h;
    return r;
}

// UpdateBoxes / ApplyCurrentActionData
HD void make_boxes(const Fighter* f, const FrameTables* T, Boxes* b) {
    const ActionData* ad = &T->actions[f->action];
    int frame = table_frame(f);
UNROLL
    for (int k = 0; k < MAX_HITBOXES; k++) {
        const HitboxEntry* e = &T->hit[ad->hit0 + (k < ad->n_hit ? k : 0)];
        int valid = k < ad->n_hit && in_range(e->lo, e->hi, frame);
        b->hit_valid[k] = (uint8_t)valid;
        b->hit[k] = valid ? place(f, e->x, e->y, e->w, e->h) : place(f, 0, 0, 0, 0);
        b->hit_attack[k] = valid ? e->attack : 0;
        b->hit_prox[k] = valid ? e->proximity : 0;
    }
UNROLL
    for (int k = 0; k < MAX_HURTBOXES; k++) {
        const RectEntry* e = &T->hurt[ad->hurt0 + (k < ad->n_hurt ? k : 0)];
        int valid = k < ad->n_hurt && in_range(e->lo, e->hi, frame);
        b->hurt_valid[k] = (uint8_t)valid;
        b->hurt[k] = valid ? place(f, e->x, e->y, e->w, e->h) : place(f, 0, 0, 0, 0);
    }
    b->push = place(f, 0, 0, 0, 0);  // no active pushbox: a zero rect
    for (int k = 0; k < ad->n_push; k++) {  // the first active entry
        const RectEntry* e = &T->push[ad->push0 + k];
        if (in_range(e->lo, e->hi, frame)) {
            b->push = place(f, e->x, e->y, e->w, e->h);
            break;
        }
    }
}

// ApplyPositionChange: the fighter and its boxes move by dx.
HD void apply_position_change(Fighter* f, Boxes* b, float dx) {
    f->pos_x = f->pos_x + dx;
UNROLL
    for (int k = 0; k < MAX_HITBOXES; k++) b->hit[k].x = b->hit[k].x + dx;
UNROLL
    for (int k = 0; k < MAX_HURTBOXES; k++) b->hurt[k].x = b->hurt[k].x + dx;
    b->push.x = b->push.x + dx;
}

// UpdatePushCharacterVsCharacter. UnityEngine.Rect semantics (x = left edge,
// strict overlap): the C# calls Rect.Overlaps / xMin / xMax on the raw rects.
HD void push_character_vs_character(Game* g, Boxes* b) {
    Rect r1 = b[0].push, r2 = b[1].push;
    int overlaps = (r2.x + r2.w > r1.x) && (r2.x < r1.x + r1.w) &&
                   (r2.y + r2.h > r1.y) && (r2.y < r1.y + r1.h);
    if (!overlaps) return;
    float p1_x = g->f[0].pos_x, p2_x = g->f[1].pos_x;
    int p1_left = p1_x < p2_x, p1_right = p1_x > p2_x;
    if (!(p1_left || p1_right)) return;
    float left_overlap = (r1.x + r1.w) - r2.x;
    float right_overlap = (r2.x + r2.w) - r1.x;
    float dx0 = p1_left ? -left_overlap * 0.5f : right_overlap * 0.5f;
    float dx1 = p1_left ? left_overlap * 0.5f : -right_overlap * 0.5f;
    apply_position_change(&g->f[0], &b[0], dx0);
    apply_position_change(&g->f[1], &b[1], dx1);
}

// PushFighterVsBackground (BoxBase xMin/xMax, x = centre)
HD void push_vs_background(Fighter* f, Boxes* b) {
    float x = b->push.x, w = b->push.w;
    float x_min = x - w * 0.5f, x_max = x + w * 0.5f;
    if (x_min < STAGE_MIN_X) apply_position_change(f, b, STAGE_MIN_X - x_min);
    else if (x_max > STAGE_MAX_X) apply_position_change(f, b, STAGE_MAX_X - x_max);
}

// BoxBase.Overlaps (x = centre, inclusive)
HD int boxes_overlap(Rect a, Rect b) {
    float a_x_min = a.x - a.w * 0.5f, a_x_max = a.x + a.w * 0.5f;
    float b_x_min = b.x - b.w * 0.5f, b_x_max = b.x + b.w * 0.5f;
    float a_y_min = a.y, a_y_max = a.y + a.h;
    float b_y_min = b.y, b_y_max = b.y + b.h;
    return (b_x_max >= a_x_min) && (b_x_min <= a_x_max) && (b_y_max >= a_y_min) &&
           (b_y_min <= a_y_max);
}

// ── Fighter: attacking and being hit ─────────────────────────────────────

// NotifyDamaged: returns the DamageResult.
HD int notify_damaged(Fighter* f, const FrameTables* T, int attack) {
    const AttackData* ad = &T->attacks[attack];
    int guard = f->guard;
    if (ad->guard_damage > 0) guard -= ad->guard_damage;
    int guard_break = ad->guard_damage > 0 && guard < 0;
    f->guard = guard_break ? 0 : guard;

    int guarding = f->action == ACT_BACKWARD ||
                   T->actions[f->action].type == ACTION_TYPE_GUARD;
    if (guarding) {
        set_current_action(f, ad->guard_action);
        if (guard_break) f->reserve_damage_action = ACT_GUARD_BREAK;
        return guard_break ? GUARD_BREAK_RESULT : GUARD_RESULT;
    }
    if (ad->vital_damage > 0) {
        int vital = f->vital - ad->vital_damage;
        f->vital = vital <= 0 ? 0 : vital;
    }
    set_current_action(f, ad->damage_action);
    return DAMAGE_RESULT;
}

// GetHitStunFrame: stun frames for both fighters after a hit.
HD int hit_stun_frame(const FrameTables* T, int result, int attack) {
    const AttackData* ad = &T->attacks[attack];
    if (result == GUARD_RESULT) return ad->guard_stun;
    if (result == GUARD_BREAK_RESULT) return ad->guard_break_stun;
    return ad->hit_stun;
}

// CheckAttack(attacker, damaged), with this frame's boxes.
HD void check_attack(Fighter* att, Fighter* dmg, const Boxes* ab, const Boxes* db,
                     const FrameTables* T) {
    int is_hit = 0, attack = 0, proximity = 0;
UNROLL
    for (int k = 0; k < MAX_HITBOXES; k++) {
        if (!ab->hit_valid[k]) continue;
        int atk = ab->hit_attack[k];
        if (!(att->hit_count < T->attacks[atk].number_of_hit)) continue;
        int touching = 0;
UNROLL
        for (int j = 0; j < MAX_HURTBOXES; j++)
            touching |= db->hurt_valid[j] && boxes_overlap(ab->hit[k], db->hurt[j]);
        if (!touching) continue;
        if (ab->hit_prox[k]) {
            proximity = 1;
        } else if (!is_hit) {  // the first damaging box in asset order
            is_hit = 1;
            attack = atk;
        }
    }
    if (is_hit) {
        int result = notify_damaged(dmg, T, attack);
        int stun = hit_stun_frame(T, result, attack);
        dmg->hit_stun = stun;
        int shake = stun / 3;  // SetSpriteShakeFrame
        if (shake > MAX_SPRITE_SHAKE_FRAME) shake = MAX_SPRITE_SHAKE_FRAME;
        dmg->sprite_shake = shake * (dmg->face_right ? -1 : 1);
        att->hit_count += 1;  // NotifyAttackHit
        att->hit_stun = stun;
    } else if (proximity) {  // NotifyInProximityGuardRange
        dmg->reserve_prox_guard = (uint8_t)(dmg->reserve_prox_guard | dmg->input_backward);
    }
}

// ── Game ─────────────────────────────────────────────────────────────────

// Field values of a freshly constructed C# Fighter.
HD void fighter_new(Fighter* f) {
    f->pos_x = 0.0f;
    f->vel_x = 0.0f;
    f->vital = f->guard = f->action = f->action_frame = f->hit_count = 0;
    f->frame_adv = f->hit_stun = f->sprite_shake = 0;
    f->buffer_action = f->reserve_damage_action = -1;
    f->in_left = f->in_right = f->in_attack = 0;
    f->face_right = f->has_won = f->input_backward = f->reserve_prox_guard = 0;
    UNROLL
    for (int k = 0; k < 4; k++) f->pad_[k] = 0;
}

// SetupBattleStart. Leaves hit stun, frame advantage, input_backward and
// reserve_prox_guard untouched: they carry over from the previous round.
HD void setup_battle_start(Fighter* f, float start_x, int face_right) {
    f->pos_x = start_x;
    f->face_right = (uint8_t)face_right;
    f->vital = 1;
    f->guard = START_GUARD_HEALTH;
    f->has_won = 0;
    f->vel_x = 0.0f;
    f->in_left = f->in_right = f->in_attack = 0;
    set_current_action(f, ACT_STAND);
}

// Reset(): P1 at -2 facing right, P2 at 2 facing left.
HD void game_reset(Game* g) {
    setup_battle_start(&g->f[0], -2.0f, 1);
    setup_battle_start(&g->f[1], 2.0f, 0);
    g->round_state = ROUND_FIGHT;
    g->frame_count = -1;
    g->done = 0;
    g->reward = 0;
}

// A freshly constructed BattleSimulation (the constructor calls Reset).
HD void game_new(Game* g) {
    fighter_new(&g->f[0]);
    fighter_new(&g->f[1]);
    g->round_state = 0;
    g->frame_count = 0;
    UNROLL
    for (int k = 0; k < 6; k++) g->pad_[k] = 0;
    game_reset(g);
}

// Step: advance one frame.
HD void game_step(Game* g, const FrameTables* T, int p1_bits, int p2_bits) {
    if (g->round_state != ROUND_FIGHT) game_reset(g);
    update_input(&g->f[0], p1_bits);
    update_input(&g->f[1], p2_bits);

    int l0 = frames_left(&g->f[0], T), l1 = frames_left(&g->f[1], T);
    g->f[0].frame_adv = l1 - l0;
    g->f[1].frame_adv = l0 - l1;

    increment_action_frame(&g->f[0], T);
    increment_action_frame(&g->f[1], T);
    update_action_request(&g->f[0], T);
    update_action_request(&g->f[1], T);
    update_movement(&g->f[0], T);
    update_movement(&g->f[1], T);

    Boxes b[2];
    make_boxes(&g->f[0], T, &b[0]);
    make_boxes(&g->f[1], T, &b[1]);
    push_character_vs_character(g, b);
    push_vs_background(&g->f[0], &b[0]);
    push_vs_background(&g->f[1], &b[1]);

    // Same boxes for both checks, but P2's check sees the effects of P1's.
    check_attack(&g->f[0], &g->f[1], &b[0], &b[1], T);
    check_attack(&g->f[1], &g->f[0], &b[1], &b[0], T);

    int d0 = g->f[0].vital <= 0, d1 = g->f[1].vital <= 0;
    g->done = (uint8_t)(d0 | d1);
    g->reward = (int8_t)((d1 && !d0) - (d0 && !d1));
    g->round_state = g->done ? ROUND_KO : ROUND_FIGHT;
    g->frame_count += 1;
}

// StepN: repeat the inputs for n frames, stopping early at a KO.
HD void game_step_n(Game* g, const FrameTables* T, int p1_bits, int p2_bits, int n) {
    for (int k = 0; k < n; k++) {
        game_step(g, T, p1_bits, p2_bits);
        if (g->done) break;
    }
}

// BatchRawState fields. action_ids: ACTION_IDS (host only, for validation).
HD void game_raw(const Game* g, const FrameTables* T, const int32_t* action_ids,
                 RawState* r) {
    UNROLL
    for (int p = 0; p < 2; p++) {
        const Fighter* f = &g->f[p];
        int count = frame_count_of(f, T);
        r->position_x[p] = f->pos_x;
        r->velocity_x[p] = f->vel_x;
        r->special_attack_progress[p] = special_progress(f);
        r->vital_health[p] = f->vital;
        r->guard_health[p] = f->guard;
        r->current_action_id[p] = action_ids[f->action];
        r->current_action_frame[p] = f->action_frame;
        r->current_action_frame_count[p] = count;
        r->current_action_hit_count[p] = f->hit_count;
        r->current_hit_stun_frame[p] = f->hit_stun;
        r->sprite_shake_position[p] = f->sprite_shake;
        r->max_sprite_shake_frame[p] = MAX_SPRITE_SHAKE_FRAME;
        r->current_frame_advantage[p] = f->frame_adv;
        r->is_dead[p] = f->vital <= 0;
        r->is_action_end[p] = f->action_frame >= count;
        r->is_always_cancelable[p] = (uint8_t)is_always_cancelable(f, T);
        r->is_in_hit_stun[p] = f->hit_stun > 0;
        r->is_face_right[p] = f->face_right;
        r->would_next_forward_input_dash[p] = (uint8_t)would_next_forward_dash(f);
        r->would_next_backward_input_dash[p] = (uint8_t)would_next_backward_dash(f);
    }
    r->round_states = g->round_state;
    r->frame_counts = g->frame_count;
    r->dones = g->done;
    r->rewards = g->reward;
}

// ── Observations (ObservationEncoder) ────────────────────────────────────

// k / d for an integer feature clipped to [-1024, 1024] (IEEE division).
HD float int_feature(int k, float d) { return (float)clampi(k, -1024, 1024) / d; }

// The 37 public features of fighter p.
HD void encode_public(const Fighter* f, const FrameTables* T, float* o) {
    int frame = f->action_frame, count = frame_count_of(f, T);
    o[0] = f->pos_x * 0.25f;
    // JAX looks velocity up in a table of IEEE quotients, with 0 -> +0.
    o[1] = f->vel_x == 0.0f ? 0.0f : f->vel_x / 5.0f;
    o[2] = f->vital <= 0 ? 1.0f : 0.0f;
    o[3] = (float)f->vital;
    UNROLL
    for (int k = 0; k < 4; k++) o[4 + k] = f->guard == k ? 1.0f : 0.0f;
    UNROLL
    for (int k = 0; k < NUM_ACTIONS; k++) o[8 + k] = f->action == k ? 1.0f : 0.0f;
    o[25] = int_feature(frame, 25.0f);
    o[26] = int_feature(count, 25.0f);
    o[27] = int_feature(count - frame, 25.0f);
    o[28] = frame >= count ? 1.0f : 0.0f;
    o[29] = is_always_cancelable(f, T) ? 1.0f : 0.0f;
    o[30] = (float)f->hit_count;
    o[31] = int_feature(f->hit_stun, 10.0f);
    o[32] = f->hit_stun > 0 ? 1.0f : 0.0f;
    o[33] = (float)f->sprite_shake;
    o[34] = int_feature(MAX_SPRITE_SHAKE_FRAME, 10.0f);
    o[35] = f->face_right ? 1.0f : 0.0f;
    o[36] = int_feature(f->frame_adv, 10.0f);
}

// The 13 privileged features of fighter p.
HD void encode_private(const Fighter* f, int prev_action, int holding, float* o) {
    o[0] = would_next_forward_dash(f) ? 1.0f : 0.0f;
    o[1] = would_next_backward_dash(f) ? 1.0f : 0.0f;
    float progress = special_progress(f);
    o[2] = progress < 1.0f ? progress : 1.0f;
    UNROLL
    for (int k = 0; k < NUM_ENV_ACTIONS; k++) o[3 + k] = prev_action == k ? 1.0f : 0.0f;
    o[12] = holding ? 1.0f : 0.0f;
}

// Both observations: obs[0:88] for P1, obs[88:176] for P2, each
// [distance | self public | self private | opponent public].
HD void env_encode(const EnvState* st, const FrameTables* T, float* obs) {
    const Game* g = &st->game;
    float distance = fabsf(g->f[0].pos_x - g->f[1].pos_x) * 0.125f;
    UNROLL
    for (int p = 0; p < 2; p++) {
        float* o = obs + p * OBS_DIM;
        o[0] = distance;
        encode_public(&g->f[p], T, o + 1);
        encode_private(&g->f[p], st->prev_selected[p], st->holding_special[p],
                       o + 1 + PUBLIC_DIM);
        encode_public(&g->f[1 - p], T, o + 1 + PUBLIC_DIM + PRIVATE_DIM);
    }
}

// ── Environment (FootsiesJaxEnv) ─────────────────────────────────────────

HD int delay_steps(const EnvConfig* cfg) { return cfg->action_delay_frames / cfg->frame_skip; }

// A fresh episode (FootsiesJaxEnv._fresh_state).
HD void env_fresh(EnvState* st) {
    game_new(&st->game);
    st->t = 0;
    st->episode_id = 0;
    st->queue_head = 0;
    UNROLL
    for (int p = 0; p < 2; p++) {
        st->budget_deductions[p] = 0;
        st->last_guard[p] = st->game.f[p].guard;
        UNROLL
        for (int k = 0; k < MAX_DELAY_STEPS; k++) st->action_queue[p][k] = 0;
        st->prev_selected[p] = 0;
        st->holding_special[p] = 0;
    }
}

// Special charge: toggle on the charge actions, then map to base actions 0-5.
HD int resolve_special_charge(EnvState* st, int p, int action) {
    if (action == ENV_SPECIAL_CHARGE || action == ENV_FORWARD_SPECIAL_CHARGE ||
        action == ENV_BACK_SPECIAL_CHARGE)
        st->holding_special[p] ^= 1;
    int base = action == ENV_SPECIAL_CHARGE           ? ENV_NONE
               : action == ENV_FORWARD_SPECIAL_CHARGE ? ENV_FORWARD
               : action == ENV_BACK_SPECIAL_CHARGE    ? ENV_BACK
                                                      : action;
    return st->holding_special[p] ? lut4(CHARGE_ACTION_LUT, clampi(base, 0, 5)) : base;
}

// FootsiesJaxEnv.step (autoreset=1) or step_env (autoreset=0). Writes both
// players' observations (obs[0:176]), rewards (rew[0:2]) and the done flags.
HD void env_step(EnvState* st, const FrameTables* T, const EnvConfig* cfg, int a1, int a2,
                 float* obs, float* rew, uint8_t* term, uint8_t* trunc, int autoreset) {
    int execute[2] = {a1, a2};

    // Action delay: a FIFO of action_delay / frame_skip steps
    if (cfg->action_delay_frames != 0) {
        // Scan with constant indices so that, on the GPU, the queue stays in
        // registers (a runtime index would put the whole state in local memory).
        int head = st->queue_head;
        UNROLL
        for (int k = 0; k < MAX_DELAY_STEPS; k++) {
            if (k == head) {
                execute[0] = st->action_queue[0][k];
                execute[1] = st->action_queue[1][k];
                st->action_queue[0][k] = (int8_t)a1;
                st->action_queue[1][k] = (int8_t)a2;
            }
        }
        st->queue_head = (head + 1) % delay_steps(cfg);
    }

    int base1 = resolve_special_charge(st, 0, execute[0]);
    int base2 = resolve_special_charge(st, 1, execute[1]);
    game_step_n(&st->game, T, lut4(P1_ENV_TO_BITS, clampi(base1, 0, 5)),
                lut4(P2_ENV_TO_BITS, clampi(base2, 0, 5)), cfg->frame_skip);

    // Rewards (RewardFunction)
    st->t += 1;
    const Fighter* f = st->game.f;
    int dead0 = f[0].vital <= 0, dead1 = f[1].vital <= 0;
    int broken0 = 0, broken1 = 0;
    if (cfg->guard_break_reward_nonzero) {
        broken0 = f[0].guard < st->last_guard[0];
        broken1 = f[1].guard < st->last_guard[1];
    }
    if (cfg->use_reward_budget) {  // P1's budget pays for breaking P2's guard
        st->budget_deductions[0] += broken1;
        st->budget_deductions[1] += broken0;
    }
    int k0 = clampi(st->budget_deductions[0], 0, MAX_BUDGET_DEDUCTIONS);
    int k1 = clampi(st->budget_deductions[1], 0, MAX_BUDGET_DEDUCTIONS);
    int idx = ((((((broken1 * 2 + broken0) * 2 + dead1) * 2 + dead0) * 4 + k0) * 4 + k1) * 2);
    rew[0] = cfg->reward_table[idx];
    rew[1] = cfg->reward_table[idx + 1];

    st->prev_selected[0] = (int8_t)a1;
    st->prev_selected[1] = (int8_t)a2;
    st->last_guard[0] = f[0].guard;
    st->last_guard[1] = f[1].guard;

    int terminated = dead0 | dead1, truncated = st->t >= cfg->max_t;
    *term = (uint8_t)terminated;
    *trunc = (uint8_t)truncated;
    if (autoreset && (terminated || truncated)) {
        int episode_id = st->episode_id + 1;
        env_fresh(st);
        st->episode_id = episode_id;
    }
    env_encode(st, T, obs);
}

#endif  // FOOTSIES_H
