// AVX-512 frame step for footsies_batch.h blocks: StepN of BATCH_W = 16
// games at once, one game per 32-bit lane.
//
// The same branch-free formulation as the lane code in footsies_batch.h,
// written with vector types so that:
//   * table lookups are register permutes (vpermi2d / vpermi2ps over at most
//     32 entries, two for the 64-entry movement table) instead of gathers;
//   * requests, hits and dash checks that no lane needs this frame are skipped;
//   * the whole frameskip runs on registers (the block is loaded and stored once).
// The float operations are the same IEEE operations in the same order, so
// the result is byte-identical (checked against footsies.h in the tests).

#ifndef FOOTSIES_AVX512_H
#define FOOTSIES_AVX512_H

#include <immintrin.h>

#include "footsies_batch.h"

typedef int32_t vi __attribute__((vector_size(64)));
typedef uint32_t vu __attribute__((vector_size(64)));
typedef float vf __attribute__((vector_size(64)));

#define VAI static inline __attribute__((always_inline))

// ── Tables padded to 32 (or 64) entries, 64-byte aligned ─────────────────

#define T32(name) int32_t name[32] __attribute__((aligned(64)))
#define F32(name) float name[32] __attribute__((aligned(64)))

typedef struct {
    // per action
    T32(fc); T32(loop_from); T32(is_loop); T32(canc); T32(is_guard); T32(hit0); T32(nhit);
    T32(hurt0); T32(nhurt); T32(push0); T32(npush); T32(move0); T32(nmove); T32(cancel0);
    T32(ncancel);
    // entries
    T32(hit_lo); T32(hit_hi); T32(hit_atk); T32(hit_prox);
    F32(hit_x); F32(hit_y); F32(hit_w); F32(hit_h);
    T32(hurt_lo); T32(hurt_hi); F32(hurt_x); F32(hurt_y); F32(hurt_w); F32(hurt_h);
    T32(push_lo); T32(push_hi); F32(push_x); F32(push_y); F32(push_w); F32(push_h);
    int32_t move_lo[64] __attribute__((aligned(64)));
    int32_t move_hi[64] __attribute__((aligned(64)));
    float move_v[64] __attribute__((aligned(64)));
    T32(cancel_lo); T32(cancel_hi); T32(cancel_targets);
    // per attack
    T32(damage_action); T32(guard_action); T32(noh); T32(vd); T32(gd); T32(hs); T32(gs);
    T32(gbs);
} VecTables;

_Static_assert(NUM_ACTIONS <= 32 && N_HIT_ENTRIES + MAX_HITBOXES <= 32 &&
                   N_HURT_ENTRIES + MAX_HURTBOXES <= 32 &&
                   N_PUSH_ENTRIES + MAX_PUSHBOXES <= 32 && N_MOVE_ENTRIES + MAX_MOVES <= 64 &&
                   N_CANCEL_ENTRIES + MAX_CANCELS <= 32 && NUM_ATTACKS <= 32,
               "tables must fit the permute lookups");

static VecTables VT;

#define VT_COPY(dst, src, n) memcpy(VT.dst, BT.src, sizeof(BT.src[0]) * (n))

static void vec_tables_init(void) {  // after batch_tables_init
    memset(&VT, 0, sizeof VT);
    VT_COPY(fc, fc, NUM_ACTIONS), VT_COPY(loop_from, loop_from, NUM_ACTIONS);
    VT_COPY(is_loop, is_loop, NUM_ACTIONS), VT_COPY(canc, canc, NUM_ACTIONS);
    for (int a = 0; a < NUM_ACTIONS; a++) VT.is_guard[a] = BT.type[a] == ACTION_TYPE_GUARD;
    VT_COPY(hit0, hit0, NUM_ACTIONS), VT_COPY(nhit, nhit, NUM_ACTIONS);
    VT_COPY(hurt0, hurt0, NUM_ACTIONS), VT_COPY(nhurt, nhurt, NUM_ACTIONS);
    VT_COPY(push0, push0, NUM_ACTIONS), VT_COPY(npush, npush, NUM_ACTIONS);
    VT_COPY(move0, move0, NUM_ACTIONS), VT_COPY(nmove, nmove, NUM_ACTIONS);
    VT_COPY(cancel0, cancel0, NUM_ACTIONS), VT_COPY(ncancel, ncancel, NUM_ACTIONS);
    int nh = N_HIT_ENTRIES + MAX_HITBOXES, nu = N_HURT_ENTRIES + MAX_HURTBOXES;
    int np = N_PUSH_ENTRIES + MAX_PUSHBOXES, nm = N_MOVE_ENTRIES + MAX_MOVES;
    int nc = N_CANCEL_ENTRIES + MAX_CANCELS;
    VT_COPY(hit_lo, hit_lo, nh), VT_COPY(hit_hi, hit_hi, nh), VT_COPY(hit_atk, hit_atk, nh);
    VT_COPY(hit_prox, hit_prox, nh), VT_COPY(hit_x, hit_x, nh), VT_COPY(hit_y, hit_y, nh);
    VT_COPY(hit_w, hit_w, nh), VT_COPY(hit_h, hit_h, nh);
    VT_COPY(hurt_lo, hurt_lo, nu), VT_COPY(hurt_hi, hurt_hi, nu), VT_COPY(hurt_x, hurt_x, nu);
    VT_COPY(hurt_y, hurt_y, nu), VT_COPY(hurt_w, hurt_w, nu), VT_COPY(hurt_h, hurt_h, nu);
    VT_COPY(push_lo, push_lo, np), VT_COPY(push_hi, push_hi, np), VT_COPY(push_x, push_x, np);
    VT_COPY(push_y, push_y, np), VT_COPY(push_w, push_w, np), VT_COPY(push_h, push_h, np);
    VT_COPY(move_lo, move_lo, nm), VT_COPY(move_hi, move_hi, nm), VT_COPY(move_v, move_v, nm);
    VT_COPY(cancel_lo, cancel_lo, nc), VT_COPY(cancel_hi, cancel_hi, nc);
    VT_COPY(cancel_targets, cancel_targets, nc);
    VT_COPY(damage_action, damage_action, NUM_ATTACKS);
    VT_COPY(guard_action, guard_action, NUM_ATTACKS), VT_COPY(noh, noh, NUM_ATTACKS);
    VT_COPY(vd, vd, NUM_ATTACKS), VT_COPY(gd, gd, NUM_ATTACKS), VT_COPY(hs, hs, NUM_ATTACKS);
    VT_COPY(gs, gs, NUM_ATTACKS), VT_COPY(gbs, gbs, NUM_ATTACKS);
}

// ── Vector helpers (masks are -1 / 0 per lane) ───────────────────────────

VAI vi vsel(vi m, vi a, vi b) { return (m & a) | (~m & b); }
VAI vu vselu(vi m, vu a, vu b) { return (vu)vsel(m, (vi)a, (vi)b); }
VAI vf vself(vi m, vf a, vf b) { return (vf)vsel(m, (vi)a, (vi)b); }
VAI int vany(vi m) { return _mm512_test_epi32_mask((__m512i)m, (__m512i)m) != 0; }
VAI vi vmask(vi x) { return x != 0; }  // 0/1 or any int -> mask
VAI vi vbool(vi m) { return m & 1; }   // mask -> 0/1
VAI vi vsplat(int x) { return (vi){} + x; }
VAI vf vsplatf(float x) { return (vf){} + x; }

VAI vi lut(const int32_t* t, vi i) {  // t[i] for i in [0, 32)
    return (vi)_mm512_permutex2var_epi32(_mm512_load_si512(t), (__m512i)i,
                                         _mm512_load_si512(t + 16));
}
VAI vf lutf(const float* t, vi i) {
    return (vf)_mm512_permutex2var_ps(_mm512_load_ps(t), (__m512i)i, _mm512_load_ps(t + 16));
}
VAI vi lut64(const int32_t* t, vi i) {  // i in [0, 64)
    return vsel(vmask(i & 32), lut(t + 32, i), lut(t, i));
}
VAI vf lut64f(const float* t, vi i) { return vself(vmask(i & 32), lutf(t + 32, i), lutf(t, i)); }

VAI vi vclamp(vi x, int lo, int hi) {
    x = vsel(x < lo, vsplat(lo), x);
    return vsel(x > hi, vsplat(hi), x);
}
VAI vi vin(vi lo, vi hi, vi fr) { return (lo <= fr) & (fr <= hi); }

// mono_move on 16 lanes: two halves of 8 doubles
VAI vf vmono_move(vf pos, vf speed, vf sign) {
    vf s = speed * sign;  // exact (sign is +-1)
    __m512 p = (__m512)pos, sp = (__m512)s;
    __m512d dt = _mm512_set1_pd((double)FIXED_DELTA_TIME);
    __m512d p0 = _mm512_cvtps_pd(_mm512_castps512_ps256(p));
    __m512d p1 = _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(p), 1)));
    __m512d s0 = _mm512_cvtps_pd(_mm512_castps512_ps256(sp));
    __m512d s1 = _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(sp), 1)));
    __m256 r0 = _mm512_cvtpd_ps(_mm512_add_pd(p0, _mm512_mul_pd(s0, dt)));
    __m256 r1 = _mm512_cvtpd_ps(_mm512_add_pd(p1, _mm512_mul_pd(s1, dt)));
    return (vf)_mm512_castpd_ps(
        _mm512_insertf64x4(_mm512_castps_pd(_mm512_castps256_ps512(r0)), _mm256_castps_pd(r1), 1));
}

// ── Fighters ─────────────────────────────────────────────────────────────

typedef struct {
    vf pos, vel;
    vi vital, guard, action, af, hit, adv, hs, shake, buf, rdmg, face, won, inback, rprox, run;
    vu ll, lh, rl, rh, al, ah;  // input histories, lo/hi halves
} VFi;

#define VLOAD(T, row) (*(const T*)(row))
#define VSTORE(T, row, v) (*(T*)(row) = (v))

VAI VFi vf_load(const GameBlock* g, int p) {
    VFi f;
    f.pos = VLOAD(vf, g->pos_x[p]), f.vel = VLOAD(vf, g->vel_x[p]);
    f.vital = VLOAD(vi, g->vital[p]), f.guard = VLOAD(vi, g->guard[p]);
    f.action = VLOAD(vi, g->action[p]), f.af = VLOAD(vi, g->action_frame[p]);
    f.hit = VLOAD(vi, g->hit_count[p]), f.adv = VLOAD(vi, g->frame_adv[p]);
    f.hs = VLOAD(vi, g->hit_stun[p]), f.shake = VLOAD(vi, g->sprite_shake[p]);
    f.buf = VLOAD(vi, g->buffer_action[p]), f.rdmg = VLOAD(vi, g->reserve_damage_action[p]);
    f.face = VLOAD(vi, g->face_right[p]), f.won = VLOAD(vi, g->has_won[p]);
    f.inback = VLOAD(vi, g->input_backward[p]), f.rprox = VLOAD(vi, g->reserve_prox_guard[p]);
    f.run = VLOAD(vi, g->attack_run[p]);
    f.ll = VLOAD(vu, g->in_left_lo[p]), f.lh = VLOAD(vu, g->in_left_hi[p]);
    f.rl = VLOAD(vu, g->in_right_lo[p]), f.rh = VLOAD(vu, g->in_right_hi[p]);
    f.al = VLOAD(vu, g->in_attack_lo[p]), f.ah = VLOAD(vu, g->in_attack_hi[p]);
    return f;
}

VAI void vf_store(GameBlock* g, int p, const VFi* f) {
    VSTORE(vf, g->pos_x[p], f->pos), VSTORE(vf, g->vel_x[p], f->vel);
    VSTORE(vi, g->vital[p], f->vital), VSTORE(vi, g->guard[p], f->guard);
    VSTORE(vi, g->action[p], f->action), VSTORE(vi, g->action_frame[p], f->af);
    VSTORE(vi, g->hit_count[p], f->hit), VSTORE(vi, g->frame_adv[p], f->adv);
    VSTORE(vi, g->hit_stun[p], f->hs), VSTORE(vi, g->sprite_shake[p], f->shake);
    VSTORE(vi, g->buffer_action[p], f->buf), VSTORE(vi, g->reserve_damage_action[p], f->rdmg);
    VSTORE(vi, g->face_right[p], f->face), VSTORE(vi, g->has_won[p], f->won);
    VSTORE(vi, g->input_backward[p], f->inback), VSTORE(vi, g->reserve_prox_guard[p], f->rprox);
    VSTORE(vi, g->attack_run[p], f->run);
    VSTORE(vu, g->in_left_lo[p], f->ll), VSTORE(vu, g->in_left_hi[p], f->lh);
    VSTORE(vu, g->in_right_lo[p], f->rl), VSTORE(vu, g->in_right_hi[p], f->rh);
    VSTORE(vu, g->in_attack_lo[p], f->al), VSTORE(vu, g->in_attack_hi[p], f->ah);
}

// Keep the lanes of `old` where m is not set.
VAI void vf_merge(VFi* f, vi m, const VFi* old) {
#define M(x) f->x = vsel(m, f->x, old->x)
#define MU(x) f->x = vselu(m, f->x, old->x)
#define MF(x) f->x = vself(m, f->x, old->x)
    MF(pos), MF(vel), M(vital), M(guard), M(action), M(af), M(hit), M(adv), M(hs), M(shake);
    M(buf), M(rdmg), M(face), M(won), M(inback), M(rprox), M(run);
    MU(ll), MU(lh), MU(rl), MU(rh), MU(al), MU(ah);
#undef M
#undef MU
#undef MF
}

VAI vi vf_frame(const VFi* f) { return vclamp(f->af, 0, NUM_FRAMES - 1); }

VAI void vset_action(VFi* f, vi m, vi a) {
    f->action = vsel(m, a, f->action);
    f->af &= ~m;
    f->hit &= ~m;
    f->buf |= m;   // -1
    f->rdmg |= m;  // -1
    f->shake &= ~m;
}

VAI vi vcancel_buffer(vi a, vi fr, vi target) {
    vi c0 = lut(VT.cancel0, a), nc = lut(VT.ncancel, a), r = vsplat(0);
    FIXED for (int k = 0; k < MAX_CANCELS; k++) {
        vi i = c0 + k;
        vi targets = lut(VT.cancel_targets, i);
        r |= (k < nc) & vin(lut(VT.cancel_lo, i), lut(VT.cancel_hi, i), fr) &
             vmask((targets >> target) & 1);
    }
    return r;
}

// request_action where en
VAI void vreq(VFi* f, vi en, vi a) {
    if (!vany(en)) return;
    vi end = f->af >= lut(VT.fc, f->action);
    vi same = f->action == a;
    vi canc = vmask(lut(VT.canc, f->action));
    vi start = en & (end | (~same & canc));
    vi maybe = en & ~end & ~same & ~canc;
    vi buffer = vsplat(0);
    if (vany(maybe)) buffer = maybe & vcancel_buffer(f->action, vf_frame(f), a);
    vset_action(f, start, a);
    f->buf = vsel(buffer, a, f->buf);
}

VAI void vupdate_input(VFi* f, vi bits) {
    vu b = (vu)bits;
    f->lh = (f->lh << 1) | (f->ll >> 31), f->ll = (f->ll << 1) | (b & 1);
    f->rh = (f->rh << 1) | (f->rl >> 31), f->rl = (f->rl << 1) | ((b >> 1) & 1);
    f->ah = (f->ah << 1) | (f->al >> 31), f->al = (f->al << 1) | ((b >> 2) & 1);
    vi held = vmask(bits & INPUT_ATTACK);
    vi up = f->run + 1;
    f->run = held & vsel(up > 64, vsplat(64), up);
}

VAI vi vframes_left(const VFi* f) {
    vi canc = vmask(lut(VT.canc, f->action));
    return ~canc & (lut(VT.fc, f->action) - f->af);
}

VAI void vincrement_action_frame(VFi* f) {
    vi neg = -f->shake;
    vi shook = neg + vsel(neg > 0, vsplat(-1), vsplat(1));
    f->shake = vsel(vmask(f->shake), shook, f->shake);
    vi stun = f->hs > 0;
    vi frame = f->af + 1;
    vi loop = (frame >= lut(VT.fc, f->action)) & vmask(lut(VT.is_loop, f->action));
    frame = vsel(loop, lut(VT.loop_from, f->action), frame);
    f->hs = vsel(stun, f->hs - 1, f->hs);
    f->af = vsel(stun, f->af, frame);
}

// dash_check on the low 32 history bits (it reads at most bit 16)
VAI vi vdash(vu toward, vu away, int first, int last) {
    vu neutral = ~(toward | away);
    vi result = vsplat(0), decided = vsplat(0);
    FIXED for (int i = first; i < last; i++) {
        vi a = vmask((vi)((away >> i) & 1)), t = vmask((vi)((toward >> i) & 1));
        vi window = vmask((vi)(neutral & (((1u << (DASH_ALLOW_FRAME - 1)) - 1) << (i + 1))));
        result = vsel(decided, result, ~a & t & window);
        decided |= a | t;
    }
    return result;
}

VAI vi vpressed(vu h) { return vmask((vi)(h & 1)) & ~vmask((vi)(h & 2)); }

#define HELD_LO ((uint32_t)(HELD_SINCE_1 & 0xFFFFFFFFu))
#define HELD_HI ((uint32_t)(HELD_SINCE_1 >> 32))

VAI void vupdate_action_request(VFi* f) {
    vi won = vmask(f->won);
    vi hs_ok = f->hs <= 0;
    vi use_res = ~won & (f->rdmg != -1) & hs_ok;
    vi use_buf = ~won & ~use_res & (f->buf != -1) &
                 (CAN_CANCEL_ON_WHIFF ? vsplat(-1) : (f->hit > 0)) & hs_ok;
    vi normal = ~(won | use_res | use_buf);
    vi res_a = f->rdmg, buf_a = f->buf;

    vi face = vmask(f->face);
    vu fwd = vselu(face, f->rl, f->ll), bwd = vselu(face, f->ll, f->rl);
    vi is_f = vmask((vi)(fwd & 1)), is_b = vmask((vi)(bwd & 1));
    vi directional = is_f | is_b;
    vi special = ~vmask((vi)(f->al & 1)) & ((vi)(f->al & HELD_LO) == (vi)vsplat((int)HELD_LO)) &
                 ((vi)(f->ah & HELD_HI) == (vi)vsplat((int)HELD_HI));
    vi attack = vpressed(f->al);
    vi in_normal = ((f->action == ACT_N_ATTACK) | (f->action == ACT_B_ATTACK)) &
                   ~(f->af >= lut(VT.fc, f->action));
    vi normal_a = vsel(in_normal, vsplat(ACT_N_SPECIAL),
                       vsel(directional, vsplat(ACT_B_ATTACK), vsplat(ACT_N_ATTACK)));
    vi fpress = normal & vpressed(fwd), bpress = normal & vpressed(bwd);
    vi fdash = vsplat(0), bdash = vsplat(0);
    if (vany(fpress)) fdash = fpress & vdash(fwd, bwd, 1, DASH_ALLOW_FRAME);
    if (vany(bpress)) bdash = bpress & vdash(bwd, fwd, 1, DASH_ALLOW_FRAME);
    vi move = vsel(is_f & ~is_b, vsplat(ACT_FORWARD),
                   vsel(is_b & ~is_f,
                        vsel(vmask(f->rprox), vsplat(ACT_GUARD_PROXIMITY), vsplat(ACT_BACKWARD)),
                        vsplat(ACT_STAND)));

    vreq(f, won, vsplat(ACT_WIN));
    vset_action(f, use_res, res_a);
    vset_action(f, use_buf, buf_a);
    vreq(f, normal & special,
         vsel(directional, vsplat(ACT_B_SPECIAL), vsplat(ACT_N_SPECIAL)));
    vreq(f, normal & ~special & attack, normal_a);
    vreq(f, fdash, vsplat(ACT_DASH_FORWARD));
    vreq(f, ~fdash & bdash, vsplat(ACT_DASH_BACKWARD));
    f->inback = vsel(normal, vbool(is_b), f->inback);
    vreq(f, normal, move);
    f->rprox &= ~normal;
}

VAI vf vsign(const VFi* f) { return vself(vmask(f->face), vsplatf(1.0f), vsplatf(-1.0f)); }

VAI void vupdate_movement(VFi* f) {
    vi a = f->action, fr = vf_frame(f);
    vi m0 = lut(VT.move0, a), nm = lut(VT.nmove, a);
    vi has = vsplat(0);
    vf v = vsplatf(0.0f);
    if (vany(nm > 0)) {
        FIXED for (int k = 0; k < MAX_MOVES; k++) {  // the first active entry
            vi i = m0 + k;
            vi ok = (k < nm) & vin(lut64(VT.move_lo, i), lut64(VT.move_hi, i), fr) & ~has;
            v = vself(ok, lut64f(VT.move_v, i), v);
            has |= ok;
        }
    }
    vi fwd = a == ACT_FORWARD, bwd = a == ACT_BACKWARD;
    vi by_data = ~fwd & ~bwd & has;
    vi moves = fwd | bwd | (by_data & (v != 0.0f));
    vf speed = vself(fwd, vsplatf(FORWARD_MOVE_SPEED),
                     vself(bwd, vsplatf(-BACKWARD_MOVE_SPEED), v));
    vi active = f->hs <= 0;
    vi go = active & moves;
    if (vany(go)) f->pos = vself(go, vmono_move(f->pos, speed, vsign(f)), f->pos);
    f->vel = vself(active & by_data, v, f->vel);
}

typedef struct {
    vf hx[MAX_HITBOXES], hy[MAX_HITBOXES], hw[MAX_HITBOXES], hh[MAX_HITBOXES];
    vi hvalid[MAX_HITBOXES], hatk[MAX_HITBOXES], hprox[MAX_HITBOXES];
    vf ux[MAX_HURTBOXES], uy[MAX_HURTBOXES], uw[MAX_HURTBOXES], uh[MAX_HURTBOXES];
    vi uvalid[MAX_HURTBOXES];
    vf px, py, pw, ph;
    vi any_hit;
} VB;

VAI void vboxes(const VFi* f, VB* b) {
    vi a = f->action, fr = vf_frame(f);
    vf sign = vsign(f), zero = vsplatf(0.0f);
    vi h0 = lut(VT.hit0, a), nh = lut(VT.nhit, a);
    b->any_hit = vsplat(0);
    FIXED for (int k = 0; k < MAX_HITBOXES; k++) {
        vi i = h0 + k;
        vi valid = (k < nh) & vin(lut(VT.hit_lo, i), lut(VT.hit_hi, i), fr);
        b->hvalid[k] = valid;
        b->any_hit |= valid;
        b->hx[k] = f->pos + vself(valid, lutf(VT.hit_x, i), zero) * sign;
        b->hy[k] = vself(valid, lutf(VT.hit_y, i), zero);
        b->hw[k] = vself(valid, lutf(VT.hit_w, i), zero);
        b->hh[k] = vself(valid, lutf(VT.hit_h, i), zero);
        b->hatk[k] = valid & lut(VT.hit_atk, i);
        b->hprox[k] = valid & vmask(lut(VT.hit_prox, i));
    }
    vi u0 = lut(VT.hurt0, a), nu = lut(VT.nhurt, a);
    FIXED for (int k = 0; k < MAX_HURTBOXES; k++) {
        vi i = u0 + k;
        vi valid = (k < nu) & vin(lut(VT.hurt_lo, i), lut(VT.hurt_hi, i), fr);
        b->uvalid[k] = valid;
        b->ux[k] = f->pos + vself(valid, lutf(VT.hurt_x, i), zero) * sign;
        b->uy[k] = vself(valid, lutf(VT.hurt_y, i), zero);
        b->uw[k] = vself(valid, lutf(VT.hurt_w, i), zero);
        b->uh[k] = vself(valid, lutf(VT.hurt_h, i), zero);
    }
    vi p0 = lut(VT.push0, a), np = lut(VT.npush, a);
    vi has = vsplat(0);
    vf x = zero, y = zero, w = zero, h = zero;
    FIXED for (int k = 0; k < MAX_PUSHBOXES; k++) {  // the first active entry
        vi i = p0 + k;
        vi ok = (k < np) & vin(lut(VT.push_lo, i), lut(VT.push_hi, i), fr) & ~has;
        x = vself(ok, lutf(VT.push_x, i), x), y = vself(ok, lutf(VT.push_y, i), y);
        w = vself(ok, lutf(VT.push_w, i), w), h = vself(ok, lutf(VT.push_h, i), h);
        has |= ok;
    }
    b->px = f->pos + x * sign, b->py = y, b->pw = w, b->ph = h;
}

VAI void vshift(VFi* f, VB* b, vi en, vf dx) {
    f->pos = vself(en, f->pos + dx, f->pos);
    FIXED for (int k = 0; k < MAX_HITBOXES; k++) b->hx[k] = vself(en, b->hx[k] + dx, b->hx[k]);
    FIXED for (int k = 0; k < MAX_HURTBOXES; k++) b->ux[k] = vself(en, b->ux[k] + dx, b->ux[k]);
    b->px = vself(en, b->px + dx, b->px);
}

VAI void vpush_character_vs_character(VFi* f0, VFi* f1, VB* b0, VB* b1) {
    vf x1 = b0->px, y1 = b0->py, w1 = b0->pw, h1 = b0->ph;
    vf x2 = b1->px, y2 = b1->py, w2 = b1->pw, h2 = b1->ph;
    vi overlaps = (x2 + w2 > x1) & (x2 < x1 + w1) & (y2 + h2 > y1) & (y2 < y1 + h1);
    vi p1_left = overlaps & (f0->pos < f1->pos);
    vi p1_right = overlaps & (f0->pos > f1->pos);
    vi apply = p1_left | p1_right;
    if (!vany(apply)) return;
    vf left_overlap = (x1 + w1) - x2;
    vf right_overlap = (x2 + w2) - x1;
    vf half = vsplatf(0.5f);
    vf dx0 = vself(p1_left, -left_overlap * half, right_overlap * half);
    vf dx1 = vself(p1_left, left_overlap * half, -right_overlap * half);
    vshift(f0, b0, apply, dx0);
    vshift(f1, b1, apply, dx1);
}

VAI void vpush_vs_background(VFi* f, VB* b) {
    vf half = vsplatf(0.5f);
    vf x_min = b->px - b->pw * half, x_max = b->px + b->pw * half;
    vi left = x_min < STAGE_MIN_X;
    vi right = ~left & (x_max > STAGE_MAX_X);
    vi apply = left | right;
    if (!vany(apply)) return;
    vf dx = vself(left, vsplatf(STAGE_MIN_X) - x_min, vsplatf(STAGE_MAX_X) - x_max);
    vshift(f, b, apply, dx);
}

VAI vi voverlap(vf ax, vf ay, vf aw, vf ah, vf bx, vf by, vf bw, vf bh) {
    vf half = vsplatf(0.5f);
    vf a_x_min = ax - aw * half, a_x_max = ax + aw * half;
    vf b_x_min = bx - bw * half, b_x_max = bx + bw * half;
    vf a_y_max = ay + ah, b_y_max = by + bh;
    return (b_x_max >= a_x_min) & (b_x_min <= a_x_max) & (b_y_max >= ay) & (by <= a_y_max);
}

VAI void vcheck_attack(VFi* att, VFi* dmg, const VB* ab, const VB* db) {
    if (!vany(ab->any_hit)) return;
    vi is_hit = vsplat(0), attack = vsplat(0), prox = vsplat(0);
    FIXED for (int k = 0; k < MAX_HITBOXES; k++) {
        vi valid = ab->hvalid[k] & (att->hit < lut(VT.noh, ab->hatk[k]));
        vi touching = vsplat(0);
        FIXED for (int j = 0; j < MAX_HURTBOXES; j++)
            touching |= db->uvalid[j] & voverlap(ab->hx[k], ab->hy[k], ab->hw[k], ab->hh[k],
                                                 db->ux[j], db->uy[j], db->uw[j], db->uh[j]);
        vi t = valid & touching;
        prox |= t & ab->hprox[k];
        vi first = t & ~ab->hprox[k] & ~is_hit;  // the first damaging box in asset order
        attack = vsel(first, ab->hatk[k], attack);
        is_hit |= first;
    }
    if (vany(is_hit)) {  // NotifyDamaged
        vi gd = lut(VT.gd, attack);
        vi guard1 = vsel(gd > 0, dmg->guard - gd, dmg->guard);
        vi guard_break = (gd > 0) & (guard1 < 0);
        vi guard2 = ~guard_break & guard1;
        vi guarding = (dmg->action == ACT_BACKWARD) | vmask(lut(VT.is_guard, dmg->action));
        vi vd = lut(VT.vd, attack);
        vi vital1 = dmg->vital - vd;
        vi vital2 = vsel(vd > 0, ~(vital1 <= 0) & vital1, dmg->vital);
        vi new_action = vsel(guarding, lut(VT.guard_action, attack), lut(VT.damage_action, attack));
        vi stun = vsel(guarding, vsel(guard_break, lut(VT.gbs, attack), lut(VT.gs, attack)),
                       lut(VT.hs, attack));
        dmg->guard = vsel(is_hit, guard2, dmg->guard);
        vset_action(dmg, is_hit, new_action);
        dmg->rdmg = vsel(is_hit & guarding & guard_break, vsplat(ACT_GUARD_BREAK), dmg->rdmg);
        dmg->vital = vsel(is_hit & ~guarding, vital2, dmg->vital);
        dmg->hs = vsel(is_hit, stun, dmg->hs);
        vi shake = (stun * 43691) >> 17;  // stun / 3 for 0 <= stun < 98304
        shake = vsel(shake > MAX_SPRITE_SHAKE_FRAME, vsplat(MAX_SPRITE_SHAKE_FRAME), shake);
        shake = vsel(vmask(dmg->face), -shake, shake);
        dmg->shake = vsel(is_hit, shake, dmg->shake);
        att->hit = vsel(is_hit, att->hit + 1, att->hit);
        att->hs = vsel(is_hit, stun, att->hs);
    }
    // NotifyInProximityGuardRange
    dmg->rprox = vsel(~is_hit & prox, dmg->rprox | dmg->inback, dmg->rprox);
}

VAI void vsetup_battle_start(VFi* f, vi en, float start_x, int face_right) {
    f->pos = vself(en, vsplatf(start_x), f->pos);
    f->face = vsel(en, vsplat(face_right), f->face);
    f->vital = vsel(en, vsplat(1), f->vital);
    f->guard = vsel(en, vsplat(START_GUARD_HEALTH), f->guard);
    f->won &= ~en;
    f->vel = vself(en, vsplatf(0.0f), f->vel);
    vu keep = (vu)~en;
    f->ll &= keep, f->lh &= keep, f->rl &= keep, f->rh &= keep, f->al &= keep, f->ah &= keep;
    f->run &= ~en;
    vset_action(f, en, vsplat(ACT_STAND));
}

// StepN of one block (footsies_batch.h's block_game_step_n), on registers.
__attribute__((noinline)) static void vblock_game_step_n(GameBlock* g, const BlockScratch* s,
                                                        int n_frames) {
    VFi f0 = vf_load(g, 0), f1 = vf_load(g, 1);
    vi round_state = VLOAD(vi, g->round_state), frame_count = VLOAD(vi, g->frame_count);
    vi done = VLOAD(vi, g->done), reward = VLOAD(vi, g->reward);
    vi bits0 = VLOAD(vi, s->bits[0]), bits1 = VLOAD(vi, s->bits[1]);
    vi running = vsplat(-1);
    for (int k = 0; k < n_frames; k++) {
        int all = !vany(~running);
        VFi o0 = f0, o1 = f1;
        vi reset = round_state != ROUND_FIGHT;
        if (vany(reset)) {
            vsetup_battle_start(&f0, reset, -2.0f, 1);
            vsetup_battle_start(&f1, reset, 2.0f, 0);
        }
        vi fc = vsel(reset, vsplat(-1), frame_count);

        vupdate_input(&f0, bits0);
        vupdate_input(&f1, bits1);
        vi l0 = vframes_left(&f0), l1 = vframes_left(&f1);
        f0.adv = l1 - l0;
        f1.adv = l0 - l1;
        vincrement_action_frame(&f0);
        vincrement_action_frame(&f1);
        vupdate_action_request(&f0);
        vupdate_action_request(&f1);
        vupdate_movement(&f0);
        vupdate_movement(&f1);

        VB b0, b1;
        vboxes(&f0, &b0);
        vboxes(&f1, &b1);
        vpush_character_vs_character(&f0, &f1, &b0, &b1);
        vpush_vs_background(&f0, &b0);
        vpush_vs_background(&f1, &b1);
        vcheck_attack(&f0, &f1, &b0, &b1);
        vcheck_attack(&f1, &f0, &b1, &b0);

        vi d0 = f0.vital <= 0, d1 = f1.vital <= 0;
        vi new_done = d0 | d1;
        vi new_reward = vbool(d1 & ~d0) - vbool(d0 & ~d1);
        if (!all) {
            vf_merge(&f0, running, &o0);
            vf_merge(&f1, running, &o1);
        }
        done = vsel(running, vbool(new_done), done);
        reward = vsel(running, new_reward, reward);
        round_state = vsel(running, vsel(new_done, vsplat(ROUND_KO), vsplat(ROUND_FIGHT)),
                           round_state);
        frame_count = vsel(running, fc + 1, frame_count);
        running &= ~new_done;
        if (!vany(running)) break;
    }
    vf_store(g, 0, &f0);
    vf_store(g, 1, &f1);
    VSTORE(vi, g->round_state, round_state), VSTORE(vi, g->frame_count, frame_count);
    VSTORE(vi, g->done, done), VSTORE(vi, g->reward, reward);
}


// ── Env step on vectors ──────────────────────────────────────────────────

VAI vf vfloat(vi x) { return __builtin_convertvector(x, vf); }
VAI vf vone(vi m) { return vself(m, vsplatf(1.0f), vsplatf(0.0f)); }
VAI vf vint_feature(vi k, float d) { return vfloat(vclamp(k, -1024, 1024)) / vsplatf(d); }
VAI vi vint8(vi x) { return (x << 24) >> 24; }  // (int32_t)(int8_t)x
VAI vi vlut4(uint32_t table, vi i) { return (vi)(((vu){} + table) >> (vu)(i * 4)) & 0xF; }

#define ROW(T, x) (*(T*)(x))

// 37 public features of fighter p into rows o[0..36]
VAI void vencode_public(const GameBlock* g, int p, float (*o)[BATCH_W]) {
    vi action = ROW(vi, g->action[p]), frame = ROW(vi, g->action_frame[p]);
    vi count = lut(VT.fc, action);
    vf vel = ROW(vf, g->vel_x[p]);
    vi vital = ROW(vi, g->vital[p]), guard = ROW(vi, g->guard[p]), hs = ROW(vi, g->hit_stun[p]);
    ROW(vf, o[0]) = ROW(vf, g->pos_x[p]) * vsplatf(0.25f);
    ROW(vf, o[1]) = vself(vel == 0.0f, vsplatf(0.0f), vel / vsplatf(5.0f));
    ROW(vf, o[2]) = vone(vital <= 0);
    ROW(vf, o[3]) = vfloat(vital);
    FIXED for (int k = 0; k < 4; k++) ROW(vf, o[4 + k]) = vone(guard == k);
    FIXED for (int k = 0; k < NUM_ACTIONS; k++) ROW(vf, o[8 + k]) = vone(action == k);
    ROW(vf, o[25]) = vint_feature(frame, 25.0f);
    ROW(vf, o[26]) = vint_feature(count, 25.0f);
    ROW(vf, o[27]) = vint_feature(count - frame, 25.0f);
    ROW(vf, o[28]) = vone(frame >= count);
    ROW(vf, o[29]) = vone(vmask(lut(VT.canc, action)));
    ROW(vf, o[30]) = vfloat(ROW(vi, g->hit_count[p]));
    ROW(vf, o[31]) = vint_feature(hs, 10.0f);
    ROW(vf, o[32]) = vone(hs > 0);
    ROW(vf, o[33]) = vfloat(ROW(vi, g->sprite_shake[p]));
    ROW(vf, o[34]) = vint_feature(vsplat(MAX_SPRITE_SHAKE_FRAME), 10.0f);
    ROW(vf, o[35]) = vone(vmask(ROW(vi, g->face_right[p])));
    ROW(vf, o[36]) = vint_feature(ROW(vi, g->frame_adv[p]), 10.0f);
}

VAI void vencode_private(const EnvBlock* e, int p, float (*o)[BATCH_W]) {
    const GameBlock* g = &e->g;
    vi face = vmask(ROW(vi, g->face_right[p]));
    vu left = ROW(vu, g->in_left_lo[p]), right = ROW(vu, g->in_right_lo[p]);
    vu fwd = vselu(face, right, left), bwd = vselu(face, left, right);
    ROW(vf, o[0]) = vone(vdash(fwd, bwd, 0, DASH_ALLOW_FRAME - 1));
    ROW(vf, o[1]) = vone(vdash(bwd, fwd, 0, DASH_ALLOW_FRAME - 1));
    vi run = ROW(vi, g->attack_run[p]);
    run = vsel(run > SPECIAL_ATTACK_HOLD_FRAME, vsplat(SPECIAL_ATTACK_HOLD_FRAME), run);
    vf progress = vfloat(run) / vsplatf((float)SPECIAL_ATTACK_HOLD_FRAME);
    ROW(vf, o[2]) = vself(progress < 1.0f, progress, vsplatf(1.0f));
    vi prev = ROW(vi, e->prev_selected[p]);
    FIXED for (int k = 0; k < NUM_ENV_ACTIONS; k++) ROW(vf, o[3 + k]) = vone(prev == k);
    ROW(vf, o[12]) = vone(vmask(ROW(vi, e->holding[p])));
}

static inline void vblock_encode(const EnvBlock* e, BlockScratch* s) {
    const GameBlock* g = &e->g;
    vf d = ROW(vf, g->pos_x[0]) - ROW(vf, g->pos_x[1]);
    d = (vf)((vi)d & 0x7FFFFFFF) * vsplatf(0.125f);  // fabsf
    ROW(vf, s->obs[0]) = d;
    ROW(vf, s->obs[OBS_DIM]) = d;
    vencode_public(g, 0, &s->obs[1]);
    vencode_public(g, 1, &s->obs[OBS_DIM + 1]);
    vencode_private(e, 0, &s->obs[1 + PUBLIC_DIM]);
    vencode_private(e, 1, &s->obs[OBS_DIM + 1 + PUBLIC_DIM]);
    for (int f = 0; f < PUBLIC_DIM; f++) {
        ROW(vf, s->obs[1 + PUBLIC_DIM + PRIVATE_DIM + f]) = ROW(vf, s->obs[OBS_DIM + 1 + f]);
        ROW(vf, s->obs[OBS_DIM + 1 + PUBLIC_DIM + PRIVATE_DIM + f]) = ROW(vf, s->obs[1 + f]);
    }
}

// 16x16 transpose of r (row i becomes column i)
VAI void vtranspose16(__m512 r[16]) {
    __m512 t[16];
    for (int i = 0; i < 8; i++) {
        t[2 * i] = _mm512_unpacklo_ps(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_ps(r[2 * i], r[2 * i + 1]);
    }
    for (int i = 0; i < 4; i++) {
        __m512d a = _mm512_castps_pd(t[4 * i]), b = _mm512_castps_pd(t[4 * i + 2]);
        __m512d c = _mm512_castps_pd(t[4 * i + 1]), d = _mm512_castps_pd(t[4 * i + 3]);
        r[4 * i] = _mm512_castpd_ps(_mm512_unpacklo_pd(a, b));
        r[4 * i + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(a, b));
        r[4 * i + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(c, d));
        r[4 * i + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(c, d));
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            t[8 * i + j] = _mm512_shuffle_f32x4(r[8 * i + j], r[8 * i + j + 4], 0x88);
            t[8 * i + j + 4] = _mm512_shuffle_f32x4(r[8 * i + j], r[8 * i + j + 4], 0xDD);
        }
    }
    for (int j = 0; j < 8; j++) {
        r[j] = _mm512_shuffle_f32x4(t[j], t[j + 8], 0x88);
        r[j + 8] = _mm512_shuffle_f32x4(t[j], t[j + 8], 0xDD);
    }
}

// The block's [feature][lane] observations to obs[lane][feature]
static inline void vblock_write_obs(const BlockScratch* s, float* obs, int count) {
    for (int g = 0; g < 2 * OBS_DIM / 16; g++) {
        __m512 r[16];
        for (int i = 0; i < 16; i++) r[i] = _mm512_load_ps(s->obs[16 * g + i]);
        vtranspose16(r);
        for (int l = 0; l < count; l++)
            _mm512_storeu_ps(obs + (size_t)l * 2 * OBS_DIM + 16 * g, r[l]);
    }
}

// Fresh episodes in the lanes of `m`
static inline void vblock_env_fresh(EnvBlock* e, vi m, vi episode_id) {
    GameBlock* g = &e->g;
    for (int p = 0; p < 2; p++) {
        VFi f = vf_load(g, p);
        f.adv &= ~m, f.hs &= ~m, f.inback &= ~m, f.rprox &= ~m;
        vsetup_battle_start(&f, m, p == 0 ? -2.0f : 2.0f, p == 0);
        vf_store(g, p, &f);
        ROW(vi, e->budget[p]) &= ~m;
        ROW(vi, e->last_guard[p]) = vsel(m, vsplat(START_GUARD_HEALTH), ROW(vi, e->last_guard[p]));
        for (int k = 0; k < MAX_DELAY_STEPS; k++) ROW(vi, e->queue[p][k]) &= ~m;
        ROW(vi, e->prev_selected[p]) &= ~m;
        ROW(vi, e->holding[p]) &= ~m;
    }
    ROW(vi, g->round_state) = vsel(m, vsplat(ROUND_FIGHT), ROW(vi, g->round_state));
    ROW(vi, g->frame_count) = vsel(m, vsplat(-1), ROW(vi, g->frame_count));
    ROW(vi, g->done) &= ~m;
    ROW(vi, g->reward) &= ~m;
    ROW(vi, e->t) &= ~m;
    ROW(vi, e->episode_id) = vsel(m, episode_id, ROW(vi, e->episode_id));
    ROW(vi, e->queue_head) &= ~m;
}

VAI vi vresolve_special_charge(EnvBlock* e, int p, vi action) {
    vi is_special = (action == ENV_SPECIAL_CHARGE) | (action == ENV_FORWARD_SPECIAL_CHARGE) |
                    (action == ENV_BACK_SPECIAL_CHARGE);
    vi holding = ROW(vi, e->holding[p]) ^ (is_special & 1);
    ROW(vi, e->holding[p]) = holding;
    vi base = vsel(action == ENV_SPECIAL_CHARGE, vsplat(ENV_NONE),
                   vsel(action == ENV_FORWARD_SPECIAL_CHARGE, vsplat(ENV_FORWARD),
                        vsel(action == ENV_BACK_SPECIAL_CHARGE, vsplat(ENV_BACK), action)));
    return vsel(vmask(holding), vlut4(CHARGE_ACTION_LUT, vclamp(base, 0, 5)), base);
}

// block_env_step on vectors
static inline void vblock_env_step(EnvBlock* e, const EnvConfig* cfg, const int32_t* actions,
                                   int count, float* obs, float* rew, uint8_t* term,
                                   uint8_t* trunc, int autoreset) {
    BlockScratch s;
    vi a0, a1;
    if (count == BATCH_W) {  // deinterleave (p1, p2) pairs
        __m512i lo = _mm512_loadu_si512(actions), hi = _mm512_loadu_si512(actions + 16);
        __m512i even = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
        a0 = (vi)_mm512_permutex2var_epi32(lo, even, hi);
        a1 = (vi)_mm512_permutex2var_epi32(lo, _mm512_add_epi32(even, _mm512_set1_epi32(1)), hi);
    } else {  // lanes past `count` idle on action 0
        int32_t t0[BATCH_W] B_ALIGNED = {0}, t1[BATCH_W] B_ALIGNED = {0};
        for (int l = 0; l < count; l++) t0[l] = actions[2 * l], t1[l] = actions[2 * l + 1];
        a0 = ROW(vi, t0), a1 = ROW(vi, t1);
    }

    vi x0 = a0, x1 = a1;
    if (cfg->action_delay_frames != 0) {
        vi head = ROW(vi, e->queue_head);
        vi q0 = vint8(a0), q1 = vint8(a1);
        for (int k = 0; k < MAX_DELAY_STEPS; k++) {
            vi here = head == k;
            vi old0 = ROW(vi, e->queue[0][k]), old1 = ROW(vi, e->queue[1][k]);
            ROW(vi, e->queue[0][k]) = vsel(here, q0, old0);
            ROW(vi, e->queue[1][k]) = vsel(here, q1, old1);
            x0 = vsel(here, old0, x0);
            x1 = vsel(here, old1, x1);
        }
        vi next = head + 1;
        ROW(vi, e->queue_head) = vsel(next == delay_steps(cfg), vsplat(0), next);
    }
    vi base0 = vresolve_special_charge(e, 0, x0), base1 = vresolve_special_charge(e, 1, x1);
    ROW(vi, s.bits[0]) = vlut4(P1_ENV_TO_BITS, vclamp(base0, 0, 5));
    ROW(vi, s.bits[1]) = vlut4(P2_ENV_TO_BITS, vclamp(base1, 0, 5));
    vblock_game_step_n(&e->g, &s, cfg->frame_skip);

    // Rewards (RewardFunction)
    GameBlock* g = &e->g;
    vi t = ROW(vi, e->t) + 1;
    ROW(vi, e->t) = t;
    vi guard0 = ROW(vi, g->guard[0]), guard1 = ROW(vi, g->guard[1]);
    vi dead0 = vbool(ROW(vi, g->vital[0]) <= 0), dead1 = vbool(ROW(vi, g->vital[1]) <= 0);
    vi broken0 = vsplat(0), broken1 = vsplat(0);
    if (cfg->guard_break_reward_nonzero) {
        broken0 = vbool(guard0 < ROW(vi, e->last_guard[0]));
        broken1 = vbool(guard1 < ROW(vi, e->last_guard[1]));
    }
    if (cfg->use_reward_budget) {
        ROW(vi, e->budget[0]) += broken1;
        ROW(vi, e->budget[1]) += broken0;
    }
    vi k0 = vclamp(ROW(vi, e->budget[0]), 0, MAX_BUDGET_DEDUCTIONS);
    vi k1 = vclamp(ROW(vi, e->budget[1]), 0, MAX_BUDGET_DEDUCTIONS);
    vi idx = ((((((broken1 * 2 + broken0) * 2 + dead1) * 2 + dead0) * 4 + k0) * 4 + k1) * 2);
    __m512 r0 = _mm512_i32gather_ps((__m512i)idx, cfg->reward_table, 4);
    __m512 r1 = _mm512_i32gather_ps((__m512i)(idx + 1), cfg->reward_table, 4);
    ROW(vi, e->prev_selected[0]) = vint8(a0);
    ROW(vi, e->prev_selected[1]) = vint8(a1);
    ROW(vi, e->last_guard[0]) = guard0;
    ROW(vi, e->last_guard[1]) = guard1;
    vi te = dead0 | dead1, tr = vbool(t >= cfg->max_t);
    vi over = vmask(te | tr);
    if (autoreset && vany(over)) vblock_env_fresh(e, over, ROW(vi, e->episode_id) + 1);

    vblock_encode(e, &s);
    vblock_write_obs(&s, obs, count);
    __mmask16 m = (__mmask16)((1u << count) - 1u);
    float rr0[BATCH_W] B_ALIGNED, rr1[BATCH_W] B_ALIGNED;
    _mm512_store_ps(rr0, r0), _mm512_store_ps(rr1, r1);
    for (int l = 0; l < count; l++) rew[2 * l] = rr0[l], rew[2 * l + 1] = rr1[l];
    _mm512_mask_cvtepi32_storeu_epi8(term, m, (__m512i)te);
    _mm512_mask_cvtepi32_storeu_epi8(trunc, m, (__m512i)tr);
}

#endif  // FOOTSIES_AVX512_H
