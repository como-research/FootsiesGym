"""Generate ``csrc/frame_data.gen.h`` from ``footsiesgym.jax.frame_data``.

The C port reads exactly the values the JAX port reads: every float goes
through the same ``np.float32`` conversion as ``frame_data._build`` and is
written as a hex literal, so no decimal parsing can change a bit. Instead
of the dense ``[action, frame]`` tables, the header holds the authored
frame ranges (a few KB), with ``use_base_rect`` and ``execute | buffer``
resolved here. Before writing, the ranges are expanded back into dense
tables and compared with ``FRAME_DATA`` bit for bit.

    python scripts/gen_c_frame_data.py           # write the header
    python scripts/gen_c_frame_data.py --check   # fail if it is out of date
"""

import argparse
import pathlib
import sys

import numpy as np

from footsiesgym.footsies.game import constants
from footsiesgym.jax import frame_data as fd

f32 = np.float32
OUT = pathlib.Path(__file__).resolve().parents[1] / "csrc" / "frame_data.gen.h"
ACTION_INDEX = {int(a): i for i, a in enumerate(fd.ACTION_IDS)}
ATTACK_INDEX = {a["attackID"]: i for i, a in enumerate(fd._ATTACKS)}


def hexf(x) -> str:
    """A C float literal with exactly the bits of float32 ``x``."""
    x = f32(x)
    assert np.isfinite(x)
    return float(x).hex() + "f"


def rect(r) -> list[str]:
    return [hexf(v) for v in np.asarray(r, f32)]


def entries():
    """The authored ranges per kind, in asset order, and per-action slices."""
    kinds = {k: [] for k in ("hit", "hurt", "push", "move", "cancel")}
    slices = []
    for a in fd._ACTIONS:
        s = {}
        for kind, key in (
            ("hit", "hitboxes"),
            ("hurt", "hurtboxes"),
            ("push", "pushboxes"),
            ("move", "movements"),
            ("cancel", "cancels"),
        ):
            start = len(kinds[kind])
            for e in a[key]:
                lo, hi = e["frames"]
                if kind == "hit":
                    kinds[kind].append(
                        dict(
                            lo=lo,
                            hi=hi,
                            rect=np.array(e["rect"], f32),
                            attack=ATTACK_INDEX[e["attack_id"]],
                            proximity=bool(e["proximity"]),
                        )
                    )
                elif kind in ("hurt", "push"):
                    base = (
                        fd.BASE_HURTBOX_RECT if kind == "hurt" else fd.BASE_PUSHBOX_RECT
                    )
                    r = base if e["use_base_rect"] else np.array(e["rect"], f32)
                    kinds[kind].append(dict(lo=lo, hi=hi, rect=np.asarray(r, f32)))
                elif kind == "move":
                    kinds[kind].append(
                        dict(lo=lo, hi=hi, velocity=f32(e["velocity_x"]))
                    )
                else:
                    if not (e["execute"] or e["buffer"]):
                        continue  # never read by RequestAction
                    mask = 0
                    for target in e["action_ids"]:
                        mask |= 1 << ACTION_INDEX[target]
                    kinds[kind].append(dict(lo=lo, hi=hi, targets=mask))
            s[kind] = (start, len(kinds[kind]) - start)
        slices.append(s)
    return kinds, slices


def expand(kinds, slices) -> dict[str, np.ndarray]:
    """Dense tables from the ranges, with the C lookup semantics."""
    A, F = fd.NUM_ACTIONS, fd.NUM_FRAMES
    t = dict(
        hitbox_rect=np.zeros((A, F, fd.MAX_HITBOXES, 4), f32),
        hitbox_valid=np.zeros((A, F, fd.MAX_HITBOXES), bool),
        hitbox_attack=np.zeros((A, F, fd.MAX_HITBOXES), np.int32),
        hitbox_proximity=np.zeros((A, F, fd.MAX_HITBOXES), bool),
        hurtbox_rect=np.zeros((A, F, fd.MAX_HURTBOXES, 4), f32),
        hurtbox_valid=np.zeros((A, F, fd.MAX_HURTBOXES), bool),
        pushbox_rect=np.zeros((A, F, 4), f32),
        has_movement=np.zeros((A, F), bool),
        movement_velocity=np.zeros((A, F), f32),
        cancel_buffer=np.zeros((A, F, A), bool),
    )
    for a, s in enumerate(slices):
        for fr in range(F):
            active = lambda kind: [
                (k, e)
                for k, e in enumerate(kinds[kind][s[kind][0] : sum(s[kind])])
                if e["lo"] <= fr <= e["hi"]
            ]
            for k, e in active("hit"):
                t["hitbox_rect"][a, fr, k] = e["rect"]
                t["hitbox_valid"][a, fr, k] = True
                t["hitbox_attack"][a, fr, k] = e["attack"]
                t["hitbox_proximity"][a, fr, k] = e["proximity"]
            for k, e in active("hurt"):
                t["hurtbox_rect"][a, fr, k] = e["rect"]
                t["hurtbox_valid"][a, fr, k] = True
            push = active("push")
            if push:
                t["pushbox_rect"][a, fr] = push[0][1]["rect"]
            move = active("move")
            if move:
                t["has_movement"][a, fr] = True
                t["movement_velocity"][a, fr] = move[0][1]["velocity"]
            for _, e in active("cancel"):
                for target in range(A):
                    if e["targets"] >> target & 1:
                        t["cancel_buffer"][a, fr, target] = True
    return t


def check_against_jax(kinds, slices):
    dense = expand(kinds, slices)
    for name, table in dense.items():
        want = getattr(fd.FRAME_DATA, name)
        if table.dtype == f32:
            same = np.array_equal(table.view(np.int32), want.view(np.int32))
        else:
            same = np.array_equal(table, want)
        if not same:
            raise AssertionError(f"{name} differs from footsiesgym.jax.frame_data")


def render() -> str:
    kinds, slices = entries()
    check_against_jax(kinds, slices)
    T = fd.FRAME_DATA
    assert fd.NUM_ACTIONS <= 32  # cancel targets are a uint32 bitmask
    assert fd.INPUT_HISTORY <= 64  # input histories are uint64 bitmasks
    assert len(kinds["hit"]) and len(kinds["cancel"])
    hold = fd.SPECIAL_ATTACK_HOLD_FRAME
    progress = (np.arange(hold + 1, dtype=f32) / f32(hold)).astype(f32)
    assert np.array_equal(
        progress.view(np.int32), T.special_attack_progress.view(np.int32)
    )

    L = []
    w = L.append
    w(
        "// GENERATED by scripts/gen_c_frame_data.py from footsiesgym/jax/frame_data.json."
    )
    w("// Do not edit. Floats are hex literals with the exact float32 bits.")
    w("#ifndef FOOTSIES_FRAME_DATA_GEN_H")
    w("#define FOOTSIES_FRAME_DATA_GEN_H")
    w("")
    w("#include <stdint.h>")
    w("")
    w("// ── Fighter constants ──")
    w(f"#define FIXED_DELTA_TIME {hexf(fd.FIXED_DELTA_TIME)}  // 1.0f / 60.0f")
    w(f"#define BATTLE_AREA_WIDTH {hexf(fd.BATTLE_AREA_WIDTH)}")
    w(f"#define STAGE_MIN_X {hexf(-fd.BATTLE_AREA_WIDTH / 2)}")
    w(f"#define STAGE_MAX_X {hexf(fd.BATTLE_AREA_WIDTH / 2)}")
    w(f"#define MAX_SPRITE_SHAKE_FRAME {fd.MAX_SPRITE_SHAKE_FRAME}")
    w(f"#define INPUT_LEFT {fd.INPUT_LEFT}")
    w(f"#define INPUT_RIGHT {fd.INPUT_RIGHT}")
    w(f"#define INPUT_ATTACK {fd.INPUT_ATTACK}")
    w(f"#define ACTION_TYPE_GUARD {fd.ACTION_TYPE_GUARD}")
    w(f"#define START_GUARD_HEALTH {fd.START_GUARD_HEALTH}")
    w(f"#define FORWARD_MOVE_SPEED {hexf(fd.FORWARD_MOVE_SPEED)}")
    w(f"#define BACKWARD_MOVE_SPEED {hexf(fd.BACKWARD_MOVE_SPEED)}")
    w(f"#define DASH_ALLOW_FRAME {fd.DASH_ALLOW_FRAME}")
    w(f"#define SPECIAL_ATTACK_HOLD_FRAME {hold}")
    w(f"#define CAN_CANCEL_ON_WHIFF {int(fd.CAN_CANCEL_ON_WHIFF)}")
    w(f"#define INPUT_HISTORY {fd.INPUT_HISTORY}")
    w(
        f"#define NUM_FRAMES {fd.NUM_FRAMES}  // lookups clamp the action frame to [0, NUM_FRAMES-1]"
    )
    w(f"#define NUM_ACTIONS {fd.NUM_ACTIONS}")
    w(f"#define NUM_ATTACKS {len(fd.ATTACK_NAMES)}")
    w(f"#define MAX_HITBOXES {fd.MAX_HITBOXES}  // box slots per frame, in asset order")
    w(f"#define MAX_HURTBOXES {fd.MAX_HURTBOXES}")
    for name, kind in (
        ("MAX_PUSHBOXES", "push"),
        ("MAX_MOVES", "move"),
        ("MAX_CANCELS", "cancel"),
    ):
        w(f"#define {name} {max(s[kind][1] for s in slices)}  // entries per action")
    w("")
    w("// Action indices (into ACTION_IDS, which is sorted, so also one-hot order)")
    for i, name in enumerate(fd.ACTION_NAMES):
        w(f"#define ACT_{name} {i}")
    w("")
    w("// ── Env constants (footsiesgym.footsies.game.constants) ──")
    w("#define NUM_ENV_ACTIONS 9")
    for name in ("SPECIAL_CHARGE", "FORWARD_SPECIAL_CHARGE", "BACK_SPECIAL_CHARGE"):
        w(f"#define ENV_{name} {getattr(constants.EnvActions, name)}")
    for name in ("NONE", "BACK", "FORWARD"):
        w(f"#define ENV_{name} {getattr(constants.EnvActions, name)}")
    w("")
    w(
        "typedef struct { int16_t lo, hi; float x, y, w, h; uint8_t attack, proximity; } HitboxEntry;"
    )
    w("typedef struct { int16_t lo, hi; float x, y, w, h; } RectEntry;  // hurt/push")
    w("typedef struct { int16_t lo, hi; float velocity; } MoveEntry;")
    w(
        "typedef struct { int16_t lo, hi; uint32_t targets; } CancelEntry;  // bit = action index"
    )
    w("typedef struct {")
    w(
        "    uint8_t hit0, n_hit, hurt0, n_hurt, push0, n_push, move0, n_move, cancel0, n_cancel;"
    )
    w("    int16_t frame_count, loop_from;")
    w("    uint8_t is_loop, always_cancelable, type;")
    w("} ActionData;")
    w("typedef struct {")
    w("    int8_t damage_action, guard_action;  // action indices")
    w(
        "    int16_t number_of_hit, vital_damage, guard_damage, hit_stun, guard_stun, guard_break_stun;"
    )
    w("} AttackData;")
    w("")
    for kind in kinds:
        w(f"#define N_{kind.upper()}_ENTRIES {len(kinds[kind])}")
    w("")
    w("typedef struct {")
    w("    ActionData actions[NUM_ACTIONS];")
    w("    AttackData attacks[NUM_ATTACKS];")
    w("    HitboxEntry hit[N_HIT_ENTRIES];")
    w("    RectEntry hurt[N_HURT_ENTRIES];")
    w("    RectEntry push[N_PUSH_ENTRIES];")
    w("    MoveEntry move[N_MOVE_ENTRIES];")
    w("    CancelEntry cancel[N_CANCEL_ENTRIES];")
    w("} FrameTables;")
    w("")

    def action_ids():
        return ", ".join(str(int(a)) for a in fd.ACTION_IDS)

    w(f"#define ACTION_IDS_INIT {{{action_ids()}}}")
    w("")
    w("#define FRAME_TABLES_INIT { \\")
    w("  { /* actions */ \\")
    for i, (a, s) in enumerate(zip(fd._ACTIONS, slices)):
        counts = ", ".join(
            f"{s[k][0]}, {s[k][1]}" for k in ("hit", "hurt", "push", "move", "cancel")
        )
        w(
            f"    {{{counts}, {int(T.frame_count[i])}, {int(T.loop_from_frame[i])}, "
            f"{int(T.is_loop[i])}, {int(T.always_cancelable[i])}, {int(T.action_type[i])}}}, "
            f"/* {a['name']} */ \\"
        )
    w("  }, \\")
    w("  { /* attacks */ \\")
    for i, name in enumerate(fd.ATTACK_NAMES):
        vals = [
            T.attack_damage_action[i],
            T.attack_guard_action[i],
            T.attack_number_of_hit[i],
            T.attack_vital_damage[i],
            T.attack_guard_damage[i],
            T.attack_hit_stun[i],
            T.attack_guard_stun[i],
            T.attack_guard_break_stun[i],
        ]
        w(f"    {{{', '.join(str(int(v)) for v in vals)}}}, /* {name} */ \\")
    w("  }, \\")
    w("  { /* hit */ \\")
    for e in kinds["hit"]:
        w(
            f"    {{{e['lo']}, {e['hi']}, {', '.join(rect(e['rect']))}, {e['attack']}, {int(e['proximity'])}}}, \\"
        )
    w("  }, \\")
    for kind in ("hurt", "push"):
        w(f"  {{ /* {kind} */ \\")
        for e in kinds[kind]:
            w(f"    {{{e['lo']}, {e['hi']}, {', '.join(rect(e['rect']))}}}, \\")
        w("  }, \\")
    w("  { /* move */ \\")
    for e in kinds["move"]:
        w(f"    {{{e['lo']}, {e['hi']}, {hexf(e['velocity'])}}}, \\")
    w("  }, \\")
    w("  { /* cancel */ \\")
    for e in kinds["cancel"]:
        w(f"    {{{e['lo']}, {e['hi']}, 0x{e['targets']:x}u}}, \\")
    w("  }, \\")
    w("}")
    w("")
    w("// Lookups over base env actions 0-5, packed as 4-bit nibbles (entry i is")
    w("// (LUT >> 4*i) & 0xF), so device code needs no host arrays.")
    w("// Env action -> input bits: P1 faces right, P2 faces left.")
    for name in ("P1_ENV_TO_BITS", "P2_ENV_TO_BITS", "CHARGE_ACTION_LUT"):
        values = [int(v) for v in getattr(constants, name)]
        assert len(values) == 6 and all(0 <= v < 16 for v in values)
        packed = sum(v << (4 * i) for i, v in enumerate(values))
        w(f"#define {name} 0x{packed:06x}u  // {values}")
    w("")
    w("#endif  // FOOTSIES_FRAME_DATA_GEN_H")
    return "\n".join(L) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if out of date")
    args = parser.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT} is out of date; run scripts/gen_c_frame_data.py")
        print(f"{OUT} is up to date")
        return
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
