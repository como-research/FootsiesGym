"""
Validation test: verify Python VectorizedEncoder produces output
identical to the C# VectorizedEncoder running server-side.

Workflow:
  1. Connect to gRPC server, start game, init vectorized envs.
  2. ResetAll (raw) -> Python-encode -> save as reference.
  3. ResetAll (encoded via C#) -> compare with Python encoding.
  4. Step with identical actions through both paths and compare at
     each step.

Because BatchResetAll resets to the same deterministic initial state,
we can compare the two paths reliably. For BatchStep, we reset,
then alternate between raw and encoded steps on the SAME envs
(resetting between comparisons so state is identical).

Usage:
    python -m scripts.validate_encoding [--host HOST] [--port PORT]
        [--num-envs N] [--steps S]
"""

import argparse
import random
import time

import grpc
import numpy as np

from footsiesgym.footsies.encoder import VectorizedEncoder
from footsiesgym.footsies.game.proto import footsies_service_pb2 as pb2
from footsiesgym.footsies.game.proto import (
    footsies_service_pb2_grpc as pb2_grpc,
)

NUM_ACTIONS = 7
GAME_ACTIONS = [0, 1, 2, 4]  # NONE, LEFT, RIGHT, ATTACK


def wait_for_ready(game_stub, timeout=10):
    for _ in range(int(timeout / 0.5)):
        if game_stub.IsReady(pb2.Empty()).value:
            return
        time.sleep(0.5)
    raise RuntimeError("Game never became ready")


def compare_arrays(
    name: str,
    python_arr: np.ndarray,
    csharp_arr: np.ndarray,
    atol: float = 1e-5,
) -> bool:
    """Compare two arrays and print diagnostics on mismatch."""
    if python_arr.shape != csharp_arr.shape:
        print(
            "  FAIL %s: shape mismatch %s vs %s"
            % (name, python_arr.shape, csharp_arr.shape)
        )
        return False

    diffs = np.abs(python_arr - csharp_arr)
    close = np.allclose(python_arr, csharp_arr, atol=atol)
    if not close:
        max_diff = diffs.max()
        max_idx = np.unravel_index(diffs.argmax(), diffs.shape)
        n_mismatched = (diffs > atol).sum()
        print(
            "  FAIL %s: %d mismatched values, max diff=%.6f at %s"
            % (name, n_mismatched, max_diff, max_idx)
        )
        print(
            "    Python[%s] = %.6f, C#[%s] = %.6f"
            % (max_idx, python_arr[max_idx], max_idx, csharp_arr[max_idx])
        )
        # Print the full row with the worst mismatch
        env_idx = max_idx[0]
        print("    Python row %d:" % env_idx)
        print("    ", python_arr[env_idx])
        print("    C# row %d:" % env_idx)
        print("    ", csharp_arr[env_idx])
        return False

    print("  OK   %s: max diff=%.2e" % (name, diffs.max()))
    return True


def validate_reset(vec_stub, encoder, num_envs):
    """Validate that Python encoding matches C# on reset."""
    print("\n=== VALIDATE RESET ===")

    # 1. C# encoded reset
    csharp_state = vec_stub.BatchResetAllEncoded(
        pb2.BatchResetAllEncodedInput(num_actions=NUM_ACTIONS)
    )
    csharp_p1 = np.array(csharp_state.p1_encodings).reshape(
        num_envs, encoder.obs_size
    )
    csharp_p2 = np.array(csharp_state.p2_encodings).reshape(
        num_envs, encoder.obs_size
    )

    # 2. Raw reset (resets to same deterministic initial state)
    raw_state = vec_stub.BatchResetAll(pb2.Empty())

    # 3. Python encode
    prev_p1 = np.zeros(num_envs, dtype=np.int64)
    prev_p2 = np.zeros(num_envs, dtype=np.int64)
    hold_p1 = np.zeros(num_envs, dtype=bool)
    hold_p2 = np.zeros(num_envs, dtype=bool)

    py_result = encoder.encode(raw_state, prev_p1, prev_p2, hold_p1, hold_p2)

    ok = True
    ok &= compare_arrays("reset p1", py_result["p1"], csharp_p1)
    ok &= compare_arrays("reset p2", py_result["p2"], csharp_p2)
    return ok


def validate_steps(vec_stub, encoder, num_envs, num_steps, n_frames):
    """Validate encoding over multiple steps.

    Strategy: Use a single set of envs. For each step:
      1. ResetAll to sync state
      2. Step with encoded (C#) -> save C# encoding
      3. ResetAll again to sync
      4. Step with raw -> Python encode -> compare with C#
    """
    print(
        "\n=== VALIDATE STEPS (%d steps, n_frames=%d) ==="
        % (num_steps, n_frames)
    )

    all_ok = True
    rng = random.Random(42)

    for step in range(num_steps):
        # Generate deterministic actions
        p1_actions = [rng.choice(GAME_ACTIONS) for _ in range(num_envs)]
        p2_actions = [rng.choice(GAME_ACTIONS) for _ in range(num_envs)]
        prev_p1 = [rng.randint(0, NUM_ACTIONS - 1) for _ in range(num_envs)]
        prev_p2 = [rng.randint(0, NUM_ACTIONS - 1) for _ in range(num_envs)]
        hold_p1 = [rng.choice([True, False]) for _ in range(num_envs)]
        hold_p2 = [rng.choice([True, False]) for _ in range(num_envs)]

        # --- Path A: C# encoded ---
        vec_stub.BatchResetAll(pb2.Empty())
        csharp_state = vec_stub.BatchStepEncoded(
            pb2.BatchStepEncodedInput(
                p1_actions=p1_actions,
                p2_actions=p2_actions,
                n_frames=n_frames,
                prev_p1_actions=prev_p1,
                prev_p2_actions=prev_p2,
                p1_holding_special=hold_p1,
                p2_holding_special=hold_p2,
                num_actions=NUM_ACTIONS,
            )
        )
        csharp_p1 = np.array(csharp_state.p1_encodings).reshape(
            num_envs, encoder.obs_size
        )
        csharp_p2 = np.array(csharp_state.p2_encodings).reshape(
            num_envs, encoder.obs_size
        )

        # --- Path B: Raw + Python encode ---
        vec_stub.BatchResetAll(pb2.Empty())
        raw_state = vec_stub.BatchStep(
            pb2.BatchStepInput(
                p1_actions=p1_actions,
                p2_actions=p2_actions,
                n_frames=n_frames,
            )
        )
        py_result = encoder.encode(
            raw_state,
            np.array(prev_p1, dtype=np.int64),
            np.array(prev_p2, dtype=np.int64),
            np.array(hold_p1, dtype=bool),
            np.array(hold_p2, dtype=bool),
        )

        ok_p1 = compare_arrays("step %d p1" % step, py_result["p1"], csharp_p1)
        ok_p2 = compare_arrays("step %d p2" % step, py_result["p2"], csharp_p2)
        all_ok &= ok_p1 and ok_p2

        if not (ok_p1 and ok_p2):
            print("  Stopping early on first mismatch.")
            break

    return all_ok


def validate_sequence(vec_stub, encoder, num_envs, num_steps, n_frames):
    """Validate Python vs C# encoding over a contiguous gameplay sequence.

    Single-pass approach: step with BatchStep (raw), then call
    GetBatchEncodedState to get C# encoding of the exact same state.
    Python-encode the raw result and compare with C#.

    This tests encoding across varied game states (mid-fight,
    near-KO, post-auto-reset) without any state divergence issues.
    """
    print(
        "\n=== VALIDATE SEQUENCE (%d steps, n_frames=%d) ==="
        % (num_steps, n_frames)
    )

    rng = random.Random(99)
    vec_stub.BatchResetAll(pb2.Empty())

    all_ok = True
    for step in range(num_steps):
        p1_actions = [rng.choice(GAME_ACTIONS) for _ in range(num_envs)]
        p2_actions = [rng.choice(GAME_ACTIONS) for _ in range(num_envs)]
        prev_p1 = [rng.randint(0, NUM_ACTIONS - 1) for _ in range(num_envs)]
        prev_p2 = [rng.randint(0, NUM_ACTIONS - 1) for _ in range(num_envs)]
        hold_p1 = [rng.choice([True, False]) for _ in range(num_envs)]
        hold_p2 = [rng.choice([True, False]) for _ in range(num_envs)]

        # Step with raw endpoint
        raw_state = vec_stub.BatchStep(
            pb2.BatchStepInput(
                p1_actions=p1_actions,
                p2_actions=p2_actions,
                n_frames=n_frames,
            )
        )

        # Python-encode the raw state
        py_result = encoder.encode(
            raw_state,
            np.array(prev_p1, dtype=np.int64),
            np.array(prev_p2, dtype=np.int64),
            np.array(hold_p1, dtype=bool),
            np.array(hold_p2, dtype=bool),
        )

        # Get C# encoding of the same state (no step, just encode)
        cs_state = vec_stub.GetBatchEncodedState(
            pb2.GetBatchEncodedStateInput(
                prev_p1_actions=prev_p1,
                prev_p2_actions=prev_p2,
                p1_holding_special=hold_p1,
                p2_holding_special=hold_p2,
                num_actions=NUM_ACTIONS,
            )
        )
        cs_p1 = np.array(cs_state.p1_encodings).reshape(
            num_envs, encoder.obs_size
        )
        cs_p2 = np.array(cs_state.p2_encodings).reshape(
            num_envs, encoder.obs_size
        )

        ok_p1 = compare_arrays("seq step %d p1" % step, py_result["p1"], cs_p1)
        ok_p2 = compare_arrays("seq step %d p2" % step, py_result["p2"], cs_p2)
        all_ok &= ok_p1 and ok_p2
        if not (ok_p1 and ok_p2):
            print("  Stopping early on first mismatch.")
            break

    return all_ok


def validate_consistency_with_scalar(vec_stub, encoder, num_envs):
    """Validate VectorizedEncoder matches FootsiesEncoder on raw data.

    This doesn't need a server for encoding - it just checks
    that VectorizedEncoder and FootsiesEncoder produce the same
    output when given the same raw state (encoded Python-side).
    """
    from footsiesgym.footsies.encoder import FootsiesEncoder

    print("\n=== VALIDATE PYTHON VECTORIZED vs SCALAR ===")

    scalar_enc = FootsiesEncoder()

    # Get a raw state from the server
    raw = vec_stub.BatchResetAll(pb2.Empty())

    prev_p1 = np.array(
        [random.randint(0, NUM_ACTIONS - 1) for _ in range(num_envs)],
        dtype=np.int64,
    )
    prev_p2 = np.array(
        [random.randint(0, NUM_ACTIONS - 1) for _ in range(num_envs)],
        dtype=np.int64,
    )
    hold_p1 = np.array(
        [random.choice([True, False]) for _ in range(num_envs)],
        dtype=bool,
    )
    hold_p2 = np.array(
        [random.choice([True, False]) for _ in range(num_envs)],
        dtype=bool,
    )

    # Vectorized encode
    vec_result = encoder.encode(raw, prev_p1, prev_p2, hold_p1, hold_p2)

    # Scalar encode each env independently
    scalar_p1 = np.zeros((num_envs, encoder.obs_size), dtype=np.float32)
    scalar_p2 = np.zeros((num_envs, encoder.obs_size), dtype=np.float32)

    for i in range(num_envs):
        # Build a single-env GameState-like object from raw fields
        game_state = _build_game_state_from_raw(raw, i)
        result = scalar_enc.encode(
            game_state,
            prev_actions={
                "p1": int(prev_p1[i]),
                "p2": int(prev_p2[i]),
            },
            is_charging_special={
                "p1": bool(hold_p1[i]),
                "p2": bool(hold_p2[i]),
            },
            num_actions=NUM_ACTIONS,
        )
        scalar_p1[i] = result["p1"]
        scalar_p2[i] = result["p2"]
        scalar_enc.reset()

    ok = True
    ok &= compare_arrays("vec vs scalar p1", vec_result["p1"], scalar_p1)
    ok &= compare_arrays("vec vs scalar p2", vec_result["p2"], scalar_p2)
    return ok


def _build_game_state_from_raw(raw, idx):
    """Construct a GameState protobuf from BatchRawState for env idx."""
    gs = pb2.GameState()

    gs.player1.player_position_x = raw.p1_position_x[idx]
    gs.player1.velocity_x = raw.p1_velocity_x[idx]
    gs.player1.is_dead = raw.p1_is_dead[idx]
    gs.player1.vital_health = raw.p1_vital_health[idx]
    gs.player1.guard_health = raw.p1_guard_health[idx]
    gs.player1.current_action_id = raw.p1_current_action_id[idx]
    gs.player1.current_action_frame = raw.p1_current_action_frame[idx]
    gs.player1.current_action_frame_count = raw.p1_current_action_frame_count[
        idx
    ]
    gs.player1.is_action_end = raw.p1_is_action_end[idx]
    gs.player1.is_always_cancelable = raw.p1_is_always_cancelable[idx]
    gs.player1.current_action_hit_count = raw.p1_current_action_hit_count[idx]
    gs.player1.current_hit_stun_frame = raw.p1_current_hit_stun_frame[idx]
    gs.player1.is_in_hit_stun = raw.p1_is_in_hit_stun[idx]
    gs.player1.sprite_shake_position = raw.p1_sprite_shake_position[idx]
    gs.player1.max_sprite_shake_frame = raw.p1_max_sprite_shake_frame[idx]
    gs.player1.is_face_right = raw.p1_is_face_right[idx]
    gs.player1.current_frame_advantage = raw.p1_current_frame_advantage[idx]
    gs.player1.would_next_forward_input_dash = (
        raw.p1_would_next_forward_input_dash[idx]
    )
    gs.player1.would_next_backward_input_dash = (
        raw.p1_would_next_backward_input_dash[idx]
    )
    gs.player1.special_attack_progress = raw.p1_special_attack_progress[idx]

    gs.player2.player_position_x = raw.p2_position_x[idx]
    gs.player2.velocity_x = raw.p2_velocity_x[idx]
    gs.player2.is_dead = raw.p2_is_dead[idx]
    gs.player2.vital_health = raw.p2_vital_health[idx]
    gs.player2.guard_health = raw.p2_guard_health[idx]
    gs.player2.current_action_id = raw.p2_current_action_id[idx]
    gs.player2.current_action_frame = raw.p2_current_action_frame[idx]
    gs.player2.current_action_frame_count = raw.p2_current_action_frame_count[
        idx
    ]
    gs.player2.is_action_end = raw.p2_is_action_end[idx]
    gs.player2.is_always_cancelable = raw.p2_is_always_cancelable[idx]
    gs.player2.current_action_hit_count = raw.p2_current_action_hit_count[idx]
    gs.player2.current_hit_stun_frame = raw.p2_current_hit_stun_frame[idx]
    gs.player2.is_in_hit_stun = raw.p2_is_in_hit_stun[idx]
    gs.player2.sprite_shake_position = raw.p2_sprite_shake_position[idx]
    gs.player2.max_sprite_shake_frame = raw.p2_max_sprite_shake_frame[idx]
    gs.player2.is_face_right = raw.p2_is_face_right[idx]
    gs.player2.current_frame_advantage = raw.p2_current_frame_advantage[idx]
    gs.player2.would_next_forward_input_dash = (
        raw.p2_would_next_forward_input_dash[idx]
    )
    gs.player2.would_next_backward_input_dash = (
        raw.p2_would_next_backward_input_dash[idx]
    )
    gs.player2.special_attack_progress = raw.p2_special_attack_progress[idx]

    gs.round_state = raw.round_states[idx]
    gs.frame_count = raw.frame_counts[idx]

    return gs


def run(host, port, num_envs, num_steps, n_frames):
    channel = grpc.insecure_channel("%s:%d" % (host, port))
    game_stub = pb2_grpc.FootsiesGameServiceStub(channel)
    vec_stub = pb2_grpc.VectorizedFootsiesServiceStub(channel)

    print("1. Starting game...")
    game_stub.StartGame(pb2.Empty())

    print("2. Waiting for ready...")
    wait_for_ready(game_stub, timeout=10)

    print("3. Initializing %d environments..." % num_envs)
    vec_stub.InitEnvironments(
        pb2.InitEnvironmentsRequest(num_environments=num_envs)
    )
    for _ in range(20):
        if vec_stub.IsVecReady(pb2.Empty()).value:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("Vectorized envs never became ready")

    encoder = VectorizedEncoder(num_actions=NUM_ACTIONS)
    all_ok = True

    # Test 1: Python vectorized vs scalar encoder (no C# needed)
    all_ok &= validate_consistency_with_scalar(vec_stub, encoder, num_envs)

    # Test 2: Python vs C# on reset
    all_ok &= validate_reset(vec_stub, encoder, num_envs)

    # Test 3: Python vs C# over individual steps (from initial state)
    all_ok &= validate_steps(vec_stub, encoder, num_envs, num_steps, n_frames)

    # Test 4: Python vs C# over a contiguous gameplay sequence
    all_ok &= validate_sequence(
        vec_stub, encoder, num_envs, num_steps, n_frames
    )

    print("\n" + "=" * 50)
    if all_ok:
        print("ALL VALIDATION TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Python vs C# encoding"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--n-frames", type=int, default=4)
    args = parser.parse_args()

    run(args.host, args.port, args.num_envs, args.steps, args.n_frames)
