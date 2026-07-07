"""
Smoke test for the VectorizedFootsiesService batch gRPC API.

Tests both raw state and encoded state endpoints:
  1. StartGame (needed to load FighterData)
  2. InitEnvironments
  3. Raw mode: BatchResetAll / BatchStep / BatchReset
  4. Encoded mode: BatchResetAllEncoded / BatchStepEncoded / BatchResetEncoded
  5. Throughput measurement for both modes

Usage:
    python -m scripts.grpc_batch_test [--host HOST] [--port PORT] [--num-envs N] [--steps S]
"""

import argparse
import random
import time

import grpc

from footsiesgym.footsies.game.proto import footsies_service_pb2 as pb2
from footsiesgym.footsies.game.proto import footsies_service_pb2_grpc as pb2_grpc

NUM_ACTIONS = (
    7  # NONE, BACK, FORWARD, ATTACK, BACK_ATTACK, FORWARD_ATTACK, SPECIAL_CHARGE
)
GAME_ACTIONS = [0, 1, 2, 4]  # Subset for random play: NONE, LEFT, RIGHT, ATTACK


def wait_for_ready(game_stub, vec_stub, timeout=10):
    for _ in range(int(timeout / 0.5)):
        if game_stub.IsReady(pb2.Empty()).value:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("Game never became ready")

    for _ in range(int(timeout / 0.5)):
        if vec_stub.IsVecReady(pb2.Empty()).value:
            return
        time.sleep(0.5)
    raise RuntimeError("Vectorized envs never became ready")


def test_raw_mode(vec_stub, num_envs, num_steps, n_frames):
    """Test raw state endpoints (BatchStep/BatchReset/BatchResetAll -> BatchRawState)."""
    print("\n=== RAW STATE MODE ===")

    # --- BatchResetAll ---
    print("  BatchResetAll...")
    state = vec_stub.BatchResetAll(pb2.Empty())
    assert len(state.round_states) == num_envs, "round_states length: %d" % len(
        state.round_states
    )
    assert len(state.dones) == num_envs
    assert len(state.rewards) == num_envs
    assert len(state.frame_counts) == num_envs
    # Check per-field raw state arrays
    assert len(state.p1_position_x) == num_envs, "p1_position_x length: %d" % len(
        state.p1_position_x
    )
    assert len(state.p2_position_x) == num_envs
    assert len(state.p1_current_action_id) == num_envs
    assert len(state.p1_guard_health) == num_envs
    assert not any(state.dones), "Expected no dones after reset"
    print("  BatchResetAll OK (%d envs, all raw fields present)" % num_envs)

    # --- BatchStep loop ---
    print("  Running %d BatchStep calls (n_frames=%d)..." % (num_steps, n_frames))
    total_done = 0
    start_time = time.perf_counter()

    for step in range(num_steps):
        p1_actions = [random.choice(GAME_ACTIONS) for _ in range(num_envs)]
        p2_actions = [random.choice(GAME_ACTIONS) for _ in range(num_envs)]

        state = vec_stub.BatchStep(
            pb2.BatchStepInput(
                p1_actions=p1_actions,
                p2_actions=p2_actions,
                n_frames=n_frames,
            )
        )

        done_count = sum(state.dones)
        total_done += done_count

        # Reset done environments
        if done_count > 0:
            reset_mask = list(state.dones)
            state = vec_stub.BatchReset(pb2.BatchResetInput(reset_mask=reset_mask))
            still_done = sum(d for d, m in zip(state.dones, reset_mask) if m)
            if still_done > 0:
                print(
                    "  WARNING: %d envs still done after reset at step %d"
                    % (still_done, step)
                )

        if (step + 1) % max(1, num_steps // 5) == 0:
            elapsed = time.perf_counter() - start_time
            fps = (step + 1) * num_envs * n_frames / elapsed
            print(
                "    Step %d/%d | episodes: %d | %s sim frames/sec"
                % (step + 1, num_steps, total_done, "{:,.0f}".format(fps))
            )

    elapsed = time.perf_counter() - start_time
    total_frames = num_steps * num_envs * n_frames
    fps = total_frames / elapsed
    print(
        "  Raw mode: %s sim frames/sec | %.2fms/step | %d episodes"
        % ("{:,.0f}".format(fps), elapsed / num_steps * 1000, total_done)
    )
    return fps


def test_encoded_mode(vec_stub, num_envs, num_steps, n_frames):
    """Test encoded state endpoints (BatchStepEncoded/etc -> BatchEncodedState)."""
    print("\n=== ENCODED STATE MODE ===")

    # Compute expected obs size: 1 + (37 + 4 + NUM_ACTIONS) + 37 = 79 + NUM_ACTIONS
    obs_size = 1 + (37 + 4 + NUM_ACTIONS) + 37  # 86 with NUM_ACTIONS=7

    # --- BatchResetAllEncoded ---
    print("  BatchResetAllEncoded...")
    state = vec_stub.BatchResetAllEncoded(
        pb2.BatchResetAllEncodedInput(
            num_actions=NUM_ACTIONS,
        )
    )
    assert (
        len(state.p1_encodings) == num_envs * obs_size
    ), "p1_encodings length: %d, expected %d" % (
        len(state.p1_encodings),
        num_envs * obs_size,
    )
    assert (
        len(state.p2_encodings) == num_envs * obs_size
    ), "p2_encodings length: %d, expected %d" % (
        len(state.p2_encodings),
        num_envs * obs_size,
    )
    assert len(state.round_states) == num_envs
    assert len(state.dones) == num_envs
    assert len(state.rewards) == num_envs
    assert not any(state.dones), "Expected no dones after reset"
    print("  BatchResetAllEncoded OK (%d envs x %d obs_size)" % (num_envs, obs_size))

    # --- BatchStepEncoded loop ---
    print(
        "  Running %d BatchStepEncoded calls (n_frames=%d)..." % (num_steps, n_frames)
    )
    total_done = 0

    # Track prev_actions and holding_special per env (mimic Python env state)
    prev_p1 = [0] * num_envs  # NONE
    prev_p2 = [0] * num_envs
    p1_holding = [False] * num_envs
    p2_holding = [False] * num_envs

    start_time = time.perf_counter()

    for step in range(num_steps):
        p1_actions = [random.choice(GAME_ACTIONS) for _ in range(num_envs)]
        p2_actions = [random.choice(GAME_ACTIONS) for _ in range(num_envs)]

        state = vec_stub.BatchStepEncoded(
            pb2.BatchStepEncodedInput(
                p1_actions=p1_actions,
                p2_actions=p2_actions,
                n_frames=n_frames,
                prev_p1_actions=prev_p1,
                prev_p2_actions=prev_p2,
                p1_holding_special=p1_holding,
                p2_holding_special=p2_holding,
                num_actions=NUM_ACTIONS,
            )
        )

        assert len(state.p1_encodings) == num_envs * obs_size
        assert len(state.p2_encodings) == num_envs * obs_size

        done_count = sum(state.dones)
        total_done += done_count

        # Update prev_actions for next step
        prev_p1 = p1_actions
        prev_p2 = p2_actions

        # Reset done environments
        if done_count > 0:
            reset_mask = list(state.dones)
            state = vec_stub.BatchResetEncoded(
                pb2.BatchResetEncodedInput(
                    reset_mask=reset_mask,
                    num_actions=NUM_ACTIONS,
                )
            )
            # Clear prev_actions for reset envs
            for i, m in enumerate(reset_mask):
                if m:
                    prev_p1[i] = 0
                    prev_p2[i] = 0
                    p1_holding[i] = False
                    p2_holding[i] = False

        if (step + 1) % max(1, num_steps // 5) == 0:
            elapsed = time.perf_counter() - start_time
            fps = (step + 1) * num_envs * n_frames / elapsed
            print(
                "    Step %d/%d | episodes: %d | %s sim frames/sec"
                % (step + 1, num_steps, total_done, "{:,.0f}".format(fps))
            )

    elapsed = time.perf_counter() - start_time
    total_frames = num_steps * num_envs * n_frames
    fps = total_frames / elapsed
    print(
        "  Encoded mode: %s sim frames/sec | %.2fms/step | %d episodes"
        % ("{:,.0f}".format(fps), elapsed / num_steps * 1000, total_done)
    )
    return fps


def run(host: str, port: int, num_envs: int, num_steps: int, n_frames: int):
    channel = grpc.insecure_channel("%s:%d" % (host, port))

    game_stub = pb2_grpc.FootsiesGameServiceStub(channel)
    vec_stub = pb2_grpc.VectorizedFootsiesServiceStub(channel)

    # --- 1. StartGame ---
    print("1. Calling StartGame...")
    game_stub.StartGame(pb2.Empty())
    print("   StartGame OK")

    # --- 2. Wait for ready ---
    print("2. Waiting for game ready...")
    wait_for_ready(game_stub, vec_stub, timeout=10)
    print("   Game is ready")

    # --- 3. InitEnvironments ---
    print("3. Calling InitEnvironments(num_environments=%d)..." % num_envs)
    vec_stub.InitEnvironments(pb2.InitEnvironmentsRequest(num_environments=num_envs))
    for _ in range(20):
        if vec_stub.IsVecReady(pb2.Empty()).value:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("Vectorized envs never became ready")
    print("   InitEnvironments OK")

    # --- 4. Test raw mode ---
    raw_fps = test_raw_mode(vec_stub, num_envs, num_steps, n_frames)

    # --- 5. Test encoded mode ---
    enc_fps = test_encoded_mode(vec_stub, num_envs, num_steps, n_frames)

    # --- Summary ---
    print("\n" + "=" * 50)
    print(
        "SUMMARY (%d envs, %d steps, %d frames/step)" % (num_envs, num_steps, n_frames)
    )
    print("  Raw mode:     %12s sim frames/sec" % "{:,.0f}".format(raw_fps))
    print("  Encoded mode: %12s sim frames/sec" % "{:,.0f}".format(enc_fps))
    if raw_fps > 0:
        ratio = enc_fps / raw_fps
        print("  Ratio:        %.2fx" % ratio)
    print("\nAll tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch gRPC smoke test for FOOTSIES")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--n-frames", type=int, default=4)
    args = parser.parse_args()

    run(args.host, args.port, args.num_envs, args.steps, args.n_frames)
