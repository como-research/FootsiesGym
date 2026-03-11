"""
Smoke test for the VectorizedFootsiesService batch gRPC API.

Launches against a running headless FOOTSIES server and tests:
  1. StartGame (needed to load FighterData)
  2. InitEnvironments
  3. BatchResetAll
  4. BatchStep loop with random actions
  5. BatchReset for done environments
  6. Throughput measurement

Usage:
    python -m scripts.grpc_batch_test [--host HOST] [--port PORT] [--num-envs N] [--steps S]
"""

import argparse
import random
import time

import grpc

from footsiesgym.footsies.game.proto import footsies_service_pb2 as pb2
from footsiesgym.footsies.game.proto import footsies_service_pb2_grpc as pb2_grpc


def run(host: str, port: int, num_envs: int, num_steps: int, n_frames: int):
    channel = grpc.insecure_channel(f"{host}:{port}")

    game_stub = pb2_grpc.FootsiesGameServiceStub(channel)
    vec_stub = pb2_grpc.VectorizedFootsiesServiceStub(channel)

    # --- 1. StartGame (loads FighterData on the Unity side) ---
    print("1. Calling StartGame...")
    game_stub.StartGame(pb2.Empty())
    print("   StartGame OK")

    # Wait for game to be ready
    for _ in range(20):
        if game_stub.IsReady(pb2.Empty()).value:
            break
        time.sleep(0.5)
    else:
        print("   ERROR: Game never became ready")
        return
    print("   Game is ready")

    # --- 2. InitEnvironments ---
    print(f"2. Calling InitEnvironments(num_environments={num_envs})...")
    vec_stub.InitEnvironments(pb2.InitEnvironmentsRequest(num_environments=num_envs))
    # Wait for vec environments to initialize
    for _ in range(20):
        if vec_stub.IsVecReady(pb2.Empty()).value:
            break
        time.sleep(0.5)
    else:
        print("   ERROR: Vectorized envs never became ready")
        return
    print("   InitEnvironments OK")

    # --- 3. BatchResetAll ---
    print("3. Calling BatchResetAll...")
    state = vec_stub.BatchResetAll(pb2.Empty())
    obs_size = len(state.p1_encodings) // num_envs
    print(f"   BatchResetAll OK")
    print(f"   p1_encodings length: {len(state.p1_encodings)} ({num_envs} envs x {obs_size} features)")
    print(f"   p2_encodings length: {len(state.p2_encodings)}")
    print(f"   round_states length: {len(state.round_states)}")
    print(f"   dones length:        {len(state.dones)}")
    print(f"   rewards length:      {len(state.rewards)}")

    # Validate shapes
    assert len(state.p1_encodings) == num_envs * obs_size, f"p1_encodings shape mismatch"
    assert len(state.p2_encodings) == num_envs * obs_size, f"p2_encodings shape mismatch"
    assert len(state.round_states) == num_envs, f"round_states shape mismatch"
    assert len(state.dones) == num_envs, f"dones shape mismatch"
    assert len(state.rewards) == num_envs, f"rewards shape mismatch"
    assert not any(state.dones), f"Expected no dones after reset, got {sum(state.dones)}"
    print("   All shape assertions passed")

    # --- 4. BatchStep loop ---
    print(f"4. Running {num_steps} BatchStep calls (n_frames={n_frames}, {num_envs} envs)...")
    total_done = 0
    total_resets = 0
    actions = [0, 1, 2, 4]  # NONE, LEFT, RIGHT, ATTACK

    start_time = time.perf_counter()

    for step in range(num_steps):
        # Random actions for all envs
        p1_actions = [random.choice(actions) for _ in range(num_envs)]
        p2_actions = [random.choice(actions) for _ in range(num_envs)]

        state = vec_stub.BatchStep(pb2.BatchStepInput(
            p1_actions=p1_actions,
            p2_actions=p2_actions,
            n_frames=n_frames,
        ))

        done_count = sum(state.dones)
        total_done += done_count

        # Reset done environments
        if done_count > 0:
            total_resets += done_count
            reset_mask = list(state.dones)
            state = vec_stub.BatchReset(pb2.BatchResetInput(reset_mask=reset_mask))
            # Verify reset envs are no longer done
            still_done = sum(state.dones)
            if still_done > 0:
                print(f"   WARNING: {still_done} envs still done after reset at step {step}")

        # Progress update
        if (step + 1) % max(1, num_steps // 5) == 0:
            elapsed = time.perf_counter() - start_time
            fps = (step + 1) * num_envs * n_frames / elapsed
            print(f"   Step {step + 1}/{num_steps} | episodes completed: {total_done} | {fps:,.0f} sim frames/sec")

    elapsed = time.perf_counter() - start_time
    total_frames = num_steps * num_envs * n_frames
    fps = total_frames / elapsed

    print(f"\n--- Results ---")
    print(f"   Environments:      {num_envs}")
    print(f"   Steps:             {num_steps}")
    print(f"   Frames per step:   {n_frames}")
    print(f"   Total sim frames:  {total_frames:,}")
    print(f"   Episodes completed:{total_done}")
    print(f"   Wall time:         {elapsed:.2f}s")
    print(f"   Throughput:        {fps:,.0f} sim frames/sec")
    print(f"   Per-step latency:  {elapsed / num_steps * 1000:.2f}ms")
    print(f"\nAll tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch gRPC smoke test for FOOTSIES")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--n-frames", type=int, default=4)
    args = parser.parse_args()

    run(args.host, args.port, args.num_envs, args.steps, args.n_frames)
