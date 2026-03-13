# Vectorized Environment Guide

This document explains how to run the FOOTSIES Unity binary in headless mode and interact with the vectorized (batched) gRPC environment for RL training.

## Running the Binary

### Headless Mode (Training)

```bash
./footsies.x86_64 -batchmode --grpc --port 50051
```

**What `-batchmode` does:**
- Runs Unity with no window, no rendering, and no GPU usage (`NullGfxDevice`)
- Uncaps the frame rate (no 60 FPS vsync limit)
- Disables all audio playback
- Skips round intro/KO/end timers — rounds transition instantly
- Skips `GC.Collect` between rounds to avoid pauses
- Disables animator and sprite updates

This is the mode you want for RL training. The game loop runs as fast as the CPU allows.

### Windowed gRPC Mode (Debugging/Watching)

```bash
./footsies.x86_64 --grpc --port 50051
```

Without `-batchmode`, the game renders normally at 60 FPS with audio and animations. Useful for watching a trained agent play via gRPC control.

### CLI Arguments

| Argument | Description |
|---|---|
| `-batchmode` | Unity built-in flag. No rendering, uncapped FPS. |
| `--grpc` or `-g` | Enable the gRPC server. |
| `--host <hostname>` | gRPC bind address (default: `0.0.0.0`). |
| `--port <port>` | gRPC listen port (default: `50051`). |

## gRPC Services

Two services run on the same port:

1. **FootsiesGameService** — Single-environment API (original). Steps one game at a time on the Unity main thread.
2. **VectorizedFootsiesService** — Batch API. Runs N independent pure-C# simulations in parallel using `Parallel.For`. No main thread dispatch needed.

For RL training, use the **VectorizedFootsiesService** exclusively. It is significantly faster because:
- Simulations are pure C# with zero Unity engine overhead
- All N environments step in parallel across CPU cores
- No rendering, no MonoBehaviour, no Transform updates
- Auto-resets on KO (no wasted frames in intro/end states)

## VectorizedFootsiesService API

### Proto Definition

The batch messages are hand-written in C# to be wire-compatible with protobuf. Use this equivalent `.proto` definition to generate Python stubs:

```protobuf
syntax = "proto3";

service VectorizedFootsiesService {
  rpc InitEnvironments(InitEnvironmentsRequest) returns (Empty);
  rpc BatchStep(BatchStepInput) returns (BatchEncodedState);
  rpc BatchReset(BatchResetInput) returns (BatchEncodedState);
  rpc BatchResetAll(Empty) returns (BatchEncodedState);
  rpc IsVecReady(Empty) returns (BoolValue);
}

message InitEnvironmentsRequest {
  int64 num_environments = 1;
}

message BatchStepInput {
  repeated int64 p1_actions = 1;
  repeated int64 p2_actions = 2;
  int64 n_frames = 3;
}

message BatchResetInput {
  repeated bool reset_mask = 1;
}

message BatchEncodedState {
  repeated float p1_encodings = 1;   // flat array: num_envs * 81 floats
  repeated float p2_encodings = 2;   // flat array: num_envs * 81 floats
  repeated int64 round_states = 3;   // one per env
  repeated bool dones = 4;           // one per env
  repeated int32 rewards = 5;        // one per env
}

// Reuse from footsies_service.proto or define:
message Empty {}
message BoolValue { bool value = 1; }
```

### Workflow

```
1. Connect to gRPC server
2. Call StartGame() on FootsiesGameService (initializes Unity scene)
3. Wait for IsReady() == true
4. Call InitEnvironments(num_environments=N)
5. Call BatchResetAll() to get initial state
6. Loop:
   a. Call BatchStep(p1_actions, p2_actions, n_frames)
   b. Read dones, rewards, encodings from response
   c. (Optional) Call BatchReset(reset_mask) for selective resets
```

**Important:** You must call `StartGame()` on the original `FootsiesGameService` first. This loads the Battle scene and initializes `BattleCore`, which provides the `FighterData` (frame data, hitboxes, etc.) needed by the vectorized environments. `InitEnvironments` reads this data from the loaded scene.

### Python Example

```python
import grpc
import footsies_service_pb2 as pb2
import footsies_service_pb2_grpc as pb2_grpc
import vectorized_service_pb2 as vec_pb2
import vectorized_service_pb2_grpc as vec_pb2_grpc
import numpy as np

channel = grpc.insecure_channel("localhost:50051")
game_stub = pb2_grpc.FootsiesGameServiceStub(channel)
vec_stub = vec_pb2_grpc.VectorizedFootsiesServiceStub(channel)

NUM_ENVS = 100
N_FRAMES = 4  # frame skip
OBS_SIZE = 81  # floats per player per env

# 1. Initialize
game_stub.StartGame(pb2.Empty())
# wait for ready...
game_stub.IsReady(pb2.Empty())

# 2. Create vectorized environments
vec_stub.InitEnvironments(vec_pb2.InitEnvironmentsRequest(num_environments=NUM_ENVS))

# 3. Reset all
state = vec_stub.BatchResetAll(vec_pb2.Empty())

# 4. Training loop
for step in range(num_steps):
    # Get actions from your policy (integers, see Action Space below)
    p1_actions = [...]  # length NUM_ENVS
    p2_actions = [...]  # length NUM_ENVS

    state = vec_stub.BatchStep(vec_pb2.BatchStepInput(
        p1_actions=p1_actions,
        p2_actions=p2_actions,
        n_frames=N_FRAMES,
    ))

    # Parse response
    p1_obs = np.array(state.p1_encodings).reshape(NUM_ENVS, OBS_SIZE)
    p2_obs = np.array(state.p2_encodings).reshape(NUM_ENVS, OBS_SIZE)
    dones = np.array(state.dones)
    rewards = np.array(state.rewards)  # +1 p1 wins, -1 p2 wins, 0 ongoing

    # Environments auto-reset on KO, but you can also manually reset:
    # vec_stub.BatchReset(vec_pb2.BatchResetInput(reset_mask=dones.tolist()))
```

## Action Space

Actions are bitmask integers using `InputDefine`:

| Bit | Value | Meaning |
|-----|-------|---------|
| 0 | 1 | Left |
| 1 | 2 | Right |
| 2 | 4 | Attack |

Combined actions:

| Action | Value | Description |
|--------|-------|-------------|
| None | 0 | Stand idle |
| Left | 1 | Move left |
| Right | 2 | Move right |
| Attack | 4 | Normal attack |
| Left + Attack | 5 | Back attack (if facing right) |
| Right + Attack | 6 | Forward attack (if facing left) |

**Direction is relative to facing:** Player 1 starts facing right, Player 2 starts facing left. "Forward" = toward opponent, "Backward" = away from opponent. The raw bitmask uses absolute Left/Right, so your agent needs to account for facing direction.

**Special attacks** are triggered by holding Attack for 60 consecutive frames, then releasing.

**Dashing** is triggered by pressing Forward twice within 10 frames.

## Observation Space

Each player's observation is 81 floats, structured as:

| Range | Size | Description |
|-------|------|-------------|
| 0 | 1 | Relative distance (normalized by 8.0) |
| 1–40 | 40 | Self state |
| 41–80 | 40 | Opponent state (delayed by 16 frames) |

**Per-player breakdown (40 floats):**

| Offset | Size | Description |
|--------|------|-------------|
| 0 | 1 | Position X (normalized) |
| 1 | 1 | Velocity X (normalized) |
| 2 | 1 | Is dead (0/1) |
| 3 | 1 | Vital health |
| 4–7 | 4 | Guard health (one-hot, 4 levels) |
| 8–24 | 17 | Current action ID (one-hot, 17 actions) |
| 25 | 1 | Action frame progress (current / total) |
| 26 | 1 | Action frame count (normalized) |
| 27 | 1 | Is action end (0/1) |
| 28 | 1 | Is always cancelable (0/1) |
| 29 | 1 | Hit count |
| 30 | 1 | Hit stun frame (normalized) |
| 31 | 1 | Is in hit stun (0/1) |
| 32 | 1 | Sprite shake position |
| 33 | 1 | Is face right (0/1) |
| 34 | 1 | Frame advantage (normalized) |
| 35 | 1 | Would next forward input dash (0/1) |
| 36 | 1 | Would next backward input dash (0/1) |
| 37 | 1 | Special attack progress (0.0–1.0) |
| 38–39 | 2 | Unused (padding) |

**Opponent observation delay:** The opponent's state is delayed by 16 frames to simulate reaction time. The agent sees its own state in real-time but the opponent's state as it was 16 frames ago.

## Rewards

| Value | Meaning |
|-------|---------|
| +1 | Player 1 wins (Player 2 died) |
| -1 | Player 2 wins (Player 1 died) |
| 0 | Round ongoing, or double KO (draw) |

Rewards are only non-zero on the frame where `done=true`.

## Round States

| Value | State | Description |
|-------|-------|-------------|
| 0 | Stop | Not started |
| 1 | Intro | Intro sequence (skipped in vectorized sim) |
| 2 | Fight | Active gameplay |
| 3 | KO | A player died |
| 4 | End | Round ended |

In the vectorized simulation, rounds go directly from Fight to KO. The auto-reset in `Step()` brings KO back to Fight on the next call.

## Architecture Notes

- **BattleSimulation** is a pure C# class that mirrors `BattleCore` fight logic without any Unity dependencies (no MonoBehaviour, Transform, Physics, rendering, or audio).
- **VectorizedEnvironmentManager** holds N `BattleSimulation` instances and steps them in parallel via `Parallel.For`.
- **FighterData** (a Unity ScriptableObject) is shared read-only across all simulations. It contains frame data, hitbox/hurtbox definitions, and action state machines.
- Output buffers (round states, dones, rewards) are pre-allocated once at init to avoid per-step GC pressure.
- The vectorized path runs entirely on the gRPC thread pool — no `UnityMainThreadDispatcher` marshaling needed, unlike the single-environment API.
