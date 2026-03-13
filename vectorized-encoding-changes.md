# Vectorized Encoding Changes Summary

## Overview

Added C#-side encoding support to the vectorized gRPC API. The system now offers **two modes**:

1. **Raw mode** — Returns per-field game state arrays. Python encodes.
2. **Encoded mode** — Returns flat pre-encoded observation arrays matching the Python `FootsiesEncoder` exactly. C# encodes inside `Parallel.For`.

Observation delay has been removed entirely from `AIEncoder`.

---

## Proto API Changes (`footsies_service.proto`)

### Renamed Messages
- `BatchEncodedState` (old, 44 per-field arrays) → **`BatchRawState`** (same wire format, fields 3–46)

### New Messages

**`BatchEncodedState`** (new) — flat encoded observations:
```protobuf
message BatchEncodedState {
    repeated float p1_encodings = 1;  // flat: num_envs * obs_size
    repeated float p2_encodings = 2;  // flat: num_envs * obs_size
    repeated int64 round_states = 3;
    repeated bool dones = 4;
    repeated int32 rewards = 5;
}
```

**`BatchStepEncodedInput`** — step request with Python-side encoding context:
```protobuf
message BatchStepEncodedInput {
    repeated int64 p1_actions = 1;       // actions to execute
    repeated int64 p2_actions = 2;
    int64 n_frames = 3;
    repeated int64 prev_p1_actions = 4;  // selected action from previous step
    repeated int64 prev_p2_actions = 5;
    repeated bool p1_holding_special = 6; // special charge toggle state
    repeated bool p2_holding_special = 7;
    int64 num_actions = 8;               // one-hot size for previous_action
}
```

**`BatchResetEncodedInput`**:
```protobuf
message BatchResetEncodedInput {
    repeated bool reset_mask = 1;
    int64 num_actions = 2;
}
```

**`BatchResetAllEncodedInput`**:
```protobuf
message BatchResetAllEncodedInput {
    int64 num_actions = 1;
}
```

### New RPC Endpoints on `VectorizedFootsiesService`

| Endpoint | Request | Response | Description |
|---|---|---|---|
| `BatchStep` | `BatchStepInput` | `BatchRawState` | Step, return raw state (unchanged) |
| `BatchReset` | `BatchResetInput` | `BatchRawState` | Reset, return raw state (unchanged) |
| `BatchResetAll` | `Empty` | `BatchRawState` | Reset all, return raw state (unchanged) |
| **`BatchStepEncoded`** | `BatchStepEncodedInput` | `BatchEncodedState` | Step, return encoded obs |
| **`BatchResetEncoded`** | `BatchResetEncodedInput` | `BatchEncodedState` | Reset, return encoded obs |
| **`BatchResetAllEncoded`** | `BatchResetAllEncodedInput` | `BatchEncodedState` | Reset all, return encoded obs |

---

## Observation Encoding Specification

The C# `VectorizedEncoder` matches Python `FootsiesEncoder.encode()` exactly.

### Per-Player Observation Layout
```
[common (1)] + [self_full (41 + num_actions)] + [opponent_well_known (37)]
```

With `num_actions=7`: **obs_size = 86**

### Feature Breakdown

**Common (1 float):**
- `abs(p1.x - p2.x) / 8.0` — normalized distance

**Well-known features per player (37 floats):**
These appear in BOTH self and opponent observations.

| # | Feature | Size | Normalization |
|---|---|---|---|
| 1 | position_x | 1 | `/ 4.0` |
| 2 | velocity_x | 1 | `/ 5.0` |
| 3 | is_dead | 1 | bool → 0/1 |
| 4 | vital_health | 1 | raw float |
| 5 | guard_health | 4 | one-hot [0,1,2,3] |
| 6 | current_action_id | 17 | one-hot over ACTION_ID_VALUES |
| 7 | current_action_frame | 1 | `/ 25.0` |
| 8 | current_action_frame_count | 1 | `/ 25.0` |
| 9 | current_action_remaining_frames | 1 | `(count - frame) / 25.0` |
| 10 | is_action_end | 1 | bool → 0/1 |
| 11 | is_always_cancelable | 1 | bool → 0/1 |
| 12 | current_action_hit_count | 1 | raw float |
| 13 | current_hit_stun_frame | 1 | `/ 10.0` |
| 14 | is_in_hit_stun | 1 | bool → 0/1 |
| 15 | sprite_shake_position | 1 | raw float |
| 16 | max_sprite_shake_frame | 1 | `/ 10.0` |
| 17 | is_face_right | 1 | bool → 0/1 |
| 18 | current_frame_advantage | 1 | `/ 10.0` |

**Privileged features (self-only, 4 + num_actions floats):**
These appear ONLY in the self observation, stripped from opponent view.

| # | Feature | Size | Notes |
|---|---|---|---|
| 19 | would_next_forward_input_dash | 1 | bool → 0/1 |
| 20 | would_next_backward_input_dash | 1 | bool → 0/1 |
| 21 | special_attack_progress | 1 | clamped to max 1.0 |
| 22 | previous_action | num_actions | one-hot of env-level action |
| 23 | is_holding_special_charge | 1 | bool → 0/1 |

### Action ID One-Hot Mapping

The `current_action_id` one-hot uses raw enum integer values (non-contiguous), indexed by position in this ordered array:

```
Index:  0    1    2    3     4     5     6     7     8     9    10   11   12   13   14   15   16
Value:  0    1    2    10    11    100   105   110   115   200  301  305  306  310  350  500  510
Name: STAND FWD  BACK DASH_ DASH_ N_ATK B_ATK N_SP  B_SP  DMG  GD_M GD_S GD_C GD_B GD_P DEAD WIN
                      FWD   BACK
```

Python equivalent: `list(constants.FOOTSIES_ACTION_IDS.values()).index(raw_action_id)`

---

## Why `prev_actions` and `is_holding_special` Are Passed from Python

These are **Python-side environment state**, not game state:

- **`prev_actions`**: The action the agent *selected* on the previous step (before action delay processing). Used as a privileged feature so the agent knows what it recently chose. Tracked by `FootsiesEnv.prev_selected_actions`.

- **`is_holding_special`**: Whether the special charge toggle is active for each agent. This is a meta-action mechanism where `SPECIAL_CHARGE` toggles continuous ATTACK holding. Tracked by `FootsiesEnv._holding_special_charge`.

The C# simulation doesn't have action delay queues or special charge toggle logic — it receives already-resolved bit-level inputs. So these must be passed from Python in `BatchStepEncodedInput`.

On **reset**, both default to 0/false (no previous action, not charging).

---

## C# File Changes

### New: `Assets/Script/VectorizedEncoder.cs`
Static, stateless encoder. Writes directly into pre-allocated `float[]` buffers — zero allocation inside `Parallel.For`. Key methods:
- `VectorizedEncoder.ObservationSize(numActions)` → int
- `VectorizedEncoder.EncodeP1Centric(f1, f2, prevP1, prevP2, p1Hold, p2Hold, numActions, buffer, offset)`
- `VectorizedEncoder.EncodeP2Centric(...)` — same signature

### Modified: `Assets/Script/BatchMessages.cs`
- Renamed `BatchEncodedState` → `BatchRawState` (same wire format)
- Added new `BatchEncodedState` (flat float arrays, fields 1–5)
- Added `BatchStepEncodedInput`, `BatchResetEncodedInput`, `BatchResetAllEncodedInput`

### Modified: `Assets/Script/VectorizedEnvironmentManager.cs`
- Added `BatchStepAndEncode()` — steps + encodes in single `Parallel.For`
- Added `BatchResetAndEncode()`, `ResetAllAndEncode()`
- Pre-allocated `float[]` encoding buffers, lazily sized
- Exposed `GetP1Encodings()` / `GetP2Encodings()`

### Modified: `Assets/Script/VectorizedGrpcService.cs`
- Raw endpoints now return `BatchRawState`
- Added `HandleBatchStepEncoded`, `HandleBatchResetEncoded`, `HandleBatchResetAllEncoded`
- `BuildEncodedBatchResponse()` copies flat encoding arrays + metadata

### Modified: `Assets/Script/AIEncoder.cs`
- Removed observation delay entirely (history queues, `_observationDelay`, `resetObsHistory()`, `setObservationDelay()`)
- Constructor is now parameterless
- Encoding is stateless

### Modified: `Assets/Script/BattleCore.cs`
- Updated `AIEncoder` constructor call (no args)
- Removed `encoder.resetObsHistory()` calls

### Modified: `Assets/Script/BattleAIBarracuda.cs`
- Updated `AIEncoder` constructor call (no args)
- Removed `encoder.setObservationDelay()` call
- `resetObsHistory()` is now a no-op

---

## Python File Changes

### Regenerated: `footsiesgym/footsies/game/proto/footsies_service_pb2.py` + `_grpc.py`
Proto stubs regenerated from updated `.proto`. New message classes and service stubs available:
- `pb2.BatchRawState` (was `BatchEncodedState`)
- `pb2.BatchEncodedState` (new, flat encodings)
- `pb2.BatchStepEncodedInput`, `pb2.BatchResetEncodedInput`, `pb2.BatchResetAllEncodedInput`
- `VectorizedFootsiesServiceStub` now has `BatchStepEncoded`, `BatchResetEncoded`, `BatchResetAllEncoded`

### Updated: `scripts/grpc_batch_test.py`
Tests both raw and encoded modes with throughput comparison.

---

## Python Code That Needs Updating

Any Python code that previously used `BatchEncodedState` response fields needs to change:

1. **Raw mode callers**: Replace `state.p1_encodings` / `state.p2_encodings` with per-field access like `state.p1_position_x[i]`, `state.p1_current_action_id[i]`, etc. The return type is now `BatchRawState`.

2. **Encoded mode callers**: Use the new `BatchStepEncoded` / `BatchResetEncoded` / `BatchResetAllEncoded` endpoints. Pass `prev_p1_actions`, `prev_p2_actions`, `p1_holding_special`, `p2_holding_special`, `num_actions` in the request. Reshape response: `np.array(state.p1_encodings).reshape(num_envs, obs_size)`.

3. **`FootsiesEncoder` class**: `observation_size = 86` is correct for `num_actions=7`. Consider making it dynamic: `observation_size = 79 + num_actions`.

4. **Any code referencing `encoder.reset()` or observation delay**: The encoder `_last_common_state` tracking may still exist but observation delay is gone from C# side. Align Python encoder if needed.
