# CleanRL PPO for FootsiesGym

This directory has two PPO examples:

- [`train_ppo_jax.py`](#jax-ppo-on-the-ccuda-environment): JAX, on the C/CUDA environment. All of the training runs on the GPU.
- `train_ppo.py` (the sections after it): PyTorch, on the Unity game server.

## JAX PPO on the C/CUDA environment

`train_ppo_jax.py` is self-play PPO in JAX, in the style of CleanRL's `ppo_atari_envpool_xla_jax_scan.py`. Both players use one policy. The environment is `footsiesgym.make(backend="c", functional=True)`. Its step is an XLA call. Thus, each PPO update (rollout, GAE and all minibatch updates) is one jitted function on the GPU.

`play_unity.py` validates the trained policy in the Unity game. The policy (p1) plays against random actions (p2) on the Unity game server. The C environment gets the same actions at each step. The script compares the observations, rewards and done flags of Unity and C, byte for byte, and gives the win rate of the policy in Unity.

### Procedure

1. Install the dependencies:

   ```bash
   pip install -e ".[jax-training]"
   pip install "jax[cuda12]"        # JAX with CUDA, for the GPU
   ```

2. Train the policy:

   ```bash
   python -m experimentation.experiments.cleanrl.train_ppo_jax
   ```

   The script saves the policy in `runs/ppo_jax/policy.msgpack` and its configuration in `runs/ppo_jax/policy.json`. Add `--track` to log to Weights & Biases. For all options, run the script with `--help`.

3. Play the policy in the Unity game and compare with the C environment:

   ```bash
   python -m experimentation.experiments.cleanrl.play_unity --policy runs/ppo_jax/policy.msgpack
   ```

   If Unity and C are not identical, the script stops and shows the first step and game that are different.

4. To look at one game in the Unity window, add `--windowed`:

   ```bash
   python -m experimentation.experiments.cleanrl.play_unity --policy runs/ppo_jax/policy.msgpack --windowed
   ```

### Results

With the default options on one RTX 3090, the training does 200M agent steps in approximately 50 seconds. The win rate against random actions is approximately 0.88. In the Unity game, the win rate of the policy is the same (0.879 over 4,883 episodes). Unity and C were identical for the first episode of each of the 128 games.

The script compares each game through its first episode only. On reset, the Unity server keeps some fighter fields from the previous episode, and the C environment starts a fully new game.

`testing/test_cleanrl_jax.py` trains a small policy and plays it in Unity (the Unity test has the `slow` marker).

## PyTorch PPO on the Unity game server

Self-play PPO training using a single vectorized FootsiesEnv server. Both agents (p1 and p2) share the same policy network.

### Setup

```bash
pip install -e ".[all]"
```

Requires `wandb` for logging and sweep orchestration. Log in first:

```bash
wandb login
```

### Launching a WandB Sweep

#### 1. Create the sweep

From the repo root:

```bash
wandb sweep experimentation/experiments/cleanrl/ppo_sweep.yaml
```

This prints a sweep ID like `your-entity/Footsies-v0/abc123def`.

#### 2. Start sweep agents

On each machine (or in each tmux pane) you want to run trials:

```bash
wandb agent <sweep-id>
```

For example:

```bash
wandb agent your-entity/Footsies-v0/abc123def
```

Each agent pulls hyperparameter configurations from the sweep controller and runs `train_ppo.py` with those values injected via `wandb.config`. You can launch multiple agents in parallel across machines to run trials concurrently.

#### 3. Monitor

Go to the sweep page on [wandb.ai](https://wandb.ai) to view:
- Parallel coordinates plot of hyperparameters vs. `eval/win_rate_vs_random`
- Per-run training curves (`losses/`, `charts/`, `eval/`)
- Bayesian optimization progress

### Sweep Configuration

`ppo_sweep.yaml` uses Bayesian optimization (`method: bayes`) to maximize `eval/win_rate_vs_random`. Tuned parameters:

| Parameter | Values |
|-----------|--------|
| `learning_rate` | 1e-4, 3e-4, 6e-4, 1e-3 |
| `ent_coef` | 0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0 |
| `gamma` | 0.99, 0.995 |
| `gae_lambda` | 0.9, 0.95, 0.99 |
| `clip_coef` | 0.1, 0.2, 0.3 |
| `vf_coef` | 0.5, 1.0 |
| `num_minibatches` | 4, 8 |
| `update_epochs` | 4, 8 |
| `anneal_lr` | true, false |
| `anneal_ent` | true, false |
| `num_steps` | 128, 256, 512, 1024, 2048 |
| `hidden_size` | 64, 128, 256 |

All other settings use the defaults in `train_ppo.py` (`DEFAULT_CONFIG`).

### Standalone Run (No Sweep)

To run a single training with default hyperparameters:

```bash
python -m experimentation.experiments.cleanrl.train_ppo
```

This still logs to wandb. Override defaults by editing `DEFAULT_CONFIG` in `train_ppo.py`.

### Environment Config

The environment settings (`ENV_CONFIG` in `train_ppo.py`) are fixed across sweep runs:

| Setting | Value |
|---------|-------|
| `frame_skip` | 4 |
| `action_delay` | 16 |
| `max_t` | 4000 |
| `use_special_charge_action` | False |
| `win_reward_scaling_coeff` | 10.0 |
| `guard_break_reward` | 0.0 |

To sweep over env settings, add them to `ppo_sweep.yaml` and update `train_ppo.py` to read them from `wandb.config`.
