# CleanRL PPO for FootsiesGym

Self-play PPO training using a single vectorized FootsiesEnv server. Both agents (p1 and p2) share the same policy network.

## Setup

```bash
pip install -e ".[all]"
```

Requires `wandb` for logging and sweep orchestration. Log in first:

```bash
wandb login
```

## Launching a WandB Sweep

### 1. Create the sweep

From the repo root:

```bash
wandb sweep experimentation/experiments/cleanrl/ppo_sweep.yaml
```

This prints a sweep ID like `your-entity/Footsies-v0/abc123def`.

### 2. Start sweep agents

On each machine (or in each tmux pane) you want to run trials:

```bash
wandb agent <sweep-id>
```

For example:

```bash
wandb agent your-entity/Footsies-v0/abc123def
```

Each agent pulls hyperparameter configurations from the sweep controller and runs `train_ppo.py` with those values injected via `wandb.config`. You can launch multiple agents in parallel across machines to run trials concurrently.

### 3. Monitor

Go to the sweep page on [wandb.ai](https://wandb.ai) to view:
- Parallel coordinates plot of hyperparameters vs. `eval/win_rate_vs_random`
- Per-run training curves (`losses/`, `charts/`, `eval/`)
- Bayesian optimization progress

## Sweep Configuration

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

## Standalone Run (No Sweep)

To run a single training with default hyperparameters:

```bash
python -m experimentation.experiments.cleanrl.train_ppo
```

This still logs to wandb. Override defaults by editing `DEFAULT_CONFIG` in `train_ppo.py`.

## Environment Config

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
