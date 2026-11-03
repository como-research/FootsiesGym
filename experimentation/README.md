# Training Examples

> [!NOTE]
> The code for the experiments of the paper is not available yet. These examples are not the code for those experiments.

Run the commands from the root of the repository. The examples have dependencies that FootsiesGym does not install. Install them with `uv pip install --python .venv`, as shown below. `uv sync` removes these packages, so install them again after each `uv sync`.

## CleanRL

PPO examples, each in one file. For more data, refer to [`experiments/cleanrl/README.md`](experiments/cleanrl/README.md).

- JAX on the C/CUDA environment. All of the training runs on the GPU. A second script plays the trained policy in the Unity game and compares Unity with the C environment.
- PyTorch on the Unity game server.

```bash
uv sync --extra cuda                                 # without a GPU: --extra jax
uv pip install --python .venv "flax>=0.10" "optax>=0.2"
uv run python -m experimentation.experiments.cleanrl.train_ppo_jax
uv run python -m experimentation.experiments.cleanrl.play_unity --policy runs/ppo_jax/policy.msgpack
```

## RLlib

[APPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#appo). You can use the RLModule stack (recommended) or the legacy ModelV2 stack:

```bash
uv sync
uv pip install --python .venv "ray[rllib]>=2.46.0" "torch>=2.0.0" "wandb>=0.15.0" natsort
uv run python -m experimentation.experiments.rllib.train_rlmodule --experiment-name <name> [--debug]
uv run python -m experimentation.experiments.rllib.train --experiment-name <name>
```
