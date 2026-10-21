"""The JAX PPO example (experimentation/experiments/cleanrl/train_ppo_jax.py)
trains on the C/CUDA env, and its policy plays the same in Unity (`slow`)."""

import importlib.util
import sys
import types

import numpy as np
import pytest

for module in ("jax", "flax", "optax"):
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"needs {module}", allow_module_level=True)

from experimentation.experiments.cleanrl import play_unity  # noqa: E402
from experimentation.experiments.cleanrl import train_ppo_jax as ppo  # noqa: E402


def _train(tmp_path, monkeypatch, **overrides):
    args = dict(
        total_timesteps=2 * 64 * 2 * 16,  # 2 updates
        num_envs=64,
        num_steps=16,
        num_minibatches=2,
        update_epochs=1,
        eval_interval=1,
        eval_envs=32,
        eval_steps=50,
        output_dir=str(tmp_path),
        max_t=100,
    )
    args.update(overrides)
    argv = ["train_ppo_jax"]
    for key, value in args.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    monkeypatch.setattr(sys, "argv", argv)
    ppo.main()
    return tmp_path / "policy.msgpack"


def test_train_saves_a_loadable_policy(tmp_path, monkeypatch):
    path = _train(tmp_path, monkeypatch)
    network, params, config = ppo.load(path)
    assert config["max_t"] == 100
    logits, value = network.apply(params, np.zeros((3, 88), np.float32))
    assert logits.shape == (3, 6) and value.shape == (3,)


@pytest.mark.slow
def test_policy_plays_the_same_in_unity(tmp_path, monkeypatch, capsys):
    path = _train(tmp_path, monkeypatch)
    act, config = play_unity.make_policy(path, seed=0)
    args = types.SimpleNamespace(num_envs=16, steps=200, seed=0)
    play_unity.compare(args, act, config)  # raises SystemExit if they differ
    assert "OK: Unity and C identical" in capsys.readouterr().out
