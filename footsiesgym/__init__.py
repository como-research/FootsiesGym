"""
FootsiesGym - A reinforcement learning environment for HiFight's Footsies game.

This package provides a Gymnasium-compatible environment for training reinforcement
learning agents on the Footsies fighting game.

Binaries are automatically downloaded from a CDN (with a GitHub Releases
fallback) when first needed and verified with SHA256 checksums.
"""

from .binary_manager import get_binary_manager
from .footsies import encoder, typing
from .footsies.footsies_env import FootsiesEnv, merge_config

__version__ = "1.0.1"
__all__ = ["FootsiesEnv", "encoder", "typing", "make", "make_class"]

# Initialize binary manager (but don't download yet - wait until needed)
_binary_manager = get_binary_manager()


def make(config: dict | None = None, **kwargs):
    """
    Create a FootsiesGym environment.

    Every setting is a config key, and can equally be passed as a keyword
    argument; ``make(config={"num_envs": 64}, backend="c")`` and
    ``make(num_envs=64, backend="c")`` are the same env. Setting a key both
    ways to different values is an error.

    Keys besides the game settings (see the README):
        backend: "grpc" (default; the Unity game server) or "c" (the
            bit-identical C port, in-process; no binaries).
        device: "cpu" (default) or "cuda" / "cuda:N" (backend "c").
        num_envs: games stepped together (default 1, the single-game
            PettingZoo API; more give arrays with a leading num_envs axis).
        tensor_type: "numpy" (default), "torch" or "jax".
        platform: "linux" (default) or "mac" (backend "grpc").
        launch_binaries: start the game server automatically (default True;
            backend "grpc").
        functional: return the env as pure JAX functions of an explicit state
            (Sardine's functional API) for jax.jit / lax.scan training loops,
            instead of a stateful env (backend "c"; needs JAX). They run where
            JAX runs them (jax.default_device, or the jit's device), so
            device is not set.

    For frameworks that build the env themselves from a class (e.g. Sardine),
    ``make_class`` takes the same settings. For RLlib, wrap the env with
    ``footsiesgym.wrappers.wrap_rllib``.

    Returns:
        A FootsiesEnv (PettingZoo ParallelEnv); with functional=True a
        functional env with reset(key) -> (obs, state, infos) and
        step(key, state, actions) -> (obs, state, rewards, terminateds,
        truncateds, infos).
    """
    return _build(_resolve(merge_config(config, kwargs)))


def make_class(config: dict | None = None, **kwargs) -> type:
    """``make()`` as a class, for frameworks that build the env themselves.

    Instances of the returned class are what ``make()`` returns for these
    settings merged with the ones the class is called with (merged like
    ``make()``'s config and keywords: setting a key to two values is an
    error). ``functional`` picks the kind of class, so it is fixed here:

    * default: a ``FootsiesEnv`` subclass, ``Cls(config)`` / ``Cls(**kwargs)``.
    * ``functional=True``: the functional env, built the way Sardine does,
      ``Cls(**env_config).vectorize(num_envs)``::

          FootsiesEnv = footsiesgym.make_class(backend="c", functional=True)
          Experiment().environment(env=FootsiesEnv, env_config={...}, num_envs=4096)

    Every class has ``possible_agents`` (``["p1", "p2"]``) as a class
    attribute, so it can be read without building an env.
    """
    preset = merge_config(config, kwargs)
    fixed = {"functional": preset.get("functional", False)}

    def settings(config, kwargs) -> dict:
        merged = merge_config(preset, merge_config(config, kwargs))
        for key, value in fixed.items():
            if merged.get(key, False) != value:
                raise ValueError(f"{key} is fixed by make_class")
        return merged

    if fixed["functional"]:
        _resolve(preset)  # check the settings now
        if "num_envs" in preset:
            raise ValueError("functional classes get num_envs from vectorize()")
        from footsiesgym.c.jax_env import FootsiesFFIEnv

        class FootsiesFunctionalEnv(FootsiesFFIEnv):
            def __init__(self, config: dict | None = None, **kwargs):
                super().__init__(_resolve(settings(config, kwargs)))

        return FootsiesFunctionalEnv

    _resolve(preset)  # check the settings now

    class FootsiesEnvClass(FootsiesEnv):
        def __init__(self, config: dict | None = None, **kwargs):
            super().__init__(config=_resolve(settings(config, kwargs)))

    return FootsiesEnvClass


def _resolve(config: dict) -> dict:
    """``make()``'s defaults added to ``config``, and the settings checked."""
    if "rllib" in config:
        raise ValueError(
            "rllib is not a setting: wrap the env for RLlib with "
            "footsiesgym.wrappers.wrap_rllib(env)"
        )
    config = {"platform": "linux", "launch_binaries": True, **config}
    backend = config.get("backend", "grpc")
    if config.get("functional", False):
        if backend != "c":
            raise ValueError("functional=True needs backend='c'")
        if config.get("tensor_type", "jax") != "jax":
            raise ValueError("functional=True returns JAX arrays (tensor_type='jax')")
        if "device" in config:
            raise ValueError(
                "functional=True runs where JAX runs it: use jax.default_device "
                "(or the jit's device) instead of device"
            )
    if config["launch_binaries"] and backend == "grpc":
        assert config["platform"] in ("linux", "mac"), (
            "Automated binary launching is supported on Linux and MacOS. "
            "Windows is not supported."
        )
    return config


def _build(config: dict):
    """What ``make()`` returns for a ``_resolve``d config."""
    config = dict(config)
    if config.pop("functional", False):
        from footsiesgym.c.jax_env import FootsiesFFIEnv

        return FootsiesFFIEnv(config).vectorize(config.get("num_envs", 1))
    return FootsiesEnv(config=config)
