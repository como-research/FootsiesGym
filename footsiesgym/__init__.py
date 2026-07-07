"""
FootsiesGym - A reinforcement learning environment for HiFight's Footsies game.

This package provides a Gymnasium-compatible environment for training reinforcement
learning agents on the Footsies fighting game.

Binaries are automatically downloaded from a CDN (with a GitHub Releases
fallback) when first needed and verified with SHA256 checksums.
"""

from .binary_manager import get_binary_manager
from .footsies import encoder, typing
from .footsies.footsies_env import FootsiesEnv

__version__ = "0.7.2"
__all__ = ["FootsiesEnv", "encoder", "typing", "make"]

# Initialize binary manager (but don't download yet - wait until needed)
_binary_manager = get_binary_manager()


def make(
    config: dict | None = None,
    platform: str = "linux",
    launch_binaries: bool = True,
    rllib: bool = False,
):
    """
    Create a FootsiesGym environment.

    Args:
        config: Configuration dictionary for the environment
        platform: Platform to run on (currently only "linux" supported for auto-launch)
        launch_binaries: Whether to automatically launch game binaries
        rllib: If True, wrap the environment for RLlib (requires ray[rllib]).
            For vectorized mode (num_envs > 1), uses VectorizedFootsiesRLlibEnv.
            For single-env mode, uses ParallelPettingZooEnv wrapper.

    Returns:
        A FootsiesEnv (PettingZoo ParallelEnv), or an RLlib MultiAgentEnv if rllib=True.
    """
    if launch_binaries:
        assert platform == "linux", (
            "Only linux is supported for automated binary launching. "
            "Create the environment manually and launch binaries by hand to use MacOS. "
            "Windows TBD."
        )

    default_config = {
        "platform": platform,
        "launch_binaries": launch_binaries,
    }

    if config is not None:
        default_config.update(config)

    if rllib and default_config.get("num_envs", 1) > 1:
        from footsiesgym.wrappers import VectorizedFootsiesRLlibEnv

        return VectorizedFootsiesRLlibEnv(default_config)

    env = FootsiesEnv(config=default_config)

    if rllib:
        from footsiesgym.wrappers import wrap_rllib

        return wrap_rllib(env)

    return env
