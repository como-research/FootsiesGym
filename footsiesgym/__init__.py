"""
FootsiesGym - A reinforcement learning environment for HiFight's Footsies game.

This package provides a Gymnasium-compatible environment for training reinforcement
learning agents on the Footsies fighting game.

Binaries are automatically downloaded from Git LFS when first needed.
"""

from .footsies.footsies_env import FootsiesEnv
from .footsies import encoder, typing
from .binary_manager import get_binary_manager

__version__ = "0.1.7"
__all__ = ["FootsiesEnv", "encoder", "typing", "make"]

# Initialize binary manager (but don't download yet - wait until needed)
_binary_manager = get_binary_manager()


def make(config: dict | None = None, platform: str = "linux", launch_binaries: bool = True):
    """
    Create a FootsiesGym environment.
    
    Args:
        config: Configuration dictionary for the environment
        platform: Platform to run on (currently only "linux" supported for auto-launch)
        launch_binaries: Whether to automatically launch game binaries
        
    Returns:
        FootsiesEnv: The configured environment instance
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
        "max_t": 4000,
        "frame_skip": 4,
        "observation_delay": 16,
        "reward_guard_break": False,
    }

    if config is not None:
        default_config.update(config)

    return FootsiesEnv(config=default_config)
