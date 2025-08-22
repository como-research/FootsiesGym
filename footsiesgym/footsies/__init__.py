"""
FootsiesGym - A reinforcement learning environment for HiFight's Footsies game.
"""

from .footsies_env import FootsiesEnv
from . import encoder, typing

__all__ = ["FootsiesEnv", "encoder", "typing"]
