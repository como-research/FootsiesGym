"""Wrappers for integrating FootsiesEnv with external frameworks."""


def wrap_rllib(env):
    """Wrap a FootsiesEnv (PettingZoo ParallelEnv) for use with RLlib.

    Args:
        env: A FootsiesEnv instance (pettingzoo.ParallelEnv).

    Returns:
        An RLlib MultiAgentEnv wrapping the given environment.

    Raises:
        ImportError: If ray[rllib] is not installed.
    """
    try:
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    except ImportError:
        raise ImportError(
            "ray[rllib] is required to use the RLlib wrapper. "
            "Install it with: pip install 'footsies-gym[rllib]'"
        )

    return ParallelPettingZooEnv(env)
