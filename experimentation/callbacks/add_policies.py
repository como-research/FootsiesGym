from ray.rllib.algorithms import algorithm as rllib_algorithm
from ray.rllib.algorithms import callbacks

from experimentation.components import module_repository


class AddPolicies(callbacks.DefaultCallbacks):
    def __init__(self, policies: list[str]):
        self.policies = policies

    def on_algorithm_init(
        self, *, algorithm: rllib_algorithm.Algorithm, metrics_logger, **kwargs
    ) -> None:
        if not self.policies:
            return

        for policy_name in self.policies:
            if algorithm.get_policy(policy_name) is None:
                policy = module_repository.ModuleRepository.get(policy_name)
                algorithm.add_policy(policy_id=policy_name, policy=policy)
