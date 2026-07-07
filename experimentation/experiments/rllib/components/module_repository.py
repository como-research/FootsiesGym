import dataclasses
import logging
import os
from typing import TYPE_CHECKING

import natsort
import numpy as np
from gymnasium import spaces
from ray.rllib import policy as rllib_policy
from ray.rllib.core.rl_module import rl_module
from ray.rllib.examples._old_api_stack.policy import random_policy
from ray.rllib.utils.framework import try_import_tf, try_import_torch

from experimentation.models.modelv2 import noop
from footsiesgym.footsies import encoder, footsies_env

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FootsiesModuleSpec:
    module_name: str
    experiment_name: str
    checkpoint_number: int = -1
    policy_id: str = "focal_policy"
    is_rlmodule: bool = False
    trial_id: str = None


class ModuleRepository:

    modules = [
        FootsiesModuleSpec(
            module_name="MY_POLICY_NAME",  # must specify
            experiment_name="MY_EXPERIMENT_NAME",  #  must specify
            trial_id="MY_TRIAL_ID",  # only required if experiment has multiple trials
            checkpoint_number=-1,  # -1 for latest checkpoint
        ),
    ]

    static_modules = {
        "random": random_policy.RandomPolicy(
            observation_space=spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(encoder.FootsiesEncoder.observation_size,),
            ),
            action_space=footsies_env.FootsiesEnv.get_action_space(
                use_special_charge_action=True
            )["p1"],
            config={},
        ),
        "noop": noop.NoOpPolicy(
            observation_space=spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(encoder.FootsiesEncoder.observation_size,),
            ),
            action_space=footsies_env.FootsiesEnv.get_action_space(
                use_special_charge_action=True
            )["p1"],
            config={},
        ),
        # "random_rlmodule": rl_module.SingleAgentRLModuleSpec(
        #     module_class=random_rlm.RandomRLModule,
        #     observation_space=footsies_env.FootsiesEnv.observation_space,
        #     action_space=footsies_env.FootsiesEnv.action_space,
        #     model_config_dict={},
        # ),
        # "noop_rlmodule": rl_module.SingleAgentRLModuleSpec(
        #     module_class=noop.NoOpRLModule,
        #     observation_space=footsies_env.FootsiesEnv.observation_space,
        #     action_space=footsies_env.FootsiesEnv.action_space,
        #     model_config_dict={},
        # ),
    }

    @classmethod
    def get(cls, module_spec_name: str) -> rllib_policy.Policy | rl_module.RLModule:
        """Retrieve the policy from the policy repository."""

        if module_spec_name in cls.static_modules:
            return (
                cls.static_modules[module_spec_name].build()
                if "rlmodule" in module_spec_name
                else cls.static_modules[module_spec_name]
            )

        for module in cls.modules:
            if module.module_name == module_spec_name:
                return get_local_checkpoint(module)

        raise ValueError(
            f"Module {module_spec_name} not found in the policy repository."
        )


def get_local_checkpoint(
    module_spec: FootsiesModuleSpec,
) -> rllib_policy.Policy:
    """Retrieve the checkpoint from the local filesystem. If checkpoint_number is -1, the latest checkpoint is retrieved."""
    base_dir = os.path.expanduser(f"~/ray_results/{module_spec.experiment_name}")

    trial_name = module_spec.trial_id

    if trial_name is None:
        num_dirs = 0
        for fname in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, fname)):
                num_dirs += 1
                trial_name = fname

        if num_dirs > 1:
            raise ValueError(
                f"More than one trial found in {base_dir}. Please specify the trial ID with FootsiesModuleSpec.trial_id."
            )

    if trial_name is None:
        raise FileNotFoundError(f"No trials found in {base_dir}")

    if module_spec.checkpoint_number == -1:
        checkpoints = natsort.natsorted(
            [
                ckpt
                for ckpt in os.listdir(os.path.join(base_dir, trial_name))
                if ckpt.startswith("checkpoint_")
            ]
        )
        checkpoint_dir = os.path.join(base_dir, trial_name, checkpoints[-1])
    else:
        checkpoint_dir = os.path.join(
            base_dir,
            trial_name,
            f"checkpoint_{module_spec.checkpoint_number:06d}",
        )
    assert os.path.exists(
        checkpoint_dir
    ), f"Checkpoint {checkpoint_dir} does not exist."

    if module_spec.is_rlmodule:
        module_dir = os.path.join(
            checkpoint_dir, "learner/module_state", module_spec.policy_id
        )

        module = rl_module.RLModule.from_checkpoint(module_dir)
    else:
        module = rllib_policy.Policy.from_checkpoint(
            checkpoint_dir, policy_ids=module_spec.policy_id
        )[module_spec.policy_id]

    return module
