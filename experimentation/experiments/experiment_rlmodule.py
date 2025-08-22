import functools

import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms import appo
from ray.rllib.algorithms import callbacks as rllib_callbacks
from ray.rllib.core.rl_module import multi_rl_module, rl_module
from ray.rllib.env import multi_agent_env_runner
from ray.rllib.examples.rl_modules.classes import random_rlm
from ray.tune import CLIReporter
from ray.tune.result import (
    EPISODE_REWARD_MEAN,
    MEAN_ACCURACY,
    MEAN_LOSS,
    TIME_TOTAL_S,
    TIMESTEPS_TOTAL,
)

from callbacks import add_policies
from footsiesgym.footsies import footsies_env
from models.rl_modules import back, lstm_module, noop
from utils import matchmaking

eval_policies = []


class Experiment:
    def __init__(self, config: dict | None = None):

        config = config or {}

        self.config = config

    def construct_run_config(self):
        reporter = CLIReporter(
            metric_columns={
                MEAN_ACCURACY: "acc",
                MEAN_LOSS: "loss",
                TIME_TOTAL_S: "total time (s)",
                TIMESTEPS_TOTAL: "ts",
                EPISODE_REWARD_MEAN: "reward",
            }
        )

        run_config = air.config.RunConfig(
            name=self.config["experiment_name"],
            callbacks=(
                [WandbLoggerCallback(project="Footsies")]
                if not self.config.get("debug", False)
                else None
            ),
            failure_config=air.config.FailureConfig(
                max_failures=self.config.get("max_failures", 0),
                fail_fast=self.config.get("fail_fast", False),
            ),
            checkpoint_config=air.config.CheckpointConfig(
                checkpoint_frequency=self.config.get("checkpoint_freq", 50),
                checkpoint_at_end=True,
            ),
            progress_reporter=reporter,
            verbose=1,
        )
        return run_config

    def construct_tune_config(self):
        tune_config = tune.TuneConfig(
            num_samples=self.config.get("num_trials", 1),
            max_concurrent_trials=self.config.get("max_concurrent_trials", 1),
        )
        return tune_config

    def construct_model_config(self, as_dict=True):

        config = (
            appo.APPOConfig()
            .environment(
                "FootsiesEnv",
                env_config={
                    "max_t": 4000,
                    "frame_skip": 4,
                    "observation_delay": 16,
                    "launch_binaries": True,
                },
            )
            .framework("torch")
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .learners(
                num_learners=1,
                num_cpus_per_learner=1,
                num_gpus_per_learner=(
                    1 if not self.config.get("debug", False) else 0
                ),
                num_aggregator_actors_per_learner=1,
            )
            .env_runners(
                env_runner_cls=multi_agent_env_runner.MultiAgentEnvRunner,
                num_env_runners=(
                    40 if not self.config.get("debug", False) else 1
                ),
                num_cpus_per_env_runner=1,
                num_envs_per_env_runner=1,
                batch_mode="truncate_episodes",
                rollout_fragment_length=256,
                episodes_to_numpy=False,
            )
            .training(
                model={"uses_new_env_runners": True},
                lr=3e-4,
                entropy_coeff=0.01,
            )
            .multi_agent(
                policies={
                    "focal_policy",
                    "random",
                    "noop",
                    "back",
                },
                # policy_mapping_fn=lambda agent_id, episode, **kwargs: "focal_policy",
                policy_mapping_fn=matchmaking.Matchmaker(
                    [matchmaking.Matchup("focal_policy", "focal_policy", 1.0)]
                ).policy_mapping_fn,
                policies_to_train=["focal_policy"],
            )
            .rl_module(
                rl_module_spec=multi_rl_module.MultiRLModuleSpec(
                    rl_module_specs={
                        "focal_policy": rl_module.RLModuleSpec(
                            module_class=lstm_module.LSTMModule,
                            model_config={
                                "lstm_cell_size": 32,
                                "dense_layers": [128, 128],
                                "max_seq_len": 32,
                            },
                        ),
                        "random": rl_module.RLModuleSpec(
                            module_class=random_rlm.RandomRLModule
                        ),
                        "noop": rl_module.RLModuleSpec(
                            module_class=noop.NoOpRLModule
                        ),
                        "back": rl_module.RLModuleSpec(
                            module_class=back.BackRLModule
                        ),
                    },
                )
            )
            .evaluation(
                evaluation_num_env_runners=(
                    5 if not self.config.get("debug", False) else 1
                ),
                evaluation_interval=1,
                evaluation_duration="auto",
                evaluation_duration_unit="episodes",
                evaluation_parallel_to_training=True,
                evaluation_config={
                    "env_config": {"evaluation": True},
                    "multiagent": {
                        "policy_mapping_fn": matchmaking.Matchmaker(
                            [
                                matchmaking.Matchup(
                                    "focal_policy",
                                    eval_policy,
                                    1 / (len(eval_policies) + 3),
                                )
                                for eval_policy in eval_policies
                                + ["random", "back", "noop"]
                            ]
                        ).policy_mapping_fn,
                    },
                },
            )
            .callbacks(
                rllib_callbacks.make_multi_callbacks(
                    [
                        functools.partial(
                            add_policies.AddPolicies, policies=eval_policies
                        ),
                    ]
                )
            )
        )

        return config

    def env_creator(self, config, **kwargs):
        # TODO(chase): Move port logic here instead of in the environment.
        return footsies_env.FootsiesEnv(config=config)

    def run(self):
        ray.init(
            local_mode=self.config.get("debug", False),
        )

        ray.tune.registry.register_env(
            "FootsiesEnv",
            env_creator=self.env_creator,
        )

        model_config = self.construct_model_config()
        tune_config = self.construct_tune_config()
        run_config = self.construct_run_config()

        tuner = tune.Tuner(
            trainable=appo.APPO,
            param_space=model_config,
            tune_config=tune_config,
            run_config=run_config,
        )

        tuner.fit()
