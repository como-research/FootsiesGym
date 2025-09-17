import functools
import os

import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms import appo
from ray.rllib.algorithms import callbacks as rllib_callbacks
from ray.rllib.examples._old_api_stack.policy import random_policy
from ray.rllib.policy import policy as rllib_policy
from ray.tune import CLIReporter
from ray.tune.result import (
    EPISODE_REWARD_MEAN,
    MEAN_ACCURACY,
    MEAN_LOSS,
    TIME_TOTAL_S,
    TIMESTEPS_TOTAL,
)
from ray.tune.search.hyperopt import HyperOptSearch

from experimentation.callbacks import add_policies, script_metrics, winrates
from footsiesgym.footsies import footsies_env
from experimentation.models.modelv2 import back, lstm_model, noop
from experimentation.utils import matchmaking
from experimentation.components import emagnet


def eval_policy_mapping_fn(*args, **kwargs): ...


class Experiment:

    NUM_ENVS_PER_ENV_RUNNER = 1

    def __init__(self, config: dict | None = None):

        config = config or {}

        self.config = config

    def construct_run_config(self):
        reporter = CLIReporter(
            metric_columns={
                MEAN_ACCURACY: "acc",
                MEAN_LOSS: "loss",
                # TRAINING_ITERATION: "iter",
                TIME_TOTAL_S: "total time (s)",
                TIMESTEPS_TOTAL: "ts",
                EPISODE_REWARD_MEAN: "reward",
            }
        )

        run_config = air.config.RunConfig(
            name=self.config["experiment_name"],
            stop={
                "num_agent_steps_trained": self.config.get(
                    "agent_steps_to_train", 300_000_000
                )
            },
            callbacks=(
                [WandbLoggerCallback(project="Footsies-v0")]
                if not self.config.get("debug", False)
                else None
            ),
            failure_config=air.config.FailureConfig(
                max_failures=self.config.get("max_failures", 0),
                fail_fast=self.config.get("fail_fast", False),
            ),
            checkpoint_config=air.config.CheckpointConfig(
                checkpoint_frequency=self.config.get("checkpoint_freq", 50),
                # num_to_keep=self.config.get("num_to_keep", 5),
                checkpoint_at_end=True,
            ),
            progress_reporter=reporter,
            verbose=1,
        )
        return run_config

    def construct_tune_config(self):

        if self.config.get("tune"):
            tune_config = tune.TuneConfig(
                num_samples=self.config.get("num_trials", 20),
                max_concurrent_trials=self.config.get(
                    "max_concurrent_trials", 1
                ),
                metric="evaluation/env_runners/custom_metrics/winrates/focal_policy/vs_random_mean",
                mode="max",
                search_alg=HyperOptSearch(),
                scheduler=tune.schedulers.ASHAScheduler(
                    time_attr="num_agent_steps_trained",
                    max_t=100_000_000,
                    grace_period=20_000_000,
                ),
            )
        else:
            tune_config = tune.TuneConfig(
                num_samples=self.config.get("num_trials", 1),
                max_concurrent_trials=self.config.get(
                    "max_concurrent_trials", 1
                ),
                trial_name_creator=lambda trial: f"{self.config.get('experiment_name')}-{str(trial).split('_')[-1]}",
            )
        return tune_config

    def construct_model_config(self, as_dict=True):

        policy_observation_space = footsies_env.FootsiesEnv.observation_space[
            "p1"
        ]
        policy_action_space = footsies_env.FootsiesEnv.action_space["p1"]

        # Add policy names stored in the ModuleRepository here to evaluate against them
        eval_policies = []

        config = (
            appo.APPOConfig()
            .environment(
                "FootsiesEnv",
                env_config={
                    "max_t": 1000,
                    "frame_skip": 4,
                    "action_delay": 16,
                    "num_envs_per_worker": self.NUM_ENVS_PER_ENV_RUNNER,
                    "guard_break_reward": 0.0,
                    "launch_binaries": True,
                },
            )
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .resources(
                num_learner_workers=1,
                num_gpus_per_learner_worker=(
                    1 if not self.config.get("debug", False) else 0
                ),
                num_gpus=(1 if not self.config.get("debug", False) else 0),
                num_cpus_for_local_worker=1,
            )
            .env_runners(
                num_env_runners=(
                    40 if not self.config.get("debug", False) else 1
                ),
                # Must be 1 unless the port configuration is changed
                # in footsies_env.py, which finds the port according
                # to the worker index.
                num_envs_per_env_runner=self.NUM_ENVS_PER_ENV_RUNNER,
            )
            .env_runners(
                rollout_fragment_length=128,
                batch_mode="truncate_episodes",
            )
            .multi_agent(
                policies={
                    "focal_policy": rllib_policy.PolicySpec(
                        config={
                            "model": {
                                "custom_model": lstm_model.LSTMModel,
                                "custom_model_config": {
                                    "lstm_cell_size": 256,
                                    "policy_dense_widths": [256, 256],
                                },
                            },
                            "max_seq_len": 32,
                        },
                        observation_space=policy_observation_space,
                        action_space=policy_action_space,
                    ),
                    "random": rllib_policy.PolicySpec(
                        policy_class=random_policy.RandomPolicy,
                        observation_space=policy_observation_space,
                        action_space=policy_action_space,
                    ),
                    "back": rllib_policy.PolicySpec(
                        policy_class=back.BackPolicy,
                        observation_space=policy_observation_space,
                        action_space=policy_action_space,
                    ),
                    "noop": rllib_policy.PolicySpec(
                        policy_class=noop.NoOpPolicy,
                        observation_space=policy_observation_space,
                        action_space=policy_action_space,
                    ),
                },
                policy_mapping_fn=matchmaking.Matchmaker(
                    [matchmaking.Matchup("focal_policy", "focal_policy", 1.0)]
                ).policy_mapping_fn,
                policies_to_train=["focal_policy"],
            )
            .evaluation(
                evaluation_num_env_runners=(
                    8 if not self.config.get("debug", False) else 1
                ),
                evaluation_interval=1,
                evaluation_duration="auto",
                evaluation_duration_unit="timesteps",
                evaluation_parallel_to_training=True,
                evaluation_config={
                    "env_config": {"evaluation": True},
                    "multiagent": {
                        "policy_mapping_fn": matchmaking.Matchmaker(
                            [
                                matchmaking.Matchup(
                                    "focal_policy",
                                    eval_policy,
                                    1 / (len(eval_policies) + 1),
                                )
                                for eval_policy in eval_policies + ["random"]
                            ]
                        ).policy_mapping_fn,
                    },
                },
            )
            .callbacks(
                rllib_callbacks.make_multi_callbacks(
                    [
                        winrates.Winrates,
                        functools.partial(
                            add_policies.AddPolicies, policies=eval_policies
                        ),
                        script_metrics.ScriptMetrics,
                    ]
                )
            )
        )

        if self.config.get("tune"):
            config.training(
                lr=tune.loguniform(1e-4, 1e-3),
                train_batch_size=1024,
                entropy_coeff=tune.loguniform(1e-4, 1e-3),
                gamma=0.99,
                vf_loss_coeff=1.0,
                lambda_=0.95,
            )
            config["magnet_learning_rate_schedule"] = tune.loguniform((1e-4)/16, (1e-3)/16)
            config["temperature_schedule"] = tune.loguniform(0.02, 0.1)
        else:

            config.training(
                train_batch_size=4096,
                # lr_schedule=[[0, 0.001], [5_000_000, 0.00075], [10_000_000, 3e-4]],
                lr=6e-4,
                entropy_coeff=0.006,
                # entropy_coeff_schedule=[[0, 0.03], [200_000_000, 0.01]],
                gamma=0.99,
                vf_loss_coeff=1.0,
                lambda_=0.95,
            )
        
            config["magnet_learning_rate_schedule"] = (6e-4)/16
            config["temperature_schedule"] = 0.04
        
        return config

    def env_creator(self, config, **kwargs):
        return footsies_env.FootsiesEnv(config=config)

    def run(self):
        ray.init(
            local_mode=self.config.get("debug", False),
        )

        ray.tune.registry.register_env(
            "FootsiesEnv",
            env_creator=self.env_creator,
        )

        ray.tune.registry.register_trainable(
            "EMAgnetAPPO", emagnet.EMAgnetAPPO
        )

        experiment_name = self.config.get("experiment_name")
        results_path = os.path.expanduser(f"~/ray_results/{experiment_name}")
        experiment_exists = os.path.exists(results_path)
        if experiment_exists and experiment_name != "test":
            print("Experiment already exists, restoring...")
            tuner = tune.Tuner.restore(
                results_path,
                trainable=emagnet.EMAgnetAPPO,
            )

        else:
            model_config = self.construct_model_config()
            tune_config = self.construct_tune_config()
            run_config = self.construct_run_config()

            tuner = tune.Tuner(
                trainable=emagnet.EMAgnetAPPO,
                param_space=model_config,
                tune_config=tune_config,
                run_config=run_config,
            )

        tuner.fit()
