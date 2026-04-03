import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms import ppo
from ray.rllib.core.rl_module import multi_rl_module, rl_module
from ray.tune import CLIReporter
from ray.tune.result import (
    EPISODE_REWARD_MEAN,
    MEAN_ACCURACY,
    MEAN_LOSS,
    TIME_TOTAL_S,
    TIMESTEPS_TOTAL,
)

import footsiesgym
from experimentation.experiments.rllib import matchmaking
from experimentation.experiments.rllib.callbacks.winrates_v2 import WinratesV2
from experimentation.experiments.rllib.env_runner import FootsiesEnvRunner

eval_policies = []

# Number of game instances per env runner (server-side vectorized)
# GAMES_PER_RUNNER = 48
TOTAL_ENV_STEPS = 10_000_000


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
            stop={
                "learners/focal_policy/num_module_steps_trained_lifetime": TOTAL_ENV_STEPS
            },
            callbacks=(
                [WandbLoggerCallback(project="Footsies-RLlib-8ad-0")]
                if not self.config.get("debug", False)
                else None
            ),
            failure_config=air.config.FailureConfig(
                max_failures=self.config.get("max_failures", 0),
                fail_fast=self.config.get("fail_fast", False),
            ),
            checkpoint_config=air.config.CheckpointConfig(
                checkpoint_frequency=self.config.get("checkpoint_freq", 100),
                checkpoint_at_end=True,
            ),
            progress_reporter=reporter,
            verbose=1,
        )
        return run_config

    def construct_tune_config(self):
        tune_config = tune.TuneConfig(
            num_samples=self.config.get("num_trials", 100),
            max_concurrent_trials=self.config.get("max_concurrent_trials", 4),
        )
        return tune_config

    def construct_model_config(self):
        debug = self.config.get("debug", False)
        num_runners = 1 if debug else 1
        # games_per_runner = (
        #     GAMES_PER_RUNNER if not self.config.get("debug", False) else 2
        # )

        config = (
            ppo.PPOConfig()
            .environment(
                "FootsiesEnv",
                env_config={
                    "max_t": 4000,
                    "frame_skip": 4,
                    "action_delay": 8,
                    "guard_break_reward": 0.0,
                    "win_reward_scaling_coeff": 10.0,
                    "use_reward_budget": False,
                    "launch_binaries": True,
                    "use_special_charge_action": False,
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
                num_gpus_per_learner=(0 if debug else 0.25),
            )
            .env_runners(
                env_runner_cls=FootsiesEnvRunner,
                num_env_runners=num_runners,
                num_cpus_per_env_runner=1,
                num_envs_per_env_runner=16,
                create_env_on_local_worker=True,
                batch_mode="truncate_episodes",
                rollout_fragment_length="auto",
            )
            .training(
                **self._training_params(),
            )
            .multi_agent(
                policies={
                    "policy_a",
                    "policy_b",
                },
                policy_mapping_fn=matchmaking.Matchmaker(
                    [
                        # 90% of the time, Policy A vs. Policy A
                        matchmaking.Matchup(
                            "policy_a",
                            "policy_a",
                            0.9,
                        ),
                        # 10% of the time, Policy A vs. Policy B
                        matchmaking.Matchup(
                            "policy_b",
                            "policy_b",
                            0.1,
                        ),
                    ]
                ).policy_mapping_fn,
                policies_to_train=["focal_policy"],
            )
            .rl_module(
                rl_module_spec=multi_rl_module.MultiRLModuleSpec(
                    rl_module_specs={
                        "policy_a": rl_module.RLModuleSpec(
                            model_config={
                                "fcnet_hiddens": [256, 256],
                                "fcnet_activation": "relu",
                            },
                        ),
                        "policy_b": rl_module.RLModuleSpec(
                            model_config={
                                "fcnet_hiddens": [256, 256],
                                "fcnet_activation": "relu",
                            },
                        ),
                    },
                )
            )
            .evaluation(
                evaluation_num_env_runners=(1 if debug else 1),
                evaluation_interval=1,
                evaluation_duration="auto",
                evaluation_duration_unit="episodes",
                evaluation_parallel_to_training=True,
                evaluation_config={
                    "env_config": {"evaluation": True},
                },
            )
            .callbacks(callbacks_class=WinratesV2)
        )

        return config

    def _training_params(self):
        return dict(
            train_batch_size=32_768,
            minibatch_size=2048,
            lr=[[0, 0.001], [TOTAL_ENV_STEPS, 0]],
            entropy_coeff=0.01,
            gamma=0.995,
            vf_loss_coeff=1.0,
            lambda_=0.95,
        )

    def env_creator(self, config, **kwargs):
        return footsiesgym.make(config=config, rllib=True)

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
            trainable=ppo.PPO,
            param_space=model_config,
            tune_config=tune_config,
            run_config=run_config,
        )

        tuner.fit()
