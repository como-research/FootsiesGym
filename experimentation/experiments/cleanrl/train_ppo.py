# Adapted from CleanRL's ppo_pettingzoo_ma_atari.py for FootsiesGym.
# Original: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
import concurrent.futures
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import footsiesgym
import wandb

# ============================================================================
# Default config — wandb sweeps override these via wandb.config
# ============================================================================
DEFAULT_CONFIG = {
    # Experiment
    "seed": 1,
    "torch_deterministic": True,
    "cuda": True,
    # PPO
    "total_timesteps": 10_000_000,
    "learning_rate": 6e-4,
    "num_envs": 48,  # agent slots (2 per game instance)
    "num_steps": 256,  # steps per rollout per env
    "anneal_lr": False,
    "anneal_ent": False,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 4,
    "update_epochs": 4,
    "norm_adv": True,
    "clip_coef": 0.2,
    "clip_vloss": True,
    "ent_coef": 0.05,
    "vf_coef": 1.0,
    "max_grad_norm": 0.5,
    "target_kl": None,
    # Evaluation
    "num_eval_envs": 48,
    "eval_episodes": 250,
    "eval_interval": 25,  # evaluate every N updates
}


def make_env():
    """Create a single FootsiesGym ParallelEnv."""
    return footsiesgym.make(
        config={
            "frame_skip": 4,
            "action_delay": 16,
            "max_t": 4000,
            "use_special_charge_action": True,
            "win_reward_scaling_coeff": 10.0,
            "guard_break_reward": 0.0,
            "headless": True,
        },
        launch_binaries=True,
    )


class FootsiesVecEnv(gym.Env):
    """Vectorized wrapper over multiple FootsiesEnv (PettingZoo ParallelEnv) instances.

    Each game instance has 2 agents (p1, p2). This wrapper flattens them into
    a single VecEnv-like interface where slot 2*i is p1 of game i and slot 2*i+1
    is p2 of game i. gRPC calls are parallelized with threads since they
    release the GIL.
    """

    def __init__(self, num_game_instances: int):
        super().__init__()
        self.envs = [make_env() for _ in range(num_game_instances)]
        self.num_game_instances = num_game_instances
        self.num_agents = 2  # p1, p2
        self.num_envs = num_game_instances * self.num_agents
        self.agents = ["p1", "p2"]
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_game_instances
        )

        # Spaces (same for all agents)
        sample_env = self.envs[0]
        self.single_observation_space = sample_env.observation_space("p1")
        self.single_action_space = sample_env.action_space("p1")
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.is_vector_env = True

        # Cumulative episode returns and lengths per slot
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, seed=None, options=None):
        all_obs = np.zeros(
            (self.num_envs,) + self.single_observation_space.shape,
            dtype=np.float32,
        )
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0

        def _reset(i):
            return i, self.envs[i].reset(seed=seed, options=options)

        for i, (obs, _) in self._pool.map(
            _reset, range(self.num_game_instances)
        ):
            all_obs[2 * i] = obs["p1"]
            all_obs[2 * i + 1] = obs["p2"]
        return all_obs, {}

    def step(self, actions):
        all_obs = np.zeros(
            (self.num_envs,) + self.single_observation_space.shape,
            dtype=np.float32,
        )
        all_rewards = np.zeros(self.num_envs, dtype=np.float32)
        all_terminated = np.zeros(self.num_envs, dtype=bool)
        all_truncated = np.zeros(self.num_envs, dtype=bool)
        all_infos = {}
        final_infos = [None] * self.num_envs

        def _step(i):
            env = self.envs[i]
            action_dict = {
                "p1": int(actions[2 * i]),
                "p2": int(actions[2 * i + 1]),
            }
            obs, rewards, terminateds, truncateds, _ = env.step(action_dict)
            done = terminateds["p1"] or truncateds["p1"]
            if done:
                obs, _ = env.reset()
            return i, obs, rewards, terminateds, truncateds, done

        for i, obs, rewards, terminateds, truncateds, done in self._pool.map(
            _step, range(self.num_game_instances)
        ):
            for j, agent in enumerate(self.agents):
                slot = 2 * i + j
                all_obs[slot] = obs[agent]
                all_rewards[slot] = rewards[agent]
                all_terminated[slot] = terminateds.get(agent, False)
                all_truncated[slot] = truncateds.get(agent, False)

                # Track cumulative episode stats
                self._episode_returns[slot] += rewards[agent]
                self._episode_lengths[slot] += 1

                if done:
                    final_infos[slot] = {
                        "episode": {
                            "r": self._episode_returns[slot],
                            "l": self._episode_lengths[slot],
                        }
                    }
                    self._episode_returns[slot] = 0.0
                    self._episode_lengths[slot] = 0

        if any(f is not None for f in final_infos):
            all_infos["final_info"] = final_infos

        return all_obs, all_rewards, all_terminated, all_truncated, all_infos

    def close(self):
        self._pool.shutdown(wait=False)
        for env in self.envs:
            env.close()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_size = np.array(envs.single_observation_space.shape).prod()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.actor = layer_init(
            nn.Linear(256, envs.single_action_space.n), std=0.01
        )
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
        )


@torch.no_grad()
def evaluate_vs_random(agent, eval_envs, device, num_episodes):
    """Play the agent (as p1) vs random (p2) across parallel envs.

    Runs episodes across all eval_envs concurrently until num_episodes
    are completed. Returns win/loss/tie rates and mean episode length.
    """
    n = len(eval_envs)
    num_actions = eval_envs[0].action_space("p2").n
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=n)

    wins, losses, ties = 0, 0, 0
    total_length = 0
    episodes_done = 0

    # Reset all envs in parallel
    def _reset(i):
        return i, eval_envs[i].reset()

    all_obs = [None] * n
    active = [True] * n
    for i, (obs, _) in pool.map(_reset, range(n)):
        all_obs[i] = obs

    while episodes_done < num_episodes:
        # Batch inference for all active envs
        active_indices = [i for i in range(n) if active[i]]
        if not active_indices:
            break

        p1_obs_batch = torch.Tensor(
            np.stack([all_obs[i]["p1"] for i in active_indices])
        ).to(device)
        p1_actions, _, _, _ = agent.get_action_and_value(p1_obs_batch)
        p1_actions = p1_actions.cpu().numpy()

        # Build action dicts
        action_dicts = {}
        for idx, i in enumerate(active_indices):
            action_dicts[i] = {
                "p1": int(p1_actions[idx]),
                "p2": np.random.randint(num_actions),
            }

        # Step all active envs in parallel
        def _step(i):
            obs, rewards, terminateds, truncateds, _ = eval_envs[i].step(
                action_dicts[i]
            )
            done = terminateds["p1"] or truncateds["p1"]
            return i, obs, rewards, done, eval_envs[i].t

        for i, obs, rewards, done, ep_len in pool.map(_step, active_indices):
            if done:
                episodes_done += 1
                total_length += ep_len
                if rewards["p1"] > 0:
                    wins += 1
                elif rewards["p1"] < 0:
                    losses += 1
                else:
                    ties += 1

                if episodes_done >= num_episodes:
                    active[i] = False
                else:
                    # Reset and continue
                    all_obs[i], _ = eval_envs[i].reset()
            else:
                all_obs[i] = obs

    pool.shutdown(wait=False)
    return {
        "eval/win_rate_vs_random": wins / num_episodes,
        "eval/loss_rate_vs_random": losses / num_episodes,
        "eval/tie_rate_vs_random": ties / num_episodes,
        "eval/mean_episode_length": total_length / num_episodes,
    }


def train():
    run = wandb.init()
    config = {**DEFAULT_CONFIG, **dict(wandb.config)}

    # Derived
    num_envs = config["num_envs"]
    num_steps = config["num_steps"]
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // config["num_minibatches"]

    # Seeding
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = config["torch_deterministic"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["cuda"] else "cpu"
    )

    # Env setup
    # Each game instance has 2 agents (p1, p2), flattened into num_envs VecEnv slots.
    assert (
        num_envs % 2 == 0
    ), "num_envs must be even (2 agents per game instance)"
    num_game_instances = num_envs // 2
    envs = FootsiesVecEnv(num_game_instances)
    eval_envs = [make_env() for _ in range(config["num_eval_envs"])]
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(
        agent.parameters(), lr=config["learning_rate"], eps=1e-5
    )

    # Storage setup
    obs = torch.zeros(
        (num_steps, num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (num_steps, num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config["seed"])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = config["total_timesteps"] // batch_size

    try:
        for update in range(1, num_updates + 1):
            frac = 1.0 - (update - 1.0) / num_updates
            if config["anneal_lr"]:
                optimizer.param_groups[0]["lr"] = (
                    frac * config["learning_rate"]
                )
            ent_coef = (
                config["ent_coef"] * frac
                if config["anneal_ent"]
                else config["ent_coef"]
            )

            for step in range(0, num_steps):
                global_step += num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminated, truncated, infos = envs.step(
                    action.cpu().numpy()
                )
                done = np.logical_or(terminated, truncated)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    device
                ), torch.Tensor(done).to(device)

                if "final_info" in infos:
                    for idx, info in enumerate(infos["final_info"]):
                        if info is None:
                            continue
                        if "episode" in info:
                            player_idx = idx % 2
                            print(
                                f"global_step={global_step}, p{player_idx+1}-episodic_return={info['episode']['r']:.2f}"
                            )
                            wandb.log(
                                {
                                    f"charts/episodic_return-p{player_idx+1}": info[
                                        "episode"
                                    ][
                                        "r"
                                    ],
                                    f"charts/episodic_length-p{player_idx+1}": info[
                                        "episode"
                                    ][
                                        "l"
                                    ],
                                    "global_step": global_step,
                                }
                            )

            # Bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + config["gamma"] * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + config["gamma"]
                        * config["gae_lambda"]
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + values

            # Flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimize policy and value network
            b_inds = np.arange(batch_size)
            clipfracs = []
            for epoch in range(config["update_epochs"]):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = (
                        agent.get_action_and_value(
                            b_obs[mb_inds], b_actions.long()[mb_inds]
                        )
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > config["clip_coef"])
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if config["norm_adv"]:
                        mb_advantages = (
                            mb_advantages - mb_advantages.mean()
                        ) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - config["clip_coef"], 1 + config["clip_coef"]
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if config["clip_vloss"]:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -config["clip_coef"],
                            config["clip_coef"],
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(
                            v_loss_unclipped, v_loss_clipped
                        )
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = (
                            0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                        )

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - ent_coef * entropy_loss
                        + v_loss * config["vf_coef"]
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        agent.parameters(), config["max_grad_norm"]
                    )
                    optimizer.step()

                if config["target_kl"] is not None:
                    if approx_kl > config["target_kl"]:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            wandb.log(
                {
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": int(
                        global_step / (time.time() - start_time)
                    ),
                    "global_step": global_step,
                }
            )
            print("SPS:", int(global_step / (time.time() - start_time)))

            # Evaluation vs random
            if update % config["eval_interval"] == 0 or update == num_updates:
                eval_metrics = evaluate_vs_random(
                    agent, eval_envs, device, config["eval_episodes"]
                )
                eval_metrics["global_step"] = global_step
                wandb.log(eval_metrics)
                print(
                    f"  eval win_rate_vs_random={eval_metrics['eval/win_rate_vs_random']:.2f}"
                )

    finally:
        envs.close()
        for ev in eval_envs:
            ev.close()
        run.finish()


if __name__ == "__main__":
    # Standalone: python train_ppo.py
    # Sweep:      wandb sweep sweep.yaml && wandb agent <sweep_id>
    train()
