# Adapted from CleanRL's ppo_pettingzoo_ma_atari.py for FootsiesGym.
# Original: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
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
    "learning_rate": 0.001,
    "num_envs": 96,  # agent slots (2 per game instance)
    "num_steps": 2048,  # steps per rollout per env
    "anneal_lr": True,
    "anneal_ent": False,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "num_minibatches": 8,
    "update_epochs": 8,
    "norm_adv": True,
    "clip_coef": 0.3,
    "clip_vloss": True,
    "ent_coef": 0.01,
    "vf_coef": 1.0,
    "max_grad_norm": 0.5,
    "target_kl": None,
    # Network
    "hidden_size": 256,
    # Evaluation
    "num_eval_envs": 96,
    "eval_episodes": 250,
    "eval_interval": 1,  # evaluate every N updates
}

ENV_CONFIG = {
    "frame_skip": 4,
    "action_delay": 8,
    "max_t": 4000,
    "use_special_charge_action": False,
    "win_reward_scaling_coeff": 10.0,
    "guard_break_reward": 0.0,
    "headless": True,
}


def make_vec_env(num_game_instances: int):
    """Create a single vectorized FootsiesEnv with N game instances."""
    return footsiesgym.make(
        config={**ENV_CONFIG, "num_envs": num_game_instances},
        launch_binaries=True,
    )


class FootsiesVecEnv(gym.Env):
    """Thin wrapper that flattens a vectorized FootsiesEnv's p1/p2
    dict outputs into flat agent slots for CleanRL's PPO loop.

    Slot layout: [p1_env0, p1_env1, ..., p1_envN-1,
                  p2_env0, p2_env1, ..., p2_envN-1]
    """

    def __init__(self, num_game_instances: int):
        super().__init__()
        self.env = make_vec_env(num_game_instances)
        self.num_game_instances = num_game_instances
        self.num_envs = num_game_instances * 2  # p1 + p2 slots
        self.N = num_game_instances

        self.single_observation_space = self.env.observation_space("p1")
        self.single_action_space = self.env.action_space("p1")
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.is_vector_env = True

        # Episode tracking per slot
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        # Concatenate p1 block and p2 block
        flat_obs = np.concatenate([obs["p1"], obs["p2"]], axis=0).astype(
            np.float32
        )
        return flat_obs, {}

    def step(self, actions):
        N = self.N
        # Split flat actions back into p1/p2 arrays
        action_dict = {
            "p1": actions[:N].astype(np.int64),
            "p2": actions[N:].astype(np.int64),
        }
        obs, rewards, terminateds, truncateds, _ = self.env.step(action_dict)

        flat_obs = np.concatenate([obs["p1"], obs["p2"]], axis=0).astype(
            np.float32
        )
        flat_rewards = np.concatenate([rewards["p1"], rewards["p2"]]).astype(
            np.float32
        )
        flat_terminated = np.concatenate([terminateds["p1"], terminateds["p2"]])
        flat_truncated = np.concatenate([truncateds["p1"], truncateds["p2"]])
        flat_done = flat_terminated | flat_truncated

        # Track episode stats
        self._episode_returns += flat_rewards
        self._episode_lengths += 1

        infos = {}
        if flat_done.any():
            final_infos = [None] * self.num_envs
            for slot in np.where(flat_done)[0]:
                final_infos[slot] = {
                    "episode": {
                        "r": self._episode_returns[slot],
                        "l": self._episode_lengths[slot],
                    }
                }
            infos["final_info"] = final_infos
            # Reset counters for done slots
            self._episode_returns[flat_done] = 0.0
            self._episode_lengths[flat_done] = 0

        return (
            flat_obs,
            flat_rewards,
            flat_terminated,
            flat_truncated,
            infos,
        )

    def close(self):
        self.env.close()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, hidden_size=256):
        super().__init__()
        obs_size = np.array(envs.single_observation_space.shape).prod()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )
        self.actor = layer_init(
            nn.Linear(hidden_size, envs.single_action_space.n),
            std=0.01,
        )
        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1)

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
def evaluate_vs_random(agent, eval_env, device, num_episodes):
    """Play the agent (as p1) vs random (p2) using a vectorized env.

    Runs until num_episodes are completed. Returns win/loss/tie rates
    and mean episode length.
    """
    N = eval_env.num_envs  # game instances
    num_actions = eval_env.action_space("p1").n

    wins, losses, ties = 0, 0, 0
    total_length = 0
    episodes_done = 0

    obs, _ = eval_env.reset()
    ep_lengths = np.zeros(N, dtype=np.int64)

    while episodes_done < num_episodes:
        ep_lengths += 1

        # Agent policy for p1
        p1_obs = torch.Tensor(obs["p1"]).to(device)
        p1_actions, _, _, _ = agent.get_action_and_value(p1_obs)
        p1_actions = p1_actions.cpu().numpy()

        # Random policy for p2
        p2_actions = np.random.randint(num_actions, size=N)

        action_dict = {"p1": p1_actions, "p2": p2_actions}
        obs, rewards, terminateds, truncateds, _ = eval_env.step(action_dict)

        done = terminateds["p1"] | truncateds["p1"]
        for i in np.where(done)[0]:
            if episodes_done >= num_episodes:
                break
            episodes_done += 1
            total_length += ep_lengths[i]
            r = rewards["p1"][i]
            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1
            else:
                ties += 1
            ep_lengths[i] = 0

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

    # Env setup — single server with N vectorized game instances
    assert (
        num_envs % 2 == 0
    ), "num_envs must be even (2 agents per game instance)"
    num_game_instances = num_envs // 2
    envs = FootsiesVecEnv(num_game_instances)

    # Eval env — single vectorized env, uses PettingZoo dict API directly
    eval_env = make_vec_env(config["num_eval_envs"])

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs, hidden_size=config["hidden_size"]).to(device)
    optimizer = optim.Adam(
        agent.parameters(),
        lr=config["learning_rate"],
        eps=1e-5,
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
        # Initial evaluation before any training
        eval_metrics = evaluate_vs_random(
            agent, eval_env, device, config["eval_episodes"]
        )
        eval_metrics["global_step"] = 0
        wandb.log(eval_metrics)
        print(
            f"  initial eval win_rate_vs_random="
            f"{eval_metrics['eval/win_rate_vs_random']:.2f}"
        )

        for update in range(1, num_updates + 1):
            frac = 1.0 - (update - 1.0) / num_updates
            if config["anneal_lr"]:
                optimizer.param_groups[0]["lr"] = frac * config["learning_rate"]
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

                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    infos,
                ) = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = (
                    torch.Tensor(next_obs).to(device),
                    torch.Tensor(done).to(device),
                )

                if "final_info" in infos:
                    for idx, info in enumerate(infos["final_info"]):
                        if info is None:
                            continue
                        if "episode" in info:
                            player = "p1" if idx < num_game_instances else "p2"
                            print(
                                f"global_step={global_step}, "
                                f"{player}-episodic_return="
                                f"{info['episode']['r']:.2f}"
                            )
                            wandb.log(
                                {
                                    f"charts/episodic_return-{player}": info[
                                        "episode"
                                    ]["r"],
                                    f"charts/episodic_length-{player}": info[
                                        "episode"
                                    ]["l"],
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
                            b_obs[mb_inds],
                            b_actions.long()[mb_inds],
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
                        ratio,
                        1 - config["clip_coef"],
                        1 + config["clip_coef"],
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
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
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
                        agent.parameters(),
                        config["max_grad_norm"],
                    )
                    optimizer.step()

                if config["target_kl"] is not None:
                    if approx_kl > config["target_kl"]:
                        break

            y_pred, y_true = (
                b_values.cpu().numpy(),
                b_returns.cpu().numpy(),
            )
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
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                    "global_step": global_step,
                }
            )
            print(
                "SPS:",
                int(global_step / (time.time() - start_time)),
            )

            # Evaluation vs random
            if update % config["eval_interval"] == 0 or update == num_updates:
                eval_metrics = evaluate_vs_random(
                    agent,
                    eval_env,
                    device,
                    config["eval_episodes"],
                )
                eval_metrics["global_step"] = global_step
                wandb.log(eval_metrics)
                print(
                    f"  eval win_rate_vs_random="
                    f"{eval_metrics['eval/win_rate_vs_random']:.2f}"
                )

    finally:
        envs.close()
        eval_env.close()
        run.finish()


if __name__ == "__main__":
    # Standalone: python train_ppo.py
    # Sweep:      wandb sweep sweep.yaml && wandb agent <sweep_id>
    train()
