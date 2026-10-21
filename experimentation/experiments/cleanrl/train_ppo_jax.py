# Self-play PPO in JAX on the C/CUDA FOOTSIES environment, in the style of
# CleanRL's ppo_atari_envpool_xla_jax_scan.py
# (https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jax_scanpy).
#
# The environment is footsiesgym.make(backend="c", functional=True): its step
# is an XLA call, so each PPO update (rollout, GAE and all minibatch updates)
# is one jitted function on the GPU. Both players share one policy, as in
# train_ppo.py. The trained parameters can be played back in the Unity game
# with play_unity.py.
#
#   python -m experimentation.experiments.cleanrl.train_ppo_jax
#   python -m experimentation.experiments.cleanrl.train_ppo_jax --total-timesteps 500000000 --track
import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Sequence

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

import footsiesgym

AGENTS = ("p1", "p2")


@dataclasses.dataclass
class Args:
    seed: int = 1
    total_timesteps: int = 200_000_000
    """agent steps (2 per game step: both players learn)"""
    num_envs: int = 4096
    """games stepped together (2 agents each)"""
    num_steps: int = 128
    """steps per rollout"""
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 256
    eval_interval: int = 10
    """updates between evaluations against a random opponent"""
    eval_envs: int = 2048
    eval_steps: int = 1000
    output_dir: str = "runs/ppo_jax"
    track: bool = False
    """log to Weights & Biases"""
    wandb_project: str = "Footsies-v0"
    # The environment (FootsiesEnv config keys)
    frame_skip: int = 4
    action_delay: int = 8
    max_t: int = 4000
    win_reward_scaling_coeff: float = 10.0
    guard_break_reward: float = 0.0
    use_special_charge_action: bool = False


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(Args):
        flag = "--" + field.name.replace("_", "-")
        if field.type is bool:
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=field.default
            )
        else:
            parser.add_argument(flag, type=field.type, default=field.default)
    return Args(**vars(parser.parse_args()))


def env_config(args: Args) -> dict:
    keys = (
        "frame_skip",
        "action_delay",
        "max_t",
        "win_reward_scaling_coeff",
        "guard_break_reward",
        "use_special_charge_action",
    )
    return {k: getattr(args, k) for k in keys}


class Network(nn.Module):
    """Shared torso, policy logits and value (as in train_ppo.py)."""

    action_dim: int
    hidden_sizes: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_sizes:
            x = nn.Dense(size, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
            x = nn.relu(x)
        logits = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.orthogonal(0.01)
        )(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return logits, value[..., 0]


def merge(per_agent: dict) -> jax.Array:
    """{"p1": (N, ...), "p2": (N, ...)} -> (2N, ...): p1 slots, then p2."""
    return jnp.concatenate([per_agent["p1"], per_agent["p2"]], axis=0)


def split(x: jax.Array, n: int) -> dict:
    return {"p1": x[:n], "p2": x[n:]}


def make_train(args: Args, env, network: Network):
    n = env.num_envs
    slots = 2 * n
    batch_size = slots * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size

    def schedule(count):
        frac = (
            1.0 - (count // (args.num_minibatches * args.update_epochs)) / num_updates
        )
        return args.learning_rate * frac

    def init(key):
        key, net_key = jax.random.split(key)
        params = network.init(net_key, jnp.zeros((1, 88), jnp.float32))
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(schedule if args.anneal_lr else args.learning_rate, eps=1e-5),
        )
        state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        obs, env_state, _ = env.reset(None)
        stats = dict(returns=jnp.zeros(slots), lengths=jnp.zeros(slots, jnp.int32))
        return state, (env_state, merge(obs), stats), key

    def rollout(state, carry, key):
        def step(carry, key):
            env_state, obs, stats = carry
            logits, value = network.apply(state.params, obs)
            action = jax.random.categorical(key, logits)
            logprob = jax.nn.log_softmax(logits)[jnp.arange(slots), action]
            next_obs, env_state, reward, term, trunc, _ = env.step(
                None, env_state, split(action, n)
            )
            reward = merge(reward)
            done = jnp.concatenate([term["p1"] | trunc["p1"]] * 2)
            returns = stats["returns"] + reward
            lengths = stats["lengths"] + 1
            finished = dict(
                count=done.sum(),
                return_sum=jnp.where(done, returns, 0.0).sum(),
                length_sum=jnp.where(done, lengths, 0).sum(),
            )
            stats = dict(
                returns=jnp.where(done, 0.0, returns),
                lengths=jnp.where(done, 0, lengths),
            )
            transition = dict(
                obs=obs,
                action=action,
                logprob=logprob,
                value=value,
                reward=reward,
                done=done,
            )
            return (env_state, merge(next_obs), stats), (transition, finished)

        keys = jax.random.split(key, args.num_steps)
        return jax.lax.scan(step, carry, keys)

    def gae(traj, last_value):
        def body(carry, t):
            next_adv, next_value = carry
            nonterminal = 1.0 - t["done"]
            delta = t["reward"] + args.gamma * next_value * nonterminal - t["value"]
            adv = delta + args.gamma * args.gae_lambda * nonterminal * next_adv
            return (adv, t["value"]), adv

        _, adv = jax.lax.scan(
            body, (jnp.zeros_like(last_value), last_value), traj, reverse=True
        )
        return adv, adv + traj["value"]

    def loss_fn(params, mb):
        logits, value = network.apply(params, mb["obs"])
        logp_all = jax.nn.log_softmax(logits)
        logprob = jnp.take_along_axis(logp_all, mb["action"][:, None], 1)[:, 0]
        logratio = logprob - mb["logprob"]
        ratio = jnp.exp(logratio)
        adv = mb["adv"]
        if args.norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg_loss = jnp.maximum(
            -adv * ratio,
            -adv * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
        ).mean()
        if args.clip_vloss:
            v_clipped = mb["value"] + jnp.clip(
                value - mb["value"], -args.clip_coef, args.clip_coef
            )
            v_loss = (
                0.5
                * jnp.maximum(
                    (value - mb["ret"]) ** 2, (v_clipped - mb["ret"]) ** 2
                ).mean()
            )
        else:
            v_loss = 0.5 * ((value - mb["ret"]) ** 2).mean()
        entropy = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss
        approx_kl = ((ratio - 1) - logratio).mean()
        return loss, dict(
            policy_loss=pg_loss, value_loss=v_loss, entropy=entropy, approx_kl=approx_kl
        )

    @jax.jit
    def update(state, carry, key):
        key, roll_key, perm_key = jax.random.split(key, 3)
        carry, (traj, finished) = rollout(state, carry, roll_key)
        _, last_value = network.apply(state.params, carry[1])
        adv, ret = gae(traj, last_value)
        batch = dict(traj, adv=adv, ret=ret)
        del batch["reward"], batch["done"]
        batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)

        def epoch(state, key):
            perm = jax.random.permutation(key, batch_size)
            mbs = jax.tree.map(
                lambda x: x[perm].reshape(
                    (args.num_minibatches, minibatch_size) + x.shape[1:]
                ),
                batch,
            )

            def minibatch(state, mb):
                (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    state.params, mb
                )
                return state.apply_gradients(grads=grads), metrics

            return jax.lax.scan(minibatch, state, mbs)

        state, metrics = jax.lax.scan(
            epoch, state, jax.random.split(perm_key, args.update_epochs)
        )
        metrics = jax.tree.map(jnp.mean, metrics)
        metrics.update(jax.tree.map(jnp.sum, finished))
        return state, carry, key, metrics

    return init, update, num_updates, batch_size


def make_evaluate(args: Args, eval_env, network: Network):
    """Win, loss and tie rates of the policy (p1) against uniform random
    actions (p2), over every episode that ends in ``eval_steps`` steps."""
    n, num_actions = eval_env.num_envs, eval_env.action_dim

    @jax.jit
    def evaluate(params, key):
        obs, env_state, _ = eval_env.reset(None)

        def step(carry, key):
            env_state, obs = carry
            k1, k2 = jax.random.split(key)
            logits, _ = network.apply(params, obs["p1"])
            actions = {
                "p1": jax.random.categorical(k1, logits),
                "p2": jax.random.randint(k2, (n,), 0, num_actions),
            }
            obs, env_state, reward, term, trunc, _ = eval_env.step(
                None, env_state, actions
            )
            done = term["p1"] | trunc["p1"]
            r = reward["p1"]
            counts = jnp.stack(
                [
                    (done & (r > 0)).sum(),
                    (done & (r < 0)).sum(),
                    (done & (r == 0)).sum(),
                ]
            )
            return (env_state, obs), counts

        _, counts = jax.lax.scan(
            step, (env_state, obs), jax.random.split(key, args.eval_steps)
        )
        wins, losses, ties = counts.sum(0)
        total = jnp.maximum(wins + losses + ties, 1)
        return dict(
            win_rate_vs_random=wins / total,
            loss_rate_vs_random=losses / total,
            tie_rate_vs_random=ties / total,
            episodes=total,
        )

    return evaluate


def save(path: Path, params, args: Args):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(flax.serialization.to_bytes(params))
    meta = dict(
        hidden_size=args.hidden_size,
        action_dim=6 + 3 * args.use_special_charge_action,
        env_config=env_config(args),
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load(path: Path):
    """(network, params, env_config) of a policy saved by this script."""
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    network = Network(meta["action_dim"], (meta["hidden_size"],) * 2)
    template = network.init(jax.random.key(0), jnp.zeros((1, 88), jnp.float32))
    params = flax.serialization.from_bytes(template, Path(path).read_bytes())
    return network, params, meta["env_config"]


def main():
    args = parse_args()
    config = env_config(args)
    env = footsiesgym.make(
        backend="c", functional=True, num_envs=args.num_envs, **config
    )
    eval_env = footsiesgym.make(
        backend="c", functional=True, num_envs=args.eval_envs, **config
    )
    network = Network(env.action_dim, (args.hidden_size,) * 2)
    init, update, num_updates, batch_size = make_train(args, env, network)
    evaluate = make_evaluate(args, eval_env, network)
    print(
        f"JAX device: {jax.devices()[0]}; {num_updates} updates of {batch_size} steps"
    )

    run = None
    if args.track:
        import wandb

        run = wandb.init(project=args.wandb_project, config=dataclasses.asdict(args))

    def log(metrics: dict, step: int):
        metrics = {k: float(v) for k, v in metrics.items()}
        if run is not None:
            run.log(metrics, step=step)
        return metrics

    state, carry, key = init(jax.random.key(args.seed))
    eval_key = jax.random.key(args.seed + 1)
    out = Path(args.output_dir)
    m = log({f"eval/{k}": v for k, v in evaluate(state.params, eval_key).items()}, 0)
    print(f"update 0: win rate vs random {m['eval/win_rate_vs_random']:.3f}")

    start = time.time()
    for i in range(1, num_updates + 1):
        state, carry, key, metrics = update(state, carry, key)
        global_step = i * batch_size
        if i % args.eval_interval == 0 or i == num_updates:
            metrics = jax.device_get(metrics)
            eval_key, k = jax.random.split(eval_key)
            ev = jax.device_get(evaluate(state.params, k))
            sps = global_step / (time.time() - start)
            done = max(int(metrics.pop("count")), 1)
            m = log(
                {
                    "charts/SPS": sps,
                    "charts/episodic_return": metrics.pop("return_sum") / done,
                    "charts/episodic_length": metrics.pop("length_sum") / done,
                    **{f"losses/{k}": v for k, v in metrics.items()},
                    **{f"eval/{k}": v for k, v in ev.items()},
                },
                global_step,
            )
            print(
                f"update {i}/{num_updates} step {global_step:,}: "
                f"{sps / 1e6:.1f}M SPS, episode length {m['charts/episodic_length']:.0f}, "
                f"win rate vs random {m['eval/win_rate_vs_random']:.3f} "
                f"(loss {m['eval/loss_rate_vs_random']:.3f}, "
                f"{int(m['eval/episodes'])} episodes)"
            )
            save(out / "policy.msgpack", state.params, args)
    print(f"saved {out / 'policy.msgpack'}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
