# Play back a policy from train_ppo_jax.py in the Unity game.
#
# The policy (p1) plays against uniform random actions (p2) on the Unity game
# server. The C environment gets the same actions at each step, and the script
# checks that the observations, rewards and done flags of the two are
# identical, byte for byte: the policy behaves in Unity as it did in training.
# (Each game is compared through its first episode: on reset, the Unity server
# keeps some fighter fields from the previous episode.) The script also gives
# the win rate of the policy in Unity.
#
#   python -m experimentation.experiments.cleanrl.play_unity --policy runs/ppo_jax/policy.msgpack
#   python -m experimentation.experiments.cleanrl.play_unity --policy ... --windowed   # watch one game
import argparse
import time

import jax
import numpy as np

import footsiesgym
from experimentation.experiments.cleanrl.train_ppo_jax import load

AGENTS = ("p1", "p2")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True, help="policy.msgpack from train_ppo_jax")
    p.add_argument("--num-envs", type=int, default=64, help="Unity games")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--windowed",
        action="store_true",
        help="play one game in the Unity window, at game speed (no comparison)",
    )
    return p.parse_args()


def make_policy(path, seed):
    network, params, config = load(path)
    apply = jax.jit(
        lambda obs, key: jax.random.categorical(key, network.apply(params, obs)[0])
    )
    key = [jax.random.key(seed)]

    def act(obs: np.ndarray) -> np.ndarray:
        key[0], k = jax.random.split(key[0])
        return np.asarray(apply(np.asarray(obs, np.float32), k))

    return act, config


def bits(x) -> np.ndarray:
    x = np.asarray(x)
    return x.view(np.int32) if x.dtype == np.float32 else x


def compare(args, act, config):
    n, rng = args.num_envs, np.random.default_rng(args.seed)
    if n < 2:
        raise SystemExit("--num-envs must be at least 2 (or use --windowed)")
    unity = footsiesgym.make(num_envs=n, **config)
    sim = footsiesgym.make(backend="c", num_envs=n, **config)
    num_actions = unity.action_space("p2").n
    try:
        obs, _ = unity.reset()
        sim_obs, _ = sim.reset()
        for a in AGENTS:
            assert (bits(obs[a]) == bits(sim_obs[a])).all(), "reset observations differ"
        live = np.ones(n, bool)  # games still in their first episode
        outcomes = np.zeros(3, int)  # wins, losses, ties of p1 in Unity
        checked = 0
        for t in range(args.steps):
            actions = {"p1": act(obs["p1"]), "p2": rng.integers(0, num_actions, n)}
            obs, rew, term, trunc, _ = unity.step(actions)
            sim_obs, sim_rew, sim_term, sim_trunc, _ = sim.step(actions)
            done = term["p1"] | trunc["p1"]
            for a in AGENTS:
                same_outcome = (
                    (bits(rew[a].astype(np.float32)) == bits(sim_rew[a]))
                    & (term[a] == sim_term[a])
                    & (trunc[a] == sim_trunc[a])
                )
                same_obs = (bits(obs[a]) == bits(sim_obs[a])).all(axis=1)
                # A finished game starts a new one in the same step: its new
                # observation comes from Unity's reset, which is not a fresh
                # game, so it is not compared.
                bad = live & (~same_outcome | (~same_obs & ~done))
                if bad.any():
                    i = int(np.flatnonzero(bad)[0])
                    raise SystemExit(
                        f"step {t}: game {i} ({a}) differs between Unity and C"
                    )
            checked += int(live.sum())
            r = rew["p1"][done]
            outcomes += [(r > 0).sum(), (r < 0).sum(), (r == 0).sum()]
            live &= ~done
    finally:
        unity.close()
    episodes = max(outcomes.sum(), 1)
    print(
        f"OK: Unity and C identical over {checked:,} game steps "
        f"(first episode of {n} games, {int((~live).sum())} of them finished)"
    )
    print(
        f"Unity, policy vs random over {outcomes.sum()} episodes: "
        f"win rate {outcomes[0] / episodes:.3f}, loss {outcomes[1] / episodes:.3f}, "
        f"tie {outcomes[2] / episodes:.3f}"
    )


def watch(args, act, config):
    env = footsiesgym.make(headless=False, **config)
    rng = np.random.default_rng(args.seed)
    num_actions = env.action_space("p2").n
    frame_time = config.get("frame_skip", 4) / 60  # the game runs at 60 frames/s
    try:
        obs, _ = env.reset()
        for _ in range(args.steps):
            start = time.time()
            actions = {
                "p1": int(act(obs["p1"][None])[0]),
                "p2": int(rng.integers(num_actions)),
            }
            obs, rew, term, trunc, _ = env.step(actions)
            if term["p1"] or trunc["p1"]:
                print(f"episode ended: reward of the policy {rew['p1']:+.1f}")
                obs, _ = env.reset()
            time.sleep(max(0.0, frame_time - (time.time() - start)))
    finally:
        env.close()


def main():
    args = parse_args()
    act, config = make_policy(args.policy, args.seed)
    if args.windowed:
        watch(args, act, config)
    else:
        compare(args, act, config)


if __name__ == "__main__":
    main()
