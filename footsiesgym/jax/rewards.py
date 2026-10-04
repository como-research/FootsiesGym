"""Rewards matching ``FootsiesEnv._step_vectorized`` bit for bit.

``FootsiesEnv`` accumulates rewards in float64. A step's rewards depend only
on whose guard broke, who died and how many guard-break deductions each
reward budget has had, so ``RewardFunction`` evaluates that exact float64
operation sequence once for every combination and looks rewards up.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

# At most three guard-health decreases per player per episode (3 -> 0).
MAX_BUDGET_DEDUCTIONS = 3


class RewardFunction:
    def __init__(
        self, win_reward: float, guard_break_reward: float, use_reward_budget: bool
    ):
        assert guard_break_reward * 3 < win_reward, (
            "Guard break reward total must be less than the win "
            "reward (guard break reward * 3 < win reward)"
        )
        self.win_reward = float(win_reward)
        self.guard_break_reward = float(guard_break_reward)
        self.use_reward_budget = bool(use_reward_budget)
        self.table = self._build_table()

    def __call__(self, last_guard_health, guard_health, dead, deductions):
        """Rewards of one step.

        :param last_guard_health: (2,) guard health before the step.
        :param guard_health: (2,) guard health after the step.
        :param dead: (2,) whether each player is dead.
        :param deductions: (2,) budget deductions so far this episode.
        :return: ((2,) rewards of P1 and P2, (2,) updated deductions).
        """
        if self.guard_break_reward != 0:
            guard_broken = guard_health < last_guard_health
        else:
            guard_broken = jnp.zeros(2, bool)
        if self.use_reward_budget:
            # P1's budget pays for breaking P2's guard and vice versa.
            deductions = deductions + guard_broken[::-1].astype(jnp.int32)
        k = jnp.clip(deductions, 0, MAX_BUDGET_DEDUCTIONS)
        i = lambda x: x.astype(jnp.int32)
        rewards = jnp.asarray(self.table)[
            i(guard_broken[1]), i(guard_broken[0]), i(dead[1]), i(dead[0]), k[0], k[1]
        ]
        return rewards, deductions

    def _build_table(self) -> np.ndarray:
        """Indexed [p2_guard_broken, p1_guard_broken, p2_dead, p1_dead,
        p1_deductions, p2_deductions] -> (p1, p2) rewards.

        float64 when ``jax_enable_x64`` is on (identical to ``FootsiesEnv``),
        otherwise float32 (the float64 value rounded once).
        """
        gb, n = self.guard_break_reward, MAX_BUDGET_DEDUCTIONS + 1
        table = np.zeros((2, 2, 2, 2, n, n, 2), dtype=np.float64)
        for p2_gb, p1_gb, p2_dead, p1_dead, k1, k2 in itertools.product(
            (0, 1), (0, 1), (0, 1), (0, 1), range(n), range(n)
        ):
            budget = np.full(2, self.win_reward, dtype=np.float64)
            for _ in range(k1):
                budget[0] -= gb
            for _ in range(k2):
                budget[1] -= gb
            p1, p2 = np.float64(0.0), np.float64(0.0)
            if gb != 0:
                if p2_gb:
                    p1 += gb
                    p2 -= gb
                if p1_gb:
                    p2 += gb
                    p1 -= gb
            p1 += budget[0] * np.float64(p2_dead)
            p2 -= budget[0] * np.float64(p2_dead)
            p2 += budget[1] * np.float64(p1_dead)
            p1 -= budget[1] * np.float64(p1_dead)
            table[p2_gb, p1_gb, p2_dead, p1_dead, k1, k2] = (p1, p2)
        return table if jax.config.jax_enable_x64 else table.astype(np.float32)
