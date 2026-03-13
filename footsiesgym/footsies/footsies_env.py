import collections
import functools
import os
import platform
import subprocess
import time
from typing import Any

import numpy as np
import portpicker
from gymnasium import spaces
from pettingzoo import ParallelEnv

from footsiesgym.footsies.game.proto import (
    footsies_service_pb2 as footsies_pb2,
)
from footsiesgym.footsies.typing import ActionType, AgentID, ObsType

from ..binary_manager import get_binary_manager
from . import encoder
from .game import constants, footsies_game


class FootsiesEnv(ParallelEnv):
    """PettingZoo ParallelEnv for the FOOTSIES fighting game.

    Supports two modes selected by config["num_envs"]:
      - Single-env (num_envs=1, default): Uses FootsiesGameService,
        scalar state, deque-based action delay. Standard PettingZoo API
        returning scalars/1-D arrays per agent.
      - Vectorized (num_envs>1): Uses VectorizedFootsiesService,
        numpy array state, circular-buffer action delay. Returns
        batched arrays of shape (num_envs, ...) per agent key.
    """

    metadata = {"render_modes": ["human"], "name": "footsies_v0"}
    LINUX_ZIP_PATH_HEADLESS = "binaries/footsies_linux_headless_9c6b36f.zip"
    LINUX_ZIP_PATH_WINDOWED = "binaries/footsies_linux_windowed_9c6b36f.zip"
    SPECIAL_CHARGE_FRAMES = 60

    def __init__(
        self,
        config: dict[Any, Any] = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        if render_mode is not None:
            raise ValueError(
                "FootsiesEnv does not support render_mode. "
                "For visual rendering, set headless=False in the "
                "env config or manually launch the windowed binaries."
            )
        self.render_mode = render_mode

        if config is None:
            config = {}
        self.config = config

        # ── Shared config ──────────────────────────────────────
        self.num_envs: int = config.get("num_envs", 1)
        self.vectorized: bool = self.num_envs > 1

        self.return_fight_state_in_infos = config.get(
            "return_fight_state_in_infos", False
        )
        self.use_build_encoding = config.get("use_build_encoding", False)
        self.agents: list[AgentID] = ["p1", "p2"]
        self.possible_agents: list[AgentID] = self.agents.copy()
        self.win_reward_scaling_coeff = config.get(
            "win_reward_scaling_coeff", 1.0
        )
        self.guard_break_reward_value = config.get("guard_break_reward", 0.0)
        self.use_reward_budget = config.get("use_reward_budget", False)
        assert (
            self.guard_break_reward_value * 3 < self.win_reward_scaling_coeff
        ), (
            "Guard break reward total must be less than the win "
            "reward (guard break reward * 3 < win reward)"
        )

        self._action_spaces: dict[AgentID, spaces.Discrete] = (
            self._build_action_spaces(
                use_special_charge_action=config.get(
                    "use_special_charge_action", False
                )
            )
        )

        self.num_actions: int = len(
            [
                constants.EnvActions.NONE,
                constants.EnvActions.BACK,
                constants.EnvActions.FORWARD,
                constants.EnvActions.ATTACK,
                constants.EnvActions.BACK_ATTACK,
                constants.EnvActions.FORWARD_ATTACK,
                constants.EnvActions.SPECIAL_CHARGE,
                constants.EnvActions.FORWARD_SPECIAL_CHARGE,
                constants.EnvActions.BACK_SPECIAL_CHARGE,
            ]
        )

        self.evaluation = config.get("evaluation", False)
        self.max_t: int = config.get("max_t", 4000)
        self.frame_skip: int = config.get("frame_skip", 4)
        self.action_delay_frames: int = config.get("action_delay", 8)

        assert (
            self.action_delay_frames % self.frame_skip == 0
        ), "action_delay must be divisible by frame_skip"

        self.action_delay_steps: int = (
            self.action_delay_frames // self.frame_skip
        )

        port = config.get("port", None)
        self.headless = config.get("headless", True)
        if port is None:
            port = portpicker.pick_unused_port()

        self.server_process = None
        launch_binaries = config.get("launch_binaries", False)
        if launch_binaries:
            self._launch_binaries(port=port)

        # ── Mode-specific init ─────────────────────────────────
        if self.vectorized:
            self._init_vectorized(config, port)
        else:
            self._init_single(config, port)

    # ──────────────────────────────────────────────────────────
    # Init helpers
    # ──────────────────────────────────────────────────────────

    def _init_single(self, config, port):
        """Initialize single-env mode state."""
        self._encoder = encoder.FootsiesEncoder()
        self.game = footsies_game.FootsiesGame(
            host=config.get("host", "localhost"),
            port=port,
        )
        self.t: int = 0
        self._action_queues: dict[AgentID, collections.deque[int]] = None
        self.prev_selected_actions: dict[AgentID, int] = {
            a: constants.EnvActions.NONE for a in self.agents
        }
        self.prev_executed_actions: dict[AgentID, int] = {
            a: constants.EnvActions.NONE for a in self.agents
        }
        self.reward_budget = {
            a: self.win_reward_scaling_coeff for a in self.agents
        }
        self._holding_special_charge = {
            "p1": False,
            "p2": False,
        }
        self.last_game_state = None
        self._reset_action_delay_queues()

    def _init_vectorized(self, config, port):
        """Initialize vectorized mode state."""
        N = self.num_envs
        K = self.action_delay_steps

        self._encoder = encoder.VectorizedEncoder(num_actions=self.num_actions)
        self.vec_game = footsies_game.VectorizedFootsiesGame(
            host=config.get("host", "localhost"),
            port=port,
            num_envs=N,
        )
        self._initialized = False

        # Step counter per env
        self._t = np.zeros(N, dtype=np.int64)

        # Action delay circular buffer: (num_envs, 2, K)
        # axis 1: 0=p1, 1=p2
        self._action_queue = np.full(
            (N, 2, max(K, 1)),
            constants.EnvActions.NONE,
            dtype=np.int64,
        )
        self._queue_head = 0

        # Per-env per-agent state: (num_envs, 2)
        self._prev_selected = np.zeros((N, 2), dtype=np.int64)
        self._prev_executed = np.zeros((N, 2), dtype=np.int64)
        self._holding_special = np.zeros((N, 2), dtype=bool)
        self._reward_budget = np.full(
            (N, 2), self.win_reward_scaling_coeff, dtype=np.float64
        )

        # Guard health tracking for guard break rewards
        self._last_guard_health = np.zeros((N, 2), dtype=np.int64)

    # ──────────────────────────────────────────────────────────
    # PettingZoo API: spaces
    # ──────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> spaces.Box:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(encoder.FootsiesEncoder.observation_size,),
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> spaces.Discrete:
        return self._action_spaces[agent]

    @classmethod
    def get_action_space(cls, use_special_charge_action: bool = False):
        return cls._build_action_spaces(use_special_charge_action)

    @classmethod
    def _build_action_spaces(cls, use_special_charge_action: bool = False):
        available_actions = [
            constants.EnvActions.NONE,
            constants.EnvActions.BACK,
            constants.EnvActions.FORWARD,
            constants.EnvActions.ATTACK,
            constants.EnvActions.BACK_ATTACK,
            constants.EnvActions.FORWARD_ATTACK,
        ]
        if use_special_charge_action:
            available_actions.extend([
                constants.EnvActions.SPECIAL_CHARGE,
                constants.EnvActions.FORWARD_SPECIAL_CHARGE,
                constants.EnvActions.BACK_SPECIAL_CHARGE,
            ])

        return {
            agent: spaces.Discrete(len(available_actions))
            for agent in ["p1", "p2"]
        }

    # ──────────────────────────────────────────────────────────
    # PettingZoo API: reset / step (dispatch)
    # ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, Any]]:
        if self.vectorized:
            return self._reset_vectorized()
        return self._reset_single()

    def step(self, actions):
        if self.vectorized:
            return self._step_vectorized(actions)
        return self._step_single(actions)

    # ──────────────────────────────────────────────────────────
    # Single-env implementation
    # ──────────────────────────────────────────────────────────

    def _reset_action_delay_queues(self):
        self._action_queues = {
            agent_id: collections.deque(
                [constants.EnvActions.NONE] * self.action_delay_steps,
                maxlen=self.action_delay_steps,
            )
            for agent_id in self.agents
        }

    def _reset_single(self):
        self.agents = self.possible_agents.copy()
        self.t = 0
        self.game.reset_game()
        self.game.start_game()

        self._reset_action_delay_queues()
        self.prev_selected_actions = {
            a: constants.EnvActions.NONE for a in self.agents
        }
        self.prev_executed_actions = {
            a: constants.EnvActions.NONE for a in self.agents
        }
        self._holding_special_charge = {
            "p1": False,
            "p2": False,
        }
        self.reward_budget = {
            a: self.win_reward_scaling_coeff for a in self.agents
        }
        self._encoder.reset()

        self.last_game_state = self.game.get_state()
        observations = self._get_obs_single(
            self.last_game_state,
            self.prev_selected_actions,
            self._holding_special_charge,
        )
        return observations, self._get_infos_single()

    def _get_obs_single(self, game_state, prev_actions, holding):
        if self.use_build_encoding:
            raise NotImplementedError(
                "Build encoder has not yet integrated action delay! "
                "Please use the default Python encoder for now."
            )
        return self._encoder.encode(
            game_state,
            prev_actions,
            holding,
            self.num_actions,
        )

    def _step_single(self, actions):
        self.t += 1

        # Action delay
        actions_to_execute: dict[AgentID, ActionType] = {}
        if self.action_delay_frames == 0:
            actions_to_execute = actions.copy()
        else:
            for agent_id in self.agents:
                actions_to_execute[agent_id] = self._action_queues[
                    agent_id
                ].popleft()
                self._action_queues[agent_id].append(actions[agent_id])

        # Special charge toggle
        for agent_id in self.agents:
            action_is_special_charge = (
                actions_to_execute[agent_id]
                in [
                    constants.EnvActions.SPECIAL_CHARGE,
                    constants.EnvActions.FORWARD_SPECIAL_CHARGE,
                    constants.EnvActions.BACK_SPECIAL_CHARGE,
                ]
            )

            if action_is_special_charge:
                self._holding_special_charge[agent_id] = (
                    not self._holding_special_charge[agent_id]
                )
                actions_to_execute[agent_id] = (
                    self._convert_special_charge_to_base_action(
                        actions_to_execute[agent_id]
                    )
                )

            if self._holding_special_charge[agent_id]:
                actions_to_execute[agent_id] = self._convert_to_charge_action(
                    actions_to_execute[agent_id]
                )

        p1_action = self.game.action_to_bits(
            actions_to_execute["p1"], is_player_1=True
        )
        p2_action = self.game.action_to_bits(
            actions_to_execute["p2"], is_player_1=False
        )

        game_state = self.game.step_n_frames(
            p1_action=p1_action,
            p2_action=p2_action,
            n_frames=self.frame_skip,
        )
        observations = self._get_obs_single(
            game_state,
            actions,
            self._holding_special_charge,
        )

        terminated = game_state.player1.is_dead or game_state.player2.is_dead

        rewards = {a_id: 0.0 for a_id in self.agents}
        if self.guard_break_reward_value != 0:
            p1_prev_gh = self.last_game_state.player1.guard_health
            p2_prev_gh = self.last_game_state.player2.guard_health
            p1_gh = game_state.player1.guard_health
            p2_gh = game_state.player2.guard_health

            if p2_gh < p2_prev_gh:
                if self.use_reward_budget:
                    self.reward_budget["p1"] -= self.guard_break_reward_value
                rewards["p1"] += self.guard_break_reward_value
                rewards["p2"] -= self.guard_break_reward_value
            if p1_gh < p1_prev_gh:
                if self.use_reward_budget:
                    self.reward_budget["p2"] -= self.guard_break_reward_value
                rewards["p2"] += self.guard_break_reward_value
                rewards["p1"] -= self.guard_break_reward_value

        opp_dead = {
            "p1": int(game_state.player2.is_dead),
            "p2": int(game_state.player1.is_dead),
        }
        for a_id, dead in opp_dead.items():
            other = "p2" if a_id == "p1" else "p1"
            rewards[a_id] += self.reward_budget[a_id] * dead
            rewards[other] -= self.reward_budget[a_id] * dead

        terminateds = {"p1": terminated, "p2": terminated}
        truncated = self.t >= self.max_t
        truncateds = {"p1": truncated, "p2": truncated}

        self.last_game_state = game_state
        self.prev_executed_actions = actions_to_execute
        self.prev_selected_actions = actions
        infos = self._get_infos_single()

        # PettingZoo convention: clear agents when episode ends
        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminateds, truncateds, infos

    def _get_infos_single(self):
        infos = {agent: {} for agent in self.agents}
        if self.return_fight_state_in_infos:
            infos.update(self._get_fight_state_dicts())
        return infos

    def _get_fight_state_dicts(self):
        fight_state_dict = {"p1": {}, "p2": {}}
        p1_state = self.last_game_state.player1
        p2_state = self.last_game_state.player2

        dist_x = np.abs(p1_state.player_position_x - p2_state.player_position_x)

        for player, opp in zip(["p1", "p2"], [p2_state, p1_state]):
            fight_state_dict[player]["distance_x"] = dist_x
            fight_state_dict[player]["is_opponent_damage"] = (
                opp.current_action_id == constants.ActionID.DAMAGE
            )
            fight_state_dict[player]["is_opponent_guard_break"] = (
                opp.current_action_id == constants.ActionID.GUARD_BREAK
            )
            fight_state_dict[player]["is_opponent_blocking"] = (
                opp.current_action_id
                in [
                    constants.ActionID.GUARD_CROUCH,
                    constants.ActionID.GUARD_STAND,
                    constants.ActionID.GUARD_M,
                ]
            )
            fight_state_dict[player]["is_opponent_normal_attack"] = (
                opp.current_action_id
                in [
                    constants.ActionID.N_ATTACK,
                    constants.ActionID.B_ATTACK,
                ]
            )
            fight_state_dict[player]["is_opponent_special_attack"] = (
                opp.current_action_id
                in [
                    constants.ActionID.N_SPECIAL,
                    constants.ActionID.B_SPECIAL,
                ]
            )

        for player, state in zip(["p1", "p2"], [p1_state, p2_state]):
            fight_state_dict[player]["is_facing_right"] = state.is_face_right

        return fight_state_dict

    # ──────────────────────────────────────────────────────────
    # Vectorized implementation
    # ──────────────────────────────────────────────────────────

    def _reset_vectorized(self):
        N = self.num_envs

        if not self._initialized:
            self.vec_game.start_and_init()
            self._initialized = True

        raw = self.vec_game.batch_reset_all()

        # Server may need a moment after init; retry if empty
        for _ in range(10):
            if len(raw.p1_guard_health) == N:
                break
            time.sleep(0.5)
            raw = self.vec_game.batch_reset_all()
        else:
            raise RuntimeError(
                f"BatchResetAll returned {len(raw.p1_guard_health)} "
                f"envs, expected {N}"
            )

        # Reset all Python-side state
        self._t[:] = 0
        self._action_queue[:] = constants.EnvActions.NONE
        self._queue_head = 0
        self._prev_selected[:] = constants.EnvActions.NONE
        self._prev_executed[:] = constants.EnvActions.NONE
        self._holding_special[:] = False
        self._reward_budget[:] = self.win_reward_scaling_coeff

        # Store initial guard health
        self._last_guard_health[:, 0] = np.array(
            raw.p1_guard_health, dtype=np.int64
        )
        self._last_guard_health[:, 1] = np.array(
            raw.p2_guard_health, dtype=np.int64
        )

        obs = self._encoder.encode(
            raw,
            self._prev_selected[:, 0],
            self._prev_selected[:, 1],
            self._holding_special[:, 0],
            self._holding_special[:, 1],
        )

        infos = {"p1": {}, "p2": {}}
        return obs, infos

    def _step_vectorized(self, actions):
        """Vectorized step over all environments.

        Args:
            actions: dict with "p1" and "p2" keys, each an
                np.ndarray of shape (num_envs,) with int actions.
        """
        N = self.num_envs
        K = self.action_delay_steps

        p1_actions = np.asarray(actions["p1"], dtype=np.int64)
        p2_actions = np.asarray(actions["p2"], dtype=np.int64)
        # Stack to (N, 2) for uniform processing
        selected = np.stack([p1_actions, p2_actions], axis=1)

        self._t += 1

        # ── Action delay ──────────────────────────────────────
        if self.action_delay_frames == 0:
            to_execute = selected.copy()
        else:
            head = self._queue_head
            # Dequeue: read at head
            to_execute = self._action_queue[:, :, head].copy()
            # Enqueue: write current selection at head
            self._action_queue[:, :, head] = selected
            self._queue_head = (head + 1) % K

        # ── Special charge toggle ─────────────────────────────
        is_special = (
            (to_execute == constants.EnvActions.SPECIAL_CHARGE)
            | (to_execute == constants.EnvActions.FORWARD_SPECIAL_CHARGE)
            | (to_execute == constants.EnvActions.BACK_SPECIAL_CHARGE)
        )
        if is_special.any():
            # Toggle holding state
            self._holding_special[is_special] = ~self._holding_special[
                is_special
            ]
            # Convert special charge actions to their base movement
            base = to_execute.copy()
            base[
                to_execute == constants.EnvActions.SPECIAL_CHARGE
            ] = constants.EnvActions.NONE
            base[
                to_execute == constants.EnvActions.FORWARD_SPECIAL_CHARGE
            ] = constants.EnvActions.FORWARD
            base[
                to_execute == constants.EnvActions.BACK_SPECIAL_CHARGE
            ] = constants.EnvActions.BACK
            to_execute[is_special] = base[is_special]

        # Apply charge conversion for all held envs/agents
        held = self._holding_special
        if held.any():
            to_execute[held] = constants.CHARGE_ACTION_LUT[to_execute[held]]

        # ── Action-to-bits via LUT ────────────────────────────
        p1_bits = constants.P1_ENV_TO_BITS[to_execute[:, 0]]
        p2_bits = constants.P2_ENV_TO_BITS[to_execute[:, 1]]

        # ── gRPC step ─────────────────────────────────────────
        raw = self.vec_game.batch_step(p1_bits, p2_bits, self.frame_skip)

        # ── Encode observations ───────────────────────────────
        obs = self._encoder.encode(
            raw,
            selected[:, 0],  # prev_selected for next step's obs
            selected[:, 1],
            self._holding_special[:, 0],
            self._holding_special[:, 1],
        )

        # ── Extract state for rewards ─────────────────────────
        p1_dead = np.array(raw.p1_is_dead, dtype=bool)
        p2_dead = np.array(raw.p2_is_dead, dtype=bool)
        p1_gh = np.array(raw.p1_guard_health, dtype=np.int64)
        p2_gh = np.array(raw.p2_guard_health, dtype=np.int64)

        terminated = p1_dead | p2_dead
        truncated = self._t >= self.max_t

        # ── Rewards ───────────────────────────────────────────
        p1_rewards = np.zeros(N, dtype=np.float64)
        p2_rewards = np.zeros(N, dtype=np.float64)

        # Guard break rewards
        if self.guard_break_reward_value != 0:
            p2_gb = p2_gh < self._last_guard_health[:, 1]
            p1_gb = p1_gh < self._last_guard_health[:, 0]

            if self.use_reward_budget:
                self._reward_budget[p2_gb, 0] -= self.guard_break_reward_value
                self._reward_budget[p1_gb, 1] -= self.guard_break_reward_value

            p1_rewards[p2_gb] += self.guard_break_reward_value
            p2_rewards[p2_gb] -= self.guard_break_reward_value
            p2_rewards[p1_gb] += self.guard_break_reward_value
            p1_rewards[p1_gb] -= self.guard_break_reward_value

        # Win/loss rewards (zero-sum via reward budget)
        p2_dead_f = p2_dead.astype(np.float64)
        p1_dead_f = p1_dead.astype(np.float64)
        p1_rewards += self._reward_budget[:, 0] * p2_dead_f
        p2_rewards -= self._reward_budget[:, 0] * p2_dead_f
        p2_rewards += self._reward_budget[:, 1] * p1_dead_f
        p1_rewards -= self._reward_budget[:, 1] * p1_dead_f

        # ── Update state ──────────────────────────────────────
        self._prev_executed = to_execute
        self._prev_selected = selected
        self._last_guard_health[:, 0] = p1_gh
        self._last_guard_health[:, 1] = p2_gh

        # ── Auto-reset done envs ──────────────────────────────
        done = terminated | truncated
        if done.any():
            # Explicitly reset all done envs. The server
            # auto-resets on KO at the next step, but that
            # skips a clean post-reset state. Calling Reset()
            # twice is idempotent so this is safe.
            self.vec_game.batch_reset(done)
            self._reset_vec_env_state(done)

        rewards = {"p1": p1_rewards, "p2": p2_rewards}
        terminateds = {"p1": terminated, "p2": terminated}
        truncateds = {"p1": truncated, "p2": truncated}
        infos = {"p1": {}, "p2": {}}

        return obs, rewards, terminateds, truncateds, infos

    def _reset_vec_env_state(self, mask: np.ndarray):
        """Reset Python-side state for environments indicated by mask."""
        self._t[mask] = 0
        self._action_queue[mask] = constants.EnvActions.NONE
        self._prev_selected[mask] = constants.EnvActions.NONE
        self._prev_executed[mask] = constants.EnvActions.NONE
        self._holding_special[mask] = False
        self._reward_budget[mask] = self.win_reward_scaling_coeff

    # ──────────────────────────────────────────────────────────
    # Shared utilities
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _convert_to_charge_action(action: int) -> int:
        if action == constants.EnvActions.BACK:
            return constants.EnvActions.BACK_ATTACK
        elif action == constants.EnvActions.FORWARD:
            return constants.EnvActions.FORWARD_ATTACK
        elif action == constants.EnvActions.BACK_ATTACK:
            return constants.EnvActions.BACK_ATTACK
        elif action == constants.EnvActions.FORWARD_ATTACK:
            return constants.EnvActions.FORWARD_ATTACK
        else:
            return constants.EnvActions.ATTACK

    @staticmethod
    def _convert_special_charge_to_base_action(action: int) -> int:
        if action == constants.EnvActions.SPECIAL_CHARGE:
            return constants.EnvActions.NONE
        elif action == constants.EnvActions.FORWARD_SPECIAL_CHARGE:
            return constants.EnvActions.FORWARD
        elif action == constants.EnvActions.BACK_SPECIAL_CHARGE:
            return constants.EnvActions.BACK
        raise ValueError(
            f"Invalid special charge action: {action}, expected "
            "one of SPECIAL_CHARGE, FORWARD_SPECIAL_CHARGE, "
            "BACK_SPECIAL_CHARGE."
        )

    def _launch_binaries(self, port: int):
        if platform.system().lower() in ["windows", "darwin"]:
            raise RuntimeError(
                "Binary launching is only supported on Linux. "
                "Please launch the footsies server manually "
                "or use a Linux system."
            )

        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        binary_subdir = (
            "footsies_binaries_headless"
            if self.headless
            else "footsies_binaries_windowed"
        )
        binary_path = os.path.join(
            project_root,
            "binaries",
            binary_subdir,
            "footsies.x86_64",
        )

        if not os.path.exists(binary_path):
            binary_manager = get_binary_manager()
            binaries_dir = os.path.join(project_root, "binaries")
            if not binary_manager.ensure_binaries_extracted(
                "linux",
                target_dir=binaries_dir,
                headless=self.headless,
            ):
                raise FileNotFoundError(
                    "Failed to download and extract footsies "
                    "binaries. Please check your internet "
                    "connection and try again."
                )
            if not os.path.exists(binary_path):
                raise FileNotFoundError(
                    f"Failed to find footsies binary at "
                    f"{binary_path} after extraction."
                )

        if not os.access(binary_path, os.X_OK):
            os.chmod(binary_path, 0o755)

        command = [
            binary_path,
            "-batchmode",
            "--grpc",
            "--port",
            str(port),
        ]

        if not self.headless and not os.environ.get("DISPLAY"):
            print(
                "Warning: DISPLAY environment variable not set. "
                "Windowed mode may not work in WSL."
            )

        print("Launching with command:", command)

        if self.headless:
            self.server_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            self.server_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        binary_type = "headless" if self.headless else "windowed"
        print(f"Launched {binary_type} footsies binary on port " f"{port}.")
        time.sleep(5)

    def close(self):
        """Clean up resources when the environment is closed."""
        if hasattr(self, "server_process") and self.server_process is not None:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print(
                    "Terminated footsies server process "
                    f"(PID: {self.server_process.pid})."
                )
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
                print(
                    "Force killed footsies server process "
                    f"(PID: {self.server_process.pid})."
                )
            except Exception as e:
                print(f"Error terminating server process: {e}")
            finally:
                self.server_process = None

    def __del__(self):
        self.close()
