import collections
import os
import platform
import subprocess
import time
from typing import Any

import numpy as np
import portpicker
from gymnasium import spaces
from ray.rllib import env

from footsiesgym.footsies.game.proto import (
    footsies_service_pb2 as footsies_pb2,
)
from footsiesgym.footsies.typing import ActionType, AgentID, ObsType

from ..binary_manager import get_binary_manager
from . import encoder
from .game import constants, footsies_game


class FootsiesEnv(env.MultiAgentEnv):
    metadata = {"render.modes": ["human"]}
    LINUX_ZIP_PATH_HEADLESS = "binaries/footsies_linux_server_021725.zip"
    LINUX_ZIP_PATH_WINDOWED = "binaries/footsies_linux_windowed_021725.zip"
    SPECIAL_CHARGE_FRAMES = 60


    SPECIAL_CHARGE_ALLOWED_ACTIONS = np.array([
        constants.EnvActions.ATTACK,
        constants.EnvActions.FORWARD_ATTACK,
        constants.EnvActions.BACK_ATTACK,
        constants.EnvActions.SPECIAL_CHARGE,
        constants.EnvActions.BACK_SPECIAL_CHARGE,
        constants.EnvActions.FORWARD_SPECIAL_CHARGE,
    ], dtype=np.int32)

    def __init__(self, config: dict[Any, Any] = None):
        super(FootsiesEnv, self).__init__()

        if config is None:
            config = {}
        self.config = config
        self.return_fight_state_in_infos = config.get(
            "return_fight_state_in_infos", False
        )
        self.return_action_mask_in_infos = config.get(
            "return_action_mask_in_infos", True
        )
        self.use_build_encoding = config.get("use_build_encoding", False)
        self.agents: list[AgentID] = ["p1", "p2"]
        self.possible_agents: list[AgentID] = self.agents.copy()
        self._agent_ids: set[AgentID] = set(self.agents)
        self.win_reward_scaling_coeff = self.config.get(
            "win_reward_scaling_coeff", 1.0
        )
        self.guard_break_reward_value = self.config.get(
            "guard_break_reward", 0.0
        )
        self.use_reward_budget = self.config.get("use_reward_budget", False)
        assert (
            self.guard_break_reward_value * 3 < self.win_reward_scaling_coeff
        ), "Guard break reward total must be less than the win reward (guard break reward * 3 < win reward)"

        # Add special charge action, if desired. The special actions
        # require that the ATTACK button be held for 60 frames. Depending
        # on enviroment parameters, this may be exceedingly long
        # for the agent to learn to hold a single button. The SPECIAL_CHARGE
        # action toggles whether or not the agent wishes to hold ATTACK.
        # For example:
        #  Agent selects:  [SPECIAL_CHARGE, NONE, SPECIAL_CHARGE]
        #  Executed Action: [ATTACK, ATTACK, NONE]
        # The second special charge deactivates the held ATTACK.
        self.action_space: dict[AgentID, spaces.Discrete] = (
            self.get_action_space(
                use_special_charge_action=config.get(
                    "use_special_charge_action", False
                )
            )
        )

        self.observation_space = self.get_observation_space(
            use_special_charge_action=config.get("use_special_charge_action", False)
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

        self.reward_budget = {
            agent: self.win_reward_scaling_coeff for agent in self.agents
        }

        self.evaluation = config.get("evaluation", False)

        self.t: int = 0
        self.max_t: int = config.get("max_t", 4000)
        self.frame_skip: int = config.get("frame_skip", 4)
        self.action_delay_frames: int = config.get("action_delay", 8)

        assert (
            self.action_delay_frames % self.frame_skip == 0
        ), "action_delay must be divisible by frame_skip"

        self.action_delay_steps: int = (
            self.action_delay_frames // self.frame_skip
        )
        self.encoder = encoder.FootsiesEncoder()
        self._action_queues: dict[AgentID, collections.deque[int]] = None

        # We track two different previous actions: the last action that was sent to the game server
        # and the last action that the policy selected. Due to action delay this represents a_{t-K}
        # and a_{t-1} respectively, where K is the action delay.
        self.prev_selected_actions: dict[AgentID, int] = {
            agent: constants.EnvActions.NONE for agent in self.agents
        }
        self.prev_executed_actions: dict[AgentID, int] = {
            agent: constants.EnvActions.NONE for agent in self.agents
        }
        self._reset_action_delay_queues()

        port = config.get("port", None)
        self.headless = config.get("headless", True)
        # Use portpicker to automatically find an available port
        if port is None:
            port = portpicker.pick_unused_port()

        # If specified, we'll launch the binaries from the environment itself.
        self.server_process = None
        launch_binaries = config.get("launch_binaries", False)
        if launch_binaries:
            self._launch_binaries(port=port)

        self.game = footsies_game.FootsiesGame(
            host=config.get("host", "localhost"),
            port=port,
        )

        self.last_game_state = None
        self._holding_special_charge = {
            "p1": False,
            "p2": False,
        }

    @classmethod
    def get_action_space(cls, use_special_charge_action: bool = False):
        available_actions = [
            constants.EnvActions.NONE,
            constants.EnvActions.BACK,
            constants.EnvActions.FORWARD,
            constants.EnvActions.ATTACK,
            constants.EnvActions.BACK_ATTACK,
            constants.EnvActions.FORWARD_ATTACK,
        ]

        # Add special charge action, if desired. The special actions
        # require that the ATTACK button be held for 60 frames. Depending
        # on enviroment parameters, this may be exceedingly long
        # for the agent to learn to hold a single button. The SPECIAL_CHARGE
        # action toggles whether or not the agent wishes to hold ATTACK.
        # For example:
        #  Agent selects:  [SPECIAL_CHARGE, NONE, SPECIAL_CHARGE]
        #  Executed Action: [ATTACK, ATTACK, NONE]
        # The second special charge deactivates the held ATTACK.
        if use_special_charge_action:
            available_actions.extend([
                constants.EnvActions.SPECIAL_CHARGE,
                constants.EnvActions.FORWARD_SPECIAL_CHARGE,
                constants.EnvActions.BACK_SPECIAL_CHARGE,
            ])

        return spaces.Dict(
            {
                agent: spaces.Discrete(len(available_actions))
                for agent in ["p1", "p2"]
            }
        )

    @classmethod
    def get_observation_space(cls, use_special_charge_action: bool = False):
        action_space = cls.get_action_space(use_special_charge_action)
        return spaces.Dict({
            agent: spaces.Dict({
                "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(encoder.FootsiesEncoder.observation_size,)),
                "action_mask": spaces.Box(low=0, high=1, shape=(action_space[agent].n,), dtype=np.float32),
            }) for agent in ["p1", "p2"]})

    def _get_fight_state_dicts(self):
        """
        class FightState:
            distance_x: float
            is_opponent_damage: bool
            is_opponent_guard_break: bool
            is_opponent_blocking: bool
            is_opponent_normal_attack: bool
            is_opponent_special_attack: bool
            is_facing_right: bool
        """
        fight_state_dict = {
            "p1": {},
            "p2": {},
        }
        p1_state, p2_state = (
            self.last_game_state.player1,
            self.last_game_state.player2,
        )

        dist_x = np.abs(
            p1_state.player_position_x - p2_state.player_position_x
        )

        for player, opp_state in zip(["p1", "p2"], [p2_state, p1_state]):
            fight_state_dict[player]["distance_x"] = dist_x
            fight_state_dict[player]["is_opponent_damage"] = (
                opp_state.current_action_id == constants.ActionID.DAMAGE
            )
            fight_state_dict[player]["is_opponent_guard_break"] = (
                opp_state.current_action_id == constants.ActionID.GUARD_BREAK
            )
            fight_state_dict[player]["is_opponent_blocking"] = (
                opp_state.current_action_id
                in [
                    constants.ActionID.GUARD_CROUCH,
                    constants.ActionID.GUARD_STAND,
                    constants.ActionID.GUARD_M,
                ]
            )
            fight_state_dict[player]["is_opponent_normal_attack"] = (
                opp_state.current_action_id
                in [constants.ActionID.N_ATTACK, constants.ActionID.B_ATTACK]
            )
            fight_state_dict[player]["is_opponent_special_attack"] = (
                opp_state.current_action_id
                in [constants.ActionID.N_SPECIAL, constants.ActionID.B_SPECIAL]
            )

        for player, state in zip(["p1", "p2"], [p1_state, p2_state]):
            fight_state_dict[player]["is_facing_right"] = state.is_face_right

        return fight_state_dict

    def _launch_binaries(self, port: int):
        # Check if we're on a supported platform
        if platform.system().lower() in ["windows", "darwin"]:
            raise RuntimeError(
                "Binary launching is only supported on Linux. "
                "Please launch the footsies server manually or use a Linux system."
            )

        # Check to ensure the linux binaries exist in the appropriate directory based on headless setting

        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        binary_subdir = (
            "footsies_binaries_headless"
            if self.headless
            else "footsies_binaries_windowed"
        )
        binary_path = os.path.join(
            project_root, "binaries", binary_subdir, "footsies.x86_64"
        )

        if not os.path.exists(binary_path):
            # Use binary manager to download and extract binaries atomically
            binary_manager = get_binary_manager()

            # Ensure binaries are downloaded and extracted (with file locking to prevent race conditions)
            binaries_dir = os.path.join(project_root, "binaries")
            if not binary_manager.ensure_binaries_extracted(
                "linux", target_dir=binaries_dir, headless=self.headless
            ):
                raise FileNotFoundError(
                    "Failed to download and extract footsies binaries. "
                    "Please check your internet connection and try again."
                )

            # Verify the binary now exists
            if not os.path.exists(binary_path):
                raise FileNotFoundError(
                    f"Failed to find footsies binary at {binary_path} after extraction."
                )

        # We'll also want to make sure the binary is executable
        if not os.access(binary_path, os.X_OK):
            # If not, make it executable
            os.chmod(binary_path, 0o755)

        # portpicker already ensures the port is available, so no need to check

        command = [binary_path, "--port", str(port)]

        # For windowed mode in WSL, check if DISPLAY is set
        if not self.headless and not os.environ.get("DISPLAY"):
            print(
                "⚠️  Warning: DISPLAY environment variable not set. Windowed mode may not work in WSL."
            )
            print(
                "   For WSL2 with Windows 11, WSLg should handle this automatically."
            )
            print(
                "   For older WSL versions, you may need to set up X11 forwarding."
            )

        print("Launching with command:", command)

        # For windowed mode, don't suppress output as it may contain important display messages
        # For headless mode, suppress output to keep it clean
        if self.headless:
            # Headless mode - suppress output
            self.server_process = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            # Windowed mode - allow output for display setup (important for WSL)
            self.server_process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

        binary_type = "headless" if self.headless else "windowed"
        print(f"Launched {binary_type} footsies binary on port {port}.")
        time.sleep(5)

    def close(self):
        """Clean up resources when the environment is closed."""
        if hasattr(self, "server_process") and self.server_process is not None:
            try:
                self.server_process.terminate()
                # Give it a moment to terminate gracefully
                self.server_process.wait(timeout=5)
                print(
                    f"Terminated footsies server process (PID: {self.server_process.pid})."
                )
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                self.server_process.kill()
                self.server_process.wait()
                print(
                    f"Force killed footsies server process (PID: {self.server_process.pid})."
                )
            except Exception as e:
                print(f"Error terminating server process: {e}")
            finally:
                self.server_process = None

    def __del__(self):
        """Ensure cleanup happens when the object is garbage collected."""
        self.close()

    def _reset_action_delay_queues(self):
        self._action_queues: dict[AgentID, collections.deque[int]] = {
            agent_id: collections.deque(
                [constants.EnvActions.NONE] * self.action_delay_steps,
                maxlen=self.action_delay_steps,
            )
            for agent_id in self.agents
        }

    def _validate_action_queues(self):
        for agent_id in self.agents:
            assert (
                len(self._action_queues[agent_id]) == self.action_delay_steps
            ), (
                f"Action queue has the incorrect number of queued actions! "
                " Observed {len(self._action_queues[agent_id])}, expected {self.action_delay_steps}"
            )

    def get_obs(
        self,
        game_state: footsies_pb2.GameState,
        prev_actions: dict[AgentID, ActionType],
        is_charging_special: dict[AgentID, bool],
        num_actions: int,
    ):
        if self.use_build_encoding:
            raise NotImplementedError(
                "Build encoder has not yet integrated action delay! "
                "Please use the default Python encoder for now."
            )
            # encoded_state = self.game.get_encoded_state()
            # encoded_state_dict = {
            #     "p1": np.asarray(
            #         encoded_state.player1_encoding, dtype=np.float32
            #     ),
            #     "p2": np.asarray(
            #         encoded_state.player2_encoding, dtype=np.float32
            #     ),
            # }
            # TODO(chase): If used, we also need to add action masking here. 
            # return encoded_state_dict
        else:
            obs = self.encoder.encode(
                game_state, prev_actions, is_charging_special, num_actions
            )

            # Add action masks to the observations
            for agent in self.agents:
                action_mask = self.get_action_mask(agent)
                obs[agent]["action_mask"] = action_mask

        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, Any]]:
        """Resets the environment to the starting state
        and returns the initial observations for all agents.

        :return: Tuple of observations and infos for each agent.
        :rtype: tuple[dict[AgentID, ObsType], dict[AgentID, Any]]
        """
        self.t = 0
        self.game.reset_game()
        self.game.start_game()

        self._reset_action_delay_queues()

        # Reset previous action trackers
        self.prev_selected_actions = {
            agent: constants.EnvActions.NONE for agent in self.agents
        }
        self.prev_executed_actions = {
            agent: constants.EnvActions.NONE for agent in self.agents
        }

        # Reset special charge tracking
        self._holding_special_charge = {
            "p1": False,
            "p2": False,
        }

        # Reset reward budget
        self.reward_budget = {
            agent: self.win_reward_scaling_coeff for agent in self.agents
        }

        self.encoder.reset()

        self.last_game_state = self.game.get_state()

        observations = self.get_obs(
            self.last_game_state,
            self.prev_selected_actions,
            self._holding_special_charge,
            self.num_actions,
        )

        return observations, self.get_infos()

    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        """Step the environment with the provided actions for all agents.

        :param actions: Dictionary mapping agent ids to their actions for this step.
        :type actions: dict[AgentID, ActionType]
        :return: Tuple of observations, rewards, terminates, truncateds and infos for all agents.
        :rtype: tuple[ dict[AgentID, ObsType], dict[AgentID, float], dict[AgentID, bool], dict[AgentID, bool], dict[AgentID, dict[str, Any]], ]


        ACTION DELAY SYSTEM
        ===================
        Actions are delayed by K steps (action_delay_steps = action_delay_frames // frame_skip)
        to simulate human reaction time. A FIFO queue stores pending actions.

        With action_delay_frames=16 and frame_skip=4, K=4 steps (~267ms at 60fps).

        Timeline (K=4):
        ---------------
                          Queue State
        Step  Selected    [oldest → newest]      Executed   prev_selected (in obs)
        ────  ────────    ─────────────────      ────────   ──────────────────────
          0   FORWARD     [NONE,NONE,NONE,FWD]   NONE       NONE
          1   BACK        [NONE,NONE,FWD,BACK]   NONE       FORWARD
          2   ATTACK      [NONE,FWD,BACK,ATK]    NONE       BACK
          3   NONE        [FWD,BACK,ATK,NONE]    NONE       ATTACK
          4   FORWARD     [BACK,ATK,NONE,FWD]    FORWARD    NONE    <-- FWD from step 0 executes
          5   BACK        [ATK,NONE,FWD,BACK]    BACK       FORWARD <-- BACK from step 1 executes

        The agent observes prev_selected_actions (what it chose last step), enabling it to
        predict what will execute when a SPECIAL_CHARGE exits the queue.


        SPECIAL CHARGE TOGGLE SYSTEM
        ============================
        Special attacks require holding ATTACK for 60 frames (15 steps at frame_skip=4).
        SPECIAL_CHARGE is a toggle that enables/disables automatic ATTACK holding.

        When SPECIAL_CHARGE executes (exits the queue):
          - If not holding: Toggle ON, continue with charged version of executed action
          - If holding: Toggle OFF, continue with base version of executed action.


        Example with K=2 delay:
        -----------------------
        Step  Selected        Queue           Dequeued        Holding  Executed
        ────  ────────        ─────           ────────        ───────  ────────
          0   BACK            [NONE,BACK]     NONE            False    NONE
          1   SPECIAL_CHARGE  [BACK,SP_CHG]   NONE            False    NONE
          2   FORWARD         [SP_CHG,FWD]    BACK            False    BACK
          3   FORWARD         [FWD,FWD]       SPECIAL_CHARGE  True     ATTACK
          4   NONE            [FWD,NONE]      FORWARD         True     FWD_ATTACK
          5   NONE            [NONE,NONE]     FORWARD         True     FWD_ATTACK
          ...                                                 True     (continue holding)
         18   SPECIAL_CHARGE  [NONE,SP_CHG]   NONE            True     ATTACK
         19   NONE            [SP_CHG,NONE]   NONE            True     ATTACK
         20   NONE            [NONE,NONE]     SPECIAL_CHARGE  False    NONE

        """
        self.t += 1

        # =====================================================================
        # ACTION DELAY: Dequeue the action from K steps ago, enqueue current action
        # =====================================================================
        actions_to_execute: dict[AgentID, ActionType] = {}
        if self.action_delay_frames == 0:
            actions_to_execute = actions
        else:
            for agent_id in self.agents:
                # Dequeue: Get action selected K steps ago (this will execute now)
                actions_to_execute[agent_id] = self._action_queues[
                    agent_id
                ].popleft()
                # Enqueue: Add current selection (will execute in K steps)
                self._action_queues[agent_id].append(actions[agent_id])

        # =====================================================================
        # SPECIAL CHARGE TOGGLE: Process SPECIAL_CHARGE actions and apply holding
        # =====================================================================
        for agent_id in self.agents:
            holding_special_charge = self._holding_special_charge[agent_id]
            action_is_special_charge = (
                actions_to_execute[agent_id]
                in [
                    constants.EnvActions.SPECIAL_CHARGE,
                    constants.EnvActions.FORWARD_SPECIAL_CHARGE,
                    constants.EnvActions.BACK_SPECIAL_CHARGE,
                ]
            )

            # Toggle special charge state when SPECIAL_CHARGE action executes.
            # Use the base action while the logic for special charging is handled separately. 
            if action_is_special_charge and not holding_special_charge:
                self._holding_special_charge[agent_id] = True
                actions_to_execute[agent_id] = self._convert_special_charge_to_base_action(
                    actions_to_execute[agent_id]
                )
            elif action_is_special_charge and holding_special_charge:
                self._holding_special_charge[agent_id] = False
                actions_to_execute[agent_id] = self._convert_special_charge_to_base_action(
                    actions_to_execute[agent_id]
                )

            # While holding special charge, convert all actions to include ATTACK.
            # This enables charging toward NEUTRAL_SPECIAL or BACK_SPECIAL based
            # on directional input when ATTACK is eventually released.
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
            p1_action=p1_action, p2_action=p2_action, n_frames=self.frame_skip
        )
        observations = self.get_obs(
            game_state=game_state,
            prev_actions=actions,
            is_charging_special=self._holding_special_charge,
            num_actions=self.num_actions,
        )

        terminated = game_state.player1.is_dead or game_state.player2.is_dead

        rewards = {a_id: 0.0 for a_id in self.agents}
        # Apply guard break reward, if using.
        if self.guard_break_reward_value != 0:
            p1_prev_guard_health = self.last_game_state.player1.guard_health
            p2_prev_guard_health = self.last_game_state.player2.guard_health
            p1_guard_health = game_state.player1.guard_health
            p2_guard_health = game_state.player2.guard_health

            # Guard break reward is deducted from the overall "budget" of reward
            # to avoid biasing gameplay towards guard break. The total reward
            # always remains the same, but we can make the signal more dense by
            # providing guard break rewards. This can be turned off with
            # "use_reward_budget=False" in the environment config.
            if p2_guard_health < p2_prev_guard_health:
                if self.use_reward_budget:
                    self.reward_budget["p1"] -= self.guard_break_reward_value
                rewards["p1"] += self.guard_break_reward_value
                rewards["p2"] -= self.guard_break_reward_value
            if p1_guard_health < p1_prev_guard_health:
                if self.use_reward_budget:
                    self.reward_budget["p2"] -= self.guard_break_reward_value
                rewards["p2"] += self.guard_break_reward_value
                rewards["p1"] -= self.guard_break_reward_value

        # If the other player is dead, reward the player who is alive.
        # We apply rewards as remaining_reward_budget * is_dead + guard_break.
        # NOTE(chase): Both players can die at the same time in which case
        # the episode will still be zero-sum, but the remaining budget rewards
        # may differ.
        opponent_is_dead = {
            "p1": int(game_state.player2.is_dead),
            "p2": int(game_state.player1.is_dead),
        }

        for a_id, opp_dead in opponent_is_dead.items():
            other_agent_id = "p2" if a_id == "p1" else "p1"

            # Reward the agent for the opponent dying
            rewards[a_id] += self.reward_budget[a_id] * opp_dead

            # Penalize the opponent for dying
            rewards[other_agent_id] -= self.reward_budget[a_id] * opp_dead

        terminateds = {
            "p1": terminated,
            "p2": terminated,
            "__all__": terminated,
        }

        truncated = self.t >= self.max_t
        truncateds = {
            "p1": truncated,
            "p2": truncated,
            "__all__": truncated,
        }

        self.last_game_state = game_state
        self.prev_executed_actions = actions_to_execute
        self.prev_selected_actions = actions

        # ~~~ START: For debugging game build! ~~~
        # encoded_state = self.game.get_encoded_state()
        # encoded_state_dict = {
        #     "p1": np.asarray(
        #         encoded_state.player1_encoding, dtype=np.float32
        #     ),
        #     "p2": np.asarray(
        #         encoded_state.player2_encoding, dtype=np.float32
        #     ),
        # }

        # for a_id, ob in observations.items():
        #     matched_obs = np.isclose(ob, encoded_state_dict[a_id]).all()
        #     assert matched_obs
        # ~~~ END: For debugging game build! ~~~
        self._validate_action_queues()

        # ~~~ END: For debugging action queue! ~~~

        # if not self.evaluation:
        #     print("===== Step:", self.t, "=====")
        #     print("Selected Action: ", actions["p1"])
        #     print("Executed Action: ", actions_to_execute["p1"])
        #     print("Holding Special Charge: ", self._holding_special_charge["p1"])
        #     print("Action Queue: ", self._action_queues["p1"])

        #     if self.t % 30 == 0:
        #         import sys
        #         sys.exit(1)
        # ~~~ END: For debugging action queue! ~~~

        return observations, rewards, terminateds, truncateds, self.get_infos()

    def get_infos(self):
        infos = {agent: {} for agent in self.agents}
        if self.return_fight_state_in_infos:
            infos.update(self._get_fight_state_dicts())

        if self.return_action_mask_in_infos:
            for agent_id in self.agents:
                infos[agent_id]["action_mask"] = self.get_action_mask(agent_id)
        
        return infos

    def get_action_mask(self, agent_id: str) -> np.ndarray:
        """Get action mask for the given agent.
        
        If they are holding special charge, then the only available actions are:
            - ATTACK
            - FORWARD_ATTACK
            - BACKWARD_ATTACK
            - SPECIAL_CHARGE
            - BACK_SPECIAL_CHARGE
            - FORWARD_SPECIAL_CHARGE
        Otherwise, the full action space is available. 
        """
        # Check if the agent is holding special charge and only allow
        # the specified actions if so. 
        if self._holding_special_charge[agent_id]:
            mask = np.zeros(self.action_space[agent_id].n, dtype=np.float32)
            mask[self.SPECIAL_CHARGE_ALLOWED_ACTIONS] = 1.0
                
            return mask
        
        # If not holding special charge, return all actions as available
        action_space = self.action_space[agent_id]
        return np.ones(action_space.n, dtype=np.float32)

    def _build_charged_special_queue(self):
        assert self.SPECIAL_CHARGE_FRAMES % self.frame_skip == 0
        steps_to_apply_attack = int(
            self.SPECIAL_CHARGE_FRAMES // self.frame_skip
        )
        return steps_to_apply_attack

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

        raise ValueError(f"Invalid special charge action: {action}, expected one of SPECIAL_CHARGE, FORWARD_SPECIAL_CHARGE, BACK_SPECIAL_CHARGE.")

    def _build_charged_queue_features(self):
        return {
            "p1": {
                "special_charge_queue": self.special_charge_queue["p1"]
                / self.SPECIAL_CHARGE_FRAMES
            },
            "p2": {
                "special_charge_queue": self.special_charge_queue["p2"]
                / self.SPECIAL_CHARGE_FRAMES
            },
        }
