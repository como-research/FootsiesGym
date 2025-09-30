from typing import Any
import time 
import numpy as np
from gymnasium import spaces
from ray.rllib import env
import collections
from ray.rllib.env import env_context

from . import encoder, typing
import os
import platform
import subprocess
import zipfile
import portpicker

from .game import constants, footsies_game
from ..binary_manager import get_binary_manager


class FootsiesEnv(env.MultiAgentEnv):
    metadata = {"render.modes": ["human"]}
    LINUX_ZIP_PATH_HEADLESS = "binaries/footsies_linux_server_021725.zip"
    LINUX_ZIP_PATH_WINDOWED = "binaries/footsies_linux_windowed_021725.zip"
    SPECIAL_CHARGE_FRAMES = 60

    observation_space = spaces.Dict(
        {
            agent: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(encoder.FootsiesEncoder.observation_size,),
            )
            for agent in ["p1", "p2"]
        }
    )

    action_space = spaces.Dict(
        {
            agent: spaces.Discrete(
                len(
                    [
                        constants.EnvActions.NONE,
                        constants.EnvActions.BACK,
                        constants.EnvActions.FORWARD,
                        constants.EnvActions.ATTACK,
                        constants.EnvActions.BACK_ATTACK,
                        constants.EnvActions.FORWARD_ATTACK,
                        # NOTE(chase): This is a special input that holds down
                        # attack for 60 frames. It's just too long of a sequence
                        # to easily learn by holding ATTACK for so long.
                        constants.EnvActions.SPECIAL_CHARGE,
                    ]
                )
            )
            for agent in ["p1", "p2"]
        }
    )

    def __init__(self, config: dict[Any, Any] = None):
        super(FootsiesEnv, self).__init__()

        if config is None:
            config = {}
        self.config = config
        self.return_fight_state_in_infos = config.get("return_fight_state_in_infos", False)
        self.use_build_encoding = config.get("use_build_encoding", False)
        self.agents: list[typing.AgentID] = ["p1", "p2"]
        self.possible_agents: list[typing.AgentID] = self.agents.copy()
        self._agent_ids: set[typing.AgentID] = set(self.agents)
        self.win_reward_scaling_coeff = self.config.get("win_reward_scaling_coeff", 10.0)
        self.guard_break_reward_value = self.config.get("guard_break_reward", 0)
        self.use_reward_budget = self.config.get("use_reward_budget", True)
        assert self.guard_break_reward_value * 3 < self.win_reward_scaling_coeff, (
            "Guard break reward total must be less than the win reward (guard break reward * 3 < win reward)"
        )

        self.reward_budget = {agent: self.win_reward_scaling_coeff for agent in self.agents}


        self.evaluation = config.get("evaluation", False)

        self.t: int = 0
        self.max_t: int = config.get("max_t", 1000)
        self.frame_skip: int = config.get("frame_skip", 4)
        self.action_delay_frames: int = config.get("action_delay", 8)

        assert (
            self.action_delay_frames % self.frame_skip == 0
        ), "action_delay must be divisible by frame_skip"

        self.action_delay_steps: int = self.action_delay_frames // self.frame_skip
        self.encoder = encoder.FootsiesEncoder()
        self._action_queues: dict[typing.AgentID, collections.deque[int]] = None
        self.prev_actions: dict[typing.AgentID, int] = {agent: constants.EnvActions.NONE for agent in self.agents}
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
        p1_state, p2_state = self.last_game_state.player1, self.last_game_state.player2

        dist_x = np.abs(p1_state.player_position_x - p2_state.player_position_x)

        for player, opp_state in zip(["p1", "p2"], [p2_state, p1_state]):
            fight_state_dict[player]["distance_x"] = dist_x
            fight_state_dict[player]["is_opponent_damage"] = opp_state.current_action_id == constants.ActionID.DAMAGE
            fight_state_dict[player]["is_opponent_guard_break"] = opp_state.current_action_id == constants.ActionID.GUARD_BREAK
            fight_state_dict[player]["is_opponent_blocking"] = opp_state.current_action_id in [constants.ActionID.GUARD_CROUCH, constants.ActionID.GUARD_STAND, constants.ActionID.GUARD_M]
            fight_state_dict[player]["is_opponent_normal_attack"] = opp_state.current_action_id in [constants.ActionID.N_ATTACK, constants.ActionID.B_ATTACK]
            fight_state_dict[player]["is_opponent_special_attack"] = opp_state.current_action_id in [constants.ActionID.N_SPECIAL, constants.ActionID.B_SPECIAL]
    

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
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        binary_subdir = "footsies_binaries_headless" if self.headless else "footsies_binaries_windowed"
        binary_path = os.path.join(project_root, "binaries", binary_subdir, "footsies.x86_64")

        if not os.path.exists(binary_path):
            # Use binary manager to download and extract binaries atomically
            binary_manager = get_binary_manager()
            
            # Ensure binaries are downloaded and extracted (with file locking to prevent race conditions)
            binaries_dir = os.path.join(project_root, "binaries")
            if not binary_manager.ensure_binaries_extracted("linux", target_dir=binaries_dir, headless=self.headless):
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
        if not self.headless and not os.environ.get('DISPLAY'):
            print("⚠️  Warning: DISPLAY environment variable not set. Windowed mode may not work in WSL.")
            print("   For WSL2 with Windows 11, WSLg should handle this automatically.")
            print("   For older WSL versions, you may need to set up X11 forwarding.")
        
        print("Launching with command:", command)
        
        # For windowed mode, don't suppress output as it may contain important display messages
        # For headless mode, suppress output to keep it clean
        if self.headless:
            # Headless mode - suppress output
            self.server_process = subprocess.Popen(
                command, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        else:
            # Windowed mode - allow output for display setup (important for WSL)
            self.server_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        binary_type = "headless" if self.headless else "windowed"
        print(f"Launched {binary_type} footsies binary on port {port}.")
        time.sleep(5)

    def close(self):
        """Clean up resources when the environment is closed."""
        if hasattr(self, 'server_process') and self.server_process is not None:
            try:
                self.server_process.terminate()
                # Give it a moment to terminate gracefully
                self.server_process.wait(timeout=5)
                print(f"Terminated footsies server process (PID: {self.server_process.pid}).")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                self.server_process.kill()
                self.server_process.wait()
                print(f"Force killed footsies server process (PID: {self.server_process.pid}).")
            except Exception as e:
                print(f"Error terminating server process: {e}")
            finally:
                self.server_process = None

    def __del__(self):
        """Ensure cleanup happens when the object is garbage collected."""
        self.close()

    def _reset_action_delay_queues(self):
        self._action_queues: dict[typing.AgentID, collections.deque[int]] = {
            agent_id: collections.deque([constants.EnvActions.NONE] * self.action_delay_steps, maxlen=self.action_delay_steps)
            for agent_id in self.agents
        }
    
    def _validate_action_queues(self):
        for agent_id in self.agents:
            assert len(self._action_queues[agent_id]) == self.action_delay_steps, (
                f"Action queue has the incorrect number of queued actions! "
                " Observed {len(self._action_queues[agent_id])}, expected {self.action_delay_steps}"
            )

    def get_obs(self, game_state, prev_actions, is_charging_special: dict[typing.AgentID, bool]):
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
            # return encoded_state_dict
        else:
            return self.encoder.encode(game_state, prev_actions, is_charging_special)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[
        dict[typing.AgentID, typing.ObsType], dict[typing.AgentID, Any]
    ]:
        """Resets the environment to the starting state
        and returns the initial observations for all agents.

        :return: Tuple of observations and infos for each agent.
        :rtype: tuple[dict[typing.AgentID, typing.ObsType], dict[typing.AgentID, Any]]
        """
        self.t = 0
        self.game.reset_game()
        self.game.start_game()

        self._reset_action_delay_queues()

        # Reset reward budget
        self.reward_budget = {agent: self.win_reward_scaling_coeff for agent in self.agents}

        self.encoder.reset()

        self.last_game_state = self.game.get_state()

        observations = self.get_obs(self.last_game_state, self.prev_actions, self._holding_special_charge)

        return observations, self.get_infos()

    def step(self, actions: dict[typing.AgentID, typing.ActionType]) -> tuple[
        dict[typing.AgentID, typing.ObsType],
        dict[typing.AgentID, float],
        dict[typing.AgentID, bool],
        dict[typing.AgentID, bool],
        dict[typing.AgentID, dict[str, Any]],
    ]:
        """Step the environment with the provided actions for all agents.

        :param actions: Dictionary mapping agent ids to their actions for this step.
        :type actions: dict[typing.AgentID, typing.ActionType]
        :return: Tuple of observations, rewards, terminates, truncateds and infos for all agents.
        :rtype: tuple[ dict[typing.AgentID, typing.ObsType], dict[typing.AgentID, float], dict[typing.AgentID, bool], dict[typing.AgentID, bool], dict[typing.AgentID, dict[str, Any]], ]
        """
        self.t += 1

        # Update action queue -> dequeue old actions, enqueue new actions
        actions_to_execute: dict[typing.AgentID, typing.ActionType] = {}
        if self.action_delay_frames == 0:
            actions_to_execute = actions
        else:
            for agent_id in self.agents:
                actions_to_execute[agent_id] = self._action_queues[agent_id].popleft()
                self._action_queues[agent_id].append(actions[agent_id])

        for agent_id in self.agents:
            holding_special_charge = self._holding_special_charge[agent_id]
            action_is_special_charge = (
                actions_to_execute[agent_id] == constants.EnvActions.SPECIAL_CHARGE
            )

            # Toggle the special charge based on whether or not we're holding special already
            if action_is_special_charge and not holding_special_charge:
                self._holding_special_charge[agent_id] = True
            elif action_is_special_charge and holding_special_charge:
                self._holding_special_charge[agent_id] = False
                actions_to_execute[agent_id] = self.prev_actions[agent_id]

            if self._holding_special_charge[agent_id]:
                actions_to_execute[agent_id] = self._convert_to_charge_action(
                    actions_to_execute[agent_id]
                )

        p1_action = self.game.action_to_bits(actions_to_execute["p1"], is_player_1=True)
        p2_action = self.game.action_to_bits(actions_to_execute["p2"], is_player_1=False)

        game_state = self.game.step_n_frames(
            p1_action=p1_action, p2_action=p2_action, n_frames=self.frame_skip
        )
        observations = self.get_obs(game_state, actions_to_execute, self._holding_special_charge)

        terminated = game_state.player1.is_dead or game_state.player2.is_dead


        rewards = {a_id: 0.0 for a_id in self.agents} 
        # Apply guard-break reward, if using. 
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
            "p2": int(game_state.player1.is_dead)
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
        self.prev_actions = actions_to_execute

        # ~~~ For debugging game build! ~~~
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
        # ~~~ END ~~~
        self._validate_action_queues()

        return observations, rewards, terminateds, truncateds, self.get_infos()

    def get_infos(self):
        infos = {agent: {} for agent in self.agents}
        if self.return_fight_state_in_infos:
            infos.update(self._get_fight_state_dicts())
        return infos

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
        else:
            return constants.EnvActions.ATTACK

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
