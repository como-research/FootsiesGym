from typing import Any
import time 
import numpy as np
from gymnasium import spaces
from ray.rllib import env

from . import encoder, typing
import os
import platform
import socket
import subprocess
import zipfile

from .game import constants, footsies_game
from ..binary_manager import get_binary_manager


class FootsiesEnv(env.MultiAgentEnv):
    metadata = {"render.modes": ["human"]}
    LINUX_ZIP_PATH_HEADLESS = "binaries/footsies_linux_server_021725.zip"
    LINUX_ZIP_PATH_WINDOWED = "binaries/footsies_linux_windowed_021725.zip"
    SPECIAL_CHARGE_FRAMES = 60
    GUARD_BREAK_REWARD = 0.3

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
        self.use_build_encoding = config.get("use_build_encoding", False)
        self.agents: list[typing.AgentID] = ["p1", "p2"]
        self.possible_agents: list[typing.AgentID] = self.agents.copy()
        self._agent_ids: set[typing.AgentID] = set(self.agents)

        self.evaluation = config.get("evaluation", False)

        self.t: int = 0
        self.max_t: int = config.get("max_t", 1000)
        self.frame_skip = config.get("frame_skip", 4)
        observation_delay = config.get("observation_delay", 16)

        assert (
            observation_delay % self.frame_skip == 0
        ), "observation_delay must be divisible by frame_skip"

        self.encoder = encoder.FootsiesEncoder(
            observation_delay=observation_delay // self.frame_skip
        )


        
        port = config.get("port", None)
        self.headless = config.get("headless", True)
        # we'll start game servers at 50051 for training
        # and 40051 for evaluation. Worker index starts at
        # 1, so we won't see 40050/50050.
        if port is None:
            start_port = 40050 if self.evaluation else 50050
            port = (
                start_port
                + int(config.get("worker_index", 0))
                * config.get("num_envs_per_worker", 1)
                + config.get("vector_index", 0)
            )

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
        self.special_charge_queue = {
            "p1": -1,
            "p2": -1,
        }

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('localhost', port))
                return False
            except OSError:
                return True

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

        # Check if the port is already in use
        if self._is_port_in_use(port):
            print(f"Port {port} is already in use. Skipping binary launch.")
            return

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


    def get_obs(self, game_state):
        if self.use_build_encoding:
            encoded_state = self.game.get_encoded_state()
            encoded_state_dict = {
                "p1": np.asarray(
                    encoded_state.player1_encoding, dtype=np.float32
                ),
                "p2": np.asarray(
                    encoded_state.player2_encoding, dtype=np.float32
                ),
            }
            return encoded_state_dict
        else:
            return self.encoder.encode(game_state)

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

        self.encoder.reset()

        if not self.use_build_encoding:
            self.last_game_state = self.game.get_state()

        observations = self.get_obs(self.last_game_state)

        return observations, {agent: {} for agent in self.agents}

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

        for agent_id in self.agents:
            empty_queue = self.special_charge_queue[agent_id] < 0
            action_is_special_charge = (
                actions[agent_id] == constants.EnvActions.SPECIAL_CHARGE
            )

            # Refill the charge queue only if we're not already in a special charge.
            if action_is_special_charge and empty_queue:
                self.special_charge_queue[agent_id] = (
                    self._build_charged_special_queue()
                )

            if self.special_charge_queue[agent_id] >= 0:
                self.special_charge_queue[agent_id] -= 1
                actions[agent_id] = self._convert_to_charge_action(
                    actions[agent_id]
                )

        p1_action = self.game.action_to_bits(actions["p1"], is_player_1=True)
        p2_action = self.game.action_to_bits(actions["p2"], is_player_1=False)

        game_state = self.game.step_n_frames(
            p1_action=p1_action, p2_action=p2_action, n_frames=self.frame_skip
        )
        observations = self.get_obs(game_state)

        terminated = game_state.player1.is_dead or game_state.player2.is_dead

        # Zero-sum game: 1 if other player is dead, -1 if you're dead:
        rewards = {
            "p1": int(game_state.player2.is_dead)
            - int(game_state.player1.is_dead),
            "p2": int(game_state.player1.is_dead)
            - int(game_state.player2.is_dead),
        }

        if self.config.get("reward_guard_break", False):
            p1_prev_guard_health = self.last_game_state.player1.guard_health
            p2_prev_guard_health = self.last_game_state.player2.guard_health
            p1_guard_health = game_state.player1.guard_health
            p2_guard_health = game_state.player2.guard_health

            if p2_guard_health < p2_prev_guard_health:
                rewards["p1"] += self.GUARD_BREAK_REWARD
                rewards["p2"] -= self.GUARD_BREAK_REWARD
            if p1_guard_health < p1_prev_guard_health:
                rewards["p2"] += self.GUARD_BREAK_REWARD
                rewards["p1"] -= self.GUARD_BREAK_REWARD

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

        return observations, rewards, terminateds, truncateds, self.get_infos()

    def get_infos(self):
        return {agent: {} for agent in self.agents}

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
