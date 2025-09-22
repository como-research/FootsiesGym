
import collections
import dataclasses 

from footsiesgym.footsies.typing import EnvID, AgentID, ActionType, ActionBits 
from footsiesgym.footsies.game import constants

class FightState:
    distance_x: float
    is_opponent_damage: bool 
    is_opponent_guard_break: bool
    is_opponent_blocking: bool
    is_opponent_normal_attack: bool
    is_opponent_special_attack: bool
    is_facing_right: bool

class ActionSequences:

    @staticmethod
    def forward_dash(is_facing_right: bool) -> list[ActionBits]:
        if is_facing_right:
            return [constants.ActionBits.RIGHT,constants.ActionBits.NONE, constants.ActionBits.RIGHT]
        else:
            return [constants.ActionBits.LEFT, constants.ActionBits.NONE, constants.ActionBits.LEFT]

    @staticmethod
    def back_dash(is_facing_right: bool) -> list[ActionBits]:
        return ActionSequences.forward_dash(not is_facing_right)

    @staticmethod
    def forward_input(is_facing_right: bool, steps: int) -> list[ActionBits]:
        if is_facing_right:
            return [constants.ActionBits.RIGHT] * steps
        else:
            return [constants.ActionBits.LEFT] * steps

    @staticmethod
    def back_input(is_facing_right: bool, steps: int) -> list[ActionBits]:
        return ActionSequences.forward_input(not is_facing_right, steps)
        

    


class FootsiesBot:
    """
    Reimplementation of the Footsies BattleAI, a rule-based agent used to benchmark performance of trained agents. 
    The original C# implementation can be found here: https://github.com/chasemcd/FootsiesV2/blob/main/Assets/Script/BattleAI.cs
    """

    def __init__(self, frame_skip: int = 4):
        self.frame_skip: int = frame_skip
        self.move_queues: dict[EnvID, dict[AgentID, collections.deque[ActionType]]] = collections.defaultdict(lambda: collections.defaultdict(collections.deque))
        self.attack_queues: dict[EnvID, dict[AgentID, collections.deque[ActionType]]] = collections.defaultdict(lambda: collections.defaultdict(collections.deque))


    def get_next_input(self, env_id: EnvID, agent_id: AgentID) -> ActionType:
        action_bits: ActionBits = constants.ActionBits.NONE
        move_queue: collections.deque[ActionType] = self.move_queues[env_id][agent_id]
        attack_queue: collections.deque[ActionType] = self.attack_queues[env_id][agent_id]
        fight_state = FightState(distance_x=0, is_opponent_damage=False, is_opponent_guard_break=False, is_opponent_blocking=False, is_opponent_normal_attack=False, is_opponent_special_attack=False, is_facing_right=True)

        if move_queue:
            action_bits |= move_queue.popleft()
        else:
            self._select_movement(fight_state)
        
        
        if attack_queue:
            action_bits |= attack_queue.popleft()
        else:
            self._select_attack(fight_state)

        return constants.BITS_TO_ACTIONS[action_bits]
        
        

    @staticmethod
    def _select_movement(self, fight_state: FightState) -> list[ActionBits]:
        if fight_state.distance_x > 4.0:
            return self.add_far_approach(dash=random.choice([True, False]), is_facing_right=fight_state.is_facing_right)
        
    @staticmethod
    def add_far_approach(self, dash: bool, is_facing_right: bool) -> list[ActionBits]:
        if dash:
            return [constants.ActionBits.DASH_FORWARD]
        

        

    @staticmethod
    def _select_attack(self, fight_state: FightState) -> list[ActionBits]:
        ...


