
import collections
import dataclasses 

from numpy import random


from footsiesgym.footsies.typing import EnvID, AgentID, ActionType, ActionBits 
from footsiesgym.footsies.game import constants

class FightState:
    def __init__(self, distance_x: float, is_opponent_damage: bool, is_opponent_guard_break: bool, is_opponent_blocking: bool, is_opponent_normal_attack: bool, is_opponent_special_attack: bool, is_facing_right: bool, **kwargs):
        
        
        self.distance_x = distance_x
        self.is_opponent_damage = is_opponent_damage
        self.is_opponent_guard_break = is_opponent_guard_break
        self.is_opponent_blocking = is_opponent_blocking
        self.is_opponent_normal_attack = is_opponent_normal_attack
        self.is_opponent_special_attack = is_opponent_special_attack
        self.is_facing_right = is_facing_right

    @classmethod
    def from_dict(cls, fight_state_dict: dict[str, int | float | bool]) -> "FightState":
        return cls(**fight_state_dict)


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

    @staticmethod
    def noop_movement(steps: int) -> list[ActionBits]:
        return [constants.ActionBits.NONE] * steps


        
class FootsiesBot:
    """
    Reimplementation of the Footsies BattleAI, a rule-based agent used to benchmark performance of trained agents. 
    The original C# implementation can be found here: https://github.com/chasemcd/FootsiesV2/blob/main/Assets/Script/BattleAI.cs
    """

    def __init__(self, frame_skip: int = 4):
        self.frame_skip: int = frame_skip
        self.move_queues: dict[EnvID, dict[AgentID, collections.deque[ActionType]]] = collections.defaultdict(lambda: collections.defaultdict(collections.deque))
        self.attack_queues: dict[EnvID, dict[AgentID, collections.deque[ActionType]]] = collections.defaultdict(lambda: collections.defaultdict(collections.deque))
        self.override_active = False

    def get_next_input(self, env_id: EnvID, agent_id: AgentID, fight_state_dict: dict[str, int | float | bool]) -> ActionType:
        action_bits: ActionBits = constants.ActionBits.NONE
        move_queue: collections.deque[ActionType] = self.move_queues[env_id][agent_id]
        attack_queue: collections.deque[ActionType] = self.attack_queues[env_id][agent_id]
        
        fight_state = FightState.from_dict(fight_state_dict)

        if not move_queue:
            move_queue.extend(self._select_movement(fight_state))
        action_bits |= move_queue.popleft()
        
        if not attack_queue:
            attack_queue.extend(self._select_attack(fight_state))
            self.override_active = False
        action_bits |= attack_queue.popleft()

        quick_whiff_punish_prob = 0.90
        if random.rand() < quick_whiff_punish_prob and not self.override_active and fight_state.distance_x < 2.5 and (fight_state.is_opponent_damage or fight_state.is_opponent_guard_break or fight_state.is_opponent_special_attack):
            attack_queue.clear()
            attack_queue.extend(self.two_hit_immediate_attack())
            action_bits = attack_queue.popleft()
            self.override_active = True
        
        return constants.BITS_TO_ACTIONS[action_bits]

    def _select_movement(self, fight_state: FightState) -> list[ActionBits]:
        if fight_state.distance_x > 4.0:
            self.last_move_dist = 4.0
            return self.far_approach(dash=random.choice([True, False]), is_facing_right=fight_state.is_facing_right)
        elif fight_state.distance_x > 3.5:
            self.last_move_dist = 3.5
            randint_ = random.randint(0, 7)
            if randint_ <= 3:
                return self.mid_approach(dash=random.choice([True, False]), is_facing_right=fight_state.is_facing_right)
            elif randint_ <= 5:
                return self.far_approach(dash=random.choice([True, False]), is_facing_right=fight_state.is_facing_right)
            else:
                return ActionSequences.noop_movement(steps=8 // self.frame_skip)
        elif fight_state.distance_x > 3.0:
            self.last_move_dist = 3.0
            randint_ = random.randint(0, 6)
            if randint_ <= 0:
                return self.mid_approach(dash=random.choice([True, False]), is_facing_right=fight_state.is_facing_right)
            elif randint_ <= 5:
                return self.fallback_movement(dash=random.choice([True, False]), is_facing_right=fight_state.is_facing_right)
            else:
                return ActionSequences.noop_movement(steps=8 // self.frame_skip)
        elif fight_state.distance_x > 2.0:
            self.last_move_dist = 2.0
            randint_ = random.randint(0, 5)
            if randint_ <= 3:
                return self.fallback_movement(dash=random.choice([True, False]), is_facing_right=fight_state.is_facing_right)
            else:
                return ActionSequences.noop_movement(steps=4 // self.frame_skip)
        else:
            self.last_move_dist = 1.0
            randint_ = random.randint(0, 3)
            if randint_ <= 1:
                return self.fallback_movement(dash=random.choice([True, False]), is_facing_right=fight_state.is_facing_right)
            else:
                return ActionSequences.noop_movement(steps=12 // self.frame_skip)

    def far_approach(self, dash: bool, is_facing_right: bool) -> list[ActionBits]:
        """Get action sequences for a far approach, either two dashes in or walking in with some backward inputs."""
        if dash:
            return [
                *ActionSequences.forward_dash(is_facing_right),
                *ActionSequences.back_input(is_facing_right, steps=10 // self.frame_skip),
            ] * 2
        
        return [
            *ActionSequences.forward_input(is_facing_right=is_facing_right, steps=16 // self.frame_skip),
            *ActionSequences.back_input(is_facing_right=is_facing_right, steps=12 // self.frame_skip),
            *ActionSequences.forward_input(is_facing_right=is_facing_right, steps=8 // self.frame_skip),
            *ActionSequences.back_input(is_facing_right=is_facing_right, steps=4 // self.frame_skip),
        ]
        
    def mid_approach(self, dash: bool, is_facing_right: bool) -> list[ActionBits]:
        """Get action sequences for a mid approach, either two dashes in or walking in with some backward inputs."""
        if dash:
            return [
                *ActionSequences.forward_dash(is_facing_right),
                *ActionSequences.back_input(is_facing_right, steps=8 // self.frame_skip),
            ]
        
        return [
            *ActionSequences.forward_input(is_facing_right=is_facing_right, steps=16 // self.frame_skip),
            *ActionSequences.back_input(is_facing_right=is_facing_right, steps=12 // self.frame_skip),
            *ActionSequences.forward_input(is_facing_right=is_facing_right, steps=16 // self.frame_skip),
            *ActionSequences.back_input(is_facing_right=is_facing_right, steps=12 // self.frame_skip),
        ]

    def fallback_movement(self, dash: bool, is_facing_right: bool) -> list[ActionBits]:
        """Get action sequences for a fallback movement."""
        if dash:
            return [
                *ActionSequences.back_dash(is_facing_right),
                *ActionSequences.back_input(is_facing_right, steps=16 // self.frame_skip),
            ] 
        
        return [
            *ActionSequences.back_input(is_facing_right=is_facing_right, steps=24 // self.frame_skip),
        ]
        
    def _select_attack(self, fight_state: FightState) -> list[ActionBits]:
        """Get action sequences for an attack."""
        if (fight_state.is_opponent_damage or fight_state.is_opponent_guard_break or fight_state.is_opponent_special_attack):
            return self.two_hit_immediate_attack()
        elif fight_state.distance_x > 3.5:
            self.last_attack_dist = 3.5
            # NOTE(chase): I know this is incorrect, I'm copying the logic exactly from Footsies
            # where the delayed special here will never be triggered (we need randint(0, 5) for that).
            randint_ = random.randint(0, 4)
            if randint_ <= 3:
                return ActionSequences.noop_movement(steps=20 // self.frame_skip)
            else: 
                # TODO: This is unreachable. If desired, fix it. 
                return self.delayed_special_attack()
        elif fight_state.distance_x > 3.0:
            self.last_attack_dist = 3.0
            if fight_state.is_opponent_normal_attack:
                return self.two_hit_immediate_attack()
            else: 
                return ActionSequences.noop_movement(steps=8 // self.frame_skip)

        elif fight_state.distance_x > 2.5:
            self.last_attack_dist = 2.5
            randint_ = random.randint(0, 3)
            if randint_ <= 1:
                return ActionSequences.noop_movement(steps=8 // self.frame_skip)
            else:
                return self.one_hit_immediate_attack()

        elif fight_state.distance_x > 2.0:
            self.last_attack_dist = 2.0
            randint_ = random.randint(0, 6)
            if randint_ <= 1:
                return ActionSequences.noop_movement(steps=8 // self.frame_skip)
            elif randint_ <= 2:
                return self.one_hit_immediate_attack()
            elif randint_ <= 3:
                return self.two_hit_immediate_attack()
            elif randint_ <= 4:
                return self.immediate_special_attack()
            else:
                return self.delayed_special_attack()
        else:
            randint_ = random.randint(0, 3)
            if randint_ == 0:
                return self.one_hit_immediate_attack()
            else:
                return self.two_hit_immediate_attack()

    def two_hit_immediate_attack(self) -> list[ActionBits]:
        """Get action sequences for a two hit immediate attack."""
        sequence = [constants.ActionBits.ATTACK]
        sequence += [constants.ActionBits.NONE] * max(3 // self.frame_skip, 1)
        sequence += [constants.ActionBits.ATTACK]
        sequence += [constants.ActionBits.NONE] * max(4 // self.frame_skip, 1)
        return sequence
    
    def one_hit_immediate_attack(self) -> list[ActionBits]:
        sequence = [constants.ActionBits.ATTACK]
        sequence += [constants.ActionBits.NONE] * max(8 // self.frame_skip, 1)
        return sequence

    def immediate_special_attack(self) -> list[ActionBits]:
        sequence = [constants.ActionBits.ATTACK] * max(60 // self.frame_skip, 1)
        sequence += [constants.ActionBits.NONE] * max(4 // self.frame_skip, 1)
        return sequence

    def delayed_special_attack(self) -> list[ActionBits]:
        sequence = [constants.ActionBits.ATTACK] * max(120 // self.frame_skip, 1)
        sequence += [constants.ActionBits.NONE] * max(4 // self.frame_skip, 1)
        return sequence
