import collections
import dataclasses

import numpy as np
from ray.rllib.utils.typing import EpisodeType


@dataclasses.dataclass
class Matchup:
    p1: str
    p2: str
    prob: float


def _parse_agent_id(agent_id: str) -> tuple[str, str]:
    """Parse agent ID into (base, game_index).

    'p1'   -> ('p1', '')
    'p1_3' -> ('p1', '3')
    """
    if "_" in agent_id:
        base, idx = agent_id.rsplit("_", 1)
        return base, idx
    return agent_id, ""


class Matchmaker:
    def __init__(self, matchups: list[Matchup]):
        self.matchups = matchups
        self.probs = [matchup.prob for matchup in matchups]
        self.current_matchups = collections.defaultdict(dict)

    def _map_agent(self, agent_id: str, episode: EpisodeType) -> str:
        """Core mapping logic shared by both API variants.

        Supports both plain agent IDs ('p1', 'p2') and indexed
        vectorized IDs ('p1_0', 'p2_3'). For indexed IDs, matchups
        are keyed per (episode, game_index) so each game instance
        gets an independent matchup sample.
        """
        ep_id = episode.env_id if hasattr(episode, "env_id") else episode.id_
        base_agent, game_idx = _parse_agent_id(agent_id)

        # Compound key: unique per episode × game instance
        matchup_key = (ep_id, game_idx)

        if self.current_matchups.get(matchup_key) is None:
            sampled_matchup = np.random.choice(self.matchups, p=self.probs)
            policies = [sampled_matchup.p1, sampled_matchup.p2]
            p1, p2 = np.random.choice(policies, size=2, replace=False)
            self.current_matchups[matchup_key]["p1"] = p1
            self.current_matchups[matchup_key]["p2"] = p2

        pid = self.current_matchups[matchup_key].pop(base_agent)

        if not self.current_matchups[matchup_key]:
            del self.current_matchups[matchup_key]

        return pid

    def policy_mapping_fn(self, agent_id: str, episode: EpisodeType, **kwargs) -> str:
        """Old API stack: policy_mapping_fn."""
        return self._map_agent(agent_id, episode)

    def agent_to_module_mapping_fn(
        self, agent_id: str, episode: EpisodeType, **kwargs
    ) -> str:
        """New API stack: agent_to_module_mapping_fn."""
        return self._map_agent(agent_id, episode)
