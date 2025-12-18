from ray.rllib.algorithms import callbacks as rllib_callbacks
from ray.rllib.connectors import connector as rllib_connector


class AttachConnectors(rllib_callbacks.DefaultCallbacks):
    def __init__(self, focal_policy_ids: list[str] = ["focal_policy"], action_connectors: list | None = None, agent_connectors: list | None = None):
        self.focal_policy_ids = focal_policy_ids
        self.action_connectors = action_connectors or []
        self.agent_connectors = agent_connectors or []

    def on_create_policy(self, *, policy_id: str, policy: "Policy") -> None:
        if policy_id not in self.focal_policy_ids:
            return
            
        ctx = rllib_connector.ConnectorContext.from_policy(policy)
        for connector in self.action_connectors:
            policy.action_connectors.append(connector(ctx))
        for connector in reversed(self.agent_connectors):
            policy.agent_connectors.prepend(connector(ctx))