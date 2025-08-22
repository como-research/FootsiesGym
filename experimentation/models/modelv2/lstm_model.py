from gymnasium import spaces
from ray.rllib.models.torch import recurrent_net

# from ray.rllib.models.torch import torch_modelv2
from ray.rllib.utils import framework
from ray.rllib.utils import typing as rllib_typing

torch, nn = framework.try_import_torch()


class LSTMModel(recurrent_net.RecurrentNetwork, nn.Module):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: rllib_typing.ModelConfigDict,
        name: str,
        **kwargs,
    ):
        recurrent_net.RecurrentNetwork.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        input_size = obs_space.shape[0]
        self.lstm_cell_size = kwargs.get("lstm_cell_size", 128)
        self.lstm = nn.LSTM(input_size, self.lstm_cell_size, batch_first=True)

        policy_dense_widths = kwargs.get("policy_dense_widths", [256, 256])
        model_body = [
            nn.Linear(self.lstm_cell_size, policy_dense_widths[0]),
            nn.ReLU(),
        ]
        for i in range(1, len(policy_dense_widths)):
            model_body.extend(
                [
                    nn.Linear(
                        policy_dense_widths[i - 1], policy_dense_widths[i]
                    ),
                    nn.ReLU(),
                ]
            )
        self.model_body = nn.Sequential(*model_body)

        self.pi = nn.Linear(policy_dense_widths[-1], num_outputs)
        self.vf = nn.Linear(policy_dense_widths[-1], 1)

    def forward_rnn(
        self,
        inputs: rllib_typing.TensorType,
        state: list[rllib_typing.TensorType],
        seq_lens: rllib_typing.TensorType,
    ) -> recurrent_net.Tuple[
        rllib_typing.TensorType, list[rllib_typing.TensorType]
    ]:

        hxs, cxs = [torch.unsqueeze(s, 0) for s in state]
        x, (hxs, cxs) = self.lstm(inputs, (hxs, cxs))
        x = self.model_body(x)
        logits = self.pi(x)
        self._value_out = self.vf(x)

        return logits, [torch.squeeze(s, 0) for s in [hxs, cxs]]

    def get_initial_state(self) -> list[rllib_typing.TensorType]:
        return [torch.zeros((self.lstm_cell_size,)) for _ in range(2)]

    def value_function(self):
        assert self._value_out is not None, "must call forward() first"
        return self._value_out.reshape(-1)
