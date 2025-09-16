import os

import numpy as np
import onnxruntime
import ray
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import add_rllib_example_script_args, check
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

from components import module_repository

torch, _ = try_import_torch()

parser = add_rllib_example_script_args()
parser.set_defaults(num_env_runners=1)


# Add a policy name in the ModuleRepository
POLICY = []


class ONNXCompatibleWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super(ONNXCompatibleWrapper, self).__init__()
        self.original_model = original_model

    def forward(self, a, b0, b1, c):
        # Convert the separate tensor inputs back into the list format
        # expected by the original model's forward method.
        b = [b0, b1]
        ret = self.original_model({"obs": a}, b, c)
        # results, state_out_0, state_out_1
        return ret[0], ret[1][0], ret[1][1]


def run(policy):

    restored_policy_0 = module_repository.ModuleRepository.get(policy)

    # Check if the restored policy is recurrent
    if not restored_policy_0.is_recurrent():
        restored_policy_0.export_model("./", onnx=13)
        return

    # If the policy is recurrent, we need to wrap it in a way that it can be exported to ONNX
    B = 1
    T = 1
    LSTM_CELL = restored_policy_0._state_inputs[0].shape[0]
    INPUT_SHAPE = restored_policy_0._get_default_view_requirements()[
        "obs"
    ].space.shape[0]

    print(INPUT_SHAPE)

    # Input data for a python inference forward call.
    test_data_python = {
        "obs": np.random.uniform(0, 1.0, size=(B * T, INPUT_SHAPE)).astype(
            np.float32
        ),
        "state_ins": [
            np.random.uniform(0, 1.0, size=(B, LSTM_CELL)).astype(np.float32),
            np.random.uniform(0, 1.0, size=(B, LSTM_CELL)).astype(np.float32),
        ],
        "seq_lens": np.array([T] * B, np.float32),
    }
    # Input data for the ONNX session.
    test_data_onnx = {
        "obs": test_data_python["obs"],
        "state_in_0": test_data_python["state_ins"][0],
        "state_in_1": test_data_python["state_ins"][1],
        "seq_lens": test_data_python["seq_lens"],
    }

    # Input data for compiling the ONNX model.
    test_data_onnx_input = convert_to_torch_tensor(test_data_onnx)

    # Let's run inference on the torch model
    result_pytorch, _ = restored_policy_0.model(
        {
            "obs": torch.tensor(test_data_python["obs"]),
        },
        [
            torch.tensor(test_data_python["state_ins"][0]),
            torch.tensor(test_data_python["state_ins"][1]),
        ],
        torch.tensor(test_data_python["seq_lens"]),
    )

    # Evaluate tensor to fetch numpy array
    result_pytorch = result_pytorch.detach().numpy()

    # Wrap the actual ModelV2 with the torch wrapper above to make this all work with
    # LSTMs (extra `state` in- and outputs and `seq_lens` inputs).
    onnx_compatible = ONNXCompatibleWrapper(restored_policy_0.model)

    os.makedirs("onnx_models", exist_ok=True)

    exported_model_file = f"onnx_models/{policy}.onnx"
    input_names = [
        "obs",
        "state_in_0",
        "state_in_1",
        "seq_lens",
    ]

    # This line will export the model to ONNX.
    torch.onnx.export(
        onnx_compatible,
        tuple(test_data_onnx_input[n] for n in input_names),
        exported_model_file,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=[
            "output",
            "state_out_0",
            "state_out_1",
        ],
        dynamic_axes={k: {0: "batch_size"} for k in input_names},
    )
    # Start an inference session for the ONNX model.
    session = onnxruntime.InferenceSession(exported_model_file, None)
    result_onnx = session.run(["output"], test_data_onnx)

    # These results should be equal!
    print("PYTORCH", result_pytorch)
    print("ONNX", result_onnx[0])

    check(result_pytorch, result_onnx[0])
    print("Model outputs are equal. PASSED")


if __name__ == "__main__":
    ray.init(local_mode=True)

    if isinstance(POLICY, list):
        for policy_id in POLICY:
            run(policy_id)
    # else:
    #     run(POLICY)
