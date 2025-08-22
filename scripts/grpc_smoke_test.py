import time

import grpc
from google.protobuf import json_format

from footsiesgym.footsies.game import constants
from footsiesgym.footsies.game.proto import footsies_service_pb2 as footsies_pb2
from footsiesgym.footsies.game.proto import footsies_service_pb2_grpc as footsies_pb2_grpc


def run():
    # Connect to the gRPC server
    channel = grpc.insecure_channel("localhost:50051")
    stub = footsies_pb2_grpc.FootsiesGameServiceStub(channel)

    # Example call to StartGame
    stub.StartGame(footsies_pb2.Empty())
    print("StartGame called successfully")

    ready = stub.IsReady(footsies_pb2.Empty()).value
    while not ready:
        print("Game not ready...")
        ready = stub.IsReady(footsies_pb2.Empty()).value
        time.sleep(0.5)

    print("Game ready!")

    # Example call to GetState
    game_state = stub.GetState(footsies_pb2.Empty())
    print("GetState response received")
    print(f"GameState: {game_state}")

    # Example call to StepNFrames
    step_input = footsies_pb2.StepInput(
        p1_action=constants.ActionBits.RIGHT,
        p2_action=constants.ActionBits.LEFT,
        nFrames=4,
    )
    game_state = stub.StepNFrames(step_input)
    print("StepNFrames response received")
    print(
        f"GameState: {json_format.MessageToDict(game_state, always_print_fields_with_no_presence=True, preserving_proto_field_name=True)}"
    )

    # Example call to ResetGame
    response = stub.ResetGame(footsies_pb2.Empty())
    print("ResetGame response received")

    # Example call to GetState
    game_state = stub.GetState(footsies_pb2.Empty())
    print("GetState response received")
    print(f"GameState: {game_state}")


if __name__ == "__main__":
    run()
