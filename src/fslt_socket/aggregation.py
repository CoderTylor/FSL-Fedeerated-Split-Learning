from __future__ import annotations

import argparse
import copy
import socket
import time
from pathlib import Path
from typing import Iterable

import torch

from .config import FSLTConfig
from .metrics import append_jsonl
from .protocol import decode_message, recv_frame, send_message


def fed_avg(client_states: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    states = list(client_states)
    if not states:
        raise ValueError("fed_avg requires at least one client state")

    averaged = copy.deepcopy(states[0])
    for key in averaged.keys():
        averaged[key] = averaged[key].clone()
        for state in states[1:]:
            averaged[key] += state[key]
        averaged[key] = torch.div(averaged[key], len(states))
    return averaged


def run_aggregation_server(config: FSLTConfig) -> None:
    print(
        "[AGGREGATOR] Listening for model updates on "
        f"{config.host}:{config.aggregation_upload_port}"
    )
    round_id = 0
    while True:
        started = time.perf_counter()
        received_models = []
        received_clients = []
        received_bytes = 0

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as upload_socket:
            upload_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            upload_socket.bind((config.host, config.aggregation_upload_port))
            upload_socket.listen()

            while len(received_models) < config.clients_per_round:
                conn, addr = upload_socket.accept()
                with conn:
                    data, bytes_read = recv_frame(conn)
                    received_bytes += bytes_read
                    message = decode_message(data, expected_type="model_update")
                    payload = message["payload"]
                    received_models.append(payload["state_dict"])
                    received_clients.append(payload["client_id"])
                    print(
                        "[AGGREGATOR] Received model from "
                        f"client {payload['client_id']} at {addr}"
                    )

        global_state = fed_avg(received_models)
        aggregation_elapsed = time.perf_counter() - started
        print(
            "[AGGREGATOR] Round "
            f"{round_id} FedAvg complete from clients {received_clients}"
        )

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as download_socket:
            download_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            download_socket.bind((config.host, config.aggregation_download_port))
            download_socket.listen(1)
            conn, addr = download_socket.accept()
            with conn:
                bytes_sent = send_message(
                    conn,
                    "global_model",
                    {"state_dict": global_state, "client_ids": received_clients},
                    round_id=round_id,
                )
                print(f"[AGGREGATOR] Sent global model to {addr}")

        append_jsonl(
            config.metrics_path,
            {
                "role": "aggregator",
                "round_id": round_id,
                "client_ids": received_clients,
                "received_bytes": received_bytes,
                "sent_bytes": bytes_sent,
                "aggregation_seconds": aggregation_elapsed,
            },
        )
        round_id += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FSL federated aggregation server.")
    FSLTConfig.add_cli_args(parser)
    args = parser.parse_args()
    config = FSLTConfig.from_args(args)
    config.validate()
    run_aggregation_server(config)


if __name__ == "__main__":
    main()
