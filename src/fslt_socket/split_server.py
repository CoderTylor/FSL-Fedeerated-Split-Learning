from __future__ import annotations

import argparse
import socket
import time

import torch
import torch.nn as nn

from .config import FSLTConfig
from .metrics import append_jsonl
from .model import ResNet18ServerSide
from .protocol import decode_message, recv_frame, send_message


def run_split_server(config: FSLTConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_model = ResNet18ServerSide(num_classes=config.num_classes).to(device)
    server_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(server_model.parameters(), lr=config.learning_rate)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((config.host, config.activation_port))
        server_socket.listen()
        print(f"[SPLIT] Listening on {config.host}:{config.activation_port}")

        while True:
            conn, addr = server_socket.accept()
            started = time.perf_counter()
            with conn:
                raw, bytes_read = recv_frame(conn)
                message = decode_message(raw, expected_type="activation")
                payload = message["payload"]
                activation = payload["activation"].to(device)
                labels = payload["labels"].to(device)
                client_id = payload["client_id"]
                round_id = message["round_id"]

                activation.requires_grad_()
                output = server_model(activation)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                gradient = activation.grad.detach().cpu()
                bytes_sent = send_message(
                    conn,
                    "activation_gradient",
                    {"gradient": gradient, "loss": float(loss.item())},
                    round_id=round_id,
                )

            elapsed = time.perf_counter() - started
            append_jsonl(
                config.metrics_path,
                {
                    "role": "split_server",
                    "round_id": round_id,
                    "client_id": client_id,
                    "peer": str(addr),
                    "loss": float(loss.item()),
                    "received_bytes": bytes_read,
                    "sent_bytes": bytes_sent,
                    "service_seconds": elapsed,
                },
            )
            print(
                "[SPLIT] Round "
                f"{round_id} client {client_id} loss={loss.item():.4f} "
                f"elapsed={elapsed:.3f}s"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FSL split-learning server.")
    FSLTConfig.add_cli_args(parser)
    args = parser.parse_args()
    config = FSLTConfig.from_args(args)
    config.validate()
    run_split_server(config)


if __name__ == "__main__":
    main()
