from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass

import torch

from .config import FSLTConfig
from .data_loader import build_client_loaders
from .metrics import append_jsonl
from .model import ResNet18ClientSide
from .protocol import decode_message, recv_frame, send_message
from .round_control import apply_round_delay, select_clients


@dataclass
class ClientRoundStats:
    activation_seconds: float = 0.0
    upload_seconds: float = 0.0
    bytes_sent: int = 0
    bytes_received: int = 0
    batches: int = 0
    server_loss_total: float = 0.0

    @property
    def mean_activation_seconds(self) -> float:
        if self.batches == 0:
            return float("inf")
        return self.activation_seconds / self.batches

    @property
    def mean_server_loss(self) -> float:
        if self.batches == 0:
            return 0.0
        return self.server_loss_total / self.batches


def train_selected_clients(config: FSLTConfig) -> None:
    config.validate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_models = []
    optimizers = []
    train_loaders = []
    for client_id in range(config.num_clients):
        model = ResNet18ClientSide().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        train_loader, _ = build_client_loaders(
            client_id=client_id,
            num_clients=config.num_clients,
            batch_size=config.batch_size,
            metadata_path=config.metadata_path,
            image_glob=config.image_glob,
        )
        client_models.append(model)
        optimizers.append(optimizer)
        train_loaders.append(train_loader)

    latency_by_client: dict[int, float] = {}
    all_client_ids = list(range(config.num_clients))

    for round_id in range(config.epochs):
        selected_clients = select_clients(
            all_client_ids,
            clients_per_round=config.clients_per_round,
            strategy=config.client_selection,
            latency_by_client=latency_by_client,
            random_seed=config.random_seed + round_id,
        )
        print(f"\n[CLIENT] Round {round_id + 1}/{config.epochs} selected {selected_clients}")

        round_stats: dict[int, ClientRoundStats] = {}
        for client_id in selected_clients:
            stats = _train_one_client_round(
                config=config,
                round_id=round_id,
                client_id=client_id,
                model=client_models[client_id],
                optimizer=optimizers[client_id],
                train_loader=train_loaders[client_id],
                device=device,
            )
            round_stats[client_id] = stats
            latency_by_client[client_id] = stats.mean_activation_seconds
            print(
                f"[CLIENT-{client_id}] batches={stats.batches} "
                f"mean_split_latency={stats.mean_activation_seconds:.3f}s "
                f"mean_server_loss={stats.mean_server_loss:.4f}"
            )

        print("[CLIENT] Uploading selected client-side models for FedAvg...")
        for client_id in selected_clients:
            upload_seconds, bytes_sent = _send_model_update(
                config=config,
                round_id=round_id,
                client_id=client_id,
                model=client_models[client_id],
            )
            round_stats[client_id].upload_seconds = upload_seconds
            round_stats[client_id].bytes_sent += bytes_sent
            print(f"[CLIENT-{client_id}] model uploaded in {upload_seconds:.3f}s")

        wait_started = time.perf_counter()
        global_state = _download_global_model(config, round_id)
        aggregation_wait_seconds = time.perf_counter() - wait_started
        for model in client_models:
            model.load_state_dict(global_state)

        for client_id, stats in round_stats.items():
            append_jsonl(
                config.metrics_path,
                {
                    "role": "client",
                    "round_id": round_id,
                    "client_id": client_id,
                    "selected_clients": selected_clients,
                    "batches": stats.batches,
                    "mean_split_latency_seconds": stats.mean_activation_seconds,
                    "mean_server_loss": stats.mean_server_loss,
                    "model_upload_seconds": stats.upload_seconds,
                    "aggregation_wait_seconds": aggregation_wait_seconds,
                    "round_delay_seconds": config.round_delay_seconds,
                    "bytes_sent": stats.bytes_sent,
                    "bytes_received": stats.bytes_received,
                },
            )

        print(
            "[CLIENT] Aggregated model received; "
            f"waiting {config.round_delay_seconds}s before next round."
        )
        apply_round_delay(
            current_round=round_id,
            total_rounds=config.epochs,
            delay_seconds=config.round_delay_seconds,
        )


def _train_one_client_round(
    config: FSLTConfig,
    round_id: int,
    client_id: int,
    model: ResNet18ClientSide,
    optimizer: torch.optim.Optimizer,
    train_loader,
    device: torch.device,
) -> ClientRoundStats:
    stats = ClientRoundStats()
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        activation = model(images)
        activation_for_server = activation.clone().detach().requires_grad_(True)
        payload = {
            "client_id": client_id,
            "activation": activation_for_server.cpu(),
            "labels": labels.cpu(),
        }

        started = time.perf_counter()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((config.host, config.activation_port))
            stats.bytes_sent += send_message(sock, "activation", payload, round_id=round_id)
            raw, bytes_read = recv_frame(sock)
            stats.bytes_received += bytes_read
        elapsed = time.perf_counter() - started

        message = decode_message(raw, expected_type="activation_gradient")
        gradient = message["payload"]["gradient"].to(device)
        activation.backward(gradient)
        optimizer.step()

        stats.activation_seconds += elapsed
        stats.server_loss_total += float(message["payload"]["loss"])
        stats.batches += 1

    return stats


def _send_model_update(
    config: FSLTConfig,
    round_id: int,
    client_id: int,
    model: ResNet18ClientSide,
) -> tuple[float, int]:
    started = time.perf_counter()
    payload = {
        "client_id": client_id,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
    }
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((config.host, config.aggregation_upload_port))
        bytes_sent = send_message(sock, "model_update", payload, round_id=round_id)
    return time.perf_counter() - started, bytes_sent


def _download_global_model(config: FSLTConfig, round_id: int) -> dict[str, torch.Tensor]:
    last_error: OSError | None = None
    for _ in range(config.aggregation_download_retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((config.host, config.aggregation_download_port))
                raw, _ = recv_frame(sock)
            message = decode_message(raw, expected_type="global_model")
            if message["round_id"] != round_id:
                raise ValueError(
                    f"Expected global model for round {round_id}, got {message['round_id']}"
                )
            return message["payload"]["state_dict"]
        except OSError as exc:
            last_error = exc
            time.sleep(config.aggregation_retry_delay_seconds)

    raise TimeoutError("Timed out waiting for aggregated global model") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UE clients for FSL socket training.")
    FSLTConfig.add_cli_args(parser)
    args = parser.parse_args()
    train_selected_clients(FSLTConfig.from_args(args))


if __name__ == "__main__":
    main()
