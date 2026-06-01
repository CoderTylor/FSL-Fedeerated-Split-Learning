from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FSLTConfig:
    host: str = "127.0.0.1"
    activation_port: int = 12349
    aggregation_upload_port: int = 12350
    aggregation_download_port: int = 12351
    num_clients: int = 3
    clients_per_round: int = 3
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_classes: int = 7
    round_delay_seconds: float = 0.0
    client_selection: str = "all"
    random_seed: int = 42
    aggregation_retry_delay_seconds: float = 1.0
    aggregation_download_retries: int = 60
    metrics_path: Path = Path("outputs/metrics.jsonl")
    metadata_path: Path = Path("data/HAM10000_metadata.csv")
    image_glob: str = "data/*/*.jpg"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--host", default=FSLTConfig.host)
        parser.add_argument("--activation-port", type=int, default=FSLTConfig.activation_port)
        parser.add_argument(
            "--aggregation-upload-port",
            type=int,
            default=FSLTConfig.aggregation_upload_port,
        )
        parser.add_argument(
            "--aggregation-download-port",
            type=int,
            default=FSLTConfig.aggregation_download_port,
        )
        parser.add_argument("--num-clients", type=int, default=FSLTConfig.num_clients)
        parser.add_argument(
            "--clients-per-round",
            type=int,
            default=FSLTConfig.clients_per_round,
            help="Number of selected UE clients that upload model updates each round.",
        )
        parser.add_argument("--epochs", type=int, default=FSLTConfig.epochs)
        parser.add_argument("--batch-size", type=int, default=FSLTConfig.batch_size)
        parser.add_argument("--learning-rate", type=float, default=FSLTConfig.learning_rate)
        parser.add_argument("--num-classes", type=int, default=FSLTConfig.num_classes)
        parser.add_argument(
            "--round-delay-seconds",
            type=float,
            default=FSLTConfig.round_delay_seconds,
            help="Delay after aggregation before the next round starts.",
        )
        parser.add_argument(
            "--client-selection",
            choices=["all", "random", "latency"],
            default=FSLTConfig.client_selection,
        )
        parser.add_argument("--random-seed", type=int, default=FSLTConfig.random_seed)
        parser.add_argument(
            "--aggregation-retry-delay-seconds",
            type=float,
            default=FSLTConfig.aggregation_retry_delay_seconds,
        )
        parser.add_argument(
            "--aggregation-download-retries",
            type=int,
            default=FSLTConfig.aggregation_download_retries,
        )
        parser.add_argument("--metrics-path", type=Path, default=FSLTConfig.metrics_path)
        parser.add_argument("--metadata-path", type=Path, default=FSLTConfig.metadata_path)
        parser.add_argument("--image-glob", default=FSLTConfig.image_glob)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FSLTConfig":
        return cls(
            host=args.host,
            activation_port=args.activation_port,
            aggregation_upload_port=args.aggregation_upload_port,
            aggregation_download_port=args.aggregation_download_port,
            num_clients=args.num_clients,
            clients_per_round=args.clients_per_round,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_classes=args.num_classes,
            round_delay_seconds=args.round_delay_seconds,
            client_selection=args.client_selection,
            random_seed=args.random_seed,
            aggregation_retry_delay_seconds=args.aggregation_retry_delay_seconds,
            aggregation_download_retries=args.aggregation_download_retries,
            metrics_path=args.metrics_path,
            metadata_path=args.metadata_path,
            image_glob=args.image_glob,
        )

    def validate(self) -> None:
        if self.clients_per_round < 1:
            raise ValueError("clients_per_round must be at least 1")
        if self.clients_per_round > self.num_clients:
            raise ValueError("clients_per_round cannot exceed num_clients")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
