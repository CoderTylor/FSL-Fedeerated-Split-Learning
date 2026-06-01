from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable


def select_clients(
    available_client_ids: Iterable[int],
    clients_per_round: int,
    strategy: str,
    latency_by_client: dict[int, float] | None = None,
    random_seed: int | None = None,
) -> list[int]:
    client_ids = list(available_client_ids)
    if clients_per_round > len(client_ids):
        raise ValueError("clients_per_round cannot exceed number of available clients")

    if strategy == "all":
        return client_ids[:clients_per_round]

    if strategy == "random":
        shuffled = client_ids[:]
        random.Random(random_seed).shuffle(shuffled)
        return shuffled[:clients_per_round]

    if strategy == "latency":
        latency_by_client = latency_by_client or {}
        return sorted(
            client_ids,
            key=lambda client_id: latency_by_client.get(client_id, float("inf")),
        )[:clients_per_round]

    raise ValueError(f"Unknown client selection strategy: {strategy}")


def apply_round_delay(
    current_round: int,
    total_rounds: int,
    delay_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    if delay_seconds <= 0:
        return
    if current_round >= total_rounds - 1:
        return
    sleep_fn(delay_seconds)
