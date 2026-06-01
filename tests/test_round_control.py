from fslt_socket.round_control import apply_round_delay, select_clients


def test_select_clients_prefers_lowest_observed_latency():
    latency_by_client = {0: 0.42, 1: 0.10, 2: 0.31, 3: 0.22}

    selected = select_clients(
        available_client_ids=[0, 1, 2, 3],
        clients_per_round=2,
        strategy="latency",
        latency_by_client=latency_by_client,
    )

    assert selected == [1, 3]


def test_select_clients_random_is_deterministic_with_seed():
    selected = select_clients(
        available_client_ids=[0, 1, 2, 3],
        clients_per_round=2,
        strategy="random",
        random_seed=7,
    )

    assert selected == [3, 1]


def test_apply_round_delay_sleeps_between_rounds_only():
    calls = []

    apply_round_delay(current_round=0, total_rounds=3, delay_seconds=1.5, sleep_fn=calls.append)
    apply_round_delay(current_round=2, total_rounds=3, delay_seconds=1.5, sleep_fn=calls.append)

    assert calls == [1.5]
