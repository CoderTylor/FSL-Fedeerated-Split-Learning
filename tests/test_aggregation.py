import torch

from fslt_socket.aggregation import fed_avg


def test_fed_avg_averages_client_state_dicts():
    states = [
        {"weight": torch.tensor([1.0, 3.0]), "bias": torch.tensor([2.0])},
        {"weight": torch.tensor([3.0, 5.0]), "bias": torch.tensor([4.0])},
        {"weight": torch.tensor([5.0, 7.0]), "bias": torch.tensor([6.0])},
    ]

    averaged = fed_avg(states)

    assert torch.equal(averaged["weight"], torch.tensor([3.0, 5.0]))
    assert torch.equal(averaged["bias"], torch.tensor([4.0]))


def test_fed_avg_rejects_empty_client_list():
    try:
        fed_avg([])
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("fed_avg should reject an empty client list")
