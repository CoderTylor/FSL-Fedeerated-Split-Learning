import pickle

from fslt_socket.protocol import decode_message, encode_message


def test_encode_message_wraps_payload_with_message_type_and_round():
    encoded = encode_message("activation", {"client_id": 2}, round_id=4)

    message = pickle.loads(encoded)

    assert message["type"] == "activation"
    assert message["round_id"] == 4
    assert message["payload"] == {"client_id": 2}


def test_decode_message_rejects_wrong_type():
    encoded = encode_message("model_update", {"client_id": 0}, round_id=1)

    try:
        decode_message(encoded, expected_type="activation")
    except ValueError as exc:
        assert "model_update" in str(exc)
    else:
        raise AssertionError("decode_message should reject an unexpected message type")
