from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

HEADER_SIZE = 8


def encode_message(message_type: str, payload: Any, round_id: int) -> bytes:
    return pickle.dumps(
        {"type": message_type, "round_id": round_id, "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def decode_message(data: bytes, expected_type: str | None = None) -> dict[str, Any]:
    message = pickle.loads(data)
    actual_type = message.get("type")
    if expected_type is not None and actual_type != expected_type:
        raise ValueError(f"Expected message type {expected_type!r}, got {actual_type!r}")
    return message


def send_frame(sock: socket.socket, data: bytes) -> int:
    header = struct.pack("!Q", len(data))
    sock.sendall(header + data)
    return len(header) + len(data)


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    chunks = []
    remaining = num_bytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed before full message was received")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> tuple[bytes, int]:
    header = recv_exact(sock, HEADER_SIZE)
    payload_size = struct.unpack("!Q", header)[0]
    payload = recv_exact(sock, payload_size)
    return payload, HEADER_SIZE + payload_size


def send_message(
    sock: socket.socket,
    message_type: str,
    payload: Any,
    round_id: int,
) -> int:
    return send_frame(sock, encode_message(message_type, payload, round_id))
