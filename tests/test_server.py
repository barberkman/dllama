import socket

import pytest

from dllama import config
from dllama.server import serve_in_thread


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cfg(tmp_path, control_port: int) -> dict:
    cfg, _ = config.load_server(tmp_path / "config.toml")
    cfg["control"]["bind"] = f"127.0.0.1:{control_port}"
    cfg["proxy"]["bind"] = f"127.0.0.1:{_free_port()}"
    cfg["ollama"]["autostart"] = False
    return cfg


def test_serve_in_thread_binds_and_stops(tmp_path):
    port = _free_port()
    stop = serve_in_thread(_cfg(tmp_path, port), tmp_path / "config.toml")
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    finally:
        stop()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


def test_serve_in_thread_reports_bind_errors(tmp_path):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        with pytest.raises(OSError):
            serve_in_thread(_cfg(tmp_path, busy.getsockname()[1]), tmp_path / "config.toml")
