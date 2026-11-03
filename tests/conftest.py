"""Make the repo's plugin module and vendored scripts importable to the tests."""
import os
import socket
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(autouse=True)
def offline_isolation(monkeypatch, request, tmp_path):
    """No real credentials, state, DNS or sockets, even when run on a live host."""
    for key in ("NOUS_API_KEY", "JEVI_API_KEY", "JEVI_API_KEY_UNUSED_BY_TEST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("unexpected network access in offline suite")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket.socket, "sendto", blocked)

    # Several test modules each load their own plugin module by path. Reset
    # each one's caches and redirect import-time paths on EVERY test.
    pl = getattr(request.module, "pl", None)
    if pl is not None and hasattr(pl, "_CREDIT"):
        monkeypatch.setattr(pl, "_CREDIT", {"ok": None, "checked_at": 0.0, "identity": None})
        with pl._DECISIONS_LOCK:
            pl._DECISIONS.clear()
        monkeypatch.setattr(pl, "_HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(pl, "CONFIG_PATH", str(tmp_path / "jev" / "lane2_config.json"))
        monkeypatch.setattr(pl, "LOG_PATH", str(tmp_path / "jev" / "lane2-live.jsonl"))
        monkeypatch.setattr(pl, "_PLUGIN_CTX", None)
    yield
    assert not attempts, "the offline suite attempted a real DNS/socket operation"
