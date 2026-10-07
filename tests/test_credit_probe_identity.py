"""Credential/endpoint cache controls; every HTTP operation is a local test double."""
import io
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    monkeypatch.setenv("NOUS_BASE_URL", "https://original.invalid/v1")
    path = Path(os.environ.get("ROUTER_SOURCE", Path(__file__).parents[1] / "__init__.py"))
    spec = importlib.util.spec_from_file_location("credit_identity_router", path)
    router = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(router)

    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected network access")

    monkeypatch.setattr(router.urllib.request, "urlopen", no_network)
    return SimpleNamespace(router=router)


def credit_probe(h, monkeypatch):
    calls, clock, alive = [], [1000.], [True]
    monkeypatch.setenv('NOUS_API_KEY', 'synthetic-account-A')
    monkeypatch.setattr(h.router.time, 'time', lambda: clock[0])
    monkeypatch.setattr(h.router.time, 'monotonic', lambda: clock[0])

    def urlopen(request, timeout):
        calls.append((request.full_url, request.get_header('Authorization')))
        if not alive[0]:
            raise OSError('synthetic unavailable account')
        return io.BytesIO(b'{"choices":[{"message":{"content":"pong"}}]}')
    monkeypatch.setattr(h.router.urllib.request, 'urlopen', urlopen)
    return calls, clock, alive


@pytest.mark.parametrize('change', ['rotate_key', 'remove_key', 'change_endpoint'])
def test_positive_cache_cannot_authorize_another_identity(harness, monkeypatch, change):
    h = harness
    calls, clock, alive = credit_probe(h, monkeypatch)
    assert h.router._paid_lane_alive() is True
    alive[0] = False
    if change == 'rotate_key':
        monkeypatch.setenv('NOUS_API_KEY', 'synthetic-account-B')
    elif change == 'remove_key':
        monkeypatch.delenv('NOUS_API_KEY')
    else:
        monkeypatch.setenv('NOUS_BASE_URL', 'https://replacement.invalid/v1')
    assert h.router._paid_lane_alive() is False
    assert len(calls) == (1 if change == 'remove_key' else 2)


def test_negative_cache_does_not_block_new_identity(harness, monkeypatch):
    h = harness
    calls, clock, alive = credit_probe(h, monkeypatch)
    alive[0] = False
    assert h.router._paid_lane_alive() is False
    monkeypatch.setenv('NOUS_API_KEY', 'synthetic-account-B')
    alive[0] = True
    assert h.router._paid_lane_alive() is True
    assert len(calls) == 2


def test_unchanged_identity_reuses_verdict(harness, monkeypatch):
    h = harness
    calls, clock, alive = credit_probe(h, monkeypatch)
    assert h.router._paid_lane_alive() is True
    assert h.router._paid_lane_alive() is True
    assert len(calls) == 1


def test_unchanged_negative_identity_reuses_verdict(harness, monkeypatch):
    h = harness
    calls, clock, alive = credit_probe(h, monkeypatch)
    alive[0] = False
    assert h.router._paid_lane_alive() is False
    assert h.router._paid_lane_alive() is False
    assert len(calls) == 1


def test_expired_verdict_reprobes(harness, monkeypatch):
    h = harness
    calls, clock, alive = credit_probe(h, monkeypatch)
    assert h.router._paid_lane_alive() is True
    clock[0] += h.router.CREDIT_TTL_S + 1
    alive[0] = False
    assert h.router._paid_lane_alive() is False
    assert len(calls) == 2


def test_missing_credential_never_probes(harness):
    assert harness.router._paid_lane_alive() is False


def test_current_account_rejection_blocks_escalation(harness, monkeypatch):
    h = harness
    calls, clock, alive = credit_probe(h, monkeypatch)
    cfg = {"enabled": True, "mode": "live", "confidence_gate": .8,
           "escalate_enabled": True, "escalate_providers": ["nous"],
           "escalate_models": ["source-mid"], "premium_model": "target-premium",
           "credit_probe": True}
    monkeypatch.setattr(h.router, "_load_config", lambda: cfg)
    monkeypatch.setattr(h.router, "_classify", lambda *args: ("paid", .99))
    monkeypatch.setattr(h.router, "_log", lambda record: None)
    alive[0] = False
    request = {"model": "source-mid", "messages": [{"role": "user", "content": "analyze"}]}
    result = h.router.on_llm_request(request=request, provider="nous", model="source-mid",
                                     session_id="s", turn_id="t", api_call_count=1)
    assert result is None
    assert request["model"] == "source-mid"
    assert len(calls) == 1
