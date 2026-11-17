"""Offline regressions for classifier privacy and per-session spend."""
import importlib.util
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import jev_client as jc


ROOT = Path(__file__).resolve().parents[1]


def load_router():
    spec = importlib.util.spec_from_file_location("router_classifier_controls", ROOT / "__init__.py")
    router = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(router)
    return router


@pytest.fixture
def setup_router(monkeypatch, tmp_path):
    router = load_router()
    config_path = tmp_path / "config.json"
    cfg = {"mode": "live", "free_model": "test-free-rung", "escalate_enabled": False,
           "classifier_spend_cap_usd": 0.000042}
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(router, "CONFIG_PATH", str(config_path))
    monkeypatch.setattr(router, "CLASSIFIER_BUDGET_PATH", str(tmp_path / "classifier.sqlite3"),
                        raising=False)
    monkeypatch.setattr(router, "_vendored_client", lambda: jc)
    monkeypatch.setenv("JEVI_API_KEY", "test-only-key")
    rows, requests = [], []
    monkeypatch.setattr(router, "_log", rows.append)

    def http(url, key, payload, timeout):
        requests.append((url, payload))
        return 200, {"model": "jev-1.13.0",
                     "usage": {"input_tokens": 1000, "output_tokens": 10},
                     "answers": {"routing_tier": {"type": "choice", "choice": "free",
                                 "probabilities": {"free": 1.0, "paid": 0.0},
                                 "confidence": 0.99}}}

    monkeypatch.setattr(jc, "_http_json", http)
    return router, cfg, config_path, rows, requests


def turn(router, session, turn_id, text="PRIVATE-CANARY-TEXT"):
    request = {"model": "default-model", "messages": [{"role": "user", "content": text}]}
    result = router.on_llm_request(provider="nous", model="default-model", request=request,
                                   session_id=session, turn_id=turn_id, api_call_count=1)
    return result, request


@pytest.mark.parametrize("source", ["code", "manifest", "example"])
def test_excerpt_defaults_off_in_all_shipped_configs(monkeypatch, tmp_path, source):
    router = load_router()
    monkeypatch.setattr(router, "CONFIG_PATH", str(tmp_path / "missing.json"))
    if source == "code":
        value = router._load_config()["send_excerpt"]
        features = router._features({"messages": [{"content": "private"}]})
        assert "task_excerpt" not in features
    elif source == "example":
        value = json.loads((ROOT / "examples/lane2_config.example.json").read_text())["send_excerpt"]
    else:
        manifest = (ROOT / "plugin.yaml").read_text()
        block = re.search(r"^  send_excerpt:\n((?:    .*\n)+)", manifest, re.M)
        assert block
        value = re.search(r"^    default:\s*(true|false)\s*$", block.group(1), re.M).group(1) == "true"
    assert value is False


def test_excerpt_opt_in_controls_wire_payload_log_and_cap(setup_router):
    router, cfg, config_path, rows, requests = setup_router
    cfg.update(mode="shadow", classifier_spend_cap_usd=1, excerpt_chars=6)
    config_path.write_text(json.dumps(cfg))
    assert turn(router, "privacy", "one")[0] is None
    assert len(requests) == 1 and requests[0][0] == "https://api.typesafe.ai/v1/systemone"
    assert "PRIVATE-CANARY-TEXT" not in json.dumps(requests[0][1])
    assert requests[0][1]["state"]["task"] == ""
    assert rows[-1]["excerpt_sent"] is False
    assert "PRIVATE-CANARY-TEXT" not in json.dumps(rows[-1])

    cfg["send_excerpt"] = True
    config_path.write_text(json.dumps(cfg))
    assert turn(router, "privacy", "two")[0] is None
    assert requests[-1][1]["state"]["task"] == "PRIVAT"
    assert rows[-1]["excerpt_sent"] is True
    assert rows[-1]["features"]["task_excerpt"] == "PRIVAT"

    cfg["send_excerpt"] = "false"  # not an explicit boolean opt-in
    config_path.write_text(json.dumps(cfg))
    assert turn(router, "privacy", "three")[0] is None
    assert requests[-1][1]["state"]["task"] == ""
    assert rows[-1]["excerpt_sent"] is False


    # Negative caps cannot use Python's negative-slice semantics to leak text.
    cfg.update(send_excerpt=True, excerpt_chars=-1)
    config_path.write_text(json.dumps(cfg))
    assert turn(router, "privacy", "four")[0] is None
    assert requests[-1][1]["state"]["task"] == ""
    assert rows[-1]["excerpt_sent"] is False


def test_session_cap_records_actual_classifier_cost_and_survives_restart(setup_router, monkeypatch):
    router, cfg, config_path, rows, requests = setup_router
    costs = iter([0.000021, 0.000021])
    # Simulate two successful calls that sum exactly to the configured ceiling.
    original_evaluate = jc.evaluate

    def evaluate(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        result["cost_usd"] = next(costs)
        return result

    monkeypatch.setattr(jc, "evaluate", evaluate)
    result, original = turn(router, "budget-a", "one")
    assert result["request"]["model"] == "test-free-rung"
    assert rows[-1]["classifier_cost_usd"] == 0.000021
    assert rows[-1]["classifier_usage"] == {"input_tokens": 1000, "output_tokens": 10}
    assert turn(router, "budget-a", "two")[0]["request"]["model"] == "test-free-rung"
    result, original = turn(router, "budget-a", "three")
    assert result is None and original["model"] == "default-model"
    assert len(requests) == 2
    assert rows[-1]["skip_reason"] == "classifier_spend_cap_reached"
    assert rows[-1]["excerpt_sent"] is False

    # A new plugin instance (e.g. gateway restart) must not reset this session.
    restarted = load_router()
    monkeypatch.setattr(restarted, "CONFIG_PATH", str(config_path))
    monkeypatch.setattr(restarted, "CLASSIFIER_BUDGET_PATH", router.CLASSIFIER_BUDGET_PATH,
                        raising=False)
    monkeypatch.setattr(restarted, "_vendored_client", lambda: jc)
    monkeypatch.setattr(restarted, "_log", rows.append)
    assert turn(restarted, "budget-a", "four")[0] is None
    assert len(requests) == 2
    monkeypatch.setattr(jc, "evaluate", original_evaluate)
    assert turn(restarted, "budget-b", "one")[0]["request"]["model"] == "test-free-rung"
    assert len(requests) == 3  # a different session has its own budget


def test_same_session_parallel_calls_do_not_bypass_ceiling(setup_router, monkeypatch):
    router, cfg, config_path, rows, requests = setup_router
    entered, release = threading.Event(), threading.Event()
    original_http = jc._http_json

    def slow_first(*args):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original_http(*args)

    monkeypatch.setattr(jc, "_http_json", slow_first)
    other = load_router()
    monkeypatch.setattr(other, "CONFIG_PATH", router.CONFIG_PATH)
    monkeypatch.setattr(other, "CLASSIFIER_BUDGET_PATH", router.CLASSIFIER_BUDGET_PATH,
                        raising=False)
    monkeypatch.setattr(other, "_vendored_client", lambda: jc)
    monkeypatch.setattr(other, "_log", rows.append)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(turn, router, "parallel", "one")
        assert entered.wait(5)
        try:
            # A separate module instance has independent Python locks and cache;
            # only the SQLite transaction coordinates the two callers.
            second, request = turn(other, "parallel", "two")
            assert second is None and request["model"] == "default-model"
            assert requests == []  # the first is blocked inside the fake transport
        finally:
            release.set()
        assert first.result()[0]["request"]["model"] == "test-free-rung"
    assert len(requests) == 1


@pytest.mark.parametrize("cap", [0, -1, "NaN", "invalid"])
def test_invalid_or_zero_cap_never_calls_vendor(setup_router, cap):
    router, cfg, config_path, rows, requests = setup_router
    cfg["classifier_spend_cap_usd"] = cap
    config_path.write_text(json.dumps(cfg))
    result, request = turn(router, "bad-cap", "one")
    assert result is None and request["model"] == "default-model"
    assert requests == []


def test_budget_unavailable_preserves_original_request(setup_router, monkeypatch):
    router, cfg, config_path, rows, requests = setup_router
    # A present but unreadable budget must never silently reset session spend.
    cfg.update(send_excerpt=False, classifier_spend_cap_usd=1)
    config_path.write_text(json.dumps(cfg))
    router = load_router()
    monkeypatch.setattr(router, "CONFIG_PATH", str(config_path))
    monkeypatch.setattr(router, "_vendored_client", lambda: jc)
    monkeypatch.setattr(router, "_log", rows.append)
    directory = config_path.parent / "directory"
    directory.mkdir()
    monkeypatch.setattr(router, "CLASSIFIER_BUDGET_PATH", str(directory), raising=False)
    result, request = turn(router, "ledger-error", "one")
    assert result is None and request["model"] == "default-model"
    assert requests == []
    assert rows[-1]["skip_reason"] == "classifier_budget_unavailable"
    assert rows[-1]["excerpt_sent"] is False


def test_unknown_billed_cost_blocks_later_calls_and_reports_uncertain_egress(setup_router, monkeypatch):
    router, cfg, config_path, rows, requests = setup_router
    cfg.update(send_excerpt=True, classifier_spend_cap_usd=1)
    config_path.write_text(json.dumps(cfg))

    def timeout(url, key, payload, timeout):
        requests.append((url, payload))
        raise TimeoutError("test timeout after dispatch")

    monkeypatch.setattr(jc, "_http_json", timeout)
    assert turn(router, "unknown", "one")[0] is None
    assert rows[-1]["excerpt_sent"] is None  # the client cannot prove whether delivery happened
    assert rows[-1]["classifier_cost_usd"] is None
    # A timeout after dispatch may have been billed.  Keep later calls blocked,
    # even when a new module instance opens the same ledger.
    restarted = load_router()
    monkeypatch.setattr(restarted, "CONFIG_PATH", str(config_path))
    monkeypatch.setattr(restarted, "CLASSIFIER_BUDGET_PATH", router.CLASSIFIER_BUDGET_PATH,
                        raising=False)
    monkeypatch.setattr(restarted, "_vendored_client", lambda: jc)
    monkeypatch.setattr(restarted, "_log", rows.append)
    assert turn(restarted, "unknown", "three")[0] is None
    assert len(requests) == 1
