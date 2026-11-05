"""Offline regressions for provider scope, cache replay and D7 routing."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "scope_router", Path(__file__).parents[1] / "__init__.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)

SOURCE = "vendor/source"
FREE = "vendor/free:free"
FLASH = "vendor/flash"
MID = "vendor/mid"
PREMIUM = "vendor/premium"


def config(**changes):
    result = {
        "enabled": True, "mode": "live", "confidence_gate": 0.65,
        "flash_gate": 0.35, "escalate_confidence_gate": 0.80,
        "route_providers": ["nous"], "escalate_providers": ["nous"],
        "escalate_enabled": True, "credit_probe": True,
        "free_model": FREE, "flash_model": FLASH, "mid_model": MID,
        "premium_model": PREMIUM,
    }
    result.update(changes)
    return result


def kwargs(turn="scope", call=1):
    return {"provider": "nous", "model": SOURCE, "session_id": "s",
            "turn_id": turn, "api_call_count": call,
            "request": {"model": SOURCE, "messages": [{"role": "user", "content": "hi"}]}}


def install(monkeypatch, cfg, pool, confidence):
    classified, probes, rows = [], [], []
    monkeypatch.setattr(pl, "_load_config", lambda: cfg)
    monkeypatch.setattr(pl, "_classify",
                        lambda *_: classified.append(1) or (pool, confidence))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: probes.append(1) or True)
    monkeypatch.setattr(pl, "_log", rows.append)
    pl._DECISIONS.clear()
    return classified, probes, rows


@pytest.mark.parametrize("confidence", [0.20, 0.35, 0.50, 0.65, 0.79, 0.80, 0.99])
def test_escalate_only_provider_never_uses_downgrade_bands(monkeypatch, confidence):
    cfg = config(route_providers=["other"], escalate_providers=["nous"])
    classified, probes, _ = install(monkeypatch, cfg, "free", confidence)
    assert pl.on_llm_request(**kwargs()) is None
    assert classified == [1] and probes == []


def test_escalate_only_provider_still_escalates(monkeypatch):
    cfg = config(route_providers=["other"], escalate_providers=["nous"])
    classified, probes, _ = install(monkeypatch, cfg, "paid", 0.95)
    assert pl.on_llm_request(**kwargs())["request"]["model"] == PREMIUM
    assert classified == [1] and probes == [1]


def test_escalate_only_provider_does_not_use_paid_routine_opt_in(monkeypatch):
    cfg = config(route_providers=["other"], escalate_providers=["nous"],
                 paid_pool_routine_band=True)
    classified, probes, _ = install(monkeypatch, cfg, "paid", .20)
    assert pl.on_llm_request(**kwargs()) is None
    assert classified == [1] and probes == []


def test_allowlisted_escalation_source_cannot_downgrade_without_route_model(monkeypatch):
    # Live-equivalent behavior: free_src includes both route_providers and the
    # route_models allowlist; escalation-only sources cannot downgrade.
    cfg = config(source_mode="allowlist", route_models=[], escalate_models=[SOURCE])
    classified, probes, _ = install(monkeypatch, cfg, "free", 0.50)
    assert pl.on_llm_request(**kwargs()) is None
    assert classified == [1] and probes == []


def test_allowlisted_route_source_still_downgrades(monkeypatch):
    cfg = config(source_mode="allowlist", route_models=[SOURCE], escalate_models=[])
    install(monkeypatch, cfg, "free", 0.50)
    assert pl.on_llm_request(**kwargs())["request"]["model"] == FLASH


@pytest.mark.parametrize("pool,confidence,target,lane", [
    ("free", 0.20, FREE, "route_providers"),
    ("free", 0.50, FLASH, "route_providers"),
    ("free", 0.70, MID, "route_providers"),
    ("free", 0.90, FREE, "route_providers"),
    ("paid", 0.90, PREMIUM, "escalate_providers"),
])
@pytest.mark.parametrize("path", ["request", "execution"])
def test_cached_replay_rechecks_its_lane_provider(monkeypatch, path, pool, confidence, target, lane):
    cfg = config()
    classified, probes, _ = install(monkeypatch, cfg, pool, confidence)
    assert pl.on_llm_request(**kwargs())["request"]["model"] == target
    cfg[lane] = ["other"]  # The other lane still includes nous.
    calls_before = len(probes)
    if path == "request":
        assert pl.on_llm_request(**kwargs(call=2)) is None
    else:
        follow = kwargs(call=2)
        request = follow.pop("request")
        seen = []
        result = pl.on_llm_execution(request=request,
                                     next_call=lambda r: seen.append(dict(r)) or r,
                                     **follow)
        assert seen == [request]
        assert result["model"] == SOURCE
    assert classified == [1] and len(probes) == calls_before


def test_cached_paid_replay_stops_if_escalation_disabled(monkeypatch):
    cfg = config()
    classified, probes, _ = install(monkeypatch, cfg, "paid", 0.90)
    assert pl.on_llm_request(**kwargs())["request"]["model"] == PREMIUM
    cfg["escalate_enabled"] = False
    assert pl.on_llm_request(**kwargs(call=2)) is None
    assert classified == [1] and probes == [1]


@pytest.mark.parametrize("path", ["request", "execution"])
@pytest.mark.parametrize("pool,lane,target", [("free", "route_models", FLASH),
                                             ("paid", "escalate_models", PREMIUM)])
def test_cached_replay_rechecks_allowlisted_original_source(monkeypatch, path, pool, lane, target):
    cfg = config(source_mode="allowlist", route_models=[SOURCE], escalate_models=[SOURCE])
    classified, probes, _ = install(monkeypatch, cfg, pool, .5 if pool == "free" else .9)
    assert pl.on_llm_request(**kwargs())["request"]["model"] == target
    cfg[lane] = []
    calls_before = len(probes)
    if path == "request":
        assert pl.on_llm_request(**kwargs(call=2)) is None
    else:
        follow = kwargs(call=2)
        request = follow.pop("request")
        seen = []
        result = pl.on_llm_execution(request=request,
                                     next_call=lambda r: seen.append(dict(r)) or r, **follow)
        assert result["model"] == SOURCE and seen == [request]
    assert classified == [1] and len(probes) == calls_before


def test_execution_replay_uses_original_source_not_rewritten_payload(monkeypatch):
    cfg = config(source_mode="allowlist", route_models=[SOURCE], escalate_models=[])
    classified, probes, rows = install(monkeypatch, cfg, "free", .5)
    first = pl.on_llm_request(**kwargs())
    assert first["request"]["model"] == FLASH
    follow = kwargs(call=2)
    follow["model"] = FLASH  # a host can report the already-rewritten model
    request = follow.pop("request")
    request["model"] = FLASH
    seen = []
    result = pl.on_llm_execution(request=request,
                                 next_call=lambda r: seen.append(dict(r)) or r, **follow)
    assert result["model"] == FLASH and seen[0]["model"] == FLASH
    assert any(row.get("reapply") and row.get("target") == FLASH for row in rows)
    assert classified == [1] and probes == []


@pytest.mark.parametrize("pool,confidence,flag,target", [
    ("paid", 0.00, None, None),
    ("paid", 0.349, None, None),
    ("paid", 0.20, False, None),
    ("paid", 0.20, True, FREE),
    ("free", 0.20, None, FREE),
    ("free", 0.20, False, FREE),
    ("paid", 0.35, True, None),
    ("free", 0.35, None, FLASH),
])
def test_d7_routine_band_pool_and_opt_in(monkeypatch, pool, confidence, flag, target):
    cfg = config()
    if flag is not None:
        cfg["paid_pool_routine_band"] = flag
    classified, probes, _ = install(monkeypatch, cfg, pool, confidence)
    out = pl.on_llm_request(**kwargs())
    assert (out["request"]["model"] if out else None) == target
    assert classified == [1] and probes == []
