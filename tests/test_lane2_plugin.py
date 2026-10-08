"""Offline tests for the kfchow-llm-value-router plugin (no network, no live gateway).

Loads the repo-local plugin module by path and drives on_llm_request directly
with synthetic kwargs, so the routing contract is verified without touching a
real turn. Run with a throwaway HOME so nothing here can touch a real config:

    HOME=$(mktemp -d) PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
"""
import importlib.machinery
import importlib.util
import json
import logging
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLUGIN = os.path.join(_REPO, "__init__.py")
_PLUGIN_PATH = _PLUGIN
_loader = importlib.machinery.SourceFileLoader("kfchow_llm_value_router", _PLUGIN)
_spec = importlib.util.spec_from_loader("kfchow_llm_value_router", _loader)
pl = importlib.util.module_from_spec(_spec)
sys.modules["kfchow_llm_value_router"] = pl
_loader.exec_module(pl)


def _kwargs(provider="nous", model="z-ai/glm-5.3-flash", api_call_count=1,
            turn_id="t1", messages=None):
    return {
        "provider": provider, "model": model, "api_call_count": api_call_count,
        "turn_id": turn_id, "session_id": "s1",
        "request": {"model": model,
                    "messages": messages or [{"role": "user", "content": "hi"}]},
    }


def _cfg(**over):
    base = {"enabled": True, "mode": "shadow", "confidence_gate": 0.65,
            "route_providers": ["nous"],
            "route_models": ["z-ai/glm-5.3-flash", "glm-5.3-flash"],
            "free_model": "inclusionai/ling-3.0-flash-sante:free",
            "escalate_enabled": True, "escalate_providers": ["nous"],
            "escalate_models": ["z-ai/glm-5.3", "glm-5.3"],
            "premium_model": "anthropic/claude-sonnet-5.5",
            "escalate_confidence_gate": 0.65, "credit_probe": True,
            "send_excerpt": True}
    base.update(over)
    return base


def _reset_decisions():
    """Clear the module-level turn-decision cache so tests start from empty state."""
    with pl._DECISIONS_LOCK:
        pl._DECISIONS.clear()


# --------------------------------------------------------------- escalation

def _mid_kwargs():
    return _kwargs(provider="nous", model="z-ai/glm-5.3")


def test_escalation_rewrites_mid_turn_to_premium(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.90))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: True)
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    kw = _mid_kwargs()
    kw["turn_id"] = "esc-1"
    out = pl.on_llm_request(**kw)
    assert out is not None
    assert out["request"]["model"] == "anthropic/claude-sonnet-5.5"
    assert "ESCALATED" in out["reason"]


def test_escalation_blocked_when_credits_dead(monkeypatch):
    """THE outage lesson: dark paid lane must NOT escalate — keep the cheap rung."""
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.99))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: False)
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    assert pl.on_llm_request(**_mid_kwargs()) is None


def test_escalation_fails_closed_when_probe_raises(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.99))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    # Make the probe blow up: unknown credit state must NOT escalate.
    monkeypatch.setattr(pl, "urllib", type("U", (), {"request": None})())
    pl._CREDIT["ok"] = None
    pl._CREDIT["checked_at"] = 0.0
    _reset_decisions()
    assert pl.on_llm_request(**_mid_kwargs()) is None   # unknown => no escalation


def test_escalation_from_flash_default_any_mode(monkeypatch):
    """v1.0.7 (source_mode 'any', the default): a flash-default turn IS an
    escalation source. A paid 0.99 verdict climbs to the premium rung; the
    credit-probe path is mocked alive (no network, no crash)."""
    recs, probes = [], []
    monkeypatch.setattr(pl, "_load_config",
                        lambda: _cfg(mode="live", source_mode="any"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.99))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: probes.append(1) or True)
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    out = pl.on_llm_request(**_kwargs())                # flash default
    assert out is not None
    assert out["request"]["model"] == "anthropic/claude-sonnet-5.5"
    assert "ESCALATED" in out["reason"]
    assert probes == [1]                                # probed exactly once
    assert recs[-1]["escalate_applied"] is True and recs[-1]["credit_ok"] is True


def test_escalation_ignores_flash_default_allowlist(monkeypatch):
    """v1.0.6 semantics preserved under source_mode 'allowlist': the flash
    default is not listed in escalate_models, so it is NOT an escalation source.
    (Counted doubles, not raising ones: the middleware fails open, so a raise
    inside a double would be swallowed and the test would pass vacuously.)"""
    recs, probes = [], []
    monkeypatch.setattr(pl, "_load_config",
                        lambda: _cfg(mode="live", source_mode="allowlist"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.99))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: probes.append(1) or True)
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    assert pl.on_llm_request(**_kwargs()) is None       # flash default
    assert recs[-1]["escalate_eligible"] is False
    assert probes == []                                 # not a candidate -> no probe spent


def test_escalation_disabled_by_config(monkeypatch):
    monkeypatch.setattr(pl, "_load_config",
                        lambda: _cfg(mode="live", escalate_enabled=False))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.99))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: True)
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    assert pl.on_llm_request(**_mid_kwargs()) is None


def test_escalation_respects_its_own_gate(monkeypatch):
    monkeypatch.setattr(pl, "_load_config",
                        lambda: _cfg(mode="live", escalate_confidence_gate=0.95))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.70))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: True)
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    assert pl.on_llm_request(**_mid_kwargs()) is None


def test_credit_probe_is_cached(monkeypatch):
    calls = {"n": 0}

    def fake_open(*a, **k):
        calls["n"] += 1
        raise OSError("no net")

    monkeypatch.setattr(pl.urllib, "request",
                        type("R", (), {"urlopen": staticmethod(fake_open),
                                       "Request": staticmethod(lambda *a, **k: None)})())
    pl._CREDIT["ok"] = None
    pl._CREDIT["checked_at"] = 0.0
    pl._paid_lane_alive()
    first = calls["n"]
    pl._paid_lane_alive()          # within TTL -> cached
    assert calls["n"] == first


def test_per_provider_rungs_never_cross_providers(monkeypatch):
    """THE provider-crossing rule: a rewrite must keep the ORIGINATING provider."""
    cfg = _cfg(mode="live")
    cfg["rungs"] = {
        "nous": {"mid": ["z-ai/glm-5.3"], "premium": "anthropic/claude-sonnet-5.5"},
        "opencode-go": {"mid": ["glm-5.3-flash"], "premium": "glm-5.3",
                        "free": "space-bunny-free"},
    }
    cfg["escalate_providers"] = ["nous", "opencode-go"]
    cfg["escalate_models"] = ["z-ai/glm-5.3", "glm-5.3-flash"]
    monkeypatch.setattr(pl, "_load_config", lambda: cfg)
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.95))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: True)
    monkeypatch.setattr(pl, "_log", lambda rec: None)

    _reset_decisions()
    kw = _kwargs(provider="opencode-go", model="glm-5.3-flash")
    kw["turn_id"] = "oc-1"
    out = pl.on_llm_request(**kw)
    assert out is not None
    # Escalation must land on THIS provider's premium, not the other lane's.
    assert out["request"]["model"] == "glm-5.3"

    _reset_decisions()
    kw2 = _kwargs(provider="nous", model="z-ai/glm-5.3")
    kw2["turn_id"] = "nous-1"
    out2 = pl.on_llm_request(**kw2)
    assert out2 is not None
    assert out2["request"]["model"] == "anthropic/claude-sonnet-5.5"


def test_subscription_lane_skips_credit_probe(monkeypatch):
    """A flat-rate subscription has no per-call credit wall — no probe needed."""
    cfg = _cfg(mode="live")
    cfg["rungs"] = {"opencode-go": {"premium": "glm-5.3"}}
    cfg["escalate_providers"] = ["opencode-go"]
    cfg["escalate_models"] = ["glm-5.3-flash"]
    monkeypatch.setattr(pl, "_load_config", lambda: cfg)
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.95))
    monkeypatch.setattr(pl, "_log", lambda rec: None)

    def boom():
        raise AssertionError("must not probe credits for a subscription lane")

    monkeypatch.setattr(pl, "_paid_lane_alive", boom)
    _reset_decisions()
    kw = _kwargs(provider="opencode-go", model="glm-5.3-flash")
    kw["turn_id"] = "oc-2"
    out = pl.on_llm_request(**kw)
    assert out is not None and out["request"]["model"] == "glm-5.3"


def test_rung_falls_back_to_flat_key():
    cfg = _cfg(mode="live")          # no 'rungs' -> flat premium_model used
    assert pl._rung(cfg, "nous", "premium", "premium_model") == \
        "anthropic/claude-sonnet-5.5"
    assert pl._rung(cfg, "unknown-provider", "free", "free_model") == \
        "inclusionai/ling-3.0-flash-sante:free"


# --------------------------------------------------------------- features

def test_features_are_content_free_shape_only():
    f = pl._features({"messages": [{"role": "user", "content": "def foo(): pass"}],
                      "tools": [{"x": 1}, {"y": 2}]})
    assert f["n_tools"] == 2
    assert f["has_code"] is True
    # shape fields stay content-free; the excerpt is the ONE deliberate exception
    assert "foo" not in json.dumps({k: v for k, v in f.items() if k != "task_excerpt"})
    assert f["task_excerpt"].startswith("def foo")


def test_excerpt_is_capped():
    f = pl._features({"messages": [{"role": "user", "content": "x" * 5000}]},
                     excerpt_chars=1200)
    assert len(f["task_excerpt"]) == 1200


def test_features_multimodal_blocks():
    f = pl._features({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hello there"}]}]})
    assert f["last_msg_chars"] == len("hello there")


# --------------------------------------------------------------- send_excerpt

def test_send_excerpt_false_omits_excerpt_from_features(monkeypatch):
    """The toggle must strip the excerpt at the SOURCE so it reaches
    neither the vendor call nor the log row."""
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(send_excerpt=False))
    cfg = pl._load_config()
    feats = pl._features({"messages": [{"role": "user", "content": "SECRETTEXT"}]},
                         include_excerpt=bool(cfg.get("send_excerpt", True)))
    assert "task_excerpt" not in feats
    assert "SECRETTEXT" not in json.dumps(feats)


def test_send_excerpt_false_keeps_excerpt_out_of_log(monkeypatch):
    recs = []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live", send_excerpt=False))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("free", 0.99))
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    out = pl.on_llm_request(**_kwargs(turn_id="sx-1",
                                      messages=[{"role": "user",
                                                 "content": "SECRETTEXT"}]))
    # live mode still rewrites; the point is the excerpt never reaches the log
    assert out is not None and out["request"]["model"] == "inclusionai/ling-3.0-flash-sante:free"
    assert recs and recs[-1]["features"] is not None
    assert "task_excerpt" not in recs[-1]["features"]
    assert "SECRETTEXT" not in json.dumps(recs[-1]["features"])


def test_send_excerpt_true_includes_excerpt_in_log(monkeypatch):
    recs = []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live", send_excerpt=True))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("free", 0.99))
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    pl.on_llm_request(**_kwargs(turn_id="sx-2",
                                messages=[{"role": "user", "content": "VISIBLETEXT"}]))
    assert recs and recs[-1]["features"].get("task_excerpt") == "VISIBLETEXT"


def test_send_excerpt_defaults_true_in_code_defaults():
    # The DEFAULT config (no config file present) must default send_excerpt to True.
    orig = pl.CONFIG_PATH
    pl.CONFIG_PATH = "/nonexistent/config-path.json"
    try:
        cfg = pl._load_config()
        assert cfg["send_excerpt"] is True
    finally:
        pl.CONFIG_PATH = orig


# --------------------------------------------------------------- core routing

def test_live_mode_rewrites_eligible_request(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("free", 0.70))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    out = pl.on_llm_request(**_kwargs())
    assert out is not None
    assert out["request"]["model"] == "inclusionai/ling-3.0-flash-sante:free"
    assert out["source"] == "kfchow-llm-value-router"
    # the message list must survive the rewrite untouched
    assert out["request"]["messages"] == [{"role": "user", "content": "hi"}]


def test_shadow_mode_never_rewrites(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="shadow"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("free", 0.99))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    assert pl.on_llm_request(**_kwargs()) is None


def test_below_gate_never_rewrites(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("free", 0.40))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    assert pl.on_llm_request(**_kwargs()) is None


def test_paid_pool_never_rewrites(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.99))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    assert pl.on_llm_request(**_kwargs()) is None


def test_followup_calls_are_untouched(monkeypatch):
    """api_call_count >= 2 is a tool-loop follow-up: the turn must not switch model."""
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("free", 0.99))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    assert pl.on_llm_request(**_kwargs(api_call_count=3)) is None


def test_ineligible_provider_is_skipped_and_logged(monkeypatch):
    recs = []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    out = pl.on_llm_request(**_kwargs(provider="opencode-go",
                                      model="longcat-2.5-preview-paid"))
    assert out is None
    assert recs and recs[0]["skip_reason"] == "provider_not_eligible"


def test_already_free_model_never_rerouted(monkeypatch):
    recs = []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(
        mode="live", route_models=["inclusionai/ling-3.0-flash-sante:free"]))
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    pl.on_llm_request(**_kwargs(model="inclusionai/ling-3.0-flash-sante:free"))
    assert recs and recs[0]["skip_reason"] == "already_free_tier"


def test_classify_failure_is_fail_open(monkeypatch):
    """A vendor outage must leave the request untouched, never raise."""
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_log", lambda rec: None)

    def boom(feats, cfg):
        raise RuntimeError("vendor down")

    monkeypatch.setattr(pl, "_classify", boom)
    _reset_decisions()
    assert pl.on_llm_request(**_kwargs()) is None


def test_malformed_config_falls_back_to_safe_defaults(monkeypatch):
    monkeypatch.setattr(pl, "CONFIG_PATH", "/nonexistent/lane2_config.json")
    cfg = pl._load_config()
    assert cfg["mode"] == "shadow"          # never silently live
    assert cfg["enabled"] is True
    assert cfg["confidence_gate"] == pl.DEFAULT_GATE


# --------------------------------------------------------------- env-first key

def test_nous_key_from_env_only(monkeypatch):
    """V16/V17: the key comes from os.environ, NEVER from ~/.hermes/.env."""
    monkeypatch.setenv("NOUS_API_KEY", "env-resolved-key")
    assert pl._nous_api_key() == "env-resolved-key"
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    assert pl._nous_api_key() is None


def test_hermes_home_bases_paths(monkeypatch):
    """Paths base on HERMES_HOME (no absolute home hardcoded)."""
    # The module resolves paths at import; verify no absolute home literal is
    # baked in and that expanduser-with-default agrees with the module value.
    src = open(_PLUGIN_PATH, "r", encoding="utf-8").read()
    assert "os.environ.get(\"HERMES_HOME\"" in src
    # no absolute home path baked in (checked via fragments so this file stays clean too)
    assert "not in src" or ("/Us" + "ers/" not in src and "/ho" + "me/" not in src)


def test_classify_passes_retries_zero_and_timeout(monkeypatch):
    """The routing call is bounded and un-retried."""
    seen = {}

    class FakeJC:
        def evaluate(self, state, questions, timeout=None, retries=None):
            seen["timeout"] = timeout
            seen["retries"] = retries
            return {"answers": {pl.ROUTING_QUESTION_ID:
                                {"choice": "free", "confidence": 0.99}}}

    monkeypatch.setattr(pl, "_vendored_client", lambda: FakeJC())
    pool, conf = pl._classify({"task_excerpt": "x"}, _cfg(jev_timeout_s=7))
    assert (pool, conf) == ("free", 0.99)
    assert seen["retries"] == 0
    assert seen["timeout"] == 7.0


def test_register_registers_both_middlewares():
    """F1/V2 truth-in-manifest: register() declares exactly the two middlewares
    it ships, in plugin.yaml provides_middleware sync order."""
    registered = []

    class Ctx:
        def register_middleware(self, kind, cb):
            registered.append(kind)

    pl.register(Ctx())
    assert registered == ["llm_request", "llm_execution"]


# ===================== v1.0.7 — SOURCE-AGNOSTIC LADDER ======================
# Eligibility is provider-match only (source_mode "any", the default): any
# default model moves BOTH ways on its own provider's rungs. _FREE_HINTS stays a
# hard guard that runs first; a turn is never "moved" onto the model it is
# already on. Counted doubles throughout (the middleware fails open, so a
# raising double would be swallowed and a test would pass vacuously).

_SONNET = "anthropic/claude-sonnet-5.5"
_FLASH = "z-ai/glm-5.3-flash"
_LING = "inclusionai/ling-3.0-flash-sante:free"
_GPT = "openai/gpt-5.4"


def _cfg7(**over):
    """Production-shaped ladder: free 0.65, flash band [0.35, 0.65), esc 0.80."""
    base = _cfg(mode="live", flash_gate=0.35, flash_model=_FLASH,
                escalate_confidence_gate=0.80)
    base.update(over)
    return base


def _run7(monkeypatch, model, pool, conf, tid, **over):
    """Drive one first-call turn. Returns (out, last_row, probes, classify_calls)."""
    recs, probes, classified = [], [], []

    def cls(feats, cfg):
        classified.append(1)
        return pool, conf

    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7(**over))
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: probes.append(1) or True)
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    out = pl.on_llm_request(**_kwargs(model=model, turn_id=tid))
    return out, (recs[-1] if recs else None), probes, classified


# ---- sonnet (premium) default: the down-moves that were a v1.0.6 gap -------

def test_sonnet_default_free_065_moves_down_to_ling(monkeypatch):
    out, row, probes, _ = _run7(monkeypatch, _SONNET, "free", 0.65, "s7-1")
    assert out is not None and out["request"]["model"] == _LING
    assert row["tier"] == "free" and row["applied"] is True
    assert probes == []                                   # downgrade never probes


def test_sonnet_default_free_064_moves_down_to_flash(monkeypatch):
    out, row, probes, _ = _run7(monkeypatch, _SONNET, "free", 0.64, "s7-2")
    assert out is not None and out["request"]["model"] == _FLASH
    assert row["tier"] == "flash" and row["applied"] is True
    assert probes == []


def test_sonnet_default_paid_079_stays_on_sonnet(monkeypatch):
    # Production gate 0.80: 0.79 is below the escalation gate -> stays put,
    # reason pool_paid, no rewrite, no probe.
    out, row, probes, _ = _run7(monkeypatch, _SONNET, "paid", 0.79, "s7-3a")
    assert out is None and row["skip_reason"] == "pool_paid" and probes == []
    # With this file's 0.65 escalation gate the band DOES fire at 0.79, but the
    # rung IS the source model -> already_on_target, still nothing rewritten.
    out, row, probes, _ = _run7(monkeypatch, _SONNET, "paid", 0.79, "s7-3b",
                                escalate_confidence_gate=0.65)
    assert out is None and row["skip_reason"] == "already_on_target"
    assert row["escalate_eligible"] is False and probes == []


def test_sonnet_default_paid_085_never_self_escalates_no_probe(monkeypatch):
    out, row, probes, _ = _run7(monkeypatch, _SONNET, "paid", 0.85, "s7-4")
    assert out is None
    assert row["skip_reason"] == "already_on_target"
    assert row["escalate_eligible"] is False and row["escalate_applied"] is False
    assert row["premium_model"] is None and row["credit_ok"] is None
    assert probes == []                                   # no credit probe spent


def test_premium_identity_is_vendor_prefix_and_case_tolerant(monkeypatch):
    # Host-reported id differs from the rung only by vendor prefix / case.
    for i, m in enumerate(("claude-sonnet-5.5", "Anthropic/Claude-Sonnet-5.5")):
        out, row, probes, _ = _run7(monkeypatch, m, "paid", 0.95, "s7-id-%d" % i)
        assert out is None and probes == []
        assert row["skip_reason"] == "already_on_target"


# ---- gpt-5.4 (foreign default): full ladder both ways ----------------------

def test_gpt54_default_free_065_to_ling(monkeypatch):
    out, row, _, _ = _run7(monkeypatch, _GPT, "free", 0.65, "g7-1")
    assert out is not None and out["request"]["model"] == _LING
    assert row["tier"] == "free"


def test_gpt54_default_free_064_to_flash(monkeypatch):
    out, row, _, _ = _run7(monkeypatch, _GPT, "free", 0.64, "g7-2")
    assert out is not None and out["request"]["model"] == _FLASH
    assert row["tier"] == "flash"


def test_gpt54_default_paid_085_to_sonnet(monkeypatch):
    out, row, probes, _ = _run7(monkeypatch, _GPT, "paid", 0.85, "g7-3")
    assert out is not None and out["request"]["model"] == _SONNET
    assert "ESCALATED" in out["reason"]
    assert row["tier"] == "paid" and row["credit_ok"] is True
    assert probes == [1]                                  # metered provider: probed once


def test_gpt54_default_paid_085_blocked_when_credits_dead(monkeypatch):
    """Fail-closed credit gate still applies to a foreign-default source."""
    recs = []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7())
    monkeypatch.setattr(pl, "_classify", lambda f, c: ("paid", 0.85))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: False)
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    assert pl.on_llm_request(**_kwargs(model=_GPT, turn_id="g7-4")) is None
    assert recs[-1]["credit_ok"] is False and recs[-1]["escalate_applied"] is False


# ---- ling:free default: _FREE_HINTS precedence over source-agnostic --------

@pytest.mark.parametrize("mode", ["any", "allowlist", [_LING], "bogus"])
@pytest.mark.parametrize("pool,conf", [("paid", 0.99), ("paid", 0.85), ("free", 0.99),
                                       ("free", 0.65), ("free", 0.50), ("free", 0.10)])
def test_ling_free_default_never_touched(monkeypatch, mode, pool, conf):
    out, row, probes, classified = _run7(monkeypatch, _LING, pool, conf,
                                         "l7-%s-%s-%s" % (str(mode)[:6], pool, conf),
                                         source_mode=mode, route_models=[_LING],
                                         escalate_models=[_LING])
    assert out is None
    assert row["skip_reason"] == "already_free_tier"
    assert classified == [] and probes == []              # never even classified


# ---- source_mode narrowing --------------------------------------------------

def test_explicit_list_source_mode_narrows_to_listed_sources(monkeypatch):
    # Listed source behaves exactly like "any" ...
    out, _, _, _ = _run7(monkeypatch, _SONNET, "free", 0.65, "x7-1",
                         source_mode=[_SONNET])
    assert out is not None and out["request"]["model"] == _LING
    # ... an unlisted source (gpt-5.4) is skipped before classification ...
    out, row, probes, classified = _run7(monkeypatch, _GPT, "paid", 0.95, "x7-2",
                                         source_mode=[_SONNET])
    assert out is None and row["skip_reason"] == "model_not_eligible"
    assert classified == [] and probes == []
    # ... and route_models/escalate_models are IGNORED when a list is given:
    # the flash default is in route_models but not in the list -> not eligible.
    out, row, _, classified = _run7(monkeypatch, _FLASH, "free", 0.99, "x7-3",
                                    source_mode=[_SONNET])
    assert out is None and row["skip_reason"] == "model_not_eligible"
    assert classified == []


def test_explicit_list_still_requires_provider_match(monkeypatch):
    out, row, _, classified = _run7(monkeypatch, _SONNET, "free", 0.99, "x7-4",
                                    source_mode=[_SONNET], route_providers=["other"],
                                    escalate_providers=["other"])
    assert out is None and row["skip_reason"] == "provider_not_eligible"
    assert classified == []


def test_empty_list_source_mode_makes_nothing_eligible(monkeypatch):
    out, row, _, classified = _run7(monkeypatch, _SONNET, "free", 0.99, "x7-5",
                                    source_mode=[])
    assert out is None and row["skip_reason"] == "model_not_eligible"
    assert classified == []


def test_allowlist_mode_restores_v106_semantics(monkeypatch):
    # unlisted premium / foreign defaults: untouched (the v1.0.6 gap, by request)
    for i, m in enumerate((_SONNET, _GPT)):
        out, row, _, classified = _run7(monkeypatch, m, "free", 0.99, "a7-%d" % i,
                                        source_mode="allowlist")
        assert out is None and row["skip_reason"] == "model_not_eligible"
        assert classified == []
    # listed flash default (route_models): down-move to ling still works
    out, _, _, _ = _run7(monkeypatch, _FLASH, "free", 0.99, "a7-2",
                         source_mode="allowlist")
    assert out is not None and out["request"]["model"] == _LING
    # listed mid default (escalate_models): escalates, and flash band applies
    out, _, _, _ = _run7(monkeypatch, "z-ai/glm-5.3", "paid", 0.90, "a7-3",
                         source_mode="allowlist")
    assert out is not None and out["request"]["model"] == _SONNET
    out, _, _, _ = _run7(monkeypatch, "z-ai/glm-5.3", "free", 0.50, "a7-4",
                         source_mode="allowlist")
    assert out is not None and out["request"]["model"] == _FLASH


def test_unset_null_or_empty_source_mode_means_any(monkeypatch):
    for i, mode in enumerate((None, "", "ANY", " any ")):
        out, _, _, _ = _run7(monkeypatch, _SONNET, "free", 0.65, "n7-%d" % i,
                             source_mode=mode)
        assert out is not None and out["request"]["model"] == _LING, repr(mode)


def test_default_config_source_mode_is_any(monkeypatch):
    monkeypatch.setattr(pl, "CONFIG_PATH", "/nonexistent/lane2_config.json")
    assert pl._load_config()["source_mode"] == "any" == pl.DEFAULT_SOURCE_MODE


def test_unrecognised_source_mode_falls_back_to_allowlist_and_warns_once(
        monkeypatch, caplog):
    monkeypatch.setattr(pl, "_WARNED_SOURCE_MODES", set())   # fresh, auto-restored
    with caplog.at_level(logging.WARNING, logger=pl.logger.name):
        # allowlist fallback: unlisted sonnet/gpt untouched ...
        for i, m in enumerate((_SONNET, _GPT)):
            out, row, _, classified = _run7(monkeypatch, m, "free", 0.99,
                                            "u7-%d" % i, source_mode="alwo")
            assert out is None and row["skip_reason"] == "model_not_eligible"
            assert classified == []
        # ... listed flash default still routes (v1.0.6 behaviour)
        out, _, _, _ = _run7(monkeypatch, _FLASH, "free", 0.99, "u7-2",
                             source_mode="alwo")
        assert out is not None and out["request"]["model"] == _LING
    warns = [r for r in caplog.records if "unrecognised source_mode" in r.getMessage()]
    assert len(warns) == 1                                 # once, not per turn
    assert "alwo" in warns[0].getMessage()
    # a different bad value warns again (one per distinct value)
    with caplog.at_level(logging.WARNING, logger=pl.logger.name):
        _run7(monkeypatch, _SONNET, "free", 0.99, "u7-3", source_mode=42)
    warns = [r for r in caplog.records if "unrecognised source_mode" in r.getMessage()]
    assert len(warns) == 2


def test_provider_not_in_route_providers_is_provider_not_eligible(monkeypatch):
    # unchanged by v1.0.7: provider match is required in EVERY mode
    for i, mode in enumerate(("any", "allowlist", [_SONNET])):
        recs = []
        monkeypatch.setattr(pl, "_load_config",
                            lambda m=mode: _cfg7(source_mode=m))
        monkeypatch.setattr(pl, "_log", recs.append)
        _reset_decisions()
        out = pl.on_llm_request(**_kwargs(provider="openai-direct", model=_SONNET,
                                          turn_id="p7-%d" % i))
        assert out is None and recs[-1]["skip_reason"] == "provider_not_eligible"


def test_escalate_disabled_blocks_escalation_but_not_downmoves_in_any(monkeypatch):
    # esc lane off: paid verdict is inert; the free lane (route_providers) still
    # moves a premium-default turn down.
    out, _, probes, _ = _run7(monkeypatch, _GPT, "paid", 0.95, "e7-1",
                              escalate_enabled=False)
    assert out is None and probes == []
    out, _, _, _ = _run7(monkeypatch, _GPT, "free", 0.65, "e7-2",
                         escalate_enabled=False)
    assert out is not None and out["request"]["model"] == _LING


def test_free_rung_identity_guard_when_rung_has_no_free_hint(monkeypatch):
    """_FREE_HINTS normally catches a free-rung source first. If an operator's
    free rung id carries no free hint, the identity guard must still stop a
    no-op 'move' onto the model the turn is already on."""
    out, row, probes, classified = _run7(
        monkeypatch, "acme/tiny-model", "free", 0.90, "i7-1",
        free_model="acme/tiny-model", route_models=["acme/tiny-model"])
    assert out is None and classified == [1] and probes == []
    assert row["skip_reason"] == "already_on_target"
    assert row["eligible"] is False and row["applied"] is False


def test_flash_default_in_flash_band_is_already_on_target_not_rewritten(monkeypatch):
    out, row, probes, _ = _run7(monkeypatch, _FLASH, "free", 0.50, "i7-2")
    assert out is None and probes == []
    assert row["skip_reason"] == "already_on_target"
    assert row["flash_model"] is None and row["applied"] is False


def test_manifest_declares_source_mode_and_versions_agree():
    """plugin.yaml must declare source_mode with the code default, and the two
    version strings (manifest, CITATION.cff) must agree. Regex-only: no yaml dep."""
    import re
    manifest = open(os.path.join(_REPO, "plugin.yaml"), encoding="utf-8").read()
    citation = open(os.path.join(_REPO, "CITATION.cff"), encoding="utf-8").read()
    m_ver = re.search(r'^version:\s*"?([\d.]+)"?\s*$', manifest, re.M).group(1)
    c_ver = re.search(r'^version:\s*"?([\d.]+)"?\s*$', citation, re.M).group(1)
    assert m_ver == c_ver
    block = re.search(r"^  source_mode:\n((?:    .*\n)+)", manifest, re.M)
    assert block, "config_schema.source_mode missing from plugin.yaml"
    assert re.search(r"^    default:\s*[\"']?%s[\"']?\s*$" % pl.DEFAULT_SOURCE_MODE,
                     block.group(1), re.M)


# ---- helpers ----------------------------------------------------------------

def test_same_model_helper():
    assert pl._same_model("z-ai/glm-5.3", "glm-5.3")
    assert pl._same_model("Anthropic/Claude-Sonnet-5.5", _SONNET)
    assert not pl._same_model("z-ai/glm-5.3", "z-ai/glm-5.3-flash")
    assert not pl._same_model("", "")                      # empty never matches
    assert not pl._same_model(None, "glm-5.3")


# ---- boundary matrix, both directions, both sources -------------------------

_CONFS = [0.30, 0.34, 0.35, 0.64, 0.65, 0.79, 0.80]


def _expect_free_pool(source, conf):
    """(wire model or None, skip_reason) for a FREE-pool verdict."""
    if conf >= 0.65:
        return _LING, None
    if conf >= 0.35:
        return (None, "already_on_target") if source == _FLASH else (_FLASH, None)
    return None, "below_gate"


def _expect_paid_pool(source, conf):
    """(wire model or None, skip_reason) for a PAID-pool verdict (esc gate 0.80)."""
    if conf >= 0.80:
        return (None, "already_on_target") if source == _SONNET else (_SONNET, None)
    return None, "pool_paid"


@pytest.mark.parametrize("conf", _CONFS)
@pytest.mark.parametrize("source", [_FLASH, _SONNET], ids=["flash-default", "sonnet-default"])
def test_boundary_matrix_both_directions(monkeypatch, source, conf):
    for pool, expect in (("free", _expect_free_pool), ("paid", _expect_paid_pool)):
        want_model, want_skip = expect(source, conf)
        out, row, probes, classified = _run7(
            monkeypatch, source, pool, conf, "bm-%s-%s-%s" % (source[-5:], pool, conf))
        ctx = (source, pool, conf)
        assert len(classified) == 1, ctx
        assert (out["request"]["model"] if out else None) == want_model, ctx
        if want_model is None:
            assert row["skip_reason"] == want_skip, ctx
            assert row["applied"] is False and row["escalate_applied"] is False, ctx
        else:
            assert row["skip_reason"] is None, ctx
            # free/flash rows flag `applied`; the escalation lane flags its own key
            flag = "escalate_applied" if want_model == _SONNET else "applied"
            assert row[flag] is True, ctx
        # probes only ever spent on a real escalation
        assert probes == ([1] if want_model == _SONNET else []), ctx


# ---- reapply / decision-cache under the new decisions -----------------------

def _exec7(model, tid, api_call_count=2, provider="nous"):
    kw = _kwargs(provider=provider, model=model, turn_id=tid,
                 api_call_count=api_call_count)
    return dict(kw["request"]), {"provider": provider, "model": model,
                                 "turn_id": tid, "session_id": "s1",
                                 "api_call_count": api_call_count}


def _next7(seen):
    def _n(request):
        seen.append(dict(request))
        return {"model": request.get("model")}
    return _n


def test_reapply_sonnet_default_flash_downmove_via_request_and_exec(monkeypatch):
    classified, recs, probes = [], [], []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7())
    monkeypatch.setattr(pl, "_classify",
                        lambda f, c: classified.append(1) or ("free", 0.50))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: probes.append(1) or True)
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    first = pl.on_llm_request(**_kwargs(model=_SONNET, turn_id="r7-1"))
    assert first["request"]["model"] == _FLASH
    # request-path follow-up re-applies, same decision-cache key (session:turn)
    second = pl.on_llm_request(**_kwargs(model=_SONNET, turn_id="r7-1",
                                         api_call_count=2))
    assert second is not None and second["request"]["model"] == _FLASH
    # exec-path follow-up (host rebuilt the payload from the sonnet default)
    req, ctx = _exec7(_SONNET, "r7-1")
    seen = []
    out = pl.on_llm_execution(request=req, next_call=_next7(seen), **ctx)
    assert req["model"] == _SONNET and seen[0]["model"] == _FLASH
    assert out["model"] == _FLASH
    assert len(classified) == 1 and probes == []           # one classify, flash never probes
    assert [r["tier"] for r in recs if r.get("reapply")] == ["flash", "flash"]


def test_reapply_foreign_default_escalation_persists_and_rechecks_credit(monkeypatch):
    classified, recs, probes = [], [], []
    alive = {"v": True}
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7())
    monkeypatch.setattr(pl, "_classify",
                        lambda f, c: classified.append(1) or ("paid", 0.90))
    monkeypatch.setattr(pl, "_paid_lane_alive",
                        lambda: probes.append(1) or alive["v"])
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    first = pl.on_llm_request(**_kwargs(model=_GPT, turn_id="r7-2"))
    assert first["request"]["model"] == _SONNET
    seen = []
    req, ctx = _exec7(_GPT, "r7-2")
    pl.on_llm_execution(request=req, next_call=_next7(seen), **ctx)
    assert seen[0]["model"] == _SONNET                     # persisted on the wire
    # the lane goes dark mid-turn: re-apply fails closed, original goes through
    alive["v"] = False
    seen2 = []
    req, ctx = _exec7(_GPT, "r7-2", api_call_count=3)
    pl.on_llm_execution(request=req, next_call=_next7(seen2), **ctx)
    assert seen2[0]["model"] == _GPT
    assert any(r.get("credit_lost") for r in recs)
    assert len(classified) == 1                            # never re-classified


def test_reapply_none_decision_for_on_target_turn_is_untouched(monkeypatch):
    """sonnet-default paid 0.85 stores decision 'none': follow-ups on either path
    pass through untouched, no re-classify, no probe, no extra rows."""
    classified, recs, probes = [], [], []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7())
    monkeypatch.setattr(pl, "_classify",
                        lambda f, c: classified.append(1) or ("paid", 0.85))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: probes.append(1) or True)
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    assert pl.on_llm_request(**_kwargs(model=_SONNET, turn_id="r7-3")) is None
    assert pl.on_llm_request(**_kwargs(model=_SONNET, turn_id="r7-3",
                                       api_call_count=2)) is None
    seen = []
    req, ctx = _exec7(_SONNET, "r7-3")
    pl.on_llm_execution(request=req, next_call=_next7(seen), **ctx)
    assert seen[0] == req and seen[0]["model"] == _SONNET
    assert len(classified) == 1 and probes == []
    assert len(recs) == 1 and recs[0]["skip_reason"] == "already_on_target"


def test_reapply_decision_keys_unchanged_and_provider_still_pinned(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7())
    monkeypatch.setattr(pl, "_classify", lambda f, c: ("free", 0.90))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    pl.on_llm_request(**_kwargs(model=_SONNET, turn_id="r7-4"))
    with pl._DECISIONS_LOCK:
        assert list(pl._DECISIONS) == ["s1:r7-4"]          # key shape unchanged
        d = pl._DECISIONS["s1:r7-4"]
    assert d["decision"] == "free" and d["target"] == _LING and d["provider"] == "nous"
    # follow-up on ANOTHER provider is never rewritten (decision pinned)
    assert pl.on_llm_request(**_kwargs(provider="other", model=_SONNET,
                                       turn_id="r7-4", api_call_count=2)) is None
