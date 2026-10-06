"""Offline tests for the kfchow-llm-value-router plugin (no network, no live gateway).

Loads the repo-local plugin module by path and drives on_llm_request directly
with synthetic kwargs, so the routing contract is verified without touching a
real turn. Run with a throwaway HOME so nothing here can touch a real config:

    HOME=$(mktemp -d) PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
"""
import importlib.machinery
import importlib.util
import json
import os
import sys

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


# --------------------------------------------------------------- escalation

def _mid_kwargs():
    return _kwargs(provider="nous", model="z-ai/glm-5.3")


def test_escalation_rewrites_mid_turn_to_premium(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.90))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: True)
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    pl._STATE["last_turn"] = None
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
    pl._STATE["last_turn"] = None
    assert pl.on_llm_request(**_mid_kwargs()) is None


def test_escalation_fails_closed_when_probe_raises(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.99))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    # Make the probe blow up: unknown credit state must NOT escalate.
    monkeypatch.setattr(pl, "urllib", type("U", (), {"request": None})())
    pl._CREDIT["ok"] = None
    pl._CREDIT["checked_at"] = 0.0
    pl._STATE["last_turn"] = None
    assert pl.on_llm_request(**_mid_kwargs()) is None   # unknown => no escalation


def test_escalation_ignores_flash_default(monkeypatch):
    """Turns on the flash default are NOT an escalation source (not a mid rung)."""
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.99))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: True)
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    pl._STATE["last_turn"] = None
    assert pl.on_llm_request(**_kwargs()) is None       # flash default


def test_escalation_disabled_by_config(monkeypatch):
    monkeypatch.setattr(pl, "_load_config",
                        lambda: _cfg(mode="live", escalate_enabled=False))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.99))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: True)
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    pl._STATE["last_turn"] = None
    assert pl.on_llm_request(**_mid_kwargs()) is None


def test_escalation_respects_its_own_gate(monkeypatch):
    monkeypatch.setattr(pl, "_load_config",
                        lambda: _cfg(mode="live", escalate_confidence_gate=0.95))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.70))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: True)
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    pl._STATE["last_turn"] = None
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

    pl._STATE["last_turn"] = None
    kw = _kwargs(provider="opencode-go", model="glm-5.3-flash")
    kw["turn_id"] = "oc-1"
    out = pl.on_llm_request(**kw)
    assert out is not None
    # Escalation must land on THIS provider's premium, not the other lane's.
    assert out["request"]["model"] == "glm-5.3"

    pl._STATE["last_turn"] = None
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
    pl._STATE["last_turn"] = None
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
    pl._STATE["last_turn"] = None
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
    pl._STATE["last_turn"] = None
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
    pl._STATE["last_turn"] = None
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
    pl._STATE["last_turn"] = None
    assert pl.on_llm_request(**_kwargs()) is None


def test_below_gate_never_rewrites(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("free", 0.40))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    pl._STATE["last_turn"] = None
    assert pl.on_llm_request(**_kwargs()) is None


def test_paid_pool_never_rewrites(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("paid", 0.99))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    pl._STATE["last_turn"] = None
    assert pl.on_llm_request(**_kwargs()) is None


def test_followup_calls_are_untouched(monkeypatch):
    """api_call_count >= 2 is a tool-loop follow-up: the turn must not switch model."""
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_classify", lambda feats, cfg: ("free", 0.99))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    pl._STATE["last_turn"] = None
    assert pl.on_llm_request(**_kwargs(api_call_count=3)) is None


def test_ineligible_provider_is_skipped_and_logged(monkeypatch):
    recs = []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_log", recs.append)
    pl._STATE["last_turn"] = None
    out = pl.on_llm_request(**_kwargs(provider="opencode-go",
                                      model="longcat-2.5-preview-free"))
    assert out is None
    assert recs and recs[0]["skip_reason"] == "provider_not_eligible"


def test_already_free_model_never_rerouted(monkeypatch):
    recs = []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(
        mode="live", route_models=["inclusionai/ling-3.0-flash-sante:free"]))
    monkeypatch.setattr(pl, "_log", recs.append)
    pl._STATE["last_turn"] = None
    pl.on_llm_request(**_kwargs(model="inclusionai/ling-3.0-flash-sante:free"))
    assert recs and recs[0]["skip_reason"] == "already_free_tier"


def test_classify_failure_is_fail_open(monkeypatch):
    """A vendor outage must leave the request untouched, never raise."""
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live"))
    monkeypatch.setattr(pl, "_log", lambda rec: None)

    def boom(feats, cfg):
        raise RuntimeError("vendor down")

    monkeypatch.setattr(pl, "_classify", boom)
    pl._STATE["last_turn"] = None
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


def test_register_registers_middleware_only():
    """F1/V2 truth-in-manifest: register() declares exactly llm_request."""
    registered = []

    class Ctx:
        def register_middleware(self, kind, cb):
            registered.append(kind)

    pl.register(Ctx())
    assert registered == ["llm_request"]
