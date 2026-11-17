"""Offline tests for the kfchow-llm-value-router plugin (no network, no live gateway).

Loads the repo-local plugin module by path and drives on_llm_request directly
with synthetic kwargs, so the routing contract is verified without touching a
real turn. Run with a throwaway HOME so nothing here can touch a real config:

    HOME=$(mktemp -d) PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
"""
import builtins
import importlib.machinery
import importlib.util
import json
import logging
import os
import stat
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
    with pl._SESSION_OVERRIDE_LOCK:
        pl._SESSION_OVERRIDE.clear()


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


def test_log_is_private_at_creation(monkeypatch, tmp_path):
    log = tmp_path / "jev" / "routing.jsonl"
    monkeypatch.setattr(pl, "LOG_PATH", str(log))
    opened = []
    builtin_open, os_open = builtins.open, os.open

    def record(fd):
        opened.append((stat.S_IMODE(os.fstat(fd).st_mode),
                       stat.S_IMODE(os.stat(log.parent).st_mode)))

    def watch_builtin(path, *args, **kwargs):
        fh = builtin_open(path, *args, **kwargs)
        if isinstance(path, (str, os.PathLike)) and os.fspath(path) == str(log):
            record(fh.fileno())
        return fh

    def watch_os_open(path, flags, mode=0o777, **kwargs):
        fd = os_open(path, flags, mode, **kwargs)
        if os.fspath(path) == str(log):
            record(fd)
        return fd

    monkeypatch.setattr(builtins, "open", watch_builtin)
    monkeypatch.setattr(os, "open", watch_os_open)
    old_umask = os.umask(0)
    try:
        pl._log({"task_excerpt": "private"})
    finally:
        os.umask(old_umask)
    assert opened == [(0o600, 0o700)]
    assert json.loads(log.read_text()) == {"task_excerpt": "private"}


@pytest.mark.parametrize("failure", ["chmod", "write"])
def test_log_failure_is_counted_and_rate_limited(monkeypatch, tmp_path, caplog, failure):
    log = tmp_path / "jev" / "routing.jsonl"
    monkeypatch.setattr(pl, "LOG_PATH", str(log))
    monkeypatch.setattr(pl, "_LOG_FAILURES", 0, raising=False)
    monkeypatch.setattr(pl, "_LOG_LAST_WARNING", None, raising=False)
    if failure == "chmod":
        def fail_chmod(*args, **kwargs):
            raise PermissionError("chmod denied")
        monkeypatch.setattr(os, "chmod", fail_chmod)
    else:
        fdopen = os.fdopen

        class FullDisk:
            def __init__(self, fh):
                self.fh = fh

            def __enter__(self):
                self.fh.__enter__()
                return self

            def __exit__(self, *args):
                return self.fh.__exit__(*args)

            def fileno(self):
                return self.fh.fileno()

            def write(self, data):
                raise OSError("disk full")

        monkeypatch.setattr(os, "fdopen", lambda *a, **kw: FullDisk(fdopen(*a, **kw)))

    caplog.set_level(logging.WARNING, logger=pl.logger.name)
    pl._log({"task_excerpt": "SECRET_PROMPT_CANARY"})
    pl._log({"task_excerpt": "SECRET_PROMPT_CANARY"})
    assert pl._LOG_FAILURES == 2
    warnings = [r.getMessage() for r in caplog.records if r.name == pl.logger.name]
    assert len(warnings) == 1 and str(log) in warnings[0]
    assert "SECRET_PROMPT_CANARY" not in warnings[0]
    assert not log.exists() or not log.read_text()


def test_log_failure_falls_back_to_stderr_without_prompt(monkeypatch, tmp_path, capsys):
    log = tmp_path / "routing.jsonl"
    log.mkdir()  # opening a directory as a file fails without special privileges
    monkeypatch.setattr(pl, "LOG_PATH", str(log))
    monkeypatch.setattr(pl, "_LOG_FAILURES", 0)
    monkeypatch.setattr(pl, "_LOG_LAST_WARNING", None)
    monkeypatch.setattr(pl.logger, "warning", lambda *args, **kwargs: None)

    pl._log({"task_excerpt": "SECRET_PROMPT_CANARY"})
    pl._log({"task_excerpt": "SECRET_PROMPT_CANARY"})
    err = capsys.readouterr().err
    assert err.count("routing log failed") == 1  # same rate limit as the warning
    assert str(log) in err
    assert "SECRET_PROMPT_CANARY" not in err


def test_log_falls_back_to_stderr_when_warning_handler_fails(monkeypatch, tmp_path, capsys):
    log = tmp_path / "routing.jsonl"
    log.mkdir()
    monkeypatch.setattr(pl, "LOG_PATH", str(log))
    monkeypatch.setattr(pl, "_LOG_FAILURES", 0)
    monkeypatch.setattr(pl, "_LOG_LAST_WARNING", None)

    class BrokenHandler(logging.Handler):
        def emit(self, record):
            raise OSError("warning handler unavailable")

    monkeypatch.setattr(pl.logger, "handlers", [BrokenHandler()])
    monkeypatch.setattr(pl.logger, "propagate", False)
    pl._log({"task_excerpt": "SECRET_PROMPT_CANARY"})
    err = capsys.readouterr().err
    assert "routing log failed" in err and str(log) in err
    assert "SECRET_PROMPT_CANARY" not in err
    assert pl.get_log_failure_count() == 1


def test_log_failure_counter_has_public_accessor(monkeypatch, tmp_path):
    log = tmp_path / "routing.jsonl"
    log.mkdir()
    monkeypatch.setattr(pl, "LOG_PATH", str(log))
    monkeypatch.setattr(pl, "_LOG_FAILURES", 0)
    monkeypatch.setattr(pl, "_LOG_LAST_WARNING", None)
    accessor = getattr(pl, "get_log_failure_count", None)
    assert callable(accessor)
    assert accessor() == 0

    pl._log({"event": "first"})
    pl._log({"event": "second"})
    assert accessor() == 2


def test_log_fails_open_when_stderr_is_unavailable(monkeypatch, tmp_path):
    log = tmp_path / "routing.jsonl"
    log.mkdir()
    monkeypatch.setattr(pl, "LOG_PATH", str(log))
    monkeypatch.setattr(pl, "_LOG_FAILURES", 0)
    monkeypatch.setattr(pl, "_LOG_LAST_WARNING", None)
    monkeypatch.setattr(pl.logger, "warning", lambda *args, **kwargs: None)

    class BrokenStderr:
        def write(self, text):
            raise OSError("stderr unavailable")

    monkeypatch.setattr(sys, "stderr", BrokenStderr())
    pl._log({"event": "first"})  # audit failure must not fail the turn
    accessor = getattr(pl, "get_log_failure_count", None)
    assert callable(accessor) and accessor() == 1


def test_existing_log_is_not_written_when_hardening_fails(monkeypatch, tmp_path, caplog):
    log = tmp_path / "jev" / "routing.jsonl"
    log.parent.mkdir()
    log.write_text("old row\n")
    log.chmod(0o644)
    monkeypatch.setattr(pl, "LOG_PATH", str(log))
    monkeypatch.setattr(pl, "_LOG_FAILURES", 0)
    monkeypatch.setattr(pl, "_LOG_LAST_WARNING", None)

    def fail_fchmod(*args):
        raise PermissionError("fchmod denied")

    monkeypatch.setattr(os, "fchmod", fail_fchmod)
    caplog.set_level(logging.WARNING, logger=pl.logger.name)
    pl._log({"task_excerpt": "SECRET_PROMPT_CANARY"})
    assert log.read_text() == "old row\n"
    assert pl._LOG_FAILURES == 1
    assert any(str(log) in r.getMessage() for r in caplog.records)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="requires O_NOFOLLOW")
def test_log_refuses_preexisting_symlink(monkeypatch, tmp_path, caplog):
    log = tmp_path / "jev" / "routing.jsonl"
    log.parent.mkdir(mode=0o777)
    victim = tmp_path / "victim"
    victim.write_text("original\n")
    victim.chmod(0o644)
    log.symlink_to(victim)
    monkeypatch.setattr(pl, "LOG_PATH", str(log))
    monkeypatch.setattr(pl, "_LOG_FAILURES", 0)
    monkeypatch.setattr(pl, "_LOG_LAST_WARNING", None)
    caplog.set_level(logging.WARNING, logger=pl.logger.name)
    pl._log({"task_excerpt": "SECRET_PROMPT_CANARY"})
    assert victim.read_text() == "original\n"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644
    assert pl._LOG_FAILURES == 1
    assert any(str(log) in r.getMessage() for r in caplog.records)


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
_MID = "z-ai/glm-5.3"
_LING = "inclusionai/ling-3.0-flash-sante:free"
_GPT = "openai/gpt-5.4"


def _cfg7(**over):
    """Production-shaped ladder: flash [0.35, 0.65), mid [0.65, 0.80), esc 0.80,
    free rung above 0.80 and below 0.35 (v1.0.10 Marc banding)."""
    base = _cfg(mode="live", flash_gate=0.35, flash_model=_FLASH,
                mid_model=_MID, escalate_confidence_gate=0.80)
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

def test_sonnet_default_free_079_moves_down_to_mid(monkeypatch):
    # v1.0.10 Marc banding: [0.65, 0.80) free pool rides the MID rung.
    out, row, probes, _ = _run7(monkeypatch, _SONNET, "free", 0.79, "s7-1")
    assert out is not None and out["request"]["model"] == _MID
    assert row["tier"] == "mid" and row["applied"] is True
    assert probes == []                                   # down-move never probes


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

def test_gpt54_default_free_070_to_mid_rung(monkeypatch):
    out, row, _, _ = _run7(monkeypatch, _GPT, "free", 0.70, "g7-1")
    assert out is not None and out["request"]["model"] == _MID
    assert row["tier"] == "mid"


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


# ---- ling:free default: free is the DEFAULT rung (v1.0.12 route-up) -------
# Marc 2026-11-16: free by default, route up on hard tasks. A free-source turn
# is therefore classified; easy verdicts hold on the free rung, harder ones
# move up (0.35-0.65 -> flash, 0.65-0.80 -> mid, 0.80+ paid -> premium).
# free_default_route_up=false restores the v1.0.11 skip-before-classify
# (covered by test_already_free_model_never_rerouted).

_FREE_ROUTE_UP_EXPECT = {
    ("paid", 0.99): _SONNET, ("paid", 0.85): _SONNET,
    ("free", 0.99): None, ("free", 0.65): _MID, ("free", 0.50): _FLASH,
    ("free", 0.10): None,
}


@pytest.mark.parametrize("mode", ["any", "allowlist", [_LING], "bogus"])
@pytest.mark.parametrize("pool,conf", [("paid", 0.99), ("paid", 0.85), ("free", 0.99),
                                       ("free", 0.65), ("free", 0.50), ("free", 0.10)])
def test_ling_free_default_routes_up_on_hard_verdicts(monkeypatch, mode, pool, conf):
    want = _FREE_ROUTE_UP_EXPECT[(pool, conf)]
    out, row, probes, classified = _run7(monkeypatch, _LING, pool, conf,
                                         "l8-%s-%s-%s" % (str(mode)[:6], pool, conf),
                                         source_mode=mode, route_models=[_LING],
                                         escalate_models=[_LING])
    got = (out["request"]["model"] if out else None)
    assert got == want, (mode, pool, conf, got, want)
    if want is None:
        # holds on the free rung: classified, never moved, never probed
        assert classified == [1] and probes == []
        assert row["skip_reason"] == "already_free_tier"
    elif want == _SONNET:
        assert row["escalate_applied"] is True and probes == [1]
    else:
        assert row["applied"] is True and probes == []


def test_free_default_route_up_paid_low_band_stays_free(monkeypatch):
    # A paid verdict below the escalation gate is NOT a route-up trigger: the
    # v1.0.10 D7 rule (paid pool stays) still holds for a free source.
    out, row, probes, classified = _run7(monkeypatch, _LING, "paid", 0.70, "l8-paid70")
    assert out is None and row["skip_reason"] == "already_free_tier"
    assert classified == [1] and probes == []


def test_free_default_route_up_credit_dead_blocks_escalation(monkeypatch):
    # Route-up to premium is credit-gated like every escalation: a dark paid
    # lane must leave the turn on the free rung, never fail it.
    probes = []
    monkeypatch.setattr(pl, "_load_config",
                        lambda: _cfg7(free_default_route_up=True))
    monkeypatch.setattr(pl, "_classify", lambda f, c: ("paid", 0.99))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: probes.append(1) or False)
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    out = pl.on_llm_request(**_kwargs(model=_LING, turn_id="l8-credit"))
    assert out is None                                  # stays on the free rung
    assert probes == [1]                                # probed once, failed
    assert pl._DECISIONS["s1:l8-credit"]["decision"] == "none"


def test_free_default_route_up_paid_mid_band_does_not_downgrade(monkeypatch):
    # D7 holds for a free source too: a PAID verdict in 0.35-0.65 must NOT be
    # pushed onto the flash rung (flash is a downgrade rung for paid verdicts).
    out, row, probes, classified = _run7(monkeypatch, _LING, "paid", 0.50, "l8-paid50")
    assert out is None and row["skip_reason"] == "already_free_tier"
    assert classified == [1] and probes == []


# ------------------------------------------------- error hook (v1.0.12)

def _err_kwargs(model, status, session="s1", reason="rate_limit"):
    return {"status_code": status, "reason": reason, "session_id": session,
            "turn_id": "e1", "provider": "nous",
            "api_kwargs": {"body": {"model": model, "messages": []}}}


@pytest.mark.parametrize("status", [401, 404, 429])
def test_error_hook_arms_from_host_request_envelope(monkeypatch, status):
    recs = []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7())
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    pl.on_api_request_error(
        status_code=status, reason="provider_error", session_id="host-session",
        turn_id="e1", provider="nous", model=_SONNET,
        request={"method": "POST", "body": {"model": _LING, "messages": []}})
    assert pl._session_override("host-session") == _FLASH
    assert recs[-1]["model"] == _LING
    assert recs[-1]["error_status"] == status


def test_error_hook_falls_back_to_top_level_model(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7())
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    pl.on_api_request_error(status_code=429, session_id="host-session",
                            provider="nous", model=_LING,
                            request={"method": "POST", "body": {}})
    assert pl._session_override("host-session") == _FLASH


@pytest.mark.parametrize("status", [401, 404, 429])
def test_error_hook_arms_next_live_rung(monkeypatch, status):
    # A 401/404/429 from the free rung arms the NEXT rung for the session;
    # the turn must not stop.
    recs = []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7())
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    pl.on_api_request_error(**_err_kwargs(_LING, status))
    assert pl._session_override("s1") == _FLASH
    assert recs[-1]["skip_reason"] == "error_override_armed"
    assert recs[-1]["error_status"] == status


def test_error_hook_non_router_status_is_noop(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7())
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    pl.on_api_request_error(**_err_kwargs(_LING, 500))
    assert pl._session_override("s1") is None


def test_error_hook_skips_empty_rungs(monkeypatch):
    # flash/mid empty -> the override jumps to the next live rung (premium).
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7(flash_model="", mid_model=""))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    pl.on_api_request_error(**_err_kwargs(_LING, 429))
    assert pl._session_override("s1") == _cfg7()["premium_model"]


def test_error_override_rewrites_next_request(monkeypatch):
    recs = []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7())
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    pl.on_api_request_error(**_err_kwargs(_LING, 429))
    kw = _kwargs(model=_LING, turn_id="after-err")
    out = pl.on_llm_request(**kw)
    assert out == {"request": {**kw["request"], "model": _FLASH}}
    assert recs[-1]["skip_reason"] == "error_override_applied"
    assert recs[-1]["applied"] is True          # live mode


def test_armed_error_override_preserves_every_request_field(monkeypatch):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7(mode="live"))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    pl.on_api_request_error(status_code=429, provider="nous", session_id="s1",
                            request={"method": "POST", "body": {"model": _LING}})
    assert pl._session_override("s1") == _FLASH

    messages = [{"role": "user", "content": "find it"},
                {"role": "assistant", "content": None, "tool_calls": []}]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    opaque = object()  # provider kwargs can contain non-JSON client objects
    request = {"model": _LING, "messages": messages, "tools": tools,
               "tool_choice": "auto", "temperature": 0.25, "stream": False,
               "metadata": {"trace": "kept"}, "future_option": opaque}
    kw = _kwargs(model=_LING, turn_id="payload-after-err")
    kw["request"] = request
    out = pl.on_llm_request(**kw)
    assert out == {"request": {**request, "model": _FLASH}}
    assert out["request"] is not request
    assert request["model"] == _LING
    assert out["request"]["messages"] == messages
    assert out["request"]["tools"] == tools
    assert out["request"]["future_option"] is opaque


@pytest.mark.parametrize("mode", ["shadow", "SHADOW", "bogus", None])
def test_armed_error_override_never_rewrites_outside_live_mode(monkeypatch, mode):
    cfg = _cfg7(mode=mode)
    if mode is None:
        cfg.pop("mode")  # absent mode defaults to shadow
    recs = []
    monkeypatch.setattr(pl, "_load_config", lambda: cfg)
    monkeypatch.setattr(pl, "_classify", lambda features, config: ("free", 0.50))
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    pl.on_api_request_error(status_code=429, provider="nous", session_id="s1",
                            request={"method": "POST", "body": {"model": _LING}})
    assert pl._session_override("s1") == _FLASH
    kw = _kwargs(model=_LING, turn_id="shadow-after-err")
    kw["request"]["tools"] = [{"type": "function", "function": {"name": "lookup"}}]
    before = json.loads(json.dumps(kw["request"]))
    assert pl.on_llm_request(**kw) is None
    assert kw["request"] == before
    assert not any(row.get("applied") for row in recs)


def test_error_override_expires(monkeypatch):
    import time as _t
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7())
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    pl.on_api_request_error(**_err_kwargs(_LING, 429))
    assert pl._session_override("s1") == _FLASH
    with pl._SESSION_OVERRIDE_LOCK:
        pl._SESSION_OVERRIDE["s1"]["ts"] -= pl._SESSION_OVERRIDE_TTL_S + 1
    assert pl._session_override("s1") is None


def test_no_rungs_configured_skips_classifier_and_excerpt(monkeypatch):
    # v1.0.12 no-egress guard: with no rung resolved for the provider the
    # classifier must never run (no vendor call, no excerpt leaves the box).
    classified = []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7(
        free_model="", flash_model="", mid_model="", premium_model="",
        rungs={}))
    monkeypatch.setattr(pl, "_classify", lambda f, c: classified.append(1) or ("paid", 0.9))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    _reset_decisions()
    out = pl.on_llm_request(**_kwargs(model=_SONNET, turn_id="l9-norungs"))
    assert out is None
    assert classified == []


def test_free_source_route_up_off_restores_v1011_skip(monkeypatch):
    out, row, probes, classified = _run7(monkeypatch, _LING, "paid", 0.99, "l8-off",
                                         free_default_route_up=False)
    assert out is None and row["skip_reason"] == "already_free_tier"
    assert classified == [] and probes == []              # never even classified


# ---- source_mode narrowing --------------------------------------------------

def test_explicit_list_source_mode_narrows_to_listed_sources(monkeypatch):
    # Listed source behaves exactly like "any" (v1.0.10: 0.65 free-pool = MID) ...
    out, _, _, _ = _run7(monkeypatch, _SONNET, "free", 0.65, "x7-1",
                         source_mode=[_SONNET])
    assert out is not None and out["request"]["model"] == _MID
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
    # listed mid default (escalate_models): escalates
    out, _, _, _ = _run7(monkeypatch, "z-ai/glm-5.3", "paid", 0.90, "a7-3",
                         source_mode="allowlist")
    assert out is not None and out["request"]["model"] == _SONNET
    # DOWN-moves additionally require route-lane eligibility: the deployed bands
    # are gated on `free_src` (v1.0.11 tightened routing scope), so a mid default
    # listed only in escalate_models stays put instead of dropping to flash.
    out, row, _, _ = _run7(monkeypatch, "z-ai/glm-5.3", "free", 0.50, "a7-4",
                           source_mode="allowlist")
    assert row is not None and row["skip_reason"] == "below_gate"
    # ...and once it IS listed for routing too, the flash band applies.
    out, _, _, _ = _run7(monkeypatch, "z-ai/glm-5.3", "free", 0.50, "a7-5",
                         source_mode="allowlist",
                         route_models=["z-ai/glm-5.3", "glm-5.3"])
    assert out is not None and out["request"]["model"] == _FLASH


def test_unset_null_or_empty_source_mode_means_any(monkeypatch):
    for i, mode in enumerate((None, "", "ANY", " any ")):
        out, _, _, _ = _run7(monkeypatch, _SONNET, "free", 0.90, "n7-%d" % i,
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
    out, _, _, _ = _run7(monkeypatch, _GPT, "free", 0.70, "e7-2",
                         escalate_enabled=False)
    assert out is not None and out["request"]["model"] == _MID


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
    """(wire model or None, skip_reason) for a FREE-pool verdict (v1.0.10)."""
    if conf >= 0.80:
        return _LING, None
    if conf >= 0.65:
        return (None, "already_on_target") if source == _MID else (_MID, None)
    if conf >= 0.35:
        return (None, "already_on_target") if source == _FLASH else (_FLASH, None)
    return _LING, None        # routine band: rides the free rung, any source


def _expect_paid_pool(source, conf):
    """(wire model or None, skip_reason) for a PAID-pool verdict (esc gate 0.80).

    D7 (Marc 2026-11-03): the routine band is FREE-POOL ONLY by default, so every
    paid-pool verdict below the escalation gate stays on its source model
    (`pool_paid`). The v1.0.10 pool-agnostic behaviour is opt-in via
    `paid_pool_routine_band: true` (covered by its own test).
    """
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


def test_routine_band_free_pool_rides_free_paid_pool_stays(monkeypatch):
    """D7 (Marc 2026-11-03): the routine band is free-pool only by default.

    Free-pool turns below the flash gate ride the free rung from any source;
    paid-pool turns stay on their source model (no downgrade) unless the
    operator opts back into the v1.0.10 pool-agnostic behaviour.
    """
    for i, src in enumerate((_SONNET, _GPT)):
        out, row, probes, _ = _run7(monkeypatch, src, "free", 0.20,
                                    "r7-low-f%d" % i)
        assert row is not None and row["tier"] == "free" and row["applied"] is True
        assert out is not None and out["request"]["model"] == _LING
        assert "routine band" in out["reason"]
        assert probes == []
    for i, src in enumerate((_SONNET, _FLASH)):
        out, row, probes, _ = _run7(monkeypatch, src, "paid", 0.20,
                                    "r7-low-p%d" % i)
        assert row is not None and row["skip_reason"] == "pool_paid"
        assert row["tier"] == "stay" and row["applied"] is False
        assert out is None and probes == []


def test_routine_band_paid_pool_opt_in_restores_v1010_behaviour(monkeypatch):
    """`paid_pool_routine_band: true` restores the v1.0.10 pool-agnostic band."""
    for i, src in enumerate((_SONNET, _FLASH)):
        out, row, probes, _ = _run7(monkeypatch, src, "paid", 0.20,
                                    "r7-low-opt%d" % i,
                                    paid_pool_routine_band=True)
        assert row is not None and row["tier"] == "free" and row["applied"] is True
        assert out is not None and out["request"]["model"] == _LING
        assert "routine band" in out["reason"]
        assert probes == []


def test_routine_band_stays_when_source_is_free_model(monkeypatch):
    # v1.0.12: a free-source turn is CLASSIFIED, but an easy verdict (free
    # pool, below the flash gate) holds on the free rung. Never moved down.
    out, row, _, classified = _run7(monkeypatch, _LING, "free", 0.20, "r7-low-s")
    assert out is None and row["skip_reason"] == "already_free_tier"
    assert classified == [1]


def test_reapply_sonnet_default_mid_downmove_via_request_and_exec(monkeypatch):
    classified, recs, probes = [], [], []
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg7())
    monkeypatch.setattr(pl, "_classify",
                        lambda f, c: classified.append(1) or ("free", 0.70))
    monkeypatch.setattr(pl, "_paid_lane_alive", lambda: probes.append(1) or True)
    monkeypatch.setattr(pl, "_log", recs.append)
    _reset_decisions()
    first = pl.on_llm_request(**_kwargs(model=_SONNET, turn_id="rm-1"))
    assert first["request"]["model"] == _MID
    second = pl.on_llm_request(**_kwargs(model=_SONNET, turn_id="rm-1",
                                         api_call_count=2))
    assert second is not None and second["request"]["model"] == _MID
    req, ctx = _exec7(_SONNET, "rm-1")
    seen = []
    out = pl.on_llm_execution(request=req, next_call=_next7(seen), **ctx)
    assert req["model"] == _SONNET and seen[0]["model"] == _MID
    assert out["model"] == _MID
    assert len(classified) == 1 and probes == []
    assert [r["tier"] for r in recs if r.get("reapply")] == ["mid", "mid"]


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
