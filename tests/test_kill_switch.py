"""v1.0.8 — per-request router kill switch (X-KFC-Router) regressions.

A client can disable routing for its OWN request with no config edit and zero
blast radius on other turns. Two interchangeable forms ride on the provider
payload (the only per-request channel the middleware layer sees):

    request["extra_headers"]["X-KFC-Router"] = "off"   # HTTP-header form
    request["metadata"]["router"] = "off"              # JSON-body form
    request["router"] = "off"                          # or top-level body

Disabling values: off / 0 / false / disable / disabled (name+value
case-insensitive). Behaviour when set: no rewrite, NO classifier call (a
counted double proves it), no credit probe, ONE log row with skip_reason
"disabled_by_request" and applied=false, and NO decision stored — so a
later callback of the turn finds nothing to re-apply (the reapply path must
honour the same flag or the rewrite reappears mid-turn). Unknown values are
ignored (fail-open, normal routing, no warning spam). The flag only turns
routing OFF; the plugin-level switch stays the config `enabled` key.

All offline: _classify and _paid_lane_alive are counted doubles, _log is
captured in memory, no network. The plugin module is loaded by path.
"""
import importlib.machinery
import importlib.util
import logging
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLUGIN = os.path.join(_REPO, "__init__.py")
_loader = importlib.machinery.SourceFileLoader("kfchow_llm_value_router", _PLUGIN)
_spec = importlib.util.spec_from_loader("kfchow_llm_value_router", _loader)
pl = importlib.util.module_from_spec(_spec)
sys.modules["kfchow_llm_value_router"] = pl
_loader.exec_module(pl)


def _reset():
    with pl._DECISIONS_LOCK:
        pl._DECISIONS.clear()


def _cfg(**over):
    base = {"enabled": True, "mode": "live", "confidence_gate": 0.65,
            "route_providers": ["nous"],
            "route_models": ["z-ai/glm-5.3-flash", "glm-5.3-flash"],
            "free_model": "inclusionai/ling-3.0-flash-sante:free",
            "flash_gate": 0.35,
            "flash_model": "z-ai/glm-5.3-flash",
            "escalate_enabled": True, "escalate_providers": ["nous"],
            "escalate_models": ["z-ai/glm-5.3", "glm-5.3"],
            "premium_model": "anthropic/claude-sonnet-5.5",
            "escalate_confidence_gate": 0.80, "credit_probe": True,
            "send_excerpt": True}
    base.update(over)
    return base


class _Classifier:
    """Counted _classify double: any call on a killed turn is a bug."""

    def __init__(self, answer=("free", 0.90)):
        self.answer = answer
        self.calls = 0

    def __call__(self, feats, cfg):
        self.calls += 1
        return self.answer


class _Probe:
    """Counted _paid_lane_alive double."""

    def __init__(self, alive=True):
        self.alive = alive
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.alive


def _kw(turn_id="k1", model="z-ai/glm-5.3-flash", api_call_count=1,
        provider="nous", messages=None):
    """Request kwargs; request payload gets the flag the caller sets."""
    return {
        "provider": provider, "model": model,
        "api_call_count": api_call_count,
        "turn_id": turn_id, "session_id": "s1",
        "request": {"model": model,
                    "messages": messages or [{"role": "user", "content": "hi"}]},
    }


def _exec(kw, api_call_count=2):
    """(request, context) as the host execution chain passes them."""
    return (dict(kw["request"]),
            {"provider": kw["provider"], "model": kw["model"],
             "turn_id": kw["turn_id"], "session_id": kw["session_id"],
             "api_call_count": api_call_count})


def _next(seen):
    def _n(request):
        seen.append(dict(request))
        return {"model": (request or {}).get("model")}
    return _n


def _install(monkeypatch, recs, cls, probe, **over):
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(**over))
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", probe)
    monkeypatch.setattr(pl, "_log", recs.append)


def _hdr(kw, name="X-KFC-Router", value="off"):
    """Set the HTTP-header form of the flag on a _kw() dict."""
    kw["request"]["extra_headers"] = {name: value}
    return kw


def _body(kw, value="off", where="metadata"):
    """Set the JSON-body form of the flag on a _kw() dict."""
    if where == "metadata":
        kw["request"]["metadata"] = {"router": value}
    else:
        kw["request"]["router"] = value
    return kw


# 1. header off: no rewrite, NO classify call, NO probe, one disabled row
def test_header_off_no_classify_no_probe_row(monkeypatch):
    _reset()
    recs, cls, probe = [], _Classifier(), _Probe()
    _install(monkeypatch, recs, cls, probe)
    kw = _hdr(_kw(turn_id="ks-1"))
    assert pl.on_llm_request(**kw) is None
    assert cls.calls == 0            # spy: the vendor was never consulted
    assert probe.calls == 0
    assert len(recs) == 1
    row = recs[0]
    assert row["skip_reason"] == "disabled_by_request"
    assert row["applied"] is False
    assert row["eligible"] is False
    assert row["pool"] is None and row["confidence"] is None
    # nothing stored: the turn stays unknown, so nothing can re-apply
    with pl._DECISIONS_LOCK:
        assert pl._DECISIONS == {}


# 2. header off mid-turn on the reapply path: a SECOND call carrying the flag
#    must not re-apply anything (and never would: nothing was stored)
def test_header_off_midturn_reapply_path_clean(monkeypatch):
    _reset()
    recs, cls, probe = [], _Classifier(), _Probe()
    _install(monkeypatch, recs, cls, probe)
    assert pl.on_llm_request(**_hdr(_kw(turn_id="ks-2", api_call_count=1))) is None
    seen = []
    req, ctx = _exec(_hdr(_kw(turn_id="ks-2", api_call_count=2)), api_call_count=2)
    out = pl.on_llm_execution(request=req, next_call=_next(seen), **ctx)
    assert out["model"] == "z-ai/glm-5.3-flash"     # original model downstream
    assert seen[0] == req                            # byte-identical payload
    assert cls.calls == 0 and probe.calls == 0
    assert [r for r in recs if r.get("reapply")] == []
    with pl._DECISIONS_LOCK:
        assert pl._DECISIONS == {}                   # reapply finds no decision


# 3. header absent: normal routing (regression — the flag changes nothing)
def test_header_absent_normal_routing(monkeypatch):
    _reset()
    recs, cls, probe = [], _Classifier(), _Probe()
    _install(monkeypatch, recs, cls, probe)
    out = pl.on_llm_request(**_kw(turn_id="ks-3"))
    assert out is not None
    assert out["request"]["model"] == "inclusionai/ling-3.0-flash-sante:free"
    assert cls.calls == 1 and probe.calls == 0
    assert recs[-1]["skip_reason"] is None
    with pl._DECISIONS_LOCK:
        assert "s1:ks-3" in pl._DECISIONS              # decision stored as usual


# 4. on-values ("on"/"1"/"true"): the flag only turns OFF — normal routing
def test_on_values_route_normally(monkeypatch):
    for i, value in enumerate(("on", "1", "true", "ON", "True")):
        _reset()
        recs, cls, probe = [], _Classifier(), _Probe()
        _install(monkeypatch, recs, cls, probe)
        out = pl.on_llm_request(**_hdr(_kw(turn_id="ks-4-%d" % i), value=value))
        assert out is not None, value
        assert out["request"]["model"] == "inclusionai/ling-3.0-flash-sante:free"
        assert cls.calls == 1
        assert recs[-1]["skip_reason"] is None, value
        with pl._DECISIONS_LOCK:
            assert "s1:ks-4-%d" % i in pl._DECISIONS


# 5. unknown value ("banana"): ignored, normal routing, NO warning spam
def test_unknown_value_ignored_no_warning(monkeypatch, caplog):
    _reset()
    recs, cls, probe = [], _Classifier(), _Probe()
    with caplog.at_level(logging.WARNING, logger=pl.logger.name):
        _install(monkeypatch, recs, cls, probe)
        out = pl.on_llm_request(**_hdr(_kw(turn_id="ks-5"), value="banana"))
        assert out is not None
        assert out["request"]["model"] == "inclusionai/ling-3.0-flash-sante:free"
        assert cls.calls == 1
        assert recs[-1]["skip_reason"] is None
        # a second turn with the SAME junk value: still no warning
        out2 = pl.on_llm_request(**_hdr(_kw(turn_id="ks-5b"), value="banana"))
        assert out2 is not None
    assert [r for r in caplog.records if "kfc-router" in r.getMessage().lower()
            or "kill switch" in r.getMessage().lower()] == []


# 6. JSON-body form: metadata["router"] = "off" behaves exactly like the header
def test_metadata_body_form_off(monkeypatch):
    _reset()
    recs, cls, probe = [], _Classifier(), _Probe()
    _install(monkeypatch, recs, cls, probe)
    assert pl.on_llm_request(**_body(_kw(turn_id="ks-6"))) is None
    assert cls.calls == 0 and probe.calls == 0
    assert recs[-1]["skip_reason"] == "disabled_by_request"
    assert recs[-1]["applied"] is False
    with pl._DECISIONS_LOCK:
        assert pl._DECISIONS == {}


# 6b. top-level body form request["router"] = "off" (some clients can't nest)
def test_top_level_body_form_off(monkeypatch):
    _reset()
    recs, cls, probe = [], _Classifier(), _Probe()
    _install(monkeypatch, recs, cls, probe)
    assert pl.on_llm_request(**_body(_kw(turn_id="ks-6b"), where="top")) is None
    assert cls.calls == 0 and probe.calls == 0
    assert recs[-1]["skip_reason"] == "disabled_by_request"


# 7. case-insensitivity: header NAME and VALUE both honoured in any casing
def test_case_insensitive_name_and_value(monkeypatch):
    cases = [("x-kfc-router", "off"), ("X-KFC-ROUTER", "OFF"),
             ("X-Kfc-Router", "Off"), ("X-KFC-ROUTER", "0"),
             ("x-KFC-router", "FALSE"), ("x-kfc-router", "disabled"),
             ("x-kfc-router", "Disable")]
    for i, (name, value) in enumerate(cases):
        _reset()
        recs, cls, probe = [], _Classifier(), _Probe()
        _install(monkeypatch, recs, cls, probe)
        out = pl.on_llm_request(**_hdr(_kw(turn_id="ks-7-%d" % i),
                                       name=name, value=value))
        assert out is None, (name, value)
        assert cls.calls == 0, (name, value)
        assert recs[-1]["skip_reason"] == "disabled_by_request", (name, value)
        with pl._DECISIONS_LOCK:
            assert pl._DECISIONS == {}, (name, value)


# 8. flag + shadow mode: still no rewrite, row still logged
def test_flag_in_shadow_mode_rows_logged(monkeypatch):
    _reset()
    recs, cls, probe = [], _Classifier(), _Probe()
    _install(monkeypatch, recs, cls, probe, mode="shadow")
    assert pl.on_llm_request(**_hdr(_kw(turn_id="ks-8"))) is None
    assert cls.calls == 0 and probe.calls == 0
    assert len(recs) == 1
    row = recs[0]
    assert row["skip_reason"] == "disabled_by_request"
    assert row["applied"] is False
    assert row["mode"] == "shadow"
    with pl._DECISIONS_LOCK:
        assert pl._DECISIONS == {}


# 9. flag + a turn that WOULD escalate (paid 0.99): no rewrite, no credit
#    probe spent — the kill switch beats the escalation path
def test_flag_beats_escalation_no_probe_spent(monkeypatch):
    _reset()
    recs, cls, probe = [], _Classifier(("paid", 0.99)), _Probe()
    _install(monkeypatch, recs, cls, probe)
    kw = _hdr(_kw(turn_id="ks-9", model="z-ai/glm-5.3"))
    assert pl.on_llm_request(**kw) is None
    assert cls.calls == 0 and probe.calls == 0      # probe never spent
    assert recs[-1]["skip_reason"] == "disabled_by_request"
    with pl._DECISIONS_LOCK:
        assert pl._DECISIONS == {}


# 10. the mid-turn scenario end to end: call 1 WITHOUT the flag routes to free
#     (a decision is stored); call 2 carries the flag — the reapply path must
#     refuse to re-apply it (on BOTH the request and the execution path).
def test_flag_midturn_blocks_reapply_of_stored_decision(monkeypatch):
    _reset()
    recs, cls, probe = [], _Classifier(), _Probe()
    _install(monkeypatch, recs, cls, probe)
    # call 1: no flag — normal routing, free tier, decision stored
    first = pl.on_llm_request(**_kw(turn_id="ks-10", api_call_count=1))
    assert first is not None
    assert first["request"]["model"] == "inclusionai/ling-3.0-flash-sante:free"
    with pl._DECISIONS_LOCK:
        assert pl._DECISIONS["s1:ks-10"]["decision"] == "free"
    # call 2 on the REQUEST reapply path with the flag: no reapply
    out2 = pl.on_llm_request(**_hdr(_kw(turn_id="ks-10", api_call_count=2)))
    assert out2 is None
    # call 2 on the EXECUTION reapply path with the flag: original goes out
    seen = []
    req, ctx = _exec(_hdr(_kw(turn_id="ks-10", api_call_count=2)),
                     api_call_count=2)
    out_exec = pl.on_llm_execution(request=req, next_call=_next(seen), **ctx)
    assert out_exec["model"] == "z-ai/glm-5.3-flash"  # NOT the free rung
    assert seen[0] == req
    assert cls.calls == 1 and probe.calls == 0
    disabled = [r for r in recs if r["skip_reason"] == "disabled_by_request"]
    assert len(disabled) == 1  # exactly one disabled row (from call 2)
    assert disabled[0]["applied"] is False


# 11. helper-level contract: the flag reader itself (values, junk, shapes)
def test_kill_switch_helper_contract():
    assert pl._kill_switch_off({"extra_headers": {"X-KFC-Router": "off"}})
    assert pl._kill_switch_off({"extra_headers": {"x-kfc-router": "0"}})
    assert pl._kill_switch_off({"metadata": {"router": "off"}})
    assert pl._kill_switch_off({"router": "off"})
    # surrounding whitespace tolerated
    assert pl._kill_switch_off({"extra_headers": {" X-KFC-Router ": " off "}})
    # non-disabling values and junk: not off
    for req in ({"extra_headers": {"X-KFC-Router": "on"}},
                {"extra_headers": {"X-KFC-Router": "banana"}},
                {"extra_headers": {"X-Other-Router": "off"}},
                {"metadata": {"router": "1"}},
                {"metadata": {"other": "off"}},
                {"router": "true"},
                {},
                None,
                "not-a-dict",
                {"extra_headers": "not-a-dict"},
                {"metadata": None},
                {"router": None}):
        assert pl._kill_switch_off(req) is False, req
    # the documented constants are wired to the helper
    for value in pl.KILL_SWITCH_OFF_VALUES:
        assert pl._kill_switch_off({"extra_headers": {pl.KILL_SWITCH_HEADER: value}})


# 12. plugin-level config switch and the per-request flag are independent:
#     enabled:false wins without any flag, and the flag cannot ENABLE a
#     config-disabled plugin (it only turns routing OFF).
def test_config_disabled_still_wins_and_flag_cannot_enable(monkeypatch):
    _reset()
    recs, cls, probe = [], _Classifier(), _Probe()
    _install(monkeypatch, recs, cls, probe, enabled=False)
    # config disabled, no flag: plain None, no row (enabled gate is upstream)
    assert pl.on_llm_request(**_kw(turn_id="ks-12a")) is None
    assert recs == []
    # config disabled WITH a flag: still disabled, still no row (the flag is
    # never even consulted — it cannot re-enable anything)
    assert pl.on_llm_request(**_hdr(_kw(turn_id="ks-12b"), value="on")) is None
    assert recs == []
    assert cls.calls == 0
