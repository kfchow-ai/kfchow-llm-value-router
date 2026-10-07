"""P1 turn-persistence regressions: one classify per turn, decision re-applied on
every later middleware callback (retries + tool-loop follow-ups), stable under
interleaving, bounded/TTL'd store, fail-closed re-escalation, additive log keys.

All offline: _classify and _paid_lane_alive are monkeypatched, _log is captured
in memory, no network, no vendor calls. The plugin module is loaded by path so
no Hermes install is needed.
"""
import importlib.machinery
import importlib.util
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
            "escalate_enabled": True, "escalate_providers": ["nous"],
            "escalate_models": ["z-ai/glm-5.3", "glm-5.3"],
            "premium_model": "anthropic/claude-sonnet-5.5",
            "escalate_confidence_gate": 0.65, "credit_probe": True,
            "send_excerpt": True}
    base.update(over)
    return base


class _Classifier:
    """Counted _classify double: returns queued (pool, conf) per call."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, feats, cfg):
        self.calls.append(feats)
        if not self.answers:
            raise AssertionError("classify called more times than the test allows")
        return self.answers.pop(0)


class _Credit:
    """Counted _paid_lane_alive double with a switchable verdict."""

    def __init__(self, alive=True):
        self.alive = alive
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.alive


def _kw(turn_id="t1", session_id="s1", api_call_count=1, provider="nous",
        model="z-ai/glm-5.3-flash", messages=None):
    return {
        "provider": provider, "model": model,
        "api_call_count": api_call_count,
        "turn_id": turn_id, "session_id": session_id,
        "request": {"model": model,
                    "messages": messages or [{"role": "user", "content": "hi"}]},
    }


# 1. downgrade + tool follow-up: call 2 returns the same rewrite, classify once
def test_downgrade_followup_reapplies_same_rewrite(monkeypatch):
    _reset()
    recs = []
    cls = _Classifier([("free", 0.90)])
    credit = _Credit(True)
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", credit)
    monkeypatch.setattr(pl, "_log", recs.append)

    first = pl.on_llm_request(**_kw(turn_id="dg-1", api_call_count=1))
    assert first is not None
    assert first["request"]["model"] == "inclusionai/ling-3.0-flash-sante:free"

    second = pl.on_llm_request(**_kw(turn_id="dg-1", api_call_count=2))
    assert second is not None
    assert second["request"]["model"] == "inclusionai/ling-3.0-flash-sante:free"
    assert len(cls.calls) == 1  # classify exactly once


# 2. repeated first-attempt callback (same turn_id, api_call_count=1 twice)
def test_repeated_first_attempt_same_rewrite(monkeypatch):
    _reset()
    recs = []
    cls = _Classifier([("free", 0.90)])
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", _Credit(True))
    monkeypatch.setattr(pl, "_log", recs.append)

    kw = _kw(turn_id="rep-1", api_call_count=1)
    out1 = pl.on_llm_request(**kw)
    out2 = pl.on_llm_request(**dict(kw))  # identical kwargs: in-attempt retry
    assert out1 is not None and out2 is not None
    assert out1["request"]["model"] == out2["request"]["model"]
    assert len(cls.calls) == 1


# 3. escalation + follow-up: premium persists (credit mock alive)
def test_escalation_followup_premium_persists(monkeypatch):
    _reset()
    recs = []
    cls = _Classifier([("paid", 0.95)])
    credit = _Credit(True)
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", credit)
    monkeypatch.setattr(pl, "_log", recs.append)

    kw = _kw(turn_id="esc-f1", model="z-ai/glm-5.3", api_call_count=1)
    first = pl.on_llm_request(**kw)
    assert first is not None
    assert first["request"]["model"] == "anthropic/claude-sonnet-5.5"

    follow = pl.on_llm_request(**_kw(turn_id="esc-f1", model="z-ai/glm-5.3",
                                     api_call_count=2))
    assert follow is not None
    assert follow["request"]["model"] == "anthropic/claude-sonnet-5.5"
    assert len(cls.calls) == 1
    assert credit.calls >= 1  # re-checked (TTL-cached, cheap)


# 4. interleaving A-first, B-first, A-retry: decisions stable, classify total == 2
def test_interleaving_decisions_stable(monkeypatch):
    _reset()
    recs = []
    cls = _Classifier([("free", 0.95), ("free", 0.95)])
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", _Credit(True))
    monkeypatch.setattr(pl, "_log", recs.append)

    a_first = pl.on_llm_request(**_kw(turn_id="turnA", api_call_count=1))
    b_first = pl.on_llm_request(**_kw(turn_id="turnB", api_call_count=1))
    a_retry = pl.on_llm_request(**_kw(turn_id="turnA", api_call_count=2))
    assert a_first is not None and b_first is not None and a_retry is not None
    assert a_retry["request"]["model"] == a_first["request"]["model"]
    assert b_first["request"]["model"] == a_first["request"]["model"]
    assert len(cls.calls) == 2  # A classified once, B classified once, A NOT re-classified


# 5. shadow: follow-ups never rewritten, no extra log rows beyond the classify row
def test_shadow_followups_never_rewritten_no_extra_rows(monkeypatch):
    _reset()
    recs = []
    cls = _Classifier([("free", 0.95)])
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="shadow"))
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", _Credit(True))
    monkeypatch.setattr(pl, "_log", recs.append)

    first = pl.on_llm_request(**_kw(turn_id="sh-1", api_call_count=1))
    assert first is None  # shadow never rewrites
    for n in (2, 3, 4):
        assert pl.on_llm_request(**_kw(turn_id="sh-1", api_call_count=n)) is None
    assert len(cls.calls) == 1
    assert len(recs) == 1  # exactly one classify row; follow-ups add none


# 6. store bound: insert 300 turns -> <=256 retained, oldest evicted, no error
def test_store_bound_300_turns(monkeypatch):
    _reset()
    recs = []
    answers = [("free", 0.95)] * 300
    cls = _Classifier(answers)
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", _Credit(True))
    monkeypatch.setattr(pl, "_log", recs.append)

    for n in range(300):
        out = pl.on_llm_request(**_kw(turn_id=f"bound-{n}", api_call_count=1))
        assert out is not None  # no error path hit
    with pl._DECISIONS_LOCK:
        size = len(pl._DECISIONS)
        keys = list(pl._DECISIONS.keys())
    assert size <= pl._DECISIONS_MAX
    assert size == 256
    assert f"s1:bound-299" in keys        # newest kept
    assert f"s1:bound-0" not in keys      # oldest evicted
    assert f"s1:bound-43" not in keys     # earlier-than-44 evicted


# 7. TTL expiry: re-apply after TTL -> no rewrite (fail-open), no crash
def test_ttl_expiry_behaves_as_unknown(monkeypatch):
    _reset()
    recs = []
    cls = _Classifier([("free", 0.95)])
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", _Credit(True))
    monkeypatch.setattr(pl, "_log", recs.append)

    first = pl.on_llm_request(**_kw(turn_id="ttl-1", api_call_count=1))
    assert first is not None
    # Age the stored decision past the TTL.
    with pl._DECISIONS_LOCK:
        pl._DECISIONS["s1:ttl-1"]["ts"] -= pl._DECISIONS_TTL_S + 1
    follow = pl.on_llm_request(**_kw(turn_id="ttl-1", api_call_count=2,
                                     messages=[{"role": "user", "content": "follow-up"}]))
    # Expired => unknown => follow-up gate (api_call_count=2) => silent None.
    assert follow is None


# 8. provider change mid-turn -> no rewrite
def test_provider_change_no_rewrite(monkeypatch):
    _reset()
    recs = []
    cls = _Classifier([("free", 0.95)])
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", _Credit(True))
    monkeypatch.setattr(pl, "_log", recs.append)

    first = pl.on_llm_request(**_kw(turn_id="prov-1", api_call_count=1,
                                    provider="nous"))
    assert first is not None
    follow = pl.on_llm_request(**_kw(turn_id="prov-1", api_call_count=2,
                                     provider="opencode-go"))
    assert follow is None
    assert len(cls.calls) == 1  # no re-classification on the provider switch


# 9. escalation follow-up with credit gone dead after classify -> no rewrite,
#    credit_lost row
def test_escalation_followup_credit_lost(monkeypatch):
    _reset()
    recs = []
    cls = _Classifier([("paid", 0.95)])
    credit = _Credit(True)
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", credit)
    monkeypatch.setattr(pl, "_log", recs.append)

    kw = _kw(turn_id="cl-1", model="z-ai/glm-5.3", api_call_count=1)
    first = pl.on_llm_request(**kw)
    assert first is not None
    assert first["request"]["model"] == "anthropic/claude-sonnet-5.5"

    credit.alive = False  # lane goes dark mid-turn
    follow = pl.on_llm_request(**_kw(turn_id="cl-1", model="z-ai/glm-5.3",
                                     api_call_count=2))
    assert follow is None
    assert len(cls.calls) == 1
    credit_rows = [r for r in recs if r.get("credit_lost")]
    assert credit_rows, "credit_lost row missing"
    row = credit_rows[-1]
    assert row["applied"] is False
    assert row["turn_id"] == "cl-1"


# 10. reapply rows carry reapply:true and no features key
def test_reapply_row_shape(monkeypatch):
    _reset()
    recs = []
    cls = _Classifier([("free", 0.90)])
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", _Credit(True))
    monkeypatch.setattr(pl, "_log", recs.append)

    pl.on_llm_request(**_kw(turn_id="rr-1", api_call_count=1))
    pl.on_llm_request(**_kw(turn_id="rr-1", api_call_count=2))
    reapply_rows = [r for r in recs if r.get("reapply")]
    assert len(reapply_rows) == 1
    row = reapply_rows[0]
    assert row["applied"] is True
    assert row["api_call_count"] == 2
    assert row["target"] == "inclusionai/ling-3.0-flash-sante:free"
    assert "features" not in row
    # and the first (classify) row keeps its existing schema untouched
    first_rows = [r for r in recs if not r.get("reapply")]
    assert first_rows and "features" in first_rows[0]


# 11. already-free source logs already_free_tier (ordering fix)
def test_already_free_logged_before_eligibility(monkeypatch):
    _reset()
    recs = []
    # Model NOT in route_models AND not on an eligible provider lane, but
    # carries a free hint: the free-hint check must win.
    cls = _Classifier([])
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="live",
                                                         route_models=["z-ai/glm-5.3-flash"]))
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_log", recs.append)

    out = pl.on_llm_request(**_kw(turn_id="af-1", model="some-other-model:free",
                                  provider="nous"))
    assert out is None
    assert recs and recs[0]["skip_reason"] == "already_free_tier"
    assert len(cls.calls) == 0  # never classified


# ===================== v1.0.5 — llm_execution path (12-17) =====================
# The real host runs llm_request ONCE per turn (first attempt); tool-loop
# follow-ups go straight to the execution chain, which runs on EVERY attempt.
# on_llm_execution re-applies the stored decision there; it never classifies.

def _exec_ctx(kw, api_call_count=2):
    """Split a _kw() dict into (request, context) as the host execution chain
    passes them: request kwarg + context kwargs (model = agent.model, i.e. the
    PRE-rewrite configured model)."""
    return (dict(kw["request"]),
            {"provider": kw["provider"], "model": kw["model"],
             "turn_id": kw["turn_id"], "session_id": kw["session_id"],
             "api_call_count": api_call_count})


def _capture_next(seen):
    """A next_call double: records the payload it was handed, returns a
    sentinel response. Raises on a second invocation, mirroring the host's
    single-use guard."""
    state = {"n": 0}

    def _next(request):
        state["n"] += 1
        assert state["n"] == 1, "next_call invoked more than once"
        seen.append(dict(request))
        return {"ok": True, "model": (request or {}).get("model")}
    return _next


# 12. THE WIRE REPRO: llm_request rewrites attempt1; the host does NOT re-run
#     llm_request for the tool follow-up; attempt2 (api_call_count=2) arrives
#     at on_llm_execution with the ORIGINAL (reverted) payload — and must go
#     out on the wire with the routed model.
def test_exec_attempt2_tool_followup_keeps_routed_model(monkeypatch):
    _reset()
    seen = []
    recs = []
    cls = _Classifier([("free", 0.90)])
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", _Credit(True))
    monkeypatch.setattr(pl, "_log", recs.append)

    # Attempt 1: the request middleware classifies + rewrites.
    first = pl.on_llm_request(**_kw(turn_id="wire-1", api_call_count=1))
    assert first is not None
    assert first["request"]["model"] == "inclusionai/ling-3.0-flash-sante:free"

    # Attempt 2 (tool follow-up): payload rebuilt by the host from agent.model,
    # llm_request NOT re-invoked — execution middleware is the last line.
    req, ctx = _exec_ctx(_kw(turn_id="wire-1", api_call_count=2))
    assert req["model"] == "z-ai/glm-5.3-flash"  # reverted, as on the wire
    out = pl.on_llm_execution(request=req, next_call=_capture_next(seen), **ctx)
    assert out == {"ok": True, "model": "inclusionai/ling-3.0-flash-sante:free"}
    assert seen and seen[0]["model"] == "inclusionai/ling-3.0-flash-sante:free"
    assert len(cls.calls) == 1  # still classified exactly once
    exec_rows = [r for r in recs if r.get("reapply") and r.get("api_call_count") == 2]
    assert exec_rows and exec_rows[0]["target"] == "inclusionai/ling-3.0-flash-sante:free"
    assert "features" not in exec_rows[0]


# 13. exec re-apply on attempt1 is idempotent: the host also runs the
#     execution chain on the first attempt, whose payload llm_request already
#     rewrote — setting the same model twice must be a no-op on the wire.
def test_exec_attempt1_reapply_idempotent(monkeypatch):
    _reset()
    seen = []
    recs = []
    cls = _Classifier([("free", 0.90)])
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", _Credit(True))
    monkeypatch.setattr(pl, "_log", recs.append)

    rewritten = pl.on_llm_request(**_kw(turn_id="idem-1", api_call_count=1))
    assert rewritten is not None
    req, ctx = _exec_ctx(_kw(turn_id="idem-1"), api_call_count=1)
    req["model"] = rewritten["request"]["model"]  # host already applied the rewrite
    out = pl.on_llm_execution(request=req, next_call=_capture_next(seen), **ctx)
    assert out["model"] == "inclusionai/ling-3.0-flash-sante:free"
    assert seen[0]["model"] == "inclusionai/ling-3.0-flash-sante:free"
    assert len(cls.calls) == 1
    # attempt1 carries the expected exec reapply row with api_call_count == 1
    a1_rows = [r for r in recs if r.get("reapply") and r.get("api_call_count") == 1]
    assert a1_rows and a1_rows[0]["target"] == "inclusionai/ling-3.0-flash-sante:free"


# 14. exec on an UNKNOWN turn -> payload untouched, no rewrite, never classifies
def test_exec_unknown_turn_untouched(monkeypatch):
    _reset()
    seen = []
    recs = []
    cls = _Classifier([])  # any classify call here is a bug
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", _Credit(True))
    monkeypatch.setattr(pl, "_log", recs.append)

    req, ctx = _exec_ctx(_kw(turn_id="unknown-1", api_call_count=1))
    out = pl.on_llm_execution(request=req, next_call=_capture_next(seen), **ctx)
    assert out["model"] == "z-ai/glm-5.3-flash"  # original model went downstream
    assert seen[0] == req  # byte-identical payload
    assert len(cls.calls) == 0
    assert recs == []  # no rows from the execution path for unknown turns


# 15. exec escalation with the paid lane dark -> untouched + credit_lost row
def test_exec_escalation_credit_lost(monkeypatch):
    _reset()
    seen = []
    recs = []
    cls = _Classifier([("paid", 0.95)])
    credit = _Credit(True)
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", credit)
    monkeypatch.setattr(pl, "_log", recs.append)

    kw = _kw(turn_id="ecl-1", model="z-ai/glm-5.3", api_call_count=1)
    first = pl.on_llm_request(**kw)
    assert first is not None
    assert first["request"]["model"] == "anthropic/claude-sonnet-5.5"

    credit.alive = False  # lane goes dark before the follow-up call
    req, ctx = _exec_ctx(_kw(turn_id="ecl-1", model="z-ai/glm-5.3", api_call_count=2))
    out = pl.on_llm_execution(request=req, next_call=_capture_next(seen), **ctx)
    assert out["model"] == "z-ai/glm-5.3"  # fail-closed: original mid model
    assert seen[0] == req
    assert len(cls.calls) == 1
    lost = [r for r in recs if r.get("credit_lost")]
    assert lost and lost[-1]["applied"] is False and lost[-1]["turn_id"] == "ecl-1"


# 16. exec in shadow mode -> never rewritten, no extra rows
def test_exec_shadow_untouched_no_rows(monkeypatch):
    _reset()
    seen = []
    recs = []
    cls = _Classifier([("free", 0.95)])
    credit = _Credit(True)
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg(mode="shadow"))
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", credit)
    monkeypatch.setattr(pl, "_log", recs.append)

    first = pl.on_llm_request(**_kw(turn_id="esh-1", api_call_count=1))
    assert first is None  # shadow request path never rewrites
    rows_after_first = len(recs)
    req, ctx = _exec_ctx(_kw(turn_id="esh-1", api_call_count=2))
    out = pl.on_llm_execution(request=req, next_call=_capture_next(seen), **ctx)
    assert out["model"] == "z-ai/glm-5.3-flash"
    assert seen[0] == req  # untouched payload
    assert len(recs) == rows_after_first  # no exec rows in shadow
    assert len(cls.calls) == 1
    assert credit.calls == 0  # shadow never probes


# 17. exec NEVER classifies: one classify (via llm_request) total across 3 exec invocations
def test_exec_never_classifies(monkeypatch):
    _reset()
    seen = []
    recs = []
    cls = _Classifier([("free", 0.90)])
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", cls)
    monkeypatch.setattr(pl, "_paid_lane_alive", _Credit(True))
    monkeypatch.setattr(pl, "_log", recs.append)

    assert pl.on_llm_request(**_kw(turn_id="nc-1", api_call_count=1)) is not None
    for n in (2, 3, 4):
        req, ctx = _exec_ctx(_kw(turn_id="nc-1", api_call_count=n,
                                 messages=[{"role": "user", "content": f"follow {n}"}]))
        out = pl.on_llm_execution(request=req, next_call=_capture_next(seen), **ctx)
        assert out["model"] == "inclusionai/ling-3.0-flash-sante:free"
    assert len(cls.calls) == 1  # classify count == 1 across 3 exec invocations
    assert len(seen) == 3


# 18. next_call semantics: our pre-dispatch bookkeeping failing (config read
#     blows up) must not swallow the call — one dispatch with the UNTOUCHED
#     request, response returned intact.
def test_exec_our_failure_still_dispatches_once_untouched(monkeypatch):
    _reset()
    seen = []
    calls = []
    real_open = pl._load_config

    def _boom():
        calls.append(1)
        raise RuntimeError("config read exploded")
    monkeypatch.setattr(pl, "_load_config", _boom)
    monkeypatch.setattr(pl, "_log", lambda rec: None)

    req, ctx = _exec_ctx(_kw(turn_id="boom-1", api_call_count=2))
    out = pl.on_llm_execution(request=req, next_call=_capture_next(seen), **ctx)
    assert out["model"] == "z-ai/glm-5.3-flash"  # untouched payload reached downstream
    assert seen[0] == req
    assert len(calls) == 1  # one bookkeeping attempt; the fail-open dispatch
    # bypasses bookkeeping (config is re-read on the NEXT invocation, not per
    # dispatch), so downstream is reached exactly once.


# 19. next_call semantics: a provider error downstream PROPAGATES verbatim
#     (after invocation) — the wrapper must not mask it with None.
def test_exec_downstream_error_propagates(monkeypatch):
    _reset()
    monkeypatch.setattr(pl, "_load_config", lambda: _cfg())
    monkeypatch.setattr(pl, "_classify", _Classifier([("free", 0.90)]))
    monkeypatch.setattr(pl, "_paid_lane_alive", _Credit(True))
    monkeypatch.setattr(pl, "_log", lambda rec: None)

    assert pl.on_llm_request(**_kw(turn_id="perr-1", api_call_count=1)) is not None

    dispatched = []

    def _provider_down(request):
        dispatched.append(True)
        raise RuntimeError("provider 503")

    req, ctx = _exec_ctx(_kw(turn_id="perr-1", api_call_count=2))
    try:
        pl.on_llm_execution(request=req, next_call=_provider_down, **ctx)
    except RuntimeError as e:
        assert str(e) == "provider 503"  # verbatim, not wrapped/masked
    else:
        raise AssertionError("downstream provider error was swallowed")
    assert len(dispatched) == 1  # dispatched exactly once, then raised


# 20. next_call semantics: even when BOTH our bookkeeping and the fail-open
#     dispatch fail (e.g. provider error), the error propagates — never a
#     silent None that would look like a successful call.
def test_exec_failopen_dispatch_failure_propagates(monkeypatch):
    _reset()
    dispatched = []

    def _boom_cfg():
        raise RuntimeError("config read exploded")
    monkeypatch.setattr(pl, "_load_config", _boom_cfg)
    monkeypatch.setattr(pl, "_log", lambda rec: None)

    def _down(request):
        dispatched.append(True)
        raise RuntimeError("downstream too")

    req, ctx = _exec_ctx(_kw(turn_id="boom-2", api_call_count=2))
    try:
        pl.on_llm_execution(request=req, next_call=_down, **ctx)
    except RuntimeError as e:
        assert str(e) == "downstream too"
    else:
        raise AssertionError("fail-open dispatch error was swallowed")
    assert dispatched == [True]  # dispatched exactly once


# 21. bookkeeping failure + healthy downstream: exactly ONE dispatch, whose
#     response is returned to the host — the turn proceeds, byte-identical
#     payload (the fail-open invariant, asserted end-to-end).
def test_exec_bookkeeping_failure_single_dispatch_returns_response(monkeypatch):
    _reset()

    def _boom_cfg():
        raise RuntimeError("config read exploded")
    monkeypatch.setattr(pl, "_load_config", _boom_cfg)
    monkeypatch.setattr(pl, "_log", lambda rec: None)

    calls = []

    def _next(request):
        calls.append(dict(request))
        return {"ok": True}

    req, ctx = _exec_ctx(_kw(turn_id="boom-3", api_call_count=2))
    out = pl.on_llm_execution(request=req, next_call=_next, **ctx)
    assert out == {"ok": True}
    assert calls == [req]  # one dispatch, untouched original payload
