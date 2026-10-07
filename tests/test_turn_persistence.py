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
