"""Tests for the vendored scripts/jev_client.py — tests are the documentation.

Offline tests use the `transport` seam; no network, no key file needed.
There are deliberately NO live-marked tests in the repo: the suite must run
green on any machine with no keys and no network.
"""
import json
import os

import pytest

import jev_client as jc

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_schema.json")


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def golden():
    with open(GOLDEN, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ok_response(payload):
    """Deterministic offline 200 response mirroring the golden schema."""
    answers = {}
    for qid, q in payload["questions"].items():
        if q["type"] == "choice":
            first = list(q["criteria"].keys())[0]
            answers[qid] = {"type": "choice", "choice": first,
                            "probabilities": {k: (1.0 if k == first else 0.0)
                                              for k in q["criteria"]},
                            "confidence": 0.9}
        elif q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.95}
        else:
            answers[qid] = {"type": "score", "score": 1.0, "confidence": 0.8,
                            "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                            "probabilities": {str(i): (1.0 if i == 1 else 0.0)
                                              for i in range(len(q["criteria"]))}}
    return 200, {"model": "jev-1.13.0", "answers": answers,
                 "usage": {"input_tokens": 1000, "output_tokens": 10}}


# ---------------------------------------------------------------- key loading

def test_load_api_key_skips_comment_lines(tmp_path):
    p = tmp_path / "creds"
    p.write_text("SK-TEST-123\n# this is a comment\n")
    assert jc.load_api_key(str(p), env_var="JEVI_API_KEY_UNUSED_BY_TEST") == "SK-TEST-123"


def test_load_api_key_env_first(monkeypatch):
    """ENV-FIRST contract: JEVI_API_KEY in the process env wins over the file."""
    monkeypatch.setenv("JEVI_API_KEY", "env-key-wins")
    assert jc.load_api_key() == "env-key-wins"


def test_load_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("JEVI_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", "/nonexistent/hermes-home")
    with pytest.raises(jc.JevError):
        jc.load_api_key("/nonexistent/creds/path")


# ---------------------------------------------------------------- fixed schema

def test_request_schema_is_exact_no_invented_fields():
    seen = {}

    def transport(payload):
        seen.update(payload)
        return ok_response(payload)

    jc.evaluate("state text", {"q": {"type": "noul", "instructions": "yes?"}},
                api_key="SK-TEST", transport=transport)
    assert set(seen.keys()) == {"state", "model", "questions"}
    assert set(seen["questions"]["q"].keys()) == {"type", "instructions"}
    assert seen["model"] == jc.DEFAULT_MODEL


def test_transport_error_mapping():
    def auth_fail(payload):
        return 401, {"error": "missing or invalid API key"}

    with pytest.raises(jc.JevAuthError) as ei:
        jc.evaluate("s", {"q": {"type": "noul", "instructions": "y?"}},
                    api_key="FIXTURE", transport=auth_fail)
    # key value must never leak into the error or its body
    assert "FIXTURE" not in str(ei.value)
    assert "FIXTURE" not in (ei.value.body or "")


def test_retry_once_on_429_then_success():
    calls = {"n": 0}

    def flaky(payload):
        calls["n"] += 1
        if calls["n"] == 1:
            return 429, {"error": "rate limited"}
        return ok_response(payload)

    r = jc.evaluate("s", {"q": {"type": "noul", "instructions": "y?"}},
                    api_key="SK-TEST", transport=flaky)
    assert calls["n"] == 2 and r["retries_used"] == 1


def test_no_retry_when_retries_zero():
    """Contract: retries=0 must attempt EXACTLY once."""
    calls = {"n": 0}

    def flaky(payload):
        calls["n"] += 1
        return 429, {"error": "rate limited"}

    with pytest.raises(jc.JevTransientError):
        jc.evaluate("s", {"q": {"type": "noul", "instructions": "y?"}},
                    api_key="SK-TEST", transport=flaky, retries=0)
    assert calls["n"] == 1


def test_no_retry_on_401():
    calls = {"n": 0}

    def auth_fail(payload):
        calls["n"] += 1
        return 401, {"error": "invalid"}

    with pytest.raises(jc.JevAuthError):
        jc.evaluate("s", {"q": {"type": "noul", "instructions": "y?"}},
                    api_key="SK-TEST", transport=auth_fail)
    assert calls["n"] == 1  # auth-fail protocol: never a retry loop


def test_cost_meter_uses_provider_counted_usage():
    r = jc.evaluate("s", {"q": {"type": "noul", "instructions": "y?"}},
                    api_key="SK-TEST", transport=ok_response)
    assert r["usage"] == {"input_tokens": 1000, "output_tokens": 10}
    assert abs(r["cost_usd"] - 1000 * 0.042 / 1e6) < 1e-9  # output free


# ---------------------------------------------------------------- validator

def test_validator_accepts_golden_example(golden):
    req = golden["request"]["example_request"]
    resp = golden["response"]["example_response"]
    assert jc.validate_response(resp, req["questions"]) == []


def test_validator_rejects_wrong_answer_type():
    questions = {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": None, "b": None}}}
    bad = {"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 1},
           "answers": {"q": {"type": "noul", "noul": 0.5}}}
    assert jc.validate_response(bad, questions)


def test_validator_rejects_choice_outside_criteria():
    questions = {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": None, "b": None}}}
    bad = {"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 1},
           "answers": {"q": {"type": "choice", "choice": "not_an_option",
                             "probabilities": {"a": 1.0}, "confidence": 0.9}}}
    errs = jc.validate_response(bad, questions)
    assert any("not in criteria" in e for e in errs)


def test_validator_rejects_missing_usage_and_tolerates_extra_fields():
    questions = {"q": {"type": "noul", "instructions": "y?"}}
    ok = {"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 1},
          "answers": {"q": {"type": "noul", "noul": 0.5}}, "request_id": "abc", "new_field": 7}
    assert jc.validate_response(ok, questions) == []
    no_usage = {k: v for k, v in ok.items() if k != "usage"}
    assert jc.validate_response(no_usage, questions)


# ---------------------------------------------------------------- chunking / batching

def test_chunking_respects_255_cap():
    entries = [("skill_%03d" % i, "desc %d" % i) for i in range(570)]
    chunks = jc.chunk_entries(entries, per_chunk=254)
    assert len(chunks) == 3  # 254 + 254 + 62
    assert all(len(c) <= 254 for c in chunks)


def test_skill_selection_batches_with_none_option_and_picks_winner():
    entries = [("alpha_option", "best match for debugging"), ("beta_option", "unrelated")]
    seen_questions = {}

    def transport(payload):
        seen_questions.update(payload["questions"])
        qid = list(payload["questions"])[0]
        payload["questions"][qid]["criteria"].setdefault("none", "")
        return ok_response(payload)

    dec = jc.skill_selection("task state", entries, "Which option matches?",
                             api_key="SK-TEST", transport=transport)
    assert dec["selected"] == "alpha_option"
    assert dec["n_requests"] == 1
    q = seen_questions["selection_chunk_0"]
    assert set(q.keys()) == {"type", "instructions", "criteria"}
    assert len(q["criteria"]) == 3  # 2 options + none = within 255 cap
    assert q["criteria"]["none"]


def test_skill_selection_chunks_across_requests():
    entries = [("s%03d" % i, "d") for i in range(510)]
    seen = {"n": 0}

    def transport(payload):
        seen["n"] += 1
        return ok_response(payload)

    dec = jc.skill_selection("task", entries, "pick", api_key="SK-TEST", transport=transport)
    assert seen["n"] == 3 and dec["n_requests"] == 3
    assert dec["usage"]["input_tokens"] == 3000  # provider-counted sum
