#!/usr/bin/env python3
"""jev_client.py — TypeSafe Jev stdlib client (vendored by kfchow-llm-value-router).

The plugin's route classifier talks to the TypeSafe System One ("Jev") API.
This is a small, dependency-free client. Design rules enforced here:

- FIXED request schema: {state, model, questions} only — no invented fields.
  Each question is exactly {type, instructions, criteria}; nothing else is sent.
- Choice chunking: <=255 options per Choice question (vendor cap); a large
  catalog decision is split into ceil(N/254) chunked requests (254 options +
  a "none" no-match option per chunk).
- Timeout 3 s/request; retry <=1, only on 429/529/network errors per vendor
  error docs. Auth (401/403) and request (422) errors are never retried.
- Typed-response validator. Unknown extra fields are tolerated (vendor
  forward-compat); missing or wrongly-typed required fields are rejected.
- Cost meter: PROVIDER-COUNTED usage tokens only; $ = input_tokens x
  $0.042/Mtok (output tokens free). Never estimated.
- Auth: Bearer-auth key resolved ENV-FIRST from the JEVI_API_KEY environment
  variable; falls back to a 0600 perms creds file. The key value never
  leaves this module and is never logged.

Stdlib only (urllib.request). Python 3.9+.
"""
import json
import os
import socket
import time
import urllib.error
import urllib.request

# Endpoint configuration. Defaults point at the vendor's public API; both the
# base URL and the model id can be overridden without touching this file.
DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-1.13.0"  # pinned versioned ID (versioned IDs always accepted)
BASE_URL = os.environ.get("JEVI_BASE_URL", DEFAULT_BASE_URL)
ENDPOINT = "/v1/systemone"
MODELS_ENDPOINT = "/v1/models"
PRICE_PER_INPUT_MTOK = 0.042  # $ per million input tokens; output tokens free
MAX_CHOICE_OPTIONS = 255      # vendor cap per Choice question
TIMEOUT_S = 3.0
RETRIES = 1                   # <=1 retry, 429/529/network only
_BACKOFF_S = 0.5

RETRYABLE_STATUS = (429, 529)


class JevError(Exception):
    """Base error. Carries HTTP status + truncated vendor body. NEVER carries the key."""

    def __init__(self, message, status=None, body=None):
        super().__init__(message)
        self.status = status
        self.body = (body or "")[:2000]


class JevAuthError(JevError):
    """401/403 — bad key. Per auth-fail protocol: count probe, do NOT retry-loop."""


class JevRequestError(JevError):
    """422 — request body rejected (our bug: malformed question / invented field)."""


class JevTransientError(JevError):
    """429/529/5xx/network — retryable per vendor docs."""


# --------------------------------------------------------------------------- auth

def load_api_key(path=None, env_var="JEVI_API_KEY"):
    """Resolve the API key, ENV FIRST.

    1. os.environ[env_var] (JEVI_API_KEY) — Hermes loads ~/.hermes/.env into the
       process environment at startup, and any CI/exported variable works too.
    2. Fallback: the 0600 perms creds file (line 1 = key; '#' lines are comments).
    The key value never leaves this module and is never logged.
    """
    from_env = os.environ.get(env_var)
    if from_env and from_env.strip():
        return from_env.strip()
    path = path or os.path.join(
        os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
        "secrets", "jev.creds")
    try:
        fh = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise JevError(
            "no API key: set %s or provide a creds file at %s (%s)"
            % (env_var, path, exc.__class__.__name__))
    with fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    raise JevError("no API key line found in creds file")


def creds_path():
    """Default creds-file path (Hermes-home aware)."""
    return os.path.join(
        os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
        "secrets", "jev.creds")


# --------------------------------------------------------------------------- transport

def _http_json(url, api_key, payload, timeout):
    """One HTTP attempt. Returns (status, body_dict). Raises urllib HTTPError/URLError."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _map_http_status(status, body):
    """Map a non-200 status to the typed error. 200-299 returns None."""
    if 200 <= status < 300:
        return None
    if status in (401, 403):
        return JevAuthError("auth rejected (status %d)" % status, status=status, body=body)
    if status == 422:
        return JevRequestError("request rejected (status %d)" % status, status=status, body=body)
    return JevTransientError("transient HTTP %d" % status, status=status, body=body)


def _request_once(payload, api_key, timeout):
    """Single attempt with error mapping. Raises typed JevError on failure."""
    url = BASE_URL + ENDPOINT
    try:
        status, body = _http_json(url, api_key, payload, timeout)
        err = _map_http_status(status, "")
        if err is not None:
            raise err
        return status, body
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        if exc.code in (401, 403):
            raise JevAuthError("auth rejected (status %d)" % exc.code, status=exc.code, body=body)
        if exc.code == 422:
            raise JevRequestError("request rejected (status %d)" % exc.code, status=exc.code, body=body)
        raise JevTransientError("transient HTTP %d" % exc.code, status=exc.code, body=body)
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
        raise JevTransientError("network failure: %s" % exc.__class__.__name__)


def evaluate(state, questions, model=DEFAULT_MODEL, api_key=None, timeout=TIMEOUT_S,
             retries=RETRIES, transport=None):
    """POST /v1/systemone once (with <=retries retries on transient errors).

    transport: optional test seam — callable(payload) -> (status, body_dict).
    Returns dict: answers, usage, cost_usd, latency_s, model, status, retries_used.
    Raises typed JevError; auth/request errors are never retried.
    """
    payload = {"state": state, "model": model, "questions": questions}  # fixed schema
    key = api_key if api_key is not None else load_api_key()
    attempt = 0
    t0 = time.monotonic()
    while True:
        try:
            if transport is not None:
                status, body = transport(payload)
                err = _map_http_status(status, json.dumps(body)[:2000] if not isinstance(body, str) else body)
                if err is not None:
                    raise err
            else:
                status, body = _request_once(payload, key, timeout)
            latency = time.monotonic() - t0
            break
        except JevTransientError:
            if attempt < retries:
                attempt += 1
                time.sleep(_BACKOFF_S * attempt)
                continue
            raise
    errors = validate_response(body, questions)
    if errors:
        # A structurally-invalid 200 response counts as a transient vendor fault.
        raise JevTransientError("response failed schema validation: %s" % "; ".join(errors[:5]),
                                status=status, body=json.dumps(body)[:2000])
    usage = body["usage"]
    cost_usd = usage["input_tokens"] * PRICE_PER_INPUT_MTOK / 1e6  # output free
    return {
        "answers": body["answers"],
        "usage": usage,
        "cost_usd": round(cost_usd, 9),
        "latency_s": round(latency, 4),
        "model": body["model"],
        "status": status,
        "retries_used": attempt,
        "validated": True,
    }


def list_models(api_key=None, timeout=TIMEOUT_S):
    """GET /v1/models — names the account can send in the model field."""
    key = api_key if api_key is not None else load_api_key()
    req = urllib.request.Request(
        BASE_URL + MODELS_ENDPOINT, method="GET",
        headers={"Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        if exc.code in (401, 403):
            raise JevAuthError("auth rejected (status %d)" % exc.code, status=exc.code, body=body)
        raise JevTransientError("HTTP %d on /v1/models" % exc.code, status=exc.code, body=body)


# --------------------------------------------------------------------------- validation

def validate_response(body, questions=None):
    """Validate a System One response.

    Returns a list of error strings (empty = valid). Tolerates unknown extra
    fields (vendor forward-compat), rejects missing/wrongly-typed required ones.
    """
    errs = []
    if not isinstance(body, dict):
        return ["response is not an object"]
    model = body.get("model")
    if not isinstance(model, str) or not model:
        errs.append("model: missing or not a string")
    usage = body.get("usage")
    if not isinstance(usage, dict):
        errs.append("usage: missing or not an object")
    else:
        for f in ("input_tokens", "output_tokens"):
            v = usage.get(f)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                errs.append("usage.%s: not a non-negative integer (%r)" % (f, v))
    answers = body.get("answers")
    if not isinstance(answers, dict):
        errs.append("answers: missing or not an object")
        return errs
    if questions:
        for qid, q in questions.items():
            a = answers.get(qid)
            if a is None:
                errs.append("answers.%s: missing answer for sent question" % qid)
                continue
            errs.extend(_validate_answer(qid, q, a))
    return errs


def _validate_answer(qid, q, a):
    errs = []
    qtype = q.get("type")
    if a.get("type") != qtype:
        errs.append("answers.%s.type: expected %r got %r" % (qid, qtype, a.get("type")))
        return errs
    if qtype == "noul":
        v = a.get("noul")
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.0 <= v <= 1.0):
            errs.append("answers.%s.noul: not a number in [0,1] (%r)" % (qid, v))
    elif qtype == "choice":
        criteria = q.get("criteria") or {}
        choice = a.get("choice")
        if not isinstance(choice, str):
            errs.append("answers.%s.choice: not a string (%r)" % (qid, choice))
        elif criteria and choice not in criteria:
            errs.append("answers.%s.choice: %r not in criteria options" % (qid, choice))
        probs = a.get("probabilities")
        if not isinstance(probs, dict):
            errs.append("answers.%s.probabilities: missing or not an object" % qid)
        else:
            bad = [k for k in probs if criteria and k not in criteria]
            if bad:
                errs.append("answers.%s.probabilities: unknown options %s" % (qid, bad[:3]))
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in probs.values()):
                errs.append("answers.%s.probabilities: non-numeric value" % qid)
        conf = a.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not (0.0 <= conf <= 1.0):
            errs.append("answers.%s.confidence: not a number in [0,1] (%r)" % (qid, conf))
    elif qtype == "score":
        criteria = q.get("criteria") or []
        n_levels = len(criteria)
        score = a.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            errs.append("answers.%s.score: not a number (%r)" % (qid, score))
        elif n_levels and not (-0.001 <= score <= n_levels - 1 + 0.001):
            errs.append("answers.%s.score: %r outside [0,%d]" % (qid, score, n_levels - 1))
        legend = a.get("legend")
        if not isinstance(legend, dict) or not legend:
            errs.append("answers.%s.legend: missing or empty" % qid)
        probs = a.get("probabilities")
        if not isinstance(probs, dict) or not probs:
            errs.append("answers.%s.probabilities: missing or empty" % qid)
        conf = a.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not (0.0 <= conf <= 1.0):
            errs.append("answers.%s.confidence: not a number in [0,1] (%r)" % (qid, conf))
    return errs


# --------------------------------------------------------------------------- batching

def chunk_entries(entries, per_chunk=254):
    """Split [(key, desc), ...] into chunks of <=(255-1) so each Choice can carry
    per_chunk options + a 'none' no-match option = <=255 options."""
    if per_chunk > MAX_CHOICE_OPTIONS - 1:
        raise ValueError("per_chunk must leave room for the 'none' option (<=254)")
    chunks = []
    for i in range(0, len(entries), per_chunk):
        chunks.append(entries[i:i + per_chunk])
    return chunks


def skill_selection(state, entries, instructions, model=DEFAULT_MODEL, api_key=None,
                    per_chunk=254, on_request=None, transport=None):
    """One full skill-selection decision = sequential chunked requests.

    entries: list of (option_key, description). Each chunk becomes ONE Choice
    question with a 'none' option. Returns a decision dict with per-request
    records, the winning option, provider-counted tokens and cost.
    on_request: optional callback(record) per request for logging.
    """
    chunks = chunk_entries(entries, per_chunk)
    per_request = []
    total_in = total_out = 0
    total_cost = 0.0
    t0 = time.monotonic()
    candidates = []
    for idx, chunk in enumerate(chunks):
        criteria = dict(chunk)
        criteria["none"] = "No listed option clearly matches the task"
        qid = "selection_chunk_%d" % idx
        questions = {qid: {"type": "choice", "instructions": instructions,
                           "criteria": criteria}}
        result = evaluate(state, questions, model=model, api_key=api_key,
                          transport=transport)
        ans = result["answers"][qid]
        rec = {
            "chunk_index": idx, "n_options": len(criteria), "latency_s": result["latency_s"],
            "usage": result["usage"], "cost_usd": result["cost_usd"], "model": result["model"],
            "choice": ans.get("choice"), "confidence": ans.get("confidence"),
            "top_probability": (ans.get("probabilities") or {}).get(ans.get("choice")),
        }
        per_request.append(rec)
        if on_request:
            on_request(rec)
        total_in += result["usage"]["input_tokens"]
        total_out += result["usage"]["output_tokens"]
        total_cost += result["cost_usd"]
        if ans.get("choice") and ans.get("choice") != "none":
            candidates.append((rec["top_probability"] or 0.0, ans["choice"], idx,
                               rec["confidence"]))
    candidates.sort(reverse=True)
    best = candidates[0] if candidates else None
    return {
        "selected": best[1] if best else "none",
        "selected_probability": best[0] if best else None,
        "selected_chunk": best[2] if best else None,
        "selected_confidence": best[3] if best else None,
        "n_requests": len(chunks),
        "wall_s": round(time.monotonic() - t0, 4),
        "usage": {"input_tokens": total_in, "output_tokens": total_out},
        "cost_usd": round(total_cost, 9),
        "per_request": per_request,
    }


if __name__ == "__main__":
    # Tiny CLI smoke: 1-question probe. Never prints the key.
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    try:
        r = evaluate("calibration probe", {
            "is_probe": {"type": "noul", "instructions": "Does this state contain the word probe?"}
        }, model=model)
        print(json.dumps({"model": r["model"], "usage": r["usage"], "cost_usd": r["cost_usd"],
                          "latency_s": r["latency_s"]}))
    except JevError as exc:
        print(json.dumps({"error": exc.__class__.__name__, "status": exc.status,
                          "body": exc.body}))
        sys.exit(1)
