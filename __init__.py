"""kfchow-llm-value-router — per-turn model routing for Hermes (llm_request + llm_execution middleware).

WHAT IT DOES
    On the FIRST provider call of each turn, asks Jev (TypeSafe System One)
    which pool the turn belongs to, then routes it down a 4-TIER ladder:
    free (routine, conf >= free gate), flash (near-routine, flash_gate <=
    conf < free gate), mid (stays on the configured model), premium (strong
    turns escalate, credit-probed). In `shadow` mode it only records the
    decision; in `live` mode it rewrites the request's model to the tier's
    rung.

    An ESCALATION path runs the other direction: when a turn starts on a "mid"
    rung and the classifier says it needs a strong model, the request can be
    escalated to the provider's premium rung — guarded by a credit probe so a
    metered provider with an exhausted balance never receives a doomed rewrite.

PER-REQUEST KILL SWITCH (v1.0.8, X-KFC-Router)
    A client can disable routing for ITS OWN request, without any config edit
    and without touching other turns. BOTH middlewares read the flag from the
    request payload on EVERY callback, so a flag set mid-turn stops both the
    first-call rewrite and the reapply of an earlier decision:
      * HTTP-header form:  request["extra_headers"]["X-KFC-Router"] = "off"
      * JSON-body form:    request["metadata"]["router"] = "off"
                          (or the top-level request["router"] = "off")
    Disabling values: off / 0 / false / disable / disabled (case-insensitive).
    When the flag is set: no rewrite, NO classifier call, no credit probe, ONE
    log row with skip_reason "disabled_by_request", and NO decision stored —
    the turn stays unknown so nothing re-applies. An unknown value is ignored
    (fail-open: normal routing, no warning). The flag only turns routing OFF;
    the plugin-level enable/disable stays the config `enabled` key.

WHY TWO MIDDLEWARE (not a cron, not a skill)
    `llm_request` is the documented cutover path: it sees the full provider
    kwargs and can replace them. A skill is advisory and a cron cannot act on
    a live turn.

    But the host's llm_request chain runs only ONCE per turn on the real
    tool-follow-up path (attempt1; retries re-invoke it, tool-loop follow-ups
    do not), so a rewrite stored there silently reverts the moment the agent
    executes a tool and calls the provider again. `llm_execution` wraps the
    actual provider call — the host runs that chain on EVERY attempt, follow-
    ups included (agent/turn_api_call.py) — so the second middleware re-applies
    the turn's stored decision on every single call. Classification stays
    exactly once per turn, in `llm_request`; the execution middleware never
    classifies, it only re-applies what was already decided.

SAFETY CONTRACT (every one of these is enforced in code, not convention)
    * Fail-open: any exception, timeout, or malformed answer returns None so
      Hermes proceeds with the ORIGINAL request, byte-identical.
    * Classified once per turn — the decision (including "no rewrite") is
      stored and re-applied on every later callback of the same turn, so the
      host rebuilding its kwargs between attempts can never silently revert a
      rewrite mid-turn. Re-application never re-classifies and never crosses
      providers.
    * Only eligible lanes are touched. v1.0.7 SOURCE-AGNOSTIC ELIGIBILITY: a
      turn is eligible when its PROVIDER is listed (route_providers /
      escalate_providers) — whatever default model the user runs, so a
      premium- or foreign-default install gets the same two-way ladder as a
      flash-default one. `source_mode` ("any" default | "allowlist" = exact
      v1.0.6 model allowlists | explicit list of model ids) narrows it. A turn
      already on a free model (_FREE_HINTS, checked FIRST), on another
      provider, or with no model field is left alone, and a turn is never
      rewritten to the model it is already on.
    * The rewritten request keeps every other key, including the message list,
      so the provider payload stays valid.
    * PER-REQUEST KILL SWITCH honoured on every callback: a request carrying
      the disabling flag (X-KFC-Router header or router body field) is never
      rewritten, never classified and never stores a decision — read fresh
      from the payload each time, so a flag set mid-turn also stops the
      reapply path.
    * A rewrite NEVER crosses providers: the rungs are resolved per provider,
      so the model id sent always belongs to the originating provider's API.
    * Mode is `shadow` until a human flips `mode` to `live` in the config.
    * The llm_execution wrapper calls next_call EXACTLY once per invocation:
      it is the provider call itself. Fail-open covers only OUR code — any
      error from downstream (the provider/next middleware) propagates after
      having been invoked, and a next_call that was never invoked is handed
      the untouched request, never swallowed.
    * Telemetry: decision logs are written to a local JSONL file only. Set
      `send_excerpt: false` to keep task text out of the vendor call AND the
      log (shape-only features still classify, with reduced accuracy).

Config: ~/.hermes/jev/lane2_config.json       Log: ~/.hermes/jev/lane2-live.jsonl
"""

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.request
from collections import OrderedDict

logger = logging.getLogger(__name__)

_HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
CONFIG_PATH = os.path.join(_HERMES_HOME, "jev", "lane2_config.json")
LOG_PATH = os.path.join(_HERMES_HOME, "jev", "lane2-live.jsonl")
ROUTING_QUESTION_ID = "routing_tier"
JEV_TIMEOUT_S = 10          # keep well under a turn's own latency budget
DEFAULT_GATE = 0.80
DEFAULT_EXCERPT_CHARS = 1200
# FLASH TIER (v1.0.6, the 4-tier ladder). free-pool turns BELOW the free gate
# but at or above the flash gate are near-routine work a small paid-by-token
# workhorse handles adequately — far cheaper than keeping them on the mid rung,
# and never a downgrade of a paid-pool verdict (those stay mid below the
# escalation gate). Below the flash gate: mid (no rewrite).
DEFAULT_FLASH_GATE = 0.35
# SOURCE-AGNOSTIC LADDER (v1.0.7): eligibility is provider-match only, so any
# default model (premium, mid, flash, foreign) moves both up and down its own
# provider's rungs. "allowlist" restores the v1.0.6 model-allowlist gate.
DEFAULT_SOURCE_MODE = "any"

# Same pool semantics as the offline measurement, so live decisions are
# comparable to the data that justified them.
POOL_CRITERIA = {
    "free": "Routine, mechanical, tool-driven work a small fast model handles "
            "adequately: file reads/searches, simple edits, formatting, running "
            "commands, short factual answers, status checks.",
    "paid": "Work needing a strong model: multi-step reasoning, architecture or "
            "plan design, subtle debugging, long-context synthesis, anything "
            "where a wrong answer is expensive.",
}

# A turn already on a free/trial model must never be "routed" again.
_FREE_HINTS = (":free", "free", "space-bunny")

# PER-REQUEST KILL SWITCH (v1.0.8). A client can disable routing for its own
# request with NO config edit and zero blast radius on other turns. The flag
# rides on the provider payload (the only per-request channel the middleware
# layer sees) in two interchangeable forms:
#   * request["extra_headers"]["X-KFC-Router"]  — the HTTP-header form
#   * request["metadata"]["router"] or request["router"] — the JSON-body form
KILL_SWITCH_HEADER = "x-kfc-router"
KILL_SWITCH_OFF_VALUES = ("off", "0", "false", "disable", "disabled")

# CREDIT GATE — checked before any escalation to a PAID model on a METERED
# provider. An account that has run out of credits refuses paid models (with
# HTTP 404 "requires available credits" in at least one provider's case, not
# 402). A gate that escalates to premium without checking would have turned
# that outage into failed turns. Fail-CLOSED: an unknown probe outcome means
# do not escalate — the turn keeps the affordable rung.
CREDIT_PROBE_MODEL = "z-ai/glm-5.3-flash"
CREDIT_TTL_S = 300          # don't probe on every turn
_CREDIT = {"ok": None, "checked_at": 0.0, "identity": None}
_CREDIT_LOCK = threading.Lock()

# TURN DECISION CACHE — the P1 fix. The host rebuilds the provider kwargs from
# the agent's configured model and re-runs the llm_request middleware chain on
# EVERY attempt of a turn (in-attempt retries and tool-loop follow-ups), so a
# rewrite that fires only on the first call would silently revert. Instead:
# classify once per (session, turn), store the RESOLVED decision, and re-apply
# it on every later callback.
_DECISIONS = OrderedDict()            # "session_id:turn_id" -> decision dict
_DECISIONS_LOCK = threading.Lock()
_DECISIONS_MAX = 256                  # evict oldest (LRU)
_DECISIONS_TTL_S = 3600               # expiry; expired => behave as unknown (fail-open)

# Resolved lazily; the vendored client lives in the plugin's own scripts/ dir.
_JC = None


def _vendored_client():
    """Import the repo-local scripts/jev_client.py (portable: no external path)."""
    global _JC
    if _JC is None:
        scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import jev_client as jc  # repo-local vendored copy
        _JC = jc
    return _JC


def _nous_api_key():
    """The Nous credit-probe key, from the PROCESS ENVIRONMENT only.

    Hermes loads the user's ~/.hermes/.env into the process env at startup, so
    os.environ.get("NOUS_API_KEY") resolves there; CI users export it manually.
    This module never opens ~/.hermes/.env itself — reading that file is both
    unnecessary and unsafe, and install-time scanners treat it as a critical
    secrets-exposure pattern.
    """
    return os.environ.get("NOUS_API_KEY")


def _nous_base_url():
    return os.environ.get("NOUS_BASE_URL", "https://inference-api.nousresearch.com/v1")


def _paid_lane_alive():
    """True if a cheap paid call succeeds. Cached CREDIT_TTL_S. Fail-CLOSED:
    unknown => False => do not escalate (stay on the affordable rung)."""
    key = _nous_api_key()
    if not key:
        return False
    base_url = _nous_base_url().rstrip("/")
    # A cached probe only describes this endpoint, credential and probe model.
    # Keep one bounded entry and never retain the plaintext credential in it.
    identity = hashlib.sha256(json.dumps(
        [base_url, key, CREDIT_PROBE_MODEL], separators=(",", ":")
    ).encode()).hexdigest()
    now = time.monotonic()
    with _CREDIT_LOCK:
        if (_CREDIT.get("identity") == identity and _CREDIT["ok"] is not None
                and (now - _CREDIT["checked_at"]) < CREDIT_TTL_S):
            return _CREDIT["ok"]
    alive = False
    try:
        if key:
            body = json.dumps({"model": CREDIT_PROBE_MODEL,
                               "messages": [{"role": "user", "content": "ping"}],
                               "max_tokens": 2, "temperature": 0}).encode()
            req = urllib.request.Request(
                base_url + "/chat/completions",
                data=body, headers={"Content-Type": "application/json",
                                    "Authorization": "Bearer " + key})
            with urllib.request.urlopen(req, timeout=15) as r:
                json.loads(r.read())
            alive = True
    except Exception as e:
        logger.debug("credit probe failed (fail-closed, no escalation): %s", e)
    with _CREDIT_LOCK:
        _CREDIT["ok"] = alive
        _CREDIT["checked_at"] = time.monotonic()
        _CREDIT["identity"] = identity
    return alive


def _load_config():
    cfg = {"enabled": True, "mode": "shadow", "confidence_gate": DEFAULT_GATE,
           "route_providers": ["nous"],
           "route_models": [],
           # v1.0.7: "any" = provider-match eligibility (source-agnostic);
           # "allowlist" = v1.0.6 (route_models/escalate_models gate the
           # source model); a list of model ids = those exact sources only.
           "source_mode": DEFAULT_SOURCE_MODE,
           "free_model": "",
           "flash_gate": DEFAULT_FLASH_GATE,
           "escalate_enabled": True,
           "escalate_providers": [],
           "escalate_models": [],
           "premium_model": "",
           "escalate_confidence_gate": None,   # falls back to DEFAULT_GATE (0.80)
           "credit_probe": True,
           "send_excerpt": True,
           "excerpt_chars": DEFAULT_EXCERPT_CHARS,
           # PER-PROVIDER RUNGS. A rewrite must NEVER cross providers: the
           # request's base_url/api_key belong to the originating provider, so
           # swapping only the model id would send the wrong id to the wrong
           # endpoint. Resolved per provider, falling back to the flat keys.
           "rungs": {}}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg.update(json.load(fh) or {})
    except Exception:
        pass  # missing/corrupt config -> conservative defaults
    return cfg


def _rung(cfg, provider, key, fallback_key):
    """Per-provider rung with a flat-key fallback. Keeps a rewrite same-provider."""
    rungs = cfg.get("rungs") or {}
    entry = rungs.get(provider) or {}
    val = entry.get(key)
    if val in (None, [], ""):
        val = cfg.get(fallback_key)
    return val


_WARNED_SOURCE_MODES = set()


def _warn_bad_source_mode(mode):
    """One warning per distinct bad value: a typo'd source_mode must be visible
    (a silently narrowed router is the 'dead rung' class of bug), but the
    middleware runs every turn, so never log it more than once."""
    key = repr(mode)
    if key not in _WARNED_SOURCE_MODES:
        _WARNED_SOURCE_MODES.add(key)
        logger.warning(
            "kfchow-llm-value-router: unrecognised source_mode %s; falling "
            "back to 'allowlist' (v1.0.6 semantics). Use 'any', 'allowlist' "
            "or a list of model ids.", key)


def _source_eligibility(cfg, provider, model):
    """(free_src, esc_src): which lanes this turn's SOURCE may use (v1.0.7).

    A provider match is ALWAYS required (a rewrite never crosses providers);
    `source_mode` then decides how the SOURCE MODEL is treated:
      * "any" (default; also unset / null / ""): provider match only. Whatever
        the user's default model is, both directions of the ladder are open
        (route_models / escalate_models are ignored).
      * "allowlist": exact v1.0.6 semantics. The free lane needs
        model in route_models, the escalation lane needs
        model in escalate_models.
      * a LIST of model ids: only those exact source models are eligible, on
        both lanes (route_models / escalate_models are ignored). An empty
        list therefore makes nothing eligible, like an empty v1.0.6 allowlist.
    An unrecognised value falls back to "allowlist": the conservative side of
    the fail-open contract (it touches fewer turns, never more).
    """
    route_ok = provider in (cfg.get("route_providers") or [])
    esc_ok = (bool(cfg.get("escalate_enabled", True))
              and provider in (cfg.get("escalate_providers") or []))
    mode = cfg.get("source_mode", DEFAULT_SOURCE_MODE)
    if isinstance(mode, (list, tuple, set)):
        allowed = {str(m) for m in mode}
        return (route_ok and model in allowed), (esc_ok and model in allowed)
    mode = str(DEFAULT_SOURCE_MODE if mode is None else mode).strip().lower()
    if mode in ("", "any"):
        return route_ok, esc_ok
    if mode != "allowlist":
        _warn_bad_source_mode(cfg.get("source_mode"))
    # "allowlist", or anything unrecognised: v1.0.6 behaviour.
    return (route_ok and model in (cfg.get("route_models") or [])), \
           (esc_ok and model in (cfg.get("escalate_models") or []))


def _same_model(a, b):
    """True if two model ids name the same model (case-insensitive, and tolerant
    of a vendor prefix: 'z-ai/glm-5.3' == 'glm-5.3'). Used so a turn is never
    'moved' onto the model it is already on (e.g. a sonnet-default user being
    'escalated' to sonnet). A false positive only skips a rewrite — the
    fail-open direction. Empty ids never match."""
    a = str(a or "").strip().lower()
    b = str(b or "").strip().lower()
    if not a or not b:
        return False
    return a == b or a.rsplit("/", 1)[-1] == b.rsplit("/", 1)[-1]


def _kill_switch_off(request):
    """True when the REQUEST itself asks the router to stand down (v1.0.8).

    Read FRESH on every middleware callback, from the payload the client sent:
      * extra_headers["X-KFC-Router"] (header name and value case-insensitive)
      * metadata["router"] or the top-level request["router"]
    Disabling values: off / 0 / false / disable / disabled. Everything else
    — including explicit on-values ("on", "1", "true") and junk like "banana"
    — is IGNORED (fail-open: route normally, and log nothing about it, so an
    unrecognised value can never spam a warning per turn). The flag only ever
    turns routing OFF; enabling the plugin stays the config `enabled` key.
    """
    if not isinstance(request, dict):
        return False
    # HTTP-header form: names are case-insensitive per RFC, values per spec.
    def _off(value):
        # Native JSON scalars: JSON clients send router: false / router: 0, not the
        # string forms — coerce before the membership check so both wire.
        if isinstance(value, bool):
            return value is False  # false disables; true is an on-value
        if isinstance(value, (int, float)):
            return value == 0
        return str(value or "").strip().lower() in KILL_SWITCH_OFF_VALUES

    headers = request.get("extra_headers")
    if isinstance(headers, dict):
        for name, value in headers.items():
            if str(name or "").strip().lower() == KILL_SWITCH_HEADER:
                return True if _off(value) else False
    # JSON-body form, for clients that cannot set headers.
    for source in (request.get("metadata"), request):
        if isinstance(source, dict) and "router" in source:
            return True if _off(source.get("router")) else False
    return False


def _log(rec):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        # Decision logs may carry task text (send_excerpt) or route metadata;
        # restrict them to the owner only.
        try:
            os.chmod(LOG_PATH, 0o600)
        except OSError:
            pass
    except Exception:
        pass  # telemetry must never break a turn


def _features(request, excerpt_chars=1200, include_excerpt=True):
    """Features of the outgoing request.

    By default INCLUDES a task excerpt (the last user message, capped). This is
    load-bearing, not decoration: measured on real traffic, a SHAPE-ONLY state
    cannot tell strategy from routine — a hard architecture question scored
    free 0.31/paid 0.23 on shape alone, because "2 messages, 27 tools, 5403
    chars" carries no meaning. Adding the excerpt moved the same turn to paid
    1.00 and a routine turn to free 1.00.

    The excerpt sends task text to the Jev vendor. Operators who prefer not to
    share task text set `send_excerpt: false` in the config: the excerpt is
    then omitted from BOTH the vendor call and the local log row (shape-only
    features still reach the classifier, with reduced accuracy).
    Excerpt is capped and only the LAST user message is used.
    """
    msgs = request.get("messages") or request.get("input") or []
    n_msgs = len(msgs) if isinstance(msgs, list) else 0
    text = ""
    tools = request.get("tools") or []
    if isinstance(msgs, list) and msgs:
        last = msgs[-1]
        if isinstance(last, dict):
            c = last.get("content")
            if isinstance(c, str):
                text = c
            elif isinstance(c, list):  # multimodal content blocks
                text = " ".join(b.get("text", "") for b in c
                                if isinstance(b, dict) and b.get("type") == "text")
    feats = {
        "n_messages": n_msgs,
        "n_tools": len(tools) if isinstance(tools, list) else 0,
        "last_msg_chars": len(text),
        "code_fences": text.count("```"),
        "has_code": bool(re.search(r"\b(def |class |import |function |SELECT )", text)),
        "has_path": bool(re.search(r"[/~][\w.-]+/", text)),
        "is_question": text.strip().endswith("?"),
    }
    if include_excerpt:
        feats["task_excerpt"] = text[:excerpt_chars]
    return feats


def _classify(features, cfg):
    """Ask Jev. Returns (pool, confidence) or (None, None) on ANY failure."""
    try:
        jc = _vendored_client()
        include_excerpt = bool(cfg.get("send_excerpt", True))
        feats = {k: v for k, v in features.items() if k != "task_excerpt"}
        task = features.get("task_excerpt", "") if include_excerpt else ""
        if not include_excerpt:
            # Strip the excerpt entirely so it reaches neither the vendor call
            # nor the log row recorded from these features.
            features = feats
        state = {"turn_features": feats,
                 "task": task,
                 "context": "live agent turn; choose the cheapest ADEQUATE pool"}
        questions = {ROUTING_QUESTION_ID: {
            "type": "choice", "criteria": dict(POOL_CRITERIA)}}
        # Bound the call: a slow vendor must not stall the user's turn.
        timeout = float(cfg.get("jev_timeout_s", JEV_TIMEOUT_S))
        result = jc.evaluate(state, questions, timeout=timeout, retries=0)
        ans = (result.get("answers") or {}).get(ROUTING_QUESTION_ID) or {}
        return ans.get("choice"), ans.get("confidence")
    except Exception as e:
        logger.debug("jev classify failed (fail-open): %s", e)
        return None, None


def on_llm_request(**kwargs):
    """llm_request middleware. Returns {'request': {...}} to rewrite, else None."""
    try:
        cfg = _load_config()
        if not cfg.get("enabled", True):
            return None

        request = kwargs.get("request") or {}
        provider = str(kwargs.get("provider") or "")
        model = str(kwargs.get("model") or request.get("model") or "")
        turn_id = str(kwargs.get("turn_id") or "")
        session_id = str(kwargs.get("session_id") or "")
        api_call_count = kwargs.get("api_call_count")

        # Every invocation records a decision with a skip_reason, so an operator
        # can always answer "why didn't this turn route?" from the log alone.
        def _skip(reason):
            _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "turn_id": turn_id,
                  "session_id": session_id,
                  "provider": provider, "model": model, "pool": None,
                  "confidence": None, "gate": None, "eligible": False,
                  "mode": str(cfg.get("mode", "shadow")).lower(),
                  "applied": False, "skip_reason": reason})
            return None

        # ---- PER-REQUEST KILL SWITCH (v1.0.8) ------------------------------
        # The client asked for no routing on THIS request. Honoured FIRST,
        # ahead of the decision cache and every eligibility rule: no rewrite,
        # no classifier call, no credit probe, one log row, and NO decision
        # stored — the turn stays unknown so nothing re-applies later in the
        # turn. Read from the payload on EVERY callback (below at the reapply
        # path too), because a flag can appear on any call of the turn.
        if _kill_switch_off(request):
            return _skip("disabled_by_request")

        # ---- TURN DECISION CACHE (P1) ------------------------------------
        # The host re-runs this middleware on every attempt of the turn
        # (retries and tool-loop follow-ups rebuild the kwargs from the
        # agent's configured model each time), so a decision made on the first
        # call must be re-applied
        # on all later callbacks or the turn silently reverts to the source
        # model. Lookup BEFORE the first-call gate: follow-ups route through
        # the stored decision; only true first calls classify.
        key = f"{session_id}:{turn_id}"
        with _DECISIONS_LOCK:
            now = time.monotonic()
            for k in [k for k, v in _DECISIONS.items()
                      if (now - v["ts"]) >= _DECISIONS_TTL_S]:
                del _DECISIONS[k]
            while len(_DECISIONS) > _DECISIONS_MAX:
                _DECISIONS.popitem(last=False)  # evict oldest (LRU)
            known = _DECISIONS.get(key)
            if known is not None:
                # Refresh recency so an active turn is not LRU-evicted mid-flight.
                _DECISIONS[key] = known
                _DECISIONS.move_to_end(key)

        if known is not None:
            # Established decision: never classify again, never recompute
            # eligibility, and NEVER route across providers.
            if provider != known["provider"]:
                return None
            if str(cfg.get("mode", "shadow")).lower() != "live":
                return None  # shadow: follow-ups never rewritten, no extra rows
            decision = known["decision"]
            if decision == "free":
                target = known["target"]
                if not target:
                    return None  # fail-open defensive: never rewrite to ''
                updated = dict(request)
                updated["model"] = target
                _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "turn_id": turn_id, "session_id": session_id,
                      "provider": provider, "model": model, "target": target,
                      "applied": True, "reapply": True, "tier": "free",
                      "api_call_count": api_call_count})
                return {"request": updated, "source": "jev-lane2-router",
                        "reason": "reapply: free tier"}
            if decision == "flash":
                # FLASH re-apply mirrors free: no credit probe (flash is the
                # stable workhorse; host retry/fail-open covers a 404).
                target = known["target"]
                if not target:
                    return None  # fail-open defensive: never rewrite to ''
                updated = dict(request)
                updated["model"] = target
                _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "turn_id": turn_id, "session_id": session_id,
                      "provider": provider, "model": model, "target": target,
                      "applied": True, "reapply": True, "tier": "flash",
                      "api_call_count": api_call_count})
                return {"request": updated, "source": "jev-lane2-router",
                        "reason": "reapply: flash tier"}
            if decision == "paid":
                target = known["target"]
                if not target:
                    return None  # fail-open defensive: never rewrite to ''
                # Re-check the credit gate on every re-apply: cheap (TTL-cached)
                # and fail-closed — a lane that went dark mid-turn must not
                # receive a doomed premium rewrite.
                if cfg.get("credit_probe", True) and provider == "nous" \
                        and not _paid_lane_alive():
                    _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                          "turn_id": turn_id, "session_id": session_id,
                          "provider": provider, "model": model,
                          "applied": False, "credit_lost": True, "tier": "paid"})
                    return None
                updated = dict(request)
                updated["model"] = target
                _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "turn_id": turn_id, "session_id": session_id,
                      "provider": provider, "model": model, "target": target,
                      "applied": True, "reapply": True, "tier": "paid",
                      "api_call_count": api_call_count})
                return {"request": updated, "source": "jev-lane2-router",
                        "reason": "reapply: premium"}
            return None  # decision "none": below-gate/credit-blocked first call

        # First provider call of the turn only. Hermes counts the first call as
        # 1, so accept 0/1/None — later callbacks carry a stored decision and
        # were handled above; an unknown turn here is a true first attempt.
        if api_call_count not in (0, 1, None):
            return None  # follow-up call: intentionally unlogged (would flood)

        # The two paths have INDEPENDENT eligibility. Checking them as one chain
        # was a real bug: the free path's route_models filter rejected the mid
        # model before escalation ever ran, so escalation could never fire.
        # Evaluate each path on its own lane.
        # _FREE_HINTS guard runs FIRST, ahead of every eligibility rule: a
        # source already on a free model is never rewritten (logs
        # already_free_tier, not *_not_eligible), whatever source_mode says.
        if any(h in model.lower() for h in _FREE_HINTS):
            return _skip("already_free_tier")
        # v1.0.7 SOURCE-AGNOSTIC: eligibility is PROVIDER-match only by
        # default (source_mode "any"), so a premium-, mid-, flash- or
        # foreign-default user all get both directions of the ladder.
        # route_models / escalate_models only gate when source_mode is
        # "allowlist" (exact v1.0.6); a list value pins explicit sources.
        free_src, esc_src = _source_eligibility(cfg, provider, model)
        if not (free_src or esc_src):
            if provider not in (cfg.get("route_providers") or []):
                return _skip("provider_not_eligible")
            return _skip("model_not_eligible")
        if not request.get("model"):
            return _skip("no_model_in_request")

        feats = _features(request, excerpt_chars=int(cfg.get("excerpt_chars",
                                                             DEFAULT_EXCERPT_CHARS)),
                          include_excerpt=bool(cfg.get("send_excerpt", True)))
        pool, confidence = _classify(feats, cfg)

        gate = float(cfg.get("confidence_gate", DEFAULT_GATE))
        free_model = _rung(cfg, provider, "free", "free_model")
        mode = str(cfg.get("mode", "shadow")).lower()

        # ---- FLASH TIER (v1.0.6, the 4-tier ladder) -----------------------
        # free-pool turns BELOW the free gate but AT/ABOVE the flash gate are
        # near-routine: cheaper to serve on the provider's flash workhorse
        # than to keep billing them on the mid rung. Fails open: without a
        # flash rung (per-provider or flat fallback) the band stays mid.
        # pool=paid NEVER lands here — a paid verdict below the escalation
        # gate stays mid (downgrading planning turns would put them on flash).
        flash_gate_raw = cfg.get("flash_gate")
        try:
            flash_gate = float(flash_gate_raw if flash_gate_raw is not None
                               else DEFAULT_FLASH_GATE)
        except (TypeError, ValueError):
            flash_gate = DEFAULT_FLASH_GATE
        flash_model = _rung(cfg, provider, "flash", "flash_model")
        flash_band = (pool == "free" and confidence is not None
                      and flash_gate <= float(confidence) < gate
                      and bool(flash_model))
        # v1.0.7 identity guard: a turn is never "moved" onto the model it is
        # already on (a flash-default user IS on the flash rung). The band
        # fired, but there is nothing to rewrite: stay on the source model.
        flash_eligible = flash_band and not _same_model(flash_model, model)

        # ---- ESCALATION (mid -> premium), the other direction -------------
        # Fires when the turn is on a MID rung and the classifier says it needs
        # a strong model. Guarded by a live credit probe: escalating into an
        # exhausted account would fail the turn, which is worse than a weaker
        # answer.
        esc_gate_raw = cfg.get("escalate_confidence_gate")
        # v1.0.6 ladder: the escalation gate defaults to the code-level 0.80
        # (DEFAULT_GATE), NOT to confidence_gate — escalation to the premium
        # rung must never silently ride the free tier's lower gate, and an
        # explicit config override still wins.
        esc_gate = float(esc_gate_raw if esc_gate_raw is not None
                         else DEFAULT_GATE)
        premium = _rung(cfg, provider, "premium", "premium_model")
        esc_band = (
            esc_src
            and pool == "paid" and confidence is not None
            and float(confidence) >= esc_gate and bool(premium))
        # v1.0.7 identity guard: a premium-default user (source == premium
        # rung) is already at the top of the ladder. Never "escalate" to
        # itself, and never spend a credit probe deciding to do so.
        esc_eligible = esc_band and not _same_model(premium, model)
        credit_ok = None
        # Only a METERED provider needs the credit probe. A flat-rate
        # subscription lane has no per-call credit wall, and probing it with a
        # metered provider's key would be meaningless.
        if esc_eligible and cfg.get("credit_probe", True) and provider == "nous":
            credit_ok = _paid_lane_alive()
            if not credit_ok:
                esc_eligible = False

        # The FREE tier rides the same eligibility union as escalation: the
        # 41 measured target turns are mid-model sources listed under the
        # escalation lane, not route_models, and their free-pool verdicts
        # below the gate are exactly the band the flash tier serves.
        free_band = ((free_src or esc_src) and pool == "free"
                     and confidence is not None and float(confidence) >= gate
                     and bool(free_model))
        eligible = free_band and not _same_model(free_model, model)
        # True when a tier fired but its rung IS the source model (identity
        # guard above): logged as already_on_target, nothing rewritten.
        on_target = bool(
            (free_band and not eligible)
            or (flash_band and not flash_eligible)
            or (esc_band and _same_model(premium, model)))

        # Store the RESOLVED decision for EVERY classified turn (incl. "none":
        # below-gate / not-eligible / credit-blocked) so retries and follow-ups
        # never re-classify and re-application stays deterministic.
        decision = "free" if eligible else ("flash" if flash_eligible else
                    ("paid" if esc_eligible else "none"))
        target = (free_model if eligible else
                  (flash_model if flash_eligible else
                   (premium if esc_eligible else None)))
        _store_decision(key, provider, decision, target,
                        float(confidence) if confidence is not None else 0.0)

        log_feats = feats if cfg.get("send_excerpt", True) else {
            k: v for k, v in feats.items() if k != "task_excerpt"}
        _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "turn_id": turn_id,
              "session_id": session_id,
              "provider": provider, "model": model, "pool": pool,
              "confidence": confidence, "gate": gate, "eligible": eligible,
              "mode": mode,
              "applied": bool((eligible or flash_eligible) and mode == "live"),
              "tier": decision if decision != "none" else "mid",
              "skip_reason": None if (eligible or flash_eligible or esc_eligible) else (
                  "already_on_target" if on_target else (
                      "below_gate" if pool == "free" else "pool_paid")),
              "escalate_eligible": esc_eligible,
              "escalate_applied": bool(esc_eligible and mode == "live"),
              "flash_model": flash_model if flash_eligible else None,
              "premium_model": premium if esc_eligible else None,
              "credit_ok": credit_ok,
              "features": log_feats})

        if mode != "live":
            return None  # shadow: decide + log only, original request untouched

        if eligible:
            updated = dict(request)
            updated["model"] = free_model
            return {"request": updated, "source": "kfchow-llm-value-router",
                    "reason": "free tier (conf %.2f >= %.2f)" % (confidence, gate)}

        if flash_eligible:
            updated = dict(request)
            updated["model"] = flash_model
            return {"request": updated, "source": "kfchow-llm-value-router",
                    "reason": "flash tier (conf %.2f in [%.2f, %.2f))"
                              % (confidence, flash_gate, gate)}

        if esc_eligible:
            updated = dict(request)
            updated["model"] = premium
            return {"request": updated, "source": "kfchow-llm-value-router",
                    "reason": "ESCALATED to premium %s (conf %.2f >= %.2f)"
                              % (premium, confidence, esc_gate)}

        return None  # below gate / not eligible: original request untouched
    except Exception as e:
        logger.warning("kfchow-llm-value-router failed open: %s", e)
        return None


def on_llm_execution(request=None, next_call=None, **context):
    """llm_execution middleware: re-apply the turn's stored decision on EVERY
    provider call, tool follow-ups included.

    Contract (hermes_cli/middleware.py _run_execution_chain, call site
    agent/turn_api_call.py): ``request`` is the provider payload and
    ``next_call(request)`` IS the downstream execution — it must be invoked
    EXACTLY once per invocation of this callback. The host's own frame guards
    double invocation (second call raises), so this wrapper:

    * calls next_call exactly once on every path, and
    * lets downstream exceptions PROPAGATE (a provider error must reach the
      host verbatim — swallowing it here would turn one doomed attempt into
      either a lost response or an illegal retry-by-proxy).

    Fail-open covers only OUR code: any exception raised before next_call is
    dispatched retries the call once with the untouched ORIGINAL request, and
    a failure of even that propagates — with next_call invoked exactly once
    either way. The rewrite is applied to a copy, so a bookkeeping failure can
    never leak a half-applied rewrite into the fail-open dispatch.
    Once next_call HAS been dispatched, this wrapper re-raises anything that
    unwinds through it (downstream truth: provider error, KeyboardInterrupt,
    a post-dispatch failure) instead of masking it with None. This function
    NEVER classifies (side effects would double with on_llm_request) and
    NEVER returns a rewrite payload — the rewrite is the request dict handed
    to next_call.
    """
    if not callable(next_call):
        raise TypeError("on_llm_execution requires a callable next_call "
                        "(host execution-chain contract)")
    invoked = [False]

    def _dispatch(payload):
        # The ONLY path that touches next_call: marks itself first so no
        # handler ever dispatches a second time (host frame: single-use).
        invoked[0] = True
        return next_call(payload)

    try:
        cfg = _load_config()
        if not cfg.get("enabled", True):
            return _dispatch(request)

        # ---- PER-REQUEST KILL SWITCH (v1.0.8) ------------------------------
        # Same flag, same payload, re-read on EVERY callback: a request whose
        # client asked for no routing is dispatched untouched — no reapply of
        # an earlier decision, no rewrite, no probe, no extra row (the request
        # middleware already logged its disabled_by_request row). Placed ahead
        # of the decision-cache lookup on purpose: without this the stored
        # rewrite would silently REAPPEAR mid-turn and the kill switch would
        # only hold for the first call.
        if _kill_switch_off(request):
            return _dispatch(request)

        # Shallow copy: our rewrite must not half-mutate the host's payload
        # if bookkeeping fails mid-flight; the fail-open dispatch then hands
        # downstream the genuinely untouched original (addendum allows
        # "the request dict (or a modified copy)").
        req = dict(request) if isinstance(request, dict) else {}
        provider = str(context.get("provider") or "")
        model = str(context.get("model") or req.get("model") or "")
        turn_id = str(context.get("turn_id") or "")
        session_id = str(context.get("session_id") or "")
        api_call_count = context.get("api_call_count")
        mode = str(cfg.get("mode", "shadow")).lower()

        # Same store, same key, same eviction policy as on_llm_request.
        key = f"{session_id}:{turn_id}"
        with _DECISIONS_LOCK:
            now = time.monotonic()
            for k in [k for k, v in _DECISIONS.items()
                      if (now - v["ts"]) >= _DECISIONS_TTL_S]:
                del _DECISIONS[k]
            while len(_DECISIONS) > _DECISIONS_MAX:
                _DECISIONS.popitem(last=False)  # evict oldest (LRU)
            known = _DECISIONS.get(key)
            if known is not None:
                # Refresh recency so an active turn is not LRU-evicted mid-flight.
                _DECISIONS[key] = known
                _DECISIONS.move_to_end(key)

        # Unknown / expired turn: execution must not classify. llm_request owns
        # first-call classification (its api_call_count gate fires there); here
        # an unknown key means "no decision yet" — pass the payload untouched.
        if known is None or provider != known["provider"]:
            return _dispatch(request)

        decision = known["decision"]
        target = known.get("target")

        if decision == "free":
            if not target:  # defensive: never rewrite to ''
                return _dispatch(request)
            if mode == "live":
                req["model"] = target
                _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "turn_id": turn_id, "session_id": session_id,
                      "provider": provider, "model": model, "target": target,
                      "applied": True, "reapply": True, "tier": "free",
                      "api_call_count": api_call_count})
            return _dispatch(req)

        if decision == "flash":
            # FLASH re-apply: NO credit probe — flash never probes (the probe
            # stays free/paid-tier only; flash is the stable workhorse and a
            # flash 404 is covered by the host's retry/fail-open).
            if not target:  # defensive: never rewrite to ''
                return _dispatch(request)
            if mode == "live":
                req["model"] = target
                _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "turn_id": turn_id, "session_id": session_id,
                      "provider": provider, "model": model, "target": target,
                      "applied": True, "reapply": True, "tier": "flash",
                      "api_call_count": api_call_count})
            return _dispatch(req)

        if decision == "paid":
            if not target:  # defensive: never rewrite to ''
                return _dispatch(request)
            # Fail-closed on a dark paid lane; the probe is TTL-cached.
            if (mode == "live" and cfg.get("credit_probe", True)
                    and provider == "nous" and not _paid_lane_alive()):
                _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "turn_id": turn_id, "session_id": session_id,
                      "provider": provider, "model": model,
                      "applied": False, "credit_lost": True, "tier": "paid"})
                return _dispatch(request)
            if mode == "live":
                req["model"] = target
                _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "turn_id": turn_id, "session_id": session_id,
                      "provider": provider, "model": model, "target": target,
                      "applied": True, "reapply": True, "tier": "paid",
                      "api_call_count": api_call_count})
            return _dispatch(req)

        # decision "none" (below-gate / not-eligible / credit-blocked) and the
        # shadow branch of free/paid above: never rewritten, never re-logged.
        return _dispatch(request)
    except Exception as e:
        # Reached ONLY by failures in OUR pre-dispatch code: a downstream
        # exception unwinding through next_call re-raises past this handler
        # unwrapped (see _run_execution_chain's _DownstreamExecutionError
        # re-raise). Fail-open = hand the host the untouched request via one
        # last dispatch; even if THAT fails, next_call was still invoked
        # exactly once and the error propagates verbatim.
        logger.warning("kfchow-llm-value-router exec failed open: %s", e)
        if not invoked[0]:
            try:
                return _dispatch(request)
            except BaseException:
                logger.exception(
                    "kfchow-llm-value-router exec: final dispatch attempt "
                    "failed; propagating to the host.")
                raise
        # Bookkeeping failed after a dispatch already happened: the call went
        # out; let the exception surface rather than mask the outcome with None.
        raise


def _store_decision(key, provider, decision, target, conf):
    """Record the classified decision for a turn (called AFTER _classify).

    Every classified turn gets a row — "free"/"flash"/"paid" carry the
    resolved target; "none" covers below-gate, not-eligible and
    credit-blocked outcomes so a retry never re-classifies. All state access
    happens under _DECISIONS_LOCK.
    """
    try:
        with _DECISIONS_LOCK:
            _DECISIONS[key] = {"decision": decision, "target": target,
                               "conf": conf, "provider": provider,
                               "ts": time.monotonic()}
            _DECISIONS.move_to_end(key)
            while len(_DECISIONS) > _DECISIONS_MAX:
                _DECISIONS.popitem(last=False)  # evict oldest (LRU)
    except Exception:
        pass  # state must never break a turn


def register(ctx):
    ctx.register_middleware("llm_request", on_llm_request)
    ctx.register_middleware("llm_execution", on_llm_execution)
