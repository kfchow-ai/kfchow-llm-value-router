"""kfchow-llm-value-router — per-turn model routing for Hermes (llm_request middleware).

WHAT IT DOES
    On the FIRST provider call of each turn, asks Jev (TypeSafe System One)
    whether the turn is routine enough to hand to the FREE tier. In `shadow`
    mode it only records the decision; in `live` mode it rewrites the request's
    model so the turn is served by the free pool instead of the paid lane.

    An ESCALATION path runs the other direction: when a turn starts on a "mid"
    rung and the classifier says it needs a strong model, the request can be
    escalated to the provider's premium rung — guarded by a credit probe so a
    metered provider with an exhausted balance never receives a doomed rewrite.

WHY MIDDLEWARE (not a cron, not a skill)
    `llm_request` is the documented cutover path: it sees the full provider
    kwargs and can replace them. A skill is advisory and a cron cannot act on
    a live turn.

SAFETY CONTRACT (every one of these is enforced in code, not convention)
    * Fail-open: any exception, timeout, or malformed answer returns None so
      Hermes proceeds with the ORIGINAL request, byte-identical.
    * First call of the turn only — tool-loop follow-ups keep whatever model the
      turn started with, so a turn never changes model mid-conversation.
    * Only eligible lanes are touched (route_providers/route_models and
      escalate_providers/escalate_models in config); a turn already on the
      free tier, on another provider, or with no model field is left alone.
    * The rewritten request keeps every other key, including the message list,
      so the provider payload stays valid.
    * A rewrite NEVER crosses providers: the rungs are resolved per provider,
      so the model id sent always belongs to the originating provider's API.
    * Mode is `shadow` until a human flips `mode` to `live` in the config.
    * Telemetry: decision logs are written to a local JSONL file only. Set
      `send_excerpt: false` to keep task text out of the vendor call AND the
      log (shape-only features still classify, with reduced accuracy).

Config: ~/.hermes/jev/lane2_config.json       Log: ~/.hermes/jev/lane2-live.jsonl
"""

import json
import logging
import os
import re
import sys
import threading
import time
import urllib.request

logger = logging.getLogger(__name__)

_HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
CONFIG_PATH = os.path.join(_HERMES_HOME, "jev", "lane2_config.json")
LOG_PATH = os.path.join(_HERMES_HOME, "jev", "lane2-live.jsonl")
ROUTING_QUESTION_ID = "routing_tier"
JEV_TIMEOUT_S = 10          # keep well under a turn's own latency budget
DEFAULT_GATE = 0.80
DEFAULT_EXCERPT_CHARS = 1200

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

# CREDIT GATE — checked before any escalation to a PAID model on a METERED
# provider. An account that has run out of credits refuses paid models (with
# HTTP 404 "requires available credits" in at least one provider's case, not
# 402). A gate that escalates to premium without checking would have turned
# that outage into failed turns. Fail-CLOSED: an unknown probe outcome means
# do not escalate — the turn keeps the affordable rung.
CREDIT_PROBE_MODEL = "z-ai/glm-5.3-flash"
CREDIT_TTL_S = 300          # don't probe on every turn
_CREDIT = {"ok": None, "checked_at": 0.0}
_CREDIT_LOCK = threading.Lock()
_STATE = {"last_turn": None}
_STATE_LOCK = threading.Lock()

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
    now = time.time()
    with _CREDIT_LOCK:
        if _CREDIT["ok"] is not None and (now - _CREDIT["checked_at"]) < CREDIT_TTL_S:
            return _CREDIT["ok"]
    alive = False
    try:
        key = _nous_api_key()
        if key:
            body = json.dumps({"model": CREDIT_PROBE_MODEL,
                               "messages": [{"role": "user", "content": "ping"}],
                               "max_tokens": 2, "temperature": 0}).encode()
            req = urllib.request.Request(
                _nous_base_url() + "/chat/completions",
                data=body, headers={"Content-Type": "application/json",
                                    "Authorization": "Bearer " + key})
            with urllib.request.urlopen(req, timeout=15) as r:
                json.loads(r.read())
            alive = True
    except Exception as e:
        logger.debug("credit probe failed (fail-closed, no escalation): %s", e)
    with _CREDIT_LOCK:
        _CREDIT["ok"] = alive
        _CREDIT["checked_at"] = now
    return alive


def _load_config():
    cfg = {"enabled": True, "mode": "shadow", "confidence_gate": DEFAULT_GATE,
           "route_providers": ["nous"],
           "route_models": [],
           "free_model": "",
           "escalate_enabled": True,
           "escalate_providers": [],
           "escalate_models": [],
           "premium_model": "",
           "escalate_confidence_gate": None,   # falls back to confidence_gate
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
        api_call_count = kwargs.get("api_call_count")

        # Every invocation records a decision with a skip_reason, so an operator
        # can always answer "why didn't this turn route?" from the log alone.
        def _skip(reason):
            _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "turn_id": turn_id,
                  "session_id": str(kwargs.get("session_id") or ""),
                  "provider": provider, "model": model, "pool": None,
                  "confidence": None, "gate": None, "eligible": False,
                  "mode": str(cfg.get("mode", "shadow")).lower(),
                  "applied": False, "skip_reason": reason})
            return None

        # First provider call of the turn only. Hermes counts the first call as
        # 1, so accept 0/1/None — tool-loop follow-ups are 2+ and must keep the
        # model the turn started with.
        if api_call_count not in (0, 1, None):
            return None  # follow-up call: intentionally unlogged (would flood)
        with _STATE_LOCK:
            if turn_id and _STATE["last_turn"] == turn_id:
                return None

        # The two paths have INDEPENDENT eligibility. Checking them as one chain
        # was a real bug: the free path's route_models filter rejected the mid
        # model before escalation ever ran, so escalation could never fire.
        # Evaluate each path on its own lane.
        free_src = (provider in (cfg.get("route_providers") or [])
                    and model in (cfg.get("route_models") or []))
        esc_src = (bool(cfg.get("escalate_enabled", True))
                   and provider in (cfg.get("escalate_providers") or [])
                   and model in (cfg.get("escalate_models") or []))
        if not (free_src or esc_src):
            if provider not in (cfg.get("route_providers") or []):
                return _skip("provider_not_eligible")
            return _skip("model_not_eligible")
        if any(h in model.lower() for h in _FREE_HINTS):
            return _skip("already_free_tier")
        if not request.get("model"):
            return _skip("no_model_in_request")

        feats = _features(request, excerpt_chars=int(cfg.get("excerpt_chars",
                                                             DEFAULT_EXCERPT_CHARS)),
                          include_excerpt=bool(cfg.get("send_excerpt", True)))
        pool, confidence = _classify(feats, cfg)
        with _STATE_LOCK:
            _STATE["last_turn"] = turn_id

        gate = float(cfg.get("confidence_gate", DEFAULT_GATE))
        free_model = _rung(cfg, provider, "free", "free_model")
        eligible = (free_src and pool == "free" and confidence is not None
                    and float(confidence) >= gate and bool(free_model))
        mode = str(cfg.get("mode", "shadow")).lower()

        # ---- ESCALATION (mid -> premium), the other direction -------------
        # Fires when the turn is on a MID rung and the classifier says it needs
        # a strong model. Guarded by a live credit probe: escalating into an
        # exhausted account would fail the turn, which is worse than a weaker
        # answer.
        esc_gate_raw = cfg.get("escalate_confidence_gate")
        esc_gate = float(esc_gate_raw if esc_gate_raw is not None
                         else cfg.get("confidence_gate", DEFAULT_GATE))
        premium = _rung(cfg, provider, "premium", "premium_model")
        esc_eligible = (
            esc_src
            and pool == "paid" and confidence is not None
            and float(confidence) >= esc_gate and bool(premium))
        credit_ok = None
        # Only a METERED provider needs the credit probe. A flat-rate
        # subscription lane has no per-call credit wall, and probing it with a
        # metered provider's key would be meaningless.
        if esc_eligible and cfg.get("credit_probe", True) and provider == "nous":
            credit_ok = _paid_lane_alive()
            if not credit_ok:
                esc_eligible = False

        log_feats = feats if cfg.get("send_excerpt", True) else {
            k: v for k, v in feats.items() if k != "task_excerpt"}
        _log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "turn_id": turn_id,
              "session_id": str(kwargs.get("session_id") or ""),
              "provider": provider, "model": model, "pool": pool,
              "confidence": confidence, "gate": gate, "eligible": eligible,
              "mode": mode, "applied": bool(eligible and mode == "live"),
              "skip_reason": None if (eligible or esc_eligible) else (
                  "below_gate" if pool == "free" else "pool_paid"),
              "escalate_eligible": esc_eligible,
              "escalate_applied": bool(esc_eligible and mode == "live"),
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


def register(ctx):
    ctx.register_middleware("llm_request", on_llm_request)
