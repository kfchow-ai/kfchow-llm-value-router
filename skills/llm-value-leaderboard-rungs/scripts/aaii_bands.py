"""Opt-in AAII-band resolver. No code here is used by the legacy RSS rule.

Board prices are always pricing.blended. Selection is confined to the Nous
catalog and an exact-id, one-token probe. The returned document is informational
until a router explicitly selects ladder_mode=aaii_bands and reads it.
"""
import copy
from datetime import datetime, timedelta, timezone
import json
import math
import os
import statistics
import sys
import tempfile
import urllib.request

BOARD_URL = "https://kfchow.com/llm-value-leaderboard/data/models.json"
NOUS_BASE_URL = "https://inference-api.nousresearch.com/v1"
TIERS = ("premium", "mid", "flash", "free")
USER_AGENT = ("Mozilla/5.0 (compatible; kfchow-llm-value-router/1.0; "
              "+https://github.com/kfchow-ai/kfchow-llm-value-router)")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def validate_board(board):
    """Reject a broken board as a whole; individual unscored rows are skipped."""
    if not isinstance(board, dict) or not isinstance(board.get("models"), list) or not board["models"]:
        raise ValueError("board schema requires a nonempty models list")
    usable_schema = False
    for row in board["models"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError("board schema requires a model id on every row")
        pricing = row.get("pricing")
        if isinstance(pricing, dict) and "intelligence_score" in row and "blended" in pricing:
            usable_schema = True
    if not usable_schema:
        raise ValueError("board schema lacks intelligence_score or pricing.blended")
    return board["models"]


def fetch_board(url=BOARD_URL, timeout=30):
    # The public board's edge can reject Python's default User-Agent.
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_catalog(key, base_url=NOUS_BASE_URL, timeout=30):
    req = urllib.request.Request(base_url.rstrip("/") + "/models",
                                 headers={"Authorization": "Bearer " + key,
                                          "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows or any(
            not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
            for row in rows):
        raise ValueError("Nous model catalog schema is invalid or empty")
    return {row["id"] for row in rows}


def probe_model(model, key, base_url=NOUS_BASE_URL, timeout=15):
    """Probe the exact model id with one output token; failures disqualify it."""
    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}],
               "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return isinstance(body, dict) and isinstance(body.get("choices"), list) and bool(body["choices"])
    except Exception:
        return False


def _candidate_rows(rows, catalog):
    candidates = []
    for row in rows:
        score = _number(row.get("intelligence_score"))
        pricing = row.get("pricing")
        price = _number(pricing.get("blended")) if isinstance(pricing, dict) else None
        if score is None or score < 25 or price is None:
            continue
        model = row["id"]
        ids = [model]
        if not model.endswith(":free"):
            ids.append(model + ":free")
        for available_id in ids:
            if available_id in catalog:
                candidates.append({"model": available_id, "price_avg": price,
                                   "aaii": score, "board_id": model})
    return candidates


def _in_band(candidate, tier):
    score = candidate["aaii"]
    if tier == "premium":
        return score > 50
    if tier == "mid":
        return 40 <= score <= 50
    if tier == "flash":
        return 25 <= score < 40
    return candidate["price_avg"] == 0 or candidate["model"].endswith(":free")


def _select(tier, candidates, probe, probes):
    """Find the cheapest probe-green price, then maximize AAII within +10%."""
    rows = sorted((c for c in candidates if _in_band(c, tier)),
                  key=lambda c: (c["price_avg"], c["model"]))
    qualified = []
    cheapest = None
    for candidate in rows:
        price, model = candidate["price_avg"], candidate["model"]
        if cheapest is not None and price > cheapest * 1.10 + 1e-12:
            break
        if model not in probes:
            try:
                probes[model] = bool(probe(model))
            except Exception:
                probes[model] = False
        if probes[model]:
            if cheapest is None:
                cheapest = price
            qualified.append(candidate)
    if not qualified:
        return None
    return min(qualified, key=lambda c: (-c["aaii"], c["price_avg"], c["model"]))


def _pin_details(old, rows):
    """Look up retained-pin data without mistaking a paid sibling's price for :free."""
    model = old["model"]
    matches = {row["id"]: row for row in rows}
    row = matches.get(model)
    if row is not None:
        pricing = row.get("pricing")
        price = _number(pricing.get("blended")) if isinstance(pricing, dict) else None
        score = _number(row.get("intelligence_score"))
        reference = model if price is not None else old.get("price_reference_model")
    else:
        # A sibling may identify the same intelligence score, but its paid Avg
        # is not the actual price of an existing :free pin. Keep any previously
        # recorded board reference only; otherwise leave price_avg unknown.
        sibling = matches.get(model[:-5]) if model.endswith(":free") else None
        score = _number(sibling.get("intelligence_score")) if sibling else None
        reference = old.get("price_reference_model")
        pricing = sibling.get("pricing") if sibling else None
        price = (_number(pricing.get("blended"))
                 if sibling and reference == sibling["id"] and isinstance(pricing, dict) else None)
    return (price if price is not None else _number(old.get("price_avg")),
            score if score is not None else _number(old.get("aaii")), reference)


def _at(text):
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _price_alerts(tiers, history, now):
    moves = []
    for tier, item in tiers.items():
        current, model = _number(item.get("price_avg")), item["model"]
        if current is None:
            continue
        past = []
        reference = item.get("price_reference_model") or model
        for record in history.get(reference, []):
            if not isinstance(record, dict):
                continue
            at, price = _at(record.get("at")), _number(record.get("price_avg"))
            if at is not None and price is not None and now - timedelta(days=7) <= at < now:
                past.append((at, price))
        # One observation per prior calendar day; the current value is not in
        # its own baseline. Same-day refreshes cannot hide a seven-day jump.
        days = {}
        for at, price in sorted(past):
            if at.date() < now.date():
                days[at.date()] = price
        if days:
            median = statistics.median(days.values())
            ratio = current / median if median else (float("inf") if current else 1.0)
            if (median == 0 and current > 0) or (median > 0 and abs(current - median) / median > .5):
                moves.append({"tier": tier, "model": model, "kind": "median_7d",
                              "price_avg": current, "baseline": median,
                              "ratio": ratio if math.isfinite(ratio) else None})
        recent = [(at, price) for at, price in past if now - at <= timedelta(days=1)]
        if recent:
            _, previous = max(recent)
            if current > 2 * previous:
                moves.append({"tier": tier, "model": model, "kind": "jump_1d",
                              "price_avg": current, "baseline": previous,
                              "ratio": current / previous if previous else None})
    return moves


def _updated_history(history, rows, now):
    result = {}
    for model, entries in history.items():
        if not isinstance(model, str) or not isinstance(entries, list):
            continue
        valid = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            at, price = _at(entry.get("at")), _number(entry.get("price_avg"))
            if at is not None and price is not None and now - timedelta(days=7) <= at <= now:
                valid.append({"at": at.isoformat(), "price_avg": price})
        if valid:
            result[model] = valid
    for row in rows:
        pricing = row.get("pricing")
        price = _number(pricing.get("blended")) if isinstance(pricing, dict) else None
        if price is None:
            continue
        model = row["id"]
        # Keep one observation per day so repeated refreshes do not bias
        # the seven-day median. Price alerts were evaluated before replacing it.
        entries = [e for e in result.get(model, []) if _at(e["at"]).date() != now.date()]
        entries.append({"at": now.isoformat(), "price_avg": price})
        result[model] = entries
    return result


def resolve_aaii(board, catalog, probe, previous, now=None, source=BOARD_URL):
    """Return a new four-slot document without mutating board or prior state.

    Two consecutive failures collapse; two consecutive passes expand. While a
    first failure is pending, an expensive retained pin is temporarily marked
    non-rewritable, so hysteresis never exposes a dominated rewrite target.
    """
    rows = validate_board(board)
    if not isinstance(catalog, (set, list, tuple)) or any(not isinstance(x, str) for x in catalog):
        raise ValueError("invalid Nous catalog")
    catalog = set(catalog)
    if not isinstance(previous, dict) or not isinstance(previous.get("tiers"), dict):
        raise ValueError("prior four-tier state is required")
    old_tiers = copy.deepcopy(previous["tiers"])
    for index, tier in enumerate(TIERS):
        old = old_tiers.get(tier)
        if not isinstance(old, dict) or not isinstance(old.get("model"), str) or not old["model"].strip():
            raise ValueError("a nonempty previous %s pin is required" % tier)
        status = old.get("status", "active")
        if status not in ("active", "collapsed"):
            raise ValueError("invalid previous %s status" % tier)
        if ((status == "collapsed" and (index == 0 or old.get("collapsed_into") != TIERS[index - 1]))
                or (status == "active" and old.get("collapsed_into") is not None)):
            raise ValueError("invalid previous %s fold target" % tier)
        streak = old.get("fold_streak", 0)
        if not isinstance(streak, int) or isinstance(streak, bool) or streak < 0:
            raise ValueError("invalid previous %s fold_streak" % tier)
    history = previous.get("price_history", {})
    if not isinstance(history, dict):
        raise ValueError("invalid prior price history")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    candidates = _candidate_rows(rows, catalog)
    probes = {}
    selected = {tier: _select(tier, candidates, probe, probes) for tier in TIERS}
    result, alerts = {}, []
    for index, tier in enumerate(TIERS):
        old, candidate = old_tiers[tier], selected[tier]
        previous_price, previous_score, price_reference = _pin_details(old, rows)
        item = {"model": old["model"], "price_avg": previous_price, "aaii": previous_score,
                "price_reference_model": price_reference,
                "status": old.get("status", "active"),
                "collapsed_into": old.get("collapsed_into"),
                "reason": "", "fold_streak": old.get("fold_streak", 0),
                "fold_signal": old.get("fold_signal"), "rewrite_allowed": False,
                "candidate": copy.deepcopy(candidate)}
        if candidate is None:
            alerts.append({"tier": tier, "code": "no_qualified_candidate",
                           "message": "kept the current pin; no board/catalog/probe candidate qualified"})
        if tier == "free" and (previous_score is None or previous_score < 25) and candidate is None:
            alerts.append({"tier": tier, "code": "free_aaii_unknown_or_below_floor",
                           "message": "current free pin retained; its AAII is unknown or below 25"})

        if tier == "premium":
            if candidate is not None:
                item.update({k: candidate[k] for k in ("model", "price_avg", "aaii")},
                            price_reference_model=candidate["board_id"])
                item["reason"] = "lowest probed Avg plus 10% tolerance, highest AAII"
            else:
                item["reason"] = "no candidate; retained current premium pin"
            item.update(status="active", collapsed_into=None, fold_streak=0,
                        fold_signal=None, rewrite_allowed=True)
            result[tier] = item
            continue

        above_name = TIERS[index - 1]
        above = result[above_name]
        # If the immediate upper rung is dormant or temporarily unsafe, compare
        # with the next routable stronger rung. This preserves monotonic costs
        # throughout hysteresis rather than only after the second refresh.
        comparable = next((result[name] for name in reversed(TIERS[:index])
                           if result[name]["status"] == "active" and result[name]["rewrite_allowed"]), None)
        above_price = _number(comparable.get("price_avg")) if comparable else None
        if candidate is None or above_price is None:
            if above_price is None:
                alerts.append({"tier": tier, "code": "dominance_price_unknown",
                               "message": "kept current pin; cannot compare the upper Avg"})
            item.update(fold_streak=0, fold_signal=None,
                        reason="candidate or upper price unavailable; retained current pin")
            item["rewrite_allowed"] = (item["status"] == "active" and
                                       previous_price is not None and above_price is not None and
                                       previous_price <= above_price + 1e-12)
            if tier == "free" and item["status"] == "active" and item["model"].endswith(":free"):
                # Never discard the operator's existing free pin merely because
                # the board lacks a qualifying free row. Its :free price is zero;
                # unknown/below-floor AAII remains separately flagged above.
                item["rewrite_allowed"] = True
                item["reason"] = "retained existing :free pin; replacement or upper Avg unavailable"
            result[tier] = item
            continue

        fails = candidate["price_avg"] > above_price + 1e-12
        signal = "fail" if fails else "pass"
        streak = min(2, (old.get("fold_streak", 0) if old.get("fold_signal") == signal else 0) + 1)
        item.update(fold_streak=streak, fold_signal=signal)
        if item["status"] == "collapsed":
            if not fails and streak >= 2:
                item.update({k: candidate[k] for k in ("model", "price_avg", "aaii")},
                            price_reference_model=candidate["board_id"])
                item.update(status="active", collapsed_into=None, rewrite_allowed=True,
                            reason="expanded after two consecutive dominance passes")
            else:
                item.update({k: above[k] for k in ("model", "price_avg", "aaii", "price_reference_model")})
                item.update(status="collapsed", collapsed_into=above_name, rewrite_allowed=False,
                            reason=("folded; waiting for a second consecutive pass" if not fails
                                    else "folded; candidate still costs more than the stronger active tier"))
        elif fails and streak >= 2:
            item.update({k: above[k] for k in ("model", "price_avg", "aaii", "price_reference_model")})
            item.update(status="collapsed", collapsed_into=above_name, rewrite_allowed=False,
                        reason="collapsed after two consecutive Avg-price dominance failures")
        elif fails:
            item.update(status="active", collapsed_into=None,
                        rewrite_allowed=(previous_price is not None and previous_price <= above_price + 1e-12),
                        reason="first dominance failure; retained pin pending confirmation")
        else:
            item.update({k: candidate[k] for k in ("model", "price_avg", "aaii")},
                        price_reference_model=candidate["board_id"])
            item.update(status="active", collapsed_into=None, rewrite_allowed=True,
                        reason="lowest probed Avg plus 10% tolerance, highest AAII")
        result[tier] = item

    ratios = {}
    for tier, above_name, threshold in (("mid", "premium", .5), ("flash", "mid", .25)):
        candidate = selected[tier]
        upper = _number(result[above_name].get("price_avg"))
        price = candidate["price_avg"] if candidate else None
        ratios[tier] = {"threshold": threshold, "candidate_price_avg": price,
                        "above_price_avg": upper,
                        "would_collapse": (price > threshold * upper + 1e-12)
                                          if price is not None and upper is not None else None}
    moves = _price_alerts(result, history, now)
    for move in moves:
        alerts.append({"tier": move["tier"], "code": "shadow_price_move",
                       "message": "%s price movement for %s (report only)" % (move["kind"], move["model"])})
    return {"ladder_mode": "aaii_bands", "provider": "nous", "resolved_at": now.isoformat(),
            "source": source, "tiers": result, "alerts": alerts,
            "shadow": {"ratio_checks": ratios, "price_moves": moves},
            "price_history": _updated_history(history, rows, now)}


def initial_state(config, out_path):
    """Resume our last document or initialize from four existing Nous pins."""
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            old = json.load(fh)
        if not isinstance(old, dict):
            raise ValueError("existing rung file is not an object")
        if old.get("ladder_mode") == "aaii_bands":
            if old.get("provider") != "nous":
                raise ValueError("existing AAII rungs belong to a different provider")
            return old
    rung_map = (config.get("rungs") or {}).get("nous") or {}
    tiers = {}
    for name in TIERS:
        pin = rung_map.get(name)
        if isinstance(pin, list):
            pin = pin[0] if pin else None
        if not isinstance(pin, str) or not pin.strip():
            raise ValueError("configure a nonempty rungs.nous.%s pin before opting in" % name)
        tiers[name] = {"model": pin, "status": "active", "fold_streak": 0}
    return {"ladder_mode": "aaii_bands", "provider": "nous", "tiers": tiers, "price_history": {}}


def run_aaii(args, config, board_fetcher=fetch_board, catalog_fetcher=fetch_catalog,
             prober=probe_model):
    """Network/CLI shell; data errors never overwrite the last good document."""
    path = os.path.expanduser(os.path.expandvars(args.out))
    try:
        board = board_fetcher(args.board_url)
        validate_board(board)  # fail before catalog/probes on a malformed board
        previous = initial_state(config, path)
        key = os.environ.get("NOUS_API_KEY")
        if not key:
            raise ValueError("NOUS_API_KEY is required for catalog and one-token probes")
        catalog = catalog_fetcher(key, args.nous_base_url)
        doc = resolve_aaii(board, catalog,
                           lambda model: prober(model, key, args.nous_base_url),
                           previous, source=args.board_url)
        print(json.dumps({"tiers": doc["tiers"], "alerts": doc["alerts"],
                          "shadow": doc["shadow"]}, indent=2))
        if args.dry_run:
            return 0
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                             prefix=".llm-rungs-", delete=False) as fh:
                temporary = fh.name
                json.dump(doc, fh, indent=2, allow_nan=False)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)
        print("wrote %s" % path)
        return 0
    except Exception as exc:
        print("resolver failed: %s: %s — keeping last good file" %
              (type(exc).__name__, exc), file=sys.stderr)
        return 1
