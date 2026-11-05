"""Synthetic, offline AAII-band selection, persistence and routing tests."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
from email.message import Message

import pytest
from urllib.error import HTTPError
import aaii_bands

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("aaii_resolver", ROOT / "scripts" / "llm-rank-resolver.py")
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)
plugin_spec = importlib.util.spec_from_file_location("aaii_plugin", ROOT / "__init__.py")
pl = importlib.util.module_from_spec(plugin_spec)
plugin_spec.loader.exec_module(pl)

PREMIUM = "openai/gpt-6.1-terra"
MID = "google/gemini-4-argon"
FLASH = "minimax/minimax-m3.1-flash"
FREE = "reflection/beam-1.0-501b-a23b:free"
NOW = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
CATALOG = {PREMIUM, MID, FLASH, FREE, "fixture/unknown-free:free"}


@pytest.fixture
def board():
    return copy.deepcopy(json.loads((ROOT / "tests/fixtures/aaii_board_today.json").read_text()))


@pytest.fixture
def prior():
    return copy.deepcopy({"ladder_mode": "aaii_bands", "provider": "nous",
        "tiers": {
            "premium": {"model": PREMIUM, "price_avg": 1.9892, "aaii": 51,
                        "status": "active", "fold_streak": 0},
            "mid": {"model": MID, "price_avg": 3.6509, "aaii": 50,
                    "status": "active", "fold_streak": 0},
            "flash": {"model": FLASH, "price_avg": .1777, "aaii": 29,
                      "status": "active", "fold_streak": 0},
            "free": {"model": FREE, "price_avg": None, "aaii": 18,
                     "status": "active", "fold_streak": 0},
        }, "price_history": {}})


def run(board, prior, catalog=None, probe=None, now=NOW):
    # Every scenario gets fully independent nested pricing and rung dictionaries.
    return resolver.resolve_aaii(copy.deepcopy(board), set(CATALOG if catalog is None else catalog),
                                 probe or (lambda model: True), copy.deepcopy(prior), now=now)


def row(board, model):
    return next(item for item in board["models"] if item["id"] == model)


def model(model_id, score, avg, lowest=99, rank=None):
    return {"id": model_id, "rank": rank, "intelligence_score": score,
            "pricing": {"blended": avg, "blended_lowest": lowest}}


def test_today_second_failure_collapses_and_mid_band_never_targets_premium(board, prior, monkeypatch):
    first = run(board, prior)
    assert first["tiers"]["mid"]["status"] == "active"
    assert first["tiers"]["mid"]["fold_streak"] == 1
    assert first["tiers"]["mid"]["rewrite_allowed"] is False  # safe even before hysteresis trips
    second = run(board, first, now=NOW + timedelta(hours=1))
    expected = json.loads((ROOT / "tests/fixtures/aaii_today_expected_rungs.json").read_text())
    assert second == expected  # full synthetic rung document, including alerts/history
    mid = second["tiers"]["mid"]
    assert (mid["model"], mid["status"], mid["collapsed_into"], mid["fold_streak"]) == (PREMIUM, "collapsed", "premium", 2)
    assert mid["price_avg"] == 1.9892 and mid["aaii"] == 51 and mid["reason"]
    assert second["tiers"]["free"]["model"] == FREE
    assert second["tiers"]["free"]["price_avg"] is None  # no Avg for the exact :free id
    assert second["tiers"]["free"]["rewrite_allowed"] is True
    assert any(a["tier"] == "free" and a["code"] == "no_qualified_candidate" for a in second["alerts"])
    assert any(a["code"] == "free_aaii_unknown_or_below_floor" for a in second["alerts"])
    # The actual middleware consumes this very rung document, not a separate simulation.
    cfg = {"enabled": True, "mode": "live", "confidence_gate": .65,
           "escalate_confidence_gate": .80, "flash_gate": .35,
           "route_providers": ["nous"], "escalate_providers": ["nous"],
           "ladder_mode": "aaii_bands", "_ladder_document": copy.deepcopy(second)}
    monkeypatch.setattr(pl, "_load_config", lambda: cfg)
    monkeypatch.setattr(pl, "_classify", lambda *args: ("free", .70))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    out = pl.on_llm_request(request={"model": "source", "messages": []},
                            provider="nous", model="source", turn_id="today", session_id="s", api_call_count=1)
    assert out is not None and out["request"]["model"] == FLASH
    assert out["request"]["model"] != PREMIUM
    # This is the resulting rung document used by the middleware.
    print(json.dumps(second["tiers"], indent=2, sort_keys=True))


def test_two_consecutive_qualifying_refreshes_expand_dormant_mid(board, prior):
    collapsed = run(board, run(board, prior))
    assert collapsed["tiers"]["mid"]["status"] == "collapsed"
    board2 = copy.deepcopy(board)
    row(board2, MID)["pricing"]["blended"] = .90  # <= 0.5 x premium
    first = run(board2, collapsed, now=NOW + timedelta(days=1))
    assert first["tiers"]["mid"]["status"] == "collapsed"
    assert first["tiers"]["mid"]["fold_streak"] == 1
    second = run(board2, first, now=NOW + timedelta(days=2))
    assert second["tiers"]["mid"]["status"] == "active"
    assert second["tiers"]["mid"]["model"] == MID
    assert second["tiers"]["mid"]["price_avg"] == .90
    assert second["tiers"]["mid"]["fold_streak"] == 2


def test_premium_price_tolerance_uses_highest_aaii_and_unranked_rows(board, prior):
    board["models"].extend([model("fixture/cheap", 52, 1.0),
                            model("fixture/strong", 58, 1.09, rank=None),
                            model("fixture/too-expensive", 90, 1.11)])
    ids = CATALOG | {"fixture/cheap", "fixture/strong", "fixture/too-expensive"}
    result = run(board, prior, ids)
    assert result["tiers"]["premium"]["model"] == "fixture/strong"


def test_candidate_absent_from_lane_is_never_probed_or_chosen(board, prior):
    board["models"].append(model("fixture/not-in-nous", 99, .01))
    probes = []
    result = run(board, prior, probe=lambda mid: probes.append(mid) or True)
    assert result["tiers"]["premium"]["model"] == PREMIUM
    assert "fixture/not-in-nous" not in probes


def test_only_average_not_lowest_controls_pins(board, prior):
    board["models"].extend([model("fixture/p1", 52, 1.0, 9), model("fixture/p2", 60, 1.2, .01)])
    ids = CATALOG | {"fixture/p1", "fixture/p2"}
    first = run(board, prior, ids)
    lowest_only = copy.deepcopy(board)
    row(lowest_only, "fixture/p1")["pricing"]["blended_lowest"] = .0001
    row(lowest_only, "fixture/p2")["pricing"]["blended_lowest"] = 999
    assert run(lowest_only, prior, ids)["tiers"]["premium"]["model"] == first["tiers"]["premium"]["model"] == "fixture/p1"
    avg_changed = copy.deepcopy(board)
    row(avg_changed, "fixture/p2")["pricing"]["blended"] = 1.05
    assert run(avg_changed, prior, ids)["tiers"]["premium"]["model"] == "fixture/p2"


def test_failed_probe_moves_to_next_available_candidate(board, prior):
    board["models"].extend([model("fixture/unservable", 60, .5), model("fixture/servable", 55, .7)])
    probes = []
    result = run(board, prior, CATALOG | {"fixture/unservable", "fixture/servable"},
                 lambda mid: probes.append(mid) or mid != "fixture/unservable")
    assert result["tiers"]["premium"]["model"] == "fixture/servable"
    assert probes.count("fixture/unservable") == probes.count("fixture/servable") == 1


@pytest.mark.parametrize("missing", ["blended", "pricing", "null", "nonfinite"])
def test_missing_blended_is_not_a_candidate(board, prior, missing):
    bad = model("fixture/no-avg", 99, .01, .0001)
    if missing == "blended":
        bad["pricing"].pop("blended")
    elif missing == "pricing":
        bad.pop("pricing")
    elif missing == "null":
        bad["pricing"]["blended"] = None
    else:
        bad["pricing"]["blended"] = float("nan")
    board["models"].append(bad)
    probes = []
    result = run(board, prior, CATALOG | {"fixture/no-avg"}, lambda mid: probes.append(mid) or True)
    assert result["tiers"]["premium"]["model"] == PREMIUM
    assert "fixture/no-avg" not in probes


def test_flash_strict_quarter_ratio_is_shadow_only(board, prior):
    board = copy.deepcopy(board)
    row(board, FLASH)["pricing"]["blended"] = .60
    # Make mid fold into premium, so the strict comparison is .60 / 1.9892.
    result = run(board, run(board, prior))
    check = result["shadow"]["ratio_checks"]["flash"]
    assert check["threshold"] == .25 and check["would_collapse"] is True
    assert result["tiers"]["flash"]["status"] == "active"
    assert result["tiers"]["flash"]["model"] == FLASH


def test_unknown_free_aaii_keeps_current_free_pin_and_flags_it(board, prior):
    row(board, "reflection/beam-1.0-501b-a23b")["intelligence_score"] = None
    prior["tiers"]["free"]["aaii"] = None
    result = run(board, prior)
    free = result["tiers"]["free"]
    assert free["model"] == FREE and free["aaii"] is None
    assert free["price_avg"] is None and free["rewrite_allowed"] is True
    assert any(a["code"] == "free_aaii_unknown_or_below_floor" for a in result["alerts"])


def test_collapsed_mid_and_flash_fall_to_free_not_premium(board, prior):
    result = run(board, run(board, prior))
    result["tiers"]["flash"].update(model=PREMIUM, status="collapsed", collapsed_into="mid",
                                    rewrite_allowed=False, price_avg=1.9892, aaii=51)
    cfg = {"ladder_mode": "aaii_bands", "_ladder_document": copy.deepcopy(result)}
    assert pl._rung(cfg, "nous", "mid", "mid_model") == FREE
    assert pl._rung(cfg, "nous", "flash", "flash_model") == FREE
    result["tiers"]["free"]["rewrite_allowed"] = False
    cfg["_ladder_document"] = copy.deepcopy(result)
    assert pl._rung(cfg, "nous", "mid", "mid_model") is None


def test_no_candidate_keeps_previous_pin_and_alerts(board, prior):
    before = copy.deepcopy(prior)
    result = run(board, prior, catalog={PREMIUM, MID, FREE})  # flash not in Nous
    assert prior == before
    assert result["tiers"]["flash"]["model"] == FLASH
    assert any(a["tier"] == "flash" and a["code"] == "no_qualified_candidate" for a in result["alerts"])
    assert all(result["tiers"][tier]["model"] for tier in ("premium", "mid", "flash", "free"))


@pytest.mark.parametrize("broken", [ValueError("board unavailable"), {"models": "not-a-list"}, {"models": []}])
def test_board_fetch_or_schema_failure_keeps_existing_file_and_exits_nonzero(monkeypatch, tmp_path, broken):
    out = tmp_path / "rungs.json"
    original = b'{"last_good": "must not change"}\n'
    out.write_bytes(original)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"ladder_mode": "aaii_bands"}))
    def fetch(*args, **kwargs):
        if isinstance(broken, Exception):
            raise broken
        return broken
    monkeypatch.setattr(resolver, "fetch_board", fetch)
    monkeypatch.setattr(resolver, "fetch_catalog", lambda *args: pytest.fail("catalog should not be read"))
    monkeypatch.setenv("NOUS_API_KEY", "synthetic")
    assert resolver.main(["--config", str(cfg), "--out", str(out)]) != 0
    assert out.read_bytes() == original


def test_shadow_price_move_over_7_day_median(board, prior):
    prior["price_history"][PREMIUM] = [
        {"at": (NOW - timedelta(days=day)).isoformat(), "price_avg": 1.0}
        for day in range(1, 8)]
    result = run(board, prior)
    alerts = result["shadow"]["price_moves"]
    assert any(a["model"] == PREMIUM and a["kind"] == "median_7d" and a["ratio"] > 1.5 for a in alerts)
    assert result["tiers"]["premium"]["model"] == PREMIUM


def test_shadow_price_jump_over_twice_in_one_day(board, prior):
    prior["price_history"][PREMIUM] = [{"at": (NOW - timedelta(hours=12)).isoformat(), "price_avg": .9}]
    result = run(board, prior)
    assert any(a["model"] == PREMIUM and a["kind"] == "jump_1d" for a in result["shadow"]["price_moves"])


def test_probe_uses_bounded_tokens_and_exact_model(monkeypatch):
    calls = []
    def fake_open(request, timeout):
        calls.append((request.full_url, json.loads(request.data), timeout))
        return io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}')
    monkeypatch.setattr(resolver.urllib.request, "urlopen", fake_open)
    assert resolver.probe_model("fixture/model:free", "synthetic", "https://unit.invalid/v1")
    assert calls[0][1]["max_tokens"] == 16
    assert calls[0][1]["model"] == "fixture/model:free"


def test_low_token_rejecting_model_is_servable_with_16(monkeypatch):
    requests = []
    def fake_open(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        if payload["max_tokens"] < 16:
            raise HTTPError(request.full_url, 400, "minimum 16", Message(), io.BytesIO(b"minimum 16"))
        return io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}')
    monkeypatch.setattr(aaii_bands.urllib.request, "urlopen", fake_open)
    assert aaii_bands.probe_model("fixture/paid", "synthetic", "https://unit.invalid/v1") is True
    assert requests[0]["max_tokens"] == 16


def test_unknown_cheapest_keeps_distinct_incumbent(board, prior):
    board["models"].extend([model("fixture/cheap-uncertain", 51, .1),
                            model("fixture/expensive-alive", 58, .2)])
    ids = CATALOG | {"fixture/cheap-uncertain", "fixture/expensive-alive"}
    attempts = []
    def probe(model_id):
        attempts.append(model_id)
        return None if model_id == "fixture/cheap-uncertain" else True
    result = run(board, prior, ids, probe)
    assert result["tiers"]["premium"]["model"] == PREMIUM
    assert result["tiers"]["premium"]["candidate"] is None
    assert "fixture/expensive-alive" not in attempts
    assert any(a["code"] == "candidate_probe_uncertain" and
               "fixture/cheap-uncertain" in a["models"] for a in result["alerts"])


def test_config_selects_aaii_mode_and_persists_fold_streak(monkeypatch, tmp_path, board):
    out = tmp_path / "rungs.json"
    cfg = tmp_path / "lane2_config.json"
    cfg.write_text(json.dumps({"ladder_mode": "aaii_bands", "rungs_path": str(out),
                               "rungs": {"nous": {"premium": PREMIUM, "mid": MID,
                                                   "flash": FLASH, "free": FREE}}}))
    monkeypatch.setenv("NOUS_API_KEY", "synthetic")
    monkeypatch.setattr(resolver, "fetch_board", lambda *args: copy.deepcopy(board))
    monkeypatch.setattr(resolver, "fetch_catalog", lambda *args: set(CATALOG))
    monkeypatch.setattr(resolver, "probe_model", lambda *args: True)
    assert resolver.main(["--config", str(cfg)]) == 0
    first = json.loads(out.read_text())
    assert first["tiers"]["mid"]["fold_streak"] == 1
    assert resolver.main(["--config", str(cfg)]) == 0
    second = json.loads(out.read_text())
    assert second["tiers"]["mid"]["status"] == "collapsed"
    assert second["tiers"]["mid"]["fold_streak"] == 2


@pytest.mark.parametrize("score,expected", [(24.9, None), (25, "flash"), (39.9, "flash"),
                                           (40, "mid"), (50, "mid"), (50.01, "premium")])
def test_aaii_band_boundaries(score, expected):
    candidate = {"model": "fixture/paid", "price_avg": 1.0, "aaii": score}
    # _candidate_rows enforces the 25 floor before _in_band.
    matching = [band for band in ("premium", "mid", "flash") if aaii_bands._in_band(candidate, band)]
    assert matching == ([] if expected is None else [expected])


def test_full_cli_request_shapes_with_fake_http(monkeypatch, tmp_path, board):
    out, config = tmp_path / "rungs.json", tmp_path / "config.json"
    config.write_text(json.dumps({"ladder_mode": "aaii_bands", "rungs_path": str(out),
        "rungs": {"nous": {"premium": PREMIUM, "mid": MID, "flash": FLASH, "free": FREE}}}))
    calls = []
    def fake_open(request, timeout):
        calls.append(request)
        if request.full_url == "https://board.invalid/models.json":
            payload = copy.deepcopy(board)
        elif request.full_url == "https://nous.invalid/v1/models":
            payload = {"data": [{"id": mid} for mid in CATALOG]}
        elif request.full_url == "https://nous.invalid/v1/chat/completions":
            body = json.loads(request.data)
            assert body["max_tokens"] == 16 and body["model"] in CATALOG
            payload = {"choices": [{"message": {"content": "ok"}}]}
        else:
            pytest.fail("unexpected URL: " + request.full_url)
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setenv("NOUS_API_KEY", "synthetic")
    monkeypatch.setattr(resolver.urllib.request, "urlopen", fake_open)
    assert resolver.main(["--config", str(config), "--board-url", "https://board.invalid/models.json",
                          "--nous-base-url", "https://nous.invalid/v1"]) == 0
    doc = json.loads(out.read_text())
    assert doc["tiers"]["mid"]["fold_streak"] == 1
    probes = [r for r in calls if r.full_url.endswith("/chat/completions")]
    assert probes and all(json.loads(r.data)["max_tokens"] == 16 for r in probes)
    assert all(r.get_header("Authorization") == "Bearer synthetic"
               for r in calls if r.full_url.startswith("https://nous.invalid/"))


def test_opt_in_requires_a_key_without_overwriting_previous(monkeypatch, tmp_path, board, prior):
    out, config = tmp_path / "rungs.json", tmp_path / "config.json"
    out.write_text(json.dumps(prior))
    original = out.read_bytes()
    config.write_text(json.dumps({"ladder_mode": "aaii_bands"}))
    monkeypatch.setattr(resolver, "fetch_board", lambda *args: copy.deepcopy(board))
    monkeypatch.setattr(resolver, "fetch_catalog", lambda *args: pytest.fail("no key means no catalog call"))
    assert resolver.main(["--config", str(config), "--out", str(out)]) != 0
    assert out.read_bytes() == original


def test_aaii_dry_run_probes_but_does_not_write_or_advance_state(monkeypatch, tmp_path, board, prior):
    out, config = tmp_path / "rungs.json", tmp_path / "config.json"
    out.write_text(json.dumps(prior))
    original = out.read_bytes()
    config.write_text(json.dumps({"ladder_mode": "aaii_bands"}))
    probes = []
    monkeypatch.setenv("NOUS_API_KEY", "synthetic")
    monkeypatch.setattr(resolver, "fetch_board", lambda *args: copy.deepcopy(board))
    monkeypatch.setattr(resolver, "fetch_catalog", lambda *args: set(CATALOG))
    monkeypatch.setattr(resolver, "probe_model", lambda model, *args: probes.append(model) or True)
    assert resolver.main(["--config", str(config), "--out", str(out), "--dry-run"]) == 0
    assert probes and out.read_bytes() == original


def test_catalog_error_preserves_last_good_file(monkeypatch, tmp_path, board, prior):
    out, config = tmp_path / "rungs.json", tmp_path / "config.json"
    out.write_text(json.dumps(prior))
    original = out.read_bytes()
    config.write_text(json.dumps({"ladder_mode": "aaii_bands"}))
    monkeypatch.setenv("NOUS_API_KEY", "synthetic")
    monkeypatch.setattr(resolver, "fetch_board", lambda *args: copy.deepcopy(board))
    def bad_catalog(*args):
        raise ValueError("synthetic malformed catalog")
    monkeypatch.setattr(resolver, "fetch_catalog", bad_catalog)
    assert resolver.main(["--config", str(config), "--out", str(out)]) != 0
    assert out.read_bytes() == original


def test_atomic_replace_error_preserves_last_good_file(monkeypatch, tmp_path, board, prior):
    out, config = tmp_path / "rungs.json", tmp_path / "config.json"
    out.write_text(json.dumps(prior))
    original = out.read_bytes()
    config.write_text(json.dumps({"ladder_mode": "aaii_bands"}))
    monkeypatch.setenv("NOUS_API_KEY", "synthetic")
    monkeypatch.setattr(resolver, "fetch_board", lambda *args: copy.deepcopy(board))
    monkeypatch.setattr(resolver, "fetch_catalog", lambda *args: set(CATALOG))
    monkeypatch.setattr(resolver, "probe_model", lambda *args: True)
    def fail_replace(*args):
        raise OSError("synthetic atomic replace failure")
    monkeypatch.setattr(aaii_bands.os, "replace", fail_replace)
    assert resolver.main(["--config", str(config), "--out", str(out)]) != 0
    assert out.read_bytes() == original
    assert list(tmp_path.glob(".llm-rungs-*")) == []


def test_standalone_skill_resolver_is_identical():
    assert (ROOT / "scripts/llm-rank-resolver.py").read_bytes() == \
           (ROOT / "skills/llm-value-leaderboard-rungs/scripts/llm-rank-resolver.py").read_bytes()
    assert (ROOT / "scripts/aaii_bands.py").read_bytes() == \
           (ROOT / "skills/llm-value-leaderboard-rungs/scripts/aaii_bands.py").read_bytes()


def test_free_suffix_is_catalogued_and_probed_as_the_exact_pin(board, prior):
    board["models"].append(model("fixture/scored", 30, 0.0))
    probes = []
    result = run(board, prior, CATALOG | {"fixture/scored:free"},
                 lambda target: probes.append(target) or True)
    assert result["tiers"]["free"]["model"] == "fixture/scored:free"
    assert "fixture/scored:free" in probes and "fixture/scored" not in probes


def test_equal_aaii_ties_break_on_price_then_model_id(board, prior):
    board["models"].extend([model("fixture/z", 55, 1.0), model("fixture/a", 55, 1.0),
                            model("fixture/higher-price", 55, 1.05)])
    result = run(board, prior, CATALOG | {"fixture/z", "fixture/a", "fixture/higher-price"})
    assert result["tiers"]["premium"]["model"] == "fixture/a"


def test_fold_failures_must_be_consecutive(board, prior):
    first_failure = run(board, prior)
    assert first_failure["tiers"]["mid"]["fold_streak"] == 1
    passing = copy.deepcopy(board)
    row(passing, MID)["pricing"]["blended"] = .90
    intervening_pass = run(passing, first_failure, now=NOW + timedelta(hours=1))
    assert intervening_pass["tiers"]["mid"]["status"] == "active"
    next_failure = run(board, intervening_pass, now=NOW + timedelta(hours=2))
    assert next_failure["tiers"]["mid"]["status"] == "active"
    assert next_failure["tiers"]["mid"]["fold_streak"] == 1
    assert run(board, next_failure, now=NOW + timedelta(hours=3))["tiers"]["mid"]["status"] == "collapsed"


def test_missing_candidate_resets_pending_streak(board, prior):
    first = run(board, prior)
    missing = run(board, first, CATALOG - {MID}, now=NOW + timedelta(hours=1))
    assert missing["tiers"]["mid"]["model"] == MID
    assert missing["tiers"]["mid"]["fold_streak"] == 0
    next_failure = run(board, missing, now=NOW + timedelta(hours=2))
    assert next_failure["tiers"]["mid"]["status"] == "active"
    assert next_failure["tiers"]["mid"]["fold_streak"] == 1


def test_no_initial_pin_never_creates_incomplete_document(monkeypatch, tmp_path, board):
    out, cfg = tmp_path / "rungs.json", tmp_path / "config.json"
    cfg.write_text(json.dumps({"ladder_mode": "aaii_bands", "rungs": {"nous": {"premium": PREMIUM}}}))
    monkeypatch.setenv("NOUS_API_KEY", "synthetic")
    monkeypatch.setattr(resolver, "fetch_board", lambda *args: copy.deepcopy(board))
    monkeypatch.setattr(resolver, "fetch_catalog", lambda *args: pytest.fail("bad pins must fail first"))
    assert resolver.main(["--config", str(cfg), "--out", str(out)]) != 0
    assert not out.exists()


def test_opt_in_config_reads_document_and_never_replays_a_folded_mid(monkeypatch, tmp_path, board, prior):
    board = copy.deepcopy(board)
    row(board, MID)["pricing"]["blended"] = .9
    active = run(board, prior)
    assert active["tiers"]["mid"]["model"] == MID
    path, config_path = tmp_path / "rungs.json", tmp_path / "lane2_config.json"
    path.write_text(json.dumps(active))
    config_path.write_text(json.dumps({"enabled": True, "mode": "live", "confidence_gate": .65,
        "escalate_confidence_gate": .8, "route_providers": ["nous"],
        "escalate_providers": ["nous"], "ladder_mode": "aaii_bands", "rungs_path": str(path)}))
    monkeypatch.setattr(pl, "CONFIG_PATH", str(config_path))
    classified = []
    monkeypatch.setattr(pl, "_classify", lambda *args: classified.append(1) or ("free", .7))
    monkeypatch.setattr(pl, "_log", lambda rec: None)
    request = {"model": "source", "messages": []}
    context = {"provider": "nous", "model": "source", "session_id": "s", "turn_id": "fold", "api_call_count": 1}
    assert pl.on_llm_request(request=request, **context)["request"]["model"] == MID
    # A new atomically published document makes the cached mid target stale.
    folded = copy.deepcopy(active)
    folded["tiers"]["mid"].update(model=PREMIUM, status="collapsed", collapsed_into="premium",
                                  price_avg=1.9892, aaii=51, rewrite_allowed=False)
    path.write_text(json.dumps(folded))
    context["api_call_count"] = 2
    assert pl.on_llm_request(request=request, **context) is None
    seen = []
    result = pl.on_llm_execution(request=request, next_call=lambda r: seen.append(dict(r)) or r, **context)
    assert result["model"] == "source" and seen == [request]
    assert classified == [1]
    # A fresh turn uses the next cheaper active rung, never the premium pin.
    context.update(turn_id="new", api_call_count=1)
    assert pl.on_llm_request(request=request, **context)["request"]["model"] == FLASH


@pytest.mark.parametrize("damage", ["missing_price", "negative_price", "bad_status", "bad_fold", "bad_streak"])
def test_malformed_aaii_document_never_routes(board, prior, damage):
    doc = run(board, prior)
    mid = doc["tiers"]["mid"]
    if damage == "missing_price":
        del mid["price_avg"]
    elif damage == "negative_price":
        mid["price_avg"] = -1
    elif damage == "bad_status":
        mid["status"] = "unknown"
    elif damage == "bad_fold":
        mid.update(status="collapsed", collapsed_into="free", rewrite_allowed=False)
    else:
        mid["fold_streak"] = -1
    cfg = {"ladder_mode": "aaii_bands", "_ladder_document": copy.deepcopy(doc)}
    assert pl._rung(cfg, "nous", "premium", "premium_model") is None
    assert pl._rung(cfg, "nous", "mid", "mid_model") is None


def test_missing_aaii_document_does_not_fall_back_to_unchecked_config(monkeypatch, tmp_path):
    path = tmp_path / "missing.json"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"ladder_mode": "aaii_bands", "rungs_path": str(path),
                                  "rungs": {"nous": {"mid": PREMIUM}}}))
    monkeypatch.setattr(pl, "CONFIG_PATH", str(config))
    cfg = pl._load_config()
    assert pl._rung(cfg, "nous", "mid", "mid_model") is None
    assert pl._rung(cfg, "nous", "premium", "premium_model") is None
