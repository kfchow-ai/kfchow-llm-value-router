"""Unit tests for scripts/llm-rank-resolver.py (offline — fixture XML, no network).

The resolver previously had ZERO tests; these pin the parsing and the
KFChow value rule, including the traps that motivated them:
- ranking lives in ITEM <description>, not the channel description;
- fewer rows than top_n must exit 1 and keep the last good file.
"""
import importlib.machinery
import importlib.util
import json
import os
import sys
import urllib.error

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_REPO, "scripts", "llm-rank-resolver.py")
_loader = importlib.machinery.SourceFileLoader("llm_rank_resolver", _PATH)
_spec = importlib.util.spec_from_loader("llm_rank_resolver", _loader)
resolver = importlib.util.module_from_spec(_spec)
sys.modules["llm_rank_resolver"] = resolver
_loader.exec_module(resolver)


def _xml(*items, channel_desc="Channel-level prose blurb — no ranking here."):
    """Build a feed XML with channel description + N items.

    Each item arg is a description string (ranking lines, one per row).
    """
    items_xml = "".join(
        "<item><title>Snapshot %d</title>"
        "<description>%s</description></item>" % (i, d)
        for i, d in enumerate(items))
    return ("<rss><channel><title>Leaderboard</title>"
            "<description>%s</description>%s</channel></rss>"
            % (channel_desc, items_xml))


def _rows_xml():
    """Six ranking lines, deliberately with prices that split around $1.00."""
    lines = [
        "1. Z.ai: GLM 5.3 (Z-Ai) — Intel 45 — 211 tok/s — $0.47",
        "2. Moonshot: Kimi K3 (Moonshot) — Intel 42 — 180 tok/s — $2.50",
        "3. DeepSeek: V4 (DeepSeek) — Intel 40 — 150 tok/s — $0.55",
        "4. Qwen: Max 3 (Alibaba) — Intel 38 — 160 tok/s — $1.20",
        "5. Inclusion: Ling Sante (InclusionAI) — Intel 30 — 220 tok/s — $0.00",
        "6. OpenAI: GPT-slow (OpenAI) — Intel 55 — 90 tok/s — $9.00",
    ]
    return _xml(" | ".join(lines))


# --------------------------------------------------------------- parsing

def test_parse_normal_item_description():
    rows = resolver.parse(_rows_xml())
    assert len(rows) == 6
    assert rows[0] == {"rank": 1, "name": "Z.ai: GLM 5.3 (Z-Ai)",
                       "intel": 45, "speed": 211.0, "price": 0.47}
    assert rows[5]["name"] == "OpenAI: GPT-slow (OpenAI)"
    assert rows[5]["intel"] == 55


def test_parse_channel_description_only_decoy():
    """THE trap: a ranking-shaped CHANNEL blurb must not contaminate the parse
    when an item with the real ranking exists (parse scopes to <item> first).
    With no <item> at all, the documented fallback scans the whole XML (so a
    channel-only feed still degrades gracefully) — tested in
    test_parse_channel_only_feed_falls_back."""
    xml = _xml("1. Z.ai: GLM 5.3 (Z-Ai) — Intel 45 — 211 tok/s — $0.47",
               channel_desc="9. Fake: Decoy (Nobody) — Intel 99 — 1 tok/s — $0.01")
    rows = resolver.parse(xml)
    assert len(rows) == 1
    assert rows[0]["name"] == "Z.ai: GLM 5.3 (Z-Ai)"
    assert rows[0]["intel"] == 45
    # the decoy row (rank 9, intel 99) must never appear
    assert all(r["intel"] != 99 for r in rows)


def test_parse_channel_only_feed_falls_back():
    """Documented fallback: no <item> -> scan the whole XML (graceful
    degradation for a channel-only feed)."""
    xml = _xml(channel_desc="1. Z.ai: GLM 5.3 (Z-Ai) — Intel 45 — 211 tok/s — $0.47")
    rows = resolver.parse(xml)
    assert len(rows) == 1
    assert rows[0]["intel"] == 45


def test_parse_channel_description_ignored_when_item_present():
    """Even when BOTH descriptions contain ranking-shaped lines, only the
    item's description counts."""
    xml = _xml("1. Z.ai: GLM 5.3 (Z-Ai) — Intel 45 — 211 tok/s — $0.47",
               channel_desc="9. Fake: Decoy (Nobody) — Intel 99 — 1 tok/s — $0.01")
    rows = resolver.parse(xml)
    assert len(rows) == 1
    assert rows[0]["name"] == "Z.ai: GLM 5.3 (Z-Ai)"
    assert rows[0]["intel"] == 45


def test_parse_missing_fields_are_skipped():
    """Lines that do not match the full pattern are skipped, not crashes."""
    xml = _xml(" | ".join([
        "garbage line with no structure",
        "2. Z.ai: GLM 5.3 (Z-Ai) — Intel 45 — 211 tok/s — $0.47",
        "3. Broken model — Intel is NaN here"]))
    rows = resolver.parse(xml)
    assert len(rows) == 1
    assert rows[0]["rank"] == 2


def test_parse_empty_xml_returns_empty():
    assert resolver.parse("<rss><channel></channel></rss>") == []


# --------------------------------------------------------------- the value rule

def test_resolve_premium_is_highest_intel_in_top_n():
    """The KFChow value rule: premium = highest AAII among the top-5 by value.
    The rank-6 model has the HIGHEST intel overall but must NOT win."""
    rows = resolver.parse(_rows_xml())
    res = resolver.resolve(rows, top_n=5)
    assert res["premium"]["name"] == "Z.ai: GLM 5.3 (Z-Ai)"       # intel 45
    assert res["premium"]["intel"] == 45
    assert res["top_n_used"] == 5
    assert all(r["rank"] <= 5 for r in res["top_n"])


def test_resolve_mid_respects_price_ceiling():
    """Mid = same rule under the $1.00/Mtok ceiling; premium candidates above
    the ceiling are excluded even when they score higher."""
    rows = resolver.parse(_rows_xml())
    res = resolver.resolve(rows, top_n=5)
    # Affordable rows in top-5: GLM 5.3 ($0.47), V4 ($0.55), Ling Sante ($0.00)
    assert res["mid"]["name"] == "Z.ai: GLM 5.3 (Z-Ai)"           # intel 45 wins mid too
    # and the $2.50 Kimi K3 (intel 42) is NOT the mid pick
    assert res["mid"]["price"] <= resolver.MID_PRICE_CEILING


def test_resolve_mid_ceiling_falls_back_to_top_when_all_expensive():
    """If nothing in the top-N is under the ceiling, mid falls back to the top
    set rather than returning nothing."""
    xml = _xml(" | ".join([
        "1. A: ModelA (X) — Intel 40 — 100 tok/s — $5.00",
        "2. B: ModelB (Y) — Intel 30 — 100 tok/s — $7.00"]))
    res = resolver.resolve(resolver.parse(xml), top_n=2)
    assert res["mid"]["name"] == "A: ModelA (X)"


def test_resolve_empty_rows_returns_none():
    assert resolver.resolve([]) is None


def test_resolve_top_n_smaller_than_rows():
    rows = resolver.parse(_rows_xml())
    res = resolver.resolve(rows, top_n=2)
    assert res["top_n_used"] == 2
    assert [r["rank"] for r in res["top_n"]] == [1, 2]
    # premium still chosen WITHIN the top-2 only (Kimi K3 intel 42 < GLM 45)
    assert res["premium"]["name"] == "Z.ai: GLM 5.3 (Z-Ai)"


# --------------------------------------------------------------- exit-1 semantics

def _run_main(monkeypatch, xml_or_exc, out_path, top_n=5):
    """Run main() with fetch stubbed. Returns (exit_code, stdout+stderr, wrote)."""
    import io

    def fake_fetch(url=None, timeout=30):
        if isinstance(xml_or_exc, Exception):
            raise xml_or_exc
        return xml_or_exc

    monkeypatch.setattr(resolver, "fetch", fake_fetch)
    monkeypatch.setattr(sys, "argv", ["llm-rank-resolver.py"])
    buf_out, buf_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    wrote = {"ok": False}

    class TrackingFile:
        def __init__(self):
            self.data = ""

        def write(self, s):
            self.data += s

    def fake_dump(payload, fh, **kw):
        wrote["ok"] = True
        wrote["payload"] = payload

    monkeypatch.setattr(resolver.json, "dump", fake_dump)
    sys.stdout, sys.stderr = buf_out, buf_err
    try:
        rc = resolver.main(["--top-n", str(top_n), "--out", out_path])
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return rc, buf_out.getvalue() + buf_err.getvalue(), wrote


def test_missing_feed_exits_1_and_writes_nothing(monkeypatch, tmp_path):
    out = str(tmp_path / "llm_rungs.json")
    rc, printed, wrote = _run_main(monkeypatch,
                                   urllib.error.URLError("feed down"), out)
    assert rc == 1
    assert "resolver failed" in printed
    assert wrote["ok"] is False


def test_fewer_rows_than_top_n_exits_1_and_writes_nothing(monkeypatch, tmp_path):
    """THE keep-last-good-file contract: a short/degraded feed must exit 1 and
    NOT overwrite the last good rungs file."""
    out = str(tmp_path / "llm_rungs.json")
    short_feed = _xml(" | ".join([
        "1. Z.ai: GLM 5.3 (Z-Ai) — Intel 45 — 211 tok/s — $0.47",
        "2. DeepSeek: V4 (DeepSeek) — Intel 40 — 150 tok/s — $0.55"]))
    rc, printed, wrote = _run_main(monkeypatch, short_feed, out, top_n=5)
    assert rc == 1
    assert "keeping last good file" in printed
    assert wrote["ok"] is False


def test_successful_run_writes_payload_and_exits_0(monkeypatch, tmp_path):
    out = str(tmp_path / "llm_rungs.json")
    rc, printed, wrote = _run_main(monkeypatch, _rows_xml(), out, top_n=5)
    assert rc == 0
    assert wrote["ok"] is True
    payload = wrote["payload"]
    assert payload["source"] == resolver.FEED
    assert payload["mid"]["name"] == "Z.ai: GLM 5.3 (Z-Ai)"
    assert payload["premium"]["name"] == "Z.ai: GLM 5.3 (Z-Ai)"
    assert "the KFChow value rule" in payload["rule"]


def test_successful_run_prints_table(monkeypatch, tmp_path):
    out = str(tmp_path / "llm_rungs.json")
    rc, printed, _ = _run_main(monkeypatch, _rows_xml(), out, top_n=5)
    assert rc == 0
    assert "MID" in printed and "PREMIUM" in printed
    assert "GLM 5.3" in printed


def test_dry_run_writes_nothing(monkeypatch, tmp_path):
    import io
    xml = _rows_xml()
    monkeypatch.setattr(resolver, "fetch", lambda url=None, timeout=30: xml)
    wrote = {"ok": False}

    def fake_dump(payload, fh, **kw):
        wrote["ok"] = True

    monkeypatch.setattr(resolver.json, "dump", fake_dump)
    buf_out, buf_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf_out, buf_err
    try:
        rc = resolver.main(["--dry-run", "--out", str(tmp_path / "x.json")])
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    assert rc == 0
    assert wrote["ok"] is False


def test_env_overrides_feed_and_out(monkeypatch, tmp_path):
    """--feed-url / KFCHOW_FEED_URL / KFCHOW_RUNGS_OUT overrides."""
    seen = {}

    def fake_fetch(url=None, timeout=30):
        seen["url"] = url
        return _rows_xml()

    monkeypatch.setattr(resolver, "fetch", fake_fetch)
    monkeypatch.setattr(resolver.json, "dump", lambda payload, fh, **kw: None)
    rc = resolver.main(["--feed-url", "https://example.com/feed.xml",
                        "--out", str(tmp_path / "r.json")])
    assert rc == 0
    assert seen["url"] == "https://example.com/feed.xml"


def test_default_out_honours_hermes_home(monkeypatch):
    """Rungs path bases on HERMES_HOME, not a hardcoded home."""
    monkeypatch.setenv("HERMES_HOME", "/tmp/fake-home-x")
    p = resolver.default_out()
    assert p == "/tmp/fake-home-x/jev/llm_rungs.json"
