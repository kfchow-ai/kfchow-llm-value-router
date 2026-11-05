#!/usr/bin/env python3
"""Resolve the mid + premium model rungs from the KFChow value leaderboard.

THE KFCHOW VALUE RULE: the premium target is the highest AAII score among the
TOP-5 models by value on https://kfchow.com/llm. Mid = the same rule under a
price ceiling, so the two rungs are separated by price band rather than by
hand-picking.

Why a resolver and not a hardcoded model: the ranking updates twice daily and
the model ids themselves rotate (free-period models expire silently). A
hardcoded premium target silently rots into a 404 — this re-derives it from
the live feed every run.

Feed format note: the channel-level <description> is prose; the ranking lives
in the ITEM <description> entries. Parsing must scope to <item> first or the
regex silently matches the channel blurb and returns zero rows.

Output: ~/.hermes/jev/llm_rungs.json (override with --out / KFCHOW_RUNGS_OUT)
Run:    python3 llm-rank-resolver.py            # resolve + write
        python3 llm-rank-resolver.py --dry-run  # print, do not write
Exit 0 on success, 1 if the feed can't be parsed (the caller keeps the last
good file — it must never be overwritten with a partial parse). The optional
ladder_mode=aaii_bands reads the full JSON board, checks the Nous catalog and
one 16-token-maximum completion probe per candidate, and persists four tiers with fold state; see aaii_bands.py.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

from aaii_bands import (BOARD_URL, NOUS_BASE_URL, fetch_board, fetch_catalog,
                        probe_model, resolve_aaii, run_aaii)

DEFAULT_FEED = "https://kfchow.com/llm-value-leaderboard/feed.xml"
FEED = os.environ.get("KFCHOW_FEED_URL", DEFAULT_FEED)
MID_PRICE_CEILING = 1.00   # $/Mtok — separates "mid" from "premium"
DEFAULT_TOP_N = 5


def default_out():
    """Default rungs path (Hermes-home aware)."""
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return os.path.join(home, "jev", "llm_rungs.json")


# Feed description lines look like:
#   3. Z.ai: GLM 5.3 (Z-Ai) — Intel 45 — 211 tok/s — $0.47
LINE = re.compile(
    r"(?P<rank>\d+)\.\s*(?P<name>.+?)\s*[—-]\s*Intel\s*(?P<intel>\d+)\s*[—-]\s*"
    r"(?P<speed>[\d.]+)\s*tok/s\s*[—-]\s*\$(?P<price>[\d.]+)")


def fetch(url=FEED, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "kfchow-llm-value-router/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors="replace")


def parse(xml):
    """Top-N by value as published. Order in the feed IS the value ranking.

    The channel <description> is prose; the ranking lives in the ITEM's
    <description>, so scope to <item> first or the regex silently matches
    the channel blurb and returns zero rows.
    """
    item = re.search(r"<item>(.*?)</item>", xml, re.S)
    blob = item.group(1) if item else xml
    desc = re.search(r"<description>(.*?)</description>", blob, re.S)
    text = desc.group(1) if desc else blob
    rows = []
    for m in LINE.finditer(text):
        rows.append({"rank": int(m.group("rank")),
                     "name": m.group("name").strip(),
                     "intel": int(m.group("intel")),
                     "speed": float(m.group("speed")),
                     "price": float(m.group("price"))})
    rows.sort(key=lambda r: r["rank"])
    return rows


def _slug(name):
    """'Z.ai: GLM 5.3 (Z-Ai)' -> a stable short key for logging."""
    n = re.sub(r"\(.*?\)", "", name).strip()
    return n


def resolve(rows, top_n=DEFAULT_TOP_N, mid_ceiling=MID_PRICE_CEILING):
    top = rows[:top_n]
    if not top:
        return None
    # Premium: highest AAII among the top-N by value (the KFChow value rule).
    premium = max(top, key=lambda r: r["intel"])
    # Mid: same rule under the price ceiling, so the rungs differ by price band.
    affordable = [r for r in top if r["price"] <= mid_ceiling] or top
    mid = max(affordable, key=lambda r: r["intel"])
    return {"premium": premium, "mid": mid, "top_n": top,
            "top_n_used": len(top), "mid_price_ceiling": mid_ceiling}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Resolve leaderboard model rungs (legacy RSS or opt-in AAII bands).")
    ap.add_argument("--dry-run", action="store_true", help="print the result, do not write")
    ap.add_argument("--feed-url", default=FEED,
                    help="leaderboard RSS/Atom feed URL (default: %s; env KFCHOW_FEED_URL)" % DEFAULT_FEED)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                    help="how many top-by-value rows to consider (default: %d)" % DEFAULT_TOP_N)
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: ~/.hermes/jev/llm_rungs.json; env KFCHOW_RUNGS_OUT)")
    ap.add_argument("--ladder-mode", choices=("legacy", "aaii_bands"), default=None,
                    help="override ladder_mode in the config; unset defaults to legacy")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(default_out()), "lane2_config.json"),
                    help="lane2 config containing ladder_mode and current Nous pins")
    ap.add_argument("--board-url", default=os.environ.get("KFCHOW_BOARD_URL", BOARD_URL),
                    help="AAII mode: leaderboard models.json URL")
    ap.add_argument("--nous-base-url", default=os.environ.get("NOUS_BASE_URL", NOUS_BASE_URL),
                    help="AAII mode: Nous OpenAI-compatible v1 base URL")
    args = ap.parse_args(argv)

    config, config_error = {}, None
    if args.ladder_mode != "legacy":
        try:
            with open(os.path.expanduser(args.config), encoding="utf-8") as fh:
                config = json.load(fh)
            if not isinstance(config, dict):
                raise ValueError("config must be a JSON object")
        except FileNotFoundError:
            config = {}
        except (OSError, ValueError) as exc:
            config, config_error = {}, exc
    # Legacy historically ignored lane2_config.json. Preserve that behavior,
    # except never overwrite an existing AAII document after config corruption.
    if config_error is not None:
        existing_out = os.path.expanduser(args.out or os.environ.get("KFCHOW_RUNGS_OUT", default_out()))
        prior_mode = None
        try:
            with open(existing_out, encoding="utf-8") as fh:
                prior = json.load(fh)
            prior_mode = prior.get("ladder_mode") if isinstance(prior, dict) else None
        except (OSError, ValueError):
            pass
        if args.ladder_mode == "aaii_bands" or prior_mode == "aaii_bands":
            print("resolver failed: invalid config: %s — keeping last good file" % config_error,
                  file=sys.stderr)
            return 1
    mode = args.ladder_mode or config.get("ladder_mode") or "legacy"
    if mode not in ("legacy", "aaii_bands"):
        print("resolver failed: unknown ladder_mode %r — keeping last good file" % mode, file=sys.stderr)
        return 1
    if args.out is None:
        args.out = (os.environ.get("KFCHOW_RUNGS_OUT") or
                    (config.get("rungs_path") if mode == "aaii_bands" else None) or default_out())
    if mode == "aaii_bands":
        return run_aaii(args, config, board_fetcher=fetch_board,
                        catalog_fetcher=fetch_catalog, prober=probe_model)

    try:
        xml = fetch(args.feed_url)
        rows = parse(xml)
    except Exception as e:
        print("resolver failed: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 1
    if len(rows) < args.top_n:
        print("resolver: only %d rows parsed (need %d) — keeping last good file"
              % (len(rows), args.top_n), file=sys.stderr)
        return 1

    res = resolve(rows, top_n=args.top_n)
    print("top-%d by value:" % res["top_n_used"])
    for r in res["top_n"]:
        print("   #%d %-34s intel=%-3d $%.2f" % (r["rank"], _slug(r["name"]),
                                                  r["intel"], r["price"]))
    print("\n  MID     -> %s (intel %d, $%.2f)"
          % (_slug(res["mid"]["name"]), res["mid"]["intel"], res["mid"]["price"]))
    print("  PREMIUM -> %s (intel %d, $%.2f)"
          % (_slug(res["premium"]["name"]), res["premium"]["intel"],
              res["premium"]["price"]))

    if args.dry_run:
        return 0
    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {"resolved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "source": args.feed_url,
               "rule": "the KFChow value rule: premium = highest AAII among top-N "
                       "by value; mid = same under $%.2f/Mtok ceiling" % MID_PRICE_CEILING,
               "mid": res["mid"], "premium": res["premium"], "top_n": res["top_n"]}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("\nwrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
