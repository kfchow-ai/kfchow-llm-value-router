---
name: llm-value-leaderboard-rungs
description: Resolve legacy leaderboard or opt-in AAII-band rungs.
author: KFChow <hotline@kfchow.com>
version: "1.0.0"
metadata:
  hermes:
    tags: [llm, model-routing, leaderboard]
---

# llm-value-leaderboard-rungs

Derive routing rungs (mid + premium model ids) from the KFChow AI Lab value
leaderboard feed instead of hardcoding model names. Powered by the KFChow AI
Lab value leaderboard (https://kfchow.com/llm).

## When to use

- You operate a model-routing gate (cost tiers: free / mid / premium) and want
  the targets derived from a maintained value ranking rather than hand-picked.
- Your leaderboard feed updates on a cadence and model ids rotate; hardcoded
  targets silently rot into 404s.
- You want a repeatable, rule-based selection you can re-run on a schedule.

## Procedure

1. Fetch the feed:
   `python3 skills/llm-value-leaderboard-rungs/scripts/llm-rank-resolver.py --dry-run`
   (override the feed with `--feed-url` or `KFCHOW_FEED_URL`; default
   https://kfchow.com/llm-value-leaderboard/feed.xml).
2. Parse the ITEM descriptions. The ranking lines live in each RSS item's
   `<description>`, NOT the channel-level description — scoping to `<item>`
   first is required or the parser silently matches the channel blurb and
   returns zero rows.
3. Apply the KFChow value rule: premium = the highest AAII score among the
   top-5 rows by value; mid = the same rule under a $1.00/Mtok price ceiling,
   so the two rungs are separated by price band rather than hand-picking.
4. Write the rungs JSON (default `~/.hermes/jev/llm_rungs.json`; override with
   `--out` or `KFCHOW_RUNGS_OUT`):
   `python3 skills/llm-value-leaderboard-rungs/scripts/llm-rank-resolver.py`
5. Wire your gate to the rungs file — e.g. point a routing plugin's
   per-provider `rungs` config at the resolved ids. Re-resolve on the feed's
   update cadence (this feed updates twice daily). Never hardcode model ids:
   free-tier ids expire silently and paid ids rotate.

## Optional AAII-band mode (Nous)

Legacy remains the default. To opt in, provide a lane2 JSON with
`"ladder_mode": "aaii_bands"`, a `rungs_path`, and four existing nonempty
`rungs.nous.{premium,mid,flash,free}` pins. Then run the same bundled script;
`--config` selects that JSON and `--ladder-mode aaii_bands` can override mode
for a single resolver invocation. The override does not change router config.

The resolver selects from all scored board rows using `pricing.blended` (Avg),
not Lowest; requires a Nous catalog match plus a bounded exact-ID completion probe; and
persists two-refresh dominance hysteresis, shadow ratio/price alerts, and
per-tier status. A collapsed tier is never a rewrite target. Missing candidates
retain pins and alert; bad board/schema/catalog data exits non-zero without
replacing the last good file. `--dry-run` does not write but still runs the
one 16-token-maximum completion probe per candidate, which may cost money. The AAII mode requires `NOUS_API_KEY`.
See the repository README for the exact bands and routing integration.

## Pitfalls (learned the hard way — do not re-learn them)

- **Item-description parse:** the ranking lives in ITEM `<description>`
  entries, not the channel description. Scope to `<item>` first.
- **Credit exhaustion looks like 404, not 402:** at least one metered provider
  refuses paid models with HTTP 404 "requires available credits" when the
  balance is empty. Gate any escalation to premium behind a live credit probe
  that fails closed (unknown state = do not escalate).
- **Heavy thinking models can return `content: null`:** some providers put the
  answer in a `reasoning_content` field for reasoning-heavy models and require
  session headers + a real user agent. Parse both fields before declaring a
  response empty, and never treat an empty parse as a routing signal.
- **Never cross providers on a rewrite:** the request's base_url/api_key
  belong to the ORIGINATING provider. Swapping only the model id sends the
  wrong id to the wrong endpoint. Resolve rungs per provider and keep a
  rewrite same-provider.
- **Free ids expire silently:** free-period model ids vanish without an error
  banner. Re-resolve from the feed on a cadence; treat a hard-coded free id as
  technical debt.

## Verification

```bash
# information only — no writes
python3 skills/llm-value-leaderboard-rungs/scripts/llm-rank-resolver.py --dry-run

# real run — writes the rungs JSON
python3 skills/llm-value-leaderboard-rungs/scripts/llm-rank-resolver.py
```

A run that cannot parse the feed exits 1 and KEEPS the last good file — a
partial parse must never overwrite a working set of rungs.

## Privacy notes

- Network: legacy fetches the public leaderboard feed. Opt-in AAII mode also
  fetches the public board and authenticated Nous catalog and sends bounded
  completion `ping` probes (up to 16 tokens) to Nous. It sends no task content in resolver calls.
- Reads: configured URLs and (in AAII mode) `NOUS_API_KEY`; writes: the rungs
  JSON path you choose, including persisted fold and board-price history.
