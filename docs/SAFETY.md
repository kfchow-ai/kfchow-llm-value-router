# SAFETY — kfchow-llm-value-router

This plugin sits in the live request path of an LLM gateway, so its failure
behaviour is specified exactly, in code, before anything else.

## Fail-open contract (routing decisions)

Any exception, timeout, malformed config, malformed vendor answer, or missing
key in the ROUTING path returns `None`, and Hermes proceeds with the ORIGINAL
request byte-identical. The plugin can fail to route; it must never fail a
turn. Verified by tests: dead vendor, dead feed, corrupt config, unknown
provider — the request payload is untouched.

Concretely enforced:

- The whole middleware body is wrapped; an unexpected error logs and returns None.
- The vendor call is bounded by `jev_timeout_s` (default 10s) with **retries
  disabled** for the routing call — a slow vendor must not stall a turn.
- Eligibility filters run BEFORE any network call: only turns on a provider
  listed in `route_providers` (downgrade) or `escalate_providers`
  (escalation) are ever considered. v1.0.7 `source_mode` (default `"any"`)
  decides how the SOURCE model is treated: `"any"` = provider match only (any
  default model moves both ways on its own provider's rungs); `"allowlist"` =
  exact v1.0.6 model allowlists (`route_models` / `escalate_models`); or an
  explicit list of model ids. An unrecognised value falls back to
  `"allowlist"` (fewer turns touched, never more) with a one-time warning. A
  source already on a free model is skipped first, in every mode.
- **Per-request kill switch (v1.0.8):** a request that carries the disabling
  flag (`X-KFC-Router` header or a `router` body field) is skipped FIRST, in
  both middlewares and on every callback — no rewrite, no classifier call, no
  credit probe, one log row, no decision stored. An unknown flag value is
  ignored (fail-open: normal routing, no warning). The flag only turns routing
  OFF; it can never enable a config-disabled plugin.

## Shadow-first

The shipped default (manifest `config_schema`, example config, and the
code-level default in `_load_config()`) is `mode: shadow`. In shadow mode the
plugin decides and logs but NEVER rewrites a request. Going live is an
explicit human action: edit the config and set `"mode": "live"`.

## Credit probe: fail-CLOSED (escalation direction only)

The downgrade path (mid/flash -> free) needs no key and makes no credit call.
The FLASH tier (mid -> flash, v1.0.6) is also a downgrade: it fires when a
free-pool verdict lands in the flash band (`flash_gate` <= confidence <
`confidence_gate`, default 0.35–0.65) and a flash rung is configured;
otherwise the turn stays mid. Flash never probes credits and never needs a
key — the credit gate exists only for the ESCALATION direction, which is the
only move that costs money the turn was not already spending.
The ESCALATION path (mid -> premium, on a METERED provider) probes the paid
lane first with a 2-token request; the probe result is cached for 5 minutes
and is FAIL-CLOSED:

- probe succeeds -> escalation may proceed (still gated by confidence);
- probe fails or is unknown (no `NOUS_API_KEY`, network down, provider error)
  -> escalation is blocked and the turn keeps the affordable rung.

The key is read from the PROCESS ENVIRONMENT only
(`os.environ.get("NOUS_API_KEY")` — Hermes loads `~/.hermes/.env` into the
process env at startup). This code never opens `~/.hermes/.env` or any secrets
file itself.

## Network calls (complete disclosure)

| Call | When | What is sent | Key |
|------|------|--------------|-----|
| `POST https://api.typesafe.ai/v1/systemone` | first provider call of an eligible turn (every turn when routing is enabled) | turn shape features (message/tool counts, char counts, boolean flags) + the capped last-user-message excerpt when `send_excerpt: true` | `JEVI_API_KEY` (env first, then 0600 creds file fallback) |
| `POST <nous>/v1/chat/completions` with a 2-token ping | escalation candidate on a metered provider, at most every 5 minutes | the literal string `"ping"` | `NOUS_API_KEY` (env only) |
| `GET https://kfchow.com/llm-value-leaderboard/feed.xml` | when YOU run `scripts/llm-rank-resolver.py` (or a schedule you set up) | nothing (public feed fetch) | none |

No telemetry, no analytics, no phone-home beyond the calls above.

## Reads

- Its own config file: `~/.hermes/jev/lane2_config.json` (override the
  location by pointing `HERMES_HOME` elsewhere).
- Environment variables: `NOUS_API_KEY`, `JEVI_API_KEY` (both optional),
  `HERMES_HOME`, `JEVI_BASE_URL`, `JEVI_MODEL`, `KFCHOW_FEED_URL`,
  `KFCHOW_RUNGS_OUT`.
- The rungs file written by the resolver, if you wire it into the config.

## Writes

- `~/.hermes/jev/lane2-live.jsonl` — one JSON decision row per eligible turn
  (permissions 0600). May contain the task excerpt when `send_excerpt: true`.
- `~/.hermes/jev/llm_rungs.json` — written by `scripts/llm-rank-resolver.py`
  when you run it (not by the middleware).

## Privacy toggle: `send_excerpt`

Default `true` because the excerpt is load-bearing: measured on real traffic,
shape-only classification mislabels hard questions as free (0.31 free /
0.23 paid) and adding a capped last-user-message excerpt moved the same turns
to 1.00/1.00. Setting `send_excerpt: false`:

- omits `task_excerpt` from the vendor request (shape-only state is sent), AND
- omits it from the local log row.

Accuracy drops — expect more turns to stay on the paid/mid rung — but no task
text ever leaves the machine or reaches the log file.

## What it never does

- Never rewrites a request in shadow mode (default).
- Never re-classifies a follow-up call: each turn is classified once and
  the stored decision (including "no rewrite") is re-applied on later
  calls of the same turn.
- Never re-applies a stored decision when the request itself carries the
  per-request kill-switch flag (v1.0.8): the flag is re-read from the
  payload on every callback, so a flagged request — even mid-turn — is
  never rewritten, never classified and never probes credits.
- Never sends a rewrite to a different provider than the request came from.
- Never rewrites a turn that is already on a free model (`:free`, `free`,
  `space-bunny`), in any `source_mode`.
- Never rewrites a turn onto the model it is already on (identity guard,
  case-insensitive and vendor-prefix tolerant): a premium-default turn is
  never "escalated" to itself and never spends a credit probe doing so; it is
  logged `already_on_target`.
- Never downgrades a paid-pool verdict: below the escalation gate a paid
  turn stays mid; only free-pool turns drop to the flash band.
- Never reads `~/.hermes/.env` or any secrets file directly.
- Never logs or transmits the API key values.
- Never overwrites the last-good rungs file on a failed feed parse
  (exit 1 keeps the previous file).
