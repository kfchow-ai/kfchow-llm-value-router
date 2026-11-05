# SAFETY — kfchow-llm-value-router

This plugin sits in an LLM gateway's request path. Its routing failure and
network behavior are specified here.

## Fail-open routing and lane eligibility

An exception, timeout, malformed vendor answer, or missing key in the routing
path leaves the original request unchanged. The plugin can fail to route; it
must not fail a turn. The execution wrapper calls the next middleware/provider
exactly once and does not swallow downstream errors.

Downgrade bands (free, flash, mid, including the routine free band) require
`route_providers` **and** the routing source eligibility. Escalation separately
requires `escalate_enabled`, `escalate_providers`, and escalation source
eligibility. With `source_mode: "any"` the provider is the source gate; with
`"allowlist"`, downgrades also require `route_models` and escalation requires
`escalate_models`. An explicit source list restricts both lanes. A provider
listed only for escalation can still escalate but cannot downgrade. A source
already on a free model is skipped before these checks.

With the example gates 0.35/0.65/0.80, free-pool verdicts use free below
0.35, flash at 0.35–<0.65, mid at 0.65–<0.80, and free at or above 0.80.
Paid-pool verdicts below the escalation gate stay on the source model by
default, **including below `flash_gate`**. The opt-in
`"paid_pool_routine_band": true` restores v1.0.10's pool-agnostic free-rung
routing below `flash_gate`. Escalation itself is unchanged by that flag.

Each turn is classified once. Both middleware paths re-check the stored
source's current lane eligibility before replay, so a provider or source
removed from scope stops receiving cached rewrites on the next call. In
`aaii_bands` mode a refreshed, different rung also blocks the stale cached
target; no re-classification occurs until a new turn. A temporarily removed
lane may resume an unexpired cached decision if re-enabled.

The per-request kill switch (`X-KFC-Router: off`, or `router: off` in the
payload) is read before the cache on every callback. A flagged request is not
rewritten, classified or credit-probed. The flag does not erase a previously
stored decision; a later unflagged callback may reuse it if still eligible.

## Shadow-first and credit probe

The shipped default is `mode: shadow`: the plugin logs decisions but does not
rewrite. Since v1.0.9, saved Desktop plugin-settings values for `enabled`,
`mode`, and `confidence_gate` override `lane2_config.json` on every call. A
panel change can therefore enable live routing without a JSON edit or gateway
restart. Other settings, including `ladder_mode` and
`paid_pool_routine_band`, are read from the JSON.

Only metered paid escalation needs the credit probe. It sends a two-token ping,
caches the result for five minutes per endpoint/key/model identity, and fails
closed: an unknown or dark paid lane blocks the premium rewrite. The downgrade
bands do not perform that credit probe. Subscription lanes do not use it.

## Optional AAII resolver

The legacy RSS resolver remains the default. `ladder_mode: "aaii_bands"` is an
explicit opt-in. The resolver uses the full board's `intelligence_score` and
`pricing.blended` (Avg), not the Lowest column; only AAII ≥ 25 rows with Avg
are candidates. Candidates must occur in the Nous model catalog (or have a
catalogued `:free` variant) and pass a bounded exact-ID completion probe before
being selected. Unknown outcomes retain the incumbent instead of promoting a
more expensive candidate. These probes can incur charges, including in `--dry-run`.

All four current Nous pins must exist before initialization. The resolver
retains a pin and reports an alert if no candidate qualifies, and records the
fold state across refreshes. Two consecutive operative 1.0× dominance failures
collapse a tier; two consecutive passes expand it. During the first failure a
retained, more-expensive tier is marked non-rewritable. A collapsed tier is
never used as a rewrite target; its band falls to a cheaper active tier or
stays on source, never to the premium pin. The stricter mid 0.5× and flash
0.25× checks and the >50% seven-day-median / >2× one-day price moves are
**report-only** and cannot cause a fold.

An existing `:free` pin without a qualifying replacement is retained; an
unknown or sub-25 AAII is flagged. If its exact board Avg is unknown, it is
reported as null rather than inventing a price. When a catalogued `:free`
variant is selected using a paid sibling's board row, `price_reference_model`
identifies that reference; its Avg is not a claim about the free variant's
actual charge.

Board fetch/parse/schema or catalog errors exit non-zero and leave the last
good rung file unchanged. The output is written atomically. If the AAII file
is missing or invalid, middleware cannot use its rungs and leaves the model
unchanged rather than falling back to unchecked pins. The legacy RSS output
remains informational; AAII mode reads the configured `rungs_path` on each
callback.

## Network calls (complete disclosure)

| Call | When | What is sent | Key |
|---|---|---|---|
| `POST https://api.typesafe.ai/v1/systemone` | First call of an eligible classified turn | Turn-shape features and, when enabled, capped task excerpt | `JEVI_API_KEY`, or the optional Jev creds-file fallback |
| `POST <nous>/v1/chat/completions`, two-token ping | Eligible paid escalation on a metered lane, cached up to five minutes | Literal `ping` | `NOUS_API_KEY` |
| `GET https://kfchow.com/llm-value-leaderboard/feed.xml` | Operator runs the legacy resolver | No task data | None |
| `GET https://kfchow.com/llm-value-leaderboard/data/models.json` | Operator runs the opt-in AAII resolver | No task data | None |
| `GET <nous>/v1/models` | AAII refresh, including dry-run | No task data | `NOUS_API_KEY` |
| `POST <nous>/v1/chat/completions`, up-to-16-token ping | AAII candidate verification, including dry-run | Exact candidate ID and literal `ping` | `NOUS_API_KEY` |

There is no analytics or telemetry upload beyond these calls. The board and
Nous URLs can be overridden as documented in the README.

## Reads and writes

- Reads the lane2 configuration from the configured Hermes home, the AAII rung
  file when opted in, and the relevant environment variables.
- `NOUS_API_KEY` is read from the process environment, not `.env` directly.
  The vendored Jev client reads `JEVI_API_KEY` first and can fall back to its
  own user-provided credentials file.
- Writes a local decision log under the configured Hermes home (0600); with
  `send_excerpt: true` this can include the task excerpt.
- The resolver writes the configured rung output file or its Hermes-home
  default only when explicitly run. AAII price history is kept in that document.
- Never logs or transmits key values except as the required authorization
  headers to the selected service endpoint.

## Privacy toggle: `send_excerpt`

Default `true` because the task excerpt is load-bearing for classification.
Set `send_excerpt: false` to omit task text from both the vendor request and
the local decision log; shape-only features still go to the vendor and
accuracy may fall.

## Other invariants

- Never rewrite in plugin shadow mode, cross providers, rewrite an already-free
  source, or rewrite to the same model.
- No classifier or credit call for a per-request kill-switch request.
- No real credentials or network in the offline pytest suite; DNS/sockets are
  blocked and per-test caches and state are isolated.
