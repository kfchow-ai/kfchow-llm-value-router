# kfchow-llm-value-router
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23186874.svg)](https://doi.org/10.5281/zenodo.23186874)

> **Cite:** KFChow AI Lab (2026). *kfchow-llm-value-router: Jev-driven per-turn model routing for Hermes Agent, resolved from the KFChow AI Lab value leaderboard.* Zenodo. https://doi.org/10.5281/zenodo.23186874

**Powered by the KFChow AI Lab value leaderboard (https://kfchow.com/llm)**

A Hermes plugin that routes each turn to the cheapest model that is *good
enough* for it — and can escalate the hard turns to a premium rung — with a
strict fail-open contract: if anything goes wrong, the request proceeds
byte-identical. Ships in **shadow mode**: it observes and logs decisions until
you deliberately flip it to live.

Developer: KFChow <hotline@kfchow.com> · License: MIT

## What it does

On the **first provider call of each turn** (tool-loop follow-ups are ignored
so a turn never changes model mid-conversation), the plugin:

1. checks eligibility — only providers/models you list in the config are ever
   touched, and a turn already on a free-tier model is never re-routed;
2. asks a vendor classifier (TypeSafe "Jev", a 3–10s bounded call) whether the
   turn is routine enough for the FREE tier;
3. in `shadow` mode (default): records the decision and leaves the request
   untouched; in `live` mode: rewrites the request's model to the provider's
   free rung;
4. conversely, when a turn starts on a **mid** rung and the classifier says it
   needs a strong model, the ESCALATION path can rewrite to the provider's
   premium rung — guarded by a credit probe that **fails closed** so an
   exhausted metered account never receives a doomed rewrite.

Model rungs come from the KFChow value leaderboard (https://kfchow.com/llm)
via the bundled resolver script — never hardcoded — because rankings update
twice daily and free-tier model ids expire silently.

## Safety & privacy (full details in [docs/SAFETY.md](docs/SAFETY.md))

- **Fail-open**: any error/timeout/malformed answer returns `None`; the
  original request is untouched. The plugin can fail to route; it never fails
  a turn.
- **Shadow by default**; live requires a deliberate config edit by a human.
- **Credit probe fails closed**: escalation to a paid premium rung only fires
  when the metered lane answers a live probe; unknown state = no escalation.
- **A rewrite never crosses providers**: rungs resolve per provider.
- **Local-only telemetry**: decision logs go to `~/.hermes/jev/lane2-live.jsonl`
  (0600), nothing is uploaded.
- **`send_excerpt` toggle**: the classifier is sent turn shape features
  (counts, flags) plus, by default, a capped 1200-char excerpt of the last
  user message — this is load-bearing (shape-only scoring mislabels hard
  questions, measured 0.31 vs 1.00 accuracy). Set `"send_excerpt": false` in
  the config to keep task text out of BOTH the vendor call and the log;
  expect reduced routing accuracy.
- **Optional keys, never forced**: `JEVI_API_KEY` (vendor classifier) and
  `NOUS_API_KEY` (credit probe) are optional environment variables. Without
  them the plugin degrades safely — without the Jev key it routes nothing
  (fail-open); without the Nous key it simply never escalates on metered
  providers. Keys come from the process environment; the vendored client has
  one fallback — a plain `~/.hermes/secrets/jev.creds` file **you own** (line
  1 = key, 0600) — used only when `JEVI_API_KEY` is unset. Nothing is ever
  read from `.env` files.

## Install (plugin)

**Option A — from the plugin catalog** (once listed):

```bash
hermes plugins install kfchow-llm-value-router
```

**Option B — manual, from this repo:**

```bash
git clone https://github.com/kfchow-ai/kfchow-llm-value-router.git
hermes plugins install ./kfchow-llm-value-router
```

Then set the optional keys (either is enough to start; both enable full
behaviour) and copy the example config:

```bash
export JEVI_API_KEY=...    # vendor classifier — omit to run resolver-only
export NOUS_API_KEY=...    # credit probe — omit to disable escalation on metered providers
mkdir -p ~/.hermes/jev
cp examples/lane2_config.example.json ~/.hermes/jev/lane2_config.json
```

Config lives at `~/.hermes/jev/lane2_config.json` (honours `HERMES_HOME`).
Resolve your rungs before flipping anything to live:

```bash
python3 scripts/llm-rank-resolver.py        # writes ~/.hermes/jev/llm_rungs.json
python3 scripts/llm-rank-resolver.py --dry-run   # print only
```

Then fill the resolved ids into the `rungs` section per provider. Stays in
`shadow` until you set `"mode": "live"`.

## Install (skill — the method, without the plugin)

The repo also ships the leaderboard-to-rungs method as a standalone Hermes
skill, so you can use the resolver without the routing middleware:

```bash
# via a skills tap
hermes skills tap add kfchow-ai/kfchow-llm-value-router
hermes skills install kfchow-ai/kfchow-llm-value-router/skills/llm-value-leaderboard-rungs

# or single-repo install
hermes skills install kfchow-ai/kfchow-llm-value-router/skills/llm-value-leaderboard-rungs
```

## The KFChow value rule

premium = the highest AAII score among the **top-5 by value** on
https://kfchow.com/llm; mid = the same rule under a **$1.00/Mtok** price
ceiling, so the rungs are separated by price band rather than hand-picking.
The resolver re-derives them from the live feed every run and exits 1 —
keeping the last good file — whenever the feed cannot be parsed.

## FAQ — the traps we hit so you don't have to

**The feed parses but returns zero rows.**
The ranking lives in each RSS ITEM's `<description>`, not the channel-level
description (that's prose). Scope your parser to `<item>` first, then read
the description inside it.

**Paid models suddenly fail with HTTP 404.**
On at least one metered provider, credit exhaustion is a **404** ("requires
available credits"), not a 402. Don't build escalation logic that assumes an
empty balance looks like a payment-required error — probe the paid lane and
fail closed before escalating.

**A strong "thinking" model returns `content: null`.**
Reasoning-heavy models on some providers return the answer in a
`reasoning_content` field with `content: null`, and the endpoint requires a
session header (`x-opencode-session`) plus a real user agent. Read both
fields and send the headers before declaring the response empty.

**My free model id stopped working with no warning.**
Free-period ids expire silently. Never hardcode them — resolve from the
leaderboard feed on a cadence (it updates twice daily).

**Can I route between different providers in one rewrite?**
No — never. The request's `base_url`/`api_key` belong to the ORIGINATING
provider; swapping only the model id sends the wrong id to the wrong
endpoint. Rungs are resolved per provider and rewrites stay same-provider.

## Files

```
├── plugin.yaml                # manifest (provides_middleware: [llm_request])
├── __init__.py                # the middleware
├── scripts/jev_client.py      # vendored vendor client (env-first key)
├── scripts/llm-rank-resolver.py  # feed -> rungs resolver
├── skills/llm-value-leaderboard-rungs/   # the method as a standalone skill
├── examples/lane2_config.example.json   # config template (shadow)
├── docs/SAFETY.md             # fail-open / fail-closed / disclosure contract
└── tests/                     # offline test suite (no network, no keys)
```

## Development

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

All tests are offline: the vendor call is exercised through a transport test
seam, the feed through fixture XML. Nothing in the suite touches a network or
a key.

## License

MIT — see [LICENSE](LICENSE).
