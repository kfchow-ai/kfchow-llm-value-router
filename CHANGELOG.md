# Changelog

## 1.0.11 — draft

- Add an opt-in Nous-only AAII-band resolver that uses the board's Avg prices;
  the legacy resolver remains the default.
- **D7 changes routing behavior relative to 1.0.10:** paid-pool turns below
  `flash_gate` remain on their source model by default. Set
  `paid_pool_routine_band: true` to restore the earlier pool-agnostic rule.
- Use a bounded completion probe with a 16-token maximum for AAII candidates.
  Unknown availability retains the incumbent rather than promoting a more
  expensive candidate; definite unavailability can be skipped.
- Keep this release in draft pending operator review and the documented live
  observation period; no implication of production deployment.
