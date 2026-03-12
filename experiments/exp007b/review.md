# exp007b/c Review Report

## Round 1 — 2026-03-12

### Verdict: PASS

Config-only experiments (no code changes). Only difference vs exp007a:
- exp007b: `GLOBAL_LOSS_SCALE: 0.25` (was 0.5)
- exp007c: `GLOBAL_LOSS_SCALE: 0.75` (was 0.5)

All code paths identical to exp007a (already reviewed and passed).
Output dirs are unique and isolated. No risk of regression.
