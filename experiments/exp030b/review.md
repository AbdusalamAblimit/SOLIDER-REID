# exp030b Review Report

## Round 1 — 2026-03-12

### Verdict: PASS

This is a config-only ablation experiment (no code changes). Only change vs exp030a:
- `POSE_PART_WEIGHT: 0.01` (was default 1.0 in exp030a)
- This makes `w_g = 0.9901 ≈ 1.0` and `w_p = 0.0099 ≈ 0`
- Global loss is effectively unscaled
- GCN still trains but with negligible loss weight

All code paths are identical to exp030a (already reviewed and passed).
No risk of regression or new bugs.
