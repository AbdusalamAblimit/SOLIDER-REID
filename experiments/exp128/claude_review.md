# Review: exp127 Stop Decision & exp128 Launch

## Findings (by severity)

### HIGH

**H1. exp127 was stopped 20 epochs early while still improving.**
- ep90 → ep100: +0.9 mAP / +0.7 R1 (59.6→60.5 / 72.4→73.1)
- Meanwhile, exp116 was *flat* in the same range (ep90=60.8, ep100=60.7)
- At the observed improvement rate, exp127's final ep120 result could plausibly land at ~61.0+, which would be competitive with or exceed exp116's 60.7
- The gap at ep100 (60.5 vs 60.7) is only **0.2 mAP** — within noise. The "clearly behind" framing overstates the evidence.

**H2. Gate collapse trajectory is unknown — only late-stage snapshots provided.**
- The log excerpt covers only ep92-102. SCRC activates after ep20, so there are ~70 missing epochs of gate dynamics.
- We don't know whether `scrc_g` was 0.999 from the start (immediate collapse, damning) or gradually increased from e.g. 0.5 (potentially indicating the model learned to prefer full completion, which could be informative rather than pathological).
- Without the early trajectory, the "gate collapsed" diagnosis is an observation at one time point, not a confirmed dynamic.

**H3. exp126 results are absent from the materials.**
- exp128 builds directly on exp126 ("exact top-k sparse routing"), yet no results are provided for exp126.
- We cannot verify that the base experiment warrants further investment. If exp126 itself underperformed, adding freeze30 is building on an unvalidated foundation.

### MEDIUM

**M1. "Collapsed to hard replace" may be a mischaracterization.**
- `scrc_g ≈ 0.999` means the gate is saturated, but whether this equals "hard replace" depends on the SCRC formulation:
  - If `out = orig + gate * residual`: gate=1.0 means *full additive residual*, not replacement. The original signal is fully preserved.
  - If `out = (1-gate) * orig + gate * bank`: gate=1.0 means true hard replace.
- The monitor claims equivalence with hard replace without citing the formulation. This distinction matters: if SCRC is additive-residual, then gate=1.0 means "always add full completion" which is functionally different from SCFR's hard replace, and the conclusion that "SCRC ≈ SCFR therefore SCRC adds nothing" doesn't follow.

**M2. exp128 design lacks clear success/fail criteria at intermediate checkpoints.**
- The design says "ep20 前后与 exp126 基本重合" and "epoch 30+ 后在 late-stage 验证上比 online teacher 更稳" but doesn't quantify what "more stable" means or define an early-stop threshold.
- Given the project's explicit stop-loss rules, exp128 should specify: "if at ep60 mAP < X, stop."

**M3. The "SCFR/SCKD series" comparison pool is inconsistent.**
- The monitor compares against exp116 (60.7/73.4), exp110 (60.8/73.4), exp114 (60.9/73.4).
- These are different methods — it's unclear whether they're all apples-to-apples comparisons. The gap between exp127 and these varies from 0.2 to 0.4 mAP, which is small.

### LOW

**L1. exp128 startup log confirms clean launch** — no issues found in the first epoch. Loss/speed are in line with previous experiments.

**L2. scrc_count is monotonically increasing within each epoch** (from ~1400 at iter 20 to ~1870 at iter 200). This is likely just a cumulative counter within the epoch, not an anomaly, but it should be confirmed.

---

## Verdict

**Partially support stop exp127 + start exp128, with reservations.**

- The decision to stop exp127 is **premature but defensible** — the gap is small (0.2 mAP) and the experiment was improving, but the gate saturation is a real concern that reduces confidence in the mechanism's value even if final numbers might close the gap.
- Starting exp128 is **reasonable in principle** but has a critical prerequisite: exp126's results must be verified as positive before building on it. If exp126 itself was neutral or negative, exp128 is wasted GPU time.

---

## Risks

1. **False negative on SCRC**: If SCRC is additive-residual (not replacement), gate saturation might actually be benign — the model decided full completion is always helpful. Stopping early means we never see whether this translates to a competitive final result.
2. **Building on unverified foundation**: Without exp126 results, exp128 could be a dead end from the start.
3. **Opportunity cost**: If the "exact sparse routing" line (exp125/126) is itself only marginally positive, layering freeze30 on top may not produce a publishable delta.

---

## Suggestions

1. **Before committing to exp128's full run**: Retrieve and document exp126's ep100/ep120 results. If exp126 didn't outperform exp116, reconsider whether the exact-topk direction is worth further investment.
2. **Clarify the SCRC gate formulation**: Read the actual code. If it's additive-residual, the "collapsed to hard replace" conclusion is wrong, and exp127's negative interpretation may need revision.
3. **Add concrete stop-loss to exp128**: e.g., "if ep60 mAP < 59.5, stop." This prevents another ambiguous late-stage stop decision.
4. **Consider letting exp127 finish in background if GPU allows**: 20 more epochs is ~20 minutes. The marginal cost is low and would provide a definitive result rather than an extrapolation.
