## Findings (by severity)

### 1. MEDIUM — Missing validation for `residual_kl` + no support teacher

The processor has:
```python
if csrd_target_mode == 'residual' and not csrd_support_teacher:
    raise ValueError(...)
```
but no equivalent guard for `'residual_kl'`. Without a support teacher, `dist_t == dist_base`, making all teacher residual logits exactly zero → uniform teacher distribution → KL loss pushes student to uniform → actively harmful.

**Not blocking for exp130** (config has `SUPPORT_TEACHER: True`), but should be added for safety.

### 2. MEDIUM — Tau calibration mismatch (not a bug, but a known risk)

`tau=0.10` was tuned for full distances (typical range 0.5–1.5 for normalized embeddings). Residual distances `dist_sc - dist_base` are likely much smaller (0.01–0.1 range), so `residual/tau` produces logit ranges of ~0.1–1.0 instead of ~5–15. This makes the softmax distribution softer (closer to uniform) than in full mode.

This is **inherent to the experimental design**, not fixable without introducing a second variable (tau tuning). If exp130 underperforms, this is one possible explanation separate from the dilution hypothesis.

### 3. LOW — Redundant detach on base_dist

```python
base_det = base_dist.detach()
```
`base_dist` already comes from `_aggregate_teacher_dist(kp_feats)` which calls `.detach()`. Harmless, cosmetic only.

### 4. LOW — Log statistics don't reflect residual semantics

`csrd_tgap`, `csrd_sgap`, `csrd_tr`, `csrd_gr` are all computed on full (absolute) distributions, not residuals. This is a pre-existing issue from exp125, not new. Monitor interpretation must account for this.

---

## Key question: Does the base cancel in the gradient?

**No. This is the critical difference from exp129.** In exp129 (Smooth L1), the gradient w.r.t. `dist_s` was:
```
dL/d(dist_s) = smooth_l1_grad((dist_s - dist_sc) / scale) / scale
```
where `dist_base` fully cancels.

In exp130 (KL with softmax), the gradient is:
```
dL/d(dist_s_i) = (p_t_i - p_s_i) / tau
```
where `p_t = softmax(-(dist_t - dist_base)/tau)` and `p_s = softmax(-(dist_s - dist_base)/tau)`. Since softmax is a **nonlinear** function of the logits, the base vector does NOT cancel — it shifts relative logit magnitudes and therefore changes the resulting probability distribution. The student receives a genuinely different supervisory signal than in full mode.

**Conclusion**: `residual_kl` is a mathematically valid, non-degenerate test of the target dilution hypothesis. exp129's review criticism does not apply here.

---

## Single-variable verification

| Dimension | exp125 | exp130 | Changed? |
|-----------|--------|--------|----------|
| Target mode | `full` | `residual_kl` | **Yes** |
| Loss family | KL-div | KL-div | No |
| Temperature | 0.10 | 0.10 | No |
| Normalization | softmax | softmax | No |
| Pair routing | delta_top, ratio=0.25 | delta_top, ratio=0.25 | No |
| Focus alpha | 1.0 | 1.0 | No |
| CSRD weight | 0.5 | 0.5 | No |
| Warmup | 20 | 20 | No |
| SC teacher params | all identical | all identical | No |
| Backbone/batch/main loss | all identical | all identical | No |

**Confirmed: true single-variable experiment.**

---

## Verdict

**Support launching exp130.** It is:
- A genuine single-variable change vs exp125
- Mathematically non-degenerate (softmax prevents base cancellation)
- The cleanest available test of the target dilution hypothesis
- Free of blocking bugs
- Properly validated in processor.py

---

## Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Soft teacher distribution due to small residual magnitudes + fixed tau | Medium | High | Monitor `csrd` loss magnitude vs exp125; if near-zero through epoch 40+, tau mismatch is confirmed |
| Missing validation for residual_kl without support teacher | Medium | Low (config is correct) | Add one-line guard after launch |
| Results near-equal to exp125 → inconclusive on dilution | Medium | Medium | Would still narrow the hypothesis space; next step would be tau sweep for residual mode |
| Log stats computed on full distributions, not residuals | Low | Certain | Note in monitor.md when interpreting; actual loss behavior may differ from logged stats |

---

## Suggestions

1. **Add the missing validation guard** (one line in processor.py):
   ```python
   if csrd_target_mode == 'residual_kl' and not csrd_support_teacher:
       raise ValueError('POSE_CSRD_TARGET_MODE=residual_kl requires POSE_CSRD_SUPPORT_TEACHER=True')
   ```

2. **Watch CSRD loss magnitude in epochs 25–40** — if `csrd` values are an order of magnitude smaller than exp125's at the same epoch, the tau mismatch is eating the signal. This would be a valid finding (not a bug), but worth noting early.

3. **If exp130 shows interesting results**, the logical next ablation is tau calibration: try `tau=0.01` or `tau=0.05` for residual mode to compensate for smaller logit ranges, keeping everything else fixed.

4. **Document in monitor.md** that logged `csrd_tgap`/`csrd_sgap` reflect full-distribution statistics, not the residual signal the loss actually optimizes.
