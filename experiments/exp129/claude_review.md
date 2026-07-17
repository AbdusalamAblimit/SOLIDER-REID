## Findings (by severity)

### CRITICAL — Residual target does NOT isolate "target dilution"; two confounded variables

The design claims "只改 CSRD 的 target 形式" but the code simultaneously changes:

1. **Target**: full → residual
2. **Loss function**: KL-divergence → Smooth L1
3. **Normalization**: softmax temperature τ → per-subset max-abs scaling

Moreover, the residual formulation is **mathematically degenerate** w.r.t. the dilution hypothesis. In `_distill_subset`:

```python
student_res = student_dist - base_det       # (dist_s - dist_base)
teacher_res = teacher_dist.detach() - base_det  # (dist_sc - dist_base)
point_loss = smooth_l1(student_res / scale, teacher_res / scale)
```

The gradient of this loss w.r.t. `dist_s` is:

```
dL/d(dist_s) = smooth_l1_grad((dist_s - dist_sc) / scale) × (1/scale)
```

**`dist_base` cancels out in the gradient.** The student receives exactly the same directional signal as if you just did `smooth_l1(dist_s / scale, dist_sc / scale)`. The "residual" framing changes nothing about what the student learns — only the scale factor (`max|dist_sc - dist_base|` instead of τ-based softmax) and the loss family (L1/L2 hybrid vs KL) differ.

This means exp129 is testing "Smooth-L1 with residual-magnitude normalization" vs "KL-div with temperature softmax", **not** "residual target vs full target". If it beats exp125, the explanation is almost certainly the loss function / normalization change, not dilution-resistance.

**Impact**: The experiment will run fine, but whatever result comes out cannot be attributed to the stated hypothesis.

---

### HIGH — Per-subset scale normalization is unstable early in training

```python
scale = teacher_res.abs().max().clamp(min=1e-6)
```

During warmup ramp (epochs 20–30), the SC bank is barely populated, so `dist_sc ≈ dist_base` for many anchors, making `teacher_res` near-zero. A max of, say, 0.0005 causes 2000× amplification. The `1e-6` clamp only protects against exact zero but not against "very small".

This could produce wildly inconsistent loss magnitudes per-anchor during the critical early-CSRD phase. The outer `anchor_weights` don't compensate because they don't know about per-subset scale.

---

### HIGH — Double emphasis on large-residual pairs

`pair_delta = |dist_sc - dist_base|` is used to compute `focus` weights via `_focus_from_delta`. In residual mode, the same `|dist_sc - dist_base|` is also the *magnitude of the target*. So large-residual pairs get:

1. Higher focus weight (from delta_top selection + alpha scaling)
2. Inherently larger loss (because the target is larger)

This double emphasis isn't present in full mode (where the target is the full teacher distribution, not the delta). It could cause the loss to be dominated by a handful of outlier-residual pairs.

---

### MEDIUM — Logged statistics won't match actual loss computation

The stats block computes:
```python
teacher_residual = (dist_t - dist_base).abs()   # full BxB matrix, absolute
student_residual = (dist_s.detach() - dist_base).abs()
```

But the actual loss uses:
- **Signed** residuals (not absolute)
- **Per-subset** normalization (not global)
- Only **selected** pairs (delta_top mask)

So `csrd_tr` / `csrd_gr` in logs will show a different picture than what the loss actually optimizes. This won't cause training errors but will make log-based debugging unreliable.

---

### MEDIUM — Config single-variable claim is technically correct but misleading

The only config change is `POSE_CSRD_TARGET_MODE: 'full' → 'residual'`. This satisfies the "one config knob" criterion. But that one knob activates a completely different code path (Smooth L1 with different normalization vs KL-div), so it's not a single-variable experiment in the scientific sense.

---

### LOW — Redundant detach

```python
base_det = base_dist.detach()
```

`base_dist` already comes from `_aggregate_teacher_dist(kp_feats)` which calls `.detach()` internally. Harmless but indicates the code was written without tracing the detach chain.

---

## Verdict

**Conditionally support continuing exp129**, but with a clear reframing:

- The experiment is safe to run (no blocking bugs, no crash risk, no data corruption)
- Early logs (epoch 1) look normal; CSRD hasn't activated yet (warmup=20)
- But **the result cannot be interpreted as evidence for or against the "target dilution" hypothesis**, because the residual base cancels in the gradient
- If exp129 outperforms exp125, the correct interpretation is: "Smooth-L1 with residual-magnitude scaling works better than KL-div with temperature softmax for CSRD"
- If exp129 underperforms, the correct interpretation is: "KL-div with softmax provides better distributional supervision than point-wise L1"

The experiment has informational value as a loss-function / normalization probe. It does not have informational value for the stated narrative.

---

## Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Per-subset scale explosion in early CSRD epochs (20-30) | High | Medium | Add floor: `clamp(min=0.01)` or use global running-average scale |
| Results misattributed to "residual target" when real cause is loss function | High | Certain | Document the mathematical equivalence now; design proper ablation later |
| Double emphasis causes outlier-pair dominance | Medium | Medium | Monitor per-batch loss variance; if huge, consider capping focus × residual |
| Log stats mislead analysis of residual dynamics | Medium | High | At minimum, note in monitor.md that logged stats don't reflect actual loss |

---

## Suggestions

1. **Document the confound immediately** in `exp129/design.md` or `monitor.md`: note that the gradient w.r.t. `dist_s` is independent of `dist_base`, so this experiment tests loss-function choice, not target-dilution resistance. This prevents future misinterpretation.

2. **If exp129 shows interesting results**, design a proper 2×2 ablation to disentangle:
   - (a) KL-div + full target (= exp125)
   - (b) KL-div + residual target (isolates target change)
   - (c) Smooth-L1 + full target (isolates loss change)
   - (d) Smooth-L1 + residual target (= exp129)

3. **Consider a scale floor** of `clamp(min=0.01)` instead of `1e-6` in the residual scale to avoid early-phase instability.

4. **Watch epochs 20–30 closely** — this is when CSRD activates and the scale normalization issue is most acute. If loss spikes or becomes erratic, the scale floor is the likely fix.
