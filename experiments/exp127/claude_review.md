## Findings (by severity)

### Medium

**M1. SCFR `replace_ratio` stat semantics changed — `support_complete_bank.py`**

The refactored `replace()` now delegates to `get_support()`, which computes `final_mask` **inclusive of the `valid_samples` filter**. Previously, `replace_ratio` was based on `replace_mask` (before the `valid_samples` guard); now it reports `support_ratio` from `final_mask` (after). Existing SCFR log comparisons across code versions will see a slight drop in `replace_ratio` numbers. Not a crash, but potentially confusing when reading historical logs.

### Low

**L1. Redundant `.detach()` — `skeleton_gcn.py:524`**

`delta = support_proto.detach() - kp_feats` — `support_proto` is already returned as `.detach()` from `get_support()`. The extra `.detach()` is harmless but unnecessary.

**L2. SCKD distillation loss is silently disabled during SCRC warmup — `processor.py`**

When `scrc_enabled=True`, the `elif not scfr_enabled and not scrc_enabled:` branch that computes the SCKD KD loss is **never reached**, even during the warmup period when `_scrc_active=False`. During warmup, neither the KD loss nor SCRC fusion runs — only bank updates occur. This is **identical to existing SCFR behavior** so it's consistent, but worth noting: SCRC warmup epochs have the skeleton branch trained purely by ID/triplet loss with no support-bank signal at all.

**L3. Full-batch `scrc_gate` forward on all (B, 17) positions — `skeleton_gcn.py:517-519`**

The gate network runs on all keypoints even though only `support_mask` positions contribute. Masked positions get zeroed out by `gate = raw_gate * support_mask.float()`. Functionally correct, minor wasted compute.

---

**无阻塞性问题。** The diff is safe to train with.

---

## Residual Risks

1. **Train/test distribution gap**: SCRC fusion only runs during training (requires labels + bank). At inference, low-visibility keypoints pass through unmodified. If the model learns to rely on the support-completed features, the gap could hurt test performance. This is inherent to the design, not a code bug.

2. **Mutual exclusion only enforced in processor**: The `skeleton_gcn.py` forward has independent SCFR and SCRC blocks with no cross-check. If someone constructs a `SkeletonGCNHead` outside the processor with both bank references set, both would run sequentially. The processor's `ValueError` guard prevents this in normal usage.

## Suggestions

1. Consider logging a message during SCRC warmup epochs (e.g., `[SCRC] warmup epoch {epoch}, bank-only`) so the log explicitly shows what's happening, rather than silent absence of both KD loss and fusion.
2. The gate bias init of −2.0 (sigmoid ≈ 0.12) is reasonable. If early training shows the gate saturating toward 0 or 1 too quickly, this is the first knob to tune.
