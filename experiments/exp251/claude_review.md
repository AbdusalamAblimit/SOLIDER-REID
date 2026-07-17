# Claude Review -- exp251 & exp252 (Joint Review)

**Date**: 2026-04-07
**Reviewer**: Claude Opus 4.6 (broad review)
**Scope**: exp251 (Tiny multi-stage PSG+PAA) and exp252 (Small multi-stage PSG+PAA)

---

## (a) design.md Review

### exp251 design.md

- **Motivation**: Clear. Extending PSG from Stage 3-only to Stage 2+3 to provide earlier pose information. References exp073 (multi-stage PAA alone = -0.5 mAP) as prior evidence. Multi-stage PSG has NOT been tested before -- this is a valid new test.
- **Single variable principle**: PARTIALLY VIOLATED. Two variables change simultaneously vs exp246b: (1) PSG stages from [-1] to [-2,-1], AND (2) POSE_ADDITIVE_ADAPTER from False to True. If results change, we cannot attribute the cause to multi-stage PSG vs PAA vs their combination. However, this is a config-only experiment and both components already exist in code, so the risk is acceptable if the purpose is to test their combined effect. A follow-up ablation separating the two would be needed for the paper.
- **Hypothesis**: Stated clearly ("Stage 2 mid-level features also benefit from pose modulation; PSG+PAA complementary").
- **Expected results**: Reasonable range (+0.5% success, +/-0.2% neutral, -1% failure).
- **Innovation concern**: This is a config combination experiment, NOT a new mechanism. Per CLAUDE.md rules, combination experiments should NOT be the main line. However, since this tests multi-stage injection (never tested before for PSG), it provides useful ablation evidence for the paper narrative. Acceptable as a supporting experiment.

### exp252 design.md

- **Motivation**: Same hypothesis on Small backbone. Valid cross-backbone verification.
- **Note**: exp252 design says "Small has 18 Stage 3 blocks + 2 Stage 2 blocks" -- this is correct per swin_small depths=(2,2,18,2). Stage 2 (index 2) has 18 blocks, Stage 3 (index 3) has 2 blocks. Wait -- this is BACKWARDS in the design doc. Stage index 2 has 18 blocks and Stage index 3 has 2 blocks for Small. The design doc says "18 Stage 3 blocks + 2 Stage 2 blocks" which reverses the count. The actual Swin-Small depths are (2, 2, 18, 2), so Stage 2 = 18 blocks and Stage 3 = 2 blocks. **The description in design.md is misleading** but the config is correct (PSG_STAGES=[-2,-1] maps to stages 2 and 3 regardless of naming).

**Verdict**: PASS with minor note on single-variable violation and design doc stage count description.

---

## (b) Code Changes -- Existing Multi-Stage PSG Logic

No new code changes. Verifying existing implementation handles multi-stage correctly.

### PSG Module Creation (pose_backbone_model.py L40-63)

```python
psg_stages = list(getattr(cfg.MODEL, 'POSE_PSG_STAGES', [-1]))
num_backbone_stages = len(self.base.stages)  # 4 for Swin
self.psg_stage_indices = set()
for s in psg_stages:
    idx = s if s >= 0 else num_backbone_stages + s
    self.psg_stage_indices.add(idx)
```

- `[-2, -1]` with 4 stages resolves to `{2, 3}`. CORRECT.
- PSG modules created for each block in each stage (L53-63):
  - Stage 2: `feat_ch = num_features[2]`. For both Tiny and Small, `embed_dims=96`, so `num_features[2] = 96 * 2^2 = 384`. CORRECT.
  - Stage 3: `feat_ch = num_features[3] = 96 * 2^3 = 768`. CORRECT.
- Keys: `s2_b0, s2_b1, ..., s2_b5` (Tiny, 6 blocks) or `s2_b0, ..., s2_b17` (Small, 18 blocks), plus `s3_b0, s3_b1` for both. CORRECT.

### PAA Module Creation (L73-93)

- Same loop structure as PSG, iterates over `psg_stage_indices` and creates PAA for each block. CORRECT.
- `feat_ch` is fetched per stage from `self.base.num_features[stage_idx]`. Stage 2 = 384, Stage 3 = 768. CORRECT.
- `paa_bottleneck = 32` from config. PAA architecture: Conv2d(17->32->feat_ch). For Stage 2: 17*32 + 32*384 = 544 + 12,288 = 12,832 params per block. For Stage 3: 17*32 + 32*768 = 544 + 24,576 = 25,120 per block. CORRECT.

### _run_backbone_with_psg (L317-380)

- Iterates all 4 stages. For `i in {2, 3}`, calls `_run_stage_with_psg`. For `i in {0, 1}`, calls normal `stage(x, hw_shape)`. CORRECT.
- Semantic weight applied after every stage (L359-363). CORRECT -- same as original backbone forward.
- Output collection at L365-371: for `i in out_indices`, norm+reshape. CORRECT.

### _run_stage_with_psg (L382-404)

- Iterates blocks manually, calling `block(x, hw_shape)`, then PSG, then PAA.
- Block call at L389: `x = block(x, hw_shape)`. This calls `SwinBlock.forward()` which internally handles `with_cp` (gradient checkpointing) at L1007-1008. CORRECT -- checkpointing still active inside each block call.
- PSG applied via key lookup in `psg_modules_dict` (L392-393). CORRECT.
- PAA applied via key lookup in `paa_modules_dict` (L396-397). Uses defensive `getattr` pattern. CORRECT.
- Downsample handling at L399-404: Stage 2 HAS downsample (PatchMerging), Stage 3 does NOT. Returns `(x_down, down_hw_shape, x, hw_shape)` when downsample exists. This means the `out` (pre-downsample features) passed back to the main loop is the PSG+PAA modulated output BEFORE downsampling. CORRECT -- matches original `SwinBlockSequence.forward()` logic exactly (L1088-1096).

**Verdict**: PASS. Multi-stage PSG+PAA code is correctly implemented for arbitrary stage configurations.

---

## (c) Config Defaults (defaults.py)

- `POSE_PSG_STAGES = [-1]` (L102): Default is Stage 3 only. Existing experiments unaffected. CORRECT.
- `POSE_ADDITIVE_ADAPTER = False` (L121): Default off. Existing experiments unaffected. CORRECT.
- `POSE_PAA_BOTTLENECK = 32` (L123): Default matches config. CORRECT.
- `POSE_PAA_ROUTED = False` (L122): Not changed in exp251/252. CORRECT.
- `POSE_PAA_ADAPTIVE_GATE = False` (L124): Not changed. CORRECT.

No risk to existing experiments from these defaults.

**Verdict**: PASS.

---

## (d) WITH_CP Compatibility (exp252 only)

### How WITH_CP works in Swin

- `SwinBlock.forward()` (L991-1012): When `self.with_cp=True` and `x.requires_grad`, wraps `_inner_forward` in `cp.checkpoint()`.
- `_inner_forward` is a nested function capturing `hw_shape` via closure.
- `cp.checkpoint` re-computes the forward pass during backward, saving memory by not storing intermediate activations.

### Interaction with _run_stage_with_psg

- `_run_stage_with_psg` calls `block(x, hw_shape)` which invokes `SwinBlock.forward()`. The block internally decides whether to use checkpoint based on `self.with_cp`. CORRECT -- checkpointing is transparent to the caller.
- PSG and PAA are applied AFTER the block call returns. Their forward passes are NOT inside the checkpoint scope. This means PSG/PAA intermediate activations ARE stored in memory (not recomputed). This is acceptable -- PSG and PAA are lightweight (small Conv2d networks), their activation memory is negligible compared to the Swin block's attention activations.
- IMPORTANT: Since PSG/PAA are outside the checkpoint boundary, their gradients flow normally. No gradient issues.

### Memory concern for Small with Stage 2 PSG

- Small Stage 2 has 18 blocks. Each block gets one PSG (Conv2d(17->64->384)) and one PAA (Conv2d(17->32->384)). Total 18 PSG + 18 PAA at Stage 2, plus 2 PSG + 2 PAA at Stage 3.
- WITH_CP saves memory inside the Swin blocks but PSG/PAA activations accumulate. For Stage 2 (spatial 24x8), PSG intermediate: `B*64*24*8 = 64*64*24*8*4 bytes = ~3MB per block, 18 blocks = ~54MB`. PAA intermediate similar. Total extra activation memory for Stage 2: ~108MB.
- exp249 (Small, WITH_CP, Stage 3-only PSG) already runs fine. Adding Stage 2 PSG+PAA adds ~108MB activations. Should be fine within 24GB.

**Verdict**: PASS. WITH_CP works correctly. Minor additional memory from PSG/PAA outside checkpoint, but well within budget.

---

## (e) Feature Dimensions

### Swin-Tiny (exp251)
- `embed_dims = 96`, `depths = (2, 2, 6, 2)`
- `num_features = [96, 192, 384, 768]`
- Stage 2: feat_ch=384, spatial=24x8 (from 384x128 input, patch_size=4 -> 96x32, then two 2x downsample -> 24x8)
- Stage 3: feat_ch=768, spatial=12x4 (one more 2x downsample from Stage 2)

### Swin-Small (exp252)
- `embed_dims = 96`, `depths = (2, 2, 18, 2)`
- `num_features = [96, 192, 384, 768]` -- SAME as Tiny (same embed_dims)
- Stage 2: feat_ch=384, spatial=24x8
- Stage 3: feat_ch=768, spatial=12x4

Both PSG and PAA take `feat_channels` from `self.base.num_features[stage_idx]`, which correctly resolves to 384 for Stage 2 and 768 for Stage 3 in both architectures.

PSG Conv2d output channels match feat_channels. PAA Conv2d output channels match feat_channels. Both reshape output to `(B, H*W, C)` matching the token layout.

**Verdict**: PASS. Dimensions are correctly handled for both backbones.

---

## (f) OOM Risk Assessment

### exp251 (Tiny, no WITH_CP)

Additional parameters vs exp246b (Stage 3-only PSG, no PAA):
- Stage 2 PSG: 6 blocks * (17*64 + 64*384 + biases) = 6 * ~25,696 = ~154K params
- Stage 2 PAA: 6 blocks * (17*32 + 32*384 + biases) = 6 * ~12,864 = ~77K params
- Stage 3 PAA: 2 blocks * (17*32 + 32*768 + biases) = 2 * ~25,152 = ~50K params
- Total new params: ~281K (negligible for a model already at ~28M)

Additional activation memory:
- Stage 2 PSG: 6 blocks * B*384*24*8 = 6 * ~4.7MB = ~28MB (at BS=64, float32)
- Stage 2 PAA: similar ~28MB
- Stage 3 PAA: 2 blocks * ~1.5MB = ~3MB
- Total extra activations: ~59MB. Tiny already fits in 3090 24GB. SAFE.

### exp252 (Small, WITH_CP)

Additional parameters:
- Stage 2 PSG: 18 blocks * ~25,696 = ~462K params
- Stage 2 PAA: 18 blocks * ~12,864 = ~231K params
- Stage 3 PAA: 2 blocks * ~25,152 = ~50K params
- Total: ~743K new params

Additional activations (WITH_CP saves Swin block activations, but PSG/PAA outside checkpoint):
- As computed in section (d): ~108MB for Stage 2 PSG+PAA, ~3MB for Stage 3 PAA
- Total: ~111MB additional. WITH_CP already saves hundreds of MB from Swin blocks. SAFE.

**Verdict**: PASS. No OOM risk for either experiment.

---

## (g) Interaction with LGPA-D (Detached LGPA + GCN)

### Training path (L521-540)

LGPA-D uses `featmaps[-1]` (Stage 3 output, detached if `_lgpa_detach=True`). Stage 2 PSG does NOT affect `featmaps[-1]` directly -- it only affects Stage 2 features which are then downsampled and passed through Stage 3. Stage 2 PSG indirectly affects Stage 3 features via the information flow: Stage 2 PSG -> downsample -> Stage 3 blocks (with Stage 3 PSG) -> featmaps[-1].

When `_lgpa_detach=True`, LGPA operates on `featmaps[-1].detach()`, so LGPA gradients do NOT flow back through Stage 2 or Stage 3 PSG. CORRECT.

GCN also uses `featmaps[-1].detach()` (L530). CORRECT.

Stage 2 features: `featmaps[-2]` is passed to GCN as `stage2_feat` (L531). With multi-stage PSG, this Stage 2 output is now PSG+PAA modulated. This means GCN gets pose-enhanced Stage 2 features. This is a FEATURE, not a bug -- it's part of the expected behavior of multi-stage injection.

### Test path (L716-728)

LGPA test uses `featmaps[-1]` (not detached at test time). GCN test also uses `featmaps[-1]` (L723-724). Same reasoning -- Stage 2 PSG indirectly affects these through the backbone forward pass.

**Verdict**: PASS. No harmful interaction. LGPA-D and GCN correctly use Stage 3 output. Stage 2 PSG enriches the features flowing into Stage 3, which is the intended effect.

---

## (h) Train/Test Symmetry

### Training forward (L406-419)

- `scene_heatmaps` prepared from `pose_dict` via `_prepare_pose`.
- `_run_backbone_with_psg(x, scene_heatmaps)` called at L421.
- PSG and PAA applied in both Stage 2 and Stage 3 during training.

### Test forward (L698-797)

- Same `_run_backbone_with_psg(x, scene_heatmaps)` call at L421 (it's the SAME forward function for both train and test).
- PSG/PAA modules have no training-specific behavior (no dropout, no batch-dependent normalization). Their forward is identical in eval mode.

### PSG: `x * (1 + gate)` where gate is deterministic from heatmaps. Same in train/test.
### PAA: `x + adapter_out` where adapter_out is deterministic from heatmaps. Same in train/test.
### Neither module uses `self.training` flag for different behavior.

**Verdict**: PASS. Perfect train/test symmetry.

---

## Additional Checks

### Optimizer inclusion
- PSG modules in `self.psg_modules_dict` (nn.ModuleDict) -- automatically included in `model.parameters()`. CORRECT.
- PAA modules in `self.paa_modules_dict` (nn.ModuleDict) -- same. CORRECT.
- No manual parameter registration needed.

### Backward compatibility
- `self.psg_modules` list (L66-71) maintained for Stage 3 as backward compat. This is only used if external code references `model.psg_modules`. Should be fine.

### Zero initialization safety
- Both PSG and PAA use zero-init on final layers. At training start, PSG gate = 0 so output = x * 1 = x. PAA adapter = 0 so output = x + 0 = x. Pretrained features are preserved initially. CORRECT.

### Heatmap resize
- PSG and PAA both resize heatmaps to match `(H, W)` from `hw_shape`. At Stage 2, hw_shape = (24, 8). At Stage 3, hw_shape = (12, 4). `F.interpolate` with `bilinear` handles this correctly for any input heatmap size. CORRECT.

### Config inheritance
- exp251 inherits from exp246b recipe (Tiny LGPA-D+GCN+OA-SD+PLBOA). Only changes: PSG_STAGES and ADDITIVE_ADAPTER. CORRECT.
- exp252 inherits from exp249 recipe (Small LGPA-D+GCN+OA-SD+PLBOA+WITH_CP). Only changes: PSG_STAGES and ADDITIVE_ADAPTER. CORRECT.

---

## Summary of Findings

| Item | Status | Notes |
|------|--------|-------|
| (a) design.md | PASS | Minor: dual-variable change, misleading stage count in exp252 |
| (b) Code (multi-stage PSG) | PASS | Correct stage resolution, module creation, block iteration |
| (c) Config defaults | PASS | No impact on existing experiments |
| (d) WITH_CP compat | PASS | Checkpointing inside blocks, PSG/PAA outside (lightweight) |
| (e) Feature dimensions | PASS | 384 for Stage 2, 768 for Stage 3, both architectures |
| (f) OOM risk | PASS | ~59MB extra (Tiny), ~111MB extra (Small WITH_CP) |
| (g) LGPA-D interaction | PASS | Uses featmaps[-1] only, Stage 2 PSG enriches indirectly |
| (h) Train/test symmetry | PASS | Same forward path, no mode-dependent behavior |

---

## Verdict

**审查通过**

Both exp251 and exp252 are safe to launch. The code correctly handles multi-stage PSG and PAA for arbitrary stage configurations. Feature dimensions are correctly resolved per stage. WITH_CP is compatible. No OOM risk. Train/test symmetry is maintained. LGPA-D and GCN are not adversely affected.

Recommendations:
1. If results are positive, run a follow-up ablation separating multi-stage PSG from PAA to attribute contributions.
2. Fix the stage block count description in exp252/design.md (Stage 2 has 18 blocks, Stage 3 has 2 blocks, not the reverse).
