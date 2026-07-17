# Claude Broad Review: exp175 PSG at ALL Stages [0,1,2,3] (no PAPE)

**Reviewer**: Opus 4.6
**Date**: 2026-03-24

---

## A. design.md Review

**Design clarity**: Adequate. Single variable vs exp166 (PSG@[3] only): extend PSG to all stages. PAPE disabled.

**Single-variable principle**: Satisfied. vs exp166 changes only `POSE_PSG_STAGES` from `[-1]` (= stage 3 only) to `[0,1,2,3]`.

**Is this just a small config change?**: This is a legitimate ablation/extension experiment. The multi-stage PSG code already exists (validated in exp173 with stages [2,3]). Extending to [0,1,2,3] is a valid ablation to answer "does full-stage PSG beat partial PSG and PAPE?" This is a fine experiment as a supporting/ablation result.

**Parameter estimates in design.md**: Let me verify:
- Stage 0: 96-d, 2 blocks. PSG = Conv2d(17->64, 1x1) + Conv2d(64->96, 1x1). Per gate: 17*64+64 + 64*96+96 = 1088+64+6144+96 = 7392. x2 blocks = 14784. Design says ~14K. **Correct.**
- Stage 1: 192-d, 2 blocks. Per gate: 17*64+64 + 64*192+192 = 1152 + 12480 = 13632. x2 = 27264. Design says ~28K. **Correct.**
- Stage 2: 384-d, 6 blocks. Per gate: 17*64+64 + 64*384+384 = 1152 + 24960 = 26112. x6 = 156672. Design says ~156K. **Correct.**
- Stage 3: 768-d, 2 blocks. Per gate: 17*64+64 + 64*768+768 = 1152 + 49920 = 51072. x2 = 102144. Design says ~102K. **Correct.**
- Total: ~300K. **Correct.**

**Severity**: No issues.

---

## B. Code Review — PoseBackboneModel (`pose_backbone_model.py`)

### B1. `__init__`: PSG module creation (lines 39-63)

```python
psg_stages = list(getattr(cfg.MODEL, 'POSE_PSG_STAGES', [-1]))
# Resolve negative indices
self.psg_stage_indices = set()
for s in psg_stages:
    idx = s if s >= 0 else num_backbone_stages + s
    self.psg_stage_indices.add(idx)
```

For `POSE_PSG_STAGES: [0, 1, 2, 3]`, all indices are non-negative, so `psg_stage_indices = {0, 1, 2, 3}`. **Correct.**

```python
feat_ch = self.base.num_features[stage_idx]
```

`num_features = [96*2^i for i in range(4)] = [96, 192, 384, 768]`.

- Stage 0: feat_ch=96. **Correct.**
- Stage 1: feat_ch=192. **Correct.**
- Stage 2: feat_ch=384. **Correct.**
- Stage 3: feat_ch=768. **Correct.**

**Severity**: No issues.

### B2. `_run_stage_with_psg` (lines 303-325) — Downsample handling

This is the **most critical check**. Stages 0, 1, 2 have downsamples. Stage 3 does not.

```python
for block_idx, block in enumerate(stage.blocks):
    x = block(x, hw_shape)     # blocks operate at stage's own dimension
    # PSG applied here — uses hw_shape (pre-downsample spatial shape)
    x = self.psg_modules_dict[key](x, hw_shape, scene_heatmaps)

# After all blocks:
if stage.downsample:
    x_down, down_hw_shape = stage.downsample(x, hw_shape)
    return x_down, down_hw_shape, x, hw_shape
else:
    return x, hw_shape, x, hw_shape
```

This exactly mirrors `SwinBlockSequence.forward()` (lines 1088-1096 of swin_transformer.py):
```python
def forward(self, x, hw_shape):
    for block in self.blocks:
        x = block(x, hw_shape)
    if self.downsample:
        x_down, down_hw_shape = self.downsample(x, hw_shape)
        return x_down, down_hw_shape, x, hw_shape
    else:
        return x, hw_shape, x, hw_shape
```

The PSG is inserted between blocks (before downsample), at the correct dimension and spatial shape. The downsample is correctly applied AFTER all PSG-gated blocks. Return values match the original stage.forward() signature.

**Severity**: No issues. **Correct.**

### B3. Semantic weight application (lines 280-284)

```python
if sem_weight is not None:
    sw = self.base.semantic_embed_w[i](sem_weight).unsqueeze(1)
    sb = self.base.semantic_embed_b[i](sem_weight).unsqueeze(1)
    x = x * self.base.softplus(sw) + sb
```

**Critical dimension check**: After stage `i` returns, `x` is the DOWNSAMPLED output (for stages 0-2). The semantic embed dimensions are:

From the backbone code (line 1280-1283):
```python
for i in range(len(depths)):
    if i >= len(depths) - 1:
        i = len(depths) - 2   # clamp last stage
    semantic_embed_w = nn.Linear(2, self.num_features[i+1])
```

So:
- Stage 0 (i=0): semantic_embed_w[0] outputs dim `num_features[1]` = 192. After stage 0 downsample, x has dim 192. **Match.**
- Stage 1 (i=1): semantic_embed_w[1] outputs dim `num_features[2]` = 384. After stage 1 downsample, x has dim 384. **Match.**
- Stage 2 (i=2): semantic_embed_w[2] outputs dim `num_features[3]` = 768. After stage 2 downsample, x has dim 768. **Match.**
- Stage 3 (i=3): clamped to i=2, semantic_embed_w[3] outputs dim `num_features[3]` = 768. Stage 3 has no downsample, output dim = 768. **Match.**

**Severity**: No issues. **Correct.**

### B4. `out_indices` norm application (lines 286-292)

The `out` variable in `_run_stage_with_psg` return is the PRE-downsample output (same as the original Swin stage). The norm layer uses `num_features[i]` which matches the pre-downsample dimension. **Correct.**

---

## C. PoseSpatialGate Module (`pose_spatial_gate.py`)

### C1. Varying `feat_channels`

The constructor parameterizes `feat_channels`. For stages [0,1,2,3] with channels [96,192,384,768], each PSG instance gets its own `feat_channels`. The encoder final layer is `Conv2d(hidden_dim, feat_channels, 1x1)`. The gate is then reshaped to `(B, H*W, C)` where C=feat_channels. This matches the input x shape `(B, H*W, C)`.

**Severity**: No issues.

### C2. Heatmap interpolation at Stage 0

Input `scene_heatmaps` has shape `(B, 17, H_hm, W_hm)` where `POSE_HEATMAP_SIZE = [96, 32]` means H_hm=96, W_hm=32.

Swin-Tiny spatial dimensions for 384x128 input (patch_size=4, then downsample 2x per stage):
- After PatchEmbed (stride 4): H=384/4=96, W=128/4=32 → hw_shape = (96, 32)
- **Stage 0**: blocks operate at (96, 32) with dim 96. After downsample: (48, 16) with dim 192.
- **Stage 1**: blocks operate at (48, 16) with dim 192. After downsample: (24, 8) with dim 384.
- **Stage 2**: blocks operate at (24, 8) with dim 384. After downsample: (12, 4) with dim 768.
- **Stage 3**: blocks operate at (12, 4) with dim 768. No downsample.

At Stage 0, `hw_shape = (96, 32)` and `scene_heatmaps.shape[2:] = (96, 32)`. The PSG forward check:
```python
if scene_heatmaps.shape[2:] != (H, W):
    hm = F.interpolate(...)
else:
    hm = scene_heatmaps  # No interpolation needed — exact match
```

**This is a no-op at Stage 0** (heatmap already matches spatial shape). At other stages:
- Stage 1: (96,32) -> (48,16) — bilinear downsample
- Stage 2: (96,32) -> (24,8) — bilinear downsample
- Stage 3: (96,32) -> (12,4) — bilinear downsample

All use `F.interpolate` with bilinear mode. **Correct.** No issues.

### C3. Sigmoid on raw heatmap logits

```python
hm = torch.sigmoid(hm)
```

Applied at every PSG instance. Scene heatmaps are raw logits from ViTPose. Sigmoid is applied after interpolation, which is the correct order (interpolating logits is more stable than interpolating probabilities). **Correct.**

---

## D. Config Validation

Base config: `pose_psg_stdpr_pertoken_plboa_pape_ms.yml`

Base config has:
- `POSE_PSG_STAGES: [2, 3]` — will be overridden to `[0, 1, 2, 3]` via CLI
- `POSE_PATCH_EMBED: True` — will be overridden to `False` via CLI
- `POSE_BACKBONE_PSG: True` — uses PoseBackboneModel. **Correct.**
- `POSE_STRUCTURAL_ROUTING: True` — STD-PR enabled with per-token classification. **Inherited from base.**
- `POSE_LOWER_BODY_OCC: True` with prob 0.7 — PLBOA enabled. **Inherited from base.**
- `POSE_TEST_FEAT: 'equal_concat'` — global + part test features. **Inherited.**

CLI overrides `POSE_PSG_STAGES [0,1,2,3] POSE_PATCH_EMBED False` — these are standard YACS key-value CLI overrides applied AFTER yaml loading. **Valid.**

### D1. `config/defaults.py` safety

`POSE_PSG_STAGES` default is `[-1]` (stage 3 only). Setting to `[0,1,2,3]` only affects `psg_stage_indices` in `PoseBackboneModel.__init__`. No other code path reads this config. **Safe.**

`POSE_PATCH_EMBED` default is `False`. Setting to `False` explicitly just confirms the default. The `use_pose_patch_embed` attribute is set in `__init__` and checked in `_run_backbone_with_psg`. **Safe.**

---

## E. Memory Estimation (5060 Ti)

The 5060 Ti has 16GB VRAM.

**Additional PSG parameters**: ~300K params = ~1.2MB in FP32 (negligible).

**Additional computation**: At Stage 0, heatmaps are (B, 17, 96, 32) and features are (B, 96*32, 96). The PSG Conv2d operations at this resolution are relatively small. Each PSG gate adds:
- Conv2d(17->64, 1x1) on (B, 17, H, W) spatial tensor
- Conv2d(64->C, 1x1) on (B, 64, H, W) spatial tensor
- Gate multiplication on (B, H*W, C) tokens

For Stage 0 at (96,32): intermediate tensor = B*64*96*32 = 64*64*96*32*4 bytes = ~50MB per batch (largest stage). This is the largest spatial stage but smallest channel count.

**Comparison**: exp173 (PAPE + PSG@[2,3]) ran on the same GPU. PSG@[0,1,2,3] replaces PAPE (which adds Conv2d(17->96, kxk) at (96,32) resolution, then adds to all tokens). The Stage 0/1 PSG operations are comparable in memory to PAPE. Should fit within similar memory budget.

**Risk level**: Low. exp173 fit on the same GPU; this experiment removes PAPE and adds Stage 0/1 PSG which should be similar or slightly less memory.

---

## F. Optimizer Check

PSG modules are registered in `self.psg_modules_dict` (an `nn.ModuleDict`), which is a proper `nn.Module` attribute. All parameters will be found by `model.parameters()` and included in the optimizer. **Correct.**

---

## G. Forward/Backward Pass Verification

1. `forward()` calls `_prepare_pose()` to get `scene_heatmaps` — same as all previous PSG experiments.
2. `_run_backbone_with_psg()` iterates through all 4 stages.
3. For stages 0-3 (all in `psg_stage_indices`), calls `_run_stage_with_psg()`.
4. Each block runs normally, then PSG gates the output.
5. Downsample happens after all blocks in stages 0-2.
6. Semantic weight applied after each stage (correct dimensions verified above).
7. Output norms applied for `out_indices` stages.
8. Final feature map from stage 3 used for GAP and STD-PR routing.

Backward: All PSG modules are differentiable (Conv2d, sigmoid, multiplication). Feature map for STD-PR is `.detach()`-ed (line 357), which is by design (stop-grad for part branch). **No issues.**

---

## H. AMP Safety

PSG uses `F.interpolate` with `mode='bilinear'`. Under AMP (FP16), bilinear interpolation can occasionally produce slightly different results but is numerically safe. The sigmoid, Conv2d, and element-wise multiply are all AMP-safe. **No issues.**

---

## I. Logging Sufficiency

The existing logging from exp166/exp173 infrastructure will report:
- PSG module creation (printed during `__init__`)
- Per-epoch losses (global ID, triplet, part ID, part triplet)
- STD-PR router stats
- Evaluation metrics

No additional logging needed for this ablation. **Sufficient.**

---

## J. Interaction with Existing Experiments

Setting `POSE_PSG_STAGES` to `[0,1,2,3]` does not affect any default config or existing experiment. The base config yaml has its own output directory (`exp173_triple_pose`) but this experiment will override `OUTPUT_DIR` via CLI. **No cross-contamination.**

---

## Summary

| Category | Status | Severity |
|----------|--------|----------|
| design.md | Pass | - |
| PSG feat_channels per stage | Pass | - |
| Downsample handling in _run_stage_with_psg | Pass | - |
| Semantic weight dimension alignment | Pass | - |
| Heatmap interpolation at Stage 0 | Pass (no-op, exact match) | - |
| Config override validity | Pass | - |
| Memory (5060 Ti) | Pass (low risk) | - |
| Optimizer coverage | Pass | - |
| Forward/backward pass | Pass | - |
| AMP safety | Pass | - |
| Logging | Pass | - |
| Reproducibility / isolation | Pass | - |

No Critical, High, Medium, or Low issues found.

---

## 审查通过

The experiment is a clean ablation extending PSG from stages [2,3] to all stages [0,1,2,3] while disabling PAPE. All code paths have been verified:
- Feature channel dimensions correctly resolved from `num_features[stage_idx]` for each stage.
- Downsample logic in `_run_stage_with_psg` exactly mirrors the original `SwinBlockSequence.forward()`.
- Semantic weight dimensions match post-downsample token dimensions at every stage.
- Stage 0 heatmap interpolation is correctly a no-op (heatmap size matches patch-embed spatial shape).
- Memory should fit on 5060 Ti (comparable to exp173 which used PAPE at same resolution).
