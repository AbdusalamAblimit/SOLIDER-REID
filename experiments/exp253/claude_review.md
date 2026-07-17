# Claude Review — exp253: Multi-Stage PSG (Stage 1+2+3, no PAA) on Tiny LGPA-D+GCN

**审查日期**: 2026-04-08
**审查范围**: design.md, pose_backbone_model.py (stage resolution, PSG creation, downsample handling, heatmap interpolation), config defaults, WITH_CP compatibility, OOM risk

---

## (a) design.md 审查

### 合理性
- 动机清晰：exp251 混合了 multi-stage PSG 和 PAA，无法分离各自贡献。exp253 去掉 PAA，纯测 multi-stage PSG 效果。同时扩展到 3 个 stage (1+2+3)。这是合理的消融设计。
- 核心假设明确：低层特征 (Stage 1, 192 dim, 48x16) 也能受益于 pose 空间门控。
- 对照组清晰：exp246b (Stage 3-only) 和 exp251 (Stage 2+3 + PAA) 形成对照链。
- 预期结果分三档 (成功/中性/失败)，判断标准合理。

### 单变量原则
exp253 vs exp246b: 唯一变量是 POSE_PSG_STAGES 从 [-1] 变为 [-3,-2,-1]。满足单变量原则。

### 创新性质疑
这是一个 config-only 消融实验，不是新方法。但作为 multi-stage PSG 的消融数据点（用于论文），合理。不属于"逃避创新的小调参"，因为它服务于已有 PSG 方法的消融表格。

**Verdict**: PASS

---

## (b) Code Review: pose_backbone_model.py

### (b1) Stage index resolution (Lines 40-46)

```python
psg_stages = list(getattr(cfg.MODEL, 'POSE_PSG_STAGES', [-1]))
num_backbone_stages = len(self.base.stages)  # = 4 for Swin-Tiny
self.psg_stage_indices = set()
for s in psg_stages:
    idx = s if s >= 0 else num_backbone_stages + s
    self.psg_stage_indices.add(idx)
```

With `POSE_PSG_STAGES = [-3, -2, -1]` and `num_backbone_stages = 4`:
- `-3` -> `4 + (-3) = 1` (Stage 1, 192 dim)
- `-2` -> `4 + (-2) = 2` (Stage 2, 384 dim)
- `-1` -> `4 + (-1) = 3` (Stage 3, 768 dim)

Result: `psg_stage_indices = {1, 2, 3}`. CORRECT.

Stage 0 (96 dim, 96x32 spatial) is intentionally excluded, matching design.md.

### (b2) PSG module creation for Stage 1 (Lines 52-63)

```python
for stage_idx in sorted(self.psg_stage_indices):  # [1, 2, 3]
    stage = self.base.stages[stage_idx]
    feat_ch = self.base.num_features[stage_idx]  # num_features[1] = 192
    for block_idx in range(len(stage.blocks)):    # Stage 1 has 2 blocks
        key = f's{stage_idx}_b{block_idx}'        # 's1_b0', 's1_b1'
        self.psg_modules_dict[key] = PoseSpatialGate(
            pose_channels=17, feat_channels=feat_ch, ...)
```

Swin-Tiny depths = (2, 2, 6, 2):
- Stage 1: `num_features[1] = 96 * 2^1 = 192`, 2 blocks -> keys `s1_b0`, `s1_b1`
- Stage 2: `num_features[2] = 96 * 2^2 = 384`, 6 blocks -> keys `s2_b0` through `s2_b5`
- Stage 3: `num_features[3] = 96 * 2^3 = 768`, 2 blocks -> keys `s3_b0`, `s3_b1`

Total: 2 + 6 + 2 = 10 PSG modules. Matches design.md. CORRECT.

### (b3) Downsample handling in _run_stage_with_psg (Lines 382-404)

```python
def _run_stage_with_psg(self, stage, x, hw_shape, scene_heatmaps, stage_idx=None):
    for block_idx, block in enumerate(stage.blocks):
        key = f's{stage_idx}_b{block_idx}'
        x = block(x, hw_shape)
        if scene_heatmaps is not None and key in self.psg_modules_dict:
            x = self.psg_modules_dict[key](x, hw_shape, scene_heatmaps)
    # Handle downsample
    if stage.downsample:
        x_down, down_hw_shape = stage.downsample(x, hw_shape)
        return x_down, down_hw_shape, x, hw_shape
    else:
        return x, hw_shape, x, hw_shape
```

Stage 1 (index 1) has `downsample = PatchMerging(192 -> 384)`. Stage 2 (index 2) has `downsample = PatchMerging(384 -> 768)`. Stage 3 (index 3) has `downsample = None`.

The method correctly checks `stage.downsample` and applies it after all blocks+PSG are done. The return signature `(x_down, down_hw_shape, x, hw_shape)` matches the original `SwinBlockSequence.forward()` exactly (swin_transformer.py L1088-1096). The caller in `_run_backbone_with_psg` (L350-357) handles both PSG and non-PSG stages identically.

CORRECT. Downsample is properly handled for Stages 1 and 2.

### (b4) Heatmap spatial interpolation in PSG (pose_spatial_gate.py L67-71)

```python
if scene_heatmaps.shape[2:] != (H, W):
    hm = F.interpolate(scene_heatmaps, size=(H, W),
                        mode='bilinear', align_corners=False)
```

Input heatmaps are (B, 17, 96, 32) for 384x128 input images. Stage spatial sizes:
- Stage 1: hw_shape = (48, 16). `(96, 32) != (48, 16)` -> interpolate to (48, 16). 2x downscale. CORRECT.
- Stage 2: hw_shape = (24, 8). Interpolate to (24, 8). 4x downscale. CORRECT.
- Stage 3: hw_shape = (12, 4). Interpolate to (12, 4). 8x downscale. CORRECT.

The bilinear interpolation handles non-integer scale factors gracefully. No alignment issues since `align_corners=False`.

Note: For Stage 1, the heatmaps are downsampled only 2x (96->48, 32->16), meaning more spatial detail is preserved. This is actually beneficial -- the pose signal has higher fidelity at earlier stages.

CORRECT.

### (b5) Semantic weight application (Lines 359-363)

After each stage (including PSG stages), semantic weight is applied:
```python
if sem_weight is not None:
    sw = self.base.semantic_embed_w[i](sem_weight).unsqueeze(1)
    sb = self.base.semantic_embed_b[i](sem_weight).unsqueeze(1)
    x = x * self.base.softplus(sw) + sb
```

This operates on `x` (the downsampled output from `_run_stage_with_psg`). The semantic embed layers are indexed by stage `i`, matching `num_features[i+1]` for stages 0-2 (with downsample) and `num_features[i]` for stage 3 (no downsample). This is identical to the original Swin forward pass. No interaction issue.

CORRECT.

**Code Verdict**: PASS. All code paths handle 3-stage PSG correctly.

---

## (c) Config Defaults

`config/defaults.py` line 102: `_C.MODEL.POSE_PSG_STAGES = [-1]`

Default is Stage 3 only. This is safe:
- All existing experiments using default get Stage 3 PSG only (unchanged behavior).
- exp253 overrides to `[-3,-2,-1]` via command-line or config file.
- No existing config file sets `POSE_PSG_STAGES` except `pose_pcg_only.yml` (which uses `[]`).

**Verdict**: PASS. Default is safe and backward-compatible.

---

## (d) WITH_CP Compatibility

exp253 runs on remote 5060 Ti (16 GB VRAM). exp251 (2-stage PSG + PAA, WITH_CP) ran at 5542 MiB on the same machine.

WITH_CP checkpoints SwinBlock.forward internally. PSG modules run OUTSIDE the checkpointed function (they are called after `block(x, hw_shape)` in `_run_stage_with_psg`). This means PSG activations are NOT recomputed during backward -- they persist in memory.

However, PSG modules are lightweight Conv2d operations:
- Per PSG intermediate: B * hidden_dim * H * W * 4 bytes (float32 under AMP forward, but stored in float16 for backward)
- Stage 1 (48x16, hidden=64): 64 * 64 * 48 * 16 * 2 = ~6.3 MB per PSG, x2 blocks = ~12.6 MB
- Stage 2 (24x8, hidden=64): 64 * 64 * 24 * 8 * 2 = ~1.6 MB per PSG, x6 blocks = ~9.4 MB
- Stage 3 (12x4, hidden=64): 64 * 64 * 12 * 4 * 2 = ~0.4 MB per PSG, x2 blocks = ~0.8 MB
- Total PSG activation overhead: ~23 MB

exp251 (Stage 2+3 PSG + PAA) used 5542 MiB. exp253 adds Stage 1 PSG (~12.6 MB) but removes PAA entirely. PAA saved memory > Stage 1 PSG added memory. Net memory should be LESS than exp251.

**Verdict**: PASS. WITH_CP works. Memory should be under exp251's 5542 MiB.

---

## (e) OOM Risk Assessment

### Parameter count

exp253 vs exp246b (Stage 3-only PSG, 2 modules):

Additional PSG modules:
- Stage 1: 2 PSG modules, each Conv2d(17->64->192). Params: 2 * (17*64 + 64 + 64*192 + 192) = 2 * (1088 + 64 + 12288 + 192) = 2 * 13632 = 27,264
- Stage 2: 6 PSG modules, each Conv2d(17->64->384). Params: 6 * (17*64 + 64 + 64*384 + 384) = 6 * (1088 + 64 + 24576 + 384) = 6 * 26112 = 156,672
- Total new params: 183,936 (~184K)

For comparison, exp246b's Stage 3 PSG: 2 * (17*64 + 64 + 64*768 + 768) = 2 * 50816 = 101,632 (~102K).

Total PSG params in exp253: 102K + 184K = 286K. Negligible relative to Swin-Tiny's ~28M params.

### Activation memory
As computed in section (d): ~23 MB total PSG activation overhead. Negligible on 16 GB VRAM.

**Verdict**: PASS. No OOM risk.

---

## (f) Train/Test Symmetry

`_run_backbone_with_psg` is called identically in both training and inference. No `self.training`-dependent branching within PSG or the stage loop (except Stochastic Pose Dropout, which is gated by `self.training` before the backbone call). The same PSG modules fire for all stages in both modes.

**Verdict**: PASS.

---

## Summary

| Section | Verdict | Notes |
|---------|---------|-------|
| (a) design.md | PASS | Valid single-variable ablation for multi-stage PSG |
| (b) Code: stage resolution | PASS | [-3,-2,-1] -> {1,2,3} correct with 4 stages |
| (b) Code: PSG creation | PASS | 10 modules (2+6+2) with correct feat_ch per stage |
| (b) Code: downsample | PASS | _run_stage_with_psg handles PatchMerging correctly |
| (b) Code: heatmap resize | PASS | F.interpolate adapts to each stage's spatial size |
| (c) Config defaults | PASS | [-1] default is safe, no side effects |
| (d) WITH_CP compat | PASS | PSG outside checkpoint, lightweight overhead |
| (e) OOM risk | PASS | ~184K new params, ~23 MB extra activations |
| (f) Train/test symmetry | PASS | Identical forward path in both modes |

---

## 结论

**审查通过**

exp253 is a clean config-only ablation experiment. The code in `pose_backbone_model.py` correctly handles 3-stage PSG injection: stage indices resolve correctly, PSG modules are created with the right feature dimensions per stage, PatchMerging downsamples are applied after PSG injection at each stage, and heatmap interpolation adapts to each stage's spatial resolution. No code modifications needed. Memory usage will be less than exp251 (which ran successfully at 5542 MiB) because PAA is removed. Safe to launch.
