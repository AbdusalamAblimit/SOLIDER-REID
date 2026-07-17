# exp223 PADPQ — Claude Review

**审查时间**: 2026-04-01
**审查范围**: design.md, skeleton_gcn.py (deformable code), pose_backbone_model.py (config passing), config/defaults.py

---

## a. design.md — 合理性

- **动机清晰**: 固定 keypoint 采样 → 可学习偏移采样，解决 pose 不精确和遮挡问题。合理。
- **创新门槛**: 满足 3/3 条（问题层面、机制层面、证据层面）。Deformable sampling 在 ReID 中确实是新的。
- **单变量原则**: 对照 exp191 (fixed sampling)，仅改采样方式。符合。
- **不属于小调参**: 新增了 deformable offset head + attention head，是结构性改动。通过。

**结论**: 通过

---

## b. 新增/修改代码 — 逐行审查

### skeleton_gcn.py: SkeletonGCNHead.__init__ (lines 360-377)

```python
self.deformable_sample = deformable_sample
if self.deformable_sample:
    self._deform_k = deformable_k
    self.deform_offset_head = nn.Sequential(
        nn.Linear(feat_dim + 2, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, deformable_k * 2),
    )
    self.deform_attn_head = nn.Sequential(
        nn.Linear(feat_dim + 2, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, deformable_k),
    )
    nn.init.zeros_(self.deform_offset_head[-1].weight)
    nn.init.zeros_(self.deform_offset_head[-1].bias)
```

- Input dim: `feat_dim + 2` = 770 (768 + 2 normalized coords). Matches `context` shape. **OK**
- Hidden dim: 128. Reasonable for a small head. **OK**
- Offset output: `deformable_k * 2` = 8 (K=4, x/y per point). **OK**
- Attn output: `deformable_k` = 4. **OK**
- Zero-init on offset head final layer (weight + bias): initial offsets = 0 → identity start. **OK**
- Attn head: no special init → default Kaiming/Xavier. Initial logits will be near-zero, softmax ≈ uniform 1/K. Since offsets are zero, all K points sample the same location, so uniform attention = identity. **OK**
- Parameters are registered as `nn.Sequential` submodules → automatically included in `model.parameters()` → optimizer picks them up. **OK**

### skeleton_gcn.py: _sample_keypoint_features deformable path (lines 461-493)

**Shape trace**:

| Step | Variable | Shape | Check |
|------|----------|-------|-------|
| Input | feat_map | (B, C, fH, fW) | OK |
| Input | grid_base | (B, 17, 2) | OK |
| 1 | grid_init | (B, 17, 1, 2) | grid_sample expects (B, H_out, W_out, 2) → H_out=17, W_out=1 → output (B, C, 17, 1) |
| 1 | init_sampled | squeeze(-1) → (B, C, 17), permute → (B, 17, C) | OK |
| 2 | context | cat([B,17,C], [B,17,2]) → (B, 17, C+2) | OK |
| 2 | offsets | (B, 17, K*2) → view → (B, 17, K, 2) | OK |
| 3 | sample_pts | (B, 17, 1, 2) + (B, 17, K, 2) → broadcast → (B, 17, K, 2) | OK |
| 3 | clamp(-1, 1) | keeps in valid grid_sample range | OK |
| 4 | pts_flat | view(B, 17*K, 1, 2) | grid_sample input: H_out=17K, W_out=1 |
| 4 | sampled_flat | squeeze → (B, C, 17K), permute → (B, 17K, C) | OK |
| 4 | sampled_k | view → (B, 17, K, C) | OK |
| 5 | attn_logits | (B, 17, K) | OK |
| 5 | attn_w | softmax(-1) → (B, 17, K) | OK |
| 5 | kp_feats | (B,17,K,C) * (B,17,K,1) → sum(dim=2) → (B, 17, C) | OK |

**Output shape (B, 17, C)** matches the standard path. **OK**

**Gradient flow**:
- `feat_map` is detached (default POSE_PART_GRAD_SCALE=0.0). No backbone gradients. **OK**
- `context` depends on detached features + non-parametric coords → no grad w.r.t. inputs, but linear layers in heads contribute their weights.
- `offsets` has grad through `deform_offset_head` weights. `F.grid_sample` backprops to grid → offsets → head weights. **OK**
- `attn_w` has grad through `deform_attn_head` weights. Weighted sum backprops to both `sampled_k` (→ grid → offsets → head) and `attn_w` (→ head). **OK**
- All new parameters receive gradients. **OK**

**AMP safety**: `F.grid_sample` runs in float32 under autocast. Linear layers auto-cast to float16. No precision issues. **OK**

**Border behavior**: `padding_mode='border'` handles out-of-bounds sampling gracefully. Combined with `clamp(-1, 1)`, this is safe. **OK**

**No issues found.**

---

## c. 配置文件

### config/defaults.py (lines 188-190)

```python
_C.MODEL.POSE_DEFORMABLE_SAMPLE = False    # default off → no impact on existing experiments
_C.MODEL.POSE_DEFORMABLE_K = 4             # reasonable default
```

- Default `False`: existing experiments unaffected. **OK**
- K=4: reasonable starting point (small overhead). **OK**

### pose_backbone_model.py (lines 142-143)

```python
deformable_sample=getattr(cfg.MODEL, 'POSE_DEFORMABLE_SAMPLE', False),
deformable_k=int(getattr(cfg.MODEL, 'POSE_DEFORMABLE_K', 4)),
```

- Uses `getattr` with safe defaults matching `defaults.py`. **OK**
- `int()` cast on K for safety. **OK**
- Correctly passed as keyword args to `SkeletonGCNHead.__init__`. **OK**

---

## d. defaults.py — 新默认值安全性

- Both defaults are `False` / `4`. No behavioral change unless explicitly enabled. **OK**
- No interaction with other flags (PSG, STD-PR, etc.) when disabled. **OK**

---

## e. Processor — loss 计算、特征提取、评估逻辑

- No changes to processor. Deformable sampling is internal to `_sample_keypoint_features` and transparent to the forward/loss/eval pipeline. The output shape `(B, 17, C)` is unchanged, so all downstream code (GCN propagation, BN, classifier, pooling, etc.) works identically. **OK**

---

## f. 与前序实验的对照

- **exp191 OA-SD** (baseline): Uses `POSE_SKELETON_GCN=True`, `POSE_DEFORMABLE_SAMPLE=False` (default). Unaffected. **OK**
- **exp220 GSPB**: Uses `POSE_PART_GRAD_SCALE > 0`. PADPQ is orthogonal (different feature). Compatible if combined. **OK**
- **消融变量隔离**: Only change is `POSE_DEFORMABLE_SAMPLE=True`. Single variable. **OK**

---

## 额外发现 (Low severity)

### pose_dual_stream_model.py 未传递 deformable 参数

`pose_dual_stream_model.py` line 67 构造 `SkeletonGCNHead` 时没有传 `deformable_sample` / `deformable_k`，会使用默认值 `False` / `4`。当前 exp223 使用 `pose_backbone_model.py`，不受影响。但如果未来在 dual stream 模型中使用 PADPQ，需要补传参数。

**严重性**: Low（不影响本实验）

---

## 总结

| 项目 | 结果 |
|------|------|
| design.md | 通过 |
| 代码正确性 | 通过 — shape、init、grad flow 全部正确 |
| 配置文件 | 通过 |
| defaults.py 安全性 | 通过 |
| Processor 兼容性 | 通过 |
| 前序实验隔离 | 通过 |
| AMP 安全 | 通过 |

**审查通过**

新增参数量估算: offset_head = (770 * 128 + 128) + (128 * 8 + 8) = 99,720; attn_head = (770 * 128 + 128) + (128 * 4 + 4) = 99,204. 总计约 199K 参数。
