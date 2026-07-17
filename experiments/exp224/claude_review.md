# exp224 KAMP Review — Claude Broad Review

## 审查范围

a. design.md — 合理性、单变量原则
b. skeleton_gcn.py — KAMP 实现（init + forward）
c. pose_backbone_model.py — Stage 2 feature 传递
d. config/defaults.py — POSE_MULTI_SCALE_KP 默认值
e. 数据流验证（feature map shapes, grid_sample, projections）

---

## A. design.md

- 动机清晰：不同 body parts 需要不同尺度的特征
- 与 exp096 (MRKF) 的区别明确：KAMP 用 per-keypoint learned scale attention，而非全图 feature map 级融合
- 对照组标注清楚（exp191, exp220）
- 创新门槛论述合理

**注意**：design.md 中 stage 编号注释：
```
outs[0]: (B, 96, 96, 32)   -- Stage 1
outs[1]: (B, 192, 48, 16)  -- Stage 2
outs[2]: (B, 384, 24, 8)   -- Stage 3
outs[3]: (B, 768, 12, 4)   -- Stage 4
```
这与代码中 "S2" 命名不一致——代码的 `kamp_s2_proj` 接收的是 `featmaps[-2]` = `outs[2]` = 384 dim（Swin Stage 3，index 2），不是 192 dim 的 Swin Stage 2。命名混淆但维度正确。

---

## B. skeleton_gcn.py — KAMP 实现

### 初始化 (lines 339-352)

```python
self.kamp_s2_proj = nn.Linear(multi_scale_s2_dim, feat_dim)  # Linear(384, 768)
self.kamp_scale_attn = nn.Sequential(
    nn.Linear(feat_dim + 1, 64),  # 769 -> 64
    nn.ReLU(inplace=True),
    nn.Linear(64, 2),             # -> 2 stages
)
nn.init.zeros_(self.kamp_scale_attn[-1].weight)
nn.init.zeros_(self.kamp_scale_attn[-1].bias)
```

- `multi_scale_s2_dim` 传入为 `self.base.num_features[-2]`，Swin-Tiny 中 = 384。**正确。**
- `feat_dim` = 768 (Swin-Tiny last stage)。**正确。**
- scale_attn 最后一层 zero-init → softmax([0,0]) = [0.5, 0.5]。

### Forward (lines 764-776)

```python
if self.multi_scale_kp and stage2_feat is not None:
    kp_s2, _ = self._sample_keypoint_features(
        stage2_feat, keypoints, scores, person_mask)  # (B, 17, 384)
    kp_s2_proj = self.kamp_s2_proj(kp_s2)             # (B, 17, 768)
    attn_input = torch.cat([kp_feats, kp_scores.unsqueeze(-1)], dim=-1)  # (B, 17, 769)
    scale_logits = self.kamp_scale_attn(attn_input)    # (B, 17, 2)
    scale_w = F.softmax(scale_logits, dim=-1)          # (B, 17, 2)
    kp_feats = scale_w[:,:,0:1] * kp_s2_proj + scale_w[:,:,1:2] * kp_feats
```

**Shape 验证**：
- `stage2_feat`: (B, 384, 24, 8) — `_sample_keypoint_features` 内部 `F.grid_sample` 使用归一化坐标 [-1,1]，与 feature map 空间分辨率无关 → **正确**
- `kp_s2`: (B, 17, 384) → `kamp_s2_proj(384→768)` → (B, 17, 768) → **正确**
- `kp_feats`: (B, 17, 768)，`kp_scores`: (B, 17) → `attn_input`: (B, 17, 769) → `kamp_scale_attn(769→64→2)` → (B, 17, 2) → **正确**
- 最终 `kp_feats`: (B, 17, 768)，维度不变 → **正确**

### AMP 安全
- `F.grid_sample` 在 AMP 下自动处理 float16 输入 → 安全
- `F.softmax` 在 float16 下可能有精度问题但对 2 个 logits 影响极小 → 安全
- `nn.Linear` 自动适配 AMP → 安全

---

## C. pose_backbone_model.py — Stage 2 传递

### 训练路径 (line 486)
```python
_s2_feat = featmaps[-2].detach() if len(featmaps) >= 2 else None
```
- `.detach()` 阻断梯度回传到 backbone Stage 2 → 有意为之，防止 KAMP loss 干扰共享 backbone 的 global 分支
- `len(featmaps) >= 2`：Swin-Tiny `out_indices=(0,1,2,3)` → `outs` 有 4 个元素 → 条件成立 → **正确**

### 测试路径 (line 609)
```python
_s2_test = featmaps[-2] if len(featmaps) >= 2 else None
```
- 测试时无需 detach → **正确**

### 构造时 dim 计算 (line 145)
```python
multi_scale_s2_dim=self.base.num_features[-2]  # = 384 for Swin-Tiny
```
- `num_features = [96, 192, 384, 768]`，`[-2] = 384` → **与 `featmaps[-2]` 的 channel dim 一致。正确。**

---

## D. config/defaults.py

```python
_C.MODEL.POSE_MULTI_SCALE_KP = False
_C.MODEL.POSE_MULTI_SCALE_STAGES = [2, 3]  # unused in current impl
```
- 默认 False → 不影响已有实验 → **正确**
- `POSE_MULTI_SCALE_STAGES` 在代码中未被引用（当前实现 hardcode 为 `featmaps[-2]`）→ Low: dead config，不影响功能

---

## E. 优化器

- `skeleton_head` 是 `nn.Module` 子模块，其所有参数（包括 `kamp_s2_proj` 和 `kamp_scale_attn`）自动被 `model.parameters()` 收集 → 优化器会覆盖 → **正确**

---

## 问题汇总

### Medium: Zero-init 不等于 identity start

**位置**: skeleton_gcn.py lines 349-351

scale_attn 最后一层 zero-init 使初始 softmax 输出 [0.5, 0.5]，但 `kamp_s2_proj` 使用默认 PyTorch 初始化（Kaiming uniform），不是 zero-init。因此初始化时模型输出 = `0.5 * random_proj(s2_feat) + 0.5 * kp_feats`，而非纯 identity。

这不是 bug（训练会调整），但与注释 "Zero-init → equal weighting initially" 的隐含意图（"starts as identity"）不完全一致。实际上模型从 epoch 1 开始就混入了 50% 的随机投影噪声。

**修复建议**（可选）：如果希望真正的 identity start，可以：
- (A) 将 scale_attn bias 初始化为 `[large_negative, 0]`（如 `[-5, 0]`），使 softmax 接近 `[0, 1]`，几乎只用 S3
- (B) 或者 zero-init `kamp_s2_proj` 的权重，使投影输出为零

当前行为不会导致训练失败，但可能导致初期 loss spike。**不阻塞训练。**

### Low: "S2" 命名混淆

代码中 `kamp_s2_proj`、`multi_scale_s2_dim` 的 "S2" 实际对应 Swin Stage 3 (index 2, 384 dim)，不是 Swin Stage 2 (index 1, 192 dim)。建议未来重命名以减少混淆。**不影响功能。**

### Low: Dead config `POSE_MULTI_SCALE_STAGES`

`defaults.py` 中定义了 `POSE_MULTI_SCALE_STAGES = [2, 3]` 但代码未引用。当前实现 hardcode 为 `featmaps[-2]`。**不影响功能。**

---

## 结论

**审查通过**

维度计算正确（`num_features[-2] = 384` for Swin-Tiny），数据流完整（训练+测试路径均传递 stage2_feat），grid_sample 自适应空间分辨率，AMP 安全，优化器覆盖完整。Medium 级 zero-init 问题不阻塞训练——初始 50% 混合实际上可能反而加速 scale attention 的学习。
