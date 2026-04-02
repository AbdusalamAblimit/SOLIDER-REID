# exp225 审查 — GSPB + PADPQ Combined (Tiny)

## 审查范围

a. design.md 合理性
b. 代码路径：GSPB gradient scaling + PADPQ deformable sampling 交互
c. 配置默认值安全性
d. 梯度流分析（核心问题）

---

## 1. GSPB + PADPQ 交互分析（核心问题）

### 梯度流追踪

```
featmaps[-1]  (backbone Stage 3 output, has grad)
    │
    ▼ GSPB (line 452, pose_backbone_model.py)
feat_map_detached = featmaps[-1].detach() + 0.05 * (featmaps[-1] - featmaps[-1].detach())
    │   forward: 值 == featmaps[-1]（数值恒等）
    │   backward: grad 缩放至 5%
    │
    ▼ skeleton_head.forward(feat_map_detached, ...) (line 487-489)
    │
    ▼ _sample_keypoint_features(feat_map, ...) (line 448/746-748)
    │   feat_map 参数 = feat_map_detached (带 5% grad)
    │
    ▼ PADPQ 分支 (line 477-509)
        │
        ├─ init_sampled = grid_sample(feat_map, grid_init)    ← 5% grad
        │   context = cat([init_sampled, kp_pos_norm])        ← 5% grad on C dims
        │
        ├─ offsets = deform_offset_head(context)              ← offset_head 的梯度来自两条路径：
        │   │   (1) 自身参数的 grad（正常 100%）
        │   │   (2) 通过 context 中 init_sampled 的 5% backbone grad
        │   │
        │   ▼ sample_pts = grid_base + offsets                ← offsets 有正常 grad
        │   ▼ sampled_flat = grid_sample(feat_map, pts_flat)  ← feat_map 有 5% grad
        │   ▼ sampled_k                                       ← 来自 feat_map 的 5% grad + 来自 sample_pts 的正常 grad
        │
        ├─ attn_logits = deform_attn_head(context)            ← 同理，attn_head 自身参数 100%，context 中 5%
        │   attn_w = softmax(attn_logits)
        │
        └─ kp_feats = (sampled_k * attn_w).sum()
            │
            ▼ GCN → BN → ID Loss / Triplet Loss
```

### 回答核心问题：GSPB 的 5% gradient scaling 是否影响 PADPQ 的 offset_head 和 attn_head？

**是的**，但只影响通过 `feat_map` 传入的那条梯度路径。具体来说：

1. **offset_head 和 attn_head 自身参数的梯度是正常的（100%）** — 它们是 SkeletonGCNHead 的子模块，loss 到它们的梯度路径不经过 GSPB 的 scale。

2. **offset_head/attn_head 的输入 `context` 包含 `init_sampled`**（从 feat_map grid_sample 得到），而 feat_map 有 5% grad scale。因此 context 中 C 维度的梯度被 scale 了，但 2 维度的 kp_pos_norm 部分不受影响（坐标是 detached 常量）。

3. **关键：offset_head 学到的 offsets 用于 `grid_sample(feat_map, sample_pts)`** — 这里 `feat_map` 同样带 5% grad。但 offsets→sample_pts 的梯度路径是通过 `grid_sample` 的 spatial gradient（对采样坐标的偏导），这条路径上 feat_map 的值（而非 grad）决定了梯度方向，所以 GSPB 的 scale 对 offset 学习的影响是间接的。

### 这是否是个问题？

**不是问题，是合理的设计**：

- PADPQ 的 offset_head/attn_head 参数仍然获得 100% 梯度 — 它们的学习不被压制
- 唯一被缩放的是"Part 分支的梯度传回 backbone"这条路径
- 这正是 GSPB 的设计意图：让 backbone 感受到 Part 分支需求（包括 PADPQ 的需求），但强度只有 5%
- PADPQ 的 deformable offsets 可以自由学习（100% grad），只是它们对 backbone 的影响被抑制到 5%

实际上，PADPQ + GSPB 的组合可能比单独 PADPQ 更好：PADPQ 在学习 offsets 时依赖 feat_map 的质量，而 GSPB 允许 backbone 稍微适应 PADPQ 的采样需求（5% 的信号），这是一个温和的协同效应。

---

## 2. design.md 审查

- **动机清晰**：两个独立改进的组合实验，有明确的对照组（exp191/220/223）
- **单变量原则**：相对于 exp220 加了 PADPQ，相对于 exp223 加了 GSPB。有两个对照组 OK
- **风险评估到位**：提到了可能的冲突和止损标准（ep10 < 30%）
- **创新门槛**：这是组合实验，不是新创新。design.md 没有声称是创新，只是验证叠加效果。符合"supporting evidence"角色

### 小问题

design.md 说"基于 pose_psg_gcn_paa_roa.yml (含 ROA) + OA-SD + PLBOA"，但没有列出具体配置文件路径或完整的命令行 override。需要确认最终运行命令包含：

```
MODEL.POSE_PART_GRAD_SCALE 0.05
MODEL.POSE_DEFORMABLE_SAMPLE True
MODEL.POSE_DEFORMABLE_K 4
```

---

## 3. 配置安全性

| 参数 | 默认值 | exp225 值 | 安全 |
|------|--------|-----------|------|
| POSE_PART_GRAD_SCALE | 0.0 | 0.05 | OK — 默认 0.0 不影响其他实验 |
| POSE_DEFORMABLE_SAMPLE | False | True | OK — 默认 False |
| POSE_DEFORMABLE_K | 4 | 4 | OK — 默认值即所需 |

无新增 defaults.py 修改。安全。

---

## 4. 代码路径审查

无新代码修改（design.md 明确说 "No new code"）。两个特性的代码已在 exp220 和 exp223 中审查通过。

需验证的唯一问题是它们能否同时启用：

- `_part_grad_scale` 在 `__init__` (line 116) 设置
- `deformable_sample` 在 `SkeletonGCNHead.__init__` (line 377) 设置
- 两者互不干扰：GSPB 修改 `feat_map_detached` 的梯度属性，PADPQ 在 `_sample_keypoint_features` 内部使用它
- 没有互斥逻辑或 assertion 阻止同时启用

**可以共存。**

---

## 5. 潜在风险

| 风险 | 严重性 | 评估 |
|------|--------|------|
| PADPQ + GSPB 梯度冲突 | Low | 如上分析，PADPQ 自身参数 grad 正常，只是对 backbone 的反传被 scale。合理 |
| 显存增加 | Low | PADPQ K=4 增加 4x grid_sample，GSPB 几乎无开销。应在 3090 24GB 内 |
| 训练不稳定 | Low | 两者都已独立验证稳定 |

---

## 审查结论

**审查通过**

所有代码路径、梯度流、配置安全性检查均通过。GSPB 的 5% gradient scale 确实作用于 PADPQ 接收的 feature map，但这是合理且有益的：PADPQ 的 offset/attention 模块自身参数保持 100% 梯度学习，仅对 backbone 的反传信号被缩放至 5%，符合 GSPB 的设计意图。两个特性可以安全共存。
