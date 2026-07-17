# Claude Review -- exp255: Small LGPA-D + GCN hidden 512 + 2-stage PSG + OA-SD

**Reviewer**: Claude Opus 4.6
**Date**: 2026-04-08
**Review round**: 1

---

## 1. design.md 审查

design.md 存在且结构完整。动机清晰：在 Small 骨干上通过增大 GCN 隐藏层维度（256->512）提升结构分支表征容量。

### 单变量原则

- vs exp254b (Small + 2-stage PSG + GCN 256): **仅 POSE_GCN_HIDDEN 改变**。满足单变量原则。
- vs exp249 (Small + 1-stage PSG + GCN 256): **两个变量改变**（GCN hidden + PSG stages）。设计文档已注明这不是单变量对照，可接受——exp254b 才是正式对照组。

### 参数增量估算（纠正 design.md）

design.md 声称 ~1.2M 新增参数。实际计算：

**GCN 内部（SkeletonGCN class, 2-layer, feat_dim=768, hidden_dim=512）**:
- Layer 0: Linear(768, 512) = 768x512 + 512 = 393,728
- Layer 1: Linear(512, 768) = 512x768 + 768 = 393,984
- LayerNorm(512) = 1,024; LayerNorm(768) = 1,536
- GCN total: ~790K

vs hidden_dim=256:
- Layer 0: Linear(768, 256) = 197,120
- Layer 1: Linear(256, 768) = 197,376
- Norms: 512 + 1,536
- GCN total: ~396K

**增量: ~394K 参数**（从 ~396K 到 ~790K），不是 design.md 声称的 ~600K。design.md 的计算有误（错误地列入了 Linear(512,512) 层，但 2-layer GCN 没有这一层——Layer 0 输出 hidden_dim=512，Layer 1 输出 feat_dim=768）。

**严重程度: Low**。参数估算不影响运行，实际增量更小（~0.4M），内存影响更小。

---

## 2. 代码审查: skeleton_gcn.py

### 2a. SkeletonGCN.__init__ 层构建逻辑

```python
# Lines 96-103
in_dim = feat_dim  # 768
for i in range(num_layers):  # num_layers=2
    out_dim = hidden_dim if i < num_layers - 1 else feat_dim
    self.layers.append(nn.Linear(in_dim, out_dim))
    self.norms.append(nn.LayerNorm(out_dim))
    in_dim = out_dim
```

With hidden_dim=512, num_layers=2:
- i=0: out_dim=512 (hidden), Linear(768, 512), LN(512). in_dim=512
- i=1: out_dim=768 (feat_dim), Linear(512, 768), LN(768)

**CORRECT.** Dimensions chain correctly. No hardcoded 256 anywhere.

### 2b. Forward pass

```python
# Lines 130-140
h = x  # (B, 17, 768)
for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
    h = torch.matmul(adj, h)  # (B, 17, dim) -- adj is (17,17) or (B,17,17)
    h = layer(h)
    h = norm(h)
    if i < len(self.layers) - 1:
        h = F.relu(h, inplace=True)
return x + h
```

- Step 0: matmul adj(17,17) @ h(B,17,768) -> (B,17,768), then Linear(768,512) -> (B,17,512), LN, ReLU
- Step 1: matmul adj(17,17) @ h(B,17,512) -> (B,17,512), then Linear(512,768) -> (B,17,768), LN, no ReLU
- Residual: x(B,17,768) + h(B,17,768) -> (B,17,768)

**CORRECT.** Shape compatibility holds. Adjacency matrix broadcast is safe (17x17 @ BxNxD).

### 2c. Adjacency matrix

COCO skeleton edges (16 edges) + extra edges (2 nose-shoulder) + self-loops. Symmetric normalization D^{-1/2}AD^{-1/2}. Standard GCN formulation. No issues.

### 2d. Zero-init last layer

Lines 106-107: `nn.init.zeros_` on last layer weight and bias. This means GCN output starts as all-zeros, residual connection makes output = input. Good for training stability. **CORRECT.**

---

## 3. 代码审查: pose_backbone_model.py

### 3a. GCN 创建 (Lines 190-211)

```python
gcn_hidden = getattr(cfg.MODEL, 'POSE_GCN_HIDDEN', 256)  # Line 193
...
self.skeleton_head = SkeletonGCNHead(
    feat_dim=self.in_planes,
    hidden_dim=gcn_hidden,  # Line 199
    ...
)
```

Config value correctly read via `getattr` with fallback 256. Passed as `hidden_dim` to `SkeletonGCNHead`. **CORRECT.**

### 3b. SkeletonGCNHead -> SkeletonGCN relay (Lines 411-417)

```python
if self.use_gcn:
    self.gcn = SkeletonGCN(
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_joints=17,
    )
```

`hidden_dim` parameter correctly passed through from `SkeletonGCNHead.__init__` arg to `SkeletonGCN`. **CORRECT.**

### 3c. self.in_planes for Swin-Small

Swin-Small: embed_dims=96, depths=(2,2,18,2), num_features = [96, 192, 384, 768].
Stage 3 output = 768 = feat_dim for GCN. Same as Tiny. **CORRECT.**

### 3d. PSG stages

`POSE_PSG_STAGES=[-2,-1]` resolved to Stage 2 (idx=2) + Stage 3 (idx=3) for 4-stage Swin. Already validated in exp251/exp254 reviews. **CORRECT.**

---

## 4. config/defaults.py 安全性

Line 136: `_C.MODEL.POSE_GCN_HIDDEN = 256`

Default remains 256. exp255 overrides via CLI `MODEL.POSE_GCN_HIDDEN 512`. No existing experiments are affected. **SAFE.**

---

## 5. Optimizer 注册

GCN parameters are part of `self.skeleton_head`, which is an `nn.Module` attribute of `PoseBackboneModel`. All parameters are auto-discovered by `model.parameters()` and included in the optimizer. No manual param group registration needed. Verified: `SkeletonGCNHead` contains `self.gcn = SkeletonGCN(...)` as `nn.Module`, whose `self.layers` and `self.norms` are `nn.ModuleList`. **CORRECT.**

---

## 6. 内存估算

- exp249 (Small, 1-stage PSG, GCN 256) used 6138 MiB on 5060 Ti (16 GB)
- 2-stage PSG adds PSG modules for Stage 2 (2 blocks): ~0.15M params, negligible memory
- GCN hidden 512 vs 256: ~394K extra params = ~1.5 MB fp32, negligible
- WITH_CP=True (gradient checkpointing) on Small backbone keeps activation memory controlled
- **Estimated total: ~6200-6400 MiB**. Well within 16 GB. **SAFE.**

---

## 7. 训练/测试对称性

GCN forward pass is identical in train and test. PSG modules are residual gating (no dropout, no stochastic behavior outside standard train/eval mode switching). LGPA, OA-SD, PLBOA only affect training. Test uses `pose_test_feat` config (inherited as `equal_concat` from base config). **CORRECT.**

---

## 8. AMP 安全性

SkeletonGCN uses standard Linear + LayerNorm + ReLU + matmul operations. No custom CUDA kernels, no fp16-unsafe operations. AMP autocast handles all these natively. **SAFE.**

---

## 9. 日志充分性

GCN head logs ID loss, triplet loss, and feature norms for the skeleton branch. PSG stages print at init time. Standard monitoring is sufficient to detect GCN collapse or training issues. **OK.**

---

## 10. 创新性质疑

This experiment is a hyperparameter sweep (GCN hidden 256 -> 512). It is not an innovation experiment. However, the context is clear: this is part of pushing the Small backbone recipe toward best results, not a main-line creative experiment. As a supporting capacity ablation for the paper's ablation table, it is acceptable.

---

## 发现汇总

| # | 级别 | 位置 | 描述 | 修复建议 |
|---|------|------|------|----------|
| 1 | Low | design.md | 参数增量计算有误 (~1.2M vs 实际 ~0.4M 增量) | 可选修正；不影响运行 |

No Critical, High, or Medium issues found.

---

## 结论

审查通过。exp255 是一个干净的单变量 GCN 容量消融实验。代码路径正确：POSE_GCN_HIDDEN=512 通过 config -> pose_backbone_model.py -> SkeletonGCNHead -> SkeletonGCN 完整传递，层维度自动适配。内存安全，optimizer 自动注册，无 AMP 风险。可以启动训练。
