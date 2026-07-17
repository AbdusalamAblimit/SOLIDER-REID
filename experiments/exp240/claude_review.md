# exp240 PPA on Small — 审查报告

审查人: Claude Opus 4.6
日期: 2026-04-04

## 审查范围

a. `experiments/exp240/design.md` — 合理性、单变量原则
b. 配置安全性 — Tiny→Small 的变更项
c. OOM 风险评估 — exp230 前例分析
d. 与 exp237 (Tiny PPA) 的对照一致性

无新代码需要审查 — PPA 模块在 exp237 审查中已逐行通过。

---

## A. 设计文档审查

**动机合理**: exp237 在 Tiny 上验证了 PPA +0.5/-0.4 (mAP 正向、R1 接近持平)。
在 Small 上验证是标准 cross-backbone validation 流程。
对照组 exp206r (Small OA-SD GCN): 70.6/82.6 是正确的 baseline。

**单变量原则**: 满足。唯一变量是 backbone (Tiny→Small)。
PPA 配置 (w=0.5, 5 parts, no GiLt) 与 exp237 完全一致。

**论文价值**: 如果 PPA 在 Small 上也正向，可以在论文中声明 cross-backbone generalization。
如果负面，说明 PPA 效果依赖 backbone capacity，也是有价值的消融。

---

## B. 配置安全性

exp240 相对 exp237 需要修改的配置项:

| 配置项 | exp237 (Tiny) | exp240 (Small) | 备注 |
|--------|---------------|-----------------|------|
| TRANSFORMER_TYPE | swin_tiny_patch4_window7_224 | swin_small_patch4_window7_224 | 核心变量 |
| PRETRAIN_PATH | pretrained/swin_tiny.pth | pretrained/swin_small.pth | 已确认文件存在 (1.1GB) |
| BASE_LR | 0.0008 | 0.0004 | Small 标准 LR (exp206r 也用 0.0004) |
| TEST.IMS_PER_BATCH | 256 | 128 | OOM 缓解 |
| OUTPUT_DIR | exp237_ppa_tiny | exp240_ppa_small | 独立目录 |

**不应改变的配置项** (与 exp237 保持一致):
- POSE_PPA: True
- POSE_PPA_ASSIGN_WEIGHT: 0.5
- POSE_PPA_NUM_PARTS: 5
- POSE_PPA_GILT: False (默认)
- POSE_SKELETON_GCN: False (PPA 替代 GCN)
- POSE_ROA: False
- POSE_LOWER_BODY_OCC: True, PROB=0.7
- POSE_OA_SD: True
- POSE_BACKBONE_PSG: True
- POSE_ADDITIVE_ADAPTER: True (PAA)
- GLOBAL_LOSS_SCALE: 1.0

**注意**: 所有 PPA 和 PAA 设置不变，只有 backbone 相关的 3 项 + OOM 缓解改动。
这是干净的单变量实验。

### 架构兼容性

Swin-Small (embed_dims=96, depths=(2,2,18,2)) 输出 `num_features[-1] = 768`，
与 Swin-Tiny (embed_dims=96, depths=(2,2,6,2)) 相同。
PPA 的 `PartAssignmentHead(feat_dim=self.in_planes)` 接收 768 维 tokens，无需修改。
PSG 在 Small Stage 3 创建 18 个 gate (vs Tiny 的 6 个)，这在 exp206r 中已验证。
PAA 同理，18 个 adapter 已在 exp206r 中运行过。

---

## C. OOM 风险评估

**exp230 前例**:
- BT-PKD on Small, non-detached graph
- 首次运行: ep20 eval 时 CUBLAS crash (OOM → GPU error state)
- 修复: TEST.IMS_PER_BATCH=128, 成功运行到 ep110
- ep120 eval 再次 OOM (可能因 epoch 越大 model 越重)

**PPA vs BT-PKD 内存对比**:
- BT-PKD: non-detached backbone → per-keypoint cosine distillation，计算图较大
- PPA: non-detached backbone → softmax weighted pooling → per-part CE + triplet
- PPA 的计算图可能比 BT-PKD 更轻 (5 个 part weighted average vs 17 个 keypoint bilinear sample)
- 但两者都在 non-detached features 上操作，eval 时仍可能 OOM

**缓解措施**: TEST.IMS_PER_BATCH=128 已在 design.md 中指定。
exp230 证明此设置可以解决 Small non-detached 的 eval OOM (至少到 ep110)。

**[Medium] 残余风险**: exp230 在 ep120 eval 仍然 OOM。
如果 exp240 也遇到同样问题，可以进一步降低到 TEST.IMS_PER_BATCH=64，
或者在 eval 前手动 `torch.cuda.empty_cache()`。
这不阻塞启动，但需要在监控中关注。

---

## D. 与 exp206r 对照的公平性

exp206r 使用: GCN + PAA + OA-SD + PLBOA + **ROA**
exp240 使用: PPA + PAA + OA-SD + PLBOA + **无 ROA**

**ROA 差异**: exp206r 有 ROA (Random Object Augmentation)，exp240 没有。
但这与 exp237 保持一致 (exp237 也没有 ROA)。
更公平的对照应该是找一个 Small + OA-SD + 无 ROA 的实验。
但 design.md 已明确对照为 exp206r，这是可接受的——因为目标是看 PPA 能否在 Small 上正向，
而不是精确量化 PPA vs GCN 的差异。

**PARALLEL_AUG 差异**: exp206r 可能使用了 PARALLEL_AUG (3-view)。
但 PARALLEL_AUG 默认 False，且 exp230 证明 Small non-detached + 3-view 会 OOM。
exp240 不使用 PARALLEL_AUG 是正确的。
如果 exp206r 也没用 PARALLEL_AUG，则对照公平。

---

## 发现的问题

### [Medium] M1: ep120 eval OOM 残余风险

即使 TEST.IMS_PER_BATCH=128，exp230 仍在 ep120 eval 时 OOM。
exp240 可能遇到相同问题。

**建议**: 监控 ep100/ep110 eval 时的 GPU 内存。如果接近上限，
在 ep120 eval 前进一步降低 batch 或分步 eval。不阻塞启动。

### [Low] L1: ROA 对照不完全公平

exp240 无 ROA vs exp206r 有 ROA。
如果 exp240 结果低于 exp206r，不能完全归因于 PPA vs GCN——ROA 缺失也是因素。

**建议**: 在 monitor.md 结论中注明 ROA 差异。不阻塞。

---

## 审查结论

这是一个干净的 backbone scaling 实验:
- 无新代码，PPA 模块已在 exp237 审查中逐行通过
- 单变量: 只有 backbone Tiny→Small (+ 对应的 LR 和 pretrained 调整)
- 配置安全: in_planes=768 对两个 backbone 一致，PPA 无需修改
- OOM 缓解: TEST.IMS_PER_BATCH=128 已有 exp230 成功先例
- 发现问题均为 Medium/Low，不阻塞启动

**审查通过**
