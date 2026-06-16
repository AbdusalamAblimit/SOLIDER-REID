# Claude Broad Review — exp331 DUL (Identity-Conditioned Aleatoric Embedding)

**Date**: 2026-06-17
**Scope**: design.md + scripts/exp331_dul.py（全范围，Opus 子代理广审 + 修复后自查）

## 第一轮（Opus 子代理）findings + 修复

### High
- **H1 — KL 锚在 N(μ,I)，μ 解析抵消 → 对 μ 零梯度**，但 design 声称"KL 反向正则 μ/拉向类心"，机制不一致、实验测不到所声称的东西。**已修**：改 real DUL KL 锚在 **N(0,I)**：`kl = 0.5·mean(μ² + σ² − logvar − 1)`，正则 μ→0 + σ→1。design 公式同步改。诚实记：μ² 项与判别性张力——若 μ→0 伤检索即 DUL 在 ReID 失败原因(本实验要测的)。
- **H2 — `model.bottleneck`(BatchNorm) 每步前向两次**：`model(img)` 内部已在 clean μ 上更 BN 一次(=det)，sample 路 `model.bottleneck(s)` 在 train 模式**二次更新** running stats(来自高方差噪声样本)→ eval 用被污染 stats 归一化 clean μ → mAP 虚低 + 破坏单变量(det 更 1 次、dul 更 2 次)。**已修**：sample 路 `model.bottleneck.eval()`(不更新 stats)，BN 只在 clean μ 上更一次=det，单变量恢复 + 防污染。

### Medium
- **M1 — 真正塌缩方向是 σ²→0**(CE-on-sample 偏好低噪声)非 design 担心的 σ²→常数1；kl_weight=0.01 可能太弱挡不住 → 最可能 σ²→小 → dul≈det no-op。原 guard(query vs gallery σ² 均值)无法区分"塌到~0"与"遮挡不敏感"。**已修**：eval 加 σ² **mean + std 量级**日志(std~0 = 塌成常数/0)。
- **M2 — `model.classifier(model.bottleneck(s))` 仅 softmax/Linear classifier 可行**(arcface/circle 需 (feat,label))。**已查**：config 无 ID_LOSS_TYPE 覆盖、classifier=`nn.Linear(in_planes,num_classes)`、cos_layer 默认 off → `classifier(feat)` 可行。**已加防御 assert**(cos_layer off + has bottleneck/classifier)。

### Low
- **L1** 2-tuple vs 3-tuple 返回：vanilla TransReID = 2-tuple(已验证)，smoke 兜底。
- **L2** fp16 下 exp(logvar) 溢出：**已修** `logvar.clamp(-10,10)`。
- **L3** det 用 model 返回 score 作 CE(正确)；dul 弃之、用 sample score(正确)。

## 第二轮自查（修复后全范围复核）
- **单变量 det vs dul**：仅差 {var_head + 采样 s + KL}。BN 现两臂都单更新(clean μ)；同 make_dataloader/make_optimizer/scheduler/seed/epoch。✓
- **DUL 正确性**：reparameterization s=μ+ε·σ(σ=exp(0.5·logvar)) ✓；零初始化 var_head → logvar=0 → σ²=1 恒等起点 ✓；KL N(0,I) 正则 μ+σ ✓；CE-on-s + triplet-on-μ 连贯 ✓。
- **eval**：val_loader query-then-gallery，按 num_query 计数切 query/gallery σ²(R1_mAP_eval 同序) ✓；mu-matching=标准 eval feature ✓；σ² mean/std/diff 全记 ✓。
- **runtime**：AMP autocast + GradScaler ✓；optimizer.add_param_group(var_head) ✓；logvar clamp ✓；model.bottleneck/classifier 存在(make_model line 77/95 确认) ✓。
- **collapse guard**：σ² std + diff 双记 → 能区分塌到~0(std~0)、遮挡不敏感(std>0 diff~0)。✓

## 剩余风险（非 bug，机制风险）
- σ²→0 仍最可能(DUL no-op)；或 μ→0 伤判别性(DUL 伤 ReID)。两者都是 DUL 在 occluded ReID 真实失败模式，kill-switch(σ² 非退化 + mAP≥+1.0)会捕获。低概率是 bet 本身、非代码。

## 结论
H1/H2/M1/M2/L2 全部修复；单变量隔离恢复(BN 单更新)；KL 改 real DUL N(0,I)。runtime 待 smoke 验。**审查通过**（pending：Codex 独立审 + --smoke 运行时验证；任一暴露问题则修复并复审）。
