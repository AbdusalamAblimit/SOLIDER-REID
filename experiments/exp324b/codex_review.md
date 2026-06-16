# Codex Review — exp324b

**Verdict**: approve
**Date**: 2026-06-16
**Review rounds**: 2（首轮 needs-attention → 修复 → 二轮 approve），`codex --search exec -s read-only`

## 第一轮 (needs-attention)

与 Claude 首轮**独立收敛到同一 High 阻断项**：
- **High — Train/test objective mismatch**：训练只监督全局 masked-mean 特征（CE on BN global + triplet on pre-BN global），但主指标 part-MaxSim 用逐部位 L2-归一化向量余弦。只优化平均向量 → 逐部位向量可能弱判别。建议：加 part-level ID/triplet/contrastive 辅助，或改全局 cosine 为主指标 + 标注 part-MaxSim 为诊断。
- Medium：design/code 损失描述不一致；CACHE_QG dead；PKSampler smoke-limit 不鲁棒。
- Low：零可见图边界安全、triplet soft-margin 正确、同cam排除正确。
- 新颖性：`frozen DINOv2 dense tokens + lightweight trained projection head + pose-anchored mutually-visible part matching` 组合 plausibly distinct，但单独的 "pose visible-part matching" / "FM ReID adaptation" 非新；最强 claim = low-trainable-param frozen-DINO dense-correspondence route for occluded ReID，需对 PVPM/KPR/PFD/CLIP-local 做对照。

## 第二轮 (approve)

> **Verdict: approve**. No blocking/high/medium findings. H1 is resolved well enough to start the gated run; remaining items are low-severity cleanup/caveats.

逐项确认修复：`step()` 七返回值在所有消费处一致；优化器 param group 只切可训练 tensor、1-D BN gamma 入 no-decay；eval 仍正确排除 same-pid/same-cam。

**Findings (all Low)**：
- 文档残留（脚本 docstring + design.md:19/21 仍描述旧损失）→ **已修**（更新脚本 docstring 与 design 数据流/损失/缓存描述）。
- H1 残余 metric-form gap：per-part CE 监督 raw 投影向量、eval 归一化做余弦 MaxSim——**acceptable proxy supervision，直接修复了旧的 global-only mismatch**；若 part-MaxSim 仍弱，下轮可用 normalized/cosine classifier 或 per-part metric loss，本轮不必。
- PKSampler `max(1,…)` 避免零 batch，但极小 `--limit_train` 仍可能 <P 身份；**全量训练不受影响**。

## 结论

codex 审查通过（approve）。首轮唯一阻断项 H1 已用 per-part 共享 ID CE（监督落在 part-MaxSim 同源投影空间）正确修复，文档残留已清，剩余均 Low 不阻断。可开训。
