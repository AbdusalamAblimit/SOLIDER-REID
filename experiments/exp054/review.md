# exp054 PGAM 审查报告

## 第一轮审查

**审查范围**: design.md, pose_attn_mask.py, pose_backbone_model.py, defaults.py, config yml

**发现的问题**:
- **[High] H1**: 阈值 0.3 应用于原始 heatmap logits（范围 -5~+15），几乎所有位置都会被判为 body，导致 PGAM 近似无效。需要加 sigmoid 归一化。
- **[Medium] M1**: design.md 中 -50 vs -100 表述略有歧义（minor）
- **[Medium] M3**: PGAM 仅在 PSG-only 分支中初始化，与 PAB/PXA 模式不兼容（safe fallback）
- **[Low] L1-L5**: 文档、padding、冗余条件等小问题

**结论**: 不通过（H1 需修复）

## 第二轮审查

**修复内容**: 在 max 之后、interpolate 之前加 `torch.sigmoid(body_conf)`

**审查结论**:
1. sigmoid 位置正确
2. 阈值 0.3 在 sigmoid 后合理（对应原始 logit ≈ -0.85，能覆盖有响应的体部区域）
3. 无其他遗漏

**结论**: ✅ 通过，可以开始训练
