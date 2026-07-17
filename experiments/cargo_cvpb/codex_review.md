# Codex Review — CVPB (OVP-Mem)

**Verdict**: approve
**Date**: 2026-06-22
**Review round**: 第 2 轮(H1 修复后重审)

## Findings

第 1 轮(原始代码): **needs-attention**
- High: OVP 冷启动梯度退化(masked CE 单类≈0 + τ=0.05 尖峰)。→ **已修**(--ovp_warmup 线性 warmup + inited 列数日志)。
- High: novelty 撞 CMPC(跨模态原型 momentum memory InfoNCE 结构同构)。→ **接受**: OVP-Mem 当 empirical auxiliary 不当 headline(out of scope for code review)。
- Medium ×3 / Low ×2: fp32 autocast / update-after-step 注释 / view-group 平均 / strict=False / device 硬编码。

第 2 轮(H1 修复后, scope=代码正确性): **approve**
- Critical/High/Medium: none。
- Low: ① OVP loss 注释说 fp32 但在 autocast 内(非 blocker, cos/τ∈[-20,20] 不溢出); ② --use_afd --afd_cvfc 时 CVFC 输出未加 loss(与当前 --ovp 纯 BoT 运行无关)。
- Checked OK: warmup 正确 wired(--ovp_warmup L194 / ovp_lambda_eff L252 / loss 用 L279 / 日志 L312); --ovp off 精确复现 baseline; BN 维度匹配 model.in_planes; cold-start mask 防未初始化原型成正/负样本; train/test 对称(OVP 仅训练期)。

## 结论
codex 审查通过(第 2 轮 verdict=approve)。第 1 轮 needs-attention 的 H1 冷启动已用 warmup+日志修复(重审确认正确 wired), novelty 撞 CMPC 接受为 empirical auxiliary(进稿前用 set-matching+Swin port 差异化 headline, 不阻断 empirical 训练)。代码正确性确认, 可启动 kill-switch #2 训练。
