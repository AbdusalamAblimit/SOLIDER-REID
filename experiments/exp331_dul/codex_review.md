# Codex Review — exp331 DUL

**Verdict**: approve（Codex 原话："needs-attention, **non-blocking after small fixes**"；3 个小 fix 已全部应用 → 放行）
**Date**: 2026-06-17
**Review round**: 1（Claude 双轮修复后的独立审）

## Codex 确认正确（substantive correctness）
- `classifier(bottleneck(s))` 在 plain softmax TransReID contract 下有效。
- **BN toggling 机制正确**：clean `model(img)` 更 BN 一次，sample 路 eval BN 不污染 stats，之后恢复 train。（Claude H2 修复确认有效）
- AMP/scaler、`add_param_group` 放置 OK。
- query/gallery 按 `seen < num_query` 切分匹配 repo 的 query-then-gallery val loader + R1_mAP_eval split。
- **KL 符号/形式正确**：dim-averaged `KL(N(mu,sigma^2)||N(0,I))`。（Claude H1 修复确认正确）
- collapse guard 充分（σ²→0 最可能失败，mean≈0 + std≈0 能抓；query/gallery diff 抓遮挡不敏感）。

## Findings（全部 non-blocking，已修）
- **Medium — eval σ² 日志未 clamp logvar**（train clamp 了）。**已修**：eval 也 `var_head(feat).clamp(-10,10)`。
- **Medium — "零初始化 identity start" 措辞不准**：σ²=1 是单位高斯**先验**起点（采样即注噪），非确定性 no-op。早期 CE 自然压 σ↓。**已修** design 措辞。
- **Low — DUL checkpoint 漏存 var_head**。**已修**：DUL 存 `{model, var_head}`。

## 结论
codex 审查通过（verdict: approve after non-blocking fixes, 3 小 fix 全部应用; 所有 substantive 正确性 Codex 已确认: BN/KL/classifier/split/AMP）。可启动训练。
