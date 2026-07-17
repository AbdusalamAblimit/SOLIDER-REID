# Codex Review — Rank-Regret Efficiency Kill-Switch (`cvpb_rankregret_killswitch.py`)

**Verdict**: needs-attention → 全部 findings 已修复（fix 只会让 RI 看起来更差，与本 kill-switch 的 NO-GO 结论同向）
**Date**: 2026-06-24
**Review round**: 1（`codex --search exec -s read-only`，联网查 novelty）

## Findings（codex 原文，逐条处置）

**High — `cheap-estRI` 在同一 eval 集上用 oracle RI 标签拟合（无 cross-fit，泄漏）。**
→ 修复：改 5-fold cross-fit（`crossfit_score`），每个 query 的路由分由没见过它自己 (cheap→target) 对的模型 OOF 产生。RI_hat 与 APgap_hat 都走 OOF。这让 deployable RI 数字**更保守**（更差），结论更稳。

**High — Test D 静态 baseline 不公平/不可部署（best_proxy 可能是 full-side，却当 cheap 用；且固定 higher→full 不按符号）。**
→ 修复：(a) 拆成 ORACLE static（含 full-side，仅作上界，标注不可部署）与 DEPLOY 系列；(b) 新增 **cheap-only static AP-gap 路由（cross-fit）= 公平正面对手**（与 RI_hat 同输入）；(c) 新增单最佳 cheap proxy 路由；(d) 所有 score 按观测 rho 符号对齐 AP_gap（`s_sign`/`bcp_sign`）。结论：deployable RI **始终输给** deployable cheap-static AP-gap（CFPER），在省算力 stage 连 random 都打不过。

**High — Test B `full_feat_norm` 控制失效（特征已 L2-norm，norm≈1）。**
→ 修复：抽取时**在 L2-norm 前**记录 full + 每个 stage GAP 的 raw 模长，存进 cache；新增 `cheap_rawnorm(neg)`（可部署）与真 `full_feat_norm(neg)`。partial 控制现 stack 8 个真静态代理。这**加强**了静态控制 → RI 偏相关更弱 → 结论更稳。

**Medium — deployable RI_hat 可能塌成 cheap difficulty ensemble（就是 CFPER）。**
→ 这正是 kill-switch 要测的。新增的 cheap-static AP-gap 公平对手直接回答：deployable RI ≤ deployable cheap-static（4/4 配置），坐实塌成 CFPER。

**Low — 设计文档默认 stage 过时（doc 说 stage2/3，脚本 stage1）。**
→ 修复：design.md 改为默认 stage1（cum 0.167，真省算力），注明 Swin FLOPs 集中 stage-2。

**Low — compute 记账公式对但打印文案误导（说 routed 付 cheap+full）。**
→ 修复：打印改为「full 子集 cheap stem，routed 付 full=1.0，cheap-exit 付 cheap_compute」。

## Novelty Check（codex 原文要点）
未找到「按 cheap-vs-full 检索排名分歧路由 person-ReID 算力」的直接先例，但 adaptive-compute/early-exit ReID 大类被占（DaReNet 1805.08805 / HashReID 2308.11900 / CtF 2008.06826 / AcuRank 2505.18512），本地 CFPER 笔记是最近的 ReID 效率先例（query difficulty / visibility early-exit）。**novelty 只在窄口「rank-regret 路由」成立，且需 Test B 把 RI 与静态难度切开。** 本次实验证明这一窄口**关不上**：RI 在所有配置都被静态难度代理碾过。

## 结论
codex 审查通过（needs-attention 的全部 findings 已修复，且修复方向与 NO-GO 结论一致——更公平的 baseline + cross-fit + 真 raw-norm 控制后，RI 仍 4/4 配置 DEAD）。
