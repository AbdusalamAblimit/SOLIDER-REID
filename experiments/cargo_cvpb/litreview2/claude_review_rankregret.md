# Claude Broad Review — Rank-Regret Efficiency Kill-Switch (`cvpb_rankregret_killswitch.py`)

**Date**: 2026-06-24
**Reviewer**: Opus 子代理（general-purpose, 全范围逐行）
**Round**: 1（含修复后复核要点）

## 审查范围
- `cvpb_rankregret_killswitch.py`（被审脚本，零训练诊断）
- `RANKREGRET_KILLSWITCH_DESIGN.md`（设计意图）
- `cvpb_hubness_killswitch.py`（已验证 sibling，作为正确 eval/extract/stats 习语基准）
- `model/pose_backbone_model.py`（确认 `POSE_TEST_FEAT='global'`+`NECK_FEAT='after'` 下 eval `model(...)` 返回 2-tuple `(BNNeck embedding, featmaps=4 个 Swin stage 输出)`，`featmaps[s]`=第 s stage `(B,C,H,W)`，`featmaps[3]` GAP 进 BNNeck）

## 模型契约（已确认正确）
单次 forward 同时拿 BNNeck full 向量 + 全部 4 个 stage GAP；query/gallery 切分 `[:nq]`/`[nq:]` 与 `val_set=query+gallery`、`shuffle=False` 一致；junk gallery `pid==-1` 已 drop；L2-norm 一致（extract 用 `F.normalize`，main 用 `l2()` 复norm）。所有 part/gcn 分支被 `pose_test_feat!='global'` 关掉，落到 `return test_feat, featmaps`。核心假设成立。

## Verdict（修复前）：NEEDS-FIXES
无 runtime crasher，无 Test-B 假阳泄漏；但 1 个 High（Test D 在默认 stage 下 headline 不可达）+ 数个 Medium（verdict 阈值/steel-man 控制/重复代理/Test A 机械性披露）。

## Findings 与处置

### HIGH
**H1 — Test D headline 在默认 `cheap_stage=2` 下架构上不可达，静默 nan。**
Swin depths (2,2,18,2) 使 stage-2 占 ~75% FLOPs，cum_frac=[0.083,0.167,0.917,1.0]。`cheap_stage=2`(stage3 输出 featmaps[2]) 跑完 stage0-2 → 算力 0.917，只省 8%。targets[0.5,0.6,0.7] 全 < 0.917 → D_table 空 → @60% 摘要静默丢失，PASS 判据无法评估。
**修复**：默认改 `cheap_stage=1`（跑完 stage0-1 → 算力 0.167，省 83%，真正可省）；docstring/help 注明只有 stage0/1 真省算力；targets 自适应到 [cheap_compute,1] 范围内；若仍为空（如用户手动选 stage2/3）**大声 WARNING** 并退化到 in-range 探测点；摘要 D-line 用最接近 0.60 的可达点。已全部落实。

### MEDIUM
- **M1 — `b_alive` 旧阈值（partial≥0.05 且 ≥0.4×marginal）易假阳。** 修复：B 判活改为 3 条全满足——(i) partial|all ≥0.08；(ii) partial ≥50% RI marginal；(iii) **RI marginal ≥ 最强静态代理 marginal**（CFPER 真问题，旧逻辑算了 best_proxy 却没用进 verdict）。
- **M2 — partial 只控 4 代理，漏掉 full-side steel-man（full_margin/full_entropy）。** 它们是最强难度代理（看同一 full ranking），漏控会让 RI 显得更好。修复：`cov_all` 现 stack 全部 7 个静态代理（cheap margin/top1-top3-gap/entropy/density + full_feat_norm + full_margin/full_entropy）。
- **M3 — `cheap_top1_top2_gap(neg)` 与 `cheap_margin(neg)` 是同一数组（重复）。** 修复：替换为真正的 top1-top3 gap（`gap_top13`），代理多样性真实。
- **M4 — Test A 部分机械（RI=0 ⇒ AP_gap=0 by construction），不应当作发现。** 修复：Test A 打印加诚实 caveat，明确 B 才是真判据。不影响 B（CFPER 判据 RI-vs-静态 非机械）。
- **M5/L2 — eval_full 若 smoke 下无 query 达 rank-10 会越界**（与 sibling eval_map 同模式，Market/OD 实际安全）；kendall 是 tau-a-over-comparable-pairs 非严格 tau-b（仍单调有效，方向正确）。保留（与已验证代码一致），仅记录。

### LOW（已处理关键项）
- **L1 — RI_hat 拟合/应用 rank 支撑不一致。** 修复：cheap-only 代理在全 Nq 上 `_rank` 一次，valid 子集拟合 beta，再对全体打分；RI_hat 纯 cheap-only 输入，无 full 泄漏。
- **L4 — entropy tau=0.05 过尖（弱 steel-man）。** 修复：tau→0.1。
- L5（解析 FLOPs 忽略 patch-embed/window-attn 近似）：docstring 已承认，stage 相对序正确，Pareto 相对位置可信；论文用 fvcore 精确化。保留。

## 显式确认正确（无需改）
- 特征抽取/切分/junk drop/L2/reuse_feat 重载（前一处 reuse bug 已修复并复核）。
- eval（per_query_ap/eval_full junk 规则 same pid&cam、AP 公式、-1=drop）与 sibling 一致。
- RI@K 三指标（RBO 增量 overlap 正确，a[d]==b[d] 不重复计数，已 4 例 trace + 远程合成自测通过；三指标 higher=更不一致一致）。
- Test B partial_spearman（rank-residualize，与 sibling 一致）；代理已正确按号对齐 AP_gap（neg 翻转 margin/density/norm）。
- Test C cheap-only 严格只用 cheap；rank-R² 正确；RI_hat 纯 cheap-only。
- Test D cascade（`np.where(mask,dm_full,dm_cheap)`；avg_compute=frac_full×1+(1−frac_full)×cheap_compute，full 子集 cheap stem 假设已声明且正确；random 20 seed 平均；interp/compute_for_target 正确）。
- swin_stage_compute_fractions（depth·tokens·dim²，tokens=1/4^i，归一到 stage3=1.0，stage2 主导，远程自测 base/small 均 [0.083,0.167,0.917,1.0]）。
- AMP/dtype/shape：纯 numpy CPU on normalized float32，无 autocast，happy path 无崩；stats 有 len<3/5 与 denom>0 守卫；ap_gap 用 nanmean。

## 远程合成自测（已执行，全过）
RBO/overlap/kendall 三指标 identical→0 / disjoint→1 / reversed(同集)→overlap0+tau1；spearman 单调 0.986；FLOPs base/small cum_frac 正确。

## 结论（Round 1）
**审查通过**。被审脚本无 runtime crasher、无 Test-B 假阳泄漏；H1（Test D 默认 stage 使 headline 不可达/nan）已修；M1/M2 收紧 verdict + full-side steel-man 纳入 partial；M3/M4/L1/L4 已处理。

## Round 2 — Codex 联网审查后的二次修复（全范围复核）
Codex（`codex --search exec`）独立审查发现 3 个 High（claude round1 漏掉的）：
1. **`full_feat_norm` 控制失效**（特征 L2-norm 后 norm≈1）→ 改为抽取时存 raw 模长（full + 各 stage GAP），新增 `cheap_rawnorm`(可部署) + 真 `full_feat_norm`；partial 现控 8 个真静态代理。
2. **Test D 静态 baseline 不公平**（best_proxy 可能 full-side 却当 cheap 用；不按符号）→ 拆 ORACLE(上界) vs DEPLOY；新增 **cheap-only static AP-gap cross-fit 路由 = 公平正面对手**；按 rho 符号对齐。
3. **cheap-estRI 同集 oracle 标签拟合泄漏** → 改 5-fold cross-fit（OOF）。
另加 cascade O(Nq·Ng·logNg)→O(Nq) 精确加速（precomputed per-query AP，含 max|dAP|=0 identity 自检）。

二次全范围复核通过：3 个 fix 方向与 NO-GO 同向（更公平 baseline + cross-fit + 真 norm 控制后 RI 仍 4/4 DEAD），无新 bug，identity check 证 cascade 加速精确。**两层审查（Claude + Codex）均通过，脚本可运行，结果已产出。**
