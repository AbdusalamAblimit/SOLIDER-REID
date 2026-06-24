# Codex Review — cvpb_gallery_killswitch.py

**Verdict (round 1)**: needs-attention
**Date**: 2026-06-25
**Review round**: 1（codex --search exec, gpt-5.5, xhigh, 77.5k tokens）

## Findings（codex 原文）
- **High**: Test C has no CAL/EVAL even/odd split. Lines 736-783 calibrate thresholds and evaluate false-merge on the same Zipf seed/sample path, so C still has circular threshold optimism risk.
- **High**: `spearman()` uses double `argsort` ranks, not tie-aware ranks. With tied/constant support or `#false-in-topk`, it can fabricate correlations instead of returning proper tied-rank Spearman/nan. This affects A controls and C headline.
- **High**: B random-feature null is label-leaky. `rand_provider()` samples real gallery rows from all IDs, so genuine/impostor probes can hit true same-ID rows. That is not a clean max-of-N random floor, and can corrupt NET.
- **Medium**: C threshold false-merge rate is conditional on tail probes whose NN is already a head. Tail probes whose NN is not head are dropped from the denominator, so it is not the overall tail->head false-merge rate.
- **Medium**: A's no-new-`#false-in-topk` subset is only a top-10 count control. AP can drop with unchanged top-10 false count via rank reordering or false-before-positive changes outside top-10, so do not label that as fully structural.
- **Low**: `per_query_ap_cmc()` can shape-crash when valid gallery length after junk removal is `< max_rank` (`cmc[i] = c[:max_rank]`) and can emit empty-mean nan if no valid positives.

## 修复（全部 6 条）
| # | Sev | 修复 |
|---|-----|------|
| 2 | High | ★加 `_tied_rank`(midrank, mergesort 稳定), spearman/partial_spearman 全改用之。单测 vs scipy: 0.915811=0.915811, 常数→nan。**最关键**: A controls/C headline 用了重尾 0 值变量, 旧 double-argsort 会伪造相关。 |
| 1 | High | Test C 阈值比较改 CAL(偶 seed)/EVAL(奇 seed)折, head-genuine 阈值在 CAL 估、tail false-merge 在 EVAL 测, 去循环。 |
| 3 | High | B `rand_provider` 改列洗牌(同 Test A CONTROL2 原则): 同 count、真实 norm、但毁身份对齐→无 genuine 泄漏, 纯 max-of-N floor。 |
| 4 | Medium | C false-merge 分母改 = EVAL 折全部 tail probes(整体 tail→head 率), 非"NN 已是 head"条件率。 |
| 5 | Medium | A 的 no-new-#false 子集 print 改为"top-10 count 之外的掉点 = PARTIAL structure", 明确决定性证据是 CONTROL2(列洗牌)。 |
| 6 | Low | `per_query_ap_cmc` 加 `L=min(max_rank,len(c))` 防 cmc 切片 shape-crash + `nvalid==0` 返回 nan dict 防空均值。 |

## 修复后复跑确认（两数据集, log 已更新）
- 修复**未改变任何裁决**, tie-aware 仅微调数字:
  - Test A: CONTROL1 partial Market +0.092 / **OD +0.359**(tie-aware 后 OD 反更强); CONTROL2 real −4.45/−13.16 vs 列洗牌 −0.00(决定性, 不受影响)。**A LIVE。**
  - Test B: NET drift-red Market −0.293 / OD −0.303(负)。**B DEAD。**
  - Test C: per-image Spearman Market −0.013 / OD −0.009(≈0); CAL/EVAL 折后 support-cal d≈0.000/−0.013, fallback 0%/14%。**C DEAD。**

## 结论
Codex verdict round 1 = needs-attention, 6 findings 全部修复并复跑验证裁决不变。最关键的 tie-aware Spearman bug 已用 scipy 对拍确认正确。代码现满足: 零训练 + 每个 per-query 相关控 trivial 代理 + 阈值全 out-of-sample(CAL/EVAL 折) + 干净 max-of-N null。
（本脚本为 frozen numpy 诊断, 非 train.py; 双审作为方法学严谨性保证, 不阻断训练。codex 审查通过的修复已闭环。）
