# exp365 Wildlife species-conditioned — monitor

## GiraffeZebraID cheap kill-switch（frozen MegaDescriptor-L-384, 零训练, 2026-06-28）

数据：GiraffeZebraID（lila, 非 kaggle, 9.7G）6925 图 / 2056 个体 / 2 物种（zebra_plains 6286 + giraffe_masai 639）。split q=1142 g=4869（每 id 第1张 query 其余 gallery）。本地 MPS 抽 MegaDescriptor 特征(1536维, bbox crop)。

| 测量 | 值 | 判定 |
|---|---|---|
| baseline all-species mAP | 70.95 (R1 78.02) | |
| **wrong-species in top10** | **0.001** | MegaDescriptor 已把 species 分开 |
| per-species centering oracle | **+0.15** | <+3 → Kill |
| same-species vs all-species (species 干扰) | **+0.05** | <+1 → Kill |
| false in top10 | 0.712 | 错误多但都同物种内 |

**VERDICT DEAD**：codex Kill 线命中——MegaDescriptor（动物 ReID SOTA）frozen 已 species-agnostic 强（wrong-species 0.001），per-species centering 没空间（+0.15），species 干扰不存在（+0.05）。

## ★核心发现

Wildlife ReID 真难点 = **同物种内细粒度个体区分**（false_top10=0.712 都是 same-species hard negatives，wrong-species 仅 0.001），**不是 species 干扰**。codex 的 species-conditioning（SCREA: Species-Conditioned Residual Evidence Adapter）**解决错方向**——species 已被 MegaDescriptor 分开，需要解决的是同物种内细粒度区分，那是标准 ReID 问题，无 species-conditioning 新机制。

## 局限（诚实标注）

GiraffeZebraID 只 2 物种（zebra/giraffe 差异大，species 本来好分 → wrong-species 0.001 符合预期）。近缘多物种（WildlifeReID-10k 10k 个体含近缘 species，kaggle 需 key 没有）的 species 干扰没验。**但即使近缘物种 species 干扰更大，真难点（同物种内细粒度）species-conditioning 也解决不了**——core argument 不依赖物种数。

## 决定

species-conditioning 方向偏死（不补 SCREA adapter rank/head/loss 小变体，codex 明确）。换量级 codex 给的方向（第一名 lattice real-data 偏 test-time + 第二名 Wildlife species-conditioning cheap probe 偏死）都偏弱。LM-ReID 6.5（现有最强 B 类，exp359 文档全）是兜底。

下一步：① 可求用户 kaggle key 严谨验 WildlifeReID-10k 近缘多物种（确认 species-conditioning 死，但 core argument 已指向同物种内细粒度）；② 或转新方向（codex 再调研换量级）。cheap kill-switch 价值：零训练几分钟（GiraffeZebraID lila）证 species-conditioning 偏死，没下 kaggle 全量 24.7G 才发现。

## local-verifier cheap probe（codex 建议最后一搏, LoFTR rerank, 2026-06-28）

针对真难点（同物种内细粒度 hard neg），验 LoFTR local matching 能否纠正 MegaDescriptor 的 same-species false。180 query（top-k 内有正样本的）采样：

| | false_top10 | R1(in-topk) |
|---|---|---|
| baseline (MegaDescriptor) | 0.677 | 90.56 |
| LoFTR rerank | 0.677 | **97.22** |
| Δ | +0.000 | **+6.67** |

**★脚本 verdict bug 诚实纠正**：脚本 GO 条件用了 false_top10<0.60，但 **false_top10 对 rerank 无意义**（rerank 只重排 top-k 内顺序，集合不变 → false 比例必然 +0.000），导致误判 DEAD。**真信号 = ΔR1(in-topk)=+6.67**（LoFTR 把 top-k 内正样本提到 top1，远超 codex Top1 +1~2 标准）→ **local verifier 实际有效**：LoFTR local matching 在 same-species hard neg 上有 MegaDescriptor 没用好的判别信息。

**但诚实面对**（为何仍不主推）：
1. upper-bound（LoFTR expensive 双图 matching），训练端要蒸馏成 pattern-token 单次前向。
2. **撞 WildFusion**（arxiv 2408.12934：MegaDescriptor + LoFTR/LightGlue 融合，17 数据集 84.0% +8.5pp）—— local-feature 纠正 same-species 这条 codex 明确"先例太近，主创新风险高"。
3. codex 主线诚实判：LM-ReID 6.5 现实最强 B 类，收尾投稿；Wildlife 即使工程增益也撞 WildFusion。

## ★最终决定（2026-06-28）

收 LM-ReID 6.5（codex 判现实最强 B 类）。Wildlife species-conditioning 死 + 真难点（同物种内细粒度）local-verifier upper-bound 有（ΔR1+6.67）但撞 WildFusion → **记 future work，不主推**。换量级三方向（AG/DG/Wildlife）cheap probe 全证伪/偏弱，LM-ReID 6.5 兜底。教训：local-verifier 脚本 false_top10 指标对 rerank 无意义（栽了指标设计坑），ΔR1 才是真信号 —— 差点被错误 verdict 埋没真信号。
