# Gallery Hubness ReID 零训练 kill-switch 设计（2026-06-24, B+GOPL 双死后, 战略重评 r_2 主推）

## re-frame（彻底绕开 SMPL/遮挡/航拍三失败区）
> 大家以为强 ReID 失败来自"某个 query 没匹配好"（pairwise 相似度病），其实是 **少数 gallery 样本变成很多 query 的吸附点**——错误是 **many-to-one 的图库拓扑病**。ReID 不是独立 query-gallery pair matching，而是 **directed kNN graph retrieval**；隐藏变量不是 hard-negative distance，而是 gallery 的 **负向 in-degree / hub mass**（≠hard negative：hard 是对一个 anchor 近；hub 是对**很多不同身份**都近，是全局误吸附点）。

## novelty 切口（待 codex 核查, r_2 初判）
- hubness 在通用高维检索是成熟概念（CSLS/Mutual Proximity 理论可借），但 **ReID 里没人把 gallery 负向 in-degree 定义成失败主变量 + 训练端 anti-hub**。
- vs k-reciprocal re-ranking: 它用 reciprocal neighbor 做 test-time ranking, 不定义 hubness 为失败变量, 不做训练端 anti-hub。
- vs Pose2ID / Feature Centralization: 正样本邻居中心化, 我们是**负向吸附点**, 反的。
- 论文卖点=**负向图库吸附点是强 ReID 的失败拓扑变量**, 不是"又一个 re-ranking/neighbor 聚合"。

## 零训练 kill-switch（冻结强 Market/MSMT ckpt, 无训练）
找团队最强 Market + MSMT ReID ckpt（agent 在 log/ 找; 没有就用 SOLIDER-Swin 标准 ckpt, 报 sanity mAP）。Market + MSMT **双数据必须都成立**, occluded_duke 只当 sanity 不写遮挡故事。

**核心量:** `H_k(g) = #{ q | g ∈ top-k(q) 且 y_g ≠ y_q }` = gallery g 的负向 in-degree（被多少不同身份 query 误放进 top-k）。query 级 hub mass `M(q) = Σ_{g∈topk(q), y_g≠y_q} H_k(g)`。

**测试:**
1. **hub 集中度**: false top-1（以及 false top-10）是否集中在少数高 H_k gallery？top 1% hub gallery 至少吃掉 **20-30% 的 false hits** 才有戏（vs 均匀分布的期望 1%）。
2. **hub mass 解释力**: per-query AP/R1 误差 ~ M(q) 的相关, 对比 ~ feature-norm / top1-margin / camera-pair / #gallery-positives。hub mass 必须**解释更多**（partial 相关控住后者仍显著）。
3. **零训练干预**: `score'(q,g) = cosine(q,g) − λ·log(1+H_k(g))`（或 CSLS / Mutual Proximity hub 校正）。扫 λ, 看 mAP/R1 是否涨, **且收益是否集中在高 M(q) query**。
4. 双数据: Market + MSMT 都要 (1)(2)(3) 成立。

**破坏对照（决定生死）:**
- D1 置换 H_k（shuffle gallery 的 H_k 值）→ 干预增益必须消失（否则不是真 hub 信号）。
- D2 camera-correction 等价: 把干预换成纯 camera-aware（同相机降权 / CA-Jaccard 式）→ 若 camera 校正涨得一样多, hub 没独立价值（撞 DART³/CA-Jaccard）。
- D3 hub mass vs cheap proxy: 控住 feature-norm + top1-margin + camera-pair 后, M(q) 偏相关必须仍显著（否则=旧难度代理）。
- D4 正 vs 负 in-degree: 用**全部** in-degree（含同 ID）当对照 → 必须**负向 in-degree（跨 ID 误吸附）** 才是关键信号, 不是单纯"热门样本"。

**通过标准:** top1% hub 吃 ≥20% false hits（双数据）+ M(q) partial 相关显著强于 norm/margin/camera + score' 干预 mAP 小涨且收益集中高 M(q) + D1 置换破 + D2 不等价于 camera + D4 负向 in-degree 是关键。
→ 全过 = hub-mass 隐藏变量真实且独立 → 写训练版（Hub-Aware Retrieval Embedding: memory bank 存 cross-ID top-k in-degree, anti-hub margin, negative 权重从"离anchor近"改"是否全局误吸附点", 测试仍单 embedding 不变 test-time trick, beat triplet/k-reciprocal/camera-aware）。
→ 不过: D2 等价 camera 或 D1 置换也涨 → 降级转 r_2 备胎 Rank-Instability Adaptive（效率 Pareto 轴）或 r_3 遮挡最后一搏。

## 资产
Market/MSMT 在 lab-3090 主仓库 `data/`。强 ckpt agent 在 `log/` 找。复用 cvpb_gopl/containment kill-switch 的 extract/per_query_ap 基建。全 frozen + numpy/torch.no_grad。
