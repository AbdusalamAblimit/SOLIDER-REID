Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/rr_logs
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019efa94-b3e7-7f83-ba97-d1b810a7bf63
--------
user
一个 ReID 团队的 **Gallery-Growth Tax** 诊断已零训练 kill-switch 证实 LIVE: frozen 强 ReID 模型(表示不变), 旧 query mAP 随**同域** gallery 膨胀 1x→10x 结构性掉点(Market −4.43 / Occluded-Duke −12.86 mAP, 量级 ≈ LReID 文献报的 catastrophic forgetting); CONTROL2 决定性(real distractor 掉 vs 列洗牌不掉 = 结构性非机械 count); CONTROL1 部分是 #false-in-topk trivial 但有结构残差(OD partial +0.36, Market 弱 +0.09)。

要做 CCF-B **方法稿**。设计最强结构:
① **re-attribution 实验(核心 headline)**: 怎么干净证明'LReID 报的 forgetting 部分是 gallery-growth-tax'? 需要什么 LReID setup(序列域 Market→MSMT→CUHK03/Duke?)? 对比 frozen 模型旧域 drop(纯 gallery-growth tax) vs 持续训练模型旧域 drop(forgetting+gallery-growth), 算 tax 占 reported forgetting 的比例? 给具体协议(用哪些数据/怎么切序列/测什么)。
② **method/remedy(必须有才是方法稿非纯分析)**: 候选——distractor-aware training(训练注入未来 gallery 压力)/ gallery-growth-aware 评测协议(分离 tax 的公平 metric)/ gallery-growth-robust embedding。哪个既是 method 又最不撞 backward-compatible LReID(arxiv 2403.10022)/re-index-free 线? 给 remedy 草案 + 它该 beat 什么 baseline + 廉价预检(若有)。
③ **撞车核查(联网 2024-26)**: backward-compatible LReID / re-index-free / historical-gallery / search-space-growth / open-world ReID 有没有做过 frozen-control 分离 forgetting? 这个 re-attribution 的确切 novelty 边界。
④ 诚实: 能撑 CCF-B **方法稿**吗(信心 1-10), 还是更像 analysis/short? 若偏 analysis, 加哪个轻 method 件能升级成方法稿?
务实中文, 给完整实验设计 + remedy 草案 + 撞车 + verdict + 信心。
codex
我先核查 2024-2026 相关线索，重点看 backward-compatible LReID、re-index-free、historical gallery / search-space growth / open-world ReID 有没有明确做 frozen-control 拆分。然后我会把协议和 remedy 设计压成可执行版本。
web search: 
web search: arXiv 2403.10022 backward-compatible lifelong person re-identification
web search: 
web search: "gallery" "growth" "person re-identification" "lifelong"
web search: 
web search: "Lifelong person re-identification" 2024 2025 benchmark gallery query together
web search: 
web search: site:arxiv.org/abs "lifelong person re-identification" "2024" "gallery"
web search: 
web search: https://arxiv.org/abs/2403.10022
web search: 
web search: 'Market' in https://arxiv.org/abs/2403.10022
web search: 
web search: https://arxiv.org/abs/2403.10022
web search: 
web search: https://arxiv.org/abs/2403.10022
web search: 
web search: https://ar5iv.org/html/2403.10022v2
web search: 
web search: "gallery growth" "re-identification"
web search: 
web search: "Generalising without forgetting" "lifelong person re-identification"
web search: 
web search: "Generalising without forgetting for lifelong person re-identification"
web search: 
web search: "PTKP" "person re-identification" "lifelong"
web search: 
web search: "open-world person re-identification" "gallery" "distractor" 2024
web search: 
web search: https://arxiv.org/abs/2407.16133
web search: 
web search: "Re-indexing Free Lifelong Person Re-identification"
web search: 
web search: "Continual learning for visual search with backward consistent feature embedding"
web search: 
web search: https://arxiv.org/abs/2103.12462
web search: 
web search: https://arxiv.org/abs/2103.12462
web search: 
web search: "survey" "open-world person re-identification" 2024
web search: 
web search: "LReID-Seen" "CUHK03" "Market-1501" "MSMT17" "DukeMTMC" "CUHK-SYSU" "order"
web search: 
web search: "LReID-Seen" person re-identification benchmark
web search: 
web search: https://arxiv.org/abs/2405.19005
web search: 
web search: "LSTKC" "Lifelong Person Re-Identification" 2024
web search: 
web search: https://arxiv.org/abs/2403.16003
web search: 
web search: "Long short-term knowledge consolidation" "lifelong person re-identification" 2024
web search: 
web search: https://arxiv.org/abs/2605.05027
web search: 
web search: "frozen" "gallery" "catastrophic forgetting" "person re-identification"
web search: 
web search: "frozen-control" "person re-identification"
codex
已经看完核心文献脉络：2024 的 backward-compatible LReID 明确把“gallery 也累积”拉进问题定义，但目标是免 backfilling / feature compatibility；2024-2026 的 LReID 新方法仍主要在抗遗忘、adapter、distillation、semantic anchor 上发力。我还没看到把旧域掉点拆成“模型遗忘 vs frozen 表示下 gallery 膨胀税”的明确控制实验。
codex
**结论先说**

能做，但不要把 claim 写成“LReID 的 forgetting 都是 gallery-growth tax”。更干净的主张是：

> 现有 LReID / backward-compatible LReID 的旧域掉点混合了两件事：模型表示漂移和 gallery 搜索空间膨胀。我们用 frozen factorial control 首次定量拆分二者，并提出 gallery-growth-robust training 来降低这部分 tax。

如果 re-attribution 证明 tax 占总旧域 drop 的 30% 以上，且 remedy 能稳定追回 tax 的 25%-40%，CCF-B 方法稿有戏。纯分析稿信心 4/10；加上方法件后信心 7/10。

**① Re-Attribution 实验**

核心不要只做 frozen vs continual 两条线，要做 2x2 factorial：

对旧域 `D_i`，阶段 `t`：

| 模型 | gallery | 分数 |
|---|---|---|
| `M_i` frozen | `G_i` | `S00`: birth isolated |
| `M_i` frozen | `G_1...G_t` | `S01`: pure gallery-growth tax |
| `M_t` continual | `G_i` | `S10`: pure model forgetting on base gallery |
| `M_t` continual | `G_1...G_t` | `S11`: deployment drop |

定义：

```text
TotalDrop = S00 - S11
FrozenTax = S00 - S01
BaseForgetting = S00 - S10
Interaction = S01 + S10 - S00 - S11

TaxShare_raw = FrozenTax / TotalDrop
TaxShare_shapley = 0.5 * [(S00-S01) + (S10-S11)] / TotalDrop
```

`TaxShare_shapley` 更适合做 headline，因为它平均了“先换 gallery”和“先换模型”的路径。

主 setup 选 2024 backward-compatible LReID 的四域顺序：`Market1501 -> DukeMTMC -> CUHK-SYSU -> MSMT17`。原因很简单：这篇已经明确把 LReID 中“gallery 也累积”作为现实问题，并提出 all-query/all-gallery evaluation；它的表里也用这四个域和这个顺序，MSMT17 gallery 很大，最容易显出 tax。论文说明其协议会把 gallery 特征累积、避免 backfilling，并用全 query / 全 gallery 做更现实评测。([ar5iv.org](https://ar5iv.org/html/2403.10022v2)) ([ar5iv.org](https://ar5iv.org/html/2403.10022v2))

辅 setup 做两组：

1. `Duke -> Market -> CUHK-SYSU -> MSMT17` 和 `Market -> CUHK-SYSU -> Duke -> MSMT17`，复用该文不同 order 对照。([ar5iv.org](https://ar5iv.org/html/2403.10022v2))  
2. AKA/LReID-Seen 五域 benchmark：`CUHK03, Market1501, MSMT17, DukeMTMC-ReID, CUHK-SYSU ReID`。AKA 定义了 LReID：当前阶段只能访问当前训练域，测试所有 encountered domains；其 benchmark 也列了这些域和 train/query/gallery 规模。([arxiv.org](https://arxiv.org/abs/2103.12462))

每个旧域测：

- `mAP / Rank-1 / mINP`
- `GGT@x`: gallery 从 `1x, 2x, 4x, 8x, full` 的 drop 曲线
- `AUGC`: area under gallery-growth curve，横轴用 `log(|G|)`
- query-level bootstrap CI，按 query ID bootstrap，不只报均值
- 每阶段 `t=i+1...T` 都测，不只 final

关键控制：

- `REAL`: 加真实新域 gallery distractors
- `SHUFFLE`: 对 distractor score column 做 query-wise permutation，保持数量但破坏 query-specific hard distractor 结构
- `COUNT-MATCH`: 匹配 `#false-in-topk` 后重算 AP residual
- `SAME-DOMAIN TAX`: Market / Occluded-Duke / MSMT 单域内按同域 distractor 扩张，作为你们 kill-switch 的外推证据

最强 headline 表：

```text
Old domain | TotalDrop | FrozenTax | BaseForgetting | Interaction | TaxShare_shapley
Market     | ...
Duke       | ...
CUHK-SYSU  | ...
Average    | ...
```

注意边界：如果某些经典 LReID 论文只在 `Q_i vs G_i` 上报旧域 mAP，那不是 gallery-growth tax，而是 base-gallery forgetting。你们只能重归因“deployment/all-gallery LReID drop”或“把 LReID 放进真实累计 gallery 后观察到的旧域 drop”。

**② Remedy 选哪个**

不要把“gallery-growth-aware evaluation protocol”单独当方法。它是贡献一，但不是方法稿核心。

最稳的 method 是：

> Gallery-Growth Robust Learning / Tax-Aware Distractor Training

它不碰 backward compatibility，也不解决 re-index-free，所以不正面撞 2024 backward-compatible LReID 或 2025/2026 re-index-free 线。它解决的是“同一 embedding 在 gallery 规模变大时极端负样本进入 top-rank”的 rank-stability 问题。

草案：

训练时为每个 anchor 构造 growth pool：

```text
P(q): same-ID positives
N_g(q): same-domain / cross-domain distractor queue, sampled at growth factor g in {1,2,4,8,10}
```

用 extreme-negative AP surrogate：

```text
p_softmin = -1/a * logsumexp_{p in P(q)}(-a * s(q,p))
n_extreme =  1/b * logsumexp_{n in N_g(q)}( b * s(q,n))

L_tax = E_g softplus(n_extreme - p_softmin + m0 + lambda * log g)
```

再加一个 rank-stability loss：

```text
L_stable = KL( topK_rank_distribution(q, G_1) || topK_rank_distribution(q, G_g) )
```

总 loss：

```text
L = L_ce + L_triplet + alpha * L_tax + beta * L_stable
```

最小版本只上 `L_tax`，别一开始堆太多。

它该 beat 的 baseline：

- CE + Triplet / Bag-of-Tricks
- Batch-hard triplet with larger batch
- SupCon / Circle loss / ArcFace
- XBM / memory-bank hard negative
- Open-Set Biometrics 2024 的 relative threshold / max-negative 类 loss，作为强撞车 baseline；那篇明确说现有 loss 忽略 imposter score 的相对大小，并提出降低最大负样本分数。([arxiv.org](https://arxiv.org/abs/2407.16133))
- LReID 侧：Fine-tuning, LwF, AKA, PTKP, KRKC, DRE, AdalReID, BC-LReID；至少做 `strong LReID + TaxLoss` 插件式对照

廉价预检：

1. 用现有 frozen score matrix 计算 `p_softmin - n_extreme` 是否能预测 10x gallery 下失败 query。AUC > 0.75 才值得训。  
2. 只训 Market/Occluded-Duke 20 epoch：标准 `1x mAP` 不能掉超过 0.5，`10x GGT` 至少追回 25%。  
3. 在 LReID 四域 final all-gallery 上插入 TaxLoss，旧域平均 tax drop 至少降 20%，新域 MSMT 不明显掉。

**③ 撞车边界**

已知近年相关线：

- 2024 backward-compatible LReID：关注 old/new model feature compatibility 和免 backfilling；它确实提出 all-query/all-gallery evaluation，但没有做 frozen model 2x2 attribution，把掉点拆成 tax / forgetting。([ar5iv.org](https://ar5iv.org/html/2403.10022v2))  
- 2024 AdalReID：adapter + auto-selection，核心是 domain-specific knowledge preservation，不是 gallery growth tax。([arxiv.org](https://arxiv.org/abs/2405.19005))  
- 2024 DRE：transformer diverse representations + knowledge update/preservation，仍是抗遗忘机制。([arxiv.org](https://arxiv.org/abs/2403.16003))  
- 2026 PAD：vision-text semantic anchor，还是 semantic drift / forgetting / adaptability。([arxiv.org](https://arxiv.org/abs/2605.05027))  
- 2024 TEMP：固定 gallery 特征下做 test-time similarity modification，关注 temporal distribution shift，不是 gallery size attribution。([arxiv.org](https://arxiv.org/abs/2403.14114))  
- 2024 Open-Set Biometrics：最接近 remedy 的负样本尾部思想，但任务是 open-set detection/identification，不是 closed-set ReID mAP under gallery growth，也没有 LReID frozen-control re-attribution。([arxiv.org](https://arxiv.org/abs/2407.16133))

所以 novelty 边界写法：

> 不是“第一个注意到 gallery 会累积”；BC-LReID 已经注意到了。  
> 是“第一个把旧域性能下降分解为 representation forgetting 与 gallery-growth tax，并证明一部分所谓 deployment forgetting 在 frozen embedding 下已经发生”。  
> 方法上不是普通 hard negative mining，而是显式优化 gallery-size-conditioned extreme distractor risk。

**④ Verdict**

能撑 CCF-B 方法稿，但条件很硬：

- `TaxShare_shapley` 平均最好 >30%，至少两个旧域显著；
- shuffle/count controls 必须证明不是机械 count；
- TaxLoss 必须降低 growth curve 斜率，而不是只涨普通 mAP；
- remedy 不能靠 re-ranking / NFC / test-time trick；
- 必须和 Open-Set Biometrics 的 max-negative loss 做 baseline，否则 remedy 会被打成 hard-negative 旧酒。

信心：  
分析发现本身 8/10；纯分析投稿 4/10；加 Tax-Aware Distractor Training 且实验达标，CCF-B 方法稿 7/10。
tokens used
115,530
**结论先说**

能做，但不要把 claim 写成“LReID 的 forgetting 都是 gallery-growth tax”。更干净的主张是：

> 现有 LReID / backward-compatible LReID 的旧域掉点混合了两件事：模型表示漂移和 gallery 搜索空间膨胀。我们用 frozen factorial control 首次定量拆分二者，并提出 gallery-growth-robust training 来降低这部分 tax。

如果 re-attribution 证明 tax 占总旧域 drop 的 30% 以上，且 remedy 能稳定追回 tax 的 25%-40%，CCF-B 方法稿有戏。纯分析稿信心 4/10；加上方法件后信心 7/10。

**① Re-Attribution 实验**

核心不要只做 frozen vs continual 两条线，要做 2x2 factorial：

对旧域 `D_i`，阶段 `t`：

| 模型 | gallery | 分数 |
|---|---|---|
| `M_i` frozen | `G_i` | `S00`: birth isolated |
| `M_i` frozen | `G_1...G_t` | `S01`: pure gallery-growth tax |
| `M_t` continual | `G_i` | `S10`: pure model forgetting on base gallery |
| `M_t` continual | `G_1...G_t` | `S11`: deployment drop |

定义：

```text
TotalDrop = S00 - S11
FrozenTax = S00 - S01
BaseForgetting = S00 - S10
Interaction = S01 + S10 - S00 - S11

TaxShare_raw = FrozenTax / TotalDrop
TaxShare_shapley = 0.5 * [(S00-S01) + (S10-S11)] / TotalDrop
```

`TaxShare_shapley` 更适合做 headline，因为它平均了“先换 gallery”和“先换模型”的路径。

主 setup 选 2024 backward-compatible LReID 的四域顺序：`Market1501 -> DukeMTMC -> CUHK-SYSU -> MSMT17`。原因很简单：这篇已经明确把 LReID 中“gallery 也累积”作为现实问题，并提出 all-query/all-gallery evaluation；它的表里也用这四个域和这个顺序，MSMT17 gallery 很大，最容易显出 tax。论文说明其协议会把 gallery 特征累积、避免 backfilling，并用全 query / 全 gallery 做更现实评测。([ar5iv.org](https://ar5iv.org/html/2403.10022v2)) ([ar5iv.org](https://ar5iv.org/html/2403.10022v2))

辅 setup 做两组：

1. `Duke -> Market -> CUHK-SYSU -> MSMT17` 和 `Market -> CUHK-SYSU -> Duke -> MSMT17`，复用该文不同 order 对照。([ar5iv.org](https://ar5iv.org/html/2403.10022v2))  
2. AKA/LReID-Seen 五域 benchmark：`CUHK03, Market1501, MSMT17, DukeMTMC-ReID, CUHK-SYSU ReID`。AKA 定义了 LReID：当前阶段只能访问当前训练域，测试所有 encountered domains；其 benchmark 也列了这些域和 train/query/gallery 规模。([arxiv.org](https://arxiv.org/abs/2103.12462))

每个旧域测：

- `mAP / Rank-1 / mINP`
- `GGT@x`: gallery 从 `1x, 2x, 4x, 8x, full` 的 drop 曲线
- `AUGC`: area under gallery-growth curve，横轴用 `log(|G|)`
- query-level bootstrap CI，按 query ID bootstrap，不只报均值
- 每阶段 `t=i+1...T` 都测，不只 final

关键控制：

- `REAL`: 加真实新域 gallery distractors
- `SHUFFLE`: 对 distractor score column 做 query-wise permutation，保持数量但破坏 query-specific hard distractor 结构
- `COUNT-MATCH`: 匹配 `#false-in-topk` 后重算 AP residual
- `SAME-DOMAIN TAX`: Market / Occluded-Duke / MSMT 单域内按同域 distractor 扩张，作为你们 kill-switch 的外推证据

最强 headline 表：

```text
Old domain | TotalDrop | FrozenTax | BaseForgetting | Interaction | TaxShare_shapley
Market     | ...
Duke       | ...
CUHK-SYSU  | ...
Average    | ...
```

注意边界：如果某些经典 LReID 论文只在 `Q_i vs G_i` 上报旧域 mAP，那不是 gallery-growth tax，而是 base-gallery forgetting。你们只能重归因“deployment/all-gallery LReID drop”或“把 LReID 放进真实累计 gallery 后观察到的旧域 drop”。

**② Remedy 选哪个**

不要把“gallery-growth-aware evaluation protocol”单独当方法。它是贡献一，但不是方法稿核心。

最稳的 method 是：

> Gallery-Growth Robust Learning / Tax-Aware Distractor Training

它不碰 backward compatibility，也不解决 re-index-free，所以不正面撞 2024 backward-compatible LReID 或 2025/2026 re-index-free 线。它解决的是“同一 embedding 在 gallery 规模变大时极端负样本进入 top-rank”的 rank-stability 问题。

草案：

训练时为每个 anchor 构造 growth pool：

```text
P(q): same-ID positives
N_g(q): same-domain / cross-domain distractor queue, sampled at growth factor g in {1,2,4,8,10}
```

用 extreme-negative AP surrogate：

```text
p_softmin = -1/a * logsumexp_{p in P(q)}(-a * s(q,p))
n_extreme =  1/b * logsumexp_{n in N_g(q)}( b * s(q,n))

L_tax = E_g softplus(n_extreme - p_softmin + m0 + lambda * log g)
```

再加一个 rank-stability loss：

```text
L_stable = KL( topK_rank_distribution(q, G_1) || topK_rank_distribution(q, G_g) )
```

总 loss：

```text
L = L_ce + L_triplet + alpha * L_tax + beta * L_stable
```

最小版本只上 `L_tax`，别一开始堆太多。

它该 beat 的 baseline：

- CE + Triplet / Bag-of-Tricks
- Batch-hard triplet with larger batch
- SupCon / Circle loss / ArcFace
- XBM / memory-bank hard negative
- Open-Set Biometrics 2024 的 relative threshold / max-negative 类 loss，作为强撞车 baseline；那篇明确说现有 loss 忽略 imposter score 的相对大小，并提出降低最大负样本分数。([arxiv.org](https://arxiv.org/abs/2407.16133))
- LReID 侧：Fine-tuning, LwF, AKA, PTKP, KRKC, DRE, AdalReID, BC-LReID；至少做 `strong LReID + TaxLoss` 插件式对照

廉价预检：

1. 用现有 frozen score matrix 计算 `p_softmin - n_extreme` 是否能预测 10x gallery 下失败 query。AUC > 0.75 才值得训。  
2. 只训 Market/Occluded-Duke 20 epoch：标准 `1x mAP` 不能掉超过 0.5，`10x GGT` 至少追回 25%。  
3. 在 LReID 四域 final all-gallery 上插入 TaxLoss，旧域平均 tax drop 至少降 20%，新域 MSMT 不明显掉。

**③ 撞车边界**

已知近年相关线：

- 2024 backward-compatible LReID：关注 old/new model feature compatibility 和免 backfilling；它确实提出 all-query/all-gallery evaluation，但没有做 frozen model 2x2 attribution，把掉点拆成 tax / forgetting。([ar5iv.org](https://ar5iv.org/html/2403.10022v2))  
- 2024 AdalReID：adapter + auto-selection，核心是 domain-specific knowledge preservation，不是 gallery growth tax。([arxiv.org](https://arxiv.org/abs/2405.19005))  
- 2024 DRE：transformer diverse representations + knowledge update/preservation，仍是抗遗忘机制。([arxiv.org](https://arxiv.org/abs/2403.16003))  
- 2026 PAD：vision-text semantic anchor，还是 semantic drift / forgetting / adaptability。([arxiv.org](https://arxiv.org/abs/2605.05027))  
- 2024 TEMP：固定 gallery 特征下做 test-time similarity modification，关注 temporal distribution shift，不是 gallery size attribution。([arxiv.org](https://arxiv.org/abs/2403.14114))  
- 2024 Open-Set Biometrics：最接近 remedy 的负样本尾部思想，但任务是 open-set detection/identification，不是 closed-set ReID mAP under gallery growth，也没有 LReID frozen-control re-attribution。([arxiv.org](https://arxiv.org/abs/2407.16133))

所以 novelty 边界写法：

> 不是“第一个注意到 gallery 会累积”；BC-LReID 已经注意到了。  
> 是“第一个把旧域性能下降分解为 representation forgetting 与 gallery-growth tax，并证明一部分所谓 deployment forgetting 在 frozen embedding 下已经发生”。  
> 方法上不是普通 hard negative mining，而是显式优化 gallery-size-conditioned extreme distractor risk。

**④ Verdict**

能撑 CCF-B 方法稿，但条件很硬：

- `TaxShare_shapley` 平均最好 >30%，至少两个旧域显著；
- shuffle/count controls 必须证明不是机械 count；
- TaxLoss 必须降低 growth curve 斜率，而不是只涨普通 mAP；
- remedy 不能靠 re-ranking / NFC / test-time trick；
- 必须和 Open-Set Biometrics 的 max-negative loss 做 baseline，否则 remedy 会被打成 hard-negative 旧酒。

信心：  
分析发现本身 8/10；纯分析投稿 4/10；加 Tax-Aware Distractor Training 且实验达标，CCF-B 方法稿 7/10。
