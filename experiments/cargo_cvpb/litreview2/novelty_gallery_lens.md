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
session id: 019efa77-b45e-7cd2-be3c-53a553b48eb0
--------
user
一个 ReID 团队要做一个统一视角的 CCF-B 方法/分析稿: **'gallery 组成(规模 size / 随时间膨胀 growth / distractor 分布 / prototype support count)是 ReID 真实失败的被忽略驱动, 不只看 query/模型'**。三个 instantiation(3 个独立 codex 收敛): ①gallery-growth-tax(LReID 旧域掉点部分是 gallery 膨胀的误归因而非 catastrophic forgetting, frozen 模型也掉); ②gallery-size-conditioned 拒识(open-set 阈值该随 watchlist size 变, 大 gallery impostor max-score 上升); ③singleton-tail-merge(尾部身份被错并入 head prototype)。

联网核查 novelty(2024-26): 这个统一视角对 CCF-B 够不够新? 还是已被覆盖——k-reciprocal/re-ranking 隐式处理 gallery topology? open-set ReID 处理 gallery size? LReID/backward-compatible 处理 gallery 累积? gallery distractor/search-space 分析已有? face/speaker watchlist-size 分析(VoxWatch)能否直接迁? 逐个 instantiation 查确切先例。
**关键**: 统一'gallery-composition lens'是真 headline 还是三个松散小点拼凑? 哪个 instantiation 最新最值得主推? 哪个最容易被说 trivial/已知? 给 verdict(统一可发/太散/被覆盖)+ 每个信心 1-10 + 最该主推哪个 + 怎么切开撞车。务实中文。
codex
我会先按五条可能撞车线索查近三年的论文和代码线：ReID gallery/search-space、open-set/watchlist-size、LReID/backward-compatible 累积、re-ranking/topology、face/speaker 里的 watchlist-size 分析。重点不是找泛泛相似词，而是找是否已经把“gallery composition”作为主因变量来建模。
web search: 
web search: 2024 2025 person re-identification gallery size distractor analysis open-set ReID watchlist size
web search: 
web search: person re-identification distractor gallery size analysis distractors Market-1501 2024
web search: 
web search: "gallery size" "person re-identification"
web search: 
web search: 2024 lifelong person re-identification survey gallery samples continual learning catastrophic forgetting
web search: 
web search: "Re-index Free Lifelong person Re-IDentification" "gallery"
web search: 
web search: 2024 open-set person re-identification survey open set ReID threshold
web search: 
web search: watchlist size verification threshold impostor max score face recognition 2024
web search: 
web search: "watchlist size" "false alarm" speaker identification VoxWatch 2024 calibration
web search: 
web search: person re-identification prototype support count singleton tail merge head prototype clustering reid 2024
web search: 
web search: 2024 unsupervised domain adaptive person ReID prototype outlier singleton cluster
web search: 
web search: "singleton" "tail" "prototype" "face recognition" "clustering"
web search: 
web search: "long-tailed" "person re-identification" 2024
web search: 
web search: "support count" "prototype" "re-identification"
web search: 
web search: "support count" "watchlist" face recognition prototype
web search: 
web search: person re-identification k-reciprocal re-ranking gallery topology 2024
web search: 
web search: 2024 person re-identification re-ranking graph gallery reciprocal nearest neighbors
web search: 
web search: "gallery growth" "person re-identification"
web search: 
web search: "incrementally accumulated" "gallery" "ReID"
web search: 
web search: https://arxiv.org/abs/2403.10022
web search: 
web search: https://arxiv.org/abs/2512.25000
web search: 
web search: https://arxiv.org/abs/2403.15119
web search: 
web search: 'open-set' in https://arxiv.org/abs/2409.03879
web search: 
web search: https://arxiv.org/abs/2307.00169
web search: 
web search: "distractor" "gallery" "person re-identification" "Market-1501"
web search: 
web search: "Market-1501" "distractor" "gallery"
web search: 
web search: Market-1501 dataset distractors junk images gallery 2793 false alarms
web search: 
web search: "gallery size" "person search" re-identification
web search: 
web search: "person search" "gallery size" "distractors"
web search: 
web search: https://arxiv.org/abs/2006.00155
web search: 
web search: https://arxiv.org/abs/1812.03282
web search: 
web search: https://arxiv.org/abs/1408.0872
web search: 
web search: "maximum score" "threshold" "watchlist size" biometrics
web search: 
web search: open set identification watchlist size false alarm threshold maximum similarity score biometrics
web search: 
web search: https://arxiv.org/abs/2407.16133
web search: 
web search: "Open-Set Biometrics: Beyond Good Closed-Set Models" "gallery" "person re-identification"
web search: 
web search: https://github.com/prevso1088/open-set-biometrics
web search: 
web search: https://arxiv.org/abs/1906.06272
web search: 
web search: https://arxiv.org/abs/1701.08398
web search: 
web search: 2024 ReID clustering prototype outliers singleton long-tail identity imbalance "person re-identification"
codex
**Verdict: 可发，但要收紧。**  
“gallery-composition lens”在 **person ReID 主流论文里还没有被系统化成一个统一诊断框架**，有 CCF-B 分析/方法稿空间；但它不能写成“首次发现 gallery 会影响 ReID”。这个说法会被已有 re-ranking、open-set、LReID/backward-compatible、biometric watchlist-size 工作打穿。更稳的 headline 是：

> ReID failure is not only query/model failure; it is conditional risk under gallery composition. We provide counterfactual gallery interventions to separate model forgetting, gallery growth, distractor distribution, and prototype support bias.

总体判断：**统一可发 6.5/10**。  
如果三点只是并列实验，**太散**；如果用同一套 counterfactual gallery intervention protocol 串起来，**可以成立**。

**逐项结论**

| Instantiation | Novelty | 撞车风险 | 判断 |
|---|---:|---:|---|
| ① gallery-growth-tax | 7/10 | 中高 | 最值得主推 |
| ② gallery-size-conditioned 拒识 | 3/10 | 很高 | 只能做 supporting，不宜当贡献 |
| ③ singleton-tail-merge | 6.5/10 | 中 | 可做第二贡献，但要证明不是 clustering/long-tail 常识 |

**① Gallery-Growth-Tax**

最接近先例是 2024 的 [Lifelong Person Re-ID with Backward-Compatibility](https://arxiv.org/abs/2403.10022)：它已经明确说 **training datasets 和 gallery images 都会 incrementally accumulated**，并提出 all query/gallery together 的更实际评测。2025/26 的 [Bi-C2R / RFL-ReID](https://arxiv.org/abs/2512.25000) 也继续沿着 historical gallery / re-index-free 方向走。

但我没查到它们做了你说的关键控制：**冻结模型后，只因 gallery 从 old-only 变成 old+new，就导致 old-domain 指标掉，并把这部分从 catastrophic forgetting 里分离出来**。这就是可争的新意。

切法：不要说“我们首次考虑 gallery accumulation”。要说：

> Existing LReID treats old-domain degradation mainly as model forgetting or compatibility failure. We show a measurable gallery-growth tax under a frozen encoder, so reported forgetting is partially a retrieval-set composition confound.

最好给分解式：

`reported old drop = encoder forgetting + gallery growth tax + compatibility/re-indexing effect + interaction`

这条最像 CCF-B 主线。

**② Gallery-Size-Conditioned 拒识**

这个点概念上基本被覆盖了。  
ReID 里早有 [Open-set Person Re-identification](https://arxiv.org/abs/1408.0872)、[Adversarial Open-World Person Re-ID](https://arxiv.org/abs/1807.10482)、2024 工业 open-set 的 [MICRO-TRACK](https://arxiv.org/abs/2409.03879)。更致命的是 biometrics/speaker 里已有非常直接的先例：[VoxWatch](https://arxiv.org/abs/2307.00169) 明确写了 watchlist 变大时 out-of-set scores 变大、false alarm 上升，并用 `max_i score_i > threshold` 做检测；2024 ECCV [Open-Set Biometrics](https://arxiv.org/abs/2407.16133) 还跨 face/gait/person ReID 优化 maximum negative score / selective thresholds。

所以这条不能当新方法 headline。它可以作为统一 lens 的一个 clean demonstration：

> ReID open-set threshold must be calibrated against gallery size and distractor mix, not transferred as a fixed scalar.

但审稿人很容易说：order statistics / max impostor score，known。

**③ Singleton-Tail-Merge**

这条直接先例相对少。相关邻近工作主要是 long-tail / prototype / clustering：如 long-tailed ReID/face 的 feature augmentation、person search long-tail、ProtoNorm、UDA ReID clustering/prototype memory 等。但我没有看到一个明确把 **gallery prototype support count** 作为检索侧失败驱动，专门分析“singleton tail 被 head prototype 吸收”的 ReID 工作。

风险是它会被说成：这不就是 class imbalance、prototype bias、clustering merge error？  
要切开，必须避免泛泛讲 long-tail training，而是固定成 gallery-side 机制：

> same embedding, different prototype support-count distribution, different merge risk.

最好做 controlled merge audit：固定模型和 query，改变 prototype support count / head-tail ratio，画 singleton-to-head merge probability 曲线。这样它才不是“又一个 long-tail loss”。

**Re-ranking / k-reciprocal 会不会覆盖统一视角？**

不会完全覆盖，但会挡住你的一部分表述。  
[k-reciprocal re-ranking](https://arxiv.org/abs/1701.08398) 本质就是利用 gallery/query 的 neighbor topology，CA-Jaccard、K-nearest weighted fusion 这类后续工作也在做 gallery-neighbor 关系。所以不能说“以前没人看 gallery topology”。

切法是：re-ranking 是 **post-hoc ranking refinement**；你们是 **counterfactual gallery composition diagnosis**。要把 k-reciprocal、re-ranking 放进 baseline：证明 gallery-growth-tax / threshold drift / support-count bias 在 no-rerank 和 rerank 下都存在，或解释 rerank 缓解了哪一部分。

**最该主推**

主推 **① gallery-growth-tax**。它最适合变成论文主问题：当前 LReID/backward-compatible 论文正在 2024-26 变热，但它们关注 model update、compatibility、backfilling；你们切的是“old-domain drop 的归因错了”。这更像分析稿的主贡献。

组合建议：

1. 主贡献：Gallery Growth Tax for LReID evaluation.
2. 第二贡献：Gallery Composition Intervention protocol，统一 size / growth / distractor / support count。
3. 方法小件：support-count-aware merge guard 或 size-conditioned calibration。
4. ② 拒识阈值降级为 sanity / cross-modal bridge，不要主推。

**一句话定稿方向**

可发版本不是“三个小点拼盘”，而是：

> 我们提出 ReID 的 gallery-composition intervention framework，并证明若不控制 gallery 组成，LReID forgetting、open-set rejection、prototype gallery maintenance 都会被系统性误判；其中 gallery-growth-tax 是核心新发现。
tokens used
126,333
**Verdict: 可发，但要收紧。**  
“gallery-composition lens”在 **person ReID 主流论文里还没有被系统化成一个统一诊断框架**，有 CCF-B 分析/方法稿空间；但它不能写成“首次发现 gallery 会影响 ReID”。这个说法会被已有 re-ranking、open-set、LReID/backward-compatible、biometric watchlist-size 工作打穿。更稳的 headline 是：

> ReID failure is not only query/model failure; it is conditional risk under gallery composition. We provide counterfactual gallery interventions to separate model forgetting, gallery growth, distractor distribution, and prototype support bias.

总体判断：**统一可发 6.5/10**。  
如果三点只是并列实验，**太散**；如果用同一套 counterfactual gallery intervention protocol 串起来，**可以成立**。

**逐项结论**

| Instantiation | Novelty | 撞车风险 | 判断 |
|---|---:|---:|---|
| ① gallery-growth-tax | 7/10 | 中高 | 最值得主推 |
| ② gallery-size-conditioned 拒识 | 3/10 | 很高 | 只能做 supporting，不宜当贡献 |
| ③ singleton-tail-merge | 6.5/10 | 中 | 可做第二贡献，但要证明不是 clustering/long-tail 常识 |

**① Gallery-Growth-Tax**

最接近先例是 2024 的 [Lifelong Person Re-ID with Backward-Compatibility](https://arxiv.org/abs/2403.10022)：它已经明确说 **training datasets 和 gallery images 都会 incrementally accumulated**，并提出 all query/gallery together 的更实际评测。2025/26 的 [Bi-C2R / RFL-ReID](https://arxiv.org/abs/2512.25000) 也继续沿着 historical gallery / re-index-free 方向走。

但我没查到它们做了你说的关键控制：**冻结模型后，只因 gallery 从 old-only 变成 old+new，就导致 old-domain 指标掉，并把这部分从 catastrophic forgetting 里分离出来**。这就是可争的新意。

切法：不要说“我们首次考虑 gallery accumulation”。要说：

> Existing LReID treats old-domain degradation mainly as model forgetting or compatibility failure. We show a measurable gallery-growth tax under a frozen encoder, so reported forgetting is partially a retrieval-set composition confound.

最好给分解式：

`reported old drop = encoder forgetting + gallery growth tax + compatibility/re-indexing effect + interaction`

这条最像 CCF-B 主线。

**② Gallery-Size-Conditioned 拒识**

这个点概念上基本被覆盖了。  
ReID 里早有 [Open-set Person Re-identification](https://arxiv.org/abs/1408.0872)、[Adversarial Open-World Person Re-ID](https://arxiv.org/abs/1807.10482)、2024 工业 open-set 的 [MICRO-TRACK](https://arxiv.org/abs/2409.03879)。更致命的是 biometrics/speaker 里已有非常直接的先例：[VoxWatch](https://arxiv.org/abs/2307.00169) 明确写了 watchlist 变大时 out-of-set scores 变大、false alarm 上升，并用 `max_i score_i > threshold` 做检测；2024 ECCV [Open-Set Biometrics](https://arxiv.org/abs/2407.16133) 还跨 face/gait/person ReID 优化 maximum negative score / selective thresholds。

所以这条不能当新方法 headline。它可以作为统一 lens 的一个 clean demonstration：

> ReID open-set threshold must be calibrated against gallery size and distractor mix, not transferred as a fixed scalar.

但审稿人很容易说：order statistics / max impostor score，known。

**③ Singleton-Tail-Merge**

这条直接先例相对少。相关邻近工作主要是 long-tail / prototype / clustering：如 long-tailed ReID/face 的 feature augmentation、person search long-tail、ProtoNorm、UDA ReID clustering/prototype memory 等。但我没有看到一个明确把 **gallery prototype support count** 作为检索侧失败驱动，专门分析“singleton tail 被 head prototype 吸收”的 ReID 工作。

风险是它会被说成：这不就是 class imbalance、prototype bias、clustering merge error？  
要切开，必须避免泛泛讲 long-tail training，而是固定成 gallery-side 机制：

> same embedding, different prototype support-count distribution, different merge risk.

最好做 controlled merge audit：固定模型和 query，改变 prototype support count / head-tail ratio，画 singleton-to-head merge probability 曲线。这样它才不是“又一个 long-tail loss”。

**Re-ranking / k-reciprocal 会不会覆盖统一视角？**

不会完全覆盖，但会挡住你的一部分表述。  
[k-reciprocal re-ranking](https://arxiv.org/abs/1701.08398) 本质就是利用 gallery/query 的 neighbor topology，CA-Jaccard、K-nearest weighted fusion 这类后续工作也在做 gallery-neighbor 关系。所以不能说“以前没人看 gallery topology”。

切法是：re-ranking 是 **post-hoc ranking refinement**；你们是 **counterfactual gallery composition diagnosis**。要把 k-reciprocal、re-ranking 放进 baseline：证明 gallery-growth-tax / threshold drift / support-count bias 在 no-rerank 和 rerank 下都存在，或解释 rerank 缓解了哪一部分。

**最该主推**

主推 **① gallery-growth-tax**。它最适合变成论文主问题：当前 LReID/backward-compatible 论文正在 2024-26 变热，但它们关注 model update、compatibility、backfilling；你们切的是“old-domain drop 的归因错了”。这更像分析稿的主贡献。

组合建议：

1. 主贡献：Gallery Growth Tax for LReID evaluation.
2. 第二贡献：Gallery Composition Intervention protocol，统一 size / growth / distractor / support count。
3. 方法小件：support-count-aware merge guard 或 size-conditioned calibration。
4. ② 拒识阈值降级为 sanity / cross-modal bridge，不要主推。

**一句话定稿方向**

可发版本不是“三个小点拼盘”，而是：

> 我们提出 ReID 的 gallery-composition intervention framework，并证明若不控制 gallery 组成，LReID forgetting、open-set rejection、prototype gallery maintenance 都会被系统性误判；其中 gallery-growth-tax 是核心新发现。
