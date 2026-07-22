# exp413 PSCCR 近期论文与代码审计（2026-07-22）

## 审计问题与检索边界

exp411证明等支持的all-identity set ranking相对clean D0能涨点，但pose×CLIP owner与正确RGB/PID绑定均未成立；
exp412又证明把梯度预算转给二维可靠视图会损害mAP。exp413因此只审计一个更窄问题：是否已有标准单图ReID方法，
用pose visibility与identity-free CLIP visible-vs-occluded margin共同定义**不丢弃support的互补覆盖顺序**，再对长度
1/2/3的嵌套support prefix逐级执行student空间的all-identity set ranking。

本轮尝试了聚焦查询`person re-identification complementary support prefix listwise ranking pose CLIP 2024 2025
2026`。Google、arXiv页面及Semantic Scholar API均在当前网络环境超时，因此不能把本轮写成覆盖完整的在线系统综述，
也不能用“在线未命中”推出“没有先例”。裁决基于仓库此前已落盘并核对到2026年的pose/CLIP/ReID论文与代码审计、
exp405--412实现谱系，以及本地语料对`prefix/submodular/coverage/listwise/multi-positive`的交叉检索。若后续网络恢复，
投稿前仍须补一次正式Scholar/DBLP/OpenAlex检索。

## 近邻对象与不能声称的新意

| 近邻对象 | 已覆盖内容 | 与PSCCR的边界 |
|---|---|---|
| SupCon与监督度量学习 | 同标签多正样本共同进入对比目标 | positive通常无序同权；不以pose×身份无关CLIP构造嵌套support prefix |
| Lifted Structured、Ranked List、Smooth-AP、ROADMAP | all-pair、listwise或AP surrogate | relevance来自标签或student距离；不能把listwise/AP导向本身写成PSCCR创新 |
| episodic/prototypical set learning | query相对prototype或support set学习 | support通常作为无序集合；PSCCR不产生外部prototype，只改变同一三图集合的前缀顺序 |
| curriculum/self-paced learning | 按难度跨step/epoch安排样本或loss | 不是在同一步内对一个无丢弃support排列的1/2/3前缀同时优化 |
| greedy/submodular coverage | 以边际覆盖增益生成集合顺序 | 贪心覆盖与单调coverage都是已知原子，绝不能声称PSCCR首次提出 |
| pose-aware sampling、part ReID | pose选择样本、部位或可见区域 | 未在已审计标准ReID实现中发现身份无关CLIP遮挡轴组织完整support prefix的同构路径 |
| CLIP-ReID及VLM-ReID | identity prompt、image-text alignment、CLIP backbone | CLIP通常提供身份或跨模态监督，不只提供train-only的support排列 |
| PAFormer、KPR、ProFD、MUVA | pose token、keypoint prompt、part text/memory、CLIP mask attention | 主要改变局部表征/attention/part retrieval，不是global student空间的逐前缀身份集合排序 |
| 2026 pose-guided enriched feature / composite-attribute ReID近邻 | pose hard positive、属性/身份分解或batch关系 | 封住宽泛的pose-guided sampling/关系主张，但未在已审计材料中出现PSCCR完整组合 |

因此PSCCR不得声称首次listwise、首次multi-positive、首次set learning、首次support curriculum、首次submodular
coverage或首次pose+CLIP。当前本地已审计语料只支持“未发现完整同构”，不支持绝对新颖性结论。

## 与仓库既有机制的不可混淆边界

1. exp405--407 CAVT是同PID donor/token transport，并连续被matcher/measurement validity阻断；PSCCR不搬运token、
   不预测状态转移，也不修donor matcher。
2. exp356 PC-MSC与exp361 PSC-JEPA已覆盖pose mask下的CLIP feature重建和latent support completion；PSCCR不蒸馏
   CLIP坐标或teacher latent。
3. exp408 PICRD做逐槽CLIP relation，exp409 PCHM选一个hard pair；PSCCR只在exp411已证有效的完整身份集合排序上
   增加无丢弃prefix训练对象。
4. exp411 owner以multiplicity改变三图集合权重且归因失败；PSCCR的三图最终只各出现一次，pose/CLIP只决定进入
   prefix的次序，长度3严格回到sealed zero-owner距离与loss。
5. exp412用同一身份无关文本轴重分配backward梯度；PSCCR不改forward或token梯度倍率，遮挡/难图仍作为query保留
   完整梯度。

## 开源接点与可证伪性

最小实现应复用`loss/pose_clip_multi_positive_set.py`的strict cache、pose visibility、PID行序和zero-owner
all-identity公式，另建coverage-chain state/loss；processor只负责在`torch.no_grad()`中生成离散链，model与eval不改。
长度3显式调用原support顺序计算，从而保证distance/loss与sealed zero-owner exact，而不是依赖浮点加法换序后的近似。

一次真实PK64合同必须同时证明：先LOO再在三support内排名且被排除图mutation不影响链、invalid/tie/rank方向通过手算
micro-oracle、三support是严格排列、coverage单调、correct与pose-only/q-only/text-shuffle的真实链改变率非零、
prefix3 exact、isolated Stage-3/backbone梯度active且相对zero-owner改变、native GradScaler真实更新。性能GO后再跑
三条matched control，可区分pose、CLIP和正确文本绑定；任何一条不被correct在mAP/R1同时严格超过，联合归因失败。

## 创新裁决

- 问题门：PASS。它把“完整三图集合一次平均”细化为“遮挡query在部分可用support下也应保持全身份排序”，并保留
  最终完整集合，不再把难图梯度转走。
- 机制门：CONDITIONAL PASS。greedy coverage、ordinal rank、prefix curriculum与listwise loss均非新原子；只有
  `pose×identity-free CLIP可靠度 + 无丢弃互补覆盖链 + 逐前缀all-identity ranking + prefix3宿主exact`整体可作为
  C类窄差分。
- 证据门：PASS。sealed zero-owner、pose-only、q-only、text-shuffle共享支持、损失形式、计算规模和正式recipe。

裁决为`C-CLASS CONDITIONAL / DESIGN ALLOWED`，不是B类创新确认。correct e120若mAP或R1任一不严格胜sealed
zero-owner `58.9/70.3`，立即性能封板；性能GO后若不同时严格胜三control，则只保留整体性能事实，不宣称pose+CLIP。
