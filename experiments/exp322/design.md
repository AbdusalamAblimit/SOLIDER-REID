# 实验 exp322: DICA — Distractor-Injected Counterfactual Anchoring

> **来源**：post-PRCV 探索新创新点。4 方向红蓝队辩论（13 agents）winner。
> **状态**：design 草案，**未过双审查、未开训**。先 Tiny 廉价证伪。
> **诚实定性**：4 个方向蓝队全部 `survives=false`（confidence 8-9）。DICA 是 **winner-by-survival 而非 winner-by-safety**——300+ 实验已把邻近方向变成雷区，DICA 是"相对最值得做的那个"，不是"安全的那个"。本 design 把蓝队对 DICA 的最强攻击写在显眼处，交由双审查再裁。

---

## 动机

- **根因诊断（项目自证）**：Occluded-Duke 训练集 person-0 平均 **95.8% 可见**（中位 100%），backbone 从未见过真实多人歧义 → 所有 target-aware 尝试（exp070/076/077/078 heatmap 硬切、exp107/108 retrieval 侧 counterfactual penalty）都在"推理时区分已经混在一起的表征"，而 backbone 表征本身从未被训练成可分——这是它们封顶的根因。
- **与 oracle 路线的分界**：exp109 oracle 的 61.88→70.40 headroom 来自 GT same-ID feature **替换**（exp109/design.md:36-39），所有兑现尝试（SCKD/CSRD bank, exp110-131）封顶 ~61.1。DICA **不走 completion/replacement 路线**，headroom 来源不同——来自把训练分布对齐 test 的多人歧义。这也是它优于 D2/D4 的关键：那两个方案都试图"兑现" replacement-oracle headroom，但结构上够不着替换上限（蓝队对 D2/D4 的致命点且成立）；DICA 不依赖这个被误读的 oracle。
- **prompt-free gap**：KPR 靠 BIPO 合成 inter-person occlusion + **外部 prompt** 指认目标；PISNet 靠额外 clean query 图。prompt-free 下"如何获得 target-attribution 监督"是空白——DICA 用"我们贴上去所以 100% 知道真实 pid 的 distractor"填补。

## 核心假设

一句话：在 batch 内注入**已知身份**的真实人体作为 pose-aware overlap distractor，并用**反事实身份锚定**（图内注入身体作 pair-specific hard negative）直接监督 backbone，能让 backbone 学到对 inter-person identity contamination 的因果归因不变性——这是注入操作本身（KPR-BIPO / exp157d 已有）所**没有**的、由真实 pid 监督带来的净增量。

## 技术方案

- **修改文件**：
  - `datasets/pose_dataset.py`（~120 LOC）：distractor 注入 + person-1 heatmap + 真实 pid 记录 + exact dedup。复用 OA-SD 双视图 + multi-person heatmap 装配（`model/pose_backbone_model.py` line 929-936 已能取 target/distractor heatmap）。
  - `loss/triplet_loss.py` + `processor/processor.py`（~50 LOC）：图内 distractor pooled 表征作额外 hard negative 注入**现有** triplet（复用 `pose_part_pooling`）——**不新增 loss head**。
  - `config/defaults.py`（~15 LOC）：`POSE_DICA` 开关族（注入概率 / 反事实 negative 权重 / dedup 阈值）。
  - [可选，第二阶段] `model/modules/counterfactual_attribution_gate.py`（~60 LOC，复用 PSG residual gate）。**第一版不加 gate**。
- **数据流**：anchor crop x（目标 pid=y）→ 从同 batch 不同 ID y'≠y 抠 pose-masked 人体块 → pose-aware 贴到目标人体边缘（制造 **overlap** 非 covering）得污染图 x̃ → 生成 multi-person heatmap（person-0=y, person-1=y'）+ 记录 distractor 真实 pid y' → backbone（可选 attribution gate）→ (a) **不变性锚**：f(x̃) global+part 经 CE/triplet 监督到 y；(b) **反事实对比锚**：triplet(anchor=f(x̃)目标, positive=同 y 另一图, negative=x̃ 内部 distractor pooled 表征)；(c) **exact dedup**（已知 pid + bbox IoU 跳过重叠/同人样本）。
- **关键超参**：注入概率（默认 **0.5**，保留干净样本主导收敛，缓解铁律3）；反事实 negative 权重（**zero-init + OA-SD 同款 cosine-decay/early-anneal**，从 D3 嫁接）；dedup IoU 阈值。
- **铁律对照**：作用点 = Swin stage-3 token（与 PSG 同层，**铁律1** backbone 修改）；全 non-detached 回流 backbone，无 bank/无 completion/无 retrieval 侧（避**铁律2**）；唯一额外梯度是融进**已有 triplet** 的一个 negative，与主 CE/triplet 同质、方差同量级，非新增 loss head（对抗**铁律3**，但需 Tiny 收敛曲线实证）。

## 预期结果

- 假设成立：Occluded-Duke mAP/R1 在 **multi-person / low-vis 子集**显著正，整体中等正；跨域 Market→Occluded-ReID 增益更明显（真实多人遮挡多）。
- 失败最可能原因：(1) **铁律3 复发**——合成图分布偏移 + 反事实 negative 后期干扰 CE+triplet（exp320 -6.4 前车）；(2) **ablation #2** 显示"注入无身份监督"≈full → 创新坍缩成 exp076/157d 换皮；(3) 合成 artifact shortcut（backbone 学贴痕而非归因，ROA 已知问题）。

## 对照组

- **baseline**：同 config 的 Tiny/Small PSG+PLBOA（无 DICA）。
- **消融变量（每次只改一个）**：
  1. 去 distractor injection（退回普通 PLBOA 物体遮挡）= 注入贡献 vs 无身份遮挡物（区别于 ROA/exp157d）。
  2. 注入但**去反事实 ID 监督**（label 只用 y，无图内 hard negative）= 退化成 exp076/157d。**切割实验（kill-switch）**：full 与它的差值 = 真实 pid 监督净贡献，**若 ≈0 立即止损**。
  3. 反事实 negative 用 batch 随机 ID vs 图内注入身体 = pair-specific 必要性。
  4. attribution gate detach vs non-detach（对照 exp320）= backbone 梯度必要性。
  5. 去 exact dedup = 去重必要性。
  6. 注入概率 0/25/50/75% 敏感性。
  7. 诊断：按 query target_vis≤5/≤8、按真实人数子集拆 mAP（复用 exp109 分桶），证增益集中在 multi/low-vis；attribution 责任图可视化（贴入身体被压制）。

---

## 红蓝队辩论记录（2026-06-15，13 agents）

| 排名 | 机制 | 方向 | 判分 | 一句话 |
|------|------|------|------|--------|
| 1 ⭐ | **DICA** | target-ambiguity | 5.5 | 问题级 novelty 最强，不依赖被误读的 replacement-oracle；但注入 ≡ KPR-BIPO + exp157d(-2.2 R1)，图内 hard-neg 是 non-detached 辅助梯度（踩铁律3 风险） |
| 2 | PAIR-CRT | common-visible-support | 4.5 | 唯一干净绕开铁律2/3（训练端零梯度）、首验最便宜；但 z-score gain ≡ exp141 LPCS(-5.3)、common-support ≡ cvk_hybrid，落禁止回退②⑥，天花板仅 +0.5~1.5 |
| 3 | RaCE | reliability-uncertainty | 3 | warp ≡ PSG x*(1+gate) 子集；FEAT_NORM=yes 把"度量曲率"抵消成 vis-weighted pooling；r* 在 95.8% 可见下退化为常数门 |
| 3 | CALM | cross-domain-transfer | 3 | 弃权评分 ≡ AQGP + scorer 家族；r_k ≡ exp140 confidence gate(→0.99)；弃权≠替换，够不着 oracle |

**为何选 DICA（非更安全的 D2）**：
1. 真 novelty（问题级+机制级）最高，且**唯一不依赖被误读的 replacement-oracle**（D2/D4 的致命点对 DICA 无效）。
2. 问题重定义经得起追问：把遮挡重定义为"对已知身份 distractor 的因果归因不变性"，prompt-free target-attribution 监督是 KPR/PISNet 都没写的空白。
3. 抗辩存活的核心是 **identity-anchoring hard-negative，不是注入本身**。

**蓝队对 DICA 的最强攻击（必须正视）**：
- **注入 ≡ KPR-BIPO**（kpr_comparison.md:33 已核实：KPR 用 BIPO 合成 inter-person occlusion 注入）+ **exp157d**（人体 bbox 贴块 = 61.0/71.5，-2.2 R1，results.md:190 已核实）。→ **击穿了"注入"这一半**。
  - 判定：未击穿真正创新载体——KPR-BIPO 无身份监督（靠 prompt），exp157d 贴块无 pid 标签。**ablation #2 正是为切干净这个差值设计的**。novelty 全押在"identity-supervised counterfactual anchoring 是 BIPO 之上的真增量"——**这条由双审查（Codex --search 查先例）+ ablation #2（实证）双重把关**。
- **图内 hard-negative = non-detached 辅助梯度**，与 exp036 per-keypoint triplet(-0.5)/CSGT(崩)同类，踩铁律3。→ 头号风险，**Tiny 收敛曲线第一关**。

## 头号风险（open risk，不可纯论证免疫）

**铁律3 复发**：反事实 negative 虽与主 triplet 同质，但合成图 x̃ 引入分布偏移，后期可能干扰 CE+triplet 收敛（exp320 LGPA detach=False -6.4 前车）。**必须先在 Tiny 上跑收敛曲线，确认 part/global loss 后期不被拖崩，再上 Small——绝不直接上 Small 才发现灾难。**

## 嫁接自落选方案的纪律（judge）

- **D2 → "命名自己的 falsifier"**：第一个实验不赌 full DICA，先验最小载体（无 gate）；ablation #2 是 kill-switch。
- **D2/D4 → 子集诊断**：按 query 真实人数 / 可见度子集拆 mAP，作 motivation figure + 机制验证（复用 exp109 分桶）。
- **D4 → 源域合成跨域视角**：distractor 注入在源域即可制造多人歧义监督，Market→Occ-ReID 跨域无需 target 遮挡标注 → 第二张表，绕开 Occ-Duke multi-query 比例可能偏低的天花板。
- **D3 → zero-init + early-anneal**：反事实 negative 权重用 OA-SD 同款 cosine-decay（exp191 OA-SD+CE 有项目内成功先例）缓解铁律3。
- **D2 → 开训前零成本 sanity check**：用现有 exp255/exp261 checkpoint + pose 标注先统计 Occ-Duke / Occ-ReID 测试集真实 multi-person query 比例，评估 DICA 整体增益上限——**比例太低则机制上限受限，应在写代码前知道**。

## 推荐首个实验（falsification-first）

- **backbone**：Swin-Tiny 快验（20-30 epoch 看收敛曲线 + 趋势，不追绝对点数）。
- **config**：最接近 exp255 的 Tiny 变体（Tiny + PSG + PLBOA/OA-SD）作 baseline 起点。
- **改 key（单变量隔离，先做最小载体不做 full DICA，第一版不加 attribution gate）**：见技术方案 1-3。
- **测什么**：① 收敛安全性（各 loss 分量曲线，对照 exp320 灾难，Tiny 上若出现立即止损）；② 趋势 mAP/R1（baseline vs +DICA-minimal）；③ ablation #2 切割（注入无身份监督，差值=身份监督净贡献，≈0 止损）；④ 子集诊断（target_vis≤8 / multi 子集）。
- **对照**：同 config Tiny PSG+PLBOA（baseline）+ ablation #2（注入无身份监督）。三组共享一切超参，只切两个开关。评测 `test.py`，`TEST.IMS_PER_BATCH 64`。
- **门槛**：先 Tiny 验证收敛曲线 + ablation #2 切割有效，再上 Swin-Small。

---

## 待办（按协议，开训前必须）

1. [ ] 红蓝队辩论已完成（本节）✅
2. [ ] 实现 DICA-minimal 代码（datasets/loss/config）
3. [ ] **Claude Broad Review**（Opus 子代理，全范围）→ `claude_review.md`
4. [ ] **Codex Review**（`codex --search exec`，查 KPR-BIPO 先例 / novelty）→ `codex_review.md`
5. [ ] 两审查 approve 后，才在空闲 GPU 启动 Tiny 首验（hook 强制阻断）
