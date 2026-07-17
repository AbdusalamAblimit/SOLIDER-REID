# LM-ReID 训练端创新 pipeline（2026-06-26，用户指令"训练端不许停"后）

## ★★★LSRC 死 → 训练端三大类全死（2026-06-26 早）

LSRC（backbone set-loss）full fine-tune eval（4090 lam0.5）vs no-LM-loss（single 77.44 / lattice 79.90 / HR-sanity 88.92）：

| 指标 | no-LM-loss | LSRC lam0.5 | Δ |
|---|---|---|---|
| HR sanity | 88.92 | **85.84** | **−3.08** |
| h16 single | 77.44 | 75.70 | −1.74 |
| h16 lattice(MaxSim) | 79.90 | **77.98** | **−1.92** |
| h24 single | — | 82.31 | |
| h24 lattice(MaxSim) | — | 83.27 | |

**LSRC 全面拉低**：训练 acc 1.000（过拟合训练集）但测试全掉 = backbone set-loss 损害判别力。marginalization 在受损 backbone 上仍 +2.288（机制工作但起点被拉低，证 lattice marg 本身没问题，是 backbone 被训坏）。3090 lam1.0 必死（更大权重）。

**★非对称 LSRC 已实测（2026-06-26 用户二次质疑"LM-ReID 没训练端创新吗"后, --lsrc_asym query-set × gallery-single, lam0.5, 40ep）= 死 −0.33**（h16 lattice MaxSim 79.57 vs no-LM 79.90, 没过 +0.3 线; HR sanity 87.63 −1.29; single 77.29 −0.15; h24 lattice 84.73 d+0.893）。**但比对称温和太多**（−0.33 vs 对称 −1.92）: 我之前"对称死→非对称必死"**结论对但严重低估温和度**——外推只给死/活, 实测才知非对称几乎不伤 single(−0.15)、lattice 仅 −0.33（差点到 0 但仍 net negative）。机理证实 backbone set-loss 损害判别力对称/非对称都有, 非对称(不给 gallery oracle)温和但回不到正。**→ 非对称 LSRC 从"推理判死"升级"实测死"，至此 LM-ReID 训练端 100% 实测穷尽零外推**（frozen LS-MRT/LATS + backbone-loss consistency/LSRC对称/LSRC非对称 + robust-ERM Hard-Lattice + input BLC证伪）。

**→ 训练端三大类全死**：
1. **frozen-feature 重投影/重加权**（LS-MRT +0.028 / LPA +0.075）— no-LM-loss 特征对边缘化已最优、无 headroom。
2. **backbone 改 loss**（LSRC −1.9 / consistency −1.73 / L_marg 有害）— 改 backbone 损害判别力。
3. **robust ERM**（Hard-Lattice 76.9<77.44）— worst-case 也救不了。

**强结论**：no-LM-loss backbone 已是 LR-ReID 好特征，**test-time decision marginalization 是唯一有效杠杆**；训练端干预要么无用（frozen）要么有害（改 backbone）。备选 BLC（input canonicalize bbox crop, design 已写）market 只裁好框受限 6.5、未验。启 2 codex（train3_{fourthclass 找第四类机制, paperstrategy 评估反例论文策略}）。

## ★★★★第四类也死/证伪（2026-06-26，codex round3）

**fourthclass codex**：第四类候选 LATS（frozen-token sidecar 6.5）/ BLC（input）/ LVAS（data sampler 5）/ residual-diversity（4.5）。**paperstrategy codex**：明确"别再碰 backbone training loss"，论文 test-time + 反例 **5.5→6.5（补跨数据集/强TTA/MLR）→7.5（仅当 BLC 过线）**；核心论点 *"Learning to be invariant is the wrong objective; marginalizing decisions over plausible observations is the right one"*。

- **BLC 被现有 LM-S4 数据证伪**（不用跑）：BLC = canonicalize bbox（固定 canonical）+ marginalize 残差。但 bbox 是主因子（marg bbox +2.84），canonicalize 它 = 放弃这大收益，只剩 phase/zoom marg ≈ **+1.7 < marginalize-all +2.557**；且 market 测试用标注框（已 canonical），BLC 让 K=9 marg 退化成 single 77.44 < 79.90。**canonicalize 主因子 < marginalize 它 → BLC 死。**
- **LATS**：frozen-adaptation 子类（LS-MRT final-feat 已证 +0.028 无 headroom），token map cache ~18GB **不 cheap**，codex 期望 +0.0~0.2。**★实测（2026-06-26 用户质疑"LM-ReID 没训练端创新吗"后, cvpb_lats_probe.py cheaper stripe-pooled 版, frozen backbone + 6-stripe token sidecar）= 死 −5.147**（uniform-global marg 80.23 → stripe-LATS 75.09；K-cos rise +0.023 变体趋同）。机理: stripe sidecar set-retrieval 训练把 K 变体拉相似破坏 marginalization 多样性, **和 LSRC 一个死法**（训练端塑造/对齐变体即损害 test-time marginalization 多样性）。**→ LATS 从"外推判死"升级"实测死"**：frozen-adaptation 类两个实测点（LS-MRT final-feat +0.028 / LATS stripe-token −5.147），连 pool-前 token 空间信息都试过。用户质疑价值=穷尽从外推→实测, "Why Training-Time Invariance Fails"论点更硬。
- **paperstrategy 论文可投性**：test-time + 反例上限 **6.5**（BLC 证伪 → 够不到 7.5）。论文转 "Why Training-Time Invariance Fails"（三类反例写成"自然但错误的解法"）+ 补实验。

**train4_final codex 判决（8.5/10）：当前设定下没有值得追的 cheap 训练端机制，别硬凑。** 4 类全封：① frozen/head/sidecar 无 headroom（含 LATS）；② backbone-loss 伤判别性（consistency/LSRC）；③ robust-ERM 没赢（Hard-Lattice）；④ BLC 逻辑封住（bbox 收益来自 marginalize 非 canonicalize，market 框已 canonical）。codex 强论点：*"sampling-lattice uncertainty 不适合训练端消除/内化；强 no-LM backbone 已近最佳判别表征；有效杠杆是 test-time decision marginalization，不是 invariance/frozen-adaptation/robust-ERM/canonicalization"*。

**→ 训练端定论穷尽**（8 机制实测/证伪 + 4 codex 收敛，8.5/10）。这不是放弃，是有证据的结论，**本身是论文强论点**。转向：test-time 论文（paperstrategy 6.5）+ 训练端反例补强（"Why Training-Time Invariance Fails" controlled-alternatives 节）。GPU 空跑 K-sweep（compute-accuracy 素材）。

---

## ★更新（2026-06-26 凌晨）：frozen-feature 类全死 → 转 LSRC（改 backbone）

- **LS-MRT 死**：冻 backbone probe，smoke 时 P(D×D linear ~1M params)过拟合暴跌 −8.694；修复（identity-reg + 降 lr + 全量 116k samples）后 **+0.028 clean FAIL**（K-cosine 不升，但 P 不帮忙）。
- **LPA 死**：query-side 加权 +0.075，预测最佳变体 acc 12.4%≈chance。
- **关键发现**：LS-MRT(+0.028)+LPA(+0.075) 两个冻结特征 probe 都 ~0 → **no-LM-loss 特征对 test-time 边缘化已近最优，重投影/重加权救不了**（oracle +4.338 是 gallery 真 ID 上界，frozen-feature 够不着）。**LCRS/LRFD/DeepSets 也是 frozen-feature 重投影，大概率同样死**。
- → 训练端价值**必须改特征本身（backbone）或改输入**。启 2 新 codex（train2_{backbone,input}）：

| 机制 | 信心 | 核心 | 依赖 | 状态 |
|---|---|---|---|---|
| **LSRC** | 7.5/10 | full fine-tune，`L_id + lam_lsrc*(bag-to-bag set-supcon logsumexp + **neg-tail 压负样本假高 lattice**)`。打 marginalization 数学瓶颈（更少假高负例+更多可复用正证据），**解释了 frozen 为何没空间**（lattice union 已固定） | 不依赖原图 ✓ | **进行中** 4090 lam0.5 + 3090 lam1.0，acc 0.98/0.97 健康，判据 lattice mAP > 79.90 +0.3 且 single 不掉 |
| BLC/LC-STN++ | 8/10→6.5 | bbox crop refiner 改输入（最贴 bbox 主导发现） | market 只裁好框→6.5 | 备选 |

---
## （旧）5 候选 frozen-feature pipeline（LS-MRT 已死，其余大概率同死）

LPA(A 加权头)定死（oracle headroom +4.338 不可达，最佳变体 query+gallery 共定单看 query 预测不出 acc≈chance）。Hard-Lattice ERM eval 中。按用户"这三个不行就找更多"启 4 新 codex（litreview2/train_more_*.md），收敛到 **5 个候选**，全避开已死的（invariance-collapse / query-side 预测最佳变体 / L_marg 分类头边缘化）。

| 机制 | 信心 | 核心 | kill-switch | 状态 |
|---|---|---|---|---|
| **LS-MRT**（set-wise 检索） | 7/10 | 把 test-time 边缘化写进**训练检索损失**：`S=logmeanexp_k sim(z_q,k, z_g)`，supervised contrastive over gallery。**在检索决策层边缘化非分类头**——直接修 L_marg 失败因（L_marg 在 train-ID classifier posterior 求均值→塌缩；LS-MRT 在 q-g 相似度证据边缘化，denominator 有真负 gallery） | 冻 backbone + cached K=9 features 训小 P(linear/BNNeck+τ)，**最廉价**；活=h12/16 ≥+0.3 且 K-cosine 不升 | **先跑** |
| LCRS（互补 residual） | 7/10 | `z_k = norm(P_shared(g_k) + α·P_k(g_k))`，shared identity core + lattice-specific residual subspace + decorr(只在分类正确后)。"准确后分工"非硬推开。test K=9 边缘化拿更富 union | 冻 backbone 训 P_shared/P_k/cls；活=K-error correlation 降+K=9 gain ≥+0.5，individual variant mAP 不掉>0.8 | 排队 |
| LRFD（disentangle） | 7/10 | `z_l=P_id, r_l=P_lat`(lattice nuisance sink, 必须能预测 lattice), per-variant CE+triplet, 推理丢 r_l 用 z_l 边缘化 | 冻 backbone 训 heads；活=with-lattice-code >+0.2 over without + r_l lattice-pred acc>60% | 排队 |
| LC-STN（对齐 canonicalize） | 7/10 | tiny localization net 预测 bbox offset(dx,dy)+grid_sample 重采样到 canonical, L_geo=SmoothL1 监督几何非身份, 残差留 marginalization | 冻 backbone 训 canonicalizer；活=offset MAE<0.35 LR px + K=1 +0.8 OR K=9 +0.3 | 排队 |
| DeepSets Marginalizer（LPA 修正） | 6.5/10 | 不预测 query 最佳变体(LPA 死因)，改 pairwise: 每个(q,g)的 K 个 cos 经 DeepSets φ/ρ 合成校准分, α 初 0 限幅, embedding 冻 | cached features 小 scorer；活=+0.4 over uniform, 且 ordinary-TTA scorer 追不平(否则只是 generic learned TTA) | 排队 |

**不做**（codex 共判）：TTT/TENT(无标签自我确认 3/10), DRO/Hard-Lattice(≈将死 4/10), PFE-Gaussian(query-side uncertainty 不够 4/10)。

**执行顺序**：LS-MRT 冻 probe 先（最廉价+最强故事）→ 活则全量；死则 LCRS→LRFD→LC-STN→DeepSets 逐个 cheap probe。**任何一个过线即训练端第二 contribution，凑成 train+test 完整方法（→7-8/10）。全死也不停——再启 codex 找更多。** novelty 共识：卖点不是各机制本身(decorr/disentangle/STN/DeepSets 都老)，是"**LR 采样格点当可枚举隐变量 + 训练端 X + 测试端 decision marginalization**"这个 problem reformulation + train/test 对齐。
