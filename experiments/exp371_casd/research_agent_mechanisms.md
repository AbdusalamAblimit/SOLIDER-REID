# exp371：LGPA 自有化机制候选组合审计

## 结论先行

围绕“保留 LGPA 已验证的约 `+0.9 mAP`，同时把方法改造成自有创新”，本轮不再把 query、GCN、matching、pose token、普通 teacher-student 或模块堆叠当作候选。机制层面一共保留六个**训练对象不同**的方案：

1. **CASD**：跨图、逐部位、严格 leave-one-view-out 的 support-advantage distillation；
2. **AERC**：把局部描述子改写成对结构化部位擦除有冗余的 error-correcting identity code；
3. **PELD**：只蒸馏 LGPA 相对 global 真正修正的 teacher-exclusive ranking events；
4. **ACC**：LGPA 只作为 privileged curriculum，选择解剖互补的同 ID 正样本；
5. **AEAD**：用 leave-one-part-out 干预估计身份有效证据，再监督无姿态 student；
6. **ASMI**：把可见部位子集视为潜变量，对结构化 visibility subsets 做 margin marginalization。

最终判断是：

> **CASD 仍是唯一同时具备较强问题新意、机制差异和清晰因果门禁的条件主线。AERC 是最正交的高风险备份，但更像“抗擦除编码”新机制，必须先过同维 frozen oracle；PELD、ACC、AEAD、ASMI 都可以很便宜地被现有 exp336/exp371 缓存特征判死，且各自已有明显的文献或仓库负证据。**

这六项不是六条待训练路线。正确执行方式是先跑共享缓存特征审计，然后最多允许一条主线进入 frozen-student kill-switch。禁止同时开六个训练实验。

## 共同事实与不可越过的边界

### 已验证资产

- `exp336` 三 seed：correct-pose LGPA 的 `equal_concat - global` 稳定为约 `+0.9 mAP`；
- `exp337` no-pose：三 seed 为负，说明无空间先验的 text query 本身不提供增益；
- `exp340/340c` canonical：CLIP `+0.7 mAP`，fixed-random `+1.1 mAP`，CLIP 词义已可判负；
- `exp357` correct 相对 cross-image pose 只多约 `0.7 mAP`；
- `exp358` anatomical channel shuffle 只掉约 `0.3 mAP`；
- 当前 descriptor 是 `global + pooled + 5 parts = 7×768 = 5376-D`；
- `exp370` 已判定结构槽写回 standard global 不成立；
- `exp109` 的 same-ID support oracle 有很大上界，但 `exp110-116` prototype bank、`exp119-131` relational transfer 只得到弱正或近中性结果。

### 不可作为贡献的内容

- GCN；
- CLIP part text 或 query 名称；
- learned matching、MaxSim、NFC、re-ranking；
- 普通 pose/parsing teacher → pose-free student；
- pose token、part token、slot、cross-attention；
- 普通 multi-shot teacher → single-shot student；
- OT、MoE、memory bank 或 loss 权重扫描；
- “测试不需要 pose”本身。

### 已知工作的统一约束

- `PGFL-KD/TSD`：普通结构 teacher → pose-free student 已有先例；
- `PAFormer`：pose heatmap 监督 pose token、推理去 pose 已直接覆盖；
- `UMTS`：multi-shot comprehensive teacher → single-shot student 已直接覆盖；
- `PFD/PAT/BPBreID/KPR/ProFD`：part decomposition、part token、part prompt、局部比较均不能单独主张；
- `PASS/PDiscoNet`：part-aware contrastive learning、part discovery/equivariance 不是空白；
- `SPT/Pose Transfer/FRT`：遮挡 patch transfer、pose transfer、检索时 complete feature recovery 已有大量近邻；
- `Residual KD/Expert-exclusive Knowledge`：只迁移 teacher 相对 student 的差异这一抽象概念已有先例；
- 2022 `Pose-guided Counterfactual Inference`：counterfactual pose 不能用作 headline。

## 总体对比

| 排名 | 方案 | 重新定义的训练对象 | 相对 LGPA 的实质变化 | 最便宜的第一刀 | 与既有工作碰撞 | 工程成本 | 当前裁决 |
|---:|---|---|---|---|---|---|---|
| 1 | **CASD** | 其他同 ID 视图形成的 LOO anatomical support advantage | LGPA 从测试描述子变为训练期 support organizer | cached support oracle | 中高，主要是 UMTS | 中 | **唯一条件主线** |
| 2 | **AERC** | 对真实部位擦除可恢复的冗余 identity code | 从 part concat 变为 erasure-resilient coding | Gate D + block-erasure codec oracle | 中，ECOC/MAE/dropout 邻域 | 中高 | 正交备份，先 oracle |
| 3 | **PELD** | teacher-exclusive ranking correction events | 从输出 teacher feature 变为输出“何时 LGPA 比 global 更对” | 三 seed ranking-event 审计 | 高，普通 KD/exclusive KD + exp119-131 | 低中 | 仅低成本备选 |
| 4 | **ACC** | anatomically complementary positive curriculum | LGPA 不作 target，只决定训练正样本 | complementarity/advantage 相关性 | 高，pose mining/metric learning | 低 | 大概率 NO-GO |
| 5 | **AEAD** | part block 的因果身份贡献 | LGPA 作为 leave-one-part-out attribution oracle | 七块 descriptor ablation | 高，attribution/exclusive KD/counterfactual | 低中 | 诊断价值大于方法价值 |
| 6 | **ASMI** | visibility subset 的边际化风险 | 从单个 part concat 变为 subset distribution objective | cached block-subset ensemble | 很高，mask consistency/PCVT | 中 | 仓库负证据强，最后考虑 |

下面每个方案都按同一九项标准展开。

---

## 候选 1：CASD

全名：**Cross-instance Anatomical Support-Advantage Distillation**。

### 1. 问题定义

遮挡图只提供不完整的 identity support。单图 LGPA 即便能稳定读取局部结构，也不能凭当前图恢复根本不可见的身份证据；训练集同一身份的其他视图却可能含有互补可见部位。

问题不再是“pose 在网络哪里注入”，而是：

> 如何把 pose 作为 privileged variable，在训练期从同 ID 其他视图中组织互补解剖证据，并且只把这些外部证据相对当前图真正增加的身份关系交给单图 student？

### 2. 训练机制

1. 用 detached LGPA 提取每张训练图的 part descriptors 与 visibility；
2. 对 anchor `i`，逐 part 只聚合 `y_j=y_i, j!=i` 的支持，硬性排除当前图；
3. 同时保留当前图 same-image teacher；
4. 计算 cross-image support 相对 same-image teacher 的 identity margin/ordering 改善；
5. 只蒸馏正 advantage，而不是对完整 multi-shot feature 做 MSE/KL；
6. correct/uniform/shuffled/wrong-ID support 是因果 controls，不是方法变体。

### 3. 测试时输入

单张 RGB。student 不读取 pose，不访问同 ID support，不使用特殊 matching；首验可诚实保留 5376-D 标准 cosine descriptor，同维化单独过 Gate D。

### 4. 相对原 LGPA 的新意

- 原 LGPA：pose 帮当前图读 local descriptors，测试仍要 pose；
- CASD：pose 只在训练期组织**其他视图**的逐部位 support；
- 方法核心不是 part query，而是 `pose-organized support + hard LOO + support-vs-self advantage` 的联合训练对象。

### 5. 与已知工作的碰撞

最高风险是 `UMTS`。如果 teacher 看到 anchor、如果直接蒸馏完整多图 feature、或如果去掉 part-wise pose organization，CASD 就退化为 UMTS 的局部版本。`PGFL-KD/TSD` 又排除了“pose teacher、测试无 pose”这一普通 claim。`Expert-exclusive Knowledge/Residual KD` 说明 advantage 本身也不能单列为首创。

因此 CASD 必须同时守住四项：

1. part-wise pose organization；
2. hard leave-one-view-out；
3. support 相对 self 的 identity-relation advantage；
4. correct vs pseudo support controls。

### 6. 最小 kill-switch

完全不训练 backbone，直接在 exp336 cached teacher parts 上做 Gate C：

- `same-image`；
- `correct LOO support`；
- `uniform/shuffled/wrong-ID support`；
- `allow-current-view` 泄漏对照；
- `full multi-shot feature` UMTS 对照。

先比较 coverage、ID CE/accuracy、hard-positive margin、positive/negative ordering 与 query-level advantage 分布。correct support 若不同时优于 self 和最强伪 control，立即停止。

### 7. 成功门槛

缓存 oracle 后，frozen-student Phase 1 必须同时满足：

- `C0 - B0 >= +0.8 mAP`；
- `C0 - max(KD0, Cx, UM0) >= +0.5 mAP`；
- pose-free student 恢复原 `+0.9 mAP` 的至少 80%，即 paired gain 不低于约 `+0.72 mAP`；
- 换任意 heatmap 后 student descriptor 逐元素不变；
- advantage 不得只来自极少数 query，需同时报告低可见/常规 query 与集中度。

### 8. 失败解释

- correct≈uniform/shuffled：pose 并未有效组织 support，CASD 的 pose-specific claim 失败；
- C0≈KD0：普通 same-image pose KD 已解释全部收益；
- C0≈UM0：方法只是 multi-shot KD，无法越过 UMTS；
- oracle 强、student 弱：跨图 evidence 对 unseen ID 不可迁移，或当前 student 不能承载 support advantage；
- 只在 allow-current-view 有效：收益来自 self leakage。

### 9. 工程成本

**中等。** Gate C 低；frozen student 中；完整训练中高。现有 P×K sampler、LGPA head、Gate B 缓存脚本可复用，不需要新增 backbone 或 test-time matcher。

---

## 候选 2：AERC

全名：**Anatomical Erasure-Resilient Coding**。

### 1. 问题定义

当前 5376-D concat 默认七个 descriptor block 均存在且等价参与距离，但遮挡本质上是结构化 part erasure。一个更根本的问题是：

> 能否把 LGPA 发现的局部身份信息编码成带冗余的定长 identity code，使若干部位缺失时仍能恢复足够的身份判别 margin？

这不是预测某个不可见部位的纹理，而是学习“哪些冗余 identity constraints 能在部分 code blocks 被擦除时仍成立”。

### 2. 训练机制

1. detached LGPA 的 pooled/五 part blocks 作为 source symbols；
2. 学习固定总维度（优先 768-D）的 source/parity projections；
3. 使用真实 pose visibility 分布采样结构化 erasure patterns，而不是独立 Bernoulli dropout；
4. 从任意合格的可见 code subset 重建 teacher identity logits/margins，而不是重建 RGB 或缺失 part feature；
5. 同时约束 full-code 与 erased-code 的 ID/triplet margin；
6. image-only encoder 最终直接输出完整 code，测试只做标准 cosine。

### 3. 测试时输入

单张 RGB，输出一个固定 768-D code；无 pose、无 part matching、无额外 support。

### 4. 相对原 LGPA 的新意

- 原 LGPA 是多个独立 part descriptors 的高维并联；
- AERC 把局部证据重写为**结构化擦除条件下的冗余身份编码问题**；
- pose 的作用是给出真实 erasure channel，而不是定位测试 descriptor。

### 5. 与已知工作的碰撞

风险来自 ECOC、dropout、masked autoencoder、robust part representation 与普通 feature compression。当前查新没有在本仓库记录的 ReID 最近邻中发现“pose-defined structural erasure channel + parity identity code”的直接完整先例，但这不是充分的新颖性证明；正式推进前仍需专项查 ECOC/erasure coding in metric learning。

若实现只是“part dropout + MLP 压缩”，立即失去新意。必须有显式 source/parity code、可解码的 margin recovery 与真实 visibility erasure controls。

### 6. 最小 kill-switch

无需训练 backbone：

1. 复用 Gate D 的 train-only JL/PCA-768；
2. 在 cached 7-block descriptors 上按真实 visibility replay block erasure；
3. 训练一个严格 train-only 的线性或单层 parity codec；
4. 比较同维 PCA/JL、普通 MLP packing 与 AERC 在 full、低可见四分位、随机 erasure、真实 erasure 下的检索指标。

若连 frozen linear/one-layer oracle 都不能在真实 erasure 下恢复 margin，禁止开 end-to-end。

### 7. 成功门槛

- full-code paired gain retention `R>=0.80`；
- 低可见四分位相对同维 PCA 至少 `+1.0 mAP`；
- 高可见四分位相对 PCA 退化不超过 `0.3 mAP`；
- correct visibility erasure 训练必须优于随机 block dropout 至少 `+0.5 mAP`，否则 pose 变量没有独立价值；
- 三 seed 的增益方向必须一致。

### 8. 失败解释

- Gate D 本身失败：LGPA 增益依赖高维 concat，定长 coding 基础不存在；
- parity 不优于 PCA/MLP：所谓 coding 只是压缩包装；
- 只对 synthetic erasure 有效：训练 channel 与真实遮挡不匹配；
- clean set 明显退化：冗余约束牺牲了可辨识细节；
- correct erasure≈random dropout：pose 不是必要变量。

### 9. 工程成本

**中高。** Frozen oracle 中等；端到端需要 codec、erasure simulator 与 image-only code head。优点是与 CASD 正交、测试系统最简；缺点是专项查新压力和“只是压缩/正则化”的审稿风险都很高。

---

## 候选 3：PELD

全名：**Pose-Expert Lift Distillation**。

### 1. 问题定义

LGPA 的平均增益只有约 `+0.9 mAP`，说明它不是对所有 pair 都更好。普通 KD 会把 teacher 的正确关系、无效关系和错误关系一起迁移。真正的问题可以改写为：

> 哪些 ranking events 是同一 checkpoint 的 LGPA descriptor 相对 global descriptor 独占修正的，能否只迁移这些 teacher-exclusive events？

### 2. 训练机制

1. 同一 detached backbone 同时产生 global control 与 LGPA teacher descriptor；
2. 对 batch/queue 中正负关系，计算 teacher 与 global 的 margin lift；
3. 只保留 teacher 将 global violation 修正、且跨增强/跨 checkpoint 稳定的事件；
4. student 只拟合这些 ordering corrections，其他 pair 继续由标准 ID/triplet 学习；
5. 不蒸馏完整 feature，不引入 pair-specific test scorer。

### 3. 测试时输入

单张 RGB，标准 image-only descriptor 与 cosine；无 pose、无 support、无特殊 matching。

### 4. 相对原 LGPA 的新意

LGPA 不再是最终 descriptor，而是一个 privileged **ranking-error corrector**。训练对象从“模仿 part feature”变成“复现 LGPA 独占修正的离散 ordering events”。

### 5. 与已知工作的碰撞

碰撞很高：`PGFL-KD/TSD` 覆盖普通 pose KD；`Expert-exclusive Knowledge/Residual KD` 覆盖“迁移老师相对学生的独占差异”；仓库 `exp119-131` 已系统测试 common-support relational、pair delta、sparse routing、residual target 与 queue coverage，最终只弱正/近中性。

PELD 只有“同 checkpoint global control 定义离散 teacher-win events，并以跨干预稳定性做硬门禁”这一窄差异。它更适合作为 CASD 的 loss control，而不是优先 headline。

### 6. 最小 kill-switch

直接读取 exp336 三 seed 的 global/equal-concat query-gallery descriptors：

- 统计 global hard-positive/negative violations 中，LGPA 修正了多少、又新增了多少错误；
- 比较 corrected/broken ratio；
- 检查同一 query 的 teacher lift 是否跨 seed/pose intervention 稳定；
- 检查正 lift 是否被少数 query 垄断。

无需训练即可判定是否存在足够密集、稳定的 teacher-exclusive supervision。

### 7. 成功门槛

- 每个 seed 的 `corrected / broken >= 2.0`；
- 至少 15% 的 global hard violations 被 LGPA 稳定修正；
- top 10% queries 不得占超过 40% 的总 positive lift；
- frozen student 相对 B0 至少 `+0.72 mAP`，且相对 full relational KD / same-image KD 至少 `+0.5 mAP`；
- 最终三 seed paired gain 均为正。

### 8. 失败解释

- corrected≈broken：`+0.9 mAP` 是连续距离的微小整体变化，不存在可迁移的离散事件；
- lift 高度集中：方法只学 benchmark 少数极难 query；
- oracle 事件清晰、student 不涨：pair-specific correction 无法压入单个 standard descriptor，这与 `exp131` 后的旧结论一致；
- 与 full KD 持平：exclusive routing 没有独立机制价值。

### 9. 工程成本

**低中。** Descriptor 审计低，frozen student 中。因为既有负证据多，不应在未过三 seed event audit 前投入训练。

---

## 候选 4：ACC

全名：**Anatomical Complementarity Curriculum**。

### 1. 问题定义

P×K sampler 给出同 ID 多图，但随机选择的正样本未必在可见部位上互补。与其蒸馏 feature，可以问：

> LGPA 能否只作为训练期 privileged sampler，把最互补、且确实能修正 global identity margin 的同 ID 图像优先暴露给 image-only student？

### 2. 训练机制

1. detached LGPA/visibility 计算同 ID 图像之间的 coverage complementarity；
2. 从 P×K batch 中选择“anchor 缺失、positive 可见”的正样本；
3. 只在 teacher 显示该 complementary pair 比普通 positive 有更好 identity margin 时启用；
4. student 使用标准 supervised contrastive/triplet，不模仿 teacher feature；
5. curriculum 从普通随机 positive 逐步过渡到 advantage-confirmed complementary positive。

### 3. 测试时输入

单张 RGB、标准 global/定长 descriptor 与 cosine；无 pose、无局部 matching。

### 4. 相对原 LGPA 的新意

LGPA 从 descriptor 变成数据选择器；pose 不进入 feature target，只决定哪些跨图关系值得学习。

### 5. 与已知工作的碰撞

pose-aware hard mining、metric learning 与 curriculum learning 都是成熟邻域。仓库 `exp027` pose-similarity triplet 中性、`exp047` common-support mining 因正负 overlap 无法区分而失败、`exp051` per-keypoint metric learning 中性。若只是按 complementarity 选 hardest positive，它只能是训练 recipe，不能成为主贡献。

唯一可能保留的差异是：positive 必须同时满足“部位互补”和“LGPA 相对 global 的实测 advantage”，而非仅凭 pose distance。

### 6. 最小 kill-switch

在 exp336 cached descriptors/visibility 上：

- 对同 ID pairs 计算 anatomical complementarity；
- 计算该 pair 的 LGPA-vs-global positive-margin lift；
- 做三 seed Spearman 相关、分位数 lift 与身份/摄像头分层；
- 与随机 positive、hardest positive、最大 pose distance 比较。

若 complementarity 不能预测 teacher advantage，立即停止。

### 7. 成功门槛

- 三 seed 的 Spearman `rho >= 0.20` 且方向一致；
- complementarity top quartile 的 median teacher margin lift 至少是 bottom quartile 的 2 倍，且其 global-violation correction rate 至少高 10 个百分点；
- frozen student 相对同 sampler B0 至少 `+0.8 mAP`；
- 相对普通 hardest-positive curriculum 至少 `+0.5 mAP`；
- clean/high-vis query 不得明显退化。

### 8. 失败解释

- complementarity 与 lift 无关：姿态互补不等于身份信息互补；
- 只比随机好、不比 hard mining 好：只是普通 curriculum；
- 训练 loss 改善而检索不涨：selected positives 过难或强化了 view invariance，而没有增加 unseen-ID identity evidence；
- 仅早期加速：重演 `exp148 PCVT` 的收敛形态。

### 9. 工程成本

**低。** 统计和 sampler 修改都便宜，适合作为快速排除项；但新颖性弱，即便成功更适合作为 CASD 的 support sampling 辅助，而不是论文主方法。

---

## 候选 5：AEAD

全名：**Anatomical Evidence-Attribution Distillation**。

### 1. 问题定义

pose 能定位身体区域，但不同图像真正有身份判别力的局部并不相同。canonical 与 channel-shuffle 仍能保留大部分 LGPA 增益，进一步提示“固定头/躯干/腿语义”可能不是关键。问题可改写为：

> 哪些 LGPA blocks 对当前 query 的正确 identity margin 有真实因果贡献，能否让 image-only student 学到这种 evidence allocation？

### 2. 训练机制

1. 对 `global + pooled + 5 parts` 做 leave-one-block-out 干预；
2. 以删除某 block 后的 positive/negative margin 变化定义其正、负或冗余贡献；
3. 只蒸馏跨增强稳定的 positive contribution；
4. student 把 768-D descriptor 分成共享 subspaces，但最终仍直接拼接/求和后做标准 cosine；
5. 不预测 pose label，不使用 test-time part matching。

### 3. 测试时输入

单张 RGB 与一个标准定长 descriptor；无 pose、无 block-wise matching。

### 4. 相对原 LGPA 的新意

原 LGPA 假定所有 part descriptors 直接参与距离；AEAD 用 LGPA 做可干预的 evidence source，训练目标是“谁真正改变了身份判断”，而不是 part feature 本身。

### 5. 与已知工作的碰撞

与 feature-attribution distillation、teacher-exclusive knowledge、counterfactual inference 高度相邻。2022 pose counterfactual inference 又使“counterfactual part evidence”不能成为 headline。`exp358` 的 channel shuffle 小损失可能直接表明 anatomical block identity 不稳定。

如果 attribution 最终只是一个 learned part weight，它会退化为 visibility scorer/matching，明确不具备创新性。

### 6. 最小 kill-switch

这是六个候选中最便宜的：对 exp336 三 seed 的七块缓存 descriptor 做 exact leave-one-block-out retrieval/margin audit：

- 单 block marginal contribution；
- block pair synergy/redundancy；
- query-level sign stability；
- correct/canonical/shuffled 下 attribution rank 的一致性；
- attribution 与 heatmap visibility 的关系。

无需训练、无需新 forward。

### 7. 成功门槛

- 至少三个 local blocks 的平均 marginal contribution 在三 seed 均为正；
- query-level positive/negative contribution sign 的跨 seed一致率至少 80%；
- 使用 oracle positive-attribution weighting 的 paired gain 至少比 uniform concat 多 `+0.5 mAP`；
- frozen student 至少恢复 `+0.72 mAP`，并领先普通 full teacher KD `+0.5 mAP`。

### 8. 失败解释

- block attribution 不稳定：LGPA 收益来自高维整体 ensemble，而不是可解释 anatomy；
- pooled block 几乎解释全部收益：五个 anatomical blocks 是冗余维度；
- correct/canonical/shuffled attribution 近似：pose-specific claim 失败；
- oracle weighting 有效但 student 不涨：贡献是 query/pair-specific 的，不能压入固定 descriptor；
- 退化为 learned weights：只能算 fusion trick。

### 9. 工程成本

**低中。** Oracle 极低，student 中。主要价值是给 LGPA 机制做归因，不足以单独保证论文新颖性。

---

## 候选 6：ASMI

全名：**Anatomical Support-Subset Marginalization**。

### 1. 问题定义

当前 concat 把可见支持视为一个确定点，但遮挡使“哪些部位可见”本身是结构化随机变量。问题可以定义为：

> 能否对真实 visibility subsets 下的 identity margin 做边际化，使 image-only descriptor 不依赖某一个特定可见部位组合？

### 2. 训练机制

1. LGPA/pose 只用于产生训练集中的真实部位子集分布；
2. 对同一图或同 ID 视图构造多个结构化 subset descriptors；
3. 优化 expected margin 与 lower-quantile/worst-subset margin，而不是逐 subset feature consistency；
4. student 输出单个 image-only descriptor，逼近 subset distribution 的稳健 identity barycenter；
5. 不在测试时枚举 subsets，不做 pair-specific matching。

### 3. 测试时输入

单张 RGB，一个标准 descriptor 与 cosine。

### 4. 相对原 LGPA 的新意

原 LGPA 输出单次 local concat；ASMI 把它变成对“可见支持集合”分布的训练期观测器，训练对象是 visibility-set risk，而不是某个 part feature。

### 5. 与已知工作的碰撞

与 part dropout、mask consistency、robust optimization、PCVT、PASS/SPT 很接近。仓库 `exp050` pose-aware masking consistency 中性，`exp148 PCVT` 早期加速但后期转负，且训练集约 95.8% 样本近乎全可见，真实 subset 信号不足。

只有“基于真实 visibility distribution 的 margin marginalization，而非 feature consistency/augmentation”可形成窄差异，创新风险仍高。

### 6. 最小 kill-switch

在 cached 7-block descriptors 上枚举或采样真实 visibility subsets：

- full concat；
- uniform block dropout；
- empirical pose-conditioned subsets；
- expected descriptor；
- lower-quantile margin optimized linear combiner。

若任何 train-only frozen marginalizer 都不能改善低可见 query，同时保持全量指标，就不训练 student。

### 7. 成功门槛

- frozen oracle 在低可见四分位至少 `+1.0 mAP`；
- 全量 mAP 相对 full LGPA 退化不超过 `0.2`；
- empirical subsets 必须领先 uniform dropout 至少 `+0.5 mAP`；
- frozen student 相对普通 mask-consistency control 至少 `+0.5 mAP`，相对 B0 至少 `+0.72 mAP`；
- 不能只表现为前 40 epoch 加速。

### 8. 失败解释

- empirical≈uniform：pose visibility 分布不是必要变量；
- 低可见涨、全量掉：只是在重新加权 benchmark 子集；
- oracle 有效、student 无效：单图无法从不可见证据分布中恢复身份信息；
- 只早期加速：重演 PCVT，最终表示没有获得新 support；
- 训练集几乎全可见：数据本身不给 subset marginalization 足够监督。

### 9. 工程成本

**中等。** Cached oracle 中等，端到端需要 subset sampler 与 robust margin objective。由于既有负证据强，排在最后。

---

## 共享的低成本研究顺序

六个候选不应各自重复提特征。一次 exp336 三 seed 缓存即可回答大部分前置问题：

### Step 1：先完成已计划的 Gate B / Gate D

1. correct/canonical/shuffled/uniform/no-pose inference intervention；
2. 缓存 train/val 的 global、pooled、五 part blocks、visibility、PID/CAMID；
3. 完成 JL/PCA-768 paired-gain retention。

这一步同时给 CASD、AERC、AEAD、ASMI 提供输入。

### Step 2：在同一缓存上并行做四个纯分析 gate

1. **CASD**：LOO support vs self/pseudo/wrong-ID；
2. **PELD**：teacher-exclusive corrected/broken ranking events；
3. **ACC**：complementarity 与 teacher lift 相关性；
4. **AEAD**：七块 exact ablation 与 attribution stability；
5. **AERC/ASMI**：同维 codec 与 subset/erasure oracle。

所有分析必须固定 evaluator，不得根据 query/gallery 结果调 projection、threshold 或 subset hyperparameter。

### Step 3：只允许一个训练 kill-switch

- CASD Gate C 通过：只跑 CASD frozen-student 六臂；
- CASD 失败但 AERC 同维/erasure oracle 强：允许 AERC frozen codec/student；
- PELD/ACC/AEAD/ASMI 即使单个统计看起来正，也必须先证明超过各自最强普通 control，不能直接开 full training；
- 两条都不过：正式停止“LGPA 自有化主线”，保留 LGPA 为实验资产，不再用 OT/MoE/slot/权重扫描救场。

## 最终推荐

### 主路线

继续 **CASD**，但必须把它视为一个待证伪假设，而不是已成立方法。最关键的不是完整训练，而是先证明：

> correct pose 组织的、严格不含 anchor 的同 ID part support，确实比 same-image teacher、uniform/shuffled/wrong-ID support 和 UMTS 式 full multi-shot teacher 提供更稳定的 identity advantage。

### 备份路线

只保留 **AERC** 作为真正正交的备份，因为它不依赖 multi-shot support 或普通 KD，而把问题重写为结构化 part erasure 下的 identity coding。但它必须先过 Gate D 与 frozen erasure codec oracle，并补专项文献查新。

### 其余路线的定位

- **PELD**：CASD 的 advantage-loss 对照；
- **ACC**：CASD 的 support sampling 分析，不作 headline；
- **AEAD**：LGPA 机制归因工具，不作主方法；
- **ASMI**：只有在 cached subset oracle 出现大幅上界时才重新考虑。

这套排序最大限度利用已验证 LGPA 资产，也明确承认：稳定 `+0.9 mAP` 只证明 local descriptor 有用，并不自动证明任何新故事。自有创新必须来自新的训练对象与可被 controls 独立裁决的机制。
