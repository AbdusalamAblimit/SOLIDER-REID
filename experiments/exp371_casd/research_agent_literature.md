# exp371 CASD：截至 2026-07 的外部系统查新

日期：2026-07-13

范围：pose-guided / pose-free occluded ReID、LUPI、同 ID 跨图 support、多视图 teacher-student、leave-one-out、关系/间隔蒸馏，以及 AERC/ECOC/feature-erasure coding。
约束：本文件只裁决**外部新颖性**；仓库内部 `exp120/123/125/129/130` 是前驱与强对照，不是外部 prior。

## 2026-07-13 未决直接近邻覆盖说明

最新审计见 `critical_prior_audit_2026.md`。Neurocomputing 2026 已出现 *Learning from multi-view fragments: An adaptive consistency distillation framework for occluded person re-identification*（DOI `10.1016/j.neucom.2026.133015`）。当前只取得 DOI、出版社、Semantic Scholar 与 ORCID 元数据，未取得合法公开摘要/全文，不能从标题推断它是否覆盖 strict LOO、pose-response 或 support-vs-self advantage。

此外，ACM MM 2023 的 `LCR²S` 已显式用同 ID 其他视图构造 support set，把 current+support 融合后的 enriched feature 与 relation matrix 蒸馏给单输入 student。所以下文关于 support set、other-view teacher 和 relation distillation 的单项新意全部作废；CASD 只剩“target 层面严格排除 current evidence + raw pose-response 逐部位 routing + support-vs-self 增量关系迁移”的联合差分。

因此，下文“尚未发现完整组合”降级为**待该论文全文审计的条件判断**。Gate C 可继续作为内部 kill-switch，但在全文八项机制核验完成前，不得给 CASD 最终外部新颖性 GO。

## 结论先行

### CASD

**当前原始故事发生直接撞车，但严格机制组合尚未发现完整先例。**

`MVI²P` 已在 Information Fusion 明确提出：同 ID 多张遮挡图提供互补信息；训练时整合这些图像形成 comprehensive representation；通过知识蒸馏让测试时单图分支获得多视图信息。其问题定义、动机措辞和训练/推理形态都与当前 CASD 原故事直接重合，而且有公开代码。

因此不能再主张：

- 首次发现 single-image support incomplete；
- 首次利用同 ID 其他图像补全遮挡证据；
- 首次 multi-view / multi-shot teacher 教 single-image student；
- 首次在遮挡 ReID 中把多图完整知识传播到单图。

截至本轮检索，仍未发现以下**完整组合**：

> 训练期 pose-aware extractor 按解剖部位组织同 ID 其他视图，严格排除 anchor，形成 leave-one-view-out support；再只迁移该 support 相对 same-image teacher 真正改善的 identity relation/margin，并在测试时完全去掉 pose 与 support。

所以 CASD 只能获得**条件 GO**：新意不再是“多图解决单图不完整”，而是：

1. **complementarity isolation**：从含 anchor 的多图 teacher 中分离“其他视图真正新增的证据”；
2. **pose-organized strict LOO support**：逐部位组织、严禁 current-view leakage；
3. **verified support-gain transfer**：只迁移相对 same-image teacher 的正向关系修正，不做完整 feature imitation；
4. **因果门禁**：correct pose support 必须优于 identity-only、uniform、slot permutation、wrong-ID、same-image KD 和 full multi-shot KD。

缺少其中任一项，CASD 都会退化为 `MVI²P / UMTS + part-wise pose`，不足以作为主创新。

### AERC

**作为独立主创新正式 NO-GO。**

2025 年 IEEE Access 的 [Neural Network Coding Layer (NNCL)](https://doi.org/10.1109/access.2025.3610080) 已经明确提出：用线性编码向中间特征加入 structured redundancy，在 feature erasure 后通过显式代数机制恢复丢失特征；支持 fixed / learnable coding matrix，并在 ResNet、EfficientNet、ViT 上验证。其[官方代码](https://github.com/quarry0226/NNCL)实现：

```text
y = A x,  dim(y) = ceil((1+k) dim(x))
Z = y - A_K x_K
x_E = pinv(A_E) Z
```

这与 AERC 的“source/parity code + 结构化擦除 + 可恢复冗余”不是概念相似，而是机制级直接重合。AERC 只剩“pose 定义人体部位擦除、ReID metric 而非分类”的任务适配差异，难以支撑独立方法首创。

AERC 可以降为：

- NNCL 在 anatomical block erasure 上的应用/强基线；
- CASD 失败后的工程性鲁棒压缩实验；
- 论文附录中的 descriptor robustness 分析。

不能再写成“首次将 error-correcting / erasure coding 引入神经特征”或“提出新的 parity identity code”。

## 最高风险直接邻居：MVI²P

### 论文与代码

- Neng Dong et al., [Multi-view Information Integration and Propagation for Occluded Person Re-identification](https://doi.org/10.1016/j.inffus.2023.102201), Information Fusion，online 2023 / journal volume 2024；[arXiv](https://arxiv.org/abs/2311.03828)。
- 官方实现：[nengdong96/MVIIP](https://github.com/nengdong96/MVIIP)。

### 它已经覆盖的内容

论文原始问题就是：遮挡下单图可见信息有限，而同一行人的多张图具有互补可见区域。方法在 P×K batch 内：

1. 用 CAM localization 过滤各图中的身份无关区域；
2. 用正确类别概率给同 ID 各图加权；
3. 对同 ID 多张 feature map 做加权相加，形成 comprehensive representation；
4. 用 L2 knowledge distillation 将综合向量传播给单图向量；
5. 测试只保留单图 baseline branch。

### 代码级核验

公开实现进一步确认：

- [`data_loader/sampler.py`](https://github.com/nengdong96/MVIIP/blob/main/data_loader/sampler.py) 组织同 ID 的 K 个实例；
- [`network/processing.py`](https://github.com/nengdong96/MVIIP/blob/main/network/processing.py) 将同 ID 四张 feature map 的 `[0..3]` 全部相加；
- [`core/train.py`](https://github.com/nengdong96/MVIIP/blob/main/core/train.py) 让单图 `bn_features` 对齐 integrated feature；
- [`tools/loss.py`](https://github.com/nengdong96/MVIIP/blob/main/tools/loss.py) 将一个 integrated feature 复制给四张单图并做 L2。

### CASD 仍可争的精确差分

MVI²P 也暴露了 CASD 可以利用的三个未解决点：

1. **teacher 包含 anchor 本身**：其 integrated target 含当前学生图，无法隔离“其他视图新增了什么”；
2. **不按跨图解剖部位对齐**：CAM 只做身份显著区域定位，不能保证同一 body part 的 evidence 被正确汇聚；
3. **完整向量蒸馏**：所有综合信息都做 L2，不判断 support 是否比当前图自身 teacher 更好，也不区分可迁移的关系改善与不可从单图推断的纹理。

这三个差分必须同时实现并以对照证明；它们不是可以只写在文字里的区别。

## 外部最近邻矩阵

| 工作 | 年份/代码 | 核心机制 | 与 CASD 的重叠 | 尚未覆盖的 CASD 条件 | 风险 |
|---|---|---|---|---|---|
| [MVI²P](https://arxiv.org/abs/2311.03828) | 2023/2024；[代码](https://github.com/nengdong96/MVIIP) | 同 ID 多图 CAM 定位、可靠性加权、综合 feature→单图 L2 propagation | **问题、数据组织、单图推理、遮挡场景全部直接重合** | strict LOO、pose part alignment、support-vs-self positive gain | **致命邻居** |
| [UMTS](https://arxiv.org/abs/2001.05197) | AAAI 2020 | 同 ID K-shot 拼接 teacher→其中一张 single-shot student，多阶段 uncertainty KD | multi-shot comprehensive teacher→single image | teacher 含 anchor；无逐部位 LOO；无 gain-only transfer | 很高 |
| [VKD](https://arxiv.org/abs/2007.04174) | ECCV 2020；[代码](https://github.com/aimagelab/VKD) | 多帧/多视图 video teacher→少帧 student | multiple-view KD、测试减少视图 | 视频任务；无 pose/LOO/gain filter | 高 |
| [Temporal Knowledge Propagation](https://doi.org/10.1109/ICCV.2019.00974) | ICCV 2019 | video representation 的 temporal knowledge 传播给 image network | privileged multi-view→single image | 非遮挡 anatomy support | 中高 |
| [Holistic Guidance](https://arxiv.org/abs/2104.06524) | BMVC 2021 | holistic reference 的 within/between-class distance distributions 教 occluded model | privileged relation / margin 与遮挡 ReID | 非同 ID LOO support；无 pose parts | 高 |
| [Metric Learning using Privileged Information](https://arxiv.org/abs/1904.05005) | TIP 2019 | privileged-space distance 作 original-space 的局部判别阈值 | train-only privileged metric/margin | 非深度 pose support、非跨实例部位聚合 | 高 |
| [Factorized Distillation](https://arxiv.org/abs/1811.08073) | 2018 | 多个 partial-ReID teachers 的 feature maps / retrieval features 蒸馏给 holistic model | part teachers→holistic image model | 无同 ID LOO support / gain isolation | 高 |
| [PGFL-KD](https://arxiv.org/abs/2108.00139) | ACM MM 2021 | pose branches 教 main branch，测试丢弃 pose | pose privileged、pose-free inference | 只做 same-image pose KD | 很高 |
| [TSD](https://arxiv.org/abs/2312.09797) | ICASSP 2024；[代码](https://github.com/hh23333/TSD) | parsing-aware teacher decoder→standard student decoder | structural teacher、part feature distill、pose-free student | 无 cross-instance LOO support | 很高 |
| [PAFormer](https://arxiv.org/abs/2408.05918) | 2024 | heatmap supervision、learnable pose tokens、visibility predictor、推理无需 pose estimator | pose-aware part tokens 与 pose-free inference | 无同 ID LOO support | 很高 |
| [PFD](https://arxiv.org/abs/2112.02466) | AAAI 2022；[代码](https://github.com/WangTaoAs/PFD_Net) | pose-guided part disentangling / visible-part matching | pose-conditioned anatomy parts | 无跨图 support distillation | 中高 |
| [Feature Completion Transformer](https://arxiv.org/abs/2303.01656) | TMM 2024 | 自生成 occluded-holistic pairs、completion decoder、distribution consistency | 缺失 feature 补全、同 ID metric bridging | support 来自同图增广，不是其他同 ID LOO | 中高 |
| [Feature Completion / RFCnet](https://arxiv.org/abs/2106.12733) | TPAMI 2021 | 空间/时间上下文恢复遮挡区域 feature | feature completion / temporal support | 无 pose LOO advantage | 中 |
| [Neighbourhood-guided Feature Reconstruction](https://arxiv.org/abs/2105.07345) | 2021 | 测试时从 gallery neighbours 图恢复 full-body feature | 外部图像 support / completion | 测试依赖 gallery；不保证同 ID；非 train-only student | 中高 |
| [RKD](https://arxiv.org/abs/1904.05068) | CVPR 2019 | distance-wise / angle-wise relation distillation | relation distillation | 无 support-vs-self conditional gain | 高（对 loss claim） |
| [DarkRank](https://doi.org/10.1609/aaai.v32i1.11783) | AAAI 2018 | 跨样本排名与 similarity transfer | ranking relation transfer | 无 pose LOO support | 高（对 loss claim） |
| [Similarity-Preserving KD](https://arxiv.org/abs/1907.09682) | ICCV 2019 | 保留 teacher 的 pairwise similarities | pair relation distillation | 无 conditional support gain | 高（对 loss claim） |
| [Triplet Distillation](https://arxiv.org/abs/1905.04457) | 2019 | teacher similarity 动态生成正负 margin | teacher-driven adaptive margin | 无 pose LOO support | 高（对 margin claim） |
| [Attribute-guided Metric Distillation](https://arxiv.org/abs/2103.01451) | ICCV 2021 | 将 ReID pair distance 分解到属性贡献 | structured semantic distance distillation | 解释器目标；非跨图 support | 中高 |

## LUPI 边界

CASD 本质上属于 learning using privileged information：训练期 teacher 额外看到 pose 和同 ID 其他视图，测试 student 只看当前 RGB。这个大框架不是新意：

- [Learning Using Privileged Information](https://doi.org/10.1162/NECO.2009.12-07-661) 已提出训练期额外信息范式；
- [Unifying Distillation and Privileged Information](https://arxiv.org/abs/1511.03643) 已把 KD 与 LUPI 统一；
- PGFL-KD、TSD、Person ReID with Metric Learning using Privileged Information 已把该范式带入 ReID / pose / metric learning；
- [PKDOT](https://arxiv.org/abs/2401.15489) 已说明 privileged teacher 的结构关系也可被蒸馏，而不只做 point-wise matching。

因此论文中应写“pose and cross-view support are privileged training variables”，不能把 privileged distillation 本身列为贡献。

## `support advantage / pair delta` 的边界

仓库内部 `exp123` 不构成外部 prior，因此它不会自动否定 CASD 的外部新颖性；但外部文献已经覆盖了关系、排名、动态 margin 和 privileged metric：

- RKD：样本间距离/角度关系；
- DarkRank：cross-sample ranking；
- Similarity-Preserving KD：pairwise similarity geometry；
- Triplet Distillation：teacher 自适应 triplet margin；
- Holistic Guidance：holistic 与 occluded 的 within/between-class distance distribution；
- Metric LUPI：privileged distance 充当局部判别阈值。

因此“只蒸馏 teacher 改变的 pair”或“teacher margin 比 student/self 更好才蒸馏”不能单独作为论文首创。它只能作为 CASD 用于**隔离跨视图 support 的新增贡献**的必要机制。

另外，“Advantage Distillation”在检索中首先对应密码学/量子密钥分发中的成熟术语。建议方法名保留 `CASD = Cross-instance Anatomical Support Distillation`，正文用 `support-gain filtering` 或 `support-relative relational distillation`，不要把 “Advantage Distillation” 放进论文标题。

## AERC / ECOC / erasure coding 专项裁决

### 直接机制撞车

[NNCL](https://doi.org/10.1109/access.2025.3610080) 的论文摘要与[官方实现](https://github.com/quarry0226/NNCL)共同覆盖：

- 对中间 feature 做线性冗余编码；
- fixed orthogonal 或 learnable coding matrix；
- 显式识别 erased / known feature indices；
- 对 erased 子矩阵求 pseudo-inverse，代数恢复丢失 feature；
- 可加 reconstruction loss；
- 在 ResNet、EfficientNet、ViT 上评估 20%～60% feature erasure。

这已经覆盖 AERC 最可能形成贡献的“显式 parity / 可解码冗余”，且比普通 dropout/MAE 更直接。

### 其他边界

| 工作 | 已覆盖内容 | 对 AERC 的约束 |
|---|---|---|
| [Error-Correcting Output Codes with Ensemble Diversity](https://doi.org/10.1609/aaai.v35i11.17169) | 设计大 Hamming distance codeword 与多 binary classifier，端到端鲁棒分类 | ECOC / error-correcting classifier 不是新意；但它面向 seen classes，不等于 unseen-ID metric code |
| [Error-correcting output codes based ensemble feature extraction](https://doi.org/10.1016/j.patcog.2012.10.015) | ECOC 驱动 ensemble feature extraction | “编码产生鲁棒 feature ensemble”也已有先例 |
| [Person Re-ID Based on Feature Erasure and Diverse Feature Learning](https://doi.org/10.1049/cvi2.12108) | DropBlock 擦除最高激活区域，多分支学习互补 ReID feature | ReID 中 feature erasure / redundancy 训练不是新意；但无代数 parity recovery |
| [RFCnet](https://arxiv.org/abs/2106.12733) / [FCFormer](https://arxiv.org/abs/2303.01656) | 从空间上下文或同图 holistic pair 恢复被遮挡 feature | “恢复缺失部位 feature”不是新问题 |

### 可保留但不能独立 claim 的差异

AERC 相对 NNCL 只剩：

1. erasure unit 是语义 anatomy block，不是随机 scalar feature；
2. erasure mask 来自 pose visibility；
3. 优化目标是 unseen-ID retrieval geometry，而非 seen-class accuracy；
4. 希望在固定 768-D 预算下保留 LGPA gain。

这些足够定义一个**任务适配实验**，不足以安全声称新的 coding mechanism。若仍做，必须把 NNCL fixed/learnable matrix 作为直接基线，并证明 pose-defined block erasure 明显优于 random erasure 与普通 block dropout；否则没有论文价值。

## 外部新颖性差分表

| 条件 | UMTS | MVI²P | PGFL-KD / TSD | CASD 目标 |
|---|---:|---:|---:|---:|
| 同 ID 多图 teacher/support | ✓ | ✓ | ✗ | ✓ |
| 专门面向 occluded ReID | ✗ | ✓ | ✓ | ✓ |
| 测试 single-image / pose-free | ✓ | ✓ | ✓ | ✓ |
| 按解剖部位跨图组织 support | ✗ | ✗ | 仅同图 | ✓ |
| teacher 严格排除 anchor | ✗ | ✗ | 不适用 | ✓ |
| 只迁移 support 相对 self 的正向关系增量 | ✗ | ✗ | ✗ | ✓ |
| correct / identity-only / shuffled / wrong-ID 因果控制 | ✗ | ✗ | ✗ | ✓ |

表中前三行已经完全不是新意。CASD 的论文贡献只能落在后三行，并且必须由实验共同成立。

## 可安全 claim

若 Gate C 和 frozen-student 对照通过，可谨慎主张：

1. 我们指出现有 multi-view ReID distillation 将 anchor self-evidence 与 cross-view complementary evidence 混在同一 teacher target 中，无法确认其他视图真正新增了什么；
2. 我们构造 pose-organized、part-wise、strict leave-one-view-out 的同 ID support，隔离跨视图新增证据；
3. 我们只迁移该 support 相对 same-image teacher 确认改善的 retrieval relations，避免完整 feature copying；
4. image-only student 在测试时不访问 pose、support、同 ID 分组或特殊 matching；
5. 通过 identity-only、slot permutation、wrong-ID、same-image KD 和 full multi-shot KD 证明收益来自 pose 组织的跨视图互补 support。

不能主张：

- 首次 multi-view / multi-shot ReID distillation；
- 首次利用同 ID 多图解决遮挡；
- 首次 pose-free pose distillation；
- 首次关系、排名、margin 或 privileged distillation；
- 首次 error-correcting / erasure-resilient neural feature coding；
- 当前 Gate B 中 `correct≈shuffled≈canonical` 已证明 pose anatomy 是 support 的关键变量。

## GO / NO-GO

### CASD：条件 GO，但必须改故事

下一步应把问题改写为：

> 现有 multi-view distillation 虽利用同 ID 多图，却把当前图也放入综合 teacher，并整向量模仿，因而无法隔离可归因、可迁移的 cross-view complement。CASD 用 pose 组织 strict LOO anatomical support，只迁移相对 self teacher 可验证的 relation gain。

不是：

> 现有方法都只看单图；我们首次用同 ID 多图补全 single-image support。

进入训练前的外部新颖性门槛：

1. Gate C 的 strict LOO correct support 必须优于 same-image 与最强 identity-only / slot-permutation control；
2. frozen student `C0` 必须优于 `KD0`、`UM0` 和 strongest pseudo-support；
3. current-view-on 对照必须显示泄漏风险，证明 LOO 不是装饰；
4. 若 correct 不优于 identity-only / permutation，则必须删除 pose-specific claim；若此时只剩同 ID 多图 support，直接被 MVI²P/UMTS 覆盖，CASD **NO-GO**；
5. 若 gain-only 不优于 full feature KD，则机制退化为 MVI²P/UMTS，CASD **NO-GO**。

### AERC：独立主创新 NO-GO

不要再为 AERC 启动独立论文主线。最多做 frozen oracle，并以 NNCL 为直接基线；除非未来提出明显不同于线性编码 + pseudo-inverse restoration 的新 coding principle，否则不应升级为主贡献。

## 本轮查新覆盖与代码核验

本轮检索覆盖 arXiv、Crossref、OpenAlex、Semantic Scholar 元数据和 GitHub；搜索词包括 pose-guided / pose-free ReID、privileged information、multi-shot / multi-view teacher-student、same-identity complementary views、feature completion、leave-one-view-out、relation/margin/triplet distillation、ECOC、feature erasure、parity / network coding。重点直接阅读或核验：

- MVI²P 论文全文与官方 `sampler / processing / train / loss`；
- UMTS 论文全文，确认 student 图属于 teacher K-shot 输入；
- VKD 官方 README 与训练路径；
- NNCL 摘要、官方 README 与 `NNCL.forward` 的线性编码 / pseudo-inverse 恢复；
- FCFormer、DPM 论文全文；
- PGFL-KD、TSD、PAFormer、PFD、Holistic Guidance、Metric LUPI、RKD、DarkRank 等论文摘要与公开实现入口。

未发现包含 CASD 后三项条件的完整公开先例，不等于绝对不存在；投稿前仍应以 `MVI²P` 为第一相关工作和第一强对照，而不是只强调 PAFormer 或 UMTS。
