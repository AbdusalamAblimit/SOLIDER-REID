# exp371 CASD：2026 未决近邻与强对照审计

日期：2026-07-13

范围：只审计 CASD 相对 `MVI²P / UMTS / PAFormer` 以及 2026 年新近邻的最窄差异，并据此冻结 Gate C 与 frozen-student 的强对照。本文不实现 Gate C，不从标题或搜索摘要推断论文机制。

## 结论先行

CASD 目前只能获得**待全文查新的条件性 GO**，不能写成已经成立的创新。ACM MM 2023 的 `LCR²S` 还进一步证明：`same-ID other-view support set + current sample fusion + full-feature/relation distillation + single-input inference` 已经存在于 person ReID 相邻任务中；support set、跨视图关系蒸馏与单输入 student 都不能单列为新意。

一个此前未列入文档的 2026 年直接近邻必须升为最高风险：

> Jianfeng Dong et al., *Learning from multi-view fragments: An adaptive consistency distillation framework for occluded person re-identification*, Neurocomputing 676 (2026), 133015, DOI: [10.1016/j.neucom.2026.133015](https://doi.org/10.1016/j.neucom.2026.133015).

其题名同时出现 `multi-view fragments`、`consistency distillation` 和 `occluded person re-identification`，与 CASD 的问题和训练形态高度接近；但当前可合法访问的元数据没有摘要或方法正文。因此：

1. 必须把它标为 **unresolved critical prior**；
2. 不能从题名推断它已经覆盖或没有覆盖 strict LOO、pose-response routing、part correspondence、support-vs-self advantage；
3. Gate C 可以继续作为内部机制 kill-switch，但在取得并审计合法全文前，不得给 CASD 最终新颖性 GO，也不得写“尚无同类方法”；
4. 若全文显示它已联合覆盖 strict anchor exclusion、跨图局部 support 与 consistency/advantage distillation，CASD 必须重新裁决，不能靠术语改名继续。

## 一、Dong et al. 2026：可核实证据

### 1.1 DOI / Crossref

[Crossref DOI 记录](https://api.crossref.org/works/10.1016/j.neucom.2026.133015)确认：

- DOI：`10.1016/j.neucom.2026.133015`；
- 期刊：*Neurocomputing*；
- published date：2026-05；
- 作者：Jianfeng Dong、Shengwei Tian、Long Yu、Hongfeng You、Qimeng Yang、Jinmiao Song、Xinjun Pei、Feng Shi、Kun Wu；
- Crossref 记录未提供 abstract。

Crossref 只能证明书目信息，不能证明任何具体方法差异。

### 1.2 出版社元数据

[Elsevier Article Retrieval API](https://api.elsevier.com/content/article/doi/10.1016/j.neucom.2026.133015?httpAccept=text/xml) 返回：

- PII：`S0925231226004121`；
- EID：`1-s2.0-S0925231226004121`；
- cover date：`2026-05-01`；
- open access：`false`；
- 文章页：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231226004121)。

未授权 API 只返回最小元数据，文章页要求人机验证且正文不是开放获取。本文没有绕过 CAPTCHA 或付费访问，也没有把搜索引擎生成摘要当作论文原文。

### 1.3 Semantic Scholar / OpenAlex / ORCID

- [Semantic Scholar 记录](https://www.semanticscholar.org/paper/85ddea08d3f1b09d1e9788bb6894b0ef78d37d43)确认题名、2026、Neurocomputing；检索 API 未返回 abstract 或 arXiv 编号。
- OpenAlex DOI 查询在本轮返回 `429` 配额限制，未取得可引用记录。该失败不等于 OpenAlex 无记录，也不能用于判断开放获取状态。
- 第一作者 [ORCID 0009-0005-2526-834X](https://orcid.org/0009-0005-2526-834X) 的公开记录未列机构、链接或 works。
- 第二作者 [ORCID 0000-0003-3525-5102](https://orcid.org/0000-0003-3525-5102) 的公开 works 列出该题名与 DOI，但没有作者预印本或代码链接。

### 1.4 预印本、机构仓储与代码检索

本轮做了以下只读检索：

1. exact-title 与 `filetype:pdf`：未发现作者预印本或机构仓储 PDF；
2. exact-title + 第一作者：结果集中于 ScienceDirect、ResearchGate、Semantic Scholar 与 DBLP；ResearchGate 仅显示 `Request PDF`，没有公开全文；
3. GitHub repository search：exact title、DOI、PII、`adaptive consistency distillation + ReID` 均未找到对应仓库；
4. 未发现 arXiv ID 或官方代码声明。

搜索不到不构成“不存在”的证明。后续若作者公开预印本、accepted manuscript、机构仓储或代码，应优先审计，不使用非授权论文镜像。

### 1.5 目前明确不能确认的机制

当前不能确认：

- `multi-view fragments` 是同 ID 多张图、单图内多尺度 fragment，还是生成/增强视图；
- teacher 是否包含当前 anchor；
- 是否 strict leave-one-view-out；
- 是否使用 pose、parsing、人体 part 或 target-only heatmap；
- `adaptive consistency` 是 feature/logit/relation consistency，还是可靠性加权；
- 是否比较 support 与 same-image self teacher；
- 是否只迁移正向 relation correction；
- 推理时是否 image-only；
- 是否有 identity-only、permutation、anchor-inclusive 等因果对照。

因此，任何把该论文写成“已经覆盖 CASD”或“显然没有覆盖 CASD”的结论都不成立。

### 1.6 合法全文取得后的强制审计清单

1. 数据单元：multi-view 是否为同 ID 的不同实例；
2. 泄漏边界：teacher/support 是否包含 student 当前图和 relation endpoint；
3. 组织变量：是否按人体 part/pose-response 对齐，还是普通全局/多尺度融合；
4. target：完整 feature、logits、pair relation、margin 还是 consistency correction；
5. 选择规则：是否显式比较 support 与 self，是否有 hindsight/验证集选择；
6. student：测试输入是否只含单张 RGB；
7. 对照：是否包含 anchor-inclusive、strict-LOO、identity-only、part permutation、same-image KD；
8. 代码：若公开，核对论文公式与真实 forward/loss/sampler，而不只读 README。

只有这八项核验后，才能决定 CASD 的联合差分是否仍存在。

## 二、MVI²P：论文与代码精确边界

来源：

- 论文：[arXiv 2311.03828](https://arxiv.org/abs/2311.03828)，[DOI 10.1016/j.inffus.2023.102201](https://doi.org/10.1016/j.inffus.2023.102201)；
- 官方代码：[nengdong96/MVIIP](https://github.com/nengdong96/MVIIP)，本轮审计 commit `4efd9fc920d2b3b5a8e9329059d81a6573f19b13`。

### 2.1 已被 MVI²P 覆盖的内容

论文第 3.1--3.4 节和公式 (2)--(9) 明确覆盖：

1. P×K batch 中同 ID 多图提供互补可见信息；
2. CAM localization 过滤身份无关空间；
3. 正确类别概率量化各视图可靠性；
4. 同 ID feature maps 加权相加为 comprehensive representation；
5. 用 L2 将综合向量传播给每张单图向量；
6. 测试仅使用 baseline 单图分支与 cosine distance。

所以 CASD 不能声称“首次用同 ID 多图补遮挡单图”“首次 multi-view teacher 教单图”“首次可靠性加权跨图传播”。

### 2.2 MVI²P 没有隔离 complementarity

论文公式 (6) 对 `m=1...M` 的所有视图求和，公式 (8) 再将该综合向量蒸馏给第 `m` 张单图。学生当前图属于其 target，本身证据与其他视图新增证据被混合。

官方代码进一步确认：

- `network/processing.py:18-21` 硬编码把每组第 0--3 张 feature map 全部相加；
- `core/train.py:122-143` 用同一个 backbone 得到单图与 integrated target，并以 `ReasoningLoss` 对齐；
- `tools/loss.py:127-132` 将 integrated vector 复制给单图后做 L2 norm；
- CAM 被 detach，但 backbone 与两套 classifier 在 `core/train.py:145-151` 联合更新，因此不是 CASD 要求的 fixed-SHA frozen teacher。

这支持一个可证伪差分：**MVI²P 是 anchor-inclusive full-feature propagation；CASD 必须证明 strict-LOO 的 part-response support 比一个严格复现论文意图的 MVI²P control 更好。**

### 2.3 官方代码复现注意事项

官方实现还有两个不能直接复制进强对照的硬编码问题：

1. `processing.py` 固定每四张整合，而 CLI 默认 `num_instances=8`；
2. `ReasoningLoss` 使用 `range(int(bn_features2.size(0) / 4))` 再把第 `i` 个综合向量写给四张单图，按表面代码只覆盖 integrated batch 的四分之一，剩余 student target 留为零。

这可能是隐藏 batch 假设或实现缺陷。CASD 的 `MVIIP-full` 强对照应实现论文公式的完整 all-student target，并报告 target coverage 与 loss mass；不能复制可疑 bug 后声称超过 MVI²P。

## 三、UMTS 与 PAFormer 的最窄边界

### 3.1 UMTS

[UMTS](https://arxiv.org/abs/2001.05197) 第 3.2--3.4 节明确：

- teacher 输入同 ID 的 K 张图沿通道拼接；
- student 输入是这 K 张中的一张；
- teacher 在多 stage 提供 projected feature target；
- UA-KDL 根据 teacher/student pair 的异方差不确定性调权；
- 测试只使用 single-shot student。

因此 teacher 明确包含 student anchor。CASD 只有在 `strict-LOO + part-response organization + support-vs-self relation correction` 联合成立时才与 UMTS 有方法级差异；单独的 uncertainty、multi-shot、pose-free student 都不新。

### 3.2 PAFormer

[PAFormer](https://arxiv.org/abs/2408.05918) 第 4 节已覆盖：

- pose heatmap 直接监督 pose-token attention；
- pose tokens 聚合同部位 patch；
- learned visibility predictor；
- 由 pose heatmap visibility 做 teacher forcing；
- 测试不再需要 pose heatmap。

但 PAFormer 是 same-image pose supervision，没有同 ID 其他视图 strict-LOO support。它因此不是 CASD 联合机制的完整先例，却是 `KD0 same-image pose teacher` 的必做外部强对照。CASD 不能把 part token、pose-free、visibility predictor 或 part-to-part comparison 列为贡献。

### 3.3 LCR²S：support set 与 relation KD 的直接术语/机制邻居

来源：

- Shuanglin Yan et al., *Learning Comprehensive Representations with Richer Self for Text-to-Image Person Re-Identification*；
- [arXiv 2310.11210](https://arxiv.org/abs/2310.11210)，[ACM MM 2023 DOI 10.1145/3581783.3611832](https://doi.org/10.1145/3581783.3611832)。

论文摘要、第 3.2--3.3 节和公式 (5)--(16) 明确：

1. 对每个 image/text，从同 ID 的其他视图随机选取样本构造 `support set`；
2. MHAF 的输入是 `current sample + support set`，因此 support set 本身排除 self，但 enriched teacher target 仍包含 current sample；
3. teacher 学习 enriched multi-view feature；
4. student 从头训练，只输入单张 image/text；
5. teacher 先训练并冻结，再用 enriched feature MSE 与 batch 内 inter-modal relation matrix Frobenius loss 蒸馏 student；
6. 测试只保留单输入 student。

这比 MVI²P 更直接地撞上 CASD 的术语和训练形态。CASD 不能声称：首次 same-ID support set、首次把 other-view support 教给单输入 student、首次同时做 feature/relation distillation、首次 frozen richer teacher。

LCR²S 仍未覆盖的联合条件是：

- 任务是 text-image ReID，不是 image-only occluded ReID；
- enriched target 显式保留 current sample，未隔离其他视图相对 self 的增量；
- support 随机选择，不按 pose/anatomical part response 路由；
- 完整 feature 与 relation matrix 全量迁移，不比较 support-vs-self gain。

因此 CASD 的差异必须准确写成 **teacher/support target 层面排除 current evidence**，不能只写“support set 排除 anchor”，因为 LCR²S 的 support set 已经这样做。外部强对照还必须包含 `current + other-view support` 的 feature+relation KD，而不能只做 MVI²P 的 feature L2。

论文未声明官方代码，本轮以 exact title、`LCR2S` 与 arXiv ID 检索未找到可核验仓库。后续强对照应依据论文公式实现，并报告支持集大小、teacher freeze、feature/relation loss mass。

## 四、第二个 2026 同组近邻

同一组作者还发表：

> Yue Dong et al., *MHSF: Multi-view hierarchical semantic fusion network for occluded person ReID*, Displays (2026), DOI: [10.1016/j.displa.2026.103424](https://doi.org/10.1016/j.displa.2026.103424), PII `S0141938226000879`。

出版社元数据标记为非开放获取，当前也没有可合法读取的摘要/正文。`multi-view` 可能指多尺度/多层视图，不能从标题推断为同 ID 跨图 support。它列为 secondary unresolved prior，全文可得后按同一八项清单核验，但风险低于题名直接包含 distillation 的 Dong et al. 2026。

### 4.1 2025/2026 补充边界：OGFR 与 PRCV pose/GCN 框架

本轮还补查了两个不会完整覆盖 CASD、但会进一步收紧单项 claim 的近邻。

1. **OGFR**：Yufei Zheng et al., *Occlusion-Guided Feature Purification Learning via Reinforced Knowledge Distillation for Occluded Person Re-Identification*, Journal of Intelligent Computing and Networking 2025, DOI [10.64509/jicn.12.31](https://doi.org/10.64509/jicn.12.31)，[公开全文](https://www.ffspub.com/index.php/jicn/article/download/31/17)。论文第 3 节明确让 teacher 读取未遮挡原图、student 读取该图的遮挡增强，并以完整 feature MSE、global/part cosine 和 KL/ID loss 迁移 holistic knowledge；测试仍用 parsing-derived part visibility 做共同可见部位距离。因此 holistic-to-occluded KD、完整图知识迁移、pose/parsing part token 与 visibility-based matching 都不能成为 CASD 单项贡献。它没有使用同 ID 的其他真实图像，也没有 strict support LOO、support-vs-self 增量隔离或 pose-response donor routing，故不是 CASD 联合机制的直接先例。
2. **Rethinking Pose Guidance**：Zengxi Huang et al., *Rethinking Pose Guidance for Occluded Person Re-identification: A Multi-granularity Feature Learning Framework*, PRCV 2026, DOI [10.1007/978-981-95-5699-1_15](https://doi.org/10.1007/978-981-95-5699-1_15)。出版社公开摘要明确覆盖多 pose cue 手工/学习 attention、pose-guided body-part partition 和改进 GCN joint-semantic features。正文非开放获取，因此不推断摘要外机制；但 pose attention、part partition、GCN joint reasoning 已更不能写成 LGPA 改造的新意。公开摘要未出现同 ID 跨图 support、strict LOO 或 student distillation。

这两篇不会解除 Dong et al. 2026 的未决状态，也不会扩大 CASD 的 claim。它们只进一步确认：最终贡献必须放在**跨图 support 的隔离、组织与增量迁移**，而不是 pose attention、GCN、part token、完整图 KD 或共同可见 matching。

## 五、CASD 的最窄可证伪 claim

在 Dong et al. 2026 全文未审计前，下述只是一条**内部假设**，不可写成最终论文首创：

> 现有 multi-view ReID distillation 将 anchor self-evidence 与其他视图的 complementary evidence 混入同一 teacher。CASD 使用 target-only pose response，从严格排除 anchor 与 relation endpoint 的同 ID 视图中组织 part-wise support，并仅迁移该 support 相对 same-image teacher 在预定义 class-free retrieval relation 上产生的改善。

这条 claim 可被直接推翻：

1. 若 Dong et al. 2026 已覆盖同一联合机制，则外部新颖性失败；
2. 若 strict-LOO full-feature/relation KD 或 LCR²S 式 current+support feature/relation KD 与 CASD 持平，则 part-response/advantage 没有独立价值；
3. 若 `PART-EQUAL / SLOT-PERM / RESPONSE-PERM / ID-MEAN` 与 CASD 持平，则 pose organization 没有独立价值；
4. 若 same-image pose KD 与 CASD 持平，则跨实例 support 没有独立价值；
5. 若只有 anchor-inclusive arm 更高，则结果主要来自 current-view leakage，不能作为 CASD 证据；
6. 若只在 5376-D 而非 matched 768-D 成立，则不能排除 descriptor capacity。

不能声称：首次 multi-view distillation、首次同 ID support set/补全、首次 pose-free pose KD、首次 feature+relation/margin KD、首次 leave-one-out prototype、首次 consistency distillation。

## 六、Gate C 必做 frozen-feature arms

所有 arm 必须复用同一 target-only frozen cache，support/reference PID 内 deterministic K-fold disjoint；每个 query 每个 arm 只能产生一个固定 descriptor。禁止验证集 hindsight `gain>0` mask，禁止按 arm 改 extraction。

| Arm | 固定变量下唯一改变 | 排除的替代解释 |
|---|---|---|
| `SELF` | 当前图 teacher | same-image 上限 |
| `ID-MEAN` | strict-LOO 同 ID descriptor mean | 类中心效应 |
| `ID-GLOBAL` | strict-LOO 同 ID global mean | local block/扩维效应 |
| `PART-EQUAL` | 同 slot donor 等权 | pose-response reliability |
| `SLOT-PERM` | donor feature slot 独立置换 | anatomy/slot correspondence |
| `RESPONSE-PERM` | feature slot 固定，只置换 response 权重 | response routing 本身 |
| `AGREE` | 仅 appearance agreement 选 donor | 非 pose 可靠性即可解释 |
| `FULL-LOO` | strict-LOO 完整 feature aggregate | MVI²P/UMTS 的最强去泄漏版本 |
| `FULL-INCL` | 完整 feature aggregate，包含 anchor | current-view leakage 与论文原型 |
| `WRONG-ID` | 错身份 support，仅安全诊断 | support 身份纯度 |

Gate C 的 pose-specific GO 条件是：correct part-response support 同时超过 `ID-MEAN`、`PART-EQUAL`、`SLOT-PERM`、`RESPONSE-PERM`、`AGREE` 和 `FULL-LOO`。只超过 `SELF` 或 `global` 不够。

## 七、Frozen-student 必做强对照

| Arm | 监督 | 作用 |
|---|---|---|
| `B0` | 无 KD | image-only 架构基线 |
| `KD0` | same-image pose KD | PAFormer/PGFL-KD 类解释 |
| `ID0` | strict-LOO identity-only aggregate | 类中心解释 |
| `P0` | slot/response permutation | part 数量或 ensemble 解释 |
| `C0` | CASD | 主方法 |
| `R0` | 同 target，但 full relation KD、无 advantage selection | advantage 机制必要性 |
| `U0` | `exp120/123`-style 内部强前驱 | 仓库内部新意 |
| `MV-INCL` | anchor-inclusive full-feature KD | MVI²P/UMTS 论文原型 |
| `LR-INCL` | current + other-view support，full-feature + relation KD | LCR²S 论文原型 |
| `MV-LOO` | strict-LOO full-feature + relation KD | 普通跨图 KD 的最强去泄漏版本 |
| `LEAK` | current-view support | 只量化泄漏虚高，不计主结果 |
| `D26` | 待 Dong et al. 2026 全文后按原方法复现 | 2026 直接近邻；机制未知前不得臆造 |

所有训练臂必须同初始化、同 runtime、同 descriptor 维度、同优化步数，并报告：有效 loss coverage、effective sample size、每样本 loss mass、梯度预算和无 support 比例。

最低结果门禁继续使用：

1. `C0-B0 >= +0.8 mAP`；
2. `C0-strongest_control >= +0.5 mAP`；
3. 至少恢复旧 LGPA `+0.9 mAP` 的 80%，即约 `+0.72 mAP`；
4. 三 seed paired mean 同向，不能由单 seed 驱动；
5. 测试 descriptor 对 pose 输入逐元素不变；
6. 5376-D 与 768-D 分开公平比较；
7. Dong et al. 2026 全文审计未完成时，即使数值过门也只能记为“机制可行、外部新颖性未决”。

## 最终裁决

`MVI²P / UMTS / LCR²S / PAFormer` 仍给 CASD 留下一条很窄但可实验裁决的联合差分：**strict LOO complementarity isolation + target-only part-response organization + support-vs-self class-free relation correction**。然而 Dong et al. 2026 的出现使这条差分在外部新颖性上暂时不可确认。

合理顺序是：继续 Gate C 廉价 kill-switch，同时寻找合法全文；Gate C 失败则直接 NO-GO，不需要再查论文救机制。Gate C 通过也不能直接进入“已创新”结论，必须先完成 Dong et al. 2026 的八项全文审计与 `D26` 强对照设计。
