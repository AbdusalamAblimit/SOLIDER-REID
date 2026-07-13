# exp371 CASD 严格红队审查

## 审查结论

**当前裁决：Phase 0 的 LGPA 归因与描述子诊断可以继续，但 Gate C 与 Phase 1 暂不具备启动条件。**

CASD 当前不是因为“实现还没写完”而被暂缓，而是存在四个会直接改变论文结论的识别漏洞：

1. 它与仓库内 `exp084/109/110-116/119/120` 的差异尚未被正面定义，当前新颖性审计过度集中于 UMTS；
2. `exp336` 使用的是场景内所有人的 max-merged heatmap，不是目标行人的专属 heatmap，“identity anatomical support”这一名称目前没有输入层事实支撑；
3. same-ID 聚合天然带来类中心降噪，现有 `KD0/Cx/UM0` 不能排除收益只是 label-conditioned prototype regularization；
4. `adv=relu(margin_support-margin_self)` 同时决定“给谁监督”和“监督多强”，会让 correct、shuffled、uniform、wrong-ID 各臂拥有不同的有效 loss 数量。尤其 wrong-ID 往往直接变成零 loss，不能作为公平反事实。

在这些问题解决前，即使 C0 涨点，也不能归因于“pose-organized complementary anatomical support advantage”。最多只能说一个同 ID 多图蒸馏变体有效。

## 一、最高风险不是 UMTS，而是仓库内已有前驱

现有 `literature_novelty.md` 正确识别了 UMTS，但遗漏了比 UMTS 更接近当前实现的本仓库证据链：

| 前驱 | 已覆盖内容 | 与 CASD 的直接重合 |
|---|---|---|
| `exp084 CIPGFR` | batch 内同 ID、可见部位到遮挡部位、detached cross-instance target | 已经提出“其他同 ID 图像的可见局部监督当前图缺失局部” |
| `exp109 Oracle Support Bank` | GT same-ID、per-keypoint、visibility-aware、leave-one-out prototype | 已经覆盖 CASD 的问题定义和 support 构造上界 |
| `exp110-116 SCKD/SCFR` | per-ID/per-keypoint support teacher、可见性门控、蒸馏或直接替换 | 已经测试 pointwise support-complete supervision，并确认 mAP 天花板约 `+0.0~0.2` |
| `exp119 CSRD` | detached part/common-support teacher 到 student 的 relation distillation | 已经覆盖“不要做 feature MSE，蒸馏 identity relation” |
| `exp120 SCRD` | same-ID support-complete bank 增强 relational teacher，不直接蒸馏点特征 | 与 CASD 当前 headline 最接近；到 e90 比 `exp119` 低 `0.2 mAP` 后停止 |

CASD 相对这条内部链条目前只剩三个候选差异：

1. 用 LGPA 的 5 个 part 代替 GCN 的 17 个 keypoint；
2. 用 batch-local leave-one-view-out support 代替 EMA identity bank；
3. 只蒸馏 support 相对 self 的正 margin advantage。

前两项单独都属于 extractor/support source 替换；第三项又与 residual/exclusive knowledge distillation 邻域接近。因此，`candidate_matrix.md` 中 CASD 的机制新意 `4/5` 目前没有证据，红队建议在通过下述差分门禁前降为 **条件性 2/5**。

### 一票否决门禁 N：内部前驱差分

在任何 CASD 训练前，必须补一张“CASD 对 exp084/109/110/119/120 的逐项差分表”，并把 Phase 1 的直接对照扩为：

- `OLD-R`：复现 `exp120` 的 support-complete relational teacher，换到与 CASD 相同的 frozen LGPA teacher/student/runtime；
- `CASD-A`：在 `OLD-R` 上只加入 advantage selection；
- `CASD-L`：在 `CASD-A` 上只把 EMA/full-ID bank 改为 batch-local strict leave-one-out；
- `CASD-P`：在 `CASD-L` 上才加入 pose-based part correspondence/visibility。

只有 `CASD-P` 相对最强内部前驱仍有稳定独立增益，才能把 CASD 写成新机制。若增益只来自 `CASD-A`，论文贡献应降为 advantage-filtered relational KD；若只来自 LGPA extractor，贡献属于更强 teacher，不属于新的 support 学习范式。

## 二、“anatomical identity support”的输入事实尚不成立

`configs/occluded_duke/exp336_swin_lgpa_nopsg.yml` 明确没有设置 `POSE_USE_TARGET_HEATMAP`，注释也写明使用 scene-merged heatmap。`pose_backbone_model.py::_prepare_pose` 先对所有检测到的人做 max-merge；`_lgpa_heatmap` 默认把该 scene heatmap 交给 LGPA。

因此当前所谓 correct pose 实际是：

> 当前场景中目标人物、遮挡者和旁人的联合人体响应。

这会造成两层问题：

1. teacher part 可能读到旁人或遮挡者的局部外观，不能直接称为目标身份的 anatomical evidence；
2. same-ID support 聚合后，模型可能只是平均掉不同场景干扰，而不是补全目标人物缺失部位。

### 一票否决门禁 T：目标人物归属

同一 frozen checkpoint/同一缓存 spatial feature 至少比较：

1. `scene-merged`：exp336 原口径；
2. `target-only`：`target_person_idx` 对应人物；
3. `distractor-only`：排除 target 后其余人物的 max-merge；
4. `canonical`：固定人体布局。

门禁不是要求 target-only 必须复现 exp336 的全部绝对指标，而是要求：

- target-only support 在 identity-only、slot-shuffle 等公平控制下仍有独立 advantage；
- distractor-only 不能得到相同 support advantage；
- 日志必须给出无 target、错误 target、多人物样本的占比。

若 scene-merged/distractor-only 与 target-only 等价或更好，应立即删除 **anatomical identity support / complementary visible body evidence** 的表述，改成更弱的 scene-structured local support；这会显著削弱论文创新性，需重新裁决是否继续。

## 三、leave-one-view-out 目前只排除了 batch 索引，没有排除图像泄漏

`RandomIdentitySampler` 在一个身份样本数少于 `NUM_INSTANCE=4` 时会有放回采样。即使 `j != i`，support 仍可能是同一路径的另一份增强。因此“排除 anchor 索引”不等于“排除 current view”。此外，连续帧/近重复图像也可能使 support 成为近似拷贝。

### 一票否决门禁 L：严格 view exclusion

support 构造必须逐 anchor 断言并落盘：

- `support_path != anchor_path`；
- 同 batch 同 PID 的 path 必须去重，不能用有放回副本补齐；
- 若有 tracklet/frame 信息，排除同 tracklet 近邻帧；若没有，至少报告图像 hash 与近重复感知相似度审计；
- 报告可用 support 数量分布、无 support 比例，而不是用重复图维持固定 K；
- 增加 `cross-camera only` 对照，区分真正跨视角互补与同摄像机近重复降噪。

若严格排除后大量 anchor 无 support，或收益只存在于 same-camera/near-duplicate support，CASD 的“跨视图互补”主张判负。

## 四、same-ID 平均是最强混淆变量，当前对照没有排除

同 ID 的三个向量取平均，即使没有任何 pose 或部位语义，也会降低实例噪声并靠近训练类中心。用训练 ID classifier 计算 `q_support` 时，这种优势几乎是构造出来的：classifier 本来就用这些训练身份和局部特征训练，same-ID average 更高的 true-class margin 不能证明 anatomical completion。

当前 `uniform/shuffled/wrong-ID` 仍不足够：

- uniform/shuffled 同时改变 teacher feature extraction 和 support routing，无法确定下降来自单图 teacher 变差还是跨图组织变差；
- wrong-ID 改变了身份内容，不是“有无 pose organization”的单变量；
- 缺少 **同一 correct teacher features 上的 identity-only aggregate**。

### 一票否决门禁 I：identity-prototype 混淆

Gate C 必须固定 `t_j,k` 为同一批 correct/target-only teacher features，只干预 support 组装，至少比较：

1. `SELF`：same-image teacher；
2. `ID-MEAN`：同 ID 其他图的全部 local blocks 等权平均，不用 visibility，不要求 part correspondence；
3. `ID-GLOBAL`：同 ID 其他图的 global feature/prototype teacher，参数与 loss 预算匹配；
4. `PART-EQUAL`：同一 correct part features，保留 slot 索引但所有 donor/part 等权；
5. `PART-PERM`：同一 correct part features，对每个 donor 独立打乱 slot 对应；
6. `CASD`：同一 correct part features，使用正确 correspondence 与绝对 visibility；
7. `WRONG-ID`：只作为污染/安全性诊断，不作为主要公平 control。

只有 `CASD > max(ID-MEAN, ID-GLOBAL, PART-EQUAL, PART-PERM)`，才能把增益归于 anatomical organization。若 CASD 只优于 SELF/wrong-ID，结论只是“同 ID prototype 有用”。

## 五、必须把 teacher extraction 与 support routing 做成正交干预

现有 Gate B 的 correct/canonical/shuffled/uniform/no-pose 改变的是 LGPA **单图特征提取**。它可以回答 checkpoint 在测试时依赖什么 pose，却不能证明 pose 在跨图 support 中起组织作用。

Gate C 若沿用这些臂重新提取 `t_j,k`，会同时改变：

- teacher feature 内容；
- part slot 含义；
- visibility；
- support donor 权重；
- advantage mask 数量。

这是典型的多重处理，不能叫因果门禁。

### 最小 2×3 因子设计

- 提取因子 `E`：`target-correct` / `canonical`；
- 组织因子 `R`：`correct correspondence+visibility` / `equal` / `independent slot permutation`。

所有 `R` 臂必须复用同一份 `E` 缓存 tensor，不能重新跑 LGPA。论文中只有 `R` 的差值能支撑“pose organizes support”；`E` 的差值只说明 pose 改善 teacher extractor。

Gate B 可继续执行，但只能作为 LGPA attribution，不能作为 CASD 因果证据。

## 六、advantage selector 存在标签回看与 loss-budget 不公平

当前设计：

```text
adv = relu(margin(q_support)-margin(q_self))
L = adv * KL(q_support || p_student)
```

这里的 margin 依赖真实训练 ID 和已经在这些 ID 上训练的 part classifier。它既筛选样本，又决定权重。因此：

1. Gate C 的 teacher ID CE/accuracy 是训练身份上的回看指标，容易自证；
2. correct arm 若产生更多正 adv，就会获得更多 KD 梯度；
3. wrong-ID 的 adv 大多为零，实际上退化为“少加或不加 loss”；
4. C0 优于 Cx 可能只说明 C0 的有效 loss 数量更多，而不是 target 更正确；
5. `relu` 会忽略 support 变差样本，主方法只挑有利案例，但对照也许不能获得同一 oracle mask。

### 一票否决门禁 A：选择器与监督内容解耦

必须同时报告两种协议：

1. **shared-mask protocol**：由 correct arm 生成 anchor/part mask 和权重，所有 control 使用完全相同的 mask、总权重和 loss 数量，只替换 teacher target；
2. **own-mask protocol**：各臂使用自己的 advantage，用于报告 coverage/安全性，但不能单独用于主因果结论。

另外至少加入：

- `COUNT-MATCHED SELF-KD`：从 self teacher 中采样相同数量、相同权重分布的 KD 项；
- `CONF-MATCHED ID-MEAN`：普通 same-ID aggregate 具有相同 effective loss mass；
- 每臂记录正 adv 比例、每 part/每 PID 覆盖、总权重、最大样本权重、有效 batch 比例。

Gate C 主指标不应使用训练 classifier CE。优先使用 class-free episodic retrieval margin，并在不参与 teacher/head 训练的身份或严格隔离的 support/query/gallery episode 上计算。若只能在原训练 ID classifier margin 上成立，support advantage 的可迁移性判负。

## 七、visibility 不是当前实现中的绝对可见性

`CLIPPartHead` 先计算每 part 的 heatmap response，随后除以五个 part 的总和。这个 `kp_weights` 是 **图内相对分配**，不是绝对 visibility：

- 全身 pose 都很弱时，仍可能有某个 part 获得很高相对权重；
- pose estimator 对遮挡关节可能产生 hallucinated heatmap；
- `m_i,k` 的 threshold 若作用在归一化权重上，不等于“该部位可见”。

CASD 需要单独缓存未归一化的 raw part response、target person detection confidence、有效像素面积，并证明其与遮挡/可见性至少有合理相关。若无法验证，应把 `visibility` 改称 `pose response/reliability`，不得写“其他视图中可见的缺失部位”。

## 八、detached teacher 不等于 frozen teacher

设计只写“teacher 输入 backbone features detach”，但没有冻结：

- teacher backbone/head/classifier 是否固定；
- BN 是否处于 eval；
- teacher 是否与 student 共享持续变化的 backbone features；
- `q_self/q_support` 的 classifier 是 exp336 已训练 classifier，还是随 CASD 联合更新。

如果 teacher target 随 student backbone/head 同步变化，`adv` 也在变化，C0 与 controls 的比较会混入 teacher dynamics。Phase 1 首验必须使用完全 frozen、eval-mode、固定 SHA 的 teacher checkpoint；student 单独初始化。任何可训练 teacher/共享移动 backbone 只能作为后续独立变量。

## 九、现有 Phase 0 各门的正确证据边界

### Gate A

canonical 条件下 fixed-random 已高于 CLIP，足以移除 CLIP 语义 claim。correct-pose learned query 只补可学习性归因，不是 CASD 的 GO 条件，也不应优先占用训练资源。

### Gate B

五臂、global SHA、stock parity 的工程约束是合理的。但它回答的是“LGPA descriptor 对测试 pose 的响应”，不能回答“pose 是否改善跨实例 support organization”。

### Gate C

当前定义必须按本文 I/A/T/L 四个门禁重写。尤其不能把训练 ID classifier accuracy 作为主判据。

### Gate D

train-only PCA/JL 的泄漏控制是合理的，但它压缩的是旧 LGPA teacher descriptor，不是未来 CASD student descriptor。Gate D 只回答旧资产是否可简单 packing，不能支撑 CASD 的因果或最终成本 claim。最终 CASD 仍需在其自身三 seed descriptor 上重新过同维门禁。

## 十、Phase 1 最小公平对照

为避免臂数失控，红队建议第一轮只保留下面七臂，全部使用相同 frozen spatial caches、student 初始化、优化步数、descriptor 维度与 loss mass：

| Arm | 监督 | 主要回答 |
|---|---|---|
| B0 | image-only student，无 KD | 架构基线 |
| KD0 | same-image pose teacher，count/weight matched | 普通 pose KD |
| ID0 | strict-LOO same-ID aggregate，无 visibility/part correspondence | 标签类中心效应 |
| P0 | correct teacher parts + independent slot permutation | part 数量/ensemble 效应 |
| C0 | strict-LOO correct routing + shared advantage mask | CASD 主方法 |
| R0 | C0 target，但普通 full relation KD、无 advantage selection | advantage 是否必要 |
| U0 | exp120-style support-complete relational teacher | 相对内部最强前驱的新意 |

`wrong-ID`、current-view leakage、scene/distractor pose 放在 oracle/安全性诊断，不必都变成完整训练臂。

### GO 条件

1. 先通过 target attribution、strict view exclusion 和 class-free support oracle；
2. `C0 - max(KD0, ID0, P0, R0, U0) >= 0.5 mAP`；
3. `C0 - B0 >= 0.8 mAP` 只能作为总效果门，不能替代第 2 条；
4. loss-mass matched 后优势仍成立；
5. 完整训练至少三 seed paired mean 为正，且置信区间/逐 seed 不由单个 seed 驱动；
6. 分别报告 `B0-global`、`KD0-B0`、`C0-KD0`、`C0-strongest-control`、`C0-global`，不能只报告总涨点；
7. 所有最终方法与基线使用相同 5376-D 或相同 768-D，不能用 CASD 5376-D 对 global 768-D 声称方法增益。

若第 2 条失败，CASD 不能作为主创新；若只有总 `C0-global` 为正，应把收益归于 image-only local router/描述子扩维，而不是 support advantage。

## 十一、若全部通过，贡献应如何精准表述

建议最高强度表述：

> 我们研究训练期同身份多视图如何为单图遮挡 ReID 提供额外局部监督。方法使用冻结的、目标人物专属 pose-aware extractor，从严格排除当前图的同 ID 视图中构造 part-wise support，并仅蒸馏该 support 相对同图 teacher 改善的检索关系。测试时 student 仅输入单张 RGB。通过固定 teacher features、只随机化 support correspondence/visibility 的受控实验，以及 same-image KD、identity-only aggregate、完整 multi-shot KD 和既有 support-complete relational teacher 对照，我们验证收益来自 pose-organized cross-instance support，而不是类中心平均、当前图泄漏或 loss 数量差异。

只有 target-only 与 slot correspondence 门禁通过，才能使用 `anatomical`；否则改成 `part-structured`。只有真正做了固定特征、共享 mask、单变量随机化，才能说“受控干预支持该解释”，仍不建议写“证明了因果”。

无论结果多好，都不能声称：

- 首次 multi-shot teacher → single-shot student；
- 首次 same-ID support completion；
- 首次 pose teacher/student 或 pose-free ReID；
- 首次 relational/exclusive/residual distillation；
- 首次 leave-one-out identity prototype；
- CLIP、GCN、matching 是贡献；
- 单个 Occluded-Duke/Swin-Tiny 实验已经证明通用性。

## 最终红队裁决

CASD 仍值得做 **重新设计后的廉价 kill-switch**，因为它至少正面连接了 LGPA 稳定局部增益与 `exp109` 的 support headroom。但当前版本把“强问题证据”误当成了“新方法证据”，并且没有排除同 ID 类中心、多人 scene pose、advantage loss 数量和仓库内部前驱四个替代解释。

下一步不是直接实现完整 CASD，而是：

1. 先补内部前驱差分与 target-only 审计；
2. 重写 Gate C 为 frozen-feature、strict-view、class-free、loss-matched 的 support oracle；
3. 只有 CASD 同时优于 identity-only aggregate、slot permutation 和 exp120-style teacher，才进入 Phase 1。

这三步任一失败，主线应正式 NO-GO，不再用 queue、温度、slot 数、OT/MoE 或更复杂 student 结构救场。
