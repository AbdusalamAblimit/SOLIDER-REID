# exp403之后：final-retrieval ownership查新记录

> 日期：2026-07-20
> 状态：`AUDIT ACTIVE / NO CANDIDATE CLEARS INNOVATION GATE / GPU NO-START`
> 目的：exp403已证明“可执行的evidence-conditioned operator”仍可在最终检索中被绕过。下一对象必须直接
> 约束最终身份排序，且不能退化为ELO-CUR换名、普通ranking loss、额外stage或retrieval-side小技巧。

## 1. 当前硬问题

exp403七臂的descriptor干预全部active，但R1/R5/R10逐项完全相同，mAP总跨度不足`0.0018 point`。
因此下一机制不能再把“descriptor变了”“operator有梯度”或训练期proxy margin当作ownership证据。最低要求是：

1. correct、matched wrong-RGB、generic、NULL和all-bypass都在设计中保留；
2. ownership目标直接作用于最终归一化检索对象或其不可绕过的组成部分；
3. shared visual trunk不能通过同步移动correct与reference来满足代理目标；
4. 不通过主动破坏wrong control制造优势；wrong evidence必须有独立、可解释的正目标；
5. 机制至少满足问题/机制/证据创新门槛中的两项，再建立新实验编号。

## 2. 近邻代码审计一：SPT（AAAI 2024）

- 论文：*Occluded Person Re-identification via Saliency-Guided Patch Transfer*
- 论文页：<https://doi.org/10.1609/aaai.v38i5.28312>
- 官方代码：<https://github.com/stone96123/SPT>
- 审计commit：`ef1e71a99bc658790d5dbbc9ab133588e849e814`

代码事实：

- `TransReID_Mask.forward_features`从第1/3/9/final transformer block拼接特征，以detach后的`mixfc`预测
  patch saliency mask；
- 第一阶段用原mask、mask与反mask三次forward，训练ID/Triplet、mask usage和两路分类正交；
- 第二阶段把二值saliency mask用于batch内patch transfer，再以普通ReID目标训练；
- 正式流程明确是先训练SPS、再训练ReID的两阶段方案。

裁决：SPT解决“哪些patch可迁移以模拟遮挡”，并不要求外部sample evidence拥有最终检索对象。它没有
matched wrong/generic/NULL/all-bypass完整执行，也没有防shared trunk绕过的所有权合同。把SPT移植为额外
augmentation或stage既不回应exp403，也违反当前“不增加stage救旧路线”的边界，故不作为下一主机制。

## 3. 近邻代码审计二：ProFD（ACM MM 2024）

- 论文/代码：*Prompt-guided Feature Disentangling for Occluded Person Re-Identification*
- 官方代码：<https://github.com/Cuixxx/ProFD>
- 审计commit：`14e47d3b04f541d2a614482848bba2071bc90cda`

代码事实：

- 训练依赖PifPaf与Mask-RCNN生成的人体解析mask；
- `PartFeatureDecoder`用prompt/part token和visual memory做双向cross-attention，生成显式part embedding；
- global、foreground、parts与concatenated parts分别接ID/metric目标，并加入pixel parsing、visibility、
  prototype memory与dissimilar loss；
- 测试通过query/gallery双方的part visibility组合pairwise distance，而非单一固定global descriptor。

裁决：ProFD证明“显式part slot + visibility-aware pairwise metric”已有完整强先例。它也提示一个真实结构方向：
让语义槽本身进入检索距离，比在global descriptor内部做小残差更难被绕过。但直接复刻会落入已有part-based
ReID与retrieval-side visibility融合，并需要当前冻结边界外的解析资产；不足以作为新贡献。

## 4. 近邻代码审计三：CHAIR（IJCAI 2024）

- 论文：*Are They the Same Picture? Adapting Concept Bottleneck Models for Human-AI Collaboration in Image Retrieval*
- arXiv：<https://arxiv.org/abs/2407.08908>
- 官方代码：<https://github.com/realize-lab/CHAIR>
- 审计commit：`8efe5a6e3f1369e558481e488645c0afbe9fb341`

代码事实：

- `FuseCBM.get_fused_embedding`先从backbone embedding预测concept，再执行
  `fused = x + fuse_layer(concepts)`；这与已失败路线一样，允许原始visual embedding绕过concept edit；
- `get_fused_embedding_with_intervention`用人工修正的concept重算同一加法edit；
- `chair_retrieval.py`和`intervene.py`都对fused embedding做`F.normalize(..., dim=1)`，直接以它执行
  Recall@1/5/10；概念干预确实落在最终归一化检索向量，而不是只用于解释分类头。

裁决：`concept correction -> additive embedding edit -> normalized retrieval`已有直接强先例。CHAIR没有
matched wrong-RGB/generic/NULL/all-bypass合同，也没有切断原始embedding的bypass；但它足以排除“把概念
修正直接加到最终检索向量”作为新机制。这个结构与exp402/403暴露的shortcut同型，不能再换名重开。

## 5. 近邻代码审计四：Minimal Concept Bottleneck Models（ICLR 2026）

- 论文：*There Was Never a Bottleneck in Concept Bottleneck Models*
- arXiv：<https://arxiv.org/abs/2506.04877v3>
- 官方代码：<https://github.com/antonioalmudevar/minimal_cbm>
- 审计commit：`9ba535c8d8e4a5b54e801a31d9db3d819d0910ab`

论文与代码事实：

- 论文精确指出：`z_j`能预测`c_j`不等于`z_j`只编码`c_j`，传统CBM/CEM可保留nuisance并使概念干预失真；
- MCBM以每个概念表示的minimal sufficient statistic为目标，通过variational information-bottleneck
  regularization压缩`I(Z_j; X | C_j)`；
- 当前公开`src/models/mcbm.py`将该约束具体化为每个`z_j`向仅由`c_j`确定的固定logit目标回归，
  `total = task + beta * concept + gamma * representation`，实现中representation项是逐概念MSE；
- 论文也明确承认：概念不完备且任务需要nuisance时，清除这些信息必然降低任务性能。

裁决：exp402/403的“proxy active但final ownership失败”与该信息泄漏诊断高度一致。然而，单独给当前
evidence code增加IB/MSE/minimality loss，只是把公开MCBM移植到ReID，而且很可能牺牲身份充分性；它最多
满足问题/证据门，机制门不足。`terminal subspace + minimality`目前只能作为待审原子，不能据此建立exp404。

## 6. 近邻代码审计五：IntCEM（NeurIPS 2023）

- 论文：*Intervention-aware Concept Embedding Models*
- arXiv：<https://arxiv.org/abs/2309.16928>
- 官方代码：<https://github.com/mateoespinosa/cem>
- 审计commit：`d67716cf8435961b41087ec884f92c74e475d890`

代码事实：

- `IntAwareConceptBottleneckModel`在训练中采样多步intervention trajectory，并逐步更新已干预concept mask；
- 对每个候选concept，它用干预后的task logit为当前真实label计算oracle target，再训练
  `concept_rank_model`预测下一次应干预哪个concept；
- 轨迹内部显式重算intervention后的task loss；代码注释还专门覆盖“用户此前提供的干预可能错误”的情形；
- 最终总目标同时包含concept loss、intervention-policy imitation loss和intervention后的task loss。

裁决：训练期暴露于干预、学习干预策略、用干预后task loss优化可干预性均已有完整先例。因此
“把wrong/NULL分支放进训练”或“在干预后再算ID loss”不能单独构成创新。IntCEM仍以分类/人类干预为对象，
没有标准欧氏instance retrieval、matched donor正目标或all-bypass终审；这些是尚存问题边界，不是现成机制。

## 7. 近邻代码审计六：PDiscoNet（ICCV 2023）

- 论文：*PDiscoNet: Semantically Consistent Part Discovery for Fine-grained Recognition*
- 官方代码：<https://github.com/robertdvdk/part_detection>
- 审计commit：`eec53f2f40602113f74c6c1f60a2034823b0fcaf`

代码事实：

- `IndividualLandmarkNet`从ResNet layer3/4拼接feature，通过prototype-distance map得到`K+1`个softmax
  part maps，再按map加权池化每个part feature；
- 分类logit来自每个part的线性头并在part维求均值；训练联合classification、concentration、presence、
  equivariance和orthogonality五项loss；
- 它学习无关键点监督的语义一致part，但没有外部sample evidence、概念干预或检索ownership目标。

裁决：unsupervised part discovery可以生成更结构化的终端槽，但不能解决“正确外部evidence是否拥有最终排序”。
把PDisco part slot接到当前route仍是常见part module拼接，且不会自动阻断shared visual trunk的shortcut，排除。

## 8. 最新硬/终端瓶颈边界：SupCBM、MM-CBM与CaBM

### 8.1 SupCBM（2024）

*Eliminating Information Leakage in Hard Concept Bottleneck Models with Supervised, Hierarchical Concept Learning*
（<https://arxiv.org/abs/2402.05945>）已经说明二值hard concept仍会借无关concept携带类别信息。其SupCBM
用层级concept set、label-supervised concept prediction和稀疏intervention matrix取代普通label predictor。
所以“硬化/离散化concept即可无泄漏”不成立，hard bottleneck本身也不是新机制。

### 8.2 MM-CBM（2026-06）

- 论文：*Multimodal Concept Bottleneck Models*（<https://arxiv.org/abs/2606.19882>）
- 官方代码：<https://github.com/Trustworthy-ML-Lab/Multi-Modal-CBM>
- 审计commit：`7add51b96c99f7bb8e7d55c3bfbbc2c8137122b8`

MM-CBM使用image/text双Concept Bottleneck Layer，把两种CLIP embedding投到同一个非负concept space；
`cbm_MM.py`在归一化后的concept response上直接计算top-k elementwise product similarity和contrastive/class
loss。论文明确主张预测“solely on concept responses”，并给出image-text retrieval case study。这是
`terminal concept-only metric`的最新直接先例；其任务不是image-image person ReID，也没有matched wrong
evidence，但已足够排除“最终距离只看concept vector”本身的新颖性。

### 8.3 Caption Bottleneck Models（2026-07）

*Caption Bottleneck Models*（<https://arxiv.org/abs/2607.00578>，代码：
<https://github.com/bariscagliyan/CaptionBottleneckModels>）让冻结LMM先生成attribute-centric captions，
下游分类器只读离散文本且不看pixel，作者据此称为leakage-free by construction。它进一步占用了“通过严格
隔离上下游、让终端任务只读语义通道来阻断bypass”的结构叙事。但它依赖测试时LMM/caption，不满足当前
teacher-free RGB-only部署合同，也不解决标准欧氏instance retrieval。

裁决：`hard/terminal bottleneck`、`concept-only metric`、`strictly isolated semantic channel`均已有近期
强先例。把它们与MCBM minimality或IntCEM intervention loss简单组合，仍是CBM原子的工程拼装，不满足当前
机制创新门。

## 9. 辅助近邻与边界

- DSANet/FDL（<https://arxiv.org/abs/2212.09498>）交换identity/camera factor做辅助分类，测试只取identity
  factor；它研究factor disentanglement，不建立sample evidence ownership；
- ISR（<https://arxiv.org/abs/2308.08887>）用可靠性引导正样本匹配，不执行外部evidence intervention；
- TokenMatcher（AAAI 2025，<https://doi.org/10.1609/aaai.v39i8.32855>）做multi-token matching与cluster
  memory，不证明correct/wrong/NULL evidence的因果顺序。

这些工作没有给出当前问题的同构答案，但也不能被当作“加一个disentanglement、reliability或token匹配模块”
的创新许可。

## 10. 当前筛选结论

本轮累计排除以下看似相关但不合格的方向：

1. `saliency mask + patch transfer + extra stage`：问题对象仍是增强，不是ownership；
2. `part decoder + visibility-weighted pairwise distance`：已有强先例，且回到被长期探索的retrieval-side路线。
3. `concept correction + additive final embedding edit`：CHAIR已直接覆盖，且保留原始embedding bypass；
4. `training-time intervention + post-intervention task loss`：IntCEM已系统覆盖；
5. `IB/minimality regularizer`：MCBM已精确定义并开源，单独移植没有机制新意；
6. `hard/terminal concept-only bottleneck`：SupCBM、MM-CBM和CaBM已覆盖三种强形式；
7. `unsupervised part slot + existing route`：PDiscoNet只提供part discovery，不提供ownership。

目前没有候选通过创新门槛，故不建立exp404、不写formal config、不运行CPU/CUDA/GPU。特别地，
`terminal concept-only subspace + minimality + intervention-aware loss`只是三项公开原子的组合，不能因为更换
loss或拼成direct-sum descriptor就升级为新机制。

下一轮查新继续收紧到：

- 对wrong evidence存在独立正目标、而非单纯被推远的counterfactual transport/equivariance；
- 结构上能证明最终标准欧氏identity descriptor对evidence path-complete，而不是靠固定norm quota宣称ownership；
- wrong donor既不能被主动破坏，也不能被当作current identity negative后丢失其自身正目标；
- student推理仍为单RGB、单descriptor、无teacher/text/pose和pair-specific scorer。

这些只是检索条件，不是已授权机制。现阶段最多确认了一个尚未被上述工作直接回答的**问题/证据缺口**：
在open-set instance retrieval中，用matched donor、generic、NULL和all-bypass同时验证sample evidence的路径
所有权。机制层仍为空，必须继续审计source-attributed representation、interventional path completeness与
conditional metric公开实现，确认不是普通CBM/IB/ranking/concat后，才能决定是否形成新编号。

## 11. 同型问题审计：multimodal modality laziness

exp403的“生产route active，但正确evidence没有最终ranking ownership”与多模态学习中的modality laziness
高度同型。为避免把已有的模态平衡方法换成ReID术语，本轮进一步审计了独立训练、latent permutation、
data remix、residual sensing、cue preservation以及直接面向evidence-use的近期工作。

### 11.1 UniCat（arXiv 2023）

- 论文：*UniCat: Crafting a Stronger Fusion Baseline for Multimodal Re-Identification*
  （<https://arxiv.org/abs/2310.18812>）；
- 截至本轮审计，论文与作者页面没有给出可核验的官方代码仓库，故只按论文v1方法定义裁决。

方法事实：

- 每个模态使用不共享backbone `f_i`，分别输出`z_i`；
- ordinary fusion-concat/avg在融合后的`z_fuse`上联合计算triplet与CE，而UniCat为每个`z_i`独立计算完整
  ReID loss；
- 训练完成后才把各模态embedding拼成最终检索descriptor，且测试时需要所有模态。

裁决：UniCat准确指出joint objective会让单模态目标松弛，并给出了multimodal ReID中的直接证据。但其解法是
“独立训练后固定拼接”，不是训练期privileged evidence、测试期单RGB的路径所有权。若把当前evidence分支做成
固定direct-sum block，仍只是在数值上保证一个子空间存在，不能保证correct优于wrong/generic/NULL；该方向已被
本轮terminal bottleneck审计和exp403结果共同限制。

### 11.2 MCR（NeurIPS 2025 Spotlight）

- 论文：*Balancing Multimodal Training Through Game-Theoretic Regularization*
  （<https://arxiv.org/abs/2411.07335>）；
- 官方代码：<https://github.com/kkontras/MCR>；审计commit：
  `0da29d0343e1f4b0b6b90425ff9ff2f71d873b54`。

代码事实：

- `MCR_Linear`的最终分类logit仍是`fc_0(a)+fc_1(v)+bias`；
- 训练时对batch内`a/v`做多次随机置换，并构造另一模态detach或双方可训练的组合；
- regularizer用置换前后预测分布的JSD近似MI分解，再以`greedy/ind/collaborative`三种game组合梯度；
- release config同时保留combined与两个unimodal CE；standalone实现还可叠加contrastive和reconstruction项。

裁决：MCR已经覆盖“latent permutation + MI surrogate + game-theoretic modality contribution regularizer”。
但置换donor只作为扰动，不保留其自身正目标；最终仍是可由任一分支主导的加性分类logit，没有标准检索descriptor
上的matched wrong/generic/NULL/all-bypass终审。因此它是重要问题近邻，却不能提供当前缺失的path-complete
ownership机制；把CUR换成MCR/JSD只是换loss。

### 11.3 Data Remixing（ICML 2025）

- 论文：*Improving Multimodal Learning Balance and Sufficiency through Data Remixing*
  （<https://arxiv.org/abs/2506.11550>）；
- 官方代码：<https://github.com/MatthewMaxy/Remix_ICML2025>；审计commit：
  `80898aa0ad8077af535e7ade4b4583525592956f`。

代码事实：先warm-up联合训练，再根据两个unimodal posterior到均匀分布的KL大小，把样本写入audio/video子集；
后续分别加载子集，在`AVClassifier.forward`中把非目标模态feature原地置零，只优化fused CE与当前模态CE。

裁决：这是明确的数据拆分、batch重组和分阶段优化，不是新的最终检索对象。它也主动置零另一模态，不执行
matched wrong donor的独立正目标；当前规则又明确禁止用增加stage或数据重混救旧臂，故排除。

### 11.4 ResTacVLA（arXiv 2026-07）

- 论文：*Feeling the Unexpected: ResTacVLA for Contact-Rich Manipulation via Residual Tactile Representation*
  （<https://arxiv.org/abs/2607.03387>）；
- 项目仓库：<https://github.com/Awilekong/ResTacVLA>；审计commit：
  `76250e58bdf8a0d927f68a818cd7ba9ca95a4b3a`。仓库当前只有论文、网页和展示资产，没有方法源码。

论文事实：

- Cross-Modal Predictor从wrist RGB预测触觉latent均值与不确定性，实际触觉编码为`z_t`，机制对象是
  `r_t=z_t-z_hat_t`；
- residual经event encoder和VQ形成contact primitive，同时通过重建、NLL与commitment loss预训练；
- 冻结CMP后，`g_t=sigmoid(MLP(sigma_t))`把primitive与learnable no-contact token插值，再将该token拼入
  action expert；
- 测试时仍需要真实触觉，且低不确定阶段明确允许路径收敛到no-contact token。

裁决：`cross-modal prediction residual`是本轮最接近“只保留强模态无法解释的证据”的结构原子，但其两阶段
训练、测试时额外传感器、VQ/gate和动作策略目标均不满足当前RGB-only single-descriptor合同。论文也没有
matched wrong tactile及donor-owned正目标。直接把CLIP evidence减去student prior会成为已有predictive-coding
原子的迁移，并不能解决teacher在测试时被移除后的ownership。

### 11.5 SCOPE（Findings of ACL 2026）

- 论文：*SCOPE: Preserving Modality-Specific Cues to Mitigate Modality Laziness in Multimodal Learning*
  （<https://aclanthology.org/2026.findings-acl.1453/>）；
- Anthology与论文未给官方源码，按正式论文公式审计。

方法事实：SCOPE以matched cross-modal cosine趋近1、cross-sample cosine趋近0实现MI surrogate；再为各模态
构造batch内kNN语义图，以图矩阵Frobenius差做结构对齐；最终feature是模态权重和，并经过graph-masked
attention与`h <- h + A_tilde h`扩散残差，联合classification loss训练。

裁决：它明确覆盖“matched/mismatched similarity + modality-specific cue preservation + relational topology +
balanced residual fusion”。但mismatched sample只是被压到正交，不承担自身身份正目标；最终加权和与残差路径
也不排除强模态shortcut。将这些loss或batch图搬到TAPF仍是辅助正则/融合模块，而非所有权机制。

## 12. 更直接的evidence-use与retrieval近邻

### 12.1 RCL：答案保持但证据依赖漂移（arXiv 2026-07）

*Hidden Forgetting in Continual Multimodal Learning: When Accuracy Survives but Grounding Fails*
（<https://arxiv.org/abs/2607.02020>）直接提出“答案未忘、evidence-use已漂移”。RCL冻结上一checkpoint，
逐一mask visual/text/OCR等channel，用full与masked输出KL及答案loss增量构成sample-level reliance vector；
再以JSD匹配teacher/student reliance profile，并联合task loss与prediction distillation。论文当前未提供官方源码。

裁决：这篇工作直接占用了“counterfactual channel suppression -> reliance profile -> teacher/student reliance
matching”的机制空间，也进一步证明仅看最终准确率不够。但RCL只保留旧模型已有的依赖分布，不知道该依赖是否
语义正确；干预仅删除channel，没有matched wrong donor，更没有donor自身正目标。把exp402/403的bypass差改成
可训练reliance loss因此既不新，也不足以建立`correct > wrong > NULL`。

### 12.2 VIGIL：seeing相对blind的视觉信息增益（ECCV 2026）

*Staying VIGILant: Mitigating Visual Laziness via Counterfactual Visual Alignment in MLLMs*
（<https://arxiv.org/abs/2606.26387>）定义
`VIG=log p(y|vision,text)-log p(y|blind,text)`；blind path保持输入tensor不变，只在所有层mask text-query到
visual-key/value的attention。CVD以DPO形式提高seeing相对blind的preferred-response likelihood，并用当前
seeing-blind gap动态衰减该正则。项目页标注`Code coming soon`。

裁决：VIGIL已系统覆盖“full与all-bypass成对执行、直接最大化输出依赖差”的二元机制。它仍只要求去掉视觉后
变差，并通过降低blind preferred-response confidence获得margin；没有wrong evidence的独立正目标，也不能辨别
模型是否使用了正确sample的视觉。因此，单纯把exp401 route gap写进训练目标不构成exp404。

### 12.3 MiMIC与VLM2Rec：retrieval中的modality collapse

- MiMIC（<https://arxiv.org/abs/2604.21326>）用分离encoder、T5 decoder cross-attention生成retrieval
  embedding；训练期随机把visual-only/text-only embedding混入fused embedding、主动drop caption，并再做
  ANCE hard-negative第二阶段。它本质上是fusion architecture + data dropout/mixin + extra stage；
- VLM2Rec（<https://arxiv.org/abs/2603.17450>）直接在推荐retrieval中定义每用户text/vision margin，冻结强
  模态margin、放大fused InfoNCE负项以补弱模态，再用两个模态对in-batch候选的soft rank distribution做双向
  KL topology alignment。最终表示仍是`z_text+z_vision`。

裁决：两者证明retrieval领域已经存在针对modality collapse的结构和目标设计。MiMIC违反当前禁止data remix/
caption dropout/额外stage的边界；VLM2Rec是动态负项加权与topology regularization，最终加性descriptor仍可
bypass，且没有matched wrong/generic/NULL合同。它们排除了把普通modality-balance retrieval loss写成新机制。

## 13. 本轮结论与下一检索对象

本轮没有发现可以建立exp404的机制。新增文献把边界收紧为：

1. `independent unimodal objective + test concat`已有UniCat；
2. `permutation/MI/game regularizer`已有MCR；
3. `data split/remix、modality dropout、extra stage`已有Data Remixing与MiMIC；
4. `predictive residual + uncertainty gate`已有ResTacVLA；
5. `matched/mismatched similarity + topology alignment`已有SCOPE与VLM2Rec；
6. `counterfactual suppression reliance matching`与`seeing > blind`分别已有RCL和VIGIL。

这些近邻使当前**问题/证据创新门更可信**：最终性能可以保持而evidence-use失败，且open-set retrieval确实需要
比full-vs-bypass更强的source-specific终审。但机制门仍为空。下一轮只继续寻找一种三方闭合对象：

- correct execution对当前identity有正目标；
- matched wrong execution对donor identity仍有独立正目标，不能仅被推远或降置信；
- 两者共享同一最终标准欧氏descriptor构造，并保留generic/NULL/all-bypass终审。

若公开方法只做full-vs-mask、modality balance、MI/topology、fixed concat、dropout/remix或额外stage，即直接排除。
当前状态维持`AUDIT ACTIVE / NO EXP404 / GPU NO-START`。

## 14. 第三轮：swap/cross-reconstruction与donor-ID合同的可定义性

上一轮暂时把下一机制冻结为`correct -> current ID`、`wrong -> donor ID`、两者共享同一最终descriptor路径。
本轮对真正执行component swap并显式分配身份标签的方法做代码级核对，结果表明这个合同不能不区分组件语义而
直接套到当前16维evidence上。

### 14.1 DG-Net：donor-ID成立的前提是交换对象本身承载身份

- 论文/代码：*Joint Discriminative and Generative Learning for Person Re-identification*；
  <https://github.com/NVlabs/DG-Net>；审计commit：
  `9855f08711df1d7ebdf976885b0fddec8e7d4a37`。
- `trainer.py`先从图像A/B得到structure code `s_a/s_b`和由ReID encoder得到的appearance/ID code
  `f_a/f_b`，再构造`x_ba=decode(s_b,f_a)`、`x_ab=decode(s_a,f_b)`；
- 交换图重新编码后的appearance预测分别以`l_a/l_b`监督，并同时使用image/feature/cycle reconstruction、
  GAN、VGG和可选teacher目标；在AB双头设置里，另一个头还显式接收structure来源标签；
- 标准测试脚本不执行交换生成路径，而是直接对原始query/gallery图像运行ID encoder并拼接其两个输出特征。

裁决：DG-Net并不是“任意donor signal都应检索到donor ID”。`f_a`本来就是由完整图像训练出的身份承载码，
所以`decode(s_b,f_a)`按A的身份监督有科学含义。它依赖生成、重建、GAN/teacher等复杂对象，且swap路径不是
最终标准检索路径。把当前低维support/appearance evidence类比成`f_a`会偷换信息充分性。

### 14.2 Hi-CMD：style donor不拥有身份标签，identity prototype才拥有

- 论文/代码：*Hi-CMD: Hierarchical Cross-Modality Disentanglement for Visible-Infrared Person
  Re-Identification*；<https://github.com/bismex/HiCMD>；审计commit：
  `48a96edafa16612725ac48b3d925a3cf00eda52f`；
- 代码把图像分为prototype code `c`与attribute/style code `s`，交换style/extrinsic索引后生成cross/cycle图，
  再做图像与attribute-code reconstruction、GAN以及ID CE/triplet；
- 关键标签规则不是“跟随被交换的style donor”：`x_ba`重新编码为`(c_b_recon,s_a_recon)`时标签为B，
  `x_ab`的`(c_a_recon,s_b_recon)`标签为A；更多组合也让标签跟随prototype/content来源；
- 测试时从原图提取prototype与identity-style块，归一化拼接后得到检索特征，不执行cross-generation路径。

裁决：Hi-CMD给出直接反例。若交换的是姿态、光照、style等非身份充分成分，donor只有该语义状态的所有权，
身份正目标仍跟随identity prototype。其生成式分解、重编码和多目标训练也已经占用了普通swap/cycle原子，
不能作为当前teacher-free、RGB-only、单descriptor机制的新意。

### 14.3 CIFT：反事实对象是graph affinity，不存在donor身份转移

- 论文：*Counterfactual Intervention Feature Transfer for Visible-Infrared Person Re-identification*
  （ECCV 2022，arXiv v4：<https://arxiv.org/abs/2208.00967v4>）；论文未给代码链接，本轮未找到可归属
  作者的官方实现，故按v4正式公式裁决；
- H2FT把单query与异模态gallery重组为不平衡图，分别执行heterogeneous/homogeneous message passing；
- CRI保持输入特征`X`不变，只用高斯重参数化的`X*`替换affinity得到
  `Y_TIE=Y_{X,A_X}-E[Y_{X,A_{X*}}]`，再对当前真实身份计算CE；
- 正式CIFT测试使用query-gallery graph feature，`CIFT†`则完全退回backbone feature。

裁决：CIFT训练的是“正确topology相对高斯反事实topology的分类效应”，反事实项没有donor实例，也没有
donor正标签。正式推理又是pair/set-dependent graph，而非当前要求的单图固定欧氏descriptor。它不能补上
source-specific ownership合同。

### 14.4 对当前16维evidence的可定义性裁决

当前evidence只描述局部support/appearance语义，并未被定义或验证为identity-sufficient code。不同身份可以
共享相同support、姿态、颜色或局部外观状态。因此要求`descriptor(x_A,e_B)`对身份B为正，等价于要求16维
`e_B`在保留A的visual trunk时仍足以唯一指定B。可行的优化解只能是让evidence泄漏身份、让shared trunk学习
新的shortcut，或把不对应任何真实人的A/B组合硬贴成B；三者都不是预期的semantic ownership。

把wrong分支改为重建B的semantic state是可定义的，但它只给auxiliary decoder正目标。DG-Net/Hi-CMD已经覆盖
swap、re-encode和semantic/style reconstruction原子，而且这种目标仍不能证明最终身份descriptor必须使用
correct evidence。故它不能单独通过机制门。

**修正后的合同**是“目标必须跟随信息承载者”，而不是无条件`wrong -> donor ID`：

1. 身份充分组件可以对其source identity为正；
2. 语义/非身份组件只能对其source semantic state为正；
3. 若主张最终identity retrieval ownership，仍须在同一部署descriptor上预注册
   `correct > matched wrong > generic/NULL`与all-bypass，但wrong不能只被训练成负例；
4. 在当前evidence没有独立且path-complete的最终检索正目标前，这四项无法同时闭合。

因此撤回上一轮把`wrong -> donor ID`作为普适准入条件的过强表述。它是一次有用但科学上不完备的候选合同，
不是新机制。第三轮只使问题定义更准确，没有找到结构对象；创新门仍为
`PROBLEM/EVIDENCE GAP ONLY / MECHANISM FAIL`。不创建exp404、不写formal config/contract、不做CPU/CUDA或GPU
执行，继续查找能让semantic source target与最终identity ranking共享不可绕过对象、同时不要求身份泄漏的机制。

## 15. 第四轮：composed retrieval揭示“语义正目标”必须可观测

第三轮确认semantic donor不能无条件继承donor identity。下一问题是：公开检索工作如何让“主体 + 语义修改”
获得合法正目标。本轮重点审计了与人检索最接近的Composed Person Retrieval，并用最新direct-composition CIR
核对其一般性。

### 15.1 Composed Person Retrieval / FAFA（NeurIPS 2025）

- 论文：*Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for Composed Person Retrieval*
  （<https://arxiv.org/abs/2311.16515v4>）；
- 官方代码：<https://github.com/Delong-liu-bupt/Composed_Person_Retrieval>；审计commit：
  `0cc16936f031f7ad166be4cce1be33d0b44b728e`。

任务与代码事实：

1. 一个监督单元是`(I_q,T_q,I_t)`：reference person image、描述从reference到target变化的relative caption、
   以及**同一身份**的真实target image。语义修改的正目标不是caption donor的身份，而是已经实现该修改的`I_t`；
2. ITCPR人工标注2,225个此类triplet。SynCPR则用Qwen生成文本quadruple、微调Flux同时合成identity-consistent
   image pair，再用Qwen-VL按图像质量、身份一致、图文对齐和triplet可推断性过滤，保留约115万triplet；
3. FAFA将reference image与relative caption经Q-Former融合成query feature，target image经视觉分支产生32个
   token feature；主FDA目标用batch内exact ID/GID软标签直接对齐query与target；
4. 公开代码的正式推理仍需要reference image和caption，并对每个gallery的token集合计算top-k平均相似度。
   因而它不是单RGB、单固定descriptor，而是带测试时文本的query-conditioned token scorer。

裁决：CPR给出了科学上正确的semantic ownership模板：`主体 + 修改 -> 已实现该修改的真实同身份target`。
但它也暴露当前合同缺少的变量。exp402/403的wrong donor B是same-camera、different-PID样本；`e_B`不是“把A
变成某状态”的相对描述，official数据中也没有一个已知的`I_t(A,e_B)`。所以既不能把组合贴成ID B，也不能
声称存在一个对ID A唯一正确的counterfactual target。

### 15.2 DiCE-CIR（arXiv 2026-07）

*DiCE-CIR: Direct Composition Learning for Efficient Zero-Shot Composed Image Retrieval*
（<https://arxiv.org/abs/2607.04665>）进一步说明真实target image不是唯一实现方式，但**target semantics仍必须
显式存在**。它从reference caption生成edit text与target caption，以冻结CLIP把
`Phi(reference image, edit text)`直接对齐target-caption embedding、edit residual方向和batch contrastive目标；
测试仍输入reference image与edit text，再对gallery image embedding检索。

裁决：target caption可以作为图像target的语义proxy，但当前16维evidence既不是relative edit，也没有对应的
target-caption/attribute annotation。把teacher code自身当target只会回到exp402/403已经失败的proxy-active
叙事；引入LLM生成描述、外部caption或合成target则改变数据与部署对象，而且其基本triplet/composition机制已
被CIR/CPR直接覆盖。

### 15.3 对exp404的可识别性门

现有official ReID训练集中的same-ID多图不能自动补上这个缺口。把A与同身份图A'配对，只给出“身份相同”，
没有证明16维`e(A')`精确描述从A到A'的修改；普通ID/metric目标又可以完全忽略该evidence。要让semantic target
可识别，至少需要以下之一：

- 已标注的relative state与实现该state的同身份target；
- 可审计、identity-consistent的counterfactual生成器；
- 测试时显式semantic query与conditional scorer。

前两项分别进入CPR的数据生成/标注与DG-Net/Hi-CMD的生成式近邻，第三项违反当前单RGB、单descriptor部署合同。
因此“在当前official数据上直接增加composition loss”不是新机制，也没有科学确定的正目标。

第四轮裁决保持`INNOVATION GATE FAIL / NO EXP404 / GPU NO-START`。下一检索对象进一步收紧为：不引入外部
annotation/generation或测试时第二输入，却能从official RGB本身构造**可验证的realized semantic target**，并让
该target与最终固定欧氏identity descriptor共享不可绕过路径。没有这样的结构与正反contract前，不进行CPU或
CUDA实验。

## 16. 第五轮：已知变换的equivariance可验证，但不能替代semantic target

第四轮要求从official RGB内部构造realized target。一个自然候选是对同一图像施加已知变换，再用equivariance
提供解析target；另一个候选是用invertible map保证所有信息进入终端表示。本轮分别核对其ReID先例与可识别性。

### 16.1 DiP：几何target来自已知affine action

- 论文：*DiP: Learning Discriminative Implicit Parts for Person Re-Identification*
  （<https://arxiv.org/abs/2212.13906v2>）；论文未提供官方代码链接，本轮仓库检索也未找到可归属作者的实现，
  故按正式公式审计；
- DiP用part token与patch feature的相关性加权坐标得到implicit position `p`，再施加已知translation/scale/
  horizontal flip矩阵`K`生成变换图`X'`和解析target `p'=Kp`；
- 原图与变换图都保留同一ID label，position-equivariance loss只回归`p'`；
- 最终推理丢弃predicted position，仅以两图各自预测的DiP weight做pair-specific part distance。

裁决：这是official RGB内部realized target的合法例子，但target只对**已知几何群作用**可定义。DiP已经占用了
“仿射图像 + 解析位置equivariance + part retrieval weighting”的机制空间，而且其位置监督不进入固定欧氏
descriptor。把相同loss移到当前router/evidence上只能证明几何proxy，不会证明correct semantic evidence拥有
身份排序。

更根本地，当前16维support/appearance evidence没有预先定义的群表示`R(K)`。颜色、遮挡、局部支持和CLIP
appearance在translation/flip后应保持、置换还是变化，不能从一个仿射矩阵唯一推出；different-PID wrong donor
更不是由host图像的已知变换生成。故不能像`p'=Kp`那样构造其解析target。

### 16.2 invertibility不等于factor ownership

针对“让descriptor与evidence双射即可防bypass”的候选，本轮没有在person ReID公开实现中找到满足当前合同的
直接结构；理论上它也不够。Locatello等人的ICML 2019工作
*Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*
（<https://arxiv.org/abs/1811.12359v4>）证明，在缺少数据/模型归纳偏置时，存在无穷多个同分布但因子完全混合的
bijective latent reparameterization。双射只保证信息可恢复，不指定哪部分latent对应哪个真实因子。

当前设置并非完全无监督：teacher提供了16维evidence target。但这份监督只定义“student能复现teacher code”，
没有定义该code如何唯一作用于final identity ranking；exp402/403正是其反例。因此将终端层改成normalizing flow、
invertible coupling或固定direct-sum，只能强迫数值信息存在，仍允许身份排序对其不敏感或以混合坐标解释它。

### 16.3 第五轮裁决

已知变换可以产生geometry target，但当前semantic evidence没有已知action；可逆性可以保证information
preservation，但不能提供factor attribution。两者组合仍是DiP式equivariance与普通invertible/disentanglement
原子的拼装，并不能直接建立`correct > wrong > generic/NULL`。

状态保持`IDENTIFIABILITY/MECHANISM GATE FAIL / NO EXP404 / GPU NO-START`。下一检索只考虑能为当前evidence
给出**可观测或解析的semantic action**，且该action直接定义最终固定identity metric；普通affine consistency、
augmentation invariance、invertible flow或part-weighted pair scorer直接排除。

## 17. 封板资产的只读CPU前提诊断

本轮没有重跑、补跑或修改exp402/403，只检查已封板落盘资产是否足以回答“16维evidence是否身份充分、是否受
camera混淆”这一前提问题。远端GPU compute process保持为0。

资产与执行器事实如下：

1. exp402与exp403的formal result都记录correct臂在运行时`captured_rows=19,871`，但两个JSON中的最长数组
   长度均只有2；没有逐样本index/path、PID、camera、evidence或descriptor数组；
2. 审计代码在进程内建立`evidence_cache[len(records),5,16]`和各臂descriptor tensor，只把metrics、
   descriptor delta统计、hash与恢复门写入result，进程退出后没有tensor落盘；
3. 两个远端audit目录只有脚本、JSON、日志与manifest；训练output只有`train_log.txt`和唯一checkpoint，
   不存在`.npy/.npz/.pt`逐样本导出；
4. Phase0E codebook也不是逐样本缓存。它只有`covariance_rows=39,249`、五个slot计数、`5x768` slot mean、
   `16x768` shared PCA basis与eigenvalue等聚合量，没有path/PID/camera/evidence/descriptor；exp403 generic
   asset则只是一个`5x16`全局常量。

因此现有封板产物不足以计算同身份/跨身份evidence距离、camera条件分类或evidence-to-descriptor互信息。为得到
这些量重新执行exp402/403会构成禁止的补跑；仅从checkpoint新导出也会产生一个新的、未预注册测量执行，不能
伪装成封板结果的离线分析。

本轮裁决为`SEALED ARTIFACT SUFFICIENCY FAIL / DIAGNOSTIC UNANSWERABLE WITHOUT NEW EXECUTION`。这不改变
exp402/403的VALIDITY PASS或科学NO-GO，只明确限制：不能据现有资产声称16维code“身份可分”或“camera无混淆”。

## 18. 第六轮：canonical action有真实target，但不闭合当前ownership

### 18.1 3D-VAN与CSCL：canonical target依赖额外可观测几何

3D-VAN（*Generalizable Person Re-Identification via Viewpoint Alignment and Fusion*，arXiv:2212.02398）用
RSC-Net、Texformer与SMPL把单图重建成3D人体，再渲染前/后/左/右四个canonical view。正式测试把原图feature与
四个canonical-view feature拼接。它证明canonical view是可执行对象，但并未移除原RGB旁路，而且“正确/错误
canonicalizer”的差只能归因于几何重建是否破坏图像，不能证明当前16维appearance evidence的source ownership。

CSCL（ACM MM 2023，官方CSE代码commit
`924d5c2b661ff2decd08450d6f42532e9437360e`）更直接地把2D pixel映射到canonical SMPL vertex。其代码读取
DP3D/CSE JSON中的`dp_x/dp_y/dp_I/dp_U/dp_V`，依赖SMPL UV、27,554个vertex geodesic matrix与surface
annotation；DP3D每图还人工标注约80–125个2D–3D correspondence。公开仓库只给CSE模块而非完整ReID fusion。
这些变量在当前official Occluded-Duke RGB与冻结pose资产中不存在，不能通过普通2D keypoint等价替代。

### 18.2 VPFA：已知nuisance action仍只是成对残差映射

最新VPFA（*Resolution as a Direction*，arXiv:2510.00936v2；官方代码commit
`13de109d72ee3a2228c959dd42046332d0b17b24`）提供另一个可观察action：同一文件的HR图与2x/3x/4x降采样图形成
exact pair，冻结ReID backbone后用MSE训练`LR feature + MLP(LR feature) -> HR feature`。官方inference代码还
从文件名suffix读取分辨率倍率，再选择对应的三套MLP。

所以VPFA的target合法，是因为resolution label与paired HR target都可观测；但其机制本质是post-hoc feature
residual completion，并使用测试时nuisance oracle。它既不定义wrong-RGB semantic target，也落入本项目已反复
否决且当前明确排除的feature-level residual completion类别。

### 18.3 第六轮裁决

canonicalization只有在action本身可观测时才可识别：3D-VAN/CSCL需要外部3D/密集对应，VPFA需要paired
degradation与resolution label。当前official RGB中没有等价的semantic action。若直接拿wrong donor的pose/code
去warp host，`correct > wrong/zero`最多证明“错误warp更具破坏性”，仍不是realized semantic target；保留原RGB
fusion又重新开放bypass。

因此canonical warp、dense-surface alignment或resolution-vector residual均不形成exp404。状态保持
`ARTIFACT/IDENTIFIABILITY/MECHANISM GATE FAIL / NO EXP404 / GPU NO-START`；下一对象必须在现有official
RGB与固定欧氏descriptor内同时给出可观测action和不靠破坏control制造的source-positive target。

## 19. 第七轮：source-provenance patch composition的三难

canonical action失败后，本轮考察一个更直接的RGB内部realization：把两张official图的patch/token按已知mask组合，
使每个输出区域的source可观测，再为source设置正目标。

### 19.1 SPT已经实现ReID中的跨身份source transfer

AAAI 2024的Saliency-Guided Patch Transfer（SPT，官方代码commit
`ef1e71a99bc658790d5dbbc9ab133588e849e814`）并非普通随机擦除。它训练SPS mask把token划为identity set与
occlusion set，再按OIoU/mask rolling选择batch donor。正式公式与代码都执行：

```text
Z_i' = M_j * Z_i + (1 - M_j) * Z_j
```

即保留target `i`在candidate `j`显著mask对应位置的token，用`j`的其余token提供真实背景/遮挡。最终仍用标准
global Euclidean ReID descriptor，测试时不需要SPT。

关键是SPT没有把candidate残留身份当正目标。论文明确把新样本标为target ID `i`，并从softmax分母和triplet
negative中忽略candidate class `j`，因为hard mask下candidate可能残留局部身体信息，却不足以让合成图拥有第二个
合法全局身份。官方源码的`RandomMix`逐token保存相同source mask，论文的class-ignoring目标正好说明
“source已知”不等于“每个source都拥有一个全局PID target”。

### 19.2 dense provenance label也不是新原子

Token Labeling（NeurIPS 2021，官方commit `9dbfd59aedecfe83f6f3253db4e99b82359d48ac`）已经为每个patch token
建立location-specific dense class target，同时联合class-token与token CE。TokenMix（ECCV 2022，官方commit
`0e17d5dda10fa4afe654aee4ca87373620b9ee2d`）进一步按teacher activation map为两个source计算content-based
mixed target，避免只按像素面积混标签。ReID自身还有Ped-Mix、Strip-Cutmix和Cutmix Dual Branch等题名直接近邻；
后3篇没有公开代码且全文受限，本轮不对其未核实细节作推断，但它们已经使“person patch mix”不能被无条件声称
为空白机制空间。

### 19.3 三种composition均不能闭合当前合同

1. **只换背景/遮挡**：SPT的host ID target合法，但donor不拥有身份正目标；它训练的是context invariance，不是
   semantic evidence ownership；
2. **换完整identity set**：把B的全部身份承载patch移到A背景后，target B合法，但evidence已变成身份充分的
   B像素/token payload。wrong臂检索B只是“实际把人换成B”，退化为SPT foreground transfer加DG-Net式
   appearance swap，不再回答当前16维support/appearance code如何作用；
3. **只换一个或若干semantic part**：每个patch的A/B provenance可以监督，但合成person没有单一PID。逐token
   source CE是合法辅助目标，最终global descriptor却没有对应的真实gallery positive；强制它远离A只会重新变成
   destructive wrong control。

因此已知source mask解决了局部标签观测，却没有同时解决非身份semantic evidence与全局identity target。将
SPT、TokenMix/Token Labeling、part descriptor和swap loss组合起来不满足新机制门。

第七轮裁决为`SOURCE-PROVENANCE TARGET TRILEMMA / MECHANISM GATE FAIL / NO EXP404 / GPU NO-START`。
下一对象必须避免“context donor无身份、full donor问题退化、partial donor无全局target”三者之一，而不能仅换一种
patch mask或mixed-label权重。
