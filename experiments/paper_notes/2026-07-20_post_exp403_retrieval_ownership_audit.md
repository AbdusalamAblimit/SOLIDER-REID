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
