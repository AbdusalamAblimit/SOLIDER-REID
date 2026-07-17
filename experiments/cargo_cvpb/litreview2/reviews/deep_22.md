Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019ef835-efa7-7ed1-b12e-2dc7230c8756
--------
user
你是 ReID 创新方法论拆解员 22。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(22)==
Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf
Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf
Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf
Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf
Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf
RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先确认目录里的文件和项目规则能读到哪些，再逐篇用 `pdftotext -f 1 -l 8` 抽方法部分。这个任务偏文献方法论拆解，我会把重点放在“观察如何被包装成问题”和“证据如何支撑视角”，不是复述模块结构。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc pwd in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title

exec
/bin/zsh -lc "sed -n '1,260p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' ../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
# CLAUDE.md · ReID 工作区（reid-paper）

## 0. 这个工作区是什么

这是一个全新的 ReID 研究工作区。目标是做一篇 CCF-B 级别、真正新颖、有效、又能讲清楚的行人重识别工作。具体做哪个子方向（遮挡、标准、跨域、长期等等），自己调研之后再定。

目录说明：
- `SOLIDER-REID/` 是干净的上游代码（github.com/tinyvision/SOLIDER-REID，版本 8c08e1c）。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这里从零写。
- `experiments/` 是实验记录，从 exp000 开始编号。
- `.claude/rules/` 是详细规则，`.claude/hooks/` 是用来强制执行纪律的检查脚本。

## 1. 接手要先做什么（最高优先级）

第一步不是开实验，而是先对齐目标，再放开去调研、把方向定下来。

1. 先读 `GOAL.md`。这是当前目标的唯一来源，由用户来写。用户随时会改它来调整方向，所以每次接手、每个大的步骤开始之前都要重新读一遍，以它为准。它说的是"现在要做什么"，你写的 `experiments/STATUS.md` 记的是"进度到了哪里"，两者不要混。
2. 读本文件，了解铁律、三条研究纪律、对新方向的要求、以及训练前的审查规矩。
3. 读 `experiments/STATUS.md`，看现在到了哪一步。
4. 按 `GOAL.md` 当前的目标，自己读文献、做差距分析：ReID 现在还有哪些没解决好的真问题？最新的工作都在改进什么、又漏掉了什么？形成你自己对"哪里能做出真正新东西"的判断，提出几个有野心、又确实能做出来的候选方向。

方向没想清楚之前，不要开实验。

## 2. 铁律（违反了基本等于白做）

1. 数字只认日志。所有指标都要用代码从日志文件里解析出来，不能凭记忆、凭印象写。
2. 凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。
3. 正常波动范围：rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在噪声范围里，不能算作成果。
4. 每涨一次点，都要换一个挑刺的角度重新核对一遍，看它是不是噪声、是不是数据泄漏、是不是评测口径前后不一致。
5. 评测口径是冻结的。要改评测口径，必须先问用户。
6. 正式训练之前先用很小的规模快跑一遍，确认不会崩、模块确实在起作用。
7. 做好实验记录（`experiments/decisions.md` 和 `results.md`）。同样的配置加同样的种子，不要重复跑。

永远不要挑随机种子，也不要挑表现最好的那个 epoch，那等于评测作弊。一律上报最后一个 epoch 的结果，不要用 best_model。

## 3. 三条研究纪律（这个项目最容易栽跟头的地方，必须遵守）

第一，判定一个方向"死了"之前，要先定好标准、并且有足够的证据。开始跑之前就把"什么样的结果才算这条路走死了"写进 `design.md`，比如三个种子配上两种配置都落在噪声范围内或者为负。只有一两个负结果的时候，只能写"还需要再试"，不能判定整条方向死掉，更不能据此就推翻方向。一个活跃的方向往往要反复试很多次才出东西，掉几个点很正常，不要一受挫就放弃。

第二，自己写的评测或分析脚本，要先用它复现一个已知的基准成绩，对得上之后才能用它的结果下结论。任何新的评测口径、新的度量、新的评测脚本，都要先拿它跑出一个已知的基线成绩，确认对得上，才能信任它的输出。一个写错的脚本足以把整条方向引到沟里去。

第三，"贡献"是个有门槛的词，不要夸大。一个结果，只有同时满足下面几条，才能叫做贡献、才能说可以投稿：通过了第 4 节对新方向的要求；跑了三个种子、报了均值和标准差；涨幅超过了噪声范围；并且和最接近的已有工作区分得清清楚楚。在那之前，一律只叫它"信号"或"探索"。复现别人的方法、公开别人没公开的基准、做一个分析，这些都不算贡献。

## 4. 一个新方向值不值得做（先过这一关）

ReID 是个活跃领域，每年都在出 B 类甚至 A 类的工作。不要一上来就觉得"能做的都被做完了"，那是错觉。你的任务是放开去找一个真正新颖、有效、又讲得清楚的角度。

一个新方向至少要满足下面三条里的两条，否则不作为主线：
1. 问题上有新意：不是"加一个模块"，而是重新定义、或者更准确地刻画一个真实存在的问题。
2. 机制上有新意：是过去的工作没有清楚写出来、而且代码上能实现的机制。
3. 证据上讲得清：能设计出干净的对照和消融，能回答"它为什么有效"。

另外几条硬要求：
- 要和最接近的已有工作区分得清清楚楚，不能是换个名字的同一个东西。
- 方向定下来之前，自己和 codex 或者子代理讨论核实，确认它确实是新的、和最接近的工作区分得清楚。
- 不能拿测试时的小技巧（重排序、特征归一化、翻转测试这类）当作主要贡献。
- 不能用"比基线高了零点几"来定义创新。

先把文献读够、把方向选准，再去花算力。要保持不轻易放弃的劲头，但动手之前多花时间读论文、做差距分析。

## 5. 正式训练前的两轮独立审查（改了方法的实验，必须做）

任何改了模型、或者有新设计的实验，在启动训练之前，都要经过两轮互不通气的独立审查：
- 一轮由 Claude 做：用 Agent 工具起一个 opus 子代理来审。
- 一轮由 codex 做：用 `codex exec`，内联在 Bash 里跑。

两个审查者互相看不到对方的结论，也不知道这是第几轮、不知道你改了什么。每一轮的结论分别写进 `experiments/expNNN/review-claude.md` 和 `review-codex.md`。

规矩是这样：
- 只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。
- 操作上的、文档上的小问题，记成待办，不拦。
- 实质问题修好之后必须再审一轮，不能修完就放行。某一轮里两个审查都没有实质问题，才算放行。
- 纯复现实验（只改随机种子）不用审查，在 `design.md` 里写一行"需要训练前审查：否"就行，检查脚本会放行。
- 检查脚本 `.claude/hooks/check_design.sh` 会在 train.py 执行前检查：设计文档在不在、两份审查结论是不是都通过。没通过会直接把命令拦下来。

## 6. 一直往下做（只要用户没说停）

默认的工作节奏是：先把方向定下来（读文献），写好 `design.md`，过两轮独立审查，先小规模快跑一遍，再正式后台训练，用 Monitor 跟着日志看，跑完立刻补好文档（results 和 decisions），然后接着做下一个。

- 每个大步骤开始之前都重新读一遍 `GOAL.md`。它变了就马上按新的来，用户是靠改这个文件来调方向的，不一定会打断你。如果 `GOAL.md` 的主目标被清空、或者写成了"暂停"，就停下来等用户，不要自己找活干。
- GPU 不要空着：要么排下一个实验，要么补文档、读文献、做消融表。
- 不要频繁问用户。长期自己往下做、自己拿主意；拿不定的先找子代理或者 codex 讨论再定。只有真正只能用户决定的事（改评测口径、大方向的取舍），才打断用户。
- 用 Monitor 或者后台的 Bash 等待器来跟日志，不要用 sleep 反复轮询。
- Claude 的额度紧张时，能独立完成的子任务（独立审查、讨论、探索）多交给 codex，省额度。

## 7. 机器和网络

- 你在 Mac 上跑，能联网（GitHub、pip 镜像、HuggingFace 镜像都通）。
- 服务器只有国内网，装包用清华源，下模型用 hf-mirror。
- 三台 GPU（详细连接方式见 `.claude/rules/remote_server.md`，连接信息在 `~/.ssh/config`）：
  - `hyy-5060ti-double`：恒源云，两块 5060Ti 16G，环境已经配好，`/hy-tmp` 只有 50G。
  - `lab-3090-d`：实验室的 RTX3090 24G，在一个 docker 容器里，经 `lab-3090` 跳板连；容器一重启就会丢掉 sshd 和 IP。
  - `lab-4090`：实验室的 RTX4090D 24G，是共享机器，只能用 `afr` 自己的空间，绝对不要碰 `/root`、`/hy-tmp` 和共享的 conda。
- 磁盘纪律：`/hy-tmp` 只有 50G，每次训练只保留最后一个 epoch 的 checkpoint，中间的和 best_model 都删掉。
- 训练在服务器后台跑（用 `setsid nohup ... </dev/null &`），Mac 这边通过 ssh 监控、解析日志。

## 8. 代码底子（上游 SOLIDER-REID）

- 上游只带了 Market-1501 和 MSMT17 的配置（Swin 的 Tiny、Small、Base）。要用别的 benchmark（比如 Occluded-Duke、Occluded-ReID），得自己加配置和数据集读取代码。
- 用 SOLIDER 的预训练权重（从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint）。Mac 能连 Google Drive。
- `SEMANTIC_WEIGHT` 默认是 0.2。
- 主干的顺序：先用 Swin-Tiny 快速验证，有信号了再放大到 Small、Base。创新点先在 Tiny 上看有没有效果，确实有效再放大，不要在没效果的点上用大主干硬磨、白白浪费算力。

## 9. 怎么写中文（这一条务必照做）

所有给人看的文字，包括文档、注释、提交信息、以及给用户的回复，都要用平实、完整的中文句子，像正常人说话写字那样。

下面这些不要出现：
- 不要把判断标准、检查步骤之类的东西生造成"某某门"（比如"创新门""双审门""新颖性门"）。中文不这么说话。要表达就直接说清楚，比如"判断方向值不值得做的标准""训练前的两轮审查"。
- 不要用箭头（→）把几个词串起来代替句子，过程要一步一步用完整的句子写出来。
- 不要用破折号（——）来连接或停顿，改用逗号、句号，或者干脆把话说完整。
- 不要在中文里夹英文当术语（比如 skeptic、smoke、gate、overclaim、free lunch 这种），有对应的中文就用中文。真正的专有名词、命令、文件名、配置字段（train.py、design.md、SOLIDER、Swin、seed、best_model 等）该用原文就用原文。
- 不要堆砌小标题、加粗、表格来充字数，能用一段话讲清楚就用一段话。
- 不要用"收口""罗盘""锚点""铺路""掺进去"这类比喻性的简写和网络口头禅。

这一条对你自己想问题的时候同样适用：思考的时候也用完整的中文句子，不要用上面这些写法。

## 10. 详细规则

放在 `.claude/rules/` 里：
- `result_discipline.md`：数字纪律和三条研究纪律（最重要，先读）。
- `experiment_protocol.md`：实验怎么命名、`design.md` 的格式、两轮审查的流程。
- `innovation_and_decisions.md`：决策怎么记录、怎么让两个子代理对着辩论、对创新方向的把关。
- `documentation.md`：文档结构和数据一致性。
- `monitoring.md`：怎么用 Monitor 和后台 Bash。
- `remote_server.md`：三台机器的详细连接方式和磁盘策略。

exec
/bin/zsh -lc 'ls -1' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
# 当前目标（唯一来源，随时改这里来调方向）

这是 Claude 当前要追的目标。要调整方向，直接改这个文件就行，不用 `/goal clear` 那一套。Claude 每次接手、每个大步骤开始前都会重新读它，以它为准（见 `CLAUDE.md` 第 1 节和第 6 节）。这个文件由用户来写，定的是"做什么"；`experiments/STATUS.md` 由 Claude 来写，记的是"进度到哪"。两者不要混。

## 主目标

做一篇 CCF-B 级别、真正新颖、有效、又讲得清楚的行人重识别工作。

## 现在这一步

自己读文献、做差距分析，提出几个有野心、又确实能做出来的候选方向。每个方向写清楚：它解决的是什么真问题，新机制是什么，打算怎么证明它有效，和最接近的已有工作差在哪里。提完拿给用户挑。先不要开实验。

## 这一步什么时候算完成

提出三到五个候选方向，每个都满足"问题、机制、证据这三条里至少占两条"，并且和最接近的工作区分清楚。和codex或者子代理去讨论之后选择其中一个，才进入写设计文档、做两轮审查、正式训练的环节。

## 不要做的事

- 方向还没定、还没通过对新方向的要求之前，不要开实验。
- 不要夸大。一个结果，没有同时满足"对新方向的要求、三个种子的均值和标准差、涨幅超过噪声范围、和已有工作区分清楚"这几条，就只能叫"信号"。

## 暂停信号

- 用户说"停"就停。
- 这个文件的主目标被清空、或者写成"暂停"，就表示当前没有目标，等用户的指示，不要自己找活干。

 succeeded in 0ms:
'Knowledge and experience' for visible-infrared person re-identification.pdf
3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf
4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf
A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf
A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf
A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
ADA framework for unsupervised domain adaptation person re-identification.pdf
Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf
Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf
Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification.pdf
Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf
Adversarial perturbation and defense for generalizable person re-identification.pdf
Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf
Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf
Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf
Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
Bidirectional modality information interaction for Visible-Infrared Person Re-identification.pdf
Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf
CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf
CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf
CLIP-driven fine-grained mining for text-based person search.pdf
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf
CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf
ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf
Confidence guided semi-supervised cross-modality person re-identification.pdf
Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf
Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf
Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf
Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf
Cross-modality average precision optimization for visible thermal person re-identification.pdf
CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf
DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf
Deep intelligent technique for person Re-identification system in surveillance images.pdf
Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf
Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf
Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf
Distribution aligned semantics adaption for lifelong person re-identification.pdf
Diverse Representations Embedding for Lifelong Person Re-Identification.pdf
DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf
Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
Exploring Part-Informed Visual-Language Learning for Person Re-Identification.pdf
FDGReID - Federated Domain Generalization for Person Re-identification.pdf
FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf
FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf
False Negatives Consensus Suppression for Text-to-Image Person Re-identification.pdf
Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Focusing on pedestrians like human for clothes changing person re-identification.pdf
GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf
GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf
GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf
Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf
Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf
HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf
Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf
Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf
Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf
Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf
InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf
Instant pose extraction based on mask transformer for occluded person re-identification.pdf
Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf
Internal-External Context Interaction Network for Person Re-Identification.pdf
Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf
Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf
Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf
Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf
MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf
Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf
MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf
Multi Queue for Unsupervised Person Re-identification.pdf
Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf
Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf
Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf
Multi-feature balanced network for clothes-changing person re-identification.pdf
Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf
Multi-year long-term person re-identification using gait and HAR features.pdf
Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf
Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf
Occluded person Re-Identification with noise injection.pdf
Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf
Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf
Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf
Privacy preserving person re-identification via anonymizing diffusion model.pdf
Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf
Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf
Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf
RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf
RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf
Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf
Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf
Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf
Shape-centered representation learning for visible-infrared person re-identification.pdf
Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf
Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf
Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf
Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
Spectrum-guided feature enhancement network for event person re-identification.pdf
Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf
Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf
TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf
TP-LReID - Lifelong person re-identification using text prompts.pdf
Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
Text-to-image Person Search based on Semantic Reorganization.pdf
Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf
Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
Two-stage knowledge distillation for visible-infrared person re-identification.pdf
UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf
Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf
Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf
Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf
Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf
Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
Unsupervised person re-identification via camera-aware multi-level label refinement.pdf
Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification.pdf
Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf
Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf
Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf
When Large Vision-Language Models Meet Person Re-Identification.pdf
manifest.csv

exec
/bin/zsh -lc "sed -n '1,220p' ../../../experiments/STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
# 活动状态（每步更新）

## ★★★ 当前最新（2026-06-14，转"补标准数据+公平对标"执行路线 决策#54，正获取标准LReID数据+搭LSTKC标准协议台子）

**方向转折(决策#54)**: 大方向取舍抛用户后用户表达失望("算了我对你绝望了")。复盘: 我一直在查先例/做归因/把候选判负/再把方向推回用户,优化"别做通用/已有的东西"而非"做出完整的东西",耗算力却没落地。**改打法,自己拿决定不再punt**: 不再"找新机制→判负"循环,直接解决卡脖子的"缺公平对标台子":
- 用标准 LReID 协议代码库 LSTKC(AAAI2024, 已 clone 到 lreid_lstkc/): 标准5域顺序(market→cuhk_sysu→duke→msmt→cuhk03)+遗忘矩阵+未见域泛化+PTKP/PatchKD/LSTKC强基线全现成,是 C2R/AKT 共用的台子。
- 补标准数据(已有Market/MSMT;取CUHK03/CUHK-SYSU/Duke[学界通用,论文不展示图]/7个unseen,子代理 a98cae3 正查获取路径)→复现LSTKC/PTKP基线对上其论文数字(纪律2)→在公平台子做方法对标C2R/AKT,三种子出稿。
- 候选二"多版本混库塌21mAP"作为干净 motivation,标准协议上做成多版本兼容/免全量重建索引的真方法。
hyy 候选二复现序列(v2 OccDuke)仍在跑,给第二条序列的多版本不兼容佐证。

**执行进展(2026-06-14 晚)**: 子代理给出 link 核验的数据获取计划(无预处理包,各自装;Drive 必须 Mac 下再 rsync;CUHK-SYSU 自裁剪;MSMT 加载器只支持 V2)。已完成:
- LSTKC 环境在 lab-4090 装好(独立 .venv_lstkc, torch2.x, continual_train.py --help 通过; uv 超时调 UV_HTTP_TIMEOUT=600 才装上)。
- Mac 下好 3 个训练域: CUHK03 cuhk-03.mat(1.1G)+新协议 mats、DukeMTMC-reID(154M,版图对)、CUHK-SYSU(160M)。脚本在 experiments/exp029_lifelong/{lreid_download_mac,setup_lstkc_env}.sh。
- **5 训练域数据已全部到手**: ①Market+MSMT17_V1 已符号链接进 PRID/(版图校验OK); ②MSMT 加载器已 patch 支持 V1(VERSION_DICT 加 V1 条目; V1≈V2,仅人脸模糊+目录名不同,身体ReID数字几乎无差); ③CUHK03 cuhk-03.mat(1.1G)+新协议mats 已下; ④Duke 已下(版图对); ⑤CUHK-SYSU **原始** person-search 已下(1.2G tar.gz,18184 SSM图+annotation),排成 cuhk-sysu/CUHK-SYSU/ 加载器自裁剪版图(之前那个160M是预裁剪版,弃用)。
- 关键发现: 服务器够不到 Google Drive/Google,但能到 Github+hf-mirror;故 Drive 数据(CUHK03/Duke/CUHK-SYSU)Mac 下再 rsync 到 lab-4090。relay4090 跳板不能并发 ssh,传输要串行。
- 剩(纯工程): rsync CUHK03+Duke(在跑 bfj7oxudl)+CUHK-SYSU raw 到 lab-4090 /mnt1/afrdata/PRID/ → 部署 msmt17.py patch(scp,等 relay 空)→ 跑 continual_train.py 复现 LSTKC 标准5域基线对上其论文数字(纪律2)→ 在公平台子做方法。
---

**全局裁决(经验,非先验; 详 results.md exp029 + 决策#51/52/53)**: 候选一(跨阶段桥)无效、候选二(多版本陈旧索引)前提判负=经典BCT不兼容悬崖非新结构、候选三(阶段顺序桥选择)前提弱=遗忘在差异极大目标域上高度一致(−20.7~−21.9,无域距离结构,LTCC偏小由数据集大小解释)。task-free无边界(4.5/10)未测且协议风险高。codex两次独立判终身方法稿在现约束下 3-4/10,binding=缺标准数据(Duke/CUHK-SYSU/CUHK03,无法公平对标)+主流子问题全拥挤。机制候选已便宜先导穷尽→剩下的路被用户专属决定主导(补敏感数据/换论文形态/离开终身)→已用 AskUserQuestion 抛大方向。hyy 候选二复现序列仍在跑确认,不空等。
---

**候选二裁决(主序列, 详 results.md exp029 + 决策#52 执行结果)**: 判负。复现校验过(v0 Market 87.01=训练日志)。Market 陈旧索引: mAP 阶梯非单调(g=v0 51.74>g=v2 50.85,是 v0 作为 Market 专家的质量混淆,非版本年龄递减); 真正大效应是"混版本图库" −21mAP,而同质旧版本单独可用(g=v0 51.74)→ 这是经典 BCT/C2R/Hot-refresh 不兼容悬崖(那条线的立项动机),不是新的可利用版本年龄结构。候选二作为新方法线判负。hyy 独立序列(Market→OccPoseTrack→OccDuke)跑完确认稳健性。
**已转候选三便宜先导(lab-4090, run_exp029_cand3.sh, 等待器 bn1jz0w7n)**: 从同一 v0 单步 fine-tune 30ep 到 OccDuke/PRCC/LTCC/MSMT,量各目标造成的 Market 遗忘,看是否随目标域/域距离有规律(校正数据规模混淆)。有强顺序效应→候选三(4/10)前提成立可继续; 遗忘与目标域无关→候选三前提弱。
---

**模块线确证关闭**：公平基线纠正后真训练的 exp028 CCS 在 PRCC 换衣给了真 +2.8 mAP/+2.3 rank1（且压方差），但归因对照判通用——randcloth(乱衣服)45.8≈CCS 45.9，增益是弱基线上辅助投影头+同身份三元组的通用正则,非衣服机制(决策#49)。exp024 部位结构 real 63.3≈random 62.9 也通用。连同 exp019/OSG,3/3 attribution-tested 模块全是通用增益 → "标准单 embedding 加头/分支/损失"方法线在新颖性上确证封顶(决策#50)。

**codex option-B 战略复审**：冻结协议内基本无可撑 CCF-B 的方法贡献；要做方法稿须放松一个约束。**用户拍板：转终身/增量 ReID(工程深度)**——真正不同的问题类(训练过程+数据流上定义新问题),离死线远,中等拥挤但活跃。

**缺口分析已出**(lifelong_reid_gap-codex.md):候选一(跨阶段同身份桥)零训练先导无效(PRCC 换衣太轻,原型检索已 100%,无空间,决策#51);候选二(多版本陈旧索引)其次,且能用手上 Market/MSMT/Occ-Duke 标准不相交身份序列直接做。候选三(阶段顺序敏感桥选择)备选,最接近 AKT/DASA/DKAR 先例压力大。

**exp029 终身最小基建已搭+先导在跑(2026-06-14 08:06)**:
- 加 MODEL.CONTINUE_FROM 配置 + train.py 注入(顺序微调:加载旧域主干+瓶颈,分类头按域重建) + eval_lifelong_stale.py(多版本陈旧索引评测,复用仓库标准 eval_func,单版本退化即标准评测做复现校验)。本地语法过、lab-4090 烟测过(CONTINUE_FROM 加载 PRCC ckpt 进 Market 训练,分类头重建,loss 9.55→正常,13.4s/epoch)。
- lab-4090 后台跑顺序微调代理序列(seed 0):v0=Market(120ep,jx_vit 初始化)→v1=+MSMT(60ep,接 v0)→v2=+Occ-Duke(60ep,接 v1)。约 70min,EXP029_SEQ_DONE 标志。这条序列同时产出朴素基线+遗忘矩阵对角。
- 训练后:先用 v0 在 Market 的训练日志 mAP 校验 eval_lifelong_stale.py(复现校验),再跑候选二探针(陈旧阶梯 g=v0/v1/v2 + 随机混版本)。判负线见 design:混版本掉点<1mAP 或阶梯非单调(悬崖式=通用不兼容非可利用结构)→候选二判负转候选三。
- 代理序列只做机制先导+工程自检,不冒充标准 SOTA(缺 Duke/CUHK-SYSU/CUHK03);若候选二过线,再补标准数据做公平对标。

**候选二先例核查(codex 3/10, candidate2_vs_c2r-codex.md, 决策#52)**: 免重建索引+持续兼容转移被 C2R(CVPR24)/Bi-C2R(TPAMI26)/URCPD(AAAI26) 占; 新旧兼容+在线部分回填+预算曲线被 BCT 系/Hot-refresh(ICLR22,已做回填比例+预算曲线)/DGR 占。窄空白=多版本原始特征同库+相机/时间非均匀陈旧+预算调度,且增益须可证来自版本年龄结构。前提门=两条独立序列在 Market 上看混版本是否明显掉点+随版本年龄单调(对照各版本 native 自评排"质量阶梯"混淆)。不过→转候选三(阶段顺序敏感桥选择,未撞免重建/BCT 工业线); 过→仍须全套破坏性归因(打乱版本/随机陈旧/全重提/年龄反转吃掉70%增益)+超 C2R 全量转移基线+三种子才算贡献(同杀模块线那道归因门)。

**两条独立先导(都在跑)**: lab-4090=Market→MSMT→OccDuke(主, ~13s/epoch); hyy GPU0=Market→OccPoseTrack→OccDuke(独立复现,不同中间域,防单序列偶然)。两条都在 Market 上做陈旧索引评测(eval_lifelong_stale.py 已用 Occ-Duke ViT-base 53.70 复现校验通过)。完成等待器 b2adeq7sh(lab-4090)/b8pv682at(hyy)。
GPU 分配: hyy GPU1 + lab-3090-d 暂空——候选二仅 3/10 的前提先导,2 个工作位已足,不在低先验方向上铺满算力; 前提过线再扩种子/破坏性对照/标准数据。

**候选三+拓宽先例核查(codex, candidate3_and_broaden-codex.md)**: 候选三(阶段顺序敏感稀疏旧域桥选择) 4/10,被 AKT/DASA/DKAR/DKP/PKA/LSTKC 从"动态迁移+旧知识过滤"两侧挤,只有当便宜先导证明域距离能预测遗忘/迁移(Spearman≥0.4、近远旧域遗忘差>1mAP)且随机相似度/最远桥/全旧域均匀对照吃掉增益才值得做。拓宽最不坏两条: task-free/无边界终身(按相机时间 shard 漂移触发 micro-domain,4.5/10,协议风险高,最接近 CIPR/CVS) > 候选三收窄版(4/10)。整体判断: 当前数据(缺 Duke/CUHK-SYSU/CUHK03)+算力下终身方法稿 3-4/10,codex 诚实建议若前提都不成立则别硬做、把 exp029 基建当协议分析素材退回更稳形态。**策略: 先用 exp029 一条序列同时验候选二(陈旧阶梯)和候选三(域距离→遗忘)两个前提的经验证据,有强信号再投;两个前提都经验证伪才把"是否离开终身方向"作为有证据的大方向问题抛用户。**

资产:PRCC/LTCC/Market/Occ-Duke 标准 ViT-base 3 种子基线 + 各探针/归因基建。lab-4090 venv=/home/afr/reid-clean/.venv、hyy venv=/hy-tmp/reid-clean/.venv(python 不在 PATH,必须用 venv 全路径)。MSMT 已 symlink MSMT17→MSMT17_V1。jx_vit 两机都在(lab-4090 /home/afr/reid-clean/weights/、hyy /hy-tmp/reid-clean/weights/)。

---

## 旧（2026-06-14，公平基线纠正——已被模块线全关+终身转向取代）

**用户关键纠正（记忆 fair-baseline-not-solider）**：为什么 SOLIDER 强主干成了否决一切的理由？全 B 类语料没人用 SOLIDER/Swin 当基线，我们自定一个全场最强、没人用的基线再否决所有方案，本末倒置、自我否决。两个真错误：(1) 我把"强主干吸收 nuisance"过度泛化成对整个鲁棒性簇的判决；(2) 用便宜探针代替真训练去否决（用户反复说的"别只probe判负、要真训练"）。

**纠正后的操作原则**：门槛=同行用的标准基线（ViT-base，我们手上 PRCC 43.0/Market 86.8/Occ-Duke 53.3/LTCC），公平对标已发表 B 类数字；不再要求打赢 SOLIDER（SOLIDER 只做附加 scale 验证）；验证用真训练三种子，探针只排优先级不一票否决。据此**重开**被基线假象/探针假象误杀的方向。

重判已出（reopen_under_fair_baseline-codex.md）：重开 3 方向（exp024 无姿态结构在标准 ViT-base 已有 +2.2~2.7 rank1 真信号被 SOLIDER 尺子误埋 / PARTIAL_EVIDENCE / 换衣稳定证据）；C 类真死保持（OC4超加性=数据属性、exp022贴图检测器、与CAL/CCIL/instance-wise同质）。

**已落到真训练（exp028 CCS 跨衣稳定证据保全，主注）**：
- 标准 ViT-base PRCC 3 种子基线已训完：CC mAP 43.0/46.1/40.3（均值 43.1±2.4，方差大，exp028 对比必须按种子配对抵消方差）、rank1 41.2/43.8/41.2。这是公平基线资产。
- exp028 CCS 插件实现+两轮独立审查通过（Claude opus 子代理 90 行 + codex 均放行）+ 自检 5/5。lab-4090 正训 3 种子（seed0 ~44/120，GPU 92%，损失含 CCS 项在降）。等待器 b026y61ao：训完报 CC，再跑 cc_eval 取 CC+SC 按判负线裁决（CC>+0.4mAP且rank1>+0.5、SC不掉、配对非采样器假象）。
- hyy: MGPARTS 占位消融因 protos/bank 路径依赖重建复杂、信号弱(flat mAP)已放弃; 改后台把 PRCC(217M)拷到 hyy，为 exp028 出信号后并行消融/LTCC 做前置。lab-3090-d 空闲（PRCC 瓶颈在 lab-4090）。

下一步取决于 exp028: 过判负线则补消融(采样器对照已=基线,需±衣服破坏视图)+LTCC+对标CAL/CCIL; 不过则按"还需再试/判负"处置,转 PARTIAL_EVIDENCE 或 exp024 结构线坐实。

注：仍真死的（机制被证伪或新颖性被占，换基线也救不了）：OC4 超加性是数据属性（决策#47）、exp022 贴图检测器机制错、与 CAL/CCIL/instance-wise invariance 同质的部分。

---

## 旧（2026-06-14 夜，自适应不变性判负——此条"整簇都死"的泛化已被上面的纠正收窄）

自适应不变性方向判负（决策#48）。Level-1+flip噪声地板: PRCC换衣 blur 改善15.5%真异质,但 Market 标准 ViT 仅8%(噪声地板6.8%)、强主干 Swin 7.5%<8.7%。Level-2 冻结选择器: PRCC oracle+6 但选择器预测线索28%≈随机25%,应用-0.94mAP。注:Level-2 是【冻结探针】非真训练,按新原则这属"探针假象"待真训确认。

正在做: 起多路 codex 先例优先搜下一方向(明确禁用鲁棒性/不变性/去偏/线索手术死簇),我用 WebSearch 逐个核实先例。资产复用: PRCC/LTCC/Market 标准 ViT-base 基线 + 三个零训练探针(指纹/oracle/冻结选择器)在 lab-4090/hyy。三机当前空闲(探针类零训练已完成),搜出方向+探针过线再占 GPU。

---

## 旧（2026-06-14，OC4 冲突角度门0 判负）

---

## 旧（exp027 VCR 实现+双审，已被门0 判负取代）

方向曾落到代码：exp027 VCR（可见衣物冲突路由，OC4-ReID 遮挡加换衣联合设定，标准 ViT-base 基线，不用 SOLIDER 强主干）。用户拍板原话"如果强基线成为劣势就别用，写完代码开双 codex 审查"。

进度：
- VCR 插件已实现（model/vcr.py + make_model/processor/make_dataloader/defaults 接线 + configs/{prcc,ltcc}/vit_base.yml + test_vcr.py + probe_superadd.py），默认关闭逐字节退化，本地 CPU 自检 6/6 过。
- 第一轮三路独立 codex 审查全部不放行，但都是可修问题：新颖性 5.0/10（窄但开放，命门是门0 超加性和交互项消融）；正确性两处（衣服关系损失 detach 切断路由器、跨衣损失用了 PRCC/LTCC 不可达的异身份同衣对）；协议两处（LTCC 探针把 general 当同衣 SC、合成贴图不能与 OC4 官方数字同台）。
- 已全部修复并验证：梯度流改进后用梯度检查确认衣服关系损失只训练路由门与关系头、不灌主干；跨衣损失改可达三元组、关系损失改身份内同衣换衣；LTCC 探针改 mode=SC、单位统一个百分点；design 收窄叙事、命名 Synth-Occ、补 CAL 式全局对照与先例核查清单。提交 0adc745、584a5dc。
- 第二轮 codex 审查：正确性 approve（无实质问题）；协议放行门0 探针与小规模冒烟，门1 三种子论文级裁决暂不放行，要求补一个与完整 VCR 同双前向口径的"只增广"控制臂加显式消融开关（这是公平性要求，不挡门0）。
- 正在跑（lab-4090，已恢复）：PRCC 标准 ViT-base 基线 seed0（约 20 秒一个 epoch，120 轮约 40 分钟）→ 门0 超加性零训练探针（两个遮挡种子）→ LTCC 基线 → LTCC 探针，远端 orchestrate_oc4.sh 串起来，Mac 侧等门0 结果通知。

下一步取决于门0：超加性 > 0.4 个百分点则方向成立，补公平控制臂与消融开关后三种子训练 VCR；否则问题定义偏弱，判负转向。hyy 两卡、lab-3090-d 空闲（都缺换衣数据，PRCC/LTCC 只在 lab-4090）。

---

## 旧（2026-06-13 20:37，用户拍板"新信息源"，3路codex深读PDF全文中，A路和C路已完成）

部位方向(exp020/024/025/026)与换衣旋钮(决策#45)全判负归档,三机空闲。用户拍板大方向="调查新信息源,仔细看168篇B类论文"。
3路codex深读全文(experiments/paper_materials/newinfo_*.md):
- A 语义/VLM teacher: 已完成,见 experiments/paper_materials/newinfo_semantic_teacher-codex.md。结论是遮挡语义 teacher 对 SOLIDER 强主干整体偏负,FLaN-Net/π-VL/AG-ReID/LVLM-ReID/RMPSNet 等发表遮挡成绩多低于或接近现有强基线; 唯一值得先探的是属性 teacher 的可靠性噪声屏蔽,其次是 teacher 只用于属性级样本组织,两者都必须先做零训练探针。
- B 生成/合成数据: CCUP/InfinitePerson/Identity Diffuser/扩散增广,找适中成本+新颖的生成机制(换衣战场PRCC 49 vs SOTA 55-66,CCUP用百万合成预训练)。
- C 特权信息蒸馏: 已完成,见 experiments/paper_materials/newinfo_privileged_distill-codex.md。结论是普通pose/parsing/silhouette teacher 已被 PGFL-KD、π-VL、FLAG、AOANet等占住,且 SOLIDER 已含 parsing 能力; 最值得继续探的是 3D体型+dense correspondence 残差蒸馏,其次是有视频数据时的3D skeleton/gait relation,parsing只适合作负控。
出齐后综合排序+零训练探针,交用户定具体押哪条。纪律不变:探针先行,强主干验证,三种子。

---

## 旧（2026-06-13 15:30，新方向调研收敛：换衣衣服捷径为第一推荐，待零训练探针）

6 路 codex 调研全出（paper_materials/newdir_*.md）。候选清单第一推荐：**基于 SEMANTIC_WEIGHT 的衣服捷径可控蒸馏**。
- 为什么排第一：换衣是强主干仍明确失败的具体干扰（PRCC 换衣远低于同衣）；我们有独有观察（PRCC 换衣对 w 有约 7 分响应、Occ-Duke/LTCC 不响应）；已发表换衣方法都靠衣服标签/解析/CLIP/生成，"用预训练自带语义旋钮控衣服捷径、无衣服标签、测试单 embedding"是干净区分点。
- 命门（红队 codex 正在核准）：笔记里"w=0.2 换衣好"与"语义权重升换衣涨"文字自相矛盾，必须从代码核准 w 到底控制什么；w 差异方向若只是通用 detuning 而非衣服捷径，方向即塌。
- 执行铁律（血换的教训）：先做零训练探针（现有 PRCC/LTCC/Occ-Duke checkpoint 上提不同 w 特征、衣服方向线性探针、闭式衣服方向擦除对随机方向），探针过线才写 design、才开训练。预注册判负线已在候选报告里。
- 后备候选：换衣状态原型(2)、状态扰动结构保持蒸馏(3)、模型敏感遮挡课程(4)、相机白化(5)。

部位/掩码方向确认种子(exp026 nd_s1/s2)在 hyy 收尾(~15min)、PoseTrack 在 3090 收尾(~1.5h)，纯清账。新方向探针不需要训练 GPU。

---

## 旧（2026-06-13 14:45，方向重置：部位/掩码方向终审判负，回语料找方法级新方向）

用户拍板转向（原话）："回去找新方向，好好看别人怎么发的，主要是别人的思路，作出能发b会的方法级成果，codex任你使用token管够"。决策#44 已记。

**部位/掩码方向终审判负**（exp022-026，约30次训练，账本在 results.md + decisions.md#44）：抗遮挡重加权、构造掩码监督(软/硬)、语言部位分支(含忠实LGPA-D)全部在强主干上不成立。三路codex法医结论：PSG的姿态接地是LGPA-D全部集成选择的承重墙，无姿态版必败。唯一幸存=遮挡物贴图增广(Swin +0.83/+1.10)，但有先例非方法级。exp026确认种子(no-detach s1/s2)在 hyy 跑完即封档，不阻塞选向。

**新方向调研进行中**：4路并行codex（思路解剖/幸存性筛选/相邻领域扫描/强论文结构逆向），写入 paper_materials/newdir_*.md。出齐后第二波（候选生成+新颖性红队），再交用户拍板。重活全交codex，Claude只做全局把控。

**机器**：hyy双卡=exp026确认种子s1/s2(~1h)；3090=PoseTrack对照s2重跑→部位s1链；4090=跳板断，恢复循环守着。新方向定下来前不开新训练。

---

## 旧（部位/掩码方向攻坚期，2026-06-11-13，已判负归档）

## ★★★ 当前最新（2026-06-11 午后 14:10，exp022 判负在途、exp023 形状池审查中）

用户硬要求不变：方法稿、双指标。主线推进到第三层迭代：

**exp021 纯增广（数据轴）：已裁决，确认信号。** 三种子 54.60±0.22 / 61.47±0.33，对无增广对照 +1.33 mAP / +0.80 rank1，双指标过噪声线。第一条 ViT-base 上活下来的训练期增益（results.md）。

**exp022 方块池掩码监督：判负在途，但诊断值钱。** s0 54.1/60.5（-0.3/-0.6 对同种子纯增广）、s1 53.2/59.2（-1.3/-2.2），s2 收尾中，数学上已够不着 +0.8 信号线。诊断（viz_occaware.py）：留出合成遮挡 IoU 0.9628（监督完全可学、跨图泛化），但真实遮挡 query 上头只认"最像贴图补丁"的物体，汽车/行人/雨伞全漏。结论：头学成了**贴图检测器**，根因是方块裁剪池与真实遮挡物分布错配，监督机制本身无罪。归因臂（W_OCC=0）双卡在 hyy 跑（~15:00 完），其 occ_bce 不降反升、α 深负的对照动态已记录。

**exp023 形状遮挡物池（诊断驱动的正路）：实现+自检全过，两轮独立审查进行中。** COCO 实例分割 512 物体库（15 类、每类≤64、种子 20260611，零依赖 pycocotools），alpha 合成贴图，token 掩码升级为形状精确（覆盖率半格规则）。自检七项含"矩形遮挡物下形状路径与矩形路径逐格互证"。新颖性边界已核：FOSENet/OCCNet 用过 COCO 形状做增广（增广本身不是主张），没人做构造副产品的 token 级监督。对照臂 MECH=False（同库只增广不建头）。裁决：方法对 aug-only ≥+0.8 mAP 且 rank1 不降。
- 审查放行后顺序：lab-4090 跑冒烟 → 方法臂 s0/s1/s2 + aug s2 链；hyy 跑 aug s0/s1（库已在传）。全部今天 ~17:30 出齐。
- Swin-Small 移植已就绪（827e023，双自检过），exp023 有信号立即上（生死线）。

**锚点收尾**：Swin-Base s2 在 lab-3090-d（~14:30 完）；s0 64.4/72.9、s1 64.5/73.3 已档。

更新（15:00）：exp022 已封档（方法 53.97±0.58/60.50±1.06，对纯增广 -0.63/-0.97，判负；α 轨迹三种子一致、死因=贴图分布错配，诊断三件套进论文）。Swin-Base 锚点封档 63.63±1.15/72.50±0.86。exp023 冒烟过闸、方法 s0 在跑。库 md5 四处字节一致（f9370729…），跨机库混杂顾虑解除。

机器：lab-4090=exp023 六臂链（方法s0→aug s0→…，~18:45 收齐，第一对同种子读数 ~16:20）；hyy=wocc0 双臂收尾（~15:00）→自动接力 Swin-Small 形状库 aug 对照 s0/s1（带冒烟闸门）；lab-3090-d=Swin-Small 形状库 aug 对照 s2（65.6s/epoch，~17:10 完）。

---

## 旧（2026-06-11 午后早些，方法稿主线：exp022 三种子链在跑）

用户两条新硬要求已生效：所有结果同时报 mAP 和 rank1；**要方法稿，不要分析稿**（强主干审计降级为动机/分析章节）。主线重组见 paper_materials/story.md（2026-06-11 版）。

**主线 exp022（构造精确遮挡掩码监督，方法稿核心）**：
- 机制：贴图增广免费产生的像素级遮挡位置 → token 网格 0/1 掩码 → 线性头在第 8 块后预测、只在合成样本上 BCE 监督 → 零初始化 α 可学重加权喂回主干。测试零外部输入。
- 两轮独立审查同轮通过（069249a），七项自检两台机器全过，冒烟通过：occ_bce 1.01→0.26（头在学）、α 在动（注意走的是负方向，收敛后符号待看）、评测路径正常。
- **三种子链在 lab-4090 跑**（4090 速度 15.7 秒/epoch，全链约 2 小时，监视器 biefwf3ui 盯 done 标记）。
- 裁决：对 exp021 纯增广对照三种子，mAP ≥+0.8 且 rank1 不降算信号；+0.4~0.8 动重试格（W_OCC 1.0 / BLOCK 6）；重试后 ≤+0.4 判死。有信号先跑 W_OCC=0 归因消融。
- 可视化脚本 viz_occaware.py 已部署（真实遮挡热图 + 合成三联图 + 留出 IoU），s0 一完就出图。

**对照臂 exp021（纯增广数据轴）**：s0 完成 54.4/61.1（对同种子控制 +1.8/+1.2）；s1（第 49 轮）/s2（第 32 轮）在 hyy 双卡，等待器 bncb0fpt4。
**锚点收尾**：Swin-Base 无 cp，s0 64.4/72.9、s1 64.5/73.3 已档，s2 在 lab-3090-d（第 34 轮，等待器 bv5tdiume；注意容器时钟 UTC+8 才是北京时间）。

机器：lab-4090=exp022 链；hyy 双卡=exp021 s1/s2（完了接 exp022 的 W_OCC=0 归因臂或重试臂）；lab-3090-d=Base s2。

---

## 旧（2026-06-11 清晨，第二夜战役：方法三连测+锚点全家桶）

**方法线（候选 A 的三种形态，全部走完预注册流程）**：
- exp018 蒸馏损失版：三种子 +0.33（无信号，归档为消融臂）。
- exp019 OSG 门控版：三种子 +0.20（无信号；门控诊断 g2≈0.001 门没开）；W_SEM 0.3 预注册重试在跑（hyy GPU0）。
- **exp020 LGP（无姿态语言部位分支，主注）**：保留 LGPA-D 被验证的两个成分（部位级 ID 监督+部位特征进表征）去掉全部姿态依赖；两轮审查同轮通过（裁决程序：对照双杆同报、不用配对差措辞），seed0 在跑（hyy GPU1）。信号线 ≥+0.8 对 53.27/53.63 双对照。
- 关键认知：探针证明"信息存在"，三连测在证明"哪种注入形态才有用"——弱监督门控不行，强监督部位分支是 PSG 验证过的形态。

**锚点全家桶（全部 w=0.2 发表级配方，数字在 results.md）**：
- Occ-Duke：Tiny 56.87±0.29（exp018 对照三种子）；Small w=1.0 66.53±0.12 + w=0.2 链（66.9/66.6/s2在跑）；Base 64.4（无cp，seed1/2 排队）；ViT-base 53.27±0.49（exp019 对照三种子）。
- Occ-PoseTrack：Tiny 76.27±0.09（exp001h，已核实 w=0.2）；Small 77.40±0.08 三种子。
- Occluded-REID 跨域容量曲线：Tiny 71.0→Small 84.2→Base 86.4 mAP（官方检查点零训练直评；裸 Base 86.4/89.2 超 FED/KPR带提示/BPBreID，距 ProFD 约 2 点）。
- 换衣：PRCC Tiny w02 三种子 46.17±1.13（w 旋钮效应 +6 mAP 坐实）、Small w02 49.4；LTCC 全档低于发表（seeds 在补）。
- **两个工程毒点已揪出并落档**：WITH_CP（重入式检查点+AMP）毁训练（41.7→64.4）禁用；SEMANTIC_WEIGHT 默认 1.0 污染（全线统一 0.2 烤进配置）。

**强主干审计三支柱（独立于方法成败的分析主线）**：同域遮挡白送（裸 Small 超全部已发表 B 类）、跨域遮挡近顶（裸 Base 第二梯队）、换衣失灵但有 w 旋钮。加 14 个方法判负的谱系和可靠检索保底稿。

在跑：hyy=OSG重试+LGP s0；lab-3090-d=Duke Small w02 s2；lab-4090=LTCC tiny w02 s1/s2。等待器全挂。

---

## 旧（2026-06-10 凌晨，用户完全放权，夜间自主战役进行中）

用户指令（原话要点）："不要把自己限定在遮挡这个领域，大胆创新，大胆做所有可能让我们稳发 B 类的实验，三台 GPU 全用，不依赖 codex（额度尽，06-11 13:01 恢复），别用 Workflow（会话额度也耗尽过一次），你自己做全部决定，我去睡觉。" 方向部署见决策 #42：锚点先行的双战场，主纲领"强人本主干 + 训练期新信息源/训练组织，测试单 embedding 冻结"。

到 09:10 为止的战果（数字都在 results.md，全部提交）：
- **★ exp015a 定稿：Occ-Duke Swin-Small 干净基线三种子 66.53±0.12 / 75.73±0.21（batch64）**。单 embedding 无测试期额外输入口径下超过全部已发表 B 类遮挡方法（最高 FLaN-Net 65.5；arXiv 的 DPL-ReID 67.2 要测试期文本）。旧的 batch32"基线"64.4 作废。
- **exp016 换衣判决（Tiny seed0 两数据集）**：LTCC 换衣 12.3/24.7、PRCC 换衣 40.1/42.4（同衣 98.2/99.5 饱和），全面大幅低于发表水位。结论：SOLIDER 人本预训练放大衣服捷径，换衣降级为分析素材（与遮挡战场形成干净对照）。协议照搬 Simple-CCReID、读取器对账 PASS、两条评测路径数字一致。
- **exp017 探针放行候选 A**：SOLIDER 特征对 CLIP 遮挡物语义零线性解码（R²=-0.01）、嵌入级超锥零假设解释力仅约 26%、残差与 AP 相关 -0.164（控 s_top1 后 -0.097 贴线）。按预注册中间地带规则的属性级判据放行。
- **exp018（语义蒸馏 v1）实现完成**：插件式 SEMDISTILL（defaults/make_model/dataloader/processor），自检五项全过（ENABLED=False 字节级退化、标签 15618 全齐、蒸馏在学、eval 前向 max|diff|=0）。CLIP 标签和代码已部署 hyy。等 10:40（Claude 子代理配额）做第一轮审查、13:01（codex 额度）做第二轮，过审即 smoke+三种子。
- Swin-Base：下载+转换+加载烟雾全过（373 键 All matched）。

正在跑：
- lab-3090-d：Occ-Duke **Swin-Base seed0**（with_cp，114 秒/epoch，约 4 小时完，首个 Base 遮挡锚点）。
- hyy 双卡：**Occ-PoseTrack Small seed1/seed2**（Epoch 29/27；注意该配置 EVAL_PERIOD=20 会打中途 mAP，等待器盯 transformer_120.pth）。
- lab-4090：Small 三连链 **Occ-PoseTrack Small seed0 → PRCC Small seed0 → LTCC Small seed0**（链 pid 1719268；后两个是换衣捷径的容量趋势分析）。
- 各完成等待器已挂；check_monitor.sh 钩子按用户指示已删（保留 check_design.sh）。

---

## 旧（2026-06-10 早，通读 167 篇 2025/26 CCF-B ReID 论文找新角度，已完成）

用户醒后对可靠检索应用稿的定位有疑虑（它不是"模块带来大提升"的故事，而且最稳的去处 PRCV 是 C 类不是 B 类），下载了 167 篇 2025/2026 年 CCF-B venue 的 ReID 论文（paper_materials/pdfs_by_title/），指令是和 codex 配合全部读完、提炼经验、找出一个能发的角度。用户确信一定有角度。范围约束不变：只做遮挡+普通 ReID。

阅读已完成：167 篇全覆盖。38 篇范围内精读 + 88 篇范围外粗读由 codex 完成（笔记 paper_notes/notes/in_*.md、out_*.md），剩 41 篇因 codex 额度耗尽（2026-06-11 13:01 恢复）由 Claude 摘要级补读（missing_41_skim_by_claude.md）。元分析和候选方向写在 **paper_notes/SYNTHESIS.md**，要点：(1) 在"单 embedding、测试无额外输入"口径下我们的 SOLIDER Swin-Small 纯基线 66.5 已处于已发表 B 类遮挡工作头部（发表水位 57.8-65.5，仅 arXiv 的 DPL-ReID 67.2 和测试期多部件距离的 AOANet 70.6 更高）；(2) 模块类增益=弱基线假象（这批论文内部基线 48-53 起步），反向验证我们 13 个判负；(3) 站得住的增益全部来自训练期新信息源（VLM 语义/外部遮挡物数据/生成数据/分布先验）；(4) 空白=遮挡结构化 VLM 语义当纯训练期监督蒸馏进非 CLIP 强人本主干、测试回纯单 embedding，没人做过。候选 A=该语义蒸馏方向（先零训练探针：冻结 SOLIDER 特征能否线性解码 CLIP 属性/遮挡物标签，解码不出的残差是否关联检索错误）；候选 B=真实遮挡物数据源+分布先验（最便宜，A 的对照轴）；候选 C=干净到遮挡的结构保持蒸馏（中风险排队）；候选 D=可靠检索稿保底投 PRCV。待办：codex 额度恢复后对候选 A 做独立新颖性红队，然后交用户挑方向。未开任何实验。

---

## ★★★ 旧（2026-06-09，转可靠检索应用稿，覆盖以下所有旧状态）★★★

**方向(决策#39，用户拍板)**：选项1 加机制涨标准 mAP 累计判负 13+(MULTIHYP/DONOR/想法2/找新机制三候选[成对匹配/episode soft-AP/框所属者蒸馏]全栽 train→test 泛化/容量律/分割失效)。转**遮挡 ReID label-free 可靠检索应用诊断稿**(非新方法稿，定位见 paper_materials/reliable_retrieval_redteam-codex.md，命中 PRCV 0.5-0.65/ICME ICPR 0.35-0.55)。

codex
项目目标仍然是先读文献、做差距分析，不开实验。现在我开始按你给的六篇逐篇抽取前 8 页文本，并定位方法章节和消融证据章节。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 6, JUNE 2026

7573

Probabilistic Distribution Alignment for
Text-Based Person Retrieval
Xi Yang , Senior Member, IEEE, Kun Chen, Chenghuan Qi , and Nannan Wang , Senior Member, IEEE

Abstract—Text-based person retrieval is a cross-modal task
that seeks to match pedestrian images with their corresponding textual descriptions. A key challenge in this task arises
from the inherent one-to-many relationships: a single image
can correspond to multiple descriptions, and a single description may relate to several images. Conventional deterministic
embedding methods, which map images and texts to fixed
feature vectors, struggle to capture such complex relationships
effectively. To overcome this limitation, we introduce Probabilistic
Distribution Alignment (PDA), a framework that represents both
pedestrian images and text as probabilistic distributions and
models the interactions between visual and linguistic modalities.
PDA comprises three main components. First, Distributional
Representation Modeling (DRM) encodes images and text into
Gaussian distributions using a specially designed distance metric,
allowing the model to capture uncertainty in the representations. Second, Cross-Modal Containment (CMC) aligns the
distributions of text and masked text with their associated image
distributions to strengthen semantic correspondence. Third,
Intra-Modal Containment (IMC) enforces structured learning
within each modality by embedding distributions alongside their
masked variants, improving robustness to incomplete observations. Experiments on standard benchmarks demonstrate that
PDA achieves superior performance compared with state-ofthe-art methods, effectively handling ambiguity and cross-modal
variability. These results highlight probabilistic distribution modeling as a powerful paradigm for vision-language alignment in
pedestrian retrieval.
Index Terms—Text-based person
alignment, representation learning.

retrieval,

cross-modal

I. I NTRODUCTION
EXT-BASED person retrieval (TBPR) aims to retrieve
specific pedestrian images from large-scale galleries
based on free-form natural language descriptions [1]. Lying
at the intersection of computer vision and natural language
processing, this task is critical for real-world applications such
as intelligent video surveillance and public security.

T

Received 8 December 2025; revised 2 February 2026; accepted 4 February
2026. Date of publication 9 February 2026; date of current version 8 June
2026. This work was supported in part by the National Natural Science
Foundation of China under Grant 62372348 and Grant U22A2096, in part by
the Key Research and Development Program of Shaanxi under Grant 2024GXZDCYL-02-10, in part by Shaanxi Outstanding Youth Science Fund Project
under Grant 2023-JC-JQ-53, in part by the Scientific and Technological
Innovation Teams in Shaanxi Province under Grant 2025RS-CXTD-011, and
in part by Shaanxi Province Core Technology Research and Development
Project under Grant 2024QY2-GJHX-11. This article was recommended by
Associate Editor W. Ji. (Corresponding author: Xi Yang.)
The authors are with the State Key Laboratory of Integrated Services Networks, School of Telecommunications Engineering, Xidian University, Xi’an
710071, China (e-mail: yangx@xidian.edu.cn; chenk@stu.xidian.edu.cn;
qch@stu.xidian.edu.cn; nnwang@xidian.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2026.3662704

Fig. 1. Problems with existing TBPR: (a) TBPR faces complex image–text
relationships. (b) Schematic diagram of existing point representation methods.
(c) Our probabilistic representation method that considers both probabilistic
matching and probabilistic inclusion, allowing the model to better capture
complex and diverse cross-modal relationships.

Unlike conventional person re-identification (Re-ID) [2],
[3], [4], which relies on matching visual appearance cues,
TBPR requires bridging the significant semantic gap between
vision and language. A fundamental, yet often overlooked,
challenge in this domain is the inherent asymmetry of abstraction between the two modalities. Textual descriptions are
typically coarse-grained and semantically general (describing a
class of attributes), whereas visual signals are fine-grained and
instance-specific. Consequently, a single text description theoretically corresponds to a “set” of valid visual appearances,
while a specific image represents just one instance within that
set. Existing methods have largely treated this task as a symmetric point-to-point matching problem. Recent approaches
focus on fine-grained alignment through masking strategies
[5], [6], [7], [8], [9] or noise suppression [10], [11], [12] to
handle annotation errors. While these deterministic embedding
frameworks have advanced the field, they suffer from two
limitations. First, by mapping images and texts to fixed points
in a joint space, they fail to capture the one-to-many nature
of TBPR, where a general text should match multiple valid
images (Strong Positives I sp and Weak Positives Iwp in Fig. 1)
while rejecting visually similar but semantically distinct negatives (Icn ). Second, even methods that introduce uncertainty
(e.g., probabilistic embeddings) typically treat variance merely
as a “margin” for noise tolerance—effectively modeling “fuzzy

1051-8215 © 2026 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:29 UTC from IEEE Xplore. Restrictions apply.

7574

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 6, JUNE 2026

points” rather than structured semantic scopes. They largely
rely on symmetric metrics (e.g., Wasserstein distance) that
ignore the logical partial-order relationship: a general text
description should probabilistically contain specific visual
instances, not just be close to them.
According to Shannon’s information theory [13], deterministic events carry less information, whereas uncertain events
contain more information. To address the aforementioned
challenges, this paper proposes a novel framework termed
Probabilistic Distribution Alignment (PDA), which represents
both pedestrian images and textual descriptions as Gaussian
probabilistic distributions. By leveraging the variance components of these distributions, PDA not only captures inherent
uncertainty but also models the semantic containment relationships between vision and language. As illustrated in Fig. 1(c),
PDA is built upon two core principles: probabilistic matching,
which aligns visual and textual distributions, and probabilistic containment, which imposes containment constraints
both across and within modalities. For probabilistic matching,
PDA introduces a Distributional Representation Modeling
(DRM) module that encodes pedestrian images and texts into
Gaussian distributions. In this formulation, the mean vector
represents the central semantic meaning, while the variance
captures modality-specific uncertainty. By learning compact
distributions for strong I sp and weak Iwp positives, and more
dispersed distributions for confusing negatives Icn , DRM alleviates the inherent conflict between retrieving relevant matches
and rejecting semantically similar but identity-mismatched
samples. To model probabilistic containment, PDA further
integrates two containment modules: Cross-Modal Containment (CMC) and Intra-Modal Containment (IMC). CMC
enforces that the Gaussian distribution of a textual description
probabilistically encompasses that of its paired image, thereby
modeling the inherent semantic generality of language relative
to the fine-grained specificity of visual content. IMC, in
parallel, constrains each masked sample to remain within the
distribution of its corresponding unmasked representation. By
doing so, IMC encourages the model to learn representations
that are robust to partial observations and missing attributes.
Together, these containment constraints guide the network
to shape distributions with meaningful inclusion relations,
which not only strengthen cross-modal alignment but also
enhance robustness against noise, occlusion, and incomplete
descriptions in real-world TBPR scenarios.
Our main contributions are summarized as follows:
• We propose the Probabilistic Distribution Alignment
(PDA) framework, which fundamentally shifts TBPR
from point-based similarity to distribution-based containment. This allows the model to interpret variance as
semantic scope, distinguishing it from prior uncertaintyaware methods that treat variance primarily as a noise
buffer.
• We introduce the Cross-Modal Containment (CMC) module to model the asymmetric partial-order relationship
between general text descriptions and specific visual
instances, effectively resolving the ambiguity between
strong/weak positives and confusing negatives. We propose Intra-Modal Containment (IMC) to bridge the gap

between complete and masked observations. By unifying
cross-modal and intra-modal learning under the same
containment logic.
• Extensive experiments on three standard TBPR benchmarks demonstrate that our method achieves stateof-the-art performance with improved robustness and
interpretability.
II. R ELATED W ORK
A. Text-Based Person Retrieval
Li et al. [1] introduced text-based person retrieval (TBPR)
by releasing the CUHK-PEDES dataset, sparking extensive
research on bridging visual and linguistic modalities for person search. Early approaches typically employed separately
pretrained feature extractors, such as VGG [14] or ResNet50/101 [15] for images and LSTM [16] or BERT [17] for text
to generate unimodal embeddings, which were then aligned
using simple similarity metrics [18], [19], [20] or matching
losses. While effective at a coarse level, these methods often
lacked fine-grained cross-modal correspondence. To address
this, works such as SSAN [21] introduced multi-view nonlocal attention networks to capture part-level relationships,
whereas other methods incorporated auxiliary semantic information like attributes [22] or color cues [23] to enhance
local visual-textual alignment. During this phase, datasets
such as ICFG-PEDES and RSTPReid were also proposed
to support finer-grained retrieval tasks. With the advent of
vision-language pretraining (VLP) frameworks like CLIP [24],
recent approaches have shifted toward unified pretrained backbones that jointly model image-text interactions at scale.
For instance, IRRA [5] leverages masked language modeling to introduce implicit semantic reasoning, while CFine
[25] performs hierarchical alignment across both global and
local representations. TBPS-CLIP [26] demonstrates strong
performance by applying CLIP-based architectures with carefully designed data augmentations and loss functions, and
RDE [10] addresses noisy image-text correspondences using
Gaussian Mixture Models and triplet ranking loss to better
separate ambiguous pairs. However, the limited scale of the
three widely used TBPR benchmarks remains a bottleneck.
To overcome this, a number of recent efforts focus on
constructing large-scale pretraining datasets followed by finetuning. UniPT [27] adopts a divide-conquer-combine paradigm
to build LUPerson-T by generating pseudo-text descriptions
from the large-scale LUPerson dataset [28]. APTM [29]
employs the BLIP [30] model to produce the MALS dataset
enriched with structured attribute phrases. Similarly, NAM
[31] utilizes Multimodal Large Language Models (MLLMs)
to augment LUPerson with more diverse and robust text
using template-based diversity enhancement and noise-aware
masking. DP [32] further explores multimodal augmentation
by integrating diffusion models for image generation and
LLMs for high-quality textual annotations. UFineBench [33]
contributes UFine6926, a benchmark enriched with ultra-finegrained human-annotated descriptions and further expanded
with LLM-generated variants. While these methods successfully tackle data scarcity and improve representation richness,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:29 UTC from IEEE Xplore. Restrictions apply.

YANG et al.: PROBABILISTIC DISTRIBUTION ALIGNMENT FOR TEXT-BASED PERSON RETRIEVAL

they fundamentally rely on deterministic, point-based embeddings that compress each modality into fixed-length vectors.
Such representations are inherently limited in capturing the
ambiguity, semantic granularity, and distributional uncertainty
present in TBPR scenarios. In contrast, our work models
both images and textual descriptions as probabilistic Gaussian
distributions, enabling a richer and more flexible framework
for representing cross-modal semantics under uncertainty.
B. Probabilistic Distribution Representations
Probabilistic distribution representations have been increasingly used to model uncertainty and improve robustness
in various machine learning tasks [34]. In face recognition
[35] and person re-identification [36], Gaussian distributions
encode feature embeddings with uncertainty estimates reflecting representation quality. In domain generalization, DSU [37]
captures domain shifts via multivariate Gaussians instead of
deterministic features. Similarly, Pr-VIPE [38] models 2D
pose embeddings as Gaussians to handle input uncertainty
and enhance robustness to viewpoint changes. Probabilistic
methods also benefit multimodal tasks: PCME [39] captures
one-to-many relationships in cross-modal retrieval, MAP [40]
incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
uncertainty via batch- and identity-level variances.
The core philosophy of our PDA framework, representing
semantics as distributions and modeling their interactions
through containment, holds significant potential for a wide
range of complex visual tasks beyond TBPR. Specifically, in group activity recognition [42] and compositional
action recognition [43], probabilistic modeling can capture
the inherent semantic hierarchy where a global activity
distribution encompasses the distributions of diverse individual actions. This inclusion-based logic is equally feasible
for few-shot action recognition [44] and compositional
zero-shot learning [45], where representing class prototypes as
probabilistic scopes rather than fixed points allows for better
handling of intra-class variations and semantic ambiguity.
Furthermore, our probabilistic paradigm can be extended to
multi-task denoising diffusion models [46] to bridge the gap
between noisy partial annotations and unified semantic targets
through distributional inclusion. In the domain of hyperspectral image classification, the containment-driven philosophy
is uniquely suited for spatial-spectral perception [47] and
cascaded cross-attention alignment [48], providing a principled
way to characterize spectral uncertainty and the spatialcontextual containment of complex mineral patterns. Finally,
such asymmetric alignment remains particularly applicable
to audio-visual event localization [49], effectively bridging the semantic gap between general acoustic signals and
fine-grained visual cues.
Beyond Gaussian distributions, other probabilistic models
can also represent feature uncertainty. Student’s t [50] distribution has heavier tails, which can capture rare or extreme
variations, but its robustness comes at the cost of more
complex parameter estimation. Mixture of Gaussians (MoG)
[51] allows modeling multi-modal uncertainty, accommodating features with multiple plausible states; however, learning

7575

and inference become computationally expensive. Laplace
distributions [52], with sharper peaks and heavier tails than
Gaussian, emphasize outliers but may over-penalize small
deviations, reducing smoothness in optimization. In contrast,
Gaussian distributions provide a balance between expressiveness, computational efficiency, and analytical tractability: their
closed-form distance metrics allow direct modeling of mean
and variance, enabling smooth containment-aware similarity
learning. These properties make Gaussian distributions particularly suitable for modeling visual-textual uncertainty in our
framework.
While prior work primarily focuses on uncertainty within
a single modality or general cross-modal alignment through
symmetric distance metrics, our approach reconceptualizes the
role of probabilistic modeling. We extend it to both image and
text features to explicitly capture complex semantic inclusion
relationships, rather than just distance-based similarity. By
shifting the focus from “point-to-set” matching to a structured containment paradigm, our method enables the model
to characterize the partial-order nature of vision-language
alignment. This probabilistic matching and containment mechanism allows for more precise and robust retrieval, effectively
addressing the inherent ambiguity and the multi-level correspondences shared by TBPR and other advanced multimodal
understanding tasks.
III. M ETHOD
In this section, we present the Probabilistic Distribution
Alignment (PDA) framework, whose overview is illustrated
in Fig. 2. We begin by introducing probabilistic distribution
representations for images and texts, followed by a detailed
description of the key modules: Distributional Representation
Modeling (DRM) and Cross-/Intra-Modal Containment (CMC
and IMC).
A. Feature Extraction Backbone
Consistent with most recent works in this community, we
initialize PDA with the full CLIP image and text encoder
where the image encoder and text encoder are both 12-layer
transformer blocks.
1) Image Encoder: Given an input image, we first resize
it to 384 × 128 pixels, denoted as I ∈ RH×W×C . A CLIP
pre-trained Vision Transformer (ViT) model is then employed
to extract the image features. The image I is divided into a
sequence of N = H×W
P2 non-overlapping patches of size 16×16,
resulting in N = 196 patches. Each patch is subsequently
N
mapped into 1D tokens { fiv }i=1
via a 2D convolutional layer. By
adding positional embeddings and an additional [CLS] token,
v
the token sequence { fcls
, f1v , . . ., fNv } is fed into L transformer
layers. The final image representation is obtained through a
linear projection. For the i-th image, we denote its representation as fIvi = { f0vi , f1vi , . . ., fNvi }, where f0vi represents the global
feature (corresponding to the [CLS] token), and { f1vi , . . ., fNvi }
denote the local patch-level features.
2) Text Encoder: For an input textual description T , we
employ the CLIP text encoder to extract its features. Following
CLIP and IRRA, the text is first tokenized using lower-cased

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:29 UTC from IEEE Xplore. Restrictions apply.

7576

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 6, JUNE 2026

Fig. 2. The framework of PDA. PDA models image and text as well as their mask forms as gaussian distributions, then narrows the distance between image
and text probabilistic representations while exploring the inclusion relationship within and between modalities.

byte pair encoding (BPE) with a vocabulary size of 49,152.
The token sequence is then enclosed with special tokens [SOS]
and [EOS] to indicate the start and end of the sequence,
t
t
respectively. The resulting token embeddings { f sos
, f1t , . . ., feos
}
are passed through transformer layers followed by a linear
projection to obtain the final Euclidean text representations.
Similarly, for the j-th text input, we denote its representation
t
t
t
t
t
as fTj = { f0 j , f1 j , . . ., f Mj }, where f0 j corresponds to the global
tj
tj
text feature and { f1 , . . ., f M } correspond to local token-level
features.
B. Distributional Representation Modeling
Pedestrian retrieval faces significant challenges from inherent uncertainties, such as ambiguous descriptions and varying
visual appearances. The Distributional Representation Modeling (DRM) module (Fig. 2) addresses this by transforming
image and text features into Gaussian distributions. This
choice is theoretically grounded in the maximum entropy
principle, as the Gaussian distribution provides the most
conservative and unbiased representation of cross-modal
uncertainty when only the first two moments are known. Crucially, the mathematical closure and analytical tractability of
Gaussian distributions enable the derivation of a closed-form
containment loss, facilitating smooth and computationally efficient similarity learning. By leveraging these properties, the
framework ensures numerically stable gradient backpropagation and a well-defined semantic manifold, allowing for precise
modeling of the complex inclusion relationships between
visual and linguistic modalities.
The derivation of the visual and textual probabilistic representations is as Fig 3. For the i-th image, we start with
the backbone output fIvi and extract the [CLS] token f0vi . The
mean µvi is obtained from f0vi through a sequence of sigmoid,
Linear layer, LayerNorm (LN), and L2 projection, while the
variance σ2vi is obtained solely via a Linear layer. The image is

Fig. 3. The derivation of the visual and textual probabilistic representations
is as follows: the mean µ is obtained from the deterministic representation
through a sequence of sigmoid, Linear layer, LayerNorm (LN), and L2
projection, while the variance δ is obtained solely via a Linear layer.

thus represented as a Gaussian distribution Zvi ∼ N (µvi , σ2vi ).
t
t
Similarly, for the j-th text, f0 j from fTj is processed to compute
the mean µt j through sigmoid, Linear, LN, and L2 projection,
and the variance σ2t j via a Linear layer, forming the Gaussian
representation Zt j ∼ N (µt j , σ2t j ).
Having modeled the image and text features as Gaussian
distributions Zvi ∼ N (µvi , σ2vi ) and Zt j ∼ N (µt j , σ2t j ), the next
step is to quantify the similarity between these probabilistic
representations. For pedestrian retrieval, we aim to ensure that
the distance between distributions is small for certain matching
image-text pairs, and large for non-matching or ambiguous
pairs. To achieve this, we adopt the following distance metric:
2

d(Zvi , Zt j ) = µvi − µt j 2 + σ2vi + σ2t j .

(1)

The distance metric defined in Eq. 1 is theoretically grounded
in the expected squared Euclidean distance between two
independent Gaussian random variables. This formulation
ensures that the visual-textual similarity is modulated by
their respective uncertainties. From an information-geometric
perspective, this allows the model to map semantic ambiguity
directly into the distance manifold, where samples with higher

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:29 UTC from IEEE Xplore. Restrictions apply.

YANG et al.: PROBABILISTIC DISTRIBUTION ALIGNMENT FOR TEXT-BASED PERSON RETRIEVAL

variance are assigned a larger “effective radius”, preventing
them from forming erroneous correspondences with specific,
low-variance samples.
This formula ensures that the distance between images and
texts with low uncertainty is primarily determined by the
difference in the means of their probabilistic representations,
while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
LDRM to align image-text pairs:
X
LDRM = −
qi j log sigmoid (−a · d + b)
i, j


− (1 − qi j ) log 1 − sigmoid (−a · d + b) ,

(2)

where qi j ∈ {0, 1} indicates whether the image-text pair
matches (qi j = 1) or not (qi j = 0), and d denotes their distance.
The logits −a · d + b (with a > 0) are designed to modulate
the influence of d on the matching probability. Specifically,
for qi j = 1, the objective is to minimize d by maximizing the
logit; conversely, for qi j = 0, the model seeks to increase d to
suppress the matching probability. The negative sign ensures
proper minimization behavior within a binary cross-entropy
formulation, while a and b control the sensitivity and bias
of the response. Parameters a and b are learnable, with their
initial values set to 1. In this way, DRM transforms point-based
representations into probabilistic distribution representations.
C. Cross-Modal Containment
In pedestrian retrieval, text descriptions such as “a person in a jacket” are expected to generalize across multiple
images. To align with this semantic property, we propose Cross-Modal Containment (CMC). Unlike traditional
uncertainty-aware approaches that treat variance merely as a
noise margin for alignment, CMC explicitly models the partialorder relationship, encouraging the text feature distribution
to probabilistically include the corresponding image feature
distribution.
Given a positive image–text pair (Ii , T j ), we model their
feature embeddings as Gaussian distributions:
Zvi ∼ N (µvi , σ2vi ),

Zt j ∼ N (µt j , σ2t j ),

(3)

where µvi and µt j denote the mean features, and σ2vi , σ2t j represent variances that capture feature uncertainty and semantic
scope. To reflect the intuition that textual descriptions are
semantically broader than single images, we define an inclusion measure that quantifies whether the image distribution Zvi
is contained within the text distribution Zt j . The log-inclusion
score is defined as:
Z ∞
H(Zvi ⊂ Zt j ) = log
p2vi (x) pt j (x) dx
−∞
Z ∞
− log
pvi (x) p2t j (x) dx.
(4)
−∞

The first term evaluates how much of the image density pvi (x)
is concentrated within pt j (x), while the second term measures
how dominant pt j (x) is over pvi (x). A positive value of H
indicates that the image distribution is well contained within
the text distribution. Intuitively, the first term is large when pvi

7577

is well aligned with pt j and has a moderate variance, ensuring
sufficient overlap, while the second term is reduced due to the
sharpening effect of squaring pt j , which decreases its overlap
with pvi . This balance between concentration and sharpening
allows H(Zvi ⊂ Zt j ) to be positive, reflecting that the image
distribution is effectively captured by the text distribution.
The log-inclusion score H in Eq. 4 is designed based
on
principle of density concentration. The first term
R ∞ the
2
p
(x)
pt j (x) dx can be interpreted as the cross-correlation
−∞ vi
between the sharpened image distribution and the text distribution. Formally, our containment objective acts as a
probabilistic proxy for partial order modeling in semantic
space. By maximizing H, we enforce a directional constraint
where the text distribution serves as a “probabilistic envelope”
that subsumes the more specific image distribution, aligning
with the asymmetric generality of language relative to visual
content.
Since both pvi (x) and pt j (x) follow Gaussian distributions,
each integral in Eq. (4) can be computed analytically. For
a one-dimensional Gaussian random variable, the general
probability density function is:


(x − µ)2
1
,
(5)
exp −
p(x) = √
2σ2
2πσ
where µ and σ2 denote the mean and variance, respectively.
Squaring the density gives:


1
(x − µ)2
p2 (x) =
exp
−
.
(6)
2π σ2
σ2
By substituting the corresponding (µvi , σvi ) and (µt j , σt j ) into
this form, the integrals in Eq. (4) can be expressed and solved
in closed form. The exponent of the integrand can be rewritten
as a quadratic form:
(x − µt j )2
(x − µvi )2
+
= Ax2 −Bx + C,
σ2vi
2σ2t j
A=

(7)

µ2t j
µt j
µ2vi
1
2µvi
1
+
,
B
=
+
,
C
=
+
.
σ2vi
σ2vi
σ2vi
2σ2t j
σ2t j
2σ2t j

(8)

Completing the square yields and using the Gaussian integral identity:

2
B
B2
2
− Ax + Bx−C = −A x −
+
− C,
(9)
2A
4A
r
Z ∞
B 2
π
e−A( x− 2A ) dx =
.
(10)
A
−∞
The first integral in Eq. (4) becomes:
r
 2

Z ∞
π
1
B
exp
−
C
.
p2vi (x)pt j (x) dx =
(2π)3/2 σ2vi σt j A
4A
−∞
(11)
Taking the logarithm and omitting constants independent of
µ and σ, we obtain:
Z ∞
log
p2vi (x)pt j (x) dx = −2 log σvi − log σt j
−∞

2

B
− 21 log A + 4A
− C.

(12)

An analogous expression is derived for the second integral in
Eq. (4) by swapping vi and t j .

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:29 UTC from IEEE Xplore. Restrictions apply.

7578

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 6, JUNE 2026

Finally, the containment loss is defined as

E. Training and Inference


LCMC = − log sigmoid(c H(Zvi ⊂ Zt j ))

− log sigmoid(c H(Zvi ⊂ Zmask
)) ,
tj

(13)

where c > 0 is a scaling factor and Zmask
denotes the
tj
masked text feature distribution. By combining both complete
and masked text distributions, CMC encourages the image
representation to align with a semantically broader textual
space, ensuring robust cross-modal correspondence even when
textual cues are partially missing.
Although the assumption that the text distribution encompasses the image distribution may not strictly hold in all
cases, our formulation is probabilistic and asymmetric, not
a hard constraint. The inclusion score H(Zvi ⊂ Zt j ) measures
the degree of containment rather than enforcing strict inclusion. This design ensures stable gradients even when textual
descriptions are highly specific or when images contain richer
contextual information. Furthermore, since both image and text
features are modeled as Gaussians with learnable variances,
the model dynamically adjusts its containment confidence
based on the relative semantic scopes of the two modalities,
providing a flexible and effective inductive bias for crossmodal alignment.

D. Intra-Modal Containment
Occlusions and incomplete descriptions (e.g., missing adjectives) are prevalent in TBPR. Intra-Modal Containment (IMC)
enhances robustness by ensuring masked distributions encompass unmasked ones, reflecting increased uncertainty in partial
inputs.
For images, Iimask occludes 25% of Ii , processed to Zmask
∼
vi
mask 2
N (µmask
,
σ
).
For
text,
20%
of
tokens
in
T
are
masked
j
vi
vi
(e.g., “blue jacket” to “[MASK] jacket”), yielding Zmask
∼
tj
mask
mask 2
N (µt j , σt j
), with ratios tuned for balance. The IMC loss,
akin to CMC, ensures masked versions have broader variances.
Similar to Eq. 13, LIMC is formulated as:

LIMC = − log sigmoid(cH(Zvi ⊂ Zmask
))
vi


− log sigmoid(cH(Zt j ⊂ Zmask
))
.
(14)
tj
The primary objective of LIMC is to enhance intra-modal
robustness, ensuring that the unmasked feature distributions
Zvi and text Zt j are contained within their corresponding
. This containment reflects
masked versions, Zmask
and Zmask
tj
vi
that masked data, which may omit certain attributes, should
exhibit greater uncertainty, thereby naturally encompassing the
distribution of the unmasked data. It is worth noting that,
although a transitive containment relation might exist between
cross-modal features, explicitly enforcing Zvi ⊂ Zmask
in
tj
the cross-modal containment loss provides direct supervision,
leading to stronger gradients and more stable convergence.
In contrast, the IMC loss focuses solely on modality-internal
consistency, complementing the cross-modal constraints and
improving robustness to missing or ambiguous attributes
within each modality.

Note that in Eq. 13, the second term explicitly models
the cross-modal robustness to incomplete text descriptions
commonly observed in TBPR. Specifically, it aligns the image
distribution Zvi with the masked text distribution Zmask
, which
tj
introduces uncertainty by randomly masking textual tokens.
On the other hand, we do not incorporate Zmask
in Equation 13,
vi
since it may not be semantically contained within the full
or masked text due to the reduced visual information and
increased ambiguity introduced by image masking. Thus, the
containment relation Zmask
⊂ Zt j or Zmask
⊂ Zmask
cannot
vi
vi
tj
be consistently guaranteed. Consequently, we only incorporate
Zmask
in the CMC loss and handle Zmask
separately in the IMC
tj
vi
loss.
Furthermore, the CMC loss is specifically designed for
positive image-text pairs (Ii , T j ), as emphasized in the main
text. In real-world TBPR scenarios, some images (e.g., I sp ) are
generally well-described by the accompanying text, making
containment relations plausible. However, for samples such
as Iwp or Itn , there might be partial mismatch or noise, for
example, illumination variations affecting perceived clothing
color (e.g., trousers appearing white in bright light). Therefore,
containment assumptions do not strictly hold across all cases.
In practice, our implementation selectively applies the CMC
loss only to semantically aligned positive samples (e.g., I sp ),
ensuring the validity of the containment constraint.
Finally, the overall optimization loss is:
Ltotal = LDRM + LCMC + LIMC .

(15)

LDRM ensures the learning of meaningful Gaussian probability
distributions, LCMC reflects the more general semantic scope
of the text than the image and LIMC enhance the model’s
robustness to occlusions and incomplete descriptions.
During the testing phase, retrieval is performed using only
the mean vector of the learned Gaussian representations, while
the variance vector is discarded. This ensures computational
efficiency and alignment with standard evaluation protocols.
IV. E XPERIMENTS
A. Datasets and Evaluation Metrics
1) Datasets: We evaluate our method on three widely
recognized challenging datasets, namely CUHK-PEDES [1],
ICFG-PEDES [21], RSTPReid [54] and adopt the Rank-k and
mAP as metrics to evaluate the performance.
• CUHK-PEDES [1] includes 40,206 images featuring
13,003 identities. Each image comes with two natural
language descriptions, totaling 80,412 sentences. The
descriptions average over 23 words with a vocabulary
of 9,408 words. The dataset splits into a training set of
34,054 images covering 11,003 individuals, a validation
set of 3,078 images spanning 1,000 individuals, and a test
set of 3,074 images across 1,000 individuals.
• ICFG-PEDES [21] offers 54,522 images representing
4,102 identities. Each image is paired with one finegrained description averaging 37.2 words, providing
detailed identity focused text. The dataset divides into a

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:29 UTC from IEEE Xplore. Restrictions apply.

YANG et al.: PROBABILISTIC DISTRIBUTION ALIGNMENT FOR TEXT-BASED PERSON RETRIEVAL

7579

TABLE I
P ERFORMANCE C OMPARISONS W ITH S TATE - OF - THE -A RT M ETHODS ON CUHK-PEDES DATASETS . T HE CLIP IN THE I MAGE B RANCH U SES CLIP-V I T
TO P ROCESS I MAGES , W HILE THE CLIP IN THE T EXT B RANCH U SES CLIP-X FORMER TO E XTRACT T EXT R EPRESENTATION

training set of 34,674 images across 3,102 individuals and
a test set of 19,848 images covering 1,000 individuals.
• RSTPReid [54] features 20,505 images across 4,101
identities. Each person appears in five images, with each
image linked to two descriptions of at least 23 words.
The vocabulary size reaches 2,204 words. The dataset
is divided into a training set spanning 3,701 individuals,
a validation set covering 200 individuals, and a test set
including 200 individuals.
2) Evaluation Metrics: We adopt the popular Rank-k (%)
metric as the primary evaluation standard, where k takes the
values 1, 5, and 10. Rank-k represents the probability that,
given a text description as a query, at least one matching
person image is included among the top-k candidate images.
Additionally, mAP (%) is used as another retrieval metric to
comprehensively evaluate model performance, which complements Rank-k by measuring the overall ranking quality across
all queries.
B. Implementation Details
We utilize the CLIP model as backbone. To ensure a fair
comparison, we employ the same CLIP-ViT/B16 version as
used in most previous works for experiments. During the
training phase, we use random horizontal flipping, random
cropping with padding, and random erasing to enrich the

training images. The image input size is set to 384 × 128,
and the maximum length of input word tokens is limited to 77.
We train our model using the Adam optimizer over 60 epochs,
incorporating a cosine learning rate decay approach. The initial
learning rate is set to 1e-5 for the full model parameters, while
the batch size is fixed at 64. We adopt an early training strategy
with a progressively increasing learning rate.
C. Comparisons With State-of-the-Art Methods
Tables I and III evaluate the effectiveness of our proposed
PDA method across three widely used datasets: CUHKPEDES, ICFG-PEDES, and RSTPReid.
1) Performance Comparisons on CUHK-PEDES: Table I
presents a comprehensive comparison of the proposed PDA
method against recent state of the art approaches on the
CUHK PEDES dataset. PDA achieves a Rank-1 accuracy of
78.07% and an mAP of 69.56%, establishing new benchmarks
under both metrics. Compared with NAM, the second best
method which utilizes Multimodal Large Language Models
(MLLMs) for diverse textual generation, PDA improves Rank1 by 1.25% and mAP by 0.01%. When compared with MUM,
a method that also employs probabilistic modeling to handle
uncertainty in cross modal retrieval, PDA demonstrates a
notable advantage of 3.82% in Rank-1 accuracy. Furthermore,
relative to APTM, a computationally expensive framework

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:29 UTC from IEEE Xplore. Restrictions apply.

7580

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 6, JUNE 2026

TABLE II
P ERFORMANCE C OMPARISONS W ITH S TATE - OF - THE -A RT
M ETHODS ON ICFG-PEDES DATASET

TABLE III
P ERFORMANCE C OMPARISONS W ITH S TATE - OF - THE -A RT
M ETHODS ON RSTPR EID DATASET

2) Performance Comparisons on ICFG-PEDES: The
results of our method on the ICFG PEDES dataset are also
reported in Table II. PDA again achieves the best performance
across all key metrics, with a Rank-1 of 69.27% and an
mAP of 42.04%. Compared to NAM, PDA secures a 2.22%
improvement in Rank-1 and a 0.53% gain in mAP. Against
MUM, our model shows a 3.65% advantage in Rank-1 accuracy, indicating its superior ability to cope with uncertainty.
Additionally, when compared with APTM, the strongest non
CLIP based method, PDA achieves higher performance by
0.76% in Rank-1 and 0.82% in mAP. These improvements
reinforce the generalizability of our approach across datasets
with differing distributions and annotation styles.
3) Performance Comparisons on RSTPReid: To further
validate the domain robustness of PDA, we evaluate it on
the RSTPReid dataset, with results summarized in Table III.
Our method continues to outperform all competing methods in
terms of Rank-1 and mAP. Specifically, PDA surpasses NAM
by 2.20% in Rank-1 and 0.77% in mAP, despite NAM’s strong
performance due to MLLM-enhanced data diversity. Although
slightly lower in Rank-5 and Rank-10, PDA demonstrates
more reliable top-1 retrieval. Furthermore, the performance
margins over APTM are also significant, with PDA improving
Rank-1 by 3.2% and mAP by 1.21%. Remarkably, the gap
between PDA and MUM widens further on this dataset, with
a Rank-1 improvement of 7.3%, highlighting PDA’s enhanced
robustness under domain shifts and its effective handling of
scene diversity.
Across all three benchmarks, our proposed PDA consistently achieves superior performance, outperforming both
VLP-based and non-VLP-based methods. These gains are
attributed to the introduction of probabilistic distribution representations and the containment-driven alignment mechanism,
which together facilitate more robust and fine-grained crossmodal matching. The consistent improvement across diverse
datasets demonstrates PDA’s strong generalization capability and practical potential for real-world text-based person
retrieval scenarios.
D. Domain Generalization Comparison

based on Swin B and BERT backbones, PDA shows improvements of 1.54% in Rank-1 and 2.65% in mAP, all while
maintaining a lighter architecture based on CLIP. These results
validate the effectiveness of our distribution based modeling
and containment driven alignment strategy on large scale real
world benchmarks.

We observe that there are two commonly adopted protocols for evaluating domain generalization performance in
text-based person retrieval. The first protocol evaluates the
model trained exclusively on the source dataset and directly
tests it on the target domain, ensuring that the target data
remains completely unseen during training. The second protocol involves monitoring performance on both the source
and target datasets during training and selecting the bestperforming result on the target domain. However, this latter
strategy may introduce data bias and compromise the reliability of domain generalization. Therefore, to ensure a fair
and rigorous comparison, we adopt the first protocol in our
experiments.
Table IV reports the domain generalization performance
of our proposed PDA method alongside several advanced
baselines. For IRRA, RDE, and NAM, we manually reproduce
the results under a consistent training setting, while for UniPT,
IRLT, and CFAM, we report the values provided in their

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:29 UTC from IEEE Xplore. Restrictions apply.


hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Prototype-Driven Multi-Feature Generation for
Visible-Infrared Person Re-identification
Jiarui Li1 , Zhen Qiu1 , Yilin Yang1, Yuqi Li1 , Zeyu Dong2 , Chuanguang Yang1†

arXiv:2409.05642v1 [cs.CV] 9 Sep 2024

1

Institute of Computing Technology, Chinese Academy of Sciences, China
2
The art & science college, Boston University, USA

Abstract—The primary challenges in visible-infrared person
re-identification arise from the differences between visible (vis)
and infrared (ir) images, including inter-modal and intra-modal
variations. These challenges are further complicated by varying
viewpoints and irregular movements. Existing methods often rely
on horizontal partitioning to align part-level features, which
can introduce inaccuracies and have limited effectiveness in
reducing modality discrepancies. In this paper, we propose
a novel Prototype-Driven Multi-feature generation framework
(PDM) aimed at mitigating cross-modal discrepancies by constructing diversified features and mining latent semantically
similar features for modal alignment. PDM comprises two key
components: Multi-Feature Generation Module (MFGM) and
Prototype Learning Module (PLM). The MFGM generates diversity features closely distributed from modality-shared features
to represent pedestrians. Additionally, the PLM utilizes learnable prototypes to excavate latent semantic similarities among
local features between visible and infrared modalities, thereby
facilitating cross-modal instance-level alignment. We introduce
the cosine heterogeneity loss to enhance prototype diversity for
extracting rich local features. Extensive experiments conducted
on the SYSU-MM01 and LLCM datasets demonstrate that our
approach achieves state-of-the-art performance. Our codes are
available at https://github.com/mmunhappy/ICASSP2025-PDM.
Index Terms—visible-infrared person re-identification, modality discrepancies, instance-level alignment

I. I NTRODUCTION
Person re-identification (ReID), a process of recognizing
individuals across various image datasets taken by different
cameras, commonly focuses on RGB images captured in ideal
daylight conditions. This preference often leads to diminished
effectiveness and unreliable outcomes in low-light or nighttime environments. As a solution to this limitation, especially
for continuous surveillance needs, the domain of visibleinfrared person re-identification (VI-ReID) has emerged as a
key area of research. The growing deployment of intelligent
surveillance cameras, which can switch automatically to infrared mode, has further accelerated progress in this field.
VI-ReID [1] presents a more complex challenge than traditional ReID. It must navigate not only intra-modality variances
but also cross-modality differences that stem from the distinct
imaging techniques of visible (VIS) and infrared (IR) cameras.
Existing approaches [2]–[4] primarily focus on mapping VIS
and IR features into a unified embedding space with the
aim of minimizing cross-modality dissimilarities. Additionally,
Jiarui Li, Zhen Qiu, Yilin Yang, Yuqi Li are interns
† Corresponding author, Email: yangchuanguang@ict.ac.cn

they attempt to address intra-modality variations – caused by
changes in viewpoint, obstruction, and background – by segmenting body features horizontally and aligning them based on
minimal feature distances. Nevertheless, such methods often
neglect the dynamic positioning of body parts, leading to
semantic misalignments that can impair the effectiveness of
ReID.
Some approaches [5]–[7] involve the use of Generative
Adversarial Networks (GANs) to convert infrared or visible images into the opposite modality, thereby bridging the
modality gap. However, these techniques are hampered by
limited training data and the intrinsic noise in the image
transformation process, affecting their overall efficacy.
In this paper, we propose a Prototype-Driven Multi-Feature
Generation (PDM) framework designed to align modal features using two primary strategies: generating diverse features
that closely match in distribution to minimize inter-modal
disparities, and extracting semantically similar local features.
The framework consists of a Multi-Feature Generation Module
(MFGM) and a Prototype Learning Module (PLM).
Specifically, the MFGM employs center-guided pair mining
loss to generate diverse features, reducing modality differences
and enriching the feature representation for PLM. The PLM
assigns weights to modality features based on the similarity
with learnable prototypes, thereby revealing latent semantically similar local features and achieving feature alignment.
Furthermore, we introduce a dual-center separation loss to
enhance the network’s ability to discriminate pedestrian relationships.
Our contributions are twofold:
• We introduce a prototype-driven multi-feature generation
framework, where the MFGM is utilized to generate diverse
features that are distributed closely. The PLM module is
responsible for mining local features by latent semantic similarity between VIS and IR modality features, thus achieving
instance-level feature alignment.
• Extensive experiments conducted on the SYSU-MM01 [8]
and LLCM datasets demonstrate that the proposed method
achieves state-of-the-art performance.
II. R ELATED W ORK
Generally speaking, there are two main categories of methods in VI-ReID: the feature-level methods and the image-level
methods.

Feature-level methods primarily focus on feature learning,
aiming to minimize the disparity between distinct features
and their common analogs in the feature space. For instance,
MSCLNet [9] bolsters the representation of modality-specific
features through a cascaded amalgamation of modality cooperative complementary learning methods. Likewise, FIENet
[3] engages intermediate features and undertakes fine-grained
learning, anchored by identity-constrained feature centers. Despite their efficacy in enhancing performance, these methods
tend to over-rely on global features, thereby neglecting vital
local information, potentially leading to suboptimal results.
Conversely, techniques such as HCT [2] and MAUM [10]
address this issue by employing Part-based Convolutional
Blocks (PCB) to directly extract features from horizontal
partitions. This approach augments feature representation.
Furthermore, HHRG [11] develops a homograph between
the component features of horizontal partitions and global
features, promoting effective alignment of local features and
further elevating saliency. However, the unpredictable movement of pedestrians may result in misalignment of horizontal
component features, which could diminish the effectiveness of
these methods.
Image-level methods primarily revolve around converting
one modality into another to alleviate the cross-modality gap
between Visible (VIS) and Infrared (IR) images. Techniques
such as cmGAN and D2RL utilize Generative Adversarial
Networks (GANs) to minimize these modality differences.
AlignGAN [6] employs GANs for aligning cross-modality
features at both the pixel and feature levels, while FMCNet
[12] implements feature-level modality compensation using
GANs. Moreover, X-modality [13] and MMN [14] introduce
an intermediate modality to bridge the gap between VIS and
IR feature distributions. Nonetheless, these methods still face
challenges in effectively mitigating modality discrepancies.
III. M ETHOD
Motivated by the need to address key challenges in VIReID, we introduce PDM. Our approach aims to overcome
limitations of existing methods that rely on constructing additional intermediate modality images. Instead, we focus on
generating diverse yet closely distributed features to effectively
represent pedestrians and bridge the modality gap. Inspired
by prototype learning, we leverage learnable prototypes to
extract semantically similar local features across modalities,
facilitating modal instance-level alignment.
The network architecture of PDM is depicted in Fig. 1,
consisting of two primary components: the Multi-Feature Generation Module (MFGM) and the Prototype Learning Module
(PLM). Initially, MFGM processes visual (VIS) and infrared
(IR) features extracted by the backbone network to generate
diverse yet closely distributed features. Subsequently, PLM
extracts semantically similar local features across VIS and IR
modalities. These combined local and global features are then
utilized for pedestrian discrimination, guided by various loss
functions during model training.

A. Multi-Feature Generation Module (MFGM)
The MFGM consists of (i) identical branches, illustrated
in Fig. 1. Initially, the feature map (f ) undergoes three
3 × 3 dilated convolutions with dilation rates of 1, 2, and
3, respectively, to capture information from varying receptive
fields. The outputs are then fused, reducing the channel
dimension to one-fourth of its original size. To enhance nonlinear representations, sequential operations include channel
attention (CA), spatial attention (SA), and ReLU activation.
A fully connected (F C) layer aligns the channel dimension
i
with the original feature map (f ). The outputs f+
from all
branches, along with f , are concatenated to form the input for
i
the next stage of the network. The resulting embeddings f+
for each branch are formulated as follows:
f i = (φ13×3 (f ) + φ23×3 (f ) + φ33×3 (f ))

(1)

i
f+
= F C(ReLU([CA(f i ), SA(f i )]))

(2)

where [·, ·] represents concatenation.
Center-Guided Pair Mining Loss. To enhance the diversity
i
of the generated embeddings f+
, we incorporate the centerguided pair mining loss Lcpm , following the DEEN [15]
approach. The Lcpm for the VIS and IR modalities are defined
as:
i,j
j
L(cv , cir , civ+ ) = [D(cjir , ci,j
v+ ) − D(cv , cv+ )

− D(cjv , ckv ) + α]+ .
j
i,j
L(cv , cir , ciir+ ) = [D(cjv , ci,j
ir+ ) − D(cir , cir+ )

− D(cjir , ckir ) + α]+ .

(3)

(4)

where D(·, ·) denotes Euclidean distance. civ and ciir represent
the original feature centers from VIS and IR modalities, while
civ+ and ciir+ are the feature centers for generated embeddings
f v+ and f ir+ . Indices j and k denote distinct identities in
a mini-batch, and [δ]+ = max(δ, 0). The margin term α is
included for balanced optimization.
Therefore, the total Lcpm can be formulated as:
Lcpm = L(cv , cir , civ+ ) + L(cv , cir , ciir+ )

(5)

B. Prototype Learning Module (PLM)
The PLM is illustrated in Fig. 1, utilizing multiple learnable
prototypes to extract semantically similar features from f v
and f ir , each represented in Rh×w×c , where h, w, and c
denote the height, width, and channel dimensions of the feature
maps. We adjust the weights of modality-specific features
based on similarity scores between prototypes and features,
where higher scores signify stronger semantic relevance. This
adaptation enables PLM to effectively capture semantically
similar local features. Specifically, we define a set of learnable
prototypes P = [P1 , P2 , . . . , Pm ] ∈ Rm×c to encapsulate
latent similar features, with Pi ∈ R1×c representing the i-th
prototype and m denoting the total number.
The process of extracting semantically similar local features
using prototypes is consistent for both f v and f ir . For the f v ,

PLM

MFGM

Backbone

ࣦ ௖௛

ࣦ ௧௥௜

ࣦ ୢ௖௦

VIS image

ࣦ ௜ௗ

PLM

MFGM

Backbone

ࣦ ௖௣௠

Embedding Space

IR image

IR Embeddings

VIS+ Embeddings

IR+ Embeddings

Multi-Feature Generation Module
CA

߮ଷ

SA

߮ଶ

CA

߮ଷ

SA

߮ଶ
Addition

Pixel-order

ࢌ૚ା

…

߮ଵ

Pixel-level Feature

FC

C

۷૚

۷૛

۷૜

۷‫ܖ‬

Positional
Encoding

ࢌ

Local Feature
ࡿ૚

FC

ࡿ૛

ࡿ૜

Similarity Score

C

Concatenate

CA

Channel
Attention

ࢌ࢏ା

SA

‫۾‬૚

‫۾‬૛

‫۾‬૜

···

ࢌ

C

···

߮ଵ

Prototype Learning Module

···

VIS Embeddings

ࡿ࢔

‫ܕ۾‬

Learnable Prototype

Spatial
Attention

Matrix Multiplication

Element-wise Multiplication

Fig. 1. The Framework of PDM.

organized pixel-wise as Iv = [I1v , I2v , . . . , Inv ] in Rn×c with
n = h × w, we incorporate position encoding for spatial
consistency. The similarity between Iv and P is calculated,
producing a similarity matrix S ∈ Rm×n , as described in
Eq. 6.
S = σ (P ⊗ Iv )

(6)

where ⊗ denotes matrix multiplication and σ(·) represents the
sigmoid activation function.
Subsequently, by weighting pixel-level features with S, we
obtain semantically similar local features. The process can be
described as follows:
1 X ij
(S ⊙ Iiv )
n i=1 v
n

piv =

(7)

where ⊙ represents element multiplication, and Sij
v represents
the similarity score between the i-th prototype and the j-th
pixel.
Finally, we concatenate the piv with the global feature to
obtain the final feature Fv ∈ R(m+1)c .
Fv = [piv , Fgv ]

where [·] denotes feature concatenation, and Fgv represents
the global feature for the VIS modality. Fv combines latent
semantic similar features and global features. Similarly, this
method is applied to f ir to obtain Fir . The learnable prototype
facilitates cross-modal semantic alignment. The identity loss
Lid is computed using batch-normalized and classified results
derived from Fv and Fir . Additionally, employing the triplet
loss Ltri supervises the global feature, guiding the model in
discerning pedestrian relationships.
Cosine Heterogeneity Loss. The Cosine Heterogeneity
Loss Lch decreases the similarity between each prototypes,
thereby enhancing the diversity of information among semantically similar local features extracted by the prototypes. The
Lch is defined as follows:

(8)

Lch = 1 −

m−1
m
X X
2
cos(Pi IT , Pj IT )
m(m − 1) i=1 j=i+1

(9)

where Pi and Pj denote the i-th and j-th learnable prototypes,
and I represents Iv and Iir .
Dual-Center Separation Loss. We introduce the DualCenter Separation Loss Ldcs to guide the network in discerning
pedestrian relationships. The goal of Ldcs is to draw samples

TABLE I
C OMPARISON WITH CROSS - MODALITY R E ID METHODS ON SYSU-MM01 AND LLCM DATASETS . 1 ST BEST RESULTS ARE IN BOLD .

Datasets
Settings
Method
AlignGAN [6]
DDAG [16]
AGW [17]
MMN [14]
CAJ [18]
DART [19]
MSCLNet [9]
PartMix [20]
SGIEL [21]
DEEN [15]
MSCMNet [22]
HOS-Net [23]
PDM

Publish
ICCV 19
ECCV 20
TPAMI 21
ACM MM 21
CVPR 21
CVPR 22
ECCV 22
CVPR 23
CVPR 23
CVPR 23
arXiv 23
AAAI 24
-

SYSU-MM01
All-search
Indoor-search
Rank-1
mAP
Rank-1
mAP
42.4
40.7
45.9
54.3
54.7
53.0
61.0
67.9
56.5
57.4
68.7
75.1
70.6
66.9
76.2
79.6
69.8
66.8
76.2
80.3
60.6
58.2
65.7
71.7
76.9
71.6
78.4
81.1
77.7
74.6
81.5
84.8
77.1
72.3
82.0
82.9
75.4
72.2
82.3
84.6
78.5
74.2
83.0
85.5
75.6
74.2
84.2
86.7
79.3
76.3
88.7
89.8

belonging to the same identity closer together while distancing
the centers of samples from different identities. We cluster
samples within a distance threshold ρ1 to enhance diversity.
The Ldcs is defined as follows:
1 X
[−ρ1 + kFi − cyi k2 ]+
N i=1

LLCM
IR-to-VIS
Rank-1
mAP
40.3
48.4
43.6
51.8
52.5
58.9
48.8
56.6
52.2
59.8
54.9
62.9
55.1
60.8
56.4
63.2
57.1
63.6

TABLE II
T HE INFLUENCE OF EACH COMPONENT ON THE PERFORMANCE OF THE
PROPOSED PDM.

PLM

Lch

Settings
Ldcs

MFGM

N

Ldcs =

M−1 M
X
X
2
[ρ2 − kcyj − cyk k2 ]+
+
M (M − 1) j=1

(10)

X
X
X
X

X
X

k=j+1

where N denotes the batch size, Fi represents the i-th feature,
yi indicates the i-th pedestrian, cyi is the centroid of yi , M
is the number of centroids, ρ1 signifies the threshold distance
from the sample to its centroid and ρ2 represents the distance
between different centroids.
C. Multi-Loss Optimization
The total loss of the PLM module is as follows:
Lplm = Ltri + Lch + Ldcs

(11)

Besides the Lcpm and Lplm , we further incorporate Lid [18]
to jointly optimize the network by minimizing these three loss
components:
Ltotal = Lid + Lplm + Lcpm

(12)

IV. E XPERIMENT
A. Datasets
We evaluate the performance of our proposed PDM by
comparing it with various state-of-the-art methods on the
SYSU-MM01 [8] and LLCM [15] datasets.
Metrics. In our evaluation, we focus on two pivotal metrics:
Cumulative Matching Characteristics (CMC) and Mean Average Precision (mAP).

VIS-to-IR
Rank-1
mAP
48.0
52.3
51.5
55.3
59.9
62.7
56.5
59.8
60.4
63.2
62.5
65.8
63.9
66.1
64.9
67.9
64.9
67.3

X

X

X

X
X
X
X
X

SYSU-MM01
Rank-1
mAP
64.7
62.0
71.6
66.9
73.0
70.2
75.7
72.2
75.6
71.4
74.2
70.9
79.3
76.3

B. Implementation Details
The PDM framework is implemented using the PyTorch
framework, runs on a single RTX 4090 GPU, utilizing ResNet50 [25] as the backbone. Initial input images are resized to a
consistent dimension of 3 × 384 × 192. Various augmentation
techniques are applied, including random horizontal flipping
and random erasing. The initial learning rate is set to 1 × 10−2
and increased to 1 × 10−1 after 10 epochs. Subsequently, at 80
and 120 epochs, it undergoes further decay to 1 × 10−3 and
1 × 10−4 , respectively, concluding a total training period of
150 epochs. The training process employs the SGD optimizer
with a momentum of 0.9. Additionally, we set the number of
learnable prototypes m to 10.
C. Main Results
As shown in Table I, PDM outperforms competing methods
in cross-modality person re-identification tasks. On the SYSUMM01 dataset, it achieves a rank-1 accuracy of 79.3% and
mAP of 76.2% in the All-search mode, and 88.7% rank-1
accuracy and 89.8% mAP in the Indoor-search mode. On the
LLCM dataset, PDM achieves a rank-1 accuracy of 57.1%

(a) Initial Distance

(b) PLM Distance

(c) MFGM Distance

(d) PDM Distance

(e) Initial Distribution

(f) PLM Distribution

(g) MFGM Distribution

(h) PDM Distribution

Fig. 2. (a-d) illustrate the intra-class and inter-class distances of cross-modality features, with intra-class and inter-class distances represented in blue and green,
respectively. In (e-h), the t-SNE [24] visualizations illustrate the 2D feature distributions, where circles and triangles denote infrared and visible modalities,
and different colors represent pedestrians from distinct categories.

TABLE III
T HE INFLUENCE OF DIFFERENT QUANTITIES OF LEARNABLE PROTOTYPES
ON THE PERFORMANCE OF THE PROPOSED PDM.
(a)

Settings

(b)

All-search

Indoor-search

Rank-1

mAP

Rank-1

mAP

m=6

78.4

75.2

86.5

88.3

m=8

78.6

75.6

85.8

87.8

m = 10

79.3

76.3

88.7

89.8

m = 12

78.1

75.8

85.2

87.1

to-IR mode. This underscores PDM’s superior performance
and effectiveness in handling modality disparities.
(c)

Fig. 3. The visualization results of attention maps. (a) represents the displayed
image, (b) and (c) show the results of baseline and PDM.

and mAP of 63.6% in the IR-to-VIS mode, and 64.9%
rank-1 accuracy and 67.3% mAP in the VIS-to-IR mode.
These results demonstrate PDM’s effectiveness in addressing
modality disparities and its exceptional performance in crossmodality person re-identification tasks. Additionally, on the
SYSU-MM01 dataset, PDM surpasses HOS-Net with a 3.7%
higher rank-1 accuracy and 2.1% higher mAP. In the LLCM
dataset, PDM outperforms HOS-Net by 0.7% in the IR-to-VIS
mode and exhibits a slightly lower mAP by 0.6% in the VIS-

D. Ablation Studies
Effectiveness of each component. The ablation studies
conducted on the SYSU-MM01 dataset, as presented in Table II, demonstrate the effectiveness of PLM and MFGM
components individually and in combination. Including Lch
and Ldcs enhances the model to achieve optimal performance.
Effectiveness of different numbers of learnable prototypes for the PLM. The PLM utilizes learnable prototypes to
discover semantically similar local features across modalities.
Our study explores different numbers of prototypes for the
PLM and finds that performance improves as the number
increases from 6 to 10. However, as shown in Table III,
performance starts to decline beyond 10 prototypes. Setting
the number to 10 achieves the best performance on the SYSUMM01 dataset, leading us to adopt this configuration for the
PLM.

E. Visualization Analysis
Feature Distribution. We conducted an analysis of intraclass and inter-class distance distributions for cross-modality
features on the SYSU-MM01 dataset, as depicted in Fig. 2
(a-d). The mean values, indicated by vertical lines, exhibit a
progressive divergence (δ1 < δ2 < δ3 < δ4 ). By integrating
PLM, we observed an increase in the inter-class distance
and an enlargement of the gap between the average intraclass distance and inter-class distance. Furthermore, with the
incorporation of MFGM, the intra-class distance decreased,
leading to a further enhancement of the gap. Notably, the
combination of both modules resulted in the maximum gap.
To visually demonstrate the discriminative capability of the
PLM, MFGM, and PDM, we conducted t-SNE visualizations
(Fig. 2 (e-h)), which illustrated the clustering of embeddings
per individual. These visualizations reaffirm that the PDM
(Prototype Distribution Mining) approach effectively addresses
intra-modal and inter-modal disparities in cross-modal person
re-identification. By leveraging diverse features that exhibit
close distributions and utilizing learnable prototypes to capture
latent semantic similarities among cross-modal features, PDM
enables a joint representation of pedestrians using multiple
partial features, effectively mitigating both intra-modal and
inter-modal variations. These comprehensive analyses consistently validate the efficiency of our proposed method in the
context of cross-modality person re-identification.
Attention Visualization. Figure 3 illustrates attention maps,
showing that PDM focuses more on pedestrian regions compared to the baseline method. These analyses validate the
effectiveness of PDM in mitigating inter-modal disparities and
capturing semantic similarities among cross-modal features.
V. C ONCLUSION
We propose PDM, a Prototype-Driven Multi-Feature Generation Network for cross-modal person re-identification. PDM
consists of two modules: Multi-Feature Generation Module
(MFGM) and Prototype Learning Module (PLM). MFGM extracts diverse features from modality-specific inputs to enhance
shared information, aligning their distributions with a centerguided pair mining loss. PLM integrates learnable prototypes to weight modality-specific features based on prototype
similarity, facilitating the discovery of semantically similar
local features across modalities for fine-grained alignment. By
combining local and diverse features, PDM effectively mitigates inter-modal and intra-modal discrepancies. Experimental
results on SYSU-MM01 and LLCM datasets demonstrate
PDM’s state-of-the-art performance in person re-identification.
In the future work, we will focus several directions to
improve VI-ReID: (1) applying more advanced attentionbased feature aggregation mechanism [26] for better representation learning ; (2) adopting contrastive learning [27],
[28] to enhance the discriminative ability; (3) introducing
CLIP [29], [30] to promote multi-modality information processing; (4) combining knowledge distillation [31]–[34] for
VI-ReID model compression.

R EFERENCES
[1] N. Huang, J. Liu, Y. Miao, Q. Zhang, and J. Han, “Deep learning for
visible-infrared cross-modality person re-identification: A comprehensive review,” Information Fusion, vol. 91, pp. 396–411, 2023. I
[2] H. Liu, X. Tan, and X. Zhou, “Parameter sharing exploration and heterocenter triplet loss for visible-thermal person re-identification,” IEEE
Transactions on Multimedia, vol. 23, pp. 4414–4425, 2020. I, II
[3] M. Qi, S. Chan, C. Hang, G. Zhang, and Z. Li, “Fine-grained learning
for visible-infrared person re-identification,” in 2023 IEEE International
Conference on Multimedia and Expo (ICME). IEEE, 2023, pp. 2417–
2422. I, II
[4] C. Fu, Y. Hu, X. Wu, H. Shi, T. Mei, and R. He, “Cm-nas: Crossmodality neural architecture search for visible-infrared person reidentification,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 11 823–11 832. I
[5] G.-A. Wang, T. Zhang, Y. Yang, J. Cheng, J. Chang, X. Liang, and Z.-G.
Hou, “Cross-modality paired-images generation for rgb-infrared person
re-identification,” in Proceedings of the AAAI conference on artificial
intelligence, vol. 34, no. 07, 2020, pp. 12 144–12 151. I
[6] G. Wang, T. Zhang, J. Cheng, S. Liu, Y. Yang, and Z. Hou, “Rgb-infrared
cross-modality person re-identification via joint pixel and feature alignment,” in Proceedings of the IEEE/CVF International Conference on
Computer Vision, 2019, pp. 3623–3632. I, II, I
[7] X. Zhang, H. Luo, X. Fan, W. Xiang, Y. Sun, Q. Xiao, W. Jiang,
C. Zhang, and J. Sun, “Alignedreid: Surpassing human-level performance in person re-identification,” 2018. I
[8] A. Wu, W.-S. Zheng, H.-X. Yu, S. Gong, and J. Lai, “Rgb-infrared
cross-modality person re-identification,” in Proceedings of the IEEE
international conference on computer vision, 2017, pp. 5380–5389. I,
IV-A
[9] Y. Zhang, S. Zhao, Y. Kang, and J. Shen, “Modality synergy complement learning with cascaded aggregation for visible-infrared person reidentification,” in European Conference on Computer Vision. Springer,
2022, pp. 462–479. II, I
[10] J. Liu, Y. Sun, F. Zhu, H. Pei, Y. Yang, and W. Li, “Learning
memory-augmented unidirectional metrics for cross-modality person reidentification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 19 366–19 375. II
[11] Y. Feng, F. Chen, J. Yu, Y. Ji, F. Wu, S. Liu, and X.-Y. Jing,
“Homogeneous and heterogeneous relational graph for visible-infrared
person re-identification,” arXiv preprint arXiv:2109.08811, 2021. II
[12] Q. Zhang, C. Lai, J. Liu, N. Huang, and J. Han, “Fmcnet: Feature-level
modality compensation for visible-infrared person re-identification,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 7349–7358. II
[13] D. Li, X. Wei, X. Hong, and Y. Gong, “Infrared-visible cross-modal
person re-identification with an x modality,” in Proceedings of the AAAI
conference on artificial intelligence, vol. 34, no. 04, 2020, pp. 4610–
4617. II
[14] Y. Zhang, Y. Yan, Y. Lu, and H. Wang, “Towards a unified middle modality learning for visible-infrared person re-identification,” in Proceedings
of the 29th ACM International Conference on Multimedia, 2021, pp.
788–796. II, I
[15] Y. Zhang and H. Wang, “Diverse embedding expansion network and
low-light cross-modality benchmark for visible-infrared person reidentification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 2153–2162. III-A, I,
IV-A
[16] M. Ye, J. Shen, D. J. Crandall, L. Shao, and J. Luo, “Dynamic
dual-attentive aggregation learning for visible-infrared person reidentification,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVII 16.
Springer, 2020, pp. 229–247. I
[17] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, and S. C. Hoi, “Deep learning
for person re-identification: A survey and outlook,” IEEE transactions
on pattern analysis and machine intelligence, vol. 44, no. 6, pp. 2872–
2893, 2021. I
[18] M. Ye, W. Ruan, B. Du, and M. Z. Shou, “Channel augmented
joint learning for visible-infrared recognition,” in Proceedings of the
IEEE/CVF International Conference on Computer Vision, 2021, pp.
13 567–13 576. I, III-C
[19] M. Yang, Z. Huang, P. Hu, T. Li, J. Lv, and X. Peng, “Learning with twin
noisy labels for visible-infrared person re-identification,” in Proceedings

of the IEEE/CVF conference on computer vision and pattern recognition,
2022, pp. 14 308–14 317. I
[20] M. Kim, S. Kim, J. Park, S. Park, and K. Sohn, “Partmix: Regularization strategy to learn part discovery for visible-infrared person reidentification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 18 621–18 632. I
[21] J. Feng, A. Wu, and W.-S. Zheng, “Shape-erased feature learning
for visible-infrared person re-identification,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 22 752–22 761. I
[22] K. Cheng, X. Hua, H. Lu, J. Tu, Y. Wang, and S. Wang, “Multi-scale
semantic correlation mining for visible-infrared person re-identification,”
arXiv preprint arXiv:2311.14395, 2023. I
[23] L. Qiu, S. Chen, Y. Yan, J.-H. Xue, D.-H. Wang, and S. Zhu, “Highorder structure based middle-feature learning for visible-infrared person
re-identification,” in Proceedings of the AAAI Conference on Artificial
Intelligence, vol. 38, no. 5, 2024, pp. 4596–4604. I
[24] L. Van der Maaten and G. Hinton, “Visualizing data using t-sne.” Journal
of machine learning research, vol. 9, no. 11, 2008. 2
[25] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in Proceedings of the IEEE conference on computer vision
and pattern recognition, 2016, pp. 770–778. IV-B
[26] C. Yang, Z. An, H. Zhu, X. Hu, K. Zhang, K. Xu, C. Li, and Y. Xu,
“Gated convolutional networks with hybrid connectivity for image
classification,” in Proceedings of the AAAI Conference on Artificial
Intelligence, 2020, pp. 12 581–12 588. V
[27] C. Yang, Z. An, L. Cai, and Y. Xu, “Mutual contrastive learning for
visual representation learning,” in Proceedings of the AAAI Conference
on Artificial Intelligence, vol. 36, no. 3, 2022, pp. 3045–3053. V
[28] C. Yang, Z. An, H. Zhou, F. Zhuang, Y. Xu, and Q. Zhang, “Online
knowledge distillation via mutual contrastive learning for visual recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence,
vol. 45, no. 8, pp. 10 212–10 227, 2023. V
[29] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable
visual models from natural language supervision,” in International
conference on machine learning. PMLR, 2021, pp. 8748–8763. V
[30] C. Yang, Z. An, L. Huang, J. Bi, X. Yu, H. Yang, B. Diao, and
Y. Xu, “Clip-kd: An empirical study of clip model distillation,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 15 952–15 962. V
[31] C. Yang, Z. An, L. Cai, and Y. Xu, “Hierarchical self-supervised
augmented knowledge distillation,” International Joint Conference on
Artificial Intelligence, pp. 1217–1223, 2021. V
[32] C. Yang, H. Zhou, Z. An, X. Jiang, Y. Xu, and Q. Zhang, “Crossimage relational knowledge distillation for semantic segmentation,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 12 319–12 328. V
[33] C. Yang, Z. An, H. Zhou, L. Cai, X. Zhi, J. Wu, Y. Xu, and Q. Zhang,
“Mixskd: Self-knowledge distillation from mixup for image recognition,” in European Conference on Computer Vision. Springer, 2022,
pp. 534–551. V
[34] W. Feng, C. Yang, Z. An, L. Huang, B. Diao, F. Wang, and Y. Xu,
“Relational diffusion distillation for efficient image generation,” in ACM
Multimedia 2024. V


codex
前两篇已经拿到方法章节。PDA 的方法里有一个关键点，作者不是把方差当作噪声缓冲，而是把它讲成“语义范围”和“包含关系”，这属于典型的视角先行；PDM 则更像把已有的“部位对齐不准”改写成“潜在语义原型对齐”。我继续读剩下四篇。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11209802

Rethinking Joint Optimization in Feature
Compression: Insights from Person Re-Identification
1

Changsheng Gao1 , Zhuoyuan Li2 , Li Li2 , Dong Liu2 , Feng Wu2 , Weisi Lin1
Nanyang Technological University, 2 University of Science and Technology of China

Abstract—Joint optimization, which jointly optimizes compression and machine vision algorithms, is widely regarded as an
effective strategy for enhancing compression performance in the
field of coding for machines. However, existing joint optimization
methods usually incorporate a semantics parsing module at
the end of the pipeline, raising a critical question: Does the
performance improvement stem from the joint optimization itself,
or is it primarily driven by the tailed semantics parsing module?
To address this, we disentangle the tailed semantics parsing
module from the joint optimization pipeline by leveraging the
simplicity of the person re-identification task, where semantics
parsing involves deterministic feature matching rather than a
learned neural network. First, we propose a separate optimization
pipeline and two joint optimization pipelines to systematically
investigate the effectiveness of joint optimization. Our findings
reveal that joint optimization alone does not necessarily guarantee performance improvement. Second, we evaluate the influence
of the tailed semantics parsing module by equipping it with
varying capabilities, demonstrating that higher parsing capability
directly correlates with better machine vision performance. These
findings underscore the pivotal role of tailed semantics parsing
in enhancing machine vision performance and challenge the assumption that joint optimization alone drives improvement. This
work offers new insights for designing effective coding methods,
emphasizing the interplay between optimization strategies and
tailed semantics parsing.
Index Terms—coding for machines, feature compression, joint
optimization

I. I NTRODUCTION
Coding for machines has emerged as a vital research area
aimed at addressing the challenges posed by the ever-growing
volume of image and video data. Unlike traditional coding
methods developed for human visual perception, coding for
machines focuses on enabling machines to analyze, understand, and make decisions based on compressed data. However, compression distortions often degrade machine vision
performance, necessitating a shift in focus from human-centric
visual quality to machine-centric accuracy.
To this end, various methods have been proposed [1]–
[35]. According to the optimization strategy, the existing
methods can be broadly grouped into two categories: separately optimized methods and jointly optimized methods.
In separately optimized methods, compression and machine
vision algorithms are optimized independently. For example,
in [1]–[8], the encoder is specialized in preserving semantic
information, whereas method proposed in [9] focuses on
This work was supported by the Ministry of Education of Singapore under
Grant T2EP20123-0006.

optimizing machine vision algorithms to adapt to distorted
images.
In contrast, joint optimization methods aim to simultaneously optimize compression and machine vision algorithms
[18]–[31]. For example, some studies optimize compression
networks using machine vision tasks like classification or
detection as part of the loss function [18], [20], [24]. Similarly,
in [18], the detection network Faster R-CNN [36] is utilized
as the loss function for the compression network. These
approaches have demonstrated improved accuracy for specific
machine vision tasks.
Interestingly, most joint optimization methods follow a
common pattern: the machine vision pipeline is divided into
two parts, referred to as the head and tail, with a compression
network inserted between them. For instance, in [25], the
tail network is fine-tuned to adapt to compressed features
to improve performance improvements across multiple tasks.
This consistent design raises a question: Does the improvement stem from the joint optimization strategy itself, or is
it primarily driven by the tail network’s enhanced semantic
parsing capabilities?
In this study, we challenge the conventional belief that joint
optimization inherently outperforms separate optimization. By
isolating the effects of tail-based semantic parsing within the
joint optimization pipeline, we provide a nuanced perspective
on its actual contribution to performance improvements.
To achieve this, we conduct an investigation using person reidentification (ReID). Specifically, we eliminate the influence
of the tail’s semantic parsing capabilities by removing this
module from the joint optimization pipeline. We then design
two distinct joint optimization strategies and compare their
performance against separate optimization. In addition to evaluating ReID accuracy, we analyze the effects of compression
on extracted features to understand the differential impacts
of these optimization strategies. Furthermore, we enhance the
tail’s semantic parsing capability by improving the decoder,
enabling us to isolate and quantify its role in machine vision
performance. Our contributions are summarized as follows.
• We rigorously investigate the effectiveness of joint optimization in feature compression by disentangling its
components. Our findings reveal the counter-intuitive fact
that joint optimization does not necessarily guarantee
performance improvement
• We design one separate optimization pipeline and two
joint optimization pipelines and analyze their impacts
on machine vision performance. Our results confirm that

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:30 UTC from IEEE Xplore. Restrictions apply.

compression negatively affects feature quality, even under
joint optimization.
• We demonstrate that the tail’s semantic parsing capability
is a critical factor driving performance improvements.
By enhancing the decoder’s semantic parsing ability, we
validate its effectiveness in improving ReID accuracy and
provide a new perspective on the limitations of joint
optimization.
II. BACKGROUND AND M OTIVATION
A. Background and Problem
Joint optimization has emerged as a promising approach in
the field of coding for machines, with the goal of improving
machine vision performance by simultaneously optimizing
compression and machine vision algorithms. Existing joint optimization methods can be categorized into two primary types:
monolithic pipeline and split pipeline. We depict and compare
them in Fig. 2. In the monolithic pipeline, the machine
vision algorithm is applied as a single unit after compression,
either processing reconstructed images or extracted features.
The algorithm serves as a loss function during optimization
to guide the compression process. In the split pipeline, the
machine vision algorithm is divided into two components: the
head (responsible for semantics representation) and the tail
(responsible for semantics parsing). The compression module
is inserted between these components, allowing for direct
optimization of compressed features. Regardless of the specific
approach, tailed semantics parsing is consistently present in
these pipelines. In the monolithic pipeline, the entire machine
vision algorithm inherently includes semantics parsing, while
in the split pipeline, the tail module explicitly handles semantics parsing after feature reconstruction. This reliance on
semantics parsing complicates the evaluation of joint optimization effectiveness, as the observed performance improvements
may stem from the semantics parsing capabilities of the tail
module rather than the joint optimization process itself.
The consistent use of tailed semantics parsing in existing
methods raises an important challenge: How can we isolate
the impact of joint optimization from the influence of tailed
semantics parsing? Addressing this challenge is critical to
understanding the true value of joint optimization in feature
compression.
B. Pipeline Design
To disentangle the contributions of joint optimization from
those of tailed semantics parsing, it is essential to design
a framework that eliminates the influence of the semantics
parsing module. However, this is not feasible for many commonly studied machine vision tasks, such as object detection,
semantic segmentation, etc. These tasks rely heavily on neural
networks for semantics parsing, making it hard to remove
the tailed semantics parsing module without disrupting the
pipeline.
To overcome this limitation, we turn to the person reidentification task. Person re-identification involves two distinct stages: feature extraction and feature matching. Fea-

Enc

Dec

Head

Tail

Dec

Tail

(a)

Head

Enc

(b)
Fig. 1. Illustration of the existing joint optimization-based pipelines. (a)
Monolithic pipeline: append the whole machine vision algorithm to the
compression module. (b) Split pipeline: split the machine vision algorithm
into two parts and insert the compression module between them.

ture extraction is responsible for deriving discriminative and
compact feature representations from input images. Feature
matching compares feature representations of a query image
with those of gallery images using similarity metrics like
Euclidean distance. In the ReID task, the feature matching
stage corresponds to tailed semantics parsing. Unlike other
tasks, it does not involve a learned neural network but instead
relies on deterministic metrics (such as MSE) for comparison.
This simplicity allows the semantics parsing module to be
removed entirely, enabling us to evaluate joint optimization
in isolation.
C. Person Re-identification
To conduct our investigation, we adopt FastReID [37], a
widely used PyTorch-based framework for person ReID. FastReID is designed to deliver high performance while offering
modularity and flexibility, making it an ideal choice for our
study. Its feature extraction consists of two main components:
backbone and aggregation. Backbone includes deep neural
networks such as ResNet, ResNeXt, and MobileNetV2 to
extract discriminative feature representations. The extracted
features are aggregated using methods like Global Average
Pooling (GAP) or Generalized Mean Pooling (GeM).
In this work, we utilize ResNet as the backbone, GeM for
feature aggregation, and Euclidean distance as the similarity
metric. The deterministic nature of the feature matching stage
in FastReID allows us to completely remove the tailed semantics parsing module while maintaining the integrity of the
joint optimization process. By leveraging the simplicity and
modularity of FastReID, we establish a robust framework to
isolate and analyze the contributions of joint optimization in
feature compression pipelines.
III. O PTIMIZATION P IPELINE AND F EATURE
C OMPRESSION M ETHODS
In this section, we present our proposed separate optimization method, two kinds of joint optimization methods, and the
feature compression method.
A. Separate Optimization
As the baseline, we introduce the separate optimization,
denoted as Opt S in Fig. 2. In this method, we separately train

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:30 UTC from IEEE Xplore. Restrictions apply.

Opt_JA

Org Img

Opt_JH

Backbone

Aggregation

Opt_S

Enc
Org
Query Feat

Feature
Extraction

Bits

Dec

Rec
Query Feat
Feature
Matching

Similarity
Measure

Rec Gallery Feats

Fig. 2. Joint optimization framework of person re-identification. Feature extraction: extract features from images and perform feature compression. Feature
matching: compare reconstructed query features with the gallery features to conduct the re-identification.

the FastReID and the feature compression network. Given that
the FastReID model is available as a pre-trained model, our
focus is primarily on training the feature compression network.
The specific structure and details of the compression network
are presented in the following subsection. In Opt S, we use
Mean Squared Error (MSE) as the optimization objective. This
is motivated by the fact that the features extracted by the pretrained FastReID model are inherently compact and discriminative. By optimizing for MSE, we aim to maintain high signal
fidelity, effectively preserving these two characteristics.

B. Joint Optimization
To eliminate the influence of tailed semantics parsing in
joint optimization pipelines, we position the feature compression module immediately before the feature matching stage.
This design ensures that reconstructed features are directly
matched without undergoing additional semantics parsing,
allowing us to isolate and evaluate the effect of joint optimization on ReID performance. We propose two joint optimization
approaches, depicted in Fig. 2.
For the first joint optimization approach, denoted as
Opt JA, we optimize all modules jointly. In contrast to
separate optimization, we optimize the whole framework with
respect to the loss function pertaining to the person reidentification task, which includes cross entropy (CE) and
triplet loss. The application of Opt JA was expected to facilitate more effective collaboration between FastReID and feature
compression and potentially lead to higher performance. However, due to the presence of joint optimization, the influence
stemming from the information capacity constraint (ICC) of
feature compression will propagate back to feature extraction.
This, in turn, could impact the feature extraction process and
result in less compact and discriminative features.
To evaluate the influence caused by the information capacity
constraint, we propose the second joint optimization approach,
denoted as Opt JH. In Opt JH, only the aggregation module
and feature compression module are jointly optimized, while
the parameters in the backbone are frozen to avoid gradient
backpropagation. By comparing Opt JA and Opt JH, we aim
to better understand the trade-offs and challenges introduced
by joint optimization in ReID pipelines.

C. Feature Compression
As our primary objective is to assess the influence of
various optimization strategies rather than design a new feature
compression method, we take a simple but effective feature
compression method. Given that the extracted features are
represented as 1 × 2048 vectors, we employ a simple feature
compression method consisting of two fully connected (FC)
layers and two batch normalization (BN) layers. The first FC
layer serves as the encoder, while the other acts as the decoder.
Considering that the BN layer is applied after each layer in
FastReID, we also incorporate a BN layer after each FC layer.
To fully investigate the impact of joint optimization, we
introduce various information capacity constraints by adjusting
the output channels of the encoder and the input channels of
the decoder. Specifically, we set five compression ratios (CR):
256, 128, 64, 32, and 16. For example, a CR of 16 involves
configuring the output channels of the encoder and the input
channels of the decoder to 128.
It is worth noting that we experimented with various compression structures, including networks with additional layers
and non-linear activation functions such as ReLU. However,
these designs resulted in poorer performance, highlighting the
inadequacy of complex compression methods for the 1 × 2048
features. Our results demonstrate that a simple compression
method is good for preserving the semantic information of
the extracted features.
IV. E XPERIMENTS
A. Experimental Settings
Dataset We use DukeMTMC-reID dataset [38] in our experiments. For joint optimization, the original images are used
for both training and evaluation. For separate optimization, we
use the features extracted from the training images to train
the feature compression network. In the evaluation phase, we
compress the extracted query and gallery features and then
reconstruct them for feature matching.
Training Configuration We adopt sbs R50 ibn configuration for the FastReID [37] network. For joint optimization,
the batch size and number of epochs are set to 128 and
1000, respectively. The backbone is frozen for the first 4000
iterations. The other training parameters are set to their default
values. For separate optimization, we only train the feature

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:30 UTC from IEEE Xplore. Restrictions apply.

TABLE I
P ERSON R E - IDENTIFICATION P ERFORMANCE C OMPARISON BETWEEN
D IFFERENT O PTIMIZATION S TRATEGIES

80
70
mAP

60
50

Opt_S
Opt_JH
Opt_JA
Opt_JH_O
Opt_JA_O

40
30
256

128

64
Compression Ratio

32

16

Fig. 3. Person re-identification performance comparison between different
optimization strategies.

compression network. We set the batch size and number of
epochs to 512 and 200, respectively. The initial learning rate
is set to 0.001 and scheduled with cosine annealing. We use
the Adam algorithm [39] as the gradient optimizer.
B. Optimization Strategy Comparison
We present the performance comparison of Opt S, Opt JA,
and Opt JH in Fig. 3, where the performance is measured by
mean Average Precision (mAP). Generally, all three optimization strategies exhibit a negative correlation between performance and CRs, i.e., lower CRs lead to higher performance.
However, the performance among the strategies varies for
different CRs. For CR 256, both joint optimization strategies
outperform Opt S, with Opt JA achieving the highest performance. For CR 128, all three optimization strategies produce
comparable results. However, for CRs smaller than 128, an
inverse trend is observed where Opt S outperforms the other
strategies. This finding challenges the common belief that
joint optimization consistently improves performance across
all CRs. Instead, it appears to be effective in extreme high
CRs but loses its advantage in medium and low CRs. In this
subsection, we present this observation, and further analyze it
from two perspectives in the subsequent subsections.
C. Influence of Feature Compression on Feature Extraction
In this subsection, we analyze the influence of feature compression on feature extraction from two perspectives: ReID
performance analysis and feature analysis.
ReID Performance Analysis We compare the person reidentification performance of the original features (the input
of the encoder) with that of the reconstructed features in
Fig. 3 and Table I. For Opt JH, the gradient backpropagation
is truncated between the backbone module and aggregation
module. As a result, the information capacity constraint does
not affect the feature extraction. Therefore, the original features in Opt JH remain as compact and discriminative as
those extracted from the pre-trained FastReID model. The
performance of the original features in Opt JH is denoted as

CR

Opt S

Opt JH

Opt JA

Opt JH O

Opt JA O

256
128
64
32
16

26.54
57.44
73.48
79.05
80.50

35.33
56.51
68.57
74.74
76.37

40.29
56.16
65.89
71.89
72.41

81.34
81.29
80.96
80.83
80.8

54.36
58.15
59.04
61.58
61.61

Opt JH O in Fig. 3, showing consistently high performance
across different CRs.
In contrast, for Opt JA, the entire FastReID network is
jointly trained with feature compression, allowing the feature
compression to influence the entire feature extraction process.
The performance of the original features in Opt JA is denoted as Opt JA O in Fig. 3. We observe two key findings.
First, the performance of Opt JA O is notably lower than
that of Opt JH O, indicating that the information capacity
constraint negatively impacts the feature extraction process.
Second, the performance varies across different CRs, with
lower CRs achieving higher performance. This is attributed to
the preservation of more information with higher information
capacity.
Feature Analysis In addition to evaluating the performance,
we use t-Distributed Stochastic Neighbor Embedding (t-SNE)
technique to visualize the original features in a 2-dimensional
space and obtain insight into how compression affects the
features. Fig. 4 presents the visualizations for 20 randomly
selected identities, represented in different colors. Each optimization strategy is visualized for three CRs. Ideally, samples
from the same identity (color) should be as concentrated as
possible (high compactness), while samples from different
identities should be as separated as possible (high distinctiveness). The first row and the second row show the visualizations of the original features extracted from Opt JH and
Opt JA, respectively. For Opt JH, although the visualizations
differ among CRs, the compactness of specific identities and
distinctiveness between identities remain similar, indicating
that feature compression has only a minor impact on feature
extraction in Opt JH. In contrast, for Opt JA, noticeable
variation is observed along CRs. Higher CRs result in poorer
feature compactness and distinctiveness. For example, the blue
and black identities become more distracted as CR increases.
In general, the feature compactness of Opt JH is better than
that of Opt JA. Based on the analysis, we can conclude that
feature compression negatively impacts feature extraction and
degrades the final performance.
D. Feature Compression Analysis
In this subsection, we analyze feature compression by
comparing the original features and the reconstructed features. For Opt JH, the original features consistently achieve
high performance, while the performance of the reconstructed
features depends on the CRs. This difference arises because
the original features contain substantial semantics, which are

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:30 UTC from IEEE Xplore. Restrictions apply.

Org_Opt_JH_16
1.0
0.8
0.6
0.4
0.2
0.00.0 0.2 0.4 0.6 0.8 1.0

1.0
0.8
0.6
0.4
0.2
0.00.0

Org_Opt_JA_16
1.0
0.8
0.6
0.4
0.2
0.00.0 0.2 0.4 0.6 0.8 1.0

1.0
0.8
0.6
0.4
0.2
0.00.0

1.0
0.8
0.6
0.4
0.2
0.00.0

Rec_Opt_JA_16

0.2

0.4

0.6

0.8

1.0

1.0
0.8
0.6
0.4
0.2
0.00.0

Org_Opt_JH_64

0.2

0.4

0.6

0.8

1.0

1.0

Org_Opt_JA_256
1.0
0.8
0.6
0.4
0.2
0.00.0 0.2 0.4 0.6 0.8 1.0

1.0

1.0
0.8
0.6
0.4
0.2
0.00.0

Org_Opt_JA_64

0.2

0.4

0.6

0.8

Rec_Opt_JA_64

0.2

0.4

0.6

0.8

Org_Opt_JH_256

1.0
0.8
0.6
0.4
0.2
0.00.0

0.2

0.4

0.6

0.8

1.0

Rec_Opt_JA_256

0.2

0.4

0.6

0.8

1.0

Fig. 4. t-SNE visualization of original features and reconstructed features for Opt JA and Opt JH. First row: original features extracted from the Opt JH
pipeline. Second row: original features extracted from the Opt JA pipeline. Third row: reconstructed features from the Opt JA pipeline.

TABLE II
P ERFORMANCE COMPARISON BETWEEN THE BASE DECODER AND THE
ENHANCED DECODER

Decoder

Opt S

Opt JH

Opt JA

Base
Enhanced

25.96
25.96

35.44
36.10

40.07
41.66

the CR 256, the reconstructed features fail to achieve higher
performance. This issue is attributed to two reasons: first, the
CR is too large to preserve sufficient semantic information,
and second, the decoder may not be powerful enough to fully
parse the semantics carried in the compressed features.
E. Tailed Semantics Parsing Analysis

partly lost during feature compression. In the case of Opt JA,
the improvements brought by joint optimization vary with
different CRs. It is worth mentioning that the performance
is not solely determined by the amount of preserved semantic
information but also by the way the semantic information is
represented. For the CR 256, the reconstructed features achieve
lower performance due to substantial semantic information
loss in the feature compression. Conversely, for CRs smaller
than 128, the semantic information is better preserved. The
decoder in Opt JA acts as a semantics parser, reconstructing
the features in a better representation, leading to higher
performance.
We visualize the reconstructed features of Opt JA in Fig. 4.
For the low CRs, especially the CR 16, feature compression
improves the compactness and distinctiveness of the original
features. This phenomenon verifies the decoder’s ability in semantics parsing. It is important to note that the jointly trained
decoder is optimized to achieve high person re-identification
accuracy. This characteristic equips the whole compression
network with the capability of semantics parsing. However, for

In this subsection, we demonstrate that tailed semantics
parsing is the key to performance improvement. Since the
feature matching is fixed as Euclidean distance, we introduce
extra semantics parsing capability into the decoder.
We propose to enhance the semantics parsing capability of
the decoder by adding two additional fully-connected layers.
Since the performance of the reconstructed features is higher
than that of the original features for most CRs, we conduct
this experiment specifically for the CR of 256. Specifically, the
modified decoder consists of three fully-connected layers, each
followed by a batch normalization layer. The input and output
dimensions of the three layers are (8, 64), (64, 512), and (512,
2048), respectively. We jointly optimize the FastReID network
and the modified feature compression network using the same
training strategy.
The performance comparison between the different decoders
is presented in Table II. The results indicate that the enhanced
decoder outperforms the base decoder in both joint optimization strategies. This finding emphasizes the significance of
semantics parsing for achieving performance improvement in
joint optimization strategy.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:30 UTC from IEEE Xplore. Restrictions apply.

V. C ONCLUSION
In this paper, we challenge the prevailing assumption that
joint optimization inherently improves compression performance in the field of coding for machines. To examine the
effectiveness of joint optimization, we designed one separate
optimization pipeline and two joint optimization pipelines
tailored to the person re-identification task. By isolating the
tailed semantics parsing module, our findings reveal that
joint optimization can negatively impact the feature extraction
process. Moreover, we demonstrate that the tailed semantics
parsing module plays a critical role in joint optimization,
as its enhanced capabilities directly improve machine vision
performance. These findings offer valuable insights into the
mechanics of joint optimization, providing a deeper understanding of its limitations and potential. Our work highlights
the importance of incorporating a robust semantics parsing
module after the compression stage to maximize system performance. We hope this study serves as a foundation for future
research, enabling researchers to more effectively leverage
joint optimization in the field of coding for machines.
R EFERENCES
[1] Xin Li, Jun Shi, and Zhibo Chen, “Task-driven semantic coding via
reinforcement learning,” IEEE Transactions on Image Processing, vol.
30, pp. 6307–6320, 2021.
[2] Hyomin Choi and Ivan V Bajić, “Deep feature compression for
collaborative object detection,” in ICIP. IEEE, 2018, pp. 3743–3747.
[3] Hyomin Choi and Ivan V Bajic, “Near-lossless deep feature compression
for collaborative intelligence,” in MMSP. IEEE, 2018, pp. 1–6.
[4] Jinyoung Choi and Bohyung Han, “Task-aware quantization network
for JPEG image compression,” in ECCV. Springer, 2020, pp. 309–324.
[5] Qi Cai et al., “A novel video coding strategy in HEVC for object
detection,” IEEE Transactions on Circuits and Systems for Video
Technology, vol. 31, no. 12, pp. 4924–4937, 2021.
[6] Ling-Yu Duan et al., “Optimizing JPEG quantization table for low bit
rate mobile visual search,” in VCIP. IEEE, 2012, pp. 1–6.
[7] Zhijing Li, Christopher De Sa, and Adrian Sampson, “Optimizing JPEG quantization for classification networks,” arXiv preprint
arXiv:2003.02874, 2020.
[8] Changsheng Gao, Dong Liu, Li Li, and Feng Wu, “Towards taskgeneric image compression: A study of semantics-oriented metrics,”
IEEE Transactions on Multimedia, vol. 25, pp. 721–735, 2023.
[9] Weiqian Wang, Ping An, Chao Yang, and Xinpeng Huang, “Intermediate
deep-feature compression for multitasking,” in Optoelectronic Imaging
and Multimedia Technology VI. SPIE, 2019, vol. 11187, pp. 206–212.
[10] Changsheng Gao, Yiheng Jiang, Li Li, Dong Liu, and Feng Wu,
“DMOFC: Discrimination metric-optimized feature compression,” in
PCS, 2024, pp. 1–5.
[11] Changsheng Gao, Yiheng Jiang, Siqi Wu, Yifan Ma, Li Li, and Dong
Liu, “IMOFC: Identity-level metric optimized feature compression for
identification tasks,” IEEE Transactions on Circuits and Systems for
Video Technology, pp. 1–1, 2024.
[12] Qiaoxi Chen, Changsheng Gao, and Dong Liu, “End-to-end learned
scalable multilayer feature compression for machine vision tasks,” in
ICIP, 2024, pp. 1781–1787.
[13] Yuan Tian, Guo Lu, Guangtao Zhai, and Zhiyong Gao, “Non-semantics
suppressed mask learning for unsupervised video semantic compression,” in ICCV, 2023, pp. 13564–13576.
[14] Yuan Tian, Guo Lu, and Guangtao Zhai, “Free-VSC: free semantics
from visual foundation models for unsupervised video semantic compression,” in ECCV, Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga
Russakovsky, Torsten Sattler, and Gül Varol, Eds., Cham, 2025, pp.
163–183, Springer Nature Switzerland.
[15] Bardia Azizian and Ivan V. Bajić, “Privacy-preserving autoencoder for
collaborative object detection,” IEEE Transactions on Image Processing,
vol. 33, pp. 4937–4951, 2024.

[16] Jaakko Laitinen, Tero Partanen, Alexandre Mercat, Jarno Vanne, Miska
Hannuksela, Honglei Zhang, Alireza Aminlou, and Francesco Cricri,
“Feasibility study of multi-layer vvc coding scheme for hybrid machinehuman consumption,” in ICME, 2024, pp. 1–6.
[17] Ying Tang, Wei Yang, Junqing Yu, and Zikai Song, “Agnostic feature
compression with semantic guided channel importance analysis,” in
ICME, 2024, pp. 1–6.
[18] Lahiru D Chamain, Fabien Racapé, Jean Bégaint, Akshay Pushparaja,
and Simon Feltman, “End-to-end optimized image compression for
machines, a study,” preprint arXiv:2011.06409, 2020.
[19] Sihui Luo et al., “DeepSIC: Deep semantic image compression,” in
ICONIP. Springer, 2018, pp. 96–106.
[20] Neel Patwa et al., “Semantic-preserving image compression,” in ICIP.
IEEE, 2020, pp. 1281–1285.
[21] Kristian Fischer, Fabian Brand, and Andre; Kaup, “Boosting neural
image compression for machines using latent space masking,” IEEE
Transactions on Circuits and Systems for Video Technology, pp. 1–1,
2022.
[22] Hyomin Choi and Ivan V Bajić, “Scalable image coding for humans
and machines,” IEEE Transactions on Image Processing, vol. 31, pp.
2739–2754, 2022.
[23] Amir Erfan Eshratifar, Amirhossein Esmaili, and Massoud Pedram, “Towards collaborative intelligence friendly architectures for deep learning,”
in International Symposium on Quality Electronic Design. IEEE, 2019,
pp. 14–19.
[24] Saurabh Singh et al., “End-to-end learning of compressible features,”
in ICIP. IEEE, 2020, pp. 3349–3353.
[25] Ruoyu Feng et al., “Image coding for machines with omnipotent feature
learning,” in ECCV. Springer, 2022, pp. 510–528.
[26] Kiran Misra, Tianying Ji, Andrew Segall, and Frank Bossen, “Video
feature compression for machine tasks,” in ICME. IEEE, 2022, pp. 1–6.
[27] Robert Henzel, Kiran Misra, and Tianying Ji, “Efficient feature compression for the object tracking task,” in ICIP. IEEE, 2022, pp. 3505–3509.
[28] Ning Yan, Changsheng Gao, Dong Liu, Houqiang Li, Li Li, and Feng
Wu, “SSSIC: Semantics-to-signal scalable image coding with learned
structural representations,” IEEE Transactions on Image Processing, vol.
30, pp. 8939–8954, 2021.
[29] Lingyu Zhu, Binzhe Li, Riyu Lu, Peilin Chen, Qi Mao, Zhao Wang,
Wenhan Yang, and Shiqi Wang, “Learned image compression for both
humans and machines via dynamic adaptation,” in ICIP, 2024, pp. 1788–
1794.
[30] Xu Zhang, Peiyao Guo, Ming Lu, and Zhan Ma, “All-in-one image
coding for joint human-machine vision with multi-path aggregation,” in
The Thirty-eighth Annual Conference on Neural Information Processing
Systems, 2024.
[31] Takahiro Shindo, Kein Yamada, Taiju Watanabe, and Hiroshi Watanabe,
“Image coding for machines with edge information learning using
segment anything,” in ICIP, 2024, pp. 3702–3708.
[32] Jinming Liu, Ruoyu Feng, Yunpeng Qi, Qiuyu Chen, Zhibo Chen,
Wenjun Zeng, and Xin Jin, “Rate-distortion-cognition controllable
versatile neural image compression,” in ECCV, Aleš Leonardis, Elisa
Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol,
Eds., Cham, 2025, pp. 329–348, Springer Nature Switzerland.
[33] Jinming Liu, Xin Jin, Ruoyu Feng, Zhibo Chen, and Wenjun Zeng,
“Composable image coding for machine via task-oriented internal adaptor and external prior,” in VCIP, 2023, pp. 1–5.
[34] Xuelin Shen, Haoqiao Ou, and Wenhan Yang, “Image coding for
machine via analytics-driven appearance redundancy reduction,” in ICIP,
2024, pp. 1883–1889.
[35] Zhongpeng Zhang and Ying Liu, “Side information driven image coding
for machines,” in PCS, 2022, pp. 193–197.
[36] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun, “Faster
R-CNN: Towards real-time object detection with region proposal networks,” in Neurips, 2015, pp. 91–99.
[37] Lingxiao He et al., “Fastreid: A pytorch toolbox for general instance
re-identification,” preprint arXiv:2006.02631, 2020.
[38] Zhedong Zheng, Liang Zheng, and Yi Yang, “Unlabeled samples
generated by gan improve the person re-identification baseline in vitro,”
in ICCV, 2017, pp. 3754–3762.
[39] Diederik P Kingma and Jimmy Ba, “Adam: A method for stochastic
optimization,” preprint arXiv:1412.6980, 2014.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:30 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Prototype-guided Knowledge Propagation with Adaptive Learning for
Lifelong Person Re-identification
Zhijie Lu1 , Wuxuan Shi1 , He Li1∗ , Mang Ye1,2∗
1
School of Computer Science, Wuhan University, Wuhan, China
2
Taikang Center for Life and Medical Sciences, Wuhan University, Wuhan, China
{zhijielu, wuxuanshi, lihe404, yemang}@whu.edu.cn
Abstract
Lifelong Person Re-identification (LReID) is essential in dynamic camera networks, which continually adapts to new environments while preserving previously acquired knowledge. Existing LReID techniques often preserve samples from
past datasets to maintain old knowledge, potentially leading to privacy risks. While prototypebased methods offer privacy advantages, current
approaches primarily focus on adjusting classifiers
for image classification tasks, neglecting representation biases between old and new identities
in person re-identification. This study introduces
a novel Prototype-guided Knowledge Propagation
(PKP) method, which mitigates discrepancies in
similar identity images between old and new tasks
by guiding prototype construction through triplet
loss constraints. Additionally, to address disparities between prototypes and the updated feature
extractor, an Adaptive Parameter Evolution (APE)
strategy is proposed. APE optimizes the integration of the old and new models by assessing the
importance of the new tasks, dynamically selecting the most pertinent parameters for updates according to their contribution to the current task.
Extensive experiments on the LReID benchmark
demonstrate that our approach surpasses state-ofthe-art prototype-based LReID methods in terms
of mAP and rank-1 accuracy. Code is available at
https://github.com/joyner-7/IJCAI2025-PKA.

1

Introduction

Person re-identification (ReID) is a fundamental task in computer vision that aims to match the same person across different locations and times [Ye et al., 2021; Leng et al., 2019;
Ye et al., 2024]. Traditional ReID methods have achieved
outstanding results by leveraging deep learning models and
large-scale static datasets, where all training data are available simultaneously [Li et al., 2024; Dai et al., 2018; Zhang
et al., 2016; Ye et al., 2018]. However, in real-world scenarios, such as surveillance systems that generate continuous
∗

Corresponding authors.

5851

streaming data, these models face significant challenges due
to the inability to handle incremental and dynamic data effectively [Ge et al., 2022; Wu and Gong, 2021]. This limitation has motivated the emergence of Lifelong Person Reidentification (LReID), which aims to enable ReID models to
acquire new knowledge from streaming data while retaining
previously learned knowledge.
The primary challenge in LReID lies in addressing catastrophic forgetting, a phenomenon common in lifelong learning tasks. This issue is particularly pronounced in LReID
due to the unique characteristics of the task. First, as a
fine-grained classification problem, the intra-person variations caused by temporal, environmental, and camera view
changes are often significant[Ye et al., 2023]. Second, subtle
inter-person differences can lead to severe distribution overlaps, making it difficult to preserve discriminative knowledge
for each individual. These factors exacerbate the forgetting
of previously learned knowledge when learning new data.
To tackle catastrophic forgetting, most existing LReID
methods employ additional memory to store exemplar data
from previous tasks, which can be reused during training
with new datasets [Ge et al., 2022; Wu and Gong, 2021;
Yu et al., 2023a]. However, such memory-based approaches
raise privacy concerns and introduce additional computational overhead [Wu et al., 2025]. These limitations are
particularly acute given the private nature of pedestrian images. Some methods replaced sample-based approaches in
Class-Incremental Learning with prototype-based methods to
solve privacy and memory issues [Xu et al., 2024a]. However, these methods struggle to adapt to the training process
of LReID, which is designed as a retrieval task and places
greater emphasis on the embedding capability of the feature extractor. This requires constructing a more discriminative feature space that is effective for both seen and unseen
domains. Therefore, prototype-based methods commonly
used in CIL face difficulties in achieving satisfactory performance in LReID. After completing each training task, existing LReID methods merge the old model with the new one to
ensure compatibility between stored prototypes and the updated feature extractor. However, the static merging strategy
ignores the unique characteristics of each task. This approach
struggles to balance old knowledge retention and new knowledge learning during training.
In this paper, we propose a novel non-exemplar-based

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

method for LReID, which provides an effective solution to
privacy and memory constraints commonly faced in LReID
tasks. When a new task is introduced, the model compares
the new features to the existing prototypes. If the new features
are similar to the old prototypes, the model pushes them apart,
ensuring a clear distinction between old and new knowledge.
By creating a clear separation, this approach not only safeguards previously learned knowledge but also sharpens the
differentiation between old and new identity features, thus
enabling better propagation of knowledge within the model
over the entire training period and maintaining powerful embedding representation capability. In addition, we introduce
an Adaptive Parameter Evolution (APE) strategy. APE evaluates the parameters in the model to assess which ones have
a greater impact on the current training task and selectively
updates them. And it dynamically evaluates the impact of
new tasks, and based on this evaluation, it updates the model
fusion method to better align with the requirements of the current task. This approach ensures robust alignment between all
prototypes and the feature extractor while maintaining a high
level of compatibility between previously acquired and newly
learned knowledge throughout the training process.
Our contributions are summarized as follows:
• We propose a non-exemplar-based LReID method that
constructs prototypes to mitigate catastrophic forgetting
while addressing privacy and memory concerns.
• We introduce an Adaptive Parameter Evolution (APE)
strategy that dynamically integrates old and new knowledge by assessing task variations and selectively updating parameters, enhancing the adaptability of the model.
• Our method achieves superior performance on benchmark datasets, demonstrating its effectiveness and setting a new standard for LReID tasks.

2

Related Work

2.1

Lifelong Person Re-identification

Lifelong person re-identification (LReID) is an emerging area
that seeks to enable ReID models to learn continuously from
non-stationary data streams, a more realistic scenario than the
traditional static batch learning setup [Pu et al., 2021]. The
core challenge in LReID is mitigating catastrophic forgetting,
which refers to the tendency of neural networks to rapidly forget previously learned knowledge when trained on new tasks.
This phenomenon is particularly pronounced in LReID due
to the fine-grained nature of the task and the variability of
person appearances over time and across different environments. Current research in LReID can be broadly categorized
into two main branches: rehearsal-based mthods and knowledge distillation-based methods. Rehearsal-based approaches
mitigate forgetting by storing exemplar images from previous
tasks and replaying these during the training process of new
tasks [Ge et al., 2022; Huang et al., 2022; Yu et al., 2023b;
Wu and Gong, 2021]. While effective, this approach raises
practical concerns, such as privacy issues related to storing
human images, and scalability problems due to the growing
memory requirements. Knowledge distillation-based methods preserve past knowledge by enforcing consistency be-

5852

tween the outputs of old and new models [Huang et al., 2023;
Pu et al., 2022; Pu et al., 2023]. While these methods
have shown promise in terms of anti-forgetting capability, the
strict consistency constraints may hinder the plasticity of the
model, limiting its ability to effectively learn new and potentially different data distributions [Xu et al., 2024a]. Our work
aims to address the challenges of both branches by exploring a non-exemplar based approach that utilizes prototypes to
represent previously learned knowledge, thereby mitigating
forgetting while avoiding data storage issues.

2.2

Prototype-based Class Incremental Learning

In the field of Class Incremental Learning (CIL), many
prototype-based methods have been proposed, which are distinguished by their ability to avoid storing historical samples, significantly reducing storage overhead and mitigating
potential privacy concerns. Prototypes are typically derived
by averaging or aggregating the features of instances belonging to the same class, and have demonstrated their effectiveness in tasks such as few-shot learning and clustering. In CIL, prototypes are widely employed to represent
past knowledge, where prototypes are mostly used for classifier calibration [Zhu et al., 2021; Goswami et al., 2024] or
knowledge distillation at the output level [Zhu et al., 2022;
Shi and Ye, 2024]. While some attempts have been made to
adapt prototype-based methods for LReID [Xu et al., 2024a],
these approaches remain inadequate as they often fail to
fully address the unique requirements of LReID, such as the
need for fine-grained feature discrimination and robust feature transfer across incremental tasks. Directly applying CIL
methods often leads to suboptimal performance, as they overlook the critical need for effective and robust feature-based
knowledge transfer that preserves the fine-grained discriminative capabilities essential for accurate and reliable performance in complex and dynamic retrieval scenarios.

3

Method

3.1

Problem Formulation

We address the challenge of Non-Exemplar Lifelong Person
Re-identification. Formally, we are given a stream of sequential training datasets D = {D1 , D2 , . . . , DT }, where
T represents the total number of tasks. Each dataset Dt =
t
{(xti , yit )}N
i=1 contains Nt samples with corresponding identity labels. During the training phase for the t-th task, access
to previous datasets {D1 , . . . , Dt−1 } is restricted due to privacy considerations. To mitigate the problem of catastrophic
t
forgetting, we construct a prototype set Pt = {pi }N
i=1 for the
t-th task. Each prototype in Pt represents a unique identity,
and it is constructed by averaging the features of the corresponding identity.

3.2

Overview

Our proposed method, Prototype-guided Knowledge Propagation with Adaptive Learning (PKA), addresses Lifelong
Person Re-identification (LReID) through two key components: Prototype-guided Knowledge Propagation (PKP) and
Adaptive Parameter Evolution (APE). At each training stage
t with dataset Dt , The sampled prototypes, enhanced with

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

gradien-based updates
no gradient updates

(a)Overall Model(at training step t)
prototype

initiate

��−1

ℒ푝�푝
PKP

�1~�−1

Model

Model from previous task

random

�1표� �2표�

...

�B
표�

Prototype of step 1～t-1

APE

sampled prototype set

Batch Size B

��푓푢�

ℒ��

Task t

Model

Model

data stream

...

�

Model

Task t

�

ℒ�� +ℒ푐

��

(b)Prototype-Guided Knowledge Propagation(PKP)

ℒ푝�푝

noise

Figure 1: Overview of the proposed model Architecture. Solid arrows indicate the gradient-based updates, and dashed arrows represent no
gradient updates. Prototype-guided Knowledge Propagation (PKP) module uses a modified triplet loss Lpkp and a standard triplet loss Ltri to
propagate knowledge while ensuring discriminability. Different colors in the figure represent features or prototypes associated with different
identities.

added noise, are passed through the PKP module. This process encourages them to diverge from the current task features, enabling the extraction of more discriminative feature
embeddings while effectively leveraging prior knowledge, as
illustrated in Fig. 1.
The APE module dynamically manages model parameters,
assessing the relevance of Dt and selecting parameters based
on their impact on the current task. It then fuses the new
model’s parameters (θt ) with those of the previous model
(θt−1 ), as shown by the dashed arrows in Fig. 1. This ensures adaptive evolution of model parameters while retaining
past knowledge, resulting in a new model θf used for the next
training stage t + 1.

3.3

Prototype-guided Knowledge Propagation

To mitigate catastrophic forgetting in lifelong person reidentification (LReID), we propose a novel prototype-based
non-exemplar learning paradigm. our approach introduces
a novel perspective by leveraging prototypes to guide both
knowledge propagation and feature learning for new tasks, as
illustrated in Fig. 2. we generate more discriminative embeddings, which in turn improves retrieval performance for the
LReID model.
Existing methods that often employ triplet loss directly on
input features of new tasks for feature discrimination. Our
method aims to leverage prototypes to push apart identities
from previous tasks and new tasks within the embedding
space, creating a clear distinction. This facilitates the generation of more refined embeddings. To achieve this, we define a prototype set P = {p1 , . . . , pM }. During training for
a new task, we randomly sample a subset Ps from it, where
Ps contains prototypes of size half the batch size. To enhance the generalization and robustness of these prototypes,
we add Gaussian noise, resulting in augmented prototypes P̃s .
Specifically, for each prototype pl ∈ Ps , we add Gaussian
noise ϵ:
p̃l = pl + βϵ, ϵ ∼ N (0, σ 2 I).
(1)

5853

Here p̃l represents the augmented prototype, ϵ is drawn from
a Gaussian distribution with zero mean and covariance matrix
σ 2 I, and β is a hyperparameter controlling the magnitude of
the noise. The prototypes enhanced with noise can cover a
broader feature space during training, preventing the prototypes from becoming too concentrated. This enables a more
comprehensive separation of the features distributed around
them in the new task, thereby improving clustering among
distinct classes and optimizing the representation capability
of the embedding space.
We utilize a modified triplet loss to encourage separation
between features from the new task and augmented prototypes. The standard triplet loss, as defined in [Sun and Mu,
2022; Yu et al., 2023a; Schroff et al., 2015], serves as our
foundation:
Ltri = max(0, ∥a − p∥22 + α − ∥a − n∥22 ).

(2)

Here a represents the anchor feature, p represents the positive
feature, and n represents the negative feature, all drawn from
the features of the new task. Triplet loss aims to reduce the
distance between same-identity embeddings and increase the
distance between embeddings of different identities, thereby
enhancing the model’s ability to distinguish between them
and improving retrieval performance.
Our method incorporates two loss terms. The first term
focuses on pushing the features of the new task away from
the augmented prototypes to guarantee the discrimination between old and new tasks. To achieve this, the triplet loss is
modified by removing the positive sample. The following
loss term is used, which aims to maximize the distance between the anchor a (new task feature) and the negative sample
n (augmented prototype):
N

Lpkp =

1
1 X
max(0, γ − ∥a − n∥22 ).
N1 i=1

(3)

Here γ is a margin to enforces a minimum distance between
the new task features and the augmented prototypes, and N1

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

(c)Adaptive Parameter Evolution(APE)
Task t-1

Previous tasks
1-α
Task t

Model

��−1

Influence Weight=α

ℒ�� +ℒ푐

Model

Model

��

Task t (during train)

Task t (after train)

Subsequent tasks

dynamic model fusion

��−1

Model

weight

=α

��

weight

Parameter selection update
initiate

Model from previous task

froze
update

weight

...
Influence score

Model

Model

=1-α
��푓푢�

Resulting Fused Model

Figure 2: The Adaptive Parameter Evolution (APE) strategy employs gradient-informed parameter selection and dynamic model parameter
fusion. We employ gradient-informed selection (left) to update high-influence parameters and performs dynamic model fusion (right) with
weight α. The resulting model is then used for the next task.

represents the number of triplets sampled from the augmented
prototypes. This loss function is denoted as Lpkp . The second
term ensures the discrimination between different identities
within the new task. We use the following standard triplet
loss function:

gradient magnitude is greater than the threshold τ , otherwise,
the parameter is preserved. This process can be expressed as:

θ − η∇θi L, if |∇θi L| > τ
θi ← i
,
(5)
θi ,
if |∇θi L| ≤ τ

N2

1 X
max 0, ∥zjt − zpt ∥22 + α − ∥zjt − znt ∥22 .
Ltri =
N2 j=1

where η is the learning rate and τ is a predefined gradient
magnitude threshold. This approach not only enhances the
efficiency of the training process by focusing on the parameters that are most influential for the current task, but also
mitigates the risk of overfitting by preserving the parameters
that are less sensitive to the current task.
After gradient descent training, dynamic model parameter fusion is performed, which emphasizes maintaining the
compatibility between new and old knowledge. Rather than
applying a static fusion strategy [Xu et al., 2024a], APE dynamically adjusts the fusion weight according to the relative
influence of the current task [Xiao et al., 2023], which is estimated by the size of its training dataset. This allows us to
overcome the shortcomings of previous methods which are
less sensitive to variation between different tasks. After traint−1
t
ing a new task t, θnew
and θold
represent the parameters of
the new and old models, respectively. The fused model parameters θft used are determined as:

(4)
Here zkt represents the feature of the k-th sample in the new
task, and zpt and znt are the positive and negative samples,
respectively, within the new task. N2 is the number of triplets
sampled from the new task.

3.4

Adaptive Parameter Evolution Strategy

To further facilitate effective knowledge propagation in lifelong person re-identification, we introduce the Adaptive Parameter Evolution (APE) strategy. a dynamic mechanism that
orchestrates the evolution of model parameters in response to
the ongoing learning process. APE is characterized by two
synergistic components: gradient-informed parameter selection and dynamic model parameter fusion. These are designed to maintain the alignment between previously learned
prototypes and the retrained feature extractor, while balancing the retention of old knowledge and the acquisition of new
knowledge throughout the training process.
We posit that not all parameters are equally relevant to a
given learning task, and that a more judicious parameter update strategy is needed [Zhang et al., 2024]. To this end, we
compute the gradient of the loss function L with respect to
each model parameter θi , denoted by ∇θi L. These gradients
quantify the sensitivity of the loss to each parameter, providing a measure of the parameter’s relevance to the task. The
absolute value of the gradient is then computed as |∇θi L|.
Instead of naively updating all parameters, we introduce a
threshold τ . We selectively update a parameter only if its

5854

t−1
t
θft used = αt θnew
+ (1 − αt )θold
.

(6)

The fusion weight αt is dynamically determined based on the
relative impact of the current task’s dataset. We recognize
that datasets with larger sizes typically offer a more comprehensive representation of the task, hence they should exert a
greater influence on the model parameters. This influence is
quantified using the dataset size, therefore, αt is defined as:
Nt
,
(7)
α t = Pt
i=1 Ni
where Nt denotes the number of samples in the training
dataset for task t.

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

NonExemplar

Method
Joint-Train [Xu et al., 2024a]
LwF [Li and Hoiem, 2017]
SPD [Tung and Mori, 2019]
PRAKA* [Shi and Ye, 2023]
PRD* [Asadi et al., 2023]

CIL

LReID

✓

CRL [Xu et al., 2024b]
AKA [Pu et al., 2021]
AKA† [Pu et al., 2021]
PatchKD [Sun and Mu, 2022]
MEGE [Pu et al., 2023]
DKP [Xu et al., 2024a]
PKA(Ours)

✓
✓

Market1501

CUHK-SYSU

DukeMTMC

mAP

R@1

mAP

R@1

mAP

R@1

mAP

MSMT17
R@1

mAP

CUHK03
R@1

mAP

Seen-Avg
R@1

mAP

Unseen-Avg
R@1

75.3
56.3
35.6
37.4
7.3

90.1
77.1
61.2
61.3
18.0

84.5
72.9
61.7
69.3
33.5

86.0
75.1
64.0
71.8
35.6

66.9
29.6
27.5
35.4
3.7

81.6
46.5
47.1
55.0
7.6

31.6
6.0
5.2
10.7
0.8

57.1
16.6
15.5
27.2
2.4

58.5
36.1
42.2
54.0
33.8

61.4
37.5
44.3
55.6
33.8

63.4
40.2
34.4
41.3
15.8

75.2
50.6
46.4
54.2
19.5

55.2
47.2
40.4
47.7
23.0

48.2
42.6
36.6
41.6
17.7

58.0
51.2
58.1
68.5
39.0
60.3
57.7

78.2
72.0
77.4
85.7
61.6
80.6
80.2

72.5
47.5
72.5
75.6
73.3
83.6
85.0

75.1
45.1
74.8
78.6
76.6
85.4
86.9

28.3
18.7
28.7
33.8
16.9
51.6
59.4

45.2
33.1
45.2
50.4
30.3
68.4
75.6

6.0
16.4
6.1
6.5
4.6
19.7
31.0

15.8
37.6
16.2
17.0
13.4
41.8
56.1

37.4
27.7
38.7
34.1
36.4
43.6
44.1

39.8
27.6
40.4
36.8
37.1
44.2
44.6

40.5
32.3
40.8
43.7
34.0
51.8
55.4

50.8
43.1
50.8
53.7
43.8
64.1
68.7

47.8
44.3
47.6
49.1
47.7
59.2
62.3

43.5
40.4
42.6
45.4
44.0
51.6
55.3

Table 1: Training Order-1: Market-1501 → CUHK-SYSU → DukeMTMC-ReID → MSMT17-V2 → CUHK03. * denotes the results are
reproduced by the released official code. † denotes the results reported by [Sun and Mu, 2022]. The best and second-best results are marked
in bold and underlined.
NonExemplar

Method
Joint-Train[Xu et al., 2024a]
LwF[Li and Hoiem, 2017]
SPD[Tung and Mori, 2019]
PRAKA* [Shi and Ye, 2023]
PRD*[Asadi et al., 2023]

CIL

LReID

✓

CRL[Xu et al., 2024b]
AKA [Pu et al., 2021]
AKA† [Pu et al., 2021]
PatchKD[Sun and Mu, 2022]
MEGE[Pu et al., 2023]
DKP[Xu et al., 2024a]
PKA(Ours)

✓
✓

Market1501

CUHK-SYSU

mAP

R@1

mAP

R@1

mAP

R@1

mAP

R@1

mAP

R@1

mAP

R@1

mAP

R@1

66.9
42.7
28.5
31.2
3.6

81.6
61.7
48.5
48.7
8.2

31.6
5.1
3.7
6.6
0.6

57.1
14.3
11.5
19.1
1.8

75.3
34.4
32.3
47.8
8.9

90.1
58.6
57.4
69.8
22.3

84.5
69.9
62.1
70.4
34.6

86.0
73.0
65.0
73.0
36.1

58.5
34.1
43.0
54.9
35.4

61.4
34.1
45.2
56.6
35.3

63.4
37.2
33.9
42.2
16.6

75.2
48.4
45.5
53.4
20.7

55.2
44.0
39.8
48.4
20.7

48.2
40.1
36.3
41.1
15.0

43.5
32.5
42.2
58.3
21.6
53.4
54.2

63.1
49.7
60.1
74.1
35.5
70.5
70.6

4.8
5.4
6.4
3.0
14.3
24.3

13.7
15.1
17.4
9.3
33.3
48.0

35.0
37.2
43.2
25.0
60.6
68.6

59.8
59.8
67.4
49.8
81.0
85.3

70.0
71.2
74.5
69.9
83.0
85.0

72.8
73.9
76.9
73.1
84.9
86.8

34.5
36.9
33.7
34.7
45.0
43.8

36.8
37.9
34.8
35.1
46.1
46.9

37.6
40.8
38.6
43.2
30.8
51.3
55.2

49.2
37.2
49.4
54.1
40.6
63.2
67.5

45.3
46.0
48.6
44.3
59.0
60.3

41.4
41.7
44.1
41.1
51.6
52.8

DukeMTMC

MSMT17

CUHK03

Seen-Avg

Unseen-Avg

Table 2: Training Order-2: DukeMTMC-reID → MSMT17-V2→ Market-1501 → CUHK-SYSU → CUHK03. *denotes the results are
reproduced by the released official code. † denotes the results reported by [Sun and Mu, 2022]. The best and second-best results are marked
in bold and underlined.

By coupling gradient-informed parameter selection with
dynamic parameter fusion, APE provides an adaptive and robust mechanism for lifelong learning, enabling the previous
prototype set to maintain strong embedding representation
capability in the new feature space, resulting in improved retrieval performance across all datasets.

3.5

Overall Loss Function

To optimize our model, we employ a composite loss function
with three distinct components: a cross-entropy loss (Lce ),
a triplet loss on the combined features of new samples and
augmented prototypes (Lpkp ), and a standard triplet loss on
new task features (Ltri ). The cross-entropy loss is defined as:
N

Lce = −

C

1 XX
yic log(ŷic ),
N i=1 c=1

(8)

where N is the number of samples, C is the number of
classes, yic is the true label (0 or 1), and ŷic is the predicted
probability for class c of sample i. The overall loss function
is defined as:
L = Lce + α(Lpkp + Ltri ).

(9)

Here α is a variable weight controlling the influence of the
two triplet loss components, ensuring a balance between preserving old knowledge and acquiring new information. This

5855

weight α is shared between the two different triplet losses.
This combined loss function allows our model to learn discriminative feature representations that are robust to the challenges of lifelong learning in person re-identification tasks.

4

Experiments

4.1

Experimental Settings

Datasets. To evaluate the effectiveness of our proposed
method, we conduct extensive experiments on five benchmark lifelong person ReID datasets:Market-1501 [Zheng et
al., 2015], CUHKSYSU [Xiao et al., 2017], DukeMTMCReID [Ristani et al., 2016], MSMT17-V2 [Wei et al., 2018],
and CUHK03 [Li et al., 2014]. To simulate a lifelong person
ReID scenario in real-world settings, we evaluate our method
using two training orders as specified in, namely: Order1: Market-1501 → CUHK-SYSU → DukeMTMC-ReID
→ MSMT17-V2 → CUHK03, and Order-2: DukeMTMCReID → MSMT17-V2 → Market-1501 → CUHK-SYSU
→ CUHK03. To further evaluate the generalization capacity of our models, we tested them on seven additional
datasets (CUHK01 [Li et al., 2013], CUHK02 [Li and Wang,
2013], VIPeR [Gray and Tao, 2008], PRID [Hirzer et al.,
2011], i-LIDS [Branch, 2006], GRID [Loy et al., 2010], and
SenseReID [Zhao et al., 2017]) as unseen domains.

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

5856

average R@1(%)

To comprehensively evaluate our method, we compare it
against various state-of-the-art(SOTA) non-exemplar LReID
approaches, including DKP [Xu et al., 2024b], AKA [Pu et
al., 2021], PatchKD [Sun and Mu, 2022], and MEGE [Pu
et al., 2023]. Additionally, several class-incremental learning (CIL) methods, such as LwF [Li and Hoiem, 2017],
SPD [Tung and Mori, 2019], PRAKA [Shi and Ye, 2023],
and PRD [Asadi et al., 2023], are also tested. To ensure a
fair comparison, all models are implemented using the same
backbone and training configurations. Furthermore, we report the results of Joint-Train, which represents the upper
bound for LReID models by assuming access to all datasets
simultaneously during training.
We present the results of different methods on each seen
domain, as well as the average performance across all seen
domains (Seen-Avg) in Tab. 1 and 2, corresponding to Training Order-1 and Training Order-2, respectively.
Seen-Domain Performance Evaluation. As shown in Tab.1
and Tab.2, our PKA significantly outperforms all existing LReID models. Compared to the second-best method,
DKP, our model achieves an improvement of 3.6%/4.6% and
3.9%/4.3% on the average mAP/R@1 performance for seen
domains. The performance of DKP deteriorates in later training stages, likely because it adopts a static fusion strategy between the old and new models, failing to properly balance the
storage of old knowledge and the acquisition of new knowledge. And our PKA demonstrates an average mAP/R@1 improvement of 14.1%/13.5% and 13%/14.1% over the best CIL
method, PRAKA, across seen domains in both training orders. This indicates that by utilizing prototypes in LReID,
we have successfully enhanced the propagation of old knowledge within the model, effectively mitigating the catastrophic
forgetting problem, while also improving the model’s generalization ability.
It should be noted that the MSMT17 dataset presents inherent challenges due to substantial variations in weather and
lighting conditions. Nevertheless, our approach significantly
outperforms existing methods on this dataset, achieving a notably higher mAP and Rank-1 accuracy, which underscores
the superior effectiveness of our method.
To visually understand the performance trends of different models in the seen domain, we present the performance
curves over different training steps in Fig 3. As can be seen in
the figure, our model PKA consistently achieves the highest
mAP and R@1 compared with the other methods. Although
the initial performance of PKA may not be the top among
these methods, it emphasizes the propagation and application
of prior knowledge, granting it superior resistance to forgetting. Consequently, PKA outperforms others in later tests and
exhibits more prominent performance.
Unseen-Domain Generalization Evaluation. The average

70

60

50

40

30

1

2
3
4
training step

80
70
60

LwF
SPD
AKA
DKP
PKA

50
40

5

LwF
SPD
AKA
DKP
PKA
1

2
3
4
training step

5

Figure 3: Illustration of performance trend on previously seen domains. After each training phase, the model is evaluated on domains
it has encountered before.
60

70

LwF
SPD
AKA
DKP
PKA

50

40

30

20

LwF
SPD
AKA
DKP
PKA

60
average R@1(%)

Comparison with State-of-the-arts Methods

90

average mAP(%)

4.2

100

80

average mAP(%)

Evaluation Metrics. The mean Average Precision (mAP)
and rank@1 accuracy (R@1) are used to evaluate the model
performance on individual datasets. In addition, the average
mAP and the average R@1 in all seen and unseen domains are
calculated to assess and compare the overall lifelong learning
and generalization capabilities of the models, respectively.

50

40

30

1

2
3
4
training step

5

20

1

2
3
4
training step

5

Figure 4: Illustration of performance trend on unseen domains. After each training phase, the model is evaluated on domains it has not
encountered before.

performance in the unseen domains is shown in the last two
columns of Tab.1 and Tab.2. Our method demonstrates superior generalization capabilities compared to SOTA CIL
models, exhibiting an average mAP/R@1 improvement of
14.6%/13.7% and 11.9%/11.7% across both training orders.
Furthermore, our model also significantly outperforms the
SOTA LReID DKP models by a margin of 3.1%/3.7% and
1.3%/1.2% in average mAP/R@1 improvement. These results show that our model effectively consolidates more generalizable knowledge. To further evaluate the generalization ability of our model on unseen domains, we provide detailed performance curves over different training steps in Fig
4. The analysis of the curves indicates that our model not
only achieves high performance but also maintains stable and
consistent performance improvement even in unseen environments. This demonstrates that our proposed Prototype-guided
Knowledge Propagation approach, compared to LReID train-

Baseline PKP APE

Seen-Avg

Unseen-Avg

mAP R@1 mAP R@1
✓
✓

✓
✓

42.8
53.4
54.5
55.4

57.4
66.0
67.8
68.7

50.7
58.7
60.9
62.3

45.0
51.4
54.1
55.3

accur acy(% )

✓
✓
✓
✓

Table 3: Ablation study of different components.

ing methods that focus solely on classifier calibration,is more
effective in enhancing the model’s capability to extract features and shape a better embedding distribution. As a result,
it achieves more discriminative embedding representations on
unseen datasets, with improved generalization and flexibility.

Seen mAP
Seen R@1
Unseen mAP
Unseen R@1
70
68
66
64
62
60
58
56
54
52
50
1 1.25 1.5 1.75 2

α

(a)The weight of Lpkp and Ltri

accur acy(% )

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Seen mAP
Seen R@1
Unseen mAP
Unseen R@1
70
68
66
64
62
60
58
56
54
52
50
0 0.1 0.2 0.3 0.5

β

(b)Noise coefficient scale

Figure 5: Ablation studies on hyperparameters.

4.3

Ablation Studies

Influence of Different Components. In this section, we conduct several ablation studies on the proposed method. To
better analyze the impact of the core designs, our approach
is divided into two components:Prototype-guided Knowledge
Propagation (PKP) and Adaptive Parameter Evolution (APE).
The baseline model refers to a framework that does not incorporate the PKP and APE modules, meaning it does not utilize
prototypes for knowledge propagation. Instead, after completing each training task, the old model and the new model
are directly integrated using a static fusion strategy. Both the
PKP and APE modules excel at propagating prior knowledge,
significantly improving performance on both seen and unseen
domains. As shown in Tab. 3, furthermore, the APE module
facilitates the transfer of prototype knowledge constructed by
the PKP module. Consequently, performance is further improved when both modules are used together.
Influence of Hyperparameters. To analyze the impact of
hyperparameters on our method, we conducted experiments
to evaluate the effects of different hyperparameter settings.
We analyzed the effects of the weights of Ltri and Lpkp .
Specifically, Ltri focuses on distinguishing the distributions
among new data, while Lpkp emphasizes the separation between the distributions of new input data and the prototypes
of old data. We set the weights of these two losses to the same
value, and based on the results shown in Fig. 5(a), we choose
α = 1.5 as the default setting. In addition, we augmented the
prototypes by adding random noise to enhance the model’s
ability to transfer old knowledge through prototypes. Appropriate noise augmentation can significantly improve performance on unseen domains, whereas excessive noise can
degrade performance on both seen and unseen domains. As
shown in Fig. 5(b) we choose β = 0.2 as the default setting.

4.4

Visualization Results

To further analyze the impact of our PKA method on the feature space, we employ t-SNE to visualize features of selected
identities. As shown in Fig. 6, two identities from each of the
five seen datasets are selected. Comparing Fig. 6(a) (baseline) with Fig. 6(b) (PKA), we observe that the PKA model
yields more compact and separable clusters. In the baseline,
clusters are scattered and overlapping, making inter-class dis-

5857

cuhk03_320

cuhk_sysu_17

duke_15

market_22

msmt_102

cuhk03_460

cuhk_sysu_32

duke_45

market_56

msmt_63

(a)

(b)

Figure 6: t-SNE results of our PKA compared with the Baseline.
Different colors represent different identities, highlighting the classwise aggregation and the distinction between different categories.
The visualization demonstrates how our PKA improves intra-class
cohesion and inter-class separation.

tinction difficult. In contrast, PKA produces tighter and more
distinct clusters, indicating that it learns more discriminative
features beneficial for classification and analysis.

5

Conclusion

In this paper, we tackle the challenging task of LReID by introducing PKA, a novel non-exemplar-based approach. PKA
mitigates catastrophic forgetting through Prototype-guided
Knowledge Propagation (PKP), which utilizes prototypes and
triplet loss to preserve and transfer knowledge, and Adaptive
Parameter Evolution (APE) to enable dynamic model updates
for task adaptation. Extensive experiments on five benchmark
datasets demonstrate PKA’s effectiveness, achieving notable
improvements in mean Average Precision (mAP) and rank1 accuracy, along with enhanced generalization. These results highlight the potential of PKA as a robust, privacypreserving, and practical solution for LReID.

Acknowledgments
This work is partially supported by National Natural Science
Foundation of China under Grant (62176188, 62361166629,

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

62225113) and Key Research and Development Project of
Hubei Province (2022BAD175). The numerical calculations
in this paper have been done on the supercomputing system
in the Supercomputing Center of Wuhan University.

Contribution Statement
Zhijie Lu and Wuxuan Shi contribute equally to this work.

References
[Asadi et al., 2023] Nader Asadi, MohammadReza Davari,
Sudhir Mudur, Rahaf Aljundi, and Eugene Belilovsky.
Prototype-sample relation distillation: towards replay-free
continual learning. In International Conference on Machine Learning, pages 1093–1106. PMLR, 2023.
[Branch, 2006] Home Office Scientific Development
Branch. Imagery library for intelligent detection systems
(i-lids). In 2006 IET conference on crime and security,
pages 445–448. IET, 2006.
[Dai et al., 2018] Pingyang Dai, Rongrong Ji, Haibin Wang,
Qiong Wu, and Yuyu Huang. Cross-modality person reidentification with generative adversarial training. In IJCAI, volume 1, page 6, 2018.
[Ge et al., 2022] Wenhang Ge, Junlong Du, Ancong Wu,
Yuqiao Xian, Ke Yan, Feiyue Huang, and Wei-Shi Zheng.
Lifelong person re-identification by pseudo task knowledge preservation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 688–696,
2022.
[Goswami et al., 2024] Dipam Goswami, Yuyang Liu,
Bartłomiej Twardowski, and Joost van de Weijer. Fecam:
Exploiting the heterogeneity of class distributions in
exemplar-free continual learning. Advances in Neural
Information Processing Systems, 36, 2024.
[Gray and Tao, 2008] Douglas Gray and Hai Tao. Viewpoint invariant pedestrian recognition with an ensemble
of localized features. In Computer Vision–ECCV 2008:
10th European Conference on Computer Vision, Marseille,
France, October 12-18, 2008, Proceedings, Part I 10,
pages 262–275. Springer, 2008.
[Hirzer et al., 2011] Martin Hirzer, Csaba Beleznai, Peter M
Roth, and Horst Bischof. Person re-identification by
descriptive and discriminative classification. In Image
Analysis: 17th Scandinavian Conference, SCIA 2011, Ystad, Sweden, May 2011. Proceedings 17, pages 91–102.
Springer, 2011.
[Huang et al., 2022] Zhipeng Huang, Zhizheng Zhang, Cuiling Lan, Wenjun Zeng, Peng Chu, Quanzeng You, Jiang
Wang, Zicheng Liu, and Zheng-jun Zha. Lifelong unsupervised domain adaptive person re-identification with coordinated anti-forgetting and adaptation. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14288–14297, 2022.
[Huang et al., 2023] Jinze Huang, Xiaohan Yu, Dong An,
Yaoguang Wei, Xiao Bai, Jin Zheng, Chen Wang, and Jun

5858

Zhou. Learning consistent region features for lifelong person re-identification. Pattern Recognition, 144:109837,
2023.
[Leng et al., 2019] Qingming Leng, Mang Ye, and Qi Tian.
A survey of open-world person re-identification. IEEE
TCSVT, 30(4):1092–1108, 2019.
[Li and Hoiem, 2017] Zhizhong Li and Derek Hoiem.
Learning without forgetting. IEEE transactions on pattern
analysis and machine intelligence, 40(12):2935–2947,
2017.
[Li and Wang, 2013] Wei Li and Xiaogang Wang. Locally
aligned feature transforms across views. In Proceedings
of the IEEE conference on computer vision and pattern
recognition, pages 3594–3601, 2013.
[Li et al., 2013] Wei Li, Rui Zhao, and Xiaogang Wang. Human reidentification with transferred metric learning. In
Computer Vision–ACCV 2012: 11th Asian Conference on
Computer Vision, Daejeon, Korea, November 5-9, 2012,
Revised Selected Papers, Part I 11, pages 31–44. Springer,
2013.
[Li et al., 2014] Wei Li, Rui Zhao, Tong Xiao, and Xiaogang
Wang. Deepreid: Deep filter pairing neural network for
person re-identification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages
152–159, 2014.
[Li et al., 2024] Zongyi Li, Jianbo Li, Yuxuan Shi, Hefei
Ling, Jiazhong Chen, Runsheng Wang, and Shijuan
Huang.
Cross-modal generation and alignment via
attribute-guided prompt for unsupervised text-based person retrieval. In Proceedings of the International Joint
Conference on Artificial Intelligence. International Joint
Conferences on Artificial Intelligence Organization, pages
1047–1055, 2024.
[Loy et al., 2010] Chen Change Loy, Tao Xiang, and Shaogang Gong. Time-delayed correlation analysis for multicamera activity understanding. International Journal of
Computer Vision, 90:106–129, 2010.
[Pu et al., 2021] Nan Pu, Wei Chen, Yu Liu, Erwin M
Bakker, and Michael S Lew.
Lifelong person reidentification via adaptive knowledge accumulation. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7901–7910, 2021.
[Pu et al., 2022] Nan Pu, Yu Liu, Wei Chen, Erwin M
Bakker, and Michael S Lew. Meta reconciliation normalization for lifelong person re-identification. In Proceedings of the 30th ACM international conference on multimedia, pages 541–549, 2022.
[Pu et al., 2023] Nan Pu, Zhun Zhong, Nicu Sebe, and
Michael S Lew. A memorizing and generalizing framework for lifelong person re-identification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.
[Ristani et al., 2016] Ergys Ristani, Francesco Solera, Roger
Zou, Rita Cucchiara, and Carlo Tomasi. Performance measures and a data set for multi-target, multi-camera tracking. In European conference on computer vision, pages
17–35. Springer, 2016.


codex
PKA 这一篇的创新构造很清楚，它不是单纯做持续学习，而是把“类增量原型方法不适合 ReID”说成“分类器校准不等于检索表征保持”，再把机制落到旧原型和新身份的距离结构。压缩那篇更像反常识论文，核心不是新模块，而是把一个公认策略拆开证明真正起作用的是尾部语义解析。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Richer Semantics, Better Alignment: Aligning Visual Features with Explicit and
Enriched Semantics for Visible-Infrared Person Re-Identification
Neng Dong1 , Shuanglin Yan1 , Liyan Zhang2∗ and Jinhui Tang1
1
School of Computer Science and Engineering, Nanjing University of Science and Technology
2
College of Computer Science and Technology, Nanjing University of Aeronautics and Astronauticsn
{neng.dong, shuanglinyan, jinhuitang}@njust.edu.cn, zhangliyan@nuaa.edu.cn
Abstract

A photo of a [X1] [X2] [X3]
[X4] person.

Visible-infrared person re-identification (VIReID)
retrieves pedestrian images with the same identity across different modalities. Existing methods
learn visual features solely from images, failing
to align them into the modality-invariant semantic
space. In this paper, we propose a novel framework, termed Richer Semantics, Better Alignment
(RSBA), to align visual features with explicit and
enriched semantics. Specifically, we first develop
an Explicit Semantics-Guided Feature Alignment
(ESFA) module, which supplements textual descriptions for cross-modality images and aligns
image-text pairs within each modality, alleviating
the distribution discrepancy of visual features. We
then devise a Consistent Similarity-Guided Indirect Alignment (CSIA) module, which constrains
the similarity between intra-modality image-text
pairs to be consistent with that between intermodality text-text pairs, indirectly aligning visual
features with cross-modality semantics. Furthermore, we design a Cross-View Semantics Compensation (CVSC) module, which integrates multiview texts and improves the image-text matching of
one-to-one in ESFA and CSIA to one-to-many, further strengthening the alignment of visual features
within the semantic space. Extensive experimental
results on three public datasets demonstrate the effectiveness and superiority of our proposed RSBA.

1

?
(a)

Learnable Textual Prompts
Detailed Language Descriptions
The pedestrian in the image is The pedestrian in the image is
a young woman wearing a
a woman with long legs. She
blue skit, carrying a should
is wearing a gray skit and
bag and holding a parasol.
carrying a shoulder bag.

(b)

The pedestrian in the image is The pedestrian in the image is
a young woman wearing a
a woman with braided hair.
blue skit, carrying a should
She is wearing blue skit and
bag and holding a parasol.
carrying a should bag.
The pedestrian in the image is The pedestrian in the image is
a woman with long legs. She
a woman with braided hair.
is wearing a gray skit and
She is wearing a gray skit and
carrying a shoulder bag.
carrying a shoulder bag.

Figure 1: The core motivation of our RSBA framework: (i) Explicit
semantics (red) in language descriptions generated by LLaVA enable
the more effective alignment of visual features than learnable textual
prompts. (ii) The conflicting semantics (green) make the alignment
of images to inter-modality texts challenging. (iii) Multi-view texts
provide complementary semantics (blue) that play a positive role in
further enhancing the modality-invariance of visual features.

et al., 2017] has been proposed to retrieve visible images that
match the identity of a given infrared query, and vice versa.
The primary challenge in VIReID lies in aligning the
feature distribution of cross-modality images, for which
two main approaches have been developed. The first is
generative-based methods [Dai et al., 2018; Choi et al., 2020;
Miao et al., 2021], which transfer the style of images to another modality. However, these algorithms often introduce
noise during the generation process, compromising feature
discriminability. The second approach, generative-free methods [Ling et al., 2021; Ye et al., 2021a; Li et al., 2022], focuses on optimizing network structures and metric functions.
Comparatively, the latter has demonstrated greater effectiveness and currently stands as the predominant solution. However, the large modality discrepancy makes it challenging to
align heterogeneous features into a suitable common space.
To address this limitation, a recent study [Yu et al.,
2025] incorporates Contrastive Language-Image Pre-training
(CLIP) [Radford et al., 2021] into VIReID, demonstrating

Introduction

Person Re-Identification (ReID) [Ye et al., 2021b; Yan et al.,
2023a; Dong et al., 2024b] aims to match images of the same
individual across cameras, a critical component of intelligent security with profound research implications. Despite
significant advancements [Li et al., 2021; Yan et al., 2022;
Gong et al., 2022; Dong et al., 2024a], most existing algorithms focus solely on visible image retrieval, failing to meet
the demands of 24-hour surveillance systems, which must
also retrieve infrared images captured at night. To overcome
this limitation, visible-infrared person ReID (VIReID) [Wu
∗

A photo of a [X1] [X2] [X3]
[X4] person.

Corresponding author

927

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Our main contributions are summarized as follows:

that semantics represented by language descriptions of heterogeneous images exhibit no modality gap, thus aligning visual features into the semantic space is beneficial for alleviating their distribution discrepancy. However, pedestrian
images typically lack accompanying language descriptions.
Learning textual prompts [Zhou et al., 2022] effectively addresses this issue, as illustrated in Figure 1(a), but it still
presents several drawbacks: 1) Uncertainty. The set trainable words are unknown, raising questions about what the semantic information they represent; 2) Coarseness. Pedestrian
images with the same identity share a common prompt, and
only four learnable tokens are allocated for identity depiction, which is insufficient for the cross-view and fine-grained
nature of VIReID; 3) Cumbersomeness. Rather than end-toend, the paradigm of learnable prompts requires a meticulously designed two-stage training process.
Recently, LLaVA [Liu et al., 2023], a prominent large
language-vision model, has demonstrated exceptional capability in image captioning. As shown in Figure 1(a), it can
generate clear and detailed descriptions for pedestrian images, whose explicit semantics, such as age and gender, are
able to facilitate the effective alignment of visual features.
This inspires us to supplement specific texts with the assistance of LLaVA and align image-text pairs within each
modality. Furthermore, the alignment of images to intermodality texts is also necessary as it can further alleviate the
distribution discrepancy between visual features. However,
descriptions of visible and infrared images may include conflicting semantics, such as color attributes, which makes the
direct alignment inappropriate. This motivates us to explore
an indirect alignment of images to inter-modality texts. In
addition, as shown in Figure 1(b), within each modality, the
descriptions corresponding to different images of the same
pedestrian contain complementary content. Integrating them
to acquire comprehensive semantics and accordingly guide
the alignment of visual features is beneficial for further enhancing their modality invariance. This prompts us to enrich pedestrian semantics with multi-view texts.
In this paper, we propose a novel framework termed
Richer Semantics, Better Alignment (RSBA), which aligns
visual features with explicit and enriched semantics for effective VIReID. As shown in Figure 2, it consists of Explicit Semantics-Guided Feature Alignment (ESFA), Consistent Similarity-Guided Indirect Alignment (CSIA), and
Cross-View Semantics Compensation (CVSC). ESFA leverages LLaVA to generate textual descriptions for visible and
infrared images, respectively, and maximizes the similarity
between visible (infrared) image-text pairs to align crossmodality visual features into the semantic space. CSIA constrains the similarity between intra-modality image-text pairs
to be consistent with that between inter-modality text-text
pairs, achieving the indirect alignment of visible visual features with infrared semantics as well as infrared visual features with visible semantics. CVSC integrates text features
from another view into the current view and accordingly improves the image-text matching in ESFA and CSIA from oneto-one to one-to-many, thereby further advancing their alignment. Our RSBA is trained end-to-end, with only the visual
side used to extract cross-modality image features for testing.

• We explore the advantages of explicit semantics in alleviating the modality gap between visible and infrared
images, and accordingly propose ESFA to align visual
features into the semantic space for effective VIReID.
• We realize the alignment of visual features with intermodality semantics, and accordingly present CSIA to
address the challenge of the direct alignment resulting
from conflicting semantics.
• We consider the comprehensiveness of multi-view semantics, and develop CVSC to achieve the one-to-many
alignment between images and texts, further strengthening the modality invariance of visual features.
• Extensive experiments across three datasets demonstrate
that RSBA achieves new state-of-the-art performance,
with each component contributing effectively.

2

Related Work

2.1

Visible-Infrared Person Re-Identification

VIReID is a challenging task due to the significant modality
gap between visible and infrared images. An intuitive approach is to transfer images from one modality to the style of
another. For instance, JSIA [Wang et al., 2020] employed
feature decoupling and cycle generation to augment crossmodality image pairs. Given the substantial gap between
heterogeneous data, MSA [Miao et al., 2021] designed a
style similarity constraint to ensure the quality of generated
images. To prevent identity information loss during transfer, ACD [Pan et al., 2024] introduced conditional probability density to optimize the generation network. Although
generative-based methods are intuitive and effective, they are
prone to model collapse and susceptible to introducing noise.
Generative-free methods have recently attracted considerable attention due to they circumvent the limitations of
generative-based approaches. These methods primarily focus on aligning cross-modality features by constructing appropriate networks or metric functions. For instance, ZeroPadding [Wu et al., 2017] evaluated the suitability of four
networks for VIReID and proposed a one-stream structure
with a zero-padding strategy. AGW [Ye et al., 2021b] devised a weighted regularization triplet loss to optimize the
relative distance between positive and negative pairs in both
intra-modality and inter-modality. To learn informative representations, DEEN [Zhang and Wang, 2023] designed an embedding expansion network to extract diverse features. However, the large modality discrepancy still makes the feature
alignment challenging. In this paper, we explore a semanticguided approach to effectively align visual features.

2.2

Large Language-Vision Models

Large language-vision models have emerged as a significant research topic, bridging computer vision and natural
language processing. CLIP, a representative model, excels
in learning visual content with high-level semantic information, showcasing exceptional potential across various downstream vision tasks [Wang et al., 2022; Zhao et al., 2022;
Yan et al., 2023b; Tang et al., 2024; Yan et al., 2024;

928

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Image
Embedding Space

Command

Please describe the
characteristic of the
pedestrian in the
image.

f vis
Lid  Lmse

Without training and
respond quickly.

tavis  tair 

f ir

Descriptions for Visible Images
The pedestrian in the image is
The
pedestrian
in
image
pedestrian
in
thethehair
image
is is
aThe
woman
with
braided
The pedestrian in the image is
woman
withisbraided
hair.
young
woman
wearing
and a a
long
legs.
She
wearinga
a young woman wearing a
She
wearing
skit and
blue
and
galsses.
ais skit
blue
skit. blue
blue skit, carrying a shoulder
carrying a shoulder bag.
bag and holding a parasol.

Textual
Embedding Space

f

 Dot Product  Element-wise Addition

f 

tivis  tiir 
Wa

tmvis,i  tmir ,i 

CVSC: Cross-View Semantics Compensation

t vis  t ir 

f vis  f



ir

Textual
Embedding Space

CVSC

aivis  aiir 



2m
Locon

Descriptions for Infrared Images
The pedestrian in the image is
The
pedestrian
in
image
pedestrian
in
thethehair
image
is is
aThe
woman
with
braided
The pedestrian in the image is
woman
withisbraided
hair.
young
woman
wearing
and a a
long
legs.
She
wearinga
a woman with long legs. She
She
wearing
skit and
blue
and
galsses.
ais skit
blue
skit. blue
is wearing a gray skit and
carrying a shoulder bag.
carrying a shoulder bag.

vis

Wv

CSIA

ESFA

Text
Encoder


Wk

tavis  tair 

Image Encoder

LLaVA

Wq

tivis  tiir 

ESFA: Explicit Semantics-Guided Feature Alignment
Taking 2 samples as
Losc2 m
ir
vis
the example
ir

t t







Text
Encoder

CVSC

t vis
Maximize Similarity

tmvis

t ir
Minimize Similarity

tmir

 Consistent Similarity

t vis  t ir 

t vis  t ir 

CSIA: Consistent Similarity-Guided Indirect Alignment

Figure 2: Overview of our RSBA. It acquires specific descriptions with LLaVA, integrates multi-view pedestrian semantics with CVSC,
aligns visual features into the semantic space with ESFA, and indirectly aligns visual features with inter-modality semantics with CSIA.

Shen et al., 2025]. In the field of ReID, CLIP-ReID [Li et
al., 2023] first introduced CLIP to advance this community.
To tackle the occlusion problem, RGANet [He et al., 2024]
employed CLIP to generate local textual prototypes for mining discriminative part features. In VIReID, CSDN [Yu et al.,
2025] incorporated trainable textual prompts to acquire implicit pedestrian descriptions, aligning visual features of visible and infrared images into the semantic space. However, the
semantics learned by CSDN are unknown and coarse, limiting its alignment ability. In this paper, we propose ESFA to
address this limitation, and further develop CSIA and CVSC
to improve our RSBA framework for more efficient VIReID.

3

Methodology

3.1

Preliminaries

equal to that between positive pairs under the intra-modality:
P

Lmse =

N
N
1 X
1 X
qi log(pvis
qi log(pir
i )−
i ),
N i=1
N i=1

(2)

k=1

K

Dintra =

1 X vis
fi − fkvis 2 ,
K −1

(3)

k=1
k̸=i

K

Dacross =

1 X vis
fi − fkir 2 ,
K

(4)

k=1

where P and K denote P identities and K visible and K
infrared images of each identity randomly sampled in each
mini-batch. ∥·∥2 represents the Euclidean distance.

3.2

Formally, we define the visible and infrared image sets as
Nv
ir Nr
{xvis
i }i=1 and {xi }i=1 , where Nv and Nr represent the sizes
of these two heterogeneous data, respectively. The label set
Np
is denoted as {yi }i=1
, with Np indicates the number of identities. In each mini-batch, N paired cross-modality images
ir N
{xvis
i , xi }i=1 are randomly sampled and their visual features
vis
N ×d
{fi , fiir }N
are extracted, where d is the dimeni=1 ∈ R
sion of features. We employ identity loss and modality-shared
enhancement loss [Lu et al., 2023] to optimize the network:
Lid = −

2K

1 X X intra
[ (Dk
− Dkacross )2 ],
2P K p=1

Explicit Semantics-Guided Feature Alignment

Most existing frameworks treat VIReID as a pure vision task,
lacking the ability to capture pedestrian semantics that is beneficial for modality alignment. Although CSDN introduces
CLIP and CoOP to address this limitation, the uncertainty and
coarseness of implicit semantics hinder the alignment of visual features into the semantic space. To this end, we propose
ESFA, which leverages LLaVA to generate explicit textual
descriptions and aligns cross-modality images with them.
As illustrated in Figure 2, given a pedestrian image, we
send the request command ’Please describe the characteristics of the pedestrian in the image’ to LLaVa. It responds
with a natural language description ’The pedestrian in the image is a young woman wearing a blue skit, carrying a shoulder bag and holding a parasol’. This description provides
clearer and more detailed explicit semantics, such as age, gender, and clothing, compared to the learnable textual prompt
’A photo of a [X1 ][X2 ][X3 ][X4 ] person’ in CSDN. Notably,

(1)

where qi is the one-hot vector of identity label yi . pvis
and pir
i
i
represent classification results of fivis and fiir , respectively.
The modality-shared enhancement loss constrains the average distance between positive pairs across modalities to be

929

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

3.4

LLaVA operates without requiring training and delivers responses quickly, taking approximately 1.2 seconds per image.
Suppose the generated language bases for visible and inir Nr
v
frared images are {livis }N
i=1 and {li }i=1 . In each mini-batch,
vis ir N
ir N
we sample {li , li }i=1 corresponding to {xvis
i , xi }i=1
and input them into the textual encoder to extract feair N
N ×d
tures {tvis
. To align {fivis , fiir }N
i , ti }i=1 ∈ R
i=1 with
vis ir N
{ti , ti }i=1 , we maximize the similarity between them:
Lcon = Li2t + Lt2i ,

The above two alignments are based on the one-to-one matching between image and text. However, within each modality,
variations in camera views result in descriptions for different
images of the same pedestrian emphasizing distinct objects.
For example, the description for a front-facing image may
highlight age and gender, while that for a rear-facing image is
more likely to focus on hairstyle and backpack. As a result,
semantics derived solely from single-view text are one-sided
and contribute limited to the robustness of visual features. To
address this limitation, we design CVSC to explore the oneto-many correspondence between images and texts.
As illustrated in Figure 2, we introduce an attention fusion module to integrate information in the textual feature
from another view into the textual feature of the current view.
Specifically, for the visible textual feature tvis
i , we randomly
select a textual feature tvis
a that shares the same identity with
while from different views. We compute the similarity
tvis
i
vis
and tvis
between tvis
a to derive the attention weight ai :
i


vis T
Wq (tvis
i )(Wk (ta ))
,
(9)
√
avis
= sof tmax
i
d
where Wq and Wk are two linear projection layers. We mulvis
tiply avis
and tvis
a to determine the contribution of ta , and
i
add the the resulting weighted feature to tvis
:
i
vis
vis
vis
tvis
=
t
+
W
(a
W
(t
(10)
a
v
m,i
i
i
a )),

(5)

where
N

Li2t = −

1 X
exp(s(fivis , tvis
i ))
log PN
vis , tvis ))
N i=1
exp(s(f
j
i
j=1
N

(6)

1 X
exp(s(fiir , tir
i ))
−
,
log PN
ir
ir
N i=1
j=1 exp(s(fi , tj ))
N

Lt2i = −

vis
1 X
exp(s(tvis
i , fi ))
log PN
vis
vis
N i=1
j=1 exp(s(ti , fj ))
N

(7)

ir
1 X
exp(s(tir
i , fi ))
−
log PN
,
ir
ir
N i=1
j=1 exp(s(ti , fj ))

where s(·) represents the cosine similarity.

3.3

Consistent Similarity-Guided Indirect
Alignment

where Wa and Wv are also linear projection layers. tvis
m,i represents the multi-view textual feature corresponding to livis ,
which contains richer pedestrian semantics as it compensates
for the missing cross-view information in tvis
i . Similarly, we
can acquire the multi-view infrared textual feature tir
m,i .
We reformulate Equations (5), (6), and (7) as the following
Equations (11), (12), and (13), which maximize the similariir
ir
ties between fivis and tvis
m,i , as well as between fi and tm,i :

ESFA achieves the alignment of images and texts in each
modality; however, it ignores the alignment of images and
inter-modality texts. A straightforward approach is to maximize the similarity between them similar to the above process. However, cross-modality texts describe the same object
with conflicting attributes due to visual ambiguity. For example, the clothing exhibits ’blue’ in the visible text while being
seen as ’gray’ in the infrared one. Forcing the maximization
of similarity between images and inter-modality texts may
disrupt the expressiveness of semantics. To this end, we develop CSIA to explore the indirect alignment between them.
As illustrated in Figure 2, for the visible visual feature fivis ,
CSIA constrains its similarity with the visible text feature tvis
i
to be equal to the similarity between the infrared text feature
vis
tir
i and visible text feature ti , thereby indirectly establishing
the alignment relationship between fivis and tir
i . Similarly,
infrared visual features fiir and visible text features tvis
are
i
indirectly aligned by constraining the similarity between fiir
vis
and tir
and tir
i to be consistent with that between ti
i :

o2m
o2m
Lo2m
con = Li2t + Lt2i ,

Lo2m
i2t = −

1 X
ir vis 2
(s(fivis , tvis
i ) − s(ti , ti ))
N i=1
N

N
exp(s(fivis , tvis
1 X
m,i ))
log PN
vis
vis
N i=1
j=1 exp(s(fi , tm,j ))

N
exp(s(fiir , tir
1 X
m,i ))
log PN
,
−
ir
ir
N i=1
j=1 exp(s(fi , tm,j ))

Lo2m
t2i = −

N
vis
exp(s(tvis
1 X
m,i , fi ))
log PN
vis
vis
N i=1
j=1 exp(s(tm,i , fj ))

(11)

(12)

(13)
N
ir
exp(s(tir
1 X
m,i , fi ))
−
log PN
.
ir
ir
N i=1
j=1 exp(s(tm,i , fj ))
This process achieves the one-to-many alignment between
images and texts within each modality. In addition, we also
redefine the Equation (8) to the following Equation (14), indirectly aligning images with multi-view inter-modality texts:
N
1 X
ir vis 2
Lo2m
=
(s(fivis , tvis
sc
m,i ) − s(ti , tm,i ))
N i=1
(14)
N
1 X
ir ir
vis ir
2
+
(s(fi , tm,i ) − s(ti , tm,i )) .
N i=1

N

Lsc =

Cross-View Semantics Compensation

(8)

1 X
vis ir 2
+
(s(fiir , tir
i ) − s(ti , ti )) .
N i=1
This similarity consistency loss not only achieves the alignment of images with inter-modality texts but also indirectly
maximizes the similarity between infrared and visible texts,
which helps alleviate cross-modality semantic discrepancy,
thus facilitating more effective alignment of visual features.

930

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

3.5

4.3

Training and Inference

SYSU-MM01. Table 1 presents the comparison results with
the state-of-the-art methods on the SYSU-MM01 dataset,
showing that RSBA consistently outperforms them across
all settings. Specifically, in the all-search testing mode,
our Rank-1 accuracy and mAP surpass those of the best
generative-based method, ACD, by 4.0% (4.2%) and 3.7%
(3.1%), respectively, while in the indoor-search mode, the
improvements are 8.8% (6.3%) and 5.1% (5.2%). These
gains are attributed to our approach aligning modalities at
the feature level, which circumvents performance limitations
imposed by the generated low-quality images. Compared
to generative-free methods, under the single-shot mode, our
Rank-1 accuracy exceeds that of CycleTrans by 1.9% (0.5%),
and our mAP surpasses HOS-Net by 0.6% (2.1%). This advantage arises from the proposed RSBA aligns visual features
with the semantic space, which are beneficial for alleviating
the modality gap. Furthermore, our RSBA also outperforms
CSDN across all settings, benefiting from its ability to capture
clear, detailed, and rich semantics, as opposed to the coarse
and ambiguous semantics learned by CSDN.
RegDB. We further evaluate the performance of RSBA on
the RegDB dataset, with the quantitative results summarized
in Table 2. Our method achieves superior recognition rates
compared to existing generative-based approaches. For instance, in the visible-to-infrared testing mode, RSBA outperforms TSME in Rank-1 accuracy by 7.9% and surpasses ACD
in mAP by 7.6%. Similarly, our method exhibits significant
performance advantages over state-of-the-art generative-free
methods, such as MBCE and HOS-Net. In comparison with
CSDN, RSBA improves the Rank-1 and mAP by 2.1% and
4.0% in the visible-to-infrared testing mode.
LLCM. We also evaluate the proposed RSBA on the challenging LLCM dataset to provide a comprehensive assessment. As detailed in Table 3, in the visible-to-infrared testing
mode, RSBA achieves a Rank-1 accuracy and mAP that are
1.1% and 0.9% higher, respectively, than those of the stateof-the-art HOS-Net. Similarly, in the infrared-to-visible testing mode, RSBA outperforms HOS-Net with improvements
of 0.9% in Rank-1 accuracy and 1.0% in mAP. These results
further highlight the superiority of our approach.

The proposed RSBA is trained in an end-to-end manner, with
the total loss expressed as:
o2m
Ltotal = Lid + Lmse + λ1 Lo2m
con + λ2 Lsc ,

(15)

where λ1 and λ2 are two hyper-parameters used to balance
o2m
the relative importance of Lo2m
con and Lsc , respectively.
Notably, the generation of language descriptions is only
performed in the training phase, ensuring the practicality of
our framework. During inference, the textual encoder and
attention fusion module are not required, reducing the model
complexity and inference time of our framework.

4

Experiments

4.1

Datasets and Evaluation Metrics

Datasets. SYSU-MM01 [Wu et al., 2017] contains 30,071
visible images captured by 4 RGB cameras and 15,792 infrared images captured by 2 IR cameras. The training set
includes 22,258 visible images and 11,909 infrared images
corresponding to 395 identities. The testing set comprises
3,803 infrared images of 96 identities and either 301 or 3,010
randomly sampled visible images for single-shot or multishot settings, respectively. RegDB [Nguyen et al., 2017] is
a small-scale VIReID dataset with 4,120 visible images and
4,120 infrared images from 412 pedestrians. Following the
standard protocol, 2,060 visible and 2,060 infrared images of
206 identities are allocated for training, while the remaining
images are used for testing. LLCM [Zhang and Wang, 2023]
is a recently released challenging VIReID dataset collected
under low-light conditions. Its training set includes 16,946
visible images and 13,975 infrared images of 713 identities,
and its testing set consists of 8,680 visible images and 7,166
infrared images corresponding to 351 identities.
Evaluation Metrics. We assess the retrieval performance
using the general indicators named mean Average Precision
(mAP) and Cumulative Matching Characteristics (CMC).

4.2

Comparison with State-of-the-Art Methods

Implementation Details

We conduct experiments using the PyTorch library on a single RTX 4090 GPU. The proposed RSBA framework incorporates a training-free LLaVA, a CLIP model comprising a
visual encoder and a textual encoder, with ResNet50 [He et
al., 2016] serving as the backbone for the visual encoder, and
an attention fusion module consisting of four linear projection
layers. Following AGW [Ye et al., 2021b], we train two parallel first convolutional layers of ResNet50 for each modality
while sharing the parameters of the subsequent four blocks.
During training, we randomly sample 8 identities, each with
4 visible and 4 infrared images. All input images are resized
to 288×144 and subjected to data augmentation techniques,
including random padding, cropping, and flipping. The training process spans 120 epochs, with the initial learning rate set
to 3e-4 for the visual encoder and 1e-6 for the textual encoder
and attention fusion module, decaying by a factor of 0.1 at the
40th and 70th epochs, respectively. The hyper-parameters are
configured as λ1 = 0.25 and λ2 = 0.2.

4.4

Ablation Studies

We evaluate the effectiveness of each component in our proposed RSBA, with the results presented in Table 4. The Rank1 and mAP of Baseline (’0’) are 71.9% and 67.6% under the
single-shot and 80.0% and 61.9% under the multi-shot.
Effectiveness of ESFA. ESFA aims to introduce explicit semantics to guide the alignment of cross-modality visual features. As shown in Table 4, it improves the Rank-1 and mAP
by 4.4% and 5.0% under the single-shot mode, which validates that aligning visual features into the semantic space is
reasonable and effective for mitigating the modality gap.
Effectiveness of CSIA. CSIA constrains the consistent similarity between intra-modality image-text pairs and intermodality text-text pairs to establish the correspondence between images and cross-modality texts. As detailed in Table
4, under the single-shot test mode, it improves the Rank-1 accuracy from 76.3% to 77.4%, which indicates that the align-

931

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Methods

Ref

cmGAN [Dai et al., 2018]
Hi-CMD [Choi et al., 2020]
JSIA [Wang et al., 2020]
MSA [Miao et al., 2021]
TSME [Liu et al., 2022b]
ACD [Pan et al., 2024]
AGW [Ye et al., 2021b]
MCSL [Ling et al., 2021]
CAJ [Ye et al., 2021a]
MMN [Zhang et al., 2021]
MAUM [Liu et al., 2022a]
CIFT [Li et al., 2022]
MBCE [Cheng et al., 2023]
DEEN [Zhang and Wang, 2023]
SEFL [Feng et al., 2023]
HOS-Net [Qiu et al., 2024]
CSCL [Liu et al., 2025]
CycleTans [Wu et al., 2025]
CSDN [Yu et al., 2025]
Ours (RSBA)

IJCAI’18
CVPR’20
AAAI’20
IJCAI’21
TCSVT’22
TIFS’24
TPAMI’21
IJCAI’21
ICCV’21
MM’21
CVPR’22
ECCV’22
AAAI’23
CVPR’23
CVPR’23
AAAI’24
TMM’24
TNNLS’24
TMM’25
IJCAI’25

All-Search
Single-Shot
Multi-Shot
R1
mAP
R1
mAP
26.9 27.8 31.4 22.2
34.9 35.9
38.1 36.9 45.1 29.5
63.1 59.2
64.2 61.2 70.3 54.3
74.4 71.1 80.4 66.9
47.5 47.6
64.8 60.8 68.0 51.4
69.8 66.8
70.6 66.9
71.6 68.7
71.7 67.6 78.0 62.4
74.7 72.0 78.3 65.7
74.7 71.8
75.1 70.1
75.6 74.2
75.7 72.0
76.5 72.6 82.8 68.5
76.7 73.0 83.5 67.9
78.4 74.8 84.6 70.0

Indoor-Search
Single-Shot
Multi-Shot
R1
mAP
R1
mAP
31.6 42.1 37.0 32.7
43.8 52.9 52.7 42.7
67.1 72.7
64.8 71.5 76.8 65.0
78.9 82.7 86.0 78.6
54.1 62.9
76.2 76.7
76.2 79.6
76.9 81.9
78.6 82.1 86.9 77.0
83.4 86.0 88.4 80.6
80.3 83.3
78.4 81.2
84.2 86.7
80.8 83.5
87.2 84.9 91.2 81.4
84.5 86.8 91.3 82.2
87.7 87.8 92.3 83.8

Table 1: Performance comparison with state-of-the-art methods on SYSU-MM01. ’-’ denotes that no reported result is available.

Methods
Hi-CMD
JSIA
MSA
TSME
ACD
AGW
MCSL
CAJ
MMN
MAUM
CIFT
MBCE
DEEN
SEFL
HOS-Net
CSCL
CycleTrans
CSDN
Ours (RSBA)

Visible to Infrared
R1
mAP
70.9
66.0
48.1
48.9
84.8
82.1
87.3
76.9
84.7
83.2
70.0
66.4
93.8
87.5
85.0
65.3
91.6
84.1
87.8
85.0
92.1
86.9
93.1
88.3
91.1
85.1
91.0
85.2
94.7
90.4
92.1
84.2
90.6
85.6
95.4
87.7
95.2
90.8

Infrared to Visible
R1
mAP
48.5
49.3
86.4
75.7
87.1
84.7
91.5
85.2
84.7
61.5
87.5
80.5
86.9
84.3
90.1
84.8
93.4
87.9
89.5
83.4
92.1
86.5
93.3
89.2
89.6
85.0
81.8
87.0
92.3
85.5
94.4
89.5

Methods
AGW
CAJ
MMN
DEEN
HOS-Net
Ours (RSBA)

Visible to Infrared
R1
mAP
51.5
55.3
56.5
59.8
59.9
62.7
62.5
65.8
64.9
67.9
66.0
68.8

Infrared to Visible
R1
mAP
43.6
51.8
48.8
56.6
52.5
58.9
54.9
62.9
56.4
63.2
57.3
64.2

Table 3: Performance comparison on LLCM.
ESFA
0
1
2
3
4

✓
✓
✓
✓

CSIA

CVSC

✓
✓

✓
✓

Single-Shot
R1
mAP
71.9 67.6
76.3 72.6
77.4 73.2
77.6 73.7
78.4 74.8

Multi-Shot
R1
mAP
80.0 61.9
82.1 66.5
82.7 68.2
83.1 68.8
84.6 70.0

Table 4: Ablation studies of our RSBA.

Table 2: Performance comparison on RegDB.

and CSIA, the recognition performance reaches a peak. These
results fully demonstrate the reasonableness of motivation behind CVSC and the effectiveness of its technology.

ment of inter-modality image-text pairs plays a positive role
in the further effective alignment of visual features.
Effectiveness of CVSC. CVSC integrates multi-view texts to
capture comprehensive semantics that are beneficial for improving the alignment in ESFA and CSIA. As illustrated in
Table 4, when it is equipped with ESFA, the Rank-1 accuracy
is improved by 1.3% and 1.0% under the two test modes, respectively. In addition, when incorporating it with both ESFA

4.5

Parameters Analysis

We introduce the hyper-parameters λ1 and λ2 to regulate the
o2m
relative importance of the loss terms Lo2m
con and Lsc . The
former optimizes the model to align image-text pairs within
each modality, while the latter drives the model to mine the
correspondence between image-text pairs across modalities.

932

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

80
78
76
74
72
70
68
66

Rank-1

mAP

0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5

1

80
78
76
74
72
70
68
66

Rank-1

further reduced. (2) Why indirectly align them? Different
from image-text pairs within each modality, which naturally
correspond to each other, images and inter-modality texts are
not completely matched. Therefore, aligning them directly by
maximizing the similarity between them may destroy the expressiveness of semantics, thereby weakening the alignment
between intra-modality image-text pairs. As shown in Table 5, the direct alignment (1) reduces the Rank-1 accuracy
and mAP of ESFA from 76.3% to 75.6% and from 72.6% to
72.1%. In contrast, our designed indirect alignment strategy
(2) improves the Rank-1 and mAP by 1.1% and 0.6%. This
proves the rationality and effectiveness of the approach.

mAP

0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5

2

Figure 3: Parameters analysis of λ1 and λ2 .

As shown in Figure 3, the optimal values for λ1 and λ2 are
0.25 and 0.2. In addition, excessively large values diminish
the contributions of the identity loss and modality-shared enhancement loss, while overly small values result in insufficient alignment, both leading to performance degradation.

4.6

The number of cross-view texts
The proposed CVSC aims to enrich pedestrian semantics with
multi-view texts, and we achieve this by integrating text with
that from an additional view. It is also feasible to integrate it
with texts from multiple additional views. However, we observe that the recognition performance degrades as the number of views increases (2 and 3). This is because the generated
descriptions may contain some inaccurate content, amplifying the noisy semantics during the information integration.
In addition, CVSC is achieved through the attention fusion
network, which requires more parameters as the number of
views increases, making model optimization challenging.

Further Discussions

In this section, we further discuss each proposed module,
with the experimental results presented in Table 5.
Modules
ESFA
CSIA
CVSC

1
2
1
2
1
2
3

Single-Shot
R1
R10 mAP
74.1 95.0 69.9
76.3 96.8 72.6
75.6 95.9 72.1
77.4 97.9 73.2
78.4 98.6 74.8
77.8 98.1 74.3
76.7 96.9 72.9

R1
80.9
82.1
80.9
82.7
84.6
83.5
81.8

Multi-Shot
R10 mAP
96.5 64.3
98.1 66.5
97.5 65.6
98.3 68.2
99.0 70.0
98.2 68.9
97.6 67.1

4.7

Limitations

This paper acquires explicit and enriched semantics to effectively alleviate the modality gap between visible and infrared
pedestrian images. However, as we discussed above, on the
one hand, the rigid semantic pattern weakens the effect of
alignment. On the other hand, this paper initially explores
the enrichment of pedestrian semantics with multi-view texts,
while we ignore the quality of texts, the number of cross-view
texts, and the strategy of text fusion, which all affect the richness of the semantics. These limitations motivate us to explore the semantics of diversity and richness more deeply.

Table 5: Further discussions of each proposed module.

The superiority of explicit semantics
Different from the implicit semantics in CSDN [Yu et al.,
2025], our ESFA acquires explicit pedestrian semantics to
align cross-modality visual features. In contrast, the latter is
more representative and thus guides the more efficient alignment. As shown in Table 5, the recognition performance
achieved by ESFA (2) is higher than that achieved based on
implicit semantic alignment (1), with improvements of 2.1%
in Rank-1 accuracy and 2.7% in mAP under the single-shot
test mode. Notably, we observe that the descriptions generated by LLaVA follow the fixed sentence structure of ‘The
pedestrian in the image is a [age group] [gender] wearing
[clothing], carrying [accessory]’, which may cause the model
to overfit to the non-differentiated semantic pattern, limiting
the effect of alignment. This motivates us to explore acquiring diverse pedestrian semantics in the future.

5

Conclusion

In this paper, we propose a novel Richer Semantics, Better
Alignment (RSBA) framework for effective VIReID. It focuses on aligning visible and infrared visual features with
explicit and enriched semantics and achieves this through
Explicit Semantics-Guided Feature Alignment (ESFA), Consistent Similarity-Guided Indirect Alignment (CSIA), and
Cross-View Semantics Compensation (CVSC). ESFA supplements language descriptions for pedestrian images and builds
the correspondence of image-text pairs, aligning visual features into the semantic space. CSIA introduces the similarity
consistency constraint to indirectly align visual features with
inter-modality semantics, further alleviating the distribution
discrepancy of visual features. CVSC mines comprehensiveness semantics to further facilitate ESFA and CSIA. Experimental results highlight the advancements RSBA achieves
over state-of-the-art methods. In the future, we will further
explore the assistance of semantic information for VIReID.

The advantage of indirect alignment
(1) Why align visual features with inter-modality semantics?
The proposed ESFA achieves alignment of visible visual features and visible semantics, as well as infrared visual features and infrared semantics. If we further align visible visual features and infrared semantics, as well as infrared visual features and visible semantics, the distribution discrepancy between visual features of visible and infrared can be

933

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Acknowledgments

[Li et al., 2021] Huafeng Li, Neng Dong, Zhengtao Yu,
Dapeng Tao, and Guanqiu Qi. Triple adversarial learning and multi-view imaginative reasoning for unsupervised domain adaptation person re-identification. IEEE
Transactions on Circuits and Systems for Video Technology, 32(5):2814–2830, 2021.
[Li et al., 2022] Xulin Li, Yan Lu, Bin Liu, Yating Liu, Guojun Yin, Qi Chu, Jinyang Huang, Feng Zhu, Rui Zhao, and
Nenghai Yu. Counterfactual intervention feature transfer
for visible-infrared person re-identification. In European
Conference on Computer Vision, pages 381–398. Springer,
2022.
[Li et al., 2023] Siyuan Li, Li Sun, and Qingli Li. Clipreid: exploiting vision-language model for image reidentification without concrete text labels. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 1405–1413, 2023.
[Ling et al., 2021] Yongguo Ling, Zhiming Luo, Yaojin Lin,
and Shaozi Li. A multi-constraint similarity learning
with adaptive weighting for visible-thermal person reidentification. In IJCAI, pages 845–851, 2021.
[Liu et al., 2022a] Jialun Liu, Yifan Sun, Feng Zhu, Hongbin Pei, Yi Yang, and Wenhui Li. Learning memoryaugmented unidirectional metrics for cross-modality person re-identification. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 19366–19375, 2022.
[Liu et al., 2022b] Jianan Liu, Jialiang Wang, Nianchang
Huang, Qiang Zhang, and Jungong Han.
Revisiting modality-specific feature compensation for visibleinfrared person re-identification. IEEE Transactions on
Circuits and Systems for Video Technology, 32(10):7226–
7240, 2022.
[Liu et al., 2023] Haotian Liu, Chunyuan Li, Qingyang Wu,
and Yong Jae Lee. Visual instruction tuning. In Advances
in Neural Information Processing Systems, volume 36,
pages 34892–34916. Curran Associates, Inc., 2023.
[Liu et al., 2025] Min Liu, Zhu Zhang, Yuan Bian, Xueping
Wang, Yeqing Sun, Baida Zhang, and Yaonan Wang.
Cross-modality semantic consistency learning for visibleinfrared person re-identification. IEEE Transactions on
Multimedia, 27:568–580, 2025.
[Lu et al., 2023] Hu Lu, Xuezhang Zou, and Pingping
Zhang. Learning progressive modality-shared transformers for effective visible-infrared person re-identification.
In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 1835–1843, 2023.
[Miao et al., 2021] Ziling Miao, Hong Liu, Wei Shi, Wanlu
Xu, and Hanrong Ye. Modality-aware style adaptation for
rgb-infrared person re-identification. In IJCAI, pages 916–
922, 2021.
[Nguyen et al., 2017] Dat Tien Nguyen, Hyung Gil Hong,
Ki Wan Kim, and Kang Ryoung Park. Person recognition
system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

This work was supported in part by the National Natural
Science Foundation of China under Grant 62172212 and
Grant 62427819, the Natural Science Foundation of Jiangsu
Province under Grant BK20230031, the Jiangsu Provincial Science and Technology Major Project under Grant
BG2024042.

References
[Cheng et al., 2023] De Cheng, Xiaolong Wang, Nannan
Wang, Zhen Wang, Xiaoyu Wang, and Xinbo Gao. Crossmodality person re-identification with memory-based contrastive embedding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 425–432,
2023.
[Choi et al., 2020] Seokeon Choi, Sumin Lee, Youngeun
Kim, Taekyung Kim, and Changick Kim. Hi-cmd: Hierarchical cross-modality disentanglement for visible-infrared
person re-identification. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 10257–10266, 2020.
[Dai et al., 2018] Pingyang Dai, Rongrong Ji, Haibin Wang,
Qiong Wu, and Yuyu Huang. Cross-modality person reidentification with generative adversarial training. In IJCAI, volume 1, page 6, 2018.
[Dong et al., 2024a] Neng Dong, Shuanglin Yan, Hao Tang,
Jinhui Tang, and Liyan Zhang. Multi-view information integration and propagation for occluded person reidentification. Information Fusion, 104:102201, 2024.
[Dong et al., 2024b] Neng Dong, Liyan Zhang, Shuanglin
Yan, Hao Tang, and Jinhui Tang. Erasing, transforming, and noising defense network for occluded person reidentification. IEEE Transactions on Circuits and Systems
for Video Technology, 34(6):4458–4472, 2024.
[Feng et al., 2023] Jiawei Feng, Ancong Wu, and Wei-Shi
Zheng. Shape-erased feature learning for visible-infrared
person re-identification. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 22752–22761, 2023.
[Gong et al., 2022] Yunpeng Gong, Liqing Huang, and Lifei
Chen. Person re-identification method based on color attack and joint defence. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 4313–4322, 2022.
[He et al., 2016] Kaiming He, Xiangyu Zhang, Shaoqing
Ren, and Jian Sun. Deep residual learning for image
recognition. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, pages 770–778,
2016.
[He et al., 2024] Shuting He, Weihua Chen, Kai Wang, Hao
Luo, Fan Wang, Wei Jiang, and Henghui Ding. Region generation and assessment network for occluded person re-identification. IEEE Transactions on Information
Forensics and Security, 19:120–132, 2024.

934


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 135ms:
etween different
local features.
Specifically, the IMRM module comprises the image
contextual relationship-mining graph (ICRMG) and text contextual relationship-mining graph (TCRMG), which encode
the mutual semantic relationships between image and text local
features within the intra-modal. The model optimizes local
features by aggregating their semantic contextual information
and interrelationships. Then, on the one hand, we fuse the
relationship-enhanced local features with the global features
as the final image and text feature expression to calculate
similarity. On the other hand, the CMRM module adopts the
relationship-enhanced local features to mine the interrelationships between different modal local features. The CMRM
module employs the nearest neighbor method to model the
semantic relationships between local features within different modalities. Consequently, the cross-modal discriminative
information has been learned, enhancing the cross-modal
semantic correspondence and expression ability of person
features. Finally, through binary classification training, it is
directly determined whether the image and text are the same
person.
B. Intra-Modal Relationship-Mining Module
1) Image Contextual Relationship-Mining Graph: In the
person ReID task, the distinguishable local features in person
images have played an important role. But only relying on
these local features is not enough. It is also very important to
model and mine the semantic relationships between different
features. As mentioned in the introduction, the left description
of Figure 1(b) is ‘A woman was walking in a gray coat.
She was carrying a black bag.’, the right description of
Figure 1(b) is ‘A middle-aged woman was wearing a gray
coat, walking and wearing a black bag.’. It can be observed
that key information such as ‘woman,’ ’gray coat,’ and ‘black
bag’ are detected in both person images. If we only use these
local features for direct matching, it may lead to recognition
errors. Clearly, the interrelationship between the bag and other
local features, such as ‘carrying the bag’ or ‘wearing the bag,’
is crucial for accurately distinguishing different person. To
this end, we design the Image Contextual Relationship-Mining
Graph (ICRMG), which leverages the GNN to model and
mine the potential interrelationship between these person local
features, the detailed architecture of the ICRMG as shown in

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.

5752

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025

Fig. 2. (a) shows the architecture of RMGNet, while (b) and (c) represent the architectures of ICRMG and TCRMG, respectively. Given N image-text pairs
as input, we first use the ViT and BETR models to extract global and local features. The local features are then fed into the IMRM module to mine the mutual
relationships among different local features by the ICRMG and the TCRMG. This can generate the relationship-enhanced image and text local features, FrIel
and FrDel , respectively. In the third step, we fuse global features with relationship-enhanced local features to perform identity recognition training with the
proposed C2FL strategy and CMPM loss. Concurrently, FrIel and FrDel are sent to the CMRM module to learn semantic interrelationships between different
features across different modalities. Finally the binary classification training is conducted on the generated relationship-enhanced cross-modal feature FrCe M .

Figure 2(b). We employ the KNN graph as the graph structure
that allows for more efficient information aggregation to better
capture relationships between different local features and
explore their semantic relevance and similarity. Meanwhile,
this also allows preventing the introduction of excessive noise.
Specifically, firstly, the person image is divided into n
patches, and ViT is employed to extract the local features
I
features FgI of the person, where FlI =
Fl I andI global
I
f 1 , f 2 . . . f n , FlI ∈ R n∗512 . After that, the local features
FlI are transformed into the d-dimensional feature space, as:
H I = w1 FlI + b1

(1)

where w1 and b1 are learning parameters of network. As a
result, the local features
of the pedestrian image can be

expressed as H I = h 1I , h 2I , h 3I . . . h nI . In the second step,
we use the local features of the person image to construct an
undirected weighted relationship graph, that is, the image con
textual relationship-mining graph denoted as G I = H I , E I ,
where H I is nodes within the graph, E I represents edges
i.e. the interrelationship between two connected nodes and
regularized by weighted adjacent matrix. In this way, the
semantic relationship between two nodes (two local features)
can be modeled through the weight of the edges in the graph,
and calculated as follows:

 

ei,I j = ReLu w1I · h iI + b1I · w2I · h Ij + b2I
(2)
where w1I and w2I are the parameters of the fully connected
layer respectively, b1I and b2I are the parameters of the B N
layer respectively, which are used to determine the interrelationship between two nodes h iI and h Ij . Further, the weighted

I
adjacency matrix A(i,
j) of the ICRMG can be expressed as:
(
ei,I j , i f i ̸ = j
I
A(i, j) =
(3)
0,
else

After constructing the graph G I . The weighted edge E I
guides the aggregation of hidden interrelationship information
in other nodes that are semantically relevant to the local feature
H I . The entire process is as follows:


n


X
I
I
I
I
I
frli
= ReLu 
A(i,
+ h iI 
(4)
j) × w · h j + b
j=1
I represents the local feature that has fused relationwhere frli
ship information between other local features. In this way, the
relationship-enhanced person image local features FrlI can be
generated. Finally, in order to prevent oscillation interference
in the early stage of model training. We aggregate the initial
local features and enhance local features through maximum
pooling and average pooling:
 
 
FrIel = α  Max Pool FlI + (1 − α)  Avg Pool FrlI
(5)

where α is the adjustment parameter, FrIel represent the final
relationship-enhanced image loacl features.
2) Text Contextual Relationship-Mining Graph: After
obtaining the relationship-enhanced person image local features, in order to obtain the relationship-enhanced person text local features. We design the text contextual
relationship-mining graph (TCRMG) as shown in Figure 2(c).
Since text is a type of data with sequential attributes, its
reading order has a significant impact on feature learning and

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: RMGNet: THE PROGRESSIVE RELATIONSHIP-MINING GRAPH NEURAL NETWORK FOR TI-ReID

matching. For example, the text description ‘A woman was
walking in a gray coat. She was carrying a black bag.’ and
‘A middle-aged woman was wearing a gray coat, walking
and wearing a black bag.’ corresponds to different person
images. When extracting features forward, the first two features described by the two sentences are “woman” and “gray
coat”, which are similar. This interferes with the matching of
image-text, that the person images corresponding to the two
sentences are considered to be the same. In contrast, when
extracting features backward, the network will first extract two
distinct features: ‘carrying a black bag’ and ‘wearing a black
bag’. This can effectively distinguish the description text of
different person images. Therefore, we model the interrelationship of person text local features from both forward and
backward perspectives, respectively.
Different from the method of directly using the bidirectional
long short-term memory (Bi-LSTM) network [46] for text
feature learning. We build two sub-graphs in TCRMG, namely
the forward text contextual relationship-mining sub-graph
(FTCRMSG) and the backward text contextual relationshipmining sub-graph (BTCRMSG). The two sub-graphs model
and learn the correlation of the text local features from the
forward and reverse perspectives, respectively. Specifically,
we first input the description text into the BERT model in forfD
ward sequence to learn the forward text local
features Fl and
o
n
fD
fD
fD
fD
fD
,
= f1 , f2 . . . fn
global features Fg , where Fl
F ∈ R n∗512 . After that, the local features FlD are transformed
to fed into FTCRMSG, as:
−→
fD
Fl = w2 · B E RT W f + b2
(6)
−→
where W f represents the forward input word vector. Similarly,
by inputting person description text in reverse, we can obtain
backward text local features FlbD , as follows:
←−
bD
Fl = w3 · B E RT W b + b3
(7)
Second, we employ GNN to model and mine the interrelationships between person text local features. We need to
construct two directed weighted graphs, that is FTCRMSG
and BTCRMSG, denoted as G f D = (H f D , E f D ) and G bD =
(H bD , E bD ) respectively. Taking the forward local feature
relationship mining as an example, H f D represents the set
of forward person text local features, that is, the nodes in
the FTCRMSG. E f D represents the relationship between two
nodes in the FTCRMSG, which is determined as follows:

 

fD
fD
fD
fD
fD
fD
fD
ei, j = ReLu w1 · h i + b1
· w2 · h j + b2
(8)
fD

where h i

fD

and h j

represent two different forward person

fD
text local features, ei, j represents the mutual semantic rela-

tionship between them. On this basis, the weighted adjacency
matrix of the FTCRMSG can be obtained as:
( fD
ei, j , i f i ̸ = j
fD
A(i, j) =
(9)
0,
else

5753

fD

where A(i, j) is the weight adjacency matrix. The edges and
weights in the graph can represent the semantic correlation
between two words. Thereby, we can model and extract
the interrelationship between different text local features as
follows:


n


X
fD
fD
fD
fD
A(i, j) × w f D · h j + b f D  + h i
frli = ReLu 
j=1

(10)
fD

where frli represents the updated forward text local feature
that has been enhanced with the relationship information.
Therefore, the relationship guide updated backward text local
features can be calculated as:



bD
ei,bDj = ReLu w1bD h ibD + b1bD w2bD h bD
+
b
j
2
(
bD
ei, j , i f i ̸ = j
AbD
(i, j) =
0,
else


n


X
bD
bD bD
bD 
frli
= ReLu 
AbD
+ h ibD
(i, j) × w h j + b
j=1

(11)
Similar to the ICRMG, the final forward and backfD
ward relationship-enhanced person text local features, Fr el
bD
and Fr el , are obtained by fusing the original and the
relationship-enhanced text local features. Finally, we weighted
and fuse the forward and backward relationship-enhanced text
local features to generate the final text relationship enhancement local features FrDel as:




fD
fD
fD
Fr el = α  Max Pool Fl
+ (1 − α)  Avg Pool Frl




bD
bD
FrbD
=
α

Max
Pool
F
+
−
α)

Avg
Pool
F
(1
el
l
rl
fD

FrDel =

Fr el + FrbD
el
2

(12)
2

C. Cross-Modal Relationship-Mining Module
In order to further model and mine the correspondence between these features within different modals,
we have designed the Cross-Modal Relationship-Mining
module (CMRM). First, we construct the Cross-Modal
Relationship-Mining Graph (CMRMG) based on the obtained
image and text relationship-enhanced features,
defined
 I
D
as
G
=
(H,
E),
where
H
=
F
,
F
=
r el
r el
 I
fr el1 , frIel2 . . . frIeln , frDel1 , frDel2 . . . frDeln represent the nodes
on the CMRMG. E represents the edge in the graph, which is
used to model and describe the high-level semantic relationships between different person local features within different
modals. To accurately identify effective relationships and
filter out irrelevant interfering relationships. We determine the
weighted adjacency matrix by computing the nearest neighbor
space of features. Specifically, the cosine distance between
different features is calculated and ranked. Afterward, it is
assumed that there is a semantic relationship between the
two features only if f i and f j are among each other’s top

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.

5754

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025

K neighbors. This is because when the two different local
features are close to each other, their semantic meanings
can be considered to correspond and have interrelationships
with each other. It should be noted that in the process of
determining semantic associations, the intra-modal and intermodal neighbor spaces are calculated separately and then
fused. The weighted adjacency matrix of the CMRMG can
be calculated as:
(

ei, j , i f f i ∈ Nintra f j and f j ∈ Nintra ( f i ) i ̸ = j
Acm =
0,
else

Fig. 3. (a) The variation curve of the Hc-Tri loss and the Acc loss. (b) The
variation curve of the cosine distance between positive samples under the Acc
loss and the Hc-Tri loss guiding.

(13)
where Acm is the weighted adjacency matrix, Nintra (·) is
the nearest neighbor space. The GNN is used to model the
semantic relationships between different local features in the
two modalities and performs learning updates as:


N
X

fr eli = ReLu 
Acm × w  h j + b + h i 
(14)
j=1

where fr eli is the cross-modal relationship-enhanced person local feature. Further, we fuse the local features after
cross-modal relationship enhancement to obtain the final crossmodal relationship-enhanced feature FrCe M . Finally, FrCe M is
used for binary classification training to determine directly
whether the person image matches the description text.
D. Training
Network training plays a crucial role in ReID research [47],
[48]. During the training process of the method, first, we use
the CMPM loss [27], the Hc-Tri loss [21], and the proposed
Adaptive Corner Center (Acc) loss to guide the network to
learn to extract pedestrian features. Specifically, we use the
CMPM loss to optimize the learning of person image and text
features to alleviate the interference of modal gaps. For a small
batch
n of training data,
o its features and labels can be expressed
as (FiI , F jD ), yi, j

K

i, j=1

, where FiI is the ith person image

feature, F jD is the jth person text feature, that is generated
by fusing the global features and relationship-enhanced local
features. When yi, j = 1, it means that the two features
are correctly matched and belong to the same person. The
probability pi, j that FiI and F jD match can be defined as:


ex p FiI D F jD
F jD
 , F jD =

(15)
pi, j = P
K
ID D
F jD
t=1 ex p Fi Ft
where F jD represents the normalized text features, and FiI D F jD
represents the projection of the image features in the text
feature space. There are multiple correctly matched images and
texts in a batch of training samples. Therefore, the normalized
matching probability is calculated as follows:

ex p yi, j
qi, j = P K
(16)

t=1 ex p yi,t
where qi, j is the normalized final matching probability. Moreover, we normalize through the softmax function, which can

also play a role in label smoothing. By calculating the K L
divergence of the image-to-text projection probability and the
true matching probability, the final image-to-text matching loss
function can be obtained, specifically:


K K
pi, j
1 XX
pi, j log
(17)
L I 2T =
K
qi, j + ε
i=1 j=1

where ε is the parameter to prevent numerical overflow.
After that, the image and text features in eq:15 also need
to be exchanged to calculate the text-to-image matching loss.
Therefore, the final CMPM loss function is expressed as:
L C M P M = L I 2D + L D2I

(18)

In addition, to reduce the intra-class distance, increase the
inter-class distance. We also introduce the Hc-Tri loss based
on the CMPM loss, calculated as follows:
K 




X
M
M̄
M
N
ρ + D Fi , Fi − min D Fi , C j
L H c_T ri =
i=1

j̸ =i

+

(19)
where M and M respresent the modal and M ̸ = M. FiM ,
FiM , and C Nj represent anchor sample features, positive sample features, and negative sample feature centre respectively.
Although the loss can effectively increase the distance between
different person features. However, during training, we find
that the Hc-Tri loss converges quickly in the early stages,
as shown in Figure 3 (a). It can be seen that the Hc-Tri loss
converges to 0 in the 20th iteration. This shows that most
central triples can easily satisfy the boundary constraints. As a
result, effective supervision signals cannot be generated in the
later stage of training. To overcome the limitations of HcTri loss, we propose the more stringent Acc loss, which is
calculated as follows:


K



h

i
X
X
µ
 1−cos FiM , FiM̄ + 1

cos FiM , C Nj
L Acc =
+
K
K
i=1

j̸ =i

(20)
where cos(·) is the cosine similarity, µ is the adaptive weight.
We utilize the cosine space which with the smaller value
range to calculate sample distances, which can better utilize
the training idea of triplet loss. It ensures that the objective
function remains continuous during the optimization process,
enhancing the overall training effectiveness of the network,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: RMGNet: THE PROGRESSIVE RELATIONSHIP-MINING GRAPH NEURAL NETWORK FOR TI-ReID

5755

IV. E XPERIMENTS AND E VALUATION
In this section, we will conduct a series of experiments to
evaluate the performance and effectiveness of the proposed
method on three benchmark TI-ReID datasets.
A. Datasets, Metrics, and Implementation Details

Fig. 4.

The illustration of the proposed Coarse-to-Fine Learning strategy.

as shown in Figure3 (a). However, as can be seen from Figure3
(b), the cosine distance of Acc loss decreases significantly
faster than the Hc-Tri loss after the warm-up stage. This is
because the network is guided by strict metric learning loss
when the initial learning ability is weak. This will cause the
network to be biased towards early learning samples and affect
the generalization ability. Therefore, we design the Coarseto-Fine Learning strategy (C2FL) to achieve better network
training effect. The strategy uses the relatively loose Hc-Tri
loss for coarse guidance in the early stages of training. When
the Hc-Tri loss converges, the strategy automatically switches
to the strictly Acc loss to further reduce modal differences,
as shown in Figure 4. We use the value of the Hc-Tri loss as
an indicator in the C2FL to control the switching, as follows:
′

L C2F L = λ1 · L H c−T ri + λ2 · wL Acc , w = e−β L H c−T ri (21)
where L ′H c−T ri is the cumulative average of L H c−T ri in each
epoch, β is the adjustment parameter, λ1 and λ2 are balance parameters. Finally, because we directly use cross-modal
relationship-enhanced features to process binary classification
tasks, that is, directly determine whether two samples belong
to the same person. Therefore, we use cross-entropy loss for
binary classification training, calculated as follows:


L C E = − y  log ŷ + (1 − y)  log 1 − ŷ
(22)
where y is the ground truth label, ŷ is the prediction results.
The total loss function of the method can be expressed as
follows:
L = ω · L C E + L C M P M + L C2F L

(23)

E. Similarity Calculation
In order to accurately match the person image and text,
we calculate their similarity from two perspectives. First,
we fuse the global features of the person image extracted by
ViT and the enhanced person image local features to generate
the final person image feature F I . In the same way, the
final person text features F D can be obtained. After that,
we calculate the cosine distance D between the two features
as part of the similarity score. Second, we also introduce the
prediction probability generated in the binary classification
task Psame , the total similarity score Ssim as follows:


Ssim = θ1 · D F I , F D + θ2 · Psame
(24)

We utilize three publicly available TI-ReID datasets:
CUHK-PEDES [26], ICFG-PEDES [39], and RSTPReid [31],
to validate the effectiveness of our approach through experiments. To comprehensively evaluate the efficacy of different methods, we adopt Cumulative Matching Characteristic
(CMC) [49] and mean Average Precision (mAP) [50] as
evaluative metrics in our experiment.
1) CUHK-PEDES: The dataset is collected from the screenshots of video and movie camera street shooting images,
comprising 13003 person with distinct identities, along with
40206 images of different person and 80440 description texts.
The training set consists of 11003 person, 34054 person
images, and 68108 person description texts. The verification
set includes 2000 person, 3078 person images, and 6156 person descriptions texts.
2) ICFG-PEDES: The person images in the dataset are all
from the MSMT17 dataset, including 4102 different identities,
and 54522 image-text pairs, and the description text contains
an average of 37.2 words. The dataset is partitioned into
training and test sets. The training set includes 34674 imagetext pairs of 3102 person, while the test set consists of 1,000
person,19,848 images, and their corresponding text descriptions.
3) RSTPReid: Person images in this dataset are gathered
from the MSMT17 dataset too. However, this dataset is
more closely aligned with real-world application scenarios.
It contains 4101 different person, 20505 person images,
and 41010 corresponding text descriptions. The training set
includes 3701 people, 18505 images, and 37010 text descriptions corresponding to images. The validation set consists of
200 people, 1000 images, and 2000 text descriptions.
4) Implementation Details: First, we employ the Pytorch
framework for training and finetuning the proposed method.
During the processing, we use ImageNet pre-trained ViT-Base
and pre-trained BERT-Base-Uncase network to extract image
and text features, respectively. We set the input image size of
the network to 384 ∗ 128, and each text length is unified to
64. The dimensions of image and text features are set to 512.
In addition, to better train the network, we employ data augmentation operations, including random erasing, flipping, and
random cropping. Each training batch consists of 32 imagetext pairs and a total of 80 epochs of iterative training are
conducted. The Stochastic Gradient Descent (SGD) optimizer
is initialized to 1e − 4 at the beginning of the training process
and subsequently reduced by 10% after 50 epochs. We set the
hyperparameter α of the fusion feature to 0.4. Additionally,
the adjustment parameters θ1 and θ2 for similarity calculation
are configured as 0.6 and 0.4 respectively. For the L C2F L
loss, the adjustment parameters λ1 and λ2 are set to 0.35 and
0.65 respectively. The adjustment parameter ω in the total loss
function is set to 0.7. All the training and experiments are
performed with GeForce RTX 3090Ti GPU.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.

5756

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025

TABLE I

TABLE II

T HE A BLATION E XPERIMENT R ESULT (%) OF THE IMRM M ODULE . T HE
B EST R ESULT FOR E ACH I NDICATOR W ILL B E B OLDED

T HE A BLATION E XPERIMENT R ESULT (%) OF THE TPRMG

TABLE III
T HE A BLATION E XPERIMENT R ESULT (%) OF THE CMRM M ODULE

B. Ablation Study
1) Analysis of Intra-Modal Relationship-Mining Module:
The experiments are conducted under the CUHK-PEDES
dataset. The comparison methods mainly select the representative excellent TI-ReID methods. We adopt ViT and BERT
networks as Baseline methods. For comparison, we select
the global matching-based TBPSLD [13] method, the local
matching-based TIPCB [18], and the LGUR [51] method.
To mitigate interference from external variables, all methods
utilized ViT and BERT as feature extractors. During experiments, we match person images and texts by computing the
cosine distance between features as the measure of similarity.
We presented the results in Table I. It can be seen from the
experiment results that the proposed method attained the best
performance, achieving 69.29% in Rank-1, 86.05% in Rank5, and 65.02% in mAP. The following conclusion can be got:
1) Compared with the global matching-based and the local
matching-based methods, our TI-ReID method exhibits superior performance, significantly outperforming them. 2) The
introduction of the interrelationship information between different local features has effectively enhanced the performance
of TI-ReID and the best performance achieved following
the introduction of the IMRM module. Consequently, it’s
from the results that the IMRM module can enhance feature
expressiveness and boost the accuracy of re-identification by
mining the mutual semantic relationships between different
features.
Moreover, to ascertain the influence of the modeling method
of text local feature relationships on the TI-ReID performance.
An analysis of the text contextual relationship-mining graph
is conducted, (a) the relation learning graph that is modeled
using only forward text local features (F-TCRMG), (b) the
relation learning graph that is modeled using only reverse
text local features (B-TCRMG), and (c) the relational learning
graph (F&B-TCRMG) that jointly utilizes both forward and
backward text local features, i.e. the method used in this paper.
We chose BERT as the baseline and Bi-LSTM [46] as the
comparison method. Table II shows the experiment result. The
F&B-TCRMG achieves the best performance with 68.05% in
Rank-1, 84.51% in Rank-5, and 64.32% in mAP. This shows
that the best effect is achieved when jointly using the text
local features in both directions for modeling and learning
semantic relationships. It can be seen from the comparison
that although the Bi-LSTM has learned some reverse text
features. However, it still cannot achieve the learning effect
obtained by inputting text in reverse. Furthermore, when

using backward text local features exclusively, the absence of
supplementation and guidance from forward text local features
would introduce noise, leading to a decrease in recognition
effectiveness. In summary, the TCRMG proposed in this article
adequately learns the relationships between different features
and enhances the distinguishability of person text features by
jointly utilizing forward and backward text local features.
2) Analysis of Cross-Modal Relationship-Mining Module:
In order to verify the effectiveness and contribution of the
design CMRM module, we conduct comparative experiments
on the CUHK-PEDES dataset. In the experiment, we selected
TBPSLD [13], IVT [52], TIPCB [18], and LGUR [51] for
comparison. The ViT and BERT are also selected as baselines,
and other experimental settings are the same as before. The
experiment results are shown in Table III.
It can be seen from Table III that the effectiveness of
TI-ReID has improved after the introduction of the CMRM
module. Compared to the best-performing local matchingbased method LGUR, our method outperforms 7.93% in
Rank-1 and 5.83% in Rank-5. After the introduction of the
IMRM module, the TI-ReID performance has been further
improved with 3.82% in Rank-1, 3.66% in Rank-5, and 4.3%
in mAP. The best results have been achieved when the IMRM
and CMRM are used together. Consequently, it’s from the
results that the designed CMRM module can effectively model
and mine the mutual semantic relationships between different
features within different modals. This relationship information
can be leveraged to enhance the expressive ability of features,
which effectively improves the performance of TI-ReID.
3) Analysis of Training Strategy: First, we perform ablation
experiments on the loss function. We choose ViT and BERT as
baseline methods that use CMPM Loss for training, to extract
person image and text features respectively. In the experiment, the extracted features are directly used for similarity
calculation to match person images and texts. Table IV shows
the TI-ReID performance of the RMGNet on different training
loss. It can be seen that the Acc loss proposed in this paper
achieved the second-best re-identification result. When the Acc

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.


codex
RSBA 把“语义无模态差异”做成主视角，但它又没有停在“用文本监督图像”这个老说法上，而是专门处理了跨模态文本语义冲突和多视角文本不完整这两个阻碍。RMGNet 的触发点则是局部对齐已经不够，真正能区分相似行人的是局部之间的关系，例如“背着包”和“穿着包”的差别。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | nl -ba | rg -n \"III\\. M|A\\. Feature|B\\. Distribution|C\\. Cross|D\\. Intra|E\\. Training|IV\\. E|Ablation|Visualization|Containment|DRM|CMC|IMC\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
20:    20	Representation Modeling (DRM) encodes images and text into
22:    22	allowing the model to capture uncertainty in the representations. Second, Cross-Modal Containment (CMC) aligns the
25:    25	Intra-Modal Containment (IMC) enforces structured learning
116:   116	(DRM) module that encodes pedestrian images and texts into
121:   121	dispersed distributions for confusing negatives Icn , DRM alleviates the inherent conflict between retrieving relevant matches
124:   124	integrates two containment modules: Cross-Modal Containment (CMC) and Intra-Modal Containment (IMC). CMC
128:   128	to the fine-grained specificity of visual content. IMC, in
131:   131	doing so, IMC encourages the model to learn representations
144:   144	• We introduce the Cross-Modal Containment (CMC) module to model the asymmetric partial-order relationship
147:   147	strong/weak positives and confusing negatives. We propose Intra-Modal Containment (IMC) to bridge the gap
267:   267	III. M ETHOD
273:   273	Modeling (DRM) and Cross-/Intra-Modal Containment (CMC
274:   274	and IMC).
275:   275	A. Feature Extraction Backbone
332:   332	B. Distributional Representation Modeling
334:   334	visual appearances. The Distributional Representation Modeling (DRM) module (Fig. 2) addresses this by transforming
395:   395	LDRM to align image-text pairs:
397:   397	LDRM = −
416:   416	initial values set to 1. In this way, DRM transforms point-based
418:   418	C. Cross-Modal Containment
420:   420	images. To align with this semantic property, we propose Cross-Modal Containment (CMC). Unlike traditional
422:   422	noise margin for alignment, CMC explicitly models the partialorder relationship, encouraging the text feature distribution
608:   608	E. Training and Inference
611:   611	LCMC = − log sigmoid(c H(Zvi ⊂ Zt j ))
623:   623	and masked text distributions, CMC encourages the image
638:   638	D. Intra-Modal Containment
639:   639	Occlusions and incomplete descriptions (e.g., missing adjectives) are prevalent in TBPR. Intra-Modal Containment (IMC)
668:   668	), with ratios tuned for balance. The IMC loss,
669:   669	akin to CMC, ensures masked versions have broader variances.
670:   670	Similar to Eq. 13, LIMC is formulated as:
672:   672	LIMC = − log sigmoid(cH(Zvi ⊂ Zmask
682:   682	The primary objective of LIMC is to enhance intra-modal
699:   699	In contrast, the IMC loss focuses solely on modality-internal
726:   726	in the CMC loss and handle Zmask
727:   727	separately in the IMC
731:   731	Furthermore, the CMC loss is specifically designed for
740:   740	In practice, our implementation selectively applies the CMC
744:   744	Ltotal = LDRM + LCMC + LIMC .
748:   748	LDRM ensures the learning of meaningful Gaussian probability
749:   749	distributions, LCMC reflects the more general semantic scope
750:   750	of the text than the image and LIMC enhance the model’s
756:   756	IV. E XPERIMENTS

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' - | nl -ba | rg -n \"3\\.3|3\\.4|3\\.5|Adaptive Parameter|θ|fused|Ablation|Table 3|Visualization|PKP|APE\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | rg -n \"3\\.2|3\\.3|3\\.4|Consistent|Cross-View|Lo2m|Lsc|Equation|Ablation|Further Discussions|Limitations|Table 4|Table 5\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
18:    18	(PKP) method, which mitigates discrepancies in
22:    22	extractor, an Adaptive Parameter Evolution (APE)
23:    23	strategy is proposed. APE optimizes the integration of the old and new models by assessing the
86:    86	an Adaptive Parameter Evolution (APE) strategy. APE evaluates the parameters in the model to assess which ones have
98:    98	• We introduce an Adaptive Parameter Evolution (APE)
183:   183	Person Re-identification (LReID) through two key components: Prototype-guided Knowledge Propagation (PKP) and
184:   184	Adaptive Parameter Evolution (APE). At each training stage
200:   200	PKP
219:   219	APE
251:   251	(b)Prototype-Guided Knowledge Propagation(PKP)
258:   258	gradient updates. Prototype-guided Knowledge Propagation (PKP) module uses a modified triplet loss Lpkp and a standard triplet loss Ltri to
262:   262	added noise, are passed through the PKP module. This process encourages them to diverge from the current task features, enabling the extraction of more discriminative feature
265:   265	The APE module dynamically manages model parameters,
268:   268	model’s parameters (θt ) with those of the previous model
269:   269	(θt−1 ), as shown by the dashed arrows in Fig. 1. This ensures adaptive evolution of model parameters while retaining
270:   270	past knowledge, resulting in a new model θf used for the next
273:   273	3.3
344:   344	(c)Adaptive Parameter Evolution(APE)
407:   407	Figure 2: The Adaptive Parameter Evolution (APE) strategy employs gradient-informed parameter selection and dynamic model parameter
420:   420	θ − η∇θi L, if |∇θi L| > τ
421:   421	θi ← i
424:   424	θi ,
425:   425	if |∇θi L| ≤ τ
441:   441	applying a static fusion strategy [Xu et al., 2024a], APE dynamically adjusts the fusion weight according to the relative
446:   446	ing a new task t, θnew
447:   447	and θold
449:   449	the new and old models, respectively. The fused model parameters θft used are determined as:
457:   457	3.4
459:   459	Adaptive Parameter Evolution Strategy
461:   461	To further facilitate effective knowledge propagation in lifelong person re-identification, we introduce the Adaptive Parameter Evolution (APE) strategy. a dynamic mechanism that
463:   463	the ongoing learning process. APE is characterized by two
470:   470	each model parameter θi , denoted by ∇θi L. These gradients
472:   472	absolute value of the gradient is then computed as |∇θi L|.
480:   480	θft used = αt θnew
481:   481	+ (1 − αt )θold
581:   581	33.5
625:   625	63.4
669:   669	73.3
709:   709	13.4
753:   753	43.5
882:   882	63.4
891:   891	53.4
906:   906	43.5
911:   911	53.4
933:   933	33.3
1023:  1023	dynamic parameter fusion, APE provides an adaptive and robust mechanism for lifelong learning, enabling the previous
1027:  1027	3.5
1110:  1110	storage of old knowledge and the acquisition of new knowledge. And our PKA demonstrates an average mAP/R@1 improvement of 14.1%/13.5% and 13%/14.1% over the best CIL
1265:  1265	Baseline PKP APE
1279:  1279	53.4
1305:  1305	Table 3: Ablation study of different components.
1358:  1358	Figure 5: Ablation studies on hyperparameters.
1362:  1362	Ablation Studies
1367:  1367	Propagation (PKP) and Adaptive Parameter Evolution (APE).
1368:  1368	The baseline model refers to a framework that does not incorporate the PKP and APE modules, meaning it does not utilize
1371:  1371	PKP and APE modules excel at propagating prior knowledge,
1373:  1373	domains. As shown in Tab. 3, furthermore, the APE module
1375:  1375	the PKP module. Consequently, performance is further improved when both modules are used together.
1392:  1392	Visualization Results
1441:  1441	Knowledge Propagation (PKP), which utilizes prototypes and
1443:  1443	Parameter Evolution (APE) to enable dynamic model updates

 succeeded in 0ms:
27:    27	then devise a Consistent Similarity-Guided Indirect Alignment (CSIA) module, which constrains
30:    30	features with cross-modality semantics. Furthermore, we design a Cross-View Semantics Compensation (CVSC) module, which integrates multiview texts and improves the image-text matching of
137:   137	visual features with explicit and enriched semantics for effective VIReID. As shown in Figure 2, it consists of Explicit Semantics-Guided Feature Alignment (ESFA), Consistent Similarity-Guided Indirect Alignment (CSIA), and
138:   138	Cross-View Semantics Compensation (CVSC). ESFA leverages LLaVA to generate textual descriptions for visible and
271:   271	CVSC: Cross-View Semantics Compensation
389:   389	 Consistent Similarity
395:   395	CSIA: Consistent Similarity-Guided Indirect Alignment
466:   466	3.2
522:   522	3.4
667:   667	3.3
669:   669	Consistent Similarity-Guided Indirect
679:   679	We reformulate Equations (5), (6), and (7) as the following
680:   680	Equations (11), (12), and (13), which maximize the similariir
713:   713	Lo2m
716:   716	Lo2m
748:   748	Lo2m
781:   781	redefine the Equation (8) to the following Equation (14), indirectly aligning images with multi-view inter-modality texts:
785:   785	Lo2m
803:   803	Lsc =
805:   805	Cross-View Semantics Compensation
867:   867	Ltotal = Lid + Lmse + λ1 Lo2m
868:   868	con + λ2 Lsc ,
874:   874	the relative importance of Lo2m
875:   875	con and Lsc , respectively.
933:   933	Ablation Studies
935:   935	We evaluate the effectiveness of each component in our proposed RSBA, with the results presented in Table 4. The Rank1 and mAP of Baseline (’0’) are 71.9% and 67.6% under the
937:   937	Effectiveness of ESFA. ESFA aims to introduce explicit semantics to guide the alignment of cross-modality visual features. As shown in Table 4, it improves the Rank-1 and mAP
1038:  1038	83.4 86.0 88.4 80.6
1039:  1039	80.3 83.3
1082:  1082	83.2
1131:  1131	93.4
1134:  1134	83.4
1137:  1137	93.3
1184:  1184	63.2
1216:  1216	77.4 73.2
1229:  1229	Table 4: Ablation studies of our RSBA.
1240:  1240	Table 4, when it is equipped with ESFA, the Rank-1 accuracy
1249:  1249	relative importance of the loss terms Lo2m
1250:  1250	con and Lsc . The
1292:  1292	between intra-modality image-text pairs. As shown in Table 5, the direct alignment (1) reduces the Rank-1 accuracy
1322:  1322	Further Discussions
1325:  1325	with the experimental results presented in Table 5.
1345:  1345	77.4 97.9 73.2
1371:  1371	Limitations
1381:  1381	Table 5: Further discussions of each proposed module.
1387:  1387	more representative and thus guides the more efficient alignment. As shown in Table 5, the recognition performance
1404:  1404	Explicit Semantics-Guided Feature Alignment (ESFA), Consistent Similarity-Guided Indirect Alignment (CSIA), and
1405:  1405	Cross-View Semantics Compensation (CVSC). ESFA supplements language descriptions for pedestrian images and builds

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' - | nl -ba | rg -n \"III\\. M|A\\. Overview|B\\. Intra|C\\. Cross|D\\. Training|E\\. Similarity|ICRMG|TCRMG|CMRM|C2FL|Acc|Ablation\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 108ms:
22:    22	Relationship-Mining (CMRM) module. These modules are
26:    26	inherent in the image and text features. The CMRM module
30:    30	we design the Adaptive Corner Center (Acc) loss and the Coarseto-Fine Learning (C2FL) strategy. These ensure the network
140:   140	Cross-Modal Relationship-Mining (CMRM) module models
154:   154	(C2FL) strategy and the novel adaptive corner center (Acc)
169:   169	(CMRM) module, which is employed to model and learn
173:   173	• The new Coarse-to-Fine Learning (C2FL) strategy and
174:   174	Adaptive Corner Center (Acc) loss are proposed to enable
251:   251	between different features and propose the RMGNet. The network employs the IMRM module and the CMRM module to
256:   256	III. M ETHOD
259:   259	RMGNet, which includes the IMRM and the CMRM module.
260:   260	Next, we propose the C2FL training strategy and the Acc loss.
266:   266	A. Overview of Framework
269:   269	module, and the CMRM module. In the single-modal feature
280:   280	contextual relationship-mining graph (ICRMG) and text contextual relationship-mining graph (TCRMG), which encode
287:   287	similarity. On the other hand, the CMRM module adopts the
288:   288	relationship-enhanced local features to mine the interrelationships between different modal local features. The CMRM
296:   296	B. Intra-Modal Relationship-Mining Module
314:   314	Graph (ICRMG), which leverages the GNN to model and
316:   316	features, the detailed architecture of the ICRMG as shown in
324:   324	Fig. 2. (a) shows the architecture of RMGNet, while (b) and (c) represent the architectures of ICRMG and TCRMG, respectively. Given N image-text pairs
326:   326	relationships among different local features by the ICRMG and the TCRMG. This can generate the relationship-enhanced image and text local features, FrIel
328:   328	proposed C2FL strategy and CMPM loss. Concurrently, FrIel and FrDel are sent to the CMRM module to learn semantic interrelationships between different
373:   373	j) of the ICRMG can be expressed as:
420:   420	relationship-mining graph (TCRMG) as shown in Figure 2(c).
443:   443	feature learning. We build two sub-graphs in TCRMG, namely
582:   582	Similar to the ICRMG, the final forward and backfD
633:   633	C. Cross-Modal Relationship-Mining Module
636:   636	module (CMRM). First, we construct the Cross-Modal
637:   637	Relationship-Mining Graph (CMRMG) based on the obtained
658:   658	on the CMRMG. E represents the edge in the graph, which is
679:   679	fused. The weighted adjacency matrix of the CMRMG can
688:   688	Fig. 3. (a) The variation curve of the Hc-Tri loss and the Acc loss. (b) The
689:   689	variation curve of the cosine distance between positive samples under the Acc
711:   711	D. Training
715:   715	Adaptive Corner Center (Acc) loss to guide the network to
815:   815	later stage of training. To overcome the limitations of HcTri loss, we propose the more stringent Acc loss, which is
832:   832	L Acc =
865:   865	(b), the cosine distance of Acc loss decreases significantly
870:   870	the generalization ability. Therefore, we design the Coarseto-Fine Learning strategy (C2FL) to achieve better network
874:   874	to the strictly Acc loss to further reduce modal differences,
876:   876	an indicator in the C2FL to control the switching, as follows:
879:   879	L C2F L = λ1 · L H c−T ri + λ2 · wL Acc , w = e−β L H c−T ri (21)
897:   897	E. Similarity Calculation
973:   973	T HE A BLATION E XPERIMENT R ESULT (%) OF THE CMRM M ODULE
975:   975	B. Ablation Study
1001:  1001	using only forward text local features (F-TCRMG), (b) the
1003:  1003	text local features (B-TCRMG), and (c) the relational learning
1004:  1004	graph (F&B-TCRMG) that jointly utilizes both forward and
1008:  1008	F&B-TCRMG achieves the best performance with 68.05% in
1020:  1020	effectiveness. In summary, the TCRMG proposed in this article
1026:  1026	design CMRM module, we conduct comparative experiments
1033:  1033	TI-ReID has improved after the introduction of the CMRM
1039:  1039	and CMRM are used together. Consequently, it’s from the
1040:  1040	results that the designed CMRM module can effectively model
1051:  1051	loss. It can be seen that the Acc loss proposed in this paper
1052:  1052	achieved the second-best re-identification result. When the Acc

codex
我现在补读几个被输出截断的段落，主要是 PDA 的分布建模、PKA 的自适应参数融合、RSBA 的间接对齐公式、RMGNet 的图建模细节。最终会按每篇五点写，不会只列模块名。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '460,875p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   460	k=1
   461	
   462	where P and K denote P identities and K visible and K
   463	infrared images of each identity randomly sampled in each
   464	mini-batch. ∥·∥2 represents the Euclidean distance.
   465	
   466	3.2
   467	
   468	Formally, we define the visible and infrared image sets as
   469	Nv
   470	ir Nr
   471	{xvis
   472	i }i=1 and {xi }i=1 , where Nv and Nr represent the sizes
   473	of these two heterogeneous data, respectively. The label set
   474	Np
   475	is denoted as {yi }i=1
   476	, with Np indicates the number of identities. In each mini-batch, N paired cross-modality images
   477	ir N
   478	{xvis
   479	i , xi }i=1 are randomly sampled and their visual features
   480	vis
   481	N ×d
   482	{fi , fiir }N
   483	are extracted, where d is the dimeni=1 ∈ R
   484	sion of features. We employ identity loss and modality-shared
   485	enhancement loss [Lu et al., 2023] to optimize the network:
   486	Lid = −
   487	
   488	2K
   489	
   490	1 X X intra
   491	[ (Dk
   492	− Dkacross )2 ],
   493	2P K p=1
   494	
   495	Explicit Semantics-Guided Feature Alignment
   496	
   497	Most existing frameworks treat VIReID as a pure vision task,
   498	lacking the ability to capture pedestrian semantics that is beneficial for modality alignment. Although CSDN introduces
   499	CLIP and CoOP to address this limitation, the uncertainty and
   500	coarseness of implicit semantics hinder the alignment of visual features into the semantic space. To this end, we propose
   501	ESFA, which leverages LLaVA to generate explicit textual
   502	descriptions and aligns cross-modality images with them.
   503	As illustrated in Figure 2, given a pedestrian image, we
   504	send the request command ’Please describe the characteristics of the pedestrian in the image’ to LLaVa. It responds
   505	with a natural language description ’The pedestrian in the image is a young woman wearing a blue skit, carrying a shoulder bag and holding a parasol’. This description provides
   506	clearer and more detailed explicit semantics, such as age, gender, and clothing, compared to the learnable textual prompt
   507	’A photo of a [X1 ][X2 ][X3 ][X4 ] person’ in CSDN. Notably,
   508	
   509	(1)
   510	
   511	where qi is the one-hot vector of identity label yi . pvis
   512	and pir
   513	i
   514	i
   515	represent classification results of fivis and fiir , respectively.
   516	The modality-shared enhancement loss constrains the average distance between positive pairs across modalities to be
   517	
   518	929
   519	
   520	Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)
   521	
   522	3.4
   523	
   524	LLaVA operates without requiring training and delivers responses quickly, taking approximately 1.2 seconds per image.
   525	Suppose the generated language bases for visible and inir Nr
   526	v
   527	frared images are {livis }N
   528	i=1 and {li }i=1 . In each mini-batch,
   529	vis ir N
   530	ir N
   531	we sample {li , li }i=1 corresponding to {xvis
   532	i , xi }i=1
   533	and input them into the textual encoder to extract feair N
   534	N ×d
   535	tures {tvis
   536	. To align {fivis , fiir }N
   537	i , ti }i=1 ∈ R
   538	i=1 with
   539	vis ir N
   540	{ti , ti }i=1 , we maximize the similarity between them:
   541	Lcon = Li2t + Lt2i ,
   542	
   543	The above two alignments are based on the one-to-one matching between image and text. However, within each modality,
   544	variations in camera views result in descriptions for different
   545	images of the same pedestrian emphasizing distinct objects.
   546	For example, the description for a front-facing image may
   547	highlight age and gender, while that for a rear-facing image is
   548	more likely to focus on hairstyle and backpack. As a result,
   549	semantics derived solely from single-view text are one-sided
   550	and contribute limited to the robustness of visual features. To
   551	address this limitation, we design CVSC to explore the oneto-many correspondence between images and texts.
   552	As illustrated in Figure 2, we introduce an attention fusion module to integrate information in the textual feature
   553	from another view into the textual feature of the current view.
   554	Specifically, for the visible textual feature tvis
   555	i , we randomly
   556	select a textual feature tvis
   557	a that shares the same identity with
   558	while from different views. We compute the similarity
   559	tvis
   560	i
   561	vis
   562	and tvis
   563	between tvis
   564	a to derive the attention weight ai :
   565	i
   566	
   567	
   568	vis T
   569	Wq (tvis
   570	i )(Wk (ta ))
   571	,
   572	(9)
   573	√
   574	avis
   575	= sof tmax
   576	i
   577	d
   578	where Wq and Wk are two linear projection layers. We mulvis
   579	tiply avis
   580	and tvis
   581	a to determine the contribution of ta , and
   582	i
   583	add the the resulting weighted feature to tvis
   584	:
   585	i
   586	vis
   587	vis
   588	vis
   589	tvis
   590	=
   591	t
   592	+
   593	W
   594	(a
   595	W
   596	(t
   597	(10)
   598	a
   599	v
   600	m,i
   601	i
   602	i
   603	a )),
   604	
   605	(5)
   606	
   607	where
   608	N
   609	
   610	Li2t = −
   611	
   612	1 X
   613	exp(s(fivis , tvis
   614	i ))
   615	log PN
   616	vis , tvis ))
   617	N i=1
   618	exp(s(f
   619	j
   620	i
   621	j=1
   622	N
   623	
   624	(6)
   625	
   626	1 X
   627	exp(s(fiir , tir
   628	i ))
   629	−
   630	,
   631	log PN
   632	ir
   633	ir
   634	N i=1
   635	j=1 exp(s(fi , tj ))
   636	N
   637	
   638	Lt2i = −
   639	
   640	vis
   641	1 X
   642	exp(s(tvis
   643	i , fi ))
   644	log PN
   645	vis
   646	vis
   647	N i=1
   648	j=1 exp(s(ti , fj ))
   649	N
   650	
   651	(7)
   652	
   653	ir
   654	1 X
   655	exp(s(tir
   656	i , fi ))
   657	−
   658	log PN
   659	,
   660	ir
   661	ir
   662	N i=1
   663	j=1 exp(s(ti , fj ))
   664	
   665	where s(·) represents the cosine similarity.
   666	
   667	3.3
   668	
   669	Consistent Similarity-Guided Indirect
   670	Alignment
   671	
   672	where Wa and Wv are also linear projection layers. tvis
   673	m,i represents the multi-view textual feature corresponding to livis ,
   674	which contains richer pedestrian semantics as it compensates
   675	for the missing cross-view information in tvis
   676	i . Similarly, we
   677	can acquire the multi-view infrared textual feature tir
   678	m,i .
   679	We reformulate Equations (5), (6), and (7) as the following
   680	Equations (11), (12), and (13), which maximize the similariir
   681	ir
   682	ties between fivis and tvis
   683	m,i , as well as between fi and tm,i :
   684	
   685	ESFA achieves the alignment of images and texts in each
   686	modality; however, it ignores the alignment of images and
   687	inter-modality texts. A straightforward approach is to maximize the similarity between them similar to the above process. However, cross-modality texts describe the same object
   688	with conflicting attributes due to visual ambiguity. For example, the clothing exhibits ’blue’ in the visible text while being
   689	seen as ’gray’ in the infrared one. Forcing the maximization
   690	of similarity between images and inter-modality texts may
   691	disrupt the expressiveness of semantics. To this end, we develop CSIA to explore the indirect alignment between them.
   692	As illustrated in Figure 2, for the visible visual feature fivis ,
   693	CSIA constrains its similarity with the visible text feature tvis
   694	i
   695	to be equal to the similarity between the infrared text feature
   696	vis
   697	tir
   698	i and visible text feature ti , thereby indirectly establishing
   699	the alignment relationship between fivis and tir
   700	i . Similarly,
   701	infrared visual features fiir and visible text features tvis
   702	are
   703	i
   704	indirectly aligned by constraining the similarity between fiir
   705	vis
   706	and tir
   707	and tir
   708	i to be consistent with that between ti
   709	i :
   710	
   711	o2m
   712	o2m
   713	Lo2m
   714	con = Li2t + Lt2i ,
   715	
   716	Lo2m
   717	i2t = −
   718	
   719	1 X
   720	ir vis 2
   721	(s(fivis , tvis
   722	i ) − s(ti , ti ))
   723	N i=1
   724	N
   725	
   726	N
   727	exp(s(fivis , tvis
   728	1 X
   729	m,i ))
   730	log PN
   731	vis
   732	vis
   733	N i=1
   734	j=1 exp(s(fi , tm,j ))
   735	
   736	N
   737	exp(s(fiir , tir
   738	1 X
   739	m,i ))
   740	log PN
   741	,
   742	−
   743	ir
   744	ir
   745	N i=1
   746	j=1 exp(s(fi , tm,j ))
   747	
   748	Lo2m
   749	t2i = −
   750	
   751	N
   752	vis
   753	exp(s(tvis
   754	1 X
   755	m,i , fi ))
   756	log PN
   757	vis
   758	vis
   759	N i=1
   760	j=1 exp(s(tm,i , fj ))
   761	
   762	(11)
   763	
   764	(12)
   765	
   766	(13)
   767	N
   768	ir
   769	exp(s(tir
   770	1 X
   771	m,i , fi ))
   772	−
   773	log PN
   774	.
   775	ir
   776	ir
   777	N i=1
   778	j=1 exp(s(tm,i , fj ))
   779	This process achieves the one-to-many alignment between
   780	images and texts within each modality. In addition, we also
   781	redefine the Equation (8) to the following Equation (14), indirectly aligning images with multi-view inter-modality texts:
   782	N
   783	1 X
   784	ir vis 2
   785	Lo2m
   786	=
   787	(s(fivis , tvis
   788	sc
   789	m,i ) − s(ti , tm,i ))
   790	N i=1
   791	(14)
   792	N
   793	1 X
   794	ir ir
   795	vis ir
   796	2
   797	+
   798	(s(fi , tm,i ) − s(ti , tm,i )) .
   799	N i=1
   800	
   801	N
   802	
   803	Lsc =
   804	
   805	Cross-View Semantics Compensation
   806	
   807	(8)
   808	
   809	1 X
   810	vis ir 2
   811	+
   812	(s(fiir , tir
   813	i ) − s(ti , ti )) .
   814	N i=1
   815	This similarity consistency loss not only achieves the alignment of images with inter-modality texts but also indirectly
   816	maximizes the similarity between infrared and visible texts,
   817	which helps alleviate cross-modality semantic discrepancy,
   818	thus facilitating more effective alignment of visual features.
   819	
   820	930
   821	
   822	Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)
   823	
   824	3.5
   825	
   826	4.3
   827	
   828	Training and Inference
   829	
   830	SYSU-MM01. Table 1 presents the comparison results with
   831	the state-of-the-art methods on the SYSU-MM01 dataset,
   832	showing that RSBA consistently outperforms them across
   833	all settings. Specifically, in the all-search testing mode,
   834	our Rank-1 accuracy and mAP surpass those of the best
   835	generative-based method, ACD, by 4.0% (4.2%) and 3.7%
   836	(3.1%), respectively, while in the indoor-search mode, the
   837	improvements are 8.8% (6.3%) and 5.1% (5.2%). These
   838	gains are attributed to our approach aligning modalities at
   839	the feature level, which circumvents performance limitations
   840	imposed by the generated low-quality images. Compared
   841	to generative-free methods, under the single-shot mode, our
   842	Rank-1 accuracy exceeds that of CycleTrans by 1.9% (0.5%),
   843	and our mAP surpasses HOS-Net by 0.6% (2.1%). This advantage arises from the proposed RSBA aligns visual features
   844	with the semantic space, which are beneficial for alleviating
   845	the modality gap. Furthermore, our RSBA also outperforms
   846	CSDN across all settings, benefiting from its ability to capture
   847	clear, detailed, and rich semantics, as opposed to the coarse
   848	and ambiguous semantics learned by CSDN.
   849	RegDB. We further evaluate the performance of RSBA on
   850	the RegDB dataset, with the quantitative results summarized
   851	in Table 2. Our method achieves superior recognition rates
   852	compared to existing generative-based approaches. For instance, in the visible-to-infrared testing mode, RSBA outperforms TSME in Rank-1 accuracy by 7.9% and surpasses ACD
   853	in mAP by 7.6%. Similarly, our method exhibits significant
   854	performance advantages over state-of-the-art generative-free
   855	methods, such as MBCE and HOS-Net. In comparison with
   856	CSDN, RSBA improves the Rank-1 and mAP by 2.1% and
   857	4.0% in the visible-to-infrared testing mode.
   858	LLCM. We also evaluate the proposed RSBA on the challenging LLCM dataset to provide a comprehensive assessment. As detailed in Table 3, in the visible-to-infrared testing
   859	mode, RSBA achieves a Rank-1 accuracy and mAP that are
   860	1.1% and 0.9% higher, respectively, than those of the stateof-the-art HOS-Net. Similarly, in the infrared-to-visible testing mode, RSBA outperforms HOS-Net with improvements
   861	of 0.9% in Rank-1 accuracy and 1.0% in mAP. These results
   862	further highlight the superiority of our approach.
   863	
   864	The proposed RSBA is trained in an end-to-end manner, with
   865	the total loss expressed as:
   866	o2m
   867	Ltotal = Lid + Lmse + λ1 Lo2m
   868	con + λ2 Lsc ,
   869	
   870	(15)
   871	
   872	where λ1 and λ2 are two hyper-parameters used to balance
   873	o2m
   874	the relative importance of Lo2m
   875	con and Lsc , respectively.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '320,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   320	projection to obtain the final Euclidean text representations.
   321	Similarly, for the j-th text input, we denote its representation
   322	t
   323	t
   324	t
   325	t
   326	t
   327	as fTj = { f0 j , f1 j , . . ., f Mj }, where f0 j corresponds to the global
   328	tj
   329	tj
   330	text feature and { f1 , . . ., f M } correspond to local token-level
   331	features.
   332	B. Distributional Representation Modeling
   333	Pedestrian retrieval faces significant challenges from inherent uncertainties, such as ambiguous descriptions and varying
   334	visual appearances. The Distributional Representation Modeling (DRM) module (Fig. 2) addresses this by transforming
   335	image and text features into Gaussian distributions. This
   336	choice is theoretically grounded in the maximum entropy
   337	principle, as the Gaussian distribution provides the most
   338	conservative and unbiased representation of cross-modal
   339	uncertainty when only the first two moments are known. Crucially, the mathematical closure and analytical tractability of
   340	Gaussian distributions enable the derivation of a closed-form
   341	containment loss, facilitating smooth and computationally efficient similarity learning. By leveraging these properties, the
   342	framework ensures numerically stable gradient backpropagation and a well-defined semantic manifold, allowing for precise
   343	modeling of the complex inclusion relationships between
   344	visual and linguistic modalities.
   345	The derivation of the visual and textual probabilistic representations is as Fig 3. For the i-th image, we start with
   346	the backbone output fIvi and extract the [CLS] token f0vi . The
   347	mean µvi is obtained from f0vi through a sequence of sigmoid,
   348	Linear layer, LayerNorm (LN), and L2 projection, while the
   349	variance σ2vi is obtained solely via a Linear layer. The image is
   350	
   351	Fig. 3. The derivation of the visual and textual probabilistic representations
   352	is as follows: the mean µ is obtained from the deterministic representation
   353	through a sequence of sigmoid, Linear layer, LayerNorm (LN), and L2
   354	projection, while the variance δ is obtained solely via a Linear layer.
   355	
   356	thus represented as a Gaussian distribution Zvi ∼ N (µvi , σ2vi ).
   357	t
   358	t
   359	Similarly, for the j-th text, f0 j from fTj is processed to compute
   360	the mean µt j through sigmoid, Linear, LN, and L2 projection,
   361	and the variance σ2t j via a Linear layer, forming the Gaussian
   362	representation Zt j ∼ N (µt j , σ2t j ).
   363	Having modeled the image and text features as Gaussian
   364	distributions Zvi ∼ N (µvi , σ2vi ) and Zt j ∼ N (µt j , σ2t j ), the next
   365	step is to quantify the similarity between these probabilistic
   366	representations. For pedestrian retrieval, we aim to ensure that
   367	the distance between distributions is small for certain matching
   368	image-text pairs, and large for non-matching or ambiguous
   369	pairs. To achieve this, we adopt the following distance metric:
   370	2
   371	
   372	d(Zvi , Zt j ) = µvi − µt j 2 + σ2vi + σ2t j .
   373	
   374	(1)
   375	
   376	The distance metric defined in Eq. 1 is theoretically grounded
   377	in the expected squared Euclidean distance between two
   378	independent Gaussian random variables. This formulation
   379	ensures that the visual-textual similarity is modulated by
   380	their respective uncertainties. From an information-geometric
   381	perspective, this allows the model to map semantic ambiguity
   382	directly into the distance manifold, where samples with higher
   383	
   384	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:29 UTC from IEEE Xplore. Restrictions apply.
   385	
   386	YANG et al.: PROBABILISTIC DISTRIBUTION ALIGNMENT FOR TEXT-BASED PERSON RETRIEVAL
   387	
   388	variance are assigned a larger “effective radius”, preventing
   389	them from forming erroneous correspondences with specific,
   390	low-variance samples.
   391	This formula ensures that the distance between images and
   392	texts with low uncertainty is primarily determined by the
   393	difference in the means of their probabilistic representations,
   394	while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
   395	LDRM to align image-text pairs:
   396	X
   397	LDRM = −
   398	qi j log sigmoid (−a · d + b)
   399	i, j
   400	
   401	
   402	− (1 − qi j ) log 1 − sigmoid (−a · d + b) ,
   403	
   404	(2)
   405	
   406	where qi j ∈ {0, 1} indicates whether the image-text pair
   407	matches (qi j = 1) or not (qi j = 0), and d denotes their distance.
   408	The logits −a · d + b (with a > 0) are designed to modulate
   409	the influence of d on the matching probability. Specifically,
   410	for qi j = 1, the objective is to minimize d by maximizing the
   411	logit; conversely, for qi j = 0, the model seeks to increase d to
   412	suppress the matching probability. The negative sign ensures
   413	proper minimization behavior within a binary cross-entropy
   414	formulation, while a and b control the sensitivity and bias
   415	of the response. Parameters a and b are learnable, with their
   416	initial values set to 1. In this way, DRM transforms point-based
   417	representations into probabilistic distribution representations.
   418	C. Cross-Modal Containment
   419	In pedestrian retrieval, text descriptions such as “a person in a jacket” are expected to generalize across multiple
   420	images. To align with this semantic property, we propose Cross-Modal Containment (CMC). Unlike traditional
   421	uncertainty-aware approaches that treat variance merely as a
   422	noise margin for alignment, CMC explicitly models the partialorder relationship, encouraging the text feature distribution
   423	to probabilistically include the corresponding image feature
   424	distribution.
   425	Given a positive image–text pair (Ii , T j ), we model their
   426	feature embeddings as Gaussian distributions:
   427	Zvi ∼ N (µvi , σ2vi ),
   428	
   429	Zt j ∼ N (µt j , σ2t j ),
   430	
   431	(3)
   432	
   433	where µvi and µt j denote the mean features, and σ2vi , σ2t j represent variances that capture feature uncertainty and semantic
   434	scope. To reflect the intuition that textual descriptions are
   435	semantically broader than single images, we define an inclusion measure that quantifies whether the image distribution Zvi
   436	is contained within the text distribution Zt j . The log-inclusion
   437	score is defined as:
   438	Z ∞
   439	H(Zvi ⊂ Zt j ) = log
   440	p2vi (x) pt j (x) dx
   441	−∞
   442	Z ∞
   443	− log
   444	pvi (x) p2t j (x) dx.
   445	(4)
   446	−∞
   447	
   448	The first term evaluates how much of the image density pvi (x)
   449	is concentrated within pt j (x), while the second term measures
   450	how dominant pt j (x) is over pvi (x). A positive value of H
   451	indicates that the image distribution is well contained within
   452	the text distribution. Intuitively, the first term is large when pvi
   453	
   454	7577
   455	
   456	is well aligned with pt j and has a moderate variance, ensuring
   457	sufficient overlap, while the second term is reduced due to the
   458	sharpening effect of squaring pt j , which decreases its overlap
   459	with pvi . This balance between concentration and sharpening
   460	allows H(Zvi ⊂ Zt j ) to be positive, reflecting that the image
   461	distribution is effectively captured by the text distribution.
   462	The log-inclusion score H in Eq. 4 is designed based
   463	on
   464	principle of density concentration. The first term
   465	R ∞ the
   466	2
   467	p
   468	(x)
   469	pt j (x) dx can be interpreted as the cross-correlation
   470	−∞ vi
   471	between the sharpened image distribution and the text distribution. Formally, our containment objective acts as a
   472	probabilistic proxy for partial order modeling in semantic
   473	space. By maximizing H, we enforce a directional constraint
   474	where the text distribution serves as a “probabilistic envelope”
   475	that subsumes the more specific image distribution, aligning
   476	with the asymmetric generality of language relative to visual
   477	content.
   478	Since both pvi (x) and pt j (x) follow Gaussian distributions,
   479	each integral in Eq. (4) can be computed analytically. For
   480	a one-dimensional Gaussian random variable, the general
   481	probability density function is:
   482	
   483	
   484	(x − µ)2
   485	1
   486	,
   487	(5)
   488	exp −
   489	p(x) = √
   490	2σ2
   491	2πσ
   492	where µ and σ2 denote the mean and variance, respectively.
   493	Squaring the density gives:
   494	
   495	
   496	1
   497	(x − µ)2
   498	p2 (x) =
   499	exp
   500	−
   501	.
   502	(6)
   503	2π σ2
   504	σ2
   505	By substituting the corresponding (µvi , σvi ) and (µt j , σt j ) into
   506	this form, the integrals in Eq. (4) can be expressed and solved
   507	in closed form. The exponent of the integrand can be rewritten
   508	as a quadratic form:
   509	(x − µt j )2
   510	(x − µvi )2
   511	+
   512	= Ax2 −Bx + C,
   513	σ2vi
   514	2σ2t j
   515	A=
   516	
   517	(7)
   518	
   519	µ2t j
   520	µt j
   521	µ2vi
   522	1
   523	2µvi
   524	1
   525	+
   526	,
   527	B
   528	=
   529	+
   530	,
   531	C
   532	=
   533	+
   534	.
   535	σ2vi
   536	σ2vi
   537	σ2vi
   538	2σ2t j
   539	σ2t j
   540	2σ2t j
   541	
   542	(8)
   543	
   544	Completing the square yields and using the Gaussian integral identity:
   545	
   546	2
   547	B
   548	B2
   549	2
   550	− Ax + Bx−C = −A x −
   551	+
   552	− C,
   553	(9)
   554	2A
   555	4A
   556	r
   557	Z ∞
   558	B 2
   559	π
   560	e−A( x− 2A ) dx =
   561	.
   562	(10)
   563	A
   564	−∞
   565	The first integral in Eq. (4) becomes:
   566	r
   567	 2
   568	
   569	Z ∞
   570	π
   571	1
   572	B
   573	exp
   574	−
   575	C
   576	.
   577	p2vi (x)pt j (x) dx =
   578	(2π)3/2 σ2vi σt j A
   579	4A
   580	−∞
   581	(11)
   582	Taking the logarithm and omitting constants independent of
   583	µ and σ, we obtain:
   584	Z ∞
   585	log
   586	p2vi (x)pt j (x) dx = −2 log σvi − log σt j
   587	−∞
   588	
   589	2
   590	
   591	B
   592	− 21 log A + 4A
   593	− C.
   594	
   595	(12)
   596	
   597	An analogous expression is derived for the second integral in
   598	Eq. (4) by swapping vi and t j .
   599	
   600	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:29 UTC from IEEE Xplore. Restrictions apply.
   601	
   602	7578
   603	
   604	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 6, JUNE 2026
   605	
   606	Finally, the containment loss is defined as
   607	
   608	E. Training and Inference
   609	
   610	
   611	LCMC = − log sigmoid(c H(Zvi ⊂ Zt j ))
   612	
   613	− log sigmoid(c H(Zvi ⊂ Zmask
   614	)) ,
   615	tj
   616	
   617	(13)
   618	
   619	where c > 0 is a scaling factor and Zmask
   620	denotes the
   621	tj
   622	masked text feature distribution. By combining both complete
   623	and masked text distributions, CMC encourages the image
   624	representation to align with a semantically broader textual
   625	space, ensuring robust cross-modal correspondence even when
   626	textual cues are partially missing.
   627	Although the assumption that the text distribution encompasses the image distribution may not strictly hold in all
   628	cases, our formulation is probabilistic and asymmetric, not
   629	a hard constraint. The inclusion score H(Zvi ⊂ Zt j ) measures
   630	the degree of containment rather than enforcing strict inclusion. This design ensures stable gradients even when textual
   631	descriptions are highly specific or when images contain richer
   632	contextual information. Furthermore, since both image and text
   633	features are modeled as Gaussians with learnable variances,
   634	the model dynamically adjusts its containment confidence
   635	based on the relative semantic scopes of the two modalities,
   636	providing a flexible and effective inductive bias for crossmodal alignment.
   637	
   638	D. Intra-Modal Containment
   639	Occlusions and incomplete descriptions (e.g., missing adjectives) are prevalent in TBPR. Intra-Modal Containment (IMC)
   640	enhances robustness by ensuring masked distributions encompass unmasked ones, reflecting increased uncertainty in partial
   641	inputs.
   642	For images, Iimask occludes 25% of Ii , processed to Zmask
   643	∼
   644	vi
   645	mask 2
   646	N (µmask
   647	,
   648	σ
   649	).
   650	For
   651	text,
   652	20%
   653	of
   654	tokens
   655	in
   656	T
   657	are
   658	masked
   659	j
   660	vi
   661	vi
   662	(e.g., “blue jacket” to “[MASK] jacket”), yielding Zmask
   663	∼
   664	tj
   665	mask
   666	mask 2
   667	N (µt j , σt j
   668	), with ratios tuned for balance. The IMC loss,
   669	akin to CMC, ensures masked versions have broader variances.
   670	Similar to Eq. 13, LIMC is formulated as:
   671	
   672	LIMC = − log sigmoid(cH(Zvi ⊂ Zmask
   673	))
   674	vi
   675	
   676	
   677	− log sigmoid(cH(Zt j ⊂ Zmask
   678	))
   679	.
   680	(14)
   681	tj
   682	The primary objective of LIMC is to enhance intra-modal
   683	robustness, ensuring that the unmasked feature distributions
   684	Zvi and text Zt j are contained within their corresponding
   685	. This containment reflects
   686	masked versions, Zmask
   687	and Zmask
   688	tj
   689	vi
   690	that masked data, which may omit certain attributes, should
   691	exhibit greater uncertainty, thereby naturally encompassing the
   692	distribution of the unmasked data. It is worth noting that,
   693	although a transitive containment relation might exist between
   694	cross-modal features, explicitly enforcing Zvi ⊂ Zmask
   695	in
   696	tj
   697	the cross-modal containment loss provides direct supervision,
   698	leading to stronger gradients and more stable convergence.
   699	In contrast, the IMC loss focuses solely on modality-internal
   700	consistency, complementing the cross-modal constraints and
   701	improving robustness to missing or ambiguous attributes
   702	within each modality.
   703	
   704	Note that in Eq. 13, the second term explicitly models
   705	the cross-modal robustness to incomplete text descriptions
   706	commonly observed in TBPR. Specifically, it aligns the image
   707	distribution Zvi with the masked text distribution Zmask
   708	, which
   709	tj
   710	introduces uncertainty by randomly masking textual tokens.
   711	On the other hand, we do not incorporate Zmask
   712	in Equation 13,
   713	vi
   714	since it may not be semantically contained within the full
   715	or masked text due to the reduced visual information and
   716	increased ambiguity introduced by image masking. Thus, the
   717	containment relation Zmask
   718	⊂ Zt j or Zmask
   719	⊂ Zmask
   720	cannot
   721	vi
   722	vi
   723	tj
   724	be consistently guaranteed. Consequently, we only incorporate
   725	Zmask
   726	in the CMC loss and handle Zmask
   727	separately in the IMC
   728	tj
   729	vi
   730	loss.
   731	Furthermore, the CMC loss is specifically designed for
   732	positive image-text pairs (Ii , T j ), as emphasized in the main
   733	text. In real-world TBPR scenarios, some images (e.g., I sp ) are
   734	generally well-described by the accompanying text, making
   735	containment relations plausible. However, for samples such
   736	as Iwp or Itn , there might be partial mismatch or noise, for
   737	example, illumination variations affecting perceived clothing
   738	color (e.g., trousers appearing white in bright light). Therefore,
   739	containment assumptions do not strictly hold across all cases.
   740	In practice, our implementation selectively applies the CMC
   741	loss only to semantically aligned positive samples (e.g., I sp ),
   742	ensuring the validity of the containment constraint.
   743	Finally, the overall optimization loss is:
   744	Ltotal = LDRM + LCMC + LIMC .
   745	
   746	(15)
   747	
   748	LDRM ensures the learning of meaningful Gaussian probability
   749	distributions, LCMC reflects the more general semantic scope
   750	of the text than the image and LIMC enhance the model’s
   751	robustness to occlusions and incomplete descriptions.
   752	During the testing phase, retrieval is performed using only
   753	the mean vector of the learned Gaussian representations, while
   754	the variance vector is discarded. This ensures computational
   755	efficiency and alignment with standard evaluation protocols.
   756	IV. E XPERIMENTS
   757	A. Datasets and Evaluation Metrics
   758	1) Datasets: We evaluate our method on three widely
   759	recognized challenging datasets, namely CUHK-PEDES [1],
   760	ICFG-PEDES [21], RSTPReid [54] and adopt the Rank-k and

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' - | nl -ba | sed -n '260,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   260	identities.
   261	
   262	added noise, are passed through the PKP module. This process encourages them to diverge from the current task features, enabling the extraction of more discriminative feature
   263	embeddings while effectively leveraging prior knowledge, as
   264	illustrated in Fig. 1.
   265	The APE module dynamically manages model parameters,
   266	assessing the relevance of Dt and selecting parameters based
   267	on their impact on the current task. It then fuses the new
   268	model’s parameters (θt ) with those of the previous model
   269	(θt−1 ), as shown by the dashed arrows in Fig. 1. This ensures adaptive evolution of model parameters while retaining
   270	past knowledge, resulting in a new model θf used for the next
   271	training stage t + 1.
   272	
   273	3.3
   274	
   275	Prototype-guided Knowledge Propagation
   276	
   277	To mitigate catastrophic forgetting in lifelong person reidentification (LReID), we propose a novel prototype-based
   278	non-exemplar learning paradigm. our approach introduces
   279	a novel perspective by leveraging prototypes to guide both
   280	knowledge propagation and feature learning for new tasks, as
   281	illustrated in Fig. 2. we generate more discriminative embeddings, which in turn improves retrieval performance for the
   282	LReID model.
   283	Existing methods that often employ triplet loss directly on
   284	input features of new tasks for feature discrimination. Our
   285	method aims to leverage prototypes to push apart identities
   286	from previous tasks and new tasks within the embedding
   287	space, creating a clear distinction. This facilitates the generation of more refined embeddings. To achieve this, we define a prototype set P = {p1 , . . . , pM }. During training for
   288	a new task, we randomly sample a subset Ps from it, where
   289	Ps contains prototypes of size half the batch size. To enhance the generalization and robustness of these prototypes,
   290	we add Gaussian noise, resulting in augmented prototypes P̃s .
   291	Specifically, for each prototype pl ∈ Ps , we add Gaussian
   292	noise ϵ:
   293	p̃l = pl + βϵ, ϵ ∼ N (0, σ 2 I).
   294	(1)
   295	
   296	5853
   297	
   298	Here p̃l represents the augmented prototype, ϵ is drawn from
   299	a Gaussian distribution with zero mean and covariance matrix
   300	σ 2 I, and β is a hyperparameter controlling the magnitude of
   301	the noise. The prototypes enhanced with noise can cover a
   302	broader feature space during training, preventing the prototypes from becoming too concentrated. This enables a more
   303	comprehensive separation of the features distributed around
   304	them in the new task, thereby improving clustering among
   305	distinct classes and optimizing the representation capability
   306	of the embedding space.
   307	We utilize a modified triplet loss to encourage separation
   308	between features from the new task and augmented prototypes. The standard triplet loss, as defined in [Sun and Mu,
   309	2022; Yu et al., 2023a; Schroff et al., 2015], serves as our
   310	foundation:
   311	Ltri = max(0, ∥a − p∥22 + α − ∥a − n∥22 ).
   312	
   313	(2)
   314	
   315	Here a represents the anchor feature, p represents the positive
   316	feature, and n represents the negative feature, all drawn from
   317	the features of the new task. Triplet loss aims to reduce the
   318	distance between same-identity embeddings and increase the
   319	distance between embeddings of different identities, thereby
   320	enhancing the model’s ability to distinguish between them
   321	and improving retrieval performance.
   322	Our method incorporates two loss terms. The first term
   323	focuses on pushing the features of the new task away from
   324	the augmented prototypes to guarantee the discrimination between old and new tasks. To achieve this, the triplet loss is
   325	modified by removing the positive sample. The following
   326	loss term is used, which aims to maximize the distance between the anchor a (new task feature) and the negative sample
   327	n (augmented prototype):
   328	N
   329	
   330	Lpkp =
   331	
   332	1
   333	1 X
   334	max(0, γ − ∥a − n∥22 ).
   335	N1 i=1
   336	
   337	(3)
   338	
   339	Here γ is a margin to enforces a minimum distance between
   340	the new task features and the augmented prototypes, and N1
   341	
   342	Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)
   343	
   344	(c)Adaptive Parameter Evolution(APE)
   345	Task t-1
   346	
   347	Previous tasks
   348	1-α
   349	Task t
   350	
   351	Model
   352	
   353	��−1
   354	
   355	Influence Weight=α
   356	
   357	ℒ�� +ℒ푐
   358	
   359	Model
   360	
   361	Model
   362	
   363	��
   364	
   365	Task t (during train)
   366	
   367	Task t (after train)
   368	
   369	Subsequent tasks
   370	
   371	dynamic model fusion
   372	
   373	��−1
   374	
   375	Model
   376	
   377	weight
   378	
   379	=α
   380	
   381	��
   382	
   383	weight
   384	
   385	Parameter selection update
   386	initiate
   387	
   388	Model from previous task
   389	
   390	froze
   391	update
   392	
   393	weight
   394	
   395	...
   396	Influence score
   397	
   398	Model
   399	
   400	Model
   401	
   402	=1-α
   403	��푓푢�
   404	
   405	Resulting Fused Model
   406	
   407	Figure 2: The Adaptive Parameter Evolution (APE) strategy employs gradient-informed parameter selection and dynamic model parameter
   408	fusion. We employ gradient-informed selection (left) to update high-influence parameters and performs dynamic model fusion (right) with
   409	weight α. The resulting model is then used for the next task.
   410	
   411	represents the number of triplets sampled from the augmented
   412	prototypes. This loss function is denoted as Lpkp . The second
   413	term ensures the discrimination between different identities
   414	within the new task. We use the following standard triplet
   415	loss function:
   416	
   417	gradient magnitude is greater than the threshold τ , otherwise,
   418	the parameter is preserved. This process can be expressed as:
   419	
   420	θ − η∇θi L, if |∇θi L| > τ
   421	θi ← i
   422	,
   423	(5)
   424	θi ,
   425	if |∇θi L| ≤ τ
   426	
   427	N2
   428	
   429	1 X
   430	max 0, ∥zjt − zpt ∥22 + α − ∥zjt − znt ∥22 .
   431	Ltri =
   432	N2 j=1
   433	
   434	where η is the learning rate and τ is a predefined gradient
   435	magnitude threshold. This approach not only enhances the
   436	efficiency of the training process by focusing on the parameters that are most influential for the current task, but also
   437	mitigates the risk of overfitting by preserving the parameters
   438	that are less sensitive to the current task.
   439	After gradient descent training, dynamic model parameter fusion is performed, which emphasizes maintaining the
   440	compatibility between new and old knowledge. Rather than
   441	applying a static fusion strategy [Xu et al., 2024a], APE dynamically adjusts the fusion weight according to the relative
   442	influence of the current task [Xiao et al., 2023], which is estimated by the size of its training dataset. This allows us to
   443	overcome the shortcomings of previous methods which are
   444	less sensitive to variation between different tasks. After traint−1
   445	t
   446	ing a new task t, θnew
   447	and θold
   448	represent the parameters of
   449	the new and old models, respectively. The fused model parameters θft used are determined as:
   450	
   451	(4)
   452	Here zkt represents the feature of the k-th sample in the new
   453	task, and zpt and znt are the positive and negative samples,
   454	respectively, within the new task. N2 is the number of triplets
   455	sampled from the new task.
   456	
   457	3.4
   458	
   459	Adaptive Parameter Evolution Strategy
   460	
   461	To further facilitate effective knowledge propagation in lifelong person re-identification, we introduce the Adaptive Parameter Evolution (APE) strategy. a dynamic mechanism that
   462	orchestrates the evolution of model parameters in response to
   463	the ongoing learning process. APE is characterized by two
   464	synergistic components: gradient-informed parameter selection and dynamic model parameter fusion. These are designed to maintain the alignment between previously learned
   465	prototypes and the retrained feature extractor, while balancing the retention of old knowledge and the acquisition of new
   466	knowledge throughout the training process.
   467	We posit that not all parameters are equally relevant to a
   468	given learning task, and that a more judicious parameter update strategy is needed [Zhang et al., 2024]. To this end, we
   469	compute the gradient of the loss function L with respect to
   470	each model parameter θi , denoted by ∇θi L. These gradients
   471	quantify the sensitivity of the loss to each parameter, providing a measure of the parameter’s relevance to the task. The
   472	absolute value of the gradient is then computed as |∇θi L|.
   473	Instead of naively updating all parameters, we introduce a
   474	threshold τ . We selectively update a parameter only if its
   475	
   476	5854
   477	
   478	t−1
   479	t
   480	θft used = αt θnew
   481	+ (1 − αt )θold
   482	.
   483	
   484	(6)
   485	
   486	The fusion weight αt is dynamically determined based on the
   487	relative impact of the current task’s dataset. We recognize
   488	that datasets with larger sizes typically offer a more comprehensive representation of the task, hence they should exert a
   489	greater influence on the model parameters. This influence is
   490	quantified using the dataset size, therefore, αt is defined as:
   491	Nt
   492	,
   493	(7)
   494	α t = Pt
   495	i=1 Ni
   496	where Nt denotes the number of samples in the training
   497	dataset for task t.
   498	
   499	Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)
   500	
   501	NonExemplar
   502	
   503	Method
   504	Joint-Train [Xu et al., 2024a]
   505	LwF [Li and Hoiem, 2017]
   506	SPD [Tung and Mori, 2019]
   507	PRAKA* [Shi and Ye, 2023]
   508	PRD* [Asadi et al., 2023]
   509	
   510	CIL
   511	
   512	LReID
   513	
   514	✓
   515	
   516	CRL [Xu et al., 2024b]
   517	AKA [Pu et al., 2021]
   518	AKA† [Pu et al., 2021]
   519	PatchKD [Sun and Mu, 2022]
   520	MEGE [Pu et al., 2023]

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' - | nl -ba | sed -n '296,910p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   296	B. Intra-Modal Relationship-Mining Module
   297	1) Image Contextual Relationship-Mining Graph: In the
   298	person ReID task, the distinguishable local features in person
   299	images have played an important role. But only relying on
   300	these local features is not enough. It is also very important to
   301	model and mine the semantic relationships between different
   302	features. As mentioned in the introduction, the left description
   303	of Figure 1(b) is ‘A woman was walking in a gray coat.
   304	She was carrying a black bag.’, the right description of
   305	Figure 1(b) is ‘A middle-aged woman was wearing a gray
   306	coat, walking and wearing a black bag.’. It can be observed
   307	that key information such as ‘woman,’ ’gray coat,’ and ‘black
   308	bag’ are detected in both person images. If we only use these
   309	local features for direct matching, it may lead to recognition
   310	errors. Clearly, the interrelationship between the bag and other
   311	local features, such as ‘carrying the bag’ or ‘wearing the bag,’
   312	is crucial for accurately distinguishing different person. To
   313	this end, we design the Image Contextual Relationship-Mining
   314	Graph (ICRMG), which leverages the GNN to model and
   315	mine the potential interrelationship between these person local
   316	features, the detailed architecture of the ICRMG as shown in
   317	
   318	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.
   319	
   320	5752
   321	
   322	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
   323	
   324	Fig. 2. (a) shows the architecture of RMGNet, while (b) and (c) represent the architectures of ICRMG and TCRMG, respectively. Given N image-text pairs
   325	as input, we first use the ViT and BETR models to extract global and local features. The local features are then fed into the IMRM module to mine the mutual
   326	relationships among different local features by the ICRMG and the TCRMG. This can generate the relationship-enhanced image and text local features, FrIel
   327	and FrDel , respectively. In the third step, we fuse global features with relationship-enhanced local features to perform identity recognition training with the
   328	proposed C2FL strategy and CMPM loss. Concurrently, FrIel and FrDel are sent to the CMRM module to learn semantic interrelationships between different
   329	features across different modalities. Finally the binary classification training is conducted on the generated relationship-enhanced cross-modal feature FrCe M .
   330	
   331	Figure 2(b). We employ the KNN graph as the graph structure
   332	that allows for more efficient information aggregation to better
   333	capture relationships between different local features and
   334	explore their semantic relevance and similarity. Meanwhile,
   335	this also allows preventing the introduction of excessive noise.
   336	Specifically, firstly, the person image is divided into n
   337	patches, and ViT is employed to extract the local features
   338	I
   339	features FgI of the person, where FlI =
   340	Fl I andI global
   341	I
   342	f 1 , f 2 . . . f n , FlI ∈ R n∗512 . After that, the local features
   343	FlI are transformed into the d-dimensional feature space, as:
   344	H I = w1 FlI + b1
   345	
   346	(1)
   347	
   348	where w1 and b1 are learning parameters of network. As a
   349	result, the local features
   350	of the pedestrian image can be
   351	
   352	expressed as H I = h 1I , h 2I , h 3I . . . h nI . In the second step,
   353	we use the local features of the person image to construct an
   354	undirected weighted relationship graph, that is, the image con
   355	textual relationship-mining graph denoted as G I = H I , E I ,
   356	where H I is nodes within the graph, E I represents edges
   357	i.e. the interrelationship between two connected nodes and
   358	regularized by weighted adjacent matrix. In this way, the
   359	semantic relationship between two nodes (two local features)
   360	can be modeled through the weight of the edges in the graph,
   361	and calculated as follows:
   362	
   363	 
   364	
   365	ei,I j = ReLu w1I · h iI + b1I · w2I · h Ij + b2I
   366	(2)
   367	where w1I and w2I are the parameters of the fully connected
   368	layer respectively, b1I and b2I are the parameters of the B N
   369	layer respectively, which are used to determine the interrelationship between two nodes h iI and h Ij . Further, the weighted
   370	
   371	I
   372	adjacency matrix A(i,
   373	j) of the ICRMG can be expressed as:
   374	(
   375	ei,I j , i f i ̸ = j
   376	I
   377	A(i, j) =
   378	(3)
   379	0,
   380	else
   381	
   382	After constructing the graph G I . The weighted edge E I
   383	guides the aggregation of hidden interrelationship information
   384	in other nodes that are semantically relevant to the local feature
   385	H I . The entire process is as follows:
   386	
   387	
   388	n
   389	
   390	
   391	X
   392	I
   393	I
   394	I
   395	I
   396	I
   397	frli
   398	= ReLu 
   399	A(i,
   400	+ h iI 
   401	(4)
   402	j) × w · h j + b
   403	j=1
   404	I represents the local feature that has fused relationwhere frli
   405	ship information between other local features. In this way, the
   406	relationship-enhanced person image local features FrlI can be
   407	generated. Finally, in order to prevent oscillation interference
   408	in the early stage of model training. We aggregate the initial
   409	local features and enhance local features through maximum
   410	pooling and average pooling:
   411	 
   412	 
   413	FrIel = α  Max Pool FlI + (1 − α)  Avg Pool FrlI
   414	(5)
   415	
   416	where α is the adjustment parameter, FrIel represent the final
   417	relationship-enhanced image loacl features.
   418	2) Text Contextual Relationship-Mining Graph: After
   419	obtaining the relationship-enhanced person image local features, in order to obtain the relationship-enhanced person text local features. We design the text contextual
   420	relationship-mining graph (TCRMG) as shown in Figure 2(c).
   421	Since text is a type of data with sequential attributes, its
   422	reading order has a significant impact on feature learning and
   423	
   424	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.
   425	
   426	ZHANG et al.: RMGNet: THE PROGRESSIVE RELATIONSHIP-MINING GRAPH NEURAL NETWORK FOR TI-ReID
   427	
   428	matching. For example, the text description ‘A woman was
   429	walking in a gray coat. She was carrying a black bag.’ and
   430	‘A middle-aged woman was wearing a gray coat, walking
   431	and wearing a black bag.’ corresponds to different person
   432	images. When extracting features forward, the first two features described by the two sentences are “woman” and “gray
   433	coat”, which are similar. This interferes with the matching of
   434	image-text, that the person images corresponding to the two
   435	sentences are considered to be the same. In contrast, when
   436	extracting features backward, the network will first extract two
   437	distinct features: ‘carrying a black bag’ and ‘wearing a black
   438	bag’. This can effectively distinguish the description text of
   439	different person images. Therefore, we model the interrelationship of person text local features from both forward and
   440	backward perspectives, respectively.
   441	Different from the method of directly using the bidirectional
   442	long short-term memory (Bi-LSTM) network [46] for text
   443	feature learning. We build two sub-graphs in TCRMG, namely
   444	the forward text contextual relationship-mining sub-graph
   445	(FTCRMSG) and the backward text contextual relationshipmining sub-graph (BTCRMSG). The two sub-graphs model
   446	and learn the correlation of the text local features from the
   447	forward and reverse perspectives, respectively. Specifically,
   448	we first input the description text into the BERT model in forfD
   449	ward sequence to learn the forward text local
   450	features Fl and
   451	o
   452	n
   453	fD
   454	fD
   455	fD
   456	fD
   457	fD
   458	,
   459	= f1 , f2 . . . fn
   460	global features Fg , where Fl
   461	F ∈ R n∗512 . After that, the local features FlD are transformed
   462	to fed into FTCRMSG, as:
   463	−→
   464	fD
   465	Fl = w2 · B E RT W f + b2
   466	(6)
   467	−→
   468	where W f represents the forward input word vector. Similarly,
   469	by inputting person description text in reverse, we can obtain
   470	backward text local features FlbD , as follows:
   471	←−
   472	bD
   473	Fl = w3 · B E RT W b + b3
   474	(7)
   475	Second, we employ GNN to model and mine the interrelationships between person text local features. We need to
   476	construct two directed weighted graphs, that is FTCRMSG
   477	and BTCRMSG, denoted as G f D = (H f D , E f D ) and G bD =
   478	(H bD , E bD ) respectively. Taking the forward local feature
   479	relationship mining as an example, H f D represents the set
   480	of forward person text local features, that is, the nodes in
   481	the FTCRMSG. E f D represents the relationship between two
   482	nodes in the FTCRMSG, which is determined as follows:
   483	
   484	 
   485	
   486	fD
   487	fD
   488	fD
   489	fD
   490	fD
   491	fD
   492	fD
   493	ei, j = ReLu w1 · h i + b1
   494	· w2 · h j + b2
   495	(8)
   496	fD
   497	
   498	where h i
   499	
   500	fD
   501	
   502	and h j
   503	
   504	represent two different forward person
   505	
   506	fD
   507	text local features, ei, j represents the mutual semantic rela-
   508	
   509	tionship between them. On this basis, the weighted adjacency
   510	matrix of the FTCRMSG can be obtained as:
   511	( fD
   512	ei, j , i f i ̸ = j
   513	fD
   514	A(i, j) =
   515	(9)
   516	0,
   517	else
   518	
   519	5753
   520	
   521	fD
   522	
   523	where A(i, j) is the weight adjacency matrix. The edges and
   524	weights in the graph can represent the semantic correlation
   525	between two words. Thereby, we can model and extract
   526	the interrelationship between different text local features as
   527	follows:
   528	
   529	
   530	n
   531	
   532	
   533	X
   534	fD
   535	fD
   536	fD
   537	fD
   538	A(i, j) × w f D · h j + b f D  + h i
   539	frli = ReLu 
   540	j=1
   541	
   542	(10)
   543	fD
   544	
   545	where frli represents the updated forward text local feature
   546	that has been enhanced with the relationship information.
   547	Therefore, the relationship guide updated backward text local
   548	features can be calculated as:
   549	
   550	
   551	
   552	bD
   553	ei,bDj = ReLu w1bD h ibD + b1bD w2bD h bD
   554	+
   555	b
   556	j
   557	2
   558	(
   559	bD
   560	ei, j , i f i ̸ = j
   561	AbD
   562	(i, j) =
   563	0,
   564	else
   565	
   566	
   567	n
   568	
   569	
   570	X
   571	bD
   572	bD bD
   573	bD 
   574	frli
   575	= ReLu 
   576	AbD
   577	+ h ibD
   578	(i, j) × w h j + b
   579	j=1
   580	
   581	(11)
   582	Similar to the ICRMG, the final forward and backfD
   583	ward relationship-enhanced person text local features, Fr el
   584	bD
   585	and Fr el , are obtained by fusing the original and the
   586	relationship-enhanced text local features. Finally, we weighted
   587	and fuse the forward and backward relationship-enhanced text
   588	local features to generate the final text relationship enhancement local features FrDel as:
   589	
   590	
   591	
   592	
   593	fD
   594	fD
   595	fD
   596	Fr el = α  Max Pool Fl
   597	+ (1 − α)  Avg Pool Frl
   598	
   599	
   600	
   601	
   602	bD
   603	bD
   604	FrbD
   605	=
   606	α
   607	
   608	Max
   609	Pool
   610	F
   611	+
   612	−
   613	α)
   614	
   615	Avg
   616	Pool
   617	F
   618	(1
   619	el
   620	l
   621	rl
   622	fD
   623	
   624	FrDel =
   625	
   626	Fr el + FrbD
   627	el
   628	2
   629	
   630	(12)
   631	2
   632	
   633	C. Cross-Modal Relationship-Mining Module
   634	In order to further model and mine the correspondence between these features within different modals,
   635	we have designed the Cross-Modal Relationship-Mining
   636	module (CMRM). First, we construct the Cross-Modal
   637	Relationship-Mining Graph (CMRMG) based on the obtained
   638	image and text relationship-enhanced features,
   639	defined
   640	 I
   641	D
   642	as
   643	G
   644	=
   645	(H,
   646	E),
   647	where
   648	H
   649	=
   650	F
   651	,
   652	F
   653	=
   654	r el
   655	r el
   656	 I
   657	fr el1 , frIel2 . . . frIeln , frDel1 , frDel2 . . . frDeln represent the nodes
   658	on the CMRMG. E represents the edge in the graph, which is
   659	used to model and describe the high-level semantic relationships between different person local features within different
   660	modals. To accurately identify effective relationships and
   661	filter out irrelevant interfering relationships. We determine the
   662	weighted adjacency matrix by computing the nearest neighbor
   663	space of features. Specifically, the cosine distance between
   664	different features is calculated and ranked. Afterward, it is
   665	assumed that there is a semantic relationship between the
   666	two features only if f i and f j are among each other’s top
   667	
   668	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.
   669	
   670	5754
   671	
   672	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025
   673	
   674	K neighbors. This is because when the two different local
   675	features are close to each other, their semantic meanings
   676	can be considered to correspond and have interrelationships
   677	with each other. It should be noted that in the process of
   678	determining semantic associations, the intra-modal and intermodal neighbor spaces are calculated separately and then
   679	fused. The weighted adjacency matrix of the CMRMG can
   680	be calculated as:
   681	(
   682	
   683	ei, j , i f f i ∈ Nintra f j and f j ∈ Nintra ( f i ) i ̸ = j
   684	Acm =
   685	0,
   686	else
   687	
   688	Fig. 3. (a) The variation curve of the Hc-Tri loss and the Acc loss. (b) The
   689	variation curve of the cosine distance between positive samples under the Acc
   690	loss and the Hc-Tri loss guiding.
   691	
   692	(13)
   693	where Acm is the weighted adjacency matrix, Nintra (·) is
   694	the nearest neighbor space. The GNN is used to model the
   695	semantic relationships between different local features in the
   696	two modalities and performs learning updates as:
   697	
   698	
   699	N
   700	X
   701	
   702	fr eli = ReLu 
   703	Acm × w  h j + b + h i 
   704	(14)
   705	j=1
   706	
   707	where fr eli is the cross-modal relationship-enhanced person local feature. Further, we fuse the local features after
   708	cross-modal relationship enhancement to obtain the final crossmodal relationship-enhanced feature FrCe M . Finally, FrCe M is
   709	used for binary classification training to determine directly
   710	whether the person image matches the description text.
   711	D. Training
   712	Network training plays a crucial role in ReID research [47],
   713	[48]. During the training process of the method, first, we use
   714	the CMPM loss [27], the Hc-Tri loss [21], and the proposed
   715	Adaptive Corner Center (Acc) loss to guide the network to
   716	learn to extract pedestrian features. Specifically, we use the
   717	CMPM loss to optimize the learning of person image and text
   718	features to alleviate the interference of modal gaps. For a small
   719	batch
   720	n of training data,
   721	o its features and labels can be expressed
   722	as (FiI , F jD ), yi, j
   723	
   724	K
   725	
   726	i, j=1
   727	
   728	, where FiI is the ith person image
   729	
   730	feature, F jD is the jth person text feature, that is generated
   731	by fusing the global features and relationship-enhanced local
   732	features. When yi, j = 1, it means that the two features
   733	are correctly matched and belong to the same person. The
   734	probability pi, j that FiI and F jD match can be defined as:
   735	
   736	
   737	ex p FiI D F jD
   738	F jD
   739	 , F jD =
   740	
   741	(15)
   742	pi, j = P
   743	K
   744	ID D
   745	F jD
   746	t=1 ex p Fi Ft
   747	where F jD represents the normalized text features, and FiI D F jD
   748	represents the projection of the image features in the text
   749	feature space. There are multiple correctly matched images and
   750	texts in a batch of training samples. Therefore, the normalized
   751	matching probability is calculated as follows:
   752	
   753	ex p yi, j
   754	qi, j = P K
   755	(16)
   756	
   757	t=1 ex p yi,t
   758	where qi, j is the normalized final matching probability. Moreover, we normalize through the softmax function, which can
   759	
   760	also play a role in label smoothing. By calculating the K L
   761	divergence of the image-to-text projection probability and the
   762	true matching probability, the final image-to-text matching loss
   763	function can be obtained, specifically:
   764	
   765	
   766	K K
   767	pi, j
   768	1 XX
   769	pi, j log
   770	(17)
   771	L I 2T =
   772	K
   773	qi, j + ε
   774	i=1 j=1
   775	
   776	where ε is the parameter to prevent numerical overflow.
   777	After that, the image and text features in eq:15 also need
   778	to be exchanged to calculate the text-to-image matching loss.
   779	Therefore, the final CMPM loss function is expressed as:
   780	L C M P M = L I 2D + L D2I
   781	
   782	(18)
   783	
   784	In addition, to reduce the intra-class distance, increase the
   785	inter-class distance. We also introduce the Hc-Tri loss based
   786	on the CMPM loss, calculated as follows:
   787	K 
   788	
   789	
   790	
   791	
   792	X
   793	M
   794	M̄
   795	M
   796	N
   797	ρ + D Fi , Fi − min D Fi , C j
   798	L H c_T ri =
   799	i=1
   800	
   801	j̸ =i
   802	
   803	+
   804	
   805	(19)
   806	where M and M respresent the modal and M ̸ = M. FiM ,
   807	FiM , and C Nj represent anchor sample features, positive sample features, and negative sample feature centre respectively.
   808	Although the loss can effectively increase the distance between
   809	different person features. However, during training, we find
   810	that the Hc-Tri loss converges quickly in the early stages,
   811	as shown in Figure 3 (a). It can be seen that the Hc-Tri loss
   812	converges to 0 in the 20th iteration. This shows that most
   813	central triples can easily satisfy the boundary constraints. As a
   814	result, effective supervision signals cannot be generated in the
   815	later stage of training. To overcome the limitations of HcTri loss, we propose the more stringent Acc loss, which is
   816	calculated as follows:
   817	
   818	
   819	K
   820	
   821	
   822	
   823	h
   824	
   825	i
   826	X
   827	X
   828	µ
   829	 1−cos FiM , FiM̄ + 1
   830	
   831	cos FiM , C Nj
   832	L Acc =
   833	+
   834	K
   835	K
   836	i=1
   837	
   838	j̸ =i
   839	
   840	(20)
   841	where cos(·) is the cosine similarity, µ is the adaptive weight.
   842	We utilize the cosine space which with the smaller value
   843	range to calculate sample distances, which can better utilize
   844	the training idea of triplet loss. It ensures that the objective
   845	function remains continuous during the optimization process,
   846	enhancing the overall training effectiveness of the network,
   847	
   848	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.
   849	
   850	ZHANG et al.: RMGNet: THE PROGRESSIVE RELATIONSHIP-MINING GRAPH NEURAL NETWORK FOR TI-ReID
   851	
   852	5755
   853	
   854	IV. E XPERIMENTS AND E VALUATION
   855	In this section, we will conduct a series of experiments to
   856	evaluate the performance and effectiveness of the proposed
   857	method on three benchmark TI-ReID datasets.
   858	A. Datasets, Metrics, and Implementation Details
   859	
   860	Fig. 4.
   861	
   862	The illustration of the proposed Coarse-to-Fine Learning strategy.
   863	
   864	as shown in Figure3 (a). However, as can be seen from Figure3
   865	(b), the cosine distance of Acc loss decreases significantly
   866	faster than the Hc-Tri loss after the warm-up stage. This is
   867	because the network is guided by strict metric learning loss
   868	when the initial learning ability is weak. This will cause the
   869	network to be biased towards early learning samples and affect
   870	the generalization ability. Therefore, we design the Coarseto-Fine Learning strategy (C2FL) to achieve better network
   871	training effect. The strategy uses the relatively loose Hc-Tri
   872	loss for coarse guidance in the early stages of training. When
   873	the Hc-Tri loss converges, the strategy automatically switches
   874	to the strictly Acc loss to further reduce modal differences,
   875	as shown in Figure 4. We use the value of the Hc-Tri loss as
   876	an indicator in the C2FL to control the switching, as follows:
   877	′
   878	
   879	L C2F L = λ1 · L H c−T ri + λ2 · wL Acc , w = e−β L H c−T ri (21)
   880	where L ′H c−T ri is the cumulative average of L H c−T ri in each
   881	epoch, β is the adjustment parameter, λ1 and λ2 are balance parameters. Finally, because we directly use cross-modal
   882	relationship-enhanced features to process binary classification
   883	tasks, that is, directly determine whether two samples belong
   884	to the same person. Therefore, we use cross-entropy loss for
   885	binary classification training, calculated as follows:
   886	
   887	
   888	L C E = − y  log ŷ + (1 − y)  log 1 − ŷ
   889	(22)
   890	where y is the ground truth label, ŷ is the prediction results.
   891	The total loss function of the method can be expressed as
   892	follows:
   893	L = ω · L C E + L C M P M + L C2F L
   894	
   895	(23)
   896	
   897	E. Similarity Calculation
   898	In order to accurately match the person image and text,
   899	we calculate their similarity from two perspectives. First,
   900	we fuse the global features of the person image extracted by
   901	ViT and the enhanced person image local features to generate
   902	the final person image feature F I . In the same way, the
   903	final person text features F D can be obtained. After that,
   904	we calculate the cosine distance D between the two features
   905	as part of the similarity score. Second, we also introduce the
   906	prediction probability generated in the binary classification
   907	task Psame , the total similarity score Ssim as follows:
   908	
   909	
   910	Ssim = θ1 · D F I , F D + θ2 · Psame

codex
为保证“证据闭环”不是我凭摘要猜，我还会定位每篇的消融或可视化。PDA 的消融可能在第 8 页之后，我会只补读和证据相关的小段，不改变你要求的主读法。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' - | nl -ba | rg -n \"Ablation|TABLE II|Feature Distribution|Attention Visualization|PLM|MFGM|Lcpm|Lch|Ldcs|t-SNE|distance\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
24:    24	components: Multi-Feature Generation Module (MFGM) and
25:    25	Prototype Learning Module (PLM). The MFGM generates diversity features closely distributed from modality-shared features
26:    26	to represent pedestrians. Additionally, the PLM utilizes learnable prototypes to excavate latent semantic similarities among
56:    56	minimal feature distances. Nevertheless, such methods often
70:    70	(MFGM) and a Prototype Learning Module (PLM).
71:    71	Specifically, the MFGM employs center-guided pair mining
73:    73	and enriching the feature representation for PLM. The PLM
80:    80	framework, where the MFGM is utilized to generate diverse
81:    81	features that are distributed closely. The PLM module is
131:   131	consisting of two primary components: the Multi-Feature Generation Module (MFGM) and the Prototype Learning Module
132:   132	(PLM). Initially, MFGM processes visual (VIS) and infrared
134:   134	diverse yet closely distributed features. Subsequently, PLM
140:   140	A. Multi-Feature Generation Module (MFGM)
141:   141	The MFGM consists of (i) identical branches, illustrated
170:   170	, we incorporate the centerguided pair mining loss Lcpm , following the DEEN [15]
171:   171	approach. The Lcpm for the VIS and IR modalities are defined
190:   190	where D(·, ·) denotes Euclidean distance. civ and ciir represent
196:   196	Therefore, the total Lcpm can be formulated as:
197:   197	Lcpm = L(cv , cir , civ+ ) + L(cv , cir , ciir+ )
201:   201	B. Prototype Learning Module (PLM)
202:   202	The PLM is illustrated in Fig. 1, utilizing multiple learnable
209:   209	adaptation enables PLM to effectively capture semantically
217:   217	PLM
219:   219	MFGM
233:   233	PLM
235:   235	MFGM
398:   398	Loss Lch decreases the similarity between each prototypes,
400:   400	Lch is defined as follows:
404:   404	Lch = 1 −
417:   417	Dual-Center Separation Loss. We introduce the DualCenter Separation Loss Ldcs to guide the network in discerning
418:   418	pedestrian relationships. The goal of Ldcs is to draw samples
517:   517	samples within a distance threshold ρ1 to enhance diversity.
518:   518	The Ldcs is defined as follows:
546:   546	TABLE II
550:   550	PLM
552:   552	Lch
555:   555	Ldcs
557:   557	MFGM
561:   561	Ldcs =
585:   585	is the number of centroids, ρ1 signifies the threshold distance
586:   586	from the sample to its centroid and ρ2 represents the distance
589:   589	The total loss of the PLM module is as follows:
590:   590	Lplm = Ltri + Lch + Ldcs
594:   594	Besides the Lcpm and Lplm , we further incorporate Lid [18]
597:   597	Ltotal = Lid + Lplm + Lcpm
682:   682	(b) PLM Distance
684:   684	(c) MFGM Distance
690:   690	(f) PLM Distribution
692:   692	(g) MFGM Distribution
696:   696	Fig. 2. (a-d) illustrate the intra-class and inter-class distances of cross-modality features, with intra-class and inter-class distances represented in blue and green,
697:   697	respectively. In (e-h), the t-SNE [24] visualizations illustrate the 2D feature distributions, where circles and triangles denote infrared and visible modalities,
700:   700	TABLE III
777:   777	D. Ablation Studies
779:   779	conducted on the SYSU-MM01 dataset, as presented in Table II, demonstrate the effectiveness of PLM and MFGM
780:   780	components individually and in combination. Including Lch
781:   781	and Ldcs enhances the model to achieve optimal performance.
782:   782	Effectiveness of different numbers of learnable prototypes for the PLM. The PLM utilizes learnable prototypes to
785:   785	PLM and finds that performance improves as the number
789:   789	PLM.
792:   792	Feature Distribution. We conducted an analysis of intraclass and inter-class distance distributions for cross-modality
796:   796	PLM, we observed an increase in the inter-class distance
797:   797	and an enlargement of the gap between the average intraclass distance and inter-class distance. Furthermore, with the
798:   798	incorporation of MFGM, the intra-class distance decreased,
802:   802	PLM, MFGM, and PDM, we conducted t-SNE visualizations
814:   814	Attention Visualization. Figure 3 illustrates attention maps,
821:   821	(MFGM) and Prototype Learning Module (PLM). MFGM extracts diverse features from modality-specific inputs to enhance
822:   822	shared information, aligning their distributions with a centerguided pair mining loss. PLM integrates learnable prototypes to weight modality-specific features based on prototype

exec
/bin/zsh -lc "pdftotext 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | nl -ba | rg -n \"Ablation|TABLE|Distribution|Visualization|Parameter|containment|CMC|IMC|DRM|masked|uncertainty|Domain Generalization|Robustness\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5:     5	Probabilistic Distribution Alignment for
16:    16	Distribution Alignment (PDA), a framework that represents both
19:    19	PDA comprises three main components. First, Distributional
20:    20	Representation Modeling (DRM) encodes images and text into
22:    22	allowing the model to capture uncertainty in the representations. Second, Cross-Modal Containment (CMC) aligns the
23:    23	distributions of text and masked text with their associated image
25:    25	Intra-Modal Containment (IMC) enforces structured learning
27:    27	masked variants, improving robustness to incomplete observations. Experiments on standard benchmarks demonstrate that
87:    87	while rejecting visually similar but semantically distinct negatives (Icn ). Second, even methods that introduce uncertainty
108:   108	Probabilistic Distribution Alignment (PDA), which represents
111:   111	uncertainty but also models the semantic containment relationships between vision and language. As illustrated in Fig. 1(c),
113:   113	which aligns visual and textual distributions, and probabilistic containment, which imposes containment constraints
115:   115	PDA introduces a Distributional Representation Modeling
116:   116	(DRM) module that encodes pedestrian images and texts into
119:   119	captures modality-specific uncertainty. By learning compact
121:   121	dispersed distributions for confusing negatives Icn , DRM alleviates the inherent conflict between retrieving relevant matches
123:   123	samples. To model probabilistic containment, PDA further
124:   124	integrates two containment modules: Cross-Modal Containment (CMC) and Intra-Modal Containment (IMC). CMC
128:   128	to the fine-grained specificity of visual content. IMC, in
129:   129	parallel, constrains each masked sample to remain within the
130:   130	distribution of its corresponding unmasked representation. By
131:   131	doing so, IMC encourages the model to learn representations
133:   133	Together, these containment constraints guide the network
139:   139	• We propose the Probabilistic Distribution Alignment
141:   141	from point-based similarity to distribution-based containment. This allows the model to interpret variance as
142:   142	semantic scope, distinguishing it from prior uncertaintyaware methods that treat variance primarily as a noise
144:   144	• We introduce the Cross-Modal Containment (CMC) module to model the asymmetric partial-order relationship
147:   147	strong/weak positives and confusing negatives. We propose Intra-Modal Containment (IMC) to bridge the gap
149:   149	between complete and masked observations. By unifying
151:   151	containment logic.
171:   171	For instance, IRRA [5] leverages masked language modeling to introduce implicit semantic reasoning, while CFine
200:   200	ambiguity, semantic granularity, and distributional uncertainty
204:   204	for representing cross-modal semantics under uncertainty.
205:   205	B. Probabilistic Distribution Representations
206:   206	Probabilistic distribution representations have been increasingly used to model uncertainty and improve robustness
209:   209	encode feature embeddings with uncertainty estimates reflecting representation quality. In domain generalization, DSU [37]
212:   212	pose embeddings as Gaussians to handle input uncertainty
216:   216	incorporates uncertainty-aware pretraining to refine imagetext interactions, and MUM [41] models multi-granularity
217:   217	uncertainty via batch- and identity-level variances.
220:   220	through containment, holds significant potential for a wide
232:   232	through distributional inclusion. In the domain of hyperspectral image classification, the containment-driven philosophy
235:   235	way to characterize spectral uncertainty and the spatialcontextual containment of complex mineral patterns. Finally,
240:   240	can also represent feature uncertainty. Student’s t [50] distribution has heavier tails, which can capture rare or extreme
243:   243	[51] allows modeling multi-modal uncertainty, accommodating features with multiple plausible states; however, learning
253:   253	and variance, enabling smooth containment-aware similarity
254:   254	learning. These properties make Gaussian distributions particularly suitable for modeling visual-textual uncertainty in our
256:   256	While prior work primarily focuses on uncertainty within
262:   262	shifting the focus from “point-to-set” matching to a structured containment paradigm, our method enables the model
264:   264	alignment. This probabilistic matching and containment mechanism allows for more precise and robust retrieval, effectively
268:   268	In this section, we present the Probabilistic Distribution
272:   272	description of the key modules: Distributional Representation
273:   273	Modeling (DRM) and Cross-/Intra-Modal Containment (CMC
274:   274	and IMC).
332:   332	B. Distributional Representation Modeling
334:   334	visual appearances. The Distributional Representation Modeling (DRM) module (Fig. 2) addresses this by transforming
339:   339	uncertainty when only the first two moments are known. Crucially, the mathematical closure and analytical tractability of
341:   341	containment loss, facilitating smooth and computationally efficient similarity learning. By leveraging these properties, the
392:   392	texts with low uncertainty is primarily determined by the
394:   394	while the distance between images and texts with high uncertainty is increased due to their larger variances. We then derive
395:   395	LDRM to align image-text pairs:
397:   397	LDRM = −
415:   415	of the response. Parameters a and b are learnable, with their
416:   416	initial values set to 1. In this way, DRM transforms point-based
420:   420	images. To align with this semantic property, we propose Cross-Modal Containment (CMC). Unlike traditional
421:   421	uncertainty-aware approaches that treat variance merely as a
422:   422	noise margin for alignment, CMC explicitly models the partialorder relationship, encouraging the text feature distribution
433:   433	where µvi and µt j denote the mean features, and σ2vi , σ2t j represent variances that capture feature uncertainty and semantic
471:   471	between the sharpened image distribution and the text distribution. Formally, our containment objective acts as a
606:   606	Finally, the containment loss is defined as
611:   611	LCMC = − log sigmoid(c H(Zvi ⊂ Zt j ))
622:   622	masked text feature distribution. By combining both complete
623:   623	and masked text distributions, CMC encourages the image
630:   630	the degree of containment rather than enforcing strict inclusion. This design ensures stable gradients even when textual
634:   634	the model dynamically adjusts its containment confidence
639:   639	Occlusions and incomplete descriptions (e.g., missing adjectives) are prevalent in TBPR. Intra-Modal Containment (IMC)
640:   640	enhances robustness by ensuring masked distributions encompass unmasked ones, reflecting increased uncertainty in partial
658:   658	masked
668:   668	), with ratios tuned for balance. The IMC loss,
669:   669	akin to CMC, ensures masked versions have broader variances.
670:   670	Similar to Eq. 13, LIMC is formulated as:
672:   672	LIMC = − log sigmoid(cH(Zvi ⊂ Zmask
682:   682	The primary objective of LIMC is to enhance intra-modal
683:   683	robustness, ensuring that the unmasked feature distributions
685:   685	. This containment reflects
686:   686	masked versions, Zmask
690:   690	that masked data, which may omit certain attributes, should
691:   691	exhibit greater uncertainty, thereby naturally encompassing the
692:   692	distribution of the unmasked data. It is worth noting that,
693:   693	although a transitive containment relation might exist between
697:   697	the cross-modal containment loss provides direct supervision,
699:   699	In contrast, the IMC loss focuses solely on modality-internal
707:   707	distribution Zvi with the masked text distribution Zmask
710:   710	introduces uncertainty by randomly masking textual tokens.
715:   715	or masked text due to the reduced visual information and
717:   717	containment relation Zmask
726:   726	in the CMC loss and handle Zmask
727:   727	separately in the IMC
731:   731	Furthermore, the CMC loss is specifically designed for
735:   735	containment relations plausible. However, for samples such
739:   739	containment assumptions do not strictly hold across all cases.
740:   740	In practice, our implementation selectively applies the CMC
742:   742	ensuring the validity of the containment constraint.
744:   744	Ltotal = LDRM + LCMC + LIMC .
748:   748	LDRM ensures the learning of meaningful Gaussian probability
749:   749	distributions, LCMC reflects the more general semantic scope
750:   750	of the text than the image and LIMC enhance the model’s
780:   780	TABLE I
827:   827	uncertainty in cross modal retrieval, PDA demonstrates a
837:   837	TABLE II
841:   841	TABLE III
851:   851	MUM, our model shows a 3.65% advantage in Rank-1 accuracy, indicating its superior ability to cope with uncertainty.
874:   874	attributed to the introduction of probabilistic distribution representations and the containment-driven alignment mechanism,
878:   878	D. Domain Generalization Comparison
883:   883	and containment driven alignment strategy on large scale real
905:   905	TABLE IV
917:   917	F. Ablation Study
951:   951	1) Ablations on Proposed Components: To evaluate the
954:   954	incorporating the DRM module (Row 1) significantly improves
958:   958	uncertainty via variances. This probabilistic modeling helps
961:   961	investigate the mechanisms of the CMC module, we decouple
962:   962	the effects of complete and masked textual distributions. As
968:   968	broadening the textual distribution’s coverage. The CMC and
969:   969	IMC modules provide further gains by enforcing structured
970:   970	containment relationships. Individually, CMC (Row 4) and
971:   971	IMC (Row 5) improve Rank-1 to 76.23% and 75.88%, respectively. When integrated with DRM, the addition of CMC (Row
972:   972	6) and IMC (Row 7) further boosts Rank-1 to 77.61% and
973:   973	77.29%. This indicates that the refined Gaussian representations learned by DRM serve as a high-quality foundation that
974:   974	facilitates more effective probabilistic containment modeling.
979:   979	roles of distributional representation and asymmetric containment in addressing cross-modal ambiguity and observational
998:   998	TABLE V
999:   999	A BLATION S TUDY ON RSTPR EID , E VALUATING THE I NDIVIDUAL AND C OMBINED E FFECTS OF DRM, CMC, AND IMC M ODULES ,
1002:  1002	TABLE VI
1004:  1004	TABLE VIII
1011:  1011	TABLE VII
1015:  1015	TABLE IX
1026:  1026	effectiveness of our Distributional Representation Modeling
1027:  1027	(DRM), we compare it with representative uncertainty-based
1034:  1034	focus on symmetric distance minimization, DRM utilizes a
1035:  1035	closed-form distance that facilitates asymmetric containmentaware similarity computation. This allows PDA to not only
1038:  1038	scopes. Compared with MUM, which estimates uncertainty via
1039:  1039	batch-level variances, DRM directly learns instance-specific
1045:  1045	uncertainty via the [CLS] token, effectively avoiding the noise
1048:  1048	AUL employs adaptive loss weighting to handle uncertainty,
1050:  1050	contrast, DRM treats features as continuous distributions, providing the necessary mathematical foundation for our unified
1051:  1051	containment-based learning. This shift from “point-to-point”
1057:  1057	distributions on DRM by replacing the Gaussian assumption
1065:  1065	containment score H (Eq. 4), Gaussian distributions allow for
1068:  1068	the optimization of asymmetric containment relationships,
1082:  1082	during the “Pull-Push” process of DRM and fails to reflect
1087:  1087	5) Analysis of Different Mask Ratios: IMC ensures that
1088:  1088	a modality’s masked distribution probabilistically includes
1089:  1089	its unmasked form, improving robustness to occlusions and
1102:  1102	TABLE X
1106:  1106	Fig. 4. Parameter sensitivity analysis of c on RSTPReid. The figure presents
1119:  1119	6) Analysis of Parameter: The scalar parameter c in the
1120:  1120	CMC and IMC losses controls the sharpness of the sigmoid function applied to the containment score H(·). A
1126:  1126	70.70% Rank-1 accuracy. A lower c weakens containment
1130:  1130	default in both CMC and IMC losses.
1146:  1146	TABLE XI
1156:  1156	proves that modeling visual-textual uncertainty through probabilistic distributions is a more efficient and powerful paradigm
1180:  1180	2) Visualization of Heatmaps: Fig. 6 visualizes the attention heatmaps of the Baseline and our proposed PDA model on
1203:  1203	across queries, and robustness to modality-specific uncertainty ultimately enabling more comprehensive and reliable
1205:  1205	Fig. 6. Visualization of attention heatmaps, where redder regions indicate
1208:  1208	Fig. 7. Visualization of distribution representations in our PDA. We display
1214:  1214	3) Visualization of Distribution Representations: To better
1222:  1222	the semantic generalization achieved through the CrossModal Containment (CMC) module, which models the textual
1231:  1231	This paper proposes a Probabilistic Distribution
1235:  1235	distributions, capturing uncertainty and enabling probabilistic
1236:  1236	alignment. We introduce Cross-Modal Containment (CMC)
1240:  1240	(IMC) preserves semantic coherence within each modality.
1245:  1245	modeling and containment-driven matching provide a robust
1251:  1251	our findings highlight probabilistic containment learning as
1296:  1296	[35] J. Chang, Z. Lan, C. Cheng, and Y. Wei, “Data uncertainty learning
1300:  1300	re-identification by modelling feature uncertainty,” in Proc. IEEE/CVF
1308:  1308	[40] Y. Ji et al., “MAP: Multimodal uncertainty-aware vision-language
1312:  1312	uncertainty modeling and semantic alignment for text-to-image person
1464:  1464	uncertainty-based learning for text-based person retrieval,” in Proc.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf' - | nl -ba | rg -n \"TABLE|Fig\\. 3|Fig\\. 4|Optimization Strategy Comparison|Feature Compression Analysis|Tailed Semantics|joint optimization|Opt S|Opt JA|Opt JH|Enhanced\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
12:    12	field of coding for machines. However, existing joint optimization
15:    15	performance improvement stem from the joint optimization itself,
18:    18	module from the joint optimization pipeline by leveraging the
22:    22	pipeline and two joint optimization pipelines to systematically
23:    23	investigate the effectiveness of joint optimization. Our findings
24:    24	reveal that joint optimization alone does not necessarily guarantee performance improvement. Second, we evaluate the influence
29:    29	in enhancing machine vision performance and challenge the assumption that joint optimization alone drives improvement. This
56:    56	In contrast, joint optimization methods aim to simultaneously optimize compression and machine vision algorithms
64:    64	Interestingly, most joint optimization methods follow a
70:    70	This consistent design raises a question: Does the improvement stem from the joint optimization strategy itself, or is
76:    76	joint optimization pipeline, we provide a nuanced perspective
80:    80	module from the joint optimization pipeline. We then design
81:    81	two distinct joint optimization strategies and compare their
88:    88	• We rigorously investigate the effectiveness of joint optimization in feature compression by disentangling its
90:    90	that joint optimization does not necessarily guarantee
93:    93	joint optimization pipelines and analyze their impacts
99:    99	joint optimization.
111:   111	compression and machine vision algorithms. Existing joint optimization methods can be categorized into two primary types:
127:   127	semantics parsing complicates the evaluation of joint optimization effectiveness, as the observed performance improvements
129:   129	module rather than the joint optimization process itself.
132:   132	the impact of joint optimization from the influence of tailed
134:   134	understanding the true value of joint optimization in feature
137:   137	To disentangle the contributions of joint optimization from
166:   166	Fig. 1. Illustration of the existing joint optimization-based pipelines. (a)
180:   180	removed entirely, enabling us to evaluate joint optimization
196:   196	joint optimization process. By leveraging the simplicity and
198:   198	isolate and analyze the contributions of joint optimization in
202:   202	In this section, we present our proposed separate optimization method, two kinds of joint optimization methods, and the
206:   206	denoted as Opt S in Fig. 2. In this method, we separately train
250:   250	are presented in the following subsection. In Opt S, we use
257:   257	joint optimization pipelines, we position the feature compression module immediately before the feature matching stage.
260:   260	allowing us to isolate and evaluate the effect of joint optimization on ReID performance. We propose two joint optimization
262:   262	For the first joint optimization approach, denoted as
263:   263	Opt JA, we optimize all modules jointly. In contrast to
266:   266	triplet loss. The application of Opt JA was expected to facilitate more effective collaboration between FastReID and feature
267:   267	compression and potentially lead to higher performance. However, due to the presence of joint optimization, the influence
273:   273	constraint, we propose the second joint optimization approach,
274:   274	denoted as Opt JH. In Opt JH, only the aggregation module
277:   277	backpropagation. By comparing Opt JA and Opt JH, we aim
279:   279	by joint optimization in ReID pipelines.
292:   292	To fully investigate the impact of joint optimization, we
308:   308	Dataset We use DukeMTMC-reID dataset [38] in our experiments. For joint optimization, the original images are used
314:   314	Training Configuration We adopt sbs R50 ibn configuration for the FastReID [37] network. For joint optimization,
322:   322	TABLE I
352:   352	Fig. 3. Person re-identification performance comparison between different
359:   359	B. Optimization Strategy Comparison
360:   360	We present the performance comparison of Opt S, Opt JA,
361:   361	and Opt JH in Fig. 3, where the performance is measured by
364:   364	different CRs. For CR 256, both joint optimization strategies
365:   365	outperform Opt S, with Opt JA achieving the highest performance. For CR 128, all three optimization strategies produce
367:   367	inverse trend is observed where Opt S outperforms the other
369:   369	joint optimization consistently improves performance across
379:   379	Fig. 3 and Table I. For Opt JH, the gradient backpropagation
382:   382	not affect the feature extraction. Therefore, the original features in Opt JH remain as compact and discriminative as
384:   384	performance of the original features in Opt JH is denoted as
388:   388	Opt S
390:   390	Opt JH
392:   392	Opt JA
394:   394	Opt JH O
396:   396	Opt JA O
434:   434	Opt JH O in Fig. 3, showing consistently high performance
436:   436	In contrast, for Opt JA, the entire FastReID network is
439:   439	The performance of the original features in Opt JA is denoted as Opt JA O in Fig. 3. We observe two key findings.
440:   440	First, the performance of Opt JA O is notably lower than
441:   441	that of Opt JH O, indicating that the information capacity
451:   451	features. Fig. 4 presents the visualizations for 20 randomly
455:   455	identities should be as separated as possible (high distinctiveness). The first row and the second row show the visualizations of the original features extracted from Opt JH and
456:   456	Opt JA, respectively. For Opt JH, although the visualizations
460:   460	extraction in Opt JH. In contrast, for Opt JA, noticeable
464:   464	In general, the feature compactness of Opt JH is better than
465:   465	that of Opt JA. Based on the analysis, we can conclude that
468:   468	D. Feature Compression Analysis
470:   470	comparing the original features and the reconstructed features. For Opt JH, the original features consistently achieve
615:   615	Fig. 4. t-SNE visualization of original features and reconstructed features for Opt JA and Opt JH. First row: original features extracted from the Opt JH
616:   616	pipeline. Second row: original features extracted from the Opt JA pipeline. Third row: reconstructed features from the Opt JA pipeline.
618:   618	TABLE II
624:   624	Opt S
626:   626	Opt JH
628:   628	Opt JA
631:   631	Enhanced
647:   647	E. Tailed Semantics Parsing Analysis
649:   649	partly lost during feature compression. In the case of Opt JA,
650:   650	the improvements brought by joint optimization vary with
658:   658	decoder in Opt JA acts as a semantics parser, reconstructing
661:   661	We visualize the reconstructed features of Opt JA in Fig. 4.
686:   686	decoder outperforms the base decoder in both joint optimization strategies. This finding emphasizes the significance of
688:   688	joint optimization strategy.
694:   694	joint optimization inherently improves compression performance in the field of coding for machines. To examine the
695:   695	effectiveness of joint optimization, we designed one separate
696:   696	optimization pipeline and two joint optimization pipelines
699:   699	joint optimization can negatively impact the feature extraction
701:   701	parsing module plays a critical role in joint optimization,
704:   704	mechanics of joint optimization, providing a deeper understanding of its limitations and potential. Our work highlights
708:   708	joint optimization in the field of coding for machines.

exec
/bin/zsh -lc "pdftotext 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' - | nl -ba | rg -n \"Ablation|TABLE I|TABLE II|TABLE III|TABLE IV|TABLE V|visual|relationship|Acc|C2FL|Hc-Tri|CMRM|IMRM\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 119ms:
676:   676	can be considered to correspond and have interrelationships
679:   679	fused. The weighted adjacency matrix of the CMRMG can
688:   688	Fig. 3. (a) The variation curve of the Hc-Tri loss and the Acc loss. (b) The
689:   689	variation curve of the cosine distance between positive samples under the Acc
690:   690	loss and the Hc-Tri loss guiding.
695:   695	semantic relationships between different local features in the
707:   707	where fr eli is the cross-modal relationship-enhanced person local feature. Further, we fuse the local features after
708:   708	cross-modal relationship enhancement to obtain the final crossmodal relationship-enhanced feature FrCe M . Finally, FrCe M is
714:   714	the CMPM loss [27], the Hc-Tri loss [21], and the proposed
715:   715	Adaptive Corner Center (Acc) loss to guide the network to
731:   731	by fusing the global features and relationship-enhanced local
785:   785	inter-class distance. We also introduce the Hc-Tri loss based
810:   810	that the Hc-Tri loss converges quickly in the early stages,
811:   811	as shown in Figure 3 (a). It can be seen that the Hc-Tri loss
815:   815	later stage of training. To overcome the limitations of HcTri loss, we propose the more stringent Acc loss, which is
832:   832	L Acc =
865:   865	(b), the cosine distance of Acc loss decreases significantly
866:   866	faster than the Hc-Tri loss after the warm-up stage. This is
870:   870	the generalization ability. Therefore, we design the Coarseto-Fine Learning strategy (C2FL) to achieve better network
871:   871	training effect. The strategy uses the relatively loose Hc-Tri
873:   873	the Hc-Tri loss converges, the strategy automatically switches
874:   874	to the strictly Acc loss to further reduce modal differences,
875:   875	as shown in Figure 4. We use the value of the Hc-Tri loss as
876:   876	an indicator in the C2FL to control the switching, as follows:
879:   879	L C2F L = λ1 · L H c−T ri + λ2 · wL Acc , w = e−β L H c−T ri (21)
882:   882	relationship-enhanced features to process binary classification
963:   963	TABLE I
965:   965	TABLE II
967:   967	T HE A BLATION E XPERIMENT R ESULT (%) OF THE IMRM M ODULE . T HE
972:   972	TABLE III
973:   973	T HE A BLATION E XPERIMENT R ESULT (%) OF THE CMRM M ODULE
975:   975	B. Ablation Study
990:   990	introduction of the interrelationship information between different local features has effectively enhanced the performance
992:   992	the introduction of the IMRM module. Consequently, it’s
993:   993	from the results that the IMRM module can enhance feature
995:   995	mining the mutual semantic relationships between different
998:   998	of text local feature relationships on the TI-ReID performance.
999:   999	An analysis of the text contextual relationship-mining graph
1012:  1012	semantic relationships. It can be seen from the comparison
1021:  1021	adequately learns the relationships between different features
1026:  1026	design CMRM module, we conduct comparative experiments
1033:  1033	TI-ReID has improved after the introduction of the CMRM
1036:  1036	IMRM module, the TI-ReID performance has been further
1038:  1038	in mAP. The best results have been achieved when the IMRM
1039:  1039	and CMRM are used together. Consequently, it’s from the
1040:  1040	results that the designed CMRM module can effectively model
1041:  1041	and mine the mutual semantic relationships between different
1042:  1042	features within different modals. This relationship information
1051:  1051	loss. It can be seen that the Acc loss proposed in this paper
1052:  1052	achieved the second-best re-identification result. When the Acc
1060:  1060	TABLE IV
1062:  1062	TABLE VII
1068:  1068	TABLE V
1071:  1071	TABLE VI
1074:  1074	loss and Hc-Tri loss are used together, the best recognition
1078:  1078	result. We can see that under the guidance of Hc-Tri loss,
1083:  1083	the Hc-Tri loss, consequently improving the feature learning
1084:  1084	ability of the network. When Acc loss is introduced, the search
1089:  1089	that the designed Acc loss can effectively alleviate the defects
1090:  1090	in Hc-Tri loss and provide effective supervision information
1096:  1096	effect is achieved when using Hc-Tri loss alone. The primary
1097:  1097	reason is that the constraints of Hc-Tri loss are relatively
1100:  1100	information for network training. After replacing the Hc-Tri
1101:  1101	loss with the Acc loss, the Rank-1, Rank-5, and mAP of
1107:  1107	the proposed C2FL strategy, the best TI-ReID performance
1112:  1112	changes in results, directly applying Acc loss leads to a
1114:  1114	designed C2FL strategy, such issue can be effectively resolved
1116:  1116	C2FL strategy not only adaptively adjusts the method training
1124:  1124	feature-based, the local feature-based and the relationshipbased matching methods, respectively. To mitigate interference
1137:  1137	TABLE VIII
1139:  1139	TABLE IX
1157:  1157	[55] which also employ the relationship between features to
1160:  1160	interrelationship between different features, its performance is
1163:  1163	attention to the interrelationships between features, the proposed method has also significantly improvement, improving
1167:  1167	suitable for learning the mutual semantic relationships between
1169:  1169	noise information. Therefore, the IMRM and CMRM modules
1170:  1170	can better model and mine the mutual semantic relationships
1172:  1172	of the proposed C2FL strategy, the designed method enhances
1182:  1182	Fig. 5. The visualization of cross-modal heatmap of TI-ReID result. The red
1183:  1183	bold font indicates the information describing of the interrelationship between
1185:  1185	feature interrelationship information.
1192:  1192	2.72% in Rank-5, and 1.79% in mAP improvement. Compared with the relationship-based method UMUMSA [55], our
1210:  1210	It can be seen that with the assistance of the proposed IMRM
1211:  1211	and CMRM modules, the network can focus on and learn the
1212:  1212	interrelationships between the local features of person in the
1233:  1233	The difference lies in the mutual semantic relationship between
1235:  1235	effectively model and extract the mutual semantic relationships
1247:  1247	TI-ReID task, which includes the IMRM module, the CMRM
1248:  1248	module, and the C2FL strategy. Specifically, the IMRM
1249:  1249	module models and mines hidden relationships between
1253:  1253	The CMRM module is employed to model the semantic
1254:  1254	correlation and intrinsic relationship between person features
1256:  1256	GNN. This extracts the affinity relationships of person features
1258:  1258	the modal gap. The C2FL learning strategy effectively
1259:  1259	addresses the drawbacks of Hc-Tri loss by employing the
1260:  1260	Acc loss which offers stricter and more effective supervision
1280:  1280	visual object tracking with selective remedy,” IEEE Trans. Circuits
1324:  1324	[15] S. Li, T. Xiao, H. Li, W. Yang, and X. Wang, “Identity-aware textualvisual matching with latent co-attention,” in Proc. IEEE Int. Conf.
1395:  1395	relationships: Attribute correlations in text attribute person search,” IEEE
1485:  1485	semantics-centric visual division,” IEEE Trans. Circuits Syst.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '910,990p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   910	7581
   911	
   912	efficiency, future work could explore lightweight designs such
   913	as partially sharing projection layers between mean and variance branches to reduce parameters, or employing knowledge
   914	distillation where PDA serves as a probabilistic teacher to
   915	guide a lightweight deterministic student model for faster
   916	inference.
   917	F. Ablation Study
   918	
   919	original publications. As shown in the table, our method
   920	consistently achieves the best performance across all evaluated
   921	metrics in both the I→C and C→I transfer scenarios. Compared with methods such as UniPT, CFAM, and NAM, which
   922	rely on the construction of additional pretraining datasets,
   923	our approach outperforms the strongest among them, NAM,
   924	by 2.99% in Rank-1 under the C→I setting and by 1.71%
   925	under the I→C setting. Furthermore, when compared with
   926	methods that aim to enhance the CLIP backbone, including IRRA, IRLT, and RDE, PDA demonstrates even more
   927	significant improvements. Specifically, it surpasses RDE, the
   928	strongest method in this group, by 3.33% in Rank-1 for the
   929	C→I case and by 11.44% for the I→C case. These results
   930	clearly demonstrate that our model not only achieves stateof-the-art performance but also exhibits superior and robust
   931	generalization ability across different domains.
   932	E. Comparisons on Computational Efficiency
   933	Table VI presents a detailed comparison of computational
   934	efficiency. In terms of model scale, PDA contains 153.8M
   935	parameters and requires 15.79G FLOPs. While this is slightly
   936	higher than the noise-reduction-focused RDE [10], PDA
   937	achieves a superior performance-to-cost ratio. Specifically,
   938	with a manageable increase of approximately 0.12 seconds
   939	in training time per step compared to RDE, PDA achieves a
   940	significant Rank-1 improvement of 2.13% on CUHK-PEDES.
   941	Furthermore, the computational demands of PDA are substantially lower than large-scale methods such as APTM [29]
   942	(44.93G FLOPs) and RaSa [64] (71.61G FLOPs). Regarding
   943	running time, PDA requires only 0.471 seconds per training
   944	step and 26.5 seconds per testing epoch on a single RTX 4090
   945	GPU, demonstrating that the probabilistic modeling overhead
   946	is well within acceptable limits for real-world deployment.
   947	These results clearly indicate that PDA strikes an excellent balance between retrieval accuracy and computational overhead,
   948	offering a highly practical solution for large-scale text-based
   949	person retrieval. Meanwhile, to further improve deployment
   950	
   951	1) Ablations on Proposed Components: To evaluate the
   952	contributions of each component in PDA, we conduct an ablation study on CUHK-PEDES, ICFG-PEDES, and RSTPReid,
   953	as summarized in Table V. Focusing on CUHK-PEDES,
   954	incorporating the DRM module (Row 1) significantly improves
   955	the baseline, elevating Rank-1 from 75.57% to 76.41%. This
   956	confirms that representing pedestrian images and texts as
   957	Gaussian distributions allows the model to capture inherent
   958	uncertainty via variances. This probabilistic modeling helps
   959	distinguish fine-grained relevant matches from visually similar
   960	but semantically mismatched “confusing negatives”. To further
   961	investigate the mechanisms of the CMC module, we decouple
   962	the effects of complete and masked textual distributions. As
   963	shown in Rows 2 and 3, while both components individually surpass the baseline, the complete text Zt j provides a
   964	stronger semantic anchor for alignment. More importantly,
   965	their combination (Row 4) yields a synergistic boost, demonstrating that while Zt j ensures precise semantic mapping, Zmask
   966	tj
   967	enhances the model’s tolerance to incomplete descriptions by
   968	broadening the textual distribution’s coverage. The CMC and
   969	IMC modules provide further gains by enforcing structured
   970	containment relationships. Individually, CMC (Row 4) and
   971	IMC (Row 5) improve Rank-1 to 76.23% and 75.88%, respectively. When integrated with DRM, the addition of CMC (Row
   972	6) and IMC (Row 7) further boosts Rank-1 to 77.61% and
   973	77.29%. This indicates that the refined Gaussian representations learned by DRM serve as a high-quality foundation that
   974	facilitates more effective probabilistic containment modeling.
   975	Finally, the full PDA framework (Row 8) achieves the best
   976	performance across all datasets. On CUHK-PEDES, it reaches
   977	Rank-1, Rank-5, and Rank-10 scores of 78.07%, 91.23%,
   978	and 94.70%, respectively. Consistent improvements on ICFGPEDES and RSTPReid further confirm the complementary
   979	roles of distributional representation and asymmetric containment in addressing cross-modal ambiguity and observational
   980	incompleteness.
   981	2) Analysis of Baseline: We build our baseline by equipping IRRA [5] with standard training strategies from TBPR
   982	and incorporating the token selection mechanism proposed
   983	in CFine [25], which is also used in RDE [10]. Although
   984	this enhances performance, the baseline remains weaker than
   985	several state-of-the-art methods. As shown in Table VII, token
   986	selection (ts) improves the baseline’s Rank-1 accuracy from
   987	62.30% to 65.10%, demonstrating its effectiveness. This controlled comparison also shows that our proposed PDA achieves
   988	substantial gains beyond the benefits of token selection alone.
   989	3) Analysis of Backbone: Table VIII shows the performance comparison of PDA and existing methods under the
   990	same Swin-B and BERT-base architecture across three TBPR

exec
/bin/zsh -lc "pdftotext 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | nl -ba | sed -n '1180,1230p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1180	2) Visualization of Heatmaps: Fig. 6 visualizes the attention heatmaps of the Baseline and our proposed PDA model on
  1181	several text–image pairs. As shown, the Baseline model tends
  1182	to focus on coarse and incomplete regions of the pedestrian,
  1183	often ignoring fine-grained visual cues such as clothing texture, color patterns, and accessories. In contrast, PDA produces
  1184	more concentrated and semantically aligned attention maps.
  1185	The highlighted areas correspond more accurately to the key
  1186	textual descriptions, such as “blue and black striped shirt”,
  1187	“white umbrella”, and “pink ruffled dress”. This indicates
  1188	that PDA is capable of capturing fine-grained visual–textual
  1189	
  1190	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:29 UTC from IEEE Xplore. Restrictions apply.
  1191	
  1192	YANG et al.: PROBABILISTIC DISTRIBUTION ALIGNMENT FOR TEXT-BASED PERSON RETRIEVAL
  1193	
  1194	7585
  1195	
  1196	of ID 3955 and ID 3940 are positioned close to each other,
  1197	suggesting that PDA captures semantic proximity between
  1198	visually and textually similar identities. This observation confirms that the probabilistic embeddings not only enhance
  1199	matching accuracy, but also reflect finer-grained cross-modal
  1200	relationships, which point-to-point embeddings often fail to
  1201	capture. In summary, the visualization results validate the
  1202	benefits of probabilistic modeling in PDA. The model demonstrates enhanced semantic alignment, better generalization
  1203	across queries, and robustness to modality-specific uncertainty ultimately enabling more comprehensive and reliable
  1204	cross-modal retrieval.
  1205	Fig. 6. Visualization of attention heatmaps, where redder regions indicate
  1206	higher response intensity.
  1207	
  1208	Fig. 7. Visualization of distribution representations in our PDA. We display
  1209	visual and textual representations of the top-10 retrieved results, with each
  1210	ellipse indicating a 50% confidence region for visual data. The white ellipse
  1211	represents the textual query with a 90% confidence region.
  1212	
  1213	correspondences. Moreover, PDA effectively suppresses irrelevant background regions and confusing negatives, demonstrating its superior discriminative ability in TBPR.
  1214	3) Visualization of Distribution Representations: To better
  1215	illustrate the effectiveness of our probabilistic modeling, we
  1216	conduct a 2D visualization of the distributed embeddings for
  1217	images and their corresponding textual queries. As shown
  1218	in Fig. 7, each white ellipse denotes the distribution of a
  1219	textual query embedding, while the colored points represent
  1220	the top-10 retrieved image embeddings from the gallery.
  1221	The broader spatial coverage of the textual ellipses reflects
  1222	the semantic generalization achieved through the CrossModal Containment (CMC) module, which models the textual
  1223	representation as a probabilistic distribution rather than a
  1224	deterministic point. This design enables the text distribution
  1225	to effectively encompass the true matching image embeddings
  1226	(e.g., ID 3955), while excluding semantically irrelevant or negative samples, indicating strong discrimination. Furthermore,
  1227	Fig. 7 reveals a clustering phenomenon among distributed
  1228	embeddings with related semantics. For instance, the ellipses
  1229	
  1230	V. C ONCLUSION

exec
/bin/zsh -lc "pdftotext 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' - | nl -ba | sed -n '1178,1238p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' - | nl -ba | sed -n '1048,1120p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 112ms:
  1048	person image and text features respectively. In the experiment, the extracted features are directly used for similarity
  1049	calculation to match person images and texts. Table IV shows
  1050	the TI-ReID performance of the RMGNet on different training
  1051	loss. It can be seen that the Acc loss proposed in this paper
  1052	achieved the second-best re-identification result. When the Acc
  1053	
  1054	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.
  1055	
  1056	ZHANG et al.: RMGNet: THE PROGRESSIVE RELATIONSHIP-MINING GRAPH NEURAL NETWORK FOR TI-ReID
  1057	
  1058	5757
  1059	
  1060	TABLE IV
  1061	
  1062	TABLE VII
  1063	
  1064	T HE C OMPARISON R ESULT (%) OF THE D IFFERENT L OSS F UNCTION
  1065	
  1066	C OMPARISON RESULT (%) W ITH OTHER M ETHODS ON CUHK-PEDES
  1067	
  1068	TABLE V
  1069	T HE A BLATION E XPERIMENT R ESULT (%) OF THE L OSS F UNCTION
  1070	
  1071	TABLE VI
  1072	T HE C OMPARISON R ESULT (%) OF THE D IFFERENT T RAINING S TRATEGY
  1073	
  1074	loss and Hc-Tri loss are used together, the best recognition
  1075	effect is achieved. Specifically, compared with other loss and
  1076	baseline, it achieves 68.13% on Rank-1, 84.87% on Rank-5,
  1077	and 63.03% on mAP. Table V shows the ablation experiment
  1078	result. We can see that under the guidance of Hc-Tri loss,
  1079	the TI-ReID model’s Rank-1 increased to 65.73%, Rank-5
  1080	increased to 83.17% and mAP increased to 60.58%. It is
  1081	evident from the results that the network can bring different
  1082	modal features of the same person closer with the aid of
  1083	the Hc-Tri loss, consequently improving the feature learning
  1084	ability of the network. When Acc loss is introduced, the search
  1085	accuracy is further improved. The best results are achieved
  1086	after introducing two loss functions simultaneously. Compared
  1087	with the Baseline method, there is an improvement of 3.99%
  1088	on Rank-1, 2.04% on Rank-5, and 4.02% on mAP. This shows
  1089	that the designed Acc loss can effectively alleviate the defects
  1090	in Hc-Tri loss and provide effective supervision information
  1091	for the network during the entire training process.
  1092	In addition, in order to verify the effectiveness of the proposed Coarse-to-Fine Learning strategy. We perform ablation
  1093	experiments on the training process, and select TIPCB [18]
  1094	for comparison. Table VI shows the experimental results.
  1095	It can be observed from the table that the least favorable
  1096	effect is achieved when using Hc-Tri loss alone. The primary
  1097	reason is that the constraints of Hc-Tri loss are relatively
  1098	loose. When the majority of triples meet the constraints in the
  1099	early stages of training, they can no longer offer supervision
  1100	information for network training. After replacing the Hc-Tri
  1101	loss with the Acc loss, the Rank-1, Rank-5, and mAP of
  1102	
  1103	our proposed method increased by 1.87%, 1.67%, and 0.82%
  1104	respectively. However, due to its relatively strict constraints,
  1105	it will cause the method to overfit some early training samples, thereby hindering the achievement of optimal results.
  1106	When the two losses are comprehensively utilized through
  1107	the proposed C2FL strategy, the best TI-ReID performance
  1108	is achieved, with 69.16% in Rank-1, 86.23% in Rank-5,
  1109	and 64.57% in mAP. Moreover, the TIPCB method, being
  1110	a TI-ReID approach based on the PCB network, exhibits
  1111	high sensitivity to constraint strength. Based on the observed
  1112	changes in results, directly applying Acc loss leads to a
  1113	decrease in performance. In contrast, after introducing the
  1114	designed C2FL strategy, such issue can be effectively resolved
  1115	and further improve accuracy. In summary, the proposed
  1116	C2FL strategy not only adaptively adjusts the method training
  1117	process that can provide continuous and effective supervision
  1118	information, but also demonstrates generalization capabilities.
  1119	C. Comparison With the State-of-the-Art Methods
  1120	To evaluate the superiority of our proposed TI-ReID

 succeeded in 112ms:
  1178	Then, the experiment results in the ICFG-PEDES dataset,
  1179	which has the more complex TI-ReID scene, are shown in
  1180	Table VIII. We can learn that the proposed method achieves
  1181	
  1182	Fig. 5. The visualization of cross-modal heatmap of TI-ReID result. The red
  1183	bold font indicates the information describing of the interrelationship between
  1184	local features. The red boundingbox area is the key area that contains local
  1185	feature interrelationship information.
  1186	
  1187	the most satisfactory performance among all approaches, with
  1188	Rank-1 = 68.35%, Rank-5 = 83.06%, and mAP = 41.62%.
  1189	Similarly, compared with the second-ranked method APTM
  1190	[63], our method has 0.13% gains in Rank-1, 0.19% improvement in Rank-5, and 2.04% in gains in mAP. Compared with
  1191	the TBPS [54], our method has achieved 3.3% in Rank-1,
  1192	2.72% in Rank-5, and 1.79% in mAP improvement. Compared with the relationship-based method UMUMSA [55], our
  1193	method improves 2.73% in Rank-1, 2.52% in Rank-5, and
  1194	2.84% in mAP. It shows that the proposed method can perform
  1195	better TI-ReID in more complex scenarios.
  1196	Finally, we also compared our method with state-of-the-art
  1197	methods on the RSTPReid dataset. The experiment results are
  1198	shown in Table IX. The experimental results reveal that our
  1199	method achieves best performance, yielding 63.67% in Rank1, 84.59% in Rank-5, and 49.58% in mAP. In comparison
  1200	with the UMUMSA [55], our approach yields improvements
  1201	of 0.27%, 1.29%, and 0.3% in Rank-1, Rank-5, and mAP,
  1202	respectively. The comparative experiments conducted across
  1203	three datasets fully prove the effectiveness and generalization
  1204	of the proposed method in the TI-ReID task.
  1205	To provide a more intuitive illustration of the recognition
  1206	effect of our method in the TI-ReID task. We randomly select
  1207	some person text descriptions from the CUHK-PEDES dataset
  1208	to retrieve their corresponding pedestrian images. Figure 5
  1209	shows the cross-modal heat map of the proposed method.
  1210	It can be seen that with the assistance of the proposed IMRM
  1211	and CMRM modules, the network can focus on and learn the
  1212	interrelationships between the local features of person in the
  1213	text and the image, enabling accurate cross-modal person re-
  1214	
  1215	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.
  1216	
  1217	ZHANG et al.: RMGNet: THE PROGRESSIVE RELATIONSHIP-MINING GRAPH NEURAL NETWORK FOR TI-ReID
  1218	
  1219	5759
  1220	
  1221	Fig. 6. Visualization of the text-to-image person re-identification comparison results on the CUHK-PEDES dataset. The first row is the retrieval results of
  1222	the Baseline method, and the second row is the RMGNet. The green bounding boxes indicate correct results, and the red ones indicate incorrect results.
  1223	
  1224	identification. The recognition results are shown in Figure 6,
  1225	where the first and second rows are the recognition results of
  1226	the Baseline and our method, respectively. It can be seen from
  1227	the retrieval results that the proposed method has obtained
  1228	correct retrieval results in Rank-1. While only one of the Rank1 search results of the Baseline method is correct. In addition,
  1229	the correct and incorrect person image results exhibit notable
  1230	similarities in both overall appearance and local details. For
  1231	example, in the 2th search result, all person images are wearing
  1232	white shorts, black pants, black shoes, and carrying black bags.
  1233	The difference lies in the mutual semantic relationship between
  1234	the bag and other features. In this situation, our method can
  1235	effectively model and extract the mutual semantic relationships
  1236	between features and enable accurate differentiation of persons
  1237	with similar appearances [71].
  1238	V. C ONCLUSION

codex
证据部分已经比较完整：几篇都不是只报总分，而是用消融、分布图、热力图或训练曲线去证明它们重定义的问题确实存在。下面我再补 PKA 的组件消融细节，之后开始整理成逐篇拆解。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf' - | nl -ba | sed -n '359,470p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   359	B. Optimization Strategy Comparison
   360	We present the performance comparison of Opt S, Opt JA,
   361	and Opt JH in Fig. 3, where the performance is measured by
   362	mean Average Precision (mAP). Generally, all three optimization strategies exhibit a negative correlation between performance and CRs, i.e., lower CRs lead to higher performance.
   363	However, the performance among the strategies varies for
   364	different CRs. For CR 256, both joint optimization strategies
   365	outperform Opt S, with Opt JA achieving the highest performance. For CR 128, all three optimization strategies produce
   366	comparable results. However, for CRs smaller than 128, an
   367	inverse trend is observed where Opt S outperforms the other
   368	strategies. This finding challenges the common belief that
   369	joint optimization consistently improves performance across
   370	all CRs. Instead, it appears to be effective in extreme high
   371	CRs but loses its advantage in medium and low CRs. In this
   372	subsection, we present this observation, and further analyze it
   373	from two perspectives in the subsequent subsections.
   374	C. Influence of Feature Compression on Feature Extraction
   375	In this subsection, we analyze the influence of feature compression on feature extraction from two perspectives: ReID
   376	performance analysis and feature analysis.
   377	ReID Performance Analysis We compare the person reidentification performance of the original features (the input
   378	of the encoder) with that of the reconstructed features in
   379	Fig. 3 and Table I. For Opt JH, the gradient backpropagation
   380	is truncated between the backbone module and aggregation
   381	module. As a result, the information capacity constraint does
   382	not affect the feature extraction. Therefore, the original features in Opt JH remain as compact and discriminative as
   383	those extracted from the pre-trained FastReID model. The
   384	performance of the original features in Opt JH is denoted as
   385	
   386	CR
   387	
   388	Opt S
   389	
   390	Opt JH
   391	
   392	Opt JA
   393	
   394	Opt JH O
   395	
   396	Opt JA O
   397	
   398	256
   399	128
   400	64
   401	32
   402	16
   403	
   404	26.54
   405	57.44
   406	73.48
   407	79.05
   408	80.50
   409	
   410	35.33
   411	56.51
   412	68.57
   413	74.74
   414	76.37
   415	
   416	40.29
   417	56.16
   418	65.89
   419	71.89
   420	72.41
   421	
   422	81.34
   423	81.29
   424	80.96
   425	80.83
   426	80.8
   427	
   428	54.36
   429	58.15
   430	59.04
   431	61.58
   432	61.61
   433	
   434	Opt JH O in Fig. 3, showing consistently high performance
   435	across different CRs.
   436	In contrast, for Opt JA, the entire FastReID network is
   437	jointly trained with feature compression, allowing the feature
   438	compression to influence the entire feature extraction process.
   439	The performance of the original features in Opt JA is denoted as Opt JA O in Fig. 3. We observe two key findings.
   440	First, the performance of Opt JA O is notably lower than
   441	that of Opt JH O, indicating that the information capacity
   442	constraint negatively impacts the feature extraction process.
   443	Second, the performance varies across different CRs, with
   444	lower CRs achieving higher performance. This is attributed to
   445	the preservation of more information with higher information
   446	capacity.
   447	Feature Analysis In addition to evaluating the performance,
   448	we use t-Distributed Stochastic Neighbor Embedding (t-SNE)
   449	technique to visualize the original features in a 2-dimensional
   450	space and obtain insight into how compression affects the
   451	features. Fig. 4 presents the visualizations for 20 randomly
   452	selected identities, represented in different colors. Each optimization strategy is visualized for three CRs. Ideally, samples
   453	from the same identity (color) should be as concentrated as
   454	possible (high compactness), while samples from different
   455	identities should be as separated as possible (high distinctiveness). The first row and the second row show the visualizations of the original features extracted from Opt JH and
   456	Opt JA, respectively. For Opt JH, although the visualizations
   457	differ among CRs, the compactness of specific identities and
   458	distinctiveness between identities remain similar, indicating
   459	that feature compression has only a minor impact on feature
   460	extraction in Opt JH. In contrast, for Opt JA, noticeable
   461	variation is observed along CRs. Higher CRs result in poorer
   462	feature compactness and distinctiveness. For example, the blue
   463	and black identities become more distracted as CR increases.
   464	In general, the feature compactness of Opt JH is better than
   465	that of Opt JA. Based on the analysis, we can conclude that
   466	feature compression negatively impacts feature extraction and
   467	degrades the final performance.
   468	D. Feature Compression Analysis
   469	In this subsection, we analyze feature compression by
   470	comparing the original features and the reconstructed features. For Opt JH, the original features consistently achieve

exec
/bin/zsh -lc "pdftotext 'Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf' - | nl -ba | sed -n '468,690p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   468	D. Feature Compression Analysis
   469	In this subsection, we analyze feature compression by
   470	comparing the original features and the reconstructed features. For Opt JH, the original features consistently achieve
   471	high performance, while the performance of the reconstructed
   472	features depends on the CRs. This difference arises because
   473	the original features contain substantial semantics, which are
   474	
   475	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:30 UTC from IEEE Xplore. Restrictions apply.
   476	
   477	Org_Opt_JH_16
   478	1.0
   479	0.8
   480	0.6
   481	0.4
   482	0.2
   483	0.00.0 0.2 0.4 0.6 0.8 1.0
   484	
   485	1.0
   486	0.8
   487	0.6
   488	0.4
   489	0.2
   490	0.00.0
   491	
   492	Org_Opt_JA_16
   493	1.0
   494	0.8
   495	0.6
   496	0.4
   497	0.2
   498	0.00.0 0.2 0.4 0.6 0.8 1.0
   499	
   500	1.0
   501	0.8
   502	0.6
   503	0.4
   504	0.2
   505	0.00.0
   506	
   507	1.0
   508	0.8
   509	0.6
   510	0.4
   511	0.2
   512	0.00.0
   513	
   514	Rec_Opt_JA_16
   515	
   516	0.2
   517	
   518	0.4
   519	
   520	0.6
   521	
   522	0.8
   523	
   524	1.0
   525	
   526	1.0
   527	0.8
   528	0.6
   529	0.4
   530	0.2
   531	0.00.0
   532	
   533	Org_Opt_JH_64
   534	
   535	0.2
   536	
   537	0.4
   538	
   539	0.6
   540	
   541	0.8
   542	
   543	1.0
   544	
   545	1.0
   546	
   547	Org_Opt_JA_256
   548	1.0
   549	0.8
   550	0.6
   551	0.4
   552	0.2
   553	0.00.0 0.2 0.4 0.6 0.8 1.0
   554	
   555	1.0
   556	
   557	1.0
   558	0.8
   559	0.6
   560	0.4
   561	0.2
   562	0.00.0
   563	
   564	Org_Opt_JA_64
   565	
   566	0.2
   567	
   568	0.4
   569	
   570	0.6
   571	
   572	0.8
   573	
   574	Rec_Opt_JA_64
   575	
   576	0.2
   577	
   578	0.4
   579	
   580	0.6
   581	
   582	0.8
   583	
   584	Org_Opt_JH_256
   585	
   586	1.0
   587	0.8
   588	0.6
   589	0.4
   590	0.2
   591	0.00.0
   592	
   593	0.2
   594	
   595	0.4
   596	
   597	0.6
   598	
   599	0.8
   600	
   601	1.0
   602	
   603	Rec_Opt_JA_256
   604	
   605	0.2
   606	
   607	0.4
   608	
   609	0.6
   610	
   611	0.8
   612	
   613	1.0
   614	
   615	Fig. 4. t-SNE visualization of original features and reconstructed features for Opt JA and Opt JH. First row: original features extracted from the Opt JH
   616	pipeline. Second row: original features extracted from the Opt JA pipeline. Third row: reconstructed features from the Opt JA pipeline.
   617	
   618	TABLE II
   619	P ERFORMANCE COMPARISON BETWEEN THE BASE DECODER AND THE
   620	ENHANCED DECODER
   621	
   622	Decoder
   623	
   624	Opt S
   625	
   626	Opt JH
   627	
   628	Opt JA
   629	
   630	Base
   631	Enhanced
   632	
   633	25.96
   634	25.96
   635	
   636	35.44
   637	36.10
   638	
   639	40.07
   640	41.66
   641	
   642	the CR 256, the reconstructed features fail to achieve higher
   643	performance. This issue is attributed to two reasons: first, the
   644	CR is too large to preserve sufficient semantic information,
   645	and second, the decoder may not be powerful enough to fully
   646	parse the semantics carried in the compressed features.
   647	E. Tailed Semantics Parsing Analysis
   648	
   649	partly lost during feature compression. In the case of Opt JA,
   650	the improvements brought by joint optimization vary with
   651	different CRs. It is worth mentioning that the performance
   652	is not solely determined by the amount of preserved semantic
   653	information but also by the way the semantic information is
   654	represented. For the CR 256, the reconstructed features achieve
   655	lower performance due to substantial semantic information
   656	loss in the feature compression. Conversely, for CRs smaller
   657	than 128, the semantic information is better preserved. The
   658	decoder in Opt JA acts as a semantics parser, reconstructing
   659	the features in a better representation, leading to higher
   660	performance.
   661	We visualize the reconstructed features of Opt JA in Fig. 4.
   662	For the low CRs, especially the CR 16, feature compression
   663	improves the compactness and distinctiveness of the original
   664	features. This phenomenon verifies the decoder’s ability in semantics parsing. It is important to note that the jointly trained
   665	decoder is optimized to achieve high person re-identification
   666	accuracy. This characteristic equips the whole compression
   667	network with the capability of semantics parsing. However, for
   668	
   669	In this subsection, we demonstrate that tailed semantics
   670	parsing is the key to performance improvement. Since the
   671	feature matching is fixed as Euclidean distance, we introduce
   672	extra semantics parsing capability into the decoder.
   673	We propose to enhance the semantics parsing capability of
   674	the decoder by adding two additional fully-connected layers.
   675	Since the performance of the reconstructed features is higher
   676	than that of the original features for most CRs, we conduct
   677	this experiment specifically for the CR of 256. Specifically, the
   678	modified decoder consists of three fully-connected layers, each
   679	followed by a batch normalization layer. The input and output
   680	dimensions of the three layers are (8, 64), (64, 512), and (512,
   681	2048), respectively. We jointly optimize the FastReID network
   682	and the modified feature compression network using the same
   683	training strategy.
   684	The performance comparison between the different decoders
   685	is presented in Table II. The results indicate that the enhanced
   686	decoder outperforms the base decoder in both joint optimization strategies. This finding emphasizes the significance of
   687	semantics parsing for achieving performance improvement in
   688	joint optimization strategy.
   689	
   690	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:30 UTC from IEEE Xplore. Restrictions apply.

exec
/bin/zsh -lc "pdftotext 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' - | nl -ba | sed -n '1390,1448p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' - | nl -ba | sed -n '1260,1405p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1390	4.4
  1391	
  1392	Visualization Results
  1393	
  1394	To further analyze the impact of our PKA method on the feature space, we employ t-SNE to visualize features of selected
  1395	identities. As shown in Fig. 6, two identities from each of the
  1396	five seen datasets are selected. Comparing Fig. 6(a) (baseline) with Fig. 6(b) (PKA), we observe that the PKA model
  1397	yields more compact and separable clusters. In the baseline,
  1398	clusters are scattered and overlapping, making inter-class dis-
  1399	
  1400	5857
  1401	
  1402	cuhk03_320
  1403	
  1404	cuhk_sysu_17
  1405	
  1406	duke_15
  1407	
  1408	market_22
  1409	
  1410	msmt_102
  1411	
  1412	cuhk03_460
  1413	
  1414	cuhk_sysu_32
  1415	
  1416	duke_45
  1417	
  1418	market_56
  1419	
  1420	msmt_63
  1421	
  1422	(a)
  1423	
  1424	(b)
  1425	
  1426	Figure 6: t-SNE results of our PKA compared with the Baseline.
  1427	Different colors represent different identities, highlighting the classwise aggregation and the distinction between different categories.
  1428	The visualization demonstrates how our PKA improves intra-class
  1429	cohesion and inter-class separation.
  1430	
  1431	tinction difficult. In contrast, PKA produces tighter and more
  1432	distinct clusters, indicating that it learns more discriminative
  1433	features beneficial for classification and analysis.
  1434	
  1435	5
  1436	
  1437	Conclusion
  1438	
  1439	In this paper, we tackle the challenging task of LReID by introducing PKA, a novel non-exemplar-based approach. PKA
  1440	mitigates catastrophic forgetting through Prototype-guided
  1441	Knowledge Propagation (PKP), which utilizes prototypes and
  1442	triplet loss to preserve and transfer knowledge, and Adaptive
  1443	Parameter Evolution (APE) to enable dynamic model updates
  1444	for task adaptation. Extensive experiments on five benchmark
  1445	datasets demonstrate PKA’s effectiveness, achieving notable
  1446	improvements in mean Average Precision (mAP) and rank1 accuracy, along with enhanced generalization. These results highlight the potential of PKA as a robust, privacypreserving, and practical solution for LReID.
  1447	
  1448	Acknowledgments

 succeeded in 0ms:
  1260	4. The analysis of the curves indicates that our model not
  1261	only achieves high performance but also maintains stable and
  1262	consistent performance improvement even in unseen environments. This demonstrates that our proposed Prototype-guided
  1263	Knowledge Propagation approach, compared to LReID train-
  1264	
  1265	Baseline PKP APE
  1266	
  1267	Seen-Avg
  1268	
  1269	Unseen-Avg
  1270	
  1271	mAP R@1 mAP R@1
  1272	✓
  1273	✓
  1274	
  1275	✓
  1276	✓
  1277	
  1278	42.8
  1279	53.4
  1280	54.5
  1281	55.4
  1282	
  1283	57.4
  1284	66.0
  1285	67.8
  1286	68.7
  1287	
  1288	50.7
  1289	58.7
  1290	60.9
  1291	62.3
  1292	
  1293	45.0
  1294	51.4
  1295	54.1
  1296	55.3
  1297	
  1298	accur acy(% )
  1299	
  1300	✓
  1301	✓
  1302	✓
  1303	✓
  1304	
  1305	Table 3: Ablation study of different components.
  1306	
  1307	ing methods that focus solely on classifier calibration,is more
  1308	effective in enhancing the model’s capability to extract features and shape a better embedding distribution. As a result,
  1309	it achieves more discriminative embedding representations on
  1310	unseen datasets, with improved generalization and flexibility.
  1311	
  1312	Seen mAP
  1313	Seen R@1
  1314	Unseen mAP
  1315	Unseen R@1
  1316	70
  1317	68
  1318	66
  1319	64
  1320	62
  1321	60
  1322	58
  1323	56
  1324	54
  1325	52
  1326	50
  1327	1 1.25 1.5 1.75 2
  1328	
  1329	α
  1330	
  1331	(a)The weight of Lpkp and Ltri
  1332	
  1333	accur acy(% )
  1334	
  1335	Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)
  1336	
  1337	Seen mAP
  1338	Seen R@1
  1339	Unseen mAP
  1340	Unseen R@1
  1341	70
  1342	68
  1343	66
  1344	64
  1345	62
  1346	60
  1347	58
  1348	56
  1349	54
  1350	52
  1351	50
  1352	0 0.1 0.2 0.3 0.5
  1353	
  1354	β
  1355	
  1356	(b)Noise coefficient scale
  1357	
  1358	Figure 5: Ablation studies on hyperparameters.
  1359	
  1360	4.3
  1361	
  1362	Ablation Studies
  1363	
  1364	Influence of Different Components. In this section, we conduct several ablation studies on the proposed method. To
  1365	better analyze the impact of the core designs, our approach
  1366	is divided into two components:Prototype-guided Knowledge
  1367	Propagation (PKP) and Adaptive Parameter Evolution (APE).
  1368	The baseline model refers to a framework that does not incorporate the PKP and APE modules, meaning it does not utilize
  1369	prototypes for knowledge propagation. Instead, after completing each training task, the old model and the new model
  1370	are directly integrated using a static fusion strategy. Both the
  1371	PKP and APE modules excel at propagating prior knowledge,
  1372	significantly improving performance on both seen and unseen
  1373	domains. As shown in Tab. 3, furthermore, the APE module
  1374	facilitates the transfer of prototype knowledge constructed by
  1375	the PKP module. Consequently, performance is further improved when both modules are used together.
  1376	Influence of Hyperparameters. To analyze the impact of
  1377	hyperparameters on our method, we conducted experiments
  1378	to evaluate the effects of different hyperparameter settings.
  1379	We analyzed the effects of the weights of Ltri and Lpkp .
  1380	Specifically, Ltri focuses on distinguishing the distributions
  1381	among new data, while Lpkp emphasizes the separation between the distributions of new input data and the prototypes
  1382	of old data. We set the weights of these two losses to the same
  1383	value, and based on the results shown in Fig. 5(a), we choose
  1384	α = 1.5 as the default setting. In addition, we augmented the
  1385	prototypes by adding random noise to enhance the model’s
  1386	ability to transfer old knowledge through prototypes. Appropriate noise augmentation can significantly improve performance on unseen domains, whereas excessive noise can
  1387	degrade performance on both seen and unseen domains. As
  1388	shown in Fig. 5(b) we choose β = 0.2 as the default setting.
  1389	
  1390	4.4
  1391	
  1392	Visualization Results
  1393	
  1394	To further analyze the impact of our PKA method on the feature space, we employ t-SNE to visualize features of selected
  1395	identities. As shown in Fig. 6, two identities from each of the
  1396	five seen datasets are selected. Comparing Fig. 6(a) (baseline) with Fig. 6(b) (PKA), we observe that the PKA model
  1397	yields more compact and separable clusters. In the baseline,
  1398	clusters are scattered and overlapping, making inter-class dis-
  1399	
  1400	5857
  1401	
  1402	cuhk03_320
  1403	
  1404	cuhk_sysu_17
  1405	

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdftotext 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' - | nl -ba | sed -n '540,620p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   540	60.8
   541	56.4
   542	63.2
   543	57.1
   544	63.6
   545	
   546	TABLE II
   547	T HE INFLUENCE OF EACH COMPONENT ON THE PERFORMANCE OF THE
   548	PROPOSED PDM.
   549	
   550	PLM
   551	
   552	Lch
   553	
   554	Settings
   555	Ldcs
   556	
   557	MFGM
   558	
   559	N
   560	
   561	Ldcs =
   562	
   563	M−1 M
   564	X
   565	X
   566	2
   567	[ρ2 − kcyj − cyk k2 ]+
   568	+
   569	M (M − 1) j=1
   570	
   571	(10)
   572	
   573	X
   574	X
   575	X
   576	X
   577	
   578	X
   579	X
   580	
   581	k=j+1
   582	
   583	where N denotes the batch size, Fi represents the i-th feature,
   584	yi indicates the i-th pedestrian, cyi is the centroid of yi , M
   585	is the number of centroids, ρ1 signifies the threshold distance
   586	from the sample to its centroid and ρ2 represents the distance
   587	between different centroids.
   588	C. Multi-Loss Optimization
   589	The total loss of the PLM module is as follows:
   590	Lplm = Ltri + Lch + Ldcs
   591	
   592	(11)
   593	
   594	Besides the Lcpm and Lplm , we further incorporate Lid [18]
   595	to jointly optimize the network by minimizing these three loss
   596	components:
   597	Ltotal = Lid + Lplm + Lcpm
   598	
   599	(12)
   600	
   601	IV. E XPERIMENT
   602	A. Datasets
   603	We evaluate the performance of our proposed PDM by
   604	comparing it with various state-of-the-art methods on the
   605	SYSU-MM01 [8] and LLCM [15] datasets.
   606	Metrics. In our evaluation, we focus on two pivotal metrics:
   607	Cumulative Matching Characteristics (CMC) and Mean Average Precision (mAP).
   608	
   609	VIS-to-IR
   610	Rank-1
   611	mAP
   612	48.0
   613	52.3
   614	51.5
   615	55.3
   616	59.9
   617	62.7
   618	56.5
   619	59.8
   620	60.4

hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' - | nl -ba | sed -n '777,815p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   777	D. Ablation Studies
   778	Effectiveness of each component. The ablation studies
   779	conducted on the SYSU-MM01 dataset, as presented in Table II, demonstrate the effectiveness of PLM and MFGM
   780	components individually and in combination. Including Lch
   781	and Ldcs enhances the model to achieve optimal performance.
   782	Effectiveness of different numbers of learnable prototypes for the PLM. The PLM utilizes learnable prototypes to
   783	discover semantically similar local features across modalities.
   784	Our study explores different numbers of prototypes for the
   785	PLM and finds that performance improves as the number
   786	increases from 6 to 10. However, as shown in Table III,
   787	performance starts to decline beyond 10 prototypes. Setting
   788	the number to 10 achieves the best performance on the SYSUMM01 dataset, leading us to adopt this configuration for the
   789	PLM.
   790	
   791	E. Visualization Analysis
   792	Feature Distribution. We conducted an analysis of intraclass and inter-class distance distributions for cross-modality
   793	features on the SYSU-MM01 dataset, as depicted in Fig. 2
   794	(a-d). The mean values, indicated by vertical lines, exhibit a
   795	progressive divergence (δ1 < δ2 < δ3 < δ4 ). By integrating
   796	PLM, we observed an increase in the inter-class distance
   797	and an enlargement of the gap between the average intraclass distance and inter-class distance. Furthermore, with the
   798	incorporation of MFGM, the intra-class distance decreased,
   799	leading to a further enhancement of the gap. Notably, the
   800	combination of both modules resulted in the maximum gap.
   801	To visually demonstrate the discriminative capability of the
   802	PLM, MFGM, and PDM, we conducted t-SNE visualizations
   803	(Fig. 2 (e-h)), which illustrated the clustering of embeddings
   804	per individual. These visualizations reaffirm that the PDM
   805	(Prototype Distribution Mining) approach effectively addresses
   806	intra-modal and inter-modal disparities in cross-modal person
   807	re-identification. By leveraging diverse features that exhibit
   808	close distributions and utilizing learnable prototypes to capture
   809	latent semantic similarities among cross-modal features, PDM
   810	enables a joint representation of pedestrians using multiple
   811	partial features, effectively mitigating both intra-modal and
   812	inter-modal variations. These comprehensive analyses consistently validate the efficiency of our proposed method in the
   813	context of cross-modality person re-identification.
   814	Attention Visualization. Figure 3 illustrates attention maps,
   815	showing that PDM focuses more on pedestrian regions compared to the baseline method. These analyses validate the

exec
/bin/zsh -lc "pdftotext 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' - | nl -ba | sed -n '792,815p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   792	Feature Distribution. We conducted an analysis of intraclass and inter-class distance distributions for cross-modality
   793	features on the SYSU-MM01 dataset, as depicted in Fig. 2
   794	(a-d). The mean values, indicated by vertical lines, exhibit a
   795	progressive divergence (δ1 < δ2 < δ3 < δ4 ). By integrating
   796	PLM, we observed an increase in the inter-class distance
   797	and an enlargement of the gap between the average intraclass distance and inter-class distance. Furthermore, with the
   798	incorporation of MFGM, the intra-class distance decreased,
   799	leading to a further enhancement of the gap. Notably, the
   800	combination of both modules resulted in the maximum gap.
   801	To visually demonstrate the discriminative capability of the
   802	PLM, MFGM, and PDM, we conducted t-SNE visualizations
   803	(Fig. 2 (e-h)), which illustrated the clustering of embeddings
   804	per individual. These visualizations reaffirm that the PDM
   805	(Prototype Distribution Mining) approach effectively addresses
   806	intra-modal and inter-modal disparities in cross-modal person
   807	re-identification. By leveraging diverse features that exhibit
   808	close distributions and utilizing learnable prototypes to capture
   809	latent semantic similarities among cross-modal features, PDM
   810	enables a joint representation of pedestrians using multiple
   811	partial features, effectively mitigating both intra-modal and
   812	inter-modal variations. These comprehensive analyses consistently validate the efficiency of our proposed method in the
   813	context of cross-modality person re-identification.
   814	Attention Visualization. Figure 3 illustrates attention maps,
   815	showing that PDM focuses more on pedestrian regions compared to the baseline method. These analyses validate the

exec
/bin/zsh -lc "pdftotext 'Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '933,1247p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   933	Ablation Studies
   934	
   935	We evaluate the effectiveness of each component in our proposed RSBA, with the results presented in Table 4. The Rank1 and mAP of Baseline (’0’) are 71.9% and 67.6% under the
   936	single-shot and 80.0% and 61.9% under the multi-shot.
   937	Effectiveness of ESFA. ESFA aims to introduce explicit semantics to guide the alignment of cross-modality visual features. As shown in Table 4, it improves the Rank-1 and mAP
   938	by 4.4% and 5.0% under the single-shot mode, which validates that aligning visual features into the semantic space is
   939	reasonable and effective for mitigating the modality gap.
   940	Effectiveness of CSIA. CSIA constrains the consistent similarity between intra-modality image-text pairs and intermodality text-text pairs to establish the correspondence between images and cross-modality texts. As detailed in Table
   941	4, under the single-shot test mode, it improves the Rank-1 accuracy from 76.3% to 77.4%, which indicates that the align-
   942	
   943	931
   944	
   945	Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)
   946	
   947	Methods
   948	
   949	Ref
   950	
   951	cmGAN [Dai et al., 2018]
   952	Hi-CMD [Choi et al., 2020]
   953	JSIA [Wang et al., 2020]
   954	MSA [Miao et al., 2021]
   955	TSME [Liu et al., 2022b]
   956	ACD [Pan et al., 2024]
   957	AGW [Ye et al., 2021b]
   958	MCSL [Ling et al., 2021]
   959	CAJ [Ye et al., 2021a]
   960	MMN [Zhang et al., 2021]
   961	MAUM [Liu et al., 2022a]
   962	CIFT [Li et al., 2022]
   963	MBCE [Cheng et al., 2023]
   964	DEEN [Zhang and Wang, 2023]
   965	SEFL [Feng et al., 2023]
   966	HOS-Net [Qiu et al., 2024]
   967	CSCL [Liu et al., 2025]
   968	CycleTans [Wu et al., 2025]
   969	CSDN [Yu et al., 2025]
   970	Ours (RSBA)
   971	
   972	IJCAI’18
   973	CVPR’20
   974	AAAI’20
   975	IJCAI’21
   976	TCSVT’22
   977	TIFS’24
   978	TPAMI’21
   979	IJCAI’21
   980	ICCV’21
   981	MM’21
   982	CVPR’22
   983	ECCV’22
   984	AAAI’23
   985	CVPR’23
   986	CVPR’23
   987	AAAI’24
   988	TMM’24
   989	TNNLS’24
   990	TMM’25
   991	IJCAI’25
   992	
   993	All-Search
   994	Single-Shot
   995	Multi-Shot
   996	R1
   997	mAP
   998	R1
   999	mAP
  1000	26.9 27.8 31.4 22.2
  1001	34.9 35.9
  1002	38.1 36.9 45.1 29.5
  1003	63.1 59.2
  1004	64.2 61.2 70.3 54.3
  1005	74.4 71.1 80.4 66.9
  1006	47.5 47.6
  1007	64.8 60.8 68.0 51.4
  1008	69.8 66.8
  1009	70.6 66.9
  1010	71.6 68.7
  1011	71.7 67.6 78.0 62.4
  1012	74.7 72.0 78.3 65.7
  1013	74.7 71.8
  1014	75.1 70.1
  1015	75.6 74.2
  1016	75.7 72.0
  1017	76.5 72.6 82.8 68.5
  1018	76.7 73.0 83.5 67.9
  1019	78.4 74.8 84.6 70.0
  1020	
  1021	Indoor-Search
  1022	Single-Shot
  1023	Multi-Shot
  1024	R1
  1025	mAP
  1026	R1
  1027	mAP
  1028	31.6 42.1 37.0 32.7
  1029	43.8 52.9 52.7 42.7
  1030	67.1 72.7
  1031	64.8 71.5 76.8 65.0
  1032	78.9 82.7 86.0 78.6
  1033	54.1 62.9
  1034	76.2 76.7
  1035	76.2 79.6
  1036	76.9 81.9
  1037	78.6 82.1 86.9 77.0
  1038	83.4 86.0 88.4 80.6
  1039	80.3 83.3
  1040	78.4 81.2
  1041	84.2 86.7
  1042	80.8 83.5
  1043	87.2 84.9 91.2 81.4
  1044	84.5 86.8 91.3 82.2
  1045	87.7 87.8 92.3 83.8
  1046	
  1047	Table 1: Performance comparison with state-of-the-art methods on SYSU-MM01. ’-’ denotes that no reported result is available.
  1048	
  1049	Methods
  1050	Hi-CMD
  1051	JSIA
  1052	MSA
  1053	TSME
  1054	ACD
  1055	AGW
  1056	MCSL
  1057	CAJ
  1058	MMN
  1059	MAUM
  1060	CIFT
  1061	MBCE
  1062	DEEN
  1063	SEFL
  1064	HOS-Net
  1065	CSCL
  1066	CycleTrans
  1067	CSDN
  1068	Ours (RSBA)
  1069	
  1070	Visible to Infrared
  1071	R1
  1072	mAP
  1073	70.9
  1074	66.0
  1075	48.1
  1076	48.9
  1077	84.8
  1078	82.1
  1079	87.3
  1080	76.9
  1081	84.7
  1082	83.2
  1083	70.0
  1084	66.4
  1085	93.8
  1086	87.5
  1087	85.0
  1088	65.3
  1089	91.6
  1090	84.1
  1091	87.8
  1092	85.0
  1093	92.1
  1094	86.9
  1095	93.1
  1096	88.3
  1097	91.1
  1098	85.1
  1099	91.0
  1100	85.2
  1101	94.7
  1102	90.4
  1103	92.1
  1104	84.2
  1105	90.6
  1106	85.6
  1107	95.4
  1108	87.7
  1109	95.2
  1110	90.8
  1111	
  1112	Infrared to Visible
  1113	R1
  1114	mAP
  1115	48.5
  1116	49.3
  1117	86.4
  1118	75.7
  1119	87.1
  1120	84.7
  1121	91.5
  1122	85.2
  1123	84.7
  1124	61.5
  1125	87.5
  1126	80.5
  1127	86.9
  1128	84.3
  1129	90.1
  1130	84.8
  1131	93.4
  1132	87.9
  1133	89.5
  1134	83.4
  1135	92.1
  1136	86.5
  1137	93.3
  1138	89.2
  1139	89.6
  1140	85.0
  1141	81.8
  1142	87.0
  1143	92.3
  1144	85.5
  1145	94.4
  1146	89.5
  1147	
  1148	Methods
  1149	AGW
  1150	CAJ
  1151	MMN
  1152	DEEN
  1153	HOS-Net
  1154	Ours (RSBA)
  1155	
  1156	Visible to Infrared
  1157	R1
  1158	mAP
  1159	51.5
  1160	55.3
  1161	56.5
  1162	59.8
  1163	59.9
  1164	62.7
  1165	62.5
  1166	65.8
  1167	64.9
  1168	67.9
  1169	66.0
  1170	68.8
  1171	
  1172	Infrared to Visible
  1173	R1
  1174	mAP
  1175	43.6
  1176	51.8
  1177	48.8
  1178	56.6
  1179	52.5
  1180	58.9
  1181	54.9
  1182	62.9
  1183	56.4
  1184	63.2
  1185	57.3
  1186	64.2
  1187	
  1188	Table 3: Performance comparison on LLCM.
  1189	ESFA
  1190	0
  1191	1
  1192	2
  1193	3
  1194	4
  1195	
  1196	✓
  1197	✓
  1198	✓
  1199	✓
  1200	
  1201	CSIA
  1202	
  1203	CVSC
  1204	
  1205	✓
  1206	✓
  1207	
  1208	✓
  1209	✓
  1210	
  1211	Single-Shot
  1212	R1
  1213	mAP
  1214	71.9 67.6
  1215	76.3 72.6
  1216	77.4 73.2
  1217	77.6 73.7
  1218	78.4 74.8
  1219	
  1220	Multi-Shot
  1221	R1
  1222	mAP
  1223	80.0 61.9
  1224	82.1 66.5
  1225	82.7 68.2
  1226	83.1 68.8
  1227	84.6 70.0
  1228	
  1229	Table 4: Ablation studies of our RSBA.
  1230	
  1231	Table 2: Performance comparison on RegDB.
  1232	
  1233	and CSIA, the recognition performance reaches a peak. These
  1234	results fully demonstrate the reasonableness of motivation behind CVSC and the effectiveness of its technology.
  1235	
  1236	ment of inter-modality image-text pairs plays a positive role
  1237	in the further effective alignment of visual features.
  1238	Effectiveness of CVSC. CVSC integrates multi-view texts to
  1239	capture comprehensive semantics that are beneficial for improving the alignment in ESFA and CSIA. As illustrated in
  1240	Table 4, when it is equipped with ESFA, the Rank-1 accuracy
  1241	is improved by 1.3% and 1.0% under the two test modes, respectively. In addition, when incorporating it with both ESFA
  1242	
  1243	4.5
  1244	
  1245	Parameters Analysis
  1246	
  1247	We introduce the hyper-parameters λ1 and λ2 to regulate the

codex
PDM 的表格在普通文本抽取里排版有些乱，但它的文字说明和图分析已经足够支撑方法论判断：组件消融证明 PLM、MFGM、两个约束项都贡献；距离分布和 t-SNE 证明它真的扩大跨身份间隔、缩小同身份跨模态间隔。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -layout -f 1 -l 8 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' - | sed -n '270,350p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
A. Datasets                                                       C. Main Results
  We evaluate the performance of our proposed PDM by                 As shown in Table I, PDM outperforms competing methods
comparing it with various state-of-the-art methods on the         in cross-modality person re-identification tasks. On the SYSU-
SYSU-MM01 [8] and LLCM [15] datasets.                             MM01 dataset, it achieves a rank-1 accuracy of 79.3% and
Metrics. In our evaluation, we focus on two pivotal metrics:      mAP of 76.2% in the All-search mode, and 88.7% rank-1
Cumulative Matching Characteristics (CMC) and Mean Aver-          accuracy and 89.8% mAP in the Indoor-search mode. On the
age Precision (mAP).                                              LLCM dataset, PDM achieves a rank-1 accuracy of 57.1%
             (a) Initial Distance                        (b) PLM Distance                          (c) MFGM Distance                        (d) PDM Distance




           (e) Initial Distribution                     (f) PLM Distribution                     (g) MFGM Distribution                    (h) PDM Distribution


Fig. 2. (a-d) illustrate the intra-class and inter-class distances of cross-modality features, with intra-class and inter-class distances represented in blue and green,
respectively. In (e-h), the t-SNE [24] visualizations illustrate the 2D feature distributions, where circles and triangles denote infrared and visible modalities,
and different colors represent pedestrians from distinct categories.



                                                                                                                 TABLE III
                                                                                      T HE INFLUENCE OF DIFFERENT QUANTITIES OF LEARNABLE PROTOTYPES
                                                                                                 ON THE PERFORMANCE OF THE PROPOSED PDM.

(a)
                                                                                                                    All-search                  Indoor-search
                                                                                          Settings
                                                                                                             Rank-1          mAP            Rank-1               mAP
                                                                                           m=6                 78.4              75.2         86.5               88.3
                                                                                           m=8                 78.6              75.6         85.8               87.8
                                                                                          m = 10               79.3              76.3         88.7               89.8
(b)
                                                                                          m = 12               78.1              75.8         85.2               87.1

                                                                                     to-IR mode. This underscores PDM’s superior performance
                                                                                     and effectiveness in handling modality disparities.

(c)                                                                                  D. Ablation Studies
                                                                                        Effectiveness of each component. The ablation studies
                                                                                     conducted on the SYSU-MM01 dataset, as presented in Ta-
                                                                                     ble II, demonstrate the effectiveness of PLM and MFGM
Fig. 3. The visualization results of attention maps. (a) represents the displayed    components individually and in combination. Including Lch
image, (b) and (c) show the results of baseline and PDM.
                                                                                     and Ldcs enhances the model to achieve optimal performance.
and mAP of 63.6% in the IR-to-VIS mode, and 64.9%                                       Effectiveness of different numbers of learnable proto-
rank-1 accuracy and 67.3% mAP in the VIS-to-IR mode.                                 types for the PLM. The PLM utilizes learnable prototypes to
These results demonstrate PDM’s effectiveness in addressing                          discover semantically similar local features across modalities.
modality disparities and its exceptional performance in cross-                       Our study explores different numbers of prototypes for the
modality person re-identification tasks. Additionally, on the                        PLM and finds that performance improves as the number
SYSU-MM01 dataset, PDM surpasses HOS-Net with a 3.7%                                 increases from 6 to 10. However, as shown in Table III,
higher rank-1 accuracy and 2.1% higher mAP. In the LLCM                              performance starts to decline beyond 10 prototypes. Setting
dataset, PDM outperforms HOS-Net by 0.7% in the IR-to-VIS                            the number to 10 achieves the best performance on the SYSU-
mode and exhibits a slightly lower mAP by 0.6% in the VIS-                           MM01 dataset, leading us to adopt this configuration for the
                                                                                     PLM.
E. Visualization Analysis                                                                          R EFERENCES
   Feature Distribution. We conducted an analysis of intra-          [1] N. Huang, J. Liu, Y. Miao, Q. Zhang, and J. Han, “Deep learning for
class and inter-class distance distributions for cross-modality          visible-infrared cross-modality person re-identification: A comprehen-
                                                                         sive review,” Information Fusion, vol. 91, pp. 396–411, 2023. I
features on the SYSU-MM01 dataset, as depicted in Fig. 2             [2] H. Liu, X. Tan, and X. Zhou, “Parameter sharing exploration and hetero-
(a-d). The mean values, indicated by vertical lines, exhibit a           center triplet loss for visible-thermal person re-identification,” IEEE
progressive divergence (δ1 < δ2 < δ3 < δ4 ). By integrating              Transactions on Multimedia, vol. 23, pp. 4414–4425, 2020. I, II
                                                                     [3] M. Qi, S. Chan, C. Hang, G. Zhang, and Z. Li, “Fine-grained learning
PLM, we observed an increase in the inter-class distance                 for visible-infrared person re-identification,” in 2023 IEEE International
and an enlargement of the gap between the average intra-                 Conference on Multimedia and Expo (ICME). IEEE, 2023, pp. 2417–
class distance and inter-class distance. Furthermore, with the           2422. I, II
                                                                     [4] C. Fu, Y. Hu, X. Wu, H. Shi, T. Mei, and R. He, “Cm-nas: Cross-
incorporation of MFGM, the intra-class distance decreased,               modality neural architecture search for visible-infrared person re-
leading to a further enhancement of the gap. Notably, the                identification,” in Proceedings of the IEEE/CVF International Confer-
combination of both modules resulted in the maximum gap.                 ence on Computer Vision, 2021, pp. 11 823–11 832. I
                                                                     [5] G.-A. Wang, T. Zhang, Y. Yang, J. Cheng, J. Chang, X. Liang, and Z.-G.
To visually demonstrate the discriminative capability of the             Hou, “Cross-modality paired-images generation for rgb-infrared person
PLM, MFGM, and PDM, we conducted t-SNE visualizations                    re-identification,” in Proceedings of the AAAI conference on artificial
(Fig. 2 (e-h)), which illustrated the clustering of embeddings           intelligence, vol. 34, no. 07, 2020, pp. 12 144–12 151. I
                                                                     [6] G. Wang, T. Zhang, J. Cheng, S. Liu, Y. Yang, and Z. Hou, “Rgb-infrared
per individual. These visualizations reaffirm that the PDM               cross-modality person re-identification via joint pixel and feature align-
(Prototype Distribution Mining) approach effectively addresses           ment,” in Proceedings of the IEEE/CVF International Conference on
intra-modal and inter-modal disparities in cross-modal person            Computer Vision, 2019, pp. 3623–3632. I, II, I
                                                                     [7] X. Zhang, H. Luo, X. Fan, W. Xiang, Y. Sun, Q. Xiao, W. Jiang,
re-identification. By leveraging diverse features that exhibit           C. Zhang, and J. Sun, “Alignedreid: Surpassing human-level perfor-
close distributions and utilizing learnable prototypes to capture        mance in person re-identification,” 2018. I

exec
/bin/zsh -lc "pdftotext -layout 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' - | sed -n '650,710p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  pervised domain adaptive person re-identification with co-                     tions on Pattern Analysis and Machine Intelligence, 2023.
  ordinated anti-forgetting and adaptation. In Proceedings                    [Ristani et al., 2016] Ergys Ristani, Francesco Solera, Roger
  of the IEEE/CVF Conference on Computer Vision and Pat-                         Zou, Rita Cucchiara, and Carlo Tomasi. Performance mea-
  tern Recognition, pages 14288–14297, 2022.                                     sures and a data set for multi-target, multi-camera track-
[Huang et al., 2023] Jinze Huang, Xiaohan Yu, Dong An,                           ing. In European conference on computer vision, pages
  Yaoguang Wei, Xiao Bai, Jin Zheng, Chen Wang, and Jun                          17–35. Springer, 2016.



                                                                       5858
                        Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)



[Schroff et al., 2015] Florian Schroff, Dmitry Kalenichenko,                  [Ye et al., 2018] Mang Ye, Zheng Wang, Xiangyuan Lan,
   and James Philbin. Facenet: A unified embedding for face                     and Pong C Yuen. Visible thermal person re-identification
   recognition and clustering. In Proceedings of the IEEE                       via dual-constrained top-ranking. In IJCAI, volume 1,
   conference on computer vision and pattern recognition,                       page 2, 2018.
   pages 815–823, 2015.                                                       [Ye et al., 2021] Mang Ye, Jianbing Shen, Gaojie Lin, Tao
[Shi and Ye, 2023] Wuxuan Shi and Mang Ye. Prototype                            Xiang, Ling Shao, and Steven CH Hoi. Deep learning
   reminiscence and augmented asymmetric knowledge ag-                          for person re-identification: A survey and outlook. IEEE
   gregation for non-exemplar class-incremental learning. In                    TPAMI, 44(6):2872–2893, 2021.
   Proceedings of the IEEE/CVF International Conference                       [Ye et al., 2023] Mang Ye, Zesen Wu, Cuiqun Chen, and
   on Computer Vision, pages 1772–1781, 2023.                                   Bo Du. Channel augmentation for visible-infrared re-
[Shi and Ye, 2024] Wuxuan Shi and Mang Ye. Prospec-                             identification. IEEE Transactions on Pattern Analysis and
   tive representation learning for non-exemplar class-                         Machine Intelligence, 46(4):2299–2315, 2023.
   incremental learning. In NeurIPS, 2024.                                    [Ye et al., 2024] Mang Ye, Shuoyi Chen, Chenyue Li, Wei-
[Sun and Mu, 2022] Zhicheng Sun and Yadong Mu. Patch-                           Shi Zheng, David Crandall, and Bo Du. Transformer for
   based knowledge distillation for lifelong person re-                         object re-identification: A survey, 2024.
   identification. In Proceedings of the 30th ACM Interna-                    [Yu et al., 2023a] Chunlin Yu, Ye Shi, Zimo Liu, Shenghua
   tional Conference on Multimedia, pages 696–707, 2022.                        Gao, and Jingya Wang. Lifelong person re-identification
[Tung and Mori, 2019] Frederick Tung and Greg Mori.                             via knowledge refreshing and consolidation. In Proceed-
                                                                                ings of the AAAI Conference on Artificial Intelligence, vol-
   Similarity-preserving knowledge distillation. In Proceed-
                                                                                ume 37, pages 3295–3303, 2023.
   ings of the IEEE/CVF international conference on com-
   puter vision, pages 1365–1374, 2019.                                       [Yu et al., 2023b] Chunlin Yu, Ye Shi, Zimo Liu, Shenghua
                                                                                Gao, and Jingya Wang. Lifelong person re-identification
[Wei et al., 2018] Longhui Wei, Shiliang Zhang, Wen Gao,                        via knowledge refreshing and consolidation. In Proceed-
  and Qi Tian. Person transfer gan to bridge domain gap for                     ings of the AAAI Conference on Artificial Intelligence, vol-
  person re-identification. In Proceedings of the IEEE con-                     ume 37, pages 3295–3303, 2023.
  ference on computer vision and pattern recognition, pages
  79–88, 2018.                                                                [Zhang et al., 2016] Yaqing Zhang, Xi Li, Liming Zhao, and
                                                                                Zhongfei Zhang. Semantics-aware deep correspondence
[Wu and Gong, 2021] Guile Wu and Shaogang Gong. Gen-                            structure learning for robust person re-identification. In
  eralising without forgetting for lifelong person re-                          IJCAI, pages 3545–3551, 2016.
  identification. In Proceedings of the AAAI conference on
                                                                              [Zhang et al., 2024] Wenxuan Zhang, Paul Janson, Rahaf
  artificial intelligence, volume 35, pages 2889–2897, 2021.
                                                                                Aljundi, and Mohamed Elhoseiny. Overcoming generic
[Wu et al., 2025] Bin Wu, Wuxuan Shi, Jinqiao Wang,                             knowledge loss with selective parameter update. In Pro-
  and Mang Ye.        Synthetic data is an elegant gift                         ceedings of the IEEE/CVF Conference on Computer Vision
  for continual vision-language models. arXiv preprint                          and Pattern Recognition, pages 24046–24056, 2024.
  arXiv:2503.04229, 2025.                                                     [Zhao et al., 2017] Haiyu Zhao, Maoqing Tian, Shuyang
[Xiao et al., 2017] Tong Xiao, Shuang Li, Bochao Wang,                          Sun, Jing Shao, Junjie Yan, Shuai Yi, Xiaogang Wang, and
  Liang Lin, and Xiaogang Wang. Joint detection and identi-                     Xiaoou Tang. Spindle net: Person re-identification with
  fication feature learning for person search. In Proceedings                   human body region guided feature decomposition and fu-
  of the IEEE conference on computer vision and pattern                         sion. In Proceedings of the IEEE conference on computer
  recognition, pages 3415–3424, 2017.                                           vision and pattern recognition, pages 1077–1085, 2017.
[Xiao et al., 2023] Jia-Wen Xiao, Chang-Bin Zhang, Jiekang                    [Zheng et al., 2015] Liang Zheng, Liyue Shen, Lu Tian,
  Feng, Xialei Liu, Joost van de Weijer, and Ming-Ming                          Shengjin Wang, Jingdong Wang, and Qi Tian. Scalable

exec
/bin/zsh -lc "pdftotext -layout 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' - | sed -n '760,840p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
                                                                                 to focus on coarse and incomplete regions of the pedestrian,
for TBPR than relying on external large-scale models for data
                                                                                 often ignoring fine-grained visual cues such as clothing tex-
enrichment. It highlights PDA’s ability to capture the essence
                                                                                 ture, color patterns, and accessories. In contrast, PDA produces
of cross-modal alignment at the representation level.
                                                                                 more concentrated and semantically aligned attention maps.
                                                                                 The highlighted areas correspond more accurately to the key
H. Qualitative Results                                                           textual descriptions, such as “blue and black striped shirt”,
  1) Top-10 Retrieval Results: To intuitively demonstrate                        “white umbrella”, and “pink ruffled dress”. This indicates
the effectiveness of our method, Fig. 5 illustrates the                          that PDA is capable of capturing fine-grained visual–textual

          Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:10:29 UTC from IEEE Xplore. Restrictions apply.
YANG et al.: PROBABILISTIC DISTRIBUTION ALIGNMENT FOR TEXT-BASED PERSON RETRIEVAL                                                                         7585



                                                                                of ID 3955 and ID 3940 are positioned close to each other,
                                                                                suggesting that PDA captures semantic proximity between
                                                                                visually and textually similar identities. This observation con-
                                                                                firms that the probabilistic embeddings not only enhance
                                                                                matching accuracy, but also reflect finer-grained cross-modal
                                                                                relationships, which point-to-point embeddings often fail to
                                                                                capture. In summary, the visualization results validate the
                                                                                benefits of probabilistic modeling in PDA. The model demon-
                                                                                strates enhanced semantic alignment, better generalization
                                                                                across queries, and robustness to modality-specific uncer-
                                                                                tainty ultimately enabling more comprehensive and reliable
                                                                                cross-modal retrieval.

Fig. 6. Visualization of attention heatmaps, where redder regions indicate                                  V. C ONCLUSION
higher response intensity.
                                                                                   This paper proposes a Probabilistic Distribution
                                                                                Alignment (PDA) network to address the inherent variability
                                                                                and ambiguity in TBPR. Unlike traditional deterministic
                                                                                embeddings, PDA models images and text as Gaussian
                                                                                distributions, capturing uncertainty and enabling probabilistic
                                                                                alignment. We introduce Cross-Modal Containment (CMC)
                                                                                to explore the broader semantic scope of text compared
                                                                                to image, ensuring that text distributions encompass paired
                                                                                image distributions. Additionally, Intra-Modal Containment
                                                                                (IMC) preserves semantic coherence within each modality.
                                                                                Extensive experiments demonstrate that PDA outperforms
                                                                                state-of-the-art methods, particularly in distinguishing
                                                                                challenging samples and handling complex relationships.
                                                                                Beyond accuracy gains, our results suggest that probabilistic
                                                                                modeling and containment-driven matching provide a robust
                                                                                and principled framework for TBPR. While PDA introduces
                                                                                a moderate computational overhead, it achieves a favorable
                                                                                accuracy. Future work will further explore lightweight
                                                                                designs and distillation-based strategies to reduce inference
Fig. 7. Visualization of distribution representations in our PDA. We display    cost, facilitating broader real-world deployment. Overall,
visual and textual representations of the top-10 retrieved results, with each
ellipse indicating a 50% confidence region for visual data. The white ellipse   our findings highlight probabilistic containment learning as
represents the textual query with a 90% confidence region.                      a promising paradigm for improving both robustness and
                                                                                reliability in TBPR.

correspondences. Moreover, PDA effectively suppresses irrele-                                                 R EFERENCES
vant background regions and confusing negatives, demonstrat-
                                                                                [1]   S. Li, T. Xiao, H. Li, B. Zhou, D. Yue, and X. Wang, “Person search
ing its superior discriminative ability in TBPR.                                      with natural language description,” in Proc. IEEE Conf. Comput. Vis.
   3) Visualization of Distribution Representations: To better                        Pattern Recognit. (CVPR), Jul. 2017, pp. 1970–1979.
illustrate the effectiveness of our probabilistic modeling, we                  [2]   Y. Sun, L. Zheng, Y. Yang, Q. Tian, and S. Wang, “Beyond part models:
                                                                                      Person retrieval with refined part pooling (and a strong convolutional
conduct a 2D visualization of the distributed embeddings for                          baseline),” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 480–496.
images and their corresponding textual queries. As shown                        [3]   S. He, H. Luo, P. Wang, F. Wang, H. Li, and W. Jiang, “TransReID:
in Fig. 7, each white ellipse denotes the distribution of a                           Transformer-based object re-identification,” in Proc. IEEE/CVF Int.
                                                                                      Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 15013–15022.
textual query embedding, while the colored points represent                     [4]   S. Li, S. Li, and Q. Li, “CLIP-ReID: Exploiting vision-language model
the top-10 retrieved image embeddings from the gallery.                               for image re-identification without concrete text labels,” in Proc. AAAI,
The broader spatial coverage of the textual ellipses reflects                         2023, vol. 37, no. 1, pp. 1405–1413.
                                                                                [5]   D. Jiang and M. Ye, “Cross-modal implicit relation reasoning and
the semantic generalization achieved through the Cross-                               aligning for text-to-image person retrieval,” in Proc. IEEE/CVF Conf.
Modal Containment (CMC) module, which models the textual                              Comput. Vis. Pattern Recognit. (CVPR), Jun. 2023, pp. 2787–2797.
representation as a probabilistic distribution rather than a                    [6]   T. Fujii and S. Tarashima, “BiLMa: Bidirectional local-matching for text-
                                                                                      based person re-identification,” in Proc. IEEE/CVF Int. Conf. Comput.
deterministic point. This design enables the text distribution                        Vis. Workshops (ICCVW), Oct. 2023, pp. 2778–2782.
to effectively encompass the true matching image embeddings                     [7]   D. Wang et al., “Fine-grained semantics-aware representation learning
(e.g., ID 3955), while excluding semantically irrelevant or neg-                      for text-based person retrieval,” in Proc. Int. Conf. Multimedia Retr.,
                                                                                      May 2024, pp. 92–100.
ative samples, indicating strong discrimination. Furthermore,                   [8]   Y. Wang, M. Yang, and R. Cao, “Fine-grained semantic alignment with

exec
/bin/zsh -lc "pdftotext -layout 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' - | sed -n '1030,1095p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 124ms:
codex
下面按“创新怎么被构造出来”拆，不按模块复述。

**1. Probabilistic Distribution Alignment for Text-Based Person Retrieval**

1. **触发观察**：作者抓住了文本行人检索里的非对称性，文本通常是粗粒度描述，图像是具体实例。一个文本本来对应一组可能图像，而不是一个点对一个点。现有点嵌入把强正样本、弱正样本、相似负样本都挤在同一套距离逻辑里，容易把“语义范围”误当成“特征距离”。

2. **重定义动作**：它把问题从“图文相似度对齐”改写成“文本分布应当包含图像分布”。关键词是 probabilistic containment、semantic scope、partial-order relationship、text as probabilistic envelope。厉害的地方是把方差从“噪声余量”重新定义成“语义覆盖范围”。

3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。

4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。

5. **reviewer 为什么买账**：这篇卖的是视角，不只是机制。它不是又加一个对齐头，而是把 TBPR 的匹配关系从点对点改成非对称包含。novelty 来自“方差等于语义范围”这个解释，以及用闭式包含损失把这个解释落成可训练目标。

**2. Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification**

1. **触发观察**：VI-ReID 里常见做法是水平切分身体做局部对齐，但行人姿态、视角和运动不规则，水平条带经常语义错位。生成式跨模态图像又容易引入噪声，所以作者转向特征层。

2. **重定义动作**：它把“局部对齐”从固定人体条带改成“潜在语义相似局部的挖掘”。关键词是 prototype-driven、latent semantically similar features、diverse yet closely distributed features、instance-level alignment。旧问题是模态差异，新说法是缺少足够多样但分布接近的共享特征，以及缺少跨模态潜在语义局部。

3. **机制怎么长出来**：MFGM 用多分支膨胀卷积、注意力和中心引导 pair mining 生成多样但靠近共享分布的特征；PLM 用可学习原型给像素级特征加权，挖出跨 VIS/IR 语义相似的局部；Lch 保证原型多样，Ldcs 让类内靠近、类间中心分离。机制和重定义有联系，但比 PDA 弱一些，更像把“潜在局部”问题工程化。

4. **证据闭环**：组件消融显示 PLM、MFGM、Lch、Ldcs 都有贡献，原型数量从 6 到 10 性能提升，超过 10 又下降，说明原型不是越多越好。距离分布图给了更关键的证据：PLM 拉大类间距离，MFGM 缩小类内距离，两者结合间隔最大。t-SNE 和注意力图进一步说明模型确实更集中在行人区域和更清晰的身份簇上。

5. **reviewer 为什么买账**：这篇主要卖机制，不是强视角。novelty 来自把“水平部位不可靠”替换成“可学习原型挖潜在语义局部”，再用多特征生成解决模态分布不够的问题。它可学的一点是：先否定一个常用结构先验，再给出一个数据驱动替代物。

**3. Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification**

1. **触发观察**：LReID 要持续学习新域又不能忘旧域，存旧图像有隐私和存储风险。已有原型方法多来自类增量分类，主要调分类器，但 ReID 的核心是检索表征，旧身份原型和新特征提取器之间会产生表征偏差。静态融合旧模型和新模型也忽略不同任务的差异。

2. **重定义动作**：它把“灾难性遗忘”重写成“旧身份原型、新任务特征、更新后特征提取器之间的兼容问题”。关键词是 non-exemplar、prototype-guided knowledge propagation、representation bias、adaptive parameter evolution、compatibility。旧问题是保留旧知识，新说法是让旧知识在新表征空间里继续能区分。

3. **机制怎么长出来**：PKP 保存旧身份原型而不是旧图像，训练新任务时抽样旧原型并加高斯噪声，用改造 triplet 把新任务特征和旧原型云推开，同时用普通 triplet 保持新身份区分。APE 只更新对当前任务梯度影响大的参数，再按任务数据规模动态融合新旧模型。这个机制和重定义绑定较紧，因为它直接处理“旧原型和新提取器不兼容”。

4. **证据闭环**：组件消融很对题。baseline 的 Seen-Avg 是 42.8 mAP/57.4 Rank-1，加入 PKP 到 53.4/66.0，加入 APE 到 54.5/67.8，两者一起到 55.4/68.7；Unseen-Avg 也从 50.7/45.0 到 62.3/55.3。t-SNE 显示 PKA 后类内更紧、类间更分开，证明它不是只调分类器，而是在改善检索表征空间。

5. **reviewer 为什么买账**：这篇卖的是“非样本终身 ReID 的问题重定义”加一个够实用的机制。原型、triplet、参数融合都不是孤立新零件，但组合起来服务于一个清楚约束：不存图像，还要让旧身份原型继续在新表征里有效。

**4. Rethinking Joint Optimization in Feature Compression: Insights from Person Re-Identification**

1. **触发观察**：机器视觉压缩领域普遍认为联合优化更好，但很多联合优化管线末端都有语义解析网络。作者怀疑性能提升到底来自联合优化，还是来自尾部语义解析能力。

2. **重定义动作**：它把问题从“怎么做更好的联合压缩”改成“联合优化的收益来源是什么”。关键词是 disentangle、tailed semantics parsing、information capacity constraint、separate optimization、joint optimization。ReID 被选作工具，是因为它的尾部匹配可以只是欧氏距离，不需要可学习语义解析器。

3. **机制怎么长出来**：机制其实是实验设计。Opt S 分别训练 ReID 和压缩器；Opt JA 联合优化全部模块；Opt JH 冻结 backbone，只联合优化聚合层和压缩器。再用不同压缩率观察什么时候联合优化有效，什么时候反而伤害特征提取。

4. **证据闭环**：关键证据很强。CR=256 时联合优化更好，Opt S 是 26.54，Opt JA 是 40.29；但 CR 小于 128 后反过来，CR=16 时 Opt S 是 80.50，Opt JH 是 76.37，Opt JA 是 72.41。更关键的是原始特征性能，Opt JH_O 稳在 80.8 左右，而 Opt JA_O 只有 54 到 62，证明联合压缩约束伤了特征提取。增强 decoder 后 Opt JA 从 40.07 到 41.66，说明尾部语义解析确实是收益来源。

5. **reviewer 为什么买账**：这篇卖的是反常识视角和干净证据，不是新模块。它用 ReID 构造了一个能隔离混杂因素的实验场景，证明“联合优化好”这个共识并不总成立。方法论价值是：如果能拆掉领域里的混淆变量，负结论也能变成贡献。

**5. Richer Semantics, Better Alignment**

1. **触发观察**：VI-ReID 的可见光和红外图像有大模态差异，视觉特征很难直接对齐。已有 CLIP prompt 方法说明语义空间有帮助，但可学习 prompt 语义不透明、太粗、还要两阶段训练。LLaVA 能生成显式细粒度描述，但跨模态描述会有冲突，比如同一衣服在可见光里是蓝色，在红外里像灰色；同一身份不同视角的描述又互补。

2. **重定义动作**：它把 VI-ReID 从“视觉特征对齐到公共空间”改成“视觉特征对齐到显式且丰富的语义空间”。关键词是 explicit semantics、enriched semantics、indirect alignment、cross-view semantics compensation、one-to-many alignment。它不是简单说“用文本监督”，而是拆成显式语义、跨模态语义冲突、多视角语义不完整三个子问题。

3. **机制怎么长出来**：ESFA 用 LLaVA 给可见光和红外图像生成描述，在各自模态内做图文对齐。CSIA 不直接强拉图像和异模态文本，而是约束“模态内图文相似度”和“跨模态文本相似度”一致，绕开颜色冲突。CVSC 用同身份其他视角文本补偿当前文本，把一对一图文匹配扩成一对多语义对齐。机制和重定义绑定非常紧。

4. **证据闭环**：消融正好对应三个子问题。baseline 单次检索是 71.9 Rank-1/67.6 mAP，ESFA 到 76.3/72.6，加入 CSIA 到 77.4/73.2，ESFA+CVSC 到 77.6/73.7，完整模型到 78.4/74.8。进一步讨论里，显式语义强于隐式 prompt，直接跨模态文本对齐会让 ESFA 变差，而间接对齐会提升，这正好证明了“语义冲突必须间接处理”。

5. **reviewer 为什么买账**：这篇卖的是视角和机制都成立。真正 novelty 不是“用了 LLaVA”，而是把外部语义引入 VI-ReID 后可能遇到的三个障碍逐个定义清楚，并且每个障碍都有对应机制和消融。

**6. RMGNet**

1. **触发观察**：TI-ReID 里很多方法已经做局部图文对齐，但作者指出只看局部对象还不够。两个人都可能有灰外套、黑包，差别在“背着包”还是“穿着包”这样的局部关系。另一个观察是 Hc-Tri loss 很早收敛到 0，后期没有有效监督。

2. **重定义动作**：它把“局部特征对齐”改写成“局部特征之间关系的建模和挖掘”。关键词是 relationship-based matching、progressive relationship-mining graph、intra-modal relationship、cross-modal relationship、continuous effective metric supervision。旧问题是找局部，新问题是理解局部之间的语义关系。

3. **机制怎么长出来**：IMRM 在图像局部和文本局部内部建图，用 GNN 聚合关系信息。TCRMG 特别做了正序和倒序文本图，因为倒序能更早看到“carrying bag”和“wearing bag”这种区分短语。CMRM 再用跨模态 KNN 图建图，学习图文局部之间的关系，并做二分类判断是否匹配。C2FL 先用较松的 Hc-Tri，等它收敛后逐步切到更严格的 Acc loss。

4. **证据闭环**：IMRM 相比全局和局部基线都有提升，F&B-TCRMG 好于只用正序或倒序，CMRM 加上 IMRM 最好。训练证据也对题：Hc-Tri 在约第 20 次迭代就收敛到 0，Acc 和 C2FL 能继续提供监督，C2FL 达到 69.16 Rank-1、86.23 Rank-5、64.57 mAP。热力图和检索例子显示，在外观细节高度相似时，模型能利用“包和人的关系”区分身份。

5. **reviewer 为什么买账**：这篇主要卖机制，但视角也足够明确。GNN 本身不是新东西，关键是作者先证明“局部物体一样但关系不同”是 TI-ReID 的真实失败点，再让图网络承担关系建模，而不是为了用图而用图。

最后看这批论文，共同套路是：先抓一个现有范式默认忽略的非对称性或混杂因素，比如点匹配忽略语义范围、局部对齐忽略关系、联合优化混入尾部解析、视觉对齐缺少显式语义。然后把旧任务重命名为一个更具体的新问题，再让机制几乎从这个新问题里长出来。能打动 reviewer 的不是模块复杂，而是消融和可视化能证明“我重新定义的问题确实存在，而且我的机制正好打在这个问题上”。
tokens used
439,366
下面按“创新怎么被构造出来”拆，不按模块复述。

**1. Probabilistic Distribution Alignment for Text-Based Person Retrieval**

1. **触发观察**：作者抓住了文本行人检索里的非对称性，文本通常是粗粒度描述，图像是具体实例。一个文本本来对应一组可能图像，而不是一个点对一个点。现有点嵌入把强正样本、弱正样本、相似负样本都挤在同一套距离逻辑里，容易把“语义范围”误当成“特征距离”。

2. **重定义动作**：它把问题从“图文相似度对齐”改写成“文本分布应当包含图像分布”。关键词是 probabilistic containment、semantic scope、partial-order relationship、text as probabilistic envelope。厉害的地方是把方差从“噪声余量”重新定义成“语义覆盖范围”。

3. **机制怎么长出来**：重定义之后，机制基本自然推出。图像和文本都建成高斯分布，均值表示语义中心，方差表示不确定性和范围；DRM 负责分布距离，CMC 让文本分布包含图像分布，IMC 让被遮挡或被掩码的样本分布包含完整样本。机制和叙事绑定很紧，因为每个损失都在维护“包含关系”。

4. **证据闭环**：关键不是只报涨点，而是证明“分布和包含”确实在起作用。消融里 DRM 把 CUHK-PEDES Rank-1 从 75.57 提到 76.41，CMC、IMC 单独和组合继续涨，完整 PDA 到 78.07。热力图显示注意力更贴合文本细节，分布可视化显示文本椭圆覆盖正确图像并排除负样本，这直接回应了“文本是语义范围”的主张。

5. **reviewer 为什么买账**：这篇卖的是视角，不只是机制。它不是又加一个对齐头，而是把 TBPR 的匹配关系从点对点改成非对称包含。novelty 来自“方差等于语义范围”这个解释，以及用闭式包含损失把这个解释落成可训练目标。

**2. Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification**

1. **触发观察**：VI-ReID 里常见做法是水平切分身体做局部对齐，但行人姿态、视角和运动不规则，水平条带经常语义错位。生成式跨模态图像又容易引入噪声，所以作者转向特征层。

2. **重定义动作**：它把“局部对齐”从固定人体条带改成“潜在语义相似局部的挖掘”。关键词是 prototype-driven、latent semantically similar features、diverse yet closely distributed features、instance-level alignment。旧问题是模态差异，新说法是缺少足够多样但分布接近的共享特征，以及缺少跨模态潜在语义局部。

3. **机制怎么长出来**：MFGM 用多分支膨胀卷积、注意力和中心引导 pair mining 生成多样但靠近共享分布的特征；PLM 用可学习原型给像素级特征加权，挖出跨 VIS/IR 语义相似的局部；Lch 保证原型多样，Ldcs 让类内靠近、类间中心分离。机制和重定义有联系，但比 PDA 弱一些，更像把“潜在局部”问题工程化。

4. **证据闭环**：组件消融显示 PLM、MFGM、Lch、Ldcs 都有贡献，原型数量从 6 到 10 性能提升，超过 10 又下降，说明原型不是越多越好。距离分布图给了更关键的证据：PLM 拉大类间距离，MFGM 缩小类内距离，两者结合间隔最大。t-SNE 和注意力图进一步说明模型确实更集中在行人区域和更清晰的身份簇上。

5. **reviewer 为什么买账**：这篇主要卖机制，不是强视角。novelty 来自把“水平部位不可靠”替换成“可学习原型挖潜在语义局部”，再用多特征生成解决模态分布不够的问题。它可学的一点是：先否定一个常用结构先验，再给出一个数据驱动替代物。

**3. Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification**

1. **触发观察**：LReID 要持续学习新域又不能忘旧域，存旧图像有隐私和存储风险。已有原型方法多来自类增量分类，主要调分类器，但 ReID 的核心是检索表征，旧身份原型和新特征提取器之间会产生表征偏差。静态融合旧模型和新模型也忽略不同任务的差异。

2. **重定义动作**：它把“灾难性遗忘”重写成“旧身份原型、新任务特征、更新后特征提取器之间的兼容问题”。关键词是 non-exemplar、prototype-guided knowledge propagation、representation bias、adaptive parameter evolution、compatibility。旧问题是保留旧知识，新说法是让旧知识在新表征空间里继续能区分。

3. **机制怎么长出来**：PKP 保存旧身份原型而不是旧图像，训练新任务时抽样旧原型并加高斯噪声，用改造 triplet 把新任务特征和旧原型云推开，同时用普通 triplet 保持新身份区分。APE 只更新对当前任务梯度影响大的参数，再按任务数据规模动态融合新旧模型。这个机制和重定义绑定较紧，因为它直接处理“旧原型和新提取器不兼容”。

4. **证据闭环**：组件消融很对题。baseline 的 Seen-Avg 是 42.8 mAP/57.4 Rank-1，加入 PKP 到 53.4/66.0，加入 APE 到 54.5/67.8，两者一起到 55.4/68.7；Unseen-Avg 也从 50.7/45.0 到 62.3/55.3。t-SNE 显示 PKA 后类内更紧、类间更分开，证明它不是只调分类器，而是在改善检索表征空间。

5. **reviewer 为什么买账**：这篇卖的是“非样本终身 ReID 的问题重定义”加一个够实用的机制。原型、triplet、参数融合都不是孤立新零件，但组合起来服务于一个清楚约束：不存图像，还要让旧身份原型继续在新表征里有效。

**4. Rethinking Joint Optimization in Feature Compression: Insights from Person Re-Identification**

1. **触发观察**：机器视觉压缩领域普遍认为联合优化更好，但很多联合优化管线末端都有语义解析网络。作者怀疑性能提升到底来自联合优化，还是来自尾部语义解析能力。

2. **重定义动作**：它把问题从“怎么做更好的联合压缩”改成“联合优化的收益来源是什么”。关键词是 disentangle、tailed semantics parsing、information capacity constraint、separate optimization、joint optimization。ReID 被选作工具，是因为它的尾部匹配可以只是欧氏距离，不需要可学习语义解析器。

3. **机制怎么长出来**：机制其实是实验设计。Opt S 分别训练 ReID 和压缩器；Opt JA 联合优化全部模块；Opt JH 冻结 backbone，只联合优化聚合层和压缩器。再用不同压缩率观察什么时候联合优化有效，什么时候反而伤害特征提取。

4. **证据闭环**：关键证据很强。CR=256 时联合优化更好，Opt S 是 26.54，Opt JA 是 40.29；但 CR 小于 128 后反过来，CR=16 时 Opt S 是 80.50，Opt JH 是 76.37，Opt JA 是 72.41。更关键的是原始特征性能，Opt JH_O 稳在 80.8 左右，而 Opt JA_O 只有 54 到 62，证明联合压缩约束伤了特征提取。增强 decoder 后 Opt JA 从 40.07 到 41.66，说明尾部语义解析确实是收益来源。

5. **reviewer 为什么买账**：这篇卖的是反常识视角和干净证据，不是新模块。它用 ReID 构造了一个能隔离混杂因素的实验场景，证明“联合优化好”这个共识并不总成立。方法论价值是：如果能拆掉领域里的混淆变量，负结论也能变成贡献。

**5. Richer Semantics, Better Alignment**

1. **触发观察**：VI-ReID 的可见光和红外图像有大模态差异，视觉特征很难直接对齐。已有 CLIP prompt 方法说明语义空间有帮助，但可学习 prompt 语义不透明、太粗、还要两阶段训练。LLaVA 能生成显式细粒度描述，但跨模态描述会有冲突，比如同一衣服在可见光里是蓝色，在红外里像灰色；同一身份不同视角的描述又互补。

2. **重定义动作**：它把 VI-ReID 从“视觉特征对齐到公共空间”改成“视觉特征对齐到显式且丰富的语义空间”。关键词是 explicit semantics、enriched semantics、indirect alignment、cross-view semantics compensation、one-to-many alignment。它不是简单说“用文本监督”，而是拆成显式语义、跨模态语义冲突、多视角语义不完整三个子问题。

3. **机制怎么长出来**：ESFA 用 LLaVA 给可见光和红外图像生成描述，在各自模态内做图文对齐。CSIA 不直接强拉图像和异模态文本，而是约束“模态内图文相似度”和“跨模态文本相似度”一致，绕开颜色冲突。CVSC 用同身份其他视角文本补偿当前文本，把一对一图文匹配扩成一对多语义对齐。机制和重定义绑定非常紧。

4. **证据闭环**：消融正好对应三个子问题。baseline 单次检索是 71.9 Rank-1/67.6 mAP，ESFA 到 76.3/72.6，加入 CSIA 到 77.4/73.2，ESFA+CVSC 到 77.6/73.7，完整模型到 78.4/74.8。进一步讨论里，显式语义强于隐式 prompt，直接跨模态文本对齐会让 ESFA 变差，而间接对齐会提升，这正好证明了“语义冲突必须间接处理”。

5. **reviewer 为什么买账**：这篇卖的是视角和机制都成立。真正 novelty 不是“用了 LLaVA”，而是把外部语义引入 VI-ReID 后可能遇到的三个障碍逐个定义清楚，并且每个障碍都有对应机制和消融。

**6. RMGNet**

1. **触发观察**：TI-ReID 里很多方法已经做局部图文对齐，但作者指出只看局部对象还不够。两个人都可能有灰外套、黑包，差别在“背着包”还是“穿着包”这样的局部关系。另一个观察是 Hc-Tri loss 很早收敛到 0，后期没有有效监督。

2. **重定义动作**：它把“局部特征对齐”改写成“局部特征之间关系的建模和挖掘”。关键词是 relationship-based matching、progressive relationship-mining graph、intra-modal relationship、cross-modal relationship、continuous effective metric supervision。旧问题是找局部，新问题是理解局部之间的语义关系。

3. **机制怎么长出来**：IMRM 在图像局部和文本局部内部建图，用 GNN 聚合关系信息。TCRMG 特别做了正序和倒序文本图，因为倒序能更早看到“carrying bag”和“wearing bag”这种区分短语。CMRM 再用跨模态 KNN 图建图，学习图文局部之间的关系，并做二分类判断是否匹配。C2FL 先用较松的 Hc-Tri，等它收敛后逐步切到更严格的 Acc loss。

4. **证据闭环**：IMRM 相比全局和局部基线都有提升，F&B-TCRMG 好于只用正序或倒序，CMRM 加上 IMRM 最好。训练证据也对题：Hc-Tri 在约第 20 次迭代就收敛到 0，Acc 和 C2FL 能继续提供监督，C2FL 达到 69.16 Rank-1、86.23 Rank-5、64.57 mAP。热力图和检索例子显示，在外观细节高度相似时，模型能利用“包和人的关系”区分身份。

5. **reviewer 为什么买账**：这篇主要卖机制，但视角也足够明确。GNN 本身不是新东西，关键是作者先证明“局部物体一样但关系不同”是 TI-ReID 的真实失败点，再让图网络承担关系建模，而不是为了用图而用图。

最后看这批论文，共同套路是：先抓一个现有范式默认忽略的非对称性或混杂因素，比如点匹配忽略语义范围、局部对齐忽略关系、联合优化混入尾部解析、视觉对齐缺少显式语义。然后把旧任务重命名为一个更具体的新问题，再让机制几乎从这个新问题里长出来。能打动 reviewer 的不是模块复杂，而是消融和可视化能证明“我重新定义的问题确实存在，而且我的机制正好打在这个问题上”。
