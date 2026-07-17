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
session id: 019ef835-f7a5-7660-851d-f24d60a84028
--------
user
你是 ReID 创新方法论拆解员 23。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(23)==
RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf
Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则确认当前目标和相关规则是否能读取，然后逐篇用 `pdftotext -f 1 -l 8` 抽取方法部分。后面如果方法部分在前 8 页截断，我会说明依据受限于该命令输出，并用检索补足背景判断。
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
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc 'pwd && ls' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "sed -n '1,220p' ../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

exec
/bin/zsh -lc "sed -n '1,240p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,220p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

exec
/bin/zsh -lc 'pdftotext -v' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
pdftotext version 26.04.0
Copyright 2005-2026 The Poppler Developers - http://poppler.freedesktop.org
Copyright 1996-2011, 2022 Glyph & Cog, LLC

codex
规则和目标已经读到，当前任务仍然是文献差距分析，不涉及开实验。我现在开始逐篇抽取前 8 页正文，重点抓作者如何从观察走到问题重定义和机制。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113919

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

RMPSNet: Occluded person re-identification via regional masking and
prompt-distribution synergy
Zan Gao a,b , Shuai Xie a , Shengxun Wei a
a
b

,∗, Yibo Zhao a

, Chunjie Ma a,b , Chen Li a

Key Laboratory of Computer Vision and System, Ministry of Education, Tianjin University of Technology, Tianjin, 300384, China
Shandong Artificial Intelligence Institute, Qilu University of Technology (Shandong Academy of Sciences), Jinan, 250014, Shandong, China

ARTICLE

INFO

Keywords:
Occluded person ReID
Prompt learning
Vision-language model

ABSTRACT
Occluded Person Re-identification (ReID) aims to retrieve images of the same individual under various
occlusion conditions. The primary obstacles arise from the semantic loss induced by occlusion and the
misalignment between the visual and textual features caused by occlusions. These challenges motivate us
to explore more robust vision-language interaction mechanisms and targeted feature enhancement strategies
to improve the resilience of the model to occlusion. To this end, we propose RMPSNet, a novel end-toend framework built upon the CLIP backbone that enhances occlusion robustness through multi-prompt
learning and distribution-level adaptation. Specifically, we propose a region-prioritized erasure augmentation
strategy that simulates realistic occlusion patterns by preferentially masking lower body regions. Furthermore,
we propose a dual-masked prompt augmentation module that performs complementary mask operations in
the textual embedding space to enhance cross-modal alignment, and a multi-branch distribution alignment
mechanism that applies diverse feature transformations and adversarial constraints to maintain consistency
with learned prototypes. Extensive experiments on occluded (Occluded-Duke, Occluded-ReID, Partial-ReID) and
general (Market-1501, DukeMTMC-ReID) ReID datasets demonstrate the superiority of the proposed method,
which achieves rank-1 accuracies of 76.0%, 93.2%, 92.0%, 95.6%, and 91.1% respectively. Compared with
KPR (ECCV 2024), ETND (TCSVT 2024) and TTPM (PR 2025), RMPSNet improves the rank-1 performance on
the Occluded-ReID dataset by 9.9%, 3.9%, and 7.0%, respectively.

1. Introduction
Person Re-identification (ReID) refers to the task of retrieving
pedestrians from non-overlapping surveillance cameras. Occluded Person ReID [1,2] has attracted significant research attention in recent
decades. In recent years, with the advancement of deep learning,
significant breakthroughs have been made in person ReID. However,
in occlusion scenarios, due to interference caused by occlusion and the
absence of some occluded features, the ReID task still faces significant
challenges [3]. To address the challenges of occluded person ReID, it
is essential to extract features that are both highly representative and
discriminative. Convolutional Neural Networks (CNNs), once dominant
in ReID tasks, have shown effectiveness in many visual recognition
scenarios. However, CNNs typically focus on limited local regions,
which may not correspond to identity-relevant areas, resulting in
suboptimal robustness and weakened discriminative capacity under
occlusion. In contrast, Vision Transformers (ViTs) have gained popularity in computer vision due to their ability to model long-range

dependencies without relying on convolutional or downsampling operations. Additionally, recent advances in cross-modal learning, such
as CLIP [4], enable joint modeling of visual features and their corresponding high-level textual descriptions through large-scale contrastive
language-image pretraining. Building upon the success of CLIP, Li
et al. [5] introduced CLIP-ReID, which extends CLIP to person ReID
by incorporating learnable textual tokens to complement visual representations. While this approach demonstrates strong performance
on general person ReID tasks, its effectiveness diminishes in occluded
scenarios due to modality misalignment. Specifically, three major limitations remain when applying CLIP to occluded person ReID: (1) Lack
of cross-modal compensation: When visual features are corrupted by
occlusion, there is no effective mechanism to enhance or supplement
them with complementary textual information. (2) Limited geometric
robustness: Standard CLIP is sensitive to spatial transformations caused
by occluders and lacks explicit mechanisms to handle such deformations. (3) Occlusion distribution mismatch: In real-world environments,

∗ Corresponding author.

E-mail addresses: zangaonsh4522@gmail.com (Z. Gao), 281298xh@gmail.com (S. Xie), wei_shengxun@163.com (S. Wei), zybtjut@163.com (Y. Zhao),
mcj@machunjie.com (C. Ma), cli@email.tjut.edu.cn (C. Li).
https://doi.org/10.1016/j.patcog.2026.113919
Received 23 September 2025; Received in revised form 20 April 2026; Accepted 2 May 2026
Available online 6 May 2026
0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 179 (2026) 113919

Z. Gao et al.

(a)

(b)

Fig. 1. (a) Most of the occlusion in daily life occurs in the head and legs, with more than 70% concentrated in the legs. (b) The simple random occlusion strategy
can only simulate a small part of the real situation, which is inconsistent with the real occlusion distribution.

occlusions frequently affect the head and, more prominently, the lower
body (see Fig. 1(a)). However, commonly used augmentation techniques such as random erasing apply masking uniformly across the
image (Fig. 1(b)), failing to reflect the true distribution of occlusions
and resulting in a domain gap between training and deployment
conditions.
To address the above problems, we propose RMPSNet. Inspired by
multi-prompt learning strategies, the Dual-Masked Prompt Augmentation (DMPA) module is designed to perform bidirectional masked
enhancement in the text feature space. By applying contrastive loss to
constrain visual-language alignment, it mitigates the semantic degradation caused by occlusion and addresses the limitations of cross-modal
feature compensation. To simulate real-world occlusion patterns more
accurately, the Region Prioritized Erasure (RPE) module is introduced.
This component selectively masks image regions based on empirical occlusion distributions, thereby reducing the mismatch between
simulated occlusion patterns in training and real-world occlusion distributions. Additionally, drawing on prior work in feature enhancement,
we develop a Multi-Enhancement Distribution Optimization (MDO)
module involving erasure, noise injection, and geometric transformation. This is coupled with adversarial optimization, where destructive
perturbations are compensated through reconstruction mechanisms, enhancing the model’s robustness to geometric variations and addressing
CLIP’s sensitivity to spatial distortions. The main contributions of this
paper are summarized as follows:

under complex and severe occlusion conditions. One common direction
is to exploit part-based representations to improve spatial alignment
and local feature robustness. Early methods either adopt rigid partitioning schemes, such as horizontal stripes or uniform blocks, or
introduce auxiliary models such as pose estimation [6] and semantic
parsing [7] to guide feature extraction. While effective in relatively
simple cases, these techniques may become less reliable under dynamic or irregular occlusions. Rigid partitioning lacks flexibility in
handling unpredictable occlusion patterns and may overlook identityrelevant regions. Similarly, auxiliary models such as OpenPose [6] can
suffer from degraded keypoint localization under occlusion while introducing additional computational cost. In addition, alignment-based
methods that rely on consistent component matching may become
less effective under heavy occlusion. Attention-based methods, particularly those built on Transformer architectures, have also attracted
increasing attention because of their ability to model long-range dependencies. However, occlusion may still disturb attention allocation
and cause the model to focus on irrelevant or noisy regions, thereby
suppressing identity-critical features. Moreover, although global attention mechanisms provide strong representational capacity, they are
often computationally intensive and may be less suitable for realtime or resource-constrained applications. Attempts to enhance local
perception, such as PAT [8], have shown promising results, but they
still remain limited in recovering occluded features without sufficient
semantic guidance.
Another line of research improves robustness through occlusion
simulation, for example by using Random Erasing [9] or occlusion
generation methods such as ISGAN [10]. Although these strategies
can improve generalization to some extent, they may also introduce
a mismatch between simulated and real occlusion patterns, resulting
in distribution bias or semantic inconsistency. Overall, existing occluded ReID methods have achieved encouraging progress, but further
improvement is still needed in robust local perception, semantic consistency, and practical efficiency under real-world occlusion scenarios.
Recent studies have further enriched this research line. For example,
occluded person ReID has benefited from multi-view information integration and propagation [11]. In person-related cross-modal retrieval,
implicit local alignment has also shown effectiveness [12]. Beyond
ReID, robust representation learning has also been explored in RGB-T
salient object detection [13]. Related advances have also been reported
in few-shot fine-grained recognition [14]. In addition, recent multimodal learning studies provide complementary insights into robust
representation learning [15]. These advances help better position our
method within the broader landscape of ReID and image recognition
research. Different from previous occluded ReID methods that mainly
rely on rigid partitioning, auxiliary parsing/pose models, or generic
random erasing, RMPSNet explicitly addresses distribution mismatch
and cross-modal semantic degradation through region-prioritized erasure, dual-masked prompt augmentation, and multi-branch robustness
optimization.

• We proposed a novel RMPSNet framework for occluded person
ReID. By simulating the occlusion scenarios in the real world, we
enhance the model’s adaptability to such scenarios.
• We designed the DMPA module, which performs bidirectional
masking enhancement in the text feature space, alleviating the
semantic degradation caused by occlusion and addressing the
limitations of cross-modal feature compensation. We proposed
the RPE module to selectively mask image regions, simulating
more realistic occlusion scenarios, and reducing the mismatch
between simulated occlusion patterns and real-world occlusion
distributions in training. Furthermore, we developed the MDO
module, which enhances the model’s robustness to geometric
changes and solves the sensitivity of CLIP to spatial distortion.
• We evaluated the RMPSNet algorithm on three occluded person
ReID datasets and two general person ReID datasets. The experimental results show that RMPSNet consistently outperforms existing methods in occluded scenarios, while achieving competitive
performance on general person ReID datasets.
2. Related work
2.1. Occluded person re-identification
The main challenge in occluded person ReID lies in the loss of
local discriminative cues and the spatial misalignment caused by partial
visibility. These issues weaken the model’s ability to capture finegrained identity information. Although various methods have been
proposed to address this problem, their performance can still be limited

2.2. Vision-language pre-training for re-identification
In recent years, vision-language pre-trained models such as CLIP [4]
have introduced cross-modal semantic reasoning into person ReID by
2

Pattern Recognition 179 (2026) 113919

Z. Gao et al.

leveraging large-scale image–text pairs. These models show strong
potential for occluded person ReID, but several limitations remain
in occlusion-sensitive scenarios. A major issue is the mismatch between coarse-grained global alignment and the fine-grained semantic degradation caused by occlusion. Methods such as CLIP-ReID [5]
align visual and textual features through prompt-based supervision
in the global embedding space, which improves identity-level semantic consistency but remains limited in associating textual cues with
localized occluded regions. ProFD [16] is a recent prompt-learningbased method for occluded person ReID under the vision-language pretraining paradigm, which enhances identity representation by leveraging learnable prompts to mitigate occlusion interference. Different from
ProFD [16], which mainly focuses on prompt-guided representation
learning within a CLIP-style backbone, our RMPSNet further addresses
the occlusion distribution mismatch through RPE and the robustness
gap between original and augmented features through MDO with stepwise adversarial training. Moreover, RMPSNet introduces DMPA with
an inter-text contrastive constraint to enhance the robustness of IDconditioned text prototypes under missing semantic components.
In addition, local semantic fusion methods often incur extra computational cost and may rely on external localization cues. For example, DenseCLIP [17] and QAConv-GLIP [18] improve region-level
understanding, but their performance can be affected by region localization quality and the additional overhead of detection-guided
processing. Generic prompt adaptation methods such as CoOp [19]
and Tip-Adapter [20] improve downstream adaptation, yet they are
primarily designed for closed-set classification and do not explicitly address the open-set, cross-view nature of person ReID. Therefore, visionlanguage ReID under occlusion still requires more effective fine-grained
semantic modeling and stronger robustness to occlusion-induced feature variation.

learning to robustness-oriented representation refinement. Specifically,
in the first stage, the parameters of the text encoder and image encoder
are frozen, and a contrastive loss is applied to guide the learning
of original prompts. Meanwhile, the prompts are enhanced through
the DMPA module, which then jointly guides the learning of original
prompts. In the second stage, input-level occlusion simulation and
feature-level robustness optimization are performed through RPE and
MDO, and an adversarial supervision loss function is constructed by
assigning positive and negative weights.
3.1. Dual-masked prompt augmentation (DMPA)
Although existing data augmentation methods provide some simulation of occluded scenes for the model. However, data augmentation
alone cannot fully address the issue of cross-modal semantic loss. Under
occlusion conditions, traditional CLIP-based ReID methods often struggle with insufficient feature disentanglement, unidirectional alignment
biases, and the risk of noise propagation. To overcome these challenges,
we propose dual-masked prompt augmentation (DMPA) module, which
enhances semantic robustness by leveraging adversarial masking and
contrastive reconstruction within the text space. Specifically, in the first
training stage, we apply double masking to the learnable text prompt
(e.g., A photo of a [𝑆]1 [𝑆]2 [𝑆]3 … [𝑆]𝑀 person.) as follows:
𝑦̂1 , 𝑦̂2 = Mask(𝑦, 𝛼),

(1)

where 𝑦̂ represents the masked text prompt, Mask is the masking
operation, which sets the value to 0, 𝑦 is the learnable parameters, and
𝛼 is the masking ratio, which we set to 0.5. Enhanced prompts derived
from the same text form positive pairs, while those from different texts
form negative pairs, with all tokens fixed during this process. The
original prompt is then encoded by the text encoder, and its feature
is aligned with the image feature via a cross-modal similarity score:

2.3. Adversarial optimization in re-identification

𝑠(𝑉𝑖 , 𝑇𝑖 ) = 𝑉𝑖 ⋅ 𝑇𝑖 = 𝑔𝑉 (img𝑖 ) ⋅ 𝑔𝑇 (text 𝑖 ),

Contrastive and adversarial optimization strategies have been widely
explored to improve robustness in person ReID. Existing methods
mainly focus on feature-level alignment, adversarial augmentation, and
collaborative optimization. Feature-level adversarial strategies reduce
camera or viewpoint discrepancies through techniques such as gradient
reversal or domain adversarial loss, but they are not specifically designed for occlusion and usually rely on global feature alignment, which
is limited in handling the semantic loss caused by local occlusions.
Recent studies have further introduced feature decoupling mechanisms
within attention modeling, such as occlusion-sensitive area suppression [21], to alleviate this problem, though at the cost of additional
complexity.
Sample-level adversarial strategies enhance robustness by generating perturbed training samples, such as AdvPattern [22]. However,
excessive perturbation may also damage identity-relevant cues and
reduce recognition accuracy. Multi-task collaborative learning has also
been explored to jointly optimize identity-related objectives [23], but
effective coordination between multiple enhanced learning branches
remains insufficiently studied. Overall, existing robust optimization
methods improve ReID performance to some extent, yet they still lack
a targeted mechanism for handling occlusion-induced feature variation and the distribution discrepancy between original and enhanced
representations.

(2)

where 𝑔𝑉 (⋅) and 𝑔𝑇 (⋅) represent the image encoder and text encoder
respectively. The image-to-text and text-to-image contrastive loss are
defined as:
exp(𝑠(𝑉𝑖 , 𝑇𝑖 ))
𝑖2𝑡 (𝑖) = − log ∑𝐵
,
(3)
𝑎=1 exp(𝑠(𝑉𝑖 , 𝑇𝑎 ))
exp(𝑠(𝑉𝑖 , 𝑇𝑖 ))
𝑡2𝑖 (𝑖) = − log ∑𝐵
,
(4)
𝑎=1 exp(𝑠(𝑉𝑎 , 𝑇𝑖 ))
where 𝑉 is the image feature embedding, 𝑇 is the text feature embedding, 𝑠(⋅, ⋅) represents the inner product similarity calculation, and 𝐵
denotes the batch size. Next, the original text prompts are enhanced
through prompt enhancements to obtain double-enhanced prompts.
The obtained double-enhanced prompts are also passed through the
text encoder to obtain the corresponding masked feature, which is
further refined using an inter-text contrastive loss. The contrastive loss
is defined as follows:
[
(
)
(
) ]
𝑁
exp 𝑠(𝑇𝑖 , 𝑇𝑖′ )
exp 𝑠(𝑇𝑖′ , 𝑇𝑖 )
1 ∑
𝑡2𝑡 = −
log ∑𝑁
(5)
(
) + log ∑𝑁
( ′
)
′
𝑁 𝑖=1
𝑗=1 exp 𝑠(𝑇𝑖 , 𝑇𝑗 )
𝑗=1 exp 𝑠(𝑇𝑖 , 𝑇𝑗 )
where 𝑇𝑖 and 𝑇𝑖′ denote two text features of the same identity obtained by applying the dual-masked prompt augmentation twice, and
𝑗 enumerates all candidate text features within the mini-batch. 𝑁
denotes the number of text prototypes involved in the batch for text-totext contrastive learning. 𝑠(⋅, ⋅) is the cosine similarity function in the
embedding space. The two log-terms implement a symmetric InfoNCE
objective. The total loss for the first stage is defined by:

3. The proposed approach
As shown in Fig. 2, RMPSNet is implemented through a two-stage
optimization strategy. This strategy enhances robustness against occlusions through multi-prompt learning and distribution-level adaptation,
and establishes a minimax game equilibrium in the feature space,
surpassing traditional single-stage adversarial training. The training
process can be understood as a transition from semantic prototype

𝐿𝑠𝑡𝑎𝑔𝑒1 = 𝑖2𝑡 + 𝑡2𝑖 + 𝜆𝑡2𝑡 𝑡2𝑡 ,

(6)

with 𝜆𝑡2𝑡 empirically set to 0.8. For computational efficiency, all image
features are extracted at the start of this stage. For a dataset with 𝐶
3

Pattern Recognition 179 (2026) 113919

Z. Gao et al.

Fig. 2. Pipeline of the proposed RMPSNet approach. It consists of the backbone, the DMPA module, the RPE module, and the MDO module, which is a two-stage
model. Specifically, in stage1, the prompt tokens are enhanced through the DMPA module and then jointly guide the learning of the original prompts. In stage2,
the RPE module is used to enhance the image input, which is then further processed by the MDO module to jointly optimize the image encoder.

identities, we store the corresponding 𝑇̂𝑦𝑖 features at the end of this
stage.
It is worth noting that the masking operation in DMPA is not
intended to establish a one-to-one correspondence between a masked
prompt token and a specific occluded image patch. Instead, we view
occlusion as a partial-information condition in the shared embedding
space. In CLIP-style ReID, the learnable text prompt acts as an identityconditioned semantic prototype for cross-modal alignment. By applying dual masking to the prompt tokens, DMPA introduces structured
missing-information perturbations on the text side and enforces the
resulting text features to remain semantically consistent through the
inter-text contrastive loss. This mechanism reduces the co-adaptation
of prompt dimensions and encourages the text prototype to remain
discriminative even when part of the semantic context is removed.
As a result, occluded images, which also contain incomplete identity
evidence, can still be aligned with a stable text-side semantic anchor,
thereby alleviating semantic degradation under occlusion.

define the aspect ratio, where ln(𝛼) ∼ 𝑈 [ln(0.3), ln(3.3)]. Based on 𝛼,
the height ℎ and width 𝑤 of the erasure region are calculated as ℎ =
⌊𝐴 ⋅ 𝛼⌋, 𝑤 = ⌊𝐴∕𝛼⌋, where 𝐴 denotes the dynamically sampled area
proportion. Finally, Gaussian noise is injected into the selected region
to simulate the occlusion effect, enhancing the model’s robustness to
real-world occluded scenarios. This targeted erasure strategy allows the
model to better focus on learning discriminative features from partially
visible body regions, thereby mitigating the impact of biased occlusion
distributions.
To further enhance occlusion robustness, the RPE strategy is also
applied at the feature erasure stage in a manner consistent with the
image-level prior. Specifically, after feature extraction, we perform
feature-level erasure on the patch-token representation while keeping
the CLS token unchanged. With a probability of 70%, the erased segment is sampled from the token sub-region corresponding to the lower
body, and with the remaining 30%, the erased segment is sampled
from the entire token sequence. The erased segment length is randomly
determined to simulate diverse occlusion scales, and the selected token
values are set to zero to mimic feature information loss caused by realistic occlusion. In this way, RPE is consistently imposed in both image
space and feature space, enabling the model to learn complementary
cues from partially visible regions and further reducing the distribution
mismatch between simulated augmentation and real-world occlusion.
This method effectively addresses the distribution mismatch problem
inherent in traditional random erasure and provides crucial technical
support for robust person ReID under occlusion. Together, they enable
the model to overcome the occlusion bottleneck and achieve more
robust identity representations.

3.2. Region Prioritized Erasure (RPE)
Traditional random erasure methods used in occluded person ReID
generally assume a spatially uniform distribution of occlusion regions,
which does not fully reflect real-world conditions. In practice, occlusions are more likely to occur in specific body regions, especially the
lower body, due to luggage, shopping carts, bicycles, or overlapping
pedestrians. Therefore, generic random erasing strategies may fail to
explicitly model the spatial prior of real occlusions, leading to a mismatch between synthetic occlusion patterns and real-world scenarios.
To reduce this distribution mismatch, we propose the region prioritized
erasure (RPE) module. It is designed to simulate realistic occlusion patterns by prioritizing lower-body regions during image augmentation.
Specifically, each input image first undergoes random augmentation
operations, including random horizontal flipping and random cropping.
Following these operations, a region erasure process is applied with a
probability of 0.5. If selected, there is a 70% probability of erasing a
region within the lower part of the image (i.e., the bottom half), and a
30% probability of performing uniform erasure across the entire image.
The erased region area is dynamically determined, ranging from 2%
to 40% of the total image area. A random parameter 𝛼 is sampled to

3.3. Multi-enhancement distribution optimization (MDO)
Traditional CLIP-based ReID methods [5,17] exhibit limitations
when addressing occluded scenarios, as they primarily rely on global
image–text alignment and lack explicit local feature enhancement
mechanisms. Moreover, there exists a distribution discrepancy between
the simulated augmented features and the original features, which
weakens generalization capability in real-world occlusion cases. To
mitigate these challenges, we propose the multi-enhancement distribution optimization (MDO) module, which integrates three distinct
4

Pattern Recognition 179 (2026) 113919

Z. Gao et al.

feature enhancement operations combined with a step-by-step adversarial optimization strategy to improve occlusion robustness. Note that
the noise branch in MDO is implemented as a simple feature-level perturbation strategy for robust optimization, rather than a diffusion-based
generative or denoising process.
Specifically, MDO extends the original feature representation by
incorporating three types of feature augmentation: region erasure, noise
injection, and geometric transformation. For region erasure, parts of
the feature map are randomly erased either within specific lower-body
regions or across the entire map to simulate partial feature loss under
occlusion. With a probability of 70%, the erasure targets only the lower
half of the pedestrian image, while the remaining 30% performs random erasure over the entire feature map. The erased region size is also
random to reflect diverse occlusion scales, thus enhancing the model’s
generalization ability. In the noise injection operation, Gaussian noise
with a fixed standard deviation of 0.2 is added to high-dimensional
features to alleviate the impact of noise disturbances commonly present
in occluded scenes. For geometric transformation, we introduce a local
feature replacement strategy to address viewpoint variations and shape
deformations. A segment within the feature sequence is replaced with
content from another region of the same sequence to mimic occlusions by external objects. Concretely, we operate on the patch-token
sequence (excluding the CLS token). We randomly sample a segment
length 𝐿 and two starting indices 𝑆𝑝 and 𝑆𝑞 , and then replace the token
segment [𝑆𝑝 , 𝑆𝑝 + 𝐿) with [𝑆𝑞 , 𝑆𝑞 + 𝐿), while keeping the CLS token
unchanged. Since this operation occurs in feature space rather than the
pixel domain, it avoids low-level distortions while enhancing geometric
robustness.
During Stage 2 training, MDO takes part using a step-by-step adversarial optimization strategy. This strategy is designed to ensure
distribution consistency between enhanced and original features. First,
we freeze the parameters of the backbone network. Then, we only
optimize the classifier to create adversarial gradients. This makes the
classifier suppress noise-sensitive dimensions. As a result, it improves
the robustness to occlusion and deformation. The corresponding loss is
defined as:
1𝑠𝑡𝑎𝑔𝑒2 = 𝑖𝑑 − 𝜆𝑛𝑜𝑖𝑠𝑒 𝑖𝑑_𝑛𝑜𝑖𝑠𝑒 − 𝜆𝑒𝑟𝑎𝑠𝑒 𝑖𝑑_𝑒𝑟𝑎𝑠𝑒 − 𝜆𝑡𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚 𝑖𝑑_𝑡𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚 ,

feature augmentations, enables the model to achieve stronger occlusion robustness and improved generalization in challenging scenarios. The process of implementing adversarial denoising with negative
weights, followed by enhancing feature robustness through collaborative optimization with positive weights, gradually optimizes the
gradient conflict problem among multiple enhancement branches. The
above optimization should not be interpreted as a naive sign-flipping
strategy. Instead, it is an alternating minimax-style optimization in
parameter space. In the first step, only the classifier is updated while
the backbone is frozen, which constructs stricter decision boundaries
for the enhanced feature branches under fixed feature representations
and prevents uncontrolled feature drift. In the second step, the full
network is optimized to reduce the vulnerability exposed by these enhanced branches, thereby improving feature robustness to perturbationinduced distribution shifts. Therefore, the adversarial effect in MDO is
realized through alternating optimization between classifier robustness
and feature adaptation, rather than through pixel-level adversarial
sample generation. This design makes the training process more stable
while still preserving the benefit of worst-case-oriented robustness
enhancement.
4. Experiments and discussion
To evaluate the effectiveness of the proposed RMPSNet, we conducted experiments on five widely used public person ReID datasets.
These datasets serve as well-recognized benchmarks in the field of
person ReID, providing a reliable basis for assessing the performance
of the RMPSNet.
4.1. Datasets
We evaluated RMPSNet on two general ReID datasets (Market1501 [2], DukeMTMC-ReID [25]) and three occluded ReID datasets
(Occluded-Duke [3], Occluded-ReID [26], Partial-ReID [27]), each
with unique characteristics that highlight different challenges. Market1501 [2] comprises 32,668 annotated images of 1501 identities captured by six cameras in an open-world setting. It provides 12,936
training images covering 751 identities, making it a standard benchmark for evaluating general person ReID performance under diverse
camera views. DukeMTMC-ReID [25] contains 36,411 images of 1404
identities collected from eight camera views. The dataset includes
16,522 training images, 17,661 gallery images, and 2228 query images. It has challenging cross-camera matching scenarios and complex
background variations. Occluded-DukeMTMC [3] is derived from
DukeMTMC-ReID and specifically focuses on occlusion. It consists of
15,618 training images of 708 identities, 2210 query images of 519
identities, and 17,661 gallery images of 1110 identities. Notably, all
query images are partially occluded by various obstacles, making this
dataset ideal for evaluating occlusion robustness. Occluded-ReID [26]
comprises 2000 images of 200 identities, with each identity represented
by 10 images-five full-body images and five occluded images. Typically,
occluded images serve as queries, while all full-body images are used as
the gallery. This dataset emphasizes testing a model’s ability to recognize individuals under severe occlusion conditions. Partial-ReID [27] is
an image-based partial person ReID dataset, encompassing images with
various perspectives, backgrounds, and occlusion types. The dataset
consists of 600 photos from 60 individuals, with each person having
5 full-body images and 5 partial images. These partial images were
manually cropped from occluded images and only include the test set
used for evaluation.
Note that as the Occluded-ReID and Partial-ReID datasets have no
official training splits, we follow the evaluation protocols in [26,27]
and train our model on Market-1501. The performance on this benchmark thus reflects the cross-dataset generalization capability of our
method under dataset distribution shift.

(7)

where 𝜆𝑛𝑜𝑖𝑠𝑒 = 0.1, 𝜆𝑒𝑟𝑎𝑠𝑒 = 0.1, and 𝜆𝑡𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚 = 0.15.
It should be noted that although both RPE and MDO are related to
occlusion robustness, their functions are complementary. RPE mainly
plays a role at the input level, reducing the mismatch between simulated occlusion patterns and actual monitoring scenarios by simulating
real occlusion distributions during the training process. In contrast,
MDO operates in the feature space, by introducing multiple perturbation branches and adversarial optimization, it enhances the robustness of features and the consistency of distributions under various
interferences.
After classifier optimization, we proceed to fine-tune the entire
network. Following prior works [5,24], we compute cross-entropy loss
and triplet loss to refine feature discrimination. The cross-entropy loss
from image to text is defined as:
𝑖2𝑡𝑐𝑒 (𝑖) =

𝑁
∑

exp(𝑠(𝑉𝑖 , 𝑇𝑦𝑘 ))
−𝑞𝑘 log ∑𝑁
,
𝑘=1
𝑦 =1 exp(𝑠(𝑉𝑖 , 𝑇𝑦𝑎 ))

(8)

𝑎

In this stage, we also construct an adversarial objective to balance
feature robustness and discriminative power:
′

1𝑠𝑡𝑎𝑔𝑒2 = 𝑖𝑑 + 𝜆𝑛𝑜𝑖𝑠𝑒 𝑖𝑑_𝑛𝑜𝑖𝑠𝑒 + 𝜆𝑒𝑟𝑎𝑠𝑒 𝑖𝑑_𝑒𝑟𝑎𝑠𝑒 + 𝜆𝑡𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚 𝑖𝑑_𝑡𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚 ,

(9)

The final loss function for Stage 2 is given by:
′

2𝑠𝑡𝑎𝑔𝑒2 = 𝑖𝑑 + 𝜆1 𝑡𝑟𝑖 + 𝑖2𝑡𝑐𝑒 + 𝜆2 1𝑠𝑡𝑎𝑔𝑒2 ,

(10)

where 𝜆1 = 0.25 and 𝜆2 = 0.9 are the weights for balancing different
losses. This two-step adversarial optimization, combined with targeted
5

Pattern Recognition 179 (2026) 113919

Z. Gao et al.

Table 1
Performance evaluation and comparison on five public datasets. The bold values indicate the best performance in each column. The symbol ‘-’ indicates that the
corresponding result was not reported in the original paper.
Methods

Backbone

Datasets
General ReID benchmarks
Market-1501

Occlusion ReID benchmarks

DukeMTMC-ReID

Occluded-Duke

Occluded-ReID

Partial-ReID

mAP

R1

mAP

R1

mAP

R1

mAP

R1

mAP

R1

BPB(WACV 2023) [29]
QPM(TMM 2023) [30]
PRE(TCSVT 2023) [31]

CNN
CNN
CNN

88.4
–
86.0

95.7
–
94.5

81.3
–
76.5

91.7
–
88.9

57.5
49.7
54.3

71.3
64.4
67.1

70.9
–
–

77.0
–
–

–
–
–

–
81.7
86.0

SPH(TCSVT 2024) [32]
THCB(TIFS 2025) [33]

CNN+Transformer
CNN+Transformer

–
90.6

–
96.2

–
83.5

–
91.7

63.7
62.6

71.9
72.3

82.1
84.5

82.6
87.3

–
84.2

–
87.4

FED(CVPR 2022) [34]
PFD(AAAI 2022) [35]
RGA(TIFS 2023) [36]
CLIP-ReID(AAAI 2023) [5]
AMG(TMM 2023) [37]
DRL(TMM 2023) [38]
SAP(AAAI 2023) [39]
TSD(ICASSP 2024) [40]
SPT(AAAI 2024) [41]
ADP (AAAI 2024) [42]
HGTDR (TCSVT 2024) [43]
ETND(TCSVT 2024) [44]
PADE(ICASSP 2024) [45]
ProFD(ACM MM 2024) [16]
KPR(ECCV 2024) [46]
FCF(TMM 2024) [47]
FRCE(ESWA 2024) [48]
CAM(InfFus 2025) [49]
MTIPE(PR 2025) [50]
TTPM(PR 2025) [51]
DPEFormer(KBS 2025) [52]

Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer
Vision Transformer

86.3
89.6
89.8
89.6
88.4
86.9
90.5
–
89.4
89.5
89.8
87.2
89.8
90.0
88.7
–
88.1
90.1
89.0
90.0
88.1

95.0
95.5
95.5
95.5
94.8
94.7
96.0
–
95.5
95.6
95.6
95.3
95.8
95.1
95.6
–
95.2
96.0
95.2
95.6
95.4

78.0
83.2
–
82.5
–
76.6
–
82.8
82.4
83.1
83.3
77.9
82.8
83.2
–
–
83.0
–
80.8
83.5
80.3

89.4
91.2
–
90.0
–
88.1
–
90.8
91.1
91.2
91.2
88.5
91.3
91.7
–
–
91.3
–
90.1
91.4
90.0

56.4
60.1
62.4
59.5
59.6
53.9
62.2
62.8
63.0
63.8
62.3
54.7
63.0
62.8
63.4
63.1
61.6
64.8
57.8
63.4
58.9

68.1
67.7
71.6
67.1
68.1
65.8
70.0
74.5
74.7
74.5
71.8
63.6
72.3
70.8
75.8
73.0
72.4
74.4
66.4
73.4
69.9

79.3
81.3
80.0
–
50.2
–
76.8
83.1
81.1
85.1
83.3
81.5
79.9
88.5
78.2
85.7
79.6
80.9
–
82.2
79.5

86.3
79.8
86.4
–
61.9
–
83.0
87.3
87.8
89.2
87.2
89.3
83.7
91.1
83.3
83.6
84.4
85.6
–
86.2
87.0

80.5
–
–
–
–
–
–
–
–
–
–
–
84.8
–
81.7
–
–
–
–
–
–

83.1
–
87.2
–
–
–
–
–
–
–
88.3
–
89.3
–
86.0
–
–
–
–
87.4
–

RMPSNet (Ours)

Vision Transformer

89.7

95.6

81.7

91.1

65.0

76.0

89.8

93.2

86.9

92.0

4.2. Implementation details

4.3. Performance evaluations and comparisons

The image and text encoders in the proposed RMPSNet both adopt
the pre-trained ViT-B/16 backbone from CLIP. Similar to CLIP-ReID,
RMPSNet is constructed based on the CLIP vision-language model
and trained in two stages. The training largely follows the CLIP-ReID
protocol, with input images resized to 256 × 128 and a batch size of
64. In Stage 1, only learnable text prompts participate in the training.
The parameters of both the text encoder and the image encoder are
fixed. The Adam optimizer is used with an initial learning rate of 3.5 ×
10−4 , and the learning rate decays according to the cosine annealing
schedule. In Stage 2, the data augmentation operations include random
flipping, region priority erasing, and random cropping. The batch size is
set to 64, and there are four images per identity. We warm up the model
for 10 epochs, during which the learning rate linearly increases from
5 × 10−7 to 5 × 10−6 . Afterwards, the learning rate is reduced by a factor
of 0.1 at the 30th and 50th epochs. All datasets in our experiment adopt
the same loss weight configuration and do not rely on hyperparameter
tuning specific to a particular dataset. Specifically, in Stage 1, we set
𝜆𝑡2𝑡 = 0.8. In Stage 2, we use 𝜆𝑛𝑜𝑖𝑠𝑒 = 0.1, 𝜆𝑒𝑟𝑎𝑠𝑒 = 0.1, 𝜆𝑡𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚 =
0.15, 𝜆1 = 0.25, and 𝜆2 = 0.9. Among them, 𝜆1 follows the original
setting of the baseline [5], while the remaining weights are consistently
adopted to balance the contributions of the enhanced branches and
the main identification objective. In addition, we follow the division
rules provided in Market-1501 [2], DukeMTMC-ReID [25], OccludedDuke [3], Occluded-ReID [26], and Partial-ReID [27] datasets. And
following the baseline model [5], we use a fixed random seed of 1234
to control model initialization, data loading, and data augmentation. In
the ReID task [5,24,26], two standard metrics are commonly used to
evaluate performance: Cumulative Matching Characteristics (CMC) at
Rank-1 and mean Average Precision (mAP). Therefore, we also strictly
adhere to these metrics in our experiments. Additionally, the evaluation
is conducted under a single-query setting without using re-ranking
methods [28].

In this section, we comprehensively evaluate RMPSNet on five
widely used public person ReID datasets and comparing it with the
competitors. The evaluation covers both general (Market-1501 [2],
DukeMTMC-ReID [25]) and occlusion-specific (Occluded-Duke [3],
Occluded-ReID [3], and Partial-ReID [27]) scenarios. The results are
shown in Table 1.
On the general benchmarks, including Market-1501 and DukeMTMCReID, RMPSNet demonstrates strong overall performance despite being designed primarily for occlusion scenarios. Specifically, RMPSNet
achieved a Rank-1 accuracy of 95.6% and an mAP of 89.7% on Market1501, slightly inferior to the current best method THCB, which has
a Rank-1 accuracy of 96.2% and an mAP of 90.6%. On DukeMTMCReID, RMPSNet attains a Rank-1 accuracy of 91.1% and an mAP of
81.7%, which is close to PADE and slightly higher than other strong
baselines such as MTIPE and CLIP-ReID. Although it is slightly lower
than the THCB model with the highest performance on general datasets,
the THCB model introduces an additional attribute model, while the
RMPSNet does not rely on additional clues. At the same time, the RMPSNet method has better performance in occluded environments. These
results confirm that RMPSNet maintains strong generalization ability
and does not significant compromise its performance on standard ReID
tasks, highlighting its versatility across different scenarios.
On the occlusion specific benchmarks, including Occluded-Duke,
Occluded-ReID, and Partial-ReID, RMPSNet demonstrates clear and
substantial performance advantages over existing methods. For
Occluded-Duke, RMPSNet achieves a Rank-1 accuracy of 76.0% and an
mAP of 65.0%, outperforming the best existing method KPR by 0.2% in
Rank-1 accuracy and 1.6% in mAP. Compared to ProFD, which achieves
70.8% in Rank-1 accuracy and 62.8% in mAP, RMPSNet shows a notable improvement of 5.2% and 2.2%, respectively. This improvement
indicates that RMPSNet effectively addresses the severe lower-body
occlusions that dominate the Occluded-Duke dataset, benefiting from
6

Pattern Recognition 179 (2026) 113919

Z. Gao et al.

the RPE module’s region-prioritized erasure strategy that simulates
realistic occlusion distributions. On Occluded-ReID, RMPSNet achieves
a Rank-1 accuracy of 93.2% and an mAP of 89.8%, surpassing ProFD by
2.1% in Rank-1 accuracy and 1.3% in mAP. Compared to other strong
baselines, such as ADP and FCF, RMPSNet exhibits improvements of
approximately 4.0% to 9.6% in Rank-1 accuracy and around 4.1% to
4.7% in mAP. The high performance on this dataset, which includes
severe and varied occlusion conditions without corresponding training
data, highlights the effectiveness of the DMPA module in facilitating
fine-grained cross-modal alignment and semantic reconstruction. On
Partial-ReID, RMPSNet achieves a Rank-1 accuracy of 92.0% and an
mAP of 86.9%, outperforming PADE by 2.7% in Rank-1 accuracy and
2.1% in mAP, and surpassing THCB by 4.6% in R1 accuracy and 2.7%
in mAP. These results suggest that RMPSNet not only excels in handling
heavy occlusion but also maintains high robustness when only partial
body regions are visible, leveraging the MDO module’s multi-branch
adversarial feature enhancement to strengthen spatial consistency and
global discriminability.
The substantial performance gains of RMPSNet can be attributed to
several key factors. First, the DMPA module enables fine-grained crossmodal alignment, allowing textual semantics to effectively complement
visual features even under heavy occlusion. Second, the RPE module
introduces realistic occlusion simulation that better matches real-world
occlusion distributions, reducing the gap between training and deployment environments. Third, the MDO module enhances feature
robustness through controlled adversarial perturbations, which balances discriminative capability and generalization. Finally, the stepwise
dual-loop optimization strategy effectively mitigates conflicts between
robustness and discriminative objectives, enabling stable and efficient
joint learning. In summary, RMPSNet not only achieves superior results
on challenging occlusion-specific datasets but also maintains competitive performance on general person ReID benchmarks. These results
clearly demonstrate its strong potential as a practical and versatile
solution for real-world pedestrian recognition applications involving
complex occlusions.
We further note that the three modules may introduce potential
optimization tension when jointly trained. Specifically, DMPA encourages semantic invariance on the text side by stabilizing ID-conditioned
prompt representations under missing semantic components, whereas
RPE increases the realism of occlusion exposure by deliberately removing visible cues at the input level. Meanwhile, MDO enforces feature
robustness and distribution consistency under multiple perturbations,
which may, in principle, conflict with fine-grained identity discrimination if the robustness constraint becomes overly strong. Therefore,
the potential conflict is not limited to a generic trade-off between
robustness and discrimination, but also involves the balance among
semantic invariance, realistic occlusion simulation, and discriminative
feature specificity. To this end, we alleviate this problem through a twostage optimization strategy. First, we learn stable cross-modal semantic
prototypes through DMPA. Then, we introduce RPE and MDO for
data-level and feature-level robustness enhancement, thereby reducing
direct gradient competition among the three objectives. This conflict
is alleviated through phased decoupling and incremental enhancement, thereby ensuring the stability of the overall training and the
improvement of the final performance.

Table 2
Effectiveness of the DMPA modules.
Methods

Datasets

Baseline
Baseline+DMPA
Baseline+DMPA+RPE
Baseline+DMPA+RPE+MDO (RMPSNet)
RMPSNet-DMPA

Occluded-Duke

Occluded-ReID

mAP

R1

mAP

R1

59.5
60.9
63.8
65.0
64.0

67.1
68.6
73.3
76.0
74.2

87.2
89.0
89.0
89.8
89.2

90.5
92.7
93.0
93.2
92.9

Table 3
Benefits of the RPE modules.
Methods

Datasets
Occluded-Duke

Baseline
Baseline+RPE
RMPSNet-RPE
RMPSNet

Occluded-ReID

mAP

R1

mAP

R1

59.5
63.7
63.2
65.0

67.1
73.2
73.8
76.0

87.2
88.4
89.0
89.8

90.5
92.1
92.7
93.2

5.1. The effectiveness of the DMPA module
To verify the effectiveness of the DMPA module, we conducted
ablation studies on the Occluded-Duke and Occluded-ReID datasets. As
presented in Table 2, the baseline model, achieves a Rank-1 accuracy
of 67.1% and an mAP of 59.5% on Occluded-Duke, and 90.5% in
Rank-1 accuracy and 87.2% in mAP on Occluded-ReID. When the
DMPA module is integrated into the baseline, the combined model
is called ‘Baseline+DMPA’, significant improvements have been observed. Specifically, on Occluded-Duke, the Rank-1 accuracy improves
by 1.5% to reach 68.6%, and the mAP increases by 1.4%, achieving 60.9%. On Occluded-ReID, the Rank-1 accuracy rises by 2.2%
to 92.7%, while the mAP increases by 1.8%, reaching 89.0%. These
results indicate that even when used alone, the DMPA module can
effectively enhance visual-language feature alignment by providing
additional semantic cues, thus improving discriminability in occluded
scenarios. To conduct a more in-depth evaluation of the contribution of
DMPA within the entire RMPSNet framework, we remove this module
from the complete RMPSNet and named it ‘RMPSNet-DMPA’. Compared
with the complete RMPSNet, excluding the DMPA module leads to
a Rank-1 accuracy drop of 1.8% and an mAP decrease of 1.0% on
Occluded-Duke. Similarly, on Occluded-ReID, there is a reduction of
0.3% in Rank-1 accuracy and 0.6% in mAP. This consistent performance degradation highlights the essential role of DMPA in enhancing
fine-grained cross-modal alignment, especially under severe occlusion
conditions. By leveraging partial textual descriptions and enforcing
contrastive learning constraints, DMPA effectively compensates for visual information loss and guides the model to focus on identity-relevant
semantic regions. Furthermore, it enhances the model’s robustness
against occlusion-induced feature perturbations, ensuring more stable
and discriminative representations.
5.2. The benefits of the RPE module

5. Ablation study
We also conducted a comprehensive ablation study on the effectiveness of the RPE module. The results are summarized in Table 3.
The baseline model, which includes standard random erasure data
augmentation [9], achieves a Rank-1 accuracy of 67.1% and an mAP
of 59.5% on Occluded-Duke, as well as 90.5% Rank-1 and 87.2%
mAP on Occluded-ReID. When adding the RPE module to the baseline, as was noted in ‘Baseline+RPE’, we observe substantial performance improvements. On Occluded-Duke, the Rank-1 accuracy increases by 6.1% to reach 73.2%, and the mAP improves by 4.2%

An ablation study was performed using RMPSNet to analyze the
contribution of each component. In our experiments, two occluded
datasets, namely Occluded-Duke and Occluded-ReID, were adopted for
evaluation. Specifically, six aspects were considered: (1) the effectiveness of the DMPA module, (2) the benefits of the RPE module, (3)
the effectiveness of the MDO module, (4) complexity analysis, (5)
sensitivity analysis, and (6) visualization. In the following, we discuss
these aspects separately.
7

Pattern Recognition 179 (2026) 113919

Z. Gao et al.

Table 4
Effectiveness of the MDO modules.
Methods

Datasets
Occluded-Duke
mAP

Baseline
Baseline+MDO (only ME)
Baseline+MDO
RMPSNet-MDO
RMPSNet

Table 5
Complexity analysis of different occluded person ReID methods. ‘FLOPs’ represents the number of floating-point operations, and ‘Params’ represents the
number of parameters.

59.5
59.5
61.9
63.8
65.0

Occluded-ReID
R1
67.1
67.1
71.0
73.3
76.0

mAP
87.2
88.6
88.7
89.0
89.8

R1
90.5
92.5
92.7
93.0
93.2

Method

FLOPs (×1012 )

Params (×107 )

mAP

R1

ISP [53]
TransReID [24]
PFD [35]
RMPSNet

1.63
5.77
2.89
5.90

3.14
9.27
9.41
11.45

52.3%
59.2%
61.8%
65.0%

62.8%
66.4%
69.5%
76.0%

and 1.2% in mAP on Occluded-Duke, and decreases of 0.2% in Rank-1
accuracy and 0.8% in mAP on Occluded-ReID. These declines confirm
the crucial role of MDO in maintaining the model’s strong occlusion
robustness and discriminative capacity. The improvements brought by
the MDO module can be attributed to its design that introduces progressive adversarial perturbations while simultaneously decoupling the
classification objective from complex feature fusion tasks. This stepwise
optimization process mitigates conflicts between feature enhancement
and identity discrimination, resulting in more stable and robust feature
representations.

to 63.7%. On Occluded-ReID, the Rank-1 accuracy rises by 1.6% to
92.1%, and the mAP increases by 1.2% to 88.4%. Similarly, we added
the RPE module to ‘Baseline+DMPA’ to obtain the combined model
‘Baseline+DMPA+RPE’. As shown in Table 2, the performance of the
model has been further improved. These consistent gains indicate that
incorporating a region-prioritized occlusion prior is more effective than
uniform random erasing in our setting. This suggests that explicitly
modeling the spatial prior of real occlusions can reduce the mismatch
between synthetic erasing and real-world occlusion patterns. Further,
to assess the specific contribution of the RPE module within the full
RMPSNet framework, we evaluate the RMPSNet variant without RPE,
denoted as ‘RMPSNet-RPE’. Compared to the complete RMPSNet, removing the RPE module results in a performance drop of 2.2% in
Rank-1 accuracy and 1.8% in mAP on Occluded-Duke. Similarly, on
Occluded-ReID, excluding RPE leads to a reduction of 0.5% in Rank1 accuracy and 0.8% in mAP. These results highlight the important
role of RPE in further strengthening occlusion robustness even when
other enhancements are present. The RPE module gives higher masking
priority to regions that are more likely to be occluded in real-world
scenarios. In this way, it introduces the spatial prior of real occlusions
into the training process and helps reduce the distribution mismatch
between synthetic augmentation and realistic occlusion patterns.

5.4. Complexity analysis
To evaluate the practical feasibility of different occluded person
ReID methods, we compare their computational complexity and model
size, as summarized in Table 5. While lightweight models such as
ISP achieve the lowest FLOPs and parameter counts, they suffer from
significant performance degradation under occlusion, limiting their
applicability in challenging scenarios. Methods like TransReID and PFD
provide a better trade-off between complexity and accuracy but still
exhibit notable gaps in occlusion robustness. In contrast, RMPSNet,
despite having slightly higher computational cost and parameter count,
achieves the highest overall accuracy, with an mAP of 65.0% and a
Rank-1 accuracy of 76.0%. These results demonstrate that our design
effectively balances model complexity and recognition performance.
The strategic incorporation of occlusion-specific modules and crossmodal enhancement enables RMPSNet to achieve superior robustness
without incurring excessive redundancy. RMPSNet justifies its moderate increase in complexity by providing significant gains in occlusion
handling and identity discrimination, highlighting its strong potential for practical real-world deployment. From a practical deployment
perspective, the increased computational cost of RMPSNet mainly manifests as higher memory consumption and inference overhead compared
with lightweight occluded ReID models. Nevertheless, this additional
cost is compensated by clear performance gains on occlusion-specific
benchmarks, where recognition reliability under severe partial visibility is usually more critical than strict model compactness. Therefore,
RMPSNet is more suitable for surveillance and retrieval scenarios that
prioritize robustness in complex occlusion conditions, whereas lighter
models may remain preferable in highly resource-constrained environments. The practical impact of the increased complexity is a trade-off
between inference efficiency and occlusion robustness, and our results
show that this trade-off is justified for challenging real-world occluded
person ReID applications.

5.3. The effectiveness of the MDO module
The MDO module in RMPSNet is designed to incorporate two key
components: multi-enhancement strategies and a step-by-step adversarial optimization strategy. To evaluate their effectiveness, we conducted
comprehensive ablation studies on the Occluded-Duke and OccludedReID datasets, as summarized in Table 4. When the introduced MDO
module contains only multiple enhancement strategies, we name it
Baseline+MDO (only ME). It is observed that on the Occluded-Duke
dataset, the Rank-1 accuracy and mAP remain unchanged, but there
is a significant improvement on the Occluded-ReID dataset, where the
Rank-1 accuracy increases to 92.5% and the mAP rises to 88.6%. These
results indicate that introducing diverse feature enhancement methods
such as erasure, noise, and geometric transformations can improve the
robustness of the model under specific occlusion patterns to a certain extent. When further combined with the step-by-step adversarial
optimization strategy to form a complete MDO module, it is called
‘Baseline+MDO’, and an additional improvement in performance is
observed. On Occluded-Duke, the Rank-1 accuracy improves to 71.0%,
and the mAP increases to 61.9%, showing a cumulative improvement
of 3.9% in Rank-1 accuracy and 2.4% in mAP over the Baseline. On
Occluded-ReID, the Rank-1 accuracy reaches 92.7%, and the mAP
further improves to 88.7%, demonstrating the synergistic effect of adversarial optimization in conjunction with feature enhancement. These
results validate the effectiveness of the proposed step-by-step adversarial optimization, which decouples the classification task from the
complex fusion task through progressive optimization and enhances
model robustness via adversarial loss. To assess the contribution of
the MDO module within the full RMPSNet framework, we evaluate
the variant without MDO, denoted as ‘RMPSNet-MDO’. Removing MDO
from RMPSNet leads to a performance drop of 2.7% in Rank-1 accuracy

5.5. Sensitivity analysis
(1) To further evaluate the robustness of RMPSNet in terms of
hyperparameter selection, we conducted a sensitivity analysis on key
weight parameters using the Occluded-ReID dataset, as shown in Fig.
3. Specifically, we changed each loss weight parameter individually
while keeping other parameters unchanged. The results indicate that
the performance curve remains relatively stable within a reasonable
range, which suggests that RMPSNet is stable within a reasonable
weight range. In addition, our model achieved excellent performance
8


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Robust mixed-degradation person Re-identification via structural consistency distillation.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 152ms:
sed as the representation space. On these
features, the proposed Structural Consistency Distillation (SCD) module
performs Fused Gromov-Wasserstein (FGW)-based clean-to-degraded
alignment, while the Clean Feature Restoration (CFR) module applies
elastic weight consolidation (EWC) regularization (see Sections 3.1,
3.2). The clean-image branch is introduced only during training as
an auxiliary reference, providing more stable structural and semantic
guidance for the degraded-image branch, thereby enhancing its robustness under mixed degradation conditions. The ‘‘Encoder’’ refers
3

Pattern Recognition 179 (2026) 113938

S. Zhao et al.

Fig. 2. Overview of the proposed MDCD framework. The framework includes two modules: (1) the SCD module, which aligns cross-domain feature distributions
using Wasserstein (WD) and Gromov-Wasserstein (GWD) distances while preserving intra-domain structure; and (2) the CFR module, which introduces a
regularization term to constrain variations in parameters critical for clean-image recognition.

Fig. 3. Comparison of Wasserstein Distance (WD) and Gromov-Wasserstein Distance (GWD). WD (left) aligns clean (green) and degraded (yellow) domains by
transporting samples based on pairwise cost 𝐶(𝑥1 , 𝑦1 ), reducing the global distribution gap. GWD (right) preserves intra-domain topology by measuring structural
discrepancy 𝐿(𝑥1 , 𝑥2 , 𝑦1 , 𝑦2 ) between pairwise relations. Squares and circles represent different IDs (ID1 and ID2), and triangles indicate structural relations. (For
interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)

to the feature extractor of the underlying Re-ID model. For instance,
when instantiated with TransReID [46], the Encoder corresponds to its
feature extraction module, and MDCD operates on the resulting features
before BNNeck. During inference, only the degraded-image branch is
used for retrieval, which aligns with the degraded-input target scenario
and introduces no additional computational branches at test time.

and intra-domain structural consistency within a unified optimal transport objective. This formulation produces representations that are both
globally aligned and structurally consistent across domains.
Given a mini-batch of size 𝐵, let {𝑥𝑖 }𝐵
and {𝑦𝑗 }𝐵
denote feature
𝑖=1
𝑗=1
vectors from the clean and degraded branches, respectively. Their
empirical measures are defined as:

3.1. Structural consistency distillation module

𝜇=

Aligning clean (green) and degraded (yellow) feature representations involves two complementary challenges: correcting global domain
shifts and preserving ID-wise geometric structure. As illustrated in
Fig. 3, the Wasserstein Distance (WD) aligns two domains by transporting samples based on cross-domain costs, thereby reducing the
global distribution discrepancy. However, WD does not account for
relational structures within each domain. Under mixed degradations,
samples belonging to the same ID may vary significantly in appearance,
which could lead WD to transport degraded samples toward visually
similar but incorrect IDs, resulting in ID confusion. In contrast, the
Gromov-Wasserstein Distance (GWD) aligns relational structures between domains by matching pairwise relations such as (𝑥1 , 𝑥2 ) and
(𝑦1 , 𝑦2 ), thus preserving ID-wise geometry. However, since GWD focuses
on structural relations rather than absolute feature locations, the global
shift between domains may remain unresolved, and corresponding
samples such as 𝑥1 and 𝑦1 can still be distant in the feature space.
To address both limitations, we adopt the Fused Gromov-Wasserstein
Distance (FGW), which jointly models cross-domain feature alignment

where 𝛿𝑥𝑖 and 𝛿𝑦𝑗 denote Dirac measures centered at 𝑥𝑖 and 𝑦𝑗 . We
define the cross-domain cost 𝑀𝑥𝑦 (𝑖, 𝑗) = ‖𝑥𝑖 −𝑦𝑗 ‖2 and the intra-domain
costs 𝑀𝑥 (𝑖, 𝑗) = ‖𝑥𝑖 − 𝑥𝑗 ‖2 and 𝑀𝑦 (𝑙, 𝑘) = ‖𝑦𝑙 − 𝑦𝑘 ‖2 . A transport matrix:
{
}
𝑇 ∈ 𝛱 (𝜇, 𝜈) = 𝑇 ∈ R𝐵×𝐵
∣ 𝑇 𝟏 = 𝜇, 𝑇 ⊤ 𝟏 = 𝜈 ,
(2)
+

1 ∑
𝛿 ,
𝐵 𝑖=1 𝑥𝑖
𝐵

1 ∑
𝛿 ,
𝐵 𝑗=1 𝑦𝑗
𝐵

𝜈=

(1)

couples the two distributions under the marginal constraints. For a
given transport plan 𝑇 , the structural mismatch term is defined as:
∑|
|2
𝐴𝑖𝑗 (𝑇 ) =
(3)
|𝑀𝑥 (𝑖, 𝑗) − 𝑀𝑦 (𝑙, 𝑘)| 𝑇𝑙𝑘 .
|
|
𝑙,𝑘

The FGW objective is formulated as:
∑
[
]
FGW (𝜇, 𝜈) = min
𝑇𝑖𝑗 𝛼𝑀𝑥𝑦 (𝑖, 𝑗) + (1 − 𝛼) 𝐴𝑖𝑗 (𝑇 ) ,
𝑇 ∈𝛱(𝜇,𝜈)

(4)

𝑖,𝑗

where 𝛼 ∈ [0, 1] balances global alignment (𝛼 = 1 recovers WD) and
structural consistency (𝛼 = 0 recovers GWD).
To integrate FGW into the training process, we adopt the entropyregularized Sinkhorn algorithm [20]. The cost matrix and kernel are
4

Pattern Recognition 179 (2026) 113938

S. Zhao et al.

4. Experimental results

defined as:
𝐶𝑖𝑗 = 𝛼𝑀𝑥𝑦 (𝑖, 𝑗) + (1 − 𝛼) 𝐴𝑖𝑗 (𝑇 ) ,
(
)
𝐶𝑖𝑗
𝐾𝑖𝑗 = exp −
, 𝑖, 𝑗 = 1, … , 𝐵,
𝜖

(5)
4.1. Dataset and implementation details
(6)
4.1.1. Degradation protocol
Due to the lack of benchmarks for mixed-degradation scenarios,
we construct two synthetic datasets, Mixed-Market1501 and MixedMSMT17. Based on Market1501 [21] and MSMT17 [22], we follow the
corruption procedure in [23], where each image is degraded by one of
four degradation types (fog, rain, snow, or brightness). For each image,
the degradation type is sampled uniformly at random, and the severity
level is drawn from five predefined levels. This procedure generates
diverse visual conditions for the training, query, and gallery sets while
preserving the original ID labels and bounding boxes.

where 𝜖 > 0 is a regularization parameter. Given target marginals
𝜇, 𝜈 ∈ R𝐵 (initialized as uniform distributions), the Sinkhorn algorithm
iteratively updates scaling vectors 𝑢, 𝑣 ∈ R𝐵 :
𝜇
𝜈
𝑢←
, 𝑣← ⊤ ,
(7)
𝐾𝑣
𝐾 𝑢
until the marginal constraints 𝑇 𝟏 = 𝜇 and 𝑇 ⊤ 𝟏 = 𝜈 are satisfied. The
optimal transport matrix is then obtained as:
𝑇 ⋆ = diag (𝑢) 𝐾diag (𝑣) .

(8)

With 𝑇 ⋆ , the FGW loss used in training becomes:
FGW =

∑

[
]
𝑇𝑖𝑗⋆ 𝛼𝑀𝑥𝑦 (𝑖, 𝑗) + (1 − 𝛼) 𝐴𝑖𝑗 (𝑇 ⋆ ) ,

(9)

4.1.2. Statistical summary
As illustrated in Fig. 4, each degradation type and severity level is
uniformly sampled, ensuring balanced coverage across corruption factors. This sampling strategy provides a controlled and diverse testbed
for evaluating model robustness under heterogeneous degradation conditions.

𝑖,𝑗

which simultaneously promotes global distribution alignment and
structural consistency between clean and degraded feature spaces.
3.2. Clean feature restoration module
Degraded images often contain noise, blur, and other artifacts that
obscure fine-grained discriminative cues present in clean images. Although training with degraded data improves robustness, it may also
cause the model to forget discriminative knowledge learned from clean
samples. To address this issue, we introduce the Clean Feature Restoration (CFR) module based on elastic weight consolidation (EWC), a
Fisher-information-based regularization technique [47]. By penalizing
parameter changes that significantly affect clean-image performance,
CFR alleviates catastrophic forgetting.
Specifically, the importance of each parameter 𝜃𝑖 is estimated using
clean-domain samples:
[(
)2 ]
𝜕
𝐹𝑖 = E(𝐼,cls)∼𝐷clean
log 𝑝 (cls ∣ 𝐼, 𝜃)
,
(10)
𝜕𝜃𝑖

4.1.3. Visual examples
Fig. 5 presents examples of synthetic degradations with five severity
levels for each weather factor: fog, rain, snow, and brightness. Each
row corresponds to one degradation type, with degradation severity
increasing from level 1 to level 5. The rightmost column shows the
original clean image for reference, illustrating the visual diversity introduced by our protocol. Fig. 5(b) reports the corresponding quantitative
metrics (SSIM and PSNR), which reflect the severity of degradation
across different levels.
4.1.4. Evaluation protocol and metrics
Following [23], we evaluate all methods under four settings: (1)
Clean Evaluation (CE): both query and gallery are clean; (2) Degraded
Evaluation (DE): both query and gallery are degraded; (3) Degraded
Query (DQ): degraded queries with a clean gallery; (4) Degraded Gallery
(DG): clean queries with a degraded gallery. We report mean average
precision (mAP) and cumulative matching characteristic (CMC) metrics
for all experiments.

where 𝑝(cls ∣ 𝐼, 𝜃) denotes the predicted probability of class cls for input
image 𝐼. A larger 𝐹𝑖 indicates that parameter 𝜃𝑖 is more important for
maintaining clean-image performance.
During training on degraded data, an EWC regularization term is
introduced to penalize deviations from the clean-trained parameters 𝜃𝑖∗ :
∑ (
)2
EWC =
𝐹𝑖 𝜃𝑖∗ − 𝜃𝑖 ,
(11)
𝑖

4.1.5. Implementation details
All images are resized to 256 × 128. Data augmentation includes
random horizontal flipping, padding, random cropping, and random
erasing. We use a batch size of 64 with four images per ID. The
optimizer, learning schedule, and training strategy follow the respective
backbone implementations. All experiments are conducted on a single
NVIDIA GeForce RTX 4090 GPU. For the proposed objectives, unless
otherwise specified, the balance coefficient in FGW is set to 𝛼 = 0.1,
the entropy regularization parameter to 𝜖 = 0.8, and the loss weights to
𝜆1 = 1 and 𝜆2 = 10, respectively. Further discussion on the sensitivity
of these hyperparameters is provided in Section 4.6.

where 𝜃𝑖∗ denotes parameters obtained from clean-image training and 𝜃𝑖
denotes the current parameters. This regularization preserves discriminative capability on clean images while allowing the model to adapt
to mixed-degradation conditions.
3.3. Overall architecture
Both modules operate at the feature level without modifying the
backbone architecture, making MDCD inherently backbone-agnostic
and easily compatible with existing Re-ID models. The training objective combines the original Re-ID loss ReID (e.g., identification and
triplet losses) with the proposed objectives, including the FGW alignment loss FGW from the SCD module and the EWC regularization loss
EWC from the CFR module. The overall training objective is formulated
as:
all = ReID + 𝜆1 FGW + 𝜆2 EWC ,

4.2. Comparison with state-of-the-art
Tables 2 and 3 compare the proposed MDCD framework with representative Re-ID methods on Mixed-Market1501 and Mixed-MSMT17.
Existing approaches typically achieve strong performance under CE but
exhibit substantial degradation in DE, DQ, and DG, highlighting their
limited robustness under mixed-degradation conditions.

(12)

where 𝜆1 and 𝜆2 balance the contributions of structural alignment and
feature restoration.
5

Pattern Recognition 179 (2026) 113938

S. Zhao et al.

Fig. 4. Distribution of degradation types and severity levels in the mixed-degradation benchmarks. The left and right panels represent Mixed-Market1501 and
Mixed-MSMT17, respectively. Numbers above each bar indicate the total number of images for each degradation type.

Fig. 5. Examples of synthetic weather-induced degradations. Fog, rain, snow, and brightness degradations are shown across severity levels 1–5. GT denotes the
original clean image. (b) Corresponding SSIM and PSNR values for each degradation type.
Table 2
Performance comparison on Mixed-Market1501. Performance comparison under four evaluation settings:
CE, DE, DQ, and DG.
Method

a

Venue

CE

DE

DQ

DG

mAP

R1

mAP

R1

mAP

R1

mAP

R1

AGW [48]
CIL [23]
TransReID [46]
TransReID-SSL [49]
PHA [50]
CLIP-ReID [51]
DenoiseRep [52]
FusionReID [53]

TPAMI’21
NeurIPS’21
ICCV’21
arXiv’21
CVPR’23
AAAI’23
NeurIPS’24
TITS’25

88.3
84.0
89.0
91.3
87.3
89.6
91.8
91.6

95.3
93.3
95.2
95.8
94.4
95.5
96.4
96.1

11.5
21.5
15.6
22.4
13.4
25.7
21.3
15.2

39.4
54.1
47.4
53.8
42.1
54.2
52.6
49.3

28.4
42.3
34.5
42.1
30.3
47.9
40.7
36.3

34.0
51.1
37.6
45.7
32.9
53.1
43.9
39.8

34.9
41.8
36.8
44.2
33.8
48.6
42.3
36.3

79.5
79.9
79.0
84.5
77.6
86.7
84.4
81.5

AGW [48]a
CIL [23]a
TransReID [46]a
TransReID-SSL [49]a
PHA [50]a
CLIP-ReID [51]a
DenoiseRep [52]a
FusionReID [53]a
MDCD w/ CLIP-ReIDa
MDCD w/ TransReIDa
MDCD w/ TransReID-SSLa

TPAMI’21
NeurIPS’21
ICCV’21
arXiv’21
CVPR’23
AAAI’23
NeurIPS’24
TITS’25
(Ours)
(Ours)
(Ours)

79.1
75.0
85.5
87.3
81.1
85.5
89.6
86.2
88.4
88.0
89.5

91.1
89.1
93.6
93.9
91.6
93.8
95.5
93.8
94.6
94.6
95.4

68.1
62.1
73.6
76.4
68.0
71.4
76.9
76.3
75.1
76.2
78.8

86.2
80.2
88.1
88.3
84.1
85.9
88.6
89.5
88.3
88.8
89.8

73.9
68.1
79.6
81.6
74.5
78.4
83.1
81.6
81.8
82.1
84.1

88.3
81.9
89.1
89.3
85.9
88.1
89.9
91.0
89.6
90.4
91.2

73.2
68.3
79.3
82.0
74.3
78.2
83.3
80.9
81.5
82.0
84.2

90.0
86.5
93.0
92.9
89.7
92.5
94.4
93.1
93.1
93.3
94.3

Models trained on degraded data, with the best results in bold.

6

Pattern Recognition 179 (2026) 113938

S. Zhao et al.

Table 3
Performance comparison on Mixed-MSMT17. Comparison of performance under CE, DE, DQ, and DG.
Method

a

Venue

CE

DE

DQ

DG

mAP

R1

mAP

R1

mAP

R1

mAP

R1

AGW [48]
CIL [23]
TransReID [46]
TransReID-SSL [49]
PHA [50]
CLIP-ReID [51]
DenoiseRep [52]
FusionReID [53]

TPAMI’21
NeurIPS’21
ICCV’21
arXiv’21
CVPR’23
AAAI’23
NeurIPS’24
TITS’25

55.3
52.6
67.8
66.2
65.1
73.4
67.1
69.7

78.8
76.4
85.4
84.8
82.4
88.8
85.3
86.4

4.1
6.5
9.6
9.6
6.5
17.3
9.3
8.8

19.1
24.8
30.9
31.0
23.0
42.9
30.0
30.9

16.7
18.2
26.0
25.7
20.9
36.1
25.4
27.3

26.6
30.0
35.9
36.0
27.9
47.3
35.9
37.6

16.8
17.3
25.4
24.6
19.3
34.3
24.4
24.1

59.4
55.4
67.8
67.5
61.4
78.1
67.8
70.1

AGW [48]a
CIL [23]a
TransReID [46]a
TransReID-SSL [49]a
PHA [50]a
CLIP-ReID [51]a
DenoiseRep [52]a
FusionReID [53]a
MDCD w/ TransReIDa
MDCD w/ TransReID-SSLa
MDCD w/ CLIP-ReIDa

TPAMI’21
NeurIPS’21
ICCV’21
arXiv’21
CVPR’23
AAAI’23
NeurIPS’24
TITS’25
(Ours)
(Ours)
(Ours)

39.4
34.9
57.7
55.2
52.6
64.8
61.8
54.0
61.8
59.5
68.2

65.8
62.2
78.5
77.4
72.4
84.3
81.9
76.4
81.3
80.6
86.2

29.6
23.6
42.2
39.5
39.1
44.8
41.5
40.9
45.8
42.4
48.1

56.3
48.5
65.9
63.8
61.6
69.9
66.7
65.9
68.8
67.2
73.0

34.9
28.5
49.4
46.7
45.5
54.1
50.7
47.5
53.2
50.3
57.6

59.1
51.4
68.6
66.6
64.1
72.7
69.5
68.4
70.9
70.3
74.8

33.9
28.1
48.9
46.2
45.2
53.2
50.0
46.6
52.6
49.5
56.3

63.9
56.7
75.3
74.4
69.9
81.7
79.0
74.2
77.9
77.5
83.9

Models trained on degraded data, with the best results in bold.

Table 4
Performance comparison on Hazy-Market1501. Comparison under CE, DE, DQ, and DG. ‘‘-Dehaze’’
indicates methods combining dehazing with Re-ID. The best results are in bold.
Method

a

Venue

CE

DE

DQ

DG

mAP

R1

mAP

R1

mAP

R1

mAP

R1

ISM [13]
CIL [23]
TransReID [46]
TransReID-SSL [49]
SJDL [8]
RVSL [9]
DI-ReID++ [11]
DenoiseRep [52]

ICME’21
NeurIPS’21
ICCV’21
arXiv’21
AAAI’22
ECCV’22
IJCV’22
NeurIPS’24

76.8
76.5
89.0
91.3
64.4
71.6
–
91.8

91.2
91.2
95.2
95.8
84.5
87.3
–
96.4

65.6
73.9
27.9
30.2
64.5
61.7
65.0
28.2

83.6
88.2
52.2
53.9
84.3
80.6
82.7
50.6

64.5
74.2
14.3
16.6
63.2
54.5
–
16.6

79.0
88.2
14.3
16.9
83.3
73.9
–
16.8

64.4
73.9
23.5
24.4
60.8
54.7
–
23.9

83.9
88.7
49.0
51.4
81.7
73.6
–
51.0

TransReID-Dehaze [46]
TransReID-SSL-Dehaze [49]
DenoiseRep-Dehaze [52]

ICCV’21
arXiv’21
NeurIPS’24

89.0
91.3
91.8

95.2
95.8
96.4

72.9
77.4
78.2

87.0
89.1
89.9

77.4
80.8
81.1

85.9
86.7
87.2

78.6
82.2
82.5

92.4
93.4
94.1

TransReID [46]a
TransReID-SSL [49]a
DenoiseRep [52]a
MDCD w/ TransReIDa
MDCD w/ TransReID-SSLa

ICCV’21
arXiv’21
NeurIPS’24
(Ours)
(Ours)

83.7
87.0
87.4
85.8
88.3

93.3
93.8
94.5
94.0
94.5

81.4
84.8
84.9
83.1
86.3

91.7
92.8
92.8
92.3
93.5

82.1
85.3
85.0
83.8
86.6

91.7
92.8
92.1
92.3
93.4

81.7
85.4
84.2
83.1
86.3

92.3
93.4
92.8
92.5
93.8

Models trained on hazy images.

4.2.1. Mixed-market1501
As shown in Table 2, AGW [48], CIL [23], TransReID [46],
TransReID-SSL [49], PHA [50], CLIP-ReID [51], DenoiseRep [52],
and FusionReID [53] achieve strong results in CE but suffer significant drops in degraded settings. In contrast, MDCD consistently
improves its corresponding baselines across all evaluation protocols.
In particular, MDCD w/ TransReID-SSL achieves the best performance
in degraded scenarios, demonstrating the effectiveness of MDCD in
mitigating domain discrepancies while preserving discriminative ID
representations.

4.3. Generalization and robustness experiments
4.3.1. Effect of a single degradation factor
To further evaluate robustness, we conduct experiments on the
single-degradation dataset Hazy-Market1501. Table 4 compares MDCD
with haze-specific approaches (e.g., ISM [13], SJDL [8], RVSL [9], DIReID++ [11]) and general Re-ID models (TransReID [46], TransReIDSSL [49], and DenoiseRep [52]). Conventional methods exhibit substantial performance drops under degraded settings relative to CE,
while two-stage pipelines often suffer from ID information loss during
dehazing. In contrast, MDCD consistently improves DE, DQ, and DG
over its baselines, particularly when combined with TransReID-SSL.
These results indicate that MDCD maintains strong robustness under
both mixed and single degradation conditions.

4.2.2. Mixed-MSMT17
On the more challenging Mixed-MSMT17 (see Table 3), overall
CE performance is lower than that on Market1501 due to greater
scene complexity and stronger degradations. Among existing methods,
CLIP-ReID [51] performs best in degraded settings, benefiting from
strong semantic pretraining. Integrating MDCD into different backbones, including TransReID [46], TransReID-SSL [49], and CLIP-ReID,
consistently improves their performance. Notably, MDCD w/ CLIPReID achieves the best mAP and Rank-1 scores under degraded conditions, demonstrating the effectiveness and generality of the proposed
framework.

4.3.2. Effect of compound degradation scenarios
We further examine robustness under alternative synthesis settings.
Specifically, we explore: (1) applying multiple degradation types simultaneously within a single image; and (2) generating degradations
using alternative synthesis algorithms. These settings reduce the risk
7

Pattern Recognition 179 (2026) 113938

S. Zhao et al.

Table 5
Performance comparison under different degradation generation settings. Comparison across degradation
settings. ‘‘Multi (in-image fusion)’’ refers to images with multiple degradation types, and ‘‘Fog (alt.
synthesis)’’ and ‘‘Rain (alt. synthesis)’’ denote images generated with different synthesis methods from
the main benchmark. The best results are in bold.

a

Degradation
type

Method
mAP

R1

mAP

R1

mAP

R1

mAP

R1

Multi
(in-image
fusion)

TransReID-SSL [49]
TransReID-SSL [49]a
MDCD w/ TransReID-SSLa

91.3
65.7
81.0

95.8
81.7
91.4

2.5
24.4
30.6

9.2
42.3
49.8

8.3
38.7
48.5

8.6
48.5
57.2

11.0
39.2
49.2

49.1
68.9
81.7

Fog
(alt. syn.)

TransReID-SSL [49]
TransReID-SSL [49]a
MDCD w/ TransReID-SSLa

91.3
67.2
69.0

95.8
82.5
84.5

6.6
65.7
69.0

15.8
81.3
84.8

1.7
57.5
56.9

0.9
71.7
72.4

6.9
56.4
54.7

17.7
74.1
72.0

Rain
(alt. syn.)

TransReID-SSL [49]
TransReID-SSL [49]a
MDCD w/ TransReID-SSLa

91.3
88.2
90.4

95.8
94.5
95.6

64.5
84.3
86.3

81.8
92.2
93.8

58.9
86.0
88.2

62.0
92.8
94.2

62.0
86.2
88.3

84.9
94.2
95.1

CE

DE

DQ

DG

Denotes models trained on degraded images.

Table 6
Performance comparison on LT-Mix long-tailed mixed-degradation evaluation. Comparison under low (L), medium (M), and high (H) degrees
of long-tailed severity, generated by discretizing a half-normal prior with
𝜎 ∈ {2.0, 1.4, 0.9} for L, M, and H. Both query and gallery follow the same
long-tailed mixture distribution. The best results are in bold.
Method

TransReID-SSL [49]
TransReID-SSL [49]a
MDCD w/ TransReID-SSLa
a

L

M

H

mAP

R1

mAP

R1

mAP

R1

62.3
85.0
86.5

81.9
93.4
94.1

65.7
85.6
87.0

84.6
93.2
94.0

68.8
85.9
87.3

85.4
93.8
94.2

Indicates models trained on degraded images.

illustrated in Fig. 6, increasing the tail degree concentrates probability mass on low-severity samples while making high-severity samples
increasingly rare. Results in Table 6 show that MDCD consistently
outperforms strong baselines across all tail degrees, demonstrating its
robustness under realistic long-tailed degradation distributions.

Fig. 6. Severity distributions in LT-Mix evaluation settings. The curves represent Low, Medium, and High tail degrees, computed from the sampled metadata
of generated test sets. As the tail degree increases, probability mass shifts
toward low-severity levels, producing an imbalanced degradation distribution.

4.4. Ablation study
of overfitting to specific corruption patterns and provide a more challenging evaluation of generalization ability under unseen degradation
distributions.
As shown in Table 5, baseline models reveal substantial performance degradation under compounded or unseen corruption patterns.
In contrast, MDCD consistently improves performance across all settings. In the ‘‘Multi’’ and ‘‘Fog (alt. synthesis)’’ scenario, structural
alignment helps preserve ID relations across complex degradations. In
the alternative synthesis settings, feature-level alignment effectively
narrows the representation gap between clean and degraded domains.
These results demonstrate that MDCD performs well under the benchmark setting and generalizes effectively beyond the original benchmark
configuration.

To evaluate the contributions of the Structural Consistency Distillation (SCD) and Clean Feature Restoration (CFR) modules, we perform
ablation experiments on Mixed-Market1501. Table 7 compares various
feature alignment losses used in SCD, including MSE, Wasserstein
distance (WD), Gromov-Wasserstein distance (GWD), their naive combination (WD&GWD), and the proposed Fused Gromov-Wasserstein
(FGW) formulation.
4.4.1. Effectiveness of SCD
The results in Table 7 show that MSE reduces feature discrepancies but fails to capture structural relations or global domain shifts.
WD improves performance by aligning distributions, while GWD preserves local geometric relations but struggles to address global shifts.
A naive combination of WD and GWD yields limited improvements
due to uncoordinated optimization. In contrast, the unified FGW objective simultaneously enforces both distribution alignment and structural consistency, achieving the best performance across all evaluation
metrics.

4.3.3. Effect of long-tailed mixed-degradation distributions
Real surveillance environments often exhibit imbalanced degradation patterns, where mild degradations dominate while severe cases are
rare. To simulate this phenomenon, we construct an LT-Mix evaluation
set with long-tailed severity distributions. Specifically, for each degradation type 𝑑 ∈ {fog, rain, snow, brightness}, the degradation severity
𝑠 ∈ {1, … , 5} is sampled from a discretized half-normal prior:
𝑥 ∼ HalfNormal (𝜎) ,

𝑠 = min (5, ⌊𝑥⌋ + 1) .

4.4.2. Effectiveness of CFR
The CFR module mitigates catastrophic forgetting of clean-image
features during training with degraded data by constraining updates
to parameters critical for clean-image recognition. Experimental results
show that CFR not only improves performance in clean-image evaluations (CE) but also provides consistent gains in degraded scenarios by
stabilizing feature representations.

(13)

We vary 𝜎 to generate three tail degrees (Low, Medium, and High),
and apply the same distribution to both query and gallery images. As
8


codex
前两篇已经读到核心：一篇把遮挡解释成真实遮挡分布、文本语义缺失和增强后特征分布不一致三件事；另一篇把退化 ReID 从“单一退化域偏移”改写成“混合退化下的分布偏移加身份拓扑扭曲”。我会补一下被输出截断的中间方法细节，然后继续后四篇。
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
exec
/bin/zsh -lc "pdftotext -f 3 -l 5 'RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113919

Z. Gao et al.

leveraging large-scale image–text pairs. These models show strong
potential for occluded person ReID, but several limitations remain
in occlusion-sensitive scenarios. A major issue is the mismatch between coarse-grained global alignment and the fine-grained semantic degradation caused by occlusion. Methods such as CLIP-ReID [5]
align visual and textual features through prompt-based supervision
in the global embedding space, which improves identity-level semantic consistency but remains limited in associating textual cues with
localized occluded regions. ProFD [16] is a recent prompt-learningbased method for occluded person ReID under the vision-language pretraining paradigm, which enhances identity representation by leveraging learnable prompts to mitigate occlusion interference. Different from
ProFD [16], which mainly focuses on prompt-guided representation
learning within a CLIP-style backbone, our RMPSNet further addresses
the occlusion distribution mismatch through RPE and the robustness
gap between original and augmented features through MDO with stepwise adversarial training. Moreover, RMPSNet introduces DMPA with
an inter-text contrastive constraint to enhance the robustness of IDconditioned text prototypes under missing semantic components.
In addition, local semantic fusion methods often incur extra computational cost and may rely on external localization cues. For example, DenseCLIP [17] and QAConv-GLIP [18] improve region-level
understanding, but their performance can be affected by region localization quality and the additional overhead of detection-guided
processing. Generic prompt adaptation methods such as CoOp [19]
and Tip-Adapter [20] improve downstream adaptation, yet they are
primarily designed for closed-set classification and do not explicitly address the open-set, cross-view nature of person ReID. Therefore, visionlanguage ReID under occlusion still requires more effective fine-grained
semantic modeling and stronger robustness to occlusion-induced feature variation.

learning to robustness-oriented representation refinement. Specifically,
in the first stage, the parameters of the text encoder and image encoder
are frozen, and a contrastive loss is applied to guide the learning
of original prompts. Meanwhile, the prompts are enhanced through
the DMPA module, which then jointly guides the learning of original
prompts. In the second stage, input-level occlusion simulation and
feature-level robustness optimization are performed through RPE and
MDO, and an adversarial supervision loss function is constructed by
assigning positive and negative weights.
3.1. Dual-masked prompt augmentation (DMPA)
Although existing data augmentation methods provide some simulation of occluded scenes for the model. However, data augmentation
alone cannot fully address the issue of cross-modal semantic loss. Under
occlusion conditions, traditional CLIP-based ReID methods often struggle with insufficient feature disentanglement, unidirectional alignment
biases, and the risk of noise propagation. To overcome these challenges,
we propose dual-masked prompt augmentation (DMPA) module, which
enhances semantic robustness by leveraging adversarial masking and
contrastive reconstruction within the text space. Specifically, in the first
training stage, we apply double masking to the learnable text prompt
(e.g., A photo of a [𝑆]1 [𝑆]2 [𝑆]3 … [𝑆]𝑀 person.) as follows:
𝑦̂1 , 𝑦̂2 = Mask(𝑦, 𝛼),

(1)

where 𝑦̂ represents the masked text prompt, Mask is the masking
operation, which sets the value to 0, 𝑦 is the learnable parameters, and
𝛼 is the masking ratio, which we set to 0.5. Enhanced prompts derived
from the same text form positive pairs, while those from different texts
form negative pairs, with all tokens fixed during this process. The
original prompt is then encoded by the text encoder, and its feature
is aligned with the image feature via a cross-modal similarity score:

2.3. Adversarial optimization in re-identification

𝑠(𝑉𝑖 , 𝑇𝑖 ) = 𝑉𝑖 ⋅ 𝑇𝑖 = 𝑔𝑉 (img𝑖 ) ⋅ 𝑔𝑇 (text 𝑖 ),

Contrastive and adversarial optimization strategies have been widely
explored to improve robustness in person ReID. Existing methods
mainly focus on feature-level alignment, adversarial augmentation, and
collaborative optimization. Feature-level adversarial strategies reduce
camera or viewpoint discrepancies through techniques such as gradient
reversal or domain adversarial loss, but they are not specifically designed for occlusion and usually rely on global feature alignment, which
is limited in handling the semantic loss caused by local occlusions.
Recent studies have further introduced feature decoupling mechanisms
within attention modeling, such as occlusion-sensitive area suppression [21], to alleviate this problem, though at the cost of additional
complexity.
Sample-level adversarial strategies enhance robustness by generating perturbed training samples, such as AdvPattern [22]. However,
excessive perturbation may also damage identity-relevant cues and
reduce recognition accuracy. Multi-task collaborative learning has also
been explored to jointly optimize identity-related objectives [23], but
effective coordination between multiple enhanced learning branches
remains insufficiently studied. Overall, existing robust optimization
methods improve ReID performance to some extent, yet they still lack
a targeted mechanism for handling occlusion-induced feature variation and the distribution discrepancy between original and enhanced
representations.

(2)

where 𝑔𝑉 (⋅) and 𝑔𝑇 (⋅) represent the image encoder and text encoder
respectively. The image-to-text and text-to-image contrastive loss are
defined as:
exp(𝑠(𝑉𝑖 , 𝑇𝑖 ))
𝑖2𝑡 (𝑖) = − log ∑𝐵
,
(3)
𝑎=1 exp(𝑠(𝑉𝑖 , 𝑇𝑎 ))
exp(𝑠(𝑉𝑖 , 𝑇𝑖 ))
𝑡2𝑖 (𝑖) = − log ∑𝐵
,
(4)
𝑎=1 exp(𝑠(𝑉𝑎 , 𝑇𝑖 ))
where 𝑉 is the image feature embedding, 𝑇 is the text feature embedding, 𝑠(⋅, ⋅) represents the inner product similarity calculation, and 𝐵
denotes the batch size. Next, the original text prompts are enhanced
through prompt enhancements to obtain double-enhanced prompts.
The obtained double-enhanced prompts are also passed through the
text encoder to obtain the corresponding masked feature, which is
further refined using an inter-text contrastive loss. The contrastive loss
is defined as follows:
[
(
)
(
) ]
𝑁
exp 𝑠(𝑇𝑖 , 𝑇𝑖′ )
exp 𝑠(𝑇𝑖′ , 𝑇𝑖 )
1 ∑
𝑡2𝑡 = −
log ∑𝑁
(5)
(
) + log ∑𝑁
( ′
)
′
𝑁 𝑖=1
𝑗=1 exp 𝑠(𝑇𝑖 , 𝑇𝑗 )
𝑗=1 exp 𝑠(𝑇𝑖 , 𝑇𝑗 )
where 𝑇𝑖 and 𝑇𝑖′ denote two text features of the same identity obtained by applying the dual-masked prompt augmentation twice, and
𝑗 enumerates all candidate text features within the mini-batch. 𝑁
denotes the number of text prototypes involved in the batch for text-totext contrastive learning. 𝑠(⋅, ⋅) is the cosine similarity function in the
embedding space. The two log-terms implement a symmetric InfoNCE
objective. The total loss for the first stage is defined by:

3. The proposed approach
As shown in Fig. 2, RMPSNet is implemented through a two-stage
optimization strategy. This strategy enhances robustness against occlusions through multi-prompt learning and distribution-level adaptation,
and establishes a minimax game equilibrium in the feature space,
surpassing traditional single-stage adversarial training. The training
process can be understood as a transition from semantic prototype

𝐿𝑠𝑡𝑎𝑔𝑒1 = 𝑖2𝑡 + 𝑡2𝑖 + 𝜆𝑡2𝑡 𝑡2𝑡 ,

(6)

with 𝜆𝑡2𝑡 empirically set to 0.8. For computational efficiency, all image
features are extracted at the start of this stage. For a dataset with 𝐶
3

Pattern Recognition 179 (2026) 113919

Z. Gao et al.

Fig. 2. Pipeline of the proposed RMPSNet approach. It consists of the backbone, the DMPA module, the RPE module, and the MDO module, which is a two-stage
model. Specifically, in stage1, the prompt tokens are enhanced through the DMPA module and then jointly guide the learning of the original prompts. In stage2,
the RPE module is used to enhance the image input, which is then further processed by the MDO module to jointly optimize the image encoder.

identities, we store the corresponding 𝑇̂𝑦𝑖 features at the end of this
stage.
It is worth noting that the masking operation in DMPA is not
intended to establish a one-to-one correspondence between a masked
prompt token and a specific occluded image patch. Instead, we view
occlusion as a partial-information condition in the shared embedding
space. In CLIP-style ReID, the learnable text prompt acts as an identityconditioned semantic prototype for cross-modal alignment. By applying dual masking to the prompt tokens, DMPA introduces structured
missing-information perturbations on the text side and enforces the
resulting text features to remain semantically consistent through the
inter-text contrastive loss. This mechanism reduces the co-adaptation
of prompt dimensions and encourages the text prototype to remain
discriminative even when part of the semantic context is removed.
As a result, occluded images, which also contain incomplete identity
evidence, can still be aligned with a stable text-side semantic anchor,
thereby alleviating semantic degradation under occlusion.

define the aspect ratio, where ln(𝛼) ∼ 𝑈 [ln(0.3), ln(3.3)]. Based on 𝛼,
the height ℎ and width 𝑤 of the erasure region are calculated as ℎ =
⌊𝐴 ⋅ 𝛼⌋, 𝑤 = ⌊𝐴∕𝛼⌋, where 𝐴 denotes the dynamically sampled area
proportion. Finally, Gaussian noise is injected into the selected region
to simulate the occlusion effect, enhancing the model’s robustness to
real-world occluded scenarios. This targeted erasure strategy allows the
model to better focus on learning discriminative features from partially
visible body regions, thereby mitigating the impact of biased occlusion
distributions.
To further enhance occlusion robustness, the RPE strategy is also
applied at the feature erasure stage in a manner consistent with the
image-level prior. Specifically, after feature extraction, we perform
feature-level erasure on the patch-token representation while keeping
the CLS token unchanged. With a probability of 70%, the erased segment is sampled from the token sub-region corresponding to the lower
body, and with the remaining 30%, the erased segment is sampled
from the entire token sequence. The erased segment length is randomly
determined to simulate diverse occlusion scales, and the selected token
values are set to zero to mimic feature information loss caused by realistic occlusion. In this way, RPE is consistently imposed in both image
space and feature space, enabling the model to learn complementary
cues from partially visible regions and further reducing the distribution
mismatch between simulated augmentation and real-world occlusion.
This method effectively addresses the distribution mismatch problem
inherent in traditional random erasure and provides crucial technical
support for robust person ReID under occlusion. Together, they enable
the model to overcome the occlusion bottleneck and achieve more
robust identity representations.

3.2. Region Prioritized Erasure (RPE)
Traditional random erasure methods used in occluded person ReID
generally assume a spatially uniform distribution of occlusion regions,
which does not fully reflect real-world conditions. In practice, occlusions are more likely to occur in specific body regions, especially the
lower body, due to luggage, shopping carts, bicycles, or overlapping
pedestrians. Therefore, generic random erasing strategies may fail to
explicitly model the spatial prior of real occlusions, leading to a mismatch between synthetic occlusion patterns and real-world scenarios.
To reduce this distribution mismatch, we propose the region prioritized
erasure (RPE) module. It is designed to simulate realistic occlusion patterns by prioritizing lower-body regions during image augmentation.
Specifically, each input image first undergoes random augmentation
operations, including random horizontal flipping and random cropping.
Following these operations, a region erasure process is applied with a
probability of 0.5. If selected, there is a 70% probability of erasing a
region within the lower part of the image (i.e., the bottom half), and a
30% probability of performing uniform erasure across the entire image.
The erased region area is dynamically determined, ranging from 2%
to 40% of the total image area. A random parameter 𝛼 is sampled to

3.3. Multi-enhancement distribution optimization (MDO)
Traditional CLIP-based ReID methods [5,17] exhibit limitations
when addressing occluded scenarios, as they primarily rely on global
image–text alignment and lack explicit local feature enhancement
mechanisms. Moreover, there exists a distribution discrepancy between
the simulated augmented features and the original features, which
weakens generalization capability in real-world occlusion cases. To
mitigate these challenges, we propose the multi-enhancement distribution optimization (MDO) module, which integrates three distinct
4

Pattern Recognition 179 (2026) 113919

Z. Gao et al.

feature enhancement operations combined with a step-by-step adversarial optimization strategy to improve occlusion robustness. Note that
the noise branch in MDO is implemented as a simple feature-level perturbation strategy for robust optimization, rather than a diffusion-based
generative or denoising process.
Specifically, MDO extends the original feature representation by
incorporating three types of feature augmentation: region erasure, noise
injection, and geometric transformation. For region erasure, parts of
the feature map are randomly erased either within specific lower-body
regions or across the entire map to simulate partial feature loss under
occlusion. With a probability of 70%, the erasure targets only the lower
half of the pedestrian image, while the remaining 30% performs random erasure over the entire feature map. The erased region size is also
random to reflect diverse occlusion scales, thus enhancing the model’s
generalization ability. In the noise injection operation, Gaussian noise
with a fixed standard deviation of 0.2 is added to high-dimensional
features to alleviate the impact of noise disturbances commonly present
in occluded scenes. For geometric transformation, we introduce a local
feature replacement strategy to address viewpoint variations and shape
deformations. A segment within the feature sequence is replaced with
content from another region of the same sequence to mimic occlusions by external objects. Concretely, we operate on the patch-token
sequence (excluding the CLS token). We randomly sample a segment
length 𝐿 and two starting indices 𝑆𝑝 and 𝑆𝑞 , and then replace the token
segment [𝑆𝑝 , 𝑆𝑝 + 𝐿) with [𝑆𝑞 , 𝑆𝑞 + 𝐿), while keeping the CLS token
unchanged. Since this operation occurs in feature space rather than the
pixel domain, it avoids low-level distortions while enhancing geometric
robustness.
During Stage 2 training, MDO takes part using a step-by-step adversarial optimization strategy. This strategy is designed to ensure
distribution consistency between enhanced and original features. First,
we freeze the parameters of the backbone network. Then, we only
optimize the classifier to create adversarial gradients. This makes the
classifier suppress noise-sensitive dimensions. As a result, it improves
the robustness to occlusion and deformation. The corresponding loss is
defined as:
1𝑠𝑡𝑎𝑔𝑒2 = 𝑖𝑑 − 𝜆𝑛𝑜𝑖𝑠𝑒 𝑖𝑑_𝑛𝑜𝑖𝑠𝑒 − 𝜆𝑒𝑟𝑎𝑠𝑒 𝑖𝑑_𝑒𝑟𝑎𝑠𝑒 − 𝜆𝑡𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚 𝑖𝑑_𝑡𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚 ,

feature augmentations, enables the model to achieve stronger occlusion robustness and improved generalization in challenging scenarios. The process of implementing adversarial denoising with negative
weights, followed by enhancing feature robustness through collaborative optimization with positive weights, gradually optimizes the
gradient conflict problem among multiple enhancement branches. The
above optimization should not be interpreted as a naive sign-flipping
strategy. Instead, it is an alternating minimax-style optimization in
parameter space. In the first step, only the classifier is updated while
the backbone is frozen, which constructs stricter decision boundaries
for the enhanced feature branches under fixed feature representations
and prevents uncontrolled feature drift. In the second step, the full
network is optimized to reduce the vulnerability exposed by these enhanced branches, thereby improving feature robustness to perturbationinduced distribution shifts. Therefore, the adversarial effect in MDO is
realized through alternating optimization between classifier robustness
and feature adaptation, rather than through pixel-level adversarial
sample generation. This design makes the training process more stable
while still preserving the benefit of worst-case-oriented robustness
enhancement.
4. Experiments and discussion
To evaluate the effectiveness of the proposed RMPSNet, we conducted experiments on five widely used public person ReID datasets.
These datasets serve as well-recognized benchmarks in the field of
person ReID, providing a reliable basis for assessing the performance
of the RMPSNet.
4.1. Datasets
We evaluated RMPSNet on two general ReID datasets (Market1501 [2], DukeMTMC-ReID [25]) and three occluded ReID datasets
(Occluded-Duke [3], Occluded-ReID [26], Partial-ReID [27]), each
with unique characteristics that highlight different challenges. Market1501 [2] comprises 32,668 annotated images of 1501 identities captured by six cameras in an open-world setting. It provides 12,936
training images covering 751 identities, making it a standard benchmark for evaluating general person ReID performance under diverse
camera views. DukeMTMC-ReID [25] contains 36,411 images of 1404
identities collected from eight camera views. The dataset includes
16,522 training images, 17,661 gallery images, and 2228 query images. It has challenging cross-camera matching scenarios and complex
background variations. Occluded-DukeMTMC [3] is derived from
DukeMTMC-ReID and specifically focuses on occlusion. It consists of
15,618 training images of 708 identities, 2210 query images of 519
identities, and 17,661 gallery images of 1110 identities. Notably, all
query images are partially occluded by various obstacles, making this
dataset ideal for evaluating occlusion robustness. Occluded-ReID [26]
comprises 2000 images of 200 identities, with each identity represented
by 10 images-five full-body images and five occluded images. Typically,
occluded images serve as queries, while all full-body images are used as
the gallery. This dataset emphasizes testing a model’s ability to recognize individuals under severe occlusion conditions. Partial-ReID [27] is
an image-based partial person ReID dataset, encompassing images with
various perspectives, backgrounds, and occlusion types. The dataset
consists of 600 photos from 60 individuals, with each person having
5 full-body images and 5 partial images. These partial images were
manually cropped from occluded images and only include the test set
used for evaluation.
Note that as the Occluded-ReID and Partial-ReID datasets have no
official training splits, we follow the evaluation protocols in [26,27]
and train our model on Market-1501. The performance on this benchmark thus reflects the cross-dataset generalization capability of our
method under dataset distribution shift.

(7)

where 𝜆𝑛𝑜𝑖𝑠𝑒 = 0.1, 𝜆𝑒𝑟𝑎𝑠𝑒 = 0.1, and 𝜆𝑡𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚 = 0.15.
It should be noted that although both RPE and MDO are related to
occlusion robustness, their functions are complementary. RPE mainly
plays a role at the input level, reducing the mismatch between simulated occlusion patterns and actual monitoring scenarios by simulating
real occlusion distributions during the training process. In contrast,
MDO operates in the feature space, by introducing multiple perturbation branches and adversarial optimization, it enhances the robustness of features and the consistency of distributions under various
interferences.
After classifier optimization, we proceed to fine-tune the entire
network. Following prior works [5,24], we compute cross-entropy loss
and triplet loss to refine feature discrimination. The cross-entropy loss
from image to text is defined as:
𝑖2𝑡𝑐𝑒 (𝑖) =

𝑁
∑

exp(𝑠(𝑉𝑖 , 𝑇𝑦𝑘 ))
−𝑞𝑘 log ∑𝑁
,
𝑘=1
𝑦 =1 exp(𝑠(𝑉𝑖 , 𝑇𝑦𝑎 ))

(8)

𝑎

In this stage, we also construct an adversarial objective to balance
feature robustness and discriminative power:
′

1𝑠𝑡𝑎𝑔𝑒2 = 𝑖𝑑 + 𝜆𝑛𝑜𝑖𝑠𝑒 𝑖𝑑_𝑛𝑜𝑖𝑠𝑒 + 𝜆𝑒𝑟𝑎𝑠𝑒 𝑖𝑑_𝑒𝑟𝑎𝑠𝑒 + 𝜆𝑡𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚 𝑖𝑑_𝑡𝑟𝑎𝑛𝑠𝑓 𝑜𝑟𝑚 ,

(9)

The final loss function for Stage 2 is given by:
′

2𝑠𝑡𝑎𝑔𝑒2 = 𝑖𝑑 + 𝜆1 𝑡𝑟𝑖 + 𝑖2𝑡𝑐𝑒 + 𝜆2 1𝑠𝑡𝑎𝑔𝑒2 ,

(10)

where 𝜆1 = 0.25 and 𝜆2 = 0.9 are the weights for balancing different
losses. This two-step adversarial optimization, combined with targeted
5


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, 2025

1

See What You Seek: Semantic Contextual Integration
for Cloth-Changing Person Re-Identification

arXiv:2412.01345v2 [cs.CV] 18 May 2025

Xiyu Han , Xian Zhong , Senior Member, IEEE, Wenxin Huang , Member, IEEE, Xuemei Jia , Xiaohan Yu ,
and Alex Chichung Kot , Life Fellow, IEEE

Abstract—Cloth-changing person re-identification (CC-ReID)
aims to match individuals across surveillance cameras despite
variations in clothing. Existing methods typically mitigate the
impact of clothing changes or enhance identity (ID)-relevant
features, but they often struggle to capture complex semantic
information. In this paper, we propose a novel prompt learning
framework Semantic Contextual Integration (SCI), which leverages
the visual-textual representation capabilities of CLIP to reduce
clothing-induced discrepancies and strengthen ID cues. Specifically,
we introduce the Semantic Separation Enhancement (SSE) module,
which employs dual learnable text tokens to disentangle clothingrelated semantics from confounding factors, thereby isolating
ID-relevant features. Furthermore, we develop a Semantic-Guided
Interaction Module (SIM) that uses orthogonalized text features to
guide visual representations, sharpening the focus of the model on
distinctive ID characteristics. This semantic integration improves
the discriminative power of the model and enriches the visual
context with high-dimensional insights. Extensive experiments on
three CC-ReID datasets demonstrate that our method outperforms
state-of-the-art techniques. The code will be released at https:
//github.com/hxy-499/CCREID-SCI.

Use explicit auxiliary features
Data
processing
…
Image
parsing gait
ID = 009

ID = 112

Same
person

encoder

skeleton

(a) Existing methods
Textual feature space Remove negative factors
Text
Encoder
Visual
Encoder

ID = 009

𝑭𝐜𝐥𝐨

𝑭𝐢𝐝

𝑭𝐜𝐥𝐨

Textual guidance

Different
person

ID = 112

Mining positive factor

(b) Our method (SCI)

Fig. 1. Comparison of traditional methods and our SCI approach. (a)
Traditional methods rely on parsing, gait analysis, skeleton extraction, and data
augmentation to suppress clothing effects, incurring significant preprocessing
overhead. (b) Our SCI approach directly removes clothing bias within the
model and exploits inherent ID-related features from images.

Index Terms—Person Re-Identification, Clothing Changes,
Vision-Language Models, Prompt Learning, Semantic Integration.

I. I NTRODUCTION

P

ERSON re-identification (ReID) is the task of matching
individuals across non-overlapping cameras over time,
with important applications in video surveillance and smart
city systems [1]–[4]. Traditional ReID methods [5], [6] exploit
the appearance of clothing, e.g., texture and color, to distinguish
Manuscript received May 17, 2025. This work was supported in part by
the National Natural Science Foundation of China under Grants 62271361
and 62301213, the Hubei Provincial Key Research and Development Program
under Grant 2024BAB039, and the Open Project Funding of the Hubei Key
Laboratory of Big Data Intelligent Analysis and Application, Hubei University
under Grant 2024BDIAA01. (Corresponding author: zhongx@whut.edu.cn)
Xiyu Han and Xian Zhong are with the Sanya Science and Education
Innovation Park, Wuhan University of Technology, Sanya 572025, and also
with the Hubei Key Laboratory of Transportation Internet of Things, School of
Computer Science and Artificial Intelligence, Wuhan University of Technology,
Wuhan 430070, China (e-mail: hanxy@whut.edu.cn; zhongx@whut.edu.cn).
Wenxin Huang is with the Hubei Key Laboratory of Big Data Intelligent
Analysis and Application, School of Computer Science and Information
Engineering, Hubei University, Wuhan 430062, China (e-mail: wenxinhuang wh@163.com).
Xuemei Jia is with the National Engineering Research Center for Multimedia
Software, School of Computer Science, Wuhan University, Wuhan 430072,
China (e-mail: jiaxuemeiL@whu.edu.cn).
Xiaohan Yu is with the School of Computing, Macquarie University, Sydney,
NSW 2109, Australia (e-mail: xiaohan.yu@mq.edu.au).
Alex Chichung Kot is with the Rapid-Rich Object Search Lab, School
of Electrical and Electronic Engineering, Nanyang Technological University,
Singapore 639798 (e-mail: eackot@ntu.edu.sg).

identities (IDs), but their performance degrades when subjects
change clothing. This limitation motivates the development of
cloth-changing ReID (CC-ReID) techniques that remain robust
under clothing variations.
Existing CC-ReID approaches can be grouped into two
categories: those that suppress clothing cues and those that
enhance ID-relevant features (see Fig. 1(a)). The first category
seeks to minimize the influence of clothing: CAL [7] employs
adversarial learning to penalize clothing-based predictions in
the RGB modality, and AIM [8] uses causal intervention
through a dual-branch model to mitigate clothing bias. The
second category injects auxiliary ID cues, such as pose, gait, or
human parsing, to strengthen discriminative features. For example, GI-ReID [9] leverages gait information to learn clothingagnostic representations, while SCNet [10] imposes semantic
consistency constraints using a human parsing network.
Although these methods partially address clothing changes,
they typically focus on minimizing negative clothing effects or
relying on explicit cues such as body contours. We argue that
visual representations inherently contain both negative factors
(e.g., clothing styles that frequently change and can confuse the
model) and positive factors (e.g., stable attributes such as hair,
glasses, or backpacks that aid ID discrimination). As illustrated
in Fig. 1(b), our goal is to remove negative influences while
making implicit positive factors more explicit.
Recent advances in vision-language learning, notably Contrastive Language-Image Pre-training (CLIP) [11], have demon-

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, 2025

strated strong abilities to bridge visual data and natural
language, yielding context-aware representations beneficial for
many downstream tasks [12], [13]. In ReID, methods like CLIPReID [14] and Instruct-ReID [15] show that aligning images
with descriptive language captures rich semantics beyond
appearance alone. Inspired by these successes, we leverage
prompt learning to extract high-dimensional semantic cues from
CLIP, thereby amplifying positive factors without additional
data preprocessing.
In this paper, we introduce the Semantic Contextual Integration (SCI) framework for CC-ReID. SCI uses CLIP
to capture rich semantic features and includes a Semantic
Separation Enhancement (SSE) module isolates clothing-related
negative factors at the text level while preserving key positive
semantics for ReID. Specifically, SSE employs dual learnable
text tokens to disentangle confounding semantics (both positive
and negative) from clothing semantics, then orthogonalizes the
token embeddings to filter out clothing factors. The refined
text features guide visual encoding via our Semantic-Guided
Interaction Module (SIM), focusing the model’s attention on IDrelevant cues beyond explicit contours. By integrating CLIP’s
multimodal strengths, SCI enriches visual representations
with high-dimensional semantic context and achieves superior
performance on three standard CC-ReID benchmarks.
Our contributions are summarized threefold:
• We propose the Semantic Contextual Integration (SCI)
framework to leverage semantic information in CC-ReID,
removing negative factors, emphasizing implicit positive
elements, and refining visual representations.
• We introduce the Semantic Separation Enhancement (SSE)
module to filter and refine text-level features, improving
the model’s ability to isolate key semantic information.
• We design the Semantic-Guided Interaction Module (SIM)
to guide visual representations using refined text features,
enhancing multi-modal integration and alignment.

2

positives to rank ahead of negatives based on distance, and
further sorting positives by similarity score. Parameterized
RV Loss [26] jointly optimizes retrieval and verification tasks
by aligning loss functions with evaluation metrics, enabling
automatic loss function search.
Although these techniques handle variations in pose, background, and viewpoint, they assume identical clothing across
views. Consequently, their performance degrades when subjects
change clothing, motivating research on CC-ReID.
B. Cloth-Changing Person ReID

Several CC-ReID datasets have been proposed, PRCC [27],
LTCC [28], COCAS+ [29], and VC-C LOTHES [30], to evaluate models under clothing variations. The primary challenge is
to learn features that remain reliable despite clothing changes.
Existing CC-ReID methods follow three main strategies:
1) Auxiliary Soft-Biometric Cues: CAMC [31] integrates
body-shape semantics into ID features, and FSAM [32]
extracts fine-grained shape information to complement clothingindependent cues. M2Net [33] uses contour and parsing maps
for appearance-robust features, PGAL [34] aligns keypoints via
pose estimation, FLAG [35] is proposed to explicitly extract
appearance and gait information, and can be integrated with
most existing video-based ReID methods, and CVSL [36]
jointly learns body-shape embeddings and appearance features.
2) Feature Disentanglement: CAL [7] employs adversarial
loss to suppress clothing-related features, and AIM [8] uses
causal intervention to remove clothing bias. DCR-ReID [37]
disentangles and reconstructs feature components, LDF [38]
separates ID, clothing, and unrelated factors via GANs, 3DInvarReID [39] disentangles features and reconstructs two-layer
3D body shapes, FIRe2 [40] augments images with fine-grained
attributes, and MAL-F [41] learns invariant features from RGB,
grayscale, and contour inputs using a ResTNet backbone.
3) Data Augmentation and Adaptation: Pos-Neg [42] and
CCFA [43] augment clothing color and texture diversity;
II. R ELATED W ORK
RCSANet [44] constructs explicit clothing-status embeddings to
A. Cloth-Consistent Person ReID
bolster feature robustness; MCSC [45] applies meta-learning to
Person ReID under consistent clothing conditions has been address clothing-distribution shifts; Zhao et al. [46] model
extensively studied [16]. These methods exploit clothing clothing changes as fine-grained domain shifts via graph
appearance, such as color and texture, to extract discriminative relations; and DCLR [47] synthesizes multi-clothing images
features. Feature representation learning in this context can be via diffusion and merges them into training data.
divided into three categories: global features, local features, and
Despite these advances, most CC-ReID methods rely solely
auxiliary information-based features. Global methods extract a on visual inputs, limiting their semantic depth. In this work,
single feature vector per image [17], while local methods aggre- we integrate visual and textual modalities via CLIP to enrich
gate part-based representations to address misalignment (e.g., semantic understanding and improve cloth-changing ReID.
human parsing [18] or horizontal partitioning [19]). Auxiliaryinformation approaches incorporate extra cues, such as semantic
attributes [20] or synthesized samples [21], to enrich context. C. Vision-Language Learning
Vision-language pre-training (VLP) trains models to align
Strong baselines in this domain include AGW [22] and
TransReID [23]. ISP [24] further refines alignment by locating images with text, improving downstream visual tasks. CLIP [11]
body parts and carried items at the pixel level. Moreover, uses paired image and text encoders to learn a shared embedsince person ReID is inherently formulated as a ranking ding space via contrastive learning, benefiting applications such
problem, class imbalance can also have a significant impact as captioning and classification.
on performance. Therefore, some works propose loss function
Prompt learning extends CLIP by making context tokens
modifications to handle data imbalance during training. For learnable. CoOp [48] transforms prompt words into trainable
example, DRSL [25] optimizes the ReID model by enforcing vectors, CoCoOp [49] generates input-conditional tokens per

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, 2025

3

Semantic Separation Enhancement

[X]1…[X]n

clothes.
person.

Text
Encoder

𝑭𝐜𝐥𝐨

𝑭𝐜𝐥𝐨

Dual Prompt

𝑭𝐢𝐝

pr oj

!-!
𝐹#&

Image
Semantic-Guide Interaction Module
A photo of the

[X]1…[X]n

clothes.

A photo of a

[X]1…[X]n

person.

Text
Encoder

Frozen

!-!
𝐹+,!

SSE
Mining positive factor

!-!
𝐹#&

Nonlocal
Operation

Visual
Encoder
#%0

Image

Training

!-!
𝐹')+

Dual Prompt

…

n

!-!
𝐹.,+/

Remove
negative
factor

ℒ!"#
+ ℒ#"!
+ ℒ$#%

Visual
Features

Visual
Encoder

…

𝑭𝐢𝐝
ecti
o

F+,#

!-!
𝐹+,!

MLP

A photo of a

[X]1…[X]n

cross
Attention

A photo of the

Textual Feature Space

!-!
𝐹')+

#%0

F&#11

F #%0

ℒ#& +
ℒ'() +
ℒ#"!'*

Fig. 2. Framework of the proposed SCI, comprising two key components: the Semantic Separation Enhancement (SSE) module and the Semantic-Guided
Interaction Module (SIM). SSE mitigates clothing bias by removing negative semantic factors, while SIM employs the refined text features to guide visual
representations, strengthening cross-modal interaction.

image, and DenseCLIP [50] applies pixel-text matching for
dense prediction.
In person ReID, CLIP-ReID [14] introduces ID-specific
tokens and a two-stage training scheme. CCLNet [51]
adapts prompt learning for unsupervised visible-infrared ReID.
CSDN [52] employs bimodal descriptions to align visible and
infrared features. RGANet [53] uses CLIP to locate informative
body parts for occluded ReID. VGSG [54] groups text features
semantically to address fine-grained misalignment. CCAFL [55]
integrates CLIP-generated textual information into a novel
semi-supervised framework to actively learn camera-agnostic
pedestrian features. MIPL [56] guides CC-ReID with multiple
common prompts, and CCAF [57] leverages CLIP’s knowledge
to learn fine-grained, clothing-independent semantics.
Although CLIP is not specifically pre-trained for ReID, its
large-scale image-text training captures high-level semantics
(e.g., body shape and context) remain invariant to clothing. We
build on this by proposing CLIP-CCReID, which uses prompt
learning to mitigate clothing changes without auxiliary models.
III. P ROPOSED M ETHOD
A. Preliminaries and Overview
The large-scale vision-language model CLIP [11] learns
joint image-text representations via a contrastive objective. It
comprises a visual encoder I(·) and a text encoder T (·), whose
embedding spaces are aligned during pre-training.

Algorithm 1: Semantic Separation Enhancement Module
: Dual textual learnable vectors promptid ,
promptclo , and person images.
txt
Output : Optimized text representations Fort
1 for each epoch e from 1 to epochs do
2
for each batch in trainloader do
3
Extract image features Fimg using the frozen visual
encoder I(·) Train context vectors for dual prompts
promptid and promptclo with Eq. (1) and Eq. (2)
txt
txt
Compute text features Fid
and Fclo
for ID and
txt
txt
clothing using Eq. (3) Project Fclo
onto Fid
using
txt
Eq. (4) Perform orthogonalization to obtain Fort
using Eq. (5) Compute Lsim loss to regulate
semantic separation using Eq. (8)
Input

4

txt
return Fort

For downstream classification, CoOp [48] refines CLIP’s
zero-shot prompts by introducing learnable context tokens.
Specifically, given a template such as [V ]1 [V ]2 . . .
[V ]M [CLS] where each [V ]i (i = 1, . . . , M ) is a trainable
vector and [CLS] is a class token, CLIP computes image-text
similarities in the shared embedding space and predicts the
class with maximal similarity.
Building on these ideas, we leverage CLIP and CoOp
to extract rich semantic prompts for CC-ReID (see Fig. 2),
enabling robust ID modeling under clothing changes.

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, 2025

4

B. Semantic Separation Enhancement

Algorithm 2: Semantic-Guided Interaction Module

CLIP cannot explicitly ignore unwanted semantics (e.g.,
clothing) via negative prompts. To address this, we propose
the Semantic Separation Enhancement (SSE) module, which
isolates and removes clothing-related components from the
high-dimensional text features, preserving only ID-relevant
semantics. The overall procedure is detailed in Algorithm 1.
We define two learnable prompts:

: Dual textual learnable vectors promptid ,
promptclo , and person images.
txt
Output : Optimized text representations Fort
1 for each epoch e from 1 to epochs do
2
for each batch in trainloader do
3
Extract dual text features Ftxt using the frozen text
encoder T (·) Compute orthogonalized text features
txt
Fort
Extract image features Fimg using the training
visual encoder I(·) Enhance image features with
contextual information using Eq. (13) Compute
semantic-guided image features with Eq. (14) and
txt
Eq. (15) Refine image features based on Fort
using
Eq. (16) Apply loss functions to regulate the visual
encoder training using Eq. (20)

promptid = A photo of a [X]1 [X]2 . . . [X]M person. (1)
promptclo = A photo of the [X]1 [X]2 . . . [X]M clothes.
(2)
where [X]i (i = 1, . . . , M ) are learnable tokens initialized
randomly, and M denotes the number of tokens. In the first
training stage, we use the pre-trained visual encoder I(·) and
text encoder T (·) to extract image and dual text features,
freezing the encoders’ parameters to focus on optimizing
the text tokens [X]i . This allows us to learn contextual
representations by updating the text tokens, thereby acquiring
distinct textual representations for each ID and their clothing:
txt
Fid
= T (promptid ) ,

txt
Fclo
= T (promptclo ) ,

(3)

Input

where λ1 and λ2 are weighting factors controlling the contributions of ID and clothing similarities.
Finally, we calculate the image-to-text contrastive loss Li2t :
txt
exp s (Vi , Fort
(i))
Li2t (i) = − log PN
,
txt
k=1 exp s (Vi , Fort (k))

(9)

txt
txt
txt
where Fid
and Fclo
represent the ID-specific text features where Vi and Fort (i) are paired visual and text embeddings,
s(·, ·) denotes the similarity function, and N is the batch size.
and clothing text features, respectively.
txt
After extracting the dual text representations, we project Fclo Since multiple images in a batch may belong to the same ID,
txt
onto Fid
to reduce the influence of clothing and emphasize meaning there may be multiple positive samples, the text-tothe implicit positive factors. This step clarifies the relation be- image contrastive loss Lt2i (yi ) is calculated as:
tween clothing representations and ID representations, thereby
X
−1
exp s (Vp , Tyi )
Lt2i (yi ) =
log PN
, (10)
minimizing the impact of clothing variations. The computation
|P (yi )|
k=1 exp s (Vk , Tyi )
p∈P (yi )
process is as follows:
txt
Fproj
=

txt txt
Fclo
Fid

F txt ,
txt ∥2 id
∥Fid

(4)

We then perform orthogonalization by subtracting the projection
txt
txt
txt
Fproj
of the clothing feature Fclo
onto the Fid
. This operation
removes the clothing component that is aligned with the ID
txt
direction. This process ensures that the features Fort
are aligned
with the positive features while removing the influence of
negative factors. The orthogonalization process is:
txt
txt
txt
Fort
= Fid
− Fproj
,

(5)

txt
where Fort
is after removing the part that is related to clothing,
ensuring the feature focuses on positive information while
reducing interference from clothing.
To optimize the text features, we design a loss function
txt
txt
that maximizes the similarity between Fort
and Fid
, while
txt
minimizing the similarity with Fclo . This enables the model
to better capture positive factors without being affected by
negative aspects, such as clothing changes. The similarity
calculations are:

txt
txt
simid (i) = mean cos Fort
(i), Fid
(i) ,
(6)

txt
txt
simclo (i) = mean cos Fort
(i), Fclo
(i) ,
(7)

where mean(·) denotes the average value, and cos(·) represents
the cosine similarity function. The loss function is as follows:
Lsim (i) = λ1 (1 − simid (i)) + λ2 simclo (i),

(8)

txt
where P (yi ) is the set of all positive indices for Fort
(yi ) in
the batch, and Tyi is the text embedding corresponding to label
yi . Therefore, the overall loss function is:
X
Lprompt =
(Li2t + Lt2i + Lsim ) .
(11)
i=1

C. Semantic-Guide Interaction Module
Previous work [7], [14] shows that reducing clothing bias
in visual or textual features improves performance in clothchanging scenarios. However, these methods typically treat
visual and textual branches independently, lacking interaction. Our method refines visual representations by leveraging
clothing-irrelevant textual features, enhancing the interaction
between visual and textual branches. This integration introduces
more robust and invariant descriptors into the visual processing,
improving the model’s ability to identify individuals across
clothing changes. The algorithm is summarized in Algorithm 2.
As illustrated in Fig. 3, we employ a Transformer decoder [58] to model interactions between visual and textual data.
To better capture long-range dependencies within the visual
domain, we replace the conventional self-attention mechanism
with a non-local operation [59], which allows the model to
aggregate information from spatially distant but semantically
relevant regions. This capability is particularly important for
handling variations in clothing. During this stage, we update

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, 2025

conv

q
%&'

F"()

v

cross
Attention

conv
%&'
F!"%

k

soft
max

conv


where y i is the true label for the i-th sample, and pid y i | xi is
the predicted probability of the true label y i . The clothes-based
adversarial loss is expressed as:

#$#
F!"#

MLP

Nonlocal Operation

%&'

F!*#

5

%&'

F+%,,

Lcal = −

NC
N
XX

q(c) log

i=1 c=1

Fig. 3. Illustration of the SIM process. Textual information refines visual
feature extraction to align features with the relevant semantic context.

the visual encoder while keeping other components frozen. The
computation is as follows:

 

img
img


θ Fori
ϕ Fori
img
img
Fcon
=
g Fori
,
(12)
N
img
where Fori
is the input feature map, θ(·), ϕ(·), and g(·) are
linear transformations, and N is the number of elements in
img
the feature map. Fcon
captures global context information.
The final visual feature map is:

img
img
img
Fres
= W Fcon
+ Fori
,
(13)
img
where W contains learnable parameters. Fres
is the final output
feature map obtained by adding this context-aware feature to
the original feature map.
To refine visual representations with semantic guidance, we
leverage textual embeddings to generate attention weights over
visual features. This allows the model to selectively emphasize
semantically relevant regions in the visual space, leading to
more discriminative and context-aware visual representations:
!
img txt T
Fres
Fort
img
img
txt
√
Fout = Fres + softmax
Fort
,
(14)
dk



img
img
Fdiff
= MLP LayerNorm Fout
,
(15)

where dk is the dimensionality of the keys, LayerNorm is
used to normalize the input features to stabilize training, and
MLP denotes a two-layer feed-forward network. This integrates
visual features under the guidance of textual embeddings, and
enables the model to extract the most informative visual cues
aligned with the semantic content of the text.
Finally, we update the visual features. This process ensures
that the visual features are enriched with relevant semantic information from the text, leading to more robust and contextually
aware visual representations:
img
img
F img = Fori
+ αFdiff
,

(16)

where α ∈ RC is a learnable parameter controlling the degree
of refinement applied to the visual features.
This module adopts an interaction-to-alignment approach,
where text embeddings guide the visual encoding process. By
emphasizing positive cues and mitigating negative ones, this
method improves the model’s ability to consistently recognize
individuals across different clothing scenarios.
For the objective function, we incorporate cross-entropy loss
Lid and clothes-based adversarial loss Lcal [7] to optimize the
visual encoder:
N
X

Lid = −
y i log pid y i | xi ,
(17)
i=1

exp (fi φc /τ )
P
,
exp (fi φc /τ ) + j∈S − exp (fi φj /τ )
i
(18)

where NC is the number of clothing categories, φc is the
clothes classifier, q(c) is the weight for the c-th class, fi is the
feature of sample i, τ is a temperature parameter, and Si− is
the set of clothes classes with the same ID. The objective of
Lcal is to encourage the backbone to extract clothes-irrelevant
features by penalizing its predictive power regarding clothing.
To fully leverage the capabilities of CLIP, we calculate the
image-to-text cross-entropy loss Li2tce as:
Li2tce (i) =

N
X

txt
exp s(Vi , Fort
(i))
−qk log PN
,
txt
k=1 exp s(Vi , Fort (k))
k=1

(19)

where label smoothing is applied to qk . The total loss for the
SIM module is:
L = Lid + Lcal + Li2tce .

(20)

By jointly optimizing Lid , Lcal , and Li2tce , the model captures
ID features from easy samples (same clothes) and progressively
learns to distinguish hard cases (same ID, different clothes)
while aligning visual and textual representations. The weights
of loss terms are basic terms [7], [14] and set as 1.
IV. E XPERIMENTAL R ESULTS
A. Datasets and Evaluation Metrics
We evaluate our method on three standard cloth-changing
datasets: LTCC [28], PRCC [27], and VC-C LOTHES [30],
following the protocols of CAL [7] and AIM [8]. Additionally,
to validate generalization, we test on two conventional ReID
datasets: M ARKET 1501 [60] and MSMT17 [61].
LTCC contains 17,138 images of 152 IDs captured by 12
cameras. The training set comprises 9,576 images of 77 IDs;
the testing set includes 7,543 images (493 queries and 7,050
gallery) of 75 IDs. This long-term dataset features frequent
clothing changes and varied environmental conditions, with
each ID wearing between two and fourteen outfits.
PRCC consists of 33,698 images of 221 IDs from three
cameras. Its training set has 22,898 images of 150 IDs, and the
testing set has 10,800 images of 71 IDs. Each person appears
in two outfits: cameras A and B share the same clothing, while
camera C uses a different outfit.
VC-C LOTHES is a synthetic dataset generated via the GTA5
engine. It contains 9,449 images of 256 IDs across four camera
views, with 1,241 distinct clothing items distributed between
training and testing sets.
M ARKET 1501 includes 12,936 training images of 751 IDs
and 19,732 testing images of 750 IDs, captured by six cameras
under the single-query evaluation protocol.
MSMT17 comprises 32,621 training images of 1,041 IDs
and 51,027 testing images of 11,659 IDs across 15 cameras, it
is one of the largest and most challenging ReID benchmarks.

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, 2025

6

We employ the cumulative matching characteristic (CMC) ID setup, improvements on other metrics are relatively smaller
curve and mean average precision (mAP) as evaluation metrics. but still notable.
On VC-C LOTHES, SCI achieves a substantial mAP gain of
Rank-k in CMC measures the probability of the correct match
appearing among the top-k retrieved images, while mAP reflects 7.4% over AIM in the cloth-changing setting, further validating
its effectiveness in challenging scenarios.
the overall retrieval performance across all queries.
2) ViT-Based Model: With ViT-B/16, SCI attains near
Experiments are conducted under three protocols: 1) General
state-of-the-art
performance, especially on LTCC, reaching
setting: exclude same-ID and same-camera samples, using both
45.9%
Rank-1
under
severe clothing variations. Crucially, these
cloth-changing and clothing-consistent pairs; 2) Same-clothes
results
are
obtained
without auxiliary models or additional
setting: exclude same-ID and same-camera samples, using only
preprocessing,
distinguishing
SCI from methods that rely on
clothing-consistent pairs; 3) Cloth-changing setting: exclude
external
components.
same-ID, same-camera and same-clothing samples, using only
cloth-changing pairs.
D. Ablation Studies and Analysis
We perform ablation studies to assess each module’s contriB. Implementation Details
bution and analyze key parameters.
We adopt two backbones for feature extraction: a modified
1) Impact of Semantic Separation Enhancement: We
ResNet-50 and ViT-B/16 pre-trained on CLIP. All images are first evaluate the baseline without additional modules,
resized to 384 × 192 pixels, with a batch size of 64. We use where the model relies solely on a prompt “a photo
the Adam optimizer [62] and apply random horizontal flipping, of a [X]1 . . . [X]M person” for classification. This
cropping, and erasing [63] for data augmentation.
approach shows reduced performance. Introducing the
For the CNN branch, a global attention pooling layer to SSE module, which uses two prompts (“a photo of
reduce the feature dimension from 2048 to 1024, matching a [X]1 . . . [X]M person”
and
“a photo of the
the text feature dimension (scaled from 512 to 1024). For the [X]1 . . . [X]M clothes”) to learn features separately at
transformer branch, we employ 12 transformer layers with the textual level. We observe consistent improvements of
hidden size 768; a linear projection reduces the image feature approximately 2.0% in both Rank-1 accuracy and mAP across
dimension from 768 to 512 to align with the text features.
datasets for both CNN-based and ViT-based models, as shown
Training proceeds in two stages: 1) Stage 1 (prompt learning): in Table II. This demonstrates the effectiveness of the SSE
train two prompt learners for 60 epochs (ResNet-50) or 120 module in filtering clothing features.
epochs (ViT-B/16) with an initial learning rate of 3.5 × 10−4
These results indicate: 1) The original generic textual
and cosine decay. 2) Stage 2 (visual encoder fine-tuning): train representations are confounding, containing both ID and
the visual encoder I(·) for 120 epochs (ResNet-50) or 30 clothing features, leading the model to focus excessively on
epochs (ViT-B/16), starting at 3.5 × 10−4 and reducing by a cloth-relevant aspects. 2) Separating these features at the textual
factor of 10 at epochs 40 and 70.
level is simple yet effective. These findings demonstrate the
All experiments are implemented in PyTorch and run on a effectiveness of setting specific prompts for each factor and
single NVIDIA A100 GPU.
the robustness of our SSE module against both cloth-changing
and general scenarios.
2) Impact of Semantic-Guided Interaction Module: We
C. Comparison with State-of-the-Art Methods
further assess SIM by integrating it with the baseline model. As
We compare SCI against state-of-the-art methods on shown in Table II, using SIM alone often degrades performance
LTCC, PRCC and VC-C LOTHES, including general ReID based on CNN, particularly in the cloth-changing scenario on
approaches (ISP [24], CLIP-ReID [14], and Instruct-ReID [15]) PRCC, where Rank-1 drops by 4.6% and mAP decreases from
and specialized cloth-changing methods (FSAM [32], RC- 55.3% to 49.9%.
SANet [44], CAL [7], GI-ReID [9], M2Net [33], AIM [8],
The decline in performance in this set of ablation experiments
Pos-Neg [42], ACID [64], 3DInvarReID [39], DCR-ReID [37], is primarily due to the absence of the SSE module. Typically,
IMS+GEP [46], AFL [4], CVSL [36], MCSC [45], MAL-F [41], the SSE module generates final textual features that guide
and DLCR [47]).
the visual representations during the updating process of the
As shown in Table I, SCI consistently outperforms baselines visual encoder in SIM. However, with only a single prompt
by effectively suppressing ID-irrelevant features and reinforcing “a photo of a [X]1 . . . [X]M person” used to extract
discriminative ID cues.
textual representations, many ID-irrelevant negative factors,
1) CNN-Based Model: On LTCC (cloth-changing), SCI particularly clothing factors, are included. This results in an
improves Rank-1 by 3.8% and mAP by 2.7% over CLIP- excessive emphasis on clothing-relevant representations during
ReID [14]. Although SCI’s mAP is marginally lower than guidance in SIM, reducing overall performance.
AIM [8] in the general setting, this is expected: by filtering
3) Feature Distribution via t-SNE: To illustrate SCI’s
out clothing information, SCI may sacrifice some performance effectiveness, we employ t-SNE visualization [65], as depicted
when clothing cues are reliable, but gains robustness under in Fig. 4. This visualization plots the statistical distribution
cloth variation.
for 20 randomly selected categories from LTCC, comparing
On PRCC, SCI surpasses AIM by 3.5% in Rank-1 accuracy the latent space distributions at different stages of the baseline
under clothing changes. Due to PRCC’s simpler two-outfit per model with our approach.

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, 2025

7

TABLE I
C OMPARISON OF R-1 ACCURACY (%) AND M AP (%) OF STATE - OF - THE - ART METHODS ON LTCC, PRCC, AND VC-C LOTHES . B OLD AND
UNDERLINE VALUES INDICATE THE BEST AND SECOND - BEST RESULTS , RESPECTIVELY. † INDICATES REPRODUCED RESULTS .
LTCC
Methods

General

Venue

ISP [24]
FSAM [32]
RCSANet [44]
GI-ReID [9]
Pos-Neg [42]
CAL [7] †
M2Net [33]
ACID [64]
CLIP-ReID [14] †
3DInvarReID [39]
DCR-ReID [37]
AIM [8] †
Instruct-ReID [15]
AFL [4]
MCSC [45]
IMS+GEP [46]
CVSL [36]
MAL-F [41]
DLCR [47]
SCI w/ ResNet-50
SCI w/ ViT-B

ECCV’20
CVPR’21
ICCV’21
CVPR’22
TIP’22
CVPR’22
ACM MM’22
TIP’23
AAAI’23
ICCV’23
TCSVT’23
CVPR’23
CVPR’24
TMM’24
TIP’24
TMM’24
WACV’24
TITS’25
WACV’25
Ours
Ours

PRCC

Cloth-changing

Same-clothes

VC-C LOTHES

Cloth-changing

General

Cloth-changing

R-1

mAP

R-1

mAP

R-1

mAP

R-1

mAP

R-1

mAP

R-1

mAP

66.3
73.2
63.2
75.7
73.4
65.1
73.0
76.1
75.7
75.8
74.4
73.9
76.4
37.5
75.7
79.9

29.6
35.4
29.4
37.0
39.4
30.6
36.5
42.3
41.3
52.0
39.1
40.2
41.9
16.2
40.6
44.4

27.8
38.5
23.7
36.2
38.0
29.1
38.3
40.9
41.1
41.8
42.1
42.2
43.4
44.5
38.8
41.3
42.1
45.9

11.9
16.2
10.4
14.4
17.2
14.5
15.9
18.9
20.4
17.9
18.4
19.4
18.2
21.3
17.0
19.6
18.6
21.6

92.8
100.0
100.0
99.5
99.1
100.0
100.0
100.0
100.0
99.8
99.7
97.5
99.6
100.0

97.2
99.6
99.1
99.0
99.5
99.7
99.7
99.7
99.8
99.8
99.1
97.7
99.4

36.6
50.2
54.9
54.7
59.3
55.4
57.0
56.5
57.2
56.3
54.2
57.4
57.8
57.3
57.5
57.1
66.5
59.8
62.8

48.6
65.8
55.4
57.7
66.1
55.3
57.2
57.4
56.5
52.3
56.5
57.3
65.8
56.9
57.7
63.0
56.2
60.0

94.5
94.7
93.1
95.1
93.1
91.2
93.9
93.2
94.7
94.9
95.4

94.7
94.8
88.3
94.7
86.9
84.9
88.3
87.9
94.9
89.2
91.6

72.0
78.6
64.5
82.6
84.3
85.9
81.0
89.7
82.5
83.3
81.8
87.1
89.2
91.7

72.1
78.9
57.8
81.7
74.2
79.3
75.7
78.9
83.0
83.2
81.7
81.1
83.1
87.5

TABLE II
A BLATION STUDY OF EACH COMPONENT OF THE SCI ON LTCC, PRCC, AND VC-C LOTHES . B OLD VALUES INDICATE THE BEST RESULTS .
LTCC
Backbone

Baseline

General

PRCC

Cloth-changing

Same-clothes

VC-C LOTHES

Cloth-changing

General

Cloth-changing

SSE

SIM

R-1

mAP

R-1

mAP

R-1

mAP

R-1

mAP

R-1

mAP

R-1

mAP

ResNet-50

#

#
#

73.0
76.5
72.8
75.7

36.5
39.7
36.3
40.6

38.3
39.8
37.8
42.1

15.9
18.3
16.1
18.6

100.0
100.0
99.7
99.6

99.5
98.4
98.0
97.7

57.0
59.0
52.4
59.8

55.3
54.7
49.9
56.2

93.1
94.1
94.0
94.9

86.9
88.0
88.9
89.2

85.9
87.6
88.4
89.2

79.3
81.1
82.4
83.1

ViT-B

#

#
#

79.5
76.3
77.5
79.9

43.6
43.9
43.9
44.4

41.6
43.4
42.9
45.9

19.4
21.0
20.5
21.6

99.9
100.0
99.8
100.0

99.2
99.3
99.2
99.4

60.9
62.3
61.4
62.8

58.5
58.7
59.1
60.0

94.4
94.6
94.3
95.4

89.8
89.8
90.4
91.6

90.6
91.4
90.9
91.7

84.7
84.8
85.7
87.5

#

#

From Fig. 4(a) and (c), it is observable that both the baseline
and our method exhibit disorganized feature distributions
with blurred class boundaries. During this stage, only the
prompts are trained while the visual and text encoders remain
frozen, limiting their capability to effectively extract diverse
representations. In contrast, as shown in Fig. 4(b) and (d),
the feature distributions for both methods become clearer,
indicating the effectiveness of the representations derived from
the text-image similarity computations.
Notably, the clusters in Fig. 4(d) are more compact and
distinct compared to those in Fig. 4(b). The red and green
dashed circles in Fig. 4(b) enclose samples with considerable
scatter, which are better clustered together in Fig. 4(d).
Therefore, this visualization not only serves as compelling
validation of our capability to effectively extract features and
discriminate IDs but also highlights our method’s potential to
significantly advance the field in cloth-changing scenarios.
4) Qualitative Prompt-Guided Refinement: To illustrate the
distinctions between the baseline [14] and SCI, we present in

Fig. 5 the attention maps activated by different prompts, highlighting how prompts guide the refinement of visual features
across LTCC, VC-C LOTHES, and PRCC. It demonstrates that
the learned prompts are capable of guiding the network to
focus on relevant semantic areas.
Fig. 6 shows VC-C LOTHES for the same ID but with different outfits, illustrating the regions of interest that our SCI model
focuses on under clothing variations. These visualizations
provide insights into the unique differences in feature activation
patterns. Moreover, while SCI leverages prompt learning to
jointly model ID and clothing information, it fundamentally
differs from attribute-based approaches, as it neither explicitly
recognizes attributes nor relies on additional attribute detectors.
As shown in Fig. 5, the baseline method tends to focus on
more dispersed areas, which include negative factors, potentially
detracting from its effectiveness. Conversely, SCI emphasizes
highly discriminative ID features, such as the person’s head,
shoulders, lower body, and shoes. This specific emphasis on
footwear aligns with findings from previous studies [66], which

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, 2025

8

scattered

assembled

assembled

scattered

(a) Baseline (stage1)

(b) Baseline (stage2)

(c) Ours (stage1)

(d) Ours (stage2)

Fig. 4. t-SNE visualization of 20 randomly selected classes from LTCC. Colors indicate ground-truth IDs. (a)–(b) depict successive stages of the baseline,
while (c)–(d) show the corresponding stages of our method.

Origin

Baseline

Ours

Origin

Baseline

Clothes

Ours

ID = 376

Clothes = 01

ID = 376

Clothes = 02

ID = 376

Clothes = 03

ID = 071

Clothes = 01

ID = 071

Clothes = 02

ID = 071

Clothes = 03

ID = 061

Clothes = 01

ID = 061

Clothes = 02

ID = 061

Clothes = 03

ID

Fig. 5. Visualization of feature maps on LTCC (first row), VC-C LOTHES
(second row), and PRCC (third row). The first column shows the original
images, while the second and third columns present the feature maps from
the baseline and our method, respectively.

Fig. 6. Visualization of feature maps on VC-C LOTHES. The vertical axis
represents different IDs, and the horizontal axis shows clothing variations. Red
circles highlight challenging cases. This figure illustrates the model’s attention
to the same ID across different clothing scenarios.

suggest that footwear often remains consistent across different time for the same individual, intra-class differences outweigh
outfits and scenarios. This indicates our method’s enhanced inter-class differences. Fig. 7(c) shows the distribution after orability to capture crucial positive factors when encountering thogonalizing the features, which achieves a clearer separation
challenging scenarios.
between intra- and inter-class similarity, indicating improved
In Fig. 6, for each person, the corresponding heatmaps feature discrimination. This demonstrates the effectiveness of
illustrate how the feature extraction model highlights various dual textual prompts and orthogonalization in enhancing feature
regions of interest across different outfits, reflecting the model’s alignment.
focus on certain body parts regardless of clothing changes.
6) Comparative Analysis of Mechanisms in SIM: As shown
However, there are some limitations. As indicated by the red in Fig. 8, we evaluate the effectiveness of the non-local
circles, the model tends to focus excessively on areas like the operation [59] compared to self-attention mechanisms [58] in
knees. If people wear long pants, this focus could potentially leveraging textual information obtained from the SSE module
affect the results. We are aware of this issue and will continue to guide visual representations. In cloth-changing scenarios, the
to optimize our model to address it.
non-local mechanism yields superior performance, achieving
5) Visualization of Feature Similarity: In Fig. 7, we visualize 42.1% in Rank-1 accuracy and 18.6% in mAP, as depicted
the similarity distributions of features between the text prompts in Fig. 8(a) and (b), thereby outperforming the self-attention
and the images. Fig. 7(a) is derived from the prompt in Eq. (1), mechanism by margins of 4.6% and 2.4%, respectively.
while the distribution is relatively well-formed, intra-class
This underscores the non-local mechanism’s enhanced casimilarity remains dispersed. Fig. 7(b) shows the distribution pability to capture comprehensive global context, which is
from the prompt in Eq. (2), which learns clothing-related vital for integrating information from disparate yet significant
semantics. However, due to significant clothing variations over segments of the visual field in cloth-changing scenarios. In


exec
/bin/zsh -lc "pdftotext -f 3 -l 5 'Robust mixed-degradation person Re-identification via structural consistency distillation.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113938

S. Zhao et al.

preserve key weather-specific visual cues and provide diverse experimental settings for analyzing cross-domain robustness in mixeddegradation Re-ID.
Our main contributions are summarized as threefold:

and self-distillation have been explored in visible-infrared Re-ID to
exploit complementary modality cues [34], while weakly supervised
collaborative consistency learning reduces reliance on paired crossmodality annotations [35]. From a domain generalization perspective,
leveraging unpaired samples has proven effective for improving crossdomain robustness in Re-ID [36]. Related alignment strategies also
appear in unregistered infrared-visible image fusion [37], underscoring the importance of alignment-aware representation learning under
heterogeneous visual conditions.
Similar robustness challenges have been studied in other vision
tasks, such as action recognition under occlusion and complex backgrounds [38,39], emphasizing the importance of structural consistency
modeling for degradation-tolerant representations. However, existing
approaches primarily address specific degradation types or related heterogeneous scenarios. In contrast, our work focuses on weather-induced
mixed degradations, aiming to jointly enhance structural consistency and
cross-domain alignment for robust person Re-ID in complex real-world
environments.

• We introduce a Mixed-Degradation Consistency Distillation
(MDCD) framework for person Re-ID under mixed-degradation
conditions, enabling robust ID representation learning across
heterogeneous degradations.
• We design a unified learning strategy that integrates a Structural
Consistency Distillation (SCD) module for joint distributional and
structural alignment with a Clean Feature Restoration (CFR) module that preserves clean-image discriminability through elastic
regularization.
• We construct two synthetic benchmarks, Mixed-Market1501 and
Mixed-MSMT17, incorporating diverse weather degradations with
varying severities, and conduct extensive experiments to demonstrate the effectiveness and generality of the proposed framework.
2. Related work

2.2. Knowledge distillation

2.1. Degraded person Re-ID

Knowledge distillation methods can generally be categorized into
logit distillation and feature distillation. Logit distillation trains a student
model to mimic the class-prediction behavior of a teacher model by
aligning their logit distributions. Hinton et al. [40] minimize the divergence between softmax outputs using Kullback–Leibler divergence
(KLD), while Zhao et al. [41] identify limitations of this coupled formulation and propose decoupled knowledge distillation (DKD), which
separately models binary and multi-class logit losses for target and
non-target classes.
In contrast, feature distillation transfers intermediate representations by encouraging the student model to emulate hidden-layer features of the teacher. Chen et al. [42] propose Wasserstein contrastive
representation distillation (WCoRD), which combines global and local
contrastive losses: the global objective minimizes mutual information
between teacher and student distributions via the dual form of the
Wasserstein distance (WD), while the local objective aligns feature sets
at the penultimate layer using WD. Lohit et al. [43] further develop a
cross-instance matching approach based on EMD and IPOT to compute
a discrete WD via an inexact proximal optimal transport algorithm, and
Lv et al. [44] model features with Gaussian distributions to enable efficient closed-form distillation. Additionally, techniques such as network
pruning [45] have been explored to improve computational efficiency
alongside distillation methods.
Inspired by these studies, our approach adopts a unified Fused
Gromov-Wasserstein objective to simultaneously model global distribution alignment and local structural consistency under mixeddegradation conditions.

Person Re-ID under various degradation conditions [8–11] has
gained significant attention in recent years. Extensive experiments
in [23] reveal a strong correlation between cross-dataset generalization
and robustness to degradation, which led to the development of the
baseline method CIL [23]. ISM [13] adopts a teacher-student paradigm,
where clean images guide the feature extraction of hazy samples,
and a domain discriminator encourages haze-invariant representations.
SJDL [8] proposes a joint dehazing and Re-ID network, where a
shared feature extractor simultaneously supports image restoration and
ID prediction within a multi-task framework. RVSL [9] introduces
a self-supervised domain translation approach that transfers cleanimage styles to the foggy domain while preserving ID semantics,
effectively mitigating haze-induced domain shifts. Task-agnostic image restoration methods have also been explored to handle diverse
degradations [24]. In low-light settings, ARN [25] explores adversarial
image recovery for Re-ID by reconstructing low-light images to recover
identity-discriminative cues. Unlike restoration-based pipelines, our
work focuses on feature-level alignment and consistency learning to
enhance robustness under weather-induced mixed degradations.
Beyond restoration-based approaches, several studies explore
feature-level strategies to improve degraded person Re-ID. Huang et al.
[11] enhance robustness by disentangling content and degradation features, while Lu et al. [26] introduce a dual-branch framework combining illumination enhancement with online distillation. Pang et al. [27]
propose a dual-resolution fusion network for cross-resolution Re-ID.
More recently, CFAT [28] addresses corruption-invariant Re-ID through
a teacher-student transformer with Brownian bridge generation, and
DM-OSNet [29] improves day-night vehicle Re-ID by modulating feature distributions without requiring paired training data. Robustness
to semantic noise has also been explored, with ICLR [30] refining
unreliable labels via instance credibility estimation to improve stability
under imperfect supervision. Zhao et al. [31] construct NightReID and
propose EDA, which enhances, denoises, and aligns nighttime features
for low-light person Re-ID. From a different perspective, SCI [32]
improves robustness under severe appearance variation by explicitly
disentangling clothing-variant semantics from ID-stable semantics and
injecting purified textual cues into visual feature learning.
In addition to image degradations, heterogeneous visual conditions such as cross-modality sensing and domain shifts have been
studied for robust Re-ID. Liu et al. [33] propose a practical mixedmodality person Re-ID paradigm, demonstrating that retrieval in mixed
visible-infrared galleries introduces a distinct modality confusion problem, which is addressed through cross-identity discrimination harmonization and bridge-based similarity optimization. Feature fusion

3. Proposed method
We propose Mixed-Degradation Consistency Distillation (MDCD), a
plug-and-play framework that integrates two feature-level modules into
standard person Re-ID models to address mixed-degradation scenarios
(see Fig. 2). During training, paired clean and degraded samples are
constructed for each ID and fed into two parallel branches with identical backbone and head architectures but separate parameters. Features
extracted before BNNeck are used as the representation space. On these
features, the proposed Structural Consistency Distillation (SCD) module
performs Fused Gromov-Wasserstein (FGW)-based clean-to-degraded
alignment, while the Clean Feature Restoration (CFR) module applies
elastic weight consolidation (EWC) regularization (see Sections 3.1,
3.2). The clean-image branch is introduced only during training as
an auxiliary reference, providing more stable structural and semantic
guidance for the degraded-image branch, thereby enhancing its robustness under mixed degradation conditions. The ‘‘Encoder’’ refers
3

Pattern Recognition 179 (2026) 113938

S. Zhao et al.

Fig. 2. Overview of the proposed MDCD framework. The framework includes two modules: (1) the SCD module, which aligns cross-domain feature distributions
using Wasserstein (WD) and Gromov-Wasserstein (GWD) distances while preserving intra-domain structure; and (2) the CFR module, which introduces a
regularization term to constrain variations in parameters critical for clean-image recognition.

Fig. 3. Comparison of Wasserstein Distance (WD) and Gromov-Wasserstein Distance (GWD). WD (left) aligns clean (green) and degraded (yellow) domains by
transporting samples based on pairwise cost 𝐶(𝑥1 , 𝑦1 ), reducing the global distribution gap. GWD (right) preserves intra-domain topology by measuring structural
discrepancy 𝐿(𝑥1 , 𝑥2 , 𝑦1 , 𝑦2 ) between pairwise relations. Squares and circles represent different IDs (ID1 and ID2), and triangles indicate structural relations. (For
interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)

to the feature extractor of the underlying Re-ID model. For instance,
when instantiated with TransReID [46], the Encoder corresponds to its
feature extraction module, and MDCD operates on the resulting features
before BNNeck. During inference, only the degraded-image branch is
used for retrieval, which aligns with the degraded-input target scenario
and introduces no additional computational branches at test time.

and intra-domain structural consistency within a unified optimal transport objective. This formulation produces representations that are both
globally aligned and structurally consistent across domains.
Given a mini-batch of size 𝐵, let {𝑥𝑖 }𝐵
and {𝑦𝑗 }𝐵
denote feature
𝑖=1
𝑗=1
vectors from the clean and degraded branches, respectively. Their
empirical measures are defined as:

3.1. Structural consistency distillation module

𝜇=

Aligning clean (green) and degraded (yellow) feature representations involves two complementary challenges: correcting global domain
shifts and preserving ID-wise geometric structure. As illustrated in
Fig. 3, the Wasserstein Distance (WD) aligns two domains by transporting samples based on cross-domain costs, thereby reducing the
global distribution discrepancy. However, WD does not account for
relational structures within each domain. Under mixed degradations,
samples belonging to the same ID may vary significantly in appearance,
which could lead WD to transport degraded samples toward visually
similar but incorrect IDs, resulting in ID confusion. In contrast, the
Gromov-Wasserstein Distance (GWD) aligns relational structures between domains by matching pairwise relations such as (𝑥1 , 𝑥2 ) and
(𝑦1 , 𝑦2 ), thus preserving ID-wise geometry. However, since GWD focuses
on structural relations rather than absolute feature locations, the global
shift between domains may remain unresolved, and corresponding
samples such as 𝑥1 and 𝑦1 can still be distant in the feature space.
To address both limitations, we adopt the Fused Gromov-Wasserstein
Distance (FGW), which jointly models cross-domain feature alignment

where 𝛿𝑥𝑖 and 𝛿𝑦𝑗 denote Dirac measures centered at 𝑥𝑖 and 𝑦𝑗 . We
define the cross-domain cost 𝑀𝑥𝑦 (𝑖, 𝑗) = ‖𝑥𝑖 −𝑦𝑗 ‖2 and the intra-domain
costs 𝑀𝑥 (𝑖, 𝑗) = ‖𝑥𝑖 − 𝑥𝑗 ‖2 and 𝑀𝑦 (𝑙, 𝑘) = ‖𝑦𝑙 − 𝑦𝑘 ‖2 . A transport matrix:
{
}
𝑇 ∈ 𝛱 (𝜇, 𝜈) = 𝑇 ∈ R𝐵×𝐵
∣ 𝑇 𝟏 = 𝜇, 𝑇 ⊤ 𝟏 = 𝜈 ,
(2)
+

1 ∑
𝛿 ,
𝐵 𝑖=1 𝑥𝑖
𝐵

1 ∑
𝛿 ,
𝐵 𝑗=1 𝑦𝑗
𝐵

𝜈=

(1)

couples the two distributions under the marginal constraints. For a
given transport plan 𝑇 , the structural mismatch term is defined as:
∑|
|2
𝐴𝑖𝑗 (𝑇 ) =
(3)
|𝑀𝑥 (𝑖, 𝑗) − 𝑀𝑦 (𝑙, 𝑘)| 𝑇𝑙𝑘 .
|
|
𝑙,𝑘

The FGW objective is formulated as:
∑
[
]
FGW (𝜇, 𝜈) = min
𝑇𝑖𝑗 𝛼𝑀𝑥𝑦 (𝑖, 𝑗) + (1 − 𝛼) 𝐴𝑖𝑗 (𝑇 ) ,
𝑇 ∈𝛱(𝜇,𝜈)

(4)

𝑖,𝑗

where 𝛼 ∈ [0, 1] balances global alignment (𝛼 = 1 recovers WD) and
structural consistency (𝛼 = 0 recovers GWD).
To integrate FGW into the training process, we adopt the entropyregularized Sinkhorn algorithm [20]. The cost matrix and kernel are
4

Pattern Recognition 179 (2026) 113938

S. Zhao et al.

4. Experimental results

defined as:
𝐶𝑖𝑗 = 𝛼𝑀𝑥𝑦 (𝑖, 𝑗) + (1 − 𝛼) 𝐴𝑖𝑗 (𝑇 ) ,
(
)
𝐶𝑖𝑗
𝐾𝑖𝑗 = exp −
, 𝑖, 𝑗 = 1, … , 𝐵,
𝜖

(5)
4.1. Dataset and implementation details
(6)
4.1.1. Degradation protocol
Due to the lack of benchmarks for mixed-degradation scenarios,
we construct two synthetic datasets, Mixed-Market1501 and MixedMSMT17. Based on Market1501 [21] and MSMT17 [22], we follow the
corruption procedure in [23], where each image is degraded by one of
four degradation types (fog, rain, snow, or brightness). For each image,
the degradation type is sampled uniformly at random, and the severity
level is drawn from five predefined levels. This procedure generates
diverse visual conditions for the training, query, and gallery sets while
preserving the original ID labels and bounding boxes.

where 𝜖 > 0 is a regularization parameter. Given target marginals
𝜇, 𝜈 ∈ R𝐵 (initialized as uniform distributions), the Sinkhorn algorithm
iteratively updates scaling vectors 𝑢, 𝑣 ∈ R𝐵 :
𝜇
𝜈
𝑢←
, 𝑣← ⊤ ,
(7)
𝐾𝑣
𝐾 𝑢
until the marginal constraints 𝑇 𝟏 = 𝜇 and 𝑇 ⊤ 𝟏 = 𝜈 are satisfied. The
optimal transport matrix is then obtained as:
𝑇 ⋆ = diag (𝑢) 𝐾diag (𝑣) .

(8)

With 𝑇 ⋆ , the FGW loss used in training becomes:
FGW =

∑

[
]
𝑇𝑖𝑗⋆ 𝛼𝑀𝑥𝑦 (𝑖, 𝑗) + (1 − 𝛼) 𝐴𝑖𝑗 (𝑇 ⋆ ) ,

(9)

4.1.2. Statistical summary
As illustrated in Fig. 4, each degradation type and severity level is
uniformly sampled, ensuring balanced coverage across corruption factors. This sampling strategy provides a controlled and diverse testbed
for evaluating model robustness under heterogeneous degradation conditions.

𝑖,𝑗

which simultaneously promotes global distribution alignment and
structural consistency between clean and degraded feature spaces.
3.2. Clean feature restoration module
Degraded images often contain noise, blur, and other artifacts that
obscure fine-grained discriminative cues present in clean images. Although training with degraded data improves robustness, it may also
cause the model to forget discriminative knowledge learned from clean
samples. To address this issue, we introduce the Clean Feature Restoration (CFR) module based on elastic weight consolidation (EWC), a
Fisher-information-based regularization technique [47]. By penalizing
parameter changes that significantly affect clean-image performance,
CFR alleviates catastrophic forgetting.
Specifically, the importance of each parameter 𝜃𝑖 is estimated using
clean-domain samples:
[(
)2 ]
𝜕
𝐹𝑖 = E(𝐼,cls)∼𝐷clean
log 𝑝 (cls ∣ 𝐼, 𝜃)
,
(10)
𝜕𝜃𝑖

4.1.3. Visual examples
Fig. 5 presents examples of synthetic degradations with five severity
levels for each weather factor: fog, rain, snow, and brightness. Each
row corresponds to one degradation type, with degradation severity
increasing from level 1 to level 5. The rightmost column shows the
original clean image for reference, illustrating the visual diversity introduced by our protocol. Fig. 5(b) reports the corresponding quantitative
metrics (SSIM and PSNR), which reflect the severity of degradation
across different levels.
4.1.4. Evaluation protocol and metrics
Following [23], we evaluate all methods under four settings: (1)
Clean Evaluation (CE): both query and gallery are clean; (2) Degraded
Evaluation (DE): both query and gallery are degraded; (3) Degraded
Query (DQ): degraded queries with a clean gallery; (4) Degraded Gallery
(DG): clean queries with a degraded gallery. We report mean average
precision (mAP) and cumulative matching characteristic (CMC) metrics
for all experiments.

where 𝑝(cls ∣ 𝐼, 𝜃) denotes the predicted probability of class cls for input
image 𝐼. A larger 𝐹𝑖 indicates that parameter 𝜃𝑖 is more important for
maintaining clean-image performance.
During training on degraded data, an EWC regularization term is
introduced to penalize deviations from the clean-trained parameters 𝜃𝑖∗ :
∑ (
)2
EWC =
𝐹𝑖 𝜃𝑖∗ − 𝜃𝑖 ,
(11)
𝑖

4.1.5. Implementation details
All images are resized to 256 × 128. Data augmentation includes
random horizontal flipping, padding, random cropping, and random
erasing. We use a batch size of 64 with four images per ID. The
optimizer, learning schedule, and training strategy follow the respective
backbone implementations. All experiments are conducted on a single
NVIDIA GeForce RTX 4090 GPU. For the proposed objectives, unless
otherwise specified, the balance coefficient in FGW is set to 𝛼 = 0.1,
the entropy regularization parameter to 𝜖 = 0.8, and the loss weights to
𝜆1 = 1 and 𝜆2 = 10, respectively. Further discussion on the sensitivity
of these hyperparameters is provided in Section 4.6.

where 𝜃𝑖∗ denotes parameters obtained from clean-image training and 𝜃𝑖
denotes the current parameters. This regularization preserves discriminative capability on clean images while allowing the model to adapt
to mixed-degradation conditions.
3.3. Overall architecture
Both modules operate at the feature level without modifying the
backbone architecture, making MDCD inherently backbone-agnostic
and easily compatible with existing Re-ID models. The training objective combines the original Re-ID loss ReID (e.g., identification and
triplet losses) with the proposed objectives, including the FGW alignment loss FGW from the SCD module and the EWC regularization loss
EWC from the CFR module. The overall training objective is formulated
as:
all = ReID + 𝜆1 FGW + 𝜆2 EWC ,

4.2. Comparison with state-of-the-art
Tables 2 and 3 compare the proposed MDCD framework with representative Re-ID methods on Mixed-Market1501 and Mixed-MSMT17.
Existing approaches typically achieve strong performance under CE but
exhibit substantial degradation in DE, DQ, and DG, highlighting their
limited robustness under mixed-degradation conditions.

(12)

where 𝜆1 and 𝜆2 balance the contributions of structural alignment and
feature restoration.
5


codex
第三篇的核心已经很清楚：它不是把换衣简单说成“去衣服”，而是拆成负因素和正因素，强调视觉里还有头发、鞋、包等可用稳定语义。接下来读后面三篇，其中两篇是可见光红外，一篇还是换衣。
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
exec
/bin/zsh -lc "pdftotext -f 4 -l 5 'See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, 2025

4

B. Semantic Separation Enhancement

Algorithm 2: Semantic-Guided Interaction Module

CLIP cannot explicitly ignore unwanted semantics (e.g.,
clothing) via negative prompts. To address this, we propose
the Semantic Separation Enhancement (SSE) module, which
isolates and removes clothing-related components from the
high-dimensional text features, preserving only ID-relevant
semantics. The overall procedure is detailed in Algorithm 1.
We define two learnable prompts:

: Dual textual learnable vectors promptid ,
promptclo , and person images.
txt
Output : Optimized text representations Fort
1 for each epoch e from 1 to epochs do
2
for each batch in trainloader do
3
Extract dual text features Ftxt using the frozen text
encoder T (·) Compute orthogonalized text features
txt
Fort
Extract image features Fimg using the training
visual encoder I(·) Enhance image features with
contextual information using Eq. (13) Compute
semantic-guided image features with Eq. (14) and
txt
Eq. (15) Refine image features based on Fort
using
Eq. (16) Apply loss functions to regulate the visual
encoder training using Eq. (20)

promptid = A photo of a [X]1 [X]2 . . . [X]M person. (1)
promptclo = A photo of the [X]1 [X]2 . . . [X]M clothes.
(2)
where [X]i (i = 1, . . . , M ) are learnable tokens initialized
randomly, and M denotes the number of tokens. In the first
training stage, we use the pre-trained visual encoder I(·) and
text encoder T (·) to extract image and dual text features,
freezing the encoders’ parameters to focus on optimizing
the text tokens [X]i . This allows us to learn contextual
representations by updating the text tokens, thereby acquiring
distinct textual representations for each ID and their clothing:
txt
Fid
= T (promptid ) ,

txt
Fclo
= T (promptclo ) ,

(3)

Input

where λ1 and λ2 are weighting factors controlling the contributions of ID and clothing similarities.
Finally, we calculate the image-to-text contrastive loss Li2t :
txt
exp s (Vi , Fort
(i))
Li2t (i) = − log PN
,
txt
k=1 exp s (Vi , Fort (k))

(9)

txt
txt
txt
where Fid
and Fclo
represent the ID-specific text features where Vi and Fort (i) are paired visual and text embeddings,
s(·, ·) denotes the similarity function, and N is the batch size.
and clothing text features, respectively.
txt
After extracting the dual text representations, we project Fclo Since multiple images in a batch may belong to the same ID,
txt
onto Fid
to reduce the influence of clothing and emphasize meaning there may be multiple positive samples, the text-tothe implicit positive factors. This step clarifies the relation be- image contrastive loss Lt2i (yi ) is calculated as:
tween clothing representations and ID representations, thereby
X
−1
exp s (Vp , Tyi )
Lt2i (yi ) =
log PN
, (10)
minimizing the impact of clothing variations. The computation
|P (yi )|
k=1 exp s (Vk , Tyi )
p∈P (yi )
process is as follows:
txt
Fproj
=

txt txt
Fclo
Fid

F txt ,
txt ∥2 id
∥Fid

(4)

We then perform orthogonalization by subtracting the projection
txt
txt
txt
Fproj
of the clothing feature Fclo
onto the Fid
. This operation
removes the clothing component that is aligned with the ID
txt
direction. This process ensures that the features Fort
are aligned
with the positive features while removing the influence of
negative factors. The orthogonalization process is:
txt
txt
txt
Fort
= Fid
− Fproj
,

(5)

txt
where Fort
is after removing the part that is related to clothing,
ensuring the feature focuses on positive information while
reducing interference from clothing.
To optimize the text features, we design a loss function
txt
txt
that maximizes the similarity between Fort
and Fid
, while
txt
minimizing the similarity with Fclo . This enables the model
to better capture positive factors without being affected by
negative aspects, such as clothing changes. The similarity
calculations are:

txt
txt
simid (i) = mean cos Fort
(i), Fid
(i) ,
(6)

txt
txt
simclo (i) = mean cos Fort
(i), Fclo
(i) ,
(7)

where mean(·) denotes the average value, and cos(·) represents
the cosine similarity function. The loss function is as follows:
Lsim (i) = λ1 (1 − simid (i)) + λ2 simclo (i),

(8)

txt
where P (yi ) is the set of all positive indices for Fort
(yi ) in
the batch, and Tyi is the text embedding corresponding to label
yi . Therefore, the overall loss function is:
X
Lprompt =
(Li2t + Lt2i + Lsim ) .
(11)
i=1

C. Semantic-Guide Interaction Module
Previous work [7], [14] shows that reducing clothing bias
in visual or textual features improves performance in clothchanging scenarios. However, these methods typically treat
visual and textual branches independently, lacking interaction. Our method refines visual representations by leveraging
clothing-irrelevant textual features, enhancing the interaction
between visual and textual branches. This integration introduces
more robust and invariant descriptors into the visual processing,
improving the model’s ability to identify individuals across
clothing changes. The algorithm is summarized in Algorithm 2.
As illustrated in Fig. 3, we employ a Transformer decoder [58] to model interactions between visual and textual data.
To better capture long-range dependencies within the visual
domain, we replace the conventional self-attention mechanism
with a non-local operation [59], which allows the model to
aggregate information from spatially distant but semantically
relevant regions. This capability is particularly important for
handling variations in clothing. During this stage, we update

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, 2025

conv

q
%&'

F"()

v

cross
Attention

conv
%&'
F!"%

k

soft
max

conv


where y i is the true label for the i-th sample, and pid y i | xi is
the predicted probability of the true label y i . The clothes-based
adversarial loss is expressed as:

#$#
F!"#

MLP

Nonlocal Operation

%&'

F!*#

5

%&'

F+%,,

Lcal = −

NC
N
XX

q(c) log

i=1 c=1

Fig. 3. Illustration of the SIM process. Textual information refines visual
feature extraction to align features with the relevant semantic context.

the visual encoder while keeping other components frozen. The
computation is as follows:

 

img
img


θ Fori
ϕ Fori
img
img
Fcon
=
g Fori
,
(12)
N
img
where Fori
is the input feature map, θ(·), ϕ(·), and g(·) are
linear transformations, and N is the number of elements in
img
the feature map. Fcon
captures global context information.
The final visual feature map is:

img
img
img
Fres
= W Fcon
+ Fori
,
(13)
img
where W contains learnable parameters. Fres
is the final output
feature map obtained by adding this context-aware feature to
the original feature map.
To refine visual representations with semantic guidance, we
leverage textual embeddings to generate attention weights over
visual features. This allows the model to selectively emphasize
semantically relevant regions in the visual space, leading to
more discriminative and context-aware visual representations:
!
img txt T
Fres
Fort
img
img
txt
√
Fout = Fres + softmax
Fort
,
(14)
dk



img
img
Fdiff
= MLP LayerNorm Fout
,
(15)

where dk is the dimensionality of the keys, LayerNorm is
used to normalize the input features to stabilize training, and
MLP denotes a two-layer feed-forward network. This integrates
visual features under the guidance of textual embeddings, and
enables the model to extract the most informative visual cues
aligned with the semantic content of the text.
Finally, we update the visual features. This process ensures
that the visual features are enriched with relevant semantic information from the text, leading to more robust and contextually
aware visual representations:
img
img
F img = Fori
+ αFdiff
,

(16)

where α ∈ RC is a learnable parameter controlling the degree
of refinement applied to the visual features.
This module adopts an interaction-to-alignment approach,
where text embeddings guide the visual encoding process. By
emphasizing positive cues and mitigating negative ones, this
method improves the model’s ability to consistently recognize
individuals across different clothing scenarios.
For the objective function, we incorporate cross-entropy loss
Lid and clothes-based adversarial loss Lcal [7] to optimize the
visual encoder:
N
X

Lid = −
y i log pid y i | xi ,
(17)
i=1

exp (fi φc /τ )
P
,
exp (fi φc /τ ) + j∈S − exp (fi φj /τ )
i
(18)

where NC is the number of clothing categories, φc is the
clothes classifier, q(c) is the weight for the c-th class, fi is the
feature of sample i, τ is a temperature parameter, and Si− is
the set of clothes classes with the same ID. The objective of
Lcal is to encourage the backbone to extract clothes-irrelevant
features by penalizing its predictive power regarding clothing.
To fully leverage the capabilities of CLIP, we calculate the
image-to-text cross-entropy loss Li2tce as:
Li2tce (i) =

N
X

txt
exp s(Vi , Fort
(i))
−qk log PN
,
txt
k=1 exp s(Vi , Fort (k))
k=1

(19)

where label smoothing is applied to qk . The total loss for the
SIM module is:
L = Lid + Lcal + Li2tce .

(20)

By jointly optimizing Lid , Lcal , and Li2tce , the model captures
ID features from easy samples (same clothes) and progressively
learns to distinguish hard cases (same ID, different clothes)
while aligning visual and textual representations. The weights
of loss terms are basic terms [7], [14] and set as 1.
IV. E XPERIMENTAL R ESULTS
A. Datasets and Evaluation Metrics
We evaluate our method on three standard cloth-changing
datasets: LTCC [28], PRCC [27], and VC-C LOTHES [30],
following the protocols of CAL [7] and AIM [8]. Additionally,
to validate generalization, we test on two conventional ReID
datasets: M ARKET 1501 [60] and MSMT17 [61].
LTCC contains 17,138 images of 152 IDs captured by 12
cameras. The training set comprises 9,576 images of 77 IDs;
the testing set includes 7,543 images (493 queries and 7,050
gallery) of 75 IDs. This long-term dataset features frequent
clothing changes and varied environmental conditions, with
each ID wearing between two and fourteen outfits.
PRCC consists of 33,698 images of 221 IDs from three
cameras. Its training set has 22,898 images of 150 IDs, and the
testing set has 10,800 images of 71 IDs. Each person appears
in two outfits: cameras A and B share the same clothing, while
camera C uses a different outfit.
VC-C LOTHES is a synthetic dataset generated via the GTA5
engine. It contains 9,449 images of 256 IDs across four camera
views, with 1,241 distinct clothing items distributed between
training and testing sets.
M ARKET 1501 includes 12,936 training images of 751 IDs
and 19,732 testing images of 750 IDs, captured by six cameras
under the single-query evaluation protocol.
MSMT17 comprises 32,621 training images of 1,041 IDs
and 51,027 testing images of 11,659 IDs across 15 cameras, it
is one of the largest and most challenging ReID benchmarks.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11209289

Semantic Alignment and Hard Sample Retraining
for Visible-Infrared Person Re-Identification
1

Jingchen Ni1,* , Keyu Lyu1,* , Yu Guo2 , Chun Yuan1,†
Shenzhen International Graduate School, Tsinghua University, 2 CEC GienTech Technology Co.,Ltd.
*
{njc24, lvky24}@mails.tsinghua.edu.cn, † yuanc@sz.tsinghua.edu.cn

Abstract—Visible-Infrared Person Re-Identification (VI-ReID)
seeks to match individuals across different modalities. Recent
methods focus on discriminative feature extraction and hard
sample learning. However, they often suffer from semantic
misalignment due to horizontal partitioning in local feature extraction and overlook global hard samples in training. Moreover,
the widely used PK Sampler cannot ensure viewpoint balance and
diversity. To overcome these limitations, we propose the Semantic
Alignment and Hard Sample Retraining (SAHSR) framework.
This framework incorporates a Recurrent Semantic Aggregation
(RSA) module that progressively aggregates and aligns regional
semantics with the help of Modality Alignment loss. Besides, we
propose a Confidence-based Hard Sample Retraining (CHSR)
strategy that identifies and retrains hard samples to improve the
model’s robustness. Additionally, we introduce the ViewpointBalanced (VB) Sampler to guarantee a balanced distribution
of viewpoints. Extensive experiments on VI-ReID benchmarks
demonstrate the significant performance gains of our approach,
showing state-of-the-art performance. Code will be available.
Index Terms—Visible-Infrared Person Re-Identification, Semantic Alignment, Hard Sample Learning, Viewpoint Balance

I. I NTRODUCTION
Person re-identification (ReID) is a pedestrian retrieval
task that matches individuals across multiple non-overlapping
cameras, essential for tracking, security, and forensics in
video surveillance. While visible-spectrum ReID [1], [2] has
advanced under good lighting conditions, its performance
declines in low-light or nighttime scenarios due to limited
discriminative features. To address this, visible-infrared ReID
(VI-ReID) integrates infrared imagery, enabling recognition in
challenging lighting environments by extending ReID across
visible and infrared modalities. However, this introduces intermodality discrepancies caused by the inherent differences
between visible and infrared data.
These cross-modal gaps are further compounded by several
additional challenges. In particular, semantic misalignment
arises when corresponding body regions fail to spatially align
across modalities due to variations in pose, viewpoint, and
imaging characteristics. Existing methods [3]–[5] primarily
utilize horizontal partitioning for feature extraction, treating
patches independently and thereby overlooking broader contextual cues necessary for precise alignment. This exacerbates
semantic misalignment, and results in subtle yet critical mismatches that diminish the discriminative power of the learned
features, as illustrated in Figure 1(c).
* Equal Contribution. † Corresponding Author.

Fig. 1.
Framework Comparison: (a) Our method incorporates CHSR
Strategy, enhancing robustness by retraining on hard samples, unlike previous
methods that rely solely on loss calculation without additional retraining steps.
(b) Rectangle colors indicate different camera labels. The Viewpoint-Balanced
(VB) sampler ensures balanced sample distribution across camera views,
unlike the conventional PK sampler. (c) For feature extraction, our method
progressively aggregates local features using RNNs, effectively avoiding
semantic misalignment and ensuring better alignment of features across the
entire feature map.

Compounding these alignment issues is the underemphasis on globally challenging instances. Although triplet-based
losses [6], [7] are designed to emphasize hard samples, their
scope is generally confined to the mini-batch level and often neglects hard samples scattered throughout the dataset.
Without a more holistic mechanism to identify and revisit
these global complex cases, the model remains vulnerable to
similar ambiguities and fails to fully exploit the discriminative
potential of the entire dataset.
A further issue is viewpoint imbalance, where uneven
camera viewpoints bias the model’s representations. Although
PK samplers [6] include both visible and infrared samples in
each batch, they do not ensure balanced viewpoints, as shown
in Figure 1(b). This leads to overfitting common viewpoints
and limits generalization to rare ones.
Taken together, these considerations highlight the necessity
of methods that not only achieve more coherent semantic
alignment but also systematically emphasize globally challenging samples and ensure a balanced distribution of viewpoints.
Addressing these issues is critical for advancing VI-ReID toward more robust, discriminative, and generalizable solutions.
To address these challenges, we propose the Semantic

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:03 UTC from IEEE Xplore. Restrictions apply.

Alignment and Hard Sample Retraining (SAHSR) framework,
comprising three components: the Recurrent Semantic Aggregation (RSA) module, the Confidence-based Hard Sample
Retraining (CHSR) strategy, and the Viewpoint-Balanced (VB)
Sampler.
The RSA module improves feature extraction by progressively aligning regional semantic features across the feature
map using a patch-based RNN, assisted by the Modality Alignment Loss to address semantic misalignment. The CHSR strategy selects hard samples globally based on confidence scores
and retrains them, enabling the model to better differentiate
between a sample and its hard negative samples. Meanwhile,
the VB Sampler balances sampling across cameras in both
modalities, reducing viewpoint discrepancies and improving
the model’s stability.
In summary, this paper makes the following contributions:
1) We introduce the SAHSR framework to simultaneously address semantic misalignment, hard sample learning, and viewpoint imbalance in VI-ReID. 2) We propose RSA, CHSR, and
VB components that respectively promote semantic alignment,
ensure thorough re-exposure to globally challenging samples,
and balance camera viewpoints. 3) Extensive experiments on
two standard VI-ReID benchmarks demonstrate that SAHSR
significantly outperforms state-of-the-art methods, confirming
its effectiveness and practical value.
II. R ELATED W ORK
Visible-Modality ReID. Person re-identification (ReID) in
the visible spectrum aims to match individuals across multiple non-overlapping cameras. Early approaches relied on
handcrafted features like SIFT [8] and metric learning [9],
which lacked robustness. The advent of deep learning, including CNN-based and transformer-based models [2], [10],
significantly enhanced discriminative representation learning.
However, these methods still struggle under poor illumination.
Visible-Infrared ReID. Visible-Infrared ReID (VI-ReID) incorporates infrared imagery to mitigate illumination issues,
facilitating recognition under varying lighting conditions. Approaches often focus on modality-invariant representations to
bridge the visible-infrared gap. For instance, Fang et al. [11]
and Zhang et al. [12] prioritize robust feature alignment, while
Ren et al. [13] use contrastive learning to reduce modality
discrepancies. Despite these advancements, existing VI-ReID
methods often overlook fine-grained semantic misalignments,
fail to address globally challenging samples, and do not ensure
balanced viewpoint distribution, highlighting the need for
more comprehensive solutions.
III. PROPOSED METHODS
The overall flow of SAHSR is shown in Figure 2. Given a
set of visible samples and infrared samples, the VB Sampler
is first used to sample P IDs, with each ID having K images,
forming a mini-batch of viewpoint-balanced samples (X m , Y )
where m ∈ {V, I} denotes the modality, with V representing
visible and I representing infrared, for training. After the
preprocessing step, a backbone network is exploited to extract

Fig. 2. The framework of the proposed SAHSR consists of three key
components: the VB Sampler, RSA Module, and CHSR Strategy.

a set of feature maps F m . Then the RSA module is utilized
to obtain global and local representations from the feature
maps for the subsequent ReID task. Finally, the CHSR strategy
selects hard samples based on confidence scores and retrains
them to further improve the model’s performance.
A. Recurrent Semantic Aggregation
The Recurrent Semantic Aggregation (RSA) module enhances feature extraction by progressively aggregating local
patch information via a BiLSTM. Unlike methods that simply
treat patches independently or rely on global attention, the
LSTM effectively models these patches as a short sequential
series, preserving their inherent spatial order. This ordersensitive modeling helps capture local dependencies—such
as the natural top-to-bottom arrangement of human body
parts—leading to more coherent and discriminative feature
representations. In contrast, Transformer-based approaches,
though powerful in global context modeling, may dilute such
local sequential cues, especially when dealing with a small
number of patches. By leveraging LSTM’s strength in handling
short sequences and local dependencies, RSA provides richer
contextual information that benefits the subsequent ReID tasks.
Visible samples are used to illustrate the process for ease of
explanation.
Specifically, after processing sample xV through the backbone, feature maps F V are obtained. To facilitate patchlevel analysis, the feature map is divided into n patches
by splitting it into a grid of size nh × nw , where nh and
nw represent the number of patches along the height and
width dimensions, respectively, satisfying n = nh × nw . Each
patch F V (k) corresponds to a spatially contiguous region of
the original feature map. For each patch, adaptive average
pooling (AAP) is applied to aggregate its spatial information
into a compact representation, yielding patch-wise features
f V (i) = avg(F V (i)), where f V (i) ∈ Rc .
Subsequently, a Bidirectional Long Short-Term Memory
(BiLSTM) network is employed to aggregate the local features, effectively capturing the interrelationships among them.
oV = BiLSTM(f V (1), f V (2), . . . , f V (n)),

(1)

where oV = [oV1 , oV2 , · · · , oVT ] ∈ RT ×2c represents the BiLSTM output at each time step, and T denotes the sequence
length. We utilize the output of the last time-step oVT to

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:03 UTC from IEEE Xplore. Restrictions apply.

V
calculate the final aggregated local feature, fpart
= W (oVT ),
where W is the learnable parameters of the fully-connected
layers.
To capture global features, global average pooling (GAP) is
applied to F V . The resulting global features are concatenated
V
with fpart
to obtain fˆV as shown in (2).
V
fˆV = [fpart
, GAP(F V )],

(2)

where fˆV combines both global and local features. A
similar process is applied to the infrared modality to extract
fˆI from F I . These fused features, fˆV and fˆI , are then passed
through a dual BNNeck [11] for normalization. This step follows the calculation of the Modality Alignment Loss, ensuring
that the features from the visible and infrared modalities are
effectively aligned.
To further reduce the inter-modality discrepancy, we introduce a custom Modality Alignment Loss (Lma ), which
is designed to progressively minimize the Kullback-Leibler
(KL) divergence between the feature probability distributions
of the two modalities at each time step. Unlike static alignment strategies, this approach incorporates a time-dependent
weighting factor g(t), which increases linearly over time. This
design prioritizes stronger alignment constraints during the
later stages of the sequence, as we hypothesize that aligning
modalities at these stages is crucial for effective modality
integration.
Specifically, we first calculate the inter-modality distance
matrix DtV I using the Euclidean distance between oVt and oIt :
DtV I = ∥oVt − oIt ∥2 ,

DtIV = (DtV I )T

(3)

Next, we compute the probability distributions AVt I and
by applying the softmax function to the negative distance
AIV
t
matrices DtV I and DtIV , respectively. The resulting probability
distributions are then used to formulate the Modality Alignment Loss Lma , as shown in Equation (4).

Lma =

T
−1
X


g(t) KL(AVt I ∥ AIV
t ) ,

g(t) =

t=0

t
, (4)
T −1

where KL(AVt I ∥ AIV
t ) represents the KL divergence between
AVt I and AIV
,
and
g(t) is the linear weighting factor that
t
increases from 0 to 1 over the sequence. By assigning higher
weights to later steps, we enforce a stronger inter-modality
alignment when the local contextual cues are more consolidated. Ablation studies in Section IV-D show that this dynamic
weighting outperforms uniform weighting schemes, validating
our time-dependent alignment strategy.
B. Confidence-based Hard Sample Retraining
Previous research [6], [7], [19] has demonstrated that enhancing a model’s ability to discriminate hard samples significantly improves its overall performance. Unlike traditional
hard sample mining techniques that operate at the mini-batch
level, our Confidence-based Hard Sample Retraining (CHSR)
strategy leverages confidence scores to dynamically identify

hard samples across the entire dataset. This global focus
on challenging instances enhances the model’s generalization
capabilities.
To implement CHSR, we divide the training process into
standard training and retraining phases. During standard training, after the RSA module, feature vectors fˆm are generated
and fed into a classifier that produces confidence scores
S ∈ R(P ×K)×N , where P is the number of identities in a
batch, K is the number of images per identity, and N is the
total number of classes (i.e., the total number of IDs in the
training set).
We organize the batch such that the images of each ID
are grouped consecutively: the first K rows correspond to the
first ID, the next K rows to the second ID, and so forth. To
reference the confidence vector for the j-th image of ID i, we
define an indexing function:
r(i, j) = (i − 1) × K + j,

(5)

where i ∈ {1, . . . , P } and j ∈ {1, . . . , K}.
With this notation, S(r(i, j)) ∈ RN denotes the confidence
score vector for the j-th image of ID i. To obtain the mean
confidence score vector for ID i, we compute:
K

S(i) =

1 X
S(r(i, j)),
K j=1

(6)

where S(i) ∈ RN represents the mean confidence score over
the K images of the i-th ID.
Let c ∈ {1, . . . , N } index the classes. Then S(i)[c] denotes
the average probability that ID i is classified as class c. If
certain classes c ̸= i attain relatively high values in S(i),
this indicates that ID i is prone to be misclassified as these
classes, suggesting that images of these classes may serve as
hard negative samples for ID i. We define the set Cm (i) as the
m classes with the highest values of S(i)[c] (including c = i).
This set comprises ID i and its highly similar IDs, which will
be used for further retraining.
For each identity i in the standard training batch of P
identities, we define a Hard Sample Batch Hi , which contains
m × K̂ images. Specifically, for each ID i, we first select
all m IDs from the set Cm (i). Then, for each of these IDs,
we randomly select K̂ images. The collected images form the
batch Hi :
Hi = {xc,j | c ∈ Cm (i), j = 1, . . . , K̂}.

(7)

Since there are P identities in the standard training batch,
this process results in P distinct Hard Sample Batches
{H1 , H2 , . . . , HP }, each tailored to the corresponding identity’s challenging hard negative samples.
In the retraining phase, the P Hard Sample Batches are
concatenated into a unified batch and fed into the network.
To specifically enhance the model’s ability to distinguish each
identity from its hard negatives, we introduce two additional
loss functions: the Hard Batch Identity Loss (Lhid ) and
the Hard Batch Center Separation Loss (Lhcs ), which are
computed on Hi .

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:03 UTC from IEEE Xplore. Restrictions apply.

TABLE I
C OMPARISON WITH THE STATE - OF - THE - ART METHODS ON SYSU-MM01 AND R EG DB.

SYSU-MM01
Method

AGW [14]
WGCN [15]
MSCLNet [16]
DEEN [12]
DMA [17]
SAAI [11]
IDKL [13]
YYDS [18]
SAHSR (ours)

Venue

TPAMI’21
TIFS’24
ECCV’22
CVPR’23
TIFS’24
ICCV’23
CVPR’24
Arxiv’24
-

All-Search

RegDB

Indoor-Search

Visible2Infrared

Infrared2Visible

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

47.50
66.82
76.99
74.70
74.57
75.90
81.42
85.54
86.26

47.65
64.88
71.64
71.80
70.41
77.03
79.85
81.64
83.91

54.17
72.64
78.49
80.30
82.85
83.20
87.14
89.13
91.44

62.97
76.19
81.17
83.30
85.10
88.01
89.37
91.00
92.17

70.05
90.61
84.17
91.10
93.30
91.07
94.72
92.67

66.37
84.53
80.99
85.10
88.34
91.45
90.19
93.66

70.49
88.77
83.86
89.50
91.50
92.09
94.22
90.20
92.77

65.90
81.61
78.31
83.40
86.80
92.01
90.43
83.50
94.26

For each Hard Sample Batch Hi , the classifier produces
confidence scores SHi ∈ R(m×K̂)×N . To compute the Hard
Batch Identity Loss, we retain only the scores for classes in
Cm (i), resulting in ŜHi ∈ R(m×K̂)×m .
The original label vector for Hi has length m × K̂ with
m unique values. These labels are mapped to {1, . . . , m}
based on their position in Cm (i) and converted into a onehot encoded matrix YHi ∈ {0, 1}(m×K̂)×m . The Hard Batch
Identity Loss (Lhid ) is then calculated as the cross-entropy
loss between ŜHi and YHi :

of training batches. Traditional PK sampling strategies [6] can
lead to imbalanced viewpoints within a batch, causing the
model to under-perform on certain viewpoints, as shown in
Figure 1(b). Inspired by [20], the VB sampler selects P IDs,
with each ID having K images. Within the same modality,
the number of images from different camera viewpoints is
equal, thereby enhancing viewpoint diversity and mitigating
viewpoint discrepancy.

D. Total Loss Function
During standard training, the model is optimized using the
identity
loss (Lid ) [21], the Center Separation loss (Lcs ) [11],
m×
K̂
m
P


X X X
1
YHi (k, c) log ŜHi (k, c) , and the modality alignment loss Lma . The standard training
Lhid = −
P × m × K̂ i=1 k=1 c=1
loss is defined as:
(8)
Lstd = Lid + Lcs + λ1 Lma .
(9)
where YHi (k, c) is the one-hot encoded label for sample k in
Hi belonging to class c, and ŜHi (k, c) is the confidence score
After epoch ϵ, we start to use the CHSR strategy, whose
for sample k and class c after filtering to Cm (i).
training iterations alternately switch between standard training
Additionally, we apply the Hard Batch Center Separation and retraining. In the retraining phase, we incorporate the Hard
Loss (Lhcs ) on each Hi . This loss computes the center Batch Loss (Lh ). The retraining loss is defined as:
separation loss [11] on Hi , allowing the model to focus
Lr = Lstd + λ2 Lh .
(10)
exclusively on distinguishing ID i from its hard negatives
without interference from easy samples, thereby improving
IV. E XPERIMENT
its discriminative capability. Finally, the Hard Batch Loss is A. Datasets and Evaluation Metrics
defined as Lh = Lhid + Lhcs .
We evaluate the proposed SAHSR on two benchmark
The CHSR strategy is activated once the training epoch
datasets, SYSU-MM01 [22] and RegDB [23], using Rank-1
exceeds a predefined threshold ϵ. Prior to ϵ, only standard
accuracy (R1) and mean average precision (mAP) as metrics.
training is employed. After ϵ, the training iterations alternate between standard training and retraining phases. This B. Implementation Details
alternating approach allows the model to benefit from both
We employ the ResNet-50 as the backbone. For each
conventional learning and focused refinement on hard samples. batch, 12 identities are randomly selected, with 12 images
A detailed description of the CHSR strategy is provided in corresponding to each identity. The images are resized to
Algorithm 1 in the supplementary material.
288 × 144. The network is optimized using Adam. The initial
learning rate of 3.5 × 10−4 is reduced by 0.1 and 0.01 at the
C. Viewpoint-Balanced Sampler
80th and 120th epochs. Training lasts for 160 epochs. For the
To mitigate viewpoint discrepancies caused by data imbal- hyperparameters in (9) and (10), λ1 is set to 0.2 and λ2 is set
ance, we designed the Viewpoint-Balanced (VB) Sampler. The to 1. CHSR threshold epoch ϵ is set to 40. Following [13],
VB Sampler ensures diversity and balance in the viewpoints [18], we use re-ranking [24] for evaluation.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:03 UTC from IEEE Xplore. Restrictions apply.

TABLE II
A NALYSIS OF THE PROPOSED MODULES ON SYSU-MM01 UNDER THE
ALL - SEARCH AND SINGLE - SHOT MODE .

Modules
1
2
3
4
5
6

Loss

Metrics

Base

RSA

CHSR

VB

Lma

Lh

Rank-1

mAP

✓
✓
✓
✓
✓
✓

×
✓
✓
✓
✓
✓

×
×
×
✓
✓
✓

×
×
×
×
×
✓

×
×
✓
✓
✓
✓

×
×
×
×
✓
✓

80.65
83.31
83.80
84.80
85.16
86.26

77.44
81.45
81.97
82.55
83.42
83.91

TABLE III
D IFFERENT RSA T YPE TEST ON THE SYSU-MM01 DATASET UNDER
ALL - SEARCH MODE .

RSA Type

Rank-1

mAP

Decoder
Encoder
LSTM
BiLSTM

84.51
83.76
85.12
86.26

82.63
82.76
82.94
83.91

C. Comparison Results
As shown in Table I, our model (SAHSR) achieves state-ofthe-art performance across all scenarios on the SYSU-MM01
dataset, with mAP improvements of 2.27% and 1.17% in
the all-search and indoor-search single-shot settings, respectively. On the RegDB dataset, SAHSR attains state-of-the-art
mAP scores of 93.66% (Visible2Infrared) and 94.26% (Infrared2Visible), along with R1 scores of 92.67% and 92.77%,
respectively, validating its effectiveness.
D. Ablation Study
In this subsection, we present the key results of our ablation
study to evaluate the effectiveness of the proposed model
components. More detailed experimental results are provided
in the supplemental material.
1) Effectiveness of Each Component: Table II demonstrates
the contribution of each component. Compared to the baseline,
adding RSA with Lma improves R1 and mAP by 3.15% and
4.53%, respectively. Incorporating the CHSR strategy and Lh
further increases R1 by 4.51% and mAP by 5.98%. The full
combination of RSA, CHSR, and VB achieves the highest
gains of 5.61% in R1 and 6.47% in mAP, underscoring the
importance of each module.
2) Hyper-parameters Analysis: Figure 3 presents an ablation study on the hyperparameters λ1 , λ2 , ϵ, and the number of

Fig. 3. Ablation Study of Hyper-parameters on the SYSU-MM01 Dataset
under all-search mode.

Fig. 4. Ablation Study of Different Lma Strategies on the SYSU-MM01
Dataset under all-search mode.

patches n using the SYSU-MM01 dataset in all-search mode.
Subfigure (a) shows that λ1 = 0.2 maximizes both R1 and
mAP, with performance declining for higher values. Similarly,
subfigure (b) indicates that λ2 = 1 achieves optimal results,
while larger λ2 slightly reduces performance. Subfigure (c)
demonstrates that ϵ = 40 epochs yields the highest R1 and
mAP, with higher ϵ introducing noise that degrades feature
representation. Finally, subfigure (d) reveals that n = 6 (grid
size nh = 3, nw = 2) offers the best balance between
feature fragmentation and aggregation, as increasing n further
decreases both R1 and mAP.
3) Analysis on the Recurrent Structure in RSA: We further
ablated the RSA module by replacing the BiLSTM with three
variants: a unidirectional LSTM, an encoder-only Transformer,
and a decoder-only Transformer. As shown in Table III,
the BiLSTM setup consistently yields higher Rank-1 and
mAP scores. While the unidirectional LSTM lacks backward
context, both Transformer variants emphasize global attention,
which may overlook local sequential cues, such as body part
arrangement in ReID, especially with fewer patches. In contrast, the BiLSTM effectively preserves these spatial sequence
dependencies, leading to more discriminative features and
better overall performance.
4) Analysis on the Progressive Alignment Strategy in Lma :
To validate the effectiveness of the Progressive Alignment
Strategy in Lma , we conducted ablation experiments comparing it against two alternative weighting strategies: a decreasing
weighting strategy (Decrease) and a static weighting strategy
(Avg). In contrast, the Decrease strategy adopts a linearly
t
decreasing weighting factor g(t) = 1 − T −1
, emphasizing
early-stage alignment. The Avg strategy, on the other hand,
applies a uniform weighting g(t) = 0.5 across all time steps,
ignoring temporal dynamics. As shown in Figure 4(b), the
Progressive strategy significantly outperforms both Decrease
and Avg in terms of system performance, highlighting the
importance of gradually increasing alignment constraints.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:03 UTC from IEEE Xplore. Restrictions apply.

Fig. 5. Visualization results: (a) attention maps of different models (b) Feature
distribution on SYSU-MM01, with colors for different identities. Circles
indicate visible features, and triangles denote infrared features.

5) Visualization Analysis: To demonstrate each module’s
impact, we apply Grad-CAM [25] to highlight the model’s
focus areas, as shown in Figure 5(a), with subscripts matching
the experiments in Table II. The results show that, for both
visible and infrared modalities, the model increasingly concentrates on body regions as more modules are added, validating
the effectiveness of our proposed module. Additionally, Figure
5(b) presents a t-SNE [23] map for a randomly selected
subset of 8 identities from the SYSU-MM01 dataset. Initially,
features from the two modalities are distinct, but after training,
features for each identity across both modalities converge
within the embedding space.
V. C ONCLUSION
We introduce the SAHSR framework, which integrates
a Recurrent Semantic Aggregation module to progressively
mitigate semantic misalignment during feature extraction, a
Confidence-based Hard Sample Retraining strategy to enhance
the model’s ability to handle global hard samples through
selection and retraining, and a Viewpoint-Balanced Sampler
to address viewpoint discrepancies by maintaining a balanced
distribution of viewpoints in training batch. Experimental
results demonstrate that our method significantly improves
baseline performance, achieving state-of-the-art results.
R EFERENCES
[1] Wei Li, Xiatian Zhu, and Shaogang Gong, “Harmonious attention
network for person re-identification,” in Proceedings of the IEEE
conference on computer vision and pattern recognition, 2018, pp. 2285–
2294.
[2] Xin Jin, Cuiling Lan, Wenjun Zeng, Zhibo Chen, and Li Zhang, “Style
normalization and restitution for generalizable person re-identification,”
in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2020, pp. 3143–3152.
[3] Jialun Liu, Yifan Sun, Feng Zhu, Hongbin Pei, Yi Yang, and Wenhui Li,
“Learning memory-augmented unidirectional metrics for cross-modality
person re-identification,” in 2022 IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2022, pp. 19344–19353.
[4] Zhipeng Huang, Jiawei Liu, Liang Li, Kecheng Zheng, and Zheng-Jun
Zha, “Modality-adaptive mixup and invariant decomposition for rgbinfrared person re-identification,” in Proceedings of the AAAI conference
on artificial intelligence, 2022, vol. 36, pp. 1034–1042.
[5] Yan Lu, Yue Wu, Bin Liu, Tianzhu Zhang, Baopu Li, Qi Chu, and
Nenghai Yu, “Cross-modality person re-identification with sharedspecific feature transfer,” in Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, 2020, pp. 13379–13389.
[6] Alexander Hermans, Lucas Beyer, and Bastian Leibe,
“In defense of the triplet loss for person re-identification,” arXiv preprint
arXiv:1703.07737, 2017.

[7] Zichang Tan, Ajian Liu, Jun Wan, Hao Liu, Zhen Lei, Guodong Guo,
and Stan Z Li, “Cross-batch hard example mining with pseudo large
batch for id vs. spot face recognition,” IEEE Transactions on Image
Processing, vol. 31, pp. 3224–3235, 2022.
[8] David G Lowe, “Distinctive image features from scale-invariant keypoints,” International journal of computer vision, vol. 60, pp. 91–110,
2004.
[9] Martin Koestinger, Martin Hirzer, Paul Wohlhart, Peter M Roth, and
Horst Bischof, “Large scale metric learning from equivalence constraints,” in 2012 IEEE conference on computer vision and pattern
recognition. IEEE, 2012, pp. 2288–2295.
[10] Shuting He, Hao Luo, Pichao Wang, Fan Wang, Hao Li, and Wei Jiang,
“Transreid: Transformer-based object re-identification,” in Proceedings
of the IEEE/CVF international conference on computer vision, 2021,
pp. 15013–15022.
[11] Xingye Fang, Yang Yang, and Ying Fu, “Visible-infrared person
re-identification via semantic alignment and affinity inference,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2023, pp. 11270–11279.
[12] Yukang Zhang and Hanzi Wang, “Diverse embedding expansion network
and low-light cross-modality benchmark for visible-infrared person reidentification,” in Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 2023, pp. 2153–2162.
[13] Kaijie Ren and Lei Zhang, “Implicit discriminative knowledge learning
for visible-infrared person re-identification,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2024, pp. 393–402.
[14] Mang Ye, Jianbing Shen, Gaojie Lin, Tao Xiang, Ling Shao, and
Steven CH Hoi, “Deep learning for person re-identification: A survey
and outlook,” IEEE transactions on pattern analysis and machine
intelligence, vol. 44, no. 6, pp. 2872–2893, 2021.
[15] Rui Sun, Long Chen, Lei Zhang, Ruirui Xie, and Jun Gao, “Robust
visible-infrared person re-identification based on polymorphic mask
and wavelet graph convolutional network,” IEEE Transactions on
Information Forensics and Security, 2024.
[16] Yiyuan Zhang, Sanyuan Zhao, Yuhao Kang, and Jianbing Shen, “Modality synergy complement learning with cascaded aggregation for visibleinfrared person re-identification,” in European Conference on Computer
Vision. Springer, 2022, pp. 462–479.
[17] Zhenyu Cui, Jiahuan Zhou, and Yuxin Peng, “Dma: Dual modalityaware alignment for visible-infrared person re-identification,” IEEE
Transactions on Information Forensics and Security, 2024.
[18] Yunhao Du, Zhicheng Zhao, and Fei Su, “Yyds: Visible-infrared
person re-identification with coarse descriptions,”
arXiv preprint
arXiv:2403.04183, 2024.
[19] Hao Sheng, Yanwei Zheng, Wei Ke, Dongxiao Yu, Xiuzhen Cheng,
Weifeng Lyu, and Zhang Xiong, “Mining hard samples globally and
efficiently for person reidentification,” IEEE Internet of Things Journal,
vol. 7, no. 10, pp. 9611–9622, 2020.
[20] Mengmeng Liu, Zhi Ma, Tao Li, Yanfeng Jiang, and Kai Wang, “Longterm person re-identification with dramatic appearance change: Algorithm and benchmark,” in Proceedings of the 30th ACM International
Conference on Multimedia, 2022, pp. 6406–6415.
[21] Houjing Huang, Dangwei Li, Zhang Zhang, Xiaotang Chen, and Kaiqi
Huang, “Adversarially occluded samples for person re-identification,”
in Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 5098–5107.
[22] Ancong Wu, Wei-Shi Zheng, Shaogang Gong, and Jianhuang Lai, “Rgbir person re-identification by cross-modality similarity preservation,”
International journal of computer vision, vol. 128, no. 6, pp. 1765–
1785, 2020.
[23] Dat Tien Nguyen, Hyung Gil Hong, Ki Wan Kim, and Kang Ryoung
Park, “Person recognition system based on a combination of body
images from visible light and thermal cameras,” Sensors, vol. 17, no.
3, pp. 605, 2017.
[24] Zhun Zhong, Liang Zheng, Donglin Cao, and Shaozi Li, “Re-ranking
person re-identification with k-reciprocal encoding,” in Proceedings of
the IEEE conference on computer vision and pattern recognition, 2017,
pp. 1318–1327.
[25] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra, “Grad-cam: Visual
explanations from deep networks via gradient-based localization,” in
Proceedings of the IEEE international conference on computer vision,
2017, pp. 618–626.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:03 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
7140

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 5, MAY 2026

Self-Rectification Historical Consistency Learning
for Coupled Noisy Visible-Infrared
Person Re-Identification
Jiacheng Zhao , Yongxi Li , and Changsheng Xu , Fellow, IEEE

Abstract—Visible-infrared person re-identification (VI-ReID)
retrieves cross-modal identity matches between visible and
infrared images, offering significant value for round-the-clock
surveillance. Despite recent advances, challenges remain: the task
relies heavily on high-quality annotations, and factors such as
occlusion, viewpoint variations, and the inherent difficulty of
labeling infrared images inevitably introduce noisy annotations
(NA) into the dataset during large-scale dataset construction.
Moreover, coupled noisy labels in two modalities lead to noisy
correspondence (NC), further complicating the learning process.
Although prior research has achieved relatively stable results
in addressing the NA and NC problem for VI-ReID through
noise detection and robust loss functions, they still exhibit certain
limitations: 1) Underutilization of training data. Existing methods
often discard noisy samples to mitigate their negative impact,
overlooking their potential value. 2) Lack of historical relevance.
Unstable learning dynamics under noisy labels lead to inconsistent outputs, yet current approaches ignore the valuable historical
information embedded in these fluctuations. Focusing on these
challenges in VI-ReID, we propose Self-Rectification Historical
Consistency Learning (SRHCL) for VI-ReID, which consists
of noise detection, self-refined label rectification, and historical
consistency learning modules. Firstly, the noise detection module
calculates confidence weights for each sample by modeling the
model’s loss response, thereby mitigating the adverse impact
of noisy samples in subsequent training phases. Secondly, we
propose a self-refined label rectification module to rectify noisy
labels by reliable historical predictions, progressively collating the
training data at fixed intervals. Finally, we introduce cross-modal
contrastive learning and early learning regularization based on
momentum-updated memories to facilitate historical consistency
learning. Extensive experiments conducted on SYSU-MM01 and
RegDB datasets demonstrate the robustness and effectiveness of
our method across varying noisy ratios.
Received 2 May 2025; revised 18 November 2025; accepted 3 December
2025. Date of publication 11 December 2025; date of current version 7 May
2026. This work was supported in part by Guangdong Science and Technology
Program under Grant 2024B01015004, in part by Beijing Natural Science
Foundation under Grant L252032, and in part by the Joint Funds of the
National Natural Science Foundation of China under Grant U23A20387. This
article was recommended by Associate Editor L. Nie. (Corresponding author:
Changsheng Xu.)
Jiacheng Zhao is with the School of Information Science and
Technology, ShanghaiTech University, Shanghai 201210, China (e-mail:
zhaojch2022@shanghaitech.edu.cn).
Yongxi Li is with the State Key Laboratory of Multimodal Artificial
Intelligence Systems, Institute of Automation, Chinese Academy of Sciences,
Beijing 100190, China (e-mail: liyongxi@outlook.com).
Changsheng Xu is with the State Key Laboratory of Multimodal Artificial
Intelligence Systems, Institute of Automation, Chinese Academy of Sciences,
Beijing 100190, China, and also with the School of Artificial Intelligence,
University of Chinese Academy of Sciences, Beijing 100049, China (e-mail:
csxu@nlpr.ia.ac.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3642770

Index Terms—Visible-infrared person re-identification, noise
detection, contrastive learning, label rectification.

I. I NTRODUCTION

V

ISIBLE-INFRARED person re-identification (VI-ReID)
is a challenging cross-modal retrieval task that seeks to
match individuals across visible and infrared modalities from
a gallery of person images [1], [2], [3], [4], [5], [6]. Due to
the advantages of infrared cameras under low-light conditions,
this task has drawn increasing attention in real-life surveillance and security systems. Although learning from different
modalities allows models to uncover rich and diverse shared
semantics [7], [8], [9], the significant modality discrepancy
between visible and infrared images poses new challenges.
Numerous efforts [9], [10], [11] have been made to address
the cross-modal discrepancy between visible and infrared
images, aiming to enhance multi-modal learning and improve
performance in VI-ReID. ADCA [12] employed image augmentations and heterogeneous feature aggregation to narrow
the differences between modalities. Ye et al. [3] designed
channel augmentations to mitigate differences and establish
relationships of input channels. Kim et al. [13] proposed a
part-mix strategy that generates part-aware augmented samples
through the mixing of part-level descriptors. These augmentations provide an intermediary modality as joint inputs for
the model, enhancing generalization capabilities for VI-ReID.
For better feature alignment, DFLN-ViT [14] introduced crossmodal matching using part and location information, along
with modifications to the model structure. Recent studies
[15], [16] further researched this problem by incorporating
structural and shape information to extract modality-irrelevant
identity features to improve the robustness.
Despite advancements in VI-ReID, there remain several
challenging issues that impede the practical application of
existing methods. The success of these supervised VI-ReID
approaches relies heavily on high-quality data annotations that
are often resource-intensive and laborious to obtain, especially
when annotating infrared data. Meanwhile, the gap between
identities within a single modality is typically smaller than the
variation between the same identity across modalities. These
inherent characteristics significantly increase the difficulty of
obtaining accurate annotations. Consequently, the noisy annotations (NA) problem inevitably exists in collected cross-modal
data, complicating the learning process and causing performance degradation.Additionally, as a multi-modal retrieval

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

ZHAO et al.: SRHCL FOR COUPLED NOISY VISIBLE-INFRARED PERSON RE-IDENTIFICATION

7141

of the dataset. However, existing label correction methods
are typically limited to single-modal embedding spaces and
primarily designed for classification tasks. Consequently, these
approaches employed in prior works may not be directly
applicable to VI-ReID with coupled noisy labels.
Consequently, the primary challenges in learning with coupled noisy labels in VI-ReID can be summarized as follows:

Fig. 1. Noisy Annotations (NA) and Noisy Correspondence (NC). In the
figure, the shape of each sample represents its corresponding modality, while
the color differentiates the correct identity of each sample.

task, VI-ReID faces the challenge of noisy correspondence
(NC), which arises from coupled noisy labels. As shown in
Figure 1, cross-modal input image pairs are randomly sampled
and composed from each dataset according to their annotated
identity labels during the training phase, leading to mismatches
of cross-modal pairs. This exacerbates the complexity of the
learning process and poses further hurdles in achieving robust
performance.
In learning with noisy labels, previous approaches mainly
concentrated on precise noise detection to filter out noise
samples and mitigate their adverse effects. Existing methods
[17], [18], [19], [20] primarily treated noisy detection as a
classification task, relying on single-modal feature mining
to identify noisy samples. For learning with coupled noisy
labels in VI-ReID, DART [21] and LCNL [22] leveraged
the memorization effect [17], [23], [24] that DNNs tend to
fit clean data simple patterns during training at the early
stage. Consequently, the model tends to show reduced loss
for simple and clean samples, while simultaneously exhibiting
an increased loss for noisy samples. By utilizing a Gaussian
Mixture Model (GMM) [25] to model the loss distributions,
the model computes confidence weights for each sample and
can effectively split noise samples from the dataset. Building
upon this, their approach suppresses noisy samples through
loss reweighting while concentrating exclusively on clean
samples for subsequent training. However, this strategy may
lead to under-utilization of the dataset.
In the past decade, negative learning with complementary
labels [26], [27], [28] has been widely studied to cut data
annotation costs. For noisy correspondence in cross-modal
learning, works [29], [30], [31] further explore complementary
learning to address mismatched data underutilization. While
directly treating noisy labels as complementary labels still
causes suboptimal utilization. Notably, negative labels have far
less information than correct identity labels. Moreover, unlike
unsupervised or semi-supervised learning, noisy label learning
has unique potential: models can learn discriminative patterns
from clean samples via initial training, then gradually make
accurate identity predictions. Unfortunately, this potential is
overlooked in existing negative learning frameworks for noisy
samples.
In response to noisy data underutilization, researchers have
proposed label rectification methods. It serves to correct the
label of noisy samples, effectively transforming them into
clean samples, which enables the comprehensive utilization

• Underutilization of training data. Previous research
on VI-ReID with NA and NC mainly concentrated on
enhancing noise detection accuracy. Such studies often
train models exclusively on clean data to attain stable
outcomes. However, this strategy causes underutilization
of training data, ignoring the value of noisy samples.
Although label rectification can mitigate this issue by
rectifying noisy labels, current approaches are primarily
designed for single-modality classification tasks. Consequently, they fail to yield reliable outcomes in VI-ReID.
• Lack of historical relevance. In the context of learning
with noisy labels, the models are susceptible to overfitting
the noisy samples, leading to historical inconsistencies in
the model’s predictions and features. Existing methods
rely solely on noise detection results and model outputs at
the current timestep during training, while neglecting the
temporal correlations with earlier model outputs. These
correlations contain valuable historical information that
could facilitate more stable learning dynamics.
Inspired by these observations, we introduce a novel
approach called Self-Rectification Historical Consistency
Learning (SRHCL) to solve the problem of learning with
coupled noisy labels for VI-ReID. In detail, SRHCL consists
of three steps: noise detection, historical consistency learning,
and self-refined label rectification. Firstly, the noise detection
module utilizes the model’s memorization effect [32] and
fits the loss response by Gaussian Mixture Models (GMMs)
to calculate confidence weights for each sample. Secondly,
we adopt contrastive learning to SRHCL for robust multimodal learning within a shared feature space. Thirdly, based
on the stable predictions from the early training stage [33],
our method designs early learning regularization loss and
self-refined label rectification modules based on the historical
memory mechanism. This approach serves to stabilize the
training dynamics and restrains drastic fluctuations in the
model’s predictions, as shown in Figure 2 (b).
The key contributions in our paper are summarized as
follows:
• We present a novel three-step pipeline for robust learning
of VI-ReID despite the presence of noisy labels. It
prevents models from overfitting to noise in the dataset
and achieves progressive learning from simple and clean
samples to hard ones.
• We integrated the cross-modal contrastive learning to
address the modality gap in VI-ReID. Specifically, we
employ the confidence weight of each sample and force
the model to focus on sample pairs with clean labels for
robust cross-modal learning.
• Our method leverages historical memorization of models
as regularization. Through self-refined label rectification,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

7142

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 5, MAY 2026

mance [44]; Sun et al. [45] and Wang et al. [46] addressed
the issue by utilizing local detail information to segment
global features and extract multi-granularity features. They
enhance the models’ ability to distinguish identities with the
concatenated features. A group of works [47], [48], [49], [50],
[51], [52] also use image parsing and the attention mechanism
to extract fine-grained features and match corresponding body
parts, obtaining aggregated multi-grained features to improve
the model’s performance.
Nonetheless, in real-world scenarios, relying solely on a
single visible modality is inadequate, particularly in low-light
conditions where visibility is severely compromised.

B. Visible-Infrared Person Re-Identification

Fig. 2. (a) Previous methods for noisy labels rely on accurate noise detection
to compute sample confidence, but they overlook the value of noisy samples,
thus underutilizing available data. (b) Our method uses momentum-updated
memories (for historical relevance) to stably predict the labels of noisy
samples. By accurate label rectification, our approach enables the model to
undergo comprehensive training with the refined dataset.

our pipeline progressively corrects the labels of noisy
samples and restrains them in the early learning stage
to offer stable and accurate predictions.
• Our method exhibits robust performance on SYSUMM01 and RegDB datasets across a range of noise
ratios, achieving new SOTA in the realm of learning with
coupled noisy labels for VI-ReID. Extensive experiments
demonstrated the effectiveness of each module in our
approach.
II. R ELATED W ORK
A. Deep Person Re-Identification
Deep person re-identification focuses on retrieving images
of individuals with the same identity across cameras using
features extracted by the deep neural networks [34], [35]. The
key to this task is learning identity-aware concepts without
the influence of environmental factors. In recent years, deep
person re-identification has experienced significant progress,
driving advancements in this domain [5], [36], [37]. Global
representations of pedestrian images have been the primary
approach [38] when deep learning was introduced into the
field.
However, the person-identification task is characterized
by inter-identity similarities and intra-identity distinctions.
Similar to fine-grained image recognition [39], [40], objects
exhibit high similarity in overall appearance but differ in
subtle features. In person re-identification, details like clothing
and texture are indispensable in improving retrieval performance [41], [42], [43]. Therefore, fully understanding and
distinguishing the subtle visual differences between objects is
crucial for this task. To this end, input images are processed
through multiple branches for more refined retrieval perfor-

To overcome the limitation of visible images in varying
environments and conditions, images captured by infrared
cameras are integrated into person re-identification systems
for cross-modal learning [53]. This integration harnesses the
strengths of both modalities, ultimately enhancing model performance across a diverse range of scenarios.
The complementary nature of visible and infrared modalities has sparked increasing interest in cross-modality person
re-identification among researchers. To address the inherent modality discrepancies present in Visible-infrared ReIdentification (VI-ReID), a variety of methods [5], [54], [55],
[56], [57], [58] have been proposed, yielding notable advancements in performance on collected datasets. These methods
can be classified into two groups according to their purposes:
image-level augmentations and feature-level alignment. 1) The
image-level input of the intermediate modality can effectively
assist the model in capturing the connections between the
visible and infrared modalities. Data augmentations [3], [13]
and generative networks [59] have been employed to generate
intermediary modality images for joint training. This approach
encourages the model to learn modality-sharing identity information, bridging the gap between visible and thermal views. 2)
From the perspective of multi-modal feature alignment, metric
learning methods are employed to train the model to obtain
discriminative identity representations. Given the fine-grained
nature of ReID, relying solely on global features leads the
model to overlook cross-modal local information matching.
To address this issue, DFLN-ViT [14] introduced a crossmodal matching approach incorporating part and location
information to address this issue. Recent research [15], [16],
[60] considered structural and shape information, formulating
frameworks to extract disentangled and modality-irrelevant
features.
However, the effectiveness of these methods relies heavily
on precise data annotations, which are resource-intensive to
obtain. Moreover, pedestrian images in datasets are commonly
captured by cameras with varying parameters under real-life
scenarios, which might lead to occlusion and blurred images.
Incorrect annotations easily occur, particularly in the case
of infrared images, where color information is not available.
Consequently, solving the problem of learning with noisy
labels in VI-ReID is significant.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

ZHAO et al.: SRHCL FOR COUPLED NOISY VISIBLE-INFRARED PERSON RE-IDENTIFICATION

C. Learning With Noisy Labels
Since deep neural networks are prone to overfitting to noise,
training models on datasets with noisy labels can be detrimental and significantly degrade performance. Numerous efforts
have been made to address the challenging issue of noisy
annotations through various strategies to alleviate the negative
effect of noise during training. From the perspective of robust
training, researchers have investigated training frameworks
like co-modeling [61] and regularization terms like neighbor
consistency [17], [18], [62] to mitigate the adverse impact
of noisy labels and achieve favorable results. In the case of
multi-modal, the gap between different modalities is often
greater than the variation among identities within the same
modality. This makes the structural relationship of the feature
space more complex, leading to inaccurate feature correlations.
The key step in loss reweighting and data resampling is noise
detection, which aims to distinguish noisy samples within the
datasets. A group of previous works attempts to leverage the
memorization effect of DNNs [17], [23], [24] by employing
a Gaussian Mixture Model (GMM) [25] or a Beta Mixture
Model (BMM) [63] to model per-sample loss distributions.
This strategy explicitly estimates the likelihood that a sample
is clean and allows the model to focus on clean samples and
ignore noisy instances.
Current methodologies for learning with noisy labels are
predominantly tailored for classification tasks within a single
modality. However, VI-ReID is a multi-modal task with a
substantial number of identity classes, which suffers from
a challenging model optimization process. Therefore, these
methods may not yield reliable outcomes for learning with
noisy labels in VI-ReID.

7143

noisy labels. Typically, these methods [70], [71], [72] involve
training a meta model on a smaller set of clean validation data,
which is subsequently utilized to provide corrected labels to
train the target model.
However, existing methods primarily focus on learning with
noisy labels for classification tasks within a single modality
and may not directly apply to VI-ReID. As mentioned above,
VI-ReID faces the challenge of noisy correspondence (NC)
resulting from coupled noisy labels. Training pairs in VIReID are randomly sampled and constructed in the training
process. Former label rectification approaches may suffer from
unstable predictions because when the model begins to overfit
is hard to perceive and control during training. To address the
challenges above, we propose self-refined label rectification
for learning with coupled noisy labels on VI-ReID. Our novel
approach leverages momentum-updated memories to obtain
stable model predictions and progressively rectifies noisy
labels over time, enhancing the overall model’s performance.
III. M ETHOD
In this section, we propose SRHCL to tackle the challenge
of noisy correspondence in VI-ReID through a systematic
three-step pipeline. Firstly, in section III-B, the noise detection
module is implemented through co-modeling to obtain the
correctly annotated confidence of each sample. Secondly, in
section III-D, we introduce historical consistency learning,
which comprises identity learning and cross-modal contrastive
learning that is reweighted by confidence scores, supplemented
with early learning regularization. Finally, section III-C outlines our self-refined label rectification strategy, detailing how
historical model predictions are harnessed to ensure stable and
effective label rectification.

D. Label Rectification
The strategies above alleviate the adverse influence of noise
by detecting and discarding noisy samples. However, this
strategy may result in the underutilization of training data,
overlooking the potential value of noisy samples. Therefore,
label rectification is introduced to predict reliable pseudolabels for those noisy samples and allow the model to gain
better performance.
These methods can be divided into three categories: correction by feature correlation, correction by model prediction,
and correction by meta-learning. Specifically, Li et al. [18]
corrects noisy labels by label propagation based on feature
correlations. Ortego et al. [17] utilized predictions from the
k-nearest neighbor to rectify noisy labels. By modeling the
global sample distribution as an optimal transport problem, OT
Cleaner [64] and CSOT [19] aimed to derive accurate pseudolabels for noisy samples utilizing feature correlation. A series
of works [65], [66], [67], [68] utilized the memorization effect
of the DNNs to rectify noisy labels for each sample. Specifically, they updated the soft labels by a convex combination
of the model’s current outputs and the noisy labels, weighted
by the predicted confidence. Based on these methods, Song et
al. [69] only updated labels for samples exhibiting consistent
predictions, thereby enhancing the accuracy of label rectification. Besides, meta learning is also introduced to rectify

A. Overview
The robust three-step model learning pipeline encompasses
a systematic approach to address noise in the learning process
of VI-ReID, as shown in Figure 3. In alignment with the cotraining strategy in [21] and [73], two networks net A and
net B with identical architectures but different initializations
are trained to enhance the robustness. SRHCL establishes an
iterative learning framework that cyclically alternates between
three modules: noisy detection, historical consistency learning,
and self-refined label rectification. Our method culminates in a
stable learning framework that guides the model to learn from
an increasing proportion of clean samples. Therefore, SRHCL
not only enhances the robustness of the model against NA and
NC but also improves the effectiveness of cross-modal feature
learning for VI-ReID.
To provide a foundation for the subsequent method descriptions, we present a concise overview of the model’s feature
extraction and classification processes applied to input samples
during training. To avoid ambiguity and for the convenience
of reference, we have summarized the notations of variables
in Table I. Let’s denote the visible and infrared image sets
r
Nv
with their annotations as Dv = {xiv , yvi }i=1
and Dr = {xrj , yrj }Nj=1
with shared identity set, where N v and N r are the size of
visible and infrared sets. Given a model with feature extractors

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

7144

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 5, MAY 2026

Fig. 3. Overview of robust three-step model learning pipeline in SRHCL. 1) Noise Detection. Networks A and B, with identical architectures, are used to
model loss distributions and calculate confidence weights for each other’s samples. This mutual evaluation helps accurately detect noisy data. 2) Historical
Consistency Learning. Our method enhances the robustness of cross-modal learning by integrating contrastive learning with early learning regularization. This
involves the use of momentum-updated memory attached to the logits and features. The regularization enforces historical consistency, thus preventing the
model from overfitting to noisy labels. 3) Self-refined Label Rectification. Historical model predictions are utilized iteratively to refine the labels of noisy
samples. This rectification progressively improves label accuracy over time. Combining the three steps, our approach implements a stable learning strategy
that ensures robust VI-ReID learning with noisy labels.

F v , F r for each modal and a classifier C, an input sample
xiv or xrj is passed into the model to compute the corresponding
feature representation fiv , f jr and predicted id logits tiv , trj of
each sample respectively.
tiv = C( fiv ),

fiv = F v (xiv ), xiv ∈ Dv ;

trj = C( f jr ),

f jr = F r (xrj ), xrj ∈ Dr .

The loss response is expected to be low for those simple
and clean samples, whereas it tends to be higher for noisy and
challenging instances. Subsequently, a two-component Gaussian Mixture Model (GMM) is employed to model the loss
distribution of samples in the training set by the ExpectationMaximization (EM) algorithm for noise detection.

(1)

For the sake of simplicity, we denote the feature, id logits,
and its label as { fim , tim , ym
i }, where m ∈ {v, r} shows their
modality.
B. Noise Detection
Building upon insights from prior research [32], [33], [65],
it has been observed that deep neural networks typically
undergo an “early learning” phase, where they tend to fit clean
data and capture simple yet discriminative patterns. In the
early training stage, the model learns those general patterns
to classify identity, giving stable predictions on samples.
Leveraging this observation, we follow the warm-up strategy
[21], [24], [74] by training on identification (cross-entropy)
loss:
m
N
`iid = − log P(ym
`id = {`iid }i=1
.
(2)
i |ti ),

p(`id |θ) =

K
X

γk N (`id |µk , σk 2 ),

K = 2,

(3)

k=1

where γk , N (µk , σk 2 ) denote each mixture coefficient and
Gaussian distribution N with mean µk and variance σk 2 ,
respectively. Among these components, the Gaussian distribution characterized by a smaller mean µc = min(µk ) is
utilized as the clean sample distribution Nc . Consequently, the
confidence weight of each sample wi can be derived from the
posterior probability associated with this distribution.
wi = p(Nc |`iid ).

(4)

Instead of splitting the train set into two parts for resampling, we take those confidence weights as soft metrics for
each sample during training. In this way, the model can adjust
its focus on samples according to their confidence.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

ZHAO et al.: SRHCL FOR COUPLED NOISY VISIBLE-INFRARED PERSON RE-IDENTIFICATION

7145

where β is the momentum with the constraint 0 ≤ β < 1.
A comprehensive description of the update processes can be
found in Eq. 10 and Eq. 15.
Based on the momentum-updated memories, we propose a
self-refined label rectification strategy that leverages historical
information. This approach is designed to enhance the reliability of label rectification by effectively integrating temporal
model outputs, thereby improving the overall accuracy and
consistency of the model’s predictions.
Furthermore, we implement label rectification exclusively
on samples in the noisy set where the networks A and B
achieve consensus:

TABLE I
T HE N OTATIONS U SED IN THE PAPER

ŷiA = argmax(t˜iA ), ŷiB = argmax(t˜iB ),

(7)

where ŷiA and ŷiB share common ID predictions.
This selective approach aims to generate label predictions
that are as accurate as possible, thereby minimizing the
influence of noise and enhancing the overall quality of the
dataset utilized for training.
(
ŷA , (xi , yi ) ∈ Dnoisy and ŷiA = ŷiB
(8)
yi = i
yi , otherwise,
D. Historical Consistency Learning
1) Lable-Wise Identity Learning: To obtain robust identity
learning under the effect of noisy annotations, our pipeline
leverages the confidence weights calculated from the comodeling networks to re-balance the original cross-entropy
loss as soft identification loss L sid :
N
X
sid
m
L =−
wi log P(ym
(9)
i |ti ).

C. Self-Refined Label Rectification
To further extend the performance of VI-ReID under noise,
fully utilizing the available data is necessary. Leveraging
confidence weights obtained from the noise detection step, the
training set can be divided into clean and noisy samples based
on a predefined threshold η.
wi = (wiA + wiB ) / 2,
Dnoisy = {(xi , yi ) | wi < η, (xi , yi ) ∈ Dv ∪ Dr }.

(5)

where we directly set the general confidence weight wi as the
average of wiA and wiB by co-modeling from net A and net B,
which means that the samples are seen as noisy ones only
when both networks assign them a low confidence weight.
Furthermore, we calibrate noisy samples to their correct
labels, which will reuse these noisy samples effectively.
Because the point at which the model overfits noise is
unstable due to the inherent variance in model architectures
and the dynamic nature of the training process, relying solely
on the current predictions from the model classifier may
yield inaccuracies. Therefore, we leverage historical model
predictions to tackle this issue.
Firstly, we design momentum-updated memories for the
logits and features to utilize the historical outputs from the
model:
t˜ ←
− βt˜ + (1 − β) t,
f˜ ←
− β f˜ + (1 − β) f ,

(6)

i=1

By applying confidence weight to the identity loss, the
model can prioritize learning from clean samples with high
confidence weight while reducing the influence of noisy
samples.
However, noisy samples often have higher loss values,
which can lead to a risk of rapid overfitting to these noisy
annotations (NA). This phenomenon can result in error accumulation, potentially leading to ambiguous noise detection and
performance degradation over time.
To address this issue, we design regularization terms to
impose consistency between the soft labels in the momentumupdated memory and the model’s current outputs.
t˜im = βt˜im + (1 − β) tim ,
 m m 
N
X
ti · t˜i
,
Lelr
=
−
log
label
m
kti k · kt˜im k

(10)

i=1

where k · k denotes the l2 norm of input vectors. The penalization is high when the model’s outputs change dramatically.
Therefore, it ensures a stable learning process for the model,
suppressing the adverse impact of noise.
Therefore, the label-wise loss for identity learning can be
summarized as:
Llabel = L sid + λLelr
label ,

(11)

where the loss weight λ is used to control the strength of
regularization and ensure stable parameter updates.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

7146

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 5, MAY 2026

2) Feature-Wise Cross-Modal Contrastive Learning: During the training, feature extractors project both visible and
infrared samples into a unified embedding space. Similar
to ADP [3], cross-modal contrastive learning aims to optimize the relative distances among positive and negative
pairs by taking into account both intra-modality and crossmodality relationships. For input feature pairs with their labels
({ fia , yai }, { f jb , ybj }), sample i is from modality a and sample j
is from modality b, with a, b ∈ {v, r}. We divide them into
positive set Pi and negative set Ni according to the anchor
label yai .
Pi = { f jb | yai = ybj , i , j, {xbj , ybj } ∈ Dv ∪ Dr },
Ni = { f jb | yai , ybj , {xbj , ybj } ∈ Dv ∪ Dr }.

(12)

To better bridge the modality gap in VI-ReID, cosine similarity is replaced with negative L2 distance as the similarity
metric between features in contrastive learning.
v
u d 
2
uX
f a [k] − f b [k] ,
(13)
sim = sim( f a , f b ) = − t
ij

i

j

i

j

k=1

where d is the dimension of the feature embedding space. The
modification aims to mitigate the influence of the modality
gap, thereby effectively bringing together positive pairs with
the same identity within the feature space.
Supervised contrastive learning is utilized to reduce the
distance between positive pairs while concurrently increasing
the distance between negative pairs:

αi j = S o f tmax −simi j /τ , j ∈ Pi ,
Lcon = −

N
X

wi

i=1

j,i
X

αi j

j∈Pi

· log P

exp(simi j /τ)
P
, (14)
k∈Pi exp(simik /τ) +
k∈Ni exp(simik /τ)

where τ is the temperature hyperparameter. In addition, we reweight the contrastive loss for each sample using confidence
weights wi and softmax weights αi j , to emphasize the learning
of challenging positive pairs.
The objective functions above alleviate the negative effect
of those noisy sample pairs and can train the model stably
when the noise detection is precise. Nonetheless, the dynamic
nature of input sample pairs during model training may still
lead the model to learn from pairs with noisy correspondence
(NC). Consequently, similar to the regularization term for soft
labels mentioned above, we also implement regularization on
the feature representations to ensure temporal consistency.
f˜im = β f˜im + (1 − β) fim ,
 m ˜m 
N
X
fi · fi
Lelr
=
−
log
,
f eat
k f m k · k f˜m k
i=1

i

(15)

i

where k · k denotes the l2 norm of input vectors. The featurewise loss for cross-modal learning can be summarized as:
L f eat = Lcon + λLelr
f eat .

(16)

By imposing constraints between the current output and the
momentum-updated historical model outputs, our pipeline can

ensure stable training under dynamic noisy pairs and facilitate
smooth updates to the model’s parameters.
Therefore, the total loss of historical consistency learning
for each batch B can be written as:
L = Llabel + L f eat .

(17)

With mutual efforts from historical consistency learning and
self-refined label rectification, the model learning from those
ambiguous samples is suppressed in the early phase and thus
avoids rapid overfitting of noise. Incorporating the self-refined
label rectification step at regular intervals T during training,
our pipeline should progressively rectify those noisy labels
into reliable ones. The whole training pipeline of SRHCL is
shown in Algorithm 1.
IV. E XPERIMENTS
In this section, we conduct experiments on SYSU-MM01
and RegDB datasets to validate the robustness of SRHCL.
Subsequently, we conduct an ablation study to demonstrate
the effectiveness of each proposed module. Furthermore, we
visualize the training process and results to provide comprehensive insight into our method.
A. Experiment Settings
To validate the robustness of our method against coupled
noisy labels for VI-ReID, we conducted experiments on two
benchmark datasets, SYSUMM01 [1] and RegDB [75].
• SYSU-MM01 [1] is a large VI-ReID dataset comprising
30,071 RGB images sourced from four visible cameras
and 15,792 thermal images captured by two near-infrared
cameras. This comprehensive dataset encompasses a total
of 491 unique identities. Within this framework, the
training set consists of 22,258 visible images and 11,909
infrared images associated with 395 identities. The query
and gallery sets include 3,803 infrared images and 301
randomly sampled visible images, adhering to a singleshot configuration.
• RegDB [75] dataset is derived from overlapping visible
and thermal cameras, comprising 412 distinct identities,
with each identity represented by 10 RGB images and
10 infrared images. In accordance with [3], [21], and
[56], we perform 10 trials utilizing different random splits
for training and testing for experiments on the RegDB
dataset. RegDB follows two testing protocols: infrared-tovisible, where the infrared modality is used as the query,
and visible-to-infrared, where the visible modality serves
as the query.
We strictly adhere to the experimental settings established
in prior literature [3], [5], [21], [22] for the noisy labels setting
by randomly injecting incorrect identities with a fixed noise
ratio. The evaluation of the SYSU-MM01 dataset contains
the all-search and indoor-search settings. In the case of the
RegDB dataset, our model is evaluated on in f rared-visible
and visible-in f rared settings with average results from 10
trials, where images in two modalities retrieve each other
alternatively. To evaluate the performance of our method, we
utilize three metrics. The mean average precision (mAP) and

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

ZHAO et al.: SRHCL FOR COUPLED NOISY VISIBLE-INFRARED PERSON RE-IDENTIFICATION

Algorithm 1 Training Pipeline of SRHCL

7147

datasets; the momentum update rate β and λ for regularization
terms are set to be 0.7 and 0.3, respectively. For self-refined
label rectification, we set a fixed noisy threshold η to 0.3,
and we incorporate it into the training process at two and
five regular intervals in experiments of SYSU-MM01 and
RegDB.
C. Results and Comparisons

the cumulative matching curve (CMC) are employed to assess
retrieval accuracy, whereas the mean Inverse Negative Penalty
(mINP) serves as an indicator of the efficiency of the most
challenging correct matches.
B. Implementation Details
Our method is implemented with PyTorch on a GeForce
RTX 3090. As to the hyper-parameters mentioned in section III-D and section III-C, temperature τ is set as 0.25
and 0.1 in contrastive learning for SYSU-MM01 and RegDB

We compare the performance of our method with AGW
[5], DDAG [76], LbA [77], MPANet [56], ADP [3], DART
[21] and LCNL [22] on two datasets under different noise
ratio settings varying from 0%, 20%, and 50%. Among them,
DART [21] and LCNL [22] are designed for Vi-ReID under
several noise settings. The results of SYSU-MM01 and RegDB
are reported in Table II and Table IV, respectively.
1) Baselines:
a) Supervised VI-ReID methods:
• AGW [5]: This article provides a thorough overview and
detailed analysis of person re-identification tasks from
three distinct perspectives. Additionally, it introduces a
robust AGW baseline, which demonstrates competitive
performance in both single-modality and cross-modality
Re-ID tasks.
• DDAG [76]: This paper proposes a dynamic dualattention aggregation learning method, which is designed
to improve the recognition capability and robustness of
the model through the incorporation of two attention
modules.
• LbA [77]: LbA proposes a CMAlign module for
aligning visible and infrared inputs to establish dense
cross-modal correspondences, to minimize the disparity between visible and infrared images at the pixel
level.
• MPANet [56]: This paper proposes a joint modality and
pattern alignment network (MPANet) to mine the subtle texture differences for VI-ReID. MPANet introduces
a modality alleviation module (MAM) to mitigate the
cross-modal sample differences and a pattern alignment
module (PAM) to discover subtle differences from different modalities.
• ADP [3]: This method presents a robust channelaugmented joint learning strategy that incorporates
random channel-exchangeable augmentation and channellevel random erasing, aimed at creating an intermediate
modality for VI-ReID.
• DART [21]: This paper proposes DART to solve the
problem of learning with noisy labels in VI-ReID. DART
detects the noisy samples in datasets by modeling the
loss response and then calculates the confidence weights.
According to the weights, DART divides input sample
pairs into four categories and handles them case by case
for cross-modal feature learning.
• LCNL [22]: This paper proposes an adaptive quadruple
loss with theoretical guarantees. With the novel objective
function, LCNL achieves robust cross-modal learning
against noisy correspondence by adaptively dividing the
training pairs into four groups.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.


codex
第四篇和第五篇的共同点是把训练协议本身写成问题：前者说噪声样本不是该丢掉的坏数据，而是要被历史预测逐步纠正的资源；后者说批内三元组不够，难样本和视角分布要从全数据和采样策略上重新组织。现在读最后一篇。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 979-8-3503-6874-1/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICASSP49660.2025.10890656

Semantic Consistency And Integrity Network For
Cloth-changing Person Re-identification
Anqi Wang

Liyan Zhang*

School of Computer Science and Technology
Nanjing University of Aeronautics and Astronautics
Nanjing, China
aqwang@nuaa.edu.cn

School of Computer Science and Technology
Nanjing University of Aeronautics and Astronautics
Nanjing, China
zhangliyan@nuaa.edu.cn

Abstract—Cloth-changing Person Re-identification aims to retrieve target pedestrians across different cameras under clothingchanging scenarios. In recent years, many scholars have made
significant explorations in this field. However, existing methods
often overlook the semantic consistency and integrity of features.
To address this issue, we design a Semantic Consistency and
Integrity Network (SCI-Net) to learn semantically invariant
features and strip clothing bias from identity features while
maintaining their semantic integrity. The network consists of
three branches: clothing branch, raw image branch, and head
feature enhancement branch. Specifically, we first propose a
Head Soft Attention Generation Module to produce head soft
attention, thereby obtaining enhanced head features. Then, to
ensure that raw features can effectively learn invariant semantic
information from head-enhanced features, Semantic Consistency
Constraint is proposed to facilitate mutual learning between
the two branches. Finally, we leverage knowledge transfer to
enable clothing branch to perceive clothing bias entangled with
raw features and simulate causal intervention to quantify and
remove clothing bias. Experiments on the LTCC-ReID and PRCC
datasets demonstrate that our model outperforms other state-ofthe-art methods.
Index Terms—cloth-changing person re-identification, causal
intervention, semantic consistency.

I. I NTRODUCTION
Person re-identification aims to identify and retrieve the
same target pedestrian from videos or images recorded by
various cameras at different times and locations, which has
significant application value in fields such as smart security,
criminal investigation, and smart shopping [12], [19], [22],
[35], [37]. In recent years, person re-identification methods
has achieved significant progress [14], [17], [28] [4]. General
person re-identification algorithms mainly focus on short-term
Re-ID scenarios [11], [16], [29], with an impractical assumption that people would not change their clothes. Therefore,
they heavily rely on appearance features of the pedestrians,
especially salient colour and texture information of clothes.
However, in long-term Re-ID scenarios as shown in Fig. 1,
clothing may exhibit considerable variability over time .
Additionally, different people may wear the same or similar
clothing.
* Corresponding Author.
+ This work was supported in part by the National Natural Science
Foundation of China under Grant 62172212 in part by the Natural Science
Foundation of Jiangsu Province under Grant BK20230031.

Short-time ReID with
clothes consistency

Day1

Query

Clothes consistency result

Long-time ReID with
clothes inconsistency

Day2

Day3

Clothes inconsistency result

Fig. 1. The difference between the short-term Re-ID and the long-term ReID. In short-term Re-ID scenarios, people hardly change their clothes. While
in long-term Re-ID scenarios, clothing may exhibit considerable variability
over time and retrieval results should include both clothes consistency results
and clothes inconsistency results.

Recently, some scholars have also explored methods for
person re-identification under a significant but challenging ReID setting, Cloth-Changing Re-ID (CC-ReID) [2], [10], [25],
[27], [33], [3], [5]. Some existing CC-ReID methods use multimodality (information such as skeletons [25], radio signals [7],
faces [26], [31], gaits [6], [21], etc.) to model discriminative
biological features. However, these methods require additional
resources and models to capture multi-modality features. More
importantly, they ignore the semantic consistency of features,
which is critical for Re-ID.
Moreover, clothing bias entangled with identity features
can significantly affect the performance of CC-ReID models.
Although eliminating clothing bias is important, it is still not
clear how to eliminate clothing bias in the feature representation space. Some CC-ReID methods try directly covering up
clothing or simply ignoring it, e.g., the mask-based methods
[15], [36] and gait-based methods [6], [21]. Although these
methods are effective, they compromise semantic integrity.
To address the above issues, we design a Semantic Consistency and Integrity Network (SCI-Net). As shown in Fig. 2,
SCI-Net consists of three branches: a clothing branch, a raw
image branch, and a head feature enhancement branch. There
are two fundamental ideas of our method: 1) Promote model

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:40 UTC from IEEE Xplore. Restrictions apply.

focus on semantically invariant regions and thereby maintain semantic consistency features. 2) Eliminating clothing
bias while maintaining the semantic integrity of the features.
Firstly, to fully exploit discriminative features of raw images,
Head Soft Attention Generation Module (HSAGM) is designed
in the head feature enhancement branch. This module learns
to generate head soft attention from the body masks obtained
through human parsing by utilizing the proposed Body Part
Matching Loss. Secondly, Semantic Consistency Constraint
(SCC) is proposed which utilizes class activation maps and
saliency maps to achieve mutual learning of raw features
and head-enhanced features at high semantic level, enabling
semantic alignment between the two branches. Finally, to eliminate clothing bias, we design two loss functions: Knowledge
Transfer Loss and Causal Intervention Loss. The former loss
aims to enhance the perception of clothing features entangled
with pedestrian features in the raw image branch, while the
latter one focuses on quantifying effect of clothing bias and
eliminate it.
The main contributions of our work are as follows:
• We propose a Semantic Consistency and Integrity
Network (SCI-Net) for Clothing-Changing Person Reidentification (CC-ReID). The designed SCI-Net can extract semantically invariant identity-related features and
eliminate clothing bias without compromising the semantic integrity of pedestrian features.
• We design a Head Soft Attention Generation Module,
which utilizes body part matching loss and gets head soft
attention from body part masks. Additionally, Semantic
Consistency Constraint is proposed to facilitate mutual
learning between the raw image branch and the head feature enhancement branch, achieving semantic alignment
between the two branches.
• Knowledge transfer and causal intervention are applied
between raw image branch and clothing branch. Clothing
bias can be quantified and stripped away while the
semantic integrity of features can be maintained.
• Experiments on two public CC-ReID datasets, LTCCReID and PRCC, demonstrate that our proposed SCI-Net
outperforms the state-of-the-art CC-ReID methods.
II. M ETHODOLOGY
A. Overall Framework
As mentioned above, some existing CC-ReID models utilize
multi-modality information as auxiliary cues for retrieval and
bring a considerable amount of time and resource. Additionally, some methods compromises semantic integrity of
features to mitigate clothing bias. To better learn semantically
invariant and integrated features from pedestrian images, we
propose a Semantic Consistency and Integrity Network (SCINet). As illustrated in Fig. 2, SCI-Net consists of three
branches: a clothing branch, a original image branch, and a
head feature enhancement branch. First, we design a Head Soft
Attention Generation Module that utilizes body part matching
loss Lmatch to obtain head-enhanced features. Moreover, to

lead model’s attention to semantically invariant regions, we
introduce Semantic Consistency Constraint LSCC which can
promote mutual learning between raw image features Fraw
and head-enhanced features Fhead . In the clothing branch,
we extract local clothing features FC . Finally, we enhance
the original image branch’s perception of clothing features
through Knowledge Transfer Loss LKT and simulate causal
intervention between clothing branch and raw image branch,
quantifying the effect of clothing bias into causal relationship
loss LCI .
B. Head Soft Attention Generation Module
The Head Soft Attention Generation Module, as shown in
Fig. 2 a), takes raw image features Fraw and body parsing
results as inputs, utilizing body part matching loss Lmatch to
learn and obtain the head soft attention.
First, we generate human parsing results by SCHPNet [23].
Each parsing result contains 18 labels, including hair, face,
coat, skirt, left/right leg, left/right arm, and so on. We set
pixel values belonging to head region in the parsing results to
1, while assigning 0 to pixels in other regions, thus obtaining
head mask. Similarly, we generate masks for upper body, lower
body, and feet using the same method. In this way, we obtain
body part masks M ∈ RH×W ×K . Next, Fraw ∈ RH×W ×C
are input into a 1 × 1 convolution layer with a softmax
activation function to produce attention maps A ∈ RH×W ×K .
H, W and C represent height, width and number of channels,
respectively. K represents the number of body parts, which is
set to be 4, corresponding to head, upper body, lower body
and feet. For AK , value of a pixel denotes probability that
current pixel position belongs to the k-th body part. Finally,
body part matching loss Lmatch is proposed to optimize the
learning of body part attention maps A from body part masks
M , which can be defined as:
N K H−1 W −1
1 1 1 XX X X
Mk (h, w) log (Ak (h, w)) ,
N H W n=1
k=1 h=0 w=0
(1)
N represents batch size. MK represents value of k-th body
part mask at (h, w). And AK represents value of k-th attention
maps at (h, w). Furthermore, the learning process of body part
attention maps is supervised under identity loss (cross-entropy
loss). Thus, compared to the results from body parsing, body
part attention maps generated by this module are more relevant
to CC-ReID.

Lmatch = −

C. Semantic Consistensy Constraint
To enable the raw image branch to fully learn the headenhanced features and enhance the model’s attention to semantically invariant regions, a Semantic Consistency Constraint
(SCC) is designed, as illustrated in the figure Fig. 3.
Specifically, Fraw and Fhead are firstly processed by a
batch normalization layer. And then corresponding results are
input into a 1 × 1 convolution layer, and we can obtain two
class activation maps Graw and Ghead ∈ RI×H×W where I

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:40 UTC from IEEE Xplore. Restrictions apply.

a) Head Soft Attention Generation Module(HSAGM)

Clothes
Encoder

Clothes
Classifier

Conv1×1 + Softmax

Assist
Clothes
Classifier

0
0
0
0
0

ID-Clothes
Classifier

0
0
0
0

ID
Encoder

ID
Classifier

0
0
0
0
0
0
0

00
00
00
00
00
10
1

00
10
10
00
00
00
0

0
0
01
01
00
00
00
10
1

01
11
10
00
00
00
0

0
0
01
01
00
00
00
00
0

01
01
00
00
00
00

0
0
0
0

0
0
0

0
0
0
0
0
0

0

0

b) Causal Relationship and Causal Intervention

Causal Intervention

Feature

Feature

Human Parsing

ID
Classifier

HSAGM

Input

Output

Intervention

Output

Fig. 2. Overall framework of the proposed Semantic Consistensy and Integrity Network (SCI-Net). It consists of three branches: a raw image branch, a head
feature enhancement branch, and a clothing branch. Features of raw image branch are fed into the head soft attention generation module to generate head soft
attention by utilizing body part matching loss and enhance head features. The Semantic Consistency Constraint LSCC facilitates raw image branch to learn
semantically invariant identity-related features. Finally, we simulate causal intervention and eliminate clothing bias between raw image branch and clothing
branch under the supervision of the Knowledge Transfer Loss LKT and Causal Intervention Loss LCI .

AvgPool

1×1 Conv

MSE Loss

ID Label

AvgPool

1×1 Conv

Batch Norm

N
2
2 i
1 Xh
LSCC =
g − Eraw + g − Ehead
.
N n=1

ID Label

Batch Norm

represents the number of identities. Since the class activation
map represents the model’s attention to each region when
distinguishing pedestrians, we select maps corresponding to
ground truth identity label in the channel dimension. Then,
we compare pixel values of the two feature maps, selecting the maximum value at each pixel location to obtain a
more effective supervision signal. We denote this signal as
g ∈ RH×W . On the other hand, we apply average pooling on
Fraw and Fhead in the channel dimension to obtain saliency
maps Eraw and Ehead ∈ RH×W . Saliency maps essentially
indicate the focused areas of the network. To maintain the
semantic invariance of pedestrian features, we extend method
in [13] and impose a Semantic Consistency Constraint LSCC ,
which can be defined as:

MSE Loss

Fig. 3. Illustration of Semantic Consistency Constraint LSCC . ID Label
reperesents the true identity label.

and optimize it by using clothing classification loss. The
clothing bias is extracted by this classifier and transferred to
the clothing branch for further knowledge transfer. We can
calculate the KL distance from FC to Fraw as follows:

(2)

D. Causal Intervention
To tackle with CC-ReID task better, causal intervention is
introduced in this work to remove the clothing bias entangled
with pedestrian features. We first extract local clothing features
FC in the clothing branch through a clothes classifier δC by
adopting pyramid matching strategy [8]. To enable clothing
branch to perceive clothing bias entangled with raw features,
we use Kullback-Leibler (KL) divergence to fit the distribution
of clothing bias in pedestrian features. Specifically, we add
an assistant clothes classifier δA to the raw image branch

p̂C = exp (δC (F C )) ,

p̂raw = exp (δA (F raw )) ,

DKL (p̂C ∥p̂raw ) =

N
X
i=1

p̂m
C log

p̂m
C
.
p̂m
raw

(3)
(4)

Due to the asymmetry of KL divergence, we calculate
DKT (p̂raw ∥p̂C ) as well. And the total knowledge transfer loss
LKT is the sum of the two KL distance.
Following [30], the impact of clothing bias on final prediction results can be quantified by the difference between actual
prediction results and predictions obtained after intervention as
shown in Fig. 2 b), that is Yef f ect = YX,C − YC . To enhace
the representation of same-located bias, we perform fusion of

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:40 UTC from IEEE Xplore. Restrictions apply.

TABLE I
C OMPARISON WITH S TATE -O F -T HE -A RT METHODS ON LTCC AND PRCC DATASETS .
Methods
HACNN (CVPR’18) [24]
PCB (ECCV’18) [29]
HPM(AAAI’19) [9]
IANet (CVPR’19) [18]
CESD (ACCV’20) [25]
ISP (ECCV’20) [39]
3DSL(CVPR’21) [1]
GI-ReID (CVPR’22) [20]
UCAD (ICJAI’22) [32]
ACID (TIP’23) [34]
SCI-Net

LTCC
General
Cloth-changing
Rank-1 mAP
Rank-1
mAP
60.2
26.7
21.6
9.3
65.1
30.6
23.5
10.0
63.7
31.0
25.0
11.9
71.4
34.3
26.1
12.4
66.3
29.6
27.8
11.9
31.2
14.8
63.2
29.4
23.7
10.4
74.4
34.8
32.5
15.1
65.1
30.6
29.1
14.5
73.6
39.5
36.7
16.8

Fraw and clothes features FC . Casuality intervention loss can
be calculated as follows:
N
X
′
LCI = −
y i log (δraw (F raw ) − δraw
(F C ⊗ F raw )) , (5)
i=1
′
is
δraw is ID classifier in the raw image branch and δraw
a ID-Clothes classifier sharing weight with δraw . Finally, by
implementing causal intervention to eliminate the clothing bias
learned through knowledge transfer, we achieve robust CCReID performance without compromising semantic integrity.

E. Training and Inference
In three branches, cross-entropy loss [28] is used to optimize
learning process and we can calculate LID , L′ID and LC . The
total training loss is formulated as follows:
L = LID + L′ID + LC + λ1 Lmatch + λ2 LSCC + LKT + LCI ,
(6)
where λ1 and λ2 control weights of loss functions. During
Inference stage, we only apply raw image branch to reduce
computational and time cost.
III. E XPERIMENTS
A. Implementation Details
Rank-1 accuracy and mean average precision (mAP) are
both used to evaluate model performance. Two test settings are
defined: 1) general setting: both clothes-changing and clothesconsistent samples are used to calculate accuracy. 2) clothchanging setting: only clothes-changing samples are used to
calculate accuracy. Batch size of training is set to be 32 and
batch size of testing is set to be 128. Adam optimizer is applied
during training stage across a span of 100 epochs. Random
horizontal flipping, random cropping, and random erasing [38]
are used for data augmentation. Learning rate is initialized to
6e − 5. λ1 and λ2 are set to be 0.8 and 0.1 respectively.
B. Comparison with State-of-art Methods
We evaluate our model on LTCC-ReID and PRCC and
compare performance with multiple state-of-the-art CC-ReID
methods. As shown in TABLE I, the best results are bolded and
the second best results are underlined. Our SCI-Net outperfoms most of state-of-the-art CC-ReID method. For LTCC, our

PRCC
General
Cloth-changing
Rank-1
mAP
Rank-1
mAP
82.5
84.8
21.8
23.2
99.8
97.0
41.8
38.7
99.4
96.9
40.4
37.2
99.4
98.3
46.3
45.9
92.8
36.6
51.3
80.0
33.3
96.5
45.3
99.8
99.1
50.5
51.0

approach achieves great improvement under cloth-changing
setting and it outperforms the SOTA method UCAD [32] by
4.2% Rank-1 and 1.7% mAP. Although Rank-1 accuracy under
general setting is 0.8% lower than that of UCAD, we obtain
a 4.7% improvement on mAP. For PRCC, our approach leads
the SOTA method IANet [18] by 4.2% Rank-1 and 5.1% mAP
under cloth-changing setteing. Besides, results show SCI-Net
improves mAP by 0.8% and achieves best performance on
Rank-1 under general setting.
C. Ablation Studys
TABLE II
A BLATION E XPERIMENT RESULTS .
Methods

Raw

Head

Cloth

1(Baseline)
2
3
4

!
!
!
!

%
!
%
!

%
%
!
!

LTCC
Rank-1 mAP
26.8
12.2
29.6
15.8
25.8
13.7
36.7
16.8

Raw, head, cloth denote raw image branch, head feature
enhancement branch and clothing branch respectively. Individually adding the head branch and the clothing branch can both
improve model performance, indicating their contributions.
When combined, model exhibits best performance, indicating
the rationality of design.
IV. C ONCLUSION
In this work, we propose a novel tri-branch network,
Semantic Consistency and Integrity Network (SCI-Net) for
CC Re-ID. Compared with existing CC-ReID methods, we
extract semantically invariant and integrated features that can
robustly tackle with CC-ReID task. In order to encourage
model focus on semantic consistent area, we propose a head
soft attention generation module to obtain head-enhanced
features and use semantic consistency constraint to promote
mutual learning of raw features and head-enhanced features.
For mitigating clothing bias, we introduce knowledge transfer
and casual intervention to quantify effect of bias and eliminate
it. Extensive experiments on two CC-ReID datasets show our
SCI-Net outperforms state-of-the-art methods.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:40 UTC from IEEE Xplore. Restrictions apply.

R EFERENCES
[1] Jiaxing Chen, Xinyang Jiang, Fudong Wang, Jun Zhang, Feng Zheng,
Xing Sun, and Wei-Shi Zheng. Learning 3d shape feature for textureinsensitive person re-identification. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 8146–
8155, 2021.
[2] Zhenyu Cui, Jiahuan Zhou, Yuxin Peng, Shiliang Zhang, and Yaowei
Wang. Dcr-reid: Deep component reconstruction for cloth-changing
person re-identification. IEEE Transactions on Circuits and Systems
for Video Technology, 33:4415–4428, 2023.
[3] Yongkang Ding, Rui Mao, Guodong Du, and Liyan Zhang. Clotheseraser: clothing-aware controllable disentanglement for clothes-changing
person re-identification. Signal, Image and Video Processing, pages 1–
12, 2024.
[4] Yongkang Ding, Anqi Wang, and Liyan Zhang. Multidimensional
semantic disentanglement network for clothes-changing person reidentification. In Proceedings of the 2024 International Conference on
Multimedia Retrieval, pages 1025–1033, 2024.
[5] Yongkang Ding, Yinghao Wu, Anqi Wang, Tiantian Gong, and Liyan
Zhang. Disentangled body features for clothing change person reidentification. Multimedia Tools and Applications, pages 1–22, 2024.
[6] Chao Fan, Yunjie Peng, Chunshui Cao, Xu Liu, Saihui Hou, Jiannan
Chi, Yongzhen Huang, Qing Li, and Zhiqiang He. Gaitpart: Temporal
part-based model for gait recognition. In 2020 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages 14213–14221,
2020.
[7] Lijie Fan, Tianhong Li, Rongyao Fang, Rumen Hristov, Yuan Yuan,
and Dina Katabi. Learning longterm representations for person reidentification using radio signals, 2020.
[8] Yang Fu, Yunchao Wei, Yuqian Zhou, Honghui Shi, Gao Huang,
Xinchao Wang, Zhiqiang Yao, and Thomas Huang. Horizontal pyramid
matching for person re-identification, 2018.
[9] Yang Fu, Yunchao Wei, Yuqian Zhou, Honghui Shi, Gao Huang,
Xinchao Wang, Zhiqiang Yao, and Thomas Huang. Horizontal pyramid
matching for person re-identification. In Proceedings of the AAAI
conference on artificial intelligence, volume 33, pages 8295–8302, 2019.
[10] Xinqian Gu, Hong Chang, Bingpeng Ma, Shutao Bai, Shiguang Shan,
and Xilin Chen. Clothes-changing person re-identification with rgb
modality only, 2022.
[11] Xinqian Gu, Hong Chang, Bingpeng Ma, Hongkai Zhang, and Xilin
Chen. Appearance-preserving 3d convolution for video-based person
re-identification. In Computer Vision–ECCV 2020: 16th European
Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II
16, pages 228–243. Springer, 2020.
[12] Xinqian Gu, Bingpeng Ma, Hong Chang, Shiguang Shan, and Xilin
Chen. Temporal knowledge propagation for image-to-video person reidentification. In Proceedings of the IEEE/CVF international conference
on computer vision, pages 9647–9656, 2019.
[13] Peini Guo, Hong Liu, Jianbing Wu, Guoquan Wang, and Tao Wang.
Semantic-aware consistency network for cloth-changing person reidentification. In Proceedings of the 31st ACM International Conference
on Multimedia, MM ’23, page 8730–8739, New York, NY, USA, 2023.
Association for Computing Machinery.
[14] Shuting He, Hao Luo, Pichao Wang, Fan Wang, Hao Li, and Wei Jiang.
Transreid: Transformer-based object re-identification, 2021.
[15] Peixian Hong, Tao Wu, Ancong Wu, Xintong Han, and Weishi Zheng.
Fine-grained shape-appearance mutual learning for cloth-changing person re-identification. 2021 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pages 10508–10517, 2021.
[16] Ruibing Hou, Hong Chang, Bingpeng Ma, Shiguang Shan, and
Xilin Chen. Temporal complementary learning for video person reidentification. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXV 16,
pages 388–405. Springer, 2020.
[17] Ruibing Hou, Bingpeng Ma, Hong Chang, Xinqian Gu, Shiguang Shan,
and Xilin Chen. Interaction-and-aggregation network for person reidentification, 2019.
[18] Ruibing Hou, Bingpeng Ma, Hong Chang, Xinqian Gu, Shiguang Shan,
and Xilin Chen. Interaction-and-aggregation network for person reidentification. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 9317–9326, 2019.

[19] Ruibing Hou, Bingpeng Ma, Hong Chang, Xinqian Gu, Shiguang
Shan, and Xilin Chen. Feature completion for occluded person reidentification. IEEE Transactions on Pattern Analysis and Machine
Intelligence, 44(9):4894–4912, 2021.
[20] Xin Jin, Tianyu He, Kecheng Zheng, Zhiheng Yin, Xu Shen, Zhen
Huang, Ruoyu Feng, Jianqiang Huang, Zhibo Chen, and Xian-Sheng
Hua. Cloth-changing person re-identification from a single image with
gait prediction and regularization. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 14278–
14287, 2022.
[21] Xin Jin, Tianyu He, Kecheng Zheng, Zhiheng Yin, Xu Shen, Zhen
Huang, Ruoyu Feng, Jianqiang Huang, Xian-Sheng Hua, and Zhibo
Chen. Cloth-changing person re-identification from a single image with
gait prediction and regularization, 2022.
[22] Qingming Leng, Mang Ye, and Qi Tian. A survey of open-world person
re-identification. IEEE Transactions on Circuits and Systems for Video
Technology, 30(4):1092–1108, 2019.
[23] Peike Li, Yunqiu Xu, Yunchao Wei, and Yi Yang. Self-correction for
human parsing, 2019.
[24] Wei Li, Xiatian Zhu, and Shaogang Gong. Harmonious attention network
for person re-identification. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 2285–2294, 2018.
[25] Xuelin Qian, Wenxuan Wang, Li Zhang, Fangrui Zhu, Yanwei Fu, Tao
Xiang, Yu-Gang Jiang, and Xiangyang Xue. Long-term cloth-changing
person re-identification, 2020.
[26] Wei Shi, Hong Liu, and Mengyuan Liu. Iranet: Identity-relevance aware
representation for cloth-changing person re-identification. Image Vision
Comput., 117(C), jan 2022.
[27] Xiujun Shu, Ge Li, Xiao Wang, Weijian Ruan, and Qi Tian. Semanticguided pixel sampling for cloth-changing person re-identification. IEEE
Signal Processing Letters, 28:1365–1369, 2021.
[28] Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, and Shengjin Wang. Beyond
part models: Person retrieval with refined part pooling (and a strong
convolutional baseline), 2018.
[29] Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, and Shengjin Wang. Beyond
part models: Person retrieval with refined part pooling (and a strong
convolutional baseline). In Proceedings of the European conference on
computer vision (ECCV), pages 480–496, 2018.
[30] Kaihua Tang, Yulei Niu, Jianqiang Huang, Jiaxin Shi, and Hanwang
Zhang. Unbiased scene graph generation from biased training, 2020.
[31] Fangbin Wan, Yang Wu, Xuelin Qian, Yixiong Chen, and Yanwei Fu.
When person re-identification meets changing clothes, 2020.
[32] Yuming Yan, Huimin Yu, Shuzhao Li, Zhaohui Lu, Jianfeng He,
Haozhuo Zhang, and Runfa Wang. Weakening the influence of clothing:
Universal clothing attribute disentanglement for person re-identification.
In IJCAI, pages 1523–1529, 2022.
[33] Qize Yang, Ancong Wu, and Wei-Shi Zheng. Person re-identification
by contour sketch under moderate clothing change. IEEE Transactions
on Pattern Analysis and Machine Intelligence, 43(6):2029–2046, June
2021.
[34] Zhengwei Yang, Xian Zhong, Zhun Zhong, Hong Liu, Zheng Wang, and
Shin’Ichi Satoh. Win-win by competition: Auxiliary-free cloth-changing
person re-identification. IEEE Transactions on Image Processing,
32:2985–2999, 2023.
[35] Mang Ye, Jianbing Shen, Gaojie Lin, Tao Xiang, Ling Shao, and
Steven CH Hoi. Deep learning for person re-identification: A survey
and outlook. IEEE transactions on pattern analysis and machine
intelligence, 44(6):2872–2893, 2021.
[36] Shijie Yu, Shihua Li, Dapeng Chen, Rui Zhao, Junjie Yan, and
Yu Qiao. Cocas: A large-scale clothes changing person dataset for reidentification, 2020.
[37] Liang Zheng, Yi Yang, and Qi Tian. Sift meets cnn: A decade survey of
instance retrieval. IEEE transactions on pattern analysis and machine
intelligence, 40(5):1224–1244, 2017.
[38] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang.
Random erasing data augmentation, 2017.
[39] Kuan Zhu, Haiyun Guo, Zhiwei Liu, Ming Tang, and Jinqiao Wang.
Identity-guided human semantic parsing for person re-identification. In
Computer Vision–ECCV 2020: 16th European Conference, Glasgow,
UK, August 23–28, 2020, Proceedings, Part III 16, pages 346–363.
Springer, 2020.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:40 UTC from IEEE Xplore. Restrictions apply.


web search: 
web search: "RMPSNet" Occluded person re-identification regional masking prompt-distribution synergy
web search: 
web search: "Semantic Alignment and Hard Sample Retraining" "Visible-Infrared"
web search: 
web search: CLIP-ReID exploiting vision-language model for image re-identification AAAI 2023
web search: 
web search: "DART" "coupled noisy labels" visible infrared person re-identification
web search: 
web search: "DART" "DART" "VI-ReID" noisy labels
codex
最后一篇已经读完。只有 SRHCL 的关键消融大概率落在第 8 页之后，前 8 页主要是方法和实验设置；为了不把证据闭环写虚，我再额外看它后面的消融页，然后汇总。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ZHAO et al.: SRHCL FOR COUPLED NOISY VISIBLE-INFRARED PERSON RE-IDENTIFICATION

Algorithm 1 Training Pipeline of SRHCL

7147

datasets; the momentum update rate β and λ for regularization
terms are set to be 0.7 and 0.3, respectively. For self-refined
label rectification, we set a fixed noisy threshold η to 0.3,
and we incorporate it into the training process at two and
five regular intervals in experiments of SYSU-MM01 and
RegDB.
C. Results and Comparisons

the cumulative matching curve (CMC) are employed to assess
retrieval accuracy, whereas the mean Inverse Negative Penalty
(mINP) serves as an indicator of the efficiency of the most
challenging correct matches.
B. Implementation Details
Our method is implemented with PyTorch on a GeForce
RTX 3090. As to the hyper-parameters mentioned in section III-D and section III-C, temperature τ is set as 0.25
and 0.1 in contrastive learning for SYSU-MM01 and RegDB

We compare the performance of our method with AGW
[5], DDAG [76], LbA [77], MPANet [56], ADP [3], DART
[21] and LCNL [22] on two datasets under different noise
ratio settings varying from 0%, 20%, and 50%. Among them,
DART [21] and LCNL [22] are designed for Vi-ReID under
several noise settings. The results of SYSU-MM01 and RegDB
are reported in Table II and Table IV, respectively.
1) Baselines:
a) Supervised VI-ReID methods:
• AGW [5]: This article provides a thorough overview and
detailed analysis of person re-identification tasks from
three distinct perspectives. Additionally, it introduces a
robust AGW baseline, which demonstrates competitive
performance in both single-modality and cross-modality
Re-ID tasks.
• DDAG [76]: This paper proposes a dynamic dualattention aggregation learning method, which is designed
to improve the recognition capability and robustness of
the model through the incorporation of two attention
modules.
• LbA [77]: LbA proposes a CMAlign module for
aligning visible and infrared inputs to establish dense
cross-modal correspondences, to minimize the disparity between visible and infrared images at the pixel
level.
• MPANet [56]: This paper proposes a joint modality and
pattern alignment network (MPANet) to mine the subtle texture differences for VI-ReID. MPANet introduces
a modality alleviation module (MAM) to mitigate the
cross-modal sample differences and a pattern alignment
module (PAM) to discover subtle differences from different modalities.
• ADP [3]: This method presents a robust channelaugmented joint learning strategy that incorporates
random channel-exchangeable augmentation and channellevel random erasing, aimed at creating an intermediate
modality for VI-ReID.
• DART [21]: This paper proposes DART to solve the
problem of learning with noisy labels in VI-ReID. DART
detects the noisy samples in datasets by modeling the
loss response and then calculates the confidence weights.
According to the weights, DART divides input sample
pairs into four categories and handles them case by case
for cross-modal feature learning.
• LCNL [22]: This paper proposes an adaptive quadruple
loss with theoretical guarantees. With the novel objective
function, LCNL achieves robust cross-modal learning
against noisy correspondence by adaptively dividing the
training pairs into four groups.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

7148

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 5, MAY 2026

TABLE II
C OMPARATIVE A NALYSES W ITH S TATE - OF - THE -A RT M ETHODS ON THE SYSU-MM01 DATASET A RE C ONDUCTED U NDER N OISE R ATIOS OF 0%, 20%
AND 50%, R ESPECTIVELY. T HE B EST AND S ECOND -B EST R ESULTS A RE H IGHLIGHTED IN B OLD AND U NDERLINED

b) Unsupervised VI-ReID methods:
• RoDE [78]: This paper proposes a novel unspervised VIReID framework to address overfitting to pseudo labels
and error accumulation. It consists of robust adaptive
learning to ensure a selective emphasis on learning and
cross-cluster matching to handle noisy cluster correspondence.
• APPD [79]: This paper presents “APPD” for unspervised
VI-ReID. It proposes adaptive calibration and division
and optimal transport prototype matching for robust
pseudo-labels and cross-modality alignments. Furthermore, debiased memory hybrid learning aims to reduce
cross-modality gaps without introducing sampling biases.
2) Results on SYSU-MM01:
a) Comparison with supervised methods: Our method
shows promising results under the experiment settings mentioned above for the SYSU-MM01 dataset. Our method
achieves competitive performance when the dataset is clean
(0% noise), although it is particularly designed for noisy situations. In the presence of noise, our approach outperforms all
baseline methods and shows its superiority in such scenarios.
In the case of 20% noise, our approach can progressively
rectify the noisy labels owing to the combination of noise
detection and self-refined label rectification, allowing the
model to be sufficiently trained on relatively clean data. The
final results achieve 66.03% mAP, 69.08% Rank-1 accuracy,
and 52.50% mINP for all-search mode and 78.87% mAP,
73.48% Rank-1 accuracy, and 75.42% mINP for indoor-search

TABLE III
C OMPARISON W ITH S TATE - OF - THE -A RT U NSUPERVISED VI-R E ID
(USVI-R E ID) M ETHODS ON THE SYSU-MM01 DATASET

mode, with marginal performance degradation from its clean
counterparts. When the noise ratio is as high as 50%, our
method consistently maintains remarkable performance. The
evaluation results of SRHCL exceed LCNL [22] by 3.01 %
mAP, 3.41% Rank-1 accuracy, and 3.25% mINP for allsearch mode, and 2.96% mAP, 3.40% Rank-1 accuracy, and
3.31% mINP for indoor-search mode, respectively. Compared
to these baselines, SRHCL achieves a new SOTA on retrieval
performance under several noise settings. Results on SYSUMM01 under variant noise ratios demonstrate the effectiveness
and robustness of our method.
b) Comparison with unsupervised methods: To comprehensively and sufficiently evaluate our method’s performance
on noisy datasets, we also selected two recent state-of-the-art
unsupervised methods (RoDE [78] and APPD [79]) for VIReID as baselines. We compared our method’s results on the
SYSU-MM01 dataset under 50% noise scenarios with these
methods. The experimental results are shown in Table III.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

ZHAO et al.: SRHCL FOR COUPLED NOISY VISIBLE-INFRARED PERSON RE-IDENTIFICATION

7149

TABLE IV

TABLE V

C OMPARATIVE A NALYSES W ITH S TATE - OF - THE -A RT M ETHODS ON THE
R EG DB DATASET A RE C ONDUCTED U NDER N OISE R ATIOS OF 0%,
20% AND 50%, R ESPECTIVELY. T HE B EST AND S ECOND -B EST
R ESULTS A RE H IGHLIGHTED IN B OLD AND U NDERLINED

T HE A BLATION S TUDY ON THE SYSU-MM01 DATASET U NDER 20%
N OISE R ATIO W ITH A LL -S EARCH M ODE

TABLE VI
C OMPARISON OF TP AND TN B ETWEEN SRHCL AND DART

D. Ablation Study

As the table shows, our SRHCL method can outperform
unsupervised approaches in high-noise scenarios (50%) by
leveraging clean identity labels within the dataset and effectively utilizing this information.
3) Results on RegDB: Since the RegDB dataset has
a limited scale, we modified the predefined rectification
interval T to every 5 epochs, allowing the model to
undergo sufficient training before each label rectification iteration. SRHCL obtains the best mAP and mINP under all
noise settings. Under the 20% noise setting, our method
reaches 76.96% mAP and 62.37% mINP for visible-tothermal search and 75.57% mAP and 58.84% mINP for
thermal-to-visible search. When the noise ratio increases to
50%, our method surpasses the sub-optimal baseline DART
by 1.79% mAP, 5.28% mINP for visible-to-thermal mode,
and 2.53% mAP, 6.04% mINP for thermal-to-visible mode,
respectively.
However, when the dataset is initially clean (0% noise), the
performance shows a decrease compared to the baseline. The
accuracy of the labels will decrease due to any rectification
attempt since the dataset is already precisely annotated. In this
case, our method assumes that noisy labels exist in the dataset,
and the self-refined label rectification step will recognize hard
samples as noise. This property may reduce performance on
clean datasets, particularly on the relatively small RegDB
dataset.
We investigate this phenomenon further in “Study on the
rectification intervals T” and Table VII below.

To validate the effectiveness of each component in SRHCL,
we conduct ablation studies on SYSU-MM01 under the 20%
noise setting. Table V shows the results from different variants
of our method: “B” denotes the baseline with GMM and
soft identification loss; “+Lcon ” represents the model training along with contrastive learning; “+Lelr ” indicates early
learning regularization is applied; finally, self-refined labelrectification “Rec” is added to form the complete version of
our pipeline.
As demonstrated in the table V, each module in SRHCL
contributes to the final performance. By comparing the baseline B with the variant B + Lcon , we can find that the
contrastive loss effectively improves mINP, which ensures
the lower bound for retrieval. It indicates that cross-modal
contrastive learning plays an important role in overcoming the
modality gap. The variant B + Lelr proves the effectiveness of
the regularization terms, facilitating robust training for noisy
samples by notably improving Rank-1 accuracy and mAP.
The last row shows the results of the complete pipeline of
SRHCL, where self-refined label rectification progressively
converts detected noisy samples into clean samples during the
model training process to gain better performance.
E. Visualization Analysis
In this section, we visualize the performance comparison
to offer an intuitive insight into how our pipeline combats
noise from the aspect of varying noise ratio and the sample
distribution during training. We also conduct experiments to
explore the impact of different hyperparameters by adjusting
them for training.
1) Study on the Generalizability: We visualize the performance comparison between SRHCL and DART under different
noise ratio settings in Figure 4.Our method can still maintain
good performance under high noise ratios. Compared with
DART, when the noise increases from 0% to 50%, mAP in
our method drops from 66.54 % to 62.81 %, with only a
slight decrease of 3.73 %, while DART drops from 66.29 %

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

7150

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 5, MAY 2026

TABLE VII
I NVESTIGATION OF SRHCL AND I TS VARIANTS ON THE C LEAN R EG DB DATASET (0% NOISE ). T HE O RIGINAL T FOR SRHCL I S 5

Fig. 4. Performance comparison between SRHCL and DART on SYSUMM01 under varying noise ratio.
Fig. 6. Comparison of the loss distribution between SRHCL and DART. The
loss distributions between 0.2 and 1.0 are zoomed in for clearer visualization.

Fig. 5. Train set label accuracy during training evaluated by ground truth
labels after each rectification step under varying noise ratios.

to 58.69 %. By comparing the tendencies of results from the
two methods against increasing noise ratio, the mAP of our
pipeline decreases much more slowly than DART when the
noise ratio increases.
2) Study on the Rectification Accuracy: To validate the
effectiveness of the self-refined label rectification in SRHCL,
we plot the curve of the label accuracy after each rectification
step in Figure 5.The curves in the figure show that the accuracy
rates increase rapidly when datasets are initialized with noise,
which proves that our method can distinguish and rectify the
noise in the data. Our pipeline achieves superior final label
accuracy progressively, suggesting that the training process in

SRHCL offers a stable learning framework that guides the
model learning from clean and simple samples to hard and
noisy ones. However, as indicated by the gray curve in the
figure, there is a slight decrease in label accuracy throughout
the training process. It is important to note that our pipeline
may be prone to over-rectifying challenging samples when the
training set is initially clean, causing performance degradation
in this scenario.
3) Study on the Loss Distribution: To explore the distribution of samples based on loss response during training, we
visualize the loss distribution at 10 and 50 epochs under a high
noise ratio(50%) on SYSU-MM01. As depicted in Figure 6(a)
and (c), the loss distribution in DART presents a notable
separation into two distinct components but retains a large
portion of the noisy samples.Although precise noise detection
ensures that the model only focuses on clean samples during
training, this approach concurrently leads to a substantial
decrease in the utilization rate of the collected data. Therefore,
the model’s learning capacity and generalization ability are
limited.
However, our method gradually rectifies the noisy labels,
transferring the noisy samples into clean ones. Comparing the
loss distribution between our methods and DART, it can be
seen that the number of noisy samples decreases significantly
and the number of clean samples increases accordingly (as
emphasized by the red box on the sample number scale). As
calculated in Table VI, where TP means those clean samples
with loss lower than 0.5 and TN means those noisy samples

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.

ZHAO et al.: SRHCL FOR COUPLED NOISY VISIBLE-INFRARED PERSON RE-IDENTIFICATION

7151

Fig. 7. Rank-10 retrieval result on SYSU-MM01 evaluation set. Both DART and our method SRHCL are trained under 20% noise.

with related loss value larger than 0.5, our method transfers an
addition of 25136 noisy samples into clean samples for model
training. This way, more data samples are exposed to our
model during training, thus achieving better performance and
generalization in noisy conditions. Therefore, the abundance of
clean samples provides the model with a substantial foundation
for improving performance.
4) Study on the Retrieval Performance: To showcase the
effectiveness of self-refined label rectification and the robust
retrieval results of our pipeline, we visualize the Rank-10
retrieval results from the evaluation set. When comparing the
first row from DART with the second row from SRHCL, it
is evident that our method delivers precise retrieval results
for this query. In contrast, DART’s performance appears to
be affected by specific patterns in the infrared modality. The
third and fourth rows illustrate a typical instance of noisy
annotations, where an incorrect identity, despite being dissimilar to the query, appears in the rank-10 retrieval results. It
demonstrates that the self-refined label rectification effectively
corrects these noisy samples, thereby reducing their negative
impact on the retrieval performance.
5) Study on the Rectification Intervals T: To explore how
the label rectification interval T influences model learning
in SRHCL, we conduct experiments on the RegDB dataset
with a high noise ratio for further analysis. Figure 8 shows
the results of experiments on the RegDB dataset under 50%
noise with different label rectification intervals (1, 2, 5, 7,
and 10). The contrastive temperature τ is fixed as 0.1. The
results indicate that the performance under a high noise ratio

Fig. 8. Impact of varying rectification interval T in section III-C on results
of the RegDB dataset under 50% noise.

improves with the increasing duration of rectification intervals,
suggesting that longer rectification intervals promote a more
stable training process when working with a small dataset.
This can be attributed to regularization terms between current
model outputs and their historical memory that prevent the
models from overfitting to noise. This mechanism ensures
that the model obtains sufficient training on the available
data through longer training intervals, fostering stability in
the model outputs and the associated historical memory. As
a result, it contributes to more reliable and precise predictions
for label rectification.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:44 UTC from IEEE Xplore. Restrictions apply.


web search: 
web search: CLIP-ReID exploiting vision-language model for image re-identification arXiv
web search: 
web search: "Corruption-Invariant Learning" person re-identification NeurIPS 2021
web search: 
web search: "Self-Rectification Historical Consistency Learning" "SRHCL"
web search: 
web search: "Fused Gromov-Wasserstein" arXiv
web search: 
web search: "Optimal Transport for structured data with application on graphs" "Fused Gromov-Wasserstein"
codex
我按指定方式读了每篇前 8 页的方法部分；SRHCL 的关键消融在后页，所以额外看了第 8 到 12 页来补证据闭环。下面只拆“创新是怎么构造出来的”。

**RMPSNet**
1. **触发观察**：作者抓住三个具体失败点。遮挡会造成视觉语义缺失，CLIP-ReID 的图文全局对齐在遮挡下会错位，常用随机擦除默认遮挡均匀分布，但真实遮挡更多集中在下半身，尤其腿部。
2. **重定义动作**：它把遮挡 ReID 从“局部特征缺失”改写成“遮挡分布不匹配加跨模态语义退化加增强特征分布不一致”。关键词是 `occlusion distribution mismatch`、`semantic degradation`、`prompt-distribution synergy`、`region-prioritized erasure`。
3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
4. **证据闭环**：关键不是只报总成绩，而是三类消融分别对应三类重定义。`Baseline+RPE` 在 Occluded-Duke 上大幅超过普通随机擦除，证明“遮挡分布先验”有用；`Baseline+DMPA` 和去掉 DMPA 的版本证明文本语义稳态不是装饰；MDO 的 only enhancement 和完整 MDO 对比证明两步对抗比普通扰动更有效。
5. **审稿人为什么买账**：它卖的是“视角加组合机制”。单个模块都不算特别新，但作者把 CLIP 遮挡失败拆成分布、语义、鲁棒性三个可验证子问题，每个子问题都有对应模块和消融，所以看起来不像随便堆模块。真正新意在“真实遮挡分布和 prompt 稳定性一起处理”，不是某个具体操作本身。

**Robust Mixed-Degradation Person Re-identification via Structural Consistency Distillation**
1. **触发观察**：以往退化 ReID 往往假设单一退化，比如雾、低光、雨。作者指出真实监控里干净图、雨、雪、雾、曝光变化会混在一起，query 和 gallery 还可能是不同退化。
2. **重定义动作**：它把“退化域偏移”改写成“混合退化下的双重错位”。第一层是多个退化造成多个中心偏移，第二层是同一身份在特征空间里的几何关系被扭曲。关键词是 `mixed degradation`、`heterogeneity gap`、`multiple centroid shifts`、`distorted topology`、`structural consistency`。
3. **机制怎么长出来**：如果问题是“中心偏移加拓扑扭曲”，那只用 Wasserstein 对齐中心不够，只用 Gromov-Wasserstein 保结构也不够，所以 SCD 用 Fused Gromov-Wasserstein 同时对齐干净和退化特征的全局分布与样本间结构。CFR 用 EWC 保住干净图识别能力，解决混合退化训练导致干净知识遗忘。这里 SCD 和重定义绑定很紧，CFR 是辅助闭环。
4. **证据闭环**：最关键的消融是 MSE、WD、GWD、WD&GWD 和 FGW 的对比。它不是证明“一个新 loss 涨点”，而是在验证“只对齐分布或只保结构都不够，混合退化必须二者统一建模”。另外 CE、DE、DQ、DG 四种评测设置和长尾退化、替代合成退化实验，也是在证明问题定义不是单一退化过拟合。
5. **审稿人为什么买账**：这篇主要卖“问题重定义”。机制使用的是已有最优传输工具，但问题被讲得很清楚：真实退化不是单域偏移，而是随机混合造成结构破坏。再加上新 benchmark 和多设置评测，审稿人容易认为它补了一个真实部署缺口。

**See What You Seek: Semantic Contextual Integration**
1. **触发观察**：换衣 ReID 里，很多方法要么压制衣服，要么引入轮廓、姿态、步态、解析等外部线索。作者的关键观察是：视觉特征里不只有负面的衣服因素，也有正面的稳定身份因素，比如头发、眼镜、包、鞋等，直接“去衣服”会漏掉这部分语义。
2. **重定义动作**：它把换衣问题从“消除衣服”改写成“分离负语义并显式挖出正语义”。关键词是 `negative factors`、`positive factors`、`semantic contextual integration`、`semantic separation`、`orthogonalized text features`。
3. **机制怎么长出来**：SSE 用两个 prompt，一个描述 person，一个描述 clothes。先得到身份文本特征和衣服文本特征，再把衣服方向从身份方向里投影并减掉，得到去衣服后的语义特征。SIM 再让这个正交化文本特征通过非局部操作和跨注意力去指导视觉特征。这个设计和“负语义分离、正语义引导”的重定义绑定很紧。
4. **证据闭环**：它的关键消融是 SSE 单独有效，SIM 单独在 CNN 上甚至会伤性能，SSE 加 SIM 才最好。这证明“文本指导本身不够，必须先把衣服语义分离干净”。t-SNE 更紧凑、prompt 引导热图更关注头肩、下身、鞋，特征相似度分布在正交化后更清晰，这些可视化都在支撑“语义分离是对的”。
5. **审稿人为什么买账**：这篇卖的是视角。CLIP-ReID 已经有 ID prompt，CAL 已经有衣服对抗，但它把换衣讲成“负因素和正因素同时存在”的语义问题，用文本空间正交化给出可解释动作。新意不是“用了 CLIP”，而是把衣服偏置从语义空间里切出来，再反过来指导视觉。

**Self-Rectification Historical Consistency Learning**
1. **触发观察**：可见光红外 ReID 的标注噪声有两层。单模态身份标签会错，这是 noisy annotation；两模态按错误标签配对后还会产生 noisy correspondence。已有 DART、LCNL 主要检测并压低噪声样本，结果是数据没被充分利用。
2. **重定义动作**：它把噪声样本从“需要丢掉的坏样本”改写成“可以被历史预测逐步纠正的潜在有效样本”。同时把训练波动从坏现象改写成信息来源。关键词是 `self-rectification`、`historical consistency`、`early learning`、`momentum-updated memories`、`coupled noisy labels`。
3. **机制怎么长出来**：先用双网络互评和 GMM 建模 loss，给每个样本算干净置信度。再用动量更新的 logits 和 feature memory 保存历史输出。只有两个网络对历史预测达成一致时，才给噪声样本改标签。训练时一边用置信度重加权身份损失和跨模态对比损失，一边用历史一致性约束防止当前输出剧烈漂移。机制和重定义绑定非常紧。
4. **证据闭环**：核心消融是 baseline、加跨模态对比、加 early learning regularization、再加 label rectification 的逐步对比。可视化更关键：不同噪声率下标签准确率随 rectification 逐步上升，loss 分布里噪声样本被转成低损失干净样本，和 DART 对比显示它不是单纯抗噪，而是在回收训练数据。
5. **审稿人为什么买账**：这篇卖的是训练范式，不是网络结构。它抓住“丢噪声样本会浪费数据”这个审稿人容易认可的矛盾，再用历史一致性避免伪标签乱改。新意来源在“历史预测驱动的噪声标签自校正”，尤其适配 VI-ReID 的双模态耦合噪声。

**Semantic Alignment and Hard Sample Retraining**
1. **触发观察**：作者指出 VI-ReID 里三个常见做法都有盲点。水平分块会造成跨模态语义错位，batch 内 hard mining 看不到全数据的难样本，PK sampler 不能保证视角均衡。
2. **重定义动作**：它把 VI-ReID 从“模态差异对齐”改写成“语义错位、全局难样本遗漏、视角采样偏置共同造成的不稳训练”。关键词是 `semantic alignment`、`global hard samples`、`viewpoint-balanced sampler`、`recurrent semantic aggregation`。
3. **机制怎么长出来**：RSA 把特征图切成局部 patch，用 BiLSTM 按人体空间顺序聚合局部语义，再拼全局特征。Lma 在每个时间步对齐可见光和红外的分布，并且越到后面权重越大，因为后期局部上下文更完整。CHSR 用分类置信度在全数据里找容易混淆的身份，重新组成 hard batch 训练。VB sampler 保证不同摄像机视角均衡进入 batch。
4. **证据闭环**：证据做得比较完整。模块消融显示 RSA、CHSR、VB 逐步增益；RSA 类型对比里 BiLSTM 优于 Transformer encoder、decoder 和单向 LSTM，支撑“人体局部顺序有价值”；Lma 的递增权重优于平均和递减权重，支撑“逐步语义聚合后再强对齐”；Grad-CAM 和 t-SNE 说明注意力更集中、两模态同身份更靠近。
5. **审稿人为什么买账**：这篇卖的是机制闭环多于视角。三个问题不算同一个根因，但每个问题都有直接干预和消融验证。它适合会议稿，因为贡献清楚、实现不重、每个模块都能解释为什么有效。

**Semantic Consistency And Integrity Network**
1. **触发观察**：作者注意到现有换衣方法常把衣服遮掉、忽略掉，或依赖姿态、步态、解析等辅助线索，这会损害行人特征的语义完整性。另一个观察是，头部等区域相对稳定，衣服偏置又和身份特征纠缠在一起。
2. **重定义动作**：它把换衣 ReID 从“去衣服偏置”改写成“在保持语义完整性的前提下，学习语义一致特征并量化衣服偏置”。关键词是 `semantic consistency`、`semantic integrity`、`head soft attention`、`clothing bias`、`causal intervention`。
3. **机制怎么长出来**：三分支分别承担三件事。头部分支用 SCHP 解析得到头、上身、下身、脚的 mask，再用 body part matching loss 学 head soft attention。SCC 用 raw branch 和 head branch 的 class activation map、saliency map 做一致性，让原图特征学到稳定语义区域。衣服分支提取局部衣服特征，通过 KL 知识转移让 raw branch 感知其中的衣服偏置，再用因果干预损失比较干预前后身份预测，逼模型剥离衣服影响。
4. **证据闭环**：主要证据是三分支消融。只加头部分支、只加衣服分支都有提升，两者合起来最好，说明“语义一致区域”和“衣服偏置剥离”是互补的。不过它对“语义完整性没有被破坏”的证据相对弱，更多是通过性能和设计叙事间接成立。
5. **审稿人为什么买账**：这篇卖的是视角包装加可实现机制。真正新意在“不要粗暴遮衣服，而是保持完整语义再做偏置剥离”。但它依赖 human parsing，因果干预也更像可解释的损失设计，严谨性不如前几篇强。能买账主要因为故事完整、模块简单、LTCC 和 PRCC 上有直接收益。

**共同套路**
这批论文的共同构造方式不是先想模块，而是先抓一个旧假设的漏洞：遮挡不是均匀的，退化不是单一的，换衣不是只去衣服，噪声样本不是只能丢，难样本不是只在 batch 里。然后把漏洞改写成两个或三个可命名的子问题，每个子问题配一个能被消融验证的机制。最值得学的是“重定义动作要能推出机制，证据也要反过来证明这个重定义”，否则就会退化成堆模块。

联网核查主要参考了这些先例：CLIP-ReID 的两阶段 prompt 训练、CAL 的衣服对抗、SCI 的公开版本、CIL-ReID 的退化鲁棒基准、FGW 的结构和特征联合对齐思想。链接分别是 [CLIP-ReID](https://arxiv.org/abs/2211.13977)、[CAL](https://arxiv.org/abs/2204.06890)、[SCI](https://arxiv.org/abs/2412.01345)、[CIL-ReID](https://arxiv.org/abs/2111.00880)、[FGW](https://arxiv.org/abs/1805.09114)。
tokens used
230,935
我按指定方式读了每篇前 8 页的方法部分；SRHCL 的关键消融在后页，所以额外看了第 8 到 12 页来补证据闭环。下面只拆“创新是怎么构造出来的”。

**RMPSNet**
1. **触发观察**：作者抓住三个具体失败点。遮挡会造成视觉语义缺失，CLIP-ReID 的图文全局对齐在遮挡下会错位，常用随机擦除默认遮挡均匀分布，但真实遮挡更多集中在下半身，尤其腿部。
2. **重定义动作**：它把遮挡 ReID 从“局部特征缺失”改写成“遮挡分布不匹配加跨模态语义退化加增强特征分布不一致”。关键词是 `occlusion distribution mismatch`、`semantic degradation`、`prompt-distribution synergy`、`region-prioritized erasure`。
3. **机制怎么长出来**：RPE 从真实遮挡分布自然推出，下半身优先擦除。DMPA 从“语义缺失”推出，在文本 prompt 上做双重 mask，用文本到文本对比让身份文本原型对缺失语义稳定。MDO 从“增强后分布不一致”推出，在特征空间做擦除、噪声、局部替换，再用两步对抗优化逼原始和扰动特征一致。RPE 和重定义绑定最紧，DMPA 次之，MDO 更像通用鲁棒性补强。
4. **证据闭环**：关键不是只报总成绩，而是三类消融分别对应三类重定义。`Baseline+RPE` 在 Occluded-Duke 上大幅超过普通随机擦除，证明“遮挡分布先验”有用；`Baseline+DMPA` 和去掉 DMPA 的版本证明文本语义稳态不是装饰；MDO 的 only enhancement 和完整 MDO 对比证明两步对抗比普通扰动更有效。
5. **审稿人为什么买账**：它卖的是“视角加组合机制”。单个模块都不算特别新，但作者把 CLIP 遮挡失败拆成分布、语义、鲁棒性三个可验证子问题，每个子问题都有对应模块和消融，所以看起来不像随便堆模块。真正新意在“真实遮挡分布和 prompt 稳定性一起处理”，不是某个具体操作本身。

**Robust Mixed-Degradation Person Re-identification via Structural Consistency Distillation**
1. **触发观察**：以往退化 ReID 往往假设单一退化，比如雾、低光、雨。作者指出真实监控里干净图、雨、雪、雾、曝光变化会混在一起，query 和 gallery 还可能是不同退化。
2. **重定义动作**：它把“退化域偏移”改写成“混合退化下的双重错位”。第一层是多个退化造成多个中心偏移，第二层是同一身份在特征空间里的几何关系被扭曲。关键词是 `mixed degradation`、`heterogeneity gap`、`multiple centroid shifts`、`distorted topology`、`structural consistency`。
3. **机制怎么长出来**：如果问题是“中心偏移加拓扑扭曲”，那只用 Wasserstein 对齐中心不够，只用 Gromov-Wasserstein 保结构也不够，所以 SCD 用 Fused Gromov-Wasserstein 同时对齐干净和退化特征的全局分布与样本间结构。CFR 用 EWC 保住干净图识别能力，解决混合退化训练导致干净知识遗忘。这里 SCD 和重定义绑定很紧，CFR 是辅助闭环。
4. **证据闭环**：最关键的消融是 MSE、WD、GWD、WD&GWD 和 FGW 的对比。它不是证明“一个新 loss 涨点”，而是在验证“只对齐分布或只保结构都不够，混合退化必须二者统一建模”。另外 CE、DE、DQ、DG 四种评测设置和长尾退化、替代合成退化实验，也是在证明问题定义不是单一退化过拟合。
5. **审稿人为什么买账**：这篇主要卖“问题重定义”。机制使用的是已有最优传输工具，但问题被讲得很清楚：真实退化不是单域偏移，而是随机混合造成结构破坏。再加上新 benchmark 和多设置评测，审稿人容易认为它补了一个真实部署缺口。

**See What You Seek: Semantic Contextual Integration**
1. **触发观察**：换衣 ReID 里，很多方法要么压制衣服，要么引入轮廓、姿态、步态、解析等外部线索。作者的关键观察是：视觉特征里不只有负面的衣服因素，也有正面的稳定身份因素，比如头发、眼镜、包、鞋等，直接“去衣服”会漏掉这部分语义。
2. **重定义动作**：它把换衣问题从“消除衣服”改写成“分离负语义并显式挖出正语义”。关键词是 `negative factors`、`positive factors`、`semantic contextual integration`、`semantic separation`、`orthogonalized text features`。
3. **机制怎么长出来**：SSE 用两个 prompt，一个描述 person，一个描述 clothes。先得到身份文本特征和衣服文本特征，再把衣服方向从身份方向里投影并减掉，得到去衣服后的语义特征。SIM 再让这个正交化文本特征通过非局部操作和跨注意力去指导视觉特征。这个设计和“负语义分离、正语义引导”的重定义绑定很紧。
4. **证据闭环**：它的关键消融是 SSE 单独有效，SIM 单独在 CNN 上甚至会伤性能，SSE 加 SIM 才最好。这证明“文本指导本身不够，必须先把衣服语义分离干净”。t-SNE 更紧凑、prompt 引导热图更关注头肩、下身、鞋，特征相似度分布在正交化后更清晰，这些可视化都在支撑“语义分离是对的”。
5. **审稿人为什么买账**：这篇卖的是视角。CLIP-ReID 已经有 ID prompt，CAL 已经有衣服对抗，但它把换衣讲成“负因素和正因素同时存在”的语义问题，用文本空间正交化给出可解释动作。新意不是“用了 CLIP”，而是把衣服偏置从语义空间里切出来，再反过来指导视觉。

**Self-Rectification Historical Consistency Learning**
1. **触发观察**：可见光红外 ReID 的标注噪声有两层。单模态身份标签会错，这是 noisy annotation；两模态按错误标签配对后还会产生 noisy correspondence。已有 DART、LCNL 主要检测并压低噪声样本，结果是数据没被充分利用。
2. **重定义动作**：它把噪声样本从“需要丢掉的坏样本”改写成“可以被历史预测逐步纠正的潜在有效样本”。同时把训练波动从坏现象改写成信息来源。关键词是 `self-rectification`、`historical consistency`、`early learning`、`momentum-updated memories`、`coupled noisy labels`。
3. **机制怎么长出来**：先用双网络互评和 GMM 建模 loss，给每个样本算干净置信度。再用动量更新的 logits 和 feature memory 保存历史输出。只有两个网络对历史预测达成一致时，才给噪声样本改标签。训练时一边用置信度重加权身份损失和跨模态对比损失，一边用历史一致性约束防止当前输出剧烈漂移。机制和重定义绑定非常紧。
4. **证据闭环**：核心消融是 baseline、加跨模态对比、加 early learning regularization、再加 label rectification 的逐步对比。可视化更关键：不同噪声率下标签准确率随 rectification 逐步上升，loss 分布里噪声样本被转成低损失干净样本，和 DART 对比显示它不是单纯抗噪，而是在回收训练数据。
5. **审稿人为什么买账**：这篇卖的是训练范式，不是网络结构。它抓住“丢噪声样本会浪费数据”这个审稿人容易认可的矛盾，再用历史一致性避免伪标签乱改。新意来源在“历史预测驱动的噪声标签自校正”，尤其适配 VI-ReID 的双模态耦合噪声。

**Semantic Alignment and Hard Sample Retraining**
1. **触发观察**：作者指出 VI-ReID 里三个常见做法都有盲点。水平分块会造成跨模态语义错位，batch 内 hard mining 看不到全数据的难样本，PK sampler 不能保证视角均衡。
2. **重定义动作**：它把 VI-ReID 从“模态差异对齐”改写成“语义错位、全局难样本遗漏、视角采样偏置共同造成的不稳训练”。关键词是 `semantic alignment`、`global hard samples`、`viewpoint-balanced sampler`、`recurrent semantic aggregation`。
3. **机制怎么长出来**：RSA 把特征图切成局部 patch，用 BiLSTM 按人体空间顺序聚合局部语义，再拼全局特征。Lma 在每个时间步对齐可见光和红外的分布，并且越到后面权重越大，因为后期局部上下文更完整。CHSR 用分类置信度在全数据里找容易混淆的身份，重新组成 hard batch 训练。VB sampler 保证不同摄像机视角均衡进入 batch。
4. **证据闭环**：证据做得比较完整。模块消融显示 RSA、CHSR、VB 逐步增益；RSA 类型对比里 BiLSTM 优于 Transformer encoder、decoder 和单向 LSTM，支撑“人体局部顺序有价值”；Lma 的递增权重优于平均和递减权重，支撑“逐步语义聚合后再强对齐”；Grad-CAM 和 t-SNE 说明注意力更集中、两模态同身份更靠近。
5. **审稿人为什么买账**：这篇卖的是机制闭环多于视角。三个问题不算同一个根因，但每个问题都有直接干预和消融验证。它适合会议稿，因为贡献清楚、实现不重、每个模块都能解释为什么有效。

**Semantic Consistency And Integrity Network**
1. **触发观察**：作者注意到现有换衣方法常把衣服遮掉、忽略掉，或依赖姿态、步态、解析等辅助线索，这会损害行人特征的语义完整性。另一个观察是，头部等区域相对稳定，衣服偏置又和身份特征纠缠在一起。
2. **重定义动作**：它把换衣 ReID 从“去衣服偏置”改写成“在保持语义完整性的前提下，学习语义一致特征并量化衣服偏置”。关键词是 `semantic consistency`、`semantic integrity`、`head soft attention`、`clothing bias`、`causal intervention`。
3. **机制怎么长出来**：三分支分别承担三件事。头部分支用 SCHP 解析得到头、上身、下身、脚的 mask，再用 body part matching loss 学 head soft attention。SCC 用 raw branch 和 head branch 的 class activation map、saliency map 做一致性，让原图特征学到稳定语义区域。衣服分支提取局部衣服特征，通过 KL 知识转移让 raw branch 感知其中的衣服偏置，再用因果干预损失比较干预前后身份预测，逼模型剥离衣服影响。
4. **证据闭环**：主要证据是三分支消融。只加头部分支、只加衣服分支都有提升，两者合起来最好，说明“语义一致区域”和“衣服偏置剥离”是互补的。不过它对“语义完整性没有被破坏”的证据相对弱，更多是通过性能和设计叙事间接成立。
5. **审稿人为什么买账**：这篇卖的是视角包装加可实现机制。真正新意在“不要粗暴遮衣服，而是保持完整语义再做偏置剥离”。但它依赖 human parsing，因果干预也更像可解释的损失设计，严谨性不如前几篇强。能买账主要因为故事完整、模块简单、LTCC 和 PRCC 上有直接收益。

**共同套路**
这批论文的共同构造方式不是先想模块，而是先抓一个旧假设的漏洞：遮挡不是均匀的，退化不是单一的，换衣不是只去衣服，噪声样本不是只能丢，难样本不是只在 batch 里。然后把漏洞改写成两个或三个可命名的子问题，每个子问题配一个能被消融验证的机制。最值得学的是“重定义动作要能推出机制，证据也要反过来证明这个重定义”，否则就会退化成堆模块。

联网核查主要参考了这些先例：CLIP-ReID 的两阶段 prompt 训练、CAL 的衣服对抗、SCI 的公开版本、CIL-ReID 的退化鲁棒基准、FGW 的结构和特征联合对齐思想。链接分别是 [CLIP-ReID](https://arxiv.org/abs/2211.13977)、[CAL](https://arxiv.org/abs/2204.06890)、[SCI](https://arxiv.org/abs/2412.01345)、[CIL-ReID](https://arxiv.org/abs/2111.00880)、[FGW](https://arxiv.org/abs/1805.09114)。
