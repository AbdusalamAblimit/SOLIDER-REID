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
session id: 019ef71f-67e3-7be3-b039-717daf4042e5
--------
user
你是 ReID 论文创新挖掘员 5/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

== 团队资产(新创新点要能挂上其中之一)==
- SOLIDER-Swin 强 backbone(自监督人体预训练,in_planes=768)
- aerial-ground 跨视角 ReID(CARGO / AG-ReID.v2,航拍↔地面极端视角+低清)
- pose 热图门控(PSG / LGPA-D,姿态引导空间 gating)
- SMPL 3D 几何(mesh/joints/2D投影,团队已打通基建)

== 目标 ==
找能投 B 类(Pattern Recognition / TMM / TCSVT / AAAI 级)的**强创新点**。不要中等工程组合,要问题层面或机制层面有真新意的。

== 每篇提取(简洁)==
(1) 创新类型: 问题重定义 / 新机制 / 新数据-设定 / 工程组合(标注哪种)
(2) 为什么能发: 填了什么 gap + 证据链怎么搭的
(3) story 套路: 怎么把卖点讲成 headline 的

== 然后综合这批,产出 2-4 个针对团队资产的强创新点候选 ==
每个候选要: a) 一句话 headline; b) 挂哪个团队资产; c) 和这批里最像的工作的区别(切开点); d) cheap kill-switch(怎么花最小代价首验真假)。

这批论文(5/20):
Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf
Confidence guided semi-supervised cross-modality person re-identification.pdf
Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf
Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf
Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf
Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我先按项目规则确认当前目标和状态，然后只用 `pdftotext -l 3` 抽每篇前三页。后面会把论文总结压缩到可用于反推创新点的层面，不做逐段综述。
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
/bin/zsh -lc "sed -n '1,180p' ../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc 'pwd && ls' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

exec
/bin/zsh -lc "sed -n '1,220p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,180p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
项目目标确认是继续做文献差距分析，先不开实验。现在开始逐篇读前三页，我会重点抓“它为什么能发”和“我们能从它反推什么更强的问题或机制”。
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
/bin/zsh -lc "pdftotext -l 3 'Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025

5307

Context-Aided Semantic-Aware Self-Alignment
for Video-Based Person Re-Identification
Zhidan Ran , Graduate Student Member, IEEE, Zhiyao Xiao, Xiaobo Lu , Xuan Wei, and Wei Liu

Abstract— Video-based person re-identification (Re-ID) aims at
associating the video sequences of the identical person across multiple cameras. The ubiquitous appearance misalignment poses a
major obstacle for video person Re-ID. Existing alignment-based
methods generally rely on off-the-shelf semantic parsing models
to locate visible human parts, which ignore identifiable personal
belongings and cannot handle various interferences (e.g., pedestrian detection errors and occlusions) in video clips. In this
work, we propose a novel framework termed Context-Aided
Semantic-Aware Self-Alignment (CSSA) for video-based person
Re-ID. First, we propose to jointly learn pixel-level part-aligned
representations and semantic-aligned global-level representations
in an end-to-end manner. Unlike most existing approaches that
depend on prior information in terms of pose for part estimation,
CSSA can locate different body parts and achieve the pixel-level
semantic alignment without extra human topology semantics.
Second, a Context-Aided Region Enhancement (CARE) module is
proposed to efficiently highlight macro-visual patterns associated
with the target pedestrian and suppress noise caused by factors
like background clutters and occlusions. Third, we propose
a Semantic-Aware Global Feature Alignment (SGFA) method
for generating pair-wise semantic-aligned global representations,
which play an essential role in both the training and inference
phases. Extensive experimental results on multiple challenging
benchmarks indicate the superiority and effectiveness of the
proposed CSSA.
Index Terms— Video-based person re-identification, semanticaware feature alignment, context-aided region enhancement,
image retrieval.

I. I NTRODUCTION
ERSON re-identification (Re-ID) [1], which attempts
to identify a particular pedestrian across multiple nonoverlapping cameras, has attracted increasing attention
because of its applications in monitoring, such as cross-scene
crowd counting, criminal investigation, and finding people of
interest (e.g., criminals or lost kids). Thanks to the significant
progress in deep learning, person Re-ID has accomplished
impressive performance. However, Re-ID remains a challenging undertaking owing to the presence of miscellaneous
occlusions (e.g., vehicles, trees, and other pedestrians), complex camera environments, illumination changes, etc.

P

Received 8 May 2024; revised 6 November 2024; accepted 17 January 2025.
Date of publication 20 January 2025; date of current version 6 June 2025. This
work was supported in part by the National Natural Science Foundation of
China under Grant 62271143 and in part by the Big Data Computing Center
of Southeast University. This article was recommended by Associate Editor
A. Iosifidis. (Corresponding author: Xiaobo Lu.)
The authors are with the School of Automation, Southeast University,
Nanjing 210096, China, and also with the Key Laboratory of Measurement
and Control of Complex Systems of Engineering, Ministry of Education,
Nanjing 210096, China (e-mail: zhidan_r@163.com; 230198548@seu.edu.cn;
xblu2013@126.com; wx1204@seu.edu.cn; liu_wei@seu.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3531883

Fig. 1. Instances of video-based Re-ID handicaps. As illustrated by the
images boxed in red rectangles, person appearance is influenced by (a-b)
pedestrian detection errors and (c) occlusions.

Generally, existing work on person Re-ID can be summarized into two classifications, i.e., image-based Re-ID [2], [3]
and video-based Re-ID [4], [5]. For the former, the gallery and
query samples are images, while for the latter, the gallery and
query samples are represented by video clips. Compared with
individual images, video data with multiple frames contains
richer appearance and temporal information. Therefore, we are
motivated to study the topic of person Re-ID in the video
setting.
The primary issue of video-based person Re-ID lies in
aggregating discriminative information from several frames in
video sequences. Most existing approaches employ a global
strategy, learning a global-level representation of the pedestrian target as a single embedding vector. However, the features
learned from the entire bounding boxes might contain misleading information due to occluding objects and background
clutter. On the other hand, global feature representation learning may ignore fine-grained information that contains essential
cues for recognizing identities.
Some recent work has shown promising results in solving
the Re-ID problem by generating several local embedding
vectors, i.e., one for each component of the target object.
Nonetheless, as shown in Fig. 1, learning such part-based
representations is inherently difficult due to the ubiquitous
misalignment problem, which is frequently caused by inaccurate human detection, partial occlusions, camera viewpoint
changes, or body posture variations. In order to alleviate the
misalignment issue, previous solutions are generally divided
into two categories: hard stripe-based [6] and external cuesbased [7]. The first group leverages uniform partitioning
to generate the horizontal stripe parts for extracting local

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:23 UTC from IEEE Xplore. Restrictions apply.

5308

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025

feature representations. Inevitably, the stripe-based partition
is sensitive to large background clutter and is too coarse to
achieve precise semantic alignment. The second group relies
on external detectors, such as pose estimation [8] and body
parsing [9], to locate human parts for semantic and spatial
alignment. This strategy results in a significant computational
overhead and is limited by complicated occlusions. More
crucially, the distinctive individual accessories (e.g., backpacks
and handbags), which are the potentially important contextual
clues for pedestrian recognition, are misinterpreted as background by these pre-trained external models.
In this work, we propose a novel video-based person
Re-ID framework, termed Context-Aided Semantic-Aware
Self-Alignment (CSSA), which can locate different body
components and achieve pixel-level semantic alignment without prior human topology information. CSSA jointly learns
pixel-level part-aligned representations and semantic-aligned
global-level representations in an end-to-end manner. In particular, we design a classifier trained with human body
pseudo-labels to achieve part estimation. For image frames in
the input video clips, we first separate the entire foreground
from the background based on the reasonable hypothesis that
the feature maps exhibit a stronger response in the foreground
than in the background [10], [11]. To ensure semantic consistency, we select a fixed number of video sequences for
each identity and cluster their foreground pixels. The cluster
IDs are employed for human body parts as pseudo-labels.
We iteratively generate pseudo-labels to supervise the semantic
estimation. The confidence maps of the classifier are used to
obtain the part-aligned representations.
For background clutters, especially occlusions, we further
propose the Context-Aided Region Enhancement (CARE)
module, which aims to highlight responses from relevant
regions and restrain noise interference responses. Existing
spatial attention modules do not fully utilize the temporal
context knowledge in the whole video sequence. By contrast,
CARE integrates global spatio-temporal information into a
video-level feature and adopts a concise autoencoder network
to suppress noise. Then, a cosine similarity matrix is computed
to highlight macro-visual patterns associated with the target
pedestrian.
In terms of global-level representation learning, we propose the Semantic-Aware Global Feature Alignment (SGFA)
scheme for generating pair-wise semantic-aligned global features. Our key observation is that the contribution of the
same semantic part is inconsistent across various video clips.
Accordingly, SGFA calculates the adjustment weight based
on the confidence map of each semantic part and obtains the
video-level global features through the weighted aggregation
of the feature maps.
In summary, the main contributions of this article are as
follows:
• We propose the CSSA framework that addresses appearance misalignment in video person Re-ID while learning
pixel-level part-aligned representations and semanticaligned global-level representations in an end-to-end
manner. CSSA can locate different body parts and
achieve pixel-level semantic alignment without prior
human topology information.
• We propose a novel Context-Aided Region Enhancement (CARE) module, which can efficiently highlight

macro-visual patterns associated with the target pedestrian and suppress noise caused by factors like background clutters and occlusions.
• A Semantic-Aware Global Feature Alignment (SGFA)
method is proposed for generating pair-wise semanticaligned global representations, which play an essential
role in both the training and inference phases.
• The proposed CSSA achieves superior performance on
three video-based person Re-ID benchmarks compared
with current state-of-the-art approaches.
II. R ELATED W ORK
A. Video-Based Person Re-ID Models
Recently, video-based Re-ID has attracted widespread attention from researchers in academia and industry [12]. Compared
with image data, video data consisting of multiple frames
contain richer temporal and spatial information, which can be
leveraged to alleviate many issues such as motion blurs and
occlusions, thereby achieving more robust Re-ID [13]. This
part reviews some closely related work on video-based Re-ID.
Some researchers use optical flow for encoding the
short-term temporal cues between adjacent image frames [14],
[15], [16]. Chung et al. [14] design a two-stream siamese
convolutional neural network for processing spatial content
and optical flow information. Simonyan et al. [15] process
optical flow and color features concurrently for the following
post-fusion of their separate classification scores. Feichtenhofer et al. [16] exploit motion information based on an input
stack of several vertical and horizontal optical flow frames.
Temporal pooling is extensively utilized for aggregating
features across different time stamps [17]. Wu et al. [18]
design the feature embedding function as an end-to-end CNN
model with temporal average pooling. Zheng et al. [19] adopt
mean/max pooling for generating video-level representations.
In [20], a global average pooling layer is employed for transforming the spatial feature maps from different spatial regions.
Temporal pooling demonstrates intriguing efficacy, but also
results in a significant loss of discriminative information.
Recurrent Neural Network (RNN) is employed for representation construction in video person Re-ID. In order to
learn the sequence-level feature representations, Yan et al. [21]
introduce the long-short term memory model for learning the
frame-wise human region information from all time stamps.
In [22], attentive temporal pooling is simultaneously learning with spatial pooling layers and recurrent-convolutional
networks, guiding the proposed ASTPN to achieve efficient
temporal information extraction. Liu et al. [23] leverage a
refinement recurrent unit for restoring the missing information
and suppressing the noisy part of the current frame feature
representations by referring to previous frames. Due to the
complicated structure, RNN is difficult to train and introduces
a large number of parameters, making it show certain limitations in modeling temporal cues for Re-ID tasks.
3D convolution is expert in encoding the relative position
and handling local temporal relations. In [24], 3D convolution
models temporal information while maintaining the appearance representation quality. Li et al. [25] employ a multi-scale
3D convolutional layer implemented using multiple parallel
temporal convolution kernels with distinct temporal ranges.
Qiu et al. [26] disassemble the 3D convolutional filters

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:23 UTC from IEEE Xplore. Restrictions apply.

RAN et al.: CONTEXT-AIDED SEMANTIC-AWARE SELF-ALIGNMENT FOR VIDEO-BASED PERSON RE-IDENTIFICATION

into temporal and spatial parts for mining relations from
videos. Tran et al. [27] build a homogeneous architecture with
3D convolution kernels that models appearance and motion
simultaneously.
Graph-based models have been applied to person Re-ID.
Yao et al. [28] design the graph network to represent the
semantic information of local regions. In [29], GPNet leverages a graph model to mine multi-granularity knowledge.
In order to enhance information propagation across features,
Wu et al. [30] propose a graph-based representation learning strategy to establish associations between node features.
Shi et al. [31] construct the graph-based network for learning
higher-order information about pedestrian attributes.
Some works [12], [13], [32] use attention mechanism to
exploit spatiotemporal information. Hou et al. [13] utilize
diverse attention models to learn fine-grained visual information and long-range dependencies. In [12], SINet constructs the
salient-to-broad module to gradually expand the attention area
of adjacent image frames. Wu et al. [32] design the attention
module to enhance the features of target objects. However, all
these methods are sensitive to appearance misalignments that
generally exists across video frames.
B. Part-Based Person Re-ID Models
Part-level features provide fine granularity for describing the images. Part-based approaches learn part aggregated
feature representations, making it robust against misalignment [1]. According to the method of obtaining body part
locations, previous studies can be roughly summarized into
three categories:
1) Rigid Stripe-Based Methods: Some work directly partitions each bounding box into stiff vertical or horizontal streaks
for local-level representation learning. Wang et al. [33] adopt
a multi-branch deep network architecture, which has branches
for obtaining local feature representations from horizontal
stripes with different granularities. To capture discriminative
information at different spatial scales, Zheng et al. [34]
introduce horizontal stripes of various sizes that can integrate progressive cues. Yan et al. [6] design a partitioning
method that separates the the entire feature map into seven
specific parts using the symmetrical information belonging to
the human body structure and the degree of discriminating
knowledge offered by various areas. However, the stripe-based
partition is helpless against background noise and semantic
misalignment.
2) Auto-Localization Based Methods: Some research utilizes regional selection sub-networks in order to automatically
locate the body parts. Li et al. [35] leverage spatial transformer networks with novel spatial constraints for learning and
localizing deformable object parts. Yao et al. [36] employ a
local loss network that automatically generates a bounding box
for each part. Shi et al. [37] design a reinforcement learning
scheme to determine the number and sizes of horizontal
streaks. However, these methods either generate a constant
number of body parts or cannot handle occluded regions.
Besides, the positioning of body parts is still rough and has a
lot of overlap.
3) Extra Semantic-Based Methods: Many approaches
exploit additional semantics in terms of posture to identify object parts, which provides aligned part features.

5309

TABLE I
S UMMARY OF S YMBOLS AND D EFINITIONS

Zhang et al. [38] adopt an off-the-shelf DensePose model
to deform the original image input into 24 semantically
densely aligned part images in the UV space. To aggregate
pixel-wise features weighted by a series of confidence maps,
Guo et al. [39] use the state-of-the-art body parsing framework CE2P for predicting the semantic human part masks.
In [40], a pre-trained pose estimator is introduced to locate
human landmarks and guide robust representation fabrication.
Miao et al. [41] employ a trained human posture estimation
method to supply visible key-point landmarks, which indicate
informative and reliable areas for both gallery and probe
images. There are two key drawbacks of these approaches:
First, they are not capable of identifying individual items
such as backpacks and handbags that are critical for person
Re-ID. Second, the off-the-shelf models are vulnerable to various Re-ID scenarios and cannot correct semantic estimation
mistakes during training.
III. P ROPOSED M ETHOD
We propose a novel CSSA framework for video-based
person Re-ID. It learns pixel-level part-aligned representations
and semantic-aligned global-level representations in an endto-end manner. In this part, we first provide a description of
the proposed CSSA framework. Next, the specifics of two
core processes, i.e., CARE and SGFA, are clarified. Finally,
we present the overall objective function of our framework.
A. The Framework
The overview of the proposed CSSA framework is illustrated in Fig. 2. Table I summarizes some key symbol
q q
q
definitions. Given a video segment I q = {I0 , I1 , . . . , IT −1 }
comprising T sampled frames, we utilize a backbone model to
q
q
q
get the intermediate feature maps X q = {X 0 , X 1 , . . . , X T −1 }.
q
Here X t ∈ RC×H ×W , where W, H, C, t are the width, height,
channel size, and index of the video frame, respectively.
Then, X q is fed into the CARE module for obtaining the
q
q
q
feature maps F q = {F0 , F1 , . . . , FT −1 }. CARE can efficiently highlight macro-visual patterns associated with the
target pedestrian and suppress noise caused by factors like
background clutters and occlusions.
We represent the homogeneous semantic part of a video clip
with all pixel representations belonging to that part. To achieve
pixel-level semantic estimation, a part classifier is designed
to localize distinct parts from each video frame according
to F q . Suppose there are one background category and N p
semantic categories in total. The classifier needs to generate
N p + 1 confidence maps of separate categories for each video
snippet. Then, the representation for the n-th semantic part can

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:23 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Content and Salient Semantics Collaboration
for Cloth-Changing Person Re-Identification
Qizao Wang 1 , Xuelin Qian 2 ,† , Bin Li 1 , Lifeng Chen 1 , Yanwei Fu 3 , Xiangyang Xue 1 ,†
1 School of Computer Science, Fudan University, Shanghai, China
2 School of Automation, Northwestern Polytechnical University, Xi’an, China
3 School of Data Science, Fudan University, Shanghai, China

arXiv:2405.16597v2 [cs.CV] 13 Mar 2025

qzwang22@m.fudan.edu.cn, xlqian@nwpu.edu.cn, {libin, chenlf, yanweifu, xyxue}@fudan.edu.cn

Abstract—Cloth-changing person re-identification aims at recognizing
the same person with clothing changes across non-overlapping cameras.
Advanced methods either resort to identity-related auxiliary modalities
(e.g., sketches, silhouettes, and keypoints) or clothing labels to mitigate
the impact of clothes. However, relying on unpractical and inflexible
auxiliary modalities or annotations limits their real-world applicability.
In this paper, we promote cloth-changing person re-identification by
leveraging abundant semantics present within pedestrian images, without
the need for any auxiliaries. Specifically, we first propose a unified
Semantics Mining and Refinement (SMR) module to extract robust
identity-related content and salient semantics, mitigating interference
from clothing appearances effectively. We further propose the Content
and Salient Semantics Collaboration (CSSC) framework to collaborate
and leverage various semantics, facilitating cross-parallel semantic interaction and refinement. Our proposed method achieves state-of-theart performance on three cloth-changing benchmarks, demonstrating
its superiority over advanced competitors. The code is available at
https://github.com/QizaoWang/CSSC-CCReID.
Index Terms—Person Re-Identification, Clothing Changes, Semantics
Collaboration

I. I NTRODUCTION
Person Re-IDentification (Re-ID) aims to recognize individuals
across different cameras and times. With the growing demand for
surveillance applications and the resurgence of deep learning, significant efforts have been devoted to the advancement of person ReID [1]–[6]. Existing person Re-ID models are principally studied
in the short-term scenario [7], where the clothing of the same
person remains consistent. Consequently, the learned features heavily
rely on clothing appearances, rendering the models ineffective when
individuals change their clothes or wear similar clothes as others. As
a result, there is a rising interest in addressing the cloth-changing
challenge in long-term real-world scenarios [8]–[11].
In this paper, we address the cloth-changing challenge without
relying on any auxiliary module [8]–[10], [12], [13] or extra annotation [14]–[16]. Interestingly, we observe that two commonly
used pooling strategies, namely average and max poolings, exhibit
distinct characteristics in information integration. As depicted in
Fig. 1, average pooling effectively preserves image content information while smoothing out some details, but max pooling excels at
capturing salient information within the content. Motivated by these
findings, we expect to leverage them to enhance the model’s ability
in content understanding and salient information capture, which are
both crucial for effective cloth-changing person Re-ID. On the one
hand, the person Re-ID model is required to comprehend the content
This work was supported in part by the National Natural Science Foundation
of China under Grant 62176061 and Grant 62406252, in part by Shanghai
Technology Development and Entrepreneurship Platform for Neuromorphic
and AI SoC, and in part by the Shanghai Research and Innovation Functional
Program under Grant 17DZ2260900.
† Co-corresponding authors.

Original
images

Average
pooling

Max
pooling

Fig. 1. Comparison of applying different poolings on pedestrian images.
Average pooling can better preserve image content while smoothing details,
while max pooling is better at capturing salient information in the content.
Incorporating both of them is expected to learn abundant semantics, and thus
improve the discriminative ability of Re-ID models.

of pedestrian images and perceive clothing information to extract
identity characteristics adaptively to tackle cloth-changing cases. On
the other hand, salient identity cues play a vital role in mitigating confusion from varying clothes and stabilizing learning features robust to
clothing variations. However, as shown in Fig. 1, employing pooling
in the pixel space of original images is susceptible to interference
from some identity-irrelevant information, such as background and
illumination, which are unrelated to pedestrian identity.
To this end, we exploit both pooling strategies to facilitate feature
learning in the high-level semantic space with the guidance of identity. Specifically, we propose the Semantics Mining and Refinement
(SMR) module. It guides the model to perceive and extract identityrelated semantics, including pedestrian body shape, faces, etc. Then,
the learned semantic features facilitate model learning adaptively. To
avoid interference from identity-irrelevant information and improve
the robustness of the Re-ID model in the cloth-changing scenario, we
propose to mine local semantics and guide the mining process with
carefully designed identity supervision. By integrating the SMR module into the backbone seamlessly, we present the Content and Salient
Semantics Collaboration (CSSC) framework capable of sequential
and parallel semantic interaction and refinement. Both content and
salient semantics play pivotal roles in recognizing and distinguishing
pedestrians with interference from varying clothing appearances.
Contributions. We summarize key contributions as follows.
(1) We propose the novel Content and Salient Semantics Collaboration (CSSC) framework incorporating effective cross-parallel semantic interaction. It effectively addresses the cloth-changing challenge in

SMR-S

SMR-C

" 𝒄𝒔
"
𝑭

"𝒄
𝑭
𝑭

SMR-S

Conv
Block

𝓛𝒄𝒔𝒔𝒄
𝑭𝒄𝒔𝒔𝒄

SMR-C

Backbone
"
" 𝒔𝒄
𝑭

"𝒔
𝑭

Element-wise
Addition

Semantics Mining and Refinement (SMR) Module

Element-wise
Multiplication

Conv
Block

Sigmoid
Function

Horizontal
Division

C

𝓛𝒊𝒅

FC
Semantics Mining

𝓛𝒕𝒓𝒊

ReLU

C

FC

Semantics Refinement

Concatenation
Operation
Average or Max
Pooling

Fig. 2. Framework of our method. The Semantics Mining and Refinement (SMR) module learns identity-related semantics without relying on any auxiliaries.
SMR modules equipped with average pooling (SMR-C) and max pooling (SMR-S) learn content and salient semantics effectively, respectively. Taking advantage
of SMR, our framework interacts and refines both semantics sequentially and parallelly to promote cloth-changing person Re-ID.

person Re-ID without relying on unpractical and inflexible auxiliary
modalities or extra annotations.
(2) We propose the Semantics Mining and Refinement (SMR)
module, which applies the distinct pooling characteristics to learn
and utilize robust identity-related content and salient semantics in a
unified manner.
(3) Extensive experiments on three cloth-changing person ReID benchmarks demonstrate the effectiveness and superiority of our
proposed method.
II. M ETHODOLOGY
A. Problem Formulation
Given a training dataset D = {xi , yi }N
i=1 containing totally N
images and N p identities, where xi and yi represent the i-th images
and its corresponding identity label, the Re-ID model G aims to
map the person image xi into a discriminative feature representation
Fi = G (xi ). Subsequently, a pooling layer is then applied to
Fi ∈ Rh×w×d , where h and w are the height and width of the feature
representation, yielding the feature vector fi ∈ Rd . The model G is
required to tolerate clothing changes and extract the robust feature fi ,
so that the similarity between features of pedestrians with the same
identity is larger than that of pedestrians with different identities,
regardless of whether they change clothes or not. In the subsequent
discussion, unless otherwise specified, we omit the subscript i of Fi
and fi to simplify notation.
B. Semantics Mining and Refinement Module
To enhance the discriminative ability of the model in the clothchanging scenario, we leverage both identity-related content and
salient semantics with our designed Semantics Mining and Refinement (SMR) module. In the following, we first illustrate the
learning of identity-related content semantics which help comprehend
pedestrian images and extract identity characteristics adaptively to
tackle cloth-changing cases.

Global content semantics mining. With F extracted by the backbone, one convolutional block is appended, resulting in F c ∈
Rh×w×d . Then, we obtain the average pooled feature,
f c = AvgPool(F c ) ∈ Rd ,

(1)

where AvgPool denotes the global average pooling operation, which
effectively integrates content semantics in feature maps.
Local content semantics mining. In addition to mining global
content semantics, the semantics of local regions are also crucial.
Therefore, we horizontally divide F c into different parts and use one
convolutional layer for dimensionality reduction, resulting in several
c P
local content semantic features {f lp }p=1 . However, in the clothchanging scenario, the reliability of local regions is questionable.
Due to clothing changes and clothing similarity between pedestrians,
if carried out as previous works [3], [17], local semantics can mislead
model learning. To leverage local content semantics effectively, we
concatenate the global and local content semantic features. Formally,
c
c
c
fˆc = [f c ; f l1 ; f l2 ; · · · ; f lP ],

(2)

where [·; ·] denotes concatenation in the feature dimension. To ensure
the mined content semantics are identity-related and discriminative,
widely-used Re-ID losses are used as follows:
Lsmr c (F c ) = Lid (fˆc ) + Ltri (f c ),

(3)

where Lid and Ltri denote widely used identity classification
loss [18] and triplet loss [19], respectively. Note that Lid includes
an identity classifier implemented as one fully connected layer.
Additonally, the identity label of fˆc (i.e., y) is required in Lid and
Ltri for identity supervision, and we omit it for notation simplicity.
Content semantics refinement. To improve the discriminative ability
of the model, the mined identity-related global content semantics are
also used to refine F c in an adaptive manner. F c is recalibrated along
the channel dimension using the learned semantics in f c . Formally,
Fec = σ(W2 ϕ(W1 f c )) ⊗ F c ,

(4)

where ϕ denotes the ReLU activation function, σ denotes the Sigmoid
d
function, and ⊗ denotes element-wise multiplication. W1 ∈ R r ×d

d

and W2 ∈ Rd× r are the weights of two Fully-Connected (FC)
layers, where the first one compresses the channel dimension to
achieve information bottleneck. r is set to 16 following [20]. The final
Sigmoid function acts as a self-attention mechanism across channels,
generating scores to refine the content semantics of pedestrian features adaptively. The ∼ notation represents our proposed semantics
mining and refinement operation applied to F .
Salient semantics mining and refinement. Salient identity cues
are helpful to mitigate confusion from varying clothes and stabilize
learning features robust to clothing variation. Following a similar
design for learning identity-related content semantics, identity-related
salient semantics can be mined with our SMR module by simply
replacing average pooling with max pooling. We denote the extracted
salient semantic feature as F s , and its max pooled feature f s and
locally augmented feature fˆs are supervised by the loss Lsmr s (F s )
(similar to Eq. 3) for identity-related salient semantics mining.
Finally, refined salient semantic feature Fes can be obtained.
C. Semantics Collaboration Framework

TABLE I
C OMPARISON OF OUR METHOD WITH STATE - OF - THE - ART METHODS ON PRCC
AND LTCC. M ETHODS IN THE GRAY REGION USE EXTRA GROUND - TRUTH
CLOTHING LABELS FOR TRAINING . “ SKETCH ”, “ SIL .”, AND “ POSE ” REPRESENT
CONTOUR SKETCHES , SILHOUETTES , AND HUMAN POSES , RESPECTIVELY. M ETHODS
MARKED WITH “∗” INVOLVE MULTIPLE TRAINING STAGES FOR EXTRA AUXILIARY
NETWORKS . “S TANDARD ” AND “C LOTH -C HANGING ” MEAN THE STANDARD AND
CLOTH - CHANGING SETTINGS , RESPECTIVELY. T HE BEST RESULTS ARE SHOWN IN
BOLD .
PRCC
Methods

Modality

Cloth-Changing

LTCC
Standard

Cloth-Changing

Standard

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

UCAD [25]
CAL [14]
DCR-ReID [24]
AIM [15]
CCFA∗ [16]

RGB+sil.
RGB
RGB+sil.+sketch
RGB
RGB

45.3
55.2
57.2
57.9
61.2

55.8
57.4
58.3
58.4

96.5
100
100
100
99.6

99.8
99.7
99.9
98.7

32.5
40.1
41.1
40.6
45.3

15.1
18.0
20.4
19.1
22.1

74.4
74.2
76.1
76.3
75.8

34.8
40.8
42.3
41.1
42.5

HA-CNN [29]
PCB [3]
IANet [2]
TransReID [30]
RCSANet∗ [31]
ACID [32]
FSAM [12]
GI-ReID∗ [13]
CAMC [10]

RGB
RGB
RGB
RGB
RGB
RGB
RGB+pose+sil.
RGB+sil.
RGB+pose

21.8
41.8
46.3
46.6
50.2
55.4
54.5
33.3
-

38.7
45.9
44.8
48.6
66.1
-

82.5
99.8
99.4
100
100
99.1
98.8
80.0
-

97.0
98.3
99.0
97.2
99.0
-

21.6
23.5
25.0
34.4
29.1
38.5
23.7
36.0

9.3
10.0
12.6
17.1
14.5
16.2
10.4
15.4

60.2
65.1
63.7
70.4
65.1
73.2
63.2
73.2

26.7
30.6
31.0
37.0
30.6
35.4
29.4
35.3

CSSC (Ours)

RGB

65.5

63.0

100

99.1

43.6

18.6

78.1

40.2

The SMR modules with average pooling and max pooling, denoted
as SMR-C and SMR-S, are used for content and salient semantics
mining and refinement, respectively. As illustrated in Fig. 2, SMR-C
and SMR-S are alternatively employed in different orders within two
branches. On the one hand, the refined content semantics Fec facilitate
e
mining salient semantics and result in Fecs . The losses involved in
the first branch are formulated as follows:

can lead to the overfitting problem and contribute to suboptimal
performance. Therefore, we empirically find it better to introduce
Ltri for model optimization after the first learning rate decay. During
inference, the comprehensive semantics feature vector f cssc is used
to compute the cosine distance between person images for retrieval.

Lbranch1 = Lsmr c (F c ) + Lsmr s (Fecs ).

III. E XPERIMENTS

(5)

On the other hand, the refined salient semantics Fes aid in mining
e
content semantics and result in Fesc . The losses involved in the second
branch are formulated as follows:
Lbranch2 = Lsmr s (F s ) + Lsmr c (Fesc ).

(6)

The cross-parallel interaction framework takes full advantage of
mined content and salient semantics, promoting the robustness of
Re-ID models in the cloth-changing scenario.
Comprehensive semantics learning. The two features from the two
branches specialized in content and salient semantics are fused to a
comprehensive semantics feature as follows:
e
e
F cssc = Conv(Fecs ⊕ Fesc ),

(7)

where ⊕ denotes element-wise addition, and Conv denotes a convolutional block. We take the max-pooled F cssc as the final Re-ID
identity representation, which is supervised by commonly used Re-ID
losses. Formally,
f cssc = MaxPool(F cssc ) ∈ Rd ,

(8)

cssc

(9)

Lcssc = Lid (f

) + Ltri (f

cssc

),

where MaxPool denotes the global max pooling operation.

Datasets. To demonstrate the effectiveness of our method, we evaluate it on three widely-used cloth-changing person Re-ID datasets,
i.e., PRCC [9], LTCC [8], and Celeb-reID [21].
Implementation details. Following previous works, we adopt
ResNet-50 [22] pre-trained on ImageNet [23] and apply our SMR
modules with P = 8 in the conv5 layer by reusing the three readymade convolutional blocks. Following [8], [14], [15], [24], [25], the
input images are resized to 384 × 192. The batch size is set to 32.
Random horizontal flipping, padding, random cropping, and random
erasing [26] are used for data augmentation. Adam optimizer [27]
with weight decay of 5×10−4 is adopted for 120 epochs. The learning
rate linearly increases from 3 × 10−5 to 3 × 10−4 in the first 10
epochs and decreases by a factor of 10 at the 30th and 60th epochs.
All experiments are conducted on one NVIDIA GeForce GTX 1080
Ti with 11GB of memory.
Evaluation metrics. For evaluation, we adopt standard metrics
as in most person Re-ID literature, namely Cumulative Matching
Characteristic (CMC) curves and mean Average Precision (mAP). For
LTCC and PRCC, we evaluate our method under both the standard
setting and the cloth-changing setting following [28].
B. Comparison with State-of-the-Art Methods

D. Training and Inference
Both content and salient semantics are mined and refined with
the help of our proposed SMR module, and effectively collaborate to
promote person Re-ID in our proposed CSSC framework. The overall
loss is computed as follows:
L = Lbranch1 + Lbranch2 + Lcssc .

A. Experimental Settings

(10)

In the cloth-changing scenario, the appearance of pedestrians
can vary significantly due to different clothing. Introducing Ltri
prematurely when the Re-ID model is not robust to clothing variations

Results on PRCC and LTCC. In Tab. I, we compare with advanced
methods designed for conventional person Re-ID [2], [3], [29], [30]
and methods tailored for cloth-changing person Re-ID using auxiliary
modalities [10], [12], [13] and using DG-Net [32], [35]. CSSC
with only RGB modality shows significant superiority over them,
without relying on extra annotations or auxiliary modalities. Recently,
some works have proposed to utilize ground-truth clothing labels to
mitigate the impact of clothes [14]–[16], and some also use auxiliary
modalities [24], [25], [36]. CSSC achieves state-of-the-art results on
PRCC and exhibits competitive performance with them on LTCC. It


exec
/bin/zsh -lc "pdftotext -l 3 'Confidence guided semi-supervised cross-modality person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 165 (2025) 111669

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Confidence guided semi-supervised cross-modality person re-identification
Xiaoke Zhu a , Lingyun Dong a , Xiaopan Chen a
Xiao-Yuan Jing d

,∗, Xinyu Zhang b , Fumin Qi c ,

a School of Computer and Information Engineering, Henan University, Kaifeng, China
b

Hunan University of Technology and Business, Changsha, China
National Supercomputing Center in Shenzhen, Shenzhen, China
d
School of Computer Science, Wuhan University, Wuhan, China
c

ARTICLE

INFO

Keywords:
Semi-supervised learning
Cross modality
Person re-identification
Confidence learning

ABSTRACT
Existing visible-infrared person re-identification (VI-ReID) methods mainly focus on supervised learning and
unsupervised learning. As a compromise between supervised learning and unsupervised learning, semisupervised learning is of great significance in practical applications. However, there is limited literature
addressing VI-ReID under the setting of semi-supervised learning. In this paper, we propose a Confidence
Guided Semi-supervised Feature Learning (CGSFL) method for cross-modality person re-identification, which
can effectively exploit the information contained in unlabeled data with the help of a small amount of labeled
data. Specifically, the CGSFL framework consists of two key modules, including confidence guided crossmodality pseudo label generation module and semi-supervised cross-modality discriminant feature learning
module. The former calculates the confidence of cross-modality matching and clustering, using them to enhance
the reliability of generated cross-modality pseudo labels. The latter designs dynamic and static memory
banks as well as a nearest neighbor class-based sampler to effectively exploit discriminative information
present in labeled and pseudo-labeled data, thereby ensuring the discriminative capability of the learned
model. Extensive experiments show that our proposed CGSFL method achieves very encouraging performance
compared to the state-of-the-art supervised and unsupervised methods, providing an effective alternative
solution for the deployment of VI-ReID to real-world environment. The code of our paper is available at
https://github.com/Dongly502/SSL-CGSFL.

1. Introduction
With the development of public video surveillance systems, modern
devices increasingly operate in dual modes: capturing visible-light images during daytime and automatically switching to the infrared mode
at night. This technological evolution has propelled visible-infrared
person re-identification (VI-ReID) as a critical research frontier, aiming
to match pedestrian images across heterogeneous modalities. Specifically, VI-ReID seeks to retrieve the same identity from infrared/visible
galleries given a visible/infrared query image. Compared with conventional single-modality ReID, this cross-modality matching faces greater
challenges due to significant modality discrepancies in image characteristics. Moreover, the need of large amount of labeled cross-modality
data further compounds these difficulties, as manual annotation of
paired visible-infrared identities remains labor-intensive and costly.
In recent years, numerous methods have been proposed for the VIReID task. These methods are broadly divided into supervised and unsupervised VI-ReID, based on the utilization of labeled cross-modality

training data. Supervised VI-ReID aims to improve the performance
of cross-modality matching by learning discriminative features from
extensive labeled cross-modality datasets. For example, FMCNet [1]
alleviates inter-modal divergence by enhancing feature discrimination
through feature-level modality compensation. In [2], the modalityspecific memory network is proposed to overcome the loss of discriminative modality-specific information, further reducing the modality
discrepancy. In [3], Zhang et al. proposed a dual-semantic consistency learning network, which can alleviate the modality difference
by reducing the channel consistency. In [4], Chen et al. proposed a
consistency-driven feature scoring and regularization network (CFSRNet) for VI-ReID to enhance the training stability. While effective, these
methods require extensive cross-modality annotations. Compared with
the single-modality person ReID, it is more expensive and difficult to
annotate such a large number of cross-modality samples for VI-ReID.
This largely limits the scalability of supervised methods in real-world
scenarios.

∗ Corresponding author.

E-mail address: xpchen@henu.edu.cn (X. Chen).
https://doi.org/10.1016/j.patcog.2025.111669
Received 14 January 2024; Received in revised form 31 March 2025; Accepted 31 March 2025
Available online 8 April 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 165 (2025) 111669

X. Zhu et al.

2. Related work

Different from supervised VI-ReID methods, unsupervised VI-ReID
trains the model by mining the potential information of unlabeled
samples. For example, Liang et al. [5] proposed a two-stage learning
approach, termed H2H, including unsupervised intra-modality selflearning stage and intra-modality heterogeneous learning stage. However, H2H is complex and must use the extra Market-1501 dataset as
labeled RGB samples for pre-training, which reduces the scalability of
this method in actual deployment. To alleviate the difficulties in generating reliable cross-modality labels and learn robust modality-invariant
features without any annotations, an Augmented Dual-Contrastive Aggregation (ADCA) learning framework is proposed in [6], which utilizes an enhanced double contrast aggregation module to capture the
complementary information on visible-infrared pedestrian features.
The aforementioned unsupervised methods can partially mitigate
the issue of data labeling costs. However, the performance of these
unsupervised methods still shows a significant gap compared to that of
supervised VI-ReID methods. For example, on SYSU-MM01 dataset, the
Rank-1 accuracy of the top-performing supervised method is 80.83%
[7], while the performance of the most advanced unsupervised method
is only 45.51% [6]. The significant performance gap may be caused by
the inherent large inter-modality discrepancy in the unlabeled crossmodality data itself. Furthermore, the process of generating crossmodality data with pseudo-labels usually lacks the guidance of real
labels, resulting in the limited reliability of the generated pseudo-labels,
which affects the performance of the learned model.
As a compromise between unsupervised and supervised learning
paradigms, semi-supervised learning strategy not only reduces dependence on labeled data but also effectively leverages limited annotated
samples to guide the exploitation of unlabeled data, thereby improving
model reliability. In recent years, this methodology has demonstrated
remarkable success in single-modality person re-identification tasks.
Huang et al. [8] developed the Multi-pseudo Regularized Label (MpRL)
mechanism to enhance pseudo-label quality by assigning appropriate virtual labels to generated data samples. Furthermore, in order
to reduce minimize the annotation cost, Hu et al. [9] proposed a
Multi-grained Active Semi-Supervised learning framework (MASS) incorporating a cluster-scatter procedure and purification loss for optimized label efficiency. Chen et al. [10] proposed a semi-supervised
ReID framework, which leverages reliable sample selection to improve
unlabeled data utilization.
However, the aforementioned methods assume that all images originate from visible modality, and thus cannot be directly employed
to effectively tackle the problem of VI-ReID due to the significant
divergence between the visible and infrared modalities. To the best of
our knowledge, limited research has explored the application of semisupervised learning methodologies for the task of cross-modality person
re-identification.
Based on the above analyses, this paper aims to minimize the
dependence on labeled data and alleviate the performance degradation
caused by the absence of labeled samples. To this end, we incorporate
semi-supervised learning techniques into the VI-ReID task and propose
a semi-supervised cross-modality person re-identification approach. By
leveraging the generated pseudo-labeled data, our method enables
the model to learn modality-irrelevant features by jointly optimizing
cluster-level pseudo-labeled features with true labeled features.
The main contributions of our approach are as follows.
(1) We design a Confidence guided Cross-modality Pseudo label
Generation (CCPG) module, which leverages the supervised information to compute the cross-modality matching confidence and the
clustering confidence of the current model, and then uses the obtained
confidence values to generate more reliable pseudo labels for the next
iteration.
(2) We design a Semi-supervised Cross-modality Discriminant
F eature Learning (SCDFL) module, which can improve the discriminability of features gradually by making use of nearest-neighbor image
pairs sampled from both the labeled and pseudo-labeled data.
(3) We conduct extensive experiments on the SYSU-MM01 and
RegDB datasets. The results demonstrate the effectiveness of our
proposed method.

2.1. Supervised cross-modality person ReID
Recent years have witnessed significant progress in Visible-Infrared
ReID (VI-ReID), with various approaches proposed to address modality
discrepancy. Wu et al. [11] established the SYSU-MM01 benchmark dataset, and proposed a zero-padding strategy for learning
modality-shared features in deep networks. Subsequent efforts explored
constrained feature learning: Hao et al. [12] designed dual-stream
networks with hypersphere manifold embedding to reduce intra- and
inter-modality divergence, while Ye et al. [13] employed bi-directional
dual-constrained top-ranking loss for modality alignment. Adversarial learning emerged as another promising direction in VI-ReID. The
cmGAN framework [14] leveraged generative adversarial networks
to minimize cross-modality divergence, while Hi-CMD [15] further
disentangled identity-related and modality-specific features through
adversarial disentanglement. In addition, Wu et al. [16] proposed joint
modality-pattern alignment to uncover subtle cross-modality differences, and Ling et al. [17] developed a cross-modality average precision
method to optimize the cross-modality overall ranking order.
Despite these advances, existing supervised methods heavily depend on extensive cross-modality identity labels, significantly limiting
their adaptability to new scenarios. Our work addresses this critical
limitation by proposing a semi-supervised framework that achieves
competitive performance with limited labeled data, enabling practical
deployment in real-world environments.

2.2. Unsupervised visible-infrared person ReID
Unsupervised Visible-Infrared Person ReID (USVI-ReID) is proposed
to cope with the cases characterized by a scarcity of annotated data
labels. The USVI-ReID task was first addressed in [5] by introducing
a two-stage method called H2H, which involves homogeneous and
heterogeneous learning techniques. Wang et al. [18] accomplished
the process of visible-infrared matching by minimizing the distance
between the feature sets of visible images and infrared images. The approach in [6] overcomes the defect of pre-training in H2H and proposes
an enhanced contrastive aggregation learning method for USVI-ReID.
In [19], Wu et al. proposed a method based on step-wise graph matching and alternating learning, achieving relatively good performance. Ji
et al. [20] designed an adversarial contrastive feature learning framework for USVI-ReID, aiming to enhance the discriminative capability
of features. Yang et al. [21] proposed a transformer-based shallowdeep collaborative learning framework, which integrates shallow patch
embeddings with deep features to address modality gaps in VI-ReID.
In [22], Yu et al. introduced a controllable diffusion framework that
generates diverse, identity-aware infrared person images to enhance
VI-ReID without extra annotations.
Although the above unsupervised methods have achieved very encouraging performance by only using unlabeled data, their performance
still shows a large gap compared with the supervised learning methods.
The major reason can be attributed to the lack of guidance from labeled
data, which hinders the reliability of the trained model.

2.3. Semi-supervised person ReID
This section reviews semi-supervised person re-identification,
covering both single-modality and cross-modality approaches.
2

Pattern Recognition 165 (2025) 111669

X. Zhu et al.

of generated noisy pseudo-labels. Different from DPIS, our approach
leverages the labeled data to enhance the reliability of generated crossmodality cluster pairs, which are then used to reduce the differences
between the infrared and visible modalities.
In summary, both our approach and DPIS aim to alleviate the
dependency on labeled data in cross-modality person re-identification
through the use of semi-supervised techniques. However, there are
significant differences in the solutions and settings employed.

2.3.1. Semi-supervised single-modality person re-identification
In recent years, many semi-supervised person re-identification
methods have been proposed. Some representative works are as follows. [23] first proposed a semi-supervised coupled dictionary learning
method that integrates labeled and unlabeled data to acquire shared
discriminant dictionaries and re-configurable sparse encoders via dictionary decomposition techniques. Han et al. [24] proposed using
CycleGAN to generate unlabeled samples from labeled training data,
and innovatively learned the pseudo relationship between labeled and
unlabeled training data. Furthermore, [25] designed an end-to-end
multiple branch network, which optimizes the model through identity
classification. Li et al. [26] proposed a semi-supervised region metric
learning method that employs label propagation and cross-person distribution alignment to handle imbalanced unlabeled data and identify
positive neighbors. Wu et al. [27] proposed a joint training method
and a progressive sampling strategy which can optimize the model
through labeled data (or pseudo-labeled) and unlabeled data. The
above semi-supervised methods leverage a small amount of labeled data
and a large amount of unlabeled data for training, and have achieved
interesting results on single-modality person ReID task. However, these
semi-supervised methods cannot be directly employed to solve the
cross-modality person ReID due to the large discrepancy between different modalities. Our approach differs from existing single-modality
methods in the following two folds.
(1) Existing single-modality methods usually do not take the modality differences into consideration during the feature extraction phase. In
contrast, our feature extraction model employs a dual-stream network,
which can reduce the influence of cross-modality discrepancy on the
feature extraction effectively.
(2) Regarding the utilization of unlabeled information, existing
methods generally generate pseudo-labels through clustering algorithms without considering the impact of modality differences on
clustering results. In this paper, we design a Confidence Guided Crossmodality Pseudo Label Generation module, which can generate the
cross-modality matching confidence and the clustering confidence.
The generated confidences are used to enhance the reliability of
cross-modality clustering pairs.

3. Proposed method
3.1. Problem formulation and overview
In this paper, we propose a confidence-aware pseudo labels
optimization-based semi-supervised cross-modality feature learning approach for VI-ReID. Our approach, as illustrated in Fig. 1, mainly
consists of two modules: confidence guided cross-modality pseudo label
generation module and semi-supervised cross-modality discriminant
feature learning module. Details of each module can be found in Sections 3.2 and 3.3. In both modules, the feature extraction blocks share
the same structure, consisting of modality-specific shallow layers and
modality-shared layers. The modality-specific layers for each modality
can capture modality-specific information. The modality-shared layers
can alleviate the divergence between the two heterogeneous modalities
by learning a multi-modality shareable space. The training process is
divided into the following two stages.
Stage 1. Training a feature extraction model with strong representation ability. Specifically, the labeled and unlabeled images are
first sent to the feature extraction module and the features of each
sample are obtained. Based on the obtained features, the unlabeled
data is clustered using the DBSCAN algorithm. Then, the cluster-center
features for each cluster are computed and stored in the static memory
bank. Finally, the loss functions 𝑠𝑖𝑑 (on labeled images) and 𝑠𝑡𝑎 (on
unlabeled images) are employed to optimize the feature extraction
model.
Stage 2. Further optimizing the feature extraction model by
introducing the confidence-guided cross-modality pseudo label
generation module and the dynamic memory bank. Specifically,
the labeled visible and infrared features obtained from feature extraction module are fed into the adaptive confidence generation block,
to compute the cross-modality matching confidence and the clustering
confidence. Then, the unlabeled images are clustered, and the pseudolabeled infrared and visible cluster pairs can be generated according to
the cross-modality matching confidence and the clustering confidence.
Next, the generated pseudo-labeled cross-modality cluster pairs are
used to update the dynamic memory bank. In this way, the differences
between cross-modality clusters of the same pedestrian can be reduced
gradually. Finally, the static memory bank and the updated dynamic
memory bank are used for semi-supervised cross-modality discriminant
feature learning. In this stage, the loss functions 𝑠𝑖𝑑 , 𝑠𝑡𝑎 and 𝑑𝑦𝑛 are
employed to optimize the feature extraction model. The flow of our
proposed approach is depicted in Algorithm 1.

2.3.2. Semi-supervised cross-modality person re-identification
Research on semi-supervised learning in cross-modality person reidentification remains in its early stages. To our knowledge, there is
only one relevant article, i.e., the literature [28], which introduces
Dual Pseudo-label Interactive Self-training (DPIS) for this task. DPIS
addresses both reliable pseudo-label generation and noisy pseudo-label
calibration. Our method differs from DPIS in three key aspects.
(1) The semi-supervised settings in our paper differ from those
in DPIS. In DPIS, labels of different rates for each identity on both
modalities are utilized in the bi-semi-supervised setting. However, our
approach selects different rates of pedestrians, and leverages the labels
of images of these pedestrians in the semi-supervised setting. Taking
the RegDB dataset as an example, the original training set consists of
206 identities, with each identity having ten visible images and ten
infrared images. Under the setting of 10% labeled data, DPIS utilizes
the labels of 10% images for each identity (i.e., one visible image
and one infrared image for each individual). In contrast, our approach
utilizes the labeled images of about twenty-one pedestrian individuals
(206×10% ≈ 21), leaving the remaining individuals’ images as unlabeled
data.
(2) The assumptions on the relationship between unlabeled and labeled images are different. Our method does not impose any restrictions
on the label space between unlabeled and labeled images, whereas DPIS
assumes that both unlabeled and labeled images have the same label
space (i.e., the labeled and unlabeled images are from the same group
of pedestrians).
(3) The goal of utilizing the small amount of labeled data is
different. DPIS utilizes labeled data to mitigate the negative impact

3.2. Confidence guided cross-modality pseudo label generation
To facilitate the description of our
approach, we first}introduced
{
the notation used in this paper. Let 𝑋𝑟 = 𝑥𝑟1 , 𝑥𝑟2 , … , 𝑥𝑟𝑀
and 𝑋𝑣 =
1
{
}
𝑣
𝑣
𝑣
𝑥1 , 𝑥2 , … , 𝑥𝑀 separately denote the features of labeled infrared and
2
{
}
{
}
visible images. Let 𝑈𝑟 = 𝑢𝑟1 , 𝑢𝑟2 , … , 𝑢𝑟𝑁
and 𝑈𝑣 = 𝑢𝑣1 , 𝑢𝑣2 , … , 𝑢𝑣𝑁
1

2

represent the features of unlabeled infrared and visible images, respectively. Here, 𝑀1 , 𝑀2 , 𝑁1 and 𝑁2 are the numbers of items in the
corresponding sets, respectively. Let 𝐶𝑢𝑟 and 𝐶𝑢𝑣 (𝐶𝑙𝑟 and 𝐶𝑙𝑣 ) represent
the clustering results of unlabeled (labeled) infrared and visible images,
𝑟 and 𝐶 𝑣 (𝐶 𝑟 and 𝐶 𝑣 ) be the 𝑖th clusters in 𝐶 𝑟 and
respectively. Let 𝐶𝑢,𝑖
𝑢
𝑢,𝑖
𝑙,𝑖
𝑙,𝑖
𝐶𝑢𝑣 (𝐶𝑙𝑟 and 𝐶𝑙𝑣 ), respectively.
3


exec
/bin/zsh -lc "pdftotext -l 3 'Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112443

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Condense loss: Exploiting vector magnitude during person Re-identiﬁcation
training process
Xi Yang
a

a,∗, Wenjiao Dong a , Yingzhi Tang b , Gu Zheng a , Nannan Wang a , Xinbo Gao

c

State Key Laboratory of Integrated Services Networks, School of Telecommunications Engineering, Xidian University, Xi’an, 710071, China

b Department of Computer Science, City University of Hong Kong, Hong Kong SAR, China
c

State Key Laboratory of Integrated Services Networks, School of Electronic Engineering, Xidian University, Xi’an, 710071, China

a r t i c l e

i n f o

a b s t r a c t

Keywords:
Re-ID
Face recognition
Softmax loss
Tangential gradient

The magnitudes of features and weights signiﬁcantly aﬀect the gradients during the training process. L2 normalized softmax losses (such as NormFace, CosFace, ArcFace, etc.) and Naive softmax losses both reduce the
magnitudes of image features in the training process and achieve good results in face recognition and person
re-identiﬁcation tasks, respectively. In this paper, we fully utilize the feature vector magnitudes and propose
Condense loss for Re-ID tasks, which replaces the inner production of Naive softmax loss with the negative Euclidean distance. Condense loss generates negative radial gradients when updating weight parameters to push all
features compacter. Because the coeﬃcients of tangential gradients (the tangential component of the gradients)
are related to feature magnitudes, it ideally provides monotonically decreasing tangential gradients, resulting
in gradually diminishing updates that enhance the stability of the training process. We also introduce a margin
parameter into Condense loss to enlarge inter-class distances and thus help the model learn more discriminative
features. Mathematical analysis is given in this paper, and we have conducted suﬃcient experiments focusing on
Re-ID tasks to prove the corresponding conclusion. The experimental results demonstrate that the Condense loss
achieves competitive results compared to the state-of-the-art methods in the person re-identiﬁcation task. At the
same time, it also has a good performance in face recognition tasks.

1. Introduction

achieved by ‖𝑥‖2 , which is the modulus of the one-dimensional vector
after the feature ﬂattening of the image). It is highly consistent with our
cognition because in the classiﬁcation task, no matter what kind of data
distribution (such as long-tail distribution, etc.), the modulus length
of the image feature vectors will be compressed during the training
process.
Based on this assumption, we conducted experiments on the Re-ID
task and face recognition to observe the relationship between feature
magnitude and accuracy. Fig. 1(a) and (b) show the variation of the
average magnitude and accuracy of images in the VGG2FP dataset with
respect to the training epochs under our baseline method [4]; (c) and (d)
show the variation of the average magnitude and retrieval accuracy of
images in the Market-1501 dataset with respect to the training epochs
under our baseline method [5]. It can be seen that with the progress
of training, the epoch increases continuously, the magnitudes of images
decrease, and the classiﬁcation accuracy improves. This conﬁrms that
the connotation behind the picture is the feature magnitude. Therefore,
we claim that classiﬁcation accuracy is related to the feature magnitudes
of the model outputs in Re-ID and face recognition.

Face recognition technology has been widely applied in the real
world because facial features are the best indicator of a person. Because
sometimes the cameras cannot capture the front side pedestrians, person Re-Identiﬁcation (Re-ID) [1,2] technology is a signiﬁcant auxiliary
to capture pedestrians through diﬀerent cameras. Although the Re-ID
task, as we know it, is more similar to the image retrieval task, the representative features are extracted from the original image data during
the training process. Then the ID or attribute of the pedestrian is used
as the training label to train the model, which is also transformed into
a ﬁne-grained classiﬁcation task.
For a face picture, the facial features have ﬁxed relative positions,
and we can use tools like MTCNN [3] to easily align the face. However, pictures of persons show various poses and diﬀerent views since
their arms and legs are not ﬁxed. In addition, people walk in diﬀerent
directions at various angles, which also makes pictures of persons differently. What is the connotation underlying these pictures? Our answer
is the feature vector magnitude (for a feature vector 𝑥, its magnitude is

∗ Corresponding author.

E-mail address: yangx@xidian.edu.cn (X. Yang).
https://doi.org/10.1016/j.patcog.2025.112443
Received 6 June 2024; Received in revised form 9 September 2025; Accepted 10 September 2025
Available online 13 September 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 172 (2026) 112443

X. Yang et al.

Fig. 1. The variation trends of feature magnitude and accuracy during the training process. (a) The average feature magnitude of the images during the training
process on the VGG2FP dataset with naive softmax loss. (b) Accuracy change curve during the training process on the VGG2FP dataset with naive softmax loss. (c)
The average feature magnitude of the images during the training process on the Market-1501 dataset with naive softmax loss. (d) Rank-1 change curve during the
training process on the Market-1501 dataset with naive softmax loss.

In the Re-ID task’s dataset, there is a small gap in the number
of pictures of each type of pedestrian, so researchers often use Naive
softmax as the internal calculation method of cross entropy classiﬁcation loss. In the data set of face recognition tasks, there is a particular gap in the number of images of each type of face, and the data
balance ability is relatively poor, so L2 normalization is commonly
used by researchers to resist balance. But in any case, both of these
loss calculation methods are to reduce the length of the characteristic mode. We can therefore claim that in face recognition and ReID tasks, the feature magnitudes should be fully utilized to train the
model.
Furthermore, Fig. 1(d) shows the relationship between classiﬁcation accuracy (rank-1) and epochs in the Market-1501 dataset under our baseline method [5]. With the increase of epochs, the overall accuracy is improved. But we can see that in the ﬁrst 30 epochs,
there is a decline in rank-1. At the same time, when rank-1 decreases, image feature magnitude increases in the reverse direction.
Therefore, we believe that the strong inverse correlation between feature magnitude and classiﬁcation accuracy further validates the importance of the magnitude of feature vectors in image classiﬁcation.
However, we also found that the compression process of feature magnitude experiences unstable phases, which can result in the magnitude not being reduced to an appropriate size at certain stages,
thereby aﬀecting classiﬁcation performance. Therefore, it is particularly important to design a classiﬁer that can stably reduce the
magnitude of feature vectors in face recognition and person Re-ID
tasks.
In the training process of neural networks, the weight parameters are
updated by calculating the gradient of the loss function. Based on our
analysis (details in Section 3.4), the tangential gradients of the Naive
Softmax loss and L2 Softmax loss usually ﬂuctuate. In other words, the
tangential gradient is not monotonically decreasing. This means that the
Naive Softmax loss could sometimes provide larger gradients in later iterations, leading the network to a local minimum. To avoid this prob-

lem, we need a loss function that can provide an ideal monotonically
decreasing tangential gradient.
In this paper, we propose a novel softmax loss formulation named
Condense loss, which takes the advantage of feature and weight vector magnitudes in Re-ID tasks to give a monotonically decreasing coeﬃcient of a tangential gradient. Diﬀerent from the Naive and L2 normalized softmax losses which are in inner product formulation between
normalized or un-normalized feature and weight, it takes the negative
Euclidean distance as the formulation. When updating weight parameters, it provides negative radial gradients, hence, the feature vector magnitudes keep reducing during the training process. Furthermore, Condense loss provides a larger tangential gradient at the beginning of the
training process, and a smaller tangential gradient at the end, with the
change in feature vector magnitudes. After training, both the magnitude
of the feature vectors and the angles between the feature vectors and the
class centers decrease, so we name this loss function “Condense Loss”.
And we also introduce a margin parameter into Condense loss to enlarge
the inter-class distances. The mathematical analysis will be given in
Section 3.
Our contributions are summarized as follows:
•

We propose Condense loss with negative Euclidean distance formulation for Re-ID tasks, which utilizes the vector magnitudes to provide
an adaptive gradient during the training phase.
• We introduce a margin parameter into Condense loss to enlarge the
inter-class distances, which helps the model learn more discriminative features.
• We give the mathematical analysis to show why Condense loss can
work and prove its superiority in Re-ID tasks when compared with
other softmax loss variants.
• We conduct extensive experiments on two Re-ID benchmarks and
face recognition tasks, which demonstrates the eﬀectiveness of Condense loss. When compared with the state-of-the-art methods, Condense loss achieves competitive results.
2

Pattern Recognition 172 (2026) 112443

X. Yang et al.

2. Related work

batches, and it has been widely adopted by many Re-ID methods. Chen
et al. [8] improved triplet loss by introducing a hard negative sample.
Therefore, we devote ourselves to researching classiﬁcation loss.

2.1. Person Re-identiﬁcation

2.2. Modiﬁed softmax losses

Because Re-ID is a complex feature learning task, many works focus
on learning and extracting highly discriminative features from pictures
or videos. The methods based on metric learning such as Contrastive
loss [6], Triplet loss [7] and Quadruplet loss [8] have been developed
quite mature.
Part-level feature learning frameworks are pioneering methods that
signiﬁcantly improve Re-ID performance. Wu et al. [9] introduced a
deep attention-based comparator module to learn relative parts of paired
pictures and produce relevant features to improve the part-based Re-ID
methods. In the meantime, the attention mechanism has been proved
to be beneﬁcial for Re-ID tasks since it makes the network focus on the
human body. Zhang et al. [10] introduced a relation-aware global attention module to learn the global feature of person pictures. Rao et
al. [11] proposed to learn the attention with counterfactual causality,
which provides a tool to measure the attention quality and a powerful supervisory signal to guide the learning process. Li et al. [1] introduced MCTN, which integrates multi-granularity feature learning with
pyramidal cross attention mechanisms to enhance local-global feature
relationships.
With the further development of the Re-ID task, more and more researchers have switched from the CNN framework to the Transformer
framework in terms of network architecture for feature extraction. He et
al. [12] proposed the ﬁrst feature extraction work that fully adopts the
Transformer framework. The side information embeddings (SIE) are introduced to mitigate feature bias towards camera/view variations; This
work strongly supports Re-ID subtasks. Based on this framework, Luo et
al. [13] and Chen et al. [14] proposed diﬀerent self-supervision modes
to provide additional prior knowledge support for Re-ID tasks, thereby
improving the recognition accuracy from another perspective. A viewdecoupled transformer is developed by Zhang et al. [15] to decouple
view and identity features for aerial-ground person Re-ID through hierarchical separation mechanisms.
Recently, the unsupervised person Re-Id task with more robust generalization performance has been further studied. Zheng et al. [16] proposed a label transfer algorithm that simultaneously uses pseudo labels
to train the data while reﬁning the pseudo labels as an online clustering algorithm. To address cross-camera domain variations, Ran et al.
[17] developed CGMAL, which leverages heterogeneous graph learning and adversarial training to conduct multi-domain feature transfer
across camera domains. Meanwhile, Chen et al. [18] tackled the reliability issue of Jaccard distance by proposing CA-Jaccard distance, where
camera-aware neighbor selection and query expansion mechanisms are
designed to mitigate the negative impact of camera variation on distance
metric computation.
Nowadays, Re-ID tasks still face a few problems in realistic scenarios, such as cross-resolution problems, cross-modality problems, etc. Li
et al. [19] designed a discriminative semi-coupled projective dictionary model to learn multiple pairs of dictionaries and multiple mapping functions in variational resolution gap scenarios to address the
cross-resolution problems. Peng et al. [20] introduced MRLReID, which
employs resolution estimation as an auxiliary task to enhance crossresolution person re-identiﬁcation through multi-task learning. Wei
et al. [21] proposed a novel syncretic modality collaborative learning (SMCL) model to bridge the cross-modality gap. Multi-granularity
frameworks (DMANet and CM2GT) are proposed by Cheng et al. [22]
and Feng et al. [23] respectively to tackle cross-modality discrepancies,
which achieved advanced results.
However, few methods focus on designing speciﬁc losses for Re-ID
tasks, most of which are metric learning losses rather than classiﬁcation losses. Siamese loss and triplet loss are two common metric learning losses; however, directly using them cannot achieve satisfactory results. Hermans et al. [24] proposed a PK sampling strategy to learn hard

The modiﬁed softmax losses are usually used in face recognition
tasks. The angular margin in the form of multiplication was ﬁrst proposed in L-Softmax [25], and Normface [26] was a pioneer work that
normalizes the feature and weight of the naive softmax loss. Shortly
afterwards A-Softmax (Sphereface) [27] combined two advantages of LSoftmax loss and Normface, normalizing the weight and giving an angular margin in the form of multiplication. AM-Softmax [28] and CosFace
[29] normalized both feature and weight and changed the multiplication angular margin as minus cosine margin. Soon after, ArcFace [4]
changed the minus cosine margin of CosFace to an additive angular margin. More recently, ElasticFace [30] replaced ﬁxed margins with adaptive elastic margins that adjust based on class separability. X2-Softmax
[32] further introduced quadratic function-based adaptive margins that
automatically scale with inter-class angles.
There are two methods in Re-ID tasks that replicate the success of
Normface. SPhereReID [34] ﬁrst normalized the feature and weight in
Re-ID tasks and HSME [35] was proposed to solve the cross modality
Re-ID problems. Recently, Circle Loss [36] uniﬁed the softmax loss and
triplet loss, and thus achieved great success in many areas. Notably,
changing the negative integer factors added in the process of backpropagation in Circle loss can be converted into all kinds of softmax loss,
which is highly innovative with similar complexity.
3. Proposed method
In this section, we ﬁrst introduce the proposed Condense loss in detail. And a mathematical analysis comparison with other classiﬁcation
losses is presented to show the superiority of Condense loss.
3.1. Condense loss
The naive softmax loss is the most widely used classiﬁcation loss in
Re-ID tasks. It is usually combined with cross-entropy loss, which can
be presented as:
1 ∑
𝑒𝑓𝑖,𝑦𝑖
𝑙𝑜𝑔 ∑𝐾
,
𝑁 𝑖=1
𝑒𝑓𝑖,𝑘
𝑁

𝐿𝐶𝐸 = −

𝑓𝑖,𝑘 = 𝑊𝑘𝑇 𝑥𝑖 + 𝑏𝑘 ,

(1)

𝑘=1

where 𝑥𝑖 ∈ ℝ𝑙 denotes the 𝑙 dim feature of the 𝑖th sample, 𝑦𝑖 denotes
the one-hot label coding of the 𝑖th sample, 𝑊𝑘 denotes the 𝑘th column
of weight 𝑊 ∈ ℝ𝑙×𝐾 and 𝑏𝑘 is the bias parameter. 𝑁 and 𝐾 denote the
size of the mini-batch and the number of classes in the training dataset,
respectively. The formulation of 𝑓𝑖,𝑘 is the classiﬁcation function. For
simplicity, we ﬁx the bias 𝑏 as 0, and thus 𝑓𝑖,𝑘 can be rewritten as:
‖‖ ‖
𝑓𝑖,𝑘 = 𝑊𝑘𝑇 𝑥𝑖 = ‖
‖𝑊𝑘 ‖‖𝑥𝑖 ‖cos𝜃𝑘 ,

(2)

where 𝜃𝑘 is the angle between the weight 𝑊𝑘 and feature 𝑥𝑖 . L2 normalized softmax losses usually normalize 𝑊 or 𝑥 and set margin parameters in 𝑓𝑖,𝑘 function to give clear boundaries between diﬀerent classes.
Normface normalizes both 𝑊 and 𝑥, and sets a scale parameter 𝑠 to converge the network; its formulation is 𝑓𝑛𝑜𝑟𝑚𝑓 𝑎𝑐𝑒 = 𝑠 cos𝜃𝑘 . Althongh classiﬁcation functions 𝑓𝑖,𝑘 focus exclusively on classiﬁcation, the feature
extracted during the testing phase may not be discriminative enough.
To solve this problem, Sphereface and CosFace set margins to 𝑓𝑖,𝑘 ; their
formulations are 𝑓𝑠𝑝ℎ𝑒𝑟𝑒𝑓 𝑎𝑐𝑒 = 𝑠 cos(𝑚𝜃𝑘 ) and 𝑓𝑐𝑜𝑠𝑓 𝑎𝑐𝑒 = 𝑠 (cos𝜃𝑘 − 𝑚), respectively.
However, all these functions are inner-product based; we show their
gradients which are updating directions of class weight 𝑊𝑦𝑖 in Fig. 2(a).
The gradient direction of 𝑓𝑖,𝑘 = 𝑊𝑘𝑇 𝑥𝑖 is the same with vector 𝑥⃗𝑖 and the
⃗ 𝑦 (the lengh of 𝑊
⃗𝑦
gradient direction of 𝑓𝑖,𝑘 = 𝑠 cos𝜃𝑘 is vertical to 𝑊
𝑖

3

𝑖


codex
前四篇已经读完。这里的共同信号很清楚：能发 B 类的论文通常不是“多堆一个分支”，而是抓住一个训练信号的盲区，例如特征范数、伪标签可信度、无外部辅助的语义自对齐，再用可视化或梯度分析把证据链补齐。继续读剩下几篇。
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
/bin/zsh -lc "pdftotext -l 3 'Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

Cross-modal Collaborative Representation Learning for Text-to-Image Person
Retrieval
Shuanglin Yan1 , Jun Liu2 , Neng Dong1 , Liyan Zhang3∗ and Jinhui Tang1
1
Nanjing University of Science and Technology
2
Lancaster University
3
Nanjing University of Aeronautics and Astronautics
{shuanglinyan, neng.dong, jinhuitang}@njust.edu.cn, j.liu81@lancaster.ac.uk, zhangliyan@nuaa.edu.cn
Abstract

1

Identity i

i

The young man
is wearing sky
blue jacket....

j

A man in his
late twenties
with short....

j

The young man
is wearing sky
blue jacket....

Labled
Identity i

i

Identity j

j

i
j
i

i

i

j

i

j

i

j

Image Embedding

Text Embedding

Visual
Backbone
Image feature
Virtualtext
Branch
Virtual-text
feature

i

j
j

j j
j j

i

j

j

i

i

j

i
j

i
j

j

i

i

Labled
Identity j

0
β
α

i

Labled

i

j

Similarity Metric Space

/ j j Strong Positive Pairs
/ j j Weak Positive Pairs
i j / j i Negative Pairs

i

i

i

i

LIBM

Enhanced
Text
image feature feature

Textual
Backbone

A man in his
late twenties
with short....

Figure 1: (1) Examples of different types of image-text pairs.
Identity-bounded Matching (IBM) loss is designed to handle different types of image-text pairs distinctly by establishing different
similarity boundaries, leading to more accurate cross-modal correspondences. (2) We propose a cross-modal collaborative representation learning framework that leverages the inter-modality complementarity to enhance image features without relying on actual texts.

Introduction

Person re-identification (ReID) aims to retrieve a personof-interest across different camera networks. ReID models [Gong et al., 2022; Li et al., 2023b; Li et al., 2019a]
trained on extensive labeled cross-camera image pairs have
shown impressive retrieval capabilities. However, the closeset assumption of paired cross-camera images severely limits
∗

i

i

Cross-modal
Collaboration

Text-to-image person retrieval (TIPR) aims to find
images of the same identity that match a given
text description. Current TIPR methods mainly
focus on mining the association between images
and texts, ignoring their potential complementarity. Besides, existing matching losses treat all positive pairs from the same identity equally, leading to noisy correspondences. In this paper, we
propose CoRL: a cross-modal Collaborative Representation Learning framework designed to improve
TIPR by effectively leveraging the complementarity between modalities. The text typically contains
identity details with less noise, which helps distinguish visually similar pedestrians. This inspires us
to integrate it into the corresponding image to emphasize identity-related and modality-shared visual
information. However, corresponding text for each
image is not always available, especially during inference. Accordingly, we introduce a Virtual-text
Embedding Synthesizer that generates high-quality
virtual-text features for cross-modal collaboration,
eliminating the need for actual texts. We then design a Cross-Modal Collaboration learning process,
incorporating a Cross-modal Relation Consistency
loss to promote interaction and fusion between image and virtual-text features for mutual enhancement. Additionally, an Identity-bounded Matching
loss is proposed to handle different types of imagetext pairs distinctly, leading to more accurate crossmodal correspondences. Extensive experiments on
multiple benchmarks demonstrate the superiority
of CoRL over existing TIPR methods.

Identity-bounded Matching

A man in his
late twenties
with short....

Labled

Corresponding author

2152

the practical deployment of these models. Recently, Text-toimage person retrieval (TIPR) [Li et al., 2017] has been proposed to address situations where images are missing under
certain cameras. As a realistic extension to ReID, TIPR assumes that text descriptions are available for the missing images, allowing the retrieval of target person images via these
descriptions. The model must accurately discover identity information and establish correspondences between images and
texts, making TIPR a challenging and meaningful task.
Previous TIPR methods [Chen et al., 2022; Yan et
al., 2023d] have relied on single-modal pre-trained models [Dosovitskiy et al., 2021; Devlin et al., 2019] as backbone networks, leveraging their robust initial representations
to facilitate fine-tuning on training data. However, these

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

single-modal pre-trained models tend to overemphasize information from one modality, leading to significant modality gaps and alignment challenges. With the rise of visionlanguage pre-training (VLP), some works [Yan et al., 2023c;
Jiang and Ye, 2023] have adopted VLP models as backbone
networks, allowing TIPR to benefit from rich multi-modal information. Although promising results have been achieved,
existing methods mainly focus on modeling the association
between images and texts, often overlooking their potential
complementarity. In reality, the text primarily contains pedestrian identity details with less noise. Even when pedestrians
are visually similar, their text descriptions are often distinct,
making text information more discriminative and robust. This
leads us to explore whether inter-modality complementary information can benefit the TIPR task. Inspired by this, we propose integrating text information into images to emphasize
discriminative and modality-shared identity details, aiding in
distinguishing fine-grained differences and enhancing robustness to background noise. However, this approach requires
access to the corresponding text for each image, which is not
always available, especially during inference.
Accordingly, we propose a cross-modal Collaborative
Representation Learning (CoRL) framework that leverages
the inter-modality complementary information to enhance
image features without relying on actual texts. Specifically,
we introduce a Virtual-text Embedding Synthesizer (VES),
which utilizes CLIP’s vision-language alignment capabilities
to produce high-quality virtual-text embeddings directly from
images, eliminating the need for actual texts. These generated
virtual-text embeddings can replace actual texts in subsequent
cross-modal collaboration. To ensure consistency, we align
the generated virtual-text embeddings with actual text embeddings at both the feature and semantic levels. To fully exploit
the complementarity between images and texts, we design a
dual-branch cross-modal collaboration learning process, incorporating a cross-modal relation consistency loss (CRCL).
The visual backbone branch focuses on generating image features. The virtual-text branch uses VES to create virtual-text
embeddings from images, which are then fed into an Adapter
to produce virtual-text features adapted to the target domain.
The CRCL loss enforces that image and virtual-text features
maintain the same relationship with modality-specific prototypes. This encourages information exchange and collaboration between the two branches, allowing virtual textual information to emphasize the discriminative and modality-shared
identity details in the images, thereby enhancing feature discriminability and reducing modality gaps.
Cross-modal matching loss is essential for learning accurate correspondences between modalities. However, existing
losses [Zhang and Lu, 2018; Ding et al., 2021; Jiang and Ye,
2023] treat all positive pairs from the same identity equally,
resulting in noisy correspondences. Typically, each batch
contains three types of image-text pairs: single-view strong
positive pairs, cross-view weak positive pairs, and negative
pairs, as illustrated in Figure 1. Due to significant appearance differences under the same identity caused by view variations, cross-view weak positive pairs may suffer from noisy
correspondences. Consequently, the similarity between these
three types of image-text pairs should decrease progressively.

2153

To address this, we propose an Identity-bounded Matching
(IBM) loss, which defines distinct similarity boundaries for
each category of image-text pairs, thereby leading to more
precise cross-modal correspondences.
Here are the main contributions of our paper: (1) We
propose a cross-modal collaborative representation learning
framework, which is the first to leverage the inter-modality
complementary information to improve the TIPR task without relying on actual texts. (2) Identity-bounded matching loss is proposed to learn precise cross-modal correspondences. (3) Extensive experiments verify the effectiveness
of our method and achieve superior performance on multiple
benchmarks.

2

Related Work

2.1

Text-to-Image Person Retrieval

TIPR extends ReID [Gong et al., 2024; Dong et al., 2024a;
Dong et al., 2024b] to a more realistic scenario. The TIPR
model mainly contains two parts: the backbone network and
the feature alignment network [Tang et al., 2025]. A common practice is to use pre-trained backbones to leverage their
strong initial representation capabilities, facilitating effective
fine-tuning on TIPR data. Earlier methods [Shen et al., 2023;
Yan et al., 2023b] employed single-modal pre-trained models
such as ViT (pre-trained on ImageNet) and BERT. Recently,
the success of vision-language pre-trained models (VLPs) has
led to their widespread adoption in TIPR [Yan et al., 2023c;
Jiang and Ye, 2023], achieving promising results by exploiting their rich multi-modal knowledge. Notably, some recent
works [Yang et al., 2023; Tan et al., 2024] have advanced
further by retraining VLPs specifically for TIPR using largescale datasets, yielding additional performance gains.
For feature alignment network, various strategies have
been proposed to align images and texts. Early methods [Li et al., 2017; Zhang and Lu, 2018] directly aligned
the global features of images and texts. To achieve finegrained correspondences, later methods [Chen et al., 2022;
Yan et al., 2023d] introduced feature aggregation schemes to
generate multiple local features, modeling fine-grained alignment through interaction or guidance between these local features. However, this approach increases storage costs and inference time, reducing practicality. To avoid explicitly generating local features, recent methods [Jiang and Ye, 2023;
Li et al., 2023a] propose to inject fine-grained information [Tang et al., 2023; Yan et al., 2023a] into global features
by designing auxiliary tasks (e.g., masked language/region
modeling) to model fine-grained matching.
Despite these advancements, existing methods mainly emphasize image-text associations but overlook their complementarity. In this study, we explore inter-modality complementarity to enhance image features with text information.
Besides, existing matching losses treat image-text pairs of the
same identity equally. Although RaSa [Bai et al., 2023] incorporates a discriminator to differentiate between positive
pairs, it still relies on conventional matching losses, with the
discriminator serving only as a regularizer. In contrast, our
IBM loss directly addresses this issue by establishing distinct
boundaries for different types of image-text pairs.

Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)

The young man
is wearing sky
blue jacket....

Stage 1: Virtual-text Embedding
Synthesizer (VES)

Et

Lscor

LIBM

Visual Backbone
Branch

CRCL

s
Lalign
2

1

...

z v,i

Freezed Network
Learnable Network
CLIP Visual/Textual FC Single Fully
Ev Et Encoders
Connected Layer

Batch

g()
Adapter

Ev

s1

Lalign

...
Virtual-text features

Virtual-text Branch

Text features

...

...

...

VES

...

z^ t,i

Visual
Backbone

Lid

Textual
Backbone

Visual/Tex
tual Spaces
Visual/Textu
al Prototypes
Visual/Virtualtext Features
Relations
Information
Transfer

A man in his
lateThe
twenties
young man
with
short.... sky
is wearing
blue jacket....

...

FC

z t,i

Image features

...

Ti
Ii

...

Stage 2: Cross-modal Collaboration

A man in his
lateThe
twenties
young man
with
short.... sky
is wearing
blue jacket....

Batch

Cross-modal Relation Consistency Loss
(CRCL)

Figure 2: Overview of our CoRL. In the first stage, we introduce a Virtual-text Embedding Synthesizer to generate a virtual-text embedding
from each image. The generated virtual-text embedding is then used to replace actual text in modeling multi-modal complementarity with
the image. In the second stage, we propose a dual-branch Cross-Modal Collaboration learning process equipped with a Cross-modal Relation
Consistency Loss, which encourages the exchange of information and mutual enhancement between image and virtual-text features.

2.2

Vision-Language Pre-Training

Vision-language pre-training involves training large-scale
models on vast collections of image-text pairs, demonstrating exceptional capabilities in semantic understanding [Shen
and Tang, 2024; Shen et al., 2025], multi-modal alignment,
and generalization. It has been widely used in diverse downstream tasks. The primary objective of pre-training is to
achieve a deep understanding of both image and text semantics and their correspondences. Some studies [Li et al.,
2019b; Li et al., 2021] focus on generating multi-modal representations by encoding interactions between images and
texts with tasks such as mask language modeling and image captioning. However, these methods typically necessitate pairwise interactions for all image-text pairs, leading
to inefficiency during training and inference, thus limiting
their practicality in large-scale applications. Consequently,
some studies [Radford et al., 2021; Yao et al., 2022] integrate
contrastive representation learning into Vision-language pretraining. These methods encode images and texts separately
into a joint space and learn modality-aligned representations
by contrasting positive and negative pairs, such as the wellknown CLIP [Radford et al., 2021], trained on 400 million
image-text pairs. The strong multi-modal alignment capability of CLIP prompts us to introduce it into the TIPR task.

3

Methods

This section presents our proposed CoRL framework, with an
overview in Figure 2 and details in the following subsections.

3.1

Problem Formulation

The TIPR dataset contains image-text pairs of persons with
multiple identities, where each identity has multiple images
collected from different cameras with their annotated text descriptions. The goal is to accurately identify images that belong to the same identity as a given text description. This requires the TIPR model to extract identity-discriminative representations and accurately establish cross-modal correspondences. To achieve this, we first use a pre-trained CLIP as

2154

the backbone to provide high-quality initialization and facilitate fine-tuning on the TIPR dataset. Next, we propose a
cross-modal collaborative representation learning framework
that leverages inter-modality complementary information to
enhance image features. Additionally, we design an identitybounded matching loss to fully leverage identity information
and precisely establish cross-modal correspondences.

3.2

Cross-modal Collaborative Representation
Learning

Current TIPR methods mainly emphasize image-text associations but overlook their complementarity. Texts offer
the following advantages over images: texts primarily contain pedestrian identity details with less noise. Even when
pedestrians are visually similar, their text descriptions are
often distinct, making text information more discriminative
and robust. Complementing images with textual information
enhances fine-grained discrimination and reduces modality
gaps. However, this requires access to corresponding text for
each image, which is not always available, especially during inference. In this paper, we tackle two key challenges:
eliminating the dependency on actual text and effectively utilizing text information to enhance image representations. To
achieve this, we propose a two-stage cross-modal collaborative representation learning strategy. In the first stage, we
develop a Virtual-text Embedding Synthesizer (VES) to generate virtual-text embeddings directly from each image, eliminating the need for actual paired text. In the second stage, we
implement a dual-branch cross-modal collaboration learning
process to integrate the virtual-text information into images,
thereby enhancing image features.
Virtual-text Embedding Synthesizer. To eliminate reliance on actual paired text, we propose a Virtual-text Embedding Synthesizer (VES) that generates a virtual-text embedding directly from each image, as illustrated in Figure 2.
CLIP, trained on vast amounts of image-text data, can generate modality-aligned image-text features. We utilize its
alignment capabilities to convert images into text embeddings. VES functions as a CLIP-based encoder-decoder
model. Specifically, an image Ii is first processed by CLIP’s


exec
/bin/zsh -lc "pdftotext -l 3 'Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1084

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025

Corruption-Invariant Person Re-Identification via
Coarse-to-Fine Feature Alignment
Xinyu Zhang , Peng Zhang , Member, IEEE, and Caifeng Shan , Senior Member, IEEE
Abstract— Corruption-invariant
Person
Re-identification
(CI-ReID) aims to build robust identity correspondence across
non-overlapped cameras even when severe image corruptions
occur. It is challenging as those corruptions contaminate intrinsic
pedestrian characteristics and cause semantic misalignment
in feature space. To address this issue, this paper proposes
a coarse-to-fine semantic alignment framework that learns
corruption-invariant pedestrian features for re-identification
from the perspective of multi-modal feature alignment. In this
framework, a Coarse-to-Fine Feature Alignment Transformer
(CFAT) is introduced to extract and align features of pedestrian
images with different corruptions. Specifically, the CFAT aligns
features of corrupted samples to that of the corresponding
clean samples in a knowledge distillation manner in the coarse
alignment stage, i.e., a teacher network distils identity-related
semantics from clean samples and supervises the student
network learning semantic-consistent features from corrupted
samples. To avoid information loss of the strict alignment,
we propose to integrate a Bridge Feature Generation (BFG)
module into CFAT to construct meaningful latent structures
among modalities in the fine alignment stage. This enables
seamless alignment of the same identity between corrupted and
clean modalities, leading to better re-identification performance.
To evaluate the effectiveness of the proposed method, extensive
experiments are conducted on three public benchmark datasets,
i.e., Market-1501, CUHK-03, and MSMT-17. The experimental
results demonstrate our CFAT outputs state-of-the-arts with a
large margin in various corrupted scenes.
Index Terms— Corruption-invariant person re-identification,
corrupt scenes, multi-modal alignment, intermediate feature
generation.

I. I NTRODUCTION

C

ORRUPTION-INVARIANT person re-identification
(CI-ReID) [1] aims at re-identifying a target person
of interest even when image corruptions occur. In addition
Received 18 May 2024; revised 1 September 2024; accepted 25 September
2024. Date of publication 2 October 2024; date of current version 13 February
2025. This work was supported in part by the Talent Introduction Program for
Youth Innovation Teams of Shandong Province, in part by the National Natural
Science Foundation of China under Grant 62202280, and in part by Shandong
Provincial Natural Science Foundation under Grant ZR2021QF017. This
article was recommended by Associate Editor H.-C. Shih. (Corresponding
authors: Peng Zhang; Caifeng Shan.)
Xinyu Zhang is with the College of Electrical Engineering and Automation,
Shandong University of Science and Technology, Qingdao 266590, China
(e-mail: zhangxinyu@sdust.edu.cn).
Peng Zhang is with the College of Computer Science and Engineering,
Shandong University of Science and Technology, Qingdao 266590, China
(e-mail: pengzhang_skd@sdust.edu.cn).
Caifeng Shan is with the College of Electrical Engineering and Automation,
Shandong University of Science and Technology, Qingdao 266590, China, and
also with the State Key Laboratory for Novel Software Technology and the
School of Intelligence Science and Technology, Nanjing University, Nanjing
210023, China (e-mail: caifeng.shan@gmail.com).
Digital Object Identifier 10.1109/TCSVT.2024.3472122

to perturbations coming from factors such as illumination
difference, deformable poses and occlusion in traditional
ReID, CI-ReID involves extra noises induced by objective
factors in real-world deployment, such as villainous weather,
digital noise, imaging blur and image compression, etc. These
factors result in issues like blurred edge, occluded feature,
color jittering, etc., which cause corrupted image quality and
contaminate intrinsic pedestrian characteristics. Therefore,
CI-ReID confuses existing ReID models and yields inferior
identification performance. This limits the scalability of the
existing ReID methods to the wild scene.
Though CI-ReID is valuable for practical deployment, it has
been overlooked by the mainstream ReID research for a long
time. Until 2021, Chen et al. [1] pioneered the research that
established benchmarks and built a baseline for CI-ReID.
Aided by some tricks, i.e., local-based augmentation [2],
consistency identity loss and inference before BNNeck [3], the
baseline improved robustness for inference with corruptions.
In spite of this, such primary work is still far from satisfaction
due to significant dataset bias [4] between training phase and
inference phase. In another word, the baseline model trained
on clean/un-corrupted data generalizes poorly to corrupted
scene due to the large distribution gap.
To better address the above issue, we carefully investigate
genesis of image corruptions and its impact on data distribution. As illustrated in [5], a total of 20 typical types of
corruptions could degrade person ReID performance in realworld deployment. These corruptions can be categorized into
four classes according to their causes, i.e., noise, blur, weather,
and digital. Fig. 1(a) shows examples of these different types
of corruptions. It can be observed that these corruptions
severely lower the image quality which further contaminates intrinsic pedestrian characteristics in monitoring scene.
We further visualize the distribution of pedestrian images with
different corruptions using t-SNE [6] in Fig. 1(b). From the
figure, we can see that these corruptions bias seriously data
distribution, which cause samples affected by the same type of
corruption to be clustered together, while there exists a certain
gap among samples affected by different types of corruptions.
This suggest CI-ReID can be regarded as pedestrian retrieval
between heterogeneous data, where each type of image corruption can be seen as an image modality. This inspires us to
address CI-ReID from the perspective of multi-modal feature
learning, and learn identity-related discriminative features in
a modality-unified semantic subspace.
Inspired by the insights mentioned above, this paper introduces a pedestrian semantic alignment framework for CI-ReID
that aligns distribution of clean and corrupted pedestrian

1051-8215 © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:25 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: CORRUPTION-INVARIANT PERSON RE-IDENTIFICATION VIA COARSE-TO-FINE FEATURE ALIGNMENT

1085

Fig. 1. An illustration of samples in CI-ReID. (a) The top two rows list different types of image corruptions in real-world deployment, and the bottom row
shows image degradation under different levels of corruptions; (b) the distribution of corrupted samples using t-SNE [6].

samples in the semantic space. To ease the feature alignment, we adopt a coarse-to-fine feature alignment strategy.
Specifically, a Dual-stream Feature Alignment (DFA) model
is proposed to achieve coarse alignment, which includes a
siamese Teacher Feature Extraction (TFE) network and a
Student Feature Extraction (SFE) network. It is motivated
by the idea of knowledge distillation [7] that the pre-trained
TFE on clean dataset supervises the further training of the
pre-trained SFE network on the corrupted dataset. This idea
regards one clean sample of the target pedestrian as the
exemplar and forces semantics of its corresponding corrupted
samples to be close with the exemplar. This helps SFE restore
pedestrian features from corrupted images and further benefits
person ReID. Particularly, we propose a weighted contrastive
loss, and the weight is positively correlated to the distance
between samples, which adaptly imposes a larger penalty to
challenging samples, such as samples with severe corruptions
in Level 5 (Lv5) as shown in Fig. 1(a).
Recent studies [8] suggests the strict modality alignment
in latent space can lead to loss of mode-specific information,
resulting in sub-optimal performance. However, by expanding
the lower bound of the approximate alignment, more useful information can be retained, and better alignment can
be obtained. In another aspect, it is difficult to completely
restore features from corrupted samples from the perspective
of information-theoretic argument. Therefore, we propose to
relax the regularization of strict alignment and construct the
latent structure between modalities. Motivated by the Brownian bridge principle [9], [10], this paper introduces the Bridge
Feature Generation (BFG) module to generate the intermediate features between modalities. By filling the intermediate
features between clean and corrupt modalities, a seamless
semantic structure is constructed. The BFG extends the range
of corrupt modality in the semantic space and achieving fine
alignment. It is worth noting that we adapt the re-parameter
trick into BFG to generate intermediate features in stochastic
path of Brownian motion. Benefiting from modality transition,

the BFG expands intra-person cluster, which enables the
proposed method to be more capable of achieving person ReID
under corruption scene.
In summary, the main contributions of this work are threefold:
• Based on our insights on corrupt images, we propose to
formulate CI-ReID as a modal alignment problem, aiming
to recover pedestrian features from corrupted images
by aligning clean and corrupt modalities. Additionally,
we introduce a novel modality alignment framework for
this problem.
• We adopt a coarse-to-fine feature alignment strategy to
optimize the framework, utilizing a novel dual-stream
feature alignment module to achieve coarse alignment
by aligning features of corrupted samples to their corresponding clean features under teacher-student paradigm.
On this basis, a novel bridge feature generation module
is introduced to construct the latent structure between
modalities for fine alignment.
• Extensive experiments are conducted on three benchmark
datasets, e.g., Market-1501 [11], CUHK-03 [12], and
MSMT-17 [13]. Under four different evaluation settings,
the proposed method outperforms the state-of-the-art
ReID methods with a large margin.
The remaining parts of this paper are organized as follows.
In Section II, we review works most closely related to this
paper. In Section III, we introduced the proposed method
in detail. Section IV presents all the experimental content.
Section V summarizes the paper.
II. R ELATED W ORK
A. Person Re-Identification (Re-ID)
In the earlier years, ReID was considered as a sub-problem
of cross-camera tracking until Gheissari et al. [14] first defined
ReID as an independent topic. ReID can be classified into
close-world problem and open-world one. With the saturation

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:25 UTC from IEEE Xplore. Restrictions apply.

1086

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025

of performance in closed-world scenarios, the focus of ReID
research has gradually shifted to open-world scenarios. For
example, Zhou et al. [15] proposed the concept of omni-scale
learning to tackle the need for attention to features at different
scales in ReID. Li et al. [16] developed a Diverse and
Compact Transformer to learn discriminative representations.
Due to various challenges in open-world scenarios, such as
diverse data collection methods, uncontrollability of pedestrians, and environmental factors, pedestrian re-identification
faces many different challenges [1]. To address these issues
in open scenarios, Pang et al. [17] proposed a cross-modality
hierarchical clustering and refinement method to address the
task of cross-modal pedestrian image retrieval. Jin et al. [18]
addressed the issue of occlusion in pedestrian re-identification
by calculating the differences between occluded and nonoccluded pedestrians. We refer [19] for more related works.
However, the study of corrupted images in the field of ReID
is lacking attention. Image corruption will cause edge blur,
feature occlusion, and color jitter, which will confuse existing
ReID models. The study of corrupted images, considered
as heterogeneous data compared to clean images, is also
overlooked in the field of ReID. Therefore, this paper conducts
research on CI-ReID, hoping to obtain a model with higher
recognition accuracy in corrupted scenes.
B. Corrupted Image Recognition
In the real world, the obtained images are often affected
by various disturbing factors such as noise, blur, weather
conditions, or data corruption, which severely degrades image
quality [20], [21] and yields superior downstream performance [1]. Research on corrupted images in the field of
computer vision has a long history [22], [23]. However, since
it is challenging to obtain certain types of corrupted images
from the real world, synthetic methods have been employed
to create datasets for corrupted images, using image quality assessment methods [20], [24] for validation. Currently,
datasets such as CIFAR-10-C, CIFAR-100-C, and ImageNet-C
[5] have been established for corrupted image recognition,
which has drawn widespread interest in the study of corrupted images. Corruption benchmarks have been proposed in
various domains, including object detection [25], pose estimation [26], and medical imaging [27]. Additionally, benchmarks
for corruption-invariant person re-identification [1] have also
been introduced, aiming to enhance the feature representation
capability and improve the model’s generalization performance
through representation learning and metric learning methods.
However, the results of this work are still far from satisfaction due to significant dataset bias between training phase
and inference phase. To mitigate the impact of dataset bias,
Wei et al. [28] introduced a novel exponential loss that
adaptively assigns weights based on the similarity of sample
pairs to address the cross-modal retrieval problem. Li et al.
[29] generated low-quality images through data augmentation
and utilized a realistic UIESR network for reconstruction,
enabling the model to enhance image texture and correct
color bias. Different from them, in this work we diversify the
inputs during the training phase through data augmentation,

introduce weighted contrastive loss, and extract the most
essential features through knowledge distillation [7], [30] to
obtain a recognition model that is more robust to diverse
corrupted data.
C. Teacher-Student Learning
Teacher-student learning regime [31] is widely utilized for knowledge distillation (KD) [7]. It builds
knowledge correspondence between models to perform
model compression and distillation. Inspired by KD,
Tarvainen et al. [32] introduced a semi-supervised Mean
Teacher model to enhance the model’s generalization capability. It forces outputs from teacher and students consistent
by adding different perturbations to them. The teacher network reflects the Student’s state over time. In addition,
Zhang et al. [33] developed a teacher-student learning strategy
to mitigate the distribution discrepancies between imbalanced
classes by constraining the predicted probabilities to achieve
alignment between the classes. Zhang et al. [34] proposed a
cross-resolution relational contrastive distillation approach for
low-quality image recognition through cross-quality knowledge transfer, achieving feature alignment between images of
different qualities. This work is also inspired by KD, but it is
built on quite another formulation. In this work, the teacher
acts as an exemplar regularizer that is pre-trained on clean
samples. It regularizes the latent feature structure constructing
from multi-source corruptions with the help of our Bridging Feature Generation module. Under this motivation, our
proposed method preserves identity information as much as
possible though various severe corruption happens.
D. Modality Alignment
The purpose of cross-modal alignment is to explore the correlations between sub-elements of multi-modal data. Modality
alignment is widely used in multi-modal tasks, and the main
methods include explicit alignment [35] and implicit alignment [36], [37]. If a model’s optimization objective is to
maximize the alignment of sub-elements of multi-modal data,
it is referred to as explicit alignment. Explicit alignment
includes unsupervised [38], [39] and supervised [40], [41]
methods. Unsupervised methods do not have annotations,
and during the process of aligning sub-elements, the model
needs to learn similarity measures and alignment methods
simultaneously. Supervised methods involve annotations and
enable the training of models to learn similarity measures.
If the final optimization objective of the model is not an alignment task, and the alignment process is only an intermediate
(or implicit) step, it is referred to as implicit alignment. Some
implicit alignment methods [42] require manually constructing
mappings between modalities. Other alignment methods [43]
are based on attention mechanisms, which compute attention
weight matrices between sub-elements of two modalities,
implicitly measuring the degree of correlation between crossmodal sub-elements. For more in-depth information, we refer
to [44]. In 2023, Jiang et al. [8] aligned text and image features
by constructing implicit structure between modalities. This
inspired us to achieve fine alignment by filling bridge features

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:25 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 166 (2025) 111702

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Cross-domain person re-identification via learning Heterogeneous Pseudo
Labels
Zhong Zhang , Di He , Shuang Liu ∗
Tianjin Key Laboratory of Wireless Mobile Communications and Power Transmission, Tianjin Normal University, Tianjin 300387, China

ARTICLE

INFO

Keywords:
Cross-domain person ReID
Heterogeneous Pseudo Labels
Contrastive learning

ABSTRACT
Assigning pseudo labels is vital for cross-domain person re-identification (ReID), and most existing methods
only assign one kind of pseudo labels to unlabeled target domain samples, which cannot describe these
unlabeled samples accurately due to large intra-class and small inter-class variances caused by diverse
environmental factors, such as occlusions, illuminations, viewpoints, and poses, etc. In this paper, we propose
a novel label learning method named Heterogeneous Pseudo Labels (HPL) for cross-domain person ReID,
which could overcome large intra-class and small inter-class variances between pedestrian images in the target
domain. For each unlabeled target domain sample, HPL simultaneously learns three different kinds of pseudo
labels, i.e., fine-grained labels, coarse-grained labels, and instance labels. With the three kinds of labels, we
could make full use of their own advantages to describe target domain samples from different perspectives.
Meanwhile, we propose the Pseudo Labels Constraint (PLC) to improve the quality of the heterogeneous labels
by using their consistency. Furthermore, in order to relieve the influence of noisy labels from the aspect of
contrastive learning, we propose the Confidence Contrastive Loss (CCL) to consider the sample confidence in
the learning process. Extensive experiments on four cross-domain tasks demonstrate that the proposed method
achieves a new state-of-the-art performance, for example, the proposed method achieves 87.2% mAP and
95.0% Rank-1 accuracy on MSMT17→Market.

1. Introduction
Person re-identification (ReID) [1–3] aims to match a target pedestrian captured from non-overlapping cameras at different time or different locations, which is of great significance for automatic surveillance,
pedestrian behavior analysis and smart city [4–7]. The single-domain
person ReID technique is where both the training and test samples
are from the same domain. Though the method achieves impressive
performance, it significantly degrades when directly applied to an
unseen domain due to the inevitable domain gap [8–10]. To reduce the
domain gap, cross-domain person ReID is proposed to simultaneously
exploit source domain samples with identity labels and target domain
samples without label annotations, so that the deep model is well
generalized to the target domain.
The clustering-based method is one kind of representative methods
for cross-domain person ReID [10–15], and it mainly contains three
steps: (1) using labeled source domain samples to obtain pre-training
models; (2) assigning pseudo labels to unlabeled target domain samples using clustering algorithms, such as 𝐾-means and DBSCAN; (3)
fine-tuning models/ learning features with the pseudo labels. Steps
(2) and (3) are iteratively performed to ameliorate each other until

convergence. The key stage for cross-domain person ReID is the pseudo
label assignment. The existing methods [10,12,13] only assign one kind
of pseudo labels using clustering algorithms based on the similarities
between samples. However, it is difficult to describe each unlabeled
target domain sample accurately by assigning only one kind of pseudo
labels for iterative optimization because these samples are captured
from unconstrained environments, where the pedestrian appearances
are easily influenced by many environmental factors such as occlusions,
illuminations, viewpoints, and poses, etc., leading to large intra-class
and small inter-class variances [16,17]. Hence, learning one kind of
label results in inaccurate pseudo labels for the target domain samples
as shown in Fig. 1(a).
In this paper, we propose a novel label learning method named
Heterogeneous Pseudo Labels (HPL) for cross-domain person ReID,
which could overcome large intra-class and small inter-class variances
between pedestrian images in the target domain. As shown in Fig.
1(b), HPL simultaneously learns three different kinds of pseudo labels
for iterative optimization to obtain a completed description of each
unlabeled target domain sample, and the three kinds of pseudo labels
include fine-grained labels, coarse-grained labels, and instance labels.

∗ Corresponding author.

E-mail address: shuangliu.tjnu@gmail.com (S. Liu).
https://doi.org/10.1016/j.patcog.2025.111702
Received 16 September 2024; Received in revised form 5 April 2025; Accepted 8 April 2025
Available online 19 April 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 166 (2025) 111702

Z. Zhang et al.

Fig. 1. (a) The most existing methods only learn one kind of pseudo label for iterative optimization, making it difficult to describe each target domain sample accurately. (b) The
proposed method learns heterogeneous pseudo labels, such as fine-grained label, coarse-grained label, and instance label, for iterative optimization to describe each target domain
sample more comprehensively from different aspects.

• We propose Heterogeneous Pseudo Labels which could overcome large intra-class and small inter-class variances in the crossdomain person ReID, where fine-grained labels, coarse-grained
labels, and instance labels are designed to describe unlabeled
target domain samples from three different perspectives. Furthermore, to improve the quality of pseudo labels, we propose the
Pseudo Labels Constraint to maintain the consistency between the
heterogeneous labels.
• We propose the Confidence Contrastive Loss to re-weight the similarity of sample pairs by considering the confidence coefficient so
as to reduce the influence of the samples with noisy labels in the
process of contrastive learning.
• Extensive experiments on four cross-domain tasks demonstrate
the superiority of the proposed method, and it achieves a new
state-of-the-art performance.

The fine-grained labels are derived from a sufficient number of cluster
centers generated by the clustering algorithm, which could capture the
subtle distinctions among different identities. Meanwhile, the coarsegrained labels are formed by fewer and broader clusters, which could
reduce the influence of noise by avoiding fine divisions of the feature
space. Moreover, the instance labels are learned by the relationships
between the samples, providing insights into the individual sample
characteristics and their similarities or differences with other samples.
In a word, the three kinds of pseudo labels describe the same target
domain sample from different perspectives, and using all three kinds
of pseudo labels makes the best of their own advantages in order to
describe target domain samples more comprehensively.
To improve the quality of generated pseudo labels, we propose the
Pseudo Labels Constraint (PLC) to regulate the process of label learning.
To this end, we enforce the instance labels to be consistent with the
fine-grained labels and the coarse-grained labels, because the three
kinds of pseudo labels describe the same target domain sample. But
these pseudo labels are heterogeneous, that is, they distribute in different feature spaces. Correspondingly, we design the label transformation
strategy so that the three kinds of pseudo labels could be constrained
after transformation.
Recently, contrastive learning combined with the memory bank
is adopted to learn discriminative features for cross-domain person
ReID [13,14,18]. Specifically, for each sample (anchor) of the target
domain in contrastive learning, the positive and negative samples are
selected from the memory bank according to pseudo labels. However,
most existing contrastive learning neglects the influence of noisy labels
which are inevitable in the process of pseudo labels generation. To
overcome this limitation, we propose the Confidence Contrastive Loss
(CCL) to consider the sample confidence in the learning process. To this
end, we first compute the cluster center of target domain samples with
the same identity in the memory bank. Then, we utilize the similarity
between the cluster centers of fine-grained labels and memory bank as
the confidence coefficient. After obtaining the confidence coefficient,
it is used to re-weight the similarity of sample pairs in order to rectify
the optimization process of contrastive learning.
The main contributions of the proposed method are summarized as
follows:

The rest of the paper is organized as follows. Section 2 gives a revisit
of related work, Section 3 details the proposed method including Heterogeneous Pseudo Labels, Pseudo Labels Constraint, and Confidence
Contrastive Loss, Section 4 shows extensive experiments to verify the
advantages of the proposed method, and finally, we make a conclusion
and discussion in Section 5.
2. Related work
Cross-domain image learning is crucial in practical applications,
because it equips models with strong generalization capabilities when
training and test image distributions do not overlap, mitigating performance degradation due to distributional shifts. Recently, cross-domain
image learning has gained popularity across various tasks, including
image classification [19,20], palmprint recognition [21–23], person reidentification [15,17,24,25], and so on. In this paper, we focus on
cross-domain person re-identification, which is mainly categorized into
three, i.e., clustering-based methods, domain transfer based methods
and memory bank based methods.
2.1. Clustering-based methods
Generating pseudo labels is a crucial stage in cross-domain person
ReID, and the quality of the generated pseudo labels determines its
2

Pattern Recognition 166 (2025) 111702

Z. Zhang et al.

performance. The pseudo labels are generally classified into two kinds,
i.e., hard labels and soft labels. Some methods utilize clustering algorithms to obtain the hard (one-hot) labels [14,26–28]. For example,
both Fan et al. [26] and Zheng et al. [14] apply the clustering algorithms once or several times to generate the hard labels, so that the
model can be trained on the target domain in a supervised way. In
order to adapt better to the feature distribution of the target domain,
Fu et al. [28] propose to assign multi-scale hard labels by exploiting
the global and local features.
In order to reduce the overfitting of the deep model, some methods [24,29,30] design different strategies to generate the soft labels.
Zhong et al. [24,29] propose to learn soft labels by exploiting the stored
averaged features in an exemplar memory module so as to alleviate the
intra-domain variations in the target domain. Yu et al. [30] learn the
soft multilabel by introducing the labeled auxiliary domain as reference
persons in the unlabeled target domain. Furthermore, the soft labels
generated from the deep network are beneficial in combination with
pseudo labels or sample denoising [10,13,31,32]. Ge et al. [10] propose
Mutual Mean-Teaching (MMT) to refine the soft labels generated by
peer networks using mutual learning. Zheng et al. [13] propose an
Uncertainty-guided Noise Resilient Network (UNRN) that estimates the
reliability of the pseudo labels and softly exploits them to alleviate the
negative effects of noise-prone samples.
However, these methods only assign one kind of pseudo labels to unlabeled target domain samples, which cannot describe these unlabeled
samples accurately. Different from the above-mentioned methods, the
proposed method is to gradually approximate the real label distribution
of the target domain through learning Heterogeneous Pseudo Labels,
where we utilize different clustering algorithms to generate three different kinds of pseudo labels. Furthermore, we propose the Pseudo Labels
Constraint to regulate the process of label learning in order to improve
the labels quality. Hence, the proposed method could describe target
domain samples more comprehensively and accurately.

Ge et al. [18] and Zheng et al. [13] combine contrastive learning
with the memory bank to jointly optimize positive and negative pairs
for a query/anchor sample, thereby enhancing discrimination of the
representations. Sun et al. [43] and Luo et al. [25] focus on the learning
process of the deep model based on the memory bank, so as to improve
the performance of cross-domain person ReID.
Furthermore, the noisy samples in contrastive learning results in
the performance degradation of the person ReID model, and some
methods [13,44,45] alleviate this problem by weighting the samples
for person ReID. For example, Zheng et al. [13] and Han et al. [45]
present to assign different weights to target domain samples based on
the uncertainty, thereby improving the discriminative ability of the
cross-domain person ReID model.
Different from these existing methods, we propose the Confidence
Contrastive Loss combined with the memory bank to obtain accurate
similarity of sample pairs by considering the sample confidence, where
the well-designed confidence coefficient is beneficial to adjusting the
similarity of sample pairs adaptively, so as to alleviate the influence of
noisy labels.
3. Approach
In this section, we first present an overview of the proposed method
and then describe the major parts, i.e., Heterogeneous Pseudo Labels,
Pseudo Labels Constraint, and Confidence Contrastive Loss in detail.
3.1. Overview
The cross-domain person ReID model is trained by the labeled
source domain and the unlabeled target domain. Formally, the samples
and the ground-truth labels in the source domain are denoted as D𝑠 =
𝑛𝑠
𝑛𝑠
{𝑥𝑠𝑖 |𝑖=1
} and {𝑦𝑠𝑖 |𝑖=1
} ∈ [1, … , 𝐾𝑠 ], where 𝑛𝑠 and 𝐾𝑠 denote the number
of samples and identities in the source domain, respectively. Similarly,
𝑛𝑡
D𝑡 = {𝑥𝑡𝑖 |𝑖=1
} represents 𝑛𝑡 samples in the target domain and there is no
ground-truth labels in the target domain.
As illustrated in Fig. 2, we adopt ResNet50 [46] as the backbone
to build a mean teacher network and a student network [47]. We
follow the general pipeline of clustering-based cross-domain person
ReID methods [12–14,26], in which they include three steps (model
pre-training, pseudo label assignment, and model fine-tuning), and the
last two steps are performed iteratively until convergence.
In the stage of the model pre-training, we exploit the labeled source
domain samples to train a ReID model and it is utilized to initialize
the mean teacher network 𝐹 (⋅) and the student network 𝐺(⋅). In the
process of clustering, we extract the features of target domain samples
𝑓𝑖𝑡 = 𝐹 (𝑥𝑡𝑖 ) (𝑖 = 1, … , 𝑛𝑡 ) from the mean teacher network. Then the finegrained labels and the coarse-grained labels are generated by clustering
these features. Meanwhile, the instance labels are learned using all the
target domain features and the source domain centers in the memory
bank. During the model fine-tuning, the three kinds of pseudo labels
are simultaneously utilized to train the student network. In particular,
the student network is updated in a real-time manner, and the mean
teacher network is updated by exponential moving average (EMA) of
parameters of the student network.

2.2. Domain transfer based methods
One of the challenges of the cross-domain person ReID task is to
align the data distributions between the source domain and the target
domain, and some methods [33–38] utilize the Generative Adversarial
Networks (GANs) with the source samples to generate specific images
or features of the target domain to align the data distributions of
different domains. For example, Deng et al. [33] and Wei et al. [34]
transfer the labeled source domain images to the style of the target
domain while preserving the identity labels so as to learn the information from the target domain. Zhong et al. [35] and Zhou et al. [38]
transfer the images in the target domain to multiple camera styles for
domain adaptation. Since the images or features generated are always
different from those of the real scene, the GANs-based approaches are
still imperfect.
2.3. Memory bank based methods
The memory bank is utilized to enhance the ability of feature representation and has been widely applied in the fields of unsupervised
and self-supervised learning, especially in contrastive learning [39–41].
The core idea of the memory bank is to store a large number of feature
representations and provide abundant negative samples for contrastive
learning, where it could keep up-to-date representations using the
specific update mechanism. In the field of cross-domain person ReID,
Zhong et al. [29] apply a memory bank for the first time, and propose
to treat each target sample as an individual identity and store them in
an exemplar memory module to adapt to the intra-domain variations
in the target domain.
Recently, a variety of strategies [13,18,25,42,43] based on contrastive learning and memory bank have been proposed to improve
the performance of cross-domain person ReID models. For example,

3.2. Heterogeneous pseudo labels
Most existing methods have achieved impressive performance by
assigning one kind of pseudo labels to the unlabeled target domain
samples for cross-domain person ReID. However, only assigning one
kind of pseudo labels is difficult to describe unlabeled target domain
samples accurately because they are easily affected by complex environments. To overcome this limitation, we propose to apply three
kinds of pseudo labels, i.e., fine-grained labels, coarse-grained labels
and instance labels, for cross-domain person ReID.
3


codex
第二组补上了跨域、腐蚀鲁棒和文本图像检索。值得注意的是，这批反复出现“把原任务重新解释成另一个结构化问题”的写法，例如把腐蚀当成多模态、把目标域样本当成多粒度伪标签集合，这比单纯加注意力更容易撑起论文故事。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -l 3 'Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Cross-Modal Full-Mode Fine-Grained Alignment for
Text-to-Image Person Retrieval
HAO YIN and XIN MAN, Shenzhen Institute for Advanced Study, University of Electronic Science and
Technology of China, Shenzhen, China

FEIYU CHEN, JIE SHAO, and HENG TAO SHEN, University of Electronic Science and Technology
of China, Chengdu, China and Sichuan Artificial Intelligence Research Institute, Yibin, China
Text-to-Image Person Retrieval (TIPR) is a cross-modal matching task designed to identify the person images
that best correspond to a given textual description. The key difficulty in TIPR is to realize robust correspondence
between the textual and visual modalities within a unified latent representation space. To address this
challenge, prior approaches incorporate attention mechanisms for implicit cross-modal local alignment.
However, they lack the ability to verify whether all local features are correctly aligned. Moreover, existing
methods tend to emphasize the utilization of hard negative samples during model optimization to strengthen
discrimination between positive and negative pairs, often neglecting incorrectly matched positive pairs. To
mitigate these problems, we propose FMFA, a cross-modal Full-Mode Fine-Grained Alignment framework,
which enhances global matching through Explicit Fine-Grained Alignment (EFA) and existing implicit relational
reasoning—hence the term “full-mode”—without introducing extra supervisory signals. In particular, we
propose an Adaptive Similarity Distribution Matching (A-SDM) module to rectify unmatched positive sample
pairs. A-SDM adaptively pulls the unmatched positive pairs closer in the joint embedding space, thereby
achieving more precise global alignment. Additionally, we introduce an EFA module, which makes up for the
lack of verification capability of implicit relational reasoning. EFA strengthens explicit cross-modal fine-grained
interactions by sparsifying the similarity matrix and employs a hard coding method for local alignment.
We evaluate our method on three public datasets, where it attains state-of-the-art results among all global
matching methods. The code for our method is publicly accessible at https://github.com/yinhao1102/FMFA.
CCS Concepts: • Information systems → Image search; • Computing methodologies → Object identification;
Additional Key Words and Phrases: Cross-modal retrieval, Person search, Fine-grained alignment

This work was supported by the National Natural Science Foundation of China (No. 62302080), Guangxi Key Research and
Development Program (No. Guike AB24010112), National Foreign Expert Project of China (No. S20240327), Sichuan Science
and Technology Program (No. 2025HJRC0021), and Sichuan Province Innovative Talent Funding Project for Postdoctoral
Fellows (No. BX202312).
Authors’ Contact Information: Hao Yin, Shenzhen Institute for Advanced Study, University of Electronic Science and
Technology of China, Shenzhen, China; e-mail: yinhao1102@std.uestc.edu.cn; Xin Man, Shenzhen Institute for Advanced
Study, University of Electronic Science and Technology of China, Shenzhen, China; e-mail: manxin@std.uestc.edu.cn;
Feiyu Chen, University of Electronic Science and Technology of China, Chengdu, China and Sichuan Artificial Intelligence
Research Institute, Yibin, China; e-mail: chenfeiyu@uestc.edu.cn; Jie Shao (corresponding author), University of Electronic
Science and Technology of China, Chengdu, China and Sichuan Artificial Intelligence Research Institute, Yibin, China;
e-mail: shaojie@uestc.edu.cn; Heng Tao Shen, University of Electronic Science and Technology of China, Chengdu, China
and Sichuan Artificial Intelligence Research Institute, Yibin, China; e-mail: shenhengtao@hotmail.com.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2026/4-ART135
https://doi.org/10.1145/3786798
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

135:2

H. Yin et al.

ACM Reference format:
Hao Yin, Xin Man, Feiyu Chen, Jie Shao, and Heng Tao Shen. 2026. Cross-Modal Full-Mode Fine-Grained
Alignment for Text-to-Image Person Retrieval. ACM Trans. Multimedia Comput. Commun. Appl. 22, 5, Article 135
(April 2026), 21 pages.
https://doi.org/10.1145/3786798

1

Introduction

Text-to-Image Person Retrieval (TIPR) seeks to understand natural language descriptions and
identify the most relevant person image within a large gallery [31]. Unlike general image-text
retrieval [5, 6, 44, 47, 50], which tends to achieve semantic-based matching between text and image,
TIPR is specifically designed for identifying individuals. TIPR requires the accurate modeling of
fine-grained correspondences between textual and visual modalities, owing to the large intra-class
variance and small inter-class difference. This substantial intra-class variation arises from two
aspects: (1) visual appearances of the same identity exhibit dramatic variations under different poses,
viewpoints, and illumination conditions, and (2) textual descriptions are influenced by differences
in phrasing, word order, and textual ambiguities. Therefore, the primary challenges in TIPR are
how to extract discriminative global representations from image-text pairs and how to achieve
precise cross-modal fine-grained alignment. Existing methods for tackling these challenges can be
roughly divided into two main categories: global matching methods and local matching methods.
Some global matching methods [58, 59] obtain discriminative global representations by aligning
images and texts, which are projected into a joint embedding space. Their widely adopted loss
functions include the Cross-Modal Projection Matching (CMPM) loss [58] and the Similarity Distribution Matching (SDM) loss [21]. The CMPM loss highlights the gap between the
scalar projections of image-text pairs and their matched label indicators. In comparison, the SDM
loss boosts global matching performance by minimizing the Kullback-Leibler (KL) divergence
between the normalized similarity profile of image-text pairs and the true label distribution. In
addition, the SDM loss incorporates a temperature hyperparameter to make model updates concentrate on hard negative samples, yet it leads to the neglect of unmatched positive pairs, as shown
in Figure 1(a). However, in TIPR, the accurate matching of positive pairs is prioritized over merely
distinguishing between positive and negative pairs. Meanwhile, some local matching methods
[2, 11, 35] incorporate attention mechanisms to achieve cross-modal fine-grained alignment. For
instance, RaSa [2] constructs a cross-modal encoder to generate multimodal representations for
subsequent fine-grained alignment. Building on RaSa, MARS [11] integrates a Masked AutoEncoder
decoder [14] to reconstruct masked image patch sequences into their original unmasked form,
thereby facilitating cross-modal fine-grained alignment. However, these methods rely on attention
mechanisms to implicitly aggregate local image-text representations. As a result, they yield only the
final multimodal representation, without revealing the details of the aggregation process. Consequently, these implicit aggregation methods make it difficult to determine whether the aggregated
multimodal representations correctly encode the corresponding visual and textual information.
To remedy these concerns, we propose a cross-modal Full-Mode Fine-Grained Alignment
(FMFA) framework that enhances global matching by including explicit fine-grained image-text
alignment and existing implicit relational reasoning. Specifically, we design an Adaptive Similarity
Distribution Matching (A-SDM) module to ensure the correct matching of positive image-text
pairs. Within the joint embedding space, the A-SDM module adaptively pulls positive pairs closer
together. In cases of mismatched positive pairs, the A-SDM module adaptively regulates the pulling
force based on their relative distance within the joint embedding space, as shown in Figure 1(b),
thus improving cross-modal global alignment. Based on the insight that each word in a caption can
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.

Cross-Modal Full-Mode Fine-Grained Alignment for TIPR

135:3

Fig. 1. Evolution of global matching methods for TIPR. (a) Existing global matching methods focus on
hard negative samples to learn a discriminative boundary in a common latent space, thus enhancing the
distinction between positive and negative samples. (b) Our adaptive global matching method builds on a
discriminative boundary and concentrates on unmatched positive samples, adaptively pulling them closer to
the corresponding query text.

be associated with several image patches [3], we introduce an Explicit Fine-Grained Alignment
(EFA) module. The EFA module derives multimodal representations through explicit aggregation
with a sparse similarity matrix. During this process, the sparse similarity matrix between text
and image reflects the contribution of textual and visual representations to the final multimodal
representation. To minimize redundancy and reduce the computational cost during training, the
EFA module employs hard coding alignment between the aggregated multimodal representation
and its original visual and textual representations. These designs allow EFA to realize fine-grained
cross-modal interactions and assist the backbone network in learning more distinctive global
image-text representations without introducing additional supervision. FMFA is evaluated on three
public benchmarks [10, 27, 60], and it attains competitive top-level performance along with high
inference efficiency. We highlight our key contributions below:
— We introduce FMFA to explicitly leverage fine-grained interactions for improving cross-modal
alignment, without incurring extra supervision or inference overhead.
— We present an A-SDM module aimed at precisely aligning image-text pairs in a shared
embedding space. It adaptively adjusts to narrow the distance between mismatched positive
pairs, ensuring more precise matching.
— We develop an EFA module, which leverages the sparse similarity matrix for explicit aggregation and employs a hard coding method in cross-modal fine-grained alignment to minimize
redundant information.
2

Related Work

TIPR was initially proposed by Li et al. [27], who created the CUHK-PEDES dataset. Unlike visualbased person retrieval [7, 8, 16, 56], the core challenge of TIPR lies in constructing a shared latent
space that enables coherent alignment between visual and textual representations. Existing methods
can be typically classified into global and local matching approaches.
Early global methods [59, 60] directly aligned the global representations of images and text in
a joint embedding space. Schroff et al. [39] proposed a triplet ranking loss to enforce a margin
constraint between positive and negative pairs, and Zhang and Lu [58] introduced the CMPM/C
loss to minimize the discrepancy between the scalar projection of image-text pairs and their labels.
However, these global methods lack cross-modal fine-grained interactions, which restrict their
ability to capture detailed semantic correspondences. To address this limitation, early local matching
methods [12, 42, 49] explicitly aligned local visual and textual features to achieve fine-grained
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 5, Article 135. Publication date: April 2026.


exec
/bin/zsh -lc "pdftotext -l 3 'Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Cross-modal Full-mode Fine-grained Alignment for
Text-to-Image Person Retrieval
HAO YIN and XIN MAN, Shenzhen Institute for Advanced Study, University of Electronic Science and
Technology of China, China

FEIYU CHEN, JIE SHAO∗ , and HENG TAO SHEN, University of Electronic Science and Technology

arXiv:2509.13754v2 [cs.CV] 28 Dec 2025

of China, China Sichuan Artificial Intelligence Research Institute, China
Text-to-Image Person Retrieval (TIPR) is a cross-modal matching task designed to identify the person images
that best correspond to a given textual description. The key difficulty in TIPR is to realize robust correspondence
between the textual and visual modalities within a unified latent representation space. To address this
challenge, prior approaches incorporate attention mechanisms for implicit cross-modal local alignment.
However, they lack the ability to verify whether all local features are correctly aligned. Moreover, existing
methods tend to emphasize the utilization of hard negative samples during model optimization to strengthen
discrimination between positive and negative pairs, often neglecting incorrectly matched positive pairs. To
mitigate these problems, we propose FMFA, a cross-modal Full-Mode Fine-grained Alignment framework,
which enhances global matching through explicit fine-grained alignment and existing implicit relational
reasoning—hence the term “full-mode”—without introducing extra supervisory signals. In particular, we
propose an Adaptive Similarity Distribution Matching (A-SDM) module to rectify unmatched positive sample
pairs. A-SDM adaptively pulls the unmatched positive pairs closer in the joint embedding space, thereby
achieving more precise global alignment. Additionally, we introduce an Explicit Fine-grained Alignment
(EFA) module, which makes up for the lack of verification capability of implicit relational reasoning. EFA
strengthens explicit cross-modal fine-grained interactions by sparsifying the similarity matrix and employs a
hard coding method for local alignment. We evaluate our method on three public datasets, where it attains
state-of-the-art results among all global matching methods. The code for our method is publicly accessible at
https://github.com/yinhao1102/FMFA.
CCS Concepts: • Information systems → Image search; • Computing methodologies → Object identification.
Additional Key Words and Phrases: Cross-modal retrieval, Person search, Fine-grained alignment
ACM Reference Format:
Hao Yin, Xin Man, Feiyu Chen, Jie Shao, and Heng Tao Shen. 2025. Cross-modal Full-mode Fine-grained
Alignment for Text-to-Image Person Retrieval. ACM Trans. Multimedia Comput. Commun. Appl. 1, 1, Article 1
(January 2025), 22 pages. https://doi.org/10.1145/3786798
∗ Corresponding author.

Authors’ Contact Information: Hao Yin, yinhao1102@std.uestc.edu.cn; Xin Man, manxin@std.uestc.edu.cn, Shenzhen
Institute for Advanced Study, University of Electronic Science and Technology of China, Shenzhen, China; Feiyu Chen,
chenfeiyu@uestc.edu.cn; Jie Shao, shaojie@uestc.edu.cn; Heng Tao Shen, shenhengtao@hotmail.com, University of Electronic Science and Technology of China, Chengdu, China and Sichuan Artificial Intelligence Research Institute, Yibin,
China.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/1-ART1
https://doi.org/10.1145/3786798
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

1:2

1

Yin et al.

Introduction

Text-to-Image Person Retrieval (TIPR) seeks to understand natural language descriptions and
identify the most relevant person image within a large gallery [31]. Unlike general image-text
retrieval [5, 6, 44, 47, 50], which tends to achieve semantic-based matching between text and image,
TIPR is specifically designed for identifying individuals. TIPR requires the accurate modeling of
fine-grained correspondences between textual and visual modalities, owing to the large intra-class
variance and small inter-class difference. This substantial intra-class variation arises from two
aspects: (1) visual appearances of the same identity exhibit dramatic variations under different poses,
viewpoints, and illumination conditions, and (2) textual descriptions are influenced by differences
in phrasing, word order and textual ambiguities. Therefore, the primary challenges in TIPR are
how to extract discriminative global representations from image-text pairs and how to achieve
precise cross-modal fine-grained alignment. Existing methods for tackling these challenges can be
roughly divided into two main categories: global matching methods and local matching methods.
Some global matching methods [58, 59] obtain discriminative global representations by aligning
images and texts, which are projected into a joint embedding space. Their widely adopted loss
functions include the Cross-Modal Projection Matching (CMPM) loss [58] and the Similarity
Distribution Matching (SDM) loss [21]. The CMPM loss highlights the gap between the scalar
projections of image-text pairs and their matched label indicators. In comparison, the SDM loss
boosts global matching performance by minimizing the Kullback-Leibler (KL) divergence between
the normalized similarity profile of image-text pairs and the true label distribution. In addition,
the SDM loss incorporates a temperature hyperparameter to make model updates concentrate
on hard negative samples, yet it leads to the neglect of unmatched positive pairs, as shown in
Figure 1a. However, in TIPR, the accurate matching of positive pairs is prioritized over merely
distinguishing between positive and negative pairs. Meanwhile, some local matching methods
[2, 11, 35] incorporate attention mechanisms to achieve cross-modal fine-grained alignment. For
instance, RaSa [2] constructs a cross-modal encoder to generate multimodal representations for
subsequent fine-grained alignment. Building on RaSa, MARS [11] integrates a Masked AutoEncoder
(MAE) decoder [14] to reconstruct masked image patch sequences into their original unmasked
form, thereby facilitating cross-modal fine-grained alignment. However, these methods rely on
attention mechanisms to implicitly aggregate local image-text representations. As a result, they
yield only the final multimodal representation, without revealing the details of the aggregation
process. Consequently, these implicit aggregation methods make it difficult to determine whether
the aggregated multimodal representations correctly encode the corresponding visual and textual
information.
To remedy these concerns, we propose FMFA, a cross-modal Full-Mode Fine-grained Alignment framework, which enhances global matching through full-mode fine-grained alignment,
including explicit fine-grained image-text alignment and existing implicit relational reasoning.
Specifically, we design an Adaptive Similarity Distribution Matching (A-SDM) module to
ensure the correct matching of positive image-text pairs. Within the joint embedding space, the
A-SDM module adaptively pulls positive pairs closer together. In cases of mismatched positive pairs,
the A-SDM module adaptively regulates the pulling force based on their relative distance within
the joint embedding space, as shown in Figure 1b, thus improving cross-modal global alignment.
Based on the insight that each word in a caption can be associated with several image patches [3],
we introduce an Explicit Fine-Grained Alignment (EFA) module. The EFA module derives multimodal representations through explicit aggregation with a sparse similarity matrix. During this
process, the sparse similarity matrix between text and image reflects the contribution of textual and
visual representations to the final multimodal representation. To minimize redundancy and reduce

ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.

Cross-modal Full-mode Fine-grained Alignment for Text-to-Image Person Retrieval

1:3

Query text
Query text

Matched positive image
Matched positive image

Unmatched positive image

Unmatched positive image

Negative image
Negative image

Hard negative image
Hard negative image

Adaptive Pull

(a) Existing global matching methods.

(b) Our adaptive global matching method.

Fig. 1. Evolution of global matching methods for text-to-image person retrieval. (a) Existing global matching
methods focus on hard negative samples to learn a discriminative boundary in a common latent space, thus
enhancing the distinction between positive and negative samples. (b) Our adaptive global matching method
builds on a discriminative boundary and concentrates on unmatched positive samples, adaptively pulling
them closer to the corresponding query text.

the computational cost during training, the EFA module employs hard coding alignment between
the aggregated multimodal representation and its original visual and textual representations. These
designs allow EFA to realize fine-grained cross-modal interactions and assist the backbone network
in learning more distinctive global image-text representations without introducing additional
supervision. FMFA is evaluated on three public benchmarks [10, 27, 60], and it attains competitive
top-level performance along with high inference efficiency. We highlight our key contributions
below:
• We introduce FMFA to explicitly leverage fine-grained interactions for improving cross-modal
alignment, without incurring extra supervision or inference overhead.
• We present an adaptive similarity distribution matching module aimed at precisely aligning
image-text pairs in a shared embedding space. It adaptively adjusts to narrow the distance
between mismatched positive pairs, ensuring more precise matching.
• We develop an explicit fine-grained alignment module, which leverages the sparse similarity
matrix for explicit aggregation and employs a hard coding method in cross-modal fine-grained
alignment to minimize redundant information.
2

Related Work

Text-to-Image Person Retrieval (TIPR) was initially proposed by Li et al. [27], who created
the CUHK-PEDES dataset. Unlike visual-based person retrieval [7, 8, 16, 56], the core challenge of
TIPR lies in constructing a shared latent space that enables coherent alignment between visual and
textual representations. Existing methods can be typically classified into global and local matching
approaches.
Early global methods [59, 60] directly aligned the global representations of images and text in
a joint embedding space. Schroff et al. [39] proposed a triplet ranking loss to enforce a margin
constraint between positive and negative pairs, and Zhang et al. [58] introduced the CMPM/C loss
to minimize the discrepancy between the scalar projection of image-text pairs and their labels.
However, these global methods lack cross-modal fine-grained interactions, which restrict their
ability to capture detailed semantic correspondences. To address this limitation, early local matching
methods [12, 42, 49] explicitly aligned local visual and textual features to achieve fine-grained
cross-modal interactions. Nevertheless, they rely on unimodal pre-trained models (e.g., BERT [9]
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 1, No. 1, Article 1. Publication date: January 2025.


codex
我只读了每篇前三页。`Cross-Modal Full-Mode...` 的 ACM 版和 arXiv 版前三页基本是同一篇，按同一工作提炼，不重复虚增一个方向。

**逐篇提炼**
1. `Condense loss`
创新类型：新机制，训练损失层面重新解释特征范数。
为什么能发：抓住 ReID 很少专门研究分类损失的空白，把“特征范数随训练压缩”和精度变化联系起来，再用径向、切向梯度分析支撑。
story 套路：特征范数不是副产物，而是控制 ReID 训练稳定性的关键变量。

2. `Confidence guided semi-supervised cross-modality person re-identification`
创新类型：新数据设定加新机制。
为什么能发：VI-ReID 里监督成本高、无监督差距大，半监督设定正好卡在实际部署痛点上。证据链是少量标注引导跨模态伪标签置信度，再用动静态记忆库学习。
story 套路：只标一小部分身份，也能把大量未标注跨模态数据变成可用训练信号。

3. `Content and Salient Semantics Collaboration`
创新类型：偏新机制，但有工程组合味道。
为什么能发：换衣 ReID 依赖姿态、轮廓、衣服标签的问题很明显，它主打不加辅助信息，只从 RGB 高层特征里挖 content semantics 和 salient semantics。
story 套路：不靠外部模态，也能让模型从衣服外观里分离更稳的身份语义。

4. `Context-Aided Semantic-Aware Self-Alignment`
创新类型：新机制。
为什么能发：视频 ReID 的部位对齐常靠姿态或人体解析，但这些会丢掉背包、手提物等 ReID 有用线索。它用前景像素聚类自生成语义部位，再用视频上下文抑制噪声。
story 套路：别让外部人体拓扑定义“什么是有用部位”，让视频自己发现可对齐的身份语义。

5. `Corruption-Invariant Person Re-Identification`
创新类型：问题重定义加新机制。
为什么能发：把图像腐蚀从“数据增强噪声”重定义成“多模态分布偏移”，然后用干净教师粗对齐、桥接特征细对齐。
story 套路：腐蚀不是随机扰动，而是一组异质模态，直接硬对齐会丢信息，需要中间桥。

6. `Cross-domain person re-identification via learning Heterogeneous Pseudo Labels`
创新类型：新机制。
为什么能发：跨域 ReID 伪标签通常只有一种，但目标域同时有大类内差异和小类间差异。它用细粒度、粗粒度、实例三种伪标签共同描述样本，再用一致性约束和置信对比损失降噪。
story 套路：一个伪标签不足以描述目标域行人，样本需要多视角伪身份描述。

7. `Cross-modal Collaborative Representation Learning`
创新类型：新机制。
为什么能发：文本图像检索以前重关联，少讲互补。它把文本当成更干净的身份信息，用虚拟文本特征增强图像，并区分强正样本、弱正样本、负样本。
story 套路：文本不只是查询模态，也可以成为图像表征的内部教师，推理时不需要真实文本。

8. `Cross-Modal Full-Mode Fine-Grained Alignment`
创新类型：新机制。
为什么能发：文本图像检索里，隐式注意力很难验证局部是否真的对齐，常规全局损失又只盯难负样本，忽视没拉近的正样本。它用自适应正样本拉近和显式稀疏局部对齐补这个缺口。
story 套路：真正的跨模态对齐不只是分开负样本，还要修复没对上的正样本，并让局部对齐可检查。

**针对团队资产的强候选**
1. Headline：3D 可见性决定 ReID 特征范数，极端跨视角不应该用同一个分类边界训练所有样本。
挂靠资产：SMPL 3D 几何、pose 热图、SOLIDER-Swin、aerial-ground。
区别：最像 `Condense loss`，但它只讲通用特征范数压缩；我们把范数和 margin 绑定到 3D 可见身体面积、关节置信度、航拍视角下的可辨识性。
廉价否决实验：冻结 SOLIDER，在 CARGO 或 AG-ReID.v2 上提特征范数、SMPL 可见面积、pose 置信度，看它们是否显著解释 hard query 错误。如果相关性不强，或只训练一个可见性条件 margin 头没有提升 hard aerial split，就砍掉。

2. Headline：航拍到地面不是两域对齐，而是沿 3D 视角轨迹的桥接对齐问题。
挂靠资产：aerial-ground、SMPL mesh、SOLIDER-Swin。
区别：最像 `Corruption-Invariant ReID` 的桥接特征，但它桥接的是腐蚀模态；我们桥接的是可解释的相机俯仰、人体投影和尺度变化，机制上有 3D 几何路径，不是随机 latent bridge。
廉价否决实验：用已有 AG 数据按视角差分桶，检查同身份特征距离是否随 SMPL 投影差连续变化。再冻结主干训练一个小投影头，加几何桥接一致性。如果只对 easy pair 有效，对最大视角差无效，就停。

3. Headline：正样本也会错对齐，ReID 应该只拉近跨视角中互相可见的 3D 身体 token。
挂靠资产：SMPL 3D、pose 热图 gating、SOLIDER-Swin。
区别：最像 `FMFA`，但它解决文本和图像的显式局部对齐；我们解决图像 ReID 里“同身份正样本局部不可见或错位”的问题，用 SMPL 投影建立可验证的正样本 token 对齐矩阵。
廉价否决实验：不用训练，先把同身份跨视角 pair 的 Swin token 相似度按 SMPL 可见部位重算，和全局相似度比 hard positive 排名。如果可见部位分数不能明显提升 hard positive 排名，这条不值得做。

4. Headline：无标注 aerial-ground ReID 的伪标签不该只有身份簇，还要同时描述视角、可见性和实例可靠性。
挂靠资产：aerial-ground、SMPL、SOLIDER-Swin。
区别：最像 `HPL` 和半监督 VI-ReID，但它们是通用跨域或可见光红外；我们把异质伪标签定义成身份簇、3D 可见性簇、实例一致性三者，专门解决航拍低清和地面细节之间的伪标签污染。
廉价否决实验：在有标签 AG 数据上伪装无标签，比较普通 DBSCAN 和几何异质伪标签的簇纯度、跨视角正样本召回。如果伪标签纯度不涨，后续训练基本不用开。
tokens used
87,398
我只读了每篇前三页。`Cross-Modal Full-Mode...` 的 ACM 版和 arXiv 版前三页基本是同一篇，按同一工作提炼，不重复虚增一个方向。

**逐篇提炼**
1. `Condense loss`
创新类型：新机制，训练损失层面重新解释特征范数。
为什么能发：抓住 ReID 很少专门研究分类损失的空白，把“特征范数随训练压缩”和精度变化联系起来，再用径向、切向梯度分析支撑。
story 套路：特征范数不是副产物，而是控制 ReID 训练稳定性的关键变量。

2. `Confidence guided semi-supervised cross-modality person re-identification`
创新类型：新数据设定加新机制。
为什么能发：VI-ReID 里监督成本高、无监督差距大，半监督设定正好卡在实际部署痛点上。证据链是少量标注引导跨模态伪标签置信度，再用动静态记忆库学习。
story 套路：只标一小部分身份，也能把大量未标注跨模态数据变成可用训练信号。

3. `Content and Salient Semantics Collaboration`
创新类型：偏新机制，但有工程组合味道。
为什么能发：换衣 ReID 依赖姿态、轮廓、衣服标签的问题很明显，它主打不加辅助信息，只从 RGB 高层特征里挖 content semantics 和 salient semantics。
story 套路：不靠外部模态，也能让模型从衣服外观里分离更稳的身份语义。

4. `Context-Aided Semantic-Aware Self-Alignment`
创新类型：新机制。
为什么能发：视频 ReID 的部位对齐常靠姿态或人体解析，但这些会丢掉背包、手提物等 ReID 有用线索。它用前景像素聚类自生成语义部位，再用视频上下文抑制噪声。
story 套路：别让外部人体拓扑定义“什么是有用部位”，让视频自己发现可对齐的身份语义。

5. `Corruption-Invariant Person Re-Identification`
创新类型：问题重定义加新机制。
为什么能发：把图像腐蚀从“数据增强噪声”重定义成“多模态分布偏移”，然后用干净教师粗对齐、桥接特征细对齐。
story 套路：腐蚀不是随机扰动，而是一组异质模态，直接硬对齐会丢信息，需要中间桥。

6. `Cross-domain person re-identification via learning Heterogeneous Pseudo Labels`
创新类型：新机制。
为什么能发：跨域 ReID 伪标签通常只有一种，但目标域同时有大类内差异和小类间差异。它用细粒度、粗粒度、实例三种伪标签共同描述样本，再用一致性约束和置信对比损失降噪。
story 套路：一个伪标签不足以描述目标域行人，样本需要多视角伪身份描述。

7. `Cross-modal Collaborative Representation Learning`
创新类型：新机制。
为什么能发：文本图像检索以前重关联，少讲互补。它把文本当成更干净的身份信息，用虚拟文本特征增强图像，并区分强正样本、弱正样本、负样本。
story 套路：文本不只是查询模态，也可以成为图像表征的内部教师，推理时不需要真实文本。

8. `Cross-Modal Full-Mode Fine-Grained Alignment`
创新类型：新机制。
为什么能发：文本图像检索里，隐式注意力很难验证局部是否真的对齐，常规全局损失又只盯难负样本，忽视没拉近的正样本。它用自适应正样本拉近和显式稀疏局部对齐补这个缺口。
story 套路：真正的跨模态对齐不只是分开负样本，还要修复没对上的正样本，并让局部对齐可检查。

**针对团队资产的强候选**
1. Headline：3D 可见性决定 ReID 特征范数，极端跨视角不应该用同一个分类边界训练所有样本。
挂靠资产：SMPL 3D 几何、pose 热图、SOLIDER-Swin、aerial-ground。
区别：最像 `Condense loss`，但它只讲通用特征范数压缩；我们把范数和 margin 绑定到 3D 可见身体面积、关节置信度、航拍视角下的可辨识性。
廉价否决实验：冻结 SOLIDER，在 CARGO 或 AG-ReID.v2 上提特征范数、SMPL 可见面积、pose 置信度，看它们是否显著解释 hard query 错误。如果相关性不强，或只训练一个可见性条件 margin 头没有提升 hard aerial split，就砍掉。

2. Headline：航拍到地面不是两域对齐，而是沿 3D 视角轨迹的桥接对齐问题。
挂靠资产：aerial-ground、SMPL mesh、SOLIDER-Swin。
区别：最像 `Corruption-Invariant ReID` 的桥接特征，但它桥接的是腐蚀模态；我们桥接的是可解释的相机俯仰、人体投影和尺度变化，机制上有 3D 几何路径，不是随机 latent bridge。
廉价否决实验：用已有 AG 数据按视角差分桶，检查同身份特征距离是否随 SMPL 投影差连续变化。再冻结主干训练一个小投影头，加几何桥接一致性。如果只对 easy pair 有效，对最大视角差无效，就停。

3. Headline：正样本也会错对齐，ReID 应该只拉近跨视角中互相可见的 3D 身体 token。
挂靠资产：SMPL 3D、pose 热图 gating、SOLIDER-Swin。
区别：最像 `FMFA`，但它解决文本和图像的显式局部对齐；我们解决图像 ReID 里“同身份正样本局部不可见或错位”的问题，用 SMPL 投影建立可验证的正样本 token 对齐矩阵。
廉价否决实验：不用训练，先把同身份跨视角 pair 的 Swin token 相似度按 SMPL 可见部位重算，和全局相似度比 hard positive 排名。如果可见部位分数不能明显提升 hard positive 排名，这条不值得做。

4. Headline：无标注 aerial-ground ReID 的伪标签不该只有身份簇，还要同时描述视角、可见性和实例可靠性。
挂靠资产：aerial-ground、SMPL、SOLIDER-Swin。
区别：最像 `HPL` 和半监督 VI-ReID，但它们是通用跨域或可见光红外；我们把异质伪标签定义成身份簇、3D 可见性簇、实例一致性三者，专门解决航拍低清和地面细节之间的伪标签污染。
廉价否决实验：在有标签 AG 数据上伪装无标签，比较普通 DBSCAN 和几何异质伪标签的簇纯度、跨视角正样本召回。如果伪标签纯度不涨，后续训练基本不用开。
