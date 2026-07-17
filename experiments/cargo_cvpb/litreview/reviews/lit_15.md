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
session id: 019ef71f-b6de-7661-9fc5-f194937a57e8
--------
user
你是 ReID 论文创新挖掘员 15/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(15/20):
Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf
Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf
Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf
Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf
Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf
RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf
RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我先按项目规则确认当前目标和状态，再只读每篇 PDF 前三页。后面会把论文总结压缩到能反推出创新方向的粒度，不做泛泛综述。
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
/bin/zsh -lc pwd in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title

exec
/bin/zsh -lc "rg --files -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf
A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf
Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf
Spectrum-guided feature enhancement network for event person re-identification.pdf
Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf
Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf
Bidirectional modality information interaction for Visible-Infrared Person Re-identification.pdf
Pose-Skeleton Guided Cross-Attention Representation Fusion for Occluded Pedestrian Re-Identification.pdf
Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf
Harnessing Knowledge From Pretrained VLMs for Unsupervised Person Search.pdf
Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification.pdf
False Negatives Consensus Suppression for Text-to-Image Person Re-identification.pdf
Exploring Part-Informed Visual-Language Learning for Person Re-Identification.pdf
Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf
Generalizable person re-identification method using bi-stream interactive learning with feature reconstruction.pdf
Unsupervised person re-identification via camera-aware multi-level label refinement.pdf
Distribution aligned semantics adaption for lifelong person re-identification.pdf
TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf
DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf
Visible-Infrared Person Re-Identification via Mutual Reinforcement of Prompts and Image Encoders.pdf
Diverse Representations Embedding for Lifelong Person Re-Identification.pdf
Adversarial perturbation and defense for generalizable person re-identification.pdf
RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf
Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf
Structured Distilled 3D Gait Fields for Clothes-Changing Person Re-Identification in Videos.pdf
Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf
GAE-Net - A gait-assisted enhancement network for video-based person re-identification.pdf
Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf
Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf
Heterogeneous Generative Tokens and Distance-Aware Recovery Network for Occluded Person Re-Identification.pdf
InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf
TP-LReID - Lifelong person re-identification using text prompts.pdf
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf
Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf
Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf
Visible-Infrared Person Re-Identification With Real-World Label Noise.pdf
Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf
Context-Aided Semantic-Aware Self-Alignment for Video-Based Person Re-Identification.pdf
Categorical Attention - Fine-grained Language-guided Noise Filtering Network for Occluded Person Re-Identification.pdf
CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM acm_browser_subscription].pdf
Multi-year long-term person re-identification using gait and HAR features.pdf
Dynamic Modality-Camera-Invariant Clustering for Unsupervised Visible-Infrared Person Re-Identification.pdf
Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf
Corruption-Invariant Person Re-Identification via Coarse-to-Fine Feature Alignment.pdf
Multi Queue for Unsupervised Person Re-identification.pdf
Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf
Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf
Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf
Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf
Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf
Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf
Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf
Occlusion-aware Cross-modality Completion Network for Occluded Visible-Infrared Person Re-Identification.pdf
Texture-aware transformer with pose-patch mapping for occluded person re-identification.pdf
Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf
Semantic Consistency And Integrity Network For Cloth-changing Person Re-identification.pdf
Meta Pairwise Relationship Distillation for Unsupervised Person Re-Identification.pdf
Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf
A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
Two-stage knowledge distillation for visible-infrared person re-identification.pdf
Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf
ADA framework for unsupervised domain adaptation person re-identification.pdf
Optimal Illumination Distance Metrics for Person Re-Identification in Complex Lighting Conditions.pdf
Local-Aware Residual Attention Vision Transformer for Visible-Infrared Person Re-Identification.pdf
Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf
Confidence guided semi-supervised cross-modality person re-identification.pdf
Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf
Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf
Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf
Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf
CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf
Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf
Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf
GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf
CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf
HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf
MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf
Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf
Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf
Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
Internal-External Context Interaction Network for Person Re-Identification.pdf
Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf
Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
Shape-centered representation learning for visible-infrared person re-identification.pdf
Multi-feature balanced network for clothes-changing person re-identification.pdf
Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf
Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf
GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf
Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification.pdf
ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf
Deep intelligent technique for person Re-identification system in surveillance images.pdf
Cross-modal Collaborative Representation Learning for Text-to-Image Person Retrieval.pdf
FDGReID - Federated Domain Generalization for Person Re-identification.pdf
Identity-aware infrared person image generation and re-identification via controllable diffusion model.pdf
HOH-Net - High-Order Hierarchical Middle-Feature Learning Network for Visible-Infrared Person Re-Identification.pdf
A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf
FMCNet+ - Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification.pdf
Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf
Focusing on pedestrians like human for clothes changing person re-identification.pdf
Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf
CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf
CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf
Instant pose extraction based on mask transformer for occluded person re-identification.pdf
FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf
Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
Text-to-image Person Search based on Semantic Reorganization.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf
Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
CLIP-driven fine-grained mining for text-based person search.pdf
'Knowledge and experience' for visible-infrared person re-identification.pdf
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
Cross-modality average precision optimization for visible thermal person re-identification.pdf
Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
Occluded person Re-Identification with noise injection.pdf
Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
Privacy preserving person re-identification via anonymizing diffusion model.pdf
RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf
Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf
Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf
When Large Vision-Language Models Meet Person Re-Identification.pdf
DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf
SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf
CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf
A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf
Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf
Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf

exec
/bin/zsh -lc "rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'AGENTS.md' -g 'STATUS.md' /Users/abdslm/Desktop/reid-paper" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/CLAUDE.md
/Users/abdslm/Desktop/reid-paper/AGENTS.md
/Users/abdslm/Desktop/reid-paper/GOAL.md
/Users/abdslm/Desktop/reid-paper/experiments/STATUS.md

codex
PDF 列表已经确认，目标批次文件都在目录里。我先读取项目目标文件，避免和当前 ReID 主线或实验纪律冲突。
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
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/Desktop/reid-paper/GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,260p' /Users/abdslm/Desktop/reid-paper/CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,260p' /Users/abdslm/Desktop/reid-paper/AGENTS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
# AGENTS.md · ReID 工作区（Codex 接手版）

本文是 Codex 会自动加载的项目规则。真正来源仍然是 `CLAUDE.md` 和 `.claude/rules/`，如果本文和它们冲突，以 `CLAUDE.md` 为准。因为 Codex 默认只自动读 `AGENTS.md`，所以这里把 Claude 那套最重要的规则展开写清楚，方便 Claude 没有额度时由 Codex 接手。

`.claude/rules/` 是给人和 agent 看的 Markdown 规则。Codex 的 `.codex/rules/` 是命令权限规则，格式不同，不要把这些 Markdown 文件直接放进去。当前项目已经有 `.codex/hooks.json` 和 `.codex/hooks/`，用于在 Codex 执行命令前做训练审查和监控检查。

## 每次接手先做什么

每次接手、每个大步骤开始之前，都先读 `GOAL.md`。这是当前目标的唯一来源，由用户来写。用户可能直接改这个文件来调整方向，所以不要只看对话历史。

接着读 `CLAUDE.md`、本文件、`experiments/STATUS.md`。如果要启动实验、审查实验、改训练代码、解析结果，继续读相关的 `.claude/rules/*.md`。其中最重要的是 `result_discipline.md` 和 `experiment_protocol.md`。

如果 `GOAL.md` 的主目标被清空，或者写成暂停，就停下来等用户。不要自己找新方向。

## 你在这个项目里的角色

第一，做训练前两轮独立审查里的 Codex 这一轮。一个改了方法的实验，在开始训练之前，主 agent 会用 `codex exec` 起你来做一次独立代码审查。你看不到另一个审查者的结论，也不知道这是第几轮，也不要假设自己知道改了什么。审查范围是设计文档、新增和改动代码、配置、对照组变量隔离、评测协议和数据流。

审查只拦实质问题：代码正确性的错误、数据泄漏、评测协议前后不一致、变量没隔离干净、比较不公平。操作上、文档上的小问题，可以记成待办，但不要拦。结论写进 `experiments/expNNN/review-codex.md`。通过时要明确写“无实质问题”“approve”“审查通过”或“放行”。

第二，做独立讨论和探索。Claude 额度紧张时，能独立完成的子任务可以交给你，比如讨论方向、读代码、查事实、解析日志、检查实验记录、做差距分析。

## 工作区是什么

这是一个新的 ReID 研究工作区，目标是做一篇 CCF-B 级别、真正新颖、有效、又能讲清楚的行人重识别工作。具体子方向由调研和实验判断，不要一开始就把方向写死。

`SOLIDER-REID/` 是干净的上游代码，来源是 `github.com/tinyvision/SOLIDER-REID`，版本是 `8c08e1c`。它只是底子，提供 SOLIDER 预训练权重和 Swin 主干，本身不是创新点。所有方法代码都在这个目录里从零写。

`experiments/` 是实验记录，从 `exp000` 开始编号。`.claude/rules/` 是详细规则。`.claude/hooks/` 和 `.codex/hooks/` 是用来强制执行纪律的检查脚本。

## 铁律

数字只认日志。所有指标都要用代码从日志文件里解析出来，不能凭记忆、凭印象写，也不要手抄。

凡是要写进论文的结论，都要把 seed 0、1、2 三个随机种子都跑一遍，报告均值和标准差。

rank1 的差异小于 0.5、mAP 的差异小于 0.4，就算落在正常波动范围里，不能算作成果。

永远不要挑随机种子，也不要挑表现最好的那个 epoch。一律上报最后一个 epoch 的结果，不要用 `best_model`。

评测口径是冻结的。要改评测口径，必须先问用户。

每涨一次点，都要换一个挑刺的角度重新核对一遍，看它是不是噪声、是不是数据泄漏、是不是评测口径前后不一致。

正式训练之前，先用很小的规模快跑一遍，确认不会崩，模块确实在起作用。

做好实验记录。`experiments/decisions.md`、`experiments/results.md`、每个实验的 `monitor.md` 都要及时更新。同样的配置加同样的种子，不要重复跑。

## 三条研究纪律

判定一个方向走死之前，要先定好标准，并且有足够证据。开始跑之前就把“什么样的结果才算这条路走死了”写进 `design.md`。只有一两个负结果时，只能写“还需要再试”，不能判定整条方向死掉。

自己写的评测脚本或分析脚本，要先用它复现一个已知的基线成绩。对得上之后，才能用它的结果下结论。新的评测口径、新的度量、新的脚本都按这条执行。

“贡献”是个有门槛的词。一个结果只有同时满足下面几条，才能叫贡献，才能说可以投稿：满足对新方向的要求；跑了三个种子并报告均值和标准差；涨幅超过正常波动范围；和最接近的已有工作区分清楚。在那之前，只能叫“信号”或“探索”。

## 一个新方向值不值得做

ReID 是活跃领域，不要一上来就觉得能做的都被做完了。先读论文、做差距分析，再决定方向。

一个新方向至少要满足下面三条里的两条，否则不作为主线。

1. 问题上有新意。不是加一个模块，而是重新定义或者更准确地刻画一个真实存在的问题。
2. 机制上有新意。是过去工作没有清楚写出来，而且代码上能实现的机制。
3. 证据上讲得清。能设计出干净的对照和消融，能回答它为什么有效。

方向必须和最接近的已有工作区分清楚。方向定下来之前，要和 Codex 或子代理讨论核实，确认它确实是新的。不能拿测试时的小技巧当主要贡献，比如重排序、特征归一化、翻转测试。不能用“比基线高了零点几”来定义创新。

## 实验命名和目录

实验目录叫 `exp{编号}_{简短描述}`，例如 `exp000_baseline`、`exp012_new_method`。

同一个实验的不同变体用字母后缀区分，例如 `exp012a`、`exp012b`、`exp012c`。判断标准是：核心方法相同，只是配置、种子、环境或超参不同，就属于同一个实验的变体。所有变体共用一个 `experiments/exp{编号}/` 目录。

每个实验用独立的 `OUTPUT_DIR`，例如 `./log/<数据集>/exp{编号}`。

训练命令一般使用这个形式：

```bash
python train.py --config_file xxx.yml \
  SOLVER.SEED NN \
  TEST.IMS_PER_BATCH 64 \
  OUTPUT_DIR ./log/.../expNNN \
  <其它覆盖项>
```

`TEST.IMS_PER_BATCH 64` 建议都加上，因为测试集大、又开了翻转测试时，默认 256 容易把显存撑爆。后台跑用 `setsid nohup python train.py ... </dev/null > /path/uniq.log 2>&1 &`。日志文件名必须唯一，不要互相覆盖。

## design.md 格式

开始训练前必须写 `experiments/exp{编号}/design.md`。格式如下：

```markdown
# 实验 exp{编号}: {名称}

## 动机
为什么做？解决什么问题？基于前面哪些实验或者论文？

## 核心假设
一句话说清楚。

## 技术方案
改了哪些文件？加了哪些模块？数据从输入到输出怎么走的？关键超参怎么定的？

## 对照组
和哪个基线比？只改了哪一个变量？

## 什么算走死
什么样的结果算“还需要再试”，什么样的结果算“这条路走死了”。

## 预期结果和失败解释
假设成立时，mAP 和 rank1 大概会怎么变；如果失败，最可能的原因是什么。

## 需要训练前审查
需要训练前审查：是
```

改了方法的实验填“需要训练前审查：是”。纯复现实验，也就是只改随机种子的，把这行改成“需要训练前审查：否”，检查脚本会放行。

## 训练前独立审查

任何改了模型或者有新设计的实验，在启动训练之前，都要经过两轮互不通气的独立审查。一轮由 Claude 做，一轮由 Codex 做。两个审查者互相看不到对方结论，也不知道这是第几轮。

Codex 这一轮要完整审一遍，不是只看某几处。要看设计文档、新增和改动的代码、配置、对照组、变量隔离和评测协议。结论写进 `experiments/exp{编号}/review-codex.md`。

实质问题修好之后，必须再审一轮。某一轮里两个审查都没有实质问题，才算放行。

检查脚本 `.codex/hooks/check_design.sh` 会在包含 `train.py` 的命令执行前检查：设计文档在不在，`review-claude.md` 是不是通过且至少三十行，`review-codex.md` 是不是通过。如果 `design.md` 里写了“需要训练前审查：否”，就只检查设计文档。

## 代码原则

新模块要插件式实现，放在 `model/` 下，用配置开关控制。默认配置必须能复现基线。

每次只改一个核心变量。可以是一个模块、一个损失、一种 pooling、一个训练机制。如果要组合几个东西，必须写清楚组合了哪些已经验证过的模块，以及为什么现在适合组合。

配置、随机种子、commit 号或代码状态都要记下来。这个工作区当前顶层不一定是 Git 仓库，所以如果没有 commit 号，就记录文件状态、命令和关键改动。

不要用有破坏性的 git 命令，不要覆盖用户已经做的改动。

## 文档纪律

没有文档的实验，等于没做过。

每个实验从头到尾要维护这些文件：`experiments/exp{编号}/design.md`、`monitor.md`、需要时的 `review-claude.md` 和 `review-codex.md`。总记录放在 `experiments/results.md`、`experiments/decisions.md`、`experiments/STATUS.md`。

每次看日志，都要更新 `monitor.md`。至少写当前到第几个 epoch、进度如何、关键损失值、评测指标，以及一句判断：继续、盯着，还是停掉，并说明原因。

同一个实验在不同文档里的数字必须完全一致。新结果推翻旧判断时，直接改文档里的措辞，不要只在对话里说。

实际实现和 `design.md` 不一致的地方，在 `monitor.md` 里写清楚。一个实验结束后，先把文档补完整，再开下一个。

论文素材要一直维护在 `experiments/paper_materials/`。`story.md` 记录当前方法主线、候选贡献、支撑实验和推翻旧说法的结果。表格和图放在 `tables/` 和 `figures/`。

## 决策记录

重大决策追加到 `experiments/decisions.md`，格式如下：

```markdown
### [年-月-日 时:分] 决策 #编号
上下文：在什么情况下做的这个决策
选项：A. 方案和预期；B. 方案和预期
选择：A 还是 B
理由：为什么
执行结果：（后面补）
```

重大决策前，尽量让两个独立视角辩一辩。一个为方案 A 辩护，一个为方案 B 辩护，从技术可行性、创新性、论文价值、风险、成本几方面讲，并给出信心分。最后综合判断，把结论写进决策记录。

自己试过并判定为负的方向，要记进 `experiments/decisions.md`，免得以后重复跑。要重新走一个已经判负的方向，先写清楚为什么这次不一样。判负必须有足够证据，不能用一两个负结果判整条方向。

## 监控和等待

不要用长 `sleep` 反复读日志。优先用 Monitor 或后台 Bash 等待器。启动长任务后，用后台方式运行，并让完成事件或完成标志来唤醒后续检查。

如果必须执行 180 秒以上的 `sleep`，先更新某个 `experiments/exp*/monitor.md`。`.codex/hooks/check_monitor.sh` 会检查这一点。

第 1 到 5 个 epoch 要勤看，确认不会崩，模块在起作用。第 6 到 30 个 epoch 中等频率。30 个 epoch 以后可以看得稀一点。每次看完都更新 `monitor.md`。

出现 NaN 或 Inf，立刻停，先查原因，再决定是降学习率还是回退。显存溢出时，先减小模块复杂度，或者把 `TEST.IMS_PER_BATCH` 调小，不要随便改训练 batch size。长时间没进展时，先把证据写下来，再决定要不要停。

DataLoader 的子进程也会显示成 `python train.py`，不要把它误当成重复训练进程。停训练时只杀主进程。

## 机器和网络

你在 Mac 上跑，能联网。服务器只有国内网，装包用清华源 `https://pypi.tuna.tsinghua.edu.cn/simple`，下模型用 `https://hf-mirror.com`。统一用 `~/.ssh/config` 里的别名连接，不要用 `sshpass`。

三台 GPU：

1. `hyy-5060ti-double`：恒源云，两块 5060Ti 16G，用户 root，直连。`/hy-tmp` 只有 50G。训练命令里用 `--gpu 0` 或 `--gpu 1` 选卡。每次训练只保留最后一个 epoch 的 checkpoint，中间的和 `best_model` 都删掉。
2. `lab-3090-d`：实验室 RTX3090 24G，单卡，在 docker 容器 `abdslm-common` 里，经 `lab-3090` 跳板连。容器一重启就会丢掉 sshd 和 IP，要在主机上执行 `docker exec abdslm-common /usr/sbin/sshd` 重新启动 sshd。IP 变了就更新 `~/.ssh/config`。
3. `lab-4090`：实验室 RTX4090D 24G，单卡，共享机器。只能用 `afr` 自己的空间，数据放 `/mnt1/afrdata`，代码、日志、虚拟环境放 `/home/afr/` 下的项目目录。绝对不要碰 `/root`、`/hy-tmp` 和共享 conda。

传大文件用 `rsync -az --partial`，放后台跑，再监控日志。本地到远程的带宽大概 2 MB/s。

如果一条远程命令反复失败，但单独拿出来测又正常，先怀疑日志没刷新，或者读到的文件不是最新的，不要急着改代码。

绝对不要用宽泛匹配的 `pkill -f codex`，那会误杀 Mac 上的 Codex.app。要杀进程就精确匹配。

## SOLIDER-REID 代码底子

上游只带了 Market-1501 和 MSMT17 的配置，主干有 Swin Tiny、Small、Base。要用别的 benchmark，比如 Occluded-Duke、Occluded-ReID，需要自己加配置和数据集读取代码。

用 SOLIDER 的预训练权重。从 SOLIDER 仓库的 Google Drive 下载，用 `convert_model.py` 转换 teacher checkpoint。Mac 能连 Google Drive。

`SEMANTIC_WEIGHT` 默认是 0.2。

主干顺序是先用 Swin-Tiny 快速验证。有信号了再放大到 Small、Base。创新点先在 Tiny 上看有没有效果，确实有效再放大，不要在没效果的点上用大主干硬磨。

## Python 和环境

如果要用 Python，使用 `uv` 在工作目录下创建虚拟环境，不要污染系统 Python。简单读取文本优先用 `rg`、`sed`、`awk`、`perl` 或项目已有脚本。需要装包或运行较复杂 Python 脚本时，先在项目里用 `uv venv` 和 `uv pip` 管理环境。

服务器上也不要动共享 conda。`lab-4090` 尤其要注意，虚拟环境放在 `/home/afr/` 下自己的项目目录。

## 写中文

所有给人看的文字，包括文档、注释、提交信息和给用户的回复，都用平实、完整的中文句子。

不要把判断标准、检查步骤生造成奇怪的简称。不要用箭头把几个词串起来代替句子。不要用破折号来连接或停顿。不要在中文里夹不必要的英文术语。真正的专有名词、命令、文件名、配置字段可以保留原文。

不要堆砌小标题、加粗和表格来充字数。能用一段话讲清楚就用一段话。

不要用比喻性的网络口头禅。思考和写作都尽量用完整、清楚的中文句子。

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/Desktop/reid-paper/experiments/STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
项目目标是只做方向调研，不开实验。现在开始按你的限制只抽每篇前三页，我会重点看它们怎么定义问题、怎么证明 gap，而不是照着方法细节复述。
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
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -l 3 'Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


codex
前四篇里已经能看到一个共同倾向：好论文通常不是只说“对齐更细”，而是把旧问题改写成“点表示不够、原型不够、压缩收益归因不清”这类可检验的结构问题。继续读剩下五篇。
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
/bin/zsh -lc "pdftotext -l 3 'See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'RMGNet - The Progressive Relationship-Mining Graph Neural Network for Text-to-Image Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025

5749

RMGNet: The Progressive Relationship-Mining
Graph Neural Network for Text-to-Image Person
Re-Identification
Xin Zhang , Kun Liu , Xinwang Wang, Zhong Zhou , Member, IEEE, and Haiyong Chen
Abstract— The
Text-to-Image
Person
Re-identification
(TI-ReID) task objective is to precisely identify the person’s
images with the textual description of the person. The
mainstream research methods focus on cross-modal aligning
local features, and overlook the learning of intra-modal and
cross-modal relationships between different features. This
renders the person features lacking in high-level semantic
information. To resolve such issues, we propose the Progressive
Relationship-Mining Graph Network (RMGNet), including the
Intra-Modal Relationship-Mining (IMRM) and the Cross-Modal
Relationship-Mining (CMRM) module. These modules are
employed to model and mine semantic relationship information
among different features. Specifically, the IMRM module
models and mines the high-level semantic interrelationships
inherent in the image and text features. The CMRM module
introduces the nearest neighbor method to model cross-modal
semantic relationships to enhance the cross-modal semantic
correspondence capabilities of person features. On this basis,
we design the Adaptive Corner Center (Acc) loss and the Coarseto-Fine Learning (C2FL) strategy. These ensure the network
receives consistent and effective metric learning supervision
throughout the entirety of the training process. To validate
the efficacy of the proposed method, extensive experiments are
conducted on three prevalent datasets: CHUK-PEDES, ICFCPEDES, and RSTPReid. The achieved mAP of 70.59%, 41.62%,
and 49.58% surpassed those current state-of-the-art methods.
Index Terms— Person re-identification, multi-modal, textto-image retrieval, relationship-mining graph, graph neural
network.

I. I NTRODUCTION

F

OR the research of Intelligent Transportation Systems,
Text-to-Image Person Re-identification [1], [2] has broad

Fig. 1.
Comparison of different matching strategies. (a) The prevalent
local-based matching strategy enhances feature expressiveness by learning
and aligning discriminative local features in images and texts. (b) Our
relationship-based matching strategy focuses on modeling and mining relationships between different features to further enhance discriminative capabilities
and distinguish persons with similar appearances.

application prospects. It can achieve the recognition of different person utilizing various devices and algorithms without
images of the target person. This facilitates the in-depth
integration and application of computer vision technology in
object tracking [3], [4], [5], [6], [7], action recognition [8], [9]
and autonomous driving [10], [11], [12]. Due to the significant
modal gap, the primary challenge lies in the efficient extraction
of cross-modal discriminative features in both person images
and text. This necessitates the exploration of their hidden
semantic correspondence relationship.
In recent years, research on TI-ReID has advanced, with
existing methods primarily following two strategies: global
matching [13], [14], [15] and local matching [16], [17], [18].
The global matching methods map person image and text
features into a unified space to reduce modal disparities
interference, same as the image-text retrieval methods [19],
[20]. However, relying solely on global features for matching
the person text and image may lead to the omission of
crucial discriminative local features. This makes it difficult to
accurately identify different person with similar appearances.
While local matching methods focus on mining salient regions
within images and discriminative words in the text, enabling
fine-grained matching between person images and textual
descriptions, as illustrated in Figure 1(a). Despite the similarity
in appearance between the two people depicted in the image,

Received 22 August 2024; revised 12 November 2024; accepted 18 January
2025. Date of publication 22 January 2025; date of current version 6 June
2025. This work was supported in part by the National Key Research
and Development Program of China under Grant 2022YFB33038004; in
part by the National Natural Science Foundation of China under Grant
U21A20482, Grant 62073117, Grant 62272018, and Grant 62173124; and
in part by the Science and Technology Project of Hainan Provincial Department of Transportation, under Grant HNJTT-KXC-2024-3-22-02. This article
was recommended by Associate Editor Z. Qian. (Corresponding author:
Haiyong Chen.)
Xin Zhang, Kun Liu, and Haiyong Chen are with the School of
Artificial Intelligence, Hebei University of Technology, Tianjin 300130,
China (e-mail: zhangxin8275@buaa.edu.cn; KunLiu@hebut.edu.cn; haiyong.
chen@hebut.edu.cn).
Xinwang Wang is with the School of Instrument Science and Engineering,
Southeast University, Nanjing, Jiangsu 210018, China, and also with the
School of Integrated Circuit, Wuxi Institute of Technology, Wuxi 214121,
China (e-mail: 230189684@seu.edu.cn).
Zhong Zhou is with the Zhongguancun Laboratory, Beijing 100086, China,
and also with the State Key Laboratory of Virtual Reality Technology and
Systems, School of Computer Science and Engineering, Beihang University,
Beijing 100191, China (e-mail: zz@buaa.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3532685
1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
SeeTIANJIN
https://www.ieee.org/publications/rights/index.html
more information.
Authorized licensed use limited to:
UNIVERSITY. Downloaded on June 09,2026 atfor
09:00:38
UTC from IEEE Xplore. Restrictions apply.

5750

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 6, JUNE 2025

accurate discrimination is still achievable through distinct local
features, such as differences in shoe color and trouser type.
While the aforementioned methods have achieved some
progress, the actual recognition results remain suboptimal.
These methods focus on mining highly discriminative local
features and achieve cross-modal alignment. The learning
of intrinsic semantic connections and interactions between
features is overlooked. Specifically, the variations in feature
relationships will lead to changes in semantic meaning, consequently altering the correspondence between images and
text. These methods ignore fully mining the complex and
varied semantic relationships between different features. This
defect results in errors in recognition outcomes, as depicted
in Figure 1(b). The two pedestrian images and their corresponding description texts are basically the same, with
nearly identical local features. However, the relationship
between the backpack and the pedestrian is different in the
image and the textual description. The textual description of
the image on the left reads “person carrying a bag”, while
the right image is “person wearing a bag”. It can be seen
that through the modeling and mining of feature relationships, the discriminative capability of features can be further
enhanced.
To effectively model and mine the relationships between
different features, we propose the Progressive Relationship Mining Graph Neural Network (RMGNet). It enhances
feature expression capabilities by learning the interrelationship between different features within intra-modal and
inter-modal. In the RMGNet, the Intra-Modal RelationshipMining (IMRM) module encodes high-level semantic concepts
associated with local features of images and texts. This
optimizes features by aggregating semantic contextual information and latent interrelationships between features. The
Cross-Modal Relationship-Mining (CMRM) module models
the cross-modal feature semantic interrelationship by fusing
the GNN and the nearest neighbor strategy. Utilizing the
powerful relational reasoning capabilities of GNN to learn the
semantic relationships between different modal features and
aggregation enhancement. In addition, we convert the imagetext cross-modal matching task into the binary classification
task. The classification probability output by the network
serves as auxiliary discriminant information to improve the
accuracy of TI-ReID.
Furthermore, the hetero-center triplet (Hc-Tri) loss [21],
tends to easily satisfy the relative distance constraint early
in the training process and converges quickly. Consequently,
there is a lack of supervision in the later stages of network
training. Therefore, we propose the coarse-to-fine learning
(C2FL) strategy and the novel adaptive corner center (Acc)
loss to train the network.
The proposed method improves the recognition accuracy
with the following four contributions:
• We propose the Progressive Relationship Mining Graph
Neural Network (RMGNet), which is used to model
and mine the hidden inter-relationship between features
within the intra-modal and inter-modal. The network is
the first to apply the GNN to learn the mutual semantic
relationships between features in the TI-ReID task.

• We design the Intra-Modal Relationship-Mining (IMRM)
module, which is used to model and mine the hidden
fine-grained semantic relationships between different features within the intra-modal.
• We design the Cross-Modal Relationship-Mining
(CMRM) module, which is employed to model and learn
the semantic correlation and affinity relationship between
person features within the inter-modal by introducing
the nearest neighbor strategy.
• The new Coarse-to-Fine Learning (C2FL) strategy and
Adaptive Corner Center (Acc) loss are proposed to enable
the network to receive effective metric learning supervision throughout the training process.
II. R ELATED W ORKS
As computer vision technology has evolved, the person
ReID task has achieved great advancements in both academic
research and practical applications [22], [23], [24], [25]. However, the image-based person ReID necessitates at least one
image of the target pedestrian in the application. Therefore, the
practical application of image-based person ReID is limited.
To address this limitation and enhance the practicality of
person ReID, Li et al. [26] proposed the text-to-image person
re-identification task. Researchers have proposed a variety of
re-identification frameworks, which can be primarily divided
into global matching methods [27], [28], [29] and local matching methods [30], [31], [32].
A. Global Matching Method
The global matching method was the main research
approach in the early TI-ReID task. It focuses on learning the correspondence between person images and text
descriptions holistically, calculating similarity based on global
features [13], [14], [28], [29], [33]. In [26], Li et al. proposed
the recurrent neural network with a gated neural attention
mechanism network (GNA-RNN) and used the Visual Geometry Group (VGG) network to extract text and image features
respectively for similarity measurement. In addition, they
proposed the CUHK-PEDES dataset. In [27], Zhang and Lu
posited that accurately measuring feature similarity across
different modalities is crucial for matching images and texts.
To tackle this, they proposed the cross-modal projection
matching (CMPM) loss and the cross-modal projection classification (CMPC) loss to effectively enhance the compactness
between each person features. Li et al. [34] proposed the visual
semantic reasoning network (VSRN), which uses the GCN
to capture the semantic relationship of salient regions in the
image. It then utilizes the gating and memory mechanism for
global reasoning, enhancing the performance of image-text
matching tasks. With the development of large pre-trained
models, some research attempts to use visual, language, and
other pre-trained models to enhance the expression ability
of the person features and distinguish discriminate between
different persons [13], [35]. Ding and Mang [35]. leveraged the
cross-modal image-text alignment capability of the Contrastive
Language-Image Pre-training (CLIP) model solely for enhancing performance using global features. While such methods

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:00:38 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: RMGNet: THE PROGRESSIVE RELATIONSHIP-MINING GRAPH NEURAL NETWORK FOR TI-ReID

have achieved certain results, the absence of constraints on
local features limits their effectiveness in real-world application scenarios.
B. Local Matching Methods
Motivated by traditional person ReID methods based on
local matching [36], [37], [38], researchers have proposed
local matching-based TI-ReID methods. This type of method
primarily focuses on learning local discriminative information
in person images and texts, which key is to performing
cross-modal alignment [16], [17], [18], [39]. Jing et al. [40]
performed cross-modal recognition by introducing pose estimation information to align local features. In [41], Aggarwal et
al. designed the cross-modal attribute-aided matching framework (CMAAM) which approach introduces and preserves
high-level semantic information in pedestrian features through
an additional attribute prediction model. This helps alleviate
the modal gap interference and improves the effect of feature
learning. Ding et al. [39] raised the semantically self-aligned
network (SSAN) to mine semantically aligned local features
from person images and texts. They also establish the correspondence between person parts and word phrases through
a multi-view non-local network. This effectively alleviates
significant modal gaps and intra-class differences. Furthermore, they proposed the widely used dataset ICFG-PEDES.
Yang et al. [42] redesigned the cross-attention module to limit
the gap between different modality features and introduced
direct constraints in local feature matching progress. In [43],
Han et al. adopted the graph convolutional network (GCN)
to extract and fuse multi-modal features. And proposed an
asymmetric multi-level alignment module to extract “local”
information more accurately from a “global” perspective. The
local matching method further improves the effectiveness of
TI-ReID. However, there is an asymmetry in the amount of
information contained in images and texts, that is, the semantic
information in images is relatively redundant, while the semantic information in texts is relatively lacking. Therefore, forced
alignment of local features will disrupt the feature-extracting
process. Consequently, it is difficult to further improve the
feature expression ability and distinguishability.
Hence, we start by learning the mutual relationships
between different features and propose the RMGNet. The network employs the IMRM module and the CMRM module to
progressively model and mine potential semantic relationships
between different features. Following this, the interrelationship
between features is utilized as a guide to aggregate and
strengthen contextual information, to augment the expressiveness and discriminative of features.
III. M ETHOD
In this section, we provide a detailed introduction to the
proposed TI-ReID method. Firstly, we introduce the proposed
RMGNet, which includes the IMRM and the CMRM module.
Next, we propose the C2FL training strategy and the Acc loss.
Finally, we explain how to calculate the similarity between the
person images and texts.

5751

A. Overview of Framework
The overall architecture of the proposed RMGNet is illustrated in Figure 2(a). The network mainly consists of three
parts: the single-modal feature extraction module, the IMRM
module, and the CMRM module. In the single-modal feature
extraction module, we utilize the pre-trained Vision Transformer (ViT) [44] and Bidirectional Encoder Representations
from Transformers (BERT) network [45] to extract image and
text features of the person, respectively. It should be noted
that, unlike other TI-ReID methods, when extracting person
text features, we extract forward and backward text features
through forward-order and reverse-order input, respectively.
Subsequently, the extracted local features are fed into the
IMRM module to learn the relationships between different
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


exec
/bin/zsh -lc "pdftotext -l 3 'RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Robust mixed-degradation person Re-identification via structural consistency distillation.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113938

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Robust mixed-degradation person Re-identification via structural
consistency distillation
Siyuan Zhao a,c , Wenxin Huang b ,∗, Wenxuan Liu d , Xuemei Jia e , Xiyu Han c , Siqi Liang f,c ,
Xian Zhong c,g ,∗
a

Sanya Science and Education Innovation Park, Wuhan University of Technology, Sanya, 572025, China
Hubei Key Laboratory of Big Data Intelligent Analysis and Application, School of Computer Science, Hubei University, Wuhan, 430062, China
c Hubei Key Laboratory of Transportation Internet of Things, School of Computer Science and Artificial Intelligence, Wuhan University of
Technology, Wuhan, 430070, China
d State Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University, Beijing, 100871, China
e
National Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan University, Wuhan, 430072, China
f
Department of Computer Science and Engineering, Shanghai Jiao Tong University, Shanghai, 200240, China
g
State Key Laboratory of Maritime Technology and Safety, Wuhan University of Technology, Wuhan, 430063, China
b

ARTICLE

INFO

Dataset link: https://github.com/SiYuanZhaoh
aha/mdcd_reid
Keywords:
Person re-identification
Mixed degradation
Structural consistency distillation
Fused Gromov-Wasserstein distance
Optimal transport

ABSTRACT
Existing person re-identification (Re-ID) studies assume a single, uniformly degraded domain. However, realworld surveillance data are significantly more complex, with clean images often coexisting alongside various
degradations, such as fog, rain, snow, and illumination variations. These mixed degradations not only alter
the global feature distribution between clean and degraded domains but also distort the geometric structure
of identities (IDs) within the degraded domains, thereby complicating reliable matching. To address these
challenges, we propose a Mixed-Degradation Consistency Distillation (MDCD) framework, which consists
of two complementary feature-level modules. The first module, Structural Consistency Distillation (SCD),
utilizes the Fused Gromov-Wasserstein distance to align global feature distributions while maintaining intraID structural relations within a unified optimal transport framework. The second module, Clean Feature
Restoration (CFR), applies elastic weight consolidation to regularize parameters critical for clean-image
recognition, thus mitigating catastrophic forgetting during mixed-degradation training. Both modules are
backbone-agnostic and can be seamlessly integrated into existing Re-ID architectures. To rigorously evaluate
robustness under mixed degradations, we construct two synthetic benchmarks, Mixed-Market1501 and MixedMSMT17, simulating various weather conditions for person Re-ID. Extensive experiments demonstrate that
MDCD effectively reduces cross-domain discrepancies and achieves state-of-the-art performance in both mAP
and CMC metrics, underscoring its strong potential for real-world deployment. The dataset and code will be
released at https://github.com/SiYuanZhaohaha/mdcd_reid.

1. Introduction
Person re-identification (Re-ID) matches individuals across different
camera views using identity (ID)-related features, making it crucial for
surveillance systems. While existing methods [1–4] perform well in
ideal conditions, their accuracy degrades significantly under adverse
weather due to severe image distortions. Some studies have improved
retrieval efficiency using model optimization techniques, such as filter pruning [5], but these still struggle with degraded images. To
address this, prior work has explored image restoration [6,7], joint
optimization [8,9], and feature disentanglement [10,11], leading to

improvements. Recent advancements also incorporate multimodal feature fusion for better performance under degraded conditions [12],
showing promising results for enhancing robustness in challenging
environments.
As shown in Fig. 1(a), existing approaches typically assume a single
degradation type, where query and gallery images share similar degradation characteristics, such as foggy conditions [8,9,13]. Under these
conditions, the domain gap arises primarily between clean and degraded domains, typically manifested as a centroid shift in the feature
space. However, in practical surveillance scenarios, images are often

∗ Corresponding authors.

E-mail addresses: siyuanzhao@whut.edu.cn (S. Zhao), wenxinhuang_wh@163.com (W. Huang), liuwx66@pku.edu.cn (W. Liu), jiaxuemeiL@whu.edu.cn
(X. Jia), hanxy@whut.edu.cn (X. Han), liangsiqi1998@sjtu.edu.cn (S. Liang), zhongx@whut.edu.cn (X. Zhong).
https://doi.org/10.1016/j.patcog.2026.113938
Received 8 December 2025; Received in revised form 3 May 2026; Accepted 4 May 2026
Available online 6 May 2026
0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 179 (2026) 113938

S. Zhao et al.

Fig. 1. Illustration of domain shift. (a) Domain gap between clean and degraded domains under a single degradation type. (b) Heterogeneity gap caused by
mixed degradations and the distribution shifts among their respective domains.
Table 1
Comparison of key characteristics across three Re-ID settings. Comparison of inter-domain gap, intra-domain
structure, and modeling difficulty between Cross-Modality Re-ID, Single-Degradation Re-ID, and Mixed-Degradation
Re-ID.
Task

Cross-Modality Re-ID

Single-Degradation Re-ID

Mixed-Degradation Re-ID

Inter-domain gap

Modality shift

Single-domain shift

Multiple centroid shifts

Intra-domain structure

Intact topology

Preserved structure and semantics

Severely distorted topology

Modeling difficulty

Cross-modality alignment

Deterministic single corruption

Open-set stochastic mixture

Dimension

affected by mixed degradations. For example, a rainy query may need
to be matched against gallery images captured under foggy, snowy, or
overexposed conditions. Such cross-degradation retrieval significantly
increases the difficulty of reliable ID matching, as the same ID must be
recognized across heterogeneous visual distortions.
To further clarify the differences, Table 1 compares three representative Re-ID settings. Cross-modality Re-ID [14–17] primarily involves
modality-induced centroid shifts while largely preserving intra-class
topology. Single-degradation Re-ID introduces a known corruption,
resulting in moderate distribution shifts with relatively stable structural relations. In contrast, mixed-degradation Re-ID simultaneously
suffers from multiple centroid shifts and severely distorted intra-class
topology, as query and gallery images may be affected by different,
unpredictable degradations (e.g., rain and fog). This dual misalignment
cannot be effectively addressed through simple distribution alignment.
To tackle these challenges, we propose a Mixed-Degradation Consistency Distillation (MDCD) framework for robust person Re-ID under
mixed-degradation conditions. The framework consists of two key
components: the Structural Consistency Distillation (SCD) module and
the Clean Feature Restoration (CFR) module. In SCD, we formulate the
alignment process using a unified Fused Gromov-Wasserstein (FGW)
objective that integrates two complementary consistencies within a
shared optimal transport formulation. The Wasserstein [18] term aligns

the global feature distributions of clean and degraded domains, reducing large centroid shifts. Meanwhile, the Gromov-Wasserstein [19]
component preserves pairwise geometry among samples, ensuring that
structural relations remain stable across domains. By alternating
Sinkhorn [20] updates within the FGW objective, SCD avoids the
additional computational overhead and gradient conflicts associated
with a naive WD+GWD combination, effectively reducing domain discrepancies while maintaining topological consistency in the degraded
feature space.
To mitigate knowledge degradation during mixed-degradation training, the CFR module preserves discriminative capability on clean images. Specifically, CFR adopts elastic weight consolidation to identify
parameters critical for clean-image recognition and introduces a regularization constraint that limits significant parameter changes when
learning from degraded data. The complementary effects of SCD and
CFR enable the model to maintain robust discriminative representations
across both clean and degraded scenarios.
To facilitate reproducible evaluation under weather-related degradations, we construct two synthetic mixed-degradation benchmarks.
Four representative weather factors, fog, rain, snow, and illumination variation, are applied to each image in Market1501 [21] and
MSMT17 [22] at randomly sampled severity levels, producing MixedMarket1501 and Mixed-MSMT17. Although synthetic, these datasets
2

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


codex
以下判断只基于每篇 PDF 前三页，也就是摘要和引言。

**逐篇提取**

1. `Probabilistic Distribution Alignment for Text-Based Person Retrieval`
创新类型是问题重定义加新机制。它把文本行人检索从“点到点匹配”改写成“语义范围包含实例”，文本是更抽象的范围，图像是具体实例。能发的点在于它抓住了一对多和语义不对称这个 gap，再用高斯分布、跨模态包含、模态内遮挡包含来证明。story 是“不要把人和描述压成一个点，要学语义作用域”。

2. `Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification`
创新类型是新机制，偏工程组合。gap 是可见光和红外的局部对齐不能靠水平切块，人体运动和视角会让条带语义错位。它用多特征生成和可学习原型挖潜在相似局部。story 是“不是硬切部位，而是让原型自己找跨模态相似语义”。

3. `Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification`
创新类型是新机制。gap 是终身 ReID 不能长期存旧图，分类增量里的原型方法又只关心分类器，不关心 ReID 的细粒度嵌入空间。它用旧身份原型把新身份特征推开，再用自适应参数融合平衡旧知识和新任务。story 是“旧知识不以图片形式回放，而以可传播的原型边界继续约束新空间”。

4. `Rethinking Joint Optimization in Feature Compression`
创新类型是问题重定义和归因拆解。它不急着提新模块，而是质疑“联合优化一定有效”这个默认叙事，指出很多收益可能来自尾部语义解析器。ReID 被用作诊断任务，因为匹配阶段是确定性距离，能拆掉解析器影响。story 是“过去认为是联合优化的收益，其实可能是后端解析能力的收益”。

5. `Richer Semantics, Better Alignment`
创新类型是新机制加新语义设定。gap 是 VI-ReID 的 CLIP prompt 往往未知、粗糙、两阶段，而且可见光和红外描述中会有冲突语义，比如颜色。它用 LLaVA 生成显式文本，用间接相似度约束处理跨模态冲突，再融合多视角文本。story 是“语义空间没有模态差，但必须用显式、多视角、可处理冲突的语义来对齐”。

6. `RMGNet`
创新类型是新机制。gap 是文本图像 ReID 只做局部词块和图块对齐，忽略“背包是背着还是拿着”这类关系语义。它用图网络挖模态内关系和跨模态关系。story 是“识别人不只看局部是什么，还要看局部之间是什么关系”。

7. `RMPSNet`
创新类型是新设定细化加工程组合。gap 是遮挡 ReID 里随机擦除和真实遮挡分布不一致，且 CLIP prompt 对缺失语义不稳。它用下半身优先擦除、双 masked prompt、多分支分布对齐来补。story 是“训练遮挡必须像真实遮挡，文本 prompt 也要学会缺词时仍保持身份语义”。

8. `Robust mixed-degradation person Re-identification`
创新类型是问题重定义加新数据设定加新机制。这篇最值得学。它把退化 ReID 从“单一退化域”改成“干净图和雾、雨、雪、光照等混合退化共存”，核心不是一个中心偏移，而是多个中心偏移加身份拓扑扭曲。机制用 Fused Gromov-Wasserstein 保结构一致，再用 EWC 保干净识别能力。story 是“真实世界不是一个退化域，而是混合退化把身份空间结构扭坏了”。

9. `See what you seek`
创新类型是新机制，偏 CLIP prompt 组合。gap 是换衣 ReID 不是只要压掉衣服，还要显式挖头发、眼镜、背包等稳定正因素。它用衣服 prompt 和身份 prompt 做语义分离，再用正交文本特征指导视觉。story 是“别只看要去掉什么，还要看换衣后还能依赖什么”。

**反推出来的强候选**

1. **航拍图不是地面图的弱版本，而是由 3D 可见性定义的身份语义范围。**  
挂团队资产：aerial-ground 数据、SMPL 3D 几何、SOLIDER-Swin。  
切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
cheap kill-switch：冻结 SOLIDER 特征，在 AG-ReID.v2 或 CARGO 上只训练一个很小的均值方差头，方差由 SMPL 可见性、投影面积、低清程度控制。若 containment 分数相对余弦检索没有超过噪声线，或者学到的方差和视角质量无关，就停。

2. **跨视角 ReID 的不变量不是部位本身，而是人体部位关系在 3D 到 2D 投影中的可恢复结构。**  
挂团队资产：SMPL mesh/joints、pose 热图门控、SOLIDER-Swin。  
切开点：最像 RMGNet，但 RMGNet 挖的是图文特征里的隐关系；我们要挖的是物理可解释的人体关节、肢体、可见区域之间的投影关系。它也不同于简单加 3D 特征，核心证据必须证明“关系结构蒸馏”比“SMPL 向量拼接”有效。  
cheap kill-switch：先不训练主模型，只抽 500 到 1000 对样本的 SMPL/joint 关系矩阵，测同身份和异身份在关系空间里的分离度，并和随机边、2D pose、水平条带关系对比。若关系矩阵本身没有任何正负分离，或者主要失败来自 SMPL 估计不可用，就停。

3. **真实航拍地面 ReID 不是单一视角差，而是高度、低清、遮挡、姿态和成像模式混在一起造成的身份拓扑扭曲。**  
挂团队资产：aerial-ground 数据、SOLIDER-Swin。  
切开点：最像 mixed-degradation ReID，但那篇是天气退化的混合；我们做的是行人跨平台观测混合，变量是高度、俯仰角、分辨率、遮挡和地面视角。关键不是再做一个鲁棒增强，而是定义“混合观测 ReID”并证明身份邻域结构在混合观测下被扭曲，普通域对齐只能修中心偏移，修不了拓扑。  
cheap kill-switch：冻结 SOLIDER，在航拍地面数据上按高度、分辨率、视角、遮挡分桶，计算同身份邻域保持率和跨桶检索掉点。若只是整体中心偏移，简单白化或域对齐就能恢复，那这个方向不够强。

4. **持续接入无人机和地面摄像头时，旧图库不应靠全量重提特征保命，而应靠几何原型保持跨版本兼容。**  
挂团队资产：当前终身 ReID 基建、aerial-ground、SMPL。  
切开点：最像 PKP 和 BCT/C2R 一类兼容学习，但它们主要处理通用旧新模型兼容或身份原型保持。我们的切口必须收窄到“跨视角摄像头持续接入造成的旧索引失配”，原型也不能只是身份均值，而要包含视角、可见部位、3D 几何状态。  
cheap kill-switch：用已有顺序训练基建模拟“地面先建库，航拍后接入”或反过来，测旧图库特征混用是否出现明显、且随视角差变化的掉点。若混版本掉点不显著，或者掉点和视角几何无关，就不要硬做。

我的排序是：优先看候选 1 和候选 3。候选 1 机制更有新意，能挂 SMPL 和 aerial-ground；候选 3 问题定义更强，容易写成 Pattern Recognition/TCSVT 风格的完整故事。候选 2 可作为候选 1 的机制分支。候选 4 和当前终身基建贴合，但先例压力最大，只适合在 stale index 掉点确实很强时继续。
tokens used
83,251
以下判断只基于每篇 PDF 前三页，也就是摘要和引言。

**逐篇提取**

1. `Probabilistic Distribution Alignment for Text-Based Person Retrieval`
创新类型是问题重定义加新机制。它把文本行人检索从“点到点匹配”改写成“语义范围包含实例”，文本是更抽象的范围，图像是具体实例。能发的点在于它抓住了一对多和语义不对称这个 gap，再用高斯分布、跨模态包含、模态内遮挡包含来证明。story 是“不要把人和描述压成一个点，要学语义作用域”。

2. `Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification`
创新类型是新机制，偏工程组合。gap 是可见光和红外的局部对齐不能靠水平切块，人体运动和视角会让条带语义错位。它用多特征生成和可学习原型挖潜在相似局部。story 是“不是硬切部位，而是让原型自己找跨模态相似语义”。

3. `Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification`
创新类型是新机制。gap 是终身 ReID 不能长期存旧图，分类增量里的原型方法又只关心分类器，不关心 ReID 的细粒度嵌入空间。它用旧身份原型把新身份特征推开，再用自适应参数融合平衡旧知识和新任务。story 是“旧知识不以图片形式回放，而以可传播的原型边界继续约束新空间”。

4. `Rethinking Joint Optimization in Feature Compression`
创新类型是问题重定义和归因拆解。它不急着提新模块，而是质疑“联合优化一定有效”这个默认叙事，指出很多收益可能来自尾部语义解析器。ReID 被用作诊断任务，因为匹配阶段是确定性距离，能拆掉解析器影响。story 是“过去认为是联合优化的收益，其实可能是后端解析能力的收益”。

5. `Richer Semantics, Better Alignment`
创新类型是新机制加新语义设定。gap 是 VI-ReID 的 CLIP prompt 往往未知、粗糙、两阶段，而且可见光和红外描述中会有冲突语义，比如颜色。它用 LLaVA 生成显式文本，用间接相似度约束处理跨模态冲突，再融合多视角文本。story 是“语义空间没有模态差，但必须用显式、多视角、可处理冲突的语义来对齐”。

6. `RMGNet`
创新类型是新机制。gap 是文本图像 ReID 只做局部词块和图块对齐，忽略“背包是背着还是拿着”这类关系语义。它用图网络挖模态内关系和跨模态关系。story 是“识别人不只看局部是什么，还要看局部之间是什么关系”。

7. `RMPSNet`
创新类型是新设定细化加工程组合。gap 是遮挡 ReID 里随机擦除和真实遮挡分布不一致，且 CLIP prompt 对缺失语义不稳。它用下半身优先擦除、双 masked prompt、多分支分布对齐来补。story 是“训练遮挡必须像真实遮挡，文本 prompt 也要学会缺词时仍保持身份语义”。

8. `Robust mixed-degradation person Re-identification`
创新类型是问题重定义加新数据设定加新机制。这篇最值得学。它把退化 ReID 从“单一退化域”改成“干净图和雾、雨、雪、光照等混合退化共存”，核心不是一个中心偏移，而是多个中心偏移加身份拓扑扭曲。机制用 Fused Gromov-Wasserstein 保结构一致，再用 EWC 保干净识别能力。story 是“真实世界不是一个退化域，而是混合退化把身份空间结构扭坏了”。

9. `See what you seek`
创新类型是新机制，偏 CLIP prompt 组合。gap 是换衣 ReID 不是只要压掉衣服，还要显式挖头发、眼镜、背包等稳定正因素。它用衣服 prompt 和身份 prompt 做语义分离，再用正交文本特征指导视觉。story 是“别只看要去掉什么，还要看换衣后还能依赖什么”。

**反推出来的强候选**

1. **航拍图不是地面图的弱版本，而是由 3D 可见性定义的身份语义范围。**  
挂团队资产：aerial-ground 数据、SMPL 3D 几何、SOLIDER-Swin。  
切开点：最像 PDA，但 PDA 的“包含关系”来自文本和图像的抽象层级；我们这里的包含关系来自航拍低清、俯视、遮挡下的 3D 可见性。它也不同于普通跨视角对齐，不是假设两个视角应映射到同一点，而是假设航拍观测对应一个由 SMPL 可见部位和投影不确定性定义的身份范围。  
cheap kill-switch：冻结 SOLIDER 特征，在 AG-ReID.v2 或 CARGO 上只训练一个很小的均值方差头，方差由 SMPL 可见性、投影面积、低清程度控制。若 containment 分数相对余弦检索没有超过噪声线，或者学到的方差和视角质量无关，就停。

2. **跨视角 ReID 的不变量不是部位本身，而是人体部位关系在 3D 到 2D 投影中的可恢复结构。**  
挂团队资产：SMPL mesh/joints、pose 热图门控、SOLIDER-Swin。  
切开点：最像 RMGNet，但 RMGNet 挖的是图文特征里的隐关系；我们要挖的是物理可解释的人体关节、肢体、可见区域之间的投影关系。它也不同于简单加 3D 特征，核心证据必须证明“关系结构蒸馏”比“SMPL 向量拼接”有效。  
cheap kill-switch：先不训练主模型，只抽 500 到 1000 对样本的 SMPL/joint 关系矩阵，测同身份和异身份在关系空间里的分离度，并和随机边、2D pose、水平条带关系对比。若关系矩阵本身没有任何正负分离，或者主要失败来自 SMPL 估计不可用，就停。

3. **真实航拍地面 ReID 不是单一视角差，而是高度、低清、遮挡、姿态和成像模式混在一起造成的身份拓扑扭曲。**  
挂团队资产：aerial-ground 数据、SOLIDER-Swin。  
切开点：最像 mixed-degradation ReID，但那篇是天气退化的混合；我们做的是行人跨平台观测混合，变量是高度、俯仰角、分辨率、遮挡和地面视角。关键不是再做一个鲁棒增强，而是定义“混合观测 ReID”并证明身份邻域结构在混合观测下被扭曲，普通域对齐只能修中心偏移，修不了拓扑。  
cheap kill-switch：冻结 SOLIDER，在航拍地面数据上按高度、分辨率、视角、遮挡分桶，计算同身份邻域保持率和跨桶检索掉点。若只是整体中心偏移，简单白化或域对齐就能恢复，那这个方向不够强。

4. **持续接入无人机和地面摄像头时，旧图库不应靠全量重提特征保命，而应靠几何原型保持跨版本兼容。**  
挂团队资产：当前终身 ReID 基建、aerial-ground、SMPL。  
切开点：最像 PKP 和 BCT/C2R 一类兼容学习，但它们主要处理通用旧新模型兼容或身份原型保持。我们的切口必须收窄到“跨视角摄像头持续接入造成的旧索引失配”，原型也不能只是身份均值，而要包含视角、可见部位、3D 几何状态。  
cheap kill-switch：用已有顺序训练基建模拟“地面先建库，航拍后接入”或反过来，测旧图库特征混用是否出现明显、且随视角差变化的掉点。若混版本掉点不显著，或者掉点和视角几何无关，就不要硬做。

我的排序是：优先看候选 1 和候选 3。候选 1 机制更有新意，能挂 SMPL 和 aerial-ground；候选 3 问题定义更强，容易写成 Pattern Recognition/TCSVT 风格的完整故事。候选 2 可作为候选 1 的机制分支。候选 4 和当前终身基建贴合，但先例压力最大，只适合在 stale index 掉点确实很强时继续。
