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
session id: 019ef835-90fb-78b2-bd65-8e54b0a40035
--------
user
你是 ReID 创新方法论拆解员 10。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(10)==
Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf
Distribution aligned semantics adaption for lifelong person re-identification.pdf
Diverse Representations Embedding for Lifelong Person Re-Identification.pdf
DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf
Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则确认当前目标和目录内容，然后逐篇用 `pdftotext` 抽方法部分。这个任务只读文件，不会改动实验记录。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc pwd in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc 'rg --files' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM arXiv].pdf
Condense loss - Exploiting vector magnitude during person Re-identification training process.pdf
Improving Text-Based Person Retrieval by Excavating All-Round Information Beyond Color.pdf
Internal-External Context Interaction Network for Person Re-Identification.pdf
Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf
Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
Shape-centered representation learning for visible-infrared person re-identification.pdf
Multi-feature balanced network for clothes-changing person re-identification.pdf
Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf
Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf
GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf
manifest.csv
A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf
Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
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
Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
Occluded person Re-Identification with noise injection.pdf
Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf
Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
CLIP-driven fine-grained mining for text-based person search.pdf
'Knowledge and experience' for visible-infrared person re-identification.pdf
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
Cross-modality average precision optimization for visible thermal person re-identification.pdf
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

hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
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

hook: PreToolUse Completed
hook: PreToolUse Completed
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

codex
我已经确认 `GOAL.md` 仍然是文献调研和方向差距分析，和这次任务一致。接下来先定位每篇的方法章节标题和页内范围，再把对应段落读完整。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf' - | rg -n '("'^[0-9]+'"\\.? |"'^[IVX]+'"\\.|Method|METHOD|Approach|APPROACH|Proposed|Framework|Overview|Experiments|EXPERIMENT)'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
45:I. I NTRODUCTION
49:2023 and 30 January 2024; accepted 28 March 2024. Date of publication
50:15 April 2024; date of current version 1 March 2025. This work was
64:feature representations due to the modality and posture factors. (b) and (c) Proposed DMPF successfully disentangles the modality and posture factors from
268:II. R ELATED W ORK
299:1) Image Alignment Methods: To address the dissimilarities
354:2) Feature Learning Methods: To address the modality gap
414:III. M ETHODS
1137:IV. E XPERIMENTS

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Distribution aligned semantics adaption for lifelong person re-identification.pdf' - | rg -n '("'^[0-9]+'"\\.? |"'^[IVX]+'"\\.|Method|METHOD|Approach|APPROACH|Proposed|Framework|Overview|Experiments|EXPERIMENT)'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
4:1 Fudan University, Shanghai, China.
8:2 Northwestern Polytechnical University, Xi’an, China.
77:1 Introduction
143:Methods can be traditionally divided into three categories, namely regularizationbased, rehearsal-based, and architecture-based methods. Regularization-based methods [17, 18] limit updating important parameters for preceding tasks to mitigate
189:3 Methodology
243:1 X
417:5 × 5 can achieve a good balance between effectiveness and efficiency.
448:4 Experiments

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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Diverse Representations Embedding for Lifelong Person Re-Identification.pdf' - | rg -n '("'^[0-9]+'"\\.? |"'^[IVX]+'"\\.|Method|METHOD|Approach|APPROACH|Proposed|Framework|Overview|Experiments|EXPERIMENT)'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
78:I. I NTRODUCTION
204:II. R ELATED WORK
267:III. P ROPOSED M ETHOD
268:A. Overview
399:1 1 XX
448:1 X n
461:1 X
492:1 X
618:1 X
659:1 X
665:IV. E XPERIMENTS
700:P ERFORMANCE COMPARISON WITH STATE - OF - THE - ART METHODS FOR TRAINING ORDER -1. B OLD FONT INDICATES OPTIMAL VALUES AND RED FONT IS
702:Method
963:P ERFORMANCE COMPARISON WITH STATE - OF - THE - ART METHODS FOR TRAINING ORDER -2. B OLD FONT INDICATES OPTIMAL VALUES AND RED FONT IS
965:Method
1673:Method

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | rg -n '("'^[0-9]+'"\\.? |"'^[IVX]+'"\\.|Method|METHOD|Approach|APPROACH|Proposed|Framework|Overview|Experiments|EXPERIMENT)'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
27:1 These authors contributed equally to this work.
32:1. Introduction
101:2. Related Work
150:3. Proposed Method
151:3.1. Overview

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf' - | rg -n '("'^[0-9]+'"\\.? |"'^[IVX]+'"\\.|Method|METHOD|Approach|APPROACH|Proposed|Framework|Overview|Experiments|EXPERIMENT)'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf' - | rg -n '("'^[0-9]+'"\\.? |"'^[IVX]+'"\\.|Method|METHOD|Approach|APPROACH|Proposed|Framework|Overview|Experiments|EXPERIMENT)'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
39:Without bells and whistles, it outperforms the state-of-the-art on various DG re-ID benchmarks. Experiments
42:1. Introduction
197:2. Related work
249:for enhancing DG performance. Experiments and ablation studies show
252:3. Methodology
263:3.2. Overview
337:1 ∑
369:1. The Frobenius norm is the sum of the squares of all elements in
403:1 ∑∑
454:1 ∑
513:1 + 𝑑𝑤𝑎𝑠𝑠
594:19 end
605:1 camera. For Protocol-3, we alternately use M and D as the source
617:4. Experiments
855:Method
997:Method
1017:4 images per identity for each domain. Random flipping and random
1060:Fig. 3. Ablation study on the position at which DSH is applied under Protocol-1. Experiments are conducted under the M+D+MS → C3 setting.
1065:Method
1178:Method
1442:Method
1452:Fig. 4. Comparisons between DSH and existing stylization methods under Protocol-2. Experiments are done under the MS+CS+C3 → M setting.
1454:Fig. 5. Ablation study on the value of 𝜔. Experiments are conducted under the MS+CS+C3 → M setting.

 succeeded in 3842ms:
76:(a) Existing Methods
88:I. I NTRODUCTION
275:II. R ELATED WORK
502:Fig. 2. Overview of the proposed DCR for LReID. First, the attribute-text generator (ATG) dynamically generates text-image pairs for each instance. Then, the
511:Proposed
537:are Method
546:III. P ROPOSED M ETHOD
547:A. Preliminary: Overview of Method
642:2. Firstly, we employ a pre-trained attribute recognition model
724:1 X
823:1 X
848:1 X
859:1 X
921:Existing Methods
1020:IV. E XPERIMENTS
1023:A. Experiments Setting
1096:II.
1104:DKP Method
1178:B. Comparison with SOTA Methods
1253:4) The Anti-forgetting Performance of Our Method: We
1265:P ERFORMANCE COMPARISON WITH STATE - OF - THE - ART METHODS ON TRAINING ORDER -1. B OLD AND RED FONTS ARE OPTIMAL AND SUBOPTIMAL
1486:Method
1489:P ERFORMANCE COMPARISON WITH STATE - OF - THE - ART METHODS ON TRAINING ORDER -2. B OLD AND RED FONTS ARE OPTIMAL AND SUBOPTIMAL
1710:Method
1811:4. t-SNE visualization

codex
6 篇题名都在目录里，用户写“10 篇”但实际列了 6 篇。我先按列出的 6 篇逐篇做，不会用目录里的其他论文补数。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf' - | sed -n '414,1136p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
III. M ETHODS
In this section, we introduce the pipeline of DMPF in
detail, and the diagram of our method is illustrated in Fig. 2.
The proposed method consists of three modules, including
TFENet, MFD, and PFD. The detailed introduction is shown
in Section III.
A. Preliminaries
nV
Given a training set consisting of VIS subset {(IiV , yiV )}i=1
N
N nN
and NIR subset {(Ii , yi )}i=1 , our objective is to learn
identity-invariant representations for IiV and IiN , where IiV
and IiN represent VIS and NIR images, respectively, while yiV
and yiN ∈ {1, 2, . . . , n c } indicate the corresponding identity
labels. The variables n V and n N denote the number of images
in VIS and NIR subsets, respectively. During the inference
stage, the well-trained model φ maps VIS and NIR images to
identity-discriminative feature embeddings, namely, E V and
E N . By comparing the similarity between E V and E N , we can
determine the likelihood of VIS-NIR image pairs sharing the
same identity. Therefore, it is crucial to learn an effective
identity representation for heterogeneous images. However,
the inclusion of modality and posture factors in E V and E N
poses a challenge for VIS-NIR identity matching. To address
this issue, we propose the DMPF model to disentangle the
two identity-unrelated factors by fusing features memory and
skeleton information. A detailed description of the DMPF
model is presented in Sections III-B–III-D.
B. Three-Stream Features Extraction Network
Due to the dramatic modality discrepancy and pedestrian
posture variation between heterogeneous images, the identity
representation of images inevitably contains modality and
posture factors, which may impact VIS-NIR identity matching.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:54 UTC from IEEE Xplore. Restrictions apply.

5498

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

Fig. 2. Diagram of our DMPF, which contains TFENet, MFD, and PFD modules. First, the TFENet is designed as a three-stream network to extract VIS-NIR
images and skeleton features. Second, the MFD is proposed to integrate features in the whole training set by memory-attention layers, which encourages the
heterogeneous features with the same identity to present higher similarity. Third, the PFD is introduced to learn posture-unrelated features with the assistance
of the skeleton features.

To solve this problem, we fuse intraclass features memory and pedestrians’ skeleton information to decouple these
two identity-independent factors, respectively. In this section,
we introduce feature extraction and intraclass features memory
construction.
1) Image and Skeleton Features Extraction: Considering the variation in pedestrian postures across images, the
extracted identity features inevitably encompass posture factors, which impact the subsequent identity-matching process.
To address this challenge, we leverage the skeleton information
of pedestrians to extract pose-related skeleton features, which
are utilized to effectively disentangle the pose factor from
the identity features. Given a VIS/NIR pedestrian image I M ,
we adopt SCHP [50] and OpenPose [51] models to extract
the parsing I P and landmark I L corresponding to pedestrian
image I M . The I P represents the shape profile information,
while the I L reflects the location information of keypoints.
Considering the complementary between I P and I L , we fuse
the two types of information as skeleton information I S by
weighting operation.
Subsequently, the image I M and skeleton I S are fed into the
backbone for corresponding feature extraction. Considering
the heterogeneity of image and skeleton information, previous
approaches have generally utilized two networks to process
image and skeleton data independently [52], [53]. Nevertheless, these approaches demand considerable computation
costs. Moreover, the practice of mapping identity features and
identity-independent features to distinct spaces significantly
inflates the parameter space for optimization. This expansion
poses a challenge for the optimizer to converge with optimal
parameters. To overcome the above limitations, we propose
the TFENet, which incorporates both special convolutional
blocks and shared convolutional blocks. The former blocks
are designed to handle the two different data, while the latter
blocks aim at mapping the two types of data into a unified

feature space. The features of I M and I S are denoted as E M
and E S , respectively. The superscript M ∈ {V, N } indicates
modality. The TFENet enables more effective fusion and
integration of image and skeleton information, which encodes
identity-related and posture-related information of image and
skeleton into E M and E S , respectively, in a unified backbone
network. The skeleton features E S are served for posture
factors disentanglement in the PFD module.
2) Intraclass Features Memory Construction: To mitigate
the modality gap between heterogeneous samples in the feature
space, previous methods designed different losses to constrain
the distribution and distance of batch features [28], [29], [30].
Nevertheless, these methods tend to overlook the modality
disparities present in heterogeneous features outside the batch,
leading to the modality factor remaining in these features.
To tackle this challenge, we introduce memory queues to store
intraclass features for the entire training set. By incorporating
features memory, we can more effectively guide the model to
learn parameters that facilitate the disentanglement of modality
factors across the training set in each batch. In this section,
we present how to initialize and update the memory.
a) Memory initializing: We employ the TFENet to
extract the instance features of heterogeneous images in
the training set. Consequently, considering that instance-level
memory requires excessive computation costs, we initialize
the memory queue with class-level center features, which are
formulated as follows:
h


 i
M M = E cM 1 , E cM 2 , . . . , E cM n c
(1)
where superscript M ∈ {V, N } indicates modality, M M
denotes M-modality intraclass features memory matrix, (E cM )i
denotes M-modality ith class center features, and n c denotes
the number of identity categories.
b) Memory updating: As training proceeds, the weight
parameters of the TFENet vary in each batch, which causes

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:54 UTC from IEEE Xplore. Restrictions apply.

LU et al.: DMPFs: MEMORY-ATTENTION AND ORTHOGONAL DECOMPOSITION

5499

a) Memory self-attention layer: First, to explore the
intramodality associations between batch and memory features, we propose the MSL to process the two modality
features separately. Specifically, the batch features are concatenated with memory features and fed into the MSL for
intramodality association mining. The process is formulated
as follows:


MbM = E M ||M M

M̂bM = MSL Norm MbM + MbM

cbM = MLP Norm M̂bM + MbM
(3)
M
Fig. 3. Diagram of memory-attention layers, consisting of an MSL and
an MCL. The MSL and MCL are presented to progressively explore the
intramodality and cross-modality relationships between batch and memory
features.

the intraclass features in M M lag behind the updated batch
features [54]. To keep both consistent, we update the memory
with a momentum strategy in each batch, which is formulated
as follows: 3
 
 
M M yi ← ηM M yi + (1 − η)E iM
(2)
where E iM and yi denote the ith batch instance features and
corresponding identity label and η denotes the updated hyperparameter. The memory M M stores the intraclass features of
the training set, enabling the model to gain a more comprehensive understanding of the modality differences. The features
memory M M is employed for modality factor decoupling in
the MFD module.
C. Modality Factor Disentanglement
To learn modality-independent feature representations
across the entire training set, we propose MFD to integrate batch features and memory features using designed
memory-attention layers. By optimizing the layers with the
OT method [55], we can progressively explore intramodality
and intermodality relationships between batch and memory
features, allowing augmented heterogeneous features with
the same identity to present high similarity. As a result,
we successfully decouple the modality factors from identity
features by the memory-attention layers, thereby extracting augmented modality-independent features. Furthermore,
to transfer the ability of modality difference elimination
from the memory-attention layers to the backbone network,
we propose the MiD loss to measure the similarity between
the original batch features and the augmented batch features,
encouraging the backbone network to acquire the knowledge
required for learning modality-independent feature representations effectively. The detailed description is presented in the
following.
1) Memory-Attention Layers: Aiming to eliminate modality differences, we design the memory-attention layers to
explore the associations between batch features and memory
features. Considering that the association of the two features
involves both intramodality and cross-modality relationships,
we adopt progressive mining strategies. As shown in Fig. 3,
the memory-attention layers are composed of two sublayers: the memory self-attention layer (MSL) and the memory
cross-attention layer (MCL), which are presented to explore
the intramodality and cross-modality relationships between
batch and memory features, respectively.

where E M and M M denote M-modality batch and memory
features, respectively. [·||·] denotes concatenation operation,
Norm(·) denotes normalization operation, MLP(·) denotes
multilayer perceptron, and MSL(·) denotes memory selfattention operation, which is formulated as follows:



QKT
M
MSL X
= softmax √
V
Dv

M
XM
Q = Ls1

M
K = Ls2
XM

M
V = Ls3
XM
(4)
where X M denotes M-modality matrix, Q, K , and V are
obtained by importing X M into three different linear projection
M
M
M
layers Ls1
(·), Ls2
(·), and Ls3
(·) and their dimension are Dq ,
Dk , and Dv , respectively. It is worth noting that the weights
of the three linear projection layers for different modalities
matrices are not shared.
b) Memory cross-attention layer:
Consequently,
to exploit the cross-modality associations between batch
and memory features, we propose the MCL to interactively
fuse the features from different modalities. Concretely, the
cM are imposed into the
intramodality augmented matrix M
b
MCL for cross-modality association mining. The process is
formulated as follows:




cbM + M
cbM , Norm M
cbM
M̃bM = MCL Norm M

fbM = MLP Norm M̃bM + M
cbM
M
 M

M
M
f =M
fb
e ||M
E
(5)
fM denotes cross-modality augmented matrix, consistwhere M
b
fM and augmented batch features
ing of augmented memory M
M
e
E , superscript M, M ∈ {V, N } indicates different modality.
fM is adopted to optimize the MSL and MCL to export
The M
modality-independent features across the entire training set
eM is employed to
by OT optimization methods. While the E
guide the backbone to learn from the MSL and MCL layers by
designed MiD loss. MCL(·) denotes memory cross-attention
operation, which is formulated as follows:




QKT
MCL X M , X M = softmax √
V
Dv

M
Q = Lc1
XM


M
K = Lc2
XM


M
V = Lc3
XM
(6)
where superscript M, M ∈ {V, N } indicates different modality.
Similar to MCL, the weights of the three linear projection layers for different modality matrices are not shared.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:54 UTC from IEEE Xplore. Restrictions apply.

5500

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

Equations (5) and (6) introduce the MCL, which plays a
crucial role in enhancing cross-modality relationship mining
by merging the memory matrices from both modalities during
the training phase. Note that each batch in the training phase
comprises images from the same individual in both modalities.
Therefore, the issue of missing modalities does not arise in the
training stage. However, a challenge arises during the testing
phase, as the MCL cannot be employed due to the modality
missing problem. To tackle this challenge, we propose the MiD
loss, a solution that enables us to distill the cross-modality
knowledge from the MCL into the backbone network, which
is introduced in Section III-C2.
2) OT Optimization for the MSL and MCL: Aiming to
eliminate modality discrepancy in the raw memories M M ,
we propose the MSL and MCL layers to process M M and
fM . In other
expose modality-unrelated augmented memories M
V
f
fN are
words, the features with the same identity in M and M
expected to share the highest similarity. To achieve the goal,
we optimize these layers by solving the OT problem.
The OT problem can be formulated as the search for an
optimal soft assignment matrix A∗ ∈ [0, 1]n c ×n c that associates
fV and M
fN with minimum
identities between the features in M
cost. In the VIS-NIR Re-ID task, heterogeneous feature pairs
with higher similarity are more likely to share the same identity [56]. Stated differently, the assignment of low-similarity
heterogeneous features incurs higher costs in the OT problem.
Therefore, we adopt the negative similarity matrix between
fV and M
fN as the cost matrix C, which can be formulated
M
as follows:

 
fV [i] T · M
fN j
Ci j = − M
(7)
and then, the optimal soft assignment matrix A∗ is formulated
as follows:
A∗ = arg min
A∈U

r (A) = −

X

nc
nc X
X

Ai j C i j −

i=1 j=1

1
r (A)
γ

Ai j log Ai j

(8)

i, j
c ×n c
where U = {A ∈ Rn>0
|A · 1n c = 1n c , AT · 1n c = 1n c } denotes
feasible domain. The regularization term r (A) aims to give
fV a more equal chance to match the features
every feature in M
fN , and γ denotes weight coefficient. According to [55],
in M
the elements in the optimal assignment matrix A∗ are of the
form

Ai∗j = αi β j e−γ Ci j
= αi β j eγ (M [i]) ·(M [ j ])
fV

T

fN

(9)

where αi and β j represent constants that ensure the matrix
A∗ falls within the feasible domain U , i.e., A∗ · 1n c = 1n c
and A∗T · 1n c = 1n c . To determine the values of αi and β j ,
we employ the Sinkhorn–Knopp algorithm [57].
The optimal assignment matrix A∗ is expected to assign
identity between heterogeneous features correctly. For this
purpose, the MCL and MSL layers are trained under the supervision of ground truth assignment matrix A gt ∈ {0, 1}n c ×n c .
The supervised loss L OT is formulated as follows:
L OT = −

nc X
nc
X
i=1 j=1


gt
Ai j log Ai∗j .

(10)

gt

As the training progresses, Ai∗j will approach Ai j , suggesting
fV and the jth feature in M
fN
that when the ith feature in M
∗
belong to the same identity, the value of Ai j tends to 1; otherwise, it tends to 0. Based on the aforementioned observation
and (9), it can be inferred that heterogeneous features from the
same identity demonstrate higher similarity, whereas heterogeneous features from different identities exhibit lower similarity.
Indeed, this signifies that the optimized MSL and MCL layers
demonstrate effective eliminating modal discrepancy between
MV and M N , consequently yielding modality-independent
feature representations across the entire training set.
Furthermore, to transfer the capability of eliminating modality discrepancy from the MSL and MCL layers to the backbone
network, we introduce the MiD loss. This loss encourages
the backbone network to learn modality-independent feature
representations by measuring the similarity between original
batch features E M and augmented modality-unrelated batch
eM , which is formulated as follows:
features E
!
T
X
eM
EM · E
L MiD =
1−
(11)
eM ∥
∥E M ∥ · ∥ E
M∈{V,N }

where L MiD drives the backbone to distill and encode
the essential information shared across different modalities.
Hence, during the inference phase, even in the absence of
memory-attention layers, the backbone network can still effectively address the modality discrepancy in heterogeneous data.
D. Posture Factor Disentanglement
Due to the diverse postures of pedestrians in different
images, the extracted identity features inherently incorporate
the posture factor, which in turn affects subsequent identity
matching. To tackle this problem, the previous method has
employed pedestrian skeleton information to extract skeleton
features. Subsequently, the skeleton features are adopted to
decouple the posture factor from the identity features [33].
However, the extracted skeleton features encompass both
pose-related and identity-related information. The entanglement between the two types of information may impact posture
factor decoupling, potentially undermining the discriminability
of the identity features. To solve the problem, we perform
subspace orthogonal decomposition [36] on the skeleton features to extract posture-related and identity-related features
of the skeleton. The posture-related features are employed
to disentangle the posture factor by the proposed PfC loss,
while the identity-related features are utilized to improve
the discriminability of identity representations. The detailed
description is presented in the following.
1) Posture Orthogonal Decomposition: The above analysis indicates that the skeleton features E S encompass
both posture-related and identity-related information, which
impacts the disentanglement of the posture factor. To separate
the two types of information, we introduce two mapping
matrices, i.e., K po and K id , for modeling posture-related and
identity-related features. The mapping process is formulated
as follows:
S
E po
= K po · E S

E idS = K id · E S

(12)

S
where E po
and E idS denote posture-related and identityrelated features of skeleton, respectively. K po and K id denote

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:54 UTC from IEEE Xplore. Restrictions apply.

LU et al.: DMPFs: MEMORY-ATTENTION AND ORTHOGONAL DECOMPOSITION

5501

posture-related and identity-related mapping matrices, respectively. Considering the subspace decomposition properties of
the matrices K po and K id , we introduce an orthogonality
constraint to make them unrelated to each other. Furthermore,
based on Lagrange’s theorem [58], the orthogonal constraint
can be transformed into the orthogonal loss

E. Overall Objectives and Inference
We adopt E M as identity features representation, which is
trained under the supervision of L R . Formulaically

T
L SO = K po
· K id .

(13)

To incorporate identity-related information into E idS ,
we introduce the identity classification loss, which is formulated as follows:
nc
X


S
L ID
=−
y[c] log ȳ S [c]

L R = L CE + L TRI + L MC

where L CE denotes cross-entropy loss, L TRI denotes triplet
loss, and L MC denotes modality consistency loss. The L MC
aims to eliminate the modality discrepancy between VIS and
NIR features, which is formulated as follows:
nc
X
 V

ȳ [c] log ȳ N [c]
L MC = −
c=1

c=1

ȳ

S

= softmax FC BN E idS



−

(14)

M
E po
= Ppo · E M

E idM = Pid · E M
M
E po

(15)

E idM

where
and
denote posture-related and identityrelated features of image, respectively. Ppo and Pid denote
posture-related and identity-related mapping matrices, respectively. Similar to (13) and (14), we introduce the orthogonal
M
M
loss L O
and identity classification loss L ID
to embed
M
posture-related and identity-related information into E po
and
M
E id , respectively.
Representing the same posture of the same pedestrian, the
S
M
posture-related features E po
and E po
are expected to exhibit
a high similarity. To achieve this goal, we propose the PfC
S
M
loss to measure the similarity between E po
and E po
, which is
formulated as follows:
D
X
 S

L PfC = −
F [d] log F M [d]
−

d=1
D
X



F M [d] log F S [d]



d=1

  
S
F S = softmax FC E po
 

M
F M = softmax FC E po

(16)

S
M
where F S and F M are obtained by feeding E po
and E po
into the same encoder, which consists of a full connection
operation and softmax operation. As training proceeds, the
M
L PfC encourages the E po
to capture posture-related information
S
M
from E po . When E po effectively represents posture-related
information, the posture factor is disentangled from image
features due to the orthogonal constraint between Ppo and Pid .

nc
X



ȳ N [c] log ȳ V [c]

c=1

nc

where y ∈ R denotes ground-truth one-hot identity label,
ȳ S ∈ Rn c denotes identity predicted probability, FC(·) denotes
fully connection operation, and BN(·) denotes batch normalization operation. When E idS effectively represents identityS
related information, E po
might capture identity-unrelated
information, i.e., posture information, due to the orthogonal constraint. Subsequently, we employ the posture-related
S
features E po
to decouple the posture factor from the image
M
features E .
2) PfC Loss: Image features E M contain both
posture-related and identity-related information as well.
Similarly, we adopt the orthogonal subspace decomposition
for modeling pose-related and identity-related features, which
is formulated as follows:

(17)

ȳ V = softmax FC BN E V



ȳ N = softmax FC BN E N


.

(18)

The overall loss is formulated as
L = λ1 L R
| {z }

ReID Loss

+ λ2 L OT + λ3 L MiD
|
{z
}
MFD Loss


M
S
M
+ λ4 L SO + L O
+ λ5 L ID
+ L ID
+ λ6 L PfC
|
{z
}

(19)

PFD Loss

where balancing coefficients λ1 , λ2 , λ3 , λ4 , λ5 , and λ6 are
employed to weigh the respective loss terms. The values of
these coefficients are determined based on the initial scales of
the corresponding loss terms, avoiding the need for extensive
tuning. Moreover, they are set as 1, 1, 1, 10, 1, and 1,
respectively. The overall loss consists of three terms, i.e., ReID, MFD, and PFD losses. First, Re-ID loss aims to encourage
the backbone to learn identity-discriminative features representation for VIS-NIR Re-ID. However, due to the modality and
posture discrepancies between heterogeneous images, the identity features inevitably contain modality and posture factors,
which impact the subsequent identity matching. To address
this issue, we introduce MFD and PFD losses to decouple the
two identity-unrelated factors, thereby learning pure identity
features.
V /N
During inference, we concatenate the E V /N , E id , and E idS
to represent the final identity representations of images, which
is formulated as follows:


E IV = E V ||E idV ||E idS


E IN = E N ||E idN ||E idS
(20)
where E IV and E IN denote the final identity representations
of VIS and NIR images, respectively. [·||·] denotes concatenate operation in feature-dimension. Subsequently, we employ
cosine distance to measure the similarity of identity representations.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Distribution aligned semantics adaption for lifelong person re-identification.pdf' - | sed -n '189,447p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3 Methodology
3.1 Preliminary
T

Problem formulation. In LReID, a stream of datasets D = {D(t) }t=1 are used for
model training sequentially. Each dataset is regarded as an application domain during
(t)
(t)
the lifelong deployment of the Re-ID model. At the t-th step, D(t) = {Dtrain , Dtest }
(t)

(t)

(t)

n(t)

(t)

contains the training set and the testing set. Dtrain = {(xi , yi )}i=1 , where xi and
(t)
(t)
(t)
yi ∈ [1, nid ] are the i-th person image and its identity label, respectively, where nid
is the total number of identity classes in the t-th training set, and n(t) is the total
number of samples in the t-th training set. At the t-th training step, the model G (t)
(t)
as feature extractor and a classifier g (t) are updated with Dtrain . Since the identity
classes for training and testing are disjoint, the classifier is discarded and the trained
model is used for evaluation. The model G (t) is expected to perform well on all testing
(t′ ) t

sets of seen domains, i.e., {Dtest }t′ =1 , respectively. The extracted feature of an input
(t)
image xi using G (t) is denoted as fi ∈ Rd , where d denotes the feature dimension.
For notation simplicity, we omit the subscript i in the following.
Knowledge distillation paradigm. The serious challenge of diverse variation across
different datasets obliges previous LReID methods [14–16] to save a small number of
exemplars from previous training steps and use them for knowledge distillation [17, 23].
In the knowledge distillation paradigm, G (t−1) and g (t−1) are used to maintain the
acquired knowledge. The distillation is performed between the old G (t−1) , g (t−1) and
the evolving G (t) , g (t) . We denote the extracted feature of an exemplar image xr using
(t)
G (t) as f r (t) ∈ Rd , and the weight parameter of the classifier g (t) as ϕ(t) ∈ RN ×d ,
P
(j)
t
where N (t) = j=1 nid . The knowledge distillation loss is formulated as follows:
Lkd = −



⊤
⊤
1 X
σ f r (t−1) · ϕ(t−1)
log σ f r (t) · ϕ(t)
,
|B| r

(1)

x ∈B

where |B| denotes the training batch size, σ is the sof tmax function. For dimension
consistency of ϕ(t−1) and ϕ(t) in Eq. 1, either only old classes of g (t) are used, or g (t−1)
5

Domain (t-1)
( t −1)

Conv

SA Module

BN

Conv

……

id

…

( t −1)

( t −1)

SAl

Share

BN l

Initialize

Share

Re-Initialize

Domain (t)
(t )

Conv

SA Module

BN

Conv

……

id

…

(t )
l

SA

BN

(t )
l

Fig. 2: The framework of DASA. The acquired robust and general human semantics from pre-training are used as shared knowledge, which is kept in the frozen Conv
layers. At each training step, we adapt the acquired knowledge from the pre-training
to application domain by tuning BN layers and adopting the lightweight Semantics
Adaption (SA) modules. During the lifelong evolution process, the previously learned
BN and SA are used for initialization in the upcoming domain for forward knowledge
transfer, while the old classifier can be discarded without increasing storage burden.
is expanded by using G (t) to calculate the center of the new classes before the new
training step. New samples can also be used for distillation [13, 17]. Previous LReID
works [14–16] all follow the knowledge distillation paradigm, but the significant task
discrepancy between classification and Re-ID makes them suboptimal. Due to privacy
issues, the training data from previous steps should not be available anymore.

3.2 Distribution Aligned Semantics Adaption
With the prevalent trend of pre-training in Re-ID, we advocate taking advantage of
pre-trained models for effective lifelong evolution across various domains. In the new
paradigm, we propose to utilize the acquired robust and general human semantics
from pre-training as shared knowledge. At each training step, our goal is transformed
to adapt the acquired semantic knowledge from the pre-training domain to the target application domain. Two problems need to be addressed, that is, the domain
distribution differences between pre-training and application, and how to effectively
utilize learned human semantics. To this end, we propose to adopt efficient distribution
alignment and lightweight semantics adaption, respectively.
Distribution alignment. Batch Normalization (BN) as a widely-used technique in
CNN can normalize the features of each domain to the same reference distribution
by domain-specific normalization statistics. It has shown its advantages in regularizing models and improving their generalization ability [27, 42]. Inspired by previous
works, we maintain multiple sets of domain-specific BN layers to estimate the feature
distribution statistics. We tune BN in each application domain so that the outputs
6

of the updated BN layers exhibit a similar distribution to the pre-training domain,
and the remaining frozen pre-trained Conv layers can receive stable input activations.
In this way, the learned human semantics can be better used, which improves the
discriminability of the model in each application domain.
Concretely, the data D(t) of each domain follows a domain-specific Gaussian distri2
bution N (µ(t) , σ (t) ). At training time, the BN at each layer l collects domain-specific
(t)

(t) 2

batch statistics (µl , σl

) of input feature maps, while updating the corresponding

(t)
(t) 2
moving averages (µl , σ l ) to approximate the domain distribution statistics. The

calculation of BN is formulated as follows:
(t)

(t)

f − µl
(t)
BN(fl ) = γl ql
+ βl ,
(t) 2
σl + ϵ
(t)

(2)

where fl is the input feature map at layer l, γl and βl are learnable affine parameters
for linear transformation, and ϵ > 0 is a small constant to avoid the numerical problem.
After training, we can obtain the statistics and affine parameters for each domain,
oLBN
n
(t)
(t) 2
, where LBN is the number of BN layers in the
i.e., BN (t) = µl , σ l , γl , βl
l=1
CNN model. To achieve the forward transfer of knowledge from the learned domain to
the next, we use the BN (t−1) of the previous domain as the initial for training BN (t)
in the new domain.
Semantics adaption. Having compensated for the distribution differences between
the pre-training domain and the application domain, it is time to consider how to
efficiently utilize the acquired human semantic knowledge in the application domain.
The learned human semantics from pre-training are regarded as robust and general
knowledge for Re-ID, so we freeze the pre-trained Conv layers to keep the shared
knowledge. Although directly utilizing these general semantics for Re-ID has achieved
decent results, it may contribute to inferior performance when the application domain
shows great complexity in scenes and pedestrians (as shown in Tab. 6). Therefore, it
is necessary to adapt the learned general semantics to specific application domains.
To this end, we introduce the lightweight Semantics Adaption (SA) module after
each frozen Conv layer to aggregate and refine learned semantics effectively. For the
sake of efficiency, it could be a good choice to implement SA as one depth-wise Conv
layer. Assuming M is the channel dimension of the input feature map, the depthwise Conv layer adopts M kernels and each for one channel of the input feature map,
respectively. One of the direct designs is using the 1 × 1 depth-wise Conv layer.
However, we find it has no effect on performance improvement and even deteriorates
the discriminative ability of the model (as shown in Fig. 5). It makes sense since there
is no interaction in the spatial dimension to achieve semantics adaption. Intuitively,
the information interaction across spatial is vital to capture discriminative semantics,
such as body figures, and improve the robustness of learned semantics in complex
scenes. Therefore, we propose to apply a relatively large kernel size to aggregate and
refine the general semantics. Note that since the SA module performs convolutional
operation at each channel separately, the increase in kernel size would not result in a

7

significant increase in parameters. In our experiments, we find that the kernel size of
5 × 5 can achieve a good balance between effectiveness and efficiency.
There are some differences when comparing the SA module with LoRA [43], which
adds trainable pairs of rank decomposition matrices in parallel to existing weight
matrices. Technically, LoRA updates the original weights and changes the calculation operator, adapting the model for different tasks. Differently, the SA module
aims to leverage general human semantics, improving the Re-ID discriminative ability
of the model in different application domains. Adapting the general human semantics to more representative identity semantics represents a process from coarse to
fine, showing great ability to evolve the model for lifelong application. Incorporating
reparametrization techniques like LoRA [43] can be a prospective future work.
After training in each domain, we can obtain a set of domain-specific SA modules,
n
oLConv
(t)
(t)
i.e., SA(t) = wl
, where wl denotes the parameters of the SA module at
l=1
layer l, and LConv is the number of Conv layers in the CNN model. Similarly to our
distribution alignment design, to achieve the forward transfer of knowledge from the
learned domain to the next, we use the SA(t−1) as the initial to train SA(t) effectively.

3.3 Overall Pipeline
The framework is shown in Fig. 2. At each training step, we use the basic identity
classification loss Lid [33] for supervision. After training the model, we can obtain the
domain-specific BN (t) and SA(t) as introduced above. Both of them are lightweight
and stored for taking advantage of the learned semantics at the inference stage. Note
that the extra parameters introduced per domain are not significant compared with
other competitors as shown in Fig. 4. During testing, according to the camera information of the person image, corresponding BN (t) and SA(t) are used. With the help
of BN (t) and SA(t) , the model can efficiently evolve and adapt to new application
domains, making it effective and practical for new deployment and wide usage in
real-world scenarios.


hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Diverse Representations Embedding for Lifelong Person Re-Identification.pdf' - | sed -n '267,664p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
III. P ROPOSED M ETHOD
A. Overview
To preserve old knowledge while adapting new information
in the face of limiting old task datasets and task-level domain gaps, we propose a Diverse Representation Embedding
(DRE) framework based on improved transformer backbone
for LReID. The input images are mapped into three representations using a transformer-based backbone. Three representations, including a primary embedding representation
and multiple auxiliary embedding representations, are operated
by Adaptive Constraint Module (ACM) to maintain richness
and discrimination of each instance. Knowledge Update (KU)
and Knowledge Preservation (KP) are proposed at the tasklevel layout by introducing the adjustment model and the
learner model. We define the problem of lifelong person reidentification in this section. Specifically, continuous person
re-identification datasets E = {E t }Tt=1 are collected in dift
t
ferent environments. E t = {Etrain
and Etest
} indicates the
training datasets and corresponding test datasets in t-th step,
respectively. The training datasets Etrain are fed into the
model in sequence for training. To store a limited number of
instances from each old task dataset, we establish a memory
buffer M t .
The DRE consists of an adjustment model Φta and a learner
model Φtl to extract diverse representations of each instance.
The adjustment model maintains the priority of old tasks,
and the learner model is responsible for adaptive learning of
information for new instances. An adjustment model and a
learner model both use improved transformer backbones. ϕta
and ϕtl serve as classifier heads for the adjustment and learner
models, providing logits of each instance for recognition, as
shown in Figure 2. The outputs of old instances xo and new
instances xn through adjustment model Φta and learner model
Φtl are as follows:
Pln , (Anl )S = Φtl (xn ); Pan , (Ana )S = Φta (xn )
Plo , (Aol )S = Φtl (xo );

Pao , (Aoa )S = Φta (xo )

(1)

Where Pa and Pl ∈ RB×D denote the primary embedding
representations from the adjustment and learner models. (Aa )S
and (Al )S ∈ RB×D are auxiliary embedding representations
from the adjustment and learner models. S is the number of
auxiliary embedding representations.
B. Transformer-based structure
Currently, LReID methods based on CNN backbone force
the introduction of multiple branches to obtain diverse representations that are independent and uncontrollable from each
other in sequential task datasets. Multiple class tokens of
transformer better fit our idea of diverse representations. Thus,
we first propose a transformer-based backbone to generate
diverse representations of each instance in LReID, as shown

4

Fig. 2. An overview of the proposed DRE for LReID. We first explore a pure transformer to LReID and equip it with an adjustment model and a learner model.
The adaptive constraint module learns rich and discriminative representations. We then introduce knowledge update and knowledge preservation strategies
based on diverse representations that collaboratively achieve knowledge interaction to preserve old knowledge while adapting to new information.

in Figure 2 (Transformer-based structure). Unlike the universal
transformer structure, we introduce multiple class tokens and
maximum embedding. For a given image x, we split x into N
fixed-size patches xi |i = 1, 2, ..., N , and then concatenate on
the first dimension.
Z0 = [F (x1 ); F (x2 ); ...; F (xN )]

(2)

where Z0 ∈RB×N ×D . F is a linear projection mapping the
patches to D dimensions. [ ; ; ] denotes concatenation on the
first dimensions.
Maximum Embedding: We further design maximum embedding (ME) to focus on critical regions of the input sequence
and reduce background interference. ME utilizes a one-hot
method [48] along the first dimension of the original input
sequence Z0 to query the index θ with the maximum value.
The prominent patch embedding matrix is obtained by setting
the value corresponding to the index θ to 1 and filling the
other positions with 0. The embedding matrix can be directly
mapped on Z0 to obtain the prominent embedding region.
Then the prominent embedding region argmaxZ0θ is embedθ

ded to the original input sequence Z0 by Hadamard product

to obtain the maximum embedding ME , and the maximum
embedding is defined as:
ME = argmaxZ0θ × Z0

(3)

θ

Multiple Class Tokens: We introduce multiple class tokens
embedded in transformer input sequence. Multiple class tokens
and patches are concatenated on the first dimension and sent
to the transformer for training on the dataset, which are
converted into multiple overlapping representations. Multiple
overlapping representations provide multiple high-level semantic information for each instance from different perspectives. The specific composition of transformer input sequence
is as follows:
(S)

Z1 = [x0cls ; ...; xcls ; ME ]

(4)

Where Z1 ∈RB×(N +S+1)×D . The input sequence Z is expressed as:
Z = Z1 + P
Where P ∈RB×(N +S+1)×D is position embedding.

(5)

5

C. Adaptive constraint module
PatchKD [18] exploits local adaptively-chosen patches to
pilot anti-forgetting distillation. However, when encountering
large occluded regions, patches inevitably introduce non-body
areas that weaken model performance in both old and new
tasks. Therefore, we propose an adaptive learning module to
enable multiple overlapping representations to retain discriminative abilities and rich body information of each instance.
It plays a crucial role in preserving old knowledge while
adapting to new tasks. As shown in Figure 2.
Specifically, the generated diverse representations are divided into a primary embedding representation and multiple auxiliary embedding representations. Multiple auxiliary
embedding representations with discrimination are beneficial
for improving the adaptive capacity of the LReID model.
Therefore, we implement the minimization of overlapping
elements among multiple auxiliary representations embedding
by orthogonal. The orthogonal loss function is defined as:
B

Lort =

S

1 1 XX
(Cos < (Anl )1 , (Anl )S > ∗I)
B S i=1 j=1

(6)

Where Cos < ., . > donates the cosine similarity, I ∈ RS×S
is a unit matrix, (Anl )S indicates auxiliary embedding
representations from learner model in new instances, S is the
number of auxiliary embedding representations.
The primary embedding representation is responsible for
learning rich body information. Given that a single embedding
representation inadequately captures rich body information,
we adaptively integrate multiple discriminative auxiliary
embedding representations into a primary embedding
representation. This ensures that the model can better
learn fine-grained and more informative representations
for classification. The integrated primary embedding
representation is formulated as:
ωS = Cos < P, (Anl )S >
P̂ = P +(1 − ω1 ) ∗ (Anl )1 + ... + (1 − ωS ) ∗ (Anl )S

(7)

Where, ωS ∈[0,1] donates cosine similarity between the
primary embedding representation and multiple auxiliary
embedding representations. P , and P̂ are primary
embedding representation, and integrated primary embedding
representation, respectively. Note that symbols P , (A)S ,
and P̂ use subscripts to indicate data flow direction in the
following text. Their superscripts represent the flow of data
(n from new instances, o from old instances), and the lower
table represents the source of the model (a from adaptation
model, l from learner model). Additionally, other symbols
are defined according to specific contexts.

D. Knowledge Update
In this section, we propose a Knowledge Update (KU) strategy that enhances the learner model’s adaptive learning ability
for new information under the adjustment model prior to new
instances, including adaptive learning, logit-level distillation
(LLD), and slow update.

Adaptive learning: We introduce cross-entropy loss and
triplet loss to facilitate the adaptability of the learner model.
The primary embedding representation of new instances are
classified by classifier head of learner model is ϕtl . Crossentropy loss is used to calculate identification loss. The
identification loss Lid is formulated as:
B

1 X n
(y )i log(ϕtl ((P̂ln )i ))
Lid = −
B i=1

(8)

Where (y n ) is the label for new instances.
Meanwhile, we use metric learning to optimize the learner
model on the primary and auxiliary embedding representations
of new instances, and the triplet loss is defined as:
B

1 X
max(d((fna )i , (fnp )i ) − d((fna )i , (fnn )i ) + m, 0)
B i=1
(9)
Where d(· , ·) denotes the euclidean distance. m is set to 0.0 as
default. (fna ,fnp ,fnn ) is a triplet of the anchor sample, positive
sample and negative sample of [P̂ln ;(Anl )1 ;...;(Anl )S ].
The base loss function for training the learner model on new
instances is formulated as:

Lntrip =

Lbase = Lid + Lttrip + Lort

(10)

Logit-level Distillation: Base loss only considers adaptability
of the learner model to new tasks, ignoring the consistency
relationship between the adjustment model and the learner
model. Guided by the feature distribution of the adjustment
model, we further introduce logit-level distillation, focusing
on mining consistency information from new samples. In particular, we employ a kullback-leibler divergence to minimize
the logit-level distribution variance of primary embedding
representation between the adjustment model and the learner
model for new samples. ϕta and ϕtl serve as classifier heads
for the adjustment and learner models. Logit-level distillation
loss is indicated as follows:
B

LLLD =

1 X
KL(ϕta ((P̂an )i )/τ )||ϕtl ((P̂ln )i )/τ ))
B i=1

(11)

Where KL(.||.) refers to a kullback-leibler divergence, and τ
is a hyperparameter called temperature [16].
Slow Update: The learner model tends to favor the distribution
of new tasks, away from fixed-parameters adjustment model
in terms of distribution. If we forcibly align the consistency
of the adjustment and learner models to mitigate the risk of
catastrophic forgetting, it will inevitably weaken adaptability
of the learner model to new tasks. Therefore, we transform the
frozen adjustment model into a dynamic adjustment model to
improve the learner model’s adaptability to new tasks. Specifically, we introduce an exponential moving average (EMA) [49]
to gradually update the adjustment model parameters using
the learner model at each iteration step. The parameters for
updating the adjustment model are formulated as:
Φta = k ∗ Φta + (1 − k) ∗ Φtl
Where k is a constant and set to 0.996.

(12)

6

E. Knowledge Preservation
Knowledge update effectively enhances the adaptability
of the learner model to new tasks. However, it biases the
distribution of the adjustment model toward new instances,
ignoring the risk of forgetting old tasks. Therefore, we propose
knowledge preservation (KP) strategy preserve old knowledge
operated by representation-level alignment and logit-level supervision in limited old task datasets while guaranteeing the
adaptive learning information capacity of the LReID model
in terms of old instances from memory buffer M t , including
representation-level alignment (RLA) and logit-level supervision (LLS).
Representation-level Alignment: Since the LReID task is
essentially a cross-camera matching problem, our goal is
to learn significant representation rather than classification
scores. We argue that the distribution between the adjustment
model and the learner model under diverse representations of
old instances should remain consistent to prevent forgetting
old tasks. To achieve this goal, we first utilize triplet loss to
optimize the primary and auxiliary embedding representations
of the learner model obtained from old instances. The triplet
loss is expressed as follows:

Where σ is softmax function.
The total loss function is formulated as:
L = Lbase + LLLD + LF LA + LLLS

(17)

TABLE I
DATASET STATISTICS OF THE LR E ID BENCHMARK . S INCE THE SAMPLING
PROCEDURE RESULTS IN THE NUMBERS OF TRAIN ID S BEING ALL 500,
THE ORIGINAL NUMBERS OF ID S ARE LISTED FOR COMPARISON . ’-’
DENOTES THAT THE DATASET IS NOT USED FOR TRAINING

Type

Datasets

Scale

Train IDs

Test IDs

Seen

Market [50]
CUHK-SYSU [51]
DukeMTMC [52]
MSMT17 V2 [53]
CUHK03 [54]

large
mid
large
large
mid

750
2900
1110
3060
700

Unseen

VIPeR [55]
GRID [56]
CUHK02 [57]
Occ Duke [58]
Occ REID [59]
PRID2011 [60]

small
small
mid
large
mid
small

500(751)
500(942)
500(702)
500(1041)
500(700)
−
−
−
−
−
−

316
126
239
1100
200
649

B

1 X
max(d((foa )i , (fop )i ) − d((foa )i , (fon )i ) + m, 0)
B i=1
(13)
Where (foa ,fop ,fon ) is a triplet of anchor sample, positive
sample and negative sample of [P̂lo ;(Aol )1 ;...;(Aol )S ].
Then, we establish a consistent loss function for
the primary and auxiliary embedding representations.
Specifically, Fa =[P̂ao ;(Aoa )1 ;...; (Aoa )S ]∈RB×(S+1)×D and
Fl =[P̂lo ;(Aol )1 ;...;(Aol )S ] represent diverse representations
concatenation of the adjustment model (Φta ) and the learner
model (Φtl ), respectively. We calculate cosine similarity
between diverse representations of the adjustment and learner
model and minimize the cosine similarity matrix by using the
L1 norm to preserve old knowledge. Consistent loss Lcon is
defined as:
Lotrip =

Lcon = ||Cos < Fa , Fa >, Cos < Fl , Fl > ||1

(14)

Finally, the representation-level alignment loss is obtained
through a linear combination of Lm
trip and Lcon . It is formulated as:
LF LA = Lm
(15)
trip + Lcon
Logit-level Supervision: Representation-level alignment only
accounts for the relationship between the adjustment model
and the learner model at the feature level while ignoring the
interdependencies at the logit level. Therefore, we propose a
logit-level supervision loss to bolster the preservation of old
knowledge as a complement to representation-level alignment.
Specifically, we leverage the logits from the primary embedding representation in the adjustment model as a supervisory
signal to constrain the learner model. Logit-level supervision
loss, named LLLS , is defined as:
B

LLLS = −

1 X
σ(ϕta ((P̂ao )i ))log(σ(ϕtl ((P̂lo )i )))
B i=1

(16)


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf' - | sed -n '252,616p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3. Methodology
3.1. Preliminaries
In the typical DG re-ID setting, we assume 𝐾 labeled source domains
{
}
{
}𝐾
 = 1 , 2 , … 𝐾 , where each source domain 𝑘 = 𝑘 , 𝑘 𝑘=1
consists of 𝑁𝑘 Image–label pairs. The label spaces for 𝐾 source sets
are disjoint. The core objective of DG re-ID is to learn a model parameterized by 𝜃 that is capable of predicting accurate identity labels on
unseen target domain  .
3.2. Overview
The overall process of the proposed framework is illustrated in Fig.
1, which includes a feature extractor 𝑓𝜃 and constructs an individual
memory-based classifier for each source domain. This paper employs
a domain-specific sampling strategy where each mini-batch contains
samples from the same domain. For a source domain 𝑘 with 𝑃𝑘
identities, the domain-specific memory 𝑘 has 𝑃𝑘 slots, where each
slot saves the feature centroid of the corresponding domain. 𝑘 is
initialized with features averaged of the corresponding identity. In
the forward propagation, the similarities between features produced
by 𝑓𝜃 and centroids of 𝑘 are used to compute the identification
loss. To maximize the dissimilarity of representations, the DED loss is
proposed to push apart features from one domain from those of other
domains. In the backward propagation, the network is optimized by
the combination of identification loss and DED loss. At the end of each
training iteration, the memory modules are updated with the features
in the current mini-batch.
3.3. Domain experts decoupling

2.3. Diversity via feature decorrelation

During the forward propagation, any image 𝑥𝑘𝑖 (where 𝑥𝑘𝑖 denotes
the 𝑖𝑡ℎ image sampled from the 𝑘𝑡ℎ dataset 𝑘 ) is passed through the
feature extractor to obtain the L2-normalized feature 𝑓𝑖𝑘 = 𝑓𝜃 (𝑥𝑘𝑖 ).
We calculate the non-parametric memory-based identification loss as
follows:
(
)
𝑁𝑘
∑
exp 𝑓𝑖𝑘 𝑐+𝑘 ∕𝜏
𝑘 = −
log ∑
(1)
(
)
𝑃𝑘
𝑘 𝑐 𝑘 ∕𝜏
𝑖=1
exp
𝑓
𝑖
𝑗
𝑗=1

Emerging evidence suggests that the correlations between features
affect or even impair the model prediction. Feature decorrelation is
a promising way to separate the explanatory factors of variations
and learn representations that are more resilient to complex variants.
Recent works realize this goal from various perspectives. For instance,
DecAug (Bai et al., 2021) decomposes the high-level features into
category-related and context-related features to deal with the diversity shift. RaMoE (Dai et al., 2021) proposes a decorrelation loss by
minimizing the L2-norm of point-wise product between features from
various domains. ABD-Net (Chen et al., 2019) incorporates a spectral
value difference orthogonality (SVDO) regularization to reduce the
correlation between learned features by constraining the condition
number of the weight Gram matrix. DACS (Yang et al., 2024) achieves
decorrelation by enlarging the distributional discrepancies between the
original and generated data with the Wasserstein distance.
Different from the above methods that require complex architectures or high computational costs, we investigate the memory-based
feature decorrelation with a novel Domain Experts Decoupling (DED)
loss, which provides a flexible and low-cost solution to enrich diversity by minimizing the correlation among the source domain experts.
Unlike DecAug (Bai et al., 2021) and ABD-Net (Chen et al., 2019),
which rely on explicit feature decomposition or matrix regularization,
our method achieves feature decorrelation by enforcing orthogonality
among the features stored in memories. This in vivo regularization not
only reduces computational overhead but also enhances adaptability
to diverse domain shifts. Moreover, our motivation is different from

where 𝑐+𝑘 refers to each centroid of the positive class prototype corresponding to 𝑓𝑖𝑘 , and 𝑐𝑗𝑘 denotes the centroid of one class prototype,
which could be either positive or negative. Here, 𝜏 is the temperature
factor that controls the scale of similarities. The memory-based identification loss computes the average loss across 𝐾 source domains, where
𝑘 is the loss for domain 𝑘, aiming to classify 𝑓𝑖𝑘 into its own identity
and the total identification loss is a combination on all source domains:
1 ∑

(2)
𝐾 𝑘=1 𝑘
Next, in order to impose inter-domain diversity and keep source
domains’ discriminative power, we propose the Domain Experts Decoupling loss to enable the feature disentanglement among various source
domains. Let 𝑓 𝑘 and 𝑓 𝑗 (𝑗 ≠ 𝑘) be representations from any two source
domains (the subscript of the image index is omitted for simplicity), we
first compute their cross-correlation matrix along the batch dimension:
∑ 𝑘 𝑗
𝑏𝑓 𝑓
𝑘𝑗 = √
(3)
√
∑ ( 𝑘 )2 ∑ ( 𝑗 )2
𝑏 𝑓
𝑏 𝑓
𝐾

𝑀 (𝜃) =

3

Neural Networks 189 (2025) 107602

J. Jia et al.

Fig. 1. The overall framework of DiverseReID. The 𝑘th memory bank serves as the expert of the domain 𝐾 and is learned with identification loss 𝑀 and Domain Experts
Decoupling loss 𝐷𝐸𝐷 to achieve inter-domain style variance. The DSH module is exploited to induce intra-domain diversity.

where 𝑏 index batch samples.  is a 𝑑 ×𝑑 square matrix with 𝑑 denoting
the dimension of features, and with values ranging between −1 and
1. The Frobenius norm is the sum of the squares of all elements in
the matrix. Then, the DED loss is introduced to minimize the squared
Frobenius norm of :
∑ ‖
‖2
𝐷𝐸𝐷 (𝜃) =
(4)
‖𝑑𝑖𝑎𝑔 ‖
‖
‖𝐹

from manually designed distributions require careful hyper-parameter
tuning and the optimal combinations of hyper-parameters are timeconsuming to acquire; (2) the fixed value of the parameters may render
synthesized styles that remain closely related to existing ones and the
insufficient diversity may notably jeopardize the generalization.
In light of this, we propose to replace this fixed mixing with a
novel Dynamic Style Hallucination module by mining the relations
across sample pairs in an adaptive manner, as shown in Fig. 2(b). Let
𝐹 ∈ 𝑅𝐶×𝐻×𝑊 be the feature maps output by a certain layer, where 𝐻
and 𝑊 indicate spatial dimensions, and 𝐶 is the number of channels.
Following existing works (Nam et al., 2021; Nuriel et al., 2021; Zhou,
Yang, Qiao et al., 2020), we adopt the channel-wise mean and standard
deviation to represent the style distribution:

𝑑𝑖𝑎𝑔=[1,𝑑]

By minimizing the correlation among the source domain experts, the
embedding vectors from each domain are decorrelated and encouraged
to be dissimilar, which can bring about a substantial amount of data
diversity and greatly promote the generalization performance. Taking
into account all the discussions above, we have the following overall
optimization objective:
(𝜃) = 𝑀 (𝜃) + 𝜔𝐷𝐸𝐷 (𝜃)

1 ∑∑
𝐹
(8)
𝐻𝑊 ℎ=1 𝑤=1 ℎ𝑤
√
√
𝐻 ∑
𝑊
√ 1 ∑
(
)2
𝜎 (𝐹 ) = √
𝐹 − 𝜇 (𝐹 )
(9)
𝐻𝑊 ℎ=1 𝑤=1 ℎ𝑤
The DSH module constructs a novel randomized style by interpolating between the styles of 𝐹 and a sampled style 𝐹 ′ . Various sampling
strategies have been proposed to determine which two instances are
to be mixed, like random sampling within one batch regardless of
domain labels, cross-domain sampling that selects from two arbitrary
domains (Zhou, Yang, Qiao et al., 2020), or farthest point sampling
(FPS) (Qi, Yi, Su, & Guibas, 2017) that select basis styles which are
most dissimilar with the current samples. The influence of sampling
strategies will be discussed in Table 9. In more detail, the generated
styles ̂
𝜇, 𝜎̂ ∈ 𝑅𝐶 can be formed as:
( )
𝜇̂ = 𝜆𝜇 (𝐹 ) + (1 − 𝜆) 𝜇 𝐹 ′
(10)
( ′)
𝜎̂ = 𝜆𝜎 (𝐹 ) + (1 − 𝜆) 𝜎 𝐹
(11)
𝐻

𝑊

𝜇 (𝐹 ) =

(5)

where 𝜔 is a positive constant trading off the importance of the first and
second terms of the loss. The overall loss is backpropagated through the
network via gradient descent, where 𝛾 stands for the learning rate:
(
)
𝜃 ′ ← 𝜃 − 𝛾 ∇𝜃 𝐿 (𝜃)
(6)
Memory Updating Strategy Our DiverseReID framework maintains
an individual memory for each source domain. At the end of each
iteration, we compute the average of all features in the current minibatch and combine it with the encoded features belonging to class
prototypes 𝑐+𝑘 , adjusting the balance between them using a parameter
𝜂:
1 ∑
𝑐+𝑘 ← 𝜂𝑐+𝑘 + (1 − 𝜂)
𝑓𝜃 (𝑥)
(7)
|𝑘 |
| | 𝑥∈𝑘
where 𝜂 ∈ [0, 1] controls the momentum updating rate. 𝑘 stands for the
samples belonging to the 𝑘𝑡ℎ domain and ||𝑘 || denotes its cardinality,
i.e., the number of samples for the 𝑘𝑡ℎ domain in the current mini-batch.

where 𝜆 is a mixing coefficient that determines the trade-off of each
term, which is usually sampled from a Beta distribution (Zhang et al.,
2018) parameterized by pre-defined 𝛼. Other works resort to sampling
from Dirichlet distribution (Shu, Cao, Wang, Wang, & Long, 2021),
Gaussian (Li et al., 2022) or Uniform distributions (Nam et al., 2021;
Tan et al., 2023) to deduce the interpolation weights, but there remain
some parameters that need to be manually set.
With this intuition, we put forward a flexible solution to derive the
value of weights 𝜆 in an adaptive manner. Specifically, we first leverage the Wasserstein distance to measure the distribution discrepancy

3.4. Dynamic style hallucination
It has been widely acknowledged that style diversifying is critical to
enhancing DG performance. A prevalent approach is to mix up existing
styles to generate novel styles (Zhang et al., 2018), with combination
weights sampled from pre-defined Beta distributions, as shown in Fig.
2(a). This can likely act as a bottleneck towards fully harnessing the
potential of style diversification in: (1) the mixing weights sampled
4

Neural Networks 189 (2025) 107602

J. Jia et al.

Fig. 2. Conceptual overview of DSH. (a) Existing stylization methods typically sample mixing coefficients from Beta distribution and require careful tuning of parameter 𝛼; (b)
the proposed Dynamic Style Hallucination formulates the weights in a dynamic manner.

between 𝐹 and 𝐹 ′ :
(
) ‖
( )‖2
𝑑𝑤𝑎𝑠𝑠 𝐹 , 𝐹 ′ = ‖𝜇 (𝐹 ) − 𝜇 𝐹 ′ ‖ +
‖
‖2
( )2
( )
𝜎(𝐹 )2 + 𝜎 𝐹 ′ − 2𝜎 (𝐹 ) 𝜎 𝐹 ′

Algorithm 1: Training Procedure of DiverseReID
(12)

Subsequently, the reciprocal of 𝑑𝑤𝑎𝑠𝑠 + 1 is computed as the initial
weight coefficient, where +1 is to make the calculation more stable.
Afterwards, the softmax is leveraged to ensure the value of 𝜆 lies within
the range [0,1] and sums up to 1:
(
)
1
𝜆 = 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥
(13)
1 + 𝑑𝑤𝑎𝑠𝑠
Finally, a new hallucinated style is synthesized by combining the
styles of 𝐹 , including its mean and variance, with the mean and
variance of a sampled style 𝐹 ′ , as shown in Eqs. (10) and (11). This
newly constructed style is then used to replace the original style 𝐹 :
(
)
𝐹 − 𝜇 (𝐹 )
𝐷𝑆𝐻 (𝐹 ) = 𝜎̂ ⋅
+ 𝜇̂
(14)
𝜎 (𝐹 )
With 𝑑𝑤𝑎𝑠𝑠 , the convex combination of two instances becomes adaptive. A small value of 𝑑𝑤𝑎𝑠𝑠 indicates that the sampled 𝐹 ′ is likely to
be very close to 𝐹 and only brings marginal diversity, thus a smaller
weight should be allocated to it. On the other hand, large distances
imply that 𝐹 ′ is mostly dissimilar with 𝐹 and contributes greatly to
enriching the style diversity, which therefore should be given more
weight. By means of this, the need for manual hyper-parameter finetuning can be eliminated, and more diverse styles that are different
from existing samples can be covered and generated. During training,
the parameter-free DSH module is inserted at the first stage of the CNN
models and adds very little training overhead. In the testing phase, the
DSH is removed from the feature extractor, which makes our network
simple and efficient for implementation. The pipeline of the proposed
DiverseReID framework is summarized in Algorithm 1.

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18

{
}𝐾
Input: Image–label pair in source domain 𝑘 , 𝑘 𝑘=1 , the number of source
domains 𝐾, ImageNet pre-trained model 𝜃, weighting parameter for loss
terms 𝜔, learning rate 𝛾, maximum number of epochs 𝑇 , momentum
updating rate 𝜂.
Output: The best model 𝜃.
for 𝑖𝑡𝑒𝑟𝑠 = 1 to 𝑇 do
{
}𝐾
Obtain batch_data from 𝑘 , 𝑘 𝑘=1 ;
for 𝑘 = 1 to 𝐾 do
Obtain feature 𝑓 𝑘 from model 𝜃 equipped with DSH;
Initialize the domain-specific memory 𝑘 ;
Compute and accumulate loss 𝑀 (𝜃) on 𝑓 𝑘 ;
Store the extracted feature 𝑓 𝑘 into outputs_list;
end
for 𝑘 = 1 to 𝐾 do
for 𝑗 = 𝑘 + 1 to 𝐾 do
Obtain 𝑓 𝑘 , 𝑓 𝑗 from outputs_list;
Compute the cross-correlation matrix  between 𝑓 𝑘 and 𝑓 𝑗 ;
Compute and accumulate loss DED ;
end
end
(𝜃) = 𝑀 (𝜃) + 𝜔𝐷𝐸𝐷 (𝜃);
(
)
Update gradient : 𝜃 ′ ← 𝜃 − 𝛾 ∇𝜃 (𝜃) ;
∑ 𝑘
𝑘 ← 𝜂𝑐 𝑘 + (1 − 𝜂) 1
Update the prototypes stored in 𝑘 : 𝑐+
𝑓 ;
| |
+
|𝑘 | 𝑥∈
𝑘

19 end

set contains 1,801,816 images from 8000 synthetic identities. Following (Wang et al., 2020), we utilize a curated subset of 132,145
images (all 8000 identities) for training. It is noteworthy that RandPerson (Wang et al., 2020) is excluded from being used as the target domain since it only includes the training subset. To ensure the
comprehensiveness of experiment results, we adopted five evaluation
protocols. For Protocol-1, we adopt a leave-one-out setting for M,
D, C3, and MS, where one dataset is selected for testing while the
remaining datasets are used for training. For Protocol-2, we use the
leave-one-out setting that selects one dataset from M, C3, and MS as
the target set and the remaining datasets plus CS are exploited for
training. Note that the CS dataset is only used for training since it covers
1 camera. For Protocol-3, we alternately use M and D as the source
and target domains, respectively. For Protocol-4, we combine all the
images in the source domains M+C2+C3+CS for training and evaluate
it on four small-scale datasets VIPeR (Gray & Tao, 2008) et al. For
Protocol-5, we follow the protocol outlined in Yan, Li, Liao, and Qin
(2025). Specifically, RP is combined with two datasets selected from
C3, M, MS for training, while the third dataset serves as the evaluation
target set. To maximize the data volume, all the identities regardless
of original train/test splits, are utilized for training. The mean average
precision (mAP) and Rank-1 accuracy are exploited as the evaluation
metrics.


codex
方法部分已经开始显出几种不同套路：有的把“旧知识保存”改写成“预训练语义适配”，有的把“跨域泛化”改写成“域专家之间必须相互去相关”。我继续把剩下两篇终身和一篇无监督跨模态的完整机制读完。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | sed -n '150,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3. Proposed Method
3.1. Overview
The framework of Dual-level Modality Debiasing Learning (DMDL) is shown in
Fig. 2, incorporating the Causality-inspired Adjustment Intervention (CAI) module
and the Collaborative Bias-free Training (CBT) strategy. In cross-modality learning,
DMDL first iteratively matches clusters across different modalities to obtain crossmodality relationships as a kind of pseudo-label. Then, CAI employs a backdoor adjustment algorithm to implement causal intervention, which guides the model to capture causal patterns, resulting in a low-biased model. Furthermore, to avoid misleading
optimization caused by biased cues, CBT incorporates label refinement and feature
alignment with modality-specific data augmentation to jointly mitigate modality bias
across different levels. This methodology leads to modality-invariant features and a
more generalized model.
6

RGB

Clustering

Causality-inspired Adjustment Intervention

optimization

…

Pseudo
label

GMM

cluster identities

Clean
𝜔𝑖

cluster feature
Noise

update
Refined label

features

Σ
𝑃(𝑌|𝑑𝑜(𝑋))

memory 𝑚𝐼

image 1: id 3

ℒ𝑓𝑎

𝑃(𝑌|𝑋, 𝐶 = 𝑉)

𝑃(𝑌|𝑋, 𝐶 = 𝐼)

row-wise

image 6: id 1
+

Iterative Maximum
Confidence Alignment

Backbone Network

Clustering

memory 𝑚

ℒ𝑐𝑎𝑖

memory 𝑚𝑉

…

aug

or

…

column-wise

+
IR

𝑃(𝐶)

𝑃(𝑌|𝑋 = 𝑥 𝑎 )

cross-modality
cluster feature

ℒ𝑖𝑑

Collaborative Bias-free Training

Σ

weighted sum

Figure 2: The framework of the proposed DMDL. After obtaining cross-modality pseudo-labels through
Iterative Maximum Confidence Alignment, the Causality-inspired Adjustment Intervention module is implemented for causal modeling to construct a low-biased model. Then, the Collaborative Bias-free Training
strategy combines label refinement and modality alignment with data augmentation to optimize the model,
further eliminating modality bias during training.

3.2. Baseline for Two-stage USL-VI-ReID
To better illustrate the design of our method and facilitate the organization of experiments, we construct a baseline for two-stage USL-VI-ReID regarding previous works
[3, 4], which contains a single-modality pre-training stage and a cross-modality learning stage.
The first single-modality learning stage is operated in a clustering-based unsupervised learning manner. Before each training epoch, we first perform clustering on the
data from each modality and construct single-modality cluster memories, denoted as
mc , by averaging the features of each cluster. mck represents the centroid of cluster k in
modality c ∈ {V, I}, where V means visible and I is infrared. Then, we train the model
by contrastive learning on the memory center and corresponding data as follows:
exp( f xc · mc+ /σ)
Lcid = −log P
,
exp( f xc · mck /σ)

(1)

k

where f xc is the feature of the image x with the modality c, m+ is the positive cluster
representation, and σ is a temperature hyper-parameter. The single-modality model is
I
trained with LVid + Lid
+ λtri · Ltri , where Ltri is the triplet loss [31], and λtri controls

7

the weight of Ltri which dynamically changes during training.
In the second stage, we initialize cross-modality learning using the pretrained singlemodality model and adopt the clustering-based unsupervised learning pipeline. To
obtain cross-modality pseudo-labels, we propose a simple yet effective Iterative Maximum Confidence Alignment (iMCA) scheme in the baseline to quickly match the
N clusters of one modality with the M clusters of the other. Let the modality with N
clusters be denoted as C N and the other with M clusters as C M . iMCA first calculates
the cosine similarity between cluster centroids to construct an N × M similarity matrix S , where S i, j represents the similarity between the i-th cluster of C N and the j-th
cluster of C M . With this, we perform two ways of matching: row-wise and columnwise. In the row-wise matching, for the i-th row of S , we find its matched cluster index
urow
∈ [0, M] as follows:
i
urow
= arg max j S i, j .
i

(2)

The pseudo-label of the urow
i -th cluster in C M is then assigned to the i-th cluster in
C N . This operation is applied to all rows (∀i ∈ [0, N]), effectively propagating labels
from C M to C N . In the column-wise matching, the same procedure is performed for
all columns to propagate labels from C N to C M . By alternating between row-wise
and column-wise matchings across different epochs, iMCA obtains cross-modality
pseudo-labels while preventing the model from being overconfident in a certain matching. Then, cross-modality cluster memories mk are established based on unified crossmodality labels for contrastive learning as follows:
exp( f x · m+ /σ)
Lid = −log P
.
exp( f x · mk /σ)

(3)

k

Finally, the unsupervised cross-modality model is trained with Lid + λtri · Ltri .
3.3. Modeling with Causal Intervention
In this section, we first consider USL-VI-ReID from the causal view, analyzing that
spurious bias patterns are captured by traditional likelihood-based modeling through a
backdoor, whereas causal intervention is not. Based on this analysis, we then illustrate
the causal modeling in the proposed CAI module, which constructs a cross-modality
model that is insensitive to modality bias.
8


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf' - | sed -n '546,1019p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 2728ms:
III. P ROPOSED M ETHOD
A. Preliminary: Overview of Method
The overview of our DCR to achieve a trade-off between
maximizing intra-domain discrimination and inter-domain
gaps is shown in Fig. 2. The DCR model learns the old
model Φt−1 and new model Φt from (t-1)-th and t-th steps,
where Φt is inherited from Φt−1 . Φt−1 and Φt with three
parts of attribute-text generator (ATG), text-guided aggregation

4

TABLE I
T HE ATTRIBUTE CATEGORIES ARE DIVIDED INTO FOUR MAJOR
CATEGORIES , INCLUDING 12 SUBCATEGORIES . ”N O ” AND ”Y ES ”
INDICATE THE ABSENCE AND PRESENCE OF THIS CATEGORY,
RESPECTIVELY.
Type

Category

0

1

Overall

Sex

man

woman

Upper body

Short sleeved top
Long sleeved top
Long coat

No
No
No

Yes
Yes
Yes

Lower body

Trousers
Shorts
Skirt

No
No
No

Yes
Yes
Yes

Decoration

Hat
Glasses
Handbag
Sholder bag
Backpack

No
No
No
No
No

Yes
Yes
Yes
Yes
Yes

network (TGA), and attribute compensation network (ACN).
ϕt−1 and ϕt serve as classifier heads for the old and new
models, providing logits of each instance for recognition.
Additionally, we define that consecutive T person datasets
D = {Dt }Tt=1 are collected from different environments, and
establish a memory buffer M to store a limited number of
samples from each previous ReID dataset. Given an image
xti ∈Dt ∪M, we forward it to Φt−1 and Φt is as follows:
Gt−1 , AGt−1 = Φt−1 (xti ); Gt , AGt = Φt (xti ).

(1)

Where G and AG are global and attribute-wise representations, respectively.
B. Attribute-Text Generator
Due to the lack of text-image pairs in ReID datasets, we
propose an attribute-text generator (ATG) to dynamically generate text-image pairs for each instance, as illustrated in Fig.
2. Firstly, we employ a pre-trained attribute recognition model
[33] (trained on the PA100K dataset [39]) to generate attribute
categories. These categories are organized into four major
types (e.g., overall, upper body, lower body, and decoration)
with 12 subcategories, as shown in TABLE I. Then, attribute
categories are converted into text descriptions for each instance
using a predefined template. Finally, the generated text-image
pairs are fed into the text-guided aggregation network (TGA)
for further processing. In this predefined template (Fig. 2),
black font denotes fixed modifiers, while red, blue, orange, and
green fonts represent the four major types, respectively. The
template dynamically replaces the colored placeholders with
the corresponding predicted attributes. To ensure robustness
across domains, we set a high confidence threshold (0.80) for
attribute classification, thereby maintaining the reliability of
text descriptions despite potential domain variations.
C. Text-Guided Aggregation Network
To enhance robust representation capabilities for each instance, we propose a text-guided aggregation network (TGA)

to explore the global representation for distinguishing identity
information in LReID, as shown in Fig. 2 (TGA). The TGA
consists of a Contrastive Language–Image Pretraining(CLIP)
model and a parallel fusion module (PFM). Note that the text
encoder is frozen in our DCR model.
1) CLIP Model: The CLIP model is a vision-language
model that learns multi-modal representations by aligning images and text in a shared embedding space. The model employs
a contrastive learning objective to maximize the similarity
between an image and its corresponding text description. In
our work, we adapt the CLIP model, which includes an image
encoder and a text encoder, to generate both image and text
embeddings. However, unlike the original CLIP model [25],
our model introduces multiple [CLS] embeddings into the
image encoder’s input sequence. This modification enables the
model to capture diverse representations from different views
for each instance, thereby enhancing its ability to distinguish
fine-grained identity information. Specifically, the clip model
generate text embedding d∗ and image embedding [v1∗ , · · · ,
∗
, v1 , · · · , vP ] for extracting rich text and vision information.
vN
image embedding consists of multiple [CLS] embeddings [v1∗ ,
∗
· · · , vN
] for multi-view representations and patch embedding
[v1 , · · · , vP ] for local feature extraction.
2) Parallel Fusion Module: To improve the performance of
the LReID model, we propose a parallel fusion module (PFM)
to explicitly explore the interactions between image and text
embeddings, as shown in Fig. 2 (PFM). Firstly, we leverage
text embedding d∗ as query and image embedding [v1∗ , · · · ,
∗
vN
, v1 , · · · , vP ] as key and value to implement operation
with cross-attention, drop, and layer normalization, getting
text-wise representations. Similarly, in another fusion branch,
image-wise representations are obtained. Finally, image-wise
and text-wise representations perform concatenation and MLP
operations to obtain global representations Gt = {Gti |i =
1, 2, · · · , N }, focusing on whole body information. We force
multiple global representations Gt at the current step to
learn more discriminative information by orthogonal loss to
minimize the overlapping elements. The orthogonal loss can
be expressed as:
LOrt =

N
−1
X

N
X

(Gti , Gtj ).

(2)

i=1 j=i+1

Then, we utilize the cross-entropy loss LCE and triplet loss
LgTri [5] to optimize our DCR at the current task.
K

1 X
yi log((ϕt (Gt ))i ).
K i=1

(3)

LgTri = max(dgp − dgn + m, 0).

(4)

LCE =

Where K is the number of classes, and m is the margin, dgp
and dgn are the distances from positive samples and negative
samples to anchor samples in global representations, respectively. Unlike some methods [10], [13], global representations
generated by the text-guided aggregation (TGA) network
present two advantages. First, we leverage text descriptions
based on the CLIP model to enhance the discrimination
capability of global representations, allowing them to better

5

distinguish identities and adapt to new knowledge. Second,
global representations facilitate knowledge transfer, improving
the model’s generalization ability.

E. Attribute-oriented Anti-Forgetting

D. Attribute Compensation Network
We force attributes to guide the attribute compensation
network (ACN) for learning attribute-wise representations. The
ACN consists of an attribute decoder and an attribute matching
component, as illustrated in Fig. 2 (ACN).
1) Attribute Decoder: Enabling attributes to better adapt
across domains, we define multiple learnable attributes semantic information S t = {Sit |i = 1, 2, · · · , N } to learn
discriminative information. The attributes undergo a linear
layer to increase their dimensions and are then multiplied with
the text-image global representation to output fAT . Attribute
semantic information S t as queries Q, fAT as keys and values
are input into the attribute decoder, which outputs the attribute
features At = {Ati |i = 1, 2, · · · , N }. The attribute decoder
utilizes six transformer blocks (T Block) referenced from [40].
2) Attribute Matching: The attribute features At = {Ati |i =
1, 2, · · · , N } focus on multiple discriminative local details
of individuals. However, it remains unclear which views
make these attribute features more prominent in the global
representations. We propose an Attribute Matching (AM)
mechanism to select the optimal alignment between attribute
features and global representations, thereby identifying the
best view for the global representation and improving the
representational ability of the attribute features. Specifically,
we first calculate the similarity between the attribute features
At = {Ati |i = 1, 2, · · · , N } and global representations
Gt = {Gti |i = 1, 2, · · · , N } from different views. Based on
this matching process, we determine the index of the most
similar global representation as follows.
k = argmax(< Ati , Gt >).

(5)

where <, > represent cosine similarity. The argmax operation
selects both the maximum similarity value and its corresponding index. We identify the most similar index between each
attribute feature Ai |i = 1, 2, · · · , N and the global representations G, and then combine the attribute feature Ai and
with its corresponding global representation Gtk (where k is
the matched index) to obtain the attribute-wise representations
AGt = {AGi |i = 1, 2, · · · , N }. This process is defined as:
AGti = Ati + Gtk .

(6)

We leverage the triplet loss to align attribute-wise representations with identity at the current step, assisting in global
representations to distinguish similar identities.
LlTri = max(dp − dn + m, 0).

datasets to minimize inter-domain gaps for better knowledge
transfer.

(7)

where, dlp and dln are the distances from positive samples
and negative samples to anchor samples in attribute-wise
representations, respectively. In this paper, attribute-wise representations that contain specific information of individuals
assist global representations in distinguishing similar identities for maximizing intra-domain discrimination. Meanwhile,
attribute-wise representations as a bridge across increasing

We develop an attribute-oriented anti-forgetting (AF) strategy to explore attribute-wise representations that align the
distributions of the old and new models, as shown in Fig. 2
(AF). The new model can adapt to new information but may
forget old knowledge from the previous datasets, while the old
model retains old knowledge. To preserve old knowledge, we
leverage attribute-wise representations as a bridge to optimize
both the old and new models by using samples from the
memory buffer. This strategy achieves domain consistency and
minimizes inter-domain gaps, alleviating the forgetting of old
knowledge, and is calculated as follows:
B

LAF =

1 X
t
KL(AGt−1
N /τ ||AGN /τ ).
B i=1

(8)

Where KL(.||.) is a kullback-leibler divergence, and τ represents a hyper-parameter called temperature [41].
F. Knowledge Consolidation
Maximizing intra-domain discrimination and minimizing
inter-domain gaps are in a contradictory relationship. Therefore, achieving a balance between them is crucial for improving the performance of LReID models. Thus, we propose a
knowledge consolidation (KC) strategy that leverages global
representations for knowledge transfer between old and new
models. This includes alignment mechanism and logit-level
distillation mechanism.
Maintaining distribution consistency between the old and
new models for previous datasets can limit the model’s ability
to learn new knowledge. Therefore, we propose an alignment
mechanism to explore global representations of knowledge
transfer from the current dataset, as follows:
B

LAL =

1 X
KL(Gt−1 /τ ||Gt /τ ).
B i=1

(9)

We further introduce a logit-level distillation mechanism to
enhance the extraction of identity information shared between
the old and new models, further improving the model’s knowledge consolidation ability. This is represented as follows:
B

1 X
LLD =
KL((ϕt−1 (Gt−1 ))i /τ ||(ϕt (Gt ))i /τ ).
B i=1

(10)

The knowledge consolidation loss is defined as:
LKC = LAL + LLD .

(11)

The total loss function is formulated as:
L = LCE + LgT ri + LlT ri + LOrt + LAF + LKC .

(12)

6

KRKC

TABLE II
class boundary
DATASET
STATISTICS FOR BOTH SEEN AND UNSEEN DOMAINS . S INCE THE
domain boundary
SELECTION PROCESS RESULTED IN 500 TRAIN ID S BEING SELECTED , THE
ID 6
ID 4
ID 5
ORIGINAL
NUMBERS OF ID S ARE LISTED FOR COMPARISON . ’-’ DENOTES
THAT THE DATASET IS NOT USED FOR TRAINING .

LSTKC

DKP

Ours

Global
representations center
attribute category
ID 2

D1

ID 3

Step t-1
Knowledge
Distllation

Step t
Type

Minimize
inter-domain gaps
Step t

Step t-1

Existing Methods
Man/woman
Shorts
Trousers
Backpack
Shoulder bag
Long sleeve
Short sleeve

Seen

Step t-1
Step t
Minimize inter-domain gaps
Maximize intra-domain discrimination
Ours

Datasets

Scale

Market1501 [42]
CUHK-SYSU [43]
DukeMTMC [44]
MSMT17 V2 [45]
CUHK03 [46]

large
mid
large
large
mid

Text
Encoder

VIPeR [47]
GRID [48]
CUHK02 [49]
Unseen
Occ Duke [50]
Occ REID [51]
Text-guided aggregation network (TGA)
Global
representations[52]
PRID2011
[CLS]
G1

Train IDs

500(751)
500(942)
500(702)
500(1041)
500(700)
−
small
−
small
−
mid
−
large
−
mid
Parallel Fusion Module (PFM)
Parallel Fusion Module (PFM)
−
small MLP

Test IDs
750
2900
1110
3060
700
316
126
239
1100
200
649

C

G2

LayerNorm

PFM

LayerNorm

GN

Image
Encoder


codex
前三篇终身论文的方法部分都读完了。无监督可见光-红外这篇的方法在第 8 页之后还没结束，所以我会继续抽后续方法段落，保证不只停在概览和基线。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 12 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | rg -n '3\\.[0-9]|4\\. Experiments|Experiment|Proposed Method|Causal|Collaborative|Label|Feature|Training'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
15:model level, we propose a Causality-inspired Adjustment Intervention (CAI) module
17:optimization level, a Collaborative Bias-free Training (CBT) strategy is introduced to
29:Keywords: Visible-infrared person re-identification, Unsupervised learning, Causal
76:To this end, a Causality-inspired Adjustment Intervention (CAI) module and a Collaborative Bias-free Training (CBT) strategy are proposed. Specifically, CAI facilitates
92:(2) We propose a Causality-inspired Adjustment Intervention (CAI) module at the
95:(3) We propose a Collaborative Bias-free Training (CBT) strategy at the optimization
127:2.2. Person ReID with Causal Inference
140:2.3. Person ReID with Noise Label Learning
150:3. Proposed Method
151:3.1. Overview
153:Fig. 2, incorporating the Causality-inspired Adjustment Intervention (CAI) module
154:and the Collaborative Bias-free Training (CBT) strategy. In cross-modality learning,
166:Causality-inspired Adjustment Intervention
243:Collaborative Bias-free Training
250:Iterative Maximum Confidence Alignment, the Causality-inspired Adjustment Intervention module is implemented for causal modeling to construct a low-biased model. Then, the Collaborative Bias-free Training
254:3.2. Baseline for Two-stage USL-VI-ReID
312:3.3. Modeling with Causal Intervention
332:𝑌 Labels
347:Causal Relation
352:3.3.1. USL-VI-ReID from Causal View
353:To illustrate our motivation for modality debiasing from the causal view, we represent the cross-modality learning process of USL-VI-ReID into the Structural Causal
377:3.3.2. Causality-inspired Adjustment Intervention
429:3.4. Collaborative Bias-free Training
437:3.4.1. Data Augmentation in CBT
503:3.4.2. Label Refinement in CBT

exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
the weight of Ltri which dynamically changes during training.
In the second stage, we initialize cross-modality learning using the pretrained singlemodality model and adopt the clustering-based unsupervised learning pipeline. To
obtain cross-modality pseudo-labels, we propose a simple yet effective Iterative Maximum Confidence Alignment (iMCA) scheme in the baseline to quickly match the
N clusters of one modality with the M clusters of the other. Let the modality with N
clusters be denoted as C N and the other with M clusters as C M . iMCA first calculates
the cosine similarity between cluster centroids to construct an N × M similarity matrix S , where S i, j represents the similarity between the i-th cluster of C N and the j-th
cluster of C M . With this, we perform two ways of matching: row-wise and columnwise. In the row-wise matching, for the i-th row of S , we find its matched cluster index
urow
∈ [0, M] as follows:
i
urow
= arg max j S i, j .
i

(2)

The pseudo-label of the urow
i -th cluster in C M is then assigned to the i-th cluster in
C N . This operation is applied to all rows (∀i ∈ [0, N]), effectively propagating labels
from C M to C N . In the column-wise matching, the same procedure is performed for
all columns to propagate labels from C N to C M . By alternating between row-wise
and column-wise matchings across different epochs, iMCA obtains cross-modality
pseudo-labels while preventing the model from being overconfident in a certain matching. Then, cross-modality cluster memories mk are established based on unified crossmodality labels for contrastive learning as follows:
exp( f x · m+ /σ)
Lid = −log P
.
exp( f x · mk /σ)

(3)

k

Finally, the unsupervised cross-modality model is trained with Lid + λtri · Ltri .
3.3. Modeling with Causal Intervention
In this section, we first consider USL-VI-ReID from the causal view, analyzing that
spurious bias patterns are captured by traditional likelihood-based modeling through a
backdoor, whereas causal intervention is not. Based on this analysis, we then illustrate
the causal modeling in the proposed CAI module, which constructs a cross-modality
model that is insensitive to modality bias.
8

𝐶

𝐶

𝐶

Modalities

𝑋

Images

𝑌 Labels
𝑋

𝑌

𝑋

(a) 𝑃(𝑌|𝑋)

𝑌

(b) 𝑃(𝑌|𝑑𝑜(𝑋))

Shortcut

Causal Relation

Figure 3: (a) The structural causal model in cross-modality learning for USL-VI-ReID. (b) The modified
structural causal model after the causal intervention.

3.3.1. USL-VI-ReID from Causal View
To illustrate our motivation for modality debiasing from the causal view, we represent the cross-modality learning process of USL-VI-ReID into the Structural Causal
Model (SCM) framework [22], as shown in Fig. 3 (a). The SCM depicts the relationships among the variables ‘images’ X, ‘labels’ Y, and ‘modalities’ C. The arrow
C → X indicates that the modality determines the image pixel values. X → Y means
causal relationships that can recognize human identity from given images. Meanwhile,
C → Y reflects the modality bias issue: due to the unsupervised learning pipeline,
cross-modality relationships are established based on single-modality clustering and
matching that are inherently influenced by modality-specific cues, resulting in biased
labels in cross-modality learning.
From this perspective, we can find that modality information influences both the observed images and the inferred labels, inducing a spurious correlation (i.e., a backdoor
path) between the input and the prediction, formulated as X ← C → Y. This backdoor
is entangled with the true causal relationship X → Y and is therefore inevitably captured by the likelihood model, which directly models P(Y|X) without distinguishing
causal identity cues from modality-dependent factors. As a result, the learned model
tends to exploit modality-induced correlations as shortcuts, resulting in biased predictions and degraded generalization.
To explicitly address this problem, we introduce causal intervention and optimize
the interventional distribution P(Y|do(X)), as illustrated in Fig. 3 (b). The intervention
probability P(Y|do(X = x)) corresponds to inferring the identity label given an intervened image X fixed to a specific input x. The intervention operation do(·) severs the

9

dependency between X and all its potential causes, thereby blocking the path C → X
and eliminating the backdoor X ← C → Y. As a result, causal intervention forces
the model to rely on identity-related causal patterns rather than modality-specific cues.
This provides a principled mechanism for modality debiasing in unsupervised crossmodality learning and motivates our implementation of intervention in CAI to prevent
the model from learning modality bias through the backdoor.
3.3.2. Causality-inspired Adjustment Intervention
Based on the above analysis, an intervention loss Lcai is constructed by maximizing
the intervention probability to eliminate the interference of the modality bias:
Lcai = E x,y [− log P(Y = y|do(X = x))],

(4)

where x denotes an input image, and y represents its associated cross-modality pseudolabel. To achieve that, CAI implements the computation of P(Y|do(X) by backdoor
adjustment [22] (the detailed derivation is provided in the supplementary material), as
follows:
P(Y|do(X)) =

X

P(Y|X, C = c) · P(C = c),

(5)

c∈{V,I}

where P(C = c) means the probabilities of modality c, and can be approximated from
the training set. P(Y|X = x, C = c) represents the classification probability of a specific
image x inferred by incorporating specific knowledge of modality c. Importantly, c
is not necessarily the original modality of x, which means that the inference needs to
combine the image with both visible-specific (V) and infrared-specific (I) knowledge.
We achieve this by using single-modality memories as follows:
exp( f x · mcy /σ)
P(Y = y|X = x, C = c) = P
,
exp( f x · mck /σ)

(6)

k

where f x is the feature extracted by the backbone model, y represents the cross-modality
pseudo-label of the image x, and mcy is the cluster centroid of y-th cluster of modality
c. With these modeled probability parts, we can train the model following Eq. (4).
We provide further analysis of CAI. Compared to the likelihood model P(Y|X)
which can be decomposed as follows:
X
P(Y|X) =
P(Y|X, C = c) · P(C = c|X),
c

10

(7)

the backdoor adjustment modifies P(C = c|X) to P(C = c), which can be seen as blocking the correlation between modalities C and images X. It eliminates the modality bias
during modeling, achieving a low-biased cross-modality model by capturing purely
causal relationships.
3.4. Collaborative Bias-free Training
Although a low-biased model is obtained through CAI, the biased modality-specific
cues existing in labels and features still mislead the model training. To tackle this
problem, we propose the CBT strategy to mitigate modality bias at the optimization
level. Specifically, considering that modality bias propagates from data into labels and
features, CBT integrates label refinement and feature alignment with well-designed
data augmentation, thereby disrupting bias propagation and promoting unbiased feature
learning.
3.4.1. Data Augmentation in CBT
CBT first introduces a modality-specific augmentation scheme to destroy modalityrelated information in images, as shown in Fig. 4. Specifically, for infrared images, we
first employ a series of color mapping methods [32] to transfer each infrared image
to multiple pseudo-color images. Then, a channel-wise sampling scheme is proposed
to increase diversity and introduce randomness to the augmentation by randomly sampling R, G, and B channels of multiple generated pseudo-color images and combining
the corresponding sampled channels into a new image. For visible images, we employ
channel augmentation (CA) [1] through channel multiplexing to generate augmented
images, which could derive a series of augmented samples that look like infrared.
This modality-specific data augmentation enables the image and its corresponding augmentation to share the same identity-discriminative information but differ in
modality-related information, mitigating the modality bias at the data level. With the
assistance of such augmentation, CBT implements label refinement and feature alignment to facilitate bias-free learning.

11

𝐼𝑅

𝑅2

𝑅1

color
mapping

𝐺1
𝐵1

𝐺2

…

𝑅𝑛

𝐺𝑛
𝐵𝑛

𝐵2

original

channel-wise
sampling

𝑅𝑖
𝐺𝑗

𝐵𝑘
augmented

𝑅

selecting 𝑅 or 𝐺 or 𝐵

𝐺

𝑅
𝑅

𝑅

𝐵

Figure 4: Illustration of the modality-specific augmentation. Circles represent channels of images. Subscript
represents the sample index of pseudo-color images. For example, R2 , G2 , and B2 are the red, green, and
blue channels from the same pseudo-color image with index 2. The grey circle with IR indicates the single
channel of the infrared image.

3.4.2. Label Refinement in CBT
To refine the noise pseudo-labels, CBT employs label smoothing by exchanging
the predictions of images and their augmented images as follows:



ei = wi yi + (1 − wi ) P(Y|X = xia )

 y



 y
ea = wi yi + (1 − wi ) P(Y|X = xi ),

(8)

i

where wi ∈ [0, 1] is the refinement weight, representing the reliability of the label yi .
The boldface yi means the one-hot label vector in which the class index yi is set to 1 and
others are 0. e
yi and e
yai represent the refined soft labels of image xi and its augmentation
xia , respectively. Then, they are used to supervise model training by modifying the Lcai
in Eq. (4) as a soft-label classification loss as follows:
X
e
Lcai = Ei [ −
yi [k] · log P(Y = k|do(X = xi ))
k

−

X

(9)
e
yai [k] · log P(Y = k|do(X = xia ))],

k

where e
yi [k] means index k-th value of e
yi . The computations of e
y and e
ya depend on
wi and P(Y|X), where the former is the certainty of yi , and the latter is the likelihood
function.
Refer to Eq. (8), the certainty wi reflects the reliability of the label for the i-th
sample. A higher wi indicates a higher-quality label, allowing yi to contribute more
significantly than P(Y|X) to the final refined label. To quantify the reliability of each
label yi as wi , we follow a common practice in noise label learning [33] by modeling
the distribution of sample losses using a Gaussian Mixture Model (GMM). Specifically,
12


codex
这篇 DMDL 的正文把“因果干预”和“协同去偏训练”分开写，前 8 页只覆盖到标签细化开头。为了满足完整方法，我会越过页码限制继续把剩下的小节读完。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 16 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | rg -n '3\\.4\\.2|3\\.4\\.3|3\\.5|4\\. Experiment|4\\. Experiments|Feature Alignment|Total|Overall|Algorithm'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 16 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | sed -n '503,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
503:3.4.2. Label Refinement in CBT
611:3.4.3. Feature Alignment in CBT
639:3.5. Total Loss of DMDL
662:4. Experiments
691:3.5 × 10−4 and a weight decay of 5 × 10−4 . The learning rate is decreased by a factor of

 succeeded in 0ms:
3.4.2. Label Refinement in CBT
To refine the noise pseudo-labels, CBT employs label smoothing by exchanging
the predictions of images and their augmented images as follows:



ei = wi yi + (1 − wi ) P(Y|X = xia )

 y



 y
ea = wi yi + (1 − wi ) P(Y|X = xi ),

(8)

i

where wi ∈ [0, 1] is the refinement weight, representing the reliability of the label yi .
The boldface yi means the one-hot label vector in which the class index yi is set to 1 and
others are 0. e
yi and e
yai represent the refined soft labels of image xi and its augmentation
xia , respectively. Then, they are used to supervise model training by modifying the Lcai
in Eq. (4) as a soft-label classification loss as follows:
X
e
Lcai = Ei [ −
yi [k] · log P(Y = k|do(X = xi ))
k

−

X

(9)
e
yai [k] · log P(Y = k|do(X = xia ))],

k

where e
yi [k] means index k-th value of e
yi . The computations of e
y and e
ya depend on
wi and P(Y|X), where the former is the certainty of yi , and the latter is the likelihood
function.
Refer to Eq. (8), the certainty wi reflects the reliability of the label for the i-th
sample. A higher wi indicates a higher-quality label, allowing yi to contribute more
significantly than P(Y|X) to the final refined label. To quantify the reliability of each
label yi as wi , we follow a common practice in noise label learning [33] by modeling
the distribution of sample losses using a Gaussian Mixture Model (GMM). Specifically,
12

we first compute the loss value for each sample via cross-entropy:
Liid = −log P(Y = yi |X = xi ).

(10)

The overall loss distribution Lid is then fitted with a two-component GMM, where
one component corresponds to low-loss samples (indicating high-quality labels) and
the other to high-loss samples (indicating low-quality labels). After training, the GMM
estimates the probability that a given loss Liid belongs to the low-loss component, which
is used as the label certainty wi .
Note that P(Y|X) plays a crucial role in computing both the certainty and the refined labels. We follow Eq. (7) to implement P(Y|X) but adopt Normalized Weighted
Geometric Mean (NWGM) [34] for simplification (details are provided in the supplementary material). In one word, P(Y|X) is computed using the modality-shared
memory:

exp( f x · my /σ)
P(Y = y|X = x) = P
,
exp( f x · mk /σ)

(11)

k

where mk denotes the centroid of the k cluster in the modality-shared memory bank. It
is evident that the quality of the memory bank directly influences the reliability of the
predictions. To enhance prediction robustness, we design a dynamic updating scheme
that iteratively updates the memory centroid features:
my ← η x my + (1 − η x ) f x ,

(12)

where η x is an adaptive coefficient determined by:
η x = η/max(e
y x [k == y], η).

(13)

Here, e
y x [k == y] represents the confidence score of sample x being assigned to class
y, which is obtained from the refined soft label e
y x , and η is a constant threshold set
to 0.2. This adaptive coefficient ensures that samples with higher label confidence
contribute more substantially to updating the corresponding memory feature my , while
low-confidence samples have limited influence.
Compared with methods [29, 30] that only penalize noisy samples based on label
certainty, the proposed label refinement directly constructs low-biased cross-modality
13

labels by incorporating modality-specific augmentations and dynamically updating the
modality-shared memory. Since an image and its modality-specific augmentation contain different modality-related information, exchanging their predictions for label smoothing effectively reduces label noise induced by modality-specific cues. Furthermore,
the dynamic memory updating scheme prevents the memory bank from accumulating
noisy representations, thereby ensuring more reliable predictions for refining labels.
3.4.3. Feature Alignment in CBT
In addition, a feature alignment loss is introduced to further enhance bias-free feature learning. It is well understood that identity-discriminative information should remain consistent under augmentation. Therefore, for an image and its modality-specific
augmentation, the model is expected to extract similar features; otherwise, it suggests
that the model is learning modality-specific knowledge. To this end, we design L f a
following the principles of MMD [35]:
n
n
X 1X

1X
c
Lfa =
∥
ϕ fi −
ϕ fica ∥2H ,
n
n
i=1
i=1
c∈{V,I}

(14)

where fica represents the features of the augmented images of modality c. ∥·∥H denotes
the distance measured by the Gaussian kernel function ϕ (·), which maps the input to
the Reproducing Kernel Hilbert Space (RKHS). This loss constrains the original image
and its augmentation representations to be close in the metric space, thereby mitigating
the modality-specific cues learned in feature representations.
3.5. Total Loss of DMDL
Following the baseline, the total loss function of our DMDL can be written as:
I
+ λcai · Lcai + λ f a · L f a + λtri · Ltri ,
L = LVid + Lid

(15)

where λcai , λ f a and λtri are weights of the corresponding loss term.
Discussion. In summary, the proposed DMDL framework establishes a unified debiasing pipeline that integrates causal modeling with bias-free optimization. At the
modeling level, the CAI module performs causal intervention via backdoor adjustment,
encouraging the model to capture causal identity patterns rather than modality-specific
14

shortcuts, thereby constructing a low-biased model. Building upon CAI, the CBT further mitigates bias propagation during the optimization process. The modality-specific
data augmentation disrupts modality cues at the data level, label refinement corrects biased pseudo-labels at the label level, and feature alignment enforces modality-invariant
representations at the feature level. These components collaboratively prevent biased
information from being amplified through iterative training. Importantly, CAI and CBT
play complementary roles. CAI suppresses modality bias at the modeling level by reshaping the learning objective, while CBT prevents residual bias from being propagated
during optimization. By jointly considering causal intervention and training dynamics,
DMDL formulates modality debiasing as an end-to-end learning problem, enabling
robust and stable bias suppression throughout the learning pipeline.

4. Experiments
4.1. Datasets and Evaluation Protocol
Dataset. In this section, we conduct comprehensive experiments to evaluate the proposed method on two widely used datasets, SYSU-MM01 [36] and RegDB [37], as
well as a more recent dataset, LLCM [38].
The SYSU-MM01 dataset with 4 visible cameras and 2 infrared cameras, capturing
395 identities for training and 96 for testing. The test query set comprises 3,803 infrared images, and the gallery set contains 301 visible images. The evaluation protocol
provides all-search and indoor-search modes.
The RegDB is a dual-camera dataset with 412 identities, each having 10 visible and
10 infrared images. It is split into 206 identities for training and 206 for testing. The
evaluation protocol includes two test modes: visible to infrared and infrared to visible.
The LLCM is the largest VI-ReID dataset that captures images with 9 cameras. It
contains 1,064 identities, of which 713 are used for training and 351 for testing. The
evaluation protocol includes two test modes: VIS to IR and IR to VIS.
Evaluation protocol. All experiments follow the standard evaluation protocol in the
VI-ReID benchmark testing. Our model is evaluated using different training/testing
splits in ten trials to ensure stable performance. Evaluation metrics include cumulative
15

matching characteristics (CMC), mean average precision (mAP), and mean inverse
negative penalty (mINP) [39].
4.2. Implementation Details
We employ ResNet-50 pre-trained on ImageNet as the backbone network and integrate Non-local Attention Blocks [39] and generalized-mean (GeM) pooling [39]. All
input images are resized to 288 × 144, and standard data augmentation techniques,
including horizontal flipping, random cropping, and random erasing, are applied. At
the beginning of each epoch, DBSCAN [40] clustering is performed independently
for each modality to generate pseudo labels. The clustering threshold and the minimum number of images are set to 0.6 and 4 on SYSU-MM01 [36] and LLCM [38],
and to 0.3 and 4 on RegDB [37], respectively. During training, 16 pseudo-identities
are sampled from each modality, with 16 instances per pseudo-identity (8 original and
8 augmented). The model is optimized using Adam with an initial learning rate of
3.5 × 10−4 and a weight decay of 5 × 10−4 . The learning rate is decreased by a factor of
ten every 20 epochs. The hyperparameter σ is set to 0.05. Training proceeds for a total
of 100 epochs, with the first 50 epochs dedicated to single-modality learning, followed
by 50 epochs of cross-modality training.
4.3. Comparison with State-of-the-art Methods
To validate the effectiveness of our DMDL, we compare it with state-of-the-art
methods under three relevant settings: supervised VI-ReID, semi-supervised VI-ReID,
and unsupervised VI-ReID. The experimental results for the SYSU-MM01 and RegDB
datasets are shown in Table 1, and the experimental results for the LLCM dataset are
presented in Table 2.
Comparison with supervised VI-ReID Methods. Encouragingly, our DMDL achieves
competitive performance compared to the supervised method FMCNet [42] on the
SYSU-MM01 and RegDB datasets, and even surpasses several supervised methods,
including AGW [39] and SPOT [41]. Moreover, on the challenging LLCM dataset, our
DMDL still demonstrates impressive performance, outperforming several supervised
methods (e.g., AGW [39] and LbA [50]). However, due to the absence of annotated
16


codex
机制部分读完后，我还需要回看每篇引言里的“问题怎么被说出来”。这些段落通常决定它到底是在卖一个模块，还是在卖一个新视角。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf' - | sed -n '45,267p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
I. I NTRODUCTION
ERSON reidentification (Re-ID) seeks to match pedestrian identities across images captured by nonoverlapping

Manuscript received 7 August 2023; revised 18 November
2023 and 30 January 2024; accepted 28 March 2024. Date of publication
15 April 2024; date of current version 1 March 2025. This work was
supported in part by the National Natural Science Foundation of China under
Grant 62076262, Grant 61673402, Grant 61273270, and Grant 60802069;
and in part by the China Scholarship Council under Grant 202306380182.
(Corresponding author: Haifeng Hu.)
The authors are with the School of Electronics and Information
Technology, Sun Yat-sen University, Guangzhou 510006, China (e-mail:
huhaif@mail.sysu.edu.cn).
This article has supplementary downloadable material available at
https://doi.org/10.1109/TNNLS.2024.3384023, provided by the authors.
Digital Object Identifier 10.1109/TNNLS.2024.3384023

Fig. 1. Illustration of motivation and methodology regarding the proposed
DMPF. (a) VIS-NIR dataset presents challenges for learning identity-invariant
feature representations due to the modality and posture factors. (b) and (c) Proposed DMPF successfully disentangles the modality and posture factors from
the identity features by the proposed MFD and PFD modules, respectively.
Specifically, to eliminate modality discrepancies across the entire training set,
the MFD is proposed to integrate features from the current batch and memory
queues by adopting the designed memory-attention layers. To disentangle the
postures factor, the PFD is designed to use subspace orthogonal decomposition
between posture-related and identity-related information in the image and
skeleton features.

cameras [1]. This field has attracted substantial attention in
the past decade, primarily owing to its extensive applications
in security surveillance [2], criminal investigation [3], and
intelligent transportation [4], [5], [6]. Previous Re-ID methods
based on deep learning [1], [7], [8], [9] have achieved remarkable success and even surpassed human performance in certain
scenarios, benefiting from large-scale annotated datasets. However, these approaches assume that the pedestrians are captured
by visible (VIS) spectrum cameras in well-lit conditions,
which limits their applicability in realistic nighttime scenarios
owing to the VIS camera’s failure to capture target appearance
in low-light scenes [10]. Fortunately, most modern surveillance
systems are equipped with the capability to switch seamlessly
from VIS to near-infrared (NIR) spectrum cameras in lowlight conditions, since NIR cameras can explore information
on pedestrian appearance in low-light environments [11],
[12]. Nevertheless, the dramatic modality discrepancy between
VIS and NIR images remains unresolved by previous ReID methods [1], [7], [8], [9]. Therefore, striving to establish
correspondences of pedestrian identities across heterogeneous
images acquired from distinct spectra, VIS-NIR Re-ID has
emerged as a relevant and challenging task.
Compared to conventional Re-ID tasks, VIS-NIR Re-ID
presents more difficulty due to the substantial intraclass
disparities exhibited by heterogeneous images [13]. As illustrated in Fig. 1(a), the intraclass differences primarily stem
from two factors: modality discrepancy and posture variation.

2162-237X © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:54 UTC from IEEE Xplore. Restrictions apply.

LU et al.: DMPFs: MEMORY-ATTENTION AND ORTHOGONAL DECOMPOSITION

5495

For the former factor, the NIR camera captures pedestrian
appearance details in low-light environments by utilizing NIR
spectral information, which inevitably sacrifices the color,
contrast, and other VIS spectral attributes in the captured
NIR images [14]. Consequently, the pronounced modality
discrepancy arises between heterogeneous images. Moreover,
pedestrians undergo significant pose variations over time,
leading to noticeable differences in the poses of the same
pedestrian captured at different moments. As a result, the
disparity in postures is notable for both NIR and VIS
images. Therefore, the modality and posture discrepancies
in heterogeneous images bring challenges in extracting the
identity-invariant features for Re-ID [15].
To learn identity-discriminative features, various VIS-NIR
Re-ID methods have focused on eliminating modality discrepancies by aligning VIS-NIR image styles from channel [16],
[17], [18] or spatial [19], [20], [21] dimension, known
as image alignment methods. However, both channel- and
spatial-based image alignment methods face challenges in
eliminating stylistic differences while preserving the appearance information of the original image [22]. On the one hand,
due to the complexity and nonlinearity of modality factors
in raw images, considerable stylistic differences still persist
between the NIR-like images generated by channel operations
and the real NIR images. On the other hand, the pixel
misalignment problem disrupts the appearance information
of pedestrians in the image generated by spatial operations, which hinders the learning of identity-invariant features.
To address these issues, recent GAN-based methods [23], [24],
[25] utilize a generative adversarial framework supervised
by identity-invariant and style-consistent losses to generate
images that exhibit consistent style and invariant appearance.
However, the introduction of the generator and discriminator in
the generative adversarial framework brings massive trainable
parameters, resulting in increasing training costs and computation complexity. Besides, the image alignment approaches
primarily focus on eliminating modality differences at the
image level, which neglects the isolation of heterogeneous
samples in the feature space. This limitation impacts the
extraction of modality-independent features and subsequently
affects the performance of the model in VIS-NIR Re-ID.
To bridge the modality gap between heterogeneous samples in the feature space, feature learning methods have
been developed to enhance the identity discriminability of
heterogeneous features by introducing discriminative local
features [26], [27]. These approaches effectively reduce the
influence of modality factors on identity matching. Differently,
to learn modality-independent features for heterogeneous
samples, feature alignment methods incorporate various heterogeneous losses to restrict the distribution or distance
between VIS and NIR features in each batch [28], [29], [30].
In addition, considering that modality factors embedded in
identity representations affect cross-modal identity matching,
disentangling-based approaches decouple modality information from identity representations by imposing orthogonal
constraints on identity-related and modality-related features
[13], [22], [31]. These feature learning methods have demonstrated their effectiveness in mitigating modality gaps in the
feature space. However, they primarily focus on eliminating
modality discrepancies at the batch-level, ignoring the modality disparities present in heterogeneous features outside the
batch, which leads to modality factors remaining in these

features. Furthermore, the aforementioned methods predominantly focus on the modality differences, while ignoring
another crucial difference between heterogeneous samples, i.e.,
posture discrepancy.
To eliminate the posture discrepancy, several approaches
generally utilize landmark [32] or parsing [33] information
of pedestrians to guide the learning of pose-invariant features. However, the extracted skeleton features encompass
both pose-related and identity-related information, which may
impact posture factor decoupling and undermine the discriminability of the identity features. Therefore, it is necessary to
disentangle them before introducing the skeleton features into
the representations. Moreover, landmark and parsing represent the information of keypoints location and shape profile,
respectively. Considering the complementary between them,
fusing the two types of information is beneficial for exploring
the pedestrian’s pose features. In addition, some methods
leverage keypoint information to extract local features, fostering the model’s ability to discern nuances among distinct
pedestrians. For instance, Liang et al. [34] proposed keypointguided modality-invariant discriminative learning (KMDL)
to extract discriminative fine-grained local representations
by adopting human keypoint information and an additional
graph convolutional network. Nevertheless, the concatenation
of local features unavoidably amplifies the dimensions of
feature representation, consequently elevating the time and
memory demands during model training. Moreover, during
the inference phase, the introduced graph-convolutional network extends the time required for feature extraction, thereby
impacting the real-time performance of the model in practical applications. Distinctly, the proposed method utilizes
keypoint information to learn pose-related features, aiding in
pose-independent identity feature extraction. Consequently, the
dimension of the feature representation is not significantly
increased, ensuring it does not impact efficiency during both
training and inference phases.
To learn identity-discriminative representations for crossmodality identity matching, we propose a disentangling
modality and posture factors (DMPFs) model to disentangle
modality and posture factors by fusing the information
of features memory and pedestrian skeleton. The former
preserves intraclass features for the entire training set,
facilitating the comprehensive elimination of modality
differences across the training set, while the latter fuses
landmark and parsing information of pedestrians, providing
pose-related information to effectively decouple pose factors
from identity features. Specifically, the DMPF comprises three
modules: three-stream features extraction network (TFENet),
modality factor disentanglement (MFD), and posture factor
disentanglement (PFD). First, to provide memory and skeleton
information for modality and posture factors disentanglement,
we designed a three-stream network named TFENet to extract
intraclass feature memories and skeleton features. Second,
as illustrated in Fig. 1(b), to eliminate modality discrepancy
across the training set, we propose the MFD to integrate batch
and memory features using designed memory-attention layers.
Optimizing by the optimal transport (OT) method, these layers
can effectively explore intramodal and intermodal relationships
by multihead self-attention [35], exposing augmented batch
features with the same identity that present high similarity.
As a result, we successfully decouple the modality factors
from identity features by the memory-attention layers.
Furthermore, to transfer the ability of modality difference

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:54 UTC from IEEE Xplore. Restrictions apply.

5496

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

elimination from the memory-attention layers to the backbone
network, we propose the modal-independent distillation
(MiD) loss to measure the similarity between the original
batch features and the augmented batch features. This loss
encourages the backbone network to learn from the memoryattention layers, thereby extracting modality-independent
feature representations effectively. Third, as illustrated in
Fig. 1(c), considering the complementary between landmark
and parsing information, we fuse both information to extract
skeleton features of pedestrians. Moreover, to handle the
entanglement between posture-related and identity-related
information in the images and skeleton features, we propose
the PFD module to extract posture-related and identity-related
features through subspace orthogonal decomposition [36].
The posture-related features are extracted to effectively
separate the posture factors by the proposed posture-features
consistency (PfC) loss, while the identity-related features
are utilized to improve the discriminability of image identity
representations by concatenating operation.
In brief, the main contributions are as follows.
1) Aiming at learning identity-invariant feature representations, the DMPF model integrates TFENet, MFD,
and PFD modules into a unified framework to
effectively disentangle modality and posture factors
from identity representations. In contrast to previous
approaches, the DMPF considers both modality and
posture discrepancies between VIS and NIR data, which
consequently learn identity-discriminative feature representations more efficiently.
2) To achieve the elimination of modality discrepancies
across the entire training set, we propose the MFD
module to integrate features from the current batch
and memory queues by adopting the designed memoryattention layers. Different from the previous method,
these layers progressively explore intramodality and
intermodality relationships between batch and memory
features under the optimization of the OT method,
thereby encouraging the extracted heterogeneous features with the same identity to present higher similarity.
Hence, we effectively decouple the modality factors
from identity features by the memory-attention layers.
3) To disentangle the postures factor from identity representation, we propose the PFD module to handle
the entanglement between posture-related and identityrelated information in the image and skeleton features
through subspace orthogonal decomposition. Distinctive
from existing methods, the PFD module decomposes
both image and skeleton features into pose-related and
identity-related features. The former is adopted to effectively separate the pose factors by the proposed PfC loss,
while the latter is utilized to improve the discriminability
of identity representations by concatenating operations.
The article proceeds as follows. Section II provides an
overview and analysis of related works. In Section III, the
proposed DMPF is presented in detail. Section IV showcases
the experimental results, demonstrating the superiority of
DMPF. Finally, Section V concludes this article.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Diverse Representations Embedding for Lifelong Person Re-Identification.pdf' - | sed -n '1,266p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1

Diverse Representations Embedding for Lifelong
Person Re-Identification

arXiv:2403.16003v2 [cs.CV] 2 Apr 2024

Shiben Liu

, Huijie Fan*

, Qiang Wang

Abstract—Lifelong Person Re-Identification (LReID) aims to
continuously learn from successive data streams, matching individuals across multiple cameras. The key challenge for LReID
is how to effectively preserve old knowledge while incrementally
learning new information, which is caused by task-level domain
gaps and limited old task datasets. Existing methods based on
CNN backbone are insufficient to explore the representation
of each instance from different perspectives, limiting model
performance on limited old task datasets and new task datasets.
Unlike these methods, we propose a Diverse Representations Embedding (DRE) framework that first explores a pure transformer
for LReID. The proposed DRE preserves old knowledge while
adapting to new information based on instance-level and tasklevel layout. Concretely, an Adaptive Constraint Module (ACM)
is proposed to implement integration and push away operations between multiple overlapping representations generated by
transformer-based backbone, obtaining rich and discriminative
representations for each instance to improve adaptive ability
of LReID. Based on the processed diverse representations, we
propose Knowledge Update (KU) and Knowledge Preservation
(KP) strategies at the task-level layout by introducing the adjustment model and the learner model. KU strategy enhances the
adaptive learning ability of learner models for new information
under the adjustment model prior, and KP strategy preserves old
knowledge operated by representation-level alignment and logitlevel supervision in limited old task datasets while guaranteeing
the adaptive learning information capacity of the LReID model.
Extensive experiments were conducted on eleven Re-ID datasets,
including five seen datasets for training in order-1 and order-2
orders and six unseen datasets for inference. Compared to stateof-the-art methods, our method achieves significantly improved
performance in holistic, large-scale, and occluded datasets. Our
code will be available soon.

, Xiai Chen

, Yandong Tang

Fig. 1. An illustration of our DRE. Each shape denotes an instance, and
color indicates a different representation of the same instance. The blue circle
indicates the primary embedding representation after integration. At the top,
the generated diverse representation is implemented for integration and push
operation. Integrated primary embedding representation (blue circle) presents
rich body information. Multiple separate auxiliary embedding representations
(orange and purple) maintain discrimination of each instance. At the bottom,
based on instance-level, we introduce Knowledge Update (KU) and Knowledge Preservation (KP) strategies to improve model performance in preserving
old knowledge and adapting to new information in sequential tasks.

Index Terms—Lifelong learning, diverse representation, adaptive constraint learning, person re-identification.

This work is supported by the National Natural Science Foundation of
China (62273339, 61991413, U20A20200), and the Youth Innovation Promotion Association of Chinese Academy of Sciences (2019203). (Corresponding
author: Huijie Fan)
Shiben Liu is with the State Key Laboratory of Robotics, Shenyang
Institute of Automation, Chinese Academy of Sciences, Shenyang 110016,
China, and with the Institutes for Robotics and Intelligent Manufacturing,
Chinese Academy of Sciences, Shenyang 110169, China, and also with the
University of Chinese Academy of Sciences, Beijing 100049, China (e-mail:
liushiben@sia.cn).
Huijie Fan, Xiai Chen, Zhi Han, and Yandong Tang are with the State Key
Laboratory of Robotics, Shenyang Institute of Automation, Chinese Academy
of Sciences, Shenyang, 110016, China, and with the Institutes for Robotics
and Intelligent Manufacturing, Chinese Academy of Sciences, Shenyang,
110016, China (e-mail: fanhuiie@sia.cn; chenxiai@sia.cn; hanzhi@sia.cn;
ytang@sia.cn).
Qiang Wang is with the Key Laboratory of Manufacturing Industrial Integrated Automation, Shenyang University, and with the State Key Laboratory
of Robotics, Shenyang Institute of Automation, Chinese Academy of Sciences,
Shenyang, 110016, China (e-mail: wangqiang@sia.cn).

, Zhi Han

I. I NTRODUCTION

P

ERSON re-identification (ReID) aims to retrieve the same
individuals across different camera views [1–4], which is
crucial in applications, such as intelligent surveillance [5–7]
and multi-camera collaborative tracking[8–10]. Most ReID
methods assume that the training datasets can be accessed all
at once, whereas ReID datasets are continuously incoming
in practical applications. Lifelong learning is necessary for
person re-identification in continuous datasets across the
scene.
Most ReID methods [11, 12] learn robust features on
specific distribution datasets through pre-trained networks.
Unlike ReID, the main purpose of LReID is to facilitate model
to efficiently preserve old knowledge while adapt to new
information in sequential tasks. The rehearsal-based [13, 14]
methods reserve a memory buffer for storing limited instances
from old tasks, and recall these instances when learning
new tasks, which ignore model representation capabilities

2

on limited old task datasets. A framework of learnable
and consistent features [15] is proposed that provides
complementary knowledge for diverse representations
generation, improving the discrimination and adaptability of
the LReID model. Inspired by knowledge distillation [16],
several methods [17, 18] in LReID incorporate distillation
loss between teacher networks with prior knowledge of
old tasks and student networks with adaptive learning of
current tasks to mitigate task-level domain gaps. These
methods all employ fixed-weight teacher networks, forcing
the model to reduce its adaptability to new tasks. Yu et al.
[19] introduced a knowledge refresh scheme that updates the
memory model with a smaller learning rate, which turns the
memory model into a dynamic teacher. The above CNN-based
methods insufficiently explore the discriminative and rich
representations of each instance, limiting the performance
of the LReID model in preserving old knowledge while
adapting to new information. We consider that the CNN
backbone (ResNet50, ResNet101) forces the introduction of
multiple branches to obtain diverse representations that are
independent and uncontrollable from each other in the LReID
model.
Recently, Vision Transformer (ViT) [20] and Data-efficient
Image Transformers (DeiT) [21] have achieved significant
success in image recognition [22, 23], detection [24, 25], and
restoration [26, 27]. In addition, TransReID [28] indicates
that pure transformer is effective in feature extraction for
ReID, because multi-head self-attention of the transformer
frame captures long-range dependencies and drives the
model to attend diverse human body parts. DC-Former[29]
gets multiple diverse and compact embedding subspaces.
Each embedding of these compact subspaces is more robust
and discriminative to identify similar classes. Although the
above methods have gained significant advantages in ReID,
transformers still need to be specifically designed for LReID
to preserve old knowledge while adapting new information.
Side information embeddings [28] (such as camera and
viewpoint-specific information) constantly change as the
number of tasks increases in LReID, leading to bias in the
LReID model. We introduce Maximum Embedding (ME)
to replace side information embeddings, focusing on critical
regions of the input sequence and reducing background
interference. Meanwhile, the class token of transformer is
transformed into a global representation of each instance in
the ReID task. We introduce multiple class tokens embedding
to learn multiple overlopping representation of each instance
in the LReID task. Multiple class tokens and patches are
concatenated in the first dimension and sent to the transformer
for training on the seen dataset. Therefore, we construct a
strong backbone based on a pure transformer in LReID, as
an adjustment model and learner model structure.
Based on the above improved transformer backbone,
we propose a diverse representation embedding that first
explores the pure transformer for LReID, named DRE.
Specifically, multiple overlopping representations are
generated by multiple class tokens of transformer, as
shown in Figure 1 (instance-level). In this paper, we set
up multiple overlopping representations. One is the primary

embedding representation for learning rich body information,
the other is multiple auxiliary embedding representations
for providing discrimination of each instance. We propose
an Adaptive Constraint Module (ACM) to implement
integration and push away operations between multiple
overlopping representations. Multiple auxiliary embedding
representations minimize overlapping elements by orthogonal
operation. We adaptively integrate multiple discriminative
auxiliary embedding representations into primary embedding
representations to enhance the representational capacity of
the LReID model, which provides rich body information for
classification. At the instance level, diverse representation
generation presents richness and discrimination of each
instance, remarkably improving feature learning and label
matching processes for models with the same ID. Based
on instance-level, we propose Knowledge Update (KU) and
Knowledge Preservation (KP) strategies at the task-level
layout by introducing the adjustment model and the learner
model. KU strategy enhances the adaptive learning ability of
learner models for new information under adjustment model
prior. Because the knowledge update strategy is prone to adapt
to new tasks, the knowledge preservation scheme is proposed
to preserve old knowledge operated by representation-level
alignment and logit-level supervision in limited old task
datasets while guaranteeing the adaptive learning information
capacity of the LReID model. The main contributions of this
paper are summarized as follows:
• We propose a diverse representation embedding framework that first exploits the transformer-based backbone
for LReID tasks. Maximum embedding Multiple class
tokens are introduced to reduce bias and generate multiple
representations of each instance.
• We design an adaptive constraint module to maintain rich
primary embedding and discriminative auxiliary embedding representations in all seen domains, improving the
adaptive capability of the LReID model.
• Based on diverse representations, knowledge update
facilitate learning new information capability of the
learner model, and knowledge preservation leverages
representation-level alignment and logit-level supervision
to preserve old knowledge while guaranteeing the learning information ability of the LReID model.
• Extensive experiments demonstrate that the proposed
DRE outperforms state-of-the-art methods on both seen
and unseen datasets. In general, our DRE outperforms
performance with CNN-based methods.
II. R ELATED WORK
A. Person Re-Identification
Research in person re-identification (ReID) make significant
achievements over the years. These methods can be broadly
divited into four kategories. First, CNN-bsed approaches
[30–33] extract discriminative features from pedestrian
images to effectively address the spatial complexities. Crossentropy loss [34] and triplet loss [35] are commonly applied
for training CNN to learn better representation. Second,

3

transformer-based methods [28, 36–38] make further progress
in ReID for their ability to capture long-range dependencies
in feature maps. In these methods, self-attention mechnism
is integrated into ReID to enhance the extraction of relevant
information. Third, domain adaptation and transfer learning
techniques [39–42] are also explored to improve model
generalization, particularly when dealing with variations
in lighting, viewpoints, and camera sources. Fourth, textto-image methods [4, 43, 44] leverage textual descriptions
and image data to improve recognition accuracy and model
performance, enabling more effective matching of individuals
in different contexts by combining text descriptions with
image-based features.
B. Lifelong Person Re-Identification
Lifelong Person Re-Identification (LReID) faces a
formidable challenge, aiming to address the evolving nature
of person identification across various scenarios and domains.
Some works [14, 45, 45, 46] are proposed to tackle the
issue of adapting ReID models over time while retaining
knowledge gained from previous experiences. Generally,
Pu et al. [14] proposed learnable knowledge graphs that
adaptively facilitate the mutual exchange of new and old
knowledge, thus achieving knowledge accumulation. Some
works [15, 17, 18, 47] aim to extract rich and discriminative
representation, mitigating the risk of knowledge forgetting.
Pu et al. [47] proposed meta-reconciliation normalization
(MRN) for mining meta-knowledge shared across different
domains. Meanwhile, ConRFL [15] maintains learnable and
consistent features across all seen domains, which improves
the discrimination and adaptation ability of the LReID
model. In addition, some methods [14, 17, 19] mitigate
catastrophic forgetting and enhance model accuracy by using
rehearsal-based strategies with images stored from previous
tasks.
C. Diverse representation learning
Representation learning methods employ specific modules
or branches to mine multiple discriminative features of each
instance, which minimize intra-class distance and maximize
inter-class distance. It effectively improves model discrimination for identifying similar classes. DEEN [32] can effectively
learn the informative representations by generating diverse
embeddings subspace of each instance. DC-Former [29] employs multiple class tokens in vision transformer to generate
multiple representations, which increases the identity density
of embedding space to help model improve its discrimination
for identifying similar classes. Sun et al. [18] employed
adaptively-chosen patches to exploit body local information.
Ge et al. [17] generated pseudo-task features by a pseudo-task
transformation module to complement the limited exemplars.
These LReID methods employ CNN as a feature extractor,
which insufficiently exploits rich and discriminative representations of each instance. We consider that diverse representations generated by multiple class tokens with transformerspecific structure play an important role in maintaining a

trade-off between preserving old knowledge and adapting to
new information. Thus, we first propose a transformer-based
diversity representation embedding for lifelong person reidentification.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Distribution aligned semantics adaption for lifelong person re-identification.pdf' - | sed -n '1,188p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Distribution Aligned Semantics Adaption for
Lifelong Person Re-Identification
Qizao Wang1 , Xuelin Qian2 , Bin Li1 , Xiangyang Xue1*
1 Fudan University, Shanghai, China.

arXiv:2405.19695v2 [cs.CV] 8 Feb 2025

2 Northwestern Polytechnical University, Xi’an, China.

*Corresponding author(s). E-mail(s): xyxue@fudan.edu.cn;
Contributing authors: qzwang22@m.fudan.edu.cn; xlqian@nwpu.edu.cn;
libin@fudan.edu.cn;
Abstract
In real-world scenarios, person Re-IDentification (Re-ID) systems need to be
adaptable to changes in space and time. Therefore, the adaptation of Re-ID models to new domains while preserving previously acquired knowledge is crucial,
known as Lifelong person Re-IDentification (LReID). Advanced LReID methods
rely on replaying exemplars from old domains and applying knowledge distillation
in logits with old models. However, due to privacy concerns, retaining previous
data is inappropriate. Additionally, the fine-grained and open-set characteristics of Re-ID limit the effectiveness of the distillation paradigm for accumulating
knowledge. We argue that a Re-ID model trained on diverse and challenging
pedestrian images at a large scale can acquire robust and general human semantic knowledge. These semantics can be readily utilized as shared knowledge for
lifelong applications. In this paper, we identify the challenges and discrepancies associated with adapting a pre-trained model to each application domain
and introduce the Distribution Aligned Semantics Adaption (DASA) framework.
It efficiently adjusts Batch Normalization (BN) to mitigate interference from
data distribution discrepancy and freezes the pre-trained convolutional layers to
preserve shared knowledge. Additionally, we propose the lightweight Semantics
Adaption (SA) module, which effectively adapts learned semantics to enhance
pedestrian representations. Extensive experiments demonstrate the remarkable
superiority of our proposed framework over advanced LReID methods, and it
exhibits significantly reduced storage consumption. DASA presents a novel and
cost-effective perspective on effectively adapting pre-trained models for LReID.
The code is available at https://github.com/QizaoWang/DASA-LReID.

1

Samples

…

New
Model

(b) Comparison of using the
same pre-trained weights

(a) Knowledge distillation paradigm

…

Old
Model
Exemplars

New
Samples

Knowledge
Distillation
Loss

…

New
Model

(b) Knowledge distillation paradigm

(a) Results of using pre-trained weights

Fig. 1: (a) Comparison of average incremental performance for different methods
using the same pre-trained weights. (b) Conventional knowledge distillation pipeline
with exemplars. Our proposed DASA paradigm shows great superiority in LReID.

Keywords: Person re-identification, Lifelong learning, Distribution alignment,
Semantics adaption

1 Introduction
Person Re-IDentification (Re-ID) aims at recognizing the same pedestrian across disjoint cameras. With significant advancements in deep learning algorithms over the
past decade, it has demonstrated remarkable performance [1–7]. However, advanced
methods usually assume the pedestrian data is provided at once. In real-world scenarios, surveillance data accumulates continuously due to successive deployments and
ongoing applications. Re-ID data expands discretely, both spatially and temporally,
necessitating the adaptation of Re-ID models to new environments. Consequently, Lifelong person Re-IDentification (LReID) seeks to continuously adapt Re-ID models to
novel domains while preserving previously acquired knowledge during the incremental
learning process. Unfortunately, due to substantial variations between different Re-ID
domains, merely updating the model with new surveillance data can lead to a loss of
discriminative ability in previously learned domains. This phenomenon is commonly
referred to as the catastrophic forgetting problem.
Recently, we have witnessed the remarkable success of pre-training in computer
vision [8, 9], which also revolutionizes the field of person Re-ID. For example, Fu et al.
[10, 11] adopt unsupervised pre-training to prevailing Convolutional Neural Network
(CNN) architectures like ResNet [12] and achieve significant performance improvement. It is promising to leverage pre-trained models that can be obtained effortlessly
for real-world applications. Nevertheless, as observed in Fig. 1 (a), directly using the
pre-trained weights for model initialization yields limited benefits in the lifelong evolution scenario. Advanced LReID methods [13–16] follow the prevailing knowledge
distillation paradigm [17]. Thus we are motivated to think whether the performance
degradation stems from the inherent drawbacks of the knowledge distillation paradigm.

2

As depicted in Fig. 1 (b), existing LReID methods follow the distillation baseline
to distill knowledge at the logit level. This paradigm heavily relies on the learned
classifier. However, unlike conventional lifelong learning [17–20], LReID is inherently
a fine-grained open-set problem. The distillation paradigm would restrict the model’s
ability to incorporate valuable knowledge for recognizing unseen identities during inference. It also limits person Re-ID in real-world applications in two aspects. Firstly,
relying on replaying data of old tasks, namely exemplars [21–24], is impractical due
to strict privacy constraints associated with pedestrian data. Secondly, the number
of classes in the LReID datasets significantly exceeds those in conventional lifelong
learning tasks, such as ImageNet [25]. The high dimensional classifier would consume
great resources when saving it for knowledge distillation.
Therefore, to achieve a balance between knowledge preservation and updating with
minimal source consumption, and without using exemplars, we introduce the Distribution Aligned Semantics Adaption (DASA) framework, which presents a novel
LReID paradigm based on pre-trained models. In this paradigm, we do not rely on
knowledge distillation that uses saved old data and models for knowledge retention.
Instead, we exert the advantages of the pre-trained model in the lifelong learning process. Specifically, we carefully handle and leverage the two crucial techniques in CNN,
i.e., Batch Normalization (BN) and Convolutional (Conv) blocks. Firstly, BN effectively regularizes the model and captures data distributions, contributing to improved
learning [26, 27]. In our proposed paradigm, we tune BN effectively to deal with
domain distribution differences between pre-training and application. Secondly, Conv
layers specialize in encoding data patterns and semantics at different depths of the
network. It is expected that the Conv layers trained on a substantial amount of pedestrian data can acquire robust and generalizable human semantic knowledge [10, 11],
thereby serving as a valuable source of shared knowledge across domains. Therefore,
we propose freezing the pre-trained Conv layers to preserve shared person semantic
knowledge. However, general semantics are not capable of distinguishing pedestrians
in complex application scenarios. To adapt the acquired general knowledge to each
application domain effectively, we introduce the lightweight Semantics Adaption (SA)
module, which can efficiently aggregate, refine, and generate improved pedestrian representations. Adapting the general human semantics to more representative identity
semantics represents a process from coarse to fine, showing great ability to evolve the
model for lifelong application. Our contributions can be summarized as follows:
(1) We advocate taking advantage of the robust and general human semantic
knowledge acquired through large-scale pre-training to effectively adapt models in
application domains, presenting a new paradigm free of exemplars for LReID.
(2) We propose the Distribution Aligned Semantics Adaption (DASA) framework
for LReID. It eliminates the interference of domain distribution discrepancy between
pre-training and application, and incorporates the lightweight semantics adaption
module to aggregate and refine learned semantics for better pedestrian representations.
(3) Extensive experiments demonstrate the superiority of the proposed framework for LReID, achieving state-of-the-art results with significantly reduced resource
consumption compared to other advanced LReID methods.

3

2 Related Work
2.1 Lifelong Learning
Lifelong learning, also known as incremental or continual learning, seeks to maintain
stable performance on old tasks while adapting the model to gain new knowledge.
Methods can be traditionally divided into three categories, namely regularizationbased, rehearsal-based, and architecture-based methods. Regularization-based methods [17, 18] limit updating important parameters for preceding tasks to mitigate
forgetting. To better keep past acquired knowledge, rehearsal-based methods [21, 23,
24] maintain a memory buffer to store finite exemplars of previous tasks. They are
prevalent for their satisfactory performance and simplicity. However, rehearsal-based
methods generally deteriorate with a smaller buffer size and do not apply to scenarios
where data privacy should be considered carefully. Architecture-based methods design
separate components or extra parameters for new tasks, so they are immune to forgetting. They expand the network with task-specific components [28, 29] or attend to
task-specific sub-networks [30, 31]. For instance, DER [32] dynamically adds a new
feature extractor per task, and HAT [31] learns a hard attention mask concurrently to
every task. Although lifelong learning has received extensive research in classification
or recognition tasks, the presence of imbalanced samples per identity and subtle interclass variations in person Re-ID poses additional challenges. The significant variations
across domains make the problem of catastrophic forgetting even more thorny.

2.2 Lifelong Person Re-Identification
Recently, there has been remarkable progress in person re-identification leveraging
pre-prepared stationary training data [1, 3, 5, 33–35]. In response to the demand
for long-term scenarios, Pu et al. [13] introduce Lifelong person Re-IDentification
(LReID) and propose to maintain a learnable knowledge graph to adaptively update
previous knowledge. However, due to significant domain variations, it struggles to
retain old knowledge without access to previous data. Recently, state-of-the-art LReID
methods have drawn inspiration from rehearsal-based lifelong learning approaches.
They adopt knowledge distillation baselines designed for traditional lifelong learning to preserve acquired knowledge. For instance, based on the distillation paradigm,
GwFReID [16] formulates a comprehensive learning objective for maintaining coherence during progressive learning. PTKP [14] proposes a pseudo task knowledge
preservation framework to alleviate the domain gap in the last BN layer. KRC [15]
introduces a dynamic memory model for bi-directional knowledge transfers and a
knowledge consolidation scheme. However, the intrinsic dissimilarities between Re-ID
with the classification task restrict these methods from effectively incorporating and
leveraging useful knowledge for LReID. Moreover, privacy concerns limit their applicability in real-world scenarios, where exemplars cannot be stored for lifelong usage.
The recent work Teata [36] achieves knowledge alignment, transfer, and accumulation
within an “image-text-image” closed loop, leveraging the advantages of text semantics
while avoiding reliance on exemplars.

4

In LReID, we aim to achieve a balance between knowledge preservation and
updating with minimal source consumption, and without using exemplars. It differs
from existing studies in Test-Time Adaption (TTA) [37, 38] and Domain Generation
(DG) [39, 40]. Specifically, in TTA, the pre-trained model is optimized during inference
[37]. For instance, BNTA [38] adapts the model using gallery data and self-supervised
auxiliary tasks, and TEMP [41] uses query images and gallery features to minimize
entropy at test time. Differently, in LReID, the model is updated during training and
directly tested on seen and unseen domains. In DG, studies aim to improve the generalization performance of the model on unseen domains [39, 40], and the model is usually
trained on multiple domains to learn domain-invariant representations. In contrast, in
LReID, the model is optimized with various domains sequentially and the data from
previous domains is inaccessible, so knowledge forgetting will be a great challenge.


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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | sed -n '1,149p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Dual-level Modality Debiasing Learning for Unsupervised
Visible-Infrared Person Re-Identification⋆
Jiaze Lia,1 , Yan Lub,1 , Bin Liua,∗, Guojun Yina , Mang Yec
a University of Science and Technology of China, Hefei, 230026, China
b Shanghai Artificial Intelligence Laboratory, Shanghai, 200233, China

arXiv:2512.03745v2 [cs.CV] 9 Apr 2026

c the School of Computer Science, Wuhan University, Wuhan, 430072, China

Abstract
Two-stage learning pipeline has achieved promising results in unsupervised visibleinfrared person re-identification (USL-VI-ReID). It first performs single-modality learning and then operates cross-modality learning to tackle the modality discrepancy. Although promising, this pipeline inevitably introduces modality bias: modality-specific
cues learned in the single-modality training naturally propagate into the following
cross-modality learning, impairing identity discrimination and generalization. To address this issue, we propose a Dual-level Modality Debiasing Learning (DMDL) framework that implements debiasing at both the model and optimization levels. At the
model level, we propose a Causality-inspired Adjustment Intervention (CAI) module
that replaces likelihood-based modeling with causal modeling, preventing modalityinduced spurious patterns from being introduced, leading to a low-biased model. At the
optimization level, a Collaborative Bias-free Training (CBT) strategy is introduced to
interrupt the propagation of modality bias across data, labels, and features by integrating modality-specific augmentation, label refinement, and feature alignment. Extensive
experiments on benchmark datasets demonstrate that DMDL could enable modalityinvariant feature learning and a more generalized model. The code is available at
https://github.com/priester3/DMDL.
⋆ This work is supported by the National Natural Science Foundation of China (Grant No. 62272430).
∗ Corresponding author

Email addresses: jz_li@mail.ustc.edu.cn (Jiaze Li), luyan@pjlab.org.cn (Yan Lu),
flowice@ustc.edu.cn (Bin Liu), gjyin@mail.ustc.edu.cn (Guojun Yin), yemang@whu.edu.cn
(Mang Ye)
1 These authors contributed equally to this work.

Keywords: Visible-infrared person re-identification, Unsupervised learning, Causal
intervention, Modality-invariant feature

1. Introduction
Visible-infrared person re-identification (VI-ReID) focuses on the identification
and matching of individuals across distinct modalities, visible and infrared. Remarkable progress has been made in this field, as evidenced by the success of existing works
[1, 2]. However, the collection of extensive cross-modality annotations is a costly and
time-consuming process, which poses limitations on its broader application. As a solution, Unsupervised Visible-infrared Person Re-identification (USL-VI-ReID) [3, 4, 5]
has emerged to facilitate VI-ReID without the reliance on human identity labels.
The main challenge in the USL-VI-ReID is the modality discrepancy, which limits the direct application of standard unsupervised learning of traditional unsupervised
ReID. Therefore, the mainstream methods for USL-VI-ReID typically follow a twostage learning pipeline [4, 5, 6, 7]: 1) In the first stage, the model is trained by operating unsupervised learning techniques [8] on each modality separately to have the
single-modality discriminative ability. 2) In the second cross-modality unsupervised
process, the model alternately establishes relationships across modalities and fits these
relationships to achieve cross-modality discrimination capabilities. Although promising, it also suffers from a modality bias issue that restricts the overall results. The first
single-modality learning process naturally captures modality-specific cues from visible/infrared data, resulting in a biased model. Initializing the second stage with this
model inevitably introduces modality bias into the cross-modality learning, leading to
biased cross-modality relationships, e.g., similar clothing color cues may result in incorrect matches across modalities, as illustrated in Fig. 1 (a). Since cross-modality
relationships (i.e., pseudo labels) are the model-fitting target in the second stage, the
biased knowledge (i.e., modality-specific cues) is gradually enhanced in the learned
patterns, leading to modality-related features. In summary, modality bias originating
from data propagates into labels and features throughout the learning pipeline, leading
the model to rely on modality-specific cues for identification and thereby significantly

2

Color

Identity

biased likelihood
model

causal model

(a) Existing biased USL-VI-ReID

(b) Our Modality Debiasing

Figure 1: Existing USL-VI-ReID methods suffer from modality bias, leading to modality-related features.
In contrast, our approach achieves modality-invariant feature learning through causal modeling and unbiased
optimization. Green, yellow, and blue circles represent visible-specific, infrared-specific, and modalityshared information, respectively.

limiting its generalization.
To address the aforementioned modality bias issue, we propose a Dual-level Modality Debiasing Learning (DMDL) framework. DMDL performs modality debiasing at
both the model and optimization levels, where the former prevents the model from
learning modality bias in incorrect cross-modality relationships, and the latter aims to
disrupt the propagation of biased knowledge from data to labels and features directly.
To this end, a Causality-inspired Adjustment Intervention (CAI) module and a Collaborative Bias-free Training (CBT) strategy are proposed. Specifically, CAI facilitates
causal intervention under cross-modality unsupervised learning with backdoor adjustment, making the model only capture the causal patterns. Compared with the traditional
likelihood method, the causal modeling in CAI is theoretically unaffected by modality bias, thereby achieving a low-biased model. To further prevent biased knowledge
from deepening during optimization, we propose the CBT strategy, integrating data
augmentation, label refinement, and feature alignment. CBT first introduces a pseudomodality augmentation scheme to modify modality-specific cues in images. Based on
the augmented images, a cross-modality label smoothing scheme and a feature alignment loss are proposed to refine the biased relationships and learn shared knowledge
across pseudo-modalities, respectively. By jointly leveraging these components, CBT
explicitly interrupts the propagation of modality bias across data, labels, and features.
Ultimately, the overall DMDL keeps an effective modality debiasing implementation,

3

achieving modality-invariant feature learning, as Fig. 1 (b) shows.
Our main contributions are summarized as follows:
(1) We investigate the modality bias issue for existing USL-VI-ReID methods and propose a Dual-level Modality Debiasing Learning (DMDL) framework performed at
both the model and optimization levels to learn modality-invariant feature representations.
(2) We propose a Causality-inspired Adjustment Intervention (CAI) module at the
model level to effectively model the causal patterns, constructing a low-biased
model.
(3) We propose a Collaborative Bias-free Training (CBT) strategy at the optimization
level, combining label refinement and feature alignment with modality-specific
data augmentation to prevent fitting biased knowledge.
(4) Extensive experiments conducted on standard visible-infrared ReID benchmarks
demonstrate the effectiveness and superiority of our method.

2. Related Work
2.1. Unsupervised Visible-Infrared Person ReID
Traditionally, visible–infrared ReID and unsupervised ReID were studied as two
largely independent tasks. For both image-level [9] and video-level [10, 11] VI-ReID,
the core objective is to construct a cross-modality identity-discriminative space that is
consistent across visible and infrared domains. In contrast, unsupervised ReID [12, 13]
typically focuses on exploiting multi-view information or local feature interactions to
generate reliable pseudo labels, thereby enabling the learning of discriminative representations without manual annotations.
By integrating these two paradigms, USL-VI-ReID naturally emerges as a promising research direction without requiring any human annotations. Most existing approaches adopted a two-stage pipeline to mitigate the significant modality discrepancy,
and most of them aimed at exploring reliable cross-modality correspondences. For instance, PGM [4] and MBCCM [5] utilized graph matching to establish reliable relationships across modalities globally, while DOTLA [14] leveraged optimal transport for
4

cross-modality matching. Other methods, such as MULT [15] and DLM [16], designed
a more complex matching scheme by integrating cluster-level matching with instancelevel structures to enhance the reliability of cross-modality association. PCLHD [6]
revisited prototype construction in contrastive learning to explore more reliable clustering. Moreover, ASM [17] improves the robustness of pseudo labels to color variations by integrating the similarity of augmented images during matching. For the
unpaired setting, MCL [18] generates pseudo cross-modality positive sample pairs
through cross-modality feature mapping, constructing a pseudo cross-modality identity space to facilitate effective feature alignment. Despite their effectiveness, these
methods are inherently constrained by the two-stage pipeline, which inevitably introduces modality bias and hinders the modality-invariant learning.
In addition, some methods [19, 20, 21] only perform a single stage of cross-modality
learning. Specifically, GUR [19] proposed a bottom-up domain learning strategy that
performs intra-camera training, inter-camera training, and inter-modality training alternately. CHCR [20] designed a cross-modality hierarchical clustering baseline that
first refines clusters within each modality before merging them cross-modally based
on similarity. SDCL [21] proposed a shallow-deep collaborative learning framework
that initializes with a pre-trained model of single-modality ReID. Although explicitly
abandoning the two-stage pipeline, these methods still suffer from the modality bias
issue since they involve single-modality training or clustering.
2.2. Person ReID with Causal Inference
Incorporating causal inference [22] into deep learning models, enabling them to
learn causal effects, can enhance the performance across various applications. There
has been research exploring the integration of causal inference into person ReID models. For instance, CIFT [23] utilized counterfactual interventions and causal effect
tools to make the graph topology structure more reliable for the VI-ReID graph model.
Zhang et al.[24] approximated causal interventions on domain-specific factors to achieve
domain-invariant representation learning for generalizable ReID. Both AIM [25] and
CCIL [26] employed causal intervention models to learn clothing-invariant features for
cloth-changing person ReID. These methods cannot be applied in the USL-VI-ReID
5

task since they are designed to mitigate biases caused by domain and clothing rather
than modality.
2.3. Person ReID with Noise Label Learning
Due to the limited availability of clean annotations in practice, cluster-based unsupervised ReID methods commonly adopt noise label learning mechanisms to refine pseudo-labels and stabilize model training. For example, STDA [27] aggregates
spatial-level neighborhood consistency to refine pseudo-labels, while PPLR [28] reduces label noise by integrating global and partial predictions with label smoothing.
However, these methods mainly operate on single-modality clustering, leveraging spatial or fine-grained contextual cues, and thus struggle to correct erroneous cross-modality
relationships. In the USL-VI-ReID task, DPIS [29] and MMM [30] incorporate noisylabel learning by fitting a two-component Gaussian Mixture Model (GMM) to the loss
distribution to estimate label confidence, which is then used to penalize noisy samples
during optimization. In contrast to such penalization-based strategies, we exploit the
estimated confidence to explicitly revise pseudo-labels, thereby mitigating modality
bias at the label level rather than merely suppressing its effect.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf' - | sed -n '1,251p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 189 (2025) 107602

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

DiverseReID: Towards generalizable person re-identification via Dynamic
Style Hallucination and decoupled domain experts
Jieru Jia ∗, Huidi Xie, Qin Huang , Yantao Song , Peng Wu
Institute of Big Data Science and Industry, Shanxi University, Taiyuan 030006, China
School of Computer and Information Technology, Shanxi University, Taiyuan, China
Engineering Research Center for Machine Vision and Data Mining of Shanxi Province, Shanxi University, Taiyuan 030006, China

ARTICLE

INFO

Keywords:
Person re-identification
Domain generalization
Data augmentation
Mixture of experts

ABSTRACT
Person re-identification (re-ID) models often fail to generalize well when deployed to other camera networks
with domain shift. A classical domain generalization (DG) solution is to enhance the diversity of source
data so that a model can learn more domain-invariant, and hence generalizable representations. Existing
methods typically mix images from different domains in a mini-batch to generate novel styles, but the mixing
coefficient sampled from predefined Beta distribution requires careful manual tuning and may render suboptimal performance. To this end, we propose a plug-and-play Dynamic Style Hallucination (DSH) module that
adaptively adjusts the mixing weights based on the style distribution discrepancy between image pairs, which is
dynamically measured with the reciprocal of Wasserstein distances. This approach not only reduces the tedious
manual tuning of parameters but also significantly enriches style diversity by expanding the perturbation space
to the utmost. In addition, to promote inter-domain diversity, we devise a Domain Experts Decoupling (DED)
loss, which constrains features from one domain to go towards the orthogonal direction against features from
other domains. The proposed approach, dubbed DiverseReID, is parameter-free and computationally efficient.
Without bells and whistles, it outperforms the state-of-the-art on various DG re-ID benchmarks. Experiments
verify that style diversity, not just the size of the training data, is crucial for enhancing generalization.

1. Introduction
Person re-identification (re-ID) aims to match pedestrian images of
the same identity in a non-overlapping camera system. Despite their
almost saturated performance in the closed-world supervised learning
testbed, most current methods suffer from drastic performance degradation if applied to a different camera network with distribution shifts.
This prevents a single re-ID system from being successfully applied to
real-world scenarios in an off-the-shelf manner.
To overcome this limitation, domain generalization (DG) is a
promising solution with the goal of training a model that generalizes
well to arbitrary unseen domains out-of-the-box, without any model
re-training or fine-tuning. This is a more challenging but practical task
compared with unsupervised domain adaptation (UDA) (Ge, Zhu, Chen,
Zhao, et al., 2020; Zhang, Li et al., 2022), as it does not require data
collection or any prior knowledge about the unseen target domain for
model training purposes. Moreover, unlike most DG methods (Huang
et al., 2023; Sun, Liu, Liu, Li, & Chu, 2023; Wang, Zhang, Lei, & Zhang,
2023) which assume a homogeneous environment where different
domains share the same label space, DG re-ID is a more challenging

heterogeneous problem, in that there is no identity overlap between
source and target domains.
To enhance the robustness to domain shifts, existing DG methods intend to learn domain-invariant representations across source
domains, mainly via data augmentation (Yang et al., 2024; Zhang,
Cisse, Dauphin, & Lopez-Paz, 2018; Zhou, Yang, Qiao & Xiang, 2020),
normalization techniques (Choi, Kim, Jeong, Park, & Kim, 2021; Jia,
Ruan, & Hospedales, 2019; Jin, Lan, Zeng, Chen, & Zhang, 2020), novel
network architecture designs (Dou, Wang, Li, & Wang, 2023; Liao &
Shao, 2020; Zhang, Dou, Yu & Li, 2022), etc. Another popular way
to address the DG problem is to develop novel learning strategies,
which can be roughly divided into but not limited to: (a) adversarial
learning (Lin, Li, & Kot, 2020); (b) mixture of experts (Dai, Li, Liu,
Tong, & Duan, 2021; Xu, Liang, He, & Sun, 2022) that learn a system
composed of many separated networks (experts); (c) meta-learning (Ni
et al., 2022; Zhao et al., 2021) that learns general knowledge by
constructing meta-learning tasks to simulate domain shift.
Among recent DG methods, feature-based data augmentation approaches like MixStyle (Zhou, Yang, Qiao et al., 2020), pAdaIN (Nuriel,

∗ Correspondence to: 030006, #92 Wucheng Street, Taiyuan, Shanxi Province, China.

E-mail address: jierujia@sxu.edu.cn (J. Jia).
https://doi.org/10.1016/j.neunet.2025.107602
Received 5 December 2024; Received in revised form 24 March 2025; Accepted 7 May 2025
Available online 24 May 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 189 (2025) 107602

J. Jia et al.

Benaim, & Wolf, 2021), and DSU (Li et al., 2022) et al. have exhibited
astonishing performance, which substantially stems from the ability to
avoid overfitting to source data by introducing channel-level style perturbations to emulate potential domain shifts. Despite their popularity
and superiority in domain generalization tasks, existing style augmentation methods are limited in: (1) when synthesizing novel styles, the
weights for two instances are obtained by sampling from the manually
designed distributions, which requires careful hyper-parameter tuning
and may be inadequate to explore a broader perturbation space. More
importantly, any fixed value of the parameters would become suboptimal at some iteration during training; (2) their success relies on the
enriched diversity of newly produced samples, which could not always
be guaranteed in practice since the generated samples may remain
closely to existing ones in the feature space. The desirable superiority
disappears when synthesized sample diversity is insufficient.
This paper attempts to solve the above two crucial issues and
proposes a novel framework dubbed DiverseReID, which expands both
inter-domain and intra-domain style space to maximize the source
diversity for generalized re-ID. Particularly, to resolve the first issue
and achieve intra-domain style diversification, we propose a Dynamic
Style Hallucination (DSH) module that dynamically generates new
styles by mining the relations across sample pairs in an adaptive
manner. Specifically, we leverage the Wasserstein distance to estimate
the style distribution discrepancy between the input image pairs, and
the reciprocal of similarity distance is utilized as the mixing coefficients. The DSH module could alleviate the tedious manual tuning of
parameters. Moreover, instead of a constant coefficient among all the
sample pairs, the DSH module adaptively adjusts the weighting factors
according to the features of different instances, which can expand a
broader perturbation space and significantly enrich the intra-domain
style diversity.
For the second issue, to achieve inter-domain variety, we maintain
a domain-specific memory for each domain (also termed as domain experts) and devise a Domain Experts Decoupling (DED) loss to constrain
the features from one domain to go towards the orthogonal direction
against other domains. The DED loss is implemented by minimizing the
squared Frobenius norm of the diagonal elements in the correlation
matrix. In this way, the domain-specific features are guaranteed to
be dissimilar from each other, which provides a substantial degree
of artificial domain shift and thereby remarkably reduces the risk of
overfitting to source training data. In summary, our main contributions
can be summarized in three aspects:

methods focus on diversifying the source data by artificially inducing
style perturbations in the training data. For instance, MixStyle (Zhou,
Yang, Qiao et al., 2020) combines the statistics of two instances to
synthesize novel samples. DomainMix (Wang, Liao, Zhao, Kang, & Shao,
2021) combines a labeled synthetic dataset with unlabeled real-world
data to learn domain-invariant features. PartMix (Kim, Kim, Park, Park,
& Sohn, 2023) synthesizes the augmented samples by mixing the part
descriptors across modalities. DACS (Yang et al., 2024) designs a style
transformation model with diversity and authenticity constraints to
generate diverse and authentic data.
Domain-invariant Representation learning approaches exploit
normalization techniques or novel network architecture designs to reduce the domain gap in the feature space. For example, DualNorm (Jia
et al., 2019) exploits instance normalization (IN) to filter out variations
in style statistics in earlier layers. SNR (Jin et al., 2020) filters out
identity-irrelevant interference and alleviates the loss of discriminative
information with an attention mechanism. MetaBIN (Choi et al., 2021)
combines batch normalization (BN) and IN with learnable balancing
parameters. DTIN (Jiao et al., 2022) introduces a novel normalization
module by employing dynamic convolutions. As for network architecture designs, QAConv (Liao & Shao, 2020) constructs query-adaptive
convolution kernels to find local correspondences in feature maps.
ACL (Zhang, Dou et al., 2022) proposes a Cross-Domain Embedding
Block to capture the domain-invariant and domain-specific features.
ISR (Dou et al., 2023) puts forward a reliability-guided contrastive
learning method to enable positive pairs to dominate the representation
learning process. MLRAT (Lin et al., 2024) proposes a multi-level
relation-aware Transformer to extract fine-grained features by learning
structural relations among key patches. ReNorm (Nie, Ding, Zhou, & Li,
2024) resolves statistical mismatch in normalization layers by mixing
statistics and freezing affine parameters, helping the model adapt to
cross-domain distribution shifts.
Lastly, some works focus on designing novel learning strategies to
learn domain-agnostic representations. For instance, MMFA-AAE (Lin
et al., 2020) employs the adversarial auto-encoder module to extract
domain-invariant features. ADNMT (Zhang, Li, Ma, & Wang, 2024)
introduces an asymmetric dual-network mutual teaching framework
to optimize pseudo-labels’ quality. Mixture of Experts (MoE) provides
another promising solution to deal with the domain gap. For instance,
META (Xu et al., 2022) functions as an ensemble of multiple experts
sharing all the parameters except for the domain-specific BN layers.
RaMoE (Dai et al., 2021) puts forward a voting network to integrate
all the experts’ features with domain relevance. Meta-learning methods
focus on simulating the training/testing discrepancy. For instance,
M3L (Zhao et al., 2021) proposes a meta BN module to integrate
domain information from the meta-train datasets into the meta-test
features. MDA (Ni et al., 2022) designs a meta-learning strategy to
simulate train–test domain shifts with prior distribution alignment
and distribution-guided refining. As an interesting alternative to metalearning, Tan, Ding, Wang, Gong, and Jia (2023) propose an efficient
Interleaved Learning (IL) technique, which simulates the domain gap
by introducing a shift between the feature extractor and classifier.
Despite the exciting progress in recent years, existing methods
often overlooked the importance of data diversity in enhancing generalization. Different from previous data augmentation methods that
mainly focus on enlarging the size of the training data, our research
underscores the effect of data diversity in DG, and expands both interdomain and intra-domain style space to maximize the source diversity.
The proposed DiverseReID is efficient but not sophisticated, and its lowcost and easy-to-use merit will be of broad interest and use for driving
future research.

• We propose a new Dynamic Style Hallucination module that
dynamically generates novel styles by mining the relations across
sample pairs in an adaptive manner, which significantly enriches
intra-domain diversity and spares the requirement of manual
parameter tuning.
• We develop a novel Domain Experts Decoupling loss for domain
generalization that guarantees the disparity between individual
source domains’ styles, which is highly efficient in disentangling
inter-domain representations and exploiting the corresponding
domain’s powerful characteristics sufficiently.
• We perform extensive experiments on multiple DG re-ID benchmarks, which show that our method consistently outperforms
state-of-the-art methods by significant margins.
2. Related work
2.1. Domain generalizable person re-ID
The goal of generalizable person re-identification is to learn a model
from one or multiple related source domains that generalizes well
to unseen target domains. Existing methods realize this goal from
various perspectives, including data augmentation, domain-invariant
representation learning, training strategy, etc. Data augmentation

2.2. Style manipulation for domain generalization
The proposed work is mainly dedicated to diversifying the styles
of the source data, i.e. data augmentation, which aids in improving
2

Neural Networks 189 (2025) 107602

J. Jia et al.

the model’s generalization ability by simulating a broader range of
scenarios during training. For instance, MixStyle (Zhou, Yang, Qiao
et al., 2020) combines the statistics of two instances to synthesize novel
samples. pAdaIN (Nuriel et al., 2021) swaps feature statistics between
samples in one batch. DSU (Li et al., 2022) imposes disturbance on the
original feature statistics by adding Gaussian noise. MOEX (Li, Wu, Lim,
Belongie, & Weinberger, 2021) encourages the model to extract training
signals from the moment information. SagNet (Nam, Lee, Park, Yoon,
& Yoo, 2021) introduces a style randomization module by interpolating
the feature statistics between different examples. SHADE (Zhao, Zhong,
Zhao, Sebe, & Lee, 2022) presents a style hallucination module that
selects basis styles from the source distribution, and simulates potential
style variations through combination and perturbation. SPC (Huang
et al., 2023) projects unseen styles into the style representation space
built on known domains. SIRL (Chang, Yang, Luo, Ji, & Wang, 2023)
devises an Adversarial Style Synthesis module that consists of a multibranch style generation and fusion strategy. DSM (Li, Song, Ni, & Shen,
2023) maintains a single-domain style queue for each source domain to
enable style mixing. DACS (Yang et al., 2024) designs a style transformation model with diversity and authenticity constraints, enabling it
to generate styles that closely resemble the true data distribution.
While delivering superior performance, there are several limitations
with existing style augmentation works: (1) their efficacy actually heavily depends on delicately designed weight parameters, whose optimal
combinations are time-consuming to acquire; (2) existing methods produce new styles via merely mixing the existing styles from few observed
source domains with the linear interpolation mechanism, leading to
insufficient diversity which may hamper the model optimization. In
this paper, we explicitly consider and explore the style diversity in
improving the model’s generalization ability. Unlike previous methods that generate new data with weights sampled from pre-defined
distributions with a fixed parameter, we devise a novel strategy to
ensure sample diversity by formulating the weights in an adaptive
manner, which is capable of broadening the perturbation space and
providing a substantial degree of style diversity. The proposed DSH
module remarkably boosts the DG re-ID performance and can be easily
integrated with other methods in a plug-and-play manner.

previous works in that the DED loss is proposed to achieve inter-domain
diversification, which is equally crucial as intra-domain style diversity
for enhancing DG performance. Experiments and ablation studies show
that DED loss is orthogonal to the intra-domain DSH module and they
can be combined to potentially further boost the re-ID performance.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf' - | sed -n '1,545p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 886ms:
1

Domain Consistency Representation Learning for Lifelong Person
Re-Identification

arXiv:2409.19954v4 [cs.CV] 3 Sep 2025

Shiben Liu

, Huijie Fan

, Qiang Wang

, Weihong Ren
Member, IEEE

Abstract—Lifelong person re-identification (LReID) exhibits a
contradictory relationship between intra-domain discrimination
and inter-domain gaps when learning from continuous data.
Intra-domain discrimination focuses on individual nuances (i.e.,
clothing type, accessories, etc.), while inter-domain gaps emphasize domain consistency. Achieving a trade-off between maximizing intra-domain discrimination and minimizing inter-domain
gaps is a crucial challenge for improving LReID performance.
Most existing methods strive to reduce inter-domain gaps through
knowledge distillation to maintain domain consistency. However,
they often ignore intra-domain discrimination. To address this
challenge, we propose a novel domain consistency representation
learning (DCR) model that explores global and attribute-wise representations as a bridge to balance intra-domain discrimination
and inter-domain gaps. At the intra-domain level, we explore
the complementary relationship between global and attributewise representations to improve discrimination among similar
identities. Excessive learning intra-domain discrimination can
lead to catastrophic forgetting. We further develop an attributeoriented anti-forgetting (AF) strategy that explores attribute-wise
representations to enhance inter-domain consistency, and propose
a knowledge consolidation (KC) strategy to facilitate knowledge
transfer. Extensive experiments show that our DCR achieves superior performance compared to state-of-the-art LReID methods.
Our code is available at https://github.com/LiuShiBen/DCR.

P

ERSON re-identification (ReID) aims to retrieve the
same individual across multiple cameras in a largescale database by using uni-modal architectures such as
convolutional neural networks (CNN) [1]–[3] or vision
transformers (ViT) [4]–[6]. However, when ReID models

, Yang Cong

Global
representations center
attribute category
ID 2

ID 1

ID 3

ID 4

Step t-1

, Senior

class boundary
domain boundary
ID 6

ID 5

Step t
Knowledge
Distllation
Minimize
inter-domain gaps
Step t

Step t-1

(a) Existing Methods
Man/woman
Shorts
Trousers
Backpack
Shoulder bag
Long sleeve
Short sleeve

Index Terms—Lifelong person re-identification, attribute-text
generator, text-image aggregation, domain consistency representation.

I. I NTRODUCTION

, Yandong Tang

Step t-1
Step t
Minimize inter-domain gaps
Maximize intra-domain discrimination
(b) Ours
Fig. 1. Comparison between our method and existing methods. (a) Existing
methods [9], [10] leverage knowledge distillation to minimize inter-domain
gaps but ignore intra-domain discrimination, which limits the LReID model’s
ability to learn new knowledge. (b) Our method explores domain consistency
representations as a bridge to achieve a trade-off between maximizing intradomain discrimination and minimizing inter-domain gaps, enhancing the
LReID model’s anti-forgetting and generalization capabilities.

Text
Encoder

This work is supported by the National Natural Science Foundation
of China (62273339, U24A201397), the Key Research and Development
are applied to continuous datasets collected by video-based
Program of Liaoning (2024JH2/102400022)
and the
LiaoNing Revitalization
Attribute-text
generator
Text-guided
aggregation
network
(TGA)
Globa
monitoring systems
[7], [8], they
exhibit notable
performance
Talents Program (XLYC2403128). (Corresponding author: Huijie Fan)
Shiben Liu is with the State KeyALaboratory
Shenyang Institute
limitations. As a result, recent works have focused on the representa
photo ofof aRobotics,
woman
of Automation, Chinese Academy
of Sciences,
110016,
(LReID),
wearing
a longShenyang
sleeved
top, China, and practical problem of lifelong person identification[CLS]
also with the University of Chinese Academy of Sciences, Beijing 100049,
which
maintains
strong
performance
with
continuously
and
a
trousers,
while
China (e-mail: liushiben@sia.cn).
Huijie Fan, and Yandong Tang
are withathe
State Key Laboratory of
updated data streams.
carrying
backpack.
Robotics, Shenyang Institute of Automation, Chinese Academy of Sciences,
[CLS] At present, lifelong person re-identification (LReID) suffers
Shenyang, 110016, China (e-mail: fanhuijie@sia.cn; ytang@sia.cn).
from the challenge of balancing the
PFManti-forgetting of old
Qiang Wang is with the Key Laboratory of Manufacturing Industrial Integrated Automation, Shenyang University, and with the State Key Laboratory
knowledge and learning new knowledge. Specifically, there
of Robotics, Shenyang Institute Attribute
of Automation, Chinese Academy of Sciences,
are two main issues to solve this challenge. 1) Intra-domain
Shenyang, 110016, China (e-mail: wangqiang@sia.cn).
1
Recognition
discrimination. Each identity may exhibit subtle nuances of
Weihong Ren is with the Harbin Institute of Technology, Shenzhen 518055,
2
China (e-mail: renweihong@hit.edu.cn).
individual information (i.e., clothing type, accessories, haircut,
Yang Cong is with the College of Automation Science and Engineering,3
etc.) and lead to severe distribution overlapping. Learning
South China University of Technology, Guangzhou, 510640, China (e-mail:
discriminative representations of individuals are effective for
congyang81@gmail.com).

Image
Encoder

distinguish identity information. 2) Inter-domain gaps. Each

Linear

T_

T_B

T_B

AN

GN

AN＋GN

2

Domain is collected in different illumination and background,
leading to inter-domain gaps. Bridging intra-domain gaps are
significant for mitigating catastrophic forgetting in LReID.
To address these issues, we aim to learn domain
consistency representations that capture individual nuances
in intra-domain and inter-domain consistency in LReID.
Knowledge distillation-based approaches [10]–[12] ensure
distribution consistency between the previous and current
domain to alleviate catastrophic forgetting. However, these
approaches impose strict constraints and ignore intradomain discrimination, [13]–[15], as outlined in Fig. 1(a).
While LReID models significantly improve intra-domain
discrimination for the current step, they inevitably damage
inter-domain consistency, leading to catastrophic forgetting.
Thus, we explore global and attribute-wise representations
to strike a trade-off between maximizing intra-domain
discrimination and minimizing inter-domain gaps, improving
the anti-forgetting and generalization capabilities of the
LReID model, as illustrated in Fig. 1(b).
Specifically, we propose a novel domain consistency
representation learning (DCR) model that first explores
attribute and text information to enhance LReID performance.
Unlike methods [16]–[18], we develop domain consistency
representations including global and attribute-wise
representations to capture individual nuances in intradomain and inter-domain consistency in LReID. We design
an attribute-text generator (ATG) to dynamically generate
text-image pairs for each instance, which are then fed into
a text-guided aggregation (TGA) network to enhance the
global representation capability, effectively distinguishing
identities in LReID. Furthermore, the attributes of each
instance guide an attribute compensation (ACN) network to
generate attribute-wise representations that focus on specific
regional information about identities. We consider that
attributes can enhance reliability by setting higher thresholds
across domains. Therefore, the generated attribute-wise
representations and text for each instance are considered
reliable in our model.
In summary, we aim to strike a balance between
maximizing intra-domain identity-discriminative information
and minimizing inter-domain gaps by exploring global and
attribute-wise representations. At the intra-domain level,
global representations capture whole-body information, while
attribute-wise representations focus on specific regional
information. When whole-body appearances or attributerelated information are similar across identities, we combine
global and attribute-wise representations to distinguish among
similar identities, maximizing intra-domain discrimination.
While the model adapts well to new information, it often
struggles with catastrophic forgetting. To preserve old
knowledge, we develop an attribute-oriented anti-forgetting
(AF) strategy that explores attribute-wise representations
to bridge inter-domain gaps across continuous datasets.
Meanwhile, striking a trade-off between maximizing intradomain discrimination and minimizing inter-domain gaps is
crucial for improving the performance of LReID models.
Knowledge consolidation (KC) is proposed to facilitate
knowledge transfer and enhance generalization capabilities,

which consists of alignment and logit-level distillation
mechanisms. The alignment mechanism explores global
representations of knowledge transfer from the current
dataset. The logit-level distillation mechanism enhances the
extraction of identity information, thereby improving the
model’s ability to consolidate knowledge. Our contributions
are as follows:
We propose a novel domain consistency representation
learning (DCR) model that explores global and attributewise representations to capture individual nuances in
intra-domain and inter-domain consistency, achieving a
trade-off between maximizing intra-domain discrimination and minimizing inter-domain gaps.
• In the intra-domain context, we explore the complementary relationship between global and attribute-wise
representations to enhance the discrimination of each
identity and adapt to new knowledge.
• In the inter-domain context, we design an attributeoriented anti-forgetting (AF) and a knowledge consolidation (KC) strategy to minimize inter-domain gaps
and facilitate knowledge transfer, improving the LReID
model’s generalization and anti-forgetting capabilities.
•

II. R ELATED WORK
A. Lifelong Person Re-Identification
Lifelong Person Re-Identification (LReID) aims to balance
intra-domain discrimination with minimizing inter-domain
gaps in continuously updated datasets across scenarios, improving the model’s anti-forgetting and generalization capabilities. LReID methods can be divided into two categories.
1) Knowledge distillation-based methods [12], [19]–[21] utilize metric strategies to achieve domain-consistent alignment
between the old model with learned knowledge distribution
and the new model that adaptively learns new knowledge. 2)
Exemplar-based methods [9], [10], [22] achieve a distribution
balance between old and new samples to prevent catastrophic
forgetting by forming a memory buffer to select the limited
samples from some identities. These methods strive to reduce
inter-domain gaps and ensure consistency across domains to
prevent catastrophic forgetting. However, they ignore intradomain identity discrimination and lack consistency optimization within the inter-domain, limiting the LReID model’s performance in learning new knowledge. In this paper, we explore
domain consistency representations as a bridge to achieve
a trade-off between maximizing intra-domain discrimination
and minimizing inter-domain gaps for enhancing the antiforgetting and generalization capabilities of the LReID model.
B. Vision-Language for Person Re-Identification
Vision-language learning paradigms [23], [24] have gained
widespread popularity in recent years. Contrastive LanguageImage Pre-training (CLIP) [25], establishes a connection between natural language and visual content through the similarity constraint of image-text pairs. CLIP has been applied
to multiple person re-identification tasks [26]–[28], including text-to-image, text-based single-modality, and text-based

3

Attribute-text generator

Text-guided aggregation network (TGA)

Text
Encoder

A photo of a woman
wearing a long sleeved top,
and a trousers, while
carrying a backpack.

[CLS]

G1

C

G2

[CLS]

PFM

1
2

LayerNorm

LayerNorm

Drop

Drop

CA

CA

GN

Image
Encoder

Attribute
Recognition

Parallel Fusion Module (PFM)
Parallel Fusion Module (PFM)
MLP

Global
representations

3

Linear

Q

AG1

T_Block

T_Block

T_Block
Attribute
semantic information

Attribute
features

Attribute Decoder

G1

A1

G2

AN＋GN

G_1
G_2
G_N

Step t+1

Old model 

AG_1
AG_2
AG_N
t−1

AG_1
AG_2

Step t

Backbone
New model 

t

Step t-1

C
Multiplication
Patch embedding

Position embedding

Hat
Shorts
Trousers
Backpack
Short sleeve
Long sleeve

Old model
distribution

Q

[CLS] embedding
Concatenation

AGN

Attribute-wise
representations

Attribute matching

K,V

K,V

AG2

Attribute compensation network (ACN)

Backbone

Step t-1

GN

AN
A2

N

 G_i

Classifier

Logits

i=1

Step t

Minimize
intra-domain gaps

AG_N

New model
distribution

G_1
G_2

Attribute-oriented anti-forgetting

Push
away
N

 G_i

LAL
Classifier

Push
away

LLD
Maximize
inter-domain
discrimination

Logits

i=1

Knowledge consolidation

G_N

Fig. 2. Overview of the proposed DCR for LReID. First, the attribute-text generator (ATG) dynamically generates text-image pairs for each instance. Then, the
text-guided aggregation network (TGA) captures global representations for each identity, while the attribute compensation network (ACN) generates attributewise representations. We explore the complementary relationship between global and attribute-wise representations to maximize intra-domain discrimination.
Meanwhile, we design attribute-oriented anti-forgetting (AF) and knowledge consolidation (KC) strategies to minimize inter-domain gaps and facilitate
knowledge transfer.

cross-modality. Text-to-image methods [28]–[30] aim to retrieve the target person based on a textual query. Text-based
single-modality works [5], [27], [31] leverage text descriptions
to generate robust visual features or integrate the beneficial
features of text and images for the person category. TextQuery
Proposed
DCR
based
cross-modality methods [32]
utilize text
descriptions to
reduce visible-infrared modality gaps. Providing insufficient
text descriptions of each identity, due to prompt learning and
text inversion. In this paper, we dynamically generate textimage pairs from single images to capture fine-grained global
representations based on the CLIP model for improving model
performance capability in terms of inter-domain and intradomain.
C. Pedestrian Attribute Recognition
Pedestrian attribute recognition aims to assign a set of
attributes (Gender, Bag, Short/Long sleeve, and etc.) to a
visual representation of a pedestrian based on their attributes.
Deep learning-based research [33] automatically learns
hierarchical features from raw images, improving recognition
accuracy. Multi-task learning methods [34]–[36] leverage
additional contextual information from multiple domains, such

as pedestrian detection or pose estimation, to significantly
improve attribute recognition. Part-based methods [37], [38]
divide the pedestrian image into several parts or regions,
providing more accurate localization. Currently, these methods
have achieved significant success in improving the accuracy
of attribute recognition. WeDKP
are Method
the first to explore the
application of attributes to LReID from two perspectives.
1) Attributes are converted into text descriptions for each
image to enhance global representation capabilities. 2) The
attributes are transformed into attribute-wise representations
by specific networks to maximize intra-domain discrimination
and minimize intra-domain gaps.


web search: 
web search: "Distribution Aligned Semantics Adaption" lifelong person re-identification DASA arXiv
web search: 
web search: "Disentangling Modality and Posture Factors" "Visible-Infrared Person Re-Identification"
web search: 
web search: "DMPF" "Visible-Infrared Person Re-Identification" modality posture factors
web search: 
web search: Disentangling Modality Posture Factors Memory Attention Orthogonal Decomposition Visible Infrared Person Re Identification TNNLS 2025
web search: 
web search: "10.1109/TNNLS.2024.3384023"
web search: 
web search: site:sciencedirect.com/science/article/pii/ DiverseReID Towards generalizable person re-identification Dynamic Style Hallucination
web search: 
web search: DiverseReID Towards generalizable person re-identification via Dynamic Style Hallucination decoupled domain experts
web search: 
web search: "Lifelong person re-identification" "PTKP" "LSTKC"
codex
接下来我补证据部分，重点不是抄表格数字，而是找它们用什么消融或可视化把“问题重定义”闭合起来。这个信息通常在实验小节末尾。
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
/bin/zsh -lc "pdftotext -f 1 -l 12 'Diverse Representations Embedding for Lifelong Person Re-Identification.pdf' - | rg -n -i 'ablation|visual|t-sne|effectiveness|analysis|module|ACM|KU|KP|maximum|class token|Fig\\.|Table'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
24:adapting to new information based on instance-level and tasklevel layout. Concretely, an Adaptive Constraint Module (ACM)
29:propose Knowledge Update (KU) and Knowledge Preservation
30:(KP) strategies at the task-level layout by introducing the adjustment model and the learner model. KU strategy enhances the
32:under the adjustment model prior, and KP strategy preserves old
45:Fig. 1. An illustration of our DRE. Each shape denotes an instance, and
52:based on instance-level, we introduce Knowledge Update (KU) and Knowledge Preservation (KP) strategies to improve model performance in preserving
141:LReID model. We introduce Maximum Embedding (ME)
144:interference. Meanwhile, the class token of transformer is
146:the ReID task. We introduce multiple class tokens embedding
148:in the LReID task. Multiple class tokens and patches are
157:generated by multiple class tokens of transformer, as
164:an Adaptive Constraint Module (ACM) to implement
176:on instance-level, we propose Knowledge Update (KU) and
177:Knowledge Preservation (KP) strategies at the task-level
179:model. KU strategy enhances the adaptive learning ability of
189:for LReID tasks. Maximum embedding Multiple class
192:• We design an adaptive constraint module to maintain rich
249:Representation learning methods employ specific modules
254:embeddings subspace of each instance. DC-Former [29] employs multiple class tokens in vision transformer to generate
260:transformation module to complement the limited exemplars.
262:which insufficiently exploits rich and discriminative representations of each instance. We consider that diverse representations generated by multiple class tokens with transformerspecific structure play an important role in maintaining a
274:by Adaptive Constraint Module (ACM) to maintain richness
275:and discrimination of each instance. Knowledge Update (KU)
276:and Knowledge Preservation (KP) are proposed at the tasklevel layout by introducing the adjustment model and the
314:other in sequential task datasets. Multiple class tokens of
321:Fig. 2. An overview of the proposed DRE for LReID. We first explore a pure transformer to LReID and equip it with an adjustment model and a learner model.
322:The adaptive constraint module learns rich and discriminative representations. We then introduce knowledge update and knowledge preservation strategies
326:transformer structure, we introduce multiple class tokens and
327:maximum embedding. For a given image x, we split x into N
337:Maximum Embedding: We further design maximum embedding (ME) to focus on critical regions of the input sequence
340:sequence Z0 to query the index θ with the maximum value.
349:to obtain the maximum embedding ME , and the maximum
357:Multiple Class Tokens: We introduce multiple class tokens
358:embedded in transformer input sequence. Multiple class tokens
378:C. Adaptive constraint module
383:tasks. Therefore, we propose an adaptive learning module to
431:table represents the source of the model (a from adaptation
436:In this section, we propose a Knowledge Update (KU) strategy that enhances the learner model’s adaptive learning ability
482:on mining consistency information from new samples. In particular, we employ a kullback-leibler divergence to minimize
498:Where KL(.||.) refers to a kullback-leibler divergence, and τ
523:knowledge preservation (KP) strategy preserve old knowledge
545:TABLE I
682:statistics for these datasets are provided in Table I.
699:TABLE II
781:PTKP[17]
859:CODA+ACM
897:ResKUP
898:DRE-KU
899:FDRE-KUP
962:TABLE III
1044:PTKP[17]
1122:CODA+ACM
1160:ResKUP
1161:DRE-KU
1162:FDRE-KUP
1230:learning methods. SPD[61], LwF[62], and CRL[63] are lifelong learning methods in natural settings, and LReID methods include AKA[14], PTKP[17], PatchKD[18], KRKC[15],
1233:method as the comparison method. The CODA+ACM method
1234:introduces our Adaptive Constraint Module (ACM) based on
1236:are ResKUP, DRE-KU and FDRE-KUP. Compared to DRE,
1237:ResKUP uses ResNet50 as the feature extractor and outputs
1238:only one representation; DRE-KU does not incorporate a
1240:knowledge preservation strategy; FDRE-KUP freezes adjustment model parameters. Table II and Table III show the results
1242:training order-1 and training order-2, respectively. Table III
1245:Performance on Seen Datasets: From Table II, it is evident
1250:and 6.1% Rank-1. From Table III, the average improvement
1259:TABLE IV
1284:PTKP[17]
1359:CODA+ACM
1403:ResKUP
1404:DRE-KU
1405:DRE-KUP
1479:PTKP[17]
1554:CODA+ACM
1598:ResKUP
1599:DRE-KU
1600:DRE-KUP
1675:Fig. 3. Generalization ability on unseen datasets during the training process.
1677:introduce transformer-based CODA method. From Table IIIV, CODA method limits the performance of the model while
1678:preserving old knowledge. CODA+ACM method outperforms
1681:datasets (MSMT17 V2). This indicates that our Adaptive Constraint Module (ACM) can effectively improve the adaptive
1683:To further validate the effectiveness of our method, we
1684:set up some variants, including ResKUP, DRE-KU, DREKUP. From Table II and Table III, ResKUP achieves poor
1686:branch to generate a representation of each instance, our Adaptive Constraint Model (ACM), Knowledge Update (KU) and
1687:Knowledge Preservation (KP) strategies, are not fully utilized
1688:in ResKUP. DRE-KU is superior to non-rehearsal methods
1690:(AKA, PatchKD, ConRFL), as it benefits from our KU and
1691:KP strategies based on diverse representations. DRE-KUP can
1697:Generalization Ability on Unseen Datasets: In Table IV, our
1698:DRE achieves a more stable result over six unseen datasets for
1703:Occ REID). We argue that ACM generates rich and discriminative representations to guide a dynamic balance between
1709:TABLE V
1747:TABLE VI
1749:REPRESENTATION FOR ACM IN TRAINING ORDER -1.
1785:training process. DRE-KU without using KP strategy, limiting
1789:Effectiveness of Transformer as a Feature Extrator: We
1792:ResNet50 as the feature extractor. Therefore, ResKUP uses
1793:only an embedding representation. Compared to ResKUP,
1796:Seen-Avg, as shown in Table III. We attribute these improvements to ACM, KU and KP strategies, which generate rich and
1798:the adjustment model and the learner model. To further validate the effectiveness of the method, we compare the effects
1799:of different transformer backbones in the table below, such as
1800:DeiT-Base, DeiT-Small, and ViT-Small, as shown in Table V.
1804:table shows that our method achieves effective performance
1809:C. Ablation Studies
1810:Effectiveness on the number of auxiliary embedding representations for ACM: The proposed ACM adaptively explores
1813:ACM. As shown in Table VI, when the number of auxiliary
1817:TABLE VII
1885:ACM incorporates a primary embedding representations and
1892:ablation studies on seen and unseen datasets. As shown in
1893:Table VII, we can observe that the performance is unsatisfactory with only LLD, indicating that the performance of
1897:effectively achieving a trade-off between retaining old knowledge and adapting to new information. By integrating KU,
1898:KP, and ACM into an end-to-end LReID model, our proposed
1903:First, the adaptive learning module facilitates diverse representations to maintain rich and discriminative body information
1937:pattern analysis and machine intelligence, 44(12):8975–
1986:distillation for lifelong person re-identification. In Proceedings of the 30th ACM International Conference on
2051:[34] Zhedong Zheng, Liang Zheng, and Yi Yang. A discriminatively learned cnn embedding for person reidentification. ACM transactions on multimedia computing,
2098:Analysis and Machine Intelligence, 2023.
2102:the 30th ACM International Conference on Multimedia,
2140:Time-delayed correlation analysis for multi-camera activity understanding. International Journal of Computer
2154:discriminative classification. In Image Analysis: 17th
2162:[62] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine

exec
/bin/zsh -lc "pdftotext -f 1 -l 12 'Distribution aligned semantics adaption for lifelong person re-identification.pdf' - | rg -n -i 'ablation|visual|t-sne|effectiveness|analysis|module|BN|SA|pre-train|storage|Fig\\.|Table'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
15:adaptable to changes in space and time. Therefore, the adaptation of Re-ID models to new domains while preserving previously acquired knowledge is crucial,
19:data is inappropriate. Additionally, the fine-grained and open-set characteristics of Re-ID limit the effectiveness of the distillation paradigm for accumulating
22:lifelong applications. In this paper, we identify the challenges and discrepancies associated with adapting a pre-trained model to each application domain
23:and introduce the Distribution Aligned Semantics Adaption (DASA) framework.
24:It efficiently adjusts Batch Normalization (BN) to mitigate interference from
25:data distribution discrepancy and freezes the pre-trained convolutional layers to
27:Adaption (SA) module, which effectively adapts learned semantics to enhance
30:exhibits significantly reduced storage consumption. DASA presents a novel and
31:cost-effective perspective on effectively adapting pre-trained models for LReID.
32:The code is available at https://github.com/QizaoWang/DASA-LReID.
36:Samples
44:same pre-trained weights
55:Samples
68:(a) Results of using pre-trained weights
70:Fig. 1: (a) Comparison of average incremental performance for different methods
71:using the same pre-trained weights. (b) Conventional knowledge distillation pipeline
72:with exemplars. Our proposed DASA paradigm shows great superiority in LReID.
78:Person Re-IDentification (Re-ID) aims at recognizing the same pedestrian across disjoint cameras. With significant advancements in deep learning algorithms over the
88:Recently, we have witnessed the remarkable success of pre-training in computer
90:[10, 11] adopt unsupervised pre-training to prevailing Convolutional Neural Network
91:(CNN) architectures like ResNet [12] and achieve significant performance improvement. It is promising to leverage pre-trained models that can be obtained effortlessly
92:for real-world applications. Nevertheless, as observed in Fig. 1 (a), directly using the
93:pre-trained weights for model initialization yields limited benefits in the lifelong evolution scenario. Advanced LReID methods [13–16] follow the prevailing knowledge
99:As depicted in Fig. 1 (b), existing LReID methods follow the distillation baseline
108:great resources when saving it for knowledge distillation.
110:minimal source consumption, and without using exemplars, we introduce the Distribution Aligned Semantics Adaption (DASA) framework, which presents a novel
111:LReID paradigm based on pre-trained models. In this paradigm, we do not rely on
112:knowledge distillation that uses saved old data and models for knowledge retention.
113:Instead, we exert the advantages of the pre-trained model in the lifelong learning process. Specifically, we carefully handle and leverage the two crucial techniques in CNN,
114:i.e., Batch Normalization (BN) and Convolutional (Conv) blocks. Firstly, BN effectively regularizes the model and captures data distributions, contributing to improved
115:learning [26, 27]. In our proposed paradigm, we tune BN effectively to deal with
116:domain distribution differences between pre-training and application. Secondly, Conv
120:we propose freezing the pre-trained Conv layers to preserve shared person semantic
123:application domain effectively, we introduce the lightweight Semantics Adaption (SA)
124:module, which can efficiently aggregate, refine, and generate improved pedestrian representations. Adapting the general human semantics to more representative identity
128:knowledge acquired through large-scale pre-training to effectively adapt models in
130:(2) We propose the Distribution Aligned Semantics Adaption (DASA) framework
132:pre-training and application, and incorporates the lightweight semantics adaption
133:module to aggregate and refine learned semantics for better pedestrian representations.
142:stable performance on old tasks while adapting the model to gain new knowledge.
143:Methods can be traditionally divided into three categories, namely regularizationbased, rehearsal-based, and architecture-based methods. Regularization-based methods [17, 18] limit updating important parameters for preceding tasks to mitigate
144:forgetting. To better keep past acquired knowledge, rehearsal-based methods [21, 23,
146:prevalent for their satisfactory performance and simplicity. However, rehearsal-based
153:or recognition tasks, the presence of imbalanced samples per identity and subtle interclass variations in person Re-ID poses additional challenges. The significant variations
163:methods have drawn inspiration from rehearsal-based lifelong learning approaches.
166:preservation framework to alleviate the domain gap in the last BN layer. KRC [15]
170:leveraging useful knowledge for LReID. Moreover, privacy concerns limit their applicability in real-world scenarios, where exemplars cannot be stored for lifelong usage.
180:(DG) [39, 40]. Specifically, in TTA, the pre-trained model is optimized during inference
181:[37]. For instance, BNTA [38] adapts the model using gallery data and self-supervised
214:number of samples in the t-th training set. At the t-th training step, the model G (t)
226:different datasets obliges previous LReID methods [14–16] to save a small number of
262:SA Module
264:BN
278:SAl
282:BN l
295:SA Module
297:BN
310:SA
312:BN
317:Fig. 2: The framework of DASA. The acquired robust and general human semantics from pre-training are used as shared knowledge, which is kept in the frozen Conv
318:layers. At each training step, we adapt the acquired knowledge from the pre-training
319:to application domain by tuning BN layers and adopting the lightweight Semantics
320:Adaption (SA) modules. During the lifelong evolution process, the previously learned
321:BN and SA are used for initialization in the upcoming domain for forward knowledge
322:transfer, while the old classifier can be discarded without increasing storage burden.
324:training step. New samples can also be used for distillation [13, 17]. Previous LReID
330:With the prevalent trend of pre-training in Re-ID, we advocate taking advantage of
331:pre-trained models for effective lifelong evolution across various domains. In the new
333:from pre-training as shared knowledge. At each training step, our goal is transformed
334:to adapt the acquired semantic knowledge from the pre-training domain to the target application domain. Two problems need to be addressed, that is, the domain
335:distribution differences between pre-training and application, and how to effectively
338:Distribution alignment. Batch Normalization (BN) as a widely-used technique in
339:CNN can normalize the features of each domain to the same reference distribution
341:works, we maintain multiple sets of domain-specific BN layers to estimate the feature
342:distribution statistics. We tune BN in each application domain so that the outputs
345:of the updated BN layers exhibit a similar distribution to the pre-training domain,
346:and the remaining frozen pre-trained Conv layers can receive stable input activations.
350:bution N (µ(t) , σ (t) ). At training time, the BN at each layer l collects domain-specific
363:calculation of BN is formulated as follows:
370:BN(fl ) = γl ql
381:oLBN
385:, where LBN is the number of BN layers in the
386:i.e., BN (t) = µl , σ l , γl , βl
389:the next, we use the BN (t−1) of the previous domain as the initial for training BN (t)
391:Semantics adaption. Having compensated for the distribution differences between
392:the pre-training domain and the application domain, it is time to consider how to
394:The learned human semantics from pre-training are regarded as robust and general
395:knowledge for Re-ID, so we freeze the pre-trained Conv layers to keep the shared
399:is necessary to adapt the learned general semantics to specific application domains.
400:To this end, we introduce the lightweight Semantics Adaption (SA) module after
402:sake of efficiency, it could be a good choice to implement SA as one depth-wise Conv
406:the discriminative ability of the model (as shown in Fig. 5). It makes sense since there
411:refine the general semantics. Note that since the SA module performs convolutional
417:5 × 5 can achieve a good balance between effectiveness and efficiency.
418:There are some differences when comparing the SA module with LoRA [43], which
420:matrices. Technically, LoRA updates the original weights and changes the calculation operator, adapting the model for different tasks. Differently, the SA module
425:After training in each domain, we can obtain a set of domain-specific SA modules,
430:i.e., SA(t) = wl
431:, where wl denotes the parameters of the SA module at
435:learned domain to the next, we use the SA(t−1) as the initial to train SA(t) effectively.
438:The framework is shown in Fig. 2. At each training step, we use the basic identity
440:domain-specific BN (t) and SA(t) as introduced above. Both of them are lightweight
443:other competitors as shown in Fig. 4. During testing, according to the camera information of the person image, corresponding BN (t) and SA(t) are used. With the help
444:of BN (t) and SA(t) , the model can efficiently evolve and adapt to new application
445:domains, making it effective and practical for new deployment and wide usage in
450:Datasets. We evaluate our proposed method following the widely-used LReID setting [14–16]. Specifically, we investigate its effectiveness in two different training
460:Table 1: Comparison with the state-of-the-art methods in the LReID setting of Order 1. “w/ Ex.” denotes rehearsal-based methods using exemplars. The
656:DASA
660:We also evaluate the influence of using different datasets for pre-training, including ImageNet [25] for image classification, LUPerson [10] and LUPerson-NL [11] for
665:use the LUPerson-NL pre-trained weights for our method, which can provide robust
676:set to 128, with 2 samples per pedestrian. Adam optimizer [55] with weight decay of
680:datasets. Each dataset is trained for 80 epochs. For the SA module, the kernel size is
681:set to 5 × 5 with stride 1, and the bias term is removed for the sake of the number of
694:Fig. 3: Comparison of average accuracies at different training steps of
696:and 2, respectively. Comparison methods for LReID are reproduced in the same experimental environment according to their released code and training settings. Fine-Tune
699:domains, although rehearsal-based methods show great performance in the classification task [21, 22], it contributes to inferior performance gain when being applied to
705:relying on impractical exemplars, DASA still outperforms them significantly.
712:are essential for LReID. As shown in Fig. 3, DASA achieves better average accuracies during the lifelong evolution process on the four datasets. The results show its
713:effectiveness in adapting to the application domains and acquiring knowledge.
714:As shown in Tab. 2, our proposed DASA also achieves state-of-the-art results when
716:images for training, DASA still shows great superiority. By comparing the results of
727:Table 2: Comparison with the state-of-the-art methods in the LReID setting of Order 2. “w/ Ex.” denotes rehearsal-based methods using exemplars. The
911:DASA
915:Table 3: Storage consumption of different components. Results are calculated after the last training step.
918:Fig. 4: Comparison of storage consumption for LReID methods. Results
921:Storage (MB)
933:BN (t)
934:SA(t)
943:Order 2. In contrast, DASA is robust to different training orders and datasets.
944:Comparison of storage consumption. Storage consumption is also crucial in lifelong scenarios, otherwise, we can save all previous models to prevent catastrophic
945:forgetting. We compare the storage consumption for different LReID methods after the
946:last training step (Order 1) in Fig. 4 and Tab. 3. All competitors [14–16] have to store
951:training step as shown in Tab. 3. What’s worse, an old model having the same large
952:classifier is used for knowledge distillation, further exacerbating storage consumption.
958:Table 4: Comparison with state-ofthe-art methods on unseen domains
963:Table 5: Comparison with state-ofthe-art methods on unseen domains
1006:DASA
1056:DASA
1064:privacy issues, saving exemplars is unrealistic. On the other hand, due to the open
1065:categories of pedestrians, saving a large number of identity images can consume storage
1066:a lot. The storage consumption of exemplars even surpasses that of both the backbone
1068:exemplar pool, but when comparing it in Fig. 4, we still assume it saves 250 identities
1069:per step, otherwise its storage cost would be enormous (about 1578MB for exemplars).
1073:lifelong evolution of the model. As shown in Tab. 3, the introduced storage consumption at each training step is only equal to 3% of the backbone model (v.s., 70% on
1075:of our proposed DASA in terms of efficiency and effectiveness in LReID.
1076:Results on unseen domains. Our proposed DASA shows great anti-forgetting
1080:domains. As shown in Tabs. 4 and 5, the generalization ability of DASA outperforms that of all other advanced methods as well as Fine-Tune and Joint-Train on all
1082:semantic knowledge is useful for generalization and DASA shows great effectiveness
1085:4.4 Ablation Studies
1086:For a fairer comparison and to validate our designs, we conduct all ablation experiments using the LUPerson-NL [11] pre-trained weights in Tab. 6. Without loss of
1087:generality, ablation experiments are conducted in Order 1 by default. Besides state-ofthe-art methods, we also compare (1) Fine-Tune: fine-tuning the pre-trained model on

exec
/bin/zsh -lc "pdftotext -f 1 -l 12 'Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf' - | rg -n -i 'ablation|visual|t-sne|effectiveness|analysis|module|MFD|PFD|memory|posture|Figure|Fig\\.|Table'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5:Disentangling Modality and Posture Factors:
6:Memory-Attention and Orthogonal Decomposition
13:images, the extracted representations inevitably comprise various modality and posture factors, impacting the matching of
14:cross-modality person identity. To solve the problem, we propose a disentangling modality and posture factors (DMPFs)
15:model to disentangle modality and posture factors by fusing
16:the information of features memory and pedestrian skeleton.
17:Specifically, the DMPF comprises three modules: three-stream
18:features extraction network (TFENet), modality factor disentanglement (MFD), and posture factor disentanglement (PFD). First,
19:aiming to provide memory and skeleton information for modality
20:and posture factors disentanglement, the TFENet is designed
23:across different batches, we maintain memory queues of previous
25:propose MFD to integrate features in the whole training set by
26:memory-attention layers. These layers explore intramodality and
28:batch and memory queues under the optimization of the optimal
31:to decouple the posture factors from representations, we introduce the PFD module to learn posture-unrelated features with the
34:to separate the posture-related and identity-related information.
35:The posture-related features are adopted to disentangle the posture factors from representations by a designed posture-features
37:The effectiveness of DMPF is validated through comprehensive
39:Index Terms— Modality-unrelated features learning, posture
62:Fig. 1. Illustration of motivation and methodology regarding the proposed
64:feature representations due to the modality and posture factors. (b) and (c) Proposed DMPF successfully disentangles the modality and posture factors from
65:the identity features by the proposed MFD and PFD modules, respectively.
67:the MFD is proposed to integrate features from the current batch and memory
68:queues by adopting the designed memory-attention layers. To disentangle the
69:postures factor, the PFD is designed to use subspace orthogonal decomposition
70:between posture-related and identity-related information in the image and
93:disparities exhibited by heterogeneous images [13]. As illustrated in Fig. 1(a), the intraclass differences primarily stem
94:from two factors: modality discrepancy and posture variation.
100:LU et al.: DMPFs: MEMORY-ATTENTION AND ORTHOGONAL DECOMPOSITION
113:disparity in postures is notable for both NIR and VIS
114:images. Therefore, the modality and posture discrepancies
153:[13], [22], [31]. These feature learning methods have demonstrated their effectiveness in mitigating modality gaps in the
160:posture discrepancy.
161:To eliminate the posture discrepancy, several approaches
165:impact posture factor decoupling and undermine the discriminability of the identity features. Therefore, it is necessary to
178:memory demands during model training. Moreover, during
187:modality and posture factors (DMPFs) model to disentangle
188:modality and posture factors by fusing the information
189:of features memory and pedestrian skeleton. The former
196:modules: three-stream features extraction network (TFENet),
197:modality factor disentanglement (MFD), and posture factor
198:disentanglement (PFD). First, to provide memory and skeleton
199:information for modality and posture factors disentanglement,
202:as illustrated in Fig. 1(b), to eliminate modality discrepancy
203:across the training set, we propose the MFD to integrate batch
204:and memory features using designed memory-attention layers.
210:from identity features by the memory-attention layers.
219:elimination from the memory-attention layers to the backbone
223:encourages the backbone network to learn from the memoryattention layers, thereby extracting modality-independent
225:Fig. 1(c), considering the complementary between landmark
228:entanglement between posture-related and identity-related
230:the PFD module to extract posture-related and identity-related
232:The posture-related features are extracted to effectively
233:separate the posture factors by the proposed posture-features
238:1) Aiming at learning identity-invariant feature representations, the DMPF model integrates TFENet, MFD,
239:and PFD modules into a unified framework to
240:effectively disentangle modality and posture factors
243:posture discrepancies between VIS and NIR data, which
246:across the entire training set, we propose the MFD
247:module to integrate features from the current batch
248:and memory queues by adopting the designed memoryattention layers. Different from the previous method,
250:intermodality relationships between batch and memory
254:from identity features by the memory-attention layers.
255:3) To disentangle the postures factor from identity representation, we propose the PFD module to handle
256:the entanglement between posture-related and identityrelated information in the image and skeleton features
258:from existing methods, the PFD module decomposes
264:overview and analysis of related works. In Section III, the
314:nonlinearity of modality factors in raw images cause notable
331:LU et al.: DMPFs: MEMORY-ATTENTION AND ORTHOGONAL DECOMPOSITION
369:a modality alleviation module and a pattern alignment module
371:effectiveness in minimizing modality differences, these local
382:to modality differences, obvious variations exist in posture,
404:methods have shown effectiveness in mitigating modality
412:overlook the posture discrepancy between heterogeneous samples, leading to that the extracted features are sensitive to
413:variations in pedestrian posture.
416:detail, and the diagram of our method is illustrated in Fig. 2.
417:The proposed method consists of three modules, including
418:TFENet, MFD, and PFD. The detailed introduction is shown
437:the inclusion of modality and posture factors in E V and E N
440:two identity-unrelated factors by fusing features memory and
445:posture variation between heterogeneous images, the identity
447:posture factors, which may impact VIS-NIR identity matching.
455:Fig. 2. Diagram of our DMPF, which contains TFENet, MFD, and PFD modules. First, the TFENet is designed as a three-stream network to extract VIS-NIR
456:images and skeleton features. Second, the MFD is proposed to integrate features in the whole training set by memory-attention layers, which encourages the
457:heterogeneous features with the same identity to present higher similarity. Third, the PFD is introduced to learn posture-unrelated features with the assistance
460:To solve this problem, we fuse intraclass features memory and pedestrians’ skeleton information to decouple these
462:we introduce feature extraction and intraclass features memory
464:1) Image and Skeleton Features Extraction: Considering the variation in pedestrian postures across images, the
465:extracted identity features inevitably encompass posture factors, which impact the subsequent identity-matching process.
496:identity-related and posture-related information of image and
498:network. The skeleton features E S are served for posture
499:factors disentanglement in the PFD module.
500:2) Intraclass Features Memory Construction: To mitigate
507:To tackle this challenge, we introduce memory queues to store
509:features memory, we can more effectively guide the model to
512:we present how to initialize and update the memory.
513:a) Memory initializing: We employ the TFENet to
516:memory requires excessive computation costs, we initialize
517:the memory queue with class-level center features, which are
526:denotes M-modality intraclass features memory matrix, (E cM )i
529:b) Memory updating: As training proceeds, the weight
534:LU et al.: DMPFs: MEMORY-ATTENTION AND ORTHOGONAL DECOMPOSITION
538:a) Memory self-attention layer: First, to explore the
539:intramodality associations between batch and memory features, we propose the MSL to process the two modality
540:features separately. Specifically, the batch features are concatenated with memory features and fed into the MSL for
552:Fig. 3. Diagram of memory-attention layers, consisting of an MSL and
554:intramodality and cross-modality relationships between batch and memory
558:features [54]. To keep both consistent, we update the memory
566:corresponding identity label and η denotes the updated hyperparameter. The memory M M stores the intraclass features of
568:memory M M is employed for modality factor decoupling in
569:the MFD module.
572:across the entire training set, we propose MFD to integrate batch features and memory features using designed
573:memory-attention layers. By optimizing the layers with the
575:and intermodality relationships between batch and memory
579:features by the memory-attention layers, thereby extracting augmented modality-independent features. Furthermore,
581:from the memory-attention layers to the backbone network,
587:1) Memory-Attention Layers: Aiming to eliminate modality differences, we design the memory-attention layers to
588:explore the associations between batch features and memory
591:we adopt progressive mining strategies. As shown in Fig. 3,
592:the memory-attention layers are composed of two sublayers: the memory self-attention layer (MSL) and the memory
595:batch and memory features, respectively.
597:where E M and M M denote M-modality batch and memory
600:multilayer perceptron, and MSL(·) denotes memory selfattention operation, which is formulated as follows:
635:b) Memory cross-attention layer:
638:and memory features, we propose the MCL to interactively
669:ing of augmented memory M
679:designed MiD loss. MCL(·) denotes memory cross-attention
715:by merging the memory matrices from both modalities during
859:memory-attention layers, the backbone network can still effectively address the modality discrepancy in heterogeneous data.
860:D. Posture Factor Disentanglement
861:Due to the diverse postures of pedestrians in different
863:the posture factor, which in turn affects subsequent identity
867:decouple the posture factor from the identity features [33].
869:pose-related and identity-related information. The entanglement between the two types of information may impact posture
872:subspace orthogonal decomposition [36] on the skeleton features to extract posture-related and identity-related features
873:of the skeleton. The posture-related features are employed
874:to disentangle the posture factor by the proposed PfC loss,
878:1) Posture Orthogonal Decomposition: The above analysis indicates that the skeleton features E S encompass
879:both posture-related and identity-related information, which
880:impacts the disentanglement of the posture factor. To separate
882:matrices, i.e., K po and K id , for modeling posture-related and
895:and E idS denote posture-related and identityrelated features of skeleton, respectively. K po and K id denote
899:LU et al.: DMPFs: MEMORY-ATTENTION AND ORTHOGONAL DECOMPOSITION
903:posture-related and identity-related mapping matrices, respectively. Considering the subspace decomposition properties of
972:denote posture-related and identityrelated features of image, respectively. Ppo and Pid denote
973:posture-related and identity-related mapping matrices, respectively. Similar to (13) and (14), we introduce the orthogonal
980:posture-related and identity-related information into E po
984:Representing the same posture of the same pedestrian, the
987:posture-related features E po
1035:to capture posture-related information
1038:from E po . When E po effectively represents posture-related
1039:information, the posture factor is disentangled from image
1058:information, i.e., posture information, due to the orthogonal constraint. Subsequently, we employ the posture-related
1061:to decouple the posture factor from the image
1065:posture-related and identity-related information as well.
1093:MFD Loss
1109:PFD Loss
1116:respectively. The overall loss consists of three terms, i.e., ReID, MFD, and PFD losses. First, Re-ID loss aims to encourage
1118:posture discrepancies between heterogeneous images, the identity features inevitably contain modality and posture factors,
1120:this issue, we introduce MFD and PFD losses to decouple the
1150:TABLE I
1153:TABLE II
1155:Fig. 4. Example images from (a) SYSU-MM01 dataset and (b) RegDB
1165:Fig. 4(a) displays a subset of NIR and VIS images sourced
1173:from the other modality (typically VIS). To provide a visual
1174:representation, Fig. 4(b) showcases a subset of NIR and VIS
1182:details of the three modules are described as follows.
1194:0.1 within the first ten epochs. To promote stable convergence,
1200:2) MFD and PFD: The MSL and MCL layers in the
1201:MFD module are implemented using a multihead self-attention
1209:PFD module, we adopt the SCHP [50] and OpenPose [51]
1215:C. Ablation Study
1216:We evaluate the effectiveness of individual modules within
1217:our proposed DMPF framework, namely, MFD and PFD.
1218:The ablation results for these modules on the SYSU-MM01
1219:and RegDB datasets are documented in Tables I and II,
1220:respectively. Moreover, to visually demonstrate the impact of
1221:each module, we present the CMC curve for both datasets in
1222:Fig. 5(a) and (b), respectively.
1223:1) Effectiveness of MFD: To address the modality discrepancy across the entire training set, we propose MFD to
1224:integrate batch and memory features by adopting the designed
1225:memory-attention layers. As demonstrated in Tables I and II,
1226:MFD markedly contributes to the enhancement on the SYSUMM01 and RegDB datasets. For instance, MFD achieves
1229:a notable 5.05% (from 66.55% to 71.60%) and 5.83% (from
1232:as depicted in Fig. 5, a comparison between the performance
1233:curves of the model with and without MFD clearly demonstrates the crucial role of MFD in enhancing the model’s
1235:memory-attention layers with the OT method, we effectively
1241:Furthermore, the MFD consists of two main components:
1242:intraclass features memory and memory-attention layers. The
1245:associations between batch and memory features. To evaluate the effectiveness of these components, we conducted
1249:LU et al.: DMPFs: MEMORY-ATTENTION AND ORTHOGONAL DECOMPOSITION
1253:Fig. 5. Illustration of CMC curves. (a) and (b) CMC curves of comparison
1255:TABLE III
1256:C OMPARISON OF M EMORY-ATTENTION M ETHODS IN MFD ON T HE
1259:Fig. 6.
1262:TABLE IV
1264:C OMPARISON OF PARSING AND L ANDMARK I NFORMATION IN PFD ON
1267:within-module ablation experiments. Specifically, we utilized
1268:the memory-attention layers to capture the associations among
1269:batch features without incorporating feature memory information, referred to as batch-attention. As shown in Table III,
1273:on the SYSU-MM01 dataset, indicating the effectiveness of
1275:fusing features memory information, referred to as memoryattention, leads to an additional 1.79% (from 64.09% to
1278:These results indicate that incorporating features memory
1281:To demonstrate the efficacy of the memory attention layers,
1283:directly from the input M M of the memory attention layers
1284:fM of the memory attention layers.
1286:As depicted in Fig. 6, the OT loss computed directly from
1289:20 000 iterations. This observation suggests that the memory
1292:The memory attention layers aim at exploring associations
1293:among heterogeneous memory matrices, effectively mitigating
1295:2) Effectiveness of PFD: Apart from the discrepancy in
1296:modality, variations in pedestrian posture also have a nonnegligible impact on the discriminability of feature representations.
1297:To handle this challenge, we introduce the PFD module,
1298:which integrates skeleton information to effectively disentangle the posture factor from the identity features. The fusion
1299:of skeleton information mitigates the influence of posture
1301:features. As presented in Tables I and II, experimental results
1302:validate the substantial contribution of the PFD module to the
1304:datasets. Notably, the PFD module achieves remarkable gains
1306:to 78.01%) in mAP, as well as notable improvements of
1309:TABLE V
1314:datasets, respectively. Moreover, Fig. 5 clearly illustrates the
1316:PFD, emphasizing the pivotal role played by PFD in enhancing
1319:The PFD module incorporates two types of information:
1324:the PFD module. As shown in Table IV, compared to the baseline method, fusing the shape profile information, i.e., parsing
1325:leads to a notable improvement of 4.67% (from 62.18% to
1328:that parsing alone may not capture the complete posture of
1330:the PFD module. Compared to parsing alone, fusing parsing
1335:more complete skeleton information to the PFD module, which
1336:is beneficial for effectively decoupling the posture factors.
1338:posture-related features to calculate the L PfC loss, it inevitably
1339:affects the decoupling of the posture factor from the identity
1342:orthogonal decomposition to obtain posture features E po
1344:computing the L PfC loss. As demonstrated in Table V, compared to using the impure skeleton features E S , the loss
1352:Fig. 7. Visualization for feature distribution by randomly sampling 14 identities of the SYSU-MM01. Each unique identity is represented by a distinct
1354:our DMPF in reducing modality variance, while the blue dotted circles indicate its success in mitigating posture variance. (a) Baseline method. (b) Proposed
1356:TABLE VI
1358:AND L O IN PFD L OSS ON T HE SYSU-MM01 DATASET
1361:computed using the posture features E po
1365:the effectiveness of subspace orthogonal decomposition in
1366:extracting posture-related features.
1367:The PFD loss comprises three subterms, i.e., L ID , L PfC and
1369:performance, we conduct ablation experiments involving the
1370:PFD loss. As illustrated in Table VI, when compared to the
1374:to 76.41%). This underscores the effectiveness of the disentangling losses. To further assess the individual contributions
1384:3) Qualitative Analysis: To analyze the effectiveness of
1385:DMPF in eliminating modality and posture discrepancies, we
1386:utilize t-SNE [64] to visualize the feature distributions of
1387:the baseline method and DMPF. As depicted in Fig. 7(a),
1391:intraclass clusters due to posture variations, as represented
1394:Fig. 8. Illustration of Re-ID results between baseline and DMPF. The green
1397:identity from the query. (a) and (d) Demonstrate the effectiveness of the
1399:the effectiveness of the proposed method in disentangling the posture factor.
1401:the baseline method is affected by both modality and posture differences. In contrast, as shown in Fig. 7(b), DMPF
1406:various postures together, as demonstrated by the blue dotted
1407:circles. The comparison between Fig. 7(a) and (b) highlights
1409:modality and posture discrepancies.
1410:To provide a more intuitive understanding of the performance of DMPF, we visually present the results of Re-ID
1411:in Fig. 8, which showcases ranking result examples of the
1418:As illustrated in Fig. 8(a) and (b), even though the pedestrian is wearing dark colors shirt such as black and red, the
1424:LU et al.: DMPFs: MEMORY-ATTENTION AND ORTHOGONAL DECOMPOSITION
1428:TABLE VII
1436:related to posture discrepancies. As depicted in Fig. 8(c), the
1437:NIR query image portrays a pedestrian in a straddle posture,
1439:samples performing the same posture. Similarly, in Fig. 8(d),
1445:postures, thus affirming the robustness of the learned identity
1446:features to posture variations.
1449:DMPF with state-of-the-art methods to showcase its effectiveness in addressing the VIS-NIR person Re-ID task. The
1453:Table VII.
1456:results are presented in Table VII. Notably, our DMPF
1467:evidence of the effectiveness of DMPF and its potential for
1471:dataset. As shown in Fig. 4, the potential reason may be the
1474:and posture information disentanglement.
1482:CAJL [17], DMPF achieves notable improvements in

codex
DMPF 的证据很完整，包含模块消融、内部分解、t-SNE 和检索例子。DASA 和 DRE 的证据更多靠“同预训练权重/存储成本/变体对照”证明视角成立，我现在把这些段落抽出来。
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
/bin/zsh -lc "pdftotext -f 1 -l 13 'Diverse Representations Embedding for Lifelong Person Re-Identification.pdf' - | sed -n '1675,1910p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Fig. 3. Generalization ability on unseen datasets during the training process.

introduce transformer-based CODA method. From Table IIIV, CODA method limits the performance of the model while
preserving old knowledge. CODA+ACM method outperforms
CODA methods in mAP and Rank-1 metrics, especially in
occluded datasets (Occ Duke and Occ REID) and large scale
datasets (MSMT17 V2). This indicates that our Adaptive Constraint Module (ACM) can effectively improve the adaptive
capability of the model, benefiting from rich and discriminative representations of each instance.
To further validate the effectiveness of our method, we
set up some variants, including ResKUP, DRE-KU, DREKUP. From Table II and Table III, ResKUP achieves poor
mAP and Rank-1 performance. Because ResNet50 employs a
branch to generate a representation of each instance, our Adaptive Constraint Model (ACM), Knowledge Update (KU) and
Knowledge Preservation (KP) strategies, are not fully utilized
in ResKUP. DRE-KU is superior to non-rehearsal methods

(AKA, PatchKD, ConRFL), as it benefits from our KU and
KP strategies based on diverse representations. DRE-KUP can
effectively alleviate catastrophic forgetting problems, but its
adaptive capacity is limited compared to DRE, due to the
freezing of adaptive model parameters. In general, our DRE
significantly improves performance in large-scale datasets to
preserve old knowledge whlie adapting new information.
Generalization Ability on Unseen Datasets: In Table IV, our
DRE achieves a more stable result over six unseen datasets for
training order-1 and training order-2. This demonstrates that
our method allows smooth transfer and robust generalization
across various unseen datasets. Compared to other methods,
our DRE is more friendly to occlusion datasets (Occ Duke and
Occ REID). We argue that ACM generates rich and discriminative representations to guide a dynamic balance between
anti-forgetting and adapting to new tasks. Figure 3 illustrates
that our method achieves optimal performance in terms of

9

TABLE V
P ERFORMANCE OF DIFFERENT TRANSFORMER - BASED BACKBONE IN
TRAINING ORDER -1.
Backbone

Seen Avg
smAP
sR−1

Unseen Avg
smAP
sR−1

DeiT-Small
DeiT-Base
ViT-Small
ViT-Base

48.9
56.4
48.3
56.8

47.0
56.5
48.4
56.7

61.0
68.4
60.3
68.2

45.6
55.5
48.0
55.0

TABLE VI
A BLATION STUDIES ON THE NUMBER OF AUXILIARY EMBEDDING
REPRESENTATION FOR ACM IN TRAINING ORDER -1.
Primary
√
√
√
√

Auxiliary
Seen Avg
Unseen Avg
1 2 3 smAP sR−1 smAP sR−1
√
√ √
√ √ √

52.8
54.3
56.8
56.4

64.2
64.4
68.2
68.7

53.4
54.8
56.7
56.0

51.5
52.7
55.0
54.6

mAP and Rank-1 metrics for all unseen datasets during the
training process. DRE-KU without using KP strategy, limiting
the generalization capability of the LReID model. In contrast,
our DRE consistently enhances generalization capabilities over
time.
Effectiveness of Transformer as a Feature Extrator: We
find that introducing diverse representations performs worse
than using only one representation separately when employing
ResNet50 as the feature extractor. Therefore, ResKUP uses
only an embedding representation. Compared to ResKUP,
DRE achieves significant performance improvement with an
8.6% increase in mAP and an 8.2% increase in Rank-1 for
Seen-Avg, as shown in Table III. We attribute these improvements to ACM, KU and KP strategies, which generate rich and
discriminative representations to interact knowledge between
the adjustment model and the learner model. To further validate the effectiveness of the method, we compare the effects
of different transformer backbones in the table below, such as
DeiT-Base, DeiT-Small, and ViT-Small, as shown in Table V.
ViT-Base and DeiT-Base are very close to mAP and Rank-1
metrics. The number of ViT-Small and DeiT-Small parameters
decreased, resulting in lower mAP and Rank-1 indicators. The
table shows that our method achieves effective performance
on the transformer backbone. In summary, our proposed
transformer-based DRE significantly improves performance on
both old and new tasks over an extended period of time.

C. Ablation Studies
Effectiveness on the number of auxiliary embedding representations for ACM: The proposed ACM adaptively explores
rich and discriminative representations. Here, we study the
suitability of multiple auxiliary embedding representations for
ACM. As shown in Table VI, when the number of auxiliary
embedding representations increases from 0 to 2, it generates
rich and discriminative representations that generate multiple

TABLE VII
P ERFORMANCE OF INDIVIDUAL COMPONENTS FOR OUR DRE IN
TRAINING ORDER -1.
LLD

RLA

LLS

√
√
√

√

√
√
√
√
√

√
√

Seen Avg
smAP sR−1

Unseen Avg
smAP sR−1

49.4
52.3

61.8
63.9

50.4
53.8

48.9
51.2

56.2
55.8
54.8
55.3
56.8

67.6
66.1
65.7
65.3
68.2

56.2
55.4
53.2
55.8
56.7

54.5
53.3
52.5
54.1
55.0

instances for the same classes to facilitate model performance
improvement on both seen and unseen datasets. Therefore,
ACM incorporates a primary embedding representations and
two auxiliary embedding representations, achieving the best
performance while striking a trade-off between metrics and
complexity on both seen and unseen datasets. Therefore, the
number of auxiliary embedding representations S is set to 2.
Performance of Individual Component: To better evaluate
the contribution of each component to DRE, we conduct some
ablation studies on seen and unseen datasets. As shown in
Table VII, we can observe that the performance is unsatisfactory with only LLD, indicating that the performance of
the knowledge update strategy reaches a limit in mitigating
catastrophic forgetting. However, after introducing the RLA
and LLS, the model performance is significantly improved,
effectively achieving a trade-off between retaining old knowledge and adapting to new information. By integrating KU,
KP, and ACM into an end-to-end LReID model, our proposed
DRE achieves impressive performance improvements.
V. C ONCLUSIONS
In this paper, we propose a diverse representation embedding framework that first exploits a pure transformer backbone
to preserve old knowledge while adapting new information.
First, the adaptive learning module facilitates diverse representations to maintain rich and discriminative body information
for each instance. Then, we explore knowledge update and
knowledge preservation strategies that collectively achieve
knowledge interaction to mitigate catastrophic forgetting and
adapt to new tasks over an extended period of time, and
sufficiently explore the diverse representations of each instance based on transformer backbone. Extensive experiments
demonstrate the superiority of our method in comparison with
state-of-the-art LReID methods. Moreover, we conduct experiments to analyze the impact of the knowledge preservation

exec
/bin/zsh -lc "pdftotext -f 1 -l 13 'Distribution aligned semantics adaption for lifelong person re-identification.pdf' - | sed -n '1085,1210p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
4.4 Ablation Studies
For a fairer comparison and to validate our designs, we conduct all ablation experiments using the LUPerson-NL [11] pre-trained weights in Tab. 6. Without loss of
generality, ablation experiments are conducted in Order 1 by default. Besides state-ofthe-art methods, we also compare (1) Fine-Tune: fine-tuning the pre-trained model on

12

Table 6: Ablation Studies of our method. All methods use LUPerson-NL pretrained weights. The results are reported after the last training step in Order 1. “∗”
denotes adding the SA module to the original ResNet-50.
Market-1501

DukeMTMC

CUHK-SYSU

MSMT17

mAP

R-1

mAP

R-1

mAP

R-1

mAP

R-1

smAP

sR−1

AKA [13]
KRC [15]
PTKP [14]

66.7
68.4
83.6

83.7
85.8
93.3

38.3
64.8
69.1

53.3
76.6
81.3

84.5
91.9
89.4

86.1
93.5
90.4

19.1
49.2
41.1

37.4
72.2
65.6

52.2
68.6
70.8

65.1
82.0
82.7

Fine-Tune∗
Fine-Tune
DA
SA

25.9
30.0
80.4
80.1

50.6
56.3
91.6
92.1

31.1
38.7
70.9
72.4

49.4
55.6
83.5
84.2

65.9
76.4
93.0
91.8

68.9
79.2
94.1
93.0

36.9
44.9
39.8
40.1

62.0
69.2
64.5
65.2

40.0

exec
/bin/zsh -lc "pdftotext -f 1 -l 14 'Disentangling Modality and Posture Factors - Memory-Attention and Orthogonal Decomposition for Visible-Infrared Person Re-Identification.pdf' - | sed -n '1215,1412p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
C. Ablation Study
We evaluate the effectiveness of individual modules within
our proposed DMPF framework, namely, MFD and PFD.
The ablation results for these modules on the SYSU-MM01
and RegDB datasets are documented in Tables I and II,
respectively. Moreover, to visually demonstrate the impact of
each module, we present the CMC curve for both datasets in
Fig. 5(a) and (b), respectively.
1) Effectiveness of MFD: To address the modality discrepancy across the entire training set, we propose MFD to
integrate batch and memory features by adopting the designed
memory-attention layers. As demonstrated in Tables I and II,
MFD markedly contributes to the enhancement on the SYSUMM01 and RegDB datasets. For instance, MFD achieves
a remarkable 3.70% (from 62.18% to 65.88%) and 7.06%
(from 72.81% to 79.87%) improvement in mAP, as well as
a notable 5.05% (from 66.55% to 71.60%) and 5.83% (from
80.92% to 86.75%) improvement in Rank-1 accuracy on the
SYSU-MM01 and RegDB datasets, respectively. In addition,
as depicted in Fig. 5, a comparison between the performance
curves of the model with and without MFD clearly demonstrates the crucial role of MFD in enhancing the model’s
performance across the Rank-1–Rank-20 range. By optimizing
memory-attention layers with the OT method, we effectively
explore intramodality and intermodality relationships, allowing
augmented heterogeneous features with the same identity
to present high similarity. Hence, we successfully decouple
the modality factors from identity features, improving model
performance on VIS-NIR Re-ID.
Furthermore, the MFD consists of two main components:
intraclass features memory and memory-attention layers. The
former provides feature information of the training set, while
the latter focuses on mining intramodality and cross-modality
associations between batch and memory features. To evaluate the effectiveness of these components, we conducted

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:54 UTC from IEEE Xplore. Restrictions apply.

LU et al.: DMPFs: MEMORY-ATTENTION AND ORTHOGONAL DECOMPOSITION

5503

Fig. 5. Illustration of CMC curves. (a) and (b) CMC curves of comparison
experiments on SYSU-MM01 and RegDB, respectively.
TABLE III
C OMPARISON OF M EMORY-ATTENTION M ETHODS IN MFD ON T HE
SYSU-MM01 DATASET

Fig. 6.

Curves of OT loss during the training on the SYSU-MM01.
TABLE IV

C OMPARISON OF PARSING AND L ANDMARK I NFORMATION IN PFD ON
T HE SYSU-MM01 DATASET

within-module ablation experiments. Specifically, we utilized
the memory-attention layers to capture the associations among
batch features without incorporating feature memory information, referred to as batch-attention. As shown in Table III,
compared to the baseline approach, the batch-attention results
in a 1.91% (from 62.18% to 64.09%) increase in mAP and
a 2.68% (from 66.55% to 69.23%) improvement in Rank-1
on the SYSU-MM01 dataset, indicating the effectiveness of
the layers. Furthermore, in comparison to the batch-attention,
fusing features memory information, referred to as memoryattention, leads to an additional 1.79% (from 64.09% to
65.88%) increase in mAP and an extra 2.37% (from 69.23% to
71.60%) improvement in Rank-1 on the SYSU-MM01 dataset.
These results indicate that incorporating features memory
information is beneficial for addressing modality discrepancy
and enhancing VIS-NIR Re-ID performance.
To demonstrate the efficacy of the memory attention layers,
we present optimization curves of the OT loss, calculated both
directly from the input M M of the memory attention layers
fM of the memory attention layers.
and from the output M
As depicted in Fig. 6, the OT loss computed directly from
M M fails to converge to 0 even after 55 000 iterations. ConfM converges to 0 after
versely, the OT loss computed using M
20 000 iterations. This observation suggests that the memory
attention layers exert a beneficial influence on enhancing the
similarity of heterogeneous features within the same identity.
The memory attention layers aim at exploring associations
among heterogeneous memory matrices, effectively mitigating
modality discrepancies.
2) Effectiveness of PFD: Apart from the discrepancy in
modality, variations in pedestrian posture also have a nonnegligible impact on the discriminability of feature representations.
To handle this challenge, we introduce the PFD module,
which integrates skeleton information to effectively disentangle the posture factor from the identity features. The fusion
of skeleton information mitigates the influence of posture
variations, enhancing the overall discriminability of the learned
features. As presented in Tables I and II, experimental results
validate the substantial contribution of the PFD module to the
performance improvement on the SYSU-MM01 and RegDB
datasets. Notably, the PFD module achieves remarkable gains
of 5.33% (from 62.18% to 67.51%) and 5.20% (from 72.81%
to 78.01%) in mAP, as well as notable improvements of
5.92% (from 66.55% to 72.47%) and 1.85% (from 80.92% to

TABLE V
C OMPARISON OF S KELETON AND P OSTURE F EATURES IN P F C L OSS ON
T HE SYSU-MM01 DATASET

82.77%) in Rank-1 accuracy for the SYSU-MM01 and RegDB
datasets, respectively. Moreover, Fig. 5 clearly illustrates the
contrasting performance curves of the model with and without
PFD, emphasizing the pivotal role played by PFD in enhancing
the model’s performance across the entire Rank-1 to Rank-20
range.
The PFD module incorporates two types of information:
parsing and landmark information. The former represents the
pedestrian’s shape profile information, while the latter locates
the keypoints of the pedestrian. In this part, we investigate the
impact of these two types of information on the performance of
the PFD module. As shown in Table IV, compared to the baseline method, fusing the shape profile information, i.e., parsing
leads to a notable improvement of 4.67% (from 62.18% to
66.85%) in mAP and 4.36% (from 66.55% to 70.91%) in
Rank-1 on the SYSU-MM01 dataset. However, considering
that parsing alone may not capture the complete posture of
pedestrians, we further incorporate landmark information into
the PFD module. Compared to parsing alone, fusing parsing
with landmark results in an additional improvement of 0.66%
(from 66.85% to 67.51%) in mAP and 1.56% (from 70.91%
to 72.47%) in Rank-1 on the SYSU-MM01 dataset. The
combination of parsing and landmark information provides
more complete skeleton information to the PFD module, which
is beneficial for effectively decoupling the posture factors.
In addition, when utilizing the impure features E S as
posture-related features to calculate the L PfC loss, it inevitably
affects the decoupling of the posture factor from the identity
representation. To address this issue, we introduce subspace
S
orthogonal decomposition to obtain posture features E po
for
computing the L PfC loss. As demonstrated in Table V, compared to using the impure skeleton features E S , the loss

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:54 UTC from IEEE Xplore. Restrictions apply.

5504

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025

Fig. 7. Visualization for feature distribution by randomly sampling 14 identities of the SYSU-MM01. Each unique identity is represented by a distinct
color. The images from the VIS and NIR modalities are denoted by “cross” and “dot” markers, respectively. The red dotted circles highlight the efficacy of
our DMPF in reducing modality variance, while the blue dotted circles indicate its success in mitigating posture variance. (a) Baseline method. (b) Proposed
method.
TABLE VI
A BLATION S TUDY R EGARDING THE T HREE S UBCOMPONENTS L ID , L P F C ,
AND L O IN PFD L OSS ON T HE SYSU-MM01 DATASET

S
computed using the posture features E po
results in an improvement of 0.42% (from 67.09% to 67.51%) in mAP metrics and
1.16% (from 71.31% to 72.47%) in Rank-1 metrics on the
SYSU-MM01 dataset. These experimental results highlight
the effectiveness of subspace orthogonal decomposition in
extracting posture-related features.
The PFD loss comprises three subterms, i.e., L ID , L PfC and
L O . To investigate their respective contributions to the model’s
performance, we conduct ablation experiments involving the
PFD loss. As illustrated in Table VI, when compared to the
baseline, implemented by removing disentangling losses L PfC
and L O from the DMPF model, the full DMPF exhibits a
3.83% improvement in Rank-1 accuracy (rising from 72.58%
to 76.41%). This underscores the effectiveness of the disentangling losses. To further assess the individual contributions
of L PfC and L O , we conduct separate evaluations for each
disentangling loss. When applied individually, L PfC and L O
exhibit minimal impact on the model’s performance. However, a substantial improvement is observed when these two
terms are combined. This may be attributed to that L PfC
is responsible for capturing pose-related information, while
L O plays a vital role in disentangling pose information from
identity features. When these two loss terms work together in
harmony, they can effectively capture pose-related information
and disentangle pose factors from identity-related features.
3) Qualitative Analysis: To analyze the effectiveness of
DMPF in eliminating modality and posture discrepancies, we
utilize t-SNE [64] to visualize the feature distributions of
the baseline method and DMPF. As depicted in Fig. 7(a),
the baseline method exhibits two primary issues. First, there
is an obvious modality gap between heterogeneous samples
belonging to the same identity, as indicated by the red dotted circles. Second, some outliers are located far from the
intraclass clusters due to posture variations, as represented
by the blue dotted circles. These observations suggest that

Fig. 8. Illustration of Re-ID results between baseline and DMPF. The green
box signifies that the retrieved image corresponds to the same identity as
the query image, while the red box indicates a pedestrian with a distinct
identity from the query. (a) and (d) Demonstrate the effectiveness of the
proposed method in disentangling the modality factor. (c) and (d) Showcase
the effectiveness of the proposed method in disentangling the posture factor.

the baseline method is affected by both modality and posture differences. In contrast, as shown in Fig. 7(b), DMPF
offers two advantages over the baseline method. On the one
hand, DMPF reduces the intraclass distance between different
modalities, as illustrated by the red dotted circles. On the
other hand, DMPF effectively gathers positive samples with
various postures together, as demonstrated by the blue dotted
circles. The comparison between Fig. 7(a) and (b) highlights
the superior performance of DMPF in efficiently addressing
modality and posture discrepancies.
To provide a more intuitive understanding of the performance of DMPF, we visually present the results of Re-ID
in Fig. 8, which showcases ranking result examples of the
baseline method and DMPF on the SYSU-MM01 dataset.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 13 'Distribution aligned semantics adaption for lifelong person re-identification.pdf' - | sed -n '1085,1165p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
4.4 Ablation Studies
For a fairer comparison and to validate our designs, we conduct all ablation experiments using the LUPerson-NL [11] pre-trained weights in Tab. 6. Without loss of
generality, ablation experiments are conducted in Order 1 by default. Besides state-ofthe-art methods, we also compare (1) Fine-Tune: fine-tuning the pre-trained model on

12

Table 6: Ablation Studies of our method. All methods use LUPerson-NL pretrained weights. The results are reported after the last training step in Order 1. “∗”
denotes adding the SA module to the original ResNet-50.
Market-1501

DukeMTMC

CUHK-SYSU

MSMT17

mAP

R-1

mAP

R-1

mAP

R-1

mAP

R-1

smAP

sR−1

AKA [13]
KRC [15]
PTKP [14]

66.7
68.4
83.6

83.7
85.8
93.3

38.3
64.8
69.1

53.3
76.6
81.3

84.5
91.9
89.4

86.1
93.5
90.4

19.1
49.2
41.1

37.4
72.2
65.6

52.2
68.6
70.8

65.1
82.0
82.7

Fine-Tune∗

codex
DASA 的表格跨页，`pdftotext` 在分页处把后半截断开了。我会再单独抽下一页，确认 BN、SA、冻结卷积和核大小这几组关键对照。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 12 -l 15 'Distribution aligned semantics adaption for lifelong person re-identification.pdf' - | sed -n '1,240p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Table 4: Comparison with state-ofthe-art methods on unseen domains
after training in Order 1. The results
(Rank-1) are given by the trained model
after the last training step.

Table 5: Comparison with state-ofthe-art methods on unseen domains
after training in Order 2. The results
(mAP) are given by the trained model
after the last training step.
Methods

Methods CUHK01 GRID SenseReID
Fine-Tune
Joint-Train

55.7
66.0

7.4
19.4

28.7
34.9

LwF [17]
SPD [19]
CRL [20]
PTKP [14]
KRC [15]

73.0
73.9
71.4
69.4
78.0

20.1
20.8
18.7
25.3
25.5

39.1
42.1
40.1
46.1
44.0

DASA

83.5

36.0

50.0

CUHK01 CUHK02 PRID

Fine-Tune
Joint-Train

48.7
65.2

40.5
56.8

11.4
14.8

LwF [17]
iCarL [23]
BiC [21]
WA [22]
PTKP [14]
KRC [15]

60.1
60.4
50.4
50.2
47.7
76.6

52.0
55.1
44.6
48.5
47.6
66.9

27.0
34.9
24.1
21.9
24.2
49.1

DASA

78.6

70.2

64.3

privacy issues, saving exemplars is unrealistic. On the other hand, due to the open
categories of pedestrians, saving a large number of identity images can consume storage
a lot. The storage consumption of exemplars even surpasses that of both the backbone
model and the classifier. Note that in GwFReID [16], all identities are selected into the
exemplar pool, but when comparing it in Fig. 4, we still assume it saves 250 identities
per step, otherwise its storage cost would be enormous (about 1578MB for exemplars).
However, our proposed method is free of exemplars. At each step, only a more
lightweight classifier is needed for identity classification during training. Once the
training is finished, the classifier can also be discarded freely and would not affect the
lifelong evolution of the model. As shown in Tab. 3, the introduced storage consumption at each training step is only equal to 3% of the backbone model (v.s., 70% on
average for advanced competitors). All the results indicate the significant advantages
of our proposed DASA in terms of efficiency and effectiveness in LReID.
Results on unseen domains. Our proposed DASA shows great anti-forgetting
capability and outstanding performance in application domains. We also investigate
its generalization capability on unseen domains. We follow the competitors to use
the model at the last training step which accumulates rich knowledge from all seen
domains. As shown in Tabs. 4 and 5, the generalization ability of DASA outperforms that of all other advanced methods as well as Fine-Tune and Joint-Train on all
datasets in both training orders. The results indicate that robust and general human
semantic knowledge is useful for generalization and DASA shows great effectiveness
in generalizing acquired knowledge.

4.4 Ablation Studies
For a fairer comparison and to validate our designs, we conduct all ablation experiments using the LUPerson-NL [11] pre-trained weights in Tab. 6. Without loss of
generality, ablation experiments are conducted in Order 1 by default. Besides state-ofthe-art methods, we also compare (1) Fine-Tune: fine-tuning the pre-trained model on

12

Table 6: Ablation Studies of our method. All methods use LUPerson-NL pretrained weights. The results are reported after the last training step in Order 1. “∗”
denotes adding the SA module to the original ResNet-50.
Market-1501

DukeMTMC

CUHK-SYSU

MSMT17

mAP

R-1

mAP

R-1

mAP

R-1

mAP

R-1

smAP

sR−1

AKA [13]
KRC [15]
PTKP [14]

66.7
68.4
83.6

83.7
85.8
93.3

38.3
64.8
69.1

53.3
76.6
81.3

84.5
91.9
89.4

86.1
93.5
90.4

19.1
49.2
41.1

37.4
72.2
65.6

52.2
68.6
70.8

65.1
82.0
82.7

Fine-Tune∗
Fine-Tune
DA
SA

25.9
30.0
80.4
80.1

50.6
56.3
91.6
92.1

31.1
38.7
70.9
72.4

49.4
55.6
83.5
84.2

65.9
76.4
93.0
91.8

68.9
79.2
94.1

exec
/bin/zsh -lc "pdftotext -f 1 -l 15 'Distribution aligned semantics adaption for lifelong person re-identification.pdf' - | rg -n -i 'Fine-Tune|DASA|DA|SA|Conv|freeze|kernel|Fig. 5|Table 6|Table 7|ablation|LUPerson'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1:Distribution Aligned Semantics Adaption for
4:1 Fudan University, Shanghai, China.
10:*Corresponding author(s). E-mail(s): xyxue@fudan.edu.cn;
11:Contributing authors: qzwang22@m.fudan.edu.cn; xlqian@nwpu.edu.cn;
12:libin@fudan.edu.cn;
15:adaptable to changes in space and time. Therefore, the adaptation of Re-ID models to new domains while preserving previously acquired knowledge is crucial,
19:data is inappropriate. Additionally, the fine-grained and open-set characteristics of Re-ID limit the effectiveness of the distillation paradigm for accumulating
22:lifelong applications. In this paper, we identify the challenges and discrepancies associated with adapting a pre-trained model to each application domain
23:and introduce the Distribution Aligned Semantics Adaption (DASA) framework.
25:data distribution discrepancy and freezes the pre-trained convolutional layers to
27:Adaption (SA) module, which effectively adapts learned semantics to enhance
30:exhibits significantly reduced storage consumption. DASA presents a novel and
31:cost-effective perspective on effectively adapting pre-trained models for LReID.
32:The code is available at https://github.com/QizaoWang/DASA-LReID.
36:Samples
44:same pre-trained weights
55:Samples
71:using the same pre-trained weights. (b) Conventional knowledge distillation pipeline
72:with exemplars. Our proposed DASA paradigm shows great superiority in LReID.
75:Semantics adaption
78:Person Re-IDentification (Re-ID) aims at recognizing the same pedestrian across disjoint cameras. With significant advancements in deep learning algorithms over the
80:methods usually assume the pedestrian data is provided at once. In real-world scenarios, surveillance data accumulates continuously due to successive deployments and
81:ongoing applications. Re-ID data expands discretely, both spatially and temporally,
82:necessitating the adaptation of Re-ID models to new environments. Consequently, Lifelong person Re-IDentification (LReID) seeks to continuously adapt Re-ID models to
85:domains, merely updating the model with new surveillance data can lead to a loss of
90:[10, 11] adopt unsupervised pre-training to prevailing Convolutional Neural Network
95:degradation stems from the inherent drawbacks of the knowledge distillation paradigm.
101:classifier. However, unlike conventional lifelong learning [17–20], LReID is inherently
104:relying on replaying data of old tasks, namely exemplars [21–24], is impractical due
105:to strict privacy constraints associated with pedestrian data. Secondly, the number
106:of classes in the LReID datasets significantly exceeds those in conventional lifelong
108:great resources when saving it for knowledge distillation.
109:Therefore, to achieve a balance between knowledge preservation and updating with
110:minimal source consumption, and without using exemplars, we introduce the Distribution Aligned Semantics Adaption (DASA) framework, which presents a novel
112:knowledge distillation that uses saved old data and models for knowledge retention.
114:i.e., Batch Normalization (BN) and Convolutional (Conv) blocks. Firstly, BN effectively regularizes the model and captures data distributions, contributing to improved
116:domain distribution differences between pre-training and application. Secondly, Conv
117:layers specialize in encoding data patterns and semantics at different depths of the
118:network. It is expected that the Conv layers trained on a substantial amount of pedestrian data can acquire robust and generalizable human semantic knowledge [10, 11],
120:we propose freezing the pre-trained Conv layers to preserve shared person semantic
122:in complex application scenarios. To adapt the acquired general knowledge to each
123:application domain effectively, we introduce the lightweight Semantics Adaption (SA)
124:module, which can efficiently aggregate, refine, and generate improved pedestrian representations. Adapting the general human semantics to more representative identity
128:knowledge acquired through large-scale pre-training to effectively adapt models in
130:(2) We propose the Distribution Aligned Semantics Adaption (DASA) framework
132:pre-training and application, and incorporates the lightweight semantics adaption
142:stable performance on old tasks while adapting the model to gain new knowledge.
143:Methods can be traditionally divided into three categories, namely regularizationbased, rehearsal-based, and architecture-based methods. Regularization-based methods [17, 18] limit updating important parameters for preceding tasks to mitigate
144:forgetting. To better keep past acquired knowledge, rehearsal-based methods [21, 23,
146:prevalent for their satisfactory performance and simplicity. However, rehearsal-based
148:where data privacy should be considered carefully. Architecture-based methods design
153:or recognition tasks, the presence of imbalanced samples per identity and subtle interclass variations in person Re-ID poses additional challenges. The significant variations
158:pre-prepared stationary training data [1, 3, 5, 33–35]. In response to the demand
160:(LReID) and propose to maintain a learnable knowledge graph to adaptively update
162:retain old knowledge without access to previous data. Recently, state-of-the-art LReID
163:methods have drawn inspiration from rehearsal-based lifelong learning approaches.
168:knowledge consolidation scheme. However, the intrinsic dissimilarities between Re-ID
170:leveraging useful knowledge for LReID. Moreover, privacy concerns limit their applicability in real-world scenarios, where exemplars cannot be stored for lifelong usage.
178:updating with minimal source consumption, and without using exemplars. It differs
179:from existing studies in Test-Time Adaption (TTA) [37, 38] and Domain Generation
181:[37]. For instance, BNTA [38] adapts the model using gallery data and self-supervised
183:entropy at test time. Differently, in LReID, the model is updated during training and
186:LReID, the model is optimized with various domains sequentially and the data from
193:Problem formulation. In LReID, a stream of datasets D = {D(t) }t=1 are used for
194:model training sequentially. Each dataset is regarded as an application domain during
214:number of samples in the t-th training set. At the t-th training step, the model G (t)
216:as feature extractor and a classifier g (t) are updated with Dtrain . Since the identity
226:different datasets obliges previous LReID methods [14–16] to save a small number of
260:Conv
262:SA Module
266:Conv
278:SAl
293:Conv
295:SA Module
299:Conv
310:SA
317:Fig. 2: The framework of DASA. The acquired robust and general human semantics from pre-training are used as shared knowledge, which is kept in the frozen Conv
318:layers. At each training step, we adapt the acquired knowledge from the pre-training
320:Adaption (SA) modules. During the lifelong evolution process, the previously learned
321:BN and SA are used for initialization in the upcoming domain for forward knowledge
324:training step. New samples can also be used for distillation [13, 17]. Previous LReID
327:issues, the training data from previous steps should not be available anymore.
329:3.2 Distribution Aligned Semantics Adaption
334:to adapt the acquired semantic knowledge from the pre-training domain to the target application domain. Two problems need to be addressed, that is, the domain
337:alignment and lightweight semantics adaption, respectively.
339:CNN can normalize the features of each domain to the same reference distribution
345:of the updated BN layers exhibit a similar distribution to the pre-training domain,
346:and the remaining frozen pre-trained Conv layers can receive stable input activations.
349:Concretely, the data D(t) of each domain follows a domain-specific Gaussian distri2
357:) of input feature maps, while updating the corresponding
391:Semantics adaption. Having compensated for the distribution differences between
395:knowledge for Re-ID, so we freeze the pre-trained Conv layers to keep the shared
399:is necessary to adapt the learned general semantics to specific application domains.
400:To this end, we introduce the lightweight Semantics Adaption (SA) module after
401:each frozen Conv layer to aggregate and refine learned semantics effectively. For the
402:sake of efficiency, it could be a good choice to implement SA as one depth-wise Conv
403:layer. Assuming M is the channel dimension of the input feature map, the depthwise Conv layer adopts M kernels and each for one channel of the input feature map,
404:respectively. One of the direct designs is using the 1 × 1 depth-wise Conv layer.
406:the discriminative ability of the model (as shown in Fig. 5). It makes sense since there
407:is no interaction in the spatial dimension to achieve semantics adaption. Intuitively,
410:scenes. Therefore, we propose to apply a relatively large kernel size to aggregate and
411:refine the general semantics. Note that since the SA module performs convolutional
412:operation at each channel separately, the increase in kernel size would not result in a
416:significant increase in parameters. In our experiments, we find that the kernel size of
418:There are some differences when comparing the SA module with LoRA [43], which
420:matrices. Technically, LoRA updates the original weights and changes the calculation operator, adapting the model for different tasks. Differently, the SA module
422:of the model in different application domains. Adapting the general human semantics to more representative identity semantics represents a process from coarse to
425:After training in each domain, we can obtain a set of domain-specific SA modules,
427:oLConv
430:i.e., SA(t) = wl
431:, where wl denotes the parameters of the SA module at
433:layer l, and LConv is the number of Conv layers in the CNN model. Similarly to our
435:learned domain to the next, we use the SA(t−1) as the initial to train SA(t) effectively.
440:domain-specific BN (t) and SA(t) as introduced above. Both of them are lightweight
443:other competitors as shown in Fig. 4. During testing, according to the camera information of the person image, corresponding BN (t) and SA(t) are used. With the help
444:of BN (t) and SA(t) , the model can efficiently evolve and adapt to new application
445:domains, making it effective and practical for new deployment and wide usage in
450:Datasets. We evaluate our proposed method following the widely-used LReID setting [14–16]. Specifically, we investigate its effectiveness in two different training
451:orders on various person Re-ID datasets. Order 1: Market-1501 [44] → DukeMTMCreID [45] → CUHK-SYSU [46] → MSMT17 [47]. Order 2: VIPeR [48] → Market-1501
454:We split VIPeR into training and testing sets following [15]. For other datasets, we
455:follow their original training and evaluation protocols. To investigate its generalization capability, we also evaluate it on the unseen datasets, including CUHK01 [49],
460:Table 1: Comparison with the state-of-the-art methods in the LReID setting of Order 1. “w/ Ex.” denotes rehearsal-based methods using exemplars. The
656:DASA
660:We also evaluate the influence of using different datasets for pre-training, including ImageNet [25] for image classification, LUPerson [10] and LUPerson-NL [11] for
661:person Re-ID. LUPerson is a large-scale unlabeled dataset of 4M images of over 200K
662:identities. LUPerson-NL is derived from LUPerson by applying an online multi-object
663:tracking system on the raw videos of LUPerson. LUPerson-NL consists of 10M images
665:use the LUPerson-NL pre-trained weights for our method, which can provide robust
675:random cropping, and random erasing [54] for data augmentation. The batch size is
676:set to 128, with 2 samples per pedestrian. Adam optimizer [55] with weight decay of
679:a factor of 10 at the 30th epoch for the first dataset and at the 10th epoch for other
680:datasets. Each dataset is trained for 80 epochs. For the SA module, the kernel size is
681:set to 5 × 5 with stride 1, and the bias term is removed for the sake of the number of
696:and 2, respectively. Comparison methods for LReID are reproduced in the same experimental environment according to their released code and training settings. Fine-Tune
697:denotes fine-tuning on different datasets sequentially. Joint-Train denotes combining
698:all datasets for joint training. As shown in Tab. 1, due to the discrepancies across
699:domains, although rehearsal-based methods show great performance in the classification task [21, 22], it contributes to inferior performance gain when being applied to
705:relying on impractical exemplars, DASA still outperforms them significantly.
709:model on seen datasets. Since our LReID paradigm avoids forgetting, we compare the
711:the capabilities in knowledge preservation and knowledge updating, both of which
712:are essential for LReID. As shown in Fig. 3, DASA achieves better average accuracies during the lifelong evolution process on the four datasets. The results show its
713:effectiveness in adapting to the application domains and acquiring knowledge.
714:As shown in Tab. 2, our proposed DASA also achieves state-of-the-art results when
715:training in Order 2. On the small-scale VIPeR [48] dataset with only 632 person
716:images for training, DASA still shows great superiority. By comparing the results of
723:from VIPeR, PTKP achieves bad incremental accuracies on the following datasets.
727:Table 2: Comparison with the state-of-the-art methods in the LReID setting of Order 2. “w/ Ex.” denotes rehearsal-based methods using exemplars. The
911:DASA
934:SA(t)
940:training datasets. (3) The catastrophic forgetting problem is not well addressed by
943:Order 2. In contrast, DASA is robust to different training orders and datasets.
944:Comparison of storage consumption. Storage consumption is also crucial in lifelong scenarios, otherwise, we can save all previous models to prevent catastrophic
951:training step as shown in Tab. 3. What’s worse, an old model having the same large
970:Fine-Tune
1006:DASA
1016:Fine-Tune
1056:DASA
1064:privacy issues, saving exemplars is unrealistic. On the other hand, due to the open
1065:categories of pedestrians, saving a large number of identity images can consume storage
1068:exemplar pool, but when comparing it in Fig. 4, we still assume it saves 250 identities
1075:of our proposed DASA in terms of efficiency and effectiveness in LReID.
1076:Results on unseen domains. Our proposed DASA shows great anti-forgetting
1080:domains. As shown in Tabs. 4 and 5, the generalization ability of DASA outperforms that of all other advanced methods as well as Fine-Tune and Joint-Train on all
1081:datasets in both training orders. The results indicate that robust and general human
1082:semantic knowledge is useful for generalization and DASA shows great effectiveness
1085:4.4 Ablation Studies
1086:For a fairer comparison and to validate our designs, we conduct all ablation experiments using the LUPerson-NL [11] pre-trained weights in Tab. 6. Without loss of
1087:generality, ablation experiments are conducted in Order 1 by default. Besides state-ofthe-art methods, we also compare (1) Fine-Tune: fine-tuning the pre-trained model on
1091:Table 6: Ablation Studies of our method. All methods use LUPerson-NL pretrained weights. The results are reported after the last training step in Order 1. “∗”
1092:denotes adding the SA module to the original ResNet-50.
1165:Fine-Tune∗
1166:Fine-Tune
1167:DA
1168:SA
1220:DASA (Ours)
1279:different datasets sequentially; (2) DA: tuning BN layers to achieve distribution alignment at each training step; (3) SA: freezing the pre-trained model and only adding and
1280:tuning our proposed SA modules; (4) our proposed DASA; (5) Joint-Train: combining
1281:all datasets for joint training. Note that for SA, since BNNeck [33] is not trained during pre-training, it is also trained at each training step. In Fig. 6, we further investigate
1283:choices of kernel sizes for the SA module, the positions to use SA, and the effectiveness
1284:of SA in adapting semantics via attention map visualization.
1285:Effectiveness of distribution alignment. As shown in Tab. 6, DA can achieve
1287:data distribution on LReID. Tuning BN layers eliminates the distribution differences
1289:human semantics effectively. Unfortunately, conventional architecture-based lifelong
1292:Effectiveness of the SA module. The results in Tab. 6 show that SA also significantly beats all advanced competitors [13–15] using the same pre-trained weights. It
1293:demonstrates that just using the pre-trained weights from large-scale person datasets
1295:paradigm limits their performance upper bound. However, our proposed SA is effective
1297:try to add the SA module to the ResNet-50 backbone. For Fine-Tune, the previously
1298:learned semantics are destroyed and tuned to fit the current domain, so incorporating the SA module may lead to inflexible knowledge updating and severe knowledge
1299:forgetting, resulting in performance degradation. In contrast, SA shows its great effectiveness in our proposed LReID paradigm. Additionally, since Joint-Train can acquire
1303:Table 7: Influence of applying the SA
1306:datasets at the last training step are reported.
1320:Fig. 5: Ablation
1322:kernel size for the SA module.
1323:The average accuracies of all datasets
1378:all data from all application domains, the SA module can also help improve semantics
1379:with few parameters. With proper design and usage, SA can bring great benefits.
1380:Necessity of adopting both distribution alignment and semantics adaption.
1381:As shown in Tab. 6, combining both DA and SA achieves the best results. On the
1383:to represent pedestrians, the data distribution discrepancy between pre-training and
1384:application would result in ineffective use of the learned semantics for specific applications. On the other hand, with the help of the SA module for aggregating semantic
1386:MSMT17 datasets, DASA brings 6.4% and 9.5% mAP improvement over DA, respectively. Note that DASA achieves competitive results with Joint-Train, and even better
1387:mAP and R-1 on the CUHK-SYSU dataset. We guess it is because the image styles and
1388:data distributions of CUHK-SYSU differ greatly from others. When trained jointly,
1389:the model has to make a compromise between datasets. On the contrary, without
1390:being influenced by distribution discrepancy, DASA leverages acquired knowledge of
1391:human semantics with the SA module to represent pedestrians effectively.
1392:Influence of different kernel sizes of the SA module. In Fig. 5, when setting the
1393:kernel size of the SA module to 1, we observe the performance decrease. The results
1394:support our motivation that since the 1×1 depth-wise convolutional layer does not
1396:and adapt acquired semantic knowledge. When increasing the kernel size to 3 or 5, we
1398:the kernel size results in less performance gain. Note that setting the kernel size to 7
1400:number of parameters, we set the kernel size of the SA module to 5.
1401:Where to use the SA module. Considering grid search on each layer takes a great
1402:effort, we explore adopting SA in different layers of ResNet-50 [12] in Tab. 7. (1) By
1403:comparing Methods 2∼4, we find that the SA module brings a better effect in deep
1412:Fig. 6: Influence of different pre-training data for LReID. (a) smAP and (b)
1414:in semantics while the shallow layers are better at capturing curves and image patterns. (2) Methods 2∼4 all outperform Method 1, showing the effectiveness of SA
1415:in adapting acquired knowledge for better pedestrian representations. (3) Method 5
1417:the better performance, we apply the lightweight SA module after each frozen Conv
1420:Influence of different pre-training data. According to the results of Fig. 6, we
1423:useful human semantics for LReID. (2) DA using LUPerson pre-trained weights outperforms that using ImageNet pre-trained weights, showing that pre-training on the
1424:person dataset provides useful knowledge for LReID. However, it still performs worse
1425:than that using LUPerson-NL pre-trained weights. The results demonstrate that when
1426:the person dataset used for pre-training is not large-scale, the model cannot acquire
1427:robust and general human semantics effectively. (3) SA outperforms DA when using
1429:knowledge for LReID, since it is still suboptimal despite aligning the data distribution via DA. On the other hand, the SA module can aggregate acquired knowledge for
1430:object classification to promote LReID to a certain extent, demonstrating its effectiveness. (4) DA achieves competitive or even better results with SA when using LUPerson
1431:or LUPerson-NL pre-trained weights, showing the nonnegligible impact of the data
1432:distribution gap. (5) Aligning distribution or adopting SA can facilitate LReID in different pre-training choices. Combining them shows great superiority over Fine-Tune,
1433:and good pre-trained weights can expand the upper bound of DASA. It is appealing
1435:Visualization of attention maps. To intuitively understand how the SA module
1436:achieves semantics adaption from pre-training to the application domain, we visualize the attention maps in Fig. 7. We observe that (1) Since Market-1501 is relatively

codex
我已拿到 DASA 的关键消融：单独调 BN、单独 SA、二者组合、SA 核大小、SA 插入位置、预训练数据来源和注意力图。接下来补后半段可视化描述，并抽 DiverseReID、DCR、DMDL 的消融线索。
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
/bin/zsh -lc "pdftotext -f 13 -l 15 'Distribution aligned semantics adaption for lifelong person re-identification.pdf' - | sed -n '180,360p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
80.5

89.1
89.6

Methods

Average

different datasets sequentially; (2) DA: tuning BN layers to achieve distribution alignment at each training step; (3) SA: freezing the pre-trained model and only adding and
tuning our proposed SA modules; (4) our proposed DASA; (5) Joint-Train: combining
all datasets for joint training. Note that for SA, since BNNeck [33] is not trained during pre-training, it is also trained at each training step. In Fig. 6, we further investigate
the influence of various pre-training choices on different methods. We also explore the
choices of kernel sizes for the SA module, the positions to use SA, and the effectiveness
of SA in adapting semantics via attention map visualization.
Effectiveness of distribution alignment. As shown in Tab. 6, DA can achieve
better results than advanced competitors [13–15], confirming the great impact of
data distribution on LReID. Tuning BN layers eliminates the distribution differences
between pre-training and application, which contributes to taking advantage of learned
human semantics effectively. Unfortunately, conventional architecture-based lifelong
learning methods designed for classification or recognition tasks [29, 30] do not concern the influence of BN layers. They try to keep and use acquired knowledge but do
not mitigate the domain distribution gap, making them unsuitable for LReID.
Effectiveness of the SA module. The results in Tab. 6 show that SA also significantly beats all advanced competitors [13–15] using the same pre-trained weights. It
demonstrates that just using the pre-trained weights from large-scale person datasets
for model initialization cannot effectively promote LReID. The knowledge distillation
paradigm limits their performance upper bound. However, our proposed SA is effective
in leveraging human semantics learned from pre-training to promote LReID. We also
try to add the SA module to the ResNet-50 backbone. For Fine-Tune, the previously
learned semantics are destroyed and tuned to fit the current domain, so incorporating the SA module may lead to inflexible knowledge updating and severe knowledge
forgetting, resulting in performance degradation. In contrast, SA shows its great effectiveness in our proposed LReID paradigm. Additionally, since Joint-Train can acquire

13

Table 7: Influence of applying the SA
module in different layers of ResNet-50.
The average incremental accuracies of four
datasets at the last training step are reported.

(b) DukeMTMC

Layers

Methods
1
2
3
4
5

(e) Average
Fig. 5: Ablation
Study of the
kernel size for the SA module.
The average accuracies of all datasets
at the last training step are reported.

Ours

1

2

✓

✓
✓

✓

3

smAP

sR-1
83.4
84.6
86.8
87.0
87.5
87.6

4

✓

✓
✓
✓

✓
✓

71.0
72.2
75.3
76.0
76.4

✓

✓

✓

76.7

(d) MSMT17

all data from all application domains, the SA module can also help improve semantics
with few parameters. With proper design and usage, SA can bring great benefits.
Necessity of adopting both distribution alignment and semantics adaption.
As shown in Tab. 6, combining both DA and SA achieves the best results. On the
one hand, although the pre-trained model can acquire human semantic knowledge
to represent pedestrians, the data distribution discrepancy between pre-training and
application would result in ineffective use of the learned semantics for specific applications. On the other hand, with the help of the SA module for aggregating semantic
representations after distribution alignment, on the challenging DukeMTMC-reID and
MSMT17 datasets, DASA brings 6.4% and 9.5% mAP improvement over DA, respectively. Note that DASA achieves competitive results with Joint-Train, and even better
mAP and R-1 on the CUHK-SYSU dataset. We guess it is because the image styles and
data distributions of CUHK-SYSU differ greatly from others. When trained jointly,
the model has to make a compromise between datasets. On the contrary, without
being influenced by distribution discrepancy, DASA leverages acquired knowledge of
human semantics with the SA module to represent pedestrians effectively.
Influence of different kernel sizes of the SA module. In Fig. 5, when setting the
kernel size of the SA module to 1, we observe the performance decrease. The results
support our motivation that since the 1×1 depth-wise convolutional layer does not
involve the spatial interaction of input feature maps technically, it cannot aggregate
and adapt acquired semantic knowledge. When increasing the kernel size to 3 or 5, we
can observe a significant performance improvement. However, continuing to increase
the kernel size results in less performance gain. Note that setting the kernel size to 7
requires almost twice as many parameters as 5. Considering the performance and the
number of parameters, we set the kernel size of the SA module to 5.
Where to use the SA module. Considering grid search on each layer takes a great
effort, we explore adopting SA in different layers of ResNet-50 [12] in Tab. 7. (1) By
comparing Methods 2∼4, we find that the SA module brings a better effect in deep
layers. It is consistent with our motivation since the deep layers are more specialized

14

(a)

(b)

Fig. 6: Influence of different pre-training data for LReID. (a) smAP and (b)
sR−1 at the last training step are reported using different methods.
in semantics while the shallow layers are better at capturing curves and image patterns. (2) Methods 2∼4 all outperform Method 1, showing the effectiveness of SA
in adapting acquired knowledge for better pedestrian representations. (3) Method 5
shows competitive performance with Ours with even less resource cost. Considering
the better performance, we apply the lightweight SA module after each frozen Conv
layer. It would be also effective to adopt it only in the deep layers for more efficient
deployment and application.
Influence of different pre-training data. According to the results of Fig. 6, we
observe that (1) Methods based on ImageNet pre-training generally achieve inferior
results. Since ImageNet includes all kinds of objects, pre-training on it brings less
useful human semantics for LReID. (2) DA using LUPerson pre-trained weights outperforms that using ImageNet pre-trained weights, showing that pre-training on the
person dataset provides useful knowledge for LReID. However, it still performs worse
than that using LUPerson-NL pre-trained weights. The results demonstrate that when
the person dataset used for pre-training is not large-scale, the model cannot acquire
robust and general human semantics effectively. (3) SA outperforms DA when using
ImageNet pre-trained weights. On the one hand, the model does not acquire useful
knowledge for LReID, since it is still suboptimal despite aligning the data distribution via DA. On the other hand, the SA module can aggregate acquired knowledge for
object classification to promote LReID to a certain extent, demonstrating its effectiveness. (4) DA achieves competitive or even better results with SA when using LUPerson
or LUPerson-NL pre-trained weights, showing the nonnegligible impact of the data
distribution gap. (5) Aligning distribution or adopting SA can facilitate LReID in different pre-training choices. Combining them shows great superiority over Fine-Tune,
and good pre-trained weights can expand the upper bound of DASA. It is appealing
to promote LReID with our paradigm following the trend of large-scale pre-training.
Visualization of attention maps. To intuitively understand how the SA module
achieves semantics adaption from pre-training to the application domain, we visualize the attention maps in Fig. 7. We observe that (1) Since Market-1501 is relatively
easy with greater inter-class differences, the model can leverage learned general human
15


exec
/bin/zsh -lc "pdftotext -f 1 -l 18 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | rg -n -i 'ablation|CAI|CBT|augmentation|label refinement|feature alignment|modality bias|visual|t-SNE|Figure|Table|Effectiveness|component'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
12:Two-stage learning pipeline has achieved promising results in unsupervised visibleinfrared person re-identification (USL-VI-ReID). It first performs single-modality learning and then operates cross-modality learning to tackle the modality discrepancy. Although promising, this pipeline inevitably introduces modality bias: modality-specific
15:model level, we propose a Causality-inspired Adjustment Intervention (CAI) module
17:optimization level, a Collaborative Bias-free Training (CBT) strategy is introduced to
18:interrupt the propagation of modality bias across data, labels, and features by integrating modality-specific augmentation, label refinement, and feature alignment. Extensive
42:relationships to achieve cross-modality discrimination capabilities. Although promising, it also suffers from a modality bias issue that restricts the overall results. The first
44:model inevitably introduces modality bias into the cross-modality learning, leading to
48:patterns, leading to modality-related features. In summary, modality bias originating
67:Figure 1: Existing USL-VI-ReID methods suffer from modality bias, leading to modality-related features.
72:To address the aforementioned modality bias issue, we propose a Dual-level Modality Debiasing Learning (DMDL) framework. DMDL performs modality debiasing at
74:learning modality bias in incorrect cross-modality relationships, and the latter aims to
76:To this end, a Causality-inspired Adjustment Intervention (CAI) module and a Collaborative Bias-free Training (CBT) strategy are proposed. Specifically, CAI facilitates
78:likelihood method, the causal modeling in CAI is theoretically unaffected by modality bias, thereby achieving a low-biased model. To further prevent biased knowledge
79:from deepening during optimization, we propose the CBT strategy, integrating data
80:augmentation, label refinement, and feature alignment. CBT first introduces a pseudomodality augmentation scheme to modify modality-specific cues in images. Based on
81:the augmented images, a cross-modality label smoothing scheme and a feature alignment loss are proposed to refine the biased relationships and learn shared knowledge
82:across pseudo-modalities, respectively. By jointly leveraging these components, CBT
83:explicitly interrupts the propagation of modality bias across data, labels, and features.
90:(1) We investigate the modality bias issue for existing USL-VI-ReID methods and propose a Dual-level Modality Debiasing Learning (DMDL) framework performed at
92:(2) We propose a Causality-inspired Adjustment Intervention (CAI) module at the
95:(3) We propose a Collaborative Bias-free Training (CBT) strategy at the optimization
96:level, combining label refinement and feature alignment with modality-specific
97:data augmentation to prevent fitting biased knowledge.
99:demonstrate the effectiveness and superiority of our method.
117:through cross-modality feature mapping, constructing a pseudo cross-modality identity space to facilitate effective feature alignment. Despite their effectiveness, these
118:methods are inherently constrained by the two-stage pipeline, which inevitably introduces modality bias and hinders the modality-invariant learning.
125:abandoning the two-stage pipeline, these methods still suffer from the modality bias
144:relationships. In the USL-VI-ReID task, DPIS [29] and MMM [30] incorporate noisylabel learning by fitting a two-component Gaussian Mixture Model (GMM) to the loss
153:Fig. 2, incorporating the Causality-inspired Adjustment Intervention (CAI) module
154:and the Collaborative Bias-free Training (CBT) strategy. In cross-modality learning,
155:DMDL first iteratively matches clusters across different modalities to obtain crossmodality relationships as a kind of pseudo-label. Then, CAI employs a backdoor adjustment algorithm to implement causal intervention, which guides the model to capture causal patterns, resulting in a low-biased model. Furthermore, to avoid misleading
156:optimization caused by biased cues, CBT incorporates label refinement and feature
157:alignment with modality-specific data augmentation to jointly mitigate modality bias
249:Figure 2: The framework of the proposed DMDL. After obtaining cross-modality pseudo-labels through
251:strategy combines label refinement and modality alignment with data augmentation to optimize the model,
252:further eliminating modality bias during training.
316:the causal modeling in the proposed CAI module, which constructs a cross-modality
317:model that is insensitive to modality bias.
349:Figure 3: (a) The structural causal model in cross-modality learning for USL-VI-ReID. (b) The modified
357:C → Y reflects the modality bias issue: due to the unsupervised learning pipeline,
375:This provides a principled mechanism for modality debiasing in unsupervised crossmodality learning and motivates our implementation of intervention in CAI to prevent
376:the model from learning modality bias through the backdoor.
378:Based on the above analysis, an intervention loss Lcai is constructed by maximizing
379:the intervention probability to eliminate the interference of the modality bias:
380:Lcai = E x,y [− log P(Y = y|do(X = x))],
384:where x denotes an input image, and y represents its associated cross-modality pseudolabel. To achieve that, CAI implements the computation of P(Y|do(X) by backdoor
415:We provide further analysis of CAI. Compared to the likelihood model P(Y|X)
426:the backdoor adjustment modifies P(C = c|X) to P(C = c), which can be seen as blocking the correlation between modalities C and images X. It eliminates the modality bias
430:Although a low-biased model is obtained through CAI, the biased modality-specific
432:problem, we propose the CBT strategy to mitigate modality bias at the optimization
433:level. Specifically, considering that modality bias propagates from data into labels and
434:features, CBT integrates label refinement and feature alignment with well-designed
435:data augmentation, thereby disrupting bias propagation and promoting unbiased feature
437:3.4.1. Data Augmentation in CBT
438:CBT first introduces a modality-specific augmentation scheme to destroy modalityrelated information in images, as shown in Fig. 4. Specifically, for infrared images, we
441:to increase diversity and introduce randomness to the augmentation by randomly sampling R, G, and B channels of multiple generated pseudo-color images and combining
443:channel augmentation (CA) [1] through channel multiplexing to generate augmented
445:This modality-specific data augmentation enables the image and its corresponding augmentation to share the same identity-discriminative information but differ in
446:modality-related information, mitigating the modality bias at the data level. With the
447:assistance of such augmentation, CBT implements label refinement and feature alignment to facilitate bias-free learning.
498:Figure 4: Illustration of the modality-specific augmentation. Circles represent channels of images. Subscript
503:3.4.2. Label Refinement in CBT
504:To refine the noise pseudo-labels, CBT employs label smoothing by exchanging
526:yai represent the refined soft labels of image xi and its augmentation
527:xia , respectively. Then, they are used to supervise model training by modifying the Lcai
531:Lcai = Ei [ −
564:The overall loss distribution Lid is then fitted with a two-component GMM, where
565:one component corresponds to low-loss samples (indicating high-quality labels) and
567:estimates the probability that a given loss Liid belongs to the low-loss component, which
604:certainty, the proposed label refinement directly constructs low-biased cross-modality
607:labels by incorporating modality-specific augmentations and dynamically updating the
608:modality-shared memory. Since an image and its modality-specific augmentation contain different modality-related information, exchanging their predictions for label smoothing effectively reduces label noise induced by modality-specific cues. Furthermore,
611:3.4.3. Feature Alignment in CBT
612:In addition, a feature alignment loss is introduced to further enhance bias-free feature learning. It is well understood that identity-discriminative information should remain consistent under augmentation. Therefore, for an image and its modality-specific
613:augmentation, the model is expected to extract similar features; otherwise, it suggests
637:and its augmentation representations to be close in the metric space, thereby mitigating
642:+ λcai · Lcai + λ f a · L f a + λtri · Ltri ,
647:where λcai , λ f a and λtri are weights of the corresponding loss term.
649:modeling level, the CAI module performs causal intervention via backdoor adjustment,
653:shortcuts, thereby constructing a low-biased model. Building upon CAI, the CBT further mitigates bias propagation during the optimization process. The modality-specific
654:data augmentation disrupts modality cues at the data level, label refinement corrects biased pseudo-labels at the label level, and feature alignment enforces modality-invariant
655:representations at the feature level. These components collaboratively prevent biased
656:information from being amplified through iterative training. Importantly, CAI and CBT
657:play complementary roles. CAI suppresses modality bias at the modeling level by reshaping the learning objective, while CBT prevents residual bias from being propagated
660:robust and stable bias suppression throughout the learning pipeline.
677:splits in ten trials to ensure stable performance. Evaluation metrics include cumulative
684:input images are resized to 288 × 144, and standard data augmentation techniques,
696:To validate the effectiveness of our DMDL, we compare it with state-of-the-art
699:datasets are shown in Table 1, and the experimental results for the LLCM dataset are
700:presented in Table 2.
1563:Table 1: Comparison with the state-of-the-art methods on SYSU-MM01 and RegDB. Rank at r accuracy(%),
1790:Table 2: Comparison with the state-of-the-art methods on the LLCM dataset. Rank at r accuracy(%), mAP
1794:semi-supervised counterparts, as reported in Table 1. These results highlight the potential of USL-VI-ReID, which eliminates the need for annotations and offers greater
1796:Comparison with unsupervised VI-ReID Methods. The results in Table 1 demonstrate that our method achieves superior performance under the unsupervised VI-ReID
1802:approaches MCL [18] and ASM [17], demonstrating the strong effectiveness of our

exec
/bin/zsh -lc "pdftotext -f 1 -l 15 'DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf' - | rg -n -i 'ablation|DSH|DED|diversity|style|sampling|visual|t-sne|Fig. 3|Fig. 4|Fig. 5|Table 9|Table|Style|Diversity'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
11:Style Hallucination and decoupled domain experts
29:with domain shift. A classical domain generalization (DG) solution is to enhance the diversity of source
31:methods typically mix images from different domains in a mini-batch to generate novel styles, but the mixing
32:coefficient sampled from predefined Beta distribution requires careful manual tuning and may render suboptimal performance. To this end, we propose a plug-and-play Dynamic Style Hallucination (DSH) module that
33:adaptively adjusts the mixing weights based on the style distribution discrepancy between image pairs, which is
35:manual tuning of parameters but also significantly enriches style diversity by expanding the perturbation space
36:to the utmost. In addition, to promote inter-domain diversity, we devise a Domain Experts Decoupling (DED)
40:verify that style diversity, not just the size of the training data, is crucial for enhancing generalization.
71:which can be roughly divided into but not limited to: (a) adversarial
77:Among recent DG methods, feature-based data augmentation approaches like MixStyle (Zhou, Yang, Qiao et al., 2020), pAdaIN (Nuriel,
93:avoid overfitting to source data by introducing channel-level style perturbations to emulate potential domain shifts. Despite their popularity
94:and superiority in domain generalization tasks, existing style augmentation methods are limited in: (1) when synthesizing novel styles, the
95:weights for two instances are obtained by sampling from the manually
99:enriched diversity of newly produced samples, which could not always
102:disappears when synthesized sample diversity is insufficient.
105:inter-domain and intra-domain style space to maximize the source
106:diversity for generalized re-ID. Particularly, to resolve the first issue
107:and achieve intra-domain style diversification, we propose a Dynamic
108:Style Hallucination (DSH) module that dynamically generates new
109:styles by mining the relations across sample pairs in an adaptive
111:the style distribution discrepancy between the input image pairs, and
112:the reciprocal of similarity distance is utilized as the mixing coefficients. The DSH module could alleviate the tedious manual tuning of
114:sample pairs, the DSH module adaptively adjusts the weighting factors
117:style diversity.
119:a domain-specific memory for each domain (also termed as domain experts) and devise a Domain Experts Decoupling (DED) loss to constrain
121:against other domains. The DED loss is implemented by minimizing the
130:style perturbations in the training data. For instance, MixStyle (Zhou,
136:descriptors across modalities. DACS (Yang et al., 2024) designs a style
137:transformation model with diversity and authenticity constraints to
142:in style statistics in earlier layers. SNR (Jin et al., 2020) filters out
151:ISR (Dou et al., 2023) puts forward a reliability-guided contrastive
175:and distribution-guided refining. As an interesting alternative to metalearning, Tan, Ding, Wang, Gong, and Jia (2023) propose an efficient
179:often overlooked the importance of data diversity in enhancing generalization. Different from previous data augmentation methods that
181:underscores the effect of data diversity in DG, and expands both interdomain and intra-domain style space to maximize the source diversity.
185:• We propose a new Dynamic Style Hallucination module that
186:dynamically generates novel styles by mining the relations across
188:intra-domain diversity and spares the requirement of manual
192:source domains’ styles, which is highly efficient in disentangling
205:2.2. Style manipulation for domain generalization
206:The proposed work is mainly dedicated to diversifying the styles
215:scenarios during training. For instance, MixStyle (Zhou, Yang, Qiao
222:& Yoo, 2021) introduces a style randomization module by interpolating
224:Zhao, Sebe, & Lee, 2022) presents a style hallucination module that
225:selects basis styles from the source distribution, and simulates potential
226:style variations through combination and perturbation. SPC (Huang
227:et al., 2023) projects unseen styles into the style representation space
229:devises an Adversarial Style Synthesis module that consists of a multibranch style generation and fusion strategy. DSM (Li, Song, Ni, & Shen,
230:2023) maintains a single-domain style queue for each source domain to
231:enable style mixing. DACS (Yang et al., 2024) designs a style transformation model with diversity and authenticity constraints, enabling it
232:to generate styles that closely resemble the true data distribution.
234:with existing style augmentation works: (1) their efficacy actually heavily depends on delicately designed weight parameters, whose optimal
235:combinations are time-consuming to acquire; (2) existing methods produce new styles via merely mixing the existing styles from few observed
237:insufficient diversity which may hamper the model optimization. In
238:this paper, we explicitly consider and explore the style diversity in
241:ensure sample diversity by formulating the weights in an adaptive
243:providing a substantial degree of style diversity. The proposed DSH
247:previous works in that the DED loss is proposed to achieve inter-domain
248:diversification, which is equally crucial as intra-domain style diversity
249:for enhancing DG performance. Experiments and ablation studies show
250:that DED loss is orthogonal to the intra-domain DSH module and they
267:a domain-specific sampling strategy where each mini-batch contains
274:loss. To maximize the dissimilarity of representations, the DED loss is
277:the combination of identification loss and DED loss. At the end of each
282:2.3. Diversity via feature decorrelation
314:category-related and context-related features to deal with the diversity shift. RaMoE (Dai et al., 2021) proposes a decorrelation loss by
323:feature decorrelation with a novel Domain Experts Decoupling (DED)
324:loss, which provides a flexible and low-cost solution to enrich diversity by minimizing the correlation among the source domain experts.
341:Next, in order to impose inter-domain diversity and keep source
365:Decoupling loss 𝐷𝐸𝐷 to achieve inter-domain style variance. The DSH module is exploited to induce intra-domain diversity.
370:the matrix. Then, the DED loss is introduced to minimize the squared
382:synthesized styles that remain closely related to existing ones and the
383:insufficient diversity may notably jeopardize the generalization.
385:novel Dynamic Style Hallucination module by mining the relations
391:deviation to represent the style distribution:
398:diversity and greatly promote the generalization performance. Taking
418:The DSH module constructs a novel randomized style by interpolating between the styles of 𝐹 and a sampled style 𝐹 ′ . Various sampling
420:to be mixed, like random sampling within one batch regardless of
421:domain labels, cross-domain sampling that selects from two arbitrary
422:domains (Zhou, Yang, Qiao et al., 2020), or farthest point sampling
423:(FPS) (Qi, Yi, Su, & Guibas, 2017) that select basis styles which are
424:most dissimilar with the current samples. The influence of sampling
425:strategies will be discussed in Table 9. In more detail, the generated
426:styles ̂
451:iteration, we compute the average of all features in the current minibatch and combine it with the encoded features belonging to class
466:2018) parameterized by pre-defined 𝛼. Other works resort to sampling
469:Tan et al., 2023) to deduce the interpolation weights, but there remain
474:3.4. Dynamic style hallucination
475:It has been widely acknowledged that style diversifying is critical to
477:styles to generate novel styles (Zhang et al., 2018), with combination
480:potential of style diversification in: (1) the mixing weights sampled
487:Fig. 2. Conceptual overview of DSH. (a) Existing stylization methods typically sample mixing coefficients from Beta distribution and require careful tuning of parameter 𝛼; (b)
488:the proposed Dynamic Style Hallucination formulates the weights in a dynamic manner.
505:weight coefficient, where +1 is to make the calculation more stable.
514:Finally, a new hallucinated style is synthesized by combining the
515:styles of 𝐹 , including its mean and variance, with the mean and
516:variance of a sampled style 𝐹 ′ , as shown in Eqs. (10) and (11). This
517:newly constructed style is then used to replace the original style 𝐹 :
526:be very close to 𝐹 and only brings marginal diversity, thus a smaller
529:enriching the style diversity, which therefore should be given more
530:weight. By means of this, the need for manual hyper-parameter finetuning can be eliminated, and more diverse styles that are different
532:the parameter-free DSH module is inserted at the first stage of the CNN
534:DSH is removed from the feature extractor, which makes our network
569:Obtain feature 𝑓 𝑘 from model 𝜃 equipped with DSH;
578:Compute and accumulate loss DED ;
597:images (all 8000 identities) for training. It is noteworthy that RandPerson (Wang et al., 2020) is excluded from being used as the target domain since it only includes the training subset. To ensure the
632:Table 1
859:Table 2
1004:validity of the proposed DED loss and DSH module, which provides
1006:sample diversity; (3) equipping the proposed approach with stronger
1023:The activation probability of DSH is set to 1.0. Besides, we conduct
1028:4.3. Ablation study
1029:4.3.1. The effect of DSH and DED
1030:Our framework has two important components: DSH in early bottlenecks to bring about style variance, and the DED loss 𝐷𝐸𝐷 to enhance
1031:the disparity between features from various source domains. To evaluate the contribution of each component, we separately add DSH and
1032:𝐷𝐸𝐷 to the Baseline and compare the performance in Tables 6 and 7.
1033:Note that the ablation study is conducted under both Protocol-2 and
1035:improvement of overall performance. Among them, adding DSH seems
1037:of data diversity, and combining two techniques leads to further performance gain, proving that they are complementary. Furthermore,
1038:it can be seen that the performance gain brought by DSH is more
1044:methods under different protocols. The results are listed in Tables 1–5
1050:source domain is involved. From Tables 1–5, the following observations
1060:Fig. 3. Ablation study on the position at which DSH is applied under Protocol-1. Experiments are conducted under the M+D+MS → C3 setting.
1062:Table 3
1173:Table 4
1322:Table 5
1452:Fig. 4. Comparisons between DSH and existing stylization methods under Protocol-2. Experiments are done under the MS+CS+C3 → M setting.
1454:Fig. 5. Ablation study on the value of 𝜔. Experiments are conducted under the MS+CS+C3 → M setting.
1457:To investigate the effect of the proposed DSH, we compare it with
1459:Fig. 4. Our competitors include MixStyle (Zhou, Yang, Qiao et al.,
1461:et al., 2021), Style Randomization (SR) in SagNet (Nam et al., 2021),
1462:Style Hallucination (SH) in SHADE (Zhao et al., 2022), Style Projected
1465:For a fair comparison, we only replace the DSH module with different
1466:style generation variants. Note that 𝐷𝐸𝐷 is removed in this section by
1467:setting 𝜔 = 0 to shed light on the effectiveness of DSH. We observe
1469:DSH, but its Rank-1 score is lower than that of DSH. Meanwhile, other
1470:comparing methods tend to deliver inferior results compared to DSH.
1473:and lead to suboptimal performance; (ii) the generated styles remain
1475:diversity and hurts the generalization capacity. On the contrary, our
1477:proposed DSH module.
1479:Table 6
1480:Ablation study on the impact of different components under Protocol-2.
1481:DSH
1543:Table 7
1544:Ablation study on the impact of different components under Protocol-5.
1545:DSH
1607:4.3.2. Ablation study on where to apply DSH
1611:ensure inter-domain diversity. The feature disentanglement technique
1615:the M+D+MS→C3 and D+C3+MS→M setting. The DSH module is
1618:in Table 8, exploiting 𝐷𝐸𝐷 can achieve the best re-ID accuracies,
1621:Despite its effectiveness, the best place to apply DSH within the
1622:ResNet50 model is unclear. We thus investigate where to apply DSH
1623:in Fig. 3. The results indicate that positioning DSH right after the
1624:first stage yields the best outcomes. Adding DSH at later stages does
1625:not contribute to further performance gains. Specifically, when DSH
1627:the features produced by stage 4 rarely contain style information, and
1636:Fig. 6. Grad-CAM visualization results. (a) Original images. (b) Heatmap of baseline. (c) Heatmap of the proposed DiverseReID.
1638:Table 8
1689:Table 9
1692:Sampling
1704:DSH
1718:DSH
1732:DSH
1744:D+C3+MS→M setting in Table 10. We can observe that our method
1746:DSH is parameter-free, the extra parameters are introduced by the
1757:Table 10
1779:In Fig. 5, we present a detailed analysis focusing on the sensitivity of
1783:is deprecated and leads to degraded performance. When 0.2 ≤ 𝜔 ≤ 1.0
1792:sufficient diversity and better generalization.
1794:4.3.8. Visualization analysis
1796:we visualized the pedestrian images and the model activation maps
1800:areas are missed. This is attributed to the enriched style diversity
1805:4.3.5. Design-space analysis on DSH
1806:In this section, we examine the design choices in DSH by altering the
1807:sampling strategy and weights computation approach in Table 9. Concretely, we investigate three popular sampling strategies, i.e. random
1808:sampling, cross-domain sampling, and farthest point sampling (FPS).
1809:As for the combination weights, we alter it by sampling from Beta
1812:be seen that although FPS can cover the rare styles to the utmost
1813:extent in theory, it is marginally superior over random sampling in
1814:our practice while requiring extra memory cost to store style basis.
1815:Hence, we suggest that random sampling is a better choice. Besides,
1816:DSH consistently outperforms Beta and Dir sampling in all settings,
1817:justifying this adaptive manner to deduce combination weights.
1820:In this paper, we propose a novel framework DiverseReID for generalized person re-ID. With a combination of Dynamic Style Hallucination and Domain Experts Decoupling, intra-domain and inter-domain
1827:style diversity among the source data is greatly enhanced respectively,
1835:Jin, X., Lan, C., Zeng, W., Chen, Z., & Zhang, L. (2020). Style normalization and
1846:Li, Y., Song, J., Ni, H., & Shen, H. T. (2023). Style-controllable generalized person reidentification. In Proceedings of the 31st ACM international conference on multimedia
1857:Liao, S., & Shao, L. (2020). Interpretable and generalizable person re-identification
1860:Liao, S., & Shao, L. (2022). Graph sampling based deep metric learning for generalizable
1863:Lin, G., Bao, Z., Huang, Z., Li, Z., Zheng, W., & Chen, Y. (2024). A multi-level relationaware transformer model for occluded person re-identification. Neural Networks,
1870:reducing style bias. In Proceedings of the IEEE/CVF conference on computer vision
1891:Grad-cam: Visual explanations from deep networks via gradient-based localization.
1902:Tan, W., Ding, C., Wang, P., Gong, M., & Jia, K. (2023). Style interleaved learning
1910:Jieru Jia: Writing – original draft, Methodology. Huidi Xie: Investigation. Qin Huang: Visualization. Yantao Song: Validation, Conceptualization. Peng Wu: Software.
1923:Chang, T., Yang, X., Luo, X., Ji, W., & Wang, M. (2023). Learning style-invariant robust
1924:representation for generalizable visual instance retrieval. In Proceedings of the 31st
1953:Huang, W., Chen, C., Li, Y., Li, J., Li, C., Song, F., et al. (2023). Style projected
1980:Yang, F., Zhong, Z., Luo, Z., He, Y., Li, S., & Sebe, N. (2024). Diversity-authenticity
1998:Zhao, Y., Zhong, Z., Zhao, N., Sebe, N., & Lee, G. H. (2022). Style-hallucinated dual
2015:Zhou, K., Yang, Y., Qiao, Y., & Xiang, T. (2020). Domain generalization with MixStyle.

exec
/bin/zsh -lc "pdftotext -f 1 -l 15 'Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf' - | rg -n -i 'ablation|DCR|ATG|TGA|ACN|attribute|anti-forgetting|knowledge consolidation|visual|t-SNE|Fig. 3|Fig. 4|Table|component|effectiveness'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 1146ms:
27:learning (DCR) model that explores global and attribute-wise representations as a bridge to balance intra-domain discrimination
29:the complementary relationship between global and attributewise representations to improve discrimination among similar
31:lead to catastrophic forgetting. We further develop an attributeoriented anti-forgetting (AF) strategy that explores attribute-wise
33:a knowledge consolidation (KC) strategy to facilitate knowledge
34:transfer. Extensive experiments show that our DCR achieves superior performance compared to state-of-the-art LReID methods.
35:Our code is available at https://github.com/LiuShiBen/DCR.
48:attribute category
85:Index Terms—Lifelong person re-identification, attribute-text
102:LReID model’s anti-forgetting and generalization capabilities.
113:Attribute-text
118:(TGA)
122:exhibit notable
160:PFManti-forgetting of old
163:of Robotics, Shenyang Institute Attribute
214:Thus, we explore global and attribute-wise representations
217:the anti-forgetting and generalization capabilities of the
220:representation learning (DCR) model that first explores
221:attribute and text information to enhance LReID performance.
223:representations including global and attribute-wise
225:an attribute-text generator (ATG) to dynamically generate
227:a text-guided aggregation (TGA) network to enhance the
229:identities in LReID. Furthermore, the attributes of each
230:instance guide an attribute compensation (ACN) network to
231:generate attribute-wise representations that focus on specific
233:attributes can enhance reliability by setting higher thresholds
234:across domains. Therefore, the generated attribute-wise
240:attribute-wise representations. At the intra-domain level,
242:attribute-wise representations focus on specific regional
243:information. When whole-body appearances or attributerelated information are similar across identities, we combine
244:global and attribute-wise representations to distinguish among
248:knowledge, we develop an attribute-oriented anti-forgetting
249:(AF) strategy that explores attribute-wise representations
253:Knowledge consolidation (KC) is proposed to facilitate
264:learning (DCR) model that explores global and attributewise representations to capture individual nuances in
267:• In the intra-domain context, we explore the complementary relationship between global and attribute-wise
270:• In the inter-domain context, we design an attributeoriented anti-forgetting (AF) and a knowledge consolidation (KC) strategy to minimize inter-domain gaps
272:model’s generalization and anti-forgetting capabilities.
279:gaps in continuously updated datasets across scenarios, improving the model’s anti-forgetting and generalization capabilities. LReID methods can be divided into two categories.
294:widespread popularity in recent years. Contrastive LanguageImage Pre-training (CLIP) [25], establishes a connection between natural language and visual content through the similarity constraint of image-text pairs. CLIP has been applied
299:Attribute-text generator
301:Text-guided aggregation network (TGA)
343:Attribute
366:Attribute
369:Attribute
372:Attribute Decoder
430:Attribute-wise
433:Attribute matching
441:Attribute compensation network (ACN)
475:Attribute-oriented anti-forgetting
498:Knowledge consolidation
502:Fig. 2. Overview of the proposed DCR for LReID. First, the attribute-text generator (ATG) dynamically generates text-image pairs for each instance. Then, the
503:text-guided aggregation network (TGA) captures global representations for each identity, while the attribute compensation network (ACN) generates attributewise representations. We explore the complementary relationship between global and attribute-wise representations to maximize intra-domain discrimination.
504:Meanwhile, we design attribute-oriented anti-forgetting (AF) and knowledge consolidation (KC) strategies to minimize inter-domain gaps and facilitate
509:to generate robust visual features or integrate the beneficial
512:DCR
522:C. Pedestrian Attribute Recognition
523:Pedestrian attribute recognition aims to assign a set of
524:attributes (Gender, Bag, Short/Long sleeve, and etc.) to a
525:visual representation of a pedestrian based on their attributes.
532:improve attribute recognition. Part-based methods [37], [38]
536:of attribute recognition. WeDKP
539:application of attributes to LReID from two perspectives.
540:1) Attributes are converted into text descriptions for each
542:attributes are transformed into attribute-wise representations
548:The overview of our DCR to achieve a trade-off between
550:gaps is shown in Fig. 2. The DCR model learns the old
553:parts of attribute-text generator (ATG), text-guided aggregation
557:TABLE I
558:T HE ATTRIBUTE CATEGORIES ARE DIVIDED INTO FOUR MAJOR
626:network (TGA), and attribute compensation network (ACN).
638:Where G and AG are global and attribute-wise representations, respectively.
639:B. Attribute-Text Generator
641:propose an attribute-text generator (ATG) to dynamically generate text-image pairs for each instance, as illustrated in Fig.
642:2. Firstly, we employ a pre-trained attribute recognition model
643:[33] (trained on the PA100K dataset [39]) to generate attribute
646:with 12 subcategories, as shown in TABLE I. Then, attribute
649:pairs are fed into the text-guided aggregation network (TGA)
654:the corresponding predicted attributes. To ensure robustness
656:attribute classification, thereby maintaining the reliability of
659:To enhance robust representation capabilities for each instance, we propose a text-guided aggregation network (TGA)
662:information in LReID, as shown in Fig. 2 (TGA). The TGA
665:encoder is frozen in our DCR model.
721:LgTri [5] to optimize our DCR at the current task.
739:generated by the text-guided aggregation (TGA) network
750:E. Attribute-oriented Anti-Forgetting
752:D. Attribute Compensation Network
753:We force attributes to guide the attribute compensation
754:network (ACN) for learning attribute-wise representations. The
755:ACN consists of an attribute decoder and an attribute matching
756:component, as illustrated in Fig. 2 (ACN).
757:1) Attribute Decoder: Enabling attributes to better adapt
758:across domains, we define multiple learnable attributes semantic information S t = {Sit |i = 1, 2, · · · , N } to learn
759:discriminative information. The attributes undergo a linear
761:the text-image global representation to output fAT . Attribute
763:are input into the attribute decoder, which outputs the attribute
764:features At = {Ati |i = 1, 2, · · · , N }. The attribute decoder
766:2) Attribute Matching: The attribute features At = {Ati |i =
769:make these attribute features more prominent in the global
770:representations. We propose an Attribute Matching (AM)
771:mechanism to select the optimal alignment between attribute
774:representational ability of the attribute features. Specifically,
775:we first calculate the similarity between the attribute features
786:attribute feature Ai |i = 1, 2, · · · , N and the global representations G, and then combine the attribute feature Ai and
788:the matched index) to obtain the attribute-wise representations
794:We leverage the triplet loss to align attribute-wise representations with identity at the current step, assisting in global
804:and negative samples to anchor samples in attribute-wise
805:representations, respectively. In this paper, attribute-wise representations that contain specific information of individuals
807:attribute-wise representations as a bridge across increasing
809:We develop an attribute-oriented anti-forgetting (AF) strategy to explore attribute-wise representations that align the
814:leverage attribute-wise representations as a bridge to optimize
832:F. Knowledge Consolidation
835:knowledge consolidation (KC) strategy that leverages global
856:the old and new models, further improving the model’s knowledge consolidation ability. This is represented as follows:
866:The knowledge consolidation loss is defined as:
880:TABLE II
901:attribute category
963:Text-guided aggregation network (TGA)
1032:anti-forgetting
1045:Attribute
1046:Attribute-wise
1047:Attribute matching
1057:compensation network (ACN)
1089:Attribute-oriented
1090:anti-forgetting for these
1095:can be shown in TABLE
1099:attribute decoder utilizes a transformer-based architecture [40].
1105:oposed DCR task. We employ mean average
1179:We compare the proposed DCR with SOTA LReID to
1183:results on training order-1 and order-2 are shown in TABLE
1184:III and TABLE IV, respectively.
1186:TABLE III and TABLE IV, our DCR significantly outperforms
1188:mAP/7.8% R-1, and 9.8% mAP/7.5% R-1 on training order1 and order-2, respectively. Meanwhile, our DCR effectively
1191:Fig. 3. Anti-forgetting curves. After each training step, we measure the
1193:anti-forgetting performance.
1207:Compared to CODA, our DCR significantly outperforms
1215:DCR achieves
1217:anti-forgetting and
1220:In Table III, compared to the KRKC method, our DCR model
1223:when evaluated against the LSTKC methods, our DCR demonstrates superior performance over LReID methods, delivering
1225:R-1, as shown in Table IV. We attribute these advancements to
1226:the complementary relationship between global and attributewise representations, which effectively enhances discrimination among similar identities. Additionally, our attribute(b) Ours
1229:anti-forgetting
1230:(AF) and knowledge consolidation
1232:seamless knowledge transfer. In summary, our DCR markedly
1243:that includes the CLIP model, an attribute-text generator, and
1244:a knowledge consolidation strategy. The Baseline outperforms
1247:of CLIP, as presented in TABLE III and TABLE IV. Compared
1248:to the Baseline, our DCR improves the Seen-Avg by 11.4%
1253:4) The Anti-forgetting Performance of Our Method: We
1255:order-1, as shown in Fig. 3. The Fig. 3 shows the metric
1264:TABLE III
1488:TABLE IV
1715:5) The effectiveness of minimizing inter-domain gaps: We
1716:visualize the feature distribution of PTKP, KRKC, Baseline,
1717:and our method across five datasets as shown in Fig. 4. The
1719:gaps, as the lack of attribute-wise representations makes it
1722:the model’s ability to prevent forgetting and enhance generalization. Compared to other methods, our DCR effectively
1724:identity information across multiple domains, which significantly improves the anti-forgetting and generalization ability
1726:6) The effectiveness of maximizing intra-domain discrimination: We visualize the feature distribution of KRKC and our
1727:method. Fig. 5 shows that our DCR can significantly cluster
1730:arrow). Compared to KRKC, our DCR improves intra-domain
1733:global and attribute-wise representations, which enables it to
1738:other methods, our DCR achieves superior performance and
1740:Thus, our attribute-oriented anti-forgetting (AF) strategy
1742:generalization ability of our model. In summary, our DCR
1743:explores global and attribute-wise representations to achieve
1747:C. Ablation Studies
1748:1) The number of global and attribute-wise representations:
1749:Global and attribute-wise representations capture individual
1751:evaluate the suitability of multiple global and attribute-wise
1811:4. t-SNE visualization
1815:domain. Our DCR effectively improves the anti-forgetting and generalization
1840:TABLE VI
1841:A BLATION STUDIES OF DIFFERENT COMPONENTS ON TRAINING ORDER -1.
1871:Components
1872:ACN
1926:Fig. 5. Visualization of intra-domain discrimination on the Market1501
1939:Our DCR can cluster images of the same identity
1942:2) Performance of Different Components: To evaluate
1943:the contribution of each component to our DCR model, we
1944:conduct ablation studies on both seen and unseen domains, as
1945:detailed in Table VI. Here, PFM denotes the Parallel Fusion
1952:Module, while ACN refers to the Attribute Compensation
1953:(b) Oursthe attribute-oriented
1956:anti-forgetting and knowledge consolidation strategies,
1967:the baseline method. This demonstrates its effectiveness
1970:PFM and ACN yields significant gains: +6.5% mAP /
1977:TABLE V
1979:A BLATION STUDIES ON THE NUMBER OF GLOBAL AND ATTRIBUTE - WISE
1982:knowledge transfer. Our DRE method unifies PFM, ACN,
2009:in our DCR.
2011:representations as shown in TABLE V. We have observed that
2012:setting the number of global and attribute-wise representations
2015:3) Performance of attribute-text generator: To better understand whether each instance’s text descriptions generated
2016:by the attribute-text generator (ATG) provide more fine-
2024:Proposed DCR
2026:Fig. 7. Visualization of retrieval results. The left column displays the query image under challenging conditions, such as occlusion, blur, side view, and back
2028:the DKP method, our DCR retrieves more pedestrians with the same identity, demonstrating superior performance.
2030:TABLE VII
2031:A BLATION OF TRAINING WITH OR WITHOUT ATTRIBUTE - TEXT
2032:GENERATOR (ATG) ON TRAINING ORDER -1.
2034:Training w/o ATG
2035:Training w/ ATG
2057:TABLE VIII
2090:person” (w/o ATG) for comparison. TABLE VII shows that
2091:the attribute-text generator obtains text descriptions to significantly improve overall performance. When using the specific
2094:the unseen domain. ATG enhances the robustness of global
2097:4) Analysis of confidence threshold: To evaluate the sensitivity of the confidence threshold in an attribute-text generation, we conduct an ablation study with threshold values
2098:of 0.7, 0.8, and 0.9, as shown in TABLE VIII. The results
2100:to 0.8 consistently enhances DCR performance across metrics.
2108:D. Visualization
2109:To further validate the effectiveness of our DCR, we conduct
2116:our DCR accurately retrieves eight out of ten individuals
2117:with the same identity, despite the limited visual information
2119:only correctly matches three pedestrians, whereas our DCR
2122:(fourth and fifth rows), DCR successfully handles clothing
2124:identity. These results demonstrate that DCR achieves superior retrieval performance, consistently identifying more
2127:of global and attribute-wise representations, which balances
2129:to complex scenarios, our DCR demonstrates significantly
2134:TABLE IX
2170:for inference. As shown in Table IX, where time measurements
2172:our DCR completes training in 19 hours and 10 minutes
2175:our DCR shows longer training and inference times compared
2179:R-1, and 9.8% mAP/7.5% R-1 on training order-1 and order2, as demonstrated in Tables III and IV. The increased
2180:computational complexity of DCR stems from its transformerbased CLIP architecture incorporating multiple class tokens
2181:and the additional attribute compensation network (ACN) that
2190:In this paper, we propose a domain consistency representation learning (DCR) model that explores global and
2191:attribute-wise representations to capture subtle nuances in
2192:intra-domain and inter-domain consistency, achieving a tradeoff between maximizing intra-domain discrimination and minimizing inter-domain gaps. Specifically, global and attributewise representations serve as complementary information to
2194:develop an attribute-oriented anti-forgetting (AF) strategy and
2195:a knowledge consolidation (KC) strategy to minimize interdomain gaps and facilitate knowledge transfer, enhancing
2277:visual models from natural language supervision,” in Proc. Int. Conf.
2298:attribute recognition: A reliable evaluation under zero-shot pedestrian
2304:pedestrian detection, tracking and attribute recognition,” Rob. Auton.
2307:multi-task network for pedestrian attribute recognition,” IEEE Trans.
2310:regularizations for pedestrian attribute recognition,” in Proc. Int. Conf.
2313:attribute representations for robust pedestrian attribute recognition,” in
2413:engaged in research and development at Stuttgart

codex
DCR 的证据链包含两个层面：跨域一致性用跨数据集 t-SNE，域内判别用身份聚类和检索可视化。DMDL 的检索结果在后页，我继续抽它的消融和可视化，不只看方法公式。
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
/bin/zsh -lc "pdftotext -f 15 -l 22 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | sed -n '1,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
shortcuts, thereby constructing a low-biased model. Building upon CAI, the CBT further mitigates bias propagation during the optimization process. The modality-specific
data augmentation disrupts modality cues at the data level, label refinement corrects biased pseudo-labels at the label level, and feature alignment enforces modality-invariant
representations at the feature level. These components collaboratively prevent biased
information from being amplified through iterative training. Importantly, CAI and CBT
play complementary roles. CAI suppresses modality bias at the modeling level by reshaping the learning objective, while CBT prevents residual bias from being propagated
during optimization. By jointly considering causal intervention and training dynamics,
DMDL formulates modality debiasing as an end-to-end learning problem, enabling
robust and stable bias suppression throughout the learning pipeline.

4. Experiments
4.1. Datasets and Evaluation Protocol
Dataset. In this section, we conduct comprehensive experiments to evaluate the proposed method on two widely used datasets, SYSU-MM01 [36] and RegDB [37], as
well as a more recent dataset, LLCM [38].
The SYSU-MM01 dataset with 4 visible cameras and 2 infrared cameras, capturing
395 identities for training and 96 for testing. The test query set comprises 3,803 infrared images, and the gallery set contains 301 visible images. The evaluation protocol
provides all-search and indoor-search modes.
The RegDB is a dual-camera dataset with 412 identities, each having 10 visible and
10 infrared images. It is split into 206 identities for training and 206 for testing. The
evaluation protocol includes two test modes: visible to infrared and infrared to visible.
The LLCM is the largest VI-ReID dataset that captures images with 9 cameras. It
contains 1,064 identities, of which 713 are used for training and 351 for testing. The
evaluation protocol includes two test modes: VIS to IR and IR to VIS.
Evaluation protocol. All experiments follow the standard evaluation protocol in the
VI-ReID benchmark testing. Our model is evaluated using different training/testing
splits in ten trials to ensure stable performance. Evaluation metrics include cumulative
15

matching characteristics (CMC), mean average precision (mAP), and mean inverse
negative penalty (mINP) [39].
4.2. Implementation Details
We employ ResNet-50 pre-trained on ImageNet as the backbone network and integrate Non-local Attention Blocks [39] and generalized-mean (GeM) pooling [39]. All
input images are resized to 288 × 144, and standard data augmentation techniques,
including horizontal flipping, random cropping, and random erasing, are applied. At
the beginning of each epoch, DBSCAN [40] clustering is performed independently
for each modality to generate pseudo labels. The clustering threshold and the minimum number of images are set to 0.6 and 4 on SYSU-MM01 [36] and LLCM [38],
and to 0.3 and 4 on RegDB [37], respectively. During training, 16 pseudo-identities
are sampled from each modality, with 16 instances per pseudo-identity (8 original and
8 augmented). The model is optimized using Adam with an initial learning rate of
3.5 × 10−4 and a weight decay of 5 × 10−4 . The learning rate is decreased by a factor of
ten every 20 epochs. The hyperparameter σ is set to 0.05. Training proceeds for a total
of 100 epochs, with the first 50 epochs dedicated to single-modality learning, followed
by 50 epochs of cross-modality training.
4.3. Comparison with State-of-the-art Methods
To validate the effectiveness of our DMDL, we compare it with state-of-the-art
methods under three relevant settings: supervised VI-ReID, semi-supervised VI-ReID,
and unsupervised VI-ReID. The experimental results for the SYSU-MM01 and RegDB
datasets are shown in Table 1, and the experimental results for the LLCM dataset are
presented in Table 2.
Comparison with supervised VI-ReID Methods. Encouragingly, our DMDL achieves
competitive performance compared to the supervised method FMCNet [42] on the
SYSU-MM01 and RegDB datasets, and even surpasses several supervised methods,
including AGW [39] and SPOT [41]. Moreover, on the challenging LLCM dataset, our
DMDL still demonstrates impressive performance, outperforming several supervised
methods (e.g., AGW [39] and LbA [50]). However, due to the absence of annotated
16

SYSU-MM01
Methods

Venue

All Search
r1

mAP

RegDB

Indoor Search
mINP

r1

mAP

mINP

Visible to Infrared

Infrared to Visible

r1

mAP

mINP

r1

mAP

mINP

Supervised VI-ReID methods
AGW [39]

TPAMI-21

47.50

47.65

35.30

54.17

62.97

59.23

70.05

66.37

50.19

70.49

65.90

51.24

CA [1]

ICCV-21

69.88

66.89

53.61

76.26

80.37

76.79

85.03

79.14

65.33

84.75

77.82

61.56

SPOT [41]

TIP-22

65.34

62.25

-

69.42

74.63

-

80.35

72.46

-

79.37

72.26

-

FMCNet [42]

CVPR-22

66.34

62.51

-

68.15

74.09

-

89.12

84.43

-

88.38

83.86

-

MUN [43]

ICCV-23

76.24

73.81

-

79.42

82.06

-

95.19

87.15

-

91.86

85.01

-

IDKL [2]

CVPR-24

81.42

79.85

-

87.14

89.37

-

94.72

90.19

-

94.22

90.43

-

TSKD [44]

PR-25

76.6

73.0

-

82.7

85.3

-

91.1

81.7

-

89.9

80.5

-

OTLA [45]

ECCV-22

48.2

43.9

-

47.4

56.8

-

49.9

41.8

-

49.6

42.8

-

DPIS [29]

ICCV-23

58.4

55.6

-

63.0

70.0

-

62.3

53.2

-

61.5

52.7

-

CGSFL [46]

PR-25

59.83

53.12

35.79

61.50

63.83

60.66

89.36

84.17

69.47

89.11

81.49

66.43

ADCA [3]

MM-22

45.51

42.73

28.29

50.60

59.11

55.17

67.20

64.05

52.67

68.48

63.81

49.62

DOTLA [14]

MM-23

50.36

47.36

32.40

53.47

61.73

57.35

85.63

76.71

61.58

82.91

74.97

58.60

MBCCM [5]

MM-23

53.14

48.16

32.41

55.21

61.98

57.13

83.79

77.87

65.04

82.82

76.74

61.73

PGM [4]

CVPR-23

57.27

51.78

34.96

56.23

62.74

58.13

69.48

65.41

52.97

69.85

65.17

-

CHCR [20]

TCSVT-23

59.47

59.14

-

-

-

-

69.31

64.74

-

69.96

65.87

-

GUR∗ [19]

ICCV-23

63.51


exec
/bin/zsh -lc "pdftotext -f 8 -l 15 'DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf' - | sed -n '360,740p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
sampling strategy and weights computation approach in Table 9. Concretely, we investigate three popular sampling strategies, i.e. random
sampling, cross-domain sampling, and farthest point sampling (FPS).
As for the combination weights, we alter it by sampling from Beta
distribution with 𝛼 = 0.1 or Dirichlet distribution with the concentration parameters set to 1 ∕𝐶 , where C denotes the number of channels.
Note that 𝐷𝐸𝐷 is removed in this section by setting 𝜔 = 0. It can
be seen that although FPS can cover the rare styles to the utmost
extent in theory, it is marginally superior over random sampling in
our practice while requiring extra memory cost to store style basis.
Hence, we suggest that random sampling is a better choice. Besides,
DSH consistently outperforms Beta and Dir sampling in all settings,
justifying this adaptive manner to deduce combination weights.

5. Conclusion
In this paper, we propose a novel framework DiverseReID for generalized person re-ID. With a combination of Dynamic Style Hallucination and Domain Experts Decoupling, intra-domain and inter-domain
9

Neural Networks 189 (2025) 107602

J. Jia et al.

style diversity among the source data is greatly enhanced respectively,
promoting generalization and transferability of deep re-ID. Experiments demonstrate that our approach achieves state-of-the-art performance under various settings and protocols. Albeit simple, the proposed
method is capable of boosting the DG re-ID performance significantly
and can be readily integrated with other DG methods in a plug-and-play
manner.

Jiao, B., Liu, L., Gao, L., Lin, G., Yang, L., Zhang, S., et al. (2022). Dynamically transformed instance normalization network for generalizable person re-identification.
In European conference on computer vision (pp. 285–301).
Jin, X., Lan, C., Zeng, W., Chen, Z., & Zhang, L. (2020). Style normalization and
restitution for generalizable person re-identification. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition (pp. 3143–3152).
Kim, M., Kim, S., Park, J., Park, S., & Sohn, K. (2023). Partmix: Regularization
strategy to learn part discovery for visible-infrared person re-identification. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition
(pp. 18621–18632).
Kingma, D. P. (2015). Adam: A method for stochastic optimization. In International
conference on learning representations.
Li, X., Dai, Y., Ge, Y., Liu, J., Shan, Y., & Duan, L.-Y. (2022). Uncertainty modeling for out-of-distribution generalization. In International conference on learning
representations.
Li, Y., Song, J., Ni, H., & Shen, H. T. (2023). Style-controllable generalized person reidentification. In Proceedings of the 31st ACM international conference on multimedia
(pp. 7912–7921).
Li, W., & Wang, X. (2013). Locally aligned feature transforms across views. In
Proceedings of the IEEE conference on computer vision and pattern recognition (pp.
3594–3601).
Li, B., Wu, F., Lim, S.-N., Belongie, S., & Weinberger, K. Q. (2021). On feature
normalization and data augmentation. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition (pp. 12383–12392).
Li, W., Zhao, R., Xiao, T., & Wang, X. (2014). Deepreid: Deep filter pairing neural
network for person re-identification. In Proceedings of the IEEE conference on
computer vision and pattern recognition (pp. 152–159).
Liao, S., & Shao, L. (2020). Interpretable and generalizable person re-identification
with query-adaptive convolution and temporal lifting. In Proceedings of the European
conference on computer vision (pp. 456–474).
Liao, S., & Shao, L. (2022). Graph sampling based deep metric learning for generalizable
person re-identification. In Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition (pp. 7359–7368).
Lin, G., Bao, Z., Huang, Z., Li, Z., Zheng, W., & Chen, Y. (2024). A multi-level relationaware transformer model for occluded person re-identification. Neural Networks,
177, Article 106382.
Lin, S., Li, C.-T., & Kot, A. C. (2020). Multi-domain adversarial feature generalization
for person re-identification. IEEE Transactions on Image Processing, 30, 1596–1607.
Loy, C. C., Xiang, T., & Gong, S. (2009). Multi-camera activity correlation analysis. In
IEEE conference on computer vision and pattern recognition (pp. 1988–1995). IEEE.
Nam, H., Lee, H., Park, J., Yoon, W., & Yoo, D. (2021). Reducing domain gap by
reducing style bias. In Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition (pp. 8690–8699).
Ni, H., Song, J., Luo, X., Zheng, F., Li, W., & Shen, H. T. (2022). Meta distribution
alignment for generalizable person re-identification. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition (pp. 2487–2496).
Nie, R., Ding, J., Zhou, X., & Li, X. (2024). Rethinking normalization layers for domain
generalizable person re-identification. In European conference on computer vision (pp.
267–284).
Nuriel, O., Benaim, S., & Wolf, L. (2021). Permuted adain: Reducing the bias towards
global statistics in image classification. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition (pp. 9482–9491).
Pan, X., Luo, P., Shi, J., & Tang, X. (2018). Two at once: Enhancing learning and
generalization capacities via ibn-net. In Proceedings of the European conference on
computer vision (pp. 464–479).
Qi, L., Wang, L., Shi, Y., & Geng, X. (2022). A novel mix-normalization method for
generalizable multi-source person re-identification. IEEE Transactions on Multimedia,
25, 4856–4867.
Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). Pointnet++: Deep hierarchical feature
learning on point sets in a metric space. Advances in Neural Information Processing
Systems, 30.
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).
Grad-cam: Visual explanations from deep networks via gradient-based localization.
In Proceedings of the IEEE international conference on computer vision (pp. 618–626).
Shankar, S., Piratla, V., Chakrabarti, S., Chaudhuri, S., Jyothi, P., & Sarawagi, S. (2018).
Generalizing across domains via cross-gradient training. In International conference
on learning representations.
Shu, Y., Cao, Z., Wang, C., Wang, J., & Long, M. (2021). Open domain generalization
with domain-augmented meta-learning. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition (pp. 9624–9633).
Sun, Y., Liu, Y., Liu, X., Li, Y., & Chu, W.-S. (2023). Rethinking domain generalization
for face anti-spoofing: Separability and alignment. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition (pp. 24563–24574).
Tan, W., Ding, C., Wang, P., Gong, M., & Jia, K. (2023). Style interleaved learning
for generalizable person re-identification. IEEE Transactions on Multimedia, 26,
1600–1612.
Wang, Y., Liao, S., & Shao, L. (2020). Surpassing real-world source training data:
Random 3d characters for generalizable person re-identification. In Proceedings of
the 28th ACM international conference on multimedia (pp. 3422–3430).

CRediT authorship contribution statement
Jieru Jia: Writing – original draft, Methodology. Huidi Xie: Investigation. Qin Huang: Visualization. Yantao Song: Validation, Conceptualization. Peng Wu: Software.
Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to
influence the work reported in this paper.
Acknowledgments
This research was supported by the National Natural Science Foundation of China (62106133), the Fundamental Research Program of
Shanxi Province (202403021211226), Central Guidance for Local Scientific and Technological Development Funds (No. YDZJSX20231B001),
and the Research Project Supported by the Shanxi Scholarship Council
of China (No. 2024-020,2023-015).
References
Bai, H., Sun, R., Hong, L., Zhou, F., Ye, N., Ye, H.-J., et al. (2021). Decaug: Outof-distribution generalization via decomposed feature representation and semantic
augmentation. In Proceedings of the AAAI conference on artificial intelligence (pp.
6705–6713).
Chang, T., Yang, X., Luo, X., Ji, W., & Wang, M. (2023). Learning style-invariant robust
representation for generalizable visual instance retrieval. In Proceedings of the 31st
ACM international conference on multimedia (pp. 6171–6180).
Chen, T., Ding, S., Xie, J., Yuan, Y., Chen, W., Yang, Y., et al. (2019). Abd-net: Attentive
but diverse person re-identification. In Proceedings of the IEEE/CVF international
conference on computer vision (pp. 8351–8361).
Chen, Z., Pan, Y., Ye, Y., Cui, H., & Xia, Y. (2023). Treasure in distribution: a domain
randomization based multi-source domain generalization for 2d medical image
segmentation. vol. 14223, In International conference on medical image computing
and computer-assisted intervention (pp. 89–99).
Choi, S., Kim, T., Jeong, M., Park, H., & Kim, C. (2021). Meta batch-instance normalization for generalizable person re-identification. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition (pp. 3425–3435).
Dai, Y., Li, X., Liu, J., Tong, Z., & Duan, L.-Y. (2021). Generalizable person reidentification with relevance-aware mixture of experts. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition (pp. 16145–16154).
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). Imagenet: A largescale hierarchical image database. In IEEE conference on computer vision and pattern
recognition (pp. 248–255). IEEE.
Dou, Z., Wang, Z., Li, Y., & Wang, S. (2023). Identity-seeking self-supervised representation learning for generalizable person re-identification. In Proceedings of the
IEEE/CVF international conference on computer vision (pp. 15847–15858).
Ge, Y., Zhu, F., Chen, D., Zhao, R., et al. (2020). Self-paced contrastive learning with
hybrid memory for domain adaptive object re-id. In Advances in neural information
processing systems (pp. 11309–11321).
Gray, D., & Tao, H. (2008). Viewpoint invariant pedestrian recognition with an
ensemble of localized features. In Proceedings of the European conference on computer
vision (pp. 262–275).
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
recognition. In Proceedings of the IEEE conference on computer vision and pattern
recognition (pp. 770–778).
Hirzer, M., Beleznai, C., Roth, P. M., & Bischof, H. (2011). Person re-identification by
descriptive and discriminative classification. In Image analysis: 17th scandinavian
conference (pp. 91–102).
Huang, W., Chen, C., Li, Y., Li, J., Li, C., Song, F., et al. (2023). Style projected
clustering for domain generalized semantic segmentation. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition (pp. 3061–3071).
Jia, J., Ruan, Q., & Hospedales, T. M. (2019). Frustratingly easy person re-identification:
Generalizing person re-id in practice. In British machine vision conference.
10

Neural Networks 189 (2025) 107602

J. Jia et al.
Wang, W., Liao, S., Zhao, F., Kang, C., & Shao, L. (2021). DomainMix: Learning
generalizable person re-identification without human annotations. In British machine
vision conference.
Wang, P., Zhang, Z., Lei, Z., & Zhang, L. (2023). Sharpness-aware gradient matching
for domain generalization. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition (pp. 3769–3778).
Wei, L., Zhang, S., Gao, W., & Tian, Q. (2018). Person transfer gan to bridge domain
gap for person re-identification. In Proceedings of the IEEE conference on computer
vision and pattern recognition (pp. 79–88).
Xiao, T., Li, S., Wang, B., Lin, L., & Wang, X. (2017). Joint detection and identification
feature learning for person search. In Proceedings of the IEEE conference on computer
vision and pattern recognition (pp. 3415–3424).
Xu, B., Liang, J., He, L., & Sun, Z. (2022). Mimic embedding via adaptive aggregation:
Learning generalizable person re-identification. In European conference on computer
vision (pp. 372–388).
Yan, Y., Li, J., Liao, S., & Qin, J. (2025). TAL: two-stream adaptive learning for
generalizable person re-identification. Machine Intelligence Research, 1–15.
Yang, F., Zhong, Z., Luo, Z., He, Y., Li, S., & Sebe, N. (2024). Diversity-authenticity
co-constrained stylization for federated domain generalization in person reidentification. In Proceedings of the AAAI conference on artificial intelligence (pp.
6477–6485).
Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical
risk minimization. In International conference on learning representations.
Zhang, P., Dou, H., Yu, Y., & Li, X. (2022). Adaptive cross-domain learning for
generalizable person re-identification. In European conference on computer vision (pp.
215–232).
Zhang, M., Li, K., Ma, J., & Wang, X. (2024). Asymmetric double networks mutual
teaching for unsupervised person re-identification. Neural Networks, 169, 744–755.

Zhang, X., Li, D., Wang, Z., Wang, J., Ding, E., Shi, J. Q., et al. (2022). Implicit sample
extension for unsupervised person re-identification. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition (pp. 7369–7378).
Zhao, Y., Zhong, Z., Yang, F., Luo, Z., Lin, Y., Li, S., et al. (2021). Learning to
generalize unseen domains via memory-based multi-source meta-learning for person
re-identification. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition (pp. 6277–6286).
Zhao, Y., Zhong, Z., Zhao, N., Sebe, N., & Lee, G. H. (2022). Style-hallucinated dual
consistency learning for domain generalized semantic segmentation. In European
conference on computer vision (pp. 535–552).
Zheng, W.-S., Gong, S., & Xiang, T. (2009). Associating groups of people.. vol. 2, In
British machine vision conference (pp. 1–11). 6.
Zheng, L., Shen, L., Tian, L., Wang, S., Wang, J., & Tian, Q. (2015). Scalable person
re-identification: A benchmark. In Proceedings of the IEEE international conference on
computer vision (pp. 1116–1124).
Zheng, Z., Zheng, L., & Yang, Y. (2017). Unlabeled samples generated by gan improve
the person re-identification baseline in vitro. In Proceedings of the IEEE international
conference on computer vision (pp. 3754–3762).
Zhou, K., Yang, Y., Cavallaro, A., & Xiang, T. (2021). Learning generalisable omni-scale
representations for person re-identification. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 44(9), 5056–5069.
Zhou, K., Yang, Y., Hospedales, T., & Xiang, T. (2020). Learning to generate novel
domains for domain generalization. In Proceedings of the European conference on
computer vision (pp. 561–578).
Zhou, K., Yang, Y., Qiao, Y., & Xiang, T. (2020). Domain generalization with MixStyle.
In International conference on learning representations.
Zhuang, Z., Wei, L., Xie, L., Zhang, T., Zhang, H., Wu, H., et al. (2020). Rethinking the
distribution gap of person re-identification with camera-based batch normalization.
In Proceedings of the European conference on computer vision (pp. 140–157).

11


exec
/bin/zsh -lc "pdftotext -f 10 -l 17 'Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf' - | sed -n '700,1180p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
codex
DMDL 第一次抽取落在结果表开头，消融还在后面。DCR 刚才页码范围没覆盖到对应行，我按全文行号重新抽。
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
/bin/zsh -lc "pdftotext -f 1 -l 24 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | sed -n '1800,2250p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 24 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | rg -n -i '4\\.[4-9]|5\\.|Ablation|CAI|CBT|label refinement|feature alignment|augmentation|visual|t-SNE|modality gap|parameter|Table 3|Table 4|Table 5|Figure 5|Figure 6|Figure 7'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
SYSU-MM01 (all-search) and 90.63% on RegDB (visible-to-infrared), respectively.
These results surpass those of existing methods, including the recent state-of-the-art
approaches MCL [18] and ASM [17], demonstrating the strong effectiveness of our

18

Components

SYSU-MM01
CBT

Index

Baseline

All Search

Indoor Search

CAI
data

1

✓

2

✓

✓

3

✓

✓

✓

4

✓

✓

✓

5

✓

✓

✓

6

✓

7

✓

✓

label

feature

r1

r5

r10

mAP

mINP

r1

r5

r10

mAP

mINP

56.26

84.12

92.19

54.60

40.75

63.98

87.77

94.06

69.68

65.30

59.61

85.34

92.94

57.85

44.31

67.07

89.67

95.02

72.94

69.00

62.17

87.01

93.82

58.76

45.59

68.07

89.92

94.51

73.08

69.25

63.11

87.67

94.66

59.61

45.34

68.34

90.22

95.29

73.06

69.09

✓

64.09

88.07

94.65

61.19

47.41

68.56

90.36

95.27

73.91

70.08

✓

✓

✓

✓

63.39

87.88

93.85

59.78

45.08

67.26

90.31

95.52

72.77

68.44

✓

✓

✓

65.90

88.56

94.71

61.86

47.53

70.66

91.18

95.78

75.45

71.66

Table 3: Ablation studies on the SYSU-MM01. Rank at r accuracy (%), mAP (%) and mINP (%) are
reported.

method. Meanwhile, on the more challenging LLCM dataset, our method showcases
remarkable performance, surpassing the state-of-the-art RoDE [48] by 14.23% mAP
and 12.52% rank-1 accuracy in the IR to VIS setting, as reported in Table 2.
These results suggest that while existing methods have made notable progress on
the USL-VI-ReID task, they still suffer from modality bias, leading to the extraction
of modality-dependent features. In contrast, our DMDL effectively learns modalityinvariant representations by addressing modality bias through causal modeling and
bias-free training optimization, achieving more robust performance.
4.4. Ablation Study
To evaluate the contribution of each component in DMDL, we conduct ablation
experiments on the SYSU-MM01 dataset, as summarized in Table 3. Note that channel
augmentation (CA) [1] for visible images is incorporated into the baseline to ensure a
fair assessment of our designed components.
Effectiveness of CAI. When CAI is applied by replacing traditional likelihood-based
modeling in baseline with causal modeling, the rank-1 accuracy and mAP of all-search
increase by 3.35% and 3.25%, respectively (Index 1 vs. 2). This demonstrates that CAI
effectively constructs a low-biased model by explicitly modeling causal relationships
between images and labels, enhancing the model’s robustness to modality variation.
Discussion of removing CAI. We conduct the experiments of removing CAI and only
using CBT with baseline (Index 6). Compared with the full DMDL (CAI+CBT), using

19

68
66

78

65.90

62

75.45

76
63.90

64
61.64

74

63.42

62.19

72

61.86
59.71

60

60.33

58.60

70.66

68

72.86

71.87

69.97

70

59.67

74.64
72.16

67.72
66.45

66.39

66

58

64

56

62
60

54
r1
CBT

r1

mAP

random aug

w/o exchange

w/o 𝜂𝑥

CBT

ℒ𝑐𝑓𝑎

mAP
random aug

w/o exchange

w/o 𝜂𝑥

ℒ𝑐𝑓𝑎

(b) Indoor search

(a) All search

Figure 5: Detailed analysis of CBT on the SYSU-MM01 dataset under (a) all-search and (b) indoor-search
modes. Rank-1 accuracy (%) and mAP (%) are reported.

CBT alone degrades the rank-1 accuracy by approximately 2–3%, suggesting that CAI
and CBT are most effective when used jointly. CAI constructs a low-biased model,
while CBT suppresses the injection of biased cues into the model during optimization.
They act on the complementary modeling and optimization stages, and reinforce each
other in modality debiasing.
Effectiveness of CBT. Compared with the results in Index 2, the experiments in Index
7 show that the proposed optimization strategy, CBT, achieves a 6.29% improvement in
rank-1 accuracy (all-search), confirming its effectiveness in bias-free feature learning.
Specifically, CBT comprises three components: data augmentation, label refinement,
and feature alignment. When integrated sequentially, these components yield consistent performance gains (Index 3–7). This trend indicates that each component mitigates
modality bias at different levels, contributes to learn modality-invariant representations,
and produces a mutually reinforcing effect. Below, we present a detailed analysis of
these components in CBT.
Effectiveness of data augmentation in CBT. To verify the effectiveness of modalityspecific augmentation, we replace it with standard random augmentation schemes (e.g.,
random cropping) within the CBT strategy. As shown in the second column “random
aug” of Fig. 5, this replacement results in a drop of about 4% in rank-1 accuracy.
This drop indicates that modality-specific augmentation plays a crucial role in CBT, as
it explicitly disrupts modality-specific information in the images and thus guides the
model to learn modality-shared representations. Moreover, when only the modality-

20

specific augmentation is applied, the model still achieves an overall performance improvement (Index 3 vs. 4), further demonstrating that this augmentation effectively
mitigates modality bias at the data level.
Effectiveness of label refinement in CBT. Based on the data augmentation, the label
refinement scheme introduces two key designs: (1) exchanging predictions between an
image and its augmentation, and (2) dynamically updating the memory with adaptive
η x . To evaluate the effectiveness of these designs, we conduct ablation experiments
under the “w/o exchange” and “w/o η x ” settings, the results of which are shown in the
third and fourth columns of Fig. 5. Specifically, the “w/o exchange” variant refines the
pseudo-label using its own prediction rather than that of its augmentation, while the
“w/o η x ” variant fixes η = 0.05 for memory updating. Both variants lead to performance degradation, confirming that: (1) exchanging predictions between images and
their modality-specific augmentations mitigates modality bias in refined labels, as the
augmentation perturbs modality-specific cues, yielding less biased predictions for refining labels, and (2) dynamic updating enhances memory reliability, enabling stable
predictions. By integrating these two designs, CBT effectively mitigates modality bias
at the label level, resulting in a substantial performance improvement (Index 3 vs. 4).
Effectiveness of feature alignment in CBT. We observe a notable performance gain
after incorporating the feature alignment loss L f a into CBT (Index 3 vs. 5), indicating that L f a effectively alleviates modality bias in feature representations by aligning
images and their augmentation. To further validate the effectiveness of our design, we
replace L f a with an MMD-based loss Lc f a commonly adopted in conventional VIReID methods, which enforces direct alignment between visible and infrared feature
distributions to reduce modality gap:
Lc f a = ∥

n
n
1 X  V 1 X  I 2
ϕ fi −
ϕ fi ∥H .
n i=1
n i=1

(16)

As shown in the fifth column of Fig. 5, Lc f a results in degraded performance, since
inconsistent cross-modality pseudo-labels lead to identity misalignment and weaken
feature discriminability. In contrast, L f a leverages the natural correspondence between
original and augmented images, aligning modalities in a label-consistent and feature-

21

67

67

65.9

65.13

65
63

62.29

61
59

58.57

65

 succeeded in 0ms:
15:model level, we propose a Causality-inspired Adjustment Intervention (CAI) module
17:optimization level, a Collaborative Bias-free Training (CBT) strategy is introduced to
18:interrupt the propagation of modality bias across data, labels, and features by integrating modality-specific augmentation, label refinement, and feature alignment. Extensive
76:To this end, a Causality-inspired Adjustment Intervention (CAI) module and a Collaborative Bias-free Training (CBT) strategy are proposed. Specifically, CAI facilitates
78:likelihood method, the causal modeling in CAI is theoretically unaffected by modality bias, thereby achieving a low-biased model. To further prevent biased knowledge
79:from deepening during optimization, we propose the CBT strategy, integrating data
80:augmentation, label refinement, and feature alignment. CBT first introduces a pseudomodality augmentation scheme to modify modality-specific cues in images. Based on
81:the augmented images, a cross-modality label smoothing scheme and a feature alignment loss are proposed to refine the biased relationships and learn shared knowledge
82:across pseudo-modalities, respectively. By jointly leveraging these components, CBT
92:(2) We propose a Causality-inspired Adjustment Intervention (CAI) module at the
95:(3) We propose a Collaborative Bias-free Training (CBT) strategy at the optimization
96:level, combining label refinement and feature alignment with modality-specific
97:data augmentation to prevent fitting biased knowledge.
117:through cross-modality feature mapping, constructing a pseudo cross-modality identity space to facilitate effective feature alignment. Despite their effectiveness, these
153:Fig. 2, incorporating the Causality-inspired Adjustment Intervention (CAI) module
154:and the Collaborative Bias-free Training (CBT) strategy. In cross-modality learning,
155:DMDL first iteratively matches clusters across different modalities to obtain crossmodality relationships as a kind of pseudo-label. Then, CAI employs a backdoor adjustment algorithm to implement causal intervention, which guides the model to capture causal patterns, resulting in a low-biased model. Furthermore, to avoid misleading
156:optimization caused by biased cues, CBT incorporates label refinement and feature
157:alignment with modality-specific data augmentation to jointly mitigate modality bias
251:strategy combines label refinement and modality alignment with data augmentation to optimize the model,
272:representation, and σ is a temperature hyper-parameter. The single-modality model is
316:the causal modeling in the proposed CAI module, which constructs a cross-modality
375:This provides a principled mechanism for modality debiasing in unsupervised crossmodality learning and motivates our implementation of intervention in CAI to prevent
378:Based on the above analysis, an intervention loss Lcai is constructed by maximizing
380:Lcai = E x,y [− log P(Y = y|do(X = x))],
384:where x denotes an input image, and y represents its associated cross-modality pseudolabel. To achieve that, CAI implements the computation of P(Y|do(X) by backdoor
415:We provide further analysis of CAI. Compared to the likelihood model P(Y|X)
430:Although a low-biased model is obtained through CAI, the biased modality-specific
432:problem, we propose the CBT strategy to mitigate modality bias at the optimization
434:features, CBT integrates label refinement and feature alignment with well-designed
435:data augmentation, thereby disrupting bias propagation and promoting unbiased feature
437:3.4.1. Data Augmentation in CBT
438:CBT first introduces a modality-specific augmentation scheme to destroy modalityrelated information in images, as shown in Fig. 4. Specifically, for infrared images, we
441:to increase diversity and introduce randomness to the augmentation by randomly sampling R, G, and B channels of multiple generated pseudo-color images and combining
443:channel augmentation (CA) [1] through channel multiplexing to generate augmented
445:This modality-specific data augmentation enables the image and its corresponding augmentation to share the same identity-discriminative information but differ in
447:assistance of such augmentation, CBT implements label refinement and feature alignment to facilitate bias-free learning.
498:Figure 4: Illustration of the modality-specific augmentation. Circles represent channels of images. Subscript
503:3.4.2. Label Refinement in CBT
504:To refine the noise pseudo-labels, CBT employs label smoothing by exchanging
526:yai represent the refined soft labels of image xi and its augmentation
527:xia , respectively. Then, they are used to supervise model training by modifying the Lcai
531:Lcai = Ei [ −
604:certainty, the proposed label refinement directly constructs low-biased cross-modality
607:labels by incorporating modality-specific augmentations and dynamically updating the
608:modality-shared memory. Since an image and its modality-specific augmentation contain different modality-related information, exchanging their predictions for label smoothing effectively reduces label noise induced by modality-specific cues. Furthermore,
611:3.4.3. Feature Alignment in CBT
612:In addition, a feature alignment loss is introduced to further enhance bias-free feature learning. It is well understood that identity-discriminative information should remain consistent under augmentation. Therefore, for an image and its modality-specific
613:augmentation, the model is expected to extract similar features; otherwise, it suggests
637:and its augmentation representations to be close in the metric space, thereby mitigating
639:3.5. Total Loss of DMDL
642:+ λcai · Lcai + λ f a · L f a + λtri · Ltri ,
647:where λcai , λ f a and λtri are weights of the corresponding loss term.
649:modeling level, the CAI module performs causal intervention via backdoor adjustment,
653:shortcuts, thereby constructing a low-biased model. Building upon CAI, the CBT further mitigates bias propagation during the optimization process. The modality-specific
654:data augmentation disrupts modality cues at the data level, label refinement corrects biased pseudo-labels at the label level, and feature alignment enforces modality-invariant
656:information from being amplified through iterative training. Importantly, CAI and CBT
657:play complementary roles. CAI suppresses modality bias at the modeling level by reshaping the learning objective, while CBT prevents residual bias from being propagated
684:input images are resized to 288 × 144, and standard data augmentation techniques,
692:ten every 20 epochs. The hyperparameter σ is set to 0.05. Training proceeds for a total
755:35.30
771:65.90
791:85.03
795:65.33
797:84.75
807:65.34
815:74.63
849:84.43
875:95.19
883:85.01
903:94.72
927:85.3
977:55.6
1007:35.79
1031:45.51
1041:55.17
1071:85.63
1079:74.97
1093:55.21
1103:65.04
1119:34.96
1129:65.41
1135:65.17
1157:64.74
1163:65.87
1189:75.00
1205:64.4
1217:85.8
1227:64.4
1235:74.4
1255:64.49
1273:85.76
1283:64.77
1289:65.34
1305:65.38
1343:45.01
1401:64.53
1413:85.78
1423:64.44
1427:45.19
1479:65.07
1507:65.90
1515:75.45
1521:85.33
1527:85.04
1537:65.42
1541:74.81
1607:55.3
1623:55.6
1663:54.9
1671:65.8
1683:65.9
1732:34.77
1768:35.13
1778:45.25
1786:55.35
1797:setting. Specifically, DMDL attains 65.42% mAP on SYSU-MM01 (all-search) and
1798:85.33% mAP on RegDB (visible-to-infrared), outperforming the method SALCR [49]
1799:by 4.98% and 1.46% mAP on the respective datasets. Notably, even without utilizing camera information, our approach achieves rank-1 accuracies of 65.90% on
1809:CBT
1819:CAI
1896:54.60
1908:65.30
1912:85.34
1924:95.02
1938:45.59
1944:94.51
1954:94.66
1958:45.34
1964:95.29
1976:94.65
1986:95.27
2008:45.08
2014:95.52
2026:65.90
2030:94.71
2040:95.78
2042:75.45
2046:Table 3: Ablation studies on the SYSU-MM01. Rank at r accuracy (%), mAP (%) and mINP (%) are
2056:4.4. Ablation Study
2057:To evaluate the contribution of each component in DMDL, we conduct ablation
2058:experiments on the SYSU-MM01 dataset, as summarized in Table 3. Note that channel
2059:augmentation (CA) [1] for visible images is incorporated into the baseline to ensure a
2061:Effectiveness of CAI. When CAI is applied by replacing traditional likelihood-based
2063:increase by 3.35% and 3.25%, respectively (Index 1 vs. 2). This demonstrates that CAI
2066:Discussion of removing CAI. We conduct the experiments of removing CAI and only
2067:using CBT with baseline (Index 6). Compared with the full DMDL (CAI+CBT), using
2076:65.90
2080:75.45
2119:74.64
2140:CBT
2152:CBT
2169:Figure 5: Detailed analysis of CBT on the SYSU-MM01 dataset under (a) all-search and (b) indoor-search
2172:CBT alone degrades the rank-1 accuracy by approximately 2–3%, suggesting that CAI
2173:and CBT are most effective when used jointly. CAI constructs a low-biased model,
2174:while CBT suppresses the injection of biased cues into the model during optimization.
2177:Effectiveness of CBT. Compared with the results in Index 2, the experiments in Index
2178:7 show that the proposed optimization strategy, CBT, achieves a 6.29% improvement in
2180:Specifically, CBT comprises three components: data augmentation, label refinement,
2181:and feature alignment. When integrated sequentially, these components yield consistent performance gains (Index 3–7). This trend indicates that each component mitigates
2184:these components in CBT.
2185:Effectiveness of data augmentation in CBT. To verify the effectiveness of modalityspecific augmentation, we replace it with standard random augmentation schemes (e.g.,
2186:random cropping) within the CBT strategy. As shown in the second column “random
2188:This drop indicates that modality-specific augmentation plays a crucial role in CBT, as
2194:specific augmentation is applied, the model still achieves an overall performance improvement (Index 3 vs. 4), further demonstrating that this augmentation effectively
2196:Effectiveness of label refinement in CBT. Based on the data augmentation, the label
2198:image and its augmentation, and (2) dynamically updating the memory with adaptive
2199:η x . To evaluate the effectiveness of these designs, we conduct ablation experiments
2201:third and fourth columns of Fig. 5. Specifically, the “w/o exchange” variant refines the
2202:pseudo-label using its own prediction rather than that of its augmentation, while the
2204:their modality-specific augmentations mitigates modality bias in refined labels, as the
2205:augmentation perturbs modality-specific cues, yielding less biased predictions for refining labels, and (2) dynamic updating enhances memory reliability, enabling stable
2206:predictions. By integrating these two designs, CBT effectively mitigates modality bias
2208:Effectiveness of feature alignment in CBT. We observe a notable performance gain
2209:after incorporating the feature alignment loss L f a into CBT (Index 3 vs. 5), indicating that L f a effectively alleviates modality bias in feature representations by aligning
2210:images and their augmentation. To further validate the effectiveness of our design, we
2212:distributions to reduce modality gap:
2236:65.9
2238:65.13
2252:64.42
2256:65.9
2309:Figure 6: Parameter analysis of λcai and λ f a on the SYSU-MM01 dataset (all-search).
2334:55.13
2360:65.90
2368:75.45
2372:Table 4: The comparison of different matching strategies on SYSU-MM01. Rank1 accuracy (%), mAP (%)
2375:discriminative manner. Furthermore, combining augmentation-based label refinement
2376:and feature alignment achieves the best result (Index 7), indicating that CBT effectively
2379:4.5. Further Analysis
2380:Parameter Analysis. The proposed DMDL introduces two key parameters, λcai and
2381:λ f a in Eq. 15, which serve as weighting factors to balance Lcai and L f a during training.
2382:Fig. 6 (a) illustrates the impact of varying λcai on rank-1 and mAP accuracy on the
2383:SYSU-MM01 dataset (all-search). When λcai = 0, the CAI is disabled, resulting in
2385:at λcai = 0.5, and this value is therefore adopted in our experiments. Fig. 6 (b) shows
2399:Figure 7: The t-SNE (first row) and similarity distribution (second row) visualization of 20 randomly selected
2400:identities on the SYSU-MM01 dataset. In t-SNE visualization, the circle and the cross represent the visible
2403:Visualization Analysis. Fig. 7 presents the t-SNE [54] plots and the cosine similarity
2407:purple circle are correctly aligned after applying DMDL. Taken together, these visualizations demonstrate that DMDL effectively narrows the modality gap and improves
2412:Score, where higher scores indicate better label quality. Notably, incorporating CBT
2418:and CAI on top of the baseline consistently improves the quality of cross-modality labels across training. This demonstrates that CBT and CAI effectively mitigate modality
2422:we compare it with two commonly adopted matching strategies, bipartite graph matching (BGM) and optimal transport (OT), under identical experimental settings, as reported in Table 4. Although both BGM and OT are capable of establishing crossmodality correspondences, their performance is consistently inferior to that of iMCA.

exec
/bin/zsh -lc "pdftotext -f 1 -l 18 'Domain Consistency Representation Learning for Lifelong Person Re-Identification.pdf' - | sed -n '1715,2130p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 966ms:
5) The effectiveness of minimizing inter-domain gaps: We
visualize the feature distribution of PTKP, KRKC, Baseline,
and our method across five datasets as shown in Fig. 4. The
Baseline shows poor performance in bridging inter-domain
gaps, as the lack of attribute-wise representations makes it
challenging to reduce inter-domain gaps. The KRKC method
effectively separates each domain, but it insufficiently distinguishes identity information within the domain, limiting
the model’s ability to prevent forgetting and enhance generalization. Compared to other methods, our DCR effectively
distinguishes identity information within a domain and spreads
identity information across multiple domains, which significantly improves the anti-forgetting and generalization ability
of the model.
6) The effectiveness of maximizing intra-domain discrimination: We visualize the feature distribution of KRKC and our
method. Fig. 5 shows that our DCR can significantly cluster
images of the same identity more tightly (circle) and increase
the distance between different identities (black bidirectional
arrow). Compared to KRKC, our DCR improves intra-domain

discrimination due to the complementary relationship between
global and attribute-wise representations, which enables it to
learn the subtle nuances of individuals.
7) Generalization Curves on the Unseen Domain: We
analyze the average performance on the unseen domain
during the training steps, as depicted in Fig. 6. Compared to
other methods, our DCR achieves superior performance and
exhibits faster performance growth across the training steps.
Thus, our attribute-oriented anti-forgetting (AF) strategy
effectively bridges inter-domain gaps and enhances the
generalization ability of our model. In summary, our DCR
explores global and attribute-wise representations to achieve
a trade-off between maximizing intra-domain discrimination
and minimizing inter-domain gaps.

C. Ablation Studies
1) The number of global and attribute-wise representations:
Global and attribute-wise representations capture individual
nuances in intra-domain and inter-domain consistency. We
evaluate the suitability of multiple global and attribute-wise

8
KRKC

LSTKC

DKP

CUHK-SYSU

Market1501

M)
FM)

Market1501

Market1501

CUHK-SYSU

CUHK-SYSU

Ours

DukeMTMC

DukeMTMC

MSMT17_V2

MSMT17_V2

DukeMTMC

MSMT17_V2

CUHK03

CUHK03

CUHK03

orm
(PFM)

pule (PFM)

ayerNorm
Drop

(a) PTKP

(b) KRKC

(d) Ours

(c) Baseline

4. t-SNE visualization
of feature distribution
five datasets from the (d)
seenOurs
domain. Our DCR effectively improves the anti-forgetting and generalization
(a) Fig.
PTKP
(b) KRKC
(c) on
Baseline
ability,
which significantly
distinguishes(c)identity
information (d)
within
a domain but also spreads identity information across multiple domains.
(a) PTKP
(b) KRKC
Ours
Baseline

Q

CA

K,V

Q

TABLE VI
A BLATION STUDIES OF DIFFERENT COMPONENTS ON TRAINING ORDER -1.

bedding

gConcatenation
tenation

Type

LLD
Maximize
inter-domain
discrimination

(a) KRKC

(b) Ours

(a)
(b)
(c)
(d)
(e)

PFM
√
√
√
√

Components
ACN
AF

√
√
√

√
√

KC

√

Seen Avg
mAP
R-1

Unseen Avg
mAP
R-1

50.4
51.7
56.9
58.7
61.8

51.8
52.5
57.6
58.5
60.8

61.5
62.1
68.3
69.2
71.9

49.4
50.3
55.4
56.8
58.3

n

LLD

aximize
r-domain
imination

Fig. 5. Visualization of intra-domain discrimination on the Market1501
AKA
PTKP 30 PatchKD
dataset. We randomly
select
identities.KRKC
Colors DKP
representOurs
different identity
(b)more
Ours tightly
(a) KRKC
information.
Our DCR can cluster images of the same identity
(circle) for minimizing inter-domian discrimination.

2) Performance of Different Components: To evaluate
the contribution of each component to our DCR model, we
conduct ablation studies on both seen and unseen domains, as
detailed in Table VI. Here, PFM denotes the Parallel Fusion
Ours
AKA
PTKP
PatchKD
KRKC
DKP
Module, while ACN refers to the Attribute Compensation
(b) Oursthe attribute-oriented
(a) KRKC
Network. AF and KC represent
anti-forgetting and knowledge consolidation strategies,
respectively. Comparing (a) and (b), we observe that PFM
improves performance by +1.3% mAP / +0.6% R-1 on
seen avg and
+0.7% mAP
Ourson unseen avg under
AKA
PTKP
PatchKD
KRKC
DKP/ +0.9% R-1
the baseline method. This demonstrates its effectiveness
in fusing text and image information to generate diverse
global representations. Comparing (a) and (c), integrating
PFM and ACN yields significant gains: +6.5% mAP /
+6.8% R-1 on seen avg and +5.8% mAP / +6.0% R-1 on
Fig. 6. Generalization curves. After each training step, the performance of
unseen avg, for generating the diverse global representations.
the unseen domain is evaluated.
Comparing (c) and (e), when integrating AF and KC,
the model achieves gains of +4.9% mAP / +3.6% R-1 on
TABLE V
seen avg and +3.2% mAP / +2.9% R-1 on unseen avg. These
A BLATION STUDIES ON THE NUMBER OF GLOBAL AND ATTRIBUTE - WISE
strategies significantly promote inter-domain consistency and
REPRESENTATIONS N ON TRAINING ORDER -1.
knowledge transfer. Our DRE method unifies PFM, ACN,
Seen Avg
Unseen Avg
AF, and KC into an end-to-end LReID model, striking a
Number (N )
mAP
R-1
mAP
R-1
balance between maximizing intra-domain discrimination and
2
60.2
68.7
59.4
56.5
minimizing inter-domain gaps. These combinations ensure
3
61.8
71.9
60.8
58.3
strong generalization while mitigating catastrophic forgetting
4
61.2
71.6
60.3
57.5
in our DCR.

representations as shown in TABLE V. We have observed that
setting the number of global and attribute-wise representations
N to 3 achieves the best performance for our method.

3) Performance of attribute-text generator: To better understand whether each instance’s text descriptions generated
by the attribute-text generator (ATG) provide more fine-

9

Query

DKP Method

Proposed DCR

Fig. 7. Visualization of retrieval results. The left column displays the query image under challenging conditions, such as occlusion, blur, side view, and back
view. In the retrieved results, correctly matched pedestrians are marked with green boxes, while incorrect matches are highlighted in red box. Compared to
the DKP method, our DCR retrieves more pedestrians with the same identity, demonstrating superior performance.

TABLE VII
A BLATION OF TRAINING WITH OR WITHOUT ATTRIBUTE - TEXT
GENERATOR (ATG) ON TRAINING ORDER -1.
Method
Training w/o ATG
Training w/ ATG

Seen Avg
mAP
R-1

Unseen Avg
mAP
R-1

60.1
61.8

59.3
60.8

70.5
71.9

56.5
58.3

TABLE VIII
A BLATION STUDIES OF CONFIDENCE THRESHOLD ON TRAINING ORDER -1.
Confidence Threshold
0.7
0.8
0.9

Seen Avg
mAP
R-1

Unseen Avg
mAP
R-1

60.9
61.8
62.0

59.7
60.8
60.5

71.1
71.9
71.6

57.6
58.3
58.0

grained guidance for learning global representations, we train
our model using the generic text descriptor ”A photo of a
person” (w/o ATG) for comparison. TABLE VII shows that
the attribute-text generator obtains text descriptions to significantly improve overall performance. When using the specific
text descriptors, the average decreases by 1.7% mAP/1.4%
R-1 on the seen domain and by 1.5% mAP/1.8% R-1 on
the unseen domain. ATG enhances the robustness of global
representations for each instance, effectively mitigating the
forgetting of old knowledge.
4) Analysis of confidence threshold: To evaluate the sensitivity of the confidence threshold in an attribute-text generation, we conduct an ablation study with threshold values
of 0.7, 0.8, and 0.9, as shown in TABLE VIII. The results
demonstrate that increasing the confidence threshold from 0.7
to 0.8 consistently enhances DCR performance across metrics.

While a confidence threshold of 0.9 achieves the highest 62.0%
mAP on seen avg, other metrics do not reach their peak
performance at this value. Overall, we find that setting the
confidence threshold to 0.8 strikes the best balance, ensuring
robust classification accuracy while maximizing model performance.

D. Visualization
To further validate the effectiveness of our DCR, we conduct
a qualitative comparison with the DKP method, as shown in
Fig. 7. The left column displays challenging query images,
including cases with occlusion, blur, side views, and back
views. In Fig. 7, correctly matched pedestrians are marked
with green boxes, while incorrect matches are highlighted
in red. Under occlusion conditions (first and second rows),
our DCR accurately retrieves eight out of ten individuals
with the same identity, despite the limited visual information
available. For blurry queries (third row), the DKP method
only correctly matches three pedestrians, whereas our DCR
achieves perfect retrieval (10/10) even with the unclear query
image. Additionally, in side-view and back-view scenarios
(fourth and fifth rows), DCR successfully handles clothing
color variations, outperforming DKP in matching the same
identity. These results demonstrate that DCR achieves superior retrieval performance, consistently identifying more
pedestrians with the same identity across diverse challenging
conditions. This advantage stems from the collaborative design
of global and attribute-wise representations, which balances
intra-domain discrimination and inter-domain gaps. By enhancing discrimination between similar identities and adapting
to complex scenarios, our DCR demonstrates significantly
more robustness than existing approaches.

codex
DMDL 的消融已经定位到：CAI、CBT、模态特异增广、标签交换、动态记忆、特征对齐、参数敏感和 t-SNE。还需要把后面的可视化与匹配策略对照读完。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 21 -l 24 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
specific augmentation is applied, the model still achieves an overall performance improvement (Index 3 vs. 4), further demonstrating that this augmentation effectively
mitigates modality bias at the data level.
Effectiveness of label refinement in CBT. Based on the data augmentation, the label
refinement scheme introduces two key designs: (1) exchanging predictions between an
image and its augmentation, and (2) dynamically updating the memory with adaptive
η x . To evaluate the effectiveness of these designs, we conduct ablation experiments
under the “w/o exchange” and “w/o η x ” settings, the results of which are shown in the
third and fourth columns of Fig. 5. Specifically, the “w/o exchange” variant refines the
pseudo-label using its own prediction rather than that of its augmentation, while the
“w/o η x ” variant fixes η = 0.05 for memory updating. Both variants lead to performance degradation, confirming that: (1) exchanging predictions between images and
their modality-specific augmentations mitigates modality bias in refined labels, as the
augmentation perturbs modality-specific cues, yielding less biased predictions for refining labels, and (2) dynamic updating enhances memory reliability, enabling stable
predictions. By integrating these two designs, CBT effectively mitigates modality bias
at the label level, resulting in a substantial performance improvement (Index 3 vs. 4).
Effectiveness of feature alignment in CBT. We observe a notable performance gain
after incorporating the feature alignment loss L f a into CBT (Index 3 vs. 5), indicating that L f a effectively alleviates modality bias in feature representations by aligning
images and their augmentation. To further validate the effectiveness of our design, we
replace L f a with an MMD-based loss Lc f a commonly adopted in conventional VIReID methods, which enforces direct alignment between visible and infrared feature
distributions to reduce modality gap:
Lc f a = ∥

n
n
1 X  V 1 X  I 2
ϕ fi −
ϕ fi ∥H .
n i=1
n i=1

(16)

As shown in the fifth column of Fig. 5, Lc f a results in degraded performance, since
inconsistent cross-modality pseudo-labels lead to identity misalignment and weaken
feature discriminability. In contrast, L f a leverages the natural correspondence between
original and augmented images, aligning modalities in a label-consistent and feature-

21

67

67

65.9

65.13

65
63

62.29

61
59

58.57

65

64.42

62.87

65.9

64.06

63.92

63.63

63
61.86

61.39

61

60.32

59.58

59

57

r1

mAP

0.75

1

57
55

55
0

0.25

0.5

0.5

1

(a) 𝜆𝑐𝑎𝑖

2.5

5

(b) 𝜆𝑓𝑎

Figure 6: Parameter analysis of λcai and λ f a on the SYSU-MM01 dataset (all-search).
SYSU-MM01

All Search

Indoor Search

methods

r1

mAP

mINP

r1

mAP

mINP

OT [14]

60.24

55.13

38.24

64.13

68.64

62.43

BGM [4]

63.74

58.87

42.88

68.57

73.06

68.48

iMCA(ours)

65.90

61.86

47.53

70.66

75.45

71.66

Table 4: The comparison of different matching strategies on SYSU-MM01. Rank1 accuracy (%), mAP (%)
and mINP (%) are reported.

discriminative manner. Furthermore, combining augmentation-based label refinement
and feature alignment achieves the best result (Index 7), indicating that CBT effectively
promotes bias-free feature learning by interrupting the propagation of modality bias
across data, labels, and features.
4.5. Further Analysis
Parameter Analysis. The proposed DMDL introduces two key parameters, λcai and
λ f a in Eq. 15, which serve as weighting factors to balance Lcai and L f a during training.
Fig. 6 (a) illustrates the impact of varying λcai on rank-1 and mAP accuracy on the
SYSU-MM01 dataset (all-search). When λcai = 0, the CAI is disabled, resulting in
poor performance, which confirms its effectiveness. The best performance is observed
at λcai = 0.5, and this value is therefore adopted in our experiments. Fig. 6 (b) shows
the rank-1 accuracy results for different λ f a values, and the model achieves the highest
accuracy at λ f a = 1, so we empirically set it to 1. For completeness, the sensitivity
analysis of λtri used in baseline is provided in the supplementary material, as it is not
our main contribution.
22

wrong
right

(a) Baseline

(b) DMDL

Figure 7: The t-SNE (first row) and similarity distribution (second row) visualization of 20 randomly selected
identities on the SYSU-MM01 dataset. In t-SNE visualization, the circle and the cross represent the visible
and infrared modalities, respectively.

Visualization Analysis. Fig. 7 presents the t-SNE [54] plots and the cosine similarity
distribution of positive and negative cross-modality pairs for randomly selected identities. Compared with the baseline, DMDL exhibits a more compact alignment between
visible and infrared samples, together with a larger separation between positive and
negative cross-modality pairs. In addition, the mismatched samples highlighted by the
purple circle are correctly aligned after applying DMDL. Taken together, these visualizations demonstrate that DMDL effectively narrows the modality gap and improves
the robustness of the learned representation against modality bias.
Cross-modality Pseudo-label Quality Analysis. We assess the quality of cross-modality
pseudo-labels generated at different training epochs on the SYSU-MM01 dataset in
Fig. 8, using two standard metrics from [55]: Homogeneity Score and Adjusted Rand
Score, where higher scores indicate better label quality. Notably, incorporating CBT

23

Figure 8: Cross-modality pseudo-label quality analysis over different epochs on the SYSU-MM01 dataset.

and CAI on top of the baseline consistently improves the quality of cross-modality labels across training. This demonstrates that CBT and CAI effectively mitigate modality
bias in the learned representations, thereby facilitating more accurate cross-modality
matching.
Effectiveness of iMCA. To evaluate the robustness of our matching strategy, iMCA,
we compare it with two commonly adopted matching strategies, bipartite graph matching (BGM) and optimal transport (OT), under identical experimental settings, as reported in Table 4. Although both BGM and OT are capable of establishing crossmodality correspondences, their performance is consistently inferior to that of iMCA.
Specifically, OT assigns samples to cross-modality clusters under an implicit uniform
assignment assumption, while BGM enforces a strict global one-to-one cluster matching. These strong assumptions make both methods more prone to erroneous assignments, especially in the presence of noisy or ambiguous cross-modality similarities. In
contrast, our iMCA performs conservative cross-modality alignment through a natural
maximum-confidence matching mechanism without any assumptions, leading to more
stable and reliable correspondences.
Retrieval Results. We qualitatively compare our DMDL with the baseline by visu-

24


codex
DMDL 的证据闭环完整了：它不仅有模块消融，还用伪标签质量曲线证明“偏置确实在标签里传播”，这比单纯涨点更接近它的核心论证。最后我会整合为逐篇方法论拆解。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 24 -l 25 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | sed -n '1,220p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Figure 8: Cross-modality pseudo-label quality analysis over different epochs on the SYSU-MM01 dataset.

and CAI on top of the baseline consistently improves the quality of cross-modality labels across training. This demonstrates that CBT and CAI effectively mitigate modality
bias in the learned representations, thereby facilitating more accurate cross-modality
matching.
Effectiveness of iMCA. To evaluate the robustness of our matching strategy, iMCA,
we compare it with two commonly adopted matching strategies, bipartite graph matching (BGM) and optimal transport (OT), under identical experimental settings, as reported in Table 4. Although both BGM and OT are capable of establishing crossmodality correspondences, their performance is consistently inferior to that of iMCA.
Specifically, OT assigns samples to cross-modality clusters under an implicit uniform
assignment assumption, while BGM enforces a strict global one-to-one cluster matching. These strong assumptions make both methods more prone to erroneous assignments, especially in the presence of noisy or ambiguous cross-modality similarities. In
contrast, our iMCA performs conservative cross-modality alignment through a natural
maximum-confidence matching mechanism without any assumptions, leading to more
stable and reliable correspondences.
Retrieval Results. We qualitatively compare our DMDL with the baseline by visu-

24

Query

Query

Ranking

Baseline

Ranking

Query

DMDL

Query

Ranking

Baseline

Ranking

DMDL

(a) SYSU-MM01

(b) RegDB

Figure 9: Visualization of the retrieval results obtained by the baseline and our DMDL on the SYSU-MM01
and RegDB datasets. The green boxes represent correct retrieval results, and the red boxes represent incorrect
retrieval results.
Query

Ranking

All Search

Query

Ranking

Query

IR to VIS

VIS to IR

(a) SYSU-MM01

Ranking

(b) LLCM

Figure 10: Visualization of representative failure examples on the SYSU-MM01 and LLCM datasets.

alizing the retrieval results of several query images on SYSU-MM01 and RegDB, as
illustrated in Fig. 9. For each query, the retrieved samples highlighted with green boxes
indicate correct matches, while those marked in red correspond to incorrect matches.
Overall, the proposed method exhibits higher robustness to modality-specific interference (e.g., color cues), whereas the baseline tends to prioritize color similarity when
retrieving results (see the first row in Fig. 9). This confirms that our method achieves
stronger cross-modality retrieval capability than the baseline, yielding consistent improvements.
Failure-case Analysis. From the challenging examples shown in Fig. 10, we observe

25


hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 25 -l 26 'Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf' - | sed -n '1,180p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Query

Query

Ranking

Baseline

Ranking

Query

DMDL

Query

Ranking

Baseline

Ranking

DMDL

(a) SYSU-MM01

(b) RegDB

Figure 9: Visualization of the retrieval results obtained by the baseline and our DMDL on the SYSU-MM01
and RegDB datasets. The green boxes represent correct retrieval results, and the red boxes represent incorrect
retrieval results.
Query

Ranking

All Search

Query

Ranking

Query

IR to VIS

VIS to IR

(a) SYSU-MM01

Ranking

(b) LLCM

Figure 10: Visualization of representative failure examples on the SYSU-MM01 and LLCM datasets.

alizing the retrieval results of several query images on SYSU-MM01 and RegDB, as
illustrated in Fig. 9. For each query, the retrieved samples highlighted with green boxes
indicate correct matches, while those marked in red correspond to incorrect matches.
Overall, the proposed method exhibits higher robustness to modality-specific interference (e.g., color cues), whereas the baseline tends to prioritize color similarity when
retrieving results (see the first row in Fig. 9). This confirms that our method achieves
stronger cross-modality retrieval capability than the baseline, yielding consistent improvements.
Failure-case Analysis. From the challenging examples shown in Fig. 10, we observe

25

that performance degradation mainly occurs under extremely difficult conditions, such
as severe occlusion, low resolution, and heavy background clutter. Similar failure cases
are also observed in the USL-VI-ReID method RoDE [48], indicating that such performance degradation stems from the inherent challenges of unsupervised VI-ReID.
When identity-discriminative cues in the query modality are weak or partially missing,
models struggle to extract sufficiently informative representations, which consequently
leads to degraded matching accuracy. In future work, incorporating richer causal structures, such as explicitly modeling environment-related factors, may help alleviate these
limitations and further improve robustness under extreme conditions.
From the challenging examples shown in Fig. 10, we observe that performance
degradation mainly occurs under extremely difficult conditions, such as severe occlusion, low resolution, and heavy background clutter. These cases are largely attributed to
the inherent challenges of both supervised and unsupervised VI-ReID. When identitydiscriminative cues in the query modality are weak or partially missing, the model
struggles to extract sufficiently informative representations, which consequently degrades matching accuracy. In future work, incorporating richer causal structures, such
as explicitly modeling environment-related factors, may help alleviate these limitations
and further improve robustness under extreme conditions.

5. Conclusion
In this paper, we investigate the modality bias issue in unsupervised VI-ReID
and propose a novel Dual-level Modality Debiasing Learning (DMDL) framework
to tackle this issue from both model and optimization perspectives, incorporating a
Causality-inspired Adjustment Intervention (CAI) module and a Collaborative Biasfree Training (CBT) strategy. CAI models causal relationships between images and
pseudo-labels to capture stable, modality-independent patterns, thereby constructing
a low-biased model. Meanwhile, CBT performs label refinement and feature alignment with modality-specific data augmentation, jointly preventing the propagation of
modality bias and thus achieving bias-free optimization. Finally, with the above designs, DMDL effectively achieves modality-invariant feature learning. Extensive ex-

26


web search: 
web search: https://arxiv.org/abs/2405.19695
web search: 
web search: https://arxiv.org/abs/2409.19954
web search: 
web search: 10.1109/TNNLS.2024.3384023 DMPF
web search: 
web search: "DiverseReID" "Neural Networks" "107602"
codex
你列的是 6 篇，不是 10 篇。我按列出的 6 篇逐篇读了方法部分和实验消融。

**1. DMPF，Disentangling Modality and Posture Factors**

1. 触发观察：作者抓住的是可见光-红外 ReID 的类内差异不只是模态差异，还混着姿态差异。已有方法大多只在图像层或批次特征层对齐模态，忽略批次外的训练集整体差异，也忽略姿态变化。
2. 重定义动作：他们把问题从“跨模态对齐”改写成“身份表征里混入了两个身份无关因子，模态因子和姿态因子”。关键词是 `disentangling modality and posture factors`、`entire training set`、`identity-unrelated factors`。
3. 机制怎么长出来：模态因子既然跨整个训练集存在，就用类别记忆队列加 memory-attention，再用 OT 让同身份跨模态记忆靠近。姿态因子既然来自人体结构，就用 parsing 和 landmark 生成 skeleton，再用正交分解把 skeleton 里的姿态信息和身份信息拆开。机制和重定义绑定较紧，但模块偏重。
4. 证据闭环：关键不是总表涨点，而是 MFD、PFD 单独消融，batch-attention 对 memory-attention，OT loss 在 memory-attention 输出上收敛，parsing 与 landmark 组合对照，impure skeleton 对正交 posture feature 对照。t-SNE 还把模态间隔和姿态离群点分别可视化出来。
5. reviewer 为什么买账：卖的是“VI-ReID 不应只讲模态，还要同时解释姿态污染身份表征”的视角，机制只是顺着两个因子分别长出来。novelty 主要来自问题拆分和证据闭环，不是 attention 或 orthogonal 本身。

**2. DASA，Distribution Aligned Semantics Adaption**

1. 触发观察：直接用预训练权重初始化，在 LReID 中收益有限；旧方法靠 exemplar replay 和 logit distillation，但 ReID 是细粒度开放集，旧分类器和旧样本既不合适也不现实。
2. 重定义动作：他们把 LReID 从“旧知识怎么蒸馏”改成“预训练得到的人体语义如何在新域中低成本适配”。关键词是 `robust and general human semantics`、`shared knowledge`、`distribution alignment`、`semantics adaption`、`exemplar-free`。
3. 机制怎么长出来：如果 Conv 里保存通用人体语义，就冻结 Conv；如果新域分布会干扰语义使用，就调 domain-specific BN；如果通用语义还不够区分身份，就加轻量 SA 模块做空间聚合。这个机制和重定义非常紧，几乎没有多余零件。
4. 证据闭环：它的核心消融是 DA 单独有效、SA 单独有效、DASA 组合最好；核大小 1 会变差，3 或 5 变好，证明 SA 不是随便加卷积，而是需要空间语义聚合；不同预训练数据对照证明“人体预训练语义”确实是上限来源；存储对照证明不用 exemplar 也能长期演化。
5. reviewer 为什么买账：这篇主要卖范式，不是卖模块。它把“终身 ReID 必须蒸馏旧模型”改成“保存预训练人体语义，再按域适配”，这个视角比一个 SA 模块更值钱。

**3. DRE，Diverse Representations Embedding**

1. 触发观察：CNN 终身 ReID 方法为了多表示要加多分支，分支之间独立且难控制；固定 teacher 又会压制新任务适应。作者看到的是旧方法在实例级表征上太单薄。
2. 重定义动作：他们把 LReID 的遗忘问题重定义为“实例级表示不够丰富，导致旧知识保存和新知识适应都受限”。关键词是 `diverse representations`、`primary embedding`、`auxiliary embeddings`、`instance-level and task-level layout`。
3. 机制怎么长出来：Transformer 的多 class token 天然能产生多视角表示，所以引入多个 class token；主表示负责丰富身体信息，辅助表示负责判别性；ACM 对辅助表示做正交分离，再自适应并入主表示；任务级再用 adjustment model 和 learner model 做 KU/KP。
4. 证据闭环：关键证据是 ResNet 版本明显弱于 transformer 版本，CODA 加 ACM 优于 CODA，DRE-KU 缺 KP 会泛化不足，冻结 adjustment model 的 DRE-KUP 适应性不如完整 DRE。辅助表示数量消融也证明不是“越多越好”，两个辅助表示较合适。
5. reviewer 为什么买账：这篇卖的是机制多于视角。视角是“实例级多表示帮助终身学习”，但说法不如 DASA 和 DMDL锋利；真正支撑 novelty 的是把纯 transformer、多 class token、ACM、KU/KP 组合成 LReID 框架。

**4. DiverseReID**

1. 触发观察：DG ReID 里，现有风格增强方法用 Beta、Dirichlet 等固定分布采样混合权重，调参麻烦，而且生成风格可能仍贴近已有源域。作者抓的是“训练数据量不是关键，风格多样性才是关键”。
2. 重定义动作：他们把域泛化从“学域不变特征”改写成“最大化源域风格空间”。关键词是 `style diversity`、`intra-domain diversity`、`inter-domain diversity`、`dynamic style hallucination`、`decoupled domain experts`。
3. 机制怎么长出来：域内多样性不足，就用 DSH 根据两张图的 style Wasserstein distance 动态决定混合权重；域间多样性不足，就维护每个源域的 memory expert，用 DED 让不同域专家的特征去相关。机制很顺，且参数很少。
4. 证据闭环：它不只做 DSH 和 DED 消融，还把 DSH 与 MixStyle、pAdaIN、DSU、SagNet 的 style randomization、SHADE 等对比，并在去掉 DED 后单独比较风格生成方式。另有 DSH 插入位置、采样策略、权重计算方式、DED 权重敏感性和 Grad-CAM。
5. reviewer 为什么买账：这篇卖的是“多样性原则”，不是某个新层。它把两个简单机制分别钉在域内和域间多样性上，且用对比证明自适应风格混合比固定分布更贴合问题。

**5. DCR，Domain Consistency Representation Learning**

1. 触发观察：LReID 有一个矛盾，域内需要更细判别相似身份，域间又需要一致性防遗忘。旧蒸馏方法主要压域间 gap，容易牺牲当前域内判别。
2. 重定义动作：他们把“保存旧知识和学习新知识”改写成“最大化域内判别和最小化域间 gap 的权衡”。关键词是 `domain consistency representation`、`global and attribute-wise representations`、`bridge`、`intra-domain discrimination`、`inter-domain gaps`。
3. 机制怎么长出来：全局表征负责整个人，属性表征负责细粒度局部属性。ATG 先用属性识别器生成属性文本，TGA 用 CLIP 融合图文得到全局表征，ACN 得到属性级表征；AF 用属性表征对齐旧新模型，KC 用全局表征做知识转移。
4. 证据闭环：它用跨五个数据集的 t-SNE 证明域间一致性，用 Market 的身份聚类可视化证明域内判别增强。组件消融显示 PFM、ACN、AF、KC 分别贡献；ATG 用具体属性文本优于泛化的 “a photo of a person”；阈值 0.8 的消融说明属性可靠性被认真控制。
5. reviewer 为什么买账：这篇卖的是 trade-off 视角。CLIP、属性、蒸馏都不新，但“属性级表征作为域内判别和域间一致性的桥”把这些零件串成了一个可审稿的故事。

**6. DMDL，Dual-level Modality Debiasing Learning**

1. 触发观察：无监督 VI-ReID 的两阶段流程先做单模态学习，再做跨模态学习。作者指出第一阶段学到的模态特异线索会进入第二阶段伪标签，并在训练中继续放大。
2. 重定义动作：他们把问题从“伪标签不准”或“模态差异大”改写成“模态偏置从数据传播到标签，再传播到特征”。关键词是 `modality bias`、`dual-level debiasing`、`causal intervention`、`bias propagation`、`model level and optimization level`。
3. 机制怎么长出来：如果模态是混杂变量，就用 CAI 做 backdoor adjustment，优化 `P(Y|do(X))` 而不是普通 `P(Y|X)`；如果偏置还会在训练中传播，就用 CBT 在数据、标签、特征三层拦截，分别是模态特异增广、原图与增广图交换预测做标签细化、原图与增广图特征对齐。
4. 证据闭环：CAI 替代似然建模有单独增益，CBT 叠加后进一步增益；只用 CBT 不如 CAI+CBT，说明两层互补。随机增广替代模态特异增广会掉点，去掉 prediction exchange 或动态 memory 会掉点，传统跨模态 MMD 对齐也会掉点。最有说服力的是伪标签质量曲线和 t-SNE，直接证明“偏置传播”这件事被缓解了。
5. reviewer 为什么买账：这篇卖的是视角，而且是很强的视角。它没有只说“我做了去偏”，而是把两阶段 USL-VI-ReID 的失败路径画成因果图，再把机制一一对应到模型层和优化层，证据也直接打在伪标签质量和模态 gap 上。

**共同套路**

这批论文共同做法不是先想一个模块，而是先把旧任务里的模糊失败重新命名成一个更尖锐的隐藏变量，比如模态加姿态、预训练语义适配、实例级多表示、风格多样性、域内域间权衡、偏置传播。然后让机制从这个命名里自然长出来，最后用消融和可视化证明“这个隐藏变量确实被处理了”，而不只是证明模型涨点。

联网核验主要用到 DASA、DRE、DCR、DMDL 的 arXiv 页面；DMPF 和 DiverseReID 主要依据本地 PDF 正文。DASA 的 arXiv 页面也确认其核心说法是 exemplar-free 的预训练语义适配范式([arxiv.org](https://arxiv.org/abs/2405.19695))；DRE 页面确认其主张是 pure transformer 加多表示用于 LReID([arxiv.org](https://arxiv.org/abs/2403.16003))；DCR 页面确认其核心问题是域内判别和域间 gap 的权衡([arxiv.org](https://arxiv.org/abs/2409.19954))；DMDL 页面确认其主张是模型层和优化层双层模态去偏([arxiv.org](https://arxiv.org/abs/2512.03745))。
tokens used
477,606
你列的是 6 篇，不是 10 篇。我按列出的 6 篇逐篇读了方法部分和实验消融。

**1. DMPF，Disentangling Modality and Posture Factors**

1. 触发观察：作者抓住的是可见光-红外 ReID 的类内差异不只是模态差异，还混着姿态差异。已有方法大多只在图像层或批次特征层对齐模态，忽略批次外的训练集整体差异，也忽略姿态变化。
2. 重定义动作：他们把问题从“跨模态对齐”改写成“身份表征里混入了两个身份无关因子，模态因子和姿态因子”。关键词是 `disentangling modality and posture factors`、`entire training set`、`identity-unrelated factors`。
3. 机制怎么长出来：模态因子既然跨整个训练集存在，就用类别记忆队列加 memory-attention，再用 OT 让同身份跨模态记忆靠近。姿态因子既然来自人体结构，就用 parsing 和 landmark 生成 skeleton，再用正交分解把 skeleton 里的姿态信息和身份信息拆开。机制和重定义绑定较紧，但模块偏重。
4. 证据闭环：关键不是总表涨点，而是 MFD、PFD 单独消融，batch-attention 对 memory-attention，OT loss 在 memory-attention 输出上收敛，parsing 与 landmark 组合对照，impure skeleton 对正交 posture feature 对照。t-SNE 还把模态间隔和姿态离群点分别可视化出来。
5. reviewer 为什么买账：卖的是“VI-ReID 不应只讲模态，还要同时解释姿态污染身份表征”的视角，机制只是顺着两个因子分别长出来。novelty 主要来自问题拆分和证据闭环，不是 attention 或 orthogonal 本身。

**2. DASA，Distribution Aligned Semantics Adaption**

1. 触发观察：直接用预训练权重初始化，在 LReID 中收益有限；旧方法靠 exemplar replay 和 logit distillation，但 ReID 是细粒度开放集，旧分类器和旧样本既不合适也不现实。
2. 重定义动作：他们把 LReID 从“旧知识怎么蒸馏”改成“预训练得到的人体语义如何在新域中低成本适配”。关键词是 `robust and general human semantics`、`shared knowledge`、`distribution alignment`、`semantics adaption`、`exemplar-free`。
3. 机制怎么长出来：如果 Conv 里保存通用人体语义，就冻结 Conv；如果新域分布会干扰语义使用，就调 domain-specific BN；如果通用语义还不够区分身份，就加轻量 SA 模块做空间聚合。这个机制和重定义非常紧，几乎没有多余零件。
4. 证据闭环：它的核心消融是 DA 单独有效、SA 单独有效、DASA 组合最好；核大小 1 会变差，3 或 5 变好，证明 SA 不是随便加卷积，而是需要空间语义聚合；不同预训练数据对照证明“人体预训练语义”确实是上限来源；存储对照证明不用 exemplar 也能长期演化。
5. reviewer 为什么买账：这篇主要卖范式，不是卖模块。它把“终身 ReID 必须蒸馏旧模型”改成“保存预训练人体语义，再按域适配”，这个视角比一个 SA 模块更值钱。

**3. DRE，Diverse Representations Embedding**

1. 触发观察：CNN 终身 ReID 方法为了多表示要加多分支，分支之间独立且难控制；固定 teacher 又会压制新任务适应。作者看到的是旧方法在实例级表征上太单薄。
2. 重定义动作：他们把 LReID 的遗忘问题重定义为“实例级表示不够丰富，导致旧知识保存和新知识适应都受限”。关键词是 `diverse representations`、`primary embedding`、`auxiliary embeddings`、`instance-level and task-level layout`。
3. 机制怎么长出来：Transformer 的多 class token 天然能产生多视角表示，所以引入多个 class token；主表示负责丰富身体信息，辅助表示负责判别性；ACM 对辅助表示做正交分离，再自适应并入主表示；任务级再用 adjustment model 和 learner model 做 KU/KP。
4. 证据闭环：关键证据是 ResNet 版本明显弱于 transformer 版本，CODA 加 ACM 优于 CODA，DRE-KU 缺 KP 会泛化不足，冻结 adjustment model 的 DRE-KUP 适应性不如完整 DRE。辅助表示数量消融也证明不是“越多越好”，两个辅助表示较合适。
5. reviewer 为什么买账：这篇卖的是机制多于视角。视角是“实例级多表示帮助终身学习”，但说法不如 DASA 和 DMDL锋利；真正支撑 novelty 的是把纯 transformer、多 class token、ACM、KU/KP 组合成 LReID 框架。

**4. DiverseReID**

1. 触发观察：DG ReID 里，现有风格增强方法用 Beta、Dirichlet 等固定分布采样混合权重，调参麻烦，而且生成风格可能仍贴近已有源域。作者抓的是“训练数据量不是关键，风格多样性才是关键”。
2. 重定义动作：他们把域泛化从“学域不变特征”改写成“最大化源域风格空间”。关键词是 `style diversity`、`intra-domain diversity`、`inter-domain diversity`、`dynamic style hallucination`、`decoupled domain experts`。
3. 机制怎么长出来：域内多样性不足，就用 DSH 根据两张图的 style Wasserstein distance 动态决定混合权重；域间多样性不足，就维护每个源域的 memory expert，用 DED 让不同域专家的特征去相关。机制很顺，且参数很少。
4. 证据闭环：它不只做 DSH 和 DED 消融，还把 DSH 与 MixStyle、pAdaIN、DSU、SagNet 的 style randomization、SHADE 等对比，并在去掉 DED 后单独比较风格生成方式。另有 DSH 插入位置、采样策略、权重计算方式、DED 权重敏感性和 Grad-CAM。
5. reviewer 为什么买账：这篇卖的是“多样性原则”，不是某个新层。它把两个简单机制分别钉在域内和域间多样性上，且用对比证明自适应风格混合比固定分布更贴合问题。

**5. DCR，Domain Consistency Representation Learning**

1. 触发观察：LReID 有一个矛盾，域内需要更细判别相似身份，域间又需要一致性防遗忘。旧蒸馏方法主要压域间 gap，容易牺牲当前域内判别。
2. 重定义动作：他们把“保存旧知识和学习新知识”改写成“最大化域内判别和最小化域间 gap 的权衡”。关键词是 `domain consistency representation`、`global and attribute-wise representations`、`bridge`、`intra-domain discrimination`、`inter-domain gaps`。
3. 机制怎么长出来：全局表征负责整个人，属性表征负责细粒度局部属性。ATG 先用属性识别器生成属性文本，TGA 用 CLIP 融合图文得到全局表征，ACN 得到属性级表征；AF 用属性表征对齐旧新模型，KC 用全局表征做知识转移。
4. 证据闭环：它用跨五个数据集的 t-SNE 证明域间一致性，用 Market 的身份聚类可视化证明域内判别增强。组件消融显示 PFM、ACN、AF、KC 分别贡献；ATG 用具体属性文本优于泛化的 “a photo of a person”；阈值 0.8 的消融说明属性可靠性被认真控制。
5. reviewer 为什么买账：这篇卖的是 trade-off 视角。CLIP、属性、蒸馏都不新，但“属性级表征作为域内判别和域间一致性的桥”把这些零件串成了一个可审稿的故事。

**6. DMDL，Dual-level Modality Debiasing Learning**

1. 触发观察：无监督 VI-ReID 的两阶段流程先做单模态学习，再做跨模态学习。作者指出第一阶段学到的模态特异线索会进入第二阶段伪标签，并在训练中继续放大。
2. 重定义动作：他们把问题从“伪标签不准”或“模态差异大”改写成“模态偏置从数据传播到标签，再传播到特征”。关键词是 `modality bias`、`dual-level debiasing`、`causal intervention`、`bias propagation`、`model level and optimization level`。
3. 机制怎么长出来：如果模态是混杂变量，就用 CAI 做 backdoor adjustment，优化 `P(Y|do(X))` 而不是普通 `P(Y|X)`；如果偏置还会在训练中传播，就用 CBT 在数据、标签、特征三层拦截，分别是模态特异增广、原图与增广图交换预测做标签细化、原图与增广图特征对齐。
4. 证据闭环：CAI 替代似然建模有单独增益，CBT 叠加后进一步增益；只用 CBT 不如 CAI+CBT，说明两层互补。随机增广替代模态特异增广会掉点，去掉 prediction exchange 或动态 memory 会掉点，传统跨模态 MMD 对齐也会掉点。最有说服力的是伪标签质量曲线和 t-SNE，直接证明“偏置传播”这件事被缓解了。
5. reviewer 为什么买账：这篇卖的是视角，而且是很强的视角。它没有只说“我做了去偏”，而是把两阶段 USL-VI-ReID 的失败路径画成因果图，再把机制一一对应到模型层和优化层，证据也直接打在伪标签质量和模态 gap 上。

**共同套路**

这批论文共同做法不是先想一个模块，而是先把旧任务里的模糊失败重新命名成一个更尖锐的隐藏变量，比如模态加姿态、预训练语义适配、实例级多表示、风格多样性、域内域间权衡、偏置传播。然后让机制从这个命名里自然长出来，最后用消融和可视化证明“这个隐藏变量确实被处理了”，而不只是证明模型涨点。

联网核验主要用到 DASA、DRE、DCR、DMDL 的 arXiv 页面；DMPF 和 DiverseReID 主要依据本地 PDF 正文。DASA 的 arXiv 页面也确认其核心说法是 exemplar-free 的预训练语义适配范式([arxiv.org](https://arxiv.org/abs/2405.19695))；DRE 页面确认其主张是 pure transformer 加多表示用于 LReID([arxiv.org](https://arxiv.org/abs/2403.16003))；DCR 页面确认其核心问题是域内判别和域间 gap 的权衡([arxiv.org](https://arxiv.org/abs/2409.19954))；DMDL 页面确认其主张是模型层和优化层双层模态去偏([arxiv.org](https://arxiv.org/abs/2512.03745))。
