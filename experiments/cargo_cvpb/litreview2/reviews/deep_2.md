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
session id: 019ef835-51dd-7180-a435-f04a69bc47a6
--------
user
你是 ReID 创新方法论拆解员 2。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(2)==
A training-free framework for text-to-image person re-identification via query-prototype matching.pdf
A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf
ADA framework for unsupervised domain adaptation person re-identification.pdf
Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf
Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf
Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我先按项目规则读当前目标和状态，再逐篇用 `pdftotext` 抽方法部分。之后会补查必要背景，重点看作者如何把观察包装成问题和证据闭环。
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
A Semantic-Aware Attention and Visual Shielding Network for Cloth-Changing Person Re-Identification.pdf
Learning Visual-Semantic Embedding for Generalizable Person Re-Identification - A Unified Perspective.pdf
Mask-Aware Hierarchical Aggregation Transformer for Occluded Person Re-Identification.pdf
Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
Internal-External Context Interaction Network for Person Re-Identification.pdf
Learning multi-granularity representation with transformer for visible-infrared person re-identification.pdf
Lifelong Visible-Infrared Person Re-Identification with Prompt Pool and Instance-level Prompt Generator.pdf
Shape-centered representation learning for visible-infrared person re-identification.pdf
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
Learning From Yourself to Others for Unsupervised Visible-Infrared Re-Identification.pdf
Text-to-image Person Search based on Semantic Reorganization.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
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

exec
/bin/zsh -lc "rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'AGENTS.md' -g 'STATUS.md' /Users/abdslm/Desktop/reid-paper" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/CLAUDE.md
/Users/abdslm/Desktop/reid-paper/AGENTS.md
/Users/abdslm/Desktop/reid-paper/GOAL.md
/Users/abdslm/Desktop/reid-paper/experiments/STATUS.md

codex
论文文件都在当前目录，目标清单中的 6 篇也能匹配到。现在先把项目目标和状态读完，避免把这次拆解写偏到实验结论上。
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
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/Desktop/reid-paper/GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,260p' /Users/abdslm/Desktop/reid-paper/CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,240p' /Users/abdslm/Desktop/reid-paper/AGENTS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,240p' /Users/abdslm/Desktop/reid-paper/experiments/STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

**核心结果(reliable_retrieval_eval.py，zero-training，主表 paper_materials/reliable_retrieval_results.md)**：三 backbone×数据集一致——RCIR 式边际名义覆盖达标但 s_top1 难层严重欠覆盖(α0.2 worst-cov 0.31-0.42)→我们分层 group-conditional 修到 0.76-0.88，候选集几乎不变甚至更小(Occ-PoseTrack α0.1 12.8→7.58)。selective mAP@80 +6.75~+10.1，s_top1 Spearman 0.59-0.75，null 对照(random/s_gap)nAURC≈1 正确。**诚实**:worst-group 覆盖修复=评价粒度/部署风险，非标准 mAP 提升；不吹"超越 RCIR 算法"，写"揭示边际 RCIR 在 ReID 隐藏 worst-group 欠覆盖+标准修复"。

**投稿打磨(2026-06-09 Codex)**：已通读 v2、红队意见和核心结果，写成 `paper_materials/paper_final_polish.md`。已小改 `paper_draft_v2.md` 三处：分组边界改为 calibration split 确定，三种子说明排除 SOLIDER Occ-Duke 单 seed 补充点，表 2 caption 同步。**已补**：codex 写出 `tables.tex`(主表+selective+单点假阳性+hard_expand 代价，未跑格子用 -- 标边界)、`refs.bib`(RCIR/DistributionNet/SSPEM/KPR/QPM/ProFD/k-reciprocal/Vovk/Barber/conformal-risk-control 10 篇)、`appendix_implementation.md`(same_camera/合法 gallery/nonconformity/split-conformal 分位数/4 层分组/20 split/AURC/random null/复现校验)。待补重点剩：图编号 caption 对齐、Market 同口径(下面在跑)、英文化、选会议、投稿格式。

**整夜自主推进中(cron 9859ade0 每30min + 等待器)**：
- **论文素材一致性+诚实性审计已做(已提交 34b50b8)**：codex 审计主表数字全对上源真值、无硬越界；修了图号错位(只3张真图却引用图1-8→归正+未画的标待绘)、tables.tex 无源单 seed 曲线点(删→--、52.60→52.57)、Market 数回填源真值标 exp011 旧口径、两处过强措辞软化、results.md 内部 α 标注矛盾。详见 consistency_audit-codex.md。
- **AURC + 完整 selective 曲线：5 个 backbone×数据集全部捕获(填用户待办#1/#2，已提交 b770118/70b1de0/4582dbd/b5b624b)**：原 eval stdout 没持久化、@80 之外全丢，重跑捕获完整 SELECTIVE 行。SOLIDER Occ-PoseTrack 三种子(AURC 0.113±0.001)、ImageNet-Swin Occ-PoseTrack 单seed(0.133)、ImageNet-Swin Occ-Duke 三种子(0.335)、ViT Occ-Duke 三种子(0.247±0.001)、SOLIDER Occ-Duke 单seed(0.228)。**每项 worst-group/Spearman/no_reject/@80 全对上★主表**(强自洽)。AURC 按难度 0.113-0.335 排序、nAURC 全<1(null random/s_gap nAURC≈1.0-1.5)。tables.tex selective 表 5 行齐、fig3 五条真实多点曲线、日志全存 eval_logs/。
- **exp014 Market 同口径对照三种子定稿(决策#41 完成，已提交 96ee1a9)**：复现通过(三种子训练 mAP 91.6/91.6/91.6 对上 exp000)。**核心结果(三种子均值)**：α0.2 边际 RCIR worst-cov 0.218±0.001→我们分层 0.774±0.005；α0.1 0.605±0.003→0.883±0.001；Spearman 0.364±0.004、selective@80 +3.14、AURC 0.043。即非遮挡 Market 上边际 conformal 条件失效比遮挡(0.30-0.42)更严重(0.218)、标准分层同样修复。**"条件失效非遮挡特有=否"的同口径三种子直接证据**。中途 seed2 等待器抢跑致 OOM 已重启恢复；Occ-Duke AURC 改在空闲 GPU1 即时跑省空转。
- **★ 至此可靠检索线实验数据全部采集完毕**：主 worst-group 表(5 组合三种子)、selective+AURC 表(5 组合完整曲线)、Market 同口径对照(三种子)、单点假阳性两划分两数据集、hard_expand 代价、正交容量律。eval_logs/ 14 个日志入库作证据。3 张配图(fig1 容量律/fig2 worst-group/fig3 selective 5曲线)。论文 v1/v2+final_polish+tables.tex+refs.bib+appendix+一致性审计全在。
- **最终红队(完整论文，redteam_final-codex.md)已做**：核心张力=数据齐全后越诚实技术增量越薄，能站"ReID 可靠检索协议+风险诊断"不能站方法稿。命中重估 **PRCV 0.52-0.66 / ICPR 0.36-0.50 / ICME 0.30-0.42**(数据补全降低"实验不足"风险,但 Market 三种子让"遮挡特有"防线消失,故 ICME/ICPR 不上调,PRCV 略稳)。3 最该补(全措辞/已有数据,无需新实验)：①拆通用 vs 遮挡贡献框架 ②代价进主叙述 ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标)。
- **我已做的非框架补充(已提交)**：hard_expand 表补全 6 组合困难组候选集绝对大小(RCIR 0.3-5.7→我们 4-95,Occ-Duke 偏大但~gallery 0.5%)，回应红队 fix#2 的数据部分。**框架/定位类(拆贡献、降级强说法、叙述 placement、选会议、英文化、投稿)全留用户决策，不擅自改大方向**。
- **conformal 方法学有效性审查(conformal_validity_review-codex.md)已做，关键正面结论**：以 conformal 理论审稿人角度逐行审 reliable_retrieval_eval.py，**无代码硬伤**，核心诊断(边际达标但 s_top1 难层欠覆盖、标准分组可修复)在实现层站得住；每组 calibration query ~276-421，worst-group 0.218/0.88 不是小样本崩坏；覆盖数字全对上日志。需收窄/限定的全是理论口径(身份可交换+固定gallery假设、RCIR-style marginal 非原文、worst 是4预固定组经验最小值、α0.2 worst 0.76-0.77 写"接近")，多数 draft 已做。总评：作经验诊断+标准分组校准闭环可投稿，作"无条件分布无关 conformal 定理/超越RCIR新算法"站不住=正是诚实定位。
- **★ 这条线的自主可做事项已全部完成**：实验数据全齐+论文素材全齐+一致性审计+两轮独立红队(通用CCF-B novelty + conformal理论validity，都无硬伤)+代价绝对量级数据。剩余纯属用户战略决策。三台机器实验程序已完，正确闲置(无非冗余线内活)。
- **④四组覆盖表已补(2026-06-09,提交 1c3e967)**：给 eval 加 GROUPS 打印(只输出已聚合的逐层覆盖/样本量,不改计算),重跑 4 代表组合,RCIR worst-group+SELECTIVE 与重跑前完全一致。新机制图景:边际 RCIR 把易层过覆盖到 0.99 凑总体 0.80、难层只 0.22-0.42,我们分组校准四组全拉到 ~0.80;每组 calib 276-422 充足(非小样本噪声)。见 results.md"四组逐层覆盖"节。
- **留给用户的提交前清单(全是战略决策或需谨慎实现的方法学变体,不擅自做)**：①拆通用vs遮挡贡献框架(战略) ②代价进主叙述(数据已备,placement战略) ③降级强说法(正交容量律→经验观察、selective"提升"→行为指标,战略=是否让步一个卖点) ④~~四组覆盖表~~已补 ⑤identity-block/每身份一 query 稳健性附录(高价值,直接消除最大理论攻击=簇内相关破坏可交换性;但属方法学变体,改 calibration 采样逻辑,按纪律新变体须先复现已知基线再下结论,留用户确认实现方式后再做,别在深夜赶) ⑥选会议(PRCV 0.52-0.66 最稳)/英文化/格式/投稿。

---

## ★★★ 旧（2026-06-08 08:25，选项1 执行中）★★★

  → 抽取 6 篇 PDF 的前 8 页文本，定位完整方法段落。
  • 必要时查网页确认题名、年份、先例和方法定位。
  • 逐篇按 5 个问题写方法论拆解，并总结共同套路。
codex
我会先直接抽每篇前 8 页的正文，再根据标题定位方法段。若某篇的方法标题在第 8 页后才收尾，我会再补读后续页，但最终只提炼方法论，不复述模块细节。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'ADA framework for unsupervised domain adaptation person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 171 (2026) 112238

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

ADA framework for unsupervised domain adaptation person
re-identiﬁcation
Wei Zhang a , Peijun Ye b , Dihu Chen c , Tao Su a,∗
School of Electronics and Information Technology, Sun Yat-sen University, Guangzhou Higher Education Mega Center, Guangzhou, China
Peng Cheng Laboratory, No. 2 Xingke 1st Street,Nanshan District, Shenzhen, China
c School of Integrated Circuits, Sun Yat-sen University, NO 66 Gongchang Road, Guangming District, Shenzhen, China
a

b

a r t i c l e

i n f o

Keywords:
Person re-identiﬁcation
Domain adaptation
Sparse self-attention

a b s t r a c t
Domain shift remains a critical barrier for generalizing person re-identiﬁcation (ReID) models across datasets.
To address this challenge, we present a sparse self-Attention augmented Domain Adaptation (ADA) framework
that learns domain-invariant identity features through three key innovations: (1) Sandwich Attention Primitive
(SAP), a novel computational unit designed to boost primitive-level domain adaptation. (2) Sparse self-Attention
Augmented Bottleneck block (SAAB block), a hierarchical block integrating SAP to enhance adaptation at the
architecture level. (3) Scalable Design, if necessary, SAAB block can be ﬂexibly cascaded to construct taskspeciﬁc ADA framework. Experiments on three benchmarks validate ADA’s superiority: (1) Achieves state-of-theart performance across domains (e.g., 16.5 % mAP gain on CUHK03→Market-1501). (2) Demonstrates consistent
generalizability and adaptability.

1. Introduction
Unsupervised Domain Adaptive Person Re-Identiﬁcation (UDA ReID) transfers recognition from labeled source to unlabeled target
domains via domain-invariant identity features. This task addresses
cross-camera or cross-domain identity matching without target domain
annotations, making it a critical technology for intelligent missing person search and smart city surveillance systems. Consequently, UDA ReID has garnered signiﬁcant research interest due to its practical applicability and technical challenges.
The domain shift in UDA Re-ID primarily stems from dataset-speciﬁc
variations in pedestrian appearance. Key factors include: (1) Illumination discrepancies: Lighting condition diﬀerences across domains alter
visual features. (2) Viewpoint variations: Camera perspectives aﬀect
pose and occlusion patterns. (3) Background clutter: Irrelevant scene elements introduce noise to feature representations. (4) Camera-speciﬁc
biases: Sensor characteristics (e.g., resolution, color proﬁles) distort appearance consistency. To mitigate these challenges, learning domaininvariant pedestrian features-representations robust to dataset-speciﬁcbiases is essential for eﬀective cross-domain adaptation.
Prior studies on UDA Re-ID have primarily focused on three dominant paradigms: (1) Clustering-based adaptation, (2) Image style
translation, and (3) Feature-level alignment. While these methods
have advanced cross-domain adaptation research, they exhibit critical

limitations: (1) Feature-level alignment often relies on auxiliary annotations or information to align distributions, increasing implementation
complexity. (2) Image style translation suﬀers from unstable optimization due to its dependency on generative adversarial networks (GANs).
(3) Clustering-based methods are prone to error propagation from noisy
pseudo-labels and imbalanced cluster distributions.
To address these challenges, we present a ADA framework -a
lightweight, sparse self-Attention augmented Domain Adaptation framework that integrates sparse token interactions to learn domain-invariant
identity representations. Our ADA framework eliminates the need for
auxiliary supervision while achieving superior cross-domain generalization.
In brief, our ADA framework advances domain adaptation in person
re-identiﬁcation (ReID) through following innovations:
•

Sandwich Attention Primitive (SAP): We introduce a novel computational primitive(SAP) integrating a sandwich attention mechanism
to enhance domain adaptation. The SAP ﬂexibly adapts to sparse
self-attention mechanisms across heterogeneous design paradigms,
eﬀectively bridging domain shifts by synergizing the complementary strengths of convolutional inductive bias and sparse token
interactions.
• Sparse self-Attention Augmented Bottleneck block (SAAB block):
Building on SAP, we propose the SAAB block, a hierarchical

∗ Corresponding author.

E-mail addresses: yepj@pcl.ac.cn (P. Ye), stscdh@mail.sysu.edu.cn (D. Chen), sutao@mail.sysu.edu.cn (T. Su).
https://doi.org/10.1016/j.patcog.2025.112238
Received 18 October 2023; Received in revised form 18 April 2025; Accepted 5 August 2025
Available online 8 August 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

bottleneck structure that systematically improves domain-invariant
feature learning. If necessary, by cascading SAAB blocks, we construct a scalable sparse self-Attention augmented Domain Adaptation framework (ADA framework), enabling ﬂexible integration into
existing architectures.
• Robustness and Superiority: Extensive experiments on three ReID
benchmarks demonstrate that the ADA framework achieves consistent, generalizable, and adaptable performance across varying sparse
self-attention conﬁgurations. As a plug-and-play module, the ADA
framework enhances model accuracy at multiple deployment stages
while surpassing many SOTA methods in cross-domain scenarios.

In contrast to these approaches, our methodology circumvents multiple limitations through three key innovations: First, it operates without
requiring auxiliary annotations from either domain, reducing implementation complexity. Second, it eliminates dependence on GAN-generated
synthetic images, thereby avoiding quality constraints of image translation. Third, the proposed architecture demonstrates superior training
stability and faster convergence compared to clustering-based alternatives while maintaining deployment ﬂexibility.
2.2. Self-attention augmented convolution
The synergistic integration of self-attention with convolution enables
complementary feature learning, achieving performance enhancements
with moderate computational overhead. This fusion paradigm addresses
the intrinsic limitations of CNNs in modeling long-range dependencies
while preserving local feature extraction capabilities.
Self-attention augmented convolution addresses the inherent locality constraints of conventional convolution through long-range dependency modeling. AA-ResNet [10] pioneered a parallel implementation
where convolutional and self-attention operators process identical input feature maps simultaneously. The ﬁnal output is generated through
channel-wise concatenation of the convolution-derived local features
and self-attention-extracted global contextual features. BoTNet [11] enhanced this paradigm by replacing spatial convolutions in ResNet’s ﬁnal
bottleneck stage with attention modules, realizing measurable performance gains.
These approaches integrate standard self-attention mechanisms with
quadratic spatial complexity 𝑂((𝐻𝑊 )2 ) (where H and W denote feature map height and width) within supervised learning frameworks.
In contrast, our method introduces two key upgradation: (1) sparse
self-attention with reduced spatial complexity, and (2) unsupervised
domain adaptation deployment without backbone architecture modiﬁcation. The proposed modular design enables ﬂexible implementation
while eﬀectively bridging domain gaps in person re-identiﬁcation tasks.
Existing supervised methods universally require structural alterations to backbone networks for performance enhancement. Our framework diﬀerentiates itself by operating in unsupervised domain adaptation (UDA) scenarios, maintaining the original backbone architecture
while achieving comparable adaptability. This non-invasive integration
ensures deployment ﬂexibility without compromising target domain
generalization.

2. Related work
Prior research relevant to this work primarily encompasses three
technical directions in unsupervised domain adaptation (UDA) for
person re-identiﬁcation (Re-ID): self-attention augmented convolution, transformer-based person re-identiﬁcation, and domain adaptation
methodologies for person re-identiﬁcation. This section provides a concise review of these foundational approaches.
2.1. UDA methods for person re-identiﬁcation
Current UDA-ReID solutions primarily address domain gaps through
three principal paradigms mentioned earlier. Each methodology
presents distinct advantages and limitations as analyzed below:
Feature-Level Alignment. This category mitigates domain discrepancies through hierarchical feature alignment across source and target
domains. Through dynamic token matching and ﬂexible allocation of
local tokens to human joint points, POSR [1] propose pose-aligned feature learning module to obtain pose-aligned features. HUDA [2] implements joint global distribution alignment and local instance matching
to enhance knowledge transferability. Similarly, TJ-AIDL [3] employs a
dual-branch network architecture with identity inference space, leveraging both semantic labels and auxiliary attribute annotations from source
domains. While eﬀective, these methods often require supplementary
annotations (e.g., attribute labels), increasing implementation complexity and data acquisition costs.
Image Style Translation. GAN-based approaches attempt to bridge
domain gaps through cross-domain image synthesis. PTGAN [4] exempliﬁes this strategy by preserving source-domain identity attributes
while transferring target-domain background characteristics. PDA-Net
[5] learns domain invariant representation with disentangled pose
and domain information. However, such methods face two inherent
constraints: (1) The training instability characteristic of adversarial
networks often prolongs convergence time compared to conventional
CNNs; (2) The synthetic image quality remains fundamentally constrained by ﬁtting quality of GANs.
Clustering-Based Adaptation. Pseudo-labeling techniques typically
employ pre-trained models for feature extraction followed by clusterguided pseudo-label assignment. Recent advances address critical limitations in this paradigm: MGCN [6] learns noise-resistant features
via prototype-, triplet-, and list-wise consistency. 𝐶 3 𝐴𝐵 [7] introduces cluster-aware boundary adaptation to reﬁne pseudo-label reliability while mitigating class imbalance through its Cluster-Adaptive Balance (CAB) loss. The MLC framework [8] further enhances robustness
through multi-granularity feature learning, memory bank-based label
assignment, and noise-resistant self-paced clustering. The CMC framework [9] employs a graph convolutional network to construct a connection estimator, which facilitates the transfer of relational knowledge
from the source domain to the target domain. Nevertheless, clusteringbased approaches inherently suﬀer from two limitations: (1) Performance ceilings imposed by clustering algorithm precision; (2) Error
propagation from noisy pseudo-labels that exacerbate intra-cluster data
imbalance.

2.3. Transformer-based person re-identiﬁcation
Transformer-based person re-identiﬁcation (ReID) methods broadly
fall into two categories: (1) hybrid architectures combining CNNs and
Transformers, and (2) pure Transformer-based approaches. APD [12]
uses a CNN as a feature extractor, followed by a modiﬁed visual Transformer for Part Mask Generation and Part Merging. TMGF [13] leverages
the ﬁnal Transformer layer to split features in a dual-branch backbone
network, enabling multi-granularity feature learning.
While these methods advance ReID research by exploring hybrid
or multi-branch architectures, they signiﬁcantly increase computational
complexity due to dense matrix operations within backbone network. In
contrast, our approach retains the original backbone architecture and
requires only sparse self-attention operations, drastically reducing computational overhead.
3. Methods
3.1. Description of the overall workﬂow
The proposed domain adaptation framework operates in joint training mode through the following workﬂow: Labeled source domain images and unlabeled target domain samples are aggregated and partitioned into mini-batches containing equal proportions of both domains.
2

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

Fig. 1. The ﬂowchart for the ADA framework: the ADA framework is deployed only once, and the eﬀective deployment location is behind any one of the stages
1/2/3 of the backbone network. The backbone is inserted a hybrid classiﬁer. The pseudo labels were generated after DBSCAN clustering and updated once at the
starting of each training epoch.

Each mini-batch is sequentially processed through our network pipeline,
which comprises:(1) A ResNet-50 backbone [14] (pre-trained on ImageNet, containing stages 0–4). (2) The ADA framework. (3) Cluster computation module. (4) Composite Re-ID loss 𝐿𝑅𝑒𝐼𝐷 .
As illustrated in Fig. 1, the Re-ID loss 𝐿𝑅𝑒𝐼𝐷 combines classiﬁcation
loss 𝐿𝑐𝑙𝑠 [15] and triplet loss 𝐿𝑡𝑟𝑖 [16]. The complete training loss is
formulated as:
𝐿𝑅𝑒𝐼𝐷 = 𝐿𝑐𝑙𝑠 + 𝐿𝑡𝑟𝑖

(1)

The ADA framework’s deployment requires balancing between computational overhead and performance optimization. Operating as a plugand-play module across both training and inference phases, the framework’s architecture permits ﬂexible scalability. In our implementation,
a simple conﬁguration is adopted with single SAAB block integration.
Note: (1)after stages 1/2/3 are all eﬀective deployment locations. This
design choice ensures computational eﬃciency while maintaining adaptation eﬀectiveness. (2) Inspired by the review [17,18], with the rapid
evolution of sparse self-attention mechanisms, it is possible to ﬂexibly
decide whether to stack SAAB blocks based on practical needs. Fig. 2
provides detailed architectural speciﬁcations.

Fig. 2. The diagram of the ADA framework: for simplicity, the BN and ReLU
operations are not drawn in the ﬁgure. In this article, stackable SAAB block is
only stacked once, and the performance improvement is signiﬁcant. Therefore,
its memory usage is relatively reasonable.

3.2. Introduction to the ADA framework

The SAP augments the capacity of domain adaptation through: (1)
Sparse Attention Eﬃciency: Implements Sparse complexity attention(<
𝑂(𝑁 2 )) versus conventional approaches(= 𝑂(𝑁 2 )). (2) Modular Flexibility: Permits plug-and-play integration of various sparse self-attention
variants. (3) Symmetrical Feature Encoding: Channel-wise symmetric
distributions emerge through bottleneck-residual coupling. Fig. 3 depicts the functional block diagram of SAP.

3.2.1. Introduction to the SAAB block and SAP
The ADA framework enhances cross-domain discriminative capability through domain-invariant identity representation learning.
Its core innovation lies in dual attention mechanisms: strong focus
on identity-salient features (pedestrian attributes) while suppressing
domain-speciﬁc distractors (background elements) via long-range dependency modeling. As joint training progresses, the framework demonstrates progressive domain adaptation through the SAAB block and the
computational primitive SAP.
To enhance domain adaptation performance, the SAAB block introduces the following key improvements over the bottleneck block structure: (1) Core Computational Primitive. The bottleneck block relies on
standard convolution, whereas the SAAB block replaces it with the SAP
(Sandwich Attention Primitive). (2) Bi-Level Bottleneck Structure. The
ﬁrst level is embedded via SAP. The second level is constructed by
the SAAB block based on SAP. (3) Expanded Receptive Field. In bottleneck block, the receptive ﬁeld is limited by a single 3 × 3 convolution. The SAAB block achieves broader contextual awareness through
self-attention mechanism. (4) Overﬁtting Integration. The SAAB block
incorporates dropout operations, which are absent in bottleneck block,
further improving robustness.

3.2.2. Algorithm description for the SAP
To rigorously characterize the SAP, we present its mathematical implementation through PyTorch-style pseudocode across ﬁve sparse selfattention variants(Linear Self-Attention [19], Permuted Self-Attention
[20], Focused Linear Attention [21], Bi-Level Routing Attention [22],
Window based Self-Attention [23]). The pseudocode description is presented in the following algorithmic listing.
#i n p u t f e a t u r e Xin a r e ( Cin , H, W) . o u t p u t f e a t u r e a r e
( Cout , H, W)
#2D c o n v o l u t i o n Conv2d_head , ( Cout , H, W)
x1 = Conv2d_head ( Xin )
#2D c o n v o l u t i o n Conv2d_down , ( Cin /4 ,H, W)
x = Conv2d_down ( Xin )
# p e r m u t e from ( Cin /4 ,H,W) t o (H,W, Cin /4)
3

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

Q = ( x ^∗∗p) ∗ | | x | | / | | x ^∗∗p | |
K = ( x ^∗∗p) ∗ | | x | | / | | x ^∗∗p | |
V = ( x ^∗∗p) ∗ | | x | | / | | x ^∗∗p | |
# a t t e n t i o n compute
a t t e n=MatrixMul (Q, K . t r a n s p o s e )
#conv2d_dwc , a d e p t h w i s e c o n v o l u t i o n
x2=MatrixMul ( a t t e n , V) + conv2d_dwc (V)
# r e s t o r e t o ( Cin /4 ,H,W)
x2 = reshape ( x2 )

x = permute ( x )
# function Sparse_attention
r e f e r s t o s p a r s e s e l f −a t t e n t i o n p r o c e s s i n g
x2= S p a r s e _ a t t e n t i o n ( x )
#2D c o n v o l u t i o n Conv2d_BneckUp r e s t o r e c h a n n e l s t o
( Cin , H, W) x2 = Conv2d_BneckUp ( x2 )
#2D c o n v o l u t i o n r e s t o r e c h a n n e l s t o ( Cout , H, W)
x2 = C o n v 2 d _ t a i l ( x2 )
# s p a r s e and c o n v o l u t i o n f e a t u r e f u s i o n
Output = x1 + x2

# c a s e 4 , Bi−L e v e l R o u t i n g A t t e n t i o n
#r e s h a p e t e n s o r t o ( S ^ 2 ,HW/S ^ 2 , Cin /4)
#S : t h e number o f r e g i o n s d i v i d e d by H and W o f
f e a t u r e map x = r e s h a p i n g ( x , reg ion_H=H// S )
#L i n e a r p r o j e c t i o n o f q u e r i e s , keys , and v a l u e s
Q, K , V = q k v _ L i n e a r P r o j e c t i o n ( x ) . chunk ( 3 , dim=−1)
#a v e r a g e on a l l r e g i o n s , g e t r e g i o n a l q u e r i e s , k e y s
( S ^ 2 , C)
Q_r , K _ r = Q. AVG( dim=1), K . AVG( dim=1)
#a d j a c e n c y m a t r i x AM ( S ^ 2 , S ^ 2 ) f o r
#r e g i o n−to−r e g i o n g r a p h
# MatrixMul : m a t r i x m u l t i p l i c a t i o n
AM = MatrixMul ( Q_r , K _ r . permute(−1, −2))
#k : t o t a l number o f e f f e c t i v e r e g i o n s t o a t t e n d
#p r o d u c e topK i n d e x m a t r i x IM ( S ^ 2 ,K)
IM = I n d e x M at r i x (AM, K ) . topK
# c o l l e c t key and v a l u e p a i r s ( S ^ 2 , kHW/S ^ 2 , C)
K_g = K V _ c o l l e c t (K , IM )
V_g = K V _ c o l l e c t (V , IM )
#t o k e n−to−t o k e n a t t e n t i o n
#MatrixMul : m a t r i x m u l t i p l i c a t i o n
Atten = MatrixMul (Q, K_g . permute(−2, −1))
Atten = softmax ( Atten , dim=−1)
# conv2d_dw ( ) : d e p t h w i s e c o n v o l u t i o n
x= MatrixMul ( Atten , V_g ) + conv2d_dw (V)
#r e s h a p e t e n s o r from ( S ^ 2 ,HW/S ^ 2 , Cin /4) t o
(H,W, Cin /4) x = unreshape ( x , reg ion_H = H// S )
#p e r m u t e t e n s o r t o ( Cin /4 ,H, W)
x2 = permute ( x )

# c a s e 1 , L i n e a r S e l f−A t t e n t i o n
#The low−rank m a t r i x a p p r o x i m a t e s
#s e l f −a t t e n t i o n mechanism .
#L i n e a r p r o j e c t i o n , Q, K and V a r e (HW, Cin /4)
Q = linear_Q (x)
K = linear_K (x)
V = linear_V (x)
# l i n e a r P r o j e c t i o n Matrix , PM_k ( P r j ,HW)
# l i n e a r P r o j e c t i o n Matrix , PM_v ( P r j ,HW)
#K_p and V_p i s ( P r j , Cin / 4 ) , and P r j << HW
#MatrixMul : m a t r i x m u l t i p l i c a t i o n
K_p = MatrixMul ( PM_k , K)
V_p = MatrixMul ( PM_v , V)
# a t t e n t i o n compute
a t t e n = MatrixMul (Q, K_p . t r a n s p o s e )
a t t e n = softmax ( a t t e n )
x2 = MatrixMul ( a t t e n , V_p )
# r e s t o r e t o ( Cin /4 ,H,W)
x2 = reshape ( x2 )
# c a s e 2 , Permuted S e l f−A t t e n t i o n
# r i s tokens reduction factor .
#S i s t h e s i d e l e n g t h o f each window .
#HW=N∗S ^ 2
#r e s h a p e t e n s o r t o (N, S ^ 2 , Cin /4)
x = reshape ( x )
# Q m a i n t a i n s (N, S ^ 2 , Cin /4)
Q = linear_query (x)
#K and V from (N, S ^ 2 , Cin /4) t o (N, S ^ 2 , Cin /(4∗ r ^ 2 ) )
K = linear_key (x)
V = linear_value (x)
#t r a n s f e r
s p a c e t o k e n s i n K and V
to channel dimention
#K_p and V_p i s (N∗S ^ 2 / r ^ 2 , Cin /4)
K_p , V_p= permute (K , V)
#s e l f −a t t e n t i o n p e r f o r m e d on Q, K_P and V_P
#PE i s an a l i g n e d r e l a t i v e p o s i t i o n embedding
a t t e n = MatrixMul (Q, K_p . t r a n s p o s e ) + PE
a t t e n = softmax ( a t t e n )
x2 = MatrixMul ( a t t e n , V_p )
# r e s t o r e t o ( Cin /4 ,H,W)
x2 = reshape ( x2 )

# c a s e 5 , Window b a s e d S e l f−A t t e n t i o n
#windows g e n e r a t i o n
#r e s h a p e t e n s o r from ( 1 , n , Cin /4) t o (Wn, n1 , Cin /4)
#Wn i s t h e number o f windows ,
#n1 i s t h e p r o d u c t o f t h e l e n g t h and w i d t h o f t h e
window x = reshape ( x )
#p r o d u c e Q, K , V by l i n e a r
#t e n s o r from (Wn, n1 , Cin /4) t o (Wn, n1 , 3 ∗ Cin /4)
QKV = l i n e a r ( x )
#t e n s o r from (Wn, n1 , 3 ∗ Cin /4) t o ( 3 ,Wn, n1 , Cin /4)
QKV = reshape (QKV)
#Q, K , V =(Wn, n1 , Cin /4)
Q = QKV[0]
K = QKV[1]
V = QKV[2]
# a t t e n t i o n compute
a t t e n = MatrixMul (Q, K . t r a n s p o s e )
a t t e n = softmax ( a t t e n )
x2 = MatrixMul ( a t t e n , V)
# r e s t o r e t o ( Cin /4 ,H,W)
x2 = reshape ( x2 )

# case 3 , Focused Linear A t t e n t i o n
# p i s t h e v a l u e o f t h e power o f X
#F o c u s e d F u n c t i o n a d j u s t t h e d i r e c t i o n o f
#each q u e r y and key f e a t u r e s , d r i v i n g s i m i l a r
#query−key p a i r s c l o s e r w h i l e p u s h i n g
# d i s s i m i l a r query−key p a i r s away
#f ( x ) i s F o c u s e d F u n c t i o n
#f ( x ) = ( x ^∗∗ p ) ∗ | | x | | / | | x ^∗∗ p | |
#x ^∗∗ p r e p r e s e n t s e l e m e n t−w i s e power p o f x
x = ReLU ( x )

Pseudocode description reveals three fundamental design principles:
•

4

(1) Spatial-Channel Feature Redistribution: (I) Permuted SelfAttention induces channel-spatial attention equilibrium through

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

Table 1
Benchmarks information of UDA re-ID.

CHUK03

market1501

MSMT17

Fig. 3. The diagram of the SAP: the SAP achieves architectural ﬂexibility, enhances feature representation , and optimizes the complexity-accuracy tradeoﬀ.

•

subset

IDs

images

cameras

train
query
gallery

757
700
700

7365
1400
5332

6
6
6

subset

IDs

images

cameras

train
query
gallery

751
750
751

12,936
3368
15,913

6
6
6

subset

IDs

images

cameras

train
query
gallery

1041
3060
3060

32,621
11,659
82,161

15
15
15

Batch Conﬁguration: 64 source-domain samples (16 identities) + 64
target-domain samples per batch.

Under ﬁxed DBSCAN hyperparameters, our methodology emphasizes systematic evaluation of the framework’s intrinsic domain adaptation capabilities. The clustering mechanism serves strictly as an auxiliary process, generating pseudo-labels across experiment. This controlled conﬁguration enables rigorous comparative analysis by isolating
attention mechanism eﬀects from clustering variability.
Architectural generalization strategy. To establish the architectural generality of the ADA framework, three uniﬁed design principles
were implemented regardless of sparse self-attention selection: (1) interface standardization, (2) hyperparameter consistency, and (3) simple
SAAB block conﬁguration. The hyperparameter space was intentionally
constrained, maintaining fewer than 4 tunable parameters per module to
ensure comparative fairness. This systematic evaluation enables direct
measurement of the ADA framework’ intrinsic adaptation capability.

dimension transposition. (II) Window-based Self-Attention maintains spatial locality constraints via ﬁxed-size partitioning.
• (2) Computational Complexity Optimization: (I) Linear SelfAttention implements low-rank matrix approximating self-attention
mechanism. (II) Bi-Level Routing Attention reduces global computation through regional token routing.
• (3) Attention Distribution Approximation: (I) Focused Linear Attention employs learnable kernel mapping for softmax approximation.
(II) Linear variants utilize low-rank matrices for attention matrix estimation.
The SAP architecture demonstrates three key capabilities through
these implementations:
•

(1) Architectural Flexibility: Seamless integration of diverse sparse
self-attention paradigms.
• (2) Feature Fusion Enhancement: Complementary convolutionattention feature synthesis.
• (3) Complexity-Accuracy Tradeoﬀ: Maintains high accuracy while
reducing FLOPs.

4.3. Comparison with state-of-the-art methods
Our framework is benchmarked against SOTA unsupervised person
re-identiﬁcation methods. As demonstrated in Table 2, our framework
achieves competitive performance while exhibiting superior architectural ﬂexibility - speciﬁcally enabling deployment at multiple network
hierarchies without structural constraints, unlike position-ﬁxed implementations in existing methods [28].
The proposed methodology demonstrates broad compatibility with
diverse baseline architectures. Comparative analysis against topperforming baselines in the tabular comparison reveals that our approach exhibits a more substantial performance improvement margin.
PP+LP [29] improves performance by leveraging memory-based
cluster contrastive learning to reﬁne features and pseudo-labels, albeit

4. Experiments
4.1. Experiments setting and evaluation protocol
We evaluate our framework on three UDA-ReID benchmarks
(CUHK03 [24], MSMT17 [4], and Market-1501 [25]) . Performance is
quantiﬁed using standard metrics: mean Average Precision (mAP) and
Cumulative Matching Characteristics (CMC) Rank-1 accuracy. Our implementation strictly uses only source domain labels and target domain
pseudo-labels generated through joint training, without additional supervision. Dataset statistics are provided in Table 1.

Table 2
Evaluation compared to leading methods on benchmark datasets. (MS→M:
MSMT17 [4] → Market-1501 [25]).

4.2. Implementation details

MS → M

M → MS

mAP R1

mAP

R1

42.4
−
81.0
81.5
82.7
85.3
85.8

−
13.4
31.3
31.4
32.1
41.8
39.5

−
34.3
60.4
61.1
59.6
71.4
67.9

high-performance baseline(IBN-ResNet50)
IBN-ResNet50+ADA (Ours)

80.1 91.3 20.2
83.4 92.8 23.1

41.3
46.0

Baseline (ResNet50)
ResNet50+ADA (Ours)

70.4 86.1 11.9
80.2 91.8 17.0

27.3
36.5

Method

Experiments were conducted on a workstation equipped with dual
NVIDIA RTX 3090 GPUs, AMD Ryzen R7 7600X CPU, and 128GB DDR5
RAM. The framework was implemented in PyTorch with the following
conﬁgurations:

HUDA [2]
MDJL [30]
SECRET [31]
PPLR [32]
MTNet [33]
CCL [28]
FP+LP [29]

Image Processing: Inputs resized to 256 × 128 pixels with standard
augmentation protocols (random ﬂipping, cropping, and erasing).
• Clustering: DBSCAN [26] with Jaccard distance metric [27] (maximum neighbor distance 𝐸𝑃 𝑆 = 0.6, minimum neighbor count
𝑛min = 4).
• Optimization: Adam optimizer with initial learning rate𝜂 = 0.001 ,
0
step-decay schedule(𝜂𝑡+20 = 0.1𝜂𝑡 ) over 50 epochs.
•

5

Reference

PR
PR
AAAI
CVPR
JVCI
TCSVT
TIP

72.3
−
92.6
92.8
93.0
94.1
94.5

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

at the expense of increased architectural and computational complexity
(due to its teacher-student framework) compared to our approach. Compared to PP+LP [29], the baseline(82.8 % mAP, 92.7 % R1) of PP+LP
is higher than our baseline(80.1 % mAP, 91.3 % R1). Although PP+LP
attains higher absolute performance metrics, the ADA framework shows
greater relative performance improvement(↑ 3.3 % mAP/1.5 % R1) over
baseline implementations.
Unlike CCL [28], which combines contrastive learning, attention
module(3DAM), and unsupervised learning with a memory bank for
performance gains, our ADA framework intentionally avoids memory
bank to assess its intrinsic capability. While 3DAM needs multi-stage
deployment for optimal results, and CCL relies on camera-count labels,
the ADA framework operates without such constraints. Currently, CCL
achieves higher performance, but integrating memory bank could further enhance the ADA framework.

directly validate the framework’s design objectives. Table 3 summarizes the peak performance metrics of four distinct dataset combinations derived from three benchmark datasets (CUHK03, Market-1501,
and MSMT17).
Through systematic analysis of Table 3, we observe the following
performance enhancement (Note: (1) sorted by improvement magnitude. (2) CUHK03 to Market-1501):
the ADA framework (Linear Self-Attention): ↑ 16.5 % mAP/10.6 %
R1.
• the ADA framework (Window based Self-Attention): ↑ 15.2 % mAP/
9.5 % R1.
• the ADA framework (Focused Linear Attention): ↑ 13.9 % mAP/9.4 %
R1.
• the ADA framework (Permuted Self-Attention): ↑ 12.3 % mAP/ 8.4 %
R1.
• the ADA framework (Bi-Level Routing Attention): ↑ 11.4 %
mAP/8.1 % R1.
•

4.4. Ablation study
We conducted ablation studies to assess three key aspects of the ADA
framework: generalizability, consistency, and adaptability. The evaluation spanned four dimensions: (1) Datasets, (2) Baseline methods,
(3) Backbone network architectures, (4) Computational primitives. The
experimental design incorporates three distinct baseline variants implemented through backbone network substitution: (1) a lightweight baseline, (2) an optimized high-performance baseline, and (3) a transformeroriented architectural implementation.
For precise interpretation of ablation results, we categorize the backbone network’s feature hierarchy as follows: (1) stages 0–1, low-level
feature. (2) stage 2, intermediate semantic representation. (3) stages
3–4, high-level feature.
The lightweight baseline comprises the following key components: a
ResNet-50 backbone network, DBSCAN clustering algorithm, and a reidentiﬁcation loss function 𝐿𝑅𝑒𝐼𝐷 . A lightweight baseline is more conducive to validating the potential of ADA framework.

The experimental data presented in Tables 4–8 demonstrate that the
ADA framework successfully fulﬁlls the ﬁrst four predeﬁned design objectives, as evidenced by a rigorous analytical evaluation. Tables 4–8
present the experimental results for ﬁve sparse self-attention variants
across various dataset conﬁgurations.
4.4.2. Performance degradation when substituting sparse self-attention
with convolution
We replaced the sparse self-attention inside SAP with conventional
2D 3 × 3 convolution, and kept everything else unchanged. This new
module repeated the experiments done by the ADA framework. The experimental results recorded in the Table 9 show that compared with
the ADA framework based on sparse self-attention, the performance and
generality of this module have signiﬁcantly decreased. The comparative
experiments further prove that SAP, SAAB block, and ADA framework
can signiﬁcantly improve domain adaptation performance.

4.4.1. The universality of the ADA framework
We systematically investigate the cross-paradigm compatibility of
the proposed framework through dual analytical dimensions: (1)
dataset generalization and (2) sparse self-attention architectural design
paradigms. Five representative sparse self-attention variants are progressively implemented within the ADA framework.
The ADA framework is designed with ﬁve objectives: (1) compatibility with diverse sparse self-attention design paradigms; (2) ﬂexible
deployment; (3) empirical eﬀectiveness across multiple datasets; (4)
simpliﬁed and uniﬁed parameter conﬁgurations; (5) multi-architecture
adaptability. Currently, no customized optimizations are implemented
for individual sparse self-attention variants. All design paradigms adopt
standardized external settings to enable three key advantages: paradigm
selection ﬂexibility, deployment simplicity, and parameter tuning eﬃciency. The framework prioritizes architectural versatility over performance consistency across varying sparse self-attention paradigms. Performance variations naturally occur when switching design paradigms.
The measured capability diﬀerences in the Tables 3 and 4–8 therefore

4.4.3. Applicability of backbone network
The ADA framework maintains robust domain adaptation performance after replacing its ResNet50 backbone with IBN-ResNet50 [34].
For empirical veriﬁcation, we employ the MSMT17 dataset-a highly
challenging cross-domain benchmark in visual recognition. Experimental results conﬁrm that the ADA framework achieves signiﬁcant performance improvements despite its stronger baseline. Detailed quantitative
metrics are provided in Table 10.
The ADA framework exhibits promising transferability to
transformer-based architectures. To validate this capability, we
systematically evaluate its integration within ViT [35] by deploying it
across distinct architectural stages. Empirical evidence demonstrates
that the ADA framework preserves its performance beneﬁts. Comprehensive testing across all transformer variants remains computationally
impractical. However, our methodology demonstrates that selective
placement at critical architectural nodes achieves optimal performance.
Table 11 records the performance data of eﬀective deployment points.

Table 3
The summary of peak performance of the ADA framework. (C → M : CUHK03 [24] to Market-1501 [25]).
(M → MS : Market-1501 [25] to MSMT17 [4]).
Setting

ResNet50 (Baseline)
Baseline + ADA (Liner Self-Attention)
Baseline + ADA (Permuted Self-Attention)
Baseline + ADA (Focused Linear Attention)
Baseline + ADA (Window based Self-Attention)
Baseline + ADA (Bi-Level Routing Attention)

C→M

M → MS

MS → M

mAP

R1

mAP

R1

mAP

R1

mAP

R1

56.7
73.2
69.0
70.6
71.9
68.1

77.3
87.9
85.7
86.7
86.8
85.4

7.2
10.8
9.5
9.0
9.2
9.2

5.8
9.0
8.5
7.6
7.9
7.2

11.9
17.0
15.7
16.0
14.0
14.7

27.3
36.5
34.0
34.1
31.1
31.4

70.4
78.9
78.7
80.2
75.6
79.2

86.1
91.2
91.1
91.8
88.9
90.4

6

M→C

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

Table 4
Ablation studies for the ADA framework with linear self-attention.

Table 8
Ablation studies for the ADA framework with window based self-attention.

ADA framework with Linear Self-Attention
Setting

Baseline
ADA after stage-0
ADA after stage-1
ADA after stage-2
ADA after stage-3
ADA after stage-4

C→M

ADA framework with Window based Self-Attention

M→C

M → MS

MS → M

mAP

R1

mAP

R1

mAP

R1

mAP

R1

56.7
53.3
73.2
72.3
66.8
54.4

77.3
73.7
87.9
87.5
82.7
73.9

7.2
6.2
7.6
8.0
10.8
5.1

5.8
4.9
5.6
6.4
9.0
3.9

11.9
14.4
17.0
14.3
13.3
10.9

27.3
32.6
36.5
31.3
28.3
24.4

70.4
69.0
78.9
78.0
77.8
71.8

86.1
85.5
91.2
90.4
89.8
86.2

Baseline
ADA after stage-0
ADA after stage-1
ADA after stage-2
ADA after stage-3
ADA after stage-4

Table 5
Ablation studies for the ADA framework with permuted self-attention.

Baseline
ADA after stage-0
ADA after stage-1
ADA after stage-2
ADA after stage-3
ADA after stage-4

C→M

M→C

M → MS

MS → M

mAP

R1

mAP

R1

mAP

R1

mAP

R1

56.7
51.5
69.0
63.5
61.0
54.2

77.3
72.5
85.7
81.9
78.1
73.5

7.2
6.0
7.7
7.6
9.5
5.5

5.8
4.4
6.0
6.7
8.5
4.5

11.9
11.8
15.7
12.9
13.9
12.0

27.3
27.3
34.0
28.3
30.4
26.9

70.4
64.3
75.1
78.7
73.2
71.2

86.1
82.2
89.3
91.1
87.9
86.6

Setting

Baseline
after stage-0
after stage-1
after stage-2
after stage-3
after stage-4

Baseline
ADA after stage-0
ADA after stage-1
ADA after stage-2
ADA after stage-3
ADA after stage-4

M→C

M → MS

MS → M

mAP

R1

mAP

R1

mAP

R1

mAP

R1

56.7
55.2
70.6
68.8
66.1
60.0

77.3
74.6
86.7
84.8
82.9
78.0

7.2
5.1
6.8
8.9
9.0
5.6

5.8
3.5
5.5
7.6
7.6
4.5

11.9
11.8
15.0
16.0
13.7
12.1

27.3
27.1
32.9
34.1
30.1
27.3

70.4
65.5
75.3
80.2
75.4
67.7

86.1
83.5
89.3
91.8
88.2
83.4

Setting

R1

mAP

R1

mAP

R1

mAP

R1

56.7
52.7
60.6
71.9
70.9
41.1

77.3
73.6
79.0
86.8
85.3
63.5

7.2
5.6
7.9
7.4
9.2
3.7

5.8
3.7
7.0
6.1
7.9
2.6

11.9
10.2
14.7
14.0
13.5
11.6

27.3
24.1
32.5
31.1
29.5
25.7

70.4
64.0
76.5
75.6
74.7
68.6

86.1
81.9
90.4
88.9
87.9
84.6

M → MS

MS → M

mAP

R1

mAP

R1

mAP

R1

mAP

R1

56.7
51.4
67.7
68.1
66.5
53.2

77.3
73.4
84.2
85.4
82.0
73.3

7.2
6.0
7.2
7.8
9.2
3.6

5.8
4.9
6.0
6.1
7.2
2.4

11.9
11.4
14.6
12.5
14.7
8.0

27.3
26.0
32.3
27.3
31.4
18.2

70.4
66.6
74.4
75.8
79.2
51.4

86.1
84.2
89.2
89.1
90.4
73.1

C→M

M→C

mAP

R1

mAP

R1

56.7
52.0
54.6
56.1
65.8
52.8

77.3
72.8
74.9
76.2
82.4
73.5

7.2
5.3
5.2
6.5
8.3
6.7

5.8
3.9
3.6
5.1
7.1
5.2

Setting
IBN-ResNet50 (high-performance baseline)
IBN-ResNet50 + ADA (Liner Self-Attention)
IBN-ResNet50 + ADA (Permuted Self-Attention)
IBN-ResNet50 + ADA (Focused Linear Attention)
IBN-ResNet50 + ADA (Window based Self-Attention)
IBN-ResNet50 + ADA (Bi-Level Routing Attention)

M → MS

MS → M

mAP

R1

mAP

R1

20.2
23.1
21.4
21.2
20.8
20.3

41.3
46.0
44.5
43.6
41.9
44.2

80.1
82.5
80.4
80.6
81.9
83.4

91.3
92.6
91.4
91.5
92.0
92.8

Table 11
Applicability of ViT backbone network. Note: ADA framework with Bi-level
routing attention. The setting of DBSCAN is not ﬁxed.

ADA framework with Bi-Level Routing Attention
M→C

mAP

ADA framework with ﬁve Sparse Self-Attention

Table 7
Ablation studies for the ADA framework with Bi-level routing attention.

C→M

MS → M

Table 10
Applicability of backbone network for the ADA framework on IBN-ResNet50.
(M → MS : Market-1501 [25] to MSMT17 [4]).

ADA framework with Focused Linear Attention
C→M

M → MS

Performance degradation when substituting sparse self-attention

Table 6
Ablation studies for the ADA framework with focused linear attention.

Setting

M→C

Table 9
Performance degradation when substituting sparse self-attention
with convolution. (C → M : CUHK03 [24] to Market-1501 [25]).

ADA framework with Permuted Self-Attention
Setting

C→M

Setting

replacing ResNet with ViT, ADA framework with Bi-Level Routing Attention
Baseline
ADA after stage-0
ADA after stage-1
ADA after stage-2
ADA after stage-3
ADA after stage-4

Setting

Baseline based on ViT
ADA framework after ViT Block3
ADA framework after ViT Block6
ADA framework after ViT Block8

The experimental results detailed in Tables 10–11 demonstrate the
framework’s attainment of the ﬁfth design objective, as evidenced by
thorough analytical validation.

C→M

MS → M

mAP

R1

mAP

R1

5.1
9.2
7.9
10.8

14.3
23.0
20.8
26.5

6.4
9.1
10.5
10.4

17.7
21.2
25.3
25.8

4.4.5. Multiple deployment and stacking of the ADA frameworks
Experimental results indicate that neither multiple deployments nor
multi-layer stacking of the ADA framework leads to further performance
enhancement. For our investigation, we randomly selected two sparse
self-attention variants(Bi-Level Routing Attention and Window based
Self-Attention) along with a high-performance baseline model(IBNResNet50). Two deployment strategies were implemented: (1) consecutive ADA framework deployment at two locations, and (2) ADA framework deployment at a single optimal location using double-stacked
SAAB blocks. Neither conﬁguration demonstrated superior performance
compared to single-instance deployment at an optimal location. The
Discussion section will analyze potential underlying causes for these
observations.

4.4.4. Analysis of the peak performance versus the optimal deployment
location
The optimal deployment position for maximum performance varies
across sparse self-attention types. This design philosophy explicitly tolerates inter-paradigm performance discrepancies in sparse attention
mechanisms as a necessary trade-oﬀ for achieving cross-scenario adaptability. Our analysis reveals this positional discrepancy stems primarily
from divergent design paradigms in sparse attention architectures. Additionally, parameter optimization levels in these models partially inﬂuence performance discrepancies between deployment locations.
7

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

4.4.6. Summary of ablation experiment
Experimental results demonstrate the strong universality of the ADA
framework. The framework exhibits compatibility with diverse backbone networks. Furthermore, it is not limited to any speciﬁc sparse selfattention mechanism, allowing ﬂexible selection according to application requirements.
The ADA framework maintains consistent performance enhancement
across diﬀerent design paradigms of sparse self-attention.
The ADA framework oﬀers considerable deployment ﬂexibility, with
eﬀective integration across diﬀerent network stages corresponding to
low-, medium-, and high-level features. This adaptability enables optimized performance across various architectural conﬁgurations.
Compared to standalone implementations in Transformer architectures [19], the ADA framework eﬀectively mitigates the performance
degradation typically associated with certain sparse self-attention mechanisms. For instance, while linear self-attention [19] in standard
Transformers incurs signiﬁcant performance drops and increased computational overhead, the ADA framework maintains competitive performance while preserving computational eﬃciency. This advantage is
particularly valuable for domain adaptation tasks, where the framework
demonstrates consistent performance improvement.

Fig. 4. The relationship between EPS of DBSCAN and mAP. (Red line: test on
baseline based on ResNet50, using the ADA framework with focused linear attention.) (Blue Line: test on baseline based on IBN-ResNet50, using the ADA
framework with bi-level routing attention). (For interpretation of the references
to color in this ﬁgure legend, the reader is referred to the web version of this
article.)

4.5. Parameters analysis
In addition to employing a bi-level bottleneck structure for parameter reduction, the ADA framework implements a simple conﬁguration
strategy for sparse self-attention. When window size and projection dimension are excluded, sparse self-attention requires only the speciﬁcation of multi-head, eliminating additional parameterization.
Comparative analysis reveals distinct window size conﬁgurations
across architectures: (1) SRFormer [20] utilizes 24 × 24 windows for Permuted Self-Attention; (2) Swin Transformer [23] employs 7 × 7 windows
for Window-based Self-Attention; (3) Flatten Transformer [21] adopts
7 × 7 windows for Focused Linear Attention. In contrast, the ADA framework implements more eﬃcient window conﬁgurations ( 4 × 4, 8 × 8,
and 4 × 4 respectively) when deploying these attention variants. For Linear Self-Attention, the projection dimensions are set to 128 and 256 a more computationally economical conﬁguration than Linformer’s implementation.
While larger window sizes typically enhance performance, the ADA
framework demonstrates that optimized smaller windows (compared
to SRFormer and Flatten Transformer) can simultaneously achieve
competitive performance and computational eﬃciency. This parameter

reduction contributes to the framework’s resource-eﬀectiveness without
compromising functionality.
4.5.1. Evaluation of parameters and computational complexity
We quantitatively evaluate the parameters and computational complexity of the ADA framework. The detailed data is shown in the
Table 12. From the perspective of performance improvement, when deploying a framework after a stage in the backbone network, the added
parameters and FLOPs are relatively reasonable compared to the scale
of the backbone network.
4.5.2. The relationship between EPS and mAP
Fig. 4 illustrates the EPS-mAP relationship, with curve analysis identifying an optimal EPS window for framework performance. Following
prior baselines, we evaluated EPS from 0.1 to 1.0 in 0.1 increments.
Experiments failed with 𝐸𝑃 𝑆 ∈ [0.1, 0.2, 0.8, 0.9, 1.0] due to insuﬃcient
clustering, while 𝐸𝑃 𝑆 ∈ [0.3, 0.4, 0.5, 0.6, 0.7] enabled successful execution. Performance improved progressively from 𝐸𝑃 𝑆 ∈ [0.3, 0.4, 0.5, 0.6],
peaking at EPS=0.6.

Table 12
Parameters and FLOPs of the ADA framework with ﬁve sparse self-attention VS ResNet50 stage
1/2/3. Calculation basis: deploy the ADA framework once and stack it once.
Part of Network

layer1
layer2
layer3
Deploy Position

after layer1
after layer2
after layer3
Deploy Position

after layer1
after layer2
after layer3

baseline

Deploy Position

Params (M)

FLOPs (G)

0.22
1.22
7.1

0.88
1.35
1.92

after layer1
after layer2
after layer3

Bi-Level Routing Attention
Params (M)

FLOPs (G)

0.43
1.7
6.8

1.76
1.74
1.74

Deploy Position

after layer1
after layer2
after layer3

Focused Linear Attention
Params (M)

FLOPs (G)

0.43
1.7
6.8

1.76
1.75
1.74

Deploy Position

after layer1
after layer2
after layer3
8

Liner Self-Attention
Params (M)

FLOPs (G)

1.99
2.07
6.77

2.13
1.92
1.77

Permuted Self-Attention
Params (M)

FLOPs (G)

0.42
1.67
6.68

1.71
1.71
1.71

Window based Self-Attention
Params (M)

FLOPs (G)

0.43
1.71
6.8

1.74
1.74
1.74


 succeeded in 0ms:
Pattern Recognition 180 (2026) 114100

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

A2HA: Attribute-aware hierarchical alignment for text–image person
re-identification
Ziyu Zhu a , Runmin Wang a ,∗, Han Xu a , Qiuju Dai a , Lingxin Cui a , Xingdong Song a ,
Congzhen Yu a , Xin Hu a , Changxin Gao b , Nong Sang b
a

School of Information Science and Engineering, Hunan Normal University, Changsha, 410081, Hunan, China

b School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan, 430074, Hubei, China

ARTICLE

INFO

Keywords:
Text–image person re-identification
Cross-modal alignment
Multi-scale alignment
Attribute-level enhancement
Semantic understanding

ABSTRACT
Text–Image Person Re-Identification (TI-ReID) aims to retrieve a target individual from an image gallery based
on auxiliary textual descriptions. Current approaches face two major limitations: (1) global matching methods
struggle to capture fine-grained attribute relations; and (2) textual descriptions are treated as a one-directional
auxiliary modality and are often dominated by global visual representations. To address these challenges,
we propose A2HA (Attribute-Aware Hierarchical Alignment), a framework designed for precise cross-modal
alignment through hierarchical optimization. Specifically, we present a dynamically weighted Hierarchical
Alignment (HA) framework that performs coarse-to-fine optimization across instance-, identity-, and attributelevel representations, enabling a unified modeling of global and fine-grained semantics. Rather than treating
objectives as an isolated combination of existing losses, this design leverages attribute-enriched benchmarks
to enable coordinated multi-level semantic alignment within a single integrated framework. To this end,
we propose a dual-dimensional and dual-directional attribute-pairing strategy, capturing semantic and modal
consistency while enabling bidirectional interaction between textual and visual attributes. Furthermore, we
extend three widely-used benchmarks by incorporating attribute-aware segmented images and annotations,
constructed with large language models and segmentation models, resulting in attribute-enriched versions of
CUHK-PEDES, ICFG-PEDES, and RSTPReid. Experimental results show that A2HA achieves improvements of
+5.02%, +4.68%, and +8.98% in mean Average Precision (mAP) on the three datasets, respectively. The code
will be released at https://github.com/ziyuzhu452-create/A2HA.

1. Introduction
Text–Image Person Re-Identification (TI-ReID) has demonstrated
significant research relevance in various security and surveillance applications [1]. In TI-ReID tasks, the model is designed to extract features from two distinct modalities and establish alignment between
them. The first challenge is bridging the modality gap to establish
unified representations. With the prevalence of Transformer [2], Vision
Transformer (ViT) [3], and Visual Language Pre-trained (VLP) models,
most current works [4–7] utilize these models to encode cross-modal information into a shared feature space to address this issue. The unified
feature space provides a gateway to alleviate modal disparities while
conducting global matching. The second challenge is alignment. Existing methods must not only bridge the modality gap but also address
the challenge of semantically aligning visual information with textual
descriptions [8,9]. As illustrated in Fig. 1(a), traditional methods [10]
that rely on global text features 𝐹𝑔𝑡 and global image features 𝐹𝑔𝑣

for alignment are inherently misleading. The global features primarily
capture coarse-grained semantics and tend to overlook fine-grained
attribute-level correspondences that are crucial for distinguishing visually similar identities. Consequently, the lack of explicit attribute-level
modeling weakens the semantic alignment between textual descriptions and corresponding visual parts, particularly in scenarios involving
subtle inter-person differences.
Although some attribute-aware methods [11–13] have considered
these issues, they have the following flaws: (1) Most of these works [12,
14] extract textual attributes and match them with global image features, which hinders their ability to achieve strong attribute-level alignment, as shown in Fig. 1(b). Noise information continues to play a
significant role in the recognition of specific attributes. This is mainly
because the semantic granularity of localized textual attributes is fundamentally mismatched with holistic visual representations, leading

∗ Corresponding author.

E-mail address: runminwang@hunnu.edu.cn (R. Wang).
https://doi.org/10.1016/j.patcog.2026.114100
Received 28 February 2026; Received in revised form 29 April 2026; Accepted 24 May 2026
Available online 29 May 2026
0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Fig. 1. A comparison of (a) traditional methods, (b) existing attribute-aware methods, and (c) our approach to aligning image–text features. 𝐹 𝑡 and 𝐹 𝑣 refer to
𝑡
𝑣
the textual and visual features. 𝐹𝑔𝑡 and 𝐹𝑔𝑣 stand for the global features of text and image. 𝐹𝑎𝑡𝑡𝑁
and 𝐹𝑎𝑡𝑡𝑁
represent the textual and visual attribute features of
the 𝑁𝑡ℎ attribute. GMP is the global max pooling.

• We propose A2HA, an attribute-aware hierarchical alignment
framework for TI-ReID, which jointly optimizes instance-level,
identity-level, and attribute-level objectives in a coarse-to-fine
manner, enabling balanced and fine-grained semantic alignment.
• To support effective attribute-level optimization, we identify and
formalize diverse attribute-level misalignment patterns in TIReID and introduce a dual-dimensional and dual-directional attribute pairing strategy, which provides explicit and balanced
cross-modal attribute supervision.
• We construct attribute-enriched versions of three public TI-ReID
benchmarks and demonstrate that A2HA achieves state-of-theart (SOTA) performance across multiple evaluation metrics on
CUHK-PEDES [17], ICFG-PEDES [18], and RSTPReid [19].

to ambiguous and noisy attribute-visual associations. (2) These methods [11,13] treat attributes as auxiliary rather than core alignment
targets and rely mostly on instance-level or identity-level constraints,
thus attribute information is implicitly optimized and easily overshadowed by dominant identity discrimination objectives. (3) Methods [11,
15,16] that segment both text and image attributes typically rely on
static attribute mapping or predefined structural priors about person attributes. Such designs assume fixed attribute importance across
samples and scenarios, ignoring the fact that different attributes contribute unequally under varying semantic and visual contexts. Without
systematic attribute-level optimization and dynamic weight adaptation, these methods struggle to flexibly emphasize discriminative attributes while suppressing less informative ones, resulting in rigid
and suboptimal attribute alignment. As a result, existing attributeaware approaches are limited in their ability to establish fine-grained
and balanced attribute-level alignment across modalities, limiting their
effectiveness in handling complex and subtle person distinctions.
To address the aforementioned challenges, we propose A2HA, an
attribute-aware hierarchical alignment framework built upon a VLP
backbone, which explicitly promotes fine-grained and balanced semantic alignment between textual descriptions and visual illustrations. At
the core of A2HA is a hierarchical alignment method that defines a unified coarse-to-fine optimization process across instance-, identity-, and
attribute-level representations. This multi-level formulation presents as
an explicitly coupled optimization scheme, where different semantic
levels are progressively refined under a shared alignment objective,
ensuring global identity consistency while progressively strengthening
attribute correspondence. As a result, A2HA enables robust cross-modal
alignment under complex and fine-grained scenarios.
To effectively support attribute-level alignment, we propose a dualdimensional and dual-directional attribute pairing strategy that defines
how image–text pairs are organized for optimization across both intramodal and inter-modal spaces. Serving as a fundamental basis for
contrastive and triplet-based objectives, this strategy goes beyond conventional instance-level pairing schemes based on global similarity
by explicitly modeling fine-grained attribute relations. It distinguishes
multiple attribute misalignment patterns, enabling the model to better identify hard false positives and false negatives at the attribute
level and to emphasize discriminative attributes during training. Furthermore, to provide reliable attribute-level supervision, we construct
attribute-enriched versions of three widely used TI-ReID benchmarks:
CUHK-PEDES [17], ICFG-PEDES [18], and RSTPReid [19]. By jointly
leveraging a powerful semantic segmentation model and a large language model, the resulting datasets offer precise and interpretable
attribute annotations, substantially reducing semantic ambiguity and
noise in cross-modal attribute alignment.
In summary, our key contributions are as follows:

2. Related works
2.1. Text–Image Person Re-Identification
Previous solutions for TI-ReID use ResNet [20] and LSTM [21]
as backbones for feature extraction, emphasizing semantic alignment
between textual tokens and visual patches. Li, et al. [17] introduces a
pioneering work with the release of the widely used CUHK-PEDES [17].
Subsequent studies [22,23] address limitations such as spatial insensitivity and an over-reliance on specific features by designing novel
loss functions and enhancing local dependency modeling. For instance,
Zhang, et al. [24] proposes Cross-Modal Projection Matching (CMPM)
and Classification (CMPC) losses to enhance discriminative feature
extraction. In recent years, with the prevalence of Vision Transformer
(ViT) [3] and BERT [25], most works adopt them as backbones. TransReID [26] is the first to apply ViT [3] to ReID, inspiring subsequent works [27–29] to explore its potential in both single and cross
modal person re-identification. In the field of text–image person reidentification, IVT [30] introduces two paradigms for implicit semantic
alignment, achieving finer and more accurate alignment between visual
and textual modalities. SDG [31] proposes a structure-level distribution guidance to calibrate instance-level feature bias which employs
inter-class and intra-class loss functions with an iterative optimization
mechanism. Meanwhile, TPAN [32] leverages aggregated text prototypes to guide the model to focus on the prime identity-related features
of target persons, mitigating the feature ambiguity caused by detection
misalignment. TP-LReID [33] proposes a novel lifelong learning method
to dynamically generate identity-consistent text prompts, enabling the
model to learn discriminative features across successive domains without catastrophic forgetting. Despite the effectiveness, these methods
still focus on the instance-level, overlooking the fine-grained alignment
among cross-modal attribute features.
2

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Fig. 2. The architecture of A2HA. The parameters in the segmentation model and LLM are frozen during training. All attribute sample pairs include both
𝑣
𝑡
inter-modal and intra-modal cases. 𝐹 𝑣 and 𝐹 𝑡 represent image and text features, while 𝐹𝑎𝑡𝑡
and 𝐹𝑎𝑡𝑡
are representation of attribute features for both image and
text inputs.

pre-trained semantic segmentation model Grounded SAM [39], both
with frozen parameters, to obtain attribute descriptions and segmenta𝑣
tion masks. In the visual branch, attribute-specific visual features 𝐹𝑎𝑡𝑡𝑞
𝑣
are obtained by fusing the global visual feature 𝐹𝑔 with the corresponding attribute mask under a ratio 𝑟. In parallel, textual attribute
descriptions are encoded to generate attribute-level textual features
𝑡 . Based on these attribute features, we employ a dual-dimensional
𝐹𝑎𝑡𝑡𝑞
and dual-directional attribute pairing strategy to categorize attribute
sample pairs across intra-modal and inter-modal spaces. The resulting
attribute pairs are then optimized through the Attribute-Level Alignment(ALA) module, which performs contrastive learning over diverse
attribute relations. Through the joint optimization of GLA and ALA,
A2HA achieves hierarchical alignment across instance, identity, and
attribute levels.

2.2. Attribute-aware methods
Numerous works [34,35] perform cross-modal alignment at the
global level. For example, IRRA [36] matches global image and text
representations based on feature similarity. To alleviate this limitation, several attribute-aware methods have been proposed [11,12,14].
ViTAA [11] pioneers attribute-level segmentation across visual and
textual modalities, but relies on NLKT-based noun phrase extraction,
which may fragment descriptive semantics, and adopts static attributevisual mappings that assume fixed attribute importance. APTM [13]
extracts attribute nouns to construct prompt-based representations;
however, decoupling attributes from their original textual context disrupts syntactic coherence and leads to misalignment when matching
attribute descriptions with global visual features. VGSG [15] further
explores attribute-aware alignment, yet still employs static attribute
mappings and performs alignment at a single semantic level, limiting
its ability to capture hierarchical and fine-grained attribute correspondence. BAMG [16] models attribute relations via graph convolution, but depends on predefined human structural priors, which may
constrain flexibility under diverse visual conditions.

3.2. Feature extraction
A2HA employs CLIP-ViT/B-16 [37] as image encoder and
BERT [25] as text encoder. In ViT-B/16 [3], the letter ‘B’ denotes the
base network scale, while ‘16’ indicates the use of a 16 × 16 pixel patch
size.
Image Encoder. The input image 𝐼 ∈ R𝐻×𝑊 ×𝐶 is divided into 𝑁
patches and passed through a linear projection layer with a learnable
position embedding 𝑃𝐸 . After concatenation with a [CLS] token, 𝐼 is
represented as 𝑉 ∈ R(𝑁+1)×𝑑 , where 𝑑 denotes the embedding dimension. The [CLS] token encodes the global information of all 𝑁 patches.
{ 𝑣
}
𝑣 , where
The extracted features are tokenized into 𝐹 𝑣 = 𝑓𝑐𝑙𝑠
, 𝑓1𝑣 , … , 𝑓𝑁
𝑣
𝑣
𝑓𝑖 represents the feature of the 𝑖𝑡ℎ patch and 𝑓𝑐𝑙𝑠 serves as the global
representation 𝐹𝑔𝑣 of the image.
Text Encoder. Following a similar approach as the image encoder,
the input text 𝑇 is tokenized by a token embedding layer. The textual
token is obtained through lower-cased Byte Pair Encoding (BPE). Starting with a [SOS] token and ending{with an [EOS] token,
we obtain
}
𝑡 , 𝑓 𝑡, … , 𝑓 𝑡
a sequence represented as 𝐹 𝑡 = 𝑓SOS
.
Additionally,
1
EOS
since we have adopted BERT as the backbone, we incorporate masked
language modeling to improve prediction accuracy.

3. Methodology
3.1. Overview
As illustrated in Fig. 2, A2HA follows an attribute-aware hierarchical alignment paradigm, consisting of a Global-Level Alignment
(GLA) branch and an Attribute-Level Alignment (ALA) branch. The two
branches collaboratively enforce coarse-to-fine cross-modal semantic
alignment. Given an input image–text pair, we adopt CLIP-ViT/B16 [37] as the backbone to extract visual and textual features, denoted
as 𝐹 𝑣 and 𝐹 𝑡 . The corresponding global representations 𝐹𝑔𝑣 and 𝐹𝑔𝑡
are first utilized to perform Global-Level Alignment(GLA), ensuring
instance-level and identity-level consistency across modalities. To enable fine-grained attribute-level alignment, the input texts and images
are further processed by a large language model ChatGPT [38] and the
3

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

3.3. Hierarchical alignment framework

3.3.3. Attribute-level alignment
However, instance-level alignment along with identity-level alignment are insufficient to resolve ambiguities caused by visually similar
pedestrians with subtle attribute differences. To address this limitation,
we introduce attribute-level alignment, which enforces fine-grained
correspondence across modalities. To effectively optimize attributelevel correspondence, it is crucial to explicitly identify and treat false
positive and false negative attribute pairs during training. To identify
false positive and false negative attribute pairs, we further design a
similarity-based attribute pairing strategy to categorize sample pairs
into positive, negative, and problematic cases.
Attribute Pairing Strategy: Some attribute-aware works explore
image–text attribute pairing for cross-modal alignment. ViTAA [11]
relies on identity-level supervision, while LAIP [40] introduces noun
phrases as attribute units. However, both approaches tend to fragment
textual descriptions, disrupting semantic integrity by discarding contextual modifiers. Although contrastive learning is employed, attribute
distance optimization is not explicitly disentangled across intra-modal
and inter-modal spaces. Consequently, alignment is mainly conducted
in a unidirectional manner from textual phrases to visual regions, without reciprocal validation from visual attributes to textual descriptions.
This limitation motivates the need for a systematic and bidirectional
attribute pairing strategy.
By using a similarity-based analysis, we categorize image–text attribute feature pairs into 12 cases, which are further grouped into 6
types of sample pairs. For the similarity metric, we use the cosine
similarity 𝑆(⋅), defined as follows:

The core of A2HA is a hierarchical alignment strategy composed
of Global-Level Alignment (GLA) and Attribute-Level Alignment (ALA).
Specifically, GLA integrates both instance-level and identity-level alignment to enforce global semantic and identity consistency, while ALA
focuses on fine-grained attribute correspondence. Through this hierarchical design, global identity consistency and local attribute alignment
are jointly optimized in a complementary manner rather than treated
independently.

3.3.1. Instance-level alignment
Instance-level alignment serves as the global semantic foundation
of our framework, ensuring that image–text pairs of the same instance
remain close in the shared embedding space. For instance-level alignment, we employ a bidirectional cross-entropy loss to enforce global
semantic consistency. As shown in Eq. (1), 𝑖𝑛𝑠 is composed of crossentropy losses computed bidirectionally: from image to text and from
text to image.
𝑖𝑛𝑠 =

)
( 𝑇
)]
1[ (
CE 𝑆𝐼2𝑇 , 𝑦 + CE 𝑆𝐼2𝑇
,𝑦 ,
2

(1)

𝑆𝐼2𝑇 =

(
)
(
)
1
1
𝑇
⋅ 𝑆 𝑓𝐼𝑔 , 𝑓𝑇𝑔 , 𝑆𝐼2𝑇
= ⋅ 𝑆 𝑓𝑇𝑔 , 𝑓𝐼𝑔 ,
𝜏
𝜏

(2)

(
)
⎛
⎞
𝐵−1
⎜ exp 𝑆𝑖,𝑦𝑖
⎟
1 ∑
CE(𝑆, 𝑦) = −
log ⎜ ∑
(
)
⎟,
𝐵−1
𝐵 𝑖=0
⎜ 𝑗=0 exp 𝑆𝑖,𝑗 ⎟
⎝
⎠

(3)

(
)
𝑗
𝑖
𝑆 𝑉𝑎𝑡𝑡𝑟
, 𝑇𝑎𝑡𝑡𝑟
=

𝑦 is a positive sample index vector of length 𝐵 (batch size), where
𝑦 ∈ [0, 𝐵 − 1] denotes the index of the text that forms a positive pair
with the 𝑖𝑡ℎ image. All texts with indices 𝑗 ≠ 𝑦𝑖 are treated as negative
samples for the 𝑖𝑡ℎ image. 𝑆𝐼2𝑇 represents the image-to-text similarity
𝑇
matrix, while 𝑆𝐼2𝑇
is its transpose, capturing text-to-image similarities
to enforce bidirectional alignment. CE(⋅) refers to the cross-entropy
loss. 𝜏 represents the temperature parameter, set to 0.07. 𝑓𝐼𝑔 and 𝑓𝑇𝑔
represent the global features of the image and text, respectively. For
the cross-entropy calculation, 𝑆𝑖,𝑗 denotes the similarity between the
𝑖𝑡ℎ image and the 𝑗𝑡ℎ text. 𝑆𝑖,𝑦𝑖 highlights the similarity score of the
positive pair for the 𝑖𝑡ℎ sample (i.e., when 𝑗 = 𝑦𝑖 ).

𝑁
(
)
1 ∑
log 𝑝𝑖,𝑦𝑖 ,
𝑁 𝑖=1

(5)

𝑖 denotes the visual feature representation of the 𝑖 sample’s
where 𝑉𝑎𝑡𝑡𝑟
𝑡ℎ
𝑗
attributes, and 𝑇𝑎𝑡𝑡𝑟
denotes the corresponding textual attribute feature
representation of the 𝑗𝑡ℎ sample. To enable fine-grained classification,
we introduce two thresholds 𝜃𝑝𝑜𝑠 and 𝜃𝑛𝑒𝑔 . The image–text attribute
sample pairs are categorized into 12 distinct cases based on these two
thresholds. To be specific, for image samples 𝑥, 𝑦 and text samples 𝑚,
𝑛, the attribute-specific features are extracted by performing feature
extraction on the segmented attributes. For example, the feature of the
𝑞𝑡ℎ attribute for the 𝑖𝑡ℎ image is denoted as 𝑥𝑖𝑎𝑡𝑡𝑞 . Pairwise similarity
measurement 𝑆(⋅) is then performed on these attribute features. Using
the thresholds 𝜃𝑝𝑜𝑠 and 𝜃𝑛𝑒𝑔 , we classify positive pairs, negative pairs,
false positive pairs, and false negative pairs, covering both intra-modal
and inter-modal cases. The detailed classification rules are provided
in Table 1. This achieves precise classification along two dimensions:
(1) semantic consistency, and (2) modal consistency among local attributes, categorizing false positives and false negatives as ‘problematic
samples’ that require strengthened constraints. Herein, we set 𝜃𝑝𝑜𝑠 =
0.5 and 𝜃𝑛𝑒𝑔 = 0.3 as the optimal values based on experimental
comparisons.
The intra-modal attribute contrastive loss 𝑖𝑛𝑡𝑟𝑎 , inter-modal attribute contrastive loss 𝑖𝑛𝑡𝑒𝑟 , and the attribute triplet loss 𝑡𝑟𝑖 constitute
the overall attribute-level constraints. For a given attribute 𝑞, we obtain
batches of false positive and false negative attribute pairs by using the
classification strategy in Table 1. To enhance the model’s ability to
identify problematic samples, we perform hard sample mining: from
false negative pairs, we select the batch pair with the smallest deviation
from the positive threshold 𝜃𝑝𝑜𝑠 as the hardest positives; similarly, from
false positive pairs, we select the pair with the largest deviation from
the negative threshold 𝜃𝑛𝑒𝑔 as the hardest negatives.
Intra-modal Attribute Loss: Intra-modal attribute alignment enforces the clustering of samples sharing the same attribute within each
modality. This additional explicit constraint encourages the model to
learn diverse stylistic variations of the same attribute, thereby improving its generalization ability to intra-modal appearance and description

3.3.2. Identity-level alignment
In addition to instance-level alignment, we introduce an explicit
identity-level alignment objective to further enhance identity discriminability in the shared embedding space. While instance-level alignment focuses on cross-modal instance correspondence, identity-level
alignment directly enforces class-level supervision by treating pedestrian identities as classification targets. Specifically, we adopt a standard identity classification loss to supervise the learning of identitydiscriminative features. The identity-level loss 𝑖𝑑 is defined as:
𝑖𝑑 = −

𝑖 ⋅ 𝑇𝑗
𝑉𝑎𝑡𝑡𝑟
𝑎𝑡𝑡𝑟
,
𝑗 ‖
‖𝑉 𝑖 ‖ × ‖
‖
‖ 𝑎𝑡𝑡𝑟 ‖ ‖𝑇𝑎𝑡𝑡𝑟 ‖
‖

(4)

where 𝑁 denotes the number of samples in a mini-batch, and 𝑝𝑖,𝑦𝑖
represents the predicted probability that the 𝑖𝑡ℎ sample belongs to its
ground-truth identity class 𝑦𝑖 .
Importantly, the identity label 𝑦𝑖 is consistent with the positive
sample indexing used in the instance-level alignment loss 𝑖𝑛𝑠 (Eq. (1)).
This design ensures coherent supervision across different alignment
levels, where instance-level learning encourages cross-modal instance
matching, while identity-level classification explicitly enlarges interclass margins and stabilizes training.
4

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Table 1
Attribute sample pair classification based on similarity measurement. 𝑥𝑎𝑡𝑡𝑞 and
𝑦𝑎𝑡𝑡𝑞 denote image samples of attribute 𝑞, while 𝑚𝑎𝑡𝑡𝑞 and 𝑛𝑎𝑡𝑡𝑞 represent text
samples of the same attribute 𝑞. The variables 𝑖 and 𝑗 represent person IDs,
and 𝑆(⋅) denotes the cosine similarity function.
pid

Similarity threshold
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑦𝑗𝑎𝑡𝑡𝑞 ≥ 𝜃𝑝𝑜𝑠
(
)
𝑗
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑦𝑎𝑡𝑡𝑞 < 𝜃𝑝𝑜𝑠
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑚𝑗𝑎𝑡𝑡𝑞 ≥ 𝜃𝑝𝑜𝑠
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑚𝑗𝑎𝑡𝑡𝑞 < 𝜃𝑝𝑜𝑠
(
)
𝑆 𝑚𝑖𝑎𝑡𝑡𝑞 , 𝑛𝑗𝑎𝑡𝑡𝑞 ≥ 𝜃𝑝𝑜𝑠
(
)
𝑆 𝑚𝑖𝑎𝑡𝑡𝑞 , 𝑛𝑗𝑎𝑡𝑡𝑞 < 𝜃𝑝𝑜𝑠
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑦𝑗𝑎𝑡𝑡𝑞 ≥ 𝜃𝑛𝑒𝑔
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑦𝑗𝑎𝑡𝑡𝑞 < 𝜃𝑛𝑒𝑔
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑚𝑗𝑎𝑡𝑡𝑞 ≥ 𝜃𝑛𝑒𝑔
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑚𝑗𝑎𝑡𝑡𝑞 < 𝜃𝑛𝑒𝑔
(
)
𝑆 𝑚𝑖𝑎𝑡𝑡𝑞 , 𝑛𝑗𝑎𝑡𝑡𝑞 ≥ 𝜃𝑛𝑒𝑔
(
)
𝑆 𝑚𝑖𝑎𝑡𝑡𝑞 , 𝑛𝑗𝑎𝑡𝑡𝑞 < 𝜃𝑛𝑒𝑔

𝑖=𝑗

𝑖≠𝑗

(belonging to a different identity but having high similarity score) as a
false positive and the text sample 𝑝 (sharing the same identity as 𝑖 but
having low similarity score) as a false negative, we apply the triplet loss
to strengthen the association between image sample 𝑖 and text sample
𝑝 while separating sample 𝑖 from 𝑛. The loss function is defined as
follows:

Category
Positive

𝐵

Intra False Negative

𝑞𝑡𝑟𝑖 =

Positive
Inter False Negative
Positive

𝐵𝑞 [ (
)
(
)
1 ∑
𝛼 𝜃pos − max S 𝑓𝐼𝑞 [𝑖], 𝑓𝐼𝑞 [𝑗]
𝑗∈𝐹 𝑁𝑖
𝐵𝑞 𝑖=1
(
)]
(
)
+ 𝛼 max S 𝑓𝐼𝑞 [𝑖], 𝑓𝐼𝑞 [𝑗] − 𝜃neg ,

Intra False Positive
Negative

3.3.4. Hierarchical loss
Based on the aforementioned classification, we propose a multiscale cross-modal composite loss that integrates contrastive learning, hard example mining, and attribute-adaptive weight adjustment.
This loss jointly optimizes fine-grained attribute alignment and global
matching, enabling more reliable discrimination among different attribute pair types. Instead of assuming uniform attribute
importance [11,13], the proposed weighting scheme adaptively adjusts
the contribution of each attribute during optimization, allowing the
model to focus more on ambiguous or misaligned attributes when necessary. Accordingly, false positive and false negative attribute pairs are
treated as problematic samples and penalized more heavily, ensuring
that features of the same pedestrian ID and attribute remain close
in both inter-modal and intra-modal spaces, while those of different
IDs and attributes are effectively separated. By integrating global-level
and attribute-level objectives, A2HA achieves hierarchical alignment in
which global identity consistency and local attribute correspondence
are optimized in a complementary manner.
Our hierarchical loss comprises instance-level, identity-level and
attribute-level losses. Specifically, 𝑖𝑛𝑠 in Eq. (1) establishes global
semantic constraints for visual and textual information; 𝑖𝑑 in Eq. (4)
achieves identity alignment, while 𝑎𝑡𝑡𝑟 enforces fine-grained attributelevel alignment for both inter-modal and intra-modal features. The
weights of 𝑖𝑛𝑠 , 𝑡𝑟𝑖 , 𝑖𝑛𝑡𝑒𝑟 , and 𝑖𝑛𝑡𝑟𝑎 are denoted as 𝜆𝑖𝑛𝑠 , 𝜆𝑡𝑟𝑖 , 𝑤𝑖𝑛𝑡𝑒𝑟 ,
and 𝑤𝑖𝑛𝑡𝑟𝑎 , respectively. Among them, 𝜆𝑖𝑛𝑠 and 𝜆𝑡𝑟𝑖 are treated as hyperparameters, while 𝑤𝑖𝑛𝑡𝑒𝑟 and 𝑤𝑖𝑛𝑡𝑟𝑎 are adaptively determined through
dynamic weight adjustment. The overall hierarchical loss function is
formulated as follows:

Inter False Positive
Negative
Intra False Positive
Negative

(6)

𝑗∈𝐹 𝑃𝑖

where 𝑞𝑖𝑛𝑡𝑟𝑎,𝐼 is the contrastive loss within images, similarly, 𝑞𝑖𝑛𝑡𝑟𝑎,𝑇
can be acquired by replacing (𝑓𝐼𝑞 [𝑖], 𝑓𝐼𝑞 [𝑗]) with (𝑓𝑇𝑞 [𝑖], 𝑓𝑇𝑞 [𝑗]). Herein,
𝐵𝑞 denotes the number of valid samples for attribute 𝑞. 𝛼(⋅) represents
the RELU activation function. 𝑓𝐼𝑞 [𝑖] represents the features of attribute 𝑞
for the 𝑖𝑡ℎ image, and 𝑓𝑇𝑞 [𝑗] represents the features of attribute 𝑞 for the
𝑗𝑡ℎ text. All features are processed through self-attention pooling and
feature fusion. 𝐹 𝑁𝑖 represents the collection of false negative samples
with the same ID as sample 𝑖, and 𝐹 𝑃𝑖 represents the collection of false
positive samples. 𝑆(⋅) represents the cosine similarity as formulated in
Eq. (5). Consequently, the overall intra-modal attribute contrastive loss
𝑞𝑖𝑛𝑡𝑟𝑎 for attribute 𝑞 is derived using Eq. (7).
(
)
1
𝑞𝑖𝑛𝑡𝑟𝑎 =
𝑞𝑖𝑛𝑡𝑟𝑎,𝐼 + 𝑞𝑖𝑛𝑡𝑟𝑎,𝑇 .
(7)
2
Inter-modal Attribute Loss: Inter-modal attribute alignment further enforces the alignment of samples with the same attribute across
different modalities in the shared embedding space. Although a shared
encoder implicitly maps different modalities into a common feature
space, it does not guarantee sufficiently compact alignment between semantically consistent instances. Therefore, this constraint is introduced
to explicitly form a tight cluster among cross-modal samples sharing the
same attribute, enhancing fine-grained semantic consistency beyond
global alignment. For inter-modal false positive and false negative
attribute pairs, we calculate 𝑞𝑖𝑛𝑡𝑒𝑟 for each attribute 𝑞 using a formula
similar to that in Eq. (6):
𝑞𝑖𝑛𝑡𝑒𝑟 =

𝐵𝑞 [ (
)
(
)
1 ∑
𝛼 𝜃pos − max S 𝑓𝐼𝑞 [𝑖], 𝑓𝑇𝑞 [𝑗]
𝑗∈𝐹 𝑁𝑖
𝐵𝑞 𝑖=1
(
)]
( 𝑞
)
+ 𝛼 max S 𝑓𝐼 [𝑖], 𝑓𝑇𝑞 [𝑗] − 𝜃neg .

(9)

where 𝜖 is the triplet margin, set as 0.2 based on experimental comparisons.

Intra False Negative

shifts. For intra-modal false positive and false negative attribute pairs,
we compute 𝑞𝑖𝑛𝑡𝑟𝑎 for each attribute 𝑞, as shown below:
𝑞𝑖𝑛𝑡𝑟𝑎,𝐼 =

𝑞
( (
(
))
1 ∑
max 0, 1 − S 𝑓𝐼𝑞 [𝑖], 𝑓𝑇𝑞 [𝑝]
𝐵𝑞 𝑖=1
(
(
))
)
− 1 − S 𝑓𝐼𝑞 [𝑖], 𝑓𝑇𝑞 [𝑛] + 𝜖 ,

𝑡𝑜𝑡𝑎𝑙 = 𝜆𝑖𝑛𝑠 ⋅ 𝑖𝑛𝑠 + 𝑖𝑑 + 𝑎𝑡𝑡𝑟 ,
𝑎𝑡𝑡𝑟 =

∑( 𝑞
)
𝑤𝑖𝑛𝑡𝑒𝑟 ⋅ 𝑞𝑖𝑛𝑡𝑒𝑟 + 𝑤𝑞𝑖𝑛𝑡𝑟𝑎 ⋅ 𝑞𝑖𝑛𝑡𝑟𝑎 + 𝜆𝑡𝑟𝑖 ⋅ 𝑞𝑡𝑟𝑖 ,

(10)
(11)

𝑞∈

 represents the set of attributes. 𝜆𝑖𝑛𝑠 and 𝜆𝑡𝑟𝑖 are both set to 0.5 based
on experimental comparisons, while 𝑤𝑞𝑖𝑛𝑡𝑒𝑟 and 𝑤𝑞𝑖𝑛𝑡𝑟𝑎 are determined by
the dynamic weight adjustment formula, as shown in Eq. (12).
(
)
Var 𝑞𝑖𝑛𝑡𝑟𝑎
𝑤𝑞𝑖𝑛𝑡𝑒𝑟 =
,
(
)
(
)
Var 𝑞𝑖𝑛𝑡𝑒𝑟 + Var 𝑞𝑖𝑛𝑡𝑟𝑎 + 𝜂
(12)
(
)
Var 𝑞𝑖𝑛𝑡𝑒𝑟
𝑤𝑞𝑖𝑛𝑡𝑟𝑎 =
,
( 𝑞 )
( 𝑞 )
Var 𝑖𝑛𝑡𝑒𝑟 + Var 𝑖𝑛𝑡𝑟𝑎 + 𝜂

(8)

where 𝜂 is a small value added to prevent division by zero, set to
1 × 10−8 , and Var(⋅) denotes the variance of the loss values, specifically
computed over all training loss values recorded at intervals of 100
batches within each individual epoch. 𝑤𝑞𝑖𝑛𝑡𝑒𝑟 and 𝑤𝑞𝑖𝑛𝑡𝑟𝑎 are both initiated
as 1 and subsequently updated at the conclusion of each training epoch.
Moreover, deviating from conventional average pooling, we leverage
self-attention pooling as a dedicated method for aggregating image–text
features.

𝑗∈𝐹 𝑃𝑖

All variable definitions follow the formula description of intramodal attribute loss (Eq. (6)) in the previous subsection.
Attribute Triplet Loss: After implementing the hard sample mining, the remaining false positive and false negative attribute pairs
are used in the attribute-aware triplet loss to strengthen cross-modal
alignment. For an image sample 𝑖, when classifying a text sample 𝑛
5

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

𝑣 . Herein, in this study, we select 𝑟 = 0.8 as the optimal value
feature 𝐹𝑎𝑡𝑡𝑞
based on experimental comparisons. In text attribute description, Fig. 3
reveals that, beyond clothing and accessories, gender is frequently mentioned. Based on this observation, we introduce a new ‘gender’ attribute
during text attribute segmentation to enhance attribute completeness.
Since gender identification requires comprehensive semantic analysis of
images, we align the textual gender attribute with global image features
to promote cross-modal consistency.
While our current framework is evaluated on a fixed set of attributes, it is important to discuss its extensibility to dynamic or larger
attribute sets. Our framework’s design inherently supports scalability
through several mechanisms. First of all, as each attribute operates
as an independent alignment unit with its own segmentation module,
adding a new attribute 𝑎𝑛𝑒𝑤 simply introduces an additional pairing
stream without affecting existing ones. The computational overhead
grows approximately linearly as (𝑛), where 𝑛 is the number of attributes. Secondly, the loss functions are designed to be attribute-aware
and naturally extendable. Our attribute-level loss Eq. (11) compute similarities independently for each attribute pair. When extending to more
attributes, the loss aggregates over all available attributes without requiring architectural changes. The threshold-based pairing mechanism
(𝜃𝑝𝑜𝑠 , 𝜃𝑛𝑒𝑔 ) remains applicable regardless of attribute count.

Fig. 3. Word cloud of the text in CUHK-PEDES [17] datasets. Larger font size
indicates higher occurrence frequency.

3.4. Attribute-enriched dataset construction
Despite the extensive collection of images and annotations in existing TI-ReID datasets, challenges such as interference, incorrect text annotations, and varied sentence structures remain, significantly limiting
the performance of TI-ReID models. By leveraging pre-trained semantic
segmentation models and large language models, we address these issues from both visual and textual perspectives, enabling more accurate
attribute segmentation. Specifically, we use Grounded SAM [39] as
the image attribute segmentation model and ChatGPT’s [38] multi-turn
dialogue for text attribute segmentation. We apply these segmentation
methods to three benchmarks: CUHK-PEDES [17], ICFG-PEDES [18],
and RSTPReid [19], resulting in attribute-enriched datasets.
To ensure controlled experiments and facilitate reproducible comparisons, we adopt a predefined closed-set attribute space, selecting
five core visual attributes: hair, upper body, lower body, bag, and
shoes. This choice is motivated by three factors: (1) To validate the
effectiveness of the hierarchical alignment framework under controlled
conditions, reducing complexity; (2) To align with common practices in
TI-ReID [11,13], where constrained attribute spaces are typically used
for benchmark comparisons; (3) As shown in Fig. 3, our word frequency
analysis reveals that these five attributes make up the majority of
descriptive nouns in text annotations and occupy significant, easily
locatable regions in images.
We also account for less prominent and infrequently mentioned attributes. While Grounded SAM [39] and ChatGPT excel in segmentation
and language understanding, challenges arise when dealing with absent
or small attributes (e.g., shoes). To avoid information loss, attributes
not mentioned in text are marked as NULL, and full text descriptions
are used for matching. For image attributes, we perform a weighted
fusion of the segmented attribute features with the original image
features to balance local details and global context. Specifically, the
fused visual feature representation is defined as:
𝑣̂ = 𝑀 ⊙ 𝐹 𝑣 ,
𝐹𝑎𝑡𝑡𝑞
𝑞
𝑔

(13)

𝑣
𝑣̂ + (1 − 𝑟) × 𝐹 𝑣 ,
𝐹𝑎𝑡𝑡𝑞
= 𝑟 × 𝐹𝑎𝑡𝑡𝑞
𝑔

(14)

3.4.1. Visual attribute segmentation
The pre-trained Grounded SAM in our work combines the object
detection capability of Grounding DINO [41] and the segmentation
capability of SAM [42], enabling a two-step segmentation process: detection followed by fine-grained segmentation. With the powerful zeroshot detection capability of Grounding DINO, Grounded SAM can locate
any object in an image using text prompts and then perform precise
segmentation with SAM’s robust capabilities. We validate segmentation
accuracy by using detection confidence scores from Grounding DINO.
For small targets (e.g., ‘shoes’), we apply a lower confidence threshold
to accommodate their inherently lower signal-to-noise ratio and avoid
information loss and excessive NULL values. Specifically, the confidence threshold is set to 0.1 for small targets and 0.3 for larger targets.
Qualitative results are shown in Fig. 7, displaying detection confidence
scores and corresponding segmentation outputs. The resulting attributeenriched datasets include comprehensive detection confidence scores
and segmentation results, which will be publicly available.
3.4.2. Textual attribute extraction
To generate text annotations for attribute labeling via large language models, we provide the following prompt to ChatGPT:
Please extract the ‘attribute’ field from the content of the ‘captions’ field
while retaining all other original fields. The required ‘attributes’ include
‘hair’, ‘upper body’, ‘lower body’, ‘bag’, ‘shoes’, and ‘gender’. Output ‘null’
for content not mentioned in the text; return ‘null’ for ambiguous gender
references. Do not add any extra attribute fields.
We limit our text attribute classification to six categories based
on two key considerations:(1) Minor features like glasses or phone
pose challenges for image segmentation accuracy, and including them
would result in a high incidence of NULL values during text attribute
parsing; (2) Overemphasizing such features may bias the model toward
non-critical information, increasing the risk of overfitting. This sixattribute framework strikes a balance by ensuring effective extraction
of textual descriptions for core features while preserving the semantic
coherence of descriptive sentences. For example, in processing a phrase
like ‘a T-shirt with a flower pattern’ a large language model can
map this description to the upper body attribute, while traditional
tokenization methods may break it into fragmented noun combinations,
undermining semantic integrity.

where 𝑀𝑞 is the binary mask of the 𝑞𝑡ℎ attribute, and 𝐹𝑔𝑣 represents
the global image feature. By performing element-wise multiplication ⊙
between the 𝑀𝑞 and the 𝐹𝑔𝑣 , we obtain the attribute-specific feature
̂𝑣 for the 𝑞 attribute. For comprehensive feature
representation 𝐹𝑎𝑡𝑡𝑞
𝑡ℎ
fusion, Eq. (14) integrates attribute-specific information with global
context through a balancing ratio 𝑟 to obtain the refined attribute
6

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Table 2
Comparison between proposed A2HA and other SOTA methods on CUHK-PEDES, ICFG-PEDES and RSTPReid.
Methods

Source

CUHK-PEDES [17]

ICFG-PEDES [18]

RSTPReid [19]

R-1

R-5

R-10

mAP

R-1

R-5

R-10

mAP

R-1

R-5

R-10

mAP

ViTAA [11]
LBUL [43]
CAIBC [44]
IVT [30]
LERF [9]
BDNet [45]
UniPT [35]
APTM [13]
TBPS-CLIP [46]
MUM [47]
PLIP-SwinB [48]
RDE [49]
WoRA [50]
HAM-PEDES [51]
RMGNet [52]
MAPS [8]
MLLM [53]

ECCV’20
ACM MM’22
ACM MM’22
ECCVW’22
PR’25
PR’23
ICCV’23
ACM MM’23
AAAI’24
AAAI’24
NeurIPS’24
CVPR’24
WWW’25
CVPR’25
TCSVT’25
PR’24
CVPR’24

54.92
64.04
64.43
65.59
65.84
66.27
68.50
76.53
73.54
74.25
75.36
75.94
76.38
77.71
77.19
77.22
78.13

75.18
82.66
82.87
83.11
84.24
85.07
84.67
90.04
88.19
89.83
90.86
90.14
89.72
91.42
92.18
–
91.19

82.90
87.22
88.37
89.21
90.22
90.27
90.38
94.15
92.35
93.58
94.87
94.12
93.49
94.57
–
96.90
94.50

51.60
–
–
–
–
–
–
66.91
65.38
66.15
–
67.56
67.22
69.68
70.59
–
68.75

–
–
–
56.04
57.23
57.31
60.09
68.51
65.05
65.62
66.17
67.68
68.35
68.25
68.35
57.22
69.37

–
–
–
73.60
76.64
76.15
76.19
82.99
80.34
80.54
83.37
82.47
83.10
83.30
83.06
–
83.55

–
–
–
80.22
83.11
81.58
82.46
87.56
85.47
85.83
88.94
87.36
87.53
88.15
–
82.70
88.18

–
–
–
–
–
–
–
41.22
39.83
38.78
–
40.06
42.60
42.30
41.62
–
42.42

–
45.55
47.35
46.70
46.75
–
51.85
67.50
62.10
63.40
–
65.35
66.85
71.69
63.67
–
69.95

–
68.20
69.55
70.00
71.30
–
74.85
85.70
81.90
83.30
–
83.95
86.45
87.85
84.59
–
87.35

–
77.85
79.00
78.80
81.60
–
82.85
91.45
87.75
90.30
–
89.90
91.10
93.30
–
–
92.30

–
–
–
–
–
–
–
52.56
48.00
49.28
–
50.88
52.49
55.19
49.58
–
54.17

CLIP [37]
CLIP+A2HA(Ours)
IRRA [36]
IRRA+A2HA(Ours)

ICML’21
–
CVPR’23
–

66.34
78.63
73.38
84.06

84.18
90.10
89.93
95.68

85.98
91.23
93.71
97.87

64.92
67.97
66.13
71.15

59.01
66.99
63.46
69.76

75.96
80.84
80.25
84.16

82.18
86.75
85.82
89.05

36.91
39.03
38.06
42.74

57.65
66.82
60.20
70.65

79.83
85.11
81.30
88.50

85.96
89.12
88.20
93.95

44.54
51.72
47.17
56.15

The best results are in bold and the second best are underlined.

4. Experiments

4.3. Comparison with state-of-the-art methods
4.3.1. Comparison on main metrics
Based on the statistics presented in Table 2, the hierarchical finegrained semantic alignment method proposed by our method is both
effective and superior.
On CUHK-PEDES [17], as shown in Table 2, our A2HA achieves
SOTA by outperforming all competitors across multiple metrics: it
yields relative improvements of +10.68%, +5.75%, +4.16%, and
+4.91% in Rank-1, Rank-5, Rank-10, and mAP over the baseline
IRRA [36], surpasses the most recent HAM-PEDES [51] by at least
+1% across all metrics, and outperforms other attribute-aware methods
like ViTAA [11] and APTM [13] on mAP by +19.55% and +4.24%,
respectively. We apply our A2HA to CLIP [37] and IRRA [36], evidence
show that with the addition of A2HA, the performance have been
notably enhanced. As the proposed hierarchical alignment relies on
attribute annotations, IRRA and CLIP, both of which lack inherent
attribute sets, require the joint use of hierarchical alignment (HA) and
our attribute-enriched datasets to evaluate the effectiveness of A2HA.
To further disentangle the contributions of the alignment strategy and
the dataset itself, we conduct additional experiments on ViTAA by
separately applying the alignment method and substituting the original
attribute sets with our enriched annotations. As shown in Table 3,
replacing the attribute sets leads to a +4.3% improvement in mAP,
while incorporating hierarchical alignment brings an additional gain of
+4.87%. Moreover, unlike methods [13,51,53] that employ synthetic
datasets for data augmentation, our approach fundamentally preserves
the original dataset content, thereby avoiding the introduction of
hallucinated information generated by the LLM. At the same time,
our method also achieves superior performance compared to these
approaches.
On ICFG-PEDES [18], as shown in Table 2, our A2HA outperforms
the most recent SOTA methods by at least +1%, further validating that
robust attribute-level alignment benefits TI-ReID even in more complex
datasets. Specifically, with the proposed A2HA, we achieve the mAP
accuracy of 42.74%, outperforms the baseline IRRA by +4.68%. Furthermore, with different baselines, A2HA also shows significant gains.
ICFG-PEDES dataset contains more complex real-world scene images,
which poses significant challenges for accurate pedestrian recognition.
Consequently, while our A2HA method demonstrates substantial improvements on CUHK-PEDES, its performance gain on ICFG-PEDES is

4.1. Datasets
CUHK-PEDES [17] is the first dataset designed for the TI-ReID task.
It contains over 40,000 pedestrian images and more than 80,000 natural language descriptions. The training set consists of 9003 identities,
26,633 images, and 26,633 textual descriptions. The validation and
test sets contain 6548 and 7025 images, and 6548 and 7025 textual
descriptions, respectively, with both sets comprising 2000 identities.
ICFG-PEDES [18] is subsequently released to enhance the network’s adaptability to real-world applications. It includes more complex
scenes, presenting a more challenging cross-modal matching task for
the network. The dataset contains a total of 54,522 images of 4102
identities, with each image having only one corresponding textual
description.
RSTPReid [19] comprises 20,505 images of 4101 unique identities.
Each identity is represented by 5 images taken from various camera
angles, and each image is annotated with 2 detailed textual descriptions. The training, validation, and test sets contain 3701, 487, and 200
identities, respectively.

4.2. Experimental settings
Our experiments are conducted using a single A800 GPU and Python
3.10. All images are resized to 384 × 128 pixels, and all text sequences
are resized to a length of 77 tokens. We apply data augmentation techniques, including standard image augmentation and random masking.
We employ the Adam optimizer for training, initializing the learning
rate to 1 × 10−5 with cosine decay. The remaining settings follow
those outlined in IRRA [36]. We employ Rank-𝑘 (𝑘 = 1,5,10) and
mean Average Precision (mAP) as the evaluation metrics. Additionally,
we also conduct comparison for the auxiliary metric, mean Inverse
Negative Penalty (mINP) [36]. For parameters, the values of 𝜆𝑖𝑛𝑠 , 𝜆𝑡𝑟𝑖 ,
𝜖, 𝑟, 𝜃𝑝𝑜𝑠 , and 𝜃𝑛𝑒𝑔 are obtained from ablation experiments, and the
remaining hyperparameters follow the settings of the baseline network
IRRA [36].
7

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Table 3
Effect of attribute-enriched datasets and hierarchical alignment on ViTAA over CUHK-PEDES.
Model

HA

Img Attr

Text Attr

ViTAA [11]
ViTAA
ViTAA
ViTAA

×
×
✓
✓

ViTAA
A2HA
ViTAA
A2HA

ViTAA
A2HA
ViTAA
A2HA

CUHK-PEDES [17]
R-1

R-5

R-10

mAP

54.92
59.31(+4.39)
60.62(+5.70)
66.41(+11.49)

68.79
73.29(+4.50)
74.13(+5.34)
79.65(+10.86)

75.78
80.99(+5.21)
81.01(+5.23)
86.08(+10.30)

51.60
55.90(+4.30)
56.47(+4.87)
60.90(+9.30)

Values in parentheses indicate performance gains.

Table 4
Comparison of A2HA with other SOTA methods on CUHK-PEDES, ICFG-PEDES and
RSTPReid on the mINP.
Method

Source

CUHK-PEDES [17]

ICFG-PEDES [18]

RSTPReid [19]

IRRA [36]
TBPS-CLIP [46]
RDE [49]
IRRA+A2HA(Ours)

CVPR’23
AAAI’24
CVPR’24
–

50.24
52.58
51.44
51.91

8.96
–
7.87
9.81

34.45
–
28.08
35.39

The best results are in bold and the second best are underlined.

Table 5
Ablation study on the effectiveness of the losses in attribute-level alignment.
ALA
#

M1
M2
M3
M4
M5
M6
M7

CUHK-PEDES [17]

ICFG-PEDES [18]

R-1

mAP

R-1

mAP

Baseline

73.38

66.13

63.46

38.06

✓

76.97
77.98
75.72
82.87
81.64
81.79
83.86

68.17
68.43
67.36
70.17
69.00
69.83
70.25

64.89
63.92
63.87
66.26
65.69
66.90
67.97

39.50
38.95
39.11
40.89
40.42
40.55
41.87

𝑖𝑛𝑡𝑒𝑟

𝑖𝑛𝑡𝑟𝑎

𝑡𝑟𝑖

✓
✓
✓
✓
✓

✓
✓
✓

✓
✓
✓

Table 6
Ablation study on the effectiveness of the component of hierarchical alignment.
ALA
M1
M2
M3
M4
M5
M6
M7

GLA
✓
✓
✓
✓
✓
✓
✓

CUHK-PEDES [17]

ICFG-PEDES [18]

R-1

mAP

R-1

mAP

80.83
81.04
80.98
83.87
83.03
83.55
84.06

69.40
69.97
69.45
71.01
70.78
70.59
71.15

65.92
64.97
65.58
68.01
66.09
67.79
69.76

40.08
39.45
39.91
41.95
41.02
41.20
42.74

The best results are in bold and the second best are underlined.

The best results are in bold and the second best are underlined.

4.4. Ablation studies

Results of the combination of ALA: In ALA, we evaluate three individual losses (𝑖𝑛𝑡𝑒𝑟 , 𝑖𝑛𝑡𝑟𝑎 , 𝑡𝑟𝑖 ) and their combinations, with detailed
findings reported in Table 5. The standalone use of each loss yields
modest improvements, while their combination significantly enhances
performance by deepening the model’s ability to distinguish positive
and negative attribute pairs. Notably, 𝑖𝑛𝑡𝑒𝑟 and 𝑖𝑛𝑡𝑟𝑎 are critical, as
they enable the model to prioritize key samples through hard sample mining and dynamic weight adjustment, thereby boosting overall
performance. When 𝑖𝑛𝑡𝑒𝑟 and 𝑖𝑛𝑡𝑟𝑎 are jointly utilized, the model exhibits the strongest efficacy in intra-modal and inter-modal contrastive
learning among multiple attributes partitioned from instances of the
same identity. Specifically, their synergistic application yields superior
performance compared to M5 and M6, which only employ a single loss
function.
Results of the addition of GLA: Table 6 illustrates the optimization
effect of hierarchical alignment (HA) when incorporating global-level
alignment (GLA). Building upon the results in Table 5, the addition of
global alignment enables the model to achieve effective alignment at
both coarse-grained and fine-grained levels. The performance trends
in Table 6 are generally consistent with those in Table 5. M4, which
concurrently applies 𝑖𝑛𝑡𝑒𝑟 , 𝑖𝑛𝑡𝑟𝑎 and GLA, achieves suboptimal performance. The optimal result is attained after integrating the attributelevel triplet loss 𝑡𝑟𝑖 . Furthermore, each method exhibits measurable
improvements in their corresponding rows of Table 6, demonstrating that the incorporation of global multi-level optimization yields
substantial benefits for alignment performance.

4.4.1. Effectiveness of different alignment methods
We conduct ablation studies to validate the effectiveness of key
alignment components in A2HA: global-level alignment (GLA) with 𝑖𝑛𝑠
and 𝑖𝑑 , and attribute-level alignment (ALA) with 𝑎𝑡𝑡𝑟 . Results for
these analyses are presented in Tables 5 and 6.

4.4.2. Effectiveness of values of hyper-parameters
We evaluate the effects of 𝜆𝑖𝑛𝑠 , 𝜆𝑡𝑟𝑖 , 𝜖, 𝑟, 𝜃𝑝𝑜𝑠 , 𝜃𝑛𝑒𝑔 , and the confidence
thresholds for the image segmentation on model performance. Through
a series of hyperparameter sensitivity analyses, we demonstrate the
effectiveness of the hyper-parameters and the rationality of their chosen

relatively modest. Nevertheless, it still achieves state-of-the-art performance and outperforms attribute alignment-based approaches such as
ViTAA [11] and APTM [13].
On RSTPReid [19], our A2HA achieves best or comparable results:
while slightly trailing the most recent work [50] in Rank-1, it excels
in all other metrics, with mAP outperforming SOTA by +0.96% and
demonstrating significant gains over the baseline IRRA [46], with the
improvement of +10.45% and +8.98% on Rank-1 and mAP. The
RSTPReid dataset exhibits a smaller scale. However, our method still
achieves exceptional performance. This indicates that A2HA is capable
of effectively operating under limited data.
4.3.2. Comparison on auxiliary metric mINP
Our A2HA demonstrates superior performance in terms of the mINP
metric. As shown in Table 4, across the CUHK-PEDES, ICFG-PEDES, and
RSTPReid datasets, it shows strong competitiveness. On CUHK-PEDES,
with a score of 51.91%, it outperforms IRRA [46] and RDE [49],
though slightly trailing TBPS-CLIP [46]. On ICFG-PEDES, its result
surpasses IRRA and RDE. On RSTPReid, A2HA achieves 35.39%, it
beats IRRA and RDE. Overall, our A2HA maintains stable and competitive performance across datasets, with notable advantages on ICFGPEDES and RSTPReid, effectively validating its good adaptability and
recognition capability in text–image person re-identification tasks.

8


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'A training-free framework for text-to-image person re-identification via query-prototype matching.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 190ms:
nd run on a single NVIDIA
A100 with 40 GB memory.

where 𝑦 = 𝑓𝑉 (𝑔) for T2I and 𝑦 = 𝑃𝑘 for T2P. The discrimination score
is
𝐷
− 𝐷intra
Disc = inter
.
(17)
𝐷inter + 𝐷intra
Since 1 − cos ∈ [0, 2], the score lies in [−1, 1]. As shown in Fig. 4,
T2P reduces 𝐷intra and increases 𝐷inter , yielding a higher Disc than
T2I. This stronger separation between matched and mismatched pairs
is consistent with the observation that T2P achieves higher mAP across
datasets. However, Fig. 5 shows that T2P does not always surpass
T2I in Rank-1 accuracy: prototype similarity is robust in a global,
distributional sense but may lose some fine-grained instance cues that
affect the very top-1 position. In addition, the trajectories in Figs. 5
and 6 indicate that T2P and T2I improve in tandem during training,
suggesting that prototypes capture complementary but not redundant
information.
These findings support our design choice in PDE: rather than replacing instance-level Text-to-Image similarity with a prototype-only
retrieval strategy, we treat Text-to-Prototype similarity as an auxiliary
prior and fuse it with T2I at inference time. In this way, PDE benefits
from the global ranking stability of prototypes while retaining the
fine-grained discrimination of instance-level similarities.

4.3. Comparison with state-of-the-art
In this section, we evaluate the effectiveness of our proposed
training-free framework by applying the PDE and PDE𝑀 to two representative backbones: CLIP and ALBEF. We demonstrate the improvements achieved by our methods on three benchmark datasets: CUHKPEDES, ICFG-PEDES, and RSTPReid. As shown in Table 1, our method
produces significant improvements, particularly on the ICFG-PEDES
dataset, which contains over 19 samples per class, significantly surpassing the 3 samples per class in CUHK-PEDES and the 5 samples per
class in RSTPReid.
For the ALBEF backbone, PDE and PDE𝑀 yield improvements of
1.1%/1.24% in Rank-1 accuracy and 3.72%/5.56% in mAP over the
RaSa model, and 0.99%/1.31% in Rank-1 and 3.38%/3.9% in mAP over
MARS. For the CLIP backbone, PDE and PDE𝑀 achieve 1.22%/1.36%
improvements in Rank-1 and 3.79%/5.77% in mAP over IRRA, and
0.68%/0.76% improvements in Rank-1 and 4.83%/5.42% in mAP over
HAM.

4. Experiment
4.1. Datasets and metrics
Our experiments are conducted on three widely used benchmark
datasets for Text-to-Image Person Re-identification: CUHK-PEDES [1],
ICFG-PEDES [37], and RSTPReid [38], which are all integral to evaluating TI-ReID methods.
6

Pattern Recognition 179 (2026) 113705

H. Yang et al.

Fig. 5. Rank-1 accuracy of IRRA on three datasets under three inference modes during training.

Fig. 6. mAP of IRRA on three datasets under three inference modes during training.
Table 1
Performance gains across different backbones.
Method

Image encoder

Text encoder

CUHK-PEDES

ICFG-PEDES

RSTPReid

R1

mAP

R1

mAP

R1

mAP

BERT-base
BERT-base
BERT-base
BERT-base
BERT-base
BERT-base
BERT-base
BERT-base

76.51
76.62+0.11
76.81+0.3
76.92+0.41
77.62
77.70+0.08
77.8+0.18
77.85+0.23

69.38
69.39+0.01
70.47+1.09
71.38+2
71.41
71.45+0.04
72.71+1.3
73.7+2.29

65.28
65.57+0.29
66.39+1.1
66.52+1.24
67.60
67.84+0.24
68.59+0.99
68.91+1.31

41.29
42.42+1.13
45.01+3.72
46.85+5.56
44.93
45.97+1.04
48.31+3.38
48.83+3.9

66.90
66.90
67.15+0.25
67.37+0.47
67.55
67.54−0.01
67.86+0.31
68.05+0.5

52.31
52.29−0.02
54.11+1.8
55.02+2.71
52.92
52.93+0.01
55.12+2.2
55.93+3.01

CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer

73.50
58.11−15.39
73.8+0.3
73.85+0.35
75.75
74.69−1.06
75.97+0.22
75.9+0.15
77.71
75.77−1.94
78.02+0.31
78.1+0.39

66.30
60.71−5.59
67.83+1.53
68.56+2.26
67.59
66.35−1.24
69.42+1.83
69.54+1.95
69.68
65.89−3.79
71.57+1.89
72.19+2.51

63.46
62.26−1.2
64.68+1.22
64.82+1.36
67.47
67.58+0.11
67.99+0.52
68.08+0.61
68.25
67.10−1.15
68.93+0.68
69.01+0.76

38.06
39.43+1.37
41.85+3.79
43.83+5.77
40.20
41.23+1.03
42.68+2.48
44.15+3.95
42.30
45.47+3.17
47.13+4.83
47.72+5.42

60.85
47.95−12.9
61.55+0.7
61.75+0.9
65.98
60.71−5.27
66.2+0.22
67.15+1.17
71.69
56.80−14.89
73.25+1.56
73.45+1.76

47.17
45.16−2.01
48.53+1.36
49.21+2.04
52.04
51.17−0.87
53.68+1.64
54.11+2.07
55.19
54.29−0.9
58.36+3.17
58.55+3.36

With ALBEF [5] backbone
RaSa [20]
RaSa-rerank
RaSa-PDE
RaSa-PDE𝑀
MARS [22]
MARS-rerank
MARS-PDE
MARS-PDE𝑀

CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT

With CLIP [4] backbone
IRRA [23]
IRRA-rerank
IRRA-PDE
IRRA-PDE𝑀
RDE [24]
RDE-rerank
RDE-PDE
RDE-PDE𝑀
HAM [28]
HAM-rerank
HAM-PDE
HAM-PDE𝑀

CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT

Notably, MARS is derived from RaSa, while RDE and HAM are
built upon IRRA. Our methods improve both models, highlighting that
modality gaps persist even in complexly modified or pretrained VLMs.
By leveraging image modality priors, our method mitigates the impact
of these gaps, resulting in consistent improvements. These findings
suggest that our approach can be broadly beneficial to other VLM-based
TI-ReID models.
When combined with our method, the best-performing models are
MARS-PDE𝑀 and HAM-PDE𝑀 . To further demonstrate their effectiveness, we compare them with recent state-of-the-art approaches on the
CUHK-PEDES, ICFG-PEDES, and RSTPReid datasets. As shown in Table
2, the results of these methods are quoted from the original papers for
fair comparison. Our method achieves competitive or superior performance across all benchmarks. Specifically, MARS-PDE𝑀 surpasses third
rank 4.02% mAP on CUHK-PEDES and 6.53% mAP on ICFG-PEDES.
HAM-PDE𝑀 surpasses third rank 0.39% Rank-1 on CUHK-PEDES, and
1.76% on RSTPReid, along with a 3.36% gain in mAP on the latter.

We further evaluate k-reciprocal re-ranking [34], a standard postprocessing method in image-based person ReID. As shown in Table 1,
it brings only marginal gains for ALBEF-based models and can even
degrade CLIP-based baselines. This suggests that neighborhood-based
refinement, which relies on reliable local structure, is less suitable for
TI-ReID because text queries and image galleries lie in heterogeneous
embedding spaces. In contrast, PDE is not a neighborhood-propagation
or diffusion-based re-ranking strategy, but an identity-level, prototypebased prior that uses global anchors derived from the image gallery
to calibrate Query–Gallery similarity. By avoiding noisy cross-modal
neighborhoods, it consistently improves retrieval quality across all
three benchmarks without retraining the backbone.
4.4. Ablation study
4.4.1. Analysis of scaling factor 𝜆 and cluster number 𝑁
We analyze the impact of cluster number 𝑁 and scaling factor 𝜆
on PDE across datasets. The results show that 𝑁 is the primary factor
7

Pattern Recognition 179 (2026) 113705

H. Yang et al.

Table 2
Comparisons with the state-of-the-art Re-ID methods.
Method
Han et al. [2]
TBPS-CLIP [42]
CFAM [19]
UMSA [43]
LSPM [44]
IRLT [45]
MDRL [46]
FSRL [47]
Propot [26]
SAP-SAM [48]
PLOT [27]
RDE [24]
APTM [21]
RaSa [20]
IRRA [23]
MARS [22]
HAM [28]
MARS-PDE𝑀
HAM-PDE𝑀

Image encoder
CLIP-RN101
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
Swin-B
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT

CUHK-PEDES

ICFG-PEDES

RSTPReid

R1

mAP

R1

mAP

R1

mAP

64.08
73.54
73.67
73.67
74.38
74.46
74.56
74.65
74.89
75.05
75.28
75.94
76.53
76.51
73.38
77.62
77.71
77.85
78.10

60.08
65.38
65.94
65.94
67.74
–
–
67.49
67.12
–
–
67.56
66.91
69.38
66.10
71.41
69.68
73.70
72.19

–
65.05
63.57
63.57
64.40
64.72
65.88
64.01
65.12
63.97
65.76
67.68
68.51
65.28
63.46
67.60
68.25
68.91
69.01

–
39.83
38.34
38.34
42.60
–
–
39.64
42.93
–
–
40.06
41.22
41.29
38.06
44.93
42.30
48.83
47.72

–
61.95
60.51
60.51
–
61.49
–
60.20
61.87
62.85
61.80
65.35
67.50
66.90
60.20
67.55
71.69
68.05
73.45

–
48.26
47.64
47.64
–
–
–
47.38
47.82
–
–
50.88
52.56
52.31
47.17
52.92
55.19
55.93
58.55

Text encoder
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
BERT-base
BERT-base
CLIP-Xformer
BERT-base
CLIP-Xformer
BERT-base
CLIP-Xformer

Fig. 7. Effect of the scaling factor 𝜆 and cluster number 𝑁 on the Rank-1 performance of HAM-PDE across three datasets.

Fig. 8. Effect of the scaling factor 𝜆 and cluster number 𝑁 on the mAP performance of HAM-PDE across three datasets.

governing performance, whereas 𝜆 mainly provides stable modulation.
As 𝑁 increases, performance improves at first and then saturates once
𝑁 exceeds a dataset-specific threshold (around 1300, 1600, and 200
on CUHK-PEDES, ICFG-PEDES, and RSTPReid, respectively), indicating
that overly fine clustering leads to over-fragmentation and weaker
prototypes.
By contrast, 𝜆 remains stable across datasets and can be fixed at 𝜆 =
0.25 without noticeable performance loss, which simplifies deployment.
As shown in Figs. 7 and 8, Rank-1 and mAP follow similar trends with
respect to 𝑁. Although the 𝑁 that maximizes Rank-1 does not always
yield the best mAP, it still gives competitive overall performance, and
is therefore adopted in our setting.
The larger fluctuation of Rank-1 on RSTPReid is likely due to
its much smaller test set, which reduces intra-cluster sample support
and weakens prototype stability. This further suggests that adapting
prototype granularity to dataset statistics is beneficial, especially in
long-tailed TI-ReID scenarios.

and 𝛽. As shown in Fig. 9, most gains are obtained in the first iteration,
followed by smaller but consistent improvements, and the method
converges within four iterations on all datasets. We therefore fix 𝑀 = 4
as a good trade-off between effectiveness and complexity.
The decay factor 𝛼 controls cluster-number reduction, while 𝛽 controls the contribution of prototypes at each stage. As shown in Fig. 11,
larger 𝛼 consistently harms mAP because excessively rapid cluster contraction leads to over-smoothed and less diverse prototypes. This effect
becomes stronger when 𝛽 is small, indicating that aggressive cluster
shrinking should be accompanied by sufficient prototype weighting.
Fig. 10 shows that Rank-1 is more variable, but larger 𝛽 generally
stabilizes performance by preserving the influence of prototypes in
later stages. However, excessively large 𝛽 weakens the adaptability
of the similarity matrix to refined cluster structures. Based on these
observations, we adopt 𝛼 = 0.25, 𝛽 = 0.8, and 𝑀 = 4, which provide
stable and consistent performance across datasets.
4.4.3. Analysis of ACE module
We evaluate ACE from two perspectives: cluster-number estimation and cluster–identity consistency. As shown in Table 3, the estimated cluster number 𝑁 is close to the hand-tuned value on all three

4.4.2. Analysis of Iteration Number 𝑀 and decay factors 𝛼, 𝛽
To improve robustness, PDE𝑀 progressively refines prototypes over
𝑀 iterations while reducing cluster granularity with decay factors 𝛼
8


codex
前三篇已经抽到方法段。两个细节需要补齐：A2HA 的属性配对和损失定义被输出截断了一段，ADA 的注意力原语伪代码也截断了一段，我会用更小页段补读，避免只靠摘要判断。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

5067

Adaptive Occlusion-Aware Network for Occluded
Person Re-Identification
Xiangzeng Liu , Member, IEEE, Jianfeng Guo , Hao Chen , Qiguang Miao , Senior Member, IEEE,
Yue Xi , and Ruyi Liu

Abstract— Occluded person re-identification (ReID) is a challenging task due to some of the essential features are interfered
by obstacles or other pedestrians. Multi-granularity local feature
extraction and recognition can effectively improve the accuracy
of ReID under occlusion. However, manual segmentation methods
for local features can lead to feature misalignment. Feature alignment based on pose estimation often ignores non-body details
(e.g., handbags, backpacks, etc.) while increasing the complexity
of the model. To address the above challenges, we propose a novel
Adaptive Occlusion-Aware Network (AOANet), which mainly
consists of two modules, the Adaptive Position Extractor (APE)
and the Occlusion Awareness Module (OAM). In order to
adaptively extract distinguishing features of body parts, APE
optimizes the representation of multi-granularity features by
the guidance of attention mechanism and keypoint features.
To further perceive the occluded region, the OAM is developed
by adaptively calculating the occlusion weights for body parts.
These weights can lead to highlighting the non-occluded parts
and suppressing the occluded parts, which in turn improves
the accuracy in the occluded situation. Extensive experimental
results confirm the advantages of our method on the MSMT17,
DukeMTMC-reID, Market-1501, Occluded-Duke and OccludedReID datasets. The comparative results demonstrate that our
method outperforms comparable methods. Especially on the
Occluded-Duke dataset, our method achieved 70.6% mAP and
81.2% Rank-1 accuracy.
Index Terms— Occluded person re-identification, body positions, transformer, local features.

I. I NTRODUCTION

A

S AN important direction of research in the field of
intelligent monitoring, person re-identification (ReID) is
a key technology to realize long-time pedestrian object tracking and cross-camera tracking. Its principle is to recognize
a specific pedestrian object by comparing the similarity of
appearance features of pedestrians in different scenes, and thus
Received 7 July 2024; revised 1 November 2024; accepted 28 December
2024. Date of publication 31 December 2024; date of current version
7 May 2025. This work was supported in part by the Natural Science Basic
Research Program of Shaanxi under Grant 2024JC-YBMS-467, in part by
the Aeronautical Science Foundation of China under Grant D023030002,
and in part by the Fundamental Research Funds for the Central Universities
under Grant QTZX24067. This article was recommended by Associate Editor
Z. Tao. (Corresponding authors: Xiangzeng Liu; Qiguang Miao.)
Xiangzeng Liu, Qiguang Miao, and Ruyi Liu are with the School
of Computer Science and Technology, Xidian University, Xi’an 710071,
China (e-mail: xzliu@xidian.edu.cn; qgmiao@xidian.edu.cn; ruyiliu@xidian.
edu.cn).
Jianfeng Guo, Hao Chen, and Yue Xi are with Guangzhou Institute
of Technology, Xidian University, Xi’an 710071, China (e-mail: jianfengguo@stu.xidian.edu.cn; haochenxd@stu.xidian.edu.cn; xiyue@xidian.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2024.3524555

realizing the continuous cross-scene tracking of the object.
It is mainly applied in the fields of public security criminal
investigation, intelligent security, intelligent campus, intelligent shopping malls, and so on. However, object occlusion,
pose uncertainty, appearance changes, and scene complexity
make person ReID still face many challenges. To address
these challenges, several methods [1], [2], [3] have recently
been proposed for ReID. For example, Luo et al. [1] used
ResNet50 [4] to extract global features of people and combined a number of ReID tricks to achieve good performance.
The occlusion phenomenon occurs frequently in different scenarios, which seriously affects the accuracy of re-recognition.
Therefore, methods relying only on global features cannot
achieve good performance in occluded environments. To cope
with occluded person ReID, some local feature-based methods [5], [6], [7] have demonstrated promising results. However,
the local features are susceptible to noise, such as background
interference, and their robustness against occlusion remains
insufficient. Therefore, the development of a robust occluded
person re-identification method is imperative.
The part-based methods show great potential in addressing the challenge of occluded person ReID, and are mainly
divided into manual splitting methods and pose estimation
based methods. Manual splitting methods are prone to feature
misalignment problems. As show in Fig. 1a, manual splitting is
employed to determine the positions of human bodies, which
in turn generates local features. However, due to differences
in object scales, this method may incorrectly compare the
head region of one image with the background region of
another image, leading to matching failures. The uniform
splitting of each person contributes to the issue of local feature
misalignment. One approach to rectify this challenge involves
leveraging pose estimation [8] to aid in position generation as
depicted in Fig. 1b. However, the local features generated by
pose estimation may become unstable due to the sensitivity
of environmental noise. In addition, the algorithm prioritizes
human features, which can lead to the omission of certain
important non-human human features, such as backpacks and
handbags. Furthermore, the implementation of this method
requires the incorporation of an auxiliary pose estimation
network, which increases the overall complexity of the model.
To address the aforementioned challenges, we introduce an
Adaptive Occlusion-Aware Network (AOANet) in this paper.
First, we employ Swin-Transformer [9] to acquire multi-scale
features and utilize them according to the semantic hierarchy of different scale features. Motivated by the concept of

1051-8215 © 2024 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.

5068

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

•

To demonstrate the effectiveness of our method, we conducted experiments on five related ReID datasets. The
experimental results demonstrate that our proposed
method achieves state-of-the-art performance.
II. R ELATED W ORK

Fig. 1.

Three different methods for human body positions extraction.

Deep Supervision [10], we adopt the Identity Loss [11] for
supervised training during this stage. Secondly, we obtain
local features for occlusion adaptation in two steps: body
part localization and local feature representation. Notably,
we constructed the Adaptive Position Extractor (APE) by
combining self-attention and cross-attention mechanisms with
keypoints for body part localization. As described in Fig. 1c,
our APE extracts human body parts with strong occlusion
adaptation. Thirdly, taking into account the effect of occlusion on the representation of local features, we design the
Occlusion Awareness Module (OAM). Since APE can extract
occlusion-adaptive local features that accurately reflect the
degree to which the object parts is occluded, we can use OAM
to compute occlusion weights for these local features. Overall,
the contributions of the paper are summarized as follows:
• We propose a novel Adaptive Occlusion-Aware Network (AOANet) for occluded person re-identification.
The AOANet enables feature adaptive representation of
occluded objects through multiple attention mechanisms
with supervised learning of key parts of the human body.
• We design an Adaptive Position Extractor (APE) that
obtains body regions that are adaptive to occlusion by
incorporating self-attention, cross-attention, and human
keypoints. Compared to the methods for pose estimation,
our model does not require pose estimation and thus has
a lower complexity.
• We develop the Occlusion Awareness Module (OAM),
which can adaptively calculate occlusion weights. These
weights have the ability to suppress the occluded part of
the feature and highlight the non-occluded part.

The challenges in ReID primarily stem from factors such
as variations in lighting, viewpoints, attitude, and occlusion.
Current approaches can be broadly categorized into two main
groups: feature representation learning [12], [13], [14] and
deep metric learning [15], [16], [17]. Within the realm of
feature representation learning, methods aim to extract discriminative features for person. On the other hand, deep
metric learning focus on learning similarity metrics capable of
measuring distances between person representations, thereby
enabling accurate identification across diverse surveillance
cameras. Nevertheless, these methods exhibit limited performance in intricate scenes, particularly occlusion changes.
In the context of occluded images, occluded person ReID
aims to match person exhibiting either holistic or occluded
appearances across different cameras. This task becomes
notably challenging owing to the presence of incomplete
information and spatial misalignment. To address these challenges, several approaches focus on information alignment.
Zhuo et al. [5] employed an occluded/non-occluded binary
classification loss to differentiate between occluded and nonoccluded images, leveraging this information to enhance
performance. Miao et al. [6] introduced a Pose-Guided Feature Alignment (PGFA), which utilizes pose landmarks to
mitigate the effects of noisy information from the occluded
regions of the target person. Wang et al. [18] applied graph
convolution to enhance the message-passing of semantic
features while suppressing that of meaningless and noisy elements. Yan et al. [19] introduced a model capable of acquiring
single-scale discriminative global features through the utilization of occlusion-based augmented data. In comparison
with these methodologies, our approach demonstrates superior
adaptability and robustness in acquiring local features. Another
class of approaches emphasizes spatial alignment, typically
seen in Part-based ReID methods. These methods extract
aggregated features from different body parts, placing emphasis on localized features and fine-grained information. Where
body parts can be generated either through specific predefined
semantic parts or with the assistance of pose estimation [11].
As a representative method for the specific predefined
semantic parts, PCB [20] divides the human body horizontally into multiple parts and then trains multiple part-level
classifiers. Moreover, some similar methods [21], [22], [23]
achieved the extraction of more discriminative features than
the original global features. However, these methods rely on
predefined parts and lack of adaptability prone to feature
misalignment.
Pose estimation based method adopted pose estimation
to integrate full-body features with local features, resulting
in commendable performance [24]. To leverage higher-order
information for feature learning and alignment, HOReID [18]
utilized high-order relations and human topology information
to achieve higher performance. BPBreID [25] incorporated

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.

LIU et al.: ADAPTIVE OCCLUSION-AWARE NETWORK FOR OCCLUDED PERSON ReID

5069

Fig. 2. Overview of our framework. It mainly consists of the Adaptive Position Extractor (APE) and the Occlusion Awareness Module (OAM). After using
Swin-Transformer as a backbone for feature extraction, human body positions are generated by APE. Weighted Average Pooling (WAP) is then used to process
the human body positions to obtain local features, and these local features are aggregated to generate global feature. OAM is used in the inference stage to
further improve recognition accuracy.

a body part focus module to improve the effectiveness of
localized features through external human semantic information. While the methods mentioned above have demonstrated
impressive performance, their vulnerability to noise in pose
estimation algorithm remains a significant concern. In addition, these methods usually require the addition of a pose
estimation model, which often generates larger feature maps
and is more complex than the ReID model itself. Our method
is more tolerant to pose estimation prediction errors, and
the extracted local features show strong adaptability under
occlusion.

body is obtained from four local features through a fully
connected layer:
f g2 = Fcat W,
(2)
h
i
where Fcat = f p1 , f p2 , f p3 , f p4 ∈ R1×(4D) represents the
feature obtained by concatenating four local features, and
W ∈ R(4D)×D is the weight matrix. Then, we present the
procedure for obtaining the four adaptive local features. Adaptive Position Extractor (APE) is developed to determine the
H
W
regions {Pi , i = 1, 2, 3, 4} ⊆ [0, 1]1× 4 × 4 of the four parts
of the human body, which can be calculated:

III. M ETHODOLOGY

P = A P E ( f 4 , F P N ( f 1 , f 2 )) ,

In this section, we first introduce the overall framework of
the proposed method, then present the structure of APE and
OAM in detail, and finally give the design of the loss function.

where P = [P1 ; P2 ; P3 ; P4 ] ∈ [0, 1]4× 4 × 4 . In particular,
f 1 and f 2 , which represent shallow features, contain more
low-level semantic information, such as clothing color and
body shape, and thus they are the best choices for extracting
part locations. We further fused these two layers of features
via the feature pyramid network (FPN) [27]. We adopt the
Weighted Averageh Pooling (WAP)
i method to extract local
features Flocal = f p1 ; f p2 ; f p3 ; f p4 ∈ R4×D , which is defined
as follows:

A. Overall Framework
Our model mainly consists of the Adaptive Position Extractor (APE) and the Occlusion Awareness Module (OAM), and
uses Swin-Transformer as the backbone network. As described
in Fig. 2, the proposed model
ultimately extractso two global
n
and four local features f g1 , f g2 , f p1 , f p2 , f p3 , f p4 ⊆ R1×D
from each person image. Firstly, we utilize the SwinTransformer [9] to obtain four distinct scale feature maps
D
H
W
D
H
W
D
H
W
{ f 1 ∈ R 8 × 4 × 4 , f 2 ∈ R 4 × 8 × 8 , f 3 ∈ R 2 × 16 × 16 , f 4 ∈
H
W
R D× 32 × 32 }. It is well known that f 4 contains a wealth of
high-level semantic information, making it a pivotal source
for both global and local features in our model. The feature
map f g1 for deep supervision can be obtained as follows:
f g1 = G A P ( f 4 ) ,

(1)

where G A P (·) represents Global Average Pooling [26].
Futhermore, global feature map f g2 for representing the human

H

(3)

W

Flocal = G A P (inter p ( f 4 ) ⊙ P) ,

(4)

where inter p (·) means using interpolation for upsampling and
⊙ represents element-wise product. After the training phase,
we introduced the Occlusion Awareness Module (OAM)
during the inference phase to further perceive occlusion information in local features.
B. Adaptive Position Extractor
The overall structure of the Adaptive Position Extractor is
illustrated in Fig. 3(a). In the APE, we define four learnable
part catchers, each representing a specific regions of the

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.

5070

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

Fig. 4. Illustration of the Occlusion Awareness Module. In the inference
stage, Scor e is used as the final similarity score of the two images.

Fig. 3. (a) Illustration of the Adaptive Position Extractor. (b) We partition
the human body into four distinct regions: the head, the upper body, the thigh,
and the lower leg.

human body (shown in Fig. 3(b)). These catchers are used to
extract adaptive part locations, which are trained supervised
by the labels extracted in the pose estimation algorithm. First,
we define four learnable part catchers {C1 , C2 , C3 , C4 } ∈
D
R1× 8 . Cross-attention is applied to motivate the learnable
part catchers to extract complex semantic information related
to body part regions from the feature map f 4 . In the above
process, the query Q c is derived from part catchers C =
D
[C1 ; C2 ; C3 ; C4 ] ∈ R4× 8 , while the key K c and value Vc are
derived from feature map f 4 , and the formula is as follows:
Q c = C WcQ ,


where f 4r
Q
and Wc

W
H
32 × 32

K c = f 4r WcK ,


Vc = f 4r WcV ,

(5)

×D

is obtained through reshaping f 4 ,
∈ R
D
D
D
D
∈ R 8 × 8 , WcK ∈ R D× 8 , WcV ∈ R D× 8
represents three different linear transformation matrices in
cross-attention. Further, we obtain the output features f ca ∈
D
R4× 8 from cross-attention:


Q c K cT
f ca = so f tmax
Vc ,
(6)
s
q
D
where s =
8 is the scaling factor. After completing the
calculations for cross-attention, we introduce self-attention
to capture the contextual information between different part
regions. In self-attention, the query Q s , key K s , and value Vs
are all derived from f ca :
Q s = f ca WsQ ,
Q

D

D

K s = f ca WsK ,
D

D

Vs = f ca WsV ,
D

D

(7)

where Ws ∈ R 8 × 8 , WsK ∈ R 8 × 8 , WsV ∈ R 8 × 8 represents
three different linear transformation matrices. Therefore, the
D
output f sa ∈ R4× 8 from self-attention is as follows:


Q s K sT
f sa = so f tmax
Vs ,
(8)
s

where s =

q

D
8

is the scaling factor. Then, we employ Feed

Forward Network (FFN) [28] to process f sa , which consists
of two fully connected layers:
f F = F F N ( f sa ) ,

(9)

4× D8

. In this case, we believe that part catchers
where f F ∈ R
require not only high-level semantic features but also depend
on the indispensability of shallow semantic features. As shown
H
W
D
in Eq. 3, We adopt FPN to generate feature f F P N ∈ R 8 × 4 × 4
for f 1 and f 2 , which is subsequently used to determine the
final part regions P. Formally,
P = σ ( f F · M L P ( f F P N )) ,

(10)

where M L P (·) is a simple neural network with three
fully connected layers and σ (·) represents the sigmoid
function.
C. Occlusion Awareness Module
To more effectively capture occlusion information from
local features, we construct the Occlusion Awareness Module (OAM), as depicted in Fig. 4. Note that this module
is exclusively utilized during the inference stage. To fully
perceive the occlusion information, we use the human body
H
W
regions P =n [P1 ; P2 ; P3 ; P4 ] ∈ o[0, 1]4× 4 × 4 and the feature
set f t = f p1 , f p2 , f p3 , f p4 , f g2 as inputs to this module.
Firstly, the part regions P are used to calculate the occlusion
′
′
′
′
weights Wo′ = [Wo1 , Wo2 , Wo3 , Wo4 ] ∈ [0, 1]1×4 :
P H4 −1 P W4 −1
h=0
w=0 Pi (h, w)
i′
Wo =
, i ∈ {1, 2, 3, 4} .
H
P4 P 4 −1 P W4 −1
P
(h,
w)
k=1
h=0
w=0 k

(11)

As shown in Fig. 4, for the two images (I1 , I2 ), the
occlusion weights Wo′ 1 and Wo′ 2 can be obtained by OAM,
respectively. When comparing two images, the proposed
model aims to minimize the influence of heavily occluded
positions in the comparison. Therefore, we take the minimum

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.

LIU et al.: ADAPTIVE OCCLUSION-AWARE NETWORK FOR OCCLUDED PERSON ReID

between the Wo′ 1 and Wo′ 2 counterparts as the final occlusion
weight Wo . Formally,
′

′

Wo = min(Woi 1 , Woi 2 ), i ∈ {1, 2, 3, 4} ,
[Wo1 , Wo2 , Wo3 , Wo4 ]

(12)

1×4

where Wo =
∈ [0, 1] . Next, it is
imperative to compute the Euclidean distance between the
respective features within the feature set f t of the two images,
yielding five distinct Euclidean distances {d1 , d2 , d3 , d4 , d2 }.
Ultimately, the definitive similarity scor e is derived through
a weight sum incorporating both the Euclidean distance and
occlusion weight:
socr e =

5
X

Woi · di ,

(13)

i=1

where Wo5 = 1 indicates that the occlusion weight associated
with the global feature f g2 is set to 1. This is because global
features typically encompass overall visual information, and
even if local regions are occluded, global features still provide
useful information.
D. Loss Function
For the global feature f g1 , we adopt the standard
cross-entropy loss denoted as L id , while the formula for the
triplet loss L tri is as follows:



L tri = d f a , f p − d ( f a , f n ) + α + ,
(14)
where α  is the margin, f a represents the anchor feature,
d f a , f p denotes the distance between positive sample pairs
and d ( f a , f n ) represents the distance between negative sample
pairs. The “+” indicates
n that the result
o is taken as positive.
1
2
3
4
For local features f p , f p , f p , f p , we adopt the average
triplet loss, denoted as L par t_tri . Formally,
L par t_tri =

4



i
1 Xh  i
i
i
i
d f p_a , f p_
−
d
f
,
f
+
α
,
p
p_a
p_n
4
i=1

(15)
i , fi , fi
where f p_a
p_ p
p_n represent anchor features, positive
sample features and negative sample features respectively.
We believe that global feature f g2 are suitable for person recognition and local features are mainly used to resist occlusion,
so we apply the ID loss and the average triplet loss to them
respectively.
Our Adaptive Position Extractor utilizes human parsing
labels Y for supervised training, where Y is generated by the
PifPaf [29] pose estimation model. We partition the human
body into four distinct regions: the head, the upper body
(comprising the torso and arms), the thigh, and the lower leg,
denoted as {Y1 , Y2 , Y3 , Y4 }. The formula for the local loss is
as follows:
4

H

−1 W −1

4
4
X
λXX
Yi (h, w) · log (Pi (h, w)) ,
L local = −
4

(16)

i=1 h=0 w=0

where λ serves as a hyperparameter to regulate adaptability
and mitigate the influence of noise labels generated by the
attitude estimation algorithm.

5071

In summary, we performed ablation experiments on the
weights between the loss functions to finally obtain the optimal
combination of loss weights. The comprehensive loss function
L total is as follows:
L total = L tri + 2L id + L par t_tri + L local .

(17)

IV. E XPERIMENTS
A. Datasets and Evaluation Metrics
We confirm the effectiveness of our approach on five
datasets, comprising three holistic datasets: MSMT17 [48],
DukeMTMC-reID [49], and Market-1501 [50], as well as two
occluded datasets: Occluded-Duke [6] and Occluded-ReID [5].
MSMT17 consists of 4,101 identities observed across
15 cameras, making it the largest person ReID dataset. The
training set encompasses 1,041 people, yielding a total of
32,621 bounding boxes. In contrast, the test set involves 3,060
people, contributing to a total of 93,820 bounding boxes.
DukeMTMC-reID comprises 36,411 images representing
1,404 identities captured by 8 cameras. The training set
consists of 16,522 images, while the query set comprises 2,228
images, and the gallery set includes 17,661 images.
Market-1501 comprises 1,501 identities captured by 6 cameras. The training set includes 12,936 images from 751 identities, the query set contains 3,368 images, and the gallery set
encompasses 19,732 images.
Occluded-Duke is derived by splitting the DukeMTMCreID dataset. A notable distinction is that both the query set
and the gallery set comprise 10% occluded person images.
It encompasses a total of 15,618 training images, 17,661
gallery images, and 2,210 occluded query images.
Occluded-ReID constitutes a dataset captured through
mobile cameras, containing 2,000 images distributed among
200 unique identities. Each identity is represented by five
full-body person images and an additional five images featuring occluded persons, showcasing diverse viewpoints and
varying degrees of severe occlusions.
Evaluation Metrics employ Cumulative Matching Characteristic (CMC) curves and mean average precision (mAP).
In addition, to ensure fairness, we only use single-query
evaluations and do not use re-ranking [51] in our experiments.
B. Implementation Details
We utilize the Swin-S structure within the SwinTransformer as our backbone, which has been pre-trained on
ImageNet-22K. The input image size is resized to 384 × 128,
and the channel dimension of the feature map D = 768. For
data augmentation, we exclusively apply random horizontal
flipping, padding, random cropping, and random erasing to
the training images. The batch size is configured to 64, with
four images allocated for each person. During training we
utilize the SGD optimizer and set its the momentum and
weight decay to 0.9 and e−4 , respectively. The initial learning
rate is established at 5e−5 , and is gradually changed with
cosine learning rate decay and warm-up learning. Furthermore,
the adaptability parameter λ in the local loss is set to 0.4.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.

5072

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

TABLE I
C OMPARISON W ITH THE S TATE - OF - THE -A RTS ON T HREE H OLISTIC DATASETS : MSMT17, D UKE MTMC- RE ID, AND M ARKET-1501

APE contains three modules including Cross-attention, Selfattention and FFN, with 6 layers in total. The shapes of the
two linear layers of FFN are 768 × 96 and 96 × 768.

TABLE II
C OMPARISON W ITH S TATE - OF - THE -A RT M ETHODS ON T WO O CCLUDED
DATASETS , O CCLUDED -D UKE AND O CCLUDED -R E ID

C. Comparative Experiment
Results on Holistic Datasets: To validate the efficacy of
our method, we conducted a comprehensive comparison with
several noteworthy works in recent years, presenting the
results in Table I. As evident from the table, our method
outperformed other approaches on the largest ReID dataset
MSMT17, achieving 74.8% mAP and 89.8% Rank-1 score.
The ISR [42] method performs unsupervised pre-training
on the large-scale person dataset LUPerson [52], and then
fine-tunes the MSMT17 dataset, and its effect is only 71.5%
mAP and 88.4% Rank-1 accuracy. This shows that our method
can outperform pre-trained models on large datasets through
structural design for ReID. Additionally, on the DukeMTMCreID dataset, our method demonstrated significant performance
with a mAP of 87.0% and a Rank-1 score of 93.6%, surpassing
other algorithms. Notably, on the Market-1501 dataset, our
method achieved remarkable results with a mAP of 93.0%
and a Rank-1 score of 97.1%. Although the mAP of NFormer
on this dataset is comparable to ours, our Rank-1 score is
1.4% higher than it. Moreover, it performs worse than our
method on the large dataset MSMT17. The above results
demonstrate the effectiveness and robustness of our method.
This is due to the multi-attention mechanism and occlusion
adaptive representation design in AOANet, which performs
particularly well on large datasets.
Results on Occluded ReID Datasets: We posit that sufficiently robust features should effectively handle diverse and
challenging scenarios. Consequently, we conducted experiments on two occluded datasets, namely Occluded-Duke and
Occluded-ReID. As depicted in Table II, our method significantly outperforms recent algorithms on these datasets.
Specifically, on the Occluded-Duke dataset, our method
achieves mAP of 70.6% and Rank-1 score of 81.2%. On the

Occluded-ReID dataset, the corresponding scores are 88.8%
for mAP and 91.7% for Rank-1 score. These results demonstrate that our model exhibits a high degree of robustness, even
in occluded environments. This is because APE can effectively
identify and highlight the features of non-occluded areas and
adaptively capture key information even in severe occlusion.
In addition, OAM effectively improves the recognition accuracy by reducing the feature weights of occluded parts.
In Fig. 5, we present the top 5 matches for eight distinct
persons, obtained by TransReID [31], FED [36], PFD [38],
and our method. For instance, in the case of the first two
persons, while other methods are capable of recognizing the
local feature of a hat, they fall short in terms of robustness
and discrimination regarding local features, resulting in subpar recognition outcomes. In contrast, our method achieves
successful matching in almost all samples. In addition, the
proposed method has higher sensitivity to unoccluded human

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.

LIU et al.: ADAPTIVE OCCLUSION-AWARE NETWORK FOR OCCLUDED PERSON ReID

Fig. 5.

5073

Visual retrieval results on Occluded-Duke.A green number indicates a correct match, and a red number indicates an incorrect match.

features compared to other methods. A case in point involves
the matching results for the second-to-last person, where the
other three methods prioritize the background trees in their
similarity assessments. Conversely, our method accentuates
white umbrella and body features, correctly assigning the
highest similarity score to the most relevant sample at the
Rank-1. A comparable scenario is observed in the matching

results for the final person, where other methods mistakenly
rely on background automotive features for retrieval, thereby
compromising the quality of the matching outcomes.
D. Ablation Study
In this section, we conducted ablation studies on the
MSMT17 and Occluded-Duke datasets to scrutinize each

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.

5074

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

TABLE III

TABLE IV

C OMPARATIVE OF P ERFORMANCE ACROSS VARIOUS C OMPONENTS

C OMPARISON OF R ESULTS FOR D IFFERENT N UMBERS OF H UMAN
P OSITIONS ON THE O CCLUDED -D UKE AND
M ARKET-1501 DATASETS

Fig. 6.

Feature map visualization.

component of our AOANet, including the Adaptive Position
Extractor (A) and the Occlusion Awareness Module (O). The
baseline configuration omits all components and solely utilizes
Swin-S as the backbone for extracting the global feature,
which denoted as f g1 in (1). The results are shown in Table III.
Effectiveness of the Adaptive Position Extractor: As indicated in index-2 of Table III, the introduction of the APE
yields a significant improvement in performance compared
to the baseline. In this instance, only the global feature f g2
from (2) is employed. On the MSMT17 dataset, both mAP
and Rank-1 score show notable improvements, increasing by
11.8% and 6.1%, respectively. Similarly, on the OccludedDuke dataset, there are substantial improvements, with mAP
and Rank-1 score increasing by 18.4% and 16.9%, respectively. The APE has the capability to extract adaptive body
positions, yielding local features that are highly robust and
discriminative. Furthermore, the global features derived from
these local features exhibit robustness as well. In Fig. 6,
we further illustrate the feature representation capability of
the proposed method through feature map visualization. It is
evident from the figure that the features extracted by Swin-S
are somewhat incomplete. Fortunately, with the addition
of APE, the robustness of the extracted features is significantly
enhanced.
Effectiveness of the Occlusion Awareness Module: As indicated in index-3 of Table III, the performance was improved on
both datasets by adding OAM on top of APE. Specifically, on
the MSMT17 dataset, there is an increase of 1.9% in mAP and
1.3% in rank-1 score. The improvement is more pronounced
on the Occluded-Duke dataset, with mAP increasing by 2.6%
and rank-1 score by 2.9%. This notable enhancement can
be attributed to OAM’s ability to further leverage occlusion

Fig. 7.
Three groups with different numbers of human body position
divisions. The top row in each group represents positions extracted by APE,
while the bottom row corresponds to labels generated by pose estimation.

information within local features, which is a particularly
advantageous property on occluded datasets.
Analysis of the Number of Human Body Positions: We
conducted quantitative experiments on the Occluded-Duke and
Market-1501 datasets to determine the optimal number of
human body positions. As depicted in Table IV, the best
performance is achieved when the number of positions is
set to 4. Furthermore, as shown in Fig. 7, we intuitively
represent the perception results of the proposed method for
different numbers of human body positions. In Fig. 7a, it is
observed that setting the number of positions to 11 results
in only some positions producing meaningful results, while
the rest are redundant. For Fig. 7b, where the number is set
to 6, we observed that APE tends to prioritize the extraction
of crucial features, disregarding the distinction between left
and right. In summary, we opt to set the number to 4,
encompassing the head, the upper body, the thigh, and the
lower leg.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.


hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025

10571

Adaptive Pseudo-Label Purification and Debiasing
for Unsupervised Visible-Infrared Person
Re-Identification
Xiangbo Yin , Jiangming Shi , Zhizhong Zhang , Member, IEEE, Yuan Xie , Member, IEEE,
and Yanyun Qu , Senior Member, IEEE
Abstract—Unsupervised
Visible-Infrared
Person
ReIdentification (USVI-ReID) aims to match visible and infrared
person images without relying on prior annotations. Recently,
unsupervised contrastive learning methods have become the
mainstream approach for USVI-ReID, leveraging clustering
algorithms to generate pseudo-labels. However, these methods
often suffer from inherent noisy pseudo-labels, which significantly
hinders their performance. To address this challenge, we propose
a Adaptive Pseudo-label Purification and Debiasing (APPD)
framework for USVI-ReID, which is designed to calibrate noisy
pseudo-labels and dynamically detects clean pseudo-labels,
thereby enhancing the model’s performance and reliability.
Specifically, we propose an Adaptive Pseudo-label Calibration
and Division (APCD) module, which calibrates noisy pseudolabels by assessing their reliability and divides pseudo-labels into
clean and noisy subsets, ensuring a more focused and accurate
learning process. Based on the calibrated pseudo-labels, we
develop an Optimal Transport Prototype Matching (OTPM)
module to establish robust cross-modality correspondences.
For clean pseudo-labels, we propose a Debiased Memory
Hybrid Learning (DMHL) module, which jointly captures
modality-specific and modality-invariant information while
addressing sampling bias to enhance feature representation.
To effectively utilize noisy pseudo-labels, we introduce a
Neighbor Relation Learning (NRL) module that mitigates
intra-class variations by exploring neighbor relationships in
the feature space. Comprehensive experiments conducted on
two widely recognized USVI-ReID benchmarks demonstrate
that APPD achieves state-of-the-art performance, significantly
outperforming existing methods. The source code will be made
available at https://github.com/XiangboYin/RPNR
Received 17 January 2025; revised 15 April 2025; accepted 13 May 2025.
Date of publication 20 May 2025; date of current version 6 October 2025.
This work was supported by in part by the National Natural Science Foundation of China under Grant 62176224, Grant 62222602, Grant 62176092,
Grant U23A20343, and Grant 62476090; in part by the Natural Science
Foundation of Shanghai under Grant 23ZR1420400; in part by Shanghai
Sailing Program under Grant 23YF1410500; in part by CCF-Tencent under
Grant RAGR20240122; and in part by the Science and Technology on Sonar
Laboratory under Grant 2024-JCJQ-LB-32/07. This article was recommended
by Associate Editor Z. Mao. (Jiangming Shi contributed equally to this work.)
(Corresponding author: Yanyun Qu.)
Xiangbo Yin and Yanyun Qu are with the School of Informatics, Xiamen University, Xiamen 361005, China (e-mail: xiangboyin@stu.xmu.edu.cn;
yyqu@xmu.edu.cn).
Jiangming Shi is with the Institute of Artificial Intelligence, Xiamen
University, Xiamen 361005, China (e-mail: jiangming.shi@outlook.com).
Zhizhong Zhang is with the School of Computer Science and Technology, East China Normal University, Shanghai 200062, China (e-mail:
zzzhang@cs.ecnu.edu.cn).
Yuan Xie is with the School of Computer Science and Technology, East
China Normal University, Shanghai 200062, China, and also with Chongqing
Institute, East China Normal University, Chongqing 401120, China (e-mail:
yxie@cs.ecnu.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3571976

Index Terms—USVI-ReID, noisy labels, optimal transport,
debiased contrastive learning, neighbor relation learning.

I. I NTRODUCTION
HE increasing demand for intelligent security has led
to the widespread adoption of smart monitoring sensor devices designed for 24-hour surveillance. The distinct
imaging principles of sensor devices during daytime and
nighttime result in significant differences between visible
and infrared images. This discrepancy has fueled growing
interest in research on visible-infrared person re-identification
(VI-ReID), which aims to accurately match visible and
infrared pedestrian images, allowing for the retrieval of a
pedestrian image from one modality based on a query from
another [1], [2], [3], [4], [5]. However, the substantial disparity
between these two modalities poses a significant challenge
for this task. Recently, numerous VI-ReID methods [6], [7],
[8], [9], [10] have focused on reducing cross-modality discrepancies by aligning visible and infrared images at both
the image and feature levels, achieving notable performance
gains. However, these approaches are highly dependent on
well-annotated cross-modality datasets, which are costly and
labor-intensive to obtain in real-world applications. As a
result, unsupervised visible-infrared person re-identification
(USVI-ReID) has garnered increasing attention.
The primary challenges of USVI-ReID lie in generating
robust pseudo-labels. Existing USVI-ReID methods [11], [12],
[13], [14] predominantly adopt the DCL [15] framework,
which utilizes DBSCAN for pseudo-label generation and
establishes cross-modality correspondences based on these
pseudo-labels. Pseudo-labels, being the result of clustering,
are inherently prone to noise. These noisy pseudo-labels can
misdirect the model, resulting in distorted learning of data
distributions and suboptimal feature representations, thereby
undermining overall performance. To address the impact of
noisy pseudo-labels, MMM [14] computes confidence scores
based on the classifier loss, using these scores to reduce
the influence of noisy labels. PGM [11], on the other hand,
minimizes the effect of noisy labels by alternately applying
two unidirectional metric losses, which helps prevent the
rapid emergence of inaccurate pseudo-labels. However, neither
method explicitly refines noisy pseudo-labels into cleaner
ones, limiting the model’s ability to fully exploit hard-todiscriminate features.
As shown in Fig. 1, noisy pseudo-labels can lead to three
negative influences for USVI-ReID: a) The centroid memory

T

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

10572

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025

Fig. 1. The negative influence of the noisy pseudo-labels. (a) Noisy pseudo-labels may initialize an unreliable memory bank, misguiding the model to train
in the wrong direction. (b) Noisy pseudo-labels can lead to unreliable cross-modality correspondences, which hinder the learning of robust cross-modality
representations. (c) “Sampling bias” problem caused by noisy pseudo-labels, i.e., the negative samples from negative sets may not be true negative samples,
which interferes with the performance improvement of the model.

bank is initialized and continuously updated by pseudo-labels.
However, since pseudo-labels inherently contain noise, the
memory bank fails to reflect the true feature distribution,
making it unreliable and leading the model to deviate from
the correct training. b) Reliable cross-modality correspondences are crucial for USVI-ReID. However, the existence
of noisy pseudo-labels may lead to unreliable cross-modality
correspondences, which hinder the learning of robust crossmodality representation. c) The sampling bias phenomenon
caused by noisy pseudo-labels, having not been explored by
previous methods, refers to negative samples from negative
sets that may not be true negative samples, which can empirically lead to a significant performance drop [16]. Therefore,
calibrating and identifying cleaner pseudo-labels can help
mitigate the aforementioned issues to a certain extent.
In this paper, we propose the Adaptive Pseudo-label
Purification and Debiasing (APPD) framework to address
the above three negative influences of noisy pseudo-labels.
Specifically, to obtain robust pseudo-labels, we propose an
Adaptive Pseudo-label Calibration and Division (APCD) module. Unlike traditional methods that directly use pseudo-labels
produced by clustering algorithms to train the model, APCD
first calibrates noisy pseudo-labels into more robust ones and
subsequently isolates clean and noisy pseudo-labels. These
calibrated pseudo-labels are then used in the Optimal Transport
Prototype Matching (OTPM) module to establish reliable
cross-modality alignments. To reduce cross-modality gaps, the
Debiased Memory Hybrid Learning (DMHL) module captures
both modality-specific and modality-invariant information,
while addressing sampling biases caused by noisy labels.
To fully utilize noisy pseudo-labels, the Neighbor Relation
Learning (NRL) module models pair-wise relationships in the

feature space, encouraging closer clustering of similar samples
and mitigating intra-class variations.
In conclusion, the main contributions of our method can be
summarized as follows:
• We propose the Adaptive Pseudo-label Purification and
Debiasing (APPD) framework to address the negative
influences of noisy pseudo-labels in USVI-ReID. Compared to the conference version, this paper has several
critical improvements: adaptive pseudo-label division,
debiased contrastive learning, and more detailed experiments.
• We design the Adaptive Pseudo-label Calibration and
Division (APCD) module to calibrate noisy pseudo-labels
into more robust ones and isolate clean and noisy pseudolabels for effective learning.
• We propose the Optimal Transport Prototype Matching
(OTPM) module to establish reliable cross-modality correspondences based on calibrated noisy pseudo-labels.
• We propose Debiased Memory Hybrid Learning (DMHL)
and Neighbor Relation Learning (NRL) modules to alleviate inter- and intra-modality gaps.
• Experiments on mainstream datasets demonstrate the
superiority of our method compared with existing methods, and APPD generates higher-quality pseudo-labels
than other methods. In addition, the performance of
APPD for SVI-ReID and SSVI-ReID is promising.
II. R ELATED W ORK
A. Unsupervised Single-Modality Person ReID
Unsupervised single-modality person ReID aims to extract
discriminative identity features from unlabeled person ReID

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

YIN et al.: ADAPTIVE PSEUDO-LABEL PURIFICATION AND DEBIASING FOR USVI-ReID

datasets. Many current unsupervised methods heavily rely on
pseudo-labels, employing an iterative process that alternates
between generating pseudo-labels and representation learning
[17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27]. The
Cluster-Contrast framework [28], [29] introduces a strategy
that utilizes unique centroid representations for cluster-level
contrastive learning, supported by a momentum update mechanism to maintain feature consistency. However, using a single
proxy to represent an entire cluster may introduce biases. To
address this limitation, multi-proxy methods [27], [30] have
been developed to enhance robustness. Given the inherent
noise in pseudo-labels, label refinement strategies [17], [31],
[32] have been proposed to improve their reliability. Although
these techniques perform well in single-modality unsupervised
ReID, their direct application to unsupervised VI-ReID faces
significant obstacles due to the large cross-modality gap.

10573

III. M ETHODOLOGY
A. Notation Definition
Given an unlabeled visible-infrared person re-identification
dataset D = {DV , DR }, where DV = {xiv | i = 1, 2, . . ., N v }
represents the unlabeled visible dataset with N v visible samples and DR = {xir | i = 1, 2, . . ., N r } denotes the unlabeled
infrared dataset with N r infrared samples. For the USVI-ReID
task, the objective is to train a robust network fθ to map
an instance xit from D into an embedding space F, where
t ∈ {v, r} denotes the visible and infrared modality. Therefore,
we can employ the encoder fθ to extract d dimensional visible
features F v = { fiv | i = 1, 2, . . ., N v } and infrared features
F r = { fir | i = 1, 2, . . ., N r }, where fit ∈ Rd .
B. Overview

B. Unsupervised Visible-Infrared Person ReID
Unsupervised visible-infrared person re-identification
(USVI-ReID) has drawn significant attention for its capability
to learn both modality-specific and modality-invariant
features without requiring cross-modality annotations. Most
USVI-ReID methods [11], [12], [13], [33], [34], [35],
[36], [37] follow the DCL [15] framework, which typically
involves two steps: (1) generating pseudo-labels through
clustering and (2) leveraging these pseudo-labels to establish
cross-modality correspondences. Methods like PGM [11]
and MBCCM [33] employ multi-stage graph matching by
constructing bipartite graphs, while OTLA [38] and DOTLA
[12] use Optimal Transport to map pseudo-labels between
modalities at the instance level. However, the inherent noise
in pseudo-labels often leads to unreliable cross-modality
correspondences, underscoring the need for strategies that
produce higher-quality pseudo-labels for USVI-ReID tasks.
C. Learning With Noisy Labels
Label noise has been shown to negatively impact the training of deep neural networks [39], [40], [41], [42]. Existing
strategies for addressing noisy labels can be broadly categorized into two approaches: label correction and sample
selection. Label correction methods [43], [44], [45], [46] focus
on using model predictions to refine noisy labels. For instance,
SMP [47] introduces an iterative learning framework to relabel
noisy samples and train the network directly on the noisy
dataset without additional clean data. Similarly, [48] employs
back-propagation to probabilistically update and correct image
labels alongside network training. In contrast, sample selection
methods [49], [50], [51] aim to identify and retain clean
samples while excluding noisy ones during training. NCE [52]
filters clean samples based on neighbor information, while
CBS [53] employs confidence-based sample augmentation to
improve the reliability of selected clean data. For the USVIReID task, pseudo-labels generated by clustering algorithms
are inherently noisy. Therefore, refining these noisy pseudolabels is essential for enhancing the model’s performance in
this domain. In this paper, we employ the calibration-thensampling strategy to shield the model from the effect of noisy
data.

The overall framework of our APPD is illustrated in Fig. 2.
Initially, we utilize the DBSCAN [54] algorithm to cluster visible and infrared features, respectively. Following the clustering
process, pseudo-label yti ∈ {1, 2, . . ., Y t } is assigned to the i-th
image from modality t, where Y t represents the total number
of clusters. Given the inherent noise in pseudo-labels, we
introduce an effective calibration-then-division module called
Adaptive Pseudo-label Calibration and Division (APCD) to
calibrate and sample cleaner pseudo-labels. First, we refine
the noisy pseudo-labels into more robust ones and assign
these calibrated pseudo-labels ŷti for each sample to obtain
Nv
Nr
the “labeled” dataset D̃V = {(xiv , ŷvi )}i=1
and D̃R = {(xir , ŷri )}i=1
.
Since the training data may still contain some noise even
after calibration, relying solely on all the data for training can
significantly impair the model’s generalization and robustness.
Therefore, we divide the visible and infrared pseudo-labels
into clean and noisy subsets, denoted as S v = S cv ∪ S nv and
S r = S cr ∪ S nr , respectively. After that, the clean set S ct is used
to perform Debiased Memory Hybrid Learning introduced in
Sec. III-E while the noisy set S nt is used to carry out Neighbor
Relation Learning introduced in Sec. III-F.
Notably, the pseudo-labels generated by two separate
clusterings for visible and infrared samples reveal a misalignment. To resolve this, we introduce the Optimal Transport
Prototype Matching (OTPM) module, which aligns visible
and infrared prototypes through optimal transport at clusterlevel, ensuring more accurate cross-modality correspondences.
Learning modality-invariant features is essential for effective
cross-modality matching. To further exploit modality-invariant
information and reduce sampling bias, we propose the Debiased Memory Hybrid Learning (DMHL) module. In this
module, we merge visible and infrared prototypes into new
modality-hybrid prototypes to better mitigate the substantial
cross-modality gaps. To address sampling bias introduced by
noisy pseudo-labels, we integrate Debiased Contrastive Learning, enabling noise-tolerant contrastive learning for improved
robustness. Moreover, DMHL does not account for potential
interactions among all noisy samples. To address this limitation, we introduce the Neighbor Relation Learning (NRL)
module, specifically designed to capture and model the complex interactions across the noisy sample set.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

10574

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025

Fig. 2. Overall of the proposed APPD. APPD begins by generating modality-specific pseudo-labels using DBSCAN at stage (a). Next, it calibrates noisy
pseudo-labels and identifies cleaner ones at stage (b). Leveraging the cleaner pseudo-labels, APPD employs optimal transport to establish cross-modality
correspondences at stage (c). For clean pseudo-labels, APPD jointly captures modality-specific and modality-invariant information while addressing sampling
bias to enhance feature representation at stage (d). To effectively utilize noisy pseudo-labels, APPD mitigates intra-class variations by exploring neighbor
relationships in the feature space (with interaction strength represented by thickness) at stage (e).

C. Adaptive Pseudo-Label Calibration and Division
Since pseudo-labels generated through clustering inherently contain noise, we introduce the Adaptive Pseudo-label
Calibration and Division (APCD) module to calibrate noisy
pseudo-labels and adaptively select cleaner data. To be specific, for the c-th cluster in modality t, represented by a
t nc
collection of d-dimensional features { fc,i
}i=1 , where nc indicates the number of features in the c-th cluster and t ∈ {v, r}
corresponds to the visible or infrared modality, we construct
the intra-class affinity matrix S using Jaccard Similarity,
defined as follows:
ˇ

ˇ

ˇ
ˇ
t
, κ ∩ R fc,t j , κ ˇ
ˇR fc,i
t

ˇ ,
Si j = ˇ
(1)

ˇ
ˇ
t
, κ ∪ R fc,t j , κ ˇ
ˇR fc,i
t
where S it j represents the affinity between fc,i
and fc,t j , while
t
t
R fc,i , κ denotes the κ-reciprocal nearest neighbors of fc,i
.
t
t
t
A higher S i j indicates greater similarity between fc,i and fc, j .
t
For fc,i
, its reliability increases if it is surrounded by more
similar samples. To identify reliable samples within a cluster,
we introduce a Similarity Counter Gtc,i for each sample:

Gtc,i =

nc
X

sign(S it j − ρ), i ∈ {1, 2, . . ., nc },

(2)

K

where id denotes the indexes of top-K similarity counts and Gtc
represents the collection of similarity counts for all samples
in the c-th cluster.
With these reliable samples, we can construct a robust
prototype for the c-th cluster as follows:
1 X t
fc,i .
(5)
ptc =
K
i∈id

Then, we can obtain a reliable prototype set pt

=

{pt1 , pt2 , . . ., ptY t }. For a given sample xit from Dt , the similarity
score δtc,i between the extracted feature fit and the prototype
ptc of c-th cluster is computed as follows:
δtc,i =

( fit ) · (ptc )T
.
fit 2 · ptc 2

(6)

A larger δtc,i indicates that the sample xit is more likely to
belong to the c-th cluster. Thus, the corrected pseudo-label can
be obtained as follows:
ŷti = arg max δtc,i , c ∈ {1, 2, . . ., Y t }.

j=1

8
ˆ
if x > 0,
<1,
sign(x) = 0,
if x = 0,
ˆ
:
−1, if x < 0,

samples tend to have higher similarity counts. Therefore, we
consider the samples with the top-K similarity counts as
reliable samples:
id = arg max Gtc ,
(4)

(7)

c

(3)

where sign(·) is a sign function denoted as Eq. (3) and ρ is a
threshold fixed to 0.5. It is evident that correctly categorized

The conference version (RPNR) solely alleviated noise
by the noisy pseudo-label calibration module (NPC), but
the calibrated pseudo-labels may still contain noise. Training
directly on entire pseudo-labels may degrade both the generalization capability and robustness of the USVI-ReID model.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

YIN et al.: ADAPTIVE PSEUDO-LABEL PURIFICATION AND DEBIASING FOR USVI-ReID

To address this limitation, we propose an adaptive pseudo-label
division strategy building upon NPC, which effectively identifies cleaner pseudo-labels for contrastive learning to enhance
model performance. To effectively distinguish clean data from
noisy ones, we unify the pseudo-labels and feature space by
establishing a linear relationship between the features fit and
the one-hot pseudo-label ỹti , as expressed by the following
formulation:
T
ỹti = fit β + ζi + εi ,
(8)
t

where ỹti ∈ RY is the one-hot label transformed by the cort
rected pseudo-label ŷti and β ∈ Rd×Y represents the coefficient
t
matrix. εi ∈ RY denotes the random noise. The noisy data
t
indicator ζi ∈ RY can be regarded as the correction of the
linear prediction, where the data will be recognized as clean
when ζi = 0.
For a well-trained network, the output prediction for clean
data should closely align with the one-hot vector, while the
feature-label pair for noisy data is less likely to be accurately
represented by the linear model. Hence, we have:
(
T
clean : ỹti ∼ N ( fit β∗ , ψ2 IY t ),
if ζi∗ = 0,
(9)
T
t
∗
t
∗
2
noisy : ỹt∗
i = ỹi − ζi ∼ N ( fi β , ψ ), otherwise,
where ψ denotes the scale estimate. Thus, we can formalize
the index of clean set S ct = {i : ζi∗ = 0} and noisy set S nt = {i :
ζi∗ , 0}. This optimization problem can be solved by the SPR
algorithm [42]. After that, the clean set S c is used to perform
Debiased Memory Hybrid Learning introduced in Sec. III-E
while the noisy set S nt is used to carry out Neighbor Relation
Learning in Sec. III-F.
D. Optimal Transport Prototype Matching
Building on the reliable prototype features, we propose
the Optimal Transport Prototype Matching (OTPM) module to establish robust cross-modality correspondences at
the cluster level. Provided with the visible prototype set
pv = {pv1 , pv2 , . . ., pvY v } and the infrared prototype set pr =
{pr1 , pr2 , . . ., prY r }, in which Y v and Y r correspond to the quantities of visible and infrared clusters, respectively. PGM [11]
has revealed that Y v > Y r , indicating an inconsistency in the
number of clusters between visible and infrared modalities.
In this case, the core challenge of cross-modality correspondences lies in the many-to-many matching of inter-modality
prototypes, which can be effectively addressed using Optimal
Transport:
1
minhQ, Ci + H(Q),
Q
λ
(
Q1 = 1 · Y1v ,
s.t.
QT 1 = 1 · Y1r ,
v

r

(10)

where Q ∈ RY ×Y denotes the transport plan for crossv
r
modality matching, while C ∈ RY ×Y represents the cost
matrix between inter-modality
prototypes. Specifically, Ci j =

v r
1/ exp cos(pi , p j ) , where cos(·) denotes cosine similarity.
The operator h·i refers to the Frobenius dot-product, and 1
is a vector with all elements being 1. Additionally, H(Q)
represents the Entropic Regularization, and λ serves as the

10575

regularization parameter. The objective function can be solved
using the Sinkhorn-Knopp algorithm [55] to derive the optimal
v
r
transport plan Q∗ ∈ RY ×Y . From this, two matched pseudolabel sets, Y v→r and Y r→v , can be obtained for network training
based on transport plan Q∗ :
Yiv→r = arg max Q∗i j , j ∈ {1, 2, . . . , Y r },
j

Y r→v
= arg max Q∗ji , i ∈ {1, 2, . . . , Y v }.
j

(11)

i

PGM enforces hard one-to-one correspondences through
the bipartite graph and then utilizes a progressive strategy to
obtain many-to-many matching correspondences, while OTPM
directly learns soft many-to-many alignments via probabilistic
transport, better handling modality-specific cluster imbalances
and establishing more relabile inter-modality correspondences.
E. Debiased Memory Hybrid Learning
To mitigate the detrimental impact of noisy pseudo-labels on
contrastive learning and to effectively extract both modalityspecific and modality-invariant information, we introduce
the Debiased Memory Hybrid Learning (DMHL) module.
DMHL improves upon RPNR’s MHL by: (i) addressing the
overlooked sampling bias issue through debiased contrastive
learning, and (ii) employing clean-sample-only from stc training instead of using all available samples.
1) Debiased Contrastive Learning: Following ClusterContrast [28], previous some works [11], [13], [14], [15]
employed ClusterNCE Loss to update the feature extractor of
visible and infrared modalities, which can be fomulated as:

LtNCE qti , pt+

NB
X

exp qti · pt+ /τ
1 X

+
exp qti · ptk /τ
log
(12)
=−
t
t
NB
exp qi · p+ /τ
i=1
k,ŷti
„
ƒ‚
… „
ƒ‚
…
positive
negative

denotes the visible/infrared query sample and pt+
is the positive representations of qti , t ∈ {v, r}. Besides, NB
denotes the batch size and τ is a temperature hyperparameter.
This loss function brings the query samples and their corresponding positive representations closer while increasing the
distance from negative representations. As a result, it makes
the feature space distribution of samples within the same
class more compact while ensuring that samples from different
classes are more dispersed.
However, despite the use of the APCD module, pseudolabels remain inherently unreliable, meaning that noise can
still persist. As a result, some of the negative samples in
Eq. (12) may actually be false negative samples, i.e., the
query sample could inadvertently share the same class as the
negative samples, which is called sampling bias in [16]. Let
the data distribution be xt ∼ P(xt ) over S ct , and Z(xt ) denotes
t
the class label of xt . Assumed that (x+
, xt ) are similar pairs,
t
t
t t
implying that Z(x+ ) = Z(x ) = c , c ∈ {1, 2, . . ., Y t }. The
class distribution of ct is denoted by Ω(ct ). Consequently,
0
the probability that xt is observed as a positive sample for
0
t
t0
t0
t
xt is given by P+
xt (x0 ) = P(x |Z(x ) = Z(x )). In contrust,
t
the probability of x being observed as a negative sample is
where qti

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

10576

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025

0

0

0

denoted by P−xt (xt ) = P(xt |Z(xt ) , Z(xt )). Here, we assume
that π+ = Ω(ct ) is uniform, while π− = 1 − π+ indicates
the probability of belonging to any other class. Therefore, to
address the sampling bias issue, we can decompose the data
0
distribution P(xt ) as:
0

0

0

− − t
t
P(xt ) = π+ P+
xt (x ) + π P xt (x ).

(13)

0

0

From Eq. (13), we can derive that P−xt (xt ) = (P(xt ) −
+ + t0
π P xt (x ))/(1 − π+ ). Therefore, the negative term in Eq. (12)
can be replaced with Eq. (15), which will lead the model to
alleviate the adverse effects of negative sample bias. Thus, the
debiased contrastive loss can be formulated as:
N

LtDCL (qti , pt+ ) = −

B
exp (qti · pt+ /τ)
1 X
,
log
NB
exp (qti · pt+ /τ) + Neg
i=1
„ƒ‚…
ƒ‚
…
„

positive

Neg =

1
1 − π+

where ŷvi and ŷrj are the pseudo-labels of query features fiv and
f jr . Mv [ŷvi ] and Mr [ŷrj ] denote the positive representations of
query features fiv and f jr , respectively.
During the backward-propagation stage, the two modalityspecific memory banks are updated by a momentum update
strategy:

(14)

negative

0
1
X
@
exp (qti · ptk /τ) − π+ exp (qti · pt+ /τ)A .
k,ŷti

(15)
2) Memory Hybrid Learning: We begin by initializing two
v
modality-specific memory banks Mv ∈ RY ×d and Mr ∈
r
RY ×d with visible and infrared prototypes. However, these two
modality-specific memory banks only store modality-specific
information, which limits their ability to capture modalityinvariant information and reduce cross-modality discrepancies.
To address this, we leverage the cross-modality correspondences obtained from OTPM and propose a memory hybrid
learning strategy.
r
To begin, we create a modality-hybrid memory Mh ∈ RY ×d
that consolidates modality-shared information by blending the
aligned visible and infrared prototypes as follows:
phi = α × pri + (1 − α) × pr→v
,
i
Mhi ← phi ,

(16)

where i ∈ {1, 2, . . ., Y r } and pr→v
denotes the visible prototype
i
which matches with the infrared prototype pri . α is a balancing
hyperparameter that balances the fusion information of the visible and infrared prototypes. Experimental results demonstrate
that the infrared prototype exhibits superior reliability. When
employed as a fusion reference, it significantly enhances the
robustness of the hybrid memory bank.
Subsequently, in the representation learning phase, we adopt
a commonly used memory-based approach [11], [12], [13],
[15], [33], which alternates between two key stages: (1)
conducting contrastive learning during forward propagation
(FP) and (2) updating the memory bank during backward
propagation (BP). To enhance representation learning, we
employ a multi-memory joint contrastive learning strategy,
comprising both modality-specific and modality-invariant contrastive learning.
3) Modality-Specific Contrastive Learning: Based on the
modality-specific memory Mv and Mr , the debiased contrastive loss is applied to learn modality-specific information
for network optimization by:
L MS = LvDCL ( fiv , Mv [ŷvi ]) + LrDCL ( f jr , Mr [ŷrj ]),

(17)

Mv [ŷvi ] ← µMv [ŷvi ] + (1 − µ) fiv ,

(18)

Mr [ŷrj ] ← µMr [ŷrj ] + (1 − µ) f jr ,

(19)

where µ is the momentum updating factor.
4) Modality-Invariant Contrastive Learning: In contrast to
the two modality-specific memories, we perform modalityinvariant contrastive learning on the modality-shared memory
Mh to capture modality-invariant features while minimizing
cross-modality discrepancies. Building on the approach of
PGM [11], we adopt an alternating contrastive learning strategy on Mh :
(
Lv ( f v , Mh [ŷv→r
]), if E poch%2 = 0,
i
L MI = rDCL ir
(20)
LDCL ( fi , Mh [ŷri ]),
if E poch%2 = 1,
where ŷv→r
denotes the visible pseudo-label ŷvi matched with
i
the infrared pseudo-label ŷri . Then, the modality-shared memory is updated jointly by visible and infrared query features:
Mh [ŷv→r
] ← µMv [ŷiv→r ]+(1 − µ) fiv ,
i
M [ŷri ] ← µMr [ŷri ] + (1 − µ) fir ,

if E poch%2 = 0,

if E poch%2 = 1.

h

(21)

The total loss of the DMHL module is:
LDMHL = L MS + k1 L MI .

(22)

F. Neighbor Relation Learning
To effectively utilize noisy pseudo-labels, we introduce the
Neighbor Relation Learning (NRL) module, which aims to
mitigate intra-class variability through complex interactions
across all pair-wise noisy samples. In this module, we train
using samples from S nt . For simplicity, we focus on the process
for visible samples. Given a pair of samples ( fiv , f jv ), the
Euclidean distance between them denotes as divj , then the NRL
loss of the visible modality can be formulated as follows:
N

LvNRL =

N

N

N

B X
B
B X
B
1 X
1 X
ωvij divj 2 +
(1 − ωvij )[γ − divj ]2+ ,
NB
NB
i=1 j=1
i=1 j=1
„
ƒ‚
… „
ƒ‚
…

attracting

repelling

(23)
where NB denotes the number of samples in each iteration
and γ is a margin hyperparameter. [x]+ denotes max(0, x).
Moreover, ωvij is the weight term, formulated by a Gaussian
kernel function based on the Euclidean distance:
!
|| fiv − f jv ||22
v
ωi j = exp −
,
(24)
σ
where σ represents the kernel bandwidth and ωvij ∈ (0, 1].
Obviously, it can be used to measure the similarity relation of
paired samples in the embedding space.
As illustrated in Eq. (23), the NRL loss consists of
two components: an attracting term and a repelling term.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

YIN et al.: ADAPTIVE PSEUDO-LABEL PURIFICATION AND DEBIASING FOR USVI-ReID

10577

The attracting term draws positive pairs closer together, while
the repelling term forces negative pairs to move apart, maintaining a margin of γ. Notably, the NRL loss doesn’t rely on
pseudo-labels that may contain noise.
Similarly, the loss of the infrared modality is defined as:
N

LrNRL =

N

N

N

B X
B
B X
B
1 X
1 X
ωrij dirj 2 +
(1 − ωrij )[γ − dirj ]2+ .
NB
NB
i=1 j=1
i=1 j=1
„
ƒ‚
… „
ƒ‚
…

attracting

repelling

(25)
Fig. 3. Example images from VI-ReID datasets: (a) Sample images from the
SYSU-MM01 dataset. (b) Sample images from the RegDB dataset.

The overall loss of the NRL module is given by:
LNRL = LvNRL + LrNRL .

(26)

G. Optimization
Algorithm 1 :Training Procedure for APPD
Input: Unlabeled VI-ReID training data D = {DV , DR }.
Initialize: ResNet50 fθ pretrained on ImageNet; maximal
epoch E; maximal iteration N.
1 for epoch = 1 : E do
2
Creating pseudo-labels for unlabeled RGB and IR data
using DBSCAN, respectively;
3
Calibrating noisy pseudo-labels by Eq. (1)–(7);
4
Dividing pseudo-labels into clean set S ct and noisy set
S nt by Eq. 9;
5
Initializing RGB memory Mv and IR memory Mr
with reliable RGB and IR prototypes;
6
Establishing cross-modality correspondences by Optimal Transport in Eq. (10);
7
Initializing hybrid memory Mh by Eq. (16);
8
for iter = 1 : N do
9
if index in S ct then
10
Training with LDMHL of Eq. (22) and updating
memory bank by the momentum strategy;
11
end if
12
if index in S nt then
13
Training fθ using LNRL in Eq. (23) without using
pseudo-labels information.
14
end if
15
end for
16 end for
The training procedure for APPD is summarized in Algorithm 1 and the total training loss of the network can be
formulated as follows:
L = LDMHL + k2 LNRL
= L MS + k1 L MI + k2 LNRL .

(27)

where k1 , k2 are balancing coefficients, which are set to 0.5
and 10.0, respectively.
IV. E XPERIMENT
A. Experiment Setting
1) Datasets: As shown in Fig. 3, we adopt the SYSUMM01 and RegDB datasets as benchmarks to evaluate the

effectiveness of our method. SYSU-MM01 stands as a largescale, publicly available benchmark tailored for the VI-ReID
task, boasting a diverse collection of 491 identities captured
across four visible cameras and two infrared cameras, spanning
both indoor and outdoor environments. Within this dataset, a
total of 22,258 visible images and 11,909 infrared images,
portraying 395 distinct identities, have been meticulously
curated for training purposes. During the inference phase, the
query set encompasses 3,803 infrared images, representative
of 96 unique identities, while the gallery set comprises 301
randomly selected visible images. In contrast, the RegDB
dataset, captured by a single visible camera and a single
infrared camera, features 4,120 visible images and 4,120
infrared images, each depicting 412 distinct identities. To
elaborate further, the dataset is partitioned into two disjoint
sets: one designated for training and the other for testing.
2) Evaluation Metrics: The experiment of APPD was carried out following the evaluation metrics in DDAG [56],
i.e., Cumulative Matching Characteristic (CMC) and mean
Average Precision (mAP). In the evaluation of our proposed
method on the SYSU-MM01 dataset, we consider two distinct search modes: the All Search mode and the Indoor
Search mode. Similarly, for the RegDB dataset, our method
is evaluated across two testing modes: Visible2Thermal and
Thermal2Visible.
3) Implementation Details: Our proposed method is executed on two TITAN RTX GPUs utilizing the PyTorch
framework. Consistent with the methodology in [15], we
leverage a two-stream feature extractor that has undergone pretraining on ImageNet so as to extract 2048-dimensional visible
and infrared features. The Adam optimizer is adopted to train
the network with a weight decay of 5e-4. The initial learning
rate is set to 3.5e-4, which decays to 1/10 of its previous
value every 20 epochs. Throughout the training process, every
input image is scaled to a resolution of 288 × 144, and image
enhancement is carried out using the augmentation strategies
specified in [15]. The number of training epochs is set to 100.
In the first 50 epochs, we employ the DCL [15] framework to
alternately offline pseudo-labels generation and online representation learning. The proposed framework is trained in the
last 50 epochs. Additionally, we store multiple proxies for
each cluster to provide complementary representation at each
stage when constructing the memory following [27], [30], [34].
The parameter κ for κ-reciprocal nearest neighbors is set to

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

10578

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025

TABLE I
C OMPARISONS W ITH S TATE - OF - THE -A RT M ETHODS ON SYSU-MM01 AND R EG DB, I NCLUDING SVI-R E ID, SSVI-R E ID, AND USVI-R E ID M ETHODS .
A LL M ETHODS A RE M EASURED BY R ANK -1 (%) AND M AP (%). GUR* D ENOTES THE R ESULTS W ITHOUT C AMERA I NFORMATION . † D ENOTES
THE R ESULTS R EPRODUCED F ROM THE O FFICIAL C ODE

30 following [75] and K is fixed to 20. The hyperparameter
λ is set to 25 following [38]. Following ADCA [15], the
momentum value µ is set to 0.1 and the temperature τ is 0.5.
The margin hyperparameter γ and the kernel bandwidth σ are
both set to 1.0 following [76]. The trade-off hyperparameter
α is set to 0.5, k1 and k2 is set to 0.5 and 10.0, respectively.
π+ is set to 0.1 following [16].
B. Comparison With State-of-the-Art Methods
To comprehensively illustrate the efficiency of our proposed
APPD, we not only compare our method with SOTA unsupervised visible-infrared person ReID (USVI-ReID) methods
but also explore the promising application of our method
on supervised visible-infrared person ReID (SVI-ReID) and
semi-supervised visible-infrared person ReID (SSVI-ReID)
methods. Unless otherwise specified, all analyses are conducted on the SYSU-MM01 dataset under the All Search
mode.

1) Comparison with USVI-ReID Methods: As shown in
Tab. I, we can see that the proposed APPD gains the best
performance compared to existing state-of-the-art USVI-ReID
methods on the SYSU-MM01 and RegDB datasets. To be
specific, for the All Search mode on SYSU-MM01, our
APPD achieves 64.7% in Rank-1 and 60.5% in mAP, which
surpasses MMM by 3.1% in Rank-1 and 2.6% in mAP. For
the Indoor Search mode on SYSU-MM01, APPD also exhibits
outstanding performance, which gains the performance with
70.6% in Rank-1 and 75.3% in mAP, exceeding MMM by a
large margin with 6.2% in Rank-1 and 4.9% in mAP. Morever,
the performance on RegDB achieves 91.1% in Rank-1 and
83.8% in mAP under the Visible2Thermal mode, and 90.4%
in Rank-1 and 82.8% in mAP under the Thermal2Visible
mode. Compared to our previous work RPNR, our APPD
demonstrates a notable performance improvement, achieving
a 1.2% increase in average Rank-1 accuracy and a 1.0% boost
in average mAP. The results strongly validate the effectiveness
of our proposed approach, showcasing that APPD generates

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 163 (2025) 111443

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Adaptive transformer with Pyramid Fusion for cloth-changing Person
Re-Identification
Guoqing Zhang a,b , Jieqiong Zhou a , Yuhui Zheng a , Gaven Martin c , Ruili Wang b,d,e ,∗
a

School of Computer Science, Nanjing University of Information Science and Technology, Nanjing, China
School of Mathematical and Computational Sciences, Massey University, Auckland, New Zealand
c
Institute for Advanced Study, Massey University, Auckland, New Zealand
d
School of Computer Science, University of Nottingham, Ningbo, China
e
School of Data Science and Artificial Intelligence , Wenzhou University of Technology, Wenzhou, China
b

ARTICLE

INFO

Keywords:
Cloth changing
Person re-identification
Vision transformer

ABSTRACT
Recently, Transformer-based methods have made great progress in person re-identification (Re-ID), especially
in handling identity changes in clothing-changing scenarios. Most current studies usually use biometric
information-assisted methods such as human pose estimation to enhance the local perception ability of clotheschanging Re-ID. However, it is usually difficult for them to establish the connection between local biometric
information and global identity semantics during training, resulting in the lack of local perception ability
during the inference phase, which limits the improvement of model performance. In this paper, we propose
a Transformer-based Adaptive-Aware Attention and Pyramid Fusion Network (𝐴3 𝑃 𝐹 𝑁) for CC Re-ID, which
can capture and integrate multi-scale visual information to enhance recognition ability. Firstly, to improve the
information utilization efficiency of the model in cloth-changing scenarios, we propose a Multi-Layer Dynamic
Concentration module (MLDC) to evaluate the importance features at each layer in real time and reduce the
computational overlap between related layers. Secondly, we propose a Local Pyramid Aggregation Module
(LPAM) to extract multi-scale features, aiming to maintain global perceptual capability and focus on key local
information. In this module, we also combine the Fast Fourier Transform (FFT) with self-attention mechanism
to more effectively identify and analyze pedestrian gait and other structural details in the frequency domain
and reduce the computational complexity of processing high-dimensional data in the self-attention mechanism.
Finally, we build a new dataset incorporating diverse atmospheric conditions (for instance wind and rain) to
more realistically simulate natural scenarios for the changing of clothes. Extensive experiments on multiple
cloth-changing datasets clearly confirm the superior performance of 𝐴3 𝑃 𝐹 𝑁. The dataset and related code are
available on the website: https://github.com/jieqiongz1999/vcclothes-w-r.

1. Introduction
Person Re-Identification (Re-ID) strives to identify the same person across different cameras and plays a vital role in public safety.
However, to date most person Re-ID methods [1–3] use clothing as
discriminative information to deal with obstacles such as item occlusion
and perspective changes. However, in a real-world scenario, such as
criminal tracking, clothing change is a common evasion strategy, and
traditional short-term Re-ID technology cannot effectively deal with
this, as shown in Fig. 1. Therefore, it is important to study more
targeted cloth-changing Person Re-ID methods.

The CC Re-ID task aims to extract identity information unaffected
by clothing changes [4–6]. One category of approaches focus on identifying clothing-independent features, such as body outlines, posture
key-points and gait information. For example, Yang et al. [7] proposed
a network that can adapt to clothing changes around pedestrian silhouette sketches, but is affected by environmental factors such as lighting
and occlusion, and may ignore key details such as facial features.
The other category of approaches focus on separating identity and
clothing features such as GAN and semantic-guided clothing erasure
network [8]. However this usually brings challenges such as additional
computational overhead, high computational requirements, and strong
dependence on data quality.

∗ Corresponding author at: School of Mathematical and Computational Sciences, Massey University, Auckland, New Zealand.

E-mail addresses: guoqingzhang@nuist.edu.cn (G. Zhang), jieqiongz331@nuist.edu.cn (J. Zhou), zheng_yuhui@nuist.edu.cn (Y. Zheng),
G.J.Martin@massey.ac.nz (G. Martin), ruili.wang@massey.ac.nz (R. Wang).
https://doi.org/10.1016/j.patcog.2025.111443
Received 16 July 2024; Received in revised form 15 January 2025; Accepted 5 February 2025
Available online 12 February 2025
0031-3203/© 2025 Published by Elsevier Ltd.

Pattern Recognition 163 (2025) 111443

G. Zhang et al.

Fig. 1. Visualization of the Top-8 ranking lists generated by MGN [3] on the MSMT17 and Celeb-reID datasets. Images with red boxes indicate incorrect matches.

Recently, Vision Transformer [9] (ViT) has demonstrate remarkable
performance in various computer vision tasks [10,11] with its multihead self-attention mechanism to effectively capture inter dependencies
within an image. However, a common limitation of the Transformer
architecture is tendency to utilize the output of a specific layer for
representation, often neglecting other valuable information embedded
in the other layers. Fig. 2 shows the attention map visualization of the
first three layers and the last three layers of the ViT model. It can be
clearly observed that there are significant differences in the focus of
attention in different layers. In addition, although existing methods
often improve the local perception ability of pedestrian Re-ID after
changing clothes through strategies such as posture estimation, but how
to effectively coordinate local details with global semantic information
remains a challenge.
To mitigate these limitations, we propose a Transformer based
Adaptive-Aware Attention and Pyramid Fusion Network (𝐴3 𝑃 𝐹 𝑁) for
CC Re-ID. We firstly design a Multi-Layer Dynamic Concentration module (MLDC) to integrate the characteristics of each layer of ViT and
reduce the redundancy between layers. MLDC fuses different layer features through weighting and adjusts the importance of each layer in real
time. Subsequently, recognizing that each layer of the ViT model concentrates on different aspects of the image, we propose a Local Pyramid
Aggregation Module (LPAM) to extract multi-scale features, thereby
maintaining attention to global perception and key local information.
In this module, we also innovatively integrate a Fast Fourier Transform
(FFT) into the self-attention mechanism to effectively identify subtle
pedestrian differences in the frequency domain (such things as gait
and clothing texture) to improve both computational efficiency and
accuracy. Finally, since the existing Re-ID datasets do not consider the
impact of weather, we propose the VC-Clothes-W&R dataset to fill this
gap by introducing wind and rain elements.
Our primary contributions are the following:

Fig. 2. Attention maps of the first three layers (first row) and the last three layers
(second row) of ViT.

2. Related works
2.1. Classical person Re-ID
Current research has focused on solving problems such as lighting
changes [12], occlusion [13], and cross resolution [14]. Jiang et al.
[11] proposed a novel cross-modal Transformer (CMT) that jointly
explores modal-level alignment modules and instance-level modules for
visible-infrared person Re-ID, aiming to alleviate the loss of modalityspecific information caused by existing methods integrating different
modalities into a unified feature space. A Pose-guided Feature Decoupling (PFD) method proposed by Wang et al. [13] utilizes pose
information to effectively decouple semantic components (such as human body or joint parts), and aligns unoccluded parts accordingly.
Zhang et al. [14] proposed a Deep High-Resolution Pseudo-Siamese
Framework (PS-HRNet), which introduces the VDSR-CA module to
restore the resolution of low-resolution images and fully utilize the
different channel information of feature maps, while using the new
representation in HRNet to extract distinguishing features, thereby
achieving excellent performance in cross-resolution scenarios. In addition, unsupervised Re-ID is also a key research focus: DHA [15]
proposed an auto encoder-based method to generate deep latent attributes without extensive annotations, thus enhancing the ability to
extract features from sparse but discriminative data to identify individuals within clues and reduce reliance on labeled data. IPES-GAN
[16] adopts loop generation to adaptively balance environment and
identity features to achieve domain adaptation, which significantly
improves the robustness to environmental changes and camera settings
in different domains.

∙ We propose a Transformer-based Adaptive Aware Attention and
Pyramid Fusion Network for CC Re-ID;
∙ We integrate the Fast Fourier Transform into the self-attention
mechanism to improve the model’s ability of identifying pedestrian features in the frequency domain and optimize computing
efficiency;
∙ We propose the VC-Clothes-W&R dataset, which fills the missing
natural weather factors in existing pedestrian re-identification
datasets by introducing wind and rain elements.
The remainder of the paper is organized as follow: Section 2
presents some related works and the details of our proposed framework
are described in Section 3. Section 4 outlines the experimental setup
and presents the results of extensive experiments on diverse datasets.
Ablation studies are reviewed in Section 5, and Section 6 presents our
conclusions.
2

Pattern Recognition 163 (2025) 111443

G. Zhang et al.

2.2. Person Re-ID under intensive cloth variations

3.1. Multi-layer dynamical concentration module

As public safety concerns become increasingly prominent, especially in the fields of monitoring and safety, there is a pressing need
for effective identification of potential threats. Therefore, accurate
identification of individuals who change their attire becomes crucial
to promptly detect and intervene in potential security risks. These
concerns have spurred many scholars to conduct in-depth research
on CC Re-ID. In recent years, some related cloth-changing datasets
have been released, such as VC-Clothes [17], Celeb-reID [18], CelebreID-light [19] and NKUP [20]. In these datasets, the same individual
switches among multiple outfits, and wears various accessories, such
as sunglasses, scarves, backpacks, etc. Frequent changes of clothing
greatly reduce the reliability of traditional appearance-based matching
methods.
To cope with the challenges brought by changing clothes, some
works learn clothing-independent features with the help of identityrelated auxiliary biological cues. For example, Hong et al. [4] proposed
a shape–appearance mutual learning framework (FSAM), which is a
dual-stream structure that acquires the detailed discriminative body
shape information in shape stream and enriches the appearance stream
with non-fabric-related details. Zhang et al. [21] proposed a novel
Multi-Biometric Unified Network (MBUNet), which applies adaptive
graph convolution to obtain relevant information between key points
of the human body, and combines multiple biological features such as
the person’s head, neck, shoulders to mitigate the influence of clothing
alterations. However, these methods have high requirements on image
quality, and when the image is affected by occlusion, low illumination
and so forth, this will limit the extraction of identity-related features,
thus limiting the performance of the model. To further reduce the
dependence on collecting a large amount of clothing change data, PosNeg [22] introduced an innovative data augmentation strategy, using
positive augmentation and negative augmentation techniques to enrich
the ID feature space and generate out-of-distribution synthetic samples,
thereby enhancing the model’s robustness to clothing changes.
Another very common methods seek to segregate clothing-related
features from irrelevant features, enabling the model to concentrate
on acquiring clothing-independent identity information. Xu et al. [8]
proposed AFD-Net, which uses GAN and semantic perception models
to distinguish the appearance and structural features of pedestrian
images to achieve the separation of identity and clothing features,
thereby enabling the model to learn identity Discriminating features.
Similarly, SAVS [23] first locates the human body and clothing area
according to the human body semantic segmentation, and introduces
the human body semantic attention module to emphasize the human
body information. Furthermore, it shields the clothing area to make the
model focus on the extraction of visual semantic information unrelated
to clothing. However, these kinds of methods generally face a challenge: in the process of separating clothing features from non-clothing
features, distorted details are inevitably generated and the accurate
expression of cloth-irrelevant features may be weakened, resulting in
unstable training processes and poor model performance. Considering
the limitation of the above two types of methods, we do not use any
biological auxiliary branches or feature decoupling to help distinguish
individuals, but make full use of the differences in features of each layer
of Transformer to learn identity-related features. Specific introduction
will be shown in the next section.

In image processing, the Transformer architecture builds a visual
feature hierarchy layer-by-layer, from edge and texture detection at
the primary layer to scene comprehension at the high-level layer.
However, previous Re-ID models often only focus on the information
of the terminal layer, while ignoring the fine details of the primary
and intermediate layers. To make up for this deficiency, we propose
the Multi-Layer Dynamical Concentration Module (MLDC) (Fig. 3).
This model dynamically synthesizes features across layers and also
includes the key visual information from each layer in the final feature
representation.
Calculation of weights. In order to effectively perform multilayer feature fusion, in our method, we assign a weight coefficient
𝑤𝑖 (i = 1...12) to each layer, the purpose of which is to evaluate the
feature importance of each layer in real time and reduce the similarity
redundancy of related layers, and the specific calculation process of 𝑤𝑖
is as follows:
)
(
𝐿
|⟨𝐹𝑖 , 𝐹𝑗 ⟩|
𝑒𝑥𝑝 𝑓𝑖 − 𝛼 𝛴𝑗=1,𝑗≠𝑖
(1)
𝑤𝑖 =
),
(
𝐿 𝑒𝑥𝑝 𝑓 − 𝛼 𝛴 𝐿
|⟨𝐹𝑘 , 𝐹𝑚 ⟩|
𝛴𝑘=1
𝑘
𝑚=1,𝑚≠𝑘
where 𝐹𝑖 ∈ R𝑁×𝐷 represents the output of the 𝑖th layer, N is the
number of image blocks and D is the feature dimension of each token,
⟨⋅, ⋅⟩ is the inner product, which measures the feature correlation of
different layers, 𝛼 is a regularization coefficient used to scale the impact
of orthogonality constraints and reduce feature overlap between layers,
𝐿 is the total number of layers. And 𝑓𝑖 is a one-dimensional scalar that
represents the importance of the output feature 𝐹𝑖 of each layer, the
specific calculation formula is as follows:
1∑
𝑚𝑒𝑎𝑛(𝐴𝑖𝑡 ),
ℎ 𝑡=1
ℎ

𝑓𝑖 =

(2)

where h represents the number of attention heads in each layer, 𝑚𝑒𝑎𝑛(⋅)
represents the mean of all elements, 𝐴𝑖𝑡 ∈ R𝑁×𝑁 represents the
attention score matrix of the 𝑡th head in the 𝑖th layer (t, i = 1,2, . . . ,12),
and the formula is as follows:
(
)
𝑄𝑖𝑡 𝐾𝑖𝑡𝑇
𝐴𝑖𝑡 = softmax
,
(3)
√
𝑑𝑡
where 𝑄𝑖𝑡 ∈ R𝑁×𝑑𝑡 and 𝐾𝑖𝑡 ∈ R𝑁×𝑑𝑡 are the query and key matrices of
is the dimension size of each
the 𝑡th head in the 𝑖th layer, and 𝑑𝑡 = 𝐷
ℎ
head, which is used to scale the dot product result to prevent too large
values from affecting the gradient of the 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥(⋅) function.
Enhanced Feature Fusion With Regularization. To mitigate the
risk of model over-fitting that may occur due to the undue influence
of specific layers, we incorporate an 𝐿2 regularization term into our
feature fusion formula:
𝐿
𝐹𝑓 𝑢𝑠𝑖𝑜𝑛 = 𝛴𝑖=1
𝑤𝑖 𝐹𝑖 − 𝜆 ⋅ ‖𝑊 ‖2𝐹 ,

(4)

where 𝜆 is a non-negative regularization parameter employed to mitigate over-fitting by constraining the magnitude of the weights within
the model. ‖𝑊 ‖2𝐹 is the Frobenius norm of the weight matrix 𝑊 and
is the sum of the squares of all layer weights.
3.2. Local pyramid aggregation module

3. The proposed method

Although the adaptive weighted average method is very effective in
integrating multi-layer features to achieve comprehensive representation, its global fusion method may ignore local information.
In order to capture image details at various scales, we design a
local pyramid aggregation module (LPAM), as shown in Fig. 3. The
module adopts a pyramid structure and gradually fuses the output
of four different layers 𝐹𝑙1 , 𝐹𝑙2 , 𝐹𝑙3 , 𝐹𝑙4 to achieve the extraction of
different fine-grained information, the specific layer selection is provided in the ablation study section. For example, in the local feature

This section elaborates on our proposed approach. We first introduce a Multi-Layer Dynamical Concentration Module to evaluate the
significance of features at each layer in real time while minimizing computational redundancy among highly correlated layers in Section 3.1. In
Section 3.2, we further adopt a Local Pyramid Aggregation Module to
enhance multi-scale features and integrate Fast Fourier Transform (FFT)
to optimize the self-attention mechanism. Finally, the optimization of
the overall framework is described in Section 3.3.
3

Pattern Recognition 163 (2025) 111443

G. Zhang et al.

Fig. 3. The architecture of our proposed 𝐴3 𝑃 𝐹 𝑁, which is built on ViT and contains two parallel modules–Multi-Layer Dynamical Concentration Module (MLDC) and Local
Pyramid Aggregation Module (LPAM). MLDC aims to obtain aggregated global features by dynamically assigning weights to each layer. LPAM is designed to fuse multi-level
features through a pyramid structure to obtain multi-scale information.

transformations on 𝑋 to convert it into Q, Kand V. Subsequently, Q, K
and V are split into multiple heads. To improve the efficiency of Fast
Fourier Transform (FFT), we appropriately fill the Q and K matrices to
the integer power of 2, and then apply FFT on the filled 𝑄𝑝𝑎𝑑 𝑑 𝑒𝑑 and
𝐾𝑝𝑎𝑑 𝑑 𝑒𝑑 and estimate their correlation in the frequency domain. The
output is formulated as below:
𝐴𝑡𝑡𝑛 = 𝑆 𝑜𝑓 𝑡𝑚𝑎𝑥(𝐹 −1 (𝐹 (𝑄𝑝𝑎𝑑 𝑑 𝑒𝑑 ) ⊙ 𝐹 (𝐾𝑝𝑎𝑑 𝑑 𝑒𝑑 ))[∶, ∶, ∶, ∶ 𝑄.𝑠𝑖𝑧𝑒(−1)]), (6)
where 𝐹 (⋅) and 𝐹 −1 (⋅) represent FFT and inverse FFT (IFFT) respectively, ⊙ is a dot product operation. Softmax function aims to normalize
the result to produce attention weights 𝐴𝑡𝑡𝑛. Finally, we obtain the
attention-weighted output through weighted calculation and residual
connection.

Fig. 4. Illustration of FFT Self-Attention, which aims to identify detailed information
of pedestrians from a frequency domain perspective.

3.3. Model optimization
aggregation (LFA) of LPAM, we implement a 1 × 1 convolution layer
and BatchNorm2D processing on features 𝐹𝑙1 and 𝐹𝑙2 , with the help
of ReLU function for size adjustment and nonlinear enhancement. In
addition, we introduce a self-attention mechanism to obtain enhanced
pedestrian information from the frequency domain perspective (see
Fig. 4). Finally, the convolved 𝐹𝑙1 and 𝐹𝑙2 are connected, and then input
into the convolution block to achieve feature fusion. The formula for
feature fusion is as follows:
(
( ( ) ( )))
𝐹𝑚𝑖𝑑1 = 𝜚 concat 𝜚 𝐹𝓁1 , 𝜚 𝐹𝓁2
,
(5)

We optimize the model through ID loss and triplet loss. The ID
loss adopts the traditional cross-entropy loss function, excluding label
smoothing, and its specific definition is as follows:
𝐿𝐼 𝐷 = −

𝐶
∑

𝑦𝑗 𝑙𝑜𝑔(𝑝𝑗 ),

(7)

𝑗

where 𝐶 is the number of categories, 𝑦𝑗 is the one-hot encoding of the
true label and 𝑝𝑗 is the probability that the model predicts that the
sample belongs to the 𝑗th category.
In triplet loss, we adopt a semi-hard sample mining strategy to
solve the over-fitting problem caused by noisy samples in hard sample
mining. This strategy selects samples that are predicted incorrectly
but with low confidence during the training phase to improve the
model’s coverage of a broad sample set and mitigate the impact of noisy
samples. The indicator function of semi-hard samples is as follows:
{
1 if 𝑑(𝑎𝑝) + 𝑚 < 𝑑(𝑎𝑛) < 𝑑(𝑎𝑝) + 𝑀
𝐼semi-hard (𝑥𝑎𝑙 , 𝑥𝑝𝑙 , 𝑥𝑛𝑙 ) =
(8)
0 otherwise ,

where 𝜚(⋅) represents the convolution block and 𝑐 𝑜𝑛𝑐 𝑎𝑡(⋅, ⋅) refers to the
splicing operation. The following fusion steps are similar to this. For
the detailed process, please refer to the framework diagram of the local
pyramid aggregation module (LPAM) shown in Fig. 3.
FFT Self-Attention. Fast Fourier Transform (FFT) is an effective
algorithm for computing the Discrete Fourier Transform, as described
in [24], which can significantly reduce the computational complexity
from 𝑂(𝑁 2 ) to 𝑂(𝑁 log 𝑁), making it crucial in signal frequency domain
analysis.
In our method, the self-attention module first accepts the input
𝑋 ∈ R𝑁×𝐷 , where N is the number of image blocks (tokens) and D is
the feature dimension of each token. Then, we perform different linear

where 𝑑(𝑎𝑝) represents the distance between the anchor sample 𝑥𝑎𝑙
and the positive sample 𝑥𝑝𝑙 , 𝑑(𝑎𝑛) represents the distance between the
anchor sample and the negative sample 𝑥𝑛𝑙 . 𝑚 means the minimum
distance difference between anchor samples and positive samples to
4

Pattern Recognition 163 (2025) 111443

G. Zhang et al.

Table 1
Clothing change statistics of the long-term pedestrian datasets used in our experiments. Please note: ‘SC’ and ‘CC’ represent the two modes of
same clothes and clothing change respectively.
Dataset

VC-Clothes
NKUP
Celeb-reID
Celeb-reID-light
PRCC
LTCC
VC-Clothes-W&R

Source

Synthetic
Real
Real
Real
Real
Real
Synthetic

Train(ID/Image)

256/9449
40/5336
1052/34,186
100/887
150/17,896
77/9576
256/9449

Test(ID/Image)
Query

Gallery

256/1020
39/332
420/2972
100/934
71/3543
75/493
256/1020

256/8591
67/4070
420/11,006
590/10,842
71/3384
75/7050
256/8591

Cameras

Data style

Weather backgrounds

4
15
Many
Many
3
12
4

SC/CC
CC
CC
CC
SC/CC
SC/CC
SC/CC

None
None
None
None
None
None
Wind & rain

prevent the model from only focusing on subtle differences between
similar samples, 𝑀 is the maximum distance between anchor samples
and negative samples to avoid selecting overly simple negative samples
and ensure that the model learns discriminative features. The triplet
loss for semi-hard sample mining is defined as follows:
𝐿𝑡𝑟𝑖 = −

𝑁
∑

(
𝐼semi-hard (𝑥𝑎𝑙 , 𝑥𝑝𝑙 , 𝑥𝑛𝑙 ) ‖𝑓 (𝑥𝑎𝑙 ) − 𝑓 (𝑥𝑝𝑙 )‖22

𝑙

)
−‖𝑓 (𝑥𝑎𝑙 ) − 𝑓 (𝑥𝑛𝑙 )‖22 + 𝑚 + ,

(9)

where 𝑓 (⋅) denotes the feature extraction operator that maps the input
image into an embedding space. ∥ ⋅ ∥2 represents the 𝐿2 -norm, the
Euclidean distance between two feature vectors. (⋅)+ is the hinge loss
function, which means that the loss is calculated only when the value in
the brackets is a positive number, otherwise the loss is 0. Consequently,
the comprehensive loss function of the model is defined as follows:
𝐿=

𝑇
∑

Fig. 5. Some images from VC-Clothes-W&R.

4.2. VC-Clothes-W & R
𝑡𝑞 (𝐿𝐼 𝐷𝑞 + 𝐿𝑡𝑟𝑖𝑞 ) (𝑞 = 1, … , 4) ,

(10)
As a synthetic dataset, VC-Clothes can provide a more controlled
experimental environment, allowing us to keep other variables (such as
lighting conditions and occlusions) relatively consistent across different
scenes. In contrast, real image datasets often have the complexity of
natural environments and introduce many uncontrollable variables,
such as fluctuating lighting conditions, varying degrees of occlusion
caused by unpredictable obstacles and other factors. These factors may
mask the specific impact of weather on the recognition task, making
it challenging to separate the variables required for testing. Therefore,
to enhance the diversity of the image backgrounds and increase the
model’s robustness, we add wind and rain scenes to the VC-Clothes
dataset [17]. The generated VC-Clothes-W&R dataset consists of images
captured from four different cameras, segmented into training and
testing sets. The training set comprises 256 unique identities with a
total of 9449 images. Similarly, the test set is structured into query
and gallery segments, maintaining the same 256 identities with 1020
images in the query and 8591 in the gallery set. Some sample images
are shown in Fig. 5 and the construction process is as follows:
Considering the complexity of precipitation levels and changes in
image brightness, we employ a refined atmospheric scattering model
[26] to generate rainy and windy scenes for images. Specifically, we use
𝐽 (𝑥, 𝑦) to represent a pixel in the original image and after adding the
wind and rain scene, the corresponding pixel 𝐼(𝑥, 𝑦) can be calculated
by the following formula:

𝑞

where 𝑇 = 4 represents the total number of output features in our
model and 𝑡𝑞 represents the weight of each output feature. While
assigning fixed weights to each part of the loss is simple, it may not
produce the best model performance. Therefore, we adapt the loss
function for each output feature to adaptive weights. Initially, the
loss weight 𝑡𝑞 for each output feature are expressed as the same size
and are subsequently dynamically adjusted during training through
back-propagation to obtain the optimal weights.

4. Experiments
4.1. Datasets and evaluation metrics
Datasets Details. To assess the performance of our proposed
𝐴3 𝑃 𝐹 𝑁, we carry out experimental evaluations on the publicly available cloth-changing pedestrian datasets, including VC-Clothes [17],
NKUP [20], Celeb-reID [18], Celeb-reID-light [19], LTCC [25] and
PRCC [7]. Table 1 provides an overview of these datasets. Meanwhile,
we build an enhanced dataset VC-Clothes-W&R for wind and rain
scenes based on VC-Clothes to make up for the lack of environmental
elements in existing datasets.

𝐼(𝑥, 𝑦) = 𝐿[𝐽 (𝑥, 𝑦)𝑡(𝑥, 𝑦) + 𝑅(𝑥, 𝑦)(1 − 𝑡(𝑥, 𝑦))𝑀(𝑥, 𝑦)],

Evaluation Metrics. We evaluate the performance of 𝐴3 𝑃 𝐹 𝑁 using
Rank-1 accuracy and mean precision (mAP) in three test scenarios: (1)
general scenarios, covering clothing changes and consistent samples;
(2) clothing change scenarios, only including clothing change samples; (3) clothing consistent scenes, only including consistent clothing
samples. In the following tables, ‘‘sil’’, ‘‘ga’’, ‘‘dg’’, ‘‘pose’’ and ‘‘bs’’ represent pedestrian semantic segmentation, gait, data generation, Human
Posture and body shape information.

(11)

where 𝐿[⋅] reflects the brightness coefficient, 𝑅(𝑥, 𝑦) represents the
radiant brightness of raindrops on the spatial coordinates (𝑥, 𝑦). Furthermore, 𝑀(𝑥, 𝑦) is the blur kernel used to simulate the width and diffusion
properties of visual distortion caused by raindrops. 𝑡(𝑥, 𝑦) represents the
medium transmission ratio at each point (𝑥, 𝑦), which quantifies the
light intensity due to the presence of raindrops at that specific location.
The formula is as follows:
𝑡(𝑥, 𝑦) = 𝑒−𝛽 𝑑(𝑥,𝑦)𝑠(𝑥,𝑦)𝑐 𝑜𝑠(𝜃) ,
5

(12)

Pattern Recognition 163 (2025) 111443

G. Zhang et al.
Table 2
Comparison with methods on VC-Clothes datasets (%).
Methods

Modality

Cross-clothes

Same-clothes

Rank-1

mAP

Rank-1

mAP

RGB
RGB+sil
RGB+dg

62.0
72.0
76.8

62.2
72.1
68.4

–
94.5
94.1

–
94.7
93.8

FSAM (CVPR 21) [4]
MBUNet (TIP 22) [21]
CAL (CVPR 22) [5]
ACID (TIP 23) [6]
AFL (TMM 23) [30]
DCR-ReID (TCSVT 23) [31]
MGP (TMM 23) [32]

RGB+bs
RGB+pose
RGB
RGB
RGB
RGB+bs
RGB+dg

78.6
82.3
81.4
84.3
82.5
83.7
81.8

78.9
68.2
81.7
74.2
83.0
82.6
81.7

94.7
95.7
95.1
95.1
–
94.6
94.7

94.8
94.2
95.3
94.7
–
94.5
94.9

Ours

RGB

89.2

83.1

96.6

95.6

Short-term based methods
PCB (ECCV 18) [27]
ISP (ECCV 20) [28]
DG-Net (CVPR 19) [29]
Cloth-changing based methods

where 𝛽 is the attenuation coefficient of the medium rain, 𝑑(𝑥, 𝑦) is the
depth of the medium at position (𝑥, 𝑦), 𝑠(𝑥, 𝑦) denotes the density of rain
at (𝑥, 𝑦) and 𝜃 is the angle of light propagation relative to the viewer.
Next, we use the fuzzy kernel function 𝑀(𝑥, 𝑦) to simulate the
scattering effect of raindrops on the imaging sensor and its effective
radius, and quantitatively describes the optical diffusion characteristics
caused by raindrops of different sizes. The kernel function is specifically
defined as follows:
2

methods MVSE [36], UCAD [8] SAVS [23] based on clothing changes
methods on the datasets NKUP collected in closed scenes. What is more,
to further verify the generalization of our model, we also conduct experiments on VC-Clothes-W&R. More specifically, FSAM, MGP, MBUNet,
SAVS, MVSE, and UCAD adopt multi-biological auxiliary modules to reduce the interference of variable appearance information; AFD-Net and
SARF introduce generative adversarial networks to decouple identityrelated and identity-independent features; other methods only use RGB
modality to convey identity information that is not affected by clothing.
Results for VC-Clothes. We evaluate our proposed 𝐴3 𝑃 𝐹 𝑁 against
seven methods based on cloth-changing and three short-term
approaches on the VC-Clothes dataset, as illustrated in Table 2. We
can notice that our method achieves the best results in both sameclothes and cross-clothes settings. Compared with FSAM [4], which
transfers fine-grained body shape knowledge from the shape to appearance stream to enhance cloth-independent features, our method
increases Rank-1 and mAP in cross-clothes scenarios by 10.6% and
4.2% respectively. This shows that the performance of FSAM is greatly
affected by the quality of body shape information. Compared with
ACID [6], a method that accumulates identity clues using a step-bystep competition strategy to accumulate identity clues, our method
increases Rank-1 and mAP in cross-clothes scenarios by 4.9% and
8.9% respectively. It shows that our method effectively integrates
fine-grained local information and global information, improving the
information utilization efficiency in processing large-scale data. DCRReID [30] proposed a component reconstruction decoupling (CRD)
module to separate clothing-related and unrelated features based on
human body component reconstruction, but it is difficult to perform
effectively when the image quality is low or there is occlusion. In
addition, although its deep assembly decoupling (DAD) module enhances feature discrimination, its high computational requirements
limit its application in resource-constrained environments. MBUNet
[21] solves the problem of frequent clothing changes in clothingchanging scenes by utilizing biological cues that are not related to
clothing (such as the head, neck, and shoulders), but these biological
features may greatly reduce their effectiveness due to occlusion or angle
changes between cameras. What is more, while the second best method
AFL [30] achieves good performance, the construction of correlation
factors requires to consider the differences and similarities of different
identities, which greatly increases the complexity of the model. In
contrast with AFL, our method not only achieves higher performance
improvements in the cross-clothes scenarios with a 6.7% improvement
in Rank-1 and a 0.1% improvement in mAP but also effectively reduces
the cost of experimental settings. This shows that our method performs
well in balancing high accuracy with lower computational and model
complexity.
Results on Celeb-reID and Celeb-reID-light. We contrast our
method with some recent short-term and cloth-changing Re-ID studies

2

1 − 𝑥 2𝑘+𝑦2
𝑒
,
(13)
2
where k represents the scale parameter of the raindrop scattering effect.
By adjusting k, we can simulate the scattering distribution caused
by raindrops of different sizes on the sensor, thus more accurately
reflecting the impact of raindrops on image quality.
The entire model not only simulates raindrops, their direction and
size, but also thoroughly considers the overall and local illumination
variations under rainy conditions, forming highly realistic and diverse
rainy scenarios.
𝑀(𝑥, 𝑦) =

4.3. Implementation details
Our method uses the Vision Transformer [9] (ViT-Base) as the
baseline model, which has 12 Transformer layers and each containing 768-dimensional embeddings, and can effectively process image
patches of 16 × 16 pixels. We resize the input images to [256, 128] to
adapt to the model’s processing requirements for patches. In addition,
we apply data augmentation techniques including random horizontal
flipping and random erasing, each with a probability of 0.5 to simulate
visual changes in the real world. During training, we adopt a softmax
triple sampling strategy with 4 instances per batch to enhance the
generalization ability of the model and use 8 worker threads for data
loading to improve data processing efficiency. The model is trained
for 120 epochs by stochastic gradient descent (SGD), and the starting
learning rate is set to 0.008. To optimize the learning rate adjustment
process, we adopt the linear warm-up technique and set a learning rate
decay factor of 1e−4. Every 10 epochs, we performed model evaluation
using batches of 64 images to regularly monitor model performance.
4.4. Performance evaluations and comparisons
For the artificially synthesized datasets VC-Clothes, we compare the
proposed 𝐴3 𝑃 𝐹 𝑁 with some short-term methods (i.e., PCB [27], ISP
[28], DG-Net [29]) and cloth-changing Re-ID methods (i.e., FSAM [4],
CAL [5], ACID [6], AFL [30], MGP [32]). In addition, we compare
our method with five clothes-changing based methods (i.e., RCSAN
[33], AFD-Net [34], MBUNet [21], SAFR [35], SAVS [23]) as well as
some short-term methods on high-resolution datasets Celeb-reID and
Celeb-reID-light collected by multiple cameras. Also, we compare with
6

Pattern Recognition 163 (2025) 111443

G. Zhang et al.
Table 3
Comparison with SOTA methods on Celeb-reID and Celeb-reID-light (%).
Methods

Modality

Celeb-reID

Celeb-reID-light

Rank-1

mAP

Rank-1

mAP

RGB
RGB
RGB+dg

10.0
8.2
50.1

49.0
37.1
10.6

13.9
9.0
23.5

21.5
16.7
12.6

RCSAN (ICCV 21) [33]
AFD-Net (IJCAL 21) [34]
MBUNet (TIP 22) [21]
SAFR (TIP 22) [35]
DCR-ReID (TCSVT 23) [31]
MGP (TMM 23) [32]
SAVS (TNNLS 23) [23]

RGB
RGB+GAN
RGB+pose
RGB+GAN
RGB+bs
RGB+bs
RGB+sil

55.6
52.1
55.3
56.0
60.8
60.5
65.9

11.9
10.6
12.1
14.2
15.7
16.1
21.3

29.3
22.2
33.9
29.5
33.5
32.8
–

16.7
11.3
21.3
16.7
22.0
21.5
–

Ours

RGB

61.4

16.9

40.6

24.2

Short-term based methods
MGN (ACMMM 18) [3]
PCB (ECCV 18) [27]
DG-Net (CVPR 19) [29]
Cloth-changing based methods

Table 4
Comparison with SOTA methods on NKUP (%).
Methods

Modality

Rank-1

mAP

RGB
RGB

18.7
20.6

14.1
16.1

MVSE (ACMMM 21) [36]
UCAD (IJCAI 22) [8]
MBUNet (TIP 22) [21]
DCR-ReID (TCSVT 23) [31]
MGP (TMM 23) [32]
SAVS (TNNLS 23) [23]

RGB+sil
RGB+bs
RGB+pose
RGB+bs
RGB+dg
RGB+sil

23.8
25.0
24.5
24.7
25.1
25.3

17.9
16.9
17.7
18.3
18.0
18.6

Ours

RGB

25.7

19.1

first cloth-changing dataset that adds weather background. Table 5
presents the comparison results between our method with short-term
and cloth-changing methods on this dataset, demonstrating that our
method achieves the best performance. Compared with ACID [6],
which achieves the second best performance, our method improves
Rank-1 and mAP by 3.6% and 5.2% respectively in the cross-clothes
setting. The main reason is that ACID gradually accumulates ID clues
through global, channel, and pixel-level feature extraction, which may
leads to feature redundancy and slow inference speed in large-scale
data processing. However, as shown in the comparison results in
Table 2, after introducing wind and rain scenarios, both our method
and the comparison methods experience a decline in experimental accuracy, and our model does not achieve the minimal decrease. Therefore,
further work is needed to address the impact of simulated weather
conditions, such as incorporating adaptive image restoration techniques
or introducing multi-modal data augmentation strategies to enhance
the model’s performance and robustness in weather environments.
Results for PRCC and LTCC. We compare with the state-of-theart methods on the PRCC and LTCC datasets, the results are shown
in Table 6. In cross-clothes settings, compared with the second-best
method MBUNet [21], our method improves Rank-1 by 1.5% and
0.6% on PRCC and LTCC, and improves mAP by 3.4% and 2.7%.
Compared with other methods, RCSAN [33] focuses on clothing state
perception, but performs poorly under dynamic scene changes; FASM
[4] enhances body shape and appearance features, but has limited
effect when data is sparse or changes extremely; GIReID [37] relies on
gait information from a single image, but its accuracy decreases when
the gait is subtle or blurred; MBUNet focuses on extracting clothingindependent biometrics, but fails when occlusion is severe. Our model
effectively improves the recognition accuracy and robustness under
variable clothing conditions by integrating the dynamic information
of each layer of the transformer and strengthening the capture of key
local information. In the same-clothes setting, AIM [38] achieves the
best results. It is an automatic intervention model based on causality,
which simulates causal intervention through a dual-branch model and
gradually separates clothing bias from entangled ID clothing representation without destroying semantic integrity. Compared with our
method, AIM’s Rank-1 is 0.2% and 4.2% higher on PRCC and LTCC,
and mAP is 1.8% and 4.4% higher. In future work, we plan to draw on
AIM’s causal inference methods to enhance our model’s ability to more
effectively distinguish identity and clothing changes.

Short-term based methods
PCB (ECCV 18) [27]
MGN (ACMMM 18) [3]
Cloth-changing based methods

on these two datasets, with the findings presented in Table 3. It
is apparent that our method surpasses all the comparative methods
on Celeb-reID-light. Relative to AFD-Net [34] and SARF [35], our
Rank-1 accuracy shows an enhancement of 18.4% and 11.1%, and
mAP increases by 12.9% and 7.5%, respectively. Since AFD-Net and
SARF have in common that they decouple information by generating
adversarial images and separate clothing regions through an additional
human parsing model, which increases the computational overhead. In
contrast, our method does not require the incorporation of additional
biological branches, but makes full use of the information differences
between different layers of ViT, effectively reducing the additional
computational burden while improving the accuracy of the model.
From Table 3, it is seen that our method achieves the second highest
performance on the Celeb-reID dataset and SAVS [23] obtains the best
results. Specifically, ours is 4.5% and 4.4% lower than SAVS in Rank-1
and mAP, respectively. The main reason is that SAVS masks clothing
clues through the visual masking module and reweights the visual
feature map in the human semantic attention module to effectively utilize human semantic information. This method of combining clothing
masking with biometrics provides new ideas for our future work.
Results on NKUP. Table 4 shows the comparison of our method
with the short-term and CC Re-ID methods on the NKUP dataset. It
can be seen that our methods significantly outperform all short-term
methods, and gives the best performance. These three cloth-changing
methods all use semantic segmentation or body shape as auxiliary cues
to support the learning of clothing-independent features. But they all ignore the local structural cues of pedestrians. Relative to the second best
method SAVS [23], our method improves Rank-1 and mAP by 0.4% and
0.5% respectively. This proves that our method successfully enhance
the discrimination of human body parts by aggregating features at
different scales.
Results for VC-Clothes-W&R. To further verify the generalization ability of our proposed method, we conduct comparative experiments on VC-Clothes-W&R. To the best of our knowledge, this is the

5. Ablation studies
Effectiveness of each module of our framework. To verify the
effectiveness of our modules, we conducted experiments on datasets
such as Celeb-reID, NKUP, Celeb-reID-light, VC-Clothes and VC-ClothesW&R. The experimental results are shown in Table 7. We can see
7

Pattern Recognition 163 (2025) 111443

G. Zhang et al.
Table 5
Comparison with SOTA methods on VC-Clothes-W&R (%).
Methods

Modality

Cross-clothes

Same-clothes

Rank-1

mAP

Rank-1

mAP

RGB+sil
RGB+dg

66.3
70.7

63.1
65.9

92.2
91.8

91.4
90.2

MBUNet (TIP 22) [21]
AFL (TMM 23) [30]
ACID (TIP 23) [6]
DCR-ReID (TCSVT 23) [31]
MGP (TMM 23) [32]

RGB+pose
RGB
RGB
RGB+bs
RGB+dg

78.1
78.9
80.5
80.1
77.5

70.6
70.2
71.3
72.0
72.3

90.9
91.6
91.1
91.9
92.4

91.1
91.3
90.8
91.4
92.1

Ours

RGB

84.1

76.5

93.1

92.7

Short-term based methods
ISP (ECCV 20) [28]
DG-Net (CVPR 19) [29]
Cloth-changing based methods

Table 6
Comparison with SOTA methods on PRCC and LTCC (%).
Methods

Modality

PRCC

LTCC

Cross-clothes

Same-clothes

Cross-clothes

Same-clothes

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

RGB
RGB+sil
RGB

22.9
36.6
53.5

–
–
53.3

86.9
92.8
98.2

–
–
98.4

23.5
27.8
25.0

10.0
11.9
12.6

61.8
66.3
68.4

27.5
29.6
34.6

RCSAN (ICCV 21) [33]
GI-ReID (CVPR 22) [37]
FSAM (CVPR 21) [4]
MBUNet (TIP 22) [21]
Chan et al (ACM 23) [39]
AIM (CVPR23) [38]

RGB
RGB+ga
RGB+bs
RGB+pose
RGB+dg
RGB

50.2
37.6
54.5
67.6
65.8
54.7

48.6
82.3
–
65.3
61.2
55.0

100
79.0
98.8
100
99.5
100

97.2
–
–
99.6
96.7
99.9

–
28.1
38.5
39.5
32.9
38.3

–
13.2
16.2
14.7
15.3
17.0

–
73.6
–
67.1
73.4
76.1

–
36.1
–
34.4
36.8
39.1

Ours

RGB

69.1

68.7

99.8

98.1

40.1

17.4

71.9

34.7

Short-term based methods
PCB (ECCV 18) [27]
ISP (ECCV 20) [28]
MGN (ACMMM 18) [3]
Long-term based methods

Table 7
Ablation study of components in our framework in Celeb-reID, NKUP, VC-Clothes (cross clothes), VC-Clothes-W&R (cross clothes), and Celeb-reID-light (%).
Methods

Baseline
+MLDC
+LPAM
+MLDC+LPAM
+MLDC+LPAM (FFT)

Celeb-reID

NKUP

VC-Clothes

Celeb-reID-light

VC-Clothes-W&R

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

57.3
59.1
60.2
60.9
61.4

15.6
16.3
15.9
16.8
16.9

21.5
22.5
23.3
24.2
25.7

15.5
16.9
17.7
18.3
19.1

83.7
85.3
84.8
87.1
89.2

80.6
81.9
80.3
82.3
83.1

29.1
33.4
32.7
36.1
40.6

20.3
21.7
21.2
23.9
24.2

80.3
81.9
81.5
83.1
84.1

74.4
75.2
74.9
75.8
76.5

that after the introduction of the MLDC module, the performance of
all datasets has improved, especially on Celeb-reID-light, where the
accuracies of Rank-1 and mAP have increased by 4.3% and 1.4%
respectively. This shows that the MLDC module effectively improves
the model’s adaptability to clothing changes by dynamically fusing
the multi-layer information of Transformer. When the proposed LPAM
is added alone, the performance of the baseline on most datasets is
further improved, especially the Rank-1 on NKUP and Celeb-reID-light
are increased by 1.8% and 3.6% respectively, highlighting the role of
the local pyramid aggregation module in extracting multi-scale features
and capturing the key role of local information.
When MLDC and LPAM are used simultaneously, the model’s performance of all datasets is significantly improved, especially on CelebreID-light, where the accuracies of Rank-1 and mAP are increased by
7.0% and 3.6% respectively. These results show that the simultaneous
use of multi-layer information fusion and multi-scale information can
further resist the interference caused by clothing changes. Next, to
further verify the superiority of the FFT self-attention mechanism we
proposed, we compare the performance of the attention mechanism
combined with FFT and the ordinary attention mechanism in the local
feature aggregation of the LPAM module. As can be seen from this
Table, after the introduction of FFT, the performance of the model on
all datasets has been further improved, especially on the VC-ClothesW&R dataset, where Rank-1 and mAP are increased by 1.0% and

0.7% respectively. This fully demonstrates that FFT can help the model
more effectively process and identify structural details in the frequency
domain by converting data to the frequency domain. These results fully
validate the effectiveness and importance of each component in our
framework.
To intuitively demonstrate the effectiveness of each module, we visualize the experimental results on the Celeb-reID dataset, as illustrated
in Fig. 6. It can be seen that the matching results of using modules
MLDC and LPAM separately are better than the baseline model. When
these two modules are used in combination, its matching rate in Ranks
1–10 is significantly improved. Especially after the introduction of FFT
self-attention, the accuracy of the model is further improved, which is
completely consistent with our ablation experimental results.
Visualization of feature distribution. To substantiate the efficacy
of the introduced components, we employ t-SNE [40] for visualizing
the distribution of features extracted by the model across different
components, as illustrated in Fig. 7. With this figure, the circles mean
randomly selected image features from the training set of Celeb-reID
dataset, with varying colors denoting distinct identities.
Specifically, Fig. 7(a) shows the extraction distribution of features
by the baseline model ViT. It can be seen that the feature points are
relatively scattered and the degree of identity aggregation is low, which
reflects the significant challenge to the effective recognition of ViT
8


codex
后三篇的方法段也抽到了。现在我补几个被截断的关键页段，并查每篇在线条目或代码页，主要确认发表时间、公开定位和它们声称的新意边界。
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
/bin/zsh -lc "pdfinfo 'A training-free framework for text-to-image person re-identification via query-prototype matching.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           A training-free framework for text-to-image person re-identification via query-prototype matching
Subject:         Pattern Recognition, 179 (2026) 113705. doi:10.1016/j.patcog.2026.113705
Keywords:        Person re-identification,Text-to-image person retrieval,Training-free
Author:          Hao Yang
Creator:         Elsevier
Producer:        Acrobat Distiller 8.1.0 (Windows)
CreationDate:    Mon Jun  8 17:29:45 2026 CST
ModDate:         Mon Jun  8 17:35:51 2026 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          yes
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           13
Encrypted:       no
Page size:       595.276 x 793.701 pts
Page rot:        0
File size:       3807986 bytes
Optimized:       no
PDF version:     1.7

exec
/bin/zsh -lc "pdftotext -f 3 -l 5 'Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 163 (2025) 111443

G. Zhang et al.

2.2. Person Re-ID under intensive cloth variations

3.1. Multi-layer dynamical concentration module

As public safety concerns become increasingly prominent, especially in the fields of monitoring and safety, there is a pressing need
for effective identification of potential threats. Therefore, accurate
identification of individuals who change their attire becomes crucial
to promptly detect and intervene in potential security risks. These
concerns have spurred many scholars to conduct in-depth research
on CC Re-ID. In recent years, some related cloth-changing datasets
have been released, such as VC-Clothes [17], Celeb-reID [18], CelebreID-light [19] and NKUP [20]. In these datasets, the same individual
switches among multiple outfits, and wears various accessories, such
as sunglasses, scarves, backpacks, etc. Frequent changes of clothing
greatly reduce the reliability of traditional appearance-based matching
methods.
To cope with the challenges brought by changing clothes, some
works learn clothing-independent features with the help of identityrelated auxiliary biological cues. For example, Hong et al. [4] proposed
a shape–appearance mutual learning framework (FSAM), which is a
dual-stream structure that acquires the detailed discriminative body
shape information in shape stream and enriches the appearance stream
with non-fabric-related details. Zhang et al. [21] proposed a novel
Multi-Biometric Unified Network (MBUNet), which applies adaptive
graph convolution to obtain relevant information between key points
of the human body, and combines multiple biological features such as
the person’s head, neck, shoulders to mitigate the influence of clothing
alterations. However, these methods have high requirements on image
quality, and when the image is affected by occlusion, low illumination
and so forth, this will limit the extraction of identity-related features,
thus limiting the performance of the model. To further reduce the
dependence on collecting a large amount of clothing change data, PosNeg [22] introduced an innovative data augmentation strategy, using
positive augmentation and negative augmentation techniques to enrich
the ID feature space and generate out-of-distribution synthetic samples,
thereby enhancing the model’s robustness to clothing changes.
Another very common methods seek to segregate clothing-related
features from irrelevant features, enabling the model to concentrate
on acquiring clothing-independent identity information. Xu et al. [8]
proposed AFD-Net, which uses GAN and semantic perception models
to distinguish the appearance and structural features of pedestrian
images to achieve the separation of identity and clothing features,
thereby enabling the model to learn identity Discriminating features.
Similarly, SAVS [23] first locates the human body and clothing area
according to the human body semantic segmentation, and introduces
the human body semantic attention module to emphasize the human
body information. Furthermore, it shields the clothing area to make the
model focus on the extraction of visual semantic information unrelated
to clothing. However, these kinds of methods generally face a challenge: in the process of separating clothing features from non-clothing
features, distorted details are inevitably generated and the accurate
expression of cloth-irrelevant features may be weakened, resulting in
unstable training processes and poor model performance. Considering
the limitation of the above two types of methods, we do not use any
biological auxiliary branches or feature decoupling to help distinguish
individuals, but make full use of the differences in features of each layer
of Transformer to learn identity-related features. Specific introduction
will be shown in the next section.

In image processing, the Transformer architecture builds a visual
feature hierarchy layer-by-layer, from edge and texture detection at
the primary layer to scene comprehension at the high-level layer.
However, previous Re-ID models often only focus on the information
of the terminal layer, while ignoring the fine details of the primary
and intermediate layers. To make up for this deficiency, we propose
the Multi-Layer Dynamical Concentration Module (MLDC) (Fig. 3).
This model dynamically synthesizes features across layers and also
includes the key visual information from each layer in the final feature
representation.
Calculation of weights. In order to effectively perform multilayer feature fusion, in our method, we assign a weight coefficient
𝑤𝑖 (i = 1...12) to each layer, the purpose of which is to evaluate the
feature importance of each layer in real time and reduce the similarity
redundancy of related layers, and the specific calculation process of 𝑤𝑖
is as follows:
)
(
𝐿
|⟨𝐹𝑖 , 𝐹𝑗 ⟩|
𝑒𝑥𝑝 𝑓𝑖 − 𝛼 𝛴𝑗=1,𝑗≠𝑖
(1)
𝑤𝑖 =
),
(
𝐿 𝑒𝑥𝑝 𝑓 − 𝛼 𝛴 𝐿
|⟨𝐹𝑘 , 𝐹𝑚 ⟩|
𝛴𝑘=1
𝑘
𝑚=1,𝑚≠𝑘
where 𝐹𝑖 ∈ R𝑁×𝐷 represents the output of the 𝑖th layer, N is the
number of image blocks and D is the feature dimension of each token,
⟨⋅, ⋅⟩ is the inner product, which measures the feature correlation of
different layers, 𝛼 is a regularization coefficient used to scale the impact
of orthogonality constraints and reduce feature overlap between layers,
𝐿 is the total number of layers. And 𝑓𝑖 is a one-dimensional scalar that
represents the importance of the output feature 𝐹𝑖 of each layer, the
specific calculation formula is as follows:
1∑
𝑚𝑒𝑎𝑛(𝐴𝑖𝑡 ),
ℎ 𝑡=1
ℎ

𝑓𝑖 =

(2)

where h represents the number of attention heads in each layer, 𝑚𝑒𝑎𝑛(⋅)
represents the mean of all elements, 𝐴𝑖𝑡 ∈ R𝑁×𝑁 represents the
attention score matrix of the 𝑡th head in the 𝑖th layer (t, i = 1,2, . . . ,12),
and the formula is as follows:
(
)
𝑄𝑖𝑡 𝐾𝑖𝑡𝑇
𝐴𝑖𝑡 = softmax
,
(3)
√
𝑑𝑡
where 𝑄𝑖𝑡 ∈ R𝑁×𝑑𝑡 and 𝐾𝑖𝑡 ∈ R𝑁×𝑑𝑡 are the query and key matrices of
is the dimension size of each
the 𝑡th head in the 𝑖th layer, and 𝑑𝑡 = 𝐷
ℎ
head, which is used to scale the dot product result to prevent too large
values from affecting the gradient of the 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥(⋅) function.
Enhanced Feature Fusion With Regularization. To mitigate the
risk of model over-fitting that may occur due to the undue influence
of specific layers, we incorporate an 𝐿2 regularization term into our
feature fusion formula:
𝐿
𝐹𝑓 𝑢𝑠𝑖𝑜𝑛 = 𝛴𝑖=1
𝑤𝑖 𝐹𝑖 − 𝜆 ⋅ ‖𝑊 ‖2𝐹 ,

(4)

where 𝜆 is a non-negative regularization parameter employed to mitigate over-fitting by constraining the magnitude of the weights within
the model. ‖𝑊 ‖2𝐹 is the Frobenius norm of the weight matrix 𝑊 and
is the sum of the squares of all layer weights.
3.2. Local pyramid aggregation module

3. The proposed method

Although the adaptive weighted average method is very effective in
integrating multi-layer features to achieve comprehensive representation, its global fusion method may ignore local information.
In order to capture image details at various scales, we design a
local pyramid aggregation module (LPAM), as shown in Fig. 3. The
module adopts a pyramid structure and gradually fuses the output
of four different layers 𝐹𝑙1 , 𝐹𝑙2 , 𝐹𝑙3 , 𝐹𝑙4 to achieve the extraction of
different fine-grained information, the specific layer selection is provided in the ablation study section. For example, in the local feature

This section elaborates on our proposed approach. We first introduce a Multi-Layer Dynamical Concentration Module to evaluate the
significance of features at each layer in real time while minimizing computational redundancy among highly correlated layers in Section 3.1. In
Section 3.2, we further adopt a Local Pyramid Aggregation Module to
enhance multi-scale features and integrate Fast Fourier Transform (FFT)
to optimize the self-attention mechanism. Finally, the optimization of
the overall framework is described in Section 3.3.
3

Pattern Recognition 163 (2025) 111443

G. Zhang et al.

Fig. 3. The architecture of our proposed 𝐴3 𝑃 𝐹 𝑁, which is built on ViT and contains two parallel modules–Multi-Layer Dynamical Concentration Module (MLDC) and Local
Pyramid Aggregation Module (LPAM). MLDC aims to obtain aggregated global features by dynamically assigning weights to each layer. LPAM is designed to fuse multi-level
features through a pyramid structure to obtain multi-scale information.

transformations on 𝑋 to convert it into Q, Kand V. Subsequently, Q, K
and V are split into multiple heads. To improve the efficiency of Fast
Fourier Transform (FFT), we appropriately fill the Q and K matrices to
the integer power of 2, and then apply FFT on the filled 𝑄𝑝𝑎𝑑 𝑑 𝑒𝑑 and
𝐾𝑝𝑎𝑑 𝑑 𝑒𝑑 and estimate their correlation in the frequency domain. The
output is formulated as below:
𝐴𝑡𝑡𝑛 = 𝑆 𝑜𝑓 𝑡𝑚𝑎𝑥(𝐹 −1 (𝐹 (𝑄𝑝𝑎𝑑 𝑑 𝑒𝑑 ) ⊙ 𝐹 (𝐾𝑝𝑎𝑑 𝑑 𝑒𝑑 ))[∶, ∶, ∶, ∶ 𝑄.𝑠𝑖𝑧𝑒(−1)]), (6)
where 𝐹 (⋅) and 𝐹 −1 (⋅) represent FFT and inverse FFT (IFFT) respectively, ⊙ is a dot product operation. Softmax function aims to normalize
the result to produce attention weights 𝐴𝑡𝑡𝑛. Finally, we obtain the
attention-weighted output through weighted calculation and residual
connection.

Fig. 4. Illustration of FFT Self-Attention, which aims to identify detailed information
of pedestrians from a frequency domain perspective.

3.3. Model optimization
aggregation (LFA) of LPAM, we implement a 1 × 1 convolution layer
and BatchNorm2D processing on features 𝐹𝑙1 and 𝐹𝑙2 , with the help
of ReLU function for size adjustment and nonlinear enhancement. In
addition, we introduce a self-attention mechanism to obtain enhanced
pedestrian information from the frequency domain perspective (see
Fig. 4). Finally, the convolved 𝐹𝑙1 and 𝐹𝑙2 are connected, and then input
into the convolution block to achieve feature fusion. The formula for
feature fusion is as follows:
(
( ( ) ( )))
𝐹𝑚𝑖𝑑1 = 𝜚 concat 𝜚 𝐹𝓁1 , 𝜚 𝐹𝓁2
,
(5)

We optimize the model through ID loss and triplet loss. The ID
loss adopts the traditional cross-entropy loss function, excluding label
smoothing, and its specific definition is as follows:
𝐿𝐼 𝐷 = −

𝐶
∑

𝑦𝑗 𝑙𝑜𝑔(𝑝𝑗 ),

(7)

𝑗

where 𝐶 is the number of categories, 𝑦𝑗 is the one-hot encoding of the
true label and 𝑝𝑗 is the probability that the model predicts that the
sample belongs to the 𝑗th category.
In triplet loss, we adopt a semi-hard sample mining strategy to
solve the over-fitting problem caused by noisy samples in hard sample
mining. This strategy selects samples that are predicted incorrectly
but with low confidence during the training phase to improve the
model’s coverage of a broad sample set and mitigate the impact of noisy
samples. The indicator function of semi-hard samples is as follows:
{
1 if 𝑑(𝑎𝑝) + 𝑚 < 𝑑(𝑎𝑛) < 𝑑(𝑎𝑝) + 𝑀
𝐼semi-hard (𝑥𝑎𝑙 , 𝑥𝑝𝑙 , 𝑥𝑛𝑙 ) =
(8)
0 otherwise ,

where 𝜚(⋅) represents the convolution block and 𝑐 𝑜𝑛𝑐 𝑎𝑡(⋅, ⋅) refers to the
splicing operation. The following fusion steps are similar to this. For
the detailed process, please refer to the framework diagram of the local
pyramid aggregation module (LPAM) shown in Fig. 3.
FFT Self-Attention. Fast Fourier Transform (FFT) is an effective
algorithm for computing the Discrete Fourier Transform, as described
in [24], which can significantly reduce the computational complexity
from 𝑂(𝑁 2 ) to 𝑂(𝑁 log 𝑁), making it crucial in signal frequency domain
analysis.
In our method, the self-attention module first accepts the input
𝑋 ∈ R𝑁×𝐷 , where N is the number of image blocks (tokens) and D is
the feature dimension of each token. Then, we perform different linear

where 𝑑(𝑎𝑝) represents the distance between the anchor sample 𝑥𝑎𝑙
and the positive sample 𝑥𝑝𝑙 , 𝑑(𝑎𝑛) represents the distance between the
anchor sample and the negative sample 𝑥𝑛𝑙 . 𝑚 means the minimum
distance difference between anchor samples and positive samples to
4

Pattern Recognition 163 (2025) 111443

G. Zhang et al.

Table 1
Clothing change statistics of the long-term pedestrian datasets used in our experiments. Please note: ‘SC’ and ‘CC’ represent the two modes of
same clothes and clothing change respectively.
Dataset

VC-Clothes
NKUP
Celeb-reID
Celeb-reID-light
PRCC
LTCC
VC-Clothes-W&R

Source

Synthetic
Real
Real
Real
Real
Real
Synthetic

Train(ID/Image)

256/9449
40/5336
1052/34,186
100/887
150/17,896
77/9576
256/9449

Test(ID/Image)
Query

Gallery

256/1020
39/332
420/2972
100/934
71/3543
75/493
256/1020

256/8591
67/4070
420/11,006
590/10,842
71/3384
75/7050
256/8591

Cameras

Data style

Weather backgrounds

4
15
Many
Many
3
12
4

SC/CC
CC
CC
CC
SC/CC
SC/CC
SC/CC

None
None
None
None
None
None
Wind & rain

prevent the model from only focusing on subtle differences between
similar samples, 𝑀 is the maximum distance between anchor samples
and negative samples to avoid selecting overly simple negative samples
and ensure that the model learns discriminative features. The triplet
loss for semi-hard sample mining is defined as follows:
𝐿𝑡𝑟𝑖 = −

𝑁
∑

(
𝐼semi-hard (𝑥𝑎𝑙 , 𝑥𝑝𝑙 , 𝑥𝑛𝑙 ) ‖𝑓 (𝑥𝑎𝑙 ) − 𝑓 (𝑥𝑝𝑙 )‖22

𝑙

)
−‖𝑓 (𝑥𝑎𝑙 ) − 𝑓 (𝑥𝑛𝑙 )‖22 + 𝑚 + ,

(9)

where 𝑓 (⋅) denotes the feature extraction operator that maps the input
image into an embedding space. ∥ ⋅ ∥2 represents the 𝐿2 -norm, the
Euclidean distance between two feature vectors. (⋅)+ is the hinge loss
function, which means that the loss is calculated only when the value in
the brackets is a positive number, otherwise the loss is 0. Consequently,
the comprehensive loss function of the model is defined as follows:
𝐿=

𝑇
∑

Fig. 5. Some images from VC-Clothes-W&R.

4.2. VC-Clothes-W & R
𝑡𝑞 (𝐿𝐼 𝐷𝑞 + 𝐿𝑡𝑟𝑖𝑞 ) (𝑞 = 1, … , 4) ,

(10)
As a synthetic dataset, VC-Clothes can provide a more controlled
experimental environment, allowing us to keep other variables (such as
lighting conditions and occlusions) relatively consistent across different
scenes. In contrast, real image datasets often have the complexity of
natural environments and introduce many uncontrollable variables,
such as fluctuating lighting conditions, varying degrees of occlusion
caused by unpredictable obstacles and other factors. These factors may
mask the specific impact of weather on the recognition task, making
it challenging to separate the variables required for testing. Therefore,
to enhance the diversity of the image backgrounds and increase the
model’s robustness, we add wind and rain scenes to the VC-Clothes
dataset [17]. The generated VC-Clothes-W&R dataset consists of images
captured from four different cameras, segmented into training and
testing sets. The training set comprises 256 unique identities with a
total of 9449 images. Similarly, the test set is structured into query
and gallery segments, maintaining the same 256 identities with 1020
images in the query and 8591 in the gallery set. Some sample images
are shown in Fig. 5 and the construction process is as follows:
Considering the complexity of precipitation levels and changes in
image brightness, we employ a refined atmospheric scattering model
[26] to generate rainy and windy scenes for images. Specifically, we use
𝐽 (𝑥, 𝑦) to represent a pixel in the original image and after adding the
wind and rain scene, the corresponding pixel 𝐼(𝑥, 𝑦) can be calculated
by the following formula:

𝑞

where 𝑇 = 4 represents the total number of output features in our
model and 𝑡𝑞 represents the weight of each output feature. While
assigning fixed weights to each part of the loss is simple, it may not
produce the best model performance. Therefore, we adapt the loss
function for each output feature to adaptive weights. Initially, the
loss weight 𝑡𝑞 for each output feature are expressed as the same size
and are subsequently dynamically adjusted during training through
back-propagation to obtain the optimal weights.

4. Experiments
4.1. Datasets and evaluation metrics
Datasets Details. To assess the performance of our proposed
𝐴3 𝑃 𝐹 𝑁, we carry out experimental evaluations on the publicly available cloth-changing pedestrian datasets, including VC-Clothes [17],
NKUP [20], Celeb-reID [18], Celeb-reID-light [19], LTCC [25] and
PRCC [7]. Table 1 provides an overview of these datasets. Meanwhile,
we build an enhanced dataset VC-Clothes-W&R for wind and rain
scenes based on VC-Clothes to make up for the lack of environmental
elements in existing datasets.

𝐼(𝑥, 𝑦) = 𝐿[𝐽 (𝑥, 𝑦)𝑡(𝑥, 𝑦) + 𝑅(𝑥, 𝑦)(1 − 𝑡(𝑥, 𝑦))𝑀(𝑥, 𝑦)],

Evaluation Metrics. We evaluate the performance of 𝐴3 𝑃 𝐹 𝑁 using
Rank-1 accuracy and mean precision (mAP) in three test scenarios: (1)
general scenarios, covering clothing changes and consistent samples;
(2) clothing change scenarios, only including clothing change samples; (3) clothing consistent scenes, only including consistent clothing
samples. In the following tables, ‘‘sil’’, ‘‘ga’’, ‘‘dg’’, ‘‘pose’’ and ‘‘bs’’ represent pedestrian semantic segmentation, gait, data generation, Human
Posture and body shape information.

(11)

where 𝐿[⋅] reflects the brightness coefficient, 𝑅(𝑥, 𝑦) represents the
radiant brightness of raindrops on the spatial coordinates (𝑥, 𝑦). Furthermore, 𝑀(𝑥, 𝑦) is the blur kernel used to simulate the width and diffusion
properties of visual distortion caused by raindrops. 𝑡(𝑥, 𝑦) represents the
medium transmission ratio at each point (𝑥, 𝑦), which quantifies the
light intensity due to the presence of raindrops at that specific location.
The formula is as follows:
𝑡(𝑥, 𝑦) = 𝑒−𝛽 𝑑(𝑥,𝑦)𝑠(𝑥,𝑦)𝑐 𝑜𝑠(𝜃) ,
5

(12)


exec
/bin/zsh -lc "pdftotext -f 3 -l 6 'A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Fig. 2. The architecture of A2HA. The parameters in the segmentation model and LLM are frozen during training. All attribute sample pairs include both
𝑣
𝑡
inter-modal and intra-modal cases. 𝐹 𝑣 and 𝐹 𝑡 represent image and text features, while 𝐹𝑎𝑡𝑡
and 𝐹𝑎𝑡𝑡
are representation of attribute features for both image and
text inputs.

pre-trained semantic segmentation model Grounded SAM [39], both
with frozen parameters, to obtain attribute descriptions and segmenta𝑣
tion masks. In the visual branch, attribute-specific visual features 𝐹𝑎𝑡𝑡𝑞
𝑣
are obtained by fusing the global visual feature 𝐹𝑔 with the corresponding attribute mask under a ratio 𝑟. In parallel, textual attribute
descriptions are encoded to generate attribute-level textual features
𝑡 . Based on these attribute features, we employ a dual-dimensional
𝐹𝑎𝑡𝑡𝑞
and dual-directional attribute pairing strategy to categorize attribute
sample pairs across intra-modal and inter-modal spaces. The resulting
attribute pairs are then optimized through the Attribute-Level Alignment(ALA) module, which performs contrastive learning over diverse
attribute relations. Through the joint optimization of GLA and ALA,
A2HA achieves hierarchical alignment across instance, identity, and
attribute levels.

2.2. Attribute-aware methods
Numerous works [34,35] perform cross-modal alignment at the
global level. For example, IRRA [36] matches global image and text
representations based on feature similarity. To alleviate this limitation, several attribute-aware methods have been proposed [11,12,14].
ViTAA [11] pioneers attribute-level segmentation across visual and
textual modalities, but relies on NLKT-based noun phrase extraction,
which may fragment descriptive semantics, and adopts static attributevisual mappings that assume fixed attribute importance. APTM [13]
extracts attribute nouns to construct prompt-based representations;
however, decoupling attributes from their original textual context disrupts syntactic coherence and leads to misalignment when matching
attribute descriptions with global visual features. VGSG [15] further
explores attribute-aware alignment, yet still employs static attribute
mappings and performs alignment at a single semantic level, limiting
its ability to capture hierarchical and fine-grained attribute correspondence. BAMG [16] models attribute relations via graph convolution, but depends on predefined human structural priors, which may
constrain flexibility under diverse visual conditions.

3.2. Feature extraction
A2HA employs CLIP-ViT/B-16 [37] as image encoder and
BERT [25] as text encoder. In ViT-B/16 [3], the letter ‘B’ denotes the
base network scale, while ‘16’ indicates the use of a 16 × 16 pixel patch
size.
Image Encoder. The input image 𝐼 ∈ R𝐻×𝑊 ×𝐶 is divided into 𝑁
patches and passed through a linear projection layer with a learnable
position embedding 𝑃𝐸 . After concatenation with a [CLS] token, 𝐼 is
represented as 𝑉 ∈ R(𝑁+1)×𝑑 , where 𝑑 denotes the embedding dimension. The [CLS] token encodes the global information of all 𝑁 patches.
{ 𝑣
}
𝑣 , where
The extracted features are tokenized into 𝐹 𝑣 = 𝑓𝑐𝑙𝑠
, 𝑓1𝑣 , … , 𝑓𝑁
𝑣
𝑣
𝑓𝑖 represents the feature of the 𝑖𝑡ℎ patch and 𝑓𝑐𝑙𝑠 serves as the global
representation 𝐹𝑔𝑣 of the image.
Text Encoder. Following a similar approach as the image encoder,
the input text 𝑇 is tokenized by a token embedding layer. The textual
token is obtained through lower-cased Byte Pair Encoding (BPE). Starting with a [SOS] token and ending{with an [EOS] token,
we obtain
}
𝑡 , 𝑓 𝑡, … , 𝑓 𝑡
a sequence represented as 𝐹 𝑡 = 𝑓SOS
.
Additionally,
1
EOS
since we have adopted BERT as the backbone, we incorporate masked
language modeling to improve prediction accuracy.

3. Methodology
3.1. Overview
As illustrated in Fig. 2, A2HA follows an attribute-aware hierarchical alignment paradigm, consisting of a Global-Level Alignment
(GLA) branch and an Attribute-Level Alignment (ALA) branch. The two
branches collaboratively enforce coarse-to-fine cross-modal semantic
alignment. Given an input image–text pair, we adopt CLIP-ViT/B16 [37] as the backbone to extract visual and textual features, denoted
as 𝐹 𝑣 and 𝐹 𝑡 . The corresponding global representations 𝐹𝑔𝑣 and 𝐹𝑔𝑡
are first utilized to perform Global-Level Alignment(GLA), ensuring
instance-level and identity-level consistency across modalities. To enable fine-grained attribute-level alignment, the input texts and images
are further processed by a large language model ChatGPT [38] and the
3

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

3.3. Hierarchical alignment framework

3.3.3. Attribute-level alignment
However, instance-level alignment along with identity-level alignment are insufficient to resolve ambiguities caused by visually similar
pedestrians with subtle attribute differences. To address this limitation,
we introduce attribute-level alignment, which enforces fine-grained
correspondence across modalities. To effectively optimize attributelevel correspondence, it is crucial to explicitly identify and treat false
positive and false negative attribute pairs during training. To identify
false positive and false negative attribute pairs, we further design a
similarity-based attribute pairing strategy to categorize sample pairs
into positive, negative, and problematic cases.
Attribute Pairing Strategy: Some attribute-aware works explore
image–text attribute pairing for cross-modal alignment. ViTAA [11]
relies on identity-level supervision, while LAIP [40] introduces noun
phrases as attribute units. However, both approaches tend to fragment
textual descriptions, disrupting semantic integrity by discarding contextual modifiers. Although contrastive learning is employed, attribute
distance optimization is not explicitly disentangled across intra-modal
and inter-modal spaces. Consequently, alignment is mainly conducted
in a unidirectional manner from textual phrases to visual regions, without reciprocal validation from visual attributes to textual descriptions.
This limitation motivates the need for a systematic and bidirectional
attribute pairing strategy.
By using a similarity-based analysis, we categorize image–text attribute feature pairs into 12 cases, which are further grouped into 6
types of sample pairs. For the similarity metric, we use the cosine
similarity 𝑆(⋅), defined as follows:

The core of A2HA is a hierarchical alignment strategy composed
of Global-Level Alignment (GLA) and Attribute-Level Alignment (ALA).
Specifically, GLA integrates both instance-level and identity-level alignment to enforce global semantic and identity consistency, while ALA
focuses on fine-grained attribute correspondence. Through this hierarchical design, global identity consistency and local attribute alignment
are jointly optimized in a complementary manner rather than treated
independently.

3.3.1. Instance-level alignment
Instance-level alignment serves as the global semantic foundation
of our framework, ensuring that image–text pairs of the same instance
remain close in the shared embedding space. For instance-level alignment, we employ a bidirectional cross-entropy loss to enforce global
semantic consistency. As shown in Eq. (1), 𝑖𝑛𝑠 is composed of crossentropy losses computed bidirectionally: from image to text and from
text to image.
𝑖𝑛𝑠 =

)
( 𝑇
)]
1[ (
CE 𝑆𝐼2𝑇 , 𝑦 + CE 𝑆𝐼2𝑇
,𝑦 ,
2

(1)

𝑆𝐼2𝑇 =

(
)
(
)
1
1
𝑇
⋅ 𝑆 𝑓𝐼𝑔 , 𝑓𝑇𝑔 , 𝑆𝐼2𝑇
= ⋅ 𝑆 𝑓𝑇𝑔 , 𝑓𝐼𝑔 ,
𝜏
𝜏

(2)

(
)
⎛
⎞
𝐵−1
⎜ exp 𝑆𝑖,𝑦𝑖
⎟
1 ∑
CE(𝑆, 𝑦) = −
log ⎜ ∑
(
)
⎟,
𝐵−1
𝐵 𝑖=0
⎜ 𝑗=0 exp 𝑆𝑖,𝑗 ⎟
⎝
⎠

(3)

(
)
𝑗
𝑖
𝑆 𝑉𝑎𝑡𝑡𝑟
, 𝑇𝑎𝑡𝑡𝑟
=

𝑦 is a positive sample index vector of length 𝐵 (batch size), where
𝑦 ∈ [0, 𝐵 − 1] denotes the index of the text that forms a positive pair
with the 𝑖𝑡ℎ image. All texts with indices 𝑗 ≠ 𝑦𝑖 are treated as negative
samples for the 𝑖𝑡ℎ image. 𝑆𝐼2𝑇 represents the image-to-text similarity
𝑇
matrix, while 𝑆𝐼2𝑇
is its transpose, capturing text-to-image similarities
to enforce bidirectional alignment. CE(⋅) refers to the cross-entropy
loss. 𝜏 represents the temperature parameter, set to 0.07. 𝑓𝐼𝑔 and 𝑓𝑇𝑔
represent the global features of the image and text, respectively. For
the cross-entropy calculation, 𝑆𝑖,𝑗 denotes the similarity between the
𝑖𝑡ℎ image and the 𝑗𝑡ℎ text. 𝑆𝑖,𝑦𝑖 highlights the similarity score of the
positive pair for the 𝑖𝑡ℎ sample (i.e., when 𝑗 = 𝑦𝑖 ).

𝑁
(
)
1 ∑
log 𝑝𝑖,𝑦𝑖 ,
𝑁 𝑖=1

(5)

𝑖 denotes the visual feature representation of the 𝑖 sample’s
where 𝑉𝑎𝑡𝑡𝑟
𝑡ℎ
𝑗
attributes, and 𝑇𝑎𝑡𝑡𝑟
denotes the corresponding textual attribute feature
representation of the 𝑗𝑡ℎ sample. To enable fine-grained classification,
we introduce two thresholds 𝜃𝑝𝑜𝑠 and 𝜃𝑛𝑒𝑔 . The image–text attribute
sample pairs are categorized into 12 distinct cases based on these two
thresholds. To be specific, for image samples 𝑥, 𝑦 and text samples 𝑚,
𝑛, the attribute-specific features are extracted by performing feature
extraction on the segmented attributes. For example, the feature of the
𝑞𝑡ℎ attribute for the 𝑖𝑡ℎ image is denoted as 𝑥𝑖𝑎𝑡𝑡𝑞 . Pairwise similarity
measurement 𝑆(⋅) is then performed on these attribute features. Using
the thresholds 𝜃𝑝𝑜𝑠 and 𝜃𝑛𝑒𝑔 , we classify positive pairs, negative pairs,
false positive pairs, and false negative pairs, covering both intra-modal
and inter-modal cases. The detailed classification rules are provided
in Table 1. This achieves precise classification along two dimensions:
(1) semantic consistency, and (2) modal consistency among local attributes, categorizing false positives and false negatives as ‘problematic
samples’ that require strengthened constraints. Herein, we set 𝜃𝑝𝑜𝑠 =
0.5 and 𝜃𝑛𝑒𝑔 = 0.3 as the optimal values based on experimental
comparisons.
The intra-modal attribute contrastive loss 𝑖𝑛𝑡𝑟𝑎 , inter-modal attribute contrastive loss 𝑖𝑛𝑡𝑒𝑟 , and the attribute triplet loss 𝑡𝑟𝑖 constitute
the overall attribute-level constraints. For a given attribute 𝑞, we obtain
batches of false positive and false negative attribute pairs by using the
classification strategy in Table 1. To enhance the model’s ability to
identify problematic samples, we perform hard sample mining: from
false negative pairs, we select the batch pair with the smallest deviation
from the positive threshold 𝜃𝑝𝑜𝑠 as the hardest positives; similarly, from
false positive pairs, we select the pair with the largest deviation from
the negative threshold 𝜃𝑛𝑒𝑔 as the hardest negatives.
Intra-modal Attribute Loss: Intra-modal attribute alignment enforces the clustering of samples sharing the same attribute within each
modality. This additional explicit constraint encourages the model to
learn diverse stylistic variations of the same attribute, thereby improving its generalization ability to intra-modal appearance and description

3.3.2. Identity-level alignment
In addition to instance-level alignment, we introduce an explicit
identity-level alignment objective to further enhance identity discriminability in the shared embedding space. While instance-level alignment focuses on cross-modal instance correspondence, identity-level
alignment directly enforces class-level supervision by treating pedestrian identities as classification targets. Specifically, we adopt a standard identity classification loss to supervise the learning of identitydiscriminative features. The identity-level loss 𝑖𝑑 is defined as:
𝑖𝑑 = −

𝑖 ⋅ 𝑇𝑗
𝑉𝑎𝑡𝑡𝑟
𝑎𝑡𝑡𝑟
,
𝑗 ‖
‖𝑉 𝑖 ‖ × ‖
‖
‖ 𝑎𝑡𝑡𝑟 ‖ ‖𝑇𝑎𝑡𝑡𝑟 ‖
‖

(4)

where 𝑁 denotes the number of samples in a mini-batch, and 𝑝𝑖,𝑦𝑖
represents the predicted probability that the 𝑖𝑡ℎ sample belongs to its
ground-truth identity class 𝑦𝑖 .
Importantly, the identity label 𝑦𝑖 is consistent with the positive
sample indexing used in the instance-level alignment loss 𝑖𝑛𝑠 (Eq. (1)).
This design ensures coherent supervision across different alignment
levels, where instance-level learning encourages cross-modal instance
matching, while identity-level classification explicitly enlarges interclass margins and stabilizes training.
4

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Table 1
Attribute sample pair classification based on similarity measurement. 𝑥𝑎𝑡𝑡𝑞 and
𝑦𝑎𝑡𝑡𝑞 denote image samples of attribute 𝑞, while 𝑚𝑎𝑡𝑡𝑞 and 𝑛𝑎𝑡𝑡𝑞 represent text
samples of the same attribute 𝑞. The variables 𝑖 and 𝑗 represent person IDs,
and 𝑆(⋅) denotes the cosine similarity function.
pid

Similarity threshold
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑦𝑗𝑎𝑡𝑡𝑞 ≥ 𝜃𝑝𝑜𝑠
(
)
𝑗
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑦𝑎𝑡𝑡𝑞 < 𝜃𝑝𝑜𝑠
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑚𝑗𝑎𝑡𝑡𝑞 ≥ 𝜃𝑝𝑜𝑠
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑚𝑗𝑎𝑡𝑡𝑞 < 𝜃𝑝𝑜𝑠
(
)
𝑆 𝑚𝑖𝑎𝑡𝑡𝑞 , 𝑛𝑗𝑎𝑡𝑡𝑞 ≥ 𝜃𝑝𝑜𝑠
(
)
𝑆 𝑚𝑖𝑎𝑡𝑡𝑞 , 𝑛𝑗𝑎𝑡𝑡𝑞 < 𝜃𝑝𝑜𝑠
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑦𝑗𝑎𝑡𝑡𝑞 ≥ 𝜃𝑛𝑒𝑔
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑦𝑗𝑎𝑡𝑡𝑞 < 𝜃𝑛𝑒𝑔
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑚𝑗𝑎𝑡𝑡𝑞 ≥ 𝜃𝑛𝑒𝑔
(
)
𝑆 𝑥𝑖𝑎𝑡𝑡𝑞 , 𝑚𝑗𝑎𝑡𝑡𝑞 < 𝜃𝑛𝑒𝑔
(
)
𝑆 𝑚𝑖𝑎𝑡𝑡𝑞 , 𝑛𝑗𝑎𝑡𝑡𝑞 ≥ 𝜃𝑛𝑒𝑔
(
)
𝑆 𝑚𝑖𝑎𝑡𝑡𝑞 , 𝑛𝑗𝑎𝑡𝑡𝑞 < 𝜃𝑛𝑒𝑔

𝑖=𝑗

𝑖≠𝑗

(belonging to a different identity but having high similarity score) as a
false positive and the text sample 𝑝 (sharing the same identity as 𝑖 but
having low similarity score) as a false negative, we apply the triplet loss
to strengthen the association between image sample 𝑖 and text sample
𝑝 while separating sample 𝑖 from 𝑛. The loss function is defined as
follows:

Category
Positive

𝐵

Intra False Negative

𝑞𝑡𝑟𝑖 =

Positive
Inter False Negative
Positive

𝐵𝑞 [ (
)
(
)
1 ∑
𝛼 𝜃pos − max S 𝑓𝐼𝑞 [𝑖], 𝑓𝐼𝑞 [𝑗]
𝑗∈𝐹 𝑁𝑖
𝐵𝑞 𝑖=1
(
)]
(
)
+ 𝛼 max S 𝑓𝐼𝑞 [𝑖], 𝑓𝐼𝑞 [𝑗] − 𝜃neg ,

Intra False Positive
Negative

3.3.4. Hierarchical loss
Based on the aforementioned classification, we propose a multiscale cross-modal composite loss that integrates contrastive learning, hard example mining, and attribute-adaptive weight adjustment.
This loss jointly optimizes fine-grained attribute alignment and global
matching, enabling more reliable discrimination among different attribute pair types. Instead of assuming uniform attribute
importance [11,13], the proposed weighting scheme adaptively adjusts
the contribution of each attribute during optimization, allowing the
model to focus more on ambiguous or misaligned attributes when necessary. Accordingly, false positive and false negative attribute pairs are
treated as problematic samples and penalized more heavily, ensuring
that features of the same pedestrian ID and attribute remain close
in both inter-modal and intra-modal spaces, while those of different
IDs and attributes are effectively separated. By integrating global-level
and attribute-level objectives, A2HA achieves hierarchical alignment in
which global identity consistency and local attribute correspondence
are optimized in a complementary manner.
Our hierarchical loss comprises instance-level, identity-level and
attribute-level losses. Specifically, 𝑖𝑛𝑠 in Eq. (1) establishes global
semantic constraints for visual and textual information; 𝑖𝑑 in Eq. (4)
achieves identity alignment, while 𝑎𝑡𝑡𝑟 enforces fine-grained attributelevel alignment for both inter-modal and intra-modal features. The
weights of 𝑖𝑛𝑠 , 𝑡𝑟𝑖 , 𝑖𝑛𝑡𝑒𝑟 , and 𝑖𝑛𝑡𝑟𝑎 are denoted as 𝜆𝑖𝑛𝑠 , 𝜆𝑡𝑟𝑖 , 𝑤𝑖𝑛𝑡𝑒𝑟 ,
and 𝑤𝑖𝑛𝑡𝑟𝑎 , respectively. Among them, 𝜆𝑖𝑛𝑠 and 𝜆𝑡𝑟𝑖 are treated as hyperparameters, while 𝑤𝑖𝑛𝑡𝑒𝑟 and 𝑤𝑖𝑛𝑡𝑟𝑎 are adaptively determined through
dynamic weight adjustment. The overall hierarchical loss function is
formulated as follows:

Inter False Positive
Negative
Intra False Positive
Negative

(6)

𝑗∈𝐹 𝑃𝑖

where 𝑞𝑖𝑛𝑡𝑟𝑎,𝐼 is the contrastive loss within images, similarly, 𝑞𝑖𝑛𝑡𝑟𝑎,𝑇
can be acquired by replacing (𝑓𝐼𝑞 [𝑖], 𝑓𝐼𝑞 [𝑗]) with (𝑓𝑇𝑞 [𝑖], 𝑓𝑇𝑞 [𝑗]). Herein,
𝐵𝑞 denotes the number of valid samples for attribute 𝑞. 𝛼(⋅) represents
the RELU activation function. 𝑓𝐼𝑞 [𝑖] represents the features of attribute 𝑞
for the 𝑖𝑡ℎ image, and 𝑓𝑇𝑞 [𝑗] represents the features of attribute 𝑞 for the
𝑗𝑡ℎ text. All features are processed through self-attention pooling and
feature fusion. 𝐹 𝑁𝑖 represents the collection of false negative samples
with the same ID as sample 𝑖, and 𝐹 𝑃𝑖 represents the collection of false
positive samples. 𝑆(⋅) represents the cosine similarity as formulated in
Eq. (5). Consequently, the overall intra-modal attribute contrastive loss
𝑞𝑖𝑛𝑡𝑟𝑎 for attribute 𝑞 is derived using Eq. (7).
(
)
1
𝑞𝑖𝑛𝑡𝑟𝑎 =
𝑞𝑖𝑛𝑡𝑟𝑎,𝐼 + 𝑞𝑖𝑛𝑡𝑟𝑎,𝑇 .
(7)
2
Inter-modal Attribute Loss: Inter-modal attribute alignment further enforces the alignment of samples with the same attribute across
different modalities in the shared embedding space. Although a shared
encoder implicitly maps different modalities into a common feature
space, it does not guarantee sufficiently compact alignment between semantically consistent instances. Therefore, this constraint is introduced
to explicitly form a tight cluster among cross-modal samples sharing the
same attribute, enhancing fine-grained semantic consistency beyond
global alignment. For inter-modal false positive and false negative
attribute pairs, we calculate 𝑞𝑖𝑛𝑡𝑒𝑟 for each attribute 𝑞 using a formula
similar to that in Eq. (6):
𝑞𝑖𝑛𝑡𝑒𝑟 =

𝐵𝑞 [ (
)
(
)
1 ∑
𝛼 𝜃pos − max S 𝑓𝐼𝑞 [𝑖], 𝑓𝑇𝑞 [𝑗]
𝑗∈𝐹 𝑁𝑖
𝐵𝑞 𝑖=1
(
)]
( 𝑞
)
+ 𝛼 max S 𝑓𝐼 [𝑖], 𝑓𝑇𝑞 [𝑗] − 𝜃neg .

(9)

where 𝜖 is the triplet margin, set as 0.2 based on experimental comparisons.

Intra False Negative

shifts. For intra-modal false positive and false negative attribute pairs,
we compute 𝑞𝑖𝑛𝑡𝑟𝑎 for each attribute 𝑞, as shown below:
𝑞𝑖𝑛𝑡𝑟𝑎,𝐼 =

𝑞
( (
(
))
1 ∑
max 0, 1 − S 𝑓𝐼𝑞 [𝑖], 𝑓𝑇𝑞 [𝑝]
𝐵𝑞 𝑖=1
(
(
))
)
− 1 − S 𝑓𝐼𝑞 [𝑖], 𝑓𝑇𝑞 [𝑛] + 𝜖 ,

𝑡𝑜𝑡𝑎𝑙 = 𝜆𝑖𝑛𝑠 ⋅ 𝑖𝑛𝑠 + 𝑖𝑑 + 𝑎𝑡𝑡𝑟 ,
𝑎𝑡𝑡𝑟 =

∑( 𝑞
)
𝑤𝑖𝑛𝑡𝑒𝑟 ⋅ 𝑞𝑖𝑛𝑡𝑒𝑟 + 𝑤𝑞𝑖𝑛𝑡𝑟𝑎 ⋅ 𝑞𝑖𝑛𝑡𝑟𝑎 + 𝜆𝑡𝑟𝑖 ⋅ 𝑞𝑡𝑟𝑖 ,

(10)
(11)

𝑞∈

 represents the set of attributes. 𝜆𝑖𝑛𝑠 and 𝜆𝑡𝑟𝑖 are both set to 0.5 based
on experimental comparisons, while 𝑤𝑞𝑖𝑛𝑡𝑒𝑟 and 𝑤𝑞𝑖𝑛𝑡𝑟𝑎 are determined by
the dynamic weight adjustment formula, as shown in Eq. (12).
(
)
Var 𝑞𝑖𝑛𝑡𝑟𝑎
𝑤𝑞𝑖𝑛𝑡𝑒𝑟 =
,
(
)
(
)
Var 𝑞𝑖𝑛𝑡𝑒𝑟 + Var 𝑞𝑖𝑛𝑡𝑟𝑎 + 𝜂
(12)
(
)
Var 𝑞𝑖𝑛𝑡𝑒𝑟
𝑤𝑞𝑖𝑛𝑡𝑟𝑎 =
,
( 𝑞 )
( 𝑞 )
Var 𝑖𝑛𝑡𝑒𝑟 + Var 𝑖𝑛𝑡𝑟𝑎 + 𝜂

(8)

where 𝜂 is a small value added to prevent division by zero, set to
1 × 10−8 , and Var(⋅) denotes the variance of the loss values, specifically
computed over all training loss values recorded at intervals of 100
batches within each individual epoch. 𝑤𝑞𝑖𝑛𝑡𝑒𝑟 and 𝑤𝑞𝑖𝑛𝑡𝑟𝑎 are both initiated
as 1 and subsequently updated at the conclusion of each training epoch.
Moreover, deviating from conventional average pooling, we leverage
self-attention pooling as a dedicated method for aggregating image–text
features.

𝑗∈𝐹 𝑃𝑖

All variable definitions follow the formula description of intramodal attribute loss (Eq. (6)) in the previous subsection.
Attribute Triplet Loss: After implementing the hard sample mining, the remaining false positive and false negative attribute pairs
are used in the attribute-aware triplet loss to strengthen cross-modal
alignment. For an image sample 𝑖, when classifying a text sample 𝑛
5

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

𝑣 . Herein, in this study, we select 𝑟 = 0.8 as the optimal value
feature 𝐹𝑎𝑡𝑡𝑞
based on experimental comparisons. In text attribute description, Fig. 3
reveals that, beyond clothing and accessories, gender is frequently mentioned. Based on this observation, we introduce a new ‘gender’ attribute
during text attribute segmentation to enhance attribute completeness.
Since gender identification requires comprehensive semantic analysis of
images, we align the textual gender attribute with global image features
to promote cross-modal consistency.
While our current framework is evaluated on a fixed set of attributes, it is important to discuss its extensibility to dynamic or larger
attribute sets. Our framework’s design inherently supports scalability
through several mechanisms. First of all, as each attribute operates
as an independent alignment unit with its own segmentation module,
adding a new attribute 𝑎𝑛𝑒𝑤 simply introduces an additional pairing
stream without affecting existing ones. The computational overhead
grows approximately linearly as (𝑛), where 𝑛 is the number of attributes. Secondly, the loss functions are designed to be attribute-aware
and naturally extendable. Our attribute-level loss Eq. (11) compute similarities independently for each attribute pair. When extending to more
attributes, the loss aggregates over all available attributes without requiring architectural changes. The threshold-based pairing mechanism
(𝜃𝑝𝑜𝑠 , 𝜃𝑛𝑒𝑔 ) remains applicable regardless of attribute count.

Fig. 3. Word cloud of the text in CUHK-PEDES [17] datasets. Larger font size
indicates higher occurrence frequency.

3.4. Attribute-enriched dataset construction
Despite the extensive collection of images and annotations in existing TI-ReID datasets, challenges such as interference, incorrect text annotations, and varied sentence structures remain, significantly limiting
the performance of TI-ReID models. By leveraging pre-trained semantic
segmentation models and large language models, we address these issues from both visual and textual perspectives, enabling more accurate
attribute segmentation. Specifically, we use Grounded SAM [39] as
the image attribute segmentation model and ChatGPT’s [38] multi-turn
dialogue for text attribute segmentation. We apply these segmentation
methods to three benchmarks: CUHK-PEDES [17], ICFG-PEDES [18],
and RSTPReid [19], resulting in attribute-enriched datasets.
To ensure controlled experiments and facilitate reproducible comparisons, we adopt a predefined closed-set attribute space, selecting
five core visual attributes: hair, upper body, lower body, bag, and
shoes. This choice is motivated by three factors: (1) To validate the
effectiveness of the hierarchical alignment framework under controlled
conditions, reducing complexity; (2) To align with common practices in
TI-ReID [11,13], where constrained attribute spaces are typically used
for benchmark comparisons; (3) As shown in Fig. 3, our word frequency
analysis reveals that these five attributes make up the majority of
descriptive nouns in text annotations and occupy significant, easily
locatable regions in images.
We also account for less prominent and infrequently mentioned attributes. While Grounded SAM [39] and ChatGPT excel in segmentation
and language understanding, challenges arise when dealing with absent
or small attributes (e.g., shoes). To avoid information loss, attributes
not mentioned in text are marked as NULL, and full text descriptions
are used for matching. For image attributes, we perform a weighted
fusion of the segmented attribute features with the original image
features to balance local details and global context. Specifically, the
fused visual feature representation is defined as:
𝑣̂ = 𝑀 ⊙ 𝐹 𝑣 ,
𝐹𝑎𝑡𝑡𝑞
𝑞
𝑔

(13)

𝑣
𝑣̂ + (1 − 𝑟) × 𝐹 𝑣 ,
𝐹𝑎𝑡𝑡𝑞
= 𝑟 × 𝐹𝑎𝑡𝑡𝑞
𝑔

(14)

3.4.1. Visual attribute segmentation
The pre-trained Grounded SAM in our work combines the object
detection capability of Grounding DINO [41] and the segmentation
capability of SAM [42], enabling a two-step segmentation process: detection followed by fine-grained segmentation. With the powerful zeroshot detection capability of Grounding DINO, Grounded SAM can locate
any object in an image using text prompts and then perform precise
segmentation with SAM’s robust capabilities. We validate segmentation
accuracy by using detection confidence scores from Grounding DINO.
For small targets (e.g., ‘shoes’), we apply a lower confidence threshold
to accommodate their inherently lower signal-to-noise ratio and avoid
information loss and excessive NULL values. Specifically, the confidence threshold is set to 0.1 for small targets and 0.3 for larger targets.
Qualitative results are shown in Fig. 7, displaying detection confidence
scores and corresponding segmentation outputs. The resulting attributeenriched datasets include comprehensive detection confidence scores
and segmentation results, which will be publicly available.
3.4.2. Textual attribute extraction
To generate text annotations for attribute labeling via large language models, we provide the following prompt to ChatGPT:
Please extract the ‘attribute’ field from the content of the ‘captions’ field
while retaining all other original fields. The required ‘attributes’ include
‘hair’, ‘upper body’, ‘lower body’, ‘bag’, ‘shoes’, and ‘gender’. Output ‘null’
for content not mentioned in the text; return ‘null’ for ambiguous gender
references. Do not add any extra attribute fields.
We limit our text attribute classification to six categories based
on two key considerations:(1) Minor features like glasses or phone
pose challenges for image segmentation accuracy, and including them
would result in a high incidence of NULL values during text attribute
parsing; (2) Overemphasizing such features may bias the model toward
non-critical information, increasing the risk of overfitting. This sixattribute framework strikes a balance by ensuring effective extraction
of textual descriptions for core features while preserving the semantic
coherence of descriptive sentences. For example, in processing a phrase
like ‘a T-shirt with a flower pattern’ a large language model can
map this description to the upper body attribute, while traditional
tokenization methods may break it into fragmented noun combinations,
undermining semantic integrity.

where 𝑀𝑞 is the binary mask of the 𝑞𝑡ℎ attribute, and 𝐹𝑔𝑣 represents
the global image feature. By performing element-wise multiplication ⊙
between the 𝑀𝑞 and the 𝐹𝑔𝑣 , we obtain the attribute-specific feature
̂𝑣 for the 𝑞 attribute. For comprehensive feature
representation 𝐹𝑎𝑡𝑡𝑞
𝑡ℎ
fusion, Eq. (14) integrates attribute-specific information with global
context through a balancing ratio 𝑟 to obtain the refined attribute
6


exec
/bin/zsh -lc "pdftotext -f 3 -l 6 'ADA framework for unsupervised domain adaptation person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 171 (2026) 112238

W. Zhang et al.

Fig. 1. The ﬂowchart for the ADA framework: the ADA framework is deployed only once, and the eﬀective deployment location is behind any one of the stages
1/2/3 of the backbone network. The backbone is inserted a hybrid classiﬁer. The pseudo labels were generated after DBSCAN clustering and updated once at the
starting of each training epoch.

Each mini-batch is sequentially processed through our network pipeline,
which comprises:(1) A ResNet-50 backbone [14] (pre-trained on ImageNet, containing stages 0–4). (2) The ADA framework. (3) Cluster computation module. (4) Composite Re-ID loss 𝐿𝑅𝑒𝐼𝐷 .
As illustrated in Fig. 1, the Re-ID loss 𝐿𝑅𝑒𝐼𝐷 combines classiﬁcation
loss 𝐿𝑐𝑙𝑠 [15] and triplet loss 𝐿𝑡𝑟𝑖 [16]. The complete training loss is
formulated as:
𝐿𝑅𝑒𝐼𝐷 = 𝐿𝑐𝑙𝑠 + 𝐿𝑡𝑟𝑖

(1)

The ADA framework’s deployment requires balancing between computational overhead and performance optimization. Operating as a plugand-play module across both training and inference phases, the framework’s architecture permits ﬂexible scalability. In our implementation,
a simple conﬁguration is adopted with single SAAB block integration.
Note: (1)after stages 1/2/3 are all eﬀective deployment locations. This
design choice ensures computational eﬃciency while maintaining adaptation eﬀectiveness. (2) Inspired by the review [17,18], with the rapid
evolution of sparse self-attention mechanisms, it is possible to ﬂexibly
decide whether to stack SAAB blocks based on practical needs. Fig. 2
provides detailed architectural speciﬁcations.

Fig. 2. The diagram of the ADA framework: for simplicity, the BN and ReLU
operations are not drawn in the ﬁgure. In this article, stackable SAAB block is
only stacked once, and the performance improvement is signiﬁcant. Therefore,
its memory usage is relatively reasonable.

3.2. Introduction to the ADA framework

The SAP augments the capacity of domain adaptation through: (1)
Sparse Attention Eﬃciency: Implements Sparse complexity attention(<
𝑂(𝑁 2 )) versus conventional approaches(= 𝑂(𝑁 2 )). (2) Modular Flexibility: Permits plug-and-play integration of various sparse self-attention
variants. (3) Symmetrical Feature Encoding: Channel-wise symmetric
distributions emerge through bottleneck-residual coupling. Fig. 3 depicts the functional block diagram of SAP.

3.2.1. Introduction to the SAAB block and SAP
The ADA framework enhances cross-domain discriminative capability through domain-invariant identity representation learning.
Its core innovation lies in dual attention mechanisms: strong focus
on identity-salient features (pedestrian attributes) while suppressing
domain-speciﬁc distractors (background elements) via long-range dependency modeling. As joint training progresses, the framework demonstrates progressive domain adaptation through the SAAB block and the
computational primitive SAP.
To enhance domain adaptation performance, the SAAB block introduces the following key improvements over the bottleneck block structure: (1) Core Computational Primitive. The bottleneck block relies on
standard convolution, whereas the SAAB block replaces it with the SAP
(Sandwich Attention Primitive). (2) Bi-Level Bottleneck Structure. The
ﬁrst level is embedded via SAP. The second level is constructed by
the SAAB block based on SAP. (3) Expanded Receptive Field. In bottleneck block, the receptive ﬁeld is limited by a single 3 × 3 convolution. The SAAB block achieves broader contextual awareness through
self-attention mechanism. (4) Overﬁtting Integration. The SAAB block
incorporates dropout operations, which are absent in bottleneck block,
further improving robustness.

3.2.2. Algorithm description for the SAP
To rigorously characterize the SAP, we present its mathematical implementation through PyTorch-style pseudocode across ﬁve sparse selfattention variants(Linear Self-Attention [19], Permuted Self-Attention
[20], Focused Linear Attention [21], Bi-Level Routing Attention [22],
Window based Self-Attention [23]). The pseudocode description is presented in the following algorithmic listing.
#i n p u t f e a t u r e Xin a r e ( Cin , H, W) . o u t p u t f e a t u r e a r e
( Cout , H, W)
#2D c o n v o l u t i o n Conv2d_head , ( Cout , H, W)
x1 = Conv2d_head ( Xin )
#2D c o n v o l u t i o n Conv2d_down , ( Cin /4 ,H, W)
x = Conv2d_down ( Xin )
# p e r m u t e from ( Cin /4 ,H,W) t o (H,W, Cin /4)
3

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

Q = ( x ^∗∗p) ∗ | | x | | / | | x ^∗∗p | |
K = ( x ^∗∗p) ∗ | | x | | / | | x ^∗∗p | |
V = ( x ^∗∗p) ∗ | | x | | / | | x ^∗∗p | |
# a t t e n t i o n compute
a t t e n=MatrixMul (Q, K . t r a n s p o s e )
#conv2d_dwc , a d e p t h w i s e c o n v o l u t i o n
x2=MatrixMul ( a t t e n , V) + conv2d_dwc (V)
# r e s t o r e t o ( Cin /4 ,H,W)
x2 = reshape ( x2 )

x = permute ( x )
# function Sparse_attention
r e f e r s t o s p a r s e s e l f −a t t e n t i o n p r o c e s s i n g
x2= S p a r s e _ a t t e n t i o n ( x )
#2D c o n v o l u t i o n Conv2d_BneckUp r e s t o r e c h a n n e l s t o
( Cin , H, W) x2 = Conv2d_BneckUp ( x2 )
#2D c o n v o l u t i o n r e s t o r e c h a n n e l s t o ( Cout , H, W)
x2 = C o n v 2 d _ t a i l ( x2 )
# s p a r s e and c o n v o l u t i o n f e a t u r e f u s i o n
Output = x1 + x2

# c a s e 4 , Bi−L e v e l R o u t i n g A t t e n t i o n
#r e s h a p e t e n s o r t o ( S ^ 2 ,HW/S ^ 2 , Cin /4)
#S : t h e number o f r e g i o n s d i v i d e d by H and W o f
f e a t u r e map x = r e s h a p i n g ( x , reg ion_H=H// S )
#L i n e a r p r o j e c t i o n o f q u e r i e s , keys , and v a l u e s
Q, K , V = q k v _ L i n e a r P r o j e c t i o n ( x ) . chunk ( 3 , dim=−1)
#a v e r a g e on a l l r e g i o n s , g e t r e g i o n a l q u e r i e s , k e y s
( S ^ 2 , C)
Q_r , K _ r = Q. AVG( dim=1), K . AVG( dim=1)
#a d j a c e n c y m a t r i x AM ( S ^ 2 , S ^ 2 ) f o r
#r e g i o n−to−r e g i o n g r a p h
# MatrixMul : m a t r i x m u l t i p l i c a t i o n
AM = MatrixMul ( Q_r , K _ r . permute(−1, −2))
#k : t o t a l number o f e f f e c t i v e r e g i o n s t o a t t e n d
#p r o d u c e topK i n d e x m a t r i x IM ( S ^ 2 ,K)
IM = I n d e x M at r i x (AM, K ) . topK
# c o l l e c t key and v a l u e p a i r s ( S ^ 2 , kHW/S ^ 2 , C)
K_g = K V _ c o l l e c t (K , IM )
V_g = K V _ c o l l e c t (V , IM )
#t o k e n−to−t o k e n a t t e n t i o n
#MatrixMul : m a t r i x m u l t i p l i c a t i o n
Atten = MatrixMul (Q, K_g . permute(−2, −1))
Atten = softmax ( Atten , dim=−1)
# conv2d_dw ( ) : d e p t h w i s e c o n v o l u t i o n
x= MatrixMul ( Atten , V_g ) + conv2d_dw (V)
#r e s h a p e t e n s o r from ( S ^ 2 ,HW/S ^ 2 , Cin /4) t o
(H,W, Cin /4) x = unreshape ( x , reg ion_H = H// S )
#p e r m u t e t e n s o r t o ( Cin /4 ,H, W)
x2 = permute ( x )

# c a s e 1 , L i n e a r S e l f−A t t e n t i o n
#The low−rank m a t r i x a p p r o x i m a t e s
#s e l f −a t t e n t i o n mechanism .
#L i n e a r p r o j e c t i o n , Q, K and V a r e (HW, Cin /4)
Q = linear_Q (x)
K = linear_K (x)
V = linear_V (x)
# l i n e a r P r o j e c t i o n Matrix , PM_k ( P r j ,HW)
# l i n e a r P r o j e c t i o n Matrix , PM_v ( P r j ,HW)
#K_p and V_p i s ( P r j , Cin / 4 ) , and P r j << HW
#MatrixMul : m a t r i x m u l t i p l i c a t i o n
K_p = MatrixMul ( PM_k , K)
V_p = MatrixMul ( PM_v , V)
# a t t e n t i o n compute
a t t e n = MatrixMul (Q, K_p . t r a n s p o s e )
a t t e n = softmax ( a t t e n )
x2 = MatrixMul ( a t t e n , V_p )
# r e s t o r e t o ( Cin /4 ,H,W)
x2 = reshape ( x2 )
# c a s e 2 , Permuted S e l f−A t t e n t i o n
# r i s tokens reduction factor .
#S i s t h e s i d e l e n g t h o f each window .
#HW=N∗S ^ 2
#r e s h a p e t e n s o r t o (N, S ^ 2 , Cin /4)
x = reshape ( x )
# Q m a i n t a i n s (N, S ^ 2 , Cin /4)
Q = linear_query (x)
#K and V from (N, S ^ 2 , Cin /4) t o (N, S ^ 2 , Cin /(4∗ r ^ 2 ) )
K = linear_key (x)
V = linear_value (x)
#t r a n s f e r
s p a c e t o k e n s i n K and V
to channel dimention
#K_p and V_p i s (N∗S ^ 2 / r ^ 2 , Cin /4)
K_p , V_p= permute (K , V)
#s e l f −a t t e n t i o n p e r f o r m e d on Q, K_P and V_P
#PE i s an a l i g n e d r e l a t i v e p o s i t i o n embedding
a t t e n = MatrixMul (Q, K_p . t r a n s p o s e ) + PE
a t t e n = softmax ( a t t e n )
x2 = MatrixMul ( a t t e n , V_p )
# r e s t o r e t o ( Cin /4 ,H,W)
x2 = reshape ( x2 )

# c a s e 5 , Window b a s e d S e l f−A t t e n t i o n
#windows g e n e r a t i o n
#r e s h a p e t e n s o r from ( 1 , n , Cin /4) t o (Wn, n1 , Cin /4)
#Wn i s t h e number o f windows ,
#n1 i s t h e p r o d u c t o f t h e l e n g t h and w i d t h o f t h e
window x = reshape ( x )
#p r o d u c e Q, K , V by l i n e a r
#t e n s o r from (Wn, n1 , Cin /4) t o (Wn, n1 , 3 ∗ Cin /4)
QKV = l i n e a r ( x )
#t e n s o r from (Wn, n1 , 3 ∗ Cin /4) t o ( 3 ,Wn, n1 , Cin /4)
QKV = reshape (QKV)
#Q, K , V =(Wn, n1 , Cin /4)
Q = QKV[0]
K = QKV[1]
V = QKV[2]
# a t t e n t i o n compute
a t t e n = MatrixMul (Q, K . t r a n s p o s e )
a t t e n = softmax ( a t t e n )
x2 = MatrixMul ( a t t e n , V)
# r e s t o r e t o ( Cin /4 ,H,W)
x2 = reshape ( x2 )

# case 3 , Focused Linear A t t e n t i o n
# p i s t h e v a l u e o f t h e power o f X
#F o c u s e d F u n c t i o n a d j u s t t h e d i r e c t i o n o f
#each q u e r y and key f e a t u r e s , d r i v i n g s i m i l a r
#query−key p a i r s c l o s e r w h i l e p u s h i n g
# d i s s i m i l a r query−key p a i r s away
#f ( x ) i s F o c u s e d F u n c t i o n
#f ( x ) = ( x ^∗∗ p ) ∗ | | x | | / | | x ^∗∗ p | |
#x ^∗∗ p r e p r e s e n t s e l e m e n t−w i s e power p o f x
x = ReLU ( x )

Pseudocode description reveals three fundamental design principles:
•

4

(1) Spatial-Channel Feature Redistribution: (I) Permuted SelfAttention induces channel-spatial attention equilibrium through

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

Table 1
Benchmarks information of UDA re-ID.

CHUK03

market1501

MSMT17

Fig. 3. The diagram of the SAP: the SAP achieves architectural ﬂexibility, enhances feature representation , and optimizes the complexity-accuracy tradeoﬀ.

•

subset

IDs

images

cameras

train
query
gallery

757
700
700

7365
1400
5332

6
6
6

subset

IDs

images

cameras

train
query
gallery

751
750
751

12,936
3368
15,913

6
6
6

subset

IDs

images

cameras

train
query
gallery

1041
3060
3060

32,621
11,659
82,161

15
15
15

Batch Conﬁguration: 64 source-domain samples (16 identities) + 64
target-domain samples per batch.

Under ﬁxed DBSCAN hyperparameters, our methodology emphasizes systematic evaluation of the framework’s intrinsic domain adaptation capabilities. The clustering mechanism serves strictly as an auxiliary process, generating pseudo-labels across experiment. This controlled conﬁguration enables rigorous comparative analysis by isolating
attention mechanism eﬀects from clustering variability.
Architectural generalization strategy. To establish the architectural generality of the ADA framework, three uniﬁed design principles
were implemented regardless of sparse self-attention selection: (1) interface standardization, (2) hyperparameter consistency, and (3) simple
SAAB block conﬁguration. The hyperparameter space was intentionally
constrained, maintaining fewer than 4 tunable parameters per module to
ensure comparative fairness. This systematic evaluation enables direct
measurement of the ADA framework’ intrinsic adaptation capability.

dimension transposition. (II) Window-based Self-Attention maintains spatial locality constraints via ﬁxed-size partitioning.
• (2) Computational Complexity Optimization: (I) Linear SelfAttention implements low-rank matrix approximating self-attention
mechanism. (II) Bi-Level Routing Attention reduces global computation through regional token routing.
• (3) Attention Distribution Approximation: (I) Focused Linear Attention employs learnable kernel mapping for softmax approximation.
(II) Linear variants utilize low-rank matrices for attention matrix estimation.
The SAP architecture demonstrates three key capabilities through
these implementations:
•

(1) Architectural Flexibility: Seamless integration of diverse sparse
self-attention paradigms.
• (2) Feature Fusion Enhancement: Complementary convolutionattention feature synthesis.
• (3) Complexity-Accuracy Tradeoﬀ: Maintains high accuracy while
reducing FLOPs.

4.3. Comparison with state-of-the-art methods
Our framework is benchmarked against SOTA unsupervised person
re-identiﬁcation methods. As demonstrated in Table 2, our framework
achieves competitive performance while exhibiting superior architectural ﬂexibility - speciﬁcally enabling deployment at multiple network
hierarchies without structural constraints, unlike position-ﬁxed implementations in existing methods [28].
The proposed methodology demonstrates broad compatibility with
diverse baseline architectures. Comparative analysis against topperforming baselines in the tabular comparison reveals that our approach exhibits a more substantial performance improvement margin.
PP+LP [29] improves performance by leveraging memory-based
cluster contrastive learning to reﬁne features and pseudo-labels, albeit

4. Experiments
4.1. Experiments setting and evaluation protocol
We evaluate our framework on three UDA-ReID benchmarks
(CUHK03 [24], MSMT17 [4], and Market-1501 [25]) . Performance is
quantiﬁed using standard metrics: mean Average Precision (mAP) and
Cumulative Matching Characteristics (CMC) Rank-1 accuracy. Our implementation strictly uses only source domain labels and target domain
pseudo-labels generated through joint training, without additional supervision. Dataset statistics are provided in Table 1.

Table 2
Evaluation compared to leading methods on benchmark datasets. (MS→M:
MSMT17 [4] → Market-1501 [25]).

4.2. Implementation details

MS → M

M → MS

mAP R1

mAP

R1

42.4
−
81.0
81.5
82.7
85.3
85.8

−
13.4
31.3
31.4
32.1
41.8
39.5

−
34.3
60.4
61.1
59.6
71.4
67.9

high-performance baseline(IBN-ResNet50)
IBN-ResNet50+ADA (Ours)

80.1 91.3 20.2
83.4 92.8 23.1

41.3
46.0

Baseline (ResNet50)
ResNet50+ADA (Ours)

70.4 86.1 11.9
80.2 91.8 17.0

27.3
36.5

Method

Experiments were conducted on a workstation equipped with dual
NVIDIA RTX 3090 GPUs, AMD Ryzen R7 7600X CPU, and 128GB DDR5
RAM. The framework was implemented in PyTorch with the following
conﬁgurations:

HUDA [2]
MDJL [30]
SECRET [31]
PPLR [32]
MTNet [33]
CCL [28]
FP+LP [29]

Image Processing: Inputs resized to 256 × 128 pixels with standard
augmentation protocols (random ﬂipping, cropping, and erasing).
• Clustering: DBSCAN [26] with Jaccard distance metric [27] (maximum neighbor distance 𝐸𝑃 𝑆 = 0.6, minimum neighbor count
𝑛min = 4).
• Optimization: Adam optimizer with initial learning rate𝜂 = 0.001 ,
0
step-decay schedule(𝜂𝑡+20 = 0.1𝜂𝑡 ) over 50 epochs.
•

5

Reference

PR
PR
AAAI
CVPR
JVCI
TCSVT
TIP

72.3
−
92.6
92.8
93.0
94.1
94.5

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

at the expense of increased architectural and computational complexity
(due to its teacher-student framework) compared to our approach. Compared to PP+LP [29], the baseline(82.8 % mAP, 92.7 % R1) of PP+LP
is higher than our baseline(80.1 % mAP, 91.3 % R1). Although PP+LP
attains higher absolute performance metrics, the ADA framework shows
greater relative performance improvement(↑ 3.3 % mAP/1.5 % R1) over
baseline implementations.
Unlike CCL [28], which combines contrastive learning, attention
module(3DAM), and unsupervised learning with a memory bank for
performance gains, our ADA framework intentionally avoids memory
bank to assess its intrinsic capability. While 3DAM needs multi-stage
deployment for optimal results, and CCL relies on camera-count labels,
the ADA framework operates without such constraints. Currently, CCL
achieves higher performance, but integrating memory bank could further enhance the ADA framework.

directly validate the framework’s design objectives. Table 3 summarizes the peak performance metrics of four distinct dataset combinations derived from three benchmark datasets (CUHK03, Market-1501,
and MSMT17).
Through systematic analysis of Table 3, we observe the following
performance enhancement (Note: (1) sorted by improvement magnitude. (2) CUHK03 to Market-1501):
the ADA framework (Linear Self-Attention): ↑ 16.5 % mAP/10.6 %
R1.
• the ADA framework (Window based Self-Attention): ↑ 15.2 % mAP/
9.5 % R1.
• the ADA framework (Focused Linear Attention): ↑ 13.9 % mAP/9.4 %
R1.
• the ADA framework (Permuted Self-Attention): ↑ 12.3 % mAP/ 8.4 %
R1.
• the ADA framework (Bi-Level Routing Attention): ↑ 11.4 %
mAP/8.1 % R1.
•

4.4. Ablation study
We conducted ablation studies to assess three key aspects of the ADA
framework: generalizability, consistency, and adaptability. The evaluation spanned four dimensions: (1) Datasets, (2) Baseline methods,
(3) Backbone network architectures, (4) Computational primitives. The
experimental design incorporates three distinct baseline variants implemented through backbone network substitution: (1) a lightweight baseline, (2) an optimized high-performance baseline, and (3) a transformeroriented architectural implementation.
For precise interpretation of ablation results, we categorize the backbone network’s feature hierarchy as follows: (1) stages 0–1, low-level
feature. (2) stage 2, intermediate semantic representation. (3) stages
3–4, high-level feature.
The lightweight baseline comprises the following key components: a
ResNet-50 backbone network, DBSCAN clustering algorithm, and a reidentiﬁcation loss function 𝐿𝑅𝑒𝐼𝐷 . A lightweight baseline is more conducive to validating the potential of ADA framework.

The experimental data presented in Tables 4–8 demonstrate that the
ADA framework successfully fulﬁlls the ﬁrst four predeﬁned design objectives, as evidenced by a rigorous analytical evaluation. Tables 4–8
present the experimental results for ﬁve sparse self-attention variants
across various dataset conﬁgurations.
4.4.2. Performance degradation when substituting sparse self-attention
with convolution
We replaced the sparse self-attention inside SAP with conventional
2D 3 × 3 convolution, and kept everything else unchanged. This new
module repeated the experiments done by the ADA framework. The experimental results recorded in the Table 9 show that compared with
the ADA framework based on sparse self-attention, the performance and
generality of this module have signiﬁcantly decreased. The comparative
experiments further prove that SAP, SAAB block, and ADA framework
can signiﬁcantly improve domain adaptation performance.

4.4.1. The universality of the ADA framework
We systematically investigate the cross-paradigm compatibility of
the proposed framework through dual analytical dimensions: (1)
dataset generalization and (2) sparse self-attention architectural design
paradigms. Five representative sparse self-attention variants are progressively implemented within the ADA framework.
The ADA framework is designed with ﬁve objectives: (1) compatibility with diverse sparse self-attention design paradigms; (2) ﬂexible
deployment; (3) empirical eﬀectiveness across multiple datasets; (4)
simpliﬁed and uniﬁed parameter conﬁgurations; (5) multi-architecture
adaptability. Currently, no customized optimizations are implemented
for individual sparse self-attention variants. All design paradigms adopt
standardized external settings to enable three key advantages: paradigm
selection ﬂexibility, deployment simplicity, and parameter tuning eﬃciency. The framework prioritizes architectural versatility over performance consistency across varying sparse self-attention paradigms. Performance variations naturally occur when switching design paradigms.
The measured capability diﬀerences in the Tables 3 and 4–8 therefore

4.4.3. Applicability of backbone network
The ADA framework maintains robust domain adaptation performance after replacing its ResNet50 backbone with IBN-ResNet50 [34].
For empirical veriﬁcation, we employ the MSMT17 dataset-a highly
challenging cross-domain benchmark in visual recognition. Experimental results conﬁrm that the ADA framework achieves signiﬁcant performance improvements despite its stronger baseline. Detailed quantitative
metrics are provided in Table 10.
The ADA framework exhibits promising transferability to
transformer-based architectures. To validate this capability, we
systematically evaluate its integration within ViT [35] by deploying it
across distinct architectural stages. Empirical evidence demonstrates
that the ADA framework preserves its performance beneﬁts. Comprehensive testing across all transformer variants remains computationally
impractical. However, our methodology demonstrates that selective
placement at critical architectural nodes achieves optimal performance.
Table 11 records the performance data of eﬀective deployment points.

Table 3
The summary of peak performance of the ADA framework. (C → M : CUHK03 [24] to Market-1501 [25]).
(M → MS : Market-1501 [25] to MSMT17 [4]).
Setting

ResNet50 (Baseline)
Baseline + ADA (Liner Self-Attention)
Baseline + ADA (Permuted Self-Attention)
Baseline + ADA (Focused Linear Attention)
Baseline + ADA (Window based Self-Attention)
Baseline + ADA (Bi-Level Routing Attention)

C→M

M → MS

MS → M

mAP

R1

mAP

R1

mAP

R1

mAP

R1

56.7
73.2
69.0
70.6
71.9
68.1

77.3
87.9
85.7
86.7
86.8
85.4

7.2
10.8
9.5
9.0
9.2
9.2

5.8
9.0
8.5
7.6
7.9
7.2

11.9
17.0
15.7
16.0
14.0
14.7

27.3
36.5
34.0
34.1
31.1
31.4

70.4
78.9
78.7
80.2
75.6
79.2

86.1
91.2
91.1
91.8
88.9
90.4

6

M→C


web search: 
web search: "A training-free framework for text-to-image person re-identification via query-prototype matching"
web search: 
web search: "Adaptive Pseudo-Label Purification and Debiasing" "Visible-Infrared"
web search: 
web search: https://doi.org/10.1016/j.patcog.2026.113705
web search: 
web search: 10.1016/j.patcog.2025.111443
web search: 
web search: site:sciencedirect.com/science/article/pii "Adaptive transformer with Pyramid Fusion"
web search: 
web search: Adaptive transformer Pyramid Fusion cloth-changing Person Re-Identification GitHub
web search: 
web search: github.com ziyuzhu452-create A2HA
web search: 
web search: "Propot" "Text-to-image" person re-identification prototype prompt
web search: 
web search: "VC-Clothes-W&R"
web search: 
web search: "AOANet" "Occluded-Duke" "70.6" "81.2"
web search: 
web search: "3524555" "Adaptive Occlusion-Aware Network"
codex
从在线检索看，有些 2025/2026 期刊题名还没有被普通搜索稳定索引，但几个关键先例能查到：文本 ReID 的身份原型训练线、属性提示线、IRRA，以及 USVI 的 RPNR 会议版。这些足够辅助判断“视角是否新”，主分析仍以本地 PDF 方法段为准。
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
/bin/zsh -lc "pdftotext -f 4 -l 6 'A training-free framework for text-to-image person re-identification via query-prototype matching.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113705

H. Yang et al.

injects identity-level structure from the image modality. The core idea
is to cluster gallery features into identity prototypes that summarize
predominant appearance patterns for each identity in the visual feature
space. Images of the same person tend to form compact clusters, so
each prototype acts as a stable identity anchor that is less sensitive to
instance-level outliers. At inference time, a text query is compared not
only with individual gallery images but also with these prototypes, so
that image-modality structure can be exploited to refine cross-modal
matching and stabilize rankings.
As illustrated in Fig. 3, given a text query and an image gallery, we
first use a pretrained vision–language model to compute instance-level
Query–Gallery Similarity (QG-Similarity) between the text query and
all gallery images. The proposed Prototype-Driven Enhancement (PDE)
module then clusters gallery features into identity-level prototypes and
computes Query–Prototype Similarity (QP-Similarity), which is fused
with QG-Similarity to yield enhanced ranking scores. The cascaded
variant PDE𝑀 extends this idea to multiple inference stages, gradually
decreasing both prototype granularity and fusion strength, so that early
stages make stronger, fine-grained corrections and later stages perform
more conservative updates guided by coarser identity structure. To
make the pipeline self-adaptive across datasets, we further introduce
the Adaptive Cluster Estimation (ACE) module, which estimates the
number of prototypes from the gallery-density distribution and passes it
to PDE/PDE𝑀 . The resulting framework is plug-and-play, operates entirely at test time without any additional training, and can be attached
to diverse VLM-based TI-ReID backbones.

2.3. Re-ranking for ReID
ReID can be seen as a retrieval task, and re-ranking strategy is
an effective post-processing method to improve performance in image retrieval. Therefore, re-ranking methods originally designed for
image-only retrieval have also been effectively applied to person reidentification. Methods such as contextual dissimilarity measure [29]
and average query expansion (AQE) [30] enhance retrieval performance by refining the initial ranking based on neighborhood relationships, feature averaging, or decision boundary distances. In the context
of person Re-ID, recent approaches, including the analysis of neighbor
information [31], and orientation-based re-ranking [32], further optimize rankings. Additionally, methods incorporating graph-based models [33], k-reciprocal encoding [34], and expanded cross-neighborhood
re-ranking [35] have shown promise in improving ranking accuracy.
Despite their success in the image-only setting, these methods typically rely on two implicit assumptions: (1) the query and the gallery
share the same feature distribution, and (2) nearest-neighbor relations
are reliable across the metric space, so that local neighborhoods can
be safely exploited for averaging or graph-based diffusion. In TI-ReID,
these assumptions no longer hold. The query is encoded as a text
embedding while the gallery consists of image embeddings from a VLM.
Even after joint training, their distributions on the unit hypersphere
remain misaligned, and text features tend to collapse into compact
clusters where multiple identities may overlap. As a result, the nearest
neighbors of a text query can contain many cross-identity images,
making conventional re-ranking operations unreliable: neighborhoodbased averaging may drift toward mismatched semantics, and graph
propagation can spread cross-modal errors instead of correcting them.
We therefore ground refinement in the more stable identity-level
structure of the image modality rather than in cross-modal neighborhoods around the text query. Specifically, we summarize gallery
samples into identity-level prototypes and refine QG-Similarity by incorporating QP-Similarity as an identity-level prior. Unlike post-hoc
refinement methods that directly revise pairwise similarities or ranking
lists, our framework performs prototype-mediated refinement that is
better aligned with the identity-aware retrieval objective of TI-ReID.

3.3. PDE: Prototype-Driven Enhancement
PDE enhances text-image retrieval by introducing identity-level prototypes constructed from the image gallery. We first perform unsupervised clustering on the gallery features {𝑓𝑉 (𝑔𝑗 )} and obtain 𝑁
clusters. We denote by 𝐶𝑘 the set of gallery images assigned to the 𝑘th
cluster. The prototype of cluster 𝐶𝑘 , which approximates an identitylevel center in the visual feature space, is computed as the mean feature
of its members:
1 ∑
𝑃𝑘 =
𝑓 (𝑔 ).
(2)
|𝐶𝑘 | 𝑔 ∈𝐶 𝑉 𝑗
𝑗

3. Method
3.1. Problem formulation
TI-ReID seeks to match individuals in an image gallery based on
𝑁𝑔
textual descriptions. Let 𝑞𝑖 represent a text query and  = {𝑔𝑗 , 𝑦𝑗 }𝑗=1
denote the gallery of images, where 𝑔𝑗 denotes an image, 𝑦𝑗 its identity
label, and 𝑁𝑔 the number of gallery images. The goal is to compute
the similarity between the text query and each image in the gallery
using features extracted from a pretrained VLMs, with query and image
features denoted as 𝑓𝑇 (𝑞𝑖 ) ∈ R𝑑 and 𝑓𝑉 (𝑔𝑗 ) ∈ R𝑑 , respectively. The
matching is performed using cosine similarity:
𝑆𝑞𝑔 (𝑖, 𝑗) = sim(𝑓𝑇 (𝑞𝑖 ), 𝑓𝑉 (𝑔𝑗 )),

𝑘

Given a text query 𝑞𝑖 , we then compute the QP-Similarity between
𝑞𝑖 and each prototype 𝑃𝑘 using cosine similarity:
(
)
𝑆𝑞𝑝 (𝑖, 𝑘) = sim 𝑓𝑇 (𝑞𝑖 ), 𝑃𝑘 .
(3)
For a gallery image 𝑔𝑗 that belongs to cluster 𝐶𝑘 , the final similarity between 𝑞𝑖 and 𝑔𝑗 is obtained by combining the instance-level
QG-Similarity in Eq. (1) with the identity-level QP-Similarity:
̂
𝑆(𝑖, 𝑗) = 𝑆𝑞𝑔 (𝑖, 𝑗) + 𝜆 𝑆𝑞𝑝 (𝑖, 𝑘),

(4)

where 𝜆 ≥ 0 is a global hyperparameter that controls the strength of
the prototype prior.
3.4. PDE𝑀 : Cascading Prototype-Driven Enhancement

(1)

A single clustering granularity and a fixed fusion weight can be
insufficiently robust because the gallery exhibits structure at multiple
scales. Small neighborhoods provide high-resolution cues that help
correct instance-level mismatches, whereas broader consensus offers
stability against noisy local clusters. To exploit both effects, we extend
PDE into a multi-stage inference scheme, denoted PDE𝑀 , that applies
a sequence of prototype-based refinements with gradually weakened
influence.
Let 𝑓𝑉(1) (𝑔𝑗 ) denote the initial image feature of gallery image 𝑔𝑗
used in the single-stage PDE. At refinement stage 𝑡 ∈ {1, … , 𝑀 − 1},
we recompute identity-level prototypes from the current features 𝑓𝑉(𝑡) .
Specifically, we perform unsupervised clustering on {𝑓𝑉(𝑡) (𝑔𝑗 )} to obtain
𝑁 (𝑡) clusters, and compute stage-wise prototypes {𝑃𝑘(𝑡) } using the same

Collecting the cosine similarities between the query and all gallery
images yields a matrix 𝑆𝑞𝑔 , referred to as the QG-Similarity, which
serves as the baseline text–image similarity throughout this paper.
However, due to the modality gap, such instance-level alignment in the
VLM feature space remains fragile. To mitigate this issue, we further
construct identity-level prototypes from the gallery, which summarize
the underlying identity structure in the image modality and provide
more reliable references for robust cross-modal retrieval.
3.2. Overview
To address the large modality gap and the instance–identity mismatch in TI-ReID, we build a training-free inference pipeline that
4

Pattern Recognition 179 (2026) 113705

H. Yang et al.

averaging rule as in Eq. (2). The number of prototypes follows a
geometric schedule
𝑁 (𝑡) = 𝑁 ⋅ 𝛼 𝑡−1 ,

𝛼 ∈ (0, 1),

To ensure robustness and remove noisy peaks, we filter out those
with densities below a predefined percentile threshold 𝑇 , typically set
to the 75th percentile, and the final number of clusters 𝑁 is determined
by the number of remaining peaks:
(
) }|
|{
|
|
|
𝑁 = | 𝑧𝑖 ∈  || 𝜌(𝑧𝑖 ) > Percentile𝑇 {𝜌(𝑧𝑗 )}𝑛𝑗=1
(13)
|,
|
|
|
|
|
This adaptively estimated 𝑁 is then seamlessly integrated into the
identity prototype construction process of both PDE and PDE𝑀 . In
PDE𝑀 , the estimated cluster count 𝑁 serves as the initialization for
the clustering procedure, and the prototypes are progressively refined
through iterative updates using decay coefficients 𝛼 and 𝛽, which
control the granularity of prototypes while maintaining their semantic
integrity. This ensures that the model can flexibly adapt to different
datasets without requiring manual tuning.
In practice, ACE is run once per dataset as an offline pre-processing
step on the gallery features. Its computational cost is modest, typically
on the order of tens of seconds. Empirically, the estimated cluster
number 𝑁 is stable across random seeds, and the procedure operates
in a fully unsupervised manner on unlabeled gallery features. For
simplicity we fix the bandwidth ℎ in our implementation, but ACE can
readily incorporate rule-based bandwidth selection (e.g., Silverman’s
rule) or lightweight validation-based tuning if desired.

(5)

so that early stages use more fine-grained prototypes, while later stages
rely on fewer, more aggregated identity centers.
Given a text query 𝑞𝑖 , we compute the stage-wise QP-Similarity
between 𝑞𝑖 and each prototype 𝑃𝑘(𝑡) as
(
)
(𝑡)
𝑆𝑞𝑝
(𝑖, 𝑘) = sim 𝑓𝑇 (𝑞𝑖 ), 𝑃𝑘(𝑡) .
(6)
For a gallery image 𝑔𝑗 assigned to cluster 𝐶𝑘(𝑡) at stage 𝑡, its feature is
updated via a residual enhancement:
𝑓𝑉(𝑡+1) (𝑔𝑗 ) = 𝑓𝑉(𝑡) (𝑔𝑗 ) + 𝜆(𝑡) 𝑃𝑘(𝑡) ,

(7)

where 𝜆(𝑡) is a stage-dependent scaling factor that controls the strength
of the prototype prior. We use a decaying schedule
𝜆(𝑡) = 𝜆 ⋅ 𝛽 𝑡−1 ,

𝛽 ∈ (0, 1),

(8)

with 𝜆 the initial fusion weight. This schedule lets early stages apply stronger corrections based on fine-grained prototypes, while later
stages make more conservative adjustments guided by coarser identity
structure.
After 𝑀 stages, the refined gallery features 𝑓𝑉(𝑀) are used to compute the final text–image similarity with the query features 𝑓𝑇 (𝑞𝑖 ). For
clarity, the resulting score can be equivalently written as
𝑆̂ = 𝑆𝑞𝑔 +

𝑀−1
∑

𝜆

(𝑡)

(𝑡)
𝑆𝑞𝑝
,

Algorithm 1: ACE: Adaptive Cluster Number Estimation
Input: Feature set 𝐗 ∈ R𝑛×𝑑 , bandwidth ℎ, percentile threshold
𝑇
Output: Estimated cluster number 𝑁
1 Feature Projection:
2
Apply t-SNE to project 𝐗 into 2D space 𝐙 = t-SNE(𝐗);
3 KDE-Based Density Estimation:
4
For each point 𝑧𝑖 ∈ 𝐙, compute density using Gaussian
kernel:
(
)
∑
‖𝑧 −𝑧 ‖2
5
𝜌(𝑧𝑖 ) = 𝑛ℎ12 𝑛𝑗=1 exp − 𝑖2ℎ2𝑗
;

(9)

𝑡=1
(𝑡)
where 𝑆𝑞𝑔 is the baseline QG-Similarity in Eq. (1) and 𝑆𝑞𝑝
denotes the
QP-Similarity computed at stage 𝑡. Note that Eq. (9) shows that PDE𝑀
adds a weighted combination of stage-wise QP-Similarities on top of
the baseline similarity. In practice we implement PDE𝑀 via the feature
refinement in Eq. (7), as illustrated in Fig. 3.
By progressively reducing the number of prototypes through 𝛼 and
decaying their influence through 𝛽, PDE𝑀 improves retrieval robustness: early stages correct prominent mismatches using localized identity
cues, and later stages stabilize rankings under broader identity consensus, which is particularly beneficial on large-scale or distributionshifted datasets.

Peak Detection:
Identify local maxima of 𝜌(𝑧) as candidate cluster centers:
8
 = {𝑧𝑖 ∣ 𝜌(𝑧𝑖 ) > 𝜌(𝑧𝑗 ), ∀𝑧𝑗 ∈  (𝑧𝑖 )};
9 Percentile-Based Filtering:
10
Retain only peaks with density above the 𝑇 -th percentile of
all values:
11
 ′ = {𝑧𝑖 ∈  ∣ 𝜌(𝑧𝑖 ) > Percentile𝑇 ({𝜌(𝑧𝑗 )}𝑛𝑗=1 )};
6
7

3.5. ACE: Adaptive Cluster Number Estimation

Return:
𝑁 = | ′ |;
14
return 𝑁

In TI-ReID galleries, multiple images usually exist for each identity,
and their embeddings tend to form locally concentrated regions, while
different identities are often more separated in the feature space. Under
this common structure, local density maxima can be viewed as robust
representatives of dominant identity-related modes. Accordingly, ACE
estimates the prototype granularity by identifying prominent density
peaks in the embedded feature space. To achieve this, we first project
the high-dimensional gallery features 𝐗 = {𝑥𝑖 ∈ R𝑑 }𝑛𝑖=1 into a twodimensional space using t-distributed stochastic neighbor embedding
(t-SNE), which helps mitigate the curse of dimensionality:

3.6. Analysis of prototype-based ranking robustness

𝐙 = t-SNE(𝐗) ∈ R𝑛×2 ,

𝑆̂𝑖𝑗 + > 𝑆̂𝑖𝑘− ,

12
13

To analyze how identity prototypes influence ranking robustness,
consider a query sample 𝑞𝑖 and two gallery samples: 𝑔𝑗 + , a positive sample, and 𝑔𝑘− , a negative sample. A desirable TI-ReID ranking satisfies

(10)

(14)

Expanding Eq. (14) with Eq. (9) gives

Next, we estimate the density distribution of the embedded points
using kernel density estimation (KDE) [36]. This step provides a continuous density function 𝜌(𝑧), which helps us capture regions of high
density that are likely to correspond to the latent identity centers:
(
)
𝑛
‖𝑧 − 𝑧𝑖 ‖2
1 ∑
𝜌(𝑧) =
exp −
,
(11)
𝑛ℎ2 𝑖=1
2ℎ2

𝑆𝑞𝑔 (𝑖𝑗 + ) − 𝑆𝑞𝑔 (𝑖𝑘− ) >

𝑀−1
∑

(𝑖)
(𝑖)
𝜆(𝑖) (𝑆𝑞𝑝
(𝑖𝑘− ) − 𝑆𝑞𝑝
(𝑖𝑗 + )),

(15)

𝑖=1

where the right-hand side aggregates prototype-based corrections
across the refinement stages. We qualitatively discuss two typical cases.
Case 1: Correct initial ranking. If 𝑆𝑞𝑔 (𝑖𝑗 + ) > 𝑆𝑞𝑔 (𝑖𝑘− ), the prototype term usually enlarges the margin. When the prototypes are close
(𝑡)
(𝑡)
to the identity centers, we have 𝑆𝑞𝑝
(𝑖𝑗 + ) ≥ 𝑆𝑞𝑝
(𝑖𝑘− ) for most stages 𝑡,
making the right-hand side non-positive. This makes Eq. (15) easier to
satisfy and strengthens an already correct decision, which contributes
to more stable rankings.

where ℎ controls the smoothness of the density estimate. The local
maxima of this density function represent the potential centers of the
identities, which can be identified by:
{
}
|
 = 𝑧𝑖 || 𝜌(𝑧𝑖 ) > 𝜌(𝑧𝑗 ), ∀𝑧𝑗 ∈  (𝑧𝑖 ) ,
(12)
|
5

Pattern Recognition 179 (2026) 113705

H. Yang et al.

Fig. 4. Comparison of Cross-Modal Feature Distances for T2I and T2P on three datasets.

Case 2: Incorrect initial ranking. If 𝑆𝑞𝑔 (𝑖𝑗 + ) < 𝑆𝑞𝑔 (𝑖𝑘− ), the prototype term provides a corrective signal. Since prototypes summarize
the majority of samples within each cluster, it typically holds that
(𝑡)
(𝑡)
𝑆𝑞𝑝
(𝑖𝑗 + ) > 𝑆𝑞𝑝
(𝑖𝑘− ) when there are sufficient intra-identity instances.
In this case, a suitable choice of 𝜆(𝑡) can flip the inequality in Eq. (15)
and rectify a mis-ranked pair.
These considerations suggest that identity prototypes can enhance
ranking robustness in two complementary ways: (1) amplifying margins
for initially correct rankings and (2) offering a chance to correct
misaligned ones, especially when the prototype prior is reliable.
Beyond the above analytical discussion, we also provide empirical
evidence that identity-level prototypes offer a stable cue that is well
suited for inference-time refinement, but not necessarily as a standalone
replacement of instance-level retrieval. Using IRRA [23] as the baseline,
we compare Text-to-Image (T2I) and Text-to-Prototype (T2P) similarities via cosine-distance statistics. Let 𝑥 denote a text feature and 𝑦
denote either an image feature or a prototype feature. We define the
intra-identity and inter-identity distances as
[
]
(
)|
𝐷intra = E 1 − cos 𝑓𝑇 (𝑞), 𝑦 | ID(𝑞) = ID(𝑦) ,
|
(16)
[
]
(
)|
𝐷inter = E 1 − cos 𝑓𝑇 (𝑞), 𝑦 | ID(𝑞) ≠ ID(𝑦) .
|

CUHK-PEDES is the first dataset dedicated to text-based person
retrieval, containing 40,206 images and 80,412 textual descriptions
of 13,003 identities. The dataset is split into a training set (11,003
identities, 34,054 images and 68,108 descriptions), a validation set
(3078 images, 6158 descriptions), and a test set (3074 images, 6156
descriptions), each containing 1000 identities.
ICFG-PEDES consists of 54,522 images of 4102 identities, with each
image corresponding to a single textual description. The dataset is split
into a training set of 34,674 text-image pairs and a test set of 19,848
pairs for 1000 identities.
RSTPReid contains 20,505 images of 4101 identities across 15
cameras, with each image annotated with two textual descriptions. The
dataset is divided into a training set (3701 identities), a validation set
(200 identities), and a test set (200 identities).
Evaluation Metrics. We use several standard evaluation metrics:
Cumulative Matching Characteristic (CMC) [39] and mean Average
Precision (mAP) [40]. Higher Rank-𝑘 and mAP values indicate better
retrieval performance.
4.2. Implementation details
Our framework builds on the PDE module and can be extended by
stacking multiple PDE units. We evaluate two configurations, PDE with
a single module and PDE𝑀 with 𝑀 cascading modules, and integrate
both into CLIP-based and ALBEF-based TI-ReID backbones. Identitylevel prototypes are obtained by unsupervised K-means [41] on gallery
features, which is parameter efficient and makes the cluster number 𝑁
the main control of prototype granularity. When ACE is enabled, 𝑁 is
selected automatically from the gallery distribution using KDE with a
percentile threshold 𝑇 = 0.75. For PDE we set 𝑁 = 1300, 𝑁 = 1600, and
𝑁 = 200 on CUHK-PEDES, ICFG-PEDES, and RSTPReid respectively,
with a fixed fusion weight 𝜆 = 0.25. For PDE𝑀 we keep these settings
and use 𝑀 = 4 with stage-wise decay 𝛽 = 0.8 and 𝛼 = 0.25. All
experiments are implemented in PyTorch and run on a single NVIDIA
A100 with 40 GB memory.

where 𝑦 = 𝑓𝑉 (𝑔) for T2I and 𝑦 = 𝑃𝑘 for T2P. The discrimination score
is
𝐷
− 𝐷intra
Disc = inter
.
(17)
𝐷inter + 𝐷intra
Since 1 − cos ∈ [0, 2], the score lies in [−1, 1]. As shown in Fig. 4,
T2P reduces 𝐷intra and increases 𝐷inter , yielding a higher Disc than
T2I. This stronger separation between matched and mismatched pairs
is consistent with the observation that T2P achieves higher mAP across
datasets. However, Fig. 5 shows that T2P does not always surpass
T2I in Rank-1 accuracy: prototype similarity is robust in a global,
distributional sense but may lose some fine-grained instance cues that
affect the very top-1 position. In addition, the trajectories in Figs. 5
and 6 indicate that T2P and T2I improve in tandem during training,
suggesting that prototypes capture complementary but not redundant
information.
These findings support our design choice in PDE: rather than replacing instance-level Text-to-Image similarity with a prototype-only
retrieval strategy, we treat Text-to-Prototype similarity as an auxiliary
prior and fuse it with T2I at inference time. In this way, PDE benefits
from the global ranking stability of prototypes while retaining the
fine-grained discrimination of instance-level similarities.

4.3. Comparison with state-of-the-art
In this section, we evaluate the effectiveness of our proposed
training-free framework by applying the PDE and PDE𝑀 to two representative backbones: CLIP and ALBEF. We demonstrate the improvements achieved by our methods on three benchmark datasets: CUHKPEDES, ICFG-PEDES, and RSTPReid. As shown in Table 1, our method
produces significant improvements, particularly on the ICFG-PEDES
dataset, which contains over 19 samples per class, significantly surpassing the 3 samples per class in CUHK-PEDES and the 5 samples per
class in RSTPReid.
For the ALBEF backbone, PDE and PDE𝑀 yield improvements of
1.1%/1.24% in Rank-1 accuracy and 3.72%/5.56% in mAP over the
RaSa model, and 0.99%/1.31% in Rank-1 and 3.38%/3.9% in mAP over
MARS. For the CLIP backbone, PDE and PDE𝑀 achieve 1.22%/1.36%
improvements in Rank-1 and 3.79%/5.77% in mAP over IRRA, and
0.68%/0.76% improvements in Rank-1 and 4.83%/5.42% in mAP over
HAM.

4. Experiment
4.1. Datasets and metrics
Our experiments are conducted on three widely used benchmark
datasets for Text-to-Image Person Re-identification: CUHK-PEDES [1],
ICFG-PEDES [37], and RSTPReid [38], which are all integral to evaluating TI-ReID methods.
6


exec
/bin/zsh -lc "pdftotext -f 8 -l 11 'A2HA - Attribute-aware hierarchical alignment for text-image person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Table 3
Effect of attribute-enriched datasets and hierarchical alignment on ViTAA over CUHK-PEDES.
Model

HA

Img Attr

Text Attr

ViTAA [11]
ViTAA
ViTAA
ViTAA

×
×
✓
✓

ViTAA
A2HA
ViTAA
A2HA

ViTAA
A2HA
ViTAA
A2HA

CUHK-PEDES [17]
R-1

R-5

R-10

mAP

54.92
59.31(+4.39)
60.62(+5.70)
66.41(+11.49)

68.79
73.29(+4.50)
74.13(+5.34)
79.65(+10.86)

75.78
80.99(+5.21)
81.01(+5.23)
86.08(+10.30)

51.60
55.90(+4.30)
56.47(+4.87)
60.90(+9.30)

Values in parentheses indicate performance gains.

Table 4
Comparison of A2HA with other SOTA methods on CUHK-PEDES, ICFG-PEDES and
RSTPReid on the mINP.
Method

Source

CUHK-PEDES [17]

ICFG-PEDES [18]

RSTPReid [19]

IRRA [36]
TBPS-CLIP [46]
RDE [49]
IRRA+A2HA(Ours)

CVPR’23
AAAI’24
CVPR’24
–

50.24
52.58
51.44
51.91

8.96
–
7.87
9.81

34.45
–
28.08
35.39

The best results are in bold and the second best are underlined.

Table 5
Ablation study on the effectiveness of the losses in attribute-level alignment.
ALA
#

M1
M2
M3
M4
M5
M6
M7

CUHK-PEDES [17]

ICFG-PEDES [18]

R-1

mAP

R-1

mAP

Baseline

73.38

66.13

63.46

38.06

✓

76.97
77.98
75.72
82.87
81.64
81.79
83.86

68.17
68.43
67.36
70.17
69.00
69.83
70.25

64.89
63.92
63.87
66.26
65.69
66.90
67.97

39.50
38.95
39.11
40.89
40.42
40.55
41.87

𝑖𝑛𝑡𝑒𝑟

𝑖𝑛𝑡𝑟𝑎

𝑡𝑟𝑖

✓
✓
✓
✓
✓

✓
✓
✓

✓
✓
✓

Table 6
Ablation study on the effectiveness of the component of hierarchical alignment.
ALA
M1
M2
M3
M4
M5
M6
M7

GLA
✓
✓
✓
✓
✓
✓
✓

CUHK-PEDES [17]

ICFG-PEDES [18]

R-1

mAP

R-1

mAP

80.83
81.04
80.98
83.87
83.03
83.55
84.06

69.40
69.97
69.45
71.01
70.78
70.59
71.15

65.92
64.97
65.58
68.01
66.09
67.79
69.76

40.08
39.45
39.91
41.95
41.02
41.20
42.74

The best results are in bold and the second best are underlined.

The best results are in bold and the second best are underlined.

4.4. Ablation studies

Results of the combination of ALA: In ALA, we evaluate three individual losses (𝑖𝑛𝑡𝑒𝑟 , 𝑖𝑛𝑡𝑟𝑎 , 𝑡𝑟𝑖 ) and their combinations, with detailed
findings reported in Table 5. The standalone use of each loss yields
modest improvements, while their combination significantly enhances
performance by deepening the model’s ability to distinguish positive
and negative attribute pairs. Notably, 𝑖𝑛𝑡𝑒𝑟 and 𝑖𝑛𝑡𝑟𝑎 are critical, as
they enable the model to prioritize key samples through hard sample mining and dynamic weight adjustment, thereby boosting overall
performance. When 𝑖𝑛𝑡𝑒𝑟 and 𝑖𝑛𝑡𝑟𝑎 are jointly utilized, the model exhibits the strongest efficacy in intra-modal and inter-modal contrastive
learning among multiple attributes partitioned from instances of the
same identity. Specifically, their synergistic application yields superior
performance compared to M5 and M6, which only employ a single loss
function.
Results of the addition of GLA: Table 6 illustrates the optimization
effect of hierarchical alignment (HA) when incorporating global-level
alignment (GLA). Building upon the results in Table 5, the addition of
global alignment enables the model to achieve effective alignment at
both coarse-grained and fine-grained levels. The performance trends
in Table 6 are generally consistent with those in Table 5. M4, which
concurrently applies 𝑖𝑛𝑡𝑒𝑟 , 𝑖𝑛𝑡𝑟𝑎 and GLA, achieves suboptimal performance. The optimal result is attained after integrating the attributelevel triplet loss 𝑡𝑟𝑖 . Furthermore, each method exhibits measurable
improvements in their corresponding rows of Table 6, demonstrating that the incorporation of global multi-level optimization yields
substantial benefits for alignment performance.

4.4.1. Effectiveness of different alignment methods
We conduct ablation studies to validate the effectiveness of key
alignment components in A2HA: global-level alignment (GLA) with 𝑖𝑛𝑠
and 𝑖𝑑 , and attribute-level alignment (ALA) with 𝑎𝑡𝑡𝑟 . Results for
these analyses are presented in Tables 5 and 6.

4.4.2. Effectiveness of values of hyper-parameters
We evaluate the effects of 𝜆𝑖𝑛𝑠 , 𝜆𝑡𝑟𝑖 , 𝜖, 𝑟, 𝜃𝑝𝑜𝑠 , 𝜃𝑛𝑒𝑔 , and the confidence
thresholds for the image segmentation on model performance. Through
a series of hyperparameter sensitivity analyses, we demonstrate the
effectiveness of the hyper-parameters and the rationality of their chosen

relatively modest. Nevertheless, it still achieves state-of-the-art performance and outperforms attribute alignment-based approaches such as
ViTAA [11] and APTM [13].
On RSTPReid [19], our A2HA achieves best or comparable results:
while slightly trailing the most recent work [50] in Rank-1, it excels
in all other metrics, with mAP outperforming SOTA by +0.96% and
demonstrating significant gains over the baseline IRRA [46], with the
improvement of +10.45% and +8.98% on Rank-1 and mAP. The
RSTPReid dataset exhibits a smaller scale. However, our method still
achieves exceptional performance. This indicates that A2HA is capable
of effectively operating under limited data.
4.3.2. Comparison on auxiliary metric mINP
Our A2HA demonstrates superior performance in terms of the mINP
metric. As shown in Table 4, across the CUHK-PEDES, ICFG-PEDES, and
RSTPReid datasets, it shows strong competitiveness. On CUHK-PEDES,
with a score of 51.91%, it outperforms IRRA [46] and RDE [49],
though slightly trailing TBPS-CLIP [46]. On ICFG-PEDES, its result
surpasses IRRA and RDE. On RSTPReid, A2HA achieves 35.39%, it
beats IRRA and RDE. Overall, our A2HA maintains stable and competitive performance across datasets, with notable advantages on ICFGPEDES and RSTPReid, effectively validating its good adaptability and
recognition capability in text–image person re-identification tasks.

8

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Fig. 4. Ablation studies on CUHK-PEDES dataset. (a) Ablation studies on values of 𝜆𝑖𝑛𝑠 and 𝜆𝑡𝑟𝑖 . (b) Ablation studies on values of 𝜖 and 𝑟.

balances these complementary information sources, resulting in the
highest model performance.
(3) Ablation on the Values of 𝜃𝑝𝑜𝑠 and 𝜃𝑛𝑒𝑔 .
For 𝜃𝑝𝑜𝑠 and 𝜃𝑛𝑒𝑔 , we adhere to the principle that 𝜃𝑝𝑜𝑠 > 𝜃𝑛𝑒𝑔 , since
positive attribute pairs inherently exhibit higher similarity than negative ones. As shown in Fig. 5, if the two values are too close, positive
and negative pairs become indistinguishable and cannot be effectively
differentiated. Conversely, an excessively large 𝜃𝑝𝑜𝑠 combined with an
overly small 𝜃𝑛𝑒𝑔 results in significant information loss, which harms
model optimization. After careful consideration, we set the optimal
values to 𝜃𝑝𝑜𝑠 = 0.5 and 𝜃𝑛𝑒𝑔 = 0.3. To determine the optimal threshold
values, we conduct extensive experiments with different combinations.
The experimental results reveal that the performance is sensitive to
the choice of these thresholds. Specifically, when 𝜃𝑛𝑒𝑔 = 0.3, the
model achieves the best overall mAP across different 𝜃𝑝𝑜𝑠 settings, with
𝜃𝑝𝑜𝑠 = 0.5 yielding the peak performance of approximately 71.3%.
This observation is consistent with the distribution characteristics of
attribute similarity scores in our feature space: positive pairs typically
concentrate in higher similarity regions (> 0.5), while negative pairs
tend to distribute in lower similarity ranges (< 0.3). The gap between
𝜃𝑝𝑜𝑠 = 0.5 and 𝜃𝑛𝑒𝑔 = 0.3 provides a reasonable margin that effectively
separates the majority of positive and negative samples while preserving sufficient informative samples for robust model training. Therefore,
we set the optimal values to 𝜃𝑝𝑜𝑠 = 0.5 and 𝜃𝑛𝑒𝑔 = 0.3.
(4) Analysis of Confidence Threshold Selection for the Image Segmentation Model.
The selection of the confidence thresholds for the image segmentation model is primarily based on the distribution of target confidence
scores obtained during the detection stage of the Grounded SAM model.
Specifically, we collect the confidence scores for each attribute after
processing the entire dataset with the segmentation model, and visualize the distribution using scatter plots. Due to the large volume of
data, we randomly sample 10,000 images for visualization. To ensure
high segmentation accuracy, for attributes that are difficult to detect
(e.g., Bag, whose confidence scores are generally lower than those of
other attributes), we select a threshold slightly below the median value.
As illustrated in Fig. 6, the median confidence for Bag is 0.18, and we
set the threshold to 0.1. For attributes that are relatively easy to detect,
the threshold is determined based on the density of the distribution.
Specifically, we choose the boundary between sparse and dense regions
in the distribution as the confidence threshold, which is set to 0.3.

Fig. 5. Ablation study on different values of 𝜃𝑝𝑜𝑠 and 𝜃𝑛𝑒𝑔 on CUHKPEDES [17]. 𝜃𝑝𝑜𝑠 and 𝜃𝑛𝑒𝑔 satisfy 𝜃𝑝𝑜𝑠 > 𝜃𝑛𝑒𝑔 .

values. All of the experiments are conducted on the CUHK-PEDES
dataset [17].
(1) Ablation on the Values of 𝜆𝑖𝑛𝑠 and 𝜆𝑡𝑟𝑖 .
As illustrated in Fig. 4(a), varying values of 𝜆𝑖𝑛𝑠 and 𝜆𝑡𝑟𝑖 result in
slight performance differences. Since 𝜆𝑖𝑛𝑠 regulates the contribution of
global information in the alignment. Excessively large values prioritize
global features, thereby regressing the model to the alignment mode
shown in Fig. 1(a) and weakening the attribute-driven fine-grained
alignment. This explains why mAP fluctuates more significantly with
𝜆𝑖𝑛𝑠 variations compared to 𝜆𝑡𝑟𝑖 . In contrast, 𝜆𝑡𝑟𝑖 controls the intensity
of the attribute triplet loss 𝑡𝑟𝑖 , which refines attribute-level alignment
by addressing residual false positive pairs and false negatives pairs
after hard sample mining. The relatively stable mAP values suggest that
𝜆𝑡𝑟𝑖 primarily fine-tunes rather than fundamentally alters the alignment
mechanism.
(2) Ablation on the Values of 𝜖 and 𝑟.
For the triplet margin 𝜖 and fusion parameter 𝑟, we conduct experiments across six distinct values within the range of 0 to 1, as
constrained by Eqs. (9) and (14). The triplet margin 𝜖 refines the distance threshold between positive and negative sample pairs to enforce
tighter intra-class clustering, where excessively small values (< 0.1)
cause premature convergence while overly large margins (> 0.5) increase optimization difficulty and risk of overfitting to noisy samples.
Experimental results in Fig. 4(b) demonstrate optimal performance
at 𝜖 = 0.2, which balances discriminative power and training stability. Regarding the fusion parameter 𝑟, which controls the integration ratio between global visual features and local attribute features.
Setting 𝑟 = 1 completely suppress global information, leading to
contextual loss, whereas 𝑟 = 0 discard segmented attribute details
and fail to achieve the attribute-driven alignment objective of the
A2HA framework. Our empirical analysis reveal that 𝑟 = 0.8 optimally

4.4.3. Effectiveness of attribute pairing strategy
Since we divide positive and negative sample pairs at the attribute
level, while general contrastive learning typically divides them at
the instance or identity level, we conduct ablation experiments to
verify that attribute-level division strengthens cross-modal alignment
constraints and enhances performance. We compare instance-level,
identity-level, and attribute-level division methods, while keeping other
9

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Fig. 6. Scatter plot of target attribute confidence score distribution after segmentation model processing.
Table 7
Ablation study on the sample pair classification strategies.

Table 9
Ablation study on the effectiveness of the add of self-attention pooling.

CUHK-PEDES [17]

Classification strategy
Baseline
Instance Level
Identity Level
Attribute Level

R-5

R-10

mAP

73.38
82.13
82.69
84.06

89.93
92.23
93.37
95.68

93.71
94.61
94.57
97.87

66.13
69.43
69.98
71.15

✓
×

Table 8
Ablation study on the effectiveness of the dynamic weight adjustment.

✓
×

R-1

R-5

R-10

mAP

84.06
83.01

95.54
95.48

97.84
97.64

71.08
70.36

Table 10
Attribute-Level text–image matching accuracy on the proposed attributeenriched datasets.

The best results are in bold and the second best are underlined.

𝑤𝑖𝑛𝑡𝑒𝑟 &𝑤𝑖𝑛𝑡𝑟𝑎

CUHK-PEDES [17]

Self-attention pooling

R-1

CUHK-PEDES [17]
R-1

R-5

R-10

mAP

84.06
82.91

95.54
94.74

97.84
96.60

71.08
69.89

Accuracy (%)

CUHK-PEDES [17]

ICFG-PEDES [18]

RSTPReid [19]

Hair (%)
Upper Body (%)
Lower Body (%)
Bag (%)
Shoes (%)

79.91
100
100
92.99
79.51

72.14
100
100
82.96
90.16

66.39
100
100
89.24
84.03

provide reliable detection confidence scores alongside precise and visually coherent segmentation masks. Meanwhile, the text attribute
parsing preserves essential semantic content, ensuring that fine-grained
descriptive information is accurately retained rather than fragmented.
The last four rows of Tables 2 and 3 clearly show that incorporating
our attribute-enriched datasets leads to consistent and significant performance gains. This observation indicates that the introduced attribute
annotations effectively enhance cross-modal alignment and provide
more informative supervision. Furthermore, we evaluate attribute-level
matching accuracy for each benchmark, as reported in Table 10. This
metric reflects the proportion of correct matches between textual and
visual attribute extractions across three datasets. The consistently high
accuracy achieved on all three benchmarks confirms the robustness
and reliability of our attribute segmentation strategy. For intra-modal
accuracy, we conduct manual verification by randomly selecting 500
identities and assessing the correctness of their visual and textual
attribute annotations. The results are reported in Table 11. Lastly,
although large-scale models are employed for both visual and textual
attribute segmentation, this procedure is conducted entirely offline
and is decoupled from the training process. As a result, it introduces
no additional computational burden during training or inference and
does not increase the model parameter count, as evidenced in Table 12. This design ensures that the performance improvements stem
from higher-quality attribute supervision rather than increased model
complexity.

loss optimization settings unchanged. As shown in Table 7, instancelevel and identity-level divisions promote alignment but yield inferior
results compared to the attribute-level approach, thus confirming the
effectiveness of this robust alignment method.
4.4.4. Effectiveness of the tricks
Ablation experiments on the CUHK-PEDES dataset, as presented
in Table 8 and 9, systematically validate the performance-enhancing
contributions of two critical components. Table 8 investigates the efficacy of dynamic weight adjustment for 𝑤𝑖𝑛𝑡𝑒𝑟 and 𝑤𝑖𝑛𝑡𝑟𝑎 . Activation of this dynamic weight mechanism yields consistent improvements across core retrieval metric Rank-1 and mean average precision (mAP). Table 9 quantifies the impact of integrating self-attention
pooling. When enabled, this module facilitates measurable gains. Collectively, these results underscore that dynamic weight adjustment
imposes fine-grained control over inter-modal and intra-modal attribute
alignment, while self-attention pooling enriches feature representation fidelity. Both mechanisms synergistically enhance the model’s discriminative capacity for cross-modal retrieval, as evidenced by robust
improvements in Rank-𝑘 and mAP metrics.
4.4.5. Evaluation on the attribute-enriched datasets
To rigorously assess the quality of the proposed attribute-enriched
datasets, we conduct both qualitative and quantitative evaluations.
From a qualitative perspective, Fig. 7 demonstrates that our datasets
10

Pattern Recognition 180 (2026) 114100

Z. Zhu et al.

Fig. 7. Segment results from Grounded SAM and ChatGPT, from left to right are the raw image, the detection confidence results and segmentation results for
‘bag’, the detection confidence results and segmentation results for ‘hair’, the detection confidence results and segmentation results for ‘upper & lower body’, the
detection confidence results and segmentation results for ‘shoes’, as well as the corresponding text attribute segmentation results.
Table 11
Intra-modal attribute annotation accuracy via manual verification on CUHK-PEDES, ICFG-PEDES, and RSTPReid
datasets.
Dataset

Modality

Hair (%)

Upper body (%)

Lower body (%)

Bag (%)

Shoes (%)

Gender (%)

CUHK-PEDES [17]

Image
Text

94.11
97.26

97.21
92.84

92.23
94.41

87.40
93.39

88.98
99.20

–
100

ICFG-PEDES [18]

Image
Text

93.41
95.09

98.82
93.41

94.26
94.35

89.36
90.13

88.34
98.82

–
100

RSTPReid [19]

Image
Text

95.76
97.35

97.20
95.73

95.14
95.02

90.00
92.90

89.57
91.89

–
100

Table 12
Model parameter count and computational complexity comparison.
Para

CUHK-PEDES [17]

ICFG-PEDES [18]

RSTPReid [19]

will open-source our extended attribute-enriched datasets and provide
their generation code in the future. We believe this will be beneficial
to attribute-related research in TI-ReID.

IRRA [36]
A2HA

194M
194M

191M
191M

367M
367M

5.2. The proposed hierarchical alignment
The proposed hierarchical alignment is motivated by the observation that most existing TI-ReID methods rely on implicit global
alignment, while local or attribute-level cues are either weakly utilized
or treated as auxiliary information. In contrast, A2HA formulates crossmodal alignment as a structured, multi-level optimization problem,
where instance-level, identity-level, and attribute-level objectives are
explicitly integrated within a unified framework. A key innovation lies
in elevating attribute alignment from an auxiliary role to a core optimization target. By explicitly modeling attribute-level correspondence
and misalignment through structured sample pairing, the proposed
hierarchical alignment enables fine-grained semantic discrimination
that cannot be achieved by global alignment alone. This design allows
global identity consistency and local attribute correspondence to be
jointly optimized in a coarse-to-fine manner, resulting in more robust
and interpretable cross-modal representations.

4.4.6. Qualitative visualization
Fig. 8 presents the top 5 retrieval results for query images. We highlight the correctly retrieved results with red bounding boxes for clearer
visualization and organize the layout by grouping high-accuracy and
low-accuracy retrieval examples. The visualization demonstrates that
our A2HA achieves superior retrieval performance, with accurate hits
consistently appearing among the top 5 results. While the model shows
slight limitations in distinguishing cases involving identical clothing
items, it delivers highly accurate retrieval outcomes overall.
5. Discussion
5.1. The proposed attribute-enriched datasets
Our approach of using Grounded SAM for image attribute segmentation and ChatGPT for text attribute division is effective and highly
accurate. Due to the rapid development of current pre-trained large
models, there are numerous large models available to assist research.
These large models not only ensure high accuracy but also apply
to various downstream tasks. Compared with traditional segmentation networks and word segmentation networks, using large models
can guarantee training accuracy without increasing the number of
additional training parameters. For instance, large language models
correct simple spelling errors (e.g., ‘co lor’ to ‘color’), thereby mitigating the impact of text inaccuracies on semantic understanding. We

5.3. Limitation and future work
Although our A2HA has been proven effective, there is still room
for improvement. Considering that some attributes are interrelated, the
current processing method may not fully capture such correlations.
For example, the relationship between ‘gender’ and ‘hair’. While such
correlations are not dominant, they still provide a starting point to
explore more robust attribute splitting methods. As illustrated in the
right column of Fig. 8, queries for a man may return images of a
woman due to similarities in their clothing. In our future work, we
will manage to model attribute dependencies by introducing techniques
11


exec
/bin/zsh -lc "pdftotext -f 8 -l 11 'A training-free framework for text-to-image person re-identification via query-prototype matching.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113705

H. Yang et al.

Table 2
Comparisons with the state-of-the-art Re-ID methods.
Method
Han et al. [2]
TBPS-CLIP [42]
CFAM [19]
UMSA [43]
LSPM [44]
IRLT [45]
MDRL [46]
FSRL [47]
Propot [26]
SAP-SAM [48]
PLOT [27]
RDE [24]
APTM [21]
RaSa [20]
IRRA [23]
MARS [22]
HAM [28]
MARS-PDE𝑀
HAM-PDE𝑀

Image encoder
CLIP-RN101
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
Swin-B
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT
CLIP-ViT

CUHK-PEDES

ICFG-PEDES

RSTPReid

R1

mAP

R1

mAP

R1

mAP

64.08
73.54
73.67
73.67
74.38
74.46
74.56
74.65
74.89
75.05
75.28
75.94
76.53
76.51
73.38
77.62
77.71
77.85
78.10

60.08
65.38
65.94
65.94
67.74
–
–
67.49
67.12
–
–
67.56
66.91
69.38
66.10
71.41
69.68
73.70
72.19

–
65.05
63.57
63.57
64.40
64.72
65.88
64.01
65.12
63.97
65.76
67.68
68.51
65.28
63.46
67.60
68.25
68.91
69.01

–
39.83
38.34
38.34
42.60
–
–
39.64
42.93
–
–
40.06
41.22
41.29
38.06
44.93
42.30
48.83
47.72

–
61.95
60.51
60.51
–
61.49
–
60.20
61.87
62.85
61.80
65.35
67.50
66.90
60.20
67.55
71.69
68.05
73.45

–
48.26
47.64
47.64
–
–
–
47.38
47.82
–
–
50.88
52.56
52.31
47.17
52.92
55.19
55.93
58.55

Text encoder
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
CLIP-Xformer
BERT-base
BERT-base
CLIP-Xformer
BERT-base
CLIP-Xformer
BERT-base
CLIP-Xformer

Fig. 7. Effect of the scaling factor 𝜆 and cluster number 𝑁 on the Rank-1 performance of HAM-PDE across three datasets.

Fig. 8. Effect of the scaling factor 𝜆 and cluster number 𝑁 on the mAP performance of HAM-PDE across three datasets.

governing performance, whereas 𝜆 mainly provides stable modulation.
As 𝑁 increases, performance improves at first and then saturates once
𝑁 exceeds a dataset-specific threshold (around 1300, 1600, and 200
on CUHK-PEDES, ICFG-PEDES, and RSTPReid, respectively), indicating
that overly fine clustering leads to over-fragmentation and weaker
prototypes.
By contrast, 𝜆 remains stable across datasets and can be fixed at 𝜆 =
0.25 without noticeable performance loss, which simplifies deployment.
As shown in Figs. 7 and 8, Rank-1 and mAP follow similar trends with
respect to 𝑁. Although the 𝑁 that maximizes Rank-1 does not always
yield the best mAP, it still gives competitive overall performance, and
is therefore adopted in our setting.
The larger fluctuation of Rank-1 on RSTPReid is likely due to
its much smaller test set, which reduces intra-cluster sample support
and weakens prototype stability. This further suggests that adapting
prototype granularity to dataset statistics is beneficial, especially in
long-tailed TI-ReID scenarios.

and 𝛽. As shown in Fig. 9, most gains are obtained in the first iteration,
followed by smaller but consistent improvements, and the method
converges within four iterations on all datasets. We therefore fix 𝑀 = 4
as a good trade-off between effectiveness and complexity.
The decay factor 𝛼 controls cluster-number reduction, while 𝛽 controls the contribution of prototypes at each stage. As shown in Fig. 11,
larger 𝛼 consistently harms mAP because excessively rapid cluster contraction leads to over-smoothed and less diverse prototypes. This effect
becomes stronger when 𝛽 is small, indicating that aggressive cluster
shrinking should be accompanied by sufficient prototype weighting.
Fig. 10 shows that Rank-1 is more variable, but larger 𝛽 generally
stabilizes performance by preserving the influence of prototypes in
later stages. However, excessively large 𝛽 weakens the adaptability
of the similarity matrix to refined cluster structures. Based on these
observations, we adopt 𝛼 = 0.25, 𝛽 = 0.8, and 𝑀 = 4, which provide
stable and consistent performance across datasets.
4.4.3. Analysis of ACE module
We evaluate ACE from two perspectives: cluster-number estimation and cluster–identity consistency. As shown in Table 3, the estimated cluster number 𝑁 is close to the hand-tuned value on all three

4.4.2. Analysis of Iteration Number 𝑀 and decay factors 𝛼, 𝛽
To improve robustness, PDE𝑀 progressively refines prototypes over
𝑀 iterations while reducing cluster granularity with decay factors 𝛼
8

Pattern Recognition 179 (2026) 113705

H. Yang et al.

Fig. 9. Effect of the iteration number 𝑀 on the mAP performance of HAM-PDE𝑀 across three datasets.

Fig. 10. Effect of decay factors 𝛼 and 𝛽 on Rank-1 performance of HAM-PDE𝑀 across three datasets.

Fig. 11. Effect of decay factors 𝛼 and 𝛽 on mAP performance of HAM-PDE𝑀 across three datasets.
Table 3
ACE estimation and cluster–identity consistency.

4.5. Robustness, fairness, and efficiency evaluation

Dataset

Est. 𝑁

Manual 𝑁

Dev.

Purity

ID Frag.

CUHK-PEDES
ICFG-PEDES
RSTPReid

1282
1615
206

1300
1600
200

−1.4%
+0.9%
+3.0%

87.6%
79.1%
85.5%

1.42
2.70
1.66

Building on the ACE analysis above, we further examine the overall behavior of prototype-based refinement from the perspectives of
robustness, fairness, and usage, as summarized in Table 5. First, the
framework is not tied to a specific clustering solver: replacing Kmeans with GMM [49] yields comparable performance and inference
cost, while DBSCAN also provides a competitive baseline. ACE-based
variants nevertheless remain slightly better overall and more efficient,
indicating a more favorable balance between effectiveness and cost.
Second, gallery-aware refinement is not uniformly beneficial in TIReID: under the same frozen HAM features, similarity computation,
and evaluation settings, methods such as k-reciprocal re-ranking, RWR,
and CSLS are less stable, whereas our prototype-based refinement
consistently achieves the best overall performance with the highest
efficiency. This suggests that the gain comes not simply from exploiting gallery structure, but from using image-side identity anchors
to calibrate Query–Gallery similarity. Finally, introducing prototype
supervision during training consistently degrades performance (see Table 4), and PDE𝑀 on top only partially recovers it. This indicates
that prototype-based signals are less suitable as training targets, but
more effective as stable identity-level priors after feature learning has
converged. Overall, these results support using prototype priors as
lightweight inference-time refinement rather than additional training
supervision.

benchmarks, with only minor deviations. To further examine how the
resulting partitions relate to ground-truth identities, we report two
statistics: purity, which measures the dominant-ID concentration within
each cluster, and ID fragmentation, which measures how many clusters
one identity spans on average. The clusters exhibit high purity (87.6%,
79.1%, and 85.5%) and moderate ID fragmentation (1.42, 2.70, and
1.66) on CUHK-PEDES, ICFG-PEDES, and RSTPReid, respectively, indicating that ACE preserves meaningful identity-related structure even
when one identity may span multiple appearance modes.
With a fixed percentile threshold 𝑇 = 0.75 and default bandwidth
ℎ = 1.5, ACE remains stable across datasets, and the retrieval accuracy
peaks within a narrow range around this default bandwidth as shown in
Figs. 12 and 13. Fig. 14 further illustrates the influence of varying ℎ on
the mAP performance of PDE𝑀 across datasets, demonstrating that ACE
maintains consistent results without frequent parameter adjustments.
In practice, we set 𝑁 using ACE and avoid per-dataset tuning. This
not only simplifies deployment but also supports the ablation findings
that 𝑁 is a key factor in performance, while 𝜆 can be fixed. With ACE,
PDE and PDE𝑀 become plug-and-play, making them easy to integrate
without additional tuning.

4.6. Robustness, fairness, and efficiency evaluation
Building on the ACE analysis above, Table 5 further shows that
prototype-based refinement remains robust across clustering choices,
9

Pattern Recognition 179 (2026) 113705

H. Yang et al.

Fig. 12. Adaptive cluster number estimation of HAM-PDE under varying KDE bandwidths across three datasets.

Fig. 13. Impact of bandwidth ℎ on Rank-1 performance of HAM-PDE𝑀 across three datasets.

Fig. 14. Impact of bandwidth ℎ on mAP performance of HAM-PDE𝑀 across three datasets.

Table 4
Exploring prototype priors during training vs. inference.
Setting

baseline (HAM)
+ Prototype-in-Training (𝑝𝑟𝑜𝑡𝑜 )
+ PDE𝑀 (inference-only)
+ Prototype-in-Training + PDE𝑀

CUHK-PEDES

ICFG-PEDES

RSTPReid

R1

mAP

R1

mAP

R1

mAP

77.71
74.49
78.10
74.80

69.68
67.22
72.19
68.13

68.25
67.83
69.01
68.11

42.30
41.70
47.72
43.36

71.69
70.03
73.45
70.57

55.19
52.91
58.55
53.76

fair under unified gallery-aware comparison, and more suitable for
inference than training supervision. The framework is not tied to a specific clustering solver: replacing K-means with GMM yields comparable
performance and inference cost, while DBSCAN [50] also provides a
competitive baseline. ACE-based variants nevertheless remain slightly
better overall and more efficient, indicating a more favorable balance between effectiveness and cost. Under the same frozen HAM
features, similarity computation, and evaluation settings, gallery-aware
refinement is not uniformly beneficial in TI-ReID: methods such as
k-reciprocal re-ranking, RWR [51], and CSLS [52] are less stable,
whereas our prototype-based refinement consistently achieves the best
overall performance with the highest efficiency. This suggests that
the gain comes not simply from exploiting gallery structure, but from
using image-side identity anchors to calibrate Query–Gallery similarity.
We also find that introducing prototype supervision during training
consistently degrades performance, and PDE𝑀 on top only partially
recovers it. This indicates that prototype-based signals are less suitable
as training targets, but more effective as stable identity-level priors after feature learning has converged. Taken together, these observations
position prototype priors more naturally as lightweight inference-time
calibration than as additional training supervision (see Table 4).

4.7. Qualitative results
4.7.1. Visualization of similarity matrix
Fig. 15 illustrates how PDE𝑀 modifies the text-image similarity
between queries and gallery images. While the original HAM model already demonstrates strong cross-modal retrieval capability (evidenced
by prominent blocks along the diagonal), modality gaps still limit
further improvements (reflected by high-similarity regions off the diagonal). PDE𝑀 leverages identity prototypes as priors to selectively
enhance similarities within the same identity, refining the matching
process over multiple inference stages. This results in an increased
margin between positive and negative pairs and ultimately improves
retrieval results, as shown by the increased mAP scores.

4.7.2. Visualization of ranking robustness
As shown in Fig. 16, we visualize the top-15 retrieval results of
the SOTA model HAM before and after applying PDE𝑀 . With PDE𝑀 ,
a larger proportion of positive samples are retrieved and ranked higher
within the returned list. This demonstrates that our method enhances
retrieval accuracy and improves the model’s robustness.
10

Pattern Recognition 179 (2026) 113705

H. Yang et al.

Table 5
Performance and inference time comparison of test-time refinement methods.
Dataset name

CUHK-PEDES

ICFG-PEDES

RSTPReid

Baseline

Test-time refinement

Rank-1

MAP

Time (s)

HAM

None
Rerank
RWR
CSLS
DBSCAN
ACE+Kmeans
ACE+GMM

77.71
75.55
77.82
77.72
77.97
78.1
78.04

69.68
65.89
69.92
69.70
71.85
72.19
72.11

88
336
183
127
254
112
120

HAM

None
Rerank
RWR
CSLS
DBSCAN
ACE+Kmeans
ACE+GMM

68.25
67.10
68.26
68.55
68.74
69.01
69.15

42.30
45.47
43.76
43.98
46.99
47.72
47.81

584
2560
1217
749
1662
656
821

HAM

None
Rerank
RWR
CSLS
DBSCAN
ACE+Kmeans
ACE+GMM

71.69
56.80
71.82
71.75
73.20
73.45
73.44

55.19
54.29
55.92
55.76
57.69
58.55
58.34

56
136
99
80
110
72
83

Fig. 15. Visualization of text-image similarity matrices on RSTPReid. PDE𝑀 selectively strengthens intra-identity similarities (diagonal blocks) through iterative
refinement, resulting in overall improved ranking of positive samples and enhanced retrieval precision.

Fig. 16. Example retrieval results for three queries on the ICFG-PEDES dataset. For each query, the top row displays the ranking produced by the HAM model,
and the bottom row shows the ranking after applying PDE𝑀 . Correct matches are highlighted with green boxes. (For interpretation of the references to color in
this figure legend, the reader is referred to the web version of this article.)

11


hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
10578

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025

TABLE I
C OMPARISONS W ITH S TATE - OF - THE -A RT M ETHODS ON SYSU-MM01 AND R EG DB, I NCLUDING SVI-R E ID, SSVI-R E ID, AND USVI-R E ID M ETHODS .
A LL M ETHODS A RE M EASURED BY R ANK -1 (%) AND M AP (%). GUR* D ENOTES THE R ESULTS W ITHOUT C AMERA I NFORMATION . † D ENOTES
THE R ESULTS R EPRODUCED F ROM THE O FFICIAL C ODE

30 following [75] and K is fixed to 20. The hyperparameter
λ is set to 25 following [38]. Following ADCA [15], the
momentum value µ is set to 0.1 and the temperature τ is 0.5.
The margin hyperparameter γ and the kernel bandwidth σ are
both set to 1.0 following [76]. The trade-off hyperparameter
α is set to 0.5, k1 and k2 is set to 0.5 and 10.0, respectively.
π+ is set to 0.1 following [16].
B. Comparison With State-of-the-Art Methods
To comprehensively illustrate the efficiency of our proposed
APPD, we not only compare our method with SOTA unsupervised visible-infrared person ReID (USVI-ReID) methods
but also explore the promising application of our method
on supervised visible-infrared person ReID (SVI-ReID) and
semi-supervised visible-infrared person ReID (SSVI-ReID)
methods. Unless otherwise specified, all analyses are conducted on the SYSU-MM01 dataset under the All Search
mode.

1) Comparison with USVI-ReID Methods: As shown in
Tab. I, we can see that the proposed APPD gains the best
performance compared to existing state-of-the-art USVI-ReID
methods on the SYSU-MM01 and RegDB datasets. To be
specific, for the All Search mode on SYSU-MM01, our
APPD achieves 64.7% in Rank-1 and 60.5% in mAP, which
surpasses MMM by 3.1% in Rank-1 and 2.6% in mAP. For
the Indoor Search mode on SYSU-MM01, APPD also exhibits
outstanding performance, which gains the performance with
70.6% in Rank-1 and 75.3% in mAP, exceeding MMM by a
large margin with 6.2% in Rank-1 and 4.9% in mAP. Morever,
the performance on RegDB achieves 91.1% in Rank-1 and
83.8% in mAP under the Visible2Thermal mode, and 90.4%
in Rank-1 and 82.8% in mAP under the Thermal2Visible
mode. Compared to our previous work RPNR, our APPD
demonstrates a notable performance improvement, achieving
a 1.2% increase in average Rank-1 accuracy and a 1.0% boost
in average mAP. The results strongly validate the effectiveness
of our proposed approach, showcasing that APPD generates

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

YIN et al.: ADAPTIVE PSEUDO-LABEL PURIFICATION AND DEBIASING FOR USVI-ReID

10579

TABLE II
A BLATION S TUDIES ON THE SYSU-MM01 DATASET. R ANK -R ACCURACY (%) AND M AP(%) A RE R EPORTED

more reliable pseudo-labels and establishes more robust crossmodality correspondences for the USVI-ReID task.
2) Comparison with SVI-ReID Methods: To ensure a fair
comparison with SVI-ReID methods, we explored the application potential of our APPD under the fully supervised setting.
Unlike the unsupervised setting, we replaced the clusteringgenerated pseudo-labels with the ground-truth visible and
infrared labels to obtain the results under the fully supervised setting, as presented in Tab. I. On the SYSU-MM01
dataset, APPD achieves a Rank-1 accuracy of 71.4% and a
mAP of 68.5%, demonstrating performance comparable to
that of many state-of-the-art SVI-ReID methods. Furthermore,
APPD delivers remarkable performance on the RegDB dataset,
achieving 94.4% in Rank-1 accuracy and 88.4% in mAP under
the Visible2Thermal setting, and 94.0% in Rank-1 accuracy
and 87.3% in mAP under the Thermal2Visible setting. In
contrast to methods specifically designed for the SVI-ReID
task, our approach, while primarily addressing the challenges
of USVI-ReID, showcases exceptional performance on the
SVI-ReID task, highlighting the robustness and effectiveness
of our proposed framework.
3) Comparison with SSVI-ReID Methods: Following the
“Uni-Semi-Supervised” setting introduced in [70], we replaced
the visible pseudo-labels with ground-truth visible labels
while retaining the infrared pseudo-labels generated by the
DBSCAN algorithm to perform the SSVI-ReID task. As
shown in Tab. I, the results demonstrate that APPD is superior
against all the existing SSVI-ReID methods. Specifically, it
achieves the performance with 92.1% in Rank-1 and 84.8%
under the Visible2Thermal setting and 91.3% in Rank-1 and
83.5% in mAP under the Thermal2Visible setting. Compared
to MUGG in the semi-supervised setting, our APPD delivers
a notable improvement of 3.1% in average Rank-1 and 4.5%
in average mAP. It is noteworthy that our APPD, without any
cross-modality annotations (unsupervised setting), achieves a
4.2% improvement in Rank-1 and a 7.1% improvement in
mAP on RegDB under the Visible2Thermal compared to the
SOTA MUGG method, which is attributed to the fact that our
proposed method can provide reliable infrared pseudo-labels
for unsupervised tasks.
C. Ablation Study
In this section, to validate the effectiveness of each module
in APPD, we employ the DCL framework with multiple

Fig. 4. The ARI metric of visible and infrared pseudo-labels on the SYSUMM01 dataset at each epoch.

proxies as the Baseline to conduct ablation experiments on
SYSU-MM01, as shown in Tab. II.
1) Effectiveness of the APCD Module: The APCD module
is designed to explicitly rectify noisy pseudo-labels and adaptively identify cleaner pseudo-labels, ensuring the generation
of more reliable labels. As observed in Order 5 and Order
6 of Tab. II, integrating the APCD in Order 6 results in an
approximate 2% enhancement in performance compared to
Order 5. Moreover, to further demonstrate the efficacy of the
APCD module, we utilize the Adjusted Rand Index (ARI)
metric to evaluate the accuracy of visible and infrared pseudolabels on SYSU-MM01 over different epochs. A higher ARI
score indicates more accurate pseudo-labels. As illustrated
in Fig. 4, incorporating the pseudo-labels calibration strategy
markedly improves the accuracy of both visible and infrared
pseudo-labels, thereby ensuring more reliable pseudo-labels to
support network training effectively. We also provide a comparison of pseudo-label accuracy between the entire dataset
and the clean subset to highlight the effectiveness of the
proposed pseudo-label division strategy compared to previous
NPC. As shown in Fig. 6, the accuracy of pseudo-labels in
the clean subset divided from the entire dataset is significantly enhanced. Additionally, we investigate how the APCD

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

10580

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025

Fig. 5. The accuracy of cross-modality correspondences on the SYSU-MM01
dataset at each epoch.

Fig. 6. The comparison of pseudo-label accuracy between all data and clean
data on the SYSU-MM01 dataset.

module enhances the performance of the OTPM module. Fig. 5
illustrates a comparison of cross-modality correspondence
accuracy between configurations with and without APCD
across different epochs. As we can see, the accuracy of crossmodality correspondences is significantly enhanced with the
inclusion of the APCD module, which effectively helps to
reduce cross-modality gaps. These results highlight the APCD
module’s ability to effectively identify cleaner pseudo-labels,
thereby significantly enhancing the model’s robustness.
2) Effectiveness of the NRL Module: The NRL module
addresses the limitations of rigid pseudo-labels by introducing
greater flexibility and adaptability, allowing it to model interaction relationships between sample pairs without relying on
pseudo-label information. Integrating the NRL module results
in a 2%-3% improvement in Rank-1 accuracy on SYSUMM01, demonstrating its ability to effectively capture complex
interactions among sample pairs. This provides complementary supervisory signals, thereby enhancing the network’s
training process.
3) Effectiveness of the OTPM Module: PGM, a crossmodality matching module based on the bipartite graph
introduced in [11], serves as a benchmark for comparison. As

Fig. 7. The accuracy of cross-modality correspondences compared with PGM
[11] on SYSU-MM01.

shown in Fig. 7, we evaluated the cross-modality matching
accuracy of OTPM against PGM using identical prototype
features on four clustering evaluation metrics, ensuring a
more equitable comparison. The results reveal that OTPM
consistently outperforms PGM across all metrics, highlighting
its superior ability to establish reliable cross-modality correspondences at the cluster level. This capability significantly
enhances the association of cross-modality samples, paving
the way for robust cross-modality contrastive learning.
4) Effectiveness of the DMHL Module: We propose
the DMHL module to jointly learn modality-specific and
modality-invariant information while addressing the sampling
bias phenomenon. It is worth noting that the DMHL module
relies on the OTPM module as its foundation and cannot function independently. Compared to the Baseline, the
combination of DMHL with OTPM leads to a significant
performance improvement, with a large margin of 21.2%
in Rank-1 accuracy and 16.8% in mAP (see Order 1 &
Order 5). This highlights the efficiency of DMHL in leveraging
modality-specific and modality-invariant information, effectively mitigating cross-modality differences and the sampling
bias phenomenon, shielding the model from the impact of false
negative samples.
D. Further Analysis
1) Differences with RPNR: The current work substantially
extends our conference version RPNR through two principal
technical contributions: Adaptive Pseudo-label Calibration and
Division (APCD) and Debiased Memory Hybrid Learning
(DMHL). Comprehensive comparative results demonstrating
their superior performance over RPNR are systematically
presented in Tab. III. Compared to NPC, APCD employs
an adaptive pseudo-label division strategy to identify cleaner
pseudo-labels, thereby enhancing the robustness of the model.
DMHL builds upon MHL by introducing debiased contrastive
learning to alleviate sampling bias, leading to improved model
performance. Overall, APPD outperforms RPNR by 1.5% in
average Rank-1 accuracy and 1.3% in mAP.
2) Hyperparameter Analysis: Our method incorporates
three key hyperparameters: α, k1 , and k2 . To assess their
impact, we provide the quantitative results under different parameter values in Fig. 8. Specifically, we adjust one

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

YIN et al.: ADAPTIVE PSEUDO-LABEL PURIFICATION AND DEBIASING FOR USVI-ReID

10581

TABLE III
Q UANTITATIVE C OMPARISON W ITH THE P REVIOUS C ONFERENCE V ERSION NAMED RPNR ON THE SYSU-MM01 DATASET. † D ENOTES THE R ESULTS
R EPRODUCED F ROM THE O FFICIAL C ODE OF RPNR

Fig. 9. Four clustering evaluation metrics compared with SOTA methods on
the SYSU-MM01 dataset. “RGB” and “IR” denote the accuracy of visible
and infrared pseudo-labels.

Fig. 8. The influence of three important hyperparameters with different values
on SYSU-MM01.

hyperparameter while keeping the others fixed to analyze its
influence on performance. The hyperparameter α controls the
balance between visible and infrared prototype fusion. The
results reveal that integrating information from both prototypes
enhances performance, with the optimal results achieved at
α = 0.5. The hyperparameters k1 and k2 serve as balancing
factors to regulate the influence of various loss components,
and the best performance is achieved when k1 is set to 0.5 and
k2 is set to 10.0.
3) Accuracy of Pseudo-Labels: As illustrated in Fig. 9, we
compare our approach with several state-of-the-art USVI-ReID
methods using four standard clustering evaluation metrics
to highlight the effectiveness of the proposed APPD. The
results demonstrate that the visible and infrared pseudo-labels

Fig. 10. The visualization of the distance distribution of cross-modality
positive and negative pairs on SYSU-MM01, with V-I positive pairs indicated
by red and V-I negative pairs indicated by blue.

generated by APPD consistently outperform those produced
by existing methods across all four metrics. This underscores
the ability of our method to generate more reliable pseudolabels, which in turn facilitates enhanced network training and
substantial performance gains.
E. Visualization
In this section, we present several visualization results,
including distance distributions, feature distributions, and
retrieval results, to intuitively demonstrate the effectiveness of
the proposed APPD.
1) Distances of Cross-Modality Positive and Negative
Pairs: To comprehensively assess the effectiveness of our
APPD approach, we visualize the distance distribution of

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.

10582

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 10, OCTOBER 2025

Fig. 11. The Top10 retrieval results of selected five identities between Baseline and our proposed APPD. The blue borders indicate correct retrievals, while
the red borders represent incorrect ones.

Fig. 12. The t-SNE visualization illustrates 10 randomly selected identities,
where each identity is represented by a distinct color, and each modality
(visible and infrared) is denoted by different shapes.

cross-modality positive and negative pairs on SYSU-MM01.
Here, V-I positive pairs represent samples from the same
identity across different modalities, while V-I negative pairs
denote samples from different identities across different modalities. As shown in Fig. 10, compared to the Baseline, our
method effectively increases the separation between the mean
distances of V-I positive and V-I negative pairs. This result
underscores the capability of APPD to significantly reduce
cross-modality discrepancies, thereby enhancing overall
performance.
2) Feature Distribution: We visualize the feature distributions of visible and infrared data using t-SNE in a 2-D
embedding space, containing 10 randomly selected identities.
As shown in Fig. 12, our approach significantly outperforms
the Baseline by producing more compact feature distributions
for the same identities within the same modality (highlighted
by the blue and green circles). Furthermore, it brings feature
distributions for the same identities across different modalities
closer together (marked by the orange and purple circles).
These results highlight the effectiveness of APPD in reducing

cross-modality disparities, thereby establishing a solid foundation for reliable cross-modality correspondences.
3) Retrieval Results: To further intuitively show the effectiveness of APPD, we also compare the top-10 retrieved
results of five randomly selected identities between APPD
and Baseline on SYSU-MM01 under the All Search mode.
The visualization of the retrieval results is shown in Fig. 11.
The images in the first column represent the query images,
while the subsequent columns display the corresponding top10 retrieval images ranked from left to right based on similarity
scores. Blue borders indicate correct retrievals, whereas red
borders signify incorrect ones. As we can see, APPD significantly enhances the ranking results, achieving more correctly
matched images in the top positions compared to the Baseline.
These results underscore the effectiveness of our proposed
method in improving feature discriminability and retrieval
accuracy. However, our method may fail to accurately detect
pedestrians in challenging scenarios, such as rear-view poses
(e.g., ID 37) or galley images with heavily cluttered backgrounds (e.g., IDs 24, 274). These factors introduce significant
retrieval difficulties, as even human evaluators struggle to
recognize highly ambiguous cases like ID 37. To address
these limitations, our future work will incorporate local feature
learning (e.g., head, body, and leg regions) to improve model
generalization and robustness in such challenging scenarios.
V. C ONCLUSION
In this paper, we present a novel and effective framework,
termed Adaptive Pseudo-label Purification and Debiasing
(APPD) for USVI-ReID, to address the unique challenges of
the USVI-ReID task. Our approach focuses on deriving more
reliable pseudo-labels and establishing robust cross-modality
correspondences, key factors for improving performance in
this domain. To achieve this, we first propose the Adaptive
Pseudo-label Calibration and Division (APCD) module, which

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:59:29 UTC from IEEE Xplore. Restrictions apply.


codex
证据部分我也补读了几篇的后续消融页。接下来会把每篇拆成“观察如何升级成问题，问题如何推出机制，证据如何闭环”，避免只列模块名。
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
/bin/zsh -lc "pdftotext -f 8 -l 10 'Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5074

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

TABLE III

TABLE IV

C OMPARATIVE OF P ERFORMANCE ACROSS VARIOUS C OMPONENTS

C OMPARISON OF R ESULTS FOR D IFFERENT N UMBERS OF H UMAN
P OSITIONS ON THE O CCLUDED -D UKE AND
M ARKET-1501 DATASETS

Fig. 6.

Feature map visualization.

component of our AOANet, including the Adaptive Position
Extractor (A) and the Occlusion Awareness Module (O). The
baseline configuration omits all components and solely utilizes
Swin-S as the backbone for extracting the global feature,
which denoted as f g1 in (1). The results are shown in Table III.
Effectiveness of the Adaptive Position Extractor: As indicated in index-2 of Table III, the introduction of the APE
yields a significant improvement in performance compared
to the baseline. In this instance, only the global feature f g2
from (2) is employed. On the MSMT17 dataset, both mAP
and Rank-1 score show notable improvements, increasing by
11.8% and 6.1%, respectively. Similarly, on the OccludedDuke dataset, there are substantial improvements, with mAP
and Rank-1 score increasing by 18.4% and 16.9%, respectively. The APE has the capability to extract adaptive body
positions, yielding local features that are highly robust and
discriminative. Furthermore, the global features derived from
these local features exhibit robustness as well. In Fig. 6,
we further illustrate the feature representation capability of
the proposed method through feature map visualization. It is
evident from the figure that the features extracted by Swin-S
are somewhat incomplete. Fortunately, with the addition
of APE, the robustness of the extracted features is significantly
enhanced.
Effectiveness of the Occlusion Awareness Module: As indicated in index-3 of Table III, the performance was improved on
both datasets by adding OAM on top of APE. Specifically, on
the MSMT17 dataset, there is an increase of 1.9% in mAP and
1.3% in rank-1 score. The improvement is more pronounced
on the Occluded-Duke dataset, with mAP increasing by 2.6%
and rank-1 score by 2.9%. This notable enhancement can
be attributed to OAM’s ability to further leverage occlusion

Fig. 7.
Three groups with different numbers of human body position
divisions. The top row in each group represents positions extracted by APE,
while the bottom row corresponds to labels generated by pose estimation.

information within local features, which is a particularly
advantageous property on occluded datasets.
Analysis of the Number of Human Body Positions: We
conducted quantitative experiments on the Occluded-Duke and
Market-1501 datasets to determine the optimal number of
human body positions. As depicted in Table IV, the best
performance is achieved when the number of positions is
set to 4. Furthermore, as shown in Fig. 7, we intuitively
represent the perception results of the proposed method for
different numbers of human body positions. In Fig. 7a, it is
observed that setting the number of positions to 11 results
in only some positions producing meaningful results, while
the rest are redundant. For Fig. 7b, where the number is set
to 6, we observed that APE tends to prioritize the extraction
of crucial features, disregarding the distinction between left
and right. In summary, we opt to set the number to 4,
encompassing the head, the upper body, the thigh, and the
lower leg.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.

LIU et al.: ADAPTIVE OCCLUSION-AWARE NETWORK FOR OCCLUDED PERSON ReID

5075

capable of extracting highly robust global and local features.
To enhance the robustness of local features to occlusion, we
constructed an Adaptive Position Extractor (APE), specifically
designed to extract adaptive and robust human body region.
Those regions are used to generate local features that adapt to
the occlusion and are further aggregated to form discriminative
global features. Furthermore, in the inference stage, we developed the Occlusion Awareness Module(OAM) to maximise
the use of occlusion information in local features, suppressing
occluded parts and highlighting non-occluded parts by calculating the occlusion weight of local features. Comprehensive
experiments conducted on both holistic and occluded datasets
validate the effectiveness of our proposed method.
R EFERENCES

Fig. 8. Visualisation of the positions extracted by APE. The top row for each
person image represents the positions extracted by APE, while the bottom row
represents the labels used during APE training.

E. Visualization of Adaptive Body Positions
We depict the human body positions extracted by APE in
Fig. 8. The results reveal that even when the pose estimation
model provides inaccurate results as the training label for
APE, our method excels in accurately extracting human body
positions, such as the first set of images in Fig. 8a. In the
first set of images in Fig. 8b, the head is covered by an
umbrella. The label provided by the pose estimation model is
an incorrect head position, and APE can accurately avoid the
incorrect head position. Fig. 8d shows that APE can maintain
accurate location extraction even when the person’s image is
scaled. Furthermore, in Fig. 8f, the pose estimation model
ignores the handbag, while the APE is able to capture the
handbag. These examples collectively demonstrate that the
positions extracted by APE surpass those generated by pose
estimation, exhibiting both robustness and discriminative.
V. C ONCLUSION
In this paper, we present an Adaptive Occlusion-Aware
Network (AOANet) for occluded person re-identification,

[1] H. Luo, Y. Gu, X. Liao, S. Lai, and W. Jiang, “Bag of tricks and a
strong baseline for deep person re-identification,” in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), Jun. 2019,
pp. 1487–1495, doi: 10.1109/CVPRW.2019.00190.
[2] Y. Lu et al., “Cross-modality person re-identification with
shared-specific feature transfer,” in Proc. IEEE/CVF Conf.
Comput. Vis. Pattern Recognit., Jun. 2020, pp. 13379–13389, doi:
10.1109/CVPR42600.2020.01339.
[3] K. Jiang, T. Zhang, Y. Zhang, F. Wu, and Y. Rui, “Self-supervised
agent learning for unsupervised cross-domain person re-identification,”
IEEE Trans. Image Process., vol. 29, pp. 8549–8560, 2020, doi:
10.1109/TIP.2020.3016869.
[4] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for
image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit.
(CVPR), Jun. 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.
[5] J. Zhuo, Z. Chen, J. Lai, and G. Wang, “Occluded person reidentification,” in Proc. IEEE Int. Conf. Multimedia Expo. (ICME),
Jul. 2018, pp. 1–6, doi: 10.1109/ICME.2018.8486568.
[6] J. Miao, Y. Wu, P. Liu, Y. Ding, and Y. Yang, “Pose-guided
feature alignment for occluded person re-identification,” in Proc.
IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2019, pp. 542–551,
doi: 10.1109/ICCV.2019.00063.
[7] Y. Cho, W. J. Kim, S. Hong, and S.-E. Yoon, “Part-based pseudo
label refinement for unsupervised person re-identification,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022,
pp. 7308–7318, doi: 10.1109/CVPR52688.2022.00716.
[8] W. Liu, Q. Bao, Y. Sun, and T. Mei, “Recent advances of monocular
2D and 3D human pose estimation: A deep learning perspective,” ACM
Comput. Surv., vol. 55, no. 4, pp. 1–41, 2022. [Online]. Available:
https://dl.acm.org/doi/10.1145/3524497
[9] Z. Liu et al., “Swin transformer: Hierarchical vision transformer using
shifted windows,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV),
Oct. 2021, pp. 10012–10022, doi: 10.1109/ICCV48922.2021.00986.
[10] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu, “Deeply-supervised
nets,” in Proc. 18th Int. Conf. Artif. Intell. Statist., 2015, pp. 562–570.
[11] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, and S. C. H. Hoi, “Deep
learning for person re-identification: A survey and outlook,” IEEE Trans.
Pattern Anal. Mach. Intell., vol. 44, no. 6, pp. 2872–2893, Jun. 2022,
doi: 10.1109/TPAMI.2021.3054775.
[12] X. Jin, C. Lan, W. Zeng, G. Wei, and Z. Chen, “Semantics-aligned
representation learning for person re-identification,” in Proc. AAAI Conf.
Artif. Intell., 2020, vol. 34, no. 7, pp. 11173–11180. [Online]. Available:
https://ojs.aaai.org/index.php/AAAI/article/view/6775
[13] C. Zhao, X. Lv, Z. Zhang, W. Zuo, J. Wu, and D. Miao, “Deep
fusion feature representation learning with hard mining center-triplet loss
for person re-identification,” IEEE Trans. Multimedia, vol. 22, no. 12,
pp. 3180–3195, Dec. 2020, doi: 10.1109/TMM.2020.2972125.
[14] L. He and W. Liu, “Guided saliency feature learning for person
re-identification in crowded scenes,” in Proc. Eur. Conf. Comput.
Vis. (ECCV). Cham, Switzerland: Springer, 2020, pp. 357–373.
[Online]. Available: https://link.springer.com/chapter/10.1007/978-3030-58604-1_22
[15] R. Yu, Z. Dou, S. Bai, Z. Zhang, Y. Xu, and X. Bai, “Hard-aware
point-to-set deep metric for person re-identification,” in Proc. Eur. Conf.
Comput. Vis. (ECCV), Sep. 2018, pp. 188–204.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.

5076

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

[16] X. Yang, P. Zhou, and M. Wang, “Person reidentification via structural deep metric learning,” IEEE Trans. Neural Netw. Learn. Syst.,
vol. 30, no. 10, pp. 2987–2998, Oct. 2019, doi: 10.1109/TNNLS.2018.
2861991.
[17] G. Chen, Y. Lu, J. Lu, and J. Zhou, “Deep credible metric learning for unsupervised domain adaptation person re-identification,” in
Proc. Eur. Conf. Comput. Vis. (ECCV). Cham, Switzerland: Springer,
2020, pp. 643–659. [Online]. Available: https://link.springer.com/
chapter/10.1007/978-3-030-58598-3_38
[18] G. Wang et al., “High-order information matters: Learning relation and topology for occluded person re-identification,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2020,
pp. 6449–6458, doi: 10.1109/CVPR42600.2020.00648.
[19] C. Yan, G. Pang, J. Jiao, X. Bai, X. Feng, and C. Shen, “Occluded person re-identification with single-scale global representations,” in Proc.
IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 11875–11884,
doi: 10.1109/ICCV48922.2021.01166.
[20] Y. Sun, L. Zheng, Y. Yang, Q. Tian, and S. Wang, “Beyond part models:
Person retrieval with refined part pooling (and a strong convolutional
baseline),” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 480–496.
[Online]. Available: https://link.springer.com/chapter/10.1007/978-3030-01225-0_30
[21] G. Wang, Y. Yuan, X. Chen, J. Li, and X. Zhou, “Learning discriminative
features with multiple granularities for person re-identification,” in
Proc. 26th ACM Int. Conf. Multimedia, Oct. 2018, pp. 274–282, doi:
10.1145/3240508.3240552.
[22] F. Zheng et al., “Pyramidal person re-identification via multiloss dynamic training,” in Proc. IEEE/CVF Conf. Comput.
Vis. Pattern Recognit. (CVPR), Jun. 2019, pp. 8514–8522, doi:
10.1109/CVPR.2019.00871.
[23] G. Wang, J. Lai, P. Huang, and X. Xie, “Spatial-temporal person
re-identification,” in Proc. AAAI Conf. Artif. Intell., 2019, vol. 33,
no. 1, pp. 8933–8940. [Online]. Available: https://ojs.aaai.org/index.
php/AAAI/article/view/4921
[24] Y. Suh, J. Wang, S. Tang, T. Mei, and K. M. Lee, “Part-aligned
bilinear representations for person re-identification,” in Proc. Eur.
Conf. Comput. Vis. (ECCV), 2018, pp. 402–419. [Online]. Available:
https://link.springer.com/chapter/10.1007/978-3-030-01264-9_25
[25] V. Somers, C. D. Vleeschouwer, and A. Alahi, “Body part-based
representation learning for occluded person re-identification,” in Proc.
IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), Jan. 2023,
pp. 1613–1623, doi: 10.1109/WACV56688.2023.00166.
[26] M. Lin, Q. Chen, and S. Yan, “Network in network,” 2013,
arXiv:1312.4400.
[27] T. Y. Lin, P. Dollàr, R. Girshick, K. He, B. Hariharan, and
S. Belongie, “Feature pyramid networks for object detection,” in Proc.
IEEE Conf. Comput. Vis. Pattern Recognit., Jul. 2017, pp. 2117–2125,
doi: 10.1109/CVPR.2017.106.
[28] A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural
Inf. Process. Syst., vol. 30, 2017, pp. 1–11, doi: 10.5555/3295222.
3295349.
[29] S. Kreiss, L. Bertoni, and A. Alahi, “PifPaf: Composite fields
for human pose estimation,” in Proc. IEEE/CVF Conf. Comput.
Vis. Pattern Recognit. (CVPR), Jun. 2019, pp. 11977–11986, doi:
10.1109/CVPR.2019.01225.
[30] K. Zhu, H. Guo, Z. Liu, M. Tang, and J. Wang, “Identity-guided human
semantic parsing for person re-identification,” in Proc. Eur. Conf.
Comput. Vis. (ECCV). Cham, Switzerland: Springer, 2020, pp. 346–363.
[Online]. Available: https://link.springer.com/chapter/10.1007/978-3030-58580-8_21
[31] S. He, H. Luo, P. Wang, F. Wang, H. Li, and W. Jiang, “TransReID: Transformer-based object re-identification,” in Proc. IEEE/CVF
Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 15013–15022, doi:
10.1109/ICCV48922.2021.01474.
[32] P. Chen et al., “Occlude them all: Occlusion-aware attention network
for occluded person re-ID,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., Oct. 2021, pp. 11833–11842, doi: 10.1109/ICCV48922.2021.
01162.
[33] Y. Li, J. He, T. Zhang, X. Liu, Y. Zhang, and F. Wu, “Diverse
part discovery: Occluded person re-identification with part-aware transformer,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), Jun. 2021, pp. 2898–2907, doi: 10.1109/CVPR46437.2021.
00292.
[34] G. Zhang, P. Zhang, J. Qi, and H. Lu, “HAT: Hierarchical aggregation
transformers for person re-identification,” in Proc. 29th ACM Int. Conf.
Multimedia, Oct. 2021, pp. 516–525.

[35] Z. Ma, Y. Zhao, and J. Li, “Pose-guided inter-and intra-part
relational transformer for occluded person re-identification,” in
Proc. 29th ACM Int. Conf. Multimedia, 2021, pp. 1487–1496, doi:
10.1145/3474085.3475283.
[36] Z. Wang, F. Zhu, S. Tang, R. Zhao, L. He, and J. Song, “Feature
erasing and diffusion network for occluded person re-identification,”
in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Jun. 2022,
pp. 4754–4763, doi: 10.1109/CVPR52688.2022.00471.
[37] H. Wang, J. Shen, Y. Liu, Y. Gao, and E. Gavves, “NFormer: Robust
person re-identification with neighbor transformer,” in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit., Jun. 2022, pp. 7297–7307, doi:
10.1109/CVPR52688.2022.00715.
[38] T. Wang, H. Liu, P. Song, T. Guo, and W. Shi, “Pose-guided
feature disentangling for occluded person re-identification based on
transformer,” in Proc. AAAI Conf. Artif. Intell., 2022, vol. 36,
no. 3, pp. 2540–2549. [Online]. Available: https://ojs.aaai.org/index.
php/AAAI/article/view/20155
[39] L. Tan, P. Dai, R. Ji, and Y. Wu, “Dynamic prototype mask for occluded
person re-identification,” in Proc. 30th ACM Int. Conf. Multimedia,
Oct. 2022, pp. 531–540, doi: 10.1145/3503161.3547764.
[40] X. Cheng, M. Jia, Q. Wang, and J. Zhang, “More is better: Multisource dynamic parsing attention for occluded person re-identification,”
in Proc. 30th ACM Int. Conf. Multimedia, Oct. 2022, pp. 6840–6849,
doi: 10.1145/3503161.3547819.
[41] G. Yan, Z. Wang, S. Geng, Y. Yu, and Y. Guo, “Part-based representation
enhancement for occluded person re-identification,” IEEE Trans. Circuits
Syst. Video Technol., vol. 33, no. 8, pp. 4217–4231, Aug. 2023, doi:
10.1109/TCSVT.2023.3241764.
[42] Z. Dou, Z. Wang, Y. Li, and S. Wang, “Identity-seeking self-supervised
representation learning for generalizable person re-identification,”
in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2023,
pp. 15847–15858, doi: 10.1109/ICCV51070.2023.01452.
[43] G. Zhang, Y. Zhang, T. Zhang, B. Li, and S. Pu, “PHA: Patch-wise highfrequency augmentation for transformer-based person re-identification,”
in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR),
Jun. 2023, pp. 14133–14142, doi: 10.1109/CVPR52729.2023.01358.
[44] M. Jia, Y. Sun, Y. Zhai, X. Cheng, Y. Yang, and Y. Li, “Semi-attention
partition for occluded person re-identification,” in Proc. AAAI Conf.
Artif. Intell., 2023, vol. 37, no. 1, pp. 998–1006. [Online]. Available:
https://ojs.aaai.org/index.php/AAAI/article/view/25180
[45] W. Liu et al., “Learning occlusion disentanglement with finegrained localization for occluded person re-identification,” in Proc.
31st ACM Int. Conf. Multimedia, Oct. 2023, pp. 6462–6471, doi:
10.1145/3581783.3612452.
[46] T. Wang et al., “Feature completion transformer for occluded person reidentification,” IEEE Trans. Multimedia, vol. 26, pp. 8529–8542, 2024,
doi: 10.1109/TMM.2024.3379908.
[47] N. Dong, L. Zhang, S. Yan, H. Tang, and J. Tang, “Erasing, transforming,
and noising defense network for occluded person re-identification,” IEEE
Trans. Circuits Syst. Video Technol., vol. 34, no. 6, pp. 4458–4472,
Jun. 2024, doi: 10.1109/TCSVT.2023.3339167.
[48] L. Wei, S. Zhang, W. Gao, and Q. Tian, “Person transfer GAN to
bridge domain gap for person re-identification,” in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit., Jun. 2018, pp. 79–88, doi:
10.1109/CVPR.2018.00016.
[49] E. Ristani, F. Solera, R. Zou, R. Cucchiara, and C. Tomasi,
“Performance measures and a data set for multi-target, multicamera tracking,” in Proc. Eur. Conf. Comput. Vis. (ECCV).
Cham, Switzerland: Springer, 2016, pp. 17–35. [Online]. Available:
https://link.springer.com/chapter/10.1007/978-3-319-48881-3_2#Sec12
[50] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian,
“Scalable person re-identification: A benchmark,” in Proc. IEEE
Int. Conf. Comput. Vis. (ICCV), Dec. 2015, pp. 1116–1124, doi:
10.1109/ICCV.2015.133.
[51] Z. Zhong, L. Zheng, D. Cao, and S. Li, “Re-ranking person reidentification with K-reciprocal encoding,” in Proc. IEEE Conf.
Comput. Vis. Pattern Recognit., Jul. 2017, pp. 1318–1327, doi:
10.1109/CVPR.2017.389.
[52] D. Fu et al., “Unsupervised pre-training for person re-identification,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR),
Jun. 2021, pp. 14750–14759, doi: 10.1109/CVPR46437.2021.
01451.
[53] Z. Gao et al., “A semantic perception and CNN-transformer hybrid
network for occluded person re-identification,” IEEE Trans. Circuits
Syst. Video Technol., vol. 34, no. 4, pp. 2010–2025, Apr. 2024, doi:
10.1109/TCSVT.2023.3296680.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:09 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 8 -l 10 'ADA framework for unsupervised domain adaptation person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 171 (2026) 112238

W. Zhang et al.

4.4.6. Summary of ablation experiment
Experimental results demonstrate the strong universality of the ADA
framework. The framework exhibits compatibility with diverse backbone networks. Furthermore, it is not limited to any speciﬁc sparse selfattention mechanism, allowing ﬂexible selection according to application requirements.
The ADA framework maintains consistent performance enhancement
across diﬀerent design paradigms of sparse self-attention.
The ADA framework oﬀers considerable deployment ﬂexibility, with
eﬀective integration across diﬀerent network stages corresponding to
low-, medium-, and high-level features. This adaptability enables optimized performance across various architectural conﬁgurations.
Compared to standalone implementations in Transformer architectures [19], the ADA framework eﬀectively mitigates the performance
degradation typically associated with certain sparse self-attention mechanisms. For instance, while linear self-attention [19] in standard
Transformers incurs signiﬁcant performance drops and increased computational overhead, the ADA framework maintains competitive performance while preserving computational eﬃciency. This advantage is
particularly valuable for domain adaptation tasks, where the framework
demonstrates consistent performance improvement.

Fig. 4. The relationship between EPS of DBSCAN and mAP. (Red line: test on
baseline based on ResNet50, using the ADA framework with focused linear attention.) (Blue Line: test on baseline based on IBN-ResNet50, using the ADA
framework with bi-level routing attention). (For interpretation of the references
to color in this ﬁgure legend, the reader is referred to the web version of this
article.)

4.5. Parameters analysis
In addition to employing a bi-level bottleneck structure for parameter reduction, the ADA framework implements a simple conﬁguration
strategy for sparse self-attention. When window size and projection dimension are excluded, sparse self-attention requires only the speciﬁcation of multi-head, eliminating additional parameterization.
Comparative analysis reveals distinct window size conﬁgurations
across architectures: (1) SRFormer [20] utilizes 24 × 24 windows for Permuted Self-Attention; (2) Swin Transformer [23] employs 7 × 7 windows
for Window-based Self-Attention; (3) Flatten Transformer [21] adopts
7 × 7 windows for Focused Linear Attention. In contrast, the ADA framework implements more eﬃcient window conﬁgurations ( 4 × 4, 8 × 8,
and 4 × 4 respectively) when deploying these attention variants. For Linear Self-Attention, the projection dimensions are set to 128 and 256 a more computationally economical conﬁguration than Linformer’s implementation.
While larger window sizes typically enhance performance, the ADA
framework demonstrates that optimized smaller windows (compared
to SRFormer and Flatten Transformer) can simultaneously achieve
competitive performance and computational eﬃciency. This parameter

reduction contributes to the framework’s resource-eﬀectiveness without
compromising functionality.
4.5.1. Evaluation of parameters and computational complexity
We quantitatively evaluate the parameters and computational complexity of the ADA framework. The detailed data is shown in the
Table 12. From the perspective of performance improvement, when deploying a framework after a stage in the backbone network, the added
parameters and FLOPs are relatively reasonable compared to the scale
of the backbone network.
4.5.2. The relationship between EPS and mAP
Fig. 4 illustrates the EPS-mAP relationship, with curve analysis identifying an optimal EPS window for framework performance. Following
prior baselines, we evaluated EPS from 0.1 to 1.0 in 0.1 increments.
Experiments failed with 𝐸𝑃 𝑆 ∈ [0.1, 0.2, 0.8, 0.9, 1.0] due to insuﬃcient
clustering, while 𝐸𝑃 𝑆 ∈ [0.3, 0.4, 0.5, 0.6, 0.7] enabled successful execution. Performance improved progressively from 𝐸𝑃 𝑆 ∈ [0.3, 0.4, 0.5, 0.6],
peaking at EPS=0.6.

Table 12
Parameters and FLOPs of the ADA framework with ﬁve sparse self-attention VS ResNet50 stage
1/2/3. Calculation basis: deploy the ADA framework once and stack it once.
Part of Network

layer1
layer2
layer3
Deploy Position

after layer1
after layer2
after layer3
Deploy Position

after layer1
after layer2
after layer3

baseline

Deploy Position

Params (M)

FLOPs (G)

0.22
1.22
7.1

0.88
1.35
1.92

after layer1
after layer2
after layer3

Bi-Level Routing Attention
Params (M)

FLOPs (G)

0.43
1.7
6.8

1.76
1.74
1.74

Deploy Position

after layer1
after layer2
after layer3

Focused Linear Attention
Params (M)

FLOPs (G)

0.43
1.7
6.8

1.76
1.75
1.74

Deploy Position

after layer1
after layer2
after layer3
8

Liner Self-Attention
Params (M)

FLOPs (G)

1.99
2.07
6.77

2.13
1.92
1.77

Permuted Self-Attention
Params (M)

FLOPs (G)

0.42
1.67
6.68

1.71
1.71
1.71

Window based Self-Attention
Params (M)

FLOPs (G)

0.43
1.71
6.8

1.74
1.74
1.74

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

4.6. Visualization

5. Discuss

We conducted a comparative visual analysis of attention weight distributions between the baseline and its ADA-enhanced counterpart using
the standardized test set. As shown in Fig. 5, the attention heatmaps
demonstrate that color intensity positively correlates with attention
weight magnitude, with deeper red hues indicating stronger inﬂuence
on model predictions. Our analytical observations reveal two critical
ﬁndings:

Comparative Analysis of Attention Mechanisms. Empirical studies
demonstrate that both standard self-attention and sparse self-attention
mechanisms eﬀectively enhance domain adaptation capabilities. However, sparse self-attention exhibits three distinct advantages: (1) Reduced Computational Complexity: Achieves lower FLOPs compared to
standard self-attention while maintaining equivalent domain adaptation
robustness. (2) Expanded Design Flexibility: Oﬀers a diverse selection of
design paradigms for architecture-speciﬁc optimization. (3) Enhanced
Architectural Compatibility: The ADA framework demonstrates seamless integration with both convolutional neural networks (CNNs) and
Transformer architectures, as validated through cross-backbone experiments.
Framework Limitations. Despite its demonstrated versatility, the current framework presents two constraints: (1) Transformer-Speciﬁc Deployment Challenges: Implementation on pure Transformer baselines
necessitates increased hyperparameter tuning complexity. (2) Static
Sparsity Constraints: The architecture primarily optimizes ﬁxed sparse
patterns, lacking dynamic attention mechanism support.
Technical Analysis of Multi-stage Deployment. The analysis reveals
that the multi-stack deployment(↑ 1.9 % mAP) failed to outperform its
single-stack counterpart(↑ 3.3 % mAP) due to two primary factors: (1)
the absence of cumulative domain adaptation eﬀects, and (2) suboptimal parameter optimization in the multi-stack conﬁguration. With the
rapid evolution of sparse self-attention mechanisms, it is possible to ﬂexibly decide whether to stack SAAB blocks based on practical needs. Note:
The reported results were obtained using the ADA framework incorporating Bi-Level Routing Attention.
Future Research Directions. Future work will proceed along three
coordinated dimensions: (1) Sparsity Pattern Innovation: Redesign domain adaptation primitives to enhance Transformer compatibility, particularly for modeling long-range dependencies in cross-domain scenarios.(2) Sparsity-Adaptation Quantiﬁcation: Establish a theoretical
framework correlating attention sparsity levels with domain adaptation
performance gains. (3) Cross-Scenario Validation: Extend evaluation to
challenging scenarios.

•

The baseline implementation exhibits diﬀuse attention patterns with
signiﬁcant weight allocation to non-salient image areas (Fig. 5, the
image in the second column). This suboptimal weight distribution
introduces performance degradation as irrelevant visual features receive disproportionate computational resources.
• The ADA-enhanced counterpart generates concentrated attention
clusters speciﬁcally aligned with discriminative identity features
(Fig. 5, images from the third to seventh columns). This targeted
focus mechanism enables progressive reﬁnement of feature discriminability during iterative training phases.

6. Conclusion
The proposed ADA framework eﬀectively addresses the limitations
of current mainstream domain adaptation methods through dual architectural advantages: (1) architectural compatibility and (2) positionﬂexible deployment. However, current analysis reveals that the framework achieves superior operational stability on CNN-based architectures
compared to Transformer implementations. Therefor, future research
directions include: (1) Optimizing attention sparsity patterns for Transformer architectures. (2) Sparsity-degree control.(3) Conducting crossscenario validation.
CRediT authorship contribution statement
Wei Zhang: Validation, Investigation, Writing – review & editing,
Methodology, Writing – original draft, Software; Peijun Ye: Data curation, Visualization, Formal analysis; Dihu Chen: Funding acquisition,
Project administration; Tao Su: Funding acquisition, Project administration.
Data availability
Data will be made available on request.
Declaration of competing interest
Fig. 5. Visualization images on the test set. (BLRA: bi-level routing attention;
FSA: focused linear attention; LSA: linear self-attention; PSA: permuted selfattention; WSA: window based self attention).

The authors declare that they have no known competing ﬁnancial
interests or personal relationships that could have appeared to inﬂuence
the work reported in this paper.
9

Pattern Recognition 171 (2026) 112238

W. Zhang et al.

Acknowledgments

[23] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo, Swin transformer:
hierarchical vision transformer using shifted windows, in: 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 9992–10002. https:
//doi.org/10.1109/ICCV48922.2021.00986
[24] W. Li, R. Zhao, T. Xiao, X. Wang, DeepReID: deep ﬁlter pairing neural network for
person re-identiﬁcation, in: 2014 IEEE Conference on Computer Vision and Pattern
Recognition, 2014, pp. 152–159. https://doi.org/10.1109/CVPR.2014.27
[25] L. Zheng, L. Shen, L. Tian, S. Wang, J. Bu, Q. Tian, Person re-identiﬁcation meets
image search, arXiv preprint arXiv:1502.02171 (2015).
[26] M. Ester, H.-P. Kriegel, J. Sander, X. Xu, et al., A density-based algorithm for discovering clusters in large spatial databases with noise, in: KDD, 96, 1996, pp. 226–231.
[27] Z. Zhong, L. Zheng, D. Cao, S. Li, Re-ranking person re-identiﬁcation with kreciprocal encoding, in: Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, 2017, pp. 1318–1327.
[28] G. Zhang, H. Zhang, W. Lin, A.K. Chandran, X. Jing, Camera contrast learning for
unsupervised person re-identiﬁcation, IEEE Trans. Circuits Syst. Video Technol. 33
(8) (2023) 4096–4107.
[29] L. Lan, X. Teng, J. Zhang, X. Zhang, D. Tao, Learning to Puriﬁcation for Unsupervised
Person Re-Identiﬁcation, 32, 2023, pp. 3338–3353. https://doi.org/10.1109/TIP.
2023.3278860
[30] F. Chen, N. Wang, J. Tang, P. Yan, J. Yu, Unsupervised person re-identiﬁcation via
multi-domain joint learning, Pattern Recognit. 138 (2023) 109369. https://doi.org/
10.1016/j.patcog.2023.109369
[31] T. He, L. Shen, Y. Guo, G. Ding, Z. Guo, SECRET: self-consistent pseudo label reﬁnement for unsupervised domain adaptive person re-identiﬁcation, in: Proceedings of
the AAAI Conference on Artiﬁcial Intelligence, 36, 2022, pp. 879–887.
[32] Y. Cho, W.J. Kim, S. Hong, S.-E. Yoon, Part-based pseudo label reﬁnement for unsupervised person re-identiﬁcation, in: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2022, pp. 7308–7318.
[33] S. Chen, L. Qiu, Z. Tian, Y. Yan, D.-H. Wang, S. Zhu, MTNet: mutual tri-training network for unsupervised domain adaptation on person re-identiﬁcation, J. Vis. Commun. Image Represent. 90 (2023) 103749.
[34] X. Pan, P. Luo, J. Shi, X. Tang, Two at once: enhancing learning and generalization
capacities via IBN-Net, 2018.
[35] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner,
M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., An image is worth 16 × 16
words: transformers for image recognition at scale, in: International Conference on
Learning Representations, 2020.

This study was supported in part by the Key Area R & D Program of
Guangdong Province under Grant 2022B0701180001.
References
[1] H. Yu, H. Fan, X. Chen, Q. Wang, Z. Han, POSR: pose-aligned outlier sample relabeling for unsupervised person reidentiﬁcation, IEEE Trans. Instrum. Meas. 74 (2025)
1–11. https://doi.org/10.1109/TIM.2025.3546396
[2] X. Lan, X. Zhu, S. Gong, Unsupervised cross-domain person re-identiﬁcation by
instance and distribution alignment, Pattern Recognit. 124 (2022) 108514. https:
//doi.org/10.1016/j.patcog.2021.108514
[3] J. Wang, X. Zhu, S. Gong, W. Li, Transferable joint attribute-identity deep learning
for unsupervised person re-identiﬁcation, in: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, 2018, pp. 2275–2284.
[4] L. Wei, S. Zhang, W. Gao, Q. Tian, Person transfer GAN to bridge domain gap for
person re-identiﬁcation, in: Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, 2018, pp. 79–88.
[5] Y.-J. Li, C.-S. Lin, Y.-B. Lin, Y.-C.F. Wang, Cross-dataset person re-identiﬁcation via
unsupervised pose disentanglement and adaptation, in: Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2019, pp. 7919–7929.
[6] Y. Zhu, Y. Zheng, J. Liu, Y. Li, Z.-j. Zha, Noise-resistance learning via multigranularity consistency for unsupervised domain adaptive person re-identiﬁcation,
ACM Trans. Multimedia Comput. Commun. Appl. 21 (1) (2024) 1–23.
[7] M. Li, H. Sun, C. Lin, C.-G. Li, J. Guo, The devil in the tail: cluster consolidation plus
cluster adaptive balancing loss for unsupervised person re-identiﬁcation, Pattern
Recognit. 129 (2022) 108763. https://doi.org/10.1016/j.patcog.2022.108763
[8] Q. Li, X. Peng, Y. Qiao, Q. Hao, Unsupervised person re-identiﬁcation with multilabel learning guided self-paced clustering, Pattern Recognit. 125 (2022) 108521.
https://doi.org/10.1016/j.patcog.2022.108521
[9] Y. Lu, W. Deng, Transferring discriminative knowledge via connective momentum
clustering on person re-identiﬁcation, Pattern Recognit. 126 (2022) 108569. https:
//doi.org/10.1016/j.patcog.2022.108569
[10] I. Bello, B. Zoph, A. Vaswani, J. Shlens, Q.V. Le, Attention augmented convolutional
networks, in: Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2019, pp. 3286–3295.
[11] A. Srinivas, T.-Y. Lin, N. Parmar, J. Shlens, P. Abbeel, A. Vaswani, Bottleneck transformers for visual recognition, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 16519–16529.
[12] S. Lai, Z. Chai, X. Wei, Transformer meets part model: adaptive part division for
person re-identiﬁcation, in: Proceedings of the IEEE/CVF International Conference
on Computer Vision, 2021, pp. 4150–4157.
[13] J. Li, M. Wang, X. Gong, Transformer based multi-grained features for unsupervised
person re-identiﬁcation, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, pp. 42–50.
[14] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in:
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
2016, pp. 770–778.
[15] A.G. Hauptmann, Y. Yang, L. Zheng, Person re-identiﬁcation: past, present and future (2016).
[16] A. Hermans, L. Beyer, B. Leibe, In defense of the triplet loss for person reIdentiﬁcation.
[17] M. Farina, U. Ahmad, A. Taha, H. Younes, Y. Mesbah, X. Yu, W. Pedrycz, Sparsity in
transformers: a systematic literature review, Neurocomputing (2024) 582 127468.
[18] Y. Tay, M. Dehghani, D. Bahri, D. Metzler, Eﬃcient transformers: a survey, ACM
Comput. Surv. 55 (6) (2022) 1–28.
[19] S. Wang, B.Z. Li, M. Khabsa, H. Fang, H. Ma, Linformer: self-attention with linear
complexity, 2020, arXiv:2006.04768
[20] Y. Zhou, Z. Li, C.-L. Guo, S. Bai, M.-M. Cheng, Q. Hou, SRFormer: permuted selfattention for single image super-resolution, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 12780–12791.
[21] D. Han, X. Pan, Y. Han, S. Song, G. Huang, Flatten transformer: vision transformer
using focused linear attention, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 5961–5971.
[22] L. Zhu, X. Wang, Z. Ke, W. Zhang, R.W.H. Lau, BiFormer: vision transformer with
bi-level routing attention, in: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023, pp. 10323–10333.

Tao Su was born in Yiyang, China. He received the B.Sc. degree in physics from the University of Science and Technology of China, Hefei, China, in 2000, the M.Sc. degree in
microwave engineering from the Munich University of Technology, Munich, Germany, in
2005, and the Doktor-Ingenieur degree in electronic engineering from the University of
Erlangen-Nurnberg, Erlangen, Germany, in 2009. He worked as a doctor student with Inﬁneon Technologies AG, Munich, from 2006 to 2009. He is now an associate professor in the
school of electronics and information technology at Sun Yat-sen University, Guangzhou,
China. His current research interest is the integrated circuits design and computer vision.
Wei Zhang was born in Shandong, China. He received the M.Sc. degree in Electronic and
Communication Engineering from Sun Yat-sen University, in 2007. He is a doctoral student
in the School of Electronics and Information Technology at Sun Yat-sen University and is
currently pursuing a doctoral degree of Electronics Science and Technology. His current
research interest is computer vision, domain speciﬁc accelerators.
Peijun Ye was born in Guangdong, China, received the Master degree in Microelectronics
and Solid State Electronics from Sun Yat-sen University in 2022, and now is an assistant
engineer at Peng Cheng Laboratory. His current research interestes are computer vision
and integrated circuits design.
Dihu Chen was born in Sichuan, China. He received the Ph.D. degree in electronic engineering from The Chinese University of Hong Kong, Hong Kong, China. He is now a professor in the school of electronics and information technology at Sun Yat-sen University,
Guangzhou, China. His current research interest is integrated circuit design methodology,
deep learning and image recognition technology, nano-functional materials, micro-nano
sensor devices and biochips.

10


exec
/bin/zsh -lc "pdftotext -f 8 -l 10 'Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 163 (2025) 111443

G. Zhang et al.
Table 5
Comparison with SOTA methods on VC-Clothes-W&R (%).
Methods

Modality

Cross-clothes

Same-clothes

Rank-1

mAP

Rank-1

mAP

RGB+sil
RGB+dg

66.3
70.7

63.1
65.9

92.2
91.8

91.4
90.2

MBUNet (TIP 22) [21]
AFL (TMM 23) [30]
ACID (TIP 23) [6]
DCR-ReID (TCSVT 23) [31]
MGP (TMM 23) [32]

RGB+pose
RGB
RGB
RGB+bs
RGB+dg

78.1
78.9
80.5
80.1
77.5

70.6
70.2
71.3
72.0
72.3

90.9
91.6
91.1
91.9
92.4

91.1
91.3
90.8
91.4
92.1

Ours

RGB

84.1

76.5

93.1

92.7

Short-term based methods
ISP (ECCV 20) [28]
DG-Net (CVPR 19) [29]
Cloth-changing based methods

Table 6
Comparison with SOTA methods on PRCC and LTCC (%).
Methods

Modality

PRCC

LTCC

Cross-clothes

Same-clothes

Cross-clothes

Same-clothes

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

RGB
RGB+sil
RGB

22.9
36.6
53.5

–
–
53.3

86.9
92.8
98.2

–
–
98.4

23.5
27.8
25.0

10.0
11.9
12.6

61.8
66.3
68.4

27.5
29.6
34.6

RCSAN (ICCV 21) [33]
GI-ReID (CVPR 22) [37]
FSAM (CVPR 21) [4]
MBUNet (TIP 22) [21]
Chan et al (ACM 23) [39]
AIM (CVPR23) [38]

RGB
RGB+ga
RGB+bs
RGB+pose
RGB+dg
RGB

50.2
37.6
54.5
67.6
65.8
54.7

48.6
82.3
–
65.3
61.2
55.0

100
79.0
98.8
100
99.5
100

97.2
–
–
99.6
96.7
99.9

–
28.1
38.5
39.5
32.9
38.3

–
13.2
16.2
14.7
15.3
17.0

–
73.6
–
67.1
73.4
76.1

–
36.1
–
34.4
36.8
39.1

Ours

RGB

69.1

68.7

99.8

98.1

40.1

17.4

71.9

34.7

Short-term based methods
PCB (ECCV 18) [27]
ISP (ECCV 20) [28]
MGN (ACMMM 18) [3]
Long-term based methods

Table 7
Ablation study of components in our framework in Celeb-reID, NKUP, VC-Clothes (cross clothes), VC-Clothes-W&R (cross clothes), and Celeb-reID-light (%).
Methods

Baseline
+MLDC
+LPAM
+MLDC+LPAM
+MLDC+LPAM (FFT)

Celeb-reID

NKUP

VC-Clothes

Celeb-reID-light

VC-Clothes-W&R

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

57.3
59.1
60.2
60.9
61.4

15.6
16.3
15.9
16.8
16.9

21.5
22.5
23.3
24.2
25.7

15.5
16.9
17.7
18.3
19.1

83.7
85.3
84.8
87.1
89.2

80.6
81.9
80.3
82.3
83.1

29.1
33.4
32.7
36.1
40.6

20.3
21.7
21.2
23.9
24.2

80.3
81.9
81.5
83.1
84.1

74.4
75.2
74.9
75.8
76.5

that after the introduction of the MLDC module, the performance of
all datasets has improved, especially on Celeb-reID-light, where the
accuracies of Rank-1 and mAP have increased by 4.3% and 1.4%
respectively. This shows that the MLDC module effectively improves
the model’s adaptability to clothing changes by dynamically fusing
the multi-layer information of Transformer. When the proposed LPAM
is added alone, the performance of the baseline on most datasets is
further improved, especially the Rank-1 on NKUP and Celeb-reID-light
are increased by 1.8% and 3.6% respectively, highlighting the role of
the local pyramid aggregation module in extracting multi-scale features
and capturing the key role of local information.
When MLDC and LPAM are used simultaneously, the model’s performance of all datasets is significantly improved, especially on CelebreID-light, where the accuracies of Rank-1 and mAP are increased by
7.0% and 3.6% respectively. These results show that the simultaneous
use of multi-layer information fusion and multi-scale information can
further resist the interference caused by clothing changes. Next, to
further verify the superiority of the FFT self-attention mechanism we
proposed, we compare the performance of the attention mechanism
combined with FFT and the ordinary attention mechanism in the local
feature aggregation of the LPAM module. As can be seen from this
Table, after the introduction of FFT, the performance of the model on
all datasets has been further improved, especially on the VC-ClothesW&R dataset, where Rank-1 and mAP are increased by 1.0% and

0.7% respectively. This fully demonstrates that FFT can help the model
more effectively process and identify structural details in the frequency
domain by converting data to the frequency domain. These results fully
validate the effectiveness and importance of each component in our
framework.
To intuitively demonstrate the effectiveness of each module, we visualize the experimental results on the Celeb-reID dataset, as illustrated
in Fig. 6. It can be seen that the matching results of using modules
MLDC and LPAM separately are better than the baseline model. When
these two modules are used in combination, its matching rate in Ranks
1–10 is significantly improved. Especially after the introduction of FFT
self-attention, the accuracy of the model is further improved, which is
completely consistent with our ablation experimental results.
Visualization of feature distribution. To substantiate the efficacy
of the introduced components, we employ t-SNE [40] for visualizing
the distribution of features extracted by the model across different
components, as illustrated in Fig. 7. With this figure, the circles mean
randomly selected image features from the training set of Celeb-reID
dataset, with varying colors denoting distinct identities.
Specifically, Fig. 7(a) shows the extraction distribution of features
by the baseline model ViT. It can be seen that the feature points are
relatively scattered and the degree of identity aggregation is low, which
reflects the significant challenge to the effective recognition of ViT
8

Pattern Recognition 163 (2025) 111443

G. Zhang et al.

Fig. 6. Qualitative visualization of the baseline and our modules and combinations on the Celeb-reID dataset. Note that the green boxes highlight the correct results and the red
boxes the incorrect results.

Fig. 7. Visual analysis of feature distribution on the Celeb-reID-light dataset. Circles represent sample features and different colors represent different identities. (a) The feature
distribution of the baseline model, (b) the feature distribution of the baseline model after adding the MLDC module, (c) the feature distribution of the baseline model after adding
the LPAM modules, (d) the feature distribution of the baseline model after adding the MLDC and LPAM modules, and (e) the feature distribution of the baseline model after adding
the MLDC and LPAM (FFT) modules.

models by pedestrians changing clothes. Fig. 7(b) shows the feature
distribution after introducing the MLDC module. This module significantly enhances the capture of identity information by weighted fusion
of Transformer’s multi-layer features, thereby making the boundaries
between classes more obvious. Fig. 7(c) shows the feature distribution after introducing the LPAM module in the baseline model. By
focusing on key local information and extracting multi-scale features,
this module significantly improves the model’s ability to capture and
perceive local details, making the feature distribution more aggregated
compared to the baseline. Fig. 7(d) shows the feature distribution
after fusing the MLDC and LPAM modules at the same time, in which
the feature point aggregation effect is better, which shows that the
combination of these two modules not only maintains global perception
capabilities, but also significantly enhances the recognition of details.
and processing, demonstrating superior overall performance. Fig. 7(e)
combines Fourier transform (FFT) on the basis of (d), which is the
complete method 𝐴3 𝑃 𝐹 𝑁 we proposed, and its feature aggregation
effect is the most outstanding. This fully illustrates the potential of
FFT in helping models more effectively process and identify structural
details in the frequency domain. In summary, our proposed component
demonstrates its excellent performance.
Transformer layers selection of LPAM. The ViT-Base [9] model
we selected contains 12 Transformer layers. Since the lower layers of

the Transformer mainly focus on detail features such as edges, colors,
and textures; the middle layers gradually turn to the local structure of
the image; and the higher layers further focus on the global semantic
understanding of the image. To enable the LPAM module to obtain
comprehensive feature integration, we empirically selected 3rd, 6th,
9th, and 12th layers as inputs. To demonstrate the effectiveness of
selecting these number of layers, we conduct partial experiments on the
Celeb-reID, NKUP, and VC-Clothes (cross-clothing) datasets. The results
are shown in Table 8 and we can see that our selected layers 3, 6, 9, and
12 achieve the best performance on all three datasets. This combination
covers various feature stages from elementary to advanced, provides
comprehensive feature integration for the LPAM module, and highlights
the importance of considering local to global information in feature
fusion.
Effectiveness of adaptive weights for loss functions. In order
to explore the specific effects of adaptive weighting and fixed weight
settings on the performance of the loss function, we implement five different fixed weight schemes on the NKUP dataset, including uniformly
distributing weights 𝑡1 to 𝑡4 and letting 𝑡1 to 𝑡4 bear the maximum
weights respectively. Through these representative experimental settings, we aim to evaluate the differences in the impact of different
manual weight assignments on model performance. From Table 9, it is
9

Pattern Recognition 163 (2025) 111443

G. Zhang et al.
Table 8
Comparison of ablation results on different Transformer layer selection.
Selected layers

1 2 3 4
5 6 7 8
9 10 11 12
2 4 6 8
3 6 9 12

Celeb-reID

NKUP

Declaration of competing interest

VC-Clothes

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

57.1
59.3
58.4
60.2
61.4

15.1
16.0
15.7
16.6
16.9

23.1
24.4
24.1
25.2
25.7

17.1
17.9
17.2
18.8
19.1

87.1
87.7
87.3
88.4
89.2

81.2
81.6
81.7
82.9
83.1

The authors confirm that 1. The work described is not under consideration for publication elsewhere; 2. All the necessary files have been
uploaded by online; 3. Each author has participated sufficiently; 4. All
the authors listed have approved the manuscript that is enclosed.
Acknowledgments
This research is supported by the National Natural Science Foundation of China under Grants 62172231, 92470202 and U22B2056, the
Natural Science Foundation of Jiangsu Province, China under Grant
BK20220107, the Preliminary Research Project on Leading Technologies by Wuxi Industrial Innovation Research Institute-Visual Intelligent
Analysis of Worker Behavior and Anomaly Warning, Wenzhou Key
Scientific and Technological Projects (No. ZG2024012), and the Ministry of Business Innovation and Employment 2020 Catalyst: Strategic –
New Zealand-Singapore Data Science Research Programme Fund (grant
number MAUX2002), New Zealand.

Table 9
Ablation study of manual weighting and adaptive weighting.
Weights

NKUP

𝑡1

𝑡2

𝑡3

𝑡4

Rank-1

Rank-5

mAP

0.25
0.7
0.1
0.1
0.1

0.25
0.1
0.7
0.1
0.1

0.25
0.1
0.1
0.7
0.1

0.25
0.1
0.1
0.1
0.7

21.5
22.7
24.4
23.9
23.3

31.2
33.3
34.3
33.1
33.5

17.5
18.3
18.1
17.9
18.5

25.7

35.1

19.1

Ours (adaptive weights)

Data availability
evident that the adaptive weights outperform the best manual weights
by 1.3%, 1.6% and 0.6% on Rank-1, Rank-5 and mAP respectively.
This shows that manual weight setting requires multiple debugging
to determine better weight distribution, which consumes a lot of time
and presents challenges in identifying the optimal weight. In contrast,
experimental results fully demonstrate that adaptive weighting has a
stronger ability to adapt to data and task differences.

Data will be made available on request.

References
[1] Y. Chen, H. Wang, X. Sun, B. Fan, C. Tang, H. Zeng, Deep attention aware feature
learning for person re-identification, Pattern Recognit. 126 (2022) 108567.
[2] Y. Lu, W. Deng, Transferring discriminative knowledge via connective momentum clustering on person re-identification, Pattern Recognit. 126 (2022)
108569.
[3] G. Wang, Y. Yuan, X. Chen, J. Li, X. Zhou, Learning discriminative features with
multiple granularities for person re-identification, in: Proceedings of the 26th
ACM International Conference on Multimedia, 2018, pp. 274–282.
[4] P. Hong, T. Wu, A. Wu, X. Han, W.-S. Zheng, Fine-grained shape-appearance
mutual learning for cloth-changing person re-identification, in: Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021,
pp. 10513–10522.
[5] X. Gu, H. Chang, B. Ma, S. Bai, S. Shan, X. Chen, Clothes-changing person
re-identification with rgb modality only, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2022, pp. 1060–1069.
[6] Z. Yang, X. Zhong, Z. Zhong, H. Liu, Z. Wang, S. Satoh, Win-win by competition:
Auxiliary-free cloth-changing person re-identification, IEEE Trans. Image Process.
(2023).
[7] Q. Yang, A. Wu, W.-S. Zheng, Person re-identification by contour sketch under
moderate clothing change, IEEE Trans. Pattern Anal. Mach. Intell. 43 (6) (2019)
2029–2046.
[8] Y. Yan, H. Yu, S. Li, Z. Lu, J. He, H. Zhang, R. Wang, Weakening the
influence of clothing: Universal clothing attribute disentanglement for person
re-identification, in: IJCAI, 2022, pp. 1523–1529.
[9] A. Dosovitskiy, An image is worth 16x16 words: Transformers for image
recognition at scale, 2020, arXiv preprint arXiv:2010.11929.
[10] G. Li, T. Zhao, Efficient image analysis with triple attention vision transformer,
Pattern Recognit. (2024) 110357.
[11] K. Jiang, T. Zhang, X. Liu, B. Qian, Y. Zhang, F. Wu, Cross-modality transformer for visible-infrared person re-identification, in: European Conference on
Computer Vision, Springer, 2022, pp. 480–496.
[12] P.K. Sarker, Q. Zhao, Enhanced visible–infrared person re-identification based
on cross-attention multiscale residual vision transformer, Pattern Recognit. 149
(2024) 110288.
[13] T. Wang, H. Liu, P. Song, T. Guo, W. Shi, Pose-guided feature disentangling for
occluded person re-identification based on transformer, in: Proceedings of the
AAAI Conference on Artificial Intelligence, Vol. 36, 2022, pp. 2540–2549.
[14] G. Zhang, Y. Ge, Z. Dong, H. Wang, Y. Zheng, S. Chen, Deep high-resolution
representation learning for cross-resolution person re-identification, IEEE Trans.
Image Process. 30 (2021) 8913–8925.
[15] Z. Wang, J. Jiang, Y. Wu, M. Ye, X. Bai, S. Satoh, Learning sparse and
identity-preserved hidden attributes for person re-identification, IEEE Trans.
Image Process. 29 (2019) 2013–2025.
[16] A. Verma, A.V. Subramanyam, Z. Wang, S. Satoh, R.R. Shah, Unsupervised
domain adaptation for person re-identification via individual-preserving and
environmental-switching cyclic generation, IEEE Trans. Multimed. 25 (2021)
364–377.

6. Conclusion
In this paper we have proposed a Transformer-based AdaptiveAware Attention and Pyramid Fusion Network for CC Re-ID. Our
method utilizes a Multi-layer Dynamic Concentration Module to evaluate the importance of features at different levels in real time, effectively
reducing computational redundancy and improving accuracy. In addition, our proposed Local Pyramid Aggregation Module optimizes the
extraction process of multi-scale features, focusing on critical local
information while maintaining global awareness capability. We also
combine the Fast Fourier transform with a self-attention mechanism,
aiming to enhance the ability to recognize fine pedestrian details.
Finally, we add wind and rain scenes to the existing dataset to fill the
lack of complex weather in existing pedestrian datasets.
Despite the good progress, our model still has some limitations in
real-world environments: we find that the drop in accuracy of our
method in rainy and windy scenes has no obvious advantage over
other methods. In addition, frequent occlusion and complex lighting conditions may also hinder the accurate extraction of pedestrian
identity features. Therefore, our future research will further integrate
environmental factors such as occlusion and low light, and introduce
more modal information such as depth map, thermal imaging or visual–
inertial sensor data to help the model enhance its adaptability and
robustness in various complex real-world scenarios.
CRediT authorship contribution statement
Guoqing Zhang: Writing – review & editing, Writing – original
draft, Validation, Resources, Methodology, Funding acquisition, Conceptualization. Jieqiong Zhou: Writing – original draft, Visualization,
Validation. Yuhui Zheng: Supervision, Funding acquisition. Gaven
Martin: Writing – review & editing, Supervision. Ruili Wang: Writing
– review & editing, Supervision.
Ethics approval
I have read and have abided by the statement of ethical standards
for manuscripts submitted to the Journal of Pattern Recognition.
10


web search: 
web search: "IRRA" "Cross-modal Implicit Relation Reasoning" "Text-to-Image Person Retrieval"
web search: 
web search: "Propot" "CLIP-derived identity prototypes" text-based person retrieval
web search: 
web search: "Occlude Them All" "Occlusion-Aware Attention Network" "occluded person re-ID"
web search: 
web search: "Sparse self-Attention augmented Domain Adaptation" "person re-identification"
web search: 
web search: "Clothes-changing person re-identification with RGB modality only" CAL
web search: 
web search: "ViTAA" "Text-Image Person Search" attributes alignments
codex
我已经读完 6 篇的方法段和关键消融页。现在进入整理阶段，会按你给的 5 个问题逐篇写，不会把它写成“模块清单”。
  ✓ 抽取 6 篇 PDF 的前 8 页文本，定位完整方法段落。
  ✓ 必要时查网页确认题名、年份、先例和方法定位。
  → 逐篇按 5 个问题写方法论拆解，并总结共同套路。
web search: 
web search: IRRA Cross-modal Implicit Relation Reasoning Text-to-Image Person Retrieval
web search: 
web search: "VITAA" "Text-Image Person Search by Natural Language"
web search: 
web search: "Occlude Them All: Occlusion-Aware Attention Network for Occluded Person Re-ID"
web search: 
web search: "Occlusion-Aware Attention Network" "Occluded Person Re-ID"
web search: 
web search: "Dual Contrastive Learning" "Unsupervised Visible-Infrared Person Re-Identification" DCL
  ✓ 抽取 6 篇 PDF 的前 8 页文本，定位完整方法段落。
  ✓ 必要时查网页确认题名、年份、先例和方法定位。
  ✓ 逐篇按 5 个问题写方法论拆解，并总结共同套路。
codex
下面按“创新怎么构造”拆，不按模块复述。

**1. A training-free framework for text-to-image person re-identification via query-prototype matching**

1. **触发观察**：作者抓住的是 Text-to-Image ReID 的一个错位，查询是文本，检索目标却是“同一个身份的所有图像”。现有方法大多仍用文本和单张图库图像做实例级相似度，作者称之为 instance-identity mismatch。另一个观察是，TI-ReID 里的文本描述很模板化，文本嵌入容易挤成团，而图库图像特征反而有更清楚的身份簇结构。

2. **重定义动作**：大家以为问题主要是“文本和图像对齐不够好”，作者改写成“实例级文本匹配不符合身份级检索目标”。关键词是 identity-level structure、Query-Prototype Similarity、image-side identity anchors、training-free refinement。它不是说再训一个更强对齐模型，而是说图库本身已经有身份结构，应该在推理时被用作先验。

3. **机制怎么长出来**：机制几乎是从重定义自然推出的。既然图像侧身份结构比文本侧稳定，就把图库图像聚类成 prototype；既然单张图像匹配容易被 hard positive 或 hard negative 干扰，就把 Query-Gallery 相似度和 Query-Prototype 相似度融合；既然聚类粒度难固定，就加多阶段 PDE 和 ACE 自动估计 prototype 数。这个机制和问题绑定得很紧，核心不是“聚类新”，而是“推理时用图像身份原型校准跨模态实例分数”。

4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。

5. **reviewer 为什么买账**：这篇卖的是视角，不是模块。聚类、原型、重排序都不新，但它把 TI-ReID 的核心矛盾讲成“身份级检索目标和实例级匹配分数不一致”，然后用训练无关的推理校准解决。真正的 novelty 是问题重定义加使用位置，原型从训练期监督转成推理期身份锚点。

**2. A2HA: Attribute-aware hierarchical alignment for text-image person re-identification**

1. **触发观察**：作者观察到两层失败。第一，global image-text matching 太粗，分不开外观相似但细节属性不同的人。第二，以前的 attribute-aware 方法也不彻底，常常只抽文本属性，再和全局图像特征匹配，粒度不对；或者把属性当辅助监督，最后被身份分类和实例对齐压住。

2. **重定义动作**：它把“跨模态对齐”改写成“多语义层级的对齐失衡”。大家以为加属性就够了，作者说不是，属性必须成为核心优化目标，而且要同时处理 instance-level、identity-level、attribute-level。关键词是 hierarchical alignment、coarse-to-fine、dual-dimensional、dual-directional、attribute-level false positive 和 false negative。

3. **机制怎么长出来**：既然问题是属性粒度不匹配，就先用 Grounded SAM 得到图像属性区域，用 ChatGPT 抽文本属性，构造 attribute-enriched datasets。既然属性对齐不能只是文本到图像单向，就把属性样本对分成跨模态和模态内两维，再按相似度阈值分成 12 种情况和 6 类样本对。既然问题样本是属性级 false positive 和 false negative，就用属性级 contrastive、triplet、动态权重去打这些样本。逻辑是完整的，但代价是框架很重。

4. **证据闭环**：最关键的是 Table 3，它把“更好的属性数据”和“层级对齐机制”拆开验证：在 ViTAA 上替换属性集有增益，加入 HA 也有增益，两者组合更高。Table 5 和 Table 6 证明 ALA 各损失、GLA 和 ALA 组合确实有效；Table 7 证明 attribute-level 样本划分优于 instance-level 和 identity-level；Table 10、Table 11 和可视化证明属性标注本身质量可用。这些证据是在证明“属性级对齐是独立有效的”，不是只证明总模块涨点。

5. **reviewer 为什么买账**：这篇卖的是“视角加资源”。视角是把属性从辅助信息抬成核心优化层级，资源是三个 attribute-enriched benchmark。机制本身有堆料感，但它用“属性误匹配类型”把堆料组织成一个新问题，因此比简单加属性分支更容易过审。

**3. ADA framework for unsupervised domain adaptation person re-identification**

1. **触发观察**：作者从 UDA ReID 三类主流方法的痛点切入：特征对齐常要辅助信息，图像风格迁移依赖 GAN 且训练不稳，聚类伪标签会传播错误。另一个观察是，卷积的局部感受野不足以处理跨域干扰，标准 self-attention 又贵且常要改 backbone。

2. **重定义动作**：它把“域适应靠伪标签或分布对齐”改写成“域适应也可以由一个轻量架构原语提供”。关键词是 Sandwich Attention Primitive、SAAB block、sparse self-attention augmented domain adaptation、plug-and-play、deployment flexibility。作者的语言重点不是“我们修伪标签”，而是“我们提供一个能插在不同网络层级的适应原语”。

3. **机制怎么长出来**：机制是把 ResNet bottleneck 里的局部卷积替换成一个“卷积分支加稀疏注意力分支”的 SAP，再用 SAAB block 包起来，插在 backbone 不同 stage 后面。稀疏注意力有五种可替换实现，训练仍然用 DBSCAN 伪标签、分类损失和 triplet loss。这个机制和“跨域需要长程身份线索、抑制背景干扰”有关系，但绑定不算特别紧，更多是把已有 attention augmented convolution 思路迁移到 UDA ReID。

4. **证据闭环**：它的证据主要证明“泛化性”和“插件价值”。五种稀疏注意力在多个数据迁移组合上都能涨；把稀疏注意力换成普通 3×3 卷积会掉；换 IBN-ResNet50 和 ViT 仍有效；不同插入位置有系统比较；复杂度表说明开销可接受；热力图显示注意力更集中在人身区域。这个闭环能证明“ADA 是个通用可插模块”，但对“为什么就是域不变身份特征”证明相对弱。

5. **reviewer 为什么买账**：这篇卖的是机制和工程覆盖面，不是特别强的新视角。它的审稿说服力来自大范围 ablation 和多配置一致性。对我们来说，这类文章提醒一点：如果只是“新 attention 插件”，必须靠非常广的消融和部署灵活性撑住；否则容易被看成架构拼装。

**4. Adaptive Occlusion-Aware Network for Occluded Person Re-Identification**

1. **触发观察**：作者抓住遮挡 ReID 里两种局部特征方案的失败。人工水平切块会因为尺度和姿态导致部位错位；姿态估计能缓解错位，但对噪声敏感，还会忽略包、伞、手提物这类非身体但有身份辨识度的线索，而且引入额外模型复杂度。

2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。

3. **机制怎么长出来**：既然固定切块错位，就用 learnable part catchers 自己找头、上身、大腿、小腿。既然姿态估计有用但不可靠，就用 PifPaf 产生训练标签，但 APE 通过 cross-attention、self-attention 和浅层 FPN 特征学出自己的区域图。既然遮挡区域不该等权比较，就在测试时用 OAM 根据 part map 面积计算可见权重，两张图比较时取对应部位较小权重，压低遮挡部位的距离贡献。

4. **证据闭环**：关键证据很直接。APE 单独带来大幅提升，在 Occluded-Duke 上 mAP 和 Rank-1 分别涨 18.4 和 16.9；OAM 再带来 2.6 和 2.9。部位数实验说明 4 个部位最合理，太多会冗余，6 个会忽略左右区分。最有说服力的是可视化，APE 在姿态标签错误时仍能抽对区域，能避开伞遮住的头部，还能抓住姿态估计忽略的手提包。这是在证明“自适应区域优于姿态输出”，不是只证明最终分数涨。

5. **reviewer 为什么买账**：这篇卖的是机制，也有一个清楚视角：姿态估计不是最终答案，而是带噪 teacher。它的新意不在“用部位”或“用遮挡权重”，而在把部位定位和遮挡权重都从图像本身自适应学出来，并用可视化证明它比姿态标签更贴合 ReID。

**5. Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification**

1. **触发观察**：作者把 USVI-ReID 的伪标签噪声拆成三个具体后果：伪标签初始化的 memory bank 不可靠；错误伪标签会建立错误跨模态对应；对比学习里的负样本可能是假负样本，形成 sampling bias。这个观察比“伪标签有噪声”更细，因为每个后果都能对应一个机制和一组证据。

2. **重定义动作**：它把“过滤或降权噪声伪标签”改写成“伪标签需要 purification and debiasing 的完整流程”。关键词是 calibration-then-division、clean subset、noisy subset、Optimal Transport Prototype Matching、Debiased Memory Hybrid Learning、Neighbor Relation Learning。它不是把所有样本丢进同一个损失，而是把干净标签和噪声标签分流处理。

3. **机制怎么长出来**：既然 cluster 内有可靠和不可靠样本，就用 κ-reciprocal Jaccard 相似度和 similarity counter 找可靠样本建 prototype，再校准伪标签。既然校准后仍有噪声，就用线性 feature-label 模型和稀疏异常项把样本分成 clean 与 noisy。既然跨模态 cluster 数不一致，就用 optimal transport 做 cluster-level many-to-many matching。既然 clean 里仍有假负样本风险，就用 debiased contrastive learning 修正负项。既然 noisy 样本不能浪费，就用 NRL 不依赖伪标签地学邻域关系。

4. **证据闭环**：这篇的证据闭环是六篇里最像“问题分解验证”的。APCD 提升伪标签 ARI，clean subset 的伪标签准确率显著高于全量数据；APCD 还提升跨模态对应准确率。OTPM 用同样 prototype 和 PGM 比，四个聚类指标都更好。DMHL 加 OTPM 相比 baseline 大幅提升，证明跨模态对应和去偏 memory 有效。距离分布、t-SNE 和检索可视化进一步证明同身份跨模态更近、异身份更远。这些证据都对应开头的三个负面后果。

5. **reviewer 为什么买账**：这篇卖的是视角化的问题分解。伪标签校正、OT、memory、邻域学习都不是完全新，但作者把“噪声伪标签伤害模型”的泛泛说法拆成可测的三类机制，再逐个闭环。真正 novelty 来自流程组织和干净、噪声两路学习，而不是某一个单独模块。

**6. Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification**

1. **触发观察**：作者抓住换衣 ReID 的两个常见依赖：一类靠姿态、轮廓、步态等生物线索，遇到遮挡、低光、质量差会不稳；一类做衣服与身份解耦，可能引入额外计算、失真和训练不稳定。另一个关键观察是 ViT 不同层的 attention map 关注内容明显不同，但很多 ReID 模型只用最后一层。

2. **重定义动作**：它把“换衣必须引入外部生物特征或做衣服解耦”改写成“RGB 图像内部的 Transformer 层级和频域结构已经含有身份线索，只是没被充分聚合”。关键词是 auxiliary-free、multi-layer dynamic concentration、local pyramid aggregation、FFT self-attention、frequency-domain structural details。天气数据集 VC-Clothes-W&R 又把问题扩展成更自然的换衣场景。

3. **机制怎么长出来**：既然不同 Transformer 层看不同内容，就用 MLDC 给 12 层动态加权，并用层间相关惩罚减少冗余。既然全局层融合会丢局部，就用 LPAM 选第 3、6、9、12 层做金字塔式局部融合。既然 gait 和结构细节可能体现在频域，就把 FFT 放进 self-attention 里做频域相关。训练上只用 ID loss、triplet loss 和自适应 loss 权重。机制和“多层多尺度”绑定较紧，和“FFT 捕捉步态结构”的绑定稍弱，需要更多专门证据。

4. **证据闭环**：Table 7 证明 MLDC、LPAM、两者组合和 FFT 逐级带来增益，而且在多个换衣数据集上都有趋势。层选择实验说明 3、6、9、12 覆盖低中高层最优；自适应 loss 权重优于手调固定权重；t-SNE 和检索可视化显示同身份聚得更紧。VC-Clothes-W&R 上也有 SOTA 结果，但作者自己承认雨风场景下降幅度不一定比别人小，所以“天气鲁棒”不是最强闭环，“RGB-only 多层多尺度换衣识别”才是主闭环。

5. **reviewer 为什么买账**：这篇卖的是机制和数据集，视角中等。它比普通模块文强的地方是把“外部生物线索依赖”转成“Transformer 内部层级身份线索复用”；弱点是多层融合、金字塔、FFT 都是常见工程语言，需要靠跨数据集消融和新数据集来撑住 novelty。

**共同套路**

这批论文共同套路不是先想一个模块，而是先找一个“任务定义和现有做法之间的错位”：身份级目标却做实例级匹配，属性级差异却只做全局对齐，伪标签噪声却只当成一个整体，遮挡局部却依赖固定切块或姿态输出。然后作者把错位改写成新问题，再让机制像自然后果一样长出来。最能打动 reviewer 的证据也不是总表涨点，而是证明这个重定义本身成立的实验，比如原型距离统计、属性划分对照、伪标签 ARI、APE 超过姿态标签的可视化。

联网辅助核对过的相关先例包括 IRRA、APTM、ViTAA、RPNR、CAL、BPBreID、PGFA、Attention Augmented Convolution 等，主要用于判断这些论文是在复用已有零件，还是改变了问题讲法。参见：IRRA citeturn9search0，APTM citeturn10search2，ViTAA citeturn10search0，RPNR citeturn9search2，CAL citeturn11search0，BPBreID citeturn12search1，PGFA citeturn13search2，Attention Augmented Convolution citeturn12search3。
tokens used
202,239
下面按“创新怎么构造”拆，不按模块复述。

**1. A training-free framework for text-to-image person re-identification via query-prototype matching**

1. **触发观察**：作者抓住的是 Text-to-Image ReID 的一个错位，查询是文本，检索目标却是“同一个身份的所有图像”。现有方法大多仍用文本和单张图库图像做实例级相似度，作者称之为 instance-identity mismatch。另一个观察是，TI-ReID 里的文本描述很模板化，文本嵌入容易挤成团，而图库图像特征反而有更清楚的身份簇结构。

2. **重定义动作**：大家以为问题主要是“文本和图像对齐不够好”，作者改写成“实例级文本匹配不符合身份级检索目标”。关键词是 identity-level structure、Query-Prototype Similarity、image-side identity anchors、training-free refinement。它不是说再训一个更强对齐模型，而是说图库本身已经有身份结构，应该在推理时被用作先验。

3. **机制怎么长出来**：机制几乎是从重定义自然推出的。既然图像侧身份结构比文本侧稳定，就把图库图像聚类成 prototype；既然单张图像匹配容易被 hard positive 或 hard negative 干扰，就把 Query-Gallery 相似度和 Query-Prototype 相似度融合；既然聚类粒度难固定，就加多阶段 PDE 和 ACE 自动估计 prototype 数。这个机制和问题绑定得很紧，核心不是“聚类新”，而是“推理时用图像身份原型校准跨模态实例分数”。

4. **证据闭环**：最关键的证据不是 SOTA 表，而是作者比较 Text-to-Image 和 Text-to-Prototype 的距离统计。T2P 能降低同身份距离、增大异身份距离，说明 prototype 确实更身份稳定；但 T2P 的 Rank-1 不总是超过 T2I，所以作者用“融合”而不是“替代”，这就把机制选择讲圆了。后续 ACE 的估计值接近手调值、cluster purity 和 ID fragmentation、与 k-reciprocal、RWR、CSLS 的公平比较，以及“Prototype-in-Training 反而掉点”都在证明：原型更适合作推理期身份先验，而不是训练监督。

5. **reviewer 为什么买账**：这篇卖的是视角，不是模块。聚类、原型、重排序都不新，但它把 TI-ReID 的核心矛盾讲成“身份级检索目标和实例级匹配分数不一致”，然后用训练无关的推理校准解决。真正的 novelty 是问题重定义加使用位置，原型从训练期监督转成推理期身份锚点。

**2. A2HA: Attribute-aware hierarchical alignment for text-image person re-identification**

1. **触发观察**：作者观察到两层失败。第一，global image-text matching 太粗，分不开外观相似但细节属性不同的人。第二，以前的 attribute-aware 方法也不彻底，常常只抽文本属性，再和全局图像特征匹配，粒度不对；或者把属性当辅助监督，最后被身份分类和实例对齐压住。

2. **重定义动作**：它把“跨模态对齐”改写成“多语义层级的对齐失衡”。大家以为加属性就够了，作者说不是，属性必须成为核心优化目标，而且要同时处理 instance-level、identity-level、attribute-level。关键词是 hierarchical alignment、coarse-to-fine、dual-dimensional、dual-directional、attribute-level false positive 和 false negative。

3. **机制怎么长出来**：既然问题是属性粒度不匹配，就先用 Grounded SAM 得到图像属性区域，用 ChatGPT 抽文本属性，构造 attribute-enriched datasets。既然属性对齐不能只是文本到图像单向，就把属性样本对分成跨模态和模态内两维，再按相似度阈值分成 12 种情况和 6 类样本对。既然问题样本是属性级 false positive 和 false negative，就用属性级 contrastive、triplet、动态权重去打这些样本。逻辑是完整的，但代价是框架很重。

4. **证据闭环**：最关键的是 Table 3，它把“更好的属性数据”和“层级对齐机制”拆开验证：在 ViTAA 上替换属性集有增益，加入 HA 也有增益，两者组合更高。Table 5 和 Table 6 证明 ALA 各损失、GLA 和 ALA 组合确实有效；Table 7 证明 attribute-level 样本划分优于 instance-level 和 identity-level；Table 10、Table 11 和可视化证明属性标注本身质量可用。这些证据是在证明“属性级对齐是独立有效的”，不是只证明总模块涨点。

5. **reviewer 为什么买账**：这篇卖的是“视角加资源”。视角是把属性从辅助信息抬成核心优化层级，资源是三个 attribute-enriched benchmark。机制本身有堆料感，但它用“属性误匹配类型”把堆料组织成一个新问题，因此比简单加属性分支更容易过审。

**3. ADA framework for unsupervised domain adaptation person re-identification**

1. **触发观察**：作者从 UDA ReID 三类主流方法的痛点切入：特征对齐常要辅助信息，图像风格迁移依赖 GAN 且训练不稳，聚类伪标签会传播错误。另一个观察是，卷积的局部感受野不足以处理跨域干扰，标准 self-attention 又贵且常要改 backbone。

2. **重定义动作**：它把“域适应靠伪标签或分布对齐”改写成“域适应也可以由一个轻量架构原语提供”。关键词是 Sandwich Attention Primitive、SAAB block、sparse self-attention augmented domain adaptation、plug-and-play、deployment flexibility。作者的语言重点不是“我们修伪标签”，而是“我们提供一个能插在不同网络层级的适应原语”。

3. **机制怎么长出来**：机制是把 ResNet bottleneck 里的局部卷积替换成一个“卷积分支加稀疏注意力分支”的 SAP，再用 SAAB block 包起来，插在 backbone 不同 stage 后面。稀疏注意力有五种可替换实现，训练仍然用 DBSCAN 伪标签、分类损失和 triplet loss。这个机制和“跨域需要长程身份线索、抑制背景干扰”有关系，但绑定不算特别紧，更多是把已有 attention augmented convolution 思路迁移到 UDA ReID。

4. **证据闭环**：它的证据主要证明“泛化性”和“插件价值”。五种稀疏注意力在多个数据迁移组合上都能涨；把稀疏注意力换成普通 3×3 卷积会掉；换 IBN-ResNet50 和 ViT 仍有效；不同插入位置有系统比较；复杂度表说明开销可接受；热力图显示注意力更集中在人身区域。这个闭环能证明“ADA 是个通用可插模块”，但对“为什么就是域不变身份特征”证明相对弱。

5. **reviewer 为什么买账**：这篇卖的是机制和工程覆盖面，不是特别强的新视角。它的审稿说服力来自大范围 ablation 和多配置一致性。对我们来说，这类文章提醒一点：如果只是“新 attention 插件”，必须靠非常广的消融和部署灵活性撑住；否则容易被看成架构拼装。

**4. Adaptive Occlusion-Aware Network for Occluded Person Re-Identification**

1. **触发观察**：作者抓住遮挡 ReID 里两种局部特征方案的失败。人工水平切块会因为尺度和姿态导致部位错位；姿态估计能缓解错位，但对噪声敏感，还会忽略包、伞、手提物这类非身体但有身份辨识度的线索，而且引入额外模型复杂度。

2. **重定义动作**：它把“遮挡 ReID 需要姿态对齐”改写成“局部区域应该是自适应且遮挡感知的，姿态只能作为弱监督，不应成为测试期依赖”。关键词是 Adaptive Position Extractor、Occlusion Awareness Module、adaptive body positions、suppress occluded parts、highlight non-occluded parts。

3. **机制怎么长出来**：既然固定切块错位，就用 learnable part catchers 自己找头、上身、大腿、小腿。既然姿态估计有用但不可靠，就用 PifPaf 产生训练标签，但 APE 通过 cross-attention、self-attention 和浅层 FPN 特征学出自己的区域图。既然遮挡区域不该等权比较，就在测试时用 OAM 根据 part map 面积计算可见权重，两张图比较时取对应部位较小权重，压低遮挡部位的距离贡献。

4. **证据闭环**：关键证据很直接。APE 单独带来大幅提升，在 Occluded-Duke 上 mAP 和 Rank-1 分别涨 18.4 和 16.9；OAM 再带来 2.6 和 2.9。部位数实验说明 4 个部位最合理，太多会冗余，6 个会忽略左右区分。最有说服力的是可视化，APE 在姿态标签错误时仍能抽对区域，能避开伞遮住的头部，还能抓住姿态估计忽略的手提包。这是在证明“自适应区域优于姿态输出”，不是只证明最终分数涨。

5. **reviewer 为什么买账**：这篇卖的是机制，也有一个清楚视角：姿态估计不是最终答案，而是带噪 teacher。它的新意不在“用部位”或“用遮挡权重”，而在把部位定位和遮挡权重都从图像本身自适应学出来，并用可视化证明它比姿态标签更贴合 ReID。

**5. Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification**

1. **触发观察**：作者把 USVI-ReID 的伪标签噪声拆成三个具体后果：伪标签初始化的 memory bank 不可靠；错误伪标签会建立错误跨模态对应；对比学习里的负样本可能是假负样本，形成 sampling bias。这个观察比“伪标签有噪声”更细，因为每个后果都能对应一个机制和一组证据。

2. **重定义动作**：它把“过滤或降权噪声伪标签”改写成“伪标签需要 purification and debiasing 的完整流程”。关键词是 calibration-then-division、clean subset、noisy subset、Optimal Transport Prototype Matching、Debiased Memory Hybrid Learning、Neighbor Relation Learning。它不是把所有样本丢进同一个损失，而是把干净标签和噪声标签分流处理。

3. **机制怎么长出来**：既然 cluster 内有可靠和不可靠样本，就用 κ-reciprocal Jaccard 相似度和 similarity counter 找可靠样本建 prototype，再校准伪标签。既然校准后仍有噪声，就用线性 feature-label 模型和稀疏异常项把样本分成 clean 与 noisy。既然跨模态 cluster 数不一致，就用 optimal transport 做 cluster-level many-to-many matching。既然 clean 里仍有假负样本风险，就用 debiased contrastive learning 修正负项。既然 noisy 样本不能浪费，就用 NRL 不依赖伪标签地学邻域关系。

4. **证据闭环**：这篇的证据闭环是六篇里最像“问题分解验证”的。APCD 提升伪标签 ARI，clean subset 的伪标签准确率显著高于全量数据；APCD 还提升跨模态对应准确率。OTPM 用同样 prototype 和 PGM 比，四个聚类指标都更好。DMHL 加 OTPM 相比 baseline 大幅提升，证明跨模态对应和去偏 memory 有效。距离分布、t-SNE 和检索可视化进一步证明同身份跨模态更近、异身份更远。这些证据都对应开头的三个负面后果。

5. **reviewer 为什么买账**：这篇卖的是视角化的问题分解。伪标签校正、OT、memory、邻域学习都不是完全新，但作者把“噪声伪标签伤害模型”的泛泛说法拆成可测的三类机制，再逐个闭环。真正 novelty 来自流程组织和干净、噪声两路学习，而不是某一个单独模块。

**6. Adaptive transformer with Pyramid Fusion for cloth-changing Person Re-Identification**

1. **触发观察**：作者抓住换衣 ReID 的两个常见依赖：一类靠姿态、轮廓、步态等生物线索，遇到遮挡、低光、质量差会不稳；一类做衣服与身份解耦，可能引入额外计算、失真和训练不稳定。另一个关键观察是 ViT 不同层的 attention map 关注内容明显不同，但很多 ReID 模型只用最后一层。

2. **重定义动作**：它把“换衣必须引入外部生物特征或做衣服解耦”改写成“RGB 图像内部的 Transformer 层级和频域结构已经含有身份线索，只是没被充分聚合”。关键词是 auxiliary-free、multi-layer dynamic concentration、local pyramid aggregation、FFT self-attention、frequency-domain structural details。天气数据集 VC-Clothes-W&R 又把问题扩展成更自然的换衣场景。

3. **机制怎么长出来**：既然不同 Transformer 层看不同内容，就用 MLDC 给 12 层动态加权，并用层间相关惩罚减少冗余。既然全局层融合会丢局部，就用 LPAM 选第 3、6、9、12 层做金字塔式局部融合。既然 gait 和结构细节可能体现在频域，就把 FFT 放进 self-attention 里做频域相关。训练上只用 ID loss、triplet loss 和自适应 loss 权重。机制和“多层多尺度”绑定较紧，和“FFT 捕捉步态结构”的绑定稍弱，需要更多专门证据。

4. **证据闭环**：Table 7 证明 MLDC、LPAM、两者组合和 FFT 逐级带来增益，而且在多个换衣数据集上都有趋势。层选择实验说明 3、6、9、12 覆盖低中高层最优；自适应 loss 权重优于手调固定权重；t-SNE 和检索可视化显示同身份聚得更紧。VC-Clothes-W&R 上也有 SOTA 结果，但作者自己承认雨风场景下降幅度不一定比别人小，所以“天气鲁棒”不是最强闭环，“RGB-only 多层多尺度换衣识别”才是主闭环。

5. **reviewer 为什么买账**：这篇卖的是机制和数据集，视角中等。它比普通模块文强的地方是把“外部生物线索依赖”转成“Transformer 内部层级身份线索复用”；弱点是多层融合、金字塔、FFT 都是常见工程语言，需要靠跨数据集消融和新数据集来撑住 novelty。

**共同套路**

这批论文共同套路不是先想一个模块，而是先找一个“任务定义和现有做法之间的错位”：身份级目标却做实例级匹配，属性级差异却只做全局对齐，伪标签噪声却只当成一个整体，遮挡局部却依赖固定切块或姿态输出。然后作者把错位改写成新问题，再让机制像自然后果一样长出来。最能打动 reviewer 的证据也不是总表涨点，而是证明这个重定义本身成立的实验，比如原型距离统计、属性划分对照、伪标签 ARI、APE 超过姿态标签的可视化。

联网辅助核对过的相关先例包括 IRRA、APTM、ViTAA、RPNR、CAL、BPBreID、PGFA、Attention Augmented Convolution 等，主要用于判断这些论文是在复用已有零件，还是改变了问题讲法。参见：IRRA citeturn9search0，APTM citeturn10search2，ViTAA citeturn10search0，RPNR citeturn9search2，CAL citeturn11search0，BPBreID citeturn12search1，PGFA citeturn13search2，Attention Augmented Convolution citeturn12search3。
