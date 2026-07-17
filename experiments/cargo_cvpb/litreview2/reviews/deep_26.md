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
session id: 019ef836-0fa1-70d2-bec4-02fc8933f583
--------
user
你是 ReID 创新方法论拆解员 26。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(26)==
TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf
Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
TP-LReID - Lifelong person re-identification using text prompts.pdf
Two-stage knowledge distillation for visible-infrared person re-identification.pdf
UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我先确认当前目录里的文件和可用文本抽取工具，然后按你要求用 `pdftotext -f 1 -l 8` 逐篇读方法部分。项目要求的 `GOAL.md` 在当前论文库目录外，我会尝试读取；如果沙箱不允许，就只基于你这条任务继续。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pwd && rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'STATUS.md' -g 'AGENTS.md'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title

hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc 'command -v pdftotext && ls -1 *.pdf' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/opt/homebrew/bin/pdftotext
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

codex
当前沙箱只显示论文库本目录文件，没有直接看到 `GOAL.md` 等项目文件。我会再尝试按相对路径读取一次；如果失败，说明当前权限只允许读论文库。
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
我已经读到了 `GOAL.md`、`CLAUDE.md` 和 `experiments/STATUS.md`，当前任务和目标一致：读文献拆方法论，不启动实验。下面开始逐篇抽取正文方法部分，我会重点看作者怎样从观察过渡到问题重定义，而不是只记模块名。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
arXiv:2509.16635v2 [cs.CV] 30 May 2026

Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification
Xulin Li1,2 , Yan Lu3 , Bin Liu1,2 ,* Jiaze Li1,2 , Qinhong Yang1,2 ,
Tao Gong1,2 , Qi Chu1,2 , Mang Ye4 , Nenghai Yu1,2
1
School of Cyber Science and Technology, University of Science and Technology of China
2
Anhui Province Key Laboratory of Digital Security
3
The Chinese University of Hong Kong
4
School of Computer Science, Wuhan University, China
lxlkw@mail.ustc.edu.cn, yanlu@cuhk.edu.hk, flowice@ustc.edu.cn, jz li@mail.ustc.edu.cn,
qhyang233@mail.ustc.edu.cn, {tgong,qchu}@ustc.edu.cn, yemang@whu.edu.cn, ynh@ustc.edu.cn

Abstract
In real applications, person re-identification (ReID) is
expected to retrieve the target person at any time, including both daytime and nighttime, ranging from short-term to
long-term. However, existing ReID tasks and datasets cannot meet this requirement, as they are constrained by available time and only provide training and evaluation for specific scenarios. Therefore, we investigate a new task called
Anytime Person Re-identification (AT-ReID), which aims to
achieve effective retrieval in multiple scenarios based on
variations in time. To address the AT-ReID problem, we collect the first large-scale dataset, AT-USTC, which contains
403k images of individuals wearing multiple clothes captured by RGB and IR cameras. Our data collection spans
21 months, and 270 volunteers were photographed on average 29.1 times across different dates or scenes, 4-15 times
more than current datasets, providing conditions for followup investigations in AT-ReID. Further, to tackle the new
challenge of multi-scenario retrieval, we propose a unified
model named Uni-AT, which comprises a multi-scenario
ReID (MS-ReID) framework for scenario-specific features
learning, a Mixture-of-Attribute-Experts (MoAE) module
to alleviate inter-scenario interference, and a Hierarchical Dynamic Weighting (HDW) strategy to ensure balanced
training across all scenarios. Extensive experiments show
that our model leads to satisfactory results and exhibits excellent generalization to all scenarios. Our dataset and
code are available at https://github.com/kw66/AT-ReID.

1. Introduction
Person re-identification (ReID) aims to retrieve specific
pedestrians with given query images. As illustrated in
* Corresponding author.

Scenario 1
Day

Scenario 2

Scenario 5

Scenario 6

Spring

Winter

Scenario 4
Night

Scenario 3

Figure 1. (a) AT-ReID aims to perform retrieval at any time, including both daytime and nighttime, ranging from short-term to
long-term.

Fig. 1, a robust ReID system is expected to retrieve a person
at any time, including daytime and nighttime, ranging from
short-term to long-term, thereby satisfying the requirements
of different surveillance scenarios. This puts more challenges on the ReID system because the capturing time of
the query image and the target image makes the task more
variable. For instance, if two images are captured during
daytime and nighttime, respectively, they will have different
modalities, and when there is a long time interval between
their capturing, the person’s appearance may change due to
alterations in clothing. Consequently, traditional ReID (TrReID) [58] may not perform effectively.
The researchers acknowledged this challenge and attempted to address these problems separately. They in-

troduced the Visible-Infrared Cross-Modality ReID (CMReID) [47] to address the issue of searching between daytime RGB images and nighttime infrared (IR) images, and
the Long-Term Cloth-Changing ReID (CC-ReID) [50] was
proposed to handle long-term retrieval in which pedestrians
change their clothes. However, existing methods designed
for these specific tasks were only able to achieve success
in one of them and incapable of retrieving targets at any
time simultaneously. This situation primarily arises from
the absence of a long-term visible-infrared dataset covering all scenarios in Fig. 2 (a), which should encompass
diverse variations in clothing and modality for each individual. The deficiency in intra-identity diversity of modalities and clothing in current ReID datasets has led to research gaps, especially in Nighttime Long-term (NT-LT)
and All-day Long-term (AD-LT) scenarios. Another issue
arises from the poor generalization of task-specific methods
in non-target scenarios. This is attributed to the differing
learning objectives across different scenarios. For instance,
prior research [29, 35] has indicated that RGB-specific cues
and clothing information are harmful to the All-day Shortterm (AD-ST, CM-ReID) and Daytime Long-term (DT-LT,
CC-ReID) scenarios, respectively, while they are crucial for
the Daytime Short-term (DT-ST, Tr-ReID) scenario.
To meet the requirements of retrieving persons at any
time, we investigate a new task called Anytime Person Reidentification (AT-ReID) and propose to focus on its exploration from dataset to model level, as depicted in Fig. 2 (b).
We collect the first corresponding large-scale dataset named
AT-USTC, which contains 403k images of 270 volunteers
and covers all six scenarios in AT-ReID. Our data collection spans 21 months, covering both day and night periods
across the seasons of spring, summer, and winter. We focus
on simultaneously providing a greater variety of clothing
and more RGB and IR cameras for each person. Through
efforts to expand in terms of capture dates, time periods, and
scene variations, our AT-USTC provides a broader intraidentity diversity and more comprehensive AT-ReID cases
than previous datasets.
To tackle the new challenge of multi-scenario retrieval
in AT-ReID, we further propose a unified model named
Uni-AT to effectively handle all scenarios. Given that the
AT-ReID encompasses six different scenarios, the information shared among all scenarios becomes limited, and learning a unified representation for all scenarios is sub-optimal.
Therefore, we propose a novel Multi-Scenario ReID (MSReID) framework with multiple classification tokens and
a scenario-aware identity loss to facilitate effective learning of specific features for each scenario. To achieve better discriminative feature extraction for different scenarios,
we improve MS-ReID at both the model structure and optimization levels. Specifically, we propose a Mixture-ofAttribute-Experts (MoAE) module, which builds the ex-

(a)

Daytime
RGB camera
Scenario : DT-ST

All-day
RGB
IR

Nighttime
IR camera

Scenario : AD-ST

Tr-ReID

CM-ReID

Scenario : NT-ST
New

Scenario : DT-LT

Scenario : AD-LT
New

Scenario : NT-LT
New

Shortterm
Same
clothes
CC-ReID

Longterm
Clothes
changes

(b)

Uni-AT method

AT-USTC dataset
DT-ST

Multi-scenario learning
Cover all scenarios

DT-LT AD-ST AD-LT NT-ST

AT-ReID task

NT-LT

Anytime retrieval

Tr-ReIDCC-ReIDCM-ReID absent absent absent

Figure 2. (a) Six non-overlapping scenarios based on variations in
time. AT-ReID aims to perform retrieval in all of these scenarios.
(b) Our solution of AT-ReID from the dataset to the model level.

pert network and assigns different experts to address distinct scenarios, thus enabling the model to alleviate interference between scenarios. Additionally, we define the attribute layer as the basic cell shared among experts with
similar scenario attributes, e.g., DT-related attribute layers
are shared among DT-LT and DT-ST experts. With this,
the model can benefit from multiple interrelated scenarios.
And we propose a Hierarchical Dynamic Weighting (HDW)
strategy, that tackles the AT-ReID training from the multitask learning view. It establishes all scenarios into several
tasks and balances the training for different tasks with a loss
weighting scheme. This method considers multiple relevant
tasks when computing weights, implicitly modeling the relationships between tasks and leading to better optimization
of the overall multi-scenario learning framework.
Our main contributions can be summarized as follows:
• We investigate a new task called AT-ReID, which aims
at enabling retrieval at any time moment and across different time intervals. We contribute for the first time a largescale dataset named AT-USTC to support the study of ATReID. Compared to existing datasets, AT-USTC stands out
for its long data collection period and the inclusion of both
RGB and IR camera footages, meeting the requirement of
AT-ReID. Importantly, our data collection has obtained the
consent of each volunteer.
• We propose a Uni-AT model to effectively handle all scenarios of AT-ReID. In Uni-AT, three components, a new
multi-scenario ReID framework, a Mixture-of-AttributeExperts module, and a Hierarchical Dynamic Weighting

training strategy are proposed to tackle the new challenges
of multi-scenario retrieval in AT-ReID tasks. Extensive experiments show that our model leads to satisfactory results
and exhibits excellent generalization to all scenarios.

Cam 15
Cam 10

Cam 16

Cam 9
Cam 1

Cam 11

Cam 12

2. Related Work

Cam 2

Cam 11
Cam 8

Cam 3
Cam 7

Person Re-Identification. Traditional ReID (Tr-ReID)
aims to achieve short-term pedestrian retrieval in the RGB
modality. The corresponding datasets, such as Market1501 [58], CUHK03 [21], and MSMT17 [46], focused
on providing more identities as well as more camera variations. Tr-ReID methods involve general pedestrian retrieval
techniques, such as the design of more robust backbone networks [13, 30, 54], effective ReID loss functions [40], and
the utilization of part-level features [39] to achieve discriminative representations of pedestrians.
Visible-Infrared Cross-Modality ReID (CM-ReID) aims
to achieve short-term pedestrian retrieval between the RGB
and the infrared (IR) modalities. The corresponding
datasets, such as SYSU-MM01 [47], RegDB [34], and
LLCM [57], focused on providing more RGB and IR cameras. Some CM-ReID methods [8, 49] aimed to project features from different modalities into the same feature space,
while others [7, 22, 29, 52] aimed to learn cross-modality
relationships.
Long-term Cloth-changing ReID (CC-ReID) aims to
achieve long-term pedestrian retrieval in the RGB modality. The corresponding datasets, such as PRCC [50],
LTCC [35], and DeepChange [48], focused on providing
clothing variations for each person. Some CC-ReID methods [4, 11, 17, 26] introduced additional data such as contour, key points, human parsing, and 3D shape for model
training, while others [9, 12, 23, 51] utilized RGB images
only to learn robust clothing-irrelevant feature.
The aforementioned tasks and datasets can only cover a
portion of the AT-ReID scenarios. In addition, some unified methods [3, 5, 14, 19, 42, 59] focus on multiple ReID
tasks, such as text/sketch-to-RGB ReID, clothes template
based CC-ReID, and occlusion ReID, as well as humancentric tasks, such as human parsing, pose estimation, and
pedestrian detection. Our research is distinct from previous
methods as it is the first to focus on the availability of ReID
at any time and proposes a relevant dataset and method to
bridge the gap between existing research and AT-ReID.
Multi-Task Learning. Multi-task learning (MTL) refers
to building a model that can handle multiple distinct
tasks [2, 33]. By sharing parameters between tasks, MTL
methods achieve efficient memory and data utilization and
expect to derive benefits from multiple related tasks. In ATReID, various input modalities and learning objectives are
present in different scenarios. Retrieval in each scenario can
be considered an individual ReID task, and it is promising

Cam 4
Cam 5

Cam 6

Cam 10
RGB
Camera
IR
Camera

Figure 3. The plan of the camera layout for collecting data.

that employing MTL methods can improve the overall efficacy of the model across all scenarios.
Some MTL methods focused on network architecture [28, 32, 45, 61] to achieve more effective parameter
sharing. Recently, some effective approaches [1, 31, 36, 41,
61] are to utilize the Mixture-of-Experts (MoE) [16] model
that employs multiple expert sub-networks to tackle multitask learning. Compared to these MoE methods, our MoAE
constructs scenario experts in a more flexible sharing manner, making the model benefit from multiple interrelated
scenarios. Other methods focused on MTL optimization,
such as manipulating gradient [25, 27, 56] and adjusting
the loss weight by task difficulty, training speed, and priority [3, 10, 18, 28]. Our HDW method groups tasks based
on their attributes and applies hierarchical dynamic weighting to the loss of each task, achieving a more effective task
balance.

3. AT-USTC Dataset
Dataset Description. AT-USTC is the first AT-ReID
benchmark that includes 403,599 (199,803 RGB and
203,796 IR) images of 270 identities and 710 sets of different clothing captured by 16 cameras. As shown in Fig. 3,
we deployed 8 RGB and 8 IR cameras across 16 nonoverlapping locations, comprising 5 indoor and 11 outdoor
scenes. We filmed videos spans 21 months including spring,
summer, and winter, with temperatures ranging from -3◦ C
to 33◦ C to cover a wider range of clothing types. Each individual in our training set has 2-14 outfits with an average of 3.6, which facilitates retrieval in long-term scenarios.
Due to the variations in both modality and clothing in ATUSTC, the process of capturing and annotating the data is
more time-consuming compared to other datasets. We made
considerable effort to provide annotations, including labels
for person, camera, and clothing.
Privacy Protection. Following the established ReID
datasets [47, 50], we made efforts for privacy protection in
five aspects: 1) Data collection was authorized by the relevant authorities, involving the deployment of cameras and
image capture. 2) The individuals we photographed did not

AT-USTC
AT-USTC

AT-USTC
AT-USTC

SYSU-MM01

SYSU-MM01
Nighttime

LLCM
DeepChange

Indoor

RegDB

Outdoor

PRCC
LTCC

CUHK03
Market1501

LTCC

29.1
7.3

AT-USTC
AT-USTC

403K

DeepChange

178K

LLCM

1.4 1.8

MSMT17

4.7

MSMT17

1.01.0

DeepChange

4.7

LLCM

47K

SYSU-MM01

4.6

SYSU-MM01

45K

Market1501

4.4

PRCC

34K

LLCM

3.2

Market1501

33K

PRCC

3.0

LTCC

17K

CUHK03

2.0

CUHK03

14K

RegDB

2.0

RegDB

8K

LTCC

3.7
3.2

PRCC

3.0

Market1501

(a)

AT-USTC
AT-USTC

1.6

RegDB

MSMT17

CrossSeason

5.3

3.0

DeepChange

ClothesChange

MSMT17

5.7

RGB
Cameras
Per Person

IR
Cameras
Per Person

4.5
4.4

CUHK03

2.0

(b)

Avg Capture
Times/Videos
Per Person

(c)

126K

Scale

(d)

Figure 4. Statistics of our AT-USTC and several popular ReID datasets.
Summer

Spring

Capture Times/Videos Per Person

MSMT17: 4.7

Market1501: 4.4

SYSU-MM01: 4.6

LLCM: 3.2

Winter

AT-USTC:

29.1

DeepChange: 4.7

LTCC: 7.3

Figure 5. Each person in AT-USTC has been photographed 29.1
times on average, resulting in higher intra-identity diversity. Images of each dataset all belong to the same individual.

include minors. 3) Each volunteer who participated in the
filming has signed a standardized consent agreement, agreeing to the release of their images for academic research purposes. 4) Our AT-USTC dataset does not include any personal information beyond the captured images. 5) Individuals or organizations seeking to use our dataset are required
to sign the corresponding dataset release agreement, which
imposes restrictions on the dataset’s copyright, usage, modification, and redistribution.
Dataset Advantages. As shown in Fig. 4 (a), our ATUSTC captured images in various scenarios, including daytime, nighttime, indoor, outdoor, and cross-season intervals,
providing comprehensive variations of modality, clothing,
camera, and scene. In contrast, the three Tr-ReID datasets,
Market1501 [58], CUHK03 [21], and MSMT17 [46]
do not include clothing changes and IR cameras; the
three CM-ReID datasets, SYSU-MM01 [47], RegDB [34],
and LLCM [57] do not consider clothing changes; the
three CC-ReID datasets, PRCC [50], LTCC [35], and
DeepChange [48] do not include IR cameras. Additionally,
certain datasets, such as synthetic datasets [44], datasets
from movies [37], and the internet [55], differ significantly

from the surveillance environment domain and are therefore
not within the scope of comparison.
As shown in Fig. 4 (b), we provide rich camera variations for each person during day and night, facilitating
cross-camera and cross-modality retrieval. The number of
cameras per identity [38] reflects the camera diversity of
the dataset. On average, each person of AT-USTC appears
in 5.7 RGB and 5.3 IR cameras, totaling 11 cameras, which
is significantly higher than other datasets.
As illustrated in Fig. 4 (c) and Fig. 5, each person in ATUSTC has been photographed 29.1 times on average, and
the identities in the training set were photographed an average of 40.0 times. Therefore, this results in significantly
higher intra-identity diversity and visual variations of our
AT-USTC dataset. Compared with existing datasets, our
AT-USTC exhibits day and night photography, diverse cameras, and captures across multiple seasons, enabling it to
encompass all scenarios of AT-ReID.
As shown in Fig. 4 (d), the scale of our AT-USTC significantly exceeds other datasets. This is due to the higher
intra-identity diversity exhibited by each individual within
the dataset. In addition to the rich variations in cameras,
modalities, and clothing, the average duration of each video
in our dataset is 50 seconds, resulting in greater diversity of
postures.
Data Split. We have a fixed split of the dataset into training and testing sets. The training set consists of 135 people with 286,087 images, and the testing set consists of
another 135 people with 117,512 images. We partitioned
20% (55,060) images from the training set for validation
purposes. Existing datasets primarily evaluate a single scenario, while we construct separate gallery sets and query
sets for all six scenarios covered by AT-ReID to facilitate
a comprehensive assessment of model performance and explore anytime retrieval. For each identity, we selected three
query images and three gallery images from video clips
featuring the same identity, captured by the same camera,
and with the same clothing. Under this configuration, the
gallery contains an average of approximately 25 images per
identity.

Task3

Position
Embedding

7

8

… 13

[ NT ]

Self-Attention

Mixture-of-AttributeExperts (MoAE)

[ LT ]

NT-ST

…

Multi-Scenario
ReID (MS-ReID)

* * *

[ ST ]

…

…
AT-USTC
Dataset

*

[ AD ]

Norm

Hierarchical Dynamic
Weighting (HDW)

DT-ST

…

6

Linear Projection

Loss weight

*

… 5*

*

[ DT ]

…

Task1

1*

*

…

Norm
Transformer Layer

0*

Expert

*

Transformer Layer

Weighted
sum

…

MoAE

*

FFN

Confidence
(Time Moment)

Gating

*

Scenario

* * *

*

Scenario
Head

* CLS Token

…

Confidence
(Time Interval)

Task2

Patch
Token

Select

Scenario-Aware
Identity Loss

*
AD-LT

Attribute Experts
layers
(FFN)

Figure 6. The pipeline of our Uni-AT. The DT, NT, AD, ST, and LT denote daytime, nighttime, all-day, short-term, and long-term cases,
respectively. With a Multi-Scenario ReID framework, a Mixture-of-Attribute-Experts module, and a Hierarchical Dynamic Weighting
scheme, Uni-AT enhances the learning of diverse scenario-specific features and improves model generalization.

4. Method
Overview. Anytime ReID (AT-ReID) aims to perform retrieval in multiple scenarios, including daytime short-term
(DT-ST), daytime long-term (DT-LT), all-day short-term
(AD-ST), all-day long-term (AD-LT), nighttime short-term
(NT-ST), and nighttime long-term (NT-LT) scenarios.
The pipeline of our proposed Unified AT-ReID model
(Uni-AT) is shown in Fig. 6. The image is fed into a MultiScenario ReID (MS-ReID) framework to extract several
types of scenario features for accurate retrieval in all covered scenarios of AT-ReID. To treat each scenario optimally,
we further propose a Mixture-of-Attribute-Experts (MoAE)
module to effectively capture scenario-specific clues and
mitigate inter-scenario interference. To balance feature
learning in different scenarios, we proposed a Hierarchical Dynamic Weighting (HDW) scheme to train the whole
model more effectively in an end-to-end way.

4.1. Multi-Scenario ReID
Model Architecture. The proposed Multi-Scenario ReID
(MS-ReID) framework is designed to extract multiple
scenario-specific features more effectively. We choose Vision Transformer (ViT) [6] as our backbone. The input image xi is split into patches and mapped to patch tokens.
Then we establish 6 CLS tokens tsi to extract image features and assign each one with a corresponding scenario s.
These CLS tokens serve as information gatherers, collecting
different scenario-specific knowledge from patch tokens by
stacked self-attention modules.
Note that the main principle of our MS-ReID is to extract
different features for different scenarios separately rather

than use a single unified representation for all scenarios because the latter sacrifices specific clues in each scenario.
The main concern is based on a prior that scenario-specific
information can lead to optimal results under specific cases.
For example, color information is suitable for daytime retrieval and clothes cues are discriminative for short-term
situations. Moreover, AT-ReID is a scenario-determinable
task, as the practical ReID involves retrieving between the
query image and the gallery images in the video surveillance, and the shooting timestamps can be easily accessed.
When faced with an uncertain scenario, the default is set to
all-day/long-term to account for potential modality variations/clothing changes. Thus, our framework is adaptable
and capable of offering more precise solutions for determined scenarios.
Scenario-aware identity loss. The common identity loss
provides undifferentiated supervision for all scenarios,
which can only learn the shared information across all scenarios but cannot capture scenario-specific cues. Therefore, we propose a scenario-aware identity loss Lsid for
our MS-ReID framework to supervise feature learning for
each covered scenario, where different scenarios have nonshared classifiers, distinct negative category sets, and different modality filtering mechanisms. Lsid is derived as follow:
Lsid (tsi ) = − log(psi,gt ),

(1)

where psi,gt is the classification probability for scenario s.
psi,gt is derived as follow:
psi,gt =

exp(osi,gt )
P
,
exp(osi,gt ) + j∈Ns exp(osi,j )

(2)

where osi is classification logit generated from the scenario
CLS token tsi and Ns is the negative category set for the
scenarios s. We employ distinct Ns for ST-related and LTrelated scenarios. Specifically, for ST cases, we treat different clothes as different categories in classification to guide
the model to attend to fine-grained discriminative information about clothes. However, we do not expect the model to
classify the same person’s images with different clothes into
different categories because they share consistent semantic
body information. To achieve this goal, we set Ns of ST
cases as the clothes ID set while each clothes in this set has
different owners with the ground-truth clothes. For the LT
cases, we follow the traditional ReID setting to define the
category space as the set of person IDs. So Ns of LT cases
are different ID persons directly. In addition to distinguishing between ST and LT scenarios, scenario-aware identity
loss also includes a modality filtering mechanism. For RGB
images, we supervise the DT-ST, DT-LT, AD-ST, and ADLT tokens of their potential scenarios while neglecting the
NT-related ones that are not available to RGB images. Similarly, for IR images, we ignore their DT-related tokens.
Our MS-ReID can fit the goal of each scenario separately, facilitating multi-scenario learning on a dataset with
modality and clothing variations.

4.2. Mixture-of-Attribute-Experts
In our MS ReID framework, CLS tokens are supervised
by distinct identity loss, aiming to extract corresponding
scenario-specific features. However, similar to the case of
multi-task learning [33], the parameter-shared ViT feature
extraction network may result in potential gradient conflicts. For instance, the all-day scenario focuses solely
on modality-shared information while the daytime scenario
needs to also consider RGB-specific information, leading to
mutual interference in feature learning between scenarios.
To tackle this problem, inspired by Mixture-of-Experts
(MoE) methods [16, 31], we propose a novel Mixture-ofAttribute-Experts (MoAE) module. As depicted in Fig. 6,
the MoAE module is added parallel with the feed-forward
network (FFN) in the transformer layers. In each transformer layer, patch tokens are fed into the original FFN,
while the CLS tokens are fed into the MoAE module. We
establish MoAE with 6n experts and assign different experts to address distinct scenarios. For each scenario s,
there are n specific experts {Ejs }nj=1 , where Ejs represents a single-layer FFN comprising two linear layers and
a gelu [15] activation function. Given an input CLS token
tsi ∈ Rd corresponding to the scenario s, the MoAE selects
and combines experts through a gating network, producing
the output y ∈ Rd . More precisely, y is the weighted sum
of the outputs from the n experts:
y=

Xn
j=1

Gs (tsi )j Ejs (tsi ),

(3)

where Gs (tsi ) ∈ Rn is the weight of n experts, calculated
by a gating network:
Gs (tsi ) = topk (softmax(Wgs · tsi )),

(4)

where, Wgs ∈ Rn×d is the trainable weights for scenario
s in gate decision, and the topk (·) operator sets all values
to zeros except the top-k largest values. Except for the selected k experts, others do not need to be computed for saving computation. Following [1, 31, 41], we set k to 1 to
obtain sparse and efficient expert networks. With this module, different inputs can utilize experts in distinct ways and
capture scenario-specific clues.
Additionally, we note the existence of potential relationships among various scenarios. For example, DT-ST and
DT-LT scenarios both have the “DT” attribute, indicating
that they require the model to extract RGB-specific features. To introduce this knowledge and establish relationships across scenarios, we construct five types of attribute
layers as the basic cells shared among experts with similar
scenario attributes. Among these, “DT”, “AD”, and “NT”
attribute layers belong to the category of Time-Moment
(TM), while “ST”, and “LT” attribute layers belong to the
category of Time-Interval (TI). In practice, attribute layers
are linear layers and we derive the experts by combining
one TM attribute layer and one TI attribute layer as follows:
E s (·) = a1 (gelu(a2 (·))),

(5)

where a1 and a2 are the attribute layers associated with scenario s. In summary, our attribute layers are shared across
different scenarios, which can serve as prior knowledge for
the relationships between scenarios. Compared to other
MoE methods [1, 31, 41], our MoAE is more flexible and
effective for expert construction, which lets the model benefit from multiple interrelated scenarios.

4.3. Hierarchical Dynamic Weighting
In our MS-ReID framework, we utilize scenario-aware
identity loss to supervise each scenario, which can be
considered as multi-task learning. Through joint learning
across multiple tasks, we aim to increase the model’s generalization ability in various scenarios. However, different
tasks have different levels of difficulty and learning curves.
Simply summing all losses with fixed weights to optimize
the overall framework may not provide adequate training
for some tasks, leading to limited robustness.
To achieve a balanced optimization across all tasks,
we propose a Hierarchical Dynamic Weighting (HDW)
scheme. The idea of HDW is that, when some tasks retrieve
corresponding images with low predicted confidence, they
should contribute more to the final loss, and vice versa. The
HDW can be exported as follows:
X
Ltotal =
ws · Lsid ,
(6)
s∈S

Test

Market1501

CUHK03

SYSU-MM01

PRCC

LTCC

AVG

Train

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

Rank-1

mAP

MSMT17 (1041 IDs)
LLCM (713 IDs)
DeepChange (450 IDs)

57.63
46.79
57.48

30.08
19.79
30.42

14.64
4.36
6.00

13.81
4.89
6.11

4.49
7.17
3.60

6.81
8.78
5.85

24.41
33.45
25.52

23.33
26.17
24.12

20.66
12.76
15.31

8.53
5.93
5.90

24.37
20.91
21.58

16.51
13.10
14.48

AT-USTC (135 IDs)

60.30

34.95

21.71

21.35

26.49

25.61

42.87

36.81

25.00

9.97

35.27

25.74

Table 1. Cross-domain generalization experiments in different datasets. Rank-1 accuracy (%) is reported.

where ws is the weight of the scenario s. Additionally,
we observe that the learning of different tasks is interrelated rather than independent. For instance, the retrieval
task in DT-ST scenarios requires RGB-specific features and
clothing features. The former can be learned from two DTrelated tasks, while the latter can be learned from three STrelated tasks. Therefore, the weight adjustment for these
tasks should also adhere to this principle. To achieve this,
ws is calculated by two terms and derived as follows:
s
s
ws = wtm
· wti
.

(7)

Method

Experts

Params

Time

Rank-1

mAP

MS-ReID

0

1.00×

1.00×

52.03

37.49

+ MMoE [31]

6
12

4.90×
8.80×

1.60×
2.18×

53.00
53.17

38.22
38.62

+ PLE [41]

6
7
12

4.90×
5.56×
8.80×

1.14×
1.26×
1.81×

52.77
53.17
53.20

38.44
38.54
38.67

+ VLMo [1]

3
6
12

2.95×
4.90×
8.80×

1.07×
1.25×
1.53×

52.81
53.30
53.40

38.10
38.59
38.57

+ MoAE (ours)

6
12

2.62×
4.25×

1.06×
1.20×

52.98
53.70

38.40
38.76

s
s
Inspired by the Focal Loss [24], wtm
and wti
can be computed as follows:
1

1

s
s
= (1 − psti ) 2 ,
wtm
= (1 − pstm ) 2 , wti

(8)

where pstm and psti corresponding time moment and time interval confidences for scenario s, respectively. For example,
if s is the DT-LT scenario, pstm is the confidence of the DT
situation while psti is the LT case confidence. We compute
the confidence by averaging the ground truth classification
probabilities of CLS tokens from scenarios s′ in each training batch:
′

′

s′tm = stm ,

′

′

s′ti = sti ,

pstm = mean(exp(−Lsid (tsi ))),
psti = mean(exp(−Lsid (tsi ))),

(9)

where scenarios s′ carry the same time moment/time interval attribute as target scenario s.
Different from traditional multi-task learning strategies [10] assuming each task is independent of the other, our
joint weighting method considers multiple relevant tasks
when computing weights, implicitly modeling the relationships between tasks and leading to better optimization of
the overall multi-task learning framework.

5. Experiments
Datasets. We primarily conducted experiments on the
proposed AT-USTC dataset due to the lack of support for multi-scenario training and evaluation of ATReID in existing datasets. To further evaluate our ATUSTC dataset and Uni-AT method, we utilized several popular ReID datasets, including MSMT17 [46],

Table 2. Comparison with other MoE methods on AT-USTC.

Market1501 [58], CUHK03 [21], SYSU-MM01 [47],
LLCM [57], DeepChange [48], PRCC [50], and LTCC [35],
for cross-domain generalization experiments.
Evaluation Protocols. The Rank-k matching accuracy
and mean average precision (mAP) are adopted as evaluation metrics. For AT-USTC, we conducted separate tests
in six different scenarios, including DT-ST, DT-LT, NT-ST,
NT-LT, AD-ST, and AD-LT. The average performance of
six scenarios is referred to as Any-Time, which evaluates
the model’s ability to retrieve at any given time. For other
datasets, we adhered to the evaluation settings of their original papers.
Implementation Details. We use a ViT-Base model with
patch size 16 and step size 16 as our backbone. Following
existing work [30], we introduce the BNNeck before the
classifier. All person images are resized to 256 × 128 and
are augmented with random horizontal flipping, padding,
random cropping, and random erasing [60] in training. The
batch size is set to 64 with 8 identities. The whole model
is trained for 120 epochs (24K iterations) with the SGD optimizer. The learning rate is initialized as 0.008 with the
warm-up scheme and cosine learning rate decay.

Lsid
!
!
!
!

Method

Any-Time

MoAE HDW

R1

mAP

R1

mAP

R1

mAP

R1

mAP

R1

50.90
52.03
53.70
53.32
55.80

34.75
37.49
38.76
39.61
41.38

95.02
96.95
97.04
97.44
97.76

80.23
85.61
86.09
86.77
87.97

32.99
33.68
35.31
33.14
36.75

21.48
22.17
23.19
22.86
25.89

74.53
78.71
79.35
79.72
81.32

43.84
50.23
50.89
52.44
53.82

38.89
38.64
38.19
36.89
39.54

!
!

!
!

DT-ST

DT-LT

NT-ST

NT-LT

AD-ST

AD-LT

mAP

R1

mAP

R1

mAP

23.88
24.61
24.86
25.76
26.93

38.05
40.78
45.16
46.61
50.25

23.74
27.62
30.95
32.50
34.94

25.92
23.44
27.15
26.15
29.21

15.31
14.71
16.58
17.32
18.71

Table 3. Ablation study on AT-USTC. Rank-1 (R1) and mAP accuracy (%) are reported.
Method
Lsid
!
!
!
!

MoAE

Avg

CUHK03

SYSU-MM01

PRCC

LTCC

HDW

R1

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

mAP

!
!

31.47
34.42
36.95
38.13
39.81

24.16
27.21
29.11
30.03
31.76

60.01
63.24
68.38
68.82
70.72

34.30
37.76
42.87
43.08
45.67

20.43
24.79
25.57
27.21
29.07

19.30
23.79
24.67
25.99
27.49

20.78
26.10
28.58
30.21
32.26

20.81
25.68
28.03
29.28
31.12

33.95
32.97
35.70
35.85
38.67

33.63
35.09
35.41
36.88
38.88

22.19
25.00
26.53
28.57
28.32

12.75
13.72
14.58
14.90
15.66

!
!

Market1501

Table 4. Ablation study with cross-dataset testing. Trained on AT-USTC and inferred on Market1501, CUHK03, SYSU-MM01, PRCC
and LTCC datasets. Rank-1 (R1) and mAP accuracy (%) are reported.

For the other comparison methods, we employed their
official code, ensuring that the image are resized to 256 ×
128. For the transformer-based methods, we maintained the
use of the ViT-Base model with a patch size of 16 and a step
size of 16.

5.1. Generalization Evaluation of AT-USTC
One of our main contributions is collecting the AT-USTC
dataset, which exhibits higher intra-identity diversity compared to existing ReID datasets. To evaluate the quality
of our dataset, we conducted domain generalization experiments, comparing it with three large-scale Tr-ReID,
CC-ReID, and CM-ReID datasets, including MSMT17,
DeepChange, and LLCM. For a fair comparison at the
dataset level, we trained the same ResNet50 model for all
datasets instead of our proposed model.
As shown in Tab. 1, the model trained on AT-USTC
achieved the best cross-dataset performance, surpassing the
model trained on MSMT17/DeepChange/LLCM by an average of 10.90% / 13.69% / 14.36% in Rank-1 accuracy and
9.23% / 11.26% / 12.64% in mAP accuracy. This indicates
that, in addition to the number of IDs, the inherent diversity
of each ID is a significant aspect of the ReID dataset. The
high intra-identity diversity in AT-USTC results in excellent scalability, thereby effectively supporting research on
the AT ReID task

5.2. Comparison with MoE Methods
As shown in Tab. 2, we compare our MoAE module with
other MoE methods under Any-Time testing on the AT-

USTC dataset. For a fair comparison, all MoE modules
are added to our MS-ReID framework in the same manner. The difference among these methods lies in the sharing
of experts. MMoE [31] constructs experts shared across all
scenarios, while PLE [41] explicitly separates the experts
into scenario-specific ones and those shared across all scenarios. VLMo [1] was originally designed to process visual
and language modalities. Here, we apply its principles to
handle the RGB and IR modalities in the ReID task, establishing three types of experts for the RGB, IR, and crossmodality data.
From the experimental results, it can be summarized that
our MoAE consistently outperforms other methods when
parameters or computational time are comparable. This
is because our MoAE introduces scenario priors and enables fine-grained expert sharing, whereas other methods
can only coarsely share experts across scenarios. Furthermore, our training or inference speed is only 1.2 times that
of the non-expert model when using 12 experts. Through
parameter sharing with our attribute layers, MoAE not only
effectively extracts scenario-specific features to enhance
performance but also exhibits greater parameter and time
efficiency than other MoE methods.

5.3. Ablation Analysis
As shown in Tab. 3 and Tab. 4, we conduct ablation studies to evaluate each component of our method. In the 1-st
row, we establish an MS-ReID baseline using the standard
identity loss for six CLS tokens, which provides undifferentiated supervision for all scenarios, failing to take advan-


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
9852

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 6, JUNE 2025

TIENet: A Tri-Interaction Enhancement Network
for Multimodal Person Reidentification
Xi Yang , Senior Member, IEEE, Wenjiao Dong, De Cheng , Nannan Wang , Senior Member, IEEE,
and Xinbo Gao , Fellow, IEEE

Abstract—Multimodal person reidentification (ReID), which
aims to learn modality-complementary information by utilizing
multimodal images simultaneously for person retrieval, is crucial for achieving all-time and all-weather monitoring. Existing
methods try to address this issue through modality fusion to
absorb complementary information. However, most of these
methods are limited to the spatial domain only and usually
overlook the intra-/intermodal interactions during feature fusion,
resulting in insufficient learning of modality-specific and complementary information. To address these issues, we propose a
tri-interaction enhancement network (TIENet), which contains
three modules: spatial-frequency interaction (SFI), intermodal
mask interaction (IMMI), and intramodal feature fusion (IMFF).
Specifically, the SFI boosts the modality-specific representation
by integrating the amplitude-guided attention mechanism into
the phase space, combined with spatial-domain convolution to
achieve fine-grained information learning. Meanwhile, the IMMI
enhances the richness of the feature descriptors by embedding
the intermodal relationships to preserve complementary information. Finally, the IMFF module considers the structure of the
human body and integrates intramodal contextual information.
Extensive experimental results demonstrate the effectiveness of
our method, achieving superior performances on RGBNT201 and
MARKET1501 RGBNT datasets.
Index Terms—Inter-/intramodal interaction, multimodal person reidentification (ReID), spatial-frequency interaction (SFI).

I. I NTRODUCTION

P

ERSON reidentification (ReID) is a technique for retrieving and identifying the same person under different
cameras, which has a wide range of applications in the fields
of security, surveillance, and intelligent transport. Different
Received 28 July 2024; revised 28 October 2024; accepted 18 February 2025. Date of publication 19 March 2025; date of current version
4 June 2025. This work was supported in part by the National Natural
Science Foundation of China under Grant 62372348, Grant 62176198, Grant
U22A2096, and Grant 62036007; in part by the Key Research and Development Program of Shaanxi under Grant 2024GX-ZDCYL-02-10; in part by
Shaanxi Outstanding Youth Science Fund Project under Grant 2023-JC-JQ53; in part by Shaanxi Province Core Technology Research and Development
Project under Grant 2024QY2-GJHX-11; and in part by the Fundamental Research Funds for the Central Universities under Grant QTZX23042.
(Corresponding author: De Cheng.)
Xi Yang, Wenjiao Dong, De Cheng, and Nannan Wang are with
the State Key Laboratory of Integrated Services Networks, School of
Telecommunications Engineering, Xidian University, Xi’an 710071, China
(e-mail: yangx@xidian.edu.cn; dwj@stu.xidian.edu.cn; dcheng@xidian.edu.
cn; nnwang@xidian.edu.cn).
Xinbo Gao is with Chongqing Key Laboratory of Image Cognition, Chongqing University of Posts and Telecommunications, Chongqing
400065, China, and also with the School of Electronic Engineering,
Xidian University, Xi’an 710071, China (e-mail: gaoxb@cqupt.edu.cn;
xbgao@mail.xidian.edu.cn).
Digital Object Identifier 10.1109/TNNLS.2025.3544679

from the traditional person ReID task, i.e., ReID based on
single visible image [1], [2], [3], [4], [5], [6], the multimodal
person ReID aims to perform robust ReID by introducing
multiple complementary modality images for each person
example. Such multimodal person ReID helps to deal with
more complex illumination scenarios, which greatly assists the
traditional ReID task and addresses its application limitations.
Besides, the popularity of diverse kinds of cameras (i.e.,
various infrared and RGB cameras) makes multimodal person
ReID possible and has attracted more interest recently. Therefore, the multimodal person ReID will have great potential
applications in the field of intelligent surveillance system, due
to its strong complementary benefits from different modalities.
As shown in Fig. 1(a), it illustrates examples from
RGBNT201 dataset [7], which includes three-modality images
for each example, i.e., visible (RGB), near-infrared (NI), and
thermal-infrared (TI) modalities. Obviously, the RGB images
can provide rich color and texture information under favorable
lighting condition, the NI images can provide clear edge
information without being affected by the light, while the TI
images can discriminate persons from surroundings by using
their temperatures without being affected by the complicated
environment [7]. Therefore, how to make full use of the
complementary information of different modalities is the key
to multimodal person ReID.
It is very different from the traditional cross-modality
person ReID [8], [9], [10], [11], [12], [13], which focuses
on reducing the modality discrepancy and learning modalityshared features. In contrast, the multimodal person ReID
focuses on effective modality fusion to absorb complementary
information from the multimodality data, and further improves
the person discriminability.
Existing fusion-based multimodal feature learning can be
roughly divided into the following three categories: imagelevel early fusion [14], [15], feature-level late fusion [16],
[17], which merges different modality features to obtain
enhanced representations, and their combinations with different strategies [18], [19]. However, these methods are limited
to the spatial domain only, and usually ignore the intermodal
interactions during feature fusion procedure. It will cause
insufficient modality-specific and complementary information
learning, and result in some redundancy or biased information
in the fused features. To address these issues, we propose
a tri-interaction enhancement network (TIENet) to boost the
modality-specific information through the spatial-frequency
interaction (SFI) module, while absorbing complementary

2162-237X © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.

YANG et al.: TIENet: A TRI-INTERACTION ENHANCEMENT NETWORK FOR MULTIMODAL PERSON ReID

Fig. 1. Frequency information for multimodal person ReID task. (a) Key
to multimodal person ReID task. (b) Fourier transforms of different modal
images. (I) Spatial-domain form of different modal pedestrian images. (II)
Different modal amplitude components. (III) Different modal phase components. (IV) Reconstructed image with phase only (phase-r). (V) Reconstructed
image with amplitude only (amplitude-r).

information between intra-/intermodalities by intermodal mask
interaction (IMMI) and intramodal feature fusion (IMFF)
module.
The frequency relationship between RGB and NI images
has been explored in [20]; in order to further explore the
relationship between RGB, NI, and TI frequencies, as shown
in Fig. 1(b), we recover the pedestrian images using only phase
(IV) and amplitude (V), respectively. It can be visualized that
the phase mainly conveys global structural information, while
amplitude could convey specific content information. To show
the difference between phase and amplitude of different modal
images more intuitively at the feature level, we take the conv1,
bn1, relu, and maxpool layers of the pretrained ResNet50 on
ImageNet as the backbone and take seven random ID image
sets to extract shallow features and draw t-SNE maps. The
following should be noted.
1) Shallow features express the basic information of the
multimodal image, while deep features reflect the
advanced semantic information of the image. Thus, we
only use shallow features to show the direct relationship
between the phase and amplitude of different modal
images.
2) The feature extraction stage is without any loss function
guidance and training process. Therefore, we can mine
the most essential properties of multimodal images in
the frequency domain. From Fig. 2(a-1) and (a-2), we
can see that the amplitude of different modalities is more
discriminative than the phase, which is consistent with
our image visualization results.
Therefore, a straightforward approach is to use an attention mechanism on amplitude in the frequency domain to
enhance the phase, as illustrated in Fig. 2(b). Compared
to Fig. 2(a), the enhanced feature modalities exhibit clearer
boundaries, indicating that amplitude-guided phase can capture
more modality-specific information in feature representation.
Additionally, features from the same ID become more compact, suggesting that enhanced features can learn ID-specific
distinguishing features.

9853

In conclusion, we deem that different modalities of the same
pedestrian can be more differentiated by amplitude than phase
in the frequency domain, as the same person usually shares
the same global structure in the phase domain, and that the
amplitudes of different pedestrians are more discriminative
because the content information of the amplitude response is
more valuable than the structural information in the pedestrian
classification task. Based on this, we propose the SFI module,
which consists of two data flows. The frequency-domain
flow implements the amplitude-guided attention mechanism
to instruct phase learning, ultimately resulting in enhanced
feature representations. The spatial-domain flow utilizes traditional convolutional neural network (CNN)-based spatial
feature learning to preserve local detailed information, and the
interaction between the two data flows achieves fine-grained
feature enhancement.
Moreover, different modalities can convey different information that complements each other in pedestrian retrieval.
In order to absorb complementary information from different
modalities, intermodal interaction is essential. Due to the existence of different relationships between different modalities,
fixed interactions will destroy the intermodal relationships
and cause inconsistent information. Therefore, we introduce
the IMMI module in the middle of modality-specific and
shared networks to guide the process of feature interaction
by learning the intermodal correlation matrix. Specifically, the
IMMI module forms a mask that adaptively reflects intermodal
correlations and complementarities by computing intermodal
similarities. Therefore, this approach ensures the consistency
of information after interaction and reduces information redundancy compared to traditional intermodal interaction.
Meanwhile, the human body structure provides rich spatial
information that can enhance the robustness of the features.
By exploiting the structural relationships of the human body,
the effects of lighting conditions, pose, and occlusion can
be alleviated and more effective pedestrian features can be
extracted under unfavorable conditions. In addition, the human
body structure is independent of the modality since the pose of
each pedestrian remains consistent across modalities. Based on
this, at the end of the network, we introduce the IMFF module
based on location attention, which takes into account the connections between human body parts. Thereby, the local feature
representation contains not only its own attribute information,
but also the contextual information of the neighboring local
features. Finally, it fully integrates intermodal features while
fusing intramodal contextual information. Also considering
the distribution of intra- and intermodal features, we utilize a
modal aggregation-separation (MAS) loss function to enhance
feature complementarity and robustness.
In summary, our contributions are as follows.
1) In the designed TIENet, we propose an SFI module, which enhances modality-specific features through
amplitude-guided phase learning to capture global information in the frequency domain, and simultaneously
preserves detailed information through convolution in
the spatial domain.
2) We propose to enhance the richness of the feature
descriptors, by taking into account the intra-/intermodal

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.

9854

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 6, JUNE 2025

Fig. 2. Distribution of features in different states. (a) Spatial-domain original image feature distribution. (a-1) Frequency-domain amplitude feature distribution.
(a-2) Frequency-domain phase feature distribution. (b) Spatial-domain enhanced feature distribution. The amplitude in (a-1) shows a clearer modal difference
compared to the phase in (a-2) and the enhanced features (b) contain more modality-specific information compared to the original features (a).

structure relationships to achieve the fusion of complementary information in different modalities.
3) Extensive experimental results demonstrate the effectiveness of the proposed method, which achieves superior
performances to the state-of-the-art methods by a large
margin on RGBNT201 and MARKET1501 RGBNT
datasets.
II. R ELATED W ORK
A. Single-Modal Person ReID
Single-modal person ReID refers to conducting a person
search within the same modality, with a key focus on extracting distinctive pedestrian features, predominantly utilizing
RGB images due to their widespread availability. In the deep
learning-based approach, Chang et al. [21] proposed a multilevel factorization net (MLFN) architecture for factorizing the
visual appearance of individuals into latent discriminative factors at multiple semantic levels. HACNN [22] integrated soft
pixel and hard regional attention mechanisms within CNNs to
enhance feature representation. Despite this, factors such as
occlusion, clothes changes, and background information limit
the further upgrading of model performance, generating the
special focus of some recent studies [23], [24], [25], [26].
However, due to the limited information conveyed by singlemodal images, it is not feasible to achieve all-time monitoring,
greatly restricting the application of person ReID.
B. Cross-Modal Person ReID
Cross-modal person ReID involves searching for pedestrians across two modalities; hence, the key is to narrow
the gap between modalities. Current major methods focus
on feature representation, metric learning, and generation
based on intermediate modalities. In terms of feature representation, the pioneering work by Wu et al. [8] addressed
this issue by proposing a single-stream network structure
with deep zero-padding to extract intermodal shared features.
Ye et al. [9] employed a twin network to extract features

and utilized a loss function to mitigate modal heterogeneity.
Wei et al. [12] and Zhang et al. [27] used local features
to capture fine-grained information and reduce intermodal
distance. Moreover, attention mechanisms are used to extract
discriminative modality-shared features from those personrelated regions in [28]. In metric learning, Feng et al. [29]
utilized Euclidean constraint to reduce the cross-modality
gaps. Zhu et al. [30] proposed a heterogeneous center loss that
extracts multimodal features through the twin-stream network
and uses the distance between the center of these modalities
as the loss to bring the feature of the same person closer
together. Liu et al. [31] designed an intercenter triplet loss
to ensure compact within-class features and discriminative
between-class features. Finally, in modal generation, Wang
et al. [32] introduced AlignGAN, converting RGB images
to infrared images with joint pixel-level and feature-level
alignment constraints to reduce modality gaps. To fully exploit
the complementary nature of the two modalities, Liu et al. [33]
utilized generative adversarial network (GAN) to generate a
high-quality missing image.
Despite the significant progress in cross-modal ReID techniques, the gap between modalities still exists, thus limiting
their accuracy and robustness. Therefore, it becomes crucial to
collaborate multimodal information simultaneously for pedestrian retrieval rather than cross-modal retrieval.
C. Multimodal Object ReID
Unlike cross-modal solutions to heterogeneous intermodal
issues, multimodal person ReID aims to fully utilize multisource information to improve pedestrian recognition accuracy.
Li et al. [35] proposed a weighted fusion approach using class
activation map (CAM) to combine different modalities. H-ViT,
presented by Pan et al. [36], integrates modal information as
tokens using Transformer architecture. They investigated the
impact of different fusion stages on model performance. Building upon H-ViT, Pan et al. [18] introduced a random hybrid
enhancement module for the image-level fusion of different modalities. However, they simply concatenate or linearly

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.

YANG et al.: TIENet: A TRI-INTERACTION ENHANCEMENT NETWORK FOR MULTIMODAL PERSON ReID

9855

Fig. 3. Overall network architecture. Our TIENet consists of three interaction parts, i.e., the SFI module, IMMI module, and IMFF module. Different colored
feature maps represent different modalities. Layer-0 to layer-4 are block-0 to block-4 of ResNet50 [34].

combine the features from different modalities, neglecting the
intermodal feature interactions. In [7], PFNet generates additional modal features by combining two modalities. Wang et al.
[37] proposed IEEE, which incorporates intermodal interaction
and global-local feature embedding to obtain the final feature
representation. However, the consideration of the intermodal
relationships during modal interaction is often overlooked,
which leads to information redundancy or feature bias in the
final representation. In [38], the GAFNet utilizes cycleGAN
to generate heterogeneous modal images and introduces a
feature-level attention module for feature fusion. LRFNet,
introduced by He et al. [17], is a low-rank fusion network
that leverages multiscale feature extraction guided to exploit
the complementarity features. However, the feature extraction
stage often focuses only on local detailed features and neglects
the global structural information of pedestrians.
Therefore, how to effectively extract intramodal features and
perform intermodal feature interactions to improve pedestrian
discriminability deserves further research.
III. M ETHOD
To learn modality-complementary representations from multimodal images for person ReID, we propose a TIENet
(as shown in Fig. 3), which achieves complementary learning through comprehensive interactions of frequency-spatial
domains, intra-/intermodal, and global-local features. Our
TIENet consists of three modules: SFI, IMMI, and IMFF. In
particular, SFI extends the interaction to the frequency domain,
where feature enhancement is performed by amplitude-guided
phase enhancement in the frequency domain, while local
details also being preserved by convolution in the spatial
domain. IMMI generates masks based on intermodal relations,
enabling adaptive interactions between modalities. IMFF utilizes the location attention mechanism to achieve contextual
interactions within modality. In addition, MAS loss takes into
account both the intramodal and intermodal feature distributions. With the above modules, our method achieves robust
feature extraction within modality and feature fusion among
different modalities.

A. Problem Definition
Define the visible image set V = {IR }, the TI image set
T = {IT }, and the NI image set N = {IN }. There are multiple
image triples {IR , IT , IN } for the same pedestrian. For a given
image set, the task of multimodal person ReID is to match
the images of the same pedestrian in all the sets and return a
ranking list, where the same pedestrian should be at the front
of the list.
B. Spatial-Frequency Interaction
Based on the Fourier transform, given a single-channel
image I with the shape of H × W, the Fourier transform
converts I in the spatial domain into the frequency domain
to obtain the complex component X ∈ RH×W
X(u, v) = F(I) = √

1

H−1 W−1
X
X

HW h=0 w=0

I(h, w)e− j2π( H u+ W ν)
h

w

(1)

where F(I) represents the fast Fourier transform (FFT) of I.
Accordingly, F −1 (I) represents the inverse FFT (IFFT) of I.
Since an image or feature may contain multiple channels, we
separately apply the Fourier transform to each channel in this
work with the FFT [39], [40]. Further, the complex component
X(u, v) can be decomposed into its amplitude component
A(X(u, v)) and phase component P(X(u, v)), allowing for a
more intuitive analysis of the frequency components [41]
p
A(X(u, ν)) = R2 (X(u, ν)) + I 2 (X(u, ν))
(2)


I (X (u, v))
P(X(u, v)) = arctan
(3)
R (X (u, v))
where R(·) and I(·) represent the real and imaginary parts of
X(u, v), respectively.
To fully utilize the detailed information in the amplitude
space as well as the global information in the phase space of
the frequency domain, we propose the two-stream SFI module,
which consists of two data flows, i.e., the frequency flow and
the spatial flow. The frequency flow adopts the amplitudeguided attention mechanism to obtain a context-enhanced
phase for fine-grained information learning. Specifically, the

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.

9856

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 6, JUNE 2025

Fig. 5. Structure of our IMMI module.
Fig. 4. Structure of our SFI module. Yellow arrows indicate frequency data
streams and blue arrows indicate spatial data streams.

amplitude-guided phase enhancement in the frequency domain
shows a strong ability to capture the global frequency representation, which represents the overall characteristics of
pedestrians. In contrast, traditional convolution operations
focus on learning local representations in the spatial domain
[42], which represent the local details of pedestrians. Combining these two representations allows for the learning of
more robust and discriminative features [40]. Therefore, the
spatial flow aims to utilize convolution to capture local details,
complementing the frequency flow.
As shown in Fig. 4, taking IN as an example, the input to the
SFI module is fNori , and the output by FFT contains the amplitude component A(X(u, v)) and phase component P(X(u, v))
in the frequency domain. Further, we apply attention map
Atten(x) of the amplitude space to extract key information,
which can be obtained as follows:
Atten(x)(X(u, v)) = softmax [GAP (A(X(u, v)))]

(4)

where softmax[·] denotes the softmax activation function, and
GAP(·) means the global average pooling (GAP). Then, we
apply the amplitude-guided mechanism to obtain enhanced
phase component with detailed information injected as
follows:
P0 (X(u, v)) = Atten(x)(X(u, v))

P(X(u, v))

(5)

where
denotes the elementwise product operation. After
obtaining the enhance phase in the frequency domain, we
exploit the IFFT to convert the frequency to the spatialdomain feature f f . To preserve local information, we also
apply traditional convolution operation on the original input
in the spatial domain to obtain f s .
Finally, by combining the features from spatial flow f s and
frequency flow f f , the final output arrives at
fN = conv([ f f , f s ]) + fNori

(6)

where [·, ·] denotes the concatenation operation, and the
obtained feature fN achieves complementary fusion of detailed
and global information.
This module enriches and enhances features, respectively,
in the spatial domain and frequency domain. First, in the
frequency domain, the amplitude-guided phase enhances the
discriminative power of features obtained in the feature

extraction stage, improving feature representation ability for
modality and pedestrian differentiation. Second, in the spatial
domain, convolutional operations preserve original detailed
features. Finally, interaction between frequency-domain and
spatial-domain features achieves reconstruction and enhancement of features at the feature stage.
C. Intermodal Mask Interaction
Different modalities provide various information. Intermodal interaction allows the information from different
modalities to be complemented and integrated with each other,
thus obtaining more comprehensive and rich description. A
two-stage channel attention mechanism [33], [43] is applied
in fusing both original and generated image features, thus
extracting useful information as well as fully fusing intermodal
features. Inspired by this, considering the relationships among
modalities, we propose an IMMI module, as shown in Fig. 5.
Taking fT as an example, the other two modal features are
added together, and then, both of them are fed into a 1 × 1
convolutional layer. Subsequently, the two sets of features are
multiplied and the resulting matrix is processed by a softmax
operation to obtain Mask1 . Using the similarity matrix Mask1
as mask, the sum of fN and fR is added to fT to obtain the
intermediate feature fT0 as follows:


Mask1 = softmax conv ( fT ) ⊗ conv ( fN + fR )
(7)
(8)
fT0 = Mask1 ⊗ conv ( fN + fR ) + conv ( fT )
where ⊗ means the matrix multiplication. To perform a more
adequate fusion, we use features obtained in the first stage fT0
as inputs for the second stage, and repeat the above process
to obtain the final fused features fTinter as follows:



Mask2 = softmax conv fT0 ⊗ conv ( fN + fR )
(9)
fTinter = Mask2 ⊗ conv ( fN + fR ) + conv ( fT ) .

(10)

The features for the other two modalities are processed similarly as described above, to obtain the final fused features for
different modalities, i.e., fNinter and fRinter .
This module improves the exploitation of complementary
information through intermodal interaction while preserving the loss of specific information in subsequent shared
networks. In addition, the similarity-based adaptive interactive fusion enables the fused features to better maintain
intermodal consistency and avoid feature conflicts. Finally,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.

YANG et al.: TIENet: A TRI-INTERACTION ENHANCEMENT NETWORK FOR MULTIMODAL PERSON ReID

compared to single-stage fusion, two-stage interaction allows
for gradual information transfer between modalities and more
adequate feature fusion, which is demonstrated in subsequent
experiments.

horizontal strips. Referring to the local feature extraction
methods in [45], we also use a uniform segmentation technique
to obtain local features, after which we impose constraints on
the cross-entropy loss function for each part of each modality
id
Lpart
=−

D. Intramodal Feature Fusion
As the human body structure provides rich spatial information, we propose to utilize the structural relationships within
the human body to extract more effective features of pedestrians under challenging conditions. Concretely, these structural
relationships are independent of specific modalities, ensuring
consistency across different modalities. Therefore, we add an
intramodal fusion module on top of the output features FT ,
F N , and FR of layer-4, to enhance the connection between
neighboring regions. Taking FT as an example, we conduct
the intramodal interaction as
FTfus = α (FT )

9857

(11)

where α(·) represents the location attention mechanism.
Following the nonlocal neural network [44], α(·) is defined
as:


1 X θ(FTl )> φ(FTq )
e
g FTq
(12)
α FTl =
C(FT ) ∀q
where C(·) is the normalization factor. l represents the target
location that refers to the index of the focused position, and
q refers to the index of the position that is being computed
for its correlation with the target location. θ(·), φ(·), and g(·)
perform dimension
transformation on the input matrix, and
q
l >
thus, eθ(FT ) φ(FT ) calculates the similarity between l and q. By
traversing all positions q globally, we can get the response of
the global feature to position l, and it enables the fusion of
contextual features within modalities.
In this way, a local feature contains not only its own attribute
information, but also the content information of neighboring
local features. With the subsequent local cross-entropy loss,
the uniqueness of the local features and the consistency of the
global features are well-preserved.
The features FTfus , F Nfus , and FRfus ∈ Rc×h×w are obtained
by taking each of the features through the above interactions.
Then, we obtain features for the three modalities, i.e., FTfin , F Nfin ,
and FRfin ∈ Rd×1 , through GAP followed by one fully connected
(FC) layer. The final representation is the concatenation of all
modality features as follows:


F = FTfin , F Nfin , FRfin ∈ R3d×1
(13)
where [·, ·, ·] denotes the concatenation operation.
E. MAS Loss
To achieve contextual interaction, we propose the IMMF
module; however, while this method captures global features,
it may weaken local salient features. To address this issue, we
introduce fine-grained cross-entropy constraints in the feature
processing, enabling the model to concentrate more on the
salient information of each part.
In single-modal ReID, a simple and typical approach for
local-level feature maps is to segment the body image into

M X
P
X

0

ymp log ymp

(14)

m=1 p=1

where M is the number of modalities,0 P is the number of strips,
ymp is the ground-truth label, and ymp is the predicted label of
the local features. Similarly, to ensure pedestrian differences
in the global features, a cross-entropy loss function is used
globally
id
Lglobal
= −ylog y0
(15)
where y is the ground-truth label of the global features and y0
is the predicted label. The final cross-entropy loss function is
as follows:
id
id
LCE = Lpart
+ Lglobal
.
(16)
To simultaneously model the distribution of intermodal and
intramodal features, we propose an MAS loss function. Specifically, we control the intermodal distance by constraining the
distribution of modal feature centers, which ensures that the
modalities remain distinct from each other after the interaction,
preserving modality-specific features.
Since each modality has numerous identities in a batch, we
calculate the center for each identity in the feature space. For
IT , the center is defined as
K

CT =

1 X fin
FTk
K

(17)

k=1

where K is the number of sample features for the same identity
in a batch, and FTfink is the final features of ITk . Similarly, we
compute the feature centers of the other two modalities to
obtain CR and C N .
The intermodal feature distribution is controlled by imposing a constraint on the distance between the centers of the two
modal features as follows:
ˇ
ˇ
ˇ
ˇ
Linter = ˇm − kCR − CT k2 ˇ1 + ˇm − kCR − C N k2 ˇ1
ˇ
ˇ
+ ˇm − kC N − CT k2 ˇ1
(18)
where m denotes the margin between two modalities, and
|·|1 represents the L1-norm. This term serves as the modalseparation loss to preserve modality-specific information
during fused feature learning.
Meanwhile, the intramodal features vary greatly due to
the large variations in pose, camera view, and background.
Therefore, intramodal constraints are required and essential
to enhance the robustness of pedestrian features to obtain
compact feature distribution. Taking IT with feature center CT
as an example, the intramodal loss LT is designed by pulling
its feature FTfin to the corresponding center CT of the same
identity within the same modality, which is defined as follows:
K

LT =

1 X fin
2
FTk − CT .
2
k=1

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.

(19)

9858

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 6, JUNE 2025

The same operation is performed for the other modalities
and thus gets LN and LR . By combing the regularization on
each modality together, our intramodal aggregation loss Lintra
arrives at
Lintra = LT + LR + LN .
(20)

TABLE I
I MPORTANCE OF M ULTIMODAL I MAGES ON RGBNT201. “+” D ENOTES
D IFFERENT M ODAL C OMBINATIONS

Finally, by integrating the intermodal separation loss Linter and
intramodal aggregation loss Lintra , the proposed MAS loss is
defined as
LMAS = Linter + δLintra
(21)
where δ is the hyperparameter to balance the two loss terms
between Linter and Lintra .
IV. E XPERIMENTS
A. Datasets and Evaluation Protocols
1) Datasets: RGBNT201 [7] is the first multimodal person
ReID dataset. It consists of 4787 image triplets (IN , IT , and
IR ) from 201 distinct individuals, with 141 individuals in
the training set, 30 in the validation set, and another 30
in the test set. MARKET1501 RGBNT [37] is an extension
of the Market1501 dataset [2] to incorporate multimodal information. The training and testing splits align with the original
Market1501 dataset, with 750 identities used for training and
751 identities used for testing.
2) Evaluation Protocols: According to existing person
ReID methods, the cumulative matching characteristics
(CMCs) at RankR (R = 1, 5, and 10) (in %) and mean average
precision (mAP) (in %) are used as evaluation metrics.
B. Implementation Details
In our PyTorch-based experiment, we utilize a computing
platform with a GeForce RTX 3090 GPU. The experiment
commences with an initial learning rate of 0.001, which is
reduced by a factor of 10 at the 20th and 40th epochs. Each
mini-batch consists of eight samples, and feature similarity is measured using the Euclidean distance. We fine-tune
the network using stochastic gradient descent (SGD) with a
momentum of 0.9 and a weight decay of 0.0005.
The intermodal interaction results in feature dimensions of
batch size 256 × 64 × 32, while the intramodal interaction
produces feature dimensions of batch size c × h × w = 2048 ×
16 × 8, with the feature maps evenly divided into six stripes.
An FC layer is employed to reduce the dimensionality of each
part feature to 128. Consequently, each modal feature has a
dimensionality of 6 × 128 = 768, and the final feature for an
individual is of size 768 × 3 = 2304.
C. Importance of Multimodal Data
In order to verify that the introduction of multimodalities
can improve the accuracy of person ReID by introducing
complementary information, we conduct experiments on the
dataset RGBNT201 with one modality, two modalities, and
three modalities of data, respectively. In the training and
testing phases, there is no intermodal interaction for the onemodal data, so to be fair, we use only ResNet50 as the
feature extraction network, and use the cross-entropy loss

part

global

function Lid + Lid
and the MAS loss LMAS to guide
the network training. The experimental results are shown in
Table I. The experimental results show that the introduction of
more data has a better performance, which is consistent with
the theoretical expectations.
D. Comparison With the State-of-the-Art Methods
To demonstrate the effectiveness of our method, we compare
it not only with existing multimodal person ReID methods,
i.e., PFNet [7], DENet [49], LRFNet [17], IEEE [37], and
MMCF [50], but also with multimodal vehicle ReID method
HAMNet [35]. Additionally, to further demonstrate the superiority of complementary information interaction, we compare
the single-modal person ReID methods by extracting features
separately for each modality and then concatenating them as
the final representation, including MLFN [21], HACNN [22],
OSNet [46], RRID [47], APNet [48], and RGA [45].
The experimental results, as shown in Table II, demonstrate
the significance of feature interaction between modalities.
Although single-modal methods achieve certain effectiveness
through feature concatenation, multimodal methods show
superior precision due to intermodal interaction. Specifically,
our method achieves the best performance among these methods on both RGBNT201 dataset and MARKET1501 RGBNT
dataset. In RGBNT201, our method gets a mAP of 54.35%,
exhibiting an improvement of at least 1.32% over other stateof-the-art methods and achieves a Rank1 accuracy of 54.43%,
surpassing the highest-performing method by a margin of
1.20%. In MARKET1501 RGBNT, our method achieves a
mAP of 67.38% and a Rank1 accuracy of 86.05%, showing
an improvement of 2.21% and 1.22% over the best-performing
method.
E. Ablation Study
The proposed method consists of four components, i.e., the
SFI module, IMMI module, IMFF module, and MAS loss.
1) Effectiveness of Each Module: In practice, we construct
the baseline by using only ResNet50 as the backbone, simply
concatenating the three modal features as the final representation, and guiding the model training using the cross-entropy
part
global
loss function Lid and Lid . To verify the impact of each
component on the model performance, we introduce the above
components sequentially based on the baseline in RGBNT201
dataset. The experimental results are shown in Table III.
Each row of Table III shows that each component has
a positive impact on our model. First, the IMMI module

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.

YANG et al.: TIENet: A TRI-INTERACTION ENHANCEMENT NETWORK FOR MULTIMODAL PERSON ReID

9859

TABLE II
C OMPARISON W ITH THE S TATE - OF - THE -A RT M ETHODS ON RGBNT201 AND MARKET1501 RGBNT. ∗ I NDICATES O UR I MPLEMENTATION . T HE B EST
I S M ARKED IN B OLD B LACK AND THE S ECOND -B EST I S IN U NDERLINE

TABLE III
A BLATION E XPERIMENTS FOR E ACH C OMPONENT OF O UR N ETWORK ON
RGBNT201

TABLE IV
A BLATION S TUDY OF P OOLING M ETHODS IN SFI ON RGBNT201 AND
M ARKET 1501 RGBNT

Fig. 6. t-SNE maps of features for different pooling methods in SFI.
(a) Original features. (b) Features enhanced using GAP (our method).
(c) Features enhanced using GMP.
TABLE V
A BLATION

implements intermodal interaction and fully fuses intermodal
complementary information; the IMFF module utilizes location attention to achieve intramodal feature fusion, making
full use of intramodal contextual structural information; MAS
enhances feature complementarity and robustness by controlling intermodal and intramodal feature distributions; and
finally, we validate the effectiveness of the dual data flows
by sequentially introducing the spatial data flow (SFI S) and
the frequency data flow (SFI F), thereby demonstrating the
necessity of these dual data flows for fine-grained feature
enhancement in both the spatial and frequency domains.
2) Further Explanation of SFI: To enhance the guidance
of amplitude features on the phase, we apply GAP to the
amplitude features and utilize softmax to generate channel
attention weights for each channel of the phase features.
As shown in Table IV, we replace GAP with global max
pooling (GMP), and the results demonstrate that GMP does
not consider global information, which negatively impacts the
model. To more intuitively showcase the advantages of our

E XPERIMENTS FOR I NTERMODAL
RGBNT201 ( IN %)

I NTERACTION

ON

method, we plot the t-SNE feature maps using the approach
described in Section I, as shown in Fig. 6. Fig. 6(a) represents
the original features, Fig. 6(b) represents the features enhanced
using GAP (our method), and Fig. 6(c) represents the features
enhanced using GMP. It is evident that, compared to Fig. 6(b),
Fig. 6(c) does not effectively distinguish between different
modalities, thus failing to fully utilize the global information
of the amplitude for feature enhancement.
3) Superiority of IMMI: To further validate the superiority of our intermodal interaction approach, we replace the
IMMI with the intermodal feature summation (SUM) and
the cross-modal interacting module (CIM) [37] in our model.
In addition, in the IMMI module, we replace the two-stage
progressive interaction with the single-stage interaction, defining this operation as single-stage IMMI (I) and multistage
interactions (II-IV). The experimental results are presented in
Table V. It can be seen that our method outperforms the others;
this is because the simple summation introduces too much
information that destroys the original modal features. CIM
utilizes the channel attention to filter the important features

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.


codex
前两篇已经读完。`TIENet` 的方法线索很清楚：先用频域可视化把“融合不够”具体化成“幅度和相位承载的信息不同”，再把三种交互模块顺着这个观察展开。`Anytime Retrieval` 更像是先重划任务边界，用新数据集把传统、跨模态、换衣三个分裂任务合并成六场景问题。继续读终身文本提示和可见红外蒸馏两篇。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Two-stage knowledge distillation for visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 169 (2026) 111850

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Two-stage knowledge distillation for visible-infrared person re-identification
Jiangming Shi a , Xiangbo Yin b , Demao Zhang b , Zhizhong Zhang c , Yuan Xie c , Yanyun Qu a,b
a
b

,∗

Institute of Artificial Intelligence, Xiamen University, Xiamen, 361005, China
School of Informatics, Xiamen University, Xiamen, 361005, China

c School of Computer Science and Technology, East China Normal University, Shanghai, 200062, China

ARTICLE

INFO

Keywords:
Visible-infrared person
Re-identification
Knowledge distillation

ABSTRACT
Visible-infrared person re-identification (VI-ReID) is an important retrieval task that has recently sparked
interest due to the requirements for continuous 24-hour surveillance. VI-ReID aims to retrieve specific visible or
infrared person images in one modality based on a query from the other modality. Visible and infrared images
have different spectra, leading to huge modality gap that is major challenge for VI-ReID. Recent methods reduce
the gap, but they ignore intra-modality discrepancy. Besides, these methods require well-annotated crossmodality data, but gathering such data is time-consuming and labor-intensive. In this paper, we propose a novel
Two-Stage Knowledge Distillation method (TSKD) for VI-ReID, which adopts a simple-to-difficult strategy for
cross-modality feature alignment and explores a way to reduce annotation costs by using only a small number
of labeled data. TSKD consists of three novel components: soft-identity learning (SI), self-mimic learning
(SM), and mutual-distillation learning (MD). SI first generates pseudo-labels with confidence for unlabeled
data, thereby decreasing the annotation cost. After that, SM learns the prototype for each person in special
modality and minimizes the intra-modality discrepancy. Finally, MD performs mutual distillation for crossmodality feature alignment in the set-level measurement rather than the instance measurement for each person.
Importantly, we demonstrate that TSKD achieves stronger robustness under weak supervision. Our experimental
results on two VI-ReID benchmarks demonstrate the effectiveness of TSKD under both full-supervision and
weak-supervision settings. The code is released at https://github.com/shijiangming1/TSKD.

1. Introduction
Person re-identification (ReID) has attracted increasing research interest in video surveillance due to its wide applications in smart city infrastructure and public security [1,2]. However, traditional ReID methods heavily rely on the visible appearance of pedestrians, which can
be unreliable in low-light conditions. With the capability to seamlessly
switch between visible and infrared modes, many surveillance cameras
enable continuous 24-hour monitoring. Consequently, the importance
of visible-infrared ReID (VI-ReID) has increased significantly.
The substantial discrepancy between visible and infrared images
makes VI-ReID much more challenging than conventional ReID focusing solely on visible images [3,4]. Many existing works try to
mitigate modality discrepancy through alignment at either the image or
feature level. The former methods generate synthetic images bridging
the modality discrepancy [5]. The latter methods aim to narrow the
modality gap by exploiting a global [6] or local [7] representation.
These methods mentioned above are too impatient to narrow the
modality discrepancy by aligning the cross-modality features in a single
stage, which disregards the fact that the intra-modality discrepancy is

substantially smaller than inter-modality, as shown in Fig. 1. Specifically, these methods randomly select samples with the same identity
from two modalities and align their features. In fact, these methods
have to face two challenges: (1) These methods aim to resolve the
challenging many-to-many alignment problem, where multiple visible
samples must be aligned with corresponding infrared samples of the
same identity, and vice versa. (2) These methods rely heavily on wellannotated cross-modality data, which demands considerable time and
effort to generate. This issue is especially noticeable when persons
reappear at long intervals or across different locations.
Given these challenges, two questions arise: Is it possible to simplify
the many-to-many alignment problem to a one-to-one alignment problem? Is it possible to train a visible-infrared person ReID model using
a small number of labels? To address the above-mentioned problems
and inspired by the success of knowledge distillation [9] in transferring knowledge, we design a straightforward yet effective two-state
knowledge distillation framework to reasonably narrow the variances
of features and decrease the annotation cost for VI-ReID, which unifies soft-identity learning, self-distillation, and mutual-learning. Softidentity learning generates pseudo-labels with confidence for unlabeled

∗ Corresponding author at: Institute of Artificial Intelligence, Xiamen University, Xiamen, 361005, China.

E-mail address: yyqu@xmu.edu.cn (Y. Qu).
https://doi.org/10.1016/j.patcog.2025.111850
Received 9 September 2024; Received in revised form 9 January 2025; Accepted 12 May 2025
Available online 28 May 2025
0031-3203/© 2025 Published by Elsevier Ltd.

Pattern Recognition 169 (2026) 111850

J. Shi et al.

Fig. 1. Distance comparison of embedding features for two identities in visible and infrared images from the SYSU-MM01 [8] test set. Figure (a) shows the results of Baseline [7],
while Figure (b) shows the results of TSKD. TSKD achieves closer intra- and cross-modality proximity for the same individual, while enhancing the distinction between different
identities.

data, addressing the expensive annotation problem. Self-mimic learning
obtains modality-specific prototypes for each person, which guide the
learning process through self-distillation. A modality-specific prototype
is the average of all features of an identity. Mutual-distillation learning
reduces the modality discrepancy between visible and infrared by
distilling infrared features into visible features for the same person
and vice versa. As shown in Fig. 1, we present the visualization of
the Euclidean distance of visible and infrared embedding from two
subjects compared between baseline [7] and TSKD. We can observe that
baseline only has a good effect on intra-modality matching but does not
well handle inter-modality matching. Our TSKD is effective not only for
intra-modality matching but also for inter-modality matching.
To summarize, our contributions are threefold:

and local fine-grained features. TS [15] developed a dual-stream network using densely semantically aligned part images to guide feature
learning.
However, due to significant modality discrepancies, many existing
single-modality methods struggle with the VI-ReID task, limiting their
effectiveness in 24-hour surveillance scenarios.
2.2. Visible-infrared person re-identification
Several approaches have been presented [16,17] to address the VIReID, focusing on developing effective embedding networks to achieve
alignment at both the feature and image levels. To be specific, featurelevel alignment methods usually transform data from multiple modalities into a common embedding, where similar characteristics are emphasized and dissimilar ones are minimized. DDAG [7] achieves superior alignment at part-level and graph-level features. Image-level
alignment methods focus on learning global representations or local
representations that can capture holistic similarities between images of
different modalities. Recently, GANs-based methods [5] have been used
for image-level alignment to address the VI-ReID task. HCML [18], the
closest related work to our method, is a two-stage feature transformation. In the first stage, HCML first learns the modality-shared features,
and subsequently extracts both modality-specific and modality-shared
features within the feature subspace created during this stage. However, HCML loses the discriminability somewhat due to the transformation in the first stage, so when it conducts feature alignment in
the second stage, it cannot achieve good results as the two stages are
separated.
The methods mentioned above primarily aim to reduce the differences between visible and infrared modality through single-stage
alignment, which often leads to suboptimal performance. To this end,
we design a two-stage feature alignment framework [19], which reduces intra-modality differences using self-mimic learning at first, and
then reduces inter-modality variance using mutual-distillation learning. Additionally, we found that reducing intra-modality differences
facilitates the reduction of inter-modality discrepancy.

• We propose a unified framework called TSKD for VI-ReID that
operates effectively in both fully supervised and weakly supervised scenarios, significantly broadening its versatility and practical utility. TSKD takes the easy-to-hard strategy to conduct
cross-modality feature alignment.
• We employ soft-identity learning to generate pseudo-labels with
confidence, self-mimic learning to align intra-modality features,
and mutual-distillation learning to align the cross-modality features.
• Extensive experiments on the SYSU-MM01 and RegDB datasets
demonstrate the effectiveness of TSKD in both fully supervised
and weakly supervised scenarios.
2. Related work
In this section, we provide a concise overview of three relevant
topics:
2.1. Single-modality person re-identification
With the advancement of deep learning, many works [10,11] have
achieved significant progress and notable success in single-modality
ReID. The work [12] extracted features by fine-tuning a pre-trained
CNN to reduce classification loss. To fully utilize different partial
information from person images, HPM [13] employed partial features
at various horizontal pyramid scales to build a comprehensive person
representation. AANet [11] constructed an attention-aligned network
to take advantage of imprecise channel attention and coarse spatial
attention across uniform scales. Recognizing the benefits of multibranch networks, MGN [14] designed a sliced network combining a
multi-branch structure with a dual learning strategy for representation
and metric learning, aimed at extracting both global coarse-grained

2.3. Knowledge distillation
Knowledge Distillation is a method in machine learning used to
distill the knowledge from a complex, high-accuracy model (Teacher)
into a smaller, more efficient model (Student). Deep mutual learning [20] involves two networks learning collaboratively, where they
continuously teach and improve each other during the training stage.
To improve generalization ability, mutual learning is extended to the
ensemble of student networks by using a gated logit function in the
work [21]. Different from the work [21], a soft target is dynamically
2

Pattern Recognition 169 (2026) 111850

J. Shi et al.

Fig. 2. The flowchart of TSKD. In Stage-I, self-mimic learning with the loss 𝑆𝑀 is conducted to decrease the intra-modality divergence within an identity. In Stage-II, mutual
distillation learning is added to make the distributions between visible and infrared modality for each person similar to reduce inter-modality divergence. Stage-I is run with the
loss 𝑆𝑀 for the first 𝑇𝑚 epochs, and at the epoch 𝑇𝑚+1 , 𝑀𝐷 is added until the final epoch 𝑇𝑛 .

generated to improve performance through collaborative learning in
KDCL [22]. OMKD [23] proposed knowledge is transferred by an ensemble of sub-network classifiers. Rather than transferring the knowledge from each student network, we utilize a mutual learning method
to minimize the distribution discrepancy between cross-modality images. SSL [24] addressed the unsupervised ReID problem by mimicking
the softened similarity to learn robust and discriminative features. In
the study of small-scale pedestrian detection, SML [25] simulates the
rich feature representations of large-scale person ReID to enhance those
of small-scale person ReID.
Unlike above methods, we employ a simulation method to make the
features of the same identity converge towards their prototypes. Additionally, our approach effectively reduces intra-modality information
redundancy, which often hinders the performance of VI-ReID.

3.2. Method overview

In this section, we provide a detailed explanation of the proposed
TSKD, which is a flexible method for both fully supervised and weakly
supervised VI-ReID.

Fig. 2 illustrates the learning process for a specific person. The
orange and green points represent the features of the visible and
infrared modalities. The red points represent the prototypes of the
visible and infrared modalities for a specific person. Unlike previous
approaches, the learning process in TSKD is split into two stages,
starting with the easier task and progressing to the more difficult
task. Before 𝑇𝑚 epochs, our method only uses the self-mimic loss to
train the model. After 𝑇𝑚 epochs, mutual-distillation loss is added to
optimize the network. Specifically, in Stage-I, samples are drawn closer
to their respective prototypes, which aligns intra-modality features
for each person and prepares for inter-modality feature alignment. In
Stage-II, mutual-distillation is conducted by making the distributions
of the visible and infrared modalities similar. We apply self-mimic
learning within each modality to reduce intra-modality variance and
use mutual-distillation to address inter-modality discrepancy for each
person. In this work, we use the Dynamic Dual-attentive aggregation
model (DDAG) as our baseline. The DDAG model has two streamlines
to process visible and infrared images separately, and the architecture
of our TSKD is shown in Fig. 3.

3.1. Problem definition

3.3. Soft-identity learning

3. Proposed method

𝑁𝑉

Suppose we are given a VI-ReID dataset. Let 𝑉𝐿 = {𝑣𝑙𝑖 }𝑖=1𝐿 be

We first employ an identity classifier that is trained using labeled
samples to establish a foundation for distinguishing between different
identities, as follows:

𝑉
𝑢 𝑁𝑈

a labeled visible dataset with 𝑁𝐿𝑉 samples and 𝑉𝑈 = {𝑣𝑖 }𝑖=1 be a
𝑁𝑅

unlabeled visible dataset with 𝑁𝑈𝑉 samples. Similarly, let 𝑅𝐿 = {𝑟𝑙𝑖 }𝑖=1𝐿

𝑁𝑡

𝑁𝑅

be a labeled infrared dataset and 𝑅𝑈 = {𝑟𝑢𝑖 }𝑖=1𝑈 be a unlabeled infrared
dataset. The sum of 𝑁𝐿𝑉 and 𝑁𝑈𝑉 , as well as the sum of 𝑁𝐿𝑅 and 𝑁𝑈𝑅 ,
represent the sample sizes of visible and infrared datasets, respectively.
Let 𝑦𝑡𝑖 ∈ R𝑃 be the one-hot encoded annotation of 𝑡𝑙𝑖 , where 𝑡 ∈ {𝑣, 𝑟}
and 𝑃 refers to the total number of identities. Our purpose is to develop a unified model that can effectively handle both fully supervised
and weakly supervised VI-ReID tasks, thereby maximizing performance
across different levels of labeled data availability. To achieve this, we
utilize the available datasets: 𝑉 = {𝑉𝐿 } and 𝑅 = {𝑅𝐿 } in the fullsupervised scenario, and 𝑉 = {𝑉𝐿 , 𝑉𝑈 } and 𝑅 = {𝑅𝐿 , 𝑅𝑈 } in the
weak-supervised scenario.

𝜃 = arg min −
𝜃

𝐿
( ( ( ) ))
1 ∑ 𝑡
𝑦𝑖 log 𝐶 𝑓 𝑡𝑙𝑖 ; 𝜃 ,
𝑡
𝑁𝐿 𝑖=1

(1)

where 𝑓 (⋅) and 𝐶(⋅) denote a function for extracting features of images
and an identity classifier, respectively. The 𝑦𝑡𝑖 is a one-hot encoded
vector corresponding to the true identity. The optimization process
of Eq. (1) aims to minimize the cross-entropy loss between predicted
distribution by 𝐶 and true class label 𝑦𝑡𝑖 .
We then use the identity classifier to predict pseudo-labels 𝑦̂𝑡 for
unlabeled samples if 𝑁𝑈𝑡 > 0, as follows:
(
) 𝑁𝑡
𝑦̂𝑡 = {𝐶 𝑓 (𝑡𝑢𝑖 ); 𝜃 }𝑖=1𝑈 ,
3

(2)

Pattern Recognition 169 (2026) 111850

J. Shi et al.

Fig. 3. Overview of the proposed TSKD (best viewed in color). Provided visible-infrared data, TSKD initially generates labels with confidence (the strength is indicated by thickness).
Subsequently, the process involves two stages: in Stage-I, intra-modality features are aligned, and in Stage-II, both intra- and inter-modality features are aligned.

{
𝑦̂𝑡 ∶=

if 𝑁𝑈𝑡 = 0,

[𝑦𝑡 ],
[𝑦𝑡 , 𝑦̂𝑡 ],

otherwise,

𝑀

(3)

𝑂𝑅𝑝 =

where 𝑡 ∈ {𝑣, 𝑟} denotes the visible or infrared modality, and 𝑁𝑈𝑡 = 0
denotes all samples have labels.
Pseudo-labels inevitably contain noise and even person annotation
is not immune to it [26], which significantly decreases the performance.
The work [27] indicated that deep neural networks learn simple structures before adjusting to noise labels. Inspired by this, we evaluate
confidence for each label. Specifically, we adopt a two-component
Gaussian Mixture Model to fit the loss distribution, as follows:
(
( ( ))) 𝑁 𝑡 +𝑁 𝑡
𝑡 = {log 𝑃 𝑦̂𝑡𝑖 ∣ 𝐶 𝑓 𝑡𝑖 }𝑖 𝐿 𝑈 ,
(4)

𝑝
1 ∑
𝑓 (𝑟𝑝𝑖 ) ⋅ 𝑤𝑝𝑟
𝑖 ,
𝑀𝑝 𝑖=1

(9)

where 𝑉 𝑝 = {𝑣𝑝1 , … , 𝑣𝑝𝑁 } represents the set of 𝑁𝑝 samples correspond𝑝

ing to identity 𝑝 in the visible modality, while 𝑅𝑝 = {𝑟𝑝1 , … , 𝑟𝑝𝑀 }
𝑝
represents the set of 𝑀𝑝 samples corresponding to identity 𝑝 in the
𝑝𝑣
𝑝𝑟
𝑝
infrared modality. 𝑤𝑖 and 𝑤𝑖 are the confidences of 𝑣𝑖 and 𝑟𝑝𝑖 ,
respectively.
We employ every sample of identity 𝑝 to align its corresponding
prototype 𝑂𝑉 𝑝 and 𝑂𝑅𝑝 in each modality. The self-mimic loss 𝑆𝑀 is
formulated as:
𝑆𝑀 = 𝑉𝑆𝑀 + 𝑅
𝑆𝑀
𝑁

𝑝(𝑡 ∣ 𝜃) =

2
∑

𝜋𝑘 𝜙(𝑡 ∣ 𝑘),

=

(5)

𝑝=1 𝑖=1

𝑘=1

where 𝜋𝑘 denotes the mixture coefficient, while 𝜙(𝑡 ∣ 𝑘) is the
probability density of the 𝑘th component. We compute the posterior
probability to determine the confidence of the 𝑖th sample as follows:
(
)
𝑤𝑡𝑖 = 𝑝 𝑘 ∣ 𝑡𝑖 ,
(6)
(
)
where 𝑘 is the Gaussian component with the lower mean, and 𝑝 𝑘 ∣ 𝑡𝑖
indicates the responsiveness of 𝑡𝑖 at the 𝑘th component.
So far, we have obtained the labels 𝑦̂𝑡 and corresponding confidence 𝑤𝑡 for visible and infrared images. For a given sample with the
{
}
identity annotation 𝑡𝑖 , 𝑦̂𝑡𝑖 , we utilize the confidence to penalize the
noise during optimization for identity-discrimination learning, which
is formulated as:
( (
( ( ))))
𝑖𝑆𝐼 = −𝑤𝑡𝑖 log 𝑃 𝑦̂𝑡𝑖 ∣ 𝐶 𝑓 𝑡𝑖
.
(7)

+

𝑝=1 𝑗=1

The central feature of a identity serves as a common representation for the identity, which we refer to as the prototype. We reduce
intra-modality variance by grouping samples with their corresponding
prototypes within each modality. Existing methods like center loss [28]
refresh centers using only small batches and treat all samples equally,
which may create a narrow-scope prototype. We contend that an effective prototype should be comprehensive, consistent, and distinct.
Therefore, we propose a self-mimic (SM) learning to iteratively refine
prototypes.
SM is influenced by self-distillation [29], where valuable information is distilled at early epochs and then used to enhance the training
process in subsequent stages. For each training epoch, the identity
prototypes are updated by distilling the features of all samples whose
identity is same. The prototypes for identity 𝑝 in both modalities are
formulated as follows:

(10)

‖ 𝑝
‖2
‖𝑓 (𝑟𝑗 ) − 𝑂𝑅𝑝 ‖ .
‖
‖2

3.5. Mutual distillation
In contrast to traditional ReID, which focuses solely on learning a
discriminative representation of identity, VI-ReID performs matching
across modalities, where a query is utilized to retrieve for corresponding images in another modality. The main obstacle lies in huge domain
gaps between visible and infrared images of the same individual, which
is larger than the gap within a single modality. Previous works [6,30]
have addressed this issue through unidirectional matching of visible
features to infrared features and the reverse However, the information
contained in these two modalities is not evenly distributed, and oneway mapping will result in the loss of discriminative features or the
introduction of undesired artifacts.
To avoid unidirectional visible-infrared feature alignment, we propose a mutual distillation (MD) learning to mitigate the distribution discrepancy between modalities in both directions. In contrast to methods
like HCML [18], which independently map inter-modality features to a

𝑁

𝑝
1 ∑
𝑓 (𝑣𝑝𝑖 ) ⋅ 𝑤𝑝𝑣
𝑖 ,
𝑁𝑝 𝑖=1

𝑀𝑝
𝑃 ∑
∑

The SM learning training pipeline is outlined below:
(1) At the first epoch, the network is trained using the loss 𝑆𝐼 and
1 for each
baseline method, and initializing the prototypes 𝑂𝑉1 𝑝 and 𝑂𝑅
𝑝
identity 𝑝 using Eqs. (8) and (9), respectively.
(2) For the 𝑒th epoch, 𝑒 = 2, … , 𝐸, the network is trained by added
𝑒−1
𝑆𝑀 , where 𝑂𝑉𝑒−1
𝑝 and 𝑂𝑅𝑝 are fixed to train network. Following the
𝑒 are adjusted.
end of training, the prototypes 𝑂𝑉𝑒 𝑝 and 𝑂𝑅
𝑝
SM has several advantages: (1) Unlike the center loss, which updates
prototypes in a localized, batch-wise manner, our approach leverages prototypes to provide a comprehensive understanding, resulting
in more inclusive identity features. (2) The prototypes are updated
together with the network training, the ID representation becomes
more discriminative. (3) The prototypes remain fixed to ensure stable
supervision, but the center loss method employs dynamical prototypes.

3.4. Self-mimic learning

𝑂𝑉 𝑝 =

𝑃 ∑
𝑝
∑
‖
‖2
𝑝
‖𝑓 (𝑣𝑖 ) − 𝑂𝑉 𝑝 ‖
‖
‖2

(8)
4

Pattern Recognition 169 (2026) 111850

J. Shi et al.

common embedding, our method integrates alignment with the encoding to improve cross-modality feature matching. We select 𝑛 visible and
infrared feature combinations, represented as 𝐹𝑉 𝑝 = {𝑓 (𝑣𝑝1 ), … , 𝑓 (𝑣𝑝𝑛 )}
and 𝐹𝑅𝑝 = {𝑓 (𝑟𝑝1 ), … , 𝑓 (𝑟𝑝𝑛 )}. We then apply the mutual distillation loss
to make the feature distribution of visible samples similar to that of the
infrared samples, and the same for the reverse.
𝑃
1 ∑ 1
1
𝑀𝐷 =
( 𝐷(𝐹𝑅𝑝 , 𝑠𝑔(𝐹𝑉 𝑝 )) + 𝐷(𝐹𝑉 𝑝 , 𝑠𝑔(𝐹𝑅𝑝 ))),
𝑃 𝑝=1 2
2

Algorithm 1 : Two-stage knowledge distillation.
{
}
{
}
Input: Training set 𝐷 = (𝑉𝐿 , 𝑅𝐿 ) or 𝐷 = (𝑉𝐿 , 𝑉𝑈 , 𝑅𝐿 , 𝑅𝑈 ) , total
training iterations 𝑚𝑎𝑥_𝑒𝑝𝑜𝑐ℎ, iterations of adding mutual distillation
𝑏𝑒𝑔𝑖𝑛_𝑀𝐷.
Initialize: Initialize parameters 𝜃.
1: for 𝑡 = 1 ∶ 𝑚𝑎𝑥_𝑒𝑝𝑜𝑐ℎ do
2:
Update label 𝑦̂𝑡 with Eq. (3)
3:
if 𝑡 < 𝑏𝑒𝑔𝑖𝑛_𝑀𝐷 then
4:
𝑀𝐷 = 0;
5:
else
6:
Compute 𝑀𝐷 with Eq.(11);
7:
end if
8:
Compute 𝑆𝑀 with Eq.(10), 𝑆𝐼 and 𝑇 𝑟𝑖 with Eq.7 and

(11)

𝑠𝑔(⋅) represents the stop-gradient operation and 𝐷(⋅, ⋅) is MMD2 [31].
MMD2 is a well-known non-parametric metric in domain adaptation
and has been empirically proven to perform better than alternatives
like KL divergence. This superior performance is attributed to its robustness and flexibility in capturing differences between complex, highdimensional distributions.
MMD2 is constructed as:
MMD2 (𝐹𝑅𝑝 , 𝐹𝑉 𝑝 ) =

1 ∑∑
𝑘(𝑓 (𝑟𝑝𝑖 ), 𝑓 (𝑟𝑝𝑗 ))
𝑛2 𝑖=1 𝑗=1

+

1 ∑∑
𝑘(𝑓 (𝑣𝑝𝑖 ), 𝑓 (𝑣𝑝𝑗 ))
𝑛2 𝑖=1 𝑗=1

𝑛

𝑛

Eq.(14), respectively;
Optimize 𝜃 based on overall  ( Eq.(13))
Update prototype for each ID in each modality with Eq.(8) and
Eq.(9).
11: end for
9:
10:

𝑛

𝑛

(12)
As shown in Fig. 1, our observation is that intra-modality samples exhibit higher similarity in the feature space compared to crossmodality samples. This implies that mutual distillation learning is inherently more challenging than self-mimic learning. Additionally, intramodality variance adds complexity to inter-modality feature alignment.
To address this, we introduce a two-stage feature alignment strategy,
where we first diminish the intra-modality divergences to tighten the
distributions and then conduct cross-modality feature alignment. The
step-by-step training process is presented in Algorithm 1.

2 ∑∑
−
𝑘(𝑓 (𝑟𝑝𝑖 ), 𝑓 (𝑣𝑝𝑗 )),
𝑛2 𝑖=1 𝑗=1
𝑛

𝑛

−‖𝑠−𝑠′ ‖2

where 𝑘(𝑠, 𝑠′ ) = exp( ‖ 2𝜎 2 ‖2 ) is a Gaussian kernel.
MD presents two major strengths: (1) Bidirectional feature alignment balances the suppression of trivial features in the visible modality and reinforcing critical features in the infrared modality. (2) MD
prioritizes the consistency of representation spaces across numerous
instances spanning both modalities, instead of attempting to align
individual pairs of samples. This approach simplifies the reduction
of statistical differences between the modalities, rather than mandating complete consistency for every pair of samples from different
modalities.

4. Experiment
In this section, we perform extensive experiments to assess our proposed approach in both fully supervised and weakly supervised settings.
To clearly outline our experiments, we first describe the experimental
setup and implementation details. Next, we present results on two
common datasets and conduct ablation studies to analyze the model’s
characteristics.

3.6. The total loss
The total loss contains four terms: the soft-identity loss, the triplet
loss, the self-mimic learning loss, and the mutual learning loss, which
is formulated as:
𝑇 𝑜𝑡𝑎𝑙 = 𝑆𝐼 + 𝑇 𝑟𝑖 + 𝜆𝑆𝑀 𝑆𝑀 + 𝜆𝑀𝐷 𝑀𝐷 ,

4.1. Datasets, setting, and implementation
SYSU-MM01 is an extensive and challenging dataset for VI-ReID [8].
It includes 287,628 visible images and 15,792 infrared images captured
across six camera views (four visible and two near-infrared). The
dataset includes 491 identities in total. For training, there are 22,258
visible images and 11,909 infrared images covering 395 individuals.
For evaluation, we use 3803 infrared images as queries and randomly
select 3010 visible images to form the gallery. RegDB [33] includes 412
individuals, each with 10 visible and 10 infrared images. The dataset
is split randomly, with half allocated for training and the other half for
testing.
Evaluation Protocols. Model performance is assessed using Cumulative Match Characteristic (CMC) and Mean Average Precision (mAP).
For an equitable comparison, results are reported for both the all-search
mode and indoor-search mode on SYSU-MM01 using the official code.
In the all-search mode, images from all available cameras are used
to form the gallery set. Conversely, in the indoor-search mode, only
images from the 1st, 2nd, 3rd, and 6th cameras, which are indoor, are
used for the gallery set. Additionally, results for RegDB are provided
according to the evaluation protocols outlined in the work [34]. This
dataset features two modes: Visible2thermal and Thermal2visible. In
the Visible2thermal mode, visible images are used as queries and
infrared images as the gallery. In the Thermal2visible mode, the roles
are reversed, with infrared images serving as queries and visible images
as the gallery.

(13)

where 𝜆𝑆𝑀 and 𝜆𝑀𝐷 serve as trade-off coefficients. The first 𝑆𝐼 and
the second 𝑇 𝑟𝑖 are soft-identity loss and triplet loss.
To maximize the inter-class discrepancy and intra-class similarity,
we employ the triplet loss [32] in the two modalities. For the visible
modality, let 𝑣𝑎𝑖 denote a randomly selected anchor, 𝑣𝑝𝑜𝑠
denote the
𝑖
feature of the hardest positive sample that has the same identity as 𝑣𝑎𝑖
within the batch and 𝑣𝑛𝑒𝑔
denotes the feature of the hardest negative
𝑖
sample that has a different identity as 𝑣𝑎𝑖 within the batch. The triplet
loss for visible images is expressed as follows:
𝑉𝑇 𝑟𝑖 =

]
𝑁𝑣 [
∑
( 𝑝𝑜𝑠 )‖2 ‖ ( )
( 𝑛𝑒𝑔 )‖2
‖ ( 𝑎)
‖𝑓 𝑣𝑖 − 𝑓 𝑣𝑖 ‖ − ‖𝑓 𝑣𝑎𝑖 − 𝑓 𝑣𝑖 ‖ + 𝛾 ,
‖
‖2 ‖
‖2
+

(14)

𝑖

where 𝛾 is a margin parameter. So do the triplet loss for the infrared
modality formulated as:
]
𝑁𝑟 [
∑
( 𝑝𝑜𝑠 )‖2 ‖ ( 𝑎 )
( 𝑛𝑒𝑔 )‖2
‖ ( 𝑎)
𝑅
=
𝑓
𝑟
−
𝑓
𝑟
−
𝑓
𝑟
−
𝑓
𝑟
+
𝛾
.
(15)
‖
‖
‖
‖
𝑖
𝑖
𝑖
𝑖
𝑇 𝑟𝑖
‖
‖2 ‖
‖2
+
𝑖

The overall 𝑇 𝑟𝑖 loss can be formulated as a combination of 𝑉𝑇 𝑟𝑖 and
𝑅
as follows:
𝑇 𝑟𝑖
𝑇 𝑟𝑖 = 𝑉𝑇 𝑟𝑖 + 𝑅
𝑇 𝑟𝑖 .

(16)
5

Pattern Recognition 169 (2026) 111850

J. Shi et al.

Table 1
Comparisons with advanced methods on SYSU-MM01. The top two results are highlighted in red and blue, representing the 1st and 2nd best
performances, respectively.
Method

Venue

All search

Indoor search

Rank-1

Rank-10

Rank-20

mAP

Rank-1

Rank-10

Rank-20

mAP

D2 RL
AlignGAN

CVPR’19
ICCV’19

28.9
42.4

70.6
85.0

82.4
93.7

29.2
40.7

–
45.9

–
87.6

–
94.4

–
54.3

JSIA
DDAG
HAT

AAAI’20
ECCV’20
TIFS’20

38.1
54.8
55.3

80.7
90.4
92.1

89.9
95.8
97.4

36.9
53.0
53.9

43.8
61.0
62.1

86.2
94.1
95.8

94.2
98.4
99.2

52.9
68.0
69.4

NFS
MCLNet
SMCL
CAJ

CVPR’21
ICCV’21
ICCV’21
ICCV’21

56.9
65.4
67.4
69.9

91.3
93.3
92.9
95.6

96.5
97.1
96.8
98.5

55.5
61.9
61.8
66.9

62.8
72.6
68.8
76.3

96.5
96.9
96.6
97.9

99.1
99.2
98.8
99.5

69.8
76.6
75.6
78.2

FMCNet
DART
CMT
DCLNet
MAUM𝐺
SPOT
SMMD

CVPR’22
CVPR’22
ECCV’22
MM’22
CVPR’22
TIP’22
ICME’22

66.3
68.7
71.9
70.8
61.6
65.3
64.8

–
96.4
96.5
–
–
92.7
91.3

–
98.7
98.9
–
–
97.0
96.6

62.5
66.3
68.6
65.2
60.0
62.3
60.2

68.2
72.5
76.9
73.5
67.1
69.4
76.5

–
97.8
97.7
–
–
96.2
96.1

–
99.5
99.6
–
–
99.1
99.0

74.1
78.2
79.9
76.8
73.6
74.6
73.8

MRCN
MCBD
DPIS
CAJ+
DEEN
𝐺2 𝐷𝐴

AAAI’23
TIFS’23
ICCV’23
TPAMI’23
CVPR’23
PR’23

68.9
71.6
68.2
71.5
74.7
63.9

95.2
95.0
–
96.2
97.6
93.3

98.4
98.4
–
98.7
99.2
97.3

65.5
67.3
65.7
68.2
71.8
60.7

76.0
79.4
71.7
78.4
80.3
71.1

98.3
98.3
–
98.4
99.0
97.3

99.7
99.9
–
99.8
99.8
99.5

79.8
79.9
77.0
82.0
83.3
76.0

WF-CAMReViT
LCNL

PR’24
IJCV’24

68.1
70.2

97.1
96.4

96.9
99.0

65.2
68.0

72.4
76.2

97.2
98.2

98.9
99.8

77.6
80.3

Ours

–

76.6

97.1

99.1

73.0

82.7

98.9

99.8

85.3

Implementation Details. Our code is implemented using PyTorch and
runs on a standard machine equipped with an RTX 3090 GPU. Following [35], a two-stream feature extractor pre-trained on ImageNet is
employed to obtain 2048d features. For data augmentation, we apply
a channel-mixed learning strategy as described in [32]. Our model is
trained using the SGD optimizer with a momentum of 0.9. The initial
learning rate is set to 0.1, employing a warm-up strategy [36], and
decreases by a factor of 0.1 at the 20th epoch and by 0.01 at the 40th
epoch, across a total of 80 epochs. The batch size in our experiment is
48, where we randomly select 6 identities and sample 4 visible and 4
infrared images per identity. The trade-off parameters 𝜆𝑆𝑀 , and 𝜆𝑀𝐷
in Eq. (13) are set to 1 and 0.05, respectively.

4.3. Results on RegDB
We compare our TSKD with advanced VI-ReID methods on RegDB
under the full-supervised setting. As illustrated in Table 2, we achieve
impressive performance and highly consistent performances across both
Visible2thermal and Thermal2visible modes. Especially, TSKD achieves
impressive results with 91.1% Rank-1 accuracy and 81.7% mAP in the
Visible2thermal mode, and 89.9% Rank-1 accuracy and 80.5% mAP
in the Thermal2visible mode. These results significantly outperform
LCNL [50], exceeding its Rank-1 accuracy by 5.5% and mAP by 3.0%
in the Visible2thermal mode, and surpassing it by 5.9% in Rank-1 and
3.6% in mAP in the Thermal2visible mode.
We also compare our TSKD with various VI-ReID methods on RegDB
under the weak-supervised setting. As illustrated in Table 3, our TSKD
can achieve superior performances across both modes. Especially, our
model with 50% labeled data achieves 86.8% Rank-1 and 77.3% mAP
in the Visible2thermal mode, and 82.1% Rank-1 and 75.2% mAP in the
Thermal2visible mode.

4.2. Results on SYSU-MM01
As shown in Table 1, we compare our TSKD with 25 advanced
methods on SYSU-MM01 under the full-supervised setting. Unless otherwise specified, all methods are analyzed in the all-search mode on the
SYSU-MM01 dataset.
Our method achieves impressive results, recording a Rank-1 accuracy of 76.6% and an mAP of 73.0%. Thanks to the proposed SI, SM,
and MD modules, TSKD can reduce the cross-modality gap. Compared
with DDAG whose backbone is similar to TSKD, TSKD outperforms it by
21.8% in Rank-1 accuracy and 20.0% in mAP without adding extra network parameters. Furthermore, our method improved Rank-1 by 1.9%
compared to the second-best DEEN, demonstrating its effectiveness for
VI-ReID.
What is more, we also compare our TSKD with 6 SOTA methods
under the weak-supervised setting: DDAG [7], AGW [35], CAJ [32],
LbA [37], DART [26], and DPIS [38], as shown in Table 3. Our TSKD
gains impressive results with Rank-1 accuracy of 58.2%, 68.5%, and
73.3% with 10%, 25%, and 50% labeled samples, respectively. It shows
that TSKD can produce dependable cross-modality pseudo-labels for
guiding the model optimization. Compared to DPIS, our TSKD surpasses
it by 5.4% in Rank-1 accuracy with 50% labeled samples. Furthermore,
compared to other advanced methods, our TSKD demonstrates impressive performance, highlighting its effectiveness for the weak-supervised
VI-ReID.

4.4. Ablation study
To further validate the effectiveness of each component of TSKD, we
perform ablation studies on the SYSU-MM01 dataset. If no otherwise
specified, we only use 10% labeled samples under the weak-supervised
setting in this section.
As illustrated in Index 1 of Table 4, the baseline of TSKD employs
DDAG equipped with the CE loss and the triplet loss, which gains
65.1% Rank-1 accuracy and 62.3% mAP under full-supervised setting
and achieves 43.0% Rank-1 accuracy and 37.8% mAP under the weaksupervised setting. Next, we present the results after integrating the
proposed SI, SM, and MD components. By replacing cross-entropy loss
with soft identity loss 𝑆𝐼 (Index 2 in Table 4), the performance
improves to 66.3% Rank-1 accuracy and 62.9% mAP under the fullsupervised setting. Besides, the performance reaches 48.3% Rank-1
and 43.2% mAP under the weak-supervised setting. By adding selfmimic learning loss 𝑆𝑀 (Index 3 in Table 4), the performance shows
a significant enhancement, achieving 71.9% Rank-1 and 68.2% mAP
6

Pattern Recognition 169 (2026) 111850

J. Shi et al.

Table 2
Comparisons with SOTA methods on RegDB. The top two results are highlighted in red and blue, representing the 1st and
2nd best performances, respectively.
Methods

Venue

Visible2thermal

Thermal2visible

Rank-1

mAP

Rank-1

mAP

JSIA [39]
DDAG [7]
Hi-CMD [5]
cm-SSFT [40]

AAAI’20
ECCV’20
CVPR’20
CVPR’20

48.5
69.3
70.9
72.3

49.3
63.5
66.0
72.9

48.1
68.1
–
71.0

48.9
61.8
–
71.7

MCLNet [41]
SMCL [42]
CM-NAS [43]
CAJ [32]

ICCV’21
ICCV’21
ICCV’21
ICCV’21

80.3
83.9
85.5
85.0

73.1
79.8
80.3
79.1

75.9
83.1
82.6
84.8

69.5
79.0
78.3
77.8

DART [26]
DCLNet [44]
MAUM𝐺 [45]
SPOT [46]

CVPR’22
MM’22
CVPR’22
TIP’22

83.6
81.2
83.4
80.4

75.7
74.3
78.8
72.5

82.0
78.0
81.1
79.4

73.8
69.5
78.9
72.3

PMT [47]
DPIS [38]
CAJ+ [48]
𝐺2 𝐷𝐴 [49]

AAAI’23
ICCV’23
TPAMI’23
PR’23

84.8
86.3
85.7
74.0

76.5
77.4
79.7
65.5

84.1
82.1
84.0
70.0

75.1
74.4
76.9
62.0

LCNL [50]

IJCV’24

85.6

78.7

84.0

76.9

Ours

–

91.1

81.7

89.9

80.5

Table 3
Comparisons with six advanced methods under the weak-supervised setting, and all methods are evaluated using Rank-1 (%) and mAP (%). All results
are re-implemented using public codes. The top two results are highlighted in red and blue, representing the 1st and 2nd best performances, respectively.
Settings

SYSU-MM01

RegDB

All search

Indoor search

Visible2thermal

Thermal2visible

Rate

Method

Venue

Year

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

10%

DDAG [7]
AGW [35]
CAJ [32]
LbA [37]
DART [26]
DPIS [38]
Ours

ECCV
TPAMI
ICCV
ICCV
CVPR
ICCV
–

2020
2021
2021
2021
2022
2023
–

29.8
32.4
35.6
30.3
31.0
57.7
58.2

27.9
33.4
37.2
29.9
29.6
55.0
56.3

33.6
38.0
38.1
32.9
39.7
62.9
63.6

42.1
49.2
47.1
43.2
22.0
69.8
70.6

44.0
22.8
21.3
41.7
46.2
55.0
56.1

38.6
18.3
17.1
36.7
40.3
50.8
51.9

44.0
20.9
21.2
42.1
45.0
55.1
55.3

37.8
19.4
17.1
36.9
37.0
49.6
50.4

25%

DDAG [7]
AGW [35]
CAJ [32]
LbA [37]
DART [26]
DPIS [38]
Ours

ECCV
TPAMI
ICCV
ICCV
CVPR
ICCV
–

2020
2021
2021
2021
2022
2023
–

39.1
36.9
48.8
49.4
51.9
65.2
68.5

37.0
37.7
89.2
47.4
49.3
61.8
62.2

43.5
43.1
95.8
54.0
57.1
70.5
72.8

51.7
54.3
48.0
62.6
64.3
75.6
76.3

52.5
57.7
57.1
51.3
60.0
77.3
78.1

48.1
49.8
93.6
43.7
49.7
67.7
69.0

52.9
57.1
97.7
47.5
57.4
75.3
76.1

47.9
49.0
64.7
43.7
49.0
66.3
67.2

50%

DDAG [7]
AGW [35]
CAJ [32]
LbA [37]
DART [26]
DPIS [38]
Ours

ECCV
TPAMI
ICCV
ICCV
CVPR
ICCV
–

2020
2021
2021
2021
2022
2023
–

42.3
44.0
59.4
53.4
56.1
67.9
73.3

38.7
40.6
57.0
52.2
55.2
64.2
67.9

49.0
48.6
62.6
58.9
62.5
71.8
77.2

55.5
53.8
64.3
62.6
69.6
76.6
80.8

59.8
70.8
77.4
67.6
78.2
85.6
86.8

53.2
62.7
66.8
61.7
67.0
76.7
77.3

57.0
68.2
74.9
65.8
75.0
81.4
82.1

51.2
61.4
63.1
60.0
68.4
74.1
75.2

Table 4
Ablation study on the SYSU-MM01 dataset under full-supervised and weak-supervised settings.
Index

Approach

All Search
Full-supervised setting

1
2
3
4
5
6

Weak-supervised setting

SI

SM

MD

Rank-1

Rank-10

Rank-20

mAP

Rank-1

Rank-10

Rank-20

mAP

×
✓
✓
✓
×
✓

×
×
✓
×
✓
✓

×
×
×
✓
✓
✓

65.1
66.3
71.9
74.7
74.9
76.6

94.7
94.6
95.8
96.9
96.2
97.1

98.1
97.9
98.6
99.1
98.9
99.1

62.3
62.9
68.2
71.3
72.1
73.0

43.0
48.3
53.5
51.4
56.0
58.2

80.8
84.4
89.1
87.5
90.1
92.3

89.7
91.9
95.3
94.5
95.9
95.2

37.8
43.2
48.6
47.2
51.7
56.3

under the full-supervised setting. Under the weak-supervised setting, it
reaches 53.5% Rank-1 and 48.6% mAP. These improvements highlight
the effectiveness of eliminating redundant intra-modality information.
With the incorporation of the mutual distillation loss 𝑀𝐷 (Index 4 in
Table 4), our TSKD gains 74.7% Rank-1 and 71.3% mAP under the fullsupervised setting, and achieves 51.4% Rank-1 and 47.2% mAP under

the weak-supervised setting, which demonstrates the crucial role of
aligning modality distributions in reducing discrepancies across modalities. To investigate the impact of noise pseudo-labels on SM and MD
(Index 5 in Table 4), we remove the identity confidence and integrate
only self-mimic learning (𝑆𝑀 ) and mutual distillation (𝑀𝐷 ), which
achieves 74.9% Rank-1 accuracy and 72.1% mAP under full-supervised
7

Pattern Recognition 169 (2026) 111850

J. Shi et al.

Fig. 4. Rank-1 and mAP with different mimic learning schemes on the SYSU-MM01 dataset.

setting and achieves 56.0% Rank-1 accuracy and 51.7% mAP under
the weak-supervised setting. Finally, by employing all loss (Index 6
in Table 4), our TSKD gains the highest results in both settings. Our
TSKD achieves 76.6% Rank-1 and 73.0% mAP under the full-supervised
setting, surpassing the baseline by substantial margins of 11.5% Rank-1
and 10.7% mAP. Under the weak-supervised setting, TSKD also gains
58.2% Rank-1 and 56.3% mAP, surpassing the baseline by 15.2% in
Rank-1 and 18.5% in mAP.
The analysis shows that removing identity confidence (Index 5 in
Table 4) leads to a clear drop in performance compared to the configuration with all components (Index 6 in Table 4). Specifically, Rank-1
accuracy decreases from 76.6% to 74.9% under the full-supervised
setting and from 58.2% to 51.7% under the weak-supervised setting,
indicating that noise in pseudo-labels has a noticeable impact on SM
and MD due to the increased risk of misalignment. However, even with
noisy pseudo-labels, SM and MD demonstrate their ability to enhance
performance. For instance, under the weak-supervised setting, Rank-1
accuracy improves from 43.0% (Index 1 in Table 4) to 65.0% (Index 5
in Table 4), showing their effectiveness in handling noise. Additionally,
TSKD (Index 6 in Table 4) significantly reduces the impact of noisy
pseudo-labels by integrating SI with SM and MD, further improving
their combined performance and achieving better overall results.

the prototypes using Eq. (10). The online mimic learning scheme is
similar to the center loss [51] that updates the prototypes as the deep
features change in a more effective way. As shown in Fig. 4, the online
mimic learning achieves 65.2% in mAP which makes an improvement
of 2.3% against the baseline model, while the center loss achieves a
gain of 2.8%. It indicates that the online learning scheme is effective
for Re-ID, and a good prototype leads to a superior performance.
From Fig. 4 we can also observe that our SM learning scheme
achieves the best results and outperforms other schemes by a large
margin. Its main difference from the center loss is described in Section 3.4. The Fig. 4 verifies the effectiveness of our SM learning scheme
which can combine the strengths of offline mimic learning and online
mimic learning to obtain globally viewed, stable, and discriminative
prototypes.
Analysis of self-mimic learning. The goal of SM is to force the
features of each identity’s images captured in the same modality to be
similar. Therefore, we visualize the mean and variance of the distances
between visible images and distances between infrared images for each
identity in Fig. 5. The results show that, compared to the baseline
model, SM significantly reduces the means and variances within the
same modality. Furthermore, we visualize the distribution plot of distances between intra-modality and inter-modality features in Fig. 6. It
is observed that SM effectively clusters the samples of each identity,
benefiting both intra-modality and inter-modality samples.
Analysis of mutual distillation. To clearly show the effectiveness
of MD, we compare three cross-modality feature alignment methods,
including the infrared feature mimics the visible feature (infrared→visible), the visible feature mimics the infrared feature (visible→infrared),
and our MD (infrared↔visible). As shown in Fig. 7, infrared→visible
has the worst performance as there is less information in infrared
images than visible images, making it difficult to map infrared features
to the visible modality. In contrast, our MD achieves the best results.
MD also can mitigate the diversity of feature distributions of intra- and
cross-modality samples. We plot 𝑆𝑀 and mAP at each epoch with and
without MD in Fig. 8. It shows that MD can reduce 𝑆𝑀 after using
𝑀𝐷 , which means that MD can reduce the distances between intramodality images. Meanwhile, as illustrated in Fig. 6, the results also
demonstrate that MD achieves the most compact distribution with the
smallest variance and mean.
Analysis of pseudo-label accuracy. We evaluate the accuracy of
pseudo-labels generated by TSKD with 10% annotation. As shown in
Fig. 9, TSKD achieves higher accuracy compared to DPIS in both RGB
and IR modalities. For the RGB modality, TSKD achieves an accuracy
of 80.7%, outperforming DPIS by 2.5%. Similarly, in the IR modality, TSKD reaches an accuracy of 78.6%, which is 2.2% higher than
DPIS. These results highlight the effectiveness of TSKD in improving
pseudo-label accuracy under weakly supervised settings.
Analysis of two-stage feature alignment strategy. To validate impact
of the two-stage feature alignment strategy, we show the Rank-1 and

4.5. More analysis of our method
Comparison of mimic schemes. To further investigate the effectiveness of the proposed self-mimic (SM) learning, we conduct a comparison of SM with other mimic learning schemes, including offline mimic
learning, online mimic learning, and center loss [51]. The results on
the SYSU-MM01 dataset are reported in Fig. 4.
(1) Offline mimic learning. In this scheme, we firstly train two
traditional person Re-ID models in the RGB and IR modalities using
our baseline model, independently. Then, we compute the prototypes of
𝑂𝑉 𝑝 and 𝑂𝑅𝑝 by collecting all samples in the training set that belong to
ID 𝑝 in the RGB and IR modalities, respectively. After that, we train the
baseline model by mimicking the prototypes using Eq. (10). The offline
mimic learning is conducted in a straightforward way that calculates
the prototypes of all identities using Eqs. (8) and (9). It updates the
prototypes considering all samples and the learned prototypes are fixed
as a guidance in Eq. (10). It is observed in Fig. 4 that the offline mimic
learning scheme achieves the mAP of 62.4% which is smaller than the
baseline model of 62.9%. The reason is that the prototypes are fixed
during updating the network, and a bad prototype fails to push other
identities.
(2) Online mimic learning. This scheme jointly learns the prototypes
of all identities and minimizes the sample distances to their corresponding prototypes. Specifically, we firstly extract the features of the
samples in every training batch. Then we update the prototypes of the
identities in the batch. Finally, we update the features by mimicking
8


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'TP-LReID - Lifelong person re-identification using text prompts.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 172 (2026) 112326

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

TP-LReID: Lifelong person re-identiﬁcation using text prompts
Zhaoshuo Liu a,1 , Zhiwei Guo a,1 , Chaolu Feng
Jinzhu Yang a,b
a

a,b,∗, Wei Li a,b , Kun Yu c , Jun Hu

d,

School of Computer Science and Engineering, Northeastern University, 110819, Shenyang Liaoning, China

b Key Laboratory of Intelligent Computing in Medical Image, Ministry of Education, 110169, Shenyang Liaoning, China
c

College of Medicine and Biological Information Engineering, Northeastern University, 110016, Shenyang Liaoning, China

d Neusoft Reach Automotive Technology Company, 110179, Shenyang Liaoning, China

a r t i c l e

i n f o

Keywords:
Lifelong person re-identiﬁcation
Model generalization
Feature diversity

a b s t r a c t
Lifelong person re-identiﬁcation (LReID) aims to develop a single model that is capable of continuously learning
from new domain (present) while retaining knowledge from previously encountered ones (past) and generalizing
to unseen domains (future). However, distribution shifts across these domains pose a signiﬁcant challenge in
maintaining performance across past, present, and future domains, that is, causing the catastrophic forgetting on
previously seen domains and limited generalization to unseen ones. To address the above issues, we propose to
guide consistent feature extraction to bridge distribution shifts using text prompts designed to remain invariant
across domains. First, identity-consistent text prompts capturing high-level image semantics are extracted and
aligned with image features throughout the lifelong learning pipeline. Moreover, to enhance generalization to
unseen domains, we introduce an adversarial training that text features are contrastively aligned with both
original and future-style image features, the latter generated by applying gradient-based perturbations in the
feature space. Compared with 21 representative models on 11 benchmark datasets, our proposed model, trained
without access to historical data, achieves performance comparable to the model trained using a joint training
approach, and it performs well on all of the past, present, and future domains. We further explored the forgetting
of the ﬁrst historical domain and the generalization to all unseen domains under all 24 orders, and the results
conﬁrmed the superiority of our model. Codes will be released if this paper is accepted.

1. Introduction
Person re-identiﬁcation (ReID) [1,2] aims to match the same person
captured by cameras at diﬀerent times and/or locations. The model has
to perform well on both historical (seen) and future (unseen) domains
while ensuring its person identiﬁcation ability on the present domain.
While image-based ReID methods have made signiﬁcant progress, recent studies reveal that hybridizing images with text can further enhance performance [3]. Text, as a high-level semantic representation of
images [4], exhibits stronger domain consistency compared to images,
as illustrated in Fig. 1(a). The advent of pre-trained vision-language
models like CLIP [5] has enabled hybridized ReID models to achieve
strong performance when trained on a single present domain. These
methods, which focus only on ﬁtting a single present-domain distribution and are referred as Common ReID in this paper, fail to learn dynamic distributions in a continual manner or generalize to unseen distributions. Continual person ReID (CReID) strives to dynamically learn

domain-consistent features and overcome catastrophic forgetting caused
by distribution shifts, using regularization [6], knowledge accumulation [7–9] and so on. CKP [7] is one of the latest model in the ﬁeld
of CReID, which performs better than Common ReID on past domains
as shown in Fig. 1(b). Note that the present domain in CReID in our
work refers to the most recently encountered domain in the training sequence and those before the present domain are past domains. However,
there is a lack of dedicated mechanisms in CReID speciﬁcally aimed at
improving generalization to future unseen domains. Generalized person ReID endeavors to adapt to the distribution of future unseen domains when trained on one or several present domains, using data augmentation, meta learning, adversarial learning and so on. As shown in
Fig. 1(b), the generalized ReID model QAConv [12] performs better than
the CReID CKP on future domains. Nevertheless, generalized ReID fails
to dynamically ﬁt the distribution in a continual fashion, which is how
the practical data stream arrives. In this paper, we propose a person ReID
model that leverages text prompts to guide domain-consistent feature

∗ Corresponding author.
1

E-mail address: fengchaolu@cse.neu.edu.cn (C. Feng).
These authors contributed equally to this work.

https://doi.org/10.1016/j.patcog.2025.112326
Received 10 March 2025; Received in revised form 18 June 2025; Accepted 21 August 2025
Available online 26 August 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 172 (2026) 112326

Z. Liu et al.

learning by aligning them with image features throughout lifelong learning. To further enhance generalization to unseen domains, we apply
style adversarial training, contrastively aligning text with both original
and future-style image features generated via gradient-based perturbations. Our method combines the advantages of common ReID, continual ReID and generalized ReID to achieve performance improvement,
forgetting prevention, and generalization promotion on domains corresponding to present, past, and future. The overall training paradigm is
illustrated in Fig. 1(c). The main contributions are summarized as follows,

and domain-consistent knowledge representation [33,34]. Additionally,
continual compatible representation and transfer mechanisms enhance
feature compatibility across domains [35], while knowledge adapters
dynamically learn and assign domain-speciﬁc knowledge for LReID tasks
[36]. To improve robustness to real-world clothing variations in CReID,
some methods incorporate personalized clothing-state tokens alongside
identity-speciﬁc tokens in CLIP-ReID, addressing challenges posed by
clothing changes in lifelong learning [37]. The aforementioned works
primarily focus on addressing catastrophic forgetting, with relatively
less emphasis on mechanisms speciﬁcally designed to enhance generalization to future domains.

– To the best of our knowledge, we are the ﬁrst to propose the mechanism that smooths distribution diﬀerences with the guidance of text
prompts in the lifelong person re-identiﬁcation across past, present,
and future domains.
– Text prompts and image encoding update alternately and sequentially with the arrival order of datasets to promote each other continually to ensure performance of the model on the past and present.
– Future style generation is introduced to simulate unseen samples in
the image embedding space, adopted as an additional regularization
together with the corresponding text embeddings on training the image encoder and therefore the text prompt generator, to ensure the
generalization on the future.

2.3. Generalized person re-identiﬁcation
The aim of generalized person ReID is to learn a generalized model
using samples from a single domain or multiple domains and then to apply it to samples coming from unseen domains. Depending on whether
the unseen domains can be accessed in advance, generalized person
ReID can be classiﬁed into methods based on domain adaptation and
methods based on domain generalization [2]. Typically, the domain
adaptation-based methods acquire samples of target domain prematurely and transfers the style of samples from training domain to target
domain for improving the performance on target domain [38]. One line
of domain generalization methods focuses on learning domain-invariant
features by ﬁltering domain-speciﬁc information, using techniques such
as instance normalization [25] and MetaBN [24]. Another direction
seeks to reduce overﬁtting to the source domain through methods like
query-adaptive convolution [12] and self-distillation [39]. In addition,
cross-camera generalization has been explored by exploiting unpaired
samples from diﬀerent cameras and introducing identity-guided perturbation prompts to enhance robustness against camera variations [40].
Recent progress in vision-language models, notably CLIP, has also been
utilized to boost generalizability in person ReID [41].

Overall structure of our paper. We begin with the motivation and
key contributions in Section 1, followed by a review of related works in
Section 2. Section 3 details our proposed TP-LReID model, while Section 4 presents the experimental setup, results, and analysis. Further
discussions on implications and limitations are in Section 5, and we conclude in Section 6 with a summary and future directions.
2. Related work
2.1. Person re-identiﬁcation

3. Method

Person ReID is a sub-problem of image retrieval, aiming to identify a
given person in an image from a gallery of images captured by cameras
at diﬀerent times and/or locations. According to diﬀerent perspectives
of methodology, person ReID can be classiﬁed into representative learning based [13] and metric learning based [14], global based [15] and
local based [16]. From the data modality perspective, ReID methods include image-based [17], video-based [18], and multimodal approaches
that incorporate complementary modalities, such as text-image pairs
[3,19,20] or visible-infrared data [21–23]. According to the perspectives
of model adaptability, it can be classiﬁed into continual [7–9] and generalized [12,24,25]. All of the above methods require annotated data
during training, i.e., they belong to supervised learning. Accordingly,
unsupervised learning-based re-identiﬁcation does not require labeled
data and focuses on learning from a large amount of unlabeled data
[26,27].

Problem formulation. As described above, the proposed TP-LReID
aims to learn a model that generalizes across both seen (i.e., past and
present) and unseen (i.e., future) domains. Let 𝐷 = {𝐷1 , 𝐷2 , … , 𝐷𝑍 } represent a sequence of datasets (domains), where training is performed sequentially from 𝐷1 to 𝐷𝑖 . At any stage 𝑖, the training process involves all
data up to and including 𝐷𝑖 , denoted as 𝐷1∶𝑖 . We deﬁne the past domains
as 𝐷past = {𝐷1 , 𝐷2 , … , 𝐷𝑖−1 }, the present domain as 𝐷present = {𝐷𝑖 }, and
the future domains as 𝐷future = {𝐷𝑖+1 , … , 𝐷𝑍 }. Each domain 𝐷𝑡 , for
|𝐷 |
𝑡 = 1, 2, … , 𝑍, consists of a set of image-label pairs 𝐷𝑡 = {(𝑥𝑘 , 𝑦𝑘 )}𝑘=1𝑡 ,
where |𝐷𝑡 | denotes the number of samples in 𝐷𝑡 . During training, two
mini-batch sets are constructed by randomly sampling from 𝐷𝑡 : one for
text prompt generation and one for image encoding, which are denoted
txt } and 𝐷img = {𝐵 img , … , 𝐵 img }. Each mini-batch
by 𝐷text = {𝐵1txt , … , 𝐵𝐻
𝑅
1
img

(𝐵ℎtxt or 𝐵𝑟 ) contains 𝑁 identities with 𝐾 samples per identity. Each
domain 𝐷𝑡 is split into disjoint training and testing subsets, denoted as
𝐷𝑡train and 𝐷𝑡test , respectively, with no class overlap between them.
Overview. The framework of the proposed model TP-LReID is given in
Fig. 2. Generally, the proposed model consists of three main modules,
namely text prompt generator (⋅), image feature encoder (⋅), and future sample simulator (⋅). To extract and preserving domain invariant
knowledge from already seen domains during training, (⋅) and (⋅) are
updated interactively by turn for each domain arrival. In addition, we
adopt the text encoder  existing in CLIP as a frozen module to extract
features from texts generated by the text prompt generator (⋅). The future sample simulator (⋅) is introduced to simulate unseen samples in
the image embedding space to ensure generalization of the proposed
model on future domains. During testing, the image encoder is used
to construct a metric space where inter-identities are far away while
intra-identities are close. Embeddings of the query and gallery images
are extracted by the image encoder and similarly measured by euclidean

2.2. Continual person re-identiﬁcation
Due to data-driven characteristics, performance of person ReID often degrades on historical domains when trained solely on current domain, leading to the issue of catastrophic forgetting. Continual person
re-identiﬁcation (CReID), also known as lifelong person re-identiﬁcation
(LReID), aims to address this issue and can be broadly categorized into
replay-based and exemplar-free methods. Replay-based methods mitigate catastrophic forgetting by storing and replaying samples from
past domains during training [28,29]. Exemplar-free methods rely on
knowledge representation and transfer to retain learned knowledge.
Some methods use graph neural networks to model and accumulate
knowledge in CReID [7–9,30]. Others employ reﬁned knowledge transfer and ﬁne-grained knowledge acquisition to further mitigate forgetting [31,32]. To address domain shifts in CReID, various strategies have been proposed, including patch-based knowledge distillation
2

Pattern Recognition 172 (2026) 112326

Z. Liu et al.

Fig. 1. Depiction of (a) domain distribution consistency, (b) comprehensive robustness across all domains, and (c) overall setting.

3

Pattern Recognition 172 (2026) 112326

Z. Liu et al.

Fig. 2. Framework of the proposed model TP-LReID that is able to applied to recognize persons in images from all of the past, present, and future domains. The
text prompt generator (⋅) and image feature encoder (⋅) are updated interactively by turn for each domain arrival to promote each other continually to ensure
performance of the model on the past and present. Future style generation is introduced to simulate unseen samples in the image embedding space to enhance the
generalization on future domains.
Table 1
Details of the benchmark datasets used in this paper.

Past & Present

Future

Testing Identities

Dataset

Training Identities

Training samples

Query

Gallery

Market1501 (MA)

751(500)

12936(9173)

750

751

CUHK03 (CU)

767(500)

7365(4867)

700

700

CUHKSYSU (SY)

942(500)

4374(2180)

2900

2900

MSMT17 (MS)

1041(500)

32621(14263)

3060

3060

Grid

125(-)

–

125

126

SenseReID

1718(-)

–

521

1718

CUHK01

485(-)

–

486

486

CUHK02

1577(-)

–

239

239

VIPER

316(-)

–

316

316

iLIDS

59(-)

–

60

60

PRID

100(-)

–

100

649

(500) denotes 500 identities are used to train the comparative models in this paper. ‘-’ denotes
the corresponding dataset is not available during training. The training samples denotes the
number of original samples and selected samples engaged in training under our strategy.

distance in the space. The gallery images are sorted based on the similarity as the output of the proposed model. The above algorithmic process
is shown in Algorithm 1.

earlier and shown in Fig. 1, we perform transfer learning only for the
text generator without employing additional complex strategies. That
is, the well-trained parameters of the prompt generator 𝑖−1 (⋅), updated
sequentially by means of image domains 𝐷1 , 𝐷2 , …, 𝐷𝑖−1 , are taken to
initialize 𝑖 (⋅) before the present 𝑖 (⋅) starts to be trained. Thus, image identities of corresponding mini-bathes are considered as input of
the text prompt generator 𝑖 (⋅), which is initialized by 𝑖−1 (⋅). We generate identity-speciﬁc text prompt vectors with a ﬁxed length of four,

3.1. Lifelong text prompt generating
For the present image domain, denoted by 𝐷𝑖 , as the domain distributions in textual embedding space are more similar as mentioned
4

Pattern Recognition 172 (2026) 112326

Z. Liu et al.

ID𝑢 } is the set of all samples in the 𝐵ℎ𝑡𝑥𝑡 sharing the same identity as 𝑢. Notably, our text prompt generator adopts a lifelong paradigm with crossdomain transfer, which oﬀers a complementary perspective to CLIPReID [3].

Algorithm 1 TP-LReID.
Input: 𝐷 = {𝐷1 , …, 𝐷𝑖−1 , 𝐷𝑖 , 𝐷𝑖+1 , …, 𝐷𝑍 }, 𝐷𝑝𝑎𝑠𝑡 = ∅, Initial Text
Prompt 0 , Initial Image Encoder 0 , Text Encoder 
Output: 𝑖 , 𝑖 , {𝑚𝐴𝑃𝑓 , 𝑅𝑎𝑛𝑘1𝑓 }𝑍
𝑓 =1
1: for 𝐷𝑒 in {𝐷1 , …, 𝐷𝑖−1 , 𝐷𝑖 }𝑡𝑟𝑎𝑖𝑛 do
⊳ Begin training
2:
𝐷𝑝𝑟𝑒𝑠𝑒𝑛𝑡 = 𝐷𝑒
3:
if e==0 then
4:
𝑒 ∶ Initialized with the 0
5:
𝑒 ∶ Initialized with the 0
6:
else
7:
𝑒 = 𝑒−1
8:
𝑒 = 𝑒−1
9:
end if
10:
Sample 𝐷𝑡_𝑡𝑥𝑡 from 𝐷𝑝𝑟𝑒𝑠𝑒𝑛𝑡 for training 𝑒
11:
Sample 𝐷𝑡_𝑖𝑚𝑔 from 𝐷𝑝𝑟𝑒𝑠𝑒𝑛𝑡 for training 𝑒
𝑡

12:

𝑗
for each mini-batch 𝐷𝑡_𝑡𝑥𝑡
in 𝐷𝑡_𝑡𝑥𝑡 do

After the text prompt generator 𝑖 (⋅) having been updated, it is used
to assist with updating of the image encoder to generate 𝑖 (⋅) by taking 𝑖−1 (⋅) as initialization. The image encoder 𝑖 (⋅) is used to extract
image embeddings of mini-batches sampling from 𝐷𝑖 . Corresponding
image identities are considered as input of the text prompt generator
𝑖 (⋅). The text prompts generated by 𝑖 (⋅) are then taken as input of the
text encoder  . The outputs are integrated with corresponding image
embeddings using the similarity function to serve as a constraint on updating of the image encoder.
To ensure performance of the image encoder on the present, the well
known cross-entropy loss 𝑐𝑒 and triplet loss 𝑡𝑟𝑖 are main objective
functions of the image encoder 𝑖 that is consider as a whole given as
follows,

⊳ Train 𝑒

𝑡𝑗
Extract text embedding 𝑇𝑦𝑙𝑡 =  (𝑒 (𝐷𝑡_𝑡𝑥𝑡
))
𝑡𝑗
Extract image embedding 𝑉𝑡𝑙𝑡 = 𝑒 (𝐷𝑡_𝑡𝑥𝑡
)
𝑗

13:
14:
15:
16:
17:
18:

3.2. Lifelong image encoding

Compute 𝑖𝑡 as Eq. (1)
Optimize 𝑖𝑡 to update 𝑒
end for
𝑟ℎ
for each mini-batch 𝐷𝑡_𝑖𝑚𝑔
in 𝐷𝑡_𝑖𝑚𝑔 do

⊳ Keep 𝑒 ﬁxed

𝑐𝑡 = 𝑐𝑒 + 𝑡𝑟𝑖

⊳ Train 𝑒

𝑐𝑒 =

𝑟

ℎ
20:
Extract image embedding 𝑉𝑟𝑙𝑡 = 𝑒 (𝐷𝑡_𝑖𝑚𝑔
)
ℎ
21:
Compute 𝑖𝑚 as Eq. (11)
22:
for 𝑚 in {1, 2, …, 𝑀} do
23:
Generate 𝑉𝑚𝑎𝑑𝑣 as Eq. (13)
⊳ Future Style Generation
24:
end for
25:
Compute 𝑎𝑑𝑣 as Eq. (14)
26:
Optimize 𝑖𝑚 + 𝑎𝑑𝑣 to update 𝑒
27:
end for
28:
𝐷𝑝𝑎𝑠𝑡 = 𝐷𝑝𝑎𝑠𝑡 + 𝐷𝑝𝑟𝑒𝑠𝑒𝑛𝑡
29: end for
⊳ End training
30: for 𝐷𝑓 in {𝐷1 , …, 𝐷𝑖 , …, 𝐷𝑍 }𝑡𝑒𝑠𝑡 do
⊳ Begin testing

31:

𝑉𝑓𝑞𝑢𝑒𝑟𝑦 , 𝑉𝑓𝑔𝑎𝑙𝑙𝑒𝑟𝑦 = 𝑖 (𝐷𝑓𝑞𝑢𝑒𝑟𝑦 ), 𝑖 (𝐷𝑓𝑔𝑎𝑙𝑙𝑒𝑟𝑦 )

32:

𝑑𝑖𝑠_𝑚𝑎𝑡𝑟𝑖𝑥𝑓 = 𝐸𝑑𝑐𝑙𝑖𝑑𝑒𝑎𝑛_𝐷𝑖𝑠𝑡𝑎𝑛𝑐𝑒(𝑉𝑓𝑞𝑢𝑒𝑟𝑦 , 𝑉𝑓𝑔𝑎𝑙𝑙𝑒𝑟𝑦 )
𝑞𝑢𝑒𝑟𝑦

33:
𝑚𝐴𝑃𝑓 , 𝑅𝑎𝑛𝑘1𝑓 = eval(𝑑𝑖𝑠_𝑚𝑎𝑡𝑟𝑖𝑥𝑓 ,ID𝑓
34: end for

𝑁
∑

𝑟

ℎ
Extract text embedding 𝑇𝑦𝑙𝑡 =  (𝑒 (𝐷𝑡_𝑖𝑚𝑔
))

19:

(4)

where 𝑐𝑒 and 𝑡𝑟𝑖 are deﬁned as
−𝑞𝑟𝑚 log(𝑝𝑚
𝑟)

(5)

𝑚=1

and
∑
1
(max 𝑑(𝑉𝑟𝑙 , 𝑉𝑟𝑝 )
𝑁 × 𝐾 𝑙=1 𝑦𝑙 =𝑦𝑝
𝑁×𝐾

𝑡𝑟𝑖 =

(6)

− min 𝑑(𝑉𝑟𝑙 , 𝑉𝑟𝑛 ) + 𝛼).
𝑦𝑙 ≠𝑦𝑛

Here, 𝑞𝑟𝑚 is the value from the target distribution and 𝑝𝑚
𝑟 is the predicted
img
logit for class 𝑚 in the image mini-batch 𝐵𝑟 . The triplet (𝑉𝑟𝑙 , 𝑉𝑟𝑝 , 𝑉𝑟𝑛 )
denotes image embeddings, where 𝑉𝑟𝑙 and 𝑉𝑟𝑝 share the same identity
(𝑦𝑙𝑟 = 𝑦𝑝𝑟 ), and 𝑉𝑟𝑛 comes from a diﬀerent identity (𝑦𝑙𝑟 ≠ 𝑦𝑛𝑟 ). The function
𝑑(⋅, ⋅) denotes the distance between two embeddings, and 𝛼 is the margin
used in the triplet loss.
To address catastrophic forgetting the well known distillation loss
deﬁned between 𝑖 (⋅) and 𝑖−1 (⋅) is introduced as a regularization on
the image encoder, namely,

,ID𝑔𝑎𝑙𝑙𝑒𝑟𝑦
)
𝑓
⊳ End testing

𝑑 = −

encoding semantic features unique to each identity. These vectors are
inserted into the template (i.e., “A photo of a [identity-speciﬁc context]
person”). The text prompts are encoded and fed into the text encoder 
to generate text embeddings, which are then aligned with image embeddings via a similarity metric to guide image encoder training. The image
encoder that has been updated sequentially by means of image domains
𝐷1 , 𝐷2 , …, 𝐷𝑖−1 , denoted by 𝑖−1 (⋅), is used to extract image embeddings
of mini-batches sampling from 𝐷𝑖 . The text prompt generator 𝑖 (⋅) is
updated similarly to CLIP-ReID [3] by minimizing the following loss on
the present domain:
∑
∑
it =
i2t (𝑢) +
t2i (𝑢)
(1)

𝑁×𝐾
∑

𝑖
𝑝𝑖−1
𝑙 log(𝑝𝑙 )

(7)

𝑙=1

where 𝑢 indexes each sample in the current mini-batch 𝐵ℎ𝑡𝑥𝑡 , and the two
terms are deﬁned as:

where 𝑝𝑖−1
and 𝑝𝑖𝑙 represent ID prediction logits of previous 𝑖−1 and
𝑙
present 𝑖 for the same input 𝑥𝑙𝑟 .
As mentioned earlier, the domain distributions in textual embedding
space are more similar than the image embedding space. To ensure performance of the image encoder, we hybridize text embeddings and image embeddings during training at each learning step (from the past to
present) by deﬁning a similarity matrix between both of the image and
text to therefore introduce constraints on the training procedure via the
text prompt generator. Due to the small distributional diﬀerence of text
features, the following loss 𝑡𝑖 , which encourages the image encoder to
learn domain-consistent features and is formally similar to CLIP-ReID
[3], is deﬁned as
∑
∑
𝑡𝑖 =
𝑖2𝑡 (𝑤) +
𝑡2𝑖 (𝑤).
(8)

exp(𝑠(𝑉𝑢 , 𝑇𝑢 ))
i2t (𝑢) = − log ∑𝑁
𝑛=1 exp(𝑠(𝑉𝑢 , 𝑇𝑛 ))

where

𝑢

𝑢

∑

𝑤

(2)

exp(𝑠(𝑉𝑏 , 𝑇𝑢 ))

1
t2i (𝑢) = −
log ∑𝑁×𝐾
|(𝑢)| 𝑏∈(𝑢)
exp(𝑠(𝑉 , 𝑇 ))
𝑎=1

𝑎

𝑤

exp(𝑠(𝑉𝑤 , 𝑇𝑤 ))
i2t (𝑤) = − log ∑𝑁
𝑛=1 exp(𝑠(𝑉𝑤 , 𝑇𝑛 ))

(3)

and

𝑢

t2i (𝑤) = −

Here, 𝑉𝑢 and 𝑇𝑢 are the image and text embeddings of sample 𝑢. 𝑠(⋅, ⋅)
denotes the similarity function (e.g., cosine similarity). (𝑢) = {𝑏 ∣ ID𝑏 =

∑
exp(𝑠(𝑉𝑏 , 𝑇𝑤 ))
1
log ∑𝑁×𝐾
.
|(𝑤)| 𝑏∈(𝑤)
exp(𝑠(𝑉 , 𝑇 ))
𝑎=1

5

(9)

𝑎

𝑤

(10)

Pattern Recognition 172 (2026) 112326

Z. Liu et al.

Here, 𝑉𝑤 and 𝑇𝑤 are the image and text embeddings of sample 𝑤 from
𝐵𝑟𝑖𝑚𝑔 . 𝑠(⋅, ⋅) denotes the similarity function. (𝑤) = {𝑏 ∣ ID𝑏 = ID𝑤 } is the
img
set of all samples in the 𝐵𝑟 sharing the same identity as 𝑤.
In summary, the image encoder 𝑖 (⋅) is updated by minimizing the
following loss on the present domain
𝑖𝑚 = 𝑐𝑡 + 𝑑 + 𝑡𝑖 .

[50] and PRID [51], each of which is considered as an image domain.
These datasets are divided into the past (Market1501, CUHKSYSU,
MSMT17), present (CUHK03) and future (Grid, SenseReId, CUHK01,
CUHK02, VIPER, ILIDS, PRID). The past and present domains arrive
in order for continually training, i.e, Market1501 (MA)→ CUHKSYSU
(SY)→ MSMT17 (MS)→ CUHK03 (CU). For each past and present domain, 500 identities from the corresponding training set are randomly
selected as training samples, while all identities in the testing set are
kept ﬁxed. The future domains are not involved in the training and are
only used for testing the generalization performance of all the comparative models adopted in this paper. For each future domain, all identities
are kept for testing. Details of the datasets used in this paper are given
in Table 1. Note that for multi-domain generalization models, which requires all the historical images are accessible, such as ERM-SDViT which
we will further discuss in Section 4.3, we train them by taking the past
and present domains as a training whole.
The performance of the model on datasets is assessed using two evaluation metrics: mean average precision (mAP) and Rank-1. Both Rank1
and mAP take values no more than 100 % with great values indicating
a better performance.

(11)

3.3. Feature space future simulating
In addition to continuously updating the text prompt generator and
the image encoder with the order of image domain appearances, a future style generation module is introduced to construct unseen samples
in the image embedding space. The image samples are considered as
samples from future domains. In this module, the image embeddings
are also combined with corresponding text embeddings considering as
an additional regularization on training the image encoder 𝑖 to therefore ensure the generalization ability of the model on unseen domains.
Speciﬁcally, the Style Adversarial training (StyleAdv [42]) module
is used to construct a candidate through the mean 𝜇 = [𝜇1 , 𝜇2 , …, 𝜇𝑀 ]
and standard deviation 𝜎 = [𝜎1 , 𝜎2 , …, 𝜎𝑀 ] patch-wisely calculated in the
image embedding 𝑉 = [𝑉1 , 𝑉2 , …, 𝑉𝑀 ]. Note that 𝑀 is depth of the image
𝑎𝑑𝑣 , 𝜎 𝑎𝑑𝑣 ) is generated by
encoder. The adversarial style (𝜇𝑚
𝑚
⎧𝜇 𝑎𝑑𝑣 = 𝜇𝑚 + 𝜖 ⋅ 𝑠𝑖𝑔𝑛(∇𝜇 𝑐𝑡 ((𝑉𝑤 , 𝜇𝑚 , 𝜎𝑚 ), 𝑦𝑤 ))
𝑚
⎪ 𝑚
⎨𝜎𝑚𝑎𝑑𝑣 = 𝜎𝑚 + 𝜖 ⋅ 𝑠𝑖𝑔𝑛(∇𝜎𝑚 𝑐𝑡 ((𝑉𝑤 , 𝜇𝑚 , 𝜎𝑚 ), 𝑦𝑤 )) + ⨏
⎪
𝑎𝑑𝑣 𝑎𝑑𝑣
⎩𝜇𝑚+1 , 𝜎𝑚+1 = 𝐿𝑎𝑦𝑒𝑟𝑚 (𝜇𝑚 , 𝜎𝑚 )

4.2. Implementation details
Our image encoder and text encoder are taken from CLIP
(transformer-based version) with pre-trained parameters from
https://huggingface.co/openai/clip-vit-base-patch16. We take the
combination of layer 11, layer 12 and the project layer of the image
encoder as the output, rather than only the output of project layer
in CLIP. The dimension of layer 11, layer 12 and the project layer is
768, 768 and 512, respectively. The dimension of the text embedding
is 512. The size of the word vector, used to describe the person
in image, generated by the prompt generator, is 4. The images are
resized to 256 × 128 and the training batch size is set to 64 where
𝑁 = 16 and 𝐾 = 4. For the present domain, we train the model in
two stages. The ﬁrst stage takes 120 epochs. Adam optimizer with
learning rate 3.5 × 10−4 is used, and the decay rate is set to 1 × 10−4 .
The CosineLRScheduler is used to automatically adjust learning rate
and the minimum is set to 1.6 × 10−5 . The second stage takes 60 epochs.
Adam optimizer with learning rate 5 × 10−6 , and decay rate is set to
1 × 10−4 . The multi-stage warm-up is used to regulate the learning rate.
The initial learning rate is set to 5 × 10−7 and increases to 5 × 10−6 at
10 epoch, then decays to 5 × 10−7 and 5 × 10−8 at 30 and 50 epoch. The
hyper-parameter ⨏ is set to 16∕255. The harmonic ratio 𝜖 is random
sampled from the set of 0.8, 0.08, 0.008. The margin 𝛼 in 𝑡𝑟𝑖 is set to 0.4.
The depth of the image encoder is set to 𝑀 = 12.

(12)

where 𝑚 = 1, 2, …, 𝑀 and 𝜖 is a harmonic ratio of signed gradients and
𝑚, varying from 1 to 12, is the index of blocks in the image encoder 𝑖 .
𝜇𝑚 and 𝜎𝑚 represent the mean and standard deviation at block 𝑚 of the
𝑎𝑑𝑣 and 𝜎 𝑎𝑑𝑣 are the adversarially perturbed versions of
image encoder. 𝜇𝑚
𝑚
𝜇𝑚 and 𝜎𝑚 after applying a perturbation based on the gradient of the loss
function. 𝑠𝑖𝑔𝑛(∇𝜇𝑚 ) and 𝑠𝑖𝑔𝑛(∇𝜎𝑚 ) represent the direction of the gradient
with respect to the mean and standard deviation, respectively, where the
𝑠𝑖𝑔𝑛 function returns the sign of the gradient. 𝐿𝑎𝑦𝑒𝑟𝑚 denotes the 𝑚-th
layer of the image encoder 𝑖 . ⨏ is a hyper-parameter that introduces
randomness. A probability, denoted as 𝜌, is assigned to each layer of
adversarial perturbations during training. And
𝑎𝑑𝑣 𝑎𝑑𝑣 𝑎𝑑𝑣
𝑉𝑚𝑎𝑑𝑣 = (𝑉𝑚−1
, 𝜇𝑚 , 𝜎𝑚 ) = 𝜎𝑚𝑎𝑑𝑣

𝑎𝑑𝑣 − 𝜇(𝑉 𝑎𝑑𝑣 )
𝑉𝑚−1
𝑚−1
𝑎𝑑𝑣 )
𝜎(𝑉𝑚−1

𝑎𝑑𝑣
+ 𝜇𝑚

(13)

where 𝑚 = 1, 2, …, 𝑀. 𝑉0𝑎𝑑𝑣 = 𝑉1 , which denotes the perturbation starts
with the input 𝑉1 . The aﬃne transformation function  applied to the
adversarially perturbed input from the previous layer. 𝑉𝑚𝑎𝑑𝑣 is the perturbed output after 𝐿𝑎𝑦𝑒𝑟𝑚 . Until now, we have generated adversarial styles (features) that diﬀer from the distribution of present domain,
which are regarded as being from possible future domains. Although the
𝑎𝑑𝑣 and original 𝑉 𝑙𝑖 share the same identities, but they are
adversarial 𝑉𝑀
𝑟ℎ
quite diﬀerent. To make adversarial style generation more eﬃcient, we
minimize the following loss function as
𝑎𝑑𝑣
𝑎𝑑𝑣 = 𝑎𝑑𝑣
𝑡𝑖 + 𝑐𝑡

4.3. Performance analysis
We compare with Lifelong ReID methods: CRL [6], AKA [9], CKP
[7], MRN [8], GCReID [30], PTKP [28], LSTKC [32], DKP [31], KRKC
[29], and 𝐶 2 𝑅 [35]; Common ReID methods: FT-RN50, BoT [1], FTViT16, TransReID [52], CLIP-ReID [3]; and Generalized ReID methods:
QAConv [12], CLIP-FGDI [41], M3L [24], DGPS [40], and ERM-SDViT
[39]. Following Section 4.1, we train on four datasets in order: Market1501 (MA) → CUHKSYSU (SY) → MSMT17 (MS) → CUHK03 (CU),
where MA, SY, MS are past and CU is the present domain. FT-RN50 and
FT-ViT are sequentially ﬁne-tuned using 𝑐𝑡 Eq. (4). BoT, TransReID,
and CLIP-ReID are trained on CUHK03 only. We also introduce CRLViT (CRL with ViT backbone), and replace the backbones of LSTKC and
KRKC with CLIP, denoted as LSTKC-CLIP* and KRKC-CLIP*.
Note that among Generalized ReID models, QAConv is single-domain
and trained only on the present domain (CU), while M3L and ERMSDViT are multi-domain and trained jointly on all four datasets. This
joint setup deviates from the LReID setting and aligns more with joint
training, which we further discuss in Section 5.

(14)

where 𝑎𝑑𝑣
is computed as the same way of Eq. (8) with the diﬀer𝑡𝑖
𝑎𝑑𝑣 . Similarly, 𝑎𝑑𝑣
ence that 𝑉𝑟𝑙𝑖 is replaced by adversarial features 𝑉𝑀
𝑐𝑡
ℎ
in Eq. (14) is computed as the same way of Eq. (4), with the diﬀerence
𝑎𝑑𝑣 . Eq. (14) and Eq. (11)
that 𝑉𝑟𝑙𝑖 is replaced by adversarial features 𝑉𝑀
ℎ
are optimized together to train the image encoder 𝑖 .
4. Experiments
4.1. Datasets and evaluation metrics
We conduct experiments on 11 person ReID benchmarks, namely
Market1501 [10], CUHKSYSU [43], MSMT17 [11], CUHK03 [44], Grid
[45], SenseReID [46], CUHK01 [47], CUHK02 [48], VIPER [49], iLIDS
6

Pattern Recognition 172 (2026) 112326

Z. Liu et al.

Table 2
Performance of the proposed model compared with representative models on the past and present (seen) domains (Bold indicates the best results, underline the
second best. ERM-SDViT requires all historical images, making it more joint-training-like, while our model only accesses the present domain and outperforms
ERM-SDViT. Past domains include Market1501, CUHKSYSU, and MSMT17, while the present domain is CUHK03 from 𝑂𝑟𝑑𝑒𝑟_1).
Past

Common ReID

Lifelong ReID

Generalized ReID

Ours

Present

Market1501

CUHKSYSU

MSMT17

CUHK03

Avg

Year

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

BoT [1]
FT-RN50 [53]
transReID [52]
CLIP-ReID [3]
FT-ViT16 [53]

2019
2020
2020
2023
2020

20.70
25.64
43.40
45.20
53.50

42.30
49.14
66.90
69.00
75.70

21.80
63.26
78.20
84.30
77.70

22.50
66.10
80.50
86.20
80.00

2.40
14.11
11.80
14.80
20.80

7.90
66.57
32.10
36.50
46.20

58.80
52.59
74.90
76.30
𝟕𝟔.𝟗𝟎

60.60
54.94
76.80
77.80
𝟕𝟗.𝟐𝟎

25.93
38.90
52.08
55.15
57.23

33.33
59.19
64.08
67.38
70.28

AKA [9]
CRL [6]
CKP [7]
GCReID [30]
𝐶 2 𝑅 [35]
MRN [8]
KRKC [29]
DKP [31]
LSTKC [32]
KRKC-CLIP* [29]
LSTKC-CLIP* [32]
PKTP [28]
CRL-ViT [6]

2021
2021
2023
2024
2024
2022
2023
2024
2024
2024
2024
2022
2021

53.63
53.56
54.88
61.47
72.71
57.60
55.70
60.60
59.50
57.00
56.90
78.60
61.20

74.08
73.43
75.68
80.82
88.84
80.60
75.40
80.00
79.50
78.60
78.40
90.90
79.70

71.00
73.14
73.55
75.56
76.56
77.50
76.60
79.10
82.10
75.80
78.50
87.20
84.60

73.86
76.72
76.76
78.59
79.17
79.80
79.60
81.00
84.00
78.50
80.30
88.50
86.20

15.59
16.67
16.56
19.45
6.15
16.50
9.50
16.00
21.60
9.10
23.60
22.20
29.50

69.38
71.62
71.24
75.43
16.02
39.90
24.00
34.50
45.60
24.50
49.00
44.50
55.60

26.05
26.86
29.32
32.73
34.32
42.90
58.60
48.50
42.00
75.90
59.80
53.50
79.30

28.16
29.16
31.66
35.71
35.57
43.70
60.10
51.10
42.90
77.60
62.80
55.00
80.40

41.57
42.56
43.58
47.30
47.43
48.63
50.10
51.05
51.30
54.45
54.70
60.38
63.65

61.37
62.73
63.84
67.64
54.90
61.00
59.78
61.65
63.00
64.80
67.63
69.73
75.53

QAConv [12]
CLIP-FGDI [41]
M3L [24]
DGPS [40]
ERM-SDViT [39]

2020
2025
2021
2025
2022

27.10
53.50
67.00
55.00
𝟖𝟑.𝟏𝟎

58.90
77.40
84.50
73.90
𝟗𝟏.𝟓𝟎

48.40
78.10
77.70
86.60
𝟗𝟒.𝟐𝟎

52.00
79.70
79.20
88.20
𝟗𝟒.𝟕𝟎

10.50
18.60
21.00
21.90
𝟔𝟏.𝟐𝟎

36.10
47.00
41.60
47.70
𝟖𝟎.𝟎𝟎

53.10
71.50
63.60
79.50
71.80

55.60
74.90
71.00
81.60
73.50

34.77
55.43
57.33
60.75
𝟕𝟕.𝟓𝟖

50.65
69.75
69.08
72.85
𝟖𝟒.𝟗𝟑

TP-LReID

2025

72.20

86.90

91.20

92.50

41.10

67.10

76.00

77.60

70.13

81.03

Performance on past and present domains. As shown in
Table 2, Lifelong ReID outperforms Common ReID on past domains but lags on the present, highlighting their respective strengths.
Generalized ReID, aiming for cross-domain generalization, also underperforms on the present due to its broader focus. ERM-SDViT achieves
the best overall performance by training on all domains jointly. In
contrast, our TP-LReID achieves comparable results without accessing
past data, operating under a continual, single-domain setting. Excluding
Generalized ReID, TP-LReID signiﬁcantly outperforms all Lifelong and
Common ReID methods.
Performance on future domains. As shown in Table 3, TP-LReID
achieves best average performance in terms of mAP and Rank1 compared with other competitive models of Lifelong, Common, Generalized
ReID on future domains, which exceeded the second best generalized
ReID by 1.67 % and 1.82 % according to average mAP and Rank1. Note
that for MRN in Table 3, we directly use the results from the corresponding paper since the dataset settings are the same. In Table 3, ‘-’ indicates
that results are not provided in the corresponding paper.

Additionally, we demonstrate the eﬀectiveness of regularization
(𝑅𝑒𝑔) and prompts (𝑃 𝑡) on struggling against catastrophic forgetting for
ﬁne-tuning (𝐹 𝑇 ) as shown in Table 6. As domain diﬀerences of textual
prompts are more minor than image embeddings, 𝐹 𝑇 + 𝑃 𝑡 demonstrates
the eﬀectiveness of textual prompts, compared to 𝐹 𝑇 in Table 6, in mitigating catastrophic forgetting caused by large diﬀerences in images.
To investigate the eﬀects of non-continual and continual text
prompting, we perform experiments based on ﬁne-tuning (FT) setting.
As shown in Table 7, 𝐹 𝑇 + 𝑛𝑃 𝑇 denotes that the text prompting is noncontinual, i.e., it is randomly initialized at each step 𝑖 of the continuous
ﬁne-tuning. 𝐹 𝑇 + 𝑃 𝑇 denotes that the text prompting is continual, i.e.,
the text prompting at step 𝑖 is initialized by the text prompting at step
𝑖 − 1 (1 < 𝑖 ≤ 4). The results of the experiments in Table 7 indicate that
the continual text prompting (𝐹 𝑇 + 𝑃 𝑇 ) perform better, so all other experiments in this paper that include text prompting are set up accordingly.
For future style generation, we explore the eﬀect of adding perturbations to each layer in transformer with diﬀerent probabilities 𝜌. As
shown in Table 8, We add the perturbation with 10 %, 60 % and 100 %
probability, and the experimental results show that adding the perturbation with 100 % probability works optimally.

4.4. Ablation experiments
In this section, we ﬁrst evaluate the eﬀectiveness of text prompts (denoted by 𝑃 𝑇 in Table 4) and adversarial future simulating (denoted by
𝐴𝐹 in Table 4) based on person ReID with regularization using 𝑑 loss
give in Eq. (7), (denoted by Baseline in Table 4) on the past, present and
future domains in terms of mAP and Rank1, where 𝐵𝑎𝑠𝑒𝑙𝑖𝑛𝑒 + 𝑃 𝑇 + 𝐴𝐹
denotes the proposed model TP-LReID. It can be seen that the proposed
modules are eﬀective to improve performance of the proposed model.
To further verify the how much boost to past and future data of
the pure AF module, we conduct dedicated experiments in more cases.
As shown in Table 5, we report the comparison results of baseline and
baseline+AF under more orders, the experimental results show that the
boost to the future is better than past’s after the addition of AF in most
cases. It proves the necessity of the AF module for simulating future
data.

5. Discussion
The impact of arrival order. We explore how well the proposed
model TP-LReID preserves performance in all 24 training orders. The
performance on the ﬁrst seen domain and the average performance on
unseen domains are shown in Figs. 3 and 4 respectively. As observed,
our model demonstrates less forgetting on the ﬁrst training domain and
better generalization to future domains compared to other Lifelong ReID
methods. Note that for the third column of Fig. 4, the generalization of
our method after training on the ﬁrst step (SY) is sub-optimal. According to our analysis, this is due to the SY domain having fewer engaged
samples compared to other domains, as indicated in Table 1. Neverthe7

Pattern Recognition 172 (2026) 112326

Z. Liu et al.

Table 3
Generalization performance of the proposed model compared with representative models on future (unseen) domains (Bolding represents the best results, underlining
represents the second best results. ‘-’ denotes results and codes are not provided by the authors of [8]. ERM-SDViT requires all historical images are accessible, it is
more joint-training-like. The proposed method achieves comparable performance by simply accessing the current domain).
Future
Grid

SenseReID

CUHK01

CUHK02

VIPeR

iLIDS

PRID

Avg

Year mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑)
FT-RN50
[53]
BoT [1]
transReID
[52]
FTViT16[53]
CLIP-ReID
[3]

2020 12.21

8.00

28.87

23.36

60.99

60.19

77.86

74.27

37.61

28.16

71.44

60.00

20.96

15.00

43.93

37.95

2019 16.20
2021 41.30

11.20
28.80

25.10
43.00

20.40
35.20

71.50
74.70

70.80
74.20

86.50
89.10

86.60
88.30

34.60
66.60

25.90
57.00

73.80
66.30

66.70
53.30

20.40
66.50

16.00
57.00

46.87
63.93

42.51
56.26

2020 44.70

32.00

49.30

40.30

77.10

76.20

91.30

90.20

61.50

50.60

70.10

60.00

58.10

46.00

64.59

56.47

2023 47.70

38.40

47.60

38.80

82.60

83.00

90.20

89.50

68.80

59.80

73.50

61.70

69.90

61.00

68.61

61.74

CRL [6]
AKA [9]
CKP[7]
𝐶 2 𝑅[35]
GCReID
[30]
KRKCCLIP*[29]
KRKC[29]
DKP[31]
LSTKC[32]
LSTKCCLIP*[32]
PTKP [28]
MRN[8]
CRL-ViT[6]

2021 13.20
2021 16.19
2023 16.98
2024 22.48
2024 31.35

7.20
11.20
9.60
15.20
22.40

33.32
34.14
33.10
37.15
39.21

27.22
27.51
26.64
29.63
32.05

62.06
60.91
64.10
66.27
64.26

62.14
59.36
63.99
65.23
63.07

68.93
61.31
69.22
69.96
68.23

65.90
57.74
65.27
68.20
64.85

37.61
40.68
38.32
41.94
45.65

28.16
31.96
29.43
32.91
37.03

71.44
71.87
70.49
60.73
63.92

60.00
60.00
56.67
50.00
53.33

20.96
23.23
25.70
27.59
29.71

15.00
15.00
19.00
20.00
21.00

43.93
44.05
45.42
46.59
48.90

37.95
37.54
38.66
40.17
41.96

2023 33.50

22.40

27.00

21.70

59.40

60.30

86.10

84.30

49.90

39.90

57.20

45.00

38.80

30.00

50.27

43.37

2023 20.70
2024 32.20
2024 33.80
2024 35.70

12.80
24.80
24.80
27.20

38.90
48.50
50.60
50.70

31.30
40.10
41.70
41.20

81.20
70.90
70.80
73.40

82.30
69.80
70.40
72.40

88.70
76.70
74.00
82.60

87.90
74.10
72.40
81.80

47.90
51.80
54.40
58.50

38.90
41.80
44.90
49.40

64.50
64.20
68.10
66.10

56.70
53.30
58.30
55.00

18.10
38.70
42.50
45.40

9.00
28.00
30.00
34.00

51.43
54.71
56.31
58.91

45.56
47.41
48.93
51.57

2020 31.60
2022 –
2021 45.40

23.20
–
33.60

53.00
–
55.40

45.50
–
45.80

81.80
–
83.40

82.70
–
82.80

87.90
–
91.10

87.90
–
89.50

56.00
–
67.80

46.20
–
59.80

72.40
–
75.20

63.30
–
66.70

31.70
–
63.40

22.00
–
53.00

59.20
60.60
68.81

52.97
56.80
61.60

QAConv
Generalized ReID [12]
M3L [24]
CLIPFGDI[41]
DGPS[40]
ERM-SDViT
[39]

2020 25.90

20.80

30.20

23.30

62.90

64.00

65.10

62.70

41.00

32.50

60.40

48.30

31.40

23.30

43.40

41.10

2021 37.50
2025 42.90

27.20
30.40

46.60
47.00

37.40
37.50

66.00
78.10

64.40
77.20

68.70
89.50

66.50
89.30

55.10
66.90

44.60
57.00

72.50
80.00

65.00
73.30

54.50
68.10

41.00
59.00

57.27
67.50

49.44
60.53

2025 51.60
2022 𝟓𝟕.𝟕𝟎

44.00
𝟒𝟔.𝟒𝟎

51.10
58.10

41.80
49.10

85.30
87.00

86.00
𝟖𝟔.𝟗𝟎

92.40
87.80

91.80
87.00

69.20
𝟕𝟕.𝟕𝟎

59.80
𝟔𝟗.𝟎𝟎

72.90
80.00

63.30
𝟕𝟏.𝟕𝟎

73.30
66.80

64.00
58.00

70.83
73.59

64.39
66.87

Ours

2025 55.00

46.40

𝟔𝟒.𝟎𝟎

𝟓𝟓.𝟏𝟎

𝟖𝟕.𝟑𝟎

𝟖𝟔.𝟗𝟎

𝟗𝟏.𝟔𝟎

𝟗𝟏.𝟐𝟎

73.20

63.90

𝟖𝟏.𝟐𝟎

𝟕𝟏.𝟕𝟎

𝟕𝟒.𝟗𝟎

𝟔𝟓.𝟎𝟎

𝟕𝟓.𝟑𝟏

𝟔𝟖.𝟔𝟎

Common ReID

Lifelong ReID

TP-LReID

Table 4
Eﬀectiveness of text prompting (PT) and adversarial future simulating (AF) on the past, present and future
domains.
Past

Baseline
Baseline+PT
Baseline+AF
Baseline+PT+AF

Present

Future

Avg

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

62.30
66.43
65.40
𝟔𝟖.𝟏𝟕

77.10
80.63
80.60
𝟖𝟐.𝟏𝟕

75.00
𝟕𝟔.𝟑𝟎
74.90
75.30

76.40
𝟕𝟖.𝟒𝟎
76.40
76.00

69.90
72.35
71.05
𝟕𝟕.𝟔𝟎

66.39
68.40
67.11
𝟔𝟖.𝟔𝟎

68.65
71.69
70.45
𝟕𝟑.𝟏𝟔

71.75
75.81
74.70
𝟕𝟔.𝟏𝟐

The Past domains include three seen domains (Market1501, CUHKSYSU, and MSMT17) and their means are
reported.The Present domain contains only CUHK03.The Future domain includes seven unseen domains (Grid,
SenseReID, CUHK01, CUHK02, VIPER, iLIDS, and PRID) and their means are reported.

less, our method overcomes this limitation and achieves optimality in
the later continual steps.
We further conduct experiments that treat MSMT17 as one of unseen
domains to investigate the impact of it. Under this setup, there are only
three training domains, i.e., Market1501 (MA), CUHKSYSU (SY), and
CUHK03 (CU) sequentially. As shown in Fig. 5, the left side reports the
forgetting eﬀect on the ﬁrst training domain (MA), while the right side
illustrates the generalization performance on one of the unseen domains
(MS).
The impact of pre-training parameters. Transformer-based methods (e.g., CLIP-ReID [3]) are pre-trained on large-scale datasets, while
CNN-based methods [7–9,34] typically use ImageNet. To ensure fairness, we pre-train the CNN backbone (ResNet50) on both ImageNet and

the large-scale dataset used by CLIP-ReID. As shown in Tables 9 and 10,
FT-RN50 (IMG) and FT-RN50 (CLIP-ReID) represent these two settings.
Results show minimal performance gain from large-scale pre-training,
so we use the original ImageNet-pretrained weights for all CNN-based
baselines.
Table 11 shows the detailed results for each training step under
the above settings according to 𝑂𝑟𝑑𝑒𝑟_1. Note that there is no past
for Step1. Speciﬁcally, the Present of Step1 denotes the performance
on itself (Market1501). For Step2, the Past denotes the performance
on Market1501, and the Present denotes the performance on itself
(CUHKSYSU). For Step3, the Past denotes the average performance on
Market1501 and CUHKSYSU, and the Present denotes the performance
on itself (MSMT17). Likewise, the performance of Step4 is calculated
8


codex
这两篇的“问题包装”比模块本身更有价值：`TP-LReID` 把终身 ReID 从“只防遗忘”扩大成过去、现在、未来三类域同时负责；`TSKD` 则把跨模态对齐改写成先易后难的对齐顺序问题。我再补抓两处被长输出截掉的方法细节，然后读后两篇。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'TP-LReID - Lifelong person re-identification using text prompts.pdf' - | awk '/3\\.2\\. Lifelong image encoding/{flag=1} /3\\.3\\. Future style generation/{if(flag){print; flag=0}} flag'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3.2. Lifelong image encoding

Compute 𝑖𝑡 as Eq. (1)
Optimize 𝑖𝑡 to update 𝑒
end for
𝑟ℎ
for each mini-batch 𝐷𝑡_𝑖𝑚𝑔
in 𝐷𝑡_𝑖𝑚𝑔 do

⊳ Keep 𝑒 ﬁxed

𝑐𝑡 = 𝑐𝑒 + 𝑡𝑟𝑖

⊳ Train 𝑒

𝑐𝑒 =

𝑟

ℎ
20:
Extract image embedding 𝑉𝑟𝑙𝑡 = 𝑒 (𝐷𝑡_𝑖𝑚𝑔
)
ℎ
21:
Compute 𝑖𝑚 as Eq. (11)
22:
for 𝑚 in {1, 2, …, 𝑀} do
23:
Generate 𝑉𝑚𝑎𝑑𝑣 as Eq. (13)
⊳ Future Style Generation
24:
end for
25:
Compute 𝑎𝑑𝑣 as Eq. (14)
26:
Optimize 𝑖𝑚 + 𝑎𝑑𝑣 to update 𝑒
27:
end for
28:
𝐷𝑝𝑎𝑠𝑡 = 𝐷𝑝𝑎𝑠𝑡 + 𝐷𝑝𝑟𝑒𝑠𝑒𝑛𝑡
29: end for
⊳ End training
30: for 𝐷𝑓 in {𝐷1 , …, 𝐷𝑖 , …, 𝐷𝑍 }𝑡𝑒𝑠𝑡 do
⊳ Begin testing

31:

𝑉𝑓𝑞𝑢𝑒𝑟𝑦 , 𝑉𝑓𝑔𝑎𝑙𝑙𝑒𝑟𝑦 = 𝑖 (𝐷𝑓𝑞𝑢𝑒𝑟𝑦 ), 𝑖 (𝐷𝑓𝑔𝑎𝑙𝑙𝑒𝑟𝑦 )

32:

𝑑𝑖𝑠_𝑚𝑎𝑡𝑟𝑖𝑥𝑓 = 𝐸𝑑𝑐𝑙𝑖𝑑𝑒𝑎𝑛_𝐷𝑖𝑠𝑡𝑎𝑛𝑐𝑒(𝑉𝑓𝑞𝑢𝑒𝑟𝑦 , 𝑉𝑓𝑔𝑎𝑙𝑙𝑒𝑟𝑦 )
𝑞𝑢𝑒𝑟𝑦

33:
𝑚𝐴𝑃𝑓 , 𝑅𝑎𝑛𝑘1𝑓 = eval(𝑑𝑖𝑠_𝑚𝑎𝑡𝑟𝑖𝑥𝑓 ,ID𝑓
34: end for

𝑁
∑

𝑟

ℎ
Extract text embedding 𝑇𝑦𝑙𝑡 =  (𝑒 (𝐷𝑡_𝑖𝑚𝑔
))

19:

(4)

where 𝑐𝑒 and 𝑡𝑟𝑖 are deﬁned as
−𝑞𝑟𝑚 log(𝑝𝑚
𝑟)

(5)

𝑚=1

and
∑
1
(max 𝑑(𝑉𝑟𝑙 , 𝑉𝑟𝑝 )
𝑁 × 𝐾 𝑙=1 𝑦𝑙 =𝑦𝑝
𝑁×𝐾

𝑡𝑟𝑖 =

(6)

− min 𝑑(𝑉𝑟𝑙 , 𝑉𝑟𝑛 ) + 𝛼).
𝑦𝑙 ≠𝑦𝑛

Here, 𝑞𝑟𝑚 is the value from the target distribution and 𝑝𝑚
𝑟 is the predicted
img
logit for class 𝑚 in the image mini-batch 𝐵𝑟 . The triplet (𝑉𝑟𝑙 , 𝑉𝑟𝑝 , 𝑉𝑟𝑛 )
denotes image embeddings, where 𝑉𝑟𝑙 and 𝑉𝑟𝑝 share the same identity
(𝑦𝑙𝑟 = 𝑦𝑝𝑟 ), and 𝑉𝑟𝑛 comes from a diﬀerent identity (𝑦𝑙𝑟 ≠ 𝑦𝑛𝑟 ). The function
𝑑(⋅, ⋅) denotes the distance between two embeddings, and 𝛼 is the margin
used in the triplet loss.
To address catastrophic forgetting the well known distillation loss
deﬁned between 𝑖 (⋅) and 𝑖−1 (⋅) is introduced as a regularization on
the image encoder, namely,

,ID𝑔𝑎𝑙𝑙𝑒𝑟𝑦
)
𝑓
⊳ End testing

𝑑 = −

encoding semantic features unique to each identity. These vectors are
inserted into the template (i.e., “A photo of a [identity-speciﬁc context]
person”). The text prompts are encoded and fed into the text encoder 
to generate text embeddings, which are then aligned with image embeddings via a similarity metric to guide image encoder training. The image
encoder that has been updated sequentially by means of image domains
𝐷1 , 𝐷2 , …, 𝐷𝑖−1 , denoted by 𝑖−1 (⋅), is used to extract image embeddings
of mini-batches sampling from 𝐷𝑖 . The text prompt generator 𝑖 (⋅) is
updated similarly to CLIP-ReID [3] by minimizing the following loss on
the present domain:
∑
∑
it =
i2t (𝑢) +
t2i (𝑢)
(1)

𝑁×𝐾
∑

𝑖
𝑝𝑖−1
𝑙 log(𝑝𝑙 )

(7)

𝑙=1

where 𝑢 indexes each sample in the current mini-batch 𝐵ℎ𝑡𝑥𝑡 , and the two
terms are deﬁned as:

where 𝑝𝑖−1
and 𝑝𝑖𝑙 represent ID prediction logits of previous 𝑖−1 and
𝑙
present 𝑖 for the same input 𝑥𝑙𝑟 .
As mentioned earlier, the domain distributions in textual embedding
space are more similar than the image embedding space. To ensure performance of the image encoder, we hybridize text embeddings and image embeddings during training at each learning step (from the past to
present) by deﬁning a similarity matrix between both of the image and
text to therefore introduce constraints on the training procedure via the
text prompt generator. Due to the small distributional diﬀerence of text
features, the following loss 𝑡𝑖 , which encourages the image encoder to
learn domain-consistent features and is formally similar to CLIP-ReID
[3], is deﬁned as
∑
∑
𝑡𝑖 =
𝑖2𝑡 (𝑤) +
𝑡2𝑖 (𝑤).
(8)

exp(𝑠(𝑉𝑢 , 𝑇𝑢 ))
i2t (𝑢) = − log ∑𝑁
𝑛=1 exp(𝑠(𝑉𝑢 , 𝑇𝑛 ))

where

𝑢

𝑢

∑

𝑤

(2)

exp(𝑠(𝑉𝑏 , 𝑇𝑢 ))

1
t2i (𝑢) = −
log ∑𝑁×𝐾
|(𝑢)| 𝑏∈(𝑢)
exp(𝑠(𝑉 , 𝑇 ))
𝑎=1

𝑎

𝑤

exp(𝑠(𝑉𝑤 , 𝑇𝑤 ))
i2t (𝑤) = − log ∑𝑁
𝑛=1 exp(𝑠(𝑉𝑤 , 𝑇𝑛 ))

(3)

and

𝑢

t2i (𝑤) = −

Here, 𝑉𝑢 and 𝑇𝑢 are the image and text embeddings of sample 𝑢. 𝑠(⋅, ⋅)
denotes the similarity function (e.g., cosine similarity). (𝑢) = {𝑏 ∣ ID𝑏 =

∑
exp(𝑠(𝑉𝑏 , 𝑇𝑤 ))
1
log ∑𝑁×𝐾
.
|(𝑤)| 𝑏∈(𝑤)
exp(𝑠(𝑉 , 𝑇 ))
𝑎=1

5

(9)

𝑎

𝑤

(10)

Pattern Recognition 172 (2026) 112326

Z. Liu et al.

Here, 𝑉𝑤 and 𝑇𝑤 are the image and text embeddings of sample 𝑤 from
𝐵𝑟𝑖𝑚𝑔 . 𝑠(⋅, ⋅) denotes the similarity function. (𝑤) = {𝑏 ∣ ID𝑏 = ID𝑤 } is the
img
set of all samples in the 𝐵𝑟 sharing the same identity as 𝑤.
In summary, the image encoder 𝑖 (⋅) is updated by minimizing the
following loss on the present domain
𝑖𝑚 = 𝑐𝑡 + 𝑑 + 𝑡𝑖 .

[50] and PRID [51], each of which is considered as an image domain.
These datasets are divided into the past (Market1501, CUHKSYSU,
MSMT17), present (CUHK03) and future (Grid, SenseReId, CUHK01,
CUHK02, VIPER, ILIDS, PRID). The past and present domains arrive
in order for continually training, i.e, Market1501 (MA)→ CUHKSYSU
(SY)→ MSMT17 (MS)→ CUHK03 (CU). For each past and present domain, 500 identities from the corresponding training set are randomly
selected as training samples, while all identities in the testing set are
kept ﬁxed. The future domains are not involved in the training and are
only used for testing the generalization performance of all the comparative models adopted in this paper. For each future domain, all identities
are kept for testing. Details of the datasets used in this paper are given
in Table 1. Note that for multi-domain generalization models, which requires all the historical images are accessible, such as ERM-SDViT which
we will further discuss in Section 4.3, we train them by taking the past
and present domains as a training whole.
The performance of the model on datasets is assessed using two evaluation metrics: mean average precision (mAP) and Rank-1. Both Rank1
and mAP take values no more than 100 % with great values indicating
a better performance.

(11)

3.3. Feature space future simulating
In addition to continuously updating the text prompt generator and
the image encoder with the order of image domain appearances, a future style generation module is introduced to construct unseen samples
in the image embedding space. The image samples are considered as
samples from future domains. In this module, the image embeddings
are also combined with corresponding text embeddings considering as
an additional regularization on training the image encoder 𝑖 to therefore ensure the generalization ability of the model on unseen domains.
Speciﬁcally, the Style Adversarial training (StyleAdv [42]) module
is used to construct a candidate through the mean 𝜇 = [𝜇1 , 𝜇2 , …, 𝜇𝑀 ]
and standard deviation 𝜎 = [𝜎1 , 𝜎2 , …, 𝜎𝑀 ] patch-wisely calculated in the
image embedding 𝑉 = [𝑉1 , 𝑉2 , …, 𝑉𝑀 ]. Note that 𝑀 is depth of the image
𝑎𝑑𝑣 , 𝜎 𝑎𝑑𝑣 ) is generated by
encoder. The adversarial style (𝜇𝑚
𝑚
⎧𝜇 𝑎𝑑𝑣 = 𝜇𝑚 + 𝜖 ⋅ 𝑠𝑖𝑔𝑛(∇𝜇 𝑐𝑡 ((𝑉𝑤 , 𝜇𝑚 , 𝜎𝑚 ), 𝑦𝑤 ))
𝑚
⎪ 𝑚
⎨𝜎𝑚𝑎𝑑𝑣 = 𝜎𝑚 + 𝜖 ⋅ 𝑠𝑖𝑔𝑛(∇𝜎𝑚 𝑐𝑡 ((𝑉𝑤 , 𝜇𝑚 , 𝜎𝑚 ), 𝑦𝑤 )) + ⨏
⎪
𝑎𝑑𝑣 𝑎𝑑𝑣
⎩𝜇𝑚+1 , 𝜎𝑚+1 = 𝐿𝑎𝑦𝑒𝑟𝑚 (𝜇𝑚 , 𝜎𝑚 )

4.2. Implementation details
Our image encoder and text encoder are taken from CLIP
(transformer-based version) with pre-trained parameters from
https://huggingface.co/openai/clip-vit-base-patch16. We take the
combination of layer 11, layer 12 and the project layer of the image
encoder as the output, rather than only the output of project layer
in CLIP. The dimension of layer 11, layer 12 and the project layer is
768, 768 and 512, respectively. The dimension of the text embedding
is 512. The size of the word vector, used to describe the person
in image, generated by the prompt generator, is 4. The images are
resized to 256 × 128 and the training batch size is set to 64 where
𝑁 = 16 and 𝐾 = 4. For the present domain, we train the model in
two stages. The ﬁrst stage takes 120 epochs. Adam optimizer with
learning rate 3.5 × 10−4 is used, and the decay rate is set to 1 × 10−4 .
The CosineLRScheduler is used to automatically adjust learning rate
and the minimum is set to 1.6 × 10−5 . The second stage takes 60 epochs.
Adam optimizer with learning rate 5 × 10−6 , and decay rate is set to
1 × 10−4 . The multi-stage warm-up is used to regulate the learning rate.
The initial learning rate is set to 5 × 10−7 and increases to 5 × 10−6 at
10 epoch, then decays to 5 × 10−7 and 5 × 10−8 at 30 and 50 epoch. The
hyper-parameter ⨏ is set to 16∕255. The harmonic ratio 𝜖 is random
sampled from the set of 0.8, 0.08, 0.008. The margin 𝛼 in 𝑡𝑟𝑖 is set to 0.4.
The depth of the image encoder is set to 𝑀 = 12.

(12)

where 𝑚 = 1, 2, …, 𝑀 and 𝜖 is a harmonic ratio of signed gradients and
𝑚, varying from 1 to 12, is the index of blocks in the image encoder 𝑖 .
𝜇𝑚 and 𝜎𝑚 represent the mean and standard deviation at block 𝑚 of the
𝑎𝑑𝑣 and 𝜎 𝑎𝑑𝑣 are the adversarially perturbed versions of
image encoder. 𝜇𝑚
𝑚
𝜇𝑚 and 𝜎𝑚 after applying a perturbation based on the gradient of the loss
function. 𝑠𝑖𝑔𝑛(∇𝜇𝑚 ) and 𝑠𝑖𝑔𝑛(∇𝜎𝑚 ) represent the direction of the gradient
with respect to the mean and standard deviation, respectively, where the
𝑠𝑖𝑔𝑛 function returns the sign of the gradient. 𝐿𝑎𝑦𝑒𝑟𝑚 denotes the 𝑚-th
layer of the image encoder 𝑖 . ⨏ is a hyper-parameter that introduces
randomness. A probability, denoted as 𝜌, is assigned to each layer of
adversarial perturbations during training. And
𝑎𝑑𝑣 𝑎𝑑𝑣 𝑎𝑑𝑣
𝑉𝑚𝑎𝑑𝑣 = (𝑉𝑚−1
, 𝜇𝑚 , 𝜎𝑚 ) = 𝜎𝑚𝑎𝑑𝑣

𝑎𝑑𝑣 − 𝜇(𝑉 𝑎𝑑𝑣 )
𝑉𝑚−1
𝑚−1
𝑎𝑑𝑣 )
𝜎(𝑉𝑚−1

𝑎𝑑𝑣
+ 𝜇𝑚

(13)

where 𝑚 = 1, 2, …, 𝑀. 𝑉0𝑎𝑑𝑣 = 𝑉1 , which denotes the perturbation starts
with the input 𝑉1 . The aﬃne transformation function  applied to the
adversarially perturbed input from the previous layer. 𝑉𝑚𝑎𝑑𝑣 is the perturbed output after 𝐿𝑎𝑦𝑒𝑟𝑚 . Until now, we have generated adversarial styles (features) that diﬀer from the distribution of present domain,
which are regarded as being from possible future domains. Although the
𝑎𝑑𝑣 and original 𝑉 𝑙𝑖 share the same identities, but they are
adversarial 𝑉𝑀
𝑟ℎ
quite diﬀerent. To make adversarial style generation more eﬃcient, we
minimize the following loss function as
𝑎𝑑𝑣
𝑎𝑑𝑣 = 𝑎𝑑𝑣
𝑡𝑖 + 𝑐𝑡

4.3. Performance analysis
We compare with Lifelong ReID methods: CRL [6], AKA [9], CKP
[7], MRN [8], GCReID [30], PTKP [28], LSTKC [32], DKP [31], KRKC
[29], and 𝐶 2 𝑅 [35]; Common ReID methods: FT-RN50, BoT [1], FTViT16, TransReID [52], CLIP-ReID [3]; and Generalized ReID methods:
QAConv [12], CLIP-FGDI [41], M3L [24], DGPS [40], and ERM-SDViT
[39]. Following Section 4.1, we train on four datasets in order: Market1501 (MA) → CUHKSYSU (SY) → MSMT17 (MS) → CUHK03 (CU),
where MA, SY, MS are past and CU is the present domain. FT-RN50 and
FT-ViT are sequentially ﬁne-tuned using 𝑐𝑡 Eq. (4). BoT, TransReID,
and CLIP-ReID are trained on CUHK03 only. We also introduce CRLViT (CRL with ViT backbone), and replace the backbones of LSTKC and
KRKC with CLIP, denoted as LSTKC-CLIP* and KRKC-CLIP*.
Note that among Generalized ReID models, QAConv is single-domain
and trained only on the present domain (CU), while M3L and ERMSDViT are multi-domain and trained jointly on all four datasets. This
joint setup deviates from the LReID setting and aligns more with joint
training, which we further discuss in Section 5.

(14)

where 𝑎𝑑𝑣
is computed as the same way of Eq. (8) with the diﬀer𝑡𝑖
𝑎𝑑𝑣 . Similarly, 𝑎𝑑𝑣
ence that 𝑉𝑟𝑙𝑖 is replaced by adversarial features 𝑉𝑀
𝑐𝑡
ℎ
in Eq. (14) is computed as the same way of Eq. (4), with the diﬀerence
𝑎𝑑𝑣 . Eq. (14) and Eq. (11)
that 𝑉𝑟𝑙𝑖 is replaced by adversarial features 𝑉𝑀
ℎ
are optimized together to train the image encoder 𝑖 .
4. Experiments
4.1. Datasets and evaluation metrics
We conduct experiments on 11 person ReID benchmarks, namely
Market1501 [10], CUHKSYSU [43], MSMT17 [11], CUHK03 [44], Grid
[45], SenseReID [46], CUHK01 [47], CUHK02 [48], VIPER [49], iLIDS
6

Pattern Recognition 172 (2026) 112326

Z. Liu et al.

Table 2
Performance of the proposed model compared with representative models on the past and present (seen) domains (Bold indicates the best results, underline the
second best. ERM-SDViT requires all historical images, making it more joint-training-like, while our model only accesses the present domain and outperforms
ERM-SDViT. Past domains include Market1501, CUHKSYSU, and MSMT17, while the present domain is CUHK03 from 𝑂𝑟𝑑𝑒𝑟_1).
Past

Common ReID

Lifelong ReID

Generalized ReID

Ours

Present

Market1501

CUHKSYSU

MSMT17

CUHK03

Avg

Year

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

BoT [1]
FT-RN50 [53]
transReID [52]
CLIP-ReID [3]
FT-ViT16 [53]

2019
2020
2020
2023
2020

20.70
25.64
43.40
45.20
53.50

42.30
49.14
66.90
69.00
75.70

21.80
63.26
78.20
84.30
77.70

22.50
66.10
80.50
86.20
80.00

2.40
14.11
11.80
14.80
20.80

7.90
66.57
32.10
36.50
46.20

58.80
52.59
74.90
76.30
𝟕𝟔.𝟗𝟎

60.60
54.94
76.80
77.80
𝟕𝟗.𝟐𝟎

25.93
38.90
52.08
55.15
57.23

33.33
59.19
64.08
67.38
70.28

AKA [9]
CRL [6]
CKP [7]
GCReID [30]
𝐶 2 𝑅 [35]
MRN [8]
KRKC [29]
DKP [31]
LSTKC [32]
KRKC-CLIP* [29]
LSTKC-CLIP* [32]
PKTP [28]
CRL-ViT [6]

2021
2021
2023
2024
2024
2022
2023
2024
2024
2024
2024
2022
2021

53.63
53.56
54.88
61.47
72.71
57.60
55.70
60.60
59.50
57.00
56.90
78.60
61.20

74.08
73.43
75.68
80.82
88.84
80.60
75.40
80.00
79.50
78.60
78.40
90.90
79.70

71.00
73.14
73.55
75.56
76.56
77.50
76.60
79.10
82.10
75.80
78.50
87.20
84.60

73.86
76.72
76.76
78.59
79.17
79.80
79.60
81.00
84.00
78.50
80.30
88.50
86.20

15.59
16.67
16.56
19.45
6.15
16.50
9.50
16.00
21.60
9.10
23.60
22.20
29.50

69.38
71.62
71.24
75.43
16.02
39.90
24.00
34.50
45.60
24.50
49.00
44.50
55.60

26.05
26.86
29.32
32.73
34.32
42.90
58.60
48.50
42.00
75.90
59.80
53.50
79.30

28.16
29.16
31.66
35.71
35.57
43.70
60.10
51.10
42.90
77.60
62.80
55.00
80.40

41.57
42.56
43.58
47.30
47.43
48.63
50.10
51.05
51.30
54.45
54.70
60.38
63.65

61.37
62.73
63.84
67.64
54.90
61.00
59.78
61.65
63.00
64.80
67.63
69.73
75.53

QAConv [12]
CLIP-FGDI [41]
M3L [24]
DGPS [40]
ERM-SDViT [39]

2020
2025
2021
2025
2022

27.10
53.50
67.00
55.00
𝟖𝟑.𝟏𝟎

58.90
77.40
84.50
73.90
𝟗𝟏.𝟓𝟎

48.40
78.10
77.70
86.60
𝟗𝟒.𝟐𝟎

52.00
79.70
79.20
88.20
𝟗𝟒.𝟕𝟎

10.50
18.60
21.00
21.90
𝟔𝟏.𝟐𝟎

36.10
47.00
41.60
47.70
𝟖𝟎.𝟎𝟎

53.10
71.50
63.60
79.50
71.80

55.60
74.90
71.00
81.60
73.50

34.77
55.43
57.33
60.75
𝟕𝟕.𝟓𝟖

50.65
69.75
69.08
72.85
𝟖𝟒.𝟗𝟑

TP-LReID

2025

72.20

86.90

91.20

92.50

41.10

67.10

76.00

77.60

70.13

81.03

Performance on past and present domains. As shown in
Table 2, Lifelong ReID outperforms Common ReID on past domains but lags on the present, highlighting their respective strengths.
Generalized ReID, aiming for cross-domain generalization, also underperforms on the present due to its broader focus. ERM-SDViT achieves
the best overall performance by training on all domains jointly. In
contrast, our TP-LReID achieves comparable results without accessing
past data, operating under a continual, single-domain setting. Excluding
Generalized ReID, TP-LReID signiﬁcantly outperforms all Lifelong and
Common ReID methods.
Performance on future domains. As shown in Table 3, TP-LReID
achieves best average performance in terms of mAP and Rank1 compared with other competitive models of Lifelong, Common, Generalized
ReID on future domains, which exceeded the second best generalized
ReID by 1.67 % and 1.82 % according to average mAP and Rank1. Note
that for MRN in Table 3, we directly use the results from the corresponding paper since the dataset settings are the same. In Table 3, ‘-’ indicates
that results are not provided in the corresponding paper.

Additionally, we demonstrate the eﬀectiveness of regularization
(𝑅𝑒𝑔) and prompts (𝑃 𝑡) on struggling against catastrophic forgetting for
ﬁne-tuning (𝐹 𝑇 ) as shown in Table 6. As domain diﬀerences of textual
prompts are more minor than image embeddings, 𝐹 𝑇 + 𝑃 𝑡 demonstrates
the eﬀectiveness of textual prompts, compared to 𝐹 𝑇 in Table 6, in mitigating catastrophic forgetting caused by large diﬀerences in images.
To investigate the eﬀects of non-continual and continual text
prompting, we perform experiments based on ﬁne-tuning (FT) setting.
As shown in Table 7, 𝐹 𝑇 + 𝑛𝑃 𝑇 denotes that the text prompting is noncontinual, i.e., it is randomly initialized at each step 𝑖 of the continuous
ﬁne-tuning. 𝐹 𝑇 + 𝑃 𝑇 denotes that the text prompting is continual, i.e.,
the text prompting at step 𝑖 is initialized by the text prompting at step
𝑖 − 1 (1 < 𝑖 ≤ 4). The results of the experiments in Table 7 indicate that
the continual text prompting (𝐹 𝑇 + 𝑃 𝑇 ) perform better, so all other experiments in this paper that include text prompting are set up accordingly.
For future style generation, we explore the eﬀect of adding perturbations to each layer in transformer with diﬀerent probabilities 𝜌. As
shown in Table 8, We add the perturbation with 10 %, 60 % and 100 %
probability, and the experimental results show that adding the perturbation with 100 % probability works optimally.

4.4. Ablation experiments
In this section, we ﬁrst evaluate the eﬀectiveness of text prompts (denoted by 𝑃 𝑇 in Table 4) and adversarial future simulating (denoted by
𝐴𝐹 in Table 4) based on person ReID with regularization using 𝑑 loss
give in Eq. (7), (denoted by Baseline in Table 4) on the past, present and
future domains in terms of mAP and Rank1, where 𝐵𝑎𝑠𝑒𝑙𝑖𝑛𝑒 + 𝑃 𝑇 + 𝐴𝐹
denotes the proposed model TP-LReID. It can be seen that the proposed
modules are eﬀective to improve performance of the proposed model.
To further verify the how much boost to past and future data of
the pure AF module, we conduct dedicated experiments in more cases.
As shown in Table 5, we report the comparison results of baseline and
baseline+AF under more orders, the experimental results show that the
boost to the future is better than past’s after the addition of AF in most
cases. It proves the necessity of the AF module for simulating future
data.

5. Discussion
The impact of arrival order. We explore how well the proposed
model TP-LReID preserves performance in all 24 training orders. The
performance on the ﬁrst seen domain and the average performance on
unseen domains are shown in Figs. 3 and 4 respectively. As observed,
our model demonstrates less forgetting on the ﬁrst training domain and
better generalization to future domains compared to other Lifelong ReID
methods. Note that for the third column of Fig. 4, the generalization of
our method after training on the ﬁrst step (SY) is sub-optimal. According to our analysis, this is due to the SY domain having fewer engaged
samples compared to other domains, as indicated in Table 1. Neverthe7

Pattern Recognition 172 (2026) 112326

Z. Liu et al.

Table 3
Generalization performance of the proposed model compared with representative models on future (unseen) domains (Bolding represents the best results, underlining
represents the second best results. ‘-’ denotes results and codes are not provided by the authors of [8]. ERM-SDViT requires all historical images are accessible, it is
more joint-training-like. The proposed method achieves comparable performance by simply accessing the current domain).
Future
Grid

SenseReID

CUHK01

CUHK02

VIPeR

iLIDS

PRID

Avg

Year mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑) mAP(↑) Rank1(↑)
FT-RN50
[53]
BoT [1]
transReID
[52]
FTViT16[53]
CLIP-ReID
[3]

2020 12.21

8.00

28.87

23.36

60.99

60.19

77.86

74.27

37.61

28.16

71.44

60.00

20.96

15.00

43.93

37.95

2019 16.20
2021 41.30

11.20
28.80

25.10
43.00

20.40
35.20

71.50
74.70

70.80
74.20

86.50
89.10

86.60
88.30

34.60
66.60

25.90
57.00

73.80
66.30

66.70
53.30

20.40
66.50

16.00
57.00

46.87
63.93

42.51
56.26

2020 44.70

32.00

49.30

40.30

77.10

76.20

91.30

90.20

61.50

50.60

70.10

60.00

58.10

46.00

64.59

56.47

2023 47.70

38.40

47.60

38.80

82.60

83.00

90.20

89.50

68.80

59.80

73.50

61.70

69.90

61.00

68.61

61.74

CRL [6]
AKA [9]
CKP[7]
𝐶 2 𝑅[35]
GCReID
[30]
KRKCCLIP*[29]
KRKC[29]
DKP[31]
LSTKC[32]
LSTKCCLIP*[32]
PTKP [28]
MRN[8]
CRL-ViT[6]

2021 13.20
2021 16.19
2023 16.98
2024 22.48
2024 31.35

7.20
11.20
9.60
15.20
22.40

33.32
34.14
33.10
37.15
39.21

27.22
27.51
26.64
29.63
32.05

62.06
60.91
64.10
66.27
64.26

62.14
59.36
63.99
65.23
63.07

68.93
61.31
69.22
69.96
68.23

65.90
57.74
65.27
68.20
64.85

37.61
40.68
38.32
41.94
45.65

28.16
31.96
29.43
32.91
37.03

71.44
71.87
70.49
60.73
63.92

60.00
60.00
56.67
50.00
53.33

20.96
23.23
25.70
27.59
29.71

15.00
15.00
19.00
20.00
21.00

43.93
44.05
45.42
46.59
48.90

37.95
37.54
38.66
40.17
41.96

2023 33.50

22.40

27.00

21.70

59.40

60.30

86.10

84.30

49.90

39.90

57.20

45.00

38.80

30.00

50.27

43.37

2023 20.70
2024 32.20
2024 33.80
2024 35.70

12.80
24.80
24.80
27.20

38.90
48.50
50.60
50.70

31.30
40.10
41.70
41.20

81.20
70.90
70.80
73.40

82.30
69.80
70.40
72.40

88.70
76.70
74.00
82.60

87.90
74.10
72.40
81.80

47.90
51.80
54.40
58.50

38.90
41.80
44.90
49.40

64.50
64.20
68.10
66.10

56.70
53.30
58.30
55.00

18.10
38.70
42.50
45.40

9.00
28.00
30.00
34.00

51.43
54.71
56.31
58.91

45.56
47.41
48.93
51.57

2020 31.60
2022 –
2021 45.40

23.20
–
33.60

53.00
–
55.40

45.50
–
45.80

81.80
–
83.40

82.70
–
82.80

87.90
–
91.10

87.90
–
89.50

56.00
–
67.80

46.20
–
59.80

72.40
–
75.20

63.30
–
66.70

31.70
–
63.40

22.00
–
53.00

59.20
60.60
68.81

52.97
56.80
61.60

QAConv
Generalized ReID [12]
M3L [24]
CLIPFGDI[41]
DGPS[40]
ERM-SDViT
[39]

2020 25.90

20.80

30.20

23.30

62.90

64.00

65.10

62.70

41.00

32.50

60.40

48.30

31.40

23.30

43.40

41.10

2021 37.50
2025 42.90

27.20
30.40

46.60
47.00

37.40
37.50

66.00
78.10

64.40
77.20

68.70
89.50

66.50
89.30

55.10
66.90

44.60
57.00

72.50
80.00

65.00
73.30

54.50
68.10

41.00
59.00

57.27
67.50

49.44
60.53

2025 51.60
2022 𝟓𝟕.𝟕𝟎

44.00
𝟒𝟔.𝟒𝟎

51.10
58.10

41.80
49.10

85.30
87.00

86.00
𝟖𝟔.𝟗𝟎

92.40
87.80

91.80
87.00

69.20
𝟕𝟕.𝟕𝟎

59.80
𝟔𝟗.𝟎𝟎

72.90
80.00

63.30
𝟕𝟏.𝟕𝟎

73.30
66.80

64.00
58.00

70.83
73.59

64.39
66.87

Ours

2025 55.00

46.40

𝟔𝟒.𝟎𝟎

𝟓𝟓.𝟏𝟎

𝟖𝟕.𝟑𝟎

𝟖𝟔.𝟗𝟎

𝟗𝟏.𝟔𝟎

𝟗𝟏.𝟐𝟎

73.20

63.90

𝟖𝟏.𝟐𝟎

𝟕𝟏.𝟕𝟎

𝟕𝟒.𝟗𝟎

𝟔𝟓.𝟎𝟎

𝟕𝟓.𝟑𝟏

𝟔𝟖.𝟔𝟎

Common ReID

Lifelong ReID

TP-LReID

Table 4
Eﬀectiveness of text prompting (PT) and adversarial future simulating (AF) on the past, present and future
domains.
Past

Baseline
Baseline+PT
Baseline+AF
Baseline+PT+AF

Present

Future

Avg

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

mAP(↑)

Rank1(↑)

62.30
66.43
65.40
𝟔𝟖.𝟏𝟕

77.10
80.63
80.60
𝟖𝟐.𝟏𝟕

75.00
𝟕𝟔.𝟑𝟎
74.90
75.30

76.40
𝟕𝟖.𝟒𝟎
76.40
76.00

69.90
72.35
71.05
𝟕𝟕.𝟔𝟎

66.39
68.40
67.11
𝟔𝟖.𝟔𝟎

68.65
71.69
70.45
𝟕𝟑.𝟏𝟔

71.75
75.81
74.70
𝟕𝟔.𝟏𝟐

The Past domains include three seen domains (Market1501, CUHKSYSU, and MSMT17) and their means are
reported.The Present domain contains only CUHK03.The Future domain includes seven unseen domains (Grid,
SenseReID, CUHK01, CUHK02, VIPER, iLIDS, and PRID) and their means are reported.

less, our method overcomes this limitation and achieves optimality in
the later continual steps.
We further conduct experiments that treat MSMT17 as one of unseen
domains to investigate the impact of it. Under this setup, there are only
three training domains, i.e., Market1501 (MA), CUHKSYSU (SY), and
CUHK03 (CU) sequentially. As shown in Fig. 5, the left side reports the
forgetting eﬀect on the ﬁrst training domain (MA), while the right side
illustrates the generalization performance on one of the unseen domains
(MS).
The impact of pre-training parameters. Transformer-based methods (e.g., CLIP-ReID [3]) are pre-trained on large-scale datasets, while
CNN-based methods [7–9,34] typically use ImageNet. To ensure fairness, we pre-train the CNN backbone (ResNet50) on both ImageNet and

the large-scale dataset used by CLIP-ReID. As shown in Tables 9 and 10,
FT-RN50 (IMG) and FT-RN50 (CLIP-ReID) represent these two settings.
Results show minimal performance gain from large-scale pre-training,
so we use the original ImageNet-pretrained weights for all CNN-based
baselines.
Table 11 shows the detailed results for each training step under
the above settings according to 𝑂𝑟𝑑𝑒𝑟_1. Note that there is no past
for Step1. Speciﬁcally, the Present of Step1 denotes the performance
on itself (Market1501). For Step2, the Past denotes the performance
on Market1501, and the Present denotes the performance on itself
(CUHKSYSU). For Step3, the Past denotes the average performance on
Market1501 and CUHKSYSU, and the Present denotes the performance
on itself (MSMT17). Likewise, the performance of Step4 is calculated
8



exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'TP-LReID - Lifelong person re-identification using text prompts.pdf' - | awk '/3\\.3\\. Future style generation/{flag=1} /4\\. Experiments/{if(flag){print; flag=0}} flag'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Two-stage knowledge distillation for visible-infrared person re-identification.pdf' - | awk '/3\\.5\\. Mutual distillation/{flag=1} /3\\.6\\. Total loss/{if(flag){print; flag=0}} flag'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
3.5. Mutual distillation
In contrast to traditional ReID, which focuses solely on learning a
discriminative representation of identity, VI-ReID performs matching
across modalities, where a query is utilized to retrieve for corresponding images in another modality. The main obstacle lies in huge domain
gaps between visible and infrared images of the same individual, which
is larger than the gap within a single modality. Previous works [6,30]
have addressed this issue through unidirectional matching of visible
features to infrared features and the reverse However, the information
contained in these two modalities is not evenly distributed, and oneway mapping will result in the loss of discriminative features or the
introduction of undesired artifacts.
To avoid unidirectional visible-infrared feature alignment, we propose a mutual distillation (MD) learning to mitigate the distribution discrepancy between modalities in both directions. In contrast to methods
like HCML [18], which independently map inter-modality features to a

𝑁

𝑝
1 ∑
𝑓 (𝑣𝑝𝑖 ) ⋅ 𝑤𝑝𝑣
𝑖 ,
𝑁𝑝 𝑖=1

𝑀𝑝
𝑃 ∑
∑

The SM learning training pipeline is outlined below:
(1) At the first epoch, the network is trained using the loss 𝑆𝐼 and
1 for each
baseline method, and initializing the prototypes 𝑂𝑉1 𝑝 and 𝑂𝑅
𝑝
identity 𝑝 using Eqs. (8) and (9), respectively.
(2) For the 𝑒th epoch, 𝑒 = 2, … , 𝐸, the network is trained by added
𝑒−1
𝑆𝑀 , where 𝑂𝑉𝑒−1
𝑝 and 𝑂𝑅𝑝 are fixed to train network. Following the
𝑒 are adjusted.
end of training, the prototypes 𝑂𝑉𝑒 𝑝 and 𝑂𝑅
𝑝
SM has several advantages: (1) Unlike the center loss, which updates
prototypes in a localized, batch-wise manner, our approach leverages prototypes to provide a comprehensive understanding, resulting
in more inclusive identity features. (2) The prototypes are updated
together with the network training, the ID representation becomes
more discriminative. (3) The prototypes remain fixed to ensure stable
supervision, but the center loss method employs dynamical prototypes.

3.4. Self-mimic learning

𝑂𝑉 𝑝 =

𝑃 ∑
𝑝
∑
‖
‖2
𝑝
‖𝑓 (𝑣𝑖 ) − 𝑂𝑉 𝑝 ‖
‖
‖2

(8)
4

Pattern Recognition 169 (2026) 111850

J. Shi et al.

common embedding, our method integrates alignment with the encoding to improve cross-modality feature matching. We select 𝑛 visible and
infrared feature combinations, represented as 𝐹𝑉 𝑝 = {𝑓 (𝑣𝑝1 ), … , 𝑓 (𝑣𝑝𝑛 )}
and 𝐹𝑅𝑝 = {𝑓 (𝑟𝑝1 ), … , 𝑓 (𝑟𝑝𝑛 )}. We then apply the mutual distillation loss
to make the feature distribution of visible samples similar to that of the
infrared samples, and the same for the reverse.
𝑃
1 ∑ 1
1
𝑀𝐷 =
( 𝐷(𝐹𝑅𝑝 , 𝑠𝑔(𝐹𝑉 𝑝 )) + 𝐷(𝐹𝑉 𝑝 , 𝑠𝑔(𝐹𝑅𝑝 ))),
𝑃 𝑝=1 2
2

Algorithm 1 : Two-stage knowledge distillation.
{
}
{
}
Input: Training set 𝐷 = (𝑉𝐿 , 𝑅𝐿 ) or 𝐷 = (𝑉𝐿 , 𝑉𝑈 , 𝑅𝐿 , 𝑅𝑈 ) , total
training iterations 𝑚𝑎𝑥_𝑒𝑝𝑜𝑐ℎ, iterations of adding mutual distillation
𝑏𝑒𝑔𝑖𝑛_𝑀𝐷.
Initialize: Initialize parameters 𝜃.
1: for 𝑡 = 1 ∶ 𝑚𝑎𝑥_𝑒𝑝𝑜𝑐ℎ do
2:
Update label 𝑦̂𝑡 with Eq. (3)
3:
if 𝑡 < 𝑏𝑒𝑔𝑖𝑛_𝑀𝐷 then
4:
𝑀𝐷 = 0;
5:
else
6:
Compute 𝑀𝐷 with Eq.(11);
7:
end if
8:
Compute 𝑆𝑀 with Eq.(10), 𝑆𝐼 and 𝑇 𝑟𝑖 with Eq.7 and

(11)

𝑠𝑔(⋅) represents the stop-gradient operation and 𝐷(⋅, ⋅) is MMD2 [31].
MMD2 is a well-known non-parametric metric in domain adaptation
and has been empirically proven to perform better than alternatives
like KL divergence. This superior performance is attributed to its robustness and flexibility in capturing differences between complex, highdimensional distributions.
MMD2 is constructed as:
MMD2 (𝐹𝑅𝑝 , 𝐹𝑉 𝑝 ) =

1 ∑∑
𝑘(𝑓 (𝑟𝑝𝑖 ), 𝑓 (𝑟𝑝𝑗 ))
𝑛2 𝑖=1 𝑗=1

+

1 ∑∑
𝑘(𝑓 (𝑣𝑝𝑖 ), 𝑓 (𝑣𝑝𝑗 ))
𝑛2 𝑖=1 𝑗=1

𝑛

𝑛

Eq.(14), respectively;
Optimize 𝜃 based on overall  ( Eq.(13))
Update prototype for each ID in each modality with Eq.(8) and
Eq.(9).
11: end for
9:
10:

𝑛

𝑛

(12)
As shown in Fig. 1, our observation is that intra-modality samples exhibit higher similarity in the feature space compared to crossmodality samples. This implies that mutual distillation learning is inherently more challenging than self-mimic learning. Additionally, intramodality variance adds complexity to inter-modality feature alignment.
To address this, we introduce a two-stage feature alignment strategy,
where we first diminish the intra-modality divergences to tighten the
distributions and then conduct cross-modality feature alignment. The
step-by-step training process is presented in Algorithm 1.

2 ∑∑
−
𝑘(𝑓 (𝑟𝑝𝑖 ), 𝑓 (𝑣𝑝𝑗 )),
𝑛2 𝑖=1 𝑗=1
𝑛

𝑛

−‖𝑠−𝑠′ ‖2

where 𝑘(𝑠, 𝑠′ ) = exp( ‖ 2𝜎 2 ‖2 ) is a Gaussian kernel.
MD presents two major strengths: (1) Bidirectional feature alignment balances the suppression of trivial features in the visible modality and reinforcing critical features in the infrared modality. (2) MD
prioritizes the consistency of representation spaces across numerous
instances spanning both modalities, instead of attempting to align
individual pairs of samples. This approach simplifies the reduction
of statistical differences between the modalities, rather than mandating complete consistency for every pair of samples from different
modalities.

4. Experiment
In this section, we perform extensive experiments to assess our proposed approach in both fully supervised and weakly supervised settings.
To clearly outline our experiments, we first describe the experimental
setup and implementation details. Next, we present results on two
common datasets and conduct ablation studies to analyze the model’s
characteristics.

3.6. The total loss
The total loss contains four terms: the soft-identity loss, the triplet
loss, the self-mimic learning loss, and the mutual learning loss, which
is formulated as:
𝑇 𝑜𝑡𝑎𝑙 = 𝑆𝐼 + 𝑇 𝑟𝑖 + 𝜆𝑆𝑀 𝑆𝑀 + 𝜆𝑀𝐷 𝑀𝐷 ,

4.1. Datasets, setting, and implementation
SYSU-MM01 is an extensive and challenging dataset for VI-ReID [8].
It includes 287,628 visible images and 15,792 infrared images captured
across six camera views (four visible and two near-infrared). The
dataset includes 491 identities in total. For training, there are 22,258
visible images and 11,909 infrared images covering 395 individuals.
For evaluation, we use 3803 infrared images as queries and randomly
select 3010 visible images to form the gallery. RegDB [33] includes 412
individuals, each with 10 visible and 10 infrared images. The dataset
is split randomly, with half allocated for training and the other half for
testing.
Evaluation Protocols. Model performance is assessed using Cumulative Match Characteristic (CMC) and Mean Average Precision (mAP).
For an equitable comparison, results are reported for both the all-search
mode and indoor-search mode on SYSU-MM01 using the official code.
In the all-search mode, images from all available cameras are used
to form the gallery set. Conversely, in the indoor-search mode, only
images from the 1st, 2nd, 3rd, and 6th cameras, which are indoor, are
used for the gallery set. Additionally, results for RegDB are provided
according to the evaluation protocols outlined in the work [34]. This
dataset features two modes: Visible2thermal and Thermal2visible. In
the Visible2thermal mode, visible images are used as queries and
infrared images as the gallery. In the Thermal2visible mode, the roles
are reversed, with infrared images serving as queries and visible images
as the gallery.

(13)

where 𝜆𝑆𝑀 and 𝜆𝑀𝐷 serve as trade-off coefficients. The first 𝑆𝐼 and
the second 𝑇 𝑟𝑖 are soft-identity loss and triplet loss.
To maximize the inter-class discrepancy and intra-class similarity,
we employ the triplet loss [32] in the two modalities. For the visible
modality, let 𝑣𝑎𝑖 denote a randomly selected anchor, 𝑣𝑝𝑜𝑠
denote the
𝑖
feature of the hardest positive sample that has the same identity as 𝑣𝑎𝑖
within the batch and 𝑣𝑛𝑒𝑔
denotes the feature of the hardest negative
𝑖
sample that has a different identity as 𝑣𝑎𝑖 within the batch. The triplet
loss for visible images is expressed as follows:
𝑉𝑇 𝑟𝑖 =

]
𝑁𝑣 [
∑
( 𝑝𝑜𝑠 )‖2 ‖ ( )
( 𝑛𝑒𝑔 )‖2
‖ ( 𝑎)
‖𝑓 𝑣𝑖 − 𝑓 𝑣𝑖 ‖ − ‖𝑓 𝑣𝑎𝑖 − 𝑓 𝑣𝑖 ‖ + 𝛾 ,
‖
‖2 ‖
‖2
+

(14)

𝑖

where 𝛾 is a margin parameter. So do the triplet loss for the infrared
modality formulated as:
]
𝑁𝑟 [
∑
( 𝑝𝑜𝑠 )‖2 ‖ ( 𝑎 )
( 𝑛𝑒𝑔 )‖2
‖ ( 𝑎)
𝑅
=
𝑓
𝑟
−
𝑓
𝑟
−
𝑓
𝑟
−
𝑓
𝑟
+
𝛾
.
(15)
‖
‖
‖
‖
𝑖
𝑖
𝑖
𝑖
𝑇 𝑟𝑖
‖
‖2 ‖
‖2
+
𝑖

The overall 𝑇 𝑟𝑖 loss can be formulated as a combination of 𝑉𝑇 𝑟𝑖 and
𝑅
as follows:
𝑇 𝑟𝑖
𝑇 𝑟𝑖 = 𝑉𝑇 𝑟𝑖 + 𝑅
𝑇 𝑟𝑖 .

(16)
5

Pattern Recognition 169 (2026) 111850

J. Shi et al.

Table 1
Comparisons with advanced methods on SYSU-MM01. The top two results are highlighted in red and blue, representing the 1st and 2nd best
performances, respectively.
Method

Venue

All search

Indoor search

Rank-1

Rank-10

Rank-20

mAP

Rank-1

Rank-10

Rank-20

mAP

D2 RL
AlignGAN

CVPR’19
ICCV’19

28.9
42.4

70.6
85.0

82.4
93.7

29.2
40.7

–
45.9

–
87.6

–
94.4

–
54.3

JSIA
DDAG
HAT

AAAI’20
ECCV’20
TIFS’20

38.1
54.8
55.3

80.7
90.4
92.1

89.9
95.8
97.4

36.9
53.0
53.9

43.8
61.0
62.1

86.2
94.1
95.8

94.2
98.4
99.2

52.9
68.0
69.4

NFS
MCLNet
SMCL
CAJ

CVPR’21
ICCV’21
ICCV’21
ICCV’21

56.9
65.4
67.4
69.9

91.3
93.3
92.9
95.6

96.5
97.1
96.8
98.5

55.5
61.9
61.8
66.9

62.8
72.6
68.8
76.3

96.5
96.9
96.6
97.9

99.1
99.2
98.8
99.5

69.8
76.6
75.6
78.2

FMCNet
DART
CMT
DCLNet
MAUM𝐺
SPOT
SMMD

CVPR’22
CVPR’22
ECCV’22
MM’22
CVPR’22
TIP’22
ICME’22

66.3
68.7
71.9
70.8
61.6
65.3
64.8

–
96.4
96.5
–
–
92.7
91.3

–
98.7
98.9
–
–
97.0
96.6

62.5
66.3
68.6
65.2
60.0
62.3
60.2

68.2
72.5
76.9
73.5
67.1
69.4
76.5

–
97.8
97.7
–
–
96.2
96.1

–
99.5
99.6
–
–
99.1
99.0

74.1
78.2
79.9
76.8
73.6
74.6
73.8

MRCN
MCBD
DPIS
CAJ+
DEEN
𝐺2 𝐷𝐴

AAAI’23
TIFS’23
ICCV’23
TPAMI’23
CVPR’23
PR’23

68.9
71.6
68.2
71.5
74.7
63.9

95.2
95.0
–
96.2
97.6
93.3

98.4
98.4
–
98.7
99.2
97.3

65.5
67.3
65.7
68.2
71.8
60.7

76.0
79.4
71.7
78.4
80.3
71.1

98.3
98.3
–
98.4
99.0
97.3

99.7
99.9
–
99.8
99.8
99.5

79.8
79.9
77.0
82.0
83.3
76.0

WF-CAMReViT
LCNL

PR’24
IJCV’24

68.1
70.2

97.1
96.4

96.9
99.0

65.2
68.0

72.4
76.2

97.2
98.2

98.9
99.8

77.6
80.3

Ours

–

76.6

97.1

99.1

73.0

82.7

98.9

99.8

85.3

Implementation Details. Our code is implemented using PyTorch and
runs on a standard machine equipped with an RTX 3090 GPU. Following [35], a two-stream feature extractor pre-trained on ImageNet is
employed to obtain 2048d features. For data augmentation, we apply
a channel-mixed learning strategy as described in [32]. Our model is
trained using the SGD optimizer with a momentum of 0.9. The initial
learning rate is set to 0.1, employing a warm-up strategy [36], and
decreases by a factor of 0.1 at the 20th epoch and by 0.01 at the 40th
epoch, across a total of 80 epochs. The batch size in our experiment is
48, where we randomly select 6 identities and sample 4 visible and 4
infrared images per identity. The trade-off parameters 𝜆𝑆𝑀 , and 𝜆𝑀𝐷
in Eq. (13) are set to 1 and 0.05, respectively.

4.3. Results on RegDB
We compare our TSKD with advanced VI-ReID methods on RegDB
under the full-supervised setting. As illustrated in Table 2, we achieve
impressive performance and highly consistent performances across both
Visible2thermal and Thermal2visible modes. Especially, TSKD achieves
impressive results with 91.1% Rank-1 accuracy and 81.7% mAP in the
Visible2thermal mode, and 89.9% Rank-1 accuracy and 80.5% mAP
in the Thermal2visible mode. These results significantly outperform
LCNL [50], exceeding its Rank-1 accuracy by 5.5% and mAP by 3.0%
in the Visible2thermal mode, and surpassing it by 5.9% in Rank-1 and
3.6% in mAP in the Thermal2visible mode.
We also compare our TSKD with various VI-ReID methods on RegDB
under the weak-supervised setting. As illustrated in Table 3, our TSKD
can achieve superior performances across both modes. Especially, our
model with 50% labeled data achieves 86.8% Rank-1 and 77.3% mAP
in the Visible2thermal mode, and 82.1% Rank-1 and 75.2% mAP in the
Thermal2visible mode.

4.2. Results on SYSU-MM01
As shown in Table 1, we compare our TSKD with 25 advanced
methods on SYSU-MM01 under the full-supervised setting. Unless otherwise specified, all methods are analyzed in the all-search mode on the
SYSU-MM01 dataset.
Our method achieves impressive results, recording a Rank-1 accuracy of 76.6% and an mAP of 73.0%. Thanks to the proposed SI, SM,
and MD modules, TSKD can reduce the cross-modality gap. Compared
with DDAG whose backbone is similar to TSKD, TSKD outperforms it by
21.8% in Rank-1 accuracy and 20.0% in mAP without adding extra network parameters. Furthermore, our method improved Rank-1 by 1.9%
compared to the second-best DEEN, demonstrating its effectiveness for
VI-ReID.
What is more, we also compare our TSKD with 6 SOTA methods
under the weak-supervised setting: DDAG [7], AGW [35], CAJ [32],
LbA [37], DART [26], and DPIS [38], as shown in Table 3. Our TSKD
gains impressive results with Rank-1 accuracy of 58.2%, 68.5%, and
73.3% with 10%, 25%, and 50% labeled samples, respectively. It shows
that TSKD can produce dependable cross-modality pseudo-labels for
guiding the model optimization. Compared to DPIS, our TSKD surpasses
it by 5.4% in Rank-1 accuracy with 50% labeled samples. Furthermore,
compared to other advanced methods, our TSKD demonstrates impressive performance, highlighting its effectiveness for the weak-supervised
VI-ReID.

4.4. Ablation study
To further validate the effectiveness of each component of TSKD, we
perform ablation studies on the SYSU-MM01 dataset. If no otherwise
specified, we only use 10% labeled samples under the weak-supervised
setting in this section.
As illustrated in Index 1 of Table 4, the baseline of TSKD employs
DDAG equipped with the CE loss and the triplet loss, which gains
65.1% Rank-1 accuracy and 62.3% mAP under full-supervised setting
and achieves 43.0% Rank-1 accuracy and 37.8% mAP under the weaksupervised setting. Next, we present the results after integrating the
proposed SI, SM, and MD components. By replacing cross-entropy loss
with soft identity loss 𝑆𝐼 (Index 2 in Table 4), the performance
improves to 66.3% Rank-1 accuracy and 62.9% mAP under the fullsupervised setting. Besides, the performance reaches 48.3% Rank-1
and 43.2% mAP under the weak-supervised setting. By adding selfmimic learning loss 𝑆𝑀 (Index 3 in Table 4), the performance shows
a significant enhancement, achieving 71.9% Rank-1 and 68.2% mAP
6

Pattern Recognition 169 (2026) 111850

J. Shi et al.

Table 2
Comparisons with SOTA methods on RegDB. The top two results are highlighted in red and blue, representing the 1st and
2nd best performances, respectively.
Methods

Venue

Visible2thermal

Thermal2visible

Rank-1

mAP

Rank-1

mAP

JSIA [39]
DDAG [7]
Hi-CMD [5]
cm-SSFT [40]

AAAI’20
ECCV’20
CVPR’20
CVPR’20

48.5
69.3
70.9
72.3

49.3
63.5
66.0
72.9

48.1
68.1
–
71.0

48.9
61.8
–
71.7

MCLNet [41]
SMCL [42]
CM-NAS [43]
CAJ [32]

ICCV’21
ICCV’21
ICCV’21
ICCV’21

80.3
83.9
85.5
85.0

73.1
79.8
80.3
79.1

75.9
83.1
82.6
84.8

69.5
79.0
78.3
77.8

DART [26]
DCLNet [44]
MAUM𝐺 [45]
SPOT [46]

CVPR’22
MM’22
CVPR’22
TIP’22

83.6
81.2
83.4
80.4

75.7
74.3
78.8
72.5

82.0
78.0
81.1
79.4

73.8
69.5
78.9
72.3

PMT [47]
DPIS [38]
CAJ+ [48]
𝐺2 𝐷𝐴 [49]

AAAI’23
ICCV’23
TPAMI’23
PR’23

84.8
86.3
85.7
74.0

76.5
77.4
79.7
65.5

84.1
82.1
84.0
70.0

75.1
74.4
76.9
62.0

LCNL [50]

IJCV’24

85.6

78.7

84.0

76.9

Ours

–

91.1

81.7

89.9

80.5

Table 3
Comparisons with six advanced methods under the weak-supervised setting, and all methods are evaluated using Rank-1 (%) and mAP (%). All results
are re-implemented using public codes. The top two results are highlighted in red and blue, representing the 1st and 2nd best performances, respectively.
Settings

SYSU-MM01

RegDB

All search

Indoor search

Visible2thermal

Thermal2visible

Rate

Method

Venue

Year

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

Rank-1

mAP

10%

DDAG [7]
AGW [35]
CAJ [32]
LbA [37]
DART [26]
DPIS [38]
Ours

ECCV
TPAMI
ICCV
ICCV
CVPR
ICCV
–

2020
2021
2021
2021
2022
2023
–

29.8
32.4
35.6
30.3
31.0
57.7
58.2

27.9
33.4
37.2
29.9
29.6
55.0
56.3

33.6
38.0
38.1
32.9
39.7
62.9
63.6

42.1
49.2
47.1
43.2
22.0
69.8
70.6

44.0
22.8
21.3
41.7
46.2
55.0
56.1

38.6
18.3
17.1
36.7
40.3
50.8
51.9

44.0
20.9
21.2
42.1
45.0
55.1
55.3

37.8
19.4
17.1
36.9
37.0
49.6
50.4

25%

DDAG [7]
AGW [35]
CAJ [32]
LbA [37]
DART [26]
DPIS [38]
Ours

ECCV
TPAMI
ICCV
ICCV
CVPR
ICCV
–

2020
2021
2021
2021
2022
2023
–

39.1
36.9
48.8
49.4
51.9
65.2
68.5

37.0
37.7
89.2
47.4
49.3
61.8
62.2

43.5
43.1
95.8
54.0
57.1
70.5
72.8

51.7
54.3
48.0
62.6
64.3
75.6
76.3

52.5
57.7
57.1
51.3
60.0
77.3
78.1

48.1
49.8
93.6
43.7
49.7
67.7
69.0

52.9
57.1
97.7
47.5
57.4
75.3
76.1

47.9
49.0
64.7
43.7
49.0
66.3
67.2

50%

DDAG [7]
AGW [35]
CAJ [32]
LbA [37]
DART [26]
DPIS [38]
Ours

ECCV
TPAMI
ICCV
ICCV
CVPR
ICCV
–

2020
2021
2021
2021
2022
2023
–

42.3
44.0
59.4
53.4
56.1
67.9
73.3

38.7
40.6
57.0
52.2
55.2
64.2
67.9

49.0
48.6
62.6
58.9
62.5
71.8
77.2

55.5
53.8
64.3
62.6
69.6
76.6
80.8

59.8
70.8
77.4
67.6
78.2
85.6
86.8

53.2
62.7
66.8
61.7
67.0
76.7
77.3

57.0
68.2
74.9
65.8
75.0
81.4
82.1

51.2
61.4
63.1
60.0
68.4
74.1
75.2

Table 4
Ablation study on the SYSU-MM01 dataset under full-supervised and weak-supervised settings.
Index

Approach

All Search
Full-supervised setting

1
2
3
4
5
6

Weak-supervised setting

SI

SM

MD

Rank-1

Rank-10

Rank-20

mAP

Rank-1

Rank-10

Rank-20

mAP

×
✓
✓
✓
×
✓

×
×
✓
×
✓
✓

×
×
×
✓
✓
✓

65.1
66.3
71.9
74.7
74.9
76.6

94.7
94.6
95.8
96.9
96.2
97.1

98.1
97.9
98.6
99.1
98.9
99.1

62.3
62.9
68.2
71.3
72.1
73.0

43.0
48.3
53.5
51.4
56.0
58.2

80.8
84.4
89.1
87.5
90.1
92.3

89.7
91.9
95.3
94.5
95.9
95.2

37.8
43.2
48.6
47.2
51.7
56.3

under the full-supervised setting. Under the weak-supervised setting, it
reaches 53.5% Rank-1 and 48.6% mAP. These improvements highlight
the effectiveness of eliminating redundant intra-modality information.
With the incorporation of the mutual distillation loss 𝑀𝐷 (Index 4 in
Table 4), our TSKD gains 74.7% Rank-1 and 71.3% mAP under the fullsupervised setting, and achieves 51.4% Rank-1 and 47.2% mAP under

the weak-supervised setting, which demonstrates the crucial role of
aligning modality distributions in reducing discrepancies across modalities. To investigate the impact of noise pseudo-labels on SM and MD
(Index 5 in Table 4), we remove the identity confidence and integrate
only self-mimic learning (𝑆𝑀 ) and mutual distillation (𝑀𝐷 ), which
achieves 74.9% Rank-1 accuracy and 72.1% mAP under full-supervised
7

Pattern Recognition 169 (2026) 111850

J. Shi et al.

Fig. 4. Rank-1 and mAP with different mimic learning schemes on the SYSU-MM01 dataset.

setting and achieves 56.0% Rank-1 accuracy and 51.7% mAP under
the weak-supervised setting. Finally, by employing all loss (Index 6
in Table 4), our TSKD gains the highest results in both settings. Our
TSKD achieves 76.6% Rank-1 and 73.0% mAP under the full-supervised
setting, surpassing the baseline by substantial margins of 11.5% Rank-1
and 10.7% mAP. Under the weak-supervised setting, TSKD also gains
58.2% Rank-1 and 56.3% mAP, surpassing the baseline by 15.2% in
Rank-1 and 18.5% in mAP.
The analysis shows that removing identity confidence (Index 5 in
Table 4) leads to a clear drop in performance compared to the configuration with all components (Index 6 in Table 4). Specifically, Rank-1
accuracy decreases from 76.6% to 74.9% under the full-supervised
setting and from 58.2% to 51.7% under the weak-supervised setting,
indicating that noise in pseudo-labels has a noticeable impact on SM
and MD due to the increased risk of misalignment. However, even with
noisy pseudo-labels, SM and MD demonstrate their ability to enhance
performance. For instance, under the weak-supervised setting, Rank-1
accuracy improves from 43.0% (Index 1 in Table 4) to 65.0% (Index 5
in Table 4), showing their effectiveness in handling noise. Additionally,
TSKD (Index 6 in Table 4) significantly reduces the impact of noisy
pseudo-labels by integrating SI with SM and MD, further improving
their combined performance and achieving better overall results.

the prototypes using Eq. (10). The online mimic learning scheme is
similar to the center loss [51] that updates the prototypes as the deep
features change in a more effective way. As shown in Fig. 4, the online
mimic learning achieves 65.2% in mAP which makes an improvement
of 2.3% against the baseline model, while the center loss achieves a
gain of 2.8%. It indicates that the online learning scheme is effective
for Re-ID, and a good prototype leads to a superior performance.
From Fig. 4 we can also observe that our SM learning scheme
achieves the best results and outperforms other schemes by a large
margin. Its main difference from the center loss is described in Section 3.4. The Fig. 4 verifies the effectiveness of our SM learning scheme
which can combine the strengths of offline mimic learning and online
mimic learning to obtain globally viewed, stable, and discriminative
prototypes.
Analysis of self-mimic learning. The goal of SM is to force the
features of each identity’s images captured in the same modality to be
similar. Therefore, we visualize the mean and variance of the distances
between visible images and distances between infrared images for each
identity in Fig. 5. The results show that, compared to the baseline
model, SM significantly reduces the means and variances within the
same modality. Furthermore, we visualize the distribution plot of distances between intra-modality and inter-modality features in Fig. 6. It
is observed that SM effectively clusters the samples of each identity,
benefiting both intra-modality and inter-modality samples.
Analysis of mutual distillation. To clearly show the effectiveness
of MD, we compare three cross-modality feature alignment methods,
including the infrared feature mimics the visible feature (infrared→visible), the visible feature mimics the infrared feature (visible→infrared),
and our MD (infrared↔visible). As shown in Fig. 7, infrared→visible
has the worst performance as there is less information in infrared
images than visible images, making it difficult to map infrared features
to the visible modality. In contrast, our MD achieves the best results.
MD also can mitigate the diversity of feature distributions of intra- and
cross-modality samples. We plot 𝑆𝑀 and mAP at each epoch with and
without MD in Fig. 8. It shows that MD can reduce 𝑆𝑀 after using
𝑀𝐷 , which means that MD can reduce the distances between intramodality images. Meanwhile, as illustrated in Fig. 6, the results also
demonstrate that MD achieves the most compact distribution with the
smallest variance and mean.
Analysis of pseudo-label accuracy. We evaluate the accuracy of
pseudo-labels generated by TSKD with 10% annotation. As shown in
Fig. 9, TSKD achieves higher accuracy compared to DPIS in both RGB
and IR modalities. For the RGB modality, TSKD achieves an accuracy
of 80.7%, outperforming DPIS by 2.5%. Similarly, in the IR modality, TSKD reaches an accuracy of 78.6%, which is 2.2% higher than
DPIS. These results highlight the effectiveness of TSKD in improving
pseudo-label accuracy under weakly supervised settings.
Analysis of two-stage feature alignment strategy. To validate impact
of the two-stage feature alignment strategy, we show the Rank-1 and

4.5. More analysis of our method
Comparison of mimic schemes. To further investigate the effectiveness of the proposed self-mimic (SM) learning, we conduct a comparison of SM with other mimic learning schemes, including offline mimic
learning, online mimic learning, and center loss [51]. The results on
the SYSU-MM01 dataset are reported in Fig. 4.
(1) Offline mimic learning. In this scheme, we firstly train two
traditional person Re-ID models in the RGB and IR modalities using
our baseline model, independently. Then, we compute the prototypes of
𝑂𝑉 𝑝 and 𝑂𝑅𝑝 by collecting all samples in the training set that belong to
ID 𝑝 in the RGB and IR modalities, respectively. After that, we train the
baseline model by mimicking the prototypes using Eq. (10). The offline
mimic learning is conducted in a straightforward way that calculates
the prototypes of all identities using Eqs. (8) and (9). It updates the
prototypes considering all samples and the learned prototypes are fixed
as a guidance in Eq. (10). It is observed in Fig. 4 that the offline mimic
learning scheme achieves the mAP of 62.4% which is smaller than the
baseline model of 62.9%. The reason is that the prototypes are fixed
during updating the network, and a bad prototype fails to push other
identities.
(2) Online mimic learning. This scheme jointly learns the prototypes
of all identities and minimizes the sample distances to their corresponding prototypes. Specifically, we firstly extract the features of the
samples in every training batch. Then we update the prototypes of the
identities in the batch. Finally, we update the features by mimicking
8



codex
补抓的细节确认了两点：`TP-LReID` 的证据闭环靠文本提示、未来风格扰动、到达顺序三类对照；`TSKD` 的核心闭环不是“蒸馏有用”，而是证明先收紧同模态分布再做双向跨模态分布对齐更合理。现在读 UAV 综述和复杂光照 Transformer。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Computer Vision and Image Understanding 251 (2025) 104261

Contents lists available at ScienceDirect

Computer Vision and Image Understanding
journal homepage: www.elsevier.com/locate/cviu

UAV-based person re-identification: A survey of UAV datasets, approaches,
and challenges
Yousaf Albaluchi a ,∗, Biying Fu b,c , Naser Damer b,d , Raghavendra Ramachandra a , Kiran Raja a
a Norwegian University of Science and Technology, 2802 Gjøvik, Norway
b

Fraunhofer Institute for Computer Graphics Research IGD, 64283 Darmstad, Germany
RheinMain University of Applied Sciences, Wiesbaden, Germany
d
Department of Computer Science, TU Darmstadt, Darmstadt, Germany
c

ARTICLE

INFO

MSC:
41A05
41A10
65D05
65D17
Keywords:
ReID
UAV
Drones
Identification
Surveillance

ABSTRACT
Person re-identification (ReID) has gained significant interest due to growing public safety concerns that
require advanced surveillance and identification mechanisms. While most existing ReID research relies on
static surveillance cameras, the use of Unmanned Aerial Vehicles (UAVs) for surveillance has recently gained
popularity. Noting the promising application of UAVs in ReID, this paper presents a comprehensive overview of
UAV-based ReID, highlighting publicly available datasets, key challenges, and methodologies. We summarize
and consolidate evaluations conducted across multiple studies, providing a unified perspective on the state of
UAV-based ReID research. Despite their limited size and diversity, We underscore current datasets’ importance
in advancing UAV-based ReID research. The survey also presents a list of all available approaches for
UAV-based ReID. The survey presents challenges associated with UAV-based ReID, including environmental
conditions, image quality issues, and privacy concerns. We discuss dynamic adaptation techniques, multi-model
fusion, and lightweight algorithms to leverage ground-based person ReID datasets for UAV applications. Finally,
we explore potential research directions, highlighting the need for diverse datasets, lightweight algorithms, and
innovative approaches to tackle the unique challenges of UAV-based person ReID.

1. Introduction
UAV technology has advanced remarkably over the past decade,
highlighted by the growing UAV market, which is projected to reach 43
billion dollars by 2025 (Abdelraziq et al., 2023). The Federal Aviation
Administration (FAA) has registered approximately 855,860 UAVs in
the United States. The drone market is expected to grow at an estimated
annual rate of 6.4%, with its size projected to double by 2024 (Sharma
and Mehra, 2023). The rapid progress in UAV technology, remarkable
maneuverability, and adaptable designs has generated substantial enthusiasm, resulting in a significant increase in their application across
various domains, including surveillance and public safety, spanning
commercial and personal uses, (Messaoudi et al., 2023; Srigrarom
et al., 2021; Bushnaq et al., 2021; Ruetten et al., 2020; Zhou et al.,
2021), such as aerial photography (Peng et al., 2017). UAVs provide
unprecedented mobility and coverage for surveillance (Kumar and Kumar, 2023), allowing for tracking individuals across large and complex
environments that traditional fixed cameras cannot cover (Mohsan
et al., 2023). This capability is essential for applications such as search
and rescue operations (Kyriakakis et al., 2022), crowd monitoring at

large events (AL-Dosari et al., 2023), and border surveillance (Martins
and Jumbert, 2023).
As the use of UAVs in surveillance grows, so does the need for
advanced identification and monitoring systems (Fang and Savkin,
2024; Wu et al., 2019; Zheng et al., 2019; Hou et al., 2019; Zhang
et al., 2021a; Wu et al., 2021b; Fan et al., 2019). This need has
brought attention to Person Re-Identification, a crucial component
of modern surveillance systems. ReID allows for the recognition of
individuals across different camera views, making it a focal point of
interest within the computer vision community (Zahra et al., 2023;
Grigorev et al., 2019; Tian et al., 2015; Liu et al., 2018). Traditionally,
person ReID has relied on static, ground-based camera systems, such
as CCTV, to recognize individuals in various views (Ye et al., 2022;
Zhang et al., 2020; Yang et al., 2014; Tang et al., 2019; Zhang et al.,
2017). However, these systems face limitations, particularly in covering wide geographical areas, adjusting to dynamic environments, and
managing changing perspectives (Zhang et al., 2021). With their ability
to provide dynamic and flexible aerial views, UAVs offer a promising
alternative to conventional surveillance systems (Zhang et al., 2021).

∗ Corresponding author.

E-mail address: ymalbalu@stud.ntnu.no (Y. Albaluchi).
https://doi.org/10.1016/j.cviu.2024.104261
Received 31 May 2024; Received in revised form 15 November 2024; Accepted 5 December 2024
Available online 14 December 2024
1077-3142/© 2024 The Author(s). Published by Elsevier Inc. This is an open access article under the CC BY license
(http://creativecommons.org/licenses/by/4.0/).

Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261

Fig. 1. Common and distinct challenges across ground-based and UAV-based person ReID.

• The challenges and opportunities for future research
In addition to these aspects, this survey further highlights other
unique challenges not present in traditional ground-based ReID systems, such as:

The comparison between ground-based and UAV-based person ReID
(Fig. 1) highlights unique and shared challenges. Ground-based ReID
operates in static environments, while UAV-based person ReID faces
dynamic backgrounds (Organisciak et al., 2021), motion blur (Koo
et al., 2020), and battery life constraints. Both ground-based and UAVbased person ReID have similar limitations, such as low resolution (Xu
et al., 2023), real-time processing (Gaikwad and Karmakar, 2022), high
computational power, privacy concerns (Alipour-Fanid et al., 2020),
dataset challenges (Moritz et al., 2021), illumination variations (Fu
et al., 2022), and viewpoint variation issues (Xu et al., 2023). These
limitations collectively influence the performance and reliability of person ReID across both domains. For several reasons, integrating person
ReID with UAV technology is particularly crucial and challenging.
Illustrated in Fig. 2 Out of more than 140 research papers in the
ReID dataset domain, only seven papers specifically address UAVbased person ReIDs, and only seven datasets currently serve as suitable
benchmarks for UAV-based person ReID. This highlights that drone
ReID is a relatively new and emerging area of research, with the initial
publicly accessible study utilizing drones to gather a dataset for ReID
dating back to Layne et al. (2014).
The availability of high-quality, diverse, and well-annotated
datasets is crucial for training and evaluating computer vision models
in this domain. Despite the potential advantages, using UAVs for
person ReID remains an emerging field with several research gaps.
The fundamental obstacle lies in the complex task of collecting and
annotating extensive datasets, as the human processing and annotation
of images necessitates a substantial commitment of time (Grigorev
et al., 2019).
Additionally, the aerial perspective presents unique challenges that
ground-based ReID systems are not equipped to address. Zhou et al.
(2023). The constantly changing viewpoints, varying altitudes, and
dynamic backgrounds in UAV footage make it significantly more difficult to maintain consistent person identification. Moreover, the lower
resolution of aerial images and the need for real-time processing on
resource-constrained UAV domains demand novel approaches to feature extraction and matching.
This survey aims to identify the gap in the literature by providing a
comprehensive overview of the state of research on UAV-based person
ReID. Specifically, it focuses on three key aspects:

• Motion Blur
• Dynamic backgrounds
• Resource constraints like battery life and computation power
These challenges underline the need for specialized research and development in UAV-based person ReID. Ground-based ReID systems are
not equipped to handle these unique aerial perspectives and conditions.
This survey focuses on UAV-based person ReID to emphasize critical
challenges and opportunities, offering a comprehensive overview of the
current state of research in this rapidly evolving field. Furthermore,
the use of UAVs for person ReID raises significant ethical concerns that
require careful consideration.
By consolidating current research, we aim to highlight the critical
challenges that need to be addressed to improve UAV-based person
ReID systems and suggest directions for future work in this evolving
field. While previous surveys have focused on ReID using traditional
cameras or object detection with UAVs, this work concentrates on the
unique aspects of person ReID in the context of UAV-based data collection. By providing valuable insights and direction for future research
in the field of UAV-based person ReID, this survey aims to contribute
to the advancement of this promising technology and its applications
in surveillance and public safety.
1.1. Organization of survey
In the subsequent sections of this survey, Section 2 provides a
chronological overview of the developments in UAV-based person ReID,
tracing the progression from traditional ground-based systems to the
emerging domain of UAV-based person ReID while focusing on the
unique challenges introduced by UAV-based surveillance. Section 3
emphasizes the common challenges faced by UAV-based person ReID
systems. Section 4 follows, which discusses the particular and distinctive difficulties related to UAV-based person ReID. Section 5 presents
a comprehensive review of the available datasets and benchmarks,
offering a critical evaluation of their contributions to the field. Section 6
explores the methodologies and approaches applied in UAV-based person ReID, spanning both traditional techniques and state-of-the-art

• The available datasets for UAV-based person ReID
• The methodologies applied in this domain
2

Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261

Fig. 2. Statistics of articles published on person ReID (up to date). *Note — ReID and UAV-based person ReID represent total number of references used in this paper.

Fig. 3. Developments in UAV-based person ReID from early 2000s.

deep learning models. Section 7 presents the results of the latest UAVbased person ReID experiments, highlighting key findings. Finally,
Sections 8, 9, and 10 discuss the broader research challenges and
provide recommendations for future research directions, proposing
strategies to address current limitations and enhance the performance
and scalability of UAV-based person ReID. This survey aims to offer
a thorough and insightful foundation for further advancements in this
rapidly evolving and promising area of computer vision.

attempt to match these features across different camera views recognizes the same person. However, these systems face several challenges,
including changes in illumination (as shown in Fig. 5), occlusion,
temporal variations in individual appearances, and discrepancies in
camera calibrations (as demonstrated in Fig. 5), which often impede
their performance (Singh et al., 2022). Moreover, the inherent spatial
constraints of static cameras further restrict their surveillance area (as
shown in Fig. 5) (Layne et al., 2014). Despite significant advancements
in addressing these challenges (Wang et al., 2020), the limitations
of fixed cameras have motivated researchers to explore alternative
solutions, such as UAVs (Layne et al., 2014). The interest in ReIDbased research, particularly UAV-based person ReID, has increased
significantly in recent years, as shown in Fig. 2, which highlights the
distribution of research publications across different digital libraries
used for this paper. We provide a list of all relevant research in Table 1
summarizing the key datasets used in UAV-based person ReID before
discussing them individually in this survey.
Early 2000s: The early 2000s saw the widespread adoption of UAVs
for military applications (Clarke, 2014).Their surveillance potential was
evident, but their specific application to tasks such as person ReID was
still in its infancy. During this period, most research focused on UAV
functionalities, such as flight dynamics (Braun et al., 2012).

2. Chronological developments in UAV-based person ReID
The development of person ReID using UAVs can be traced back to
the emergence of advanced UAV technology in the late 20th century,
as illustrated in Fig. 3. Improvements in sensor size, battery life,
and wireless communication have enabled UAVs in various domains,
including surveillance and monitoring (Muchiri and Kimathi, 2022).
Prior works have used static and ground-based camera systems like
CCTV to recognize individuals across various views (Zhang et al., 2024;
Zheng et al., 2015). These systems employ various methods to extract
features of individuals, such as their clothing (Li et al., 2024; Wan
et al., 2020) or gait analysis (Ye et al., 2024; Liu et al., 2015) and then
3

Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261

Table 1
Summary of UAV-based person ReID datasets and other relevant UAV-based datasets.
Dataset

Approach

Key Outcome

MRP (Layne
et al., 2014)

Simple nearest-neighbor matching

Need for generalizable ReID models; Unsupervised
domain adaptation for mobile ReID

DRHIT01
(Grigorev et al.,
2019)

Multi-branch deep learning; channel group
learning; triplet and L-GM losses

75.5% rank-1 accuracy, 72.7% mAP on DRHIT01;
Importance of transfer learning

P-DESTRE
(Kumar et al.,
2021)a

Novel UAV-based dataset for pedestrian detection,
tracking, and ReID

New challenges for ReID from UAV data; Short-term ReID
methods robust to UAV data

PRAI-1581
(Zhang et al.,
2021)b

Subspace pooling method for person ReID in aerial
imagery

State-of-the-art performance on PRAI-1581; Low accuracy
indicates challenging nature

AG-ReID
(Nguyen et al.,
2024)c

Three-stream Explainable Elevated-View Attention
(EP+EVA) network

80.72% mAP, 88.77% Rank-1 in aerial-to-CCTV; Effective
for aerial-ground person ReID

AHU-UAV
(Zhang et al.,
2021b)d

Fed-UAV: edge-based federated learning with
knowledge distillation

79% Rank-1 with FL vs. 65% local on AHU-UAV;
Reduced model size and inference time

UAV-Human (Li
et al., 2021b)e

Large-scale benchmark dataset for human behavior
understanding

Comprehensive annotated UAV videos/images; Enables
various behavior understanding tasks

UAV-Based Person ReID Datasets

Other UAV-Based Relevant Datasets
BRIAR (Cornett
et al., 2023)

Large-scale, multi-modal biometric dataset with
long-range, high-angle, and UAV imagery

350,000+ images, 1,300+ hours of video; Supports
person recognition in extreme conditions

Manipal-UAV
(Akshatha et al.,
2023)f

Large-scale UAV dataset for small-person detection

DDOD detector achieved 70.3% AP50; Challenges of
small person detection in aerial images

UAV123 (Nousi
et al., 2020)

Long-term tracking framework with object
re-detection and ReID

State-of-the-art accuracy on multiple benchmarks;
Maintains real-time speed

D-PTUAC
(Alansari et al.,
2024)g

Novel dataset for drone-based single-object
tracking in uniform crowds

Reveals gaps in state-of-the-art trackers; Challenges in
tracking uniform objects in UAV-based surveillance

a

http://p-destre.di.ubi.pt/
https://github.com/biyisi/UAV-AHU?tab=readme-ov-file
c
https://drive.google.com/drive/folders/16r7G_CuUqfWG6_UCT7goIGRMqJird6vK
d
https://drive.google.com/file/d/1ws1memBDz9viROf6_hLmDfeQLw35FITO/view?usp=sharing
e
https://sutdcv.github.io/uav-human-web/
f https://docs.google.com/forms/d/e/1FAIpQLScGsutUkmxbZIaiJxsknezz-rd-OuhUfFZRML6RInolf9UlxA/viewform
g https://github.com/HamadYA/D-PTUAC
b

Mid-2000s to Early 2010s: From the mid-2000s to the early 2010s,
as UAVs became more accessible, their potential for civilian applications, including computer vision tasks, began to be explored. However, person ReID research primarily focused on traditional camera
systems (Scaramuzza et al., 2014).
Late 2010s and 2015 : Researchers began to explore the feasibility of using drones for collecting ReID datasets in 2015 and the
late 2010s. Layne et al. (2014), and other researchers have investigated the challenges and opportunities presented by UAV-based person
ReID (Barekatain et al., 2017). Integrating deep learning with UAV
technology marked a significant shift in the field. Creating and utilizing
datasets specifically designed for drone-based ReID became a central
research focus (Barekatain et al., 2017).
2020 and Beyond: Since 2020, there has been an exponential
increase in interest in UAV-based person ReID (Moritz et al., 2021).
Advancements in Convolutional Neural Networks (CNN) (Habib and
Qureshi, 2022) and their application in UAV-based systems have become more prominent. Research has begun to focus on innovative
solutions to bridge the gap between traditional and UAV-based person
ReID systems (Zhang et al., 2021).

2021), has become integral to Person ReID systems. Fig. 4 illustrates
the interconnected subfields of AI that contribute to ReID technology.
UAV-based person ReID exemplifies the convergence of multiple AI
domains:
1. Robotics: Utilizes UAVs to capture’s images for ReID (Vrontis
et al., 2023; Rezwan and Choi, 2022).
2. Computer Vision: Focuses on interpreting visual data through
techniques such as:
• Object detection (Zou et al., 2023)
• Face recognition (Adjabi et al., 2020)
• Motion analysis (Poppe, 2007)
• Face Detection (Kumar et al., 2019)
• Scene Understanding (Zhou et al., 2016)
These components are crucial for detecting and recognizing
individuals in UAV-based person ReID systems (Ye et al., 2022).
3. Deep Learning (DL): Employs sophisticated neural network
architectures to tackle complex ReID tasks (Dong et al., 2021),
including:
• Convolutional Neural Networks (CNNs) (Li et al., 2021a)
• Recurrent Neural Networks (RNNs) (Sherstinsky, 2020)
• Generative Adversarial Networks (GANs) (Aggarwal et al.,
2021)
• Transformer models (Han et al., 2021)
• Diffusion Models (Croitoru et al., 2023)

2.1. Artificial intelligence and its role in person ReID
Artificial Intelligence (AI), first conceptualized in 1956 as ‘‘the
science and engineering of creating intelligent machines’’ (Ng et al.,
4

Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261

Fig. 4. UAV-based person ReID relation with AI and computer vision.
Fig. 5. Illustration of common challenges seen in UAV-based ReID datasets: A-blur, Bdynamic background, C-illumination changes, D-low resolution, E-viewpoint variation,
F-weather challenges (Li et al., 2021b; Zhang et al., 2021).

4. Machine Learning: Applies either supervised or unsupervised
learning approaches, depending on the nature of the available
image data (e.g., labeled or unlabeled) (Mahesh, 2020). Supervised learning methods, such as triplet loss models, depend
on labeled datasets that group images of the same person together (Hastie et al., 2009). In contrast, unsupervised learning
approaches (Naeem et al., 2023), like clustering or domain adaptation, do not require labeled data, instead leveraging patterns
in the data to group individuals or adapt to new domains.

from UAV-based footage, in particular when combined with factors like
distance, occlusions, and environment conditions (Singh et al., 2022).
Further, unforeseen positional changes, background clutter, and occlusions frequently result in low-resolution images captured by public
surveillance cameras. These factors have a major effect on the accuracy
of the person ReID (Jiao et al., 2018; Cheng et al., 2020).
3.2. Illumination changes

This integration of AI subfields enables UAV-based person ReID systems
to effectively capture, process, and analyze visual data for accurate
person identification and tracking.

UAVs operate in different environments and times of the day,
which makes them exposed to a wide range of illumination conditions.
Changes in illumination can significantly affect an image, making it
challenging to maintain consistent identification across different scenarios (Fu et al., 2022) as shown in Fig. 5. Furthermore, the aerial perspective of UAVs can affect the reflecting characteristics of materials,
which can result in unpredictable lighting effects.

3. Common challenges in ground-based and UAV-based person
ReID
Ground-based and UAV-based person ReID share several inherent common challenges across different setups. These include lowresolution images, illumination changes, viewpoint variations, realtime processing, and privacy concerns, as discussed further. We first
discuss the common challenges across both ReID and UAV-based person
ReID in this section and supplement with the specific problems in
UAV-based person ReID.

3.3. Viewpoint variations
ReID for UAVs poses different challenges because of variations in
viewpoint, with changes in UAV altitude, angle, and lens distortion
impacting the clarity and consistency of pedestrian images (Xu et al.,
2023). UAV-based person ReID systems face difficulties in matching
an individual’s appearance across varying perspectives, as individuals
can look substantially different when captured from multiple angles
and altitudes. Traditional ReID methods struggle in this environment,
necessitating the development of advanced algorithms tailored to handle such significant viewpoint-induced appearance changes (Alanezi
et al., 2022; Moritz et al., 2021). While recent advancements have improved supervised ReID algorithms, challenges persist specifically from
viewpoint variations, significantly affecting ReID accuracy (Zhu et al.,
2020a). The substantial difference in the appearance of individuals due
to viewpoint changes can increase the visual distance within inter-class
comparisons, complicating matching processes and impacting overall
ReID performance (Zhao et al., 2022a).

3.1. Low-resolution images
Low-resolution images as shown in Fig. 5 pose a significant challenge for UAV imaging systems. Challenges such as weak pedestrian
appearance features and large resolution variations caused by factors
such as inconsistencies in the shooting height of the UAV, changes in
the angle of the UAV camera, and wide-angle distortion (Xu et al.,
2023). The representation of people is never complete, as (Zheng et al.,
2016b) noted in the context of low-resolution images. The lack of pixels
reduces the ability to differentiate between distinct persons, hence complicating the person ReID process, and such limitations are significant.
UAVs, which operate at changing altitudes, usually capture images and
videos with insufficient resolution to detect specific features. This limitation poses significant challenges in accurately re-identifying persons
5

Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261

3.4. Real time processing

4.1. Dynamic backgrounds

The importance of real-time processing for person ReID in UAVs to
explore areas of interest and the ability to quickly identify and track
is critical, especially in situations requiring immediate action, such
as security risks and rescue operations. However, real-time processing
has its own challenges (Rieke et al., 2012). Person-ReID involves deep
learning models and complex feature extraction techniques, which
require high computational processing resources (Gaikwad and Karmakar, 2022), and there is a trade-off between speed and accuracy.
Real-time data processing needs quick algorithms and efficient data
pipelines that are capable of handling high-throughput video inputs
without bottlenecks (Rieke et al., 2012).

When UAVs fly over various terrains and landscapes, they capture
images and videos that are different from those captured by traditional
ground-based cameras. The dynamic backgrounds faced by UAVs are
one of the most pronounced challenges (Organisciak et al., 2021).
Dynamic background includes moving trees, traffic, and other objects. The altitude of UAVs plus their mobility means the background
is constantly changing. Even with stabilization technology in modern
UAVs, slight drift can result in motion blur (Han et al., 2018). Dynamic
backgrounds in UAV-based person ReID can significantly affect performance by introducing visual noise and distractions, making it harder for
the model to focus on the target person. Constantly changing scenery,
moving objects, and varying angles can lead to misidentification or
failure to track individuals consistently (Ye et al., 2022). Fig. 5 (B)
illustrates this challenge by showing how background elements, such
as trees, cause challenges in distinguishing the target person from the
environment.

3.5. Other challenges
In addition to the common challenges mentioned above, other
relevant challenges common to both ReID and UAV-based ReID are as
follows

4.2. Motion blur
3.5.1. Dataset
The progress of person ReID using UAVs has been hindered by the
lack of UAV-captured datasets and the specific requirements associated
with these projects. As a result, relatively few research studies have
been published focusing on ReID of UAV-based person ReID (Moritz
et al., 2021). The datasets used in UAV-based person ReID differ from
those applied in regular ReID due to unique image/video collection circumstances. Each ReID dataset for UAV-based person ReID is collected
under different conditions, considering factors such as the equipment
used, the altitude of the UAV, the type of camera installed on the
UAV, the resolution, frames, and methods used to detect pedestrians.
These variations in the characteristics of the dataset pose significant
challenges for researchers working on UAV-based person ReID.
Matching individuals in aerial images captured by UAVs is considerably more difficult than matching individuals in regular surveillance
scenarios with fixed camera installations (Zhang et al., 2021; Mueller
et al., 2016). This increased difficulty can be attributed to several
factors, including variations in viewpoint, scale, and occlusion, which
are more pronounced in aerial images compared to ground images.
Specialized datasets and increased availability of public datasets are,
therefore, essential for furthering research in this field.

Motion blur occurring due to the motion of UAV and changing
backgrounds has a substantial influence on person ReID as it distorts
vital details necessary for accurate recognition (Koo et al., 2020).
Research demonstrates that blurring of motion has an impact on critical
features used for person ReID, which presents difficulties in obtaining
perfect recognition (Gunawan et al., 2018). This challenge is illustrated
in Fig. 5 (A), where the effects of motion blur on UAV-based person
ReID are highlighted.
4.3. Battery life and operational time
Despite the significant potential of deploying UAVs with high computational capabilities, the limited battery life remains a primary concern (Hao et al., 2024). Most battery-powered UAVs are typically rated
for less than one hour of flight time (Qin et al., 2020); however, in
practical applications, the actual flight duration is often considerably
shorter than the manufacturer’s specifications. This limitation on flight
time presents a substantial challenge to the use of UAVs in the biometric industry, as it restricts the volume of data that can be collected
during a single flight (Siddiqi et al., 2022).
5. Datasets and benchmarks in UAV-based person ReID

3.5.2. Privacy concerns
Maintaining a person’s privacy is a significant challenge in this
area. UAVs can serve a wide range of applications, but their uses
raises issues about public security and privacy (Alipour-Fanid et al.,
2020). The compromise in privacy can occur in several ways, including
the recognition and location of the individuals in UAV images and
monitoring the behavior of individuals in the private space (Gevaert
et al., 2018). Due to advancements in UAVs and camera technology,
the level of intrusiveness exceeds normal surveillance technologies and
raises serious privacy concerns (Cawthorne and Devos, 2020). Privacy
concerns are a common challenge in both ground-based and UAVbased ReID systems. However, UAVs can intrude into private spaces
and capture wide areas, amplifying these concerns compared to fixed
ground-based systems.

To facilitate research on person ReID using UAV, several benchmark
datasets have been developed. These datasets, as listed in Table 2 serve
as the foundation for training, validating, and testing models, enabling
researchers to evaluate the performance of their algorithms and compare them with existing state-of-the-art methods. The availability of
high-quality datasets is essential for the development and advancement
of person ReID techniques in aerial images (Correa Martins et al.,
2023).
5.1. UAV-based person ReID datasets
Fig. 6 presents an overview of all publicly accessible ReID datasets,
including UAV-based person ReIDs from 2015 onward. Additionally,
Table 2 provides a comprehensive summary of the existing datasets and
highlights their key characteristics for understanding the performance
limitations.

4. Unique challenges in UAV-based person ReID
UAV-based person ReID systems face another distinct set of challenges due to the dynamic nature of UAV domains. These unique
challenges include dynamic backgrounds, motion blur, battery life,
and operational time, as discussed below. As shown in Fig. 1, which
highlights the unique challenges in UAV-based person ReID.

5.1.1. MRP dataset: Characteristics and initial challenges
Mobile ReID Platform (MRP) is an introductory study on ReID using
drones. It introduced the first ReID dataset based on UAV-captured
6

Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261

place just before noon, a time when many people were present. The
individuals have distinct postures and attire, with some overlapping
due to the crowd (Zhang et al., 2021b).
5.1.5. PRAI-1581 dataset: Large-scale aerial ReID benchmark
Zhang et al. (2021) highlights the fact that well-studied datasets
like Market-1501 (Zheng et al., 2015) and CUHK-03 (Li et al., 2014)
are public datasets recorded with fixed security cameras. UAV person
ReID research has received less interest, which is likely due to the lack
of a large-scale publicly available UAV person ReID dataset. Person
ReID in Aerial Images (PRAI-1581) is a large-scale aerial person ReID
dataset that contains 39,461 photos and 1581 person identifiers. Two
DJI consumer UAVs flew at an altitude of 20 to 60 meters to capture the
dataset’s images, which covered the majority of real-world UAV surveillance scenarios. Two UAVs were employed to film at two adjacent sites
with non-overlapping visible areas. PRAI-1581 comprises data from
two flights that took place in both unrestricted and heavily populated
outdoor areas. In order to obtain enough films with a variety of angles
and backgrounds, hovering, cruising, and rotating sports models were
used in the UAV control procedure.
5.1.6. AG-ReId dataset: Mixed aerial and ground scenarios
The AG-ReID dataset is a comprehensive benchmark designed for
person ReID in mixed aerial and ground scenarios. As an extension of
the AG-ReID.v1 dataset (Nguyen et al., 2023a), it provides a large-scale,
diverse, and publicly accessible resource for ReID research, reflecting a variety of real-world surveillance and monitoring applications.
The dataset comprises 100,502 images of 1,615 unique individuals,
each annotated with matching IDs and 15 soft biometric attributes
such as age, gender, and clothing style. Images were collected from
various perspectives using a UAV at altitudes ranging from 15 to 45
m, stationary CCTV, and smart glasses-integrated camera (wearable),
offering a rich variety of intra-identity variations and realistic challenges, including occlusions, blur, resolution variations, and diverse
poses and lighting conditions. Methodically collected over 5 months in
20 sessions (Nguyen et al., 2024).

Fig. 6. UAV-based and ground-based datasets.

images, and the unique challenges of UAV ReID were highlighted.
The process of feature extraction in MRP entails the utilization of
deep learning techniques, specifically CNNs (Matsukawa and Suzuki,
2016), to extract distinctive characteristics from videos or images.
These features capture unique characteristics of individuals that aid in
matching them across different camera views (Layne et al., 2014).
5.1.2. DRHIT01 dataset: High-resolution UAV imaging for ReID
DRHIT01 dataset was collected around the university campus; a
typical remote-controlled quadcopter was utilized. The UAV was flown
at a height of around 25 meters and was equipped with a highdefinition camera capable of recording footage at a resolution of 1920
𝑥 1080 pixels at 30 frames per second. There are around 5000 frames
in each video segment. In the collected videos, a deep CNN was used
to recognize pedestrians. Special annotation software was built to make
the annotation process easier. A total of 101 pedestrian IDs, each person
having around 459 photos (Grigorev et al., 2019).

5.1.7. UAV-Human dataset: Comprehensive action and pose recognition
UAV-Human dataset contains 67,428 annotated video sequences
of 119 subjects for action recognition, 22,476 annotated frames for
pose estimation, 41,290 annotated frames of 1,144 identities for person
ReID, and 22,263 annotated frames for attribute recognition. The data
is gathered using a DJI Matrice 100 platform. UAV-Human is one of
the biggest, most challenging, and most complete UAV datasets for
analyzing human action, position, and behavior (Li et al., 2021b).

5.1.3. P-DESTRE dataset: Multi-session collection for real-world scenarios
Pedestrian analysis research has mainly relied on datasets that
provide data with limited time gaps between successive observations
of each ID, often within a single day. The joint research to collect
the P-DESTRE dataset, a multi-session collection of video shots in busy
outdoor spaces using three (DJI Phantom 4) drones piloted by human
operators, flew over various settings at both institutions’ camps, collecting data that simulated everyday surveillance conditions. Furthermore,
skilled individuals have thoroughly annotated the P-DESTRE set at the
frame level, offering four categories of meta-data: bounding boxes, IDs,
soft biometric labels, and head pose. The 269 participants in the dataset
willingly volunteered and were instructed to ignore the UAVs, which
flew at heights ranging from 5.5 to 6.7 meters and with camera pitch
angles ranging from 45 to 90 degrees (Kumar et al., 2021).

5.2. Other relevant UAV-based person ReID datasets
In addition to commonly used UAV datasets, as discussed in the
previous section, we also note some other datasets that can be relevant
for training models for UAV-based person ReID.
5.2.1. BRIAR dataset: Biometric recognition in challenging aerial conditions
The Biometric Recognition and Identification at Altitude and Range
(BRIAR) dataset is a unique collection designed to advance state-ofthe-art face and whole-body recognition under challenging, real-world
conditions. The dataset includes over 350,000 images and more than
1,300 h of video footage captured from a diverse set of cameras and
sensors, including commercial surveillance cameras, specialized longrange research cameras, and various UAVs and fixed-wing aircraft.
One of the most novel aspects of the BRIAR dataset is the inclusion
of imagery collected from UAVs and fixed-wing platforms at a wide
range of altitudes and distances. The aerial domains, ranging from
small quadcopters to Group 2 fixed-wing aircraft, were selected to
represent operationally relevant scenarios encountered by military and

5.1.4. AHU-UAV dataset: Campus-based collection with varying lighting
conditions
Two DJI UAVs equipped with 4K Full HD video cameras were used
to take photos of 212 individuals on the campus of Anhui University
in Hefei, China. A total of 424 images were taken from two different
perspectives. The dataset was collected outside the campus canteen,
where lighting conditions affected the images, as the collection took
7

Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261

Table 2
Characteristics of UAV ReID datasets.
UAV-Based Person ReID
Dataset Name

Year

Identities

Images

Resolution

UAV Type

Altitudes (M)

Annotation

MRP (Layne
et al., 2014)

2015

51

51,922

640 × 360

Standard
Quadrocopter Drone

–

Bounding Box, IDs

DRHIT01
(Grigorev
et al., 2019)

2019

101

46,000

1920 × 1080

Standard
Quadrocopter Drone

25

Pedestrians manually labeled
and extracted using special
annotation software

AG-ReID
(Nguyen et al.,
2024)

2024

1,615

100,502

3840 × 2160

DJI M600

15–45

YOLO detector, StrongSORT,
Manual attribute annotation

P-DESTRE
(Kumar et al.,
2021)

2021

269

128,000

3840 × 2160

DJI Phantom 4

5.5–6.7

Bounding Boxes, IDs, Soft
Biometrics, Head Pose

AHU-UAV
(Zhang et al.,
2021b)

2021

212

424

3840 × 2160

DJI

–

Bounding Box, Manual
Identity Labeling

PRAI-1581
(Zhang et al.,
2021)

2021

1,581

39,461

2560 × 1440

DJI

20–60

Bounding Box, Manual
Identity Labeling

UAV-Human
(Li et al.,
2021b)

2021

1,144

41,290 (for
ReID)

1920 × 1080
– 640 × 480

DJI Matrice 100

2–8

Bounding Boxes, IDs,
Attributes

Other Relevant UAV-Based Person ReID
UAV123 (Nousi
et al., 2020)

2020

-

110,000

High
Resolution

–

–

Bounding Box

BRIAR (Cornett
et al., 2023)

2023

1055

350,000 images
– 1,300h video

4k – 10
pixels

Skydio X2, Autel
Evo II Dual, Parrot
Anafi, Parrot Anafi
USA, FLIR R80D,
Aerovironment
Puma, The Hybrid
Project SuperVolo

15–400

Automatic face and whole
body bounding boxes, Manual
verification and gap filling

Manipal-UAV
(Akshatha
et al., 2023)

2023

13,462

153,112

1280 × 720

DJI Phantom ,
MAVIC Pro

10–50

CVAT, Manual annotation

D-PTUAC
(Alansari et al.,
2024)

2024

50

121,000

3840 × 2160

DJI Mavic 3 Pro

–

Bounding Box, Manual
annotation

law enforcement agencies during surveillance operations. By capturing
subjects from multiple viewpoints and at distances up to 1,000 m, the
BRIAR dataset enables research into whole-body and gait recognition
for person ReID under challenging conditions (Cornett et al., 2023).

annotate 153,112 person-object instances using rectangular bounding
boxes. The dataset is divided into training, validation, and test sets,
with additional test subsets based on camera altitudes and low-light
conditions. The Manipal-UAV dataset introduces unique challenges for
person detection algorithms, with approximately 70% of objects categorized as tiny or small. This dataset is designed to advance research
in person detection within aerial imagery and the detection of small
objects. (Akshatha et al., 2023). Although the Manipal UAV dataset was
designed and annotated primarily for person detection in aerial images,
it could potentially be adapted and used for person ReID experiments.

5.2.2. UAV123 dataset: High-resolution tracking for aerial object detection
The UAV123 dataset features 123 high-resolution UAV video sequences containing over 110,000 frames, designed to highlight challenges in videos captured from an aerial perspective. The UAV20L
dataset, a subset of concatenated sequences from UAV123, includes 20
UAV videos with frame counts ranging from 1,717 to 5,527, focusing
on long-term surveillance applications. (Nousi et al., 2020).
The primary purpose of UAV123 is to track visual objects. However,
many of the video sequences in the dataset contain people as the
main tracking targets. Therefore, the dataset could be interesting and
challenging to explore person ReID in aerial surveillance scenarios.

5.2.4. D-PTUAC dataset: Tracking in uniform crowds with visual challenges
Drone-Person Tracking in Uniform Appearance Crowd (D-PTUAC) is
a novel dataset designed to evaluate visual object tracking algorithms in
the challenging scenario of using a drone to follow an individual moving through a crowd of similarly dressed people. The dataset fills a gap
in existing tracking benchmarks by focusing on this practical use case,
which is relevant for surveillance and security applications. D-PTUAC
contains 138 carefully selected video sequences totaling over 121,000
frames. The videos were captured in diverse real-world settings using a
DJI Mavic 3 Pro drone. Sequences cover 18 challenging attributes, such
as target scale variations, occlusions, background clutter, and illumination changes. Notably, the dataset emphasizes the unique challenge
of discerning individuals in a crowd when everyone is wearing similar
attire (Alansari et al., 2024).
Although the D-PTUAC dataset is not specifically designed for person ReID, it does contain some elements relevant to ReID research. The

5.2.3. Manipal-UAV dataset: Small person detection in varied environments
The Manipal-UAV Person detection dataset is a newly developed
large-scale dataset specifically designed for person ReID. The dataset
consists of 33 videos captured using two UAVs, DJI Phantom and
MAVIC Pro, flying at 10 to 50 m. The videos are recorded in an
unconstrained environment, covering various locations in and around
the MIT Manipal campus, and include variations in scale, pose, and
illumination. Researchers sample 13,462 frames from the videos and
8


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Uniform Light Transformer for Person Re-identification
under Complex Illumination
XIANG GUO, RUIMIN HU, DONG LIANG ZHU, and MEI WANG, National Engineering
Research Center for Multimedia Software, School of Computer Science, Wuhan University, Wuhan, China
The quality of pedestrian image retrieval is affected by the difference in illumination between images. Previous
studies have used one-to-one lighting transformers to convert images taken under different lighting conditions
into the target lighting. However, can we use a single lighting transformer to convert input images with
various lighting conditions to the target lighting? This question motivated us to investigate the discrepancy
between images generated by a Unified Lighting Transformer and the ground truth images across different
illumination scales. We discovered that the modeling capability of the Unified Lighting Transformer for lowfrequency information decreases gradually with an increase in the number of illuminant variations. Therefore,
based on this insight, we proposed a Discriminative Feature Spectrum Consistency and Low-Frequency
Information Constrained method. This method employs two constraints to enhance the Unified Lighting
Transformer’s modeling capability for low-frequency information. The first mechanism enforces the constraint
at the feature level by comparing the spectrum information between real and fake discriminative features. The
second approach constrains the differences in pedestrian recognition features caused by the differences in
low-frequency information between real and virtual images composed of low-frequency information from
fake images and high-frequency information from authentic images. Our experiments show that our method
outperforms other approaches and performs best across all metrics.
CCS Concepts: • Computing methodologies → Matching; Visual content-based indexing and retrieval;
Additional Key Words and Phrases: Person re-identification, generative adversarial network, illuminationadaptive
ACM Reference format:
Xiang Guo, Ruimin Hu, Dong Liang Zhu, and Mei Wang. 2025. Uniform Light Transformer for Person Reidentification under Complex Illumination. ACM Trans. Multimedia Comput. Commun. Appl. 21, 9, Article 272
(September 2025), 18 pages.
https://doi.org/10.1145/3745786

This work is partially supported by the National Natural Science Foundation of China (Grant Nos. U22A2035, U1736206,
U1803262), and the National Social Science Fund of China (Grant No. 19ZDA113).
Authors’ Contact Information: Xiang Guo, National Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan University, Wuhan, China; e-mail: nanqiaobei@163.com; Ruimin Hu (corresponding author), National
Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan University, Wuhan, China;
e-mail: hrm@whu.edu.cn; Dong Liang Zhu, National Engineering Research Center for Multimedia Software, School of
Computer Science, Wuhan University, Wuhan, China; e-mail: zhudongliang@whu.edu.cn; Mei Wang, National Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan University, Wuhan, China; e-mail:
dr.mei.wang@whu.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/9-ART272
https://doi.org/10.1145/3745786
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.

272:2
1

X. Guo et al.

Introduction

Person Re-identification (re-ID) retrieves all the images in a database containing the same
person as the query image of interest. This task is crucial for filtering out the routes where the
person appears under different cameras, enabling the tracking of suspects, and finding critical
applications such as public security and video surveillance. However, existing person re-ID work
[2, 23] are typically composed of query and gallery images captured under mild variations in
lighting conditions. In real-world scenarios, images captured by cameras are influenced by various
external factors such as time, location, weather, and other environmental conditions, which would
result in significant variations in lighting conditions among the captured images. Severely affects
the retrieval performance of the re-ID methods [26]. As a result, it has become an urgent problem
to mitigate the impact of lighting variations on person recognition effectively.
This article focuses on the task of person re-ID in complicated lighting conditions [20, 22, 27].
This task can be defined as: Given a query image of a pedestrian with unknown lighting conditions,
retrieve all containing the same pedestrian images from a retrieval gallery of images captured
under different lighting conditions. Existing research on person re-ID in complex lighting scenarios
can be classified into two categories: (1) Illumination-Invariant Methods [9, 26], which project the
features of images captured under varying lighting conditions into the same feature space and
extract their share features to serve as pedestrian identification characteristics; (2) IlluminationUnification Methods [27], which convert images under varying lighting conditions into images with
uniform lighting conditions and leverage existing re-ID models to extract features from the images
after standardizing the lighting. Whether based on color invariance [13], color spatial structure
invariance [13], or decoupling lighting information from features [8], the richness of identification
features obtained by illumination-invariant methods is inevitably reduced. Therefore, regarding
the richness of identifiable features, they are perceived as inferior to illumination-unification
methods. The illumination-unification methods still preserve identification features such as color
characteristics and structural features. However, as the number of lighting conditions in the image
database increases, the requirement for multiple illumination transformers also increases, which
becomes inefficient and redundant [27]. Compared to the use of advanced methods [27] that employ
precise one-to-one illumination transformation, is it possible to use a single illumination converter
(which is called Unified Lighting Transformer) to transform images under any lighting conditions
into images under target lighting conditions? Assuming a successful scenario, what is the difference
between the image converted by the Unified Lighting Transformer and the image under real lighting
conditions?
To clarify the above two questions: (1) The feasibility of a Unified Lighting Transformer and the
effectiveness of a Unified Lighting Transformer? Based on [10, 28] inspiration, from a visual and
frequency perspective, we analyze the differences between the generated standard illumination
images by the Unified Lighting Transformer for various illumination variations and the ground truth
illumination images. As shown in Figure 1(a), we observed that the Unified Lighting Transformer
is capable of effectively transforming low (high) illumination images into standard illumination
images. However, its performance in practical retrieval tasks is not as effective. We believe that the
converted standard illumination images and the ground truth standard illumination images exhibit
minimal visual disparities, but they display noticeable distinctions in low-frequency information.
Figure 1(b) shows that the distance between the generated standard light image and the real standard
light image is mainly low frequency. The high-frequency distance is small and stays the same with
the increase of the input scale. In contrast, the low-frequency distance gradually increases with
the rise in the number of scales input to the Unified Lighting Transformer (the number of scales is
3, 5, 7). However, it fluctuates back and forth after reaching a certain level (the number of scales is
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.

Uniform Light Transformer for Person re-ID under Complex Illumination

272:3

Fig. 1. The differences between the images generated by the Unified Light Transformer and the ground truth
images. (a) Differences between generating and real target light images. (b) We employed the Gaussian kernel
function to decouple the low-frequency and high-frequency information of the images. We compared their
low-frequency and high-frequency information modeling capabilities at different illumination scales.

7, 10, 13). Based on this observation, we attribute the potential reasons: modeling capability of the
Unified Lighting Transformer for low-frequency information decreases gradually as the number of
illuminant variations increases, and the deficiency in modeling low-frequency information leads to
the challenge for the Unified Lighting Transformer in effectively performing the conversion between
illumination information.
To address the issue above, it is necessary to enhance the Unified Lighting Transformer’s
ability to model low-frequency information. Therefore, we propose a framework Decoupling
Low-Frequency Information Constraints (DLFC) that utilizes two constraint mechanisms.
The first is based on a discriminative feature spectrum consistency constraint module, while
the second is based on a Low-Frequency Information Constraint Module. The Low-Frequency
Information Constraints method calculates the difference in low-frequency information between
the virtual and real target illumination. It applies a loss at the feature level to strengthen the
Unified Lighting Transformer’s ability to model low-frequency information. In concrete terms, the
Low-Frequency Information Constraints method decomposes generated and authentic images into
low-frequency and high-frequency information using a Gaussian kernel function. It combines the
high-frequency information from real images with the low-frequency information from generated
images to create virtual samples. Then, the pre-trained re-ID model extracts features from real and
virtual images, and knowledge distillation is employed to capture the differences in low-frequency
information at the feature level. Additionally, the Discriminative Feature Spectrum Consistency
module addresses the issue of spectral inconsistency in discriminative features extracted by the
discriminator. This inconsistency may cause the generator to prioritize generating high-frequency
information at the expense of low-frequency information. The module leverages the spectral
difference in discriminative features between “real” based on the generated image’s feature map
transformer and generated discriminative features as supervisory information to compel the
generator to focus more on generating low-frequency information. In short, our framework DLFC
achieves the state-of-the-art performance on two synthetic datasets, and the contributions of this
article can be summarized as follows:
— We find that the modeling capability of the Unified Lighting Transformer for low-frequency
information diminishes gradually with an increasing number of input illuminant variations.
— We propose a novel DLFC model to address the insufficient modeling capability of the Unified
Lighting Transformer for low-frequency information caused by multi-illuminant variations
as input. Specifically, the Discriminative Feature Spectrum Consistency and Low-Frequency
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.

272:4

X. Guo et al.

Information Constraints Modules will be introduced to constrain the Unified Lighting Transformer.
— We validated the efficacy of our proposed method on two datasets and achieved the state-ofthe-art performance compared to existing approaches.
2
2.1

Related Work
Illumination Adaptive Person re-ID

Compared to traditional person re-ID tasks [21, 23] under soft illumination variations, person re-ID
tasks in complex lighting conditions face the interference of illumination noise. On the one hand,
illumination variations cause changes in the color of pedestrian images, making it difficult to rely
on color information as a distinguishing feature. On the other hand, the significant illumination
discrepancies between images result in intra-class distances being much more critical than interclass distances. Researchers have attempted to identify pedestrian recognition features invariant
to illumination changes in such cases. Bouma et al. [1] argued that while color is susceptible to
illumination variations, transforming the color space can reduce the impact of complex lighting.
Through the comparison of clothing type features, particularly local clothing type features, Wang
et al. [20] believed that different image pairs exhibit varying recognition under different lighting conditions. Therefore, the minimum recognition distance based on lighting is sought as the
recognition distance for the image pair. However, these shallow semantic features based on image
content are only suitable for distinguishing pedestrians with significant clothing color and type
differences [24]. They do not apply to cases where the clothing is similar between pedestrians.
To enable effective pedestrian recognition, deep semantic representation features are crucial.
Based on the Retinex theory [9], Huang et al. decoupled the essential features from pedestrian
images, independent of lighting conditions, and combined them with lighting-independent features
for feature matching [26]. While this approach can separate the lighting information from the
pedestrian features and eliminate the impact of lighting changes, it may also reduce the discriminative features of pedestrians. Zhang et al. [27] employ lighting transformation techniques to
transform images taken under different lighting conditions into images with the same lighting. This
allows traditional pedestrian recognition methods to extract more identifiable features. However,
having specific transformers for each lighting condition can take time and effort. Therefore, this
article proposes a novel Unified Lighting Transformer that can transform images from any lighting
condition to the target lighting condition using a single transformer.
2.2

Frequency Domain Analysis on Generative Adversarial Network (GAN)

Recent research has been focusing on analyzing GAN models from a frequency perspective. In
style transfer tasks using GANs, all GAN models are found to achieve low errors in the lowfrequency range but struggle with generating high-frequency details, according to researchers
[28]. Similarly, researchers have observed differences between generated and authentic images
in the frequency domain in image reconstruction and generation tasks and proposed a range of
solutions. For instance, some studies [10] have introduced frequency domain losses based on focal
loss, which focuses the generator on more challenging frequency components by reducing the
weight of relatively simple frequency components. This approach aims to generate higher-quality
images. In addition, another study has proposed [25], a wavelet-based image restoration network, to
address frequency deviations in image restoration tasks. Moreover, researchers [18] have found that
discriminators better perceive high-frequency information than low-frequency information. This
inconsistency in frequency spectra can impact the quality of generated images. Similarly, Li et al.
[14] suggest that discriminators directly utilize high-frequency information for data discrimination.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.

Uniform Light Transformer for Person re-ID under Complex Illumination

272:5

Fig. 2. Overview of the proposed DLFC framework. The training loss of the Unified Lighting Transformer
includes the generator loss of the GAN and the losses of the proposed Discriminative Feature Spectrum
Consistency and Low-Frequency Information Constraints modules.

This leads to generators focusing on generating high-frequency information while neglecting
low-frequency information. The authors propose adding high-frequency information from real
images to the generated images to guide the discriminator’s focus on low-frequency components.
The ability of generators to capture high-frequency and low-frequency information in images
varies across different tasks. Hence, from a frequency perspective, we analyze the impact of
increasing multi-scale lighting variations on a unified lighting model. It reveals that input lighting
differences increase, and there is a growing gap in low-frequency information between the generated
target lighting images by the unified lighting transformer and the actual target lighting images.
Thus, there is a need to constrain the modeling ability of the unified lighting transformer for
low-frequency information in images.
3
3.1

Proposed Method
Framework Overview

As shown in Figure 2, we propose a DLFC framework that can transform pedestrian images under
arbitrary lighting conditions into images under the same lighting conditions. The DLFC consists of
four modules: the Unified Lighting Transformer, Discriminator, Discriminative Feature Spectrum
Consistency, and Low-Frequency Information Constraints. Unified Lighting Transformer: used
to convert pedestrian pictures under arbitrary lighting conditions to uniform lighting conditions;
Discriminator: used to distinguish whether the generated lighting images are real or fake; Discriminative Feature Spectrum Consistency: used to address the issue of spectral inconsistency
in discriminative features extracted by the discriminator, which leads to a stronger emphasis on
high-frequency information and neglect of low-frequency information in generation, resulting
in weak modeling of low-frequency information. Moreover, Low-Frequency Information Constraints are used to reduce the disparity between the generated and actual light pictures in terms of
low-frequency information. During training, the Discriminator, Discriminative Feature Spectrum
Consistency, and Low-Frequency Information Constraints modules work together to constrain the
Unified Lighting Transformer, enabling it to generate high-quality unified illumination images. In
the following sections, we will introduce some definitions and then explain each module based on
the training procedure.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.

272:6
3.2

X. Guo et al.
Problem Definition and Notation

Let X𝑠 = {𝑥𝑖𝑠 }𝑛 be a set of 𝑛 standard lighting training images, with corresponding identities labels
𝑌𝑠 = {𝑦𝑖𝑠 }𝑛 , where 𝑥𝑖𝑠 ∈ 𝑅 𝐻 ×𝑊 ×3 is the 𝑖 𝑡ℎ standard lighting image and 𝑦𝑖𝑠 ⊂ {0, 1} indicate the
one-hot vector of the ground truth label. We aim to enhance the model’s capability to address the
problem of significant illumination variations between pedestrian images. We generate images
under different lighting conditions by applying gamma correction to the images under standard
lighting. These synthesized images constitute a diverse lighting training dataset called X𝑐 = {𝑥𝑖𝑐 }𝑛 .
Based on this, we can obtain a sufficient number of image pairs {𝑥𝑖𝑠 , 𝑥𝑖𝑐 } for training a unified
lighting transformer, enabling it to learn the transformation patterns and effectively convert any
input image 𝑥 𝑐 under arbitrary lighting conditions into the target lighting condition 𝑥 𝑠 .
In complex lighting scenarios, we assume that the lighting conditions in the original dataset are
considered the standard lighting condition and experiment following [27] setting, where the query
images are in the standard lighting condition. The retrieval gallery is composed of images captured
under different lighting conditions.
3.3

Discriminator

The Unified Light Transformer aims to transform images captured under arbitrary lighting into
images under standard illumination conditions. Therefore, a series of image pairs {𝑥𝑖𝑐 , 𝑥𝑖𝑠 } can be
used to train the Unified Light Transformer. First, we input the image with unknown lighting
𝑔
conditions 𝑥𝑖𝑐 into the Unified Light Transformer 𝐺 to get the generated image 𝑥𝑖 :
𝑔

𝑥𝑖 = 𝐺 (𝑥𝑖𝑐 ).

(1)

Before training the unified Light transformer 𝐺, it is necessary to train a well-performing
𝑔
discriminator 𝐷. The discriminator is trained to distinguish whether the lighting 𝑥𝑖 is standard. To
achieve this, we input both the generated standard lighting images and authentic standard lighting
images into the discriminator, obtaining their respective discrimination features:
𝑑𝑖 = 𝐷 (𝑥𝑖 ),

(2)

𝑑𝑖𝑠 = 𝐷 (𝑥𝑖𝑠 ),

(3)

𝑔

𝑔

where 𝑑𝑖 and 𝑑𝑖𝑠 represent the discriminative features of the generated and real data, respectively.
Following the setup of GAN, we aim for the discriminator to identify the generated images as fake
with distribution features close to 0 while identifying the real images as genuine with discriminator
features close to 1. Therefore, the loss function for the discriminator L𝐷 can be expressed as follows:
𝑔
𝑛
1 Õ (𝐵𝐶𝐸 (𝒅𝑖 , 0) + 𝐵𝐶𝐸 (𝒅𝑖𝑠 , 1))
L𝐷 =
,
(4)
𝑛 𝑖=1
2
𝑔

where 𝐵𝐶𝐸 is Binary Cross Entropy, our Unified Lighting Transformer and Discriminator adopt
a network architecture similar to CycleGAN’s generator and discriminator [33]. Discriminator
𝐷 adopts the patchGAN’s discriminator structure, which outputs a discriminative feature map
instead of a single value. This structure is a foundation for subsequent learning in the Spectrum
Constrained Network (SCN).
3.4

Discriminative Feature Spectrum Consistency Module

After training the discriminator, we train the Spectral Constrained Network in the Discriminative
Feature Spectrum Consistency Module. This module addresses the issue where the discriminator
overly focuses on high-frequency information and neglects low-frequency information, leading the
generator to overemphasize high-frequency details and overlook low-frequency information in its
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.

Uniform Light Transformer for Person re-ID under Complex Illumination

272:7

Fig. 3. The motivation of SCN. We selected and presented the results from five consecutive rounds during
the training process of the unified illumination transformer. The similarity between the feature map and
discriminant features (Tf_d, Ff_d) is higher than between pedestrian recognition features and discriminant
features (Tc_d, Tf_d).

generation. This scenario poses a more significant challenge for the unified illumination translator
as it needs to handle the unified transformation of multi-scale illuminations.
The core of this innovative approach lies in treating the discriminative features and the features
map extracted through the re-ID model as a combination of high-frequency and low-frequency
information. By utilizing a spectral constraint network, the method learns the correct spectral
correlations between the discriminative features of real images and the feature maps of real
images extracted by the pre-trained re-ID model. Subsequently, we attempt to utilize this spectral
constraint network to transform the feature maps extracted from generated images into “real”
discriminative features. This enables a direct measurement of the spectral discrepancies between the
discriminative features of generated images and the “real” discriminative features, then using this
spectral discrepancies as new supervisory information to guide the illumination unified transformer
in generating more accurate image content. This method inspired by the work of cross-resolution
person re-ID [4], but difference is that we explored and utilized the relationship between the feature
maps of images and the discriminative features of the images instead of relationship between the
Pedestrian identification feature of discriminative feature. Because we used the Centered Kernel
Alignment [12] to calculate the similarity between the Pedestrian identification features of the
𝑔
𝑔
real and fake images (𝒇𝑖𝑠 and 𝒇𝑖 ), the feature map of the real and fake images (𝒎𝑠𝑖 and 𝒎𝑖 ), and
𝑠
the discriminative feature of the real image 𝒅𝑖 , respectively, as shown in Figure 3, we find that the
inherent relationship between feature maps and discriminative features is more informative than
between pedestrian identification features and discriminative features, enabling a more effective
transformation of the image’s feature maps into “real” discriminative features.
We use the pre-trained re-ID model 𝐹𝑐 , which is trained under standard lighting conditions, to
extract feature maps of standard lighting images 𝒎𝑠𝑖 :
𝒎𝑠𝑖 = 𝐹𝑐 (𝑥𝑖𝑠 ).

(5)

Next, we input the feature maps 𝒎𝑠𝑖 into the SCN 𝐹𝜃 1 to obtain transformed discriminative
features 𝒅𝑖 = 𝐹𝜃 1 (𝒎𝑠𝑖 ). We aim for the SCN to map the feature maps to the actual discriminative
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.

272:8

X. Guo et al.

Fig. 4. The framework of SCN. SCN computational flow and each layer output feature shape.

features 𝒅𝑖𝑠 . Therefore, the loss 𝐿𝜃 1 of the SCN is represented by:
𝑛

1Õ
L𝜃 1 =
||𝒅𝑖 − 𝒅𝑖𝑠 || 2 .
𝑛 𝑖=1

(6)

To facilitate rapid convergence of the SCN, we designed an Additional Network 𝐹𝜃 2 to bridge
the relationship between the SCN‘s intermediate features 𝒅𝑖𝑚 and the discriminative features 𝒅𝑖𝑠 .
This additional network transforms the input discriminative features 𝒅𝑖𝑠 into intermediate features
𝑝
𝒅𝑖 = 𝐹𝜃 2 (𝒅𝑖𝑠 ). Therefore, the optimization loss of the Additional Network L𝜃 2 can be expressed as
follows:
𝑛
1Õ 𝑝
L𝜃 2 =
||𝒅 − 𝒅𝑖𝑚 || 2 .
(7)
𝑛 𝑖=1 𝑖
During the training process, we jointly train the parameters of the SCN 𝐹𝜃 1 and the Additional
Network 𝐹𝜃 2 . The overall optimization loss L𝜃 is represented by:
L𝜃 = L𝜃 1 + L𝜃 2 .

(8)

The SCN consists of four layers of hidden layers. The detailed computational flow and each layer
output feature shape can be found in Figure 4.
3.5

Low-Frequency Information Constraints Module

The Low-Frequency Information Constraints combine the low-frequency information of generated
standard lighting images with the high-frequency information of authentic lighting images to create
virtual samples. Subsequently, we utilize a pre-trained re-ID model to extract feature classification
logits separately from the virtual samples and authentic lighting images, which can effectively
quantify the impact of the difference in low-frequency information on the feature level. This
approach provides a direct and effective constraint on the Unified Lighting Transformer’s modeling
capability for low-frequency information [10]. Specifically, we employ a Gaussian kernel:
𝐾𝜙 [𝑖, 𝑗] =

1 − 12 ( 𝑖 2 +𝑗2 2 )
𝜎
𝑒
,
2𝜋𝜎 2

(9)

where [𝑖, 𝑗] represents the spatial location within the image, and 𝜎 2 denotes the square of the
Gaussian kernel function. It filters the high-frequency information of the image and retains the
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.


codex
`UAV` 是综述，没有自己的算法模块，但它的“创新构造”是把一个应用域重新界定为独立 ReID 子问题，证据来自数据集稀缺、挑战差异和方法谱系整理。`Uniform Light Transformer` 的方法部分还在第 8 页之后，我继续补读它的低频约束和训练目标。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf' - | awk '/3\\.5[[:space:]]+Low-Frequency Information Constraints Module/{flag=1} /4[[:space:]]+Experimental Settings/{if(flag){print; flag=0}} flag'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf' - | awk '/3\\.6[[:space:]]+Optimization/{flag=1} /4[[:space:]]+Experimental Settings/{if(flag){print; flag=0}} flag'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
codex
刚才的标题匹配没有命中，可能是 `pdftotext` 把标题断行了。我直接读第 8 到 12 页，再按内容定位。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 12 'Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
272:8

X. Guo et al.

Fig. 4. The framework of SCN. SCN computational flow and each layer output feature shape.

features 𝒅𝑖𝑠 . Therefore, the loss 𝐿𝜃 1 of the SCN is represented by:
𝑛

1Õ
L𝜃 1 =
||𝒅𝑖 − 𝒅𝑖𝑠 || 2 .
𝑛 𝑖=1

(6)

To facilitate rapid convergence of the SCN, we designed an Additional Network 𝐹𝜃 2 to bridge
the relationship between the SCN‘s intermediate features 𝒅𝑖𝑚 and the discriminative features 𝒅𝑖𝑠 .
This additional network transforms the input discriminative features 𝒅𝑖𝑠 into intermediate features
𝑝
𝒅𝑖 = 𝐹𝜃 2 (𝒅𝑖𝑠 ). Therefore, the optimization loss of the Additional Network L𝜃 2 can be expressed as
follows:
𝑛
1Õ 𝑝
L𝜃 2 =
||𝒅 − 𝒅𝑖𝑚 || 2 .
(7)
𝑛 𝑖=1 𝑖
During the training process, we jointly train the parameters of the SCN 𝐹𝜃 1 and the Additional
Network 𝐹𝜃 2 . The overall optimization loss L𝜃 is represented by:
L𝜃 = L𝜃 1 + L𝜃 2 .

(8)

The SCN consists of four layers of hidden layers. The detailed computational flow and each layer
output feature shape can be found in Figure 4.
3.5

Low-Frequency Information Constraints Module

The Low-Frequency Information Constraints combine the low-frequency information of generated
standard lighting images with the high-frequency information of authentic lighting images to create
virtual samples. Subsequently, we utilize a pre-trained re-ID model to extract feature classification
logits separately from the virtual samples and authentic lighting images, which can effectively
quantify the impact of the difference in low-frequency information on the feature level. This
approach provides a direct and effective constraint on the Unified Lighting Transformer’s modeling
capability for low-frequency information [10]. Specifically, we employ a Gaussian kernel:
𝐾𝜙 [𝑖, 𝑗] =

1 − 12 ( 𝑖 2 +𝑗2 2 )
𝜎
𝑒
,
2𝜋𝜎 2

(9)

where [𝑖, 𝑗] represents the spatial location within the image, and 𝜎 2 denotes the square of the
Gaussian kernel function. It filters the high-frequency information of the image and retains the
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.

Uniform Light Transformer for Person re-ID under Complex Illumination

272:9

low-frequency information. We obtain the low-frequency information 𝑥𝑖 of the generated standard
lighting image and the low-frequency information 𝑥𝑖𝑙𝑠 of the ground truth standard lighting image:
ÕÕ
𝑙𝑔
𝑔
𝑥𝑖 [𝑖, 𝑗] =
𝑘 [𝑎, 𝑏] · 𝑥𝑖 [𝑖 + 𝑎, 𝑗 + 𝑏],
(10)
𝑙𝑔

𝑎

𝑥𝑖𝑙𝑠 [𝑖, 𝑗] =

𝑏

ÕÕ
𝑎

𝑘 [𝑎, 𝑏] · 𝑥𝑖𝑠 [𝑖 + 𝑎, 𝑗 + 𝑏],

(11)

𝑏

(𝑘 −1)
where 𝑎 and 𝑏 denote the indices of the two-dimensional Gaussian kernel, 𝑎, 𝑏 ∈ [− (𝑘 −1)
2 ,
2 ].
ℎ𝑔
To obtain the generated standard lighting image’s high-frequency information 𝑥𝑖 and the actual
standard lighting image’s high-frequency information 𝑥𝑖ℎ𝑠 , we convert the color image into grayscale
and then subtract the low-frequency information:

𝑥𝑖 = 𝑟𝑔𝑏2𝑔𝑟𝑎𝑦 (𝑥𝑖 ) − (𝑟𝑔𝑏2𝑔𝑟𝑎𝑦 (𝑥𝑖 ))𝑖 ,

𝑙𝑔

(12)

𝑥𝑖ℎ𝑠 = 𝑟𝑔𝑏2𝑔𝑟𝑎𝑦 (𝑥𝑖𝑠 ) − (𝑟𝑔𝑏2𝑔𝑟𝑎𝑦 (𝑥𝑖𝑠 ))𝑖𝑙𝑠 ,

(13)

ℎ𝑔

𝑔

𝑔

where the function 𝑟𝑔𝑏2𝑔𝑟𝑎𝑦 converts a color image into a gray-scale image, next, we add the
high-frequency information of the actual standard lighting image to the low-frequency information
of the generated standard lighting image, resulting in the virtual image 𝑥𝑖𝑣 :
𝑙𝑔

𝑥𝑖𝑣 = 𝑥𝑖 + 𝑥𝑖ℎ𝑠 .

(14)

The virtual image and the standard illumination image only differ in terms of low-frequency information. Therefore, we can directly use the kullback–leibler divergence to measure the differences
in low-frequency information that lead to discrepancies in pedestrian identification features. Given
𝑔
the virtual images 𝑥𝑖𝑣 and the real sample 𝑥𝑖 , we input them into the re-ID model 𝐹𝑐 . This yields
𝑣
𝑣
the logits 𝒒𝑖 = 𝑠𝑜 𝑓 𝑡𝑚𝑎𝑥 (𝐹𝑐 (𝑥𝑖 )/T) (T is a hyperparameter coefficient for temperature) for the real
image and the softmax probabilities 𝒑𝑖𝑠 = 𝑠𝑜 𝑓 𝑡𝑚𝑎𝑥 (𝐹𝑐 (𝑥𝑖𝑠 )/T) for the virtual image. Therefore, the
L𝐿𝐹𝐶𝑀 can be expressed as:
𝑛

L𝐿𝐹𝐶𝑀 =

1Õ
𝐾𝐿(𝒒𝑖𝑣 ||𝒑𝑖𝑠 ).
𝑛 𝑖=1

(15)

We utilize the idea of knowledge distillation [3] when calculating the 𝐾𝐿 divergence between the
two distributions. Adding a temperature parameter 𝑇 makes the distributions smoother and more
accessible to learning the distance information between the two distributions.
3.6 Unified Light Transformer
The total loss of the Unified Lighting Transformer consists of three components. The first component
is the loss of the GAN generator L𝐺 [7]:
𝑛

L𝐺 =

1Õ
𝑔
𝑔
(||(𝑥𝑖 , 𝑥𝑖𝑠 || 2 ) + 𝑏𝑐𝑒 (𝒅𝑖 , 1)).
𝑛 𝑖=1

(16)

The second component is the loss of the Discriminative Feature Spectrum Consistency Module. We
𝑔
𝑔
utilize a pre-trained re-ID model and a discriminator to extract the feature map 𝒎𝑖 = 𝐹𝑐 (𝑥𝑖 ) and
𝑔
𝑔
its discriminative feature 𝒅𝑖 = 𝐷 (𝑥𝑖 ) from the generated standard lighting images. Subsequently,
0
𝑔
we use the SCN to transform the feature map into a “real” discriminative feature 𝒅𝑖 = 𝐹𝜃 1 (𝒎𝑖 ). The
spectral distance between the generated and real standard lighting images can be represented as
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.

272:10

X. Guo et al.

Fig. 5. Different gamma adjustment was applied to each channel of the original image to generate images
suitable for various lighting conditions.

0

L𝐷𝐹𝑆𝐶 = ||𝒅𝑖 − 𝒅𝑖 || 2 , and the third component is the LFCM loss L𝐿𝐹𝐶𝑀 . Therefore, the total loss of
the ULT L𝑈 𝐿𝑇 can be expressed as:
𝑔

L𝑈 𝐿𝑇 = L𝐺 + L𝐷𝐹𝑆𝐶 + L𝐿𝐹𝐶𝑀 .

(17)

We need to iteratively train the discriminator, SCN, and the unified lighting transformer until
the unified lighting transformer is converged.
4
4.1

Experiments
Datasets and Evaluation Settings

Datasets. We synthesized complex illumination datasets, namely Market1501-AI and Duke-AI. Based
on the original datasets, i.e., Market1501 [29], which includes images captured by six cameras,
consisting of 12,936 training images with 751 identities and 19,732 images with 750 identities
in the testing set. And DukeMTMC-reID [31] consists of images captured from 8 cameras, with
16,522 training images belonging to 702 different identities and 19,889 testing images representing
the remaining 702 identities. Similar to reference [15], we apply gamma correction to create
illumination-adaptive datasets for training and evaluating our proposed method. Gamma correction
[15] is a nonlinear transfer function widely used in video and image systems. It has also been
extensively employed in references [9, 26] to simulate images with varying illumination scales due
to its ability to approximate the nonlinear relationship between light intensity and image output
values.
In our experiments, following [27], we utilize gamma correction with 13 specific gamma values
(0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2.1, 2.5, 2.9, 3.3) to simulate images under different illumination
conditions, ranging from dim light to glare. These synthesized datasets’ training and query sets
remain the same as the original ones. However, for the gallery part, each image was randomly
selected from 13 lighting scales (0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2.1, 2.5, 2.9, 3.3). A random
gamma adjustment was applied to each channel of the original image to generate images suitable
for different lighting conditions. As shown in Figure 5, the results show the synthesis of different
individuals under varying lighting conditions.
Evaluation Metrics. Following [17], we assess the performance of our algorithm using rank-K
(K = 1, 5, 10) and Mean Average Precision (mAP). The rank-K metric evaluates the percentage of
successful searches where at least one relevant individual is retrieved from the top K outcomes.
The mAP metric allows for the analysis of the model’s capability to predict the sequential order
accurately.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.

Uniform Light Transformer for Person re-ID under Complex Illumination

272:11

Table 1. Overall Performance Comparison on Market1501-AI and Duke-AI
Methods
IDE
PCB
BOT
AGW
Camstyle
DG-Net
CBN
IS-GAN
IUReid
OIDM(AGW)
DLFC(BOT)
DLFC(AGW)

mAP
27.1
33.1
29.9
51.5
37.8
49.0
51.8
45.9
72.1
76.9
72.4
78.1

Market1501-AI
rank-1
rank-5
69.7
84.9
79.1
90.0
70.8
86.8
90.0
95.9
76.0
89.4
83.0
95.1
84.4
94.1
87.4
95.7
89.2
96.1
90.8
96.7
91.0
96.5
93.4
97.2

rank-10
89.3
92.4
90.7
97.4
92.4
96.8
96.2
96.5
97.7
98.1
97.8
98.5

mAP
25.0
36.5
29.6
50.4
38.8
51.8
50.9
48.0
63.2
64.5
64.1
73.5

rank-1
63.2
76.3
69.0
86.0
69.7
83.5
78.2
84.3
81.7
86.7
83.6
89.5

Duke-AI
rank-5
77.5
87.6
83.4
93.3
82.3
92.1
88.4
92.1
90.0
93.8
91.9
94.7

rank-10
82.2
90.5
87.7
95.1
85.9
94.0
91.2
94.4
92.8
95.7
94.5
96.2

Bold numbers indicate the best performance.

Baseline Methods. Our proposed method compares with several advanced methods, including (1)
traditional Re-ID methods: IDE [32], PCB [19], BOT [16], CBN [34], and AGW [23], (2) methodsbased on GAN: Camstyle [32], DG-Net [30], IS-GAN [5], UiReid [27], and OIDM [20].
Implementation Details. We used the Adam optimizer with a batch size of 32 for training. The
Unified Lighting Transformer and discriminator were trained with a learning rate of 2e-4, while
the SCNemployed a learning rate of 1e-3. We trained the models for 100 epochs and updated the
learning rate with a decay rate of 0.1 every 30 epochs. During the re-ID model training, all images
were resized to 256×128×3. To augment the data, we applied random cropping and horizontal
flipping. The learning rate was set to 3e-4, and the Adam optimizer was utilized to update the
learning rate with a decay rate of 0.1 every 30 epochs. The re-ID model was trained for 60 epochs,
but we use the pre-trained AGW model downloaded in [23].
4.2

Performance Evaluation

This section compares our proposed method with several state-of-the-art methods on two synthetic
datasets. As shown in Table 1, we can observe the following: (1) When comparing the direct feature
extraction methods (e.g., IDE, PCB, BOT, AGW) and the GAN-based methods (e.g., Camstyle,
DG-Net, IS-GAN, IUReid), it is found that although powerful feature extraction methods exhibit
certain robustness to illumination, i.e., the stronger the feature extractor, the less affected it is by
illumination, the impact of illumination remains non-negligible. Better recognition performance
can be achieved through simple illumination transformation methods, indicating the necessity
of illumination normalization to mitigate the influence of illumination variations on pedestrian
recognition performance. (2) A visual comparison between BOT and DLFC (BOT) demonstrates
that the Unified Lighting Transformer significantly improves the performance, indicating that
unifying illumination effectively addresses the performance degradation issue in existing person
re-ID methods caused by lighting variations. This result indicates that our unified illumination
transformer shows specific feasibility in performing multi-illumination transformation tasks, which
is capable of converting various illumination inputs into a standardized illumination, thereby
eliminating the influence of illumination discrepancies on the results. (3) Compared to IUReid, the
state-of-the-art illumination normalization method utilizing a more powerful BOT+GeM as the
feature extractor, our method still outperforms it under the unequal conditions of feature extractors;
this result demonstrates that the method we proposed has achieved significant advancements in
effectiveness and potential, reaching the highest level in this field.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.

272:12

X. Guo et al.
Table 2. Ablation Study on Market1501-AI
ULT

Components
DFSC
LFCM

3
3
3
3

3
3

3
3

mAP
51.5
71.2
77.8
76.5
78.1

Market1501-AI
rank-1
rank-5
90.0
95.9
89.6
96.0
93.0
97.3
92.8
97.0
93.4
97.2

rank-10
97.4
97.7
98.6
98.3
98.5

(4) DLFC (BOT) outperforms all other methods on the Market1501 dataset. However, the Duke-AI
dataset falls behind the IS-GAN and AGW methods. Specifically, compared to AGW, we found
a decrease of approximately 2.4% in rank-1, 1.4% in rank-5, and 0.6% in rank-10. The potential
reason is that AGW outperforms BOT regarding feature extraction capability and possesses strong
robustness in extracting pedestrian illumination-invariant features. Therefore, the lower identification performance is not due to the poor quality of standard illumination images transformed by
our Unified Lighting Transformer but rather the inadequate feature extraction capability. Hence,
we employed AGW as the feature extractor, and DLFC(AGW) achieved the best experimental
results on both datasets. Especially on Duke-AI, it further increased rank-1, rank-5, and rank-10
by approximately 3.5%, 1.4%, and 0.1%, respectively, once again proving the effectiveness of our
method.
4.3

Ablation Study

To demonstrate the effectiveness of the modules in our proposed method, we conducted a series of
ablation experiments on Market1501-AI. The rank-1, rank-5, and rank-10 mAP accuracies (%) are
reported in Table 2.
— First, we compared the performance of ULT + AGW with AGW alone and found that simple
ULT struggles to transform images under arbitrary lighting conditions to a unified illumination,
resulting in inferior performance compared to AGW, the rank-1 accuracy decreased by 0.4%;
this suggests that using a unified illumination transformer alone is insufficient for effective
illumination conversion.
— Next, we compared AGW + DFSC and AGW + LFCM modules with AGW alone and observed
that both modules contribute to improved retrieval performance, with a rank-1 accuracy
improvement of approximately 3% and 2.8%, respectively; this demonstrates that our proposed method is effective in enhancing the low-frequency modeling capability of the unified
illumination transformer, enabling the capture of differences in low-frequency information to
generate high-quality illumination images.
— However, AGW + DFSC outperformed AGW + LFCM by an additional 0.2% in rank-1 accuracy,
indicating greater effectiveness; this suggests that improving the discriminator’s ability to
capture low-frequency information may be more crucial for feature discrimination than
capturing low-frequency information itself.
— Moreover, when these two modules were combined to form AGW + DFSC+ LFCM, further
enhancement, DLFC (AGW + DFSC+LFCM) achieved a rank-1 accuracy improvement of 0.4%
compared to AGW + DFSC alone, this indicates that the ability to model low-frequency information from different perspectives is essential as a single angle may have limited capability
in low-frequency information modeling.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 9, Article 272. Publication date: September 2025.


codex
`UAV` 综述的前 8 页只覆盖到数据集部分，后面的“methodologies and approaches”没有完全出现。为了不把综述误拆成摘要，我会继续定位它的 Section 6 方法谱系和后续挑战建议。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdfinfo 'UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf' | sed -n '1,40p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Custom Metadata: no
Metadata Stream: yes
Tagged:          yes
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           19
Encrypted:       no
Page size:       595.276 x 793.701 pts
Page rot:        0
File size:       3198767 bytes
Optimized:       no
PDF version:     1.7

exec
/bin/zsh -lc "pdftotext 'UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf' - | rg -n '"'^6'"\\.|methodolog|approach|7\\.|8\\.|Future|challenges'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
8:UAV-based person re-identification: A survey of UAV datasets, approaches,
9:and challenges
41:UAV-based ReID, highlighting publicly available datasets, key challenges, and methodologies. We summarize
44:in advancing UAV-based ReID research. The survey also presents a list of all available approaches for
45:UAV-based ReID. The survey presents challenges associated with UAV-based ReID, including environmental
49:innovative approaches to tackle the unique challenges of UAV-based person ReID.
101:Fig. 1. Common and distinct challenges across ground-based and UAV-based person ReID.
103:• The challenges and opportunities for future research
105:unique challenges not present in traditional ground-based ReID systems, such as:
108:(Fig. 1) highlights unique and shared challenges. Ground-based ReID
114:dataset challenges (Moritz et al., 2021), illumination variations (Fu
132:Additionally, the aerial perspective presents unique challenges that
137:resource-constrained UAV domains demand novel approaches to feature extraction and matching.
145:These challenges underline the need for specialized research and development in UAV-based person ReID. Ground-based ReID systems are
148:challenges and opportunities, offering a comprehensive overview of the
153:challenges that need to be addressed to improve UAV-based person
166:unique challenges introduced by UAV-based surveillance. Section 3
167:emphasizes the common challenges faced by UAV-based person ReID
171:explores the methodologies and approaches applied in UAV-based person ReID, spanning both traditional techniques and state-of-the-art
174:• The methodologies applied in this domain
186:Sections 8, 9, and 10 discuss the broader research challenges and
193:attempt to match these features across different camera views recognizes the same person. However, these systems face several challenges,
200:in addressing these challenges (Wang et al., 2020), the limitations
264:New challenges for ReID from UAV data; Short-term ReID
284:80.72% mAP, 88.77% Rank-1 in aerial-to-CCTV; Effective
364:late 2010s. Layne et al. (2014), and other researchers have investigated the challenges and opportunities presented by UAV-based person
411:Fig. 5. Illustration of common challenges seen in UAV-based ReID datasets: A-blur, Bdynamic background, C-illumination changes, D-low resolution, E-viewpoint variation,
412:F-weather challenges (Li et al., 2021b; Zhang et al., 2021).
415:learning approaches, depending on the nature of the available
418:approaches (Naeem et al., 2023), like clustering or domain adaptation, do not require labeled data, instead leveraging patterns
438:3. Common challenges in ground-based and UAV-based person
440:Ground-based and UAV-based person ReID share several inherent common challenges across different setups. These include lowresolution images, illumination changes, viewpoint variations, realtime processing, and privacy concerns, as discussed further. We first
441:discuss the common challenges across both ReID and UAV-based person
446:ReID for UAVs poses different challenges because of variations in
454:et al., 2022; Moritz et al., 2021). While recent advancements have improved supervised ReID algorithms, challenges persist specifically from
470:videos with insufficient resolution to detect specific features. This limitation poses significant challenges in accurately re-identifying persons
485:has its own challenges (Rieke et al., 2012). Person-ReID involves deep
495:one of the most pronounced challenges (Organisciak et al., 2021).
504:as trees, cause challenges in distinguishing the target person from the
507:3.5. Other challenges
508:In addition to the common challenges mentioned above, other
509:relevant challenges common to both ReID and UAV-based ReID are as
524:challenges for researchers working on UAV-based person ReID.
578:4. Unique challenges in UAV-based person ReID
579:UAV-based person ReID systems face another distinct set of challenges due to the dynamic nature of UAV domains. These unique
580:challenges include dynamic backgrounds, motion blur, battery life,
582:highlights the unique challenges in UAV-based person ReID.
584:5.1.1. MRP dataset: Characteristics and initial challenges
621:offering a rich variety of intra-identity variations and realistic challenges, including occlusions, blur, resolution variations, and diverse
627:images, and the unique challenges of UAV ReID were highlighted.
643:5.1.7. UAV-Human dataset: Comprehensive action and pose recognition
942:conditions. The Manipal-UAV dataset introduces unique challenges for
950:The UAV123 dataset features 123 high-resolution UAV video sequences containing over 110,000 frames, designed to highlight challenges in videos captured from an aerial perspective. The UAV20L
959:5.2.4. D-PTUAC dataset: Tracking in uniform crowds with visual challenges
987:Methodologies and approaches used by UAV ReID datasets.
996:problem in person ReID. The dataset incorporates occlusion and lowresolution attributes, common challenges in person ReID, especially
1003:5.3. ReID data collection and evaluating challenges
1005:types of challenges, as discussed below.
1007:dataset collection faced significant challenges due to adverse weather
1023:presents its own set of challenges, primarily due to the nature of
1026:angle inherently presented challenges for ReID, which typically performs better with various perspectives and angles for subject identification.
1028:where lighting conditions varied significantly. Shadows, glare, and inconsistent lighting across images introduced challenges for model training, as the lighting variations affected the model’s ability to achieve
1041:The above challenges highlight the complexities involved in the data
1121:6. Methodologies and approaches in UAV-based person ReID
1122:Most methods and approaches in person ReID are adapted from
1123:conventional camera benchmarks (Remigereau et al., 2022). UAVbased person ReID methodologies combine traditional computer vision
1124:techniques with modern deep learning approaches. This section examines the key methodologies and approaches utilized in UAV-based
1127:6.1. MRP dataset - ensemble and machine learning approaches
1133:6.2. DRHIT01 dataset - deep learning and triplet loss techniques
1166:These diverse methodologies showcase the advancements in UAVbased person ReID and highlight the importance of leveraging both
1168:models to tackle the unique challenges posed by UAV footage (see
1171:6.3. P-DESTRE dataset - advanced temporal and spatial attention mechanisms
1175:Other methodologies include RetinaNet (Li and Ren, 2019), R-FCN (Dai
1187:6.4. AHU-UAV dataset - federated learning and knowledge distillation
1188:The AHU-UAV dataset (Zhang et al., 2021b) applies deep neural networks (DNNs) alongside federated learning (FL) approaches. Notable
1191:6.5. PRAI-1581 dataset - subspace pooling and cross-entropy loss techniques
1192:PRAI-1581 (Zhang et al., 2021) uses an array of methodologies such
1196:7. UAV-ReID - state-of-the-art results
1197:6.6. UAV-Human dataset - fisheye distortion and KL divergence loss
1206:7.1. Evaluation metrics
1212:6.7. AG-ReID dataset - vision transformers and metric distillation
1239:6.8. Other relevant UAV-based person ReID datasets: UAV123, D-PTUAC,
1242:of approaches, including Kernelized Correlation Filter (KCF) (Tang
1290:17.53%
1310:68.90%
1316:78.30%
1347:37.64%
1348:38.45%
1350:47.47%
1353:48.54%
1354:48.07%
1401:77.03%
1421:78.51%
1424:88.77%
1426:87.86%
1427:88.61%
1434:7.2. MRP
1447:7.4. P-DESTRE
1452:benchmark for this task. The rank-1 accuracy and mAP scores were similar across both datasets, indicating that existing short-term ReID techniques are relatively robust to the challenges posed by UAV-based person ReID. Specifically, NVAN achieved a Rank-1 accuracy of 83.64% on
1458:In contrast, the performance of long-term ReID methods was significantly lower, illustrating the challenges of identifying individuals
1459:without the use of clothing-based features, which are critical for shortterm ReID. The long-term ReID approach, which combined ArcFace
1469:handle these challenges effectively.
1471:7.3. DRHIT01
1495:88.77% in the aerial-to-CCTV setting. This represents an improvement
1499:7.5. PARI-1581
1503:and a Rank-1 accuracy of 49.79%, outperforming the other approaches.
1506:resolutions introduce additional challenges. However, despite the improvements brought by SP, the absolute mAP and Rank-1 accuracies
1513:7.9. Discussion on results
1514:The results noted in the table 4 highlight the challenges and
1518:Rank-1 accuracy of 88.77% in the aerial-to-CCTV setting. This demonstrates the effectiveness of combining a Vision Transformer backbone
1523:across edge servers. This approach is well-suited for real-world UAV
1527:pooling approach. However, there remains significant room for improvement, particularly in the more challenging long-term ReID setting
1535:learning approaches.
1537:7.6. AHU-UAV
1554:7.7. UAV-Human
1556:7.10. Evaluation across datasets
1563:accuracies at 65.81%. Despite these relatively high scores for UAVbased person ReID, the challenges presented by dynamic viewpoints,
1578:7.8. AG-ReID
1579:8. Discussion and challenges
1647:concerted efforts in both data collection and algorithm design. Future
1661:challenges from motion blur and view variability.
1668:- UAV-Human: Broader modalities and challenges
1689:8.1. Comparative analysis of UAV-based person ReID methods
1691:varies significantly depending on the challenges posed by low resolution, dynamic viewpoints, and environmental changes. Methods such
1696:2019), have demonstrated robustness in handling viewpoint variations, achieving comparable results to ground-based ReID approaches
1700:8.4. Ethical implications of UAV-based person ReID
1719:8.2. Environmental conditions and image quality
1720:The challenges associated with capturing high-quality UAV images
1722:weather, and camera specifications. These challenges make it difficult
1723:to create standardized datasets that are applicable in different scenarios. To address these challenges, future research must focus on the
1729:8.3. Imbalance and bias
1738:Fig. 7. Gender and time-of-day distribution in UAV-Human dataset.
1740:9. Future directions
1750:In their study, Cao et al. (2022) employed the approach of
1754:model called Shift Gaussian Convolution Network (SGCNet). This approach is in line with the goal of creating efficient algorithms that are
1764:an understanding of strategic approaches. An essential factor entails
1780:approaches, promoting the advancement of this exciting and rapidly
1782:Scale and Resolution Variations: The challenges of person ReID
1784:To address these challenges, advanced methods and techniques are
1786:learning network that uses residual blocks. This approach aims to tackle
1793:approach helps overcome the difficulties related to scaling and variations in resolution. It can contribute to the advancement of methods
1798:9.3. Domain adaptation and unsupervised approaches
1806:develop UAV-based person ReID models presents a promising approach
1807:to overcoming the challenges posed by UAV person ReID.
1808:Additionally, it is necessary to explore alternative innovative approaches that can effectively address the unique issues associated with
1836:Adaptive fusion approaches improve this process by dynamically
1841:Fig. 8. UAV-based and ground-based viewpoint difference.
1848:overcoming challenges such as limited annotated datasets and diverse
1870:addressing challenges such as data scarcity and variability in real-world
1882:9.7. Explainability
1889:elevated dynamic perspectives. This approach aids in clarifying the
1895:explainability models directly to UAV-based settings, as shown in Fig. 8.
1896:The elevated and mobile perspectives in UAV-based person ReID require tailored approaches to account for these unique challenges. Future research should focus on developing more sophisticated explainability techniques tailored to UAV-based person ReID scenarios.
1908:Despite the challenges, the relevance of UAV ReID datasets cannot
1932:Future research should prioritize the development of more diverse
1935:the exploration of innovative approaches such as generative AI models to address the unique challenges of UAV-based person ReID. In
1939:Collecting UAV ReID datasets is not without its challenges. Major
1945:UAVs present both opportunities and challenges. While current datasets
1947:the challenges related to dataset collection and analysis will be essential
1957:Prendinger, H., 2017. Okutama-action: An aerial view video dataset for concurrent
1970:2019. Springer, pp. 657–668.
1971:Bochinski, E., Eiselein, V., Sikora, T., 2017. High-speed tracking-by-detection without
1983:Carreira, J., Zisserman, A., 2017. Quo vadis, action recognition? A new model and the
1985:Pattern Recognition. pp. 6299–6308.
1994:Multimedia. pp. 4939–4948.
2016:2018. The unmanned aerial vehicle benchmark: Object detection and tracking. In:
2020:51–58.
2025:arxiv.2208.06818.
2027:grouping: A simple unsupervised cross domain adaptation approach for person
2047:of face recognition: A review. Electronics 9 (8), 1188.
2060:10, 45001–45028.
2072:Gevaert, C.M., Sliuzas, R., Persello, C., Vosselman, G., 2018. Evaluating the societal
2085:Gunawan, F.E., Harefa, J., Sekishita, N., 2018. Face recognition on linear motionblurred image. TELKOMNIKA (Telecommun. Comput. Electron. Control) 16 (3),
2088:networks: A survey. J. King Saud Univ.-Comput. Inf. Sci. 34 (7), 4244–4268.
2090:Jung, J., Chang, A., Malambo, L., Popescu, S.C., Gates, I.T., Cope, D., 2018.
2107:Hermans, A., Beyer, L., Leibe, B., 2017. In defense of the triplet loss for person
2108:re-identification. ArXiv preprint arXiv:1703.07737.
2112:Hsieh, M.-R., Lin, Y.-L., Hsu, W.H., 2017. Drone-based object counting by spatially
2118:Iscen, A., Tolias, G., Avrithis, Y., Furon, T., Chum, O., 2017. Efficient diffusion on
2122:Jiao, J., Zheng, W.-S., Wu, A., Zhu, X., Gong, S., 2018. Deep low-resolution person
2135:Rev. 52, 927–948.
2140:1696–1708.
2142:unmanned aerial vehicle routing problem approach for humanitarian coverage path
2157:International Conference on Computer Vision. pp. 3958–3967.
2167:Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S., 2017. Feature
2173:I 14. Springer, pp. 21–37.
2176:preprint arXiv:1908.01683.
2177:Liu, J., Zha, Z.-J., Xie, H., Xiong, Z., Zhang, Y., 2018. CA 3 net. In: Proceedings of the
2195:and Pattern Recognition. pp. 4040–4048.
2203:aerial vehicles (UAVs): Practical aspects, applications, open challenges, security
2204:issues, and future trends. Intell. Serv. Robot. 16 (1), 109–137.
2247:Peng, K., Lin, F., Chen, B.M., 2017. Online schedule for autonomy of multiple unmanned
2249:Poppe, R., 2007. Vision-based human motion analysis: An overview. Comput. Vis. Image
2250:Underst. 108 (1–2), 4–18.
2255:networks with battery-limited drones. IEEE Commun. Lett. 24, 2664–2668. http:
2264:IEEE, pp. 3853–3857.
2269:Asian J. Comput. Sci. Technol. 8, 13–17. http://dx.doi.org/10.51983/ajcst-2019.8.
2271:Rezwan, S., Choi, W., 2022. Artificial intelligence approaches for UAV navigation:
2272:Recent advances and future challenges. IEEE Access 10, 26320–26339.
2279:Communication Workshop and Conference. CCWC, IEEE, pp. 0613–0618.
2280:Samek, W., 2017. Explainable artificial intelligence: Understanding, visualizing and
2281:interpreting deep learning models. ArXiv preprint arXiv:1708.08296.
2293:re-identification approaches: various aspects. Multimedia Tools Appl. 81 (11),
2302:Suh, Y., Wang, J., Tang, S., Mei, T., Lee, K.M., 2018. Part-aligned bilinear representations for person re-identification. In: Proceedings of the European Conference on
2309:Sun, Y., Zheng, L., Yang, Y., Tian, Q., Wang, S., 2018. Beyond part models: Person
2319:442, 337–347.
2323:Tang, M., Yu, B., Zhang, F., Wang, J., 2018. High-speed tracking with multi-kernel
2328:2726–2738.
2329:Taud, H., Mas, J.-F., 2018. Multilayer perceptron (MLP). In: Geomatic Approaches for
2335:Voigt, P., Von dem Bussche, A., 2017. The eu general data protection regulation (gdpr).
2342:Wang, X., Doretto, G., Sebastian, T., Rittscher, J., Tu, P., 2007. Shape and appearance
2344:IEEE, pp. 1–8.
2346:re-identification approaches. IEEE Access 8, 45556–45583.
2352:559–568.
2355:Vision and Pattern Recognition. pp. 79–88.
2367:Xiao, T., Li, S., Wang, B., Lin, L., Wang, X., 2017. Joint detection and identification
2371:(12), e7687.
2387:Yi, Z., Zhang, H., Tan, P., Gong, M., 2017. Dualgan: Unsupervised dual learning for
2389:on Computer Vision. pp. 2849–2857.
2394:A retrospective on domain specific open challenges and future trends. Pattern
2398:Zhang, S., Lan, X., Qi, Y., Yuen, P.C., 2017. Robust visual tracking via basis matching.
2403:Pattern Recognition. pp. 1278–1287.
2422:Dependable and Secure Computing. DSC, IEEE, pp. 1–8.
2434:Conference on Computer Vision and Pattern Recognition. pp. 2138–2147.
2435:Zheng, Z., Zheng, L., Yang, Y., 2017. Unlabeled samples generated by gan improve the
2438:Zhong, Z., Zheng, L., Cao, D., Li, S., 2017. Re-ranking person re-identification with
2440:and Pattern Recognition. pp. 1318–1327.
2453:Artif. Intell. vol. 34, 13114–13121. http://dx.doi.org/10.1609/aaai.v34i07.7014.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 14 'UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261
Table 3
Methodologies and approaches used by UAV ReID datasets.

dataset includes high-resolution gallery images captured in constrained
settings. These images encompass four distinct poses and serve multiple
purposes, including developing a facial identification system capable of
recognizing individuals in aerial images, even when the facial area is
limited to only a few pixels. The dataset simulates scenarios where an
individual (an ‘‘intruder’’) moves within a crowd of people wearing similar attire. This setup resembles the challenge of re-identifying a specific
person across different camera views or time instances, which is a core
problem in person ReID. The dataset incorporates occlusion and lowresolution attributes, common challenges in person ReID, especially
in real-world surveillance scenarios. However, it is essential to note
that the primary focus of the D-PTUAC dataset is on the single-object
tracking (SOT) problem in the context of drone-based surveillance. The
dataset evaluates the performance of visual object trackers by requiring
them to track a specific target person within a crowd of individuals
wearing similar clothing.
5.3. ReID data collection and evaluating challenges
The dataset collection and evaluation process encounter different
types of challenges, as discussed below.
Data Preprocessing and Occlusion Issues: The UAV-Human
dataset collection faced significant challenges due to adverse weather
conditions such as rain and wind. These environmental factors caused
several issues:
- Occlusions from Umbrellas: During rain, the use of umbrellas
obstructs individuals from view and creates occlusions that complicate
the ReID process (Huang et al., 2022).
- UAV Shaking: Strong winds cause UAV instability, resulting in
blurry and unstable image captures. This further hinders the clarity
and quality of the data, adding complexity to the identification task (Li
et al., 2021b).
- Low Image Quality: Rain and fog reduce the overall quality of
the images, making it difficult to maintain high-resolution data for
effective ReID. Preprocessing the data became more challenging due
to these weather-induced noise factors, which necessitated addressing
occlusions and blurriness.
Model Training and Convergence Issues: The AHU-UAV dataset
presents its own set of challenges, primarily due to the nature of
UAV-based data collection:
- Overhead Angle: The UAVs were required to fly at certain altitudes, resulting in predominantly overhead views of individuals. This
angle inherently presented challenges for ReID, which typically performs better with various perspectives and angles for subject identification.
- Lighting Conditions: The dataset was collected outside a canteen,
where lighting conditions varied significantly. Shadows, glare, and inconsistent lighting across images introduced challenges for model training, as the lighting variations affected the model’s ability to achieve
stability during the training process.
- Time of Collection and Crowd Density: Data collection during
lunchtime increased crowd density, making it harder to accurately
re-identify individuals in a crowded environment. The many people
introduced additional complexity due to posture variability, clothing diversity, and frequent occlusions between individuals, which negatively
affected model convergence and identification performance.
Hardware Constraints: Despite efforts to collect high-quality data,
hardware limitations sometimes impact the stability and quality of
the images. Operating UAVs in adverse weather conditions placed
additional strain on the equipment, leading to reduced capture quality
in some instances. This, in turn, affected the data evaluation process,
particularly in maintaining consistent quality across the dataset.
The above challenges highlight the complexities involved in the data
collection and evaluation process for UAV-based person ReID.

Dataset Name

Methodologies and Approaches

MRP (Layne
et al., 2014)

Ensemble of Local Features (ELF) - RankSVM - BR-SVM KISS

DRHIT01
(Grigorev
et al., 2019)

Object Detection with Faster R-CNN - CNN - Triple Loss L-GM Loss - Random Horizontal Flip

UAV123 (Nousi
et al., 2020)

Kernelized Correlation Filter (KCF) - CNN - Fully
Convolutional Siamese Networks (SiamFC) - Multilayer
Perceptron (MLP) - Log-loss Objective Function

P-DESTRE
(Kumar et al.,
2021)

Bounding Boxes - Soft Biometrics Labels - Mask R-CNN Deep SORT - Deep Head Pose - RetinaNet - R-FCN - SSD
- TracktorCV - V-IOU - GLTR (Global-Local Temporal
Representations) - COSAM - NVAN (Non-local Video
Attention Network) - ArcFace

AHU-UAV
(Zhang et al.,
2021b)

Deep Neural Networks (DNNs) - Federated Learning (FL)
- Knowledge Distillation - Federated Average (FedAvg)
Algorithm - Stochastic Gradient Descent (SGD)

PRAI-1581
(Zhang et al.,
2021)

SVDNet- PCANet FD-GAN - Subspace Pooling - Cross-Entropy Loss - Triplet Loss

UAV-Human
(Li et al.,
2021b)

Fisheye Distortion Mitigation - I3D Network Kullback–Leibler (KL) Divergence Loss

BRIAR (Cornett
et al., 2023)

Atmospheric Turbulence Mitigation - YOLOv5 InsightFace

Manipal-UAV
(Akshatha
et al., 2023)

CVAT annotation tool - Feature Pyramid Networks (FPN)
- Faster R-CNN - Cascade R-CNN - Sparse R-CNN CornerNet - RetinaNet - Deformable DETR - YOLO-F DDOD - Intersection over Union (IoU)

AG-ReID
(Nguyen et al.,
2024)

Vision Transformer (ViT) - Cross-Entropy Loss - Triplet
Loss - Metric Distillation Loss - Attribute Prior Loss

D-PTUAC
(Alansari et al.,
2024)

Monocular Depth Estimation - Segmentation Mask
Generation

6. Methodologies and approaches in UAV-based person ReID
Most methods and approaches in person ReID are adapted from
conventional camera benchmarks (Remigereau et al., 2022). UAVbased person ReID methodologies combine traditional computer vision
techniques with modern deep learning approaches. This section examines the key methodologies and approaches utilized in UAV-based
person ReID datasets, summarizing notable datasets such as DRHIT01,
PRAI-1581, UAV-HUMAN, AG-ReID, AHU-UAV, P-DESTRE, and MRP.
6.1. MRP dataset - ensemble and machine learning approaches
The MRP dataset (Layne et al., 2014) incorporates an ensemble
of local features (ELF) combined with various machine learning algorithms, including RankSVM (Zhang et al., 2016), BR-SVM (Tsujitani
and Tanaka, 2011), and KISS (Tao et al., 2016). These techniques aim
to improve ReID accuracy by focusing on local features, ranking-based
learning, and similarity measures.
6.2. DRHIT01 dataset - deep learning and triplet loss techniques
DRHIT01 (Grigorev et al., 2019) utilizes advanced deep learning
methods such as Faster R-CNN (Bharati and Pramanik, 2020) for object
detection and CNNs in conjunction with techniques like triplet loss and
large-margin Gaussian mixture (L-GM) loss. Data augmentation strategies, such as random horizontal flips, enhance the model’s robustness
and performance.
9

Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261

D-PTUAC Dataset - Depth Estimation and Segmentation Mask
Generation Lastly, the D-PTUAC dataset (Alansari et al., 2024) employs monocular depth estimation and segmentation mask generation
to facilitate re-identification in challenging drone-based environments.
BRIAR Dataset - Turbulence Mitigation and YOLOv5 Implementation BRIAR (Cornett et al., 2023) focuses on atmospheric turbulence
mitigation, which is crucial for UAV imagery. Additionally, the dataset
employs the YOLOv5 model (Wu et al., 2021a) and InsightFace for face
detection and recognition, respectively.
Manipal-UAV Dataset - Object Detection and Annotation Techniques The Manipal-UAV dataset (Akshatha et al., 2023) includes
annotations via the Computer Vision Annotation Tool (CVAT tool),
an open-source annotation tool designed for labeling data used in
computer vision tasks (Krenzer et al., 2022) and a range of object
detection frameworks, such as Feature Pyramid Networks (FPN) (Lin
et al., 2017), Faster R-CNN (Ren et al., 2016), Cascade R-CNN (Yu et al.,
2022), and Sparse R-CNN (Sun et al., 2021), are designed to enhance
detection accuracy and efficiency. FPN improves multi-scale detection,
while Faster R-CNN and Cascade R-CNN progressively refine object proposals. Sparse R-CNN reduces computational complexity using sparse
proposals. State-of-the-art models like Deformable DETR (Zhu et al.,
2020b), YOLO-F (Cao et al., 2023), and DDOD (Chen et al., 2021b) further advance detection capabilities. Deformable DETR handles variable
object scales, YOLO-F focuses on speed, and DDOD separates classification from localization for improved precision. Intersection over Union
(IoU) (Rahman and Wang, 2016) serves as the key metric to evaluate
these models’ performance.
These diverse methodologies showcase the advancements in UAVbased person ReID and highlight the importance of leveraging both
classic machine learning algorithms and cutting-edge deep learning
models to tackle the unique challenges posed by UAV footage (see
Table 3).

6.3. P-DESTRE dataset - advanced temporal and spatial attention mechanisms
P-DESTRE (Kumar et al., 2021) features a comprehensive set of
techniques, ranging from bounding boxes and soft biometrics labels
to more sophisticated models such as Mask R-CNN and Deep SORT.
Other methodologies include RetinaNet (Li and Ren, 2019), R-FCN (Dai
et al., 2016), Single Shot Multibox Detector (SSD) (Liu et al., 2016),
TracktorCV (Bergmann et al., 2019), V-IOU (Bochinski et al., 2017),
and Global-Local Temporal Representations (GLTR) (Li et al., 2019).
Attention mechanisms such as COSAM (Subramaniam et al., 2019) and
Non-local Video Attention Network (NVAN) (Liu et al., 2019) are also
employed alongside ArcFace for face recognition tasks. Faster R-CNN
with ResNet50 is chosen for this dataset due to its superior object
detection capabilities, making it ideal for detecting pedestrians in UAV
images. The model combines region proposal networks (RPN) with CNN
to provide accurate object localization and classification (Tanatipuknon
et al., 2021).
6.4. AHU-UAV dataset - federated learning and knowledge distillation
The AHU-UAV dataset (Zhang et al., 2021b) applies deep neural networks (DNNs) alongside federated learning (FL) approaches. Notable
techniques include knowledge distillation and the Federated Average (FedAvg) algorithm, with optimization performed via stochastic
gradient descent (SGD).
6.5. PRAI-1581 dataset - subspace pooling and cross-entropy loss techniques
PRAI-1581 (Zhang et al., 2021) uses an array of methodologies such
as SVDNet, PCANet, and FD-GAN. It employs subspace pooling, crossentropy loss, and triplet loss functions, which enhance the dataset’s
ability to distinguish between individuals in re-identification tasks.

7. UAV-ReID - state-of-the-art results
6.6. UAV-Human dataset - fisheye distortion and KL divergence loss
We first discuss commonly used metrics and discuss the results.
The UAV-Human dataset (Li et al., 2021b), a fisheye-based action recognition method, is proposed to mitigate distortions in fisheye
videos through the use of unbounded transformations guided by flat
RGB videos. The dataset also makes use of the I3D network (Carreira
and Zisserman, 2017) and Kullback–Leibler (KL) (Zeng et al., 2014)
divergence loss to improve model learning and reduce error in person
ReID.

7.1. Evaluation metrics
mAP. The mean Average Precision (mAP) is a metric commonly used
to evaluate the accuracy of computer vision models across all classes
within a given dataset. It provides a comprehensive measure by averaging the precision at various recall levels for each class, offering an
overall assessment of the model’s performance (Padilla et al., 2020).

6.7. AG-ReID dataset - vision transformers and metric distillation

1 ∑
AP(𝑖)
𝑁 𝑖=1
𝑁

mAP =
The AG-ReID dataset (Nguyen et al., 2024) utilizes the Vision Transformer (ViT) model alongside various loss functions such as crossentropy, triplet loss, metric distillation loss (Chen et al., 2021a), and
attribute prior loss, which contribute to more accurate re-identification.
Metric distillation loss focuses on transferring the knowledge of distance metrics from a teacher model to a student model, enhancing the
student’s ability to learn robust feature representations by preserving
the relationships between data points in the embedding space.

where:
• 𝑁 is the total number of classes being evaluated.
• AP(𝑖) is the Average Precision for the 𝑖th class.
Rank-1 accuracy. ReID is evaluated as a retrieval task, where the
objective is to accurately identify a target individual within a set of
candidates. A key performance metric employed in this survey is the
Rank-1 accuracy.
The Rank-1 accuracy is calculated as:
# correct matches at rank-1
× 100%
𝑅𝑎𝑛𝑘 − 1 𝐴𝑐 𝑐 =
Total # of queries

6.8. Other relevant UAV-based person ReID datasets: UAV123, D-PTUAC,
BRIAR, and Manipal-UAV
UAV123 Dataset - Kernelized Correlation Filter and CNN Approaches The UAV123 dataset (Nousi et al., 2020) employs a variety
of approaches, including Kernelized Correlation Filter (KCF) (Tang
et al., 2018), CNNs, and fully convolutional Siamese networks (SiamFC)
(Bertinetto et al., 2016). A multilayer perceptron (MLP) (Taud and Mas,
2018) and a log-loss objective function are also utilized to improve the
tracking and identification performance.

This measure quantifies the frequency with which the correct match
is positioned at the top of the ranked list of potential matches, effectively representing the system’s precision in immediate identification. Zhong et al. (2017).
10

Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261

Table 4
Performance of UAV-based person ReID on different datasets.
Dataset

Network (pre-training dataset)

Method

mAP

Rank-1

MRP (Layne et al., 2014)

–
–
–
–
–

Nearest Neighbor (NN)
RankSVM
KISS
BR-SVM
Nearest Neighbor (NN) + DA

47%
12.32%
28%
34%
56%

–
9.76%
17.53%
24.00%
–

P-DESTRE Short-Term
ReID (Kumar et al.,
2021)

ResNet-50 (CUHK-SYSU)
ResNet-50 (MARS)
ResNet-50 (CUHK03)
Se-ResNet-50

NVAN
COSAM
GLTR
COSAM+Temporal

79.18%
72.10%
68.90%
70.50%

83.64%
76.20%
74.50%
78.30%

P-DESTRE Long-Term
ReID (Kumar et al.,
2021)

MobileNetV2 + Se-ResNet50
ResNet-50

ArcFace + COSAM
ArcFace + COSAM

–
–

49.88%
51.34%

PRAI-1581 (Zhang et al., 2021)

ResNet-50 (ImageNet)

Triplet Loss
Subspace Pooling (SP) + TL
SVDNet
AlignedReID
PCB + RPP

36.49%
39.58%
36.7%
37.64%
38.45%

47.47%
49.79%
46.1%
48.54%
48.07%

AHU-UAV (Zhang et al., 2021b)

ResNet-50 - IDE

Local Model Training
Federated Learning
Knowledge Distillation

–
–
–

65%
79%
69%

UAV-Human (Li et al., 2021b)

–
–
–
–

Tricks
DG-Net
PCB
Part-Aligned

63.41%
61.97%
61.05%
60.86%

62.48%
65.81%
62.19%
60.86%

ViT (ImageNET)

Aerial to CCTV
Aerial to Wearable
CCTV to Aerial
Wearable to Aerial

77.03%
80.48%
75.90%
76.59%

85.40%
89.77%
84.65%
84.27%

ViT+EVA+EP
(ImageNET)

Aerial to CCTV
Aerial to Wearable
CCTV to Aerial
Wearable to Aerial

80.72%
84.85%
78.51%
80.11%

88.77%
93.62%
87.86%
88.61%

AG-ReID (Nguyen et al., 2024)

combination of loss functions helps the model better handle the challenging variations in UAV-based images, such as changing scales and
viewpoints.

7.2. MRP
Nearest Neighbor (NN) (Li et al., 2012) showed relatively strong
performance with an mAP of 47%, outperforming more complex methods like RankSVM (Zhang et al., 2016) and KISS (Tao et al., 2016),
which struggled to generalize across the dynamically changing views of
mobile platforms. This aligns with findings from the MRP paper, where
simpler models excelled due to their robustness against overfitting
to fixed camera views. Domain Adaptation (DA) (Fu et al., 2019)
improved the performance of Nearest Neighbor (NN) from 47% mAP to
56% mAP on the MRP dataset. However, the improvements for more
complex models like KISS (Tao et al., 2016) and BR-SVM (Tsujitani and
Tanaka, 2011) were marginal, indicating that simpler models benefit
more from these techniques in UAV-based scenarios.

7.4. P-DESTRE
For short-term ReID, the performance of GLTR (Li et al., 2019), CoSegmentation Based Attention Module (COSAM) (Subramaniam et al.,
2019), and Non-Local Video Attention Network (NVAN) (Kumar et al.,
2021) on the P-DESTRE dataset was comparable to their performance
on the MARS dataset (Zheng et al., 2016a), which is a widely-used
benchmark for this task. The rank-1 accuracy and mAP scores were similar across both datasets, indicating that existing short-term ReID techniques are relatively robust to the challenges posed by UAV-based person ReID. Specifically, NVAN achieved a Rank-1 accuracy of 83.64% on
P-DESTRE, which is close to its performance on MARS, demonstrating
its effectiveness in scenarios with moving cameras and dynamic environments. Similarly, GLTR (Li et al., 2019) and COSAM (Subramaniam
et al., 2019) performed well, achieving Rank-1 accuracies of 74.5% and
76.2%, respectively, showcasing their robustness in handling temporal
and spatial variations in the data.
In contrast, the performance of long-term ReID methods was significantly lower, illustrating the challenges of identifying individuals
without the use of clothing-based features, which are critical for shortterm ReID. The long-term ReID approach, which combined ArcFace
for face features and COSAM (Subramaniam et al., 2019) for body
features, yielded a Rank-1 accuracy of only 49.88%, substantially lower
than the short-term ReID methods. This performance gap highlights
the increased difficulty of reidentifying individuals in different sessions
in long-term scenarios, where factors such as changes in appearance
(e.g. clothing, hairstyle) and pose significantly affect the model’s ability
to maintain high accuracy (Qian et al., 2020). Despite combining
both facial and body features, the performance remains much lower,
indicating that long-term ReID still requires further advancements to
handle these challenges effectively.

7.3. DRHIT01
Pre-training the ResNet-50 model on a large-scale person ReID
dataset (like CUHK-SYSU (Xiao et al., 2017), Market-1501 (Zheng et al.,
2015), or CUHK03 (Li et al., 2014)) before fine-tuning on DRHIT01
significantly improves performance compared to pre-training only on
the general ImageNet dataset. The choice of the pre-training dataset is
important. Among the ReID datasets, pre-training on CUHK-SYSU yields
the best results when fine-tuned on DRHIT01. This implies that CUHKSYSU likely shares more similar data characteristics with DRHIT01
compared to datasets like Market-1501 or CUHK03. Furthermore, the
use of multiple loss functions proved beneficial. Adding the L-GM
(Large Margin Gaussian Mixture) loss (Grigorev et al., 2019) to the
Triplet loss function generally led to improvements in both mAP and
rank-1 accuracy over using Triplet loss (Hermans et al., 2017) alone.
This indicates that employing complementary loss functions can enhance the discriminative power of the learned features by encouraging
the network to better separate identities in the feature space. This
11

Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261

combines a Vision Transformer (ViT) (Han et al., 2022) backbone
with Explainable Processing (EP) and Elevated-View Attention (EVA)
components, achieves an mAP of 80.72% and a Rank-1 accuracy of
88.77% in the aerial-to-CCTV setting. This represents an improvement
of 3.69% in mAP and 3.37% in Rank-1 accuracy compared to using the
ViT backbone alone.

7.5. PARI-1581
The proposed subspace pooling (SP) (Wei et al., 2018b) method
demonstrates superior performance compared to these methods. When
combined with triplet loss, SP achieved the highest mAP of 39.58%
and a Rank-1 accuracy of 49.79%, outperforming the other approaches.
This indicates that subspace pooling is particularly effective for ReID
in aerial images, where the diversity of viewpoints and lower image
resolutions introduce additional challenges. However, despite the improvements brought by SP, the absolute mAP and Rank-1 accuracies
remain below 50%, which highlights the difficulty of this new ReID
scenario. The results from all methods demonstrate that PARI-1581 is
a challenging dataset, emphasizing the need for further advancements
in person ReID techniques to handle the unique conditions presented
by UAV-based person ReID.

7.9. Discussion on results
The results noted in the table 4 highlight the challenges and
progress in aerial-ground person ReID. The proposed Explainable
Elevated-View Attention (EP+EVA) method achieves the best overall
performance on the AG-ReID dataset, with an mAP of 80.72% and a
Rank-1 accuracy of 88.77% in the aerial-to-CCTV setting. This demonstrates the effectiveness of combining a Vision Transformer backbone
with explainable processing and attention mechanisms tailored for
elevated views. The Fed-UAV framework also shows promise on the
AHU-UAV dataset, achieving a Rank-1 accuracy of 0.79 when using
federated learning to train the person ReID model collaboratively
across edge servers. This approach is well-suited for real-world UAV
scenarios where data may be distributed and privacy is a concern. In the
other datasets, state-of-the-art ReID methods achieve Rank-1 accuracies
of 65.81% on UAV-Human, around 80% on P-DESTRE for shortterm ReID, and 49.79% on PARI-1581 using the proposed subspace
pooling approach. However, there remains significant room for improvement, particularly in the more challenging long-term ReID setting
on P-DESTRE, where accuracy drops to 49.88%. Overall, the EP+EVA
method and Fed-UAV framework represent important advances in
aerial-ground person ReID, while the other datasets and benchmarks
highlight the need for continued research to boost performance in this
domain. Key areas for future work include developing more robust
feature representations, addressing domain gaps between aerial and
ground views, and leveraging larger-scale datasets and collaborative
learning approaches.

7.6. AHU-UAV
Table 4 shows the Rank-1 accuracy results in the AHU-UAV dataset
using three different training methods in the Fed-UAV framework.
Training the person ReID model only on the local AHU-UAV dataset
without federated learning, the Rank-1 accuracy is 0.65. Training the
model using federated learning (FL) (Konečnỳ, 2016), which aggregates
the model updates from different edge servers, significantly improves
the accuracy of Rank-1 to 0.79. Model training with knowledge distillation (KD) (Gou et al., 2021), after FL, the global model is compressed
using knowledge distillation. The Rank-1 accuracy is 0.69.
FL can substantially improve the accuracy of the person ReID compared to local training by enabling collaborative learning across edge
servers. Although knowledge distillation leads to a slight decrease
in accuracy compared to FL, it still outperforms local training while
significantly reducing the model size and inference time.
Overall, the Fed-UAV framework (Zhang et al., 2021b) that uses
federated learning and knowledge distillation is effective for person
ReID in real-world UAV delivery scenarios, as represented by the AHUUAV dataset. It achieves a good balance between model accuracy and
efficiency.
7.7. UAV-Human

7.10. Evaluation across datasets

Four state-of-the-art ReID methods were evaluated on the UAVHuman dataset: Tricks (Luo et al., 2019), DG-Net (Zheng et al., 2019),
PCB (Sun et al., 2018), and Part-Aligned (Suh et al., 2018). The
results show that these methods achieve mAP scores around 60%
and Rank-1 accuracies between 60%–65%. Specifically, Tricks obtains
the highest mAP at 63.41%, while DG-Net achieves the best Rank-1
accuracies at 65.81%. Despite these relatively high scores for UAVbased person ReID, the challenges presented by dynamic viewpoints,
motion blur, and varying resolutions remain, suggesting that further
improvements are necessary for robust ReID performance in real-world
UAV applications.

As detailed in Table 5, the datasets, including AHU-UAV (Zhang
et al., 2021b), AG-ReID (Nguyen et al., 2024), MRP (Layne et al.,
2014), P-DESTRE (Kumar et al., 2021), PRAI-1581 (Zhang et al., 2021),
DRHIT01 (Grigorev et al., 2019), and UAV-Human (Li et al., 2021b),
are evaluated based on key factors such as real-time applicability, memory consumption, power efficiency, and performance in aerial surveillance and re-identification tasks. Table 5 also highlights the main
optimizations, including federated learning, knowledge distillation, and
attention-based feature extraction, which improve the computational
performance of these models in resource-constrained environments
such as the UAV domain.

7.8. AG-ReID
8. Discussion and challenges
The evaluation results of the proposed Explainable Elevated-View
Attention (EP+EVA) method on the AG-ReID dataset (Nguyen et al.,
2024) demonstrate its effectiveness in aerial-ground person ReID. The
AG-ReID dataset is split evenly into training and testing sets, with the
testing set further divided into aerial-ground and ground-aerial categories. The primary evaluation metrics used are mAP (mean Average
Precision) and CMC-k (Cumulative Matching Characteristic at rank-k),
with a focus on rank-1 accuracy. CMC-k measures the probability that
a correct match for a query appears within the top-k ranked results,
making it a key metric for assessing retrieval performance, particularly
at rank-1, which indicates the likelihood that the correct match is the
top-ranked result (Wang et al., 2007). The EP+EVA method, which

The lack of publicly accessible datasets presents a significant challenge for researchers aiming to conduct studies on UAV-based person
ReID. Although earlier sections have outlined the existing datasets and
their limitations, it is crucial to emphasize that the development of new,
more diverse datasets is essential for advancing person ReID algorithms
in UAV applications (Mueller et al., 2016; Du et al., 2018; Hsieh et al.,
2017). The variety of datasets that are currently available, whether they
are designed for specific tasks like finding forest fires or keeping an eye
on the environment, still does not give ReID full coverage (Guan et al.,
2022; Hayes et al., 2022; Rominger and Meyer, 2021).
12

Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261

Table 5
Comparison with existing methods and optimizations across all datasets.
Aspect

many ways, including the disproportionate representation of certain demographic groups, environmental conditions, or viewpoints within the
data. For instance, UAV-based datasets may over-represent certain demographics, such as age groups or ethnicities, while under-representing
others. Such imbalances lead to biased models that may perform well
on over-represented groups but poorly on under-represented ones,
reducing the generalizability of the ReID system.
Geographic bias remains a common concern across all these
datasets. Most of the data is collected in specific regions, such as
university campuses or urban areas, which may not be globally representative. Several datasets have acknowledged and attempted to
mitigate these issues. For example, the UAV-Human dataset (Li et al.,
2021b) specifically addresses diversity by collecting data from 45
different sites, spanning urban, rural, and natural environments such
as forests and riversides. It also includes data captured under various
weather conditions (sunny, cloudy, rainy, and windy) and at different
times of the day (daytime and nighttime), which helps ensure a more
balanced representation of environmental factors.
Additionally, AG-ReID (Nguyen et al., 2024) and AHU-UAV (Zhang
et al., 2021b) make efforts to ensure that subjects in the datasets come
from a range of demographics, capturing different age groups, genders,
and clothing styles to balance the representation of individuals. This
type of soft-biometric annotation helps mitigate bias that could arise
from homogeneous subject pools.
To combat these biases, researchers are exploring data augmentation techniques and domain adaptation methods. For example, some
datasets, like UAV-Human (Li et al., 2021b), explore using multiple
data modalities (RGB, infrared, depth, fisheye, and night vision) to
capture human behavior across various perspectives and environments,
making models more robust to diverse scenarios. However, despite
the diverse data modalities, the UAV-Human dataset still presents
significant biases. Out of the 119 subjects, only 20 (16.8%) are female,
while 99 (83.2%) are male, resulting in a gender imbalance that could
influence model development. Additionally, of the 41,254 frames in the
dataset, only 1,412 (3.4%) were captured at night, meaning that nighttime data is severely underrepresented. These biases could potentially
limit the generalization of models trained on the dataset, particularly
for gender-specific behavior analysis and low-light (nighttime) conditions. As shown in Fig. 7 these imbalances highlight the need for
addressing these issues through targeted data collection or synthetic
techniques, which remains essential for improving model robustness.
Addressing imbalance and bias in UAV-based person ReID requires
concerted efforts in both data collection and algorithm design. Future
research in UAV-based person ReID must prioritize the development of
more diverse and representative datasets

Evaluation Across All Datasets
- AHU-UAV: More efficient than traditional cloud-based
models for real-time UAV use.

Comparison with
Existing Methods

- AG-ReID.v2: Superior accuracy, but slower than CNNs
like HRNet for real-time applications.
- MRP: More dynamic views than CCTV, but faces
challenges from motion blur and view variability.
- P-DESTRE: Higher resolution causes performance
drops compared to fixed-camera datasets.
- PRAI-1581: Outperforms models like SVDNet and PCB
but is challenged by aerial views.
- DRHIT01: Outperforms CNN baselines (PCB,
AlignedReID) due to multi-branch architecture.
- UAV-Human: Broader modalities and challenges
compared to PRAI-1581; more real-world conditions.
- AHU-UAV: Federated learning and knowledge
distillation improve efficiency.

Optimizations

- AG-ReID.v2: Mixed-precision training and
attention-based feature localization reduce computation.
- MRP: Corrective transformations (e.g., roll correction)
and thresholding reduce false positives.
- P-DESTRE: Mask R-CNN and Deep SORT improve
detection/tracking performance.
- PRAI-1581: Subspace pooling, combined triplet and
cross-entropy loss enhance learning.
- DRHIT01: Transfer learning and channel group
learning reduce redundancy and improve accuracy.
- UAV-Human: Guided Transformer Modules handle
fisheye distortions, KL Divergence guides fisheye
models.

8.1. Comparative analysis of UAV-based person ReID methods
In UAV-based person ReID, the performance of various methods
varies significantly depending on the challenges posed by low resolution, dynamic viewpoints, and environmental changes. Methods such
as Subspace Pooling combined with Triplet Loss have shown promising results in addressing low-resolution issues in datasets like PRAI1581 (Zhang et al., 2021). However, despite these advancements, the
overall mean average precision (mAP) and Rank-1 accuracy scores remain below 50%, highlighting the continued difficulty of re-identifying
individuals in low-resolution aerial images. Attention-based models,
such as NVAN (Liu et al., 2019) and COSAM (Subramaniam et al.,
2019), have demonstrated robustness in handling viewpoint variations, achieving comparable results to ground-based ReID approaches
in dynamic UAV environments like those presented in the P-DESTRE
dataset.

8.4. Ethical implications of UAV-based person ReID
One of the major privacy concerns with UAVs is the potential for
spying on people. This highlights the need for clear regulations to govern their use in civilian airspace. Additionally, UAVs can be equipped
with directional microphones, allowing them to eavesdrop on private
conversations (Mekdad et al., 2023). In addition to privacy concerns,
ethical considerations are essential for the operation of UAVs, ensuring
their compliance with legal frameworks such as the GDPR (Voigt and
Von dem Bussche, 2017). The GDPR, a comprehensive data protection
regulation in the EU, imposes strict guidelines on the collection and
processing of personal data to protect individual privacy rights. The
principles of privacy by design and privacy by default are particularly
important. As outlined in the GDPR and supported by the European
Data Protection Supervisor (EDPS), UAS operations must ensure that
data protection is embedded in their design and functionality. This
includes minimizing data collection, safeguarding personal privacy, and
ensuring that operations do not infringe on fundamental rights such as
human dignity, freedom of association, and non-discrimination (Bassi,
2020).

8.2. Environmental conditions and image quality
The challenges associated with capturing high-quality UAV images
have been discussed in earlier sections, including variations in altitude,
weather, and camera specifications. These challenges make it difficult
to create standardized datasets that are applicable in different scenarios. To address these challenges, future research must focus on the
creation of UAV datasets that encompass a broader range of attributes
and environmental conditions. Standardizing data collection protocols
and establishing benchmark datasets that represent diverse real-world
scenarios will be key to moving the field forward (Mayer et al., 2016;
Zhang et al., 2021).
8.3. Imbalance and bias
A significant challenge in UAV-based person ReID is the presence of
dataset imbalances and inherent biases. This imbalance can manifest in
13

Y. Albaluchi, B. Fu, N. Damer et al.

Computer Vision and Image Understanding 251 (2025) 104261

Fig. 7. Gender and time-of-day distribution in UAV-Human dataset.

9. Future directions

into account the balance between computational efficiency and model
performance when developing lightweight algorithms. Numerous studies have endeavored to tackle this challenge through the proposition of
lightweight person ReID models and the optimization of algorithms to
enable real-time processing (Zhou et al., 2019; Ke et al., 2021).
Additionally, Wang et al. (2019) have successfully devised an unsupervised face ReID system that operates in real-time. This system has
demonstrated impressive accuracy rates of 93.55 percent and 90.41
percent on diverse datasets, all while maintaining efficient processing speeds. This research provides effective strategies for achieving
real-time processing while maintaining accuracy in ReID tasks.
In their study, Cao et al. (2022) employed the approach of
algorithm-hardware co-design to achieve compression of the structure
and optimization of the operator for a person ReID network with
reduced weight. This work led to the development of a novel network
model called Shift Gaussian Convolution Network (SGCNet). This approach is in line with the goal of creating efficient algorithms that are
suitable for UAVs with limited resources. In addition, Neff et al. (2019)
were able to attain a pedestrian ReID accuracy of 74.8 percent using a
real-time edge video analytics system. This showcases the practicality
of real-time processing while maintaining a high level of accuracy.
This research offers valuable insights into the advancement of real-time
person ReID systems specifically designed for UAVs.

9.1. Diverse datasets
Advancing the development of UAV-based person ReID requires
an understanding of strategic approaches. An essential factor entails
expanding and improving ReID databases created specifically using
UAV images. The datasets should include a wide range of situations that
cover difficult settings, such as different lighting conditions, various
views, and objects obstructing the vision. By highlighting the integration of these complexities into the dataset, a thorough assessment of
algorithmic performance can be conducted, promoting improvements
in both reliability and accuracy. To advance the field, the focus
should be placed on diversifying datasets, innovating algorithms, using
standardized evaluation methods, and exploring the integration of
multi-model data. These efforts are expected to strengthen and make
person ReID frameworks more adaptable and robust in the context of
UAV applications.
The availability of diverse and comprehensive datasets is essential
for the development of robust and generalized ReID algorithms that
can effectively handle the complexities of real-world scenarios. Furthermore, the establishment of standardized evaluation metrics and
protocols will facilitate the comparison and validation of different
approaches, promoting the advancement of this exciting and rapidly
growing field of research.
Scale and Resolution Variations: The challenges of person ReID
using UAVs are primarily related to scaling and resolution variations.
To address these challenges, advanced methods and techniques are
required. One potential solution for UAV person ReID using a deep
learning network that uses residual blocks. This approach aims to tackle
issues of lower resolution and deformation conditions in aerial images
of people, which are frequently encountered in UAV surveillance scenarios. The proposed solution utilizes deep learning techniques and
residual blocks to enhance the reliability of person ReID in scenarios
involving scaling and resolution variations. Another significant solution
is the application of meta-transfer learning to generalize UAV-based
person ReID. Meta-transfer learning shows potential in adjusting models trained on high-resolution data to low-resolution aerial images. This
approach helps overcome the difficulties related to scaling and variations in resolution. It can contribute to the advancement of methods
that can efficiently manage variations in image quality and scale, which
are commonly encountered in scenarios involving the identification of
individuals using UAVs.

9.3. Domain adaptation and unsupervised approaches
Static ReID datasets or ground-based person ReID datasets, such
as Market-1501, CUHK03, and DukeMTMC-reID, are considered rich
sources of data for training person ReID models. These datasets offer
diverse scenarios and substantial amounts of annotated data, which can
be leveraged to develop domain adaptation techniques for UAV-based
person ReID. As UAV technology continues to evolve, it is believed
that utilizing existing ground-based person ReID datasets to train and
develop UAV-based person ReID models presents a promising approach
to overcoming the challenges posed by UAV person ReID.
Additionally, it is necessary to explore alternative innovative approaches that can effectively address the unique issues associated with
person ReID in aerial settings. The effort involves creating sophisticated
algorithms that are skilled at reducing the influence of environmental
factors such as changes in lighting and obstructions. This, in turn,
improves the dependability and effectiveness of person ReID systems
can be improved.

9.2. Lightweight and efficient algorithms

9.4. Multi-model fusion

In order to achieve real-time processing of person ReID on resourceconstrained UAVs without sacrificing accuracy, it is imperative to take

In order to improve the reliability of person ReID in UAV systems,
it is crucial that several visual characteristics, such as appearance,
14


codex
我已经补读了 UAV 综述的方法谱系和结果讨论。最后再对 `TIENet` 的消融部分补一小段，确保“证据闭环”不是只靠介绍里的可视化。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 10 'TIENet - A Tri-Interaction Enhancement Network for Multimodal Person Reidentification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
YANG et al.: TIENet: A TRI-INTERACTION ENHANCEMENT NETWORK FOR MULTIMODAL PERSON ReID

9859

TABLE II
C OMPARISON W ITH THE S TATE - OF - THE -A RT M ETHODS ON RGBNT201 AND MARKET1501 RGBNT. ∗ I NDICATES O UR I MPLEMENTATION . T HE B EST
I S M ARKED IN B OLD B LACK AND THE S ECOND -B EST I S IN U NDERLINE

TABLE III
A BLATION E XPERIMENTS FOR E ACH C OMPONENT OF O UR N ETWORK ON
RGBNT201

TABLE IV
A BLATION S TUDY OF P OOLING M ETHODS IN SFI ON RGBNT201 AND
M ARKET 1501 RGBNT

Fig. 6. t-SNE maps of features for different pooling methods in SFI.
(a) Original features. (b) Features enhanced using GAP (our method).
(c) Features enhanced using GMP.
TABLE V
A BLATION

implements intermodal interaction and fully fuses intermodal
complementary information; the IMFF module utilizes location attention to achieve intramodal feature fusion, making
full use of intramodal contextual structural information; MAS
enhances feature complementarity and robustness by controlling intermodal and intramodal feature distributions; and
finally, we validate the effectiveness of the dual data flows
by sequentially introducing the spatial data flow (SFI S) and
the frequency data flow (SFI F), thereby demonstrating the
necessity of these dual data flows for fine-grained feature
enhancement in both the spatial and frequency domains.
2) Further Explanation of SFI: To enhance the guidance
of amplitude features on the phase, we apply GAP to the
amplitude features and utilize softmax to generate channel
attention weights for each channel of the phase features.
As shown in Table IV, we replace GAP with global max
pooling (GMP), and the results demonstrate that GMP does
not consider global information, which negatively impacts the
model. To more intuitively showcase the advantages of our

E XPERIMENTS FOR I NTERMODAL
RGBNT201 ( IN %)

I NTERACTION

ON

method, we plot the t-SNE feature maps using the approach
described in Section I, as shown in Fig. 6. Fig. 6(a) represents
the original features, Fig. 6(b) represents the features enhanced
using GAP (our method), and Fig. 6(c) represents the features
enhanced using GMP. It is evident that, compared to Fig. 6(b),
Fig. 6(c) does not effectively distinguish between different
modalities, thus failing to fully utilize the global information
of the amplitude for feature enhancement.
3) Superiority of IMMI: To further validate the superiority of our intermodal interaction approach, we replace the
IMMI with the intermodal feature summation (SUM) and
the cross-modal interacting module (CIM) [37] in our model.
In addition, in the IMMI module, we replace the two-stage
progressive interaction with the single-stage interaction, defining this operation as single-stage IMMI (I) and multistage
interactions (II-IV). The experimental results are presented in
Table V. It can be seen that our method outperforms the others;
this is because the simple summation introduces too much
information that destroys the original modal features. CIM
utilizes the channel attention to filter the important features

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.

9860

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 6, JUNE 2025

TABLE VII
A BLATION E XPERIMENTS FOR L OSS F UNCTION ON RGBNT201

Fig. 7. IMMI in different locations in RestNet50. Layer0–4 represent
block0–4 of RestNet50.
TABLE VI
I MPACT OF D IFFERENT P OSITIONS OF THE IMMI M ODULE

Fig. 9. Parameter sensitivity analysis. (a) Impact of m. (b) Impact of δ.

Fig. 8. Impact of different positions of the SFI module. Layer0–4 represent
block0–4 of RestNet50.

but ignores the intermodal relationships. Single-stage IMMI
considers intermodal relationships but the fusion between
modalities is not sufficient, leading to the loss of specific information in the shared network. Moreover, having more stages
causes confusion of modal information and adds pressure on
computation and memory.
4) Location of IMMI and SFI: First, we explore the effect
of IMMI location on the model. There are several scenarios
for the location of the IMMI in the backbone RestNet50, as
shown in Fig. 7; note that for simplicity, we have removed the
SFI as well as the IMFI module. As shown in Table VI, we
insert the IMMI module after layer n (where n = 0, 2, and 4).
It can be seen that, although the model accuracy fluctuates
with changes in the IMMI position, the variations are not
significant. What is more, using a shared network not only
projects heterogeneous images into the same space, but also
reduces the number of parameters and improves the training
efficiency. The experiments prove that modal interaction before
extracting the features using the shared network does not have
a great impact on the model accuracy. Therefore, we chose to
extract the three modal features using the shared network after
layer0.
For the SFI module, we first embed it after the modalindependent network layer0 to enhance modal-specific features
with amplitude-guided phases. In order to locate the SFI
module position in the subsequent network, we insert SFI

after layer1–layer4, respectively, and the experimental results
are shown in Fig. 8. It can be seen that when the SFI
module is inserted into the subsequent network, the performance of the model decreases as the network layers get
deeper. This is because the shallow network extracts pedestrian
representation information while the deep network extracts
pedestrian semantic information. Guiding the phase with
amplitude in the deep network will break the overall semantic
continuity and lead to a significant performance degradation.
5) Necessity of MAS Loss: As shown in Table VII, the
removal of the MAS loss LMAS = Linter + δLintra has a significant impact on the model, thus resulting in a decrease of 8.83%
in mAP and 11.94% in Rank1 accuracy. Linter is favorable to
the model because it broadens the intermodal distance so that
complementary features are preserved. In addition, with the
introduction of the loss Lintra , the accuracy of mAP and Rank1
rises significantly. This is because Lintra considers adverse
factors such as pose occlusion and aligns with the motivation
of multimodal ReID and the dataset settings (e.g., complex
weather and unfavorable lighting conditions). This approach
enhances the robustness of person features to environmental
changes.
F. Parameter Sensitivity Analysis
1) Impact of m: m is the margin between intermodal centers, which controls the distribution of specific-shared features
between modalities. We explore the effect of different m, and
the experimental results are shown in Fig. 9(a). When m is
too small, the modalities lose their characteristics after going
through the previous networks, and the multimodal feature
cannot provide complementary information. When m is too
large, the modalities lose too many shared features, resulting in
a lack of discriminative information in the feature expression.
Therefore, we finally take m to be 1.
2) Impact of δ: δ is the weight of the intramodal aggregation loss function. We explore the effect of different δ on the
model performance, and the results are shown in Fig. 9(b).
It can be seen that the model performance fluctuates with a
change in δ and gets optimal when δ takes 1.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.

YANG et al.: TIENet: A TRI-INTERACTION ENHANCEMENT NETWORK FOR MULTIMODAL PERSON ReID

9861

Fig. 10. Distribution of distances for positive/negative sample pairs (the same ID features/different ID features). The black straight lines indicate the respective
mean distances, and δ indicates the difference between the mean distances of the positive and negative samples. Our method reduces the distance between
positive sample pairs and increases the negative sample pair distance compared to baseline. (a) Baseline. (b) TIENet.

Fig. 11. t-SNE results of the intersample feature distribution of RGBNT201. On the left is baseline, and on the right is our method. Different colors represent
different identities in RGBNT201. (a) Baseline. (b) TIENet.

Fig. 12. Top10 visualization of RGB query image retrieval results in RGBNT201. (a) Baseline method. (b) Our method. On the left is the query image, and
the next 10 are the search results inside each image. The correct ones are boxed in green, and the wrong ones in red.

G. Visualization Analysis
To statistically reflect the effectiveness of our method, we
randomly select 1000 pairs of positive and negative samples,
calculate the distances between them, and plot probability
distribution histograms. As shown in Fig. 10, compared to the
baseline, our method reduces the average distance between
positive sample pairs and increases the distance between
negative sample pairs, further demonstrating that our method
effectively enhances interperson diversity.
Furthermore, as shown in Fig. 11, we use t-SNE [51] to
visualize and analyze the distribution of 15 random sample
features in RGBNT201. It should be noted that each ID in the

dataset contains samples from two cameras with significantly
different resolutions, resulting in each ID’s features being
divided into two parts. It can be observed that compared
to the baseline, our method narrows the intraclass feature
distribution within the same ID, making the features more
compact; between different IDs, our method clarifies the
interclass boundaries. Therefore, it can be demonstrated that
our method effectively leverages multimodal information and
fully utilizes it to enhance the model’s performance.
At last, we randomly select four identities and show the top
ten images of the retrieved images, where correct retrievals are
indicated by green boxes and incorrect retrievals are indicated

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:13:12 UTC from IEEE Xplore. Restrictions apply.


codex
我基于本地 PDF 的方法和补读页拆解。判断重点放在“观察怎样变成问题”，不是复述模块。

**1. TIENet**

1. 触发观察：多模态 ReID 不是缺一个融合层，而是现有方法大多只在空间域融合，忽略频域差异和模态间关系。作者用傅里叶可视化发现，相位主要保留结构，幅度更能区分模态和身份。

2. 重定义动作：他们把“多模态融合”重定义为“三类交互不足”。关键词是频域和空间域交互、模态间交互、模态内人体结构交互。这样旧问题从“怎么拼 RGB、近红外、热红外”变成“怎么分别保留模态特异性和互补性”。

3. 机制怎么长出来：SFI 用幅度注意力引导相位，再用空间卷积保留局部细节；IMMI 用相似度掩码做自适应模态交互；IMFF 用位置注意力建模身体局部上下文；MAS loss 同时做模态间分离和模态内聚合。SFI 和观察绑定最紧，IMFF 更像补强模块，逻辑稍泛。

4. 证据闭环：关键不是最终涨点，而是频域重构和 t-SNE 先证明幅度、相位确实承载不同信息；GAP 和 GMP 替换证明“幅度全局信息”有用；IMMI 对比 SUM、CIM 和不同交互阶段证明固定融合会引入冲突；MAS removal、距离分布、t-SNE 和检索可视化补足“特异性和互补性同时改善”。

5. reviewer 为什么买账：这篇卖的是“多模态融合不只是拼接，而是频域、模态关系、人体结构三重交互”的视角，加上频域观察让模块堆叠看起来有根。真正的新颖性来自频域解释和三重交互框架，不是单个注意力模块。

**2. Towards Anytime Retrieval**

1. 触发观察：真实系统要求任何时间都能检索，包括白天、夜晚、短期、长期换衣。但传统 ReID、可见红外 ReID、换衣 ReID 被拆成不同任务，数据集也各自只覆盖一部分场景。

2. 重定义动作：作者把 ReID 从单一任务重定义为 Anytime ReID，用“时间点”和“时间间隔”划出六个场景。关键词是 anytime retrieval、multi-scenario retrieval、scenario-specific cues、inter-scenario interference。

3. 机制怎么长出来：既然六个场景依赖的线索不同，一个统一特征会互相伤害，所以 MS-ReID 用六个 CLS token 学不同场景特征；短期用衣服类别监督，长期用身份监督；MoAE 按白天、夜晚、全天、短期、长期这些属性共享专家；HDW 按任务难度动态加权。机制基本是从任务重定义自然推出的。

4. 证据闭环：数据集 AT-USTC 本身证明旧数据集缺口，21 个月、RGB 和红外、多衣服、多季节支撑“anytime”成立。模型证据包括六场景分别评测、跨数据集泛化、MS-ReID、MoAE、HDW 消融，以及和其他 MoE 方法对比，证明不是单纯多头分类。

5. reviewer 为什么买账：这篇主要卖问题和 benchmark，机制是配套证明新任务可做。新颖性核心是把三个割裂子任务合成一个时间驱动的统一任务空间。

**3. TP-LReID**

1. 触发观察：终身 ReID 主要解决旧域遗忘，但现实还要求未来未见域泛化。作者观察到普通 ReID 当前域强，终身 ReID 旧域强，泛化 ReID 未来域强，三者各管一段；同时文本特征比图像特征更跨域一致。

2. 重定义动作：他们把 LReID 从“防止忘记过去”重定义为“过去、现在、未来三类域同时负责”。关键词是 text prompts、domain-consistent features、future-style generation、past-present-future domains。

3. 机制怎么长出来：文本更稳定，所以用文本 prompt 作为跨域语义锚；域按顺序到来，所以 prompt generator 和 image encoder 交替、连续更新；未来域不可见，所以在特征空间用风格对抗扰动模拟未来样本，再让文本和原图、未来风格图一起对齐。逻辑比较紧。

4. 证据闭环：它不只报平均成绩，而是分 past、present、future 比较普通 ReID、终身 ReID、泛化 ReID；消融分别看文本 prompt 和未来扰动；比较 continual prompt 和非 continual prompt；还跑 24 种到达顺序，看第一个旧域遗忘和所有未见域泛化。

5. reviewer 为什么买账：这篇卖的是视角，终身 ReID 不应只防遗忘，还要负责未来域。机制里的 CLIP、prompt、StyleAdv 都不是全新零件，但组合被“过去、现在、未来”这个新定义绑住了。

**4. Two-stage knowledge distillation**

1. 触发观察：可见红外 ReID 的主流做法急着做跨模态对齐，但同模态差异本来比跨模态小很多。直接多对多跨模态对齐既难，又依赖完整标注。

2. 重定义动作：作者把“跨模态 gap 太大”重定义为“对齐顺序错了”。关键词是 simple-to-difficult strategy、intra-modality discrepancy、self-mimic learning、mutual distillation、weak supervision。

3. 机制怎么长出来：先用少量标注训练分类器，给无标注样本生成带置信度的软身份标签；再用 self-mimic 把同一身份在同一模态内拉向模态特定原型；最后用 mutual distillation 做可见和红外的双向分布对齐。先同模态收紧，再跨模态对齐，机制和重定义绑定很紧。

4. 证据闭环：关键消融是 SI、SM、MD 在全监督和弱监督下逐项加入；还比较离线原型、在线原型、center loss 和 self-mimic；可视化同模态距离均值和方差下降；比较红外到可见、可见到红外、双向蒸馏，证明单向对齐会损失信息；伪标签准确率也单独验证。

5. reviewer 为什么买账：这篇卖的是机制加顺序视角。蒸馏本身不新，但“先易后难，把多对多跨模态对齐拆成同模态原型收紧和跨模态分布互蒸馏”是它的真正新意。

**5. UAV-based person re-identification survey**

1. 触发观察：大多数 ReID 研究默认固定地面摄像头，但 UAV 有高空视角、动态背景、运动模糊、低分辨率、续航和算力约束。作者还指出 UAV-ReID 数据集非常少，和普通 ReID 数据集规模不成比例。

2. 重定义动作：这篇不是方法论文，它把 UAV-ReID 从“普通 ReID 的一个数据集变体”重定义成独立问题域。关键词是 aerial perspective、dynamic background、motion blur、resource constraints、aerial-ground gap、privacy concerns。

3. 机制怎么长出来：作为综述，它的“机制”是分类框架。它按数据集、挑战、方法谱系、结果和未来方向组织领域，把 MRP、DRHIT01、P-DESTRE、PRAI-1581、AHU-UAV、UAV-Human、AG-ReID 等串起来，再把方法从 ELF、SVM、triplet loss、CNN、subspace pooling、federated learning、ViT、elevated-view attention 统一归档。

4. 证据闭环：它用数据集表证明资源稀缺，用方法表证明现有技术大多迁移自地面 ReID，用结果表证明不同 UAV 场景难度差异很大。例如 PRAI-1581 的绝对精度仍低，P-DESTRE 短期较稳但长期明显掉，AG-ReID 的 aerial-ground 设置需要专门的 elevated-view attention。

5. reviewer 为什么买账：这篇卖的是领域地图和问题边界，而不是算法。它让 reviewer 相信 UAV-ReID 不是普通 ReID 换相机，而是数据、视角、部署约束和伦理问题共同定义的新应用域。

**6. Uniform Light Transformer**

1. 触发观察：已有复杂光照 ReID 用一对一光照转换器，随着光照种类增加会变得冗余。作者问一个反直觉问题：能不能只用一个统一光照转换器处理任意光照？实验发现视觉上转换还行，但低频信息差异随光照尺度增加而变大，检索效果受损。

2. 重定义动作：他们把问题从“光照归一化不够好”重定义为“统一光照转换器在多光照输入下低频建模能力不足”。关键词是 Unified Lighting Transformer、low-frequency information、discriminative feature spectrum consistency、low-frequency information constraints。

3. 机制怎么长出来：既然问题被定位到低频信息，方法就围绕低频约束展开。DFSC 用谱约束网络把 ReID 特征图映射成真实判别特征，约束生成图的判别频谱；LFCM 把生成图低频和真实图高频拼成虚拟样本，再用预训练 ReID 模型的分类 logits 做蒸馏，让低频差异在身份特征层面被惩罚；总损失是 GAN 生成损失、DFSC、LFCM 三部分。

4. 证据闭环：最关键的是图 1 的频域分析，说明高频差异小且稳定，低频差异随光照尺度增加而扩大。消融里单独 ULT 甚至不如强 ReID baseline，说明“一个转换器”不是自然有效；加入 DFSC、LFCM 都提升，合起来最好，证明重定义的低频瓶颈确实被命中。

5. reviewer 为什么买账：这篇卖的是诊断视角加机制。它不是又做一个 GAN，而是先解释为什么统一光照转换失败，再提出两个专门约束低频的证据化模块。

这批论文共同套路很明显：先抓一个具体失败或缺口，不急着说“我加了模块”，而是把旧问题改写成更尖锐的新问题。然后机制要从新定义里自然长出来，证据也要证明这个新定义成立，而不是只证明最终 mAP 涨了。对我们最有用的是，创新最好从“任务边界、干扰来源、训练顺序、信息载体”里长出来，模块只是这个判断的执行形式。
tokens used
212,193
我基于本地 PDF 的方法和补读页拆解。判断重点放在“观察怎样变成问题”，不是复述模块。

**1. TIENet**

1. 触发观察：多模态 ReID 不是缺一个融合层，而是现有方法大多只在空间域融合，忽略频域差异和模态间关系。作者用傅里叶可视化发现，相位主要保留结构，幅度更能区分模态和身份。

2. 重定义动作：他们把“多模态融合”重定义为“三类交互不足”。关键词是频域和空间域交互、模态间交互、模态内人体结构交互。这样旧问题从“怎么拼 RGB、近红外、热红外”变成“怎么分别保留模态特异性和互补性”。

3. 机制怎么长出来：SFI 用幅度注意力引导相位，再用空间卷积保留局部细节；IMMI 用相似度掩码做自适应模态交互；IMFF 用位置注意力建模身体局部上下文；MAS loss 同时做模态间分离和模态内聚合。SFI 和观察绑定最紧，IMFF 更像补强模块，逻辑稍泛。

4. 证据闭环：关键不是最终涨点，而是频域重构和 t-SNE 先证明幅度、相位确实承载不同信息；GAP 和 GMP 替换证明“幅度全局信息”有用；IMMI 对比 SUM、CIM 和不同交互阶段证明固定融合会引入冲突；MAS removal、距离分布、t-SNE 和检索可视化补足“特异性和互补性同时改善”。

5. reviewer 为什么买账：这篇卖的是“多模态融合不只是拼接，而是频域、模态关系、人体结构三重交互”的视角，加上频域观察让模块堆叠看起来有根。真正的新颖性来自频域解释和三重交互框架，不是单个注意力模块。

**2. Towards Anytime Retrieval**

1. 触发观察：真实系统要求任何时间都能检索，包括白天、夜晚、短期、长期换衣。但传统 ReID、可见红外 ReID、换衣 ReID 被拆成不同任务，数据集也各自只覆盖一部分场景。

2. 重定义动作：作者把 ReID 从单一任务重定义为 Anytime ReID，用“时间点”和“时间间隔”划出六个场景。关键词是 anytime retrieval、multi-scenario retrieval、scenario-specific cues、inter-scenario interference。

3. 机制怎么长出来：既然六个场景依赖的线索不同，一个统一特征会互相伤害，所以 MS-ReID 用六个 CLS token 学不同场景特征；短期用衣服类别监督，长期用身份监督；MoAE 按白天、夜晚、全天、短期、长期这些属性共享专家；HDW 按任务难度动态加权。机制基本是从任务重定义自然推出的。

4. 证据闭环：数据集 AT-USTC 本身证明旧数据集缺口，21 个月、RGB 和红外、多衣服、多季节支撑“anytime”成立。模型证据包括六场景分别评测、跨数据集泛化、MS-ReID、MoAE、HDW 消融，以及和其他 MoE 方法对比，证明不是单纯多头分类。

5. reviewer 为什么买账：这篇主要卖问题和 benchmark，机制是配套证明新任务可做。新颖性核心是把三个割裂子任务合成一个时间驱动的统一任务空间。

**3. TP-LReID**

1. 触发观察：终身 ReID 主要解决旧域遗忘，但现实还要求未来未见域泛化。作者观察到普通 ReID 当前域强，终身 ReID 旧域强，泛化 ReID 未来域强，三者各管一段；同时文本特征比图像特征更跨域一致。

2. 重定义动作：他们把 LReID 从“防止忘记过去”重定义为“过去、现在、未来三类域同时负责”。关键词是 text prompts、domain-consistent features、future-style generation、past-present-future domains。

3. 机制怎么长出来：文本更稳定，所以用文本 prompt 作为跨域语义锚；域按顺序到来，所以 prompt generator 和 image encoder 交替、连续更新；未来域不可见，所以在特征空间用风格对抗扰动模拟未来样本，再让文本和原图、未来风格图一起对齐。逻辑比较紧。

4. 证据闭环：它不只报平均成绩，而是分 past、present、future 比较普通 ReID、终身 ReID、泛化 ReID；消融分别看文本 prompt 和未来扰动；比较 continual prompt 和非 continual prompt；还跑 24 种到达顺序，看第一个旧域遗忘和所有未见域泛化。

5. reviewer 为什么买账：这篇卖的是视角，终身 ReID 不应只防遗忘，还要负责未来域。机制里的 CLIP、prompt、StyleAdv 都不是全新零件，但组合被“过去、现在、未来”这个新定义绑住了。

**4. Two-stage knowledge distillation**

1. 触发观察：可见红外 ReID 的主流做法急着做跨模态对齐，但同模态差异本来比跨模态小很多。直接多对多跨模态对齐既难，又依赖完整标注。

2. 重定义动作：作者把“跨模态 gap 太大”重定义为“对齐顺序错了”。关键词是 simple-to-difficult strategy、intra-modality discrepancy、self-mimic learning、mutual distillation、weak supervision。

3. 机制怎么长出来：先用少量标注训练分类器，给无标注样本生成带置信度的软身份标签；再用 self-mimic 把同一身份在同一模态内拉向模态特定原型；最后用 mutual distillation 做可见和红外的双向分布对齐。先同模态收紧，再跨模态对齐，机制和重定义绑定很紧。

4. 证据闭环：关键消融是 SI、SM、MD 在全监督和弱监督下逐项加入；还比较离线原型、在线原型、center loss 和 self-mimic；可视化同模态距离均值和方差下降；比较红外到可见、可见到红外、双向蒸馏，证明单向对齐会损失信息；伪标签准确率也单独验证。

5. reviewer 为什么买账：这篇卖的是机制加顺序视角。蒸馏本身不新，但“先易后难，把多对多跨模态对齐拆成同模态原型收紧和跨模态分布互蒸馏”是它的真正新意。

**5. UAV-based person re-identification survey**

1. 触发观察：大多数 ReID 研究默认固定地面摄像头，但 UAV 有高空视角、动态背景、运动模糊、低分辨率、续航和算力约束。作者还指出 UAV-ReID 数据集非常少，和普通 ReID 数据集规模不成比例。

2. 重定义动作：这篇不是方法论文，它把 UAV-ReID 从“普通 ReID 的一个数据集变体”重定义成独立问题域。关键词是 aerial perspective、dynamic background、motion blur、resource constraints、aerial-ground gap、privacy concerns。

3. 机制怎么长出来：作为综述，它的“机制”是分类框架。它按数据集、挑战、方法谱系、结果和未来方向组织领域，把 MRP、DRHIT01、P-DESTRE、PRAI-1581、AHU-UAV、UAV-Human、AG-ReID 等串起来，再把方法从 ELF、SVM、triplet loss、CNN、subspace pooling、federated learning、ViT、elevated-view attention 统一归档。

4. 证据闭环：它用数据集表证明资源稀缺，用方法表证明现有技术大多迁移自地面 ReID，用结果表证明不同 UAV 场景难度差异很大。例如 PRAI-1581 的绝对精度仍低，P-DESTRE 短期较稳但长期明显掉，AG-ReID 的 aerial-ground 设置需要专门的 elevated-view attention。

5. reviewer 为什么买账：这篇卖的是领域地图和问题边界，而不是算法。它让 reviewer 相信 UAV-ReID 不是普通 ReID 换相机，而是数据、视角、部署约束和伦理问题共同定义的新应用域。

**6. Uniform Light Transformer**

1. 触发观察：已有复杂光照 ReID 用一对一光照转换器，随着光照种类增加会变得冗余。作者问一个反直觉问题：能不能只用一个统一光照转换器处理任意光照？实验发现视觉上转换还行，但低频信息差异随光照尺度增加而变大，检索效果受损。

2. 重定义动作：他们把问题从“光照归一化不够好”重定义为“统一光照转换器在多光照输入下低频建模能力不足”。关键词是 Unified Lighting Transformer、low-frequency information、discriminative feature spectrum consistency、low-frequency information constraints。

3. 机制怎么长出来：既然问题被定位到低频信息，方法就围绕低频约束展开。DFSC 用谱约束网络把 ReID 特征图映射成真实判别特征，约束生成图的判别频谱；LFCM 把生成图低频和真实图高频拼成虚拟样本，再用预训练 ReID 模型的分类 logits 做蒸馏，让低频差异在身份特征层面被惩罚；总损失是 GAN 生成损失、DFSC、LFCM 三部分。

4. 证据闭环：最关键的是图 1 的频域分析，说明高频差异小且稳定，低频差异随光照尺度增加而扩大。消融里单独 ULT 甚至不如强 ReID baseline，说明“一个转换器”不是自然有效；加入 DFSC、LFCM 都提升，合起来最好，证明重定义的低频瓶颈确实被命中。

5. reviewer 为什么买账：这篇卖的是诊断视角加机制。它不是又做一个 GAN，而是先解释为什么统一光照转换失败，再提出两个专门约束低频的证据化模块。

这批论文共同套路很明显：先抓一个具体失败或缺口，不急着说“我加了模块”，而是把旧问题改写成更尖锐的新问题。然后机制要从新定义里自然长出来，证据也要证明这个新定义成立，而不是只证明最终 mAP 涨了。对我们最有用的是，创新最好从“任务边界、干扰来源、训练顺序、信息载体”里长出来，模块只是这个判断的执行形式。
