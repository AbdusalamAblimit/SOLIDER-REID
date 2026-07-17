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
session id: 019ef836-16ed-7f60-a19b-64398efcde4d
--------
user
你是 ReID 创新方法论拆解员 27。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(27)==
Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf
Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf
Unsupervised person re-identification via camera-aware multi-level label refinement.pdf
Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf
Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我先按项目规则确认目标文件和论文文件是否在当前可读范围内，然后用 `pdftotext` 直接抽取方法部分。当前沙箱是只读的，所以我只会读文件和检索，不会改任何项目记录。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc 'ls -1' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "pwd && rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'STATUS.md' -g 'AGENTS.md' .. ../.. ../../.. 2>/dev/null" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
../../../CLAUDE.md
../../../AGENTS.md
../../../GOAL.md
../../STATUS.md
../../../experiments/STATUS.md

codex
我能读到项目目标和状态文件。先快速确认这些规则文件，避免后面把论文拆解写偏成实验结论或模块复刻。
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
/bin/zsh -lc "sed -n '1,220p' ../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,220p' ../../../AGENTS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


codex
我已经确认当前任务属于文献方法论拆解，不需要启动实验。接下来我会逐篇抽取前 8 页，并重点读 `method`、`approach` 或对应小节，而不是只看摘要和引言。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Unsupervised Lifelong Person Re-Identification via Affinity
Harmonization
JICAN TAN, JINJIA PENG, and SONGYU ZHANG, School of Cyber Security and Computer, Hebei
University, Baoding, China
ZHEN WANG, Tianjin University of Finance and Economics, Tianjin, China
HUIBING WANG, Information Science and Technology College, Dalian Maritime University, Dalian,
China
Lifelong Person Re-Identification (LReID) seeks to continuously train models across multiple target domains,
enabling effective generalization in both known and unseen domains. Achieving a balance between “plasticity”
(the ability to adapt to new knowledge) and “stability” (the capacity to prevent forgetting) is crucial in
lifelong learning. However, most existing LReID methods primarily focus on enhancing model stability or
plasticity, often neglecting the critical balance between them. Moreover, current LReID approaches largely
rely on supervised learning, which necessitates large-scale pre-labeled datasets—a process that is both timeconsuming and labor-intensive in practical applications. To address these challenges, this article proposes
an Unsupervised LReID approach called the Affinity Harmonization Network (AHN). AHN includes an Old
Domain Affinity Constraint (ODAC) module, which builds an expert model for the old domain to provide
affinity relationships as references. This helps limit changes among old representations, enabling the model
to integrate new knowledge while preserving compatibility with previous representations. To harmonize
stability and plasticity while guiding the model in acquiring new knowledge, AHN incorporates a Current
Domain Affinity Guidance (CDAG) module. This module builds an expert model for the new domain and
uses the generated affinity relationships to assist in training the model. Furthermore, this article proposes
the Old Domain Intra-class Variance Constraint (OIVC) module, which mitigates potential deviations in
the intra-class variance of legacy samples by limiting the distance between replay samples and old domain
camera prototypes. Extensive experiments demonstrate that our method achieves significant performance
improvements over existing unsupervised lifelong ReID methods, with an average gain of 5.3% in mAP and
5.2% in Rank-1 accuracy.
CCS Concepts: • Computing methodologies → Computer vision tasks; Unsupervised learning; Lifelong
machine learning; Object identification;

This work was supported in part by the National Natural Science Foundation of China Grant (62501226, 62576067),
National Key Research and Development Program of China Grant (2024YFB4710800), Natural Science Foundation of Hebei
Province (F2025201037), Basic Research Project of Shijiazhuang Municipal Universities in Hebei Province (241791387A), and
Interdisciplinary Research Program of Hebei University (DXK202404).
Authors’ Contact Information: Jican Tan, School of Cyber Security and Computer, Hebei University, Baoding, China;
e-mail: tanjican@163.com; Jinjia Peng (corresponding author), School of Cyber Security and Computer, Hebei University,
Baoding, China; e-mail: pengjinjia@hbu.edu.cn; Songyu Zhang, School of Cyber Security and Computer, Hebei University,
Baoding, China; e-mail: zhangsongyu@stumail.hbu.edu.cn; Zhen Wang, Tianjin University of Finance and Economics,
Tianjin, China; e-mail: wangzhen@tjufe.edu.cn; Huibing Wang (corresponding author), Information Science and Technology
College, Dalian Maritime University, Dalian, China; e-mail: huibing.wang@dlmu.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2026/3-ART103
https://doi.org/10.1145/3779124
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 103. Publication date: March 2026.

103:2

J. Tan et al.

Additional Key Words and Phrases: Person Re-identification, Affinity Constraints, Knowledge balance, Unsupervised lifelong learning, Intra-class Variance Constraint
ACM Reference format:
Jican Tan, Jinjia Peng, Songyu Zhang, Zhen Wang, and Huibing Wang. 2026. Unsupervised Lifelong Person
Re-Identification via Affinity Harmonization. ACM Trans. Multimedia Comput. Commun. Appl. 22, 4, Article 103
(March 2026), 22 pages.
https://doi.org/10.1145/3779124

1

Introduction

The primary objective of person Re-Identification (ReID) is to recognize and distinguish pedestrian images taken by different cameras or at different times by the same camera. However, numerous
studies [21, 41, 50, 53] have demonstrated that re-identification models trained on a single target
domain often perform poorly when faced with continuously emerging unseen domains. To address
this issue, increasing attention has been directed toward Lifelong Person ReID (LReID) [28, 42].
Unlike standard lifelong learning, LReID places greater emphasis on cross-domain Person ReID
and similarity management, while traditional lifelong learning primarily focuses on broader task
learning and mitigating forgetting.
In recent years, most LReID approaches [29, 30, 36] have been based on supervised cross-domain
training, where manual annotation of new sample data is required before deployment when learning
new task domains. Although these methods have led to strong model performance, obtaining fully
annotated datasets in real-world scenarios is often challenging, and the manual annotation process
is both time-consuming and labor-intensive. Therefore, research on unsupervised training methods
have become increasingly important. By utilizing unsupervised domain adaptation to replace
supervised cross-domain training, the effectiveness and applicability of LReID algorithms in realworld settings can be greatly enhanced. However, the lack of labeled samples in unsupervised
learning significantly increases the difficulty of model training. Consequently, in unsupervised
environments, achieving a balance between the plasticity (the capacity to acquire new knowledge)
of LReID models and their stability (the ability to retain previously acquired knowledge) has
emerged as a pivotal challenge in the development of robust and highly generalizable person ReID
models capable of effectively mitigating catastrophic forgetting.
One typical approach in current LReID research [9, 48] is to retain part of the data from previous
domains, such as exemplar samples or copies of the old model and its parameters, to facilitate new
learning. In this paradigm, old samples are used for replay, while the old model is frozen and serves
as a teacher for knowledge distillation, helping the current model preserve its memory of prior
knowledge. Several replay and distillation strategies [36, 47] have demonstrated strong effectiveness
in alleviating catastrophic forgetting. However, as illustrated in Figure 1, the main drawback of
these traditional methods is their overemphasis on stability: while they succeed in retaining old
knowledge, they provide limited guidance for acquiring and integrating new knowledge from
the current domain. As a result, the model often suffers from an imbalance between stability and
plasticity. To overcome this limitation, we propose a new framework that introduces two temporary
expert models during training: a frozen old domain expert to anchor prior knowledge (stability) and
a newly trained expert to guide the acquisition of current knowledge (plasticity). After training,
both experts are discarded, ensuring that the overall model capacity remains fixed. This design
explicitly balances stability and plasticity, enabling the model not only to retain previously learned
knowledge but also to effectively acquire new domain knowledge, which is consistent with natural
human learning processes.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 103. Publication date: March 2026.

Unsupervised Lifelong Person Re-Identification via Affinity Harmonization

103:3

Fig. 1. The illustration of our motivation. (a) Finetune: Without any forgetting-prevention design, it inevitably
leads to the conflict between different knowledge. (b) Traditional method: The frozen old model serves as a
teacher to help the main model retain prior knowledge, but this limits new knowledge acquisition. (c) By
contrast, our proposed method uses two expert models to balance stability and plasticity, enabling the model
to retain old knowledge while effectively learning new information.

Additionally, in LReID, images of the same individual may be captured by different cameras, and
factors such as time, weather, and lighting can result in stylistic variations. To mitigate the intra-class
variance caused by differences in camera perspectives and environmental changes, and to enhance
the model’s ability to recognize the same person across different cameras, camera prototype
contrastive loss [4] has been proposed. This method has proven highly effective, significantly
improving model performance. However, it still overlooks a critical issue: as new knowledge is
introduced, it may interfere with the intra-class variance of old samples, thereby diminishing the
model’s recognition capability for these older samples.
To achieve this, we propose a novel Affinity Harmonization Network (AHN) for Unsupervised
LReID, which can not only mitigate the interference of new knowledge on old knowledge by
constraining affinity relationships among old domain images but also guide the model’s adaptation
to the knowledge within the new domain. Specifically, AHN initially encodes transferable knowledge
in the form of a pairwise Relation Matrix (RM), with each element representing the affinity
between two samples within a mini-batch. Based on a pair of RMs generated from the old and new
LReID models, the Old Domain Affinity Constraint (ODAC) module is proposed. In this module,
by limiting changes in the RMs, the model is guided to update new knowledge in a way that remains
compatible with old knowledge, thereby reducing interference with previously acquired knowledge.
Furthermore, the Current Domain Affinity Guidance (CDAG) module is proposed to guide
the model’s learning of new domain knowledge. In this module, a new domain expert model is
constructed as an additional teacher to work with the main model, jointly generating an additional
pair of RMs. These RMs are then used to guide the learning of affinity relationships among images
in the new domain, helping the model to adapt more effectively to new knowledge. These two
modules work in harmony to achieve a balance between stability and plasticity. Additionally, to
tackle the challenge of increased intra-class variance in old samples, this article proposes the Old
Domain Intra-class Variance Constraint (OIVC) module, which further improves the model’s
ability to recognize old domain data by limiting the distance between replay samples and old
domain camera prototypes. The contributions of our article are summarized as follows:
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 103. Publication date: March 2026.

103:4

J. Tan et al.

— This article proposes the AHN for Unsupervised LReID, where the network framework employs
an Affinity Harmonization approach to perform domain-by-domain learning, achieving a
balance between plasticity and stability.
— The ODAC module and the CDAG module are proposed in AHN. The ODAC module builds
a high-stability knowledge base to help the model retain old knowledge while learning
new information, while the CDAG module establishes a high-plasticity knowledge base that
effectively guides the model in acquiring new knowledge. Together, these two modules enable
the model to achieve Affinity Harmonization during training.
— This article also proposes the OIVC module to mitigate performance degradation caused
by high intra-class variance in old domain image recognition. By constraining the intraclass variance of samples from previous domains, the model is able to retain a high level of
discriminative capability for identifying the same individual across different camera views,
even after continual learning over multiple tasks.
2

Related Works

Person ReID. Person ReID aims to accurately identify whether individuals in images captured
at different times and across different cameras are the same person [1, 20, 37]. In recent years,
significant research has been conducted in the field of Person ReID, with varying focuses depending
on the size of the training and test datasets and whether the samples are annotated. Supervised
Person ReID [26, 38, 39], where all training samples are labeled and the test samples share the
same distribution as the training set, has been the most actively researched area. In this context,
the primary focus has been on developing various network architectures, such as part-based
networks [35] and pyramid networks [14]. While supervised Person ReID methods [6, 25, 52]
perform well on large-scale datasets, they heavily rely on a vast amount of labeled data. Manual
data annotation is not only time-consuming but also prone to errors, which can significantly
degrade model performance if mislabeled samples are introduced. To address this issue, researchers
have proposed unsupervised domain adaptation [7, 10, 11, 27] and fully unsupervised Person ReID
methods [40] to adapt ReID models to target domains that contain unlabeled samples. However,
despite their promising performance on seen datasets, these methods often suffer significant
performance degradation when applied to new target domains, limiting the practical application of
current ReID models in real-world scenarios. In response to this challenge, this article explores the
task of unsupervised LReID, aiming to enable models to continuously learn from different datasets
without any labeled data. This approach allows the model to adapt to various scenarios, meeting the
demands of real-world applications. By addressing the limitations of current methods in handling
cross-domain tasks, unsupervised LReID provides a novel direction for improving the applicability
of ReID models in real-world environments.
LReID. Recent advances in LReID have focused on addressing catastrophic forgetting while
maintaining adaptability to new domains. Among them, LSTKC [45, 47] consolidates long shortterm knowledge by leveraging knowledge distillation to correct outdated representations, achieving
strong performance under the supervised lifelong setting. Building upon LSTKC, DASK [44]
introduces adaptive style rehearsal to dynamically model domain distributions, further enhancing
plasticity while preserving stability. More recently, AdalReID [31] employs lightweight domainspecific adapters built on a CLIP backbone, combined with an automatic selection mechanism, to
achieve modular expansion and inference flexibility. In the unsupervised setting, UCR [5] represents
the most relevant baseline, which leverages contrastive rehearsal with exemplar replay to mitigate
catastrophic forgetting. However, its unidirectional distillation and reliance on clustering-based
pseudo labels limit its ability to balance stability and plasticity across diverse domains. While
these approaches have achieved State-of-the-Art (SOTA) results, they still fail to strike a proper
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 103. Publication date: March 2026.

Unsupervised Lifelong Person Re-Identification via Affinity Harmonization

103:5

balance between stability and plasticity, which is crucial for lifelong learning. In contrast, our
work addresses the more challenging unsupervised lifelong scenario by introducing the AHN.
Instead of maintaining permanent domain-specific modules, AHN employs temporary experts with
affinity-based constraints, striking a balance between stability and plasticity without requiring
manual annotations or additional inference cost.
3

Problem Formulation

Unsupervised LReID assumes that a sequence of training datasets from diverse domains is provided
for incremental learning, with no labeled data accessible during the training process. Moreover,
data from other datasets is not directly accessible during the training process [28]. Specifically,
𝑡
given a stream of 𝑇 ReID datasets 𝐷 = {𝐷 1, . . . , 𝐷𝑇 }, where each 𝐷𝑡 = {𝑥𝑖 }𝑛𝑖=1
consists of 𝑛𝑡 images
𝑥𝑖 without identity labels. It is crucial to emphasize that the identities across different training
datasets are non-overlapping. Following the 𝑡th training step, the model obtained is represented
as 𝜃 𝑡 . This process continues iteratively, with the final model represented as 𝜃𝑇 . During testing,
to evaluate the model 𝜃𝑇 ’s ability to learn new knowledge and mitigate forgetting, it is tested
on the test datasets from all seen domains, 𝐷 test = {𝐷𝑡test }𝑇𝑡=1 . Additionally, to assess the model’s
generalization capability, it is evaluated on several unseen domains, 𝐷 un = {𝐷𝑡un }𝑈𝑡=1 . 𝑈 represents
the number of unseen domains.
4
4.1

Proposed Method
Overview

As illustrated in Algorithm 1, during the training process on domain 𝐷 1 , a SOTA fully unsupervised
ReID method is adopted in this article, with pseudo-label-based contrastive learning used as the
baseline. The momentum encoder is capable of generating more stable feature representations, and
its slower update rate facilitates better retention of previously learned representations, thereby
gradually accumulating knowledge from each observed domain. Therefore, the baseline method
integrates the strengths of both the online encoder and the momentum encoder. To retain the
learned knowledge within the momentum representation, the momentum encoder (with weights
denoted as 𝜃𝑚 ) is updated based on the exponential moving average of the online encoder weights
(denoted as 𝜃 ):
𝑡
𝑡 −1
𝜃𝑚
= 𝛼𝜃𝑚
+ (1 − 𝛼)𝜃 𝑡 ,

(1)

where the hyperparameter 𝛼 regulates the update rate of the momentum encoder, with 𝑡 and 𝑡 − 1
representing the current and preceding iterations, respectively. Due to the greater stability of the
momentum encoder, it is used as the feature extractor to obtain representations for all images in
the current domain. Subsequently, a density-based clustering algorithm, DBSCAN [8], is employed
to generate pseudo labels for each image. Based on the pseudo labels obtained from clustering, a
contrastive loss L𝑏𝑎𝑠𝑒 (described in Section 4.2) is constructed between the images and prototypes in
the current domain 𝐷 1 . To mitigate the forgetting of previously learned knowledge, AHN establishes
two memory components: one for storing images and the other for retaining camera prototypes
from previous domains. The method updates these memory buffers after completing the learning
on each domain.
The overall architecture of AHN during training on unlabeled samples from 𝐷𝑡 is illustrated in
Figure 2. (a) represents the backbone network: 𝜃 𝑡 and 𝜃𝑚 , while (b) is the expert model 𝜃 𝑛𝑒 trained
on the current domain using a duplicated copy of the backbone 𝜃𝑐𝑜𝑝𝑦 ; (c) denotes the baseline
loss term; (d) is the CDAG module, which leverages the expert model from (b) to guide backbone
adaptation to the current domain; (e) is the ODAC module, which preserves affinity consistency
among previous feature representations by freezing the old model 𝜃𝑜𝑒 ; and (f) is the OIVC module,
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 103. Publication date: March 2026.

103:6

J. Tan et al.

Fig. 2. The overall pipeline of our proposed AHN during training on the tth domain: (a) shows the main model
training process; (c) presents the baseline loss; (d) illustrates the CDAG Module, which uses the frozen expert
model from (b) to help the main model adapt to the current domain; (e) is the ODAC Module, leveraging
the frozen old model to constrain changes in affinities among old representations; (f) is the OIVC Module,
mitigating disruptions in old sample intra-class variance.

designed to mitigate intra-class variance drift in stored samples. The overall Unsupervised LReID
loss is defined as:
L𝑜𝑣𝑒𝑟𝑎𝑙𝑙 = L𝑏𝑎𝑠𝑒 + LOIVC + 𝜆ODAC LODAC + 𝜆CDAG LCDAG .
4.2

(2)

Base Model

In the Unsupervised LReID method proposed in this article, the lack of ground truth annotations
renders cross-entropy loss based on identity labels unsuitable for this task. Therefore, this article
utilizes a cluster prototype contrastive loss, which does not requiren identity labels,
 for model
o

𝑡
𝑡
𝑡 , 𝑦𝑡
training. When training in domain 𝑡, this domain is denoted as 𝐷𝑡 = 𝑥 1, 𝑦1 , . . . , 𝑥 𝑁
𝑁𝐷𝑖 ,
𝐷𝑖
where 𝑁𝐷𝑖 represents the number of images within the domain, and 𝑦 denotes the cluster pseudolabel for image 𝑥. For the current domain image 𝑥𝑖𝑡 , the output representations of the online
encoder and momentum encoder are 𝑓 (𝑥𝑖𝑡 |𝜃 𝑡 ) and 𝑓 (𝑥𝑖𝑡 |𝜃𝑚 ), respectively. The average momentum
representation, calculated over the sample set from domain 𝐷𝑡 that shares the same pseudo-label
𝑦𝑎 , is defined as the prototype of cluster 𝑎:
1 Õ
𝑝𝑎𝑡 =
𝑓 (𝑥𝑖𝑡 |𝜃𝑚 ).
(3)
𝑁𝑎 𝑡
𝑥𝑖 ∈𝑦𝑎

Thus, if 𝑥𝑖𝑐 belongs to cluster 𝑎, the contrastive loss for the cluster prototype can be defined as:
 



exp 𝑓 (𝑥𝑖𝑡 |𝜃 𝑡 ) · 𝑝𝑎𝑡 /𝜏𝑝
,
L𝑐𝑙𝑢𝑠𝑡𝑒𝑟 = E − log Í |𝑃 𝑡 |
(4)



exp 𝑓 (𝑥𝑖𝑡 |𝜃 𝑡 ) · 𝑝 𝑗 /𝜏𝑝 
𝑗=1


𝑡
where |𝑃 | represents the total number of clusters present in the domain 𝐷𝑡 , and 𝜏𝑝 serves as the
temperature hyperparameter.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 103. Publication date: March 2026.

Unsupervised Lifelong Person Re-Identification via Affinity Harmonization

103:7

The objective of Lcluster is to reduce the distance between a sample and the prototype of its
corresponding class, while increasing the separation from other clusters. Additionally, to improve
the model’s ability to distinguish samples from different clusters, a normalization-guided triplet
loss [22] is employed for training:
!!
2
𝑓 (𝑥 𝑝𝑡 |𝜃 𝑡 )
𝑓 (𝑥𝑎𝑡 |𝜃 𝑡 )
𝑓 (𝑥𝑎𝑡 |𝜃 𝑡 )
𝑓 (𝑥𝑛𝑡 |𝜃 𝑡 ) 2
L𝑡𝑟𝑖 = log 1 + exp
−
−
−
, (5)
k 𝑓 (𝑥𝑎𝑡 |𝜃 𝑡 )k 2 k 𝑓 (𝑥 𝑝𝑡 |𝜃 𝑡 )k 2
k 𝑓 (𝑥𝑎𝑡 |𝜃 𝑡 )k 2 k 𝑓 (𝑥𝑛𝑡 |𝜃 𝑡 )k 2 2
2

where 𝑥 𝑝𝑡 represents the positive samples of 𝑥𝑖𝑡 , and 𝑥𝑛𝑡 represents the negative samples. Both
are drawn from the current domain 𝐷𝑡 . Given that images in person ReID datasets generally
originate from different cameras, improving the Unsupervised LReID model’s performance and
enhancing its discriminative ability requires minimizing intra-class variance across different camera
styles. Suppose the current domain contains 𝑁𝑡 cameras, and the set of cameras is denoted as
𝐶 = {𝑐 1, . . . , 𝑐 𝑁𝑡 }. The intra-camera prototype is defined as the average momentum representation
of all samples recorded by the same camera 𝑐𝑏 and assigned the same pseudo-label 𝑦𝑎 :
Õ
1
𝑡
𝑝𝑎𝑏
=
𝑓 (𝑥𝑖𝑡 |𝜃𝑚 ).
(6)
𝑁𝑎𝑏 𝑡
𝑡
𝑥𝑖 ∈𝑦𝑎 ∩𝑥𝑖 ∈𝑐𝑏

When the pseudo-label of image 𝑥𝑖𝑐 is 𝑦𝑎 and the image is captured by camera 𝑐𝑏 , the contrastive

loss associated with the camera prototype can be expressed as:




 1 Õ

exp 𝑓 (𝑥𝑖𝑡 |𝜃 𝑡 ) · 𝑝𝑎𝑡 𝑗 /𝜏𝑐



,
L𝑐𝑎𝑚 = E −
log Í
(7)
𝑁 neg +1
 𝑁𝐶

𝑡
𝑡
exp
𝑓
(𝑥
|𝜃
)
·
𝑝
/𝜏
𝑗∈C


𝑡
𝑐
𝑖
𝑘=1
𝑘


where 𝜏𝑐 denotes the camera contrastive temperature hyperparameter. To further enhance the
model’s ability to identify individuals, the most challenging 𝑁𝑛𝑒𝑔 negative camera prototypes in
the current domain are selected for training. The loss L𝑐𝑎𝑚 aims to increase the affinity between
the sample representation and the camera prototypes belonging to the same cluster, effectively
minimizing intra-cluster variance.
The loss function for the base model is composed of (4), (5), and (7), balanced by the hyperparameters 𝜆𝑐𝑎𝑚 and 𝜆𝑡𝑟𝑖 :
L𝑏𝑎𝑠𝑒 = L𝑐𝑙𝑢𝑠𝑡𝑒𝑟 + 𝜆𝑐𝑎𝑚 L𝑐𝑎𝑚 + 𝜆𝑡𝑟𝑖 L𝑡𝑟𝑖 .
4.3

(8)

Old Domain Image-to-Image Affinity Constraint (ODAC)

From a technical perspective, ReID is essentially a feature-based matching task, where the core
objective is to identify the same individual by comparing the affinity between image features. The
model aims to bring images of the same identity closer in the feature space, yielding higher affinity
scores, while pushing images of different identities further apart, resulting in lower affinity scores.
However, In Unsupervised LReID, training each domain sequentially with only the base model can
result in the affinity relationships among images in the new domain interfering with previously
learned knowledge, which may ultimately lead to catastrophic forgetting.
To address this issue and ensure that new knowledge is acquired without forgetting previously
learned information, this article proposes an image-to-image affinity constraint loss for the old
domain, building on Lbase . The affinity relationships in the current domain are updated by this loss
function in a manner that remains compatible, while its impact on the relationships among images
in the old domain is minimized. Since the frozen model from the previous domain is considered an
expert for that domain, it is represented as 𝜃𝑜𝑒 . The affinity relationships it computes serve as a
reference to regulate the updates of the current online encoder 𝜃 𝑡 .
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 103. Publication date: March 2026.

103:8

J. Tan et al.

𝑜 }, where 𝑁
Consider a mini-batch of old images {𝑥 1𝑜 , . . . , 𝑥 𝑁
𝑏𝑠 indicates the batch size. The
𝑏𝑠
affinity distribution between images can be derived by applying the softmax function to the cosine
similarity of each image pair in the mini-batch. Furthermore, the affinity relationships in the current
domain are not solely computed by the online encoder 𝜃 , but rather by both the online encoder 𝜃
and the momentum encoder 𝜃𝑚 :
D
E 
exp 𝑓 (𝑥𝑖𝑜 |𝜃 𝑡 ) · 𝑓 (𝑥 𝑜𝑗 |𝜃𝑚 ) /𝜏𝑠
𝑜

,
𝑃𝑖,𝑗
=Í
(9)
𝑁 bs
exp 𝑓 (𝑥𝑖𝑜 |𝜃 𝑡 ) · 𝑓 (𝑥𝑘𝑜 |𝜃𝑚 ) /𝜏𝑠
𝑘=1

where h·i denotes the standardized cosine similarity, and 𝜏𝑠 is a temperature coefficient used
for scaling the affinity scores. 𝜃 𝑡 represents the online encoder at domain 𝑡, while 𝜃𝑚 is the
𝑜 is computed based on the
corresponding momentum encoder. It is worth noting that the affinity 𝑃𝑖,𝑗
feature representations of both 𝜃 𝑡 and 𝜃𝑚 , denoted as 𝜃 and 𝜃𝑚 , respectively. This reflects the affinity
between the current online features and the accumulated momentum features. In comparison to
using only the affinity between online encoders, where both encoders are represented by 𝜃 , this
online-to-momentum affinity mechanism more effectively guides the online encoder 𝜃 to update in
the direction of the accumulated momentum encoder 𝜃𝑚 .
For this batch of samples, generated by applying data augmentation to the aforementioned
old domain samples, the method utilizes the frozen previous model 𝜃𝑜𝑒 to compute the affinity
distribution between images as a reference. This calculated distribution then serves as a constraint,
guiding the updates in the current model. Specifically, the affinity reference between two augmented
images from the old domain, 𝑥˜𝑖𝑜 and 𝑥˜ 𝑜𝑗 , is defined as follows:
D
E 
exp 𝑓 (𝑥˜𝑖𝑜 |𝜃𝑜𝑒 ) · 𝑓 (𝑥˜ 𝑜𝑗 |𝜃𝑜𝑒 ) /𝜏𝑠
𝑜

,
𝑄𝑖,𝑗
=Í
(10)
𝑁 bs
𝑜 |𝜃 ) · 𝑓 (𝑥˜ 𝑜 |𝜃 ) /𝜏
exp
𝑓
(
𝑥
˜
𝑜𝑒
𝑜𝑒
𝑠
𝑖
𝑘=1
𝑘
where 𝜃𝑜𝑒 is obtained by freezing the online encoder 𝜃 𝑡 −1 .
Thus, this article formulates the affinity constraint between images for the previous domain
using the Kullback-Leibler (KL) divergence between the two distributions mentioned above:
LODAC = D𝐾𝐿 (𝑃 𝑜 k 𝑄 𝑜 ).

(11)

By minimizing LODAC , the current model is encouraged to align its computed affinity distribution
𝑃 for the old domain samples with the reference affinity distribution 𝑄 obtained from the frozen
model.
4.4

Current Domain Image-to-Image Affinity Guidance (CDAG)

Lifelong learning methods focus on acquiring new knowledge effectively while minimizing interference with previously acquired information. A fundamental challenge in continual learning,
particularly relevant to Unsupervised LReID, is achieving a balance between stability and plasticity
[3]. In the previous module, the ODAC component effectively constrained affinity relationships
among images within the old domain. However, an additional mechanism is required to guide the
model in learning the affinities among images within the domain 𝐷𝑡 , thereby better harmonizing stability and plasticity. To tackle this, the article proposes an affinity-guided loss tailored for
the current domain, which steers the model toward capturing the unique affinity relationships
characteristic of the current domain.
Similar to the ODAC module, this approach requires the establishment of a model to serve as
an expert within the knowledge base of the new domain. To achieve this, the current model 𝜃 𝑡
is duplicated to create 𝜃𝑐𝑜𝑝𝑦 , which is subsequently trained on the current domain using only
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 103. Publication date: March 2026.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113509

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Unleashing the potential of traditional person re-ID methods to clothes
changed scenario via curriculum learning
Yuxuan Xiao
a

a , Shanshan Zhang

a,b,∗, Jian Yang

a,b

PCA Lab, School of Computer Science and Engineering, Nanjing University of Science and Technology, China

b Nanjing University, Nanjing, China

a r t i c l e

i n f o

a b s t r a c t

Keywords:
Curriculum learning
Person re-ID
Clothes changed
Data scheduling
Data weighting

Most existing clothes changed person re-ID (CC re-ID) methods require personalized modules to recognize the
same ID wearing diﬀerent clothes, largely diﬀering from re-ID methods designed under the traditional same
clothes scenario (SC re-ID) in terms of model architecture. This makes it hard to unify methods under diﬀerent
clothes conditions, and thus is not friendly to real-world applications. In this work, we aim to unleash the potential
of the existing SC re-ID methods to the CC scenario without modiﬁcations to the model architecture, but with
a properly designed curriculum for training. Our curriculum learning involves clothes-level data scheduling and
data weighting. Speciﬁcally, we simulate the gradual cognitive process of humans through data scheduling,
which proposes to start the training with one piece of clothes per ID and then increase the variation gradually by
allowing the well-recognized IDs to pick up the clothes that are more diﬃcult than the current, thus encouraging
the model to learn clothes invariant features. To mitigate the learning bias caused by the imbalanced number
of clothes samples, we perform clothes data weighting during training, which assigns higher weights to samples
with lower accuracy. Extensive experiments on PRCC, LTCC, VC-Clothes, LaST, and DeepChange validate the
eﬀectiveness of our method. With our curriculum learning, the SC re-ID method, CLIP-ReID, outperforms top CC
re-ID methods on most datasets. The code will be released at https://github.com/YuSuen/CL-CC-ReID.

1. Introduction

as CLIP-ReID [24], can achieve competitive performance under the CC
scenario, as shown in Fig. 1. This indicates that SC re-ID methods have
large potential to eﬀectively capture discriminative pedestrian features
with clothes changed. Nevertheless, we must acknowledge that there is
still a gap, when compared with the top CC re-ID methods with additional components handling changes clothes. For example, the SC re-ID
method CLIP-ReID [24] underperforms the top CC re-ID method SCNet [20] by ∼8 pp w.r.t. Rank-1 under CC. This is because the SC re-ID
method without a clothes processing component is struggling to cope
with clothes changed interference within the same ID. As shown in Fig. 2
(w/o data scheduling), the SC re-ID model needs to handle almost ﬁve
on average clothes within each ID simultaneously, which poses great
challenges for the re-ID model as the intra-ID variation is too high, particularly in the initial stage of model training.
Inspired by the gradual cognition process of humans, we propose to
start the training with a fairly low level of intra-ID variation (i.e. one
piece of clothes per ID), and then increase the variation gradually by allowing the well-recognized IDs to pick up more clothes. Following such
a curriculum learning scheme, the SC re-ID method is expected to better handle clothes changes within each ID in the end. Here come two

Person re-identiﬁcation (re-ID) [1–3] aims to retrieve the same person in non-overlapping camera views based on the person image features, which draws great attention due to its wide applications. Early
person re-ID methods assume that an individual’s clothes do not change
in the short term, so the clothes can serve as an important clue for eﬀective recognition [4–9]. We refer to these methods as same clothes person re-ID (SC re-ID). However, people in real long-term environments
do not always maintain their clothes consistent. Thus, numerous eﬀorts
have been made to tackle the challenges posed by clothes changed (CC
re-ID), such as introducing additional clothes-irrelevant cues [10–17]
or disentangling clothes information at the image [18–20] or feature
level [21–23]. The additional processing of clothes information naturally makes these methods diﬀerent from SC re-ID in design, thus weakening the versatility of the CC re-ID methods. This is not conducive to
the promotion of re-ID systems in real-world applications.
To investigate the applicability of SC re-ID methods under clothes
changed, we ﬁrst directly apply SC re-ID methods to the clothes changed
scenario and ﬁnd that current state-of-the-art SC re-ID methods, such

∗ Corresponding author.

E-mail addresses: xiaoyuxuan@njust.edu.cn (Y. Xiao), shanshan.zhang@njust.edu.cn (S. Zhang), csjyang@njust.edu.cn (J. Yang).
https://doi.org/10.1016/j.patcog.2026.113509
Received 14 August 2025; Received in revised form 16 February 2026; Accepted 15 March 2026
Available online 18 March 2026
0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 179 (2026) 113509

Y. Xiao et al.

In summary, our contributions are as follows:
•

For the ﬁrst time, we point out that SC re-ID methods are potential
to well cope with the challenges under the CC scenario.
• We design a novel clothes-level curriculum to increase the intra-ID
variation gradually by allowing the distribution of training clothes
data to be adjusted dynamically, which involves clothes-level data
scheduling and data weighting, thereby enhancing the model’s ability to distinguish samples from diﬀerent IDs under the CC scenario.
• We apply our curriculum learning to several state-of-the-art SC re-ID
methods and obtain consistently signiﬁcant improvements over the
baselines under CC. Thanks to our proposed curriculum learning,
these SC re-ID methods achieve comparable performance under CC
and even outperform the current top CC re-ID methods.

Fig. 1. On LTCC [14], some top SC re-ID methods achieve consistent improvements under CC by using our curriculum learning and compete with existing top
CC re-ID methods.

2. Related work
In this work, we make an attempt to unleash the potential of the existing the traditional same clothes re-ID methods to the clothes changed
scenario via curriculum learning. Therefore, we ﬁrst review related work
on person re-ID, including SC re-ID and CC re-ID; and then we discuss
recent work of person re-ID in curriculum learning.
2.1. Person re-identiﬁcation
Same Clothes Person Re-identiﬁcation. Early person reidentiﬁcation refers to same clothes person re-ID (SC re-ID), as it
assumes that the clothes of a pedestrian do not change in a short
period of time. Therefore, the color and texture information of
clothes can be used as favorable features for identity recognition
under this scenario [4,5,7–9]. Recently, a two-stage method named
CLIP-ReID integrates CLIP [26] and prompt learning [27,28] into
person re-identiﬁcation and achieves encouraging results. CLIP-ReID
ﬁrst learns an ambiguous text prompt for each ID and then aligns the
image with the corresponding text prompt to capture the discriminative
features of each ID. To improve eﬃciency, PCL [29] directly uses the
visual encoder in CLIP to build a prototype for each ID, replacing
the text prompt in CLIP-ReID, and updates it through EMA during
training; while TF [30] introduces the Sequence-Speciﬁc Prompt (SSP)
module to update the prototype online. Although these methods have
achieved state-of-the-art performance on SC re-ID, they do not consider
applicability under clothes changed scenario. In this work, we propose
to directly transfer these methods to the clothes changed scenario via
curriculum learning.
Clothes Changed Person Re-identiﬁcation. SC re-ID assumes
that the clothes of the pedestrian do not change, but this assumption does not always hold true for person re-identiﬁcation in longterm environments. Therefore, research on clothes changed person
re-ID (CC re-ID) has emerged. The CC re-ID focuses on obtaining
clothes-irrelevant features for retrieval. Early methods employ additional clothes-irrelevant cues such as radio [10], contour sketches [11],
silhouette [12], gait [13], skeletons [14], 3D shape [15–17], etc., to
assist the model in learning clothes-irrelevant discriminative features.
Since these cues require additional equipment to obtain, some methods utilize existing human parsing models such as [31] to locate the
clothes area of the person image to eliminate or replace the pixels in
the clothes area, thereby preventing the model from learning clothesrelevant features [18–20]. Other methods achieve clothes-irrelevant feature learning by decoupling clothes information from identity information in the feature space [21–23]. Due to the additional processing of
clothes, these methods are weakened in generality and are not able to
handle same clothes scenario. We propose to customize the curriculum
for SC re-ID methods according to the clothes to explore the potential
of SC re-ID methods under clothes changed scenario, thereby achieving
methodological generality between SC and CC re-ID.

Fig. 2. Average numbers of clothes included for training within each ID.

Fig. 3. The counts of samples under diﬀerent clothes are imbalanced.

questions regarding the curriculum design: (1) How to decide whether
each ID is well-recognized? (2) Which piece of clothes should be added
to the training pool? For the ﬁrst question, we employ the silhouette
coeﬃcient [25] to measure for each ID the level of discrimination from
other IDs; once the silhouette coeﬃcient for a certain ID exceeds a given
threshold, meaning the this ID is well recognized, we add a new piece of
clothes for it. For the second question, we always choose the most different piece from previous ones, encouraging the model to learn clothes
invariant features. In addition, we notice that there is a severe imbalance in the counts of samples across diﬀerent clothes (Fig. 3). To mitigate the negative impact of clothes bias, we introduce a clothes-level
data weighting strategy, i.e. we assign higher weights to clothes samples
with lower accuracy, allowing the model to balance the optimization of
diﬀerent clothes in a dynamic way over the whole training process.
2

Pattern Recognition 179 (2026) 113509

Y. Xiao et al.

Fig. 4. Overview of our pipeline. We apply our curriculum learning, including clothes-level data scheduling and data weighting, on the standard SC re-ID pipeline.
Here, 𝐼𝐷1 represents the ID label, 𝐶𝐿𝑇 1 represents the clothes label, 𝑄 represents the classiﬁcation accuracy, 𝑤 represents the sample weight, 𝑘 represents the
threshold, and 𝑆𝐼𝐷1 represents the current silhouette coeﬃcient of 𝐼𝐷1. First, we simulate the human cognitive process by scheduling clothes data. Our scheduling
strategy involves silhouette coeﬃcient evaluation and maximization selection strategy (See Section 3.2 for details). Second, we dynamically weight the clothes data
during training to alleviate learning bias caused by the imbalance in the number of clothes samples (See Section 3.3 for details). Best viewed in color.

2.2. Curriculum learning for person re-ID

First, we start with clothes-level data scheduling. We randomly select
a set of clothes samples from each ID to build an initial training set.
In the following epochs, we decide whether to schedule new clothes
data for an ID before each training epoch by evaluating the silhouette
coeﬃcient of the ID. If the silhouette coeﬃcient of the ID is greater than
the threshold, we schedule a new set of clothes under that ID to form
the current epoch of training data, along with the previously scheduled
data. Our data scheduling involves a maximization strategy, that is, we
select the sample from the unselected clothes that is furthest from the
previous ones. When all the samples under an ID have been added to the
training data, it means that the data scheduling for that ID is ﬁnished,
see more details in Section 3.2.
During training, we performed data weighting to mitigate the learning bias caused by the imbalance in the number of samples for diﬀerent
clothes. We ﬁrst measure the ID accuracy of each piece of clothes by
multiplying its ID-level and clothes-level ID accuracies. Then, we assign
higher weights to samples with lower accuracy, allowing the model to
focus more on diﬃcult samples in a dynamic way based on the current
training status, see more details in Section 3.3.
The object function of our pipeline is written as follows:

Curriculum learning is ﬁrst proposed by [32]. It trains models using courses or meaningful sequences to imitate the process by which
humans learn from easy to diﬃcult tasks. Early studies on curriculum
learning often measure sample diﬃculty based on prior heuristic rules.
Self-paced learning [33] propose measuring the diﬃculty of samples by
training loss. Therefore, the model can adjust the sample curriculum
according to its own training state. Curriculum learning has recently
been shown to be eﬀective in solving domain adaptation person reidentiﬁcation problems. For example, SPCL [34] proposes to use loose
and tight clustering methods before each epoch training to retain the
reliable clusters , thereby providing more accurate pseudo-labels for the
target domain; SpML [35] deﬁnes the easy-to-diﬃcult sequence through
Style-uncertainty Augmentation, then learns the domain-invariant gradient from the easy domain, and ﬁnally adjusts the gradient through the
diﬃcult domain; CaCL [36] arranges the target domain domain adaptation curriculum sequence by measuring the distance between the target domain and the source domain on the camera. Diﬀerent from these
methods, we focus on clothes curriculum learning for each ID under
clothes changed scenario and achieve it through data scheduling and
data weighting.

arg min𝔼(𝜃) =
𝜃

3. Methodology

𝑁
∑

𝑤𝑥 𝑖 𝐿𝑥 𝑖 ,

(1)

𝑖=1

where 𝑁 indicates the number of samples, 𝜃 indicates the model parameters, 𝐿𝑥𝑖 indicates the loss function of the SC re-ID method at sample
𝑥𝑖 , and 𝑤𝑥𝑖 indicates the weight of the sample 𝑥𝑖 . We summarize in
Algorithm 1 an overall training process of our pipeline.

In this section, we ﬁrst give an overview of our proposed curriculum learning, followed by detailed descriptions of its two key designs:
clothes-level data scheduling and data weighting.

3.2. Data scheduling
3.1. Overview
The SC re-ID methods do not include additional processing for
clothes changed within same ID. If data with changed clothes is fed
in randomly, multiple samples of diﬀerent clothes associated with the

As shown in Fig. 4, our curriculum learning includes clothes-level
data scheduling and data weighting.
3

Pattern Recognition 179 (2026) 113509

Y. Xiao et al.

fore each epoch training. The formula of silhouette coeﬃcient is written
as follows:
𝑏(𝑥𝑖 ) − 𝑎(𝑥𝑖 )
𝑆𝐶(𝑥𝑖 ) =
, −1 ≤ 𝑆𝐶(𝑥𝑖 ) ≤ 1,
(3)
max{𝑎(𝑥𝑖 ), 𝑏(𝑥𝑖 )}

Algorithm 1 Training.
Input: The training dataset  = {(𝑥𝑖 , 𝑦𝑖 , 𝑧𝑖 )𝑁
}, Model 𝜑(⋅) with param𝑖=1
eter 𝜃, Threshold 𝑘
Output: Model Parameter 𝜃
1: Initial weights 𝑤 = 1.0
2: Initial training data 𝑡𝑟𝑎𝑖𝑛 by Eq. (2)
3: 𝑢𝑛𝑠𝑒𝑙 =  − 𝑡𝑟𝑎𝑖𝑛
4: for 𝑒𝑝𝑜𝑐ℎ = 0 to epochs do
5: if 𝑒𝑝𝑜𝑐ℎ > 1 ∧ 𝑢𝑛𝑠𝑒𝑙 ≠ ∅ then
6: Compute 𝑆𝑦 by Eq. (3) and Eq. (4)
7: if 𝑆𝑦 > 𝑘 then
8: 𝑛𝑒𝑤 = 𝑆𝑐ℎ𝑒𝑑𝑢𝑙𝑒(𝑡𝑟𝑎𝑖𝑛 , 𝑢𝑛𝑠𝑒𝑙 )
9: Update 𝑡𝑟𝑎𝑖𝑛 : 𝑡𝑟𝑎𝑖𝑛 ← 𝑡𝑟𝑎𝑖𝑛 ∪ 𝑛𝑒𝑤
10: Update 𝑢𝑛𝑠𝑒𝑙 : 𝑢𝑛𝑠𝑒𝑙 ← 𝑢𝑛𝑠𝑒𝑙 − 𝑛𝑒𝑤
11: end if
12: end if
13: Optimization 𝜃 with 𝑡𝑟𝑎𝑖𝑛 by Eq. (1)
14: update 𝑤 by Eq. (6), Eq. (7) and Eq. (8)
15: end for
16: return Model Parameter 𝜃

where 𝑆𝐶(𝑥𝑖 ) is the silhouette coeﬃcient for a given sample point 𝑥𝑖 ;
𝑎(𝑥𝑖 ) represents the average distance between 𝑥𝑖 and all other samples
within the same cluster (measuring how closely 𝑥𝑖 ﬁts within its cluster);
𝑏(𝑥𝑖 ) is the average distance from 𝑥𝑖 to points in the nearest neighboring
cluster (measuring how far 𝑥𝑖 is from other clusters). When 𝑏(𝑥𝑖 ) > 𝑎(𝑥𝑖 ),
it indicates that the sample point is closer to its own cluster, resulting
in a positive silhouette coeﬃcient. We calculate the mean 𝑆𝑦 of the silhouette coeﬃcients for all samples within identity 𝑦. The formula of 𝑆𝑦
is written as follows:
𝑁

𝑆𝑦 =

𝑦
1 ∑
𝑆𝐶(𝑥𝑖 ),
𝑁𝑦 𝑖=1

where 𝑁𝑦 represents the number of all samples labeled 𝑦 in the 𝑡𝑟𝑎𝑖𝑛 . 𝑆𝑦
evaluates the overall silhouette coeﬃcient of ID 𝑦, that is, the intra-class
compactness and inter-class separability of ID 𝑦. Fig. 5 illustrates the
diﬀerence in silhouette coeﬃcients for IDs with varying inter-class and
intra-class distances. The larger the silhouette coeﬃcient 𝑆𝑦 , the better
the model ﬁts the sample with ID 𝑦 in current training data, otherwise
it is not good.
Training pool updating. We control data scheduling by a threshold 𝑘. Speciﬁcally, if 𝑆𝑦 exceeds the threshold 𝑘, the current model is
deemed capable of recognizing ID 𝑦 suﬃciently to allow the addition of
new clothes curriculum data 𝑛𝑒𝑤 to it.
𝑛𝑒𝑤 = 𝑆𝑐ℎ𝑒𝑑𝑢𝑙𝑒(𝑡𝑟𝑎𝑖𝑛 , 𝑢𝑛𝑠𝑒𝑙
𝑦,𝑧 ),

(5)

where 𝑆𝑐ℎ𝑒𝑑𝑢𝑙𝑒(𝑡𝑟𝑎𝑖𝑛 , 𝑢𝑛𝑠𝑒𝑙
𝑦,𝑧 ) indicates the scheduling function, which
schedules new clothes data based on the current training data 𝑡𝑟𝑎𝑖𝑛 and
the unselected data 𝑢𝑛𝑠𝑒𝑙
𝑦,𝑧 . Our scheduling function involves maximization strategy. It entails ﬁrst calculating the Euclidean distance between
the feature center of the ID in the current 𝑡𝑟𝑎𝑖𝑛 and the feature center of
the clothes sample from the 𝑢𝑛𝑠𝑒𝑙
𝑦,𝑧 associated with the same ID. Then, all
samples under the clothes sample center with the furthest distance are
selected as 𝑛𝑒𝑤 . Fig. 4 illustrates the data scheduling process following
the maximization strategy.
After scheduled new clothes data for IDs that meet the threshold, the
training data for current epoch updated by 𝑡𝑟𝑎𝑖𝑛 ← 𝑡𝑟𝑎𝑖𝑛 ∪ 𝑛𝑒𝑤 , and
𝑢𝑛𝑠𝑒𝑙
𝑢𝑛𝑠𝑒𝑙
the unselected data updated by 𝑢𝑛𝑠𝑒𝑙
𝑦,𝑧 ← 𝑦,𝑧 − 𝑛𝑒𝑤 . If 𝑦,𝑧 = ∅, the
whole data scheduling ends, which means all training data are added to
the training.

Fig. 5. The diﬀerence in silhouette coeﬃcients for IDs with varying inter-class
and intra-class distances. Diﬀerent colors represent diﬀerent IDs, and the dashed
lines are classiﬁcation boundaries. The larger the silhouette coeﬃcient of the ID,
the better its intra-class compactness and inter-class separation, such as 𝑆𝐴 > 𝑆𝐵 .
Best viewed in color.

same ID may be input simultaneously, which raises a challenge to the
optimization during the initial training of model as the intra-ID variation is too high. Inspired by the gradual cognition process of humans,
we believe that the SC re-ID model’s learning of the clothes changed ID
also needs to follow the gradual principles, i.e. start the training with
a fairly low level of intra-ID variation. To this end, we propose to design a proper curriculum schedule for the SC re-ID method under clothes
changed.
Initialization. Formally, given a clothes changed dataset  =
{(𝑥𝑖 , 𝑦𝑖 , 𝑧𝑖 )𝑁
}, 𝑥𝑖 is a pedestrian sample, 𝑦𝑖 is an identity label, and 𝑧𝑖
𝑖=1
is a clothes label. The set of all samples under ID label 𝑦 and clothes
label 𝑧 is expressed as {𝑦,𝑧 , ∀𝑧 ∈ 𝑦 }, where 𝑦 represents the set of
all clothes 𝑧 associated with identity 𝑦. Then, we randomly select a
clothes set under identity 𝑦 to form the initial curriculum training data
𝑡𝑟𝑎𝑖𝑛 = 𝑖𝑛𝑖𝑡 , which provides a simple initial cognition of the training
ID for the model:
𝑖𝑛𝑖𝑡 = 𝑅𝑎𝑛𝑑𝑜𝑚𝑆𝑒𝑙𝑒𝑐𝑡({𝑦,𝑧 |𝑧 ∈ 𝑦 }),

(4)

3.3. Data weighting
As previously shown in Fig. 3, there are imbalance in the number
of clothes samples, which easily cause the SC re-ID model to focus on
the clothes samples with a large number, thus leading to biased feature
learning. To this end, we introduce a dynamically weighting for each
clothes sample during training. We ﬁrst estimate the diﬃculty of each
sample based on its classiﬁcation accuracy, as the fewer samples available for training, the harder it is for the model to classify the sample
accurately. Considering the diﬀerent diﬃculty of the model’s recognition between the whole ID and the ID with speciﬁc clothes, we calculate
the ID level and clothes level classiﬁcation accuracy of the ID in current
training data based on the ID and clothes respectively. Let 𝑦̂ = 𝜑(𝑥, 𝜃) be
the identity label predicted of 𝑥 by the model 𝜑(⋅). The accuracy calculation at the ID-level can be written as:

(2)

𝑁

𝑄𝑦 =

where 𝑅𝑎𝑛𝑑𝑜𝑚𝑆𝑒𝑙𝑒𝑐𝑡 indicates random selection. The unselected data
can be expressed as 𝑢𝑛𝑠𝑒𝑙
𝑦,𝑧 = 𝑦,𝑧 − 𝑡𝑟𝑎𝑖𝑛 .
Status assessment. In order to evaluate the recognition of the
model to the ID, we employ the silhouette coeﬃcient [25] as indictor to
decide whether to add new clothes curriculum data to the each ID be-

𝑦
1 ∑
1(𝑦̂𝑖 = 𝑦𝑖 ),
𝑁𝑦 𝑖=1

(6)

where 𝑁𝑦 indicates the number of the sample with identity label 𝑦, and
1(𝑦̂𝑖 = 𝑦𝑖 ) indicates indicator function. The value is 1 when 𝑦̂𝑖 = 𝑦𝑖 , otherwise it is 0. The accuracy calculation of the ID with the speciﬁc clothes
4

Pattern Recognition 179 (2026) 113509

Y. Xiao et al.

Table 1
Comparison with state-of-the-art methods on PRCC. Bold indicates the best results.

𝑧 can be written as:
𝑁𝑦,𝑧

𝑄𝑦,𝑧 =

1 ∑
1(𝑦̂𝑖 = 𝑦𝑖 ),
𝑁𝑦,𝑧 𝑖=1

(7)

Methods

where 𝑁𝑦,𝑧 indicates the number of the sample with identity label 𝑦 and
clothes label 𝑧. Then, we take the product of the ID-level ID accuracy, 𝑄𝑦 ,
and the clothes-level ID accuracy, 𝑄𝑦,𝑧 , as the accuracy of each clothes
under the ID, and give higher weights to the clothes with low accuracy.
The weights formula of sample 𝑥 with identity label 𝑦 and clothes label
𝑧 can be written as follows:
𝑤𝑥 = 1 − 𝑄𝑦 × 𝑄𝑦,𝑧 .

Extra info

PRCC
CC

(8)

Our weights are dynamically adjusted based on the model’s classiﬁcation accuracy of samples, which allows the model to dynamically balance the learning bias caused by the number of clothes samples.
4. Experiments
We combine our curriculum learning with the existing SC re-ID
method, CLIP-ReID, to compete with state-of-the-art CC re-ID methods.
Finally, we show a series of ablation studies. Let’s start from experimental settings.

SC

Rank-1

mAP

Rank-1

mAP

3DSL [15]
FSAM [12]
RCSANet [40]
GI-ReID [13]
CAL [21]
AIM [22]
3DInvarReID [16]
SCNet [20]
DCR-ReID [18]
IRM(STL) [42]
CCFA [23]
CLIP3DReID [43]
MCSC [41]

3D shape
Silhouette
–
Gait
–
–
3D shape
Parsing
Parsing
–
–
3D shape
–

51.3
54.5
50.2
37.6
55.2
57.9
57.2
61.3
57.2
48.1
61.2
60.6
57.8

–
–
48.6
–
55.8
58.3
56.5
59.9
57.4
46.0
58.4
59.3
57.3

–
98.8
100
–
100
100
–
100
100
–
99.6
–
99.8

–
–
97.2
–
99.8
99.9
–
97.8
99.7
–
98.7
–
99.8

Ours

–

63.8

60.0

100

99.5

Table 2
Comparison with state-of-the-art methods on LTCC. Bold indicates the
best results. DIFFER† reports the performance of the same architecture
as ours.

4.1. Experimental settings
Datasets. We conduct our experiments on PRCC [11], LTCC [14],
VC-Clothes [37], LaST [38], and DeepChange [39]. PRCC consists of
33,698 images captured by 3 cameras, including 221 identities, each
with two diﬀerent clothes. The same person wears the same clothes under cameras A and B, but diﬀerent clothes under camera C. The dataset
is divided into a training set consisting of 150 identities and a testing set
consisting of 71 identities. LTCC includes a total of 17,119 images from
152 identities. Each identity appeared in front of at least two cameras,
and the number of costumes ranged from 2 to 14. A total of 77 identities are selected for training, consisting of 46 identities with clothes
variations and 31 identities with consistent clothes. Another 75 identities are used for testing, including 45 clothes-changed identities and
30 clothes-consistent identities. VC-Clothes is a virtual dataset synthesized by GTA5. It contains 512 identities of 19,060 images in 4 diﬀerent
scenes. Each identity has 1 to 3 suits of clothes. LaST contains 10,862
identities and 228,156 images. It consists of a training set with 70,923
images for 5000 identities, an evaluation set with 20,584 images for
56 identities, and a test set with 133,214 images for 5803 identities.
DeepChange contains 178,407 images of 1121 identities, recorded by
a real-world surveillance system using 17 outdoor cameras of varying
resolutions. All identities are captured by at least two cameras, with the
majority captured by 2 to 6 cameras.
Training details. We reproduce the top SC re-ID methods CLIPReID (ViT-B/16) [24] on clothes changed datasets as baseline following
the default experimental settings, and apply our curriculum learning on
it. Note that apart from the data adjustments involved in our curriculum
learning, we do not make any changes to the source model and hyperparameters.
Evaluation protocols. Following the established evaluation protocol from previous works [18,21,22,40]: for PRCC, we conduct the
comparison under Same-Clothes (SC) and Cloth-Changed (CC) settings,
where SC means that the same identity always wears the same clothes
in the query and gallery images, and CC means that the same identity
wears diﬀerent clothes in the query and gallery images; for LTCC and
VC-Clothes, we compare our method with the state-of-the-art methods
under CC and General (GEN) settings, where the GEN means that for
each identity, some instances in the gallery wear the same clothes with
the query, while others may not; for LaST and DeepChange, as with
most methods, the comparison is performed under GEN. We employ Cumulative Matching Characteristic Rank-1 accuracy and Mean Average

Methods

Extra info

LTCC
CC

3DSL [15]
FSAM [12]
GI-ReID [13]
CAL [21]
AIM [22]
3DInvarReID [16]
SCNet [20]
DCR-ReID [18]
IRM(STL) [42]
CCFA [23]
CLIP3DReID [43]
MCSC [41]
DIFFER† [44]
Ours

3D shape
Silhouette
Gait
–
–
3D shape
Parsing
Parsing
–
–
3D shape
–
Text
–

GEN

Rank-1

mAP

Rank-1

mAP

31.2
38.5
28.9
40.1
40.6
40.9
47.5
41.1
–
45.3
42.1
42.2
36.5
46.2

14.8
16.2
14.2
18.0
19.1
18.9
25.5
20.4
–
22.1
21.7
19.4
15.6
25.3

–
73.2
73.6
74.2
76.3
73.2
76.3
76.1
66.7
75.8
–
73.9
72.6
77.5

–
35.4
36.1
40.8
41.1
35.4
43.6
42.3
46.7
42.5
–
40.2
35.2
47.8

Precision (mAP) for evaluation. Note that consistent with most methods [12,13,15,21,41], we report the results from a subset of cameras 3
and 4 on VC-Clothes as the performance for the CC setting.
4.2. Comparison with state-of-the-art methods
We compare our method with state-of-the-art CC re-ID methods on
PRCC, LTCC, VC-Clothes, LaST and DeepChange. As can be seen in Table 1, 2, 3, 4, our method outperforms the existing top CC re-ID methods on most metrics across the ﬁve datasets, achieving state-of-the-art
performance on PRCC, VC-Clothes, LaST and DeepChange. It is worth
mentioning that we do not employ additional clothes processing, such
as disentangling clothes information or introducing additional clothesirrelevant information. In detail, on PRCC, our method outperforms the
existing top CC re-ID method, SC-Net [20], by 2.5 pp w.r.t. Rank-1 under CC. Due to the simplicity of the SC scenario on PRCC, the performance reached saturation.Our method achieves 100% Rank-1 accuracy,
which is on par with the performance of previous methods. On challenge
dataset, LTCC, our method achieves competitive performance with the
top CC re-ID method, SC-Net, under CC; however, our method outperforms it by 4.2 pp w.r.t. mAP under GEN. On virtual dataset, VC-Clothes,
5

Pattern Recognition 179 (2026) 113509

Y. Xiao et al.

Table 3
Comparison with state-of-the-art methods on VCClothes. Bold indicates the best results.
Methods

Table 6
Comparison of ﬁxed and silhouette coeﬃcient
based adaptive scheduling.

VC-Clothes

epoch

CC

GEN

Rank-1

mAP

Rank-1

mAP

PCB [45]
3DSL [15]
FSAM [12]
GI-ReID [13]
CAL [21]
MCSC [41]

62.0
79.9
78.6
64.5
81.4
83.3

62.2
81.2
78.9
57.8
81.7
83.2

87.7
–
–
–
92.9
93.2

74.6
–
–
–
87.2
87.9

Ours

85.3

85.3

94.6

89.5

1
5
ours

LaST
GEN

DS

✓

mAP

Rank-1

mAP

BOT [4]
mAPLoss [38]
CAL [21]
MCSC [41]

68.3
69.9
73.7
75.4

25.3
27.6
28.2
31.0

47.5
–
54.0
56.9

13.0
–
19.0
21.5

Ours

85.6

52.9

66.3

26.7

VC-Clothes

LTCC

Rank-1

mAP

Rank-1

mAP

TF [30]
+ Random
+ Minimum
+ Maximum (ours)

82.7
84.1
83.9
84.9

83.1
84.1
84.8
85.4

39.5
43.6
44.4
44.9

18.3
24.3
24.1
24.3

PCL [29]
+ Random
+ Minimum
+ Maximum (ours)

84.1
85.1
84.7
86.5

84.8
85.3
86.0
86.2

37.8
42.6
42.9
43.1

18.8
24.6
25.1
25.1

CLIP-ReID [24]
+ Random
+ Minimum
+ Maximum (ours)

83.5
84.5
84.7
85.3

83.2
84.3
84.2
85.3

40.3
43.4
45.2
46.2

20.0
24.9
25.5
25.3

Rank-1

mAP

Rank-1

mAP

61.0
60.0
63.8

58.5
58.1
60.0

43.6
40.1
46.2

24.6
20.2
25.3

✓
✓

PRCC

LTCC

Rank-1

mAP

Rank-1

mAP

58.0
59.4
59.9
63.8

56.6
57.1
57.8
60.0

40.3
41.1
42.3
46.2

20.0
22.9
23.1
25.3

three diﬀerent SC re-ID methods on both datasets, even larger gains on
the more challenging dataset LTCC.
Comparison of diﬀerent scheduling strategies. (1) How to
schedule? To verify the superiority of our maximum scheduling strategy, we also report the results of using random or minimization strategies in Table 5. It can be found that our maximization strategy achieves
the largest improvement on the three SC re-ID methods consistently.
By selecting the clothes instance that is most dissimilar from those previously introduced, we deliberately increase intra-ID appearance variation at each stage of the curriculum. This progressively shifts the
model’s optimization focus from appearance-dominant cues to identityconsistent features, thereby reducing the tendency to overﬁt clothesrelated characteristics and encouraging more robust clothes-invariant
learning. These results demonstrate that establishing a rational curriculum for SC re-ID methods under CC is instrumental in unleashing its
potential. (2) When to schedule? We compare the results of scheduling
clothes data after every 1 or 5 epochs with our silhouette coeﬃcient
based approach and report the results in Table 6. It can be seen that
our scheduling based on the silhouette coeﬃcient performs better than
ﬁxed-period scheduling because the silhouette coeﬃcient can well evaluate the current model’s recognition ability of IDs, thereby enabling
the appropriate and adaptive introduction of new clothes data into the
training process.
Eﬀects of both components. We conduct ablation experiments by
adding components one by one to analyze the contributions of clotheslevel data scheduling (DS) and data weighting (DW). We show quantitative results in Table 7 and qualitative results in Fig. 6. When both DS
and DW are removed, our method degenerates to the baseline SC re-ID
method CLIP-ReID. As reported in Table 7, it can be seen that both DS
and DW bring consistent improvements w.r.t. all metrics. In detail, by
using DS, we obtain an improvement of 1.4 pp and 0.8 pp w.r.t. Rank-1
under CC on PRCC and LTCC, respectively; by using DW, we obtain an
improvement of 1.9 pp and 2.0 pp w.r.t. Rank-1 under CC on PRCC and
LTCC, respectively; when adding both DW and DS, we obtain a signiﬁcant improvement of 5.8 pp and 5.9 pp w.r.t. Rank-1 under CC on PRCC
and LTCC, which beneﬁts from the complementary nature of DW and
DS in the clothes curriculum setting: DW schedules the overall clothes
data from the label level, and DS carefully considers the impact of the
diﬀerence in the number of samples of diﬀerent clothes. These results
well demonstrate the eﬀectiveness of both two components. Additionally, the increased training time due to adding components is tolerable.
Take the large scale dataset DeepChange experiment conducted on a
single 24G 3090 as an example. based model CLIP-ReID takes 12h 41m
to train, while our method takes 13 h 33 m, with a slight increase of
6.8%.

Table 5
Consistent improvements on VC-Clothes and LTCC.
“+Random” means that our curriculum learning adopts
random strategy for data scheduling after the initial stage.
Methods

DW

✓

DeepChange
GEN

Rank-1

LTCC

Table 7
Eﬀects of both components.

Table 4
Comparison with state-of-the-art methods on LaST
and DeepChange. Bold indicates the best results.
Methods

PRCC

our method outperforms the existing top CC re-ID method, MCSC [41],
by 2.0 pp and 2.1 pp w.r.t. Rank-1 and mAP under CC. On two largescale datasets, LaST and DeepChange, our method outperforms the existing top CC re-ID method, MCSC, by 10.2 pp and 9.4 pp w.r.t. Rank-1
under GEN, and by 21.9 pp and 5.2 pp w.r.t. mAP, respectively.
4.3. Ablation study
To understand the generalization and individual contributions of our
curriculum learning, we perform a series of ablation studies. Please note
that unless explicitly mentioned, these experiments are evaluated under
CC.
Generalization to diﬀerent baselines. Our method is modelagnostic. It can be easily applied to any SC re-ID method. To demonstrate
the generalization ability of our method, we apply our curriculum training method to three state-of-the-art SC re-ID methods, including CLIPReID [24], TF [30] and PCL [29], and report the results on VC-Clothes
and LTCC. As shown in Table 5. It can be found that our curriculum
learning achieves consistent improvement under the CC scenario across
6

Pattern Recognition 179 (2026) 113509

Y. Xiao et al.

Fig. 6. Qualitative results of ablation experiments on PRCC under CC. Red boxes indicate mis-identiﬁcations. As depicted in the ﬁgure, both our two components
(DW and DS) are helpful to achieve more accurate retrieval results. Best viewed in color.

Table 8
Results of our method without manual clothes label.
Clothes Label

Manual (Ours)
Clustering

PRCC

Table 9
Analysis of threshold 𝑘.
𝑘

LTCC

Rank-1

mAP

Rank-1

mAP

63.8
62.2

60.0
58.8

46.2
45.2

25.3
25.2

0
0.1
0.2
0.3

Getting rid of manual clothes labels. In practice, manual clothes
labels are sometimes not available. Thus, we develop an alternative version of our method, without relying on manual clothes labels. In fact,
we can obtain pseudo clothes labels via clustering. Here in our experiments, we ﬁrst extract features for each patch using the CLIP-ReID model
trained on Market1501 [46], and them employ the classic clustering algorithm of DBSCAN [47] to achieve pseudo clothes labels. As shown in
Table 8, our method still achieves reasonable performance using noisy
pseudo clothes labels, only dropping by around 1 pp compared to using the accurate manual clothes labels. These results indicate that our
method can be also be applied on those data without manual clothes
labels.
Visualization of feature learning. We use t-SNE and GradCAM1
to visualize the feature learning of the model at training epochs 0, 10,
20, and 30. For t-SNE, we randomly select 100 samples from LTCC and
visualize the last output features of image encoder. We use the same
color to represent the same ID and diﬀerent numbers to represent diﬀerent clothes. As shown in Fig. 7, our method gradually clusters the same
ID samples under diﬀerent clothes together and distinguishes them from
other IDs. For GradCAM, we visualize the output of the last transformer
layer of the image encoder. As shown in Fig. 8, as the training epochs
increase, the model gradually shifts its focus from clothes areas (i.e.,
the high-response parts of the top) to clothes-irrelevant areas (i.e., the
head and body contours). These results well indicate that our curriculum learning enhances the SC method’s ability to extract discriminative
ID features under CC.
Analysis of threshold 𝑘. The threshold 𝑘 plays an important role
in determining when to schedule new clothes data, we conduct an

PRCC

LTCC

Rank-1

mAP

Rank-1

mAP

61.2
63.8
61.9
60.3

58.9
60.0
58.5
57.1

44.1
46.2
42.3
42.9

23.3
25.3
23.4
23.6

Table 10
Diﬀerent weighting strategies.
𝛼, 𝛽

PRCC

LTCC

Rank-1

mAP

Rank-1

mAP

0.3, 0.7
0.5, 0.5
0.7, 0.3

60.2
59.8
59.7

56.9
57.0
56.9

43.9
43.9
43.6

24.0
23.8
23.5

Ours

63.8

60.0

46.2

25.3

analysis of 𝑘. When 𝑆𝑦 > 0, it indicates that the model is capable of
correctly distinguishing the majority of samples under the ID [25].
As shown in Table 9, we ﬁnd that setting 𝑘 to 0 is not the best
choice for scheduling new clothes data for the ID. However, when
the value increases to 0.2 and 0.3, the performance of the model becomes unimpressive. The best results are achieved on both PRCC and
LTCC when we set 𝑘 to 0.1. We believe this is due to the fact that
a high threshold can cause the model to over-ﬁt to the current curriculum data, while a low threshold can hinder the model from stably adapting to CC scenario. Therefore, we ﬁnally set 𝑘 to 0.1 in all
experiments.
Diﬀerent weighting strategies. We modify Eq. (8) of our submission into weighted average form, 𝑤𝑥 = 1 − (𝛼𝑄𝑦 + 𝛽𝑄𝑦,𝑧 ), and report its
results and ours in Table 10. It can be found that our method is better
than averaging, as ours selects hard samples in a more strict way and
assigns them higher weights so that the model is better optimized during
training.

1 https://github.com/jacobgil/pytorch-grad-cam

7

Pattern Recognition 179 (2026) 113509

Y. Xiao et al.

Fig. 7. t-SNE visualization of the distribution of the features learned at each epoch. We randomly select 100 samples from LTCC and visualized. The same color
represents the same ID, and the number represents the clothes label. Our curriculum learning allows the SC method CLIP-ReID to gradually learn to extract ID features
with clothes robustness under CC. Please zoom in for better view.

Fig. 8. GradCAM visualization of the response of the features learned at each epoch. Thanks to our proper scheduling of training data, the features focus more on
clothes-irrelevant area (such as head, body shape etc.) as training progresses.

5. Conclusion

clothes, encouraging the model to learn clothes invariant features. Furthermore, to alleviate the learning bias caused by the imbalanced number of clothes samples, we perform data weighting for clothes samples
during training, which assigns higher weights to samples with lower
accuracy, allowing the model to balance the optimization of diﬀerent
clothes in a dynamic way over the whole training process. Extensive experiments on ﬁve CC re-ID benchmark datasets, including PRCC, LTCC,
VC-Clothes, LaST and DeepChange, verify the eﬀectiveness of our approach. By applying our curriculum learning, the performance of SC reID method, CLIP-ReID, under CC even outperforms the current top CC

In this paper, we point out that SC re-ID methods are potential to well
cope with the challenges under the CC scenario and make an attempt to
unleash their potential in this scenario. To this end, we design a novel
clothes-level curriculum, which involves clothes-level data scheduling
and data weighting. Speciﬁcally, we propose to start the training with
a fairly low level of intra-ID variation, and then increase the variation
gradually by allowing the well-recognized IDs to schedule new clothes
data, i.e. gradually adding the one that is furthest from the previous
8


codex
前两篇的方法部分已经读到核心：一篇把换衣问题重写成训练顺序问题，另一篇把终身无监督问题重写成新旧亲和关系的平衡问题。第二篇方法延伸到了第 8 页之后，我会补读后续方法页，保证机制链条不缺。
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
/bin/zsh -lc "pdftotext -f 9 -l 12 'Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Unsupervised Lifelong Person Re-Identification via Affinity Harmonization

103:9

the baseline method, without the application of any anti-forgetting mechanisms. This approach
allows 𝜃𝑐𝑜𝑝𝑦 to focus exclusively on learning affinity relationships among images within the current
domain:


𝜃𝑐𝑜𝑝𝑦 = 𝜃𝑐𝑜𝑝𝑦 − 𝜂∇𝜃𝑐𝑜𝑝𝑦 E𝑥∼𝐷 𝑐 L𝑏𝑎𝑠𝑒 (𝑥, 𝜃𝑐𝑜𝑝𝑦 ) .
(12)
It is noteworthy that, in contrast to the main model’s regular training, a different clustering scale is
utilized during the training of 𝜃𝑐𝑜𝑝𝑦 . This approach incorporates multiscale clustering information,
enhancing the model’s generalization capability for subsequent training phases. After training,
𝜃𝑐𝑜𝑝𝑦 is frozen and assigned as the expert model for the new domain knowledge base, denoted
as 𝜃 𝑛𝑒 .
The calculation of image-to-image affinity relationships in the current domain is similar to that
𝑡 } represent a batch of images taken from the domain
described in the ODAC module. Let {𝑥 1𝑡 , ..., 𝑥 𝑁
𝑏𝑠
𝑡 between these images are jointly
𝐷𝑡 , where 𝑁𝑏𝑠 denotes the batch size. The affinity relationships 𝑃𝑖,𝑗
computed by 𝜃 𝑡 and 𝜃𝑚 :
D
E 
exp 𝑓 (𝑥𝑖𝑡 |𝜃 𝑡 ) · 𝑓 (𝑥 𝑡𝑗 |𝜃𝑚 ) /𝜏𝑠
𝑡

.
𝑃𝑖,𝑗
=Í
(13)
𝑁 bs
𝑡 |𝜃 ) · 𝑓 (𝑥 𝑡 |𝜃 ) /𝜏
exp
𝑓
(𝑥
𝑠
𝑖 𝑡
𝑘=1
𝑘 𝑚
For the same batch of augmented samples, the frozen model 𝜃 𝑛𝑒 is utilized as the expert for new
knowledge to calculate the affinity relationships among images within the current domain 𝐷𝑡 ,
guiding the updates of the current model. The reference affinity between two augmented images 𝑥˜𝑖𝑐
and 𝑥˜ 𝑐𝑗 in the current domain is defined as:
D
E 
exp 𝑓 (𝑥˜𝑖𝑡 |𝜃 𝑛𝑒 ) · 𝑓 (𝑥˜ 𝑡𝑗 |𝜃 𝑛𝑒 ) /𝜏𝑠
𝑡

.
𝑄𝑖,𝑗
=Í
(14)
𝑁 bs
𝑡 |𝜃 ) · 𝑓 (𝑥˜ 𝑡 |𝜃 ) /𝜏
exp
𝑓
(
𝑥
˜
𝑛𝑒
𝑛𝑒
𝑠
𝑖
𝑘=1
𝑘
Finally, the affinity guidance loss for image-to-image relations in the current domain is formulated
based on the 𝐾𝐿 divergence between the two distributions described above:
LCDAG = D𝐾𝐿 (𝑃 𝑡 k 𝑄 𝑡 ).

(15)

By minimizing LCDAG , the model is encouraged to update the image-to-image affinity relationships
in the direction guided by the current domain knowledge expert, thereby facilitating more effective
learning of new knowledge.
4.5

OIVC

Within the Unsupervised LReID algorithm, training on the current domain involves the application
of stringent constraints on intra-class variance within clusters. By maximizing the similarity of the
same identity across different cameras, this approach effectively diminishes the intra-class variance
that typically arises from variations in camera styles, thus substantially enhancing the performance
of cross-camera person ReID. However, with domain shifts, the intra-class variance within the old
domain may increase due to the influence of newly introduced knowledge, potentially causing a
decline in the model’s ability to accurately recognize previously learned samples. To address this
issue, the AHN framework proposes the OIVC module, which mitigates fluctuations in intra-class
variance within the old domain. This approach effectively preserves the model’s ability to recognize
samples from previously learned domains.
At the end of training for each domain, all camera-aware prototypes are saved into the prototype
memory, and 𝐾𝑚𝑒𝑚 images from each cluster are stored in the image memory. To minimize the noise
of pseudo-labels during replay, the 𝐾𝑚𝑒𝑚 images with the highest cosine similarity to the cluster
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 103. Publication date: March 2026.

103:10

J. Tan et al.

Algorithm 1: AHN

prototype are selected from each cluster as the reliable images to be stored, thereby maintaining a
reasonable memory buffer size. At the beginning of adaptation to domain 𝐷𝑡 , the camera-aware
prototypes stored from previous domains, 𝑃𝑜 = {𝑃𝐷 1 , ..., 𝑃𝐷𝑡 −1 }, are merged with the prototypes
from the current domain, 𝑃𝑡 = {𝑃𝐷𝑖 }. This process updates the prototype memory to 𝑃 = 𝑃𝑜 ∪ 𝑃𝑡 .
When training in domain 𝑡, for a given old sample 𝑥𝑖𝑜 with identity 𝑦𝑎 , the model’s knowledge of
the old domain can be preserved by enhancing the similarity between the online representation
𝑓 (𝑥𝑖𝑜 |𝜃 𝑡 ), generated by the current domain online encoder 𝜃 𝑡 , and the camera-specific prototypes of
the same identity 𝑦𝑎 stored in memory. However, since the number of stored old domain samples is
significantly smaller than that of the current domain, directly using 𝑓 (𝑥𝑖𝑜 |𝜃 ) for computation may
lead to overfitting. To mitigate this issue, this method maintains a feature buffer that stores the
average features of various old domain samples during replay. For the old sample 𝑥𝑖𝑜 with identity
𝑦𝑎 , the feature memory contains the corresponding average feature 𝑓¯𝑎 . The weighted fused features
𝑓𝑤 , which can be utilized to compute the loss, are derived using the following formula:
𝑓𝑤 = 𝛾 𝑓 (𝑥𝑖𝑜 |𝜃 𝑡 ) + (1 − 𝛾) 𝑓¯𝑎 .
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 103. Publication date: March 2026.

(16)

Unsupervised Lifelong Person Re-Identification via Affinity Harmonization

103:11

Algorithm 2: CDAG

By combining the online feature representation of the current old domain sample 𝑓 (𝑥𝑖𝑜 |𝜃 𝑡 ) with its
corresponding average feature representation 𝑓¯𝑎 , the model is able to account for both individual
sample characteristics and the overall class feature distribution, thereby effectively mitigating the
risk of overfitting. Based on this, the OIVC Loss can be formulated as follows:




 1 Õ

exp 𝑓𝑤 · 𝑝𝑎𝑜 𝑗 /𝜏𝑐



 .
LOIVC = E −
log Í
(17)
𝑁 neg +1
 𝑁𝐶

𝑜
exp
𝑓
·
𝑝
/𝜏
𝑗
∈
C

𝑤
𝑘=1
𝑘 𝑐 

where 𝜏𝑜 represents the temperature hyperparameter used to control the intra-class variance
constraint. During training, the 𝑁𝑛𝑒𝑔 most challenging negative camera prototypes from the previous
domain are chosen. By minimizing LOIVC , the intra-class variance in the previous domain can be
effectively reduced, which improves the model’s proficiency in identifying data from prior datasets.
5

Experiments

5.1

Datasets

In this article, we conduct comprehensive experiments on the Unsupervised LReID benchmark,
utilizing a total of 12 ReID datasets. Among these, five datasets—Market-1501 [50], CUHK-SYSU
[43], DukeMTMC-reID [33], MSMT17 [41], and CUHK03 [19]—are used for training and assessing
forgetting mitigation in Unsupervised LReID. The remaining seven datasets—VIPeR [12], PRID
[15], GRID [23], iLIDS [51], CUHK01 [18], CUHK02 [17], and SenseReID [49]—are employed to
evaluate generalization capability on unseen data. Importantly, no ground-truth labels are used
during training in the Unsupervised LReID benchmark. We test two different training sequences:
Market-1501 → CUHK-SYSU → DukeMTMC-reID → MSMT17 → CUHK03, and DukeMTMC-reID
→ MSMT17 → Market-1501 → CUHK-SYSU → CUHK03. Detailed statistics for these datasets
are provided in Table 1. Following the evaluation setup described by Pu et al. [28], we compute
the Mean Average Precision (mAP) and Rank-1 Accuracy (R1) for each dataset. The overall
performance of our proposed approach is then assessed based on mAP and R1 scores across seen
and unseen data.
5.2

Implementation Details

The proposed method is implemented using the PyTorch framework [24], and all experiments
are performed on three Nvidia A4000 GPUs. We adopt the ResNet50 backbone [13], pre-trained
on ImageNet [34]. The input images are resized to 256 × 128, and data augmentation techniques
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 103. Publication date: March 2026.

103:12

J. Tan et al.
Table 1. Dataset Statistics of the Unsupervised LReID Benchmark [28]
Type

Dataset

Scale

# Train IDs

# Test IDs

Seen

Market-1501 [50]
CUHK-SYSU [43]
DukeMTMC-reID [33]
MSMT17_V2 [41]
CUHK03 [19]

Large
Mid
Large
Large
Mid

751
942
702
1,041
700

750
2,900
1,110
3,060
700

Unseen

VIPeR [12]
PRID [15]
GRID [23]
i-LIDS [51]
CUHK01 [18]
CUHK02 [17]
SenseReID [49]

Small
Small
Small
Small
Small
Mid
Mid

–
–
–
–
–
–
–

316
649
126
60
486
239
1,718

“–”denotes that the dataset is not used for training.

such as random horizontal flipping, cropping, Gaussian blurring, and erasing [55] are applied. For
each training phase (domain), we employ the Adam optimizer [16] with a weight decay of 0.0005,
training for 30 epochs with 400 iterations per epoch. The learning rate is initialized to 0.00035,
with a warm-up period spanning the first 10 epochs, and remains constant without any decay
throughout the training process.
At the beginning of each epoch, DBSCAN [8] is applied on the re-ranked Jaccard distance [54]
to update pseudo-labels for the new domain, using a minimum cluster size of 4 and a distance
threshold of 0.55. When training the new domain knowledge expert 𝜃𝑐 , the minimum cluster size
remains 4, but the distance threshold is adjusted to 0.60. The momentum hyperparameter is set
to 𝛼 = 0.999. Following Chen et al. [4], we set 𝜏𝑝 = 0.5, 𝜏𝑐 = 0.07, 𝑁𝑛𝑒𝑔 = 50, and 𝜆𝑐𝑎𝑚 = 0.5 in the
baseline, as shown in Equation (7).
Additionally, we employ a random identity sampler to generate minibatches, effectively addressing the imbalance among different identities. According to the clustering configuration of
the current domain, which requires each cluster to have at least 4 neighbors, the 32 images are
grouped into 8 identities, with each identity comprising 4 images. We perform a grid search to tune
𝜏𝑠 , 𝜆ODAC , and 𝜆CDAG (as described in Section 5.5). To ensure a fair comparison with other LReID
methods incorporating replay, we set 𝐾𝑚𝑒𝑚 = 2, retaining 2 images per cluster. Consequently, the
32 images from the previous domain represent 16 identities, with 2 images per identity. At the end
of training, only the momentum encoder is retained for inference.
5.3

Comparison with the SOTA

Comparison Methods. Due to the limited references available for unsupervised LReID, specifically
only UCR [5], the forthcoming experiments will compare the proposed AHN method not only
against UCR but also with several recent supervised lifelong learning methods, including AKA [28],
PatchKD [36], LSTKC [47], DKP [46], LSTKC++ [45], and DASK [44]. For completeness, we also
adapt two replay-based lifelong learning approaches, iCaRLL [32] and Co2LL [2], for the person
ReID task. To ensure fairness, we further re-train the SOTA supervised method DASK under the
unsupervised lifelong with replay setting for direct comparison with our approach.
Results on Seen Datasets. Tables 2 and 3 report the results under training orders 1 and 2. Compared
with the latest unsupervised baseline UCR, our approach achieves consistent improvements of
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 4, Article 103. Publication date: March 2026.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Unsupervised person re-identification via camera-aware multi-level label refinement.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 195 (2026) 108292

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

Unsupervised person re-identiﬁcation via camera-aware multi-level label
reﬁnement
Ning Tang

a , Zheyi Fan

b,∗, Yixuan Zhu

a , Tingting Zhang

a

a School of Integrated Circuits and Electronics, Beijing Institute of Technology, Beijing, 100081, China
b

School of Information and Electronics, Beijing Institute of Technology, Beijing, 100081, China

a r t i c l e

i n f o

Keywords:
Unsupervised person re-identiﬁcation
Label reﬁnement
Camera variation
Contrastive learning

a b s t r a c t
Unsupervised person re-identiﬁcation (re-ID) aims to match individuals across camera views without manual annotations, making it a challenging yet promising task. Although recent methods have made notable progress by
leveraging pseudo-labels, two key challenges remain insuﬃciently addressed: (1) the inherent noise in pseudolabels stemming from clustering, and (2) the limited discriminability of features resulting from camera variation.
To address these issues, we propose a camera-aware multi-level label reﬁnement (CMLR) framework, which
jointly reﬁnes labels at both cluster and instance levels to facilitate more eﬀective contrastive learning and enhance feature discrimination. At the cluster level, our dual-level intra-inter reﬁnement (DIIR) module exploits
intra- and inter-camera relationships to improve global and local pseudo-labels. At the instance level, the aﬃnityguided mutual reﬁnement (AGMR) module computes aﬃnity scores between samples based on selected informative nodes, adaptively pulling reliable positive pairs closer while pushing negative ones apart. By integrating
camera-aware cues into multi-level reﬁnement, CMLR enhances intra-class cohesion and inter-class separation,
enabling more robust feature learning. Extensive experiments on Market-1501 and MSMT17 demonstrate the
superiority of our CMLR compared to state-of-the-art unsupervised re-ID approaches.

1. Introduction
Person re-identiﬁcation (re-ID) aims to retrieve speciﬁc individuals
from videos or images captured by non-overlapping cameras. It plays
a crucial role in numerous real-world applications, such as intelligent
surveillance, public security, and urban management. In recent years,
person re-ID has attracted considerable research attention in the computer vision community due to its signiﬁcance and challenges. Traditionally, person re-ID relies on fully supervised training with extensive
cross-camera identity annotations, which are costly and impractical to
obtain in large-scale or real-time scenarios. To address this limitation,
unsupervised person re-ID has emerged as a promising alternative, offering better scalability and generalization capabilities without the need
for manual labeling.
Unsupervised person re-ID methods can be broadly categorized into
two groups: unsupervised domain adaptation (UDA) and fully unsupervised (FU) approaches. UDA assumes access to a labeled source domain
and an unlabeled target domain, with the goal of transferring the learned
knowledge from the source to the target domain. A variety of strategies have been proposed for UDA, including feature alignment, image
translation, to mitigate the domain gap between labeled and unlabeled

data Toldo et al. (2020), Zhong et al. (2018). While UDA methods have
demonstrated strong performance under certain conditions, they still
rely on the availability of a labeled dataset, and their generalization is
often limited when applied to unseen target domains with signiﬁcantly
diﬀerent distributions.
In contrast, fully unsupervised person re-ID assumes no access to
any labeled data, making it more challenging but also more applicable
to real-world deployments. This setting demands that models learn discriminative features solely from the unlabeled target domain. A dominant line of work adopts a clustering-based self-training paradigmHe
et al. (2024), Lin et al. (2019), Zeng et al. (2020), where the model
initially extracts features for all samples, computes Jaccard similarities,
and applies clustering algorithms (e.g., DBSCAN Ester et al. (1996), kmeans MacQueen (1967)) to group samples into clusters. These cluster
assignments serve as pseudo-labels, which are then used to ﬁne-tune the
feature extractor through supervised learning objectives.
Recent advances in unsupervised person re-ID have largely focused
on improving the clustering-based self-training paradigm. Among these
eﬀorts, cluster-level contrastive learning has emerged as a dominant
strategy due to its eﬀectiveness in capturing global structure and enhancing feature discrimination. As a benchmark, CC Dai et al. (2022)

∗ Corresponding author.

E-mail address: funye@bit.edu.cn (Z. Fan).
https://doi.org/10.1016/j.neunet.2025.108292
Received 25 April 2025; Received in revised form 18 September 2025; Accepted 2 November 2025
Available online 5 November 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 195 (2026) 108292

N. Tang et al.

Fig. 1. (a) In the clustering stage, the variation curve of the number of clusters generated by DBSCAN on two benchmark datasets during the training process. (b) tSNE visualization of feature embeddings on MSMT17 after clustering. Intra-camera samples tend to cluster closely even across diﬀerent identities, while inter-camera
samples of the same identity are scattered, revealing signiﬁcant camera-speciﬁc bias.

stores cluster-level representations to reduce memory overhead and stabilize training. DCMIP Zou et al. (2023) further introduces multiple cluster proxies to improve both intra-class compactness and inter-class separability. More recently, DHCL-HNM Zhao and Shu (2025) designs a
debiased hybrid contrastive learning strategy combined with hard negative mining to enhance sample discrimination.
In parallel, a number of approaches target the improvement of
pseudo-label reliability and robustness to noise. For example, DSCE
Yang et al. (2021) proposes a dynamic symmetric cross-entropy loss to
tolerate noisy pseudo-labels. LP Lan et al. (2023) builds feature and label
puriﬁcation modules by integrating local and global views under oﬄine
teacher supervision. DCCT Chen et al. (2023) adopts dual clustering coteaching with dynamic pseudo-label assignment and consistent sample
mining to suppress noise accumulation. To address camera domain shift,
CAP Wang et al. (2021) divides clusters into camera-aware subgroups,
enabling ﬁner-grained supervision. In a more recent study, CGMAL Ran
et al. (2025) integrates camera-aware semantic associations into contrastive learning and leverages graph convolution to propagate robust
domain-invariant features. Despite these progresses, two persistent challenges still remain: the accumulation of pseudo-label noise during iterative training and the large intra-class variation caused by camera
changes.
On one hand, while visually similar samples may naturally form
reasonable groupings, such groupings inevitably introduce incorrect
pseudo-labels due to imperfect feature representations and ambiguous clustering boundaries. In a fully unsupervised iterative training
paradigm, these potentially erroneous groupings are nonetheless treated
as ground-truth supervision, leading to the propagation and ampliﬁcation of errors through subsequent training rounds. This accumulation of label noise can misguide model optimization and degrade
ﬁnal performance, particularly during the early training stages when
feature embeddings remain immature and less discriminative. As shown
in Fig. 1(a), the number of clusters generated by DBSCAN on Market1501 Zheng et al. (2015) and MSMT17 Wei et al. (2018) deviates substantially from the ground-truth identity count and ﬂuctuates throughout training, especially in the early stages. These ﬂuctuations highlight
the instability of clustering results and further indicate the presence of
label noise, which can hinder robust representation learning.
On the other hand, images of the same individual captured by diﬀerent cameras often exhibit signiﬁcant variations in lighting, viewpoint,
pose, and background. Conversely, visually similar appearances may
arise from diﬀerent individuals captured by the same camera, due to

shared environmental factors. These phenomena compromise the reliability of feature-based similarity metrics and lead to cross-camera misgrouping during clustering. Fig. 1(b) shows the t-SNE Van der Maaten
and Hinton (2008) visualization of feature distributions after clustering.
Same-identity samples often scatter across clusters due to cross-camera
appearance shifts, while diﬀerent identities from the same camera may
cluster closely. This highlights that clustering based solely on global similarity, without camera context, can lead to inaccurate pseudo-labels and
hinder robust representation learning.
To address the limitations, we propose the camera-aware multi-level
label reﬁnement (CMLR) framework, which employs a hierarchical approach consisting of dual-level intra-inter reﬁnement (DIIR) at the cluster level and aﬃnity-guided mutual reﬁnement (AGMR) at the instance
level. Speciﬁcally, to reduce the over-reliance on global features, DIIR
processes samples separately using both global and local features, reﬁning distinct pseudo-labels for each feature branch. This strategy leverages intra-camera and inter-camera relationships to pull samples closer
to their correct clusters, thereby improving the reliability of pseudolabels. Meanwhile, AGMR reﬁnes instance-level labels by mining the
most relevant inter-camera informative nodes and computing aﬃnity
scores between samples, which further enhances the discriminative capability of the learned features. By integrating these two levels of reﬁnement, CMLR eﬀectively alleviates the challenges of label noise and
camera variation in unsupervised person re-ID, achieving state-of-theart performance on benchmark datasets. In summary, the contributions
of this work are structured as follows:

2

•

We introduce a dual-level intra-inter reﬁnement (DIIR) method for
cluster-level contrastive learning, which reﬁnes global and local
pseudo-labels by leveraging camera information. This reﬁnement
narrows the gap between hard positive samples and their reliable
cluster centroids.

•

We further propose an aﬃnity-guided mutual reﬁnement (AGMR)
met-hod, which mitigates the impact of camera variation by pulling
positive pairs across diﬀerent cameras closer. This is achieved by
reﬁning instance-level pseudo-labels based on aﬃnity scores with
informative nodes, allowing for a more ﬂexible similarity estimation
beyond rigid one-hot assignments.

•

Extensive experiments on benchmark datasets demonstrate that our
proposed method achieves state-of-the-art performance, outperforming existing unsupervised re-ID methods.

Neural Networks 195 (2026) 108292

N. Tang et al.

2. Related work

Lan et al. (2023) designs a feature puriﬁcation module that integrates
local view features within a cluster contrastive learning framework, effectively addressing biases related to global features.
Compared to these methods, our approach further integrates camera
information with both global and local features to uncover more comprehensive and robust identity representations.

2.1. Fully unsupervised approaches for person re-ID
Unsupervised person re-ID methods can be broadly categorized into
unsupervised domain adaptation (UDA) Tang et al. (2019) and fully unsupervised learning (FU) Yin et al. (2023), depending on whether external labeled data from the source domain is utilized. UDA methods Li
and Zhang (2020), Wang et al. (2018) aim to bridge the domain gap
between a labeled source domain and an unlabeled target domain by
aligning feature distributions Deng et al. (2021), Toldo et al. (2020) or
transferring image styles across domains Deng et al. (2018), Zhong et al.
(2018, 2019).
In contrast, FU methods directly train models on the unlabeled target domain. These approaches focus on discovering and leveraging the
intrinsic structure of the target data for representation learning, without any external supervision. A dominant strategy in fully unsupervised
re-ID is to generate pseudo-labels via clustering techniques Cheng et al.
(2022), Li et al. (2023), which are subsequently used as supervision signals to guide model training.
Building upon this clustering-based paradigm, recent methods in
fully unsupervised person re-ID have introduced memory-based contrastive learning frameworks to further enhance representation quality.
For instance, CC Dai et al. (2022) maintains a memory bank of cluster features and performs contrastive learning at the cluster level. CAP
Wang et al. (2021) introduces camera-aware proxies for each cluster,
promoting intra-class compactness and inter-class separation. Similarly,
ICE Chen et al. (2021) divides camera-speciﬁc proxies within clusters,
eﬀectively reducing intra-class variance. DCMIP Zou et al. (2023) enhances clustering performance by generating multi-attribute proxies for
each cluster and jointly optimizing sample-to-proxy and proxy-to-proxy
distances. IICS Xuan and Zhang (2021) performs camera-speciﬁc clustering and estimates the probability of cross-camera pairs belonging to
the same identity, enabling ﬁne-grained identity discrimination. MGCE
Sun et al. (2021) conducts multiple clustering iterations per epoch to reliably select samples and suppress noise. Motivated by these advances,
we focus on fully unsupervised person re-ID, and our work is built upon
the memory-based contrastive learning framework.

2.3. Noisy labels
In unsupervised person re-ID, noisy pseudo-labels are inevitable. In
the early training stages, limited feature extraction capability and ambiguous cluster boundaries result in substantial label noise. Moreover,
such noise can accumulate over iterations, hindering model optimization.
To address the challenge, various methods have been proposed for
label reﬁnement. Yu et al. Yu et al. (2019) reduce the inﬂuence of hard
samples by exploiting comparative consistency among reﬁned labels.
MMT Ge et al. (2020a) improves accuracy by reﬁning hard labels ofﬂine and soft labels online. MMCL Wang and Zhang (2020) reformulates unsupervised person re-ID as a multi-label classiﬁcation task, assigning a single-class label to each image and using a reliable sample
set to perform multi-label prediction for label reﬁnement. RLCC Zhang
et al. (2021b) propagates pseudo-labels and conﬁdence scores across
iterations to progressively reﬁne noisy labels. ISE Zhang et al. (2022)
enhances clustering quality by introducing a sample extension strategy that generates support samples near cluster boundaries. SECRET
He et al. (2022) improves label quality by ﬁltering out inconsistent clustering results based on the consistency between global and local features
under camera constraints. In this work, we mitigate label noise from
multiple perspectives by encouraging sample-to-cluster alignment with
reliable clusters and bringing positive sample pairs closer at the instance
level.
3. Methodology
We propose the camera-aware multi-level label reﬁnement (CMLR)
framework to mitigate label noise and the adverse impact of camera
variation on feature learning. The framework consists of two key modules: dual-level intra-inter reﬁnement (DIIR) and aﬃnity-guided mutual reﬁnement (AGMR), which respectively reﬁne cluster-level and
instance-level labels thro-ugh a camera-aware hierarchical optimization
strategy. The overall unsupervised person re-ID workﬂow is shown in
Fig. 2.

2.2. Part-based person re-ID
Traditional person re-ID methods predominantly rely on global features for identiﬁcation, often overlooking ﬁne-grained local details. This
limitation may result in error accumulation and suboptimal model performance. To address this issue, recent research has increasingly focused
on part-based discriminative feature learning to enhance re-ID accuracy.
Part-based person re-ID methods can generally be categorized into three
groups. The ﬁrst category leverages prior knowledge, such as pose estimation or body landmark detection Zhang et al. (2021a), to locate
speciﬁc body parts. However, the performance of these methods largely
depends on the accuracy and robustness of the auxiliary models. The second category adopts attention mechanisms to emphasize high-activation
regions in feature maps Fu et al. (2019). While this strategy is ﬂexible,
the resulting regions often lack semantic consistency, potentially leading
to incorrect part selection. The third category partitions features using
predeﬁned horizontal stripes Sun et al. (2018). Compared to the previous two, this approach is more scalable and robust, as it does not rely
on additional pretrained models.
In unsupervised person re-ID, horizontal partition methods have
proven eﬀective for learning discriminative representations. PAUL
Yang et al. (2019) proposes a patch-based method that extracts local
discriminative features from unlabeled patches instead of whole images.
PPLR Cho et al. (2022) introduces a cross-consistency score to assess the
complementarity between local and global features. SSL Lin et al. (2020)
divides images into patches and calculates the average distance between corresponding patches of sample pairs to adjust class ranking. LP

3.1. Preliminary
Given an unlabeled person re-identiﬁcation dataset 𝑋 = {𝑥𝑖 }𝑁
,
𝑖=1
where 𝑁 denotes the total number of images and 𝑥𝑖 represents the 𝑖-th
image. Let 𝐹 = {𝑓𝑖 }𝑁
∈ ℝ𝑁×𝐶×𝐻×𝑊 denote the corresponding feature
𝑖=1
maps extracted from 𝑋 using the model 𝑓𝜃 , where 𝐶 is the number of
channels, 𝐻 and 𝑊 are the height and width of the image, respectively.
𝑓𝑖 = 𝑓𝜃 (𝑥𝑖 ), where 𝜃 represents the learnable parameters of the model.
The feature maps in 𝐹 are partitioned into two components, denoted
𝐻

as 𝐹𝑢𝑝 = {𝑓𝑢𝑝,𝑖 }𝑁
and 𝐹𝑙𝑜𝑤 = {𝑓𝑙𝑜𝑤,𝑖 }𝑁
, with 𝐹𝑢𝑝 , 𝐹𝑙𝑜𝑤 ∈ ℝ𝑁×𝐶× 2 ×𝑊 .
𝑖=1
𝑖=1
A generalized mean pooling operation is independently applied to
these feature branches, yielding three sets of 𝑑-dimensional feature vectors: the global features 𝑈𝑔 = {𝑢𝑔,𝑖 }𝑁
, the upper local features 𝑈𝑢𝑝 =
𝑖=1
{𝑢𝑢𝑝,𝑖 }𝑁
, and the lower local features 𝑈𝑙𝑜𝑤 = {𝑢𝑙𝑜𝑤,𝑖 }𝑁
.
𝑖=1
𝑖=1
At the initial stage of each training epoch, we compute the Jaccard
distance matrix and employ the DBSCAN clustering algorithm to cluster
the global features, generating hard pseudo-labels 𝑌 = {𝑦𝑖 }𝑁
, where
𝑖=1
𝑌 ∈ {1, … , 𝑁 ′ } and 𝑁 ′ denotes the total number of clusters. The corre′
sponding one-hot label vector is denoted as 𝑌̂ = {𝑦̂𝑖 }𝑁
, where 𝑦̂𝑖 ∈ ℝ𝑁 .
𝑖=1
Following the methodology described in Dai et al. (2022), a cluster′
level memory bank 𝑀 ∈ ℝ𝑑×𝑁 is established and maintained, where 𝑀
stores the centroid feature vector of each cluster, and 𝑑 is the dimension
3

Neural Networks 195 (2026) 108292

N. Tang et al.

Fig. 2. The overall framework of the proposed method. In the clustering phase, DBSCAN is applied to group global features and assign initial hard pseudo-labels. In the
training phase, the DIIR module reﬁnes these labels for cluster-level contrastive learning by leveraging camera-aware neighbors: inter-camera neighbors for global
features and intra-camera neighbors for local features. Meanwhile, the AGMR module performs instance-level contrastive learning by constructing aﬃnity-based
labels through informative node selection, further mitigating camera-style discrepancies and enhancing the reliability of the clustering results.

of the feature vector. Speciﬁcally, the cluster centroids in the memory
bank are initialized using the corresponding average feature vectors and
pseudo-labels, as follows:
1 ∑
𝑀[𝑖] =
𝑢𝑗
(1)
|𝐶̃𝑖 | 𝑢 ∈𝐶̃
𝑗

This cluster-level loss eﬀectively reduces the distance between samples and the centroids of their respective clusters while pushing them
away from other clusters. Following the cluster-level contrastive learning paradigm outlined in Eq. (3), separate losses are computed for global
and local features, yielding three distinct cluster-level losses: 𝐿𝑔 , 𝐿𝑢𝑝 ,
and 𝐿𝑙𝑜𝑤 . These losses are then weighted to obtain the total cluster-level
loss 𝐿𝑐𝑙𝑢𝑠𝑡𝑒𝑟 , as shown in Eq. (4):

𝑖

where 𝐶̃𝑖 denotes the cluster with pseudo-label 𝑖 and | ⋅ | represents the
counting of elements. The cluster centroids stored in 𝑀 are updated via
a moving average strategy:
𝑀[𝑦𝑖 ] = 𝑚𝑀[𝑦𝑖 ] + (1 − 𝑚)𝑢𝑖

𝐿𝑐𝑙𝑢𝑠𝑡𝑒𝑟 = (1 − 𝜆1 )𝐿𝑔 + 𝜆1 (𝐿𝑢𝑝 + 𝐿𝑙𝑜𝑤 )

(4)

where 𝜆1 is the weight parameter that balances the global loss and the
local loss.

(2)

where 𝑚 denotes the momentum update factor.
To complement the limited representation of global features, memory banks for local features are also set to facilitate the discovery of more
detailed information during the learning process. The three branches independently compute cluster centroids based on their respective feature
vectors according to Eq. (1) while sharing the same set of pseudo-labels
𝑌 . These centroids are then updated following Eq. (2).
The cluster-level contrastive loss formulation is presented as follows:

3.2. Dual-level intra-inter reﬁnement

(3)

In each training epoch, hard pseudo-labels are generated from the
initial clustering, oﬀering only a coarse approximation of the feature
distribution. This often results in features being assigned to incorrect
clusters, with label noise compounding over time and hindering model
optimization. To address this, we propose the dual-level intra-inter reﬁnement (DIIR) method, as illustrated in Fig. 3. DIIR reﬁnes pseudolabels from both global and local perspectives, enhancing label quality
and improving clustering accuracy.

where |𝐵| represents the number of samples in each mini-batch, 𝑦̂𝑖 represents the one-hot encoded form of the pseudo-label for sample 𝑥𝑖 , ∗
denotes three branches: global (g), upper local (up), and lower local
(low) features, 𝑀∗ represents the cluster-level memory bank that stores
the centroids of features from three branches, and 𝜏 is the temperature
parameter.

3.2.1. Reﬁnement for labels of global features
Due to camera style variation, features captured by the same camera tend to cluster tightly, while those from diﬀerent cameras are more
dispersed. As a result, samples belonging to the same identity may be
assigned to diﬀerent clusters simply because they were captured by different cameras. To alleviate this issue, we aim to pull closer the centroids of inter-camera clusters that are likely to correspond to the same

𝐿∗ = −

|𝐵| 𝑁 ′
⎛ exp ((𝑀 [𝑙]𝑇 ⋅ 𝑢 )∕𝜏 ) ⎞
∗
∗,𝑖
1 ∑∑
⎟
𝑦̂𝑖 [𝑙] ⋅ log ⎜ ∑ ′
⎜ 𝑁 exp ((𝑀 [𝑝]𝑇 ⋅ 𝑢 )∕𝜏 ) ⎟
|𝐵| 𝑖=1 𝑙=1
∗
∗,𝑖
⎝ 𝑝=1
⎠

4

Neural Networks 195 (2026) 108292

N. Tang et al.

Fig. 3. The workﬂow of DIIR. For sample 𝑥𝑖 , we separately search the inter-camera 𝑘-nearest neighbors of global features and the intra-camera 𝑘-nearest neighbors
of local features. Their cluster assignments are then recorded in the current epoch. (1) Global reﬁnement: We compute the top-k probabilities for diﬀerent clusters.
For example, samples from the yellow cluster appear once in the top-5 of the neighbor list, while samples from the green cluster appear once in the top-2. Therefore,
the top-k probabilities for the yellow and green clusters are 1/5 and 1/2, respectively. (2) Local reﬁnement: We take the intersection of intra-camera neighboring
samples for local upper and lower features to obtain more reliable neighbors. Then, we calculate the top-k probabilities of their assigned clusters and reﬁne the local
labels.

identity, while pushing away unrelated ones. Speciﬁcally, for a given
sample 𝑥𝑖 , we compute the Euclidean distances between its global representation and those of samples from other cameras. We then sort these
distances in ascending order to obtain the inter-camera 𝑘-nearest neigh𝑔
bor list 𝑆𝑖𝑛𝑡𝑒𝑟
(𝑖, 𝑘), which stores the indices of reliable neighbors from
diﬀerent cameras, as deﬁned in Eq. (5):
𝑔
𝑆𝑖𝑛𝑡𝑒𝑟
(𝑖, 𝑘) = 𝑎𝑟𝑔𝑠𝑜𝑟𝑡(‖𝑢𝑔,𝑖 − 𝑢𝑔,𝑗 ‖2 )[∶ 𝑘], 𝑐𝑖 ≠ 𝑐𝑗

adaptively pulled toward the centroids of its neighbors’ clusters. Although the original cluster remains the primary inﬂuence, its role in
determining the pseudo-label is no longer absolute. As the model’s feature extraction capability improves, misclassiﬁed samples in the early
stage are more likely to be reassigned to their true clusters. Compared
to directly using initial hard pseudo-labels for supervised learning, this
reﬁned labeling scheme signiﬁcantly reduces the accumulation of label
noise.

(5)

where 𝑢𝑔,𝑖 and 𝑢𝑔,𝑗 denote the global features of samples 𝑖 and 𝑗, respectively, 𝑐𝑖 and 𝑐𝑗 are their corresponding camera labels, 𝑗 ranges over
all samples from cameras diﬀerent from that of sample 𝑥𝑖 , ‖ ⋅ ‖2 represents the L2 norm, and 𝑎𝑟𝑔𝑠𝑜𝑟𝑡(⋅)[∶ 𝑘] represents the indices of the top-𝑘
smallest elements in ascending order.
Based on the distribution of the pseudo-labels of these neighbors,
we calculate the probability that sample 𝑥𝑖 belongs to alternative clusters, termed as the top-k probability. The computation is formulated as
follows:
𝑃𝑖𝑛𝑡𝑒𝑟 (𝑖, 𝑙) =

𝑔
|𝑆𝑖𝑛𝑡𝑒𝑟
(𝑙|𝑖, 𝑘)|
𝑔
𝐼𝑛𝑑𝑒𝑥(𝑆𝑖𝑛𝑡𝑒𝑟
(𝑙|𝑖, 𝑘))𝑚𝑎𝑥

3.2.2. Reﬁnement for labels of local features
In order to address the issue that global features alone cannot effectively distinguish intra-camera heterogeneous samples, local features
are utilized to capture more detailed and discriminative information.
In the previous method Lan et al. (2023), global and local contrastive
learning share the same global hard pseudo-labels. However, this strategy suﬀers from several limitations. First, although global and local features originate from the same sample, global pseudo-labels cannot fully
reﬂect the underlying distribution of local features, resulting in suboptimal supervision for local-level representation learning. Second, the
inherent noise in global clustering results tends to propagate throughout the training process, creating compounded error accumulation that
degrades local feature discriminability.
To address these limitations, we reﬁne the hard labels into two sets
𝑙𝑜𝑤
of local soft labels, 𝑦̂𝑢𝑝
𝑖 and 𝑦̂𝑖 , which supervise contrastive learning
on the corresponding upper and lower local features. Due to the incomplete semantic representation of local features, they tend to be less stable than global features. This instability makes inter-camera neighbor
search prone to introducing considerable noise. To ensure the discriminability of local information while minimizing the inﬂuence of camera
style, we restrict the 𝑘-nearest neighbor search for upper and lower lo𝑢𝑝
𝑙𝑜𝑤 (𝑖, 𝑘),
cal features within the same camera, yielding 𝑆𝑖𝑛𝑡𝑟𝑎
(𝑖, 𝑘) and 𝑆𝑖𝑛𝑡𝑟𝑎
𝑝𝑎𝑟𝑡
respectively. Their intersection, 𝑆𝑖𝑛𝑡𝑟𝑎 (𝑖, 𝑘) serves as a ranked list of reliable intra-camera neighbors for sample 𝑥𝑖 , ordered by ascending feature distance. Following the same strategy as in the global reﬁnement,
we compute the top-k probabilities over the local reliable neighbors to

(6)

𝑔
𝑔
where 𝑆𝑖𝑛𝑡𝑒𝑟
(𝑙|𝑖, 𝑘) denotes the samples with label 𝑙 in 𝑆𝑖𝑛𝑡𝑒𝑟
(𝑖, 𝑘),
𝐼𝑛𝑑𝑒𝑥(⋅)𝑚𝑎𝑥 represents the maximum index value of all elements inside, and 𝑃𝑖𝑛𝑡𝑒𝑟 (𝑖, 𝑙) represents the probability of the label 𝑙 occurring
𝑔
in 𝑆𝑖𝑛𝑡𝑒𝑟
(𝑖, 𝑘), up to its last occurrence. The probability 𝑃𝑖𝑛𝑡𝑒𝑟 is determined by the distance rank of neighboring samples and the occurrence
frequency of their respective clusters, aiming to assign higher weights
to closer clusters that appear more frequently.
The reﬁned labels 𝑦̂𝑖 𝑔 are then constructed as follows:

⎧1 − 𝛼,
⎪
𝑃
(𝑖, 𝑙)
𝑦̂𝑔𝑖 [𝑙] = ⎨
𝛼 × ∑ 𝑖𝑛𝑡𝑒𝑟
,
′
⎪
′ ≠𝑦 𝑃𝑖𝑛𝑡𝑒𝑟 (𝑖, 𝑙 )
𝑙
𝑖
⎩

𝑙 = 𝑦𝑖
𝑙 ≠ 𝑦𝑖

(7)

where 𝛼 ∈ [0, 1] represents the weight for label reﬁnement. By incorporating the clustering information of reliable neighbors, sample 𝑥𝑖 is
5

Neural Networks 195 (2026) 108292

N. Tang et al.

construct reﬁned cluster-level labels for the local features, as deﬁned in
Eq. (8):
𝑝𝑎𝑟𝑡
𝑢𝑝
𝑙𝑜𝑤
𝑆𝑖𝑛𝑡𝑟𝑎
(𝑖, 𝑘) = 𝑆𝑖𝑛𝑡𝑟𝑎
(𝑖, 𝑘) ∩ 𝑆𝑖𝑛𝑡𝑟𝑎
(𝑖, 𝑘)
| 𝑝𝑎𝑟𝑡
|
|𝑆𝑖𝑛𝑡𝑟𝑎 (𝑙|𝑖, 𝑘)|
|
|
𝑃𝑖𝑛𝑡𝑟𝑎 (𝑖, 𝑙) =
( 𝑝𝑎𝑟𝑡
)
𝐼𝑛𝑑𝑒𝑥 𝑆𝑖𝑛𝑡𝑟𝑎
(𝑙|𝑖, 𝑘) 𝑚𝑎𝑥

⎧1 − 𝛼,
⎪
𝑃
(𝑖, 𝑙)
𝑢𝑝
𝑙𝑜𝑤
𝑦̂𝑖 [𝑙] = 𝑦̂𝑖 [𝑙] = ⎨𝛼 × ∑ 𝑖𝑛𝑡𝑟𝑎
,
𝑃𝑖𝑛𝑡𝑟𝑎 (𝑖, 𝑙′ )
⎪
⎩
𝑙′ ≠𝑦𝑖

(8)
𝑙 = 𝑦𝑖
𝑙 ≠ 𝑦𝑖

where 𝑃𝑖𝑛𝑡𝑟𝑎 (𝑖, 𝐿) is the top-k probability of the pseudo-label 𝑙 in
𝑝𝑎𝑟𝑡
𝑆𝑖𝑛𝑡𝑟𝑎
(𝑖, 𝑘).
We replace the original hard labels 𝑦̂𝑖 in Eq. (3) with the reﬁned
labels 𝑦̂∗𝑖 , where yielding the reﬁned cluster-level losses 𝐿′∗ as follows:
(
)
|𝐵| 𝑁 ′
exp 𝑀∗ [𝑙]𝑇 ⋅ 𝑢∗,𝑖 ∕𝜏
1 ∑∑ ∗
𝐿′∗ = −
𝑦̂𝑖 [𝑙] ⋅ log ∑ ′
(9)
(
)
𝑁
|𝐵| 𝑖=1 𝐿=1
𝑇
𝑝=1 exp 𝑀∗ [𝑝] ⋅ 𝑢∗,𝑖 ∕𝜏

Fig. 4. Identifying informative node collections. Candidate nodes marked with
a red ✓ are reliable informative nodes, while those marked with a blue × are
ﬁltered-out candidates.

The proposed total DIIR loss 𝐿DIIR is formulated by integrating the reﬁned global and local cluster-level contrastive losses. Speciﬁcally, 𝐿′𝑔 ,
𝐿′𝑢𝑝 , and 𝐿′𝑙𝑜𝑤 are substituted into Eq. (4) in place of 𝐿𝑔 , 𝐿𝑢𝑝 , and 𝐿𝑙𝑜𝑤 ,
respectively:
𝐿𝐷𝐼𝐼𝑅 = (1 − 𝜆1 )𝐿′𝑔 + 𝜆1 (𝐿′𝑢𝑝 + 𝐿′𝑙𝑜𝑤 )

the same camera - are selected as candidate informative nodes, as deﬁned in Eq. (13):
{
(
)}
|
𝐼𝐹𝑖 = 𝑢𝑔,𝑗 ∈ 𝑈𝑔 | 𝑐𝑗 ≠ 𝑐𝑖 , 𝑢𝑔,𝑗 ∈ TopK 𝑘 ⟨𝑢𝑔,𝑖 , 𝑢𝑔,𝑗 ⟩
(13)
|
where ⟨⋅, ⋅⟩ denotes the cosine similarity, and TopK 𝑘 (⟨𝑢𝑔,𝑖 , 𝑢𝑔,𝑗 ⟩) returns
the set of top-𝑘 feature vectors 𝑢𝑔,𝑗 ∈ 𝑈𝑔 that have the highest similarity
scores with the query feature 𝑢𝑔,𝑖 , excluding those from the same camera
(𝑐𝑗 ≠ 𝑐𝑖 ). The resulting set 𝐼𝐹𝑖 contains the candidate informative nodes
for the query.
For these candidates, we further compute their cosine similarity with
all other samples and extract their own top-𝑘 inter-camera neighbors
as sub-candidates. For each candidate 𝑢𝑔,𝑗 ∈ 𝐼𝐹𝑖 , we check whether the
query 𝑢𝑔,𝑖 appears in its sub-candidate set 𝐼𝐹𝑖,𝑗 . If this mutual similarity
is conﬁrmed, the candidate 𝑢𝑔,𝑗 is promoted to a reliable informative
node. This process is formalized in Eq. (14):
{
(
)}
|
𝐼𝐹𝑖,𝑗 = 𝑢𝑔,𝑛 ∈ 𝑈𝑔 | 𝑐𝑛 ≠ 𝑐𝑗 , 𝑢𝑔,𝑛 ∈ TopK 𝑘 ⟨𝑢𝑔,𝑗 , 𝑢𝑔,𝑛 ⟩
|
(14)
̃𝑖 = {𝑢𝑔,𝑗 ∈ 𝐼𝐹𝑖 ∣ 𝑢𝑔,𝑖 ∈ 𝐼𝐹𝑖,𝑗 }
𝐼𝐹

(10)

By reﬁning both global and local labels, our method eﬀectively mitigates label noise and contributes to more stable model optimization.
3.3. Aﬃnity-Guided mutual reﬁnement
Cluster-level contrastive learning focuses solely on the relationships
between instance features and their corresponding cluster centroids,
while overlooking ﬁne-grained relationships between individual sample
pairs. To address this limitation and improve representational granularity, we construct an additional instance-level memory bank 𝐼 ∈ ℝ𝑑×𝑁 on
top of the baseline. This memory bank stores global features and adopts
the same update strategy as 𝑀, as shown in Eq. (11):
𝐼[𝑖] = 𝑢𝑔,𝑖
𝐼[𝑖] = 𝑚𝐼[𝑖] + (1 − 𝑚)𝑢𝑔,𝑖

(11)

To facilitate instance-level contrastive learning, each sample 𝑥𝑖 is
initially assigned a one-hot label 𝑛̂ 𝑖 ∈ ℝ𝑁 , treating itself as the only positive. The instance-level contrastive loss is formulated as follows:
(
)
|𝐵| 𝑁
exp 𝐼[𝑗]𝑇 ⋅ 𝑢𝑔,𝑖 ∕𝜏
1 ∑∑
𝐿𝑖𝑛𝑠 = −
𝑛̂ 𝑖 [𝑗] ⋅ log ∑𝑁
(12)
(
)
𝑇
|𝐵| 𝑖=1 𝑗=1
𝑞=1 exp 𝐼[𝑞] ⋅ 𝑢𝑔,𝑖 ∕𝜏

where 𝐼𝐹𝑖,𝑗 corresponds to the sub-candidate nodes for 𝑢𝑔,𝑗 , 𝑢𝑔,𝑗 ∈ 𝐼𝐹𝑖 ,
̃𝑖 indicates the reliable informative nodes for 𝑢𝑔,𝑖 . These collecand 𝐼𝐹
tions of informative nodes, which hold the most relevant information
for sample 𝑥𝑖 , are then used to form positive pairs.

where 𝜏 is the temperature parameter. Although Eq. (12) enhances the
discriminability of individual samples, it fails to capture the underlying
relationships among them, leading to overly separated sample representations and the neglect of their semantic aﬃnities.

3.3.2. Aﬃnity-guided reﬁnement
To mitigate he inﬂuence of camera bias and enhance intra-class
compactness, we propose an aﬃnity-guided mutual reﬁnement (AGMR)
mechanism, which leverages reliable informative nodes to perform ﬁnegrained reﬁnement of the initial instance-level labels, as illustrated in

3.3.1. Informative node collections
To address the limitation of instance-level contrastive learning in
capturing semantic relationships between samples, we propose a mechanism for constructing more meaningful positive pairs. Instead of treating
each sample as its own positive, we identify a set of informative nodesreliable samples that share strong semantic similarity and discriminative
relevance with the current query sample. Paired with the query, these
informative nodes form semantically consistent positive pairs, replacing
the trivial self-positive assumption. This design encourages feature compactness among similar instances and helps to mitigate camera-induced
biases. The identiﬁcation process is illustrated in Fig. 4.
Concretely, given the feature representation of the query sample 𝑢𝑔,𝑗 ,
we compute its cosine similarity with all other samples in the instance
memory 𝐼 within the current batch, followed by normalization. The
Top-𝑘 samples with the highest similarity scores - excluding those from

Fig. 5. The workﬂow of AGMR. In each batch, we identify informative nodes
for the current sample. Sample pairs are then constructed, and their similarities are computed. After softmax processing, aﬃnity labels between samples are
obtained. (The shade of gray in the labels represents the magnitude of aﬃnity).
6

Neural Networks 195 (2026) 108292

N. Tang et al.

Fig. 5. For each query sample 𝑥𝑖 , we construct positive pairs by assõ𝑖 . The aﬃnity
ciating it with each of its reliable informative nodes 𝐼𝐹
̃𝑖 , is computed by
score 𝑠𝑐𝑜𝑟𝑒(𝑗 ∣ 𝑥𝑖 ) for each pair (𝑥𝑖 ,𝑥𝑗 ), where 𝑢𝑔,𝑗 ∈ 𝐼𝐹
measuring the cosine similarity between their global representations,
followed by a softmax normalization over all such pairs:
𝑠𝑐𝑜𝑟𝑒(𝑗 ∣ 𝑥𝑖 ) = 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥(⟨𝑢𝑔,𝑖 , 𝑢𝑔,𝑗 ⟩)

Algorithm 1 Training process of our person re-ID model.
Require: Unlabeled training data 𝑋;
Require: Initialize the encoder 𝑓𝜃 with ImageNet-pretrained ResNet50;
1: for 𝑒𝑝𝑜𝑐ℎ ∈ [1, 𝑛𝑢𝑚_𝑒𝑝𝑜𝑐ℎ𝑠] do
2:
Extract feature vector sets {𝑈𝑔 , 𝑈𝑢𝑝 , 𝑈𝑙𝑜𝑤 } from 𝑋 by 𝑓𝜃 ;
3:
Cluster {𝑈𝑔 , 𝑈𝑢𝑝 , 𝑈𝑙𝑜𝑤 } into 𝑁 ′ clusters;
4:
Initialize three cluster-level memories and one instance-level
memory;
5:
Search for 𝑘 reliable neighbors for intra-camera and inter-camera
with Eq. (5) and Eq. (8);
6: for 𝑛 ∈ [1, 𝑛𝑢𝑚_𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠] do
7:
Sample 𝑃 × 𝐾 query images from 𝑋;
8:
Reﬁne cluster-level pseudo-labels with Eq. (10);
9:
Identify informative node collections with Eq. (14);
10:
Reﬁne instance-level labels with Eq. (17);
11:
Update memories with Eq. (2) and Eq. (11);
12: end for
13: end for

(15)

̃𝑖 denotes the set of informative nodes associated with sample
where 𝐼𝐹
𝑥𝑖 . The softmax operation transforms the similarities into a probability
distribution, enabling the model to weigh each positive pair based on
relative similarity.
Subsequently, the aﬃnity label 𝐴𝑖 for sample 𝑥𝑖 is constructed as
follows:
⎧1,
𝑢𝑔,𝑖 = 𝑢𝑔,𝑗
⎪
̃𝑖
𝐴𝑖 [𝑗] = ⎨𝑠𝑐𝑜𝑟𝑒(𝑗 ∣ 𝑥𝑖 ), 𝑢𝑔,𝑗 ∈ 𝐼𝐹
⎪
𝑜𝑡ℎ𝑒𝑟𝑠
⎩0,

(16)

where 𝑠𝑐𝑜𝑟𝑒(𝑗 ∣ 𝑥𝑖 ) is the aﬃnity score between sample 𝑥𝑖 and sample 𝑥𝑗 .
In the aﬃnity label 𝐴𝑖 for sample 𝑥𝑖 , we assign a weight of 1 to sample
𝑥𝑖 itself to enforce strong self-supervision. The weights for its informative nodes are adaptively determined by their aﬃnity scores 𝑠𝑐𝑜𝑟𝑒(𝑗 ∣ 𝑥𝑖 ),
which reﬂect the semantic relevance between sample 𝑥𝑖 and each informative node. All other samples are assigned a weight of 0 to avoid introducing noise or misleading supervision signals. This selective weighting
strategy enables the model to focus on learning from high-quality and
semantically meaningful relationships while suppressing the inﬂuence
of uncertain or irrelevant samples.
The reﬁned instance-level contrastive loss is computed based on the
memory bank 𝐼 and the aﬃnity label 𝐴, with the formulation deﬁned
in Eq. (17):
|𝐵|
𝑁
∑
1 ∑
1
𝐴𝑖 [𝑗]
∑
𝑁
|𝐵| 𝑖=1
𝑛=1 𝐴𝑖 [𝑛] 𝑗=1
(
)
exp 𝐼[𝑗]𝑇 ⋅ 𝑢𝑔,𝑖 ∕𝜏
⋅ log ∑𝑁
(
)
𝑇
𝑞=1 exp 𝐼[𝑞] ⋅ 𝑢𝑔,𝑖 ∕𝜏

Our CMLR framework facilitates feature learning at diﬀerent hierarchical levels. At the cluster level, it reﬁnes pseudo-labels from both
global and local perspectives to improve clustering quality and reduce
the impact of label noise. At the instance level, an aﬃnity-guided contrastive learning strategy is employed to capture ﬁne-grained semantic relationships between samples. By integrating these complementary
strategies, CMLR enhances pseudo-label reliability and signiﬁcantly improves the model’s generalization and clustering performance in complex cross-view scenarios.

𝐿𝐴𝐺𝑀𝑅 = −

(17)
4. Experiments
4.1. Datasets and evaluation protocols

The proposed loss 𝐿𝐴𝐺𝑀𝑅 enhances instance-level contrastive learning by replacing the initial one-hot label 𝑛̂ in Eq. (12) with the aﬃnityguided label 𝐴. This reﬁned label captures ﬁne-grained, mutual relationships between each query sample and its semantically relevant neighbors. By eﬀectively drawing together cross-camera sample pairs with
high semantic similarity, 𝐿𝐴𝐺𝑀𝑅 alleviates the inﬂuence of camera bias
and leads to more accurate and robust clustering.

We evaluate the eﬀectiveness of the proposed method on two widely
recognized person re-ID datasets: Market-1501 Zheng et al. (2015) and
MSMT17 Wei et al. (2018).
Market-1501 is a prominent dataset extensively used in person re-ID
research. It consists of 32,668 images captured by six diﬀerent cameras,
oﬀering a diverse range of viewpoints and environmental conditions.
The dataset includes 1501 unique individuals, making it a comprehensive resource for studying pedestrian re-identiﬁcation under varying scenarios.
MSMT17 is another highly signiﬁcant dataset in the ﬁeld, known for
its complexity and scale. It contains 126,441 images collected from 15
cameras, providing a wide variety of viewpoints, lighting conditions,
and environmental settings. The dataset features 4101 unique pedestrians, making it a valuable resource for advanced research. Notably, in
comparison to Market-1501, MSMT17 exhibits signiﬁcantly higher complexity, characterized by greater environmental variability, increased
occlusions, and inconsistent image quality, which collectively elevate
the challenges of person re-ID.
Evaluation protocols. To evaluate the performance of the proposed method, we utilize two widely accepted metrics in person reidentiﬁcation research: mean Average Precision (mAP) and Cumulative
Matching Characteristic (CMC) accuracy. mAP assesses the overall retrieval quality by integrating both precision and recall across all ranking
positions, providing a comprehensive measure of the method’s eﬀectiveness. CMC quantiﬁes the probability of the correct match appearing within the top-ranked results. In this study, we report the Rank-1,
Rank-5, and Rank-10 accuracies of the CMC metric to provide a detailed
analysis of the method’s performance.

3.4. Joint learning
The detailed training process of our proposed method is shown in
Algorithm 1. To mitigate the impact of label noise and enhance the
reliability of pseudo-labels, global and local labels are reﬁned at the
beginning of each epoch based on the 𝑘 reliable neighbors extracted
from the multi-branch features. The updated formulations Eq. (9) and
Eq. (10) are then computed. By leveraging these reﬁned pseudo-labels
in cluster-level contrastive loss 𝐿𝐷𝐼𝐼𝑅 , the model is guided to adaptively
pull each sample closer to alternative reliable clusters. Additionally, informative node collections are identiﬁed to construct aﬃnity-guided labels for instance-level supervision. Within each batch, the aﬃnities of
sample pairs from diﬀerent cameras are dynamically updated, and an
aﬃnity-guided instance-level contrastive loss 𝐿𝐴𝐺𝑀𝑅 is formulated, as
shown in Eq. (17). The re-ID model is encouraged to pull semantically
similar samples closer, thereby promoting intra-class compactness and
eﬀectively mitigating the interference of camera variation. The total loss
function of our CMLR framework is deﬁned as follows:
𝐿𝐶𝑀𝐿𝑅 = (1 − 𝜆2 )𝐿𝐷𝐼𝐼𝑅 + 𝜆2 𝐿𝐴𝐺𝑀𝑅

(18)

where 𝜆2 is a parameter that balances 𝐿𝐷𝐼𝐼𝑅 and 𝐿𝐴𝐺𝑀𝑅 , and 𝐿𝐶𝑀𝐿𝑅
represents the total loss of our method.
7

Neural Networks 195 (2026) 108292

N. Tang et al.

Table 1
Comparison with state-of-the-art methods on Market-1501 and MSMT17. The bold font indicates the optimal
performance, and the underlined text indicates the second-best performance.
Methods

Reference

Market-1501

MSMT17

mAP

R1

R5

R10

mAP

R1

R5

R10

NeurIPS’20
ICCV’21
AAAI’22
AAAI’22
ICCV’23

76.7
80.2
81.0
83.8
84.7

90.3
91.4
92.6
93.8
93.8

96.2
–
–
97.5
97.7

97.7
–
–
98.5
98.6

26.8
29.3
31.3
35.7
36.5

53.7
56.1
60.4
67.5
66.6

65.0
–
–
77.9
75.3

69.8
–
–
81.6
80.1

MMCL Wang and Zhang (2020)
CAP Wang et al. (2021)
ICE Chen et al. (2021)
O2CAP Wang et al. (2022)
CC Dai et al. (2022)
PPLR Cho et al. (2022)
ISE Zhang et al. (2022)
CCL Zhang et al. (2023)
LP Lan et al. (2023)
DHCCN Li et al. (2024)
HCACE Luo et al. (2024)
CAJ Chen et al. (2024)
CGMAL Ran et al. (2025)

CVPR’20
AAAI’21
ICCV’21
TIP’22
ACCV’22
CVPR’22
CVPR’22
TCSVT’23
TIP’23
TCSVT’24
TMM’24
CVPR’24
PR’25

45.5
79.2
82.3
82.7
80.7
84.4
84.7
85.3
85.8
85.6
83.4
86.1
86.8

80.3
91.4
93.8
92.5
91.9
94.3
94.0
94.1
94.5
94.1
93.7
94.4
94.9

89.4
96.3
97.6
96.9
96.4
97.8
97.8
97.8
97.8
–
97.5
97.9
98.2

92.3
97.7
98.4
98.0
97.6
98.6
98.8
98.8
98.7
–
98.1
98.7
98.9

11.2
36.9
38.9
42.4
31.1
42.2
35.0
41.8
39.5
36.4
41.6
44.3
44.1

35.4
67.4
70.2
72.0
60.2
73.3
64.7
71.4
67.9
65.9
72.4
75.1
69.7

44.8
78.0
80.5
81.9
71.3
83.5
75.5
81.8
78.0
–
81.8
84.3
79.9

49.8
81.4
84.4
85.4
75.7
86.5
79.4
85.1
81.6
–
84.9
87.3
83.2

Baseline
CMLR (ours)

–
–

83.3
86.7

93.1
94.6

96.7
98.2

97.6
98.7

39.5
49.1

66.1
75.9

76.7
85.4

80.8
88.0

Unsupervised Domain Adaptation
SpCL Ge et al. (2020b)
HCD Zheng et al. (2021)
SECRET He et al. (2022)
MCRN Wu et al. (2022)
CaCL Lee et al. (2023)
Fully Unsupervised

4.2. Implementation details

the-art UDA approach CaCL Lee et al. (2023), which also utilizes camera information without any annotations. Notably, on the challenging
MSMT17 dataset, our model achieves a 12.6 % improvement in mAP
and a 9.3 % increase in Rank-1 accuracy.
Comparison with fully unsupervised methods. In the ﬁeld of fully
unsupervised person re-identiﬁcation, our CMLR has demonstrated remarkable performance, signiﬁcantly surpassing the current state-of-theart methods. Compared to methods such as CAP Wang et al. (2021), ICE
Chen et al. (2021), O2CAP Wang et al. (2022), PPLR Cho et al. (2022),
CCL Zhang et al. (2023), HCACE Luo et al. (2024), CAJ Chen et al.
(2024) and CGMAL Ran et al. (2025), which also utilize camera labels,
CMLR achieves signiﬁcant improvement by making full use of camera
information to alleviate label noise and camera bias. On the Market1501 dataset, our method achieves mAP and Rank-1 scores of 86.7 %
and 94.6 %, respectively, while on the MSMT17 dataset, it achieves mAP
and Rank-1 scores of 49.1 % and 75.9 %, signiﬁcantly surpassing the
best-known results of existing methods. For feature partitioning methods like PPLR Cho et al. (2022), DHCCN Li et al. (2024), and LP Lan et al.
(2023), CMLR enhances pseudo-label reliability through ﬁne-grained exploration of global and local features, leading to superior optimization
outcomes.
It is evident that our proposed method, CMLR, demonstrates robust
performance across both datasets utilized in the experiments, with particularly outstanding results on the MSMT17 dataset. We attribute this
superior performance to the inherently more complex composition of
the MSMT17 dataset and the signiﬁcant presence of label noise, both
of which are eﬀectively mitigated through the optimization strategies
implemented in our model.

Our framework employs ResNet-50 He et al. (2016), initialized with
weights pre-trained on ImageNet Deng et al. (2009), as the backbone
encoder. To adapt the network for our task, we modify the architecture by removing all layers beyond layer-4 and adding a global
generalized mean pooling layer, followed by batch normalization and
L2-normalization layers. During the training phase, pseudo-labels are
dynamically generated at the beginning of each epoch using the DBSCAN clustering algorithm. For inference, computational eﬃciency is
prioritized by extracting feature vectors exclusively from the ﬁrst global
feature branch.
Input images are resized to 256 × 128, and data augmentation techniques such as random horizontal ﬂipping, random cropping, and random erasing Zhong et al. (2020) are applied. The training process employs a mini-batch size of 256, comprising 16 pseudo-classes with 16
images per class. Optimization is performed using the Adam Kingma
and Ba (2014) optimizer with a weight decay of 5 × 10−4 and an initial learning rate of 3.5 × 10−4 , which is reduced by a factor of 10 every
20 epochs. The model is trained for a total of 50 epochs, with each
epoch comprising 200 iterations for the Market-1501 dataset and 400
iterations for the MSMT17 dataset. Following LP Lan et al. (2023), we
empirically set the parameters as: 𝑚 = 0.2, and 𝜏 = 0.05. In our method,
we set 𝜆1 = 0.2, 𝜆2 = 0.6, 𝛼 = 0.08, and 𝑘 = 4 for Market-1501 and 8 for
MSMT17.
4.3. Comparison with state-of-the-art
We compare our CMLR with the state-of-the-art re-ID methods on
the Market-1501 and MSMT17 datasets. The comparison results are presented in Table 1.
Comparison with UDA-based methods. Recent advances in Unsupervised Domain Adaptation (UDA) for person re-identiﬁcation have
demonstrated signiﬁcant progress by incorporating additional labeled
data to enhance model performance, such as MMCL Wang and Zhang
(2020), SpCL Ge et al. (2020b), HCD Zheng et al. (2021), SECRET He
et al. (2022), MCRN Wu et al. (2022) and CaCL Lee et al. (2023). Surprisingly, our proposed method signiﬁcantly outperforms the state-of-

4.4. Ablation studies
In this subsection, we validate the eﬀectiveness of diﬀerent components in our proposed method. The experimental results are shown in
Table 2.
Our baseline is a simpliﬁed variant built upon LP Lan et al. (2023),
where we remove the student training module and generate pseudolabels solely based on the Jaccard distance computed from global
features, as described in Section 3.1. It does not include any feature
8


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 3, MARCH 2026

2999

Unsupervised Person Re-Identification With
Diffusion Model via Semantic-Aware
Disentanglement Representation Learning
Xuefeng Tao , Jun Kong , Member, IEEE, Min Jiang , Member, IEEE, Jiayi Li ,
and Ajmal Mian , Senior Member, IEEE

Abstract—Unsupervised person re-identification (Re-ID)
requires learning semantic representation without identity
labels. Existing methods entangle identity-related person
features with camera-related background features, hindering
discriminative feature learning. Also, these methods often
disrupt the semantic structure of the person, weakening
the semantic representation. In this paper, we propose the
Semantic-Aware Disentanglement Representation Learning
(SDRL) framework with diffusion models for unsupervised
person Re-ID. Firstly, to enhance feature learning, we propose
the Disentanglement Aggregation Model (DAM). This model
disentangles identity-related features from camera-related
features to generate multi-view features. Secondly, to promote
the consistency of multi-view features, we design the multi-view
similarity consistency (MSC) loss to constrain intra-camera
and cross-camera similarity distributions. Thirdly, to generate
semantically meaningful patches, we propose the Semantic
Spatial Diffusion Model (SSDM). This model operates on
identity-related features to perform the denoising diffusion
process over spatial transformer parameters. Finally, to further
enhance the semantic representation of generated patches, we
design the Semantic Decoupled Contrastive (SDC) loss to perceive
the inherent semantic structure. Numerous experiments on three
demanding datasets prove that our approach is superior to the
current unsupervised Re-ID approaches. The source code will be
publicly available at https://github.com/taoxuefong/SDRL-reid
Index Terms—Unsupervised person re-identification, semanticaware disentanglement representation learning, disentanglement
aggregation model, semantic spatial diffusion model.

Received 8 August 2025; revised 25 September 2025; accepted 10 October
2025. Date of publication 14 October 2025; date of current version 9 March
2026. This work was supported in part by the National Natural Science
Foundation of China under Grant 62371209 and Grant 62371208, in part
by the Postgraduate Research and Practice Innovation Program of Jiangsu
Province (the Fundamental Research Funds for the Central Universities) under
Grant KYCX24 2515, and in part by the 111 Projects under Grant B12018.
This article was recommended by Associate Editor H. Liu. (Corresponding
author: Jun Kong.)
Xuefeng Tao and Jun Kong are with the Key Laboratory of Advanced Process Control for Light Industry, Ministry of Education, Jiangnan University,
Wuxi 214122, China (e-mail: kongjun@jiangnan.edu.cn).
Min Jiang and Jiayi Li are with the Engineering Research Center of Intelligent Technology for Healthcare, Ministry of Education, Jiangnan University,
Wuxi 214122, China.
Ajmal Mian is with the School of Physics, Mathematics and Computing,
Department of Computer Science and Software Engineering, The University
of Western Australia, Crawley, WA 6009, Australia.
Digital Object Identifier 10.1109/TCSVT.2025.3621439

I. I NTRODUCTION
ERSON re-identification (Re-ID) [1], [2] aims to match
the same identity across different cameras. Supervised
methods [3], [4] have achieved excellent performance, but
necessitate vast quantities of labeled data, limiting their
scalability in real-world scenarios. Therefore, unsupervised
methods [5], [6] have gained popularity to handle unlabeled
data.
Unsupervised person Re-ID often generates pseudo-labels
through clustering [7], [8] or k-nearest neighbor search [9].
However, these pseudo-labels inherently contain noise, which
impedes effective discriminative learning. To improve pseudolabels purity, methods such as pseudo-label refinement [10]
or robust clustering [11] are employed. Nonetheless, the features extracted by these methods still entangle identity-related
person features with camera-related background features. As
shown in Fig. 1 (a), the background and obstacles captured
by different cameras show substantial variation. These entangled features combine background and identity information,
resulting in large cross-camera distances for the same identity
and small distances between different identities with similar
backgrounds and appearances. Recent methods [12], [13] aim
to disentangle shallow-layer features into identity-related and
style-related components. However, they overlook the finegrained semantic information that is crucial for accurately
representing identity-related features.
In addition, patch features containing semantic information
can help reduce noise in global feature clustering [14], [15].
However, real-world scenarios often present challenges such
as variations in viewpoints, poses, and the absence of prior
knowledge. Uniform slicing methods [6], [16] divide person
images into multiple patches, disrupting the inherent semantic
structure. Pose estimation [17], [18], [19] and segmentation
methods [20] depend on human pose knowledge and segmentation priors, which limits their generalization and increases
computational costs. As illustrated in Fig. 2 (a), the Spatial
Transformer Network (STN) [21] directly generates spatial
transformer parameters to sample patch features. However,
STN’s architecture is limited in generating semantically meaningful patches, as it lacks access to semantic labels and relies
solely on low-level spatial transformations without understanding the semantic contents. Consequently, in unsupervised
Re-ID, STN alone is insufficient to capture the semantic

P

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:33 UTC from IEEE Xplore. Restrictions apply.

3000

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 3, MARCH 2026

Fig. 1. The illustration of our motivation using disentanglement learning for
unsupervised person Re-ID. Entangled features mix background and identity
information, leading to large intra-class distances across cameras and small
inter-class distances in similar backgrounds. SDRL disentangles these features
into camera-related features and identity-related features. Concentric circles
depict distances in the feature space relative to the query. The shapes of the
embedding vectors signify different feature spaces.

Fig. 2. An illustration of our motivation using diffusion model for unsupervised person Re-ID. In Fig. (a), the Spatial Transformer Network (STN)
generates spatial transformer parameters to sample patch features, but lacks
semantic guidance and relies only on low-level spatial transformations.
In Fig. (b), we operate on identity-related features to perform the denoising
diffusion process over spatial transformer parameters, where q represents the
diffusion stage and p represents the reverse stage.

structure accurately. Directly applying the generated spatial
transformer parameters to affine transformations can disrupt
the semantic integrity of the patches.

To generate semantically meaningful patches within available computational resources, we propose the Semantic-Aware
Disentanglement Representation Learning (SDRL) framework,
which integrates disentanglement learning and diffusion model
into unsupervised Re-ID. Our motivation is shown in Fig. 1
and Fig. 2. As shown in Fig. 1 (b, c), separating identityrelated features from camera-related features forces the model
to concentrate solely on identifying the target person. In
unsupervised Re-ID, this disentanglement learning mitigates
interference from factors such as camera viewpoints and
backgrounds. This enhances discriminability, and improves
generalization across cameras. As illustrated in Fig. 2 (b),
rather than applying computationally expensive diffusion models to entire images, we condition it on spatial transformer
parameters derived from disentangled identity-related features.
These parameters allow for affine transformations tailored
to specific viewpoints and poses, thereby generating semantically meaningful patches while maintaining computational
efficiency. Additionally, the inference process starts from randomly initialized spatial transformer parameters, eliminating
the need for additional networks or prior knowledge, which
further reduces computational overhead and enhances model
generalization.
In this paper, we first present the Disentanglement Aggregation Model (DAM), which disentangles the identity-related
features from the camera-related features. Unlike existing
disentanglement methods that only separate features at the
shallow layer, DAM operates on semantic features and explicitly trains the model on cross-camera scenarios. DAM relies on
person detection boxes, which are directly available in Re-ID
datasets, to generate precise person masks for switching source
views and enhanced views. Since DAM is trained end-to-end
with the Re-ID task, the model can learn to compensate for
noisy masks, making it less sensitive to segmentation results.
Second, building upon the disentangled features from DAM,
we design the Multi-view Similarity Consistency (MSC) loss
to constrain feature similarity distributions for intra-camera
and cross-camera matching. This loss constrains the distance
between source and enhanced views within each class, thereby
extracting camera-invariant features. Third, we propose the
Semantic Spatial Diffusion Model (SSDM) that formulates
patch generation as a denoising diffusion process over spatial
transformer parameters. SSDM exploits the generative modeling capabilities of diffusion models to learn the underlying
distribution of identity-related features, enabling the generation of semantically meaningful patches. Finally, to ensure the
quality of patches generated by SSDM, we design the Semantic Decoupled Contrastive (SDC) loss to strengthen semantic
representation through decoupled contrastive learning across
adjacent denoising steps. Our contribution is summarized as
follows:
1) DAM is presented to disentangle identity-related and
camera-related features for enhanced multi-view feature
generation. To ensure consistency across multi-view
features, we design the MSC loss to enforce consistent
similarity distributions for both intra-camera and crosscamera matching.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:33 UTC from IEEE Xplore. Restrictions apply.

TAO et al.: UNSUPERVISED PERSON RE-IDENTIFICATION WITH DIFFUSION MODEL VIA SDRL

2) SSDM is proposed to perform a denoising diffusion
process over spatial transformer parameters to sample
patches. To ensure that SSDM generates semantically meaningful patches, we design the SDC loss to
strengthen semantic representation through decoupled
contrastive learning across two adjacent denoising steps.
3) Experiments validate the efficacy of the proposed SDRL
framework and achieve competitive performance on
Market-1501 [22], DukeMTMC-reid [23], [24], and
MSMT17 [3].
II. R ELATED W ORK
A. Feature Disentanglement
Feature Disentanglement [25], [26] is mainly used to synthesize images or separate features, but it struggles with
the discriminative learning for specific tasks. Recently, some
methods have applied flexible clustering to enhance discriminative learning. Li et al. [27] calibrate features using a
nonparametric graph attention network before pseudo-label
generation. Xiong et al. [28] dynamically adjust the clustering threshold and use weighted regularization to improve
pseudo-label accuracy. However, these methods have not
effectively isolated identity-related features from confounding
background or occluding factors. Subsequently, Sun et al.
[13] disentangle the shallow layer features into identity-related
features and style-related features. Ji et al. [29] separate
person images into identity-relevant and identity-irrelevant
factors, creating disentangled positive and negative groups.
However, these methods fail to capture fine-grained semantic
details tied to identity-related features, and overlook the consistency of similarity distributions both within and between
cameras. Unlike previous methods, we propose the DAM to
switch identity-related and camera-related features, enabling
the aggregation of enhanced multi-view features. In addition,
we propose the MSC loss to enforce consistency of similarity
distributions, improving cross-camera generalization.
B. Patch-Based Unsupervised Person Re-ID
Fine-grained patch features play a crucial role in enhancing
discriminative representation for unsupervised Re-ID. Several
approaches [14], [30] have been presented to generate patch
features. In [16], uniform partitioning disrupts semantic structure as divisions may not align with semantics. To preserve
it, pose-based methods [4], [31] segment features by joints
and limbs, better handling variations but relying on potentially
unavailable pose information. Segmentation methods [20],
[32] provide fine-grained patches but increase complexity
with pixel-wise parsing. Appearance reconstruction methods
[33] extract cloth-independent features but may lose intrinsic
identity information. Recently, Hu et al. [34] leverage pairwise semantic information as additional supervisory signals
to enhance representation learning. The Spatial Transformer
Network (STN) [21] has also been introduced to sample
patch features. However, these methods require pre-training
with identity labels to learn the feature distribution, and
they struggle to capture the semantic structure accurately in
unsupervised settings. In contrast, we propose the SSDM,

3001

which operates identity-related features to formulate patch
generation as a denoising diffusion process. Inference starts
directly from random spatial transformer parameters, allowing
the sampling of semantic patches without relying on additional
networks or prior knowledge.
C. Controllable Generation With Diffusion Models
Diffusion models are widely employed to generate realistic
and diverse images from random noise [35], [36]. They adjust
the generated content and style by conditioning on various
inputs [9], [37]. However, most methods [38], [39] offer
limited control over the generated content. Therefore, some
methods introduce guidance signals for explicit control [40],
[41]. They control content generation by fine-tuning models
[42] or manipulating pre-trained models [43]. However, these
methods directly diffuse on the entire images or features,
which requires costly training on carefully designed datasets.
In this paper, our proposed SSDM is conditioned on spatial
transformer parameters to control the content of the generated patches. This reduces computational complexity while
enhancing the generalization. In addition, to further guide
the diffusion model, we design the SDC loss to refine the
parameters to sample semantically meaningful patches.
III. P ROPOSED M ETHOD
A. The Overall Architecture
As shown in Fig. 3, we propose the Semantic-Aware Disentanglement Representation Learning (SDRL) framework for
unsupervised person Re-ID. We ˚access unlabeled source views
N
with camera information S = si,c i=1 , where si,c represents
the image of the i-th person from camera c (c = 1, . . . , Nc ,
where Nc is the total number of cameras). We design the
DAM to disentangle the identity-related features from the
camera-related features, generating enhanced views E = ei,c .
The source and enhanced views, xi = {si , ei }, are then fed
into a Resnet-50 network to extract shared representations
F(xi ) ∈ RC×H×W , where C, H, and W denote the number of
channels, height, and width, respectively. Getting this feature
maps, the global features F(xig ) are obtained through global
average pooling. We design the MSC loss to ensure consistent
similarity distributions for source and enhanced views. In
addition,
We propose the SSDM to generate P patch features
˚
p P
F(xi ) p=1 . During training, semantic parameters diffuse from
the ground-truth generated by Spatial Transformer Network
(STN) [21] to noisy parameters, and the model reverses this
diffusion process to discover semantic information. During
inference, starting from random spatial transformer parameters, we aim to refine them to sample semantically meaningful
patches. To further guide the diffusion model, we design the
SDC loss to improve the semantic representation of generated
patches.
We use DBSCAN [44] to assign pseudo-labels yi ∈ RK to
the image (si , ei ). These pseudo-labels are then used to learn
both global and patch-level discriminative representations by
computing the cross-entropy loss Lces [45] and softmax-triplet
loss Ltri [46]. The overall loss function LS DRL is:
LS DRL = Lces + Ltri + λmsc Lmsc + λ sdc L sdc ,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:33 UTC from IEEE Xplore. Restrictions apply.

(1)

3002

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 3, MARCH 2026

Fig. 3. The framework of our proposed SDRL. DAM is designed to disentangle the identity-related features from the camera-related features to generate
enhanced multi-view features. Source views and enhanced views are split into multiple cameras and fed into the SSDM. During the training stage, we use STN
to obtain spatial transformer parameters and add Gaussian noise to produce the noisy parameters. The loss functions Lces , Ltri , Lmsc and L sdc are combined
for training. During the inference stage, SSDM generates the semantic patches by reversing the diffusion process, which adjusts a noisy prior distribution to
a learned distribution over spatial transformer parameters.

where λmsc and λ sdc are the weight parameters that balance the
importance of each loss terms. The specific details of DAM,
SSDM, MSC and SDC are shown in the rest of this section.
B. Disentanglement Aggregation Model (DAM)
Feature disentanglement has primarily been applied to
image synthesis and feature separation, but it faces challenges
in discriminative learning for specific tasks. Recent methods
in person Re-ID focus on disentangling fine-grained components or features but often struggle to effectively capture
identity-related features. To address this limitation, we present
the DAM, which disentangles identity-related features from
camera-related features to generate enhanced multi-view features, providing diverse training samples for the subsequent
feature learning. The Segment Anything Model (SAM) [47]
can produce accurate masks from input prompts like boxes
or points. For person Re-ID datasets, bounding boxes are
conveniently generated by detectors, such as the Deformable
Part Model (DPM) [48] in Market-1501 [22] and Faster RCNN [49] in MSMT17 [3]. Therefore, we can also use
detectors to obtain bounding boxes and then apply SAM to
generate person masks, making the approach feasible in realworld scenarios. Specifically, we apply SAM to segment the
source images si and s j from different cameras by entering

an easily accessible person detection box, resulting in binary
masks mi and m j . The masked images are denoted as:
Mi = si

mi ,

(2)

Mj = sj

m j,

(3)

where
denotes the element-wise multiplication. The seggi and
mented person that are masked off are represented as M
g
gi
M j . To place the segmented person in a new context, M
g
and M j are overlaid on the masked images M j and Mi to
obtain aggregation results. The mask mi , m j is stacked with
the aggregation results, leading to a four-channel input tensor:
gi ⊕ M j , m j ),
s0i = stack(M
gj ⊕ Mi , mi ),
s0j = stack(M

(4)
(5)

where stack(·) denotes stacking of feature maps into a higherdimensional tensor, and ⊕ denotes the concatenation operation.
Getting s0i , s0j , the Aggregation Network fan (·) is able to inpaint
the rest of the image such that it is consistent with the person
Re-ID task.
To fill large-sized masks, generating effective image inpainting results requires considering the global features. The model
should utilize units with the largest receptive field as early as
possible. Conventional fully convolutional models lack global

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:33 UTC from IEEE Xplore. Restrictions apply.

TAO et al.: UNSUPERVISED PERSON RE-IDENTIFICATION WITH DIFFUSION MODEL VIA SDRL

3003

se
within-class and cross-camera distance dcd
between the source
views and the enhanced views is calculated:
se
=k F(si,c ) − F(ei,c0 ) k,
dcd

(8)

where k·k denotes the L2 -norm, si,c represents source view
images, and ei,c0 represents enhanced view images captured
by different camera c and c0 . Both the within-class and crossss
camera distance dcd
between the source views is calculated:
ss
=k F(si,c ) − F(s j,c0 ) k, i , j.
dcd

Fig. 4. The proposed Aggregation Network aims to generate enhanced views
by camera switching and aggregating.

feature and waste computational resources. The Aggregation
Network addresses this by incorporating global features in
the early layers. As shown in Fig. 4, it relies on a channelwise fast Fourier transform (FFT) [50] and has a receptive
field that spans the whole image. The branch divides the
channels into two parallel networks: the local network employs
conventional convolutions, while the global network utilizes
real FFT to capture global features. Since real FFT processes
the real-valued features, the inverse real FFT guarantees the
output remains real-valued. Compared to standard FFT, real
FFT utilizes only half of the spectrum. The outputs from both
branches are then fused. The Aggregation Network addresses
the source views using the fully convolutional manner, producing three-channel enhanced views ei and e j :
ei = fan (s0i ),
e j = fan (s0j ).

(6)
(7)

It is fully differentiable and can seamlessly replace conventional convolutions as a drop-in solution. With its image-wide
receptive field, the network allows the generator to incorporate
global features from the early layers, which is essential for
person image aggregating.
C. Multi-View Similarity Consistency (MSC) Loss
Building upon the enhanced multi-view features generated
by DAM, we propose the MSC loss to ensure the consistency
of multi-view features and reduce the distribution discrepancy
across cameras. The proposed DAM manipulates the source
views to generate enhanced multi-view features, which serve
as the foundation for our MSC loss computation. The reliable
similarity distribution between the source and enhanced views
guides the feature distribution of the original images. While the
distribution constraint indirectly optimizes feature similarity,
it overlooks intra-class variations between different instances.
Instances of the same class continue to exhibit a considerable
distance from each other. As a result, we incorporate both
distribution and instance constraints simultaneously.
As shown in Fig. 3, for distribution constraint, the crosscamera and intra-camera distance distributions are calculated
respectively. Specifically, source view features F(si,c ) and
enhanced view features F(ei,c ) are extracted. Both the

(9)

where si,c and s j,c0 represent different source view images
captured by different cameras c and c0 . Both the within-class
and intra-camera distance didse between the source views and
the enhanced views is calculated:
didse =k F(si,c ) − F(ei,c ) k .

(10)

where si,c and ei,c represent source view images and enhanced
view images captured by the same camera c. Both the withinclass and intra-camera distance didss between the source views
is calculated:
didss =k F(si,c ) − F(s j,c ) k, i , j.

(11)

where si,c and s j,c represent different source view images
captured by the same camera c. We use Maximum Mean
Discrepancy (MMD) [51] to mitigate the discrepancy of the
cross-camera and intra-camera distance distributions. The distribution constraint Ldis
msc is calculated as follows:
se
ss
se ss
Ldis
msc = MMD(dcd , dcd ) + MMD(did , did ).

(12)

As shown in Fig. 5, for instance constraint, we calculate
the cross-camera matching and intra-camera matching respectively. We maintain memory bank W s , W e = wis , wei to update
source and enhanced features. The features in the key memory
are updated with the training samples si and ei by:
{wis , wei } ← ε{wis , wei } + (1 − ε)F(si , ei ),

(13)

where ε ∈ [0, 1] denotes the updating rate of memory bank. For
cross-camera matching, we utilize DAM to switch the source
features si,c to generate enhanced features ei,c0 :
ei,c0 = DAM(si,c ), c0 , c.

(14)

For each generated ei,c0 , suppose wc0 = wcs0 , wec0 denotes the
set of features whose camera labels match that of ei,c0 . The
cross-camera matching merely considers the features within
wc0 . Therefore, we can get the set Ki,c0 of its nearest-neighbors
on the whole dataset. The cross-camera matching is formulated
as follows:
Lcc
msc =

Nc
X
c0 =1

P

exp((Ki,c0 )T F(ei,c0 )/τ)
.
T
j<Ki,c0 exp((w j,c0 ) F(ei,c0 )/τ)

(15)

τ denotes the temperature factor, set to 0.1.
For each source image si,c , suppose wc = wcs , wec denotes
the set of features whose camera labels match that of si,c . The
intra-camera matching merely considers the features within
wc . Therefore, we can get the set Ki,c of its nearest-neighbors

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:33 UTC from IEEE Xplore. Restrictions apply.

3004

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 3, MARCH 2026

Fig. 6. The illustration representation of the enhanced (blue) and source
(orange) distance distributions. Fig. (a) and (c) show that the cross-camera
and intra-camera distance distribution are significant differences between
the source and enhanced views. Fig. (b) denotes the cross-camera distance
distribution with the proposed MSC to pull the distribution closer. Fig. (d)
depicts the distance distribution using Lmsc to ensure the consistency between
the source and enhanced views.

Fig. 5. Illustration of camera instance constraint. The green border denotes
positive matching, while the red border denotes negative matching. For crosscamera matching, we manipulate the source views to generate the enhanced
views of the corresponding camera, thus pulling the high similarity positive
sample closer and pushing the low similarity negative sample away under
the same camera view. For intra-camera matching, we constrain the instance
similarity relationship of the same camera view in both the source views and
the enhanced views.

on the whole dataset. The intra-camera matching is formulated
as follows:
Lic
msc = P

exp((Ki,c )T F(si,c )/τ)
.
T
j<Ki,c exp((w j,c ) F(si,c )/τ)

(16)

The instance constraint Lins
MS C is formulated as follows:
ic
cc
Lins
msc = Lmsc + Lmsc .

(17)

So our final MSC loss is:
ins
Lmsc = Ldis
msc + Lmsc .

(18)

As shown in Fig. 6, by minimizing Lmsc , the pair-wise
distance distributions are consistently aligned, ensuring that
the distance distributions between the source and enhanced
multi-views more consistent.
D. Semantic Spatial Diffusion Model (SSDM)
We propose the SSDM to introduce diffusion models
for sampling semantically meaningful patch features. Unlike
traditional spatial sampling methods that apply predefined
transformations, our SSDM learns the underlying distribution
of identity-related features by performing denoising diffusion
process over spatial transformer parameters. SSDM integrates

STN and diffusion models, where the STN is composed of
three parts including a localization network, a patch sampling
grid, and a sampler. The localization network predicts P
spatial locations, parameterized by a set of spatial transformer
parameters Θ = [θ1 , . . . , θ p , . . . , θ P ]. Each parameter is used to
compute a patch sampling grid, which determines the sampling
points on the input feature map to create the patches. The
sampler extracts P patches from each feature map. However,
directly sampling patch features with the generated parameters can distort the semantic content of the person images.
Therefore, we incorporate diffusion models to learn and refine
these parameters through generative modeling, enabling them
to better capture the semantic structure and generate semantic
parameters. Specifically, to minimize background interference,
we mask the background, allowing the model to focus on
task-related features. The localization network takes the input
feature map U = F(xi ) ⊗ mi and outputs Θ:
Θ = floc (F(xi ) ⊗ mi ).

(19)

The size of Θ relies on the dimensions of the transformation being parameterized. The localization network floc (·) can
receive various forms but must include a final regression
layer to obtain the spatial transformer parameters Θ. We
utilize the diffusion model to define a Markovian chain for
the forward diffusion process q, progressively introducing
noise to the spatial transformer parameters. Beginning with an
initial parameter distribution Θ0 ∼ q(Θ0 ), the model generate
parameter samples Θ0 , Θ1 , . . . , ΘT at timesteps t ∈ [0, T ]. The
noise added at each timestep is governed by the variance
βt ∈ (0, 1). The forward diffusion process is defined as:
q(Θ1:T |Θ0 ) :=

T
Y

q(Θt |Θt−1 ),

(20)

t=1

q(Θt |Θt−1 ) := N (Θt ;

p
1 − βt Θt−1 , βt I).

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:33 UTC from IEEE Xplore. Restrictions apply.

(21)

TAO et al.: UNSUPERVISED PERSON RE-IDENTIFICATION WITH DIFFUSION MODEL VIA SDRL

We can sample parameter Θt at an arbitrary timestep t without
the need of applying q repeatedly:
p
√
(22)
q(Θt |Θ0 ) := ᾱt Θ0 + ς 1 − ᾱt , ς ∈ N (0, I),
Qt
where ᾱt := i=0 α s and αt := 1 − βt .
In the reverse process, we obtain a sample from q(Θ0 )
by first sampling from q(ΘT ) and performing the reverse
steps q(Θt−1 |Θt ) until reaching q(Θ0 ). The distribution of
q(ΘT ) approximates an isotropic Gaussian distribution when
T is sufficiently large and with a well-designed schedule for
βt (βt → 0), making it straightforward to sample ΘT ∼ N (0, I).
Since computing q(Θt−1 |Θt ) would require knowledge of the
entire data distribution, we approximate it using a neural
network trained to predict the mean µ and the diagonal
covariance matrix Σ:
p(Θt−1 |Θt ) := N (Θt−1 ; µ(Θt , t), Σ(Θt , t)).

(23)

During training, the network F(Θt , t) is optimized to predict
the semantic parameters Θ0 from the noisy inputs Θt by
minimizing the training objective with L scd . This enforces
the preservation of semantic information necessary to generate meaningful and discriminative spatial patches. Unlike
conventional data augmentation, this process leverages the
generative modeling capabilities of diffusion models to learn
a structured semantic space of spatial transformations. During
inference, the model F iteratively denoises a noise sample
ΘT to reconstruct Θ0 , enabling the exploration of diverse and
semantic transformation parameters beyond those seen during
training. This generative process enables SSDM to move
beyond simple perturbations by modeling semantic spatial
transformations, thereby enhancing representation learning.
E. Semantic Decoupled Contrastive (SDC) Loss
Complementing the SSDM’s patch generation capabilities,
we propose the SDC loss to capture semantic structure of
various person images from the spatial transformer parameters.
SDC encourages the patches produced at each denoising step
to closely resemble the original patches before the diffusion
process. This approach improves the flexibility of the diffusion
model, allowing it to better adapt to controlled semantic patch
generation. Specifically, we introduce a decoupled contrastive
learning paradigm for evolving the semantic parameters over
the spatial transformer parameters. This paradigm performs
contrastive learning on the parameter pairs (Θt−1 , Θt ), with
different levels of added Gaussian noise, to extract the target
features of the original input Θ0 .
Each generated parameter is used to compute a sampling
grid. To apply a spatial transformation to the input feature
map, a sampler takes the set of sampling points Tθ (·) along
with the input feature map, producing the corresponding
sampled output. This transformation enables the application of
translation, rotation, and scaling operations to the input feature
map using the generated parameters Θ:
i
P
{F(xt−1
)}i=1
, {F(xtj )}Pj=1 = Tθ (F(xi ) ⊗ mi , Θt−1 , Θt ).

(24)

Sampling is conducted uniformly across each input channel,
ensuring that all channels undergo consistent transformation.

3005

This creates a differentiable sampling process, enabling loss
gradients to propagate back to the spatial transformer parameters Θ. For a specific patch feature F(xti ), there are 2P − 1
i
pairs in total.
corresponding feature F(xt−1
) form a
˚ The
i
i
positive pair F(xt−1 ), F(xt ) , while the other 2P − 2 pairs are
considered negative. The L2 distance is employed to calculate
the similarity between different patch features:
i
i
) − F(xtj ) ,
d(xt−1
, xtj ) = F(xt−1

(25)

where i, j ∈ [1, P]. The following loss is utilized to adaptively
perceive the semantic properties of different person images
by distinguishing patch feature F(xti ) from all others except
i
F(xt−1
):
i
exp(d(xt−1
, xti ))

i
lt−1
= −log PP

j
j
i
i
j=1 exp(d(xt−1 , xt−1 )) + exp(d(xt−1 , xt ))

. (26)

By traversing all semantic patches, the SDC is finally computed as follows:
P

L sdc =

1 X i
(lt−1 + lti ).
2P

(27)

i=1

Algorithm 1 SDRL With Diffusion Model
Require: unlabeled source view S = si,c , enhanced view E =
ei,c and camera label c
Input: Φ . Semantic Spatial Diffusion ModelF(si,c , ei,c ) ∈
RC×H×W
.
image
space
mappingsΘ
=
[θ1 , . . . , θ p , . . . , θ P ] . spatial transformer parameters
1: for t = T , . . . , 1 steps do
2:
The forward noise process is defined with Eq. (22)
3:
The reverse process is defined with Eq. (23)
4:
The pointwise transformation
5:
Perform spatial transformation
6: end for
7: for n ∈ [1, E] epochs do
8:
for each mini-batch B ⊂ S do
9:
generate enhanced views with Eq. (6) and (7)
10:
Compute distribution constraint Ldis
msc with Eq. (12)
11:
Maintain a memory bank with Eq. (13)
12:
Compute instance constraint Lins
msc with Eq. (17)
13:
Compute the MSC loss Lmsc with Eq. (18)
14:
Manipulate patch generation process with Eq. (24)
15:
Measure the similarity between different patch features with Eq. (25)
16:
Compute the SDC loss L sdc with Eq. (27)
17:
Optimize SDRL based on overall LS DRL Eq. (1)
18:
end for
19: end for
The steps for unsupervised training based on SDRL are
outlined in Algorithm 1. First, we propose DAM to generate
enahnced views. The diffusion model is introduced to design
SSDM and take T iterations to produce patch features. Second,
E epochs are required during training. Lmsc is computed to
enhance intra-class compactness by utilizing the enhanced
multi-view images. L sdc is computed to capture the semantic patches from the spatial transformer parameters. Finally,
SDRL is optimized using the overall loss LS DRL Eq. (1).

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:33 UTC from IEEE Xplore. Restrictions apply.

3006

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 3, MARCH 2026

Fig. 7. Evaluation of λmsc and λ sdc in Market-1501 and MSMT17 datasets.

Fig. 8. Evaluation of P in Market-1501 and MSMT17 datasets.

IV. E XPERIMENTS
A. Datasets and Experimental Settings
We evaluate the performance of our method using three
widely employed person Re-ID datasets: Market-1501 [22],
DukeMTMC-reid [23], [24], and MSMT17 [3]. Market-1501
[22] is composed of 32,668 images from 1,501 identities
captured across 6 cameras. The dataset is divided into 12,936
training images (751 identities) and 19,732 testing images
(750 identities). DukeMTMC-reid [24] is derived from the
DukeMTMC dataset [23] and is composed of 36,411 images of
1,404 identities from 8 cameras. It is divided into 16,522 training images (702 identities), 2,228 query images, and 17,661
gallery images. MSMT17 [3] is composed of 126,411 images
from 4,101 identities captured by 15 cameras (12 outdoor and
3 indoor). This dataset is challenging due to its diverse data
collected across multiple time periods with significant lighting
variations, reflecting more realistic, real-world conditions. For
evaluation, we follow the standard Re-ID protocol and report
the Cumulative Matching Characteristics (CMC) for Rank-1,
Rank-5, and Rank-10 accuracies, as well as mean Average
Precision (mAP), without applying re-ranking.
B. Implementation Details
We employ ResNet-50 as the backbone pre-trained on
ImageNet [52]. SAM [47] is employed to generate accurate masks from input person detection boxes. To generate
semantic patches, we incorporate diffusion model and STN
[21] to construct the SSDM. The SSDM generates P patch
features per person image. The momentum coefficient ε in
the memory bank is set to 0.2. For each mini-batch of size
72, we select two different cameras for each label, with each
camera containing two samples. Each input image is resized to
384 × 128, and data augmentation techniques such as random
horizontal flipping, cropping, and erasing [53] are applied.
Training is conducted using backpropagation with the Adam
optimizer, and the learning rate starts at 3.5 × 10−4 , decaying
by a factor of 10 every 20 epochs. The combined loss Lces ,
Ltri , Lmsc , and L sdc are used during training.
C. Parameter Analysis
1) The Weight λmsc and λ sdc : The weights λmsc and λ sdc
in Eq. (1) control the relative importance of each loss term.
By adjusting their values, we evaluate the Rank-1 accuracy on
the Market-1501 dataset, as shown in Fig. 7. For Lmsc , smaller
values limit the influence of camera constraints, while larger
values reduce the impact of other loss terms. The optimal
performance is achieved when λmsc = 0.3. Similarly, for L sdc ,

Fig. 9. Evaluation of T in Market-1501 and MSMT17 datasets.

the best results occur when λ sdc = 0.6, ensuring the diffusion
model’s controllability and enhancing the adaptability of the
generated semantic patches for the Re-ID task.
2) The Number of Patches P: The number of patches P
in Eq. (24) defines the number of patch features extracted by
the SSDM from each feature map. We explore the impact of
varying P within the range [2, 6]. As shown in Fig. 8, both
Rank-1 and mAP accuracy improve with increasing P. When P
is too small, the model struggles to capture fine-grained details,
producing ineffective patch features for global clustering. We
find that the optimal value for P is 3. However, increasing
P beyond 3 can degrade accuracy by disrupting the semantic
structure and introducing noise. A larger number of patches
also increases the number of spatial transformer parameters,
raising the training cost of the diffusion model.
3) The Diffusion Step T : The diffusion step T in Eq. (20)
is crucial for achieving a balance between training efficiency
and generation quality. As shown in Fig. 9, a small T limits
the model to learn the distribution of spatial transformer
parameters, resulting in semantically poor patch generation.
Our experiments show that T = 1000 offers an optimal tradeoff. Further increasing T leads to diminishing returns and
harms performance due to over-smoothing. Also, a larger
T incurs considerable computational overhead during both
training and inference, reducing its applicability.
D. Ablation Study
We perform ablation experiments to investigate each component of our approach on the Market-1501 and MSMT17
dataset. These datasets provide a balanced combination of
scale, complexity, and diversity, making them ideal for assessing the effectiveness and robustness of our method. We use the
patch-based unsupervised Re-ID framework (Sec.III-A) using
Lces and Ltri as baseline. The results, shown in Table. I and
Table. II, demonstrate the effectiveness of each components.
Our method achieves significant performance improvements,
boosting Rank-1/mAP accuracy by 6.3%/13.0% on Market1501 and 24.8%/20.5% on MSMT17 compared to the baseline.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:33 UTC from IEEE Xplore. Restrictions apply.


codex
第三、四篇也读到了方法主体。`CMLR` 的 cluster-level 细节在工具输出里被折叠了一段，我会单独补读那几页，避免把“相机感知”理解成泛泛的 camera proxy。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 4 -l 6 'Unsupervised person re-identification via camera-aware multi-level label refinement.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 195 (2026) 108292

N. Tang et al.

Fig. 2. The overall framework of the proposed method. In the clustering phase, DBSCAN is applied to group global features and assign initial hard pseudo-labels. In the
training phase, the DIIR module reﬁnes these labels for cluster-level contrastive learning by leveraging camera-aware neighbors: inter-camera neighbors for global
features and intra-camera neighbors for local features. Meanwhile, the AGMR module performs instance-level contrastive learning by constructing aﬃnity-based
labels through informative node selection, further mitigating camera-style discrepancies and enhancing the reliability of the clustering results.

of the feature vector. Speciﬁcally, the cluster centroids in the memory
bank are initialized using the corresponding average feature vectors and
pseudo-labels, as follows:
1 ∑
𝑀[𝑖] =
𝑢𝑗
(1)
|𝐶̃𝑖 | 𝑢 ∈𝐶̃
𝑗

This cluster-level loss eﬀectively reduces the distance between samples and the centroids of their respective clusters while pushing them
away from other clusters. Following the cluster-level contrastive learning paradigm outlined in Eq. (3), separate losses are computed for global
and local features, yielding three distinct cluster-level losses: 𝐿𝑔 , 𝐿𝑢𝑝 ,
and 𝐿𝑙𝑜𝑤 . These losses are then weighted to obtain the total cluster-level
loss 𝐿𝑐𝑙𝑢𝑠𝑡𝑒𝑟 , as shown in Eq. (4):

𝑖

where 𝐶̃𝑖 denotes the cluster with pseudo-label 𝑖 and | ⋅ | represents the
counting of elements. The cluster centroids stored in 𝑀 are updated via
a moving average strategy:
𝑀[𝑦𝑖 ] = 𝑚𝑀[𝑦𝑖 ] + (1 − 𝑚)𝑢𝑖

𝐿𝑐𝑙𝑢𝑠𝑡𝑒𝑟 = (1 − 𝜆1 )𝐿𝑔 + 𝜆1 (𝐿𝑢𝑝 + 𝐿𝑙𝑜𝑤 )

(4)

where 𝜆1 is the weight parameter that balances the global loss and the
local loss.

(2)

where 𝑚 denotes the momentum update factor.
To complement the limited representation of global features, memory banks for local features are also set to facilitate the discovery of more
detailed information during the learning process. The three branches independently compute cluster centroids based on their respective feature
vectors according to Eq. (1) while sharing the same set of pseudo-labels
𝑌 . These centroids are then updated following Eq. (2).
The cluster-level contrastive loss formulation is presented as follows:

3.2. Dual-level intra-inter reﬁnement

(3)

In each training epoch, hard pseudo-labels are generated from the
initial clustering, oﬀering only a coarse approximation of the feature
distribution. This often results in features being assigned to incorrect
clusters, with label noise compounding over time and hindering model
optimization. To address this, we propose the dual-level intra-inter reﬁnement (DIIR) method, as illustrated in Fig. 3. DIIR reﬁnes pseudolabels from both global and local perspectives, enhancing label quality
and improving clustering accuracy.

where |𝐵| represents the number of samples in each mini-batch, 𝑦̂𝑖 represents the one-hot encoded form of the pseudo-label for sample 𝑥𝑖 , ∗
denotes three branches: global (g), upper local (up), and lower local
(low) features, 𝑀∗ represents the cluster-level memory bank that stores
the centroids of features from three branches, and 𝜏 is the temperature
parameter.

3.2.1. Reﬁnement for labels of global features
Due to camera style variation, features captured by the same camera tend to cluster tightly, while those from diﬀerent cameras are more
dispersed. As a result, samples belonging to the same identity may be
assigned to diﬀerent clusters simply because they were captured by different cameras. To alleviate this issue, we aim to pull closer the centroids of inter-camera clusters that are likely to correspond to the same

𝐿∗ = −

|𝐵| 𝑁 ′
⎛ exp ((𝑀 [𝑙]𝑇 ⋅ 𝑢 )∕𝜏 ) ⎞
∗
∗,𝑖
1 ∑∑
⎟
𝑦̂𝑖 [𝑙] ⋅ log ⎜ ∑ ′
⎜ 𝑁 exp ((𝑀 [𝑝]𝑇 ⋅ 𝑢 )∕𝜏 ) ⎟
|𝐵| 𝑖=1 𝑙=1
∗
∗,𝑖
⎝ 𝑝=1
⎠

4

Neural Networks 195 (2026) 108292

N. Tang et al.

Fig. 3. The workﬂow of DIIR. For sample 𝑥𝑖 , we separately search the inter-camera 𝑘-nearest neighbors of global features and the intra-camera 𝑘-nearest neighbors
of local features. Their cluster assignments are then recorded in the current epoch. (1) Global reﬁnement: We compute the top-k probabilities for diﬀerent clusters.
For example, samples from the yellow cluster appear once in the top-5 of the neighbor list, while samples from the green cluster appear once in the top-2. Therefore,
the top-k probabilities for the yellow and green clusters are 1/5 and 1/2, respectively. (2) Local reﬁnement: We take the intersection of intra-camera neighboring
samples for local upper and lower features to obtain more reliable neighbors. Then, we calculate the top-k probabilities of their assigned clusters and reﬁne the local
labels.

identity, while pushing away unrelated ones. Speciﬁcally, for a given
sample 𝑥𝑖 , we compute the Euclidean distances between its global representation and those of samples from other cameras. We then sort these
distances in ascending order to obtain the inter-camera 𝑘-nearest neigh𝑔
bor list 𝑆𝑖𝑛𝑡𝑒𝑟
(𝑖, 𝑘), which stores the indices of reliable neighbors from
diﬀerent cameras, as deﬁned in Eq. (5):
𝑔
𝑆𝑖𝑛𝑡𝑒𝑟
(𝑖, 𝑘) = 𝑎𝑟𝑔𝑠𝑜𝑟𝑡(‖𝑢𝑔,𝑖 − 𝑢𝑔,𝑗 ‖2 )[∶ 𝑘], 𝑐𝑖 ≠ 𝑐𝑗

adaptively pulled toward the centroids of its neighbors’ clusters. Although the original cluster remains the primary inﬂuence, its role in
determining the pseudo-label is no longer absolute. As the model’s feature extraction capability improves, misclassiﬁed samples in the early
stage are more likely to be reassigned to their true clusters. Compared
to directly using initial hard pseudo-labels for supervised learning, this
reﬁned labeling scheme signiﬁcantly reduces the accumulation of label
noise.

(5)

where 𝑢𝑔,𝑖 and 𝑢𝑔,𝑗 denote the global features of samples 𝑖 and 𝑗, respectively, 𝑐𝑖 and 𝑐𝑗 are their corresponding camera labels, 𝑗 ranges over
all samples from cameras diﬀerent from that of sample 𝑥𝑖 , ‖ ⋅ ‖2 represents the L2 norm, and 𝑎𝑟𝑔𝑠𝑜𝑟𝑡(⋅)[∶ 𝑘] represents the indices of the top-𝑘
smallest elements in ascending order.
Based on the distribution of the pseudo-labels of these neighbors,
we calculate the probability that sample 𝑥𝑖 belongs to alternative clusters, termed as the top-k probability. The computation is formulated as
follows:
𝑃𝑖𝑛𝑡𝑒𝑟 (𝑖, 𝑙) =

𝑔
|𝑆𝑖𝑛𝑡𝑒𝑟
(𝑙|𝑖, 𝑘)|
𝑔
𝐼𝑛𝑑𝑒𝑥(𝑆𝑖𝑛𝑡𝑒𝑟
(𝑙|𝑖, 𝑘))𝑚𝑎𝑥

3.2.2. Reﬁnement for labels of local features
In order to address the issue that global features alone cannot effectively distinguish intra-camera heterogeneous samples, local features
are utilized to capture more detailed and discriminative information.
In the previous method Lan et al. (2023), global and local contrastive
learning share the same global hard pseudo-labels. However, this strategy suﬀers from several limitations. First, although global and local features originate from the same sample, global pseudo-labels cannot fully
reﬂect the underlying distribution of local features, resulting in suboptimal supervision for local-level representation learning. Second, the
inherent noise in global clustering results tends to propagate throughout the training process, creating compounded error accumulation that
degrades local feature discriminability.
To address these limitations, we reﬁne the hard labels into two sets
𝑙𝑜𝑤
of local soft labels, 𝑦̂𝑢𝑝
𝑖 and 𝑦̂𝑖 , which supervise contrastive learning
on the corresponding upper and lower local features. Due to the incomplete semantic representation of local features, they tend to be less stable than global features. This instability makes inter-camera neighbor
search prone to introducing considerable noise. To ensure the discriminability of local information while minimizing the inﬂuence of camera
style, we restrict the 𝑘-nearest neighbor search for upper and lower lo𝑢𝑝
𝑙𝑜𝑤 (𝑖, 𝑘),
cal features within the same camera, yielding 𝑆𝑖𝑛𝑡𝑟𝑎
(𝑖, 𝑘) and 𝑆𝑖𝑛𝑡𝑟𝑎
𝑝𝑎𝑟𝑡
respectively. Their intersection, 𝑆𝑖𝑛𝑡𝑟𝑎 (𝑖, 𝑘) serves as a ranked list of reliable intra-camera neighbors for sample 𝑥𝑖 , ordered by ascending feature distance. Following the same strategy as in the global reﬁnement,
we compute the top-k probabilities over the local reliable neighbors to

(6)

𝑔
𝑔
where 𝑆𝑖𝑛𝑡𝑒𝑟
(𝑙|𝑖, 𝑘) denotes the samples with label 𝑙 in 𝑆𝑖𝑛𝑡𝑒𝑟
(𝑖, 𝑘),
𝐼𝑛𝑑𝑒𝑥(⋅)𝑚𝑎𝑥 represents the maximum index value of all elements inside, and 𝑃𝑖𝑛𝑡𝑒𝑟 (𝑖, 𝑙) represents the probability of the label 𝑙 occurring
𝑔
in 𝑆𝑖𝑛𝑡𝑒𝑟
(𝑖, 𝑘), up to its last occurrence. The probability 𝑃𝑖𝑛𝑡𝑒𝑟 is determined by the distance rank of neighboring samples and the occurrence
frequency of their respective clusters, aiming to assign higher weights
to closer clusters that appear more frequently.
The reﬁned labels 𝑦̂𝑖 𝑔 are then constructed as follows:

⎧1 − 𝛼,
⎪
𝑃
(𝑖, 𝑙)
𝑦̂𝑔𝑖 [𝑙] = ⎨
𝛼 × ∑ 𝑖𝑛𝑡𝑒𝑟
,
′
⎪
′ ≠𝑦 𝑃𝑖𝑛𝑡𝑒𝑟 (𝑖, 𝑙 )
𝑙
𝑖
⎩

𝑙 = 𝑦𝑖
𝑙 ≠ 𝑦𝑖

(7)

where 𝛼 ∈ [0, 1] represents the weight for label reﬁnement. By incorporating the clustering information of reliable neighbors, sample 𝑥𝑖 is
5

Neural Networks 195 (2026) 108292

N. Tang et al.

construct reﬁned cluster-level labels for the local features, as deﬁned in
Eq. (8):
𝑝𝑎𝑟𝑡
𝑢𝑝
𝑙𝑜𝑤
𝑆𝑖𝑛𝑡𝑟𝑎
(𝑖, 𝑘) = 𝑆𝑖𝑛𝑡𝑟𝑎
(𝑖, 𝑘) ∩ 𝑆𝑖𝑛𝑡𝑟𝑎
(𝑖, 𝑘)
| 𝑝𝑎𝑟𝑡
|
|𝑆𝑖𝑛𝑡𝑟𝑎 (𝑙|𝑖, 𝑘)|
|
|
𝑃𝑖𝑛𝑡𝑟𝑎 (𝑖, 𝑙) =
( 𝑝𝑎𝑟𝑡
)
𝐼𝑛𝑑𝑒𝑥 𝑆𝑖𝑛𝑡𝑟𝑎
(𝑙|𝑖, 𝑘) 𝑚𝑎𝑥

⎧1 − 𝛼,
⎪
𝑃
(𝑖, 𝑙)
𝑢𝑝
𝑙𝑜𝑤
𝑦̂𝑖 [𝑙] = 𝑦̂𝑖 [𝑙] = ⎨𝛼 × ∑ 𝑖𝑛𝑡𝑟𝑎
,
𝑃𝑖𝑛𝑡𝑟𝑎 (𝑖, 𝑙′ )
⎪
⎩
𝑙′ ≠𝑦𝑖

(8)
𝑙 = 𝑦𝑖
𝑙 ≠ 𝑦𝑖

where 𝑃𝑖𝑛𝑡𝑟𝑎 (𝑖, 𝐿) is the top-k probability of the pseudo-label 𝑙 in
𝑝𝑎𝑟𝑡
𝑆𝑖𝑛𝑡𝑟𝑎
(𝑖, 𝑘).
We replace the original hard labels 𝑦̂𝑖 in Eq. (3) with the reﬁned
labels 𝑦̂∗𝑖 , where yielding the reﬁned cluster-level losses 𝐿′∗ as follows:
(
)
|𝐵| 𝑁 ′
exp 𝑀∗ [𝑙]𝑇 ⋅ 𝑢∗,𝑖 ∕𝜏
1 ∑∑ ∗
𝐿′∗ = −
𝑦̂𝑖 [𝑙] ⋅ log ∑ ′
(9)
(
)
𝑁
|𝐵| 𝑖=1 𝐿=1
𝑇
𝑝=1 exp 𝑀∗ [𝑝] ⋅ 𝑢∗,𝑖 ∕𝜏

Fig. 4. Identifying informative node collections. Candidate nodes marked with
a red ✓ are reliable informative nodes, while those marked with a blue × are
ﬁltered-out candidates.

The proposed total DIIR loss 𝐿DIIR is formulated by integrating the reﬁned global and local cluster-level contrastive losses. Speciﬁcally, 𝐿′𝑔 ,
𝐿′𝑢𝑝 , and 𝐿′𝑙𝑜𝑤 are substituted into Eq. (4) in place of 𝐿𝑔 , 𝐿𝑢𝑝 , and 𝐿𝑙𝑜𝑤 ,
respectively:
𝐿𝐷𝐼𝐼𝑅 = (1 − 𝜆1 )𝐿′𝑔 + 𝜆1 (𝐿′𝑢𝑝 + 𝐿′𝑙𝑜𝑤 )

the same camera - are selected as candidate informative nodes, as deﬁned in Eq. (13):
{
(
)}
|
𝐼𝐹𝑖 = 𝑢𝑔,𝑗 ∈ 𝑈𝑔 | 𝑐𝑗 ≠ 𝑐𝑖 , 𝑢𝑔,𝑗 ∈ TopK 𝑘 ⟨𝑢𝑔,𝑖 , 𝑢𝑔,𝑗 ⟩
(13)
|
where ⟨⋅, ⋅⟩ denotes the cosine similarity, and TopK 𝑘 (⟨𝑢𝑔,𝑖 , 𝑢𝑔,𝑗 ⟩) returns
the set of top-𝑘 feature vectors 𝑢𝑔,𝑗 ∈ 𝑈𝑔 that have the highest similarity
scores with the query feature 𝑢𝑔,𝑖 , excluding those from the same camera
(𝑐𝑗 ≠ 𝑐𝑖 ). The resulting set 𝐼𝐹𝑖 contains the candidate informative nodes
for the query.
For these candidates, we further compute their cosine similarity with
all other samples and extract their own top-𝑘 inter-camera neighbors
as sub-candidates. For each candidate 𝑢𝑔,𝑗 ∈ 𝐼𝐹𝑖 , we check whether the
query 𝑢𝑔,𝑖 appears in its sub-candidate set 𝐼𝐹𝑖,𝑗 . If this mutual similarity
is conﬁrmed, the candidate 𝑢𝑔,𝑗 is promoted to a reliable informative
node. This process is formalized in Eq. (14):
{
(
)}
|
𝐼𝐹𝑖,𝑗 = 𝑢𝑔,𝑛 ∈ 𝑈𝑔 | 𝑐𝑛 ≠ 𝑐𝑗 , 𝑢𝑔,𝑛 ∈ TopK 𝑘 ⟨𝑢𝑔,𝑗 , 𝑢𝑔,𝑛 ⟩
|
(14)
̃𝑖 = {𝑢𝑔,𝑗 ∈ 𝐼𝐹𝑖 ∣ 𝑢𝑔,𝑖 ∈ 𝐼𝐹𝑖,𝑗 }
𝐼𝐹

(10)

By reﬁning both global and local labels, our method eﬀectively mitigates label noise and contributes to more stable model optimization.
3.3. Aﬃnity-Guided mutual reﬁnement
Cluster-level contrastive learning focuses solely on the relationships
between instance features and their corresponding cluster centroids,
while overlooking ﬁne-grained relationships between individual sample
pairs. To address this limitation and improve representational granularity, we construct an additional instance-level memory bank 𝐼 ∈ ℝ𝑑×𝑁 on
top of the baseline. This memory bank stores global features and adopts
the same update strategy as 𝑀, as shown in Eq. (11):
𝐼[𝑖] = 𝑢𝑔,𝑖
𝐼[𝑖] = 𝑚𝐼[𝑖] + (1 − 𝑚)𝑢𝑔,𝑖

(11)

To facilitate instance-level contrastive learning, each sample 𝑥𝑖 is
initially assigned a one-hot label 𝑛̂ 𝑖 ∈ ℝ𝑁 , treating itself as the only positive. The instance-level contrastive loss is formulated as follows:
(
)
|𝐵| 𝑁
exp 𝐼[𝑗]𝑇 ⋅ 𝑢𝑔,𝑖 ∕𝜏
1 ∑∑
𝐿𝑖𝑛𝑠 = −
𝑛̂ 𝑖 [𝑗] ⋅ log ∑𝑁
(12)
(
)
𝑇
|𝐵| 𝑖=1 𝑗=1
𝑞=1 exp 𝐼[𝑞] ⋅ 𝑢𝑔,𝑖 ∕𝜏

where 𝐼𝐹𝑖,𝑗 corresponds to the sub-candidate nodes for 𝑢𝑔,𝑗 , 𝑢𝑔,𝑗 ∈ 𝐼𝐹𝑖 ,
̃𝑖 indicates the reliable informative nodes for 𝑢𝑔,𝑖 . These collecand 𝐼𝐹
tions of informative nodes, which hold the most relevant information
for sample 𝑥𝑖 , are then used to form positive pairs.

where 𝜏 is the temperature parameter. Although Eq. (12) enhances the
discriminability of individual samples, it fails to capture the underlying
relationships among them, leading to overly separated sample representations and the neglect of their semantic aﬃnities.

3.3.2. Aﬃnity-guided reﬁnement
To mitigate he inﬂuence of camera bias and enhance intra-class
compactness, we propose an aﬃnity-guided mutual reﬁnement (AGMR)
mechanism, which leverages reliable informative nodes to perform ﬁnegrained reﬁnement of the initial instance-level labels, as illustrated in

3.3.1. Informative node collections
To address the limitation of instance-level contrastive learning in
capturing semantic relationships between samples, we propose a mechanism for constructing more meaningful positive pairs. Instead of treating
each sample as its own positive, we identify a set of informative nodesreliable samples that share strong semantic similarity and discriminative
relevance with the current query sample. Paired with the query, these
informative nodes form semantically consistent positive pairs, replacing
the trivial self-positive assumption. This design encourages feature compactness among similar instances and helps to mitigate camera-induced
biases. The identiﬁcation process is illustrated in Fig. 4.
Concretely, given the feature representation of the query sample 𝑢𝑔,𝑗 ,
we compute its cosine similarity with all other samples in the instance
memory 𝐼 within the current batch, followed by normalization. The
Top-𝑘 samples with the highest similarity scores - excluding those from

Fig. 5. The workﬂow of AGMR. In each batch, we identify informative nodes
for the current sample. Sample pairs are then constructed, and their similarities are computed. After softmax processing, aﬃnity labels between samples are
obtained. (The shade of gray in the labels represents the magnitude of aﬃnity).
6


 succeeded in 0ms:
Unsupervised Visible-Infrared Person ReID via
Modality-Camera Balance Label Refinement
JIAKAI HE, YIMING YANG, HAIFENG HU, and RUIXING WU, School of Electronics and
Information Technology, Sun Yat-Sen University, Guangzhou, China
Unsupervised Learning Visible-Infrared Person Re-Identification (USL-VI-ReID) focuses on developing a
cross-modality retrieval model without the need for labels, minimizing the dependence on costly manual
annotation across modalities. Recently, various approaches focus on reducing the cross-modality discrepancies.
However, they ignore that USL-VI-ReID is also a task of solving discrepancies while exploring fine-grained
information in hierarchical domains. In this article, we propose a hierarchical Modality-Camera Balance Label
Refinement (MCBL) framework to balance the contributions of each camera-modality. Meanwhile, we explore
the fine-grained features and refine the noise labels at each training stages. Specifically, our MCBL naturally
combines Modality-Camera Balanced Label Mining (MBLM), Unreliable Pseudo-Label Re-align (UPR), and
Hybrid Modality-Camera Contrastive Learning (HMCCL) into a unified framework, which balances the
association information for each hierarchical domain through refining noise labels. Technically, MBLM filters
cluster-level noise samples utilizing a modality-camera balance strategy, thereby ensuring that reliable samples
are stored in memory for effective contrast learning. UPR refines the noise labels through the re-alignment
methods at the instance level, thus improving the accuracy of labels and further enhancing the model’s
generalization ability. Moreover, the key of HMCCL is optimizing the distribution at both the instance and
cluster levels, which forces the sample to be close to its cluster proxy while being far from others in a real-time
memory update phase. Extensive experiments have shown that our MCBL addresses the current limitations of
camera discrepancy and achieves competitive performance.
CCS Concepts: • Computing methodologies → Computer vision; Object identification; Neural networks;
Additional Key Words and Phrases: Person re-identification (ReID), cross-modality, unsupervised learning,
clustering
ACM Reference format:
Jiakai He, Yiming Yang, Haifeng Hu, and Ruixing Wu. 2025. Unsupervised Visible-Infrared Person ReID
via Modality-Camera Balance Label Refinement. ACM Trans. Multimedia Comput. Commun. Appl. 21, 12,
Article 357 (November 2025), 24 pages.
https://doi.org/10.1145/3772086

Authors’ Contact Information: Jiakai He, School of Electronics and Information Technology, Sun Yat-Sen University,
Guangzhou, China; e-mail: hejk26@mail2.sysu.edu.cn; Yiming Yang, School of Electronics and Information Technology, Sun
Yat-Sen University, Guangzhou, China; e-mail: yangym53@mail2.sysu.edu.cn; Haifeng Hu (corresponding author), School
of Electronics and Information Technology, Sun Yat-Sen University, Guangzhou, China; e-mail: huhaif@mail.sysu.edu.cn;
Ruixing Wu, School of Electronics and Information Technology, Sun Yat-Sen University, Guangzhou, China; e-mail:
wurx29@mail2.sysu.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/11-ART357
https://doi.org/10.1145/3772086
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.

357:2
1

J. He et al.

Introduction

Person Re-Identification (ReID), which aims to match the same person across multiple different
views, has been a topic of significant interest in the field of computer vision for an extended
period. Due to its importance in multimedia data retrieval and criminal investigation [8, 56], this
technology has been under development for the past decade or so. Initially, the matching task
of person ReID is based on the images captured by RGB cameras. However, these methods are
sensitive to illumination, especially in low-light conditions, resulting in poor performance. This is
mainly attributable to the limitations of images captured by RGB cameras under dark conditions.
Consequently, Visible-Infrared ReID (VI-ReID) is conceived and put into practice for 24-hour
surveillance. It synchronizes the infrared image acquired during low-light situations and the visible
image taken in favorable lighting conditions.
Several existing VI-ReID methods learn the modality invariant representation through generation
and subspace mapping techniques and achieve remarkable performance [7, 6, 29, 43]. Despite this,
their success relies on the massive amount of manually labeled data between visible and infrared
modality, which is expensive and time-consuming, making it more difficult to scale and deploy
VI-ReID models. Consequently, Unsupervised Learning VI-ReID (USL-VI-ReID) has been
suggested to mitigate this dependence on extensive annotations and has gained growing attention
owing to its promising prospects.
The USL-VI-ReID approach eliminates the requirement of numerous manual identity labels
by enabling cross-modality association. As a density-based clustering algorithm, DBSCAN [12]
has proven to be very successful in most existing methods for pseudo-labeling unlabeled data.
However, it still suffers from many challenges and difficulties on account of the significant crossmodality and cross-camera differences present within the dataset. The challenge of clustering
cross-modal pedestrian data is illustrated in Figure 1. Discrepancies between different cameras
and modalities can result in excessive identity segmentation and the inability to assign accurate
labels [16, 22, 52]. Fine-tuning the neural network with these error labels can even lead to intracluster distances beyond inter-cluster distances. In this case, the pseudo-labels are directly obtained
through DBSCAN, resulting in a high proportion of noise labels, which adversely affects model
exploring the fine-grained features. More importantly, each camera carries unique information
leading to inconsistent cluster numbers between two modalities and the introduction of noise labels.
Therefore, it is considerable to balancing the alignment between hierarchical features and labels.
The current methods [26, 32, 45, 51] emphasize eliminating differences between modalities. For
example, graph matching [45] and optimal transmission methods [41] are widely used to strengthen
the associations in cross-modal clustering. Nevertheless, methods mentioned above are hindered
by the problem of extensive identity division, which could affect the precision of the association.
To address the above-mentioned issues, we put forward a Modality-Camera Balance Label
Refinement (MCBL) framework to mine the modality-camera balance label while exploring the
fine-grained information of hierarchical domains. The flowchart of MCBL is shown in Figure 2.
Specifically, MCBL adopts a comprehensive bottom-up clustering and optimization framework
through intra-camera, inter-camera, and inter-modal stages, which combines Modality-Camera
Balanced Label Mining (MBLM) strategy, Unreliable Pseudo-Label Re-align (UPR) strategy,
and Hybrid Modality-Camera Contrastive Learning (HMCCL) into the framework. MBLM
integrates inter-modal and inter-camera similarity and employs the balance of associations within
modality-camera to assess the reliability of pseudo-labels. By investigating the distribution of
the pseudo-label across different cameras and modalities, we eliminate unreliable pseudo-labels
and identify accurate pseudo-labels to enhance the recognition capability of the model. However,
excluding unreliable samples leads to reduction of sample quantity in the initial training stage.
Therefore, to further utilize the fine-grained features at the intra-camera training stage, we propose
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.

Unsupervised Visible-Infrared Person ReID via MCBL

357:3

Fig. 1. Illustration of noise pseudo-labels generated in clustering due to cross-modal and cross-camera
discrepancies with each modality with two cameras as an example. Circles and squares represent the samples
with same identity after clustering, respectively. Different colors indicate whether the samples in the cluster
are reliable. The variations of inter-cameras and inter-modality collectively lead to the introduction of noisy
samples in the clustering process.

the UPR strategy to re-alignm the unreliable samples. Specifically, in the initial phase of the model,
we address the rejected samples through a re-alignment method and enhance the accuracy by
refining noisy labels. By seamlessly integrating MBLM and UPR, we can enhance the model’s ability
to differentiate between noise labels and optimize them accordingly. Importantly, this approach
facilitates more efficient contrastive learning within the intra-camera memory. Inspired by [58],
HMCCL optimizes the distribution at the instance level and the cluster level in real time. Finally, with
the fine-grained bottom-up framework, MCBL learns a modality-camera invariant representation,
achieving the balance in different modalities and cameras.
In conclusion, the key achievements of this article are outlined below:
— We propose a comprehensive bottom-up MCBL framework for USL-VI-ReID, which is designed
to learn hierarchical fine-grained features by achieving a balance between inter-modality and
inter-cameras, thus improving the robustness and generalization capabilities of the model.
— The MBLM combines the similarity information between inter-cameras and inter-modality to
eliminate the unreliable labels in the cluster, so as to achieve the modal-camera balance.
— The UPR leverage the rejected samples effectively and re-aligns the noise labels at the instance
level, which improves the label accuracy in the initial stage of the model.
— HMCCL conducts cluster-wise and instance-wise contrastive learning in different stages,
specifically in the intra-camera, intra-modality, and inter-modality stages. The cluster-wise
representations are modified through randomly drawn cross-modal data to achieve a more
condensed cross-modal pattern.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.

357:4

J. He et al.

Fig. 2. The flowchart of MCBL framework. The MCBL model is composed of three training stages, i.e., intracamera, inter-camera, and inter-modality training stages. In each stage, we adopted the key modules: MBLM,
UPR, and HMCCL. MBLM evaluates each sample uniformly for intra-modal and intra-camera similarity
information, thereby eliminating unreliable labels. UPR further leverages these unreliable instances to
compensate for the model’s weak discrimination ability in early stages, ultimately enhancing its generalization
capability. The refine pseudo-labels generated by MBLM and UPR are employed to initiate the instance-wise
and cluster-wise memories. HMCCL conducts contrastive learning at each training stage to learn the modality
and camera invariant representations. In the testing phase, our framework employs only the backbone for
assessment.

— Extensive experiments on the SYSU-MM01 and RegDB datasets demonstrate the efficacy and
competitive superiority of MCBL framework.
2

Related Work

In the following part, we will provide an overview of the following areas: Supervised VI Person
ReID (SVI-ReID), Unsupervised Single-Modality Person ReID (USL-ReID), and USL-VI-ReID.
2.1

SVI-ReID

SVI-ReID has been the subject of widespread concern due to its application of infrared for person
retrieval in low-light conditions at night. Among the existing methods, SVI-ReID can be generally
categorized into two groups: feature-level and image-level modal alignment methods.
The key to the feature-level modality alignment methods is the mapping of cross-modal features
to a shared subspace. Most methods also incorporate metric learning techniques to constrain
the distribution. Achieving that goal, Zhang et al. [63] proposed a diverse embedding expansion
network, which can effectively generate diverse embeddings and reduce the modality discrepancy
through triple-level constraints. Ren and Zhang [35] not only embed images from various modalities
into an integrated space but also seek specific information within these shared features, thereby
reducing modal variations while preserving identity awareness. Furthermore, in pursuit of digging
more diverse modality-shared cues, Feng et al. [13] remove body-shape-associated semantic notions
from the learned features and prompt the ReID model to retrieve further and diverse modality-shared
characteristics for identification.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.

Unsupervised Visible-Infrared Person ReID via MCBL

357:5

On the other hand, given the modal differences between pixels, the modal alignment technique
at the image-level is usually solved using generative network or image enhancement technologies.
Ye et al. [54] utilized a Channel Exchangeable Augmentation (CA) to bridge the gap between
the visible and infrared modalities by incorporating randomly generated color-irrelevant images.
Kim et al. [23] tailored a part-based data augmentation technique by combining body-regions
across different modalities to enhance the performance of the model. Liu et al. [27] developed the
unidirectional metrics from image to proxy, which can establish effective cross-modal correlation.
In the aspect of generated network, various methods [6, 9, 40, 43, 65] demonstrated the powerful
transitions capabilities of Generative Adversarial Networks (GANs), which can simultaneously
bridge the gap between different modalities and preserve identity details. The modal alignment
methods, both at the feature-level and image-level, inevitably introduce noise. Nevertheless, the
distribution of these noises between modalities and cameras is uneven, which greatly hinders the
real-time performance and scalability of ReID.
2.2

USL-ReID

USL-ReID mainly focus on assigning pseudo-label to visible modality, in order to addressing the
problem of time-consuming and expensive labeling. Among the various methods of pseudo-label
assignment, clustering algorithms are the most widely used. Moreover, memory-based methods for
optimizing associations between instances and clusters have recently emerged as a predominant
approach in the field. Ge et al. [16] contributed the memory-based contrastive learning framework
SPCL to distinguish the distribution between clusters and generate supervisory label, which plays a
vital role in the subsequent research. In the context of pseudo-label generation through clustering,
Zhang et al. [61] employed the similarity of labels between consecutive training iterations to
achieve label refinement. In order to address the issue of delayed updating of pseudo-labels and the
criterion for simple clustering methods, Zheng et al. [66] proposed a novel label update method
aimed at enhancing the accuracy of ReID.
Recently, a series of papers explores the other difficulties of ReID, while maintaining the contrastive learning framework. Dai et al. [10] considered cluster consistency and optimized unsupervised clustering to enhance the performance of model. Further considered the identity centroids,
both of them [22, 42] incorporate contrast learning into the camera. By adding cross-camera proxy
contrast loss, Hao et al. [2] alleviated the negative influence between the cameras. However, the
discrepancy between cross-cameras still results in incorrect cluster associations and introduces
noise labels, which significantly impacts memory-based contrast learning.
2.3

USL-VI-ReID

USL-VI-ReID is a project dedicated to the field of cross-modality person retrieval, with the goal
of matching daytime visible and nighttime infrared images by establishing stable cross-modality
associations. Yang et al. [51] designed a count priority selected module in ADCA to associate positive
cross-modality identity. But employing this top-k strategy [50, 51] to discard unreliable instances
often causes feature shifts. In addressing this issue, our UPR strategy eliminates label selection
instead of re-aligning unreliable samples at the instance level, thereby preserving more useful
data and improving label reliability. GUR [49] further introduced a bottom-up domain learning
framework based on ADCA, which considers the consistency at cluster level to perform label mining
strategy. Unlike the label strategy in GUR, we introduce MBLM module that leverages a instancelevel information, fine-grained modality–camera perspective to refine pseudo-labels, achieving
more reliable and balanced label mining. Furthermore, a series of works [4, 5, 21, 41, 45] have also
effectively utilized graph matching and optimal transport methods to enhance the precision of
cross-modal association. Shi et al. [36] utilised the generating multiple prototypes and designed
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.

357:6

J. He et al.

a learning strategy with progressive attribute to shift the model’s attention, thereby preventing
clustering deterioration. Taking advantage of the visual text representation capability of CLIP, Chen
et al. [3] incorporated the resulting text representations into the subsequent unsupervised training
process. Pang et al. [32] tried to isolate three channels from visible images and subsequently merge
them with infrared images for clustering, whereas adopted the IoU metric method to achieve the
label balance refinement in RGB three-channel. However, these aforementioned methods do not
take into account the discrepancies between camera domains, which can make a significant impact
in the cluster stage. Particularly in large datasets, camera discrepancy can exacerbate the effects of
noise labels in cross-modal associations due to the view and illumination of each camera. Therefore,
in the pseudo-label generation stage, we optimize the identity noise problem by balancing the
influence between cameras within modality and the clusters.
3

The Proposed Model

In this section, we detail the proposed MCBL framework, which accomplishes pseudo-label refinement based on modality-camera balance. Our framework is illustrated in Figure 2.
3.1

Problem Modeling

In USL-VI-ReID, we employ an effective network known as AGW [56] as the backbone, f . For the
purpose of unsupervised training, the identity labels in the given datasets are disregarded. Formally,
the visible-infrared pedestrian dataset can be expressed in terms of D = {V, R}, in which the
𝑁𝑉
𝑁𝑅
visible and infrared images can be defined as V = {𝑥𝑖𝑣 }𝑖=1
and R = {𝑥𝑖𝑟 }𝑖=1
. 𝑁𝑉 and 𝑁𝑅 indicate the
image numbers from visible and infrared modalities. Similar to [54], we implement CA to bridge the
discrepancy between the visible and infrared stream. Therefore, the visible stream has an additional
𝑁𝐶
CA branch, in which the data can be expressed as C = {𝑥𝑖𝑐 }𝑖=1
. Note that we only apply the CA
technique to enhance the visible images during the training process, ensuring that the sampled
image number in mini-batches is consistent across both modalities, i.e., 𝑁𝑉 = 𝑁𝑅 = 𝑁𝐶 = 𝑃 × 𝐾.
Here, 𝑃 is the number of different categories of clusters, and 𝐾 is the number of images per cluster
in single modality.
Specifically, our framework employs the bottom-up learning strategy that consists of three
training stages: intra-camera training, inter-camera training, and inter-modal training. At each stage,
we extract features from all the training samples and then adopt DBSCAN [12] clustering algorithm
to allocate pseudo-labels. However, in contrast to GUR [49], we are of the opinion that alternating
ADCA [51] in each camera domain has a detrimental impact on the generalization performance.
Consequently, the intra-camera stage aims to develop the initial discriminability of the model via
independent memory for each camera. We can represent the intra-camera images’ corresponding
labels as follows: {𝑦ˆ𝑐𝑣𝑖 , 𝑦ˆ𝑐𝑟𝑖 }, where 𝑐𝑖 denotes the camera sequence number in each modality. In intercamera training stage, the MBLM strategy calculates plausible associations at camera-level between
distinct person classes and each camera (intra-modality) to mine for reliable labels. Additionally,
we adopt UPR module to refine the unreliable labels and result in modal-specific labels {𝑦ˆ𝑣 , 𝑦ˆ𝑟 }.
For inter-modality training stage, the objective is to improve the model’s cross-modality and crosscamera generalization. We also employ the two modules mentioned above, and adopt inter-modal
clustering to generate modality-shared labels {𝑦ˆ𝑚 }. Eventually, all the training samples will be
assigned two pseudo-labels, which are {𝑦ˆ𝑟 , 𝑦ˆ𝑚 } for infrared samples and {𝑦ˆ𝑣 , 𝑦ˆ𝑚 } for visible samples.
3.2

MBLM

Previous researches [32, 36–38, 51, 59] have calculated the Jaccard distance and utilize distance
encoding to achieve its results. However, the differences in illumination and viewpoint are highly
concerning in visible modality, which can cause identity splitting. To mitigate cross-camera and
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.

Unsupervised Visible-Infrared Person ReID via MCBL

357:7

cross-modality discrepancies, we propose the MBLM module to eliminate unreliable labels in
clustering and establish effective cluster associations.
In existing cluster-based methods, all of the features are typically utilized as a probe to calculate
the k-reciprocal encoding vector [67] in relation to features, thereby obtaining the distance encoding:
𝐷𝑖 = [𝑑𝑖,1, 𝑑𝑖,2, ...𝑑𝑖,𝑛 ],
(
𝑒𝑥𝑝 (−M (𝑓𝑖 , 𝑓 𝑗 )) if 𝑓 𝑗 ∈ K (𝑓𝑖 , 𝑘 1 )
𝑑𝑖,𝑗 =
0
𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒.

(1)
(2)

where 𝑓 𝑗 and 𝑓𝑖 indicate the feature of gallery and probe features, respectively. M (., .) is the function
that represents the Mahalanobis distance. Furthermore, K (𝑓𝑖 , 𝑘 1 ) denotes the set of k-reciprocal
nearest neighboring feature for 𝑓𝑖 , while 𝑘 1 serves as that adjusts the proximity range in order to
discriminate these nearest neighbors effectively.
After acquiring k-reciprocal encoding vector 𝐷𝑖 , we expand the vector further to associate more
hard samples. To be specific, it is extended by aggregating the distance encoding of top-𝑘 2 most
similar gallery instances from reciprocal neighbors. This enables a comprehensive relationship
between the samples. Additionally, by averaging these distance encodings, it can further integrate
their contextual information:
𝑘2
1 Õ
𝐷˜𝑖 =
𝐷𝑗,
𝑘 2 𝑗=1

(3)

where 𝑘 2 should be less than 𝑘 1 to avoid noise introduction and 𝐷˜ 𝑖 is the expanded distance
encoding. Afterwards, to obtain the embedding for clustering, we calculate the Jaccard distance
between the probe instance and other instances:
Í𝑛
𝑚𝑖𝑛(𝑑𝑖,𝜋 , 𝑑 𝑗,𝜋 )
𝐽 (𝑓𝑖 , 𝑓 𝑗 ) = 1 − Í𝑛𝜋 =1
,
(4)
𝑚𝑎𝑥
(𝑑𝑖,𝜋 , 𝑑 𝑗,𝜋 )
𝜋 =1
𝐽 (𝑓𝑖 ) = [𝐽 (𝑓𝑖 , 𝑓1 ), 𝐽 (𝑓𝑖 , 𝑓2 ), · · · , 𝐽 (𝑓𝑖 , 𝑓𝑛 )],
(5)
where 𝐽 (𝑓𝑖 ) indicates the embedding of 𝑓𝑖 , whereas 𝐽 = [𝐽 (𝑓𝑖 , 𝑓1 ), 𝐽 (𝑓𝑖 , 𝑓2 ), · · · , 𝐽 (𝑓𝑖 , 𝑓𝑛 )]𝑇 is served
as the precomputed matrix for clustering algorithm [12] to generate pseudo-labels 𝑦ˆ = DBSCAN(𝐽 ).
In addition, min and max are element-wise minimum and maximum computations of two input
vectors, respectively.
Nevertheless, despite obtaining pseudo-labels {𝑦ˆ𝑐𝑣𝑖 , 𝑦ˆ𝑐𝑟𝑖 } through the DBSCAN clustering algorithm during the inter-camera training stage, a large number of noise labels are simultaneously
generated, leading to the splitting of each identity. To address this problem, we refine samples and
pseudo-labels from different cameras to guarantee that identities are consistent between cameras.
In particular, we first adopt the following formula to calculate the cluster centroid within each
camera, which have same pseudo-label:
1 Õ
𝑐𝑣
G𝑘𝑖 = 𝑣
1{𝑦ˆ𝑐𝑣𝑖 = 𝑘 }𝑓 (𝑥 𝑣𝑗 ),
(6)
𝑛𝑘 𝑗
1 Õ
𝑐𝑟
G𝑘𝑖 = 𝑟
1{𝑦ˆ𝑐𝑟𝑖 = 𝑘 }𝑓 (𝑥 𝑟𝑗 ),
(7)
𝑛𝑘 𝑗
where 𝑛𝑟𝑘 and 𝑛𝑘𝑣 represent the number of infrared and visible samples for the kth identity in
camera {𝑐𝑖𝑣 , 𝑐𝑖𝑟 }, respectively. 1{.} is the indicator function. During the process of calculating the
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.

357:8

J. He et al.
𝑐𝑣

𝑐𝑟

𝑐𝑣

intra-camera cluster centroid {G𝑘𝑖 , G𝑘𝑖 }, we are able to obtain the instance set 𝑍𝑘𝑖 = {𝑥 𝑣𝑗 } 𝑦ˆ𝑐𝑣𝑖 =𝑘 and
𝑐𝑟

𝑍𝑘𝑖 = {𝑥 𝑟𝑗 } 𝑦ˆ𝑐𝑟𝑖 =𝑘 with the same label within each camera.
Generally, given instances with the same label, these instances can be classified into clusters of
consistent identities. Moreover, these cluster centers should ensure consistent similarity with all
instances within the cluster. In other words, if there are multiple instances belongs to the same
identity, then the similarity between these instances and the cluster center of that identity should
be the highest. Subsequently, we compute the similarity between each camera’s cluster instance
and the cluster centroids of the remaining cameras within the same modality. For example, given a
𝑐𝑣
𝑐𝑟
pair of visible features < 𝑥 𝑣𝑗 , G𝑘𝑖 > 𝑥 𝑣𝑗 ∈ 𝑍𝑘𝑖 , we compute the similarity by:
𝑐𝑣

𝑐𝑣
𝑐𝑣
𝑆 𝑗 𝑖 = 𝑠𝑖𝑚(𝑥 𝑣𝑗 , G𝑘𝑖 ) =

𝑥 𝑣𝑗 · G𝑘𝑖

,

𝑐𝑣

(8)

||𝑥 𝑣𝑗 || 2 ||G𝑘𝑖 || 2

𝑐𝑣

where 𝑆 𝑗 𝑖 is the similarity matrix between each instance and other cameras’ cluster centroids.
𝑐𝑣

Particularly, the cluster centroids G𝑘𝑖 and the instance 𝑥 𝑣𝑗 should not belong to the same camera
𝑐𝑟

𝑐𝑖𝑣 . Moreover, the similarity matrix 𝑆 𝑗 𝑖 in infrared cameras is calculated in the same way. Upon
converting the matrix to the one-hot format, we can classify the sample into two categories by
analyzing the similarity between the sample and the centroid of other cameras. Most of the samples
show high similarity with the same camera centroid, while some samples show high similarity with
other camera centroids. We represent them as reliable samples and unreliable samples, respectively.
Consequently, by setting the label of unreliable sample to −1, we are able to effectively balance the
influence of each camera on the labels, while exploring the fine-grained features.
Analogous to inter-camera training stage, we introduce the MBLM module to inter-modal training
stage. Following the assignment of modality-specific labels {𝑦ˆ𝑣 , 𝑦ˆ𝑟 }, we obtain the cluster centroid
{G𝑘𝑣 , G𝑟𝑘 } and instance set, respectively. Additionally, the instance set in different modalities can be
expressed in terms of 𝑍𝑘𝑣 = {𝑥 𝑣𝑗 } 𝑦ˆ 𝑣 =𝑘 and 𝑍𝑘𝑟 = {𝑥 𝑟𝑗 } 𝑦ˆ𝑟 =𝑘 . Therefore, for the VIS modality, we have:
𝑆 𝑣𝑗 = 𝑠𝑖𝑚(𝑥 𝑣𝑗 , G𝑟𝑘 ) =

𝑥 𝑣𝑗 · G𝑘𝑣
||𝑥 𝑣𝑗 || 2 ||G𝑘𝑣 || 2

,

(9)

,

(10)

and for the Near Infrared (NIR) modality, we have:
𝑆 𝑟𝑗 = 𝑠𝑖𝑚(𝑥 𝑟𝑗 , G𝑘𝑣 ) =

𝑥 𝑟𝑗 · G𝑟𝑘
||𝑥 𝑟𝑗 || 2 ||G𝑟𝑘 || 2

where {𝑆 𝑣𝑗 , 𝑆 𝑟𝑗 } represent the similarity matrix between cluster centroids and instance in different
modality. Similar to the inter-camera training stage, by converting {𝑆 𝑣𝑗 , 𝑆 𝑟𝑗 } into one-hot format, the
unreliable label can be identified from the matrix.
In brief, the MBLM is employed between clustering and contrastive learning stages. Unlike
conventional approaches, MBLM explicitly incorporates both camera and modality associations
to assess label reliability. This allows the framework to filter out unreliable pseudo-labels caused
by viewpoint and illumination discrepancies. By balancing contributions from different cameras
and modalities, MBLM prevents identity fragmentation and reduces label noise, leading to robust
cross-camera and modal alignment and higher-quality contrastive learning.
3.3

UPR

The reliance on MBLM exclusively during the intra-camera training phase inevitably results in the
loss of a considerable number of instances, which leads to a lack of training samples. However,
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 979-8-3503-6874-1/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICASSP49660.2025.10887740

Unveiling Local Well-posedness Influence for
Cross-modal Person Re-Identification
Yumeng Yang1

Guan-Nan Dong*2

Aichun Zhu3

Mingcheng Ni4

Yifeng Li5

Nanjing Tech University
yangyumeng@njtech.edu.cn, guannandong@outlook.com, aichun.zhu@njtech.edu.cn, nmc.njtech@gmail.com,
lyffz4637@163.com

Abstract—The existing cross-modal retrieval methods trend
toward the conventional multi-modal alignment while ignoring
the localization bias caused by visual hallucination, including
color pollution and appearance-like occlusion due to uncontrollable factors such as weather, illumination, and occlusion. This
feature blinding misleads the model to lock in the pseudo-real
position and further leads to local unmatched. To this end, we
discuss cross-modal local alignment well-posedness by making
a phased local modal-masking to calibrate the undisturbed
actual local alignment from entity, attribute, and appearance.
Specifically, we introduce a mask-based local well-posedness
modeling (MLWM) strategy, including text-based entity masking
(TEM), text-based attribute-specific masking (TAM), and imagebased appearance masking (IAM) to phased collaboratively
consider image prompting-based text entities, image promptingbased text attributes, and text prompting-based appearance
inference contrast, respectively. Finally, we dynamically optimize
the weights of positively correlated image-text pairs by comparing the similarity between original and reconstructed features.
Experimental results demonstrate that our method is effective on
three public datasets.
Index Terms—Person re-identification, text-to-image, local
well-posedness

I. I NTRODUCTION
Cross-modal person Re-ID focuses on aligning fine-grained
and global/local feature matching [1, 2] between the two
modalities. For example, local matching approaches [3, 4, 5,
6] is introduced by establishing the correspondence between
body parts and text, and Fujii et al. [7] achieved bidirectional semantic alignment by introducing unmasked tokens
to predict randomly masked image and text tokens. Lin et
al. [8] utilized the BLIP visual-language backbone network
to extract features and establish local alignment between
textual attributes and image patches. Despite the advancements
achieved by the aforementioned methods, aligning image-text
pairs is predominantly accomplished through the application
of beam splitters or alignment algorithms on sample pairs.
These techniques exhibit limitations under conditions of extreme interference. For instance, color distortion caused by
severe weather phenomena (such as rain, snow, and fog),
variations caused by illumination (including extreme exposure
and differing times of day), and appearance-like occlusions
caused by specific materials (like wood, opaque substances,
and glass) can lead to weakly aligned or even misaligned
* Corresponding author.

image-text pairs [9]. The misalignment at a local level results in discrepancies in viewpoints, locations, and scales
for identical objects; consequently, this leads to mismatched
real descriptions at corresponding locations, which diminishes
the accuracy of existing detection methods and complicates
convergence during training processes. Specifically, clothing
items of varying colors may appear similar under different
lighting conditions, potentially resulting in erroneous textual
descriptions. Furthermore, entities that resemble the target may
be incorrectly identified as such when occlusions are present.
To address the above problems, we explore the image-text
pair’s local well-posedness in different scenarios to measure
the tolerance of the image-text local misalignment produced by
severe factors. We propose a mask-based local well-posedness
modeling strategy (MLWM) to correct for misalignment of the
corresponding visible and text features from entity, attribute,
and appearance and further infer the fact that there may not
be features within the visible feature map that correspond
to the text description. Specifically, our method introduces
text-based entity masking (TEM), text-based attribute-specific
masking (TAM), and image-based appearance masking (IAM),
respectively, to better capture the correlation between the
visible and text domains. As shown in the Fig. 1, the main
idea is to build the information from the unmasked modality
corresponding to the masked region of the other modality.
The main contributions of this paper can be summarized as
follows:
We introduce local well-posedness to mitigate the impact of the image-text local misalignment under extreme
conditions by reconstructing condition-unrelated text and
appearance prompts from entity, attribute, and appearance
so that we can enlarge the tolerance for extreme conditions.
• The proposed mask-based local well-posedness modeling
approach considers all strong correlations between local
information in a phased way, including image promptingbased text entities (TEM), image prompting-based text
attributes contrast (TAM), and text prompting-based appearance inference (IAM), respectively.
• Extensive experiments on three public benchmark
datasets, i.e., CUHK-PEDES [10], ICFG-PEDES [11],
and RSTPReid [12], show the effectiveness of MLWM.
•

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:45 UTC from IEEE Xplore. Restrictions apply.

Fig. 1: Overview of our MLWM framework. It consists of three key components: 1) Text-based Entity Masking (TAM)
reconstructs the masked text entity regions using image prompts; 2) Text-based Attribute Masking (TAM) reconstructs the
masked text attribute regions utilizing image prompts; 3) Image-based appearance masking (IAM) infers the masked appearance
regions by text prompts.

II. M ETHOD
A. Architecture Overview
As shown in Fig. 1, the MLWM framework is based on an
encoder-decoder structure. For the encoder, we incorporate a
BLIP pre-trained VIT model [13, 14] and CNN as the image
encoder to obtain image features. Additionally, add a learnable
sequence aggregate information [CLS]v token at the beginning of the sequence. The final image embeddings are represented as Vg {vg icls , vg i1 , . . . , vg in } and Vl {vl icls , vl i1 , . . . , vl in },
respectively. Meanwhile, the text features are extracted by
using BERT [15], and appending special tokens [CLS]t and
[EOS] at the beginning and end of the text embeddings,
respectively, which denoted as {ticls , ti1 , . . . , tin , tieos }. For the
decoder, we propose a mask-based local well-posedness modeling module, which reconstructs the masked regions within
a modality based on the corresponding information from the
unmasked modality to balance the tolerance of the local
misalignment produced by severe factors and enhance the local
well-posedness of image-text pairs.
Global Association The global association is obtained by
aligning global features of matching image-text pairs in a
shared cross-modal space. Given an image I and a text T ,
i
we obtain feature representations Vo {vcls
, v1i , . . . , vni } and
i
i
i
i
To {tcls , t1 , . . . , tn , teos }, where Vo is combined of Vg and
Vl . We define image-text pairs as (vo , to )y=i , where y = 1
indicates a positive sample (strong correlation) between the

image and text, and y = 0 indicates a negative sample (weak
or no correlation). The probability of matching from image to
text can be calculated by:
exp(sim(I, T )/τ )
P i2t = PN
k=1 exp(sim(I, Tk )/τ )

(1)

where τ is a temperature parameter, and sim(vo , to ) =
vo⊤ to /∥vo ∥∥to ∥ The cross-modal matching loss is formulated:
Li2t = KL(pi2t ∥q i2t ) + KL(q i2t ∥pi2t )

(2)

where q i represents the probability of a positive sample.
KL(pi ∥q i ) represents the KL divergence between p and
q, measuring the information loss when the distribution pi
approximates the true distribution q i .
B. Mask-based local well-posedness modeling
The global information matching approach [16] addresses
the holistic feature representation of images and text. However, it is susceptible to localization bias caused by visual
hallucination under extreme conditions, which can hinder the
precise localization of local information in subsequent tasks.
To address this problem, we propose a mask-based local wellposedness modeling method that corrects for misalignment of
the corresponding image and text and enhances tolerance to
mismatched image-text pairs caused by local misalignment.
Specifically, our method introduces text-based entity masking (TEM), text-based attribute-specific masking (TAM), and

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:45 UTC from IEEE Xplore. Restrictions apply.

image-based appearance masking (IAM), respectively. This
method infers the content of the masked region in the other
modality by the information from the unmasked modality,
which can better capture the correlation between the visual
and text domains.
1) Text-based Entity Masking (TEM): We leverage the
image prompt-based text entity reconstruction to build the
interaction between images and text entities. To effectively
distinguish between textual entities and attributes, we use the
Natural Language Toolkit (NLTK) for phrase tokenization,
segmenting a text into multiple phrases in the format of
[attribute][entity], such as [red][shoes]. Specifically, given a
mini-batch of image-text pairs I − T , the entity/attribute
words in the text are masked to obtain I − T ′ . We then feed
I − T ′ into the decoder to generate the masked embeddings
I − Tl′ for entity masking and I − Ta′ for attribute masking.
Sequence positions in paired embeddings are aligned to obtain
corresponding positional feature representations. Based on the
contextual information of the masked text regions (the visual
features Vo and the unmasked text features To ), the possible
words are inferred from the vocabulary. These inferred words
are then input into a classifier to compute the matching
probability of each word corresponding to the masked position.
2) Text-based Attribute-specific Masking (TAM) : We leverage the image prompt-based text attribute reconstruction to
build the interaction between images and text attributes. Identifying whether a person is the same across different spaces
relies heavily on external attributes, which, apart from facial
features, are the most easily correlated. Therefore, exploring cross-modal interactions of attributes-level in person reidentification tasks is essential. Similar to TEM, given the
embedding pairs I − Ta′ of masked text attributes and the
complete visual context, the original signals are reconstructed
by cross-modal interactions. The cross-entropy loss function
is used to calculate the difference between the model’s predictions and the true labels. This module associates image
vision and text attributes, learns the visual differences caused
by different external factors, improves the local matching of
image-text pairs with different attribute features and positive
correlation, and reduces the learning of locally mismatched
image-text pairs.
X
1
(
KL(ymask ∥pmask ))
(3)
LT EM,T AM =
|P |
(Vo ,To )

where |P | represents the set of positive image-text pairs in the
mini-batch. ymask represents the masked words selected from
the vocabulary. pmask represents the prediction for masked
token.
3) Image-based Appearance Masking (IAM) : Due to the
local alignment from text to image is different from the above
content, which need to have a better understanding of the
text and image, and can better capture the correlation between
text and vision. So we proposed the IAM module for further
optimization.
Specifically, given an image I, we randomly mask out the
image with a probability of α. The masked image is defined as

Im . Then, we utilize the decoder fe , including multiple residual attention modules, where each residual attention module
consists of a multi-head attention layer followed by a feedforward network, utilizing the QuickGELU activation function.
The masked image Im is fed into an encoder to obtain masked
image features Vm {vm icls , vm i1 , . . . , vm imask , . . . , vm in }, utilizing an attention module to align the masked image features
with text features, and using a given MIM head to reconstruct
the original image. The IAM loss LIAM is defined as follows:
LIAM =

1
∥I − fd (fe (Vm , To ))∥1
Ω(I)

(4)

where fd is the cross-modal decoder, and ∥ · ∥1 denotes the
L1 loss, which measures the difference between the predicted
and actual images.
In summary, the overall loss of our local well-posedness
modeling method can be formulated as follows:
LM LW M = LT EM + LT AM + LIAM

(5)

III. E XPERIMENTS
A. Implementation Details
During training, random horizontal flipping, image border
augmentation and padding, random cropping, and random
erasing are used for image data augmentation. WordNet synonym replacement, random insertion of synonyms, random
swapping, and random deletion are used for text data augmentation. The mask rates for TEM, TAM, and IAM are set to 0.8,
and the temperature parameter is set to 0.02. All images are
resized to 224 × 224. The maximum text sequence length is
set to 77. Our model is trained with AdamW optimizer with
a batch size of 40 for 50 epochs, an initial learning rate of
1e−5, and cosine learning rate decay.
B. Comparison with other SOTA methods
For all experiments, we adopt Rank-1, Rank-5, and Rank-10
as the primary evaluation metrics. Additionally, we comprehensively use mean Average Precision (mAP) [17] to evaluate
model performance. We compare our method with SOTA
methods on three public datasets. As shown in Tab. I.
CUHK-PEDES: We evaluate the proposed method on
CUHK-PEDES. As shown in the first column of Tab. I,
our proposed MLWM achieved 78.46% Rank-1 accuracy
and 69.11% mAP and improving by 0.09% and 0.24%,
respectively, compared to the best performing method. The
experimental results demonstrate that MLWM can effectively
achieve cross-model interaction and matching between visual
and textual.
ICFG-PEDES: Our experimental results as shown in the
second column of Tab. I. Our proposed MLWM outperforms
existing methods in terms of Rank-1 performance under
the hybrid feature fusion and matching inference, achieving
a Rank-1 accuracy of 68.80%, surpassing the most recent
method by + 0.99%. These results demonstrate that MLWM
can effectively enhance the tolerance of the image-text pair’s

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:45 UTC from IEEE Xplore. Restrictions apply.

TABLE I: Performance comparisons with state-of-the-art methods on CUHK-PEDES, ICFG-PEDES and RSTPReid dataset.
“G” and “L” in “Type” denote the global-matching/local-matching method. Results are ordered based on the Rank-1 accuracy.
Method

Type

MIA [18]
ViTAA [19]
DSSL [12]
SSAN [11]
LBUL [20]
IVT [21]
ACSA [22]
CFine [23]
IRRA [24]
CADA [8]
MLWM (Ours)

L
L
L
L
L
G
L
G
G
L
L

Rank-1
53.10
55.97
59.98
61.37
64.04
65.59
68.67
69.57
73.38
78.37
78.46

CUHK-PEDES
Rank-5
Rank-10
75.00
82.90
75.84
83.52
80.41
87.56
80.15
86.73
82.66
87.22
83.11
89.21
85.61
90.66
85.93
91.15
89.93
93.71
91.57
94.58
91.22
94.11

mAP
51.60
66.13
68.87
69.11

Rank-1
54.23
56.04
60.83
63.46
67.81
68.80

ICFG-PEDES
Rank-5
Rank-10
72.63
79.53
73.60
80.22
76.55
82.42
80.25
85.82
82.34
87.14
82.27
85.96

mAP
38.06
39.85
37.10

Rank-1
46.49
50.98
39.05
43.50
45.55
46.70
48.40
50.55
60.20
69.60
70.90

RSTPReid
Rank-5
Rank-10
67.14
75.18
68.79
75.78
62.60
73.95
67.80
77.15
68.2
77.85
70.00
78.80
71.85
81.45
72.50
81.60
81.30
88.20
86.75
92.40
86.85
91.55

mAP
47.17
52.74
53.35

TABLE II: Ablation study on each component of MLWM on
RSTPReid
Components
TEM
TAM
IAM
✓
✓
✓

✓

✓
✓

Rank-1
68.75
69.60
69.60
69.40
70.90

RSTPReid
Rank-5
Rank-10
85.05
90.30
86.55
92.65
91.57
92.40
86.05
91.10
86.85
91.55

local well-posedness. However, the method is slightly less effective in terms of Rank-5, Rank-10, and mAP. This method is
slightly less effective in terms of Rank-5, Rank-10, and mAP,
but we optimize the accuracy of the top-ranked result more,
which does not significantly impact the overall performance
of the model.
RSTPReid: The evaluation results of our proposed method
on the RSTPReid dataset are shown in the third column of
Tab. I. Our method significantly outperforms existing methods,
surpassing CADA by 1.3% and 0.61% in Rank-1 accuracy and
mAP, respectively.
C. Ablation Study
To fully demonstrate the effectiveness of different components within the MLWM framework, we conducted experiments on RSTPReid, comparing the performance across Rank1, Rank-5, and Rank-10 metrics. The experimental results are
presented in Tab. II. From this table, the best performance is
achieved when using TEM, TAM, and IAM simultaneously.
Comparing the results in NO.1 vs. NO.5, it is evident that constructing local detail associations between modalities greatly
enhances precision. Specifically, obtaining improvements of
+ 2.25%, 1.80% and 1.25% on Rank-1, Rank-5, and Rank-10
accuracy across RSTPReid, respectively. These results demonstrate that MLWM considers all strong correlations in local
information, uses modality prompting-based to introduce local
well-posedness to mitigate the mismatch of local information
under extreme conditions.
D. Qualitative Results
Fig. 2 presents the top-5 retrieval results from the Baseline
(first row) and our model (second row) given the text query
shown at the top. An image with a green frame are true

Fig. 2: Comparison of top-5 retrieved results on RSTPReid
between Baseline (the first row) and MLWM (the second row)
for each text query. The matched and mismatched images are
marked with green and red rectangles, respectively.

positives while the one with a red frame are false positives.
The results clearly demonstrate that the MLWM method, based
on Mask-based Local Well-posedness Modeling, can retrieve
the correct target person more correctly.
IV. C ONCLUSION
In this paper, we propose a mask-based local well-posedness
modeling method to establish representational relationships
between the two modalities. This method combines TEM,
TAM and IAM to phased collaboratively consider image
prompting-based text entities, image prompting-based text
attributes and text prompting-based appearance inference contrast, respectively. Calculating a similarity matching loss between the original and reconstructed features and dynamically
optimize the weights of positively correlated image-text pairs.
Experimental results demonstrate that the Mask-based Local
Well-posedness Modeling method is effective on three public
datasets.
Future Work Our future work will proceed in the following two directions: 1) Extend the component to other
tasks/domains to assess its applicability. 2) Develop crossmodal models for better feature extraction and alignment.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:45 UTC from IEEE Xplore. Restrictions apply.

R EFERENCES
[1] Chen, Y., Zhang, G., Lu, Y., Wang, Z. & Zheng, Y.
TIPCB: A simple but effective part-based convolutional
baseline for text-based person search. Neurocomputing.
494 pp. 171-181 (2022)
[2] Liu, J., Zha, Z., Hong, R., Wang, M. & Zhang, Y. Deep
adversarial graph attention convolution network for textbased person search. Proceedings Of The 27th ACM
International Conference On Multimedia. pp. 665-673
(2019)
[3] Jing, Y., Si, C., Wang, J., Wang, W., Wang, L. &
Tan, T. Pose-guided multi-granularity attention network
for text-based person search. Proceedings Of The AAAI
Conference On Artificial Intelligence. 34, 11189-11196
(2020)
[4] Taylor, W. “Cloze procedure”: A new tool for measuring
readability. Journalism Quarterly. 30, 415-433 (1953)
[5] Gong, X., Yao, Z., Li, X., Fan, Y., Luo, B., Fan, J. &
Lao, B. LAG-Net: Multi-granularity network for person
re-identification via local attention system. IEEE Transactions On Multimedia. 24 pp. 217-229 (2021)
[6] Shao, Z., Zhang, X., Fang, M., Lin, Z., Wang, J. & Ding,
C. Learning granularity-unified representations for textto-image person re-identification. Proceedings Of The
30th Acm International Conference On Multimedia. pp.
5566-5574 (2022)
[7] Fujii, T. & Tarashima, S. Bilma: Bidirectional localmatching for text-based person re-identification. Proceedings Of The IEEE/CVF International Conference On
Computer Vision. pp. 2786-2790 (2023)
[8] Lin, D., Peng, Y., Meng, J. & Zheng, W. Cross-Modal
Adaptive Dual Association for Text-to-Image Person
Retrieval. IEEE Transactions On Multimedia. (2024)
[9] Lin, X., Yang, Y. & Niu, Z. Enhance Part-Based Model
for Person Re-Identification with Fused Multi-Scale Features. ICASSP 2020 - 2020 IEEE International Conference On Acoustics, Speech And Signal Processing
(ICASSP). pp. 4092-4096 (2020)
[10] Li, S., Xiao, T., Li, H., Zhou, B., Yue, D. & Wang, X.
Person search with natural language description. ArXiv
Preprint ArXiv:1702.05729. (2017)
[11] Ding, Z., Ding, C., Shao, Z. & Tao, D. Semantically selfaligned network for text-to-image part-aware person reidentification. ArXiv Preprint ArXiv:2107.12666. (2021)
[12] Zhu, A., Wang, Z., Li, Y., Wan, X., Jin, J., Wang, T., Hu,
F. & Hua, G. Dssl: Deep surroundings-person separation
learning for text-based person retrieval. Proceedings Of
The 29th ACM International Conference On Multimedia.
pp. 209-217 (2021)
[13] Alexey, D. An image is worth 16x16 words: Transformers for image recognition at scale. ArXiv Preprint ArXiv:
2010.11929. (2020)
[14] Li, J., Li, D., Xiong, C. & Hoi, S. Blip: Bootstrapping
language-image pre-training for unified vision-language
understanding and generation. International Conference

On Machine Learning. pp. 12888-12900 (2022)
[15] Kenton, J. & Toutanova, L. Bert: Pre-training of deep
bidirectional transformers for language understanding.
Proceedings Of NaacL-HLT. 1 pp. 2 (2019)
[16] Zheng, Z., Zheng, L., Garrett, M., Yang, Y., Xu, M.
& Shen, Y. Dual-path convolutional image-text embeddings with instance loss. ACM Transactions On Multimedia Computing, Communications, And Applications
(TOMM). 16, 1-23 (2020)
[17] Ye, M., Shen, J., Lin, G., Xiang, T., Shao, L. & Hoi, S.
Deep Learning for Person Re-Identification: A Survey
and Outlook. IEEE Transactions On Pattern Analysis
And Machine Intelligence. 44, 2872-2893 (2022)
[18] Niu, K., Huang, Y., Ouyang, W. & Wang, L. Improving description-based person re-identification by multigranularity image-text alignments. IEEE Transactions On
Image Processing. 29 pp. 5542-5556 (2020)
[19] Wang, Z., Fang, Z., Wang, J. & Yang, Y. Vitaa: Visualtextual attributes alignment in person search by natural
language. Computer Vision–ECCV 2020: 16th European
Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16. pp. 402-420 (2020)
[20] Wang, Z., Zhu, A., Xue, J., Wan, X., Liu, C., Wang, T. &
Li, Y. Look before you leap: Improving text-based person
retrieval by learning a consistent cross-modal common
manifold. Proceedings Of The 30th ACM International
Conference On Multimedia. pp. 1984-1992 (2022)
[21] Shu, X., Wen, W., Wu, H., Chen, K., Song, Y., Qiao, R.,
Ren, B. & Wang, X. See finer, see more: Implicit modality alignment for text-based person retrieval. European
Conference On Computer Vision. pp. 624-641 (2022)
[22] Ji, Z., Hu, J., Liu, D., Wu, L. & Zhao, Y. Asymmetric
cross-scale alignment for text-based person search. IEEE
Transactions On Multimedia. 25 pp. 7699-7709 (2022)
[23] Yan, S., Dong, N., Zhang, L. & Tang, J. Clip-driven
fine-grained text-image person re-identification. IEEE
Transactions On Image Processing. (2023)
[24] Jiang, D. & Ye, M. Cross-modal implicit relation reasoning and aligning for text-to-image person retrieval.
Proceedings Of The IEEE/CVF Conference On Computer
Vision And Pattern Recognition. pp. 2787-2797 (2023)

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:45 UTC from IEEE Xplore. Restrictions apply.


codex
第五篇的第 8 页停在 `UPR` 开头，方法还没有完整闭合。我会继续读后续方法页，重点补齐被拒伪标签怎么重对齐，以及混合对比学习怎么把层级训练串起来。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 13 'Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Unsupervised Visible-Infrared Person ReID via MCBL

357:9

Fig. 3. Diagram of the intra-camera training stage, MBLM and UPR in visible modality. Stars denote the
cluster centroids generated through clustering, while circles represent samples belonging to the same identity.
Notably, “−
→” indicates the correlation after assessing similarity between samples and all other cluster centers
within the same camera, whereas “d” outlines the strategy for re-aligning the sample when it is determined
that its label is unreliable.

in the initial phase, the model has limited discriminative ability, and these samples are unable to
satisfy the training requirements. UPR can effectively address this discrepancy by utilizing the
similarity between instances and each clustering centroid to redistribute the pseudo-labels.
As shown in Figure 3, our motivation is to reassign the pseudo-labels for the unreliable samples to
avoid a large number of missing samples. In MBLM strategy, we obtain the samples with unreliable
labels according to the similarity. Likewise, taking the RGB modality as an example, we represent
𝑐𝑣
the unreliable samples from different cameras as 𝑢 𝑗 𝑖 , and their pseudo-label are temporarily set
to −1. To facilitate the reassignment of labels, we compute the similarity between the features of
instances and cluster centroids within the same camera:
𝑐𝑣

𝑐𝑣
𝑐𝑣
𝑐𝑣
𝑆 𝑗 𝑖 = 𝑠𝑖𝑚(𝑢 𝑗 𝑖 , G𝑘𝑖 ) =
𝑐𝑣

𝑐𝑣

𝑢 𝑗 𝑖 · G𝑘𝑖
𝑐𝑣

,

(11)

||𝑢 𝑗 𝑖 || 2 ||G𝑟𝑘 || 2

𝑐𝑟

where {𝑆 𝑗 𝑖 , 𝑆 𝑗 𝑖 } is the similarity matrix used to align samples. In this case, 𝑐𝑖𝑣 denotes that the
sample and cluster centroids belong to the same camera. Thus, we reassign new labels to the
unreliable samples through a process of re-alignment. Subsequently, we recalculated the center
of each cluster based on these updated labels to achieve a more accurate representation of their
identities:
1 Õ
𝑐𝑣
Ĝ𝑘𝑖 = 𝑣
1{[𝑦ˆ𝑐𝑟𝑒𝑖 , 𝑦ˆ𝑐𝑣𝑖 ] = 𝑘 }𝑓 (𝑥 𝑣𝑗 ),
(12)
𝑛ˆ𝑘 𝑗
𝑐𝑣

where Ĝ𝑘𝑖 represents the new cluster centroids after the sample is re-aligned in each RGB cameras.
[ · ] denotes the concatenation operation, which integrates previous labels with re-aligned labels.
Similarly, the infrared modality performs the same inconsistent label alignment procedure. More𝑐𝑣
𝑐𝑟
over, The recalibrated cluster centers {Ĝ𝑘𝑖 , Ĝ𝑘𝑖 } are stored in intra-camera memory for comparative
learning purposes.
While conventional methods typically discard unreliable samples, UPR instead leverages similarity with camera centroids to reassign labels for low-confidence samples in the same camera domain.
This not only preserves valuable training data but also prevents performance degradation in the
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.

357:10

J. He et al.

early stages when discriminative ability is limited. Furthermore, by integrating UPR with MBLM,
the framework can progressively refine noisy labels rather than simply ignoring them, thus ensuring
higher label accuracy and more stable memory updates for subsequent contrastive learning.
3.4

HMCCL

According to the MBLM and UPR modules, the intra-camera pseudo labels {𝑦ˆ𝑐𝑣𝑖 , 𝑦ˆ𝑐𝑟𝑖 }, the intra-modal
pseudo labels {𝑦ˆ𝑣 , 𝑦ˆ𝑟 }, and inter-modal pseudo labels {𝑦ˆ𝑚 } are obtained. Moreover, to better refine
the distribution between clusters and instances in hierarchical learning framework, we develop
HMCCL.
Among unsupervised methods, the refined InfoNCE [10, 19] is the most common used, and its
loss function is expressed as:
L =−

𝑃Õ
×𝐾

𝑒𝑥𝑝 (𝑞𝑖 · 𝜙 [𝑦ˆ𝑖 ]/𝜏)
𝑙𝑜𝑔 Í𝐼
,
𝑖=1
𝑘=0 𝑒𝑥𝑝 (𝑞𝑖 · 𝜙 [𝑘]/𝜏)

(13)

where 𝑃 is the number of different categories of sampled clusters, and 𝐾 is the number of images
per cluster in single modality, respectively. 𝑞𝑖 indicates the query instance normalized by L2 in
training mini-batch. 𝜙 [𝑦ˆ𝑖 ] is the positive cluster representation of 𝑞𝑖 while 𝜙 [𝑘] is the centroid
of cluster in memory. More specifically, the cluster centroids are determined by taking the mean
of the instance features among the clusters. 𝜏 is temperature factor and 𝐼 represents the cluster
number. After each iteration, the momentum update strategy [19] is used to update the memory:
𝜙 [𝑦ˆ𝑖 ] ← 𝜆𝜙 [𝑦ˆ𝑖 ] + (1 − 𝜆)𝑞𝑖

(14)

(𝑖 = 1, 2, ..., 𝑃 × 𝐾).

As in the previous researches [58], we utilize the method with sampling instances randomly
to update the representation of each clusters. Then, we introduce instance-wise and cluster-wise
losses thereby refining partial and global distributions. Notably, we employ Hybrid Contrastive
Learning Strategy in the hierarchical domains, which enhance the model exploring the fine-grained
features in each training stages.
In intra-camera training, we establish same types of memories, i.e., cluster-wise and instance-wise
memories:
Õ
1
𝑐𝑟
𝑐𝑟
𝜙 𝐼 𝑖 [𝑖] = 𝑓𝑖𝑟 , 𝜙𝐶𝑖 [𝑦ˆ𝑐𝑟𝑖 ] =
𝑓𝑖𝑟 ,
𝑂 𝑟𝑦ˆ𝑟 𝑓𝑖𝑟 ∈𝑂 𝑟𝑦ˆ 𝑟
𝑐𝑖
𝑐𝑖
(15)
Õ
𝑣
1
𝑐𝑖𝑣
𝑐
𝜙 𝐼 [𝑖] = 𝑓𝑖𝑣 , 𝜙𝐶𝑖 [𝑦ˆ𝑐𝑣𝑖 ] =
𝑓𝑖𝑣 ,
𝑂 𝑦𝑣ˆ 𝑣 𝑓𝑖𝑣 ∈𝑂 𝑦𝑣ˆ 𝑣
𝑐𝑖

𝑐𝑖𝑟

𝑐𝑖𝑣

𝑐𝑖

𝑐𝑖𝑟

𝑐𝑣

where {𝜙 𝐼 , 𝜙 𝐼 } are the instance-wise memories, whereas {𝜙𝐶 , 𝜙𝐶𝑖 } are the cluster-wise memories,
both of them are based on the cameras within infrared and visible modality. {𝑂 𝑟𝑦ˆ𝑟 , 𝑂 𝑦𝑣ˆ 𝑣 } represent
𝑐𝑖

𝑐𝑖

the 𝑦ˆ𝑖𝑟 (𝑣) th cluster in infrared and visible modality, and | · | indicates the instances number in a

particular cluster. In order to ensure the starting point in the optimization process is stable, we
perform the cluster-wise memory initialization via averaging features. In addition, we randomly
select instances to update the instance-wise memory after each iteration in order to implement
real-time memory updates. Moreover, we implement memory-based Contrastive Learning for each
camera in intra-camera training stage, in order to balance the impact of each camera’s information
on the model.
Considering the essential role of CA in the visible modality, we combine CA data {C} with visible
data {V} and extract features from the merged dataset for memory update using random index.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.

Unsupervised Visible-Infrared Person ReID via MCBL

357:11

𝑐𝑣

𝑐𝑣

Concretely, we update the memory by {𝜙 𝐼 𝑖 [𝑖] ← 𝑓𝑖 (𝑣,𝑐 ) , 𝜙𝐶𝑖 [𝑦ˆ𝑐𝑣𝑖 ] ← 𝑓𝑖 (𝑣,𝑐 ) } in visible modality and
𝑐𝑟

𝑐𝑟

{𝜙 𝐼 𝑖 [𝑖] ← 𝑓𝑖𝑟 , 𝜙𝐶𝑖 [𝑦ˆ𝑐𝑟𝑖 ] ← 𝑓𝑖𝑟 } in infrared modality, respectively.
Furthermore, the contrast learning loss of the visible and infrared modalities at the cluster-level
can be expressed in the following way:
𝑐𝑣
L𝐶𝑖 = −

2×𝑃
Õ×𝑍
𝑖=1

𝑐

(𝑣,𝑐 )

𝑐𝑣

𝑒𝑥𝑝 (𝑞𝑖 𝑖 · 𝜙𝐶𝑖 [𝑦ˆ𝑐𝑣𝑣 ]/𝜏)
𝑖
𝑙𝑜𝑔 Í
,
𝑐𝑖𝑣
𝑐𝑖𝑣
𝐼
𝑘=0 𝑒𝑥𝑝 (𝑞𝑖 · 𝜙𝐶 [𝑘]/𝜏)
𝑐𝑟

𝑐𝑟

(16)

𝑒𝑥𝑝 (𝑞𝑖 𝑖 · 𝜙𝐶𝑖 [𝑦ˆ𝑐𝑟 𝑟 ]/𝜏)
𝑖
L𝐶 = −
𝑙𝑜𝑔 Í
,
𝑐𝑖𝑟
𝑐𝑖𝑟
𝐼
𝑒𝑥𝑝
(𝑞
·
𝜙
[𝑘]/𝜏)
𝑖=1
𝑖
𝑘=0
𝐶
𝑃Õ
×𝑍

𝑐𝑖𝑟

𝑐

(𝑣,𝑐 )

where 𝑞𝑖 𝑖 represents the query features within the camera from the merged dataset, which
combine CA and visible modalities in the mini-batch. 𝑞𝑖𝑟 represents the query features from the
camera in infrared modality. In the training process, we also employ instance-wise contrastive
learning to investigate the relationship between instances and the loss of contrastive learning can
be formulated as follows:
𝑐𝑣
L𝐼 𝑖 = −

2×𝑃
Õ×𝑍
𝑖=1

𝑐𝑖𝑟

L𝐼

𝑐

(𝑣,𝑐 )

𝑐𝑣

𝑒𝑥𝑝 (𝑞𝑖 𝑖 · 𝜙𝐶𝑖 [𝑦ˆ𝑐𝑣𝑣 ]/𝜏)
𝑖
𝑙𝑜𝑔 Í
,
𝑐𝑖𝑣
𝑐𝑖𝑣
𝐼
𝑒𝑥𝑝
(𝑞
·
𝜙
[𝑘]/𝜏)
𝑖
𝑘=0
𝐶
𝑐𝑟

𝑐𝑟

(17)

𝑒𝑥𝑝 (𝑞𝑖 𝑖 · 𝜙𝐶𝑖 [𝑦ˆ𝑐𝑟 𝑟 ]/𝜏)
𝑖
=−
𝑙𝑜𝑔 Í
.
𝑐𝑖𝑟
𝑐𝑖𝑟
𝐼
𝑖=1
𝑘=0 𝑒𝑥𝑝 (𝑞𝑖 · 𝜙𝐶 [𝑘]/𝜏)
𝑃Õ
×𝑍

Different from the above stage, we initialize both types of memory and focus on the intra-modality
by the following equations:
Õ
1
𝜙 𝐼𝑟 [𝑖] = 𝑓𝑖𝑟 , 𝜙𝐶𝑟 [𝑦ˆ𝑖𝑟 ] =
𝑓𝑖𝑟 ,
𝑟
𝑟
𝑟
𝑂 𝑦ˆ𝑟 𝑓𝑖 ∈𝑂 𝑦ˆ 𝑟
𝑖
𝑖
(18)
Õ
1
𝜙 𝐼𝑣 [𝑖] = 𝑓𝑖𝑣 , 𝜙𝐶𝑣 [𝑦ˆ𝑖𝑣 ] =
𝑓𝑖𝑣 ,
𝑂 𝑦𝑣ˆ 𝑣 𝑓𝑖𝑣 ∈𝑂 𝑦𝑣ˆ 𝑣
𝑖

𝑖

where {𝜙𝑖𝑟 , 𝜙𝑖𝑣 } are instance-wise memories for infrared and visible modalities, and {𝜙𝐶𝑟 , 𝜙𝐶𝑣 } are
cluster-wise memories. Similar to intra-camera stage, {𝑂 𝑟𝑦ˆ𝑟 , 𝑂 𝑦𝑣ˆ 𝑣 } represent the set of 𝑦ˆ𝑖𝑟 (𝑣) th cluster.
𝑖
𝑖
Primarily, we randomly leverage CA features and visible features for updates the memory in the
visible modality {𝜙 𝐼𝑣 [𝑖] ← 𝑓𝑖 (𝑣,𝑐 ) , 𝜙𝐶𝑣 [𝑦ˆ𝑖𝑣 ] ← 𝑓𝑖 (𝑣,𝑐 ) }. In the case of the infrared modality, we update
the memory by {𝜙 𝐼𝑟 [𝑖] ← 𝑓𝑖𝑟 , 𝜙𝐶𝑟 [𝑦ˆ𝑖𝑟 ] ← 𝑓𝑖𝑟 }. Additionally, the contrastive loss at the cluster-wise
for both visible and infrared modalities can be formulated as:
2×𝑃
Õ×𝑍
𝑒𝑥𝑝 (𝑞𝑖(𝑣,𝑐 ) · 𝜙𝐶𝑣 [𝑦ˆ𝑖𝑣 ]/𝜏)
L𝐶𝑣 = −
𝑙𝑜𝑔 Í𝐼
,
𝑣
𝑣
𝑖=1
𝑘=0 𝑒𝑥𝑝 (𝑞𝑖 · 𝜙𝐶 [𝑘]/𝜏)
(19)
𝑃Õ
×𝑍
𝑒𝑥𝑝 (𝑞𝑖𝑟 · 𝜙𝐶𝑟 [𝑦ˆ𝑖𝑟 ]/𝜏)
𝑟
L𝐶 = −
𝑙𝑜𝑔 Í𝐼
,
𝑟
𝑟
𝑖=1
𝑘=0 𝑒𝑥𝑝 (𝑞𝑖 · 𝜙𝐶 [𝑘]/𝜏)
where 𝑞𝑖(𝑣,𝑐 ) indicates the query features within the mini-batch derived from visible and CA
modalities, while 𝑞𝑖𝑟 indicates the query feature belongs to infrared modality. The instance-wise
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.

357:12

J. He et al.

contrastive loss can be expressed as:
L𝐼𝑣 = −
L𝐼𝑟 = −

2×𝑃
Õ×𝑍
𝑖=1
𝑃Õ
×𝑍

𝑒𝑥𝑝 (𝑞𝑖(𝑣,𝑐 ) · 𝜙 𝐼𝑣 [𝑦ˆ𝑖𝑣 ]/𝜏)
𝑙𝑜𝑔 Í𝐼
,
𝑣
𝑣
𝑘=0 𝑒𝑥𝑝 (𝑞𝑖 · 𝜙 𝐼 [𝑘]/𝜏)
𝑒𝑥𝑝 (𝑞𝑖𝑟 · 𝜙 𝐼𝑟 [𝑦ˆ𝑖𝑟 ]/𝜏)

𝑙𝑜𝑔 Í𝐼

(20)

.

𝑟
𝑟
𝑘=0 𝑒𝑥𝑝 (𝑞𝑖 · 𝜙 𝐼 [𝑘]/𝜏)

𝑖=1

To optimize the training process, we incorporate cluster-wise memory and instance-wise memory,
which are initialized as:
𝜙 𝐼𝑚 [𝑖] = 𝑓𝑖 (𝑣,𝑟 ) ,
Õ
1
𝜙𝐶𝑚 [𝑦ˆ𝑖𝑚 ] =
𝑓𝑖 (𝑣,𝑟 ) ,
𝑚
𝑂 𝑦ˆ𝑚 𝑓 (𝑣,𝑟 ) ∈𝑂𝑚𝑚
𝑖

𝑖

(21)

𝑦ˆ𝑖

where 𝑂 𝑚
indicates the 𝑦ˆ𝑖𝑚 th cluster set from the inter-modality clustering. In the memory-update
𝑦ˆ𝑖𝑚

phase, we concatenate all modalities and update memory through random selection, as indicated
{𝜙 𝐼𝑚 [𝑖] ← 𝑓𝑖 (𝑣,𝑟,𝑐 ) , 𝜙𝐶𝑚 [𝑖] ← 𝑓𝑖 (𝑣,𝑟,𝑐 ) }. Likewise, for inter-modality training stage, the contrastive
losses in cluster-wise and instance-wise can be expressed as:
L𝐶𝑚 = −

3×𝑃
Õ×𝑍
𝑖=1

L𝑚
𝐼 =−

3×𝑃
Õ×𝑍
𝑖=1

exp(𝑞𝑖(𝑣,𝑟,𝑐 ) · 𝜙𝐶𝑚 [𝑦ˆ𝑖𝑚 ]/𝜏)
log Í
,
(𝑣,𝑟,𝑐 )
𝐼
· 𝜙𝐶𝑚 [𝑘]/𝜏)
𝑘=0 exp(𝑞𝑖
exp(𝑞𝑖(𝑣,𝑟,𝑐 ) · 𝜙 𝐼𝑚 [𝑦ˆ𝑖𝑚 ]/𝜏)
log Í
,
(𝑣,𝑟,𝑐 )
𝐼
· 𝜙 𝐼𝑚 [𝑘]/𝜏)
𝑘=0 exp(𝑞𝑖

(22)

where 𝑞𝑖(𝑣,𝑟,𝑐 ) demonstrates that the sampled query features from different modality, including

visible, infrared, and CA.
In the beginning of training process, HMCCL conducts intra-camera learning and balances the
influence of each camera. In contrast, the implementation of direct intra- and inter-modal clustering
is prone to introduce a lot of noise labels and identity splitting. On top of that, we allow model to
learn for each camera, which enable to acquire a balanced modal representation. Hence, shifting
from intra-camera to intra-modal learning aids in guiding the model towards a more balanced
optimization. On the other hands, through MBLM and UPR, we establish a hierarchical learning
framework and learn the fine-grained features in each training stages.
3.5

Optimization Algorithm

The proposed MCBL network seamlessly integrates MBLM, UPR, and HMCCL into a unified
framework. It is worth noting that the aforementioned modules can establish synergies that balance
the impact of different cameras and modalities on the model, thus improving the robustness of
model against cross-modal and cross-camera differences. The MCBL framework consists of three
hierarchical training stages, i.e., intra-camera, inter-camera, and inter-modality training stages, since
its loss function in both level can be expressed as:
𝑐𝑟

𝑐𝑣

𝑐𝑟

𝑐𝑣

L𝐼 = L𝐼 𝑖 + L𝐼 𝑖 + L𝐼𝑟 + L𝐼𝑣 + L𝑚
𝐼 ,

(23)

L𝐶 = L𝐶𝑖 + L𝐶𝑖 + L𝐶𝑟 + L𝐶𝑣 + L𝐶𝑚 .
Consequently, the overall loss function for MCBL can be represented as:
L = 𝜑𝐼 L𝐼 + 𝜑𝐶 L𝐶 ,
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.

(24)

Unsupervised Visible-Infrared Person ReID via MCBL

357:13

Algorithm 1: The Flow of MCBL Approach

where {𝜑𝐼 , 𝜑𝐶 } indicate the tradeoff weights in instance-wise and cluster-wise loss function, respectively. The pseudo-codes generated by MCBL are delineated in Algorithm 1 and its flowchart
for the optimization process is illustrated in Figure 2, while epoch indicates the number of epoch in
each training stage, 𝜅 denotes the index of epoch, and Maxiter represents the number of iteration
within per training.
4

Experiments Evaluation

In this section, we will first introduce two widely used VI-ReID benchmarks: SYSU-MM01 [44]
and RegDB [30], and the details of our experimental implementation. Subsequently, we conduct
experiments on both challenging cross-modal datasets, comparing the proposed MCBL framework
with other State-of-the-Art (SOTA) methods to present the performance.
Eventually, we conduct several performance analyses on the proposed model in order to illustrate
the effects of each component.
4.1

Datasets and Settings

SYSU-MM01 is a large-scale visible-infrared pedestrian dataset, which is always utilized for validating effectiveness of the published model. This dataset includes 287,628 visible images and 15,792
infrared images, which are collected from 491 participants. Moreover, the images are gathered
from 6 cameras, comprising 4 RGB cameras and 2 infrared cameras. For the training set, it contains
22,258 VIS images and 11,909 NIR images with 395 identities. Conversely, in order to avoid any
overlap between the training set and the testing set, the latter is based on the remaining 96 individuals. In addition, it contains 3,803 infrared images in the query set, while 301 visible images are
randomly sampled for the gallery set. Accordingly, considerable variations in pedestrian postures
and clothing, the viewpoint camera, and the background illumination in different cameras are
depicted in Figure 4(a). In light of our objective to filter identity-irrelevant interference and learn
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 12, Article 357. Publication date: November 2025.


codex
现在六篇都已经读完方法主体，另补了两篇超过第 8 页的后续方法段。最后我会按你要的五个问题逐篇写，不复述模块流水账，而是抽它们如何从观察到问题定义，再到证据闭环。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 3 -l 5 'Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113509

Y. Xiao et al.

Fig. 4. Overview of our pipeline. We apply our curriculum learning, including clothes-level data scheduling and data weighting, on the standard SC re-ID pipeline.
Here, 𝐼𝐷1 represents the ID label, 𝐶𝐿𝑇 1 represents the clothes label, 𝑄 represents the classiﬁcation accuracy, 𝑤 represents the sample weight, 𝑘 represents the
threshold, and 𝑆𝐼𝐷1 represents the current silhouette coeﬃcient of 𝐼𝐷1. First, we simulate the human cognitive process by scheduling clothes data. Our scheduling
strategy involves silhouette coeﬃcient evaluation and maximization selection strategy (See Section 3.2 for details). Second, we dynamically weight the clothes data
during training to alleviate learning bias caused by the imbalance in the number of clothes samples (See Section 3.3 for details). Best viewed in color.

2.2. Curriculum learning for person re-ID

First, we start with clothes-level data scheduling. We randomly select
a set of clothes samples from each ID to build an initial training set.
In the following epochs, we decide whether to schedule new clothes
data for an ID before each training epoch by evaluating the silhouette
coeﬃcient of the ID. If the silhouette coeﬃcient of the ID is greater than
the threshold, we schedule a new set of clothes under that ID to form
the current epoch of training data, along with the previously scheduled
data. Our data scheduling involves a maximization strategy, that is, we
select the sample from the unselected clothes that is furthest from the
previous ones. When all the samples under an ID have been added to the
training data, it means that the data scheduling for that ID is ﬁnished,
see more details in Section 3.2.
During training, we performed data weighting to mitigate the learning bias caused by the imbalance in the number of samples for diﬀerent
clothes. We ﬁrst measure the ID accuracy of each piece of clothes by
multiplying its ID-level and clothes-level ID accuracies. Then, we assign
higher weights to samples with lower accuracy, allowing the model to
focus more on diﬃcult samples in a dynamic way based on the current
training status, see more details in Section 3.3.
The object function of our pipeline is written as follows:

Curriculum learning is ﬁrst proposed by [32]. It trains models using courses or meaningful sequences to imitate the process by which
humans learn from easy to diﬃcult tasks. Early studies on curriculum
learning often measure sample diﬃculty based on prior heuristic rules.
Self-paced learning [33] propose measuring the diﬃculty of samples by
training loss. Therefore, the model can adjust the sample curriculum
according to its own training state. Curriculum learning has recently
been shown to be eﬀective in solving domain adaptation person reidentiﬁcation problems. For example, SPCL [34] proposes to use loose
and tight clustering methods before each epoch training to retain the
reliable clusters , thereby providing more accurate pseudo-labels for the
target domain; SpML [35] deﬁnes the easy-to-diﬃcult sequence through
Style-uncertainty Augmentation, then learns the domain-invariant gradient from the easy domain, and ﬁnally adjusts the gradient through the
diﬃcult domain; CaCL [36] arranges the target domain domain adaptation curriculum sequence by measuring the distance between the target domain and the source domain on the camera. Diﬀerent from these
methods, we focus on clothes curriculum learning for each ID under
clothes changed scenario and achieve it through data scheduling and
data weighting.

arg min𝔼(𝜃) =
𝜃

3. Methodology

𝑁
∑

𝑤𝑥 𝑖 𝐿𝑥 𝑖 ,

(1)

𝑖=1

where 𝑁 indicates the number of samples, 𝜃 indicates the model parameters, 𝐿𝑥𝑖 indicates the loss function of the SC re-ID method at sample
𝑥𝑖 , and 𝑤𝑥𝑖 indicates the weight of the sample 𝑥𝑖 . We summarize in
Algorithm 1 an overall training process of our pipeline.

In this section, we ﬁrst give an overview of our proposed curriculum learning, followed by detailed descriptions of its two key designs:
clothes-level data scheduling and data weighting.

3.2. Data scheduling
3.1. Overview
The SC re-ID methods do not include additional processing for
clothes changed within same ID. If data with changed clothes is fed
in randomly, multiple samples of diﬀerent clothes associated with the

As shown in Fig. 4, our curriculum learning includes clothes-level
data scheduling and data weighting.
3

Pattern Recognition 179 (2026) 113509

Y. Xiao et al.

fore each epoch training. The formula of silhouette coeﬃcient is written
as follows:
𝑏(𝑥𝑖 ) − 𝑎(𝑥𝑖 )
𝑆𝐶(𝑥𝑖 ) =
, −1 ≤ 𝑆𝐶(𝑥𝑖 ) ≤ 1,
(3)
max{𝑎(𝑥𝑖 ), 𝑏(𝑥𝑖 )}

Algorithm 1 Training.
Input: The training dataset  = {(𝑥𝑖 , 𝑦𝑖 , 𝑧𝑖 )𝑁
}, Model 𝜑(⋅) with param𝑖=1
eter 𝜃, Threshold 𝑘
Output: Model Parameter 𝜃
1: Initial weights 𝑤 = 1.0
2: Initial training data 𝑡𝑟𝑎𝑖𝑛 by Eq. (2)
3: 𝑢𝑛𝑠𝑒𝑙 =  − 𝑡𝑟𝑎𝑖𝑛
4: for 𝑒𝑝𝑜𝑐ℎ = 0 to epochs do
5: if 𝑒𝑝𝑜𝑐ℎ > 1 ∧ 𝑢𝑛𝑠𝑒𝑙 ≠ ∅ then
6: Compute 𝑆𝑦 by Eq. (3) and Eq. (4)
7: if 𝑆𝑦 > 𝑘 then
8: 𝑛𝑒𝑤 = 𝑆𝑐ℎ𝑒𝑑𝑢𝑙𝑒(𝑡𝑟𝑎𝑖𝑛 , 𝑢𝑛𝑠𝑒𝑙 )
9: Update 𝑡𝑟𝑎𝑖𝑛 : 𝑡𝑟𝑎𝑖𝑛 ← 𝑡𝑟𝑎𝑖𝑛 ∪ 𝑛𝑒𝑤
10: Update 𝑢𝑛𝑠𝑒𝑙 : 𝑢𝑛𝑠𝑒𝑙 ← 𝑢𝑛𝑠𝑒𝑙 − 𝑛𝑒𝑤
11: end if
12: end if
13: Optimization 𝜃 with 𝑡𝑟𝑎𝑖𝑛 by Eq. (1)
14: update 𝑤 by Eq. (6), Eq. (7) and Eq. (8)
15: end for
16: return Model Parameter 𝜃

where 𝑆𝐶(𝑥𝑖 ) is the silhouette coeﬃcient for a given sample point 𝑥𝑖 ;
𝑎(𝑥𝑖 ) represents the average distance between 𝑥𝑖 and all other samples
within the same cluster (measuring how closely 𝑥𝑖 ﬁts within its cluster);
𝑏(𝑥𝑖 ) is the average distance from 𝑥𝑖 to points in the nearest neighboring
cluster (measuring how far 𝑥𝑖 is from other clusters). When 𝑏(𝑥𝑖 ) > 𝑎(𝑥𝑖 ),
it indicates that the sample point is closer to its own cluster, resulting
in a positive silhouette coeﬃcient. We calculate the mean 𝑆𝑦 of the silhouette coeﬃcients for all samples within identity 𝑦. The formula of 𝑆𝑦
is written as follows:
𝑁

𝑆𝑦 =

𝑦
1 ∑
𝑆𝐶(𝑥𝑖 ),
𝑁𝑦 𝑖=1

where 𝑁𝑦 represents the number of all samples labeled 𝑦 in the 𝑡𝑟𝑎𝑖𝑛 . 𝑆𝑦
evaluates the overall silhouette coeﬃcient of ID 𝑦, that is, the intra-class
compactness and inter-class separability of ID 𝑦. Fig. 5 illustrates the
diﬀerence in silhouette coeﬃcients for IDs with varying inter-class and
intra-class distances. The larger the silhouette coeﬃcient 𝑆𝑦 , the better
the model ﬁts the sample with ID 𝑦 in current training data, otherwise
it is not good.
Training pool updating. We control data scheduling by a threshold 𝑘. Speciﬁcally, if 𝑆𝑦 exceeds the threshold 𝑘, the current model is
deemed capable of recognizing ID 𝑦 suﬃciently to allow the addition of
new clothes curriculum data 𝑛𝑒𝑤 to it.
𝑛𝑒𝑤 = 𝑆𝑐ℎ𝑒𝑑𝑢𝑙𝑒(𝑡𝑟𝑎𝑖𝑛 , 𝑢𝑛𝑠𝑒𝑙
𝑦,𝑧 ),

(5)

where 𝑆𝑐ℎ𝑒𝑑𝑢𝑙𝑒(𝑡𝑟𝑎𝑖𝑛 , 𝑢𝑛𝑠𝑒𝑙
𝑦,𝑧 ) indicates the scheduling function, which
schedules new clothes data based on the current training data 𝑡𝑟𝑎𝑖𝑛 and
the unselected data 𝑢𝑛𝑠𝑒𝑙
𝑦,𝑧 . Our scheduling function involves maximization strategy. It entails ﬁrst calculating the Euclidean distance between
the feature center of the ID in the current 𝑡𝑟𝑎𝑖𝑛 and the feature center of
the clothes sample from the 𝑢𝑛𝑠𝑒𝑙
𝑦,𝑧 associated with the same ID. Then, all
samples under the clothes sample center with the furthest distance are
selected as 𝑛𝑒𝑤 . Fig. 4 illustrates the data scheduling process following
the maximization strategy.
After scheduled new clothes data for IDs that meet the threshold, the
training data for current epoch updated by 𝑡𝑟𝑎𝑖𝑛 ← 𝑡𝑟𝑎𝑖𝑛 ∪ 𝑛𝑒𝑤 , and
𝑢𝑛𝑠𝑒𝑙
𝑢𝑛𝑠𝑒𝑙
the unselected data updated by 𝑢𝑛𝑠𝑒𝑙
𝑦,𝑧 ← 𝑦,𝑧 − 𝑛𝑒𝑤 . If 𝑦,𝑧 = ∅, the
whole data scheduling ends, which means all training data are added to
the training.

Fig. 5. The diﬀerence in silhouette coeﬃcients for IDs with varying inter-class
and intra-class distances. Diﬀerent colors represent diﬀerent IDs, and the dashed
lines are classiﬁcation boundaries. The larger the silhouette coeﬃcient of the ID,
the better its intra-class compactness and inter-class separation, such as 𝑆𝐴 > 𝑆𝐵 .
Best viewed in color.

same ID may be input simultaneously, which raises a challenge to the
optimization during the initial training of model as the intra-ID variation is too high. Inspired by the gradual cognition process of humans,
we believe that the SC re-ID model’s learning of the clothes changed ID
also needs to follow the gradual principles, i.e. start the training with
a fairly low level of intra-ID variation. To this end, we propose to design a proper curriculum schedule for the SC re-ID method under clothes
changed.
Initialization. Formally, given a clothes changed dataset  =
{(𝑥𝑖 , 𝑦𝑖 , 𝑧𝑖 )𝑁
}, 𝑥𝑖 is a pedestrian sample, 𝑦𝑖 is an identity label, and 𝑧𝑖
𝑖=1
is a clothes label. The set of all samples under ID label 𝑦 and clothes
label 𝑧 is expressed as {𝑦,𝑧 , ∀𝑧 ∈ 𝑦 }, where 𝑦 represents the set of
all clothes 𝑧 associated with identity 𝑦. Then, we randomly select a
clothes set under identity 𝑦 to form the initial curriculum training data
𝑡𝑟𝑎𝑖𝑛 = 𝑖𝑛𝑖𝑡 , which provides a simple initial cognition of the training
ID for the model:
𝑖𝑛𝑖𝑡 = 𝑅𝑎𝑛𝑑𝑜𝑚𝑆𝑒𝑙𝑒𝑐𝑡({𝑦,𝑧 |𝑧 ∈ 𝑦 }),

(4)

3.3. Data weighting
As previously shown in Fig. 3, there are imbalance in the number
of clothes samples, which easily cause the SC re-ID model to focus on
the clothes samples with a large number, thus leading to biased feature
learning. To this end, we introduce a dynamically weighting for each
clothes sample during training. We ﬁrst estimate the diﬃculty of each
sample based on its classiﬁcation accuracy, as the fewer samples available for training, the harder it is for the model to classify the sample
accurately. Considering the diﬀerent diﬃculty of the model’s recognition between the whole ID and the ID with speciﬁc clothes, we calculate
the ID level and clothes level classiﬁcation accuracy of the ID in current
training data based on the ID and clothes respectively. Let 𝑦̂ = 𝜑(𝑥, 𝜃) be
the identity label predicted of 𝑥 by the model 𝜑(⋅). The accuracy calculation at the ID-level can be written as:

(2)

𝑁

𝑄𝑦 =

where 𝑅𝑎𝑛𝑑𝑜𝑚𝑆𝑒𝑙𝑒𝑐𝑡 indicates random selection. The unselected data
can be expressed as 𝑢𝑛𝑠𝑒𝑙
𝑦,𝑧 = 𝑦,𝑧 − 𝑡𝑟𝑎𝑖𝑛 .
Status assessment. In order to evaluate the recognition of the
model to the ID, we employ the silhouette coeﬃcient [25] as indictor to
decide whether to add new clothes curriculum data to the each ID be-

𝑦
1 ∑
1(𝑦̂𝑖 = 𝑦𝑖 ),
𝑁𝑦 𝑖=1

(6)

where 𝑁𝑦 indicates the number of the sample with identity label 𝑦, and
1(𝑦̂𝑖 = 𝑦𝑖 ) indicates indicator function. The value is 1 when 𝑦̂𝑖 = 𝑦𝑖 , otherwise it is 0. The accuracy calculation of the ID with the speciﬁc clothes
4

Pattern Recognition 179 (2026) 113509

Y. Xiao et al.

Table 1
Comparison with state-of-the-art methods on PRCC. Bold indicates the best results.

𝑧 can be written as:
𝑁𝑦,𝑧

𝑄𝑦,𝑧 =

1 ∑
1(𝑦̂𝑖 = 𝑦𝑖 ),
𝑁𝑦,𝑧 𝑖=1

(7)

Methods

where 𝑁𝑦,𝑧 indicates the number of the sample with identity label 𝑦 and
clothes label 𝑧. Then, we take the product of the ID-level ID accuracy, 𝑄𝑦 ,
and the clothes-level ID accuracy, 𝑄𝑦,𝑧 , as the accuracy of each clothes
under the ID, and give higher weights to the clothes with low accuracy.
The weights formula of sample 𝑥 with identity label 𝑦 and clothes label
𝑧 can be written as follows:
𝑤𝑥 = 1 − 𝑄𝑦 × 𝑄𝑦,𝑧 .

Extra info

PRCC
CC

(8)

Our weights are dynamically adjusted based on the model’s classiﬁcation accuracy of samples, which allows the model to dynamically balance the learning bias caused by the number of clothes samples.
4. Experiments
We combine our curriculum learning with the existing SC re-ID
method, CLIP-ReID, to compete with state-of-the-art CC re-ID methods.
Finally, we show a series of ablation studies. Let’s start from experimental settings.

SC

Rank-1

mAP

Rank-1

mAP

3DSL [15]
FSAM [12]
RCSANet [40]
GI-ReID [13]
CAL [21]
AIM [22]
3DInvarReID [16]
SCNet [20]
DCR-ReID [18]
IRM(STL) [42]
CCFA [23]
CLIP3DReID [43]
MCSC [41]

3D shape
Silhouette
–
Gait
–
–
3D shape
Parsing
Parsing
–
–
3D shape
–

51.3
54.5
50.2
37.6
55.2
57.9
57.2
61.3
57.2
48.1
61.2
60.6
57.8

–
–
48.6
–
55.8
58.3
56.5
59.9
57.4
46.0
58.4
59.3
57.3

–
98.8
100
–
100
100
–
100
100
–
99.6
–
99.8

–
–
97.2
–
99.8
99.9
–
97.8
99.7
–
98.7
–
99.8

Ours

–

63.8

60.0

100

99.5

Table 2
Comparison with state-of-the-art methods on LTCC. Bold indicates the
best results. DIFFER† reports the performance of the same architecture
as ours.

4.1. Experimental settings
Datasets. We conduct our experiments on PRCC [11], LTCC [14],
VC-Clothes [37], LaST [38], and DeepChange [39]. PRCC consists of
33,698 images captured by 3 cameras, including 221 identities, each
with two diﬀerent clothes. The same person wears the same clothes under cameras A and B, but diﬀerent clothes under camera C. The dataset
is divided into a training set consisting of 150 identities and a testing set
consisting of 71 identities. LTCC includes a total of 17,119 images from
152 identities. Each identity appeared in front of at least two cameras,
and the number of costumes ranged from 2 to 14. A total of 77 identities are selected for training, consisting of 46 identities with clothes
variations and 31 identities with consistent clothes. Another 75 identities are used for testing, including 45 clothes-changed identities and
30 clothes-consistent identities. VC-Clothes is a virtual dataset synthesized by GTA5. It contains 512 identities of 19,060 images in 4 diﬀerent
scenes. Each identity has 1 to 3 suits of clothes. LaST contains 10,862
identities and 228,156 images. It consists of a training set with 70,923
images for 5000 identities, an evaluation set with 20,584 images for
56 identities, and a test set with 133,214 images for 5803 identities.
DeepChange contains 178,407 images of 1121 identities, recorded by
a real-world surveillance system using 17 outdoor cameras of varying
resolutions. All identities are captured by at least two cameras, with the
majority captured by 2 to 6 cameras.
Training details. We reproduce the top SC re-ID methods CLIPReID (ViT-B/16) [24] on clothes changed datasets as baseline following
the default experimental settings, and apply our curriculum learning on
it. Note that apart from the data adjustments involved in our curriculum
learning, we do not make any changes to the source model and hyperparameters.
Evaluation protocols. Following the established evaluation protocol from previous works [18,21,22,40]: for PRCC, we conduct the
comparison under Same-Clothes (SC) and Cloth-Changed (CC) settings,
where SC means that the same identity always wears the same clothes
in the query and gallery images, and CC means that the same identity
wears diﬀerent clothes in the query and gallery images; for LTCC and
VC-Clothes, we compare our method with the state-of-the-art methods
under CC and General (GEN) settings, where the GEN means that for
each identity, some instances in the gallery wear the same clothes with
the query, while others may not; for LaST and DeepChange, as with
most methods, the comparison is performed under GEN. We employ Cumulative Matching Characteristic Rank-1 accuracy and Mean Average

Methods

Extra info

LTCC
CC

3DSL [15]
FSAM [12]
GI-ReID [13]
CAL [21]
AIM [22]
3DInvarReID [16]
SCNet [20]
DCR-ReID [18]
IRM(STL) [42]
CCFA [23]
CLIP3DReID [43]
MCSC [41]
DIFFER† [44]
Ours

3D shape
Silhouette
Gait
–
–
3D shape
Parsing
Parsing
–
–
3D shape
–
Text
–

GEN

Rank-1

mAP

Rank-1

mAP

31.2
38.5
28.9
40.1
40.6
40.9
47.5
41.1
–
45.3
42.1
42.2
36.5
46.2

14.8
16.2
14.2
18.0
19.1
18.9
25.5
20.4
–
22.1
21.7
19.4
15.6
25.3

–
73.2
73.6
74.2
76.3
73.2
76.3
76.1
66.7
75.8
–
73.9
72.6
77.5

–
35.4
36.1
40.8
41.1
35.4
43.6
42.3
46.7
42.5
–
40.2
35.2
47.8

Precision (mAP) for evaluation. Note that consistent with most methods [12,13,15,21,41], we report the results from a subset of cameras 3
and 4 on VC-Clothes as the performance for the CC setting.
4.2. Comparison with state-of-the-art methods
We compare our method with state-of-the-art CC re-ID methods on
PRCC, LTCC, VC-Clothes, LaST and DeepChange. As can be seen in Table 1, 2, 3, 4, our method outperforms the existing top CC re-ID methods on most metrics across the ﬁve datasets, achieving state-of-the-art
performance on PRCC, VC-Clothes, LaST and DeepChange. It is worth
mentioning that we do not employ additional clothes processing, such
as disentangling clothes information or introducing additional clothesirrelevant information. In detail, on PRCC, our method outperforms the
existing top CC re-ID method, SC-Net [20], by 2.5 pp w.r.t. Rank-1 under CC. Due to the simplicity of the SC scenario on PRCC, the performance reached saturation.Our method achieves 100% Rank-1 accuracy,
which is on par with the performance of previous methods. On challenge
dataset, LTCC, our method achieves competitive performance with the
top CC re-ID method, SC-Net, under CC; however, our method outperforms it by 4.2 pp w.r.t. mAP under GEN. On virtual dataset, VC-Clothes,
5


codex
说明：我按要求用 `pdftotext -f 1 -l 8` 读了六篇。`Affinity Harmonization` 和 `MCBL` 的方法超过第 8 页，我额外补读了后续方法页，避免机制链条缺失。

**1. Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning**

1. 触发观察：作者先发现传统同衣 ReID 方法不是完全不能做换衣。以 CLIP-ReID 为代表的同衣方法直接迁到换衣场景已经有竞争力，真正卡住的是训练初期同一个身份同时出现多套衣服，类内变化太大，模型过早被打乱。
2. 重定义动作：大家以为换衣 ReID 必须加衣服无关线索、解析、轮廓、骨架或特征解耦；作者改成说，问题其实是训练课程不合理。同衣方法的能力没有被释放，关键词是 `unleash potential`、`clothes-level curriculum`、`gradual cognition`、`intra-ID variation`。
3. 机制怎么长出来：这个重定义几乎直接推出机制。既然问题是过早暴露高类内变化，就先让每个身份只用一套衣服训练；等 silhouette coefficient 说明该身份已经可分，再加入最不像当前衣服的新衣服；再用衣服级准确率给难衣服更高权重，补样本数量不均衡。
4. 证据闭环：关键不是只报涨点，而是证明“换衣困难可以由训练顺序缓解”。他们做了数据调度和数据加权消融，比较随机、最小差异、最大差异三种加衣服策略，比较固定周期和 silhouette 自适应调度，还用 t-SNE 和 GradCAM 展示同身份多衣服逐渐聚合、注意力从衣服区域转到头部和轮廓。
5. reviewer 为什么买账：这篇卖的主要是视角，不是复杂模块。它把“换衣必须改模型”改成“先把训练组织对”，而且不改测试模型、不加额外输入，现实部署友好；这比再堆一个衣服去偏分支更容易讲清楚。

**2. Unsupervised Lifelong Person Re-Identification via Affinity Harmonization**

1. 触发观察：现有终身 ReID 多数靠旧样本回放和旧模型蒸馏来防遗忘，但这种做法偏向稳定性，容易限制新域学习。作者还指出，旧域样本的相机内类内方差会被新知识扰动，导致旧域识别能力下降。
2. 重定义动作：大家以为终身 ReID 的核心是“保旧知识”；作者改成“协调新旧域的亲和关系”。关键词是 `affinity harmonization`、`plasticity-stability balance`、`old domain expert`、`current domain expert`、`intra-class variance constraint`。
3. 机制怎么长出来：如果知识以样本间亲和关系存在，那么机制自然变成关系矩阵约束。ODAC 用旧域专家约束旧样本的关系矩阵不要漂；CDAG 训练一个当前域专家，引导主模型学习新域关系；OIVC 再用旧域相机原型约束旧样本类内方差。
4. 证据闭环：关键消融应当证明三件事：ODAC 保旧域，CDAG 提升新域学习，OIVC 抑制旧域类内方差漂移。它的闭环不是只看最终平均 mAP，而是看 seen domains、unseen domains、不同训练顺序，以及新旧性能是否同时改善。
5. reviewer 为什么买账：这篇是机制和视角一起卖。亲和关系作为终身 ReID 的知识单位，比普通参数蒸馏更贴合 ReID 的匹配任务；两个临时专家训练后丢弃，也避免了永久扩模型的质疑。

**3. Unsupervised person re-identification via camera-aware multi-level label refinement**

1. 触发观察：作者用两个很具体的现象开局。DBSCAN 的聚类数量在训练中波动并偏离真实身份数；t-SNE 显示同摄像头的不同身份容易聚在一起，而同一身份跨摄像头反而分散。
2. 重定义动作：大家以为无监督 ReID 的伪标签噪声只是聚类不准；作者改成说，噪声是由摄像头结构驱动的，而且同时存在于 cluster level 和 instance level。关键词是 `camera-aware`、`multi-level label refinement`、`dual-level intra-inter refinement`、`affinity-guided mutual refinement`。
3. 机制怎么长出来：机制和重定义绑定很紧。全局特征用于找跨摄像头近邻，把可能被摄像头分开的同身份拉近；局部特征不稳定，所以只在同摄像头内找上下半身共同可靠近邻；实例级再找互为 top-k 的跨摄像头 informative nodes，把 one-hot 自正样本改成软亲和正样本。
4. 证据闭环：真正支撑问题定义的是 Fig.1 的聚类波动和摄像头偏置可视化，再加 DIIR、AGMR、全局和局部分支的消融。尤其是 MSMT17 这种摄像头变化更强的数据上提升更大，能说明它不是普通 label smoothing，而是在处理 camera-induced noise。
5. reviewer 为什么买账：这篇卖的是结构化修伪标签的视角。单独“用 camera 信息”不新，CAP、ICE、CGMAL 都做过；它的新意在于把 camera bias 拆到全局、局部、实例三个层级，每层的邻居选择规则不同，逻辑比较完整。

**4. Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning**

1. 触发观察：作者抓了两个失败点。无监督 ReID 的特征把身份相关的人体信息和摄像头背景信息缠在一起，导致同身份跨摄像头距离大；同时，常见 patch 方法会破坏人体语义结构，STN 又缺少语义指导。
2. 重定义动作：大家以为主要矛盾是伪标签净化或 patch 设计；作者改成“先做语义感知的身份与摄像头解耦，再在身份相关特征上生成语义 patch”。关键词是 `semantic-aware disentanglement`、`identity-related features`、`camera-related features`、`diffusion over spatial transformer parameters`。
3. 机制怎么长出来：DAM 用人像 mask 做跨摄像头聚合，生成增强视图，把身份和摄像头背景拆开；MSC 约束源视图和增强视图的跨摄像头、同摄像头相似度分布一致；SSDM 不直接扩散整图，而是扩散 STN 的空间变换参数；SDC 用相邻去噪步的 patch 对比保持语义。
4. 证据闭环：这篇需要靠强消融来成立。DAM、MSC、SSDM、SDC 都要分别拿掉；还要展示距离分布被 MSC 拉齐、SSDM 生成的 patch 确实有语义，而不是随机局部块。参数分析里 patch 数、扩散步数和损失权重也在证明它不是随便堆扩散。
5. reviewer 为什么买账：这篇更偏机制稿。真正的新意不是“用扩散模型”，而是把扩散对象换成空间变换参数，避开整图生成的高成本，并把它嵌进无监督 ReID 的语义 patch 学习里。风险是模块多，必须靠消融证明不是堆料。

**5. Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement**

1. 触发观察：现有无监督可见光红外 ReID 主要盯着模态差异，但作者指出摄像头差异同样会造成身份拆裂和噪声标签。更具体地说，每个 camera-modality 单元的信息量不同，聚类数和可靠性不平衡，直接跨模态关联会放大错标。
2. 重定义动作：大家以为 USL-VI-ReID 是跨模态对齐问题；作者改成“层级域内的模态和摄像头平衡问题”。关键词是 `modality-camera balance`、`hierarchical domains`、`bottom-up learning`、`intra-camera`、`inter-camera`、`inter-modality`。
3. 机制怎么长出来：重定义直接推出三阶段训练。先 intra-camera 建立每个摄像头内的基本判别力，再 inter-camera 做同模态跨摄像头平衡，最后 inter-modality 做跨模态统一。MBLM 用样本到其他摄像头或模态 cluster centroid 的相似度筛掉不可靠标签；UPR 又把这些不可靠样本在同摄像头内重对齐，避免早期丢太多样本；HMCCL 在每个层级同时做实例级和簇级对比学习。
4. 证据闭环：闭环重点应该是逐阶段和逐模块消融。只做模态对齐、只过滤不重对齐、只 cluster-wise 或只 instance-wise，都要输给完整 MCBL；如果 SYSU-MM01 这种多摄像头数据上收益更明显，就能支撑“camera-modality balance”这个定义。
5. reviewer 为什么买账：这篇卖的是问题拆分方式。它不是再提出一个跨模态生成或图匹配模块，而是把训练流程改成从摄像头内到模态间的层级标签修正，和无监督场景里的伪标签噪声天然绑定。

**6. Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification**

1. 触发观察：作者看到图文局部对齐在极端干扰下会失效。雨雪雾、强光、遮挡和相似外观物体会造成 visual hallucination，模型以为某个局部位置有对应文本，实际是伪真实位置，导致局部错配。
2. 重定义动作：大家以为文本到图像 ReID 要做更细粒度 local alignment；作者改成“先判断局部对齐是否 well-posed”。关键词是 `local well-posedness`、`local misalignment`、`localization bias`、`pseudo-real position`、`mask-based modeling`。
3. 机制怎么长出来：如果问题是局部信息是否能被另一模态可靠恢复，那么 masked reconstruction 很自然。TEM 遮文本实体词，用图像和上下文重建；TAM 遮属性词，检验颜色、衣物等属性是否被图像支持；IAM 遮图像区域，用文本提示重建外观。三者分别覆盖实体、属性、外观。
4. 证据闭环：消融 TEM、TAM、IAM 是基本证据，检索可视化说明局部错配减少。但从“well-posedness”这个概念看，最强证据应该是极端天气、强遮挡、颜色污染子集上的鲁棒性分析；论文主要用三数据集和定性图支撑，闭环不算特别强。
5. reviewer 为什么买账：这篇主要卖视角和命名，机制本身接近图文 masked modeling。ICASSP 这种篇幅下，它把一个常见操作包装成“局部对齐可解性检查”，再给出实体、属性、外观三阶段，故事完整，所以能买账。

总体看，这批论文共同套路很明显：先抓一个具体失败现象，不说“我要加模块”，而说“大家把问题定义错了”。然后把新定义落成一个几乎顺理成章的训练组织、标签修正或关系约束，再用消融和可视化证明“这个定义解释了涨点”。真正值得学的是这个顺序：先重写问题，再让机制像答案一样长出来。
tokens used
120,413
说明：我按要求用 `pdftotext -f 1 -l 8` 读了六篇。`Affinity Harmonization` 和 `MCBL` 的方法超过第 8 页，我额外补读了后续方法页，避免机制链条缺失。

**1. Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning**

1. 触发观察：作者先发现传统同衣 ReID 方法不是完全不能做换衣。以 CLIP-ReID 为代表的同衣方法直接迁到换衣场景已经有竞争力，真正卡住的是训练初期同一个身份同时出现多套衣服，类内变化太大，模型过早被打乱。
2. 重定义动作：大家以为换衣 ReID 必须加衣服无关线索、解析、轮廓、骨架或特征解耦；作者改成说，问题其实是训练课程不合理。同衣方法的能力没有被释放，关键词是 `unleash potential`、`clothes-level curriculum`、`gradual cognition`、`intra-ID variation`。
3. 机制怎么长出来：这个重定义几乎直接推出机制。既然问题是过早暴露高类内变化，就先让每个身份只用一套衣服训练；等 silhouette coefficient 说明该身份已经可分，再加入最不像当前衣服的新衣服；再用衣服级准确率给难衣服更高权重，补样本数量不均衡。
4. 证据闭环：关键不是只报涨点，而是证明“换衣困难可以由训练顺序缓解”。他们做了数据调度和数据加权消融，比较随机、最小差异、最大差异三种加衣服策略，比较固定周期和 silhouette 自适应调度，还用 t-SNE 和 GradCAM 展示同身份多衣服逐渐聚合、注意力从衣服区域转到头部和轮廓。
5. reviewer 为什么买账：这篇卖的主要是视角，不是复杂模块。它把“换衣必须改模型”改成“先把训练组织对”，而且不改测试模型、不加额外输入，现实部署友好；这比再堆一个衣服去偏分支更容易讲清楚。

**2. Unsupervised Lifelong Person Re-Identification via Affinity Harmonization**

1. 触发观察：现有终身 ReID 多数靠旧样本回放和旧模型蒸馏来防遗忘，但这种做法偏向稳定性，容易限制新域学习。作者还指出，旧域样本的相机内类内方差会被新知识扰动，导致旧域识别能力下降。
2. 重定义动作：大家以为终身 ReID 的核心是“保旧知识”；作者改成“协调新旧域的亲和关系”。关键词是 `affinity harmonization`、`plasticity-stability balance`、`old domain expert`、`current domain expert`、`intra-class variance constraint`。
3. 机制怎么长出来：如果知识以样本间亲和关系存在，那么机制自然变成关系矩阵约束。ODAC 用旧域专家约束旧样本的关系矩阵不要漂；CDAG 训练一个当前域专家，引导主模型学习新域关系；OIVC 再用旧域相机原型约束旧样本类内方差。
4. 证据闭环：关键消融应当证明三件事：ODAC 保旧域，CDAG 提升新域学习，OIVC 抑制旧域类内方差漂移。它的闭环不是只看最终平均 mAP，而是看 seen domains、unseen domains、不同训练顺序，以及新旧性能是否同时改善。
5. reviewer 为什么买账：这篇是机制和视角一起卖。亲和关系作为终身 ReID 的知识单位，比普通参数蒸馏更贴合 ReID 的匹配任务；两个临时专家训练后丢弃，也避免了永久扩模型的质疑。

**3. Unsupervised person re-identification via camera-aware multi-level label refinement**

1. 触发观察：作者用两个很具体的现象开局。DBSCAN 的聚类数量在训练中波动并偏离真实身份数；t-SNE 显示同摄像头的不同身份容易聚在一起，而同一身份跨摄像头反而分散。
2. 重定义动作：大家以为无监督 ReID 的伪标签噪声只是聚类不准；作者改成说，噪声是由摄像头结构驱动的，而且同时存在于 cluster level 和 instance level。关键词是 `camera-aware`、`multi-level label refinement`、`dual-level intra-inter refinement`、`affinity-guided mutual refinement`。
3. 机制怎么长出来：机制和重定义绑定很紧。全局特征用于找跨摄像头近邻，把可能被摄像头分开的同身份拉近；局部特征不稳定，所以只在同摄像头内找上下半身共同可靠近邻；实例级再找互为 top-k 的跨摄像头 informative nodes，把 one-hot 自正样本改成软亲和正样本。
4. 证据闭环：真正支撑问题定义的是 Fig.1 的聚类波动和摄像头偏置可视化，再加 DIIR、AGMR、全局和局部分支的消融。尤其是 MSMT17 这种摄像头变化更强的数据上提升更大，能说明它不是普通 label smoothing，而是在处理 camera-induced noise。
5. reviewer 为什么买账：这篇卖的是结构化修伪标签的视角。单独“用 camera 信息”不新，CAP、ICE、CGMAL 都做过；它的新意在于把 camera bias 拆到全局、局部、实例三个层级，每层的邻居选择规则不同，逻辑比较完整。

**4. Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning**

1. 触发观察：作者抓了两个失败点。无监督 ReID 的特征把身份相关的人体信息和摄像头背景信息缠在一起，导致同身份跨摄像头距离大；同时，常见 patch 方法会破坏人体语义结构，STN 又缺少语义指导。
2. 重定义动作：大家以为主要矛盾是伪标签净化或 patch 设计；作者改成“先做语义感知的身份与摄像头解耦，再在身份相关特征上生成语义 patch”。关键词是 `semantic-aware disentanglement`、`identity-related features`、`camera-related features`、`diffusion over spatial transformer parameters`。
3. 机制怎么长出来：DAM 用人像 mask 做跨摄像头聚合，生成增强视图，把身份和摄像头背景拆开；MSC 约束源视图和增强视图的跨摄像头、同摄像头相似度分布一致；SSDM 不直接扩散整图，而是扩散 STN 的空间变换参数；SDC 用相邻去噪步的 patch 对比保持语义。
4. 证据闭环：这篇需要靠强消融来成立。DAM、MSC、SSDM、SDC 都要分别拿掉；还要展示距离分布被 MSC 拉齐、SSDM 生成的 patch 确实有语义，而不是随机局部块。参数分析里 patch 数、扩散步数和损失权重也在证明它不是随便堆扩散。
5. reviewer 为什么买账：这篇更偏机制稿。真正的新意不是“用扩散模型”，而是把扩散对象换成空间变换参数，避开整图生成的高成本，并把它嵌进无监督 ReID 的语义 patch 学习里。风险是模块多，必须靠消融证明不是堆料。

**5. Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement**

1. 触发观察：现有无监督可见光红外 ReID 主要盯着模态差异，但作者指出摄像头差异同样会造成身份拆裂和噪声标签。更具体地说，每个 camera-modality 单元的信息量不同，聚类数和可靠性不平衡，直接跨模态关联会放大错标。
2. 重定义动作：大家以为 USL-VI-ReID 是跨模态对齐问题；作者改成“层级域内的模态和摄像头平衡问题”。关键词是 `modality-camera balance`、`hierarchical domains`、`bottom-up learning`、`intra-camera`、`inter-camera`、`inter-modality`。
3. 机制怎么长出来：重定义直接推出三阶段训练。先 intra-camera 建立每个摄像头内的基本判别力，再 inter-camera 做同模态跨摄像头平衡，最后 inter-modality 做跨模态统一。MBLM 用样本到其他摄像头或模态 cluster centroid 的相似度筛掉不可靠标签；UPR 又把这些不可靠样本在同摄像头内重对齐，避免早期丢太多样本；HMCCL 在每个层级同时做实例级和簇级对比学习。
4. 证据闭环：闭环重点应该是逐阶段和逐模块消融。只做模态对齐、只过滤不重对齐、只 cluster-wise 或只 instance-wise，都要输给完整 MCBL；如果 SYSU-MM01 这种多摄像头数据上收益更明显，就能支撑“camera-modality balance”这个定义。
5. reviewer 为什么买账：这篇卖的是问题拆分方式。它不是再提出一个跨模态生成或图匹配模块，而是把训练流程改成从摄像头内到模态间的层级标签修正，和无监督场景里的伪标签噪声天然绑定。

**6. Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification**

1. 触发观察：作者看到图文局部对齐在极端干扰下会失效。雨雪雾、强光、遮挡和相似外观物体会造成 visual hallucination，模型以为某个局部位置有对应文本，实际是伪真实位置，导致局部错配。
2. 重定义动作：大家以为文本到图像 ReID 要做更细粒度 local alignment；作者改成“先判断局部对齐是否 well-posed”。关键词是 `local well-posedness`、`local misalignment`、`localization bias`、`pseudo-real position`、`mask-based modeling`。
3. 机制怎么长出来：如果问题是局部信息是否能被另一模态可靠恢复，那么 masked reconstruction 很自然。TEM 遮文本实体词，用图像和上下文重建；TAM 遮属性词，检验颜色、衣物等属性是否被图像支持；IAM 遮图像区域，用文本提示重建外观。三者分别覆盖实体、属性、外观。
4. 证据闭环：消融 TEM、TAM、IAM 是基本证据，检索可视化说明局部错配减少。但从“well-posedness”这个概念看，最强证据应该是极端天气、强遮挡、颜色污染子集上的鲁棒性分析；论文主要用三数据集和定性图支撑，闭环不算特别强。
5. reviewer 为什么买账：这篇主要卖视角和命名，机制本身接近图文 masked modeling。ICASSP 这种篇幅下，它把一个常见操作包装成“局部对齐可解性检查”，再给出实体、属性、外观三阶段，故事完整，所以能买账。

总体看，这批论文共同套路很明显：先抓一个具体失败现象，不说“我要加模块”，而说“大家把问题定义错了”。然后把新定义落成一个几乎顺理成章的训练组织、标签修正或关系约束，再用消融和可视化证明“这个定义解释了涨点”。真正值得学的是这个顺序：先重写问题，再让机制像答案一样长出来。
