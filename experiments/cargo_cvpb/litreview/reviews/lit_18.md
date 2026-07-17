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
session id: 019ef71f-ce93-7d91-a3f4-e3fd8c092ab2
--------
user
你是 ReID 论文创新挖掘员 18/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(18/20):
Two-stage knowledge distillation for visible-infrared person re-identification.pdf
UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf
Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf
Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf
Unsupervised person re-identification via camera-aware multi-level label refinement.pdf
Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf
Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf
Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我先按项目规则读目标和状态文件，再只用 `pdftotext -l 3` 抽取这批论文的前三页内容。随后会把每篇论文压缩成创新反推要点，而不是做普通综述。
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
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
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

exec
/bin/zsh -lc "sed -n '1,260p' /Users/abdslm/Desktop/reid-paper/CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.622.11653/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/Desktop/reid-paper/experiments/STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
---
name: "pdf"
description: "Read, create, inspect, render, and verify PDF files where visual layout matters. Use Poppler rendering plus Python tools such as reportlab, pdfplumber, and pypdf for generation and extraction."
---

# PDF Skill

## When To Use

- Read or review PDF content where layout and visuals matter.
- Create PDFs programmatically with reliable formatting.
- Validate final rendering before delivery.

## Workflow

1. Prefer visual review: render PDF pages to PNGs and inspect them.
   - Use `pdftoppm` from the bundled runtime or system Poppler when available.
   - If unavailable, install Poppler or ask the user to review the output locally.
2. Use `reportlab` to generate PDFs when creating new documents.
3. Use `pdfplumber` or `pypdf` for text extraction and quick checks; do not rely on text extraction for layout fidelity.
4. After each meaningful update, re-render pages and verify alignment, spacing, and legibility.

## Temp And Output Conventions

- Use `tmp/pdfs/` for intermediate files; delete them when done.
- Write final artifacts under `output/pdf/` when working in this repo.
- Keep filenames stable and descriptive.

## Dependencies

Prefer the Codex bundled workspace/runtime dependencies when available. The primary runtime is expected to include:

- Python packages: `reportlab`, `pdfplumber`, `pypdf`
- Rendering tools: `pdftoppm` and `pdfinfo` from Poppler

If a dependency is missing, install only what is needed.

Python packages:

```bash
uv pip install reportlab pdfplumber pypdf
```

If `uv` is unavailable:

```bash
python3 -m pip install reportlab pdfplumber pypdf
```

System tools for rendering:

```bash
# macOS (Homebrew)
brew install poppler

# Ubuntu/Debian
sudo apt-get install -y poppler-utils
```

If installation is not possible in this environment, tell the user which dependency is missing and how to install it locally.

## Environment

No required environment variables.

## Rendering Command

```bash
pdftoppm -png "$INPUT_PDF" "$OUTPUT_PREFIX"
```

## Quality Expectations

- Maintain polished visual design: consistent typography, spacing, margins, and section hierarchy.
- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
- Charts, tables, and images must be sharp, aligned, and clearly labeled.
- Use ASCII hyphens only. Avoid U+2011 and other Unicode dashes.
- Citations and references must be human-readable; never leave tool tokens or placeholder strings.

## Final Checks

- Do not deliver until the latest PNG inspection shows zero visual or formatting defects.
- Confirm headers, footers, page numbering, and section transitions look polished.
- Keep intermediate files organized or remove them after final approval.

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
我已经确认当前目标还是“读文献做差距分析，先不要开实验”。下面开始逐篇抽前三页，重点抓摘要、引言里的问题定义、机制和证据套路。
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
/bin/zsh -lc "pdftotext -l 3 'Uniform Light Transformer for Person Re-identification under Complex Illumination.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Two-stage knowledge distillation for visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


codex
前三篇已经读完。一个明显模式是，B 类论文常把“单步对齐太硬”或“统一变换器在某个频段失真”说成核心问题，然后用可视化或频谱/距离分解证明这个问题真的存在。
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
/bin/zsh -lc "pdftotext -l 3 'Unsupervised Lifelong Person Re-Identification via Affinity Harmonization.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Unsupervised person re-identification via camera-aware multi-level label refinement.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


codex
第二组读完。这里出现两个对我们有用的套路：一是“训练组织本身就是方法”，比如按衣服难度或域专家关系来安排学习；二是把相机、域、衣服这些干扰因素变成显式的关系约束，而不是只加一个注意力模块。
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
/bin/zsh -lc "pdftotext -l 3 'Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


exec
/bin/zsh -lc "pdftotext -l 3 'Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


codex
基于每篇前三页。下面只抽对“怎么反推新方向”有用的部分。

**逐篇提取**

1. `Two-stage knowledge distillation for visible-infrared person re-identification`
创新类型：新机制。把 VI-ReID 从“一步跨模态硬对齐”改成“先收紧同模态，再跨模态互蒸馏”。
为什么能发：指出已有方法忽略同模态内部差异，还依赖完整标注。证据链是特征距离可视化、弱监督设置、SYSU-MM01 和 RegDB 对照。
Story 套路：跨模态不是一上来就对齐，而是从简单到困难，把多对多问题先压成原型级关系。

2. `UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges`
创新类型：新数据设定和问题重定义。
为什么能发：UAV ReID 不是普通 ReID 的小分支，而是动态背景、运动模糊、低清、视角高度变化、机载算力约束的组合问题。前三页强调公开数据和方法都很少。
Story 套路：地面摄像头 ReID 的假设在航拍里系统性失效，所以需要专门机制，不是直接迁移模型。

3. `Uniform Light Transformer for Person Re-identification under Complex Illumination`
创新类型：问题重定义加新机制。
为什么能发：把复杂光照下的失败定位到统一光照转换器的低频建模不足，而不是泛泛说光照变化。证据链是频谱分解，低频差异随光照尺度增多而增大。
Story 套路：视觉上像修好了，但频域里身份相关的低频结构坏了，所以要约束频谱和低频特征一致性。

4. `Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning`
创新类型：训练机制和问题重定义。
为什么能发：不再做专门换衣模块，而是证明传统同衣 ReID 方法有潜力，只是训练初期同时面对过大同身份衣服差异。证据链是多个 SC 方法在 PRCC、LTCC、VC-Clothes 等数据集上通过衣服课程学习超过 CC 方法。
Story 套路：不是模型不会换衣，而是学习顺序太粗暴。先学一个衣服，再逐步增加最难衣服。

5. `Unsupervised Lifelong Person Re-Identification via Affinity Harmonization`
创新类型：新机制。
为什么能发：无监督终身 ReID 的核心不是只防遗忘，而是稳定性和可塑性的平衡。证据链是旧域专家约束旧关系，新域专家指导新关系，再用旧域相机原型限制类内方差。
Story 套路：保留旧知识不能靠冻结模型压住一切，而要同时维护旧关系和吸收新关系。

6. `Unsupervised person re-identification via camera-aware multi-level label refinement`
创新类型：新机制。
为什么能发：把无监督 ReID 的伪标签噪声拆成聚类噪声和相机偏置。证据链是 DBSCAN 簇数波动和 t-SNE 相机聚团现象，再做簇级和实例级双层精炼。
Story 套路：相机不是元数据，而是伪标签错误的结构来源，必须进入标签精炼过程。

7. `Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning`
创新类型：新机制。
为什么能发：把无监督 ReID 的问题说成身份特征和相机背景特征纠缠，以及普通切块破坏语义结构。机制是身份和相机特征解耦，再用扩散模型生成语义空间变换参数。
Story 套路：扩散模型不是用来生成整个人，而是用来在无标签条件下找到语义局部。

8. `Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement`
创新类型：新机制。
为什么能发：指出 USL-VI-ReID 不只是模态差异，还有模态和相机共同导致的层级伪标签噪声。证据链是模态相机不平衡导致聚类错误，再用筛选、重对齐和混合对比学习修正。
Story 套路：跨模态无监督不能只喊模态对齐，要在相机和模态两个层级上平衡伪标签。

9. `Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification`
创新类型：问题重定义加新机制。
为什么能发：把图文 ReID 的局部错配定义成“局部可解性”问题，极端天气、光照、遮挡会让文本和图像局部根本不对应。机制是实体、属性、外观三阶段掩码重建。
Story 套路：先判断局部对应是否可解，再做局部对齐，否则对齐会把模型带偏。

**这批论文给我们的启发**

最有价值的不是“加蒸馏、加扩散、加相机标签”，而是三个写法：把旧任务重新刻画成“局部是否可解”“学习顺序是否合理”“伪标签错误是否有结构”；先用可视化或统计证明这个结构确实存在；再把机制严格扣到这个结构上。

**强创新点候选**

1. 标题句：航拍地面 ReID 的核心不是视角差，而是很多身体局部在三维投影上本来不可对齐，模型应该先判断哪些局部可解，再学习身份。
挂载资产：aerial-ground 数据、SMPL 3D 几何、pose 热图门控、SOLIDER-Swin。
区别切开点：最像的是 local well-posedness 那篇，但它做图文局部掩码重建；我们做图像到图像的三维可见性、投影面积、视角夹角和低清约束。和 UAV survey 的区别是它只总结挑战，我们给出可计算的“局部可解性”机制。
最小首验：在 CARGO 或 AG-ReID.v2 上用现有 SMPL 或姿态估计得到每个身体部位的可见性、投影面积和视角差，检查这个分数是否能预测 SOLIDER 的失败样本。再做冻结特征的部位重加权重排。若相关性不明显，且重排 mAP 不到 0.4、rank1 不到 0.5，就杀掉。

2. 标题句：跨航拍地面训练不该一开始硬吃极端视角，而要按三维可观测性从容易样本逐步推进到极端样本。
挂载资产：aerial-ground 数据、SMPL 几何、SOLIDER-Swin。
区别切开点：最像的是换衣课程学习那篇，但它的难度轴是衣服数量和衣服差异；我们用三维可见性、人体投影尺度、俯仰角和部位缺失定义难度。和 TSKD 的区别是它做模态蒸馏阶段，我们做样本进入训练的几何课程。
最小首验：先不改模型，只改采样器。用同一个 SOLIDER-Swin Tiny，在前 20 到 30 个 epoch 比较随机采样和几何课程采样。若几何难度不能预测训练损失或错误对，且短训无稳定正信号，就杀掉。

3. 标题句：无监督航拍地面 ReID 的伪标签噪声来自连续视角和低清失衡，不是传统相机类别，应该按三维姿态可比性筛选、补回和重对齐伪标签。
挂载资产：aerial-ground 数据、SMPL 几何、pose 热图门控。
区别切开点：最像的是 CMLR 和 MCBL，但它们用相机或模态这种离散标签做平衡；航拍地面更关键的是连续视角、人体投影、可见部位和分辨率。我们做的是几何平衡标签精炼，不是普通 camera-aware refinement。
最小首验：隐藏训练标签，用 SOLIDER 特征跑 DBSCAN，统计伪标签错误是否集中在大视角差、低可见部位、低投影面积区间。再做一个后处理版几何重对齐，看聚类纯度和 NMI 是否明显提升。若几何分组解释不了错误，直接杀掉。

我最建议先验一号。它的问题定义最硬，也最能把团队的 SMPL、pose gating 和 aerial-ground 资产合成一个别人不容易复刻的机制。二号可以作为一号的低成本前置实验，三号适合在有无监督设定价值时另开。
tokens used
77,646
基于每篇前三页。下面只抽对“怎么反推新方向”有用的部分。

**逐篇提取**

1. `Two-stage knowledge distillation for visible-infrared person re-identification`
创新类型：新机制。把 VI-ReID 从“一步跨模态硬对齐”改成“先收紧同模态，再跨模态互蒸馏”。
为什么能发：指出已有方法忽略同模态内部差异，还依赖完整标注。证据链是特征距离可视化、弱监督设置、SYSU-MM01 和 RegDB 对照。
Story 套路：跨模态不是一上来就对齐，而是从简单到困难，把多对多问题先压成原型级关系。

2. `UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges`
创新类型：新数据设定和问题重定义。
为什么能发：UAV ReID 不是普通 ReID 的小分支，而是动态背景、运动模糊、低清、视角高度变化、机载算力约束的组合问题。前三页强调公开数据和方法都很少。
Story 套路：地面摄像头 ReID 的假设在航拍里系统性失效，所以需要专门机制，不是直接迁移模型。

3. `Uniform Light Transformer for Person Re-identification under Complex Illumination`
创新类型：问题重定义加新机制。
为什么能发：把复杂光照下的失败定位到统一光照转换器的低频建模不足，而不是泛泛说光照变化。证据链是频谱分解，低频差异随光照尺度增多而增大。
Story 套路：视觉上像修好了，但频域里身份相关的低频结构坏了，所以要约束频谱和低频特征一致性。

4. `Unleashing the potential of traditional person re-ID methods to clothes changed scenario via curriculum learning`
创新类型：训练机制和问题重定义。
为什么能发：不再做专门换衣模块，而是证明传统同衣 ReID 方法有潜力，只是训练初期同时面对过大同身份衣服差异。证据链是多个 SC 方法在 PRCC、LTCC、VC-Clothes 等数据集上通过衣服课程学习超过 CC 方法。
Story 套路：不是模型不会换衣，而是学习顺序太粗暴。先学一个衣服，再逐步增加最难衣服。

5. `Unsupervised Lifelong Person Re-Identification via Affinity Harmonization`
创新类型：新机制。
为什么能发：无监督终身 ReID 的核心不是只防遗忘，而是稳定性和可塑性的平衡。证据链是旧域专家约束旧关系，新域专家指导新关系，再用旧域相机原型限制类内方差。
Story 套路：保留旧知识不能靠冻结模型压住一切，而要同时维护旧关系和吸收新关系。

6. `Unsupervised person re-identification via camera-aware multi-level label refinement`
创新类型：新机制。
为什么能发：把无监督 ReID 的伪标签噪声拆成聚类噪声和相机偏置。证据链是 DBSCAN 簇数波动和 t-SNE 相机聚团现象，再做簇级和实例级双层精炼。
Story 套路：相机不是元数据，而是伪标签错误的结构来源，必须进入标签精炼过程。

7. `Unsupervised Person Re-Identification With Diffusion Model via Semantic-Aware Disentanglement Representation Learning`
创新类型：新机制。
为什么能发：把无监督 ReID 的问题说成身份特征和相机背景特征纠缠，以及普通切块破坏语义结构。机制是身份和相机特征解耦，再用扩散模型生成语义空间变换参数。
Story 套路：扩散模型不是用来生成整个人，而是用来在无标签条件下找到语义局部。

8. `Unsupervised Visible-Infrared Person ReID via Modality-Camera Balance Label Refinement`
创新类型：新机制。
为什么能发：指出 USL-VI-ReID 不只是模态差异，还有模态和相机共同导致的层级伪标签噪声。证据链是模态相机不平衡导致聚类错误，再用筛选、重对齐和混合对比学习修正。
Story 套路：跨模态无监督不能只喊模态对齐，要在相机和模态两个层级上平衡伪标签。

9. `Unveiling Local Well-posedness Influence for Cross-modal Person Re-Identification`
创新类型：问题重定义加新机制。
为什么能发：把图文 ReID 的局部错配定义成“局部可解性”问题，极端天气、光照、遮挡会让文本和图像局部根本不对应。机制是实体、属性、外观三阶段掩码重建。
Story 套路：先判断局部对应是否可解，再做局部对齐，否则对齐会把模型带偏。

**这批论文给我们的启发**

最有价值的不是“加蒸馏、加扩散、加相机标签”，而是三个写法：把旧任务重新刻画成“局部是否可解”“学习顺序是否合理”“伪标签错误是否有结构”；先用可视化或统计证明这个结构确实存在；再把机制严格扣到这个结构上。

**强创新点候选**

1. 标题句：航拍地面 ReID 的核心不是视角差，而是很多身体局部在三维投影上本来不可对齐，模型应该先判断哪些局部可解，再学习身份。
挂载资产：aerial-ground 数据、SMPL 3D 几何、pose 热图门控、SOLIDER-Swin。
区别切开点：最像的是 local well-posedness 那篇，但它做图文局部掩码重建；我们做图像到图像的三维可见性、投影面积、视角夹角和低清约束。和 UAV survey 的区别是它只总结挑战，我们给出可计算的“局部可解性”机制。
最小首验：在 CARGO 或 AG-ReID.v2 上用现有 SMPL 或姿态估计得到每个身体部位的可见性、投影面积和视角差，检查这个分数是否能预测 SOLIDER 的失败样本。再做冻结特征的部位重加权重排。若相关性不明显，且重排 mAP 不到 0.4、rank1 不到 0.5，就杀掉。

2. 标题句：跨航拍地面训练不该一开始硬吃极端视角，而要按三维可观测性从容易样本逐步推进到极端样本。
挂载资产：aerial-ground 数据、SMPL 几何、SOLIDER-Swin。
区别切开点：最像的是换衣课程学习那篇，但它的难度轴是衣服数量和衣服差异；我们用三维可见性、人体投影尺度、俯仰角和部位缺失定义难度。和 TSKD 的区别是它做模态蒸馏阶段，我们做样本进入训练的几何课程。
最小首验：先不改模型，只改采样器。用同一个 SOLIDER-Swin Tiny，在前 20 到 30 个 epoch 比较随机采样和几何课程采样。若几何难度不能预测训练损失或错误对，且短训无稳定正信号，就杀掉。

3. 标题句：无监督航拍地面 ReID 的伪标签噪声来自连续视角和低清失衡，不是传统相机类别，应该按三维姿态可比性筛选、补回和重对齐伪标签。
挂载资产：aerial-ground 数据、SMPL 几何、pose 热图门控。
区别切开点：最像的是 CMLR 和 MCBL，但它们用相机或模态这种离散标签做平衡；航拍地面更关键的是连续视角、人体投影、可见部位和分辨率。我们做的是几何平衡标签精炼，不是普通 camera-aware refinement。
最小首验：隐藏训练标签，用 SOLIDER 特征跑 DBSCAN，统计伪标签错误是否集中在大视角差、低可见部位、低投影面积区间。再做一个后处理版几何重对齐，看聚类纯度和 NMI 是否明显提升。若几何分组解释不了错误，直接杀掉。

我最建议先验一号。它的问题定义最硬，也最能把团队的 SMPL、pose gating 和 aerial-ground 资产合成一个别人不容易复刻的机制。二号可以作为一号的低成本前置实验，三号适合在有无监督设定价值时另开。
