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
session id: 019ef71f-7f8a-7f13-b383-040275531e67
--------
user
你是 ReID 论文创新挖掘员 8/20。当前目录就是论文库,下面这批 PDF 在本目录。用 `pdftotext -l 3 '文件名' -` 读每篇的摘要+引言(只读前3页,不读全文,省时间)。

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

这批论文(8/20):
Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
Exploring Part-Informed Visual-Language Learning for Person Re-Identification.pdf
False Negatives Consensus Suppression for Text-to-Image Person Re-identification.pdf
FDGReID - Federated Domain Generalization for Person Re-identification.pdf
Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf

中文输出,务实。重点不是总结论文,是**反推出能让我们发 B 类的新强创新点**。
codex
我先按项目规则确认当前目标和实验状态，然后只用 `pdftotext -l 3` 抽取这 9 篇的前三页。后面会把每篇压缩成创新反推信息，再合并成可首验的强候选点。
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
/bin/zsh -lc "rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'AGENTS.md' -g 'STATUS.md' /Users/abdslm/Desktop/reid-paper" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/CLAUDE.md
/Users/abdslm/Desktop/reid-paper/AGENTS.md
/Users/abdslm/Desktop/reid-paper/GOAL.md
/Users/abdslm/Desktop/reid-paper/experiments/STATUS.md

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

codex
目录和目标文件位置已经确认。现在先读项目目标文件，避免把这批论文分析偏到当前主线之外。
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
/bin/zsh -lc "sed -n '1,240p' /Users/abdslm/Desktop/reid-paper/AGENTS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
目标文件说明当前就是文献差距分析阶段，不开实验。下面我开始按批读前三页，先抓每篇的摘要、引言、贡献点和它们自称的缺口。
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
/bin/zsh -lc "pdftotext -l 3 'Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Efficient Lightweight Multi-Source Domain Adaptation for
Person Re-ID via Self-paced Meta-Learning
XIAOYU ZHANG, School of Artificial Intelligence, Shenyang University of Technology, Shenyang, China
CHUANQI QIN and XITING PENG, School of Information Science and Engineering, Shenyang
University of Technology, Shenyang, China
XIAOLING ZHANG, School of Artificial Intelligence, Shenyang University of Technology,
Shenyang, China
LEXI XU, Research Institute, China United Network Communications Corporation, Beijing, China
HUAXUAN ZHAO, School of Information Science and Engineering, Shenyang University of Technology,
Shenyang, China
Person re-identification (Re-ID) aims to match individuals across different cameras, a task complicated by
variations in camera positions, resolutions, and lighting conditions. While supervised training improves Re-ID
model accuracy, it requires significant annotation efforts. Unsupervised domain adaptation (UDA) methods
address this by leveraging unlabeled target domain data but often fail to fully utilize multiple source domains
and are constrained by computational resources. This article introduces a lightweight multi-source domain
adaptation method for person Re-ID that combines meta-learning with pseudo-label-based UDA. By employing
Self-paced Meta-Learning (SpML) and style enhancement techniques, the model learns domain-invariant
knowledge from easy to difficult source domains, enhancing pseudo-label quality during adaptation. Our
approach, based on an omni-scale feature extraction network using deep separable convolution, combines
global and partial feature branches to capture richer pedestrian features. Experiments on public and real-world
datasets demonstrate that our method achieves competitive performance with significantly fewer parameters
and Floating Point Operations (FLOPs) compared to state-of-the-art models, proving its effectiveness and
practicality.
CCS Concepts: • Computing methodologies → Object identification;
Additional Key Words and Phrases: Person Re-Identification, Unsupervised Learning, Multi-source Domain
Adaptation, Meta-learning
This study is supported in part by the Key Technologies Research and Development Program (grant no. 2024YFF0617200),
Liaoning Science and Technology Major Project (grant no. 2024JH1/11700043), the Natural Science Foundation of Liaoning
Province (grant no. 2024-bs-102, 2025-MSLH-539), the Basic Scientific Research Project of the Education Department of
Liaoning Province (grant no. LJ222410142043).
Authors’ Contact Information: Xiaoyu Zhang, School of Artificial Intelligence, Shenyang University of Technology, Shenyang, China; e-mail: xy.zhang@sut.edu.cn; Chuanqi Qin School of Information Science and Engineering,
Shenyang University of Technology, Shenyang, China; e-mail: qinchuanqi@smail.sut.edu.cn; Xiting Peng (corresponding
author), School of Information Science and Engineering, Shenyang University of Technology, Shenyang, China; e-mail:
xt.peng@sut.edu.cn; Xiaoling Zhang, School of Artificial Intelligence, Shenyang University of Technology, Shenyang, China;
e-mail: zhangxiaoling@sut.edu.cn; Lexi Xu, Research Institute, China United Network Communications Corporation,
Beijing, China; e-mail: xulx29@chinaunicom.cn; Huaxuan Zhao, School of Information Science and Engineering, Shenyang
University of Technology, Shenyang, China; e-mail: zhaohuaxuan22@smail.sut.edu.cn.

This work is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives International
4.0.
© 2026 Copyright held by the owner/author(s).
ACM 1551-6865/2026/6-ART171
https://doi.org/10.1145/3798053
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.

171:2

X. Zhang et al.

ACM Reference format:
Xiaoyu Zhang, Chuanqi Qin, Xiting Peng, Xiaoling Zhang, Lexi Xu, and Huaxuan Zhao. 2026. Efficient
Lightweight Multi-Source Domain Adaptation for Person Re-ID via Self-paced Meta-Learning. ACM Trans.
Multimedia Comput. Commun. Appl. 22, 6, Article 171 (June 2026), 18 pages.
https://doi.org/10.1145/3798053

1

Introduction

Person re-identification (Re-ID), a technology focused on cross-camera identity matching in
disjoint surveillance networks, has emerged as a critical research area owing to its vital role in
intelligent security and public safety applications. Supervised Re-ID methods have seen remarkable
improvements due to the development of deep learning structures, particularly convolutional neural
network architectures. Concurrently, researchers have progressively developed expanded benchmark datasets with growing image volumes to support methodological innovations in this domain
[15]. However, in the context of training sets and test sets coming from the same domain, even when
trained on extensive datasets, Re-ID models typically exhibit considerable performance degradation
because of domain gaps resulting from differences in illumination and camera perspective. On the
other hand, annotating datasets to improve performance consumes a lot of manpower and resources.
Therefore, scholars have expressed interest in unsupervised domain adaptation (UDA).
In UDA, a model initially trained on labeled source data is adapted to perform effectively on an
unlabeled target domain, making it a cross-domain problem. Existing approaches can be classified
into two categories: one leverages domain alignment techniques such as using cross-domain similarity transfer frameworks to enhance domain alignment [7]. The other employs clustering algorithms
to group unlabeled datasets by ID and trains the network using pseudo-labels generated from clustering. Generally, pseudo-label-based algorithms achieve better performance on the target domain.
Most recent UDA methods focus on refining pseudo-labels to reduce the impact of noisy labels [29],
but these methods ignore the rich source domain knowledge. If a more powerful and better learning
initial model can be learned from multiple source domains first, then the subsequent pseudo-label
learning can be supervised from less noisy labels, making feature learning more accurate, thereby
further refining more accurate pseudo-labels. How to obtain a model with cross-domain capabilities
from multiple existing datasets, which introduces domain generalization (DG). DG refers to
learning a generalizable model from one or more datasets that can be applied to any unknown
domain. Existing DG methods use meta-learning, style transfer, and instance normalization (IN)
to learn domain-invariant knowledge from multiple source domains [23]. However, most DG works
only focus on generalization capabilities in unseen domains. With the development of algorithms
such as object detection, it is becoming easier to obtain unlabeled person image data. Therefore,
the combination of DG and domain adaptation methods is an important part of UDA.
As a key technology in Internet of Things (IoT) surveillance systems, person Re-ID has certain
requirements for the models deployed on related devices. Edge devices typically face constraints
in processing power and memory capacity, making it difficult to support complex and largescale models [21]. Additionally, many IoT applications require local data processing at the edge
to safeguard user privacy or minimize data transmission needs. To address these challenges,
lightweight models can operate locally in real-time using low resources, eliminating the need
for cloud transmission while meeting the demands of low resource consumption and privacy. In
conclusion, developing a lightweight, UDA Re-ID method is essential for the practical deployment
of person Re-ID systems.
In addition to lightweight requirements, person Re-ID performance is often affected by various
factors in real-world scenarios, such as significant variations in pose and lighting, degraded pixel
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.

Efficient Lightweight Multi-Source Domain Adaptation for Person Re-ID

171:3

quality, and occlusions [22]. Most domain adaptation methods focus solely on global features
during feature extraction, which may fail to accurately represent pedestrian information in practical
settings. In recent years, transformer-based models have gained popularity among researchers
due to their ability to directly model relationships between arbitrary image patches. For instance,
introducing attention pyramid Transformers or improving positional encoding can enhance the
accuracy of visual tasks [16]. Additionally, some approaches integrate other pedestrian information,
such as human pose estimation [10] and motion [26] into Re-ID. However, these models tend to
be large and complex. Therefore, if a model can learn more fine-grained and richer features while
remaining lightweight, it would be better suited for adapting to new domains.
In general, in view of the fact that existing methods do not fully utilize multiple source domains
and do not consider environmental issues such as device limitations and light occlusion in actual
scenes, this study proposes a lightweight multi-source UDA method based on overall level features
and partial level features. Specifically, the work of this study can be summarized as follows:
— In order to make full use of the source domain data in multi-source domain UDA, this article
combines the pseudo-label-based domain adaptation method with the meta-learning method.
Through Self-paced Meta-Learning (SpML) and style enhancement technology, the model
simulates the human learning process and learns the source domain knowledge from easy
to difficult so that the model can learn domain-invariant knowledge, thereby improving the
quality of pseudo-labels in the domain adaptation stage.
— In response to the problems of limited processing equipment resources in actual scenarios
and environmental factors such as object occlusion and light changes that interfere with
image recognition accuracy, this article uses deep separable convolution to build a lightweight
network and designs a feature extraction network consisting of overall level branches and
partial level branches to obtain richer feature information from pedestrian images and improve
recognition capabilities.
— This article validates the model on multiple public datasets and their different combinations
and also tests the model performance using a set of datasets from real application scenarios.
2
2.1

Related Work
Supervised Person Re-ID

Supervised person Re-ID is the most widely studied and outstanding direction. This method requires
a big amount of training data with annotated information, which can fully utilize the advantages
of annotated data and achieve high recognition accuracy in specific scenarios. With the continuous
development of deep learning and attention mechanisms, supervised Re-ID methods have achieved
superior performance. For example, Luo et al. [18] presented an efficient multi-scale channel
attention network to learn robust and more discriminative features through cross-channel attention
modules. In recent years, with the popularity of Transformer, He et al. [9] proposed a Transformerbased Re-ID framework and tried to use pure Transformer for person Re-ID research for the first
time. Subsequently, Li et al. [14] studied how to apply the pre-trained visual language model
contrastive language–image pretraining to fine-grained image Re-ID tasks, further improving
the accuracy of Re-ID. Furthermore, Liu et al. [17] proposed Mix-Modality Re-ID with CrossIdentity Discrimination Harmonization loss for hyperspherical identity-modality harmonization
and bridge-based similarity optimization strategy to enable robust day-night mixed retrieval.
While supervised techniques have demonstrated outstanding performance in the same scenario, in
practical applications, due to the difficulty of data annotation, more and more researchers have
turned their attention to the task of unsupervised method.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 22, No. 6, Article 171. Publication date: June 2026.


exec
/bin/zsh -lc "pdftotext -l 3 'Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11210054

Dynamic Token Selective Transformer for
Aerial-Ground Person Re-Identification
Yuhai Wang*

Maryam Pishgar

Department of Computer Science
University of Southern California
Los Angeles, United States
yuhaiwan@usc.edu

Department of Industrial and Systems Engineering
University of Southern California
Los Angeles, United States
pishgar@usc.edu

Abstract—Aerial-Ground Person Re-identification (AGPReID)
holds significant practical value but faces unique challenges due
to pronounced variations in viewing angles, lighting conditions,
and background interference. Traditional methods, often involving a global analysis of the entire image, frequently lead to inefficiencies and susceptibility to irrelevant data. In this paper, we
propose a novel Dynamic Token Selective Transformer (DTST)
tailored for AGPReID, which dynamically selects pivotal tokens
to concentrate on pertinent regions. Specifically, we segment the
input image into multiple tokens, with each token representing
a unique region or feature within the image. Using a Top-k
strategy, we extract the k most significant tokens that contain
vital information essential for identity recognition. Subsequently,
an attention mechanism is employed to discern interrelations
among diverse tokens, thereby enhancing the representation of
identity features. Extensive experiments on benchmark datasets
showcases the superiority of our method over existing works.
Notably, on the CARGO dataset, our proposed method gains
1.18% mAP improvements when compared to the second place.
In addition, we comprehensively analyze the impact of different
numbers of tokens, token insertion positions, and numbers of
heads on model performance. Please checkout our website for
code and dataset: https://yuhaiw.github.io/DTS-AGPReID/
Index Terms—Aerial Ground Person Re-identification, Top-k
Token Selective Transformer, Attention Mechanism

I. I NTRODUCTION
Person Re-identification (ReID) is crucial for surveillance
and tracking, identifying individuals across camera views. Advances in deep learning have improved feature extraction and
matching accuracy [1]–[5]. However, most methods rely on
global image features, making them vulnerable to background
noise and irrelevant regions, particularly in cases of occlusion
or complex backgrounds. This limits their effectiveness in
diverse real-world scenarios with cross-camera variations and
environmental inconsistencies [6]–[8].
To address these challenges, recent studies have emphasized the importance of more targeted and efficient feature
extraction approaches. For instance, Zhang et al. [9] propose
a separable attention mechanism to focus on discriminative regions while suppressing irrelevant background features.
Tang et al. [10] introduce adaptive context-aware selection to
dynamically enhance feature representations under complex
conditions. Similarly, Qiu et al. [11] develop a salient feature
*Corresponding Author, yuhaiwan@usc.edu

Fig. 1: A straightforward description of Aerial-Ground Person Re-identification (AGPReID) involves the utilization of
an aerial-ground mixed camera network, enabling matching
across aerial-aerial, ground-ground, and aerial-ground scenarios. Thus, it presents greater challenges and practical applications compared to traditional single-camera person ReID
methods.

extraction framework that prioritizes key object parts even in
scenarios involving significant occlusion. These advancements
show promising progress in overcoming the limitations of
the reliance on global feature in View-homogeneous person
ReID. However, when applied to Aerial-Ground Person Reidentification (AGPReID) tasks (View-heterogeneous person
ReID), which are valuable in real-world scenarios for addressing complex aerial-to-ground matching challenges and
encompassing diverse camera perspectives [12], these methods
often fall short. Fig. 1 demonstrates the AGPReID problem.
This discrepancy may stem from the scale diversity and redundancy characteristics observed in large-area observational
scenarios, leading to notable appearance differences for the
same individual across various cameras. Therefore, there is an
urgent need to develop innovative strategies that effectively
address these specific challenges in AGPReID.
To this end, we propose a Dynamic Token Selective Transformer (DTST) that enhances identity representation by focusing on the most critical spatial features. Our DTST module
contains two steps: First, a Predictor Local-Global network

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:56 UTC from IEEE Xplore. Restrictions apply.

Fig. 2: Illustration of the proposed Dynamic Token Selective Transformer (DTST) framework. The framework incorporates N
Token Selection view-decoupled transformer (VDT) blocks, where each block consists of an encoder layer and a visual token
selector. The loss function is designed to account for both view-related and view-unrelated features, while an orthogonal loss
ensures that these features remain independent from each other, further enhancing feature disentanglement and robustness.

Fig. 3: The Illustration of Visual Token Selector (VTS). The process involves selecting the Top-K informative tokens from the
original token set to be used in the subsequent feature aggregation.

computes relevance scores for each token, integrating local and
global spatial semantics using multi-head attention. Second, a
Perturbation-Based Top-K Selector chooses the most relevant
tokens based on the predicted scores, ensuring robustness by
adding noise perturbations. The selected tokens are combined
with a global class token, enabling efficient and compact representation while reducing computational overhead. Extensive
experiments validate our method’s state-of-the-art performance
on AGPReID tasks, showcasing its robustness in handling
occlusions, complex backgrounds, and viewpoint variations.
Our main contributions are as follows.
We propose a Top-k Token Selective Transformer for
AGPReID, to better model identity representation spatially. We further comprehensively study the impact of
the insertion position and the number of tokens selected
on the model’s performance.
• To eliminate the interference of irrelevant tokens, our
method adaptively selects the most critical tokens based
on the top-k selective mechanism, making the long-range
modeling more effective and compact.
• Extensive experiments on various datasets demonstrate
that our proposed model achieves state-of-the-art performance on AGPReID tasks.

•

II. R ELATED W ORK
A. Person Re-identification
Person re-identification (ReID) is essential for retrieving
images of the same individual across different camera views.
It can be categorized into view-homogeneous and viewheterogeneous ReID. View-homogeneous ReID pertains to
scenarios with a single camera type, such as ground-only or
aerial-only networks, while view-heterogeneous ReID such
as Aerial-Ground Person ReID (AGPReID), deals with networks featuring diverse camera perspectives. In terms of
view-homogeneous ReID, ground-only camera networks have
received more attention compared to aerial-only networks. For
example, some ground-only datasets are well established such
as Market1501 [13] and MSMT17 [14]. As a consequence,
a multitude of methods have been proposed, such as handcrafted feature-based, CNN-based, and transformer-based approaches, facilitating the development of ReID. However, these
methods overlook the significant view differences between
aerial and ground cameras, leading to poor performance
faced with diverse view-point scenarios. Fortunately, viewheterogeneous ReID can address this issue. Recently, researchers in [12] propose the AG-ReID dataset, which includes

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:56 UTC from IEEE Xplore. Restrictions apply.

identity and attribute labels, and put forward an attributeguided model. Another work extends this by introducing
the CARGO dataset with multiple matching scenarios and
proposes a view-decoupled transformer (VDT) that decouples
view-related features using hierarchical separation and orthogonal loss, improving performance and reducing reliance on
extensive attribute labeling [15]. However, this approach does
not dynamically select key tokens related to the target object,
fails to reduce redundant computation, and lacks enhanced
model capability to focus specifically on critical regions of
interest.
B. Token Selection in Vision Transformers
Token selection is crucial for addressing redundancy issues
in transformer-based vision models, particularly in tasks involving dense visual data. Despite their success, transformers
often suffer from computational inefficiencies due to the need
to process numerous redundant tokens. Token selection methods can effectively mitigate this issue by focusing on only the
most informative tokens for further processing. For example,
STTS [16], as a representative work, utilizes token selection
to enhance computational efficiency by dynamically reducing
the number of tokens processed at each transformer layer.
These approaches have demonstrated substantial reductions in
computation while maintaining performance. To address the
challenge of differentiability in token selection, a perturbed
maximum strategy is introduced [17], enabling top-K selection
to be differentiable, thereby facilitating end-to-end training.
Building on the principles of differentiable top-K selection
[18], we develop a lightweight token selection module specifically designed to enhance temporal-spatial modeling in our
view-decoupled transformer. By selecting only the most informative tokens, this module reduces redundancy and improves
both efficiency and performance, especially in modeling visual
data across multiple viewpoints.
III. M ETHOD
A. Formulation
Aerial-Ground Person ReID aims to match images from
ground- or aerial-only camera networks. In a training dataset
|D tr |
Dtr = {(xi , yi , vi )}i=1 , each instance consists of an image
xi depicting a person, along with identity label yi and view
label vi . The view label vi ∈ {v a , v g } is determined by
the known camera labels in D, distinguishing between aerial
(v a ) and ground (v g ) views. A substantial distinction in
views between v a and v g results in a biased feature space,
characterized by low intra-identity similarity and high interidentity dissimilarity.
B. Overview
As illustrated in Fig.2, we propose a token enhanced framework based on the View-Decoupled Transformer (VDT) to
tackle the view discrepancy challenge in AGPReID. Input
images that include both aerial (va ) and ground (vg ) views
are tokenized into a sequence of tokens. To encompass both
global and view-specific details, meta tokens and view tokens

are added to these image tokens before they are inputted into
our VDT.
Comprising N blocks, the VDT framework initiates each
block with a conventional self-attention encoding process,
succeeded by a subtraction operation between meta and view
tokens to explicitly disentangle view-specific characteristics
from the overarching ones. This facilitates a distinct segregation of features influenced by diverse viewpoints.
Subsequently, the updated meta and view tokens produced
by the VDT are supervised by identity and view classifiers. To
enforce the independence of meta and view tokens, we introduce an orthogonal loss, facilitating the successful separation
of view-based and view-agnostic attributes. To select the most
critical tokens, a visual token selector module is proposed to
enhance the identity representation, with further elaboration
provided in subsequent sections.
We introduce the Visual Token Selector (VTS), as shown in
Fig. 3, designed to dynamically refine the token representation
by selecting the most informative tokens for subsequent analysis. This module aims to reduce redundancy and enhance the
model’s ability to focus on critical regions, thereby optimizing
computational efficiency while preserving feature quality. The
VTS mechanism can be understood as a dynamic token
selection process that leverages attention scores to determine
the importance of each token.
For a sequence of tokens {ti }M
i=1 , where M is the number
of tokens, the VTS computes importance scores for each token
si using a lightweight attention mechanism. The score si is
obtained as:

 ⊤
ti Wq Wk⊤ ti
√
,
si = softmax
d
where ti is the i-th token, Wq and Wk are learnable matrices
representing query and key transformations, and d is the
dimensionality of the tokens. The softmax function normalizes
the scores to ensure they sum to 1, thus creating a probabilistic
distribution over the tokens.
These tokens are then ranked based on their importance
scores, and we select the top-K tokens with the highest
scores, where K < M is a hyperparameter that controls the
number of tokens retained. Mathematically, this selection can
be represented as:
{tselected
} = TopK({si }M
i
i=1 ),
where TopK(·) returns the indices corresponding to the topK scores. The retained tokens, {tselected
}, are then passed to the
i
subsequent layers or directly to the final classification head.
To ensure that the VTS can be used in an end-to-end
training fashion, we adopt a differentiable approach for the
token selection. Specifically, we use a continuous relaxation
of the TopK function by employing a Gumbel-Softmax trick:
exp((si + gi )/τ )
ŝi = PM
,
j=1 exp((sj + gj )/τ )

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:56 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
8086

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025

Enhancing Visible-Infrared Person Re-Identification
With Modality- and Instance-Aware
Adaptation Learning
Ruiqi Wu , Bingliang Jiao , Meng Liu, Shining Wang , Wenxuan Wang , Member, IEEE,
and Peng Wang , Member, IEEE

Abstract—The Visible-Infrared Person Re-identification (VI
ReID) aims to achieve cross-modality re-identification by matching pedestrian images from visible and infrared illumination.
A crucial challenge in this task is mitigating the impact of
modality divergence to enable the VI ReID model to learn
cross-modality correspondence. Regarding this challenge, existing
methods primarily focus on eliminating the information gap
between different modalities by extracting modality-invariant
information or supplementing inputs with specific information
from another modality. However, these methods may overly
focus on bridging the information gap, a challenging issue
that could potentially overshadow the inherent complexities of
cross-modality ReID itself. Based on this insight, we propose
a straightforward yet effective strategy to empower the VI
ReID model with sufficient flexibility to adapt diverse modality
inputs to achieve cross-modality ReID effectively. Specifically, we
introduce a Modality-aware and Instance-aware Visual Prompts
(MIP) network, leveraging transformer architecture with customized visual prompts. In our MIP, a set of modality-aware
prompts is designed to enable our model to dynamically adapt
diverse modality inputs and effectively extract information for
identification, thereby alleviating the interference of modality
divergence. Besides, we also propose the instance-aware prompts,
which are responsible for guiding the model to adapt individual
pedestrians and capture discriminative clues for accurate identification. Through extensive experiments on four mainstream
VI ReID datasets, the effectiveness of our designed modules is

Received 22 August 2024; revised 13 February 2025; accepted 7 April 2025.
Date of publication 11 April 2025; date of current version 6 August 2025.
This work was supported in part by the National Natural Science Foundation
of China under Grant 62476226 and in part by Guangdong Basic and Applied
Basic Research Foundation under Grant 2025A1515011465. This article was
recommended by Associate Editor X. Chang. (Ruiqi Wu and Bingliang Jiao
contributed equally to this work.) (Corresponding author: Wenxuan Wang.)
Ruiqi Wu, Bingliang Jiao, Shining Wang, and Peng Wang are
with the School of Computer Science, Northwestern Polytechnical
University, Xi’an 710072, China, also with Ningbo Institute, Northwestern
Polytechnical University, Ningbo 315000, China, and also with the National
Engineering Laboratory for Integrated Aero-Space-Ground-Ocean Big
Data Application Technology, Xi’an 710072, China (e-mail: wurq@
mail.nwpu.edu.cn;
bingliang.jiao@mail.nwpu.edu.cn;
wangshining@mail.nwpu.edu.cn; peng.wang@nwpu.edu.cn).
Meng Liu is with the School of Electronics and Information,
Northwestern Polytechnical University, Xi’an 710072, China (e-mail:
meng.liu@mail.nwpu.edu.cn).
Wenxuan Wang is with the School of Computer Science, Northwestern
Polytechnical University, Xi’an 710072, China, also with Shenzhen Research
Institute, Northwestern Polytechnical University, Shenzhen 518057, China,
and also with the National Engineering Laboratory for Integrated AeroSpace-Ground-Ocean Big Data Application Technology, Xi’an 710072, China
(e-mail: wxwang@nwpu.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2025.3560118

evaluated. Furthermore, our proposed MIP network outperforms
most current state-of-the-art methods.
Index Terms—Visible-infrared person re-identification, crossmodality person re-identification, visual prompt learning.

I. I NTRODUCTION

P

ERSON Re-Identification (ReID) aims to retrieve images
of the same individuals from different cameras. With
its wide-ranging applications in public security and video
surveillance, ReID has sparked significant interest and notable
advancements in the field. Many existing methods [1], [2], [3],
[4] primarily focus on re-identification in daytime scenarios,
overlooking low-light conditions. However, treating ReID as
a single-modality problem is unreasonable, as this inevitably
causes existing methods to underperform in low-light environments. To overcome this limitation, the adoption of infrared
camera technology for continuous, all-conditions surveillance
has given rise to the Visible-Infrared Person Re-Identification
(VI ReID) task, enhancing the system’s robustness and effectiveness across diverse scenarios.
Different from the ReID task merely built upon the visible illumination, the query and gallery sets in VI ReID
are captured by cameras with distinct modalities, resulting
in significant modality gaps among the compared person
images. Existing methods typically focus on reducing modality
discrepancies by eliminating information gaps. For example,
some modality supplementing methods [5], [6], [7], [8] utilize Generative Adversarial Networks (GAN) [9] to generate
specific information from another modality for supplementing
inputs, achieving effective cross-modality matching. However, challenges arise because the brightness of infrared
images may not perfectly correspond to the color of visible
images [10], and the high computational cost of GANs also
makes stable modality transfer difficult [11]. Moreover, some
methods involve extracting discriminative modality-invariant
features and focusing on the commonality of visual features
between different modalities, to address modality discrepancies. For instance, SPOT [12] employs multi-level alignment
mechanisms and leverages physics knowledge, such as body
structure, to learn discriminative cross-modality invariant features.
However, these existing methods may overly focus on
bridging the information gap. In fact, we contend that merely

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.

WU et al.: ENHANCING VI ReID WITH MODALITY- AND INSTANCE-AWARE ADAPTATION LEARNING

eliminating the information gap does not fully address all the
challenges of the VI ReID task, such as capturing discriminative pedestrian clues for identification. Moreover, completely
bridging the information gap between diverse modalities could
be a very challenging issue, potentially even more than crossmodality re-identification itself. Here, let us consider the
fundamental challenge of the VI ReID task, which we believe
involves training a model to consistently extract discriminative
information for ReID from various instances within different
modalities. Based on this, in this work, we propose a more
straightforward and effective strategy that directly enhances
the model’s adaptability, enabling it to flexibly adapt to various
instances and diverse modalities. The core insight of our
work is that when a model has sufficient flexibility, it can
dynamically adapt to the characteristics of different modalities
and consistently extract significant information from diverse
modality inputs for identification. This enables our model
to effectively mitigate the impact of modality divergence.
Besides, this flexibility also allows our model to adapt to
different instance inputs and adaptively recognize their discriminative clues.
Based on this idea, in this paper, we propose a novel and
effective Modality-aware and Instance-aware Visual Prompts
(MIP) network to address the VI ReID task. Our key innovation is to endow the model with sufficient flexibility, enabling
it to adapt to various modalities and instances. In this step, we
notice that visual prompts could be a good tool to accomplish
this. Recently, Visual Prompt Tuning (VPT) [13] and its
extensive use in numerous existing works [13], [14], [15],
[16], [17], [18] showcases its ability to adapt the origin
models efficiently to various target tasks. Inspired by this,
in this work, we customize two types of visual prompts
namely, modality-aware prompts and instance-aware prompts
to adapt our model. The modality-aware prompts are designed
to learn and equip our model with the characteristics of
the current inputs. This enables the model to dynamically
adapt to diverse modalities, thereby alleviating the interference
caused by modality divergence. Regarding the instance-aware
prompts, they are responsible for guiding our model to adapt
to the input instances, thereby enabling our model to capture instance-aware discriminative clues for identification. As
shown in Fig. 1, our method focuses on model adaptability to
different modalities and instances. Using modality and instance
prompts, the model’s parameters and feature extraction process
are adjusted to dynamically adapt to the feature distributions of
input images from different modalities and instances, learning
discriminative information in an adaptive feature space and
overcoming more out-of-distribution samples that existing
methods struggle to process, thereby improving identification
performance. In contrast, existing methods typically focus
on reducing the distribution gap between different modalities
by mapping input images to a shared feature space, aiming
to facilitate subsequent matching and recognition. However,
eliminating the distribution gap is a challenging task, and
certain hard samples may fail to map correctly to the shared
space, impacting identification accuracy.
Practically, the MIP network comprises a global backbone
and three prompt learning modules: a Modality-aware Prompt

8087

Fig. 1. The illustrations of the modality- and instance-aware adaptation,
and the difference between existing methods and our method. The circles in
different colors represent inputs from different modalities. Existing methods
typically focus on reducing the distribution gap between different modalities
to map input images to a shared feature space, aiming to facilitate subsequent
matching and recognition. However, eliminating the distribution gap is a
challenging task, and certain hard samples may fail to map correctly to the
shared space, impacting recognition accuracy. In contrast, our method avoids
this challenge by focusing on model adaptability to different modalities and
instances. Using modality and instance prompts, the model’s parameters and
feature extraction process are adjusted to dynamically adapt to the feature
distributions of input images from different modalities and instances, learning
discriminative information in an adaptive feature space and overcoming outof-distribution samples, thereby improving identification performance.

Learning (MPL) module and two Instance-aware Prompt
Generator (IPG) modules, i.e., a Self-guiding IPG (SIPG)
module, and a Query-guiding IPG (QIPG) module. In terms
of structure, the MPL consists of two sets of learnable vectors corresponding to infrared and visible modalities. These
prompt vectors are responsible for learning the characteristics
of distinct modalities and guiding our model to adapt to
them. Moreover, we have devised two innovative instanceaware prompt generators, namely SIPG and QIPG, based on
transformer architecture. In the two IPG modules, we employ
a transformer layer to transfer the identity-related information
from the image features into a group of learnable vectors
to construct instance-aware prompts. The distinction between
SIPG and QIPG lies in the source of image features: SIPG
receives features from the current input instances, while QIPG
receives features from the query instances to be matched. The
prompts generated by two IPG modules are supplied to the
backbone model to guide it in dynamically adapting to the
input instances and the query instances, respectively, thereby
capturing discriminative clues for identification. Additionally,
we have designed a Customized Prompt Fusion (CPF) module
to adaptively integrate modality-aware and instance-aware
prompts, so as to provide more effective guidance for our
model. We also designed some auxiliary loss functions, the
Instance-aware Enhancement Loss (IAEL) and the Causality
Enhancement Loss (CEL) to help the modules we designed to
function more effectively.
We summarize the contribution of our work as follows:
• We propose a novel method, the Modality-aware and
Instance-aware Visual Prompts (MIP) network, incorporating visual prompt learning into the VI ReID field.
• We design a Modality-aware Prompt Learning (MPL)
module, two Instance-aware Prompt Generators (IPG),

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.

8088

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 8, AUGUST 2025

i.e. the Self-guiding IPG (SIPG) and the Query-guiding
IPG (QIPG), and the Customized Prompt Fusion (CPF)
module to generate modality-aware and instance-aware
prompts for the ReID model, which guide the model
to adapt to the diverse modalities and instances. These
enable our model to alleviate the impact of modality
divergence and effectively capture discriminative instance
clues for identification.
• We execute extensive experiments on VI ReID benchmarks SYSU-MM01, RegDB, LLCM, and RGBN300 (for
Vehicle), which validate the effectiveness of both our
designed modules and demonstrate that MIP performs
better than most state-of-the-art methods.
Statement. This paper is an extended version of our previous work [19]. This version significantly expands upon the
original work by following several key aspects.
1) Theoretical Improvement:
This version provides a more comprehensive theoretical
analysis of how the model’s adaptability and flexibility help
in VI ReID. Then this version extends the adaptation on
modalities and instances to the adaptation on input modalities,
input instances, and the query instances to be matched.
2) Methodological Enhancements:
This version includes significant improvements in the
methodology, as follows:
• a) Based on the model with only two modules, the
Modality-aware Prompt Learning (MPL) module and
the Instance-aware Prompt Generator (IPG) module, the
journal version extra designs two IPG modules, namely
SIPG and QIPG. The SIPG takes the place of IPG in the
previous version, and the QIPG produces query-guiding
instance-aware prompts to guide the model to adapt to
the query instances to be matched.
• b) This version introduces a novel CFP module to dynamically integrate modality-aware prompts and instanceaware prompts and discusses how to deploy the prompts
better, comparing two kinds of fusion strategies to
integrate modality-aware prompts and instance-aware
prompts, so as to provide more effective guidance for
the model.
• c) This version designs a new CEL loss and introduces a
cross-modality triplet loss to help the model training.
3) Expanded Experimental Validation:
This version reports more extensive new experimental
results, validating the effectiveness of our proposed methods,
including: comparisons with more existing methods (including
some CLIP-based methods); comparisons with existing methods on two other mainstream datasets, namely LLCM and
RGBN300 (vehicle); ablations about the trade-off parameters
in hybrid objection functions; ablations about the new proposed modules; ablation about the length of prompts; ablation
about the QIPG module and its CEL loss constraint; discussion
on the parameters and computational complexity.

II. R ELATED W ORK
A. Person Re-Identification
Person Re-identification (ReID) is a crucial task that
involves matching query images of individuals with corresponding target images from a gallery set. Its importance
in real-world applications has garnered significant attention
and driven the development of various methodologies [1],
[3], [20], [21], [22], [23], [24], [25], [26], [27], [28]. Similar
tasks include vehicle ReID [29], [30], animal ReID [31],
etc.
ReID methods typically comprise two key components:
feature representation learning and deep metric learning.
Global-based approaches like VLAD [20], BNNeck [21],
among others, have been introduced to extract global-level
feature representations for individuals’ images. Furthermore,
part-based methods, such as PCB-RPP [3], leverage part-level
clues to amalgamate more robust representations for retrieval
purposes. Some algorithms combine both global and local features to exploit their respective advantages. For instance, Wang
et al. [22] proposed a multiple granularity network with one
branch for global feature representation and two branches for
local feature representation. Deep metric learning techniques
[23], [25], [26], such as triplet-loss [25] and quadrupleloss [26], aim to increase inter-identity feature distance and
reduce intra-identity variation. While many ReID methods
excel merely built upon the visible illumination, they may face
challenges in low-light scenarios due to insufficient handling
of the significant domain gap between visible and infrared
modalities. Addressing this gap is essential for improving the
versatility and applicability of ReID techniques in various
environmental conditions.
B. Visible-Infrared Person Re-Identification
Visible-Infrared Person Re-identification (VI ReID) focuses
on matching visible and infrared images of the same individuals across different cameras. The query and gallery sets
are captured by cameras using different modalities. However,
directly applying ReID methods designed for visible-light
images to VI ReID results in poor performance due to modality
discrepancies and differing distributions between the modalities [32]. Existing methods mainly focus on bridging the
modality gap, and they can generally be grouped into two
approaches: modality compensation and modality-invariant
feature learning.
Modality compensation represents a VI ReID paradigm that
supplements inputs with specific information from another
modality. This kind of method usually initially employs a Generative Adversarial Network (GAN) [9] to generate another
modality from the available one, then compensates the original
image with the generated image, to mitigate the modality
discrepancy. The cmGAN [5] is the first research introducing
GAN into the VI ReID task, but it utilizes GAN to better
distinguish between different modalities of images rather than
to generate images of missing modalities. CycleGAN [33] are
typical methodologies via modality compensation, which uses
GAN for style transfer of training images across different
cameras to create an augmented training set, which helps

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:52 UTC from IEEE Xplore. Restrictions apply.


codex
前三篇读完了。一个明显模式是：能发的文章经常不是把特征做得更复杂，而是把问题说成“全局统一处理不合适”，然后引入样本、模态、视角或难度感知的动态处理。
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
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -l 3 'Exploring Part-Informed Visual-Language Learning for Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Exploring Part-Informed Visual-Language Learning
for Person Re-Identification
Yin Lin1,2 , Yehansen Chen2 , Baocai Yin2 , Jinshui Hu2 , Bing Yin2 , Cong Liu2 , Zengfu Wang1 *
1 University of Science and Technology of China, Hefei, China
2 iFLYTEK Research, Hefei, China

arXiv:2308.02738v2 [cs.CV] 21 Mar 2025

lin5875@mail.ustc.edu.cn; zfwang@ustc.edu.cn;
{yinlin, yhschen, bcyin, jshu, bingyin, congliu2}@iflytek.com

Abstract—Recently, visual-language learning (VLL) has shown
great potential in enhancing visual-based person re-identification
(ReID). Existing VLL-based ReID methods typically focus on
image-text feature alignment at the whole-body level, while
neglecting supervision on fine-grained part features, thus lacking
constraints for local feature semantic consistency. To this end,
we propose Part-Informed Visual-language Learning (π-VL) to
enhance fine-grained visual features with part-informed language
supervisions for ReID tasks. Specifically, π-VL introduces a
human parsing-guided prompt tuning strategy and a hierarchical
visual-language alignment paradigm to ensure within-part feature semantic consistency. The former combines both identity
labels and human parsing maps to constitute pixel-level text
prompts, and the latter fuses multi-scale visual features with
a light-weight auxiliary head to perform fine-grained image-text
alignment. As a plug-and-play and inference-free solution, our
π-VL achieves performance comparable to or better than stateof-the-art methods on four commonly used ReID benchmarks.
Notably, it reports 91.0% Rank-1 and 76.9% mAP on the
challenging MSMT17 database, without bells and whistles.
Index Terms—Person re-identification, Visual-language learning, Fine-grained image-text alignment

I. I NTRODUCTION
Person re-identification (ReID) refers to the task of retrieving the query person-of-interest from large-scale gallery
databases captured by non-overlapping camera views [1].
Owing to its practical importance for intelligent video surveillance, ReID has gained ever-growing attention from both
academia and industry in recent years [2]–[4].
As appearance biometrics serve as the most fundamental
and well-studied cues for identity recognition [2], [5]–[7],
appearance-based ReID has achieved considerable success
across a wide range of applications. However, human body
semantics are not readily apparent in raw pixels, making it
challenging to learn semantic information under the single
supervision of one-hot or pair-wise identity labels [8].
Inspired by the recent success of visual-language models
[9], [10], CLIP-ReID [8] is one of the pioneer attempts
that leverages natural texts to specify visual concepts beyond
appearance. By tuning identity-specific text prompts [11],
it uses text representations generated by a powerful text
encoder [9] to deliver the image encoder a broader source
∗ Corresponding author

Learnable Identity-Specific Prompts

A photo of a [X]1[X]2[X]3[X]4 person

Text
Encoder

Text Embeddings
Inner Product

Image
Encoder

Identity Loss

Image Embeddings

(A) CLIP-ReID based on Global Image-Text Alignment
Learnable Identity-Specific Part Prompts
Head

Bag

…

Shoes

A photo of a [X]1[X]2[X]3[X]4 person’s head

Human
Parsing

Text
Encoder
Label
Guiding

Text Feature Map

MSE Loss

Image
Encoder
Visual Feature Map

(B) Our Part-Informed Visual-Language Learning

Fig. 1. Comparison of CLIP-ReID [8] and our part-informed visual-language
learning (π-VL) framework. (a) CLIP-ReID based on global image-text
alignment. (b) Our π-VL based on pixel-level image-text alignment.

of supervisions, leading to more discriminative global features. However, naively porting ideas from global image-text
alignment may not suffice for ReID. Several studies [3], [12]
have demonstrated that some non-salient details can be easily
overwhelmed, raising the within-part semantic inconsistency
issue (see Fig.2). And they also revealed that introducing partinformed identity supervisions is a promising solution to this
issue [3]. This motivates us to ask: Is learning fine-grained
body semantics as easy as global image-text alignment in ReID
task? An obstacle to addressing this issue lies in the ambiguous
boundaries between different parts of the human body. While
the human parsing task [13], [14] has effectively tackled
this problem, it introduces a new issue: supervision conflict.
Human parsing distinguishes identity-agnostic body part semantics, whereas ReID requires identity-specific discriminative
cues. This conflict can lead to reduced feature diversity and a
confused decision boundary for identity recognition.
In this paper, we address the above problems by introducing a Part-Informed Visual-Language learning framework,
termed π-VL, for person ReID tasks. Unlike existing works
that apply parsing maps for background elimination or body

alignment, we propose to construct pixel-level text prompts via
human parsing, and perform per-pixel image-text alignment to
enhance visual features. To alleviate the supervision conflict
problem, we combine both global-level identity labels and
pixel-level parsing semantics for contrastive prompt tuning,
leading to more discriminative part text embeddings. Furthermore, considering the hierarchical nature of visual backbones
[15], we propose a light-weight auxiliary head to fuse multistage visual features and design a parsing confidence weighted
alignment loss for robust semantic enhancement. It is worth
noting that our π-VL is a plug-and-play and loss functionbased solution, it is highly compatible with existing ReID
models. Experimental results on both CNN and ViT-based
backbones suggest that π-VL has the potential to be used as
a universal front-end capable of handling various model architectures. It achieves highly competitive results, i.e., 91.0%
Rank-1 and 76.9% mAP, on the MSMT17 benchmark, and
shows consistent improvements over mainstream person ReID
databases. Our contributions are summarized as follows:
• We propose a part-informed visual-language learning
framework, named π-VL, for person ReID. To our best
knowledge, this is one of the first attempts to introduce
fine-grained visual-language learning for ReID tasks.
• We present an identity-aware part-informed prompt tuning strategy based on human parsing. With this strategy,
we can generate pixel-level text prompts based on both
identity labels and parsing maps, strengthening the visual
encoder to spot more discriminative features.
• We design a novel fine-grained alignment mechanism for
ReID tasks. It integrates confidence scores from human
parsing to weight the alignment loss, leading to a more
semantically rich feature space for person image retrieval.
• Extensive experiments on mainstream ReID benchmarks
not only demonstrate the superior performance of the
proposed method, but also validate its generalization
ability to various visual encoders.
II. R ELATED W ORK
A. Appearance-based Person ReID
Appearance-based ReID aims to match a target pedestrian
across disjoint visible camera views at varying places and
times. It is challenging to learn suitable feature representations
robust enough to withstand large intra-class variations of
illumination, poses, and background clutter [2], [16].
Nowadays, deep learning methods show powerful capacity
of automatically extracting features from large-scale image
datasets and have achieved state-of-the-art results on RGBbased person ReID tasks [16]. Building on various sophisticated CNN architectures, deep ReID models are doing exceptionally well on visual matching by learning robust crosscamera feature representations and optimal distance metrics in
an end-to-end manner [1], [2]. To learn more discriminative
features, part information and contextual information are also
exploited in recent works [16], [17]. For example, methods
like PCB [3] and MGN [12] utilize hand-crafted partitioning

to split feature maps into grid cells or horizontal stripes for
local feature learning. Another line of researches adopt off-theshelf pose estimation or attention module [17] to extract the
human part aligned features. Although these approaches have
reported encouraging performance, the learned part features
still lack high-level semantics under the single supervision of
discrete identity labels [8].
B. Visual-language Pre-training
Over the past years, the emergence of visual-language pretraining models has led to substantial improvements to many
downstream tasks [9], [10]. Based on the idea of contrastive
image-text alignment, CLIP exploits a two-directional InfoNCE loss [9] to pre-train a pair of image and text encoders, leading to semantic meaningful visual representations
in harmony with manually-designed text prompts. To our best
knowledge, CLIP-ReID [8] is the first milestone that deals
with ReID tasks based on CLIP. By tuning identity-specific
text prompts [11], it uses text representations generated by a
powerful text encoder [9] to distill the image encoder, leading
to more discriminative global features. However, the local
part features still lack meaningful semantics under the only
supervision of global identity text embeddings.
III. M ETHODOLOGY
A. Preliminaries: Overview of CLIP-ReID
CLIP-ReID [8] is the first milestone that applies visuallanguage pre-training models to appearance-based ReID. As
one-hot identity labels used by ReID are meaningless to
construct high-quality text prompts, it proposes a two-stage
training procedure for visual-language learning.
The first training stage aims to optimize identityspecific text tokens with CLIP-style supervisions. By passing a pre-designed text description TD = ‘A photo of a
[X]1 [X]2 [X]3 ...[X]M person.’ and the corresponding person
image Ii through a frozen text encoder T (·) and a frozen
image encoder I(·) respectively, a text embedding Tyi and an
image embedding Vyi could be obtained, where [X] denotes
a learnable text token with the word embedding dimension,
M is the number of learnable text tokens, and yi indicates the person identity label. Then, CLIP-style contrastive
learning losses Li2t and Lt2i [9] are computed to optimize
[X]1 [X]2 [X]3 ...[X]M :
Lstage1 = Li2t + Lt2i .

(1)

In the second training stage, the learned identity-specific text
embeddings are treated as a classifier, and the image encoder
I(·) is fully optimized under the supervision of identity loss
Lid with label smoothing and triplet loss Ltri [2]:
Li2tce =

N
X

exp (Vi · Tyk )
−qk log PN
ya =1 exp (Vi · Tya )
k=1

Lstage2 = Lid + Ltri + Li2tce ,

(2)
(3)

where qk represents the soft label in the target distribution, N
is the number of identities, and i denotes the image index.

Head
Body
Shoes
(a) CLIP-ReID

Head
Body
Shoes
(b) Ours

But unlike [8], we reformulate the identity text prompt
at the pixel level using parsing maps, thereby generating fine-grained text prompts Tipart , such as, ‘A photo
of a [X]1 [X]2 [X]3 ...[X]M person’s head.’, ‘A photo of a
[X]1 [X]2 [X]3 ...[X]M person’s shoes.’. Then we can obtain
by passing fine-grained
a fine-grained text embedding tpart
i
text prompts Tipart through the tokenizer T :

Fig. 2. Illustration of within-part semantic inconsistency. Colors indicate
different body parts, symbols denote human identities, and the red dashed
line represents the decision boundary for identity recognition.

B. The Within-part Semantic Inconsistency Issue
CLIP-ReID is simple and effective, yet to be improved.
Since the natural language supervision is limited to the
whole-body scale (Eq.(2)), some non-salient or infrequent
part features can be easily overwhelmed, and still lack highlevel semantics [4]. As shown in Fig.2, we use t-SNE [18]
to visualize the pixel-level feature distributions produced by
CLIP-ReID, and adopt human parsing [13] to assign semantic
labels for each pixel. Here, colors indicate different body parts,
while symbols denote human identities.
We observe that although the decision boundary of identities
(the red dashed line) is generally clear, features of different
body parts are hard to distinguish. Furthermore, for several
confusing identities (e.g., identities denoted as crosses and
circles), the classification boundary of their part features is
even more difficult to be recognized. We term this issue as
within-part semantic inconsistency, which directly hinders the
performance of person retrieval.
C. Part-Informed Prompt Tuning
To address the issue of within-part semantic inconsistency,
an intuitive approach is to aggregate part features that share
the same semantics while separating those that are irrelevant.
This intuition, however, further raises two questions: (1) How
to identify the semantics of fine-grained part features? (2) How
to design the supervision signal for part distinction?
The first question has already been answered by state-of-theart human parsing models, which are robust to the ambiguous
boundaries between different body parts. Thus, we employ a
human parsing model H [13] to generate a pixel-level parsing
map P for person image Ii . Specifically, we follow the setup
from [13] and classify each pixel into N (N=20) semantic
categories, including ‘Background’, ‘Hat’, ‘Hair’, etc (see the
appendix for details). This allows us to generate per-pixel text
prompts based on the semantic labels and CLIP text encoder.
However, human parsing inherently introduces a new obstacle in addressing the second question. That is, human parsing only distinguishes identity-agnostic body part semantics,
whereas ReID requires learning identity-specific discriminative
cues. This conflict can suppress the diversity and discriminability of ReID features to some extent, leading to inferior
performance. Inspired by [11], we propose a part-informed
prompt tuning strategy to solve the supervision conflict issue.
As illustrated in Fig.3(a), similar to [8], we first learn identityspecific tokens with the text prompt Ti , i.e., ‘A photo of a
[X]1 [X]2 [X]3 ...[X]M person’, through optimizing Eq.(1).

tpart
= T (Tipart ).
i

(4)

Then, we align the spatial resolution of visual feature maps
and parsing map via downsampling, and rearrange the finegrained text embedding based on the spital arrangement of
parsing maps, leading to a ‘text embedding map’ (see appendix
for details), i.e.,
) tfi ull ∈ RH×W ,
tfi ull = rearrange(tpart
i

(5)

Next, we propose to learn our part-informed text prompts with
pixel-level dense contrastive learning. Specifically, let vif ull
denote the visual feature map extracted by the visual encoder,
we treat pixel-wise text embedding tji ∈ tfi ull and visual
embeddings vij ∈ vif ull of the j-th semantic label of the ij
th person (tji , vi+
), (vij , tji+ ) as positive pairs, while counting
j
the others as negative pairs (tji , vi−
), (vij , tji− ):

N
j
exp tji · vi+
/τ
1 X
log
,
 P
j
j
N j=1
/τ
exp tji · vi+
+ vj exp tji · vi−
i−

N
(6)
X
exp vij · tji+ /τ
Lpart
=
log

,
P
i2t
j
j
j
j
exp vi · ti+ + tj exp vi · ti− /τ
j=1

Lpart
t2i =

i−

part
Lpart = Lpart
t2i + Li2t ,

where τ is the temperature coefficient of the InfoNCE loss.
During the prompt tuning process, only the learnable text
tokens [X]1 [X]2 [X]3 ...[X]M are optimized, while the image
and text encoders are frozen (see the appendix for details).
Therefore, the loss function of the first training stage can
be defined as:
Lstage1′ = Lstage1 + Lpart ,

(7)

D. Part-Informed Visual-Language ReID
With part-informed prompt tuning, we are able to generate
identity-specific text embeddings with discriminative body
semantics. However, for image-text alignment, two issues
remain: 1) Which resolution of visual features is suitable for
supervision by language signals and 2) How to make the
alignment process robust against inevitable noises generated
by off-the-shelf parsing models?
For the first issue, downsampling operations can easily lead
to information loss, causing supervision signals to become inaccurate for small-scale features. Therefore, directly imposing
language supervision on all intermediate visual feature maps
is not advisable. To address this, we introduce an auxiliary
head that fuses multi-scale visual features to a relatively higher
resolution. This allows both low- and high-level visual features
to directly receive gradients from language guidance, thereby


exec
/bin/zsh -lc "pdftotext -l 3 'ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ESTI: An Efficient Spatial-Temporal Interaction
Network For Video-Based Person Re-Identification
2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11209571

Guquan Jing1,2 , Peng Gao1,2 , Yiyang Hu1,2 , Yujian Lee1,2 , Hui Zhang2 ∗
1

2

Department of Computer Science, Hong Kong Baptist University, Hong Kong, China
Department of Computer Science, Beijing Normal-Hong Kong Baptist University, Guangdong, China
{guquanjing1, gaopeng1225, yiyanghu124, yujianlee1119}@gmail.com, amyzhang@uic.edu.cn

Abstract—Video-based person re-identification (Re-ID) aims
to identify the target pedestrian from video sequences. However, redundant information exist in input frames. Extracting
spatial-temporal features in whole adjacent frames can introduce
additional computational overhead. Furthermore, this process
leads to the loss of critical spatial and temporal details, causing
suboptimal representations. To mitigate these issues, we propose
an Efficient Spatial-Temporal Interaction (ESTI) network, which
processes half of the input sequence separately through spatial
and temporal branches, extracting high-level discriminative features across multiple layers and avoiding redundancy computations. In particular, we propose a Feature Enhancement Module
(FEM) for the spatial branch to focus on enhancing spatial dependencies adaptively, and a Temporal Interaction Module (TIM)
for temporal branch to capture temporal correlations effectively.
Spatial-temporal interaction is performed at the final layer to
generate distinctive representations. Extensive experiments on
three challenging video Re-ID datasets show that our ESTI
achieves competitive results while maintaining low computational
complexity.
Index Terms—Video-based person re-identification, spatialtemporal information

I. I NTRODUCTION
Video-based person Re-Identification (Re-ID) [1]–[4] aims
to identify the same pedestrian from video sequences across
non-overlapping cameras, which is a crucial task in intelligent
surveillance and video retrieval. Different from the imagebased Re-ID that relies on single-shot images, video-based
Re-ID offers richer spatial-temporal information. Efficiently
leveraging these spatial-temporal cues is significant for achieving robust performance.
Early methods to video Re-ID adapt models from other
video tasks, such as 3D CNN [5] and RNN [6], to learn
video temporal information, which are not suitable for video
Re-ID. Recent studies [1]–[4], [7]–[16] first extract framelevel features, then aggregate them temporally to learn spatialtemporal representations. However, these methods face significant limitations due to the redundancy within video sequences.
Figure 1 shows two sampled sequences through the Restricted
Random Sampling (RRS) strategy [7] from the iLIDS-VID
∗ Corresponding author.
This work is supported in part by the Natural Science Foundation of
China (62076029); in part by the National Key R&D Program of China
(2022YFE0201400); in part by the Guangdong Provincial Key Laboratory
of Interdisciplinary Research and Application for Data Science, BNU-HKBU
United International College (2022B1212010006).

Spatial Feature

Discriminative
Spatial-temporal Representation

Temporal Feature

Spatial-temporal Representation
Multi-Layer
Enhancement

Spatial Feature

Temporal Feature

Multi-Layer
Interaction

Spatial-Temporal Module
Similar Feature Representations

Similar Frames
Fig. 1. Two sampled sequences from iLIDS-VID (the image sequence above)
and MARS (the image sequence below) dataset with previous video Re-ID
methods (black boxes and lines) and our method (red boxes and lines).

[17] and the MARS [18] datasets. Input frames in video ReID often exhibit minimal variation, as pedestrians perform
limited actions with subtle frame differences. Recent methods
aggregate features across such whole adjacent frames, leading
to computational inefficiency due to redundant computations
that contribute minimally to the overall representation but
still require full processing. Moreover, existing aggregation
strategies on similar frames often cause spatial and temporal
information loss. As shown in Figure 1, recent methods tend to
aggregate similar features temporally, causing an overemphasis
on redundant areas while ignoring critical local regions. These
inefficiencies dilute key features and restrict the generation of
high-level spatial and temporal representations.
To tackle aforementioned issues, we propose an Efficient Spatial-Temporal Interaction (ESTI) network for videobased person Re-ID (see Figure 1 for a conceptual illustration). Specifically, the network divides input frames into
two branches: a spatial branch and a temporal branch, each
processing half of the sequence to extract high-level spatial
and temporal features, respectively. This half-sequence extraction strategy balances computational efficiency and pedestrian

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:45 UTC from IEEE Xplore. Restrictions apply.

representation quality through capturing comprehensive spatial
and temporal information without over-processing spatialtemporal aggregation of similar content. Inspired by [3], a
pyramid structure is adopted to generate distinctive features
across layers. Our design significantly reduces computational
cost and mitigates redundancy often introduced by the spatialtemporal aggregation across whole adjacent frames. By extracting critical spatial and temporal features from separate
halves of the input sequence and performing spatial-temporal
interaction at the final stage, the ESTI maintains efficiency
while preserving crucial details. To obtain high-level spatial
and temporal dependencies, we propose the Feature Enhancement Module (FEM) for the spatial branch and the Temporal
Interaction Module (TIM) for the temporal branch. The FEM
emphasizes spatial cues and local details, while the TIM
models temporal correlations across adjacent features to capture both local and global dynamics. Both modules leverage
deformable attention [19] to adaptively extract crucial pedestrian information while reducing computational overhead. To
facilitate effective training, a Feature Aggregation (FA) module
is designed to generate pedestrian representations at each layer
under supervision for feature refinement. Consequently, highlevel spatial and temporal features are aggregated via the
Spatial-Temporal Interaction Module (STIM), which shares
a similar structure with the TIM. By incorporating these
components, the ESTI efficiently captures the interaction of
spatial and temporal information, generating discriminative
video-level representations.
The contributions of this paper can be summarized as
follows:
• We propose an Efficient Spatial-Temporal Interaction
(ESTI) Network, incorporating spatial and temporal
branches to minimize redundancy and efficiently obtain
high-level spatial-temporal dependencies.
• We propose a Feature Enhancement Module (FEM) to enhance spatial features and a Temporal Interaction Module
(TIM) to capture temporal correlations. The high-level
spatial and temporal features are generated for further
spatial-temporal interaction. A Feature Aggregation (FA)
module is designed to create pedestrian representations
at each layer under supervision, facilitating effective
training.
• Extensive experiments on three challenging video Re-ID
datasets demonstrate that our network achieve a competitive performance with a low computational cost compared
to state-of-the-art methods.
II. R ELATED W ORKS
Video-based person re-identification (Re-ID) aims to retrieve the target pedestrian from video sequences. Early methods primarily utilize models for other video tasks, such as
RNN [6] and 3D CNN [5], to directly model temporal information. For example, Eom et al. [6] exploit RNNs to encode a
sequence temporally, enabling access to the temporal memory.
Gu et al. [5] propose a network that uses 3D convolutions
to model temporal information while preserving appearance

information. However, these models are not suitable for video
Re-ID task as they are not designed specifically for this
domain and introduce a mass of parameters. Recent methods
[1]–[4], [7]–[16] obtain features in each frame, subsequently
aggregate multi-frame features. For instance, Wang et al.
[3] propose a feature aggregation framework with a pyramid
structure to aggregate frame-level features temporally. Wu et
al. [16] enhance extracted features based on the pedestrian
relative state before aggregating them. Despite the remarkable
progress achieved, these methods fail to mitigate the impact
of redundant information in video Re-ID data, which can lead
to computational overhead and critical information loss.
III. M ETHOD
A. overview
As shown in Figure 2 (left), the overall structure of our
proposed network consists of a spatial branch and a temporal
branch. Specifically, given a video tracklet with T frames V =
{It }Tt=1 , it is first fed into a backbone network (e.g., ResNet50 [20]) to extract frame-level features F 0 = {Ft0 }Tt=1 ,
where Ft0 ∈ RC×H×W . We separate these features along the
time axis, with the spatial branch focusing on the first half
{F10 ...F T0 } to enhance spatial dependencies and the temporal
2
branch processing the rest {F T0 +1 ...FT0 } to capture temporal
2
correlations. Inspired by [3], we utilize a pyramid structure to
extract multi-layer features while mitigating irrelevant information. For each branch, the proposed Feature Enhancement
Module (FEM) and Temporal Interaction Module (TIM) are
applied at each layer to extract distinctive spatial and temporal
features F n = {Fln }L
l=1 , where n denotes the index of layer
starts from 1, and L = 2Tn . At the final layer, the SpatialTemporal Interaction Module (STIM) integrates the high-level
spatial and temporal features, capturing comprehensive spatialtemporal correlations. Similar to [3], the ESTI supervises
multi-layer features for effective learning. However, we enhance this process with a Feature Aggregation (FA) module,
which combines features across layers for discriminative representations. The details are presented below.
B. Spatial-Temporal Interaction
1) Spatial Branch: We extract spatial features from half of
the input sequence to focus on discriminative spatial representations. To achieve this, we propose the Feature Enhancement
Module (FEM), which emphasizes the target pedestrian in the
spatial domain using deformable attention [19], as shown in
Figure 2 (right). In detail, given the feature map Ft of the
t-th feature in the spatial branch, we first generate embedded
features Qt and Vt through embedding layers θs and θs′ , and a
linear layer φs . A set of 2-D reference points rt is derived from
Qt via a linear projection φr , indicating sampled positions on
Vt . We can express these as:
Qt = θs (Ft ), Vt = φs (θs′ (Ft )), rt = φr (Qt ).

(1)

Afterward, we define j index both the feature Qt and reference
t
points rt . The learnable offset ∆rmjk
for each reference point

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:45 UTC from IEEE Xplore. Restrictions apply.

2-th Layer
1-th Frame

TIM

Temporal Interaction Module

FEM

Feature Enhancement Module

STIM

Spatial-temporal Interaction Module

AG

0.1 0.4 0.3 0.2

Reference Point
Addition

FA
AG

FA

Feature Aggregation

Aggregation

Residual
RCB Connection Block

The Spatial-Temporal Feature

Layer-2
FA

RCB

STIM
0.1 0.4 0.3 0.2
Layer-1

AG

FEM

TIM

FA

Feature Enhancement
Module (FEM)

Layer-0

FEM

FEM

TIM

TIM
AG
0 0.5 0.3 0.2

RCB

Temporal Branch

Spatial Branch
Image Backbone

0.1 0.4 0.3 0.2
AG

Temporal Interaction
Module (TIM)

Input Frames

Fig. 2. The overall architecture of our proposed ESTI (left) and the illustration of the Feature Enhancement Module (FEM) and the Temporal Interaction
Module (TIM) (right). In the overall architecture (left), we use eight frames (T = 8) as an example.

rtj , and attention weight Atmjk (j-th Qt to k-th Vt at m-th
head) are acquired from two linear projections (θo and θa ) and
a softmax function to search local crucial positions around rtj
and generate distinctive features. These can be expressed as:
t
∆rmjk
= θo (Qjt ), Atmjk = softmax(θa (Qjt )).

(2)

Subsequently, the sampled positions Vtk are aggregated with
the attention weight Atmjk . The enhancement process E(·) is
formulated as:
E(Qjt , rtj , Vt ) =
M
X

Wm

m=1

"K
X

′
t
Atmjk · Wm Vt (rtj + ∆rmjk
)

k=1

#

,

(3)

Qt+1 = θt+1 (Ft+1 ), Vt = φt (θt′ (Ft )),

′

where Wm and Wm are learnable weights. M and K are the
total attention head and sampling point number, respectively.
Therefore, an enhanced feature map F̂tS in the spatial branch
is obtained by implementing enhancement process to Qt ,
followed by a shortcut connection to the initial feature map
Ft . This processing enables the FEM to adaptively focus on
local relevant regions, thereby enhancing the discriminative
capacity of spatial features. Eventually, we aggregate the
adjacent feature maps by applying the element-wise addition
and a Residual Connection Block (RCB) [20]. The feature for
the input of next layer FS in the spatial branch is calculated
by:
S
FS = RCB(F̂tS + F̂t+1
).

By leveraging these enhancements, the FEM ensures that our
network captures critical spatial details, contributing to the
generation of discriminative spatial representations.
2) Temporal Branch: For the remaining feature maps in
the input sequence, we apply the Temporal Interaction Module
(TIM) to capture temporal correlations, as shown in Figure 2
(right). Similar to the FEM, the TIM utilizes the deformable
attention [19] to efficiently model temporal dependencies.
Differently, the TIM focuses on interaction between adjacent
feature maps. These feature maps inquire each other to generate distinctive one. Formally, given adjacent feature maps
Ft and Ft+1 in the temporal branch, the temporal interaction
process I(·) that inquires the temporal information in Ft using
Ft+1 can be formulated as:

(4)

rt+1 = φt+1
r (Qt+1 ),
j
I(Qjt+1 , rt+1
, Vt ) =
"K
#
M
X
X
′
j
t+1
t+1
Amjk · Wm Vt (rt+1 + ∆rmjk ) ,
Wm
m=1

(5)

(6)

k=1

where θt′ and θt+1 are embedding layers. φt and φt+1
are
r
t+1
linear layers. The sampling offsets ∆rmjk
and the attention
j
weight At+1
mjk are generated from Qt+1 . This allows our
network to establish temporal correlations between adjacent
frames. We also inquires the temporal information in Ft+1
using Ft . By performing mutual inquiry between Ft and Ft+1
during the interaction and incorporating a shortcut connection

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:50:45 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'False Negatives Consensus Suppression for Text-to-Image Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11209807

False Negatives Consensus Suppression for
Text-to-Image Person Re-identificatio
Ruigeng Zeng1,2,3 , Wentao Ma4,* , Qinglin Wang1,2,3* , Xinjun Mao3 and Jie Liu1,2,3
1

2

Laboratory of Digitizing Software for Frontier Equipment, National University of Defense Technology, Changsha, China
National Key Laboratory of Parallel and Distributed Computing, National University of Defense Technology, Changsha, China
3
College of Computer Science and Technology, National University of Defense Technology, Changsha, China
4
School of Information and Artificial Intelligence, Anhui Agricultural University, Hefei, China

Abstract—Text-Image Person Re-identification (TIReID) aims
to retrieve the relevant pedestrian images according to the
given textual query. Recent methods typically achieve this goal
through image-text contrastive learning, which assumes that
only paired images and texts from the same pedestrian are
considered positive samples. However, we observe that there
exist negative samples, termed false negatives, that are highly
semantically related to the anchor in practice. Training with
these false negatives may adversely affect feature representation
learning and semantic alignment between modalities. This work
proposed a false negative detection and suppression (FNDS)
method to mitigate their adverse impact. Our FNCD consists
of a False Negative Consensus Detection (FNCD) mechanism
and an Adaptive False Negative Suppression (AFNS) method.
FNCD combines dual-grained detection to consensually identify
potential false negatives, while AFNS assigns adaptive weights
to the false negative similarities for more robust suppression.
Extensive experiments conducted on three public benchmark
datasets demonstrate the effectiveness of the proposed method.
Index Terms—Text-image person Re-identification, false negative, cross-modal contrastive learning

I. I NTRODUCTION
Text-Image Person Re-identification (TIReID) [1], a subtask of Person Re-identification (ReID), aims to retrieve the
most semantically related pedestrian images from a large
candidate gallery based on the given text query. Due to its
practical relevance in the fields of public safety and smart
cities, TI-ReID has garnered increasing attention in recent
years. However, TIReID remains a challenging task as it
requires fine-grained feature representation of pedestrians’
complex semantic visual and textual information, as well as
accurate visual-textual alignment.
To tackle these challenges, most previous works adopt
Visual Semantic Embedding (VSE) methods to learn the
correspondence between the image and text modalities. These
methods [2]–[5] generally follow a common model structure: “image/text encoders + feature embedding”. In this
framework, image/text features are first extracted using the
respective encoders, and then these features are embedded
(model-specific) into a shared latent space for cross-modal
alignment. Image/text encoders typically use single-modality
* Corresponding authors(email: wtma@ahau.edu.cn, wangqinglin@
nudt.edu.cn).
Code is available at https://github.com/Ray-Zhen/FNDS.

Anchor: A man has his head bent down with arms at his sides. His
right leg is extended behind him. He wears a black, short-sleeve top,
blue shorts, ending below the knees, and dark sandals.

attract

Positive

repel

repel

repel

False Negative True Negative True Negative

Fig. 1: The illustration of false negatives. There exist negative
samples that share the same semantics with the anchor due to
the data noise. Repeling such false negative samples from the
anchor harms the representation learning.

networks initialized with pre-trained models (e.g., ViT [6]
on ImageNet for the image encoder and BERT [7] for the
text encoder) to facilitate cross-modal learning. The most
recent works use the Contrastive Language-Image Pre-training
(CLIP) model [8], pre-trained on large image-text datasets,
as image/text encoders to leverage multi-modal semantic correspondence, greatly enhancing retrieval performance over
single-modal methods.
Most TIReID methods employ image-text contrastive learning method to establish semantic correspondences between
image and text modalities. Specifically, considering the query
text as anchor, the paired person image is viewed as positive, while all other images in the mini-batch are treated
as negative samples. The optimization objective is to bring
the positive samples closer to the anchor while pushing the
negative samples farther away in the shared latent space.
Although image-text contrastive learning has demonstrated
impressive performance, these methods often overlook the
semantic relationships between image-text pairs of each individual. We observe that some negative samples share the
same semantic concept as the anchor, due to incorrect person
ID labeling and the semantic diversity of text descriptions,
where a single description may refer to multiple pedestrians.
A typical example is shown in Figure 1. During training, these
negative samples, defined as false negatives in this paper,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:31 UTC from IEEE Xplore. Restrictions apply.

are still treated as true negatives and pushed away from the
anchor, contradicting the optimization objective and adversely
impacting cross-modal representation learning and alignment.
As far as we know, there is litter work has been devoted
to studying the false negative problem in TIReID, and the
closest related work is in the task of image-text matching [9],
[10]. However, these works may be inefficient or ineffective in
TIReID because they rely solely on global image-text features
for false negative detection, neglecting fine-grained local cues,
where subtle differences in local details often exist between
pedestrians.
To address the false negative problem in TIReID, we
propose a false negative detection and suppression framework,
called FNDS, to identify potential false negative samples and
mitigate their adverse impact. Our FNDS consists of a False
Negative Consensus Detection (FNCD) mechanism and an
Adaptive False Negative Suppression (AFNS) method. FNCD
fuses dual-grained detection to consensually identify potential false negative samples, thereby providing more reliable
negative samples for robust contrastive learning. Specifically,
we leverage the similarity distribution differences between
false negatives and true negatives to screen out potential
false negatives. To enrich feature granularity, we propose a
dual-representation method that combines coarse-grained basic
global representation (BGR) with fine-grained token selection
representation (TSR) to enhance cross-modal correspondence
comprehensively. Based on this, we conduct consensus false
negative detection for more robust identification. The proposed
AFNS assigns adaptive weights, adjusted based on false negative confidence, to the similarities between the anchor and
detected false negative samples, enabling more robust false
negative suppression. Moreover, the Cross-modal Momentum
Contrastive (CM-MoC) module is introduced for a more
accurate estimation of false negative similarity distribution.
The main contribution can be summarized as follows: (1) We
reveal and investigate the inevitable false negative problem
in TIReID. We propose a FNDS framework to suppress the
adverse impact of false negatives through the False Negative Consensus Detection (FNCD) mechanism and Adaptive
False Negative Suppression (AFNS) method. (2) We introduce
Cross-modal Momentum Contrastive (CM-MoC) to expand
the training data in each training epoch, enabling a more accurate estimation of false negative similarities. (3) We conducted
extensive experiments on three widely used datasets: CUHKPEDES, ICFG-PEDES, and RSTPReid. The comprehensive
results demonstrate that our method surpasses all current stateof-the-art approaches, confirming its effectiveness.
II. R ELATED WORK
A. Text-image Person Re-identification
TIReID, first introduced by [1], is a subtask of crossmodal retrieval with challenges in fine-grained alignment
due to intra- and inter-modal variations. According to the
alignment strategy, the existing approaches can be generally
classified into two categories: the global-matching methods

and local-matching methods. Global-matching methods focus on designing models or objective functions to learn
image-text correspondence within a shared latent space [1],
[11]. However, global-matching methods focus exclusively
on global-level feature representation, neglecting informative
local details, which hinders performance improvements. To
address this limitation, local-matching methods [2]–[4], [12]–
[14] have been proposed to capture fine-grained local crossmodal alignment between visual scenes and text descriptions.
Recently, CLIP [8], a landmark in visual-language pre-training
(VLP), has garnered remarkable success owing to its robust and comprehensive multi-modal representations. Consequently, numerous studies [12], [13] have integrated CLIP into
TIReID to improve cross-modal representation and alignment.
For instance, [12] introduced a CLIP-driven framework to
extract fine-grained visual information, while [13] utilized
both the visual and language encoders of CLIP to capture implicit fine-grained cross-modal relations. In this paper, we do
not aim to design elaborate cross-modal alignment strategies
or introduce powerful backbone networks. Instead, we focus
on addressing the inevitable and challenging false negative
problem in TIReID.
B. Learning with False Negative
Research on false negatives, a crucial issue in noisy data
tasks, has gained increasing attention in fields such as imagetext matching [9], [10], graph representation learning [15], [16]
and sound source localization [17]. To tackle the false negative
challenge, many approaches have been introduced, which can
generally be grouped into robust loss function methods [16],
[18] and sample selection methods [9], [10], [19]. The former
approaches focus on developing loss functions that are tolerant
to noise, aiming to reduce the negative impact of false negative
samples. In contrast, the sample selection approaches focus
on formulating effective techniques to identify false negative
samples and mitigate their impact. The methods mentioned
above have made considerable advancements in various tasks.
However, they are not specifically tailored for TIReID. Therefore, in this study, we propose a novel approach to mitigate
false negatives and tackle the false negative challenge in
TIReID.
III. T HE P ROPOSED M ETHOD
A. Feature Representation
Image-Text Feature Representation. Following the previous work [13], we adopt the CLIP backbone to extract imagetext features. For image feature representation, given an input
image Ii ∈ Rc×w×h , we adopt the visual encoder of CLIP
to extract token feature sequence fiv = {vgi , v1i , v2i , ..., vni v }
, with a total length of nv + 1. Where, vg is the image-level
v
global feature of [CLS] token, {vji }nj=1
is the patch-level local
features. For Text Feature Representation, given the input text
Ti , we obtain the text features fit = {tisos , ti1 , ti2 , ..., tint , tieos }
with textual encoder of CLIP. Where, nt denotes the text token
length, tisos and tieos are the specific token features for the
[SOS] and [EOS] tokens, respectively, tieos serve as text-level

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:31 UTC from IEEE Xplore. Restrictions apply.

negative
similarities

...
vg

FNDS

negative samples

true negatives

false negatives

tg

...

Text
Encoder

(a) The Model Framework

Token Selection

Momentum
Encoder

She is wearing a white
shirt and denim shorts
and black tennis shoes.

Momentum
Encoder

Token Selection

Image
Encoder

vtsr

GMM with TSR similarities
Component 1
Component 2
Mixture PDF

GMM with BGR similarities
Component 1
Component 2
Mixture PDF

−

ttsr

False Negative Consensus Detection
positive
+
+
−

−

+

+
−
−

ρ
+

+

−

−

true negative

−

false negative

similarity

Adaptive False Negative Suppression
(b) False Negative Detection and Suppression

Fig. 2: The overview of our method. (a) is the illustration
of the model framework, which consists of token selection
representation (TSR) module, False Negative Detection and
Suppression (FNCD) and Cross-modal Momentum Contrastive
(CM-MoC) module. (b) illustrates the core of FNDS, which
consists of False Negative Consensus Detection (FNCD)
mechanism and Adaptive False Negative Suppression (AFNS)
method.
t
global feature of tig , and {tij }nj=1
represent the word-level local

features.
Dual Representation. Most previous TIReID works [13],
[20], [21] only adopt vgi and tig token features as basic global
representation (BGR) for cross-modal alignment. However, using only BGR may overlook fine-grained intra-modal semantics, hindering detailed cross-modal alignment. To overcome
this, we aggregate local features from image patches and word
tokens for fine-grained feature representations.
Specifically, inspired by the previous works [12], [22], we
select and transform informative local features to generate finegrained token selection representation (TSR). In practice, take
the visual process for example, we first obtain the attention
map Avi ∈ R(1+nv )×(1+nv ) from the last transformer block
of the image encoder, which reflects the importance scores
between 1 + nv tokens. Then the correlation weight between
[CLS] token and local tokens avi = Avi [0, 1 :] ∈ Rnv are
used to select the top-K informative local token features fˆiv =
{vki v , vki v , ..., vki nv }, where knv = R × nv denotes the indices of
1
2
the selected local tokens and R is the token selection ratio. In
terms of textual procedure, we obtain the selected local text
features in a similar way as fˆit = {tikt , tikt , ..., tikt }. Finally,
n
1
2
the selected local visual and textual token features are linearly
transformed and aggregated via:
titsr = M axP ool(σ(BN (W1t fˆit ) + W2t fˆit ))
(1)
i
vtsr
= M axP ool(σ(BN (W1v fˆiv ) + W2v fˆiv ))
i
where vtsr
and titsr is the TSR for image and text,
M axP ool(·) is the max-pooling function, σ(·) is the ReLU
activation function, BN (·) is the batch normalization, and W
denotes the linear transformation parameter.

B. False Negative Consensus Detection
To alleviate the adverse impact of false negatives, the
primary challenge is to identify and remove the potential false

negative samples during training. Intuitively, false negative
samples are negative samples that exhibit high semantic similarity to positive samples, which can be treated as anomalies among negative samples. Building on this observation,
we exploit the differences in similarity distributions between
false negatives and true negatives to identify potential false
negatives. To this end, we employ a two-component Gaussian
Mixture Model (GMM) to fit the similarity distributions of
negative samples within the current mini-batch. Specifically,
we first compute the cosine similarity set S ∈ RB×B across
all image-text feature pairs {vi , ti }B
i=1 in a mini-batch:
B
S(vi , ti )|B
i=1 = {sim(vk , tl )}k,l=1

(2)

Where sim(vk , tl ) = vk tTl /||vk ||||tl ||, B is the batch size.
Then we collect the negative pair similarities, which are
B(1−B)
denoted as S − = {sim(vk , tl )}B
. At last,
k,l=1,k̸=l ∈ R
we fit the negative pair similarities by using the GMM to find
out the potential false negative samples:
p(s− | θ) =

2
X

βk ϕ(s− | k)

(3)

k=1

where βk is the mixture coefficient and ϕ(s− | k) is the
probability density of negative pair similarities of the k-th
component. We adopt the Expectation Maximization algorithm
to optimize the GMM. Subsequently, the posterior probability
−
p(k|s−
i ) = p(k)p(si |k) of i-th negative pair, as either a false
negative or true negative, can be computed, where k ∈ {0, 1},
k = 0 indicates a true negative with a lower similarity mean,
and k = 1 indicates a false negative with a higher similarity
mean. By setting a threshold δ for p(k = 1|s−
i ), the negative
pairs in the current mini-batch can be divided into true negative
and false negative subsets:
−
−
−
S f n = {s−
i |p(k = 1|si ) ≥ δ, ∀si ∈ S }
−
−
−
S tn = {s−
i |p(k = 0|si ) < δ, ∀si ∈ S }

(4)

where S f n denotes the false negative set, S tn denotes the
true negative set, and S = S tn ∪ S f n . Note that we adopt
both BGR and TSR for false negative identification, denoting
fn
fn
their respective false negative sets as Sbgr
and Stsr
. To obtain
a more reliable false negative set, we propose exploiting the
fn
fn
consistency between Sbgr
and Stsr
to identify their consensus
as the final false negative set. The process can be described
as follows:
tn
tn
Ŝ tn = Sbgr
∩ Stsr
fn
fn
Ŝ f n = Sbgr
∩ Stsr
u

S = S − (Ŝ

tn

(5)
tn

∪ Ŝ )

where, Ŝ tn is the final consensus true negative set, Ŝ f n is the
final consensus false negative set, S u is the uncertain set. For
the negative samples in uncertain set S u , we randomly labeled
them as true negative or false negative.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:31 UTC from IEEE Xplore. Restrictions apply.


codex
第二批也看完了。这里的有价值信号是两类：一类把“冗余帧、冗余 token”变成核心问题，另一类把“训练监督本身有冲突或假负样本”变成核心问题，这比单纯加注意力更容易讲出论文级故事。
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
/bin/zsh -lc "pdftotext -l 3 'FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025

1801

FLAG: A Framework With Explicit Learning Based
on Appearance and Gait for Video-Based
Clothes-Changing Person Re-Identification
Hengjie Lu , Yilin Gao , and Shugong Xu , Fellow, IEEE
Abstract— Person re-identification (ReID) aims to search
for the target person among the non-overlapping surveillance
cameras. Video-based clothes-changing person re-identification
(VCC-ReID) has become an essential branch of ReID due to
the rich spatial and temporal information in the videos and
the broad application of the scenarios. Appearance and gait are
discriminative features in the video-based ReID, but appearance
information is limited due to the clothes changing, which makes
the VCC-ReID challenging. To solve this challenge, we propose a
Framework with explicit Learning based on Appearance and Gait
(FLAG), which can explicitly extract two corresponding types
of information and be combined with most existing video-based
ReID methods. The FLAG includes a multi-modal and multigranularities Architecture (MGA), which is a large model,
and a Cross-Modal Knowledge Distillation Scheme (CMKDS),
which has a small model. They can be applied to devices
with different computing resources. The MGA is designed to
simultaneously take the visible light and silhouette modalities
as input to explicitly learn the appearance and gait features,
respectively. The silhouette modalities are composed of several
levels of granularities to model global and local gait features and
independently serve as input for MGA. The Embedding-Based
parallel fusion module is proposed to fuse the appearance and
multi-granularities gait feature efficiently. The CMKDS is present
to distill the MGA to a small single-modal model that only uses
the visible light modality as input. The Embedding-Based direct
and indirect distillation strategies are designed in the CMKDS.
Experimental results demonstrate that the FLAG combined with
the existing video-based ReID methods can significantly improve
their performance. In addition, when FLAG is combined with
the AP3D method, the MGA can outperform state-of-the-art
accuracy by 4.2%.
Index Terms— Video-based person re-identification, clotheschanging person re-identification, multi-modal learning, knowledge distillation.

I. I NTRODUCTION

R

ECENTLY, Person Re-Identification (ReID), which aims
to match the same person from multiple non-overlapping
cameras, has become a popular research area because of its
wide application, such as intelligent surveillance, criminal
Received 12 April 2024; revised 11 September 2024; accepted 14 October
2024. Date of publication 18 October 2024; date of current version
13 February 2025. This work was supported in part by the National High
Quality Program under Grant TC220H07D, in part by the National Key
Research and Development Program of China under Grant 2022YFB2902002,
and in part by the Innovation Program of Shanghai Municipal Science and
Technology Commission under Grant 20511106603. This article was recommended by Associate Editor J. Shen. (Corresponding author: Shugong Xu.)
The authors are with the School of Communication and Information Engineering, Shanghai University, Shanghai 200444, China (e-mail:
luhengjie@shu.edu.cn; gaoyilin@shu.edu.cn; shugong@shu.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2024.3483265

investigation, and so on. With the rise of deep learning
techniques [1], [2], [3], [4], significant progress have been
achieved in ReID [5], [6], [7], [8], [9], [10]. Compared with
image-based ReID, video-based ReID, which belongs to the
video analysis, can perform better due to the rich spatial and
temporal information in the video sequences. Clothes changing
is a common scenario when we want to re-identify over a long
period of time. Besides, clothes changing also exists in some
short-time ReID cases, e.g., the clothes changing caused by a
criminal or hot weather. Overall, video-based clothes-changing
person re-identification (VCC-ReID) is crucial in computer
vision.
Most researchers focus on the video-based ReID with the
same-clothes setting instead of the clothes-changing setting
due to its significant challenges. Specifically, video-based
ReID mainly relies on the appearance and gait (the way
of walking) information of pedestrians, and the appearance
information is dominated by the clothes from the surveillance
camera. Clothes changing will result in limited appearance
information, which makes the VCC-ReID become a challenging task. In this challenging VCC-ReID, the core is how to
mine the limited remaining appearance information (e.g., the
human face) and the gait information. Although some existing
video-based ReID methods can simultaneously extract spatial
and temporal information in video sequences, decoupling and
extracting appearance and gait information still needs to be
improved.
Gait recognition is also a technique for identifying pedestrians, which typically takes silhouettes as input. As shown in
Fig. 1, the silhouettes are segmented by the semantic segmentation model from the visible light images. The silhouettes do
not contain appearance information and are not affected by
the clothes. When using the silhouettes as input, the model
can focus on the gait information. If the existing video-based
ReID methods, which only use visible light modality as input,
can be extended to take visible light and silhouette modalities
as input simultaneously, their performance on VCC-ReID
can be improved by explicitly extracting appearance and gait
information separately.
Based on this idea, we propose a Framework with explicit
Learning based on Appearance and Gait (FLAG), which can
extend most existing video-based ReID methods to extract
appearance and gait information explicitly. Specifically, our
FLAG includes a multi-Modal and multi-Granularities Architecture (MGA) which is a large model, and a Cross-Modal
Knowledge Distillation Scheme (CMKDS) which has a small

1051-8215 © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

1802

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 2, FEBRUARY 2025

Fig. 1.
Examples of the visible light and silhouette modalities. The
silhouettes are generated from the visible light modalities through the semantic
segmentation model.

model. They can be applied to devices with different computing power.
The MGA simultaneously uses the visible light and silhouette modalities as input to extract appearance and gait
information explicitly. The silhouette modality comprises
N levels of granularity, which help the model extract global
and local gait information. Therefore, the MGA has N+1
branches to process visible light and silhouette modalities.
N branches are used to process the N-granularities silhouette
modality (gait), and one branch is used to process the visible
light modality (appearance). Most existing video-based ReID
methods can used as the appearance and gait branches of the
MGA. Through the MGA, they can be extended to extract
appearance and gait information explicitly. To efficiently
fuse the appearance and multi-granularities gait features, the
Embedding-Based parallel fusion module is designed in the
MGA. The MGA’s performance is excellent but large and
needs silhouettes generated by the semantic segmentation
model. To expand our method to devices with limited computing power, the CMKDS is proposed to distill the MGA
to a small single-modal model that only uses the visible
light modality as input. For example, we can use AP3D [9],
which is a video-based ReID method, as the branches of
MGA and distill this MGA (teacher) to the AP3D (student). The performance of AP3D can be improved with
the explicit appearance and gait features from the MGA
and the inference cost of AP3D will not be increased.
The Embedding-Based direct and indirect distillation strategies are designed to realize efficient cross-modal knowledge
distillation.
Experimental results on CCVID [11], a VCC-ReID
dataset, demonstrate the generalization of our FLAG on
the existing ReID methods. Specifically, when combined
with the existing video-based ReID methods (AP3D [9] and
TCLNet [10]), the MGA can significantly improve their
performance, and the CMKDS can also improve their accuracy without increasing inference cost. In addition, when
combined with AP3D, the MGA can achieve state-of-the-art
performance.

The main contributions can be summarized as follows:
1) A framework with explicit learning based on appearance and gait (FLAG) is proposed to explicitly extract
appearance and gait information on VCC-ReID, and it
can be combined with most existing video-based ReID
methods.
2) A multi-modal and multi-granularities architecture
(MGA) in FLAG, which takes the visible light
and multi-granularities silhouette modalities as input,
is designed to explicitly extract appearance and gait
features and fuse them. The MGA can be applied to
devices with powerful computing power.
3) A cross-modal knowledge distillation scheme (CMKDS)
in FLAG is designed to distill the MGA to a small
single-modal model that only uses the visible light
modality as input. The small model from CMKDS can
be applied to devices with limited computing power.
4) Experimental results demonstrate the generalization
of our FLAG on the existing video-based ReID
methods. In addition, the MGA can outperform stateof-the-art accuracy by 4.2% when combined with
AP3D.
II. R ELATED W ORK
The video-based ReID can be divided into two categories:
the same-clothes setting and the clothes-changing setting.
The video-based ReID with the same-clothes setting assumes
people will not change their clothes. This kind of video-based
ReID is easier but has limited practicality, which is the
mainstream research direction. The video-based ReID with the
clothes-changing setting can be simultaneously applied in the
clothes-consistent and clothes-changing scenes. This kind of
video-based ReID is more practical but received less attention
due to its difficulty. In this section, we will introduce these
two kinds of video-based ReID.
A. Video-Based ReID With Same-Clothes Setting
The video-based ReID with the same-clothes setting attracts
much attention and performs well. Several datasets have been
published to support this task, such as PRID-2011 [12],
iLIDS-VID [13], MARS [14], DukeMTMC-VideoReID [15]
and LS-VID [16]. These datasets only contain the same-clothes
scenes, so the model trained with them can not applied to the
clothes-changing scenes.
The methods in the video-based ReID with the same-clothes
setting can be divided into two types according to how
they model temporal information. One type is directly
using the existing operator, such as CNN [9], [17], [18],
RNN [19], [20] GNN [21], [22], [23], [24] and Transformer [8], [25], [26], [27] to model temporal information.
The other type is designing the specialized module to model
temporal information, such as the module to mine interframe difference [10], [28], [29], [30], [31], [32], [33],
[34], [35], and the module to evaluate importance of frames
[36], [37], [38], [39], [40], [41], [42], [43], [44], [45].
Gu et al. [9] proposed an appearance-preserving module

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.

LU et al.: FLAG FOR VIDEO-BASED CLOTHES-CHANGING PERSON ReID

to align the feature maps according to semantic similarity.
Hou et al. [18] proposed a bilateral complementary network
for spatial complementarity modeling and a temporal kernel
selection block to capture short-term and long-term temporal relations. Chen et al. [19] present competitive similarity
aggregation and co-attentive snippet embedding to reduce the
intra-person variation in each sample. McLaughlin et al. [20]
designed a video-based ReID system for wide area tracking
based on an RNN architecture. Yang et al. [21] proposed
a spatial-temporal graph convolutional network containing
spatial and temporal GCN branches. Li et al. [24] present
a spatial-temporal graph-guided global attention network that
can mine spatial-temporal knowledge through graph modeling. He et al. [26] designed a dense interaction learning
framework to reduce the difficulties of multi-grained spatialtemporal interaction modeling. Yang et al. [27] designed a
spatiotemporal interaction Transformer network to effectively
extract the discriminative robust representation. Tang et al. [8]
designed a novel multi-stage spatial-temporal aggregation
Transformer with two designed proxy embedding modules.
Hou et al. [10] designed a temporal complementary learning
network that extracts complementary information of consecutive frames. Chen et al. [30] designed a region-level
saliency and granularities mining network to discover temporal
coherence. Leng et al. [33] proposed a multi-granularities
occlusion aware framework to extract multi-granularities features by precisely erasing the occlusion. Eom et al. [37]
designed a spatial and temporal memory network to extract
robust person representations against spatial and temporal
distractors. Wang et al. [39] proposed a hierarchical mining
network to extract discriminative representations with high
integrity even over sequences where the characteristics of
pedestrians are not consecutive. Tao et al. [43] proposed an
adaptive interference removal framework to remove various
interference.
Significant progress has been achieved in the video-based
ReID with the same-clothes setting. For example, AP3D [9]
and TCLNet [10] are representation methods of the two types
mentioned above; they can achieve 97.2 and 96.9 on Rank-1
metric in the DukeMTMC-VideoReID [15] dataset, respectively. This means the existing methods have a strong practical
application in clothes-consistent scenarios. However, because
these methods are not optimized for the clothes-changing
scene, their performance in it still needs improvement.
As mentioned in Section I, clothes changing is a common scenario in the actual use of video-based person re-identification,
so the video-based ReID with the clothes-changing setting has
great research prospects.
B. Video-Based ReID With Clothes-Changing Setting
Recently, a few researchers have focused on the video-based
ReID with the clothes-changing setting. Gu et al. [11] proposed the first publicly available VCC-ReID dataset named
CCVID. This dataset contains clothes-changing and clothesconsistent scenes, so the models trained with it are more
practical.

1803

The methods in the video-based ReID with the clotheschanging setting can also be divided into two types according
to how they are optimized in clothes-changing scenes. One
type is introducing the additional input with information
that is irrelevant to clothes but relevant to identity, such
as the face [46] and gait (our work). The other type is
introducing the additional task at the output to assist in
decoupling the clothes-irrelevant identity features, such as
the clothes classification [11], [47] and human reconstruction
tasks [47], [48], [49]. Arkushin et al. [46] proposed a method
that combines pre-trained face recognition and ReID models
and created an enriched gallery from the given query and
gallery samples. Gu et al. [11] proposed a clothes-based
adversarial loss to force the backbone of the ReID model
to learn clothes-irrelevant features. Cui et al. [47] designed
a deep component reconstruction ReID framework to disentangle the clothes-irrelevant and the clothes-relevant features.
Liu et al. [48] proposed a joint two-layer shape and texture
representation of a 3D clothed human model to disentangle
identity from non-identity components of 3D clothed humans
and reconstruct accurate 3D clothed body shapes and learn
discriminative features of naked body shapes for person ReID
in a joint manner. Nguyen et al. [49] proposed a temporal
3D shape modeling that can leverage human 3D shape to
assist ReID.
The first publicly available VCC-ReID dataset
(CCVID [11]) is proposed in 2022. After that, some
progress has been achieved in this task. Due to the short
development time and the task’s difficulty, the performance of
current methods in this field still needs to be improved. For
example, GEFF [46] and DCR-ReID [47] are representation
methods of the two types mentioned above; they can
only achieve 89.2 and 84.7 on the Rank-1 metric in the
CCVID [11] dataset, respectively. Such performance is still a
certain distance from actual use. The weaker performance of
current methods and the more practical application value of
VCC-ReID mean that this task has great research prospects.
Therefore, we will focus on the VCC-ReID in this paper.
As mentioned above, in the video-based ReID task, more
progress has been achieved on the same-clothes setting
compared to the clothes-changing setting. So, the methods
designed for the same-clothes setting are extensive and have
potential, as they have yet to be optimized for the clotheschanging setting. Therefore, we propose the FLAG, which
can combined with the existing video-based ReID methods,
to transfer the methods designed for the same-clothes setting
to the clothes-changing setting. Through our FLAG, we can
significantly improve the performance of the methods designed
for the same-clothes setting in the clothes-changing setting and
fully utilize these methods. Specifically, we will combine our
FLAG with AP3D [9] and TCLNet [10], which are designed
for the same-clothes setting. As mentioned in Section II-A,
AP3D and TCLNet represent two types of methods in the
video-based ReID with the same-clothes setting according
to how they model temporal information. Combining with
them can comprehensively demonstrate the generality of
our FLAG.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:09 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -l 3 'FDGReID - Federated Domain Generalization for Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Machine Learning (2026) 115:22
https://doi.org/10.1007/s10994-025-06974-z

FDGReID: Federated Domain Generalization for Person Reidentification
Ke Niu1 · Haiyang Yu1,3 · Teng Fu1 · Mengyang Zhao1 · Bin Li1 · Xuelin Qian2 ·
Xiangyang Xue1
Received: 31 May 2025 / Revised: 17 September 2025 / Accepted: 18 December 2025 /
Published online: 13 January 2026
© The Author(s), under exclusive licence to Springer Science+Business Media LLC, part of Springer Nature 2026

Abstract
Person re-identification (Re-ID) has become a critical task in cross-camera retrieval systems. While deep learning-based approaches have made significant strides under controlled conditions, real-world deployment remains hindered by two major challenges: domain drift and data privacy. To address these challenges, we propose FDGReID, a novel
federated learning framework designed to achieve domain generalization in person Re-ID
without compromising user privacy. FDGReID introduces two core components: style
information sharing (SIS) and viewpoint-aware contrastive learning (VCL). SIS diversifies stylistic exposure among distributed clients by sharing style representations during
federated training, improving resilience to visual appearance changes. VCL, in contrast,
mitigates spatial viewpoint shifts by enforcing identity consistency via contrastive objectives across varied perspectives at each client. Together, these modules enable FDGReID
to learn robust, domain-invariant person representations without direct data exchange. We
conduct extensive experiments on widely-used cross-domain Re-ID benchmarks, demonstrating that FDGReID consistently outperforms existing federated and generalizable
Re-ID baselines. Moreover, it ensures strict data privacy compliance by keeping all raw
images localized. Our results highlight FDGReID’s effectiveness and practicality in building scalable, privacy-preserving Re-ID systems for real-world applications.
Keywords Person re-identification · Federated learning

1 Introduction
Person re-identification (Re-ID) (Zheng et al., 2017a, b) is a critical cross-camera retrieval
task with broad real-world applications in intelligent transportation, smart cities, and public
safety. The goal of Re-ID is to retrieve images of the same pedestrian captured by differ-

Ke Niu and Haiyang Yu contributed equally to this work
Editors: Chun-Yi Lee, Andy Song, Jhih-Ciang Wu, Hung Guei
Extended author information available on the last page of the article

13

22 Page 2 of 20

Machine Learning (2026) 115:22

ent non-overlapping cameras. In recent years, significant progress has been achieved, with
many state-of-the-art methods attaining remarkable accuracy on standard benchmarks.
Despite these advancements, deploying Re-ID systems in real-world environments
remains highly challenging. One of the most fundamental obstacles is the issue of domain
drift between the training and inference phases, which transforms the Re-ID task into a
zero-shot learning problem. This challenge stems from limited data availability and the
complexity of real-world operational environments. Specifically, while training datasets
are typically collected under controlled settings with pre-defined identities, practical Re-ID
systems cannot anticipate or pre-collect data for target identities at deployment time. Consequently, identities encountered during inference are entirely unseen. Furthermore, dynamic
factors such as illumination changes, diverse camera angles, and environmental conditions
exacerbate domain divergence, leading to significant degradation in model performance.
To tackle this, recent research has explored domain generalizable Re-ID (DG-ReID) (Choi
et al., 2021b; Ni et al., 2023), where models are trained across multiple source domains to
generalize to unseen target domains. However, most DG-ReID approaches rely heavily on
direct data or feature alignment, introducing privacy concerns–a critical barrier in sensitive
or regulated scenarios.
To address the privacy issue, federated learning (FL) (Zhuang et al., 2020) emerges as a
promising solution. FL enables collaborative model training across distributed clients without sharing raw data, thus preserving user privacy. While FL-based Re-ID methods primarily enhance training strategies (e.g., improving aggregation schemes), they often overlook
domain drift–resulting in biased local models. Aggregating these biased models without
addressing distributional shifts leads to suboptimal generalization in the global model.
Through our analysis of popular Re-ID datasets, we identify that domain drift arises
primarily from two factors: (1) stylistic discrepancies such as image tone and brightness,
and (2) viewpoint inconsistencies due to varied camera placements. As illustrated in Fig. 1,
datasets like iLIDs (Zheng et al., 2009) predominantly capture side-view images, while
GRID (Loy et al., 2010) suffers from color bias and severe distortion. These inconsistencies
hinder effective generalization across domains.

Fig. 1 Illustration of the heterogeneity across person Re-ID datasets

13

Machine Learning (2026) 115:22

Page 3 of 20 22

In this paper, we propose FDGReID, a novel decentralized Re-ID framework that integrates federated learning with domain generalization principles to simultaneously tackle
privacy and domain drift challenges. FDGReID introduces two key components: Style
Information Sharing (SIS): A mechanism that promotes stylistic diversity across federated clients by enabling early-layer style information exchange. This encourages models to
become more robust to varied visual styles. Viewpoint-aware Contrastive Learning (VCL):
A contrastive learning strategy that enforces viewpoint-invariant identity representations,
improving model robustness to spatial domain shifts. Extensive experiments across multiple
Re-ID benchmarks validate that FDGReID significantly enhances cross-domain generalization while safeguarding user data, demonstrating its practical value for real-world Re-ID
deployments. Our main contributions are summarized as follows:
● We propose a novel Style Information Sharing (SIS) module within the FL framework,
which explicitly increases style diversity among clients to mitigate style-related domain
drift.
● We introduce Viewpoint-aware Contrastive Learning (VCL), a client-side contrastive
learning approach that improves viewpoint invariance in person embeddings.
● We conduct comprehensive experiments showing that FDGReID achieves superior generalization performance on unseen domains while ensuring privacy preservation, making it well-suited for deployment in privacy-sensitive real-world scenarios.

2 Related Work
2.1 Supervised Person Re-ID
Most existing person re-identification (Re-ID) approaches adopt supervised learning paradigms, typically employing classification loss to derive discriminative global representations (Zheng et al., 2017a, b). Recent advancements have explored feature invariance
across various factors, including image resolution (Li et al., 2019a), human pose (Li et al.,
2019b), domain shift (Huang et al., 2019; Jin et al., 2020), and clothing changes (Qian et
al., 2020). However, viewpoint variation remains an under-addressed yet critical challenge
in person Re-ID. Filax and Ortmeier (2021) provide empirical evidence that differences in
camera viewpoints significantly impact the performance of metric-based learning models.
Similarly, Sun and Zheng (2019) highlight substantial performance degradation when query
and gallery images exhibit viewpoint discrepancies. Motivated by these observations, we
propose a Viewpoint-aware Contrastive Learning (VCL) module that explicitly tackles the
domain drift caused by viewpoint variations. VCL encourages identity consistency across
diverse viewpoints by enforcing contrastive objectives during local client training, thereby
enhancing the robustness of the learned representations in federated Re-ID settings.

2.2 Domain-Generalizable Re-ID
Existing domain generalizable (DG) person Re-ID approaches can be broadly categorized
into three main paradigms: (1) Meta-Learning: Choi et al. (2021b) simulate generalization
failures within a meta-learning framework, enabling normalization layers to adapt more

13


exec
/bin/zsh -lc "pdftotext -l 3 'Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 11, NOVEMBER 2025

19879

Find Hidden Modality Divergence: Adversarial
Aware Learning for Unsupervised Visible–Infrared
Person Re-Identification
Yuxuan Liu , Hongwei Ge , Yong Luo , and Chunguo Wu

Abstract—Unsupervised visible–infrared person re-identification (Unsupervised VI-ReID) aims to learn discriminative
identity features under the large modality gap without any
labeled data. Currently, the state-of-the-art methods optimize
cross-modality differences by using contrastive learning as the
underlying paradigm. However, they neglect the problem of
modality divergence during the cross-modality optimization process. This problem means that the interclass instances between
the cross-modality intraclass gaps can make cross-modality
intraclass instances difficult to get closer to each other in
the feature space due to the effect of contrastive learning on
these interclass instances. To alleviate the negative impact of
the modality divergence problem, we propose an adversarial
aware learning (ADAL) framework to explore the instances
that generate modal divergence and adversarially optimize these
explored instances. Specifically, on the one hand, we explore the
optimization directions of each cluster during the cross-modality
optimization process, and the cluster centroids generating
positive optimization are facilitated, while the others generating negative optimization are penalized. On the other hand,
we further consider the instance-level optimization process,
which increases the affinities of the positive instance pairs
with large cross-modality gaps to further improve the centroidlevel optimization. Extensive experiments conducted on the
visible–infrared person Re-ID datasets show that the proposed
method is used as a universally applicable plug-in module to add
the existing unsupervised VI-ReID methods, which outperforms
the existing state-of-the-art approaches.
Index Terms—Adversarial aware learning (ADAL), person reidentification (Re-ID), unsupervised learning, visible–infrared.

P

I. I NTRODUCTION
ERSON re-identification (Re-ID) aims at matching the
consistent pedestrians in different cameras by learning

Received 19 August 2024; revised 18 January 2025 and 10 May 2025;
accepted 13 July 2025. Date of publication 29 July 2025; date of current
version 31 October 2025. This work was supported in part by the National
Natural Science Foundation of China under Grant 61976034, in part by Dalian
Science and Technology Innovation Fund under Grant 2022JJ12GX013, and
in part by Liaoning Natural Science Foundation under Grant 2022-YGJC-20.
(Corresponding author: Hongwei Ge.)
Yuxuan Liu and Yong Luo are with the School of Computer Science and
Technology, Dalian University of Technology, Dalian 116024, China (e-mail:
lyx8880lzc@mail.dlut.edu.cn).
Hongwei Ge is with the School of Computer Science and Technology,
Dalian University of Technology, Dalian 116023, China, and also with the
Key Laboratory of Social Computing and Cognitive Intelligence, Ministry of
Education, Dalian University of Technology, Dalian 116024, China (e-mail:
hwge@dlut.edu.cn).
Chunguo Wu is with the Key Laboratory of Symbolic Computation and
Knowledge Engineering of Ministry of Education, College of Computer
Science and Technology, Jilin University, Changchun 130012, China.
Digital Object Identifier 10.1109/TNNLS.2025.3591116

diverse pedestrian feature representations [1], [2], [3], which
can serve as the continuous learning system apply in the intelligent surveillance and security [4], [5], [6]. Current approaches
consider that pedestrians are matched during the daytime by
intelligent surveillance systems and primarily depend on rich
visual texture information to address the single visible modality problems in person Re-ID. However, these single-modality
techniques have weak capabilities for retrieving persons under
poor lighting conditions, limiting their applicability in realworld surveillance scenarios.
Currently, a growing number of approaches have focused
on the visible–infrared person Re-ID tasks and made many
progresses [7], [8], [9]. However, the current visible–infrared
person re-identification (VI-ReID) methods are mainly trained
in supervised settings, which require large amounts of
labeled data. Since the identity annotations of visible to
infrared modality are more costly than single modal person Re-ID annotations, limiting the scalability of supervised
visible–infrared person Re-ID methods in real-world surveillance systems. Currently, unsupervised person Re-ID methods
[10], [11], [12] have attracted increasing attention. Current
unsupervised visible-modality person Re-ID methods widely
utilize cluster-based methods to obtain clustering results and
continuously optimize the clusters to generate precise pseudo
labels. These visible-modality methods can generate better
initial clustering results due to primarily depending on rich
clothes texture information, making the subsequent optimization process easier based on initial clustering results. In
contrast, it is more difficult to optimize intraclass variance on
unsupervised visible–infrared person Re-ID task [13], [14],
[15] because of the huge cross-modality intraclass gaps.
Due to the identity labels in two modalities being unseen
in the unsupervised settings, the state-of-the-art methods [14],
[15] address the unsupervised visible–infrared person Re-ID
problem by exploring cross-modality correspondences of
the same identity. However, there are huge cross-modality
intraclass gaps in the unsupervised VI-ReID, which will
undoubtedly negatively impact on exploring the cross-modality
correspondences of the same identity. Therefore, reducing
the cross-modality intraclass gaps through effective crossmodality optimization methods can further improve the
ability to explore cross-modality correspondences of the same
identity.
The existing unsupervised VI-ReID methods primarily use
contrastive learning to perform cross-modality optimization,

2162-237X © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

19880

IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 11, NOVEMBER 2025

Fig. 1. Representation graph of motivation. Colors and shapes represent the pedestrians and modalities, respectively. The instances within the
black circle are denoted as representative cross-modality intraclass instances.
(a) Cross-modality intraclass instances are directly pulled closer to each other
during the training process, resulting in a smaller cross-modality intraclass
gap. (b) Feature distribution under the modality divergence problem, where
the interclass instances within the cross-modality gaps hinder cross-modality
intraclass instances from drawing closer, leading to a larger cross-modality
intraclass gap.

aiming to pull cross-modality intraclass instances closer while
pushing all interclass instances farther apart. Due to the
large cross-modality intraclass gaps, a considerable number
of interclass instances must exist between the cross-modality
intraclass gaps. In this case, the modality divergence problem
arises, meaning that these interclass instances between the
cross-modality intraclass gaps hinder cross-modality intraclass
instances from moving closer in the feature space. This occurs
due to the influence of contrastive learning on these interclass
instances, resulting in a larger cross-modality intraclass gap, as
shown in Fig. 1(b). In contrast, when the interclass instances
do not exist between the cross-modality intraclass gaps, the
intraclass instances are directly pulled closer to each other
by the effect of contrastive learning on positives, which
further reduces the cross-modality intraclass gaps, as shown
in Fig. 1(a). This effectively promotes modality alignment and
enhances robustness to cross-modality variations. Therefore,
such interclass instances between the cross-modality intraclass
gaps can deteriorate cross-modality intraclass instances from
converging in the feature space during the contrastive learning
process, which demonstrates the importance of the proposed
modality divergence problem in the unsupervised VI-ReID
task.
Based on the above motivation, we propose an adversarial
aware learning (ADAL) to alleviate the negative impact of
the modality divergence problem on unsupervised VI-ReID.
Specifically, as shown in Fig. 2(a), we explore the cluster
instances that contribute to negative and positive optimization. Negative optimization hinders cross-modality intraclass
clusters from converging, while positive optimization promotes
the alignment of intraclass clusters across visible and infrared
modalities. Therefore, we facilitate the positive optimization

Fig. 2.
Representation graph of our ADAL. We facilitate the positive
optimization process and penalizes the negative optimization process to
increase the cross-modality intraclass compactness. (a) Exploration process
in negative and positive optimization. (b) Adversarial optimization process
for negative and positive optimization.

process by further pushing these cluster instances away from
each other and penalize the negative optimization process by
pulling these cluster instances closer toward each other, as
shown in Fig. 2(b), which increases cross-modality intraclass
compactness.
However, the above cross-modality optimization process
only focuses on cluster-level optimization but is weak at
optimizing those hard positive instances at the cluster edges.
In general, the hard positive instances are the important
cause of generating large cross-modality intraclass variance.
Therefore, we further consider the instance-level optimization
process, increasing the intraclass affinities in large crossmodality gaps. Specifically, we merge the highest similarity
clusters in two modalities based on the similarity ranking of
cross-modality cluster centroids. Then, the easiest and hardest
positive instance pairs are explored from the merged clusters.
Since easy positive instance distribution can represent the most
compact degree of each cluster, we further enforce the model
to learn the ability to converge all cross-modality hard feature
distribution into easy feature distribution, increasing the crossmodality intraclass compactness.
We summarize our contributions as follows.
1) We raise a new modality divergence problem during
the cross-modality optimization process. To alleviate
the negative impact of the modality divergence problem on unsupervised VI-ReID, we propose an ADAL
framework, which can adversarially optimize the clusters
that generate the modality divergence, improving the
intraclass compactness of the cross-modality.
2) To further improve the cluster-level optimization process, we enforce the model to learn the ability that
converges all cross-modality hard positive feature distribution into easy positive feature distribution, which

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.

LIU et al.: FIND HIDDEN MODALITY DIVERGENCE: ADVERSARIAL AWARE LEARNING VI-ReID

increases the affinities of the intraclass instance pairs
with large cross-modality gaps.
3) We conduct extensive experiments in the large-scale
datasets to verify the effectiveness of the proposed
ADAL framework. Our framework achieves the state-ofthe-art performance on widely adopted VI-ReID person
datasets.
II. R ELATED W ORK
A. Visible–Infrared Person Re-ID
1) Supervised Visible–Infrared Person Re-ID: has recently
received increasing attention due to its potential for all-day
surveillance [16], [17], [18]. The key technical challenge is the
large domain gap between visible and infrared images from the
same individuals [19]. Wang et al. [20] introduce a dual-level
visible–infrared person Re-ID framework, which decomposes
the mixed discrepancies and handles them separately. Additionally, Liu et al. [21] propose a memory-augmented learning
framework, which learns and enhances cross-modality metrics. However, the supervised VI-ReID methods require large
amounts of labeled data in cross-modality, limiting the scalability of supervised methods in real-world deployments.
2) Unsupervised Visible–Infrared Person Re-ID: aims at
learning discriminative identity features under the large
modality gaps without any labeled data [13], [14], [22].
Liang et al. [13] first propose a two-stage method named
homogeneous-to-heterogeneous learning to address the unsupervised visible–infrared person Re-ID problem. Based on the
two-stage optimization method, the state-of-the-art unsupervised VI-ReID methods [14], [15] reduce huge cross-modality
intraclass gaps by finding cross-modality correspondences
of the same identity. Yang et al. [14] associate positive
cross-modality identities to learn the intramodality person
representation. Wu and Ye [15] formulate correspondence
mining as a graph-matching process to explore cross-modality
correspondences.
However, these approaches ignore the negative impact of the
huge cross-modality intraclass gaps on exploring the crossmodality correspondences of the same identity. The key to
reduce cross-modality intraclass gaps is addressing the modality divergence problem during the cross-modality optimization
process. The modality divergence problem means that simultaneously optimizing intra- and interclass gaps of cross-modality
will mutually generate negative optimization directions. The
proposed ADAL can effectively explore the instances that
generate modality divergence, alleviating the negative impact
of the modality divergence problem on unsupervised VI-ReID
by adversarially optimizing these explored instances.
B. Unsupervised Person Re-ID
The unsupervised learning method aims to learn discriminative feature representations without relying on any
labels. Current methods have emerged in the field of person
Re-ID research. Two interest areas in unsupervised person
Re-ID methods are unsupervised domain adaptation (UDA)
approaches [11], [23], [24] and fully unsupervised approaches
[10], [25], [26]. UDA-based approaches aim to minimize the

19881

domain gap by learning domain-invariant features from labeled
datasets and unlabeled target datasets. In contrast, fully unsupervised approaches can direct training on unlabeled target
datasets, enhancing the efficiency of real-world surveillance
systems. However, current fully unsupervised methods are
used for the single visible modality. Due to the large visible
and infrared modality differences, the existing unsupervised
methods are unsuitable for visible–infrared person Re-ID that
requires the learning of cross-modality invariant features in
the interference of the large modality differences.

C. Adversarial Learning Method for Person Re-ID
Adversarial learning is initially introduced in the generative
adversarial network (GAN) [27], which can produce realistic
images by training the generative model. This technique has
found applications in diverse tasks, such as domain adaptation
[28] and representation learning [29]. Over recent years, adversarial learning has been applied in the field of person Re-ID.
Gu et al. [30] propose a clothes-based adversarial loss, which
makes the trained clothing classifier unable to distinguish
between instances with the same identity and different clothes.
In contrast, our method aims to penalize the instances that
generate the negative optimization, which imposes the opposite
optimization direction on these negative instances to reduce
the cross-modality gaps. It is more suitable for unsupervised
visible–infrared person Re-ID tasks.
III. M ETHOD
A. Modality Divergence Problem Formulation
The modality divergence problem manifests as follows:
during the cross-modal optimization process, the interclass
instances in the cross-modal intraclass gaps can make crossmodal intraclass instances difficult to get closer to each other
in the feature space due to the effect of contrastive learning
on these interclass instances, thus negatively affecting the
modality alignment.
Specifically, given a pair of cross-modal intraclass instance
normalization features f (Xvi ) ∈ V and f (Xri ) ∈ R to be
optimized by contrastive learning, where V and R denote the
set of visible and infrared instance features, respectively. The
cross-modal intraclass gap can be defined by the following
formula:


Dintra = f Xvi − f Xri 2
(1)
where k · k2 denotes the Euclidean distance.
In the feature space, it is assumed that there exists an interclass instance feature f (X j ) ∈ R ∪ V between the cross-modal
intraclass instance pair f (Xvi ) and f (Xri ), and the interclass
instance feature (hardest negative instance) with minimum
Euclidean distance to the instance pair can be formulated as




f X j = arg min f Xvi − f (X) 2 + f Xri − f (X) 2
f (X)∈N

(2)
where N denotes the set of all negative instances. It is worth
noting that the above formula uses the similarity calculation
to define the hardest negative interclass instance that generates

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:04 UTC from IEEE Xplore. Restrictions apply.


codex
我只按前三页读摘要和引言，下面判断偏“论文怎么立住”和“我们能反推出什么方向”。

**逐篇提取**
1. `Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification`  
创新类型：新机制偏工程组合。能发是因为 AGPReID 里全图 token 冗余、背景和视角变化严重，已有 VDT 没做动态关键 token 选择。story 是“航拍地面匹配不是看全图，而是只让关键 token 参与身份建模”。注意：这篇已经直接占了“AGPReID 动态 token 选择”这个坑。

2. `Efficient Lightweight Multi-Source Domain Adaptation...`  
创新类型：工程组合加应用设定。能发是因为多源 UDA、边缘部署、伪标签噪声、轻量模型四个现实痛点打包，证据链靠多数据集、真实场景、参数量和 FLOPs。story 是“先从易到难学源域泛化，再用更干净伪标签适配目标域”。

3. `Enhancing Visible-Infrared... Modality- and Instance-Aware Adaptation`  
创新类型：问题重定义加新机制。它不再说“消除模态差”，而是说“模型要能适应不同模态和不同实例”。证据链靠 modality prompt、instance prompt、query prompt 的消融和四个数据集。story 是“与其强行对齐所有模态，不如让模型按输入动态改变特征提取方式”。

4. `ESTI`  
创新类型：问题重定义偏效率机制。能发是因为视频 ReID 的邻帧高度冗余，全帧时空建模既浪费又会淹没关键细节。story 是“空间和时间不必每帧都全量交互，分支提炼后再交互”。

5. `Exploring Part-Informed Visual-Language Learning`  
创新类型：新机制。能发是因为 CLIP-ReID 只做全身文本对齐，局部部位特征没有语义一致性；它用 parsing 生成像素级部位文本监督，并处理 parsing 的身份无关语义和 ReID 身份判别之间的冲突。story 是“全身语义不够，ReID 需要身份相关的局部语义”。

6. `False Negatives Consensus Suppression...`  
创新类型：问题重定义。能发是因为对比学习默认未配对样本全是负样本，但文本描述会让很多“负样本”语义高度相近，强推开会伤害跨模态对齐。story 是“不是对齐网络弱，而是训练目标把不该推开的样本推开了”。

7. `FDGReID`  
创新类型：新数据设定加工程机制。能发是因为真实 ReID 部署同时有隐私和域漂移，联邦 ReID 只解决隐私，DG-ReID 又常要共享数据或特征。story 是“不共享图像，只共享风格信息，并在本地做视角一致性”。

8. `Find Hidden Modality Divergence`  
创新类型：问题重定义加新机制。能发是因为它指出无监督 VI-ReID 的对比学习里，某些类间样本卡在跨模态同类样本之间，会阻塞同类跨模态靠近。story 是“跨模态优化不是简单拉正推负，有些负样本的梯度方向本身在制造模态分歧”。

9. `FLAG`  
创新类型：特权信息蒸馏加应用设定。能发是因为换衣视频 ReID 里外观失效，步态是互补身份线索；训练时显式用 RGB 和 silhouette 学外观与步态，再蒸馏到单 RGB 小模型。story 是“训练期用多模态把外观和步态分开学，部署期不增加输入成本”。

**强候选方向**
1. **把 aerial-ground ReID 从图像 token 匹配改成“共同可见 3D 人体表面匹配”。**  
挂靠资产：aerial-ground 数据、SMPL 3D 几何、SOLIDER-Swin。  
和最像工作的区别：DTST 只是在 2D 图像平面里选 top-k token，它不知道一个航拍头顶 token 和地面正面 token 是否对应同一人体表面。我们的切开点是用 SMPL mesh、joints 或 2D 投影估计两张图的可见身体表面，只比较共同可见或可几何对应的表面区域。  
便宜首验：不训练，先用现有 SOLIDER 特征加 SMPL/pose 分区，做共同可见部位加权相似度。如果 AG hard subset 上 mAP 不到 +0.4、rank1 不到 +0.5，或者航拍低清导致可用姿态低于约七成，就先杀掉。

2. **几何感知的负样本冲突抑制：解决 aerial-ground 对比学习里“不该强推的负样本”。**  
挂靠资产：aerial-ground、SMPL/pose、SOLIDER。  
和最像工作的区别：ADAL 讲 VI-ReID 的模态分歧，FNDS 讲文本图像里的语义假负样本；我们讲的是 aerial-ground 里由于可见人体证据不重叠，某些类间样本会卡在同一身份跨视角样本之间，普通 triplet 或 contrastive 会制造错误梯度。  
便宜首验：用基线 embedding 统计失败的跨视角正样本对，看它们之间是否更常出现“几何可见性相近但身份不同”的阻塞负样本。如果失败正样本的阻塞分数没有明显高于成功正样本，比如不足 1.5 倍，先不做训练。

3. **实例可靠性提示：让 SOLIDER 按每张图的 pose/SMPL 可靠证据动态改变信息通路。**  
挂靠资产：SOLIDER-Swin、pose 热图门控、SMPL。  
和最像工作的区别：MIP 的 prompt 是模态和实例向量，DTST 是 learned token 选择，π-VL 是 parsing 文本监督。我们的切开点是显式建模“这张图哪些人体证据可靠”，由 pose 置信度、遮挡、视角、mesh 可见性生成 reliability prompt 或 gating，指导 Swin 只强化可信身体证据。  
便宜首验：先不改模型，按 pose/SMPL 可靠性把 query 分桶。如果低可靠桶不是主要错误来源，或者随机可靠性分桶也能解释同样错误，就杀掉；如果错误高度集中，再做一个只加可靠性 gate 的 Tiny 单种子快跑。

4. **3D 步态特权蒸馏：把 FLAG 的 2D silhouette 步态升级成视角规范化的 SMPL 关节运动。**  
挂靠资产：SMPL 3D 几何、SOLIDER、可能的视频 ReID 数据。  
和最像工作的区别：FLAG 用 silhouette 做外观和步态双模态，再蒸馏到 RGB；我们用 3D joints 或 mesh motion 做 view-canonical gait teacher，重点解决跨衣和极端视角下 2D 轮廓不稳定的问题。  
便宜首验：先抽一个小视频子集跑 SMPL/joints，做简单 3D gait descriptor 或 teacher 相似度。如果同身份跨衣、跨视角的 3D gait margin 不优于 RGB baseline，或者 SMPL 跟踪质量不稳定，就不要推进。这个方向还需要先查“3D gait fields”类先例，撞车风险比前两个高。

我的排序是：先验最高是第 1 个，其次是第 2 个。第 1 个最能避开 DTST 的“2D token 选择”先例，同时和团队的 aerial-ground 与 SMPL 资产贴得最紧。第 3 个更像可落地模块，适合作为第 1 个或第 2 个的轻量实现。第 4 个有潜力，但依赖视频数据和先例核查，不建议第一个押。
tokens used
132,533
我只按前三页读摘要和引言，下面判断偏“论文怎么立住”和“我们能反推出什么方向”。

**逐篇提取**
1. `Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification`  
创新类型：新机制偏工程组合。能发是因为 AGPReID 里全图 token 冗余、背景和视角变化严重，已有 VDT 没做动态关键 token 选择。story 是“航拍地面匹配不是看全图，而是只让关键 token 参与身份建模”。注意：这篇已经直接占了“AGPReID 动态 token 选择”这个坑。

2. `Efficient Lightweight Multi-Source Domain Adaptation...`  
创新类型：工程组合加应用设定。能发是因为多源 UDA、边缘部署、伪标签噪声、轻量模型四个现实痛点打包，证据链靠多数据集、真实场景、参数量和 FLOPs。story 是“先从易到难学源域泛化，再用更干净伪标签适配目标域”。

3. `Enhancing Visible-Infrared... Modality- and Instance-Aware Adaptation`  
创新类型：问题重定义加新机制。它不再说“消除模态差”，而是说“模型要能适应不同模态和不同实例”。证据链靠 modality prompt、instance prompt、query prompt 的消融和四个数据集。story 是“与其强行对齐所有模态，不如让模型按输入动态改变特征提取方式”。

4. `ESTI`  
创新类型：问题重定义偏效率机制。能发是因为视频 ReID 的邻帧高度冗余，全帧时空建模既浪费又会淹没关键细节。story 是“空间和时间不必每帧都全量交互，分支提炼后再交互”。

5. `Exploring Part-Informed Visual-Language Learning`  
创新类型：新机制。能发是因为 CLIP-ReID 只做全身文本对齐，局部部位特征没有语义一致性；它用 parsing 生成像素级部位文本监督，并处理 parsing 的身份无关语义和 ReID 身份判别之间的冲突。story 是“全身语义不够，ReID 需要身份相关的局部语义”。

6. `False Negatives Consensus Suppression...`  
创新类型：问题重定义。能发是因为对比学习默认未配对样本全是负样本，但文本描述会让很多“负样本”语义高度相近，强推开会伤害跨模态对齐。story 是“不是对齐网络弱，而是训练目标把不该推开的样本推开了”。

7. `FDGReID`  
创新类型：新数据设定加工程机制。能发是因为真实 ReID 部署同时有隐私和域漂移，联邦 ReID 只解决隐私，DG-ReID 又常要共享数据或特征。story 是“不共享图像，只共享风格信息，并在本地做视角一致性”。

8. `Find Hidden Modality Divergence`  
创新类型：问题重定义加新机制。能发是因为它指出无监督 VI-ReID 的对比学习里，某些类间样本卡在跨模态同类样本之间，会阻塞同类跨模态靠近。story 是“跨模态优化不是简单拉正推负，有些负样本的梯度方向本身在制造模态分歧”。

9. `FLAG`  
创新类型：特权信息蒸馏加应用设定。能发是因为换衣视频 ReID 里外观失效，步态是互补身份线索；训练时显式用 RGB 和 silhouette 学外观与步态，再蒸馏到单 RGB 小模型。story 是“训练期用多模态把外观和步态分开学，部署期不增加输入成本”。

**强候选方向**
1. **把 aerial-ground ReID 从图像 token 匹配改成“共同可见 3D 人体表面匹配”。**  
挂靠资产：aerial-ground 数据、SMPL 3D 几何、SOLIDER-Swin。  
和最像工作的区别：DTST 只是在 2D 图像平面里选 top-k token，它不知道一个航拍头顶 token 和地面正面 token 是否对应同一人体表面。我们的切开点是用 SMPL mesh、joints 或 2D 投影估计两张图的可见身体表面，只比较共同可见或可几何对应的表面区域。  
便宜首验：不训练，先用现有 SOLIDER 特征加 SMPL/pose 分区，做共同可见部位加权相似度。如果 AG hard subset 上 mAP 不到 +0.4、rank1 不到 +0.5，或者航拍低清导致可用姿态低于约七成，就先杀掉。

2. **几何感知的负样本冲突抑制：解决 aerial-ground 对比学习里“不该强推的负样本”。**  
挂靠资产：aerial-ground、SMPL/pose、SOLIDER。  
和最像工作的区别：ADAL 讲 VI-ReID 的模态分歧，FNDS 讲文本图像里的语义假负样本；我们讲的是 aerial-ground 里由于可见人体证据不重叠，某些类间样本会卡在同一身份跨视角样本之间，普通 triplet 或 contrastive 会制造错误梯度。  
便宜首验：用基线 embedding 统计失败的跨视角正样本对，看它们之间是否更常出现“几何可见性相近但身份不同”的阻塞负样本。如果失败正样本的阻塞分数没有明显高于成功正样本，比如不足 1.5 倍，先不做训练。

3. **实例可靠性提示：让 SOLIDER 按每张图的 pose/SMPL 可靠证据动态改变信息通路。**  
挂靠资产：SOLIDER-Swin、pose 热图门控、SMPL。  
和最像工作的区别：MIP 的 prompt 是模态和实例向量，DTST 是 learned token 选择，π-VL 是 parsing 文本监督。我们的切开点是显式建模“这张图哪些人体证据可靠”，由 pose 置信度、遮挡、视角、mesh 可见性生成 reliability prompt 或 gating，指导 Swin 只强化可信身体证据。  
便宜首验：先不改模型，按 pose/SMPL 可靠性把 query 分桶。如果低可靠桶不是主要错误来源，或者随机可靠性分桶也能解释同样错误，就杀掉；如果错误高度集中，再做一个只加可靠性 gate 的 Tiny 单种子快跑。

4. **3D 步态特权蒸馏：把 FLAG 的 2D silhouette 步态升级成视角规范化的 SMPL 关节运动。**  
挂靠资产：SMPL 3D 几何、SOLIDER、可能的视频 ReID 数据。  
和最像工作的区别：FLAG 用 silhouette 做外观和步态双模态，再蒸馏到 RGB；我们用 3D joints 或 mesh motion 做 view-canonical gait teacher，重点解决跨衣和极端视角下 2D 轮廓不稳定的问题。  
便宜首验：先抽一个小视频子集跑 SMPL/joints，做简单 3D gait descriptor 或 teacher 相似度。如果同身份跨衣、跨视角的 3D gait margin 不优于 RGB baseline，或者 SMPL 跟踪质量不稳定，就不要推进。这个方向还需要先查“3D gait fields”类先例，撞车风险比前两个高。

我的排序是：先验最高是第 1 个，其次是第 2 个。第 1 个最能避开 DTST 的“2D token 选择”先例，同时和团队的 aerial-ground 与 SMPL 资产贴得最紧。第 3 个更像可落地模块，适合作为第 1 个或第 2 个的轻量实现。第 4 个有潜力，但依赖视频数据和先例核查，不建议第一个押。
