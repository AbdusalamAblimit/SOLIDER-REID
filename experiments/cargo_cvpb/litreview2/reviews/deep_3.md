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
session id: 019ef835-59c2-7121-9c10-e136ccedcc36
--------
user
你是 ReID 创新方法论拆解员 3。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(3)==
Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf
Adversarial perturbation and defense for generalizable person re-identification.pdf
Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf
Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf
Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf
Beyond geometry - The power of texture in interpretable 3D person ReID.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则确认当前目标和可读范围，然后用 PDF 技能的流程读方法部分。这个目录是只读沙箱，所以我会只做读取、检索和整理，不写入文件。
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
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc pwd in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title

exec
/bin/zsh -lc 'rg --files' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.623.12021/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
Multi Queue for Unsupervised Person Re-identification.pdf
Interactive Sketch-Based Person Re-Identification with Text Feedback.pdf
Camera-aware graph multi-domain adaptive learning for unsupervised person re-identification.pdf
Lifelong visible-infrared person re-identification via replay samples domain-modality-mix reconstruction and cross-domain cognitive network.pdf
Richer Semantics, Better Alignment - Aligning Visual Features with Explicit and Enriched Semantics for Visible-Infrared Person Re-Identification.pdf
Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification.pdf
Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf
UP-Person - Unified Parameter-Efficient Transfer Learning for Text-Based Person Retrieval.pdf
A Survey on 3D Skeleton Based Person Re-Identification - Taxonomy, Advances, Challenges, and Interdisciplinary Prospects.pdf
Find Hidden Modality Divergence - Adversarial Aware Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
4SNet - Spatial and Spectrum Self-adaptive Synergy Network for Visible-Infrared Person Re-identification.pdf
Self-Rectification Historical Consistency Learning for Coupled Noisy Visible-Infrared Person Re-Identification.pdf
Spectrum-guided feature enhancement network for event person re-identification.pdf
Stochastic style perturbation modelling for visible-Infrared person re-Identification with severely modality imbalance.pdf
Visible-Infrared Person Re-Identification Based on Feature Decoupling and Refinement.pdf
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
Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf
Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf
TP-LReID - Lifelong person re-identification using text prompts.pdf
Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf
Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf
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
Multi-feature balanced network for clothes-changing person re-identification.pdf
Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf
Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification.pdf
GSTNET - A Geospatial-Temporal Graph Network for Group Person Re-Identification.pdf
manifest.csv
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
Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf
CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf
Instant pose extraction based on mask transformer for occluded person re-identification.pdf
FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf
Cross-modality average precision optimization for visible thermal person re-identification.pdf
CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
'Knowledge and experience' for visible-infrared person re-identification.pdf
CLIP-driven fine-grained mining for text-based person search.pdf
Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf
Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
Occluded person Re-Identification with noise injection.pdf
Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf
Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf
DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf
SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf
CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf
A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf
When Large Vision-Language Models Meet Person Re-Identification.pdf
Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf
Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf
Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
Privacy preserving person re-identification via anonymizing diffusion model.pdf

exec
/bin/zsh -lc "sed -n '1,180p' /Users/abdslm/Desktop/reid-paper/experiments/STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/Desktop/reid-paper/CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
我已经读到当前目标，确认这一步就是读文献、做差距分析和学习别人如何构造创新。接下来我先定位每篇的方法章节边界，再把方法正文读完整，避免只凭摘要判断。
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
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf' - | nl -ba | rg -n \"Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|FRAMEWORK|Experiments|EXPERIMENT|III|IV|3\\.|4\\.\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
161:   161	Method
163:   163	3.1 Overview
174:   174	3.2 Cross-Modal Alignment Stage
175:   175	3.2.1
188:   188	3.2.2
228:   228	3.2.3
315:   315	3.3 Dual Guidance Stage
397:   397	Experiments
399:   399	4.1 Experimental Settings
400:   400	4.1.1
402:   402	4.1.2
427:   427	horizontal flipping, cropping, and erasing [30] are used. In the crossmodal alignment stage, the Adam optimizer is used with a base learning rate of 3.5e-04 and a warm-up learning rate that increases linearly
436:   436	2.76M additional parameters, which represents only about 3.2% of
443:   443	4.2 Performance Comparison
471:   471	4.2.1
479:   479	achieves a 3.2% improvement in mAP and a 2.6% increase in R@1.
483:   483	(e.g., reaching 75.4% mAP and 74.1 R@1 on Occ-Duke). Detailed
487:   487	achieving 55.1% mAP and 63.7% Rank-1, compared to 53.5% mAP
489:   489	Method
538:   538	43.8
540:   540	53.6
541:   541	64.5
550:   550	63.0
551:   551	74.7
552:   552	63.8
553:   553	74.5
556:   556	63.2
568:   568	83.0
586:   586	83.1
590:   590	84.5
598:   598	4.2.2
618:   618	74.4
619:   619	64.6
620:   620	83.7
632:   632	Table 3.
649:   649	84.9
650:   650	94.2
675:   675	83.1
677:   677	83.2
687:   687	83.1
689:   689	83.3
705:   705	4.3 Ablation Studies and Analysis
706:   706	4.3.1
717:   717	63.2
736:   736	Table 4. Ablation studies for AG-ReID, where “AT” is attribute-prompt
741:   741	Method
763:   763	4.3.2
780:   780	63.0
782:   782	63.2
832:   832	Figure 3.
842:   842	4.3.3
857:   857	63.2
890:   890	4.3.4
894:   894	We employed the Grad-CAM method [16] to visualize the key regions of interest for the model, shown in Figure 3. Compared to the
898:   898	4.3.5
929:   929	Profd: Prompt-guided feature disentangling for occluded person reidentification. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 1583–1592, 2024.
956:   956	1413, 2023.
968:   968	visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
977:   977	pages 1613–1623, 2023.
979:   979	re-identification. arXiv preprint arXiv:2407.18112, 2024.
998:   998	intelligence, volume 38, pages 6162–6170, 2024.
999:   999	[25] J. Xia, L. Tan, P. Dai, M. Zhao, Y. Wu, and L. Cao. Attention disturbance and dual-path constraint network for occluded person reidentification. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 6198–6206, 2024.
1002:  1002	Conference on Multimedia, pages 4492–4501, 2023.
1009:  1009	Intelligence, volume 38, pages 6979–6987, 2024.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf' - | nl -ba | rg -n \"Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|FRAMEWORK|Experiments|EXPERIMENT|III|IV|3\\.|4\\.\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | rg -n \"Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|FRAMEWORK|Experiments|EXPERIMENT|III|IV|3\\.|4\\.\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
90:    90	in generating pedestrian images. Experiments on public reidentification datasets, such as Market1501 and DukeMTMC,
92:    92	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:51 UTC from IEEE Xplore. Restrictions apply.
156:   156	Fig. 3.
195:   195	Fig. 3. The Identity Diffuser model uses a pre-trained re-identification net to
257:   257	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:51 UTC from IEEE Xplore. Restrictions apply.
268:   268	the ϵ in Eq. 4. The loss function is defined as:
301:   301	models, with detailed experiment results shown in Sec. III.
314:   314	III. E XPERIMENTS
328:   328	We visually compare Identity Diffuser with the current stateof-the-art PIDM [9], as shown in Fig. 4. Each row depicts a
355:   355	Fig. 4. Visual comparison of Identity Diffuser and PIDM[9] for market1501
373:   373	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:51 UTC from IEEE Xplore. Restrictions apply.
376:   376	C OMPARISON OF DIFFERENT GENERATIVE METHODS ON REALISM IN
378:   378	Methods
385:   385	Methods
391:   391	54.23
405:   405	14.45
407:   407	13.29
496:   496	83.14
499:   499	83.40
500:   500	84.17
504:   504	shown in Table III, using ImageNet for identity embedding
528:   528	4.71% improvement at the 0.8 level, highlighting the greater
535:   535	are provided in the ablation study section III-D.
549:   549	TABLE III
586:   586	83.93
587:   587	84.17
591:   591	data pretraining? As indicated in Table IV, two different reidentification models, ResNet50 and ResNet100 are compared
599:   599	TABLE IV
630:   630	83.44
633:   633	83.29
647:   647	43.41
665:   665	84.17
667:   667	IV. C ONCLUSION
676:   676	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:51 UTC from IEEE Xplore. Restrictions apply.
703:   703	Conf. Comput. Vis. ICCV, Oct. 2023. 1
710:   710	Models,” in The Twelfth International Conference on Learning Representations, Oct. 2023. 1
711:   711	[11] F. Shen and J. Tang, “IMAGPose: A Unified Conditional Framework
712:   712	for Pose-Guided Person Generation,” in NIPS, 2024. 1
732:   732	International Conference on Computer Vision, 2015, pp. 1116–1124. 3
756:   756	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:51 UTC from IEEE Xplore. Restrictions apply.

 succeeded in 0ms:
1:     1	Base-Detail Feature Learning Framework for Visible-Infrared Person
13:    13	gongzhh888@gmail.com, wulian best@163.com, laterfall@hit.edu.cn
19:    19	neglecting modality-specific details. To fully utilize differentiated minutiae, we propose a BaseDetail Feature Learning Framework (BDLF) that
68:    68	Framework (BDLF), as shown in Figure 1(c). This framework is designed to extract modality-shared base features
90:    90	• We propose an end-to-end Base-Detail Feature Learning Framework (BDLF) for VIReID that integrates extracts of modality-shared base knowledge and modalityspecific detail knowledge.
113:   113	another modality. XIV [Li et al., 2020] introduces the Xmodality generated by a lightweight auxiliary network to decrease discrepancies between the two modalities. LUPI [Alehdaghi et al., 2022] establishes an intermediate domain between VIS and IR modalities. Furthermore, it generates images that belong to this intermediate domain to guide the
123:   123	Methodology
125:   125	3.1
127:   127	Overall Framework
256:   256	Figure 2: The pipeline of the proposed Base-Detail Feature Learning Framework (BDLF), which consists of a Detail Feature Extraction
284:   284	3.2
315:   315	3.3
463:   463	3.4
588:   588	3.5
641:   641	Experiments
646:   646	4.1
660:   660	Methods
681:   681	74.7
687:   687	73.9
710:   710	73.0 79.3
712:   712	72.7 83.5
714:   714	76.8 74.6 84.2
719:   719	74.1 89.1
721:   721	81.2 84.2
722:   722	83.3 91.1
724:   724	78.9 93.0
725:   725	84.6 92.6
727:   727	85.8 94.1
729:   729	79.1 84.8
730:   730	84.1 87.5
731:   731	84.4 88.4
733:   733	81.0 83.7
736:   736	84.3 87.4
738:   738	86.0 93.0
739:   739	88.8 93.3
743:   743	83.9
746:   746	83.4 62.5 65.8 54.9 62.9
749:   749	84.9
754:   754	85.8 94.4 90.1 94.5 89.6 67.0 68.9 58.1 64.5
780:   780	73.7
782:   782	73.7
783:   783	74.0
793:   793	74.6
826:   826	4.2
848:   848	4.3
850:   850	Comparison with State-of-the-art Methods
860:   860	rank-1 accuracy of 76.8% and a mAP of 74.6%, in the IndoorSearch mode, BDLF achieved a rank-1 accuracy of 84.2%
862:   862	achieved a rank-1 accuracy of 94.4% and a mAP of 90.1%
864:   864	94.5% and a mAP of 89.6% for the IR to VIS search. These
872:   872	of 64.5% in IR to VIS search. It is evident that our BDLF is
875:   875	4.4
898:   898	73.4
902:   902	74.6
943:   943	4.5
983:   983	17999. IEEE, 2023.
996:   996	22761. IEEE, 2023.
1006:  1006	Montreal, Quebec, Canada, pages 2672–2680, 2014.
1017:  1017	and Bastian Leibe. In defense of the triplet loss for person re-identification. CoRR, abs/1703.07737, 2017.
1031:  1031	Orleans, LA, USA, June 18-24, 2022, pages 19344–19353.
1035:  1035	visible-infrared person re-identification. IEEE Transactions on Multimedia, 26:2700–2714, 2024.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf' - | nl -ba | rg -n \"Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|FRAMEWORK|Experiments|EXPERIMENT|III|IV|3\\.|4\\.\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
169:   169	Fig. 2. Framework of Flow2Flow, in which the blue arrows denote training sample generation (𝑧𝑉𝑖 → 𝑥̂ 𝑉𝑖 2𝑉 and 𝑧𝑅
222:   222	3. Preliminaries
228:   228	2.3. Flow-based generative models
275:   275	2.4. Generative adversarial network
351:   351	and fake visible modality, respectively. And vice versa for 𝑅 . The detailed training objectives are presented in Section 4.2.
356:   356	4.2. Objectives for model training
432:   432	4. Methodologies
433:   433	4.1. Model architecture
447:   447	1 × 1 convolution layers and aﬃne coupling layers as the standard ﬂowbased model [15,16] of Section 3. Moreover, to increase the nonlinearity
768:   768	4.3. Image generation
940:   940	54.75
944:   944	53.02
946:   946	53.63
949:   949	64.18
1028:  1028	83.18 84.60 W=0.0, p=0.0423
1029:  1029	84.00 W=0.0, p=0.0637
1033:  1033	Method
1089:  1089	5. Experiments
1099:  1099	performance of training sample expansion (TSE) and cross-modal generation (CMG). We report the results in Table 3. As can be seen, TSE and
1146:  1146	Fig. 3. As can be seen, expansion on two modalities could achieve better
1153:  1153	Fig. 3. Performance of training sample expansion, in which the dotted line denoted the performance of baseline model. X-axis represents the expansion multiple,
1162:  1162	Method
1235:  1235	74.13
1284:  1284	Method
1305:  1305	Method
1371:  1371	5.2.3. Veriﬁcation of cross-Modal image generation
1406:  1406	Fig. 4. Visualization of generated samples by image space-based and latent Gaussian space-based methods. For the latent space interpolation, we set 𝑞 = 2 and 𝑝 = 1.
1414:  1414	Method
1437:  1437	Method
1516:  1516	74.97
1538:  1538	64.04
1543:  1543	73.31
1570:  1570	74.85
1580:  1580	5.2.4. Veriﬁcation of nonlinear activation layer
1617:  1617	5.3. Comparison and visualization
1618:  1618	5.3.1. Quantitative comparison with SOTA

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Beyond geometry - The power of texture in interpretable 3D person ReID.pdf' - | nl -ba | rg -n \"Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|FRAMEWORK|Experiments|EXPERIMENT|III|IV|3\\.|4\\.\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
10:    10	Signal Processing, Artificial Intelligence and Vision Technologies (SAIVT), Queensland University of Technology (QUT), Brisbane, QLD 4000, Australia
83:    83	(http://creativecommons.org/licenses/by/4.0/).
219:   219	3.1. Preliminaries: UVTexture and 3D reconstruction
239:   239	3.2. Framework overview
255:   255	3. Methodology
259:   259	Section 3.1, which establish the foundational concepts essential for
261:   261	overview of the overall framework architecture in Section 3.2. Subsequently, we will detail the two main processing streams: the UVTextureemphasized ReID stream in Section 3.3 and the 3D point cloud processing stream in Section 3.4. Finally, we will present our novel explainable module that provides interpretability through UVTexture-based
262:   262	visualization in Section 3.5.
270:   270	FusionTexReIDNet: Proposed method summary.
345:   345	3.4. Stream 2 — 3DReIDNet
372:   372	3.3. Stream 1 — UVTextureNet: UVTexture-emphasized ReID
375:   375	in Section 3.1. The UVTexture maps provide a 3D representation of
393:   393	3.5. Explainable module — UVTexture-inspired explainability
412:   412	Fig. 4. Overview of attribute-guided attention map generation using Attribute
417:   417	Fig. 3. Explainability comparison between 2D and 3D person ReID. Top: 2D
684:   684	Section 4.4.
685:   685	4. Experimental results and discussions
686:   686	4.1. Datasets and evaluation metrics
723:   723	and 13.2% in mAP, validating the effectiveness of our dual-stream
727:   727	and 93.5% mAP on Market-1501, representing substantial improvements over the baseline configuration. The optimization techniques
758:   758	4.2. Implementation details
761:   761	of several key components. Python version used was 3.7.12. The CUDA
764:   764	PyTorch version 1.4.
766:   766	optimizer (Kingma and Ba, 2015) with a learning rate of 10𝑒 − 4. The
768:   768	size of 128 × 64. We obtain 6890 points with RGB information using
790:   790	4.3. Comparison with the state-of-the-art methods
805:   805	Comprehensive comparison of person ReID methods across different modalities and datasets. Methods are evaluated using Rank-1 accuracy and mean Average
809:   809	Methods
919:   919	44.80
939:   939	33.16
941:   941	14.00
947:   947	63.30
951:   951	53.80
963:   963	4.16
981:   981	64.50
997:   997	4.16
1015:  1015	63.40
1019:  1019	54.10
1033:  1033	83.14
1035:  1035	63.36
1037:  1037	73.16
1045:  1045	63.50
1059:  1059	44.72
1067:  1067	84.26
1071:  1071	74.28
1079:  1079	64.80
1085:  1085	63.50
1099:  1099	24.56
1101:  1101	84.59
1105:  1105	73.20
1119:  1119	64.50
1133:  1133	23.51
1139:  1139	74.15
1149:  1149	53.90
1175:  1175	93.80
1176:  1176	94.70
1177:  1177	94.80
1183:  1183	83.30
1189:  1189	74.80
1192:  1192	83.10
1196:  1196	64.00
1208:  1208	73.80
1217:  1217	64.80
1219:  1219	53.50
1223:  1223	64.80
1229:  1229	53.40
1260:  1260	83.90
1268:  1268	73.40
1269:  1269	74.50
1276:  1276	74.20
1293:  1293	73.20
1317:  1317	74.80
1329:  1329	73.50
1366:  1366	13.33
1384:  1384	24.56
1388:  1388	23.98
1410:  1410	53.45
1414:  1414	54.89
1416:  1416	33.45
1444:  1444	63.45
1448:  1448	64.78
1470:  1470	64.23
1472:  1472	44.19
1480:  1480	44.56
1486:  1486	64.23
1490:  1490	63.78
1502:  1502	54.18
1522:  1522	43.23
1524:  1524	64.90
1542:  1542	43.84
1556:  1556	44.89
1560:  1560	44.12
1576:  1576	44.27
1606:  1606	63.39
1618:  1618	73.67
1644:  1644	44.56
1648:  1648	73.45
1652:  1652	74.56
1654:  1654	53.23
1680:  1680	24.07
1682:  1682	74.23
1684:  1684	53.12
1688:  1688	54.12
1722:  1722	63.45
1746:  1746	54.04
1780:  1780	63.67
1790:  1790	83.89
1813:  1813	84.26
1815:  1815	64.93
1835:  1835	44.12
1839:  1839	43.56
1855:  1855	44.23
1889:  1889	64.11
1895:  1895	84.23
1905:  1905	64.34
1909:  1909	63.67
1911:  1911	‡ Total parameters including ResNet-50 backbone (23.51M) and our 3D processing modules (2.47M).
1917:  1917	Methods
1953:  1953	83.34
1954:  1954	83.34
1957:  1957	84.45
1958:  1958	84.45
2001:  2001	64.34
2002:  2002	64.34
2029:  2029	63.40 89.20
2035:  2035	84.50 89.40
2069:  2069	93.00
2070:  2070	93.89
2071:  2071	94.39
2074:  2074	73.84
2076:  2076	83.12
2078:  2078	83.01
2081:  2081	54.08
2084:  2084	83.16
2085:  2085	83.74
2097:  2097	73.67
2102:  2102	24.56
2105:  2105	33.90
2106:  2106	34.45
2107:  2107	64.60
2138:  2138	84.59
2140:  2140	64.78
2154:  2154	93.50 92.90
2170:  2170	83.45
2179:  2179	73.67
2180:  2180	83.78
2181:  2181	84.56
2193:  2193	84.34
2196:  2196	83.12
2201:  2201	63.34
2202:  2202	63.34
2203:  2203	83.80

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Adversarial perturbation and defense for generalizable person re-identification.pdf' - | nl -ba | rg -n \"Method|METHOD|Approach|APPROACH|Proposed|PROPOSED|Framework|FRAMEWORK|Experiments|EXPERIMENT|III|IV|3\\.|4\\.\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
38:    38	(APD) Re-identification Method. In the APD, to synthesize hard matching samples, we introduce a MetricPerturbation Generation Network (MPG-Net) grounded in the concept of metric adversariality. In the MPG-Net,
107:   107	Above all, we propose a novel Adversarial Perturbation and Defense (APD) Re-identification Method for the Domain Generalization
269:   269	3. Method
276:   276	3.1. MPG-Net
322:   322	Fig. 3. The Architecture of the proposed SP-Net. LPP: Linear Projection of Patches; MSP: Multi-scale Semantic Perception Module; ViT: ViT-Transformer; CEL: Cross-entropy loss;
325:   325	3.2. SP-Net
329:   329	SP-Net is shown in Fig. 3. It contains several components (Fig. 3):
333:   333	3.2.1. Baseline/backbone
345:   345	Fig. 4. Structure of the proposed Multi-scale Semantic Perception (MSP) Module. The
352:   352	3.2.2. Multi-scale semantic perception (MSP) module
380:   380	3.2.3. Semantic self-perturbation and defense (SSD) scheme
499:   499	3.2.4. Loss functions in SP-Net
508:   508	4. Experimental results
526:   526	4.1. Datasets and settings
559:   559	Method
594:   594	43.7
595:   595	44.7
610:   610	63.5
612:   612	64.5
628:   628	74.4
633:   633	24.3
640:   640	33.6
641:   641	33.1
642:   642	34.4
651:   651	63.8
661:   661	74.0
662:   662	73.5
667:   667	83.1
673:   673	74.7
677:   677	84.6
685:   685	23.5
686:   686	24.0
692:   692	33.9
704:   704	84.1
714:   714	44.4
750:   750	73.7
763:   763	43.1
765:   765	53.0
789:   789	23.7
807:   807	54.3
852:   852	4.2. Comparison with state-of-the-art methods
872:   872	Method
920:   920	34.2
926:   926	24.8
928:   928	34.4
939:   939	23.6
949:   949	34.6
950:   950	63.1
958:   958	83.7
969:   969	34.8
971:   971	43.6
987:   987	43.0
992:   992	33.3
993:   993	33.8
995:   995	43.1
1004:  1004	24.4
1005:  1005	24.5
1027:  1027	84.5
1035:  1035	43.0
1068:  1068	results are reported in Table 3. From Table 3, we can see that APD
1113:  1113	33.3
1121:  1121	24.1
1127:  1127	24.8
1133:  1133	24.9
1162:  1162	94.4
1166:  1166	14.3
1195:  1195	4.3. Ablation study
1199:  1199	results are documented in Table 4. The Baseline (Base.) structure is
1200:  1200	described in Section 3.2.1. As reported in He et al. (2021), the Baseline achieved ∼ 94% Rank-1 score (see Table 5) on the Market1501

codex
第一轮边界定位显示，这几篇的方法章节大多在第 3 节或第 4 节，少数生成类短文在第 2 节就已经进入模型定义。我现在按章节把引言中的动机段和完整方法段一起读，因为“触发观察”和“重定义动作”通常埋在引言而不是方法小节标题里。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf' - | nl -ba | sed -n '1,320p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 979-8-3503-6874-1/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICASSP49660.2025.10887789
     2	
     3	Attribute Conditional Diffusion-Augmented Person
     4	Re-Identification
     5	Shijie Nie1 , Ziqiang Shi1 , Rujie Liu1 , Song Guo1 , Meng Zhang1 , Mengjiao Wang1 ,
     6	Kazuki Osamura2 , Lina Septiana2 , Abe Narishige2
     7	1 Fujitsu Research & Development Center, China. 2 Fujitsu Limited, Japan
     8	nieshijie@fujitsu.com
     9	Abstract—Due to privacy and cost issues, the lack of largescale labeled datasets limits the advancement of person reidentification. Existing methods use generative adversarial networks or game engine rendering for data augmentation to improve re-identification performance. However, these approaches
    10	struggle to maintain realistic images. This paper introduces a
    11	novel approach called Identity Diffuser, which uses diffusion
    12	models to generate synthetic data for the same identity with
    13	different poses. Our proposed framework incorporates identityspecific embeddings and target poses into the diffusion process,
    14	enabling the generation of realistic and diverse images that
    15	consistently preserve identity features. Guided by pretrained
    16	re-identification net and target pose heatmap, the framework
    17	learns transformation trajectories through forward and backward denoising steps in the diffusion models. This approach
    18	effectively maintains key pedestrian attributes across various
    19	poses. Experimental results on the Market1501 and DukeMTMC
    20	datasets demonstrate a notable improvement in performance,
    21	with a 1.73%/0.80% mAp increase in Market1501/DukeMTMC
    22	datasets compared with current state-of-the-art method. When
    23	less real data is included, the increment can be 5.1%/1.5%,
    24	separately.
    25	Index Terms—Person re-identification, conditional diffusion
    26	models, attribute augmentation
    27	
    28	poses. Due to one-step generation process, these methods
    29	face challenges with image diversity and quality, and GANs
    30	often struggle with convergence instability and mode collapse.
    31	For example, PG-GAN often produces striped noise or blurry
    32	images. Recently, diffusion models [6] have shown promise
    33	in generating high-quality images [7] and have been applied
    34	to various tasks [8]. Diffusion-based methods for pedestrian
    35	synthesis, such as PIDM [9] (Fig. 1(c)), PCDM [10], and
    36	IMAGPose [11] have made progress. However, controlling
    37	the generation process to ensuring identity consistency across
    38	different poses for re-identification remains challenging.
    39	
    40	(a) Randperson
    41	Rendering based methods
    42	
    43	(b) PG-GAN
    44	GAN based methods
    45	
    46	(c) PIDM
    47	Diffusion based methods
    48	
    49	(d) Identity Diffuser
    50	Our methods
    51	
    52	I. I NTRODUCTION
    53	Person re-identification involves matching individuals
    54	across different cameras, a task complicated by variations
    55	in pose, low-resolution surveillance footage, occlusions, and
    56	other factors. While deep learning has driven significant
    57	progress in this field [1], the shortage of large-scale labeled
    58	datasets remains a significant challenge. Privacy concerns and
    59	the high cost of manual annotation further complicate data
    60	collection.
    61	To address this issue, synthetic data generation methods
    62	have been explored, broadly categorized into model-based
    63	rendering and data-driven generative approaches. Model-based
    64	rendering methods, such as PersonX [2] and Randperson [3]
    65	shown in Fig. 1(a), create large-scale synthetic data by simulating diverse human poses and environments. While these
    66	datasets improve re-identification performance when combined
    67	with real data, they rely on expensive and complex 3D models
    68	for high-quality results.
    69	On the other hand, data-driven generative methods, especially those using GANs, offer an alternative approach.
    70	Techniques like FD-GAN [4] and PG-GAN [5] (Fig. 1(b))
    71	focus on augmenting data by generating images in various
    72	
    73	Fig. 1. A visual comparison of various typical re-identification synthetic
    74	data generation methods highlights specific challenges. The PIDM dataset
    75	occasionally suffers from background noise and loss of cloth texture detail.
    76	GAN-based methods often produce artifacts, such as striped images, while
    77	rendering engine-based methods face issues with coarse 3D human modeling.
    78	
    79	In this work, we propose the Identity Diffuser framework,
    80	which takes the first step in integrating prior knowledge
    81	from the identity embedding space into diffusion models
    82	to generate high-quality pedestrian images with consistent
    83	identity features. In scenarios where less real data per person
    84	is available, our approach can generate novel images given
    85	arbitrary poses and a source identity image. Our approach uses
    86	a pre-trained encoder to guide the diffusion process, ensuring
    87	identity consistency across pose variations.
    88	Our contributions include the development of the Identity
    89	Diffuser model, which outperforms state-of-the-art methods
    90	in generating pedestrian images. Experiments on public reidentification datasets, such as Market1501 and DukeMTMC,
    91	
    92	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:51 UTC from IEEE Xplore. Restrictions apply.
    93	
    94	Real dataset
    95	
    96	Reid
    97	model
    98	
    99	Train 1
   100	
   101	Synthetic data
   102	Finetune 4
   103	
   104	ID embedding 2
   105	
   106	Image pairs 2
   107	
   108	Augmentation
   109	
   110	Pre-trained
   111	Reid-net
   112	
   113	C
   114	
   115	xs
   116	
   117	Input
   118	1
   119	
   120	2
   121	
   122	Diffusion model
   123	3
   124	
   125	4
   126	
   127	Fig. 2. The proposed person re-identification framework consists of four
   128	steps: training a re-identification model on a real dataset, using pose and
   129	image embeddings to train a diffusion model, generating synthetic images
   130	with the diffusion model, and pre-training the re-identification model with
   131	these synthetic images before fine-tuning on the real dataset.
   132	
   133	demonstrate that our model improves mAP by 1.73%/0.8%,
   134	with even larger gains of up to 5.1%/1.5% when less real data
   135	is available.
   136	II. M ETHODS
   137	In this section, we first present an overview of our generative
   138	pipeline for pedestrian synthetic data generation, followed by
   139	a detailed explanation of the model architecture and training
   140	process.
   141	A. Overview
   142	Fig. 2 illustrates the pipeline. Starting with a real pedestrian dataset, We then randomly sample source/target image
   143	pairs with same identity for training diffusion models. In
   144	the following sections, we will not distinguish between the
   145	source ID and target ID, as they are the same. We train a reidentification model [12] and use its backbone to extract source
   146	identity embedding. OpenPose [13] is applied to extract pose
   147	information from the target image. By combining the source
   148	image, target pose map and identity embedding as input, we
   149	train a diffusion model to generate synthetic pedestrian images
   150	that matches the target image. Finally, we create a synthetic
   151	dataset conditioned on arbitrary poses and source images to
   152	pre-train the re-identification model, which is later fine-tuned
   153	with the real dataset. Note that in the training stage, pose and
   154	target image is aligned, while in the inference stage is not. The
   155	detailed model architecture and training process are shown in
   156	Fig. 3.
   157	B. Human Attribute-conditioned Diffusion Model
   158	In the detailed framework shown in Fig. 3, a pre-trained
   159	ResNet model is used to extract identity embeddings, which
   160	are then processed by an ID adapter and expanded to match
   161	the dimensions of the source image. These embeddings are
   162	combined with the pose heatmap and the source image using
   163	channel-wise concatenation. The combined inputs are further
   164	refined through global source image feature extractor using
   165	cross-attention. The architecture is designed to estimate noise
   166	and progressively update the image across multiple timesteps
   167	during the diffusion process.
   168	
   169	Timesteps
   170	
   171	t
   172	
   173	ID adapter
   174	
   175	3
   176	
   177	Pose 2
   178	Condition
   179	Random pose pool
   180	
   181	xid
   182	
   183	t+1
   184	θ (yt , t, xp , xs )
   185	
   186	yt
   187	
   188	xp
   189	
   190	Cross-Attention Estimated noise
   191	
   192	Global feature
   193	Extractor
   194	
   195	Fig. 3. The Identity Diffuser model uses a pre-trained re-identification net to
   196	extract identity embeddings, processes them with an ID adapter, and combines
   197	them with the pose heatmap and source image. These inputs are refined
   198	through a global feature extractor through cross-attention, and the model
   199	estimates noise to iteratively denoise the image during the diffusion process.
   200	
   201	To formulate this process, our objective is to train a diffusion
   202	model G. Given a source image xs , a target pose xp , and
   203	a target image y, the goal is to generate a synthetic image
   204	xt that matches the pose of xp while preserving the identity
   205	embedding of xs . The model is composed of three main
   206	components: an offline pose extractor Ep , an identity encoder
   207	Eid , and a diffusion model generator G. Let Pi represent the
   208	distribution of all images corresponding to identity si with
   209	arbitrary poses, and let Yi = {yij ∈ Pi }N
   210	j=1 denote the set of
   211	target images that share the same identity as the source images
   212	but with different poses. Our objective is to train the generator
   213	G such that it can produce synthetic images conditioned on
   214	the identity representation and poses extracted from Yi . The
   215	problem can be defined as follows:
   216	G(ϵ, Eid (xs ), Ep (Yi )) ∼ Pi
   217	
   218	(1)
   219	
   220	where ϵ is random noise, starting with pure Gaussian noise
   221	ϵ ∼ N (0, I). Let G = pθ (y|xs , xp , zid ) represent a diffusion
   222	model conditioned on the target pose xp , source image xs , and
   223	source identity embedding zid , which is extracted by Eid :
   224	zid = Eid (xs )
   225	
   226	(2)
   227	
   228	Suppose the denoising diffusion probabilistic model
   229	(DDPM) adds noise from y0 ∼ q(y0 ) to an isotropic Gaussian
   230	noise yT ∼ N (0, I) in T steps. The forward process is:
   231	q(yt |yt−1 ) = N (yt ;
   232	
   233	p
   234	
   235	1 − βt yt−1 , βt I)
   236	
   237	(3)
   238	
   239	where t ∼ [1, T ] and β1 , β2 , ..., βT is a fixed variance schedule
   240	with
   241	Qt βt ∈ (0, 1). Using the notation αt = 1 − βt and ᾱt =
   242	i=1 αi , we can sample from q(yt |y0 ) in a closed form at an
   243	arbitrary timestep t:
   244	√
   245	q(yt |y0 ) = N (yt ; ᾱt y0 , 1 − ᾱt I)
   246	√
   247	√
   248	= ᾱt y0 + 1 − ᾱt ϵ
   249	
   250	(4)
   251	
   252	In the denosing process, the slightly denoised yt−1 is
   253	sampled from yt , from the distribution pθ (yt−1 |yt ). This can
   254	be approximated by a deep neural network to predict the mean
   255	and variance, and is parameterized as:
   256	
   257	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:51 UTC from IEEE Xplore. Restrictions apply.
   258	
   259	pθ (yt−1 |yt , xp , xs , zid ) = N (yt−1 ; µθ (yt , t, xp , xs , zid ),
   260	Σθ (yt , t, xp , xs , zid )).
   261	
   262	feature extractor, based on ResNet, maps inputs to specific
   263	resolutions: 32x32, 16x16, and 8x8, which are then combined
   264	with predicted noise using attention blocks.
   265	
   266	(5)
   267	Instead of directly predict the mean and variance, we predict
   268	the ϵ in Eq. 4. The loss function is defined as:
   269	2
   270	
   271	Lmse = Eyt ,t,xp ,xs ,zid ,ϵ ∥ϵ − ϵθ (yt , t, xp , xs , zid )∥ .
   272	The mean µθ can be calculated from ϵθ as:
   273	
   274	
   275	1 − αt
   276	1
   277	yt − √
   278	µθ (yt , t) = √
   279	ϵθ (yt , t) .
   280	α
   281	1 − αt
   282	
   283	(6)
   284	
   285	(7)
   286	
   287	We have ommitted various conditions in Eq. 6. In DDPM,
   288	the variance Σθ is fixed, but in the improved DDPM [14], the
   289	variance is learnable by an additional term Lvlb with relatively
   290	small weight. Substitute mean and variance into Eq. 5, the
   291	iterative denoising equation is:
   292	yt−1 = µθ (yt , t) + Σθ ϵ,
   293	
   294	ϵ ∼ N (0, I).
   295	
   296	(8)
   297	
   298	As shown in Fig. 3, we propose to use pose xp and id
   299	embedding zid as the condition for the diffusion model. These
   300	conditions are found to be effective in controlling the diffusion
   301	models, with detailed experiment results shown in Sec. III.
   302	a) Classifier-free Guidance (CFG): CFG [15] is an extended version of classifier-guidance [7], where constraint
   303	information is incorporated during training, enhancing the
   304	ability to generate detailed image structures. As is well known,
   305	CFG performs better than classifier-guidance, but requires a
   306	extra training stage. Given the relatively small scale of the
   307	re-id dataset, we introduce pose and ID embedding guidance
   308	during the training stage to ensure stricter alignment. Inspired
   309	by CFG to achieve pose and identity jointly sampling during
   310	training, we use the following equation to compute ϵθ :
   311	ϵθ = ω1 ϵθ (yt , t, xp , xs , zid ) + ω2 ϵθ (yt , t, xp )
   312	+ (1 − ω1 − ω2 )ϵθ (yt , t).
   313	
   314	III. E XPERIMENTS
   315	A. Dataset
   316	We evaluate using two public re-identification datasets:
   317	Market-1501 [18] and DukeMTMC-reID [19]. Market-1501
   318	has 12,936 training images and 19,732 gallery images across
   319	1,501 identities from six cameras. DukeMTMC-reID contains
   320	36,411 images of 1,404 identities from eight cameras. We

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf' - | nl -ba | sed -n '1,850p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Pattern Recognition 172 (2026) 112622
     2	
     3	Contents lists available at ScienceDirect
     4	
     5	Pattern Recognition
     6	journal homepage: www.elsevier.com/locate/pr
     7	
     8	Adversarial ﬂow-based generative models for visible-to-Infrared person
     9	re-Identiﬁcation
    10	Honghu Pan
    11	a
    12	
    13	a , Yongyong Chen a , Xin Li b , Zhenyu He a,∗
    14	
    15	School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen, Shenzhen, 518055, China
    16	
    17	b Peng Cheng Laboratory, Shenzhen, 518055, China
    18	
    19	a r t i c l e
    20	
    21	i n f o
    22	
    23	Keywords:
    24	Visible-to-Infrared person re-Identiﬁcation
    25	Data augmentation
    26	Flow-based generative model
    27	Adversarial training
    28	
    29	a b s t r a c t
    30	The task of visible-to-infrared (V2I) person re-identiﬁcation (ReID) presents greater challenges than visible-tovisible (V2V) ReID, primarily due to limited availability of training samples and signiﬁcant cross-modal discrepancy. To address these challenges, we propose Flow2Flow, a uniﬁed framework designed to simultaneously
    31	expand training samples and generate cross-modal images for V2I person ReID. Flow2Flow operates by learning bijective transformations from both the visible and infrared image domains to a shared isotropic Gaussian
    32	domain, utilizing invertible ﬂow-based generators for each modality. This framework enables the generation
    33	of training samples by transforming latent Gaussian noise into visible or infrared images and the generation of
    34	cross-modal images by transforming existing modality images through the latent Gaussian space into the target
    35	modality. To ensure proper identity and modality alignment of the generated images, we devise two adversarial
    36	training strategies. Speciﬁcally, we design an image encoder and a modality discriminator for each modality. The
    37	image encoder enhances the similarity between generated images and real images of the same identity through
    38	identity adversarial training, while the modality discriminator ensures the generated images are indistinguishable from real images through modality adversarial training. Experimental results on the SYSU-MM01 and RegDB
    39	datasets demonstrate that both training sample generation and cross-modal image generation substantially improve V2I ReID accuracy.
    40	
    41	1. Introduction
    42	Person re-identiﬁcation (ReID) aims to match pedestrian images captured by non-overlapping cameras, which is achieved by training a deep
    43	neural network to learn discriminative pedestrian representations, enabling cross-camera matching through feature similarity comparison.
    44	Recent advancements [1,2] in person ReID have achieved human-level
    45	accuracy on large-scale datasets [3,4]. However, these methods typically assume that pedestrian images are captured by visible-spectrum
    46	cameras in well-lit environments, limiting their eﬀectiveness in nighttime surveillance scenarios. Given that infrared radiation is unaﬀected
    47	by lighting conditions, visible-to-infrared (V2I) person ReID [5–10], a
    48	cross-spectrum or cross-modal matching task, has garnered signiﬁcant
    49	attention within the computer vision community. V2I person ReID is
    50	a critical task for 24/7 intelligent surveillance systems, enabling crossmodal matching between daytime and nighttime images. It has a wide
    51	range of applications in public security (tracking suspects or miss-
    52	
    53	ing persons across diﬀerent lighting conditions), autonomous driving
    54	(enhancing pedestrian detection in low-light environments), and smart
    55	cities (enabling seamless person retrieval in day-night surveillance networks).
    56	Unlike traditional single-modality ReID, V2I ReID must overcome
    57	signiﬁcant cross-modal discrepancies, making it more challenging yet
    58	practically indispensable. Despite recent progress in V2I ReID, the task
    59	remains highly challenging for two primary reasons. First, the modality
    60	discrepancy between the visible and infrared spectra is substantial. Visible (RGB) and infrared (grayscale, heat-based) images exhibit vast differences in texture, color, and illumination, making feature alignment
    61	diﬃcult. Existing V2I ReID methods [11–14] mainly focus on reducing this cross-modal discrepancy through cross-modal image generation,
    62	often employing generative adversarial networks (GANs) to generate
    63	the target modality from the existing one. However, it is challenging
    64	for GAN-based generators to transform infrared images, which contain
    65	limited information, into visible images, which are information-rich.
    66	
    67	∗ Corresponding author.
    68	
    69	E-mail address: 19b951002@stu.hit.edu.cn, zhenyuhe@hit.edu.cn (Z. He).
    70	
    71	https://doi.org/10.1016/j.patcog.2025.112622
    72	Received 12 February 2025; Received in revised form 14 October 2025; Accepted 15 October 2025
    73	Available online 29 October 2025
    74	0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    75	
    76	Pattern Recognition 172 (2026) 112622
    77	
    78	H. Pan et al.
    79	
    80	This challenge motivates the need for learning bijective transformations between the visible and infrared modalities. Second, the number of
    81	training images in V2I datasets [5,6] is insuﬃcient to train deep models
    82	eﬀectively. Compared to V2V ReId, most datasets in this task contain
    83	few visible and infrared images, restricting supervised learning. For instance, the training sets of SYSU-MM01 [5] and RegDB [6] contain only
    84	9929 and 2060 infrared images, respectively. Obtaining pedestrian images in real scenarios requires lots of manual collections and annotations, thereby, this paper proposes to expand training datasets by generative models.
    85	In this paper, we introduce Flow2Flow, a uniﬁed framework that facilitates both cross-modal generation and training sample generation.
    86	Our framework comprises two ﬂow-based generative models [15,16],
    87	i.e., a visible ﬂow and an infrared ﬂow, each of which learns bijective transformations from the visible or infrared image domain to a
    88	shared isotropic Gaussian domain. This design allows for the generation of visible or infrared training samples via forward ﬂow propagation from the latent noise domain to the respective image domain.
    89	Additionally, target-modality images can be generated from sourcemodality images by transforming them from the source-modality domain through the Gaussian noise domain to the target-modality domain,
    90	enabling cross-modal generation. Fig. 1 provides a schematic overview
    91	of the training sample generation and cross-modal image generation
    92	processes.
    93	To ensure invertibility and exact log-likelihood computation, existing ﬂow models [15,16] typically employ multiple 1 × 1 convolutional
    94	layers and linear coupling layers, which leads to insuﬃcient nonlinearity. To address this, we introduce an additional invertible activation layer in the ﬁnal block of both the visible and infrared ﬂows to
    95	enhance model nonlinearity. Furthermore, we propose an identity adversarial training strategy and a modality adversarial training strategy
    96	to ensure that the generated images align with speciﬁc identities and
    97	modalities. For adversarial training, we implement two discriminators
    98	for each modality: an image encoder for identity alignment and a modality discriminator for modality alignment. To achieve identity alignment
    99	between real and generated images, we minimize the distance between
   100	their encoded features during generator training, while maximizing it
   101	during discriminator training. Meanwhile, the modality discriminators
   102	work to distinguish whether images are generated or belong to a speciﬁc
   103	real modality.
   104	Experimental results on SYSU-MM01 [5] and RegDB [6] demonstrate that our method can eﬀectively improve the V2I ReID performance. Models trained on real and generated samples outperform models trained on real samples; meanwhile, models trained on dual-modal
   105	images are superior to those trained on single-modal images. The main
   106	contributions of this paper are threefold:
   107	
   108	Fig. 1. Diagram illustrating (a) training sample generation and (b) (c) crossmodal image generation. In these diagrams, images outlined by blue boxes represent fake training samples, while those by red boxes represent cross-modal
   109	images. The proposed Flow2Flow framework consists of a visible ﬂow and an
   110	infrared ﬂow, each of which learns bijective transformation from the visible or
   111	infrared image domain to a shared isotropic Gaussian domain. In (a), training
   112	sample generation is accomplished through the forward propagation of ﬂow
   113	models from the Gaussian domain to the image domains. In (b) and (c), crossmodal image generation is achieved by transforming existing modality images
   114	(𝑥𝑉𝑖 or 𝑥𝑅
   115	) through the Gaussian noise space to the target modality images (𝑥̂ 𝑉𝑖 2𝑅
   116	𝑖
   117	or 𝑥̂ 𝑅2𝑉
   118	). (For interpretation of the references to colour in this ﬁgure legend, the
   119	𝑖
   120	reader is referred to the web version of this article.)
   121	
   122	2. Related works
   123	2.1. Visible-to-visible person reid
   124	The V2V person ReID [17,18] is a single-modality image retrieval
   125	task, which aims to enlarge the inter-class variance and reduce the intraclass variance. To this end, existing methods mainly consider three levels of factors: objective-level, network-level, and data-level. For the objectives or loss functions, TriNet [2] proposed the hard triplet mining
   126	strategy on the basis of triplet loss to learn pedestrian representations;
   127	BoT [19] combined the cross entropy loss and triplet loss to train network; moreover, the center loss [20] and angular loss [21] have also
   128	been successfully applied in the V2V person ReID. For the network, early
   129	works [2] learned the global features from pedestrian images via a single CNN branch. Next, the multi-branch architecture has been adopted
   130	to learn the multi-granularity or part-level features [22,23]. Furthermore, data augmentation or generation [24,25] could also improve the
   131	ReID accuracy, which belongs to the data-based category. For example,
   132	PN-GAN [25] generated multi-pose pedestrian images via GAN model,
   133	which could reduce the pedestrian view variance; DG-Net [26] disentangled the pedestrian images into appearance and pose structure, and
   134	generated multi-pose pedestrian images for each sample by changing
   135	the pose structure; JVTC [24] conducted the online data augmentation
   136	for contrastive learning, in which the mesh projections were taken as
   137	the references to generate multi-view images.
   138	
   139	•
   140	
   141	We propose Flow2Flow, a uniﬁed framework, to jointly generate
   142	training samples and cross-modal images, which leverages a visible
   143	ﬂow and an infrared ﬂow to learn bijective transformations from
   144	image domains to a shared Gaussian domain;
   145	• For the purpose of identity alignment and modality alignment of generated images, we develop an image encoder and a modality discriminator for each modality to perform the identity adversarial training
   146	and modality adversarial training, respectively;
   147	• Experimental results show that both the training sample generation
   148	and cross-modal generation can eﬀectively improve the performance
   149	of existing V2I ReID baselines, demonstrating the eﬀectiveness and
   150	generalization of Flow2Flow.
   151	
   152	2.2. Visible-to-infrared person reid
   153	The remainder of this paper is organized as follows: Section 2 introduces recent literature related to this paper; Section 3 simply reviews
   154	theoretical backgrounds of the ﬂow-based generative models; Section 4
   155	elaborates the Flow2Flow model in detail; Section 5 presents the ablation studies, visualizations and comparisons with the SOTA; Section 6
   156	draws brief conclusions.
   157	
   158	The V2I person ReID enables cross-spectrum pedestrian retrieval,
   159	whose crux is to reduce the large cross-modal discrepancy. Existing
   160	V2I ReID methods mainly have two techniques to reduce the modal
   161	discrepancy: 1) learning the modality-shared pedestrian representation
   162	and 2) compensating information of target modality via generative
   163	2
   164	
   165	Pattern Recognition 172 (2026) 112622
   166	
   167	H. Pan et al.
   168	
   169	Fig. 2. Framework of Flow2Flow, in which the blue arrows denote training sample generation (𝑧𝑉𝑖 → 𝑥̂ 𝑉𝑖 2𝑉 and 𝑧𝑅
   170	→ 𝑥̂ 𝑅2𝑅
   171	), and the red arrows denote cross-modal
   172	𝑖
   173	𝑖
   174	𝑅2𝑉
   175	𝑅 →𝑥
   176	generation (𝑥𝑉𝑖 →𝑧𝑉𝑖 →𝑥̂ 𝑉𝑖 2𝑅 and 𝑥𝑅
   177	→𝑧
   178	̂
   179	).
   180	It
   181	consists
   182	of
   183	visible
   184	and
   185	infrared
   186	ﬂow-based
   187	generators
   188	
   189	and
   190	
   191	,
   192	𝑉
   193	𝑅 visible and infrared encoders 𝑉 and 𝑅 , visible
   194	𝑖
   195	𝑖
   196	𝑖
   197	and infrared modality discriminators 𝑉 and 𝑅 : 𝑉 or 𝑅 learns a bijective transformation from the visible domain 𝑃 (𝑋 𝑉 ) or infrared domain 𝑃 (𝑋 𝑅 ) to the
   198	latent Gaussian domain Π(𝑍);  and  encourage the generated images to match true identity and modality by identity and modality adversarial training. (For
   199	interpretation of the references to colour in this ﬁgure legend, the reader is referred to the web version of this article.)
   200	
   201	models [27–29]. The modality-shared ones [7,30–37] projected the visible and infrared pedestrian images into a shared Euclidean space, in
   202	which the intra-class similarity and inter-class similarity are maximized
   203	and minimized, respectively. For example, DDAG [38] proposed a dualattentive aggregation learning method to mine both intra-modality partlevel and cross-modality graph-level contextual cues. MPANet [7] aimed
   204	to capture the nuances of cross-modal images via a modality alleviation module and a pattern alignment module. SGIEL [32] leveraged
   205	the body shape of pedestrians as the signiﬁcant modality-shared cues,
   206	and devised a shape-erased feature learning paradigm to decorrelate
   207	modality-shared features in two orthogonal subspaces. The modality
   208	compensation ones [11,12,39,40] usually generated target modality information from existing modality data: DDRL [41] proposed an imagelevel sub-network based on GAN model, which could translate a visible
   209	(infrared) image to a corresponding infrared (visible) one; cmPIG [12]
   210	employed the set-level alignment information to generate instance alignment cross-modal paired-images; FMCNet [42] utilized the feature-level
   211	modality compensation to reduce modality discrepancy, which generated the cross-modal features rather than images.
   212	
   213	whether the data is true or fake to beat the generator. Recently, the
   214	GAN architectures have been heavily reﬁned to adapt various application scenarios. For instance, the Conditional GAN [46,47] could generate samples corresponding to speciﬁc condition labels; CycleGAN [28]
   215	enabled the unpaired cross-domain image translation by the cycle consistency loss. Meanwhile, the GAN model also showed its priority in
   216	the V2I person ReID [11,41,42] and V2I person ReID areas [24,25].
   217	Unlike the ﬂow-based model [15,16] which could exactly compute the
   218	log-likelihood of true data, GAN model implicitly minimizes the KL divergence between the true data and data generated from noises. To make
   219	the generated data indistinguishable from the real data, training a GAN
   220	model pursues an equilibrium between the generator and discriminator,
   221	which requires careful experimental setup tuning.
   222	3. Preliminaries
   223	The ﬂow-based generative model aims to learn a bijective transformation from a complex distribution 𝑋 ∼ 𝑃 (𝑋) to a simple distribution
   224	𝑍 ∼ Π(𝑍) with a known probability density function, in which 𝑋 denotes the true training data and Π(𝑍) is usually a Gaussian distribution.
   225	For the purpose of bijective mapping, the ﬂow-based model consists of
   226	a sequence of invertible generators  = 1 ∙ ⋯ ∙ 𝐿 :
   227	
   228	2.3. Flow-based generative models
   229	The ﬂow-based generative model constructs an invertible mapping
   230	from the complex distribution of true data to a simple distribution
   231	(e.g., isotropic Gaussian distribution). Layers in the ﬂow-based model
   232	should be carefully designed to match the goal of invertibility and exact log-likelihood computation. RealNVP [15] proposed the aﬃne coupling layer, which could easily compute the determinant of Jacobian
   233	matrix; Glow [16] presented an invertible 1 × 1 convolution layer, meanwhile the LU decomposition was utilized to speed up the computation of determinants; cAttnFlow [43] introduced the invertible attentions to increase the nonlinearity of ﬂow-based model. Recently, a great
   234	number of works have extended the ﬂow-based model image generation [16,44,45]. For example, on the image super-resolution ﬁeld, SRFlow [44] and HCFlow [45] took the low-resolution images as the condition, and thus learned the high-resolution images via a conditional
   235	ﬂow.
   236	
   237	𝑥𝑖 = (𝑧𝑖 ), 𝑧𝑖 = −1 (𝑥𝑖 ).
   238	
   239	(1)
   240	
   241	By the change of variable formula, 𝑃 (𝑋) and Π(𝑍) satisfy the following
   242	transformation:
   243	|
   244	|
   245	𝑃 (𝑋) = Π(𝑍)|det(𝐽−1 )|,
   246	|
   247	|
   248	
   249	(2)
   250	
   251	where det(𝐽−1 ) denotes the determinant of Jacobian matrix. Then the
   252	objective of max{log(𝑃 (𝑋))} can be converted to:
   253	𝐿
   254	∑
   255	∑
   256	|
   257	|
   258	max{ log(Π(𝑧𝑖 )) +
   259	log ||det(𝐽−1 )||}.
   260	𝑙
   261	|
   262	|
   263	𝑖
   264	𝑙=1
   265	
   266	(3)
   267	
   268	From Eq. (1), Eq. (2) and Eq. (3), we could know that the training
   269	process of the ﬂow-based model follows the reverse propagation, and
   270	the inference or generation process follows the forward propagation.
   271	A standard ﬂow-based model mainly contains two categories of layers: invertible 1 × 1 convolution layer [16] and aﬃne coupling layer [15,
   272	48]. For a single generator 𝑙 in , the reverse and forward projection
   273	of the 1 × 1 convolution layer has the following expression:
   274	
   275	2.4. Generative adversarial network
   276	The ﬁrst GAN model was proposed in [27], which consists of a generator and a discriminator, and they could improve each other by the
   277	adversarial training. In GAN model, the generator generates samples
   278	from noise variables with a known probability density function (PDF)
   279	and tries to fool the discriminator, and the discriminator distinguishes
   280	
   281	<𝑙>
   282	𝑧<𝑙−1>
   283	= 𝑊𝑙 𝑧<𝑙>
   284	= 𝑊𝑙−1 𝑧<𝑙−1>
   285	,
   286	𝑖
   287	𝑖 , 𝑧𝑖
   288	𝑖
   289	
   290	(4)
   291	
   292	where 𝑍 <0> and 𝑍 <𝐿> denote 𝑍 and 𝑋, respectively. The design of the
   293	aﬃne coupling layer should allow 1) invertible transformation and 2)
   294	3
   295	
   296	Pattern Recognition 172 (2026) 112622
   297	
   298	H. Pan et al.
   299	
   300	Table 1
   301	Notations and descriptions.
   302	Notations
   303	
   304	Descriptions
   305	
   306	𝑥𝑉𝑖 , 𝑥𝑅
   307	𝑖
   308	𝑥̂ 𝑉𝑖 2𝑉 , 𝑥̂ 𝑅2𝑅
   309	𝑖
   310	𝑉 2𝑅
   311	𝑥̂ 𝑖 , 𝑥̂ 𝑅2𝑉
   312	𝑖
   313	𝑉
   314	𝑅
   315	𝑦𝑖 , 𝑦𝑖
   316	𝑉 , 𝑅
   317	𝑧𝑉𝑖 , 𝑧𝑅
   318	𝑖
   319	𝑉 , 𝑅
   320	𝑓𝑖𝑉 , 𝑓𝑖𝑅
   321	𝑓̂𝑖𝑉 2𝑅 , 𝑓̂𝑖𝑅2𝑉
   322	𝑉 , 𝑅
   323	
   324	The true visible and infrared images
   325	The generated training samples
   326	The generated cross-modal images
   327	The identity labels of 𝑥𝑉𝑖 and 𝑥𝑅
   328	𝑖
   329	The visible and infrared generators
   330	The latent noises learned from 𝑥𝑉𝑖 and 𝑥𝑅
   331	𝑖
   332	The visible and infrared image encoders
   333	The encoded features of 𝑥𝑉𝑖 and 𝑥𝑅
   334	𝑖
   335	The encoded features of 𝑥̂ 𝑉𝑖 2𝑅 and 𝑥̂ 𝑅2𝑉
   336	𝑖
   337	The visible and infrared modality discriminators
   338	
   339	to train 𝑉 and 𝑅 : in the stage of training generators, we minimize the
   340	distance between 𝑓𝑖 and 𝑓̂𝑖 if they correspond the same identity label
   341	𝑦𝑖 ; while in the stage of training discriminators, we maximize the distance between them. The modality discriminators 𝑉 and 𝑅 aim to
   342	achieve the modality alignment for true and fake images. Compared to
   343	𝑉 and 𝑅 , 𝑉 and 𝑅 have one more CNN layer to map the features
   344	into binary modality logits. Motivated by CycleGAN [28], we devise a
   345	modality adversarial training strategy to train 𝑉 and 𝑅 : when training generators, we hope that the generated visible image 𝑥̂ 𝑅2𝑉
   346	could be
   347	𝑖
   348	classiﬁed to the real visible modality by 𝑉 ; when training discriminators, 𝑉 struggles to classify 𝑥𝑉𝑖 and 𝑥̂ 𝑅2𝑉
   349	as the real visible modality
   350	𝑖
   351	and fake visible modality, respectively. And vice versa for 𝑅 . The detailed training objectives are presented in Section 4.2.
   352	
   353	exact computation of the Jacobian determinant det(𝐽−1 ). Its reverse and
   354	forward transformation from ℝ𝑛 to ℝ𝑛 can be denoted as:
   355	
   356	4.2. Objectives for model training
   357	
   358	𝑧<𝑙−1>
   359	= 𝑧<𝑙>
   360	,
   361	𝑖(1∶𝑑)
   362	𝑖(1∶𝑑)
   363	𝑧<𝑙−1>
   364	= 𝑧<𝑙>
   365	⊙ sigmoid(𝑆Θ (𝑧<𝑙>
   366	)) + 𝑇Θ (𝑧<𝑙>
   367	);
   368	𝑖(𝑑+1∶𝑛)
   369	𝑖(1∶𝑑)
   370	𝑖(1∶𝑑)
   371	𝑖(1∶𝑑)
   372	
   373	Our Flow2Flow architecture consists of two types of models: 1) the
   374	ﬂow-based generators 𝑉 and 𝑅 that map the true images into latent
   375	noise; 2) discriminators (𝑉 , 𝑅 , 𝑉 and 𝑅 ) that encourage generators to generate images corresponding to speciﬁc identities and speciﬁc
   376	modalities via adversarial training.
   377	Training ﬂow-based generators. By the ﬂow objective of Eq. (3),
   378	to maximize the log-likelihood of the training data, we need to maximize the log-likelihood of latent noises and the log-determinants of Jacobian matrices. Maximizing log-likelihood of latent noises is equivalent
   379	to minimizing its negative log-likelihood (NLL). The NLL of Gaussian dis∑
   380	tribution Π(𝑍) can be denoted as 𝑚2 log(2𝜋) + 𝑚 log(𝜎) + 2𝜎1 2 (𝑧𝑖 − 𝜇)2 ,
   381	where 𝑚, 𝜇 and 𝜎 are the number of samples, mean and standard deviation, respectively. Here we omit the ﬁrst two terms since they are
   382	constant terms. Then the ﬂow loss can be denoted as:
   383	∑
   384	|
   385	|
   386	2
   387	1 ∑ 𝑉
   388	𝐿𝑓 𝑙𝑜𝑤 =
   389	(𝑧𝑖 − 𝜇) −
   390	log ||det(𝐽−1 )||
   391	2
   392	𝑉
   393	2𝜎
   394	|
   395	|
   396	(9)
   397	∑
   398	|
   399	|
   400	2
   401	1 ∑ 𝑅
   402	+
   403	(𝑧𝑖 − 𝜇) −
   404	log ||det(𝐽−1 )||.
   405	2
   406	𝑅 |
   407	2𝜎
   408	|
   409	
   410	(5)
   411	
   412	𝑧<𝑙>
   413	= 𝑧<𝑙−1>
   414	,
   415	𝑖(1∶𝑑)
   416	𝑖(1∶𝑑)
   417	𝑧<𝑙>
   418	= (𝑧<𝑙−1>
   419	− 𝑇Θ (𝑧<𝑙−1>
   420	))∕sigmoid(𝑆Θ (𝑧<𝑙−1>
   421	)).
   422	𝑖(𝑑+1∶𝑛)
   423	𝑖(𝑑+1∶𝑛)
   424	𝑖(1∶𝑑)
   425	𝑖(1∶𝑑)
   426	
   427	(6)
   428	
   429	In Eq. (5) and Eq. (6), 𝑆Θ and 𝑇Θ are learning-based neural networks,
   430	and sigmoid is the activation function. Readers could refer to the original
   431	papers [15,16,48] for more algorithm details.
   432	4. Methodologies
   433	4.1. Model architecture
   434	This paper aims to combine the ﬂow-based generative model and
   435	adversarial training to jointly achieve expansion of training samples
   436	and cross-modal image generation. To this end, we devise a Flow2Flow
   437	model as shown in Fig. 2, which consists of a visible ﬂow 𝑉 , an infrared
   438	ﬂow 𝑅 , visible encoder 𝑉 , infrared encoder 𝑅 , visible modality discriminator 𝑉 and infrared modality discriminator 𝑅 . To make this
   439	paper clear, we present the notations and their corresponding descriptions in Table 1.
   440	In our Flow2Flow architecture, the visible models (𝑉 , 𝑉 and 𝑉 )
   441	and their infrared counterparts have the same network structure but do
   442	not share weights. Motivated by GAN models, we adopt the adversarial
   443	training to generate high-quality fake images: 𝑉 and 𝑅 denote the
   444	generators, while encoders 𝑉 and 𝑅 , modality discriminators 𝑉 and
   445	𝑅 refer to as the discriminator models.
   446	For the generators 𝑉 and 𝑅 , we ﬁrst implement multiple invertible
   447	1 × 1 convolution layers and aﬃne coupling layers as the standard ﬂowbased model [15,16] of Section 3. Moreover, to increase the nonlinearity
   448	of the standard ﬂow, we propose an extra invertible nonlinear activation
   449	layer 𝜙. Speciﬁcally, for the reverse propagation from 𝑍 <𝑙> to 𝑍 <𝑙−1> ,
   450	the nonlinear layer follows tanh activation function:
   451	𝑧<𝑙−1>
   452	=
   453	𝑖
   454	
   455	<𝑙>
   456	exp{𝑧<𝑙>
   457	𝑖 } − exp{−𝑧𝑖 }
   458	<𝑙>
   459	exp{𝑧<𝑙>
   460	𝑖 } + exp{−𝑧𝑖 }
   461	
   462	.
   463	
   464	In addition to maximizing log(𝑃 (𝑋)), we add a cluster constraint on
   465	the latent noises, which encourages noises with the same identity to
   466	be close to each other. Speciﬁcally, we minimize the distance between
   467	intra-class noises and maximize that between inter-class ones:
   468	1 ∑
   469	1 ∑
   470	𝐿𝑛𝑜𝑖𝑠𝑒 =
   471	𝑑(𝑧𝑖 , 𝑧𝑗 ) −
   472	𝑑(𝑧𝑖 , 𝑧𝑘 ),
   473	(10)
   474	𝑛1 𝑦 =𝑦
   475	𝑛2 𝑦 ≠𝑦
   476	𝑖
   477	
   478	where 𝑧𝑖 can be 𝑧𝑉𝑖
   479	
   480	1 + 𝑧<𝑙−1>
   481	1
   482	𝑖
   483	log
   484	.
   485	2
   486	1 − 𝑧<𝑙−1>
   487	
   488	𝑖
   489	
   490	𝑘
   491	
   492	or 𝑧𝑅
   493	𝑖 ; 𝑑(⋅, ⋅) indicates the Euclidean distance; 𝑛1
   494	
   495	and 𝑛2 denote the number of intra-class pairs and inter-class pairs, respectively. Then the total generator loss 𝐿 for training the visible and
   496	infrared ﬂows can be deﬁned as the combination of the ﬂow loss and
   497	latent noise loss:
   498	𝐿 = 𝐿𝑓 𝑙𝑜𝑤 + 𝜆𝐿𝑛𝑜𝑖𝑠𝑒 .
   499	
   500	(11)
   501	
   502	Identity adversarial training of image encoders. The encoders 𝑉
   503	and 𝑅 belong to the discriminator models, thereby their weights are
   504	frozen when training generators. To achieve the set-level alignment, 𝑉
   505	and 𝑅 encourage the fake images generated by 𝑉 and 𝑅 could be
   506	similar to true images of the same identity:
   507	(
   508	)
   509	1 ∑
   510	𝑅
   511	𝐿 = 𝑉
   512	𝑑 𝑉 (𝑥𝑉𝑖 ), 𝑉 (𝑉 (−1
   513	𝑅 (𝑥𝑗 )))
   514	𝑛1 𝑦𝑉 =𝑦𝑅
   515	𝑖
   516	𝑗
   517	(12)
   518	(
   519	)
   520	1 ∑
   521	−1 𝑉
   522	+ 𝑅
   523	𝑑 𝑅 (𝑥𝑅
   524	𝑖 ), 𝑅 (𝑅 (𝑉 (𝑥𝑗 ))) .
   525	𝑛1 𝑦𝑅 =𝑦𝑉
   526	
   527	(7)
   528	
   529	The forward propagation from 𝑍 <𝑙−1> to 𝑍 <𝑙> has the following expression:
   530	𝑧<𝑙>
   531	=
   532	𝑖
   533	
   534	𝑗
   535	
   536	(8)
   537	
   538	𝑖
   539	
   540	We call 𝑉 and 𝑅 with the activation layer as the nonlinear ﬂow. The
   541	reverse propagation, forward propagation and log-determinant of three
   542	main components in 𝑉 and 𝑅 are summarized in Table 2.
   543	The encoders 𝑉 and 𝑅 aim to achieve the identity alignment for
   544	true and fake images with the same identity. They are composed of multiple CNN layers, which encode the true image 𝑥𝑖 and fake image 𝑥̂ 𝑖 as 𝑓𝑖
   545	and 𝑓̂𝑖 , respectively. We devise an identity adversarial training strategy
   546	
   547	𝑖
   548	
   549	𝑗
   550	
   551	In Eq. (12), (⋅) denotes the encoded feature, such as 𝑓𝑖𝑉 = 𝑉 (𝑥𝑉𝑖 ) and
   552	𝑉
   553	𝑅
   554	𝑓̂𝑗𝑅2𝑉 = 𝑉 (𝑉 (−1
   555	(𝑥𝑅
   556	𝑗 ))); 𝑛1 and 𝑛1 denote the number of intra-class
   557	𝑅
   558	visible pairs and intra-class infrared pairs, respectively.
   559	In the stage of training discriminators, we freeze the weights of 𝑉
   560	and 𝑅 and update the weights of 𝑉 and 𝑅 . At this time, we minimize
   561	4
   562	
   563	Pattern Recognition 172 (2026) 112622
   564	
   565	H. Pan et al.
   566	
   567	Table 2
   568	The reverse propagation, forward propagation and log-determinant of three main components in the nonlinear ﬂow.
   569	Description
   570	
   571	Reverse propagation
   572	
   573	Forward propagation
   574	
   575	Invertible convolution layer [16]
   576	
   577	𝑧<𝑙−1>
   578	= 𝑊𝑙 𝑧<𝑙>
   579	𝑖
   580	𝑖
   581	
   582	𝑧<𝑙>
   583	= 𝑊𝑙−1 𝑧<𝑙−1>
   584	𝑖
   585	𝑖
   586	
   587	Aﬃne coupling layer [15]
   588	
   589	𝑧<𝑙−1>
   590	= 𝑧<𝑙>
   591	𝑖(1∶𝑑)
   592	𝑖(1∶𝑑)
   593	𝑠 = sigmoid(𝑆Θ (𝑧<𝑙>
   594	))
   595	𝑖(1∶𝑑)
   596	𝑡 = 𝑇Θ (𝑧<𝑙>
   597	)
   598	𝑖(1∶𝑑)
   599	𝑧<𝑙−1>
   600	= 𝑧<𝑙>
   601	⊙𝑠+𝑡
   602	𝑖(𝑑+1∶𝑛)
   603	𝑖(1∶𝑑)
   604	
   605	𝑧<𝑙>
   606	= 𝑧<𝑙−1>
   607	𝑖(1∶𝑑)
   608	𝑖(1∶𝑑)
   609	𝑠 = sigmoid(𝑆Θ (𝑧<𝑙−1>
   610	))
   611	𝑖(1∶𝑑)
   612	𝑡 = 𝑇Θ (𝑧<𝑙−1>
   613	)
   614	𝑖(1∶𝑑)
   615	𝑧<𝑙>
   616	= (𝑧<𝑙−1>
   617	− 𝑡)∕𝑠
   618	𝑖(𝑑+1∶𝑛)
   619	𝑖(1∶𝑑)
   620	
   621	Nonlinear activation layer
   622	
   623	𝑧<𝑙−1>
   624	= exp{𝑧𝑖<𝑙> }−exp{−𝑧𝑖<𝑙> }
   625	𝑖
   626	
   627	exp{𝑧<𝑙> }−exp{−𝑧<𝑙> }
   628	𝑖
   629	
   630	𝑖
   631	
   632	𝑖
   633	
   634	Algorithm 1: Training Process of Flow2Flow.
   635	
   636	▿𝜃 |𝑑(𝑅 (𝑥𝑅 ), 𝑅 (𝑅 (−1
   637	(𝑥𝑉 )))|
   638	𝑉
   639	
   640	(15)
   641	
   642	1
   643	
   644	and 𝑥𝑉𝑖 corresponding to identity label 𝑦𝑉𝑖 , we ﬁrst feed them into 𝑉
   645	2
   646	
   647	for reverse propagation to learn their respective latent noise 𝑧𝑉𝑖 and 𝑧𝑉𝑖 :
   648	1
   649	
   650	𝑉
   651	𝑉
   652	−1 𝑉
   653	𝑧𝑉𝑖 = −1
   654	𝑉 (𝑥𝑖 ), 𝑧𝑖 = 𝑉 (𝑥𝑖 ).
   655	1
   656	
   657	1
   658	
   659	2
   660	
   661	2
   662	
   663	2
   664	
   665	(16)
   666	
   667	We then obtain a fake visible image 𝑥̂ 𝑉𝑖 2𝑉 corresponding to identity 𝑦𝑉𝑖
   668	by the following equation:
   669	𝑝
   670	𝑥̂ 𝑉𝑖 2𝑉 = 𝑉 (𝑧𝑉𝑖 + (𝑧𝑉𝑖 − 𝑧𝑉𝑖 )),
   671	(17)
   672	1
   673	1
   674	𝑞 2
   675	where 𝑝, 𝑞 ∈ 𝑁 + and 𝑞 > 𝑝 ≥ 1. For simplicity, we omit the process of
   676	generating fake infrared image 𝑥̂ 𝑅2𝑅
   677	from 𝑥𝑅
   678	𝑖
   679	𝑖 .
   680	Cross-modal image generation aims to generate images of target
   681	modality from images of existing modality. Given a visible image 𝑥𝑉𝑖 ,
   682	we ﬁrst learns its latent noise 𝑧𝑉𝑖 by the reverse propagation of 𝑉 , then
   683	generate its corresponding infrared image 𝑥̂ 𝑉𝑖 2𝑅 by the forward propagation of 𝑅 :
   684	
   685	end
   686	14
   687	for 𝑘 steps do
   688	15
   689	Update 𝑉 , 𝑅 , 𝑉 , 𝑅 by descending gradients:
   690	[
   691	]
   692	16
   693	▿𝜃 ‖1 − 𝑉 (𝑥𝑉 )‖22 + ‖𝑉 (𝑥̂ 𝑅2𝑉 )‖22
   694	[
   695	]
   696	2
   697	2
   698	𝑅
   699	𝑉
   700	2𝑅
   701	17
   702	▿𝜃 ‖1 − 𝑅 (𝑥 )‖2 + ‖𝑅 (𝑥̂
   703	)‖2
   704	𝑅 )))|
   705	18
   706	▿𝜃 |2 − 𝑑(𝑉 (𝑥𝑉 ), 𝑉 (𝑉 (−1
   707	(𝑥
   708	𝑅
   709	19
   710	▿𝜃 |2 − 𝑑(𝑅 (𝑥𝑅 ), 𝑅 (𝑅 (−1
   711	(𝑥𝑉 )))|
   712	𝑉
   713	20
   714	end
   715	21 until converged;
   716	
   717	𝑉
   718	𝑧𝑉𝑖 = −1
   719	̂ 𝑉𝑖 2𝑅 = 𝑅 (𝑧𝑉𝑖 ).
   720	𝑉 (𝑥𝑖 ), 𝑥
   721	
   722	(18)
   723	
   724	Similarly, we could generate a visible image 𝑥̂ 𝑅2𝑉
   725	from a given infrared
   726	𝑖
   727	image 𝑥𝑅
   728	by
   729	the
   730	following
   731	equations:
   732	𝑖
   733	−1 𝑅
   734	𝑧𝑅
   735	̂ 𝑅2𝑉
   736	= 𝑉 (𝑧𝑅
   737	𝑖 = 𝑅 (𝑥𝑖 ), 𝑥
   738	𝑖
   739	𝑖 ).
   740	
   741	Modality adversarial training of modality discriminators. When
   742	training generators, the weights of 𝑉 and 𝑅 are frozen. We hope the
   743	generated image 𝑥̂ 𝑅2𝑉
   744	or 𝑥̂ 𝑉𝑖 2𝑅 can be classiﬁed to the visible or infrared
   745	𝑖
   746	modality by 𝑉 or 𝑅 :
   747	
   748	𝑉
   749	+ (1 − 𝑅 (𝑅 (−1
   750	𝑉 (𝑥𝑖 )))),
   751	
   752	)
   753	
   754	Thanks to the invertibility property of the ﬂow-based models,
   755	Flow2Flow of Fig. 2 could jointly achieve training sample generation
   756	and cross-modal image generation.
   757	Training sample generation aims to generate images corresponding to speciﬁc identities. To this end, we fully exploit the invertibility
   758	of ﬂow and latent space interpolation technique. For visible images 𝑥𝑉𝑖
   759	
   760	13
   761	
   762	𝑅
   763	𝐿 = (1 − 𝑉 (𝑉 (−1
   764	𝑅 (𝑥𝑖 ))))
   765	
   766	2
   767	
   768	4.3. Image generation
   769	
   770	Input: 𝑥𝑉 and 𝑥𝑅 , hyperparameters 𝑘 and 𝑘
   771	Output: Generators 𝑉 and 𝑅 , discriminators 𝑉 , 𝑅 , 𝑉 and
   772	𝑅
   773	1 repeat
   774	2
   775	𝑧𝑉 ← −1
   776	(𝑥𝑉 ), 𝑧𝑅 ← −1
   777	(𝑥𝑅 )
   778	𝑉
   779	𝑅
   780	𝑉
   781	2𝑅
   782	𝑉
   783	𝑅2𝑉
   784	3
   785	𝑥̂
   786	← 𝑅 (𝑧 ), 𝑥̂
   787	← 𝑉 (𝑧𝑅 )
   788	4
   789	for 𝑘 steps do
   790	5
   791	Update 𝑉 and 𝑅 by descending gradients:
   792	[
   793	]
   794	∑
   795	∑
   796	2
   797	6
   798	▿𝜃 2𝜎1 2 (𝑧𝑉 − 𝜇) − log | det(𝐽−1 )|
   799	𝑉
   800	[
   801	]
   802	∑
   803	∑
   804	2
   805	7
   806	▿𝜃 2𝜎1 2 (𝑧𝑅 − 𝜇) − log | det(𝐽−1 )|
   807	𝑅
   808	∑
   809	∑
   810	8
   811	▿𝜃 𝑛1 𝑦𝑖 =𝑦𝑗 𝑑(𝑧𝑖 , 𝑧𝑗 ) − 𝑛1 𝑦𝑖 ≠𝑦𝑘 𝑑(𝑧𝑖 , 𝑧𝑘 )
   812	2]
   813	[1
   814	9
   815	▿𝜃 1 − 𝑉 (𝑉 (−1
   816	(𝑥𝑅 )))
   817	𝑅
   818	[
   819	]
   820	−1
   821	10
   822	▿𝜃 1 − 𝑅 (𝑅 (𝑉 (𝑥𝑉 )))
   823	12
   824	
   825	4⋅exp{2⋅𝑧<𝑙>
   826	}
   827	𝑖
   828	(exp{2⋅𝑧<𝑙>
   829	}+1)
   830	𝑖
   831	
   832	The training details of Flow2Flow are summarized in Algorithm 1, in
   833	which 𝑘 and 𝑘 are predeﬁned hyperparameters for adversarial training. In the stage of training ﬂow-based generators, we have three categories of losses: latent space constraint from Eq. (9) and Eq. (10), identity adversarial losses from Eq. (12), and modality adversarial losses
   834	from Eq. (14). In the stage of training discriminators, we have two categories of losses: identity adversarial losses from Eq. (13), and modality
   835	adversarial losses from Eq. (15).
   836	
   837	(13)
   838	
   839	in which features learned by the encoders are normalized to unit-length,
   840	so that the distance is within the interval [0, 2].
   841	
   842	▿𝜃 |𝑑(𝑉 (𝑥𝑉 ), 𝑉 (𝑉 (−1
   843	(𝑥𝑅 )))|
   844	𝑅
   845	
   846	(log
   847	
   848	(
   849	)
   850	𝑉

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Adversarial perturbation and defense for generalizable person re-identification.pdf' - | nl -ba | sed -n '1,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Neural Networks 186 (2025) 107287
     2	
     3	Contents lists available at ScienceDirect
     4	
     5	Neural Networks
     6	journal homepage: www.elsevier.com/locate/neunet
     7	
     8	Full Length Article
     9	
    10	Adversarial perturbation and defense for generalizable person
    11	re-identification
    12	Hongchen Tan a , Kaiqiang Xu b , Pingping Tao c , Xiuping Liu b ,∗
    13	a
    14	
    15	Institute of Future Technology, Dalian University of Technology, Dalian, Dalian 116024, China
    16	School of Mathematical Sciences, Dalian University of Technology, Dalian 116024, China
    17	c
    18	Shandong University, Weihai 264209, China
    19	b
    20	
    21	ARTICLE
    22	
    23	INFO
    24	
    25	Keywords:
    26	Generalizable person Re-ID
    27	Hard matching samples
    28	Semantics purification
    29	Metric perturbation
    30	
    31	ABSTRACT
    32	In the Domain Generalizable Person Re-Identification (DG Re-ID) task, the quality of identity-relevant
    33	descriptor is crucial for domain generalization performance. However, for hard-matching samples, it is
    34	difficult to separate high-quality identity-relevant feature from identity-irrelevant feature. It will inevitably
    35	affect the domain generalization performance. Thus, in this paper, we try to enhance the model’s ability
    36	to separate identity-relevant feature from identity-irrelevant feature of hard matching samples, to achieve
    37	high-performance domain generalization. To this end, we propose an Adversarial Perturbation and Defense
    38	(APD) Re-identification Method. In the APD, to synthesize hard matching samples, we introduce a MetricPerturbation Generation Network (MPG-Net) grounded in the concept of metric adversariality. In the MPG-Net,
    39	we try to perturb the metric relationship of samples in the latent space, while preserving the essential visual
    40	details of the original samples. Then, to capture high-quality identity-relevant feature, we propose a Semantic
    41	Purification Network (SP-Net). The hard matching samples synthesized by MPG-Net is used to train the SPNet. In the SP-Net, we further design the Semantic Self-perturbation and Defense (SSD) Scheme, to better
    42	disentangle and purify identity-relevant feature from these hard matching samples. Above all, through extensive
    43	experimentation, we validate the effectiveness of the APD method in the DG Re-ID task.
    44	
    45	1. Introduction
    46	Person re-identification (Re-ID) endeavors to accurately identify a
    47	specific individual across non-overlapping cameras, despite variations
    48	in viewpoints, times, and locations (Liu, Feng, Chen, & Hu, 2023;
    49	Ning, Wang, Wang, Zhang, & Ning, 2023; Wang, Huang, Yang, Tiwari,
    50	& Zhang, 2024; Zhu et al., 2024). While employing deep learning
    51	method, person Re-ID techniques (Li, Zhang, Tian, Wang, & Gao, 2022;
    52	Tan, Liu, Bian, Wang, & Yin, 2022; Tan, Liu, Yin, & Li, 2023) have
    53	exhibited significant performance on various publicly available benchmarks. However, a pivotal limitation remains; most of these methods
    54	are trained and evaluated on identical datasets/domains. Consequently,
    55	when deployed in practical scenarios involving new environments or
    56	unseen domains, their performance often suffers a substantial decline.
    57	This performance degradation is primarily attributed to the significant
    58	discrepancy between the source/training and target/test domains.
    59	To realize effective Re-ID in new target domains, recent DG ReID methods (Dai, Li, Liu, Tong, & yu Duan, 2021; Li, Zhang, Hu,
    60	Zhang, & Yu, 2024; Liao & Shao, 2022; Zhuang et al., 2020) integrate
    61	multiple source samples or multiple domain-specific models to cover
    62	
    63	data distribution from various domains as much as possible. This
    64	approach attempts to improve domain generalization by increasing the
    65	amount of data/models to approximate or cover the distribution of
    66	the target scenario. In addition, disentanglement-based methods (Eom
    67	& Ham, 2019; Jin, Lan, Zeng, Chen, & Zhang, 2020; Zhang et al.,
    68	2021; Yi-Fan Zhang et al., 2021; Zou, Yang, Yu, Kumar, & Kautz,
    69	2020) believe that the key obstacle to domain generalization is the
    70	identity-irrelevant feature. To this, they (Eom & Ham, 2019; Jin et al.,
    71	2020; Zhang, Lan, et al., 2021; Zhang, Zhang, et al., 2021; Zou et al.,
    72	2020) rely on a hypothesis that identity-irrelevant features are those whose
    73	intra-class spacing is greater than their inter-class spacing, to disentangle
    74	the identity-relevant and -irrelevant feature, and only use the former
    75	for DG Re-ID. However, their assumption is correct only when the
    76	feature space is well-designed. Specifically, before a well-trained model
    77	and effective feature space are obtained, it is difficult to differentiate identity-relevant feature from identity-irrelevant ones. Especially
    78	when there are many hard matching samples, their identity-relevant
    79	and -irrelevant components are heavily entangled and hard to separate. Since this, the identity-relevant descriptor, which contains some
    80	
    81	∗ Corresponding author.
    82	
    83	E-mail addresses: tanhongchenphd@bjut.edu.cn (H. Tan), 459553299@qq.com (K. Xu), pingping.tao@sdu.edu.cn (P. Tao), xpliu@dlut.edu.cn (X. Liu).
    84	https://doi.org/10.1016/j.neunet.2025.107287
    85	Received 15 August 2024; Received in revised form 10 December 2024; Accepted 13 February 2025
    86	Available online 22 February 2025
    87	0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    88	
    89	Neural Networks 186 (2025) 107287
    90	
    91	H. Tan et al.
    92	
    93	Fig. 1. Flowchart of the APD method. We first train the MPG-Net using Source Domain data. MPG-Net adopts the metric adversariality strategy to synthesize hard matching
    94	samples (i.e. ‘‘New Source Domain’’ in Figure) based on Source Domain data. Secondly, SP-Net accommodates the training of hard matching samples synthesize by MPG-Net and
    95	achieve high-quality person feature capturing capabilities. Finally, SP-Net performs person matching on the Unseen Domain.
    96	
    97	misclassified identity-irrelevant feature, is prone to interference from
    98	non-key factors of target domain such as background feature. Recently,
    99	a serious of outstanding methods (Chen, Li, Wu, Liang, & Jha, 2021;
   100	Wang, Wang, Zhu, & Wang, 2022; Zhu et al., 2023) have demonstrated
   101	that enhancing the model’s ability to perceive and capture effective
   102	semantics from difficult samples can effectively improve the model’s
   103	generalization capability. Therefore, inspired by them (Chen et al.,
   104	2021; Wang et al., 2022; Zhu et al., 2023), we try to enhance the ability
   105	of the Re-ID model to separate identity-relevant feature from identityirrelevant feature of hard matching samples, to capture high-quality
   106	identity-relevant feature applications in the face of unseen domains.
   107	Above all, we propose a novel Adversarial Perturbation and Defense (APD) Re-identification Method for the Domain Generalization
   108	person Re-identification (DG Re-ID) task. In the APD, we firstly try
   109	to synthesize hard matching samples; we use these hard matching
   110	samples to train the Re-ID model; Secondly, we try to separate and
   111	purify the identity-relevant feature. Flowchart of the APD method is
   112	shown in Fig. 1. For synthesizing hard matching samples, we design a
   113	Metric-Perturbation Generation Network (MPG-Net). In the MPG-Net,
   114	we aim to achieve a scenario where the inter-class distances between
   115	samples of different person IDs are smaller than the intra-class distances
   116	between samples of the same person ID, essentially adopting an inverse
   117	strategy to metric learning. Moreover, during the generation process,
   118	we adopt a shallow content consistency strategy to maintain minimal
   119	variations in the person appearance. To capture high-quality identityrelevant feature, we propose a Semantic Purification Network (SP-Net).
   120	And hard matching samples synthesized by MPG-Net is used to the
   121	SP-Net. In the SP-Net, we propose a Semantic Self-perturbation and
   122	Defense (SSD) Scheme to separate identity-relevant and -irrelevant
   123	feature; and we purify the identity-relevant feature through a feature
   124	attack manner. Specifically, based on the constraint that the intra-class
   125	distance is less than the inter-class distance, the feature captured by the
   126	model is the identity-relevant feature; otherwise, it is identity-irrelevant
   127	feature; We further use identity-irrelevant feature to perform a disorder
   128	perturb on identity-relevant feature to purify and improve the quality of
   129	identity-relevant feature. This work makes the following contributions:
   130	(i) We design a Metric-Perturbation Generation Network (MPG-Net)
   131	to synthesize hard matching person samples.
   132	(ii) We design a Semantic Purification Network (SP-Net) to perceive
   133	and separate identity-relevant and -irrelevant feature from these hard
   134	matching samples.
   135	(iii) Extensive experiments on the many datasets demonstrate that
   136	our APD achieves the competitive performance. A series of ablation studies have also validated that each design component significantly improves the performance of Domain Generalization person
   137	Re-identification (DG Re-ID) task.
   138	
   139	data to bolster the model’s adaptability across domains. Style Transferbased approaches (Chen, Zhu, & Gong, 2019; Huang, Wu, Xu, & Zhong,
   140	2019; Lin, Wu, Yan, Xu, & Yang, 2020; Zhong, Zheng, Zheng, Li,
   141	& Yang, 2019) initially leverage Generative Adversarial Networks
   142	(GANs) (Goodfellow et al., 2014) to transform the style of unlabeled
   143	target data to match the source data. This style-translated source data
   144	is then used to train the Re-ID model, improving its robustness to
   145	domain shifts. Attribute Recognition-based methods (Li, Chen, Tao, Yu, &
   146	Qi, 2021; Wang, Zhu, Gong, & Li, 2018; Xu, Luo, & Hu, 2021) initially
   147	utilize a Re-ID model to extract person descriptors. Subsequently, an
   148	attribute detector is employed to extract attribute embeddings. These
   149	person descriptors are then combined with the attribute embeddings
   150	to facilitate person matching in the target domain, leveraging the
   151	discriminative power of attributes. Pseudo Label-based methods (Dai, Liu,
   152	Bai, Tong, & Duan, 2021; Feng et al., 2021; Zheng, Liu, He, Mei, Luo,
   153	& Zha, 2021; Zheng et al., 2021) typically begin by utilizing a trained
   154	Re-ID model to perform sample clustering through various clustering
   155	strategies. Pseudo-labels are then predicted for unlabeled target data,
   156	enabling the utilization of these pseudo-labeled target domain data to
   157	fine-tune the Re-ID model. However, these methods still heavily rely on
   158	data collection and often perform poorly in unseen scenarios, limiting
   159	their practical applicability and generalization capabilities.
   160	2.2. Domain generalization person Re-ID
   161	Due to their practical relevance, Domain Generalization (DG) person
   162	Re-ID methods have garnered increasing attention in recent years.
   163	Meta Learning-based approaches (Choi, Kim, Jeong, Park, & Kim,
   164	2021; Zhao et al., 2021) incorporate meta-learning and normalization
   165	techniques (Huang & Belongie, 2017) to mimic real train–test domain
   166	shifts and bolster generalization capabilities. The normalization technique, in particular, can mitigate identity-independent style variations
   167	and extract identity-relevant descriptors, effectively mitigating the impact of different scene styles on generalization performance. However,
   168	these methods often overlook the influence of different normalization
   169	techniques and their positioning on the corresponding person image
   170	features. Additionally, normalization techniques can potentially lose
   171	parts of identity-relevant feature.
   172	Ensemble Learning-based methods (Dai, Li, et al., 2021; Liao & Shao,
   173	2022; Mancini, Bulò, Caputo, & Ricci, 2018; Zhuang et al., 2020) leverage multiple source domain data to train the Re-ID model, breaking
   174	the confinement of a single source domain’s style. For instance, Song,
   175	Yang, Song, Xiang, and Hospedales (2019) constructed a large-scale
   176	domain generalization person Re-ID database and employed a metalearning training scheme (Finn, Abbeel, & Levine, 2017) along with a
   177	proposed domain invariant mapping network. However, the additional
   178	mapping network can slow down inference. In response, Tamura and
   179	Murakami (2019) proposed a lightweight data augmentation selection
   180	strategy that is easily applicable to other models or tasks. Nevertheless, these methods heavily rely on the collection and construction of
   181	databases, making it challenging to predict whether the collected data
   182	will adequately cover the target data distribution.
   183	
   184	2. Related work
   185	2.1. Domain adaptation person Re-ID
   186	To enhance the domain generalization capability of the model,
   187	Domain Adaptation Re-ID methods strive to collect unlabeled target
   188	2
   189	
   190	Neural Networks 186 (2025) 107287
   191	
   192	H. Tan et al.
   193	
   194	main visual content of samples unchanged, under perturbing sample
   195	metric process. Next, we will gradually achieve the aforementioned
   196	objective through specific formulations and equations.
   197	First, we describe the sample partitioning strategy. As shown in
   198	Fig. 2-(a), we denote the training dataset as 𝑋 = {𝑥1 , 𝑥2 , … , 𝑥𝑁 }, where
   199	𝑁 is the number of person samples. For each sample 𝑥𝑖 ∈ 𝑋, we build
   200	its negative sample subset 𝐴𝑖 by randomly picking 𝑀 persons whose ID
   201	are different from 𝑥𝑖 . Namely, 𝐴𝑖 contains 𝑀 negative samples, and is
   202	⋃ ⋃ ⋃
   203	denoted as 𝐴𝑖 = {𝑥𝑖𝑗 |𝑗 = 1, 2, … , 𝑀} (𝐴 = 𝐴1 𝐴2 𝐴3 ⋯). Note
   204	that 𝑥𝑖𝑗 ∈ 𝐴𝑖 is the negative sample of 𝑥𝑖 , and 𝐴𝑖 is the set of 𝑀
   205	randomly selected negative samples of 𝑥𝑖 .
   206	Second, we adopt GAN to push visual semantics of anchor sample
   207	𝑥𝑖 to approximate its negative samples 𝐴𝑖 = {𝑥𝑖𝑗 |𝑗 = 1, 2, … , 𝑀}. The
   208	generator loss between sample 𝑥𝑖 and negative sample 𝑥𝑖𝑗 is defined as:
   209	1
   210	𝐿𝐺 = − E𝑥𝑖 [𝑙𝑜𝑔 𝐷(𝐺(𝑥𝑖 ))].
   211	(1)
   212	2
   213	
   214	Fig. 2. Schematic diagram of the MPG-Net. In sub-figure (a), we present the planning
   215	of the samples and the negative sample set to prepare for the subsequent modification
   216	of sample feature. The sub-figure (b) shows that we aim to reduce the distance between
   217	samples of the same person ID and increase the distance between samples of different
   218	person IDs in the deep latent space. In sub-figure (c), while the metric/distances changes
   219	between samples, the person’s appearance feature should not change dramatically.
   220	
   221	𝐿𝐺 drives anchor sample 𝐺(𝑥𝑖 ) to approximate the negative sample
   222	set 𝐴𝑖 , aiming to fool the discriminator 𝐷. The discriminator is trained
   223	to classify the input image into the ‘‘Fake/True’’ category by
   224	1
   225	1
   226	𝐿𝐷 = − E𝑥𝑖 [𝑙𝑜𝑔(1 − 𝐷(𝐺(𝑥𝑖 )))] − E𝑥𝑖𝑗 [𝑙𝑜𝑔(𝐷(𝑥𝑖𝑗 ))].
   227	(2)
   228	2
   229	2
   230	
   231	Disentanglement-based methods (Eom & Ham, 2019; Jin et al., 2020;
   232	Zhang, Lan, et al., 2021; Zhang, Zhang, et al., 2021; Zou et al., 2020)
   233	strive to separate identity-relevant features from identity-irrelevant
   234	ones, utilizing only the former for person Re-ID. However, as mentioned
   235	in Section 1, under their current framework, identity-relevant feature
   236	are often misinterpreted as identity-irrelevant, leading to inaccuracies.
   237	To address this, based on the semantic disentanglement, we try to
   238	purify identity-relevant feature through the semantics perturbation
   239	strategy. Furthermore, we also adopt hard matching samples to further
   240	push our proposed Re-ID model to refine and improve person descriptor
   241	extraction.
   242	
   243	For achieving the metric perturbation, we intentionally make (see
   244	Fig. 2-(b)): (1) the distance between the synthesized sample 𝐺(𝑥𝑖 ) and
   245	the negative sample 𝑥𝑖𝑗 to be as close as possible, and (2) the distance
   246	between the synthesized sample 𝐺(𝑥𝑖 ) and its original sample 𝑥𝑖 to be
   247	as far as possible. Therefore, we define a metric perturbation loss 𝐿𝑀 𝑃
   248	by revising the triplet loss (Schroff, Kalenichenko, & Philbin, 2015):
   249	𝐿𝑀 𝑃 = 𝑆 𝑃 (𝑑(𝜓(𝐺(𝑥𝑖 )), 𝜓(𝑥𝑖𝑗 )) − 𝑑(𝜓(𝐺(𝑥𝑖 )), 𝜓(𝑥𝑖 ))).
   250	
   251	(3)
   252	
   253	Here 𝑆 𝑃 (𝑥) = 𝑙𝑛(1 + 𝑒𝑥 ), distance 𝑑(𝑎, 𝑏) is the euclidean distance
   254	between vectors 𝑎 and 𝑏, and 𝜓(⋅) is the deep neural layer to extract person semantics. To improve the quality of deep semantics, we adopted
   255	ResNet-50 pre-trained on ImageNet (Jia et al., 2009) to construct the
   256	𝜓(⋅). Note that the 𝜓(⋅) parameter is frozen and does not participate
   257	in model training. We use ‘‘ResNet-50 Stage 4’’ to capture the deep
   258	semantics. With the design of this strategy, in the deep latent space,
   259	the distances between samples of the same ID will be increased, while
   260	the distances between samples of different IDs will be reduced.
   261	When using 𝐿𝑀 𝑃 to adjust the inter-/intra-class spacing in the latent
   262	space, the content of the sample 𝐺(𝑥𝑖 ) could get modified and distorted
   263	through this adjustment. Third, we would like to keep basic content
   264	including the structure and body silhouette of the person relatively
   265	stable (see Fig. 2-c). We know that in deep models, shallow features
   266	primarily contains the content aspects of a person’s appearance. Thus,
   267	we design a Content Consistency loss
   268	
   269	3. Method
   270	Flowchart of the APD method is shown in Fig. 1. To this end,
   271	our APD contains two components: a Metric-Perturbation Generation
   272	Network (MPG-Net) and a Semantic Purification Network (SP-Net).
   273	The MPG-Net is designed to adopt the metric adversariality strategy
   274	to synthesize hard matching samples. The SP-Net is designed to better accommodate the training of hard matching samples and achieve
   275	high-quality person feature capturing capabilities.
   276	3.1. MPG-Net
   277	As delineated in Section 1, our first objective is to synthesize hard
   278	matching samples. Consequently, in this section, we strive to realize a
   279	scenario where the inter-class distances between samples of different
   280	person IDs are smaller than the intra-class distances between samples
   281	of the same person ID. It essentially reverses the conventional metric learning strategy. In addition, while the metric/distances changes
   282	between samples, the person’s appearance feature should not change
   283	dramatically. To achieve the goal, we propose the Metric-Perturbation
   284	Generation Network (MPG-Net).
   285	As an excellent generation model, Generative Adversarial Networks
   286	(GANs) (Goodfellow et al., 2014) have outstanding performance in
   287	modifying content and semantics. Therefore, we build our MPG-Net
   288	based on GANs. However, simply using GAN does not effectively modify the metric space of samples from the person ID perspective. To this,
   289	we try to design a sample partitioning strategy to build a negative sample set for each sample. And then GAN is used to push visual feature of
   290	each sample to approximate its negative samples. To perturb the sample
   291	metric, we further design the metric perturbation constraint to increase
   292	intra-class spacing and narrow inter-class spacing. However, such a
   293	strategy will seriously distort or change the key semantics of persons,
   294	which is not conducive to the capture of high-quality descriptors. Thus,
   295	we further to introduce the content consistency constraint to keep the
   296	
   297	𝐿𝐶 = ‖𝜑(𝐺(𝑥𝑖 )) − 𝜑(𝑥𝑖 )‖1 ,
   298	
   299	(4)
   300	
   301	to penalize basic content change after modification. The 𝜑(⋅) is a
   302	shallow neural layer to encode the basic content. We adopted ‘‘ResNet50 Stage 2’’ pre-trained on ImageNet (Jia et al., 2009) to construct the
   303	𝜑(⋅). Note that the 𝜑(⋅) parameter is frozen and does not participate in
   304	model training.
   305	In addition, we also need to ensure the stability of samples’ visual semantics. We hope that if 𝐺(𝑥𝑖 ) does further sample synthesis
   306	(i.e., 𝐺(𝐺(𝑥𝑖 ))), the visual semantics should be consistent with 𝐺(𝑥𝑖 ).
   307	Thus, the loss can be defined as
   308	𝐿𝑆 = ‖𝐺(𝐺(𝑥𝑖 )) − 𝐺(𝑥𝑖 )‖1 .
   309	
   310	(5)
   311	
   312	Finally, generator has the combined loss 𝐿𝑀 𝑃 𝐺 = 𝐿𝐺 + 𝜆1 𝐿𝑀 𝑃 +
   313	𝜆2 𝐿𝑆 + 𝜆3 𝐿𝐶 . And the discriminator adopts the cross-entropy loss,
   314	𝐿𝐷𝑀 𝑃 𝐺 = 𝐿𝐷 . Based on control variable strategy of parameter selection,
   315	we set 𝜆1 = 2.0, 𝜆2 = 1.0, and 𝜆3 = 1.0.
   316	3
   317	
   318	Neural Networks 186 (2025) 107287
   319	
   320	H. Tan et al.
   321	
   322	Fig. 3. The Architecture of the proposed SP-Net. LPP: Linear Projection of Patches; MSP: Multi-scale Semantic Perception Module; ViT: ViT-Transformer; CEL: Cross-entropy loss;
   323	TL: Triplet Loss.
   324	
   325	3.2. SP-Net
   326	To better perceive and capture high-quality person descriptor from
   327	these hard matching samples generated by the MPG-Net, we propose
   328	a Semantic Purification Network (SP-Net). The Network Structure of
   329	SP-Net is shown in Fig. 3. It contains several components (Fig. 3):
   330	Backbone (Dosovitskiy et al., 2021), a Multi-scale Semantic Perception (MSP) Module, and a Semantic Self-perturbation and Defense
   331	(SSD) Scheme. Next, we introduce the design details of each strategy
   332	respectively.
   333	3.2.1. Baseline/backbone
   334	We construct our SP-Net based on the ViT-Transformer architecture.
   335	Given a person image 𝑥 ∈ R𝐻×𝑊 ×𝐶 , first, the image 𝑥 is divided into
   336	𝑁 fixed-sized and non-overlapped patches 𝑥 = {𝑥𝑘 |𝑘 = 1, 2, … , 𝑄};
   337	then, these patches are fed to a Linear Projection of Patches (LPP)
   338	module, which outputs patch descriptors  (𝑥) = { (𝑥𝑘 )|𝑘 = 1, 2, … , 𝑄};
   339	then,  (𝑥) is flattened and combined with position embedding , and
   340	then sent to the ViT-Transformer; finally, the ViT-Transformer outputs a
   341	global person descriptor 𝑓 . The baseline is trained by the Cross-entropy
   342	loss and the Triplet loss (Schroff et al., 2015). More details can be found
   343	in Dosovitskiy et al. (2021), He, Luo, Wang, Wang, Li, and Jiang (2021).
   344	
   345	Fig. 4. Structure of the proposed Multi-scale Semantic Perception (MSP) Module. The
   346	MSP module is designed to quickly realize the perception and interaction between
   347	feature in a two-dimensional structure using convolutional receptive fields of different
   348	sizes, to supplement the subsequent Transformer with some structured contextual
   349	feature. To achieve a larger receptive field without increasing computational load, we
   350	adopt Dilated Convolution to build the MSP module.
   351	
   352	3.2.2. Multi-scale semantic perception (MSP) module
   353	The ViT-Transformer can quickly perceive and capture global feature of image. However, the ‘‘flatten’’ operation in Transformer destroys the two-dimensional structure of the image. Thus, Transformer’s
   354	perception of local structured contextual feature is not as good as
   355	Convolution. Although position embedding can complement contextual
   356	feature to a certain extent, the contextual feature of neighboring feature
   357	in a two-dimensional structure is still lacking. Therefore, we consider
   358	introducing the convolutional module into ViT-Transformer. However,
   359	too many convolutional modules tend to lose the dependency between
   360	local feature, and too few convolutional modules have insufficient
   361	local feature awareness. Compared to the general convolutional structure, the Inception structure (Szegedy et al., 2015) can simultaneously
   362	perceive contextual feature at different scales. For the size of final
   363	Re-ID model, it is impossible to stack the Inception model and the
   364	Transformer model. Inspired by the Inception structure, we attempt to
   365	design such an idea into a lightweight attention mechanism to achieve
   366	rapid perception of multi-scale feature.
   367	Before the Transformer performs ‘‘flatten’’ operation, we leverage
   368	convolutional operations to design a Multi-scale Semantic Perception
   369	(MSP) Module (as shown in Fig. 4), to help ViT-Transformer supplement structured contextual feature. In the MSP Module, to achieve a
   370	larger receptive field without increasing computational load, we adopt
   371	dilated convolution to build this MSP Module. As shown in Fig. 4, the
   372	specific formalized expression of MSP Module is:
   373	𝑀 𝑆 𝑃1 (𝑥) = 𝑆 𝑖.(𝐷𝐶1 ( (𝑥))),
   374	
   375	where 𝐷𝐶𝑖 (⋅), 𝑖 = 1 … 4 are dilated convolution layers, 𝑆 𝑖.(⋅) is the
   376	sigmoid activation function, 𝑀 𝑆 𝑃𝑖 (𝑥), 𝑖 = 1, 2, 3 are the attention
   377	weights, and 𝐾(𝑥) is the identity-relevant feature. 𝐷𝐶1 and 𝐷𝐶3 have
   378	a kernel size of 3, and a dilation rate of 2. 𝐷𝐶2 and 𝐷𝐶4 have a kernel
   379	size of 2, and a dilation rate of 2.
   380	3.2.3. Semantic self-perturbation and defense (SSD) scheme
   381	Although the MSP is expected to better perceive contextual and
   382	global feature, it still cannot capture effective discriminative feature
   383	in the face of a series of hard matching samples. As mentioned in
   384	Section 1, an effective pattern to face hard matching samples is to
   385	separate identity-relevant feature from identity-irrelevant feature and
   386	utilize the identity-relevant feature for person matching. Currently,
   387	many disentanglement based methods (Eom & Ham, 2019; Jin et al.,
   388	2020; Zhang, Lan, et al., 2021; Zhang, Zhang, et al., 2021; Zou et al.,
   389	2020) can disentangle the identity-relevant and -irrelevant feature.
   390	However, their designs are prone to misclassification of identity feature.
   391	To relieve the issue, we propose a Semantic Self-perturbation and
   392	Defense (SSD) Scheme to separate identity-relevant and -irrelevant
   393	feature; and we purify the identity-relevant feature through a feature
   394	attack manner. Specifically, based on the constraint that the intra-class
   395	distance is less than the inter-class distance, the feature captured by the
   396	model is the identity-relevant feature; otherwise, it is identity-irrelevant
   397	feature; We further use identity-irrelevant feature to perform a disorder
   398	perturb on identity-relevant feature to purify and improve the quality
   399	of identity-relevant feature.
   400	Semantic Disentanglement Strategy (SDS). To filter out identityrelevant and -irrelevant feature, we also conduct the semantic disentanglement strategy following Eom and Ham (2019), Jin et al. (2020),
   401	
   402	𝑀 𝑆 𝑃2 (𝑥) = 𝑆 𝑖.(𝐷𝐶2 (𝑆 𝑖.(𝐷𝐶3 ( (𝑥))))),
   403	𝑀 𝑆 𝑃3 (𝑥) = 𝑆 𝑖.(𝐷𝐶4 ( (𝑥))),
   404	1
   405	𝐾(𝑥) = ⋅ (𝑀 𝑆 𝑃1 (𝑥) + 𝑀 𝑆 𝑃2 (𝑥) + 𝑀 𝑆 𝑃3 (𝑥)) ⊙  (𝑥),
   406	3
   407	
   408	(6)
   409	
   410	4
   411	
   412	Neural Networks 186 (2025) 107287
   413	
   414	H. Tan et al.
   415	
   416	Zhang, Lan, et al. (2021), Zhang, Zhang, et al. (2021), Zou et al. (2020).
   417	However, unlike them, we separate identity-relevant and -irrelevant
   418	feature only at the shallow stage. This is because when the network
   419	becomes deeper, the ID-relevant and -irrelevant feature in Transformer
   420	become entangled and cannot be untangled easily. At the same time,
   421	the MSP is designed to quickly perceive structured contextual feature
   422	at different scales; and it is located in the shallow stage of the entire
   423	network. Therefore, we introduce a Semantic Disentanglement Strategy (SDS) to separate identity-relevant feature from identity-irrelevant
   424	feature in the MSP stage.
   425	In the SDS, we represent the identity-relevant feature as 𝐾(𝑥) and
   426	the identity-irrelevant feature as 𝑁 𝐾(𝑥) =  (𝑥) − 𝐾(𝑥). It is not enough
   427	to only represent identity-relevant and identity-irrelevant feature. The
   428	separation of feature requires subsequent models and loss for response
   429	to drive it. Next, we will introduce the extraction of subsequent feature
   430	and the setting of loss functions.
   431	For Identity-relevant Feature, 𝐾(𝑥) are firstly fed into the ‘‘Flatten’’ operation, and then the first ViT-Transformer branch 𝑉 𝑖𝑇1 (See Fig. 3). This
   432	ViT-Transformer branch 𝑉 𝑖𝑇1 outputs an identity-relevant descriptor 𝑓 ;
   433	This branch is trained using the Cross-Entropy (CE) loss and the Triplet
   434	(Tri.) loss. The loss for identity-relevant descriptor 𝑓 can be denoted
   435	as 𝐿𝐾 = 𝐿𝑇 𝑟𝑖𝑝𝑙𝑒𝑡−𝐾 + 𝐿𝐶 𝐸−𝐾 . Specifically, for a triplet set {𝑥𝑎 , 𝑥𝑝 , 𝑥𝑛 }
   436	where 𝑓𝑎 is the anchor sample, identity-relevant descriptors of positive
   437	sample pair are {𝑓𝑎 , 𝑓𝑝 }, and identity-relevant descriptors of negative
   438	sample pair are {𝑓𝑎 , 𝑓𝑛 }, the Triplet loss is shown as
   439	𝐿𝑇 𝑟𝑖𝑝𝑙𝑒𝑡−𝐾 = 𝑆 𝑃 (𝑑(𝑓𝑎 , 𝑓𝑝 ) − 𝑑(𝑓𝑎 , 𝑓𝑛 )).
   440	
   441	identity-relevant feature. To relieve this issue, some methods (Gao,
   442	Wang, Lu, & Liu, 2020; Sun et al., 2021) adopted human masks
   443	to capture the person body feature. However, they usually require
   444	either additional mask labels (often unavailable) or pretrained maskgeneration models. But this pretrained model is often trained on
   445	another segmentation dataset, which introduces another domain gap
   446	from the Re-ID dataset. From another perspective, ‘‘Adversity Makes
   447	a Man Wise’’; and in the person image with mixed feature, the Re-ID
   448	model indeed adaptively selects effective feature conducive to person
   449	matching. However, this pattern of the Re-ID model is not enough
   450	for hard matching person samples. Thus, inspired by above statement,
   451	we try to adopt the identity-irrelevant feature to attack/perturb the
   452	person matching of the identity-relevant feature, to drive the Re-ID
   453	model capture more robust descriptors to defend this perturbation. To
   454	this end, we design a Semantic Perturbation Strategy (SPS). Next, we
   455	describe the construction details.
   456	(i) We randomly shuffle the identity-irrelevant feature by person
   457	ID in the training batch. Specifically, the 𝑁 𝐾(𝑥) is randomly shuffled
   458	̂
   459	according to person ID; the shuffled 𝑁 𝐾(𝑥) is denoted as the 𝑁 𝐾(𝑥).
   460	(ii) The shuffled identity-irrelevant feature 𝑁 ̂
   461	𝐾(𝑥) is used to attack/perturb the identity-relevant feature 𝐾(𝑥) by 𝐾 ∗ (𝑥) = 𝑁 ̂
   462	𝐾(𝑥) +
   463	𝐾(𝑥). Notice that the person ID for 𝑁 𝐾(𝑥) and 𝐾(𝑥) is not the same.
   464	(iii) The perturbed person feature 𝐾 ∗ (𝑥) is fed into the ‘‘Flatten’’ operation, then the second ViT-Transformer branch 𝑉 𝑖𝑇2 (See Fig. 3), which
   465	outputs the person descriptor 𝑓 ∗ . We constrain the person ID classification of the descriptor 𝑓 ∗ to not get affected by the identity-irrelevant
   466	̂
   467	feature 𝑁 𝐾(𝑥)
   468	during training. Hence, the loss for the descriptor 𝑓 ∗ can
   469	be denoted as 𝐿𝐾 ∗ = 𝐿𝑇 𝑟𝑖𝑝𝑙𝑒𝑡−𝐾 ∗ + 𝐿𝐶 𝐸−𝐾 ∗ . Here 𝐿𝑇 𝑟𝑖𝑝𝑙𝑒𝑡−𝐾 ∗ is the triplet
   470	loss (Eq. (7)), 𝐿𝐶 𝐸−𝐾 ∗ is the Cross Entropy loss (Eq. (8)).
   471	Based on the Semantic Self-perturbation and Defense (SSD) Scheme,
   472	SP-Net is drived to purify and capture the robust identity-relevant feature. Accordingly, the proportion of identity-relevant feature misjudged
   473	as identity-irrelevant feature will be smaller and smaller.
   474	
   475	(7)
   476	
   477	Here 𝑆 𝑃 (𝑥) = 𝑙𝑛(1+𝑒𝑥 ) is a monotonically increasing function that helps
   478	numerical optimization by avoiding negative loss values, the distance
   479	𝑑(𝑎, 𝑏) is the Euclidean distance between vectors 𝑎 and 𝑏. The CE loss
   480	is shown as
   481	𝐿𝐶 𝐸−𝐾 = −𝑦𝑎 ⋅ 𝑙𝑜𝑔(𝑝(𝑥𝑎 )).
   482	
   483	(8)
   484	
   485	Here the 𝑦𝑎 is the person ID label of the anchor sample 𝑥𝑎 .
   486	For Identity-irrelevant Feature, 𝑁 𝐾(𝑥) also is fed into the ‘‘Flatten’’
   487	operation, and then fed into the third ViT-Transformer branch 𝑉 𝑖𝑇3 (See
   488	Fig. 3). This branch 𝑉 𝑖𝑇3 outputs identity-irrelevant descriptor 𝑓 𝑛 . To
   489	disentangle the identity-irrelevant feature, (1) we constrain the intraclass spacing of samples to be greater than the inter-class spacing by
   490	adopting the Triplet loss, and (2) we classify anchor sample 𝑥𝑎 to the
   491	person ID of the negative sample 𝑥𝑛 in the CE loss. The losses for
   492	identity-irrelevant descriptor 𝑓 𝑛 can be denoted as 𝐿𝑁 𝐾 = 𝐿𝑇 𝑟𝑖𝑝𝑙𝑒𝑡−𝑁 𝐾 +
   493	𝐿𝐶 𝐸−𝑁 𝐾 . Specifically, for a triplet set {𝑥𝑎 , 𝑥𝑝 , 𝑥𝑛 } where 𝑓𝑎𝑛 is the
   494	anchor sample, identity-irrelevant descriptors of positive sample pair
   495	are {𝑓𝑎𝑛 , 𝑓𝑝𝑛 }, and identity-irrelevant descriptors of negative sample pair
   496	are {𝑓𝑎𝑛 , 𝑓𝑛𝑛 }, we modified the Triplet loss to
   497	𝐿𝑇 𝑟𝑖𝑝𝑙𝑒𝑡−𝑁 𝐾 = 𝑆 𝑃 (𝑑(𝑓𝑎𝑛 , 𝑓𝑛𝑛 ) − 𝑑(𝑓𝑎𝑛 , 𝑓𝑝𝑛 )).
   498	
   499	3.2.4. Loss functions in SP-Net
   500	The loss functions for the SP-Net can be summarized as
   501	𝐿𝑆 𝑃 = 𝜆4 𝐿𝐾 + 𝜆5 𝐿𝑁 𝐾 + 𝜆6 𝐿𝐾 ∗ ,
   502	
   503	Based on control variable strategy of parameter selection, we set 𝜆4 =
   504	1.0, 𝜆5 = 0.5, and 𝜆6 = 0.1. To further enhance the robustness
   505	of the descriptor and reduce learnable parameters, these three ViTTransformer branches share parameters. In the testing stage, we only
   506	extract identity-relevant descriptor 𝑓 to conduct the person matching
   507	in unseen target domain (see Fig. 5).
   508	4. Experimental results
   509	
   510	(9)
   511	
   512	We evaluated the proposed APD qualitatively and quantitatively.
   513	Firstly, we compare the performance of the APD with many DG person
   514	Re-ID methods. Secondly, we conduct ablation studies to validate the
   515	effectiveness of each strategy.
   516	
   517	Besides, we exchange the person ID of the sample 𝑥𝑎 and the sample 𝑥𝑛 ,
   518	and use the Cross-Entropy (CE) loss to optimize the model. Specifically,
   519	the CE loss is defined as
   520	𝐿𝐶 𝐸−𝑁 𝐾 = −𝑦𝑛(𝑎≠𝑛) ⋅ 𝑙𝑜𝑔(𝑝𝑎𝑛 (𝑥𝑎 )).

codex
三篇已经读完主要方法段：第一篇的文本抽取顺序有两栏错位，我会补读算法和生成小节后半段；第二篇方法比较完整；第三篇是短会论文，方法到第 3 页结束。现在继续读剩下三篇。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '1,930p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Attribute Guidance With Inherent Pseudo-label For
     2	Occluded Person Re-identification
     3	Rui Zhia , Zhen Yanga and Haiyang Zhanga, *
     4	
     5	arXiv:2508.04998v1 [cs.CV] 7 Aug 2025
     6	
     7	a Beijing University of Post and Telecommunication
     8	
     9	Abstract. Person re-identification (Re-ID) aims to match person
    10	images across different camera views, with occluded Re-ID addressing scenarios where pedestrians are partially visible. While pretrained vision-language models have shown effectiveness in Re-ID
    11	tasks, they face significant challenges in occluded scenarios by focusing on holistic image semantics while neglecting fine-grained
    12	attribute information. This limitation becomes particularly evident
    13	when dealing with partially occluded pedestrians or when distinguishing between individuals with subtle appearance differences.
    14	To address this limitation, we propose Attribute-Guide ReID (AGReID), a novel framework that leverages pre-trained models’ inherent capabilities to extract fine-grained semantic attributes without
    15	additional data or annotations. Our framework operates through a
    16	two-stage process: first generating attribute pseudo-labels that capture subtle visual characteristics, then introducing a dual-guidance
    17	mechanism that combines holistic and fine-grained attribute information to enhance image feature extraction.
    18	Extensive experiments demonstrate that AG-ReID achieves stateof-the-art results on multiple widely-used Re-ID datasets, showing
    19	significant improvements in handling occlusions and subtle attribute
    20	differences while maintaining competitive performance on standard
    21	Re-ID scenarios.
    22	
    23	1
    24	
    25	Introduction
    26	
    27	Person Re-Identification (ReID) is a computer vision task that aims
    28	to match person images across different camera views [27]. With the
    29	rapid development of smart cities and surveillance systems, ReID
    30	has become increasingly crucial in various real-world applications,
    31	processing millions of surveillance videos daily. It requires the extraction of unique features from images to recognize the same person, even when there are alterations in pose, lighting, and viewpoint.
    32	Among various challenges in ReID, occlusion presents a particularly
    33	significant obstacle. This necessitates advanced algorithms capable
    34	of effectively managing partial occlusions while maintaining precise
    35	identification capabilities.
    36	Existing approaches to address occlusion in ReID can be categorized based on their feature extraction strategies. Global methods,
    37	which are commonly used in general ReID tasks, learn a single feature to represent the entire image. While effective in standard scenarios, these methods often fail to handle occluded regions effectively
    38	as they cannot distinguish between visible and occluded parts. To
    39	address this limitation, researchers have proposed part-based techniques [20, 11, 17] and attribute-based methods. Part-based tech∗ Corresponding Author. Email: zhhy@bupt.edu.cn
    40	
    41	Figure 1. Comparison of retrieval results between CLIP-ReID and
    42	AG-ReID on challenging cases. Case1: A person wearing a black-red jacket
    43	with subtle color variations. Case2: A person with an umbrella in occluded
    44	scenario. For each case, the first column shows the query image, followed by
    45	top-10 retrieval results. Incorrect matches (marked as ’0’) are outlined in red,
    46	while correct matches (marked as ’1’) are outlined in green.
    47	
    48	niques focus on comparing visible body parts between images, which
    49	helps to identify individuals even when some parts are occluded. Recent studies [4, 28, 26] have shown that attribute-based methods can
    50	significantly enhance model performance by leveraging detailed visual characteristics such as clothing style, accessories, and physical
    51	features. These two approaches have shown promising results in handling occluded scenarios, but they inevitably require additional supervision signals, such as pose estimation data, human parsing labels,
    52	or manually annotated attributes. This dependence on extra annotation data significantly limits their practical application in large-scale
    53	scenarios.
    54	Recent advances in pre-trained vision-language models present a
    55	promising direction for addressing these limitations. Studies [10, 28]
    56	have established that models like CLIP (Contrastive LanguageImage Pre-training) [14] are highly effective in extracting comprehensive semantic information from images, achieving significant performance improvements in ReID tasks. By learning combined representations of text and visuals through contrastive learning on massive image-text pairs, these models excel at grasping complex semantic concepts and transferring knowledge to downstream tasks.
    57	However, in the context of occlusion ReID tasks, these models face
    58	similar challenges as global methods. Relying solely on holistic in-
    59	
    60	formation from textual prompts to guide image feature extraction
    61	has significant limitations. This approach leads to the omission of
    62	fine-grained semantic information within the image, causing the pretrained model to focus only on primary features that may be partially visible or occluded. As a result, performance degrades substantially in occlusion scenarios. As illustrated in Figure 1, the tendency
    63	of CLIP-ReID to focus on holistic features while overlooking finegrained details significantly impacts its ability to distinguish between
    64	individuals with similar appearances or identify partially occluded
    65	persons.
    66	This paper explores the potential of mining fine-grained feature
    67	semantic information from the pre-trained vision-language model
    68	CLIP to improve its performance in occluded ReID tasks. Our approach leverages attribute guidance, thereby eliminating the necessity for additional data or capabilities and representing a promising avenue for enhancing the model’s efficacy in this challenging
    69	domain. We propose a novel attribute-guided method called AGReID that supports this challenging task through a two-stage training process. In the first stage, AG-ReID acquires image attribute
    70	pseudo-labels by leveraging the detailed semantic information within
    71	CLIP through context optimization. In the second stage, it guides
    72	the extraction of image features with both holistic and fine-grained
    73	attribute information, improving the retrieval performance of the
    74	model. Specifically, the dual guidance includes: 1) Attribute-prompt
    75	guidance: using the overall attribute prompt text feature to guide image features through contrastive learning. 2) Fine-grained attribute
    76	pseudo-label guidance: the learnable tokens are implicitly trained
    77	through the CoOp [31] method to obtain fine-grained semantics,
    78	thereby guiding the extraction of image features. Furthermore, we
    79	propose an attribute encoder for aligning image features with attribute pseudo-labels, and an attribute loss for measuring the semantic difference between them. To handle inconsistent features in occluded scenarios, we introduce a noise-masking mechanism that selectively considers attribute pairs based on their semantic similarity.
    80	The efficacy of AG-ReID was assessed through experimentation on
    81	multiple well-known occluded and holistic ReID datasets. The results demonstrated that AG-ReID outperformed a number of existing
    82	methods. The primary contributions of this work are summarized as
    83	follows:
    84	
    85	• To the best of our knowledge, this is the first attempt to improve occluded ReID by embedding fine-grained attribute semantic into image features through the inherent capabilities of the
    86	CLIP model, without the need for extra data or annotations, significantly enhancing model performance in occlusion scenarios.
    87	• We present a novel attribute dual-guidance ReID framework
    88	called AG-ReID, which effectively guides the extraction of image features through both holistic text features and fine-grained attribute pseudo-labels, improving feature accuracy and robustness.
    89	• We propose an innovative method that implicitly trains attribute
    90	pseudo-labels through context optimization, and design corresponding attribute encoder module and attribute loss to achieve
    91	effective alignment between image features and attribute pseudolabels.
    92	• We conduct evaluations on multiple challenging datasets, including Occluded-ReID, P-Duke and MSMT17, demonstrating that
    93	AG-ReID achieves state-of-the-art performance in both occluded
    94	and holistic scenarios.
    95	
    96	2
    97	
    98	Related Works
    99	
   100	2.1 Pre-trained Vision-Language Learning
   101	Pre-trained vision-language models are a class of machine learning
   102	models trained on large-scale datasets to understand and process both
   103	visual and textual data. These models can understand the relationships between images and text, enabling them to perform a variety
   104	of tasks such as image captioning, visual question answering, and
   105	image-text retrieval. The power of these models lies in their ability
   106	to generalize well to various downstream tasks. This is achieved by
   107	pre-training on extensive datasets, which enables the models to learn
   108	a wide range of visual-textual relationships. Once pre-trained, these
   109	models can be fine-tuned on specific tasks, making them versatile
   110	and effective in real-world applications.
   111	The dual encoder architecture is a prevalent pre-trained visionlanguage model architecture that employs two separate unimodal
   112	encoders to independently process images and text. It utilizes shallow attention layers or dot products to align the embeddings of both
   113	modalities into a unified semantic space, enhancing efficiency in
   114	tasks like image-text retrieval. Nonetheless, the limited depth of interaction between the modalities can pose challenges in complex
   115	visual-language understanding tasks. Both the widely used CLIP and
   116	ALIGN [7] models incorporate this dual encoder architecture.
   117	Contrastive Language–Image Pre-training (CLIP) is a pre-trained
   118	model introduced by OpenAI that efficiently learns visual concepts
   119	from natural language supervision and is adaptable to various downstream tasks [14]. It consists of two encoders, an Image Encoder
   120	and a Text Encoder. Image encoders, with architectures like ViT
   121	[3] or ResNet50 [5], are designed to transform images from a highdimensional RGB space into a low-dimensional embedding space.
   122	The text encoder converts each word in the given prompt to a unique
   123	numeric ID, maps IDs into embedding vectors, and finally encodes
   124	them to a text feature that contains prompt semantic information.
   125	During the training phase, CLIP optimizes a symmetric cross entropy loss to achieve the target of maximizing the cosine similarity
   126	for matched pairs while minimizing the cosine similarity for all other
   127	unmatched pairs.
   128	One of the challenges in applying pre-trained models to downstream tasks is the time-consuming and domain-expertise-required
   129	prompt engineering. Context Optimization (CoOp) automates this
   130	process by modeling prompt context words using learnable vectors
   131	while keeping the pre-trained parameters frozen [31]. CoOp significantly enhances prompt engineering performance and demonstrates robust domain generalization capabilities compared to manual
   132	prompts. In essence, CoOp transforms static text prompts into learnable text templates. It acquires and learns text descriptions directly
   133	through the intrinsic multi-modal abilities of the pre-trained model,
   134	avoiding intricate manual word tuning. Specifically, the prompt embedding given to the text encoder is designed with trainable tokens
   135	and fixed tokens, where the number and position of trainable tokens
   136	can be adjusted according to the requirements of downstream tasks.
   137	
   138	2.2 Attribute-based Person ReID
   139	Person ReID has been extensively studied due to its critical applications in surveillance and security. Traditional person ReID methods
   140	primarily focus on visual features extracted from images, and combine additional information such as pose, body mask, and visible infrared to address occlusion, light changes, and more [20, 19, 17, 22].
   141	Similarly, recent research has explored the use of fine-grained at-
   142	
   143	tributes, such as clothing color, accessories, and physical characteristics, to enhance ReID performance.
   144	Attribute-based Person ReID leverages these descriptive attributes
   145	to provide additional semantic information, improving the robustness
   146	and accuracy of person matching. Notable work in this area includes
   147	artificially annotating image attribute features in existing datasets [9],
   148	using language models to automatically generate and utilize attribute
   149	descriptions [26], and integrating multi-modal data to bridge the gap
   150	between textual and visual information [28]. Recent advances have
   151	also explored prompt-guided approaches for feature disentangling in
   152	occluded scenarios [2] and text-based multi-granularity contrastive
   153	learning for occluded person ReID [24].
   154	In addition, recent work has also explored the interpretability of
   155	attribute-guided methods. For example, AMD method [1] provides
   156	post-mortem explanations for existing ReID models by identifying
   157	and quantifying the contributions of different attributes.
   158	
   159	3
   160	
   161	Method
   162	
   163	3.1 Overview
   164	This section provides a detailed introduction to the AG-ReID model,
   165	which leverages the rich semantic information inherent in pre-trained
   166	models to enhance image retrieval performance for occluded person
   167	ReID tasks. Our framework operates through a two-stage process
   168	that aligns visual and textual modalities. In the first stage, we establish a semantic bridge between images and their fine-grained attributes, generating attribute pseudo-labels that capture subtle visual
   169	characteristics. In the second stage, we introduce a dual-guidance
   170	mechanism that combines holistic attribute-prompt features and finegrained attribute pseudo-labels to enhance image feature extraction.
   171	To handle inconsistent features in occluded scenarios, we propose
   172	a noise-masking mechanism that selectively focuses on reliable attribute matches while filtering out those affected by occlusions.
   173	
   174	3.2 Cross-Modal Alignment Stage
   175	3.2.1
   176	
   177	Preliminary
   178	
   179	To adapt the text encoder for specific downstream tasks without
   180	extensive prompt engineering, we leverage a technique that incorporates learnable context vectors into the prompt embedding. Specifically, the input Ti to the text encoder ET is structured to include r
   181	trainable vectors [v] alongside k − r fixed tokens [t]. The number r
   182	and the position of [v] can be adjusted based on task requirements:
   183	Ti = [t]i1 [v]i1 ...[v]ir [t]ik−r
   184	
   185	During fine-tuning on a downstream task, only these learnable vectors [v] are optimized, utilizing the semantic knowledge encoded
   186	within the frozen pre-trained model parameters to capture taskrelevant information.
   187	
   188	3.2.2
   189	
   190	Ti = [t]i1 [t]i2 ...[t]ik = Embd(pi )
   191	
   192	(1)
   193	
   194	fiT = ET (Ti )
   195	
   196	(2)
   197	
   198	During pre-training, such architectures typically optimize a symmetric contrastive loss to align the image and text representations
   199	in a shared embedding space. This objective maximizes the cosine
   200	similarity for matched image-text pairs while minimizing it for unmatched pairs:
   201	s(fM , fT ) = L2(projM (fM )) · L2(projT (fT ))T
   202	
   203	(3)
   204	
   205	where projM and projT are projection layers, L2 denotes L2normalization, and t is a learned temperature parameter.
   206	
   207	Attribute Prompt Template
   208	
   209	To effectively capture fine-grained attributes of person images, we
   210	design a set of attribute prompt templates that cover various visual
   211	characteristics. These templates are constructed using a combination
   212	of fixed and learnable tokens, following the CoOp framework. The
   213	attribute prompt templates are designed to describe different aspects
   214	of a person’s appearance, including:
   215	• Clothing attributes (e.g., color, style, pattern)
   216	• Accessories (e.g., bags, hats, glasses)
   217	• Body characteristics (e.g., height, build)
   218	• Pose and movement
   219	Each attribute prompt template follows the structure:
   220	Tattr = [t]prefix [v]1 [v]2 ...[v]r [t]suffix
   221	
   222	(5)
   223	
   224	where [t]prefix and [t]suffix are fixed tokens that provide context, and
   225	[v]1 through [v]r are learnable tokens that adapt to capture specific attribute information. The number of learnable tokens r is determined
   226	based on the complexity of the attribute being described.
   227	
   228	3.2.3
   229	
   230	Our approach is built upon a dual-encoder architecture designed to
   231	process visual and textual information. Specifically, i ∈ {1...B}
   232	denotes the index within a batch. Given the image batch M =
   233	{m1 , m2, ..., mB } and their text descriptions P = {p1 , p2, ..., pB },
   234	an image encoder EM maps images to 512-d features fiM =
   235	EM (mi ). For text prompts pi , each prompt is first tokenized (e.g.,
   236	using byte pair encoding) and then embedded into 512-d word tokens
   237	Ti via an embedding layer Embd. These tokens are subsequently processed by a text encoder ET to produce the final text feature fiT .
   238	
   239	(4)
   240	
   241	Attribute Pseudo-label Generation
   242	
   243	Inspired by the positive impact of fine-grained attributes in ReID, we
   244	utilize the trainable attribute prompt structure defined above. Unlike
   245	holistic prompts used in some prior work [10], our attribute prompts
   246	feature learnable tokens dispersed throughout the text, each intended
   247	to capture a specific fine-grained feature. This design is critical for
   248	recognizing subtle differences, especially under occlusion, setting
   249	the stage for effective pseudo-label generation as described next.
   250	In our framework, attribute pseudo-labels refer to learnable token
   251	representations automatically derived from CLIP and CoOp, requiring no manual annotation. These pseudo-labels capture fine-grained
   252	visible attribute features (e.g., hair style, clothing color, accessories)
   253	and serve as auxiliary supervision signals to guide the image encoder towards learning more discriminative attribute-aware features.
   254	Unlike traditional labels that require human annotation, our pseudolabels are semantically trained to describe the collective attribute
   255	characteristics of all images belonging to the same identity.
   256	Our selection of attributes is informed by prior work demonstrating their effectiveness in person ReID. Specifically, we adopt key attribute categories identified as discriminative in ATPM [26], encompassing aspects like gender, hairstyle, posture, and clothing characteristics. Furthermore, the designed prompt template module offers
   257	inherent flexibility; it facilitates the straightforward incorporation of
   258	alternative attribute sets or descriptive contexts, such as explicitly
   259	
   260	Figure 2. Overview of the AG-ReID framework’s two-stage training. Stage 1 (Cross-modal Alignment): Using frozen Image/Text Encoders, learnable
   261	tokens within attribute prompts are trained via contrastive loss (Lfeat ) to generate attribute pseudo-labels. Simultaneously, a trainable Attribute Encoder learns
   262	to predict attribute tokens aligned with these pseudo-labels via LattrA . Stage 2 (Dual Guidance): The Image Encoder is trained with guidance from both
   263	attribute-prompt text features (LCE ) and the generated attribute pseudo-labels (LattrG ), leveraging a fine-tuned Attribute Encoder. Standard ReID losses
   264	(Lid , Ltri ) are also applied to the image features. The legend indicates module status (frozen/trainable) and token types.
   265	
   266	mentioning potential occlusion, through configuration adjustments
   267	without requiring modifications to the underlying code structure.
   268	Given a ReID dataset batch of n images M = {m1 , m2 , ..., mn },
   269	representing K unique identities, let P = {p1 , p2 , ..., pK } be the
   270	corresponding attribute prompts for each identity. For an image mi
   271	associated with identity k, its corresponding attribute prompt pk contains r learnable tokens, as defined previously. We denote this set of
   272	learnable tokens as Vi = {[v]k1 , [v]k2 , ..., [v]kr }.
   273	In the alignment stage of training, the text features fT of attribute
   274	prompt are aligned with the image features fM by SupConLoss [8],
   275	which is improved on top of the cross-entropy loss function for the
   276	supervised contrastive learning task.
   277	Lfeat = SupCon(fM , fT ) + SupCon(fT , fM )
   278	
   279	(6)
   280	
   281	In this way, not only the text feature fT gets the semantics corresponding to the image, but also each learnable token [v] gets the
   282	semantics of its corresponding attribute. At this time, we will call the
   283	learnable attribute token set V as semantics attribute pseudo-label.
   284	To use attribute pseudo-labels for guiding image features during
   285	the following stage, we propose the attribute encoder EA with 4
   286	i
   287	self-attention layers to align image feature fM
   288	with attribute pseudolabels Vi , which predicts attributes set Ai by image features.
   289	i
   290	Ai = [ai1 , ai2 , ..., air ] = EA (fM
   291	)
   292	
   293	The attribute align loss LattrA in this stage is formulated as,
   294	
   295	(7)
   296	
   297	n
   298	
   299	LattrA = −
   300	
   301	r
   302	
   303	1 XX
   304	cos(aij , vji )
   305	r i=1 j=1
   306	
   307	(8)
   308	
   309	The loss function in the cross-modal alignment stage Lalign consists of Lfeat and LattrA , and λ is a hyper-parameter for adjusting loss
   310	weights.
   311	Lalign = Lfeat + λLattrA
   312	
   313	(9)
   314	
   315	3.3 Dual Guidance Stage
   316	Building on the cross-modal alignment stage, we obtain text features
   317	encapsulating the holistic semantic information of the text prompt,
   318	the attribute pseudo-label denoting fine-grained attribute information, and an attribute encoder designed to align image features with
   319	their respective attribute features. At this stage, the integration of
   320	both holistic and fine-grained information guides the feature extraction process for images, which is why it is referred to as the dual
   321	guidance stage.
   322	The former computes the loss using the cross entropy loss function
   323	LCE , which is frequently employed in contrastive learning, whereas
   324	the latter is optimized by the improved previously mentioned attributes alignment loss LattrG .
   325	LCE = CrossEntropy(fM , fT )
   326	
   327	(10)
   328	
   329	Additionally, the ID loss Lid and triplet loss Ltri , commonly employed in ReID tasks, are also combined.
   330	
   331	Lid = −
   332	
   333	n
   334	X
   335	
   336	qi log(pi )
   337	
   338	Dataset
   339	Occ-Duke
   340	Occ-ReID
   341	P-Duke
   342	MSMT17
   343	Market-1501
   344	DukeMTMC
   345	
   346	(11)
   347	
   348	i=1
   349	
   350	Ltri = max(dp − dn + α, 0)
   351	
   352	(12)
   353	
   354	where qi denotes the value in the target distribution, and pi represents ID prediction logits of class i, dp and dn are feature distances
   355	of positive pair and negative pair, while α is the margin of triple loss.
   356	The idea of fine-tuning image feature encoder directly using
   357	pseudo-labels is expected to yield good results in holistic datasets,
   358	and this is confirmed in subsequent experiments.
   359	However, it is crucial to recognize that images with the same ID
   360	do not always share consistent semantics. In other words, attribute
   361	pseudo-labels reflect the collective attribute semantics of all images
   362	with a common ID, which may cause some tokens a predicted by the
   363	attribute encoder to mismatch their pseudo-labels v in the occluded
   364	case. Such mismatched pairs are referred to as “noise”. This issue
   365	arises when parts of some images differ from the majority within the
   366	same ID group. For instance, if the lower half of a person is visible
   367	in most images with ID i, but obscured in others, the corresponding
   368	pseudo-label v will predominantly contain the semantic information
   369	of the visible parts. This leads to the token a at the corresponding
   370	positions in the occluded image having semantic discrepancies, potentially misleading the feature extraction process.
   371	To address this issue, we suggest a slight modification to the attribute alignment loss for occluded ReID datasets, termed the noisemask method. In this approach, only pairs with a similarity above the
   372	threshold γ will be considered for loss calculation, while the rest will
   373	be masked. For holistic datasets, γ is assigned a value of −1, which
   374	means that loss will be computed for every pair.
   375	(
   376	1 if cos(aij , tij ) > γ
   377	D(i, j) =
   378	(13)
   379	0 otherwise.
   380	Thus, the attribute pseudo-label guidance loss LattrG is calculated
   381	as follows,
   382	n
   383	
   384	LattrG = −
   385	
   386	r
   387	
   388	1 XX
   389	D(i, j) · cos(aij , vji )
   390	r i=1 j=1
   391	
   392	(14)
   393	
   394	The final dual guidance stage loss is formulated as,
   395	Lguide = Lid + Ltri + LCE + βLattrG
   396	
   397	Experiments
   398	
   399	4.1 Experimental Settings
   400	4.1.1
   401	
   402	4.1.2
   403	
   404	ID
   405	1,404
   406	200
   407	1299
   408	4,101
   409	1,501
   410	1,404
   411	
   412	Camera
   413	8
   414	6
   415	2
   416	15
   417	6
   418	8
   419	
   420	Datasets Statistics.
   421	
   422	Implementation Details
   423	
   424	Our method is implemented on a server equipped with a NVIDIA
   425	GeForce RTX 3090 Ti GPU. We utilize the pre-trained CLIP with
   426	ViT-B/16 as the backbone, resizing all images to 256 × 128. During the training phase, data augmentation techniques such as random
   427	horizontal flipping, cropping, and erasing [30] are used. In the crossmodal alignment stage, the Adam optimizer is used with a base learning rate of 3.5e-04 and a warm-up learning rate that increases linearly
   428	from 1e-06. In the dual guidance stage, the base learning rate is set
   429	at 5e-06. The batch size is set to 32, with K to 4 instances sampled
   430	for each identity per batch. In addition, side information embeddings
   431	(SIE) and overlap patches (OLP) are utilized to further improve the
   432	model performance [6]. The margin for triplet loss α is set to 0.3,
   433	and the weight λ and β in the loss function are set to 1 and 0.01,
   434	respectively.
   435	During training, the attribute encoder introduces approximately
   436	2.76M additional parameters, which represents only about 3.2% of
   437	the ViT-B/16 backbone (86M parameters). Crucially, during inference, AG-ReID only requires the image encoder (ViT-B/16), resulting in identical computational complexity and model size as CLIPReID and other ViT-based methods. The attribute encoder is exclusively used during training and does not affect deployment efficiency.
   438	The selection of specific attributes in the prompt-attribute templates refers to the table of attributes available in ATPM [26]. For
   439	threshold γ in the dual guidance stage, we choose the Otsu method
   440	[12] for occluded datasets to automatically calculate the threshold to
   441	maximize the variance between classes.
   442	
   443	4.2 Performance Comparison
   444	(15)
   445	
   446	which consists of three components: the basic ReID losses Lid and
   447	Ltri , the text prompt guidance loss LCE , and the attribute pseudolabel guidance loss LattrG . These components work together to guide
   448	the effective extraction of image features, where β is the weight of
   449	LattrG .
   450	
   451	4
   452	
   453	Table 1.
   454	
   455	Image
   456	35,489
   457	2000
   458	24143
   459	126,441
   460	32,668
   461	36,411
   462	
   463	Datasets and Evaluation Protocols
   464	
   465	We evaluated the proposed AG-ReID method on both occluded, partial, and holistic ReID benchmarks, including Occluded-Duke [13].
   466	Occluded-ReID [32], P-DukeMTMC [32], MSMT17 [23], Market1501 [29] and DukeMTMC-reID [15]. The details of the datasets are
   467	summarized in Tab. 1.
   468	Following common practices, we employ the cumulative matching characteristics (CMC) [21] at Rank-1 (R1) and the mean average
   469	precision (mAP) [29] for performance evaluation.
   470	
   471	4.2.1
   472	
   473	Performance on Occluded datasets
   474	
   475	Table 2 presents the comparison between AG-ReID and state-of-theart methods, including recent approaches like KRP [18] and ADP
   476	[25], on three widely-used occluded/partial person ReID datasets:
   477	Occluded-Duke, Occluded-ReID, and P-Duke. Compared to the
   478	baseline CLIP-ReID on the Occluded-Duke dataset, our method
   479	achieves a 3.2% improvement in mAP and a 2.6% increase in R@1.
   480	Besides, AG-ReID achieves state-of-the-art mAP and R@1 performance on both Occluded-ReID and P-Duke datasets. This highlights
   481	the effectiveness of fully utilizing the fine-grained semantic information within the pre-trained model without requiring extra annotation data. Furthermore, applying standard k-reciprocal re-ranking
   482	significantly boosts the performance on these challenging datasets
   483	(e.g., reaching 75.4% mAP and 74.1 R@1 on Occ-Duke). Detailed
   484	re-ranking results are provided in supplementary materials.
   485	We primarily report results using ViT-B/16 as it generally outperforms ResNet-50 in recent ReID works. For completeness, we also
   486	evaluated AG-ReID with ResNet-50 backbone on Occluded-ReID,
   487	achieving 55.1% mAP and 63.7% Rank-1, compared to 53.5% mAP
   488	
   489	Method
   490	
   491	Reference
   492	
   493	HOReID
   494	PAT
   495	FED
   496	PFD
   497	BPBReIDRI
   498	KRPIN
   499	DPM-SPT
   500	ADP
   501	CLIP-ReID
   502	AG-ReID
   503	
   504	CVPR2020
   505	CVPR2021
   506	CVPR2022
   507	AAAI2022
   508	WACV2023
   509	ECCV2024
   510	AAAI2024
   511	AAAI2024
   512	AAAI2023
   513	-
   514	
   515	Category
   516	
   517	Extra data
   518	
   519	Pretrain model
   520	Baseline
   521	Our
   522	
   523	Detail
   524	Pose Estimator
   525	Part Prototype
   526	Manually Crop
   527	Pose Estimator
   528	Human parsing labels
   529	Manual keypoint labels
   530	Pre-trained ViT
   531	Pre-trained ViT
   532	Pre-trained CLIP
   533	Pre-trained CLIP
   534	
   535	Occ-Duke
   536	mAP
   537	R@1
   538	43.8
   539	55.1
   540	53.6
   541	64.5
   542	56.4
   543	68.1
   544	61.8
   545	69.5
   546	57.5
   547	71.3
   548	67.1
   549	79.8
   550	63.0
   551	74.7
   552	63.8
   553	74.5
   554	60.0
   555	67.8
   556	63.2
   557	70.4
   558	
   559	Occ-ReID
   560	mAP
   561	R@1
   562	70.2
   563	80.3
   564	72.1
   565	81.6
   566	79.3
   567	86.3
   568	83.0
   569	81.5
   570	70.9
   571	77.0
   572	79.1
   573	85.4
   574	81.1
   575	87.8
   576	82.0
   577	88.2
   578	87.6
   579	90.1
   580	
   581	P-Duke
   582	mAP
   583	R@1
   584	88.0
   585	80.5
   586	83.1
   587	79.2
   588	91.3
   589	86.0
   590	84.5
   591	91.8
   592	
   593	Table 2. Performance on occluded datasets.
   594	
   595	and 61.0% Rank-1 for CLIP-ReID (RN50), demonstrating the effectiveness of our approach across different architectures.
   596	Case studies in Figure 1 further reveal that our method effectively addresses errors caused by overlooking fine-grained differences, aligning with our initial design expectations.
   597	
   598	4.2.2
   599	
   600	PAT
   601	FED
   602	BPBReIDRI
   603	DPM-SPT
   604	ADP
   605	PFD
   606	HOReID
   607	CC-ViT
   608	SOLIDER
   609	TransReID
   610	DiP
   611	CLIP-ReID
   612	AG-ReID
   613	
   614	MSMT17
   615	mAP
   616	R@1
   617	50.4
   618	74.4
   619	64.6
   620	83.7
   621	67.4
   622	85.9
   623	67.4
   624	85.3
   625	71.8
   626	87.3
   627	75.8
   628	89.7
   629	76.7
   630	90.1
   631	
   632	Table 3.
   633	
   634	Market-1501
   635	mAP
   636	R@1
   637	88.0
   638	95.4
   639	86.3
   640	95.0
   641	88.4
   642	95.7
   643	89.4
   644	95.5
   645	89.5
   646	95.6
   647	89.7
   648	95.5
   649	84.9
   650	94.2
   651	90.4
   652	96.0
   653	91.6
   654	96.1
   655	88.9
   656	95.3
   657	90.3
   658	95.8
   659	90.5
   660	95.4
   661	90.8
   662	95.8
   663	
   664	DukeMTMC
   665	mAP
   666	R@1
   667	78.2
   668	88.8
   669	78.0
   670	89.4
   671	81.3
   672	91.7
   673	82.4
   674	91.1
   675	83.1
   676	91.2
   677	83.2
   678	91.2
   679	75.6
   680	86.9
   681	81.2
   682	90.4
   683	82.0
   684	90.7
   685	85.2
   686	91.7
   687	83.1
   688	90.8
   689	83.3
   690	91.0
   691	
   692	Performance on holistic datasets.
   693	
   694	Table 3 presents the results of AG-ReID compared to other
   695	methods on three widely used holistic person ReID datasets. Our
   696	model shows improvement over the CLIP-ReID baseline on all three
   697	datasets. Moreover, our approach attains state-of-the-art performance
   698	on the large-scale MSMT17 dataset, surpassing the compared methods. It is also observed that the CLIP-ReID baseline, while performing well on MSMT17, was less competitive on Market-1501 and
   699	DukeMTMC. This strong performance on MSMT17 likely benefits from the dataset’s large scale and diversity, which may better
   700	align with the data distribution encountered during the model’s extensive pre-training. When combined with k-reciprocal re-ranking,
   701	AG-ReID’s performance is further enhanced on holistic datasets as
   702	well (e.g., achieving 86.6% mAP and 91.1% R@1 on MSMT17),
   703	demonstrating its compatibility with standard post-processing techniques.
   704	
   705	4.3 Ablation Studies and Analysis
   706	4.3.1
   707	
   708	AP
   709	
   710	✓
   711	✓
   712	
   713	mAP
   714	60.0
   715	62.4
   716	62.2
   717	63.2
   718	
   719	Result for Occ-Duke
   720	R@1
   721	R@5
   722	R@10
   723	67.8
   724	80.1
   725	85.2
   726	69.7
   727	81.7
   728	86.2
   729	69.2
   730	81.9
   731	86.3
   732	70.4
   733	82.4
   734	86.4
   735	
   736	Table 4. Ablation studies for AG-ReID, where “AT” is attribute-prompt
   737	template guidance and “AP” is attribute pseudo-label guidance. The baseline
   738	is CLIP-ReID model with SIE and OLP.
   739	
   740	Performance on Holistic datasets
   741	Method
   742	
   743	Strategies
   744	Baseline
   745	AT
   746	✓
   747	✓
   748	✓
   749	✓
   750	✓
   751	✓
   752	
   753	Ablation study for dual guidance
   754	
   755	To further confirm the significance of the attribute-prompt template
   756	(AT) and attribute pseudo-label guidance (AP) in our AG-ReID
   757	
   758	framework, we conduct ablation studies using the CLIP-ReID model
   759	with SIE and OLP as the baseline. As shown in Table 4, introducing either the attribute prompt template or the attribute pseudolabel guidance individually yields performance improvements over
   760	the baseline. Combining both components (AG-ReID) further enhances the guiding effect of fine-grained attributes on image features,
   761	achieving the best results.
   762	
   763	4.3.2
   764	
   765	Attribute Prompt Template
   766	Template
   767	Default Prompt
   768	Longer Prompt
   769	Random Prompt
   770	Attribute Prompt a
   771	Attribute Prompt b
   772	Attribute Prompt c
   773	Attribute Prompt d
   774	
   775	mAP
   776	62.2
   777	62.0
   778	62.1
   779	62.6
   780	63.0
   781	62.8
   782	63.2
   783	
   784	Occ-Duke
   785	R@1
   786	R@5
   787	69.2
   788	81.9
   789	69.9
   790	82.1
   791	69.5
   792	81.6
   793	69.8
   794	82.4
   795	70.5
   796	82.0
   797	69.9
   798	82.6
   799	70.4
   800	82.4
   801	
   802	R@10
   803	86.3
   804	86.2
   805	86.1
   806	86.2
   807	86.5
   808	86.1
   809	86.4
   810	
   811	Table 5. Result comparison for different prompt template. The default
   812	prompt is the text prompt proposed by CLIP-ReID, the longer prompt
   813	increases the number of trainable tokens on the basis of default, while the
   814	random prompt makes the number of fixed tokens become 0, making all
   815	tokens learnable, and the attribute prompt a, b, c, d are our trainable attribute
   816	prompt templates with different attributes selection.
   817	
   818	We investigate the impact of different prompt template designs
   819	in Table 5. While our pseudo-label guidance mechanism can operate with various prompt types, we hypothesized that attributespecific templates would better enrich the fine-grained semantics
   820	of the learned pseudo-labels compared to holistic prompts. The results confirm this hypothesis: templates explicitly structured around
   821	attributes (Attribute Prompt a, b, c, d) consistently outperform the
   822	default holistic prompt, longer holistic prompts, and random token
   823	prompts. This indicates that the core concept of attribute-structured
   824	
   825	successful mining and utilization of fine-grained attribute semantics inherent within the pre-trained vision-language model. The dualguidance mechanism ensures that both holistic context and detailed
   826	attributes contribute to the final feature representation, while the
   827	noise-masking strategy provides robustness against inconsistent visual cues common in occlusion. This approach, which avoids the
   828	need for external annotations or specialized modules like pose estimators, offers a promising direction for developing more practical
   829	and data-efficient ReID systems, potentially extendable to other finegrained recognition tasks facing similar challenges of partial information or subtle distinctions.
   830	
   831	5
   832	Figure 3.
   833	
   834	Heat map for visualization. (a) Source images, (b) Baseline, (d)
   835	AG-ReID.
   836	
   837	prompting is beneficial, and suggests the method is relatively robust
   838	to specific template wording, reducing the need for extensive prompt
   839	engineering. The detailed content of these templates can be found in
   840	the supplementary materials.
   841	
   842	4.3.3
   843	
   844	Dual guidance threshold
   845	
   846	Threshold
   847	P50
   848	P75
   849	P90
   850	Otsu
   851	Table 6.
   852	
   853	mAP
   854	61.9
   855	62.5
   856	62.8
   857	63.2
   858	
   859	Occ-Duke
   860	R@1
   861	R@5
   862	69.9
   863	82.7
   864	70.6
   865	82.5
   866	70.5
   867	82.3
   868	70.4
   869	82.4
   870	
   871	R@10
   872	86.0
   873	86.3
   874	86.0
   875	86.4
   876	
   877	Result comparison for different threshold γ.
   878	
   879	In the dual guidance stage, to address the issue where samples
   880	within the same ID may exhibit inconsistent features in occluded
   881	datasets, we introduce the noise-mask method with an attribute
   882	loss threshold γ. Loss is calculated only for predicted attribute and
   883	pseudo-label pairs whose similarity exceeds this threshold. As shown
   884	in Table 6, we compare different strategies for setting γ: fixed percentiles (P50, P75, P90) and the Otsu method [12], which automatically determines a threshold to maximize inter-class variance.
   885	Higher γ values impose stricter constraints, ensuring only wellaligned attribute features contribute to the loss, mitigating noise
   886	from inconsistent image details. Experimental results demonstrate
   887	that the Otsu method effectively determines an optimal threshold autonomously, eliminating the need for manual tuning per dataset and
   888	achieving the best performance.
   889	
   890	4.3.4
   891	
   892	Visualization
   893	
   894	We employed the Grad-CAM method [16] to visualize the key regions of interest for the model, shown in Figure 3. Compared to the
   895	baseline model, our AG-ReID model exhibits a heightened focus on
   896	fine-grained attributes such as hair, bags, and clothing.
   897	
   898	4.3.5
   899	
   900	Discussion
   901	
   902	Overall, our experiments validate the effectiveness of AG-ReID. The
   903	performance gains, particularly on occluded datasets, stem from the
   904	
   905	Conclusion
   906	
   907	In this paper, we introduce AG-ReID, a novel framework for occluded person re-identification that leverages the fine-grained semantics of pre-trained vision-language models. By employing attribute
   908	text prompts and attribute pseudo-labels, our method significantly
   909	enhances the descriptiveness of image features.
   910	Through extensive experiments on multiple datasets, AG-ReID
   911	demonstrates superior performance compared to existing methods,
   912	achieving state-of-the-art results on both occluded and holistic ReID
   913	tasks. Specifically, our method shows significant improvements in
   914	handling challenging cases involving occlusions and subtle attribute
   915	differences, while maintaining competitive performance on standard
   916	ReID scenarios. The experimental results validate the effectiveness
   917	of our approach in mining and utilizing fine-grained attribute semantics inherent within pre-trained vision-language models.
   918	This work highlights the potential of integrating fine-grained attribute information from pre-trained vision-language models to improve ReID accuracy without the need for additional data or complex manual annotations. Our approach offers several advantages: 1)
   919	It eliminates the dependency on external supervision signals such as
   920	pose estimation data or manually annotated attributes; 2) It effectively handles occluded scenarios by focusing on reliable attribute
   921	matches; 3) It maintains robust performance across different datasets
   922	and scenarios. Future work will explore the effectiveness of this approach in cross-dataset scenarios, as well as further optimization and
   923	application in real-world settings.
   924	
   925	References
   926	[1] X. Chen, X. Liu, W. Liu, X.-P. Zhang, Y. Zhang, and T. Mei. Explainable person re-identification with attribute-guided metric distillation. In
   927	Proceedings of the IEEE/CVF international conference on computer vision, pages 11813–11822, 2021.
   928	[2] C. Cui, S. Huang, W. Song, P. Ding, M. Zhang, and D. Wang.
   929	Profd: Prompt-guided feature disentangling for occluded person reidentification. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 1583–1592, 2024.
   930	[3] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1,950p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Base-Detail Feature Learning Framework for Visible-Infrared Person
     2	Re-Identification
     3	
     4	arXiv:2505.03286v1 [cs.CV] 6 May 2025
     5	
     6	Zhihao Gong1 , Lian Wu2 , Yong Xu3,∗
     7	1
     8	Harbin Institute of Technology (Shenzhen)
     9	2
    10	GuiZhou Education University
    11	3
    12	Harbin Institute of Technology (Shenzhen)
    13	gongzhh888@gmail.com, wulian best@163.com, laterfall@hit.edu.cn
    14	Abstract
    15	Visible-infrared person re-identification (VIReID)
    16	provides a solution for ReID tasks in 24-hour scenarios; however, significant challenges persist in
    17	achieving satisfactory performance due to the substantial discrepancies between visible (VIS) and infrared (IR) modalities. Existing methods inadequately leverage information from different modalities, primarily focusing on digging distinguishing
    18	features from modality-shared information while
    19	neglecting modality-specific details. To fully utilize differentiated minutiae, we propose a BaseDetail Feature Learning Framework (BDLF) that
    20	enhances the learning of both base and detail
    21	knowledge, thereby capitalizing on both modalityshared and modality-specific information. Specifically, the proposed BDLF mines detail and base
    22	features through a lossless detail feature extraction module and a complementary base embedding
    23	generation mechanism, respectively, supported by
    24	a novel correlation restriction method that ensures
    25	the features gained by BDLF enrich both detail and
    26	base knowledge across VIS and IR features. Comprehensive experiments conducted on the SYSUMM01, RegDB, and LLCM datasets validate the
    27	effectiveness of BDLF.
    28	
    29	1
    30	
    31	Introduction
    32	
    33	Person re-identification (ReID) aims to retrieve a target identity from gallery images captured by different cameras [Liu
    34	et al., 2022] and has recently demonstrated significant advancements in the fields of security and public surveillance
    35	[Ye et al., 2022a]. However, most existing methods [Cao
    36	et al., 2023][Wang et al., 2022][Yan et al., 2021] primarily focus on utilizing RGB images captured by visible (VIS)
    37	cameras during the daytime, which are inadequate for accommodating 24-hour scenarios that involve infrared (IR)
    38	images captured by IR cameras. To address the substantial
    39	cross-modality gap and facilitate operation in all-day scenarios, visible-infrared person re-identification (VIReID) methods [Chen et al., 2022][Park et al., 2021] have been developed, enabling the matching of IR (RGB) images given an
    40	interest in a specific RGB (IR) pedestrian image.
    41	
    42	(a) Features Learning with align- (b) Features Learning with crossing cross-modalities knowledge modalities knowledge compensation
    43	
    44	(c) Feature Learning with the proposed BDLF
    45	Figure 1: Motivation of the proposed BDLF, which focuses on sufficiently mining the modality-shared and modality-specific knowledge simultaneously and are not applicable for additional auxiliary
    46	data.
    47	
    48	The existing research on VIReID can generally be categorized into two principal methods: extracting distinguishing
    49	modality-shared features from VIS and IR modalities[Park
    50	et al., 2021][Zhang and Wang, 2023] and compensating for
    51	modality-specific or modality-shared features [Zhang et al.,
    52	2022a]. As shown in Figure 1(a), the former method aims to
    53	reduce cross-modality discrepancies by aligning comprehensive cross-modality features into a common semantic space.
    54	However, it neglects to leverage modality-specific and shared
    55	cues, which inevitably leads to performance bottlenecks. The
    56	latter approach, depicted in Figure 1(b) can be further divided into embedding-level and image-level methods. These
    57	methods generate compensatory knowledge in the embedding space and at the pixel level respectively, using auxiliary models(e.g., GANs[Goodfellow et al., 2014], segmentation networks, part alignment networks, etc.). However, these
    58	methods typically introduce losses and noise into the generated features or require additional data processing by other
    59	models, making them less effective and convenient. Consequently, advancing the development of VIReID to a more
    60	
    61	comprehensive level remains a significant challenge.
    62	Inspired by the analyses presented above, it is essential to
    63	recognize that modality-shared information, such as the contour and movement characteristics of pedestrians, can be considered base features. In contrast, modality-specific information, including the color and texture details of the RGB
    64	modality and the thermal characteristics of the IR modality,
    65	can be regarded as detail features. Both types of them should
    66	be integrated and utilized effectively together. Therefore, in
    67	this paper, we propose a novel Base-Detail Feature Learning
    68	Framework (BDLF), as shown in Figure 1(c). This framework is designed to extract modality-shared base features
    69	and modality-specific detail features from the original images
    70	with minimal additional computational costs, while jointly
    71	optimizing modality-shared, modality-specific, and comprehensive features.
    72	The proposed BDLF comprises a modality-specific detail
    73	feature extraction (DFE) module and a modality-shared base
    74	embedding generation (BEG) block, which ultimately combine the optimized features collected. Inspired by [Zhao
    75	et al., 2023], we designed the DFE module to mine the
    76	modality-specific detail information losslessly. Subsequently,
    77	the BEG block derives modality-shared base features. To
    78	fully capture both specific and shared information, we proposed a novel specific-shared knowledge distillation(SKD)
    79	loss. It encourages the detail (base) features to effectively
    80	incorporate modality-specific (modality-shared) knowledge
    81	by imposing a constraint on the correlation that the crossmodality detail and base features should exhibit. Specifically,
    82	it ensures that the correlations across RGB and IR modalities are indistinct and notable, respectively. Perspectives in
    83	[Feng et al., 2023] explain that the independent decomposition of features can maximize the mutual information of subfeatures; therefore, we introduced an independence constraint
    84	in the semantic space between the derived detail and base
    85	features. This indicates that the base feature exclusively encompasses modality-shared knowledge, while the detail feature contains modality-specific information. In summary, the
    86	main contributions of our work are as follows:
    87	• A novel correlation optimization method is proposed
    88	that effectively generates both modality-shared and
    89	modality-specific features using a non-parametric approach, rather than relying on classifiers.
    90	• We propose an end-to-end Base-Detail Feature Learning Framework (BDLF) for VIReID that integrates extracts of modality-shared base knowledge and modalityspecific detail knowledge.
    91	• Extensive experiments have demonstrated that the proposed BDLF outperforms other state-of-the-art methods
    92	for the VIReID task on the SYSU-MM01, RegDB, and
    93	LLCM datasets.
    94	
    95	2
    96	
    97	Related Work
    98	
    99	The main idea for solution VI-ReID task is decreasing the
   100	notable discrepence across VIS and IR modalities, thereby
   101	the existing methods consist of aligning the cross-modality
   102	features and utilizing the auxiliary data or features generated
   103	by other models.
   104	
   105	The alignment of feature representation methods seeks to
   106	convert cross-modality features into a unified semantic space
   107	through either metric learning techniques [Liu et al., 2022]
   108	[Park et al., 2021] [Luo et al., 2019] or by enhancing networks with more effective feature extraction components
   109	[Zhang and Wang, 2023] [Sarker and Zhao, 2024]. However, these approaches ultimately encounter performance bottlenecks due to the loss of modality-specific information.
   110	The methods for utilizing auxiliary information produced
   111	by other models are proposed to enhance identifiable knowledge. GAN-based methods [Zhang et al., 2022a]d[Wang et
   112	al., 2020] generate compensatory features at either the image level or the embedding level to simulate features from
   113	another modality. XIV [Li et al., 2020] introduces the Xmodality generated by a lightweight auxiliary network to decrease discrepancies between the two modalities. LUPI [Alehdaghi et al., 2022] establishes an intermediate domain between VIS and IR modalities. Furthermore, it generates images that belong to this intermediate domain to guide the
   114	network in acquiring more discernible information. SGIEL
   115	[Feng et al., 2023] innovatively adopts the shape knowledge
   116	of identity generated by segmentation models to enrich supplementary information. TMD [Lu et al., 2024] generates
   117	style-aligned images to minimize differences at the image
   118	level, subsequently aligning cross-modality features to eliminate discrepancies in feature distribution and instance features. However, this remains a challenging field of research
   119	because these methods either inevitably introduce information distortion during the generation process or fail to completely capture modality-specific and modality-shared information.
   120	
   121	3
   122	
   123	Methodology
   124	
   125	3.1
   126	
   127	Overall Framework
   128	
   129	The pipeline of our proposed method, referred to as BDLF, is
   130	illustrated in Figure 2. This method utilizes a single-stream
   131	ResNet-50 network[He et al., 2016a] as its backbone. The intermediate features Z M ∈ RB×C×H×W , which pass through
   132	a portion of the backbone, are fed into the proposed detail
   133	feature extraction (DFE) module to yield detail features Z D .
   134	Additionally, the base feature Z B is generated by inputting
   135	the output Z ∈ RB×C from the backbone into the proposed
   136	base embedding generation (BEG) block. A novel specificshared knowledge distillation (SKD) loss is proposed to ensure that the generated detail(base) features contain as much
   137	modality-specific (modality-shared) knowledge as possible,
   138	thereby effectively leveraging modality-specific and shared
   139	information. Furthermore, we construct a modality-shared
   140	feature Z F using a cross-modality feature fusion method to
   141	optimally supplement the base features. During the inference phase, only the comprehensive feature Z yielded by the
   142	backbone is used for performance evaluation. This is because
   143	the proposed DFE and BEG modules effectively enhance the
   144	comprehensive feature by incorporating additional detail and
   145	base information.
   146	Given an identity image from either the visible or infrared modality, VIReID intends to identify the most similar sequence of that identity in another modality. Let
   147	
   148	cat
   149	
   150	Z M (1 : C)
   151	
   152	ZD
   153	
   154	Z̄ D
   155	
   156	GAP
   157	
   158	lorth
   159	Independence Restrict
   160	
   161	Z̄ B
   162	I −P
   163	Z
   164	
   165	ZFB
   166	
   167	Projection matrix
   168	
   169	Detail Feature
   170	
   171	Base Feature
   172	
   173	lokl
   174	lDF E
   175	
   176	RGB Feature
   177	
   178	lBEG
   179	lf bkl
   180	
   181	CLSB
   182	
   183	lskd
   184	ZFB
   185	Base Subspace
   186	
   187	Z̄FB
   188	
   189	Base Embedding Generation
   190	
   191	Comprehensive Feature
   192	
   193	ltri
   194	
   195	Z̄ B
   196	
   197	Batch Attention Channel Attention
   198	
   199	Middle Feature
   200	
   201	CLSD
   202	
   203	ZkM (c + 1 : C)
   204	
   205	P
   206	
   207	lid
   208	
   209	lapp
   210	
   211	ZD
   212	
   213	lcorr
   214	
   215	Z1M (c + 1 : C)
   216	
   217	Z̄ D
   218	
   219	Correlation Restrict
   220	
   221	Gap
   222	
   223	Cross Attention
   224	
   225	INN Block
   226	
   227	Detail Subspace
   228	
   229	ZkM (1 : c)
   230	
   231	Layer Normalization
   232	
   233	INN Block
   234	
   235	ZM
   236	
   237	Layer Normalization
   238	
   239	Detail Feature Extraction
   240	Z1M (1 : c)
   241	
   242	Conv Block 4
   243	
   244	Conv Block 1
   245	Conv Block 2
   246	Conv Block 3
   247	
   248	Backbone Network
   249	
   250	IR Feature
   251	
   252	Multiplication
   253	
   254	Addition
   255	
   256	Figure 2: The pipeline of the proposed Base-Detail Feature Learning Framework (BDLF), which consists of a Detail Feature Extraction
   257	(DFE) module and a Base Embedding Generation (BEG) block, and jointly optimizes the extracted detail, base, and comprehensive features.
   258	
   259	the training set {XV , XI } consist of B identities, with
   260	each identity including P samples. Therefore, XV =
   261	
   262	xV b,p , b = 1, ..., B; p = 1,
   263	 ..., P symbolizes the set of visible images, while XI = xI b,p , b = 1, ..., B; p = 1, ..., P
   264	denotes the set of infrared images. As illustrated in Figure 2,
   265	the VIS and IR images are processed through the backbone
   266	network, i.e,
   267	ZVM/I =E f ore (XV /I )
   268	ZV /I =E rear (ZVM/I )
   269	Z = cat(ZV , ZI )
   270	
   271	(1)
   272	
   273	where E f ore (·) and E rear (·) are the former and latter
   274	parts of the backbone network, the embeddings ZV /I M ∈
   275	B
   276	
   277	B
   278	
   279	R 2 ×C×H×W and ZV /I ∈ R 2 ×C denote the intermediate
   280	and complete outputs from the backbone for the VIS and IR
   281	modalities, cat(·) refers to the concatenation operation along
   282	the batch dimension.
   283	
   284	3.2
   285	
   286	Specific-shared Knowledge Distillation
   287	
   288	We observe that the similarity of base information, such as
   289	contours and movements, between the VIS and IR modalities is noticeable. In contrast, the similarity of detail information including color, texture, and thermal details between
   290	the two modalities is suppressed. Inspired by [Zhao et al.,
   291	2023], as shown in Figure 3, the base and detail features can
   292	be generated by increasing and reducing the correlation between the two modalities respectively. Based on this, we
   293	propose a novel specific-shared knowledge distillation (SKD)
   294	
   295	loss, which is numerically smoother and easier to optimize,
   296	formulated as follows:
   297	log[Corr(ZVB , ZIB )]
   298	lskd = q
   299	3
   300	log[Corr(ZVD , ZID )] + γ
   301	
   302	(2)
   303	
   304	in which ZVB/I denotes the base features generated by the
   305	proposed BEG block, and ZVD/I denotes the detail features
   306	extracted from the proposed DFE module. Corr(·) is the
   307	Pearson correlation coefficient operation, while γ represents a
   308	constant that ensures the denominator remains non-zero. According to optimize the SKD loss, the correlation between
   309	the VIS and IR modalities of both base and detail features(i.e,
   310	Corr(ZVB , ZIB ) and Corr(ZVD , ZID ) in formula (2)) is simultaneously increased and decreased. This approach allows the
   311	proposed DFE module to extract embeddings rich in detailed
   312	knowledge. Consequently, the proposed BEG block is capable of generating base embeddings that contain a greater
   313	amount of modality-shared knowledge.
   314	
   315	3.3
   316	
   317	Detail Feature Extraction
   318	
   319	The proposed DFE module aims to acquire detail features
   320	that imply modality-specific information from the intermediate embedding ZVM/I by utilizing a series of invertible neural network (INN) blocks[Zhao et al., 2023][Dinh et al.,
   321	2017][Zhou et al., 2022], which can effectively preserves detailed characteristics and mitigates information loss during
   322	feature extraction by making its input and output embeddings
   323	are mutually generated. Taking the VIS case as an exam-
   324	
   325	VIS Base Embeddings VIS Detail Embeddings VIS Detail Embeddings From BEG
   326	
   327	obtained by concatenating the VIS and IR detail embeddings
   328	along the batch dimension:
   329	
   330	D
   331	D
   332	Corr(ZV
   333	, Z̄V
   334	)
   335	
   336	B
   337	Corr(ZV
   338	, ZIB )
   339	
   340	D
   341	Corr(ZV
   342	, ZID )
   343	
   344	Z D = cat(ZVD , ZID )
   345	D
   346	Corr(Z̄V
   347	, Z̄ID )
   348	
   349	D
   350	D
   351	Corr(ZV
   352	, Z̄V
   353	)
   354	
   355	IR Base Embeddings
   356	
   357	IR Detail Embeddings
   358	
   359	Pull away
   360	
   361	IR Detail Embeddings From BEG
   362	Pull close
   363	
   364	Figure 3: Illustration of correlation instruct to learn modality specific and shared information.
   365	
   366	With the proposed SKD loss formulated in formula (2), the
   367	extracted detail feature Z D can significantly enrich modalityspecific detail knowledge. Thus a private classifier CLSD
   368	that is specially designed for the detail feature Z D is constructed, alongside a communal classifier CLSB that processes the base embeddings and the comprehensive feature
   369	Z obtained from formula (1), as illustrated in Figure 2. Furthermore, the commonly used idP
   370	loss[Luo et al., 2019] driven
   371	n
   372	by cross-entropy (ce(p, q) = − i=1 qi log(pi )) was applied
   373	to strengthen the distinguishable information of detail feature
   374	Z D , i.e,
   375	D
   376	lid
   377	= E(zD ∼Z D ) ce(CLSD (z D ), Y )
   378	
   379	M
   380	
   381	ple, we obtain the input for the DFE module, ZV (1 : c) and
   382	C
   383	B
   384	M
   385	ZV (c + 1 : C) ∈ R 2 × 2 ×H×W by splitting ZV M in half
   386	along the channel dimension. The transformations in each
   387	block can be denoted as follows:
   388	M
   389	M
   390	M
   391	ZV,k+1
   392	(c + 1 : C) = ZV,k
   393	(c + 1 : C) + F1 [ZV,k
   394	(1 : c)]
   395	M
   396	M
   397	ZV,k+1
   398	(1 : c) = F2 [ZV,k+1
   399	(c + 1 : C)]
   400	M
   401	M
   402	+ ZV,k
   403	(1 : c) • exp{F3 [ZV,k+1
   404	(c + 1 : C)]}
   405	
   406	Here,
   407	is the input of the kth (k ∈ 1, ..., K) block,
   408	Fi (·)(i ∈ 1, 2, 3) denotes the convolution blocks. The symbol
   409	• indicates element-wise multiplication of matrices, LN (·)
   410	represents layer normalization in the Lite Transformer[Wu et
   411	al., 2020] and cat(·) is the channel concatenation operation.
   412	The IR situation can be easily derived by substituting I for
   413	the subscript V in the aforementioned formulas.
   414	At the final stage of DFE, we consider the detail features of both modalities integrally, as concatenating the extracted detail embeddings from the two modalities can help
   415	reduce computational complexity. Therefore, we feed the extracted detail embeddings into a cross-attention-based transformer to facilitate cross-modality reasoning and information
   416	exchange. This process enables the detail feature integration of knowledge from various modalities and allows for a
   417	more effective focus on distinguishable information, thereby
   418	enhancing the robustness and efficacy of semantic representation. Inspired by[Li et al., 2022], the transformer can be
   419	denoted as follows:
   420	ZVP/I = GAP (ZVM/I,K )
   421	ZVD = LN {sof tmax[(ZVP Wq )(ZIP Wk )T ](ZIP Wv ) + ZVP }
   422	ZID = LN {sof tmax[(ZIP Wq )(ZVP Wk )T ](ZVP Wv ) + ZIP }
   423	(4)
   424	where GAP (·) is the global average pooling operation,
   425	C
   426	B
   427	ZVP/I ∈ R 2 × 2 denotes the embeddings after pooling. Wq ,
   428	Wk , and Wv are the learnable parameters for DFE, LN (·)
   429	refers to layer normalization, and sof tmax(·) indicates the
   430	calculation of the softmax by row. Ultimately, the detail feaC
   431	ture Z D ∈ RB× 2 produced by the proposed DFE module is
   432	
   433	(6)
   434	
   435	Since there are differences in the distribution of classification results between the detail feature Z D and the comprehensive feature Z, this misalignment may impede our goal of
   436	enhancing the representation ability of Z leveraging detailed
   437	knowledge. Therefore, we constrain the probability distribution predicted from Z D to align with the distribution from
   438	Z, ensuring that their semantic representations are consistent.
   439	This process can be expressed as follows:
   440	
   441	M
   442	M
   443	M
   444	ZV,k+1
   445	= LN {cat[ZV,k+1
   446	(1 : c), ZV,k+1
   447	(c + 1 : C)]} (3)
   448	M
   449	ZV,k
   450	
   451	(5)
   452	
   453	lodkl = E(z,zD ∼Z,Z D ) ce(CLSD (z D ), CLSB (z))
   454	
   455	(7)
   456	
   457	The total loss of the proposed DFE module can be obtained
   458	by combining formulas (6) and (7):
   459	D
   460	lDF E = lid
   461	+ lodkl
   462	
   463	3.4
   464	
   465	(8)
   466	
   467	Base Embedding Generation
   468	
   469	The proposed BEG block is designed to produce the base
   470	embeddings from Z utilizing the acquired detail feature
   471	Z D . Take notice that there are significant semantic differences between modality-specific detail information such
   472	as color and texture and modality-shared base information,
   473	which includes movements, contours, and so on. For this
   474	reason, inspired by [Feng et al., 2023], we have developed a method to ensure that the detail(base) features can
   475	only contain modality-specific(modality-shared) distinguishable knowledge, thereby maximizing the collection of both
   476	modality-specific and modality-shared information. Furthermore, the proposed DFE and BEG blocks can learn these two
   477	categories of knowledge simultaneously without interfering
   478	with each other. Based on this premise, we consider the detail and base embeddings to be independent of each other,
   479	i.e, Z D ⊥ Z B . According to the approach of making Z̄ D
   480	comprehensively converge to Z D and impose the independence restriction between the detail and base embedding, the
   481	proposed BEG block can then generate modality-shared base
   482	embedding by excluding detailed knowledge from Z in the
   483	semantic space,i.e,
   484	
   485	Z × P = Z̄ D
   486	, Z̄ D → Z D
   487	(9)
   488	Z × (I − P ) = Z B
   489	in which Z is the output of backbone network, I is the identity
   490	matrix, → denotes approximating, Z̄ D and Z B ∈ RB×C are
   491	
   492	the gained detail and base embeddings by using a projection
   493	matrix P ∈ RC×C to decompose Z into mutually orthogonal
   494	subspaces. By the properties of orthogonal projection matrix,
   495	P should be a conjugate symmetric idempotent matrix and
   496	must satisfy the following constraints in the real number case:
   497	2
   498	
   499	P = P, P
   500	
   501	T
   502	
   503	=P
   504	
   505	(10)
   506	
   507	The process of approaching can be divided into three components: approximating in the feature space, semantic representation, and the correlation between Z D and Z̄ D . In the
   508	case of approximating on feature space, we first calculate the
   509	distances between all embeddings in a mini-batch for Z̄ D and
   510	Z D respectively, and obtain the difference map M by:
   511	M = ||sof tmax[Z̄ D (Z̄ D )T − Z D (Z D )T ]||2
   512	
   513	(11)
   514	
   515	dimensions. We then enhance the similarity between ZVB and
   516	ZIB by aligning them with ZFB :
   517	lf bkl = E(zFB ,zB
   518	
   519	V /I
   520	
   521	B
   522	B
   523	B ,Z B ) ce(CLSB (z
   524	∼ZF
   525	V /I ), CLSB (zF ))
   526	V /I
   527	
   528	(17)
   529	This approach ensures that Z B contains only the knowledge
   530	shared between the modalities. In addition, we also utilize
   531	cross-modality semantic alignment for ZVB/I to strengthen the
   532	collection of modality-shared knowledge:
   533	lbkl = E(zB
   534	
   535	V /I
   536	
   537	B
   538	B
   539	B ) ce(CLSB (z ), CLSB (z ))
   540	∼ZV
   541	V
   542	I
   543	/I
   544	
   545	(18)
   546	
   547	The id loss for both was also employed to enhance the distinguishable information of Z B and ZFB , and the loss for crossmodality feature fusion method is:
   548	
   549	Then we enforce the distance distribution of Z̄ D to converge
   550	to that of Z D by optimizing the following loss:
   551	
   552	F
   553	lcmf = lid
   554	+ lf bkl
   555	
   556	lf kl = E(ai,j ∼M ) ai,j
   557	
   558	Consequently, the total loss for the BEG block can be summarized as follows:
   559	
   560	(12)
   561	
   562	Furthermore, we aligned the semantic representation between
   563	Z D and Z̄ D by adjusting the predicted probability distribution of Z̄ D closer to that of Z D . By drawing an analogy with
   564	formula (7), we have:
   565	ldkl = E(z̄D ,zD ∼Z̄ D ,Z D ) ce(CLSD (z̄ D ), CLSD (z D )) (13)
   566	Considering that the detail feature Z̄ D generated by the
   567	BEG block should exhibit the same correlation properties as
   568	Z D . As illustrated on the right side of the dashed line in Figure 3, we achieved consistency in correlation between Z̄ D
   569	and Z D by pulling close their cross-modalities correlations
   570	denoted as Corr(ZVp , ZIp ), p ∈ {D, /D̄} and by reducing the
   571	discrepancy in correlation within the same modality, repreD̄
   572	D
   573	), m ∈ {V, I}. This is accomplished
   574	, Zm
   575	sented as Corr(Zm
   576	by optimizing the follows loss:
   577	ldcorr =
   578	
   579	(Corr(Z̄VD , Z̄ID ) − Corr(ZVD , ZID ))2
   580	Corr(Z̄VD , ZVD )2 + Corr(Z̄ID , ZID )2 + γ
   581	
   582	(14)
   583	
   584	B
   585	lBEG = lid
   586	+ lapp + lbkl + lcmf + lorth
   587	
   588	3.5
   589	
   590	Optimization
   591	
   592	In the preceding section, the proposed DFE module extracted
   593	detailed knowledge from the intermediate feature Z M and
   594	subsequently produced the detail feature Z D , the proposed
   595	BEG block produced the base feature by eliminating the detailed knowledge from the comprehensive feature Z, the proposed SKD loss ensures that both the detail and base features
   596	effectively capture modality-specific and shared information.
   597	We also incorporated the commonly used id and triplet loss
   598	[Hermans et al., 2017] ltri for Z into our method. Similar to
   599	(18), we enforce cross-modality consistency for Z by:
   600	lokl = E(zV /I ∼ZV /I ) ce(CLSB (zV ), CLSB (zI ))
   601	
   602	(21)
   603	
   604	Eventually, the total loss of BDLF is defined as:
   605	(15)
   606	
   607	After the description provided above, we generated the
   608	base feature Z B by eliminating the detail feature Z D from
   609	Z. Given that the base information across modalities, such as
   610	contours and movements, should exhibit significant similarities, we constructed a cross-modality feature fusion method
   611	that integrates the base feature ZVB and ZIB to generate an
   612	auxiliary feature ZFB . Inspired by [Li et al., 2022] [Wang et
   613	al., 2018], the fusion method can be formulated as follows:
   614	1
   615	Z̄FB = [(ZVB Pq )T (ZIB Pk )](ZIB Pv ) + ZVB
   616	C
   617	2
   618	B
   619	ZF = [(ZIB Qq )(Z̄FB Qk )T ](Z̄FB Qv ) + ZIB
   620	(16)
   621	B
   622	B
   623	
   624	(20)
   625	
   626	where lorth represents the constraint in formula (10) for parameter P to achieve the decomposition of orthogonal subspaces.
   627	
   628	Thereby, the total approaching function for Z̄ is:
   629	lapp = lf kl + ldkl + ldcorr
   630	
   631	(19)
   632	
   633	Here, ZVB/I ∈ R 2 ×C represents the cross-modality base embedding, P ,Q are the learnable parameters. The fused ZFB aggregates the base knowledge from VIS and IR modalities, employing attention mechanisms across both channel and batch
   634	
   635	ltotel = lid + ltri + lokl + lDF E + lBEG + lskd
   636	
   637	4
   638	
   639	(22)
   640	
   641	Experiments
   642	
   643	In this section, we validate the effectiveness of our BDLF
   644	by conducting experiments on the widely recognized SYSUMM01, RegDB and LLCM benchmarks.
   645	
   646	4.1
   647	
   648	Datasets and Evaluation Protocol
   649	
   650	SYSU-MM01 dataset [Wu et al., 2017] comprises 287,628
   651	VIS and 15,792 IR images from 491 identities captured by
   652	4 RGB and 2 IR cameras. It features both All-Search and
   653	Indoor-Search modes for evaluation. RegDB [Nguyen et al.,
   654	2017] contains 412 identities, each represented by 10 VIS and
   655	10 IR images captured from a pair of cameras. We adhere to
   656	the evaluation protocol outlined in [Ye et al., 2022b] to randomly split the identities into training and testing sets of equal
   657	size. LLCM [Zhang and Wang, 2023]is a challenging largescale low-light dataset for VI-ReID task, which contains 713
   658	
   659	SYSU-MM01
   660	Methods
   661	
   662	Venue
   663	
   664	LLCM
   665	
   666	All-Search Indoor-Search VIS to IR IR to VIS VIS to IR IR to VIS
   667	R-1 mAP R-1
   668	
   669	CAJ[Ye et al., 2021a]
   670	ICCV’21
   671	69.9
   672	MMN[Zhang et al., 2021]
   673	ACMMM’21 70.6
   674	FMCNet[Zhang et al., 2022a] CVPR’22
   675	66.3
   676	LUPI[Alehdaghi et al., 2022] ECCV’22
   677	71.1
   678	MSCLNet[Zhang et al., 2022b] ECCV’22
   679	77.0
   680	DEEN[Zhang and Wang, 2023] CVPR’23
   681	74.7
   682	SGIEL[Feng et al., 2023]
   683	CVPR’23
   684	75.2
   685	TMD[Lu et al., 2024]
   686	TMM’23
   687	73.9
   688	AGCC[Yu et al., 2024]
   689	PR’24
   690	75.9
   691	ReViT[Sarker and Zhao, 2024] PR’24
   692	68.1
   693	STAR[Wu et al., 2024]
   694	TMM’24
   695	76.1
   696	BDLF(ours)
   697	
   698	RegDB
   699	
   700	-
   701	
   702	66.9 76.3
   703	66.9 76.2
   704	62.5 68.2
   705	67.6 82.4
   706	71.6 78.5
   707	71.8 80.3
   708	70.1 78.4
   709	67.8 81.2
   710	73.0 79.3
   711	65.1 72.4
   712	72.7 83.5
   713	
   714	76.8 74.6 84.2
   715	
   716	mAP R-1 mAP R-1 mAP R-1 mAP R-1 mAP
   717	80.4 85.0
   718	79.6 91.6
   719	74.1 89.1
   720	82.7 88.0
   721	81.2 84.2
   722	83.3 91.1
   723	81.2 92.2
   724	78.9 93.0
   725	84.6 92.6
   726	77.6 91.7
   727	85.8 94.1
   728	
   729	79.1 84.8
   730	84.1 87.5
   731	84.4 88.4
   732	82.7 86.8
   733	81.0 83.7
   734	85.1 89.5
   735	86.6 91.1
   736	84.3 87.4
   737	86.2 91.4
   738	86.0 93.0
   739	88.8 93.3
   740	
   741	77.8 56.5 59.8 48.8 56.6
   742	80.5 59.9 62.7 52.5 58.9
   743	83.9
   744	81.3
   745	78.3
   746	83.4 62.5 65.8 54.9 62.9
   747	85.2
   748	81.3
   749	84.9
   750	86.1
   751	88.2
   752	-
   753	
   754	85.8 94.4 90.1 94.5 89.6 67.0 68.9 58.1 64.5
   755	
   756	Table 1: Comparisons between the proposed BDLF and several state-of-the-art methods on the SYSU-MM01, RegDB, and LLCM datasets.
   757	
   758	Settings
   759	DFE
   760	
   761	BEG
   762	
   763	lapp
   764	
   765	SYSU-MM01
   766	lorth
   767	
   768	lskd
   769	
   770	R-1
   771	
   772	mAP
   773	
   774	✓
   775	✓
   776	✓
   777	✓
   778	
   779	72.7
   780	73.7
   781	75.3
   782	73.7
   783	74.0
   784	75.5
   785	76.8
   786	
   787	68.1
   788	69.6
   789	72.1
   790	69.0
   791	70.9
   792	72.3
   793	74.6
   794	
   795	✓
   796	✓
   797	✓
   798	✓
   799	✓
   800	✓
   801	
   802	✓
   803	✓
   804	✓
   805	✓
   806	✓
   807	✓
   808	✓
   809	
   810	✓
   811	✓
   812	✓
   813	✓
   814	
   815	✓
   816	✓
   817	
   818	Table 2: Effectiveness of each component for the proposed BDLF.
   819	
   820	identities with 25,626 VIS and 21,141 IR images, all captured
   821	by 9 cameras in both RGB and IR modalities
   822	The Cumulative Matching Characteristic curve (CMC) and
   823	mean Average Precision(mAP) are adopted as standard evaluation metrics in our experiments to comprehensively assess
   824	the performance of our framework.
   825	
   826	4.2
   827	
   828	Implementation Details
   829	
   830	The entire framework is implemented using PyTorch and runs
   831	on a single NVIDIA RTX3090 GPU with 24GB VRAM. We
   832	employed a pre-trained ResNet-50[He et al., 2016b] as the
   833	backbone network and incorporated INN blocks with affine
   834	coupling layers[Dinh et al., 2017][Zhou et al., 2022] to construct the DFE module, setting the number of INN blocks to
   835	6. All images are resized to 3×384×144, and we adopted the
   836	Random Channel Exchangeable Augmentation and ChannelLevel Random Erasing techniques proposed in [Ye et al.,
   837	2021b] during the training phase. The SGD optimizer was
   838	used, with the initial learning rate set to 1 × 10−2 , which was
   839	warmed up to 1 × 10−1 during the first 10 epochs, then we
   840	decayed the learning rate to 1 × 10−2 and 1 × 10−3 at epochs
   841	20 and 95 for SYSU-MM01, and at epochs 70 and 130 for
   842	
   843	RegDB and LLCM, respectively. The learning rate was further decayed to 1 × 10−4 at 180 epoch, with a total of 220
   844	epochs. For each mini-batch, we randomly sampled 8 identities, each consisting of 4 VIS and 4 IR images for training.
   845	Additionally, the exponential moving average (EMA) model
   846	[Ge et al., 2020] also employed in our method.
   847	
   848	4.3
   849	
   850	Comparison with State-of-the-art Methods
   851	
   852	We demonstrate the superiority of our BDLF by comparing
   853	performance with several existing state-of-the-art methods on
   854	the SYSU-MM01, RegDB, and LLCM datasets. The performance of these methods is presented in Table 1, with optimal
   855	performances annotated by underlining.
   856	Comparison on SYSU-MM01 and RegDB. Table 1
   857	presents the results of our BDLF alongside selected outstanding methods, confirming the superiority of our BDLF, which
   858	almost outperforms all other state-of-the-art methods. In the
   859	All-Search mode of SYSU-MM01, our method achieved a
   860	rank-1 accuracy of 76.8% and a mAP of 74.6%, in the IndoorSearch mode, BDLF achieved a rank-1 accuracy of 84.2%
   861	and a mAP of 85.8%. On the RegDB dataset, our method
   862	achieved a rank-1 accuracy of 94.4% and a mAP of 90.1%
   863	for the VIS to IR search, and attained a rank-1 accuracy of
   864	94.5% and a mAP of 89.6% for the IR to VIS search. These
   865	results validate the effectiveness of BDLF that independently
   866	learns the detail and base information and sufficiently utilizes
   867	cross-modalities knowledge.
   868	Comparison on LLCM. According to Table 1, our method
   869	outperformed other approaches. Specifically, BDLF achieved
   870	a rank-1 accuracy of 67.0% and a mAP of 68.9% in VIS to
   871	IR search, as well as a rank-1 accuracy of 58.1% and a mAP
   872	of 64.5% in IR to VIS search. It is evident that our BDLF is
   873	well-equipped to handle challenging scenarios.
   874	
   875	4.4
   876	
   877	Ablation Studies
   878	
   879	Effectiveness of each component. In this section, we designed an ablation experiment to validate the effectiveness
   880	
   881	Figure 4: Effectiveness of how many INN blocks are more favorable
   882	for the proposed DFE.
   883	
   884	Location of DFE
   885	After stage-1
   886	After stage-2
   887	After stage-3
   888	After stage-4
   889	
   890	SYSU-MM01
   891	R-1
   892	
   893	mAP
   894	
   895	59.9
   896	71.1
   897	76.8
   898	73.4
   899	
   900	55.3
   901	67.1
   902	74.6
   903	71.1
   904	
   905	Table 3: Effectiveness of which stage of ResNet-50 to combine the
   906	proposed DFE.
   907	
   908	of certain components of BDLF. Specifically, we removed
   909	the DFE, lapp , lorth and lskd modules from BDLF, while
   910	retaining the backbone with the BEG block as the baseline. All experiments adopted the same training settings, and
   911	we evaluated their performance in the All-Search mode of
   912	SYSU-MM01. The results are presented in Table 2, Notably, the removal of the DFE module resulted in poor precision, demonstrating the effective detail extraction capability
   913	of DFE. The experiments also indicated that the lapp loss enhances the model’s distinguishing performance by effectively
   914	aiding in the generation of base embeddings, eliminating detailed knowledge from the comprehensive feature. Although
   915	the DFE module significantly promotes the mining of detail information, its performance remains suboptimal, as the
   916	model cannot extract all modality-specific and shared information without interference each other due to the absence of
   917	correlation constraint lcorr and independent constraint lorth .
   918	Effectiveness of how many INN blocks are more favorable
   919	for DFE. The proposed DFE module consists of a series
   920	of INN blocks with an LN layer to extract detail information non-destructively. We conducted experiments to determine the optimal number of blocks for our framework. As
   921	shown in Figure 4, we modified the number of INN blocks
   922	and evaluated performance in the All-Search mode of SYSUMM01. The results indicate accuracy gradually improves as
   923	the number of INN blocks increases, reaching a plateau when
   924	the count is 6. This confirm that a balance exists between
   925	accuracy and computational complexity when the number of
   926	INN blocks is set to 6.
   927	Effectiveness of which stage of ResNet-50 to combine DFE
   928	module. In this section, we implement experiments to assess which stage of ResNet-50 is most suitable for serving
   929	as the input to the proposed DFE module. All experiments
   930	maintain consistent settings, except for the locations of the
   931	
   932	Figure 5: Visualization of the comprehensive and detailed features.
   933	
   934	DFE module within ResNet-50. The results are presented
   935	in Table 3, we observed that connecting the DFE module to
   936	stage-3 of ResNet-50 yielded the best accuracy in the AllSearch mode of SYSU-MM01. This can be attributed to the
   937	fact that modality-shared information is more prominent in
   938	the high-level features produced by stages-4, which impedes
   939	the extraction of modality-specific detail information. Furthermore, the low-level features generated by stages 1 and 2
   940	are inadequate for effectively expressing the semantics necessary to distinguish between different identities. These findings elucidate why the best accuracy is achieved when the
   941	DFE module is connected to stage-3 of ResNet-50.
   942	
   943	4.5
   944	
   945	Visualization
   946	
   947	To investigate the detail information extraction capabilities
   948	of the proposed DFE, we visualize the comprehensive and
   949	detailed features of several identities produced by BDLF. As
   950	illustrated in Figure 5, a comparison of the images of comprehensive and detailed features reveals that the attention regions

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf' - | nl -ba | sed -n '850,1120p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   850	𝑉
   851	𝐿
   852	̂ 𝑅2𝑉
   853	)
   854	𝑖
   855	 = (1 − 𝑉 (𝑥𝑖 ) + (0 − 𝑉 (𝑥
   856	(
   857	)
   858	+ (1 − 𝑅 (𝑥𝑅
   859	̂ 𝑉𝑖 2𝑅 ) .
   860	𝑖 ) + (0 − 𝑅 (𝑥
   861	
   862	𝑗
   863	
   864	11
   865	
   866	(log |𝑠|)
   867	
   868	tively:
   869	
   870	𝑗
   871	
   872	𝑖
   873	
   874	∑
   875	
   876	1+𝑧<𝑙−1>
   877	
   878	1 ∑
   879	𝑑(𝑉 (𝑥𝑉𝑖 ), 𝑉 (𝑥̂ 𝑅2𝑉
   880	))
   881	𝑗
   882	𝑛𝑉1 𝑦𝑉 =𝑦𝑅
   883	
   884	1 ∑
   885	+2− 𝑅
   886	𝑑(𝑅 (𝑥𝑅
   887	̂ 𝑉𝑗 2𝑅 )),
   888	𝑖 ), 𝑅 (𝑥
   889	𝑛1 𝑦𝑅 =𝑦𝑉
   890	
   891	∑
   892	
   893	𝑧<𝑙>
   894	= 12 log 1−𝑧𝑖<𝑙−1>
   895	𝑖
   896	
   897	𝑖
   898	
   899	the similarity between the true images and generated images of the same
   900	identity:
   901	𝐿
   902	 =2−
   903	
   904	log-determinant
   905	∑
   906	(log ||det(𝑊𝑙 )||)
   907	
   908	(19)
   909	
   910	Summary on Flow2Flow. For the purpose of reducing crossmodality discrepancy and expanding training samples, we have two objectives for model design: dual generation capability and uniﬁed framework development. The proposed Flow2Flow is the ﬁrst uniﬁed architecture combining ﬂow-based generation with adversarial training for
   911	visible-to-infrared (V2I) person ReID, which can simultaneously achieve
   912	training sample expansion and cross-modal image synthesis within a single model. The rational behind Flow2Flow is the invertibility of ﬂowbased generative models. Given a visible image 𝑥𝑉 and an infrared one
   913	𝑥𝑅 , the visible ﬂow and infrared ﬂow map 𝑥𝑉 and 𝑥𝑅 to a shared Gaussian latent space 𝑧. Thus, the cross-modal generation can be achieved by
   914	
   915	(14)
   916	
   917	in which 𝑉 and 𝑅 output the modality logits. While in the stage of
   918	training discriminators, the modality discriminators 𝑉 and 𝑅 struggle to classify the true images and generated ones as 1 and 0, respec5
   919	
   920	Pattern Recognition 172 (2026) 112622
   921	
   922	H. Pan et al.
   923	
   924	Table 3
   925	Generalization veriﬁcation on multiple baselines. In Wilcoxon signed-rank test,
   926	W and p denote the test statistic and p-value, respectively.
   927	Baseline
   928	
   929	Augmentation All-Search
   930	Rank1 mAP
   931	
   932	Rank1 mAP
   933	
   934	DDAG [38]
   935	
   936	None
   937	TSE
   938	CMG
   939	
   940	54.75
   941	58.80
   942	55.61
   943	
   944	53.02
   945	57.48
   946	53.63
   947	
   948	61.02
   949	64.18
   950	61.91
   951	
   952	67.98
   953	70.39
   954	68.34
   955	
   956	W=0.0, p=0.0431
   957	W=1.0, p=0.0696
   958	
   959	MPANet [7] None
   960	TSE
   961	CMG
   962	
   963	70.58
   964	72.40
   965	71.75
   966	
   967	68.24
   968	69.77
   969	69.30
   970	
   971	76.74
   972	77.02
   973	77.66
   974	
   975	80.95
   976	81.24
   977	81.52
   978	
   979	W=0.0, p=0.0339
   980	W=0.0, p=0.0521
   981	
   982	SGIEL [32]
   983	
   984	75.47
   985	77.18
   986	75.91
   987	
   988	71.11 81.34
   989	72.79 82.70
   990	71.72 81.82
   991	
   992	None
   993	TSE
   994	CMG
   995	
   996	Indoor-Search
   997	
   998	Table 4
   999	Veriﬁcation of conﬁdence interval (CI). We repeat the experiment 10 times
  1000	to calculate each CI value.
  1001	
  1002	Wilcoxon signed-rank test
  1003	
  1004	All-Search
  1005	
  1006	Setting
  1007	
  1008	Flow2Flow-TSE
  1009	
  1010	One-shot 72.40
  1011	69.77
  1012	77.02
  1013	81.24
  1014	CI
  1015	72.23±0.39 69.48±0.53 69.94±0.42 81.28±0.50
  1016	
  1017	Flow2Flow-CMG
  1018	
  1019	One-shot 71.75
  1020	69.30
  1021	77.66
  1022	81.52
  1023	CI
  1024	71.08±0.28 69.32±0.32 77.60±0.22 81.47±0.35
  1025	
  1026	Rank1
  1027	
  1028	83.18 84.60 W=0.0, p=0.0423
  1029	84.00 W=0.0, p=0.0637
  1030	
  1031	Indoor-Search
  1032	
  1033	Method
  1034	
  1035	mAP
  1036	
  1037	Rank1
  1038	
  1039	mAP
  1040	
  1041	ent from cmPIG [12] that directly concatenates the true images and
  1042	generated images, we perform the feature-level concatenation. For the
  1043	evaluation metrics, we report mAP and Rank1 of CMC. The mAP is
  1044	computed as:
  1045	
  1046	𝑥𝑉 → 𝑧 → 𝑥𝑅 and 𝑥𝑅 → 𝑧 → 𝑥𝑉 , and the training samples can be derived
  1047	by mapping from latent space to image space. The proposed Flow2Flow
  1048	enjoys two innovations: framework contribution and new adversarial
  1049	mechanism. For the ﬁrst one, unlike prior GAN/ﬂow-based methods
  1050	(e.g., AlignGAN [11], cmGAN [8]) that perform one-directional modality translation, our Flow2Flow introduces two new innovations: bidirectional cross-modal generation (visible ↔ infrared) via coupled invertible ﬂows and joint training sample expansion while preserving identity, addressing data scarcity without external datasets. For the second
  1051	one, unlike standard GAN discriminators, our identity-modality decoupled adversarial training simultaneously enforces identity consistency
  1052	via encoder  and modality realism via discriminator . Our model
  1053	ensures mathematical alignment in the latent space for visible and infrared modality, which existing GAN-based methods cannot achieve.
  1054	
  1055	1 ∑
  1056	AP𝑖 ,
  1057	𝑁 𝑖=1
  1058	𝑁
  1059	
  1060	mAP =
  1061	
  1062	(20)
  1063	
  1064	where AP is computed using precision (𝑝) and recall (𝑟):
  1065	AP𝑖 =
  1066	
  1067	𝐾
  1068	∑
  1069	(𝑝𝑘 + 𝑝𝑘−1 )(𝑟𝑘 − 𝑟𝑘−1 )
  1070	
  1071	2
  1072	
  1073	𝑘=1
  1074	
  1075	.
  1076	
  1077	(21)
  1078	
  1079	For CMC, we compute the top-K accuracy Acc𝑖 @𝐾 for each query sample, and deﬁne rank K as following:
  1080	1 ∑
  1081	Acc𝑖 @𝐾.
  1082	𝑁 𝑖=1
  1083	𝑁
  1084	
  1085	𝑅𝐾 =
  1086	
  1087	(22)
  1088	
  1089	5. Experiments
  1090	
  1091	5.2. Ablation studies
  1092	
  1093	5.1. Experimental settings
  1094	
  1095	5.2.1. Eﬀectiveness and generalization veriﬁcation
  1096	In this section, we ﬁrst verify the eﬀectiveness and generalization of
  1097	data generation by Flow2Flow. Speciﬁcally, we conduct the experiments
  1098	on the basis of DDAG [38], MPANet [7] and SGIEL [32], and test the
  1099	performance of training sample expansion (TSE) and cross-modal generation (CMG). We report the results in Table 3. As can be seen, TSE and
  1100	CMG could signiﬁcantly improve the performance of the baseline models, which demonstrated the generalization of Flow2Flow. In the next
  1101	sections, the baseline model for ablation studies denotes MPANet [7].
  1102	Statistical veriﬁcation. To further verify the statistical conﬁdence,
  1103	we perform Wilcoxon signed-rank test on the metric of Rank1. Specifically, we repeated the experiment ﬁve times and used the results of
  1104	Rank1 to perform the Wilcoxon signed-rank test. We compare the model
  1105	of TSE or CMG with baseline models. In this test, p-value < 0.05 indicates signiﬁcant diﬀerence between groups. As can be seen in Table 3,
  1106	models with TSE obtains p-value < 0.05, meaning that training sample
  1107	expansion obtains signiﬁcant improvements. While p-value in CMG is
  1108	slightly larger than 0.05. Another statistical veriﬁcation is to calculate
  1109	the conﬁdence interval (CI). We repeat the experiment 10 times, and use
  1110	Bootstrap CI calculation method to compute the 95 % CI for Rank-1 accuracy or mAP on a test set. As can be seen in Table 4, for TSE and CMG,
  1111	our method achieves 72.23 % Rank1 accuracy (95 % CI [71.84, 72.62])
  1112	and 71.08 % Rank1 accuracy (95 % CI [70.80, 71.36]), respectively.
  1113	
  1114	Benchmarks. To validate the eﬀectiveness of Flow2Flow, we
  1115	conduct experiments on two widely-used V2I ReID datasets: SYSUMM01 [5] and RegDB [6]. SYSU-MM01 is composed of 491 identities
  1116	captured by 4 visible and 2 infrared cameras. Its training set consists of
  1117	20,284 visible images and 9929 infrared images from 296 persons, and
  1118	the query set contains 3803 infrared images from 96 identities. RegDB
  1119	dataset is composed of 4120 visible images and 4120 infrared images
  1120	from 412 identities, where each identity contains 10 visible images and

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Beyond geometry - The power of texture in interpretable 3D person ReID.pdf' - | nl -ba | sed -n '1,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Computer Vision and Image Understanding 261 (2025) 104517
     2	
     3	Contents lists available at ScienceDirect
     4	
     5	Computer Vision and Image Understanding
     6	journal homepage: www.elsevier.com/locate/cviu
     7	
     8	Beyond geometry: The power of texture in interpretable 3D person ReID
     9	Huy Nguyen ∗, Kien Nguyen, Akila Pemasiri, Sridha Sridharan, Clinton Fookes
    10	Signal Processing, Artificial Intelligence and Vision Technologies (SAIVT), Queensland University of Technology (QUT), Brisbane, QLD 4000, Australia
    11	
    12	ARTICLE
    13	
    14	INFO
    15	
    16	Communicated by Shiliang Zhang
    17	Keywords:
    18	3D person re-identification
    19	Texture UV mapping
    20	3D explainability
    21	
    22	ABSTRACT
    23	This paper presents FusionTexReIDNet, a robust framework for 3D person re-identification that uniquely
    24	leverages UVTexture to enhance both performance and explainability. Unlike existing 3D person ReID
    25	approaches that simply overlay textures on point clouds, our method exploits the full potential of UVTexture
    26	through its high resolution and normalized coordinate properties. The framework consists of two main
    27	streams: a UVTexture stream that processes appearance features and a 3D stream that handles geometric
    28	information. These streams are fused through an effective combination of KNN, attribute-based, and explainable
    29	re-ranking strategies. Our approach introduces explainability to 3D person ReID through the visualization of
    30	activation maps on UVTextures, providing insights into the model’s decision-making process by highlighting
    31	discriminative regions. By incorporating the Intersection-Alignment Score derived from activation maps and
    32	visible clothing masks, we further improve the ReID accuracy. Extensive experiments demonstrate that
    33	FusionTexReIDNet achieves state-of-the-art performance across various scenarios, with Rank-1 accuracies of
    34	98.5% and 89.7% Rank-1 on benchmark datasets, while providing interpretable results through its explainable
    35	component.
    36	
    37	1. Introduction
    38	Person re-identification (ReID) is the task of recognizing individuals across different non-overlapping camera views in a surveillance
    39	system (Nguyen et al., 2022, 2023). This task involves matching individuals based on their visual characteristics, such as appearance, body
    40	shape, and clothing, across different camera views. The importance
    41	of person ReID has been highlighted by its security surveillance applications, which has spurred significant progress in the field, driven
    42	by the availability of larger and more diverse datasets (Zheng et al.,
    43	2015; Ristani et al., 2016; Nguyen et al., 2024) and advancements in
    44	deep learning techniques for person representation (Khatun et al., 2020;
    45	Hafner et al., 2022; Liu et al., 2023b).
    46	While 2D person ReID methods have advanced significantly (Zhang
    47	et al., 2023; Lin et al., 2023; Weng et al., 2023), they miss a crucial
    48	aspect: humans are inherently three-dimensional (3D) entities. Traditional 2D ReID approaches often struggle with unpredictable viewpoint
    49	changes and geometric deformations caused by camera positioning,
    50	which we identify as significant factors responsible for poor generalization performance across domains. Shifting person ReID from 2D to
    51	3D space can enhance the field by leveraging the additional information
    52	provided by 3D human models. These models can normalize variations
    53	in appearance due to human movements, dynamics, camera distances,
    54	and lighting conditions, providing a consistent identity representation (Loper et al., 2015; Xu and Loy, 2021). Moreover, 3D models
    55	
    56	can match identities across different viewing modalities, such as aerial
    57	versus ground-based perspectives, by accounting for the changes in visible body parts and explicitly addressing viewpoint variations (Nguyen
    58	et al., 2023, 2024).
    59	However, state-of-the-art (SOTA) 3D person ReID approaches still
    60	lag behind their 2D counterparts in performance and face significant
    61	challenges in handling unpredictable viewpoint changes across domains. Most existing 3D person ReID methods (Wang et al., 2023;
    62	Zheng et al., 2020; Chen et al., 2021a) simply reconstruct 3D models
    63	from input images and directly perform classification on them without
    64	explicitly addressing the viewpoint normalization problem. For example, PointReIDNet (Wang et al., 2023), 3DInvarReID (Liu et al., 2023a)
    65	and OG-Net (Zheng et al., 2020) first infer an SMPL-based 3D human
    66	model (Loper et al., 2015) from the input image, overlay the 3D model
    67	with texture, encode the overlayed 3D point cloud, then classify the
    68	encoded vector for person ReID. However, these approaches fail to
    69	leverage the explicit viewpoint alignment capabilities that 3D reconstruction can provide, missing the opportunity to normalize geometric
    70	deformations and appearance variations caused by camera positioning.
    71	Unlike existing methods that simply overlay textures on 3D models
    72	or ignore viewpoint variations, our work addresses these fundamental
    73	limitations through explicit 3D viewpoint alignment and robust texture
    74	representation. We propose FusionTexReIDNet, an innovative method
    75	
    76	∗ Corresponding author.
    77	
    78	E-mail address: nguyet91@qut.edu.au (H. Nguyen).
    79	https://doi.org/10.1016/j.cviu.2025.104517
    80	Received 9 December 2024; Received in revised form 20 August 2025; Accepted 18 September 2025
    81	Available online 26 September 2025
    82	1077-3142/© 2025 The Author(s). Published by Elsevier Inc. This is an open access article under the CC BY license
    83	(http://creativecommons.org/licenses/by/4.0/).
    84	
    85	H. Nguyen, K. Nguyen, A. Pemasiri et al.
    86	
    87	Computer Vision and Image Understanding 261 (2025) 104517
    88	
    89	• We pioneer the integration of explainability into 3D person ReID
    90	by developing a visualization technique that projects activation maps onto UVTextures and 3D models in canonical viewpoints. This provides clear insights into discriminative features
    91	free from viewpoint-induced distortions, enabling better understanding of model decisions and facilitating performance improvements through explainable re-ranking strategies.
    92	• We achieve unprecedented performance across seven diverse
    93	datasets spanning ground–ground, aerial–aerial, and aerial–ground
    94	scenarios, with our method reaching 98.5% Rank-1 accuracy
    95	on Market-1501 and 89.7% on challenging AG-ReID.v2. These
    96	results demonstrate the first instance where 3D person ReID
    97	methods surpass their 2D counterparts through explicit viewpoint
    98	alignment.
    99	2. Related work
   100	This section reviews and discusses prior work in person ReID in the
   101	3D space and explainability in person ReID.
   102	Person ReID in 3D. Literature of existing approaches in 3D person
   103	ReID can generally be organized into two main categories based on
   104	their utilization of the reconstructed 3D human models:
   105	(1) Direct 3D model-based classification: Leveraging 3D information
   106	in person ReID provides an effective approach by utilizing shape and
   107	spatial depth features, which are particularly robust to variations in
   108	texture. Point clouds, recognized for their depth information, add a
   109	valuable dimension to data representation (Qi et al., 2016). Innovations
   110	such as OG-Net by Zheng et al. (2020) convert 2D images into a 3D
   111	data framework, incorporating structural and appearance information.
   112	This approach is further explored by Liu et al. (2023a) and Wang
   113	et al. (2023), who delve into distinguishing identity-specific features
   114	from changeable aspects like clothing and posture in 3D shape representations. These advancements highlight the significant potential of
   115	3D information in enhancing person ReID beyond the specific challenges of occlusion. However, these methods often struggle to capture
   116	fine-grained texture details, which are crucial for accurate person ReID.
   117	(2) 3D model-based data augmentation: Addressing occlusions in 3D
   118	Re-ID, contributions such as PersonX (Sun and Zheng, 2018) and Wang
   119	et al. (2022a) enhance data representation by utilizing 3D scanning
   120	and UV mapping techniques, further enriched by the integration of
   121	point clouds to incorporate depth information. Works like ASSP (Chen
   122	et al., 2021a) and JGCL (Chen et al., 2020) show advancements in
   123	merging 2D and 3D data, applying adversarial and contrastive learning
   124	strategies, while Zhang et al.’s 3DT model (Zhang et al., 2022a) tackles
   125	the challenge of group ReID in occluded environments through 3D
   126	transformations. TranSG (Rao and Miao, 2023) introduces a focus on
   127	utilizing 3D skeleton data for Re-ID, emphasizing skeletal graphs and
   128	spatial–temporal semantics. However, it potentially overlooks crucial
   129	details like body size, hairstyle, and clothing, pointing to the nuanced
   130	challenges of clothing representation and diverse body movements in
   131	3D person ReID research. Notably, while there is significant progress,
   132	studies focusing on 3D ReID using video inputs, such as those by Liao
   133	et al. (2019), Han et al. (2022), outline the breadth of methodologies
   134	beyond the current discussion’s scope, highlighting the dynamic and
   135	evolving nature of the field. Despite the advancements in synthesizing additional data, these methods often lack the ability to generate
   136	highly realistic and diverse textures, which limits their effectiveness in
   137	real-world scenarios.
   138	Explainability in Person ReID. The blackbox nature of CNN models (Somers et al., 2022; Wang et al., 2021) has led to a growing
   139	interest in developing explainable approaches for person ReID. Existing
   140	works attempt to explain CNN models through various techniques,
   141	such as visualizing salient maps (Selvaraju et al., 2019), distilling
   142	knowledge (Chen et al., 2019a), or learning with decision trees (Zhang
   143	et al., 2019a). In the person ReID domain, attention learning has been
   144	investigated as a means to explain the model’s predictions (Zhang et al.,
   145	
   146	Fig. 1. Overview of the proposed explainable 3D person ReID approach.
   147	From person images captured at different viewpoints, our method reconstructs
   148	3D human models and generates high-resolution UVTexture representations.
   149	The framework leverages three key components: (1) UVTexture maps that
   150	provide viewpoint-invariant appearance representation, (2) visible clothing
   151	masks that identify non-occluded regions, and (3) activation map alignment
   152	that highlights discriminative features. Unlike 2D methods that suffer from
   153	viewpoint-dependent activation patterns and background bias, our 3D approach enables spatially consistent feature analysis across different camera
   154	angles, providing interpretable insights into which body regions contribute to
   155	identity matching.
   156	
   157	that leverages 3D human models to perform explicit viewpoint normalization, projecting pedestrian images from arbitrary camera views into
   158	canonical viewpoints to mitigate geometric deformations and appearance discrepancies. Our approach combines UVTexture representation
   159	with 3D viewpoint alignment to achieve improved performance and
   160	explainability, as illustrated in Fig. 1. By utilizing the high-resolution
   161	and normalized coordinate properties of UVTexture, our method captures fine-grained appearance details while enabling precise localization of discriminative features on the human body. Additionally, we
   162	introduce a transformer-based fusion module that compensates for
   163	reconstruction errors by aligning and fusing visual cues from original
   164	and canonical view images, ensuring robust performance even when
   165	3D reconstruction is imperfect.
   166	With these contributions, we make 3D person ReID performance
   167	surpass 2D counterparts for the first time while introducing explainability through viewpoint-aware feature analysis. Our work addresses the
   168	critical challenge of unpredictable viewpoint changes in cross-domain
   169	scenarios and sets a new benchmark for 3D person ReID systems. The
   170	core contributions are summarized as follows:
   171	• We introduce a novel 3D viewpoint alignment framework that
   172	explicitly addresses unpredictable camera view changes by projecting pedestrian images from arbitrary viewpoints into canonical views. This approach mitigates geometric deformations and
   173	appearance variations caused by viewpoint changes, achieving
   174	superior generalization performance across domains with up to
   175	7.7% improvement in Rank-1 accuracy over state-of-the-art 3D
   176	approaches.
   177	• We develop a dual-stream architecture that strategically combines
   178	explicit viewpoint normalization with high-resolution UVTexture
   179	representation. Our transformer-based fusion module enables effective alignment and information compensation between original and canonical view images, addressing reconstruction errors
   180	while preserving discriminative features for accurate identification.
   181	2
   182	
   183	H. Nguyen, K. Nguyen, A. Pemasiri et al.
   184	
   185	Computer Vision and Image Understanding 261 (2025) 104517
   186	
   187	Fig. 2. The proposed model has two streams: UVTextureNet extracts appearance features 𝐹𝐴 from UVTexture images, while 3DReIDNet processes 3D rendered
   188	point clouds using KNN graphs to extract geometry features 𝐹𝑆 . Each stream produces distance matrices (𝐷1 , 𝐷2 ), fused to yield joint distance matrix 𝐷. The
   189	Explainable Module overlays attention maps on UVTexture, visualizing model focus and incorporating visibility masks for enhanced explainability. It computes
   190	Intersection-Alignment Score (IAS) from IoU between attention maps and visible clothes masks, plus activation alignment scores. Three re-ranking methods (KNN,
   191	attribute, explainable) refine distance matrix 𝐷, with IAS improving ReID performance.
   192	
   193	2020; Nguyen et al., 2023). Some notable examples of explainable
   194	techniques in the 2D domain include Grad-CAM (Selvaraju et al.,
   195	2019), which highlights areas of interest in images, and attributebased methods that visualize differences or similarities between person
   196	matching (Chen et al., 2021b; Nguyen et al., 2023). Despite the progress
   197	made in explainable person ReID, the majority of these approaches
   198	focus on 2D image-based methods, and there are limitations to their
   199	effectiveness. For instance, 2D explainable methods often struggle to
   200	capture the complex spatial relationships and depth information that is
   201	crucial for understanding person appearance and behavior in real-world
   202	scenarios. The problem of domain generalization in person ReID has
   203	gained increasing attention (Bhuiyan et al., 2024; Liu et al., 2024), particularly when dealing with varying viewpoints and camera conditions.
   204	Additionally, 2D methods may be sensitive to variations in viewpoint,
   205	occlusion, and illumination, which can hinder their ability to provide
   206	reliable explanations.
   207	In contrast, 3D data offers potential for explainability that 2D
   208	does not, presenting a more comprehensive view of the spatial relationships and depth information that are crucial for understanding
   209	complex scenes. Our paper investigates this potential, exploring how
   210	3D data can enhance the explainability of person ReID models by
   211	leveraging the additional dimension to provide deeper insights into
   212	the model’s decision-making processes. The limitations of existing 2D
   213	explainable methods and the lack of established 3D explainable approaches motivate our proposed UVTexture-inspired explainable person
   214	ReID method, which aims to bridge this gap and provide a more
   215	comprehensive understanding of the ReID process. By utilizing the
   216	rich information available in 3D data, our approach addresses the
   217	limitations of 2D methods and offers explainability in person ReID.
   218	
   219	3.1. Preliminaries: UVTexture and 3D reconstruction
   220	Inputs for our model are 3D human models and UVTexture. To
   221	reconstruct 3D human models and UVTexture for our inputs, we employ
   222	RSC-Net (Xu et al., 2021) and Texformer (Xu and Loy, 2021). It is worth
   223	noting that other models and approaches can also be used. RSC-Net
   224	is an algorithm designed to address the challenges of estimating 3D
   225	human pose from low-resolution images and videos (Xu et al., 2021,
   226	2020). RSC-Net achieves impressive results even when working with
   227	low-resolution input data by integrating a resolution-aware network
   228	that adapts to different resolutions, employing self-supervision loss, and
   229	leveraging contrastive learning for high-quality 3D reconstructions.
   230	For creating 3D textures via UVTexture maps for humans, we adopt
   231	a transformer-based framework, Texformer (Xu and Loy, 2021). Texformer leverages global information, enabling the generation of realistic texture maps from input images. Recent advances in transformer
   232	architectures have shown promising results in person ReID (Li et al.,
   233	2024; Lin et al., 2024), particularly in handling complex visual relationships and feature extraction. Unlike traditional UV maps, which may
   234	lack spatial consistency across different individuals and conditions (Xiu
   235	et al., 2023; Pumarola et al., 2019), the UVTexture maps produced
   236	by Texformer are particularly effective for supporting the downstream
   237	task of explainable 3D person ReID due to their enhanced spatial
   238	consistency.
   239	3.2. Framework overview
   240	This section presents the overall architecture of FusionTexReIDNet, a comprehensive framework for 3D person re-identification that
   241	uniquely leverages both 3D structural information and high-resolution
   242	UVTexture details. The proposed framework addresses the fundamental
   243	limitations of existing 3D person ReID methods by introducing explicit
   244	viewpoint normalization and robust texture representation through a
   245	dual-stream architecture. As illustrated in Fig. 2, FusionTexReIDNet
   246	consists of four main components working in concert: (1) 3D reconstruction and UVTexture generation — the preprocessing stage that transforms
   247	2D pedestrian images into 3D human models with corresponding UVTexture maps using RSC-Net and Texformer; (2) Dual-stream feature extraction — comprising Stream 1 (UVTextureNet) for appearance feature
   248	extraction and Stream 2 (3DReIDNet) for geometric feature processing; (3) Multi-level fusion module — integrating distance matrices from
   249	both streams through weighted combination and multiple re-ranking
   250	strategies; and (4) Explainable module — providing interpretability
   251	through UVTexture-based activation map visualization and explainable
   252	re-ranking using the Intersection-Alignment Score (IAS). A detailed
   253	breakdown of the method components is provided in Table 1.
   254	
   255	3. Methodology
   256	In this section, we will elaborate on the proposed FusionTexReIDNet
   257	framework and its key components in detail. First, we will introduce
   258	the preliminaries of UVTexture and 3D reconstruction methods in
   259	Section 3.1, which establish the foundational concepts essential for
   260	understanding our approach. Then, we will provide a comprehensive
   261	overview of the overall framework architecture in Section 3.2. Subsequently, we will detail the two main processing streams: the UVTextureemphasized ReID stream in Section 3.3 and the 3D point cloud processing stream in Section 3.4. Finally, we will present our novel explainable module that provides interpretability through UVTexture-based
   262	visualization in Section 3.5.
   263	3
   264	
   265	H. Nguyen, K. Nguyen, A. Pemasiri et al.
   266	
   267	Computer Vision and Image Understanding 261 (2025) 104517
   268	
   269	Table 1
   270	FusionTexReIDNet: Proposed method summary.
   271	Stream/Module
   272	
   273	Input data
   274	
   275	Techniques
   276	
   277	Key contributions
   278	
   279	Stream 1 - UVTextureNet
   280	
   281	Reconstructed UVTexture images
   282	
   283	ResNet50 backbone within SBS framework
   284	
   285	- Extracts discriminative texture features
   286	- Leverages subtle texture variations
   287	
   288	Stream 2 - 3DReIDNet
   289	
   290	3D point cloud with XYZ
   291	and RGB information
   292	
   293	Dynamic graph convolution
   294	Omni-scale Module
   295	
   296	- Processes unordered point cloud data
   297	- Extracts multi-scale features
   298	
   299	Fusion Module
   300	
   301	Distance matrices from both streams
   302	
   303	Weighted sum
   304	KNN and attribute-based re-ranking
   305	Explainable re-ranking
   306	
   307	- Combines 3D structural and texture information
   308	- Refines distance metric results
   309	- Refines results using IAS value
   310	
   311	Explainable Module
   312	
   313	Features from Stream 1, UVTexture model
   314	
   315	Activation map generation
   316	Actmap and visible clothing mask fusion
   317	
   318	- Visualizes relevant texture regions
   319	- Compute IAS value for explainable re-ranking
   320	
   321	The framework operates through a two-stage process that fundamentally differs from existing approaches in its treatment of viewpoint
   322	variations and texture representation. Unlike traditional 2D person
   323	ReID methods that suffer from background bias and viewpointdependent activation patterns, our approach leverages UVTexture’s
   324	viewpoint-invariant properties to provide consistent explanations
   325	across different camera angles. In the first stage, 2D pedestrian
   326	images are processed through 3D reconstruction networks to generate
   327	both geometric point cloud data and high-resolution UVTexture
   328	representations. This preprocessing addresses the critical challenge
   329	of viewpoint variations by projecting arbitrary camera views into
   330	canonical viewpoints, enabling consistent identity representation
   331	across different poses and perspectives while eliminating spurious
   332	background correlations that plague 2D methods.
   333	The second stage employs a dual-stream architecture that processes the reconstructed data in parallel, with Stream 1 (UVTextureNet)
   334	extracting discriminative appearance features from UVTexture maps
   335	and Stream 2 (3DReIDNet) processing geometric features from 3D
   336	point clouds. The explainable component distinguishes our approach by
   337	providing interpretable insights through UVTexture-based visualization
   338	that maintains spatial consistency across viewpoints, enabling precise
   339	feature localization and occlusion-aware matching. The IntersectionAlignment Score (IAS) quantifies how well the model focuses on visible clothing regions, creating a feedback loop where interpretability
   340	directly improves performance through explainable re-ranking. This
   341	comprehensive approach enables FusionTexReIDNet to achieve stateof-the-art performance while providing reliable interpretations that
   342	remain consistent across challenging aerial–ground scenarios where
   343	traditional 2D explanations fail due to extreme viewpoint differences.
   344	
   345	3.4. Stream 2 — 3DReIDNet
   346	While Stream 1 processes high-resolution appearance features from
   347	UVTexture maps, Stream 2 extracts geometric information from 3D
   348	point clouds using our 3DReIDNet architecture inspired by Zheng et al.
   349	(2020). Both our approach and prior 3D ReID methods use RGBenhanced point clouds (XYZ+RGB), but our dual-stream design strategically separates texture and geometry processing to enable domainspecific optimization before fusion. We adopt a Dynamic Graph Convolution (DGC) module that constructs a dynamic graph for feature 𝑓𝑖
   350	using the 𝑘-nearest neighbors (KNN) method.
   351	We also adopt the basic structure of the Omni-scale Module, which
   352	consists of dynamic graph convolution for local feature aggregation
   353	and three branches with different grouping rates to capture multi-scale
   354	patterns. Each branch includes a grouping layer, linear layers, batch
   355	normalization, a squeeze-excitation block, and a group max pooling
   356	layer. The outputs of the three branches are summed to accumulate
   357	different scale patterns. By incorporating the adapted Omni-scale Module into our network architecture, we aim to leverage the benefits of
   358	multi-scale information aggregation and improve the performance of
   359	our proposed method on point cloud processing tasks.
   360	While our point cloud processing module shares similarities with
   361	the work of Zheng et al. (2020), our overall network architecture
   362	introduces a key difference to enhance its performance and adaptability
   363	to our specific task. We extend their work by incorporating a texture
   364	stream that leverages UVTexture to extract discriminative features and
   365	enhance the network’s ability to capture subtle variations in appearance. This texture stream complements the point cloud stream, enabling
   366	our network to learn more comprehensive and robust representations.
   367	To fuse the information from both streams, we employ a weighted
   368	sum of the distance matrices obtained from the texture stream and the
   369	point cloud stream, allowing our network to adaptively balance their
   370	contributions for accurate and robust ReID, as detailed in Appendix A.
   371	
   372	3.3. Stream 1 — UVTextureNet: UVTexture-emphasized ReID
   373	In the first stream of our proposed method, we train a person ReID
   374	model using a ResNet50 backbone architecture (He et al., 2015) on UVTexture maps generated from the 3D reconstruction process described
   375	in Section 3.1. The UVTexture maps provide a 3D representation of
   376	a person’s appearance, independent of their pose or viewpoint in the
   377	original image. We reformulate the dataset into 𝑆 ′ = {(𝑢𝑖 , 𝑦𝑖 )}𝑁 , where
   378	𝑢𝑖 represents the UVTexture map corresponding to a person’s identity
   379	𝑦𝑖 . During training, the ResNet50 model takes a UVTexture map as
   380	input and learns to extract discriminative features for person ReID. This
   381	approach aims to utilize the localized textural features on the UV maps
   382	for ReID, enhancing the model’s ability to distinguish identities based
   383	on the surface texture information.
   384	As shown in Fig. 2, the trained network generates a corresponding
   385	feature map 𝐹𝐴 from the input UVTexture map. The feature map is
   386	then passed through a classifier to obtain a compact feature vector
   387	that encodes the person’s identity. During inference, we compute the
   388	distances 𝐷1 between the feature vectors of the query and gallery
   389	images using cosine distance. The resulting distance matrix, which
   390	captures the similarity between the query and gallery images in the
   391	UVTexture space, is utilized in the fusion module for the final ReID.
   392	
   393	3.5. Explainable module — UVTexture-inspired explainability
   394	Traditional 2D person ReID models suffer from significant interpretability challenges, particularly their susceptibility to background
   395	bias and viewpoint-dependent activation patterns. As demonstrated in
   396	the top row of Fig. 3, 2D ReID activation maps often highlight spurious
   397	background correlations rather than person-specific features, leading
   398	to incorrect matches. This viewpoint dependency makes it difficult
   399	to identify which features truly contribute to identity matching, as
   400	activation patterns vary inconsistently across different camera angles,
   401	preventing systematic analysis of discriminative regions.
   402	In contrast, our 3D UVTexture-based approach provides viewpointinvariant explanations that maintain spatial consistency across different
   403	camera perspectives. The bottom row of Fig. 3 illustrates how UVTexture representation enables precise localization of discriminative
   404	features on a canonical body representation. By projecting person appearance onto a normalized UV coordinate system, our method ensures
   405	that activation maps consistently align regardless of the original camera
   406	4
   407	
   408	H. Nguyen, K. Nguyen, A. Pemasiri et al.
   409	
   410	Computer Vision and Image Understanding 261 (2025) 104517
   411	
   412	Fig. 4. Overview of attribute-guided attention map generation using Attribute
   413	Decomposition Head (ADH). Input image pairs and attributes pass through
   414	CNN for feature extraction, then ADH generates Attribute-guided Attention
   415	Maps (AAM) highlighting each attribute’s contribution to pairwise distance.
   416	
   417	Fig. 3. Explainability comparison between 2D and 3D person ReID. Top: 2D
   418	ReID shows query with rank-1 (incorrect, red) and rank-7 (correct, green)
   419	matches plus activation maps from ResNet-50 model. Bottom: Our 3D approach uses UVTexture mapping with texture, visibility masks, and activation
   420	maps generated by our fusion module showing high (red) and low (blue)
   421	contribution.
   422	
   423	Fig. 5. Comparison of person images (top) and UVTextures (bottom).
   424	Green/red borders indicate same/different persons. Attention maps show
   425	misalignment on images but clear alignment on UVTextures. Demonstrates
   426	UVTexture effectiveness for attribute analysis.
   427	
   428	viewpoint, eliminating background bias and providing clear visualization of which body parts (e.g., upper torso patterns, clothing designs)
   429	contribute to identity matching. This spatial consistency is particularly
   430	crucial for cross-view scenarios such as aerial–ground matching, where
   431	traditional 2D explanations fail due to extreme geometric deformations.
   432	A key innovation of our approach is the integration of visible clothing masks with activation maps to enable occlusion-aware matching.
   433	As shown in Fig. 3, the visibility masks (yellow regions) indicate which
   434	parts of the person are visible in the original image, while the activation
   435	maps reveal where the model focuses its attention. By computing the
   436	Intersection-Alignment Score (IAS) between these two components, we
   437	quantify how well the model concentrates on visible, discriminative regions rather than occluded areas. This explainability-driven refinement
   438	creates a feedback loop where interpretability directly improves performance — when the model inappropriately focuses on occluded regions
   439	(low IAS), the explainable re-ranking mechanism adjusts the similarity
   440	scores to discourage such matches. In other words, if the visualization
   441	reveals that the model is placing more emphasis on occluded parts, it suggests
   442	that the model may not be effectively learning discriminative features from
   443	the visible regions.
   444	The proposed UVTexture-based visualization technique transforms
   445	explainability from a post-hoc analysis tool into an integral component of the ReID pipeline. By providing a background-free, viewpointinvariant representation with consistent spatial correspondence, our
   446	method enables researchers to gain deeper insights into model behavior and make informed decisions to optimize performance. The
   447	high-resolution nature of UVTextures preserves fine-grained details
   448	while maintaining global structure, enabling multi-scale explainability
   449	that captures both subtle texture variations and overall appearance
   450	patterns. This comprehensive approach not only improves the accuracy
   451	of 3D person ReID but also provides interpretable results that build trust
   452	and understanding in the model’s decision-making process.
   453	To further investigate the explainability of our approach, we explore
   454	the use of attribute attention maps to demonstrate the consistency
   455	
   456	and alignment of UVTexture attention maps compared to normal 2D
   457	images. Fig. 4 provides an overview of the attribute-guided attention
   458	map generation process. Fig. 5 presents a side-by-side comparison of
   459	different individuals and their corresponding UVTextures, revealing
   460	that the attention maps in the image space are misaligned, while the
   461	UVTextures exhibit better alignment, particularly in terms of clothing
   462	details and gender-specific features. Similarly, Fig. 5 also showcases
   463	a comparison between a single individual and their UVTexture, highlighting the misalignment of attention maps in the image space and the
   464	improved alignment in the UVTexture, especially in capturing clothing
   465	specifics and gender characteristics. These observations underscore
   466	the effectiveness of our approach in providing more explainable and
   467	interpretable attention maps by leveraging the UVTexture and utilizing
   468	attribute attention maps to showcase the consistency and alignment of
   469	UVTexture attention maps compared to normal 2D images.
   470	Expanding on the insights gained from the UVTexture and attribute attention maps, we now shift our focus to the explainable
   471	re-ranking component, which further enhances the explainability and
   472	performance of 3D person ReID. The proposed re-ranking method
   473	leverages the alignment of low-level activation patterns to improve the
   474	ReID accuracy while maintaining interpretability. By focusing on these
   475	fine-grained patterns, the re-ranking component can effectively capture
   476	the subtle differences between individuals, leading to more accurate
   477	and explainable ReID results.
   478	The explainable re-ranking component leverages two key scores: the
   479	Intersection over Union (IoU) score and the activation alignment score.
   480	The IoU score measures how focused the activation map is within the
   481	visible clothing area mask. It is calculated using the following equation:
   482	𝐼𝑜𝑈 =
   483	5
   484	
   485	𝐴𝑚 ∩ 𝑀
   486	,
   487	𝐴𝑚 ∪ 𝑀
   488	
   489	(1)
   490	
   491	H. Nguyen, K. Nguyen, A. Pemasiri et al.
   492	
   493	Computer Vision and Image Understanding 261 (2025) 104517
   494	
   495	Table 2
   496	Person ReID datasets used for evaluation. ∗ indicates attributes are automatically labeled using off-the-shelf vision models (e.g.,
   497	CLIP (Radford et al., 2021)) following the 15 attribute categories defined in AG-ReID (Nguyen et al., 2023).
   498	Datasets
   499	
   500	Type
   501	
   502	Market-1501 (Zheng et al., 2015)
   503	DukeMTMC-reID (Zheng et al., 2017)
   504	MSMT-17 (Wei et al., 2018)
   505	PRAI-1581 (Zhang et al., 2019b)
   506	UAV–Human (Li et al., 2021b)
   507	AG-ReID.v1 (Nguyen et al., 2023)
   508	AG-ReID.v2 (Nguyen et al., 2024)
   509	
   510	Ground
   511	Ground
   512	Ground
   513	Aerial
   514	Aerial
   515	Aerial–Ground
   516	Aerial–Ground
   517	
   518	Cameras
   519	
   520	6
   521	8
   522	15
   523	2
   524	1
   525	2
   526	3
   527	
   528	26
   529	23
   530	15∗
   531	15∗
   532	7
   533	15
   534	15
   535	
   536	where 𝐴𝑚 represents the activation map and 𝑀 represents the mask.
   537	By setting a threshold for the IoU score, we classify the activation map
   538	as ‘‘focused’’ or ‘‘unfocused’’. If the IoU score is above the threshold,
   539	indicating a focused activation map, we encourage the similarity by
   540	reducing the distance in the distance matrix. Conversely, if the IoU
   541	score is below the threshold, suggesting an unfocused activation map,
   542	we discourage the similarity by increasing the distance.
   543	In parallel, we compute the activation alignment score, which
   544	measures the similarity between the activation patterns of the query
   545	and gallery images. It is calculated by computing the cosine similarity
   546	between the flattened activation maps of the query and gallery images:
   547	𝑆𝑎 (𝑖, 𝑗) =
   548	
   549	𝐴𝑞 (𝑖) ⋅ 𝐴𝑔 (𝑗)
   550	‖𝐴𝑞 (𝑖)‖‖𝐴𝑔 (𝑗)‖
   551	
   552	,
   553	
   554	(2)
   555	
   556	if 𝐼𝑜𝑈𝑞 (𝑖) > 𝜏𝑞 and
   557	𝐼𝑜𝑈𝑔 (𝑗) > 𝜏𝑔 and
   558	𝑆𝑎 (𝑖, 𝑗) > 𝜏𝑎
   559	
   560	Training
   561	
   562	Testing
   563	
   564	IDs
   565	
   566	Images
   567	
   568	IDs
   569	
   570	Images
   571	
   572	751
   573	702
   574	1041
   575	782
   576	619
   577	199
   578	807
   579	
   580	12,936
   581	16,522
   582	32,621
   583	19,523
   584	11,805
   585	11,554
   586	51,530
   587	
   588	750
   589	1110
   590	3060
   591	799
   592	525
   593	189
   594	808
   595	
   596	19,732
   597	17,661
   598	11,659
   599	19,938
   600	29,485
   601	12,464
   602	48,972
   603	
   604	DukeMTMC-reID (Zheng et al., 2017), and MSMT-17 (Wei et al.,
   605	2018)) captured by fixed CCTV cameras below 10 m; aerial–aerial
   606	datasets (PRAI-1581 (Zhang et al., 2019b) and UAV–Human (Li et al.,
   607	2021b)) featuring drone-captured imagery from various altitudes; and
   608	aerial–ground datasets (AG-ReID.v1 (Nguyen et al., 2023) and AGReID.v2 (Nguyen et al., 2024)) that combine multiple viewing perspectives. Notably, AG-ReID.v2 introduces a triple-platform approach by
   609	integrating CCTV, UAV, and wearable device perspectives, making it
   610	particularly valuable for evaluating cross-view person ReID capabilities
   611	across different viewing angles and capturing conditions.
   612	Market-1501: Zheng et al. (2015), introduced in 2015, is a widelyused dataset for person ReID tasks. It consists of 32,668 images capturing 1501 unique individuals. The dataset is divided into a training
   613	set with 12,936 images of 751 identities and a test set containing 3368
   614	query images and 19,732 gallery images of 750 identities. Additionally, 26 attributes labeled by Lin et al. (2019) are available for each
   615	individual.
   616	DukeMTMC-reID: Zheng et al. (2017), released in 2017, is another
   617	prominent dataset in the person ReID domain. It comprises 36,411
   618	images of 1404 individuals, captured by 8 different cameras. The
   619	dataset is split into a training set with 16,522 images of 702 identities
   620	and a test set with 2228 unique query images and 17,661 gallery images
   621	representing 702 identities. The DukeMTMC-reID dataset contains 23
   622	attributes for each individual.
   623	MSMT-17: Wei et al. (2018), published in 2017, is the largest
   624	image-based person ReID dataset, featuring data from 15 cameras in
   625	both indoor and outdoor scenarios. The detection boxes for each pedestrian are generated using Faster RCNN. The dataset comprises 126,441
   626	images representing 4101 identities, with the training set containing
   627	32,621 images of 1041 identities and the test set including 11,659
   628	images.
   629	PRAI-1581: Zhang et al. (2019b), released in 2019, is an aerialbased person ReID dataset containing 39,461 images of 1581 individuals. The images were captured by two UAVs flying at altitudes between
   630	20 to 60 m, making it one of the first large-scale aerial person ReID
   631	datasets.
   632	UAV–Human: Li et al. (2021b), introduced in 2021, is a comprehensive aerial dataset consisting of 41,290 images of 1144 individuals.
   633	The dataset was captured using a drone flying at low altitudes between
   634	2 to 8 m across various locations and time periods. It includes 7
   635	different attributes for each person and features diverse backgrounds
   636	and occlusion scenarios.
   637	AG-ReID.v1: Nguyen et al. (2023) is the first aerial–ground person
   638	ReID dataset that combines both UAV and CCTV camera perspectives.
   639	It contains 21,983 images of 388 unique individuals, captured from
   640	heights of 15 to 45 m. The dataset includes 15 distinct attributes for
   641	each person and features mixed camera views from different platforms.
   642	AG-ReID.v2: Nguyen et al. (2024) is a large-scale aerial–ground
   643	dataset containing 100,502 images of 1615 unique individuals. The
   644	images were captured by CCTV, UAVs, and Wearable devices from
   645	heights ranging from 15 to 45 m to tackle the matching across aerial
   646	
   647	where 𝑆𝑎 (𝑖, 𝑗) is the activation alignment score for the query–gallery
   648	pair, 𝐴𝑞 (𝑖) and 𝐴𝑔 (𝑗) are the flattened activation maps of the query
   649	and gallery images, respectively, and ‖ ⋅ ‖ denotes the Euclidean norm.
   650	A higher alignment score indicates a stronger similarity between the
   651	activation patterns, suggesting a better match.
   652	By incorporating both the IoU score and the activation alignment
   653	score into the re-ranking process, we refine the distance matrix to
   654	prioritize matches that satisfy two criteria: (1) the activation map
   655	concentrates on the relevant clothing regions, and (2) the activation
   656	patterns exhibit strong similarity between the query and gallery images.
   657	The distance matrix is refined using the following equation:
   658	⎧𝐷 (𝑖, 𝑗) ∗ 𝛼,
   659	⎪ 𝑜
   660	⎪
   661	𝐷𝑟 (𝑖, 𝑗) = ⎨
   662	⎪
   663	⎪𝐷 (𝑖, 𝑗) ∗ 𝛽,
   664	⎩ 𝑜
   665	
   666	Attributes
   667	
   668	(3)
   669	
   670	otherwise
   671	
   672	where 𝐷𝑟 (𝑖, 𝑗) is the refined distance matrix, 𝐷𝑜 (𝑖, 𝑗) is the original distance matrix, 𝐼𝑜𝑈𝑞 (𝑖) and 𝐼𝑜𝑈𝑔 (𝑗) are the IoU scores for the query and
   673	gallery images, respectively, 𝑆𝑎 (𝑖, 𝑗) is the activation alignment score for
   674	the query–gallery pair, 𝜏𝑞 , 𝜏𝑔 , and 𝜏𝑎 are the threshold values for the
   675	query IoU, gallery IoU, and activation alignment, respectively, and 𝛼
   676	and 𝛽 are the encouragement and discouragement factors, respectively.
   677	The proposed explainable re-ranking component enhances the 3D
   678	person ReID pipeline by incorporating additional information from the
   679	IoU scores and activation alignment. By encouraging matches with high
   680	IoU scores and activation alignment while discouraging those with low
   681	scores, the refined distance matrix is expected to improve the overall
   682	ReID performance. The effectiveness of this component will be further
   683	demonstrated through qualitative examples and experimental results in
   684	Section 4.4.
   685	4. Experimental results and discussions
   686	4.1. Datasets and evaluation metrics
   687	We conducted a comprehensive comparative analysis of FusionTexReIDNet across several commonly used large-scale benchmark
   688	datasets, as presented in Table 2. These datasets span three distinct
   689	categories: ground–ground datasets (Market-1501 (Zheng et al., 2015),
   690	6
   691	
   692	H. Nguyen, K. Nguyen, A. Pemasiri et al.
   693	
   694	Computer Vision and Image Understanding 261 (2025) 104517
   695	
   696	and ground cameras (Nguyen et al., 2024, 2023). Each individual has
   697	15 distinct attributes annotated. The dataset is split into a training set
   698	with 51,530 images of 807 identities and a testing set with 48,972
   699	images of the remaining 808 identities.
   700	Evaluation Metrics: We employ two widely used evaluation metrics for person ReID: rank-k accuracy of cumulative matching features
   701	(CMC) (Wang et al., 2007) and mean average precision (mAP) (Zheng
   702	et al., 2015). Rank-k represents the probability that the true match
   703	appears in the top-k retrieval results, with Rank-1 being the most
   704	commonly reported. On the other hand, mAP is a more comprehensive
   705	metric that considers both precision and recall, reflecting the average
   706	performance across all queries. It is calculated as the mean of the
   707	average precision (AP) for each query, where AP is the area under the
   708	Precision–Recall curve. Additionally, we report the number of model
   709	parameters (#params) to provide insights into the model complexity.
   710	
   711	different input modalities and datasets. In the image-based category,
   712	existing methods show a clear progression from lightweight architectures to transformer-based models. While lightweight networks like
   713	ShuffleNetV2 and MobileNetV2 achieve moderate results (e.g., ShuffleNetV2 with CE+Circle: 79.8% Rank-1 on Market-1501), recent advanced architectures like DC-Former and TransReID demonstrate strong
   714	performance (DC-Former: 96.0% Rank-1, 90.5% mAP on Market-1501)
   715	across ground–ground, aerial–aerial, and aerial–ground scenarios.
   716	For point cloud-based approaches using HMR reconstruction (pc∗ ),
   717	baseline architectures like DGCNN and PointNet++ show limited performance (DGCNN: 28.9% Rank-1 on Market-1501). More recent methods like PointReIDNet achieve better results (90.6% Rank-1, 75.3%
   718	mAP on Market-1501). Our proposed FusionTexReIDNet with pc∗ input
   719	achieves 91.2% Rank-1 and 76.9% mAP on Market-1501, demonstrating competitive performance in this category.
   720	When using Texformer-based point clouds (pc† ), our FusionTexReIDNet architecture shows substantial improvements. The baseline configuration achieves 92.9% Rank-1 and 79.9% mAP on Market-1501,
   721	significantly outperforming OG-Net Deep’s 85.2% Rank-1 and 66.7%
   722	mAP. This represents an improvement of 7.7% in Rank-1 accuracy
   723	and 13.2% in mAP, validating the effectiveness of our dual-stream
   724	architecture and UVTexture representation.
   725	Impact of Optimization Techniques. As shown in Table 4, various optimization strategies can significantly enhance the baseline performance. Our framework incorporates three key optimization techniques: KNN re-ranking (1 ), attribute-based re-ranking (2 ), and optimal weight fusion (3 ). When all optimization techniques are applied
   726	(FusionTexReIDNet1,2,3 with pc† ), our method achieves 98.5% Rank-1
   727	and 93.5% mAP on Market-1501, representing substantial improvements over the baseline configuration. The optimization techniques
   728	demonstrate consistent improvements across all datasets, with our
   729	FusionTexReIDNet1,2,3 achieving 89.7% Rank-1 accuracy on the challenging AG-ReID.v2 dataset. The superior performance validates our
   730	core hypothesis that 3D person ReID can surpass 2D methods through
   731	explicit viewpoint normalization and robust texture representation,
   732	where our UVTexture-based framework provides viewpoint-invariant
   733	explanations with spatially consistent activation maps across different
   734	camera angles. Additional experimental comparisons with recent stateof-the-art methods across multiple datasets can be found in Table A.8
   735	in the Appendix C.
   736	Performance on Aerial–Ground Scenarios. Our proposed FusionTexReIDNet demonstrates exceptional performance in challenging
   737	aerial–ground scenarios, which represent the most difficult cross-view
   738	matching tasks in person ReID. On AG-ReID.v1, our method achieves
   739	90.5% Rank-1 and 86.6% mAP, while on the larger AG-ReID.v2 dataset,
   740	FusionTexReIDNet attains 89.7% Rank-1 and 87.0% mAP. These results
   741	significantly outperform state-of-the-art 2D methods, with improvements of 16.2% and 26.9% respectively on AG-ReID.v2 compared
   742	to DC-Former. The substantial performance gains can be attributed
   743	to our viewpoint-invariant UVTexture representation that maintains
   744	spatial consistency across extreme viewing angles, enabling reliable
   745	feature correspondence between aerial top-down and ground lateral
   746	perspectives where traditional 2D approaches fail due to geometric
   747	deformations and appearance variations caused by drastic viewpoint
   748	differences.
   749	Qualitative Results Fig. 6 illustrates the performance of FusionTexReIDNet in successful cases, where the original 2D query image,
   750	its corresponding 3D rendered, and the top-10 retrieved candidates
   751	from the Market1501 dataset are presented. The model demonstrates
   752	its ability to retrieve true matches from various viewpoints, leveraging
   753	the 3D human reconstruction effectively. More results on other datasets
   754	can be found in Appendix C.
   755	To further understand the performance of FusionTexReIDNet in
   756	challenging scenarios, we selected and visualized the queries with
   757	
   758	4.2. Implementation details
   759	Hardware settings. The experiments were conducted on a Linuxbased operating system with an x86_64 architecture. High-performance
   760	computing (HPC) nodes equipped with NVIDIA A100 GPUs were utilized to perform the computational tasks. The software stack consisted

codex
第四、第五篇的方法结构已经足够清楚，但属性引导那篇在正文中段被输出截断，我会补读伪标签生成到双引导损失的完整段落。3D 纹理那篇也要补读双流网络和解释模块的中间段。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification.pdf' - | nl -ba | sed -n '228,397p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   228	3.2.3
   229	
   230	Our approach is built upon a dual-encoder architecture designed to
   231	process visual and textual information. Specifically, i ∈ {1...B}
   232	denotes the index within a batch. Given the image batch M =
   233	{m1 , m2, ..., mB } and their text descriptions P = {p1 , p2, ..., pB },
   234	an image encoder EM maps images to 512-d features fiM =
   235	EM (mi ). For text prompts pi , each prompt is first tokenized (e.g.,
   236	using byte pair encoding) and then embedded into 512-d word tokens
   237	Ti via an embedding layer Embd. These tokens are subsequently processed by a text encoder ET to produce the final text feature fiT .
   238	
   239	(4)
   240	
   241	Attribute Pseudo-label Generation
   242	
   243	Inspired by the positive impact of fine-grained attributes in ReID, we
   244	utilize the trainable attribute prompt structure defined above. Unlike
   245	holistic prompts used in some prior work [10], our attribute prompts
   246	feature learnable tokens dispersed throughout the text, each intended
   247	to capture a specific fine-grained feature. This design is critical for
   248	recognizing subtle differences, especially under occlusion, setting
   249	the stage for effective pseudo-label generation as described next.
   250	In our framework, attribute pseudo-labels refer to learnable token
   251	representations automatically derived from CLIP and CoOp, requiring no manual annotation. These pseudo-labels capture fine-grained
   252	visible attribute features (e.g., hair style, clothing color, accessories)
   253	and serve as auxiliary supervision signals to guide the image encoder towards learning more discriminative attribute-aware features.
   254	Unlike traditional labels that require human annotation, our pseudolabels are semantically trained to describe the collective attribute
   255	characteristics of all images belonging to the same identity.
   256	Our selection of attributes is informed by prior work demonstrating their effectiveness in person ReID. Specifically, we adopt key attribute categories identified as discriminative in ATPM [26], encompassing aspects like gender, hairstyle, posture, and clothing characteristics. Furthermore, the designed prompt template module offers
   257	inherent flexibility; it facilitates the straightforward incorporation of
   258	alternative attribute sets or descriptive contexts, such as explicitly
   259	
   260	Figure 2. Overview of the AG-ReID framework’s two-stage training. Stage 1 (Cross-modal Alignment): Using frozen Image/Text Encoders, learnable
   261	tokens within attribute prompts are trained via contrastive loss (Lfeat ) to generate attribute pseudo-labels. Simultaneously, a trainable Attribute Encoder learns
   262	to predict attribute tokens aligned with these pseudo-labels via LattrA . Stage 2 (Dual Guidance): The Image Encoder is trained with guidance from both
   263	attribute-prompt text features (LCE ) and the generated attribute pseudo-labels (LattrG ), leveraging a fine-tuned Attribute Encoder. Standard ReID losses
   264	(Lid , Ltri ) are also applied to the image features. The legend indicates module status (frozen/trainable) and token types.
   265	
   266	mentioning potential occlusion, through configuration adjustments
   267	without requiring modifications to the underlying code structure.
   268	Given a ReID dataset batch of n images M = {m1 , m2 , ..., mn },
   269	representing K unique identities, let P = {p1 , p2 , ..., pK } be the
   270	corresponding attribute prompts for each identity. For an image mi
   271	associated with identity k, its corresponding attribute prompt pk contains r learnable tokens, as defined previously. We denote this set of
   272	learnable tokens as Vi = {[v]k1 , [v]k2 , ..., [v]kr }.
   273	In the alignment stage of training, the text features fT of attribute
   274	prompt are aligned with the image features fM by SupConLoss [8],
   275	which is improved on top of the cross-entropy loss function for the
   276	supervised contrastive learning task.
   277	Lfeat = SupCon(fM , fT ) + SupCon(fT , fM )
   278	
   279	(6)
   280	
   281	In this way, not only the text feature fT gets the semantics corresponding to the image, but also each learnable token [v] gets the
   282	semantics of its corresponding attribute. At this time, we will call the
   283	learnable attribute token set V as semantics attribute pseudo-label.
   284	To use attribute pseudo-labels for guiding image features during
   285	the following stage, we propose the attribute encoder EA with 4
   286	i
   287	self-attention layers to align image feature fM
   288	with attribute pseudolabels Vi , which predicts attributes set Ai by image features.
   289	i
   290	Ai = [ai1 , ai2 , ..., air ] = EA (fM
   291	)
   292	
   293	The attribute align loss LattrA in this stage is formulated as,
   294	
   295	(7)
   296	
   297	n
   298	
   299	LattrA = −
   300	
   301	r
   302	
   303	1 XX
   304	cos(aij , vji )
   305	r i=1 j=1
   306	
   307	(8)
   308	
   309	The loss function in the cross-modal alignment stage Lalign consists of Lfeat and LattrA , and λ is a hyper-parameter for adjusting loss
   310	weights.
   311	Lalign = Lfeat + λLattrA
   312	
   313	(9)
   314	
   315	3.3 Dual Guidance Stage
   316	Building on the cross-modal alignment stage, we obtain text features
   317	encapsulating the holistic semantic information of the text prompt,
   318	the attribute pseudo-label denoting fine-grained attribute information, and an attribute encoder designed to align image features with
   319	their respective attribute features. At this stage, the integration of
   320	both holistic and fine-grained information guides the feature extraction process for images, which is why it is referred to as the dual
   321	guidance stage.
   322	The former computes the loss using the cross entropy loss function
   323	LCE , which is frequently employed in contrastive learning, whereas
   324	the latter is optimized by the improved previously mentioned attributes alignment loss LattrG .
   325	LCE = CrossEntropy(fM , fT )
   326	
   327	(10)
   328	
   329	Additionally, the ID loss Lid and triplet loss Ltri , commonly employed in ReID tasks, are also combined.
   330	
   331	Lid = −
   332	
   333	n
   334	X
   335	
   336	qi log(pi )
   337	
   338	Dataset
   339	Occ-Duke
   340	Occ-ReID
   341	P-Duke
   342	MSMT17
   343	Market-1501
   344	DukeMTMC
   345	
   346	(11)
   347	
   348	i=1
   349	
   350	Ltri = max(dp − dn + α, 0)
   351	
   352	(12)
   353	
   354	where qi denotes the value in the target distribution, and pi represents ID prediction logits of class i, dp and dn are feature distances
   355	of positive pair and negative pair, while α is the margin of triple loss.
   356	The idea of fine-tuning image feature encoder directly using
   357	pseudo-labels is expected to yield good results in holistic datasets,
   358	and this is confirmed in subsequent experiments.
   359	However, it is crucial to recognize that images with the same ID
   360	do not always share consistent semantics. In other words, attribute
   361	pseudo-labels reflect the collective attribute semantics of all images
   362	with a common ID, which may cause some tokens a predicted by the
   363	attribute encoder to mismatch their pseudo-labels v in the occluded
   364	case. Such mismatched pairs are referred to as “noise”. This issue
   365	arises when parts of some images differ from the majority within the
   366	same ID group. For instance, if the lower half of a person is visible
   367	in most images with ID i, but obscured in others, the corresponding
   368	pseudo-label v will predominantly contain the semantic information
   369	of the visible parts. This leads to the token a at the corresponding
   370	positions in the occluded image having semantic discrepancies, potentially misleading the feature extraction process.
   371	To address this issue, we suggest a slight modification to the attribute alignment loss for occluded ReID datasets, termed the noisemask method. In this approach, only pairs with a similarity above the
   372	threshold γ will be considered for loss calculation, while the rest will
   373	be masked. For holistic datasets, γ is assigned a value of −1, which
   374	means that loss will be computed for every pair.
   375	(
   376	1 if cos(aij , tij ) > γ
   377	D(i, j) =
   378	(13)
   379	0 otherwise.
   380	Thus, the attribute pseudo-label guidance loss LattrG is calculated
   381	as follows,
   382	n
   383	
   384	LattrG = −
   385	
   386	r
   387	
   388	1 XX
   389	D(i, j) · cos(aij , vji )
   390	r i=1 j=1
   391	
   392	(14)
   393	
   394	The final dual guidance stage loss is formulated as,
   395	Lguide = Lid + Ltri + LCE + βLattrG
   396	
   397	Experiments

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '943,1020p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   943	4.5
   944	
   945	Visualization
   946	
   947	To investigate the detail information extraction capabilities
   948	of the proposed DFE, we visualize the comprehensive and
   949	detailed features of several identities produced by BDLF. As
   950	illustrated in Figure 5, a comparison of the images of comprehensive and detailed features reveals that the attention regions
   951	of the comprehensive features is broader and more dispersed
   952	than that of the detailed features. This observation indicates
   953	that the DFE module has the capacity to focus on subtly distinguishable characteristics.
   954	
   955	5
   956	
   957	CONCLUSION
   958	
   959	In this paper, we propose a novel base-detail feature learning
   960	framework(BDLF) that learns detail and base features from
   961	a correlation and mutual information maximization for the
   962	VI-ReID task. The proposed BDLF consists of a DFE module and a BEG block. The DFE module non-destructively
   963	extracts detail information, while the BEG block generates
   964	base features by eliminating detail information from the output of the backbone network. By imposing constraints of
   965	independence and correlation on the detail and base embeddings, the proposed BDLF can capture detail and base features that retain as much modality-specific and shared information as possible, thereby effectively leveraging the differentiated minutiae. Extensive experiments on the SYSUMM01, RegDB, and LLCM datasets have demonstrated the
   966	superiority of BDLF.
   967	
   968	Acknowledgments
   969	This work was partially supported by the Shenzhen Science
   970	and Technology Program (KJZD20230923114600002), and
   971	the Guangdong Major Project of Basic and Applied Basic Research (2023B0303000010).
   972	
   973	References
   974	[Alehdaghi et al., 2022] Mahdi Alehdaghi, Arthur Josi,
   975	Rafael M. O. Cruz, and Eric Granger. Visible-infrared
   976	person re-identification using privileged intermediate information. In Leonid Karlinsky, Tomer Michaeli, and
   977	Ko Nishino, editors, Computer Vision - ECCV 2022 Workshops - Tel Aviv, Israel, October 23-27, 2022, Proceedings,
   978	Part V, volume 13805 of Lecture Notes in Computer Science, pages 720–737. Springer, 2022.
   979	[Cao et al., 2023] Chengzhi Cao, Xueyang Fu, Hongjian
   980	Liu, Yukun Huang, Kunyu Wang, Jiebo Luo, and ZhengJun Zha. Event-guided person re-identification via sparsedense complementary learning. In IEEE/CVF Conference
   981	on Computer Vision and Pattern Recognition, CVPR 2023,
   982	Vancouver, BC, Canada, June 17-24, 2023, pages 17990–
   983	17999. IEEE, 2023.
   984	[Chen et al., 2022] Cuiqun Chen, Mang Ye, Meibin Qi,
   985	Jingjing Wu, Jianguo Jiang, and Chia-Wen Lin. Structureaware positional transformer for visible-infrared person reidentification. IEEE Transactions on Image Processing,
   986	31:2352–2364, 2022.
   987	[Dinh et al., 2017] Laurent Dinh, Jascha Sohl-Dickstein, and
   988	Samy Bengio. Density estimation using real NVP. In 5th
   989	International Conference on Learning Representations,
   990	ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017.
   991	[Feng et al., 2023] Jiawei Feng, Ancong Wu, and Wei-Shi
   992	Zheng. Shape-erased feature learning for visible-infrared
   993	person re-identification. In IEEE/CVF Conference on
   994	Computer Vision and Pattern Recognition, CVPR 2023,
   995	Vancouver, BC, Canada, June 17-24, 2023, pages 22752–
   996	22761. IEEE, 2023.
   997	[Ge et al., 2020] Yixiao Ge, Dapeng Chen, and Hongsheng
   998	Li. Mutual mean-teaching: Pseudo label refinery for unsupervised domain adaptation on person re-identification.
   999	In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30,
  1000	2020. OpenReview.net, 2020.
  1001	[Goodfellow et al., 2014] Ian J. Goodfellow, Jean PougetAbadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
  1002	Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio.
  1003	Generative adversarial nets. In Zoubin Ghahramani, Max
  1004	Welling, Corinna Cortes, Neil D. Lawrence, and Kilian Q.
  1005	Weinberger, editors, Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems 2014, December 8-13 2014,
  1006	Montreal, Quebec, Canada, pages 2672–2680, 2014.
  1007	[He et al., 2016a] Kaiming He, Xiangyu Zhang, Shaoqing
  1008	Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and
  1009	
  1010	Pattern Recognition, CVPR 2016, Las Vegas, NV, USA,
  1011	June 27-30, 2016, pages 770–778. IEEE Computer Society, 2016.
  1012	[He et al., 2016b] Kaiming He, Xiangyu Zhang, Shaoqing
  1013	Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and
  1014	Pattern Recognition, CVPR 2016, Las Vegas, NV, USA,
  1015	June 27-30, 2016, pages 770–778. IEEE Computer Society, 2016.
  1016	[Hermans et al., 2017] Alexander Hermans, Lucas Beyer,
  1017	and Bastian Leibe. In defense of the triplet loss for person re-identification. CoRR, abs/1703.07737, 2017.
  1018	[Li et al., 2020] Diangang Li, Xing Wei, Xiaopeng Hong,
  1019	and Yihong Gong. Infrared-visible cross-modal person reidentification with an X modality. In The Thirty-Fourth
  1020	AAAI Conference on Artificial Intelligence, AAAI 2020,

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Beyond geometry - The power of texture in interpretable 3D person ReID.pdf' - | nl -ba | sed -n '255,410p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   255	3. Methodology
   256	In this section, we will elaborate on the proposed FusionTexReIDNet
   257	framework and its key components in detail. First, we will introduce
   258	the preliminaries of UVTexture and 3D reconstruction methods in
   259	Section 3.1, which establish the foundational concepts essential for
   260	understanding our approach. Then, we will provide a comprehensive
   261	overview of the overall framework architecture in Section 3.2. Subsequently, we will detail the two main processing streams: the UVTextureemphasized ReID stream in Section 3.3 and the 3D point cloud processing stream in Section 3.4. Finally, we will present our novel explainable module that provides interpretability through UVTexture-based
   262	visualization in Section 3.5.
   263	3
   264	
   265	H. Nguyen, K. Nguyen, A. Pemasiri et al.
   266	
   267	Computer Vision and Image Understanding 261 (2025) 104517
   268	
   269	Table 1
   270	FusionTexReIDNet: Proposed method summary.
   271	Stream/Module
   272	
   273	Input data
   274	
   275	Techniques
   276	
   277	Key contributions
   278	
   279	Stream 1 - UVTextureNet
   280	
   281	Reconstructed UVTexture images
   282	
   283	ResNet50 backbone within SBS framework
   284	
   285	- Extracts discriminative texture features
   286	- Leverages subtle texture variations
   287	
   288	Stream 2 - 3DReIDNet
   289	
   290	3D point cloud with XYZ
   291	and RGB information
   292	
   293	Dynamic graph convolution
   294	Omni-scale Module
   295	
   296	- Processes unordered point cloud data
   297	- Extracts multi-scale features
   298	
   299	Fusion Module
   300	
   301	Distance matrices from both streams
   302	
   303	Weighted sum
   304	KNN and attribute-based re-ranking
   305	Explainable re-ranking
   306	
   307	- Combines 3D structural and texture information
   308	- Refines distance metric results
   309	- Refines results using IAS value
   310	
   311	Explainable Module
   312	
   313	Features from Stream 1, UVTexture model
   314	
   315	Activation map generation
   316	Actmap and visible clothing mask fusion
   317	
   318	- Visualizes relevant texture regions
   319	- Compute IAS value for explainable re-ranking
   320	
   321	The framework operates through a two-stage process that fundamentally differs from existing approaches in its treatment of viewpoint
   322	variations and texture representation. Unlike traditional 2D person
   323	ReID methods that suffer from background bias and viewpointdependent activation patterns, our approach leverages UVTexture’s
   324	viewpoint-invariant properties to provide consistent explanations
   325	across different camera angles. In the first stage, 2D pedestrian
   326	images are processed through 3D reconstruction networks to generate
   327	both geometric point cloud data and high-resolution UVTexture
   328	representations. This preprocessing addresses the critical challenge
   329	of viewpoint variations by projecting arbitrary camera views into
   330	canonical viewpoints, enabling consistent identity representation
   331	across different poses and perspectives while eliminating spurious
   332	background correlations that plague 2D methods.
   333	The second stage employs a dual-stream architecture that processes the reconstructed data in parallel, with Stream 1 (UVTextureNet)
   334	extracting discriminative appearance features from UVTexture maps
   335	and Stream 2 (3DReIDNet) processing geometric features from 3D
   336	point clouds. The explainable component distinguishes our approach by
   337	providing interpretable insights through UVTexture-based visualization
   338	that maintains spatial consistency across viewpoints, enabling precise
   339	feature localization and occlusion-aware matching. The IntersectionAlignment Score (IAS) quantifies how well the model focuses on visible clothing regions, creating a feedback loop where interpretability
   340	directly improves performance through explainable re-ranking. This
   341	comprehensive approach enables FusionTexReIDNet to achieve stateof-the-art performance while providing reliable interpretations that
   342	remain consistent across challenging aerial–ground scenarios where
   343	traditional 2D explanations fail due to extreme viewpoint differences.
   344	
   345	3.4. Stream 2 — 3DReIDNet
   346	While Stream 1 processes high-resolution appearance features from
   347	UVTexture maps, Stream 2 extracts geometric information from 3D
   348	point clouds using our 3DReIDNet architecture inspired by Zheng et al.
   349	(2020). Both our approach and prior 3D ReID methods use RGBenhanced point clouds (XYZ+RGB), but our dual-stream design strategically separates texture and geometry processing to enable domainspecific optimization before fusion. We adopt a Dynamic Graph Convolution (DGC) module that constructs a dynamic graph for feature 𝑓𝑖
   350	using the 𝑘-nearest neighbors (KNN) method.
   351	We also adopt the basic structure of the Omni-scale Module, which
   352	consists of dynamic graph convolution for local feature aggregation
   353	and three branches with different grouping rates to capture multi-scale
   354	patterns. Each branch includes a grouping layer, linear layers, batch
   355	normalization, a squeeze-excitation block, and a group max pooling
   356	layer. The outputs of the three branches are summed to accumulate
   357	different scale patterns. By incorporating the adapted Omni-scale Module into our network architecture, we aim to leverage the benefits of
   358	multi-scale information aggregation and improve the performance of
   359	our proposed method on point cloud processing tasks.
   360	While our point cloud processing module shares similarities with
   361	the work of Zheng et al. (2020), our overall network architecture
   362	introduces a key difference to enhance its performance and adaptability
   363	to our specific task. We extend their work by incorporating a texture
   364	stream that leverages UVTexture to extract discriminative features and
   365	enhance the network’s ability to capture subtle variations in appearance. This texture stream complements the point cloud stream, enabling
   366	our network to learn more comprehensive and robust representations.
   367	To fuse the information from both streams, we employ a weighted
   368	sum of the distance matrices obtained from the texture stream and the
   369	point cloud stream, allowing our network to adaptively balance their
   370	contributions for accurate and robust ReID, as detailed in Appendix A.
   371	
   372	3.3. Stream 1 — UVTextureNet: UVTexture-emphasized ReID
   373	In the first stream of our proposed method, we train a person ReID
   374	model using a ResNet50 backbone architecture (He et al., 2015) on UVTexture maps generated from the 3D reconstruction process described
   375	in Section 3.1. The UVTexture maps provide a 3D representation of
   376	a person’s appearance, independent of their pose or viewpoint in the
   377	original image. We reformulate the dataset into 𝑆 ′ = {(𝑢𝑖 , 𝑦𝑖 )}𝑁 , where
   378	𝑢𝑖 represents the UVTexture map corresponding to a person’s identity
   379	𝑦𝑖 . During training, the ResNet50 model takes a UVTexture map as
   380	input and learns to extract discriminative features for person ReID. This
   381	approach aims to utilize the localized textural features on the UV maps
   382	for ReID, enhancing the model’s ability to distinguish identities based
   383	on the surface texture information.
   384	As shown in Fig. 2, the trained network generates a corresponding
   385	feature map 𝐹𝐴 from the input UVTexture map. The feature map is
   386	then passed through a classifier to obtain a compact feature vector
   387	that encodes the person’s identity. During inference, we compute the
   388	distances 𝐷1 between the feature vectors of the query and gallery
   389	images using cosine distance. The resulting distance matrix, which
   390	captures the similarity between the query and gallery images in the
   391	UVTexture space, is utilized in the fusion module for the final ReID.
   392	
   393	3.5. Explainable module — UVTexture-inspired explainability
   394	Traditional 2D person ReID models suffer from significant interpretability challenges, particularly their susceptibility to background
   395	bias and viewpoint-dependent activation patterns. As demonstrated in
   396	the top row of Fig. 3, 2D ReID activation maps often highlight spurious
   397	background correlations rather than person-specific features, leading
   398	to incorrect matches. This viewpoint dependency makes it difficult
   399	to identify which features truly contribute to identity matching, as
   400	activation patterns vary inconsistently across different camera angles,
   401	preventing systematic analysis of discriminative regions.
   402	In contrast, our 3D UVTexture-based approach provides viewpointinvariant explanations that maintain spatial consistency across different
   403	camera perspectives. The bottom row of Fig. 3 illustrates how UVTexture representation enables precise localization of discriminative
   404	features on a canonical body representation. By projecting person appearance onto a normalized UV coordinate system, our method ensures
   405	that activation maps consistently align regardless of the original camera
   406	4
   407	
   408	H. Nguyen, K. Nguyen, A. Pemasiri et al.
   409	
   410	Computer Vision and Image Understanding 261 (2025) 104517

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Adversarial perturbation and defense for generalizable person re-identification.pdf' - | nl -ba | sed -n '1190,1290p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1190	Table 5. As shown in Table 5, the result of Baseline (Base.) on the
  1191	‘‘Original Testing Set (OST)’’ is reported in He et al. (2021). When
  1192	we replace the ‘‘Original Testing Set’’ with the ‘‘Hard Testing Set
  1193	(HST)’’ synthesized by MPG-Net, the performance of the Re-ID model
  1194	
  1195	4.3. Ablation study
  1196	In this subsection, we perform comprehensive ablation studies to
  1197	validate the effectiveness of each component in our proposed APD.
  1198	Contribution of Each Component in the APD. The experiment
  1199	results are documented in Table 4. The Baseline (Base.) structure is
  1200	described in Section 3.2.1. As reported in He et al. (2021), the Baseline achieved ∼ 94% Rank-1 score (see Table 5) on the Market1501
  1201	dataset in the supervised person Re-ID task. When the Baseline is
  1202	directly tested on other testing sets, including DukeMTMC, CUHK03Detected (CUHK03-Det.), and CUHK03-Labeled (CUHK03-Lab.), the
  1203	model performance dropped significantly. This means that the domain
  1204	generalization performance of the Baseline is bad. We added each of
  1205	the different components from APD into the Baseline step by step to
  1206	7
  1207	
  1208	Neural Networks 186 (2025) 107287
  1209	
  1210	H. Tan et al.
  1211	
  1212	Fig. 6. Original Images and Images synthesized by MPG-NET∗ and MPG-Net. MPG-NET∗ represents MPG-Net without the metric perturbation loss 𝐿𝑀 𝑃 Eq. (3). The MPG-Net
  1213	can effectively modify the appearance feature of persons. We also can clearly observe that the metric perturbation loss 𝐿𝑀 𝑃 Eq. (3) plays a very important role in this synthesis
  1214	process. Besides, the basic appearance features of persons are basically unchanged.
  1215	
  1216	Fig. 7. Original Images and Images synthesized by MPG-Net. We can see that the MPG-Net makes the appearance feature of different ID samples very similar, which indeed
  1217	increases the difficulty of sample matching.
  1218	
  1219	shows a cliff-like decline. This means that we have indeed modified the
  1220	metric relationship between samples, significantly increasing the difficulty of
  1221	sample matching. (ii) We further show the effectiveness of sample metric
  1222	perturbation strategy (i.e. remove the metric perturbation loss 𝐿𝑀 𝑃
  1223	Eq. (3) in the MPG-Net) on Re-ID model performance, which is named
  1224	as Base.+MPG-Net∗ . As shown in Table 4, compared to MPG-Net, when
  1225	we remove the metric perturbation loss 𝐿𝑀 𝑃 Eq. (3) in the MPG-Net,
  1226	the performance of the model decreases. That is, the performance of
  1227	Base.+MPG-Net∗ shown in Table 4 is worse than that of Base.+MPGNet. In all, the experimental result means that the metric perturbation
  1228	strategy in MPG-Net is useful for improving the domain generalization
  1229	of the Re-ID model. (iii) As shown in Table 4, the performance of the
  1230	Base.+MPG-Net∗ is also better than that of Baseline (Base.). In this case,
  1231	even if we do not introduce the metric perturbation loss 𝐿𝑀 𝑃 (Eq. (3)),
  1232	each sample will implicitly change the metrics between samples when
  1233	it is pushed towards its negative samples. Accordingly, the samples
  1234	generated by Base.+MPG-Net∗ can also be considered hard matching
  1235	samples compared to the original samples.
  1236	From Figs. 6 and 7, we can observe how the MPG-Net synthesizes
  1237	new samples with modified part feature. In Fig. 6, MPG-NET∗ represents MPG-Net without the metric perturbation loss 𝐿𝑀 𝑃 Eq. (3). We
  1238	can clearly see that the MPG-Net can effectively modify the appearance
  1239	feature of persons. Firstly, from Fig. 6, we can clearly observe that the
  1240	metric perturbation loss 𝐿𝑀 𝑃 Eq. (3) plays a very important role in this
  1241	process. Secondly, the basic appearance features of persons are basically unchanged. Finally, Fig. 7, we can see that the MPG-Net makes the
  1242	
  1243	appearance feature of different ID samples very similar, which indeed
  1244	increases the difficulty of sample matching. Specifically, (1) the basic
  1245	content in the person sample is unchanged; (2) the background/style
  1246	and the partial clothing color of the person are modified; and (3)
  1247	compared with original images, the clothing colors of person ID 𝑎.𝑏.𝑐
  1248	are similar in the synthesized images, and this implies a significantly
  1249	increased sample difficulty. Of course, Fig. 6 can also verify that the
  1250	diversity of samples can be guaranteed. Besides, in https://pan.baidu.
  1251	com/s/16hNErZVQaVkkTWpuonl1Cg and password: 4iyo, we further
  1252	show a large number of images generated by MPG-Net. The images
  1253	generated by MPG-Net still maintain diversity. Therefore, we do not
  1254	have to worry about the collapse of the MPG-Net model.
  1255	In all, MPG-Net can indeed increase the difficulty of sample matching, and can ensure that main content does not change. These hard
  1256	samples will encourage the SP-Net to learn and capture high-quality
  1257	person descriptors to meet the challenges from target domains.
  1258	Effectiveness Discussion about SP-Net. In this part, we discuss
  1259	the effectiveness of each strategy in SP-Net. The experimental results
  1260	are documented in Table 6. (i) The MSP is designed to quickly realize
  1261	the perception and interaction between features in a two-dimensional
  1262	structure, to further supplement the subsequent Transformer with some
  1263	structured contextual feature. So, we try to verify the effectiveness
  1264	of the Multi-scale Semantic Perception (MSP) Module. We introduce
  1265	the MSP module into Baseline (Base.), i.e. Base.+MSP. Compared with
  1266	Baseline (Base.), Base.+MSP gains better performance. This means that
  1267	8
  1268	
  1269	
web search: 
web search: "Adversarial flow-based generative models" "visible-to-Infrared" person re-Identification Flow2Flow
web search: 
web search: "Base-Detail Feature Learning Framework" "Visible-Infrared Person Re-Identification" BDLF
web search: 
web search: "Adversarial perturbation and defense for generalizable person re-identification" DOI
web search: 
web search: "Adversarial Perturbation and Defense" "Re-identification" Neural Networks 107287
web search: 
web search: https://doi.org/10.1016/j.neunet.2025.107287
web search: 
web search: 10.1016/j.neunet.2025.107287
web search: 
web search: Adversarial perturbation defense generalizable person re identification 107287
web search: 
web search: "Identity Diffuser" "Market1501" "DukeMTMC"
web search: 
web search: "FusionTexReIDNet"
web search: 
web search: "CLIP-ReID" person re-identification arxiv
web search: 
web search: "CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification"
web search: 
web search: "Learning to Prompt for Vision-Language Models" CoOp arxiv
codex
我还会补一下第三篇的实验和消融段，因为它的方法简单，但“证据闭环”主要靠生成质量、少数据收益和条件消融来支撑。这个部分决定它卖的是扩散模型本身，还是“身份条件生成”这个视角。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf' - | nl -ba | sed -n '328,670p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   328	We visually compare Identity Diffuser with the current stateof-the-art PIDM [9], as shown in Fig. 4. Each row depicts a
   329	different person synthesized with random poses. Our method
   330	produces more stable and consistent pedestrian images. We
   331	use the DDIM sampling method with 100 sampling steps,
   332	consistent with PIDM’s settings.
   333	Identity Diffuser
   334	
   335	PIDM
   336	
   337	(9)
   338	
   339	In the code, we set zero tensor for condition that is not used.
   340	ω1 = 0.1 and ω2 = 0.9 are the hyperparameters to control the
   341	weight of the conditions.
   342	b) Network Architecture: The components in our framework shown in Fig.3 are primarily based on Guided
   343	Diffusion[7], U-Net [16], and ResNet50 [17]. All input images are resized to a resolution of 128x64 pixels. A U-Net
   344	structured network maps the input image, conditions, and
   345	timestep embeddings to predict noise at specific timesteps.
   346	The pre-trained reid-net in the figure serves as the backbone of
   347	ResNet50, mapping inputs to a 2048-dimensional embedding
   348	vector. This net is freezed during training. The ID adapter
   349	consists of several bilinear upsampling and convolution layers,
   350	mapping the embedding vector back to the original image size
   351	with a channel number of N = 10. The pose is processed
   352	as a heatmap of the same size as the input image, with the
   353	number of channels equal to the keypoints number. The global
   354	
   355	Fig. 4. Visual comparison of Identity Diffuser and PIDM[9] for market1501
   356	generation
   357	
   358	Frechet Inception Distance (FID) are used to evaluate the
   359	realism of the generated images. FID measures how close the
   360	distribution of generated images is to the real. The FID of our
   361	method and other methods are shown in Table. I. Comparing
   362	our method with the current methods, the realism (FID) is
   363	superior. This finding suggests that our approach is capable of
   364	producing more authentic human images.
   365	C. Re-Identification Performance
   366	ReID accuracy, measured by mAP (mean Average Precision), is used to evaluate the impact of generated data on
   367	person reID tasks. A random test image and 20 poses from
   368	real training data are fed into the diffusion model to generate
   369	a synthetic training set Dsyn .
   370	We initialize a ResNet50 network with Dsyn and randomly
   371	select 20%, 40%, 60%, and 80% of the real re-identification
   372	
   373	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:51 UTC from IEEE Xplore. Restrictions apply.
   374	
   375	TABLE I
   376	C OMPARISON OF DIFFERENT GENERATIVE METHODS ON REALISM IN
   377	M ARKET 1501
   378	Methods
   379	
   380	PG2 -GAN [22]
   381	
   382	LSGAN [21]
   383	
   384	Realism (FID)
   385	Methods
   386	
   387	136.26
   388	
   389	151.16
   390	
   391	54.23
   392	
   393	FD-GAN [4]
   394	
   395	JDGL [24]
   396	
   397	PIDM [9]
   398	
   399	Ours
   400	
   401	257.00
   402	
   403	18.24
   404	
   405	14.45
   406	
   407	13.29
   408	
   409	Realism (FID)
   410	
   411	Market1501
   412	
   413	85
   414	
   415	PN-GAN [23]
   416	
   417	DukeMTMC
   418	
   419	75
   420	
   421	Identity diffuser
   422	UnrealPerson
   423	PIDM
   424	Baseline
   425	
   426	55
   427	
   428	40
   429	
   430	0.2
   431	
   432	0.4
   433	
   434	0.6
   435	Ratio of real data
   436	(a)
   437	
   438	0.8
   439	
   440	1.0
   441	
   442	mAp
   443	
   444	mAp
   445	
   446	66
   447	70
   448	
   449	57
   450	
   451	Identity diffuser
   452	UnrealPerson
   453	PIDM
   454	Baseline
   455	
   456	48
   457	39
   458	30
   459	
   460	0.2
   461	
   462	0.4
   463	
   464	0.6
   465	
   466	0.8
   467	
   468	TABLE II
   469	M A P PERFORMANCE WITH DIFFERENT REAL DATA PERCENTAGES VS .
   470	TRAINING EPOCHS OF OUR IDENTITY DIFFUSER
   471	
   472	Training epochs
   473	
   474	0.2
   475	
   476	0.4
   477	
   478	0.6
   479	
   480	0.8
   481	
   482	1.0
   483	
   484	100 epochs
   485	300 epochs
   486	
   487	76.81
   488	76.16
   489	
   490	75.13
   491	75.92
   492	
   493	78.83
   494	80.81
   495	
   496	83.14
   497	82.77
   498	
   499	83.40
   500	84.17
   501	
   502	We initialized ResNet50 with pre-trained weights from both
   503	embeddings and guided the diffusion model training. As
   504	shown in Table III, using ImageNet for identity embedding
   505	outperforms the current SOTA PIDM in mAP performance,
   506	indicating that our model enhances image feature extraction
   507	and maintains identity consistency.
   508	
   509	1.0
   510	
   511	Ratio of real data
   512	(b)
   513	
   514	Fig. 5. The re-identification mAP score on ResNet50 using images generated
   515	by our identity diffuser is assessed on Market1501 and DukeMTMC datasets.
   516	The yellow line represents the baseline mAP score using real data only. The
   517	x-axis represents the number of real data used in fine-tuning, while the y-axis
   518	represents the mAP score on the test query and gallery images.
   519	
   520	dataset, ensuring at least one image per ID, to simulate limited
   521	data scenarios. The sampled data Dtr is used to fine-tune the
   522	network.
   523	In Fig. 5(a), our method is compared with others on
   524	Market1501. With limited real data (e.g., 0.2), both PIDM and
   525	our method show significant improvement (27.13%/32.75%
   526	mAP), with our method outperforming PIDM by 5.62%. As
   527	real data increases, the benefit decreases, showing only a
   528	4.71% improvement at the 0.8 level, highlighting the greater
   529	effectiveness of real data over synthetic. Unrealperson shows
   530	only slight improvement due to its rough 3D modeling.
   531	In the DukeMTMC dataset (Fig. 5(b)), our method still
   532	outperforms PIDM, though the gap is smaller at 0.82% mAP
   533	at the 1.0 level. Our consistent performance across datasets
   534	is due to considering reID feature consistency. Further details
   535	are provided in the ablation study section III-D.
   536	D. Ablation Study
   537	We evaluate the influence of different network backbones,
   538	pre-train datasets and training strategies on our identity diffuser on Market1501 datasets.
   539	a) Different Epochs of Training Diffusion Modal: We
   540	monitored the visual quality of synthetic data at each epoch.
   541	After about 50 epochs, distinguishing between real and synthetic data becomes difficult for humans. However, as shown
   542	in Table II, performance on downstream tasks remains suboptimal if training is incomplete, despite good synthetic quality.
   543	This indicates that increasing training epochs is essential for
   544	optimizing synthetic data for downstream tasks.
   545	b) Imagenet Pretrain vs. Market1501 Pretrain: To assess
   546	the impact of re-identification feature embedding, we conducted two experiments: one with a general ImageNet embedding and another with a specific Market1501 embedding. We
   547	compared results to evaluate each embedding’s contribution.
   548	
   549	TABLE III
   550	I MAGENET PRETRAIN VS . MARKET 1501 PRETRAIN OF I DENTITY
   551	DIFFUSER
   552	
   553	Real data percentage
   554	
   555	0.2
   556	
   557	0.4
   558	
   559	0.6
   560	
   561	0.8
   562	
   563	1
   564	
   565	PIDM [9]
   566	ImageNet pretrain
   567	Market1501 pretrain
   568	
   569	70.54
   570	75.64
   571	76.16
   572	
   573	72.44
   574	76.39
   575	75.92
   576	
   577	78.66
   578	80.41
   579	80.80
   580	
   581	80.56
   582	82.50
   583	82.77
   584	
   585	82.44
   586	83.93
   587	84.17
   588	
   589	c) Different Backbones: The influence of various reidentification architectures is compared to address the research
   590	question: Does a larger model benefit more from synthetic
   591	data pretraining? As indicated in Table IV, two different reidentification models, ResNet50 and ResNet100 are compared
   592	during the fine-tuning task. In Resnet50 and Resnet100, the
   593	improvement of our method with synthetic data pretraining is
   594	2.54/1.44 separately. The observation suggests that synthetic
   595	data has a more substantial impact on smaller models than on
   596	larger ones. We attribute this to the varying representational
   597	capabilities of the models, where smaller models are more
   598	prone to underfitting and thus benefit more from synthetic data.
   599	TABLE IV
   600	T HE IMPACT OF DIFFERENT BACKBONES
   601	Backbones
   602	
   603	Percentage
   604	
   605	0.2
   606	
   607	0.4
   608	
   609	0.6
   610	
   611	0.8
   612	
   613	1
   614	
   615	Resnet100
   616	
   617	Baseline
   618	PIDM [9]
   619	Identity diffuser
   620	
   621	50.37
   622	78.16
   623	80.25
   624	
   625	62.69
   626	78.40
   627	81.68
   628	
   629	77.97
   630	83.44
   631	85.18
   632	
   633	83.29
   634	85.14
   635	86.17
   636	
   637	85.52
   638	86.08
   639	86.96
   640	
   641	Resnet50
   642	
   643	Baseline
   644	PIDM [9]
   645	Identity diffuser
   646	
   647	43.41
   648	70.54
   649	76.16
   650	
   651	61.29
   652	72.44
   653	75.92
   654	
   655	72.21
   656	78.66
   657	80.81
   658	
   659	78.06
   660	80.56
   661	82.77
   662	
   663	81.63
   664	82.44
   665	84.17
   666	
   667	IV. C ONCLUSION
   668	In this study, we propose a novel generative re-identification
   669	method called Identity Diffuser. We integrate a pre-trained
   670	re-identification network into a diffusion model to generate

web search: 
web search: https://arxiv.org/abs/2508.04998
web search: 
web search: https://doi.org/10.1109/ICASSP49660.2025.10887789
web search: 
web search: "10.1016/j.patcog.2025.112622"
web search: 
web search: "10.1016/j.patcog.2025.112622"
web search: 
web search: "Adversarial flow-based generative models for visible-to-Infrared person re-Identification"
web search: 
web search: "Learning to Prompt for Vision-Language Models" "Context Optimization"
web search: 
web search: "PointReIDNet"
web search: 
web search: "How Image Generation Helps" "Visible-Infrared Person ReID" "Flow"
web search: 
web search: "adversarial perturbation" "person re-identification"
web search: 
web search: "Shape-erased feature learning" "visible-infrared person re-identification"
web search: 
web search: "3DInvarReID" "person re-identification"
web search: 
web search: "InfinitePerson" "generalization person re-identification"
codex
我按你列出的 6 篇拆，不按标题里的“3”处理。方法正文来自本地 PDF 的前 8 页，联网主要核了相近先例和论文来源。

**1. Adversarial flow-based generative models for visible-to-Infrared person Re-Identification**

1. 触发观察：V2I ReID 同时缺数据、跨模态差异大。已有 GAN 翻译主要做“补目标模态”，但红外到可见光是信息少到信息多，天然难；而 SYSU-MM01、RegDB 这类数据又不够大。
2. 重定义动作：作者没有把问题讲成“再做一个翻译器”，而是讲成“可见光和红外都应该映射到同一个可逆高斯潜空间”。关键词是 shared Gaussian domain、bijective transformation、training sample expansion、cross-modal generation。
3. 机制怎么长出来：一旦有两个可逆流，可见光和红外都能进同一个潜空间，那么同模态扩增就是在潜空间插值后映回原模态，跨模态生成就是源模态进潜空间再从目标流出来。身份编码器和模态判别器是自然补上的约束，用来保证“像同一个人”和“像目标模态”。
4. 证据闭环：关键不是只报 SOTA，而是分别验证 training sample expansion 和 cross-modal generation 对多个基线都有效，并做生成图、潜空间插值、非线性层消融、统计检验。它在证明“同一个潜空间确实能同时支撑两种生成任务”。
5. reviewer 为什么买账：卖的是视角加框架，不是某个小层。真正 novelty 是把“数据少”和“模态缺口”统一成一个可逆潜空间操作。不过现在对我们不能新抄，生成式 VI-ReID、GAN 翻译、flow/latent bridge 这条线已经有明显先例。

**2. Adversarial perturbation and defense for generalizable person re-identification**

1. 触发观察：DG ReID 的常规说法是“源域覆盖不够”或“身份无关特征干扰”。作者抓到更具体的一点：在 hard matching samples 上，身份相关和无关特征纠缠，已有按类内类间距离拆特征的假设，在特征空间还没学好时并不稳。
2. 重定义动作：把 DG ReID 重讲成“模型要能面对度量关系被故意扰乱的困难样本，并能防御身份无关语义的攻击”。关键词是 metric adversariality、hard matching samples、adversarial perturbation and defense、semantic purification。
3. 机制怎么长出来：先用 MPG-Net 反过来做 metric learning，让生成样本靠近负样本、远离原样本，同时用浅层内容一致性保持人形和主体外观。再用 SP-Net 学防御，浅层拆出身份相关特征和身份无关特征，把打乱后的身份无关特征加回去攻击身份相关特征，强迫模型仍能分类正确。
4. 证据闭环：最关键的是 hard testing set 让普通基线出现断崖式下降，说明生成样本真的改变了匹配难度；再通过去掉 metric perturbation loss、生成图可视化、MPG-Net 与 SP-Net 逐步消融，证明不是普通 GAN 增广。
5. reviewer 为什么买账：卖的是“用反度量困难样本训练泛化能力”的视角。机制有些堆，但 perturbation 到 defense 的逻辑很顺。需要注意，ReID 对抗攻击早有先例，例如 MUAP 这类工作已经证明 ReID 排序容易被扰动攻击，APD 的新意在于把攻击思想转成 DG 训练样本和语义防御。([arxiv.org](https://arxiv.org/abs/1910.14184?utm_source=openai))

**3. Attribute Conditional Diffusion-Augmented Person Re-Identification**

1. 触发观察：ReID 缺标注数据，渲染数据贵且粗糙，GAN 增广容易伪影和模式坍塌；扩散模型图像质量高，但用于 ReID 时最要命的是换姿态后身份一致性不稳。PIDM 这类扩散人像生成是直接先例。([arxiv.org](https://arxiv.org/abs/2211.12500?utm_source=openai))
2. 重定义动作：作者把“生成更多图”重定义为“给同一身份生成不同姿态、且保持 ReID 身份嵌入一致的训练数据”。关键词是 Identity Diffuser、identity-specific embedding、target pose heatmap、identity consistency、synthetic pretraining。
3. 机制怎么长出来：先训练一个 ReID 网络提身份 embedding，再用 OpenPose 提目标姿态，把源图、姿态热图、身份 embedding 一起作为扩散条件。生成后的合成集不是直接混训，而是先预训练 ReID，再用真实数据微调。
4. 证据闭环：它的闭环是三件事：FID 证明生成质量好，和 PIDM/UnrealPerson 比下游 mAP，少真实数据比例下收益更大；再做身份 embedding 来源、训练轮数、不同 backbone 的消融，证明“身份条件”比普通生成更有用。
5. reviewer 为什么买账：卖的是机制应用，不是扩散模型本身。真正能打动人的点是“生成质量”和“ReID 身份保持”被绑定到下游 mAP，尤其少数据设定下增益大。弱点是创新更像条件设计，容易被问和 PIDM、CCUP、InfinitePerson 这类合成数据线的边界。

**4. Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification**

1. 触发观察：CLIP-ReID 类方法已经很强，但它依赖整体语义提示，在遮挡和细粒度相似行人上容易只看主特征，漏掉头发、包、衣服细节。CLIP-ReID 与 CoOp 是它的直接技术背景。([arxiv.org](https://arxiv.org/abs/2211.13977?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2109.01134?utm_source=openai))
2. 重定义动作：作者把遮挡 ReID 的问题从“缺 pose、parsing、人工属性标签”重讲成“预训练 VLM 内部已有细粒度属性语义，只是没有被挖出来”。关键词是 inherent pseudo-label、attribute guidance、fine-grained attribute semantics、dual guidance、noise mask。论文来源显示它就是 ECAI 2025 的 AG-ReID。([arxiv.org](https://arxiv.org/abs/2508.04998))
3. 机制怎么长出来：第一阶段冻结 CLIP 图文编码器，用 CoOp 学属性 prompt 的可学习 token，把这些 token 当属性伪标签，同时训练 attribute encoder 从图像特征预测这些 token。第二阶段用整体文本特征和属性伪标签双重引导图像编码器，再用 noise-mask 跳过遮挡导致的不可靠属性匹配。
4. 证据闭环：最关键的消融是 attribute prompt guidance 和 attribute pseudo-label guidance 分别有效、合起来最好；属性模板比默认整体 prompt、长 prompt、随机 prompt 更好；Otsu 阈值的 noise-mask 优于固定阈值；Grad-CAM 看到模型更关注头发、包、衣服。
5. reviewer 为什么买账：卖的是“无需额外标注，从 CLIP 内部挖属性监督”的视角。模块本身不重，但问题重定义漂亮：不是再接一个属性标注器，而是把可学习 token 变成训练监督。它的风险是 CLIP prompt ReID 已经拥挤，必须靠“遮挡噪声属性对齐”这条证据站住。

**5. Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification**

1. 触发观察：VI-ReID 过去常把跨模态差异当作要消除的东西，结果共享对齐会丢掉可见光颜色纹理、红外热特征这些细节；生成式补偿又会引入噪声或依赖辅助模型。BDLF 论文页面也明确把缺口说成“只挖 shared，忽略 modality-specific details”。([arxiv.org](https://arxiv.org/abs/2505.03286))
2. 重定义动作：作者把“模态差异”从敌人改写成两类知识：shared base 和 modality-specific detail。关键词是 base-detail、modality-shared base knowledge、modality-specific detail knowledge、correlation restriction、independent decomposition。
3. 机制怎么长出来：如果 base 是跨模态相关的，detail 是跨模态不相关但有辨识度的，那么自然要用相关性约束：拉高 base 的跨模态相关，压低 detail 的跨模态相关。DFE 用可逆网络从中层无损提 detail；BEG 用投影矩阵把综合特征拆成 detail 子空间和 base 子空间，并加正交独立约束。
4. 证据闭环：消融逐个去掉 DFE、BEG 相关损失、正交约束、SKD；还做 INN block 数量、DFE 接入 ResNet 哪一层的实验。最有说服力的是 stage-3 最好，因为太浅没身份语义，太深 shared 太强不利于 detail，这和他们的 base/detail 定义能对上。
5. reviewer 为什么买账：卖的是视角和形式化机制。它不是说“我又对齐了一下”，而是说“过度对齐会损失有用细节”。近邻工作里 SGIEL 已经用正交子空间和形状擦除来挖 shared cue，LUPI 也用中间域补信息，所以 BDLF 的边界在“base/detail 相关性分解，而非额外生成或形状特权信息”。([arxiv.org](https://arxiv.org/abs/2304.04205?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2209.09348?utm_source=openai))

**6. Beyond geometry: The power of texture in interpretable 3D person ReID**

1. 触发观察：2D ReID 受视角和背景影响大，3D ReID 理应解决，但已有 3D 方法多是把纹理贴到点云上做分类，实际没有充分利用 3D 的视角归一化，也没用好高分辨率纹理。3DInvarReID 等说明 3D 形状线已经存在。([arxiv.org](https://arxiv.org/abs/2308.10658?utm_source=openai))
2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
3. 机制怎么长出来：先用 3D 重建和 Texformer 得到点云与 UVTexture。UVTextureNet 从标准化 UV 纹理中学外观，3DReIDNet 从点云中学几何，两个距离矩阵融合。解释模块把激活图投到 UVTexture 上，再和 visible clothing mask 算 IoU 与 activation alignment，最后用这些分数重排距离。
4. 证据闭环：它必须证明两件事：UVTexture 的注意力比 2D 图像更对齐、更少受背景和视角影响；IAS 不只是好看，还能提升重排序结果。论文用 2D/UVTexture 激活图对比、双流对比、KNN/属性/解释重排消融和 aerial-ground 数据集结果形成闭环。AG-ReID.v2 这类跨空地数据也说明视角差异是真问题。([arxiv.org](https://arxiv.org/abs/2401.02634?utm_source=openai))
5. reviewer 为什么买账：卖的是大视角，不是普通双流。它把 3D 的价值从“多一个点云模态”升级成“规范化纹理坐标让外观、解释、匹配都可对齐”。风险也明显：依赖离线重建和重排序，严格方法投稿时要小心被认为是系统工程而不是核心学习机制。

**共同套路**

这批论文共同不是从“我加了一个模块”开始，而是先把已有失败讲成一个更精确的新问题：过度对齐丢细节、整体 CLIP 忽略属性、3D 只用几何没用纹理、生成图像缺身份一致性。然后机制基本从这个新定义里长出来，最后用一个能直接验证新定义的消融或可视化闭环兜住，而不是只报涨点。
tokens used
162,771
我按你列出的 6 篇拆，不按标题里的“3”处理。方法正文来自本地 PDF 的前 8 页，联网主要核了相近先例和论文来源。

**1. Adversarial flow-based generative models for visible-to-Infrared person Re-Identification**

1. 触发观察：V2I ReID 同时缺数据、跨模态差异大。已有 GAN 翻译主要做“补目标模态”，但红外到可见光是信息少到信息多，天然难；而 SYSU-MM01、RegDB 这类数据又不够大。
2. 重定义动作：作者没有把问题讲成“再做一个翻译器”，而是讲成“可见光和红外都应该映射到同一个可逆高斯潜空间”。关键词是 shared Gaussian domain、bijective transformation、training sample expansion、cross-modal generation。
3. 机制怎么长出来：一旦有两个可逆流，可见光和红外都能进同一个潜空间，那么同模态扩增就是在潜空间插值后映回原模态，跨模态生成就是源模态进潜空间再从目标流出来。身份编码器和模态判别器是自然补上的约束，用来保证“像同一个人”和“像目标模态”。
4. 证据闭环：关键不是只报 SOTA，而是分别验证 training sample expansion 和 cross-modal generation 对多个基线都有效，并做生成图、潜空间插值、非线性层消融、统计检验。它在证明“同一个潜空间确实能同时支撑两种生成任务”。
5. reviewer 为什么买账：卖的是视角加框架，不是某个小层。真正 novelty 是把“数据少”和“模态缺口”统一成一个可逆潜空间操作。不过现在对我们不能新抄，生成式 VI-ReID、GAN 翻译、flow/latent bridge 这条线已经有明显先例。

**2. Adversarial perturbation and defense for generalizable person re-identification**

1. 触发观察：DG ReID 的常规说法是“源域覆盖不够”或“身份无关特征干扰”。作者抓到更具体的一点：在 hard matching samples 上，身份相关和无关特征纠缠，已有按类内类间距离拆特征的假设，在特征空间还没学好时并不稳。
2. 重定义动作：把 DG ReID 重讲成“模型要能面对度量关系被故意扰乱的困难样本，并能防御身份无关语义的攻击”。关键词是 metric adversariality、hard matching samples、adversarial perturbation and defense、semantic purification。
3. 机制怎么长出来：先用 MPG-Net 反过来做 metric learning，让生成样本靠近负样本、远离原样本，同时用浅层内容一致性保持人形和主体外观。再用 SP-Net 学防御，浅层拆出身份相关特征和身份无关特征，把打乱后的身份无关特征加回去攻击身份相关特征，强迫模型仍能分类正确。
4. 证据闭环：最关键的是 hard testing set 让普通基线出现断崖式下降，说明生成样本真的改变了匹配难度；再通过去掉 metric perturbation loss、生成图可视化、MPG-Net 与 SP-Net 逐步消融，证明不是普通 GAN 增广。
5. reviewer 为什么买账：卖的是“用反度量困难样本训练泛化能力”的视角。机制有些堆，但 perturbation 到 defense 的逻辑很顺。需要注意，ReID 对抗攻击早有先例，例如 MUAP 这类工作已经证明 ReID 排序容易被扰动攻击，APD 的新意在于把攻击思想转成 DG 训练样本和语义防御。([arxiv.org](https://arxiv.org/abs/1910.14184?utm_source=openai))

**3. Attribute Conditional Diffusion-Augmented Person Re-Identification**

1. 触发观察：ReID 缺标注数据，渲染数据贵且粗糙，GAN 增广容易伪影和模式坍塌；扩散模型图像质量高，但用于 ReID 时最要命的是换姿态后身份一致性不稳。PIDM 这类扩散人像生成是直接先例。([arxiv.org](https://arxiv.org/abs/2211.12500?utm_source=openai))
2. 重定义动作：作者把“生成更多图”重定义为“给同一身份生成不同姿态、且保持 ReID 身份嵌入一致的训练数据”。关键词是 Identity Diffuser、identity-specific embedding、target pose heatmap、identity consistency、synthetic pretraining。
3. 机制怎么长出来：先训练一个 ReID 网络提身份 embedding，再用 OpenPose 提目标姿态，把源图、姿态热图、身份 embedding 一起作为扩散条件。生成后的合成集不是直接混训，而是先预训练 ReID，再用真实数据微调。
4. 证据闭环：它的闭环是三件事：FID 证明生成质量好，和 PIDM/UnrealPerson 比下游 mAP，少真实数据比例下收益更大；再做身份 embedding 来源、训练轮数、不同 backbone 的消融，证明“身份条件”比普通生成更有用。
5. reviewer 为什么买账：卖的是机制应用，不是扩散模型本身。真正能打动人的点是“生成质量”和“ReID 身份保持”被绑定到下游 mAP，尤其少数据设定下增益大。弱点是创新更像条件设计，容易被问和 PIDM、CCUP、InfinitePerson 这类合成数据线的边界。

**4. Attribute Guidance with Inherent Pseudo-Label for Occluded Person Re-Identification**

1. 触发观察：CLIP-ReID 类方法已经很强，但它依赖整体语义提示，在遮挡和细粒度相似行人上容易只看主特征，漏掉头发、包、衣服细节。CLIP-ReID 与 CoOp 是它的直接技术背景。([arxiv.org](https://arxiv.org/abs/2211.13977?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2109.01134?utm_source=openai))
2. 重定义动作：作者把遮挡 ReID 的问题从“缺 pose、parsing、人工属性标签”重讲成“预训练 VLM 内部已有细粒度属性语义，只是没有被挖出来”。关键词是 inherent pseudo-label、attribute guidance、fine-grained attribute semantics、dual guidance、noise mask。论文来源显示它就是 ECAI 2025 的 AG-ReID。([arxiv.org](https://arxiv.org/abs/2508.04998))
3. 机制怎么长出来：第一阶段冻结 CLIP 图文编码器，用 CoOp 学属性 prompt 的可学习 token，把这些 token 当属性伪标签，同时训练 attribute encoder 从图像特征预测这些 token。第二阶段用整体文本特征和属性伪标签双重引导图像编码器，再用 noise-mask 跳过遮挡导致的不可靠属性匹配。
4. 证据闭环：最关键的消融是 attribute prompt guidance 和 attribute pseudo-label guidance 分别有效、合起来最好；属性模板比默认整体 prompt、长 prompt、随机 prompt 更好；Otsu 阈值的 noise-mask 优于固定阈值；Grad-CAM 看到模型更关注头发、包、衣服。
5. reviewer 为什么买账：卖的是“无需额外标注，从 CLIP 内部挖属性监督”的视角。模块本身不重，但问题重定义漂亮：不是再接一个属性标注器，而是把可学习 token 变成训练监督。它的风险是 CLIP prompt ReID 已经拥挤，必须靠“遮挡噪声属性对齐”这条证据站住。

**5. Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification**

1. 触发观察：VI-ReID 过去常把跨模态差异当作要消除的东西，结果共享对齐会丢掉可见光颜色纹理、红外热特征这些细节；生成式补偿又会引入噪声或依赖辅助模型。BDLF 论文页面也明确把缺口说成“只挖 shared，忽略 modality-specific details”。([arxiv.org](https://arxiv.org/abs/2505.03286))
2. 重定义动作：作者把“模态差异”从敌人改写成两类知识：shared base 和 modality-specific detail。关键词是 base-detail、modality-shared base knowledge、modality-specific detail knowledge、correlation restriction、independent decomposition。
3. 机制怎么长出来：如果 base 是跨模态相关的，detail 是跨模态不相关但有辨识度的，那么自然要用相关性约束：拉高 base 的跨模态相关，压低 detail 的跨模态相关。DFE 用可逆网络从中层无损提 detail；BEG 用投影矩阵把综合特征拆成 detail 子空间和 base 子空间，并加正交独立约束。
4. 证据闭环：消融逐个去掉 DFE、BEG 相关损失、正交约束、SKD；还做 INN block 数量、DFE 接入 ResNet 哪一层的实验。最有说服力的是 stage-3 最好，因为太浅没身份语义，太深 shared 太强不利于 detail，这和他们的 base/detail 定义能对上。
5. reviewer 为什么买账：卖的是视角和形式化机制。它不是说“我又对齐了一下”，而是说“过度对齐会损失有用细节”。近邻工作里 SGIEL 已经用正交子空间和形状擦除来挖 shared cue，LUPI 也用中间域补信息，所以 BDLF 的边界在“base/detail 相关性分解，而非额外生成或形状特权信息”。([arxiv.org](https://arxiv.org/abs/2304.04205?utm_source=openai)) ([arxiv.org](https://arxiv.org/abs/2209.09348?utm_source=openai))

**6. Beyond geometry: The power of texture in interpretable 3D person ReID**

1. 触发观察：2D ReID 受视角和背景影响大，3D ReID 理应解决，但已有 3D 方法多是把纹理贴到点云上做分类，实际没有充分利用 3D 的视角归一化，也没用好高分辨率纹理。3DInvarReID 等说明 3D 形状线已经存在。([arxiv.org](https://arxiv.org/abs/2308.10658?utm_source=openai))
2. 重定义动作：标题已经在重定义：“不要只讲 geometry，texture 才是 3D ReID 的关键”。进一步把解释性从事后可视化改成检索信号。关键词是 UVTexture、canonical viewpoint、viewpoint-invariant explanation、Intersection-Alignment Score、explainable re-ranking。
3. 机制怎么长出来：先用 3D 重建和 Texformer 得到点云与 UVTexture。UVTextureNet 从标准化 UV 纹理中学外观，3DReIDNet 从点云中学几何，两个距离矩阵融合。解释模块把激活图投到 UVTexture 上，再和 visible clothing mask 算 IoU 与 activation alignment，最后用这些分数重排距离。
4. 证据闭环：它必须证明两件事：UVTexture 的注意力比 2D 图像更对齐、更少受背景和视角影响；IAS 不只是好看，还能提升重排序结果。论文用 2D/UVTexture 激活图对比、双流对比、KNN/属性/解释重排消融和 aerial-ground 数据集结果形成闭环。AG-ReID.v2 这类跨空地数据也说明视角差异是真问题。([arxiv.org](https://arxiv.org/abs/2401.02634?utm_source=openai))
5. reviewer 为什么买账：卖的是大视角，不是普通双流。它把 3D 的价值从“多一个点云模态”升级成“规范化纹理坐标让外观、解释、匹配都可对齐”。风险也明显：依赖离线重建和重排序，严格方法投稿时要小心被认为是系统工程而不是核心学习机制。

**共同套路**

这批论文共同不是从“我加了一个模块”开始，而是先把已有失败讲成一个更精确的新问题：过度对齐丢细节、整体 CLIP 忽略属性、3D 只用几何没用纹理、生成图像缺身份一致性。然后机制基本从这个新定义里长出来，最后用一个能直接验证新定义的消融或可视化闭环兜住，而不是只报涨点。
