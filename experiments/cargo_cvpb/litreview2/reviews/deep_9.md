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
session id: 019ef835-891c-72f3-9a6f-4eed87b2c4fc
--------
user
你是 ReID 创新方法论拆解员 9。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(9)==
CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf
DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf
Deep intelligent technique for person Re-identification system in surveillance images.pdf
Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf
DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf
Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会按项目规则先确认能读到的上下文，然后用 PDF 技能和 `pdftotext` 逐篇抽方法部分。当前沙箱是只读，我不会写入任何文件。
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
/bin/zsh -lc "pwd && rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'AGENTS.md' -g 'STATUS.md' .. ../.. ../../.. 2>/dev/null" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.623.12021/skills/pdf/SKILL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
../../../CLAUDE.md
../../../AGENTS.md
../../../GOAL.md
../../../experiments/STATUS.md
../../STATUS.md

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

codex
我已经确认目录里有你列出的 6 篇目标 PDF，题头写“9”但实际清单只有 6 个文件。我先按这 6 篇逐篇拆，不会把目录里其他论文混进来。
hook: PreToolUse
hook: PreToolUse
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

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "sed -n '1,260p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,260p' ../../../AGENTS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
我看过当前目标和状态了，这次任务和“读文献、做差距分析”一致，不涉及开实验或改代码。接下来我会先定位每篇的 method/approach 区间，再针对创新构造逻辑做拆解。
  → 定位 6 篇论文的方法区间和核心术语
  • 逐篇提炼观察、问题重定义、机制、证据和 novelty 来源
  • 汇总共同的创新构造套路
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "for f in \\
'CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf' \\
'DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf' \\
'Deep intelligent technique for person Re-identification system in surveillance images.pdf' \\
'Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf' \\
'DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf' \\
'Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf'; do
  printf '\\n===== %s =====\\n' \""'$f"
  pdfinfo "$f" 2>/dev/null | rg '"'"'^(Title|Subject|Pages):'"' || true
  pdftotext -f 1 -l 8 \""'$f" - 2>/dev/null | rg -n -i '"'"'^('"\\s*)((abstract|introduction|related work|method|methods|methodology|approach|proposed|experiments?|results|conclusion|i+\\.|[0-9]+\\.?\\s+[A-Z]|[IVX]+\\.?\\s+[A-Z]))' | head -80
 done" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 699ms:
10:Abstract— Visible-infrared person re-identification (VI-ReID)
33:20 March 2024. Date of publication 9 April 2024; date of current version
34:1 March 2025. This work was supported in part by the National Key Research
61:I. I NTRODUCTION
118:II. R ELATED W ORK
147:method can well model general features across modalities
151:experiments on two benchmarks, namely SYSU-MM01 [1]
154:methods, e.g., +1.88% Rank-1 on SYSU-MM01 and +1.1%
229:III. P RELIMINARY
248:I d2 xi , x j > d2 (xi , xr ) ,
291:IV. M ETHOD
295:1 In testing, gallery set contain a series of pedestrian images whose identity
488:1 X
551:1 X X
611:V. E XPERIMENTS
619:96 identities. Under the single-shot and multishot setting, there
662:64 images of eight identities. We randomly sample four
692:I MPACT OF D IFFERENT A LTERNATIVES OF DMM ON SYSU-MM01
740:method, i.e., SMCL [33], CycleTrans can obviously improve

===== DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf =====
Title:           DATE: Dual Asymmetric Textual Embedding guided Person Re-Identification
Subject:         2025 IEEE International Conference on Multimedia and Expo (ICME);2025; ; ;10.1109/ICME59968.2025.11209501
Pages:           6
3:2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11209501
17:Abstract—Inspired by the development of the Visual-Language
31:proposed to inject the description-based textual embedding into
41:I. I NTRODUCTION
72:methods to propose a reasonable fusion strategy for jointly
98:proposed to fuse the visual embeddings and learnable-based
116:II. R ELATED W ORKS
130:method uses learnable-based textual embeddings to boost the
151:III. M ETHODS
373:i fi
418:i i
435:method, where the image-level visual space is used to optimize
531:IV. E XPERIEMENTS
550:Methods
726:proposed Dual Asymmetric Textual Embedding.
733:proposed Cross-attention Adapters. Here, CT denotes crossattention mechanism; AD is the add mechanism, which can
807:V. C ONCLUSION
845:VI. ACKNOWLEDGEMENT

===== Deep intelligent technique for person Re-identification system in surveillance images.pdf =====
Title:           Deep intelligent technique for person Re -identification system in surveillance images
Subject:         Pattern Recognition, 162 (2025) 111349. doi:10.1016/j.patcog.2025.111349
Pages:           10
44:1. Introduction
167:2. Related work
208:approach are insufficient to estimate the recorded performance through
248:Method
354:3. Proposed methodology: HLGFS
359:approach effectively balances the exploration and exploitation phases. It
501:results validated based on these step processes. The outcome of the
521:4. Results and discussion
557:4 GB
558:15 GB
600:1812 individuals. The learning set consists of 16,522 pictures of 702
611:32. The accuracy and loss graph of the training and testing is shown in
697:approach is 97.6 %. The mAP comparison with the prevailing mecha­
902:results of the developed HLGFS model are observed and compared in
910:proposed model must be efficient in identifying the person from
914:Methods
977:Methods
996:Proposed

===== Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf =====
Title:           Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification
Subject:         IEEE Transactions on Circuits and Systems for Video Technology;2026;36;1;10.1109/TCSVT.2025.3600759
Pages:           12
10:Abstract—Text-to-image person re-identification (TIReID)
28:Experimental results demonstrate that DFLSG achieves state-ofthe-art performance compared with other unsupervised methods.
32:I. I NTRODUCTION
37:methods [4], [5], [6], TIReID utilizes diverse text descriptions
45:2026. This work was supported in part by the National Natural Science
120:method to automatically obtain accurate text descriptions.
167:II. R ELATED W ORK
184:methods aim to transfer knowledge learned from labeled
226:methods introduce extra branches to learn pedestrian key
257:III. A PPROACH
346:results and obtain dependable image-text pairs with label
362:1 X
389:1 denotes the image
392:2 represents that the cosine similarity between text features and class
394:3 indicates abnormal text features are detected
396:4 represents
485:1 XX
547:v j ∈Mv
557:vi and t i , t vhn and vthn are the hardest negative text/image features of vi and t i . ϕvij is the image-to-text
623:IV. E XPERIMENTAL R ESULTS
646:200 identities, respectively. UFine6926 [48] is the first ultrafine granularity TIReID database. It contains 6,926 identities
672:approaches, this method obtains the worst results, which shows
699:method and possesses strong generalization ability.
718:experimental results, which fully demonstrate the effectiveness
723:experiments to compare IQR with other filtering algorithms.
741:method achieves the best identification performance among

===== DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf =====
Title:           DIRL: Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID
Subject:         -  Information systems  ->  Information retrieval;-  Computing methodologies  ->  Computer vision tasks;
Pages:           16
41:Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
53:Introduction
59:approaches are highly dependent on adequate lighting conditions [3–5]. However, in real-world
122:experiments are conducted on two public video ReID datasets. The experimental results
127:Related Work
189:proposed a dual-gaussian VAE that models ID-related and ambiguous subspaces with distinct
233:Methodology
419:methods focus on IVI-ReID and face difficulties in solving VVI-ReID. (2) Our method explicitly
426:method first leverages TBM to obtain a unified modal-invariant representation, then uses CIM to

===== Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf =====
Title:           Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification
Subject:         IEEE Transactions on Circuits and Systems for Video Technology;2026;36;2;10.1109/TCSVT.2025.3612751
Pages:           15
9:Abstract—Visible-Infrared Person Re-identification (VI-ReID)
27:experiments demonstrate the state-of-the-art performance of our
28:method, achieving 61.06% Rank-1 and 67.75% mAP in the
34:I. I NTRODUCTION
45:2025. Date of publication 22 September 2025; date of current version
46:5 February 2026. This work was supported in part by the National Natural
106:method. Existing methods either perform broad-spectrum feature mining
205:II. R ELATED W ORK
293:III. M ETHODOLOGY
505:methods.
643:1 X
752:introduction of the margin m > 0, the cross-modality hard
756:1 X h (V,I)
769:approach only considers the maximum and minimum sample
823:1 X h (V,I)
1025:IV. E XPERIMENTS
1071:methods for VI-ReID, in mainstream datasets SYSU-MM01,

codex
我开始逐篇抽文本。第一步先看标题、引言里的问题设定，再读方法区间到实验前，避免只从摘要倒推。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '60,620p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 107ms:
   352	Note that the sum of each row in A equals to 1, and P′ =
   353	P − f̂ I . Thus, (7) can be rewritten as
   354	
   355	F N = F′I + A P − f̂ I
   356	(8)
   357	
   358	′
   359	F N = F I − f̂ I + AP.
   360	(9)
   361	Considering that f̂ I is the averaged vector of F′I , the term of
   362	(F′I − f̂ I ) in (9) will result in an informative sparse tensor.
   363	In this case, F N is mainly composed of the newly aggregated
   364	prototype features, i.e., AP, thereby achieving the alignment
   365	of cross-modality distributions.
   366	To enhance the neutral features, we also place trainable
   367	weights to adaptively adjust the contribution of each pattern
   368	in F N , which is achieved by
   369	ewi
   370	F Ni → Pk
   371	
   372	j=1 e
   373	
   374	wj
   375	
   376	F Ni
   377	
   378	(10)
   379	
   380	where w j is the weight for jth pattern of neutral feature. And
   381	“→” refers to weighting up the neutral feature F Ni extracted
   382	according to the ith pseudo anchor. In this way, the path of
   383	gradient backward in DMM is similar to cross-attention [55].
   384	From (8) and (9), we can see that F N contains a certain
   385	amount of discriminative information form F′I − f̂ I , but it
   386	is still hard to ensure that they are discriminative enough
   387	for VI-ReID. In this case, we further implement Cycle Constructions to enhance their descriptive power.
   388	
   389	C. Discrepancy Modeling Module
   390	DMM acts to mitigate the modality gap of VI-ReID.
   391	Instead of directly embedding the modality-relevant features
   392	into a common semantic space, DMM learns the neutral
   393	features via aggregating information from a set of modalityirrelevant prototypes. The prototypes P ∈ Rn×d consist of
   394	a set of learnable vectors that represent the semantics of
   395	appearance [55]. Considering the actual appearance of the
   396	pedestrian is modality-independent, so the two modalities
   397	share the prototypes.
   398	Concretely, given the discriminative modality-relevant features learned by KCM, denoted as F′I , we first calculate their
   399	discrepancy to the trainable prototypes, where n is the number
   400	of prototypes
   401	P = P − f̂ I .
   402	′
   403	
   404	(6)
   405	
   406	Here, P′ ∈ Rn×d refers to the obtained discrepancy tensor and
   407	f̂ I denotes the averaged feature of F′I . The gradient will not
   408	be zero due to the difference between the average feature f̂ I
   409	and the modality-dependent feature F′I . Afterward, the neutral
   410	features F N ∈ Rk×d are obtained via a residual connection and
   411	a cross attention
   412	F N = F′I + AP′ ,
   413	
   414	!
   415	F′I Wψ (PWσ )T
   416	.
   417	where A = Softmax
   418	√
   419	c
   420	
   421	D. Cycle Construction
   422	The main assumption of Cycle Construction is that if the
   423	learned neutral features can recover modality-relevant information well, they are capable of both cross-modality alignment
   424	and prominent feature discrimination.
   425	Specifically, the proposed cycle construction consists of two
   426	processes, which transform the neutral features into visible and
   427	infrared ones, respectively. Taking a visible image for example,
   428	of which feature maps are denoted as FvI ∈ Rhw×c , we apply
   429	the proposed KCM to reconstruct its modality-relevant features
   430	
   431	FvRe = KCM FvI , F N
   432	(11)
   433	where FvRe is the recovered features and F N acts the role
   434	of pseudo anchors described in (5). During training, we will
   435	minimize the l-1 distance between the recover features FvRe
   436	′
   437	and the modality-relevant ones FvI defined in (5) for the
   438	discriminability of neutral features F NI .
   439	In the other stream, CycleTrans project the neutral features
   440	to the other modality through KCM, i.e., the infrared one here,
   441	defined as
   442	h
   443	i
   444	
   445	FrRe = KCM FkI 1 , FkI 2 , . . . , FkI h , F N
   446	where yi = yk j , mi ̸ = mk j , j = 1, 2, . . . , h.
   447	
   448	(7)
   449	
   450	Here, the attention weights A ∈ Rk×n are also the weighted
   451	adjacent matrix between F′I and P. The Wψ and Wσ here
   452	are weight matrices. A can reformulate semantics in a general
   453	space according to the modality-relevant features F′I .
   454	
   455	(12)
   456	
   457	Here, FrRe denotes the recovered infrared features and
   458	[FkI 1 , FkI 2 , . . . , FkI h ] denotes h feature maps that have the same
   459	identity but from the infrared modality in the batch.
   460	In (12), the neutral features are regarded as the pseudo
   461	anchors for KCM to aggregate semantics from all feature maps
   462	that may provide valuable information. It can help to rule out
   463	
   464	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:38 UTC from IEEE Xplore. Restrictions apply.
   465	
   466	WU et al.: CycleTrans: LEARNING NEUTRAL YET DISCRIMINATIVE FEATURES VIA CYCLE CONSTRUCTION
   467	
   468	the factors that may affect appearance discrepancy between
   469	samples for more accurate reconstruction, e.g., viewpoints,
   470	body poses, and obstructions.
   471	To ensure the reconstruction, we also minimize the semantic
   472	distance between two generated features, i.e., F′I , and FrRe .
   473	This objective is also beneficial for alleviating the modality
   474	gap. For an infrared image, the process of cycle construction
   475	is the same. To maintain the compactness of CycleTrans,
   476	we share the parameters of the three KCMs. The gradient is
   477	only backward through the anchor features to the backbone.
   478	E. Optimization
   479	During training, we apply the following objectives to optimize CycleTrans.
   480	1) Cross-Entropy Loss: As the main objective of VI-ReID,
   481	cross-entropy loss is used to learn the identities of samples
   482	with classifier C(·) under the supervision of the label yi
   483	Lid = −
   484	
   485	B
   486	
   487	 
   488	1 X
   489	logP yi |C f(i)
   490	N
   491	B i=1
   492	
   493	(13)
   494	
   495	where C(f(i)
   496	N ) is the predicted identity based on the flattened
   497	kd
   498	neutral feature f(i)
   499	of the sample xi .
   500	N ∈R
   501	2) Metric Loss: To semantically separate the obtained neutral features, we apply a metric loss to CycleTrans
   502	Lme =
   503	
   504	B
   505	X
   506	
   507	B
   508	X
   509	
   510	1
   511	B 2 i=1 j=1,y ̸=y
   512	i
   513	j
   514	
   515	
   516	
   517	i
   518	( j) ( j)
   519	(i) (i)
   520	+d f N , f̄ N + d f N , f̄ N
   521	
   522	h
   523	
   524	
   525	( j)
   526	ρ − d f(i)
   527	,
   528	f
   529	N
   530	N
   531	(14)
   532	
   533	+
   534	
   535	where [·]+ represents max{·, 0}, B denotes the batch size.
   536	d(·) is the distance function, which is l-2 here. c(i) denotes
   537	the center of class f(i)
   538	N belong to, which is calculated in each
   539	batch, and ρ is the least margin between two classes. Via (14),
   540	CycleTrans can well separate the neural features of different
   541	identities and minimize the distance between the example and
   542	its multi-modality anchor, i.e., the class center c. In this case,
   543	it is much easier to obtain the general and cross-modality
   544	representation, which is critical in VI-ReID.
   545	3) Separation Loss: To learn neutral features with more
   546	diverse patterns, we define the following regularization term:
   547	Lsep =
   548	
   549	k−2 k−1
   550	F Ni F N j
   551	1 X X
   552	2
   553	k i=1 i= j+1 |F Ni |2 |F N j |2
   554	
   555	(15)
   556	
   557	where F Ni is the ith pattern of the neutral feature. Note that,
   558	the last pattern of neutral features F N is not involved in the
   559	Lsep , which plays the role of global representation.
   560	4) Modality Fusion Loss: We also apply the Multikernel
   561	Maximum Mean Discrepancy (MMD) [56] with Gaussian
   562	kernel to make features following a similar distribution:
   563	
   564	
   565	
   566	
   567	LMMD = ||Ev φ FvN − Er φ FrN ||2Hk
   568	(16)
   569	where φ(·) is an implicit feature mapping function and
   570	Hk represents the Reproducing Kernel Hilbert Space (RKHS).
   571	FvN and FrN denote the neutral features of visible and infrared
   572	images, respectively. Equation (16) can ensure the consistency
   573	between the neural features of different modalities.
   574	
   575	5473
   576	
   577	5) Reconstruction Loss: To ensure the discriminability of
   578	neutral features and the quality of reconstructions, we propose
   579	a distance-based reconstruction loss
   580	Lrec = |FvRe − FvI |1 .
   581	′
   582	
   583	(17)
   584	
   585	Here, the | · |1 represents the l-1 distance. By decreasing the distance between reconstructed features FvRe and
   586	′
   587	modality-relevant features FvI , we can keep semantic consistency during transformation.
   588	6) Alignment Loss: We also introduce an Alignment loss to
   589	ensure the quality of recovered cross-modality features, which
   590	is defined by
   591	Laln = |FrRe − FvI |2
   592	′
   593	
   594	(18)
   595	
   596	where | · | denote the l-2 distance. Equation (18) can also
   597	serve to reduce the gap between visible and infrared images
   598	by aligning two types of features.
   599	Notably, in (17) and (18), we use the reconstruction of
   600	visible features as an example. During training, these loss
   601	terms are also applied to infrared images.
   602	In summary, the overall objective function of the proposed
   603	CycleTrans is defined as
   604	L = Lid + Lme + λ1 Lsep + λ2 LMMD + λ3 Lrec + λ4 Laln
   605	(19)
   606	where λ1 , λ2 , λ3 and λ4 are hype-parameters. They are
   607	set mainly based on our empirical knowledge and put the
   608	cross-entropy loss at the center. Specifically, both λ3 and λ4 are
   609	directly set to 0.1 based on their scales of gradients. Only
   610	λ1 and λ2 will be tuned during experiments.
   611	V. E XPERIMENTS
   612	A. Datasets and Metrics
   613	We validate the proposed CycleTrans on two VI-ReID
   614	benchmarks, namely SYSU-MM01 [1] and RegDB [17].
   615	SYSU-MM01 is a large-scale dataset consisting of both
   616	indoor and outdoor images captured by four visible cameras
   617	and two near-infrared ones. The training set contains 395 identities with 22 258 visible images and 11 909 infrared ones. The
   618	query set has 3803 infrared images and the gallery set shares
   619	96 identities. Under the single-shot and multishot setting, there
   620	are 301 and 3010 randomly sampled visible images in the

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf' - | nl -ba | sed -n '40,540p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    40	
    41	I. I NTRODUCTION
    42	Person Re-Identification(ReID) aims to retrieve query images from large-scale gallery images. Previous methods generate the visual embedding based on the metric-based learning [1], part-based methods [2], [3], and self-attention learning [4], [5]. However, the above-mentioned methods only consider the visual clues inferred from the given images, lacking
    43	clues from other modalities. Inspired by the development of
    44	the Visual-Language Models(VLM), the textual embedding
    45	generated from the VLM has been explored to boost the person
    46	representation learning in person ReID [6], denoted as multimodal ReID.
    47	*Corresponding author
    48	
    49	Among all multi-modal ReID methods, CLIP-ReID [7] is
    50	the first method using identity-level textual space to align the
    51	visual embedding. Unlike the traditional metric term used in
    52	existing visual-based ReID methods, CLIP-ReID applies an
    53	additional contrastive loss between the visual embeddings and
    54	the textual-level identity-aware embeddings inferred from the
    55	identity-aware prompts, denoted as the learnable-based textual
    56	embedding. However, learnable-based textual embedding does
    57	not capture the text description containing human priors,
    58	leading to a limited ability to describe the specific information
    59	of each person’s image. Recently, multiple types of textual
    60	embeddings have been employed to enhance the diversity of
    61	textual space [3], [8], [9], e.g., MP-ReID [3] firstly applies
    62	MLLM to extract the image-level description, which is further
    63	fused with the learnable textual tokens and visual tokens by a
    64	synchronization strategy. We define the embedding generated
    65	by the learnable tokens and the generated description as
    66	the learnable-based textual embedding and description-based
    67	textual embedding, respectively. However, the synchronization strategy treats the learnable-based and description-based
    68	textual embedding equally, leading to the generated visual
    69	representation easily affected by the noise contained in each
    70	textual embedding, especially for the description-based textual
    71	embedding. Consequently, it is critical for multi-modal ReID
    72	methods to propose a reasonable fusion strategy for jointly
    73	considering the benefits of learnable-based and descriptionbased textual embedding.
    74	To address the above shortcomings, we introduce an asymmetrical strategy that fuses learnable-based and descriptionbased textual embedding with visual embedding. Although
    75	description-based textual embedding can provide essential
    76	clues containing the human prior, low-quality text description
    77	would lead to the description-based textual embedding being
    78	less discriminative. Note that low-quality images and meaningless descriptions like noise can easily generate low-quality text
    79	descriptions. Therefore, it is a reasonable motivation to treat
    80	the description-based textual embeddings as auxiliary clues
    81	to boost visual and textual representation learning rather than
    82	decisive information. Consequently, we inject the descriptionbased textual embedding into the learnable-based and visual
    83	embedding with the fuse module. To reduce the effect of noise
    84	
    85	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:22 UTC from IEEE Xplore. Restrictions apply.
    86	
    87	in textual description, the identity-aware description-based
    88	textual embedding is generated by averaging the descriptionbased textual embeddings belonging to the same identity,
    89	which is used to boost the discriminative to infer the learnablebased textual space used for aligning the visual representation
    90	learning.
    91	In this work, we propose a novel Dual Asymmetric Textual
    92	Embedding(DATE) that uses learnable-based and descriptionbased textual embedding to asymmetrically guide person representation learning, as shown in Figure 1. We utilize MLLM
    93	to generate consistent-granularity descriptions for each image
    94	under specific prompts, which are then processed through a
    95	Text Encoder to obtain description-based textual embeddings.
    96	Meanwhile, similar to CLIP-ReID, we generate visual embedding and learnable-based textual embedding. After that,
    97	Textual-to-Visual Adapter and Textual-to-Textual Adapter are
    98	proposed to fuse the visual embeddings and learnable-based
    99	textual embeddings with the description-based textual embeddings for injecting the human-level knowledge inferred from
   100	the textual description. We then implement a cross-granularity
   101	training approach, where identity-level learnable tokens are
   102	supervised using image-level description-based textual embeddings and visual embeddings, and image-level visual embeddings are supervised using identity-level description-based
   103	textual embeddings. This method asymmetrically leverages the
   104	description-based textual space and the learnable-based textual
   105	space to facilitate the visual space in learning more robust
   106	visual representations.
   107	Our main contributions are summarized as follows:
   108	1) An asymmetrical fusion strategy is introduced to jointly
   109	consider the benefits of different types of textual embeddings, such as learnable-based and description-based
   110	textual embeddings.
   111	2) We propose a novel Dual Asymmetric Textual Embedding(DATE) that uses learnable-based and descriptionbased textual embedding to asymmetrically guide person
   112	representation learning.
   113	3) Evaluation on several benchmarks verify the effectiveness of the proposed method, e.g., obtaining 90.7%,
   114	83.6%, and 74.7% for Market-1501, DukeMTMC, and
   115	MSMT17, respectively.
   116	II. R ELATED W ORKS
   117	A. Prompt Learning
   118	Inspired by the NLP realm where researchers use prompt
   119	learning methods to avoid the high costs of directly training
   120	models [10]. The rise of ChatGPT and GPT-4 offers a idea
   121	of using instruction-tuning methodology to transfer its ability
   122	to downstream tasks, e.g., InstructGPT [11], FLAN-T5 [12],
   123	which shows the ability to use prompt learning method on
   124	zero-shot and few-shot tasks. Some works transferred this idea
   125	to the vision realm. BLIP-2 [13], Flamingo [14] learns from
   126	image-text pairs and has shown promising ability. CoOp [15]
   127	uses a set of global prompts to help the model fit in downstream tasks. CLIP-ReID [7] trains an identity-level prompt
   128	
   129	to cooperate with the training of the visual encoder. Our
   130	method uses learnable-based textual embeddings to boost the
   131	performance of visual space.
   132	B. Person Re-identification
   133	Person Re-Identification (ReID) is a pivotal task in computer
   134	vision. The attention mechanism shows its feature extraction
   135	ability, enhancing its performance on ReID tasks [4]. Vision
   136	Transformer (ViT) application for image feature extraction
   137	introduces innovative approaches to the ReID task [16]. The
   138	trivial idea is to leverage ViT’s robust feature aggregation capability to consolidate hierarchical and regional features [17].
   139	The development of Vision-Language models [18] have also
   140	provided solutions for ReID tasks. CLIP-ReID [7] uses a
   141	prompt learning strategy [15] to generate a set of discriminative text embeddings and then use it to enhance the visual space by minimizing InfoNCE loss [19] between visual
   142	embeddings and textual embeddings. Instruct-ReID [9] uses
   143	different prompts to help the model shift to different tasks.
   144	MLLMReID [8] uses an MLLM to directly produce visual
   145	embeddings, effectively applying the powerful generalization
   146	ability of MLLM to the ReID tasks.
   147	Previous cross-modal ReID methods have predominantly
   148	employed synchronous fusion of multiple modalities, overlooking the noise within the description-based textual embeddings. In our approach, we achieve a more effective model by
   149	treating the description-based textual embeddings as auxiliary
   150	clues to boost visual and textual representation learning.
   151	III. M ETHODS
   152	A. Overall Framework
   153	This work proposes a novel Dual Asymmetric Textual
   154	Embedding(DATE) that uses learnable-based and descriptionbased textual embedding to asymmetrically guide person representation learning. As shown in Figure 1, the proposed
   155	DATE consists of three types of embedding spaces: visual
   156	space, learnable-based textual space, and description-based
   157	textual space. The description-based textual space first applies
   158	MLLM to extract the textual description for each image and
   159	then uses the frozen Text Encoder to generate the imagelevel description-based embedding. After that, a Textual-toVisual Adapter At2v is proposed to fuse the description-based
   160	embedding and the image’s visual embedding extracted by the
   161	frozen Visual Encoder. Moreover, a Textual-to-Textual Adapter
   162	At2t injects the description-based textual embedding into the
   163	learnable-based textual embedding generated by feeding the
   164	identity-aware prompt into the frozen Text Encoder. Through a
   165	cross-granularity training method we can boost the robustness
   166	within the visual space.
   167	B. Dual Asymmetric Textual Embedding
   168	Previous methods applies adversarial learning between the
   169	visual and textual embeddings inferred from the identity-aware
   170	prompts, denoted as the learnable-based textual embedding.
   171	However, learnable-based textual embedding does not capture
   172	the text description containing human priors, leading to a
   173	
   174	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:22 UTC from IEEE Xplore. Restrictions apply.
   175	
   176	“The person in
   177	the image is
   178	wearing a white
   179	T-shirt and black
   180	pants”
   181	
   182	Matmul
   183	
   184	Matmul
   185	
   186	Softmax
   187	
   188	Enhanced
   189	Visual
   190	Embeddings ෡
   191	𝑭
   192	
   193	ℒ𝑐𝑙𝑠
   194	
   195	ℒ𝑐𝑜𝑛
   196	
   197	…
   198	
   199	Learnable-based
   200	Textual
   201	Embeddings 𝑻𝒍
   202	
   203	Retrieval
   204	
   205	GeLU
   206	
   207	Softmax
   208	
   209	Matmul
   210	
   211	Matmul
   212	
   213	…
   214	
   215	Text
   216	Encoder 𝓔
   217	
   218	ℒ𝑑
   219	
   220	ℒ𝑐𝑒
   221	
   222	…
   223	…
   224	
   225	…
   226	
   227	Learnable
   228	-based
   229	Textual
   230	Learnable
   231	Space
   232	Prompts 𝔼
   233	
   234	𝕋
   235	
   236	…
   237	
   238	Textual-to-Visual Adapter
   239	
   240	ID-level
   241	Descriptionbased
   242	Textual
   243	Embeddings
   244	𝕋෢𝒅
   245	
   246	…
   247	…
   248	
   249	…
   250	
   251	…
   252	…
   253	
   254	Visual
   255	Space
   256	
   257	…
   258	
   259	Visual
   260	Encoder 𝓥
   261	
   262	Visual Embeddings
   263	𝑭
   264	
   265	GeLU
   266	
   267	𝐌𝐋𝐋𝐌
   268	𝒢
   269	
   270	Image-level
   271	Descriptionbased
   272	Textual
   273	Embeddings
   274	𝒅
   275	
   276	…
   277	
   278	“Please describe the
   279	appearance of the
   280	person in the picture.
   281	Do not add any other
   282	sentences.”
   283	
   284	Text
   285	Encoder 𝓔
   286	
   287	Semantic
   288	Descriptions
   289	𝑻𝒍
   290	
   291	MLLM Prompts
   292	
   293	…
   294	
   295	Description
   296	-based
   297	Textual
   298	Space
   299	
   300	Enhanced
   301	Learnable-based
   302	Textual
   303	Embeddings 𝑻෡𝒍
   304	
   305	Textual-to-Textual Adapter
   306	
   307	Fig. 1. The framework of DATE. DATE employs a description-based textual space as complementary to the visual and learnable-based textual space. We
   308	use two cross-attention adapters to fuse description-based textual descriptions with visual and learnable-based textual embedding, respectively. We conduct a
   309	cross-granularity training method, guiding the construction of visual space with learnable-based and description-based textual embeddings.
   310	
   311	limited ability to describe the specific information of each
   312	image. Moreover, the text description generated by MLLM
   313	can be used to construct a complementary textual space to
   314	enhance its diversity. Therefore, a novel Dual Asymmetric
   315	Textual Embedding(DATE) is proposed to use learnable-based
   316	and description-based textual embedding to asymmetrically
   317	guide person representation learning.
   318	Given the training dataset D = {xi , yi }N
   319	i=1 sampled from
   320	M identities, for the learnable-based textual space, CLIP-ReID
   321	defines a set of learnable prompts E = {E 1 , E 2 , · · · , E M },
   322	where E i denotes the identity-aware prompt of i-th identity. Specifically, the learnable prompt E i is designed as
   323	“A photo of a [e1i ][e2i ] · · · [eni ] person.”, where eni stands for
   324	learnable tokens and n is the length of prompt. After that,
   325	the learnable-based textual embeddings of all M identities
   326	T l = {tl1 , ..., tlM } can be generated by feeding all learnable
   327	prompts E into the Text Encoder E:
   328	T l = E(E),
   329	
   330	(1)
   331	
   332	where E are learnable tokens needed for optimization.
   333	Meanwhile, we apply the frozen Visual Encoder V to extract
   334	the visual embedding f i for the image xi ,
   335	f i = V(xi ),
   336	
   337	(2)
   338	
   339	where V denotes for the visual encoder.
   340	The proposed Dual Asymmetric Textual Embedding D aims
   341	to take the visual embedding f i , the image xi , and MLLM
   342	
   343	G as input, for generating the enhanced visual embeddings
   344	F̂ = {f̂ 1 , ..., f̂ N }, which can formulated as Eq. (3),
   345	f̂ i = D(xi , f i , G).
   346	
   347	(3)
   348	
   349	Formally, given the image xi , LLaVA(G) is employed to
   350	generate the image-level description Tdi based on the prompt
   351	such as “{Please describe the appearance of the person in the
   352	picture,...}”. After that, the Text Encoder E is applied to generate the corresponding description-based textual embedding
   353	tdi = E(Tdi ). Once obtaining the pair of visual embedding f i
   354	and the description-based textual embedding tdi , a Textual-toVisual Adapter At2v is proposed to fuse those two types of
   355	embeddings,
   356	f̂ i = At2v (f i , tdi ).
   357	(4)
   358	To better fuse the visual and textual embedding, a multihead attention module and two fully connected layers implement the Textual-to-Visual Adapter At2v . Since the visual
   359	embedding f i contains enough discriminative information,
   360	it is treated as the value and key terms in the multi-head
   361	attention module, and the description-based textual embedding
   362	is served as the query term for providing the complementary
   363	clues to generated the fused visual embedding. After that, two
   364	additional fully connected layers are employed to project the
   365	fused visual embedding. Finally, the Textual-to-Visual Adapter
   366	At2v can be formulated as Eq. (5),
   367	d
   368	
   369	At2v (f i , ti ) = F2 (GeLU(F1 (softmax
   370	
   371	⊤
   372	td
   373	i fi
   374	√
   375	d
   376	
   377	!
   378	f i + f i ))) + f i ,
   379	
   380	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:22 UTC from IEEE Xplore. Restrictions apply.
   381	
   382	(5)
   383	
   384	where F1 and F2 are two fully-connected layers.
   385	In additional, we also employ the description-level textual embedding tdi to enhance the learnable-based textual
   386	embedding tlyi to formulate the enhanced learnable textual
   387	embeddings Tˆl = {t̂ , ..., t̂ } with a proposed Textual-to1
   388	
   389	N
   390	
   391	Textual Adapter At2t :
   392	l
   393	
   394	t̂i = At2t (tlyi , tdi ),
   395	
   396	(6)
   397	
   398	where yi is the identity label of image xi . Note that Textualto-Textual Adapter At2t can project the identity-aware textual
   399	l
   400	embedding tlyi into the image-aware textual embedding t̂i
   401	by injecting the the image-aware description-based textual
   402	embedding tdi .
   403	Similar to the Textual-to-Visual Adapter, the Textual-toTextual Adapter comprises a multi-head attention module
   404	and fully connected layers. Since learnable-based textual embedding is more discriminative than description-based visual
   405	embedding, learnable-based textual embedding is treated as the
   406	value and key terms, and description-based textual embedding
   407	serves as the query term for multi-head attention. Therefore,
   408	Textual-to-Textual Adapter At2t can be formulated as Eq. (7),
   409	l
   410	
   411	d
   412	
   413	At2t (ty , ti ) = GeLU(F (softmax
   414	i
   415	
   416	⊤
   417	tly td
   418	i i
   419	√
   420	d
   421	
   422	!
   423	d
   424	
   425	d
   426	
   427	d
   428	
   429	ti + ti )) + ti ,
   430	
   431	(7)
   432	
   433	where F is the fully-connected layer.
   434	Subsequently, we implement a cross-granularity training
   435	method, where the image-level visual space is used to optimize
   436	the identity-level learnable prompts E, and the identity-level
   437	textual embeddings are used to optimize the image-level visual
   438	space. We optimizes the learnable prompt E by minimizing
   439	the two types of losses: Lcon (Tˆl , F̂ ) denotes the contrastive
   440	loss between learnable-based textual embeddings Tˆl and visual embedding F̂ , and Lcon (Tˆl , Td ) denotes the contrastive
   441	loss between learnable-based textual embeddings Tˆl and the
   442	description-based textual embeddings Td , which is formulated
   443	as Eq. (8),
   444	Ll = Lcon (Tˆl , F̂ ) + Lcon (Tˆl , Td ),
   445	
   446	(8)
   447	
   448	where Lcon is the contrastive loss with formulated as Eq. (9),
   449	N
   450	X
   451	exp(s(tˆli , tdi ))
   452	Lcon (Tˆl , Td ) = −
   453	log P
   454	.
   455	N
   456	ˆl , td ))
   457	exp(s(
   458	t
   459	i=1
   460	j
   461	i
   462	j=1
   463	
   464	(9)
   465	
   466	To enable the visual space to learn more robust representations, we optimize the visual space at the identity level using
   467	both the learnable and description-based textual embeddings.
   468	Therefore, the final objective is:
   469	exp(s(f i , T yi ))
   470	,
   471	Lce (f i , T , yi ) = −log PM
   472	j=1 exp(s(f j , T yj ))
   473	Lv = Ld + Lce + Lcls ,
   474	
   475	(10)
   476	(11)
   477	
   478	where Lcls is the standard classification loss, and Lcls is
   479	the contrastive loss. Moreover, Ld is the additional visual
   480	
   481	contrastive loss that supervises the visual space from the
   482	identity level description-based embeddings.
   483	The identity-aware description-based textual embeddings
   484	are used to maintain a general description for each person
   485	by using a memory bank M ∈ RM ×D initialized with the
   486	enhanced learnable-based textual embeddings. Especially, the
   487	memory bank M is initialized with Tˆd = {tˆd1 , ..., tˆdM } which
   488	is the identity-aware learnable-based textual embedding, as
   489	shown in Eq. (12),
   490	P
   491	d
   492	j∈Pyi (tj )
   493	ˆ
   494	d
   495	,
   496	(12)
   497	M yi = tyi =
   498	|Pyi |
   499	where Pyi is the set of images belong to yi -th identity.
   500	Different from the textual-level classifier T than contains
   501	the textual knowledge, the contrastive loss between the visual
   502	embedding f i and the memory bank M is used to optimized
   503	the Visual Encoder,
   504	exp(d(fˆi , Myi )/τ )
   505	,
   506	Ld (xi , yi ) = PM
   507	ˆ
   508	j=1 exp(d(f i , Mj )/τ )
   509	
   510	(13)
   511	
   512	where d() denotes for the Euclidean distance, and τ is for the
   513	temperature factor.
   514	Moreover, the visual embedding is used to momentum
   515	update the cluster memory M during backward propagation
   516	with Eq. (14):
   517	Myi = ωMyi + (1 − ω) · fˆi ,
   518	
   519	(14)
   520	
   521	where fˆi and yi is the visual embedding and label for image
   522	xi , respectively, and ω as the momentum factor. Myi is the
   523	yi -th cluster feature in cluster memory M.
   524	By minimizing Ll in Eq. (8) and Lv in Eq. (11), we
   525	asymmetrically optimize the visual space and learnable-based
   526	textual space, allowing them to mutually enhance each other.
   527	Under the cross-granularity training paradigm, we leverage
   528	image-level information to boost the discriminativity of the
   529	learnable-based textual space, while employing identity-level
   530	information to boost the robustness within the visual space.
   531	IV. E XPERIEMENTS
   532	Datasets: We conduct experiments on three benchmarks:
   533	MSMT17 [20], Market-1501 [21], and DukeMTMC [22].
   534	Market-1501 contains 32,668 images from six different cameras and 1,501 identities, 751 of which are used for training
   535	and 750 for evaluation. DukeMTMC consists of 1,812 identities captured by eight cameras with 702 training identities
   536	and 1,110 testing identities. MSMT17 is a challenging person
   537	ReID benchmark containing 126,441 images captured from
   538	4,101 identities collected from 15 cameras.
   539	Protocols: Two evaluation metrics are used to quantitatively
   540	evaluate the performance of ReID methods, including the

codex
前两篇已经抽完。它们的共同点很明显：都不是直接说“我加一个注意力模块”，而是先把旧范式里的副作用单独拎出来，再让模块服务这个副作用。现在继续读第三篇。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Deep intelligent technique for person Re-identification system in surveillance images.pdf' - | nl -ba | sed -n '40,540p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    40	ciency of the approach is validated and related to existing techniques. The accuracy of HLGFS is 97.8 %, and the
    41	mAP is 97.6 % for the CUHK03 dataset, with 97.68 % accuracy, and 98.87 % mAP for the Market1501 dataset
    42	and for the DukeMTMC re-id dataset, the model achieved 96.65 % accuracy and 96.65 % mAP.
    43	
    44	1. Introduction
    45	Person re-identification employs visual data from surveillance film in
    46	non-overlapping views to match persons to a query [1]. It is frequently
    47	and extensively used in public security and video surveillance [2]. The
    48	photographic photos of the same people recorded by several cameras
    49	may change dramatically due to the background clutter and the sharp
    50	variances in angles and lighting [3]. These features act as obstacles to
    51	proper camera matching of pedestrians [4]. Since pedestrians come in a
    52	variety of forms, a quadratic image patch that always includes some
    53	backdrop regions is used to represent a pedestrian for person
    54	re-identification [5]. The majority of low-quality photos are made with
    55	various camera angles [6] and an unregulated backdrop, making faces
    56	and features impractical [7]. In this case, various people are seen as
    57	being quite similar or the same person’s picture seems different [8]. As a
    58	result, creating an intelligent model to examine the security footage is
    59	rapidly approaching [9].
    60	
    61	The widespread machine learning and deep learning models for
    62	pedestrian re-identification are employed [10]. Significant advance­
    63	ments in this sector have been made possible by deep learning’s ongoing
    64	development [11]. The learning architectures examined a variety of
    65	factors, including deep visual information, semantic traits, and super­
    66	ficial visual features, to classify people [12]. Significant advancements
    67	in this sector have been developed to examine a variety of factors,
    68	including deep visual information, semantic traits, and superficial visual
    69	features, to classify people [13]. The presence of background noises
    70	makes the model complex to provide an accurate prediction [14].
    71	However, these models have gone through pose misalignment problems
    72	[15]. To overcome this, an optimized intelligent prediction system is
    73	employed. However, the proposed research gives equal importance to all
    74	the parts that extract sufficient features and make it easier for the system
    75	to notice the individual’s identity in the surveillance cameras, and by
    76	selecting the optimal features it improves the prediction accuracy.
    77	
    78	* Corresponding author.
    79	E-mail addresses: mallikapandeeswari@francisxavier.ac.in (Ms.R.M.A. Pandeeswari), gmanly12@gmail.com (D.G. Rajakumar).
    80	https://doi.org/10.1016/j.patcog.2025.111349
    81	Received 14 March 2024; Received in revised form 4 December 2024; Accepted 6 January 2025
    82	Available online 10 January 2025
    83	0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    84	
    85	Ms.R.M.A. Pandeeswari and D.G. Rajakumar
    86	
    87	Pattern Recognition 162 (2025) 111349
    88	
    89	1.1. Problem statement and objectives
    90	
    91	formulation. Henceforth, the present features were analyzed, and the reperson identification was initiated. The Re-person identification was
    92	executed by analyzing the body shape and capturing body or face fea­
    93	tures from the specific test images. Finally, some traditional models
    94	measured and validated the key robustness metrics. The unique contri­
    95	bution and novelty of this study are integrating the Horned Lizard
    96	optimization process with the Googlenet model to make a hybrid person
    97	identification system. Here, before, this Hybrid novel HLGFS is not
    98	tested for person-identification applications. The model includes pre­
    99	processing and feature analysis modules to enhance re-identification
   100	accuracy. The model removed low-level noise features that enhance
   101	the image quality and make the images feasible for feature extraction.
   102	The fitness procedure for the Horned Lizard selected and extracted
   103	sufficient features such as body shape, face, and clothing information for
   104	better prediction and also tuned and adjusted the hyperparameters of
   105	the network model for the robust learning of the features, leading to
   106	increased prediction accuracy. Several studies were done based on
   107	intelligent models for re-person identification, and the models were used
   108	in real-world environments. Each model showcases its benefits and also
   109	exhibits some limitations that forbid it from being used in real-time
   110	applications. The recent model surveyed in the related work section
   111	has various limitations, such as lower identification accuracy due to the
   112	lower image quality, insufficient features identification stage, compu­
   113	tation complexity, increased cost, and increased time consumption. A
   114	new model must be introduced to address all these limitations of the
   115	prevailing person re-identification system. The recommended model
   116	carried out robust preprocessing steps that enhance the quality of the
   117	images for easy identification, and essential features are extracted by the
   118	optimal feature selection process, reducing the time and computation
   119	complexity.
   120	The article is arranged as given: Section 2 reviews the traditional
   121	literature, Section 3 explains the problems in the re-identification
   122	frameworks, Section 4 describes the solution for the stated problems
   123	with a new method, Section 5 discusses the results of the created solu­
   124	tion, and Section 6 noted the conclusion of the research with future
   125	work.
   126	
   127	The urgency of public safety has increased surveillance systems; reperson identification is imperative in the intelligence surveillance sys­
   128	tem. Re-person identification is a widely used method of retrieving
   129	specific persons from different camera angles. It is one of the very
   130	challenging tasks due to the different viewpoints of the particular per­
   131	sons, varying illuminations, camera environments, occlusion, back­
   132	ground, etc. The images from the surveillance camera are too noisy,
   133	increasing the complexity score in identifying the persons who have
   134	been recorded for crimes. Several bio-inspired models were imple­
   135	mented in the past for this re-person identification system, but it could
   136	not gain the expected optimal solutions, due to their lack of feature
   137	selection. All optimization algorithms like particle swarm algorithm,
   138	grey wolf, Ant colony, Hyena optimal model, etc., can executed up to the
   139	optimal solution iteration, once, the optimal solution is found then the
   140	optimization iteration is stopped. In this case, the features identifying
   141	face challenges due to the stop iteration of optimal solution inaccurate
   142	feature selection is recorded. HLGFS is developed to address several
   143	critical challenges in person re-identification, particularly within sur­
   144	veillance systems used for public safety. It incorporates horned lizard
   145	optimization because it is ideal for hunting. Its hunting fitness is much
   146	better than bio-inspired mechanics, due to their skin characteristics
   147	changing behaviour. This behavior has attracted this research study for
   148	identifying the features of every different characteristic video frame, by
   149	changing its behavior fitness. Here, the skin color-changing fitness is
   150	updated in the dense layer of Googlenet, which afforded the finest
   151	feature selection outcome for all tested video frames. The main reason
   152	for choosing the googlenet, it can train more and different format video
   153	frames. Hence, incorporating the horned lizard features in the Googlenet
   154	and making the hybrid prediction mechanism “HLGFS” is considered the
   155	key novelty of this study. Practically, it is utilized for criminal or crime
   156	identification by conducting the investigation on surveillance cameras
   157	in public areas. In addition, for identifying the person in crowded scenes,
   158	and dynamic lighting, the skin change behavior of the horned lizard was
   159	utilized for recognizing the unique characteristics of the tested person
   160	from the recorded crowed video frames.
   161	The prime objectives of this work are re-person predictive framework
   162	was introduced in the artificial intelligence field with the bioinspired
   163	model to enhance the camera identification system. Here, the Re-person
   164	was identified by processing the intelligent concepts. Hence, the specific
   165	objectives of this study are:
   166	
   167	2. Related work
   168	Existing works related to the re-identification task are described
   169	below,
   170	Nguyen et al. [16] provided an enlarged dataset with wearable,
   171	CCTV, and aerial technologies for aerial-ground ReID study. To assist
   172	ongoing studies in this area, the company will make the baseline code
   173	and enriched dataset freely available to the community. However, po­
   174	tential bias in the data and reliance on the head region could prevent
   175	generalization. Huang et al. [17] propose a sequential step learning
   176	architecture (SSLA) that improves the feature extraction performance of
   177	the re-identity network by co-segmentation. Focusing on large-scale
   178	spatiotemporal human re-identification, the Large Scale Spatio Tempo­
   179	ral (LaST) dataset is the most significant and efficient labeled re-ID
   180	benchmark, by Xiujun et al. [18]. Long-term and cloth-changing cir­
   181	cumstances demonstrate LaST’s strong generalization capacity.
   182	Although it may not provide all the obstacles needed to explore
   183	long-term scenarios, the dataset seeks to stimulate investigation in the
   184	re-ID domain and promote the development of re-ID methods for
   185	real-world scenarios. To re-identify people, Yiheng et al. [19] suggested
   186	an end-to-end foreground-aware network (EFAN) that uses camera IDs
   187	from pre-existing datasets to locate background regions and create a soft
   188	foreground mask. This method promotes foreground and background
   189	branches for more robust and discriminative feature representations,
   190	using target improvement modules and attention loss. A deep
   191	learning-based technique for surveillance system human recognition is
   192	presented by Choudhary et al. [20]. It trains a multi-tasking model using
   193	a Siamese architecture (SA) with similarity and classification con­
   194	straints. The technique may not be as applicable in the actual world due
   195	
   196	• Introduces the HLGFS hybrid system that combines horned lizardinspired optimization and the Googlenet architecture to enhance
   197	person Re-identification capabilities.
   198	• Improves feature extraction by the horned lizard optimization
   199	mechanism so the adaptive identification of relevant features within
   200	noisy, occluded, or dynamic conditions can be done by the system.
   201	• Avoids the problems faced by bio-inspired optimization algorithms
   202	by maintaining a continuously adapting fitness mechanism to ensure
   203	a robust and accurate selection of features.
   204	• Validates the performance of the HLGFS on widely recognized
   205	datasets CUHK03, Market1501, and DukeMTMC to testify to its
   206	robustness.
   207	Research Question: The features of the traditional intelligent
   208	approach are insufficient to estimate the recorded performance through
   209	re-person identification. It increased the identification error with less
   210	accuracy. To resolve these issues, the current work has focused on
   211	building a novel tuned forecasting framework for Re-person identifica­
   212	tion. The essential contribution is explained as follows,
   213	The surveillance image databases were gathered and taken as Python
   214	input during the prime process. Consequently, a novel HLGFS was built
   215	with the required critical functional parameters. Here, the image noises
   216	were analyzed and eliminated at the preprocessing stage by filtering
   217	2
   218	
   219	Ms.R.M.A. Pandeeswari and D.G. Rajakumar
   220	
   221	Pattern Recognition 162 (2025) 111349
   222	
   223	to its reliance on paired input during training. Zhu et al. [21] introduced
   224	an automatic aligning transformer (AAT) for the person’s re-identity
   225	prediction. However, its response is low for the background patches.
   226	Chen et al. [22] used attention pyramid architecture (APS) to exploit the
   227	attention area in the person re-identification. The model works based on
   228	the split and merge principle. The computational cost and time is very
   229	high. Dong et al. [23] proposed a framework that includes the
   230	multi-view characteristics of the person images. It integrates feature
   231	maps to describe the target pedestrian. The model achieved higher
   232	performance benefits. However, it may not be optimal for superior in­
   233	formation extraction. The attention mechanism may ignore essential
   234	feature information. Wang et al. [24] proposed a multi-deep supervision
   235	with attention features to address this issue. Multi-structure and deep
   236	supervision have been used to remedy the essential global feature in­
   237	formation loss. However, the model faced cross-modality issues due to
   238	omitting local and salient features. Gupta et al. [25] suggested the re­
   239	sidual neural network and transfer learning for the better learning of
   240	visual features in the person-identification process. In addition, hyper­
   241	parameters’ influence is explored in this model. The overall comparison
   242	of the discussed related works is presented in Table 1.
   243	
   244	Table 1
   245	Advantages and limitations of related works.
   246	Authors
   247	
   248	Method
   249	
   250	Advantages
   251	
   252	Disadvantages
   253	
   254	Nguyen
   255	et al. [16]
   256	
   257	Attribute-based,
   258	three-stream
   259	ReID technique
   260	
   261	Both rank accuracy
   262	and total
   263	performance have
   264	improved.
   265	
   266	Huang et al.
   267	[17]
   268	
   269	Sequential step
   270	learning
   271	architecture
   272	
   273	Xiujun et al.
   274	[18]
   275	
   276	LaST
   277	
   278	Yiheng et al.
   279	[19]
   280	
   281	End-to-end
   282	foregroundaware network
   283	
   284	Choudhary
   285	et al. [20]
   286	
   287	A multi-tasking
   288	model using a
   289	Siamese
   290	architecture
   291	
   292	It achieves notable
   293	performance gains
   294	with more distinct
   295	and temporal
   296	invariant body
   297	attributes.
   298	Strong
   299	generalization
   300	capacity
   301	The method
   302	promotes
   303	foreground and
   304	background
   305	branches for more
   306	robust and
   307	discriminative
   308	feature
   309	representations.
   310	Increased accuracy
   311	performance and
   312	efficient validation
   313	process
   314	
   315	Potential bias in the
   316	data and reliance on
   317	the head region could
   318	prevent
   319	generalization.
   320	Complexity,
   321	computing expenses,
   322	and parameterization
   323	are increased.
   324	
   325	Zhu et al.
   326	[21]
   327	
   328	AAT
   329	
   330	Chen et al.
   331	[22]
   332	
   333	Attention
   334	pyramid
   335	architecture
   336	
   337	Dong et al.
   338	[23]
   339	
   340	Multi-view
   341	characteristicsbased framework
   342	
   343	Wang et al.
   344	[24]
   345	
   346	Multi-deep
   347	supervision with
   348	attention
   349	features
   350	Residual neural
   351	network and
   352	transfer learning
   353	
   354	3. Proposed methodology: HLGFS
   355	The proposed architecture for the Re-ID works is based on integrating
   356	the Horned Lizard optimization [26] and GoogleNet [27]. The horned
   357	lizard optimization is the metaheuristic function designed based on the
   358	adaptive and defensive behavior of the horned Lizard. This optimization
   359	approach effectively balances the exploration and exploitation phases. It
   360	converges robustly toward near-optimal solutions. By mimicking the
   361	lizard’s adaptive color changing, it adapts its search strategy dynami­
   362	cally according to the problem space to enable the re-identification
   363	system to navigate diverse solution areas. Moreover, the use of an
   364	adaptive mechanism based on iteration progress and solution quality
   365	adjusts the step size to avoid the inefficiencies associated with a balance
   366	between exploration and exploitation. Integration into the Horned Liz­
   367	ard GoogLeNet Forecasting System achieves an improved ability to
   368	extract critical features to identify people through the optimization
   369	technique. Utilizing GoogLeNet’s inception layers for multi-scale feature
   370	extraction coupled with HLO’s adaptive exploration, HLGFS provides a
   371	more detailed and nuanced understanding of the input data. This syn­
   372	ergy enables the capturing of subtle details of the features. HLGFS
   373	thereby enhances identification and demonstrates the novelty of HLO in
   374	feature extraction and re-identification systems. The block structure of
   375	the proposed schema is visualized in Fig. 1.
   376	The process starts with the data training function. The data input and
   377	the learning process of the proposed HLGFS model are detailed in Eqn.
   378	(1).
   379	T(Sd ) = in (n = 1, 2, ...x)
   380	
   381	Gupta et al.
   382	[25]
   383	
   384	(1)
   385	
   386	It significantly beats
   387	the other researched
   388	models, even in
   389	large datasets.
   390	Better
   391	understanding of the
   392	features and
   393	enhanced
   394	identification
   395	accuracy
   396	
   397	Its adaptation to
   398	different surroundings
   399	may be limited due to
   400	its dependence on
   401	camera IDs.
   402	
   403	The technique may
   404	not be as applicable in
   405	the actual world due
   406	to its reliance on
   407	paired input during
   408	training.
   409	Its response to the
   410	background patches is
   411	low.
   412	The computational
   413	cost and time is very
   414	high.
   415	It may not be optimal
   416	for superior
   417	information
   418	extraction.
   419	The model faced
   420	cross-modality issues
   421	due to omitting local
   422	and salient features.
   423	Influence of
   424	hyperparameters
   425	
   426	the noise tracing variable, f indicates the standard pedestrian features
   427	and the term k denotes the low-level features. Based on this Eqn. (2), the
   428	preprocessing layers eliminated the several noisy features for better
   429	identification.
   430	
   431	Here T indicates the data training function, Sd indicates the sur­
   432	veillance image database, in represents the trained person images, and x
   433	represents the total number of images present in the dataset. The layers
   434	of the designed HLGFS model are visualized in Fig. 2.
   435	The designed architecture comprises 11 layers: an input layer, 4
   436	convolution layers (C1, C2, C3, and C4), 3 pooling layers (P1, P2, and
   437	P3), an inception layer, a fully connected layer, and an output layer.
   438	
   439	3.2. Feature extraction
   440	The proposed HLGFS for re-person identification follows feature
   441	extraction and comparing those features with the other images. The
   442	success of the re-person identification relies on the extraction of better
   443	features. In the proposed research, the best features were extracted
   444	based on the cryptic behavior of the horned Lizard. In this phase, the
   445	Lizard imitates the characteristics of the environment, such as color,
   446	texture, etc., to blend with the surroundings. Using this behavior, the
   447	presented system extracts a person’s body shape, face, and clothing at­
   448	tributes. The features are earned by Eqn. (3).
   449	
   450	3.1. Preprocessing
   451	The collected surveillance images contain noises that make the sys­
   452	tem complex to extract sufficient features and identify the re-person. The
   453	preprocessing phase reduces the identification complexity and thereby
   454	increases the model’s overall accuracy.
   455	P = Sd × λ [(f, k) − (k)]
   456	
   457	The system locates
   458	the human and nonhuman regions at
   459	the patch level in
   460	self-manner
   461	The model focuses
   462	on the clothing
   463	features such as
   464	color, even in the
   465	faded background.
   466	The model achieved
   467	higher performance
   468	benefits
   469	
   470	It doesn’t use for longterm scenarios
   471	
   472	(2)
   473	
   474	Here, the preprocessing step is defined as the variable P, λ denotes
   475	3
   476	
   477	Ms.R.M.A. Pandeeswari and D.G. Rajakumar
   478	
   479	Pattern Recognition 162 (2025) 111349
   480	
   481	Fig. 1. Proposed methodology.
   482	
   483	Fig. 2. HLGFS layer architecture.
   484	
   485	(
   486	)
   487	∂.in
   488	F = Sd + ∂ −
   489	(h − (a ∗ , b ∗ , c ∗ ))
   490	imax
   491	
   492	between − 1 and 1, ε indicates the ranking variable that attains the value
   493	from 0 to 1, and (a∗, b∗, c ∗ ) are the best features extracted at the feature
   494	extraction phase. Thus, the re-person in the other images is identified
   495	using the designed HLGFS.
   496	The flow chart displayed in Fig. 3 is designed based on the GoogleNet
   497	and horned lizard model. Based on the specific application need, the
   498	flow chart was modified with more detailed comments. The established
   499	mathematical formulation for every stage process is ordered in the flow
   500	model which is illustrated in Fig. 3. The Python code has been run and
   501	results validated based on these step processes. The outcome of the
   502	discussed methodology is shown in the following results and discussion
   503	section by validating the proposed model’s performance with three
   504	different databases.
   505	
   506	(3)
   507	
   508	Here, the feature extraction function is denoted as F, ∂ indicates the
   509	required feature tracking variable, imax denotes the maximum iteration,
   510	a∗ represents the face attributes, b∗ represents the body features, and c ∗
   511	is the clothing features. Thus, sufficient features for re-person identifi­
   512	cation were extracted during the feature extraction phase of the
   513	designed HLGFS.
   514	3.3. Re-person identification
   515	Further, the extracted features are entered into the prediction layer
   516	of the HLGFS. The system analyzes the extracted features for the reperson identification. The re-person was identified by matching and
   517	ranking the extracted features with the other images. This process has
   518	been validated using the formulation given in Eqn. (4).
   519	RID = m.imax + w(1 / 2 − ε)(a ∗ , b ∗ , c ∗ )
   520	
   521	4. Results and discussion
   522	The developed HLGFS model is executed in the Python platform. The
   523	working performance of the created HLGFS approach is validated by
   524	testing it with data collected from the Kaggle site. The implementation
   525	requirements for the proposed model are listed in Table 2.
   526	
   527	(4)
   528	
   529	Here, the re-person identification process is represented as RID , m
   530	indicates the feature matching variable, w is the random number
   531	4
   532	
   533	Ms.R.M.A. Pandeeswari and D.G. Rajakumar
   534	
   535	Pattern Recognition 162 (2025) 111349
   536	
   537	Fig. 3. Flowchart of HLGFS.
   538	
   539	website.
   540	

codex
第三篇的方法部分读完了。这篇的写法和前两篇差别很大，更像“优化器加网络”的工程组合，方法论价值要单独按它真实的叙事强弱来拆，不能把它当成扎实机制论文来学。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf' - | nl -ba | sed -n '30,625p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    30	Index Terms—Text-to-image person re-identification, unsupervised learning, deep learning.
    31	
    32	I. I NTRODUCTION
    33	EXT-TO-IMAGE person re-identification (TIReID) aims
    34	to retrieve the target pedestrians from a large-scale image
    35	gallery based on specific textual description queries [1], [2],
    36	[3]. Different from single-modality person re-identification
    37	methods [4], [5], [6], TIReID utilizes diverse text descriptions
    38	as query information to match target images, which is considered a cross-modality task. In recent years, due to its potential
    39	application value in smart cities and intelligent transportation
    40	
    41	T
    42	
    43	Received 16 May 2025; revised 4 August 2025; accepted 18 August 2025.
    44	Date of publication 20 August 2025; date of current version 22 January
    45	2026. This work was supported in part by the National Natural Science
    46	Foundation of China under Grant 62222110 and Grant 61603151; in part by
    47	Taishan Scholar Project of Shandong Province under Grant tsqn202103001;
    48	in part by Shandong Provincial Natural Science Foundation under Grant
    49	ZR2023LZH013, Grant ZR2024QF185, Grant ZR2022MF263, and Grant
    50	ZR2023LZH006; and in part by the New Introduced Talents Program of
    51	University of Jinan under Grant 1009569. This article was recommended by
    52	Associate Editor G. Xu. (Corresponding authors: Tongzhen Si; Xiaohui Yang.)
    53	Chenglong Shao, Tongzhen Si, and Xiaohui Yang are with Shandong Key
    54	Laboratory of Ubiquitous Intelligent Computing, University of Jinan, Jinan
    55	250022, China (e-mail: ise sitz@ujn.edu.cn; ise xhyang@ujn.edu.cn).
    56	Hui Yuan is with the School of Control Science and Engineering, Shandong
    57	University, Jinan 250061, China (e-mail: huiyuan@sdu.edu.cn).
    58	Digital Object Identifier 10.1109/TCSVT.2025.3600759
    59	
    60	Fig. 1. (a) Supervised TIReID setting with matched image-text pairs and
    61	identity labels. (b) Unsupervised TIReID settings without image-text match
    62	relationship and identity labels. (c) Description of the proposed dependable
    63	text generation process for building image-text match relationship.
    64	
    65	systems, TIReID has received widespread attention. However,
    66	the inherent modality differences cause the cross-modality
    67	matching process to be difficult.
    68	Several works leverage pedestrian identity information to
    69	explore invariant features at different granularities for bridging
    70	the modality gap [7], [8]. In addition, some methods generate
    71	additional image-text pairs according to identity information
    72	for enhancing the diversity of training samples [9], [10].
    73	These methods require image-text pairs to possess explicitly
    74	matching relationship and identity information, as shown in
    75	Fig. 1 (a).
    76	In practical scenarios, image samples are easily obtainable
    77	and abundant, but text descriptions are typically manually
    78	annotated. In addition, annotating identity labels for different
    79	pedestrian samples consumes much time and is not feasible.
    80	Unsupervised TIReID aims to explore internal pedestrian
    81	information from unlabeled samples, which has broad application prospects. As for unlabeled image and text samples, they
    82	not only lack the label information but also miss the matching
    83	relationship as shown in Fig. 1 (b). This significantly increases
    84	
    85	1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
    86	similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
    87	See https://www.ieee.org/publications/rights/index.html for more information.
    88	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:02 UTC from IEEE Xplore. Restrictions apply.
    89	
    90	1004
    91	
    92	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
    93	
    94	the complexity of the task. Existing image-based unsupervised
    95	learning methods usually perform the clustering operation to
    96	assign pseudo-labels for different pedestrian images [11], [12].
    97	However, these methods could only process image samples
    98	and fail to construct image-text matching relationships. Moreover, they neglect error category samples during the clustering
    99	process, which undermines the reliability of identity labels.
   100	Therefore, they are not suitable for unsupervised TIReID task.
   101	Multimodal Large Language Model (MLLM) possesses
   102	powerful cross-modality understanding and generation capabilities [13], [14]. Some researchers utilize a kind of MLLM to
   103	generate text descriptions with pedestrian attributes according
   104	to given different prompts [15]. Nevertheless, the reliability of
   105	these generated text descriptions remains doubtful. In addition,
   106	some methods cluster sample features to generate identity
   107	information and employ cross-modality matching losses to
   108	enhance semantic consistency [16], [17]. When using all
   109	pseudo-labels, many generated unreliable label information
   110	hinders model optimization. Although these methods do not
   111	employ the labeled text descriptions, they utilize the label
   112	information of images or the image-text match relationship, which belongs to the weak-supervised learning task.
   113	In contrast, we focus on addressing two important problems
   114	for unsupervised TIReID. Firstly, how to obtain better text
   115	descriptions to construct reliable image-text matching relationships. Secondly, how to utilize clustering information to
   116	generate reliable identity labels.
   117	To this end, we propose a Dependability Feature Learning
   118	based on Sample Generation (DFLSG) method for unsupervised TIReID. Dependability is solved from two key aspects.
   119	On the one hand, we introduce a dependable text generation
   120	method to automatically obtain accurate text descriptions.
   121	Considering that MLLM has a strong generation ability and
   122	text samples are uncertain and diverse, we design different
   123	prompts to guide two MLLMs to generate diverse text descriptions. Then, the dependable text is selected as the training data
   124	based on the cross-modality correlation guided by the prior
   125	knowledge from the TIReID model. In this way, we build
   126	the image-text matching relationship as shown in Fig. 1 (c).
   127	On the other hand, since incorrect pseudo-labels could affect
   128	the training process, we propose an Error Sample Filtering
   129	Module (ESFM) to eliminate abnormal samples. We perform
   130	the clustering operation to assign pseudo-labels for image-text
   131	pairs and compute the image class center features. Generally,
   132	the similarity between correct category text descriptions and
   133	the image class centers is higher than that of incorrect category.
   134	Hence, we utilize the Interquartile Range (IQR) filtering algorithm to identify and filter out abnormal samples. In this way,
   135	we refine the clustering results and obtain more dependable
   136	image-text pairs with label information.
   137	Moreover, images and texts belong to two different modalities and follow one-to-one matching relationships. Image-text
   138	pairs with the same identity possess a large modality difference, which increases the intra-class variations. Hence, we
   139	design the Multilevel Triplet Joint Learning (MTJL) process
   140	to explore cross-modality dependable features from the center
   141	and instance views. Firstly, we present a center-level matching
   142	loss that minimizes the distance between class center feature
   143	
   144	and instance features within the same category. Secondly, we
   145	introduce an instance-level matching loss that continuously
   146	optimizes the intra-modality and inter-modality feature distribution to reduce intra-class variations and increase inter-class
   147	distances. The MTJL method effectively aggregates the two
   148	levels of matching loss to optimize the feature distribution
   149	across different modalities and facilitate the model to learn
   150	dependability pedestrian features.
   151	In this study, four major contributions are summarized as
   152	follows.
   153	(1) We propose a dependable text generation process that
   154	leverages the complementary strengths of MLLMs to
   155	generate and filter accurate descriptions for constructing
   156	image-text matching relationships.
   157	(2) We design an ESFM to integrate statistical method
   158	and cross-modality correlation for eliminating abnormal
   159	image-text pairs, which effectively enhances the reliability of identity labels.
   160	(3) We construct an MTJL process to continuously learn
   161	cross-modality dependability features from the center
   162	and instance views.
   163	(4) We conduct numerous experiments on three public
   164	benchmark datasets, and experiment results demonstrate
   165	that DFLSG achieves state-of-the-art performance for
   166	unsupervised TIReID task.
   167	II. R ELATED W ORK
   168	A. Person Re-Identification
   169	Person re-identification (ReID) aims to search and locate
   170	target pedestrian images across cameras according to given
   171	image queries [18]. With the development of deep learning
   172	technologies, many researchers have focused on the person
   173	identification task and proposed some advanced methods [5],
   174	[19], [20], [21]. Some methods construct different network
   175	structures to extract discriminative pedestrian features at different granularities [22], [23], [24]. In addition, some works
   176	design different loss functions to continuously optimize feature
   177	distribution from the representation learning or metric learning
   178	views [25], [26], [27]. The above methods require pedestrian
   179	images with identity information as training data, but identity
   180	information annotation is extremely time-consuming.
   181	Therefore, researchers have begun exploring unsupervised
   182	ReID methods to mine pedestrian key attributes from unlabeled data. Unsupervised ReID methods can be divided into
   183	two categories: unsupervised domain adaptation (UDA) methods and fully unsupervised learning (USL) methods. UDA
   184	methods aim to transfer knowledge learned from labeled
   185	source domain database to unlabeled target domain database
   186	[28], [29]. For example, Wei et al. [30] construct a GAN-based
   187	image style transfer network to change the pedestrian style
   188	from the source domain to the target domain for reducing the
   189	domain gap. Zhai et al. [31] design a mutual learning strategy
   190	and introduce regularization mechanisms in the target domain
   191	to adaptively learn different feature distributions. However,
   192	UDA methods heavily rely on the data quality of the source
   193	domain, and annotating the source domain data also generates
   194	additional resource costs. Hence, some studies have started to
   195	
   196	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:02 UTC from IEEE Xplore. Restrictions apply.
   197	
   198	SHAO et al.: DFLSG FOR UNSUPERVISED TEXT-TO-IMAGE PERSON RE-IDENTIFICATION
   199	
   200	design USL methods that do not require any identity information in the training process [32], [33]. For instance, Dai et al.
   201	[11] employ clustering algorithms to assign pseudo-labels for
   202	unlabeled images and design a clustering-level contrastive loss,
   203	effectively reducing intra-class variations. Some researchers
   204	[34] leverage similarity distributions as soft labels to explore
   205	the relationships among unlabeled images, which promotes
   206	similar images to have consistent feature representations.
   207	These methods have achieved good performance in processing image-based unsupervised ReID task. Unfortunately, when
   208	facing text descriptions, these methods fail to establish the
   209	image-text matching relationship and cannot solve the crossmodality unsupervised learning problem. In this study, we
   210	design an effective method to handle the unsupervised TIReID
   211	task, which greatly improves the model performance.
   212	B. Text-to-Image Person Re-Identification
   213	Li et al. [1] first study the TIReID task, which aims to match
   214	target pedestrian images according to given textual descriptions. They design a recurrent neural network that outputs the
   215	overall sample correlations and publish the first cross-modality
   216	database, CUHK-PEDES. Recently, this challenging task has
   217	received significant attention due to its importance in practical
   218	applications.
   219	Some studies mine robust pedestrian features from the
   220	global view to bridge the inter-modality gap [27], [35]. For
   221	example, Chen et al. [36] design a cross-modality knowledge
   222	adaptive framework to balance the information volume of
   223	two modalities and enhance image-text semantic consistency.
   224	Afterwards, Jiang and Ye [2] utilize implicit relation reasoning and similarity distribution loss to model the relationship
   225	between visual and textual representations. In addition, some
   226	methods introduce extra branches to learn pedestrian key
   227	attributes for improving the fine-grained perception ability. For
   228	example, Shu et al. [37] propose an Implicit Visual-Textual
   229	(IVT) network that leverages multi-level alignment and bidirectional masking to explore sample fine-grained information.
   230	Yan et al. [38] leverage K-Q matrix relations in the transformer
   231	to select important local features and aggregate local attributes
   232	to enhance final feature representation. To further enhance
   233	data diversity, some researchers utilize identity information to
   234	generate additional image-text pairs. For instance, Song et al.
   235	[9] employ the diffusion model guided by clothing accessories
   236	to reconstruct original image-text pairs.
   237	Benefiting from image-text pairs with identity information,
   238	existing supervised methods achieve relatively good performance [39], [40]. However, cross-modality data annotation
   239	typically requires numerous resource consumption. To this
   240	end, researchers have started exploring weakly supervised or
   241	unsupervised TIReID tasks. For example, Bai et al. [15] finetune BLIP to generate text descriptions with key attributes
   242	and introduce text confidence score to mitigate noise attribute
   243	impact. In addition, Zhao et al. [16] utilize clustering operation
   244	to assign pseudo-labels for unlabeled image-text pairs and
   245	design a text-guided matching loss to learn discriminative
   246	visual-textual joint embedding.
   247	Differently, we extend a new method to address how to
   248	obtain dependable text descriptions for building the image-text
   249	
   250	1005
   251	
   252	matching relationship. In addition, we design ESFM to eliminate abnormal samples for acquiring dependable identity
   253	labels. Finally, we propose MTJL to extract dependability features from center and instance views, which could effectively
   254	mitigate the inter-modality discrepancy. Hence, our method
   255	significantly improves the retrieval performance over other
   256	unsupervised TIReID methods.
   257	III. A PPROACH
   258	In this section, we construct a dependability feature learning
   259	framework based on sample generation, termed DFLSG, to
   260	address the unsupervised TIReID task. We first introduce
   261	the overall structure of DFLSG. Then, the dependable text
   262	generation process is described to construct the image-text
   263	matching relationship and ESFM is illustrated how to obtain
   264	dependability labels. Finally, we introduce MTJL that continuously optimizes intra-modality and inter-modality feature
   265	distribution to bridge the inter-modality gap.
   266	A. Structure of DFLSG
   267	As shown in Fig. 2, we present the overall model structure of
   268	DFLSG. Specifically, considering that CLIP [41] can capture
   269	rich semantic information between image and text samples,
   270	we adopt CLIP as the backbone network. Initially, to obtain
   271	reliable image-text pairs for constructing the cross-modality
   272	dataset, we propose a dependable text generation process.
   273	Then, all input images are resized to 384 × 128, and the
   274	maximum text length is limited to 77 tokens. The DBSCAN
   275	clustering algorithm [42] is utilized to assign pseudo-labels
   276	for the image-text pairs. To enhance the reliability of identity
   277	labels, we design ESFM to identify and eliminate abnormal
   278	samples. Based on refined clustering results, we randomly
   279	select P pedestrian identities where each identity contains K
   280	image-text pairs within each batch. Random padding, random
   281	flipping, random erasing, and random masking are applied
   282	as data augmentation strategies. After extracting pedestrian
   283	features by CLIP encoders, the image and text features are
   284	denoted as vi ∈ R512×1 and t i ∈ R512×1 (i = 1, 2, . . . , PK).
   285	Finally, we construct MTJL to explore dependable feature
   286	distribution from center and instance views.
   287	B. Dependable Text Generation Process
   288	Since texts require manual description, this process is
   289	extremely time-consuming when facing large-scale datasets.
   290	Moreover, textual information possesses diversity and uncertainty, which increases the difficulty of obtaining meaningful
   291	descriptions. Hence, generating rich and reliable text descriptions for different pedestrian images is an urgent problem.
   292	Previous methods fine-tune MLLM to generate text descriptions with key attributes according to specific prompts [15],
   293	[43]. Due to the influence of illumination, occlusion, and
   294	resolution, MLLM may generate hallucinations and output
   295	erroneous attributes, which results in unreliable texts. To this
   296	end, we propose a dependable text generation method to
   297	build the image-text matching relationship and enhance the
   298	dependability of training data. Specifically, ChatGPT [44] possesses strong text understanding and generation capabilities,
   299	
   300	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:02 UTC from IEEE Xplore. Restrictions apply.
   301	
   302	1006
   303	
   304	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   305	
   306	Fig. 2. The whole framework of DFLSG. In the dependable text generation phase (a), we leverage the diversity strengths of MLLMs to generate reliable
   307	text descriptions for constructing the cross-modality dataset. Image and text samples are fed into the corresponding encoders to extract pedestrian features.
   308	(b) illustrates that the Error Sample Filtering Module (ESFM) is designed to perform the clustering operation and eliminate abnormal samples. Meanwhile, an
   309	image memory bank is created to store and update the class center features. Finally, (c) represents that Multilevel Triplet Joint Learning (MTJL) is proposed
   310	to optimize the cross-modality reliable feature distribution from the center and instance views.
   311	
   312	which could summarize sentence structures and contextual
   313	information. To increase the diversity of text descriptions, we
   314	first guide ChatGPT to generate various generic templates that
   315	represent general descriptions for different pedestrians, such as
   316	follows:
   317	“With [hair description], the [person/woman/man] is wearing [clothing description] and is also carrying [belongings
   318	description].”
   319	“In [clothing description] and [accessory description], the
   320	[person/woman/man] also has [hair description].”
   321	Then, we randomly select one template as the prompt, which
   322	is fed into multiple MLLMs to generate diverse descriptions
   323	for the same image. However, due to the performance differences among large models, the quality of the generated text
   324	descriptions is different. As shown in Fig. 1 (c), we present two
   325	text descriptions generated by different MLLMs. Obviously,
   326	the text generated by MLLM2 provides a more accurate and
   327	detailed description. Our objective is to select high-quality
   328	text from two different text descriptions. In previous work [2],
   329	Jiang et al. develop a well-performing TIReID model by establishing cross-modality implicit associations. This model not
   330	only possesses rich semantic knowledge but also effectively
   331	perceives the correlation for image-text pairs. Based on previous prior knowledge, we calculate cross-modality similarity to
   332	evaluate the semantic correlation between generated texts and
   333	images. Typically, reliable text descriptions contain the key
   334	pedestrian attributes and possess a high similarity with the
   335	corresponding images. Hence, we select high similarity texts
   336	as training samples to construct reliable image-text matching
   337	relationships.
   338	C. Error Sample Filtering Module
   339	In image-based unsupervised ReID methods, most pedestrian images exhibit clear color and shape information.
   340	
   341	Researchers utilize clustering algorithms to assign pseudolabels for different images, which demonstrates the effectiveness of clustering algorithms [11], [12]. However, because the
   342	clustering process inevitably generates incorrect pseudo-labels,
   343	using all pseudo-labels may introduce wrong information that
   344	damages model performance. To address this, we construct an
   345	Error Sample Filtering Module (ESFM) to refine the clustering
   346	results and obtain dependable image-text pairs with label
   347	information.
   348	In the cross-modality task, image and text have different
   349	feature representations. Image typically contains rich visual
   350	information, while text is based on contextual semantic
   351	information, which results in distinct feature spaces. Hence,
   352	performing clustering operation on the entire cross-modality
   353	database is not suitable. To this end, we perform clustering
   354	operation only on the image samples. Specifically, we utilize
   355	the DBSCAN clustering algorithm [42] to assign pseudo-labels
   356	for pedestrian images at the beginning of each epoch. Since
   357	the image-text matching relationship has been established, the
   358	matched texts also receive corresponding pseudo-labels. Then,
   359	we create an image memory bank C to save the class center
   360	features. The C[a] represents the a-th identity center feature,
   361	which is initialized as follows:
   362	1 X
   363	C[a] = a
   364	v,
   365	(1)
   366	N
   367	a
   368	v∈I
   369	
   370	a
   371	
   372	where I is the image feature set with the a-th pedestrian
   373	identity and N a denotes the number of features within Ia .
   374	Since image and text are two different modalities of representing pedestrians, texts provide semantic attributes and
   375	reflects the image information. For learning cross-modality
   376	features, we evaluate the correlation between the image class
   377	center and text description to remove abnormal sample pairs
   378	in the clustering process. Specifically, we calculate the cosine
   379	similarity between each text feature with the a-th identity and
   380	
   381	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:02 UTC from IEEE Xplore. Restrictions apply.
   382	
   383	SHAO et al.: DFLSG FOR UNSUPERVISED TEXT-TO-IMAGE PERSON RE-IDENTIFICATION
   384	
   385	1007
   386	
   387	D. Multilevel Triplet Joint Learning
   388	
   389	1 denotes the image
   390	Fig. 3. Illustration of Error Sample Filtering Module. O
   391	features with the same pseudo label and their corresponding class center
   392	2 represents that the cosine similarity between text features and class
   393	feature. O
   394	3 indicates abnormal text features are detected
   395	center feature is calculated. O
   396	4 represents
   397	by conducting an Interquartile Range (IQR) filtering strategy. O
   398	that the error image-text sample pairs are removed based on the imagetext matching relationship. Error category samples and texts with redundant
   399	attributes are marked in red and purple, respectively.
   400	
   401	C[a] to obtain the similarity set S = {sim(t, C[a])|t ∈ T },
   402	where sim() denotes the cosine similarity and Ta represents the
   403	text feature set with the a-th identity. Typically, the similarity
   404	between correct text samples and the corresponding image
   405	class center is usually higher than that of incorrect samples. Moreover, if generated texts contain excessive redundant
   406	information, they cannot accurately represent key pedestrian
   407	attributes and lead to lower similarity. Because the number of
   408	abnormal samples is limited and uncertain, directly removing
   409	low similarity sample pairs is infeasible. The IQR algorithm
   410	identifies outliers by defining a abnormal boundary based
   411	on the similarity score range, which makes it suitable for
   412	filtering abnormal samples in small data distributions. Hence,
   413	we employ the IQR filtering algorithm to remove abnormal
   414	image-text pairs with low similarities. Fig. 3 illustrates the
   415	computing process of abnormal samples, where red represents
   416	that the image-text pairs with incorrect labels are removed,
   417	and purple denotes that texts with redundant attributes are
   418	eliminated. The filtered text feature set Taf ilter is represented
   419	as:
   420	Taf ilter = {sim(t, C[a]) ≥ Q1 − β · (Q3 − Q1 )},
   421	(2)
   422	a
   423	
   424	a
   425	
   426	where β is a hyperparameter that controls the filtering strength,
   427	Q1 represents the first quartile (25-th percentile) of the sorted
   428	S a , and Q3 is the third quartile (75-th percentile) of the sorted
   429	S a . Then, we leverage the image-text matching relationship
   430	to search filtered image features. In this way, we construct a
   431	dependable dataset of image-text pairs.
   432	In the training process, we compute the cosine similarity
   433	between each sample and corresponding class center feature.
   434	Then, the sample with the lowest similarity score for each
   435	identity is selected as the hardest positive feature, which is
   436	utilized to update the image memory bank C as follows:
   437	C[a] ← ηC[a] + (1 − η)vah p ,
   438	
   439	(3)
   440	
   441	where η represents the updating rate and vah p is the hardest
   442	positive feature of C[a].
   443	
   444	Because image and text modalities are heterogeneous, the
   445	inherent modality differences increase intra-class variations.
   446	This is the main challenge for the cross-modality TIReID
   447	task. In this section, we present the Multilevel Triplet Joint
   448	Learning (MTJL) that continuously optimizes cross-modality
   449	dependable feature distribution from center and instance views
   450	to bridge the inter-modality gap.
   451	Previous studies [38], [45] have demonstrated that the triplet
   452	loss is effective for supervised TIReID task, which could
   453	continuously optimize the relative distance between positive
   454	and negative sample pairs. Hence, we extend the triplet loss to
   455	the unsupervised TIReID task to mitigate intra-class variations.
   456	Specifically, we first employ triplet loss to constrain the
   457	distance between class center features and instance features
   458	as follows:
   459	Lctrl =
   460	
   461	P X
   462	K
   463	X
   464	[m − sim(C[r], vrk ) + sim(C[r], vrhn )]+ ,
   465	
   466	(4)
   467	
   468	r=1 k=1
   469	
   470	where vrk denotes the k-th image feature with r-th identity, vrhn
   471	represents the hardest negative sample, which is selected with
   472	the highest similarity from different identities, and m is the
   473	margin that controls the distance between positive and negative
   474	samples.
   475	Considering that the traditional triplet loss only optimizes
   476	the relative distance between positive and negative samples.
   477	We extend the center loss to directly constrain the absolute
   478	distance between class center features and instance features as
   479	follows:
   480	Lcen =
   481	
   482	P
   483	K
   484	
   485	1 XX
   486	1 − sim(C[r], vrk ) .
   487	PK
   488	
   489	(5)
   490	
   491	r=1 k=1
   492	
   493	By combining triplet loss and center loss, we leverage
   494	their complementary advantages to jointly constrain intra-class
   495	feature distribution. The center-level matching loss is defined
   496	as follows:
   497	Lcenter = λ1 Lctrl + λ2 Lcen ,
   498	
   499	(6)
   500	
   501	where λ1 and λ2 denote the importance of the center loss and
   502	triplet loss, respectively.
   503	The center-level matching loss primarily focuses on the
   504	feature distribution between class center and instance features
   505	among the image modality. To constrain the features of different cross-modality samples, we introduce an instance-level
   506	matching loss from inter-modality and intra-modality views.
   507	Specifically, the traditional triplet loss calculates the correlation between positive sample pairs by a static focus. It neglects
   508	the similarity discrepancies among different positive sample
   509	pairs. In contrast, a weighted average similarity approach is
   510	adopted to learn image-text pair features by considering the
   511	different feature correlations. The inter-modality matching loss
   512	is defined as follows:
   513	2
   514	3
   515	PK 
   516	X
   517	X
   518	
   519	4m −
   520	Linter =
   521	ϕvij · sim(vi , t j ) + sim(vi , t vh )5
   522	n
   523	
   524	i=1
   525	
   526	t j ∈Mt
   527	
   528	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:02 UTC from IEEE Xplore. Restrictions apply.
   529	
   530	+
   531	
   532	1008
   533	
   534	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   535	
   536	2
   537	
   538	3
   539	
   540	+ 4m −
   541	
   542	X
   543	
   544	ϕti j · sim(t i , v j ) + sim(t i , vthn )5
   545	
   546	
   547	v j ∈Mv
   548	
   549	
   550	
   551	,
   552	
   553	+
   554	
   555	(7)
   556	where Mt and Mv represent positive text/image feature sets of
   557	vi and t i , t vhn and vthn are the hardest negative text/image features of vi and t i . ϕvij is the image-to-text
   558	P similarity weight, calculated as: ϕvij = exp(sim(vi , t j )/τ)/ t j ∈Mt exp(sim(vi , t j )/τ),
   559	where τ serves as a temperature hyperparameter.
   560	Similarly, ϕti j is the text-to-image similarity weight.
   561	Meanwhile, we introduce the intra-modality matching loss
   562	to facilitate all instance features for generating the compact
   563	feature distribution among the same modality:
   564	2
   565	3
   566	PK 
   567	X
   568	X
   569	
   570	4m −
   571	Lintra =
   572	αvil · sim(vi , vl ) + sim(vi , vvh )5
   573	n
   574	
   575	vl ∈Hv
   576	
   577	i=1
   578	
   579	2
   580	+ 4m −
   581	
   582	X
   583	
   584	
   585	
   586	αtil · sim(t i , t l ) + sim(t i , t thn )5 , (8)
   587	
   588	t l ∈Ht
   589	v
   590	
   591	+
   592	
   593	3
   594	
   595	+
   596	
   597	t
   598	
   599	where H and H represent positive image/text feature sets of vi
   600	and t i , vvhn and t thn are the hardest negative image/text features
   601	of vi and t i . αvil is the image-to-image
   602	P similarity weight, calculated as: αvil = exp(sim(vi , vl )/τ)/ vl ∈Hv exp(sim(vi , vl )/τ), and
   603	αtil denotes the text-to-text similarity weight. The optimization
   604	objective of the instance-level matching loss is represented as:
   605	Linstance = λ3 Linter + λ4 Lintra ,
   606	
   607	(9)
   608	
   609	where λ3 and λ4 are hyperparameters that control the weights
   610	of the inter-modality and intra-modality matching losses.
   611	In this study, we present the multilevel triplet joint learning
   612	from both center and instance views to optimize dependable feature distribution, which effectively reduces intra-class
   613	variations and increases inter-class distances. The overall optimization objective can be written as:
   614	L MT JL = Lcenter + Linstance ,
   615	
   616	(10)
   617	
   618	In the inference stage, the query texts and image gallery are
   619	separately fed into our model to extract pedestrian features.
   620	Afterwards, the feature similarity between the two modalities
   621	is calculated, and images with the highest similarity score are
   622	returned as matching results.
   623	IV. E XPERIMENTAL R ESULTS
   624	In this section, we systematically evaluate the proposed
   625	DFLSG on three public TIReID databases. The three databases

codex
第四篇已经能看到更典型的“可靠性”叙事：无监督文本图像 ReID 不是单纯缺标签，而是同时缺匹配关系和可靠伪标签。这个重定义比“用大模型生成文本”更关键。继续读视频可见红外那篇。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf' - | nl -ba | sed -n '50,620p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    50	
    51	J. Wang et al.
    52	
    53	Introduction
    54	
    55	Person Re-Identification (ReID) is a crucial task in computer vision that focuses on identifying the
    56	same individual across different times, cameras, or scenes. It plays a significant role in applications
    57	such as video surveillance, intelligent security systems, crowd monitoring, and other related fields.
    58	Although some deep end-to-end methods [1, 2] have achieved notable success. Most of these
    59	approaches are highly dependent on adequate lighting conditions [3–5]. However, in real-world
    60	applications, such optimal lighting conditions are not always available. For example, in low-light
    61	environments such as nighttime, visible light images (RGB images) often struggle to capture clear
    62	and accurate features of a person. In these cases, infrared images provide a practical alternative by
    63	preserving essential person-related information, even when visible images fall short. Leveraging
    64	this advantage, Visible-Infrared Person Re-Identification (VI-ReID) offers a viable approach
    65	to tackle the challenges posed by such conditions.
    66	VI-ReID aims to match the same individual from a still-image or video sequence across visible and
    67	infrared camera views, resulting in Image-Based Visible-Infrared Person Re-Identification
    68	(IVI-ReID) task [6–8] and Video-Based Visible-Infrared Person Re-Identification (VVIReID) task [9–11]. The IVI-ReID task primarily involves addressing the challenge of cross-modality
    69	feature alignment. For example, Chen et al. [6] proposed a quadruplet deep network with an
    70	attention mechanism to capture global spatial features, thereby reducing spatial misalignment and
    71	semantic inconsistency across modalities. Hermans et al. [7] introduced a triplet loss to minimize
    72	intra-class cross-modality distance while maximizing inter-class separation, enhancing identity
    73	feature alignment. Wang et al. [8] proposed MPMN with modality-specific and shared branches to
    74	disentangle modality noise and preserve identity semantics, effectively addressing feature alignment
    75	at both global and local levels. However, these IVI-ReID methods are designed based on still images,
    76	and their performance often degrades when applied to real-world scenarios (e.g., video data). Due
    77	to variations in pose, movement, perspective, lighting conditions, and spatial placement over time,
    78	video data poses unique challenges for accurate identification.
    79	In contrast, VVI-ReID can effectively handle video data. VVI-ReID primarily focuses on addressing the challenges of spatial-temporal feature learning and cross-modality feature alignment.
    80	For example, Li et al. [9] proposed the Intermediary-Guided Bidirectional Spatial–Temporal
    81	Aggregation Network (IBAN), which uses Convolutional Neural Networks (CNNs) to capture
    82	local features within individual frames, uses Long Short-Term Memory (LSTM) networks to
    83	extract temporal cues from sequential data, and uses an intermediary-guided strategy to bridge
    84	the gap between different modalities. However, when processing lengthy video sequences, LSTM
    85	models struggle to effectively capture temporal nuances and long-range dependencies [10], due
    86	to the sequential nature of their computation and the inherent forgetting mechanism. Moreover,
    87	when handling scenarios involving modality fusion, especially in later video sequences with occlusions and background interference, the model tends to overlook modal invariance. This issue
    88	is particularly evident in earlier video segments. This further exacerbates the gap between features
    89	from different modalities, allowing modal-related information to interfere with the re-identification
    90	task, ultimately diminishing the model’s discriminative ability. Similarly, Lin et al. [11] proposed
    91	MITML, which uses a weight-shared CNN to construct spatio-temporal features, facilitating the
    92	learning of frame-level modal-invariant features. It then incorporates a modal-invariant adversarial
    93	loss to align cross-modality features, ensuring that these features preserve both ID-related and
    94	modality-related information. However, this method only obtains frame-level modal invariance and
    95	does not account for modality invariance when computing sequence-level features, exacerbating
    96	the gap between sequential features of different modalities (as shown in Figure 1). Furthermore,
    97	without further separating modal-invariant features, modal-related information can interfere with
    98	the re-identification task. This interference ultimately reduces the model’s discriminative ability.
    99	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.
   100	
   101	DIRL for Video Visible-Infrared Person ReID
   102	
   103	238:3
   104	
   105	Fig. 1. Compared to the existing methods, our approach emphasizes the interaction of information between
   106	modalities when extracting sequence-level features, further reducing the feature gap between different
   107	modalities.
   108	
   109	To address the aforementioned issues, we propose a Discriminative ID-Related Representation Learning (DIRL) network for VVI-ReID. DIRL network consists of three key components,
   110	that is, Two-Stream Backbone Module (TBM), Cross-Modality Interaction Module (CIM),
   111	and Feature Decoupling Module (FDM). Due to the superiority of the CNN structure in capturing local features of pedestrians, we use two-stream CNN network as our backbone to capture
   112	frame-level modal-invariant features. The CIM, an improved version of attention mechanisms,
   113	enables modality interaction during the adaptive aggregation of temporal information, thereby
   114	obtaining sequence-level modal-invariant features. The FDM is composed of two trainable feature
   115	decoupling matrices, which explicitly separate ID-related information from modal-related information in the modal-invariant features, resulting in more discriminative and modal-sharing ID-related
   116	representations. The contributions of our work are summarized as follows:
   117	— We proposed a CIM to perform modality information interaction and temporal feature aggregation simultaneously, thereby obtaining sequence-level modal-invariant features.
   118	— We developed an FDM and designed its corresponding loss function to effectively separate
   119	ID-related information from modal-related information, thereby leaving only discriminative
   120	ID-related representations.
   121	— By integrating TBM, CIM, and FDM, we proposed the DIRL network for VVI-ReID. Extensive
   122	experiments are conducted on two public video ReID datasets. The experimental results
   123	demonstrate that the proposed method achieves competitive performance compared to some
   124	State-of-the-Art (SOTA) methods.
   125	2
   126	
   127	Related Work
   128	
   129	In this section, we will briefly review some related works on IVI-ReID and VVI-ReID methods.
   130	2.1
   131	
   132	IVI-ReID
   133	
   134	IVI-ReID involves matching images in visible and infrared modalities. Existing IVI-ReID methods
   135	can be categorized into four groups: modal-sharing feature learning-based methods [12–15], modalspecific information compensation-based methods [16–18], modality generation methods [19, 20],
   136	data augmentation methods [21–24], and feature decoupling methods [25–27].
   137	Modal-sharing feature learning-based methods focus on aligning features from different modalities in a shared feature space. They aim to reduce cross-modality discrepancies through feature-level
   138	constraints. For example, Wei et al. [12] proposed a flexible body partition model-based adversarial learning method, incorporating fine-grained part representations and adversarial learning
   139	with a modality classifier to reduce the cross-modality gap and enhance feature distinguishability.
   140	Similarly, Feng et al. [13] proposed a shape-removed feature learning framework that decorrelates
   141	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.
   142	
   143	238:4
   144	
   145	J. Wang et al.
   146	
   147	modal sharing features into two orthogonal subspaces, allowing the model to extract more diverse
   148	modal-sharing features by erasing body shape information. Furthermore, Feng et al. [14] proposed
   149	a cross-modality interactive transformer framework that acquires long-range dependencies and enhanced feature discriminativeness by exchanging information across modalities for modal-invariant
   150	representations. Additionally, Feng et al. [15] proposed a correspondence framework for obstructed
   151	scenarios with a Local Feature Enhance Module (LFEM) and a Modality Information Fusion
   152	Module (MIFM), where LFEM use transformers to enhance non-occluded local features, and MIFM
   153	employed co-attention to align cross-modality correlations.
   154	Modal-specific information compensation-based methods aim to address the issue of missing
   155	modality-specific details by utilizing available modalities. For example, Liu et al. [16] proposed a
   156	two-stage modality enhancement network, which combined improved image generation through
   157	DSGAN and feature-level fusion (PwIF) to effectively reduce modality discrepancy and boost
   158	performance. Similarly, Li et al. [17] proposed MSMNet to address modality discrepancies by
   159	filling in missing modality data and integrating visible and infrared features. This approach utilizes
   160	memory-guided strategies to facilitate the transfer of identity information between modalities.
   161	Furthermore, Kansal et al. [18] proposed spectrum-disentangled representation learning, which
   162	introduced a two-branch network with a dual-branch network with disentanglement loss to extract
   163	identity features while removing spectrum-related information. These approaches collectively
   164	highlight the importance of balancing modal-specific and modal-sharing information in addressing
   165	cross-modality challenges.
   166	Modality generation methods focus on translating one modality into another to ensure modality
   167	consistency. For instance, Wang et al. [19] proposed AlignGAN, which transformed RGB images
   168	into their infrared counterparts by confusing a discriminator, enabling subsequent retrieval in
   169	the infrared feature embedding space. However, directly generating such images may lead to
   170	considerable noise due to the large discrepancies between modalities. To address this issue, Zhang
   171	et al. [20] proposed a novel feature-level modality compensation network, which applied GAN-based
   172	to generate feature vector, leveraging cross-modality features for effective modality compensation
   173	and reducing noise. These approaches illustrate the challenge of maintaining modality consistency
   174	while minimizing noise in cross-modality generation.
   175	Data augmentation methods are widely used to enhance VI-ReID performance by generating more
   176	diverse training data. For example, Fan et al. [21] proposed the cross-spectrum dual-subspace pairing
   177	model, which improved data diversity by splitting RGB images into three channels and adding a
   178	grayscale channel to create similar data. Similarly, Ye et al. [24] expanded a randomly selected RGB
   179	channel into a three-channel image, offering additional variety for training. Moving beyond purely
   180	RGB-based methods, Qian et al. [23] bridged the modality gap by combining segments from visible
   181	and infrared data to form a transitional modality. Additionally, Liang et al. [22] enriched human
   182	body color information by introducing human key-point heatmaps combined with ColorJitter
   183	operations. These strategies highlight how augmenting different channels or combining modalities
   184	can effectively reduce modality gaps and boost model performance.
   185	Feature decoupling methods are effective in mitigating modality gaps in VI-ReID by separating
   186	ID-related features from modal-related noise. For example, Yuan et al. [25] proposed a causal
   187	intervention-based method that decouples modality-specific information from identity features,
   188	effectively reducing spurious correlations and improving cross-modality alignment. Pu et al. [26]
   189	proposed a dual-gaussian VAE that models ID-related and ambiguous subspaces with distinct
   190	priors, enhancing robustness via a triplet swap strategy. Furthermore, Choi et al. [27] propose a
   191	Hierarchical Cross-Modality Disentanglement (Hi-CMD) method, which separate ID-related
   192	information and ID-independent information in visible-infrared images. By alternating between
   193	ID-related and ID-independent information, such as pose or structural details, the model generates
   194	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.
   195	
   196	DIRL for Video Visible-Infrared Person ReID
   197	
   198	238:5
   199	
   200	pseudo-pedestrian images with specific attributes, thereby improving identity representation
   201	learning. These studies highlight the importance of identity–nonidentity disentanglement for
   202	cross-modal alignment.
   203	Although these methods demonstrate improved performance in the IVI-ReID task, they face
   204	difficulties in effectively addressing VVI-ReID. The significant challenges arise from the dynamic
   205	changes in pedestrian appearance, movement, posture, and surroundings over time within video
   206	sequences.
   207	2.2
   208	
   209	VVI-ReID
   210	
   211	VVI-ReID gained more focus because of the availability of video tracklets, which provide more
   212	comprehensive spatial–temporal information compared to still images. This wealth of information
   213	has the potential to greatly enhance retrieval accuracy but also introduces new challenges. Recently,
   214	Lin et al. [11] introduced the HITSZ-VCM dataset, the first extensive dataset for VVI-ReID task.
   215	In their study, Lin et al. [11] leveraged convolutional neural networks to extract local person features
   216	from individual image and utilized long short-term memory networks to acquire temporal patterns
   217	across image sequences. Additionally, adversarial learning was used to extract modality-invariant
   218	features. Following this, Li et al. [9] proposed the IBAN, which used stereoscopic images as an
   219	intermediate step to reduce modality differences and direct the extraction of modality-independent
   220	features. LSTM networks were also employed to capture temporal patterns in video sequences.
   221	Recently, Hou et al. [28] proposed a three-stage decomposition-mining-aggregation framework for
   222	VVI-ReID, which systematically addresses modality discrepancy by decomposing features, mining
   223	identity information, and aggregating the results to enhance feature extraction and improve model
   224	accuracy.
   225	However, existing VVI-ReID methods often focus on frame-level features, overlooking modality
   226	invariance at the sequence level, which leads to discrepancies between sequence-level features of
   227	different modalities. Furthermore, insufficient disentanglement of modal-invariant features allows
   228	modal-related information to interfere, reducing the model’s discriminative capability. To overcome
   229	these limitations, we propose a DIRL network to capture both the long-range temporal dependencies
   230	and ID-related information between ultra-long frames.
   231	3
   232	
   233	Methodology
   234	
   235	In this section, we will provide a detailed exposition of the composition of DIRL which consists of
   236	three key components: TBM, CIM, and FDM. First, to address the challenge of handling diverse
   237	data from distinct modalities, we design TBM following the idea proposed in [29]. It is initially
   238	constructed to preliminarily capture frame-level modal-invariant features, laying the groundwork
   239	for more robust cross-modality interactions in subsequent modules. Then, we use CIM to integrate
   240	modal information and aggregate frame-level features into sequence-level features simultaneously.
   241	Finally, we use FDM to separate modal-related information and ID-related information, leaving
   242	only identity information that is beneficial for VVI-ReID. The architecture of our model is illustrated
   243	in Figure 2.
   244	3.1
   245	
   246	TBM
   247	
   248	
   249	𝑇
   250	Here, an RGB sequence and an infrared sequence are denoted as 𝑉 = 𝑉 𝑡 |𝑉 𝑡 ∈ 𝑅 𝐻 ×𝑊 ×𝐶 𝑡 =1 and
   251	𝑡 𝑡
   252	𝑇
   253	𝐼 = 𝐼 |𝐼 ∈ 𝑅 𝐻 ×𝑊 ×𝐶 𝑡 =1 , where 𝐻 and 𝑊 denote the height and weight of the images, 𝑡 means the
   254	𝑡th frame of this sequence, and 𝑇 is the total number of images in a tracklet. Thus, the identity
   255	labels are represented as 𝑝 𝑣 and 𝑝𝑖 , while 𝑚 𝑣 and 𝑚𝑖 represent the modality labels.
   256	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.
   257	
   258	238:6
   259	
   260	J. Wang et al.
   261	
   262	Fig. 2. The architecture of Discriminative ID-related Representation Learning network. Visible and infrared
   263	video sequences are considered as cross-modality inputs. TBM is first constructed to preliminarily capture
   264	frame-level modal-invariant features. Building on this, CIM is designed to aggregate frame-level features into
   265	sequence-level features. Finally, FDM is employed to remove modal-related information from cross-modality
   266	data and preserve only ID-related features.
   267	
   268	TBM is composed of four convolutional blocks. In the initial block, the two branches use distinct weights to capture modal-specific features for the RGB and infrared sequences. In contrast, the remaining three blocks share weights between the branches, enabling the extraction
   269	of modal-invariant features across both modalities. The convolution blocks
   270	 in TBM are denoted
   271	𝑇
   272	as 𝐸𝑟𝑒𝑠 . We denote the frame-level features from 𝐸𝑟𝑒𝑠 ’s output as 𝑓𝑣 = 𝑓𝑣𝑡 |𝑓𝑣𝑡 ∈ 𝑅 1×𝑃 𝑡 =1 and
   273	
   274	𝑇
   275	𝑓𝑖 = 𝑓𝑖𝑡 |𝑓𝑖𝑡 ∈ 𝑅 1×𝑃 𝑡 =1 . To organize these features across the entire sequence and facilitate further processing, we stack the frame-level outputs for each modality. Accordingly, we design
   276	𝐹 𝑣 = [𝑓𝑣1 ; 𝑓𝑣2 ; . . . ; 𝑓𝑣𝑇 ] ∈ 𝑅𝑇 ×𝑃 and 𝐹𝑖 = [𝑓𝑖1 ; 𝑓𝑖2 ; . . . ; 𝑓𝑖𝑇 ] ∈ 𝑅𝑇 ×𝑃 as the output frame-level features
   277	of TBM.
   278	3.2
   279	
   280	CIM
   281	
   282	To integrate modal information and aggregates frame-level features into sequence-level, we proposed a CIM. The structure of CIM is shown in Figure 2. Since the attention mechanism is highly
   283	effective at capturing global features and plays a key role in extracting temporal features, we will
   284	leverage it to aggregate frame-level features and obtain sequence-level feature representations for
   285	each video sequence. However, applying the attention mechanism independently to each modality’s
   286	input can limit the discriminative power of modal-invariant features. To address this issue, we
   287	propose an enhancement to the traditional Query–Key–Value paradigm. Specifically, based on the
   288	inputs 𝐹 𝑣 and 𝐹𝑖 from each modal, we first compute their corresponding 𝑄 and 𝑉 in Equations (1)
   289	and (2).
   290	𝑄 𝑣 = 𝐹 𝑣 × 𝑊𝑣 ,
   291	
   292	𝑄
   293	
   294	𝑉𝑣 = 𝐹 𝑣 × 𝑊𝑣𝑉 ,
   295	
   296	(1)
   297	
   298	𝑄
   299	
   300	𝑉𝑖 = 𝐹𝑖 × 𝑊𝑖𝑉 ,
   301	
   302	(2)
   303	
   304	𝑄𝑖 = 𝐹𝑖 × 𝑊𝑖 ,
   305	
   306	𝑄
   307	where 𝑊𝑣 ∈ 𝑅 𝑃 ×𝑃 and 𝑊𝑣𝑉 ∈ 𝑅 𝑃 ×𝑃 represent the query and value metrics for RGB modality, while
   308	𝑄
   309	𝑊𝑖 ∈ 𝑅 𝑃 ×𝑃 and 𝑊𝑖𝑉 ∈ 𝑅 𝑃 ×𝑃 stand for query and value metrics for infrared modality, respectively.
   310	Then, to enable information interaction between different modals, we set the 𝐾 ∈ 𝑅 𝑃 ×𝑃 value as a
   311	
   312	shared, learnable parameter. Finally, the calculation process of the CIM can be described as follows:
   313	𝑀𝑖𝐸 = 𝛿 𝑗 (𝑄𝑖 × 𝐾),
   314	
   315	(3)
   316	
   317	𝑉𝑣 = [𝑧 𝑣1 ; 𝑧 𝑣2 ; ...; 𝑧𝑇𝑣 ] ∈ 𝑅𝑇 ×𝑃 ,
   318	
   319	(4)
   320	
   321	𝑀𝑣𝐸 = 𝛿 𝑗 (𝑄 𝑣 × 𝐾),
   322	𝑍 𝑣 = 𝑀𝑣𝐸
   323	
   324	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.
   325	
   326	DIRL for Video Visible-Infrared Person ReID
   327	
   328	238:7
   329	
   330	Fig. 3. The architecture of Feature Decoupling Module. The yellow rectangles represent RGB modal-related
   331	features, the green rectangles represent infrared modal-related features, and the purple rectangles represent
   332	ID-related features. The two light yellow trapezoids represent the RGB modal encoder and decoder, while the
   333	two light green trapezoids represent the infrared modal encoder and decoder.
   334	
   335	𝑍𝑖 = 𝑀𝑖𝐸
   336	
   337	𝑉𝑖 = [𝑧𝑖1 ; 𝑧𝑖2 ; ...; 𝑧𝑇𝑖 ] ∈ 𝑅𝑇 ×𝑃 ,
   338	
   339	(5)
   340	
   341	𝑓𝑣_𝑜𝑢𝑡 =
   342	
   343	𝑇
   344	𝑇
   345	1Õ 𝑗 1Õ 𝑗
   346	𝑧𝑣 +
   347	𝑓 ,
   348	𝑇 𝑗=1
   349	𝑇 𝑗=1 𝑣
   350	
   351	(6)
   352	
   353	𝑓𝑖_𝑜𝑢𝑡 =
   354	
   355	𝑇
   356	𝑇
   357	1Õ 𝑗 1Õ 𝑗
   358	𝑧𝑖 +
   359	𝑓 ,
   360	𝑇 𝑗=1
   361	𝑇 𝑗=1 𝑖
   362	
   363	(7)
   364	
   365	where 𝑀𝑣𝐸 ∈ 𝑅𝑇 ×𝑃 and 𝑀𝑖𝐸 ∈ 𝑅𝑇 ×𝑃 are weight matrices normalized by column-wise softmax 𝛿 𝑗 ,
   366	representing the element-wise external attention matrices of 𝐹 𝑣 and 𝐹𝑖 , respectively. The symbol
   367	denotes the Hadamard product. More specifically, by using Equations (3)–(5), we obtain the postinteraction frame-level features 𝑍 𝑣 and 𝑍𝑖 . Then, we apply average pooling to 𝑍 𝑣 and 𝑍𝑖 to aggregate
   368	the new frame-level features into sequence-level features 𝑓𝑣_𝑜𝑢𝑡 ∈ 𝑅 1×𝑃 , 𝑓𝑖_𝑜𝑢𝑡 ∈ 𝑅 1×𝑃 . Besides,
   369	to ensure the robustness of sequence-level features, the skip connections are introduced in the
   370	calculation of 𝑓𝑣_𝑜𝑢𝑡 , 𝑓𝑖_𝑜𝑢𝑡 . CIM significantly enhances performance and serves as the foundation
   371	of our efficient and effective visible-infrared retrieval network.
   372	3.3
   373	
   374	FDM
   375	
   376	After leveraging CIM to integrate modal information, we obtain modal-invariant features that
   377	contain rich modal-sharing information. Building on this, we generally believe that modal-invariant
   378	feature for each modal is a mixture of modal-related information and ID-related information. And
   379	only ID-related information is beneficial for re-identification. Therefore, our goal is to separate
   380	the modal-related information and ID-related information within the modal-invariant features. To
   381	achieve this goal, we introduce the FDM with two trainable feature decoupling matrices 𝐷 and 𝐸.
   382	The structure of FDM is shown in Figure 3.
   383	According to Equations (6) and (7), the sequence-level features for RGB and infrared tracklets
   384	are 𝑓𝑣_𝑜𝑢𝑡 and 𝑓𝑖_𝑜𝑢𝑡 , respectively. Then, they are input into FDM, and we have:
   385	𝑓𝑣_𝑚𝑜𝑑𝑎𝑙 = 𝑓𝑣_𝑜𝑢𝑡 × 𝐸 𝑣 × 𝐷 𝑣 ,
   386	
   387	(8)
   388	
   389	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.
   390	
   391	238:8
   392	
   393	J. Wang et al.
   394	
   395	∈ 𝑅 𝑃 ×𝑆 and 𝐷
   396	
   397	𝑓𝑖_𝑚𝑜𝑑𝑎𝑙 = 𝑓𝑖_𝑜𝑢𝑡 × 𝐸𝑖 × 𝐷𝑖 ,
   398	
   399	(9)
   400	
   401	𝑓𝑣 = 𝑓𝑣_𝑜𝑢𝑡 − 𝑓𝑣_𝑚𝑜𝑑𝑎𝑙 ,
   402	
   403	(10)
   404	
   405	𝑓𝑖 = 𝑓𝑖_𝑜𝑢𝑡 − 𝑓𝑖_𝑚𝑜𝑑𝑎𝑙 ,
   406	
   407	(11)
   408	
   409	where 𝐸 𝑣
   410	𝑣
   411	modality, while 𝐸𝑖 ∈ 𝑅 𝑃 ×𝑆 and 𝐷𝑖 ∈ 𝑅𝑆 ×𝑃 stand for modal encoder matrix and modal decoder
   412	matrix for infrared modality, respectively. Besides, to ensure the stability of the modal, a residual
   413	connection is added. As for RGB data, 𝑓𝑣_𝑚𝑜𝑑𝑎𝑙 ∈ 𝑅 1×𝑃 and 𝑓𝑣 ∈ 𝑅 1×𝑃 are modal-related feature
   414	and ID-related feature. Next, different loss functions are used to ensure the effectiveness of the
   415	modal-related features and ID-related features, respectively.
   416	So far, many feature decoupling methods have emerged in the field of IVI-ReID, such as UCT
   417	[25], DG-VAE [26], Hi-CMD [27], and so on. The main differences between our proposed method
   418	and the above methods are: (1) Our method aims to solve the VVI-ReID task. In contrast, the above
   419	methods focus on IVI-ReID and face difficulties in solving VVI-ReID. (2) Our method explicitly
   420	separates the sequence-level modal-related features from the ID-related ones through the FDM
   421	module. In contrast, UCT [25] uses causal intervention, DG-VAE [26] employs MoG prior and
   422	standard Gaussian distribution prior, and Hi-CMD [27] utilizes an image generation network to
   423	achieve frame-level feature decoupling. Their model principles and network architectures are all
   424	different. (3) Existing feature decoupling methods usually use convolutional blocks to perform
   425	disentanglement, without adequately integrating cross-modal cues, which can leave residual modalrelated information and weaken the discriminative ability of the learned features. In contrast, our
   426	method first leverages TBM to obtain a unified modal-invariant representation, then uses CIM to
   427	perform modality information interaction and temporal feature aggregation, and finally utilizes
   428	FDM to perform feature decoupling. This strategy enables a more structured decoupling of IDrelated and modal-related information. Comprehensive comparative experiments in Section 4
   429	demonstrate that our approach achieves more effective feature disentanglement and superior
   430	performance than existing methods.
   431	3.4
   432	
   433	∈ 𝑅𝑆 ×𝑃 are modal encoder matrix and modal decoder matrix for RGB
   434	
   435	Loss Function
   436	
   437	In theory, if 𝑓𝑣 and 𝑓𝑖 solely contain identity-related information without modality-specific information, they cannot be classified as belonging to 𝑚 𝑣 or 𝑚𝑖 . To address this, a classifier 𝑊𝑚 is
   438	introduced, which outputs a 3 × 1 vector representing the probabilities of a tracklet belonging to the
   439	RGB modality, the infrared modality, or neither. We consider the third class 𝑚 3 as an embedding
   440	space containing only identity information. Thus, the loss functions can be formulated as follows:
   441	𝐿𝑚1 = 𝐶𝐸 (𝑊𝑚 (𝑓𝑣 ), 𝑚 3 ) + 𝐶𝐸 (𝑊𝑚 (𝑓𝑖 ), 𝑚 3 ),
   442	
   443	(12)
   444	
   445	𝐿𝑚2 = 𝐶𝐸 (𝑊𝑚 (𝑓𝑣_𝑜𝑢𝑡 ), 𝑚 𝑣 ) + 𝐶𝐸 (𝑊𝑚 (𝑓𝑖_𝑜𝑢𝑡 ), 𝑚𝑖 ),
   446	
   447	(13)
   448	
   449	𝐿𝑚 = 𝐿𝑚1 + 𝐿𝑚2,
   450	
   451	(14)
   452	
   453	where 𝐶𝐸 (·) represents the cross-entropy loss function. 𝑚 3 denotes the third category, which
   454	represents a latent space that contains only ID-related information.
   455	To promote the learning of discriminative identity information in 𝑓𝑣 and 𝑓𝑖 , we introduce the
   456	cross-entropy loss and triplet loss via a separate classifier, 𝑊𝑖𝑑 . Consequently, the ID-related,
   457	modal-invariant loss function is expressed as follows:
   458	𝑖𝑑
   459	𝐿𝑖𝑑 = 𝐶𝐸 (𝑊𝑖𝑑 (𝑓𝑣 ), 𝑝 𝑣 ) + 𝐶𝐸 (𝑊𝑖𝑑 (𝑓𝑖 ), 𝑝𝑖 ) + 𝐿𝑡𝑟𝑖𝑝
   460	,
   461	
   462	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.
   463	
   464	(15)
   465	
   466	
codex
DIRL 的前 8 页刚好到损失函数，没有到实验细节；这篇我会把“序列级模态不变性”和“序列级去模态信息”作为它的主叙事来拆。为了证据闭环，我后面会再查它的消融标题和关键表。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf' - | nl -ba | sed -n '620,930p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 10 'DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf' - | nl -ba | sed -n '1,260p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	DIRL for Video Visible-Infrared Person ReID
     2	
     3	238:9
     4	
     5	𝑖𝑑 is triplet loss [7] and 𝑝 , 𝑝 are ID labels. The final loss function is as follows:
     6	where 𝐿𝑡𝑟𝑖𝑝
     7	𝑣 𝑖
     8	
     9	𝐿 = 𝐿𝑖𝑑 + 𝐿𝑚 .
    10	
    11	(16)
    12	
    13	4 Experiment
    14	In this section, we will conduct experiments to validate the effectiveness of our proposed method.
    15	Our experimental results will be presented from two perspectives. First, we will compare our
    16	method with other SOTA approaches. Second, we will conduct ablation experiments to evaluate the
    17	impact of different framework stages and parameter settings on the experimental results. Finally,
    18	we will conduct some visual analysis on our model.
    19	4.1
    20	
    21	Implementation Details
    22	
    23	The experimental configuration comprises an Intel 13900k CPU and an NVIDIA GTX 4090 GPU
    24	with 24GB of memory. For the backbone architecture, we employ the ResNet50 model, which has
    25	been pretrained on the ImageNet dataset. In order to create training sample sequences that are
    26	compatible with the model, we establish a batch size of 16, incorporating 8 distinct identities, with
    27	2 tracklets allocated to each identity. Each tracklet consists of eight frames. The dimensions of
    28	each frame are adjusted to 288 × 144 pixels, and various data augmentation techniques are applied
    29	to each training sample, including random cropping, horizontal flipping, and random erasing.
    30	The parameter 𝑆 in the FDM is set to 5,120. The Stochastic Gradient Descent optimizer is utilized,
    31	featuring a momentum of 0.9 and a weight decay of 1𝑒 − 4. The initial learning rate is established at
    32	0.01, with a maximum training duration of 200 epochs, during which a cosine learning rate decay
    33	strategy is implemented.
    34	4.2
    35	
    36	Datasets
    37	
    38	To validate the effectiveness of our proposed method, we performed experiments on two widely
    39	recognized VVI-ReID datasets: HITSZ-VCM [11] and BUPTCampus [30]. These are two recently
    40	collected datasets from 2022 and 2023, respectively.
    41	The HITSZ-VCM dataset comprises 251,452 RGB images and 211,807 infrared images, featuring
    42	927 distinct identities, gathered from 12 different, non-overlapping cameras. Its training set includes
    43	9,751 tracklets from 500 pedestrians, with 4,291 tracklets from infrared pedestrians and 5,460
    44	from visible pedestrians. In the testing set, the query consists of 4,884 tracklets from different 427
    45	pedestrians, while the gallery contains 5,099 tracklets from the same 427 pedestrians.
    46	The BUPTCampus dataset collected 16,826 tracklets of 3,080 pedestrians from 6 cameras, totaling
    47	1,869,066 images. These samples are randomly divided into training and testing sets. Its training
    48	set includes 3,574 tracklets from 1,017 pedestrians. In the testing set, the query consists of 1,076
    49	tracklets, while the gallery contains 4,844 tracklets. Notably, nearly 30% of the pedestrians are
    50	captured by only one camera. Additionally, the dataset includes an auxiliary set with 930 pedestrians
    51	to assist in model training.
    52	4.3
    53	
    54	Evaluation Metrics
    55	
    56	The performance evaluation in the VVI-ReID task employs two widely utilized metrics: the Cumulative Matching Characteristics (CMC) curve and the Mean Average Precision (mAP)
    57	[31]. The CMC curve, depicted as rank-k, assesses the top-k result accuracy. On the other hand,
    58	the mAP metric evaluates our method’s performance by computing the average precision across
    59	various recall rates and subsequently averaging these values. In addition, we introduce the Avg
    60	metric, which represents the average of each evaluation metric such as rank-k and mAP. It is used
    61	to assess the overall performance of the model. This comprehensive measure provides a balanced
    62	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.
    63	
    64	238:10
    65	
    66	J. Wang et al.
    67	
    68	Table 1. Comparison of CMC (%) and mAP (%) Performances with the State-of-the-Art Methods on the
    69	HITSZ-VCM Dataset
    70	
    71	Method
    72	
    73	Year
    74	
    75	DDAG [29]
    76	DG-VAE [26]
    77	Hi-CMD [27]
    78	Lba [32]
    79	MPANet [33]
    80	VSD [34]
    81	CAJL [24]
    82	MITML [11]
    83	AuxNet [30]
    84	DEEN [35]
    85	HOS-Net [36]
    86	UCT [25]
    87	CST [37]
    88	Ours
    89	
    90	2020
    91	2020
    92	2020
    93	2021
    94	2021
    95	2021
    96	2021
    97	2022
    98	2023
    99	2023
   100	2024
   101	2024
   102	2024
   103	
   104	Infrared to Visible
   105	Rank-1 Rank-5 Rank-10
   106	54.62
   107	69.79
   108	76.05
   109	47.23
   110	62.85
   111	73.12
   112	51.34
   113	67.91
   114	73.53
   115	46.38
   116	65.29
   117	72.23
   118	46.51
   119	63.07
   120	70.51
   121	54.53
   122	70.01
   123	76.28
   124	56.59
   125	73.49
   126	79.52
   127	63.74
   128	76.88
   129	81.72
   130	51.05
   131	53.70
   132	74.81
   133	80.74
   134	61.27
   135	75.14
   136	79.02
   137	58.39
   138	76.81
   139	80.19
   140	64.97
   141	65.18
   142	79.12
   143	84.60
   144	
   145	mAP Rank-1
   146	39.26 59.03
   147	31.27 51.15
   148	39.59 54.26
   149	30.96 49.30
   150	35.26 50.32
   151	41.18 57.52
   152	41.49 60.13
   153	45.31 64.54
   154	45.99 54.58
   155	50.43 49.81
   156	46.02 63.91
   157	43.18 61.23
   158	46.12 68.12
   159	47.93 66.99
   160	
   161	Visible to Infrared
   162	Rank-5 Rank-10
   163	74.64
   164	79.53
   165	70.22
   166	76.54
   167	70.61
   168	78.13
   169	69.27
   170	75.90
   171	67.31
   172	73.56
   173	73.66
   174	79.38
   175	74.62
   176	79.86
   177	78.96
   178	82.98
   179	71.64
   180	80.97
   181	74.62
   182	81.41
   183	75.83
   184	81.47
   185	81.72
   186	84.19
   187	
   188	mAP
   189	41.50
   190	42.25
   191	41.49
   192	32.38
   193	37.80
   194	43.45
   195	42.81
   196	47.69
   197	48.70
   198	48.59
   199	47.90
   200	49.01
   201	48.27
   202	50.17
   203	
   204	Avg
   205	61.82
   206	56.82
   207	59.60
   208	55.21
   209	55.54
   210	62.01
   211	63.56
   212	67.72
   213	63.83
   214	66.17
   215	65.76
   216	69.98
   217	
   218	Avg represents the mean of all evaluation metrics. The best-performing model is highlighted in red, followed by
   219	the second-best in green, and the third-best in blue for comparative clarity.
   220	
   221	perspective on the model’s effectiveness across multiple dimensions, offering a clearer indication
   222	of its strengths and weaknesses in various aspects of evaluation.
   223	4.4
   224	
   225	Comparative Experiments on the HITSZ-VCM Dataset
   226	
   227	In this section, we will assess the effectiveness of our proposed method by comparing it with
   228	established SOTA methods, which includes IVI-ReID methods: LbA [32], MPANet [33], DDAG [29],
   229	DG-VAE[26], Hi-CMD [27], VSD [34], CAJL [24], DEEN [35], HOS-Net [36], and UCT [25]; and VVIReID methods: MITML [11], AuxNet [30], and CST [37]. For IVI-ReID methods, an average pooling
   230	layer is applied to their generated sequence-level features. All these methods utilize ResNet50
   231	as backbones. It is worth emphasizing that when compared to CST [37], we chose the results at
   232	frames = 8 for fair comparison.
   233	Table 1 reports the comparison results on the HITSZ-VCM dataset. As shown in the table,
   234	compared with SOTA methods, our approach achieves significant improvements across various
   235	evaluation metrics. Specifically, in the infrared-to-visible task, our Rank-1, Rank-5, and Rank10 accuracies reached 65.18%, 79.12%, and 84.60%, respectively, surpassing all listed methods.
   236	Moreover, compared to other approaches, our method shows at least a 0.21%, 2.24%, and 2.88%
   237	improvement in Rank-1, Rank-5, and Rank-10 accuracy, respectively. In the visible-to-infrared
   238	task, our Rank-5 and Rank-10 results reached 81.72% and 84.19%, respectively, outperforming the
   239	next-best methods by at least 2.76% and 1.21%. Additionally, our method achieves an mAP of
   240	50.17%, reflecting a 1.16% improvement. Finally, our model obtained 69.98% on the avg metric,
   241	surpassing the current best-performing model by 2.26%. In addition, compared to existing feature decoupling methods (i.e., UCT, DG-VAE, Hi-CMD), our method also achieves much better
   242	performance.
   243	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.
   244	
   245	
codex
DIRL 的方法区间比前 8 页长，我补读了第 9 到 10 页把损失和实验开头接上。现在读 Multi-Frequency 这篇，这类题目通常靠“频域成分不是一个整体”来构造新问题。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '30,1028p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    30	LLCM, surpassing existing methods by significant margins without resorting to reranking or additional labeled data. Code is
    31	available at https://github.com/GuHY777/MFENet-VIReID.
    32	Index Terms—Person re-identification, cross-modality, visibleinfrared, Fourier transform, retrieval loss.
    33	
    34	I. I NTRODUCTION
    35	
    36	P
    37	
    38	ERSON re-identification (ReID), a critical technology for
    39	intelligent surveillance systems and urban management
    40	[1], aims to learn discriminative features that maintain identity consistency across non-overlapping camera views. While
    41	recent advances in visible-spectrum recognition [2], [3], [4]
    42	have achieved remarkable progress, these methods remain
    43	
    44	Received 30 May 2025; revised 26 August 2025; accepted 18 September
    45	2025. Date of publication 22 September 2025; date of current version
    46	5 February 2026. This work was supported in part by the National Natural
    47	Science Foundation of China under Grant 62401609, in part by the Natural
    48	Science Basic Research Plan in Shaanxi Province of China under Grant
    49	2024JC-YBQN-0628 and Grant 2025JC-YBMS-730, and in part by the China
    50	Postdoctoral Science Foundation under Grant 2024M754275. This article
    51	was recommended by Associate Editor X. Shu. (Corresponding author:
    52	Hongyang Gu.)
    53	The authors are with the Rocket Force University of Engineering,
    54	Xi’an 710025, China (e-mail: guhy7@outlook.com; doctoryxg@163.com;
    55	lrt19880220@163.com; warmstoner@163.com; hansm119@outlook.com;
    56	hyacinth531@163.com).
    57	This article has supplementary downloadable material available at
    58	https://doi.org/10.1109/TCSVT.2025.3612751, provided by the authors.
    59	Digital Object Identifier 10.1109/TCSVT.2025.3612751
    60	
    61	fundamentally limited to daylight applications due to their
    62	dependence on optimal illumination conditions.
    63	To enable round-the-clock surveillance capabilities, crossmodality visible-infrared person re-identification (VI-ReID)
    64	[5] has emerged as a pivotal research direction, addressing the
    65	significant modality gap between visible (VIS) and infrared
    66	(IR) imaging. Current VI-ReID methods can be broadly categorized into two paradigms: 1) Image–level methods employ
    67	generative models [6], [7], [8], [9], [10] or basic transformations [11], [12], [13], [14], [15], [16] to bridge the modality
    68	gap through input-space alignment. However, generative models suffer from training instability and detail degradation, while
    69	simple transformations prove inadequate for handling complex cross-modality variations. 2) Feature-level methods focus
    70	on architectural innovations and specialized loss functions.
    71	ResNet50 [17] remains the predominant backbone in VI-ReID
    72	research, while emerging architectures like Vision Transformers (ViTs) [18] demonstrate comparable efficacy [19], [20],
    73	[21]. Both frameworks achieve satisfactory performance, with
    74	ResNet50-based methods dominating current implementations
    75	and ViT methods showing growing promise for cross-modality
    76	scenarios, especially with additional multimodal information
    77	[22], [23]. For loss function design, in addition to the crossentropy loss [24] that implicitly aligns modalities, specialized
    78	cross-modality losses [1], [25], [26] explicitly minimize intraclass distances while maximizing inter-class separability and
    79	mitigating modality discrepancies. In practice, compared with
    80	image-level methods, feature-level methods tend to deliver
    81	stronger recognition performance. However, most existing
    82	feature-level methods predominantly focus on spatial-domain
    83	features, leaving the frequency domain insufficiently leveraged
    84	for modality-invariant learning.
    85	Recent advances in frequency domain analysis have
    86	demonstrated remarkable success across various vision tasks,
    87	including image deraining [27], denoising [28], and low-light
    88	enhancement [29]. As shown in Fig. 2, Fourier analysis reveals that amplitude spectra (Fig. 2(b)) primarily
    89	encode modality-specific characteristics, while phase spectra
    90	(Fig. 2(c)) preserve structural information across modalities. These findings have motivated several frequency-based
    91	VI-ReID methods. FDMNet [30] and FDNM [31] focus on
    92	amplitude spectrum alignment while preserving phase information. In addition, DSSF3 [32] advocates joint learning of
    93	both spectral components. These methods perform broadspectrum feature mining across the entire frequency band (top
    94	left of Fig. 1), but overlook the distinct roles of low/highfrequency in modality-invariant feature learning.
    95	
    96	1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
    97	similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
    98	See https://www.ieee.org/publications/rights/index.html for more information.
    99	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.
   100	
   101	GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
   102	
   103	1767
   104	
   105	Fig. 1. The comparison of existing frequency-based methods and our proposed
   106	method. Existing methods either perform broad-spectrum feature mining
   107	across the entire frequency band or rely on simple low/high-frequency decomposition, which often adopts coarse and inefficient feature learning in the
   108	spatial domain. Conversely, our method achieves efficient low/high-frequency
   109	feature learning through the meticulously designed HiLo-FM module
   110	(Sec. III-B). Meanwhile, the FADE module (Sec. III-C) leverages multifrequency attention to generate richer, more robust, and modality-invariant
   111	features.
   112	
   113	Further decomposition into low-frequency (Fig. 2(d))
   114	and high-frequency (Fig. 2(e)) components provides deeper
   115	insights. The low-frequency component, capturing the overall
   116	shape of a person, is heavily influenced by modality-specific
   117	information, such as color and illumination in VIS images
   118	or thermal radiation in IR images. In contrast, the highfrequency component effectively filters out modality-specific
   119	styles, preserving shared structural details like edges and contours, thereby making the two modalities appear more similar
   120	[19]. Notable methods for low/high-frequency decomposition
   121	include FSDF [33] and BiFFN [34] (top right of Fig. 1).
   122	FSDF leverages discrete cosine transform in conjunction with
   123	fundamental 1 × 1 convolutional operators, whereas BiFFN
   124	integrates wavelet transforms with resource-demanding Graph
   125	Neural Networks (GNNs) [35]. Although these two methods
   126	have demonstrated certain effectiveness, two critical challenges
   127	persist. First, they predominantly execute low/high-frequency
   128	feature learning in the spatial domain, thereby neglecting
   129	the potential for efficient frequency-domain processing. Second, and more critically, they rely on limited frequency
   130	decompositions, focusing solely on low/high-frequency components. This limitation is particularly suboptimal, as our
   131	visual analysis in Fig. 2 reveals. While low-frequencies
   132	(Fig. 2(d)) are dominated by modality-specific styles and highfrequencies (Fig. 2(e)) better preserve shared structures, a
   133	person’s identity is inherently multi-scale. Crucial cues, from
   134	overall posture to fine-grained textures (e.g., the logo on
   135	
   136	Fig. 2. The visualization of frequency components for a pair of VIS (odd
   137	rows) and IR (even rows) images. The figure shows the (a) original images
   138	and the reconstructions from (b) amplitude (A), and (c) phase (P). The lowfrequency (L) and high-frequency (H) components, shown in (d) and (e), are
   139	separated using a frequency mask as detailed in Eq.(4). The multi-frequency
   140	decompositions in (f) are generated by partitioning the spectrum into multiple
   141	non-overlapping bands, formally defined in Eq.(17). This visualization highlights how modality-invariant cues are distributed across different frequency
   142	bands, motivating our method.
   143	
   144	the person’s clothing), are scattered across various frequency
   145	bands, with their prominence varying between modalities,
   146	as shown in the multi-frequency decompositions (Fig. 2(f)).
   147	Relying on a single, handcrafted frequency cutoff thus forces
   148	a poor trade-off, either discarding valuable identity cues or
   149	retaining excessive modality-specific noise. This underscores
   150	the necessity of a more flexible, multi-frequency analysis to
   151	capture these scattered yet discriminative features.
   152	Motivated by the aforementioned findings, we propose the
   153	Multi-Frequency Embedding Network (MFENet), a featurelevel method that learns modality-invariant representations
   154	from the frequency domain via multi-frequency decompositions. Specifically, to address the challenge of efficient
   155	low/high-frequency feature extraction, we introduce the HiLoFrequency Modulation (HiLo-FM) module. This module
   156	employs the more flexible Fast Fourier Transform (FFT)
   157	[36] for low/high-frequency decomposition, directly performs
   158	efficient filtering on low-frequency features in the frequency
   159	domain, and conducts lightweight multi-scale feature mining
   160	
   161	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.
   162	
   163	1768
   164	
   165	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   166	
   167	on high-frequency features in the spatial domain. To mitigate
   168	unavoidable noise, these low/high-frequency features are then
   169	fused using spatial-attention-based modulation. To achieve
   170	multi-frequency feature learning, we propose the FrequencyAware Diversity Enhancer (FADE) module, which can be
   171	viewed as an extension of Instance Normalization (IN) [37]
   172	in the frequency domain. By weighting multiple pre-set frequencies from low to high, the module accentuates feature
   173	differences across bands and learns rich modality-invariant
   174	features with a multi-branch structure. Additionally, considering the cross-modality retrieval characteristics of the VI-ReID
   175	task, we propose two loss functions to further improve the
   176	robustness and diversity of cross-modality features: CrossModality Soft Retrieval (CMSR) loss and Cross-Modality
   177	Ranking Regularization (CMRR) loss. The CMSR loss is
   178	specifically devised to accentuate cross-modality matching. It
   179	enforces that the maximum inter-modality distance between
   180	instances of the same class should be smaller than the
   181	minimum intra-modality distance between instances of the
   182	same class. This mechanism directs the focus more towards
   183	cross-modality features rather than intra-modality features.
   184	Meanwhile, the CMRR loss is dedicated to augmenting the
   185	diversity of cross-modality features. It achieves this by maximizing the inconsistency in ranking among features extracted
   186	by different branches in the FADE module. To tackle the
   187	challenge posed by non-differentiable ranking, a differentiable
   188	Spearman rank function is incorporated.
   189	The contributions of our work are summarized as follows:
   190	• We propose MFENet, a feature-level method that operates
   191	in the frequency domain and introduces multi-frequency
   192	decompositions to learn robust modality-invariant features
   193	for VI-ReID. This method incorporates two key modules: the HiLo-FM module, which efficiently extracts and
   194	fuses low/high-frequency features, and the FADE module, which effectively enhances feature discriminability
   195	through diverse frequency responses.
   196	• We propose two novel loss functions for cross-modality
   197	retrieval tasks: CMSR loss, which enhances the robustness of cross-modality features by focusing more
   198	on cross-modality features learning rather than intramodality features learning, and CMRR loss, which
   199	promotes the diversity of cross-modality features by
   200	maximizing the inconsistency in ranking among features
   201	extracted by different branches.
   202	• We conduct extensive experiments on VI-ReID benchmarks SYSU-MM01, RegDB, and LLCM, which validate
   203	the effectiveness of our designed modules and loss functions, and demonstrate that MFENet outperforms most
   204	state-of-the-art methods.
   205	II. R ELATED W ORK
   206	A. Visible-Infrared Person Re-Identification
   207	The field of VI-ReID [5] enables 24-hour surveillance capabilities by matching cross-modality images, yet it is confronted
   208	with significant modality discrepancies between visible and
   209	infrared images. To address these disparities, existing methods
   210	primarily focus on two aspects:
   211	
   212	1) Image-level methods leverage Generative Adversarial
   213	Networks (GANs) [38] and Diffusion Models (DMs) [39], or
   214	employ basic operations and modules to enrich input images
   215	and bridge the VIS-IR gap. Although GAN-based methods [6],
   216	[7], [8] are widely used, they generally exhibit inferior image
   217	generation quality when compared to DM-based methods [9],
   218	[10]. Moreover, generative models typically require substantial
   219	computational resources and time, and are prone to introducing
   220	noise. Some methods either design intuitive enhancement
   221	operations [14], [15], [16] or utilize simple modules [11], [13]
   222	for intermediate modality generation. However, these methods
   223	often rely heavily on domain-specific knowledge, are overly
   224	simplistic, and struggle to adapt to the complex scenarios
   225	inherent in VI-ReID tasks effectively.
   226	2) Feature-level methods concentrate on developing
   227	advanced network architectures or loss functions. The majority
   228	of existing works utilize ResNet50 [17] as the backbone
   229	network, and enhance performance through attention mechanisms [1], local feature mining [40], multi-scale learning [41],
   230	and high-order structure [25]. These methods have achieved
   231	promising results. With the advent of ViTs [18] in recent years,
   232	numerous studies have explored ViT-based architectures [19],
   233	[20], [21], [23], [40], [42]. Despite their good performance,
   234	these architectures still lag behind ResNet50 under the same
   235	ImageNet pre-trained model settings. Their high computational
   236	complexity and large number of parameters also limit their
   237	application in scenarios with limited computational resources.
   238	In terms of loss function design, in addition to the commonly
   239	used cross-entropy loss [24], various loss functions specifically
   240	designed for cross-modality scenarios [1], [25], [26] have
   241	been employed. These functions aim to minimize intra-class
   242	distances, maximize inter-class distances, and account for the
   243	differences between modalities.
   244	In practice, feature-level methods tend to be more stable
   245	and computationally economical than image-level methods.
   246	Nevertheless, the vast majority of feature-level VI-ReID methods still learn in the spatial domain, and thus under-utilize
   247	frequency-domain cues that can be modality-invariant. In this
   248	work, we advance the feature-level line by explicitly operating
   249	in the frequency domain, conducting multi-frequency feature
   250	learning to mine robust and diverse modality-invariant representations.
   251	B. Frequency-Domain Analysis in Deep Learning
   252	Frequency-domain analysis [43], a subfield of image
   253	processing, converts spatial domain images into frequency
   254	features to uncover concealed patterns within image data.
   255	Unlike spatial-domain analysis, which directly manipulates
   256	pixel values, frequency-domain analysis dissects images into
   257	various frequency components, thereby facilitating the differentiation and manipulation of subtle features such as noise and
   258	texture. Recent advancements in frequency-domain analysis
   259	have achieved remarkable success across a wide range of
   260	vision tasks, including image deraining [27], denoising [28],
   261	and low-light enhancement [29].
   262	In the field of VI-ReID, several recent works [30], [31], [32],
   263	[33], [34] have initiated the exploration of frequency-domain
   264	feature learning for VI-ReID tasks. Notably, FDNM [31] and
   265	
   266	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.
   267	
   268	GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
   269	
   270	FDMNet [30] adopt a strategy that preserves phase features
   271	while conducting feature learning on the amplitude. In contrast, DSSF3 [32] posits that both phase and amplitude features
   272	are crucial for effective feature extraction. However, these pioneering works have largely overlooked the frequency-domain
   273	decomposition, thereby neglecting the distinct contributions of
   274	low/high-frequency components to modality-invariant feature
   275	learning [19].
   276	To address these limitations, recent works have attempted
   277	to decouple high/low-frequency branches for more robust
   278	feature extraction. For instance, FSDF [33] employs the
   279	discrete cosine transform to decompose low/high-frequency
   280	components and utilizes simple 1 × 1 convolution for feature
   281	extraction in the spatial domain. However, this approach has
   282	certain limitations in feature extraction capabilities. Similarly, BiFFN [34] adopts the discrete wavelet transform for
   283	low/high-frequency decomposition and processes these features in the spatial domain using computationally intensive
   284	GNNs [35]. Although both methods aim to extract more robust
   285	low/high-frequency features, they both still perform feature
   286	extraction in the spatial domain, and the adopted extraction
   287	modules are either too simplistic or overly complex.
   288	In this work, as Fig. 1 illustrates, we first achieve efficient
   289	low/high-frequency feature learning and fusion through meticulously designed spatial/frequency operations. Subsequently,
   290	we leverage multi-frequency attention to generate richer, more
   291	robust, and modality-invariant features.
   292	
   293	III. M ETHODOLOGY
   294	A. Overall Framework
   295	In the task of VI-ReID [1], the dataset is comprised of
   296	images captured across both the visible and infrared modalities. The dataset is formally represented as {I(M) ∈ Rc×h×w |
   297	M ∈ {V, I}}, where I(M) denotes images from either the visible
   298	(V) or infrared (I) modality. The c, h, and w correspond to
   299	the number of channels, height, and width of the images,
   300	respectively.
   301	PK
   302	As illustrated in Fig. 3(a), the input images {I(V)
   303	i }i=1 and
   304	(I) PK
   305	{Ii }i=1 , consisting of P identities with K visible images
   306	and K infrared images per identity, respectively, are initially
   307	processed by the shared layer0∼3 of ResNet-50 [17]. Considering that low frequencies typically represent the overall
   308	style of an image, while high frequencies are often modalityagnostic and contain detailed features [44], a HiLo-Frequency
   309	Modulation (HiLo-FM) module (as shown in Fig. 3(b), in
   310	Sec. III-B) is introduced in layer2 and layer3 to perform
   311	modality-invariant feature learning on low/high-frequency features, respectively. Inspired by Instance Normalization [26],
   312	[30], [37], a Frequency-Aware Diversity Enhancer (FADE)
   313	module (as shown in Fig. 3(c), in Sec. III-C) is incorporated at the end of layer3 . This module further explores rich
   314	and diverse modality-invariant features by applying attention
   315	weighting across multi-frequency bands from low to high,
   316	leveraging subsequent N non-shared layer4 branch structures.
   317	(V)
   318	(I)
   319	This process yields features F(M)
   320	n,i = {Fn,i , Fn,i }, where n =
   321	1, . . . , N; i = 1, . . . , PK. Subsequently, for the n-th branch, the
   322	
   323	1769
   324	
   325	final image embeddings are extracted by performing global
   326	pooling and batch normalization operations:
   327	
   328	
   329	
   330	
   331	(M)
   332	(M)
   333	(M)
   334	fn,i
   335	= GeM F(M)
   336	,
   337	f̃
   338	=
   339	BN
   340	f
   341	,
   342	(1)
   343	n,i
   344	n,i
   345	n,i
   346	(M) (M)
   347	where fn,i
   348	/f̃n,i ∈ RD , and D denotes the feature dimension.
   349	GeM represents Generalized Mean Pooling [45], and BN
   350	represents Batch Normalization [46].
   351	During the training phase, in addition to the basic identity
   352	loss and weighted regularization triplet loss [1], a CrossModality Soft Retrieval (CMSR) loss and a Cross-Modality
   353	Ranking Regularization (CMRR) loss are specifically designed
   354	to address the characteristics of cross-modality retrieval in
   355	VI-ReID (in Sec. III-D). In the testing phase, the concatenation
   356	(M) (M)
   357	(M) (M)
   358	of [f1,i
   359	; f̃1,i ; · · · ; fN,i
   360	; f̃N,i ] is used as the final feature.
   361	
   362	B. HiLo-Frequency Modulation Module
   363	Inspired by Fig.2(d,e), explicitly decomposing features
   364	into low/high frequencies helps isolate modality-specific
   365	style and illumination (predominantly low-frequency) from
   366	identity-bearing structures and textures (predominantly highfrequency), which facilitates cross-modality alignment in
   367	VI-ReID. Unlike prior frequency-based methods that merely
   368	split into low/high bands and then learn mainly in the
   369	spatial domain with coarse operators or fixed filters, our
   370	HiLo-Frequency Modulation (HiLo-FM) module (Fig. 3(b))
   371	performs learnable frequency-domain filtering to efficiently
   372	normalize low-frequency style, enhances high-frequency identity cues via lightweight multiscale convolutions in the spatial
   373	domain, and fuses the two with spatial attention, suppressing noise while preserving discriminative, modality-invariant
   374	information. The module operates through four key stages:
   375	1) HiLo-Frequency Decomposition: Given an input feature
   376	map X ∈ RC×H×W , we first transform it to the frequency
   377	domain using the FFT [36]:
   378	X̃ = F(X) ∈ CC×H×W ,
   379	
   380	(2)
   381	
   382	where F denotes the 2D real-valued channel-wise FFT, which
   383	can be formally defined as:
   384	F(X)u,v = √
   385	
   386	1
   387	
   388	H−1 W−1
   389	X
   390	X
   391	
   392	HW h=0 w=0
   393	
   394	Xh,w e−2 jπ( H u+ W v) ,
   395	h
   396	
   397	w
   398	
   399	(3)
   400	
   401	where j is the imaginary unit. u ∈ [0, H − 1] and v ∈ [0, W − 1]
   402	are the spatial frequencies in the horizontal and vertical
   403	directions, respectively. Accordingly, F −1 (X) represents the
   404	inverse Fast Fourier transform (IFFT) of the feature X.
   405	In the frequency domain, a low-frequency binary mask Π ∈
   406	{0, 1}H×W is predefined as:
   407	(
   408	1, max (u/H, v/W) ⩽ ρ
   409	Πu,v =
   410	(4)
   411	0, max (u/H, v/W) > ρ
   412	where the hyperparameter ρ ∈ [0, 1] balances low/highfrequency components, and a square-shaped method is used to
   413	separate these components. Alternatively, circular or diamondshaped methods could be applied, and subsequent ablation
   414	studies are conducted to compare these three methods. Accordingly, the high-frequency binary mask can be obtained by
   415	
   416	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.
   417	
   418	1770
   419	
   420	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   421	
   422	Fig. 3. Overall framework of Multi-Frequency Embedding Network (MFENet). (a) The high-level architecture, showing the integration of our novel modules
   423	and loss functions. (b, c) Detailed schematics of our core technical innovations: the HiLo-Frequency Modulation (HiLo-FM) and Frequency-Aware Diversity
   424	Enhancer (FADE) modules. (d, e) Illustrations of the key principles behind our novel Cross-Modality Soft Retrieval (CMSR) and Cross-Modality Ranking
   425	Regularization (CMRR) loss functions.
   426	
   427	Π̄ = 1 − Π, then the low/high-frequency components can be
   428	decomposed as:
   429	X̃L = X̃
   430	
   431	Π, X̃H = X̃
   432	
   433	Π̄,
   434	
   435	(5)
   436	
   437	where denotes the element-wise multiplication.
   438	2) Low-Frequency Adaptation: Given that low-frequency
   439	components carry global style information, employing smallkernel convolutions in the spatial domain is insufficient for
   440	capturing long-range dependencies. Meanwhile, Transformers
   441	[18] exhibit prohibitive quadratic computational complexity,
   442	especially in shallow network layers. Drawing inspiration from
   443	GFNet [47], we propose to enhance low-frequency features
   444	directly in the frequency domain by introducing a parameterized complex weight matrix W̃L ∈ CC×H×W :
   445	X̂L = X̃L
   446	
   447	W̃L .
   448	
   449	(6)
   450	
   451	Compared to GFNet, our approach achieves a significant
   452	reduction in both parameter count and computational complexity. Specifically, the matrix X̃L contains non-zero values
   453	exclusively within its central region, which is determined
   454	by the low-frequency mask. Additionally, the real-valued 2D
   455	FFT inherently reduces the dimensionality by half compared
   456	to retaining the full complex frequency domain size. As a
   457	result, the effective parameter count of W̃L is reduced to
   458	
   459	CbρHcbρ(W/2 + 1)c. Finally, we transform the adapted lowfrequency component back to the spatial domain via iFFT:
   460	XL = F −1 (X̂L ).
   461	
   462	(7)
   463	
   464	3) High-Frequency Enhancement: High-frequency features, which capture various scales of details and contours of
   465	persons, are crucial for extracting modality-invariant features
   466	and are well-suited for feature extraction in the spatial domain
   467	via convolutions. As shown in Fig. 2, a single 3×3 convolution
   468	is insufficient for extracting multi-scale features. To address
   469	this, we design the Lite Multiscale (LM) module (as shown
   470	in Fig. 4), inspired by OSNet [48], for efficient multi-scale
   471	feature learning.
   472	First, we obtain the spatial domain features X̂H =
   473	F −1 (X̃H ) ∈ RC×H×W . To integrate the sparse high-frequency
   474	features across channels, we use a 1×1 pointwise convolution
   475	(with batch normalization and ReLU activation) to reduce the
   476	dimensionality of X̂H :
   477	
   478	X̄H = Conv1×1 X̂H ∈ RC/r×H×W ,
   479	
   480	(8)
   481	
   482	where r is the dimensionality reduction factor. We then construct multi-scale convolutions by stacking different numbers
   483	
   484	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.
   485	
   486	GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
   487	
   488	1771
   489	
   490	where Conv3×3 denotes a convolution with a kernel size of
   491	3 × 3, which reduces the number of input channels from C to
   492	(s)
   493	2, corresponding to the spatial masks M(s)
   494	L /MH . To stabilize
   495	training, the final output combines the fused features with the
   496	original input via a residual connection and zero-initialized
   497	batch normalization:
   498	
   499	X̂ = X + BN X0 .
   500	(15)
   501	In summary, the HiLo-FM module effectively integrates
   502	comprehensive frequency domain processing, efficient feature enhancement, and spatial-attention-based modulation
   503	to achieve robust modality-invariant feature learning for
   504	VI-ReID, offering significant advantages over existing
   505	methods.
   506	C. Frequency-Aware Diversity Enhancer Module
   507	Fig. 4. The architecture of the Lite Multiscale (LM) module for HiLo-FM.
   508	The batch normalization and ReLU are omitted for simplicity, and the specific
   509	details are elaborated in the main text.
   510	
   511	of lite 3 × 3 convolutions [48]. For scale s ∈ {1, 2, · · · , S }, the
   512	resulting features are:
   513	
   514	s
   515	X̄(s)
   516	LiteConv3×3
   517	X̄H ,
   518	(9)
   519	H =
   520	„ ƒ‚ …
   521	DWConv3×3 (Conv1×1 (·))
   522	s
   523	where LiteConv3×3
   524	denotes s lite 3 × 3 convolutions, and
   525	DWConv3×3 is followed by batch normalization and ReLU
   526	S
   527	activation. To fuse the features from S scales {X̄(s)
   528	H } s=1 , we
   529	use a shared efficient aggregation gate (EAG) that employs a
   530	one-dimensional convolution in the channel dimension, thus
   531	making it lightweight and efficient. The final features are
   532	obtained by summing the weighted features and restoring the
   533	original channel dimension using a 1×1 pointwise convolution
   534	(with batch normalization and ReLU activation):
   535	 3
   536	2
   537	
   538	(s)
   539	avgpooling
   540	X̄
   541	 H  5 ∈ R2×C/r ,
   542	4
   543	(10)
   544	X̄0(s)
   545	H =
   546	max pooling X̄(s)
   547	H
   548	
   549	
   550	
   551	w(s) = σ Conv5 X̄0(s)
   552	∈ R1×C/r ,
   553	(11)
   554	H
   555	!
   556	S
   557	X
   558	XH = Conv1×1
   559	w(s) X̄(s)
   560	∈ RC×H×W ,
   561	(12)
   562	H
   563	s=1
   564	
   565	where Conv5 is a one-dimensional convolution with a kernel
   566	size of 5 [49], and σ is the sigmoid activation function.
   567	4) Spatial-Attention Based Modulation and Fusion: Based
   568	on Eq.(7) and (12), the processed low/high-frequency features
   569	can be obtained respectively. Considering the inevitable noise
   570	introduced in the low/high-frequency features, spatial masks
   571	(s)
   572	H×W
   573	M(s)
   574	are generated from X to modulate the
   575	L /MH ∈ R
   576	low-frequency feature XL and the high-frequency feature XH
   577	spatially. The modulated low/high-frequency features are then
   578	fused directly by addition:
   579	h
   580	i
   581	
   582	(s)
   583	M(s)
   584	= σ Conv3×3 (X) ,
   585	(13)
   586	H ; ML
   587	X0 = M(s)
   588	H
   589	
   590	XH + M(s)
   591	L
   592	
   593	XL ,
   594	
   595	(14)
   596	
   597	Instance Normalization (IN) [37] has gained widespread
   598	adoption for suppressing modality-specific features by normalizing frequency components: zeroing the DC component while
   599	scaling others via standard deviation:
   600	(
   601	(u, v) = (0, 0)
   602	0,
   603	,
   604	(16)
   605	|F (IN (X))| =
   606	otherwise
   607	|F (X)| /σ (X) ,
   608	where |F (X)| denotes the amplitude of F(X), and σ(X)
   609	represents the standard deviation of the input feature map X.
   610	To prevent the elimination of some modality-invariant features,
   611	existing methods [26], [30] fuse original and IN-processed
   612	features.
   613	IN-style designs effectively impose a coarse “DC vs.
   614	others” weighting, which can wash out some modality-specific
   615	content. Inspired by the multi-frequency decompositions
   616	in Fig. 2(f), we introduce the Frequency-Aware Diversity
   617	Enhancer (FADE) (Fig. 3(c)) to learn frequency-band–specific
   618	importance and to encourage diverse yet complementary representations. The module operates through three key stages:
   619	1) Multi-Frequency Decomposition: First, the amplitude
   620	X̃amp = |F (X)| is decomposed into B non-overlapping regions
   621	B
   622	{Rb }b=1
   623	according to frequency levels:
   624	Rb = {(u, v) | (b − 1) /B ⩽ max (u/H, v/W) < b/B} ,
   625	
   626	(17)
   627	
   628	where H and W denote the height and width of the
   629	input feature map X, respectively. It should be noted
   630	that the decomposition here is performed in square-shaped
   631	regions, consistent with the HiLo-frequency decomposition in
   632	Sec. III-B. Other decomposition methods are also verified in
   633	the subsequent ablation studies.
   634	Then, the amplitudes within the same region are subjected to avgpooling and concatenation (labeled as “P&C” in
   635	Fig. 3(c)):
   636	
   637	
   638	X̃v = X̃v,1 ; . . . ; X̃v,B ,
   639	(18)
   640	where
   641	X̃v,b =
   642	
   643	1 X
   644	X̃amp (u, v),
   645	|Rb |
   646	
   647	(19)
   648	
   649	(u,v)∈Rb
   650	
   651	[·; ·] denotes concatenation, X̃v ∈ RC×B , and C is the number
   652	of channels of X.
   653	
   654	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.
   655	
   656	1772
   657	
   658	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   659	
   660	2) Frequency-Aware Attention: Next, a lightweight
   661	frequency-aware module is employed to generate attention
   662	weights for each region Rb :
   663	
   664	M1( f ) = σ Conv1 X̃v ∈ RB ,
   665	(20)
   666	where σ is the sigmoid activation function, and Conv1 denotes
   667	a one-dimensional convolution with a kernel size of 1, which
   668	can be viewed as performing point-wise convolution across
   669	the regions. This approach is not only more efficient than
   670	directly convolving on X̃amp but also fully leverages the
   671	spatial characteristics of frequencies from low to high. The
   672	convolution reduces the number of channels from C to 1,
   673	and the final output M(1f ) is a B-dimensional attention weight.
   674	Using this attention weight, we can weight the amplitude
   675	features of each region:
   676	(f)
   677	X̃(1)
   678	amp (u, v) = M1,b × X̃amp (u, v),
   679	
   680	(21)
   681	
   682	where (u, v) ∈ Rb .
   683	3) Diversity Enhancer: To increase feature diversity, N
   684	parallel attention branches (Eq.(20)) generate distinct attention
   685	N
   686	. Subsequently, through Eq.(21), each branch
   687	weights {Mn( f ) }n=1
   688	N
   689	obtains N different {X̃(n)
   690	amp }n=1 , which are combined with the
   691	phase X̃ pha and then inverse-transformed using iFFT to yield
   692	N
   693	N distinct features {X0(n) }n=1
   694	:
   695	
   696	0(n)
   697	−1
   698	X =F
   699	X̃(n)
   700	(22)
   701	amp exp jX̃ pha .
   702	To ensure the stability of optimization, batch normalization
   703	(BN) is introduced in each branch:
   704	
   705	(23)
   706	X̂(n) = BNn X0(n) .
   707	In summary, the FADE module provides superior frequency
   708	adaptability compared to conventional IN-based methods,
   709	enabling more precise preservation of modality-invariant characteristics across different frequency bands.
   710	D. Objective Function
   711	Following the AGW [1], we adopt the identity loss LID
   712	and weighted regularization triplet (WRT) loss LWRT as our
   713	baseline objectives. However, the conventional WRT loss
   714	does not explicitly handle the cross-modality characteristics
   715	in VI-ReID. To enhance the robustness and diversity of crossmodality features, we propose two novel loss:
   716	1) Cross-Modality Soft Retrieval Loss: VI-ReID is a crossmodality retrieval task, so training objectives that only tighten
   717	intra-modality positives can mismatch the test protocol and
   718	leave a residual modality gap. To align training with evaluation, we propose the Cross-Modality Soft Retrieval (CMSR)
   719	loss, which aims to enforce that the maximum inter-modality
   720	distance between same-class instances should be smaller
   721	than the minimum intra-modality distance between sameclass instances. Formally, given visible feature embeddings
   722	PK
   723	PK
   724	{fi(V) }i=1
   725	and the infrared feature embeddings {fi(I) }i=1
   726	, the maximum inter-modality distance between same-class instances
   727	and the minimum intra-modality distance between same-class
   728	instances are defined as (taking V → I as an example, and
   729	I → V is the reverse):
   730	(V,I)
   731	di,max
   732	= max di(V,I)
   733	j ,
   734	y j =yi
   735	
   736	(24)
   737	
   738	(V,V)
   739	,
   740	di,min
   741	= min di(V,V)
   742	j
   743	y j =yi
   744	
   745	(25)
   746	
   747	(M ,M )
   748	
   749	where di j i j denotes the euclidean distance between the i-th
   750	feature of modality Mi and the j-th feature of modality M j .
   751	yi represents the person category of the i-th feature. With the
   752	introduction of the margin m > 0, the cross-modality hard
   753	retrieval loss can be defined as:
   754	PK
   755	i
   756	1 X h (V,I)
   757	(V,V)
   758	(V,I)
   759	di,max − di,min
   760	+m ,
   761	(26)
   762	LCMHR
   763	=
   764	+
   765	PK
   766	i=1
   767	
   768	where [x]+ denotes max(0, x). This hard retrieval loss
   769	approach only considers the maximum and minimum sample
   770	pairs, ignoring the effects of the remaining sample pairs.
   771	Therefore, inspired by the WRT loss [1], we replace the max
   772	and min operations in Eq.(24) and Eq.(25) with softmax and
   773	softmin operations [50]:
   774	
   775	
   776	exp di(V,I)
   777	/γ
   778	X
   779	j
   780	(V,I)
   781	
   782	 di(V,I)
   783	,
   784	(27)
   785	d̃i,max
   786	=
   787	j
   788	P
   789	(V,I)
   790	exp
   791	d
   792	/γ
   793	y j =yi
   794	yk =yi
   795	ik
   796	
   797	
   798	exp −di(V,V)
   799	/γ
   800	X
   801	j
   802	(V,V)
   803	
   804	 di(V,V)
   805	,
   806	(28)
   807	d̃i,min
   808	=
   809	j
   810	P
   811	(V,V)
   812	exp
   813	−d
   814	/γ
   815	y j =yi
   816	yk =yi
   817	ik
   818	where γ controls the degree of smoothing. Substituting these
   819	soft operators into Eq.(26), we obtain the cross-modality soft
   820	retrieval loss:
   821	PK
   822	i
   823	1 X h (V,I)
   824	(V,I)
   825	(V,V)
   826	LCMS
   827	=
   828	d̃i,max − d̃i,min
   829	(29)
   830	+m .
   831	R
   832	+
   833	PK
   834	i=1
   835	
   836	Considering both V → I and I → V, the final cross-modality
   837	soft retrieval loss is:
   838	
   839	1  (V,I)
   840	(I,V)
   841	LCMS R =
   842	(30)
   843	LCMS R + LCMS
   844	R .
   845	2
   846	(V,I)
   847	(V,V)
   848	It is worth noting that when γ → ∞, d̃i,max
   849	and d̃i,min
   850	correspond to the mean distances between the anchor sample
   851	and other samples, making LCMS R equivalent to L MS EL [20].
   852	Conversely, when γ → 0, they correspond to the maximum and
   853	minimum distances, respectively, making LCMS R equivalent to
   854	LCMR [15]. Thus, our proposed cross-modality soft retrieval
   855	loss can be seen as a compromise between MSEL and CMR.
   856	2) Cross-Modality Ranking Regularization Loss: To
   857	enhance the diversity of features obtained by the FADE
   858	module, we propose a Cross-Modality Ranking Regularization
   859	(CMRR) loss based on differentiable Spearman rank
   860	correlation, starting from the inconsistency in the ranking of
   861	cross-modality positive pairs. Given N sets of visible features
   862	(V)
   863	{fn,i
   864	|i = 1, . . . , PK; n = 1, . . . , N} and N sets of infrared
   865	(I)
   866	features {fn,i
   867	|i = 1, . . . , PK; n = 1, . . . , N}, we obtain the
   868	distance list between features of the same class but different
   869	modalities within the n-th set (taking V → I as an example,
   870	and I → V is the reverse):
   871	h
   872	i
   873	(V,I)
   874	(V,I)
   875	(V,I)
   876	d(V,I)
   877	(31)
   878	n,i = dn,i, j1 , dn,i, j2 , . . . , dn,i, jK ,
   879	
   880	where dn,i, jk denotes the Euclidean distance from the i-th
   881	sample to the jk -th sample in the n-th set of features, and
   882	
   883	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.
   884	
   885	GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
   886	
   887	K is the number of samples of the same class within the same
   888	modality in a minibatch. Based on the above distance list, we
   889	can obtain the ranking list of features within the set:
   890	i
   891	h
   892	(V,I)
   893	(V,I) (V,I)
   894	,
   895	.
   896	.
   897	.
   898	,
   899	r
   900	(32)
   901	,
   902	r
   903	r(V,I)
   904	=
   905	r
   906	n,i,K ,
   907	n,i
   908	n,i,1 n,i,2
   909	where rn,i,k ∈ [1, K] indicates the index corresponding to the
   910	k-th smallest distance in the distance list, that is:
   911	(V,I)
   912	(V,I)
   913	(V,I)
   914	dn,i,
   915	j (V,I) ⩽ dn,i, j (V,I) ⩽ · · · ⩽ dn,i, j (V,I) .
   916	r
   917	n,i,1
   918	
   919	r
   920	
   921	(33)
   922	
   923	rn,i,K
   924	
   925	n,i,2
   926	
   927	Based on the ranking lists of two different sets, the Spearman
   928	rank correlation coefficient can be defined as:
   929	D
   930	E
   931	(V,I) (V,I)
   932	(V,I)
   933	r(V,I)
   934	−
   935	r̄
   936	,
   937	r
   938	−
   939	r̄
   940	n,i
   941	n,i
   942	m,i
   943	m,i
   944	(V,I)
   945	, (34)
   946	S R(r(V,I)
   947	n,i , rm,i ) =
   948	(V,I)
   949	(V,I)
   950	(V,I)
   951	rn,i − r̄n,i
   952	rm,i − r̄(V,I)
   953	m,i
   954	2
   955	
   956	2
   957	
   958	where r̄(V,I)
   959	represents the mean index of r(V,I)
   960	n,i
   961	n,i . S R ∈ [−1, 1],
   962	and the smaller this value is, the more inconsistent the ranking
   963	lists of the two sets are, thus achieving the goal of ranking
   964	diversity. Since Eq.(32) is non-differentiable, we introduce
   965	the softrank operation [51] to obtain the differentiable crossmodality ranking regularization loss:
   966	PK
   967	
   968	(V,I)
   969	LCMRR
   970	=
   971	
   972	XX
   973	2
   974	(V,I)
   975	S R(r̃(V,I)
   976	n,i , r̃m,i ),
   977	PKN (N − 1)
   978	n<m
   979	
   980	(35)
   981	
   982	i=1
   983	
   984	where
   985	
   986	
   987	(V,I)
   988	r̃(V,I)
   989	=
   990	softrank
   991	d
   992	.
   993	n,i
   994	n,i
   995	
   996	(36)
   997	
   998	Considering both V → I and I → V, the final cross-modality
   999	ranking regularization loss is:
  1000	LCMRR =
  1001	
  1002	
  1003	1  (V,I)
  1004	(I,V)
  1005	LCMRR + LCMRR
  1006	.
  1007	2
  1008	
  1009	(37)
  1010	
  1011	3) Overall Objective Function: The complete training
  1012	objective combines all above losses:
  1013	Ltotal = LID + LWRT + λ1 LCMS R + λ2 LCMRR ,
  1014	
  1015	(38)
  1016	
  1017	where λ1 and λ2 are hyperparameters controlling the tradeoff between different objectives. We apply LCMS R and LCMRR
  1018	to features both before and after batch normalization (BN).
  1019	For features after BN, we first perform L2 normalization
  1020	before applying these two loss functions. For convenience in
  1021	hyperparameter tuning, we share the same set of hyperparameters for features both before and after BN, except for the
  1022	hyperparameter λ2 of LCMRR , which is set to one-tenth of
  1023	the value used for features before BN by default. For specific
  1024	details, please refer to the code.
  1025	IV. E XPERIMENTS
  1026	A. Datasets and Implementation Details
  1027	Three mainstream public VI-ReID datasets, SYSU-MM01
  1028	[52], RedDB [53] and LLCM [41] are used for experiments.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '293,506p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   293	III. M ETHODOLOGY
   294	A. Overall Framework
   295	In the task of VI-ReID [1], the dataset is comprised of
   296	images captured across both the visible and infrared modalities. The dataset is formally represented as {I(M) ∈ Rc×h×w |
   297	M ∈ {V, I}}, where I(M) denotes images from either the visible
   298	(V) or infrared (I) modality. The c, h, and w correspond to
   299	the number of channels, height, and width of the images,
   300	respectively.
   301	PK
   302	As illustrated in Fig. 3(a), the input images {I(V)
   303	i }i=1 and
   304	(I) PK
   305	{Ii }i=1 , consisting of P identities with K visible images
   306	and K infrared images per identity, respectively, are initially
   307	processed by the shared layer0∼3 of ResNet-50 [17]. Considering that low frequencies typically represent the overall
   308	style of an image, while high frequencies are often modalityagnostic and contain detailed features [44], a HiLo-Frequency
   309	Modulation (HiLo-FM) module (as shown in Fig. 3(b), in
   310	Sec. III-B) is introduced in layer2 and layer3 to perform
   311	modality-invariant feature learning on low/high-frequency features, respectively. Inspired by Instance Normalization [26],
   312	[30], [37], a Frequency-Aware Diversity Enhancer (FADE)
   313	module (as shown in Fig. 3(c), in Sec. III-C) is incorporated at the end of layer3 . This module further explores rich
   314	and diverse modality-invariant features by applying attention
   315	weighting across multi-frequency bands from low to high,
   316	leveraging subsequent N non-shared layer4 branch structures.
   317	(V)
   318	(I)
   319	This process yields features F(M)
   320	n,i = {Fn,i , Fn,i }, where n =
   321	1, . . . , N; i = 1, . . . , PK. Subsequently, for the n-th branch, the
   322	
   323	1769
   324	
   325	final image embeddings are extracted by performing global
   326	pooling and batch normalization operations:
   327	
   328	
   329	
   330	
   331	(M)
   332	(M)
   333	(M)
   334	fn,i
   335	= GeM F(M)
   336	,
   337	f̃
   338	=
   339	BN
   340	f
   341	,
   342	(1)
   343	n,i
   344	n,i
   345	n,i
   346	(M) (M)
   347	where fn,i
   348	/f̃n,i ∈ RD , and D denotes the feature dimension.
   349	GeM represents Generalized Mean Pooling [45], and BN
   350	represents Batch Normalization [46].
   351	During the training phase, in addition to the basic identity
   352	loss and weighted regularization triplet loss [1], a CrossModality Soft Retrieval (CMSR) loss and a Cross-Modality
   353	Ranking Regularization (CMRR) loss are specifically designed
   354	to address the characteristics of cross-modality retrieval in
   355	VI-ReID (in Sec. III-D). In the testing phase, the concatenation
   356	(M) (M)
   357	(M) (M)
   358	of [f1,i
   359	; f̃1,i ; · · · ; fN,i
   360	; f̃N,i ] is used as the final feature.
   361	
   362	B. HiLo-Frequency Modulation Module
   363	Inspired by Fig.2(d,e), explicitly decomposing features
   364	into low/high frequencies helps isolate modality-specific
   365	style and illumination (predominantly low-frequency) from
   366	identity-bearing structures and textures (predominantly highfrequency), which facilitates cross-modality alignment in
   367	VI-ReID. Unlike prior frequency-based methods that merely
   368	split into low/high bands and then learn mainly in the
   369	spatial domain with coarse operators or fixed filters, our
   370	HiLo-Frequency Modulation (HiLo-FM) module (Fig. 3(b))
   371	performs learnable frequency-domain filtering to efficiently
   372	normalize low-frequency style, enhances high-frequency identity cues via lightweight multiscale convolutions in the spatial
   373	domain, and fuses the two with spatial attention, suppressing noise while preserving discriminative, modality-invariant
   374	information. The module operates through four key stages:
   375	1) HiLo-Frequency Decomposition: Given an input feature
   376	map X ∈ RC×H×W , we first transform it to the frequency
   377	domain using the FFT [36]:
   378	X̃ = F(X) ∈ CC×H×W ,
   379	
   380	(2)
   381	
   382	where F denotes the 2D real-valued channel-wise FFT, which
   383	can be formally defined as:
   384	F(X)u,v = √
   385	
   386	1
   387	
   388	H−1 W−1
   389	X
   390	X
   391	
   392	HW h=0 w=0
   393	
   394	Xh,w e−2 jπ( H u+ W v) ,
   395	h
   396	
   397	w
   398	
   399	(3)
   400	
   401	where j is the imaginary unit. u ∈ [0, H − 1] and v ∈ [0, W − 1]
   402	are the spatial frequencies in the horizontal and vertical
   403	directions, respectively. Accordingly, F −1 (X) represents the
   404	inverse Fast Fourier transform (IFFT) of the feature X.
   405	In the frequency domain, a low-frequency binary mask Π ∈
   406	{0, 1}H×W is predefined as:
   407	(
   408	1, max (u/H, v/W) ⩽ ρ
   409	Πu,v =
   410	(4)
   411	0, max (u/H, v/W) > ρ
   412	where the hyperparameter ρ ∈ [0, 1] balances low/highfrequency components, and a square-shaped method is used to
   413	separate these components. Alternatively, circular or diamondshaped methods could be applied, and subsequent ablation
   414	studies are conducted to compare these three methods. Accordingly, the high-frequency binary mask can be obtained by
   415	
   416	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.
   417	
   418	1770
   419	
   420	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
   421	
   422	Fig. 3. Overall framework of Multi-Frequency Embedding Network (MFENet). (a) The high-level architecture, showing the integration of our novel modules
   423	and loss functions. (b, c) Detailed schematics of our core technical innovations: the HiLo-Frequency Modulation (HiLo-FM) and Frequency-Aware Diversity
   424	Enhancer (FADE) modules. (d, e) Illustrations of the key principles behind our novel Cross-Modality Soft Retrieval (CMSR) and Cross-Modality Ranking
   425	Regularization (CMRR) loss functions.
   426	
   427	Π̄ = 1 − Π, then the low/high-frequency components can be
   428	decomposed as:
   429	X̃L = X̃
   430	
   431	Π, X̃H = X̃
   432	
   433	Π̄,
   434	
   435	(5)
   436	
   437	where denotes the element-wise multiplication.
   438	2) Low-Frequency Adaptation: Given that low-frequency
   439	components carry global style information, employing smallkernel convolutions in the spatial domain is insufficient for
   440	capturing long-range dependencies. Meanwhile, Transformers
   441	[18] exhibit prohibitive quadratic computational complexity,
   442	especially in shallow network layers. Drawing inspiration from
   443	GFNet [47], we propose to enhance low-frequency features
   444	directly in the frequency domain by introducing a parameterized complex weight matrix W̃L ∈ CC×H×W :
   445	X̂L = X̃L
   446	
   447	W̃L .
   448	
   449	(6)
   450	
   451	Compared to GFNet, our approach achieves a significant
   452	reduction in both parameter count and computational complexity. Specifically, the matrix X̃L contains non-zero values
   453	exclusively within its central region, which is determined
   454	by the low-frequency mask. Additionally, the real-valued 2D
   455	FFT inherently reduces the dimensionality by half compared
   456	to retaining the full complex frequency domain size. As a
   457	result, the effective parameter count of W̃L is reduced to
   458	
   459	CbρHcbρ(W/2 + 1)c. Finally, we transform the adapted lowfrequency component back to the spatial domain via iFFT:
   460	XL = F −1 (X̂L ).
   461	
   462	(7)
   463	
   464	3) High-Frequency Enhancement: High-frequency features, which capture various scales of details and contours of
   465	persons, are crucial for extracting modality-invariant features
   466	and are well-suited for feature extraction in the spatial domain
   467	via convolutions. As shown in Fig. 2, a single 3×3 convolution
   468	is insufficient for extracting multi-scale features. To address
   469	this, we design the Lite Multiscale (LM) module (as shown
   470	in Fig. 4), inspired by OSNet [48], for efficient multi-scale
   471	feature learning.
   472	First, we obtain the spatial domain features X̂H =
   473	F −1 (X̃H ) ∈ RC×H×W . To integrate the sparse high-frequency
   474	features across channels, we use a 1×1 pointwise convolution
   475	(with batch normalization and ReLU activation) to reduce the
   476	dimensionality of X̂H :
   477	
   478	X̄H = Conv1×1 X̂H ∈ RC/r×H×W ,
   479	
   480	(8)
   481	
   482	where r is the dimensionality reduction factor. We then construct multi-scale convolutions by stacking different numbers
   483	
   484	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.
   485	
   486	GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
   487	
   488	1771
   489	
   490	where Conv3×3 denotes a convolution with a kernel size of
   491	3 × 3, which reduces the number of input channels from C to
   492	(s)
   493	2, corresponding to the spatial masks M(s)
   494	L /MH . To stabilize
   495	training, the final output combines the fused features with the
   496	original input via a residual connection and zero-initialized
   497	batch normalization:
   498	
   499	X̂ = X + BN X0 .
   500	(15)
   501	In summary, the HiLo-FM module effectively integrates
   502	comprehensive frequency domain processing, efficient feature enhancement, and spatial-attention-based modulation
   503	to achieve robust modality-invariant feature learning for
   504	VI-ReID, offering significant advantages over existing
   505	methods.
   506	C. Frequency-Aware Diversity Enhancer Module

codex
Multi-Frequency 的方法抽完了。现在我补查每篇的消融和可视化部分，因为你要的是“证据闭环”，只读方法容易把作者真正说服审稿人的证据漏掉。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "for f in \\
'CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf' \\
'DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf' \\
'Deep intelligent technique for person Re-identification system in surveillance images.pdf' \\
'Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf' \\
'DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf' \\
'Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf'; do
  printf '\\n===== %s =====\\n' \""'$f"
  pdftotext "$f" - 2>/dev/null | rg -n -i '"'ablation|component|visual|visualization|t-sne|effect|parameter|analysis|filter|IQR|divers|frequency|cycle|dependab|Table [0-9IVX]+|Fig\\. [0-9]+' | head -120
 done" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 324ms:
5:Dependability Feature Learning Based on Sample
16:descriptions or pseudo-labels but neglect the dependability of
18:this end, we propose a Dependability Feature Learning based
20:we introduce a dependable text generation method that leverages
21:multimodal large language models to generate diverse texts
22:and further filtrate dependable texts for establishing imagetext matching relationships. Second, we design an Error Sample
23:Filtering Module (ESFM) to eliminate abnormal samples and
25:optimizes the cross-modality dependable feature from center and
37:methods [4], [5], [6], TIReID utilizes diverse text descriptions
60:Fig. 1. (a) Supervised TIReID setting with matched image-text pairs and
62:relationship and identity labels. (c) Description of the proposed dependable
72:for enhancing the diversity of training samples [9], [10].
75:Fig. 1 (a).
83:relationship as shown in Fig. 1 (b). This significantly increases
117:To this end, we propose a Dependability Feature Learning
118:based on Sample Generation (DFLSG) method for unsupervised TIReID. Dependability is solved from two key aspects.
119:On the one hand, we introduce a dependable text generation
122:text samples are uncertain and diverse, we design different
123:prompts to guide two MLLMs to generate diverse text descriptions. Then, the dependable text is selected as the training data
126:the image-text matching relationship as shown in Fig. 1 (c).
128:the training process, we propose an Error Sample Filtering
134:Hence, we utilize the Interquartile Range (IQR) filtering algorithm to identify and filter out abnormal samples. In this way,
135:we refine the clustering results and obtain more dependable
140:to explore cross-modality dependable features from the center
147:distances. The MTJL method effectively aggregates the two
150:dependability pedestrian features.
153:(1) We propose a dependable text generation process that
155:generate and filter accurate descriptions for constructing
159:image-text pairs, which effectively enhances the reliability of identity labels.
161:cross-modality dependability features from the center
203:effectively reducing intra-class variations. Some researchers
210:design an effective method to handle the unsupervised TIReID
225:between visual and textual representations. In addition, some
228:example, Shu et al. [37] propose an Implicit Visual-Textual
233:data diversity, some researchers utilize identity information to
246:visual-textual joint embedding.
248:obtain dependable text descriptions for building the image-text
252:matching relationship. In addition, we design ESFM to eliminate abnormal samples for acquiring dependable identity
253:labels. Finally, we propose MTJL to extract dependability features from center and instance views, which could effectively
258:In this section, we construct a dependability feature learning
261:the overall structure of DFLSG. Then, the dependable text
264:dependability labels. Finally, we introduce MTJL that continuously optimizes intra-modality and inter-modality feature
267:As shown in Fig. 2, we present the overall model structure of
272:dataset, we propose a dependable text generation process.
285:Finally, we construct MTJL to explore dependable feature
287:B. Dependable Text Generation Process
290:Moreover, textual information possesses diversity and uncertainty, which increases the difficulty of obtaining meaningful
296:end, we propose a dependable text generation method to
298:dependability of training data. Specifically, ChatGPT [44] possesses strong text understanding and generation capabilities,
306:Fig. 2. The whole framework of DFLSG. In the dependable text generation phase (a), we leverage the diversity strengths of MLLMs to generate reliable
308:(b) illustrates that the Error Sample Filtering Module (ESFM) is designed to perform the clustering operation and eliminate abnormal samples. Meanwhile, an
313:information. To increase the diversity of text descriptions, we
322:is fed into multiple MLLMs to generate diverse descriptions
324:descriptions is different. As shown in Fig. 1 (c), we present two
330:only possesses rich semantic knowledge but also effectively
338:C. Error Sample Filtering Module
341:Researchers utilize clustering algorithms to assign pseudolabels for different images, which demonstrates the effectiveness of clustering algorithms [11], [12]. However, because the
345:Error Sample Filtering Module (ESFM) to refine the clustering
346:results and obtain dependable image-text pairs with label
349:feature representations. Image typically contains rich visual
390:Fig. 3. Illustration of Error Sample Filtering Module. O
397:by conducting an Interquartile Range (IQR) filtering strategy. O
409:low similarity sample pairs is infeasible. The IQR algorithm
412:filtering abnormal samples in small data distributions. Hence,
413:we employ the IQR filtering algorithm to remove abnormal
414:image-text pairs with low similarities. Fig. 3 illustrates the
418:eliminated. The filtered text feature set Taf ilter is represented
426:where β is a hyperparameter that controls the filtering strength,
430:to search filtered image features. In this way, we construct a
431:dependable dataset of image-text pairs.
449:dependable feature distribution from center and instance views
452:loss is effective for supervised TIReID task, which could
559:where τ serves as a temperature hyperparameter.
609:where λ3 and λ4 are hyperparameters that control the weights
612:from both center and instance views to optimize dependable feature distribution, which effectively reduces intra-class
627:conduct ablation studies to validate the effectiveness of each
628:component. Subsequently, we compare DFLSG with other
629:advanced TIReID methods and evaluate key hyperparameters.
630:Finally, some visualization results are presented to demonstrate
659:parameter to 0.02 and the margin in MTJL to 0.1. To optimize
661:rate of 1 × 10−5 , which gradually cosine decays. In addition, we further evaluate several important hyperparameters in
663:B. Ablation Studies
664:As shown in Table I, we conduct ablation studies on three
665:cross-modality databases to evaluate the effectiveness of each
666:component in DFLSG.
677:which demonstrates the effectiveness of MTJL for unsupervised TIReID in reducing intra-class variations and increasing
690:TABLE I
693:TABLE II
702:TABLE III
708:In No.5, we propose a dependable text generation method
710:to No.2 and No.3, the dependable text generation method
714:TIReID framework effectively learns dependable features.
717:process. From Table I, the No.6 setting achieves the best
718:experimental results, which fully demonstrate the effectiveness
720:DFLSG is effective for the unsupervised TIReID task, which
723:experiments to compare IQR with other filtering algorithms.
724:As shown in Table II, Z-score and LOF represent two different
727:ESFM with the IQR filtering algorithm achieves the best
731:In addition, we assess the model performance under different clustering algorithms, as shown in Table III. K-means
740:benchmark dataset, CUHK-PEDES, as shown in Table IV. Our
745:demonstrate that DFLSG can effectively learn dependable
749:in Table IV, we present the experimental results on ICFGPEDES, which achieves excellent performance, with a Rank-1
750:accuracy of 37.41% and an mAP accuracy of 22.37%. Compared to GAAP, the best unsupervised method in Table IV,
753:effective for the unsupervised TIReID task and significantly
756:database. As shown in Table IV, the key metrics of Rank-1
758:all other unsupervised competitive methods and some supervised learning methods. This is because the dependable text
759:generation and ESFM effectively enhance the reliability of
761:which could learn dependable cross-modality pedestrian
763:D. Parameter Analysis
766:achieves the best performance. The appropriate parameter promotes intra-class features to generate the compact distribution.
767:The parameter λ2 represents the importance of the extended
777:TABLE IV
781:Fig. 4. Evaluation of four important loss parameters in MTJL on the ICFG-PEDES and CUHK-PEDES databases.
783:TABLE V
785:TABLE VI
799:and intra-modality losses in Eq. 9, as shown in Fig. 4 (d)
800:and Fig. 4 (e). The experimental results indicate that when
804:β represents the strength of the filtering algorithm in ESFM.
813:TABLE VII
817:effectively eliminate abnormal samples. We conduct extensive experiments to evaluate the parameter β as shown in

===== DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf =====
13:consists of three key components, that is Two-Stream Backbone Module (TBM), Cross-Modality Interaction
40:full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
74:disentangle modality noise and preserve identity semantics, effectively addressing feature alignment
79:In contrast, VVI-ReID can effectively handle video data. VVI-ReID primarily focuses on addressing the challenges of spatial-temporal feature learning and cross-modality feature alignment.
85:models struggle to effectively capture temporal nuances and long-range dependencies [10], due
105:Fig. 1. Compared to the existing methods, our approach emphasizes the interaction of information between
109:To address the aforementioned issues, we propose a Discriminative ID-Related Representation Learning (DIRL) network for VVI-ReID. DIRL network consists of three key components,
118:— We developed an FDM and designed its corresponding loss function to effectively separate
147:modal sharing features into two orthogonal subspaces, allowing the model to extract more diverse
157:DSGAN and feature-level fusion (PwIF) to effectively reduce modality discrepancy and boost
172:to generate feature vector, leveraging cross-modality features for effective modality compensation
176:diverse training data. For example, Fan et al. [21] proposed the cross-spectrum dual-subspace pairing
177:model, which improved data diversity by splitting RGB images into three channels and adding a
184:can effectively reduce modality gaps and boost model performance.
185:Feature decoupling methods are effective in mitigating modality gaps in VI-ReID by separating
188:effectively reducing spurious correlations and improving cross-modality alignment. Pu et al. [26]
204:difficulties in effectively addressing VVI-ReID. The significant challenges arise from the dynamic
236:three key components: TBM, CIM, and FDM. First, to address the challenge of handling diverse
262:Fig. 2. The architecture of Discriminative ID-related Representation Learning network. Visible and infrared
283:effective at capturing global features and plays a key role in extracting temporal features, we will
312:shared, learnable parameter. Finally, the calculation process of the CIM can be described as follows:
330:Fig. 3. The architecture of Feature Decoupling Module. The yellow rectangles represent RGB modal-related
371:of our efficient and effective visible-infrared retrieval network.
414:and ID-related feature. Next, different loss functions are used to ensure the effectiveness of the
429:demonstrate that our approach achieves more effective feature disentanglement and superior
479:In this section, we will conduct experiments to validate the effectiveness of our proposed method.
481:method with other SOTA approaches. Second, we will conduct ablation experiments to evaluate the
482:impact of different framework stages and parameter settings on the experimental results. Finally,
483:we will conduct some visual analysis on our model.
495:The parameter 𝑆 in the FDM is set to 5,120. The Stochastic Gradient Descent optimizer is utilized,
503:To validate the effectiveness of our proposed method, we performed experiments on two widely
533:Table 1. Comparison of CMC (%) and mAP (%) Performances with the State-of-the-Art Methods on the
686:perspective on the model’s effectiveness across multiple dimensions, offering a clearer indication
692:In this section, we will assess the effectiveness of our proposed method by comparing it with
698:Table 1 reports the comparison results on the HITSZ-VCM dataset. As shown in the table,
714:Table 2. Comparison of CMC (%) and mAP (%) Performances with the State-of-the-Art Methods on the
888:These results highlight the effectiveness of our proposed approach in VVI-ReID. The significant
894:In this section, we will evaluate the effectiveness of our proposed method by comparing it with
900:in Table 2.
901:From Table 2, it is evident that the proposed approach consistently out performs all other
916:Table 3. Ablation Results of CMC (%) and mAP (%) Performances of Key Components on HITSZ-VCM
985:Ablation Experiments
987:In this section, we performed ablation experiments to evaluate the effectiveness of each module. All
988:experiment were trained and evaluated on the HITSZ-VCM datasets. Table 3 presents the ablation
989:results for the key components of our method, namely CIM and FDM. From this table, we can make
991:— The introduced CIM module provides an 8.97% improvement in Rank-1 accuracy and approximately a 6.7% boost in mAP over the baseline. This demonstrates its effectiveness in enhancing
995:— The components above work synergistically. By integrating them, our final method achieves
997:results clearly highlight the effectiveness of the two proposed modules.
1000:Visualization
1008:visible-infrared mode, our approach effectively retrieves the correct clips by recognizing the
1010:In Figure 5, we utilize t-SNE [42] to illustrate the distributions of features. We randomly choose
1013:and our method, respectively. In Figure 5(a), the baseline method effectively clusters intra-class
1018:by increasing sample diversity, allowing the model to more effectively distinguish positive samples
1026:Fig. 4. The Rank-5 results for certain queries using our approach in the infrared-visible and visible-infrared
1031:Fig. 5. The visualization illustrates the feature distributions for both the baseline method and our approach.
1037:Parameter Analysis
1039:Our model encompasses various parameters, including the number of frames in each video sequence
1040:and the value of parameter 𝑆 in FDM. These parameters may have an impact on the model’s
1041:performance. Therefore, we designed a series of parameters analysis experiments to examine the
1042:effects of these parameters on the model.
1048:maintaining this vital information, our approach effectively reduces the impact of noise and captures
1051:resilience and adaptability across diverse frame lengths.
1052:As depicted in Figure 7, we showcase the experimental results of the six parameters, analyzing a
1063:Fig. 6. Performance comparison across different frame counts in the infrared-visible and visible-infrared
1066:Fig. 7. Comparison of performance with varying values of S in the FDM for both infrared-visible and visibleinfrared modes on the HITSZ-VCM dataset, measured by Rank-1 accuracy.
1071:improved performance. While high-dimensional matrices can effectively manage the diversity and
1072:richness of data, excessive redundancy in information and related features may have adverse effects
1084:achieves competitive performance on the VVI-ReID, showcasing its effectiveness in extracting
1085:modal-invariant features. Ablation studies and detailed analyses further confirm the contributions
1104:person re-identification. IEEE Transactions on Pattern Analysis and Machine Intelligence 43, 8 (2020), 2723–2738.
1119:Transactions on Pattern Analysis and Machine Intelligence 45, 11 (2022), 12783–12797.
1182:[35] Yukang Zhang and Hanzi Wang. 2023. Diverse embedding expansion network and low-light cross-modality benchmark
1192:re-identification: A survey and outlook. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 6 (2021),
1201:[42] Pavlin G. Poličar, Martin Stražar, and Blaž Zupan. 2024. Opentsne: A modular python library for t-sne dimensionality

===== Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf =====
5:Discovering Multi-Frequency Embedding for
12:to fully exploit frequency domain information, focusing predominantly on spatial domain feature learning or limited frequency
13:decompositions. To address this, we propose the Multi-Frequency
15:operates in the frequency domain through multi-frequency
16:decomposition to learn discriminative and modality-invariant features. Specifically, the HiLo-Frequency Modulation (HiLo-FM)
17:module efficiently extracts low-frequency features via frequencydomain filtering and high-frequency details through lightweight
19:Frequency-Aware Diversity Enhancer (FADE) module further
20:enriches feature discriminability by weighting multi-frequency
21:components and learning diverse features through multi-branch
25:Ranking Regularization (CMRR) loss enhances feature diversity
83:features, leaving the frequency domain insufficiently leveraged
85:Recent advances in frequency domain analysis have
88:enhancement [29]. As shown in Fig. 2, Fourier analysis reveals that amplitude spectra (Fig. 2(b)) primarily
90:(Fig. 2(c)) preserve structural information across modalities. These findings have motivated several frequency-based
93:both spectral components. These methods perform broadspectrum feature mining across the entire frequency band (top
94:left of Fig. 1), but overlook the distinct roles of low/highfrequency in modality-invariant feature learning.
101:GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
105:Fig. 1. The comparison of existing frequency-based methods and our proposed
107:across the entire frequency band or rely on simple low/high-frequency decomposition, which often adopts coarse and inefficient feature learning in the
108:spatial domain. Conversely, our method achieves efficient low/high-frequency
110:(Sec. III-B). Meanwhile, the FADE module (Sec. III-C) leverages multifrequency attention to generate richer, more robust, and modality-invariant
113:Further decomposition into low-frequency (Fig. 2(d))
114:and high-frequency (Fig. 2(e)) components provides deeper
115:insights. The low-frequency component, capturing the overall
118:or thermal radiation in IR images. In contrast, the highfrequency component effectively filters out modality-specific
120:[19]. Notable methods for low/high-frequency decomposition
121:include FSDF [33] and BiFFN [34] (top right of Fig. 1).
126:have demonstrated certain effectiveness, two critical challenges
127:persist. First, they predominantly execute low/high-frequency
129:the potential for efficient frequency-domain processing. Second, and more critically, they rely on limited frequency
130:decompositions, focusing solely on low/high-frequency components. This limitation is particularly suboptimal, as our
131:visual analysis in Fig. 2 reveals. While low-frequencies
132:(Fig. 2(d)) are dominated by modality-specific styles and highfrequencies (Fig. 2(e)) better preserve shared structures, a
136:Fig. 2. The visualization of frequency components for a pair of VIS (odd
138:and the reconstructions from (b) amplitude (A), and (c) phase (P). The lowfrequency (L) and high-frequency (H) components, shown in (d) and (e), are
139:separated using a frequency mask as detailed in Eq.(4). The multi-frequency
141:non-overlapping bands, formally defined in Eq.(17). This visualization highlights how modality-invariant cues are distributed across different frequency
144:the person’s clothing), are scattered across various frequency
146:as shown in the multi-frequency decompositions (Fig. 2(f)).
147:Relying on a single, handcrafted frequency cutoff thus forces
150:the necessity of a more flexible, multi-frequency analysis to
153:Multi-Frequency Embedding Network (MFENet), a featurelevel method that learns modality-invariant representations
154:from the frequency domain via multi-frequency decompositions. Specifically, to address the challenge of efficient
155:low/high-frequency feature extraction, we introduce the HiLoFrequency Modulation (HiLo-FM) module. This module
157:[36] for low/high-frequency decomposition, directly performs
158:efficient filtering on low-frequency features in the frequency
167:on high-frequency features in the spatial domain. To mitigate
168:unavoidable noise, these low/high-frequency features are then
170:multi-frequency feature learning, we propose the FrequencyAware Diversity Enhancer (FADE) module, which can be
172:in the frequency domain. By weighting multiple pre-set frequencies from low to high, the module accentuates feature
176:robustness and diversity of cross-modality features: CrossModality Soft Retrieval (CMSR) loss and Cross-Modality
185:diversity of cross-modality features. It achieves this by maximizing the inconsistency in ranking among features extracted
191:in the frequency domain and introduces multi-frequency
194:fuses low/high-frequency features, and the FADE module, which effectively enhances feature discriminability
195:through diverse frequency responses.
199:promotes the diversity of cross-modality features by
203:the effectiveness of our designed modules and loss functions, and demonstrate that MFENet outperforms most
225:inherent in VI-ReID tasks effectively.
236:complexity and large number of parameters also limit their
247:frequency-domain cues that can be modality-invariant. In this
249:in the frequency domain, conducting multi-frequency feature
250:learning to mine robust and diverse modality-invariant representations.
251:B. Frequency-Domain Analysis in Deep Learning
252:Frequency-domain analysis [43], a subfield of image
253:processing, converts spatial domain images into frequency
255:Unlike spatial-domain analysis, which directly manipulates
256:pixel values, frequency-domain analysis dissects images into
257:various frequency components, thereby facilitating the differentiation and manipulation of subtle features such as noise and
258:texture. Recent advancements in frequency-domain analysis
263:[33], [34] have initiated the exploration of frequency-domain
268:GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
272:are crucial for effective feature extraction. However, these pioneering works have largely overlooked the frequency-domain
274:low/high-frequency components to modality-invariant feature
277:to decouple high/low-frequency branches for more robust
279:discrete cosine transform to decompose low/high-frequency
280:components and utilizes simple 1 × 1 convolution for feature
283:low/high-frequency decomposition and processes these features in the spatial domain using computationally intensive
285:low/high-frequency features, they both still perform feature
288:In this work, as Fig. 1 illustrates, we first achieve efficient
289:low/high-frequency feature learning and fusion through meticulously designed spatial/frequency operations. Subsequently,
290:we leverage multi-frequency attention to generate richer, more
302:As illustrated in Fig. 3(a), the input images {I(V)
308:style of an image, while high frequencies are often modalityagnostic and contain detailed features [44], a HiLo-Frequency
309:Modulation (HiLo-FM) module (as shown in Fig. 3(b), in
311:modality-invariant feature learning on low/high-frequency features, respectively. Inspired by Instance Normalization [26],
312:[30], [37], a Frequency-Aware Diversity Enhancer (FADE)
313:module (as shown in Fig. 3(c), in Sec. III-C) is incorporated at the end of layer3 . This module further explores rich
314:and diverse modality-invariant features by applying attention
315:weighting across multi-frequency bands from low to high,
362:B. HiLo-Frequency Modulation Module
365:style and illumination (predominantly low-frequency) from
366:identity-bearing structures and textures (predominantly highfrequency), which facilitates cross-modality alignment in
367:VI-ReID. Unlike prior frequency-based methods that merely
369:spatial domain with coarse operators or fixed filters, our
370:HiLo-Frequency Modulation (HiLo-FM) module (Fig. 3(b))
371:performs learnable frequency-domain filtering to efficiently
372:normalize low-frequency style, enhances high-frequency identity cues via lightweight multiscale convolutions in the spatial
375:1) HiLo-Frequency Decomposition: Given an input feature
376:map X ∈ RC×H×W , we first transform it to the frequency
405:In the frequency domain, a low-frequency binary mask Π ∈
412:where the hyperparameter ρ ∈ [0, 1] balances low/highfrequency components, and a square-shaped method is used to
413:separate these components. Alternatively, circular or diamondshaped methods could be applied, and subsequent ablation
414:studies are conducted to compare these three methods. Accordingly, the high-frequency binary mask can be obtained by
422:Fig. 3. Overall framework of Multi-Frequency Embedding Network (MFENet). (a) The high-level architecture, showing the integration of our novel modules
423:and loss functions. (b, c) Detailed schematics of our core technical innovations: the HiLo-Frequency Modulation (HiLo-FM) and Frequency-Aware Diversity
427:Π̄ = 1 − Π, then the low/high-frequency components can be
438:2) Low-Frequency Adaptation: Given that low-frequency
439:components carry global style information, employing smallkernel convolutions in the spatial domain is insufficient for
443:GFNet [47], we propose to enhance low-frequency features
444:directly in the frequency domain by introducing a parameterized complex weight matrix W̃L ∈ CC×H×W :
452:reduction in both parameter count and computational complexity. Specifically, the matrix X̃L contains non-zero values
454:by the low-frequency mask. Additionally, the real-valued 2D
456:to retaining the full complex frequency domain size. As a
457:result, the effective parameter count of W̃L is reduced to
459:CbρHcbρ(W/2 + 1)c. Finally, we transform the adapted lowfrequency component back to the spatial domain via iFFT:
464:3) High-Frequency Enhancement: High-frequency features, which capture various scales of details and contours of
467:via convolutions. As shown in Fig. 2, a single 3×3 convolution
470:in Fig. 4), inspired by OSNet [48], for efficient multi-scale
473:F −1 (X̃H ) ∈ RC×H×W . To integrate the sparse high-frequency
486:GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '668,845p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   668	C. Ablation Study
   669	We first evaluate the influence of prototypes and anchors by
   670	adjusting their numbers. As shown in Table I, more or fewer
   671	prototypes both degrade the performance of our CycleTrans.
   672	The more prototypes make features from two modalities
   673	that have no common representation and cannot alleviate the
   674	modality discrepancy. While too few prototypes are missing to
   675	represent the necessary information about a person. Similarly,
   676	too much anchors cause the conflict in (15) and less anchors
   677	capture inadequate information for a person to decline the
   678	performance.
   679	We then ablate our CycleTrans on SYSU-MM01 under allsearch single-shot setting [1], of which results are given in
   680	Table II. Here, baseline denotes that the model only consists
   681	of the convolution backbone and is trained merely with the
   682	cross-entropy loss Lid .
   683	Table II shows the cumulative results of each design
   684	in CycleTrans. From this table, we can first observe that
   685	the proposed KCM and DMM can significantly improve
   686	model performance, achieving +3.33% and +1.67% gains on
   687	Rank-1, respectively. The use of cycle construction, i.e., +Lr ec
   688	and +Laln , can also improve performance to a large extent,
   689	e.g., +4.56% on Rank-1 compared to “+D M M.” Meanwhile,
   690	
   691	TABLE III
   692	I MPACT OF D IFFERENT A LTERNATIVES OF DMM ON SYSU-MM01
   693	U NDER A LL -S EARCH S INGLE -S HOT S ETTING
   694	
   695	we also notice that the metric loss Lme can also bring
   696	improvements on all metrics, suggesting its benefits for neutral
   697	features. Lastly, combining all designs proposed in CycleTrans
   698	can improve the baseline by up to +12.97% Rank-1, strongly
   699	validating their effectiveness.
   700	We also examine different alternatives of the proposed
   701	DMM, i.e., (7), of which results are given in Table III. The
   702	second block of Table III shows the different choices of DMM,
   703	including the one aggregating prototypes without residual
   704	connection, i.e., F N = AP, and the one without discrepancy
   705	modeling, i.e., F N = F′I + AP. We also use a Transformer
   706	layer [67] for comparison.
   707	The first alternative only uses the aggregated prototypes
   708	as neutral features, which can strictly follow the distribution of prototype information. However, this alternative will
   709	make the convolution backbone hard to optimize, since the
   710	image features are not directly involved in the objective
   711	functions. Meanwhile, the lack of fine-grained image semantics from residual connection also limits its performance
   712	upper-bound. Compared to DMM, the second alternative does
   713	not include discrepancy modeling, which leads to obvious
   714	performance degradation. One hypothesis is that without discrepancy modeling, the obtained neutral features are still
   715	highly modality-relevant, making the model fail in crossmodal retrieval. The use of a Transformer layer is a good
   716	choice for neutral feature transformation, which takes the
   717	modality-relevant features as queries and the prototypes as
   718	keys and values. However, its performance is still inferior to
   719	our DMM, e.g., −3.52% Rank-1 and −3.69% mAP. Overall,
   720	these designs well confirm the effectiveness of our DMM in
   721	neutral feature learning for VI-ReID.
   722	D. Comparison With SOTA Methods
   723	We then compare our CycleTrans with a set of SOTAs
   724	on SYSU-MM01 and RegDB, of which results are given in
   725	Tables IV and V, respectively.
   726	
   727	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:38 UTC from IEEE Xplore. Restrictions apply.
   728	
   729	WU et al.: CycleTrans: LEARNING NEUTRAL YET DISCRIMINATIVE FEATURES VIA CYCLE CONSTRUCTION
   730	
   731	5475
   732	
   733	TABLE IV
   734	C OMPARISON B ETWEEN C YCLE T RANS AND THE SOTA M ETHODS ON SYSU-MM01.T HE B EST P ERFORMANCE I S BOLD , AND THE S ECOND
   735	B EST I S UNDERLINED .T HE M ETHODS T HAT TAKE THE S AME BACKBONE AND S ETTINGS AS AGW [17] A RE M ARKED W ITH “*”
   736	
   737	1) Comparisons on SYSU-MM01: As shown in Table IV,
   738	the proposed CycleTrans outperforms existing SOTAs by large
   739	margins on SYSU-MM01. Specifically, compared to the latest
   740	method, i.e., SMCL [33], CycleTrans can obviously improve
   741	the performance of all metrics under All-Search setting, e.g.,
   742	+4.57% on Rank-1 and +5.46% on mAP. Under the setting
   743	of Indoor-Search, the advantages of CycleTrans are further
   744	expanded. For instance, the SOTA performance on Singleshot Rank-1 and Multi-shot Rank-1 is improved by +9.99%
   745	and +9.71% by our method, which is indeed very significant.
   746	When taking the same backbone and settings as AGW [17],
   747	our CycleTrans maintains its advantage. Specifically, compared to DEEN [51], CycleTrans improves performance under
   748	All-Search Single-Shot, e.g., +1.88% on Rank-1 and +0.82%
   749	on mAP.
   750	2) Comparisons on RegDB: Similar advantages of CycleTrans can be also witnessed on RegDB in Table V, which
   751	is a smaller-scale dataset. Under two cross-modality settings,
   752	our method achieves new SOTA performance on all metrics. Notably, the latest method FMCNet [47] has already
   753	achieved obvious gains over previous VI-ReID methods, but
   754	our CycleTrans can further improve performance, e.g., +2.2%
   755	and +1.9% Rank-1 on two settings. When taking the same
   756	backbone and settings as AGW [17], the proposed CycleTrans
   757	
   758	achieves competitive performance. Under both Infrared to Visible and Visible to Infrared settings, the proposed CycleTrans
   759	improve mAP by +2.2% and +1.9%, compare to DEEN [51].
   760	Considering SYSU-MM01 and RegDB are two highly
   761	competitive benchmarks, these significant performance gains
   762	strongly validate the effectiveness of the proposed CycleTrans
   763	and our motivation about the modality gap.
   764	E. Quantitative Analysis
   765	1) Impact of Hyper-Parameters: In Fig. 3, we report
   766	the impact of hyper-parameters in the proposed CycleTrans.
   767	We can first observe that CycleTrans is reasonably robust to
   768	the values used to control the impact of different modules
   769	in cycle construction and identification (i.e., λ1 , λ2 , λ3 , and
   770	λ4 ). In particular, the most significant difference appears
   771	in λ1 , when the difference between 0.3 and 0.5 reaches
   772	3.23% in Rank-1. Especially when the value is too large,
   773	the performance reduction is particularly obvious. Since the
   774	model gives up some valuable information, it ensures that
   775	the information captured by each anchor is not duplicated.
   776	We can also observe that the loss function used to eliminate
   777	modality differences is not sensitive to their weights, i.e.,
   778	λ2 and λ4 . This shows that the proposed KCM and DMM
   779	play an effective role in alleviating modality gap. As for
   780	
   781	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:38 UTC from IEEE Xplore. Restrictions apply.
   782	
   783	5476
   784	
   785	IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 3, MARCH 2025
   786	
   787	TABLE V
   788	
   789	TABLE VI
   790	
   791	C OMPARISON W ITH SOTA M ETHODS ON R EG DB.T HE B EST
   792	P ERFORMANCE I S BOLD , AND THE S ECOND B EST I S
   793	UNDERLINED . T HE M ETHODS T HAT TAKE THE S AME
   794	BACKBONE AND S ETTINGS AS AGW [17]
   795	A RE M ARKED W ITH “*”
   796	
   797	C OMPUTATION OVERHEAD OF THE P ROPOSED C YCLE T RANS .
   798	T HE M ETHODS T HAT TAKE THE S AME BACKBONE AND
   799	S ETTINGS AS AGW [17] A RE M ARKED W ITH “*”
   800	
   801	testing stages are reported in Table VI. We can first observe
   802	that the proposed CycleTrans significantly improves the performance, i.e., +17.59% and +18.33% on Rank-1 and mAP,
   803	with limited increase in the computation overhead, i.e., +0.6h
   804	in training. As for the representative method, e.g., DEEN [51],
   805	CycleTrans* achieves +1.88% and +0.82% on Rank-1 and
   806	mAP with 33.3% training time. Overall, the proposed CycleTrans method is an effective and efficient way to address the
   807	visible-infrared person ReID.
   808	F. Qualitative Analysis
   809	
   810	Fig. 3. Impact of hyper-parameters λ1 , λ2 , λ3 and λ4 . The performances
   811	are evaluated on SYSU-MM01 under all-search single-shot.
   812	
   813	reconstruction loss, the λ2 is limited to a small range to ensure
   814	that the model can be optimized normally. Experimental results
   815	well confirm the effectiveness of the proposed CycleTrans in
   816	alleviating modality gap and extracting discriminative features.
   817	2) Inference Efficiency: We further compare the actual
   818	inference efficiencies of CycleTrans and representative methods. The computation overhead during both the training and
   819	
   820	To gain deep insight into the proposed CycleTrans, we further visualize the distributions of different features extracted
   821	by the baseline and our CycleTrans in Fig. 4. We randomly
   822	visualize samples of ten identities from the testing set via
   823	t-SNE [68]. Fig. 4(a) shows the feature distribution of the
   824	baseline. We can see that although these features can be
   825	mapped to different clusters, images of the same identity but
   826	different modalities are still hard to distinguish. For instance,
   827	the blue and yellow features of the same modalities are
   828	closely distributed in this space and hard to identify. Fig. 4(b)
   829	shows the results of CycleTrans without Cycle Construction. It illustrates that CycleTrans can well transform these
   830	modality-relevant features into neutral ones with the help of
   831	the proposed DMM, resulting in better clusters than Fig. 4(a).
   832	However, due to the lack of enough feature discriminability,
   833	the cross-modality features of some identities still do not
   834	exhibit clear semantic margins, e.g., the yellow and blue
   835	examples. With cycle construction, this problem is greatly
   836	alleviated, as shown in Fig. 4(c). From this figure, we can see
   837	that our CycleTrans can learn clear margins between features
   838	of different identities. Meanwhile, the better clustering result
   839	of CycleTrans than the other two methods suggests a stronger
   840	descriptive power.
   841	Furthermore, cycle construction can also improve discriminability by effectively expanding the model’s attention scopes.
   842	We visualize the attention results of CycleTrans and its alternatives with Grad-CAM [69] in Fig. 5. Fig. 5(a) shows the
   843	heat maps of the baseline. It only focuses on the information
   844	of a small region that can be generalized across modalities. However, such information is not sufficient for ReID.
   845	Fig. 5(b) and (c) show the results of CycleTrans without

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '841,890p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   841	Furthermore, cycle construction can also improve discriminability by effectively expanding the model’s attention scopes.
   842	We visualize the attention results of CycleTrans and its alternatives with Grad-CAM [69] in Fig. 5. Fig. 5(a) shows the
   843	heat maps of the baseline. It only focuses on the information
   844	of a small region that can be generalized across modalities. However, such information is not sufficient for ReID.
   845	Fig. 5(b) and (c) show the results of CycleTrans without
   846	and with cycle construction. Benefiting from the first KCM,
   847	alternative (b) can capture more information for VI-ReID. But
   848	without cycle construction, its attention is likely to become
   849	
   850	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:12:38 UTC from IEEE Xplore. Restrictions apply.
   851	
   852	WU et al.: CycleTrans: LEARNING NEUTRAL YET DISCRIMINATIVE FEATURES VIA CYCLE CONSTRUCTION
   853	
   854	5477
   855	
   856	Fig. 4. Feature visualizations. Circles and triangles denote the features of visible and infrared images, respectively, and the colors represent different identities.
   857	(a) Baseline refers to the basic setting described in Table II. The middle plot shows the results of our CycleTrans (b) without cycle construction. Compared
   858	to the other two models, our (c) CycleTrans can well cluster features of different modalities but with the same identity. It also exhibits more clear semantic
   859	margins between identities.
   860	
   861	Trans first use a novel KCM to mine salient semantics from
   862	convolution feature maps based on pseudo anchors. Afterward,
   863	we propose a DMM to transform these semantics into neutral
   864	features based on the modality-irrelevant prototypes. To ensure
   865	the descriptive power of the neutral features, feature cycle
   866	constructions are performed via another two KCMs sharing
   867	the same parameters. To validate our CycleTrans, we conduct
   868	extensive experiments on two highly competitive benchmarks,
   869	namely SYSU-MM01 and RegDB. The experimental results
   870	not only report the new SOTA performance achieved by
   871	CycleTrans with great advantages to existing methods, e.g.,
   872	+1.88% Rank-1 and +1.1% Rank-1 on SYSU-MM01 and
   873	RegDB, but also greatly validate the effectiveness of our
   874	method toward the modality gap.
   875	R EFERENCES
   876	Fig. 5. Visualizations of attention results. (a) Baseline refers to the basic
   877	setting described in Table II. (b) w/o Cycle refers to the CycleTrans without
   878	cycle construction. Compared to the other two alternatives, (c) CycleTrans
   879	can grasp more details for VI-ReID.
   880	
   881	noisy and sparse, e.g., attending to the background areas.
   882	In stark contrast, CycleTrans [Fig. 5(c)] can grasp more
   883	comprehensive and salient visual cues, and this visual information can also be well aligned across modalities. Furthermore,
   884	we can observe from Fig. 5(c) that the source of the knowledge
   885	captured by CycleTrans is the appearance of a pedestrian.
   886	According to (7), prototypes are aggregated based on their
   887	correlation with modality-relevant features. To this end, heat
   888	maps can well illustrate where the knowledge in a prototype
   889	comes from.
   890	Overall, the visualization results well confirm the effectiveness of the proposed CycleTrans toward neutral yet

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'CycleTrans - Learning Neutral Yet Discriminative Features via Cycle Construction for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '890,910p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   890	Overall, the visualization results well confirm the effectiveness of the proposed CycleTrans toward neutral yet
   891	discriminative feature learning for VI-ReID.
   892	VI. C ONCLUSION
   893	In this article, we aim to address the modality gap in
   894	VI-ReID via learning neutral yet discriminative features.
   895	To approach this target, we propose a cycle-construction-based
   896	model for VI-ReID, termed CycleTrans. Specifically, Cycle-
   897	
   898	[1] A. Wu, W. Zheng, H. Yu, S. Gong, and J. Lai, “RGB-infrared crossmodality person re-identification,” in Proc. ICCV, 2017, pp. 5390–5399.
   899	[2] P. Dai, R. Ji, H. Wang, Q. Wu, and Y. Huang, “Cross-modality person
   900	re-identification with generative adversarial training,” in Proc. 27th Int.
   901	Joint Conf. Artif. Intell., Jul. 2018, pp. 677–683.
   902	[3] M. Ye, X. Lan, and Q. Leng, “Modality-aware collaborative learning
   903	for visible thermal person re-identification,” in Proc. ACM Multimedia,
   904	L. Amsaleg et al., Eds. 2019, pp. 347–355.
   905	[4] H. Luo et al., “A strong baseline and batch normalization neck for
   906	deep person re-identification,” IEEE Trans. Multimedia, vol. 22, no. 10,
   907	pp. 2597–2609, Oct. 2020.
   908	[5] Y. Lu et al., “Cross-modality person re-identification with sharedspecific feature transfer,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
   909	Recognit. (CVPR), Jun. 2020, pp. 13376–13386.
   910	[6] C. Fu, Y. Hu, X. Wu, H. Shi, T. Mei, and R. He, “CM-NAS:

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'DATE - Dual Asymmetric Textual Embedding guided Person Re-Identification.pdf' - | nl -ba | sed -n '680,830p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   680	90.7
   681	95.4
   682	
   683	TABLE II
   684	E FFECT OF DIFFERENT FUSION STRATEGIES ON MSMT17. “CT” IS
   685	CROSS - ATTENTION METHOD , AND “AD” IS ADD METHOD .
   686	CT
   687	
   688	AD
   689	√
   690	
   691	√
   692	
   693	mAP
   694	74.0
   695	40.2
   696	74.7
   697	
   698	R@1
   699	89.2
   700	74.3
   701	89.4
   702	
   703	R@5
   704	94.6
   705	81.9
   706	94.6
   707	
   708	Lp is 4. The Adam optimizer with a learning rate initialized
   709	at 0.00035 and decayed by a cosine schedule with a batch
   710	size 64. We utilize LLaVA v1.6 to generate semantic descriptions for each image. The textual embeddings are integrated
   711	with the original visual embeddings through a cross-attention
   712	mechanism to obtain visual embeddings, which are then used
   713	for retrieval.
   714	A. Comparisons with State-of-the-arts
   715	In this section, we compare the proposed DATE against
   716	state-of-the-art techniques across three extensively utilized
   717	person ReID datasets and summarized the related results in
   718	Table I. We can observe that the proposed DATE outperforms
   719	existing methods on all three datasets, e.g., obtaining the mAP
   720	of 90.7%, 83.6% and 74.7% for Market-1501, DukeMTMC,
   721	and MSMT17, respectively. Moreover, it can be observed
   722	that the proposed DATE obtains an obvious improvement
   723	upon our baseline CLIP-ReID, e.g., improving mAP from
   724	73.4%/82.5%/89.6% for MSMT17/DukeMTMC/Market-1501.
   725	The superior performance verifies the effectiveness and reasonability of using the generated textual-level description with the
   726	proposed Dual Asymmetric Textual Embedding.
   727	B. Ablation Studies
   728	To further evaluate the effectiveness of description-based
   729	textual space in ReID tasks, we conducted several ablation
   730	studies about the proposed component in this section.
   731	1) Effect of Cross-attention Adapter: Table II compares two
   732	different adapter strategies to verify the effectiveness of the
   733	proposed Cross-attention Adapters. Here, CT denotes crossattention mechanism; AD is the add mechanism, which can
   734	be formulated as Aad (f i , di ) = f i + di . We can observe that
   735	using the CT strategy obtains mAP of 74.7%, obtaining 0.7%
   736	
   737	TABLE III
   738	E FFECT OF Lce , Lcon AND Ld IN DATE ON MSMT17.
   739	Lce
   740	
   741	Lcon
   742	
   743	Ld
   744	√
   745	
   746	√
   747	√
   748	√
   749	
   750	√
   751	
   752	√
   753	√
   754	
   755	mAP
   756	1.2
   757	72.1
   758	73.4
   759	74.4
   760	74.7
   761	
   762	R@1
   763	5.6
   764	87.9
   765	88.7
   766	89.3
   767	89.4
   768	
   769	R@5
   770	12.5
   771	94.1
   772	94.5
   773	94.6
   774	
   775	improvement upon the baseline. The superior performance
   776	demonstrates the reasonability and necessity of using the
   777	cross-attention mechanism to fuse description-based embedding with visual embedding. Compared to the cross-attention
   778	mechanism, the add mechanism treats the fused embedding
   779	equally while ignoring the importance of each embedding.
   780	Consequently, the AD strategy obtains a worse performance
   781	than the baseline and CT strategy.
   782	2) Effect of Description-based Textual Embedding: Table III illustrates the effect of description-based textual embedding on DATE, where Lce denotes the alignment between
   783	learnable-based textual embedding and the visual embedding, and Lcon stands for the constraint between descriptionbased to learnable-based textual embeddings. Moreover, Ld
   784	is the loss between visual embedding and the identity-aware
   785	description-based embeddings. Without additional constraints,
   786	the pre-trained CLIP visual encoder performs poorly at 1.2%
   787	mAP, showing that the pre-trained CLIP is unsuitable for the
   788	task of person ReID. Using learnable-based textual embedding
   789	with Lce as a supervision gives the model an mAP of 73.4%,
   790	while Ld utilizes the identity-level descriptions and yields an
   791	mAP of 72.1%. Incorporating supervision from the learnablebased textual embedding with Ld , the mAP reaches 74.4%.
   792	By considering all three type of constraints to fully fuse
   793	the description-based textual embedding, we obtain the final
   794	best mAP of 74.7%. This demonstrates that incorporating
   795	description-based information during training can effectively
   796	enhance the performance of both the learnable-based textual
   797	space and the visual space, thereby improving DATE’s discriminative capability.
   798	3) Effect of Cross-Granularity Training: Table IV illustrates the effect of two types of description on the learnablebased textual space, where Td represents the image-level
   799	description, and Tˆd represents the identity-level descriptions.
   800	We observed that the inclusion of image-level descriptionbased textual embeddings improved the model’s mAP from
   801	74.4% to 74.7%. This indicates that incorporating image-level
   802	description-based textual embeddings effectively enhances the
   803	discriminative capability of the textual space. Using identitylevel description-based textual embeddings resulted in a 0.2%
   804	lower mAP compared to using image-level embeddings, which
   805	suggests that the cross-granularity training leads to a stronger
   806	generalization capability.
   807	V. C ONCLUSION
   808	In this work, we propose a novel cross-modal pedestrian reidentification method, Dual Asymmetric Textual Embedding
   809	
   810	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:22 UTC from IEEE Xplore. Restrictions apply.
   811	
   812	TABLE IV
   813	E FFECT OF Td AND Tˆd ON THE TRAINING PROCESS OF
   814	LEARNABLE - BASED TEXTUAL SPACE IN DATE ON MSMT17.
   815	Td
   816	
   817	Tˆd
   818	√
   819	
   820	√
   821	
   822	mAP
   823	74.4
   824	74.5
   825	74.7
   826	
   827	R@1
   828	89.3
   829	89.5
   830	89.4

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Deep intelligent technique for person Re-identification system in surveillance images.pdf' - | nl -ba | sed -n '600,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   600	1812 individuals. The learning set consists of 16,522 pictures of 702
   601	beings, and the remaining 19,889 images of 702 individuals are
   602	considered for the model testing. Among 19,889 testing images, 2228
   603	are query images, and the remaining 17,661 are gallery images.
   604	4.2. Implementation results
   605	The proposed HLGFS model incorporated four steps: preprocessing,
   606	feature analysis, prediction, and similarity finding. The model extracted
   607	the body shape, face, and clothing features for the re-person identifi­
   608	cation. The horned lizard fitness process is utilized for the fine-tuned
   609	network training, which enhances the model’s accuracy. The learning
   610	rate of the model is 0.0002, the dropout ratio is 0.5, and the batch size is
   611	32. The accuracy and loss graph of the training and testing is shown in
   612	Figs. 4 and 5.
   613	Some of the sample images taken from the CUHK03, Market1501,
   614	and DukeMTMC re-id datasets and the results given by the proposed
   615	HLGFS model at each process are shown in Table 3 and Table 4.
   616	Here, the person in the query image is effectively identified by the
   617	designed HLGFS system. The model tuned their parameters to accurately
   618	track the body parts and extract the fittest features for the prediction.
   619	The HLGFS in CUHK03, Market1501, and DukeMTMC datasets show
   620	super consistency in various surveillance scenarios and overall effec­
   621	tiveness. The model attained high accuracy and mAP values on CUHK03,
   622	Market1501and DukeMTMC. Further, it ensures high performance at
   623	different ranks and shows reliability in fetching relevant matches. This
   624	ensures robust ranking and matching performance. Similar trends are
   625	also noted in various datasets where Rank-1 to Rank-5 accuracies are
   626	constant. These results confirm the ability of this model to adapt and
   627	perform well on datasets with different complexities and may have
   628	broad applicability in person re-identification and intelligent surveil­
   629	lance scenarios. The accuracy and mAP of the considered three datasets
   630	using the recommended HLGFS model are arranged in Table 5. Here, the
   631	accuracy is validated for images ranked 1 to 5.
   632	
   633	Fig. 5. Loss curve.
   634	
   635	4.4. Comparative analysis
   636	The proposed system is compared with the existing methods such as
   637	SSLA [13], EFAN [19], SA [20], AAT [21], APS [22], Hybrid Ranking
   638	Framework (HRF) [29], Lightweight Multi-Branch System (LMS) [30],
   639	and k Reciprocal Nearest Neighbor (kRNN) [31] for Market1501,
   640	DukeMTMC re-id and CUHK03dataset. The image quality and resolution
   641	highly influence the feature extraction.
   642	4.4.1. Accuracy (A)
   643	The accuracy metric quantifies the percentage of queries where the
   644	correct matching image is ranked within the top n retrieved images from
   645	the gallery set. It shows how often the model finds the right person
   646	among the most similar candidates. The accuracy of the model is
   647	measured using Eqn. (6).
   648	A=
   649	
   650	correctly matched images
   651	total query images
   652	
   653	(6)
   654	
   655	Rank-1 accuracy indicates the proportion of queries where the model
   656	retrieves the correct match as the most similar image, while higher n
   657	values represent a more lenient criterion for successful identification.
   658	Here, the accuracy of the designed HLGFS model is compared with some
   659	of the existing re-identification frameworks, such as HRF, LMS, AbRF,
   660	and kRNN tested on the same CUHK03 dataset. The accuracy value
   661	obtained by the recommended HLGFS model is 97.8 %. The accuracy
   662	comparison is shown in Table 8. Here, the recommended technique
   663	attained the leading accuracy of the other existing frameworks.
   664	
   665	4.3. Ablation study
   666	The ablation study was conducted to evaluate how the ratio of
   667	training data affects the performance of the designed re-identification
   668	system. The accuracy and precision rates recorded at the different
   669	training sample ratios for each dataset are recorded in Table 6.
   670	Accuracy and mAP generally rise with increasing training dataset
   671	size in all datasets. For example, compared to CUHK03, Market1501
   672	exhibits somewhat stronger mAP at lower training sizes.
   673	
   674	4.4.2. Mean average precision (mAP))
   675	In Re-ID, mAP measures the mean of the Average Precision (AP)
   676	calculated for each query in the dataset. Each query involves searching
   677	for a specific person’s image (probe) within a more extensive set of
   678	images (gallery) captured by different cameras or at different times. The
   679	mAP of the designed re-identification framework is shown in Eqn. (7).
   680	Y
   681	∑
   682	
   683	mAP =
   684	
   685	y=1
   686	
   687	AP(y)
   688	Y
   689	
   690	(7)
   691	
   692	Here y indicates the query images present in the considered dataset,
   693	Y represents the number of query images, and AP denotes the average
   694	precision value. The average precision value is validated under the
   695	precision-recall curve. Here, the designed HLGFS attained a higher mAP
   696	rate than the existing schemas. The achieved mAP rate of the proposed
   697	approach is 97.6 %. The mAP comparison with the prevailing mecha­
   698	nisms is shown in Table 8.
   699	Fig. 4. Accuracy curve.
   700	6
   701	
   702	Ms.R.M.A. Pandeeswari and D.G. Rajakumar
   703	
   704	Pattern Recognition 162 (2025) 111349
   705	
   706	Table 3
   707	HLGFS sample outputs of CUHK03 database.
   708	
   709	Table 4
   710	Sample results of Market1501 and DukeMTMC re-id dataset.
   711	
   712	4.4.3. F-score
   713	The percentage of correctly labeled pairs divided by the entire
   714	amount of pairs with the same pseudo label is known as recall (R). At the
   715	same time, the F-score (F) is the harmonic mean of the precision and the
   716	recall value. The f-score value can be computed through the expression
   717	given in Eqn. (8).
   718	
   719	Table 5
   720	Performance results of CUHK03, Market1501 and DukeMTMC re-id datasets.
   721	Datasets
   722	CUHK03
   723	Market1501
   724	DukeMTMC reid
   725	
   726	Accuracy (%)
   727	
   728	mAP (%)
   729	
   730	Rank 1
   731	
   732	Rank 2
   733	
   734	Rank 3
   735	
   736	Rank 4
   737	
   738	Rank 5
   739	
   740	97.8
   741	97.68
   742	96.65
   743	
   744	97.28
   745	98.2
   746	96.34
   747	
   748	97.08
   749	98.98
   750	96.08
   751	
   752	98
   753	98.24
   754	97.98
   755	
   756	98.61
   757	98.34
   758	97.56
   759	
   760	97.6

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Deep intelligent technique for person Re-identification system in surveillance images.pdf' - | nl -ba | sed -n '760,1045p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   760	97.6
   761	98.87
   762	96.65
   763	
   764	F − score = 2 ×
   765	
   766	mAP × recall
   767	mAP + recall
   768	
   769	(8)
   770	
   771	All the metrics value of the researched technique is superior to the
   772	7
   773	
   774	Ms.R.M.A. Pandeeswari and D.G. Rajakumar
   775	
   776	Pattern Recognition 162 (2025) 111349
   777	
   778	different image sets. Hence, gaining the finest outcome than other
   779	models is more important. Here, outstanding results were gained
   780	because of processing the horned lizard features in the feature analysis
   781	phase. It is apt for this application to analyze the features in any envi­
   782	ronment with different quality due to their skin-changing behavior.
   783	
   784	Table 6
   785	Ablation study.
   786	Training
   787	dataset
   788	size
   789	
   790	25 %
   791	50 %
   792	75 %
   793	100 %
   794	
   795	Datasets
   796	CUHK03
   797	
   798	Market1501
   799	
   800	DukeMTMC re-id
   801	
   802	Accuracy
   803	(%)
   804	
   805	mAP
   806	(%)
   807	
   808	Accuracy
   809	(%)
   810	
   811	mAP
   812	(%)
   813	
   814	Accuracy
   815	(%)
   816	
   817	mAP
   818	(%)
   819	
   820	86.5
   821	90.3
   822	94.2
   823	97.8
   824	
   825	86.4
   826	91
   827	95.3
   828	97.6
   829	
   830	86.4
   831	92.1
   832	95.2
   833	97.68
   834	
   835	85.3
   836	90.6
   837	94.9
   838	98.87
   839	
   840	85.3
   841	90.4
   842	93.5
   843	96.65
   844	
   845	84.6
   846	92.6
   847	92.3
   848	96.65
   849	
   850	4.5. Discussion
   851	The discussion section is based on the practical implications, key
   852	strengths, and limitations of this study. Moreover, to verify the proposed
   853	model performance against other optimization techniques such as par­
   854	ticle swarm (PS) [32], grey wolf (GW) [33], Ant colony (AC) [34], and
   855	Hyena optimization (HO) [35], the Googlenet (GN) is integrated with
   856	these optimization techniques and the performance metrics such as ac­
   857	curacy, mAP, recall, F-score and error rate are evaluated and the results
   858	are described in Table 8.
   859	The proposed method meets with the highest performance compared
   860	to other techniques like PS, GW, AC, and HO in overall metrics. This
   861	shows its robust performance on complex high-dimensional data, which
   862	is one of the significant challenges in person re-identification. The
   863	exploration-exploitation balance of HL allows the method to converge
   864	rapidly on optimal solutions and prevent falling into local optima.
   865	Moreover, the promising performance of the proposed GN+HL in pre­
   866	cision and recall measures indicates the ability of HL to discover
   867	discriminative features more accurately, which in turn allows for better
   868	generalization and matching accuracy across camera views. Thus the
   869	presented framework substantially outperforms competing approaches.
   870	Practical implication: In the real-time scenario, the model works
   871	based on the real captured images from the CCTV or surveillance cam­
   872	era, which is specifically more suitable for predicting the criminalist by
   873	tracing the crime events. It enhances the public safety from the theft and
   874	other crime events. Hence, Surveillance camera images are the only
   875	requirements to execute this present proposed model in the real-time
   876	scenario. The HLGFS is a highly accurate and precise model for sur­
   877	veillance and public security applications. It uses advanced feature
   878	
   879	considered existing methods. Hence, the recall and F-score metrics are
   880	exposed in Table 8.
   881	4.4.4. Error rate
   882	The error rate can be interpreted as the proportion of images that the
   883	model makes a mistake in identifying the correct person. The identifi­
   884	cation of the irrelevant image (image of a different person) for the
   885	respective query image is validated as an error rate. Compared to the
   886	prevailing techniques, the designed HLGFS model provided a low error
   887	rate of 2.4 %, statistics given in Table 8.
   888	4.4.5. Dataset results comparison
   889	The efficiency of the HLGFS model for the Market1501, DukeMTMC
   890	re-id, and CUHK03dataset are compared with HRF, LMS, AbRF, kRNN,
   891	SSLA, EFAN, SA, AAT, and APS models. The comparison is shown in
   892	Table 7. Here, all the approaches were tested, and the same proposed
   893	platform and performance were compared to each other. The proposed
   894	model maintained the maximum efficiency of the other prevailing
   895	models, which shows the robust identification of the proposed frame­
   896	work. Here, the Market1501 dataset is referred to as A, the DukeMTMC
   897	re-id dataset is referred to as B, and the CUHK03 dataset is referred to as
   898	C.
   899	The proposed method gained higher results than the other models for
   900	all three datasets. However, a comparison with the related works is
   901	added to this section to emphasize the proposed model. The average
   902	results of the developed HLGFS model are observed and compared in
   903	terms of accuracy, mAP, recall, and f-value and error metrics. The pro­
   904	cessing time of this proposed model is 20 to 60 s computational time
   905	while dealing with the above-mentioned three databases. It is quite in an
   906	optimal state.
   907	Considering all other traditional models, the proposed model has
   908	earned the finest outcome, it is the most important element for appli­
   909	cability in a real-time environment. To analyze CCTV-based videos, the
   910	proposed model must be efficient in identifying the person from
   911	
   912	Table 8
   913	Combining Google Net with Other Optimization.
   914	Methods
   915	
   916	Accuracy
   917	(%)
   918	
   919	mAP
   920	(%)
   921	
   922	Recall
   923	(%)
   924	
   925	F-score
   926	(%)
   927	
   928	Error
   929	(%)
   930	
   931	GN+PS
   932	GN+GW
   933	GN+AC
   934	GN+HO
   935	GN+ HL
   936	(Proposed)
   937	
   938	73.5
   939	86.2
   940	77.9
   941	93.2
   942	97.68
   943	
   944	72.6
   945	85.7
   946	78.6
   947	91.6
   948	98.87
   949	
   950	72.3
   951	86.1
   952	77.5
   953	91.4
   954	97.45
   955	
   956	72.5
   957	85.9
   958	78.1
   959	91.5
   960	98.4
   961	
   962	26.5
   963	13.8
   964	22.1
   965	6.8
   966	2.32
   967	
   968	Table 7
   969	Market1501, DukeMTMC and CUHK03 re-id dataset performance analysis.
   970	Metrics
   971	Accuracy (%)
   972	mAP (%)
   973	Recall (%)
   974	F-score (%)
   975	Error rate (%)
   976	
   977	Methods
   978	Dataset
   979	
   980	SSLA
   981	
   982	EFAN
   983	
   984	SA
   985	
   986	HRF
   987	
   988	LMS
   989	
   990	kRNN
   991	
   992	AAT
   993	
   994	APS
   995	
   996	Proposed
   997	
   998	A
   999	B
  1000	C
  1001	A
  1002	B
  1003	C
  1004	A
  1005	B
  1006	C
  1007	A
  1008	B
  1009	C
  1010	A
  1011	B
  1012	C
  1013	
  1014	86.5
  1015	86
  1016	86.5
  1017	86.4
  1018	85
  1019	86.4
  1020	90.3
  1021	91
  1022	90.3
  1023	91
  1024	90
  1025	91
  1026	13.5
  1027	14
  1028	13.5
  1029	
  1030	94.2
  1031	93.6
  1032	94.2
  1033	95.3
  1034	94.3
  1035	95.3
  1036	92.1
  1037	91.1
  1038	92.1
  1039	95.2
  1040	94
  1041	95.2
  1042	5.8
  1043	6.4
  1044	5.8
  1045	

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Dependability Feature Learning Based on Sample Generation for Unsupervised Text-to-Image Person Re-Identification.pdf' - | nl -ba | sed -n '660,825p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   660	DFLSG, we adopt the Adam optimizer with an initial learning
   661	rate of 1 × 10−5 , which gradually cosine decays. In addition, we further evaluate several important hyperparameters in
   662	Section IV-D.
   663	B. Ablation Studies
   664	As shown in Table I, we conduct ablation studies on three
   665	cross-modality databases to evaluate the effectiveness of each
   666	component in DFLSG.
   667	No.1 indicates that a kind of MLLM is utilized to generate
   668	text descriptions for building the cross-modality dataset. Then,
   669	we perform the clustering operation to assign pseudo-labels
   670	for unlabeled image-text pairs. Contrastive loss and SDM loss
   671	are applied to constrain pedestrian features. Compared to other
   672	approaches, this method obtains the worst results, which shows
   673	that the model fails to learn reliable pedestrian features.
   674	No.2 represents that MTJL is added to No.1, which continuously constrains cross-modality feature distribution from
   675	center and instance views. Comparing the results in No.1 vs.
   676	No.2, we observe a significant performance improvement,
   677	which demonstrates the effectiveness of MTJL for unsupervised TIReID in reducing intra-class variations and increasing
   678	inter-class distances.
   679	No.3 denotes that MLLM1 in No.2 is replaced by MLLM2
   680	to evaluate the impact of text descriptions from different
   681	MLLMs. Comparing the results in No.2 vs. No.3, they illustrate that the quality of text descriptions generated by different
   682	MLLMs is inconsistent. Using a kind of MLLM to generate
   683	
   684	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:02 UTC from IEEE Xplore. Restrictions apply.
   685	
   686	SHAO et al.: DFLSG FOR UNSUPERVISED TEXT-TO-IMAGE PERSON RE-IDENTIFICATION
   687	
   688	1009
   689	
   690	TABLE I
   691	A BLATION S TUDY ON T HREE M AINSTREAM DATABASES
   692	
   693	TABLE II
   694	E VALUATION OF D IFFERENT F ILTERING M ETHODS
   695	ON THE CUHK-PEDES DATABASE
   696	
   697	The experimental results show that three clustering algorithms
   698	achieve similar performance. This indicates that our unsupervised framework does not rely on any specific clustering
   699	method and possesses strong generalization ability.
   700	C. Comparison With State-of-the-Art Methods
   701	
   702	TABLE III
   703	E VALUATION OF D IFFERENT C LUSTERING A LGORITHMS
   704	ON THE ICFG-PEDES DATABASE
   705	
   706	texts may be unreliable, which results in fluctuations in model
   707	performance.
   708	In No.5, we propose a dependable text generation method
   709	to construct the image-text matching relationship. Compared
   710	to No.2 and No.3, the dependable text generation method
   711	significantly enhances retrieve performance. This not only
   712	demonstrates that the method could obtain reliable text
   713	descriptions but also indicates that the proposed unsupervised
   714	TIReID framework effectively learns dependable features.
   715	No.4 and No.6 represent that ESFM is added to No.3 and
   716	No.5 methods to eliminate abnormal samples in the clustering
   717	process. From Table I, the No.6 setting achieves the best
   718	experimental results, which fully demonstrate the effectiveness
   719	of ESFM. In addition, all results also prove that the proposed
   720	DFLSG is effective for the unsupervised TIReID task, which
   721	bridges the large inter-modality gap.
   722	To further evaluate the performance of ESFM, we conduct
   723	experiments to compare IQR with other filtering algorithms.
   724	As shown in Table II, Z-score and LOF represent two different
   725	algorithms that measure the average deviations or local density
   726	differences to detect abnormal samples. Compared to them,
   727	ESFM with the IQR filtering algorithm achieves the best
   728	performance. This not only proves the importance of removing
   729	abnormal samples but also illustrates that our method obtains
   730	reliable image-text pairs.
   731	In addition, we assess the model performance under different clustering algorithms, as shown in Table III. K-means
   732	and Hierarchical represent the other two common clustering
   733	algorithms, which are utilized to generate identity labels.
   734	Since they require a predefined cluster number, we leverage
   735	the approximate range produced by DBSCAN as guidance.
   736	
   737	In this section, we compare DFLSG with other advanced
   738	supervised and unsupervised TIReID methods on three mainstream databases.
   739	We first evaluate the performance of DFLSG on the first
   740	benchmark dataset, CUHK-PEDES, as shown in Table IV. Our
   741	method achieves the best identification performance among
   742	unsupervised methods, with a Rank-1 accuracy of 50.20%
   743	and an mAP accuracy of 46.93%. Notably, our method
   744	even outperforms supervised learning method. These results
   745	demonstrate that DFLSG can effectively learn dependable
   746	cross-modality features from unlabeled samples and enhance
   747	semantic consistency between image and text modalities.
   748	Since the query texts in ICFG-PEDES are close to realworld scenarios, this database is more challenging. As shown
   749	in Table IV, we present the experimental results on ICFGPEDES, which achieves excellent performance, with a Rank-1
   750	accuracy of 37.41% and an mAP accuracy of 22.37%. Compared to GAAP, the best unsupervised method in Table IV,
   751	our method surpasses it by 10.29% in Rank-1 accuracy and
   752	10.94% in mAP accuracy. This demonstrates that DFLSG is
   753	effective for the unsupervised TIReID task and significantly
   754	improve retrieval performance.
   755	Finally, we conduct extensive experiments on RSTPReID
   756	database. As shown in Table IV, the key metrics of Rank-1
   757	and mAP reach 47.05% and 37.61%, respectively, surpassing
   758	all other unsupervised competitive methods and some supervised learning methods. This is because the dependable text
   759	generation and ESFM effectively enhance the reliability of
   760	the training data, and MTJL continuously optimizes the crossmodality feature distribution from center and instance views,
   761	which could learn dependable cross-modality pedestrian
   762	features.
   763	D. Parameter Analysis
   764	λ1 denotes the importance of the triplet loss in Eq. 6 and the
   765	impact is exhibited in Fig 4 (a). When λ1 is set to 1, DFLSG
   766	achieves the best performance. The appropriate parameter promotes intra-class features to generate the compact distribution.
   767	The parameter λ2 represents the importance of the extended
   768	center loss, and the evaluation results are shown in Fig 4 (b)
   769	and Fig 4 (c). If λ2 is too small, it fails to constrain the
   770	
   771	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:02 UTC from IEEE Xplore. Restrictions apply.
   772	
   773	1010
   774	
   775	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   776	
   777	TABLE IV
   778	P ERFORMANCE C OMPARISON W ITH A DVANCED M ETHODS ON CUHK-PEDES, ICFG-PEDES, AND RSTPR EID DATABASES .
   779	* R EPRESENTS THE P ERFORMANCE OF THE M ETHOD U NDER U NSUPERVISED S ETTINGS
   780	
   781	Fig. 4. Evaluation of four important loss parameters in MTJL on the ICFG-PEDES and CUHK-PEDES databases.
   782	
   783	TABLE V
   784	
   785	TABLE VI
   786	
   787	E VALUATION OF D IFFERENT F ILTERING S TRENGTH
   788	ON THE CUHK-PEDES DATABASE
   789	
   790	E VALUATION OF D IFFERENT F ILTERING S TRENGTH ON
   791	THE RSTPR E ID DATABASE
   792	
   793	feature distances, while if λ2 too large, the model heavily
   794	relies on the pedestrian class center. When λ2 is 0.08, the
   795	model achieves the best performance on ICFG-PEDES and
   796	RSTPReID. In addition, when λ2 is 0.04, the performance
   797	generates the highest accuracy on CUHK-PEDES.
   798	λ3 and λ4 represent the importance of inter-modality
   799	and intra-modality losses in Eq. 9, as shown in Fig. 4 (d)
   800	and Fig. 4 (e). The experimental results indicate that when
   801	λ3 and λ4 are both set to 1, the model achieves the best
   802	performance. However, when λ3 is equal to 0, the model
   803	performance significantly declines. This proves that the intermodality loss plays a crucial role in unsupervised TIReID.
   804	β represents the strength of the filtering algorithm in ESFM.
   805	The selection of β is determined based on the clustering
   806	stability across different databases. Specifically, for databases
   807	with rough pedestrian images, the clustering results tend to
   808	be unstable. Selecting a smaller β can retain more potentially
   809	
   810	E VALUATION OF D IFFERENT M AXIMUM S AMPLE D ISTANCE
   811	IN DBSCAN A LGORITHM
   812	
   813	TABLE VII
   814	
   815	useful information. In contrast, for databases with high quality
   816	and stable clustering performance, a larger β can be set to
   817	effectively eliminate abnormal samples. We conduct extensive experiments to evaluate the parameter β as shown in
   818	Table V and Table VI. For CUHK-PEDES, the best accuracy
   819	is obtained when β is equal to 1.0. For RSTPReID and
   820	ICFG-PEDES, DFLSG achieves the highest identification
   821	
   822	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:04:02 UTC from IEEE Xplore. Restrictions apply.
   823	
   824	SHAO et al.: DFLSG FOR UNSUPERVISED TEXT-TO-IMAGE PERSON RE-IDENTIFICATION
   825	

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf' - | nl -ba | sed -n '985,1065p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   985	Ablation Experiments
   986	
   987	In this section, we performed ablation experiments to evaluate the effectiveness of each module. All
   988	experiment were trained and evaluated on the HITSZ-VCM datasets. Table 3 presents the ablation
   989	results for the key components of our method, namely CIM and FDM. From this table, we can make
   990	the following interesting observations:
   991	— The introduced CIM module provides an 8.97% improvement in Rank-1 accuracy and approximately a 6.7% boost in mAP over the baseline. This demonstrates its effectiveness in enhancing
   992	representation learning.
   993	— The designed FDM method enhances Rank-1 accuracy by 8.16% and mAP by 4.51%, highlighting
   994	the importance of temporal cues in video sequences.
   995	— The components above work synergistically. By integrating them, our final method achieves
   996	a 9.6% improvement in Rank-1 accuracy and a 7.13% boost in mAP over the baseline. These
   997	results clearly highlight the effectiveness of the two proposed modules.
   998	4.7
   999	
  1000	Visualization
  1001	
  1002	To showcase the success of our method, we randomly choose video clips to illustrate retrieval
  1003	outcomes in both infrared-visible and visible-infrared modes. Figure 4 illustrates the Rank-5 retrieval
  1004	results for two person queries in both modes. Video outlined in green boundaries represent accurate
  1005	matches related to the query’s ID, whereas those outlined in red signify incorrect matches from
  1006	different IDs. In the infrared-visible mode, our method excels at identifying pedestrian clips with
  1007	similar color patterns, especially by matching the unique back designs of their clothing. In the
  1008	visible-infrared mode, our approach effectively retrieves the correct clips by recognizing the
  1009	distinctive poses and actions of pedestrians.
  1010	In Figure 5, we utilize t-SNE [42] to illustrate the distributions of features. We randomly choose
  1011	eight classes from the HITSZ-VCM dataset to compare the feature distributions of our method
  1012	against the baseline [29]. Figure 5(a) and (b) presents the feature distributions from the baseline
  1013	and our method, respectively. In Figure 5(a), the baseline method effectively clusters intra-class
  1014	samples across different modalities but struggles to differentiate samples from various classes.
  1015	Conversely, Figure 5(b) shows that our method not only groups samples from the same class across
  1016	two modalities but also more successfully separates samples from different classes, preventing the
  1017	overlap of dissimilar samples. This indicates that our approach improves classification performance
  1018	by increasing sample diversity, allowing the model to more effectively distinguish positive samples
  1019	from challenging negative ones.
  1020	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.
  1021	
  1022	DIRL for Video Visible-Infrared Person ReID
  1023	
  1024	238:13
  1025	
  1026	Fig. 4. The Rank-5 results for certain queries using our approach in the infrared-visible and visible-infrared
  1027	modes on the HITSZ-VCM dataset. True matches are indicated by a green rectangle, whereas false matches
  1028	are highlighted with a red rectangle. The top row corresponds to visible-infrared retrieval, while the second
  1029	row pertains to infrared-visible retrieval.
  1030	
  1031	Fig. 5. The visualization illustrates the feature distributions for both the baseline method and our approach.
  1032	Each color corresponds to a unique identity, with visible and infrared modalities represented by solid circles
  1033	and triangles, respectively.
  1034	
  1035	4.8
  1036	
  1037	Parameter Analysis
  1038	
  1039	Our model encompasses various parameters, including the number of frames in each video sequence
  1040	and the value of parameter 𝑆 in FDM. These parameters may have an impact on the model’s
  1041	performance. Therefore, we designed a series of parameters analysis experiments to examine the
  1042	effects of these parameters on the model.
  1043	The experimental results depicted in Figure 6 showcase the comparative performance of our
  1044	methods across a range of frames extending from 4 to 8. Notably, our method demonstrates
  1045	a consistent and stable enhancement in performance as the number of frames increases. This
  1046	improvement can be attributed to the method’s proficiency in establishing enduring dependencies
  1047	within exceptionally long videos while also preserving the integrity of identity information. By
  1048	maintaining this vital information, our approach effectively reduces the impact of noise and captures
  1049	robust discriminative features. This inherent capability not only aids in noise reduction but also
  1050	facilitates the extraction of durable discriminative features, thereby underscoring the method’s
  1051	resilience and adaptability across diverse frame lengths.
  1052	As depicted in Figure 7, we showcase the experimental results of the six parameters, analyzing a
  1053	spectrum of 𝑆 values ranging from 1,024 to 6,144. The accuracy of our model improves with the
  1054	increase in the value of S. This clearly demonstrates the positive impact of higher-dimensional
  1055	modality encoder and modality decoder matrices on enhancing the model’s accuracy. Higherdimensional modality encoder and modality decoder matrices offer a more expressive representation
  1056	of the data. With increased dimensionality, the model can capture more intricate patterns and
  1057	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.
  1058	
  1059	238:14
  1060	
  1061	J. Wang et al.
  1062	
  1063	Fig. 6. Performance comparison across different frame counts in the infrared-visible and visible-infrared
  1064	modes on the HITSZ-VCM dataset.
  1065	

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf' - | nl -ba | sed -n '910,985p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   910	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 238. Publication date: August 2025.
   911	
   912	238:12
   913	
   914	J. Wang et al.
   915	
   916	Table 3. Ablation Results of CMC (%) and mAP (%) Performances of Key Components on HITSZ-VCM
   917	Method
   918	
   919	CIM FDM
   920	
   921	Baseline
   922	
   923	Ø
   924	Ø
   925	
   926	Ø
   927	Ø
   928	
   929	Rank-1
   930	55.58
   931	64.55
   932	63.74
   933	65.18
   934	
   935	Infrared to Visible
   936	Rank-5 Rank-10 Rank-20 mAP
   937	70.75
   938	77.01
   939	82.16
   940	40.80
   941	78.23
   942	82.72
   943	86.69
   944	47.50
   945	76.88
   946	81.72
   947	86.28
   948	45.31
   949	79.12
   950	84.60
   951	87.78
   952	47.93
   953	
   954	Rank-1
   955	56.59
   956	66.64
   957	64.54
   958	66.99
   959	
   960	Visible to Infrared
   961	Rank-5 Rank-10 Rank-20 mAP
   962	73.49
   963	79.58
   964	83.51
   965	45.99
   966	79.45
   967	84.08
   968	87.80
   969	49.64
   970	76.28
   971	82.98
   972	86.28
   973	47.69
   974	81.72
   975	84.19
   976	88.48
   977	50.17
   978	
   979	Bold means the best result.
   980	
   981	proposed. Furthermore, we once again found that our method performs better than existing feature
   982	decoupling methods.
   983	4.6
   984	
   985	Ablation Experiments

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf' - | rg -n -i 'Ablation|Effect|HiLo|FADE|CMSR|CMRR|visual|frequency|Table [IVX]+|Fig\\. [0-9]' | sed -n '1,220p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
5:Discovering Multi-Frequency Embedding for
12:to fully exploit frequency domain information, focusing predominantly on spatial domain feature learning or limited frequency
13:decompositions. To address this, we propose the Multi-Frequency
15:operates in the frequency domain through multi-frequency
16:decomposition to learn discriminative and modality-invariant features. Specifically, the HiLo-Frequency Modulation (HiLo-FM)
17:module efficiently extracts low-frequency features via frequencydomain filtering and high-frequency details through lightweight
19:Frequency-Aware Diversity Enhancer (FADE) module further
20:enriches feature discriminability by weighting multi-frequency
24:Soft Retrieval (CMSR) loss prioritizes cross-modality consistency over intra-modality similarity, while the Cross-Modality
25:Ranking Regularization (CMRR) loss enhances feature diversity
83:features, leaving the frequency domain insufficiently leveraged
85:Recent advances in frequency domain analysis have
88:enhancement [29]. As shown in Fig. 2, Fourier analysis reveals that amplitude spectra (Fig. 2(b)) primarily
90:(Fig. 2(c)) preserve structural information across modalities. These findings have motivated several frequency-based
93:both spectral components. These methods perform broadspectrum feature mining across the entire frequency band (top
94:left of Fig. 1), but overlook the distinct roles of low/highfrequency in modality-invariant feature learning.
101:GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
105:Fig. 1. The comparison of existing frequency-based methods and our proposed
107:across the entire frequency band or rely on simple low/high-frequency decomposition, which often adopts coarse and inefficient feature learning in the
108:spatial domain. Conversely, our method achieves efficient low/high-frequency
109:feature learning through the meticulously designed HiLo-FM module
110:(Sec. III-B). Meanwhile, the FADE module (Sec. III-C) leverages multifrequency attention to generate richer, more robust, and modality-invariant
113:Further decomposition into low-frequency (Fig. 2(d))
114:and high-frequency (Fig. 2(e)) components provides deeper
115:insights. The low-frequency component, capturing the overall
118:or thermal radiation in IR images. In contrast, the highfrequency component effectively filters out modality-specific
120:[19]. Notable methods for low/high-frequency decomposition
121:include FSDF [33] and BiFFN [34] (top right of Fig. 1).
126:have demonstrated certain effectiveness, two critical challenges
127:persist. First, they predominantly execute low/high-frequency
129:the potential for efficient frequency-domain processing. Second, and more critically, they rely on limited frequency
130:decompositions, focusing solely on low/high-frequency components. This limitation is particularly suboptimal, as our
131:visual analysis in Fig. 2 reveals. While low-frequencies
132:(Fig. 2(d)) are dominated by modality-specific styles and highfrequencies (Fig. 2(e)) better preserve shared structures, a
136:Fig. 2. The visualization of frequency components for a pair of VIS (odd
138:and the reconstructions from (b) amplitude (A), and (c) phase (P). The lowfrequency (L) and high-frequency (H) components, shown in (d) and (e), are
139:separated using a frequency mask as detailed in Eq.(4). The multi-frequency
141:non-overlapping bands, formally defined in Eq.(17). This visualization highlights how modality-invariant cues are distributed across different frequency
144:the person’s clothing), are scattered across various frequency
146:as shown in the multi-frequency decompositions (Fig. 2(f)).
147:Relying on a single, handcrafted frequency cutoff thus forces
150:the necessity of a more flexible, multi-frequency analysis to
153:Multi-Frequency Embedding Network (MFENet), a featurelevel method that learns modality-invariant representations
154:from the frequency domain via multi-frequency decompositions. Specifically, to address the challenge of efficient
155:low/high-frequency feature extraction, we introduce the HiLoFrequency Modulation (HiLo-FM) module. This module
157:[36] for low/high-frequency decomposition, directly performs
158:efficient filtering on low-frequency features in the frequency
167:on high-frequency features in the spatial domain. To mitigate
168:unavoidable noise, these low/high-frequency features are then
170:multi-frequency feature learning, we propose the FrequencyAware Diversity Enhancer (FADE) module, which can be
172:in the frequency domain. By weighting multiple pre-set frequencies from low to high, the module accentuates feature
176:robustness and diversity of cross-modality features: CrossModality Soft Retrieval (CMSR) loss and Cross-Modality
177:Ranking Regularization (CMRR) loss. The CMSR loss is
184:Meanwhile, the CMRR loss is dedicated to augmenting the
186:by different branches in the FADE module. To tackle the
191:in the frequency domain and introduces multi-frequency
193:for VI-ReID. This method incorporates two key modules: the HiLo-FM module, which efficiently extracts and
194:fuses low/high-frequency features, and the FADE module, which effectively enhances feature discriminability
195:through diverse frequency responses.
197:retrieval tasks: CMSR loss, which enhances the robustness of cross-modality features by focusing more
198:on cross-modality features learning rather than intramodality features learning, and CMRR loss, which
203:the effectiveness of our designed modules and loss functions, and demonstrate that MFENet outperforms most
225:inherent in VI-ReID tasks effectively.
247:frequency-domain cues that can be modality-invariant. In this
249:in the frequency domain, conducting multi-frequency feature
251:B. Frequency-Domain Analysis in Deep Learning
252:Frequency-domain analysis [43], a subfield of image
253:processing, converts spatial domain images into frequency
256:pixel values, frequency-domain analysis dissects images into
257:various frequency components, thereby facilitating the differentiation and manipulation of subtle features such as noise and
258:texture. Recent advancements in frequency-domain analysis
263:[33], [34] have initiated the exploration of frequency-domain
268:GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
272:are crucial for effective feature extraction. However, these pioneering works have largely overlooked the frequency-domain
274:low/high-frequency components to modality-invariant feature
277:to decouple high/low-frequency branches for more robust
279:discrete cosine transform to decompose low/high-frequency
283:low/high-frequency decomposition and processes these features in the spatial domain using computationally intensive
285:low/high-frequency features, they both still perform feature
288:In this work, as Fig. 1 illustrates, we first achieve efficient
289:low/high-frequency feature learning and fusion through meticulously designed spatial/frequency operations. Subsequently,
290:we leverage multi-frequency attention to generate richer, more
302:As illustrated in Fig. 3(a), the input images {I(V)
308:style of an image, while high frequencies are often modalityagnostic and contain detailed features [44], a HiLo-Frequency
309:Modulation (HiLo-FM) module (as shown in Fig. 3(b), in
311:modality-invariant feature learning on low/high-frequency features, respectively. Inspired by Instance Normalization [26],
312:[30], [37], a Frequency-Aware Diversity Enhancer (FADE)
313:module (as shown in Fig. 3(c), in Sec. III-C) is incorporated at the end of layer3 . This module further explores rich
315:weighting across multi-frequency bands from low to high,
352:loss and weighted regularization triplet loss [1], a CrossModality Soft Retrieval (CMSR) loss and a Cross-Modality
353:Ranking Regularization (CMRR) loss are specifically designed
362:B. HiLo-Frequency Modulation Module
365:style and illumination (predominantly low-frequency) from
366:identity-bearing structures and textures (predominantly highfrequency), which facilitates cross-modality alignment in
367:VI-ReID. Unlike prior frequency-based methods that merely
370:HiLo-Frequency Modulation (HiLo-FM) module (Fig. 3(b))
371:performs learnable frequency-domain filtering to efficiently
372:normalize low-frequency style, enhances high-frequency identity cues via lightweight multiscale convolutions in the spatial
375:1) HiLo-Frequency Decomposition: Given an input feature
376:map X ∈ RC×H×W , we first transform it to the frequency
405:In the frequency domain, a low-frequency binary mask Π ∈
412:where the hyperparameter ρ ∈ [0, 1] balances low/highfrequency components, and a square-shaped method is used to
413:separate these components. Alternatively, circular or diamondshaped methods could be applied, and subsequent ablation
414:studies are conducted to compare these three methods. Accordingly, the high-frequency binary mask can be obtained by
422:Fig. 3. Overall framework of Multi-Frequency Embedding Network (MFENet). (a) The high-level architecture, showing the integration of our novel modules
423:and loss functions. (b, c) Detailed schematics of our core technical innovations: the HiLo-Frequency Modulation (HiLo-FM) and Frequency-Aware Diversity
424:Enhancer (FADE) modules. (d, e) Illustrations of the key principles behind our novel Cross-Modality Soft Retrieval (CMSR) and Cross-Modality Ranking
425:Regularization (CMRR) loss functions.
427:Π̄ = 1 − Π, then the low/high-frequency components can be
438:2) Low-Frequency Adaptation: Given that low-frequency
443:GFNet [47], we propose to enhance low-frequency features
444:directly in the frequency domain by introducing a parameterized complex weight matrix W̃L ∈ CC×H×W :
454:by the low-frequency mask. Additionally, the real-valued 2D
456:to retaining the full complex frequency domain size. As a
457:result, the effective parameter count of W̃L is reduced to
459:CbρHcbρ(W/2 + 1)c. Finally, we transform the adapted lowfrequency component back to the spatial domain via iFFT:
464:3) High-Frequency Enhancement: High-frequency features, which capture various scales of details and contours of
467:via convolutions. As shown in Fig. 2, a single 3×3 convolution
470:in Fig. 4), inspired by OSNet [48], for efficient multi-scale
473:F −1 (X̃H ) ∈ RC×H×W . To integrate the sparse high-frequency
486:GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
501:In summary, the HiLo-FM module effectively integrates
502:comprehensive frequency domain processing, efficient feature enhancement, and spatial-attention-based modulation
506:C. Frequency-Aware Diversity Enhancer Module
507:Fig. 4. The architecture of the Lite Multiscale (LM) module for HiLo-FM.
568:on Eq.(7) and (12), the processed low/high-frequency features
570:introduced in the low/high-frequency features, spatial masks
576:low-frequency feature XL and the high-frequency feature XH
577:spatially. The modulated low/high-frequency features are then
598:adoption for suppressing modality-specific features by normalizing frequency components: zeroing the DC component while
613:IN-style designs effectively impose a coarse “DC vs.
615:content. Inspired by the multi-frequency decompositions
616:in Fig. 2(f), we introduce the Frequency-Aware Diversity
617:Enhancer (FADE) (Fig. 3(c)) to learn frequency-band–specific
619:1) Multi-Frequency Decomposition: First, the amplitude
623:according to frequency levels:
631:regions, consistent with the HiLo-frequency decomposition in
633:the subsequent ablation studies.
635:Fig. 3(c)):
660:2) Frequency-Aware Attention: Next, a lightweight
661:frequency-aware module is employed to generate attention
707:In summary, the FADE module provides superior frequency
709:enabling more precise preservation of modality-invariant characteristics across different frequency bands.
718:leave a residual modality gap. To align training with evaluation, we propose the Cross-Modality Soft Retrieval (CMSR)
770:pairs, ignoring the effects of the remaining sample pairs.
857:enhance the diversity of features obtained by the FADE
859:(CMRR) loss based on differentiable Spearman rank
885:GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
969:LCMRR
1000:LCMRR =
1005:LCMRR + LCMRR
1013:Ltotal = LID + LWRT + λ1 LCMS R + λ2 LCMRR ,
1017:where λ1 and λ2 are hyperparameters controlling the tradeoff between different objectives. We apply LCMS R and LCMRR
1022:hyperparameter λ2 of LCMRR , which is set to one-tenth of
1047:well-known AGW baseline [1], our HiLo-FM module and
1048:FADE module are positioned after the final two residual blocks
1051:are shown in Fig. 3(a). In HiLo-FM, ρ = 0.2; in FADE,
1064:The parameters are set as γ = 0.1, m = 0.1 in CMSR, and
1073:Table I and Table II, where bold and underlined fonts indicate
1075:1) SYSU-MM01: As shown in Table I, MFENet achieves
1084:our approach. This could be a potential direction for further improvement. Among methods using frequency-domain
1094:TABLE I
1105:MFENet’s robustness and effectiveness in various search scenarios.
1106:2) RegDB: As depicted in Table I, within the IR to VIS
1110:(Rank-1 94.82%, the best among frequency domain methods)
1125:3) LLCM: As shown in Table II, in the IR to VIS
1139:GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
1143:TABLE II
1155:Fig. 5. A diagram illustrating the relationship between (a) other methods and
1156:(b) our frequency-aware attention.
1158:TABLE III
1168:of four evaluated metrics. This underscores the effectiveness
1171:C. Ablation Studies
1172:In this subsection, we conduct a series of ablation experiments to evaluate the effectiveness of our proposed MFENet.
1173:1) Effectiveness of Proposed Components: The effectiveness of the proposed components is evaluated on the
1174:SYSU-MM01 and LLCM datasets, with detailed results presented in Table III. Variants-1, serving as the baseline, utilizes
1176:LWRT as loss functions. Variants-2, incorporating the HiLoFM module, achieves a significant performance improvement,
1179:Fig. 6. The effectiveness of decomposition hyperparameters in HiLo-FM
1180:and FADE modules on the SYSU-MM01 dataset (Single-Shot mode of
1182:(Eq.(4)) and the different decomposition strategies in HiLo-FM. (c) and (d)
1184:decomposition strategies in FADE.
1186:Variants-1. The FADE module in Variants-3 further enhances
1188:in mAP. When the LCMRR loss function is introduced in
1192:indicate that the HiLo-FM and FADE modules have distinct
1193:and complementary roles. Variants-6 and Variants-7, employing LCMHR and LCMSR respectively, demonstrate enhanced
1195:with the soft version of LCMSR providing a more stable
1197:proposed components, achieves the best performance, validating the effectiveness and complementarity of the proposed
1199:2) Ablation Study on the HiLo-Frequency Decomposition Hyperparameter: The influence of the hyperparameter
1208:within the HiLo-FM module is illustrated by the deep blue
1209:curve in Fig. 6 (a)(b). Specifically, when ρ = 0, the frequency
1212:ρ = 1, all frequency components are processed by the filtering,
1214:Fig. 6 (a)(b) represents the scenario where all frequency components are processed by the LM module, akin to the strategy
1217:incorporating a small amount of low-frequency information,
1220:partial integration of low-frequency components significantly
1224:reliance on low-frequency components (i.e., larger ρ values)
1225:undermines the learning of high-frequency information, which
1227:highlight the importance of balancing low/high-frequency
1229:capabilities. The selection of ρ thus represents a critical tradeoff between these two types of frequency components.
1230:3) Comparison with the HiLo-Frequency Decomposition
1232:three distinct decomposition strategies within the HiLo-FM
1234:Fig. 6 (a)(b). All three strategies achieve their peak performance at ρ = 0.2, indicating that a moderate incorporation
1235:of low-frequency information is universally advantageous for
1238:suggests that the square decomposition is more effective in
1239:leveraging frequency-decomposed features for the VI-ReID
1241:performance of HiLo-frequency decomposition for VI-ReID
1243:4) Ablation Study on the Number of Multi-Frequency
1245:B (Eq. (17)) on the performance of the FADE module is
1246:illustrated by the deep blue curve in Fig. 6 (c)(d). Specifically,
1249:frequency components, similar to the approach in IN.
1250:The second scenario mirrors the HiLo-FM method, where
1251:low/high-frequency components are separated using ρ = 0.2.
1252:The results indicate that separating low/high-frequency
1254:and other frequency components are considered. This finding
1255:underscores the benefits of employing multiple frequency
1259:performance of feature learning using multiple frequency
1261:TABLE IV
1272:frequency components are learned in a point-wise manner, as
1273:depicted by the red dashed line in the Fig. 6 (c)(d). Although
1280:5) Comparison with Multi-Frequency Decomposition Methods: A thorough comparison was conducted among three
1281:distinct decomposition strategies within the FADE module:

codex
证据部分已补到：CycleTrans 有模块消融、DMM 替代、t-SNE 和 Grad-CAM；DATE 有融合方式、文本约束和跨粒度训练消融；DFLSG 有文本生成、异常过滤、聚类替换和参数分析；DIRL 有 CIM/FDM 消融和可视化。Multi-Frequency 的消融很多，我再把具体段落读完整。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1168,1308p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1168	of four evaluated metrics. This underscores the effectiveness
  1169	of MFENet’s approach in learning discriminative features for
  1170	cross-modality re-identification on the LLCM dataset.
  1171	C. Ablation Studies
  1172	In this subsection, we conduct a series of ablation experiments to evaluate the effectiveness of our proposed MFENet.
  1173	1) Effectiveness of Proposed Components: The effectiveness of the proposed components is evaluated on the
  1174	SYSU-MM01 and LLCM datasets, with detailed results presented in Table III. Variants-1, serving as the baseline, utilizes
  1175	the original ResNet50 architecture and employs LID and
  1176	LWRT as loss functions. Variants-2, incorporating the HiLoFM module, achieves a significant performance improvement,
  1177	with an approximate 7% to 8% increase in mAP compared to
  1178	
  1179	Fig. 6. The effectiveness of decomposition hyperparameters in HiLo-FM
  1180	and FADE modules on the SYSU-MM01 dataset (Single-Shot mode of
  1181	All-Search): (a) and (b) denote the performances of the hyperparameter ρ
  1182	(Eq.(4)) and the different decomposition strategies in HiLo-FM. (c) and (d)
  1183	denote the performances of the hyperparameter B (Eq.(17)) and the different
  1184	decomposition strategies in FADE.
  1185	
  1186	Variants-1. The FADE module in Variants-3 further enhances
  1187	performance, yielding an approximate 7.4% to 9% increase
  1188	in mAP. When the LCMRR loss function is introduced in
  1189	Variants-4 (based on Variants-3), feature diversity is further enhanced, resulting in an approximate 8.7% to 10.6%
  1190	increase in mAP compared to Variants-1. The performance
  1191	improvements of Variants-5 over Variants-2 and Variants-3
  1192	indicate that the HiLo-FM and FADE modules have distinct
  1193	and complementary roles. Variants-6 and Variants-7, employing LCMHR and LCMSR respectively, demonstrate enhanced
  1194	robustness and better performance compared to Variants-5,
  1195	with the soft version of LCMSR providing a more stable
  1196	increase in mAP. Finally, Variants-8, which incorporates all
  1197	proposed components, achieves the best performance, validating the effectiveness and complementarity of the proposed
  1198	components.
  1199	2) Ablation Study on the HiLo-Frequency Decomposition Hyperparameter: The influence of the hyperparameter
  1200	
  1201	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.
  1202	
  1203	1776
  1204	
  1205	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
  1206	
  1207	ρ (Eq. (4)) on the performance of the square-shaped strategy
  1208	within the HiLo-FM module is illustrated by the deep blue
  1209	curve in Fig. 6 (a)(b). Specifically, when ρ = 0, the frequency
  1210	domain is divided into two segments: the direct current (DC)
  1211	component and the remaining components. In contrast, when
  1212	ρ = 1, all frequency components are processed by the filtering,
  1213	similar to the approach in GFNet [47]. The dashed line (M) in
  1214	Fig. 6 (a)(b) represents the scenario where all frequency components are processed by the LM module, akin to the strategy
  1215	in OSNet [48]. As ρ increases from 0.0 to 0.2, a significant
  1216	performance improvement is observed. This indicates that
  1217	incorporating a small amount of low-frequency information,
  1218	controlled by ρ = 0.2, is highly beneficial for learning
  1219	modality-invariant features. This optimal point suggests that
  1220	partial integration of low-frequency components significantly
  1221	enhances the model’s discriminative power. However, as ρ
  1222	continues to increase towards 1.0, a consistent degradation in
  1223	performance is observed. This decline implies that excessive
  1224	reliance on low-frequency components (i.e., larger ρ values)
  1225	undermines the learning of high-frequency information, which
  1226	is crucial for modality invariance. Overall, these observations
  1227	highlight the importance of balancing low/high-frequency
  1228	features to improve model robustness and feature learning
  1229	capabilities. The selection of ρ thus represents a critical tradeoff between these two types of frequency components.
  1230	3) Comparison with the HiLo-Frequency Decomposition
  1231	Methods: A comprehensive comparison is conducted among
  1232	three distinct decomposition strategies within the HiLo-FM
  1233	module: square, circular, and diamond, as illustrated in
  1234	Fig. 6 (a)(b). All three strategies achieve their peak performance at ρ = 0.2, indicating that a moderate incorporation
  1235	of low-frequency information is universally advantageous for
  1236	feature learning. Specifically, the square decomposition strategy consistently outperforms both the circular and diamond
  1237	decomposition strategies across the majority of ρ values. This
  1238	suggests that the square decomposition is more effective in
  1239	leveraging frequency-decomposed features for the VI-ReID
  1240	task. These highlight the critical role of both the decomposition strategy and the hyperparameter ρ in optimizing the
  1241	performance of HiLo-frequency decomposition for VI-ReID
  1242	tasks.
  1243	4) Ablation Study on the Number of Multi-Frequency
  1244	Decompositions: The impact of the hyperparameter
  1245	B (Eq. (17)) on the performance of the FADE module is
  1246	illustrated by the deep blue curve in Fig. 6 (c)(d). Specifically,
  1247	when B = 2, two distinct scenarios were examined. The
  1248	first scenario considers only the DC component and other
  1249	frequency components, similar to the approach in IN.
  1250	The second scenario mirrors the HiLo-FM method, where
  1251	low/high-frequency components are separated using ρ = 0.2.
  1252	The results indicate that separating low/high-frequency
  1253	components outperforms the scenario where only the DC
  1254	and other frequency components are considered. This finding
  1255	underscores the benefits of employing multiple frequency
  1256	decompositions for enhancing the model’s robustness and
  1257	feature learning capability. As B increases, this advantage
  1258	is further corroborated. However, when B exceeds 7, the
  1259	performance of feature learning using multiple frequency
  1260	
  1261	TABLE IV
  1262	T HE PARAMETERS AND FLOP S S TATISTICS OF O UR P ROPOSED MFEN ET
  1263	AND OTHER M ETHODS D URING I NFERENCE . M ETHODS U SING CLIP
  1264	AS THE BACKBONE A RE M ARKED BY ∗ . I N CLIP-BASED M ETH ODS , THE PARAMETER Q UANTITY IN PARENTHESES I S THE
  1265	N UMBER OF M ODEL PARAMETERS D URING THE T RAIN ING P HASE . T HE † I NDICATES T HAT THE R ESULT I S
  1266	E STIMATED BASED ON THE M ODEL F RAMEWORK
  1267	D ESCRIBED IN T HEIR PAPER . T HE ‡ I NDI CATES T HAT THE R ESULT I S R EIMPLEMENTED
  1268	BASED ON THE O PEN -S OURCE C ODE
  1269	
  1270	decompositions begins to deteriorate. Additionally, we
  1271	compared this approach with a more extreme case, where all
  1272	frequency components are learned in a point-wise manner, as
  1273	depicted by the red dashed line in the Fig. 6 (c)(d). Although
  1274	the point-wise learning approach can achieve relatively good
  1275	performance, it still underperforms compared to the limited
  1276	number of low-to-high decomposition strategies. This finding
  1277	further demonstrates that our proposed method can fully
  1278	leverage the spatial characteristics of frequencies from low to
  1279	high.
  1280	5) Comparison with Multi-Frequency Decomposition Methods: A thorough comparison was conducted among three
  1281	distinct decomposition strategies within the FADE module:
  1282	square, circular, and diamond, as shown in Fig. 6 (c)(d). The
  1283	performance trends of these three decomposition methods are
  1284	generally consistent, with the square decomposition method
  1285	achieving marginally superior results.
  1286	6) Discussion on the Parameters and Computational Complexity: Table IV provides a detailed comparison of our
  1287	proposed MFENet with existing methods in terms of model
  1288	efficiency (parameters and computational load) and performance metrics (Rank-1 and mAP) on the SYSU-MM01 and
  1289	LLCM datasets. Specifically, MFENet strikes a favorable
  1290	balance between efficiency and performance. While methods
  1291	like AGW (23.55M parameters, 9.21 GFLOPs) are more
  1292	parameter- and computationally-efficient, MFENet consistently delivers superior retrieval performance. For example, on
  1293	the SYSU-MM01 dataset, MFENet surpasses AGW’s Rank-1
  1294	and mAP scores, demonstrating that its architecture is more
  1295	effective at learning robust cross-modality features. Moreover,
  1296	MFENet is notably more parameter-efficient than several highcapacity models like HOS-Net (83.40M) and DEEN (89.00M),
  1297	while offering superior performance. In comparison to the
  1298	growing trend of highly efficient, CLIP-based methods (e.g.,
  1299	CSDN, DSFAD), a crucial trade-off is highlighted. While
  1300	these methods achieve impressively low FLOPs and inferencetime parameters, this apparent efficiency is enabled by a
  1301	substantially larger architecture during the training phase. As
  1302	shown in Table IV, these models rely on over 100M parameters
  1303	
  1304	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:03:00 UTC from IEEE Xplore. Restrictions apply.
  1305	
  1306	GU et al.: DISCOVERING MULTI-FREQUENCY EMBEDDING FOR VISIBLE-INFRARED PERSON RE-IDENTIFICATION
  1307	
  1308	1777

codex
Multi-Frequency 的消融也看完了。它的闭环比较完整：先用频率可视化提出多频问题，再用 HiLo/FADE/损失逐项消融和频率分割参数曲线证明这个问题不是随便编出来的。
  ✓ 定位 6 篇论文的方法区间和核心术语
  → 逐篇提炼观察、问题重定义、机制、证据和 novelty 来源
  • 汇总共同的创新构造套路
  ✓ 定位 6 篇论文的方法区间和核心术语
  ✓ 逐篇提炼观察、问题重定义、机制、证据和 novelty 来源
  → 汇总共同的创新构造套路
codex
我已经整理出逐篇五点。最后我会按你要的“方法论，不抄模块”来写，重点放在作者如何把观察升级成可发表的问题，而不是复述结构图。
  ✓ 定位 6 篇论文的方法区间和核心术语
  ✓ 逐篇提炼观察、问题重定义、机制、证据和 novelty 来源
  ✓ 汇总共同的创新构造套路
codex
我按你实际列出的 6 篇拆了。题头写“9”，但清单里只有 6 个 PDF。

**CycleTrans**
1. 触发观察：作者抓到的是“对齐会伤判别性”。VI-ReID 里大家都在缩小可见光和红外的模态差异，但直接做分布对齐会把不同身份也拉近，颜色、纹理等细粒度线索也被洗掉。
2. 重定义动作：它把旧问题“模态对齐”改成“学习 neutral yet discriminative features”。关键词是 neutral、discriminative、modality-relevant、modality-irrelevant prototypes、cycle construction。核心说法是，跨模态特征不能只中性，还必须能保留身份细节。
3. 机制怎么长出来：机制和重定义绑得比较紧。KCM 先抽“模态相关但有判别性”的语义，DMM 再用共享原型把它变成中性特征，最后用 cycle reconstruction 逼中性特征恢复原模态和异模态语义。逻辑是，如果中性特征还能重构模态相关信息，就说明它没有把身份细节洗掉。
4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
5. reviewer 为什么买账：买的是视角加机制，不只是模块。它把“对齐和判别冲突”讲成一个明确矛盾，再用 cycle 作为保持判别性的检验器。novelty 真正来自“中性特征必须可逆地保留身份语义”这个问题重定义。

**DATE**
1. 触发观察：CLIP-ReID 的 learnable prompt 有判别性，但缺少具体人的描述先验。MLLM 生成的描述有人的属性细节，但文本会有噪声。已有多模态 ReID 把 learnable text 和 description text 同步融合，等价于把可靠和不可靠信息同等对待。
2. 重定义动作：它把问题从“文本能不能帮助 ReID”改成“两类文本嵌入的可靠性和粒度不同，不能对称使用”。关键词是 learnable-based textual embedding、description-based textual embedding、asymmetric fusion、auxiliary clues、cross-granularity training。
3. 机制怎么长出来：因为 description text 有信息但不够可靠，所以它不让描述文本当决定性监督，而是作为辅助线索注入 visual embedding 和 learnable textual embedding。Textual-to-Visual Adapter 用描述去补视觉，Textual-to-Textual Adapter 用描述去补身份 prompt，再用图像级描述和身份级文本做交叉粒度训练。
4. 证据闭环：核心消融是融合方式和文本约束。cross-attention 比直接相加好很多，直接相加甚至会明显伤性能，说明“非对称融合”不是装饰。损失消融里，只有 learnable text、只有 description text、二者组合和完整约束逐步提升，说明描述文本确实是辅助增强而不是替代视觉或替代 prompt。
5. reviewer 为什么买账：买的是“文本空间不等价”的融合视角。真正 novelty 不是 adapter，而是把 MLLM 描述的噪声风险讲清楚，然后给出不让噪声文本主导训练的非对称设计。不过这篇的增益相对克制，更像 ICME 级别的稳健增量，不是特别强的问题重定义范式。

**Deep intelligent technique**
1. 触发观察：作者说监控图像有噪声、视角、光照、背景、姿态错位等问题，传统智能模型特征选择不足，生物启发优化算法容易因为停止迭代而错过更合适的特征。
2. 重定义动作：它没有把 ReID 重新定义成一个更深的问题，而是改写成“在噪声监控图像中做最优特征选择和超参数调节”。关键词是 HLGFS、Horned Lizard optimization、GoogLeNet、feature selection、fitness。
3. 机制怎么长出来：机制主要是把 Horned Lizard 优化和 GoogLeNet 组合。预处理去噪，GoogLeNet 抽特征，Horned Lizard 优化用于选择脸、体型、衣服等特征并调参数。这里逻辑绑定很弱，蜥蜴变色行为和 ReID 视觉机制之间更多是类比，不是从 ReID 失败现象自然推出。
4. 证据闭环：它报了 CUHK03、Market1501、DukeMTMC 上很高的 accuracy 和 mAP，也做了训练数据比例消融，以及 GN+PS、GN+GW、GN+AC、GN+HO 和 GN+HL 的优化器对比。但这些证据主要证明“这个组合分数高”，没有干净证明问题重定义成立，也缺少标准 ReID 协议下的强基线、模块隔离和可解释分析。
5. reviewer 为什么买账：如果买账，买的是应用型混合智能系统和高指标，不是 ReID 方法视角。对我们要做 CCF-B 方法稿来说，这篇更像反例：不要靠“某优化器加某网络”当创新，也不要用很泛的噪声、复杂度、特征选择来替代具体观察。

**DFLSG**
1. 触发观察：无监督 text-to-image ReID 不只是没有身份标签，还没有可靠的图文匹配关系。图像聚类能给伪标签，但文本不一定和图像匹配，MLLM 生成文本也会有幻觉和冗余属性。
2. 重定义动作：它把旧问题“无监督跨模态学习”改成“训练样本和伪标签的 dependability 问题”。关键词是 dependable text generation、reliable image-text matching relationship、Error Sample Filtering、dependable identity labels。
3. 机制怎么长出来：既然核心是可靠性，它先用多个 MLLM 生成候选描述，再用已有 TIReID 模型的图文相似度选择更可靠文本。然后只对图像聚类，把伪标签传给匹配文本，再用文本和图像类中心的相似度做 IQR 异常过滤。最后 MTJL 从类中心和实例两个层面收紧跨模态特征。
4. 证据闭环：消融是比较完整的。单 MLLM 生成文本加聚类最差，加入 MTJL 有明显提升，换 MLLM 会波动，说明文本质量不稳定。dependable text generation 明显提升，ESFM 再提升。IQR 和 Z-score、LOF 对比，聚类方法替换，过滤强度参数分析，都在证明“可靠样本构造”是承重点。
5. reviewer 为什么买账：买的是问题分解。它没有简单说“大模型生成文本能用”，而是说无监督 TIReID 的难点在两层可靠性：图文关系可靠、伪身份可靠。novelty 来源是数据构造和可靠性过滤的闭环，机制和重定义绑定较紧。

**DIRL**
1. 触发观察：图像级 VI-ReID 方法处理不了视频里的姿态、运动、遮挡和时序变化。已有 VVI-ReID 多在帧级做模态不变，序列级特征聚合后仍有模态差异，而且 modal-related 信息会继续干扰身份判别。
2. 重定义动作：它把 VVI-ReID 从“时序聚合加跨模态对齐”改成“序列级模态交互之后，显式学习 ID-related representation”。关键词是 sequence-level modal-invariant features、modality information interaction、feature decoupling、ID-related 和 modal-related。
3. 机制怎么长出来：先用 TBM 抽帧级特征，第一层分模态，后面共享。CIM 在聚合时序信息时引入跨模态交互，用共享 key 让可见光和红外在序列级产生共同参照。FDM 再学习模态相关分量并从序列特征中减掉，只留下 ID-related 表示。损失里还用第三个“neither modality”类别逼 ID 特征不能被模态分类。
4. 证据闭环：HITSZ-VCM 消融显示 CIM 单独带来 Rank-1 和 mAP 大提升，FDM 单独也提升，两者组合最好。t-SNE 显示同身份跨模态更聚、不同身份更分开，Rank-5 可视化展示视频检索结果。参数分析说明帧数增加和 FDM 维度变化会影响性能。
5. reviewer 为什么买账：买的是“序列级”这个落点。feature decoupling 本身不是新概念，但它把解耦从图像帧级挪到视频序列级，并放在跨模态交互之后，形成了和 VVI-ReID 场景绑定的说法。novelty 是问题位置的新，不是基础算子的新。

**Discovering Multi-Frequency Embedding**
1. 触发观察：已有 VI-ReID 多在空间域做特征，频域方法也常常只做全频段或低高频二分。作者用频率可视化说明，低频更容易含模态风格和照明，高频更保留边缘结构，但身份线索并不只在高频，而是分散在多个频带。
2. 重定义动作：它把问题从“频域能减少模态差异”改成“身份是 multi-frequency 的，单一频率切分会在保留身份线索和去除模态噪声之间做坏取舍”。关键词是 multi-frequency embedding、frequency-aware diversity、band-specific importance、cross-modality retrieval。
3. 机制怎么长出来：HiLo-FM 对低频和高频分工处理，低频在频域用可学习复权重做风格调节，高频回到空间域用轻量多尺度卷积抽结构细节，再用空间注意力融合。FADE 把频谱拆成多个频带，对每个频带学注意力，多分支产生多样特征。CMSR 把训练目标改成更贴近跨模态检索，CMRR 用排序不一致约束鼓励分支多样性。
4. 证据闭环：这篇闭环很完整。Fig. 2 先用可视化证明多频观察。组件消融显示 HiLo-FM、FADE、CMRR、CMSR 都有增益，完整组合最好。ρ 曲线证明低频不能全丢也不能过多依赖，0.2 附近最好。频带数 B 的曲线证明多频拆分优于 DC/others 或简单低高频，但过多频带会退化。方形、圆形、菱形分割对比进一步证明不是随便切。
5. reviewer 为什么买账：买的是视角加证据。FFT、IN、多尺度卷积都不是新零件，真正 novelty 是“多频身份线索分布”这个观察，以及用参数曲线证明单一频率划分确实不够。机制和重定义绑定很紧，是这批里最值得学的创新构造方式之一。

这批论文共同套路是：先不急着加模块，而是把旧范式的副作用单独命名，比如对齐损判别性、文本有噪声、无监督样本不可靠、序列级仍残留模态信息、单频切分会丢身份线索。然后造一个中间概念承接新问题，比如 neutral feature、dependable pair、ID-related sequence representation、multi-frequency embedding，再让模块围绕这个中间概念服务。真正能让 reviewer 买账的不是模块复杂，而是消融和可视化能证明“这个新问题定义本身是对的”。
tokens used
197,011
我按你实际列出的 6 篇拆了。题头写“9”，但清单里只有 6 个 PDF。

**CycleTrans**
1. 触发观察：作者抓到的是“对齐会伤判别性”。VI-ReID 里大家都在缩小可见光和红外的模态差异，但直接做分布对齐会把不同身份也拉近，颜色、纹理等细粒度线索也被洗掉。
2. 重定义动作：它把旧问题“模态对齐”改成“学习 neutral yet discriminative features”。关键词是 neutral、discriminative、modality-relevant、modality-irrelevant prototypes、cycle construction。核心说法是，跨模态特征不能只中性，还必须能保留身份细节。
3. 机制怎么长出来：机制和重定义绑得比较紧。KCM 先抽“模态相关但有判别性”的语义，DMM 再用共享原型把它变成中性特征，最后用 cycle reconstruction 逼中性特征恢复原模态和异模态语义。逻辑是，如果中性特征还能重构模态相关信息，就说明它没有把身份细节洗掉。
4. 证据闭环：关键不是只报 SOTA，而是消融显示 KCM、DMM、cycle 都有贡献。文中说 KCM 和 DMM 分别带来 Rank-1 提升，cycle 相比 DMM 后又明显提升，完整方法比基线高很多。DMM 替代实验说明“直接用原型”“不用 discrepancy”“Transformer 替代”都不如 DMM。t-SNE 证明无 cycle 时中性但边界不清，Grad-CAM 证明 cycle 让模型看更完整的人体细节。
5. reviewer 为什么买账：买的是视角加机制，不只是模块。它把“对齐和判别冲突”讲成一个明确矛盾，再用 cycle 作为保持判别性的检验器。novelty 真正来自“中性特征必须可逆地保留身份语义”这个问题重定义。

**DATE**
1. 触发观察：CLIP-ReID 的 learnable prompt 有判别性，但缺少具体人的描述先验。MLLM 生成的描述有人的属性细节，但文本会有噪声。已有多模态 ReID 把 learnable text 和 description text 同步融合，等价于把可靠和不可靠信息同等对待。
2. 重定义动作：它把问题从“文本能不能帮助 ReID”改成“两类文本嵌入的可靠性和粒度不同，不能对称使用”。关键词是 learnable-based textual embedding、description-based textual embedding、asymmetric fusion、auxiliary clues、cross-granularity training。
3. 机制怎么长出来：因为 description text 有信息但不够可靠，所以它不让描述文本当决定性监督，而是作为辅助线索注入 visual embedding 和 learnable textual embedding。Textual-to-Visual Adapter 用描述去补视觉，Textual-to-Textual Adapter 用描述去补身份 prompt，再用图像级描述和身份级文本做交叉粒度训练。
4. 证据闭环：核心消融是融合方式和文本约束。cross-attention 比直接相加好很多，直接相加甚至会明显伤性能，说明“非对称融合”不是装饰。损失消融里，只有 learnable text、只有 description text、二者组合和完整约束逐步提升，说明描述文本确实是辅助增强而不是替代视觉或替代 prompt。
5. reviewer 为什么买账：买的是“文本空间不等价”的融合视角。真正 novelty 不是 adapter，而是把 MLLM 描述的噪声风险讲清楚，然后给出不让噪声文本主导训练的非对称设计。不过这篇的增益相对克制，更像 ICME 级别的稳健增量，不是特别强的问题重定义范式。

**Deep intelligent technique**
1. 触发观察：作者说监控图像有噪声、视角、光照、背景、姿态错位等问题，传统智能模型特征选择不足，生物启发优化算法容易因为停止迭代而错过更合适的特征。
2. 重定义动作：它没有把 ReID 重新定义成一个更深的问题，而是改写成“在噪声监控图像中做最优特征选择和超参数调节”。关键词是 HLGFS、Horned Lizard optimization、GoogLeNet、feature selection、fitness。
3. 机制怎么长出来：机制主要是把 Horned Lizard 优化和 GoogLeNet 组合。预处理去噪，GoogLeNet 抽特征，Horned Lizard 优化用于选择脸、体型、衣服等特征并调参数。这里逻辑绑定很弱，蜥蜴变色行为和 ReID 视觉机制之间更多是类比，不是从 ReID 失败现象自然推出。
4. 证据闭环：它报了 CUHK03、Market1501、DukeMTMC 上很高的 accuracy 和 mAP，也做了训练数据比例消融，以及 GN+PS、GN+GW、GN+AC、GN+HO 和 GN+HL 的优化器对比。但这些证据主要证明“这个组合分数高”，没有干净证明问题重定义成立，也缺少标准 ReID 协议下的强基线、模块隔离和可解释分析。
5. reviewer 为什么买账：如果买账，买的是应用型混合智能系统和高指标，不是 ReID 方法视角。对我们要做 CCF-B 方法稿来说，这篇更像反例：不要靠“某优化器加某网络”当创新，也不要用很泛的噪声、复杂度、特征选择来替代具体观察。

**DFLSG**
1. 触发观察：无监督 text-to-image ReID 不只是没有身份标签，还没有可靠的图文匹配关系。图像聚类能给伪标签，但文本不一定和图像匹配，MLLM 生成文本也会有幻觉和冗余属性。
2. 重定义动作：它把旧问题“无监督跨模态学习”改成“训练样本和伪标签的 dependability 问题”。关键词是 dependable text generation、reliable image-text matching relationship、Error Sample Filtering、dependable identity labels。
3. 机制怎么长出来：既然核心是可靠性，它先用多个 MLLM 生成候选描述，再用已有 TIReID 模型的图文相似度选择更可靠文本。然后只对图像聚类，把伪标签传给匹配文本，再用文本和图像类中心的相似度做 IQR 异常过滤。最后 MTJL 从类中心和实例两个层面收紧跨模态特征。
4. 证据闭环：消融是比较完整的。单 MLLM 生成文本加聚类最差，加入 MTJL 有明显提升，换 MLLM 会波动，说明文本质量不稳定。dependable text generation 明显提升，ESFM 再提升。IQR 和 Z-score、LOF 对比，聚类方法替换，过滤强度参数分析，都在证明“可靠样本构造”是承重点。
5. reviewer 为什么买账：买的是问题分解。它没有简单说“大模型生成文本能用”，而是说无监督 TIReID 的难点在两层可靠性：图文关系可靠、伪身份可靠。novelty 来源是数据构造和可靠性过滤的闭环，机制和重定义绑定较紧。

**DIRL**
1. 触发观察：图像级 VI-ReID 方法处理不了视频里的姿态、运动、遮挡和时序变化。已有 VVI-ReID 多在帧级做模态不变，序列级特征聚合后仍有模态差异，而且 modal-related 信息会继续干扰身份判别。
2. 重定义动作：它把 VVI-ReID 从“时序聚合加跨模态对齐”改成“序列级模态交互之后，显式学习 ID-related representation”。关键词是 sequence-level modal-invariant features、modality information interaction、feature decoupling、ID-related 和 modal-related。
3. 机制怎么长出来：先用 TBM 抽帧级特征，第一层分模态，后面共享。CIM 在聚合时序信息时引入跨模态交互，用共享 key 让可见光和红外在序列级产生共同参照。FDM 再学习模态相关分量并从序列特征中减掉，只留下 ID-related 表示。损失里还用第三个“neither modality”类别逼 ID 特征不能被模态分类。
4. 证据闭环：HITSZ-VCM 消融显示 CIM 单独带来 Rank-1 和 mAP 大提升，FDM 单独也提升，两者组合最好。t-SNE 显示同身份跨模态更聚、不同身份更分开，Rank-5 可视化展示视频检索结果。参数分析说明帧数增加和 FDM 维度变化会影响性能。
5. reviewer 为什么买账：买的是“序列级”这个落点。feature decoupling 本身不是新概念，但它把解耦从图像帧级挪到视频序列级，并放在跨模态交互之后，形成了和 VVI-ReID 场景绑定的说法。novelty 是问题位置的新，不是基础算子的新。

**Discovering Multi-Frequency Embedding**
1. 触发观察：已有 VI-ReID 多在空间域做特征，频域方法也常常只做全频段或低高频二分。作者用频率可视化说明，低频更容易含模态风格和照明，高频更保留边缘结构，但身份线索并不只在高频，而是分散在多个频带。
2. 重定义动作：它把问题从“频域能减少模态差异”改成“身份是 multi-frequency 的，单一频率切分会在保留身份线索和去除模态噪声之间做坏取舍”。关键词是 multi-frequency embedding、frequency-aware diversity、band-specific importance、cross-modality retrieval。
3. 机制怎么长出来：HiLo-FM 对低频和高频分工处理，低频在频域用可学习复权重做风格调节，高频回到空间域用轻量多尺度卷积抽结构细节，再用空间注意力融合。FADE 把频谱拆成多个频带，对每个频带学注意力，多分支产生多样特征。CMSR 把训练目标改成更贴近跨模态检索，CMRR 用排序不一致约束鼓励分支多样性。
4. 证据闭环：这篇闭环很完整。Fig. 2 先用可视化证明多频观察。组件消融显示 HiLo-FM、FADE、CMRR、CMSR 都有增益，完整组合最好。ρ 曲线证明低频不能全丢也不能过多依赖，0.2 附近最好。频带数 B 的曲线证明多频拆分优于 DC/others 或简单低高频，但过多频带会退化。方形、圆形、菱形分割对比进一步证明不是随便切。
5. reviewer 为什么买账：买的是视角加证据。FFT、IN、多尺度卷积都不是新零件，真正 novelty 是“多频身份线索分布”这个观察，以及用参数曲线证明单一频率划分确实不够。机制和重定义绑定很紧，是这批里最值得学的创新构造方式之一。

这批论文共同套路是：先不急着加模块，而是把旧范式的副作用单独命名，比如对齐损判别性、文本有噪声、无监督样本不可靠、序列级仍残留模态信息、单频切分会丢身份线索。然后造一个中间概念承接新问题，比如 neutral feature、dependable pair、ID-related sequence representation、multi-frequency embedding，再让模块围绕这个中间概念服务。真正能让 reviewer 买账的不是模块复杂，而是消融和可视化能证明“这个新问题定义本身是对的”。
