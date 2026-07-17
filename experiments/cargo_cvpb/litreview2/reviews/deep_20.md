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
session id: 019ef835-dfc1-7b11-83f9-985c35cd3741
--------
user
你是 ReID 创新方法论拆解员 20。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(20)==
Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf
Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf
Multi-year long-term person re-identification using gait and HAR features.pdf
Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf
Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf
Occluded person Re-Identification with noise injection.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则读必要的目标和状态文件，然后用 `pdftotext -f 1 -l 8` 抽每篇的方法部分。因为当前任务是 PDF 论文拆解，我会按 PDF 技能的读取和核验思路处理，但具体读取命令遵循你给的 `pdftotext`。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pwd && rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'AGENTS.md' -g 'STATUS.md' -g 'SKILL.md' /Users/abdslm/Desktop/reid-paper /Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.623.12021/skills/pdf 2>/dev/null" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
/Users/abdslm/Desktop/reid-paper/CLAUDE.md
/Users/abdslm/Desktop/reid-paper/AGENTS.md
/Users/abdslm/Desktop/reid-paper/GOAL.md
/Users/abdslm/Desktop/reid-paper/experiments/STATUS.md
/Users/abdslm/.codex/plugins/cache/openai-primary-runtime/pdf/26.623.12021/skills/pdf/SKILL.md

hook: PreToolUse
hook: PreToolUse
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

codex
目录里列出的目标论文是 6 篇，不是 20 篇；我会按你列出的 6 个文件逐篇拆。现在先把项目规则和 PDF 处理规则读完，避免后面引用口径跑偏。
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
/bin/zsh -lc "sed -n '1,260p' /Users/abdslm/Desktop/reid-paper/experiments/STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

**08:25 滚动进展**：
- **DONOR**：Occ-Duke ImageNet-Swin 两种子稳 +1.5 mAP(45.9/45.7 vs 纯基线 44.3)，选项1 第一个正信号。归因(aug-only 三种子)~1.5h 出，定 +1.5 是 paste-aug 增广还是真解耦机制(exp009 在 SOLIDER 上是增广)。seed2 在跑。
- **ViT DONOR scale**：codex 实现的 ViT patch 空间图改动(让 DONOR 能在 ViT 跑)+自检(纯 ViT 基线 CLS diff=0)+两轮独立 codex 审查都放行。配置就绪，等 GPU1 空(归因时)smoke。**注意用不覆盖 PRETRAIN_PATH 的 wrapper**。
- **想法2(多点证据一致性，exp013)**：零训练探针两数据集都证实"遮挡 top-1 假阳性=单点局部巧合"(same_cam 假阳性 Occ-Duke 91.8%/PoseTrack 93.9% 单点，真阳性多 0.8~0.9 独立证据，bootstrap 显著)。design 就绪(纯训练侧、区别 MULTIHYP/TARDIS、detach 防污染、预注册判负线)，codex 实现+退化自检中。是 DONOR 死后的主线接棒 or 第二贡献候选。
- MULTIHYP 在常规 ImageNet-Swin 两数据集都判负(决策#34)。
- 四卡满载+三处链式接续，全程用 codex(不用 Workflow/子 agent)。

---

**方向（用户决策#33=选项1）**：放开 SOLIDER 强主干，回常规 ImageNet 预训练主干(TransReID ViT-base、ImageNet 预训练 Swin-Tiny)刷标准遮挡 ReID mAP/rank1。可靠性/conformal 线归档(决策#28-#32 的负结果+诊断作分析素材)。

**进展**：
- ViT-base 基线 Occ-Duke seed0 = 53.7/61.0，正中文献护栏，环境+ViT 适配可信。
- **MULTIHYP 在常规 ImageNet-Swin 两数据集都判负(决策#34)**：Occ-Duke within-ckpt α0→0.3 +0.2、Occ-PoseTrack +0.2，都噪声内、都没过 +0.5。即"换较弱常规主干救回 SOLIDER 上塌掉的增益"对 MULTIHYP 没兑现。MULTIHYP 越来越像"特定弱主干+特定数据集"的偶然。
- **DONOR 待生死**：Occ-Duke ImageNet-Swin seed0 末 epoch 45.9，等 hyy GPU1 纯基线 seed0 对照(~45min)。纯基线≈44.x→DONOR 有 +1.x 信号(再查是否又来自 paste-aug)，≈45.9→按 exp009 同款判负。

**四线满(机器不空转)**：hyy GPU1=纯基线 Occ-Duke seed0(DONOR 对照,~45min)；hyy GPU0=纯基线 Occ-Duke seed1(复用控制)；lab-3090-d=纯基线 Occ-PoseTrack seed0(复用控制)；lab-4090=DONOR Occ-Duke seed1(afr，并行抢跑)。纯基线是所有方法共用控制，正负都不浪费。

**并行**：codex 头脑风暴下一个真正新的遮挡机制(bmjb1csj8，省 token；DONOR 也是 SOLIDER 判负回收的去偏模块，若它也塌则"回收旧模块到常规主干"赌注基本证伪，需要新机制)，写到 exp012_baseline_pivot/brainstorm-codex.md。

codex
状态文件里当前阶段仍是读文献、做差距分析、提炼候选方向，和这次任务一致。接下来我先定位每篇论文的方法章节边界，再把方法段完整抽出来读，避免只抓摘要或贡献列表。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "for f in 'Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf' 'Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf' 'Multi-year long-term person re-identification using gait and HAR features.pdf'; do printf '\\n===== %s =====\\n' \""'$f"; pdftotext -f 1 -l 8 "$f" - | nl -ba | rg -i '"'("'^'"\\s*[0-9]+\\s+(abstract|introduction|related work|method|methodology|approach|proposed|framework|experiments|experiment|evaluation|conclusion|results)|\\bmethod\\b|\\bapproach\\b|\\bframework\\b|\\bmodule\\b|\\bloss\\b)' | head -n 80; done" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "for f in 'Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf' 'Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf' 'Occluded person Re-Identification with noise injection.pdf'; do printf '\\n===== %s =====\\n' \""'$f"; pdftotext -f 1 -l 8 "$f" - | nl -ba | rg -i '"'("'^'"\\s*[0-9]+\\s+(abstract|introduction|related work|method|methodology|approach|proposed|framework|experiments|experiment|evaluation|conclusion|results)|\\bmethod\\b|\\bapproach\\b|\\bframework\\b|\\bmodule\\b|\\bloss\\b)' | head -n 80; done" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 183ms:
    10	Abstract—Unsupervised person re-identification aims to
    12	method of clustering and assigning pseudo-labels has become
    19	paper proposes an unsupervised person re-identification method
    27	improve clustering accuracy, an inter-class balance loss (CB loss)
    30	design a Feature Reconstruction (ODE-FR) module with residual
    32	experimental results on Market-1501, DukeMTMC-reID, and
    33	MSMT17 demonstrate that our proposed method is superior to
    76	proposed method. First, insufficient feature representation
    92	contains multiple positive instances. The SpCL [9] method
    95	contrast learning method [10] updates the memory bank and
    96	computes the contrastive loss at the cluster level, making the
    97	contrastive loss the most suitable metric learning algorithm for
   128	person Re-ID method based on nearest neighbor sample
   138	we also design an inter-class balance loss (CB loss), which
   149	methods of mining encoder features at various depths, such
   157	module. Meanwhile, we insert a simple PReLU layer [16] and
   169	• We design the inter-class balance loss (CB loss) better to
   173	(ODE-FR) module, which leverages the implicit thirdorder Adam solution of an ordinary differential equation
   181	The clustering algorithm is the most common method for
   195	Zhang et al. [20] proposed a time-integrated approach to
   200	the mainstream method of generating pseudo-labels by clustering. In addition, some methods [22], [23], [24], [25], [26],
   204	re-identification framework, optimizing pseudo-labels through
   205	a spatial diffusion model and semantic consistency loss, thus
   224	address this issue, Pang et al. proposed the CIFL framework
   225	[33] and the MSC-GDC method [34], both of which incorporate camera label information: CIFL employs an ensemble
   229	results. In contrast to these camera-aware approaches, this
   236	a progressive learning method integrating self-step learning
   248	learning method has become a vital helper.
   250	introduces a memory bank and proposes a new selfscheduling contrast learning framework with mixed memory.
   251	Subsequently, Sun et al. [37] optimized the contrast loss
   252	and proposed a contrast learning method based on multigranularity cluster integration to mine the priority information
   254	a cluster-guided asymmetric contrast learning method that
   255	successfully combines classification loss and contrast loss by
   257	designed asymmetric contrast learning framework. In addition,
   260	loss at the cluster level. Moreover, the real-time memory
   267	modality contrastive loss, which establishes cross-modal constraints at the cluster centroid level to promote the generation
   270	semantic information fusion framework that leverages nonparametric graph convolution to estimate pair-wise similarities
   271	and employs relaxed contrastive loss to integrate hard pseudolabels, pair-wise semantics, and neighborhood semantics as
   280	classes. Coupled with the class balance loss, which is theoretically grounded in concave function optimization via Jensen’s
   281	inequality, our approach dynamically repels confusing samples
   292	[43] designed a PolyInception module for each residual block
   304	multistep solution method of ordinary differential equations,
   306	Leapfrog et al. [46] in numerical ordinary differential equations, He et al. [45] designed a network module for image
   322	of hard samples between classes; CB loss can better identify hard samples between the nearest neighbor classes and improve clustering accuracy; ODE-FR
   331	features, performs pseudo-label refinement, and optimizes features through global and local mutual refinement loss. Global
   333	smoothing loss Laals , the part-guided label refinement loss
   334	L pglr , and the combination of cross-entropy loss Lce and triplet
   335	loss Ltri refines the pseudo-labels. The total PPLR loss is as
   349	loss, the nearest neighbor sample constraint strategy finds
   353	balance loss, the classification criteria of the nearest cluster
   356	the ODE guided feature reconstruction module is inserted into
   367	loss every time when mini-batch samples are trained. To
   371	between samples of different classes by contrast loss Lq :
   402	contrast loss is defined as:
   414	hyperparameter. Unlike the contrast loss Lq , the negative
   433	in addition to the clustering centroids, the memory module
   438	from traditional sampling strategies (such as the triplet loss)
   441	method is based on querying the negative samples on all
   443	can be ensured under the constraint of class centroid loss,
   447	loss of the nearest neighbor sample mining strategy is defined
   481	centroid contrast loss Lcen and the nearest neighbor sample
   482	contrast loss Lnea together constitute the global contrast loss:
   489	CB loss; (b) with CB loss. The red circles highlight hard samples that
   491	loss in (b), these hard samples are better separated and classified more
   492	clearly, demonstrating the effectiveness of CB loss in improving feature
   495	Under the guidance of global contrast loss, the clustering
   504	C. Inter-Class Balance Loss
   508	and propose the balance loss between classes, which is defined
   531	fixed constant that prevents a loss explosion which is set as
   532	1e − 8. In the inter-class balance loss, consider the function:
   559	to minimize the loss. Therefore, based on the hard sample
   560	constraint strategy, the inter-class balance loss is designed to
   573	Thus, the total loss of our proposed algorithm is obtained
   579	Discussion: The design of the Inter-Class Balance loss
   599	their approach of designing various ODE-blocks as the main
   601	Runge-Kutta methods, our ODE-FR module is designed based
   611	form a richer feature representation. This approach integrates
   614	a hierarchical feature representation. This method emphasizes
   618	design the ODE-FR module (Fig. 5 (c)) based on the thirdorder implicit Adam solution formula of ordinary differential
   647	Adams method. Therefore, with step size h = 1, the third-order

===== Occluded person Re-Identification with noise injection.pdf =====
    30	ABSTRACT
    43	method achieves state-of-the-art performance on two occluded person ReID benchmarks: Occluded-Duke and
    44	P-DukeMTMC. Additionally, we validate the approach on four standard person ReID benchmarks, Market1501, DukeMTMC-ReID, CUHK03-NP, and MSMT17, further demonstrating its effectiveness. The code is at
    54	crowds obscure one another. Both scenarios lead to the loss of critical
   109	Our approach consists of two key components: (1) Epoch-wise
   151	task complexity under certain conditions. This framework demonstrates
   171	occluded ReID settings. In contrast, our method introduces structured
   174	Our approach extends these insights by treating vehicle images as
   195	method employs IoU-based filtering to ensure occlusion realism while
   211	architectures. In contrast, our method improves occluded ReID from a
   215	3. Proposed approach
   216	As illustrated in Fig. 3, our proposed framework addresses the
   228	phases and mixed person-vehicle phases. This approach serves a dual
   231	The extracted features from both branches are fused into a comprehensive representation that captures both holistic appearance and discriminative local patterns. This dual-representation approach ensures
   249	Fig. 3. Overview of the proposed NIReID. The framework utilizes vehicle images as positive noise to improve generalization in occluded person ReID by adopting
   260	Output 𝑂, Loss Function 𝐿𝐹
   261	Result: Loss Values 𝐿𝑖 , 𝑖 ∈ [1, 𝑁]
   382	resistance to adversarial examples. However, this method requires that
   393	operations. After the shared branch-specific embedding module, the
   403	branches, leveraging the complementary nature of global holistic features and local part-based features. This dual-representation approach
   408	Comparison with Hide-and-Seek. The Hide-and-Seek method [26]
   410	one is hidden. Our approach, INI, replaces these hidden patches with
   413	Comparison with GridMask. GridMask [27] is a variant of Hide-andSeek, and it similarly removes regions, thereby disrupting semantic information. In contrast, our INI approach better preserves this semantic
   419	integrated into any training framework.
   426	To optimize the network, we employ a multi-loss training strategy.
   427	Cross-entropy loss supervises identity classification for both global and
   428	local features, ensuring both branches contribute to identity discrimination. In contrast, the triplet loss with hard negative mining is applied
   431	total loss function is formulated as:
   442	methods, similar to RandAugment, function primarily as data augmentation strategies. Besides, our strategies are well-designed for the
   453	where 𝜆 balances the contribution of the triplet loss.
   458	The proposed noise-injection re-identification framework can be
   501	Results on Occluded-Duke and P-DukeMTMC datasets. Best results in BOLD.
   502	Method
   744	Results on Occluded-Duke and P-DukeMTMC Datasets with additional parsing
   746	Method
   753	proposed positive noise injection can be interpreted as an instance
   837	and P-DukeMTMC [1]. First, we compare our method against existing approaches using different backbone architectures to demonstrate
   841	annotations, to highlight the practical advantages of our approach.
   842	Comparison with Different Backbone Architectures. Table 3 compares our method against existing approaches using different backbone architectures. Despite utilizing CNN-based backbones, our NIReID
   843	method achieves superior performance compared to more parameterheavy ViT-based approaches. Specifically, our NIReID+R-1.6G achieves
   845	the best ViT-based method ADP [40] by 2.0% and 1.6%, respectively.
   848	same backbone architecture, our approach demonstrates substantial
   852	our approach under fair comparison conditions.
   856	To validate the efficacy of our approach, rigorous assessments were
   863	All experiments are conducted with the PyTorch framework on
   874	compares our method with approaches that rely on additional supervision, such as human parsing networks or manual annotations. Notably,
   877	dataset, our method outperforms most parsing-enhanced techniques,
   878	demonstrating that the proposed vehicle noise injection strategy effectively eliminates the need for costly manual annotations. This comparison underscores the practical advantage of our approach: it attains
   884	To comprehensively evaluate the effectiveness of our approach, we
   893	Results on Market-1501 and DukeMTMC-ReID datasets.
   894	Method
  1059	Results on CUHK03-NP dataset.
  1060	Method
  1202	validate that our vehicle noise injection approach enhances model
  1206	Results on MSMT17 dataset.
  1207	Method
  1278	results.
  1285	To comprehensively evaluate the effectiveness of our proposed approach, we conduct ablation studies on the Occluded-Duke dataset [3]
  1289	To validate the generalizability of our approach beyond occluded
  1293	Our method demonstrates substantial improvements over existing
  1294	approaches, as shown in Table 5. On Market-1501, NIReID+R-1.6G
  1295	surpasses the previous best method FMGNet [46] by 0.5% Rank-1
  1301	best method, APD [47] by 2.0% Rank-1. In the labeled setting, we
  1305	NIReID+R-1.6G surpasses the previous best method, TransReID [8] by
  1327	Method
  1740	Method

 succeeded in 252ms:
    11	Abstract—Open-world person re-identification aims to train
    14	methods primarily employ the equality training paradigm to
    19	Perception (MSP) framework equipped with an Asynchronous
    27	Perception (SSP) module that constructs the feature relationship
    36	experiments demonstrate that our method outperforms state-ofthe-arts for open-domain person ReID.
    76	methods [7], [8], [9] assume a homogeneous environment
   118	In this paper, we propose a Multi-model Synergy Perception (MSP) framework to construct balanced domains for
   138	propose a Structure Synergy Perception (SSP) module that
   160	learning. Furthermore, existing ReID loss functions primarily
   177	model, we propose an adaptive-weight method to aggregate
   179	augmented randomly and we employ contrastive loss function
   187	framework equipped with an Asynchronous Training
   193	we construct a Structure Synergy Perception (SSP) module based on the feature relationship. SSP constructs the
   217	methods [20], [24] and (3) attention based methods [21], [25].
   218	Hermans et al. [23] propose a hard sample based triplet loss
   268	methods to DG ReID.
   277	bias. However, this method can eliminate some important
   279	a Style Normalization and Restitution (SNR) module to filter
   290	a novel Meta Distribution Alignment (MDA) method [55]
   295	a multi-scale query-adaptive convolution framework to encode
   305	method to leverage multi-model to learn a system composed of
   311	Embedding via adapTive Aggregation (META) method, which
   320	Fig. 2. Structure of our proposed method. Source 1 and Source 2 represent small- and large-scale datasets respectively. Our method divides the biased source
   335	propose a multi-model synergy perception framework that
   337	imbalance issue and develops a synergy learning approach
   343	framework of our method is shown in Fig. 2. To relieve
   351	module, which aligns the feature relationship distribution
   390	(MSP) framework equipped with an Asynchronous Training
   399	Under-sampling is an efficient method for data-imbalance
   402	such under-sampling leads to the loss of valuable information
   482	perception (SSP) module for synergy learning among submodels. The feature structure distribution of each subdoamin is first
   529	develop a synergy perception loss function LS S P to guide the
   557	To further improve the model generalization ability, we propose a Synergy Distillation Perception(SDP) module to exploit
   576	a central distillation loss LDis based on ATP, which extracts
   581	models. Next, we leverage the proposed central distillation loss
   609	the proposed central distillation loss LDis , represented as:
   625	fixed copy of the parameter θ in our method. Following [67],
   634	module that leverages this valuable prior knowledge to explore
   637	domain noise filter loss L f to construct the parts graph of each
   639	ability, we develop a parts distillation loss L p to align the
   685	Based on the guidance of this loss, the sub-model is forced to
   712	of the same ID, we develop a domain noise filter loss L f to
   726	by circumventing negative values in loss. i and j represent
   749	cross-model parts distillation loss L p and construct the intraand inter-submodel parts graph for synergy distillation. The
   750	parts distillation loss L p is defined as:
   758	The triplet loss of Ltri employs a widely-used batch-hard
   759	triplet loss [23]. Considering that the number of conventional
   837	In this work, we use both identification loss and triplet loss.
   838	The total loss of the proposed method is calculated as:
   849	1) Datasets: To evaluate our method, we conduct extensive
   850	experiments on four large-scale datasets including Market1501 [69], CUHK03 [70], CUHK-SYSU [71] and MSMT17
   908	identity and we set the margin in the triplet loss to 0.3 in our
   909	experiment. For the memory identity loss function, we set the
   919	1) Compared Methods Under the Protocol-2: To demonstrate the superiority of our method, we compare with a
   924	methods on our experiments setting by running their released

===== Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf =====
    17	DFF module dynamically extracts local and long-range dependencies among features to obtain finer-grained
    18	discriminative features. The DPE module extracts multi-scale features from both visible and infrared modalities
    19	to generate diverse embeddings. The FRS module mitigates the impact of information imbalance between
    40	Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
    59	Introduction
    80	methods, the fusion of multi-modal data requires finding a suitable method to bridge the gap
    87	(MDFN), which comprises three modules: dynamic feature fusion (DFF), dynamic perceptual enhancement (DPE), and feature reweighting with similarity (FRS). The DFF module
    90	The DPE module employs deformable dilated convolution with various dilation rates to capture
    92	hierarchical contrast (MHC) loss. Additionally, to address inaccuracies in the query library affinity matrix, we introduce the FRS module, which optimizes matching performance by leveraging
   109	MDFN framework to improve feature extraction and enhancement, as well as modality
   111	— Our MDFN framework integrates the DFF module to extract fine-grained features by fusing
   112	the extracted local and long-range dependencies among features, the DPE module to enhance
   113	multi-scale spatial dependencies and feature representation, and the FRS module to generate
   115	query and gallery sets. Therefore, our MDFN framework is capable of tackling complex
   122	Related Work
   131	shape representation as a pixel-level guide. This approach incorporates body shape knowledge in
   133	computational cost. EIE [24] proposes an enhancement, integration and extension approach aimed
   135	representation of the model are improved by means of a self-enhancement module, a cointegration module, and a matching extension module. SecureReID [50] proposes a federated learning
   136	reversible anonymization framework that generates full-body anonymized images with little or no
   146	but seamlessly recoverable without loss via the corresponding keys. The DKP [44] captures the
   150	further designs a prototype-based knowledge transfer module to enhance the knowledge-forgetting
   159	and feature-level methods. Image-level methods typically involve generating intermediate modalities to bridge the gap between the two modalities. The feature-level methods approach maps the
   161	the method employed in this paper is a feature-level approach.
   164	enhances the accuracy of cross-modal matching. For example, AlignGAN [34] is the first method to
   167	Another notable approach is D2 RL [36], which uniquely separates appearance differences in traditional re-identification tasks from modality differences introduced by spectral imaging processes.
   179	achieves strong results, our approach further refines this alignment through a dynamic, hierarchical
   181	synergy module to fuse VIS and IR features, enhancing identity discrimination while reducing
   182	noise; however, our method introduces an adaptive, multi-scale fusion mechanism that allows
   186	incorporating historical information during training. SPOT [1], which incorporates a structureaware module, uses structural information to guide modal feature extraction and dynamically selects
   188	our approach enables more flexible, adaptive decision-making, dynamically refining features to
   190	extension module that generates additional embeddings constrained by center-guided pair mining
   197	loss, helping the network capture rich representations and reduce modality differences. Inspired
   201	Compared with existing methods, our approach shares key principles such as modal fusion and
   207	Methodology
   213	In this section, we provide a detailed description of the proposed MDFN framework, as illustrated
   214	in Figure 2. The framework processes two modalities of input images: VIS for the VIS branch and
   215	IR for the IR branch. The MDFN comprises four main components: the backbone, the DFF module,
   216	the dynamic perception enhancement (DPE) module, and the FRS module. In this paper, we
   219	extraction. These extracted VIS-IR feature pairs are then input into the DFF module, where local
   221	features are subsequently fed into the DPE module, which captures contextual information at
   222	various scales and enriches the feature representation. Additionally, we design an MHC loss to
   224	the DPE module. Furthermore, we employ the orthogonal Gramian matrix loss (OGM loss) to
   229	DFF Module
   235	ability to learn representations effectively. To address this limitation, we propose a DFF module,
   236	which includes both an attention branch and a dynamic convolutional branch. The DFF module
   237	introduces a multi-scale fusion approach that combines both local and global feature extraction to
   240	contextual information across larger scales [31]. This multi-scale fusion enables the DFF module
   243	global features, the DFF module strengthens cross-modal alignment, effectively bridging modality
   246	applications requiring high performance. The overall workflow of the proposed DFF module is
   259	and local feature extraction are performed by LEMA (b) and AGC (c) in the DFF module, respectively. The
   260	fused features are fed to the DPE module for informative embedding generation by our proposed Deformable
   261	Dilated Conv with different dilation rates, where the generation is constrained by our proposed multi-loss
   270	enhanced multi-head attention (LEMA). This approach captures both local details and the
   271	global context within a single framework. By combining these techniques, the design overcomes
   280	the use of pure self-attention mechanisms alone, this hybrid approach offers superior computational
   321	This approach effectively adjusts the range of values of the input feature map, which improves the
   341	these issues, we introduce a local feature enhancement approach. This method can effectively
   439	via the Kaiming initialization strategy [7]. This approach ensures effective gradient propagation
   441	module dynamically generates convolution kernels and biases for adaptive processing of various

===== Multi-year long-term person re-identification using gait and HAR features.pdf =====
    27	We propose a two-stream person re-identiﬁcation (Re-ID) framework that integrates gait and human action recognition (HAR) through cross-attention fusion. The model processes gait sequences via a BiLSTM-based encoder to
    31	method on a newly curated long-term spatio-temporal dataset of ultra-distance runners captured in natural outdoor settings across multiple locations spanning three years (2020 to 2023). Experimental results demonstrate
    32	that integrating HAR signiﬁcantly enhances gait-based Re-ID performance. Compared to gait-only models, our approach yields a 12 % improvement in mean Average Precision (mAP) in cross-year scenarios and up to an 11.6 %
    57	light-weight. Then, we incorporate a cross-attention scheme enabling interactive mixing of the two feature streams, allowing for a ﬂexible combination of activity and motion cues. We utilized a triplet loss function
    62	Experiments show that fusing gait and HAR outperforms singlestream and visual-only baselines, yielding more robust embeddings. Results highlight the potential of motion-based multimodal representations
    92	evaluation of gait-based Re-ID under realistic conditions of appearance
   116	changes on recognition performance [13]. However, their approach
   133	training objective using triplet loss with semi-hard negative sampling.
   213	train in parallel with the combining module, aligning with typical practice in recent video recognition backbones (e.g., C2D [21], I3D [22],
   280	of the framework is identity recognition rather than activity categorization. Gait embeddings are explicitly trained to discriminate identities
   328	3.3. Triplet loss for metric learning
   336	loss function. It compares three examples at a time:
   369	extra margin 𝛼 for safety. The loss is deﬁned as:
   390	don’t penalize it. If it doesn’t, the loss becomes positive and the model
   448	We deﬁne a modality-aware distance used both in the loss function
   518	form the foundation of our two-stream framework, with the gait backbone focusing on ﬁne-grained motion dynamics and the HAR backbone
   527	GaitGL [25] reﬁnes this approach by combining global and local feature extraction branches with 3D convolutional layers to capture holistic
   567	results over several runs to mitigate variance introduced by temporal
   574	to monitor performance. We employed the Triplet Loss with a margin (𝛼) 0.6 to encourage a clear separation between positive and negative pairs in the embedding space. Performance was assessed using
   578	Experimental protocol. We performed a ﬁve times 10-fold split
   580	training, one for validation, and one for internal testing. This approach

codex
我已经定位到几篇的方法边界：有的章节标题叫 `Methodology`，有的叫 `Proposed approach`，也有的把方法放在第 3 节但标题提取不明显。下面我会分块读方法正文，重点看作者从观察到机制的论证链，而不是只记模块名。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '207,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   207	Methodology
   208	
   209	3.1
   210	
   211	Overview
   212	
   213	In this section, we provide a detailed description of the proposed MDFN framework, as illustrated
   214	in Figure 2. The framework processes two modalities of input images: VIS for the VIS branch and
   215	IR for the IR branch. The MDFN comprises four main components: the backbone, the DFF module,
   216	the dynamic perception enhancement (DPE) module, and the FRS module. In this paper, we
   217	employ ResNet-50 [8] as the backbone network. Given VIS-infrared (VIS-IR) image pairs with
   218	corresponding labels, we first pass them through Stages 0-3 of the backbone for initial feature
   219	extraction. These extracted VIS-IR feature pairs are then input into the DFF module, where local
   220	and long-range dependencies among features for each modality are extracted and fused. The fused
   221	features are subsequently fed into the DPE module, which captures contextual information at
   222	various scales and enriches the feature representation. Additionally, we design an MHC loss to
   223	constrain the distance between VIS and IR features across different scale embeddings generated by
   224	the DPE module. Furthermore, we employ the orthogonal Gramian matrix loss (OGM loss) to
   225	encourage the independence or orthogonality of the generated embeddings, thereby optimizing
   226	the utilization of the embedding information.
   227	3.2
   228	
   229	DFF Module
   230	
   231	In traditional VI-ReID methods, feature extraction is typically performed via convolutional
   232	neural networks (CNNs) [14] and vision transformers (ViTs) [3]. CNNs excel at capturing
   233	local features, whereas ViTs are proficient at capturing long-range dependencies among features.
   234	However, static convolution introduces an inductive bias that, although beneficial, limits the model’s
   235	ability to learn representations effectively. To address this limitation, we propose a DFF module,
   236	which includes both an attention branch and a dynamic convolutional branch. The DFF module
   237	introduces a multi-scale fusion approach that combines both local and global feature extraction to
   238	increase feature robustness. Specifically, the local unit uses AGC to capture fine-grained details,
   239	whereas the global unit leverages an improved multi-head self-attention mechanism to capture
   240	contextual information across larger scales [31]. This multi-scale fusion enables the DFF module
   241	to adapt dynamically to varying conditions, such as lighting changes and occlusions, enhancing
   242	the model’s generalizability in complex and dynamic environments. By integrating both local and
   243	global features, the DFF module strengthens cross-modal alignment, effectively bridging modality
   244	gaps and improving matching accuracy across VIS and IR images. Additionally, DFF achieves an
   245	efficient balance between computational cost and precision, making it well-suited for real-time
   246	applications requiring high performance. The overall workflow of the proposed DFF module is
   247	illustrated in Figure 2.
   248	For an IR-VIS light hybrid feature 𝐹 𝑣𝑖 ∈ R𝐶 ×𝐻 ×𝑊 extracted by the backbone network, we first
   249	input it to the position embedding (PE), where the convolution kernel of PE(·) is 7 × 7, the group
   250	is the channel dimension of the input feature map, and padding is 3. The results are then summed
   251	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.
   252	
   253	101:6
   254	
   255	S. Wang et al.
   256	
   257	Fig. 2. Architecture of the proposed MDFN. (a) MDFN overview. The input VIS and IR images go through
   258	a preliminary feature extraction by stages 0-3, and then the long-distance dependencies between features
   259	and local feature extraction are performed by LEMA (b) and AGC (c) in the DFF module, respectively. The
   260	fused features are fed to the DPE module for informative embedding generation by our proposed Deformable
   261	Dilated Conv with different dilation rates, where the generation is constrained by our proposed multi-loss
   262	functions. The features are finally enhanced by our proposed FRS in matching phase.
   263	
   264	with the original input to generate feature maps 𝐹 𝑣𝑖+ ∈ R𝐶 ×𝐻 ×𝑊 :
   265	𝐹 𝑣𝑖+ = 𝐹 𝑣𝑖 + PE(𝐹 𝑣𝑖 ).
   266	
   267	(1)
   268	
   269	Next, 𝐹 𝑣𝑖+ is normalized and split into two parts, which are processed through AGC and local
   270	enhanced multi-head attention (LEMA). This approach captures both local details and the
   271	global context within a single framework. By combining these techniques, the design overcomes
   272	the limitations of relying solely on either convolutional or self-attention mechanisms, thereby
   273	enhancing the model’s flexibility and expressiveness when handling diverse visual features. The
   274	AGC component employs a dynamic convolution kernel that is adjusted adaptively on the basis
   275	of the input features. This adaptability allows the network to automatically optimize convolution
   276	operations to adapt to various input data, a capability that traditional static convolutional layers
   277	lack. This dynamic adjustment significantly improves the model’s adaptability and representation
   278	power. In tandem, LEMA combines the multi-head self-attention mechanism with local convolution
   279	to effectively capture long-range dependencies while representing local structures. Compared with
   280	the use of pure self-attention mechanisms alone, this hybrid approach offers superior computational
   281	efficiency.
   282	For the input feature 𝐹 𝑣𝑖+ , we first evenly divide it into two subfeature mappings along the
   283	
   284	𝐶
   285	channel dimension, denoted as 𝐹 1𝑣𝑖+, 𝐹 2𝑣𝑖+ ∈ R 2 ×𝐻 ×𝑊 . 𝐹 1𝑣𝑖+ and 𝐹 2𝑣𝑖+ are then fed into the attention branch LEMA(·) and the dynamic convolution branch AGC(·), respectively, to produce the
   286	
   287	𝐶
   288	corresponding feature maps 𝐹 1𝐺 , 𝐹 2𝐿 ∈ R 2 ×𝐻 ×𝑊 , which are subsequently concatenated along the
   289	channel dimension. Next, we apply perceptual embedding enhancement (PEE) to the concatenated
   290	𝐹 1𝐺 and 𝐹 2𝐿 (F ∈ R𝐶 ×𝐻 ×𝑊 ) to extract richer feature representations and enhance the model’s
   291	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.
   292	
   293	Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification
   294	
   295	101:7
   296	
   297	expressive power. These enhanced features are then summed with F to obtain the feature vector
   298	𝐹 + ∈ R𝐶 ×𝐻 ×𝑊 .
   299	PEE(·) employs depthwise separable convolution followed by pointwise convolution, with additional nonlinear activation and normalization layers, which maintain high efficiency while
   300	preserving high performance. Specifically, the input feature map F is first processed through a
   301	depthwise separable convolutional layer that applies a single deep convolution kernel K𝑑𝑤 to
   302	each input channel. This is followed by the rectified linear unit (ReLU) and batch normalization
   303	(BN). Next, the convolution kernel K𝑝𝑤1 is used for pointwise convolution, reducing the number
   304	of channels to 1/4 of the original number. This is then input to ReLU(·) and BN(·) to enhance the
   305	model’s nonlinear representation and batch normalization, respectively. Finally, the convolution
   306	kernel K𝑝𝑤2 is used to restore the original number of channels through pointwise convolution,
   307	followed by BN(·) processing. The overall transformation can be expressed as follows:
   308	PEE(F ) = BN(K𝑝𝑤2 ∗ BN(ReLU(K𝑝𝑤1 ∗ BN(ReLU(K𝑑𝑤 ∗ F ))))),
   309	
   310	(2)
   311	
   312	where K𝑑𝑤 has a convolution kernel of 3 × 3 and a padding of 1, the K𝑝𝑤1 kernel K𝑝𝑤2 has a
   313	convolution kernel of 1 × 1 and * denotes the convolution operation. Overall, 𝐹 + is represented as:
   314	𝐹 + = F + PEE(F ).
   315	
   316	(3)
   317	
   318	Finally, 𝐹 + is fed into the LayerScale (LS) layer, where LS(·) adjusts the values of the input
   319	feature maps by introducing learnable scaling factors and bias terms. Specifically, it scales the
   320	input via a trainable weight parameter 𝑊 ∈ R𝐶 ×1×1×1 and adds a trainable bias parameter 𝑏 ∈ R𝐶 .
   321	This approach effectively adjusts the range of values of the input feature map, which improves the
   322	training stability of the model and the representation capability. LS(·) is equivalent to the following
   323	calculation for each input channel c:
   324	LS(𝐹𝑐+ ) = 𝑊𝑐 · 𝐹𝑐+ + 𝑏𝑐 ,
   325	
   326	(4)
   327	
   328	where 𝑊𝑐 and 𝑏𝑐 are the learnable weight and bias parameters for channel c, respectively. The
   329	outputs obtained from LS(·) and 𝐹 𝑣𝑖+ are added to obtain the final output 𝐺 ∈ R𝐶 ×𝐻 ×𝑊 . The overall
   330	process can be expressed as:
   331	𝐺 = 𝐹 𝑣𝑖+ + LS(𝐹𝑐+ ).
   332	
   333	(5)
   334	
   335	LEMA. Multi-head self-attention mechanisms are commonly employed in vision tasks to capture
   336	long-range dependencies among features. However, standard multi-head self-attention involves
   337	pairwise interactions between all positions in the sequence, resulting in a quadratic increase in
   338	computational complexity and memory requirements as the input sequence length increases [13].
   339	Furthermore, while these mechanisms excel at capturing long-range dependencies, they often
   340	overlook the significance of local spatial structures, which are crucial for VI-ReID tasks. To address
   341	these issues, we introduce a local feature enhancement approach. This method can effectively
   342	capture the local correlation between neighboring pixels or features, preserve spatial information,
   343	and improve the efficiency of the model in extracting fine-grained details. The specific process is
   344	illustrated in Figure 2(b).
   345	𝐶
   346	Specifically, for a given input feature 𝐹 1𝑣𝑖+ ∈ R 2 ×𝐻 ×𝑊 , a linear layer is first used and reshaped to
   347	𝐶
   348	map the input feature to a query vector 𝑄 ∈ Rℎ× (𝐻 ·𝑊 ) × 2×ℎ , where C is the number of channels, H
   349	and W are the height and width of the feature map, respectively, and h is the number of attention
   350	heads. A Local Conv (LC) with 3 × 3 groups as dimensions is then applied to enhance its local
   351	features, which are then summed with the original input, followed by a 1 × 1 convolution KV(·)
   352	with a doubled number of channels, which is then split along the channel dimension into two
   353	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.
   354	
   355	101:8
   356	
   357	S. Wang et al.
   358	𝐶
   359	
   360	key-value pairs 𝐾, 𝑉 ∈ Rℎ× (𝐻 ·𝑊 ) × 2×ℎ . Finally, we calculate the scaled dot-product attention based
   361	𝐶
   362	on Q, K, and V to obtain 𝐹 1𝐺 ∈ R 2 ×𝐻 ×𝑊 , as shown in the following procedure:
   363	𝐾, 𝑉 = Split(KV(𝐹 1𝑣𝑖+ + LC(𝐹 1𝑣𝑖+ )))
   364	(6)
   365	
   366	© 𝑄𝐾 𝑇 ª
   367	® ·𝑉.
   368	𝐹 1𝐺 = Softmax ­­ q
   369	®
   370	«
   371	
   372	𝐶
   373	2×ℎ ¬
   374	
   375	AGC. To inject inductive bias and perform local feature aggregation in a dynamic input-dependent
   376	manner, we propose a novel dynamic deep convolution called AGC. As shown in Figure 2(c), an
   377	𝐶
   378	input feature map 𝐹 2𝑣𝑖+ ∈ R 2 ×𝐻 ×𝑊 is taken, and the global contextual information is extracted
   379	via adaptive pooling, which compresses the spatial dimension to 𝐾 2 . Moreover, to increase the
   380	feature representation power and reduce the computational complexity, the input is then fed to the
   381	projection layer, which effectively adjusts the dimensionality and expression capability of the input
   382	features through a series of 1×1 convolutional layer dimensionality reductions, batch normalization,
   383	nonlinear activation and 1 × 1 convolutional layer dimensionality enhancements to improve the
   384	performance and efficiency of the whole neural network. It is then deformed and normalized in
   385	𝐶
   386	2
   387	the G dimension via the Softmax function to generate the attention weights 𝑆 𝑤 ∈ R𝐵×𝐺 × 2 ×𝐾 ,
   388	with the attention weights 𝑆 𝑤 reflecting the importance of each channel and group. G denotes the
   389	number of attention groups and K denotes the size of the convolutional kernel. A set of predefined
   390	𝐶
   391	2
   392	base convolution kernels 𝑊 ∈ R𝐺 × 2 ×𝐾 are then utilized in conjunction with the attentional
   393	𝐶
   394	2
   395	weights 𝑆 𝑤 to generate the dynamic convolution kernel 𝑊𝑑 𝑦𝑛 ∈ R𝐵× 2 ×𝐾 . Specifically, the output
   396	is summed over the G-dimension via element-by-element multiplication:
   397	𝑊𝑑𝑦𝑛 =
   398	
   399	𝐺
   400	Õ
   401	
   402	𝑆 𝑤,𝑔
   403	
   404	(7)
   405	
   406	𝑊𝑔 ,
   407	
   408	𝑔=1
   409	
   410	where denotes element-by-element multiplication, ensuring that each set of convolutional weights
   411	is dynamically adjusted to the global contextual information of the input features. In the same way,
   412	replacing AP(·) with mean(M), we can generate dynamic attention weights 𝑏𝑑 𝑦𝑛 ∈ R𝐵×𝐶 :
   413	𝑏𝑑𝑦𝑛 =
   414	
   415	𝐺
   416	Õ
   417	
   418	𝑆𝑏,𝑔
   419	
   420	(8)
   421	
   422	𝑏𝑔 ,
   423	
   424	𝑔=1
   425	
   426	where 𝑆𝑏 ∈ R𝐵×𝐺 × 2 is the generated adaptive bias weight and where 𝑏 ∈ R𝐺 × 2 is the initial
   427	bias. Finally, the dynamically generated convolution kernels 𝑊𝑑 𝑦𝑛 and 𝑏𝑑 𝑦𝑛 are applied to perform
   428	𝐶
   429	convolution operations on the input feature map to obtain 𝐹 2𝐿 ∈ R 2 ×𝐻 ×𝑊 :
   430	𝐶
   431	
   432	𝐶
   433	
   434	𝐹 2𝐿 = Conv(𝐹 2𝑣𝑖+,𝑊𝑑 𝑦𝑛 , 𝑏𝑑 𝑦𝑛 )
   435	
   436	(9)
   437	
   438	To facilitate efficient training and prevent gradient explosion, all weight parameters are initialized
   439	via the Kaiming initialization strategy [7]. This approach ensures effective gradient propagation
   440	following nonlinear activation functions. Since different inputs produce distinct attention maps, the
   441	module dynamically generates convolution kernels and biases for adaptive processing of various
   442	input samples and features. This dynamic adjustment enhances the model’s expressiveness and
   443	adaptability in complex and varying environments.
   444	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.
   445	
   446	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf' - | nl -ba | sed -n '320,848p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 146ms:
   320	Fig. 2. Structure of our proposed method. Source 1 and Source 2 represent small- and large-scale datasets respectively. Our method divides the biased source
   321	domains into multiple debiased sub-source domains and learn these subdomains in parallel to achieve both domain balance and domain diversity. Beside, the
   322	structure synergy perception, cross-model central distillation and parts graph distillation are introduced to intensify the collaborative learning among submodels
   323	j
   324	for improving the model semantic and domain generalization ability. Ri is the affinity matrix of the jth sobdomain modeled by the i-th submodel.
   325	
   326	layers. META introduces Instance Normalization (IN) into the
   327	global branch to explore domain invariant features and leverages other models to exploit domain specific features. And
   328	multi models is aggregated by considering the relevance of an
   329	unseen target sample and source domains via normalization
   330	statistics. Federated learning methods [62], [63], [64] leverages
   331	multiple models to explore the privacy-preserving datasets,
   332	which employs a central training strategy to integrate multimodel knowledge. During training process, a central model is
   333	leveraged to aggregate and average multi-model parameters
   334	and then return it to each model. Unlike these works, we
   335	propose a multi-model synergy perception framework that
   336	employs a asynchronous training paradigm to relieve domain
   337	imbalance issue and develops a synergy learning approach
   338	to reinforce the collaborative learning among submodels for
   339	improving the model both semantic and domain generalization
   340	ability.
   341	
   342	Paradigm (ATP) for biased open-world ReID. The overall
   343	framework of our method is shown in Fig. 2. To relieve
   344	the issue of domain bias among different sources, we first
   345	propose an asynchronous sampling paradigm for large- and
   346	small-scale domains to form multiple debiased subdomains.
   347	With the philosophy of divide and conquer, a multi-network
   348	architecture is employed to learn all subdomains in parallel.
   349	Then to better generalize knowledge across these sub-source
   350	domains, we propose a Structure Synergy Perception (SSP)
   351	module, which aligns the feature relationship distribution
   352	among subdomains for unique knowledge synergy learning.
   353	Furthermore, considering the single-view limitation of each
   354	submodel, we propose a Synergy Distillation Perception (SDP)
   355	to aggregate views of different submodels. The SDP comprises
   356	semantic based cross-model central distillation (CCD) and
   357	parts graph distillation (PGD), which uses the center guided
   358	soft label and the part based triplet graph respectively to distill
   359	each submodel for generalization learning.
   360	
   361	III. M ETHOD
   362	Problem Setup. For multi-source domain generalization
   363	(DG) in person ReID, we are provided with K source domains
   364	K
   365	D = {Dk }k=1
   366	in the training stage.
   367	Each domain contains
   368	˚
   369	Nk
   370	its own image-label space Dk = (xik , yki ) i=1
   371	, where Nk is the
   372	number of images in the source domain Dk . Each sample
   373	xik ∈ Xk is labeled as an identity yki ∈ Yk = {1, 2, . . . , Hk },
   374	where Hk is the number of identities in the domain Dk .
   375	The conventional DG methods for classification tasks assume
   376	that different source domains share the same label space as
   377	Yi = Y j = Y∗ , ∀i, j, 1 ≤ i, j ≤ K. While in DG ReID
   378	setting, the label spaces of the source domains are disjointed
   379	as Yi , Y j , Y∗ . So
   380	DG ReID, the number of identities is
   381	Pin
   382	K
   383	expressed as H = k=1
   384	Hk . During training phase, all source
   385	domains are aggregated to jointly train a generalizable model.
   386	During testing phase, the trained model is directly applied to
   387	an open domain for person image retrieval.
   388	A. Overview
   389	This paper proposes a Multi-model Synergy Perception
   390	(MSP) framework equipped with an Asynchronous Training
   391	
   392	B. Asynchronous Training Paradigm
   393	The conventional DG ReID methods [13], [54] often rely
   394	on the domain diversity of multi-source domains to explore
   395	domain-invariant representations, while neglecting the domain
   396	imbalance issue that can lead to sub-optimal generalization.
   397	To relieve this problem, we develop a Asynchronous Training
   398	Paradigm (ATP) to exploit the source domains through multiparallel submodels.
   399	Under-sampling is an efficient method for data-imbalance
   400	issue, which uses only subset of the large-scale domain to train
   401	the model. Since the trained large-scale data is equal to smallscale data, the domain balance can be achieved. However,
   402	such under-sampling leads to the loss of valuable information
   403	in large-scale domain. The proposed asynchronous training
   404	paradigm can effectively exploit all valuable knowledge, while
   405	keeping the domain balance. As illustrated in Fig. 2, the
   406	ATP first divides the biased source domains into multiple
   407	debiased subdomains without compromising domain diversity.
   408	To maintain domain balance and domain diversity, we construct a multi-network architecture to exploit these subdomains
   409	
   410	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:29 UTC from IEEE Xplore. Restrictions apply.
   411	
   412	LIU AND ZHANG: MULTI-MODEL SYNERGY PERCEPTION FOR OPEN-WORLD PERSON RE-IDENTIFICATION
   413	
   414	in parallel. Specifically, during training process, we sample
   415	n batches from the large-scale domain (L-Batch) and one
   416	batch from the small-scale domain (S-Batch), represented
   417	as S 11 and [S 1K , S 2K , · · ·, S nK ] in Fig. 2, where S 1 and S K
   418	denote the small- and large-scale domain respectively and the
   419	superscript denotes different batch data. And the number of
   420	batches of other domains ranges from 1 to n. Then each
   421	L-Batch is combined with the shared S-Batch ˇto ˇform
   422	ˇ ˇa
   423	subdomain and we adjust the parameter n to make ˇS 11 ˇ=ˇS nK ˇ,
   424	which effectively keeps domain balance of each subdomain.
   425	And in this way, we can obtain n debiased subdomains, and
   426	each subdomain can keep both domain balance and domain
   427	diversity that is beneficial for domain invariant representations.
   428	Since domain imbalance also exists among subdomains, we
   429	employ a multi-network architecture to exploit all subdomains
   430	in parallel instead of simple one-stream learning. Different
   431	submodels are independent of each other and have different
   432	parameters to keep
   433	balance during
   434	training.
   435	˚ domain
   436	˚
   437	Nl
   438	N
   439	and D s = (xis , yis ) i=1s to represent
   440	We use Dl = (xil , yli ) i=1
   441	the large- and small-scale domain, where Nl and N s are
   442	the number of images of the source domain Dl and D s
   443	respectively. And the number of batch size is denoted as B,
   444	so the small-scale domain needs N s /B iterations to traverse
   445	all data. But the large-scale needs Nl /B iterations, which is
   446	larger than that of small-scale domain. And to keep them in
   447	sync, we sample n batches from large-scale domain in each
   448	iteration, and the n can be computed by:
   449	  
   450	
   451	Nl
   452	Nl /B
   453	=
   454	(1)
   455	n=
   456	N s /B
   457	Ns
   458	where d·e represents the ceiling function.
   459	Suppose the i-th and j-th submodel to be optimised with a
   460	learning rate η, then its model parameters wi,t+1 and w j,t+1 at
   461	the (t + 1)-th local step are updated as:
   462	wi,t+1 = wi,t − η∇G(S 11 , · · ·S li%nl , · · ·S iK )
   463	w j,t+1 = w j,t − η∇G(S 11 , · · ·S lj%nl , · · ·S Kj )
   464	
   465	(2)
   466	
   467	where ∇G(·) represents the computed gradient. S iK and S Kj
   468	are different batch data sampled from the largest-scale source
   469	domain and S 11 is the small-scale data shared between local
   470	sub-models. S l represents the l − th source domain, whose size
   471	lies between that of the 1−th and the K −th source domains. nl
   472	is the number of samples from the l−th source domain, which
   473	is computed by Eq. 1. And % denotes the modulo operation.
   474	C. Structure Synergy Perception Learning
   475	Conventional multi-model methods such as federated learning [62], [63], [64] often employ parameters integration
   476	manner to jointly learn multiple models, which may impair
   477	generalization learning due to domain imbalance in DG ReID.
   478	Many methods [65], [66] leverage the feature relationship
   479	to exploit potential knowledge of model, but they directly
   480	align the features of different models that can hardly explore
   481	generalizable knowledge. So we propose a structure synergy
   482	perception (SSP) module for synergy learning among submodels. The feature structure distribution of each subdoamin is first
   483	built based on feature distance relationship, and then we align
   484	
   485	10895
   486	
   487	Fig. 3. The basic idea of the structure synergy perception learning. The
   488	feature-level relationship matrix is first built in each submodel. Then we align
   489	this relationship matrix among local sub-models to reinforce the collaborative
   490	learning among them.
   491	
   492	the feature structure among submodels to facilitate the model
   493	exploring generalizable knowledge, as shown in Fig. 3.
   494	Specifically, we first build up the feature-level relationship distribution matrix for each dubdomain. Let F ij =
   495	i
   496	i
   497	i
   498	} be the features of the i-th subdomain
   499	, . . . , f jB
   500	, f j2
   501	{ f j1
   502	extracted from the j-th submodel, where B is the batch size
   503	and f ∈ RC . It can be known that the parallel learning of
   504	ATP is employed on the features F ij when j = i for supervised learning. And to reinforce synergy perception among
   505	submodels, we map the unique supervised knowledge of ATP
   506	to other submodels by structure synergy learning, represented
   507	as RF ij |( j=i) → RF ij |( j,i) , where R(·) denotes the structure relationship of features. And we can obtain the feature structure of
   508	each subdomain, i.e., the affinity matrix, by calculating the
   509	relationships between features:
   510	R(a,b) = P
   511	
   512	exp(D( fa , fb ))
   513	k∈N exp(D( fa , fk ))
   514	
   515	(3)
   516	
   517	where N denotes the sample space and R(a,b) represents
   518	the relationship of feature fa and fb . D(·) represents the
   519	Euclidean distance and exp(·) denotes exponential function.
   520	The affinity matrix is obtained by computing the similarity
   521	scores among features through softmax function. Specifically,
   522	given the subdomain i, we use all submodels to extract
   523	its features and model its relationship structure respectively,
   524	represented as {Ri1 , Ri2 , . . . , Rii , Rin }, where the subscript represents different submodel. Note that for the i-th subdomain,
   525	only the features extracted from the i-th submodel are trained
   526	in parallel learning to keep the domain balance. So we leverage
   527	the i-th relationship structure Rii to guide the learning of
   528	other submodels for synergy learning of multi-submodels. We
   529	develop a synergy perception loss function LS S P to guide the
   530	structure distribution learning, and it can be represented as:
   531	LS S P =
   532	
   533	n
   534	n
   535	X
   536	X
   537	X
   538	i=1 j=1, j,i
   539	
   540	(a,b)∈Ni
   541	
   542	KL(Rii(a,b) , Rij (a,b) )
   543	
   544	(4)
   545	
   546	where KL(·) represents the Kullback-Leibler (KL) divergence.
   547	n and Ni represent the number of subdomains and the i-th subdomain space respectively. And the superscript and subscript
   548	of Rij denote the subdomain and submodel respectively.
   549	
   550	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:29 UTC from IEEE Xplore. Restrictions apply.
   551	
   552	10896
   553	
   554	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 11, NOVEMBER 2025
   555	
   556	D. Synergy Distillation Perception
   557	To further improve the model generalization ability, we propose a Synergy Distillation Perception(SDP) module to exploit
   558	multi-view generalizable knowledge of different submodels.
   559	The SDP primarily comprises cross-model central distillation
   560	(CCD) and parts graph distillation (PGD), which uses the
   561	center guided soft label and the parts based triplet graph to
   562	distill each submodel respectively to improve the model both
   563	semantic and domain generalization ability.
   564	Fig. 4. Overview of cross-model central distillation (CCD). CCD extracts the
   565	center logits value of different models as soft label to distill each local model
   566	for generalizable model learning.
   567	
   568	E. Cross-Model Central Distillation
   569	The conventional supervision of hard labels (i.e., one-hot
   570	labels) can easily lead to the model over-fitting the source
   571	identities [13] and is difficult to generalize the knowledge to
   572	open domains with different IDs and domain styles. Considering the different views of different submodels, we aim to
   573	design a soft label of multi-views as supervision signal to
   574	guide the model to exploit the generalized semantic content
   575	for retrieval rather than over-fitting. In our paper, we propose
   576	a central distillation loss LDis based on ATP, which extracts
   577	the center value of different models as soft label to distill each
   578	local model for the generalized semantic learning.
   579	Specifically, we generate the center value by averaging the
   580	output logits from the shared data of different parallel local
   581	models. Next, we leverage the proposed central distillation loss
   582	to penalize each output mismatched with the center, which
   583	encourages the parallel models to learn generalized semantic
   584	features. Formally, given an input image of x, we denote its
   585	output logit of the i-th sub-model as zi (x). Firstly, we obtain
   586	the center ensemble logits of the parallel sub-models by taking
   587	an average as:
   588	n
   589	1X
   590	zi (x)
   591	(5)
   592	z̄(x) =
   593	n
   594	i=1
   595	
   596	where n is the number of parallel sub-models. Then, we
   597	generate the soft label by computing the predictive distribution
   598	of the center ensemble:
   599	exp(z̄c (x)/τ)
   600	P̄(c|x; θ, τ) = PC
   601	j
   602	j=1 exp(z̄ (x)/τ)
   603	
   604	(6)
   605	
   606	where C represents the number of identities and z̄c (x) denotes
   607	the center value of the c-th label. The θ and τ represent the
   608	model parameters and temperature scaling parameter respectively. We then propose to distill each local sub-model by using
   609	the proposed central distillation loss LDis , represented as:
   610	LDis (x; θ, τ) =
   611	
   612	n X
   613	C
   614	X
   615	
   616	KL(P̄(c|x; θ̂, τ)||Pi (c|x; θ, τ))
   617	
   618	(7)
   619	
   620	i=1 c=1
   621	
   622	where Pi (c|x; θ, τ) represents the predictive distribution of the
   623	i-th model and KL(·||·) represents the Kullback-Leibler (KL)
   624	divergence. θ̂ denotes the parameters of teacher, which is a
   625	fixed copy of the parameter θ in our method. Following [67],
   626	[68], we stop the gradient to be propagated through θ̂ to
   627	prevent the model from falling into some trivial solutions.
   628	
   629	F. Parts Graph Distillation
   630	Since the image-level visual similarities is unable to eliminate the domain noise as this noise may exist in all features
   631	extracted from the same model, we aim to explore the parts
   632	relationship among intra- and inter-subdomains for generalization learning. Considering the visual difference of different
   633	local parts, we develop a Parts Graph Distillation (PGD)
   634	module that leverages this valuable prior knowledge to explore
   635	the domain invariant features. There should be a large distance
   636	among different parts even from the same ID, and we develop a
   637	domain noise filter loss L f to construct the parts graph of each
   638	subdomain. Then to further improve the model generalization
   639	ability, we develop a parts distillation loss L p to align the
   640	parts graph among submodels for exploiting domain invariant
   641	features.
   642	Specifically, given images xi and x j of the same ID, its
   643	features extracted from the a-th submodel is represented as
   644	fai and faj . We split the images into m local parts and its
   645	representation can be represented as { fai |1 , fai |2 , . . . , fai |m } and
   646	{ faj |1 , faj |2 , . . . , faj |m } respectively. Each feature can disentangled into tow parts: identity feature f + and domain bias feature
   647	f − . Due to the consistency of domain bias, the domain bias
   648	can be similar within the same sub-model, represented as fa− ,
   649	where a is the a − th sub-model. And the local parts can be
   650	−
   651	i+
   652	−
   653	i+
   654	−
   655	redefined as {[ fai |+
   656	1 + fa ], [ fa |2 + fa ], . . . , [ fa |m + fa ]} and
   657	j+
   658	j+
   659	j+
   660	−
   661	−
   662	−
   663	{[ fa |1 + fa ], [ fa |2 + fa ], . . . , [ fa |m + fa ]}. The optimization of
   664	j+
   665	−
   666	semantic content can be represented as min([ fai |+
   667	1 + fa ], [ fa |1 +
   668	−
   669	fa ]). Clearly, such intra-submodel optimization is unable to
   670	remove domain noise. In contrast, different submodels can
   671	share the same semantics while having different domain bias.
   672	Based on this idea, we design the parts graph distillation for
   673	domain generalization learning. The inter-subdomain PGD can
   674	be represented as:
   675	j)
   676	(i, j)
   677	L = KL(P(i,
   678	(a,a) , P(a,b) )
   679	
   680	= P( fai |+ + fa− , fbj |+ + fb− ) → P( fai |+ , faj |+ )
   681	
   682	(8)
   683	
   684	where P is the parts relationship graph, as shown in Eq. 11.
   685	Based on the guidance of this loss, the sub-model is forced to
   686	exploit the domain generalization knowledge.
   687	Specifically, we set each part as a node and connect all
   688	nodes of these two images to build up the parts relationship
   689	matrix, as shown in fig 5. For example, the relationship
   690	
   691	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:29 UTC from IEEE Xplore. Restrictions apply.
   692	
   693	LIU AND ZHANG: MULTI-MODEL SYNERGY PERCEPTION FOR OPEN-WORLD PERSON RE-IDENTIFICATION
   694	
   695	10897
   696	
   697	TABLE II
   698	S UMMARY OF A LL THE DATASETS
   699	
   700	Fig. 5. Overview of parts graph distillation (PGD). The PGD first models the
   701	parts graph of subdomains and then exploits the domain invariant features by
   702	intra- and inter-subdomain distillation.
   703	
   704	between the parts fai | p and faj |q of images xi and x j can be
   705	represented as:
   706	mi·p·qj = D( fai | p , faj |q )
   707	(9)
   708	where D(·, ·) represents the Euclidean distance and Mai· j =
   709	{mi·p·qj } ∈ Rm×m denotes the relationship matrix of the images
   710	xi and x j in the a-th submodel. Based on the prior knowledge
   711	that there should be a large distance between different parts
   712	of the same ID, we develop a domain noise filter loss L f to
   713	relieve domain noise, which can be represented as:
   714	m
   715	X
   716	
   717	Lf =
   718	
   719	(S o f tplus(d( fai | p , faj | p ) − d( fai | p , faj |q )))
   720	
   721	(10)
   722	
   723	q=1,q,p
   724	
   725	where S o f tplus(·) = ln(1 + exp(·)) is a function with monotonic increase, designed to alleviate optimization challenges
   726	by circumventing negative values in loss. i and j represent
   727	different images of the same identity and subscript a denotes
   728	the submodel. And in this way, we can construct the parts
   729	relationship graph by exploiting its distribution, which can be
   730	defined as:
   731	(i, j)
   732	|(p,q) = P
   733	P(a,b)
   734	
   735	exp(D( fai | p , fbj |q ))
   736	j
   737	i
   738	p∈N exp(D( fa | p , fb |q ))
   739	
   740	(11)
   741	
   742	(i, j)
   743	|(p,q) represents relationship of the parts p and q,
   744	where P(a,b)
   745	which is from the image i and the image j respectively. a and
   746	b denote the submodel and P(a,b|b=a) and P(a,b|b,a) represent the
   747	intra- and inter-submodel parts graph respectively. To obtain
   748	multi-view generalization knowledge, we further propose a
   749	cross-model parts distillation loss L p and construct the intraand inter-submodel parts graph for synergy distillation. The
   750	parts distillation loss L p is defined as:
   751	
   752	Lp =
   753	
   754	n
   755	X
   756	X
   757	
   758	The triplet loss of Ltri employs a widely-used batch-hard
   759	triplet loss [23]. Considering that the number of conventional
   760	classifier parameters will surge when multiple domains are
   761	used for training, we employ a more efficient non-parametric
   762	memory-based classifier as [54]:
   763	exp(M[i]T f (xi )/τ)
   764	Lid = − log PnT
   765	k=1 exp(M[k]T f (xi )/τ)
   766	
   767	(15)
   768	
   769	where T and nT denote the source domain T and the total
   770	identities in this domain and τ is the temperature factor that
   771	controls the scale of distribution. M represents the memory
   772	bank of identity centroid, which is initialized by the initial
   773	embedding features and updated by:
   774	1 X
   775	f (xi )
   776	(16)
   777	M[k] ← ϕ · M[k] + (1 − ϕ)
   778	|Vk |
   779	xi ∈Vk
   780	
   781	where Vk and |Vk | denote the samples of the kth identity and
   782	the number. ϕ ∈ [0, 1] controls the updating rate.
   783	During testing, we propose to leverage the self-contrast to
   784	learn adaptive weights of each submodel. Given the testing
   785	image xt , and we augment it randomly, represented as x̂t . We
   786	use the fta and fˆta to represent the representation of these two
   787	images extracted from the a-th submodel. And its similarity
   788	can be represented as:
   789	S ta = D( fta , fˆta )
   790	
   791	(17)
   792	
   793	where D(·, ·) represents the Euclidean distance. And its adaptive weight wat can be computed by:
   794	wat =
   795	
   796	exp(−S ta )
   797	n
   798	P
   799	exp(−S tk )
   800	
   801	(18)
   802	
   803	k=1
   804	(i, j)
   805	(i, j)
   806	(i, j)
   807	(i, j)
   808	(KL(P(a,a)
   809	, P(b,b)
   810	) + KL(P(a,a)
   811	, P(a,b)
   812	)
   813	
   814	a,b=1 (i, j)∈N
   815	(i, j)
   816	(i, j)
   817	+ KL(P(b,b)
   818	, P(a,b)
   819	))
   820	
   821	(12)
   822	
   823	(i, j)
   824	(i, j)
   825	where P(a,a)
   826	and P(a,b)
   827	represent intra- and inter-submodel parts
   828	graph respectively. KL(·) denotes the Kullback-Leibler (KL)
   829	divergence and N represents the image space of subdomain.
   830	So the synergy distillation LS D can be represented as:
   831	
   832	LS D = LDis + L f + L p
   833	
   834	(13)
   835	
   836	G. Objective Function
   837	In this work, we use both identification loss and triplet loss.
   838	The total loss of the proposed method is calculated as:
   839	L = Lid + Ltri + LS S P + LS D
   840	
   841	(14)
   842	
   843	where exp(·) represents exponential function and n is the
   844	number of submodel. To prevent the randomization of individual image, we employ a batch of images to determine this
   845	parameter, which induces almost no extra inference overhead
   846	in real-world scenarios.
   847	IV. E XPERIMENTS
   848	A. Datasets and Settings

codex
第一篇的主线很清楚：它不是从“多模型更强”起步，而是从多源 ReID 的数据规模不平衡起步，再把“平衡采样会丢信息”改写成“要把大域切成多个平衡子域并并行协同”。第二篇第 8 页还没结束方法，我会继续只补足后面的 DPE、损失和匹配模块，否则会漏掉它真正的证据设计。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 14 'Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | rg -i '("'^'"\\s*[0-9]+\\s+([0-9]+\\.[0-9]|DPE|FRS|Loss|Experiments|Experimental|Evaluation|Conclusion)|DPE|FRS|MHC|OGM|loss|reweight|matching|dynamic perception|feature reweight)' | head -n 160" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     2	3.3
     6	DPE Module
    13	[53], we propose a DPE module. The DPE module introduces a novel approach to capture multi-scale
    18	the DPE module employs the MHC loss to refine feature embeddings, which strengthens crossmodal alignment by reducing modality disparities at multiple feature levels. This combination of
    20	ensures that critical identity cues are preserved and aligned across modalities, making the DPE
    22	As illustrated in Figure 2, our proposed DPE module generates multiple embeddings and focuses
    24	branches. Specifically, each branch of the DPE module utilizes three parallel 3 × 3 deformable dilated
    79	Fig. 3. The proposed method FRS, primarily involves two summations: one for the original distances and
    82	top-k feature enhancement, FRS addresses the noise issue stemming from information disparities, resulting
    83	ˆ Furthermore, FRS demonstrates significantly improved accuracy in
    87	A ReLU activation layer is then used to improve the nonlinear representation of the DPE. Next,
    89	map in order to change its size so that it has the same size as G. The DPE is then used to generate
   104	3.4
   106	FRS
   114	features influences the results. To address these challenges, we introduce feature reweighting on the
   115	basis of similarity (FRS). The FRS module uses advanced feature enhancement technology to solve
   117	the matching performance. The specific effects of the FRS are illustrated in Figure 3.
   118	FRS is a similarity-driven feature alignment method, with the core innovation lying in combining
   130	optimizing the matching performance, so we compute the feature similarity matrix 𝐺 · 𝐺𝑇 within
   152	only preserves the matching information in the original feature space, but also incorporates the
   155	3.5
   157	Multi-Loss Optimization
   160	crucial for improving model performance. To this end, our DPE module generates 𝐺𝑖+ embeddings
   167	and negative samples, MHCLoss enables detailed feature optimization from the macroscopic intermodal level to the microscopic intra-modal level.
   168	The core design principle of MHC loss is adaptive feature alignment across modalities, which is
   169	achieved by generating multiple contrastive pairs within a hierarchical structure. At each hierarchical level, MHC loss promotes feature similarity between corresponding VIS and IR samples while
   170	maximizing the distance between different identities. This hierarchical structure allows the loss
   172	at higher levels, effectively addressing the modality gap. The primary advantage of MHC loss is
   180	Fig. 4. The proposed loss function MHC, mainly primarily aims to minimize the distance between embeddings
   183	their initial counterparts, fostering intra-modality contrast. Through this dual process, MHC encourages the
   188	MHC loss enables robust cross-modal feature alignment, enhancing the model’s adaptability to
   192	Specifically, the inputs to MHCLoss are the VIS and IR modality features corresponding to the
   196	and where 𝑓𝑣𝑖 and 𝑓𝑖𝑟𝑖 are the new embeddings generated by the DPE module. To compute the loss,
   198	𝑖 . For the VIS embeddings, the loss can be computed
   220	where T𝑎 is a triplet loss function [10, 26], 𝑎𝑐 𝑖 ,𝑐 𝑗 denotes min 𝑑 (𝑐 𝑖 , 𝑐 𝑗 ), and 𝑝𝑐 𝑖 ,𝑐 𝑗 denotes max
   221	𝑑 (𝑐 𝑖 , 𝑐 𝑗 ). The loss function H (𝑐 𝑣 , 𝑐𝑖𝑟 , 𝑐 𝑖𝑣 ) is intended to maximize the difference between 𝑓𝑣 and
   233	𝑖 ). The overall MHC loss can then be
   234	embeddings of the IR, we define a loss function H (𝑐𝑖𝑟 , 𝑐 𝑣 , 𝑐𝑖𝑟
   257	the model, we propose an orthogonal Gramian metric loss (OGM loss). This loss function effectively
   271	triangular redundancy loss L𝑂𝐺𝑀 :
   290	triangular position. The orthogonal Gramian metric loss L𝑂𝐺𝑀 penalizes the nonorthogonality
   292	and extracting its upper triangular portion. Minimizing this loss encourages the embeddings to
   294	In addition to the proposed L𝑀𝐻𝐶 and L𝑂𝐺𝑀 , we also combine identity loss L𝑖𝑑 [56], and the
   295	total loss is formulated as follows:
   300	where 𝜆1 and 𝜆2 are coefficients controlling for the relative importance of the loss term.
   302	4.1
   304	Experiments
   315	Evaluation metrics. The experiments use the existing evaluation criteria of VI-ReID, and two
   316	metrics, cumulative matching characteristics (CMC) and mean average precision (mAP),
   317	are used to evaluate the performance of the method. In particular, CMC (rank-k matching accuracy)
   333	decays to 1 × 10−3 and 1 × 10−4 at epochs 80 and 120, respectively. For 𝑘 1 in the FRS module, we
   336	4.2
   347	Evaluation on the SYSU-MM01 dataset. The SYSU-MM01 dataset consists of both indoor and
   350	experiments are shown in Table 1. For the all-search mode, compared with the GGMMNet approach,
   361	Evaluation on the RegDB dataset. Since the RegDB dataset is relatively small, we use a 10-fold

exec
/bin/zsh -lc "pdftotext -f 9 -l 14 'Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '1,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification
     2	3.3
     3	
     4	101:9
     5	
     6	DPE Module
     7	
     8	In VI-ReID, an accurate understanding of image content relies heavily on features at different scales,
     9	as these features are crucial for capturing both contextual information and fine-grained details. Finescale features, such as edges and textures, are vital for recognizing fine details, whereas coarse-scale
    10	features provide insight into the overall structure of an image. Conventional convolution operations
    11	often struggle to flexibly capture effective features when faced with challenges such as complex
    12	occlusions, background shifts, and resolution discrepancies in VI-ReID tasks. Inspired by DEEN
    13	[53], we propose a DPE module. The DPE module introduces a novel approach to capture multi-scale
    14	spatial information through the use of deformable dilated convolution with varying dilation rates,
    15	allowing for enhanced feature extraction across diverse spatial resolutions. This design enables
    16	the model to dynamically perceive and adapt to complex scene variations, such as pose changes
    17	and occlusions, by capturing both fine-grained details and broad contextual cues. Additionally,
    18	the DPE module employs the MHC loss to refine feature embeddings, which strengthens crossmodal alignment by reducing modality disparities at multiple feature levels. This combination of
    19	deformable convolution and hierarchical contrast not only improves feature robustness but also
    20	ensures that critical identity cues are preserved and aligned across modalities, making the DPE
    21	module a powerful component for enhancing cross-modal representation in dynamic environments.
    22	As illustrated in Figure 2, our proposed DPE module generates multiple embeddings and focuses
    23	on more fine-grained features through an adaptive convolutional generation structure with multiple
    24	branches. Specifically, each branch of the DPE module utilizes three parallel 3 × 3 deformable dilated
    25	1 , D 2 , and D 3 , with dilation
    26	convolutional layers (illustrated in Figure 2(d)), denoted as D3×3
    27	3×3
    28	3×3
    29	ratios of 1, 2, and 3, respectively. These layers reduce the number of channels in the input feature
    30	map G to 1/4 of its original size and then combine them to form the output feature map. Each
    31	𝑗
    32	D3×3
    33	consists of two convolutional layers: one for computing offsets and the other for performing
    34	regular convolution. The offsets are computed via offset dilated convolution and are then applied in
    35	deformable dilated convolution operations in conjunction with the weights of the regular dilated
    36	𝑗
    37	convolution. The offset computation layer O3×3
    38	outputs 2 × 𝐾 2 channels, where K is the kernel
    39	𝑗
    40	size (in this case, 3), providing x and y offsets for each kernel position. The second layer (R 3×3
    41	)
    42	performs a regular dilation convolution operation on the basis of the offset. The convolutional layer
    43	weights are initialized via the Kaiming initialization method to maintain stable activation function
    44	variance throughout the network, whereas the bias term is initialized to zero. For a given input
    45	feature map G, the offset Δ𝑝 is calculated as follows:
    46	𝑗
    47	Δ𝑝 = O3×3
    48	(𝐺;𝑊𝑜 𝑓 𝑓 𝑠𝑒𝑡 , 𝑏𝑜 𝑓 𝑓 𝑠𝑒𝑡 ),
    49	
    50	(10)
    51	
    52	where 𝑊𝑜 𝑓 𝑓 𝑠𝑒𝑡 and 𝑏𝑜 𝑓 𝑓 𝑠𝑒𝑡 denote the weight and bias of the offset convolution layer, respectively.
    53	The output of deformable dilated convolution 𝑦 (𝑝 0 ) can subsequently be expressed as:
    54	𝑦 (𝑝 0 ) =
    55	
    56	𝐾2
    57	Õ
    58	
    59	Wk (𝑝𝑘 ) · BI (𝑝 0 + 𝑝𝑘 + Δ𝑝𝑘 ) + 𝑏,
    60	
    61	(11)
    62	
    63	𝑘=1
    64	
    65	where 𝑝 0 is the position of each point of the output feature map corresponding to the center of
    66	the convolution kernel, which is then mapped to the feature map; 𝑝𝑘 are the relative coordinates
    67	of 𝑝 0 corresponding to each position within the convolution kernel field; Δ𝑝𝑘 is the offset vector
    68	calculated by offset dilated convolution; 𝑝 0 + 𝑝𝑘 + Δ𝑝𝑘 is the position of the convolutional kernel
    69	field point corresponding to each point of the output feature map plus the learned offsets; Wk (𝑝𝑘 ) is
    70	the weight of the regular dilated convolution; b is the bias; and BI(p) refers to the input eigenvalue
    71	obtained by bilinear interpolation at position p to cope with the sampling problem posed by
    72	noninteger offsets.
    73	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.
    74	
    75	101:10
    76	
    77	S. Wang et al.
    78	
    79	Fig. 3. The proposed method FRS, primarily involves two summations: one for the original distances and
    80	another for the enhanced distances. This approach considers both the similarity metric between Query and
    81	Gallery, as well as the similarity metric among elements within the Gallery, which is a crucial feature. Through
    82	top-k feature enhancement, FRS addresses the noise issue stemming from information disparities, resulting
    83	ˆ Furthermore, FRS demonstrates significantly improved accuracy in
    84	in enhanced representations of 𝑄ˆ and 𝐺.
    85	the retrieve stage compared to the original method.
    86	
    87	A ReLU activation layer is then used to improve the nonlinear representation of the DPE. Next,
    88	another convolutional layer Conv1×1 with a kernel size of 1 × 1 is applied to the obtained feature
    89	map in order to change its size so that it has the same size as G. The DPE is then used to generate
    90	diverse embeddings. Thus, the generative embeddings 𝐺𝑖+ of the ith branch can be written in the
    91	following form:
    92	!!
    93	𝑛∈Õ
    94	(1,2,3)
    95	𝑗
    96	+
    97	𝐺𝑖 = Conv1×1 ReLU
    98	D3×3 (𝐺) .
    99	(12)
   100	𝑗=1
   101	
   102	Finally, the generated embeddings of all branches are joined together and used as inputs for the
   103	next stage.
   104	3.4
   105	
   106	FRS
   107	
   108	In VI-ReID, balancing the disparity between the training and testing phases poses a significant
   109	challenge because of the different amounts of data in each phase. During training, each batch
   110	contains an equal number of VIS and IR samples, ensuring that the model learns from balanced
   111	modal information. However, in the testing phase, this balance is disrupted: a single query may
   112	retrieve multiple gallery samples, resulting in an imbalance in modal information availability. This
   113	discrepancy can lead to notable performance degradation. Additionally, the similarity among gallery
   114	features influences the results. To address these challenges, we introduce feature reweighting on the
   115	basis of similarity (FRS). The FRS module uses advanced feature enhancement technology to solve
   116	the problem of uneven information distribution between modalities, thereby effectively improving
   117	the matching performance. The specific effects of the FRS are illustrated in Figure 3.
   118	FRS is a similarity-driven feature alignment method, with the core innovation lying in combining
   119	similarity calculations and feature fusion at multiple scales. By leveraging a multi-scale top-k [11]
   120	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.
   121	
   122	Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification
   123	
   124	101:11
   125	
   126	strategy and local averaging technique, this method enhances feature alignment and similaritybased feature refinement, similar in effect to the re-ranking [57] technique. Specifically, given a set
   127	of query features 𝑄 ∈ R𝑁𝑞 ×𝐶 and a set of gallery features 𝐺 ∈ R𝑁𝑔 ×𝐶 , we first normalize all feature
   128	vectors to eliminate scale differences. Next, we calculate the feature similarity matrix 𝑄 · 𝐺𝑇 to
   129	obtain M𝑞𝑔 ∈ R𝑁𝑞 ×𝑁𝑔 . We note that the similarity between features inside the gallery can assist in
   130	optimizing the matching performance, so we compute the feature similarity matrix 𝐺 · 𝐺𝑇 within
   131	the gallery to obtain M𝑔𝑔 ∈ R𝑁𝑔 ×𝑁𝑔 .
   132	Owing to the unequal amount of information in the query and gallery, the computation of
   133	the generated feature similarity matrix involves mean noise values, and these tiny noise values
   134	interfere with the retrieval results, leading to performance degradation and inaccurate results.
   135	To address this, we apply multi-scale feature aggregation by selecting the strongest similarity
   136	connections across multiple scales for each sample in M𝑞𝑔 and M𝑔𝑔 . Specifically, we use a set of
   137	scales {𝑘 1(1) , 𝑘 1(2) , . . . , 𝑘 1(𝑛) } for query-gallery connections, where at each scale, the top 𝑘 1(𝑖 ) similarity
   138	values are retained, and other values are set to zero. The resulting multi-scale similarity patterns
   139	are concatenated to create a more expressive representation 𝑄ˆ ∈ R𝑁𝑞 ×𝑁𝑔 .
   140	Similarly, within the gallery-to-gallery similarity matrix M𝑔𝑔 , we retain the strongest connections
   141	at multiple scales {𝑘 1(1) , 𝑘 1(2) , . . . , 𝑘 1(𝑛) }, and the highest 𝑘 2 values are selected for the weighted
   142	average to improve feature robustness and reduce informativeness inequality. This produces a
   143	new gallery feature representation 𝐺ˆ ∈ R𝑁𝑔 ×𝑁𝑔 , which better captures the dominant similarity
   144	patterns. This multi-scale feature aggregation thus results in a more comprehensive and stable
   145	feature alignment, significantly enhancing the retrieval accuracy. Finally, a composite final matrix
   146	S is constructed:
   147	S = 𝑄ˆ · 𝐺ˆ𝑇 + 𝑄 · 𝐺𝑇 ,
   148	
   149	(13)
   150	
   151	where S combines the original matrix 𝑄 · 𝐺𝑇 with the enhanced matrix 𝑄ˆ · 𝐺ˆ𝑇 . This strategy not
   152	only preserves the matching information in the original feature space, but also incorporates the
   153	enhanced similarity information, thus achieving a more robust and accurate similarity evaluation
   154	mechanism.
   155	3.5
   156	
   157	Multi-Loss Optimization
   158	
   159	In the VI-ReID domain, effectively fusing and comparing features from different modalities is
   160	crucial for improving model performance. To this end, our DPE module generates 𝐺𝑖+ embeddings
   161	to capture more effective multimodal features. As shown in Figure 4, it encourages samples from
   162	different modalities with different embeddings to pull close to each other in the feature space,
   163	while it encourages samples from the same modality with different embeddings to pull away from
   164	each other in the feature space. This mechanism facilitates the transfer and sharing of cross-modal
   165	information and makes effective use of diverse embeddings. Consequently, this approach enhances
   166	the model’s cross-modal generalization capability. By hierarchically constructing pairs of positive
   167	and negative samples, MHCLoss enables detailed feature optimization from the macroscopic intermodal level to the microscopic intra-modal level.
   168	The core design principle of MHC loss is adaptive feature alignment across modalities, which is
   169	achieved by generating multiple contrastive pairs within a hierarchical structure. At each hierarchical level, MHC loss promotes feature similarity between corresponding VIS and IR samples while
   170	maximizing the distance between different identities. This hierarchical structure allows the loss
   171	function to capture modality-specific details at lower levels and enforce cross-modal consistency
   172	at higher levels, effectively addressing the modality gap. The primary advantage of MHC loss is
   173	its ability to mitigate modality differences, especially in challenging conditions such as lighting
   174	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.
   175	
   176	101:12
   177	
   178	S. Wang et al.
   179	
   180	Fig. 4. The proposed loss function MHC, mainly primarily aims to minimize the distance between embeddings
   181	derived from different modalities, thereby promoting inter-modality alignment. Conversely, it seeks to
   182	maximize the distance between embeddings of the same modality when they are not closely aligned with
   183	their initial counterparts, fostering intra-modality contrast. Through this dual process, MHC encourages the
   184	generation of embeddings that are not only representative but also capture a broader spectrum of information
   185	across modalities.
   186	
   187	variations, pose changes, and occlusions. Operating across multiple levels of feature abstraction,
   188	MHC loss enables robust cross-modal feature alignment, enhancing the model’s adaptability to
   189	various environments. This multi-scale contrastive approach enhances the discriminative power of
   190	learned embeddings while increasing model robustness and accuracy, making it highly suitable for
   191	real-world VI-ReID applications.
   192	Specifically, the inputs to MHCLoss are the VIS and IR modality features corresponding to the
   193	two embeddings. The input feature vectors are partitioned into multiple modal and embedded
   194	feature representations. These multimodal features are grouped into categories: 𝑓𝑣 , 𝑓𝑖𝑟 ∈ 𝐺 and
   195	𝑓𝑣𝑖 , 𝑓𝑖𝑟𝑖 ∈ 𝐺𝑖+ , where 𝑓𝑣 and 𝑓𝑖𝑟 are the original embeddings of the VIS and IR modalities, respectively,
   196	and where 𝑓𝑣𝑖 and 𝑓𝑖𝑟𝑖 are the new embeddings generated by the DPE module. To compute the loss,
   197	we first calculate the mean of all sample features belonging to the same category to form the set of
   198	𝑖 . For the VIS embeddings, the loss can be computed
   199	category-centered feature vectors 𝑐 𝑣 , 𝑐𝑖𝑟 , 𝑐 𝑖𝑣 , 𝑐𝑖𝑟
   200	as follows:
   201	𝑝𝑐
   202	
   203	,𝑐 𝑖
   204	
   205	𝑝𝑐
   206	
   207	,𝑐 𝑖
   208	
   209	𝑝𝑐
   210	
   211	,𝑐 𝑖
   212	
   213	H (𝑐 𝑣 , 𝑐𝑖𝑟 , 𝑐 𝑖𝑣 ) = T𝑎𝑐 ,𝑐𝑖𝑟𝑖 𝑣 + T𝑎𝑐 𝑣 ,𝑐𝑖𝑟𝑣 𝑣 + T𝑎𝑐𝑖 ,𝑐𝑖𝑟𝑖 𝑣 ,
   214	𝑣 𝑣
   215	
   216	(14)
   217	
   218	𝑣 𝑣
   219	
   220	where T𝑎 is a triplet loss function [10, 26], 𝑎𝑐 𝑖 ,𝑐 𝑗 denotes min 𝑑 (𝑐 𝑖 , 𝑐 𝑗 ), and 𝑝𝑐 𝑖 ,𝑐 𝑗 denotes max
   221	𝑑 (𝑐 𝑖 , 𝑐 𝑗 ). The loss function H (𝑐 𝑣 , 𝑐𝑖𝑟 , 𝑐 𝑖𝑣 ) is intended to maximize the difference between 𝑓𝑣 and
   222	𝑓𝑣𝑖 to ensure that the generated new embeddings 𝑓𝑣𝑖 have independent features rather than simply copying the original embeddings. Minimizing the difference between 𝑓𝑖𝑟 and 𝑓𝑣𝑗 ensures that
   223	the generated new embeddings 𝑓𝑣𝑗 effectively capture the similarity of the different modal fea𝑝
   224	tures, which further enhances the cross-modal consistency and semantic relevance. T𝑎 can be
   225	specified as:
   226	𝑝
   227	
   228	T𝑝𝑎 = 𝑚𝑎𝑥 {𝑑 (𝑎, 𝑝) − 𝑑 (𝑎, 1) + 𝛼, 0},
   229	
   230	(15)
   231	
   232	where d(·, ·) denotes the Euclidean distance between two embeddings and where 𝛼 is a predetermined threshold to control the difference between positive and negative samples. Similarly, for the
   233	𝑖 ). The overall MHC loss can then be
   234	embeddings of the IR, we define a loss function H (𝑐𝑖𝑟 , 𝑐 𝑣 , 𝑐𝑖𝑟
   235	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.
   236	
   237	Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification
   238	
   239	101:13
   240	
   241	expressed as:
   242	L𝑀𝐻𝐶 =
   243	
   244	𝑁
   245	Õ
   246	
   247	𝑖
   248	(H (𝑐 𝑣 , 𝑐𝑖𝑟 , 𝑐 𝑖𝑣 ) + H (𝑐𝑖𝑟 , 𝑐 𝑣 , 𝑐𝑖𝑟
   249	)).
   250	
   251	(16)
   252	
   253	𝑖
   254	
   255	To further enhance the model’s ability to learn feature representations with more decoupling
   256	properties, optimize the interdependence between features, and improve the generalization ability of
   257	the model, we propose an orthogonal Gramian metric loss (OGM loss). This loss function effectively
   258	quantifies and penalizes the nonorthogonality between the generated embedded features.
   259	Specifically, for the generated embeddings 𝑥𝑝𝑖+ ∈ R𝐵×𝐷 ×𝐶 , we first perform two-by-two splicing
   260	ˆ ∈ R𝐵×2𝐷 ×𝐶 , where B denotes the batch
   261	along the channel dimension to form a new tensor 𝑥𝑝
   262	size, D denotes the dimensionality of the feature, and C denotes the number of channels. Next,
   263	ˆ with its transpose 𝑥𝑝
   264	ˆ 𝑇 to obtain the Gramian matrix
   265	we compute the batch matrix product of 𝑥𝑝
   266	𝐵×2𝐷
   267	×2𝐷
   268	M∈R
   269	. Finally, we extract the upper triangular part of the Gramian matrix M, excluding
   270	the diagonal, and denote it as U. The sum of the elements in U is calculated to form the upper
   271	triangular redundancy loss L𝑂𝐺𝑀 :
   272	𝐵
   273	
   274	L𝑂𝐺𝑀 =
   275	
   276	2𝐷
   277	
   278	2𝐷
   279	
   280	1 ÕÕ Õ
   281	U𝑏,𝑖,𝑗 ,
   282	𝐵
   283	𝑖 𝑗=𝑖+1
   284	
   285	(17)
   286	
   287	𝑏=1
   288	
   289	where U𝑏,𝑖,𝑗 denotes the element of the Gramian matrix M for the bth sample that lies in the upper
   290	triangular position. The orthogonal Gramian metric loss L𝑂𝐺𝑀 penalizes the nonorthogonality
   291	between the feature vectors by computing the Gramian matrix of the spliced embedded features
   292	and extracting its upper triangular portion. Minimizing this loss encourages the embeddings to
   293	become more diverse and less redundant.
   294	In addition to the proposed L𝑀𝐻𝐶 and L𝑂𝐺𝑀 , we also combine identity loss L𝑖𝑑 [56], and the
   295	total loss is formulated as follows:
   296	L𝑡𝑜𝑡𝑎𝑙 = L𝑖𝑑 + 𝜆1 L𝑀𝐻𝐶 + 𝜆2 L𝑂𝐺𝑀 ,
   297	
   298	(18)
   299	
   300	where 𝜆1 and 𝜆2 are coefficients controlling for the relative importance of the loss term.
   301	4
   302	4.1
   303	
   304	Experiments
   305	Datasets and Settings
   306	
   307	Datasets. Two publicly available VI-ReID datasets were used to evaluate our model: SYSU-MM01
   308	[38] and RegDB [23]. The SYSU-MM01 dataset includes images captured by six different cameras,
   309	comprising two IR cameras and four VIS cameras. It features a total of 491 valid identities, with
   310	296 identities designated for training, 99 for verification, and 96 for testing. During the training
   311	phase, images from all the cameras are utilized for the 296 training identities. The RegDB dataset
   312	contains 412 valid identities, each represented by ten VIS images and ten IR images. This dataset is
   313	divided equally into training and testing sets, each consisting of 206 identities. The performance is
   314	evaluated separately for querying IR images with VIS images.
   315	Evaluation metrics. The experiments use the existing evaluation criteria of VI-ReID, and two
   316	metrics, cumulative matching characteristics (CMC) and mean average precision (mAP),
   317	are used to evaluate the performance of the method. In particular, CMC (rank-k matching accuracy)
   318	indicates the probability of having a correct match in the top-k retrieval results, and mAP measures
   319	the average retrieval performance when there are multiple correct matches.
   320	Implementation details. Our MDFN framework is implemented via the PyTorch framework,
   321	employs ResNet-50 as a backbone, and runs on a single Tesla V100 GPU. During the training phase,
   322	all the images were adjusted to 384 × 144 for data enhancement via horizontal flipping, random
   323	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.
   324	
   325	101:14
   326	
   327	S. Wang et al.
   328	
   329	erasure and channel enhancement. For each batch size, we randomly selected 6 identities, where
   330	4 VIS images and 4 IR images were selected for each identity. We adopt a learning rate warm-up
   331	strategy; specifically, the initial learning rate is set to 1 × 10−2 and then gradually increases to
   332	1 × 10−1 after 10 epochs. After that, the learning rate decays to 1 × 10−2 at the 20th epoch and further
   333	decays to 1 × 10−3 and 1 × 10−4 at epochs 80 and 120, respectively. For 𝑘 1 in the FRS module, we
   334	choose the values {4, 8} and 𝑘 2 to be 2. We use SGD as the optimizer with the momentum parameter
   335	set to 0.9. The total number of training epochs is set to 150.
   336	4.2
   337	
   338	Comparison with State-of-the-Art Methods
   339	
   340	We evaluate our MDFN framework on SYSU-MM01 and RegDB. Also, our MDFN is compared with
   341	many state-of-the-art (SOTA) methods, including image-level cmGAN [2], D2 RL [36], AlignGAN
   342	[34], JSIA-ReID [35], X-Modality [16], FMCNet [52], PMT [21], ACD-Intra [25], GGMMNet [45]
   343	and feature-level Zero-Padding [38], BDTR [51], DDAG [48], cm-SSFT [22], AGW [49], SMCL [37],
   344	MPANet [42], MMN [54], SPOT [1], DCLNet [29], ReCF [9], DPIS [28], DGRNet [15], CM-EMD
   345	[18], DEEN [53], CIIM [46], PMWGCN [30], LCNL [47], TMAL [4], MIP [43], FFDML [17], TRR
   346	[55], and the experimental results are shown in Table 1.
   347	Evaluation on the SYSU-MM01 dataset. The SYSU-MM01 dataset consists of both indoor and
   348	outdoor data, so we use both all-search and indoor-search strategies to evaluate the performance of
   349	our approach and state-of-the-art methods on the SYSU-MM01 dataset. The results of the comparison
   350	experiments are shown in Table 1. For the all-search mode, compared with the GGMMNet approach,
   351	while our proposed MDFN performs on par with GGMMNet in Rank-1, we achieve a significant
   352	mAP improvement of 4.92%. Furthermore, MDFN outperforms approaches such as DEEN and
   353	TMAL, surpassing DEEN by 0.40% in Rank-1 and 4.42% in mAP, and DGRNet by 4.52% in Rank-1
   354	and 2.26% in mAP. For the indoor-search mode, MDFN provides impressive results with a Rank-1
   355	accuracy of 81.46% and an mAP of 85.45%. Compared with DEEN, MDFN improves Rank-1 by 1.16%
   356	and mAP by 2.15%, and outperforms TRR by 0.14% and 2.23% in Rank-1 and mAP, respectively. These
   357	results demonstrate that our MDFN approach offers significant improvements over other state-ofthe-art approaches. In summary, the proposed MDFN method achieves significant improvements
   358	over existing methods and provides the best performance in terms of Rank-1 accuracy and mAP
   359	under both the all-search and indoor-search protocols, highlighting its effectiveness in real-world
   360	surveillance scenarios.
   361	Evaluation on the RegDB dataset. Since the RegDB dataset is relatively small, we use a 10-fold
   362	cross-validation method to evaluate the performance of MDFN in both VI and IR-VIS test modes.
   363	The comparison results on the RegDB dataset are presented in Table 1. In the VIS-to-IR mode, the
   364	MDFN achieves a Rank-1 accuracy of 91.28% and an mAP of 92.15%. In the IR-to-VIS mode, the
   365	Rank-1 accuracy is 90.03% and the mAP is 91.20%. Although the MDFN’s Rank-1 accuracy in both
   366	modes is slightly lower than that of state-of-the-art methods such as GGMMNet and CM-EMD, this
   367	performance gap may be attributed to the smaller scale and lower complexity of the RegDB dataset.
   368	Specifically, RegDB lacks the complex dynamics found in larger datasets, such as occlusions, diverse
   369	poses, and significant environmental changes. As a result, the limited variability in the dataset may
   370	not sufficiently challenge the model to capture real-world scene diversity, leading to incomplete
   371	training of the model’s learnable prototypes. This explains the slight drop in performance compared
   372	with methods evaluated on more diverse and challenging datasets.
   373	Among the methods we compared, DEEN, FFDML, and TRR all employ feature-level methods,
   374	which focus on learning shared feature representations between modalities to address the modality
   375	gap. These methods aim to extract discriminative features at the feature level but face challenges in
   376	dynamic environments, particularly when handling issues such as lighting variations, occlusions,
   377	and pose changes. In contrast, our method effectively incorporates multi-scale feature fusion and
   378	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.
   379	
   380	
codex
第二篇的卖点是一个典型的“动态、多尺度、融合”组合稿，但它真正把观察升级的地方在测试期信息不平衡：训练时可成对平衡，测试时一个 query 对多个 gallery，query-gallery 和 gallery-gallery 的结构都参与匹配，所以它把一个后处理式相似度增强也包装进方法闭环。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '274,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   274	proposed Mutual Distillation driven Dual-space Matching
   275	(MDDM) framework. Firstly, we present the overall architecture in Section III-A. Secondly, the proposed Dual Level Fusion (DLF) is described in Section III-B. Thirdly, Section III-C
   276	and Section III-D introduce the Modality Invariant Augmentation (MIA) module and the Dual-Space Matching (DSM),
   277	respectively. Finally, the optimization scheme is discussed in
   278	Section III-E.
   279	
   280	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:25 UTC from IEEE Xplore. Restrictions apply.
   281	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   282	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   283	
   284	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   285	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3682328
   286	
   287	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
   288	
   289	4
   290	
   291	Fig. 2: Overview of the proposed framework for VIReID. (a) The Mutual Distillation driven Dual-space Matching (MDDM)
   292	framework includes the dual-level fusion and modality invariant augmentation modules to address modality discrepancies. (b)
   293	The DLF module integrates low- and high-level semantics across spatial and channel dimensions for modality alignment. (c)
   294	The MIA module captures fine-grained semantics to enhance identity discriminability and facilitate visible-infrared modality
   295	alignment.
   296	
   297	A. Overall Architecture
   298	
   299	B. Dual Level Fusion Module
   300	
   301	An overview of the proposed MDDM framework is illustrated in Fig. 2, which mainly consists of three components:
   302	input, feature extraction, and dual-space metric learning. For
   303	the input, we denote the training image from modality t as
   304	xt , where t ∈ {v, r}, and v and r represent the visible
   305	and infrared modalities, respectively. The visible and infrared
   306	v
   307	samples in the dataset are denoted as V = {xvi , yiv }N
   308	i=0 and
   309	r r Nr
   310	t
   311	R = {xi , yi }i=0 , respectively, where N denotes the number
   312	of samples in modality t, and yit is the identity label of the
   313	i-th sample from modality t.
   314	In the feature extraction stage, we adopt the pre-trained
   315	ResNet-50 [60] as the backbone network. Two specific ResBlocks are employed to extract low-level modality semantic
   316	features, and the DLF is incorporated after layer1 and layer2
   317	to capture and adaptively integrate hierarchical semantics
   318	containing more Modality Invariant features. Subsequently, a
   319	shared ResBlock is applied to learn high-level modality-shared
   320	features from both modalities. The proposed MIA module is
   321	integrated after layer3 and layer4 to obtain enhanced modalityshared representations.
   322	In the metric learning stage, MDDM employs dual-space
   323	matching with a mutual distillation strategy, which encourages
   324	spatial consistency while reducing modality-specific discrepancies.
   325	
   326	Low-level semantic features mainly capture image details
   327	and local structures, which help the model recognize textures,
   328	edges, and other fundamental patterns. In contrast, highlevel semantic features provide global context and abstract
   329	representations, enabling comprehension of more complex
   330	patterns and scene structures. To fully exploit the complementary information from different semantic levels and enhance
   331	feature stability, we propose the DLF module, which integrates
   332	low-level details with high-level semantics through crosslevel fusion, achieving a balance between fine-grained detail
   333	and holistic semantic understanding, thereby improving both
   334	robustness and overall performance.
   335	In each stage of the backbone network, we consider two
   336	types of source features. Taking the DLF module after layer1
   337	as an example, the inputs are defined as the feature map before
   338	layer1, Flt ∈ RC×H×W , and the feature map after layer1,
   339	Fht ∈ RC×H×W , where C, H, and W denote the number
   340	of channels, width, and height, respectively. Here, l and h
   341	indicate low-level and high-level semantic features. As shown
   342	in Fig. 2 (b), the first DLF module takes Flt and Fht as input.
   343	Similarly, in the second DLF module, the low-level feature
   344	remains Flt , while the high-level feature is replaced with the
   345	t
   346	output of the first DLF module, denoted as Fm
   347	. The process
   348	
   349	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:25 UTC from IEEE Xplore. Restrictions apply.
   350	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   351	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   352	
   353	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   354	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3682328
   355	
   356	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
   357	
   358	can be formulated as
   359	Fnt = layer1(Flt ),
   360	
   361	(1)
   362	
   363	t
   364	t
   365	Fm
   366	= DLF1 (Flt , Fnt ), F̂ t = DLF2 (Flt , Fm
   367	).
   368	
   369	(2)
   370	
   371	We first employ an attention mechanism to fuse low-level
   372	and high-level semantics. Specifically, three 1 × 1 convolutional layers, Q1 , K1 , and V1 , are applied to generate
   373	the corresponding embeddings. The channel-wise similarity
   374	matrix S t ∈ RC×C is then computed by matrix multiplication
   375	followed by the softmax function
   376	S t = Softmax(Q1 (Fht ) × (K1 (Flt ))T ).
   377	
   378	(3)
   379	
   380	Next, we leverage Flt again to restore the channel dimension
   381	of S t , thereby enabling cross-level feature fusion. A 1 × 1
   382	convolutional layer θ1×1 is further applied to adjust the feature
   383	dimensions. To enhance flexibility, we introduce a gating
   384	mechanism, and the output of the first DLF module can be
   385	expressed as
   386	W1 = S t ⊙ V1 (Flt ),
   387	(4)
   388	t
   389	Fm
   390	= σ(W1 ) ⊙ W1 + (1 − σ(W1 )) ⊙ Fht ,
   391	
   392	(5)
   393	
   394	where W1 denotes the adjusted feature map, σ is the sigmoid
   395	t
   396	represents the intermediate fused features.
   397	function, and Fm
   398	t
   399	with Flt to
   400	Following the same procedure, we fuse Fm
   401	obtain the output of the second DLF module, namely the
   402	enhanced feature F̂ t , which integrates both low- and highlevel semantics
   403	F̂
   404	
   405	t
   406	
   407	t
   408	= σ(W2 ) ⊙ W2 + (1 − σ(W2 )) ⊙ Fm
   409	.
   410	
   411	(6)
   412	
   413	C. Modality Invariant Augmentation Module
   414	While DLF leverages cross-level semantic fusion to mitigate feature inconsistencies, visible–infrared person reidentification still faces significant fine-grained modality discrepancies, such as variations in texture, shape, and local
   415	structural information. The proposed Modality Invariant Augmentation (MIA) module, as shown in Fig. 2 (c), is designed
   416	to capture fine-grained semantic cues and ensure the discriminative power of each identity sample, thereby facilitating a
   417	robust association between visible and infrared modalities.
   418	Consequently, this enhances the discriminative capacity of the
   419	modality-shared representations.
   420	Specifically, given the input feature map F t ∈ RC×H×W
   421	from the visible-infrared pair, we first perform global average pooling (GAP) operations along both the horizontal and
   422	vertical directions to fuse modality-shared information. This
   423	operation can be expressed as
   424	Fwt = GAPx (F t ),
   425	
   426	Fht = GAPy (F t ),
   427	
   428	(7)
   429	
   430	where GAPx (·) and GAPy (·) denote the global average
   431	pooling operations with kernels of size (h, 1) and (1, w),
   432	respectively. Fwt aggregates information along the rows, capturing detailed semantic features, while Fht fuses information
   433	along the columns, aiding in the understanding of pedestrian
   434	positions within the image. This mechanism mitigates the
   435	impact of pose variations and noise introduced by different
   436	cameras, viewpoints, and modalities.
   437	
   438	5
   439	
   440	Subsequently, the modality-specific information in Fwt and
   441	t
   442	Fh is converted into distributions ranging from 0 to 1 using the
   443	softmax function. By multiplying the distributions from both
   444	directions, we obtain the modality-specific distribution mask
   445	M t , which is defined as
   446	M t = σ(Fht ) ⊗ σ(Fwt ),
   447	
   448	(8)
   449	
   450	where M t ∈ RC×H×W is the modality-specific distribution
   451	mask, with each value incorporating fine-grained semantic
   452	information. This mask reduces the impact of noise and
   453	highlights the significance of modality-specific information.
   454	Furthermore, cross-modal contextual information plays a
   455	pivotal role in the VI-ReID task, especially in accurately
   456	aligning visible and infrared images. Given the substantial
   457	differences in the visual information presented by these modalities in terms of shape and features, leveraging cross-modal
   458	contextual information allows the model to better understand
   459	these discrepancies and achieve more precise alignment. We
   460	employ large convolution decompositions to aid the module in
   461	effectively processing multi-scale contextual features. By decomposing a large convolution operation into smaller kernels
   462	with varying dilation rates, we efficiently capture a broader
   463	receptive field. Specifically, we use two convolution layers
   464	Conv1 and Conv2 with kernel sizes 5×5 and 7×7, and dilation
   465	rates 1 and 3, respectively, to simulate a larger convolution
   466	kernel of size 23 × 23. This is expressed as
   467	F1t = Conv1(F t ),
   468	
   469	F2t = Conv2(F1t ),
   470	
   471	(9)
   472	
   473	where F1t ∈ R1×H×W and F2t ∈ R1×H×W are feature
   474	representations with different receptive fields.
   475	To focus on the most relevant spatial regions, we concatenate the features from different scales along the channel
   476	dimension. Then, we apply global average pooling and global
   477	max pooling to compress the global channel information, i.e.,
   478	t
   479	FM
   480	= MaxPool(Concat(F1t , F2t )),
   481	
   482	(10)
   483	
   484	FAt = AvgPool(Concat(F1t , F2t )),
   485	
   486	(11)
   487	
   488	t
   489	where FM
   490	∈ RC×H×W and FAt ∈ RC×H×W are the maxpooled and average-pooled spatial descriptions, respectively,
   491	enabling information exchange across different spatial scales.
   492	Next, we concatenate the two spatial descriptions along the
   493	channel dimension and apply a convolutional layer to recover
   494	the channel dimension from 2 to C. The resulting spatial
   495	attention map is then applied with the sigmoid function to
   496	t
   497	obtain the weight representation for each scale feature FM
   498	Ai :
   499	
   500	t
   501	t
   502	t
   503	FM Ai = σ Conv Concat FM , FA
   504	, i ∈ {1, 2}. (12)
   505	
   506	The weight representation is multiplied with the corresponding feature maps, and the attention features are obtained as
   507	t
   508	t
   509	t
   510	At = F1t ⊗ FM
   511	A1 + F2 ⊗ FM A2 ,
   512	
   513	(13)
   514	
   515	where At ∈ RC×H×W denotes the attention features fused
   516	from different scales, and ⊗ denotes element-wise multiplication.
   517	Finally, the distribution mask, which integrates vertical
   518	position and horizontal semantic information, is fused with the
   519	
   520	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:25 UTC from IEEE Xplore. Restrictions apply.
   521	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   522	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   523	
   524	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   525	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3682328
   526	
   527	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
   528	
   529	6
   530	
   531	1) Hilbert Space Matching Loss: First, the modality-shared
   532	knowledge captured by the MIA module is aggregated using
   533	both Generalized Mean Pooling (GeMP) and Global Average
   534	Pooling (GAP) to obtain f t and f˜t , denoted as
   535	
   536	t
   537	
   538	f t = GeMP(F̃ t ),
   539	
   540	t ∈ {v, r},
   541	
   542	(15)
   543	
   544	f˜t = GAP(F̃ t ),
   545	
   546	t ∈ {v, r},
   547	
   548	(16)
   549	
   550	B×C
   551	
   552	where f ∈ R
   553	represents high-dimensional features, and
   554	f˜t ∈ RB×C represents relatively low-dimensional features.
   555	Based on f t and f˜t , we design an inter-class loss (Linter ) to
   556	alleviate the inter-class discrepancy between the two modalities, defined as
   557	Linter = HSIC(f v , f r ) + HSIC(f˜v , f˜r ),
   558	Fig. 3: The proposed Hilbert Space Matching loss. Different
   559	colors represent different feature spaces. The triangle and
   560	circle symbols represent the feature embeddings of visible and
   561	infrared images, respectively.
   562	multi-scale attention features to obtain the Modality Invariant
   563	enhanced feature, expressed as
   564	F̃ t = M t + At + F t .
   565	
   566	(14)
   567	
   568	(17)
   569	
   570	where HSIC(·, ·) denotes the hilbert-schmidt independence
   571	criterion loss [62], which is formulated as
   572	m
   573	X
   574	1
   575	(
   576	Φ(fiv , fjv ) Φ(fir , fjr )
   577	HSIC(f v , f r ) =
   578	(m − 1)2 i,j=1
   579	m
   580	m
   581	m
   582	X
   583	2 XX
   584	v
   585	v
   586	(
   587	Φ(fi , fj ))(
   588	Φ(fir , fjr ))
   589	−
   590	m i=1 j=1
   591	j=1
   592	m
   593	m
   594	X
   595	1 X
   596	v
   597	v
   598	(
   599	Φ(f
   600	,
   601	f
   602	))(
   603	Φ(fir , fjr ))),
   604	i
   605	j
   606	m2 i,j=1
   607	i,j=1
   608	(18)
   609	where fiv and fir denote the visible and infrared feature vectors
   610	of the i-th sample, m is the number of visible and infrared
   611	features in the current batch. Φ(·, ·) is the Gaussian kernel that
   612	maps features into Hilbert space.
   613	Additionally, we consider an intra-class loss (Lintra ) to
   614	address intra-class variations, defined as
   615	
   616	+
   617	
   618	D. Dual-Space Matching
   619	The infinite nature of the latent feature space makes it
   620	difficult to define clear boundaries for sample distributions,
   621	complicating the alignment of visible and infrared modalities.
   622	Traditional methods [47], [48], [51], [52] typically employ
   623	Euclidean distance for measurement. Euclidean space constraints preserve local discriminative features by minimizing
   624	intra-class distances while maximizing inter-class distances,
   625	which helps distinguish fine-grained identity differences, but
   626	they may fail to capture global structural relationships across
   627	modalities. To complement this limitation, some studies [19],
   628	[61] project features into Hilbert space. Hilbert space constraints model the statistical dependence of cross-modal features in a high-dimensional kernel space, ensuring that shared
   629	features across modalities remain consistent at the global
   630	distribution level. Such a strategy not only preserves local
   631	details but also reinforces global semantic consistency, effectively mitigating deep-layer semantic misalignment and
   632	cross-modality discrepancies. Inspired by these insights, we
   633	propose the Dual-Space Matching (DSM) strategy, which
   634	projects features into both Euclidean and Hilbert spaces and
   635	aligns the shared features consistent in both spaces via mutual
   636	distillation. The proposed method simultaneously maintains
   637	local discriminative power and captures global statistical dependencies, significantly improving the robustness of crossmodality feature alignment.
   638	DSM consists of two components: Hilbert Space Matching (HSM) loss and Euclidean Space Matching (ESM) loss.
   639	Lhsm aligns features in the Hilbert space for better crossmodal alignment, while Lesm applies mutual distillation for bidirectional knowledge transfer, enhancing feature consistency
   640	and classification performance.
   641	
   642	Lintra = HSIC(f v , f˜v ) + HSIC(f r , f˜r ).
   643	
   644	(19)
   645	
   646	As shown in Fig. 3, the formulation of both inter-class and
   647	intra-class losses contributes to the overall HSM loss. Finally,
   648	HSM loss is formulated as:
   649	Lhsm = αLinter + Lintra ,
   650	
   651	(20)
   652	
   653	where α is hyperparameter that balance inter-class and intraclass losses. The proposed Lhsm accounts for both cross-modal
   654	and intra-modal discrepancies, and leverages HSIC to mitigate
   655	misalignment, thereby effectively reducing the cross-modal
   656	gap.
   657	2) Euclidean Space Matching Loss: In cross-modality retrieval tasks, there often exists a significant discrepancy between the feature distributions of different modalities, leading
   658	to inconsistent projections of visible and infrared features
   659	under the same semantic context. To mitigate this modality
   660	mismatch, we propose a Euclidean Space Matching loss
   661	(Lesm ). Within a unified framework, this loss implements
   662	Mutual Distillation in the pixel-level feature space to enforce
   663	modality-consistent constraints, thereby enhancing both the
   664	discriminability and consistency of cross-modality representations. Moreover, it is seamlessly integrated with Identity
   665	Supervision, ensuring not only feature alignment but also
   666	category discrimination.
   667	
   668	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:25 UTC from IEEE Xplore. Restrictions apply.
   669	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   670	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   671	
   672	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   673	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3682328
   674	
   675	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
   676	
   677	7
   678	
   679	the prediction space, achieving adaptive alignment across
   680	modalities.
   681	Finally, the ESM loss is defined as
   682	Lesm = Lkl1 + Lkl2 .
   683	
   684	(27)
   685	
   686	To preserve identity discriminability, we further incorporate
   687	an identity classification loss
   688	Lid = CE(Pt , y t ) + CE(P̂t , y t ),
   689	Fig. 4: The proposed Euclidean Space Matching loss. Different
   690	colors represent different feature spaces. The triangle and
   691	circle symbols represent the feature embeddings of visible and
   692	infrared images, respectively.
   693	
   694	As shown in Fig. 4, given the fine-grained shared features from the visible and infrared modalities, denoted as
   695	At = {At1 , At2 , · · · , AtB } and Ât = {Ât1 , Ât2 , · · · , ÂtB }, where
   696	Ati and Âti represent the i-th shared feature in a batch of
   697	size B. Correspondingly, we construct two modality-specific
   698	classifiers that are updated via Exponential Moving Average
   699	(EMA) to predict the opposite modality. The update rules are
   700	formulated as
   701	[Φv r ]k = (1 − τ )[Φv r ]k−1 + τ [Φv ]k ,
   702	
   703	(21)
   704	
   705	[Φr v ]k = (1 − τ )[Φr v ]k−1 + τ [Φr ]k ,
   706	
   707	(22)
   708	
   709	where [Φv r ]k and [Φr v ]k denote the parameters of the crossmodality teacher classifiers at the k-th iteration, while Φv
   710	and Φr represent the corresponding student classifiers. The
   711	parameter τ is the update ratio, set to 0.2 in the paper. In the
   712	mutual distillation process, the EMA update strategy smooths
   713	the parameter changes of the teacher classifiers, providing
   714	stable and consistent supervision for the student. Additionally,
   715	since the teachers for the visible and infrared modalities are
   716	updated separately, EMA helps prevent them from drifting in
   717	different directions, thereby avoiding training divergence and
   718	ensuring more robust cross-modality distillation.
   719	For the visible and infrared features, we first obtain their
   720	predicted probability distributions through their respective
   721	student and teacher classifiers
   722	Pt = Softmax(concat(Φv (Av ), Φr (Ar ))),
   723	
   724	(23)
   725	
   726	Pt′ = Softmax(concat(Φr v (Av ), Φv r (Ar ))),
   727	
   728	(24)
   729	
   730	where Pt and Pt′ are the probability distributions of At after
   731	mutual learning between the two classifiers. Similarly, we can
   732	obtain P̂t and P̂t′ for the corresponding modality.
   733	To enforce cross-modality consistency, we adopt a mutual
   734	distillation scheme based on bidirectional Kullback–Leibler
   735	(KL) divergence, formulated as
   736	Lkl1 = DKL (Pt ∥Pt′ ) + DKL (P̂t ∥P̂t′ ),
   737	
   738	(25)
   739	
   740	Lkl2 = DKL (Pt′ ∥Pt ) + DKL (P̂t′ ∥P̂t ),
   741	
   742	(26)
   743	
   744	where DKL (·, ·) denotes the KL divergence, which measures
   745	the information discrepancy between two probability distributions. Through this bidirectional constraint, the student and
   746	teacher classifiers distill knowledge from each other within
   747	
   748	(28)
   749	
   750	where CE(·) denotes the cross-entropy loss, and y t represents
   751	the identity labels corresponding to each modality.
   752	E. Overall Objective Function
   753	In addition to the HSM loss (Lhsm ) and the ESM loss
   754	(Lesm ), both the cross-entropy loss (Lid ) and the triplet loss
   755	(Ltri ) are employed to jointly optimize the proposed MDDM
   756	framework. The overall training objective of MDDM can thus
   757	be formulated as
   758	Lall = Lid + Ltri + Lhsm + Lesm .
   759	
   760	(29)

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-year long-term person re-identification using gait and HAR features.pdf' - | nl -ba | sed -n '1,620p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	Pattern Recognition 172 (2026) 112627
     2	
     3	Contents lists available at ScienceDirect
     4	
     5	Pattern Recognition
     6	journal homepage: www.elsevier.com/locate/pr
     7	
     8	Multi-year long-term person re-identiﬁcation using gait and HAR features
     9	David Freire-Obregón ∗, Oliverio J. Santana , Javier Lorenzo-Navarro
    10	Daniel Hernández-Sosa, Modesto Castrillón-Santana
    11	
    12	,
    13	
    14	Universidad de Las Palmas de Gran Canaria, Institute of Intelligent Systems and Numeric Applications in Engineering, Las Palmas de Gran Canaria, 35007, Spain
    15	
    16	a r t i c l e
    17	
    18	i n f o
    19	
    20	Keywords:
    21	Person re-identiﬁcation
    22	Biometrics
    23	Gait
    24	Human action recognition
    25	
    26	a b s t r a c t
    27	We propose a two-stream person re-identiﬁcation (Re-ID) framework that integrates gait and human action recognition (HAR) through cross-attention fusion. The model processes gait sequences via a BiLSTM-based encoder to
    28	capture temporal motion dynamics. At the same time, HAR embeddings are extracted using pre-trained video
    29	backbones and distilled into compact behavioral features. These two modalities are fused using a cross-attention
    30	mechanism, enriching gait-based identity representations with context-aware activity cues. We evaluate our
    31	method on a newly curated long-term spatio-temporal dataset of ultra-distance runners captured in natural outdoor settings across multiple locations spanning three years (2020 to 2023). Experimental results demonstrate
    32	that integrating HAR signiﬁcantly enhances gait-based Re-ID performance. Compared to gait-only models, our approach yields a 12 % improvement in mean Average Precision (mAP) in cross-year scenarios and up to an 11.6 %
    33	gain in same-year evaluations. The HAR-enhanced models also exhibit faster convergence and higher Rank-1
    34	accuracy, establishing the eﬀectiveness of multi-modal motion-based representations for long-term, real-world
    35	person Re-ID.
    36	
    37	1. Introduction
    38	Humans can recognize familiar individuals across diverse contexts
    39	and times. Biometrics automates this process using physical, chemical,
    40	or behavioral traits [1]. Typically, biometric veriﬁcation systems assume a known gallery of identities created during registration. In contrast, when identities are not pre-registered, the task shifts to linking
    41	observations of the same individual across time and space, regardless
    42	of their true name. This problem, known as person re-identiﬁcation
    43	(Re-ID), involves retrieving an individual across diﬀerent cameras or
    44	time spans [2]. In this work, we focus on supervised Re-ID, where labeled identities guide the learning of embeddings. Most research emphasizes short-term image-based Re-ID, where appearance remains stable. Real deployments, however, require long-term Re-ID, where clothing and visual cues change over time. Video provides richer temporal
    45	information, with gait serving as a stable biometric foundation. Nevertheless, benchmarks for long-term, video-based Re-ID remain scarce.
    46	DeepChange [3] is among the few, though its imbalance in identity frequency poses challenges, and its use in video-based scenarios is limited.
    47	Long-term Re-ID in crowded, dynamic environments is particularly difﬁcult: individuals may look alike, appear brieﬂy, or undergo occlusions,
    48	motion blur, and domain shifts. Appearance-based cues often fail under
    49	such conditions. Motion signals such as gait, together with human ac-
    50	
    51	tion recognition (HAR), oﬀer complementary stability and behavioral
    52	context. Despite this, they are rarely combined in trainable, end-to-end
    53	systems.
    54	Our main contributions are as follows. First, we propose a twostream architecture that combines gait and HAR features for person ReID. A gait dynamics is learned in a structured way by a BiLSTM branch,
    55	and a light-weight and well-cited baseline in video recognition is obtained by the HAR branch, producing compact clip-level embeddings by
    56	global-temporal average pooling, in order to keep the auxiliary branch
    57	light-weight. Then, we incorporate a cross-attention scheme enabling interactive mixing of the two feature streams, allowing for a ﬂexible combination of activity and motion cues. We utilized a triplet loss function
    58	to uncover discriminative and generalizable representations. Second, we
    59	introduce a new dataset collected in unconstrained environments, where
    60	ultra-distance race participants were recorded at two locations in 2020
    61	and at two additional locations in 2023. This design facilitates evaluation under both short-term conditions (same-day/within-year) and longterm conditions (across multiple years), thereby capturing realistic variations in appearance, viewpoint, and environment.
    62	Experiments show that fusing gait and HAR outperforms singlestream and visual-only baselines, yielding more robust embeddings. Results highlight the potential of motion-based multimodal representations
    63	for Re-ID in unconstrained conditions.
    64	
    65	∗ Corresponding author.
    66	
    67	E-mail address: david.freire@ulpgc.es (D. Freire-Obregón).
    68	https://doi.org/10.1016/j.patcog.2025.112627
    69	Received 2 June 2025; Received in revised form 12 September 2025; Accepted 18 October 2025
    70	Available online 23 October 2025
    71	0031-3203/© 2025 The Author(s). Published by Elsevier Ltd. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).
    72	
    73	Pattern Recognition 172 (2026) 112627
    74	
    75	D. Freire-Obregón et al.
    76	
    77	Fig. 1. Pipeline of the proposed two-stream architecture. Gait and HAR features are extracted independently, aligned in dimension, and fused via a cross-attention
    78	mechanism where gait attends to HAR. The fused embedding is used for identity representation. This design prioritizes gait as the primary cue while enriching it
    79	with complementary HAR context.
    80	
    81	2. Related work
    82	
    83	26,000 subjects in unconstrained environments. Other datasets, such
    84	as FVG [17], contribute to advancing cross-view and in-the-wild gait
    85	recognition by oﬀering high-quality visual gait data under naturalistic
    86	conditions.
    87	However, these datasets typically lack long-term variability, as they
    88	do not include recordings of the same individuals across multiple years.
    89	In contrast, one of our key contributions is a new dataset comprising
    90	real-world race footage recorded over three years (2020 and 2023) and
    91	from diﬀerent distant locations within the same year. This enables the
    92	evaluation of gait-based Re-ID under realistic conditions of appearance
    93	change, long-term temporal gaps, and spatial variability, a scenario not
    94	addressed in existing benchmarks.
    95	
    96	Person Re-ID has traditionally focused on visual appearance cues
    97	such as color, clothing, and facial features. However, in real-world longterm scenarios involving large-scale outdoor events or cross-year comparisons, appearance cues often fall short due to lighting variations,
    98	occlusions, and clothing changes [4]. To address these challenges, researchers have explored motion-based modalities such as gait as complementary or alternative identity signals [5]. Our work builds on this
    99	line of research by jointly modeling gait and HAR in a uniﬁed architecture for spatio-temporal Re-ID [6].
   100	Re-ID in time and space. Spatio-temporal Re-ID aims to recognize individuals across signiﬁcant temporal and spatial gaps, where
   101	appearance-based models are particularly vulnerable to domain shift
   102	and context changes [7]. Existing methods often address these issues by
   103	incorporating temporal modeling [8] or leveraging soft biometrics such
   104	as gait [9]. In this regard, gait is especially promising in long-term ReID due to its inherent stability and robustness against changes in clothing and viewpoint. However, many current models still struggle when
   105	faced with unconstrained real-world variability, such as that found in
   106	long-term datasets.
   107	Appearance-based gait recognition. Gait recognition methods can
   108	be broadly categorized into skeleton-based and appearance-based approaches [9]. Our work considers 2D silhouettes used to extract discriminative motion patterns. More recent deep learning-based models,
   109	such as GaitSet [10], use set-based learning to handle unaligned input
   110	frames. GaitPart [11] introduces part-based modeling to capture local
   111	motion features better. These approaches have demonstrated strong performance under controlled conditions but often lack robustness in dynamic, real-world environments.
   112	HAR and multi-modal cues. HAR provides a higher-level understanding of behavior that can complement gait in identity reasoning. Although HAR has been widely studied for activity classiﬁcation
   113	tasks [12], its use in person Re-ID remains limited. Recently, pre-trained
   114	HAR models for person Re-ID in ultra-distance sports scenarios have
   115	been explored, emphasizing the impact of fatigue-induced movement
   116	changes on recognition performance [13]. However, their approach
   117	does not explicitly incorporate gait analysis. We aim to bridge this gap
   118	by jointly embedding HAR and gait cues, enabling richer, context-aware
   119	representations for spatio-temporal Re-ID.
   120	Datasets. Several benchmark datasets support gait-based person
   121	Re-ID research. CASIA-B [14] remains one of the most widely used
   122	datasets, featuring gait sequences from 124 subjects under varying viewpoints and conditions (e.g., normal, bag-carrying, and coat-wearing).
   123	The OU-ISIR gait datasets, including the large-scale OU-MVLP [15], provide extensive samples across diﬀerent age groups, clothing variations,
   124	and viewpoints. OU-MVLP, in particular, includes over 10,000 subjects
   125	captured from 14 view angles, making it one of the most extensive
   126	multi-view gait datasets available. GREW [16] is a recent large-scale
   127	dataset collected from real-world surveillance footage containing over
   128	
   129	3. Methodology
   130	This section describes the proposed two-stream architecture for
   131	spatio-temporal person Re-ID (see Fig. 1), which integrates gait and human action information through pre-trained backbones. We ﬁrst formalize the problem, then detail the structure and roles of the gait and HAR
   132	backbones, the fusion strategy based on cross-attention, and ﬁnally, the
   133	training objective using triplet loss with semi-hard negative sampling.
   134	3.1. Problem formulation
   135	Let  = {(𝑣𝑖 , 𝑦𝑖 )}𝑁
   136	be a dataset of 𝑁 raw video recordings, where:
   137	𝑖=1
   138	𝑣𝑖 denotes the 𝑖th input video captured at a particular time and location,
   139	• 𝑦 ∈ {1, … , 𝐶} is the identity label associated with 𝑣 .
   140	𝑖
   141	𝑖
   142	•
   143	
   144	Each video 𝑣𝑖 is processed through two modality-speciﬁc pipelines
   145	to extract temporal embeddings for gait and HAR.
   146	Gait Processing Pipeline. The gait modality aims to model the subtle motion patterns unique to each individual. To extract these cues, the
   147	raw video 𝑣𝑖 is ﬁrst processed into a sequence of binary silhouettes by
   148	combining person detection and pose estimation. Speciﬁcally, we use
   149	YOLOv8 [18] and Bot-SORT [19] for robust multi-object tracking, enabling consistent localization of the subject across frames. Silhouettes
   150	are later generated using SAMURAI [20]. The resulting silhouette sequence serves as input to a pre-trained gait encoder GAIT , which has
   151	been trained on large-scale public datasets such as CASIA-B, OUMVLP,
   152	and GREW.
   153	The encoder produces a feature tensor from the silhouette sequence:
   154	𝐸𝑖GAIT = GAIT (𝑣silhouettes
   155	) ∈ ℝ𝐷𝑔 ×𝑃
   156	𝑖
   157	Where 𝐷𝑔 is the feature dimension and 𝑃 represents the number of horizontal partitions of the body used during Horizontal Pyramid Pooling
   158	(HPP). This output captures part-level information rather than temporal dynamics; each row corresponds to a distinct horizontal region of
   159	the body (e.g., upper torso, lower legs), not a time step.
   160	2
   161	
   162	Pattern Recognition 172 (2026) 112627
   163	
   164	D. Freire-Obregón et al.
   165	
   166	To ensure stability, each part embedding vector is ﬁrst standardized:
   167	𝐸̃ 𝑖 [∶, 𝑝] =
   168	
   169	Since the output dimensions of the gait and HAR streams may diﬀer
   170	(𝐷𝑔 ≠ 𝐷ℎ ), we apply a projection layer to transform the HAR representation into the common dimension 𝐷:
   171	
   172	𝐸𝑖GAIT [∶, 𝑝] − 𝜇train
   173	𝜎train
   174	
   175	ℎHAR
   176	= 𝜙proj (ℎ̄ HAR
   177	) ∈ ℝ𝐷
   178	𝑖
   179	𝑖
   180	
   181	and then L2-normalized:
   182	𝐸̂ 𝑖 [∶, 𝑝] =
   183	
   184	Both ℎGAIT
   185	∈ ℝ𝐷 and ℎHAR
   186	∈ ℝ𝐷 are projected into a shared embed𝑖
   187	𝑖
   188	ding space of dimension 𝐷, where 𝐷 denotes the common projection
   189	size (i.e., 𝐷𝑔 = 𝐷), ensuring compatibility for subsequent fusion. The
   190	resulting pair of embeddings (ℎGAIT
   191	, ℎHAR
   192	) are aligned in dimension and
   193	𝑖
   194	𝑖
   195	subsequently used for cross-attention-based fusion and identity embedding computation.
   196	
   197	𝐸̃ 𝑖 [∶, 𝑝]
   198	‖𝐸̃ 𝑖 [∶, 𝑝]‖2
   199	
   200	This results in a sequence of normalized part-based embeddings
   201	𝐸̂ 𝑖GAIT ∈ ℝ𝐷𝑔 ×𝑃 . Although they form a sequence-like structure, the order of parts corresponds to spatial locations rather than chronological
   202	time.
   203	Part-wise Encoding. To capture the structured spatial information encoded in the body parts, the normalized part descriptors 𝐸̂ 𝑖GAIT are interpreted as a sequence and passed through a BiLSTM. While this sequence
   204	does not represent time, the recurrent architecture allows for context
   205	modeling across adjacent body regions. This allows the model to capture structured co-movement patterns (e.g., torso-leg coordination) that
   206	are informative for identity. We use a bidirectional LSTM to aggregate
   207	such spatial dependencies. Let 𝐸̃ 𝑖 = (𝐸̂ 𝑖GAIT )⊤ ∈ ℝ𝑃 ×𝐷𝑔 , where the part
   208	dimension 𝑃 is treated as the temporal axis.
   209	
   210	Rationale for temporal aggregation. In the HAR stream, we use global
   211	temporal average pooling in order to obtain clip-level embeddings. This
   212	keeps the HAR stream lightweight, computationally robust, and easy to
   213	train in parallel with the combining module, aligning with typical practice in recent video recognition backbones (e.g., C2D [21], I3D [22],
   214	SlowFast [23], X3D [24]). Shallow recurrent layers and temporal 1D
   215	convolutions in our experiments did not show consistent improvement
   216	under long-term Re-ID, while average pooling kept accuracy with a modest cost. Even though more intricate aggregation methods are potentially
   217	available (e.g., self-attention, Transformer pooling), we intentionally select simplicity and robustness in order to uncover the isolated contribution of HAR cues towards gait-based recognition.
   218	
   219	(
   220	(
   221	(
   222	)))
   223	ℎGAIT
   224	= 𝜙 BiLSTM256 Dropout BiLSTM128 (𝐸̃ 𝑖 )
   225	𝑖
   226	Here, 𝜙 denotes a dense layer with ReLU activation. This conﬁguration models spatial body part embeddings as a pseudo-temporal sequence, allowing the BiLSTM to capture part-to-part relational dynamics
   227	relevant to identity.
   228	HAR Processing Pipeline. The HAR stream is designed to capture
   229	high-level activity patterns from the subject’s movement. However, raw
   230	videos may contain multiple actors, visual clutter, or background distractions To ensure that the extracted embeddings focus solely on the
   231	subject of interest, we apply a context-constrained preprocessing step.
   232	Step 1: Video Preprocessing - Context Constraint. We use the silhouettes
   233	previously computed to crop a tight region around the individual in each
   234	frame. The rest of the frame is suppressed by superimposing the silhouette onto a static mode frame 𝑓 . For an individual 𝑖 at time 𝑡 ∈ [0, 𝑇 ], let
   235	𝑆𝑖𝑙(𝑖) (𝑡) be the silhouette, and 𝐹 (𝑖) (𝑡) the raw frame. The preprocessed
   236	frame is deﬁned as:
   237	(
   238	)
   239	𝐹 ′(𝑖) (𝑡) = Crop(𝐹 (𝑖) (𝑡), 𝑆𝑖𝑙(𝑖) (𝑡)) + 1 − 1𝑆𝑖𝑙(𝑖) (𝑡) ⋅ 𝑓
   240	
   241	3.2. Cross-attention fusion
   242	We apply a cross-attention mechanism to fuse both modalities, where
   243	the gait embedding attends to the HAR representation. Given the aligned
   244	embeddings (ℎGAIT
   245	, ℎHAR
   246	) for sample 𝑖, we deﬁne:
   247	𝑖
   248	𝑖
   249	𝑄𝑖 = ℎGAIT
   250	∈ ℝ1×𝐷
   251	𝑖
   252	𝐾𝑖 = 𝑉𝑖 = ℎHAR
   253	∈ ℝ1×𝐷
   254	𝑖
   255	The attention weights and attended HAR embedding are computed
   256	as:
   257	
   258	′
   259	
   260	where 𝑇 ′ is the number of temporal segments or frames output by the
   261	HAR backbone and 𝐷ℎ is the dimensionality of each HAR embedding
   262	vector.
   263	After extracting the sequence of embeddings from the HAR backbone, we apply normalization frame-wise before pooling. First, the embeddings are standardized using the mean and standard deviation computed across the training set, and then, each frame is L2-normalized:
   264	
   265	HAR
   266	𝜎train
   267	
   268	,
   269	
   270	𝐸̂ 𝑖HAR =
   271	
   272	)
   273	,
   274	
   275	𝑧𝑖 = 𝛼𝑖 ⋅ 𝑉𝑖
   276	
   277	where [ ; ] denotes concatenation and 𝜙proj is a dense projection layer.
   278	In this setup, gait is deﬁned as the querying modality (serving as
   279	the Query in the attention mechanism) because the primary objective
   280	of the framework is identity recognition rather than activity categorization. Gait embeddings are explicitly trained to discriminate identities
   281	and thus form the base representation to be preserved. HAR features, in
   282	turn, provide the Keys and Values, oﬀering complementary contextual
   283	cues that enrich the gait signal without shifting the focus toward action
   284	recognition. Through the attention mechanism, the model selectively
   285	integrates relevant HAR information into the gait embedding. The resulting attended representation is then combined with the original gait
   286	embedding, ensuring that gait remains the dominant identity signal. Finally, a projection layer fuses both sources into a uniﬁed identity representation in a common embedding space, where samples can be directly
   287	compared using L2 distance for Re-ID.
   288	
   289	𝐸𝑖HAR = HAR (𝑣context
   290	) ∈ ℝ𝑇 ×𝐷ℎ
   291	𝑖
   292	
   293	HAR
   294	𝐸𝑖HAR − 𝜇train
   295	
   296	𝑄𝑖 𝐾𝑖⊤
   297	√
   298	𝐷
   299	
   300	We then concatenate the original gait embedding with the attended
   301	HAR vector and apply a projection layer 𝜙 to obtain the ﬁnal fused
   302	identity representation:
   303	(
   304	)
   305	𝑓𝑖 = 𝜙proj [ℎGAIT
   306	; 𝑧𝑖 ] ∈ ℝ𝐷
   307	𝑖
   308	
   309	This produces a context-constrained video 𝑣context
   310	= {𝐹 ′(𝑖) (𝑡)}𝑇𝑡=0 fo𝑖
   311	cused solely on the individual.
   312	Step 2: HAR Feature Extraction. The processed video 𝑣context
   313	is then
   314	𝑖
   315	passed through a pre-trained action recognition backbone HAR , which
   316	outputs a sequence of temporal embeddings:
   317	
   318	𝐸̃ 𝑖HAR =
   319	
   320	(
   321	
   322	𝛼𝑖 = softmax
   323	
   324	𝐸̃ 𝑖HAR
   325	‖𝐸̃ HAR ‖2
   326	𝑖
   327	
   328	3.3. Triplet loss for metric learning
   329	
   330	Although temporal modeling techniques, such as those used in gait
   331	recognition, aim to capture patterns, we observed that naive temporal
   332	processing alone did not yield signiﬁcant improvements in Re-ID performance. Then, average pooling is applied over the temporal dimension
   333	to aggregate frame-level features into a sequence-level representation:
   334	
   335	To teach the model how to tell diﬀerent people apart, we use a triplet
   336	loss function. It compares three examples at a time:
   337	•
   338	
   339	ℎ̄ HAR
   340	= AvgPool(𝐸̂ 𝑖HAR ) ∈ ℝ𝐷ℎ
   341	𝑖
   342	
   343	•
   344	
   345	3
   346	
   347	an anchor (𝑎), a reference video of a person,
   348	a positive sample (𝑝), another video of the same person,
   349	
   350	Pattern Recognition 172 (2026) 112627
   351	
   352	D. Freire-Obregón et al.
   353	•
   354	
   355	Table 1
   356	Number of runners and videos per location pair in
   357	the training and test sets. The training set contains
   358	204 runners: 189 observed in two diﬀerent locations
   359	and 15 observed in three, resulting in a total of 423
   360	labeled videos. Because some runners contribute to
   361	multiple location pairs, the sum of runner and video
   362	counts across pairs exceeds the total number of unique
   363	training videos. The test set comprises 17 runners observed in three diﬀerent locations, resulting in 51 labeled videos.
   364	
   365	a negative sample (𝑛), a video of a diﬀerent person.
   366	
   367	The goal is to make sure that the model places the anchor closer to
   368	the positive than to the negative in the embedding space, with some
   369	extra margin 𝛼 for safety. The loss is deﬁned as:
   370	triplet =
   371	
   372	𝑁
   373	∑
   374	
   375	(
   376	)
   377	max ‖𝑓 (𝑎𝑗 ) − 𝑓 (𝑝𝑗 )‖22 − ‖𝑓 (𝑎𝑗 ) − 𝑓 (𝑛𝑗 )‖22 + 𝛼, 0
   378	
   379	𝑗=1
   380	
   381	Here:
   382	𝑓 (⋅) is the embedding function (i.e., the model’s output for each input),
   383	• ‖ ⋅ ‖ is the L2 distance between embeddings,
   384	2
   385	• 𝛼 is a ﬁxed margin (e.g., 0.5),
   386	• The max(⋅, 0) makes sure we only penalize the model when the negative is too close to the anchor.
   387	•
   388	
   389	If the model already separates the anchor and negative correctly, we
   390	don’t penalize it. If it doesn’t, the loss becomes positive and the model
   391	learns from that mistake.
   392	To construct eﬀective triplets, we apply semi-hard negative mining,
   393	selecting negatives that satisfy:
   394	
   395	Set
   396	
   397	Location Pair
   398	
   399	# Runners
   400	
   401	# Videos
   402	
   403	Train
   404	Train
   405	Train
   406	Train
   407	Train
   408	Train
   409	
   410	RP0_20 ↔ RP1_20
   411	RP2_23 ↔ RP3_23
   412	RP1_20 ↔ RP3_23
   413	RP0_20 ↔ RP3_23
   414	RP2_23 ↔ RP1_20
   415	RP0_20 ↔ RP2_23
   416	
   417	158
   418	39
   419	15
   420	11
   421	6
   422	5
   423	
   424	316
   425	78
   426	30
   427	22
   428	12
   429	10
   430	
   431	Test
   432	Test
   433	Test
   434	
   435	RP1_20 ↔ RP2_23
   436	RP1_20 ↔ RP3_23
   437	RP2_23 ↔ RP3_23
   438	
   439	17
   440	17
   441	17
   442	
   443	34
   444	34
   445	34
   446	
   447	‖𝑓 (𝑎) − 𝑓 (𝑝)‖22 < ‖𝑓 (𝑎) − 𝑓 (𝑛)‖22 < ‖𝑓 (𝑎) − 𝑓 (𝑝)‖22 + 𝛼
   448	We deﬁne a modality-aware distance used both in the loss function
   449	and during negative sample selection:
   450	𝑑combined (𝑥, 𝑦) = 𝜆 ⋅ ‖ℎGAIT
   451	− ℎGAIT
   452	‖2 + (1 − 𝜆) ⋅ ‖ℎHAR
   453	− ℎHAR
   454	‖2
   455	𝑥
   456	𝑦
   457	𝑥
   458	𝑦
   459	This combined distance selects negative samples across identities and
   460	checkpoints during mining, with 𝜆 = 0.5 balancing the contribution of
   461	gait and HAR modalities equally.
   462	4. Dataset
   463	Our study is based on a dataset collected at four diﬀerent locations during the 2020 and 2023 editions of the Transgrancanaria ultradistance running competition. In this event, athletes compete on a 128kilometer course that typically takes between 12 and 30 h to complete.
   464	Each sample in the dataset consists of a short video clip, typically lasting no more than ten s, recorded at 25 frames per second. The dataset
   465	does not exhibit a symmetric structure in which each runner is recorded
   466	at every location. Instead, most runners appear in two locations, while
   467	a subset of ﬁfteen runners appear in three within the training set and
   468	seventeen in three within the test set. As a result, location pairs diﬀer
   469	in how much they share subjects. Some pairs involve identical sets of
   470	runners, while others involve partially overlapping or entirely distinct
   471	subsets. This necessitates a pair-level analysis when interpreting training and test splits, as seen in Table 1.
   472	The training set of the dataset comprises a total of 423 labeled videos,
   473	each corresponding to a unique observation of a runner at a speciﬁc
   474	Recording Point (RP), identiﬁed by a location-year code (RPloc_year).
   475	These observations are distributed across four distinct points in diﬀerent years: RP0_20, RP1_20, RP2_23, and RP3_23. Runners are most often recorded in exactly two of these locations, forming 234 unique unordered pairs, which serve as the fundamental unit of co-occurrence
   476	in the analysis. The pair RP0_20 ↔ RP1_20 emerges as the most frequent, appearing in 158 instances, indicating a strong connection between these two 2020 locations. Similarly, the 2023 pair RP2_23 ↔
   477	RP3_23 appears 39 times, reﬂecting a notable but less dominant colocation pattern.
   478	The training dataset includes 15 runners who appear in three distinct locations. Each of these runners contributes three unique location
   479	pairs, increasing the number of co-occurrence pairs without a proportional increase in video samples, as explained in Table 1. Their presence
   480	adds structural complexity to the network of relationships by connecting more location pairs per individual rather than inﬂating the dataset
   481	size.
   482	
   483	Fig. 2. These dataset samples illustrate short-term and long-term Re-ID scenarios. Each column corresponds to the same individual, while each row shows data
   484	captured at diﬀerent locations and years. This ﬁgure has been anonymized for
   485	privacy purposes.
   486	
   487	Moreover, the training dataset contains both same-year pairs, such
   488	as RP0_20 ↔ RP1_20 and RP2_23 ↔ RP3_23, and cross-year pairs, including examples like RP0_20 ↔ RP3_23 and RP1_20 ↔ RP2_23. These
   489	temporal pairings provide insight into longitudinal movement patterns,
   490	highlighting transitions and tracking continuity across years (see Fig. 2).
   491	This structure oﬀers a rich foundation for spatio-temporal analysis and
   492	deeper modeling of athlete behavior across time and locations.
   493	The test partition contains 51 videos from 17 runners, each recorded
   494	in three locations: RP1_20, RP2_23, and RP3_23. Evaluation is performed through pairwise comparisons between locations in both directions (e.g., RP1_20 to RP2_23 and RP2_23 to RP1_20), yielding six directional evaluation scenarios. For each direction, the 17 runners from
   495	the source location are matched against the 17 runners from the target location, producing 289 comparisons per direction. This results in a
   496	total of 1734 comparisons across the test set. The design ensures a comprehensive and balanced evaluation of generalization across spatial and
   497	temporal dimensions while maintaining a subject-disjoint protocol.
   498	4
   499	
   500	Pattern Recognition 172 (2026) 112627
   501	
   502	D. Freire-Obregón et al.
   503	
   504	5. Gait & HAR backbones: From motion to identity
   505	
   506	poral variants via progressive scaling. The four X3D versions diﬀer in
   507	complexity and cost: XS applies multiple scaling stages, S reduces the
   508	frame rate, M increases spatial resolution, and L adds depth through
   509	deeper residual blocks. This structured design strikes a balance between
   510	eﬃciency and performance.
   511	Several models (I3D, Slow, SlowFast) are enhanced with Non-local
   512	Networks (NLN) [29], which compute global pairwise dependencies
   513	across space and time. All backbones are pre-trained on Kinetics400 [30], enabling robust and transferable HAR performance across varied conditions.
   514	
   515	In this section, we provide a detailed description of the backbones
   516	used for gait and HAR, highlighting their architectural design and the
   517	type of information each captures. These modality-speciﬁc backbones
   518	form the foundation of our two-stream framework, with the gait backbone focusing on ﬁne-grained motion dynamics and the HAR backbone
   519	capturing high-level activity patterns.
   520	Gait Backbones. Silhouette-based gait recognition methods focus
   521	on extracting discriminative features from silhouette sequences to identify individuals based on their walking patterns. In our work, we employ several representative models-GaitBase, GLN Phase 1, GLN Phase
   522	2, GaitGL, GaitPart, and GaitSet-which share the common objective of
   523	leveraging silhouette information yet diﬀer signiﬁcantly in their architectural designs. GaitSet treats gait sequences as unordered frame sets,
   524	using temporal pooling to aggregate frame-level features without modeling explicit spatial relationships. Building on this, GaitPart [11] incorporates part-based modeling using Focal Convolution (FConv) to extract
   525	local features across horizontal body regions, enhancing ﬁne-grained
   526	spatial detail but potentially introducing sensitivity to alignment errors.
   527	GaitGL [25] reﬁnes this approach by combining global and local feature extraction branches with 3D convolutional layers to capture holistic
   528	and localized temporal-spatial dynamics. However, its added complexity does not always yield consistent improvements in real-world scenarios. GLN Phase 1 introduces a grouped latent representation strategy
   529	to disentangle feature learning, while GLN Phase 2 extends this with
   530	a reﬁnement stage to progressively enhance feature granularity across
   531	network layers [26]. In contrast, GaitBase [27] adopts a deep residual
   532	network architecture to serve as a strong baseline backbone, demonstrating the beneﬁts of increased network capacity for capturing robust
   533	gait patterns.
   534	As previously mentioned in Section 3, the models have been trained
   535	and evaluated on three benchmark datasets-OU-MVLP, CASIA-B, and
   536	GREW-which vary signiﬁcantly in terms of scale, environmental conditions, and overall complexity. OU-MVLP [15] is a large-scale constrained
   537	dataset collected in controlled indoor conditions with consistent camera
   538	viewpoints, oﬀering extensive identity coverage but limited environmental variation. Although smaller in scale, CASIA-B [14] introduces
   539	covariates such as clothing and carrying conditions in a multi-view indoor setup, making it valuable for analyzing speciﬁc intra-subject variations. In contrast, GREW [16] is a real-world, in-the-wild dataset captured in unconstrained environments with diverse backgrounds, lighting
   540	conditions, and occlusions, reﬂecting practical deployment challenges
   541	and more related to the running scenario of our dataset. Due to these
   542	diﬀerences, not all models are trained across all datasets. Some architectures, particularly those relying on sensitive spatial modeling (e.g.,
   543	GaitPart, GaitGL), may struggle to generalize on GREW without extensive reconﬁguration. Furthermore, training high-capacity models such
   544	as GLN Phase 2 or GaitBase on GREW can be computationally demanding due to the dataset’s scale and complexity. Consequently, model training choices are inﬂuenced by dataset characteristics and the architectural robustness and scalability of the gait backbones.
   545	HAR Backbones. We employed a diverse set of backbone architectures with varying capacities to model spatial and temporal dynamics, including C2D, I3D, Slow8x8, Slow4x16, SlowFast8x8, SlowFast4x16, and X3D variants (L, M, S, XS). The C2D model [21] employs 2D convolutions on individual frames, treating the video as a sequence of static images. While eﬃcient, it lacks explicit temporal modeling. I3D [22] overcomes this by inﬂating 2D ﬁlters to 3D and using
   546	two streams (RGB and optical ﬂow) to capture appearance and motion
   547	jointly. SlowNet [28] processes fewer frames at high spatial resolution
   548	to model long-term patterns; we use Slow8x8 and Slow4x16 to vary
   549	temporal coverage.
   550	SlowFast [23] adds a high-frame-rate path for fast motion, complementing the slow path’s semantic focus. We evaluate SLF_8x8 and
   551	SLF_4x16. The X3D family [24] expands a 2D base model into spatiotem-
   552	
   553	6. Experimental setup
   554	Baselines. As baseline models for person Re-ID, we consider OSNet, DenseNet-121, MobileNet V2, and AlignedReID. All models are
   555	trained on the Market1501 dataset, a widely used benchmark in the
   556	Re-ID community. OSNet (Omni-Scale Network) [31] is designed explicitly for person Re-ID, featuring a multi-stream architecture that captures
   557	local and global features through dynamic, omni-scale feature aggregation while remaining lightweight and eﬃcient. DenseNet-121 [32] is a
   558	popular backbone known for its densely connected layers, which encourage feature reuse and eﬃcient gradient propagation, oﬀering robust performance at a higher computational cost. MobileNet V2 [33] is
   559	a compact and eﬃcient model optimized for mobile and real-time applications, utilizing depthwise separable convolutions to signiﬁcantly reduce model size and inference time, with a trade-oﬀ in accuracy. AlignedReID [34] builds upon the ResNet backbone and incorporates local
   560	feature alignment through part-based matching, enhancing the model’s
   561	ability to handle misalignment and pose variation-two common challenges in Re-ID scenarios. Together, these baselines oﬀer a range of
   562	architectural styles, from lightweight mobile networks to specialized
   563	Re-ID solutions. These models are adapted for our task by leveraging
   564	tracklets extracted from the input videos. For each individual, we isolate
   565	sequences of bounding boxes (tracklets) corresponding to their appearances across the video timeline. To ensure robust and fair evaluation, we
   566	perform multiple comparisons between diﬀerent runners, averaging the
   567	results over several runs to mitigate variance introduced by temporal
   568	sampling or environmental noise.
   569	Implementation details. First, the gait pipeline is pre-trained; once
   570	fusion is enabled, we ﬁrst train the fusion layers (lr = 6e-5) and then
   571	ﬁne-tune the last gait layers with a reduced learning rate (lr = 1e5). We trained the network with Adam. The training was conducted
   572	using a batch size of 32, and each iteration involved generating semihard triplets to enhance convergence and model robustness. The model
   573	was trained for 2000 iterations, with evaluations performed every 500
   574	to monitor performance. We employed the Triplet Loss with a margin (𝛼) 0.6 to encourage a clear separation between positive and negative pairs in the embedding space. Performance was assessed using
   575	standard retrieval metrics, including Cumulative Match Characteristic
   576	(CMC) curves and mean Average Precision (mAP), reported per location
   577	segment to capture spatial consistency across diﬀerent checkpoints.
   578	Experimental protocol. We performed a ﬁve times 10-fold split
   579	strategy on the training set. In each run, eight folds were used for
   580	training, one for validation, and one for internal testing. This approach
   581	thoroughly evaluated various hyperparameter conﬁgurations and fusion
   582	strategies while keeping the external test set completely untouched. The
   583	conﬁguration that achieved the best performance was then selected to
   584	train the ﬁnal model using the entire training set. The results reported in
   585	this work are based exclusively on evaluating this ﬁnal model on the untouched test set, ensuring a fair and unbiased assessment of the system’s
   586	generalization capabilities.
   587	7. Long-Term Re-ID experiments
   588	Long-term Re-ID is a key focus of this work, as it reﬂects real-world
   589	challenges where individuals must be re-identiﬁed after extended tem5
   590	
   591	Pattern Recognition 172 (2026) 112627
   592	
   593	D. Freire-Obregón et al.
   594	
   595	Table 2
   596	Long-term Re-ID using only gait features. Cross-year evaluations between
   597	RP1_20 and RP2_23. Metrics reported as mAP / Rank-1 / Rank-5. Baseline
   598	performances are shown in gray, and the best-performing backbone is highlighted in blue.
   599	Gait Backbone
   600	
   601	RP1_20 → RP2_23
   602	mAP↑
   603	
   604	MobileNet V2 [33]
   605	DenseNet 121 [32]
   606	OSNet [31]
   607	AlignedReID [34]
   608	GaitBase_C [27]
   609	GaitBase_O [27]
   610	GLNp1_C [26]
   611	GLNp2_C [26]
   612	GaitGL_C [25]
   613	GaitGL_G [25]
   614	GaitGL_O [25]
   615	GaitPart_C [11]
   616	GaitPart_G [11]
   617	GaitPart_O [11]
   618	GaitSet_C [10]
   619	GaitSet_G [10]
   620	GaitSet_O [10]

codex
第三篇不是普通图像 ReID，它把“长时换衣导致外观不可靠”重定义成“运动生物特征加行为上下文”的问题，因此贡献有一半是数据设定本身。第四篇的重定义更工程化：欧氏空间保局部身份差异，希尔伯特空间保全局跨模态统计依赖，二者互相蒸馏才叫 dual-space。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '760,820p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   760	(29)
   761	
   762	IV. E XPERIMENTS
   763	A. Datasets and Implementation Details
   764	SYSU-MM01. The SYSU-MM01 [24] dataset consists of
   765	491 identities captured by four VIS cameras and two IR
   766	cameras in both indoor and outdoor environments. The training
   767	set includes 19,659 visible light (VIS) images and 12,792
   768	infrared (IR) images from 395 identities, while the test set
   769	comprises 96 identities and 3,803 IR query images. According
   770	to [24], the gallery set depends on the testing mode, which
   771	can either be the all-search mode or the indoor-search mode.
   772	In the all-search mode, all images captured by VIS cameras
   773	are used as gallery images, while in the indoor-search mode,
   774	only images from two indoor VIS cameras are used as gallery
   775	images.
   776	RegDB. The RegDB [39] dataset contains 412 identities,
   777	captured by two aligned cameras, with 10 VIS images and 10
   778	IR images per identity, totaling 8,240 images. In accordance
   779	with existing protocols, we randomly select 206 identities for
   780	the training set, and the remaining 206 identities are used for
   781	testing. There are two test modes: VIS-to-IR and IR-to-VIS.
   782	In the VIS-to-IR mode, both VIS and IR images are used as
   783	queries and gallery images, and vice versa. Performance is
   784	calculated based on the experimental results from 10 random
   785	splits of the training and testing sets.
   786	LLCM. The LLCM [58] dataset consists of 46,767 images
   787	from 1,064 identities, captured by nine VIS and IR cameras.
   788	The training set includes 30,921 images from 713 identities
   789	(16,946 VIS images and 13,975 IR images), while the test set
   790	contains 13,909 images from 351 identities (8,680 VIS images
   791	and 7,166 IR images). As in [58], we evaluate the proposed
   792	MDDM method using both VIS-to-IR and IR-to-VIS modes.
   793	Evaluation Metrics. We adopt the standard Cumulative
   794	Matching Characteristics (CMC) and mean Average Precision
   795	(mAP) as evaluation metrics.
   796	Implementation Details. We implemented our model using
   797	the Pytorch framework and trained it on a single RTX 4090
   798	
   799	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:25 UTC from IEEE Xplore. Restrictions apply.
   800	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
   801	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
   802	
   803	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
   804	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3682328
   805	
   806	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
   807	
   808	8
   809	
   810	TABLE I
   811	C OMPARISON WITH SOTA METHODS ON THE SYSU-MM01 AND R EG DB DATASETS . R1 AND R10 DENOTE THE R ANK -1
   812	(%) AND R ANK -10 (%), RESPECTIVELY. W E ROUGHLY DIVIDE THE COMPARED METHODS INTO TWO CATEGORIES :
   813	AUXILIARY- BASED AND FEATURE - BASED METHODS . T HE RED BOLD AND BLUE BOLD FONT DENOTE THE BEST AND
   814	SECOND - BEST PERFORMANCES .
   815	Methods
   816	
   817	Venue
   818	
   819	AlignGAN [24]
   820	JSIA [63]

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '320,760p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   320	
   321	Fig. 2. Network structure of NNSC-FR. NNSC strategy applies the nearest neighbor constraint to each instance sample and realizes the fine classification
   322	of hard samples between classes; CB loss can better identify hard samples between the nearest neighbor classes and improve clustering accuracy; ODE-FR
   323	improves the model’s feature representation ability.
   324	
   325	III. M ETHODS
   326	A. Baseline and Overall Structure
   327	1) Baseline: Global and local feature representation learning methods have shown effectiveness in various subtasks of
   328	person Re-ID. Therefore, we take PPLR [23] based on global
   329	and part feature learning as our baseline. PPLR calculates
   330	the cross-consistency scores between part and global image
   331	features, performs pseudo-label refinement, and optimizes features through global and local mutual refinement loss. Global
   332	and part features are refined by the agreement-aware label
   333	smoothing loss Laals , the part-guided label refinement loss
   334	L pglr , and the combination of cross-entropy loss Lce and triplet
   335	loss Ltri refines the pseudo-labels. The total PPLR loss is as
   336	follows:
   337	LPPLR = Lce + Ltri + Laals + L pglr .
   338	(1)
   339	2) Overall Structure: Although PPLR refines the global and
   340	part features and captures some local details, it is still difficult
   341	to classify some easily confused samples. On the one hand,
   342	the classification accuracy of hard cluster boundary samples
   343	needs improvement. On the other hand, the model’s feature
   344	representation capability requires enhancement. Therefore, we
   345	design three modules to optimize the clustering accuracy of
   346	features and improve the quality of pseudo-labels. As shown
   347	in Fig. 2, the total network structure of NNSC-FR is proposed.
   348	First, under the stable constraint of class centroid contrast
   349	loss, the nearest neighbor sample constraint strategy finds
   350	the farthest positive sample and the nearest negative sample
   351	between classes for each training instance and pulls out the
   352	two cluster blocks. Secondly, with the help of inter-class
   353	balance loss, the classification criteria of the nearest cluster
   354	are further tightened to improve the clustering accuracy. In
   355	addition, through the design of the residual network structure,
   356	the ODE guided feature reconstruction module is inserted into
   357	the third layer of ResNet-50, and the problem of reduced
   358	representational ability is solved by embedding the PReLU
   359	layer and convolutional layer.
   360	
   361	Fig. 3. Nearest neighbor sample constraint strategy.
   362	
   363	B. Nearest Neighbor Sample Constraint
   364	To store and update feature clusters, we introduce a memory
   365	bank that retains the global features after feature extraction
   366	and clustering. The memory bank is updated through contrast
   367	loss every time when mini-batch samples are trained. To
   368	increase intra-class compactness and inter-class separability,
   369	the mainstream advanced methods minimize the distance
   370	between samples of the same class and maximize the distance
   371	between samples of different classes by contrast loss Lq :
   372	˝
   373	˛ 
   374	exp q, k+ /τ
   375	Lq = −log PK
   376	(2)
   377	˝
   378	˛ ,
   379	i
   380	i=1 exp q, k /τ
   381	where q is characteristic of the query from a set of candidate
   382	k1 , k2 ,. . ., k K , k+ is positive characteristics with the same label
   383	as q, τ is the temperature hyperparameter that controls the
   384	similarity range.
   385	Some instance-level memory algorithms [47] maintain each
   386	feature of the pedestrian image in the dataset and update the
   387	corresponding memory dictionary with its instance feature in
   388	each small batch. Still, there is a memory update consistency problem [10]; because of the unbalanced cluster size
   389	distribution, smaller clusters may update more instances than
   390	larger clusters. As shown in Fig. 3 (a), we reserve the cluster
   391	centroid feature for each cluster rather than each instance
   392	feature. Regardless of cluster size, the corresponding features
   393	
   394	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:00 UTC from IEEE Xplore. Restrictions apply.
   395	
   396	5612
   397	
   398	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 4, APRIL 2026
   399	
   400	are updated in the memory to ensure the consistency of feature
   401	updates within the same cluster. Thus, the cluster centroid
   402	contrast loss is defined as:
   403	˝
   404	˛ 
   405	exp q, c+ /τc
   406	Lcen = −log PC
   407	(3)
   408	˝
   409	˛ ,
   410	i
   411	i=1 exp q, c /τc
   412	where C is the number of clusters in the training stage, c+
   413	is the positive cluster centroid, and τc is the temperature
   414	hyperparameter. Unlike the contrast loss Lq , the negative
   415	impact of outliers in Lcen , such as boundary hard samples,
   416	on model learning is reduced. Cluster centroid c1 ,c2 ,. . .,cC is
   417	calculated and stored in memory for cluster centroid contrast
   418	losses. We update the cluster memory bank as follows:
   419	ci ← αci + (1 − α) c̄i ,
   420	
   421	(4)
   422	
   423	i
   424	
   425	where c̄ is the average of the features of the class instance i
   426	in the small batch, and α is the hyperparameter adjusting the
   427	cluster center update weight.
   428	To further distinguish easily confused samples, we propose
   429	the nearest neighbor sample constraint strategy, in which we
   430	construct a pair of contrast samples for each query sample (the
   431	sample that needs to be matched and classified), including a
   432	positive sample that belongs to the same category and a negative sample that belongs to a different category. Specifically,
   433	in addition to the clustering centroids, the memory module
   434	also stores K = P × C instance features. These features span
   435	C pseudo-label categories, with each category containing P
   436	instances, where the values of both P and C are determined
   437	by the clustering process. As shown in Fig. 3 (b), different
   438	from traditional sampling strategies (such as the triplet loss)
   439	that calculate the distance between the hardest positive and
   440	negative cases in mini-batches based on paired losses, our
   441	method is based on querying the negative samples on all
   442	pseudo-labels in the memory. The stability of the whole cluster
   443	can be ensured under the constraint of class centroid loss,
   444	while the nearest neighbor sample constraint can optimize the
   445	hard samples more carefully at the instance level and reduce
   446	the feature perturbations inside other classes. The contrastive
   447	loss of the nearest neighbor sample mining strategy is defined
   448	as follows:
   449	˝
   450	˛
   451	
   452	exp q, z+
   453	hard /τnea
   454	˝
   455	˛
   456	
   457	˝ − ˛
   458	 , (5)
   459	Lnea = −log
   460	exp q, z+
   461	hard /τnea + exp q, zhard /τnea
   462	where τnea is the instance temperature hyperparameter, z+
   463	hard
   464	is the hard positive instance feature with the lowest cosine
   465	similarity to the query sample in the same cluster, and z−hard
   466	is the hard negative instance feature with the highest cosine
   467	similarity in the memory bank.
   468	˝ + ˛
   469	z+
   470	, k = 1, . . . , K;
   471	hard = argmin q, zk
   472	˝ − ˛
   473	−
   474	zhard = argmax q, zk , k = 1, . . . , K.
   475	(6)
   476	−
   477	where z+
   478	k is the positive instance and where zk is the negative
   479	one. Similarly, to ensure the consistency of memory updates,
   480	all instance features of the corresponding K IDs in a minibatch are updated in each training iteration. Finally, the cluster
   481	centroid contrast loss Lcen and the nearest neighbor sample
   482	contrast loss Lnea together constitute the global contrast loss:
   483	
   484	LGC = γLcen + (1 − γ) Lnea .
   485	
   486	(7)
   487	
   488	Fig. 4. t-SNE visualization comparison of feature distributions. (a) without
   489	CB loss; (b) with CB loss. The red circles highlight hard samples that
   490	are ambiguous and close to decision boundaries in (a). After applying CB
   491	loss in (b), these hard samples are better separated and classified more
   492	clearly, demonstrating the effectiveness of CB loss in improving feature
   493	discrimination.
   494	
   495	Under the guidance of global contrast loss, the clustering
   496	algorithm can better classify the hard samples and push the
   497	nearest neighbors to a suitable distance. As the hard samples
   498	are optimized, the model progressively refines the decision
   499	boundaries in the feature space, ultimately achieving a more
   500	accurate classification of all samples. Moreover, the constraints
   501	between the nearest neighbors can also avoid problems such
   502	as the inconsistent updating direction of the samples of multineighbor classes and play an excellent guiding role in the
   503	training of an unsupervised person Re-ID model.
   504	C. Inter-Class Balance Loss
   505	As shown in Fig. 4, to encourage the network to move the
   506	nearest neighbor hard samples closer to the correct cluster
   507	centroid, we construct a simple binary classification network
   508	and propose the balance loss between classes, which is defined
   509	as follows:
   510	0
   511	1
   512	Np
   513	n 
   514	
   515	
   516	
   517	X
   518	X
   519	1
   520	@
   521	LCB = −
   522	(8)
   523	xij log xij + a A ,
   524	Np
   525	i=1
   526	
   527	j=1
   528	
   529	where N p is the batch number of images, xij is the j coordinates
   530	of Cd (F (xi | θ)), Cd is a simple classification network, a is the
   531	fixed constant that prevents a loss explosion which is set as
   532	1e − 8. In the inter-class balance loss, consider the function:
   533	f (x) = − (xlog (x) + a) , x ∈ (0, 1) .
   534	
   535	(9)
   536	
   537	The second derivative of f (x) is:
   538	f 00 (x) = −
   539	
   540	1
   541	< 0, x ∈ (0, 1) .
   542	x
   543	
   544	(10)
   545	
   546	Therefore, it is a concavePfunction. According to
   547	j
   548	j
   549	n
   550	Jensen’s inequality [48], given
   551	j=1 xi = 1, when xi =
   552	1
   553	(
   554	n j = 1, 2, . . . , n), the maximum value of the function can be
   555	reached. When LCB is maximum, the distance between xij
   556	and n1 is shortened, and the class of hard samples between
   557	the nearest neighbor classes is most difficult to distinguish.
   558	The function of neural network gradient backpropagation is
   559	to minimize the loss. Therefore, based on the hard sample
   560	constraint strategy, the inter-class balance loss is designed to
   561	reduce its value so that the hard samples that are difficult
   562	to distinguish between neighboring classes can be further
   563	moved towards the classification direction of the centroid, the
   564	
   565	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:00 UTC from IEEE Xplore. Restrictions apply.
   566	
   567	YANG et al.: NNSC AND ODE GUIDED FEATURE RECONSTRUCTION FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   568	
   569	5613
   570	
   571	clustering is clearer, and the confidence of the pseudo-labels
   572	after refinement is higher.
   573	Thus, the total loss of our proposed algorithm is obtained
   574	as follows:
   575	LNNS C−FR = LPPLR + LGC + λLCB .
   576	
   577	(11)
   578	
   579	Discussion: The design of the Inter-Class Balance loss
   580	function is fundamentally based on the probabilistic distribution constraint mechanism constructed from the properties of
   581	concave functions and Jensen’s inequality. When LCB reaches
   582	its maximum value, the predicted probabilities of samples
   583	across all categories tend to a uniform distribution. At this
   584	point, there is significant confusion in the decision boundary
   585	regions of adjacent categories in the feature space, making
   586	it difficult to effectively distinguish difficult samples. By
   587	minimizing LCB through backpropagation, the model enforces
   588	optimization during the gradient descent process: it applies
   589	penalties to difficult samples that are close to the interclass boundary, compelling the feature vectors to escape the
   590	state of uniform classification distribution. This mechanism
   591	dynamically pushes difficult samples away from the confusion
   592	region during optimization, ultimately achieving minimized
   593	intra-class variance and clarified inter-class boundaries.
   594	D. ODE Guided Feature Reconstruction
   595	Many research [13], [45], [46] have explained the design
   596	of deep neural networks in terms of differential equations. For
   597	example, He et al. [45] derived network structures corresponding to two solution methods of ordinary differential equations
   598	by using numerical schemes in ODEs as blueprints. Unlike
   599	their approach of designing various ODE-blocks as the main
   600	building blocks of super-resolution networks using explicit
   601	Runge-Kutta methods, our ODE-FR module is designed based
   602	on the implicit third-order Adam solution and embedded
   603	in the third layer of ResNet-50 as a feature enhancement
   604	component. This multi-step structure can leverage multi-layer
   605	historical feature information for feature reconstruction, and
   606	enhance feature discriminability by embedding PReLU layers
   607	and convolutional layers, which is well-suited to the demands
   608	for feature representation capability in Re-ID tasks.
   609	As shown in Fig. 5 (a), it illustrates feature fusion, which
   610	refers to the combination of features from different branches to
   611	form a richer feature representation. This approach integrates
   612	diverse information, thereby enhancing the model’s expressive
   613	capability. Fig. 5 (b) represents feature cascading, where features from different levels are sequentially connected to create
   614	a hierarchical feature representation. This method emphasizes
   615	the relationships between features learned at various depths of
   616	the model. However, both methods exhibit issues of blindness
   617	and dependency on specific tasks. In contrast, we derive and
   618	design the ODE-FR module (Fig. 5 (c)) based on the thirdorder implicit Adam solution formula of ordinary differential
   619	equations, effectively overcoming these shortcomings. Additionally, ODE-FR allows the insertion of function modules
   620	at the same depth to guide feature reconstruction, making
   621	network design more convenient.
   622	In practice, the third-order implicit adam solution formula
   623	is:
   624	
   625	h
   626	yn+2 = yn+1 +
   627	5 fn+2 + 8 fn+1 − fn ,
   628	(12)
   629	12
   630	
   631	Fig. 5. Diagram of feature fusion, feature cascade and our proposed ODE-FR.
   632	
   633	where y∗ is the output at step *, h is the step size and
   634	ft is the function containing the relationship between y and
   635	x obtained from the ordinary differential equation, that is,
   636	ft = ft (x, y). From the dynamic systems perspective, the
   637	convolutional neural network defines a mapping that receives
   638	the first unit time of the input state in the phase space. The time
   639	range corresponds to the layer that can be selected adaptively,
   640	and t indicates the chronological order. In the context of
   641	feature reconstruction, we define ft = yt − xt as the feature
   642	residual, which represents the state transition from current
   643	feature xt to target feature yt , capturing both the direction and
   644	magnitude of correction needed. This formulation enables us
   645	to model feature reconstruction as an ODE numerical solution
   646	problem and leverage the stability of the third-order implicit
   647	Adams method. Therefore, with step size h = 1, the third-order
   648	implicit Adams solution formula can be rewritten as:
   649	
   650	5
   651	yn+2 = yn+1 +
   652	yn+2 − xn+2
   653	12
   654	 1
   655	3
   656	(yn − xn ) ,
   657	+
   658	yn+1 − xn+1 −
   659	(13)
   660	4
   661	12
   662	where the output of the previous level of the feature reconstruction network is the input of the latter level, that is,
   663	yn−1 = xn , yn+1 = xn+2 , and the formula is rewritten as:
   664	5
   665	16
   666	10
   667	1
   668	yn+2 + yn+1 − yn + yn−1
   669	12
   670	12
   671	12
   672	12
   673	16
   674	10
   675	1
   676	=
   677	yn+1 − yn + yn−1
   678	(14)
   679	7
   680	7
   681	7
   682	Because ∆yn+1 = yn+1 − yn , ∆yn = yn − yn−1 , the formula is
   683	rewritten as:
   684	 1
   685	9
   686	yn+2 = yn+1 +
   687	yn+1 − yn − (yn − yn−1 ) .
   688	(15)
   689	7
   690	7
   691	Based on the above formula, our ODE-FR module is
   692	designed and its structure is shown in Fig. 5, consisting of
   693	two groups of Conv layers and PReLU layers. Therefore,
   694	yn+2 =
   695	
   696	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:00 UTC from IEEE Xplore. Restrictions apply.
   697	
   698	5614
   699	
   700	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 4, APRIL 2026
   701	
   702	on the one hand, when the convolutional layer does not
   703	change the number of channels, the ODE-FR is relatively
   704	free and can be plugged and played. At the same time, the
   705	PReLU layer can decouple features and solve the problem of
   706	overfitting. On the other hand, the residual structure guided
   707	by the third-order implicit adam solution formula of ordinary
   708	differential equation can reconstruct features and give a new
   709	way to shape image features. In general, the ODE-FR applies
   710	the third-order implicit adam solution formula of ordinary
   711	differential equation and designs the residual module based
   712	on this formula to reconstruct the feature.
   713	Discussion: The structure of traditional residual networks is
   714	yn = xn + F(xn , wn ), where xn is the input, wn represents the
   715	network parameters, and yn is the output. At the same time,
   716	the forward difference
   717	equation represents the first
   718	
   719	 derivative
   720	
   721	yn+1 −yn
   722	dy
   723	dy
   724	as follows: dx = ∆x , that is: yn+1 = yn +∆x dx
   725	. Then
   726	n
   727	n 
   728	dy
   729	we can express dx in terms of F (xn , yn ) and ∆x in terms of
   730	n
   731	step size p, yn+1 can be rewritten as yn+1 = yn + pF (xn , yn ).
   732	From the above equation, it can be seen that the difference
   733	expression for the first derivative shares the same form as the
   734	structural expression of the residual network. This enables us
   735	to map the forward difference equation to the residual block,
   736	thus facilitating the design of the residual network architecture.
   737	It is important to note that traditional residual networks correspond to first-order methods in numerical ordinary differential
   738	equations, while higher-order methods can achieve smaller
   739	local truncation errors [13], offering certain advantages in
   740	obtaining more accurate solutions.
   741	IV. E XPERIMENTS
   742	A. Datasets and Evaluation Protocal
   743	We utilize Market-1501 [49], DukeMTMC-reID [50] and
   744	MSMT17 [51] for Re-ID tasks. Market-1501 is collected by
   745	6 cameras and contains 1,501 pedestrians; there are 12,936
   746	images of 751 pedestrians used for training, 19,732 images of
   747	750 pedestrians used for testing. DukeMTMC-reID is a subset
   748	of DukeMTMC, which is captured by 8 cameras. This dataset
   749	is split into 3 partitions, i.e., 16,522 images of 702 identities
   750	used for training, 17,661 images of 702 identities used for
   751	testing and 2,228 images used as query images. MSMT17 is
   752	the largest dataset in the Re-ID community which contains
   753	4,101 and 126,411 images captured with 15 cameras.
   754	To evaluate the performance of different methods in the
   755	unsupervised person Re-ID task, we conduct comparison
   756	and ablation experiments on the above three datasets. Mean
   757	average precision (mAP) and CMC rank-1, rank-5, rank-10
   758	accuracies are adopted to evaluate the methods’ performances.
   759	B. Implementation Details
   760	The backbone network used by our algorithm to extract

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf' - | nl -ba | sed -n '70,180p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    70	increasing research has focused on unsupervised person Re-ID,
    71	which learns directly from unlabeled data, exploring potential
    72	feature representations. This technique is more scalable and
    73	has more significant potential for deployment in the real
    74	world.
    75	Existing unsupervised person Re-ID faces two core challenges, which together constitute the motivation for our
    76	proposed method. First, insufficient feature representation
    77	capability: although existing deep networks (e.g., ResNet)
    78	can extract multi-level features, there is a lack of effective
    79	fusion mechanisms among different hierarchical feature information, and traditional residual connections lack mathematical
    80	modeling of the feature evolution process. Second, lack of
    81	optimization mechanisms for cluster boundary regions: in
    82	the iterative training process of existing methods, boundary
    83	hard samples often receive unstable pseudo-labels, and these
    84	errors accumulate and amplify in subsequent training, leading
    85	to insufficient inter-class separability and degraded model
    86	performance.
    87	Recently, many advanced methods [7], [8] utilize memory
    88	banks to store all instance features, treating each image as an
    89	independent class and learning representations by matching
    90	features of the same instance across different augmented
    91	views. However, in person Re-ID datasets, each class typically
    92	contains multiple positive instances. The SpCL [9] method
    93	alleviates this issue by matching instances with multiple centroids of positive values. To further ensure that each positive
    94	value converges to its centroid at a uniform rate, the cluster
    95	contrast learning method [10] updates the memory bank and
    96	computes the contrastive loss at the cluster level, making the
    97	contrastive loss the most suitable metric learning algorithm for
    98	updating memory bank features.
    99	Although cluster contrast learning achieves impressive performance, applying contrast learning only at the cluster level
   100	does not consider the relationships among difficult samples at
   101	the instance level. Furthermore, previous deep metric learning works focus on difficult sample mining to place greater
   102	emphasis on hard samples within a class. These methods
   103	aim to distinguish between different classes of samples and
   104	bring samples of the same class closer together. However,
   105	
   106	1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
   107	similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
   108	See https://www.ieee.org/publications/rights/index.html for more information.
   109	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:00 UTC from IEEE Xplore. Restrictions apply.
   110	
   111	YANG et al.: NNSC AND ODE GUIDED FEATURE RECONSTRUCTION FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   112	
   113	Fig. 1. Examples of confusing images for the unsupervised person Re-ID
   114	task.
   115	
   116	these methods often employ small batch-based depth measurement losses, such as hard triplet losses [11] and multiple
   117	similarity losses [12], which utilize only a small subset of
   118	data and do not account for information across all classes,
   119	and they are incompatible with memory bank architectures
   120	designed to ensure consistency across the entire dataset. As
   121	shown in Fig. 1, appropriate constraints are needed between
   122	easily confused sample classes due to various reasons such
   123	as pedestrian occlusion or similar clothing. At the same time,
   124	features with rich representation capabilities can refine pseudolabels, thereby improving the accuracy of model training.
   125	Therefore, we also need to find ways to enhance the model’s
   126	representation ability.
   127	To solve these problems, we propose an unsupervised
   128	person Re-ID method based on nearest neighbor sample
   129	constraint and ordinary differential equation guided feature
   130	reconstruction (NNSC-FR) to improve the clustering accuracy
   131	and pseudo-label quality while enhancing the representation of
   132	features. Specifically, we propose a nearest neighbor constraint
   133	(NNSC) based relative distance, which explicitly identifies
   134	hard samples located near decision boundaries by computing
   135	sample pair distances. NNSC enables the model to focus
   136	computational resources on the most informative samples
   137	rather than treating all samples uniformly. Based on NNSC,
   138	we also design an inter-class balance loss (CB loss), which
   139	works together to classify the hard samples between the
   140	neighboring domains and distinguish the two clusters to the
   141	greatest extent by forcing feature vectors to escape from the
   142	uniform classification distribution state. In addition, works on
   143	the design of residual structures guided by the solution of
   144	ODEs [13], [14], [15] have shown that it can significantly
   145	help the representation and generalization of the model. Thus,
   146	we apply the implicit third-order adam solution of ODEs to
   147	design a suitable residual structure guided by the solution of
   148	differential equations in numerical form. Different from the
   149	methods of mining encoder features at various depths, such
   150	as feature fusion and cascade, the residual structure can be
   151	inserted at any depth of the encoder, and function modules
   152	can be designed according to different task characteristics,
   153	so we call it ODE guided feature reconstruction (ODE-FR)
   154	
   155	5609
   156	
   157	module. Meanwhile, we insert a simple PReLU layer [16] and
   158	a Conv layer to reconstruct the backbone network’s features
   159	appropriately. Combined with the anti-degradation ability of
   160	the residual structure, we improve the feature characterization
   161	ability of the unsupervised model.
   162	Overall, we improve the unsupervised person Re-ID model’s
   163	performance from two perspectives: improving clustering
   164	accuracy and feature representation capability. Our contribution can be summarized as follows:
   165	• We introduce the nearest neighbor sample constraint
   166	strategy (NNSC), which aims to enhance clustering accuracy by mining positive samples within the same class
   167	and negative samples between neighboring classes at the
   168	instance level.
   169	• We design the inter-class balance loss (CB loss) better to
   170	identify the hard samples between the nearest neighbor
   171	classes and improve the clustering accuracy and pseudolabel quality.
   172	• We incorporate an ODE-guided feature reconstruction
   173	(ODE-FR) module, which leverages the implicit thirdorder Adam solution of an ordinary differential equation
   174	to create the residual structure of the architecture and
   175	improve the model representation ability.
   176	• Extensive experiments verify that our proposed NNSC-FR
   177	is superior to existing unsupervised person Re-ID methods on Market-1501, DukeMTMC-reID, and MSMT17
   178	datasets.
   179	II. R ELATED W ORK
   180	A. Unsupervised Person Re-Identification

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Occluded person Re-Identification with noise injection.pdf' - | nl -ba | sed -n '90,460p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    90	
    91	(b)
    92	
    93	(c)
    94	
    95	(d)
    96	
    97	Fig. 2. The visualization of data. (a) Person, (b) Inter-person occlusion, (c) Vehicle-induced occlusion, (d) Our synthetic Vehicle-Person.
    98	
    99	which can be readily collected via automated license plate recognition
   100	systems, offer two principal advantages: First, their inherent visual
   101	diversity (e.g., in color and texture) encourages models to focus on discriminative pedestrian features, reducing dependence on global apparel
   102	patterns and mitigating overfitting. Second, vehicle images captured
   103	from varying camera viewpoints introduce additional camera bias and
   104	domain-specific variations, thereby improving model generalization.
   105	Consequently, we incorporate vehicle images into the training process
   106	for occluded person ReID, which can also be interpreted as a form of
   107	noisy learning. We systematically examine their dual function as both
   108	a regularizer and a semantic enhancer.
   109	Our approach consists of two key components: (1) Epoch-wise
   110	Noise Injection (ENI) strategy, which periodically alternates the training dataset between person and vehicle images, serving as a regularizer
   111	to prevent overfitting. (2) Image-wise Noise Injection (INI) strategy, which synthesizes occluded person images by inserting vehicle
   112	segments to create physically plausible occlusion patterns, thereby
   113	enriching the semantic content of the training data. As shown in Fig.
   114	1, based on the above strategies, we could improve the performance
   115	significantly. Besides, we visualize several training images in Fig. 2. Our
   116	synthetic images are very similar to person images with vehicle-induced
   117	occlusion.
   118	
   119	Our contributions can be summarized as follows:
   120	• We demonstrate that vehicle data is not merely noise but also
   121	contributes positively to the training process in occluded person
   122	ReID.
   123	• We propose Epoch-wise Noise Injection and Image-wise Noise
   124	Injection strategies, which inject controlled noise to bridge the domain gap between vehicle and pedestrian images while avoiding
   125	overfitting.
   126	• Extensive experiments on two occluded person ReID datasets and
   127	four standard person ReID benchmarks show that incorporating vehicle data yields significant performance improvements,
   128	even surpassing state-of-the-art methods that rely on manual
   129	annotations.
   130	2. Related works
   131	2.1. Person ReID
   132	Early studies on convolutional neural networks (CNNs) established
   133	robust representation learning frameworks for person ReID, achieving
   134	2
   135	
   136	Pattern Recognition 179 (2026) 113689
   137	
   138	C. Yao et al.
   139	
   140	significant progress through global descriptor learning [4] and local
   141	feature alignment strategies. Although these methods perform well in
   142	holistic scenarios, they suffer from inherent limitations in generalization and efficiency. For instance, rigid partitioning strategies (e.g., fixed
   143	stripes in MPPM [5]) assume consistent spatial alignment of body parts
   144	across viewpoints, leading to misalignment when parts are occluded,
   145	deformed, or inconsistently visible. To mitigate these issues, poseguided methods have been proposed, leveraging keypoints or semantic
   146	parsing for improved part alignment. However, such approaches introduce new challenges: auxiliary modules (e.g., pose estimators in
   147	HOReID [6] and co-parsing networks in HCGA [7]) require additional manual annotations and increase inference latency compared to
   148	standard CNNs.
   149	
   150	the concept of ‘‘Positive-incentive noise’’ (𝜋-noise), which can reduce
   151	task complexity under certain conditions. This framework demonstrates
   152	that even random noise can benefit learning by simplifying the task
   153	entropy when applied appropriately.
   154	In computer vision, this principle manifests in various forms. Adversarial training deliberately introduces worst-case perturbations to
   155	enhance model robustness against distribution shifts. AutoAugment
   156	and RandAugment search for optimal noise policies that expand the
   157	effective training distribution without compromising label integrity.
   158	These methods validate that controlled noise injection can improve
   159	generalization beyond traditional data augmentation.
   160	Cross-domain learning presents a compelling application of noise
   161	principles. While domain gaps are traditionally considered harmful
   162	[20], recent studies demonstrate that they can be beneficial under appropriate constraints. For instance, progressive domain adaptation [21]
   163	reveals that incrementally introducing target domain characteristics
   164	as ‘‘noise’’ facilitates the learning of domain-invariant features. Similarly, CleanAdapt [22] treats source-free video domain adaptation as
   165	a noisy-label learning problem and improves training through reliable pseudo-label refinement. These studies suggest that structured
   166	cross-domain variation, when properly controlled, can act as useful regularization rather than pure interference. This insight is also relevant to
   167	person re-identification. Recent work has shown that background clutter, occlusion, and model imperfections can introduce substantial noise
   168	into person representations. For example, Pose2ID [23] reduces such
   169	noise through feature centralization and identity-preserving generation,
   170	improving representation stability across standard, cross-modality, and
   171	occluded ReID settings. In contrast, our method introduces structured
   172	cross-domain noise directly during training, allowing the network to
   173	learn occlusion-robust representations from augmented observations.
   174	Our approach extends these insights by treating vehicle images as
   175	structured noise for person ReID. Unlike random perturbations, vehicle
   176	data provides semantic consistency through license plate associations
   177	while introducing appearance variations that enhance occlusion robustness. The person–vehicle copy–paste strategy bridges domain gaps
   178	by creating hybrid samples that preserve identity information while
   179	diversifying visual patterns. This represents a novel instantiation of
   180	𝜋-noise theory, where cross-domain data serves as constructive noise
   181	that reduces the complexity of occluded person ReID without requiring
   182	additional human annotations.
   183	
   184	2.2. Occluded person ReID
   185	The introduction of Vision Transformers marked a turning point in
   186	handling occlusion challenges. TransReID [8] pioneered this direction
   187	by treating images as patch sequences and leveraging self-attention to
   188	capture long-range dependencies. Building upon this foundation, subsequent works have explored various strategies to enhance occlusion
   189	robustness while addressing computational efficiency.
   190	DRL-Net [9] decouples identity-relevant features from occlusion
   191	patterns through disentangled representation learning. By introducing
   192	semantic preference queries, it learns to focus on discriminative regions without explicit part annotations. Data-centric approaches offer
   193	complementary solutions to architectural innovations. SPT [10] synthesizes realistic occlusion scenarios by transferring salient person regions
   194	onto diverse backgrounds, creating a rich augmentation pipeline. The
   195	method employs IoU-based filtering to ensure occlusion realism while
   196	maintaining identity consistency. CAAO [11] advances this concept
   197	through reinforcement learning, where an occlusion controller learns
   198	optimal augmentation policies based on model feedback. The controller
   199	adapts occlusion patterns to current model weaknesses, creating a
   200	curriculum learning effect.
   201	Recent studies have further advanced occluded person ReID from
   202	several complementary perspectives. Cross-erasure [12] enhanced network improves robustness by strengthening feature learning under
   203	partial visibility and occlusion-induced corruption. Occlusion-attribute
   204	based methods [13] explicitly exploit occlusion cues to guide identity learning under incomplete observations. Feature complement and
   205	dual-attention based approaches [14] enhance the interaction between
   206	visible body cues and attention refinement, improving discriminability
   207	when informative regions are partially missing. In addition, maskaware hierarchical transformer architectures [15] incorporate mask semantics to emphasize non-occluded body parts and suppress corrupted
   208	regions.
   209	Despite their effectiveness, these recent methods still mainly rely
   210	on explicit occlusion modeling, auxiliary cues, or more sophisticated
   211	architectures. In contrast, our method improves occluded ReID from a
   212	data-centric perspective by injecting structured cross-domain noise during training, without requiring extra manual annotations or specialized
   213	occlusion masks.
   214	
   215	3. Proposed approach
   216	As illustrated in Fig. 3, our proposed framework addresses the
   217	challenge of occluded person ReID through a noise injection strategy
   218	based on a dual-branch architecture [5]. The core innovation lies in
   219	reconceptualizing vehicle data not as mere noise but as a valuable
   220	training resource in occluded scenarios.
   221	We think that controlled noise injection during training improves
   222	model robustness and generalization. Specifically, we leverage largescale vehicle datasets, readily available from automated license plate
   223	recognition systems, to augment conventional person ReID training.
   224	The training begins by preparing two distinct data sources: person
   225	images from standard ReID datasets and vehicle images with consistent
   226	license plate associations. Our noise injection strategy dynamically
   227	modulates input composition, alternating between pure vehicle training
   228	phases and mixed person-vehicle phases. This approach serves a dual
   229	purpose: it regularizes the model to prevent overfitting to specific appearance patterns while enriching semantic diversity through realistic
   230	occlusion simulation.
   231	The extracted features from both branches are fused into a comprehensive representation that captures both holistic appearance and discriminative local patterns. This dual-representation approach ensures
   232	robust performance across diverse occlusion scenarios while maintaining high discriminative capability for person ReID.
   233	
   234	2.3. Noise data learning
   235	Noise-aware learning has been explored in different forms across
   236	representation learning and recognition tasks. Click Prediction [16]
   237	highlights early noise-aware representation learning. SFE-Net [17] provides a complementary view on robustness under sparse and noisy
   238	signals. GAGA [18] contextualizes our work from an adversarial robustness perspective. These studies collectively suggest that noise should
   239	not always be viewed as purely harmful, but can sometimes play a
   240	constructive role in improving representation quality and robustness.
   241	Recent theoretical work by Li [19] further challenges the conventional
   242	view that noise is always detrimental to learning systems. Li introduces
   243	3
   244	
   245	Pattern Recognition 179 (2026) 113689
   246	
   247	C. Yao et al.
   248	
   249	Fig. 3. Overview of the proposed NIReID. The framework utilizes vehicle images as positive noise to improve generalization in occluded person ReID by adopting
   250	two noise injection strategies: ENI, which alternates person and vehicle images during training, and INI, which generates occluded person images by inserting
   251	vehicle segments. The dual-branch network extracts both global and local features, which are aggregated using average and max pooling, and are supervised by
   252	a combination of identity and triplet losses.
   253	
   254	Algorithm 1: Epoch-wise Noise Injection
   255	
   256	Algorithm 2: Image-wise Noise Injection
   257	
   258	Data: Epoch 𝑁, Switching Interval 𝑆, Vehicle Images 𝐼𝑣 ,
   259	Person Images 𝐼𝑝 , Model Inputs 𝐼, Model 𝑀, Model
   260	Output 𝑂, Loss Function 𝐿𝐹
   261	Result: Loss Values 𝐿𝑖 , 𝑖 ∈ [1, 𝑁]
   262	1 initialization;
   263	2 for i=1, i≤ N do
   264	3
   265	if i % 𝑆 == 0 then
   266	4
   267	Select vehicle images 𝐼𝑣,𝑖 ;
   268	5
   269	𝐼𝑖 = 𝐼𝑣,𝑖 ;
   270	6
   271	else
   272	7
   273	Select vehicle images 𝐼𝑣,𝑖 ;
   274	8
   275	Select person images 𝐼𝑝,𝑖 ;
   276	9
   277	Copy-Paste 𝐼𝑣,𝑖 and 𝐼𝑝,𝑖 to produce 𝐼𝑖 ;
   278	10
   279	end
   280	11
   281	𝑂𝑖 = 𝑀(𝐼𝑖 );
   282	12
   283	𝐿𝑖 = 𝐿𝐹 (𝑂𝑖 );
   284	13
   285	Back propagate 𝐿𝑖 ;
   286	14 end
   287	
   288	Data: Vehicle Images 𝐼𝑣 , Cropped Vehicle Image 𝐶𝑣 , Person
   289	Images 𝐼𝑝 , Person Images Number 𝑁, Person Image
   290	Height 𝐻, Person Image Width 𝑊 , Random Height Ratio
   291	𝑟, Top Left Coordinates (𝑥, 𝑦),
   292	Result: Person-Vehicle Mixed Images 𝐼𝑝𝑣
   293	1 initialization;
   294	2 for i=1, i≤ N do
   295	3
   296	Get person image 𝐼𝑝,𝑖 from 𝐼𝑝 ;
   297	4
   298	𝐼𝑝𝑣,𝑖 = 𝐼𝑝,𝑖 ;
   299	5
   300	Randomly select a vehicle image 𝐼𝑣′ from vehicle images 𝐼𝑣 ;
   301	6
   302	Randomly produce coordinates
   303	(𝑥, 𝑦), 𝑥 ∈ [0, 𝑊 ], 𝑦 ∈ [𝑟 ⋅ 𝐻, 𝐻];
   304	7
   305	𝐶𝑣 = 𝐼𝑣′ [∶ 𝐻 − 𝑦, ∶ 𝑊 − 𝑥];
   306	8
   307	𝐼𝑝𝑣,𝑖 [𝑦 ∶ 𝐻, 𝑥 ∶ 𝑊 ] = 𝐶𝑣 ;
   308	9 end
   309	
   310	from varying viewpoints. This exposure to diverse visual contexts improves the model’s ability to handle domain shifts and camera variations commonly encountered in practical ReID scenarios. Furthermore, the periodic alternation prevents the model from becoming
   311	overly specialized to either domain, maintaining a balanced learning
   312	trajectory.
   313	
   314	3.1. Noise injection strategy
   315	Unlike conventional methods that rely solely on person datasets, we
   316	use two noise injection strategies to enhance model training. They enable the network to extract discriminative features from the large-scale
   317	vehicle dataset while preserving its focus on the primary person ReID
   318	task. By treating vehicle data as controlled noise, we enrich feature
   319	representations and enhance generalization, particularly in occluded
   320	scenarios.
   321	
   322	3.1.2. Image-wise noise injection
   323	The Image-wise Noise Injection (INI) strategy focuses on synthesizing physically plausible occluded person images by strategically
   324	inserting vehicle segments into person images. Unlike ENI operating
   325	at the epoch level, INI performs augmentation at the instance image
   326	level, creating a finer-grained control over occlusion patterns.
   327	As outlined in Algorithm 2, INI begins by randomly selecting vehicle
   328	images 𝐼𝑣′ from the vehicle dataset for each person image 𝐼𝑝,𝑖 . The
   329	algorithm then generates random coordinates (𝑥, 𝑦) where 𝑥 ∈ [0, 𝑊 ]
   330	and 𝑦 ∈ [𝑟 ⋅ 𝐻, 𝐻], with 𝑟 representing a height ratio parameter that
   331	controls the vertical placement of the vehicle image. This constraint
   332	ensures that vehicle occlusions primarily affect the lower portions of
   333	person images, mimicking realistic occlusions where vehicles typically
   334	obstruct the lower body regions of pedestrians. The cropping operation
   335	𝐶𝑣 = 𝐼𝑣′ [∶ 𝐻 − 𝑦, ∶ 𝑊 − 𝑥] extracts appropriately sized vehicle segments.
   336	Subsequently, the insertion operation 𝐼𝑝𝑣,𝑖 [𝑦 ∶ 𝐻, 𝑥 ∶ 𝑊 ] = 𝐶𝑣 replaces
   337	the corresponding region in the person image with the vehicle segment,
   338	creating a seamlessly integrated occluded person image.
   339	The INI strategy presents several advantages compared to traditional occlusion simulation methods. First, it produces realistic occlusions that closely mimic real-world surveillance scenarios, where
   340	vehicles often obstruct pedestrians. Second, incorporating real vehicle
   341	images provides authentic texture and color variations, which synthetic
   342	occlusion methods fail to replicate accurately. Third, randomized positioning ensures diverse occlusion patterns across training samples,
   343	mitigating the risk of the model developing position-specific biases.
   344	
   345	3.1.1. Epoch-wise noise injection
   346	The Epoch-wise Noise Injection (ENI) strategy implements a temporal alternation that periodically switches the training dataset between
   347	person and vehicle images.
   348	As detailed in Algorithm 1, the ENI process operates with a predefined switching interval 𝑆 that determines the frequency of dataset
   349	alternation. During vehicle-only epochs, the model trains exclusively
   350	on vehicle images 𝐼𝑣,𝑖 , forcing the network to extract features that are
   351	not dependent on human-specific visual cues. This constraint compels
   352	the model to develop more robust and transferable feature extraction
   353	capabilities. In contrast, during mixed epochs, the training process
   354	combines both person 𝐼𝑝,𝑖 and vehicle images 𝐼𝑣,𝑖 through a copy–paste
   355	process. This combination creates a diverse training environment where
   356	the model must discriminate between different types of visual content
   357	while maintaining its ReID capabilities. This operation strategically
   358	places cropped vehicle images within person images, creating realistic
   359	occlusion scenarios that mirror real-world surveillance conditions.
   360	The key advantage of ENI lies in its ability to introduce domainspecific variations and camera bias through vehicle images captured
   361	4
   362	
   363	Pattern Recognition 179 (2026) 113689
   364	
   365	C. Yao et al.
   366	
   367	aggregation. After passing through the branch-specific feature embeddings, the feature maps are processed by both global average pooling
   368	and global max pooling. The global average pooling captures the
   369	overall statistical distribution of features across the entire spatial extent, providing a compact representation of the complete appearance.
   370	Complementarily, the global max pooling extracts the most discriminative features from each channel, highlighting the most salient visual
   371	elements regardless of their spatial location. The combination of these
   372	two pooling strategies ensures that the coarse-grained branch captures
   373	both the general appearance distribution and the most distinctive visual
   374	characteristics.
   375	
   376	3.1.3. Comparison with other methods
   377	We also compare our INI and ENI with several augmentations and
   378	strategies.
   379	Comparison with MixUp. MixUp [24] enhances model robustness by
   380	generating novel training images from the combination of two labeled
   381	inputs, which reduces memorization of corrupt labels and improves
   382	resistance to adversarial examples. However, this method requires that
   383	both source images have clean labels, a requirement incompatible with
   384	our unlabeled vehicle images in the INI strategy.
   385	Comparison with CutMix. Like MixUp, the CutMix [25] requires a
   386	label for the vehicle image, which the INI strategy cannot provide.
   387	Consequently, CutMix without a vehicle label can be viewed as a generalized form of INI, but breaks spatial priors. For instance, a pedestrian
   388	image where only the upper body is occluded by a vehicle does not
   389	represent a realistic scenario.
   390	
   391	Fine-grained Branch: The fine-grained branch emphasizes part-based
   392	feature extraction through spatial decomposition and localized pooling
   393	operations. After the shared branch-specific embedding module, the
   394	feature map is processed along two parallel paths: one path applies
   395	global average pooling directly to the feature map to retain global context, while the other path first performs spatial partitioning, followed
   396	by global max pooling within each partition. This spatial partitioning
   397	allows the model to focus on specific anatomical regions, while the
   398	subsequent max pooling identifies the most discriminative features
   399	within each region. This combination ensures that the fine-grained
   400	branch captures both global and part-based peak responses, resulting
   401	in a comprehensive representation of person-specific details.
   402	The final feature representation combines outputs from both
   403	branches, leveraging the complementary nature of global holistic features and local part-based features. This dual-representation approach
   404	ensures robust performance across various scenarios, particularly in
   405	occluded conditions where local features may be compromised but
   406	global context remains informative, or vice versa.
   407	
   408	Comparison with Hide-and-Seek. The Hide-and-Seek method [26]
   409	randomly hides patches in a training image, thereby forcing the network to rely on other relevant features when the most discriminative
   410	one is hidden. Our approach, INI, replaces these hidden patches with
   411	vehicle images. This substitution provides richer semantic information,
   412	resulting in a more discriminative model.
   413	Comparison with GridMask. GridMask [27] is a variant of Hide-andSeek, and it similarly removes regions, thereby disrupting semantic information. In contrast, our INI approach better preserves this semantic
   414	content.
   415	Comparison with domain-perturbation. Most existing domain perturbation methods rely on auxiliary modules, such as multi-layer perceptrons (MLPs) [28] or generative adversarial networks (GANs) [29],
   416	which inevitably increase computational load. In contrast, our proposed
   417	ENI and INI methods operate solely by adjusting the input image
   418	and introduce no new modules. Consequently, it can be seamlessly
   419	integrated into any training framework.
   420	
   421	3.3. Feature learning and optimization
   422	The final feature representation 𝐹final is obtained by concatenating
   423	global features 𝐹global and local features 𝐹local . This combined representation captures both holistic appearance patterns and detailed local
   424	characteristics, providing comprehensive discriminative information
   425	for person ReID.
   426	To optimize the network, we employ a multi-loss training strategy.
   427	Cross-entropy loss supervises identity classification for both global and
   428	local features, ensuring both branches contribute to identity discrimination. In contrast, the triplet loss with hard negative mining is applied
   429	exclusively to the global features, further enhancing the discriminative
   430	capability of the global representation in the embedding space. The
   431	total loss function is formulated as:
   432	(
   433	)
   434	(
   435	)
   436	total = id 𝐹final + 𝜆tri 𝐹global
   437	(1)
   438	
   439	Comparison with noise-injection. Like domain perturbation, many
   440	noise injection techniques also depend on complex components like
   441	neural architecture search (NAS) or GANs [30]. Both our INI and ENI
   442	methods, similar to RandAugment, function primarily as data augmentation strategies. Besides, our strategies are well-designed for the
   443	occluded person re-identification, thus are more suitable than universal
   444	AutoAugment and RandAugment.
   445	Comparison with curriculum learning. The curriculum learning typically follows an easy-to-hard schedule. ENI adopts a nonmonotonic,
   446	periodic alternation between person-only and vehicle-only (or mixed)
   447	training phases.
   448	Comparison with domain alternation. The standard domain alternation or multi-domain training assumes semantic consistency across
   449	domains, whereas ENI deliberately introduces semantically mismatched
   450	but structurally correlated data (vehicles vs. pedestrians) as positive
   451	noise.
   452	
   453	where 𝜆 balances the contribution of the triplet loss.
   454	3.4. Theoretical perspective of positive noise
   455	
   456	3.2. Network architecture
   457	
   458	The proposed noise-injection re-identification framework can be
   459	interpreted from an information-theoretic perspective inspired by the
   460	concept of Positive-Incentive Noise (𝜋-noise) [19]. Let 𝑇 denote the

codex
第五篇的触发点比较实用：无监督 ReID 的伪标签错误主要堆在聚类边界，单纯按聚类中心做对比学习太粗，按小批次挖 hard sample 又看不到全局类别。第六篇把“车是干扰物”反过来讲成“有结构的正向噪声”，这是这一批里最值得学的叙事动作之一。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Occluded person Re-Identification with noise injection.pdf' - | nl -ba | sed -n '454,520p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   454	3.4. Theoretical perspective of positive noise
   455	
   456	3.2. Network architecture
   457	
   458	The proposed noise-injection re-identification framework can be
   459	interpreted from an information-theoretic perspective inspired by the
   460	concept of Positive-Incentive Noise (𝜋-noise) [19]. Let 𝑇 denote the
   461	re-identification task and 𝜀 the injected noise. The task entropy 𝐻(𝑇 )
   462	measures the uncertainty or complexity of the tasks. The mutual information between the task and the injected noise is given by
   463	
   464	The core network exclusively employs a CNN architecture as the
   465	backbone to extract robust feature representations. These features are
   466	subsequently processed by a dual-branch architecture that generates
   467	complementary visual representations.
   468	The two branches share identical initial weights from the pretrained backbone but diverge in their subsequent processing to learn
   469	distinct visual patterns. Each branch incorporates a convolutional layer
   470	followed by a convolutional head that refines the extracted representations. These components are designed to produce branch-specific
   471	feature embeddings that capture different aspects of the input image.
   472	
   473	𝑀𝐼(𝑇 , 𝜀) = 𝐻(𝑇 ) − 𝐻(𝑇 ∣ 𝜀).
   474	
   475	(2)
   476	
   477	When the injected noise satisfies 𝑀𝐼(𝑇 , 𝜀) > 0, or equivalently
   478	𝐻(𝑇 ∣ 𝜀) < 𝐻(𝑇 ).
   479	
   480	(3)
   481	
   482	It is regarded as positive-incentive noise, meaning that it simplifies the
   483	task by reducing uncertainty rather than introducing disturbance.
   484	
   485	Coarse-grained Branch: The coarse-grained branch focuses on capturing holistic appearance characteristics through comprehensive spatial
   486	5
   487	
   488	Pattern Recognition 179 (2026) 113689
   489	
   490	C. Yao et al.
   491	
   492	Table 1
   493	Datasets for Occlusion, General, and Noisy Vehicle ReID.
   494	Dataset
   495	
   496	Train set
   497	
   498	Query set
   499	
   500	Table 3
   501	Results on Occluded-Duke and P-DukeMTMC datasets. Best results in BOLD.
   502	Method
   503	
   504	Gallery set
   505	
   506	Number
   507	
   508	ID
   509	
   510	Number
   511	
   512	ID
   513	
   514	Number
   515	
   516	ID
   517	
   518	Occluded-Duke
   519	P-DukeMTMC
   520	

hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Occluded person Re-Identification with noise injection.pdf' - | nl -ba | sed -n '1280,1425p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1280	across all configurations, suggesting that enriching training data with
  1281	realistic occlusion patterns through vehicle segment insertion contributes more significantly to model performance than the regularization effect of epoch-wise alternation. The consistent improvement
  1282	patterns observed across different backbone architectures validate the
  1283	generalizability and robustness of our proposed noise injection strategies.
  1284	
  1285	To comprehensively evaluate the effectiveness of our proposed approach, we conduct ablation studies on the Occluded-Duke dataset [3]
  1286	to analyze the individual contributions of each component and investigate the optimal hyperparameter settings.
  1287	
  1288	4.4. Results on person ReID
  1289	To validate the generalizability of our approach beyond occluded
  1290	scenarios, we conduct additional experiments on four standard person
  1291	ReID benchmarks: Market-1501 [31], DukeMTMC-ReID [32], CUHK03
  1292	[33], and MSMT17 [2].
  1293	Our method demonstrates substantial improvements over existing
  1294	approaches, as shown in Table 5. On Market-1501, NIReID+R-1.6G
  1295	surpasses the previous best method FMGNet [46] by 0.5% Rank-1
  1296	and 1.4% mAP. Notably, even our lightweight version NIReID+R50
  1297	achieves 96.1% Rank-1 accuracy, outperforming all existing state-ofthe-art methods. On DukeMTMC-ReID, we achieve comparable Rank-1
  1298	performance to OAT [54] while improving mAP by 0.2%.
  1299	Table 6 shows significant advantages in both labeled and detected
  1300	settings. In the detected setting, NIReID+R-1.6G surpasses the previous
  1301	best method, APD [47] by 2.0% Rank-1. In the labeled setting, we
  1302	achieve even more substantial improvements, outperforming DAA [48]
  1303	by 3.0% in Rank-1 accuracy and 2.8% in mAP.
  1304	On MSMT17, this challenging large-scale dataset shows that
  1305	NIReID+R-1.6G surpasses the previous best method, TransReID [8] by
  1306	3.1% in both Rank-1 accuracy and mAP (see Table 7). Remarkably,
  1307	our NIReID+R50 also outperforms TransReID [8] by 1.0% in Rank-1
  1308	accuracy, demonstrating that our noise injection strategies can achieve
  1309	competitive performance even with simpler backbone architectures.
  1310	
  1311	Hyperparameter Analysis for ENI. The epoch interval in ENI determines how frequently the training alternates between person and
  1312	vehicle datasets. We systematically investigate different interval values
  1313	ranging from 3 to 30 epochs, as shown in Table 8.
  1314	7
  1315	
  1316	Pattern Recognition 179 (2026) 113689
  1317	
  1318	C. Yao et al.
  1319	
  1320	Fig. 4. Ablation study on two noisy injection strategies.
  1321	Table 10
  1322	Ablation study on INI and other augmentations.
  1323	
  1324	Table 12
  1325	Ablation study on Pooling Type.
  1326	
  1327	Method
  1328	
  1329	Rank-1
  1330	
  1331	Rank-5
  1332	
  1333	Rank-10
  1334	
  1335	mAP
  1336	
  1337	–
  1338	
  1339	62.3
  1340	
  1341	76.4
  1342	
  1343	81.2
  1344	
  1345	53.1
  1346	
  1347	Branch-1
  1348	
  1349	INI
  1350	
  1351	69.1
  1352	
  1353	82.7
  1354	
  1355	86.7
  1356	
  1357	57.7
  1358	
  1359	Type-1
  1360	
  1361	Type-2
  1362	
  1363	Type-1
  1364	
  1365	Type-2
  1366	
  1367	INI (brightness = 0.1)
  1368	INI (brightness = 0.5)
  1369	INI (contrast = 0.1)
  1370	INI (contrast = 0.5)
  1371	INI (saturation = 0.1)
  1372	INI (saturation = 0.5)
  1373	
  1374	69.0
  1375	66.1
  1376	68.9
  1377	67.5
  1378	69.1
  1379	66.2
  1380	
  1381	82.8
  1382	81.0
  1383	82.7
  1384	81.2
  1385	82.6
  1386	80.7
  1387	
  1388	86.7
  1389	86.0
  1390	86.6
  1391	85.7
  1392	86.7
  1393	85.6
  1394	
  1395	57.6
  1396	56.5
  1397	57.5
  1398	57.2
  1399	57.6
  1400	55.7
  1401	
  1402	Avg
  1403	
  1404	Max
  1405	
  1406	Avg
  1407	
  1408	Avg
  1409	Max
  1410	Avg
  1411	Avg
  1412	
  1413	Avg
  1414	Max
  1415	Max
  1416	Max
  1417	
  1418	Avg
  1419	Avg
  1420	Avg
  1421	Max
  1422	
  1423	MixUp (p = 0.1, 𝛼 = 0.1)
  1424	MixUp (p = 0.2, 𝛼 = 0.1)
  1425	MixUp (p = 0.3, 𝛼 = 0.1)

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf' - | nl -ba | sed -n '90,160p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    90	training an vanilla DG model (ERM) on 4 source domains,
    91	as shown in Fig. 1. From Fig. 1(a), we see that small-scale
    92	domain 1&2 converges faster than large-scale domain 3&4.
    93	With further iterations, as is shown in Fig. 1(b), overfitting
    94	happens and the performances of domain 1&2 and the open
    95	test domain are degraded. This indicates the domain imbalance
    96	leads to a collapse risk of DG ReID model, which, however, is
    97	seriously overlooked. To solve this issue, an intuitive way is to
    98	re-weight the samples, a widely-used strategy in the long-tailed
    99	recognition task [17], [18], [19]. However, such re-weighting
   100	leads to inadequate learning of domain-invariant representation
   101	by rudely reducing importance of some domains. This is
   102	because each domain is valuable in domain-invariant learning,
   103	as a rather challenging task.
   104	
   105	1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
   106	similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
   107	See https://www.ieee.org/publications/rights/index.html for more information.
   108	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:29 UTC from IEEE Xplore. Restrictions apply.
   109	
   110	10892
   111	
   112	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 11, NOVEMBER 2025
   113	
   114	Fig. 1. An exploratory experiment by training a standard DG model on 4
   115	source domains, where domain 1&2 represent small-scale datasets and domain
   116	3&4 represent large-scale datasets.
   117	
   118	In this paper, we propose a Multi-model Synergy Perception (MSP) framework to construct balanced domains for
   119	exploring domain-invariant features. For this purpose, we
   120	first develop an Asynchronous Training Paradigm (ATP) that
   121	divides the biased source domains into multiple debiased
   122	subdomains and each subdomain has rich domain diversity.
   123	To fully exploit multiple subdomains for domain-invariant
   124	features learning, inspired by the philosophy of divide and
   125	conquer, we construct a multi-network architecture to learn all
   126	subdomains in parallel. Specifically, we use multiple parallel
   127	sub-models to learn different subdomains to exploit different
   128	batch knowledge of the large-scale domain, while sharing
   129	the same batch knowledge of the small-scale domain for
   130	knowledge asynchronous learning. To keep domain balance,
   131	different sub-models are independent of each other during
   132	training process. Based on the idea of multi-submodel training,
   133	another problem is faced: naively aggregating all parallel
   134	local sub-models as federated learning may lead to inferior
   135	asynchronous optimization due to lacking synergy perception
   136	among sub-models. Considering that the distance relationship
   137	of features can represent the retrieval model knowledge, we
   138	propose a Structure Synergy Perception (SSP) module that
   139	constructs the feature relationship distribution for each subdomain and aligns them to map the unique knowledge to
   140	each other for synergy learning. Specifically, the structure
   141	distribution is built based on the feature relationship of each
   142	subdomain and then we align structure distribution of different subdomains to better generalize knowledge across these
   143	subdomains.
   144	Additionally, each submodel can only exploit singleview generalizable knowledge, and the conventional multisubmodel training strategies are unable to complement the
   145	insufficient view with each other due to lacking interaction
   146	learning among sub-models. Considering the consistency of
   147	sub-source domains, we further propose a Synergy Distillation
   148	Perception (SDP) to aggregate views of different submodels to
   149	improve the model both semantic and domain generalization
   150	ability. The SDP primarily comprises semantic based cross-
   151	
   152	model central distillation (CCD) and part graph distillation
   153	(PGD). The main idea of SDP is to use the center guided
   154	soft label and the part based triplet graph to distill each
   155	subnetwork, which facilitates the network to explore domaininvariant representations of images. Specifically, considering
   156	the importance of logits in semantic classification, CCD first
   157	aggregates the output logits of different submodels to obtain
   158	multi-view knowledge. And then this value is used as guided
   159	soft labels to distill each submodel for semantic generalization
   160	learning. Furthermore, existing ReID loss functions primarily

codex
第六篇的方法页有双栏抽取错序，我按章节标题把 3.1、3.2、3.3、3.4 对齐读了。它的关键证据不只是 SOTA 表，而是消融里 INI 单独贡献大于 ENI，以及和 MixUp、CutMix、Hide-and-Seek、GridMask 等通用增广的区分。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '70,118p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    70	VIS and IR light differ, resulting in distinct sharpness and lighting conditions, which can produce
    71	very different results in the images. Therefore, a new paradigm of visible-infrared ReID (VIReID) [38] has emerged and has been rapidly deployed for video surveillance applications in
    72	recent years.
    73	VI-ReID is essential for surveillance and security systems, particularly in complex, dynamic
    74	environments. In scenarios such as public security monitoring [20], search and rescue operations,
    75	and night-time surveillance, VI-ReID enables consistent monitoring under variable lighting and
    76	weather conditions, allowing reliable identification even when VIS cameras lack detail under
    77	low-light conditions [41]. However, implementing VI-ReID in real-world applications remains
    78	challenging, especially due to occlusions, pose changes, and background shifts (illustrated in
    79	Figure 1) that hinder accurate cross-modal matching. Different from traditional single-modal
    80	methods, the fusion of multi-modal data requires finding a suitable method to bridge the gap
    81	between the two modalities [12]. These modal differences can lead to significant matching difficulties
    82	due to differences in imaging principles and environmental factors such as illumination conditions
    83	and sensor characteristics. Therefore, while VI-ReID holds great promise for improving security and
    84	surveillance, addressing challenges such as modality gaps, occlusion, and environmental variability
    85	is critical to achieving reliable performance in diverse, dynamic scenarios.
    86	To address the aforementioned challenges, we propose a multi-scale dynamic fusion network
    87	(MDFN), which comprises three modules: dynamic feature fusion (DFF), dynamic perceptual enhancement (DPE), and feature reweighting with similarity (FRS). The DFF module
    88	incorporates a local unit that uses adaptive group convolution (AGC) to capture local features,
    89	alongside a global unit enhanced with an improved multi-head self-attention mechanism [33].
    90	The DPE module employs deformable dilated convolution with various dilation rates to capture
    91	multi-scale spatial information and enhance feature embeddings through the modality-aware
    92	hierarchical contrast (MHC) loss. Additionally, to address inaccuracies in the query library affinity matrix, we introduce the FRS module, which optimizes matching performance by leveraging
    93	similarities between internal gallery features. By combining these modules, fine-grained features
    94	can be extracted efficiently to address dynamic challenges such as lighting variations, occlusion,
    95	pose changes, and complex background shifts.
    96	ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 101. Publication date: March 2025.
    97	
    98	Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification
    99	
   100	101:3
   101	
   102	Fig. 1. The challenges of VI-ReID: the samples (a), (b), and (c) provide three visible-infrared paired images of
   103	persons. Visible images suffer from occlusion, background shifts and pose changes, while infrared images
   104	yield low clarity and high noises. Developing an effective VI-ReID solution robust to these dynamic changes
   105	remains a challenge to be addressed.
   106	
   107	Our main contributions are summarized as follows:
   108	— To address the inherent challenges of complex dynamic scenes in VI-ReID, we propose an
   109	MDFN framework to improve feature extraction and enhancement, as well as modality
   110	alignment from VIS and IR images.
   111	— Our MDFN framework integrates the DFF module to extract fine-grained features by fusing
   112	the extracted local and long-range dependencies among features, the DPE module to enhance
   113	multi-scale spatial dependencies and feature representation, and the FRS module to generate
   114	accurate feature vectors for retrieval by utilizing higher-order similarity information in the
   115	query and gallery sets. Therefore, our MDFN framework is capable of tackling complex
   116	dynamic changes across VIS and IR modalities.
   117	— Extensive experiments demonstrate that our proposed MDFN outperforms the existing stateof-the-art VI-ReID methods overall in terms of the Rank-1 and mAP metrics on the commonly
   118	used SYSU-MM01 and RegDB datasets.

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '80,160p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    80	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:07:25 UTC from IEEE Xplore. Restrictions apply.
    81	© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
    82	but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
    83	
    84	This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
    85	content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3682328
    86	
    87	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
    88	
    89	matching difficult. Specifically, VIS light has a wavelength
    90	range of 400–700 nm, and is typically stored as RGB threechannel images in Re-ID tasks, providing rich color and
    91	texture information. In contrast, IR light ranges from 700
    92	nm to 1 mm, among which near-infrared (NIR, 700–1000
    93	nm) is most commonly used in Re-ID. NIR images are usually single-channel grayscale images that reflect the intensity
    94	of infrared radiation, lacking color information and certain
    95	fine details [19], [20]. Moreover, the inherent heterogeneity
    96	across modalities results in distinct feature distributions. While
    97	VIS images can focus on pedestrian regions with reduced
    98	background interference, IR images often suffer from blurred
    99	boundaries between pedestrians and background. This discrepancy leads to severe feature misalignment and erroneous
   100	matches in the latent feature space. In addition, VI-ReID faces
   101	intra-modality variations such as viewpoint, pose, clothing, and
   102	background changes, further increasing task complexity.
   103	Recently, numerous approaches have been proposed to address VI-ReID [21]–[25], which can generally be categorized
   104	into image-level and feature-level methods, as illustrated in
   105	Fig. 1. Image-level methods [15], [26] typically employ Generative Adversarial Network (GAN) to synthesize VIS images
   106	from IR inputs or vice versa, thereby bridging the modality
   107	gap. However, these methods require precisely aligned VIS–IR
   108	image pairs with consistent viewpoint, time, and resolution.
   109	Despite their promising results, image-level approaches are
   110	often computationally expensive, prone to noise generation,
   111	and limited in flexibility. Feature-level methods [27]–[30], in
   112	contrast, generally adopt dual-stream architectures to extract
   113	both modality-specific and modality-shared representations in
   114	the latent space. Owing to their higher efficiency and stability,
   115	feature-level techniques have become the dominant paradigm
   116	in recent VI-ReID research. Nevertheless, due to the large
   117	modality discrepancy and the unbounded nature of latent
   118	feature spaces, these methods generally capture only coarsegrained patterns (e.g., human contours and structures) and
   119	struggle to learn discriminative identity-consistent knowledge
   120	across modalities. Furthermore, as the network deepens, a
   121	substantial amount of discriminative semantic information may
   122	be lost, and the intrinsic modality gap continues to drive the
   123	model toward inconsistent cues, thereby hindering accurate
   124	cross-modal matching, as empirically observed in previous
   125	studies [31]–[33]. In essence, the large modality gap remains
   126	the primary bottleneck limiting VI-ReID performance, and
   127	effectively bridging this gap to discover cross-modal consistent
   128	information is still a major challenge.
   129	Based on these observations, we summarize the challenges
   130	of VI-ReID into two aspects: (i) difficulty in aligning lowand high-level semantics across modalities, and (ii) distribution discrepancies between VIS and IR features. To address
   131	these challenges, we propose a simple yet effective framework named Mutual Distillation driven Dual-Space Matching (MDDM). Specifically, for challenge (i), we design a
   132	Dual Level Fusion (DLF) module and a Modality Invariant
   133	Augmentation (MIA) module. DLF captures and adaptively
   134	fuses hierarchical features, aligning modalities by integrating
   135	both low- and high-level semantics across spatial and channel
   136	dimensions. MIA extracts fine-grained semantic cues and
   137	
   138	2
   139	
   140	enhances identity discrimination, thereby reinforcing VIS–IR
   141	correlations and enabling the learning of robust modalityshared representations. For challenge (ii), we propose a novel
   142	Dual-Space Matching (DSM), which aligns modalities in both
   143	Euclidean and Hilbert spaces at the pixel level. By mapping
   144	shared features into these two complementary spaces and
   145	employing a mutual distillation strategy, DSM encourages
   146	cross-space consistency, alleviates the unboundedness of latent spaces, and strengthens the representation capacity of
   147	shared features. Moreover, DSM guides the model to focus
   148	on consistent cues across spaces, which can be regarded as
   149	invariant, robust, and discriminative, thereby reducing modality discrepancies. In summary, the contributions of this paper
   150	are as follows:
   151	• We propose a novel Mutual Distillation driven DualSpace Matching (MDDM) framework, which performs
   152	pixel-level alignment in both Euclidean and Hilbert
   153	spaces to mitigate modality discrepancies in VI-ReID.
   154	• We design a Dual Level Fusion (DLF) to capture hierarchical semantics across channel and spatial dimensions,
   155	and a Modality Invariant Augmentation (MIA) module
   156	to learn discriminative modality-shared knowledge by
   157	establishing VIS–IR correlations.
   158	• Extensive experiments on SYSU-MM01, RegDB, and
   159	LLCM demonstrate the effectiveness of MDDM, achieving competitive performance compared with state-of-theart methods.
   160	II. R ELATED W ORK

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf' - | nl -ba | sed -n '160,205p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   160	learning. Furthermore, existing ReID loss functions primarily
   161	leverage the image-level visual similarities of IDs to explore
   162	the discriminative knowledge, which, however, is unable to
   163	eliminate the domain noise as this noise may exist in all
   164	features extracted from the same model. Considering the visual
   165	difference of different local parts, we develop a Parts Graph
   166	Distillation (PGD) that uses this valuable prior knowledge to
   167	explore the domain invariant features. The PGD divides the
   168	image into several local parts and leverages the parts visual
   169	similarities to construct the parts graph for each subdomain.
   170	And then we employ intra- and inter-subdomain distillation
   171	based on the parts graph to further improve the model domain
   172	generalization ability.
   173	Due to the limitation of small-scale domain, each sub-model
   174	can only acquire partial knowledge of large-scale domain.
   175	And different knowledge allows the sub-models to possess
   176	varying discriminative ability. To obtain the optimal final
   177	model, we propose an adaptive-weight method to aggregate
   178	all submodels during testing process. The testing images are
   179	augmented randomly and we employ contrastive loss function
   180	to compute its similarity, which serves as the weights for the
   181	corresponding submodel. And to improve the reliability of the
   182	learned weights, we augment the image with different ways
   183	and employ cross-computation among the augmented images
   184	for weighs learning.
   185	In summary, the contributions of this paper are three-fold.
   186	• We propose a Multi-model Synergy Perception (MSP)
   187	framework equipped with an Asynchronous Training
   188	Paradigm (ATP) for open-world ReID. With the philosophy of divide and conquer, we divide the biased
   189	source domains into multiple debiased subdomains and
   190	employ a multi-network architecture to learn these subdomains in parallel, which facilitates domain-invariant
   191	representations.
   192	• To reinforce the collaborative learning among submodels,
   193	we construct a Structure Synergy Perception (SSP) module based on the feature relationship. SSP constructs the
   194	feature relationships distribution of each subdomain and
   195	aligns them to map the unique knowledge to each other
   196	for synergy learning.
   197	• Since each submodel can only exploit single-view generalizable knowledge, we further propose a Synergy
   198	Distillation Perception (SDP) to aggregate views of different submodels. The SDP comprises semantic based
   199	cross-model central distillation (CCD) and parts graph
   200	distillation (PGD), which uses the center guided soft label
   201	and the part based triplet graph respectively to distill each
   202	subnetwork for improving the model both semantic and
   203	domain generalization ability.
   204	
   205	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:57:29 UTC from IEEE Xplore. Restrictions apply.

codex
方法主体已经读完。为了回答你要的“证据闭环”，我还要补看各文的消融和可视化标题，重点不是抄数字，而是看他们用什么实验让审稿人相信“问题重定义是对的”。
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
/bin/zsh -lc "pdftotext 'Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | rg -i 'ablation|visual|t-sne|sensitivity|component|DFF|DPE|FRS|MHC|OGM|Table|Fig\\.' | head -n 240" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    16	(DFF), Dynamic Perception Enhancement (DPE), and Feature Reweighting with Similarity (FRS) modules. The
    17	DFF module dynamically extracts local and long-range dependencies among features to obtain finer-grained
    18	discriminative features. The DPE module extracts multi-scale features from both visible and infrared modalities
    19	to generate diverse embeddings. The FRS module mitigates the impact of information imbalance between
    39	full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
    80	methods, the fusion of multi-modal data requires finding a suitable method to bridge the gap
    87	(MDFN), which comprises three modules: dynamic feature fusion (DFF), dynamic perceptual enhancement (DPE), and feature reweighting with similarity (FRS). The DFF module
    90	The DPE module employs deformable dilated convolution with various dilation rates to capture
    92	hierarchical contrast (MHC) loss. Additionally, to address inaccuracies in the query library affinity matrix, we introduce the FRS module, which optimizes matching performance by leveraging
   102	Fig. 1. The challenges of VI-ReID: the samples (a), (b), and (c) provide three visible-infrared paired images of
   111	— Our MDFN framework integrates the DFF module to extract fine-grained features by fusing
   112	the extracted local and long-range dependencies among features, the DPE module to enhance
   113	multi-scale spatial dependencies and feature representation, and the FRS module to generate
   167	Another notable approach is D2 RL [36], which uniquely separates appearance differences in traditional re-identification tasks from modality differences introduced by spectral imaging processes.
   215	IR for the IR branch. The MDFN comprises four main components: the backbone, the DFF module,
   216	the dynamic perception enhancement (DPE) module, and the FRS module. In this paper, we
   219	extraction. These extracted VIS-IR feature pairs are then input into the DFF module, where local
   221	features are subsequently fed into the DPE module, which captures contextual information at
   222	various scales and enriches the feature representation. Additionally, we design an MHC loss to
   224	the DPE module. Furthermore, we employ the orthogonal Gramian matrix loss (OGM loss) to
   229	DFF Module
   235	ability to learn representations effectively. To address this limitation, we propose a DFF module,
   236	which includes both an attention branch and a dynamic convolutional branch. The DFF module
   240	contextual information across larger scales [31]. This multi-scale fusion enables the DFF module
   243	global features, the DFF module strengthens cross-modal alignment, effectively bridging modality
   244	gaps and improving matching accuracy across VIS and IR images. Additionally, DFF achieves an
   246	applications requiring high performance. The overall workflow of the proposed DFF module is
   257	Fig. 2. Architecture of the proposed MDFN. (a) MDFN overview. The input VIS and IR images go through
   259	and local feature extraction are performed by LEMA (b) and AGC (c) in the DFF module, respectively. The
   260	fused features are fed to the DPE module for informative embedding generation by our proposed Deformable
   262	functions. The features are finally enhanced by our proposed FRS in matching phase.
   273	enhancing the model’s flexibility and expressiveness when handling diverse visual features. The
   274	AGC component employs a dynamic convolution kernel that is adjusted adaptively on the basis
   451	DPE Module
   458	[53], we propose a DPE module. The DPE module introduces a novel approach to capture multi-scale
   463	the DPE module employs the MHC loss to refine feature embeddings, which strengthens crossmodal alignment by reducing modality disparities at multiple feature levels. This combination of
   465	ensures that critical identity cues are preserved and aligned across modalities, making the DPE
   466	module a powerful component for enhancing cross-modal representation in dynamic environments.
   467	As illustrated in Figure 2, our proposed DPE module generates multiple embeddings and focuses
   469	branches. Specifically, each branch of the DPE module utilizes three parallel 3 × 3 deformable dilated
   488	weights are initialized via the Kaiming initialization method to maintain stable activation function
   524	Fig. 3. The proposed method FRS, primarily involves two summations: one for the original distances and
   527	top-k feature enhancement, FRS addresses the noise issue stemming from information disparities, resulting
   528	ˆ Furthermore, FRS demonstrates significantly improved accuracy in
   532	A ReLU activation layer is then used to improve the nonlinear representation of the DPE. Next,
   534	map in order to change its size so that it has the same size as G. The DPE is then used to generate
   551	FRS
   558	discrepancy can lead to notable performance degradation. Additionally, the similarity among gallery
   560	basis of similarity (FRS). The FRS module uses advanced feature enhancement technology to solve
   562	the matching performance. The specific effects of the FRS are illustrated in Figure 3.
   563	FRS is a similarity-driven feature alignment method, with the core innovation lying in combining
   589	patterns. This multi-scale feature aggregation thus results in a more comprehensive and stable
   605	crucial for improving model performance. To this end, our DPE module generates 𝐺𝑖+ embeddings
   612	and negative samples, MHCLoss enables detailed feature optimization from the macroscopic intermodal level to the microscopic intra-modal level.
   613	The core design principle of MHC loss is adaptive feature alignment across modalities, which is
   614	achieved by generating multiple contrastive pairs within a hierarchical structure. At each hierarchical level, MHC loss promotes feature similarity between corresponding VIS and IR samples while
   617	at higher levels, effectively addressing the modality gap. The primary advantage of MHC loss is
   625	Fig. 4. The proposed loss function MHC, mainly primarily aims to minimize the distance between embeddings
   628	their initial counterparts, fostering intra-modality contrast. Through this dual process, MHC encourages the
   633	MHC loss enables robust cross-modal feature alignment, enhancing the model’s adaptability to
   635	learned embeddings while increasing model robustness and accuracy, making it highly suitable for
   637	Specifically, the inputs to MHCLoss are the VIS and IR modality features corresponding to the
   641	and where 𝑓𝑣𝑖 and 𝑓𝑖𝑟𝑖 are the new embeddings generated by the DPE module. To compute the loss,
   678	𝑖 ). The overall MHC loss can then be
   702	the model, we propose an orthogonal Gramian metric loss (OGM loss). This loss function effectively
   778	decays to 1 × 10−3 and 1 × 10−4 at epochs 80 and 120, respectively. For 𝑘 1 in the FRS module, we
   791	[55], and the experimental results are shown in Table 1.
   795	experiments are shown in Table 1. For the all-search mode, compared with the GGMMNet approach,
   808	The comparison results on the RegDB dataset are presented in Table 1. In the VIS-to-IR mode, the
   829	Table 1. Comparison of Rank-1 (%) and mAP (%) Performances with the State-of-the-art Methods on
  1159	contrast, our MDFN focuses on fine-grained feature extraction at multiple scales through DFF and
  1160	DPE modules, capturing local and global features in a robust and adaptive manner. This multi-scale
  1172	Fig. 5. (a–c) Show the results of the features distribution in the 2D feature space on SYSU-MM01, where
  1178	Multi-head self-attention is integrated into our DFF module to account for contextual dependencies
  1189	Visualization Analysis
  1191	Feature distribution. We utilize t-SNE [32] for feature map visualization to assess the specific effects
  1200	are more pronounced. Additionally, we visualize the inter-class and intra-class distances, as shown
  1209	Fig. 6. Some Rank-10 retrieval results obtained by the baseline and the proposed MDFN on SYSU-MM01
  1211	Table 2. Effectiveness of Each Component on SYSU-MM01 under the All-Search Mode
  1212	(a) Analysis of the proposed DFF, DPE and FRS.
  1213	Baseline DFF DPE FRS Rank-1 mAP
  1269	Retrieval results. To better reflect the superiority of our approach, we visualize the Rank-10
  1276	Ablation Studies
  1278	Effectiveness of each component. We evaluate the effectiveness of individual key components (including DPE, DFF, L𝑀𝐻𝐶 , 𝐿𝑂𝐺𝑀 , and FRS) in all-search modes on the SYSU-MM01 dataset, with the
  1279	components added independently. The baseline is trained by L𝑖𝑑 , and we keep the other settings
  1280	consistent with our approach. The experimental results are shown in Table 2(a), where × indicates
  1282	Table 2(a) shows the performance improvement achieved by each module. Compared with the
  1283	baseline, DFF improves the Rank-1 accuracy by 5.50% and the mAP by 5.78%. DPE improves the
  1285	DFF and DPE are used jointly, confirming the effectiveness of DFF and DPE when used together.
  1286	FRS optimizes the matching performance more effectively by enhancing the representation of the
  1288	two modalities. The FRS algorithm without any computational overhead in the inference phase
  1290	effective role of FRS in the post-processing retrieval matching phase.
  1292	shown in Table 2(b), L𝑀𝐻𝐶 improves the Rank-1 accuracy by 2.55% and mAP by 2.71%. L𝑂𝐺𝑀
  1299	Fig. 7. Analysis of the influence of which stage of ResNet-50 to plug the DFF module and DPE module.
  1304	Impact of the insertion stage choice. Our proposed DFF and DPE modules can be integrated at any
  1307	DFF and DPE into different stages of ResNet50 on the performance of MDFN across all-search
  1312	The DFF module enhances the model’s ability to learn complex feature patterns through its local
  1316	additional computational burden. Simultaneously, the AGC component efficiently refines local
  1321	The DPE module is designed to enhance multi-scale feature representation and is especially
  1324	suitable point for DPE insertion. At this stage, the feature maps have undergone preliminary downsampling and multiple convolution operations compared to stages 0 to 2, resulting in more abstract
  1325	features enriched with high-level semantic information. Inserting DPE at this stage allows the
  1331	the DEE module in DEEN [53], we propose the DPE module for generating diverse embeddings.
  1332	To evaluate the effectiveness of our DPE module, we compare it with the DEE module, and the
  1333	results are presented in Figure 8(a). Our DPE module achieves an improvement of 0.74% in Rank-1
  1340	Fig. 8. Comparison with the DEE module and impact of the number of DPE branches.
  1342	Fig. 9. Influence of different 𝛼, 𝜆1 and 𝜆2 values on SYSU-MM01.
  1344	accuracy and 1.05% in mAP compared with those of the DEE module. Additionally, when the DFF
  1345	module is incorporated, the DPE module further achieves a 0.75% improvement in Rank-1 accuracy
  1347	of the DPE module in extracting fine-grained features.
  1348	Impact of the number of DPE branches. Our proposed DPE module uses multiple convolutional
  1371	Fig. 10. Comparison of class activation maps and retrieval results of our proposed method with the Baseline.
  1387	We use Grad-CAM [27] to visualize the class activation maps of the baseline and our method in
  1388	complex dynamic environments, as shown in Figure 10. The DPE takes two branches, in addition
  1401	In this article, we introduce the MDFN, which integrates three core components: DFF, DPE, and FRS.
  1402	The DFF module leverages both local and long-range feature dependencies to expand the receptive
  1404	DPE module uses deformable dilated convolutions with variable dilation rates to capture multi-scale
  1412	scenes. During the feature matching phase, the FRS module refines feature representations by
  1414	incorporates a modality-aware hierarchical contrastive loss (MHCLoss) to enhance multi-scale
  1491	file fragment classification through visual perspectives. IEEE Transactions on Multimedia (2024), 1–14.
  1508	2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE
  1528	[32] Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. Journal of Machine Learning
  1556	re-identification with modality-and instance-aware visual prompt learning. In Proceedings of the 2024 International

exec
/bin/zsh -lc "pdftotext 'Multi-year long-term person re-identification using gait and HAR features.pdf' - | nl -ba | rg -i 'ablation|comparison|cross-year|same-year|fusion|attention|HAR|gait-only|visual-only|Table|Fig\\.' | head -n 240" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     8	Multi-year long-term person re-identiﬁcation using gait and HAR features
    27	We propose a two-stream person re-identiﬁcation (Re-ID) framework that integrates gait and human action recognition (HAR) through cross-attention fusion. The model processes gait sequences via a BiLSTM-based encoder to
    28	capture temporal motion dynamics. At the same time, HAR embeddings are extracted using pre-trained video
    29	backbones and distilled into compact behavioral features. These two modalities are fused using a cross-attention
    32	that integrating HAR signiﬁcantly enhances gait-based Re-ID performance. Compared to gait-only models, our approach yields a 12 % improvement in mean Average Precision (mAP) in cross-year scenarios and up to an 11.6 %
    33	gain in same-year evaluations. The HAR-enhanced models also exhibit faster convergence and higher Rank-1
    44	time spans [2]. In this work, we focus on supervised Re-ID, where labeled identities guide the learning of embeddings. Most research emphasizes short-term image-based Re-ID, where appearance remains stable. Real deployments, however, require long-term Re-ID, where clothing and visual cues change over time. Video provides richer temporal
    45	information, with gait serving as a stable biometric foundation. Nevertheless, benchmarks for long-term, video-based Re-ID remain scarce.
    51	tion recognition (HAR), oﬀer complementary stability and behavioral
    54	Our main contributions are as follows. First, we propose a twostream architecture that combines gait and HAR features for person ReID. A gait dynamics is learned in a structured way by a BiLSTM branch,
    55	and a light-weight and well-cited baseline in video recognition is obtained by the HAR branch, producing compact clip-level embeddings by
    57	light-weight. Then, we incorporate a cross-attention scheme enabling interactive mixing of the two feature streams, allowing for a ﬂexible combination of activity and motion cues. We utilized a triplet loss function
    62	Experiments show that fusing gait and HAR outperforms singlestream and visual-only baselines, yielding more robust embeddings. Results highlight the potential of motion-based multimodal representations
    77	Fig. 1. Pipeline of the proposed two-stream architecture. Gait and HAR features are extracted independently, aligned in dimension, and fused via a cross-attention
    78	mechanism where gait attends to HAR. The fused embedding is used for identity representation. This design prioritizes gait as the primary cue while enriching it
    79	with complementary HAR context.
    97	such as color, clothing, and facial features. However, in real-world longterm scenarios involving large-scale outdoor events or cross-year comparisons, appearance cues often fall short due to lighting variations,
    99	line of research by jointly modeling gait and HAR in a uniﬁed architecture for spatio-temporal Re-ID [6].
   112	HAR and multi-modal cues. HAR provides a higher-level understanding of behavior that can complement gait in identity reasoning. Although HAR has been widely studied for activity classiﬁcation
   114	HAR models for person Re-ID in ultra-distance sports scenarios have
   118	by jointly embedding HAR and gait cues, enabling richer, context-aware
   131	spatio-temporal person Re-ID (see Fig. 1), which integrates gait and human action information through pre-trained backbones. We ﬁrst formalize the problem, then detail the structure and roles of the gait and HAR
   132	backbones, the fusion strategy based on cross-attention, and ﬁnally, the
   133	training objective using triplet loss with semi-hard negative sampling.
   145	to extract temporal embeddings for gait and HAR.
   169	Since the output dimensions of the gait and HAR streams may diﬀer
   170	(𝐷𝑔 ≠ 𝐷ℎ ), we apply a projection layer to transform the HAR representation into the common dimension 𝐷:
   175	ℎHAR
   176	= 𝜙proj (ℎ̄ HAR
   185	∈ ℝ𝐷 and ℎHAR
   186	∈ ℝ𝐷 are projected into a shared embed𝑖
   189	size (i.e., 𝐷𝑔 = 𝐷), ensuring compatibility for subsequent fusion. The
   191	, ℎHAR
   195	subsequently used for cross-attention-based fusion and identity embedding computation.
   210	Rationale for temporal aggregation. In the HAR stream, we use global
   212	keeps the HAR stream lightweight, computationally robust, and easy to
   217	available (e.g., self-attention, Transformer pooling), we intentionally select simplicity and robustness in order to uncover the isolated contribution of HAR cues towards gait-based recognition.
   228	HAR Processing Pipeline. The HAR stream is designed to capture
   241	3.2. Cross-attention fusion
   242	We apply a cross-attention mechanism to fuse both modalities, where
   243	the gait embedding attends to the HAR representation. Given the aligned
   245	, ℎHAR
   252	𝐾𝑖 = 𝑉𝑖 = ℎHAR
   255	The attention weights and attended HAR embedding are computed
   261	HAR backbone and 𝐷ℎ is the dimensionality of each HAR embedding
   263	After extracting the sequence of embeddings from the HAR backbone, we apply normalization frame-wise before pooling. First, the embeddings are standardized using the mean and standard deviation computed across the training set, and then, each frame is L2-normalized:
   265	HAR
   270	𝐸̂ 𝑖HAR =
   279	the Query in the attention mechanism) because the primary objective
   281	and thus form the base representation to be preserved. HAR features, in
   284	recognition. Through the attention mechanism, the model selectively
   285	integrates relevant HAR information into the gait embedding. The resulting attended representation is then combined with the original gait
   289	𝐸𝑖HAR = HAR (𝑣context
   293	HAR
   294	𝐸𝑖HAR − 𝜇train
   301	HAR vector and apply a projection layer 𝜙 to obtain the ﬁnal fused
   312	Step 2: HAR Feature Extraction. The processed video 𝑣context
   315	passed through a pre-trained action recognition backbone HAR , which
   318	𝐸̃ 𝑖HAR =
   324	𝐸̃ 𝑖HAR
   325	‖𝐸̃ HAR ‖2
   339	ℎ̄ HAR
   340	= AvgPool(𝐸̂ 𝑖HAR ) ∈ ℝ𝐷ℎ
   355	Table 1
   392	To construct eﬀective triplets, we apply semi-hard negative mining,
   452	‖2 + (1 − 𝜆) ⋅ ‖ℎHAR
   453	− ℎHAR
   461	gait and HAR modalities equally.
   469	in how much they share subjects. Some pairs involve identical sets of
   471	subsets. This necessitates a pair-level analysis when interpreting training and test splits, as seen in Table 1.
   477	RP3_23 appears 39 times, reﬂecting a notable but less dominant colocation pattern.
   479	pairs, increasing the number of co-occurrence pairs without a proportional increase in video samples, as explained in Table 1. Their presence
   483	Fig. 2. These dataset samples illustrate short-term and long-term Re-ID scenarios. Each column corresponds to the same individual, while each row shows data
   487	Moreover, the training dataset contains both same-year pairs, such
   488	as RP0_20 ↔ RP1_20 and RP2_23 ↔ RP3_23, and cross-year pairs, including examples like RP0_20 ↔ RP3_23 and RP1_20 ↔ RP2_23. These
   490	highlighting transitions and tracking continuity across years (see Fig. 2).
   494	in three locations: RP1_20, RP2_23, and RP3_23. Evaluation is performed through pairwise comparisons between locations in both directions (e.g., RP1_20 to RP2_23 and RP2_23 to RP1_20), yielding six directional evaluation scenarios. For each direction, the 17 runners from
   495	the source location are matched against the 17 runners from the target location, producing 289 comparisons per direction. This results in a
   496	total of 1734 comparisons across the test set. The design ensures a comprehensive and balanced evaluation of generalization across spatial and
   504	5. Gait & HAR backbones: From motion to identity
   513	across space and time. All backbones are pre-trained on Kinetics400 [30], enabling robust and transferable HAR performance across varied conditions.
   516	used for gait and HAR, highlighting their architectural design and the
   518	form the foundation of our two-stream framework, with the gait backbone focusing on ﬁne-grained motion dynamics and the HAR backbone
   522	2, GaitGL, GaitPart, and GaitSet-which share the common objective of
   544	as GLN Phase 2 or GaitBase on GREW can be computationally demanding due to the dataset’s scale and complexity. Consequently, model training choices are inﬂuenced by dataset characteristics and the architectural robustness and scalability of the gait backbones.
   545	HAR Backbones. We employed a diverse set of backbone architectures with varying capacities to model spatial and temporal dynamics, including C2D, I3D, Slow8x8, Slow4x16, SlowFast8x8, SlowFast4x16, and X3D variants (L, M, S, XS). The C2D model [21] employs 2D convolutions on individual frames, treating the video as a sequence of static images. While eﬃcient, it lacks explicit temporal modeling. I3D [22] overcomes this by inﬂating 2D ﬁlters to 3D and using
   566	perform multiple comparisons between diﬀerent runners, averaging the
   570	fusion is enabled, we ﬁrst train the fusion layers (lr = 6e-5) and then
   572	using a batch size of 32, and each iteration involved generating semihard triplets to enhance convergence and model robustness. The model
   575	standard retrieval metrics, including Cumulative Match Characteristic
   581	thoroughly evaluated various hyperparameter conﬁgurations and fusion
   595	Table 2
   596	Long-term Re-ID using only gait features. Cross-year evaluations between
   659	Table 4
   660	Long-term Re-ID fusing gait and HAR. Performance on cross-year evaluations
   663	the gait-based backbone under consideration is shown in blue, and the bestperforming HAR model is highlighted in green.
   723	HAR Backbone
   765	Table 3
   766	Long-term Re-ID using only gait features. Cross-year evaluations between
  1048	Table 5
  1049	Long-term Re-ID fusing gait and HAR. Performance on cross-year evaluations
  1052	the gait-based backbone under consideration is shown in blue, and the bestperforming HAR model is highlighted in green.
  1060	HAR Backbone
  1087	suitable architecture for the Re-ID task. This allows us to establish a
  1089	gait model. Second, we assess the impact of incorporating HAR embeddings by comparing performance before and after their integration. As
  1090	described in Section 3, the gait-only experiments use the ℎGAIT
  1093	refer to Fig. 1) outputs as identity embeddings for discrimination.
  1238	is signiﬁcantly more challenging than same-year evaluations due to
  1250	Tables 2 and 3 present the performance of various gait and
  1251	appearance-based models in cross-year Re-ID scenarios, speciﬁcally
  1258	Fig. 3. CMC curves for the long-term evaluation between RP1_20 and RP2_23, comparing HAR-enhanced models. GaitGL_G is shown in black dotted, and AlignedReID
  1265	Performance generally declines in the second half of the table, covering the RP1_20 ↔ RP3_23 setting, indicating greater diﬃculty in this
  1271	backbone across evaluation protocols and the most suitable candidate
  1272	for further integration with HAR embeddings.
  1275	insight into where HAR contributes most through early rank improvements (e.g., Rank-1 and Rank-5) or enhanced overall retrieval consistency.
  1276	Fig. 3 shows the CMC curves for the long-term Re-ID between
  1278	of HAR demonstrates consistent improvements over both the gait-only
  1283	the HAR-enhanced models in the ﬁnal ranks. AlignedReID performs the
  1286	In the reverse direction (RP2_23 → RP1_20), all three HAR-enhanced
  1290	weaker early-rank performance than the HAR models (29.4 % Rank-1)
  1294	These results demonstrate that incorporating HAR features usually
  1295	boosts early-rank retrieval performance while contributing to more stable performance across higher ranks. The performance gap between
  1296	GaitGL_G and HAR-fused models is more pronounced in the cross-year
  1297	setting than in the same-year evaluation, reinforcing the robustness and
  1298	long-term Re-ID capacity of HAR-enhanced architectures.
  1299	Fig. 4 illustrates the most challenging cross-year evaluation setting:
  1300	RP1_20 ↔ RP3_23. Despite the increased diﬃculty caused by the threeyear gap and diﬀerent locations, HAR-enhanced models outperform the
  1301	gait-only and appearance-based baselines.
  1307	solid performance (35.3 % Rank-1), trails behind the HAR models at
  1314	7.2. HAR integration
  1315	Tables 4 and 5 present the performance of the proposed two-stream
  1316	model on the long-term Re-ID tasks, where HAR features are integrated with the gait backbone via cross-attention, as detailed in Section 3. Compared to the gait-only baseline GaitGL_G (highlighted in
  1317	blue), adding HAR consistently improves performance across all metrics, particularly in mAP and Rank-1 accuracy. In the RP1_20 → RP2_23
  1319	46.2 % and 46.6 %, respectively. When fused with HAR, models such as
  1328	Rank-1 improves from 35.3 % to 52.9 % in the RP3_23 → RP1_20 direction. Other strong HAR models like SLF_8x8𝑁𝐿𝑁 , SLF_4x16𝑁𝐿𝑁 , and
  1329	SL_8x8𝑁𝐿𝑁 also show consistent improvements over the gait-only setting.
  1330	We also present CMC curves to analyze how HAR integration impacts person Re-ID performance. These curves visualize the rank-based
  1331	retrieval accuracy of the most promising HAR-enhanced conﬁgurations
  1332	(e.g., X3D_L, SL_4x16𝑁𝐿𝑁 , and SL_8x8𝑁𝐿𝑁 ) in comparison with the bestperforming appearance-based baseline (AlignedReID) and the gait-only
  1341	Fig. 4. CMC curves for the long-term evaluation between RP1_20 and RP3_23, comparing HAR-enhanced models. GaitGL_G is shown in black dotted, and AlignedReID
  1347	Comparing these results with both the RP1_20 ↔ RP2_23 evaluation and the same-year setting, the improvements introduced by HAR
  1348	are more substantial in these highly unconstrained, long-term scenarios. HAR-enhanced models oﬀer better Rank-1 and Rank-5 performance
  1363	Comparing Tables 2, 3 and 6, all models show a noticeable drop in
  1364	performance under long-term Re-ID (Tables 2 and 3), particularly the
  1368	stark contrast demonstrates the limitations of appearance cues in longterm Re-ID tasks. Gait-based methods, while also aﬀected, maintain relatively stable performance over time. Notably, GaitGL_G is the only backbone to consistently perform well in both short-term and long-term settings, achieving 73.6 % / 64.7 % / 94.1 % in the short-term scenario and
  1376	Table 6 presents the results for the short-term Re-ID scenario, where
  1379	8.2. HAR integration
  1380	Table 6
  1381	Short-term Re-ID using only gait features. Performance on same-year crosslocation evaluations (probe → gallery). Metrics reported as mAP / Rank-1 /
  1406	Table 7 presents the results of the short-term Re-ID experiments
  1407	when HAR features are integrated into the pipeline through the crossattention mechanism described in Section 3. In this conﬁguration, the
  1409	Fig. 1). The gait-based model GaitGL_G is used as the backbone and
  1410	is highlighted in blue, while each HAR model is tested in combination
  1411	with this backbone. The results demonstrate that incorporating HAR signiﬁcantly enhances Re-ID performance compared to using gait alone.
  1414	mAP and 76.5 % Rank-1 in the reverse direction. When HAR is added,
  1418	at 84.7 % mAP and 76.5 % Rank-1 in reverse. Other top-performing HAR
  1420	in the forward direction, all of which outperform the gait-only baseline.
  1421	Even lightweight HAR models such as X3D_S and C2D yield notable
  1422	improvements, suggesting that HAR features consistently provide complementary motion and activity-level cues that enhance identity discrimination. Overall, these ﬁndings validate the eﬀectiveness of the
  1551	Fig. 5. CMC curves for short-term evaluation between RP2_23 and RP3_23, comparing HAR-enhanced models. GaitGL_G is shown in black dotted, and AlignedReID
  1554	Table 7
  1555	Short-term Re-ID results fusing gait and HAR. Performance evaluation results
  1556	for same-year, cross-location Re-ID (probe → gallery). Performance is reported
  1559	blue, and the best-performing HAR model is highlighted in green.
  1560	HAR Backbone
  1586	Fig. 5 presents the CMC curves for the short-term Re-ID evaluations.
  1587	Across both directions, integrating HAR features leads to consistent performance gains over the gait-only and appearance-only baselines.
  1594	100.0 %, indicating weaker retrieval consistency at higher ranks compared to HAR-enhanced models.
  1596	with all HAR-enhanced models surpassing both baselines. Slow_8x8𝑁𝐿𝑁
  1599	While GaitGL_G maintains a strong baseline (76.5 % Rank-1), it converges more slowly than the HAR-fused models. AlignedReID again performs the weakest at early and mid ranks, achieving only 64.7 % Rank-1
  1601	These results conﬁrm that HAR integration improves early retrieval
  1603	HAR-enhanced models, Slow_4x16𝑁𝐿𝑁 , X3D_L, and Slow_8x8𝑁𝐿𝑁 consistently outperform the gait-only and appearance-only baselines, reinforcing the value of motion-aware representations in realistic, locationvarying settings.
  1745	cross-attention fusion strategy and demonstrate that augmenting gait
  1746	with HAR improves robustness in short-term Re-ID tasks.
  1747	When comparing these results to those in Tables 4 and 5 (longterm Re-ID with HAR), we observe that performance also drops across
  1751	(RP1_20 → RP3_23), the best mAP is 63.8 % (X3D_L). However, the relative improvements introduced by HAR remain consistent in both scenarios, validating the generalization capacity of fused motion/activitybased representations. Moreover, HAR-enhanced models consistently
  1752	outperform the GaitGL_G baseline (in blue) in all short-term and longterm cases, establishing HAR integration-especially with two models
  1757	insight into where HAR contributes most through early rank improve-
  1759	9. Ablation study on fusion strategies
  1760	To further analyze the role of the fusion mechanism in our framework, we provide an ablation study using the best-performing gait backbone (GaitGL trained on GREW) and the best-performing HAR backbone (X3D-L) identiﬁed in our previous experiments. We evaluate three
  1761	alternative fusion strategies: cross-attention, cosine similarity-based fusion, and concatenation. Each represents a distinct way of integrating
  1762	gait and HAR embeddings. Cross-attention adaptively emphasizes behavioral cues conditioned on gait, while cosine similarity enforces alignment by maximizing directional closeness between modalities. In contrast, concatenation stacks the embeddings without explicit interaction,
  1765	in Section 3, using triplet loss with semi-hard negative mining and
  1767	of gait and HAR streams. By ﬁxing the underlying encoders to their
  1774	Fig. 6. CMC curves for long-term evaluation (RP1_20–RP2_23) with HAR model X3DL under diﬀerent fusion strategies. Solid lines show the best model per fusion;
  1778	Fig. 7. CMC curves for long-term evaluation (RP1_20–RP3_23) with HAR model X3DL under diﬀerent fusion strategies. Solid lines show the best model per fusion;
  1782	strongest conﬁgurations, this ablation isolates the impact of the fusion strategy itself. The comparative results quantify the impact of each
  1783	method on retrieval performance in both short-term and long-term settings, highlighting the advantages of attention-based fusion over simpler
  1785	In the long-term ReID evaluation, cross-attention consistently outperformed the other fusion strategies across all transfer scenarios.
  1786	For RP2_23→RP1_20 and the reverse direction (Fig. 6), cross-attention
  1789	(Fig. 7), where the cross-attention fusion reached 63.8 % mAP for
  1792	cross-attention also outperformed the baselines GaitGL_G and AlignedReID, demonstrating superior generalization capability in long-term
  1795	Our fused gait-HAR model outperformed appearance-based and gaitonly baselines, with +12 % mAP in long-term and +11.6 % in shortterm setups. HAR-enhanced models also improved early-rank retrieval,
  1796	conﬁrming the value of multimodal fusion in unconstrained conditions.
  1797	Limitations include reliance on reliable silhouettes and a HAR branch
  1799	51 videos, resulting in 1734 comparisons under a subject-disjoint protocol, which ensures a meaningful and comprehensive assessment.
  1811	that jointly models gait and HAR features. Our framework leverages existing components (BiLSTMs, pre-trained HAR extractors, attention) but
  1814	through cross-attention, thereby improving both short-term and longterm performance.
  1833	[15] N. Takemura, Y. Makihara, D. Muramatsu, T. Echigo, Y. Yagi, Multi-view large population gait dataset and its performance evaluation for cross-view gait recognition,
  1842	[19] N. Aharon, R. Orfaig, B.-Z. Bobrovsky, Bot-Sort: Robust associations multi-pedestrian
  1898	[4] G. Zhang, J. Zhou, Y. Zheng, G. Martin, R. Wang, Adaptive transformer with pyramid fusion for cloth-changing person re-identiﬁcation, Pattern Recognit. 163 (2025)
  1900	[5] Y. Makihara, D. Muramatsu, Y. Yagi, Gait Recognition: Databases, Representations,

exec
/bin/zsh -lc "pdftotext 'Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf' - | nl -ba | rg -i 'ablation|visual|t-sne|sensitivity|component|SSP|SDP|ATP|PGD|CCD|Table|Fig\\.' | head -n 220" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 169ms:
    20	Training Paradigm (ATP) on biased domains to maintain the
    27	Perception (SSP) module that constructs the feature relationship
    31	Distillation Perception (SDP) to improve the model both semantic
    32	and domain generalization ability. The main idea of SDP is to
    66	TABLE I
    72	is an intractable challenge.
    82	Table I, obvious domain imbalance among public benchmarks
    91	as shown in Fig. 1. From Fig. 1(a), we see that small-scale
    93	With further iterations, as is shown in Fig. 1(b), overfitting
   114	Fig. 1. An exploratory experiment by training a standard DG model on 4
   120	first develop an Asynchronous Training Paradigm (ATP) that
   138	propose a Structure Synergy Perception (SSP) module that
   148	Perception (SDP) to aggregate views of different submodels to
   150	ability. The SDP primarily comprises semantic based cross-
   152	model central distillation (CCD) and part graph distillation
   153	(PGD). The main idea of SDP is to use the center guided
   156	the importance of logits in semantic classification, CCD first
   161	leverage the image-level visual similarities of IDs to explore
   164	features extracted from the same model. Considering the visual
   166	Distillation (PGD) that uses this valuable prior knowledge to
   167	explore the domain invariant features. The PGD divides the
   168	image into several local parts and leverages the parts visual
   188	Paradigm (ATP) for open-world ReID. With the philosophy of divide and conquer, we divide the biased
   193	we construct a Structure Synergy Perception (SSP) module based on the feature relationship. SSP constructs the
   198	Distillation Perception (SDP) to aggregate views of different submodels. The SDP comprises semantic based
   199	cross-model central distillation (CCD) and parts graph
   200	distillation (PGD), which uses the center guided soft label
   293	for DG-ReID by designing a proxy task to mine local visual
   320	Fig. 2. Structure of our proposed method. Source 1 and Source 2 represent small- and large-scale datasets respectively. Our method divides the biased source
   342	Paradigm (ATP) for biased open-world ReID. The overall
   343	framework of our method is shown in Fig. 2. To relieve
   350	domains, we propose a Structure Synergy Perception (SSP)
   354	submodel, we propose a Synergy Distillation Perception (SDP)
   355	to aggregate views of different submodels. The SDP comprises
   356	semantic based cross-model central distillation (CCD) and
   357	parts graph distillation (PGD), which uses the center guided
   398	Paradigm (ATP) to exploit the source domains through multiparallel submodels.
   405	keeping the domain balance. As illustrated in Fig. 2, the
   406	ATP first divides the biased source domains into multiple
   417	as S 11 and [S 1K , S 2K , · · ·, S nK ] in Fig. 2, where S 1 and S K
   482	perception (SSP) module for synergy learning among submodels. The feature structure distribution of each subdoamin is first
   487	Fig. 3. The basic idea of the structure synergy perception learning. The
   493	exploring generalizable knowledge, as shown in Fig. 3.
   504	ATP is employed on the features F ij when j = i for supervised learning. And to reinforce synergy perception among
   505	submodels, we map the unique supervised knowledge of ATP
   557	To further improve the model generalization ability, we propose a Synergy Distillation Perception(SDP) module to exploit
   559	The SDP primarily comprises cross-model central distillation
   560	(CCD) and parts graph distillation (PGD), which uses the
   564	Fig. 4. Overview of cross-model central distillation (CCD). CCD extracts the
   576	a central distillation loss LDis based on ATP, which extracts
   630	Since the image-level visual similarities is unable to eliminate the domain noise as this noise may exist in all features
   632	relationship among intra- and inter-subdomains for generalization learning. Considering the visual difference of different
   633	local parts, we develop a Parts Graph Distillation (PGD)
   673	domain generalization learning. The inter-subdomain PGD can
   697	TABLE II
   700	Fig. 5. Overview of parts graph distillation (PGD). The PGD first models the
   861	TABLE III
   864	TABLE IV
   897	train the model, as shown in Table IV. During testing, we
   920	number of state-of-the-art methods on four widely used benchmark datasets presented in Table III, where M, CS, C and MS
   932	2) Evaluations on CUHK03: From Table III, we can see
   956	Table III. Specifically, our method achieves 49.4% rank-1 and
   967	“MS+CS+C3 →M” in Table III. On this dataset, the ACL
   990	Fig. 6. Performance variation with increased training epochs on Market1501
  1001	in Table V, our method also achieves superior performance on
  1009	D. Ablation Study
  1010	In this section, we conduct ablation studies to analyze the
  1011	effectiveness of each component of the proposed method.
  1012	The ablation studies are implemented on three settings of
  1014	and the experimental results with different components are
  1015	shown in Table VI.
  1017	shown in Table VI, the proposed ATP consistently improves
  1018	the model generalization. Specifically, on the experimental setting of “M+MS+CS→C3”, the ATP increases its performance
  1022	ATP also achieves remarkable improvements with 1.5% mAP
  1027	the training epochs. As shown in Fig. 6, the performance of
  1030	that the ATP training paradigm helps the model to exploit
  1034	in Table VI, we combine the asynchronous training paradigm
  1042	TABLE V
  1046	TABLE VI
  1047	A BLATION S TUDIES ON A SYNCHRONOUS T RAINING PARADIGM (ATP), S TRUCTURE S YNERGY P ERCEPTION (SSP), C ROSS -M ODEL C ENTRAL D ISTIL LATION (CCD) AND PARTS G RAPH D ISTILLATION (PGD). M ODELS A RE T RAINED W ITH THE OTHER T HREE DATASETS
  1052	generalization ability, represented as “+ATP+SSP”. It can be
  1053	seen that adding SSP increases the ATP by 1.9% mAP &
  1057	the SSP achieves improvements of 2.0% mAP and 1.9% rank1. The results validate that the collaborative learning among
  1061	Synergy Distillation Perception(SDP) aims to exploit multiview generalizable knowledge of different submodels, which
  1062	comprises cross-model center distillation (CCD) and parts
  1063	graph distillation (PGD). As shown in Table VI, adding the
  1064	CCD can further improve the performance to 41.2% mAP &
  1068	0.9% and 1.6% rank-1 respectively. And adding the PGD can
  1073	After combining the CCD and PGD, the performance of
  1077	“+ATP+SSP+SDP” in Table VI.
  1080	Fig. 7. Performance variation in extreme domain imbalance scenarios.
  1082	conduct the sensitivity studies in the extreme domain imbalance scenarios. We choose Market-1501 and a subset of
  1085	CUHK03 data is shown in Fig. 7. It can be seen that different
  1086	module are complementary and effective. The ATP divides the
  1090	SSP leverages the debiased knowledge to guide the synergy
  1091	learning among sub-models, which is complementary to ATP.
  1092	As shown in Table VI and Fig. 7, after integrating the ATP with
  1093	SSP, the model generalization ability can be further improved.
  1097	demand for the synergy distillation SDP to exploit multi-view
  1099	in Fig. 7, when choosing 1000 images from CUHK03, the
  1100	domain imbalance issue is further exacerbated, and the SDP
  1109	TABLE VII
  1114	TABLE IX
  1118	TABLE VIII
  1121	TABLE X
  1130	shown in Table VII. It can be seen that the ensemble manner
  1133	Table VII. And we further compare the central training strategy
  1136	represented as “Centralized Training” in Table VII. The results
  1142	shown in Table VIII. It can be seen that our method of
  1150	shown in Table IX. The cross entropy (CE) loss and triplet
  1159	Fig. 8. Evaluation on the number of parallel sub-models on Market1501 and
  1162	Fig. 9. Evaluation on the number of parts in PGD on Market1501 and
  1166	represented as “Non-parametric” in Table IX.
  1168	different methods in Table X. The experiments are conducted
  1187	Fig. 10. Evaluation on the quantity of selected images on Market1501 and
  1189	TABLE XI
  1193	of images on performance is shown in Fig. 10. It can be
  1199	conduct a ablation study on the occluded image. In conventional datasets, there is less data with occlusions, which causes
  1204	Table XI. The “Baseline (Single OD)” represents using the
  1215	10) Analysis of Hyper-Parameters: We conduct the sensitivity studies on the number of parallel sub-models, and the
  1216	results are shown in Fig. 8. It can be seen that when testing
  1218	performance first shows an upward trend and then kept stable,
  1228	we set different number of local parts in the PGD to analyse
  1229	its influence on our model. As shown in Fig. 9, the best
  1231	Fig. 11. The visualization of the SDP. From top to bottom are the original
  1232	image, the feature visualization of the ATP without and with SDP, respectively.
  1234	Fig. 12. Visual distributions of four person ReID benchmarks. The distributions are obtained from inference features of (a) Baseline trained with 30
  1243	E. Visualization
  1244	We use the Grad-CAM [91] to show four visualization
  1246	effectiveness of the proposed SDP, as shown in Fig. 11. In
  1248	the feature visualization of the ATP without and with SDP,
  1249	respectively. It can be seen that the features without SDP lack
  1252	After combining the SDP, the features of the bottom visualization can effectively provide more comprehensive information
  1255	visualize the t-SNE [90] distributions of features on four
  1262	Fig. 13. The visualization of the failure retrieval results. Images with red
  1268	shown in Fig. 12. We train the baseline with 30 epochs and
  1270	Fig. 12(a) and Fig. 12(b) respectively. It can be seen that
  1272	large-scale domain as shown in Fig. 12(b) and stopping early
  1274	Fig. 12(a). Comparing Fig. 12(b) with Fig. 12(c), we observe
  1276	(Fig. 12(c)) are clustered more compactly without domain
  1281	improvements, we present the visualization of the failure
  1282	retrieval results in Fig. 13. The query images are shown in
  1287	Fig. 13(a) and (b), the incorrect retrieval results are primarily
  1290	the bag in Fig. 13(b). Fig. 13(c), (d) and (e) show the erroneous
  1300	Paradigm (ATP) for biased open-world person ReID, where
  1302	The ATP is a parallel asynchronous learning paradigm with
  1306	Perception (SSP) to intensify the collaborative learning among
  1309	Perception (SDP) that uses the center guided soft label and
  1320	create a sufficient visual obfuscation effect on anonymized
  1391	[12] S. Liao and L. Shao, “Interpretable and generalizable person reidentification with query-adaptive convolution and temporal lifting,” in
  1410	[19] J. Ren et al., “Balanced meta-softmax for long-tailed visual recognition,”
  1549	“Collaborative visual place recognition through federated learning,”
  1587	Speech Signal Process. (ICASSP), Apr. 2024, pp. 2880–2884.
  1621	[90] L. Van der Maaten and G. Hinton, “Visualizing data using t-SNE,”
  1624	D. Batra, “Grad-CAM: Visual explanations from deep networks via

codex
消融信息补完后可以看到，这些文章基本都在做“组件可加性”之外的第二层证明：不只说模块涨点，还用极端不平衡、插入阶段、融合策略、可视化或同类增广对照去支撑原始观察。下面我再补第五、第六篇的消融细节，然后整理成逐篇方法论拆解。
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
/bin/zsh -lc "pdftotext 'Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification.pdf' - | nl -ba | rg -i 'ablation|visual|t-sne|sensitivity|component|DLF|MIA|DSM|HSM|ESM|Hilbert|Euclidean|Table|Fig\\.' | head -n 240" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    21	challenge (i), we design a Dual Level Fusion (DLF) module
    25	Invariant Augmentation (MIA) module is developed to extract
    29	(DSM), which aligns features in both Hilbert and Euclidean
    56	Fig. 1: Illustration of the differences between the proposed
    63	in both Euclidean and Hilbert spaces via mutual distillation,
   105	Fig. 1. Image-level methods [15], [26] typically employ Generative Adversarial Network (GAN) to synthesize VIS images
   132	Dual Level Fusion (DLF) module and a Modality Invariant
   133	Augmentation (MIA) module. DLF captures and adaptively
   136	dimensions. MIA extracts fine-grained semantic cues and
   142	Dual-Space Matching (DSM), which aligns modalities in both
   143	Euclidean and Hilbert spaces at the pixel level. By mapping
   145	employing a mutual distillation strategy, DSM encourages
   147	shared features. Moreover, DSM guides the model to focus
   152	pixel-level alignment in both Euclidean and Hilbert
   154	• We design a Dual Level Fusion (DLF) to capture hierarchical semantics across channel and spatial dimensions,
   155	and a Modality Invariant Augmentation (MIA) module
   266	by establishing cross-modality interactions in both the Euclidean space and an implicit Hilbert space. To realize this
   275	(MDDM) framework. Firstly, we present the overall architecture in Section III-A. Secondly, the proposed Dual Level Fusion (DLF) is described in Section III-B. Thirdly, Section III-C
   276	and Section III-D introduce the Modality Invariant Augmentation (MIA) module and the Dual-Space Matching (DSM),
   291	Fig. 2: Overview of the proposed framework for VIReID. (a) The Mutual Distillation driven Dual-space Matching (MDDM)
   293	The DLF module integrates low- and high-level semantics across spatial and channel dimensions for modality alignment. (c)
   294	The MIA module captures fine-grained semantics to enhance identity discriminability and facilitate visible-infrared modality
   301	An overview of the proposed MDDM framework is illustrated in Fig. 2, which mainly consists of three components:
   316	features, and the DLF is incorporated after layer1 and layer2
   320	features from both modalities. The proposed MIA module is
   331	feature stability, we propose the DLF module, which integrates
   336	types of source features. Taking the DLF module after layer1
   342	in Fig. 2 (b), the first DLF module takes Flt and Fht as input.
   343	Similarly, in the second DLF module, the low-level feature
   346	output of the first DLF module, denoted as Fm
   366	= DLF1 (Flt , Fnt ), F̂ t = DLF2 (Flt , Fm
   384	mechanism, and the output of the first DLF module can be
   401	obtain the output of the second DLF module, namely the
   414	While DLF leverages cross-level semantic fusion to mitigate feature inconsistencies, visible–infrared person reidentification still faces significant fine-grained modality discrepancies, such as variations in texture, shape, and local
   415	structural information. The proposed Modality Invariant Augmentation (MIA) module, as shown in Fig. 2 (c), is designed
   457	differences in the visual information presented by these modalities in terms of shape and features, leveraging cross-modal
   531	1) Hilbert Space Matching Loss: First, the modality-shared
   532	knowledge captured by the MIA module is aggregated using
   558	Fig. 3: The proposed Hilbert Space Matching loss. Different
   570	where HSIC(·, ·) denotes the hilbert-schmidt independence
   612	maps features into Hilbert space.
   623	Euclidean distance for measurement. Euclidean space constraints preserve local discriminative features by minimizing
   628	[61] project features into Hilbert space. Hilbert space constraints model the statistical dependence of cross-modal features in a high-dimensional kernel space, ensuring that shared
   633	propose the Dual-Space Matching (DSM) strategy, which
   634	projects features into both Euclidean and Hilbert spaces and
   638	DSM consists of two components: Hilbert Space Matching (HSM) loss and Euclidean Space Matching (ESM) loss.
   639	Lhsm aligns features in the Hilbert space for better crossmodal alignment, while Lesm applies mutual distillation for bidirectional knowledge transfer, enhancing feature consistency
   646	As shown in Fig. 3, the formulation of both inter-class and
   647	intra-class losses contributes to the overall HSM loss. Finally,
   648	HSM loss is formulated as:
   649	Lhsm = αLinter + Lintra ,
   653	where α is hyperparameter that balance inter-class and intraclass losses. The proposed Lhsm accounts for both cross-modal
   657	2) Euclidean Space Matching Loss: In cross-modality retrieval tasks, there often exists a significant discrepancy between the feature distributions of different modalities, leading
   660	mismatch, we propose a Euclidean Space Matching loss
   661	(Lesm ). Within a unified framework, this loss implements
   681	Finally, the ESM loss is defined as
   682	Lesm = Lkl1 + Lkl2 .
   689	Fig. 4: The proposed Euclidean Space Matching loss. Different
   694	As shown in Fig. 4, given the fine-grained shared features from the visible and infrared modalities, denoted as
   714	stable and consistent supervision for the student. Additionally,
   753	In addition to the HSM loss (Lhsm ) and the ESM loss
   754	(Lesm ), both the cross-entropy loss (Lid ) and the triplet loss
   758	Lall = Lid + Ltri + Lhsm + Lesm .
   810	TABLE I
  1002	TABLE II
  1086	remaining layers as shared ResBlocks. The proposed DLF is
  1087	added after the first and second layers, while the MIA module
  1235	in Table I, the proposed MDDM demonstrates robust performance on the SYSU-MM01 dataset. Specifically, in the
  1255	TABLE III
  1256	A BLATION STUDIES FOR DIFFERENT COMPONENTS OF THE
  1286	DLF
  1288	MIA
  1290	DSM
  1291	Lhsm Lesm
  1343	TABLE IV
  1344	A BLATION STUDY OF DLF AND MIA MODULES ON THE
  1346	POSITION OF DLF. ( B ) E VALUATION OF THE POSITION OF
  1347	MIA. T HE PARAMETERS AND FLOP S ARE CALCULATED
  1349	(a) Evaluation of the position of DLF
  1371	(b) Evaluation of the position of MIA
  1394	Fig. 5: Parameter analysis on the SYSU-MM01 dataset. (a)
  1418	Comparisons on the RegDB Dataset. As shown in Table I,
  1442	C. Ablation Study and Analysis
  1443	Effectiveness of Each Component. To demonstrate the
  1444	contribution of each component in MDDM, we conduct ablation studies by evaluating the individual components on
  1445	the SYSU-MM01 dataset, as shown in Table III. We first
  1446	assess the independent impact of each proposed component
  1447	on the baseline model. Specifically, the proposed DLF module (Index-2) improves the Rank-1 accuracy by 1.86% and
  1448	mAP by 3.4%, indicating that the DLF module captures and
  1449	effectively integrates hierarchical features. The proposed MIA
  1451	mAP by 1.69%, demonstrating that the MIA module captures
  1454	between visible and infrared modalities. By individually introducing the Lhsm (Index-4) and Lesm (Index-5), Rank-1
  1471	(b) Baseline+DLF
  1473	(c) Baseline+DLF+MIA
  1479	(f) Baseline+DLF
  1481	(g) Baseline+DLF+MIA
  1485	Fig. 6: Visualization of the learned features on SYSU-MM01. (a)-(d) are the distribution of the cosine distance between VIS-IR
  1490	Fig. 7: Visualization of attention maps using Grad-CAM. From
  1493	further increases to 4.17%. This is because the DSM constraint
  1495	of the latent feature space and effectively bridging the substantial modality gap. Based on DSM, DLF and MIA (Index-8
  1497	all components (Index-10), MDDM effectively mitigates the
  1500	that all proposed components consistently contribute to the
  1502	proposed components.
  1503	Analysis of DLF and MIA. We conduct experiments to
  1504	thoroughly examine the effectiveness of the proposed DLF and
  1505	MIA modules, as shown in Table IV. Since both proposed
  1507	of DLF and MIA at different positions by integrating them
  1509	as presented in Table IV (a). When DLF is integrated after
  1512	an additional 0.1M model parameters. However, when DLF is
  1514	improvement. The performance further improves when DLF
  1518	0.5M model parameters. Therefore, we integrate DLF after
  1519	Layer 1 and Layer 2. Similarly, MIA is evaluated in the same
  1520	manner, and when MIA is integrated into both Layer 3 and
  1523	Analysis of DSM. The Hilbert Space Matching loss and
  1524	Euclidean Space Matching loss in the proposed DSM use
  1525	hyperparameters α and τ to establish stable relationships
  1526	between modalities in the Euclidean space and Hilbert space.
  1527	As shown in Fig. 5, the best performance is achieved when
  1534	D. Visualization
  1539	in Fig. 6 (a–d). By comparing the baseline feature distances
  1540	(Fig. 6 (a)) with those obtained using the proposed method
  1554	setting. As shown in Fig. 8, our proposed MDDM significantly
  1563	Fig. 8: The Rank-10 retrieval results obtained by the baseline
  1568	(Fig. 6 (b–d)), it is evident that the proposed components
  1574	of the proposed MDDM, we visualized the feature distribution
  1575	on the SYSU-MM01 dataset using t-SNE [77]. As shown
  1576	in Fig. 6 (e–h), 10 identities were randomly selected from
  1580	Moreover, the proposed components effectively mitigate crossmodality discrepancies, resulting in significant improvements
  1582	Attention Heat Maps. To better visualize the proposed
  1584	are presented in Fig. 7. Compared with the baseline method,
  1601	discrepancies through pixel-level alignment performed simultaneously in both the Euclidean and Hilbert spaces. To further
  1603	introduce a plug-and-play DLF module, which establishes
  1605	Building on this, a MIA module is designed to dynamically capture identity-consistent patterns in accordance with
  1694	[25] R. Wu, B. Jiao, W. Wang, M. Liu, and P. Wang, “Enhancing visibleinfrared person re-identification with modality-and instance-aware visual
  1745	[41] Z. Miao, H. Liu, W. Shi, W. Xu, and H. Ye, “Modality-aware style
  1801	[57] Y. Hao, N. Wang, J. Li, and X. Gao, “Hsme: Hypersphere manifold
  1817	[62] D. Greenfeld and U. Shalit, “Robust learning with the hilbert-schmidt
  1868	[77] L. van der Maaten and G. Hinton, “Visualizing data using t-sne,” J.
  1871	D. Batra, “Grad-cam: Visual explanations from deep networks via

exec
/bin/zsh -lc "pdftotext 'Occluded person Re-Identification with noise injection.pdf' - | nl -ba | rg -i 'ablation|visual|sensitivity|component|ENI|INI|vehicle|MixUp|CutMix|Hide-and-Seek|GridMask|Table|Fig\\.' | head -n 280" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
    27	Vehicle
    36	innovative solution by leveraging large-scale vehicle images, which can be automatically associated using
    37	consistent license plate information. Although vehicle images are typically considered noise in person ReID,
    38	we demonstrate that incorporating them into training significantly improves discrimination in occluded ReID.
    39	Specifically, we propose two vehicle noise injection strategies: Epoch-wise Noise Injection (ENI) and Imagewise Noise Injection (INI). ENI periodically alternates the training dataset between person and vehicle images,
    40	while INI synthesizes occluded person images by inserting vehicle segments to generate physically plausible
    41	occlusion patterns. Both strategies deliberately introduce controlled noise during training, bridging the domain
    42	gap between vehicle and pedestrian images while avoiding overfitting. Extensive experiments show that our
    52	difficulty stems from dual occlusion patterns: (1) object-induced occlusion, where environmental elements (e.g., vehicles, buildings) obstruct
    68	also prioritize static pedestrian scenes, neglecting the dynamic occlusion patterns caused by moving vehicles that are prevalent in real-world
    69	urban environments. Paradoxically, although vehicle-induced occlusions account for over 40% (statistics in Occluded-Duke [3] query set)
    70	in cross-camera retrieval, vehicle images are typically discarded as
    72	This study challenges the conventional view that vehicle data constitute mere noise in person ReID. Specifically, large-scale vehicle images,
    81	0031-3203/© 2026 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
    87	Fig. 1. Our noise injection enhances the Rank-1 accuracy on Occluded-Duke [3] of various vision backbones without additional manual annotations.
    97	Fig. 2. The visualization of data. (a) Person, (b) Inter-person occlusion, (c) Vehicle-induced occlusion, (d) Our synthetic Vehicle-Person.
   100	systems, offer two principal advantages: First, their inherent visual
   102	patterns and mitigating overfitting. Second, vehicle images captured
   105	Consequently, we incorporate vehicle images into the training process
   109	Our approach consists of two key components: (1) Epoch-wise
   110	Noise Injection (ENI) strategy, which periodically alternates the training dataset between person and vehicle images, serving as a regularizer
   111	to prevent overfitting. (2) Image-wise Noise Injection (INI) strategy, which synthesizes occluded person images by inserting vehicle
   113	enriching the semantic content of the training data. As shown in Fig.
   115	significantly. Besides, we visualize several training images in Fig. 2. Our
   116	synthetic images are very similar to person images with vehicle-induced
   120	• We demonstrate that vehicle data is not merely noise but also
   121	contributes positively to the training process in occluded person
   124	Injection strategies, which inject controlled noise to bridge the domain gap between vehicle and pedestrian images while avoiding
   127	four standard person ReID benchmarks show that incorporating vehicle data yields significant performance improvements,
   154	In computer vision, this principle manifests in various forms. Adversarial training deliberately introduces worst-case perturbations to
   157	effective training distribution without compromising label integrity.
   165	a noisy-label learning problem and improves training through reliable pseudo-label refinement. These studies suggest that structured
   172	cross-domain noise directly during training, allowing the network to
   174	Our approach extends these insights by treating vehicle images as
   175	structured noise for person ReID. Unlike random perturbations, vehicle
   177	while introducing appearance variations that enhance occlusion robustness. The person–vehicle copy–paste strategy bridges domain gaps
   179	diversifying visual patterns. This represents a novel instantiation of
   196	maintaining identity consistency. CAAO [11] advances this concept
   202	several complementary perspectives. Cross-erasure [12] enhanced network improves robustness by strengthening feature learning under
   212	data-centric perspective by injecting structured cross-domain noise during training, without requiring extra manual annotations or specialized
   216	As illustrated in Fig. 3, our proposed framework addresses the
   219	reconceptualizing vehicle data not as mere noise but as a valuable
   220	training resource in occluded scenarios.
   221	We think that controlled noise injection during training improves
   222	model robustness and generalization. Specifically, we leverage largescale vehicle datasets, readily available from automated license plate
   223	recognition systems, to augment conventional person ReID training.
   224	The training begins by preparing two distinct data sources: person
   225	images from standard ReID datasets and vehicle images with consistent
   227	modulates input composition, alternating between pure vehicle training
   228	phases and mixed person-vehicle phases. This approach serves a dual
   232	robust performance across diverse occlusion scenarios while maintaining high discriminative capability for person ReID.
   249	Fig. 3. Overview of the proposed NIReID. The framework utilizes vehicle images as positive noise to improve generalization in occluded person ReID by adopting
   250	two noise injection strategies: ENI, which alternates person and vehicle images during training, and INI, which generates occluded person images by inserting
   251	vehicle segments. The dual-branch network extracts both global and local features, which are aggregated using average and max pooling, and are supervised by
   258	Data: Epoch 𝑁, Switching Interval 𝑆, Vehicle Images 𝐼𝑣 ,
   262	1 initialization;
   267	Select vehicle images 𝐼𝑣,𝑖 ;
   273	Select vehicle images 𝐼𝑣,𝑖 ;
   288	Data: Vehicle Images 𝐼𝑣 , Cropped Vehicle Image 𝐶𝑣 , Person
   292	Result: Person-Vehicle Mixed Images 𝐼𝑝𝑣
   293	1 initialization;
   300	Randomly select a vehicle image 𝐼𝑣′ from vehicle images 𝐼𝑣 ;
   310	from varying viewpoints. This exposure to diverse visual contexts improves the model’s ability to handle domain shifts and camera variations commonly encountered in practical ReID scenarios. Furthermore, the periodic alternation prevents the model from becoming
   311	overly specialized to either domain, maintaining a balanced learning
   316	use two noise injection strategies to enhance model training. They enable the network to extract discriminative features from the large-scale
   317	vehicle dataset while preserving its focus on the primary person ReID
   318	task. By treating vehicle data as controlled noise, we enrich feature
   323	The Image-wise Noise Injection (INI) strategy focuses on synthesizing physically plausible occluded person images by strategically
   324	inserting vehicle segments into person images. Unlike ENI operating
   325	at the epoch level, INI performs augmentation at the instance image
   327	As outlined in Algorithm 2, INI begins by randomly selecting vehicle
   328	images 𝐼𝑣′ from the vehicle dataset for each person image 𝐼𝑝,𝑖 . The
   331	controls the vertical placement of the vehicle image. This constraint
   332	ensures that vehicle occlusions primarily affect the lower portions of
   333	person images, mimicking realistic occlusions where vehicles typically
   335	𝐶𝑣 = 𝐼𝑣′ [∶ 𝐻 − 𝑦, ∶ 𝑊 − 𝑥] extracts appropriately sized vehicle segments.
   337	the corresponding region in the person image with the vehicle segment,
   339	The INI strategy presents several advantages compared to traditional occlusion simulation methods. First, it produces realistic occlusions that closely mimic real-world surveillance scenarios, where
   340	vehicles often obstruct pedestrians. Second, incorporating real vehicle
   342	occlusion methods fail to replicate accurately. Third, randomized positioning ensures diverse occlusion patterns across training samples,
   346	The Epoch-wise Noise Injection (ENI) strategy implements a temporal alternation that periodically switches the training dataset between
   347	person and vehicle images.
   348	As detailed in Algorithm 1, the ENI process operates with a predefined switching interval 𝑆 that determines the frequency of dataset
   349	alternation. During vehicle-only epochs, the model trains exclusively
   350	on vehicle images 𝐼𝑣,𝑖 , forcing the network to extract features that are
   351	not dependent on human-specific visual cues. This constraint compels
   353	capabilities. In contrast, during mixed epochs, the training process
   354	combines both person 𝐼𝑝,𝑖 and vehicle images 𝐼𝑣,𝑖 through a copy–paste
   355	process. This combination creates a diverse training environment where
   356	the model must discriminate between different types of visual content
   357	while maintaining its ReID capabilities. This operation strategically
   358	places cropped vehicle images within person images, creating realistic
   360	The key advantage of ENI lies in its ability to introduce domainspecific variations and camera bias through vehicle images captured
   370	Complementarily, the global max pooling extracts the most discriminative features from each channel, highlighting the most salient visual
   373	both the general appearance distribution and the most distinctive visual
   377	We also compare our INI and ENI with several augmentations and
   379	Comparison with MixUp. MixUp [24] enhances model robustness by
   380	generating novel training images from the combination of two labeled
   384	our unlabeled vehicle images in the INI strategy.
   385	Comparison with CutMix. Like MixUp, the CutMix [25] requires a
   386	label for the vehicle image, which the INI strategy cannot provide.
   387	Consequently, CutMix without a vehicle label can be viewed as a generalized form of INI, but breaks spatial priors. For instance, a pedestrian
   388	image where only the upper body is occluded by a vehicle does not
   408	Comparison with Hide-and-Seek. The Hide-and-Seek method [26]
   409	randomly hides patches in a training image, thereby forcing the network to rely on other relevant features when the most discriminative
   410	one is hidden. Our approach, INI, replaces these hidden patches with
   411	vehicle images. This substitution provides richer semantic information,
   413	Comparison with GridMask. GridMask [27] is a variant of Hide-andSeek, and it similarly removes regions, thereby disrupting semantic information. In contrast, our INI approach better preserves this semantic
   417	ENI and INI methods operate solely by adjusting the input image
   419	integrated into any training framework.
   426	To optimize the network, we employ a multi-loss training strategy.
   428	local features, ensuring both branches contribute to identity discrimination. In contrast, the triplet loss with hard negative mining is applied
   440	noise injection techniques also depend on complex components like
   441	neural architecture search (NAS) or GANs [30]. Both our INI and ENI
   443	occluded person re-identification, thus are more suitable than universal
   445	Comparison with curriculum learning. The curriculum learning typically follows an easy-to-hard schedule. ENI adopts a nonmonotonic,
   446	periodic alternation between person-only and vehicle-only (or mixed)
   447	training phases.
   448	Comparison with domain alternation. The standard domain alternation or multi-domain training assumes semantic consistency across
   449	domains, whereas ENI deliberately introduces semantically mismatched
   450	but structurally correlated data (vehicles vs. pedestrians) as positive
   467	complementary visual representations.
   468	The two branches share identical initial weights from the pretrained backbone but diverge in their subsequent processing to learn
   469	distinct visual patterns. Each branch incorporates a convolutional layer
   470	followed by a convolutional head that refines the extracted representations. These components are designed to produce branch-specific
   492	Table 1
   493	Datasets for Occlusion, General, and Noisy Vehicle ReID.
   500	Table 3
   588	Table 2
   589	Training settings. BS means batch size, and LR means max learning rate. Three
   743	Table 4
   748	In the context of occluded person re-identification, occlusions increase task entropy because visual cues become incomplete and ambiguous. By injecting structured ‘‘vehicle noise’’ in the form of semantically coherent patterns (e.g., vehicle textures and contours), the
   754	of 𝜋-noise learning, where the auxiliary domain (vehicles) introduces
   842	Comparison with Different Backbone Architectures. Table 3 compares our method against existing approaches using different backbone architectures. Despite utilizing CNN-based backbones, our NIReID
   860	MSMT17 [2]). The noisy vehicle dataset is VeRi-776 [34]. Detailed
   861	information about these datasets is provided in Table 1.
   869	marked as NIReID+R50 and NIReID+R-1.6G. Training settings are
   870	shown in Table 2. A linear warm-up strategy is applied, followed by
   873	Comparison with Methods Using Additional Supervision. Table 4
   878	demonstrating that the proposed vehicle noise injection strategy effectively eliminates the need for costly manual annotations. This comparison underscores the practical advantage of our approach: it attains
   880	consistent improvements across both datasets further validate the effectiveness and generalizability of leveraging vehicle data as positive
   892	Table 5
   900	Table 8
   901	Ablation study on epoch interval in ENI.
  1055	Table 9
  1056	Ablation study on height ratio in INI.
  1058	Table 6
  1202	validate that our vehicle noise injection approach enhances model
  1205	Table 7
  1251	4.5. Ablation study
  1265	Component Analysis. We first analyze the individual and combined
  1267	(ENI) and Image-wise Noise Injection (INI). As illustrated in Fig. 4,
  1269	injection), ENI only, INI only, and the combination (ENI+INI).
  1272	the baseline achieves 61.3% Rank-1 accuracy, which is modestly enhanced to 62.3% with ENI alone (+1.0%), significantly improved to
  1273	68.4% with INI alone (+7.1%), and reaches the highest performance of
  1276	with INI consistently outperforming ENI in terms of improvement
  1279	Notably, INI demonstrates superior effectiveness compared to ENI
  1280	across all configurations, suggesting that enriching training data with
  1281	realistic occlusion patterns through vehicle segment insertion contributes more significantly to model performance than the regularization effect of epoch-wise alternation. The consistent improvement
  1285	To comprehensively evaluate the effectiveness of our proposed approach, we conduct ablation studies on the Occluded-Duke dataset [3]
  1286	to analyze the individual contributions of each component and investigate the optimal hyperparameter settings.
  1294	approaches, as shown in Table 5. On Market-1501, NIReID+R-1.6G
  1299	Table 6 shows significant advantages in both labeled and detected
  1306	3.1% in both Rank-1 accuracy and mAP (see Table 7). Remarkably,
  1311	Hyperparameter Analysis for ENI. The epoch interval in ENI determines how frequently the training alternates between person and
  1312	vehicle datasets. We systematically investigate different interval values
  1313	ranging from 3 to 30 epochs, as shown in Table 8.
  1320	Fig. 4. Ablation study on two noisy injection strategies.
  1321	Table 10
  1322	Ablation study on INI and other augmentations.
  1324	Table 12
  1325	Ablation study on Pooling Type.
  1349	INI
  1367	INI (brightness = 0.1)
  1368	INI (brightness = 0.5)
  1369	INI (contrast = 0.1)
  1370	INI (contrast = 0.5)
  1371	INI (saturation = 0.1)
  1372	INI (saturation = 0.5)
  1423	MixUp (p = 0.1, 𝛼 = 0.1)
  1424	MixUp (p = 0.2, 𝛼 = 0.1)
  1425	MixUp (p = 0.3, 𝛼 = 0.1)
  1426	MixUp (p = 0.1, 𝛼 = 0.2)
  1427	MixUp (p = 0.1, 𝛼 = 0.3)
  1453	CutMix (p = 0.1, 𝛼 = 0.1)
  1454	CutMix (p = 0.2, 𝛼 = 0.1)
  1455	CutMix (p = 0.3, 𝛼 = 0.1)
  1456	CutMix (p = 0.1, 𝛼 = 0.2)
  1457	CutMix (p = 0.1, 𝛼 = 0.3)
  1503	GridMask (p = 0.2)
  1504	GridMask (p = 0.3)
  1505	GridMask (p = 0.4)
  1576	Table 13
  1577	Ablation study on Pooling Number.
  1686	Table 11
  1687	Ablation study on Vehicle Resolution.
  1728	Table 14
  1729	Ablation study on Inference Speed, which is measured on a Nvidia 4090 with
  1734	to insufficient training stability, as frequent dataset switching prevents
  1792	strikes an optimal balance between training stability and regularization
  1800	model to adapt to each domain while maintaining frequent enough
  1803	pooling operations is detrimental, as it introduces noise from vehicles
  1806	Hyperparameter Analysis for INI. The height ratio parameter in INI
  1807	controls the proportion of vehicle segments inserted into person images,
  1810	synthetic occlusion, as presented in Table 9.
  1819	where vehicle segments dominate the image and obscure critical person identification features, leading to performance degradation. The
  1824	Ablation of Inference Speed. Table 14 reports the inference speed of
  1827	performance while maintaining the highest inference speed.
  1830	ReID by reconceptualizing vehicle images as valuable training resources
  1832	in person ReID and demonstrates that large-scale vehicle datasets
  1835	strategies: ENI, which alternates training between person and vehicle
  1836	datasets to prevent overfitting, and INI, which synthesizes realistic
  1837	occlusion patterns by inserting cropped vehicle images into person
  1845	significance lies in utilizing readily available vehicle data through
  1847	systems without additional annotation costs. Besides, this work establishes a new perspective on noise learning in ReID, opening directions
  1850	Comparison with Other Augmentations. Table 10 presents a comparison between INI and several augmentation methods, including the
  1851	color jitter, MixUp [24], CutMix [25], HaS [26], and GridMask [27].
  1852	We apply color jitter to the vehicle images in the INI. The results
  1856	baseline. The results show that MixUp leads to a significant performance degradation, which we attribute to the absence of vehicle labels
  1857	in its formulation. Both HaS and GridMask cause a slight performance
  1859	their effects. Although CutMix yields some improvement, its performance remains inferior to that of INI, primarily because it disrupts
  1867	scenarios. Second, the current method focuses primarily on vehicleinduced occlusion and does not account for other common occluders,
  1875	Ablation of Vehicle Resolution. Table 11 presents the impact of input
  1876	resolution within the INI strategy. As expected, reducing the resolution
  1877	results in performance degradation. This decline is likely attributable
  1878	to frequent vehicle occlusion of the lower body in typical scenarios.
  1880	a partially visible vehicle is uncommon, which may introduce bias into
  1882	Ablation of Pooling Type and Number. Tables 12 and 13 present
  1886	statistics that are relatively stable when partial regions are missing,
  1889	By combining coarse-grained global pooling with fine-grained local
  1892	global appearance cues. This complementary design explains the superior performance observed in the pooling ablation results and highlights
  1903	editing, Visualization, Software. Deng Cai: Writing – review & editing, Project administration. Shenqi Lai: Writing – review & editing,
  1918	[26] Krishna Kumar Singh, Yong Jae Lee, Hide-and-seek: Forcing a network to be
  1924	[27] Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia, GridMask data
  2000	poses to identity: Training-free person re-identification via feature centralization,
  2003	[24] Hongyi Zhang, Moustapha Cissé, Yann N. Dauphin, David Lopez-Paz, Mixup:
  2004	Beyond empirical risk minimization, in: ICLR, 2018.
  2006	Junsuk Choe, CutMix: Regularization strategy to train strong classifiers with
  2013	[34] Xinchen Liu, Wu Liu, Tao Mei, Huadong Ma, PROVID: Progressive and multimodal vehicle reidentification for large-scale urban surveillance, IEEE Trans.
  2060	and refining hybrid compressors for efficient object re-identification, IEEE Trans.
  2082	computer vision, data mining, and information retrieval.

exec
/bin/zsh -lc "pdftotext 'Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification.pdf' - | nl -ba | rg -i 'ablation|visual|t-sne|sensitivity|component|nearest|NNSC|CB|ODE|PPLR|Table|Fig\\.' | head -n 260" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     5	Nearest Neighbor Sample Constraint and ODE
    18	also crucial for the model’s performance. To this end, this
    20	based on nearest neighbor sample constraint and ordinary
    21	differential equation guided feature reconstruction (NNSC-FR)
    24	present a novel nearest neighbor sample constraint (NNSC) after
    27	improve clustering accuracy, an inter-class balance loss (CB loss)
    29	nearest neighbor classes. In addition, guided by the thirdorder adam solution of the Ordinary Differential Equation, we
    30	design a Feature Reconstruction (ODE-FR) module with residual
    31	structure to improve the model representation ability. Extensive
    35	Index Terms—Unsupervised person re-identification, nearest
    80	modeling of the feature evolution process. Second, lack of
    83	hard samples often receive unstable pseudo-labels, and these
    85	to insufficient inter-class separability and degraded model
    97	contrastive loss the most suitable metric learning algorithm for
   111	YANG et al.: NNSC AND ODE GUIDED FEATURE RECONSTRUCTION FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   113	Fig. 1. Examples of confusing images for the unsupervised person Re-ID
   121	shown in Fig. 1, appropriate constraints are needed between
   124	features with rich representation capabilities can refine pseudolabels, thereby improving the accuracy of model training.
   125	Therefore, we also need to find ways to enhance the model’s
   128	person Re-ID method based on nearest neighbor sample
   130	reconstruction (NNSC-FR) to improve the clustering accuracy
   132	features. Specifically, we propose a nearest neighbor constraint
   133	(NNSC) based relative distance, which explicitly identifies
   135	sample pair distances. NNSC enables the model to focus
   137	rather than treating all samples uniformly. Based on NNSC,
   138	we also design an inter-class balance loss (CB loss), which
   144	ODEs [13], [14], [15] have shown that it can significantly
   145	help the representation and generalization of the model. Thus,
   146	we apply the implicit third-order adam solution of ODEs to
   147	design a suitable residual structure guided by the solution of
   149	methods of mining encoder features at various depths, such
   151	inserted at any depth of the encoder, and function modules
   153	so we call it ODE guided feature reconstruction (ODE-FR)
   161	ability of the unsupervised model.
   162	Overall, we improve the unsupervised person Re-ID model’s
   165	• We introduce the nearest neighbor sample constraint
   166	strategy (NNSC), which aims to enhance clustering accuracy by mining positive samples within the same class
   169	• We design the inter-class balance loss (CB loss) better to
   170	identify the hard samples between the nearest neighbor
   172	• We incorporate an ODE-guided feature reconstruction
   173	(ODE-FR) module, which leverages the implicit thirdorder Adam solution of an ordinary differential equation
   175	improve the model representation ability.
   176	• Extensive experiments verify that our proposed NNSC-FR
   192	supervision teacher-student model to refine the pseudo-labels
   203	[28] proposed a diffusion model-based unsupervised person
   205	a spatial diffusion model and semantic consistency loss, thus
   213	enhancing current state-of-the-art methods. The recently introduced Relation-Preserving Feature Embedding model [29]
   221	algorithms, accurately clustering visually ambiguous samples
   230	study proposes a nearest neighbor constraint mechanism that
   237	into training. In the initial training of the model, only a small
   239	were considered for training. The model would self-adapt
   276	NNSC explicitly identifies the hardest sample pairs at the
   279	class and the nearest negative samples from neighboring
   285	C. Network Design With ODEs
   287	networks in recent years due to its stable training process and
   300	deep neural networks as a discrete dynamical system, identifying similarities between ResNet and ODE discretization
   310	Re-ID task to fine granularity, we can use ODEs to design an
   313	of the Re-ID model.
   317	YANG et al.: NNSC AND ODE GUIDED FEATURE RECONSTRUCTION FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   321	Fig. 2. Network structure of NNSC-FR. NNSC strategy applies the nearest neighbor constraint to each instance sample and realizes the fine classification
   322	of hard samples between classes; CB loss can better identify hard samples between the nearest neighbor classes and improve clustering accuracy; ODE-FR
   323	improves the model’s feature representation ability.
   328	person Re-ID. Therefore, we take PPLR [23] based on global
   329	and part feature learning as our baseline. PPLR calculates
   335	loss Ltri refines the pseudo-labels. The total PPLR loss is as
   337	LPPLR = Lce + Ltri + Laals + L pglr .
   339	2) Overall Structure: Although PPLR refines the global and
   343	needs improvement. On the other hand, the model’s feature
   347	in Fig. 2, the total network structure of NNSC-FR is proposed.
   348	First, under the stable constraint of class centroid contrast
   349	loss, the nearest neighbor sample constraint strategy finds
   350	the farthest positive sample and the nearest negative sample
   353	balance loss, the classification criteria of the nearest cluster
   356	the ODE guided feature reconstruction module is inserted into
   361	Fig. 3. Nearest neighbor sample constraint strategy.
   363	B. Nearest Neighbor Sample Constraint
   390	larger clusters. As shown in Fig. 3 (a), we reserve the cluster
   416	on model learning is reduced. Cluster centroid c1 ,c2 ,. . .,cC is
   429	the nearest neighbor sample constraint strategy, in which we
   437	by the clustering process. As shown in Fig. 3 (b), different
   444	while the nearest neighbor sample constraint can optimize the
   447	loss of the nearest neighbor sample mining strategy is defined
   481	centroid contrast loss Lcen and the nearest neighbor sample
   488	Fig. 4. t-SNE visualization comparison of feature distributions. (a) without
   489	CB loss; (b) with CB loss. The red circles highlight hard samples that
   490	are ambiguous and close to decision boundaries in (a). After applying CB
   492	clearly, demonstrating the effectiveness of CB loss in improving feature
   497	nearest neighbors to a suitable distance. As the hard samples
   498	are optimized, the model progressively refines the decision
   501	between the nearest neighbors can also avoid problems such
   503	training of an unsupervised person Re-ID model.
   505	As shown in Fig. 4, to encourage the network to move the
   506	nearest neighbor hard samples closer to the correct cluster
   521	LCB = −
   555	reached. When LCB is maximum, the distance between xij
   557	the nearest neighbor classes is most difficult to distinguish.
   567	YANG et al.: NNSC AND ODE GUIDED FEATURE RECONSTRUCTION FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   575	LNNS C−FR = LPPLR + LGC + λLCB .
   581	concave functions and Jensen’s inequality. When LCB reaches
   587	minimizing LCB through backpropagation, the model enforces
   594	D. ODE Guided Feature Reconstruction
   598	by using numerical schemes in ODEs as blueprints. Unlike
   599	their approach of designing various ODE-blocks as the main
   601	Runge-Kutta methods, our ODE-FR module is designed based
   604	component. This multi-step structure can leverage multi-layer
   609	As shown in Fig. 5 (a), it illustrates feature fusion, which
   612	diverse information, thereby enhancing the model’s expressive
   613	capability. Fig. 5 (b) represents feature cascading, where features from different levels are sequentially connected to create
   616	the model. However, both methods exhibit issues of blindness
   618	design the ODE-FR module (Fig. 5 (c)) based on the thirdorder implicit Adam solution formula of ordinary differential
   619	equations, effectively overcoming these shortcomings. Additionally, ODE-FR allows the insertion of function modules
   631	Fig. 5. Diagram of feature fusion, feature cascade and our proposed ODE-FR.
   645	to model feature reconstruction as an ODE numerical solution
   691	Based on the above formula, our ODE-FR module is
   692	designed and its structure is shown in Fig. 5, consisting of
   703	change the number of channels, the ODE-FR is relatively
   709	way to shape image features. In general, the ODE-FR applies
   756	and ablation experiments on the above three datasets. Mean
   789	the state-of-the-art methods in the Market-1501, DukeMTMCreID, and MSMT17 datasets. NNSC-FR achieves the best
   792	missing results in these methods implemented with opensource code. For the comparison method, we start when the
   796	of generating pseudo-labels by clustering. MMT [19] dualteacher model uses four feature extraction networks, and the
   797	model is large.
   800	PISL [28], utilizes a spatial diffusion model to optimize
   805	Unlike the above methods, our proposed NNSC-FR introduces
   808	nearest neighbor sample constraints and feature reconstruction,
   810	time, PPLR [23] and INCLR [24] make full use of the
   811	complementary of global and part features to optimize pseudolabels. Our proposed NNSC-FR is an improvement on PPLR.
   818	sample mining strategy to mine samples in the k-nearest
   822	point of our proposed NNSC-FR, and both have achieved good
   826	YANG et al.: NNSC AND ODE GUIDED FEATURE RECONSTRUCTION FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   830	TABLE I
   836	the state-of-the-art method HHCL, NNSC-FR achieves the
   843	challenging dataset, MSMT17, NNSC-FR achieves the most
   847	TABLE II
   851	D. Ablation Studies
   852	1) Effectiveness of Each Component: We conduct ablation
   855	Tab. II, in the experiments on the MSMT dataset, the NNSC
   858	gain of the CB loss and ODE-FR on the baseline network is
   860	and 0.3%, respectively. It proves that NNSC and CB loss
   861	can optimize clusters, and the ODE-FR can improve feature
   862	representation to a certain extent. After introducing the NNSC
   863	strategy, the CB loss focuses on further discrimination and
   864	differentiation of the nearest neighbor classes, which results in
   865	0.8% and 0.5% improvements on mAP and Rank-1, respectively. Finally, the ODE-FR effectively reconstructs the feature
   877	TABLE III
   878	C OMPARISON B ETWEEN NNSC AND T RADITIONAL C ONTRASTIVE L OSS
   881	TABLE V
   883	ODE-FR ON M ARKET-1501 (%)
   885	TABLE IV
   886	TABLE VI
   888	C OMPARISON B ETWEEN CB L OSS AND T RADITIONAL L OSS M ETHODS ON
   891	A BLATION E XPERIMENT OF ODE-FR ON M ARKET-1501(%)
   898	2) Necessity of NNSC: In our NNSC module, we still
   905	our NNSC, and CB loss lost its effect under the traditional
   908	instance level samples when the cluster centroid is stable. This
   909	allows the model to adjust clustering boundaries in feature
   911	constraint, CB loss can be introduced to further optimize the
   913	3) Comparison of CB Loss: We replace CB Loss with
   915	Loss. As shown in Tab. IV, when removing CB Loss (w/o
   916	CB Loss) or substituting it with other loss functions, the
   917	model performance significantly declines. Specifically, the
   922	adaptability to the feature space. CB Loss dynamically pushes
   927	4) Location and Superiority of ODE-FR: First, we investigate the impact of the embedding position of ODE-FR on the
   928	model’s performance on Market-1501. As shown in Tab. V,
   929	we embed ODE-FR into four different layers of ResNet-50.
   933	on the model’s accuracy. As the embedding position moves
   934	deeper, the model begins to demonstrate significant positive
   937	abstract features are not suitable for feature reconstruction,
   941	Furthermore, to validate the role of ODE-FR in enhancing
   942	the model’s feature representation capability, as shown in
   943	Tab. VI, we add different forms of modules at the same position to conduct ablation experiments. Firstly, in cases where
   948	residual module of ODE, the recognition accuracy significantly
   953	ODEs [57], which is based on adaptive time steps, suffers from
   958	YANG et al.: NNSC AND ODE GUIDED FEATURE RECONSTRUCTION FOR UNSUPERVISED PERSON RE-IDENTIFICATION
   962	TABLE VII
   966	Fig. 6. The visualization of distance distribution of randomly selected
   971	sampling points prevent stable learning of hierarchical feature
   973	efficiency. Our ODE-FR addresses these limitations through
   976	Therefore, the design of the residual network guided by ODEs,
   977	combined with the embedding of the required modules, effectively enhances the model’s feature representation capability,
   979	models.
   980	Finally, as shown in Fig. 6, we randomly construct 60,000
   984	ODE-FR reduces the distance between pairs of positive samples while increasing the distance between pairs of negative
   986	Fig. 7 (a) and (b) of the t-SNE visualization.
   989	cross-dataset experiments where models trained on one dataset
   997	correction γ = 2.5 on Market-1501). Models trained on standard datasets are directly tested without adaptation. As shown
  1000	TABLE VIII
  1005	visual degradation.
  1009	NNSC-FR method requires 125.70 MB parameters and 9.754
  1011	only 0.087 ms more than PPLR, achieving 1661.69 FPS and
  1016	G. Parameter Sensitivity Analysis
  1018	PPLR, we propose two additional loss functions. In the
  1019	NNSC, the CB loss coefficient is λ, aimed at improving
  1021	clustering center contrastive loss and the nearest neighbor
  1027	H. Visualization Research
  1037	Fig. 7. The t-SNE visualization of 10 person. (a) The baseline method. (b) Baseline+OED-FR. (c) Our method. Different colors in the plot represent different
  1040	Fig. 8. The top8 visualization of query image retrieval results. (a) The baseline method. (b) HHCL. (c) Our method. On the left is the query image, and the
  1042	TABLE IX
  1046	extraction and visualized the results using a t-SNE plot (as
  1047	shown in Fig. 7). It is evident that with the introduction of the
  1048	ODE-FR module and the NNSC module (with CB loss), both
  1050	challenging nearest neighbor samples have been optimized,
  1056	As shown in Fig. 8, we conduct a visual experiment to
  1057	compare the classification of hard samples between the nearest
  1061	umbrella, so it is the nearest neighbor hard sample that
  1064	our method, NNSC-FR does an excellent job of accurately
  1070	Through the visualization experiment, it is easy to see the
  1072	We also visually compare the NNSC-FR and HHCL. HHCL
  1073	used the k-nearest neighbor strategy to constrain hard samples
  1080	YANG et al.: NNSC AND ODE GUIDED FEATURE RECONSTRUCTION FOR UNSUPERVISED PERSON RE-IDENTIFICATION
  1084	baseline, our method constrains the nearest neighbor samples
  1086	and improve the quality of pseudo-labels. As shown in Fig. 8,
  1088	or other people’s occlusion, our proposed NNSC-FR has a
  1092	the model feature representation ability in unsupervised person
  1093	Re-ID tasks, we propose an algorithm based on nearest neighbor sample constraint and ODE guided feature reconstruction
  1094	(NNSC-FR), which make full use of features to optimize the
  1096	propose a nearest neighbor sample constraint strategy, which
  1098	between the nearest neighbor classes for each small-batch
  1101	the nearest neighbor classes. In addition, we also create a
  1104	ability of feature representation. These components effectively
  1109	and erasing, to enhance the model’s robustness and effectively
  1112	show that our proposed NNSC-FR achieves a very competitive
  1132	model with noisy labels,” IEEE Trans. Inf. Forensics Security, vol. 15,
  1135	unsupervised visual representation learning,” in Proc. IEEE/CVF Conf.
  1188	prototypes for visual pattern debiasing,” Int. J. Comput. Vis., vol. 132,
  1196	of intrinsic semantics with diffusion model for person re-identification,”
  1211	[34] Z. Pang, J. Guo, Z. Ma, W. Sun, and Y. Xiao, “Median stable
  1250	[45] X. He, Z. Mo, P. Wang, Y. Liu, M. Yang, and J. Cheng, “ODEinspired network design for single image super-resolution,” in Proc.

codex
我没有额外联网，这次判断基于本地 PDF 的方法、相关工作和消融。你当前列了 6 篇，我按这 6 篇逐篇拆。

**1. Multi-Model Synergy Perception for Open-World Person Re-Identification**

1. 触发观察：作者先做了一个探索实验，用普通多源 DG ReID 训练 4 个源域，发现小规模域先收敛，大规模域还没学完时，小规模域和开放测试域已经开始退化。这个观察把问题从“多源数据越多越好”拉回到“多源规模不平衡会导致训练崩塌”。

2. 重定义动作：他们把旧问题“学习域不变特征”改写成“在保持域多样性的同时构造域平衡”。关键词是 biased source domains、domain balance、domain diversity、asynchronous training、divide and conquer、synergy perception。核心话术是，简单重采样会丢掉大域知识，简单联合训练会被规模偏置拖垮，所以需要一种不丢信息的平衡学习范式。

3. 机制怎么长出来：ATP 很自然，大域拆成多个批次，每个批次和小域共享批次组合成平衡子域，再用多个子模型并行学。接着他们说多子模型会带来“各自只看一部分知识”的问题，于是用 SSP 对齐特征关系矩阵，用 CCD 做中心 logit 蒸馏，用 PGD 做部位关系图蒸馏。ATP 和重定义绑定最紧，SSP/SDP 是从“多子模型副作用”推出的补丁，PGD 稍微有堆模块感。

4. 证据闭环：最关键不是 SOTA 表，而是 Fig.1 的训练退化观察、ATP/SSP/CCD/PGD 的逐项消融、极端域不平衡设置下的敏感性实验、和集中训练或简单集成的比较。可视化用 Grad-CAM 和 t-SNE 说明 SDP 后关注区域更完整、域分布更紧。这个闭环在证明“域规模偏置确实存在，平衡子域加协同学习确实对症”。

5. reviewer 为什么买账：这篇卖的主要是视角和训练范式，不是某个单独模块。真正 novelty 来自把开放世界 ReID 的多源训练失败归因到“规模不平衡导致的 collapse risk”，再给出一个能保持大域信息的异步并行范式。

**2. Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification**

1. 触发观察：VI-ReID 中可见光和红外的成像差异导致清晰度、纹理、噪声、背景和姿态都不一致。作者还抓了一个测试期现象，训练时 VIS 和 IR 可以按批次平衡，但测试时一个 query 对多个 gallery，query-gallery 和 gallery-gallery 的信息量不均衡会影响匹配。

2. 重定义动作：他们没有只说“模态差距”，而是把问题拆成复杂动态场景下的局部细节、长程上下文、多尺度空间结构、测试期相似度噪声。关键词是 multi-scale dynamic fusion、dynamic perceptual enhancement、feature reweighting with similarity、local and long-range dependencies、information imbalance。

3. 机制怎么长出来：DFF 用动态卷积分支抓局部，用注意力分支抓长程依赖。DPE 用不同 dilation 的 deformable dilated convolution 生成多尺度 embedding。MHC loss 让跨模态 embedding 拉近，让同模态生成 embedding 保持差异，OGM loss 抑制冗余。FRS 在匹配阶段利用 query-gallery 和 gallery-gallery 的 top-k 相似度增强。机制和问题是一一对应的，但整体更像“把每个痛点配一个模块”，不是从单一观察自然推出。

4. 证据闭环：他们做了 DFF、DPE、FRS 的逐项消融，也做了 MHC 和 OGM 的损失消融。还有模块插入 stage 的实验、DPE 和已有 DEE 模块的对比、DPE 分支数和超参敏感性、t-SNE、类内类间距离、Grad-CAM 和检索结果可视化。闭环重点是证明每个动态因素都能单独贡献，而不是一个模块偶然涨点。

5. reviewer 为什么买账：这篇主要卖机制组合，不是特别锐利的新视角。novelty 来源是“动态、多尺度、局部加全局、训练加匹配”的完整工程闭环。对我们有用的教训是，如果观察不够尖，消融就必须非常密，才能让组合稿显得可信。

**3. Multi-year long-term person re-identification using gait and HAR features**

1. 触发观察：普通 ReID 默认外观短期稳定，但真实长时场景里衣服、光照、地点和年份都会变。作者把场景放到 2020 到 2023 的越野跑视频，外观线索不可靠，步态相对稳定，但单纯步态又缺少行为上下文。

2. 重定义动作：他们把 ReID 从“图像外观匹配”重定义成“跨地点、跨年份的运动生物特征识别”。HAR 也不是作为动作分类任务，而是作为步态身份表示的上下文补充。关键词是 long-term、spatio-temporal ReID、gait、HAR、cross-year、motion-based multimodal representation、context-aware activity cues。

3. 机制怎么长出来：先用检测、跟踪和分割得到 silhouette 序列，再用 gait backbone 提取部位级步态特征。方法里一个细节很关键，他们承认 BiLSTM 处理的是身体水平部位序列，不是真正时间序列，用它建模部位之间的结构关系。HAR 分支用预训练视频 backbone，做 context-constrained crop，最后让 gait 作为 query 去 attend HAR，保证身份主线仍由 gait 主导。机制和重定义绑定比较紧，技术本身不新，但角色分配清楚。

4. 证据闭环：关键证据是新数据集的 same-year 和 cross-year 协议、外观模型、gait-only 和 gait-HAR 的对比、CMC 曲线，以及 cross-attention、cosine fusion、concatenation 的融合策略消融。他们还强调 HAR 在更难的跨年设置里增益更明显，这直接支撑“行为上下文在长时 ReID 中有价值”。

5. reviewer 为什么买账：这篇卖的是问题设定和证据，不是复杂模块。novelty 来源是长时真实数据、跨年评测和“HAR 作为步态上下文”的视角。审稿人买账的前提是，这个场景足够真实，且对照能说明不是普通视频 backbone 更强，而是 HAR 补了长期身份线索。

**4. Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification**

1. 触发观察：VIS 有颜色和纹理，IR 是灰度、边界模糊、细节少，导致特征空间错位。作者还指出图像级生成方法需要严格配对、成本高；特征级方法更稳定，但常只学到轮廓和粗结构，深层语义还会丢失。

2. 重定义动作：他们把 VI-ReID 的瓶颈重定义成两个问题：低层和高层语义难以跨模态对齐，VIS 和 IR 在无边界 latent space 中存在分布不一致。关键词是 dual-level fusion、modality invariant augmentation、dual-space matching、Euclidean space、Hilbert space、mutual distillation、pixel-level alignment。

3. 机制怎么长出来：DLF 在浅层融合低级细节和高级语义，MIA 用水平和垂直池化、分布 mask、多尺度卷积注意力提取细粒度共享信息。DSM 是主叙事，Hilbert Space Matching 用 HSIC 约束全局统计依赖，Euclidean Space Matching 用 EMA teacher classifier 和双向 KL 做互蒸馏，保局部判别和预测一致性。DSM 和“欧氏不够、希尔伯特补全局”的重定义绑定最紧。

4. 证据闭环：Table III 做 DLF、MIA、HSM、ESM 的组件消融，Table IV 验证 DLF 和 MIA 放在哪些层最合适。还有 α 和 τ 的参数分析、VIS-IR cosine distance 分布、t-SNE、Grad-CAM 和 Rank-10 检索可视化。它的证据闭环是证明双空间不只是换名词，确实让跨模态分布更紧、注意区域更对。

5. reviewer 为什么买账：这篇卖的是视角加机制。真正 novelty 是把常见的“跨模态对齐”包装成“欧氏空间保局部身份差异，希尔伯特空间保全局统计依赖，再用互蒸馏让二者一致”。这比单纯说加一个 alignment loss 更像可审的理论化问题定义。

**5. Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification**

1. 触发观察：无监督 ReID 的伪标签错误主要集中在聚类边界。聚类中心对比学习稳定但太粗，不处理实例级边界 hard sample；小批次 hard mining 又只能看局部样本，和全局 memory bank 不匹配。同时，作者认为普通残差连接缺少对特征演化过程的建模。

2. 重定义动作：他们把“伪标签噪声”重定义成“最近邻类别边界样本没有被精细约束”。关键词是 nearest neighbor sample constraint、cluster boundary、hard samples、inter-class balance、pseudo-label quality、ODE guided feature reconstruction。这个重定义比“改进聚类”更具体，因为它指出了错误发生的位置。

3. 机制怎么长出来：NNSC 保留聚类中心 memory 维持稳定，同时在实例 memory 中为每个 query 找最远正样本和最近负样本，专门拉开最近邻类别边界。CB loss 用凹函数和 Jensen 不等式解释，当分类概率接近均匀时说明样本处于混淆区，最小化它就是把 hard sample 推离边界。ODE-FR 用三阶隐式 Adams 公式构造残差模块，插到 ResNet 第三层增强表示。NNSC 和 CB 与重定义绑定很紧，ODE-FR 更像第二条“表示能力不足”的补强线。

4. 证据闭环：最关键的是 CB 前后的 t-SNE 边界样本可视化、NNSC/CB/ODE-FR 的逐项消融、NNSC 和传统 contrastive loss 的对比、CB 和其他 loss 的对比、ODE-FR 插入位置和替代模块对比、正负样本距离分布、Top-8 检索可视化。这个闭环能支撑“边界 hard sample 被更好地区分”，但 ODE 理论本身的证据相对弱一些。

5. reviewer 为什么买账：这篇主要卖机制，但机制有一个明确失败点托住。novelty 不是“又挖 hard sample”，而是“中心稳定约束加全 memory 最近邻实例修正”。ODE-FR 提供额外新鲜感，不过真正承重的是边界样本重定义。

**6. Occluded person Re-Identification with noise injection**

1. 触发观察：遮挡 ReID 里车辆遮挡很常见，论文说 Occluded-Duke query 中 vehicle-induced occlusion 占比超过 40%，但车辆图像通常被当成无关噪声丢掉。作者反过来观察到，车辆图像有真实纹理、颜色、视角和相机偏置，而且很容易从车牌系统收集。

2. 重定义动作：这是最值得学的一篇。它把“车辆是噪声”改写成“车辆是有结构的正向噪声”。关键词是 positive noise、positive-incentive noise、structured cross-domain noise、data-centric perspective、vehicle data as valuable training resource、without manual annotations。旧问题是遮挡建模，新问题是如何用真实跨域干扰物降低遮挡识别的不确定性。

3. 机制怎么长出来：INI 很自然，既然车辆会真实遮挡行人，就把真实车辆块贴到行人下半身，形成更物理合理的遮挡样本。ENI 则在 epoch 级别交替车辆和行人，让模型不只记住人类外观模式。网络本身是双分支全局和局部特征，损失是常规 ID loss 加 triplet。机制简单，但和重定义绑定紧，尤其 INI 几乎是从观察直接推出的。

4. 证据闭环：他们做了 ENI only、INI only、ENI+INI 的组件消融，结果显示 INI 单独贡献最大，说明核心不是周期训练，而是真实车辆遮挡模拟。还和 MixUp、CutMix、Hide-and-Seek、GridMask、color jitter 比较，证明不是普通增广。再加 ENI 间隔、INI 高度比例、车辆分辨率、池化类型和推理速度消融，闭环是在证明“车辆作为结构化遮挡源”比随机遮挡和通用增广更对题。

5. reviewer 为什么买账：这篇卖的是视角，不是模块。机制很朴素，但“把会被丢弃的车辆数据重定义为 positive-incentive noise”有反直觉性，又有现实采集优势和强对照消融，所以审稿人容易接受它是数据中心方法创新，而不是普通 copy-paste。

共同套路可以概括成三句话：能发的方法不是先有模块，而是先把一个具体失败现象改写成更窄、更可验证的新问题。机制最好像是从这个新问题自然长出来的，至少要让每个模块都有明确的失败点对应。证据上不能只给 SOTA 和组件消融，还要有一个能反过来证明“原始观察确实成立”的实验，比如极端场景、替代机制对照、可视化、或同类通用方法对照。
tokens used
247,133
我没有额外联网，这次判断基于本地 PDF 的方法、相关工作和消融。你当前列了 6 篇，我按这 6 篇逐篇拆。

**1. Multi-Model Synergy Perception for Open-World Person Re-Identification**

1. 触发观察：作者先做了一个探索实验，用普通多源 DG ReID 训练 4 个源域，发现小规模域先收敛，大规模域还没学完时，小规模域和开放测试域已经开始退化。这个观察把问题从“多源数据越多越好”拉回到“多源规模不平衡会导致训练崩塌”。

2. 重定义动作：他们把旧问题“学习域不变特征”改写成“在保持域多样性的同时构造域平衡”。关键词是 biased source domains、domain balance、domain diversity、asynchronous training、divide and conquer、synergy perception。核心话术是，简单重采样会丢掉大域知识，简单联合训练会被规模偏置拖垮，所以需要一种不丢信息的平衡学习范式。

3. 机制怎么长出来：ATP 很自然，大域拆成多个批次，每个批次和小域共享批次组合成平衡子域，再用多个子模型并行学。接着他们说多子模型会带来“各自只看一部分知识”的问题，于是用 SSP 对齐特征关系矩阵，用 CCD 做中心 logit 蒸馏，用 PGD 做部位关系图蒸馏。ATP 和重定义绑定最紧，SSP/SDP 是从“多子模型副作用”推出的补丁，PGD 稍微有堆模块感。

4. 证据闭环：最关键不是 SOTA 表，而是 Fig.1 的训练退化观察、ATP/SSP/CCD/PGD 的逐项消融、极端域不平衡设置下的敏感性实验、和集中训练或简单集成的比较。可视化用 Grad-CAM 和 t-SNE 说明 SDP 后关注区域更完整、域分布更紧。这个闭环在证明“域规模偏置确实存在，平衡子域加协同学习确实对症”。

5. reviewer 为什么买账：这篇卖的主要是视角和训练范式，不是某个单独模块。真正 novelty 来自把开放世界 ReID 的多源训练失败归因到“规模不平衡导致的 collapse risk”，再给出一个能保持大域信息的异步并行范式。

**2. Multi-Scale Dynamic Fusion for Visible-Infrared Person Re-Identification**

1. 触发观察：VI-ReID 中可见光和红外的成像差异导致清晰度、纹理、噪声、背景和姿态都不一致。作者还抓了一个测试期现象，训练时 VIS 和 IR 可以按批次平衡，但测试时一个 query 对多个 gallery，query-gallery 和 gallery-gallery 的信息量不均衡会影响匹配。

2. 重定义动作：他们没有只说“模态差距”，而是把问题拆成复杂动态场景下的局部细节、长程上下文、多尺度空间结构、测试期相似度噪声。关键词是 multi-scale dynamic fusion、dynamic perceptual enhancement、feature reweighting with similarity、local and long-range dependencies、information imbalance。

3. 机制怎么长出来：DFF 用动态卷积分支抓局部，用注意力分支抓长程依赖。DPE 用不同 dilation 的 deformable dilated convolution 生成多尺度 embedding。MHC loss 让跨模态 embedding 拉近，让同模态生成 embedding 保持差异，OGM loss 抑制冗余。FRS 在匹配阶段利用 query-gallery 和 gallery-gallery 的 top-k 相似度增强。机制和问题是一一对应的，但整体更像“把每个痛点配一个模块”，不是从单一观察自然推出。

4. 证据闭环：他们做了 DFF、DPE、FRS 的逐项消融，也做了 MHC 和 OGM 的损失消融。还有模块插入 stage 的实验、DPE 和已有 DEE 模块的对比、DPE 分支数和超参敏感性、t-SNE、类内类间距离、Grad-CAM 和检索结果可视化。闭环重点是证明每个动态因素都能单独贡献，而不是一个模块偶然涨点。

5. reviewer 为什么买账：这篇主要卖机制组合，不是特别锐利的新视角。novelty 来源是“动态、多尺度、局部加全局、训练加匹配”的完整工程闭环。对我们有用的教训是，如果观察不够尖，消融就必须非常密，才能让组合稿显得可信。

**3. Multi-year long-term person re-identification using gait and HAR features**

1. 触发观察：普通 ReID 默认外观短期稳定，但真实长时场景里衣服、光照、地点和年份都会变。作者把场景放到 2020 到 2023 的越野跑视频，外观线索不可靠，步态相对稳定，但单纯步态又缺少行为上下文。

2. 重定义动作：他们把 ReID 从“图像外观匹配”重定义成“跨地点、跨年份的运动生物特征识别”。HAR 也不是作为动作分类任务，而是作为步态身份表示的上下文补充。关键词是 long-term、spatio-temporal ReID、gait、HAR、cross-year、motion-based multimodal representation、context-aware activity cues。

3. 机制怎么长出来：先用检测、跟踪和分割得到 silhouette 序列，再用 gait backbone 提取部位级步态特征。方法里一个细节很关键，他们承认 BiLSTM 处理的是身体水平部位序列，不是真正时间序列，用它建模部位之间的结构关系。HAR 分支用预训练视频 backbone，做 context-constrained crop，最后让 gait 作为 query 去 attend HAR，保证身份主线仍由 gait 主导。机制和重定义绑定比较紧，技术本身不新，但角色分配清楚。

4. 证据闭环：关键证据是新数据集的 same-year 和 cross-year 协议、外观模型、gait-only 和 gait-HAR 的对比、CMC 曲线，以及 cross-attention、cosine fusion、concatenation 的融合策略消融。他们还强调 HAR 在更难的跨年设置里增益更明显，这直接支撑“行为上下文在长时 ReID 中有价值”。

5. reviewer 为什么买账：这篇卖的是问题设定和证据，不是复杂模块。novelty 来源是长时真实数据、跨年评测和“HAR 作为步态上下文”的视角。审稿人买账的前提是，这个场景足够真实，且对照能说明不是普通视频 backbone 更强，而是 HAR 补了长期身份线索。

**4. Mutual Distillation Driven Dual-Space Matching for Visible-Infrared Person Re-Identification**

1. 触发观察：VIS 有颜色和纹理，IR 是灰度、边界模糊、细节少，导致特征空间错位。作者还指出图像级生成方法需要严格配对、成本高；特征级方法更稳定，但常只学到轮廓和粗结构，深层语义还会丢失。

2. 重定义动作：他们把 VI-ReID 的瓶颈重定义成两个问题：低层和高层语义难以跨模态对齐，VIS 和 IR 在无边界 latent space 中存在分布不一致。关键词是 dual-level fusion、modality invariant augmentation、dual-space matching、Euclidean space、Hilbert space、mutual distillation、pixel-level alignment。

3. 机制怎么长出来：DLF 在浅层融合低级细节和高级语义，MIA 用水平和垂直池化、分布 mask、多尺度卷积注意力提取细粒度共享信息。DSM 是主叙事，Hilbert Space Matching 用 HSIC 约束全局统计依赖，Euclidean Space Matching 用 EMA teacher classifier 和双向 KL 做互蒸馏，保局部判别和预测一致性。DSM 和“欧氏不够、希尔伯特补全局”的重定义绑定最紧。

4. 证据闭环：Table III 做 DLF、MIA、HSM、ESM 的组件消融，Table IV 验证 DLF 和 MIA 放在哪些层最合适。还有 α 和 τ 的参数分析、VIS-IR cosine distance 分布、t-SNE、Grad-CAM 和 Rank-10 检索可视化。它的证据闭环是证明双空间不只是换名词，确实让跨模态分布更紧、注意区域更对。

5. reviewer 为什么买账：这篇卖的是视角加机制。真正 novelty 是把常见的“跨模态对齐”包装成“欧氏空间保局部身份差异，希尔伯特空间保全局统计依赖，再用互蒸馏让二者一致”。这比单纯说加一个 alignment loss 更像可审的理论化问题定义。

**5. Nearest Neighbor Sample Constraint and ODE Guided Feature Reconstruction for Unsupervised Person Re-Identification**

1. 触发观察：无监督 ReID 的伪标签错误主要集中在聚类边界。聚类中心对比学习稳定但太粗，不处理实例级边界 hard sample；小批次 hard mining 又只能看局部样本，和全局 memory bank 不匹配。同时，作者认为普通残差连接缺少对特征演化过程的建模。

2. 重定义动作：他们把“伪标签噪声”重定义成“最近邻类别边界样本没有被精细约束”。关键词是 nearest neighbor sample constraint、cluster boundary、hard samples、inter-class balance、pseudo-label quality、ODE guided feature reconstruction。这个重定义比“改进聚类”更具体，因为它指出了错误发生的位置。

3. 机制怎么长出来：NNSC 保留聚类中心 memory 维持稳定，同时在实例 memory 中为每个 query 找最远正样本和最近负样本，专门拉开最近邻类别边界。CB loss 用凹函数和 Jensen 不等式解释，当分类概率接近均匀时说明样本处于混淆区，最小化它就是把 hard sample 推离边界。ODE-FR 用三阶隐式 Adams 公式构造残差模块，插到 ResNet 第三层增强表示。NNSC 和 CB 与重定义绑定很紧，ODE-FR 更像第二条“表示能力不足”的补强线。

4. 证据闭环：最关键的是 CB 前后的 t-SNE 边界样本可视化、NNSC/CB/ODE-FR 的逐项消融、NNSC 和传统 contrastive loss 的对比、CB 和其他 loss 的对比、ODE-FR 插入位置和替代模块对比、正负样本距离分布、Top-8 检索可视化。这个闭环能支撑“边界 hard sample 被更好地区分”，但 ODE 理论本身的证据相对弱一些。

5. reviewer 为什么买账：这篇主要卖机制，但机制有一个明确失败点托住。novelty 不是“又挖 hard sample”，而是“中心稳定约束加全 memory 最近邻实例修正”。ODE-FR 提供额外新鲜感，不过真正承重的是边界样本重定义。

**6. Occluded person Re-Identification with noise injection**

1. 触发观察：遮挡 ReID 里车辆遮挡很常见，论文说 Occluded-Duke query 中 vehicle-induced occlusion 占比超过 40%，但车辆图像通常被当成无关噪声丢掉。作者反过来观察到，车辆图像有真实纹理、颜色、视角和相机偏置，而且很容易从车牌系统收集。

2. 重定义动作：这是最值得学的一篇。它把“车辆是噪声”改写成“车辆是有结构的正向噪声”。关键词是 positive noise、positive-incentive noise、structured cross-domain noise、data-centric perspective、vehicle data as valuable training resource、without manual annotations。旧问题是遮挡建模，新问题是如何用真实跨域干扰物降低遮挡识别的不确定性。

3. 机制怎么长出来：INI 很自然，既然车辆会真实遮挡行人，就把真实车辆块贴到行人下半身，形成更物理合理的遮挡样本。ENI 则在 epoch 级别交替车辆和行人，让模型不只记住人类外观模式。网络本身是双分支全局和局部特征，损失是常规 ID loss 加 triplet。机制简单，但和重定义绑定紧，尤其 INI 几乎是从观察直接推出的。

4. 证据闭环：他们做了 ENI only、INI only、ENI+INI 的组件消融，结果显示 INI 单独贡献最大，说明核心不是周期训练，而是真实车辆遮挡模拟。还和 MixUp、CutMix、Hide-and-Seek、GridMask、color jitter 比较，证明不是普通增广。再加 ENI 间隔、INI 高度比例、车辆分辨率、池化类型和推理速度消融，闭环是在证明“车辆作为结构化遮挡源”比随机遮挡和通用增广更对题。

5. reviewer 为什么买账：这篇卖的是视角，不是模块。机制很朴素，但“把会被丢弃的车辆数据重定义为 positive-incentive noise”有反直觉性，又有现实采集优势和强对照消融，所以审稿人容易接受它是数据中心方法创新，而不是普通 copy-paste。

共同套路可以概括成三句话：能发的方法不是先有模块，而是先把一个具体失败现象改写成更窄、更可验证的新问题。机制最好像是从这个新问题自然长出来的，至少要让每个模块都有明确的失败点对应。证据上不能只给 SOTA 和组件消融，还要有一个能反过来证明“原始观察确实成立”的实验，比如极端场景、替代机制对照、可视化、或同类通用方法对照。
