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
session id: 019ef836-001a-7e91-ab9d-5a69e1feee5d
--------
user
你是 ReID 创新方法论拆解员 24。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(24)==
Shape-centered representation learning for visible-infrared person re-identification.pdf
Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf
Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf
Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf
Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会按项目规则先确认当前目标和规则文件是否可读，然后用 PDF 技能和 `pdftotext` 逐篇读方法部分；如果论文自身不足以判断视角的新意，我再补少量联网核对先例。
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
/bin/zsh -lc "rg --files -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
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
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
Latent Diffusion-Guided Feature Inpainting for Occluded Person Re-Identification With Hybrid Re-Ranking.pdf
CVAF - A CLIP-Based View-Consistent Alignment Framework for Aerial-Ground Person Re-Identification.pdf
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

exec
/bin/zsh -lc "pwd && rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'AGENTS.md' -g 'STATUS.md' .. ../.. ../../.. ../../../.. 2>/dev/null" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 353ms:
../../../CLAUDE.md
../../../AGENTS.md
../../../GOAL.md
../../../../prcv/prcv-paper-writing/SOLIDER-REID/CLAUDE.md
../../../../prcv/prcv-paper-writing/SOLIDER-REID/AGENTS.md
../../../experiments/STATUS.md
../../STATUS.md
../../../../SOLIDER-REID/CLAUDE.md
../../../../SOLIDER-REID/AGENTS.md
../../../../prcv/SOLIDER-REID/CLAUDE.md
../../../../thundera/AGENTS.md
../../../../prcv/SOLIDER-REID/AGENTS.md
../../../../reid-paper/CLAUDE.md
../../../../reid-paper/AGENTS.md
../../../../reid-paper/GOAL.md
../../../../reid-paper/experiments/STATUS.md
../../../../work/2606/CLAUDE.md
../../../../work/2606/reid-opl/CLAUDE.md
../../../../work/2606/reid-opl/AGENTS.md
../../../../work/2606/reid-autolab/CLAUDE.md
../../../../work/2606/reid-autolab/GOAL.md
../../../../work/2606/reid-autolab/experiments/STATUS.md
../../../../browser-ext/thundera-github/AGENTS.md

codex
我已经确认本轮任务是 PDF 方法论拆解，并读取了 PDF 技能说明。现在先补读项目的 `GOAL.md`、`CLAUDE.md` 和实验状态，确保不和本工作区纪律冲突。
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
/bin/zsh -lc "sed -n '1,240p' ../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,260p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
项目目标仍然是读文献做差距分析，和你这轮要求一致。我接下来会逐篇抽取前 8 页文本，重点抓作者如何从观察过渡到问题定义、机制和证据。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdfinfo 'Shape-centered representation learning for visible-infrared person re-identification.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           
Subject:         
Keywords:        
Author:          
Creator:         LaTeX with hyperref
Producer:        pdfTeX-1.40.25
CreationDate:    Tue Apr 29 09:15:59 2025 CST
ModDate:         Tue Apr 29 09:15:59 2025 CST
Custom Metadata: yes
Metadata Stream: no
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           33
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       1772957 bytes
Optimized:       no
PDF version:     1.5

hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdfinfo 'Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           
Subject:         
Creator:         LaTeX with hyperref
Producer:        LuaHBTeX, Version 1.18.1 (MiKTeX 24.4)
CreationDate:    Wed Mar 12 20:17:50 2025 CST
ModDate:         Wed Mar 12 20:17:50 2025 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           19
Encrypted:       no
Page size:       486 x 720 pts
Page rot:        0
File size:       4249881 bytes
Optimized:       no
PDF version:     2.0

hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdfinfo 'Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification
Subject:         IEEE Transactions on Circuits and Systems for Video Technology; ;PP;99;10.1109/TCSVT.2026.3670874
Creator:         LaTeX with hyperref
Producer:        pdfTeX-1.40.26; modified using iText® 7.1.12 ©2000-2020 iText Group NV (AGPL-version); modified using iText® Core 7.2.4 (AGPL version) ©2000-2022 iText Group NV
CreationDate:    Sat Jul 26 15:50:52 2025 CST
ModDate:         Thu Mar  5 07:25:45 2026 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           15
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       4610796 bytes
Optimized:       no
PDF version:     1.5

exec
/bin/zsh -lc "pdfinfo 'Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           
Subject:         
Keywords:        
Author:          
Creator:         LaTeX with hyperref
Producer:        pdfTeX-1.40.25
CreationDate:    Thu Dec 12 09:51:19 2024 CST
ModDate:         Thu Dec 12 09:51:19 2024 CST
Custom Metadata: yes
Metadata Stream: no
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           14
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       3154098 bytes
Optimized:       no
PDF version:     1.5

exec
/bin/zsh -lc "pdfinfo 'Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification
Subject:         Pattern Recognition, 168 (2025) 111813. doi:10.1016/j.patcog.2025.111813
Keywords:        Feature level augmentation,Vision Transformer,Video-based person re-identification
Author:          Minjung Kim
Creator:         Elsevier
Producer:        Acrobat Distiller 8.1.0 (Windows)
CreationDate:    Mon Jun  9 17:17:03 2025 CST
ModDate:         Mon Jun  9 17:24:13 2025 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          yes
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           11
Encrypted:       no
Page size:       595.276 x 793.701 pts
Page rot:        0
File size:       3506760 bytes
Optimized:       no
PDF version:     1.7

exec
/bin/zsh -lc "pdfinfo 'SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           SPCL: Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval
Subject:         IEEE Transactions on Circuits and Systems for Video Technology;2026;36;2;10.1109/TCSVT.2025.3601071
Creator:         LaTeX with hyperref package
Producer:        pdfTeX-1.40.18; modified using iText® Core 7.2.4 (AGPL version) ©2000-2022 iText Group NV
CreationDate:    Tue Feb  3 22:57:39 2026 CST
ModDate:         Wed Feb 11 10:25:28 2026 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           14
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       6836471 bytes
Optimized:       no
PDF version:     1.5

codex
第一页到第八页通常覆盖引言、相关工作和方法主体。我会先保留这些上下文，因为“触发观察”和“重定义动作”经常藏在引言里的反驳句，而不只是在方法小节里。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Shape-centered representation learning for visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Shape-centered Representation Learning for Visible-Infrared Person
Re-identification
Shuang Li,a , Jiaxu Lenga , Ji Gana , Mengjingcheng Moa , Xinbo Gao,a,∗

arXiv:2310.17952v3 [cs.CV] 28 Apr 2025

a. Chongqing Key Laboratory of Image Cognition, Chongqing University of Posts and Telecommunications, Chongqing
400065, China.

Abstract
Visible-Infrared Person Re-Identification (VI-ReID) plays a critical role in all-day surveillance systems. However, existing methods primarily focus on learning appearance features while overlooking
body shape features, which not only complement appearance features but also exhibit inherent robustness to modality variations. Despite their potential, effectively integrating shape and appearance
features remains challenging. Appearance features are highly susceptible to modality variations and
background noise, while shape features often suffer from inaccurate infrared shape estimation due to
the limitations of auxiliary models. To address these challenges, we propose the Shape-centered Representation Learning (ScRL) framework, which enhances VI-ReID performance by innovatively integrating shape and appearance features. Specifically, we introduce Infrared Shape Restoration (ISR)
to restore inaccuracies in infrared body shape representations at the feature level by leveraging infrared appearance features. In addition, we propose Shape Feature Propagation (SFP), which enables
the direct extraction of shape features from original images during inference with minimal computational complexity. Furthermore, we design Appearance Feature Enhancement (AFE), which utilizes
shape features to emphasize shape-related appearance features while effectively suppressing identityunrelated noise. Benefiting from the effective integration of shape and appearance features, ScRL
demonstrates superior performance through extensive experiments. On the SYSU-MM01, HITSZVCM, and RegDB datasets, it achieves Rank-1 (mAP) accuracies of 76.1% (72.6%), 71.2% (52.9%),
and 92.4% (86.7%), respectively, surpassing existing state-of-the-art methods. The code will be released at https://github.com/Visuang/ScRL.
Keywords: VI-ReID, Shape Feature Propagation, Infrared Shape Restoration, Appearance Feature
Enhancement.

Preprint submitted to Pattern Recognition

April 29, 2025

Infrared

Visible

Infrared

Shapes

Images

Visible

Pedestrian A

Pedestrian B

Figure 1: The visible (infrared) images and their corresponding body shapes and the orange box indicate an incorrect area of
the infrared body shape.

1. Introduction
Person re-identification (ReID) aims to identify specific individuals across non-overlapping camera views, playing a crucial role in intelligent surveillance systems [1]. Consequently, it has attracted
significant attention from researchers and has seen rapid advancements in recent years [2]. However,
most existing methods are limited to scenarios where pedestrians are visible only during daylight, relying heavily on visible appearances. This limitation leads to notable performance degradation when
matching pedestrians captured by both visible (VIS) and infrared (IR) cameras. To address this issue,
the visible-infrared person re-identification (VI-ReID) task [3] was introduced, aiming to enable the
retrieval of pedestrians across the distinct spectra of IR and VIS [4]. In contrast to the extensively
studied ReID within the visible spectrum, the VI-ReID presents significantly greater challenges. This
difficulty arises primarily due to the substantial intra- and inter-modality variations between images
captured in the VIS and IR spectra [5].
While existing VI-ReID methods predominantly emphasize modality-shared appearance cues, incorporating body shape features can provide additional identity-discriminative information. Since
shape and appearance features are inherently complementary, leveraging both is essential for robust
person ReID. To further highlight the importance of body shape, we identify three key reasons why
it should be considered alongside appearance features. 1) The body shape’s natural resistance to
modality changes is a primary reason. As illustrated in Figure. 1, there is no discrepancy in body
shape between IR and VIS images. 2) The identity-discriminative nature of body shape is another
crucial factor. As shown in Figure. 1, pedestrian A is slightly heavier than pedestrian B, which
∗ Corresponding author E-mail: gaoxb@cqupt.edu.cn.

2

is evident in their global body shapes and local characteristics such as facial shape, hair shape, and
limb shape. Therefore, body shape analysis can aid in pedestrian identification, even when changes
in modality make color texture features unreliable. 3) Body shape estimation can be accomplished
using the pre-trained human parsing model, thereby eliminating the need for human annotation [6]. Additionally, single-modality ReID methods have demonstrated success in leveraging body
shape cues [7].
Nevertheless, when applying body shape estimation to VI-ReID images, as illustrated in Figure.
1, inaccuracies occur in the body shapes extracted from infrared images. These inaccuracies are primarily observed in the limbs, appearing as missing or incorrectly represented local shapes. This issue
occurs because the pedestrian’s skin color is very similar to the background color in infrared images,
causing the human parsing model to mistakenly identify exposed arms and legs as background. Although body shape does indeed carry identity-related information within the range of modality-shared
cues, the presence of these inaccuracies in infrared body shapes limits the effective utilization of these
cues. Moreover, although body shape contributes to pedestrian identification, relying solely on it is
insufficient, as VIS (IR) images contain richer identity cues, such as clothing, facial features, and hair.
Shape and appearance are inherently complementary—shape provides modality-invariant structural
information, while appearance captures fine-grained identity details. However, extracting reliable appearance features remains challenging due to modality-specific noise (e.g., color in visible images,
temperature variations in infrared images) and background clutter. Importantly, identity-relevant appearance features exhibit a strong correlation with body shape, whereas noise and background elements do not. To fully leverage body shape, it is essential not only to extract discriminative
shape representations but also to enhance appearance features by exploiting their correlation
with body shape. Integrating these appearance features with shape representations results in a
more comprehensive, identity-discriminative person representation.
In the field of VI-ReID, two methods closely related to body shape are CMMTL [8] and SEFL
[9]. As shown in Figure 2(a), CMMTL implicitly learns shape features by using human parsing as an
auxiliary task. However, this approach fails to effectively address the potential issues with infrared
shape representations and does not explore the relationship between shape features and appearance
features. In contrast, SEFL, as shown in Figure 2(b), assumes that body shape cues are unreliable and
seeks to obtain diverse modality-shared features by disentangling and discarding potentially unreliable shape features. While SEFL achieves competitive performance, we argue that discarding body

3

ReID
Model

Shape
Extract

ReID
Model

Shape
ReID
Extract restitute Model

decouple
Mask
Generation

enhance

ReID Train

guide

supervise

identity
(a) multi-task learning

guide

supervise

identity

test
supervise

discard

identity

(b) learning non-shape feature

concat
supervise
supervise
test
identity
identity
(c) learning shape-centered feature

Figure 2: Framework comparison of VI-ReID methods that explore the utilization of body shape. (a) Learning the features
associated with shapes through multi-task learning in CMMTL[8]. (b) Learning diverse appearance features through decoupling
and discarding shape features in SEFL[9]. (c) Learning shape features and enhancing appearance features through shape
features.

shape features overlooks their inherent identity-discriminative potential and robustness to modality
variations. In contrast to SEFL’s perspective, we contend that properly leveraging body shape cues
can significantly enhance VI-ReID performance. To this end, we focus on designing effective strategies to explicitly extract robust body shape features and enhance appearance feature representations
by exploring the underlying relationship between shape and appearance features.
Based on the above analysis, we propose the Shape-centered Representation Learning (ScRL)
framework that explicitly integrates both shape and appearance features to construct a modality-robust
pedestrian representation, addressing the limitations of prior approaches. As illustrated in Figure 3,
the proposed framework comprises two branches: the appearance stream and the shape stream, incorporating three key components: Infrared Shape Restoration (ISR), Shape Feature Propagation (SFP),
and Appearance Feature Enhancement (AFE). The appearance stream encodes appearance features,
while the shape stream encodes shape features. Given the inaccuracies in infrared shapes, we first propose ISR to capture the missing infrared shape features from appearance features to restore infrared
shape features, thus enabling the shape stream to encode better shape features. To further improve
computational efficiency during inference, we introduce SFP, which transfers the capabilities of the
shape stream to the appearance stream, allowing the model to directly extract shape features from
pedestrian images without requiring an additional auxiliary network during inference. Finally, we incorporate AFE to enhance appearance features. Utilizing a two-stage cascaded attention mechanism,
AFE, directly and indirectly, emphasizes shape-related features while suppressing identity-unrelated
features, thereby obtaining shape-related appearance features. Through the interaction of ISR, SFP,
and AFE, our framework enables the shape and appearance streams to mutually refine each other,
4

leading to a more discriminative and stable person representation across different modalities.
Our main contributions are summarized as follows:
• We propose a novel framework that leverages the complementarity between shape and appearance features to construct a more robust cross-modality pedestrian representation, effectively
mitigating the impact of modality variations on recognition performance.
• We introduce Infrared Shape Restoration (ISR), which restores inaccurate infrared body shapes
from human parsing networks, enhancing the discriminative ability of shape features.
• We design the Appearance Feature Enhancement (AFE) to boost appearance features by leveraging the inherent relationship between shape and appearance features.
• Extensive experimental results on SYSU-MM01, RegDB, and HITSZ-VCM datasets show that
the proposed ScRL achieves a new state-of-the-art performance.
The rest of this paper is organized as follows. Section II introduces related work; Section III elaborates the proposed method; Section IV analyzes the comparative experimental results; and Section
V concludes this paper.

2. Related work
2.1. Visible Person Re-Identification
Visible Person Re-Identification (ReID) aims to match visible pedestrian images with the same
identity under non-overlapping cameras. With the introduction of large-scale datasets [10], visible
person ReID based on deep learning has rapidly developed [11]. In order to directly conduct end-toend training in the expected embedding space, [12] improved the hard sample mining of the classic
triplet loss, which improved the discriminability of pedestrian features. In order to obtain fine-grained
discriminative features, PCB[13] proposes to horizontally divide pedestrian feature maps into 6 parts
to learn part-level features. In addition, changes in illumination, pose, and perspective also pose
challenges for extracting discriminative pedestrian features. In response to the issue of illumination
changes, IID[14] proposes to eliminate the adverse effects of illumination changes by decoupling illumination features and identity features. In order to align with the standard posture of pedestrians,
PIE[15] introduces the PoseBox structure to obtain pose-invariant embedded features. In response to

5

the adverse impact of camera style changes on pedestrian matching, Camstyle[16] proposed using CycleGAN to achieve transfer between different camera styles, smoothing out camera style differences at
the data level. To improve the efficiency and accuracy of ReID models in real-world applications, various methods have been proposed to enhance representation learning and improve feature embeddings.
SAT [17] employs a Siamese network for similarity learning, enhancing re-identification accuracy and
tracking stability. Fast re-OBJ [18] introduces an efficient embedding learning strategy for real-time
rigid object ReID, improving feature discriminability while maintaining a high processing speed of
15Hz on a standard PC. IO-ReID [19] enhances object re-identification by optimizing embedding generation, improving retrieval accuracy while maintaining real-time efficiency in cluttered rigid scenes.
However, these methods are primarily designed for feature learning within the visible spectrum and
struggle to adapt to all-day surveillance systems, especially in nighttime scenarios where near-infrared
cameras are commonly used, significantly limiting their performance.
2.2. Visible-Infrared Person Re-Identification
In order to match pedestrians from different modalities, researchers have contributed a lot of
excellent work, VI-ReID methods can be roughly divided into two main categories: feature-level
modality alignment and image-level modality alignment methods.
The feature-level modality alignment methods aim to learn modality-shared features by aligning IR and VIS features at the feature level. HSME [20] maps features of different modalities to
a unified hypersphere. MBCE [21] proposes a memory-based prototype feature learning method to
suppress the modality discrepancy. MRCN [22] reduces the modality discrepancy by decoupling the
modality-relevant and modality-irrelevant features. To more effectively mine diverse cross-modality
cues, MPANet [23] is designed to discover the nuanced modality-shared features. DEEN [24] enhances the embedding representation in the embedding space by generating diverse embeddings.
The image level modality alignment methods alleviate modality differences by generating images
with the target or intermediate modality styles. D2 RL [25] transfers the style of IR (VIS) images to
VIS (IR) images through the GAN network, compensating for the missing modality information. XIV
[26] transform IR and VIS images into auxiliary X-modality images respectively and perform X-IRVIS three-mode learning. SMCL [27] generates syncretic modality images that contain information
from both modalities to steer modality-invariant feature learning. However, the modality-shared cues
(like shape-centered cues) have not been fully explored, which limits the discriminability of features.

6

2.3. Body shape for Person Re-Identification
With the rapid advancement of deep learning, ReID has made significant progress. However,
traditional methods heavily rely on color and texture features, leading to performance degradation under clothing changes or modality variations (e.g., visible-infrared transformation). Researchers have
explored body shape as a more stable biometric feature to address this issue to enhance ReID robustness. To mitigate the impact of clothing changes, FSAM [7] introduces interactive mutual learning,
transferring knowledge from the shape stream to the appearance stream to improve cloth-invariant
feature representation. Similarly, GI-ReID [28] adopts a two-stream framework, where an auxiliary
Gait-Stream assists the main ReID-Stream in learning gait-based identity features. By incorporating
Gait Sequence Prediction (GSP), GI-ReID extracts temporal gait cues from a single image, significantly improving robustness against clothing variations. In gait-based ReID, continuous body shapes
are utilized to learn gait features. To this end, GaitPart [29] enhances feature extraction through local
temporal modeling, capturing fine-grained motion details from body parts. GaitBase [30] simplifies
gait recognition pipelines, improving generalization across diverse scenarios. Furthermore, DeepGaitV2 [31] leverages deep convolutional architectures to enhance gait feature learning, achieving
superior performance in challenging environments with occlusions and background clutter. In VIReID, body shape is used to improve cross-modality feature alignment. CMMTL [8] utilizes body
shape as a semantic label, jointly training VI-ReID and human semantic segmentation to implicitly
learn shape-related features. In contrast, SEFL [9] employs disentanglement learning to discard shape
features, focusing on modality-shared appearance representations to enhance generalization across
modalities.
Compared to their work, our method aims to integrate modality-robust shape and appearance
features while leveraging the interaction between shape and appearance to enhance their respective
feature representations.

3. Proposed Method
3.1. Preliminaries and Overview
vis , xir , y )|N } represent the training dataset, where N reprePreliminaries. Let X = {(xivis , xiir , xs,i
s,i i i=1

sents the total number of pedestrian images, yi ∈ {1, 2, · · · , K} represents its corresponding identity
vis (xir )
label, K represents the total number of identities. xivis (xiir ) represents the i-th VIS(IR) image, xs,i
s,i

7

Shape stream
Fˆs ,i

I
S
R

Lsid

f s,i

Fs,iir, j

Wids

Lkd -ce

Fi ir , j

 GeM
query

key

value

Stage 1
I
S
R

Layer4 of ResNet

Feature vector

key

value

Appearance stream

AFE
Cross
Attention

Infrared Images

Lqid Lqwrt

Cross
Attention

Ea

F̂s,iir, j

query

Visible Images



value

ISR

Fs,i

Fi

query

key

GeM
Es



Wids

SFP
Fi ir ,2

Fi ir ,1

f̂ s,i

Lkd

Es

Infrared Shapes

GeM

Cross
Attention

I
S
R

Visible Shapes

Lswrt


GeM
Laid Lawrt

Stage 2



Feature map Infrared Shape Restoration Element-wise addition

Data flow

Loss flow

Figure 3: The pipeline of our proposed ScRL framework consists of two branches: the shape stream and the appearance
stream. The shape stream includes the shape feature learning network Es and Infrared Shape Restoration (ISR). Es encodes
the input shape into shape features F̂s,i , with ISR applied at an intermediate stage to restore erroneous infrared shape features by
leveraging appearance features Fiir,1 and Fiir,2 . The appearance stream comprises the appearance feature learning network Ea ,
the shape sub-network Es̃ , and Appearance Feature Enhancement (AFE). Ea encodes the pedestrian image into appearance
features Fi . To improve inference efficiency, Es̃ encodes shape features F̄s,i from the third block of Ea , guided by Shape
Feature Propagation (SFP). Finally, AFE employs a cascaded two-stage cross-attention mechanism, enhancing the interaction
between F̄s,i and Fi , which results in shape-centered pedestrian feature representations.

represents the body shape corresponding to xivis (xiir ), which is obtained by feeding xivis (xiir ) into the
pre-trained Self-Correction Human Parsing (SCHP) [32] model.
Overview of ScRL. As shown in Figure.3, we first extract the pedestrian appearance feature map
Fi through the image encoder Ea of the appearance stream, while introducing the shape sub-network
Es̃ to encode the shape feature F̄s,i from the features outputted by the third block of Ea . In shape
stream, to restore inaccurate IR shape features, Infrared Shape Restoration (ISR) employs a crossattention mechanism, applying it to the features outputted by the intermediate block of Ea and Es , to
obtain the restorative infrared shape feature map and fed into the subsequent blocks of Es to extract
the final IR shape feature. And the VIS shape feature is directly obtained by feeding the visible shape
xivis into Es , without involving ISR. For improved computational efficiency during inference, F̄s,i is
fed into SFP, transferring the capability of the shape stream to the appearance stream through prototype and instance-level distillation. To enhance appearance features, Appearance Feature Enhancement (AFE) introduces a cascaded two-stage cross-attention mechanism, emphasizing shape-centered
appearance features by establishing interactions between F̄s,i and Fi .

8


codex
第一篇的第 8 页只到方法总览，具体的 ISR、SFP、AFE 还在后面。为了满足“完整方法部分”，我会在这种情况下继续读后续方法页，不把拆解建立在摘要级信息上。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 15 'Shape-centered representation learning for visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Shape stream
Fˆs ,i

I
S
R

Lsid

f s,i

Fs,iir, j

Wids

Lkd -ce

Fi ir , j

 GeM
query

key

value

Stage 1
I
S
R

Layer4 of ResNet

Feature vector

key

value

Appearance stream

AFE
Cross
Attention

Infrared Images

Lqid Lqwrt

Cross
Attention

Ea

F̂s,iir, j

query

Visible Images



value

ISR

Fs,i

Fi

query

key

GeM
Es



Wids

SFP
Fi ir ,2

Fi ir ,1

f̂ s,i

Lkd

Es

Infrared Shapes

GeM

Cross
Attention

I
S
R

Visible Shapes

Lswrt


GeM
Laid Lawrt

Stage 2



Feature map Infrared Shape Restoration Element-wise addition

Data flow

Loss flow

Figure 3: The pipeline of our proposed ScRL framework consists of two branches: the shape stream and the appearance
stream. The shape stream includes the shape feature learning network Es and Infrared Shape Restoration (ISR). Es encodes
the input shape into shape features F̂s,i , with ISR applied at an intermediate stage to restore erroneous infrared shape features by
leveraging appearance features Fiir,1 and Fiir,2 . The appearance stream comprises the appearance feature learning network Ea ,
the shape sub-network Es̃ , and Appearance Feature Enhancement (AFE). Ea encodes the pedestrian image into appearance
features Fi . To improve inference efficiency, Es̃ encodes shape features F̄s,i from the third block of Ea , guided by Shape
Feature Propagation (SFP). Finally, AFE employs a cascaded two-stage cross-attention mechanism, enhancing the interaction
between F̄s,i and Fi , which results in shape-centered pedestrian feature representations.

represents the body shape corresponding to xivis (xiir ), which is obtained by feeding xivis (xiir ) into the
pre-trained Self-Correction Human Parsing (SCHP) [32] model.
Overview of ScRL. As shown in Figure.3, we first extract the pedestrian appearance feature map
Fi through the image encoder Ea of the appearance stream, while introducing the shape sub-network
Es̃ to encode the shape feature F̄s,i from the features outputted by the third block of Ea . In shape
stream, to restore inaccurate IR shape features, Infrared Shape Restoration (ISR) employs a crossattention mechanism, applying it to the features outputted by the intermediate block of Ea and Es , to
obtain the restorative infrared shape feature map and fed into the subsequent blocks of Es to extract
the final IR shape feature. And the VIS shape feature is directly obtained by feeding the visible shape
xivis into Es , without involving ISR. For improved computational efficiency during inference, F̄s,i is
fed into SFP, transferring the capability of the shape stream to the appearance stream through prototype and instance-level distillation. To enhance appearance features, Appearance Feature Enhancement (AFE) introduces a cascaded two-stage cross-attention mechanism, emphasizing shape-centered
appearance features by establishing interactions between F̄s,i and Fi .

8

3.2. Appearance Feature Learning
Referring to the ReID framework named AGW [33], we adopted a dual stream network that removed the Non-local Attention as the Baseline. Specifically, a dual stream network consists of two
parallel convolutional layers with unshared parameters and four blocks with shared parameters, named
Ea . Two parallel convolutional layers independently process VIS and IR images to extract low-level
features, then these features are fed to the subsequent four blocks to extract high-level features fim .
The above process can be formalized as:

fim = GeM(Ea (xim ))

m = {vis, ir}.

(1)

where the GeM represents the Generalized-mean (GeM) Pooling layer[34]. To ensure the identity
discriminability of fim , the Cross-Entropy (CE) loss Lid and the Weighted Regularization Triplet
(WRT)[34] loss Lwrt were adopt to constrain it as follows:
Lid = −

1 nb
∑ qi log(Wid (fim )) ,
nb i=1

1 nb
∑ log(1 + exp(∑ wi,p j di,p j − ∑ wni,k dni,k ),
nb i=1
i, j
i,k
,
p
n )
exp(−d
exp(d
)
i, j
i,k
p
n
wi, j =
p , wi,k =
∑di,k ∈Ni exp(−dni,k )
∑di, j ∈Pi exp(di, j )

(2)

Lwrt =

(3)

where nb represents the batch size, Wid represents the shared identity classifier for IR and VIS pedestrian features, qi ∈ RK×1 is a one-hot vector, and only the element at yi is 1. For WRT loss, j and
k represent the index of the positive and negative samples corresponding to the anchor sample xi
within a batch, respectively. Pi and Ni represent the positive and negative set corresponding to the
anchor sample xi within a batch, respectively. And di, j represents the Euclidean distance between two
features: di, j = ∥fim − f m
j ∥2 .
3.3. Shape Feature Learning
Although the design of the dual stream network is aimed at extracting modality-shared features,
it focuses on learning pedestrian appearance features and lacks the ability to learn pedestrian body
shape features. Compared to appearance features, the body shape features of pedestrians are more
robust against modality changes. Therefore, in this section, we aim to learn the body shape features

9

reshape
q

ir, j
s,i

F

Fi ir, j



C  H W

transpose
(H W )  (H W )

reshape
k

softmax

reshape
v



F̂s,iir, j


v 2

C  H W

Figure 4: Illustration of the proposed ISR that is used to obtain missing shape features from appearance features for restoring
IR shapes in feature level.

by the SCHP and a shape feature learning network Es . In order to obtain discriminative body shape
vis generated by the SCHP to E to extract the VIS shape
features, we first feed the VIS body shape xs,i
s
vis = E (xvis ). For IR body shapes, as discussed in the introduction and shown in Figure.1,
features fs,i
s s,i

inherent errors are present within the IR body shapes, which leads to the inability of Es to obtain
adequately discriminative shape features from input IR shapes. To address this issue, we propose the
Infrared Shape Restoration (ISR) to restore infrared shapes at the feature level.
Infrared Shape Restoration: Although IR shapes may contain some inaccuracies, the corresponding original IR images still contain shape-related cues. Additionally, SCHP can effectively
parse VIS images to obtain accurate VIS shapes, which are error-free and consistent with the identities of the infrared shapes. Therefore, our motivation is to utilize the reliable VIS shape features
to guide the extraction of required shape information from the IR image features and achieve
IR shape restoration at the feature level.
However, both directly relying on IR shape features for restoration and learning shape features
solely from IR images pose challenges, as the former may introduce shape inaccuracies, while the
latter can be affected by appearance noise, making it difficult to extract clean and reliable shape
information. A naı̈ve approach, such as simple feature fusion, may fail to effectively correct these
inaccuracies, as it lacks a mechanism to adaptively search for required shape information from IR
images and utilize it to restore IR shapes. To address this, considering that shape inaccuracies are
primarily spatial, particularly in local structures such as hands and feet, we introduce a spatial crossattention mechanism that enables the model to search for required shape information from the IR
image features. Specifically, let the IR shape feature map and IR feature map output from the j-th layer
ir, j
of Es and Ea be denoted as Fs,i
and Fiir, j , respectively. To ensure more reliable attention allocation,
ir, j
we construct the query Q by integrating both IR shape feature map Fs,i
and IR appearance feature

10

map Fiir, j . This fusion not only incorporates initially estimated shape information but also captures
correct shape cues embedded in the appearance features, providing richer contextual information for
shape restoration. Moreover, since both originate from the same IR image, their summation ensures
spatial alignment, while any misalignment or anomalies indicate potential shape errors. The query Q
is computed as follows:
ir, j
Q = φq (Fiir, j + Fs,i
),

(4)

where φq represents the 2D convolution layer with kernel sizes of 1 × 1. The query Q contains richer
contextual information, which interacts with K to generate attention and searches for the required
body shape information in the value V to restore the IR body shape. The key K and value V are
represented as:
K = φk (Fiir, j ), V = φv (Fiir, j ),

(5)

where φk and φv represent the 2D convolution layer with kernel sizes of 1 × 1, respectively. Then, we
ir, j
, this process similar to self-attention, as shown below:
obtain the restored IR shape feature map F̂s,i

ir, j
ir, j
,
= φv2 (BN (N orm(QK T )V )) + Fs,i
F̂s,i

(6)

where φv2 represents the 2D convolution layer with kernel sizes of 1 × 1, BN , N orm and (·)T
represents the batch normalization layer, the normalization operation and the transpose operation.
Following the above steps, we incorporated the above attention following the first and second blocks
of Es . In order to guide the training of Es and ISR, we introduce CE loss and WRT loss to constrain
the shape features fˆs,i output from Es with a GeM pooling layer:
Lsid = −

1 nb
∑ qi log(Wids (fˆs,i )) ,
nb i=1

1 nb
s,p
s,n s,n
∑ log(1 + exp(∑ ws,p
i, j di, j − ∑ wi,k di,k ),
nb i=1
i, j
i,k
,
s,n
s,p
exp(−di,k
)
exp(di, j )
s,p
s,n
wi, j =
=
s,p , w
s,n
∑di,k ∈Nis exp(−di,k )
∑di, j ∈Pis exp(di, j ) i,k

(7)

s
=
Lwrt

(8)

where Wids represents the shared identity classifier for IR and VIS shape features, j and k represent
the index of the positive and negative samples corresponding to the anchor shape feature fˆs,i within
a batch, respectively. Pis and Nis represent the positive and negative set corresponding to the anchor

11

sample xi within a batch, respectively. And dsi, j represents the euclidean distance between two shape
features: dsi, j = ∥fˆs,i − fˆs, j ∥2 . Importantly, it should be highlighted that the VIS shape features do
not require restoration and we mix them with restored IR shape features in batch to participate in the
loss calculation. Therefore, the VIS shape features can guide the learning of the ISR at the loss level.
Shape Feature Propagation: we can obtain shape features through the collaboration of appearance feature extraction network Ea , shape feature extraction network Es , and ISR. However, this
poses challenges to model deployment due to the increased parameters and computational complexity
from SCHP and the shape stream. Therefore, it is crucial to transfer the ability of the shape stream
network to the appearance stream network so that the testing phase does not require the participation
of the shape stream network. Towards this objective, we replicate the fourth block of Ea as the shape
subnetwork Es̃ with GeM pooling layer and apply it to the output features of the third block of Ea
to obtain the shape features f˜s,i under the guidance of the shape features fˆs,i output from the Es at
instance (Eq.9) and prototype level (Eq.10):

Lkd =

1 nb ˜
∑ fs,i − fˆs,i
nb i=1

Lkd−ce = −

,

(9)

1 nb
∑ qi log( ˜fs,i ΘT ),
nb i=1

(10)

2

where ||·||2 represents l2 -norm, Θ ∈ RC×K represents the class prototype from the classifier Wids .
3.4. Appearance Feature Enhancement
To effectively leverage pedestrian appearance features in VI-ReID, it is crucial to extract identitydiscriminative information, such as clothing patterns and facial structures, while suppressing modalityspecific noise (e.g., color in visible images and temperature distributions in infrared images). However, the direct extraction of robust appearance features is highly challenging due to the inherent
variations across modalities. Notably, many identity-related appearance features exhibit strong correlations with body shape—for example, appearance features associated with head shape primarily
include hair and facial attributes, whereas background elements are clearly unrelated to shape. This
underlying relationship is typically reflected spatially. Therefore, to effectively capture this spatial
dependency, AFE adopts spatial attention for more precise alignment between shape and appearance
representations.
Given this strong correlation, we propose to utilize body shape to enhance appearance features.
12

Fs,i
C  H W

reshape
 q1
reshape
 k1

Fi
C  H W

Fi fuse
C  H W

softmax



reshape
 q2

transpose

Fi

reshape
C  H W

(H W )  (H W )

reshape
 v1

reshape
 k2

 v2

Stage 1

transpose

 v12

Lqid



Lqwrt
Stage 2

(H W )  (H W )

softmax


 v22



Laid
Lawrt

Figure 5: Illustration of the proposed AFE, which can mine shape-centered appearance features guided by shape features.

Specifically, we aim to mine modality-shared appearance features while filtering out identity-irrelevant
modality-specific features. Since only a subset of appearance features is directly associated with body
shape, we adopt a progressive extraction strategy: first identifying directly shape-related appearance
features, then leveraging them to discover indirectly shape-related appearance features that further
enrich identity representation. To achieve this, we introduce a cascading two-stage attention mechanism, systematically refining the appearance feature. The first stage utilizes shape features as a query
to extract appearance features with strong shape dependencies. The second stage then employs these
refined features along with shape features to further extract appearance features that are both directly
and indirectly linked to shape, enhancing the appearance representation.
In the first stage, let the shape feature map F̄s,i output by Es̃ serve as query Q, and the appearance
feature map Fi extracted by Ea serve as key K and value V :
Q = ψq1 (F̄s,i ), K = ψk1 (Fi ), V = ψv1 (Fi ),

(11)

where ψq1 , ψk1 , and ψv1 represents the 2D convolution layer with kernel sizes of 1 × 1. Then, similar
to ISR, the correlation score between Q and K is used to search features directly related to shape in

13

V , and fused with shape feature map F̄s,i to obtain F̃i f use .
1
F̃i f use = ψv2
(BN (N orm(QK T )V )) + F̄s,i ,

(12)

1 represents the 2D convolution layer with kernel sizes of 1 × 1. Considering F̃ f use contains
where ψv2
i

shape feature and appearance feature directly related to body shape, F̃i f use can be effectively employed
as a query during the second stage of attention, which facilitates the acquisition of appearance features
that are both directly and indirectly associated with the body shape. As the query feature of the
second-stage attention, F̃i f use plays a pivotal role in determining whether discriminative modalityshared features, both directly and indirectly associated with body shape, can be effectively extracted
from the appearance features Fi . To ensure the discriminability of the query F̃i f use , we also employ
CE loss and WRT loss to jointly constrain f˜if use = GeM(F̃i f use ):
Lqid = −

1 nb
∑ qi log(Widq (f˜if use )) ,
nb i=1

1 nb
q,n q,n
q,p
∑ log(1 + exp(∑ wq,p
i, j di, j − ∑ wi,k di,k ),
nb i=1
i, j
i,k
,
q,n
q,p
exp(−di,k
)
exp(di, j )
q,p
q,n
wi, j =
=
q,p , w
q,n
∑di, j ∈Pq exp(di, j ) i,k
∑di,k ∈N q exp(−di,k )

(13)

q
Lwrt
=

i

(14)

i

where Widq represents the shared identity classifier for infrared and visible pedestrian query features,
j and k represent the index of the positive and negative samples corresponding to the anchor shape
feature f˜if use within a batch, respectively. Piq and Niq represent the positive and negative set corresponding to the anchor f˜if use within a batch, respectively. And dqi, j represents the euclidean distance
between two query features: dqi, j = ∥f˜if use − f˜jf use ∥2 .
In the second stage, we employed the output feature F̃i f use of the first stage attention as a query to
emphasize the appearance features directly and indirectly related to the body shape in Fi as follows:
2
F̃i = ψv2
(BN (N orm(QK T )V )) + Fi ,

(15)

2 represents the 2D conwhere Q = ψq2 (F̃i f use ),K = ψk2 (Fi ),V = ψv2 (Fi ), and ψq2 , ψk2 , ψv2 and ψv2

volution layer with kernel sizes of 1 × 1. We use the enhanced appearance features f˜i = GeM(F̃i ),
which are closely related to the shape, as the final appearance features. Therefore, the loss functions

14

Algorithm 1 Training process of the proposed ScRL
vis (xir ).
Require: A mini-batch that consists of nb VIS(IR) images xivis (xiir ) and VIS(IR) shapes xs,i
s,i
Ensure: Trained ScRL model.
1: for i = 1; i < iteration; i + + do
2:
Extract Fi ,Fiir, j by inputting xivis , xiir into the appearance stream network Fa .
ir, j
ir into the shape stream network E .
3:
Extract Fi,s
by inputting xi,s
s
4:

ir, j
ir, j
Send Fiir, j ,Fi,s
to the ISR module to obtain the restored infrared shape feature F̂s,i
.

ir, j
Send F̂s,i
to the subsequent network of the shape stream Es to obtain the final shape features
ir into E ).
F̂s,i (contain VIS shape features obtained by directly inputting xi,s
s
6:
Extract F̃s,i by feeding the output features of the third block of Ea into the shape subnetwork
Es̃ .
7:
Transfer the capability to extract shape features from the shape stream network Es̃ to the appearance stream network Ea , Es̃ by the SFP module, and obtain the shape features F̄s,i
8:
Extract appearance features F̃i f use directly related to body shape by inputting F̄s,i , Fi into stage
1 of the AFE module.
9:
Extract appearance features F̃i both directly and indirectly related to body shape by inputting
F̃i f use , Fi into stage 2 of the AFE module.
10:
Optimize Ea , Es , Es̃ , ISR module and AFE module through loss functions Lsid (Eq.7),
Lswrt (Eq.8), Lkd (Eq.9), Lkd ce (Eq.10), Lqid (Eq.13), Lqwrt (Eq.14), Laid (Eq.16) and Lawrt (Eq.17).
11: end for

5:

Lid and Lwrt in Appearance Feature Learning are replaced by Laid and Lawrt to ensure the identity
discriminability and modality invariance of the appearance feature, as shown below:
Laid = −

1 nb
∑ qi log(Wid (f˜i )) ,
nb i=1

1 nb
a,p
a,n a,n
∑ log(1 + exp(∑ wa,p
i, j di, j − ∑ wi,k di,k ),
nb i=1
i, j
i,k
,
a,n
a,p
exp(−d
exp(d
)
i, j
i,k )
a,p
a,n
wi, j =
=
a,p , w
a,n
∑di,k ∈Nia exp(−di,k )
∑dai, j ∈Pia exp(di, j ) i,k

(16)

a
Lwrt
=

(17)

where j and k represent the index of the positive and negative samples corresponding to the anchor
shape feature f˜i within a batch size, respectively. Pia and Nia represent the positive and negative set
corresponding to the anchor f˜i within a batch, respectively. And dai, j represents the euclidean distance
between two features: dai, j = ∥f˜i − f˜j ∥2 .
3.5. Training and Inference
In the training process, we employ the appearance stream network Ea for extracting appearance
features and the shape stream network Es for extracting shape features. To further enhance this

15


codex
第一篇已经读到三个核心模块的细节：作者不是简单“加形状分支”，而是先指出红外解析形状会错，再把机制拆成修红外形状、把形状能力迁回主干、用形状反过来净化外观。后面我会继续对其余论文做同样粒度。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 15 -l 17 'Shape-centered representation learning for visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Algorithm 1 Training process of the proposed ScRL
vis (xir ).
Require: A mini-batch that consists of nb VIS(IR) images xivis (xiir ) and VIS(IR) shapes xs,i
s,i
Ensure: Trained ScRL model.
1: for i = 1; i < iteration; i + + do
2:
Extract Fi ,Fiir, j by inputting xivis , xiir into the appearance stream network Fa .
ir, j
ir into the shape stream network E .
3:
Extract Fi,s
by inputting xi,s
s
4:

ir, j
ir, j
Send Fiir, j ,Fi,s
to the ISR module to obtain the restored infrared shape feature F̂s,i
.

ir, j
Send F̂s,i
to the subsequent network of the shape stream Es to obtain the final shape features
ir into E ).
F̂s,i (contain VIS shape features obtained by directly inputting xi,s
s
6:
Extract F̃s,i by feeding the output features of the third block of Ea into the shape subnetwork
Es̃ .
7:
Transfer the capability to extract shape features from the shape stream network Es̃ to the appearance stream network Ea , Es̃ by the SFP module, and obtain the shape features F̄s,i
8:
Extract appearance features F̃i f use directly related to body shape by inputting F̄s,i , Fi into stage
1 of the AFE module.
9:
Extract appearance features F̃i both directly and indirectly related to body shape by inputting
F̃i f use , Fi into stage 2 of the AFE module.
10:
Optimize Ea , Es , Es̃ , ISR module and AFE module through loss functions Lsid (Eq.7),
Lswrt (Eq.8), Lkd (Eq.9), Lkd ce (Eq.10), Lqid (Eq.13), Lqwrt (Eq.14), Laid (Eq.16) and Lawrt (Eq.17).
11: end for

5:

Lid and Lwrt in Appearance Feature Learning are replaced by Laid and Lawrt to ensure the identity
discriminability and modality invariance of the appearance feature, as shown below:
Laid = −

1 nb
∑ qi log(Wid (f˜i )) ,
nb i=1

1 nb
a,p
a,n a,n
∑ log(1 + exp(∑ wa,p
i, j di, j − ∑ wi,k di,k ),
nb i=1
i, j
i,k
,
a,n
a,p
exp(−d
exp(d
)
i, j
i,k )
a,p
a,n
wi, j =
=
a,p , w
a,n
∑di,k ∈Nia exp(−di,k )
∑dai, j ∈Pia exp(di, j ) i,k

(16)

a
Lwrt
=

(17)

where j and k represent the index of the positive and negative samples corresponding to the anchor
shape feature f˜i within a batch size, respectively. Pia and Nia represent the positive and negative set
corresponding to the anchor f˜i within a batch, respectively. And dai, j represents the euclidean distance
between two features: dai, j = ∥f˜i − f˜j ∥2 .
3.5. Training and Inference
In the training process, we employ the appearance stream network Ea for extracting appearance
features and the shape stream network Es for extracting shape features. To further enhance this

15

process, we introduce the ISR to restore inaccuracies in IR shape features. Additionally, the SFP is
integrated to transfer the shape feature extraction capabilities of the shape stream to the appearance
stream. Finally, the AFE is introduced to mine appearance features that have both direct and indirect
associations with the body shape. The complete training process is executed in an end-to-end fashion,
as illustrated in Algorithm 1. As shown above, the total objective function of this method can be
formalized as:
Ltotal = La + Ls + Lkd + Lkd−ce + Lq .

(18)

where La = Laid + Lawrt , Ls = Lsid + Lswrt , Lq = Lqid + Lqwrt . During the testing process, we concatenate the shape features output by Es̃ with the appearance features output by Appearance Feature
Enhancement (AFE) for inference. Therefore, we solely utilize the appearance stream network Ea ,
Es̃ , and the AFE, without involving the shape stream network Es and the human parsing network
SCHP.

4. Experiments
4.1. Datasets
SYSU-MM01[3] is a large-scale dataset with complex environments. The training set consists of
11909 IR (22258 VIS) images of 395 identities captured across 2 IR (4 VIS) cameras. For the testing
set, there are 96 pedestrians, with a total of 3,803 IR pedestrian images and 301 randomly selected
VIS images.
RegDB[35] is a small dataset consisting of 8420 images of 421 identities captured by a single
VIS(IR) camera. Each pedestrian has 10 VIS(IR) images. We followed BDTR[36] and randomly
divided the dataset into training and testing sets for training and evaluation.
HITSZ-VCM[37] is a video-based VI-ReID dataset that contains 251452 VIS images and 211807
IR images of 927 identities, with each track containing 24 consecutive images. The training(testing)
set encompasses 11061(10802) tracks of 500(427) identities.
Evaluation Metrics We employ Cumulative Matching Characteristics (CMC), mean Average Precision (mAP), and mean Inverse Negative Penalty (mINP)[38] as evaluation metrics to assess the
performance of the ScRL and the methods compared in this paper.

16

Table 1: The Settings of different datasets of VI-ReID. “–” denotes that the dataset has no tracklets.

Datasets
SYSU-MM01[3]
RegDB[35]
HITSZ-VCM [37]

Type
Image
Image
Video

#Identites
412
491
927

#RGB cam.
1
4
12

#IR cam.
1
2
12

#Images
8240
303420
463259

#Tracklets
21863

#Evaluation
CMC+mAP
CMC+mAP
CMC+mAP

Table 2: Results of mAP and CMC (%) obtained by our proposed method and the state-of-the-art Re-ID methods on SYSUMM01. “R1”, “R10” and “R20” denote Rank-1,Rank-10 and Rank-20, respectively. These results are copied from their
papers. “–” denotes that no reported result is available. The underline indicates the second-best performance, while bold values
represent the best performance.

Methods

Reference

Zero-Pad[3]
HCML[39]
HSME[20]
D2 RL[25]
X-Modal[26]
DDAG[40]
MPANet[23]
CAJ[34]
AGW[38]
CMMTL[8]
PMT[41]
MRCN[22]
DEEN[24]
SEFL[9]
CSMSS[42]
CSC-Net[43]
ScRL(our)

ICCV’2017
AAAI’2018
AAAI’2019
CVPR’2019
AAAI’2020
ECCV’2020
CVPR’2021
ICCV’2021
TPAMI’2022
PR’2022
AAAI’2022
AAAI’2023
CVPR’2023
CVPR’2023
TMM’2024
TCSVT’2024

R1
14.8
14.3
20.7
28.9
49.9
54.8
70.6
69.9
47.5
67.3
67.5
68.9
74.7
75.2
70.6
72.7
76.1

R10
54.1
53.2
32.7
70.6
89.8
90.4
96.2
95.7
84.4
95.4
95.4
95.2
97.6
96.9
96.2
95.7
97.6

All search
R20 mAP
71.3 16.0
69.2 16.2
78.0 23.1
82.4 29.2
96.0 50.7
95.8 53.0
98.8 68.2
98.5 66.9
92.1 47.7
98.5 64.3
98.6 65.0
98.4 65.5
99.2 71.8
99.1 70.1
98.8 67.5
98.3 69.6
99.4 72.6

mINP
39.6
35.3
51.9
59.8

R1
20.6
24.5
61.0
76.7
76.3
54.2
69.6
71.7
76.0
80.3
78.4
76.0
78.6
82.4

Indoor search
R10 R20 mAP
68.4 85.8 26.9
73.3 86.7 30.1
94.1 98.4 68.0
98.2 99.6 81.0
97.9 99.5 80.4
91.1 96.0 63.0
96.7 99.0 74.4
96.7 99.3 76.5
98.3 99.7 79.8
99.0 99.8 83.3
97.5 98.9 81.2
98.1 99.6 80.2
98.3 99.6 82.1
98.8 99.8 85.4

mINP
62.6
59.2
72.7
82.2

4.2. Implementation Details
Similar to DEEN [24], we adopted ResNet50 pre-trained on ImageNet as the backbone and replaced the average pooling layer with the GeM pooling layer, with all input image sizes resized to
384 × 144. In the training phase, we adopted Random Crop, Random Horizontal Flip, Channel Random Erasing and Channel AdapGray [34] to enhance the IR(VIS) images, and we adopted Random
Crop and Random Horizontal Flip for the shape image. We adopt the Adam optimizer for optimization, the learning rate of the classifier Ws , Wa was set to 0.0007, and the learning rate of other
networks was set to 0.00035. The model trained a total of 120 epochs, in the first 10 epochs, the
learning rate is dynamically adjusted through the warmup strategy, in the 40th and 60th epochs, the
learning rate decreases by 10%. At every batch size, we randomly sample 64 images from 8 identities,
with 4 VIS images and 4 IR images for each identity and we follow the sampling settings of MITML
for HITSZ-VCM[37].

17


codex
我会额外抓每篇的消融和可视化段落，因为你要的是“证据闭环”。只读方法不够判断作者到底证明了重定义，还是只证明了模块能涨点。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Shape-centered representation learning for visible-infrared person re-identification.pdf' - | rg -n -C 3 'Ablation|ablation|visual|Visualization|shape|ISR|SFP|AFE|Figure 7|Figure 8|Table 5|Table 6|Table 7'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
9-
10-Abstract
11-Visible-Infrared Person Re-Identification (VI-ReID) plays a critical role in all-day surveillance systems. However, existing methods primarily focus on learning appearance features while overlooking
12:body shape features, which not only complement appearance features but also exhibit inherent robustness to modality variations. Despite their potential, effectively integrating shape and appearance
13-features remains challenging. Appearance features are highly susceptible to modality variations and
14:background noise, while shape features often suffer from inaccurate infrared shape estimation due to
15:the limitations of auxiliary models. To address these challenges, we propose the Shape-centered Representation Learning (ScRL) framework, which enhances VI-ReID performance by innovatively integrating shape and appearance features. Specifically, we introduce Infrared Shape Restoration (ISR)
16:to restore inaccuracies in infrared body shape representations at the feature level by leveraging infrared appearance features. In addition, we propose Shape Feature Propagation (SFP), which enables
17:the direct extraction of shape features from original images during inference with minimal computational complexity. Furthermore, we design Appearance Feature Enhancement (AFE), which utilizes
18:shape features to emphasize shape-related appearance features while effectively suppressing identityunrelated noise. Benefiting from the effective integration of shape and appearance features, ScRL
19-demonstrates superior performance through extensive experiments. On the SYSU-MM01, HITSZVCM, and RegDB datasets, it achieves Rank-1 (mAP) accuracies of 76.1% (72.6%), 71.2% (52.9%),
20-and 92.4% (86.7%), respectively, surpassing existing state-of-the-art methods. The code will be released at https://github.com/Visuang/ScRL.
21-Keywords: VI-ReID, Shape Feature Propagation, Infrared Shape Restoration, Appearance Feature
--
41-
42-Pedestrian B
43-
44:Figure 1: The visible (infrared) images and their corresponding body shapes and the orange box indicate an incorrect area of
45:the infrared body shape.
46-
47-1. Introduction
48-Person re-identification (ReID) aims to identify specific individuals across non-overlapping camera views, playing a crucial role in intelligent surveillance systems [1]. Consequently, it has attracted
--
54-studied ReID within the visible spectrum, the VI-ReID presents significantly greater challenges. This
55-difficulty arises primarily due to the substantial intra- and inter-modality variations between images
56-captured in the VIS and IR spectra [5].
57:While existing VI-ReID methods predominantly emphasize modality-shared appearance cues, incorporating body shape features can provide additional identity-discriminative information. Since
58:shape and appearance features are inherently complementary, leveraging both is essential for robust
59:person ReID. To further highlight the importance of body shape, we identify three key reasons why
60:it should be considered alongside appearance features. 1) The body shape’s natural resistance to
61-modality changes is a primary reason. As illustrated in Figure. 1, there is no discrepancy in body
62:shape between IR and VIS images. 2) The identity-discriminative nature of body shape is another
63-crucial factor. As shown in Figure. 1, pedestrian A is slightly heavier than pedestrian B, which
64-∗ Corresponding author E-mail: gaoxb@cqupt.edu.cn.
65-
66-2
67-
68:is evident in their global body shapes and local characteristics such as facial shape, hair shape, and
69:limb shape. Therefore, body shape analysis can aid in pedestrian identification, even when changes
70:in modality make color texture features unreliable. 3) Body shape estimation can be accomplished
71-using the pre-trained human parsing model, thereby eliminating the need for human annotation [6]. Additionally, single-modality ReID methods have demonstrated success in leveraging body
72:shape cues [7].
73:Nevertheless, when applying body shape estimation to VI-ReID images, as illustrated in Figure.
74:1, inaccuracies occur in the body shapes extracted from infrared images. These inaccuracies are primarily observed in the limbs, appearing as missing or incorrectly represented local shapes. This issue
75-occurs because the pedestrian’s skin color is very similar to the background color in infrared images,
76:causing the human parsing model to mistakenly identify exposed arms and legs as background. Although body shape does indeed carry identity-related information within the range of modality-shared
77:cues, the presence of these inaccuracies in infrared body shapes limits the effective utilization of these
78:cues. Moreover, although body shape contributes to pedestrian identification, relying solely on it is
79-insufficient, as VIS (IR) images contain richer identity cues, such as clothing, facial features, and hair.
80:Shape and appearance are inherently complementary—shape provides modality-invariant structural
81-information, while appearance captures fine-grained identity details. However, extracting reliable appearance features remains challenging due to modality-specific noise (e.g., color in visible images,
82:temperature variations in infrared images) and background clutter. Importantly, identity-relevant appearance features exhibit a strong correlation with body shape, whereas noise and background elements do not. To fully leverage body shape, it is essential not only to extract discriminative
83:shape representations but also to enhance appearance features by exploiting their correlation
84:with body shape. Integrating these appearance features with shape representations results in a
85-more comprehensive, identity-discriminative person representation.
86:In the field of VI-ReID, two methods closely related to body shape are CMMTL [8] and SEFL
87:[9]. As shown in Figure 2(a), CMMTL implicitly learns shape features by using human parsing as an
88-auxiliary task. However, this approach fails to effectively address the potential issues with infrared
89:shape representations and does not explore the relationship between shape features and appearance
90:features. In contrast, SEFL, as shown in Figure 2(b), assumes that body shape cues are unreliable and
91:seeks to obtain diverse modality-shared features by disentangling and discarding potentially unreliable shape features. While SEFL achieves competitive performance, we argue that discarding body
92-
93-3
94-
--
133-
134-identity
135-
136:(b) learning non-shape feature
137-
138-concat
139-supervise
--
141-test
142-identity
143-identity
144:(c) learning shape-centered feature
145-
146:Figure 2: Framework comparison of VI-ReID methods that explore the utilization of body shape. (a) Learning the features
147:associated with shapes through multi-task learning in CMMTL[8]. (b) Learning diverse appearance features through decoupling
148:and discarding shape features in SEFL[9]. (c) Learning shape features and enhancing appearance features through shape
149-features.
150-
151:shape features overlooks their inherent identity-discriminative potential and robustness to modality
152:variations. In contrast to SEFL’s perspective, we contend that properly leveraging body shape cues
153:can significantly enhance VI-ReID performance. To this end, we focus on designing effective strategies to explicitly extract robust body shape features and enhance appearance feature representations
154:by exploring the underlying relationship between shape and appearance features.
155-Based on the above analysis, we propose the Shape-centered Representation Learning (ScRL)
156:framework that explicitly integrates both shape and appearance features to construct a modality-robust
157-pedestrian representation, addressing the limitations of prior approaches. As illustrated in Figure 3,
158:the proposed framework comprises two branches: the appearance stream and the shape stream, incorporating three key components: Infrared Shape Restoration (ISR), Shape Feature Propagation (SFP),
159:and Appearance Feature Enhancement (AFE). The appearance stream encodes appearance features,
160:while the shape stream encodes shape features. Given the inaccuracies in infrared shapes, we first propose ISR to capture the missing infrared shape features from appearance features to restore infrared
161:shape features, thus enabling the shape stream to encode better shape features. To further improve
162:computational efficiency during inference, we introduce SFP, which transfers the capabilities of the
163:shape stream to the appearance stream, allowing the model to directly extract shape features from
164:pedestrian images without requiring an additional auxiliary network during inference. Finally, we incorporate AFE to enhance appearance features. Utilizing a two-stage cascaded attention mechanism,
165:AFE, directly and indirectly, emphasizes shape-related features while suppressing identity-unrelated
166:features, thereby obtaining shape-related appearance features. Through the interaction of ISR, SFP,
167:and AFE, our framework enables the shape and appearance streams to mutually refine each other,
168-4
169-
170-leading to a more discriminative and stable person representation across different modalities.
171-Our main contributions are summarized as follows:
172:• We propose a novel framework that leverages the complementarity between shape and appearance features to construct a more robust cross-modality pedestrian representation, effectively
173-mitigating the impact of modality variations on recognition performance.
174:• We introduce Infrared Shape Restoration (ISR), which restores inaccurate infrared body shapes
175:from human parsing networks, enhancing the discriminative ability of shape features.
176:• We design the Appearance Feature Enhancement (AFE) to boost appearance features by leveraging the inherent relationship between shape and appearance features.
177-• Extensive experimental results on SYSU-MM01, RegDB, and HITSZ-VCM datasets show that
178-the proposed ScRL achieves a new state-of-the-art performance.
179-The rest of this paper is organized as follows. Section II introduces related work; Section III elaborates the proposed method; Section IV analyzes the comparative experimental results; and Section
--
216-VIS (IR) images through the GAN network, compensating for the missing modality information. XIV
217-[26] transform IR and VIS images into auxiliary X-modality images respectively and perform X-IRVIS three-mode learning. SMCL [27] generates syncretic modality images that contain information
218-from both modalities to steer modality-invariant feature learning. However, the modality-shared cues
219:(like shape-centered cues) have not been fully explored, which limits the discriminability of features.
220-
221-6
222-
223:2.3. Body shape for Person Re-Identification
224-With the rapid advancement of deep learning, ReID has made significant progress. However,
225-traditional methods heavily rely on color and texture features, leading to performance degradation under clothing changes or modality variations (e.g., visible-infrared transformation). Researchers have
226:explored body shape as a more stable biometric feature to address this issue to enhance ReID robustness. To mitigate the impact of clothing changes, FSAM [7] introduces interactive mutual learning,
227:transferring knowledge from the shape stream to the appearance stream to improve cloth-invariant
228-feature representation. Similarly, GI-ReID [28] adopts a two-stream framework, where an auxiliary
229-Gait-Stream assists the main ReID-Stream in learning gait-based identity features. By incorporating
230:Gait Sequence Prediction (GSP), GI-ReID extracts temporal gait cues from a single image, significantly improving robustness against clothing variations. In gait-based ReID, continuous body shapes
231-are utilized to learn gait features. To this end, GaitPart [29] enhances feature extraction through local
232-temporal modeling, capturing fine-grained motion details from body parts. GaitBase [30] simplifies
233-gait recognition pipelines, improving generalization across diverse scenarios. Furthermore, DeepGaitV2 [31] leverages deep convolutional architectures to enhance gait feature learning, achieving
234:superior performance in challenging environments with occlusions and background clutter. In VIReID, body shape is used to improve cross-modality feature alignment. CMMTL [8] utilizes body
235:shape as a semantic label, jointly training VI-ReID and human semantic segmentation to implicitly
236:learn shape-related features. In contrast, SEFL [9] employs disentanglement learning to discard shape
237-features, focusing on modality-shared appearance representations to enhance generalization across
238-modalities.
239:Compared to their work, our method aims to integrate modality-robust shape and appearance
240:features while leveraging the interaction between shape and appearance to enhance their respective
241-feature representations.
242-
243-3. Proposed Method
--
293-
294-Appearance stream
295-
296:AFE
297-Cross
298-Attention
299-
--
316-
317-value
318-
319:ISR
320-
321-Fs,i
322-
--
333-
334-Wids
335-
336:SFP
337-Fi ir ,2
338-
339-Fi ir ,1
--
373-
374-Loss flow
375-
376:Figure 3: The pipeline of our proposed ScRL framework consists of two branches: the shape stream and the appearance
377:stream. The shape stream includes the shape feature learning network Es and Infrared Shape Restoration (ISR). Es encodes
378:the input shape into shape features F̂s,i , with ISR applied at an intermediate stage to restore erroneous infrared shape features by
379-leveraging appearance features Fiir,1 and Fiir,2 . The appearance stream comprises the appearance feature learning network Ea ,
380:the shape sub-network Es̃ , and Appearance Feature Enhancement (AFE). Ea encodes the pedestrian image into appearance
381:features Fi . To improve inference efficiency, Es̃ encodes shape features F̄s,i from the third block of Ea , guided by Shape
382:Feature Propagation (SFP). Finally, AFE employs a cascaded two-stage cross-attention mechanism, enhancing the interaction
383:between F̄s,i and Fi , which results in shape-centered pedestrian feature representations.
384-
385:represents the body shape corresponding to xivis (xiir ), which is obtained by feeding xivis (xiir ) into the
386-pre-trained Self-Correction Human Parsing (SCHP) [32] model.
387-Overview of ScRL. As shown in Figure.3, we first extract the pedestrian appearance feature map
388:Fi through the image encoder Ea of the appearance stream, while introducing the shape sub-network
389:Es̃ to encode the shape feature F̄s,i from the features outputted by the third block of Ea . In shape
390:stream, to restore inaccurate IR shape features, Infrared Shape Restoration (ISR) employs a crossattention mechanism, applying it to the features outputted by the intermediate block of Ea and Es , to
391:obtain the restorative infrared shape feature map and fed into the subsequent blocks of Es to extract
392:the final IR shape feature. And the VIS shape feature is directly obtained by feeding the visible shape
393:xivis into Es , without involving ISR. For improved computational efficiency during inference, F̄s,i is
394:fed into SFP, transferring the capability of the shape stream to the appearance stream through prototype and instance-level distillation. To enhance appearance features, Appearance Feature Enhancement (AFE) introduces a cascaded two-stage cross-attention mechanism, emphasizing shape-centered
395-appearance features by establishing interactions between F̄s,i and Fi .
396-
397-8
--
453-3.3. Shape Feature Learning
454-Although the design of the dual stream network is aimed at extracting modality-shared features,
455-it focuses on learning pedestrian appearance features and lacks the ability to learn pedestrian body
456:shape features. Compared to appearance features, the body shape features of pedestrians are more
457:robust against modality changes. Therefore, in this section, we aim to learn the body shape features
458-
459-9
460-
461:reshape
462-q
463-
464-ir, j
--
475-transpose
476-(H W )  (H W )
477-
478:reshape
479-k
480-
481-softmax
482-
483:reshape
484-v
485-
486-
--
492-
493-C  H W
494-
495:Figure 4: Illustration of the proposed ISR that is used to obtain missing shape features from appearance features for restoring
496:IR shapes in feature level.
497-
498:by the SCHP and a shape feature learning network Es . In order to obtain discriminative body shape
499:vis generated by the SCHP to E to extract the VIS shape
500:features, we first feed the VIS body shape xs,i
501-s
502:vis = E (xvis ). For IR body shapes, as discussed in the introduction and shown in Figure.1,
503-features fs,i
504-s s,i
505-
506:inherent errors are present within the IR body shapes, which leads to the inability of Es to obtain
507:adequately discriminative shape features from input IR shapes. To address this issue, we propose the
508:Infrared Shape Restoration (ISR) to restore infrared shapes at the feature level.
509:Infrared Shape Restoration: Although IR shapes may contain some inaccuracies, the corresponding original IR images still contain shape-related cues. Additionally, SCHP can effectively
510:parse VIS images to obtain accurate VIS shapes, which are error-free and consistent with the identities of the infrared shapes. Therefore, our motivation is to utilize the reliable VIS shape features
511:to guide the extraction of required shape information from the IR image features and achieve
512:IR shape restoration at the feature level.
513:However, both directly relying on IR shape features for restoration and learning shape features
514:solely from IR images pose challenges, as the former may introduce shape inaccuracies, while the
515:latter can be affected by appearance noise, making it difficult to extract clean and reliable shape
516-information. A naı̈ve approach, such as simple feature fusion, may fail to effectively correct these
517:inaccuracies, as it lacks a mechanism to adaptively search for required shape information from IR
518:images and utilize it to restore IR shapes. To address this, considering that shape inaccuracies are
519:primarily spatial, particularly in local structures such as hands and feet, we introduce a spatial crossattention mechanism that enables the model to search for required shape information from the IR
520:image features. Specifically, let the IR shape feature map and IR feature map output from the j-th layer
521-ir, j
522-of Es and Ea be denoted as Fs,i
523-and Fiir, j , respectively. To ensure more reliable attention allocation,
524-ir, j
525:we construct the query Q by integrating both IR shape feature map Fs,i
526-and IR appearance feature
527-
528-10
529-
530:map Fiir, j . This fusion not only incorporates initially estimated shape information but also captures
531:correct shape cues embedded in the appearance features, providing richer contextual information for
532:shape restoration. Moreover, since both originate from the same IR image, their summation ensures
533:spatial alignment, while any misalignment or anomalies indicate potential shape errors. The query Q
534-is computed as follows:
535-ir, j
536-Q = φq (Fiir, j + Fs,i
--
540-
541-where φq represents the 2D convolution layer with kernel sizes of 1 × 1. The query Q contains richer
542-contextual information, which interacts with K to generate attention and searches for the required
543:body shape information in the value V to restore the IR body shape. The key K and value V are
544-represented as:
545-K = φk (Fiir, j ), V = φv (Fiir, j ),
546-
--
549-where φk and φv represent the 2D convolution layer with kernel sizes of 1 × 1, respectively. Then, we
550-ir, j
551-, this process similar to self-attention, as shown below:
552:obtain the restored IR shape feature map F̂s,i
553-
554-ir, j
555-ir, j
--
562-where φv2 represents the 2D convolution layer with kernel sizes of 1 × 1, BN , N orm and (·)T
563-represents the batch normalization layer, the normalization operation and the transpose operation.
564-Following the above steps, we incorporated the above attention following the first and second blocks
565:of Es . In order to guide the training of Es and ISR, we introduce CE loss and WRT loss to constrain
566:the shape features fˆs,i output from Es with a GeM pooling layer:
567-Lsid = −
568-
569-1 nb
--
601-
602-(8)
603-
604:where Wids represents the shared identity classifier for IR and VIS shape features, j and k represent
605:the index of the positive and negative samples corresponding to the anchor shape feature fˆs,i within
606-a batch, respectively. Pis and Nis represent the positive and negative set corresponding to the anchor
607-
608-11
609-
610:sample xi within a batch, respectively. And dsi, j represents the euclidean distance between two shape
611:features: dsi, j = ∥fˆs,i − fˆs, j ∥2 . Importantly, it should be highlighted that the VIS shape features do
612:not require restoration and we mix them with restored IR shape features in batch to participate in the
613:loss calculation. Therefore, the VIS shape features can guide the learning of the ISR at the loss level.
614:Shape Feature Propagation: we can obtain shape features through the collaboration of appearance feature extraction network Ea , shape feature extraction network Es , and ISR. However, this
615-poses challenges to model deployment due to the increased parameters and computational complexity
616:from SCHP and the shape stream. Therefore, it is crucial to transfer the ability of the shape stream
617-network to the appearance stream network so that the testing phase does not require the participation
618:of the shape stream network. Towards this objective, we replicate the fourth block of Ea as the shape
619-subnetwork Es̃ with GeM pooling layer and apply it to the output features of the third block of Ea
620:to obtain the shape features f˜s,i under the guidance of the shape features fˆs,i output from the Es at
621-instance (Eq.9) and prototype level (Eq.10):
622-
623-Lkd =
--
643-where ||·||2 represents l2 -norm, Θ ∈ RC×K represents the class prototype from the classifier Wids .
644-3.4. Appearance Feature Enhancement
645-To effectively leverage pedestrian appearance features in VI-ReID, it is crucial to extract identitydiscriminative information, such as clothing patterns and facial structures, while suppressing modalityspecific noise (e.g., color in visible images and temperature distributions in infrared images). However, the direct extraction of robust appearance features is highly challenging due to the inherent
646:variations across modalities. Notably, many identity-related appearance features exhibit strong correlations with body shape—for example, appearance features associated with head shape primarily
647:include hair and facial attributes, whereas background elements are clearly unrelated to shape. This
648-underlying relationship is typically reflected spatially. Therefore, to effectively capture this spatial
649:dependency, AFE adopts spatial attention for more precise alignment between shape and appearance
650-representations.
651:Given this strong correlation, we propose to utilize body shape to enhance appearance features.
652-12
653-
654-Fs,i
655-C  H W
656-
657:reshape
658- q1
659:reshape
660- k1
661-
662-Fi
--
669-
670-
671-
672:reshape
673- q2
674-
675-transpose
676-
677-Fi
678-
679:reshape
680-C  H W
681-
682-(H W )  (H W )
683-
684:reshape
685- v1
686-
687:reshape
688- k2
689-
690- v2
--
714-Laid
715-Lawrt
716-
717:Figure 5: Illustration of the proposed AFE, which can mine shape-centered appearance features guided by shape features.
718-
719-Specifically, we aim to mine modality-shared appearance features while filtering out identity-irrelevant
720-modality-specific features. Since only a subset of appearance features is directly associated with body
721:shape, we adopt a progressive extraction strategy: first identifying directly shape-related appearance
722:features, then leveraging them to discover indirectly shape-related appearance features that further
723:enrich identity representation. To achieve this, we introduce a cascading two-stage attention mechanism, systematically refining the appearance feature. The first stage utilizes shape features as a query
724:to extract appearance features with strong shape dependencies. The second stage then employs these
725:refined features along with shape features to further extract appearance features that are both directly
726:and indirectly linked to shape, enhancing the appearance representation.
727:In the first stage, let the shape feature map F̄s,i output by Es̃ serve as query Q, and the appearance
728-feature map Fi extracted by Ea serve as key K and value V :
729-Q = ψq1 (F̄s,i ), K = ψk1 (Fi ), V = ψv1 (Fi ),
730-
731-(11)
732-
733-where ψq1 , ψk1 , and ψv1 represents the 2D convolution layer with kernel sizes of 1 × 1. Then, similar
734:to ISR, the correlation score between Q and K is used to search features directly related to shape in
735-
736-13
737-
738:V , and fused with shape feature map F̄s,i to obtain F̃i f use .
739-1
740-F̃i f use = ψv2
741-(BN (N orm(QK T )V )) + F̄s,i ,
--
746-where ψv2
747-i
748-
749:shape feature and appearance feature directly related to body shape, F̃i f use can be effectively employed
750-as a query during the second stage of attention, which facilitates the acquisition of appearance features
751:that are both directly and indirectly associated with the body shape. As the query feature of the
752:second-stage attention, F̃i f use plays a pivotal role in determining whether discriminative modalityshared features, both directly and indirectly associated with body shape, can be effectively extracted
753-from the appearance features Fi . To ensure the discriminability of the query F̃i f use , we also employ
754-CE loss and WRT loss to jointly constrain f˜if use = GeM(F̃i f use ):
755-Lqid = −
--
794-i
795-
796-where Widq represents the shared identity classifier for infrared and visible pedestrian query features,
797:j and k represent the index of the positive and negative samples corresponding to the anchor shape
798-feature f˜if use within a batch, respectively. Piq and Niq represent the positive and negative set corresponding to the anchor f˜if use within a batch, respectively. And dqi, j represents the euclidean distance
799-between two query features: dqi, j = ∥f˜if use − f˜jf use ∥2 .
800-In the second stage, we employed the output feature F̃i f use of the first stage attention as a query to
801:emphasize the appearance features directly and indirectly related to the body shape in Fi as follows:
802-2
803-F̃i = ψv2
804-(BN (N orm(QK T )V )) + Fi ,
--
808-2 represents the 2D conwhere Q = ψq2 (F̃i f use ),K = ψk2 (Fi ),V = ψv2 (Fi ), and ψq2 , ψk2 , ψv2 and ψv2
809-
810-volution layer with kernel sizes of 1 × 1. We use the enhanced appearance features f˜i = GeM(F̃i ),
811:which are closely related to the shape, as the final appearance features. Therefore, the loss functions
812-
813-14
814-
815-Algorithm 1 Training process of the proposed ScRL
816-vis (xir ).
817:Require: A mini-batch that consists of nb VIS(IR) images xivis (xiir ) and VIS(IR) shapes xs,i
818-s,i
819-Ensure: Trained ScRL model.
820-1: for i = 1; i < iteration; i + + do
821-2:
822-Extract Fi ,Fiir, j by inputting xivis , xiir into the appearance stream network Fa .
823-ir, j
824:ir into the shape stream network E .
825-3:
826-Extract Fi,s
827-by inputting xi,s
--
831-ir, j
832-ir, j
833-Send Fiir, j ,Fi,s
834:to the ISR module to obtain the restored infrared shape feature F̂s,i
835-.
836-
837-ir, j
838-Send F̂s,i
839:to the subsequent network of the shape stream Es to obtain the final shape features
840-ir into E ).
841:F̂s,i (contain VIS shape features obtained by directly inputting xi,s
842-s
843-6:
844:Extract F̃s,i by feeding the output features of the third block of Ea into the shape subnetwork
845-Es̃ .
846-7:
847:Transfer the capability to extract shape features from the shape stream network Es̃ to the appearance stream network Ea , Es̃ by the SFP module, and obtain the shape features F̄s,i
848-8:
849:Extract appearance features F̃i f use directly related to body shape by inputting F̄s,i , Fi into stage
850:1 of the AFE module.
851-9:
852:Extract appearance features F̃i both directly and indirectly related to body shape by inputting
853:F̃i f use , Fi into stage 2 of the AFE module.
854-10:
855:Optimize Ea , Es , Es̃ , ISR module and AFE module through loss functions Lsid (Eq.7),
856-Lswrt (Eq.8), Lkd (Eq.9), Lkd ce (Eq.10), Lqid (Eq.13), Lqwrt (Eq.14), Laid (Eq.16) and Lawrt (Eq.17).
857-11: end for
858-
--
900-(17)
901-
902-where j and k represent the index of the positive and negative samples corresponding to the anchor
903:shape feature f˜i within a batch size, respectively. Pia and Nia represent the positive and negative set
904-corresponding to the anchor f˜i within a batch, respectively. And dai, j represents the euclidean distance
905-between two features: dai, j = ∥f˜i − f˜j ∥2 .
906-3.5. Training and Inference
907-In the training process, we employ the appearance stream network Ea for extracting appearance
908:features and the shape stream network Es for extracting shape features. To further enhance this
909-
910-15
911-
912:process, we introduce the ISR to restore inaccuracies in IR shape features. Additionally, the SFP is
913:integrated to transfer the shape feature extraction capabilities of the shape stream to the appearance
914:stream. Finally, the AFE is introduced to mine appearance features that have both direct and indirect
915:associations with the body shape. The complete training process is executed in an end-to-end fashion,
916-as illustrated in Algorithm 1. As shown above, the total objective function of this method can be
917-formalized as:
918-Ltotal = La + Ls + Lkd + Lkd−ce + Lq .
919-
920-(18)
921-
922:where La = Laid + Lawrt , Ls = Lsid + Lswrt , Lq = Lqid + Lqwrt . During the testing process, we concatenate the shape features output by Es̃ with the appearance features output by Appearance Feature
923:Enhancement (AFE) for inference. Therefore, we solely utilize the appearance stream network Ea ,
924:Es̃ , and the AFE, without involving the shape stream network Es and the human parsing network
925-SCHP.
926-
927-4. Experiments
--
1130-4.2. Implementation Details
1131-Similar to DEEN [24], we adopted ResNet50 pre-trained on ImageNet as the backbone and replaced the average pooling layer with the GeM pooling layer, with all input image sizes resized to
1132-384 × 144. In the training phase, we adopted Random Crop, Random Horizontal Flip, Channel Random Erasing and Channel AdapGray [34] to enhance the IR(VIS) images, and we adopted Random
1133:Crop and Random Horizontal Flip for the shape image. We adopt the Adam optimizer for optimization, the learning rate of the classifier Ws , Wa was set to 0.0007, and the learning rate of other
1134-networks was set to 0.00035. The model trained a total of 120 epochs, in the first 10 epochs, the
1135-learning rate is dynamically adjusted through the warmup strategy, in the 40th and 60th epochs, the
1136-learning rate decreases by 10%. At every batch size, we randomly sample 64 images from 8 identities,
--
1150-indicators, respectively. Similarly, in the “indoor search” mode, the proposed method delivered an
1151-accuracy of 82.4%, 85.4%, and 82.2% for the Rank-1, mAP, and mINP indicators, respectively. Furthermore, our method outperforms the suboptimal method by 0.9%(2.1%), 0.8%( 2.1%), and 7.9%
1152-(9.5%) on Rank-1, mAP, and mINP in “all search”(“indoor search”) mode, respectively. The results
1153:suggest that the proposed ScRL, which focuses on pedestrian features with shape as the central component, effectively mitigates modality changes and enhances the accuracy of cross-modality pedestrian
1154-matching.
1155-RegDB. As illustrated in Tab. 3, the proposed approach has showcased commendable performance even on the limited-scale dataset RegDB. Notably, within the “IR to VIS” mode, our method
1156-achieved 91.8% and 85.3% accuracy in Rank-1 and mAP indicators, respectively, and has demonstrated superiority over the suboptimal SEFL approach by 0.7% in terms of Rank-1. Similarly, in
--
1358-the suboptimal CST approach by 0.7% in terms of Rank-1 in the “VIS to IR” mode. Furthermore,
1359-when compared to the image-based approach, our method has achieved remarkable results in the “IR
1360-to VIS” (“VIS to IR”) mode, surpassing the suboptimal SEFL approach by 3.5% (3.1%) in Rank-1
1361:and 0.6% (0.5%) in mAP. The experimental results suggest that our method can extract shape-centered
1362-features that remain insensitive to modality variations at the frame level, ensuring robustness in pedestrian feature representation at the sequence level. As a result, it achieves the best performance.
1363-The comparative experimental results validate the effectiveness and superiority of our method for
1364-both image-based and video-based VI-ReID tasks. Specifically, our approach outperforms SEFL, the
1365-
1366-19
1367-
1368:Table 5: Ablation studies of the proposed ScRL.
1369-
1370-Setting
1371-Baseline
1372:+SFP
1373:+ISR
1374:+AFE
1375-
1376-Component
1377:SFP ISR AFE
1378-✓
1379-✓
1380-✓
--
1418-
1419-second-best method, across three datasets: the large-scale SYSU-MM01 and the small-scale RegDB
1420-for image-based VI-ReID, as well as the large-scale MITML video dataset for video-based VI-ReID.
1421:This performance gain is attributed to our explicit integration of body shape features alongside appearance features, whereas SEFL primarily relies on appearance cues alone. These findings underscore
1422:the importance of learning shape-centered features, which provide substantial benefits for VI-ReID.
1423:4.4. Ablation Studies
1424:In this subsection, we analyze the contribution of different components, including SFP (Shape
1425:Feature Propagation), ISR (Infrared Shape Restoration), and AFE (Appearance Feature Enhancement), based on the SYSU-MM01 dataset.
1426:Effectiveness of SFP. To verify the effectiveness of SFP, we integrated the SFP into the Baseline,
1427:resulting in “Baseline+SFP”. This augmentation enables the model to autonomously acquire shape
1428:features from the original image during inference, eliminating the need for shape feature learning and
1429-human parsing networks. Specifically, as shown in Tab. 5, in “All search”(“Indoor search”) mode,
1430:“Baseline+SFP” has demonstrated improvements of 3.4%(1.4%) in Rank-1, 3.1%(1.7%) in mAP, and
1431-3.6%(2.0%) in mINP when contrasted with the Baseline. These improvements are attributed to the fact
1432:that shape features, which remain unaffected by modality changes, complement appearance features,
1433-thereby enhancing the overall performance.
1434:Effectiveness of ISR. Incorporating SFP extends the shape stream’s capabilities to the appearance stream. However, the incorrect IR shape restricts the potential of the shape stream, subsequently curbing the efficacy of the appearance stream. To tackle this challenge, ISR was introduced
1435:to the “Baseline+SFP”, yielding “Baseline+SFP+ISR”. As highlighted in Tab. 5, the Rank-1, mAP,
1436-and mINP metrics have collectively improved by 1.0%(0.4%), 1.0%(0.4%), and 1.2%(0.4%) in “All
1437-search”(“Indoor search”) mode respectively. The performance improvement suggests the feasibility
1438:of extracting relevant information for storing IR shape features from the intermediate features of the
1439:appearance stream network. It also highlights ISR’s capability to store inaccurate IR shapes.
1440-
1441-20
1442-
1443:Table 6: Ablation study on the design of the proposed ISR. This table presents a comparison of different design choices for the
1444:ISR. Setting 1: direct learning from infrared features, Setting 2: adds the infrared shape feature map and appearance feature
1445:map, Setting 3: only shape feature map as query, Setting 4: replaces spatial attention with channel attention, and Setting 5:
1446:spatial Attention-based ISR, our proposed method.
1447-
1448-Setting
1449-Setting 1
--
1489-98.7 99.6 84.7 81.5
1490-98.8 99.8 85.4 82.2
1491-
1492:To further evaluate the effectiveness of ISR, we conducted a detailed analysis with five different settings, as shown in Tab. 6. Taking the All search mode as an example, in Setting 1, only the
1493:IR appearance feature map was fed into the shape stream, assuming it could implicitly learn the IR
1494:shape feature. However, results indicate that relying solely on IR appearance features fails to extract
1495:meaningful shape cues, leading to suboptimal performance (Rank-1: 74.6%, mAP: 71.6%). Setting 2
1496-improved performance (Rank-1: 75.4%, mAP: 71.9%) by directly adding the appearance feature map
1497:to the IR shape feature map, and then inputting it into the shape stream network. However, without an
1498:adaptive feature selection mechanism, it struggled to identify the required shape features for restoring
1499:IR shape. To address this, we introduced a cross-attention mechanism that adaptively extracts required
1500:shape information from IR images and utilizes it for IR shape restoration at the feature level. In query
1501-design, we compared two strategies: Setting 3 (Rank-1: 74.9%, mAP: 71.8%), where only the IR
1502:shape feature map was used as the query, and Setting 5, where both the IR shape and appearance
1503-feature maps were combined. The superior performance of Setting 5 (Rank-1: 76.1%, mAP: 72.6%)
1504:suggests that integrating both shape and appearance features provides better guidance for extracting
1505:the required shape features from IR appearance features and effectively restoring IR shape representations. Additionally, in Setting 4, replacing spatial attention with channel attention resulted in the
1506-performance similar to Setting 2 (Rank-1: 75.5%, mAP: 72.2%), indicating that channel attention is
1507-insufficient. Finally, Setting 5, which employs spatial cross-attention, achieved the best performance
1508:across all evaluation metrics, confirming that ISR effectively restores IR shapes at the feature level.
1509:Effectiveness of AFE. In order to emphasize appearance features related to shape, we incorporated the AFE into the “Baseline+SFP+ISR”. The corresponding results, as depicted in Tab. 5, reflect
1510-an enhancement in Rank-1, mAP, and mINP, in “All search” mode, these metrics increased from
1511-74.6%, 71.0%, and 57.7% to 76.1%, 72.6%, and 59.8%, respectively. In “Indoor search” mode, the
1512-
1513-21
1514-
1515:Table 7: Ablation study on the design of the proposed AFE. This table presents a comparison of different design choices for
1516:the AFE. Setting 1: concatenates shape and appearance features, Setting 2: replaces spatial attention with channel attention,
1517:Setting 3: excludes the query loss, and Setting 4: our proposed AFE.
1518-
1519-Setting
1520-Setting 1
--
1560-
1561-metrics improved from 79.4%, 83.2%, and 79.8% to 82.4%, 85.4%, and 82.2% in Rank-1, mAP, and
1562-mINP, respectively. This suggests that in contrast to typical appearance features, appearance features
1563:centered on shape are better at emphasizing identity-discriminative features related to individuals
1564-while minimizing the influence of modality, background, and other irrelevant noise features.
1565:To further investigate the role of AFE and validate its design choices, we conduct an in-depth
1566:analysis, as shown in Tab. 7, comparing different configurations and their impact on model performance. Taking the All search mode as an example, in Setting 1, shape and appearance features are
1567-directly concatenated as the final representation (Rank-1: 74.6%, mAP: 71.0%), without leveraging
1568:shape information to enhance appearance features. Due to the presence of modality-specific information and background noise in the appearance features, overall performance is limited. In Setting 2,
1569-spatial attention is replaced with channel attention (Rank-1: 74.9%, mAP: 71.8%). While this leads to
1570-a slight improvement in mAP and mINP compared to Setting 1, the gains are minimal, indicating that
1571:channel attention is insufficient for effectively learning shape-related appearance features. In Setting
1572:3, we remove the query loss term Lqwrt of AFE to evaluate its impact (Rank-1: 74.9%, mAP: 71.4%).
1573-The results show a decline in Rank-1 and mAP, primarily due to the reduced discriminability of query
1574-features in the second-stage attention. This, in turn, weakens the model’s ability to capture both direct
1575:and indirect shape-related features, ultimately degrading recognition performance. Finally, in Setting 4 (AFE), both spatial attention and query loss are incorporated, achieving the best performance
1576-(Rank-1: 76.1%, mAP: 72.6%) across all evaluation metrics, and demonstrating its effectiveness in
1577-enhancing identity-discriminative features.
1578-To further explore the design of cascaded multi-stage attention, we conducted additional experiments with four different settings, as shown in Tab. 8. Setting 1 represents the configuration without
1579:AFE, corresponding to Baseline+SFP+ISR. Building upon this, Setting 2 (Baseline+SFP+ISR+S1)
1580-incorporates a single-stage attention. The results show improvements in Rank-1, mAP, and mINP.
1581-Specifically, compared to Setting 1, all search and indoor search modes exhibit gains of 0.7% (1.0%)
1582-22
1583-
1584:Table 8: Ablation studies of the proposed AFE. S1: first-stage attention mechanism. S2: second-stage attention mechanism.
1585-S3: third-stage attention mechanism.
1586-
1587-Setting
--
1637-98.8 99.8 85.4 82.2
1638-98.6 99.8 84.1 80.8
1639-
1640:Table 9: Ablation Study on the Effectiveness of Shape and Appearance Features in ScRL. Where “App” represents appearance
1641:features, while “Both” indicates the use of both appearance and shape features. The performance of CSDN is sourced from its
1642-original paper.
1643-
1644-Settings
1645-B(App)
1646:B + AFE(w/o ISR)
1647:B + AFE(w ISR)
1648-CSDN[48]
1649:B(shape)
1650:B(shape) + ISR
1651-ScRL
1652-CSDN+ScRL
1653-
--
1711-98.8 99.4 87.2 84.2
1712-
1713-in Rank-1, 0.9% (0.8%) in mAP, and 1.2% (0.9%) in mINP, respectively. Further incorporating
1714:second-stage attention leads to Setting 3 (Baseline+SFP+AFE), forming a two-stage attention approach, which is our proposed AFE. In all search, Rank-1, mAP, and mINP improved from 75.3%,
1715-71.9%, and 58.9% to 76.1%, 72.6%, and 59.8%, respectively, while in indoor search, they increase
1716-from 80.4%, 84.0%, and 80.7% to 82.4%, 85.4%, and 82.2%. However, adding a third attention stage
1717:in Setting 4 (three-stage attention) results in a slight performance drop, indicating that two-stage attention is sufficient for learning both direct and indirect shape-related appearance features. Adding
1718-additional stages may introduce redundancy, leading to diminished performance. These results confirm that a two-stage attention mechanism is the optimal design choice.
1719:Complementarity of Shape and Appearance Features. To evaluate the complementarity between shape and appearance features, we conducted experiments using only appearance features and
1720:only shape features, as presented in Tab. 9. First, comparing the appearance-based baseline “B(App)”
1721:(using only appearance features) and the shape-based baseline “B(shape)” (using only shape features),
1722:it is evident that appearance features outperform shape features overall, achieving higher Rank-1 and
1723:mAP. This is because appearance features contain finer-grained identity cues, such as clothing details and local textures, which contribute to stronger identity discrimination. Next, incorporating ISR
1724:into the shape stream “B(shape) + ISR” significantly improves performance. This indicates that ISR
1725-23
1726-
1727:Table 10: Different shape feature extractors are used for the shape stream. “R1”,“R10”, and “R20” denote Rank-1, Rank-10,
1728-and Rank-20, respectively.
1729-
1730-Methods
--
1786-98.8 99.8 85.4
1787-82.2
1788-
1789:effectively restores inaccurate infrared shape features at the feature level, thereby enhancing shape
1790:feature representation. Additionally, comparing “B + AFE (w/ ISR)” and “B + AFE (w/o ISR)” further validates the role of ISR in the AFE module, indicating that ISR enhances the representation
1791:of appearance features by improving the quality of infrared shape features. When ScRL integrates
1792:both shape and appearance features, its performance improves significantly. In the all search mode,
1793-Rank-1/mAP reaches 76.1%/72.6%, while in the indoor search mode, the performance increases to
1794:82.4%/85.3%, further confirming the complementary nature of shape and appearance features. To
1795-further explore the potential of ScRL, we applied it to a stronger appearance-based baseline, CSDN,
1796-which is driven by the Contrastive Language-Image Pretraining (CLIP). The results show that in the all
1797-search mode, Rank-1/mAP improves from 75.2%/71.8% to 79.1%/75.2%, while in the indoor search
1798-mode, the performance rises from 82.0%/85.0% to 85.3%/87.2%. These results indicate that our proposed method can further enhance recognition performance even on a strong baseline, validating that
1799:combining shape and appearance features leads to a more robust and discriminative representation for
1800-VI-ReID.
1801:Selection of Shape Feature Extractors. We replaced the shape feature extractor with deepgaitv2
1802-[31], gaitpart [29], and gaitbase [30] to evaluate their impact on overall performance. However, since
1803-these networks have different architectures compared to our appearance stream network (ResNet50),
1804-their integration led to suboptimal interactions and guidance, negatively affecting performance. As
1805:shown in Tab. 10, compared to ScRL with ResNet50 as the shape feature extractor, all three shape
1806-feature extractors result in varying degrees of performance degradation. On the other hand, using
1807:ResNet50 as the shape feature extractor, which aligns structurally with the appearance stream, resulted
1808-in significant performance improvements. Specifically, in all search settings, ResNet50 achieved
1809:Rank-1 of 76.1% and mAP of 72.6%, outperforming other shape baselines. Similarly, in the indoor
1810-search setting, ResNet50 attained Rank-1 of 82.4% and mAP of 85.4%, demonstrating its superiority
1811-
1812-24
1813-
1814-Table 11: Comparison of performance and computational costs on different settings in the training stage. Setting 1: Baseline.
1815:Setting 2: Two independent networks extract shape and appearance features respectively, and concatenate the two features for
1816-inference. Setting 3: The ScRL. HP: human parsing network.
1817-
1818-Settings
--
1830-Training
1831-times/hours
1832-mAP mINP
1833:ISR
1834:AFE
1835-Es
1836-Ea
1837-Total
--
1850-12.8
1851-
1852-Table 12: Comparison of performance and computational costs on different settings in the inference stage. Setting 1: Baseline.
1853:Setting 2: Two independent networks extract shape and appearance features respectively, and concatenate the two features for
1854:inference. Setting 3: Baseline + SFP. Setting 4: the ScRL. HP: human parsing network.
1855-
1856-Settings
1857-Setting 1
--
1902-38.48(10.15)
1903-38.48(10.15)
1904-
1905:in learning shape features effectively. Based on these results, we chose ResNet50 as the shape feature
1906-extraction network to ensure better compatibility and information sharing between the two streams.
1907-4.5. Computing Complexity
1908-This section analyzes the computational complexity of the proposed ScRL during both the training
1909-and inference stages, as shown in Tab. 11 and Tab. 12.
1910-During the training phase, Tab. 11 shows that Setting 2, which employs two independent networks
1911:to extract shape and appearance features separately, significantly increases computational complexity
1912-compared to Baseline (Setting 1). The total number of parameters rises from 23.51M to 47.02M, and
1913-FLOPs increase from 6.91G to 13.82G, while training time extends from 4.37 hours to 7.47 hours. In
1914-contrast, our proposed ScRL (Setting 3) further improves performance, achieving a Rank-1 accuracy
1915:of 76.1% and an mAP of 72.6% while introducing ISR and AFE modules. This increases the total
1916-number of parameters to 62.02M and FLOPs to 17.06G, with a training time of 12.8 hours. Although
1917-the computational complexity increases, the significant improvement in retrieval performance makes
1918-the additional training overhead acceptable.
1919-Efficient inference is a key requirement for real-world deployment. As shown in Tab. 12, Setting
1920:2 requires additional human parsing (HP) and shape feature extraction, leading to a sharp increase
1921-in model complexity, with the total number of parameters reaching 113.64M and FLOPs soaring to
1922:90.38G. In contrast, Setting 3 (Baseline+SFP) eliminates the need for an additional shape stream and
1923-25
1924-
1925-Rank-6
--
1946-
1947-d
1948-
1949:Figure 6: Retrieval results under different ablation settings on SYSU-MM01. a, b, c, and d denote “Baseline”, “Baseline+SFP”,
1950:“Baseline+SFP+ISR”, and “Baseline+SFP+ISR+AFE”, respectively. The green (red) border represents the same (different)
1951-identity. It is evident that the incorporation of various modules has led to a notable enhancement in the hit rate for pedestrian
1952-retrieval.
1953-
1954-human parsing networks, significantly reducing computational overhead. Building upon this, Setting
1955:4 (ScRL) introduces AFE, which imposes a negligible computational cost while further enhancing
1956-performance. ScRL incurs only a slight increase in computational complexity compared to the baseline, with the final model size reaching 38.48M parameters and 10.15G FLOPs. This demonstrates
1957:that by integrating shape features into the appearance stream, our method preserves shape-related features during inference without imposing significant computational overhead, making it more practical
1958-for real-world deployment.
1959:4.6. Visualization
1960:This section encompasses visualization experiments on retrieval results and feature similarity distribution.
1961:Retrieval result. In order to further analyze the effectiveness of different ablation settings, as
1962:shown in Figure. 6, we visualized the retrieval results of different ablation settings. Moving left to
1963-right, the similarity score with the query image progressively decreases. Compared to the Baseline, it
1964-is clear that the addition of each module yields improvements in retrieval performance.
1965:Heatmap interest. The proposed ScRL method is primarily dedicated to learning shape-centered,
1966-modality-invariant pedestrian feature representations. To illustrate the advantages of our method,
1967:we have visualized the heatmap, as shown in Figure. 7. Compared to the Baseline method, the
1968-proposed ScRL effectively highlights more discriminative information about the human body area
1969-26
1970-
1971:Figure 7: Visualization of regions that the model interests. The first row depicts the original input pedestrian images, with four
1972-pedestrians, each having both an infrared and a visible image. The second row illustrates the region where the Baseline method
1973-captures features, while the third row showcases the region where the proposed ScRL method captures features. It’s evident
1974-that the ScRL method outperforms the Baseline significantly in capturing these features.
--
1980-
1981-5. Conclusion and Future Works
1982-This paper proposes the innovative ScRL framework, designed for cross-day and night person retrieval, significantly enhancing the robustness of surveillance systems in real-world applications. By
1983:explicitly integrating body shape and appearance features, ScRL effectively mitigates the challenges
1984-posed by modality variations in VI-ReID, ensuring reliable person retrieval under diverse lighting
1985:conditions. Specifically, we introduce Shape Feature Propagation (SFP), which directly extracts shape
1986:features from visible and infrared pedestrian images, eliminating the need for auxiliary models and reducing computational complexity. Infrared Shape Restoration (ISR) corrects errors in infrared shape
1987:representations at the feature level, improving the discriminability of infrared shape features. Furthermore, Appearance Feature Enhancement (AFE) learns shape-centered appearance features while
1988-suppressing identity-irrelevant noise, thereby strengthening identity representation. Extensive experiments on the SYSU-MM01 and RegDB image datasets, as well as the HITSZ-VCM video dataset,
1989-validate the superiority of ScRL. Beyond its technical contributions, ScRL offers practical benefits
1990-for security personnel, law enforcement agencies, and intelligent surveillance systems by improving
--
1995-surveillance environments.
1996-Despite its effectiveness, ScRL has limitations when handling cases where individuals undergo
1997-drastic clothing changes, such as long-term cross-day and night scenarios. Large clothing variations
1998:may obscure or distort body shape features, reducing their reliability for person retrieval. Future research will explore more advanced body modeling techniques, such as 3D shape modeling, to enhance
1999-robustness under extreme clothing variations and investigate the integration of ScRL into broader VIReID scenarios, such as the thermal infrared modality, ensuring robust performance across various
2000-real-world conditions.
2001-
--
2024-[6] Z. Cui, J. Zhou, Y. Peng, S. Zhang, Y. Wang, Dcr-reid: Deep component reconstruction for
2025-cloth-changing person re-identification, IEEE Transactions on Circuits and Systems for Video
2026-Technology.
2027:[7] P. Hong, T. Wu, A. Wu, X. Han, W.-S. Zheng, Fine-grained shape-appearance mutual learning
2028-for cloth-changing person re-identification, in: Proceedings of the IEEE/CVF conference on
2029-computer vision and pattern recognition, 2021, pp. 10513–10522.
2030-[8] N. Huang, K. Liu, Y. Liu, Q. Zhang, J. Han, Cross-modality person re-identification via multitask learning, Pattern Recognition 128 (2022) 108653.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 19 -l 23 'Shape-centered representation learning for visible-infrared person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Table 3: Results of mAP and CMC (%) obtained by our proposed method and the state-of-the-art Re-ID methods on RegDB.
“R1”, “R10” and “R20” denote Rank-1,Rank-10 and Rank-20, respectively. These results are copied from their papers. “–”
denotes that no reported result is available. The underline indicates the second-best performance, while bold values represent
the best performance.

Methods

Reference

Zero-Pad[3]
HCML[39]
HSME[20]
D2 RL[25]
XModal[26]
DDAG[40]
MPANet[23]
CAJ[34]
AGW[38]
CMMTL[8]
PMT[41]
MRCN[22]
DEEN[24]
SEFL[9]
CSMSS[42]
CSC-Net[43]
ScRL(our)

ICCV’2017
AAAI’2018
AAAI’2019
CVPR’2019
AAAI’2020
ECCV’2020
CVPR’2021
ICCV’2021
TPAMI’2022
PR’2022
AAAI’2022
AAAI’2023
CVPR’2023
CVPR’2023
TMM’2024
TCSVT’2024

R1
17.8
24.4
50.9
43.4
62.2
69.3
83.7
85.0
70.1
89.9
84.8
91.4
91.1
92.2
85.3
91.0
92.4

Visible to Infrared
R10 R20 mAP
34.2 44.4 18.9
47.5 56.8 20.1
73.4 81.7 47.0
66.1 76.3 44.1
83.1 91.7 60.2
86.2 91.5 63.5
80.9
95.5 97.5 79.1
86.2 91.6 66.4
96.6 98.3 85.6
76.6
98.0 99.0 84.6
97.8 98.9 85.1
86.6
76.4
86.4
98.1 99.1 86.7

mINP
49.2
65.3
50.2
73.6

R1
16.6
21.7
50.2
68.1
82.8
84.8
70.5
88.3
84.2
88.3
89.5
91.1
83.9
89.4
91.8

Infrared to Visible
R10 R20 mAP
34.7 44.3 17.8
45.0 55.6 22.2
72.4 81.1 46.2
85.2 90.3 61.8
80.7
95.3 97.5 77.8
87.2 91.8 65.9
96.2 98.0 84.1
75.1
96.7 98.5 81.9
96.8 98.4 83.4
85.2
75.2
85.7
98.0 99.1 85.3

mINP
48.6
61.6
51.2
70.9

Table 4: Results of mAP and CMC (%) obtained by our proposed method and the state-of-the-art Re-ID methods on HITSZVCM.“R1”, “R5” and “R10” denote Rank-1,Rank-5 and Rank-10, respectively. These results are copied from their papers.

Strategy
Video

Image

Methods

Reference

MITML[37]
IBAN[44]
CST[45]
Lba[46]
MPANet[23]
VSD[47]
CAJ[34]
SEFL[9]
ScRL(our)

CVPR’22
TCSVT’23
TMM’24
ICCV’21
CVPR’21
CVPR’21
ICCV’21
CVPR’23

Infrared to Visible
R1
R5
R10 mAP
63.7 76.9 81.7 45.3
65.0 78.3 83.0 48.8
69.4 81.1 85.6 51.2
46.4 65.3 72.2 30.7
46.5 63.1 70.5 35.3
54.5 70.0 76.3 41.2
56.6 73.5 79.5 41.5
67.7 80.3 84.7 52.3
71.2 83.2 87.4 52.9

Visible to Infrared
R1
R5
R10 mAP
64.5 79.0 83.0 47.7
69.6 81.5 85.4 51.0
72.6 83.4 86.7 53.0
49.3 69.3 75.9 32.4
50.3 67.3 73.6 37.8
57.5 73.7 79.4 43.5
60.1 74.6 79.9 42.8
70.2 82.2 86.1 52.5
73.3 84.4 87.7 53.0

the suboptimal CST approach by 0.7% in terms of Rank-1 in the “VIS to IR” mode. Furthermore,
when compared to the image-based approach, our method has achieved remarkable results in the “IR
to VIS” (“VIS to IR”) mode, surpassing the suboptimal SEFL approach by 3.5% (3.1%) in Rank-1
and 0.6% (0.5%) in mAP. The experimental results suggest that our method can extract shape-centered
features that remain insensitive to modality variations at the frame level, ensuring robustness in pedestrian feature representation at the sequence level. As a result, it achieves the best performance.
The comparative experimental results validate the effectiveness and superiority of our method for
both image-based and video-based VI-ReID tasks. Specifically, our approach outperforms SEFL, the

19

Table 5: Ablation studies of the proposed ScRL.

Setting
Baseline
+SFP
+ISR
+AFE

Component
SFP ISR AFE
✓
✓
✓

✓
✓

✓

R1
70.2
73.6
74.6
76.1

R10
95.7
97.1
97.4
97.6

All search
R20 mAP mINP
98.6 66.9 52.9
99.2 70.0 56.5
99.3 71.0 57.7
99.4 72.6 59.8

R1
77.6
79.0
79.4
82.4

Indoor search
R10 R20 mAP mINP
97.8 99.5 81.1 77.4
98.2 99.4 82.8 79.4
98.4 99.3 83.2 79.8
98.8 99.8 85.4 82.2

second-best method, across three datasets: the large-scale SYSU-MM01 and the small-scale RegDB
for image-based VI-ReID, as well as the large-scale MITML video dataset for video-based VI-ReID.
This performance gain is attributed to our explicit integration of body shape features alongside appearance features, whereas SEFL primarily relies on appearance cues alone. These findings underscore
the importance of learning shape-centered features, which provide substantial benefits for VI-ReID.
4.4. Ablation Studies
In this subsection, we analyze the contribution of different components, including SFP (Shape
Feature Propagation), ISR (Infrared Shape Restoration), and AFE (Appearance Feature Enhancement), based on the SYSU-MM01 dataset.
Effectiveness of SFP. To verify the effectiveness of SFP, we integrated the SFP into the Baseline,
resulting in “Baseline+SFP”. This augmentation enables the model to autonomously acquire shape
features from the original image during inference, eliminating the need for shape feature learning and
human parsing networks. Specifically, as shown in Tab. 5, in “All search”(“Indoor search”) mode,
“Baseline+SFP” has demonstrated improvements of 3.4%(1.4%) in Rank-1, 3.1%(1.7%) in mAP, and
3.6%(2.0%) in mINP when contrasted with the Baseline. These improvements are attributed to the fact
that shape features, which remain unaffected by modality changes, complement appearance features,
thereby enhancing the overall performance.
Effectiveness of ISR. Incorporating SFP extends the shape stream’s capabilities to the appearance stream. However, the incorrect IR shape restricts the potential of the shape stream, subsequently curbing the efficacy of the appearance stream. To tackle this challenge, ISR was introduced
to the “Baseline+SFP”, yielding “Baseline+SFP+ISR”. As highlighted in Tab. 5, the Rank-1, mAP,
and mINP metrics have collectively improved by 1.0%(0.4%), 1.0%(0.4%), and 1.2%(0.4%) in “All
search”(“Indoor search”) mode respectively. The performance improvement suggests the feasibility
of extracting relevant information for storing IR shape features from the intermediate features of the
appearance stream network. It also highlights ISR’s capability to store inaccurate IR shapes.

20

Table 6: Ablation study on the design of the proposed ISR. This table presents a comparison of different design choices for the
ISR. Setting 1: direct learning from infrared features, Setting 2: adds the infrared shape feature map and appearance feature
map, Setting 3: only shape feature map as query, Setting 4: replaces spatial attention with channel attention, and Setting 5:
spatial Attention-based ISR, our proposed method.

Setting
Setting 1
Setting 2
Setting 3
Setting 4
Setting 5

R1
74.6
75.4
74.9
75.5
76.1

R10
97.8
97.5
97.6
98.1
97.6

All search
R20 mAP mINP
99.5 71.6 58.6
99.4 71.9 58.8
99.4 71.8 58.9
99.5 72.2 59.3
99.4 72.6 59.8

R1
80.9
81.2
81.2
81.3
82.4

Indoor search
R10 R20 mAP mINP
98.4 99.4 84.1 80.9
98.6 99.4 84.4 81.2
98.5 99.5 84.5 81.4
98.7 99.6 84.7 81.5
98.8 99.8 85.4 82.2

To further evaluate the effectiveness of ISR, we conducted a detailed analysis with five different settings, as shown in Tab. 6. Taking the All search mode as an example, in Setting 1, only the
IR appearance feature map was fed into the shape stream, assuming it could implicitly learn the IR
shape feature. However, results indicate that relying solely on IR appearance features fails to extract
meaningful shape cues, leading to suboptimal performance (Rank-1: 74.6%, mAP: 71.6%). Setting 2
improved performance (Rank-1: 75.4%, mAP: 71.9%) by directly adding the appearance feature map
to the IR shape feature map, and then inputting it into the shape stream network. However, without an
adaptive feature selection mechanism, it struggled to identify the required shape features for restoring
IR shape. To address this, we introduced a cross-attention mechanism that adaptively extracts required
shape information from IR images and utilizes it for IR shape restoration at the feature level. In query
design, we compared two strategies: Setting 3 (Rank-1: 74.9%, mAP: 71.8%), where only the IR
shape feature map was used as the query, and Setting 5, where both the IR shape and appearance
feature maps were combined. The superior performance of Setting 5 (Rank-1: 76.1%, mAP: 72.6%)
suggests that integrating both shape and appearance features provides better guidance for extracting
the required shape features from IR appearance features and effectively restoring IR shape representations. Additionally, in Setting 4, replacing spatial attention with channel attention resulted in the
performance similar to Setting 2 (Rank-1: 75.5%, mAP: 72.2%), indicating that channel attention is
insufficient. Finally, Setting 5, which employs spatial cross-attention, achieved the best performance
across all evaluation metrics, confirming that ISR effectively restores IR shapes at the feature level.
Effectiveness of AFE. In order to emphasize appearance features related to shape, we incorporated the AFE into the “Baseline+SFP+ISR”. The corresponding results, as depicted in Tab. 5, reflect
an enhancement in Rank-1, mAP, and mINP, in “All search” mode, these metrics increased from
74.6%, 71.0%, and 57.7% to 76.1%, 72.6%, and 59.8%, respectively. In “Indoor search” mode, the

21

Table 7: Ablation study on the design of the proposed AFE. This table presents a comparison of different design choices for
the AFE. Setting 1: concatenates shape and appearance features, Setting 2: replaces spatial attention with channel attention,
Setting 3: excludes the query loss, and Setting 4: our proposed AFE.

Setting
Setting 1
Setting 2
Setting 3
Setting 4

R1
74.6
74.9
74.9
76.1

R10
97.4
97.7
97.4
97.6

All search
R20 mAP mINP
99.3 71.0 57.7
99.4 71.8 59.0
99.2 71.4 58.3
99.4 72.6 59.8

R1
79.4
80.2
80.0
82.4

Indoor search
R10 R20 mAP mINP
98.4 99.3 83.2
79.8
98.7 99.4 83.8
80.5
98.9 99.8 83.7
80.4
98.8 99.8 85.4
82.2

metrics improved from 79.4%, 83.2%, and 79.8% to 82.4%, 85.4%, and 82.2% in Rank-1, mAP, and
mINP, respectively. This suggests that in contrast to typical appearance features, appearance features
centered on shape are better at emphasizing identity-discriminative features related to individuals
while minimizing the influence of modality, background, and other irrelevant noise features.
To further investigate the role of AFE and validate its design choices, we conduct an in-depth
analysis, as shown in Tab. 7, comparing different configurations and their impact on model performance. Taking the All search mode as an example, in Setting 1, shape and appearance features are
directly concatenated as the final representation (Rank-1: 74.6%, mAP: 71.0%), without leveraging
shape information to enhance appearance features. Due to the presence of modality-specific information and background noise in the appearance features, overall performance is limited. In Setting 2,
spatial attention is replaced with channel attention (Rank-1: 74.9%, mAP: 71.8%). While this leads to
a slight improvement in mAP and mINP compared to Setting 1, the gains are minimal, indicating that
channel attention is insufficient for effectively learning shape-related appearance features. In Setting
3, we remove the query loss term Lqwrt of AFE to evaluate its impact (Rank-1: 74.9%, mAP: 71.4%).
The results show a decline in Rank-1 and mAP, primarily due to the reduced discriminability of query
features in the second-stage attention. This, in turn, weakens the model’s ability to capture both direct
and indirect shape-related features, ultimately degrading recognition performance. Finally, in Setting 4 (AFE), both spatial attention and query loss are incorporated, achieving the best performance
(Rank-1: 76.1%, mAP: 72.6%) across all evaluation metrics, and demonstrating its effectiveness in
enhancing identity-discriminative features.
To further explore the design of cascaded multi-stage attention, we conducted additional experiments with four different settings, as shown in Tab. 8. Setting 1 represents the configuration without
AFE, corresponding to Baseline+SFP+ISR. Building upon this, Setting 2 (Baseline+SFP+ISR+S1)
incorporates a single-stage attention. The results show improvements in Rank-1, mAP, and mINP.
Specifically, compared to Setting 1, all search and indoor search modes exhibit gains of 0.7% (1.0%)
22

Table 8: Ablation studies of the proposed AFE. S1: first-stage attention mechanism. S2: second-stage attention mechanism.
S3: third-stage attention mechanism.

Setting
Setting 1
Setting 2
Setting 3
Setting 4

S1
✓
✓
✓

Stages
S2
S3

✓
✓

✓

R1
74.6
75.3
76.1
76.0

R10
97.4
97.6
97.6
97.4

All search
R20 mAP mINP
99.3 71.0 57.7
99.3 71.9 58.9
99.4 72.6 59.8
99.5 72.6 59.7

R1
79.4
80.4
82.4
80.9

Indoor search
R5
R10 mAP mINP
98.4 99.3 83.2 79.8
98.6 99.7 84.0 80.7
98.8 99.8 85.4 82.2
98.6 99.8 84.1 80.8

Table 9: Ablation Study on the Effectiveness of Shape and Appearance Features in ScRL. Where “App” represents appearance
features, while “Both” indicates the use of both appearance and shape features. The performance of CSDN is sourced from its
original paper.

Settings
B(App)
B + AFE(w/o ISR)
B + AFE(w ISR)
CSDN[48]
B(shape)
B(shape) + ISR
ScRL
CSDN+ScRL

Type

App

Shape
Both

R1
70.2
70.5
72.6
75.2
65.9
71.6
76.1
79.1

R10
95.7
96.3
96.7
96.6
95.7
96.8
97.6
98.1

All search
R20 mAP mINP
98.6 66.9 52.9
98.7 67.2 53.4
99.1 70.0 57.2
98.8 71.8
98.6 62.1 47.1
99.0 67.4 53.0
99.4 72.6 59.8
99.6 75.2 62.5

R1
77.6
77.3
80.6
82.0
68.9
75.3
82.4
85.3

Indoor search
R10 R20 mAP mINP
97.8 99.5 81.1 77.4
98.4 99.6 81.3 77.7
98.5 99.8 83.8 80.5
98.7 99.5 85.0
97.5 99.2 75.2 71.2
98.1 99.3 79.8 76.0
98.8 99.8 85.4 82.2
98.8 99.4 87.2 84.2

in Rank-1, 0.9% (0.8%) in mAP, and 1.2% (0.9%) in mINP, respectively. Further incorporating
second-stage attention leads to Setting 3 (Baseline+SFP+AFE), forming a two-stage attention approach, which is our proposed AFE. In all search, Rank-1, mAP, and mINP improved from 75.3%,
71.9%, and 58.9% to 76.1%, 72.6%, and 59.8%, respectively, while in indoor search, they increase
from 80.4%, 84.0%, and 80.7% to 82.4%, 85.4%, and 82.2%. However, adding a third attention stage
in Setting 4 (three-stage attention) results in a slight performance drop, indicating that two-stage attention is sufficient for learning both direct and indirect shape-related appearance features. Adding
additional stages may introduce redundancy, leading to diminished performance. These results confirm that a two-stage attention mechanism is the optimal design choice.
Complementarity of Shape and Appearance Features. To evaluate the complementarity between shape and appearance features, we conducted experiments using only appearance features and
only shape features, as presented in Tab. 9. First, comparing the appearance-based baseline “B(App)”
(using only appearance features) and the shape-based baseline “B(shape)” (using only shape features),
it is evident that appearance features outperform shape features overall, achieving higher Rank-1 and
mAP. This is because appearance features contain finer-grained identity cues, such as clothing details and local textures, which contribute to stronger identity discrimination. Next, incorporating ISR
into the shape stream “B(shape) + ISR” significantly improves performance. This indicates that ISR
23


hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Similarity Regulation and Calibration Alignment for Weakly
Supervised Text-Based Person Re-Identification
AO FU, JIAQI ZHAO, YONG ZHOU, WENLIANG DU, and RUI YAO, School of Computer
Science and Technology, China University of Mining and Technology, Xuzhou, China and Mine Digitization
Engineering Research Center of the Ministry of Education, China University of Mining and Technology,
Xuzhou, China
ABDULMOTALEB EL SADDIK, EECS, University of Ottawa, Ottawa, Ontario, Canada and Computer
Vision, Mohamed Bin Zayed University for Humanities, Abu Dhabi, United Arab Emirates
Traditional text-based person re-identification relies on identity labels. However, it is impossible to annotate
large datasets, since identity annotation is expensive and time-consuming. Weakly supervised text-based
person re-identification, where only text–image pairs are available without annotation of identities, is very
practical in real life. While dealing with the weakly supervised person re-identification, two issues should
be strengthed, i.e., alignment caused by different modal, and cross-modal matching ambiguity caused by the
lack of identity labels. In this article, we propose a similarity regulation and calibration alignment (SRCA)
framework, which consists of two unimodal encoders for images and text, respectively, and a multi-modal
encoder for the masked language modeling task. First, a similarity regulation (SR) strategy is proposed to relax
the strict one-to-one constraints for the local similarities between different pairs by introducing a novel soft
objective. The soft objective can adjust hard objectives to achieve soft cross-modal alignment by establishing
a many-to-many relationship between two modalities. Second, the calibration alignment (CA) module is
proposed to improve intra-class compactness by modeling pseudo-label assignment as optimal transport.
The ambiguity of cross-modal matching can be reduced by aligning features and pseudo-labels of different
modalities and gradually calibrating the distribution of pseudo-labels. Experimental results show that our
This work was supported by the National Natural Science Foundation of China (Nos. 62272461, 62172417, 62276266, and
62277046), and the “Double First-Class” Project of China University of Mining and Technology for Independent Innovation
and Social Service under Grant 2022ZZCX06, the Six Talent Peaks Project in Jiangsu Province (Nos. 2015-DZXX-010 and
2018-XYDXX-044).
Authors’ Contact Information: Ao Fu, School of Computer Science and Technology, China University of Mining and
Technology, Xuzhou, China and Mine Digitization Engineering Research Center of the Ministry of Education, China
University of Mining and Technology, Xuzhou, China; e-mail: fuao@cumt.edu.cn; Jiaqi Zhao (corresponding author), School
of Computer Science and Technology, China University of Mining and Technology, Xuzhou, China and Mine Digitization
Engineering Research Center of the Ministry of Education, China University of Mining and Technology, Xuzhou, China;
e-mail: jiaqizhao@cumt.edu.cn; Yong Zhou, School of Computer Science and Technology, China University of Mining
and Technology, Xuzhou, China and Mine Digitization Engineering Research Center of the Ministry of Education, China
University of Mining and Technology, Xuzhou, China; e-mail: yzhou@cumt.edu.cn; Wenliang Du, School of Computer
Science and Technology, China University of Mining and Technology, Xuzhou, China and Mine Digitization Engineering
Research Center of the Ministry of Education, China University of Mining and Technology, Xuzhou, China; e-mail:
wldu@cumt.edu.cn; Rui Yao, School of Computer Science and Technology, China University of Mining and Technology,
Xuzhou, China and Mine Digitization Engineering Research Center of the Ministry of Education, China University of
Mining and Technology, Xuzhou, China; e-mail: ruiyao@cumt.edu.cn; Abdulmotaleb El Saddik, EECS, University of Ottawa,
Ottawa, Ontario, Canada and Computer Vision, Mohamed Bin Zayed University for Humanities, Abu Dhabi, United Arab
Emirates; e-mail: elsaddik@uottawa.ca.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/3-ART96
https://doi.org/10.1145/3711861
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

96:2

A. Fu et al.

method has achieved obvious advantages compared with existing methods and also demonstrated competitive
performance compared with fully supervised methods.
CCS Concepts: • Computing methodologies → Object recognition; Computer vision; Natural language
processing; Search methodologies;
Additional Key Words and Phrases: Person Re-Identification, Cross-modal, Weakly Supervised
ACM Reference format:
Ao Fu, Jiaqi Zhao, Yong Zhou, Wenliang Du, Rui Yao, and Abdulmotaleb El Saddik. 2025. Similarity Regulation
and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification. ACM Trans. Multimedia
Comput. Commun. Appl. 21, 3, Article 96 (March 2025), 19 pages.
https://doi.org/10.1145/3711861

1

Introduction

Text-based person re-identification aims to retrieve person images that are highly semantically
related to a given text description . Text descriptions can provide more detailed and specific information that is easier to obtain than image information. It is easier to apply to actual projects
such as public security. Therefore, text-based person re-identification has received widespread
attention in recent years. However, traditional text-based person re-identification relies on identity annotation, and the process of annotating identities is expensive and time-consuming. Zhao
et al. [36] first proposed the weakly supervised text-based person re-identification task, weakly
supervised text-based person re-identification only requires text–image pairs, without any identity
annotations available. Since identity annotations are not required, the size of the dataset can be
increased more easily, and it has broader application prospects.
Due to the lack of identity annotations, weakly supervised text-based person re-identification
must address not only cross-modal alignment but also cross-modal matching ambiguities. As shown
in Figure 1, for a given text description, the cross-modal matching process cannot assign positive or
negative labels to any images other than the paired image. A text description can be semantically
paired with multiple images, leading to instances where false negatives and text anchors belong
to the same identity and exhibit local consistency. Furthermore, person re-identification datasets
often exhibit significant intra-class variations and minor inter-class differences, compounded by
the absence of identity annotations. This makes it challenging to mitigate the impact of intra-class
differences effectively.
To address these issues above, we propose a novel similarity regulation and calibration
alignment (SRCA) framework to enhance weakly supervised text-based person re-identification.
First, due to the absence of identity annotations, the system cannot assign positive or negative
labels to samples other than the paired ones during cross-modal matching. Given that there can
be local similarities between different image–text pairs, indicating a many-to-many relationship
rather than a perfect one-to-one correspondence, we introduce the similarity regulation (SR)
strategy. This strategy incorporates a novel soft objective to model the local similarity between
different pairs. However, directly optimizing with a vanilla soft objective cannot fully leverage the
significant one-to-one relationships between image–text pairs to enhance inter-class differences. To
address this, we combine the novel soft objective with the hard objective, thereby relaxing the strict
one-to-one constraint. The soft objective serves as a regularizer for the hard objective, establishing
a many-to-many relationship between the two modalities while maintaining a large inter-class
margin. This approach achieves soft cross-modal alignment by using soft objectives to enhance the
flexibility and robustness of the alignment process.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

SRCA for Weakly Supervised Text-Based Person Re-Identification

96:3

Fig. 1. Conceptual note of the proposed framework. This figure shows that due to the lack of identity
annotation, there are false negatives and partially similar negatives in the training data of weakly supervised
text-based person re-identification.

In addition, to further improve the intra-class compactness and increase the inter-class difference,
we propose a calibration alignment (CA) module. This module aligns the features of one modality
with the pseudo-labels of another modality to couple the features of different modalities. We noticed
that the dataset for person re-identification has a large number of categories, a small number of
instances of each category, and different categories are approximately uniformly distributed in the
dataset. Therefore, we use optimal transport for pseudo-label allocation and gradually calibrate the
distribution of pseudo-labels during the training process to further achieve cross-modal alignment
and reduce the ambiguity of cross-modal matching.
The pre-trained cross-modal model contrastive language-image pre-training (CLIP) [19]
is employed as the backbone of our framework. CLIP, which leverages an extensive dataset of
text–image pairs for pre-training, possesses robust cross-modal alignment capabilities. By utilizing
CLIP as the foundational model, we can transfer its powerful text–image matching abilities to the
task of text-based person re-identification. In addition, our model adopts a two-tower architecture,
allowing for the pre-computation of features. During the inference stage, only the cosine similarity
between features from different modalities needs to be calculated, ensuring a fast and efficient
text–image search process. Our main contributions are as follows:
— The SRCA framework is proposed for weakly supervised text-based person re-identification.
— The SR strategy is proposed to model local similarities between different pairs by introducing
a novel soft objective.
— The CA module is designed to improve intra-class compactness and increase inter-class
differences.
— Extensive experiments are conducted to demonstrate that the proposed method outperforms
the current state-of-the-art weakly supervised methods without any identity supervision and
is also competitive with fully supervised methods.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

96:4
2
2.1

A. Fu et al.

Related Work
Text-Based Person Search

Text-based person re-identification was introduced by Li et al. [17], who collected and released the
first large-scale text-person dataset and proposed a recurrent neural network with gated neural
attention. Wang et al. [26] utilized a special human attribute extraction network to align specific
attribute phrases to the corresponding visual areas. Gao et al. [8] proposed a method to adaptively
align image and text features at full scale. Zhu et al. [38] decoupled personal information and
surrounding information of visual features to effectively extract and match features about people.
Wu et al. [30] proposed a representation learning method for language-guided human search based
on color reasoning, which can explicitly establish fine-grained cross-modal associations. Han et al.
[12] used incremental contrastive learning for cross-modal alignment. Shao et al. [22] proposed
an end-to-end transformer-based framework to learn granular unified representations of both
modalities. Wei et al. [29] applied the masked language modeling (MLM) task to text-based
person re-identification for the first time. Wang et al. [24] applied complete CLIP to text-based
person re-identification tasks for the first time. Jiang and Ye [14] leveraged pre-trained CLIP and
MLM tasks to achieve state-of-the-art performance on text-based person re-identification using a
dual-stream encoder. The methods mentioned above rely on identity annotations. However, identity
annotation is quite expensive and time-consuming, which is not conducive to the expansion of
dataset size and limits its application in practical scenarios.
2.2

Weakly Supervised Text-Based Person Re-Identification

There are only a few studies [36] about the weakly supervised text-based person re-identification.
Patel et al. [18] leverage entire text articles and image captions to supervise the textual-visual
embeddings at both the local and global levels. Gomez et al. [11] extract feature embeddings from
images and the paired captions, which then are utilized to learn textual-visual joint embeddings.
These works are aimed at studying general text–image retrieval problems. Unlike general text–image
retrieval problems, identity information is crucial for learning specific identity feature embeddings
for text-based person re-identification. However, identity annotation is very expensive, which
hinders the development of text-based person re-identification. Text-based person re-identification
without identity annotation has wider application prospects. Zhao et al. [36] proposed weakly
supervised text-based person re-identification for the first time, utilizing pseudo-labels for selftraining in each modality and utilizing the text-IoU score as similarity soft-labels to facilitate the
cross-modal matching learning and the hard sample mining.
In this article, the proposed SRCA framework does not require identity annotation and introduces a novel soft objective through SR to relax the strict one-to-one constraints and establish
a many-to-many relationship between the two modalities. At the same time, the CA module
introduces optimal transport to solve pseudo-label assignment and further improves intra-class
compactness and inter-class distinguishability by mutually aligning features and pseudo-labels of
different modalities.
2.3

Vision-Language Pre-Training

With the development of CLIP [19], the visual-language pre-training (VLP) has been fully
developed. It has become the mainstream paradigm for learning multi-modal tasks [25, 34]. Existing
VLP models are mainly divided into two types, the two-tower model and the single-tower model.
The two-tower model independently extracts visual and text features and aligns them in the feature
space. The two-tower model can receive single-modal input and process them independently. Since
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

SRCA for Weakly Supervised Text-Based Person Re-Identification

96:5

Fig. 2. The overview of the proposed SRCA framework. (a) The proposed model consists of two unimodal
encoders for images and text, respectively, and a multi-modal encoder for the MLM task. The text data will
be randomly masked and input to the text encoder. The multi-modal encoder will be removed during the
inference stage. (b) Our SR module, which introduces soft objectives generated by semantic similarity. The
soft objectives serve as a regularizer for the hard objectives to relax the strict one-to-one constraints. Achieve
soft cross-modal alignment by establishing a many-to-many relationship between two modalities. (c) Our CA
module is used to ensure large margins between different classes to improve intra-class compactness.

the two-tower model can pre-compute single-modal features, it performs well in the field of crossmodal retrieval. Text-based person search is a sub-problem of the cross-modal retrieval problem,
usually using the twin-tower model to ensure efficiency and performance. The single-tower model
requires complex cross-modal fusion during the reasoning process and cannot store single-modal
feature vectors in advance, so it cannot be applied to large-scale text-based person re-identification.
3

Method

In this section, we detail the implementation of the proposed SRCA framework, an overview of
which is shown in Figure 2. It consists of two branches, one for image encoders and another for
text encoders. Cross-modal encoders perform fine-grained alignment by performing MLM. We
further propose two modules, including the SR module that establishes a many-to-many relationship
between two modalities to achieve soft cross-modal alignment. A CA module that increases margins
between different classes and improves compactness within classes. The details will be introduced
in the following.
3.1

Image–Text Encoder

In previous work on text-based person search, the pre-trained models ViT [6] and BERT [4] are
usually used as visual and textual encoders. With the development of cross-modal pre-training
tasks, recent work [12, 14] adopted cross-modal pre-trained CLIP [19] as the backbone of the model
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

96:6

A. Fu et al.

and obtained good performance. Inspired by related work, The pre-trained CLIP was adopted as
our visual and textual encoders to transfer the knowledge of the pre-trained cross-modal model.
Image Encoder. Feeding an input image 𝐼ˆ ∈ 𝑅 𝐻 ×𝑊 ×𝐶 to CLIP’s image encoder to obtain image
features. Similar to the procedure of VIT, first dividing 𝐼ˆ into 𝑁 = (𝐻 · 𝑊 )/(𝑃1 · 𝑃2 ) patches, where
(𝑃1, 𝑃 2 ) is the size of each patch, and then using a learnable linear projection layer to map each
patch to 1D tokens I = {𝐼𝑖 } |𝑖𝑁 , and adding an extra [CLS] token at the front, and then adding
positional embedding to these tokens. Sending the tokens sequence {𝐼𝑐𝑙𝑠 , 𝐼 1, · · ·𝐼 𝑁 } to the L-layer
transformer for modeling, and finally feeding the [CLS] tokens to the linear mapping layer to obtain
the global image embedding.
Text Encoder. Given an input text 𝑇ˆ , we extract text embeddings using CLIP’s text encoder.
According to CLIP, using a vocabulary of 49,152 to mark the text description, inserting an extra
[SOS] token {𝑇𝑆𝑂𝑆 } at the front of the text description T = {𝑇𝑖 } |𝑖𝑁 and inserting a [EOS] token
{𝑇𝐸𝑂𝑆 } at the end to mark the beginning and end of the text description. We randomly mask
out the input tokens with a probability of 15% and replace them with the special token [MASK ].
Following BERT, the replacements are 10% random tokens, 10% unchanged, and 80% [MASK ]. The
tokenized text {𝑇𝑆𝑂𝑆 ,𝑇1, · · ·𝑇𝑁 ,𝑇𝐸𝑂𝑆 } description is then feeding into an L-layer transformer, modeled using attention, and finally the [EOS] token {𝑇𝐸𝑂𝑆 } is linearly projected to obtain the final text
global embedding.
3.2

SR

Traditional contrastive learning treats all other samples as negative samples and negative samples
will be pushed away from the anchor point in the same way. However, weakly supervised text-based
person re-ID lacks identity annotations, false negatives and partially consistent negatives exist in
cross-modal data, and traditional contrastive learning is detrimentally affected by these samples.
Traditional text-based person re-identification assigns an identity tag to each image–text pair to
mark the same pedestrian identity. However, weakly supervised text-based person re-identification
lacks identity annotations, and samples of the same identity but different image–text pairs are
called false negatives. Obviously, if false negatives cannot be identified and are treated as correct
negative samples, the intra-class difference will increase, which is not conducive to model training.
Therefore, reducing the impact of false-negative samples on the model is crucial for text-based
person re-identification.
In order to reduce the impact of false negatives, we explored three different approaches to SR.
As shown in Figure 3(a), we first transfer part of the confidence from positive samples to negative
samples and use semantic similarity to generate soft objectives for optimization. This method
is ineffective in weakly supervised text-based person re-identification tasks. Directly using soft
objectives cannot model details well, cannot distinguish samples with small differences, and cannot
focus on hard samples. Second, we try to optimize the soft objective separately to regulate the
positive objective. This method can use the relationship between image and text pairs to better
distinguish hard samples with less difference. Although soft objectives can be adjusted for hard
objectives, positive samples will dominate vanilla soft objectives and cannot effectively model manyto-many relationships. To improve this, as shown in Figure 3(c), we propose the SR module. The
strict one-to-one constraints are relaxed and local similarities between different pairs are modeled
by introducing a novel soft objective. To avoid the dominance of positive samples in vanilla soft
objectives, we fill the diagonal elements of the soft objective matrix generated by semantic similarity
with 0 instead of 1. The soft objective becomes the regularization of the hard objective, which not
only can better distinguish hard samples with less difference but also establishes a many-to-many
relationship between the two modalities to achieve better soft cross-modal alignment.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

SRCA for Weakly Supervised Text-Based Person Re-Identification

96:7

Fig. 3. (a) The method of using only soft objectives. The method of directly using soft objectives cannot model
details well and cannot distinguish samples with small differences. (b) The method using a combination of
vanilla soft objective and hard objective. However, the improvement brought by the vanilla soft objective is
limited. Hard objectives still dominate, failing to narrow the margin of potential false negatives and positives.
(c) Our proposed SR, which can reduce the distance between positives and false negatives and achieve better
soft cross-modal alignment.

Formally, we input a mini-batch of each image–text pair (I, T) into the dual-stream model,
respectively. The image I is processed by the image encoder, the
by the text
 text T is processed

encoder, and the global features of the image and text 𝑭 𝑣 = 𝒇 𝑖𝑣 |𝑖𝐵 , 𝑭 𝑡 = 𝒇 𝑖𝑡 |𝑖𝐵 ∈ R𝐵×𝐷 are
obtained by normalization. B is the batch size and D is the dimension. Let 𝑠𝑖𝑚(𝒖, 𝒗) = 𝒖𝒗 > /k𝒖 k𝒗 k
represent the dot product between L2 normalized 𝒖 and 𝒗 (i.e., cosine similarity). Then, the crossmodal similarity probability can be calculated using the following softmax function:




exp sim 𝑓𝑖𝑣 , 𝑓 𝑗𝑡 /𝜏0
𝑣2𝑡



,
𝑝𝑖,𝑗
=Í
(1)
𝐵
𝑣 , 𝑓 𝑡 /𝜏
exp
sim
𝑓
0
𝑖
𝑘=1
𝑘
where 𝜏0 is the temperature hyper-parameter that controls the peak value of the probability
distribution. The probability 𝑝𝑖,𝑗 can be viewed as the ratio of the cosine similarity score between
𝑓𝑖𝑣 and 𝑓 𝑗𝑡 in the mini-batch to the cosine similarities between 𝑓𝑖𝑣 and {𝑓 𝑗𝑡 }𝐵𝑗=1 .
In order to distinguish hard samples with small differences and establish many-to-many relationships between different modalities, we set two objectives, namely hard objectives and soft
ℎ𝑎𝑟𝑑 = 1 means that the
objectives. The hard objective 𝒚ℎ𝑎𝑟𝑑 is defined by the identity matrix, 𝑦𝑖,𝑗
ℎ𝑎𝑟𝑑 = 0 means a non-matching pair. The hard
image and text are the same matching pair, while 𝑦𝑖,𝑗
objective treats negative samples equally without considering their semantic similarity to anchors.
Since hard objectives alone cannot model many-to-many relationships, we propose a novel soft
objective. We hypothesize that the similarity between images and text is positively correlated with
their semantic similarity. Therefore, we use semantic similarity within text modality to represent
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

96:8

A. Fu et al.

soft objectives and semantic similarity within text modality is calculated as follows:
>
e
𝑺 = 𝑭 𝑡 ⊗ 𝑭 𝑡 ∈ R𝐵×𝐵 ,

(2)

where ⊗ is matrix multiplication. However, vanilla semantic similarity as a soft objective will
cause positive sample pairs to dominate and cannot prominently modulate other sample pairs.
We fill the diagonal of the soft objective matrix generated by semantic similarity with 0 instead
of 1, which is the opposite direction of optimization for hard objectives. Soft objectives serve as
regulators for hard objectives. This regulator drives the model to appropriately optimize hard goals,
maintain small margins between positives and false negatives, relax strict one-to-one constraints,
and establish many-to-many relationships between different modalities. The revised soft objective
calculation method is as follows:
 
𝒚𝑠𝑜 𝑓 𝑡 = e
𝑺 − 𝑑𝑖𝑎𝑔 e
𝑺 ,
(3)
where diag extracts the diagonal elements of the matrix and creates a diagonal matrix. It is worth
noting that the model in the early training stage is unreliable because the semantic structure of
the embedding space is still being optimized, and it is unreasonable to directly use the semantic
similarity calculated by the text encoder. Therefore, in the beginning, we use sentence transformers
[20] to obtain semantic features and use them to estimate the semantic similarity of image–text
pairs in the early training stage. In this way, the semantic similarity can be gradually refined with
training, providing more reasonable soft objectives.
Next, we use the hard objective 𝑦ℎ𝑎𝑟𝑑 and the soft objective 𝑦𝑠𝑜 𝑓 𝑡 as labels, respectively, and
use KL divergence supervision to train the cross-modal similarity matrix 𝑆𝑡 2𝑣 , 𝑆 𝑣2𝑡 . The formula is
as follows:
𝐵


1Õ
𝑠𝑜 𝑓 𝑡
Lsoft-v2t =
KL 𝒑𝑖𝑣2𝑡 k𝒚ˆ𝑖
𝐵 𝑖=1
(4)
𝑣2𝑡
𝐵 𝐵
1 Õ Õ 𝑣2𝑡
© 𝑝𝑖,𝑗 ª
=
𝑝 log ­ 𝑠𝑜 𝑓 𝑡
®,
𝐵 𝑖=1 𝑗=1 𝑖,𝑗
𝑦ˆ𝑖,𝑗 + 𝜖
«
¬
𝐵


1Õ
𝑣2𝑡
ℎ𝑎𝑟𝑑
Lhard-v2t =
KL 𝒑𝑖 k𝒚ˆ𝑖
𝐵 𝑖=1
!
(5)
𝑣2𝑡
𝐵 𝐵
𝑝𝑖,𝑗
1 Õ Õ 𝑣2𝑡
=
𝑝 log ℎ𝑎𝑟𝑑
,
𝐵 𝑖=1 𝑗=1 𝑖,𝑗
𝑦ˆ𝑖,𝑗 + 𝜖
Í𝐵
where 𝜖 is a small number to avoid numerical problems, and 𝑦ˆ𝑖,𝑗 = 𝑦𝑖,𝑗 / 𝑘=1
𝑦𝑖,𝑘 is the probability
label. In fact, Lsoft-v2t can be regarded as a regular term, and the rule for calculating the gradient
becomes the following formula:
𝐵

∇𝜃 LSR = −

1Õ
𝑠𝑜 𝑓 𝑡
𝒑 ∇𝜃 (log 𝒚ˆℎ𝑎𝑟𝑑
+ log 𝒚ˆ𝑖 ),
𝑖
𝐵 𝑖=1 𝑖

(6)

where 𝜃 is the parameter of the model. This means that the model not only maximizes the difference
between the image–text pair and other samples during the optimization process but also needs
to consider the similarity with other samples. By using soft objectives to adjust hard objectives,
the model maximizes the gap between positive and negative samples, while paying attention to
the similarity between different samples and minimizes the impact of false negatives and partially
similar negative samples on model training.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.


codex
第二篇的重定义很清楚：弱监督文本 ReID 不是“缺身份标签所以只能做配对对比”，而是存在假负样本和局部相似负样本，严格一对一目标本身会伤训练。方法后半还在第 9 页之后，我继续读 CA 和训练目标。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 8 -l 13 'Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
96:8

A. Fu et al.

soft objectives and semantic similarity within text modality is calculated as follows:
>
e
𝑺 = 𝑭 𝑡 ⊗ 𝑭 𝑡 ∈ R𝐵×𝐵 ,

(2)

where ⊗ is matrix multiplication. However, vanilla semantic similarity as a soft objective will
cause positive sample pairs to dominate and cannot prominently modulate other sample pairs.
We fill the diagonal of the soft objective matrix generated by semantic similarity with 0 instead
of 1, which is the opposite direction of optimization for hard objectives. Soft objectives serve as
regulators for hard objectives. This regulator drives the model to appropriately optimize hard goals,
maintain small margins between positives and false negatives, relax strict one-to-one constraints,
and establish many-to-many relationships between different modalities. The revised soft objective
calculation method is as follows:
 
𝒚𝑠𝑜 𝑓 𝑡 = e
𝑺 − 𝑑𝑖𝑎𝑔 e
𝑺 ,
(3)
where diag extracts the diagonal elements of the matrix and creates a diagonal matrix. It is worth
noting that the model in the early training stage is unreliable because the semantic structure of
the embedding space is still being optimized, and it is unreasonable to directly use the semantic
similarity calculated by the text encoder. Therefore, in the beginning, we use sentence transformers
[20] to obtain semantic features and use them to estimate the semantic similarity of image–text
pairs in the early training stage. In this way, the semantic similarity can be gradually refined with
training, providing more reasonable soft objectives.
Next, we use the hard objective 𝑦ℎ𝑎𝑟𝑑 and the soft objective 𝑦𝑠𝑜 𝑓 𝑡 as labels, respectively, and
use KL divergence supervision to train the cross-modal similarity matrix 𝑆𝑡 2𝑣 , 𝑆 𝑣2𝑡 . The formula is
as follows:
𝐵


1Õ
𝑠𝑜 𝑓 𝑡
Lsoft-v2t =
KL 𝒑𝑖𝑣2𝑡 k𝒚ˆ𝑖
𝐵 𝑖=1
(4)
𝑣2𝑡
𝐵 𝐵
1 Õ Õ 𝑣2𝑡
© 𝑝𝑖,𝑗 ª
=
𝑝 log ­ 𝑠𝑜 𝑓 𝑡
®,
𝐵 𝑖=1 𝑗=1 𝑖,𝑗
𝑦ˆ𝑖,𝑗 + 𝜖
«
¬
𝐵


1Õ
𝑣2𝑡
ℎ𝑎𝑟𝑑
Lhard-v2t =
KL 𝒑𝑖 k𝒚ˆ𝑖
𝐵 𝑖=1
!
(5)
𝑣2𝑡
𝐵 𝐵
𝑝𝑖,𝑗
1 Õ Õ 𝑣2𝑡
=
𝑝 log ℎ𝑎𝑟𝑑
,
𝐵 𝑖=1 𝑗=1 𝑖,𝑗
𝑦ˆ𝑖,𝑗 + 𝜖
Í𝐵
where 𝜖 is a small number to avoid numerical problems, and 𝑦ˆ𝑖,𝑗 = 𝑦𝑖,𝑗 / 𝑘=1
𝑦𝑖,𝑘 is the probability
label. In fact, Lsoft-v2t can be regarded as a regular term, and the rule for calculating the gradient
becomes the following formula:
𝐵

∇𝜃 LSR = −

1Õ
𝑠𝑜 𝑓 𝑡
𝒑 ∇𝜃 (log 𝒚ˆℎ𝑎𝑟𝑑
+ log 𝒚ˆ𝑖 ),
𝑖
𝐵 𝑖=1 𝑖

(6)

where 𝜃 is the parameter of the model. This means that the model not only maximizes the difference
between the image–text pair and other samples during the optimization process but also needs
to consider the similarity with other samples. By using soft objectives to adjust hard objectives,
the model maximizes the gap between positive and negative samples, while paying attention to
the similarity between different samples and minimizes the impact of false negatives and partially
similar negative samples on model training.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

SRCA for Weakly Supervised Text-Based Person Re-Identification

96:9

Fig. 4. Illustration of the purpose of the CA module. The CA module can further reduce the margins of false
negatives and positives based on SR and enhance cross-modal intra-class compactness.
𝑡 2𝑣 can be defined by exchanging 𝐹 𝑣 and 𝐹 𝑡 in Equation (1), and L
Symmetrically, 𝑝𝑖,𝑗
hard-t2v, Lsoft-t2v
𝑡 2𝑣 instead of 𝑝 𝑣2𝑡 . The loss of SR is calculated by
can be calculated in Equations (4) and (5) using 𝑝𝑖,𝑗
𝑖,𝑗
the following formula:

LSR = Lhard-v2t + L hard-t2v + 𝜆0 (Lsoft-v2t + Lsoft-t2v ) ,

(7)

where 𝜆0 is the hyper-parameter that adjusts the loss weight.
3.3

CA

The SR module establishes a many-to-many relationship between different modalities. However,
the SR module has limited effect in shortening false negatives and anchor distance. As shown in
Figure 4, in order to further improve intra-class compactness and reduce the margin between false
negatives and anchors, we propose the CA module. The CA module aligns features of different
modalities and pseudo-labels of other modalities with each other, while gradually calibrating the
distribution of pseudo-labels to reduce the harmful effects of false negatives and further achieve
cross-modal alignment.
Traditional clustering methods may produce degenerate solutions, that is, all data points are
mapped to the same cluster. Unlike some other classification tasks that lack labels, in the text-based
person re-identification task, samples of different identities are evenly distributed, and the number
of person identities can be roughly estimated, which means that we have prior information about
the number of categories, which is often ignored by previous work. So we can use this information
to add the constraint that the labels must induce an equipartition of the data. Inspired by [1],
the pseudo-label assignment problem is transformed into optimal transport, which avoids the
degenerate solution that may be produced by traditional algorithms by maximizing the mutual
information between labels and input data indices. Taking the visual branch as an example, consider
the classification problem optimized using cross-entropy loss:
𝐵

𝐸 (𝒑 𝑣 , 𝒒 𝑣 ) = −

𝐾



1 ÕÕ 𝑣
𝑞 𝑦 | 𝒇 𝑖𝑣 log 𝑝 𝑣 𝑦 | 𝒇 𝑖𝑣 ,
𝐵 𝑖=1 𝑦=1

(8)

where {𝒇𝑖𝑣 , 𝒇 𝑖𝑡 } represents the features of the unlabeled image–text pair obtained by the model,
𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 is the pseudo-label. The basic idea of pseudo-labeling is to use an initial model to predict
unlabeled data and use these prediction results as “labels” to train the model. Here, we model the
calculation of pseudo-labels as an optimal transport problem. The pseudo-label is an unknown
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

96:10

A. Fu et al.

quantity in this formula and is obtained by
transport problem. Each epoch
 solving the optimal

needs to calculate pseudo-labels 𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 and use 𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 to optimize model parameters.
Then, each sample feature is input into the classifier for prediction, and the probability distribution
vector of each sample belonging to each category is obtained:

exp 𝑾 𝑦𝑖 >𝒇 𝑖𝑣 /𝜏1
𝑣
𝑣
𝑝 𝑦 | 𝒇 𝑖 = Í𝐾
(9)
,
> 𝑣
𝑗=1 exp 𝑾 𝑗 𝒇 𝑖 /𝜏1

where 𝑝 𝑣 𝑦 | 𝒇 𝑖𝑣 is the output of the classifier, W is the cross-modal shared learnable normalized
weight, 𝜏 is the scaling factor.
Since text-based person re-identification datasets have a large number of categories and few
instances of each category. This means that the number of instances of different classes
in the

dataset is evenly distributed, so equipartition constraint can be imposed on 𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 . As shown
in the following formula:

min
𝐸 (𝒑 𝑣 , 𝒒 𝑣 ) s.t. ∀𝑦 : 𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 ∈ {0, 1},
𝑣
𝒒

 𝐵
𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 = .
𝑖=1
𝐾

Õ𝐵

(10)



𝑣 = 1 𝑝 𝑣 𝑦 | 𝒇 𝑣 , 𝑄 𝑣 = 1 𝑞𝑣 𝑦 | 𝒇 𝑣 ,
At the same time, we define 𝑷 𝑣 ∈ R𝐾 ×𝐵 , 𝑸 𝑣 ∈ R𝐾 ×𝐵 , where 𝑃 𝑦𝑖
𝑖
𝑖
𝑦𝑖
𝐵
𝐵
K is the number of categories. Fix 𝑝 𝑣 , optimize 𝑞 𝑣 individually, and define the feasible solution
space of 𝑸 𝑣 that satisfies the equipartition constraint as 𝑼 𝑣 (𝒓, 𝒄):

𝑼 𝑣 (𝒓, 𝒄) = 𝑸 𝑣 ∈ R𝐾+ ×𝐵 | 𝑸 𝑣 1𝐵 = 𝒓, 𝑸 𝑣 > 1𝐾 = 𝒄 ,
(11)
where 𝒓 = 𝐾1 1𝐾 , 𝒄 = 𝐵1 1𝐵 . Therefore, formula (7) can be transformed into:
𝐵

𝐸 (𝒑 𝑣 , 𝒒 𝑣 ) = −

=−

=−

−

𝐾



1 ÕÕ 𝑣
𝑞 𝑦 | 𝒇 𝑖𝑣 log 𝑝 𝑣 𝑦 | 𝒇 𝑖𝑣
𝐵 𝑖=1 𝑦=1
𝐵 Õ
𝐾
Õ
1



1
𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 log 𝑝 𝑣 𝑦 | 𝒇 𝑖𝑣 × 𝐵
𝐵
𝐵
𝑖=1 𝑦=1

𝐵 Õ
𝐾
Õ
1



1
𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 log 𝑝 𝑣 𝑦 | 𝒇 𝑖𝑣
𝐵
𝐵
𝑖=1 𝑦=1
𝐵 Õ
𝐾
Õ
1

𝐵


𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 log 𝐵

𝑖=1 𝑦=1
𝑣

= h𝑸 , − log 𝑷 𝑣 i − log 𝐵,

(12)

where <, > is Frobenius inner product. Therefore, formula (7) is transformed into the optimal
transport problem:
min

𝑸 𝑣 ∈𝑼 𝑣 (𝑟,𝑐 )

h𝑸 𝑣 , − log 𝑷 𝑣 i,

(13)

where − log 𝑷 𝑣 is the cost matrix. The Sinkhorn-Knopp
[3] algorithm can be used to iteratively

solve to obtain the pseudo-label 𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 . Similarly, the pseudo-label 𝑞𝑡 𝑦 | 𝒇 𝑖𝑡 of the text branch
can be obtained. Then we fix the pseudo-label 𝒒 𝑣 , 𝒒𝑡 and optimize the model parameters. Text
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

SRCA for Weakly Supervised Text-Based Person Re-Identification

96:11

pseudo-labels to train image representation, image pseudo-labels to train text representation:
𝐵

𝐾

LCA-t = −



1 ÕÕ 𝑣
𝑞 𝑦 | 𝒇 𝑖𝑣 log 𝑝 𝑡 𝑦 | 𝒇 𝑖𝑡 ,
𝐵 𝑖=1 𝑦=1

LCA-v = −



1 ÕÕ 𝑡
𝑞 𝑦 | 𝒇 𝑖𝑡 log 𝑝 𝑣 𝑦 | 𝒇 𝑖𝑣 .
𝐵 𝑖=1 𝑦=1

𝐵

(14)

𝐾

(15)

The CA module gradually calibrates the pseudo-label distribution 𝑸 𝑣 , 𝑸 𝑡 during the model
training process. Assign features of different modalities to pseudo-labels of other modalities to
reduce intra-class gaps between different modalities. The CA loss is as follows:
LCA = 𝜆1 (LCA-v + LCA-t ),

(16)

where 𝜆1 is the hyper-parameter that adjusts the loss weight.
3.4

Objective

MLM utilizes both the image and the contextual text to predict the masked words. Following BERT
[4], we randomly mask out the input tokens with a probability of 15% and replace them with
the special token [MASK]. Let 𝑇 msk denote a masked text and 𝒑 msk (𝐼,𝑇 msk ) denote the model’s
predicted probability for a masked token. The objective of MLM is to minimize the cross-entropy
loss Lmlm :


Lmlm = E (𝐼,𝑇 msk )∼D H 𝒚 msk , 𝒑 msk (𝐼,𝑇 msk ) ,
(17)
where 𝑦𝑚𝑠𝑘 is a one-hot vocabulary distribution where the ground-truth token has a probability of 1.
We optimize pseudo-label assignment via optimal transport and jointly optimize three cross-modal
alignment modules to train our SRCA framework, encouraging the network to learn discriminative
and generalized text–image representations. The overall training objective can be represented as
LSRCA = LSR + LCA + Lmlm .
4
4.1

(18)

Experiments
Experimental Setup

In this section, three common benchmark datasets are used to evaluate the proposed method.
Dataset. CUHK-PEDES [17] is the first and most popular evaluation benchmark for text-based
person search. Its training set consists of 34,054 images 11,003 person identities, and 2 corresponding
sentence captions for each image, and its validation and test sets, respectively, consist of 3,078
and 3,074 images of 1,000 identities. ICFG-PEDES [5] contains a total of 54,522 images for 4,102
identities. Each image has only one corresponding textual description. The dataset is divided into
a training set and a test set, the former comprises 34,674 image–text pairs of 3,102 identities,
while the latter contains 19,848 image–text pairs for the remaining 1,000 identities. RSTPReid [39]
contains 20,505 images of 4,101 identities from 15 cameras. Each identity has five corresponding
images taken by different cameras and each image is annotated with two textual descriptions.
Following the official data split, the training, validation and test sets contain 3,701, 200, and 200
identities, respectively.
Evaluation Metrics. We adopt the popular metric Rank-k as the main evaluation metric. Given
the specified query text to retrieve its corresponding person in the image library. Rank images by
text–image similarity. Rank-k is the probability of finding at least one matching person image in
the top k candidate lists when given a textual description as a query. We evaluate the accuracy of
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

96:12

A. Fu et al.

rank-1, 5, and 10 in experiments. In addition, we also adopted the mAP evaluation metric. Higher
Rank-k and mAP indicate better performance.
Implementation Details. Our model includes pre-trained CLIP’s text encoder and image encoder as
the backbone, as well as our proposed ESA module and a multi-modal encoder. The input embedding
dimension is set to 512, the input image size is resized to 384 × 128, and the maximum text sequence
length is set to 77. During training, horizontal flipping, random cropping, and random wiping are
used for image enhancement. 𝜆0 is set to 3.0 and 𝜆1 is set to 1.0. 𝜏0 is set to 0.02, 𝜏1 is set to 0.2. Our
model has been trained for 60 epochs with the Adam [15] optimizer, the learning rate is initialized
to 1 × 10−5 , and the cosine learning rate decays. In the beginning, we spend 5 warmup epochs,
linearly increasing the learning rate from 1 × 10−6 to 1 × 10−5 . For randomly initialized modules,
we set the initial learning rate to 5 × 10−5 . We perform our experiments on two RTX2080Ti GPUs
with PyTorch framework.
4.2

Comparison with State-of-the-Art Methods

Methods such as TIPCB [2] use traditional convolutional neural networks as backbones. LGUR
[22] proposed PGU and DGA modules to improve the fine-grained resolution ability of the model.
CAIBC [27] improves the performance of the model through color information. Han et al. [12]
first introduced the pre-trained CLIP image encoder and BERT [4] as the backbone, and CFine [32]
improved the fine-grained recognition of the model on this basis. TP-TPS [24] pointed out that the
previous VLP method only used a separate image encoder ignored the role of the text encoder and
proposed two pre-training mode models for the text encoder. IRRA [14] goes a step further and does
not require a separate pre-trained text encoder. It uses the fully pre-trained CLIP text encoder and
image encoder for fine-tuning. PLIP [40] automatically generates text annotations and constructs a
large-scale image–text pair task dataset to pre-train the VLP model of the person image–text pair
dataset. In recent years, these works have mainly focused on the fine-grained recognition ability of
the backbone and model, while ignoring the reality that the labeling of person image–text pairs
is expensive.
At the same time, due to the similarities between different person image–text pairs, it is unreasonable for these methods to completely rely on annotated hard labels to push away all negative
samples. However, few studies have been conducted on this. CMMT [36] first introduced weakly
supervised text-based person re-identification, considering more realistic problems. It does not
use labels that are difficult and expensive to obtain and uses text-IOU to measure the similarity
between different text pairs. However, this method strictly relies on the prior similarity of words
between texts and ignores the semantic similarity between different descriptions. In order to solve
this problem, this article proposes a SRCA framework. We propose a SR strategy, which introduces
a novel soft objective to relax the strict one-to-one constraints and model the local similarities
between different pairs. To further improve intra-class compactness, we propose a CA module that
models pseudo-label assignment as optimal transport, further achieves cross-modal alignment and
reduces the ambiguity of cross-modal matching.
In this section, we will show the comparison results with the state-of-the-art methods on three
public benchmark datasets to show the superiority and universality of our method.
CUHK-PEDES. As shown in Table 1, we compare the proposed SRCA method with other stateof-the-art methods. Our proposed SRCA achieved 73.23% Rank-1 accuracy and 65.60% mAP, significantly surpassing the previous best model CMMT with an R1 of 16.13%. To the best of our
knowledge, this is the first among all weakly supervised text-based person re-identification methods
to achieve an R1 accuracy of over 73%. It is worth noting that our method still achieves competitive
performance even when compared with fully supervised methods based on twin-tower encoders.
Our method performs comparably well to the fully supervised method IRRA [14]. This performance
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

SRCA for Weakly Supervised Text-Based Person Re-Identification

96:13

Table 1. Comparison of State-of-the-Art Methods on CUHK-PEDES Dataset

Method
CMPM/C [35]
TIMAM [21]
ViTAA [26]
NAFS [8]
DSSL [38]
SSAN [5]
LapsCore [30]
ACSA [13]
MFPE [31]
ISANet [33]
LBUL [28]
Han et al. [12]
SAF [16]
TIPCB [2]
CAIBC [27]
AXM-Net [7]
LGUR [22]
IVT [23]
PLIP [40]
CFine [32]
Wei et al. [29]
TP-TPS [24]
IRRA [14]
MM-TIM [11]
CMPM [35]+SpCL [10]
CMPM [35]+MMT [9]
CMMT [36]
SRCA

Ref
ECCV18
ICCV19
ECCV20
arXiv21
MM21
arXiv21
ICCV21
TMM
arXiv23
arXiv22
MM22
BMVC21
ICASSP22
Neuro22
MM22
AAAI23
MM22
ECCVW22
arXiv23
arXiv22
arXiv23
arXiv23
CVPR23
MSU19
NIPS20
ICLR20
ICCV21

Supervise
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Weakly Sup
Weakly Sup
Weakly Sup
Weakly Sup
Weakly Sup

R1
49.37
54.51
54.92
59.36
59.98
61.37
63.4
63.56
63.82
63.92
64.04
64.08
64.13
64.26
64.43
64.44
65.25
65.59
69.23
69.57
69.61
70.16
73.38
45.35
51.13
50.51
57.10
73.23

R5
77.56
75.18
79.13
80.41
80.15
81.4
82.63
82.15
82.66
81.73
82.62
83.19
82.87
80.52
83.12
83.11
85.84
85.93
86.01
86.1
89.93
63.78
71.54
70.23
78.14
89.47

R10
79.27
79.27
82.9
86
87.56
86.73
87.8
87.7
88.66
87.69
87.22
88.19
88.4
89.1
88.37
86.77
89
89.21
91.16
91.15
90.9
90.98
93.71
70.63
80.03
78.98
85.23
93.73

mAP
51.6
54.07
60.08
58.73
66.13
65.60

Results are ordered based on the Rank-1 accuracy.

is achieved jointly by the three modules we set up: SR establishes many-to-many relationships
between different modalities, and CA further increases cross-modal intra-class compactness. Finally,
the MLM task enhances the fine-grained perception capabilities of the model.
ICFG-PEDES. The comparison results are shown in Table 2. Our proposed SRCA method achieves
63.11% Rank-1 accuracy and 36.61% mAP on the ICFG-PEDES dataset. The performance of weakly
supervised SRCA is close to the current best fully supervised method IRRA. Even compared with
the fully supervised method PLIP, which uses extra data pre-training, the SRCA method shows
competitive performance. The effectiveness of our proposed SRCA on this dataset was verified
through experiments.
RSTPReid. According to the comparison results in Table 3, our proposed SRCA method also has
significant advantages, with Rnak-1 accuracy of 61.2% and mAP of 46.93%. Our proposed SRCA
achieves 1% higher R1 than the current best fully supervised method IRRA and achieves 22.15%
higher Rank-1 accuracy than DSSL [39].
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.


hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf' - | rg -n -C 3 'Ablation|ablation|Effectiveness|SR module|CA module|visual|pseudo|Table 4|Table 5|Table 6|Figure 5|similarity|false negative'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
11-person re-identification, where only text–image pairs are available without annotation of identities, is very
12-practical in real life. While dealing with the weakly supervised person re-identification, two issues should
13-be strengthed, i.e., alignment caused by different modal, and cross-modal matching ambiguity caused by the
14:lack of identity labels. In this article, we propose a similarity regulation and calibration alignment (SRCA)
15-framework, which consists of two unimodal encoders for images and text, respectively, and a multi-modal
16:encoder for the masked language modeling task. First, a similarity regulation (SR) strategy is proposed to relax
17-the strict one-to-one constraints for the local similarities between different pairs by introducing a novel soft
18-objective. The soft objective can adjust hard objectives to achieve soft cross-modal alignment by establishing
19-a many-to-many relationship between two modalities. Second, the calibration alignment (CA) module is
20:proposed to improve intra-class compactness by modeling pseudo-label assignment as optimal transport.
21:The ambiguity of cross-modal matching can be reduced by aligning features and pseudo-labels of different
22:modalities and gradually calibrating the distribution of pseudo-labels. Experimental results show that our
23-This work was supported by the National Natural Science Foundation of China (Nos. 62272461, 62172417, 62276266, and
24-62277046), and the “Double First-Class” Project of China University of Mining and Technology for Independent Innovation
25-and Social Service under Grant 2022ZZCX06, the Six Talent Peaks Project in Jiangsu Province (Nos. 2015-DZXX-010 and
--
80-must address not only cross-modal alignment but also cross-modal matching ambiguities. As shown
81-in Figure 1, for a given text description, the cross-modal matching process cannot assign positive or
82-negative labels to any images other than the paired image. A text description can be semantically
83:paired with multiple images, leading to instances where false negatives and text anchors belong
84-to the same identity and exhibit local consistency. Furthermore, person re-identification datasets
85-often exhibit significant intra-class variations and minor inter-class differences, compounded by
86-the absence of identity annotations. This makes it challenging to mitigate the impact of intra-class
87-differences effectively.
88:To address these issues above, we propose a novel similarity regulation and calibration
89-alignment (SRCA) framework to enhance weakly supervised text-based person re-identification.
90-First, due to the absence of identity annotations, the system cannot assign positive or negative
91-labels to samples other than the paired ones during cross-modal matching. Given that there can
92-be local similarities between different image–text pairs, indicating a many-to-many relationship
93:rather than a perfect one-to-one correspondence, we introduce the similarity regulation (SR)
94:strategy. This strategy incorporates a novel soft objective to model the local similarity between
95-different pairs. However, directly optimizing with a vanilla soft objective cannot fully leverage the
96-significant one-to-one relationships between image–text pairs to enhance inter-class differences. To
97-address this, we combine the novel soft objective with the hard objective, thereby relaxing the strict
--
106-96:3
107-
108-Fig. 1. Conceptual note of the proposed framework. This figure shows that due to the lack of identity
109:annotation, there are false negatives and partially similar negatives in the training data of weakly supervised
110-text-based person re-identification.
111-
112-In addition, to further improve the intra-class compactness and increase the inter-class difference,
113-we propose a calibration alignment (CA) module. This module aligns the features of one modality
114:with the pseudo-labels of another modality to couple the features of different modalities. We noticed
115-that the dataset for person re-identification has a large number of categories, a small number of
116-instances of each category, and different categories are approximately uniformly distributed in the
117:dataset. Therefore, we use optimal transport for pseudo-label allocation and gradually calibrate the
118:distribution of pseudo-labels during the training process to further achieve cross-modal alignment
119-and reduce the ambiguity of cross-modal matching.
120-The pre-trained cross-modal model contrastive language-image pre-training (CLIP) [19]
121-is employed as the backbone of our framework. CLIP, which leverages an extensive dataset of
122-text–image pairs for pre-training, possesses robust cross-modal alignment capabilities. By utilizing
123-CLIP as the foundational model, we can transfer its powerful text–image matching abilities to the
124-task of text-based person re-identification. In addition, our model adopts a two-tower architecture,
125:allowing for the pre-computation of features. During the inference stage, only the cosine similarity
126-between features from different modalities needs to be calculated, ensuring a fast and efficient
127-text–image search process. Our main contributions are as follows:
128-— The SRCA framework is proposed for weakly supervised text-based person re-identification.
129-— The SR strategy is proposed to model local similarities between different pairs by introducing
130-a novel soft objective.
131:— The CA module is designed to improve intra-class compactness and increase inter-class
132-differences.
133-— Extensive experiments are conducted to demonstrate that the proposed method outperforms
134-the current state-of-the-art weakly supervised methods without any identity supervision and
--
147-Text-based person re-identification was introduced by Li et al. [17], who collected and released the
148-first large-scale text-person dataset and proposed a recurrent neural network with gated neural
149-attention. Wang et al. [26] utilized a special human attribute extraction network to align specific
150:attribute phrases to the corresponding visual areas. Gao et al. [8] proposed a method to adaptively
151-align image and text features at full scale. Zhu et al. [38] decoupled personal information and
152:surrounding information of visual features to effectively extract and match features about people.
153-Wu et al. [30] proposed a representation learning method for language-guided human search based
154-on color reasoning, which can explicitly establish fine-grained cross-modal associations. Han et al.
155-[12] used incremental contrastive learning for cross-modal alignment. Shao et al. [22] proposed
--
166-Weakly Supervised Text-Based Person Re-Identification
167-
168-There are only a few studies [36] about the weakly supervised text-based person re-identification.
169:Patel et al. [18] leverage entire text articles and image captions to supervise the textual-visual
170-embeddings at both the local and global levels. Gomez et al. [11] extract feature embeddings from
171:images and the paired captions, which then are utilized to learn textual-visual joint embeddings.
172-These works are aimed at studying general text–image retrieval problems. Unlike general text–image
173-retrieval problems, identity information is crucial for learning specific identity feature embeddings
174-for text-based person re-identification. However, identity annotation is very expensive, which
175-hinders the development of text-based person re-identification. Text-based person re-identification
176-without identity annotation has wider application prospects. Zhao et al. [36] proposed weakly
177:supervised text-based person re-identification for the first time, utilizing pseudo-labels for selftraining in each modality and utilizing the text-IoU score as similarity soft-labels to facilitate the
178-cross-modal matching learning and the hard sample mining.
179-In this article, the proposed SRCA framework does not require identity annotation and introduces a novel soft objective through SR to relax the strict one-to-one constraints and establish
180:a many-to-many relationship between the two modalities. At the same time, the CA module
181:introduces optimal transport to solve pseudo-label assignment and further improves intra-class
182:compactness and inter-class distinguishability by mutually aligning features and pseudo-labels of
183-different modalities.
184-2.3
185-
186-Vision-Language Pre-Training
187-
188:With the development of CLIP [19], the visual-language pre-training (VLP) has been fully
189-developed. It has become the mainstream paradigm for learning multi-modal tasks [25, 34]. Existing
190-VLP models are mainly divided into two types, the two-tower model and the single-tower model.
191:The two-tower model independently extracts visual and text features and aligns them in the feature
192-space. The two-tower model can receive single-modal input and process them independently. Since
193-ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.
194-
--
199-Fig. 2. The overview of the proposed SRCA framework. (a) The proposed model consists of two unimodal
200-encoders for images and text, respectively, and a multi-modal encoder for the MLM task. The text data will
201-be randomly masked and input to the text encoder. The multi-modal encoder will be removed during the
202:inference stage. (b) Our SR module, which introduces soft objectives generated by semantic similarity. The
203-soft objectives serve as a regularizer for the hard objectives to relax the strict one-to-one constraints. Achieve
204-soft cross-modal alignment by establishing a many-to-many relationship between two modalities. (c) Our CA
205-module is used to ensure large margins between different classes to improve intra-class compactness.
--
215-In this section, we detail the implementation of the proposed SRCA framework, an overview of
216-which is shown in Figure 2. It consists of two branches, one for image encoders and another for
217-text encoders. Cross-modal encoders perform fine-grained alignment by performing MLM. We
218:further propose two modules, including the SR module that establishes a many-to-many relationship
219:between two modalities to achieve soft cross-modal alignment. A CA module that increases margins
220-between different classes and improves compactness within classes. The details will be introduced
221-in the following.
222-3.1
--
224-Image–Text Encoder
225-
226-In previous work on text-based person search, the pre-trained models ViT [6] and BERT [4] are
227:usually used as visual and textual encoders. With the development of cross-modal pre-training
228-tasks, recent work [12, 14] adopted cross-modal pre-trained CLIP [19] as the backbone of the model
229-ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.
230-
--
233-A. Fu et al.
234-
235-and obtained good performance. Inspired by related work, The pre-trained CLIP was adopted as
236:our visual and textual encoders to transfer the knowledge of the pre-trained cross-modal model.
237-Image Encoder. Feeding an input image 𝐼ˆ ∈ 𝑅 𝐻 ×𝑊 ×𝐶 to CLIP’s image encoder to obtain image
238-features. Similar to the procedure of VIT, first dividing 𝐼ˆ into 𝑁 = (𝐻 · 𝑊 )/(𝑃1 · 𝑃2 ) patches, where
239-(𝑃1, 𝑃 2 ) is the size of each patch, and then using a learnable linear projection layer to map each
--
255-
256-Traditional contrastive learning treats all other samples as negative samples and negative samples
257-will be pushed away from the anchor point in the same way. However, weakly supervised text-based
258:person re-ID lacks identity annotations, false negatives and partially consistent negatives exist in
259-cross-modal data, and traditional contrastive learning is detrimentally affected by these samples.
260-Traditional text-based person re-identification assigns an identity tag to each image–text pair to
261-mark the same pedestrian identity. However, weakly supervised text-based person re-identification
262-lacks identity annotations, and samples of the same identity but different image–text pairs are
263:called false negatives. Obviously, if false negatives cannot be identified and are treated as correct
264-negative samples, the intra-class difference will increase, which is not conducive to model training.
265-Therefore, reducing the impact of false-negative samples on the model is crucial for text-based
266-person re-identification.
267:In order to reduce the impact of false negatives, we explored three different approaches to SR.
268-As shown in Figure 3(a), we first transfer part of the confidence from positive samples to negative
269:samples and use semantic similarity to generate soft objectives for optimization. This method
270-is ineffective in weakly supervised text-based person re-identification tasks. Directly using soft
271-objectives cannot model details well, cannot distinguish samples with small differences, and cannot
272-focus on hard samples. Second, we try to optimize the soft objective separately to regulate the
273-positive objective. This method can use the relationship between image and text pairs to better
274-distinguish hard samples with less difference. Although soft objectives can be adjusted for hard
275:objectives, positive samples will dominate vanilla soft objectives and cannot effectively model manyto-many relationships. To improve this, as shown in Figure 3(c), we propose the SR module. The
276-strict one-to-one constraints are relaxed and local similarities between different pairs are modeled
277-by introducing a novel soft objective. To avoid the dominance of positive samples in vanilla soft
278:objectives, we fill the diagonal elements of the soft objective matrix generated by semantic similarity
279-with 0 instead of 1. The soft objective becomes the regularization of the hard objective, which not
280-only can better distinguish hard samples with less difference but also establishes a many-to-many
281-relationship between the two modalities to achieve better soft cross-modal alignment.
--
288-Fig. 3. (a) The method of using only soft objectives. The method of directly using soft objectives cannot model
289-details well and cannot distinguish samples with small differences. (b) The method using a combination of
290-vanilla soft objective and hard objective. However, the improvement brought by the vanilla soft objective is
291:limited. Hard objectives still dominate, failing to narrow the margin of potential false negatives and positives.
292:(c) Our proposed SR, which can reduce the distance between positives and false negatives and achieve better
293-soft cross-modal alignment.
294-
295-Formally, we input a mini-batch of each image–text pair (I, T) into the dual-stream model,
--
299-
300-encoder, and the global features of the image and text 𝑭 𝑣 = 𝒇 𝑖𝑣 |𝑖𝐵 , 𝑭 𝑡 = 𝒇 𝑖𝑡 |𝑖𝐵 ∈ R𝐵×𝐷 are
301-obtained by normalization. B is the batch size and D is the dimension. Let 𝑠𝑖𝑚(𝒖, 𝒗) = 𝒖𝒗 > /k𝒖 k𝒗 k
302:represent the dot product between L2 normalized 𝒖 and 𝒗 (i.e., cosine similarity). Then, the crossmodal similarity probability can be calculated using the following softmax function:
303-
304-
305-
--
323-𝑘=1
324-𝑘
325-where 𝜏0 is the temperature hyper-parameter that controls the peak value of the probability
326:distribution. The probability 𝑝𝑖,𝑗 can be viewed as the ratio of the cosine similarity score between
327-𝑓𝑖𝑣 and 𝑓 𝑗𝑡 in the mini-batch to the cosine similarities between 𝑓𝑖𝑣 and {𝑓 𝑗𝑡 }𝐵𝑗=1 .
328-In order to distinguish hard samples with small differences and establish many-to-many relationships between different modalities, we set two objectives, namely hard objectives and soft
329-ℎ𝑎𝑟𝑑 = 1 means that the
330-objectives. The hard objective 𝒚ℎ𝑎𝑟𝑑 is defined by the identity matrix, 𝑦𝑖,𝑗
331-ℎ𝑎𝑟𝑑 = 0 means a non-matching pair. The hard
332-image and text are the same matching pair, while 𝑦𝑖,𝑗
333:objective treats negative samples equally without considering their semantic similarity to anchors.
334-Since hard objectives alone cannot model many-to-many relationships, we propose a novel soft
335:objective. We hypothesize that the similarity between images and text is positively correlated with
336:their semantic similarity. Therefore, we use semantic similarity within text modality to represent
337-ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.
338-
339-96:8
340-
341-A. Fu et al.
342-
343:soft objectives and semantic similarity within text modality is calculated as follows:
344->
345-e
346-𝑺 = 𝑭 𝑡 ⊗ 𝑭 𝑡 ∈ R𝐵×𝐵 ,
347-
348-(2)
349-
350:where ⊗ is matrix multiplication. However, vanilla semantic similarity as a soft objective will
351-cause positive sample pairs to dominate and cannot prominently modulate other sample pairs.
352:We fill the diagonal of the soft objective matrix generated by semantic similarity with 0 instead
353-of 1, which is the opposite direction of optimization for hard objectives. Soft objectives serve as
354-regulators for hard objectives. This regulator drives the model to appropriately optimize hard goals,
355:maintain small margins between positives and false negatives, relax strict one-to-one constraints,
356-and establish many-to-many relationships between different modalities. The revised soft objective
357-calculation method is as follows:
358- 
--
363-where diag extracts the diagonal elements of the matrix and creates a diagonal matrix. It is worth
364-noting that the model in the early training stage is unreliable because the semantic structure of
365-the embedding space is still being optimized, and it is unreasonable to directly use the semantic
366:similarity calculated by the text encoder. Therefore, in the beginning, we use sentence transformers
367:[20] to obtain semantic features and use them to estimate the semantic similarity of image–text
368:pairs in the early training stage. In this way, the semantic similarity can be gradually refined with
369-training, providing more reasonable soft objectives.
370-Next, we use the hard objective 𝑦ℎ𝑎𝑟𝑑 and the soft objective 𝑦𝑠𝑜 𝑓 𝑡 as labels, respectively, and
371:use KL divergence supervision to train the cross-modal similarity matrix 𝑆𝑡 2𝑣 , 𝑆 𝑣2𝑡 . The formula is
372-as follows:
373-𝐵
374-
--
430-
431-where 𝜃 is the parameter of the model. This means that the model not only maximizes the difference
432-between the image–text pair and other samples during the optimization process but also needs
433:to consider the similarity with other samples. By using soft objectives to adjust hard objectives,
434-the model maximizes the gap between positive and negative samples, while paying attention to
435:the similarity between different samples and minimizes the impact of false negatives and partially
436-similar negative samples on model training.
437-ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.
438-
--
440-
441-96:9
442-
443:Fig. 4. Illustration of the purpose of the CA module. The CA module can further reduce the margins of false
444-negatives and positives based on SR and enhance cross-modal intra-class compactness.
445-𝑡 2𝑣 can be defined by exchanging 𝐹 𝑣 and 𝐹 𝑡 in Equation (1), and L
446-Symmetrically, 𝑝𝑖,𝑗
--
459-
460-CA
461-
462:The SR module establishes a many-to-many relationship between different modalities. However,
463:the SR module has limited effect in shortening false negatives and anchor distance. As shown in
464-Figure 4, in order to further improve intra-class compactness and reduce the margin between false
465:negatives and anchors, we propose the CA module. The CA module aligns features of different
466:modalities and pseudo-labels of other modalities with each other, while gradually calibrating the
467:distribution of pseudo-labels to reduce the harmful effects of false negatives and further achieve
468-cross-modal alignment.
469-Traditional clustering methods may produce degenerate solutions, that is, all data points are
470-mapped to the same cluster. Unlike some other classification tasks that lack labels, in the text-based
--
472-of person identities can be roughly estimated, which means that we have prior information about
473-the number of categories, which is often ignored by previous work. So we can use this information
474-to add the constraint that the labels must induce an equipartition of the data. Inspired by [1],
475:the pseudo-label assignment problem is transformed into optimal transport, which avoids the
476-degenerate solution that may be produced by traditional algorithms by maximizing the mutual
477:information between labels and input data indices. Taking the visual branch as an example, consider
478-the classification problem optimized using cross-entropy loss:
479-𝐵
480-
--
491-(8)
492-
493-where {𝒇𝑖𝑣 , 𝒇 𝑖𝑡 } represents the features of the unlabeled image–text pair obtained by the model,
494:𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 is the pseudo-label. The basic idea of pseudo-labeling is to use an initial model to predict
495-unlabeled data and use these prediction results as “labels” to train the model. Here, we model the
496:calculation of pseudo-labels as an optimal transport problem. The pseudo-label is an unknown
497-ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.
498-
499-96:10
--
504-transport problem. Each epoch
505- solving the optimal
506-
507:needs to calculate pseudo-labels 𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 and use 𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 to optimize model parameters.
508-Then, each sample feature is input into the classifier for prediction, and the probability distribution
509-vector of each sample belonging to each category is obtained:
510-
--
627-where − log 𝑷 𝑣 is the cost matrix. The Sinkhorn-Knopp
628-[3] algorithm can be used to iteratively
629-
630:solve to obtain the pseudo-label 𝑞 𝑣 𝑦 | 𝒇 𝑖𝑣 . Similarly, the pseudo-label 𝑞𝑡 𝑦 | 𝒇 𝑖𝑡 of the text branch
631:can be obtained. Then we fix the pseudo-label 𝒒 𝑣 , 𝒒𝑡 and optimize the model parameters. Text
632-ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.
633-
634-SRCA for Weakly Supervised Text-Based Person Re-Identification
635-
636-96:11
637-
638:pseudo-labels to train image representation, image pseudo-labels to train text representation:
639-𝐵
640-
641-𝐾
--
664-
665-(15)
666-
667:The CA module gradually calibrates the pseudo-label distribution 𝑸 𝑣 , 𝑸 𝑡 during the model
668:training process. Assign features of different modalities to pseudo-labels of other modalities to
669-reduce intra-class gaps between different modalities. The CA loss is as follows:
670-LCA = 𝜆1 (LCA-v + LCA-t ),
671-
--
686-Lmlm = E (𝐼,𝑇 msk )∼D H 𝒚 msk , 𝒑 msk (𝐼,𝑇 msk ) ,
687-(17)
688-where 𝑦𝑚𝑠𝑘 is a one-hot vocabulary distribution where the ground-truth token has a probability of 1.
689:We optimize pseudo-label assignment via optimal transport and jointly optimize three cross-modal
690-alignment modules to train our SRCA framework, encouraging the network to learn discriminative
691-and generalized text–image representations. The overall training objective can be represented as
692-LSRCA = LSR + LCA + Lmlm .
--
712-identities, respectively.
713-Evaluation Metrics. We adopt the popular metric Rank-k as the main evaluation metric. Given
714-the specified query text to retrieve its corresponding person in the image library. Rank images by
715:text–image similarity. Rank-k is the probability of finding at least one matching person image in
716-the top k candidate lists when given a textual description as a query. We evaluate the accuracy of
717-ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.
718-
--
752-At the same time, due to the similarities between different person image–text pairs, it is unreasonable for these methods to completely rely on annotated hard labels to push away all negative
753-samples. However, few studies have been conducted on this. CMMT [36] first introduced weakly
754-supervised text-based person re-identification, considering more realistic problems. It does not
755:use labels that are difficult and expensive to obtain and uses text-IOU to measure the similarity
756:between different text pairs. However, this method strictly relies on the prior similarity of words
757:between texts and ignores the semantic similarity between different descriptions. In order to solve
758-this problem, this article proposes a SRCA framework. We propose a SR strategy, which introduces
759-a novel soft objective to relax the strict one-to-one constraints and model the local similarities
760:between different pairs. To further improve intra-class compactness, we propose a CA module that
761:models pseudo-label assignment as optimal transport, further achieves cross-modal alignment and
762-reduces the ambiguity of cross-modal matching.
763-In this section, we will show the comparison results with the state-of-the-art methods on three
764-public benchmark datasets to show the superiority and universality of our method.
--
1127-demonstrates the generality and effectiveness of our proposed method.
1128-4.3
1129-
1130:Ablation Study
1131-
1132-In this subsection, we analyze the effectiveness of each component in the SRCA framework and
1133:present some major ablation studies.
1134:Effectiveness of SR. To fully illustrate the effectiveness of our proposed SR module, we tested the
1135:performance of SR in Table 4. The SR module can relax the strict one-to-one constraints between
1136-different modalities and establish many-to-many relationships between different modalities. The
1137-efficacy of our proposed SR is revealed through the experimental results of Nos. 2, 3, 4, and 5.
1138:Compared with traditional contrast loss, the SR module can improve R1 by 0.5% and mAP by 0.2%.
1139:The above results clearly show that the SR module is beneficial for weakly supervised text-based
1140-person re-identification.
1141-In addition, we analyze the performance comparison of our proposed SR and other similar
1142:methods in Table 5. Table 5 shows that since the relationship between image and text pairs is not
1143-used to push the distance between anchors and negative samples, the performance of the model
1144-using only soft objectives is very poor. The performance of naively combining vanilla soft objective
1145-ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.
--
1148-
1149-96:15
1150-
1151:Table 4. Ablation Results of All Components Evaluated on
1152-CUHKPEDES Dataset
1153-
1154-No.
--
1208-
1209-L𝑖𝑡𝑐 represents the contrast loss used in CLIP.
1210-
1211:Table 5. Ablation Results of Different Objectives of SR on the
1212-CUHK-PEDES Dataset
1213-
1214-Method
--
1247-proposed effectively uses the soft objective to adjust the hard objective, establishing a many-tomany relationship between different modes while ensuring a large margin between the anchor and
1248-the negative sample. Our method achieved an R1 of 0.8% and a performance improvement of 0.85%,
1249-proving the superiority of our proposed SR method compared to other methods.
1250:Effectiveness of CA. To fully illustrate the effectiveness of our proposed CA module, we test the
1251:performance of CA in Table 4. The CA module can further improve the intra-class compactness
1252:on the basis of the SR module, shorten the distance between false negatives and anchors, and
1253-reduce the impact of false-negative samples on training. The efficacy of our proposed CA is revealed
1254-through the experimental results of Nos. 2, 3, 4, and 5. Compared with the method without CA, the
1255:CA module can improve R1 by 0.57% and mAP by 0.9%. In addition, we show the impact of different
1256:numbers of clusters on the CA module in Figure 5(a). It can be found that different numbers of
1257-clusters can improve the performance of the model. When the number of clusters is 2,000, the
1258-performance is the best.
1259:Analysis on Acquisition of Intra-Modal Semantic Similarity. Table 6 shows the results of different
1260:semantic similarity acquisition methods. The performance of Rank-1 is between 72.74% and 73.23%,
1261-ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.
1262-
1263-96:16
1264-
1265-A. Fu et al.
1266:Table 6. Ablation Results of Different Acquisition Methods of Semantic
1267-Similarity
1268-
1269-Method
--
1287-93.73
1288-65.60
1289-
1290:Fig. 6. Attention visualization for image and text encoders.
1291-
1292-and the performance of mAP is between 65.60% and 65.63%. The results show that using only
1293-the text encoder has the worst effect. Compared with the method using only the text encoder,
1294-our method can improve Rank-1 by 0.49%. Our method combines a text encoder with an extra
1295:language model and extracts semantic similarity from the external language model in the early
1296-stages of training. When the cross-modal model is trained for a period of time, we obtain it from
1297-the text encoder of the cross-modal model. Similarity. This can avoid the problem of insufficient
1298-semantic information obtained by the text encoder in the early stages of training. We show the
1299:results of changing the semantic extractor at different epochs in Figure 5(b) and achieved the best
1300-performance at epoch = 25.
1301-4.4
1302-
1303-Visualization
1304-
1305-Figure 6 shows the heatmap generated using the last layer of attention for text and image encoders.
1306:We can observe from the visualization results that the SRCA framework exhibits good correlation
1307-effects in text and images. It can be noted that our proposed SRCA can accurately capture the key
1308-ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.
1309-
--
1312-96:17
1313-
1314-correspondences in the image and text, such as “glasses,” “shirt with dots,” and “feather skirt” of the
1315:text are activated in the image. This is attributed to our proposed SR and CA modules. Our SRCA
1316-framework only requires simpler weak supervision conditional constraints and does not require
1317-any identity labels to achieve similar performance to fully supervised models.
1318-5
--
1320-Conclusion
1321-
1322-In this article, we proposed the SRCA framework to deal with weakly supervised text-based
1323:person re-identification. In the framework, we designed two modules, i.e., SR and CA. The proposed SR module was used to relax the strict one-to-one constraints and establish many-tomany relations. In the module, a novel soft objective was adopted to avoid the negative effects
1324-of false-negative samples and partially similar negative samples by constraining and adjusting
1325:the hard objective. The CA module was designed to better remove the negative impact of false
1326:negatives and improve intra-class compactness. In the module, the pseudo-labels are allocated
1327:through the theory of optimal transport, and the pseudo-labels are gradually calibrated during the
1328-training process to better aggregate features of the same category in different modalities. These
1329-modules cooperate and complement each other. Significant performance improvements on three
1330-widely used benchmark datasets demonstrate the superiority and effectiveness of our proposed
--
1376-‘22), 2724–2728. DOI: https://doi.org/10.1109/icassp43922.2022.9746846
1377-[17] Shuang Li, Tong Xiao, Hongsheng Li, Bolei Zhou, Dayu Yue, and Xiaogang Wang. 2017. Person search with natural
1378-language description. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 1970–1979.
1379:[18] Yash Patel, Lluis Gomez, Marçal Rusiñol, Dimosthenis Karatzas, and C. V. Jawahar. 2019. Self-supervised visual
1380-representations for cross-modal retrieval. In Proceedings of the 2019 on International Conference on Multimedia
1381-Retrieval, 182–186.
1382-[19] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda
1383:Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision.
1384-In Proceedings of the International Conference on Machine Learning. PMLR, 8748–8763.
1385-[20] Nils Reimers and Iryna Gurevych. 2020. Making monolingual sentence embeddings multilingual using knowledge
1386-distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. Association

codex
第二篇的证据段重点不只是“SR、CA 都涨点”，还包括作者把三种软目标方案摆出来，让自己的“去掉对角线的软目标”成为对错误问题定义的修正。这个证据形态对我们很有参考价值。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 14 -l 16 'Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
96:14

A. Fu et al.
Table 2. Comparison of State-of-the-Art Methods on ICFGPEDES

Method
Dual Path [37]
CMPM/C [35]
ViTAA [26]
SSAN [5]
IVT [23]
ISANet [33]
TP-TPS [24]
CFine [32]
Wei et al. [29]
IRRA [14]
PLIP [40]
SRCA

Ref
TOMM20
ECCV18
ECCV20
arXiv21
ECCVW22
arXiv22
arXiv23
arXiv22
BMVC21
CVPR23
arXiv23

Supervise
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Weakly Sup

R1
38.99
43.51
50.98
54.23
56.04
57.73
60.64
60.83
62.29
63.46
64.25
63.11

R5
59.44
65.44
68.79
72.63
73.6
75.42
75.97
76.55
77.15
80.25
80.88
79.87

R10
68.41
74.26
75.78
79.53
80.22
81.72
81.76
82.42
82.52
85.82
86.32
85.30

mAP
38.06
36.61

Table 3. Comparison of State-of-the-Art Methods on RSTPReid

Method
DSSL [38]
SSAN [5]
LBUL [28]
IVT [23]
ACSA [13]
CFine [32]
TP-TPS [24]
Wei et al. [29]
IRRA [14]
SRCA

Ref
MM21
arXiv21
MM20
ECCVW22
TMM22
arXiv22
arXiv23
BMVC21
CVPR23

Supervise
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Full Sup
Weakly Sup

R1
R5
R10 mAP
39.05 62.6 73.95
43.5 67.8 77.15
45.55 68.2 77.85
46.7
70
78.8
48.4 71.85 81.45
50.55 72.5 81.6
50.65 72.45 81.2
51.95 73.5 82.45
60.2 81.3 88.2 47.17
61.2 79.25 87.65 46.93

In summary, our SRCA method achieves excellent performance on three common datasets with
a more relaxed weak supervision setting, even compared with fully supervised methods. This
demonstrates the generality and effectiveness of our proposed method.
4.3

Ablation Study

In this subsection, we analyze the effectiveness of each component in the SRCA framework and
present some major ablation studies.
Effectiveness of SR. To fully illustrate the effectiveness of our proposed SR module, we tested the
performance of SR in Table 4. The SR module can relax the strict one-to-one constraints between
different modalities and establish many-to-many relationships between different modalities. The
efficacy of our proposed SR is revealed through the experimental results of Nos. 2, 3, 4, and 5.
Compared with traditional contrast loss, the SR module can improve R1 by 0.5% and mAP by 0.2%.
The above results clearly show that the SR module is beneficial for weakly supervised text-based
person re-identification.
In addition, we analyze the performance comparison of our proposed SR and other similar
methods in Table 5. Table 5 shows that since the relationship between image and text pairs is not
used to push the distance between anchors and negative samples, the performance of the model
using only soft objectives is very poor. The performance of naively combining vanilla soft objective
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

SRCA for Weakly Supervised Text-Based Person Re-Identification

96:15

Table 4. Ablation Results of All Components Evaluated on
CUHKPEDES Dataset

No.
1
2
3
4
5

L𝑖𝑡𝑐
Ø
Ø

L𝑚𝑙𝑚
Ø
Ø
Ø
Ø

SR

CA

Ø
Ø

Ø
Ø

R1
72.06
72.21
72.71
72.78
73.23

R5
88.48
88.22
89.41
89.41
89.47

R10
92.79
92.92
93.50
93.57
93.73

mAP
64.32
64.59
64.79
65.49
65.60

L𝑖𝑡𝑐 represents the contrast loss used in CLIP.

Table 5. Ablation Results of Different Objectives of SR on the
CUHK-PEDES Dataset

Method
Only soft objective
Only hard objective
Vanilla soft + hard
Ours

Rank-1
34.19
72.66
73.08
73.23

Rank-5
56.34
89.20
89.17
89.47

Rank-10
66.88
93.41
93.71
93.73

mAP
33.13
64.83
65.71
65.60

Fig. 5. Evaluation results for different parameter settings.

and hard objective is even worse than using only hard objective. Experiments show that the SR we
proposed effectively uses the soft objective to adjust the hard objective, establishing a many-tomany relationship between different modes while ensuring a large margin between the anchor and
the negative sample. Our method achieved an R1 of 0.8% and a performance improvement of 0.85%,
proving the superiority of our proposed SR method compared to other methods.
Effectiveness of CA. To fully illustrate the effectiveness of our proposed CA module, we test the
performance of CA in Table 4. The CA module can further improve the intra-class compactness
on the basis of the SR module, shorten the distance between false negatives and anchors, and
reduce the impact of false-negative samples on training. The efficacy of our proposed CA is revealed
through the experimental results of Nos. 2, 3, 4, and 5. Compared with the method without CA, the
CA module can improve R1 by 0.57% and mAP by 0.9%. In addition, we show the impact of different
numbers of clusters on the CA module in Figure 5(a). It can be found that different numbers of
clusters can improve the performance of the model. When the number of clusters is 2,000, the
performance is the best.
Analysis on Acquisition of Intra-Modal Semantic Similarity. Table 6 shows the results of different
semantic similarity acquisition methods. The performance of Rank-1 is between 72.74% and 73.23%,
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.

96:16

A. Fu et al.
Table 6. Ablation Results of Different Acquisition Methods of Semantic
Similarity

Method
Rank-1
Only text encoder
72.74
Only extra language model 73.03
Ours
73.23

Rank-5
89.36
89.43
89.47

Rank-10 mAP
93.57
65.60
93.88
65.63
93.73
65.60

Fig. 6. Attention visualization for image and text encoders.

and the performance of mAP is between 65.60% and 65.63%. The results show that using only
the text encoder has the worst effect. Compared with the method using only the text encoder,
our method can improve Rank-1 by 0.49%. Our method combines a text encoder with an extra
language model and extracts semantic similarity from the external language model in the early
stages of training. When the cross-modal model is trained for a period of time, we obtain it from
the text encoder of the cross-modal model. Similarity. This can avoid the problem of insufficient
semantic information obtained by the text encoder in the early stages of training. We show the
results of changing the semantic extractor at different epochs in Figure 5(b) and achieved the best
performance at epoch = 25.
4.4

Visualization

Figure 6 shows the heatmap generated using the last layer of attention for text and image encoders.
We can observe from the visualization results that the SRCA framework exhibits good correlation
effects in text and images. It can be noted that our proposed SRCA can accurately capture the key
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 3, Article 96. Publication date: March 2025.


codex
接下来是终身联邦这篇。我会特别看它是不是把“终身 ReID”重新定义成边缘端的空间域和时间流同时变化，而不是常规持续学习套壳。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 121ms:
 with
integrated knowledge from relevant edge clients. The base
parameters Bi will be dispatched to edge client i to help the
optimization of the local model.
C. Training Methodology

.

(5)

t′ =t−k

Our intuition is that if the historical tasks of edge client j have
a higher similarity with the new task on edge client i, then the

The distributed training procedures of FedSTIL are illustrated in Algorithm 1. The edge clients collect new drifted
data as new tasks, and input these tasks into their extraction
layers to generate feature vectors as the prototypes. Then, the

7

V. E XPERIMENT
In this section, we present the experimental settings and
evaluate the performances of our method by comparing with
the baseline methods. We also conduct ablation and visualization studies to validate the effectiveness of our method.
A. Experimental Setting
1) Dataset: To simulate the real-world scenarios of spatialtemporal person ReID [1], we apply the benchmark combined
from five popular person ReID datasets 1 . In this benchmark,
we split the images into five partitions, and each partition
has the person images of different camera-IDs. Then, we
shuffle these partitions into five distributed edge clients. As
shown in Fig. 6, we ensure the person images of different
edge clients are from different camera views, which aims
to align with the real-world camera distributions [6]. Then,
we group the images of each edge client into 6 sequential
tasks, and each edge sequential task uses 60% of the images
as training data and 40% as query data. The gallery images
are organized from other edges’ query images, where we can
ensure the camera-IDs of gallery images are different from
query images. Additionally, we simulate the data drift on
an edge client by changing person identities distribution. As
shown in Fig. 7(a), the data drift (e.g., T1 → T2) in prior
efforts can be measured as the class shift (e.g., Person-ID: 0800 → 800-1600). However, in the real-world scenario, the
1 https://github.com/MSNLAB/ST-ReID-Datasets

Edge 1
𝑻𝑻𝟏𝟏

→ 𝑻𝑻𝟐𝟐

…

Edge 2
→

𝑻𝑻𝟏𝟏

𝑻𝑻𝟑𝟑

→ 𝑻𝑻𝟐𝟐

Edge 5
𝑻𝑻𝟏𝟏

→ 𝑻𝑻𝟑𝟑

→ 𝑻𝑻𝟐𝟐

→

𝑻𝑻𝟑𝟑

Person Identity

10

10

Person Identity

2 3 4 5 6
00 000 000 000 000 000

2 3 4 5 6
00 000 000 000 000 000

Fig. 6. The visualization of some examples in our datasets. We maintain the
distribution gaps among edge clients with different camera views.

T1

T2

T3

T4

T5

0

0

edge clients upload the task features, which are the average
of the prototypes, to the parameter server. The parameter
server calculates the similarities with the task features of
the historical tasks from different edge clients. Based on
these similarities, the parameter server will integrate the taskrelevant knowledge as the base parameters for each edge
client, and the edge clients can continuously learn from the
new tasks based on the integrated knowledge from other edge
clients. Finally, when the training of the edge clients’ models
converges, the parameters of the adaptive layers of the edge
models will be uploaded to the parameter server to further
improve other spatial-temporal correlated edge clients.
For each task, the edge clients store some of the task
prototypes for future rehearsal, which can alleviate forgetting
of previously learnt knowledge. Specifically, we periodically
select a batch of prototypes from the current task and previously stored prototypes to train the parameters θc of the
adaptive layers for each task. The loss function can be either
cross-entropy loss or triplet loss.
The training samples on the edge clients may be insufficient
due to the limited data. If we directly use the limited data
to train models, the models could easily fall into overfitting.
Hence, we adopt parameter tying to tackle this issue by
regularizing or penalizing model weights [43], where all parameter changes are summarized as a penalty loss to get sparse
gradients for parameters optimization. By tying the parameters
of edge clients’ models, the models can converge with less
overfitting due to the minimal changes in prior knowledge.

T6

(a) Hard-divided Datasets (Prior)

T1

T2

T3

T4

T5

T6

(b) Soft-divided Datasets (Ours)

Fig. 7. Person identities distribution at different moments. We adopt softdivided datasets to enable the data distribution changes gradually.

person identities between adjacent moments always change
gradually rather than completely. Therefore, as shown in
Fig. 7(b), we enable the person identities to change gradually
in the form of normal distribution, where some old and new
identities can be all involved during one moment [37]
2) Backbone Network: The backbone network for person
ReID is modified from ResNet-18 [44]. Compared with regular
ResNet-18, the differences are as follows: 1) the last stride of
backbone is set to be 1 to enrich representation granularity;
2) a batch normalization layer is added after representation
for better convergence; 3) the bias of the classifier is removed
to avoid representation bias. To verify the compatibility of
our method with different backbones, we also evaluate the
performances with ResNet-50 and Swin-Transformer [45].
3) Training Setting: We adopt Adam optimizer with learning rate 10−3 and weight decaying rate 10−5 . The edge models
are trained for 5 epochs at each communication round between
parameter server and edge clients. To avoid the overfitting
issue, the training phases will stop if the loss stops decreasing
for 3 epochs. Considering the limited computing capacity of
the edge devices, only the last residual block and the classifier
will be updated during training, and the other layers are fixed
with pre-trained weights.
4) Performance Metric: We adopt the following performance metrics to evaluate federated lifelong person ReID,
which are in line with the previous works [17], [34].
(1) Accuracy: we adopt the mean average precision (mAP)
and cumulative match characteristic (CMC) [17] to measure the retrieval accuracy for each edge client. The test
accuracy (i.e., mAP, CMC) at communication round r is
defined as the average retrieval accuracy of all training
tasks on edge client c as follows,
Nc

A(r)
=
c

1 X
a(r; Dc(i) ),
Nc i=1

(7)

8

TABLE III
T HE PERFORMANCE COMPARISON OF DIFFERENT METHODS .
Method

mAP (%)

R1 (%)

R3 (%)

R5 (%)

Storage

S2C

C2S

Baseline

STL

54.39

51.62

60.27

63.70

354MB

NaN

NaN

Lifelong
(Regularization)

EWC [35]
MAS [36]

52.42
52.32

49.56
49.35

58.34
58.53

61.87
62.31

154MB
154MB

NaN
NaN

NaN
NaN

Lifelong
(Rehearsal)

LwF [46]
GwF [11]
iCaRL [42]
AKA [3]
PTKP [12]

52.19
52.67
54.52
55.02
55.22

49.13
49.87
51.78
52.23
52.53

58.02
58.49
60.33
60.75
60.90

61.65
61.87
63.63
64.10
64.15

106MB
287MB
696MB
178MB
395MB

NaN
NaN
NaN
NaN
NaN

NaN
NaN
NaN
NaN
NaN

Federated

FedSGD [38]
FedAvg [38]
FedProx [47]

60.53
62.47
62.48

57.68
60.26
60.26

66.42
67.20
67.30

69.07
69.92
69.97

59MB
59MB
106MB

2.8GB
2.8GB
2.8GB

2.8GB
2.8GB
2.8GB

FedCurv [31]
CFeD [32]
FedWeIT (a) [34]
FedWeIT (b) [34]

58.26
60.09
53.86
64.06

55.42
57.71
50.93
61.55

64.37
65.28
59.92
68.89

67.75
68.02
63.67
71.38

631MB
378MB
611MB
993MB

30.7GB
2.8GB
3.2GB
8.1GB

6.1GB
2.8GB
1.1GB
2.7GB

EWC [35] + FedAvg [38]
EWC [35] + FedProx [47]
iCaRL [42] + FedAvg [38]
iCaRL [42] + FedProx [47]

53.16
53.60
59.98
60.55

50.29
50.73
57.59
58.32

59.14
59.69
65.14
65.35

62.66
63.18
67.84
67.94

154MB
154MB
758MB
612MB

2.8GB
2.8GB
2.8GB
2.8GB

2.8GB
2.8GB
2.8GB
2.8GB

FedSTIL (ours)

68.16

66.05

72.03

74.05

825MB

2.8GB

2.8GB

Type

Federated
Lifelong

(i)

where a(r; Dc ) is the retrieval accuracy of the i-th task
(i)
Dc on edge client c at r-th communication round and
Nc is the number of training tasks for edge client c.
(2) Forgetting: we measure the forgetting for each client by
calculating the decreasing accuracy compared with the
maximum value of each task during training [34]. The
forgetting of client c can be calculated as follow,
Fc(r) =

NX
c −1
1
max a(t; Dc(i) ) − a(r; Dc(i) ), (8)
Nc − 1 i=1 t∈{1,...,r}

(i)

where a(r; Dc ) is the retrieval accuracy as mentioned
above. Note that the forgetting for the last training task
does not exist because there is no further task for training.

B. Performance Comparison
We compare the accuracy (mAP, Rank-1, Rank-3, Rank5), storage cost (model size + memory size), server-to-client
(S2C) and client-to-server (C2S) communication cost of our
method with the following baseline methods: 1) Single task
learning (STL) with multi-models for all tasks; 2) Lifelong
learning: EWC [35], MAS [36], iCaRL [42], LwF [46],
GwF [11], AKA [3], and PTKP [12]; 3) Federated learning:
FedSGD [38], FedAvg [38], and FedProx [47]; 4) Federated
lifelong learning: CFeD [32], FedCurv [31], FedWeIT [34],
and some combinations of federated and lifelong methods.
In the real-life deployments of person ReID, task IDs are
unavailable both in the training and inference stages. FedWeIT
[34] requires task IDs for training and inference, and we
assume that task IDs are given for FedWeIT [34] to obtain
its performance. Moreover, we validate FedWeIT [34] under
different settings to balance its accuracy and communication
cost. FedWeIT (a) is set with l1 = 1.0×10−4 , l2 = 1.0×10−6 .
FedWeIT (b) is set with l1 = 5.0 × 10−6 , l2 = 1.0 × 10−3 .

Table III illustrates the performance comparison of different
methods. Our FedSTIL can outperform other baseline methods
in accuracy and communication cost. The most competitive
baseline method, FedWeIT [34], is about 4% lower than
our method on mAP, and the S2C communication cost is
2.89 times higher compared with our method. The higher
accuracy and lower communication cost make our method
more applicable in real-world person ReID scenarios. The
detailed analysis for each performance metric are as follows.
1) Comparison of Accuracy: We evaluate the average accuracy of different edge clients under different federated lifelong
learning methods (i.e., FedSTIL, FedWeIT [34], FedCurv [31],
and CFeD [32]) in Fig 8. Compared with the baseline methods,
our method can achieve higher accuracy over 60 rounds of
training. Besides, with continuous training on the new tasks,
the accuracy can increase stably with less fluctuation, which
also indicates the robustness of our methods. This improvement is largely owing to the effective knowledge sharing and
the alleviation of the forgetting of the learnt knowledge during
federated lifelong learning. We then analyze the impact of
knowledge sharing and forgetting on federated lifelong person
ReID.
Knowledge Sharing. Our FedSTIL can achieve higher accuracy because it can effectively exchange knowledge among
edge clients. In general, most federated learning-based methods (i.e., FedSGD [38], FedAvg [38], FedProx [47], CFeD
[32], FedCurv [31], FedWeIT [34], and our FedSTIL) can
achieve higher accuracy than other local-based training methods (i.e., STL, EWC [35], MAS [36], iCaRL [42], LwF [46],
GwF [11], AKA [3], and PTKP [12]). These results indicate
that exchanging on-edge knowledge can significantly improve
the overall accuracy, because the local data of an edge client
is limited for learning, such as insufficient data or limited
camera views. Thus, sharing the knowledge across edge clients


codex
这篇前 8 页已经覆盖了问题定义、系统设计、训练算法和主要结果。刚才输出中间有截断，我会单独重读第 4 到第 6 页，补齐自适应层、原型回放和空间时间知识聚合的具体逻辑。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 4 -l 6 'Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
4

Adaptive Layer Architecture
Edge Framework

𝒍−𝟏

Prototypes

…

𝝁𝒍

𝒍+𝟏
…

𝑨𝑫𝑫
𝑴𝑼𝑳

Extraction Layer

Adaptive Layer

Sampling

Raw Data

𝒍

𝝁 𝒍−𝟏

𝑨𝒄

Rehearsal

Local
Storage

𝝁𝒍

𝝁 𝒍−𝟏
𝑩𝒄

𝜶𝒄

Fig. 3. The architecture of the adaptive layers. The on-edge models continuously learn knowledge from the forthcoming tasks and meanwhile maintain the
knowledge from prior tasks. The adaptive layers adaptively balance the tradeoff between local knowledge and global spatial-temporal knowledge.

TABLE II
M AIN N OTATIONS .
(t)

Dc
(t)
Pc
(t)
Pc

the drifted training data for edge client c at round t
(t)
the compressed prototypes for raw data Dc
(t)
the task feature averaged on prototypes Pc

Gc
Fc
θc

the extraction layers for edge client c
the adaptive layers for edge client c
the parameters of adaptive layers for edge client c

Ac
Bc
αc

the adaptive parameters with local knowledge
the base parameters with global spatial-temporal knowledge
the attention to selectively capture the task-specific knowledge

Π(·)

the similarity function to measure task features

(t,t′ )

Sij

(t)

the similarity between task feature P i

(t′ )

and P j

Wij

(t)

the knowledge relevance between i- and j-th edge client

(r)

the retrieval accuracy of edge client c at the r-th round
the forgetting of edge client c at the r-th round

Ac
(r)
Fc

for personalized lifelong learning. Considering the limited
computing capacity of the edge clients [39]–[41], our framework adopts the last several layers of the backbone as adaptive
layers for training, and the rest layers are extraction layers.
The learning procedures of FedSTIL are as follows. The
edge clients collect the new drifted data as incremental tasks
and extract the general representation from the drifted data
as prototypes of raw data. The edge clients learn incremental
knowledge from these prototypes of drift data, and then upload
learnt knowledge to the parameter server. The parameter server
aggregates these incremental knowledge based on the spatialtemporal correlation of neighbors’ task characteristics, and
delivers the task-relevant knowledge to the edge clients. The
edge clients adaptively utilize dispatched knowledge and previously learnt knowledge to optimize models for new scenarios
continuously. Meanwhile, some general prototypes of raw data
will be stored in each edge client for future rehearsal to
alleviate the forgetting of previously learnt knowledge.

A. Problem Definition
We assume that the distributed person ReID system has C
edge clients. Each edge client c continuously learns from its
(t)
(t)
arriving task stream Dc , where Dc denotes the drift data
arriving at the t-th round on the edge client c. We assume
(t)
that Dc is only available for edge client c on the t-th round,
and the previous training data are no longer accessible due to
the limited storage space of the edge clients. Our goal is to
determine how to continuously learn from both on-edge task
streams and across-edge knowledge to improve performances
without sharing sensitive raw data among edge clients.
B. System Overview
As illustrated in Fig. 2, our federated lifelong framework,
FedSTIL, consists of one central parameter server and several
distributed edge clients. The edge clients continuously retrain
local models with newly obtained drift data, and the parameter
server aggregates spatial-temporal knowledge from the edge
clients’ local models for knowledge sharing. The network
backbones are alternative, such as MobileNet, ResNet, and
ViT. We divide the backbone of each edge model into two
parts: 1) the extraction layers initialized with global pre-trained
weights to extract task prototypes, and 2) the adaptive layers

IV. L EARNING M ETHODOLOGY
In this section, we first present the framework of our
FedSTIL and then illustrate the training methodology.
A. Lifelong Learning on Distributed Edges
We enable the distributed edges to perform on-edge lifelong
learning for every new scenario. Considering the limited edge
computing capacity, we only enable the last several layers,
which we refer to as adaptive layers, for lifelong learning.
The rest layers are pre-trained and frozen as the extraction
layers. In the real-world deployment, the drifted data from new
scenarios are coming as a task stream, and the edge client will
perform on-edge lifelong learning by the following procedures.
Task Prototype Extraction. The drift data arrives as task
(t)
stream Dc , where t = 0, 1, 2, .... Considering the limited
resources of the edge clients, we utilize the extraction layers
(t)
to encode the raw tasks Dc to compressed prototypes to
represent the original task for training and inference. Specifically, the drift data of edge client c at the t-th round can
(t)
(t)
(t)
(t)
be represented as Dc = {(Xi , Yi )}, where Xi is the i(t)
(t)
th training image of Dc and Yi is the corresponding label.

5

Parameter Server

Task Feature

Task
Distance

Time Axis

Time Axis

Edge 3 Edge 1 Edge 2

…

× Forget

Task Feature

× Forget

Ratio

Distributed Space

𝑾𝑾𝟑𝟑𝟑𝟑

𝑾𝑾𝟐𝟐𝟐𝟐 𝑾𝑾𝟐𝟐𝟐𝟐 𝑾𝑾𝟐𝟐𝟐𝟐

⋯

⋮

⋱

⋮

⋮

Edge N 𝑾𝑾𝑵𝑵𝑵𝑵 𝑾𝑾𝑵𝑵𝑵𝑵 𝑾𝑾𝑵𝑵𝑵𝑵 ⋯

Ratio

Task Distance Space

Edge 3 𝑾𝑾𝟑𝟑𝟑𝟑 𝑾𝑾𝟑𝟑𝟑𝟑 𝑾𝑾𝟑𝟑𝟑𝟑 ⋯

Edge 2

𝑵𝑵

𝑩𝑩𝟏𝟏 = � 𝑾𝑾𝟏𝟏𝟏𝟏 𝜽𝜽𝒊𝒊
𝒊𝒊

𝑵𝑵

𝑾𝑾𝟐𝟐𝟐𝟐

𝑩𝑩𝟐𝟐 = � 𝑾𝑾𝟐𝟐𝒊𝒊 𝜽𝜽𝒊𝒊
…

× Forget

𝑾𝑾𝟏𝟏𝟏𝟏

…

Task Feature

Softmax

Ratio

Edge 1 𝑾𝑾𝟏𝟏𝟏𝟏 𝑾𝑾𝟏𝟏𝟏𝟏 𝑾𝑾𝟏𝟏𝟏𝟏 ⋯

⋮

𝑵𝑵

𝑾𝑾𝑵𝑵𝑵𝑵

𝑩𝑩𝑵𝑵 = � 𝑾𝑾𝑵𝑵𝒊𝒊 𝜽𝜽𝒊𝒊
𝒊𝒊

Knowledge Relevance

Task Similarity Matrix

𝒊𝒊

Model Aggregation

Fig. 4. The framework of the spatial-temporal knowledge integration on the parameter server. The parameter server can automatically measure the spatialtemporal correlations for edge clients based on their task features. The task-specific knowledge is organized for edge clients for continuously learning.

(t)

(t)

(t)

We input each raw data (Xi , Yi ) ∈ Dc into the extraction
layers Gc to extract prototype as
(t)

(t)

Pc(t) = {(Gc (Xi ), Yi )},

(1)

(t)

Backtrack
Prototypes

Prototypes

Prototypes
Prototypes

where Gc is the extraction layers, and Pc is the extracted
(t)
prototype set for the raw data set Dc . Hence, the raw task
(t)
(t)
Dc can be represented as the prototypes Pc after being
processed by the extraction layers. Compared with raw data,
prototypes are more generalized to represent different tasks
and also smaller to store. Moreover, the prototypes with
compressed semantics can also reduce data transmission costs
and avoid potential privacy leakage.
Adaptive Lifelong Learning. The training and inference
tasks on different edge clients have different characteristics due
to the different camera environment. If all of edge clients adopt
one single unified model for lifelong training or inference, it is
hard to achieve the optimal performance for each edge client.
To address the problem of heterogeneous tasks on different
edge clients, each edge client needs to have its personalized
model, which can continuously learn from both local knowledge and other edge neighbors’ knowledge. Inspired by [34],
we present the adaptive layers that leverage global and local
knowledge for personalized model training. As illustrated in
Fig. 3, the parameters θc of the adaptive layers for the model of
edge client c consist of three parts: 1) the adaptive parameters
Ac with knowledge learnt from local incremental tasks, 2)
the base parameters Bc with the spatial-temporal knowledge
from other correlated edge neighbors, and 3) the attention
parameters αc to capture the task-specific knowledge from the
base parameters Bc by attention mechanism,

will disturb the prior learnt knowledge, and thus the recognition accuracy for the previous domains will decrease during
continuous learning [11]. To alleviate catastrophic forgetting of
previously learnt knowledge, we design a prototype rehearsal
method which utilizes the stored general prototypes of the
former tasks in the local storage for knowledge rehearsal. As
shown in Fig. 5, we periodically sample some representative
prototypes of the newly arriving incremental task with the
nearest-mean-of-exemplars strategy [42], and store them in the
local storage of the edge clients. Specifically, when a training
task arrives, we first input the prototypes of the task into the
adaptive layers. Then, we calculate the mean center of the
outputs of the adaptive layers for each person identity. We
will store some prototypes whose outputs are closest to the
mean centers of different person identities. Those prototypes
are informative to maintain the representation of previously
learnt knowledge. During the training phases for new tasks,
we periodically sample some stored prototypes for training to
mitigate the forgetting of the previously learnt knowledge.
Compared with other rehearsal-based lifelong learning
methods (e.g., iCaRL [42], and GwFReID [11]), our strategy is
more friendly to edge devices with limited storage resources.
That is because prototypes are smaller than raw images for
storage. Meanwhile, rehearsing the prototypes to alleviate
catastrophic forgetting takes smaller computation overhead.

Prototypes

Save

Local
Storage

Backtrack

θc = Bc ⊙ αc + Ac .

(2)

Adaptive Layer
Backtrack

As shown in Eq. (2), the parameters θc of the adaptive
layers are combined by the following terms. The first term
(Bc ⊙ αc ) enables learnable attention αc for the spatialtemporal knowledge Bc to capture the personalized specific
knowledge for local tasks. The second term (Ac ) is to learn
incremental knowledge from the drifted data of local tasks. We
adaptively combine knowledge from two different perspectives
into the adaptive layers to continuously optimize models for
the newly arriving tasks.
Prototype Rehearsal to Alleviate Forgetting. The local
models of the edge clients are continuously updated with
newly arriving tasks. However, the new incremental knowledge

Prototypes

Fig. 5.
The data flow for sampling prototypes into local storage. We
dynamically store some identities’ prototypes for future rehearsal which are
near the corresponding mean center.

B. Spatial-Temporal Knowledge Integration on Server
The distributed person ReID system consists of many edge
clients. However, not all knowledge learnt from edge clients
are informative and relevant to others. It may even hinder the
training of an edge client if the irrelevant knowledge from
others are transferred to the edge client [34]. To extract the

6

task-relevant knowledge from relevant neighbors, we design
a spatial-temporal knowledge integration method based on
the characteristics of the tasks from different locations and
moments. The framework of the spatial-temporal knowledge
integration on the parameter server is illustrated in Fig. 4.
Next, we elaborate the procedure of spatial-temporal knowledge integration on the parameter server.
Task Similarity across Spatial-Temporal Dimension. The
(t)
raw data of training task Dc is encoded into task prototypes
(t)
Pc , which consist of the compressed semantics of the raw
data. To identify the spatial-temporal correlations of the edge
tasks, however, directly centralizing and analyzing the raw data
(t)
(t)
Dc or task prototypes Pc is consuming for communication,
and exists the potential risk of privacy leakage. Instead,
the parameter server only collects the average value of the
(t)
prototypes as the task feature P c for different edge tasks,
X
1
(t)
P c = (t)
p,
(3)
|Pc |
(t)

Algorithm 1: Training Procedures of FedSTIL
(0)

Require: Pre-trained weights θc
(t)
Require: Task streams {Dc }∞
t=1
Require: Shared layers {Gc }C
c=1
Require: Adaptive layers {Fc }C
c=1
(0)
(0) C
1 Initialize weights {αc ; Ac }c=1
2 for round t = 1, 2, 3, ... do
3
for client c ∈ C do
(t)
4
Collect incremental task Dc on edge client c
/* Task Prototype Extraction
5

in the task, and p is the vector of each prototype.
To evaluate the relevance of the tasks across different edge
(t,t′ )
clients and moments, we calculate the task similarity Sij
between the tasks of edge client i at the t-th round and edge
client j at the t′ -th round as
(t,t′ )

Sij

(t′ )

(t)

= Π(P i , P j ),

(4)

where Π(·) is the similarity function to measure between task
(t)
(t′ )
feature P i and task feature P j . In this work, we adopt
Kullback-Leibler Divergence as the similarity function because
it can effectively measure the information difference between
different distributions.
Knowledge Relevance for Distributed Edge Clients. To
organize the task-relevant knowledge for sharing, we calculate
the knowledge relevance for distributed edge clients based on
their task similarity across the spatial-temporal dimension.
We measure the knowledge relevance by the similarities
of all historical tasks for different edge clients. However, the
tasks which arrive recently have a more significant impact on
the model knowledge, while those coming earlier have less
influence because the knowledge learnt from earlier tasks may
be forgotten or updated over time. Hence, we introduce the
forgetting ratio λf (0 < λf < 1) for historical task when
calculating the knowledge relevance. Formally, to calculate
(t)
the knowledge relevance Wij between edge client i and edge
client j at the t-th round, we first evaluate the task similarity
between the current task on edge client i and the past k tasks
on edge
client j. Then, we accumulate the task similarities
(t,t′ )
Sij
with the forgetting ratio as,
(t)

Wij =

t
X

′

(t,t′ )

λft−t · Sij

*/
(t)

and P c

/* Spatial-Temporal Integration
6

7

8

p∈Pc

(t)
where P c is the task feature for the training task on edge
(t)
client c at the t-th round, |Pc | is the number of prototypes

(t)

Calculate task prototypes Pc

(t,t′ )
Calculate task similarity {Sci }i∈C/c by

Eq.(4)
(t)
Integrate knowledge relevance {Wci }i∈C/c by
Eq.(5)
Aggregate spatial-temporal knowledge into Bc
by Eq.(6)
/* Adaptive Lifelong Learning

9

12

Sample training data X from current and
stored prototypes
for epoch = 1, 2, ... do
Update θc ← θc − η∇L(θc ; X)

13

Upload parameters θc to the parameter server

11

*/

Set parameters of local adaptive layers
θc ← Bc ⊙ αc + Ac
/* Prototypes Rehearsal

10

*/

*/

model knowledge of edge client j has a greater relevance with
the new task on edge client i.
Personalized Model Aggregation. The parameter server
needs to integrate the task-relevant knowledge for each distributed edge client to improve the recognition accuracy.
We adopt the parameter-sharing approach for spatial-temporal
knowledge transfer, and the parameters of models are the
carrier of spatial-temporal knowledge for transferring. The
parameter server will aggregate the parameters of relevant edge
(t)
models with corresponding knowledge relevance weight Wij .
Then the aggregated parameters will be dispatched to edge
client i as the base parameters Bi for the training,
X
(t)
Bi =
Wij · θj ,
(6)
j∈C/i

where θj are the model parameters of edge client j, and Bi
are the task-specific base parameters for edge client i with
integrated knowledge from relevant edge clients. The base
parameters Bi will be dispatched to edge client i to help the
optimization of the local model.
C. Training Methodology

.

(5)

t′ =t−k

Our intuition is that if the historical tasks of edge client j have
a higher similarity with the new task on edge client i, then the

The distributed training procedures of FedSTIL are illustrated in Algorithm 1. The edge clients collect new drifted
data as new tasks, and input these tasks into their extraction
layers to generate feature vectors as the prototypes. Then, the


codex
这篇机制和问题定义绑定得比较紧：如果边缘端既有空间异构又有时间漂移，那么“谁的知识该传给谁”就不能用平均聚合，而要用任务相似度和时间衰减做个性化聚合。现在补读消融，看它有没有证明空间和时间两个维度都必要。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf' - | rg -n -C 4 'Ablation|ablation|Effectiveness|visual|Visualization|spatial|temporal|Spatial|Temporal|forgetting|TABLE IV|TABLE V|Fig. 9|Fig. 10|Fig. 11|Storage|communication|knowledge'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 148ms:
952-parameter server will aggregate the parameters of relevant edge
953-(t)
954:models with corresponding knowledge relevance weight Wij .
955-Then the aggregated parameters will be dispatched to edge
956-client i as the base parameters Bi for the training,
957-X
958-(t)
--
962-j∈C/i
963-
964-where θj are the model parameters of edge client j, and Bi
965-are the task-specific base parameters for edge client i with
966:integrated knowledge from relevant edge clients. The base
967-parameters Bi will be dispatched to edge client i to help the
968-optimization of the local model.
969-C. Training Methodology
970-
--
985-
986-V. E XPERIMENT
987-In this section, we present the experimental settings and
988-evaluate the performances of our method by comparing with
989:the baseline methods. We also conduct ablation and visualization studies to validate the effectiveness of our method.
990-A. Experimental Setting
991:1) Dataset: To simulate the real-world scenarios of spatialtemporal person ReID [1], we apply the benchmark combined
992-from five popular person ReID datasets 1 . In this benchmark,
993-we split the images into five partitions, and each partition
994-has the person images of different camera-IDs. Then, we
995-shuffle these partitions into five distributed edge clients. As
--
1047-
1048-2 3 4 5 6
1049-00 000 000 000 000 000
1050-
1051:Fig. 6. The visualization of some examples in our datasets. We maintain the
1052-distribution gaps among edge clients with different camera views.
1053-
1054-T1
1055-
--
1068-edge clients upload the task features, which are the average
1069-of the prototypes, to the parameter server. The parameter
1070-server calculates the similarities with the task features of
1071-the historical tasks from different edge clients. Based on
1072:these similarities, the parameter server will integrate the taskrelevant knowledge as the base parameters for each edge
1073-client, and the edge clients can continuously learn from the
1074:new tasks based on the integrated knowledge from other edge
1075-clients. Finally, when the training of the edge clients’ models
1076-converges, the parameters of the adaptive layers of the edge
1077-models will be uploaded to the parameter server to further
1078:improve other spatial-temporal correlated edge clients.
1079-For each task, the edge clients store some of the task
1080:prototypes for future rehearsal, which can alleviate forgetting
1081:of previously learnt knowledge. Specifically, we periodically
1082-select a batch of prototypes from the current task and previously stored prototypes to train the parameters θc of the
1083-adaptive layers for each task. The loss function can be either
1084-cross-entropy loss or triplet loss.
1085-The training samples on the edge clients may be insufficient
--
1088-Hence, we adopt parameter tying to tackle this issue by
1089-regularizing or penalizing model weights [43], where all parameter changes are summarized as a penalty loss to get sparse
1090-gradients for parameters optimization. By tying the parameters
1091-of edge clients’ models, the models can converge with less
1092:overfitting due to the minimal changes in prior knowledge.
1093-
1094-T6
1095-
1096-(a) Hard-divided Datasets (Prior)
--
1125-to avoid representation bias. To verify the compatibility of
1126-our method with different backbones, we also evaluate the
1127-performances with ResNet-50 and Swin-Transformer [45].
1128-3) Training Setting: We adopt Adam optimizer with learning rate 10−3 and weight decaying rate 10−5 . The edge models
1129:are trained for 5 epochs at each communication round between
1130-parameter server and edge clients. To avoid the overfitting
1131-issue, the training phases will stop if the loss stops decreasing
1132-for 3 epochs. Considering the limited computing capacity of
1133-the edge devices, only the last residual block and the classifier
--
1136-4) Performance Metric: We adopt the following performance metrics to evaluate federated lifelong person ReID,
1137-which are in line with the previous works [17], [34].
1138-(1) Accuracy: we adopt the mean average precision (mAP)
1139-and cumulative match characteristic (CMC) [17] to measure the retrieval accuracy for each edge client. The test
1140:accuracy (i.e., mAP, CMC) at communication round r is
1141-defined as the average retrieval accuracy of all training
1142-tasks on edge client c as follows,
1143-Nc
1144-
--
1165-R3 (%)
1166-
1167-R5 (%)
1168-
1169:Storage
1170-
1171-S2C
1172-
1173-C2S
--
1406-(i)
1407-
1408-where a(r; Dc ) is the retrieval accuracy of the i-th task
1409-(i)
1410:Dc on edge client c at r-th communication round and
1411-Nc is the number of training tasks for edge client c.
1412:(2) Forgetting: we measure the forgetting for each client by
1413-calculating the decreasing accuracy compared with the
1414-maximum value of each task during training [34]. The
1415:forgetting of client c can be calculated as follow,
1416-Fc(r) =
1417-
1418-NX
1419-c −1
--
1423-
1424-(i)
1425-
1426-where a(r; Dc ) is the retrieval accuracy as mentioned
1427:above. Note that the forgetting for the last training task
1428-does not exist because there is no further task for training.
1429-
1430-B. Performance Comparison
1431-We compare the accuracy (mAP, Rank-1, Rank-3, Rank5), storage cost (model size + memory size), server-to-client
1432:(S2C) and client-to-server (C2S) communication cost of our
1433-method with the following baseline methods: 1) Single task
1434-learning (STL) with multi-models for all tasks; 2) Lifelong
1435-learning: EWC [35], MAS [36], iCaRL [42], LwF [46],
1436-GwF [11], AKA [3], and PTKP [12]; 3) Federated learning:
--
1441-unavailable both in the training and inference stages. FedWeIT
1442-[34] requires task IDs for training and inference, and we
1443-assume that task IDs are given for FedWeIT [34] to obtain
1444-its performance. Moreover, we validate FedWeIT [34] under
1445:different settings to balance its accuracy and communication
1446-cost. FedWeIT (a) is set with l1 = 1.0×10−4 , l2 = 1.0×10−6 .
1447-FedWeIT (b) is set with l1 = 5.0 × 10−6 , l2 = 1.0 × 10−3 .
1448-
1449-Table III illustrates the performance comparison of different
1450-methods. Our FedSTIL can outperform other baseline methods
1451:in accuracy and communication cost. The most competitive
1452-baseline method, FedWeIT [34], is about 4% lower than
1453:our method on mAP, and the S2C communication cost is
1454-2.89 times higher compared with our method. The higher
1455:accuracy and lower communication cost make our method
1456-more applicable in real-world person ReID scenarios. The
1457-detailed analysis for each performance metric are as follows.
1458-1) Comparison of Accuracy: We evaluate the average accuracy of different edge clients under different federated lifelong
1459-learning methods (i.e., FedSTIL, FedWeIT [34], FedCurv [31],
1460-and CFeD [32]) in Fig 8. Compared with the baseline methods,
1461-our method can achieve higher accuracy over 60 rounds of
1462-training. Besides, with continuous training on the new tasks,
1463-the accuracy can increase stably with less fluctuation, which
1464:also indicates the robustness of our methods. This improvement is largely owing to the effective knowledge sharing and
1465:the alleviation of the forgetting of the learnt knowledge during
1466-federated lifelong learning. We then analyze the impact of
1467:knowledge sharing and forgetting on federated lifelong person
1468-ReID.
1469:Knowledge Sharing. Our FedSTIL can achieve higher accuracy because it can effectively exchange knowledge among
1470-edge clients. In general, most federated learning-based methods (i.e., FedSGD [38], FedAvg [38], FedProx [47], CFeD
1471-[32], FedCurv [31], FedWeIT [34], and our FedSTIL) can
1472-achieve higher accuracy than other local-based training methods (i.e., STL, EWC [35], MAS [36], iCaRL [42], LwF [46],
1473-GwF [11], AKA [3], and PTKP [12]). These results indicate
1474:that exchanging on-edge knowledge can significantly improve
1475-the overall accuracy, because the local data of an edge client
1476-is limited for learning, such as insufficient data or limited
1477:camera views. Thus, sharing the knowledge across edge clients
1478-
1479-9
1480-
1481-(a) mAP Accuracy
1482-
1483-(b) Rank-1 Accuracy
1484:Fig. 8. The comparison of average mAP and Rank-1 accuracy of 6 local tasks on 5 edge clients during 60 communication rounds. Our FedSTIL achieves
1485-higher and stable accuracy with less fluctuation during the federated lifelong learning.
1486-
1487:Fig. 9. The comparison of forgetting for trained tasks over different communication rounds. Our FedSTIL can maintain higher accuracy for ever-learnt tasks
1488:with less forgetting when training for new tasks.
1489-
1490-Fine-tuning
1491-AKA
1492-EWC
--
1501-10
1502-better
1503-
1504-Alleviation of Forgetting. Our FedSTIL can maintain the
1505:accuracy with less forgetting of the previous knowledge during
1506-the continuous training on the subsequent tasks. We can
1507-observe in Fig. 8 that both the Rank-1 and mAP accuracy
1508-of federated learning methods (i.e., FedAvg [38] and FedProx
1509-[47]) will not increase after the 40th round. As shown in Fig.
1510:9, this is because the knowledge learnt from the past tasks
1511-is gradually forgotten during the learning of the new tasks.
1512-In contrast, federated lifelong methods (i.e., FedCurv [31],
1513:FedWeIT [34], and FedSTIL) can alleviate forgetting on the
1514-past tasks, which help to achieve higher accuracy. To future
1515-investigate the effectiveness of our methods, we compare
1516:catastrophic forgetting with the fine-tuning baseline (without
1517-lifelong learning) and some other lifelong learning methods on
1518-
1519-25
1520-
--
1523-can make up for the limitations of local data. Moreover, we
1524-can observe that our FedSTIL achieves the highest accuracy
1525-compared with other federated learning-based methods (i.e.,
1526-FedSGD [38], FedAvg [38], FedProx [47], CFeD [32], FedCurv [31], and FedWeIT [34]). The improvement mainly owes
1527:to the task-relevant knowledge sharing among edge clients,
1528-which alleviates the interference incurred by the irrelevant
1529:knowledge transferring among the edge clients’ models.
1530-
1531-5
1532-
1533-mAP-F
--
1535-R1-F
1536-
1537-R3-F
1538-
1539:Fig. 10. The comparison of the catastrophic forgetting under the MSMT17
1540:dataset. Our FedSTIL alleviates the forgetting under lifelong learning for new
1541-scenarios.
1542-
1543-MSMT17 dataset [5]. Also, we divide the MSMT17 dataset
1544-into 6 task sequences, and the model backbone is ResNet50.
1545:Fig. 10 shows us the forgetting after finishing lifelong learning
1546-on the MSMT17 dataset. We can observe that our method
1547:FedSTIL can still effectively alleviate the forgetting compared
1548-with other lifelong person ReID baselines.
1549-
1550-Better
1551-
--
1609-
1610-64.0
1611-CFeD
1612-
1613:Fig. 11. The comparison of mAP accuracy over communication cost. Our
1614:FedSTIL achieves the highest retrieval accuracy with less communication cost.
1615-
1616-2) Comparison of Communication Cost: We illustrate the
1617:mAP accuracy over total communication costs (client-toserver and server-to-client cost) in Fig. 11. FedSTIL is
1618:communication-efficient and can achieve higher accuracy compared with the other federated lifelong learning methods. FedWeIT [34] can adjust the communication cost by l1 pruning for
1619-the transferred parameters, however, the accuracy of FedWeIT
1620-(a) is still much lower than our method under comparable
1621:communication cost. FedCurv [31] has lower accuracy with
1622:higher communication cost. That is because FedCurv [31]
1623-needs to transfer excessive additional information to alleviate
1624:catastrophic forgetting. Our FedSTIL is more communicationefficient because it does not need to exchange additional
1625-parameters among edge clients except model weights. This
1626:improvement is largely because FedSTIL only requests taskspecific parameters, which have been aggregated by the parameter server based on the relevant spatial-temporal knowledge.
1627-Moreover, our FedSTIL can achieve higher accuracy compared
1628-with the other baseline methods by using the cheaper model
1629-ResNet18, which has lower computation cost and storage
1630-consumption. Therefore, FedSTIL is more applicable for
1631-deployment in real-life person ReID scenarios by reducing
1632:communication and computational costs.
1633:TABLE IV
1634-T HE ABLATION STUDY OF THE INFLUENCES OF EDGE AND SERVER
1635-MODULES ON THE ACCURACY OF OUR METHOD .
1636-Variant
1637-
--
1658-58.18 (-7.87)
1659-60.33 (-5.72)
1660-
1661-Server Modules
1662:w/o Spatial Model Aggregation
1663:w/o Temporal Model Aggregation
1664:w/o Spatial-Temporal Integration
1665-
1666-58.72 (-9.44)
1667-61.04 (-7.12)
1668-54.26 (-13.90)
--
1670-57.03 (-9.02)
1671-59.53 (-6.52)
1672-51.51 (-14.54)
1673-
1674:C. Ablation Study
1675:We conduct the ablation study by removing some key
1676-components of our method to analyze their influences on
1677-the performances, and the results are shown in Table IV.
1678-
1679-Specifically, we first remove the edge modules to verify our
--
1682-kernels and MLP blocks. The ”w/o Prototypes Rehearsal”
1683-setting eliminates the rehearsal of historical prototypes in
1684-lifelong learning. The ”w/o Parameter Tying” setting removes
1685-the tying regularization terms in the loss functions. Moreover,
1686:we also verify the distributed knowledge sharing by removing
1687:the parameter server modules. Specifically, the ”w/o Spatial
1688-Model Aggregation” setting removes the model aggregation
1689:on the spatial-relevant edges. The ”w/o Temporal Model
1690-Aggregation” setting omits the historical task features when
1691-calculating task distances in Eq. (5). The ”w/o S-T Integration”
1692:setting removes both spatial- and temporal- knowledge sharing
1693-among edge clients. Table IV demonstrates that mAP and
1694-Rank-1 accuracy will decrease significantly without these
1695-modules, which also verifies that these modules can contribute
1696-to our method’s performance improvements. We then conduct
1697:detailed ablation studies to verify the effectiveness of each
1698-design of our FedSTIL.
1699:TABLE V
1700-T HE COMPARISON OF CATASTROPHIC FORGETTING WITH DIFFERENT
1701-MEMORY SIZES FOR PROTOTYPES REHEARSAL .
1702-Variant
1703-
--
1754-+ PR: 18K
1755-+ PR: 20K
1756-
1757-Influence of Memory Size on Forgetting. We analyze
1758:the catastrophic forgetting of our method as Eq. (8) under
1759-different memory sizes for prototype rehearsal. As shown in
1760-Table V, the Rank-1 Forgetting (R1-F), Rank-5 Forgetting
1761-(R5-F), and mAP Forgetting (mAP-F) keep decreasing as
1762-the memory size increases. With a larger memory size for
1763:storing prototypes, the knowledge from more historical tasks
1764-can be saved for future rehearsal. Therefore, the rehearsal
1765-with more historical prototypes can effectively alleviate the
1766:forgetting of the previous knowledge. In addition, the Rank1 Forgetting with the memory size 20,000 (20K) is nearly
1767-2% lower than without prototype rehearsal, which is within
1768-the acceptable range in the real-life person ReID scenario.
1769:Additionally, we also observe that forgetting decreases slowly
1770-or even rebounds (e.g., PR: 16K) as we give more memory
1771-size. The most probable reason is that some stored prototypes
1772-may be redundant or even noisy with larger memory space.
1773-In other words, although we store more prototypes, we can
1774:hardly gain equivalent knowledge for future rehearsal. Hence,
1775-one needs to carefully pick up the PR threshold to balance
1776:the tradeoff between memory usage and knowledge forgetting.
1777-Nevertheless, these results can still indicate that our prototype
1778:rehearsal effectively alleviate the catastrophic forgetting and
1779:retain the knowledge from the previous tasks during federated
1780-lifelong learning.
1781-
1782-Loss
1783-
--
1871-
1872-Fig. 12. The comparison of the averaged loss for different tasks w/ and w/o parameter tying. Parameter Tying can achieve convergence faster under
1873-continuously training for sequential tasks.
1874-
1875:TABLE VII
1876-P ERFORMANCES UNDER DIFFERENT DISTANCE METRICS .
1877-
1878:TABLE VI
1879-P ERFORMANCES WITH DIFFERENT BACKBONES .
1880-R5
1881-
1882:Storage
1883-
1884-TC
1885-
1886-Distance
--
2066-E5
2067-
2068-Performances with Different Backbones. We analyze the
2069-influences of choosing different network backbones on the
2070:accuracy, storage cost, and communication cost. Specifically,
2071-we utilize ResNet18, ResNet50, and Swin-Transformer (SwinT) [45] as backbones for training. Considering the limited
2072-computing capacity of edge devices, only the last block of
2073-these backbones are trainable during learning, and the rest
2074-blocks are fixed with the pre-trained weights. As shown in
2075-Table VI, our FedSTIL can outperform the other baselines on
2076:accuracy (i.e., mAP, Rank-1, Rank-5), storage, and total communication cost (TC) with different backbones. The storage
2077-size of FedSTIL is comparable with FedWeIT [34] and larger
2078:than FedCurv [31]. The communication cost of FedSTIL is
2079-lower than other baselines, while the accuracy (e.g., mAP,
2080-Rank-1, and Rank-5) of FedSTIL is the highest with different backbones. These results verify the compatibility of our
2081-method with different backbones. Therefore, one can choose
2082-suitable backbones based on the edge computing capacity.
--
2086-parameter tying from Task-1 to Task-5 during different training
2087-epochs. Our FedSTIL with the parameter tying can achieve
2088-lower loss and converge much faster than eliminating the parameter tying. The improvement is mainly because parameter
2089-tying enables on-edge models to continuously fit new tasks
2090:with minimal parameter changes and forgetting of previous
2091:knowledge learnt in the past. Thus, the design of tying the
2092-
2093:spatial-temporal correlated edge models for jointly optimizing
2094-can achieve better convergence and generalization.
2095-Performances under Different Distance Metrics. Considering the characteristics of the task features, we adopt
2096:Kullback-Leibler (KL) Divergence to compute the spatialtemporal correlation ratio when aggregating the personalized
2097:sharing knowledge. We also evaluate the retrieval accuracy of
2098-our method under some other distance metrics, namely, Cosine
2099-Distance and Euclidean Distance. As shown in Table VII,
2100-KL can achieve higher accuracy than other metrics. Different
2101-from Cosine and Euclidean, KL can effectively measure the
2102-information difference among the different distributions of task
2103-features. Hence, KL can help the parameter server precisely
2104:measure the edge similarities to aggregate the spatial-temporal
2105:knowledge with low noise and less interference.
2106-
2107-E4
2108-
2109-Swin-T
--
2125-E1
2126-
2127-Backbone
2128-
2129:Fig. 13. The visualization of the correlation weights at 20th , 40th and 60th
2130-rounds. X-axis and Y-axis show the edge IDs.
2131-
2132:Edge Correlations at Different Rounds. To better understand the effectiveness of our spatial-temporal knowledge
2133:integration, we visualize the edge task correlation weights
2134-(t)
2135-Wij calculated by our parameter server for all edges at 20th ,
2136-th
2137-40 and 60th round as shown in Fig. 13. Specifically, we predefine the ground-truth correlation groups that the 20th round
2138-are 1-3, 4-5; the 40th round are 1-2, 3-5; the 60th round are
2139-1-2, 3, 4-5. As shown in Fig. 13, we can observe that the pairwise correlation weights with light color almost corresponds to
2140:the ground-truth, which reflect the effectiveness of our spatialtemporal knowledge integration.
2141-
2142-12
2143-
2144-Positive Attention
--
2155-
2156-Fig. 14. The heat maps of sampled person images from five datasets. The hot-spots are what the models focus on, which also reflect the generalization
2157-degree of representation. Images with red and blue boundaries denote positive attention and negative attention from the perspective of models.
2158-
2159:D. Visualization Study
2160-To further verify the effectiveness of some designs in our
2161:method FedSTIL, we visualize the heat-maps of our method
2162:by removing adaptive lifelong learning and spatial-temporal
2163-integration to demonstrate their influences on the representation. We utilize the gradient-weighted class activation map
2164-(GradCAM) [48] to generate the heat maps, where the hotspots are the positive focus of the models for re-identification.
2165:We can observe in Fig. 14 that without the spatial-temporal
2166:knowledge integration, the model will only focus on some
2167-limited regions, such as shoes, coats, or bags. On the contrary,
2168-the attention regions of our oracle method FedSTIL are mainly
2169-on the overall person’s body and have more generalized representations. This improvement is largely due to the effectiveness
2170:of knowledge sharing to improve the generalizing capacities
2171-for edge clients. Moreover, if without our adaptive lifelong
2172-learning, we can also observe that the model attention may
2173-deviate from the regions of person body, and may even focus
2174-on the surrounding or background regions. This is probably
2175:because the learnt knowledge from the prior tasks will be
2176-gradually forgotten during the lifelong learning of new scenarios. However, our FedSTIL with the prototype rehearsal will
2177:mitigate the forgetting of the former knowledge and maintain
2178-the focus on person regions during lifelong learning.
2179-VI. F UTURE W ORK
2180-Domain Generalization. The edge clients may have completely different source domains. To adapt to diverse domains,
2181-distributed person ReID models may require frequent retraining or fine-tuning, resulting in large training time and resource
--
2190-lifelong learning with ResNet50, while some weak devices
2191-(e.g., Raspberry Pi 4B) can barely afford MobileNetV2. To
2192-meet the diverse computing capacities, the edge clients need
2193-to be deployed with different backbone networks. Thus, we
2194:will consider how to support knowledge sharing under heterogeneous backbones among distributed edges in the future.
2195-Occluded Person ReID. The real-world person ReID scenarios are occasionally crowded, and the captured person
2196-images may be overlapped or occluded by some obstacles.
2197-Occluded person ReID significantly differs from the conventional person ReID due to the lack of full-body person images.
2198-However, the missing information of the occluded person
2199:images may be complemented by judicious organizing spatialtemporal knowledge among distributed edges. Therefore, implementing the occluded person ReID techniques [23], [53]
2200-into our framework to enhance model robustness is also a
2201-promising direction for future work.
2202-VII. C ONCLUSION
2203-In this work, we proposed a federated lifelong learning
2204-framework, FedSTIL, which enables the distributed edge
2205-clients to learn collaboratively and continuously in real-life
2206-for person ReID in new scenarios. Our proposed method can
2207:utilize the spatial-temporal correlated knowledge among the
2208-edge clients to improve the data representation for person
2209-ReID and achieve better performance. Meanwhile, the learnt
2210:knowledge from the historical tasks can remain long-term effective with only slight forgetting during continuously learning
2211-for new scenarios. Experimental results verified that FedSTIL
2212-improves the accuracy of person ReID while reducing the
2213:communication cost for distributed learning.
2214-
2215-13
2216-
2217-ACKNOWLEDGMENT
2218-This research was supported by the Natural Science Foundation of Jiangsu Province under Grant BK20200486 and NSFC
2219-No. 61972086.
2220-R EFERENCES
2221-[1] X. Shu, X. Wang, X. Zang, S. Zhang, Y. Chen, G. Li, and Q. Tian,
2222:“Large-scale spatio-temporal person re-identification: Algorithms and
2223-benchmark,” IEEE Transactions on Circuits and Systems for Video
2224-Technology, 2021.
2225-[2] C. Zhao, X. Wang, W. Zuo, F. Shen, L. Shao, and D. Miao, “Similarity
2226-learning with joint transfer constraints for person re-identification,”
2227-Pattern Recognition, vol. 97, p. 107014, 2020.
2228-[3] N. Pu, W. Chen, Y. Liu, E. M. Bakker, and M. S. Lew, “Lifelong person
2229:re-identification via adaptive knowledge accumulation,” in Proceedings
2230-of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 7901–7910.
2231-[4] Y. Ge, F. Zhu, D. Chen, R. Zhao et al., “Self-paced contrastive learning
2232-with hybrid memory for domain adaptive object re-id,” Advances in
2233-Neural Information Processing Systems, 2020.
--
2249-[10] B. Zhao, S. Tang, D. Chen, H. Bilen, and R. Zhao, “Continual
2250-representation learning for biometric identification,” in Proceedings of
2251-the IEEE/CVF Winter Conference on Applications of Computer Vision,
2252-2021, pp. 1198–1208.
2253:[11] G. Wu and S. Gong, “Generalising without forgetting for lifelong person
2254-re-identification,” in Proceedings of the AAAI Conference on Artificial
2255-Intelligence, vol. 35, no. 4, 2021, pp. 2889–2897.
2256-[12] W. Ge, J. Du, A. Wu, Y. Xian, K. Yan, F. Huang, and W.-S. Zheng,
2257:“Lifelong person re-identification by pseudo task knowledge preservation,” 2022.
2258-[13] C. Zhang, X. Liu, J. Xu, T. Chen, G. Li, F. Jiang, and X. Li, “An
2259-edge based federated learning framework for person re-identification in
2260-uav delivery service,” in 2021 IEEE International Conference on Web
2261-Services (ICWS). IEEE, 2021, pp. 500–505.
2262-[14] S. Sun, G. Wu, and S. Gong, “Decentralised person re-identification with
2263:selective knowledge aggregation,” arXiv preprint: 2110.11384, 2021.
2264-[15] F. Yang, Z. Zhong, Z. Luo, S. Li, and N. Sebe, “Federated and generalized person re-identification through domain and feature hallucinating,”
2265-arXiv preprint arXiv:2203.02689, 2022.
2266-[16] G. Wu and S. Gong, “Decentralised learning from independent multidomain labels for person re-identification,” in Proceedings of the AAAI
2267-Conference on Artificial Intelligence, vol. 35, no. 4, 2021, pp. 2898–
--
2270-Technology, vol. 30, no. 4, pp. 1092–1108, 2019.
2271-[18] L. Zhang, G. Du, F. Liu, H. Tu, and X. Shu, “Global-local multiple
2272-granularity learning for cross-modality visible-infrared person reidentification,” IEEE Transactions on Neural Networks and Learning Systems,
2273-2021.
2274:[19] X. Shu, L. Zhang, G.-J. Qi, W. Liu, and J. Tang, “Spatiotemporal coattention recurrent neural networks for human-skeleton motion prediction,” IEEE Transactions on Pattern Analysis and Machine Intelligence,
2275-vol. 44, no. 6, pp. 3300–3315, 2021.
2276-[20] K. Wang, H. Wang, M. Liu, X. Xing, and T. Han, “Survey on person
2277-re-identification based on deep learning,” CAAI Transactions on Intelligence Technology, vol. 3, no. 4, pp. 219–227, 2018.
2278-
--
2310-in heterogeneous federated learning,” in Proceedings of the IEEE/CVF
2311-Conference on Computer Vision and Pattern Recognition, 2022, pp.
2312-10 143–10 153.
2313-[31] N. Shoham, T. Avidor, A. Keren, N. Israel, D. Benditkis, L. Mor-Yosef,
2314:and I. Zeitak, “Overcoming forgetting in federated learning on non-iid
2315-data,” arXiv preprint arXiv:1910.07796, 2019.
2316-[32] Y. Ma, Z. Xie, J. Wang, K. Chen, and L. Shou, “Continual federated
2317:learning based on knowledge distillation,” in Proceedings of the
2318-Thirty-First International Joint Conference on Artificial Intelligence,
2319-IJCAI-22, L. D. Raedt, Ed. International Joint Conferences on
2320-Artificial Intelligence Organization, 7 2022, pp. 2182–2188, main
2321-Track. [Online]. Available: https://doi.org/10.24963/ijcai.2022/303
--
2327-continual learning with weighted inter-client transfer,” in International
2328-Conference on Machine Learning. PMLR, 2021, pp. 12 073–12 086.
2329-[35] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins,
2330-A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska
2331:et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the national academy of sciences, 2017.
2332-[36] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars,
2333-“Memory aware synapses: Learning what (not) to forget,” in Proceedings
2334-of the European Conference on Computer Vision (ECCV), 2018.
2335-[37] G. M. van de Ven and A. S. Tolias, “Three continual learning scenarios
2336-and a case for generative replay,” 2018.
2337-[38] J. Konečnỳ, H. B. McMahan, F. X. Yu, P. Richtárik, A. T. Suresh, and
2338:D. Bacon, “Federated learning: Strategies for improving communication
2339-efficiency,” arXiv preprint arXiv:1610.05492, 2016.
2340-[39] X. Wang and G. Gao, “Smarteye: An open source framework for realtime video analytics with edge-cloud collaboration,” in Proceedings of
2341-the 29th ACM International Conference on Multimedia, 2021, pp. 3767–
2342-3770.
--
2364-[45] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and
2365-B. Guo, “Swin transformer: Hierarchical vision transformer using shifted
2366-windows,” in Proceedings of the IEEE/CVF International Conference on
2367-Computer Vision, 2021, pp. 10 012–10 022.
2368:[46] Z. Li and D. Hoiem, “Learning without forgetting,” IEEE transactions
2369-on pattern analysis and machine intelligence, vol. 40, no. 12, pp. 2935–
2370-2947, 2017.
2371-[47] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith,
2372-“Federated optimization in heterogeneous networks,” Proceedings of
--
2379-“Joint noise-tolerant learning and meta camera shift adaptation for
2380-unsupervised person re-identification,” in Proceedings of the IEEE/CVF
2381-conference on computer vision and pattern recognition, 2021.
2382-[50] X. Shu, G.-J. Qi, J. Tang, and J. Wang, “Weakly-shared deep transfer
2383:networks for heterogeneous-domain knowledge propagation,” in Proceedings of the 23rd ACM international conference on Multimedia, 2015,
2384-pp. 35–44.
2385-[51] C. Zhao, X. Lv, Z. Zhang, W. Zuo, J. Wu, and D. Miao, “Deep fusion
2386-feature representation learning with hard mining center-triplet loss for
2387-person re-identification,” IEEE Transactions on Multimedia, vol. 22,

codex
消融里它确实把服务器端的空间聚合、时间聚合、空间时间联合聚合分别去掉，跌幅很大；这比只报一个总消融更能支撑“空间时间相关性是问题核心”。我把表格页完整抓出来，避免只依赖搜索片段。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 11 'Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
9

(a) mAP Accuracy

(b) Rank-1 Accuracy
Fig. 8. The comparison of average mAP and Rank-1 accuracy of 6 local tasks on 5 edge clients during 60 communication rounds. Our FedSTIL achieves
higher and stable accuracy with less fluctuation during the federated lifelong learning.

Fig. 9. The comparison of forgetting for trained tasks over different communication rounds. Our FedSTIL can maintain higher accuracy for ever-learnt tasks
with less forgetting when training for new tasks.

Fine-tuning
AKA
EWC

20

GwF
PTKP
FedSTIL*

15
10
better

Alleviation of Forgetting. Our FedSTIL can maintain the
accuracy with less forgetting of the previous knowledge during
the continuous training on the subsequent tasks. We can
observe in Fig. 8 that both the Rank-1 and mAP accuracy
of federated learning methods (i.e., FedAvg [38] and FedProx
[47]) will not increase after the 40th round. As shown in Fig.
9, this is because the knowledge learnt from the past tasks
is gradually forgotten during the learning of the new tasks.
In contrast, federated lifelong methods (i.e., FedCurv [31],
FedWeIT [34], and FedSTIL) can alleviate forgetting on the
past tasks, which help to achieve higher accuracy. To future
investigate the effectiveness of our methods, we compare
catastrophic forgetting with the fine-tuning baseline (without
lifelong learning) and some other lifelong learning methods on

25

Forgetting (%)

can make up for the limitations of local data. Moreover, we
can observe that our FedSTIL achieves the highest accuracy
compared with other federated learning-based methods (i.e.,
FedSGD [38], FedAvg [38], FedProx [47], CFeD [32], FedCurv [31], and FedWeIT [34]). The improvement mainly owes
to the task-relevant knowledge sharing among edge clients,
which alleviates the interference incurred by the irrelevant
knowledge transferring among the edge clients’ models.

5

mAP-F

R1-F

R3-F

Fig. 10. The comparison of the catastrophic forgetting under the MSMT17
dataset. Our FedSTIL alleviates the forgetting under lifelong learning for new
scenarios.

MSMT17 dataset [5]. Also, we divide the MSMT17 dataset
into 6 task sequences, and the model backbone is ResNet50.
Fig. 10 shows us the forgetting after finishing lifelong learning
on the MSMT17 dataset. We can observe that our method
FedSTIL can still effectively alleviate the forgetting compared
with other lifelong person ReID baselines.

Better

70

→

mAP Accuracy (%)

10

65

FedSTIL*
(ResNet12)

60
55
50

FedSTIL*
(ResNet18)

FedSTIL*
(ResNet50)
FedWeIT(b)
(ResNet18)

CFeD
(ResNet18)

CFeD
(ResNet12)

FedWeIT
(ResNet50)
CFeD
(ResNet50)

FedCurv
(ResNet50)

FedCurv
(ResNet18)

FedWeIT(a)
(ResNet18)

4.0

8.0
16.0
32.0
Communication Cost (GB)

FedSTIL*

FedWeIT

FedCurv

64.0
CFeD

Fig. 11. The comparison of mAP accuracy over communication cost. Our
FedSTIL achieves the highest retrieval accuracy with less communication cost.

2) Comparison of Communication Cost: We illustrate the
mAP accuracy over total communication costs (client-toserver and server-to-client cost) in Fig. 11. FedSTIL is
communication-efficient and can achieve higher accuracy compared with the other federated lifelong learning methods. FedWeIT [34] can adjust the communication cost by l1 pruning for
the transferred parameters, however, the accuracy of FedWeIT
(a) is still much lower than our method under comparable
communication cost. FedCurv [31] has lower accuracy with
higher communication cost. That is because FedCurv [31]
needs to transfer excessive additional information to alleviate
catastrophic forgetting. Our FedSTIL is more communicationefficient because it does not need to exchange additional
parameters among edge clients except model weights. This
improvement is largely because FedSTIL only requests taskspecific parameters, which have been aggregated by the parameter server based on the relevant spatial-temporal knowledge.
Moreover, our FedSTIL can achieve higher accuracy compared
with the other baseline methods by using the cheaper model
ResNet18, which has lower computation cost and storage
consumption. Therefore, FedSTIL is more applicable for
deployment in real-life person ReID scenarios by reducing
communication and computational costs.
TABLE IV
T HE ABLATION STUDY OF THE INFLUENCES OF EDGE AND SERVER
MODULES ON THE ACCURACY OF OUR METHOD .
Variant

mAP (%)

R1 (%)

FedSTIL

68.16

66.05

Edge Modules
w/o Adaptive Layer
w/o Prototype Rehearsal
w/o Parameter Tying

63.71 (-4.45)
60.73 (-7.43)
62.53 (-5.63)

62.54 (-3.51)
58.18 (-7.87)
60.33 (-5.72)

Server Modules
w/o Spatial Model Aggregation
w/o Temporal Model Aggregation
w/o Spatial-Temporal Integration

58.72 (-9.44)
61.04 (-7.12)
54.26 (-13.90)

57.03 (-9.02)
59.53 (-6.52)
51.51 (-14.54)

C. Ablation Study
We conduct the ablation study by removing some key
components of our method to analyze their influences on
the performances, and the results are shown in Table IV.

Specifically, we first remove the edge modules to verify our
lifelong learning effectiveness. The ”w/o Adaptive Layer”
setting replaces our adaptive layer with standard convolution
kernels and MLP blocks. The ”w/o Prototypes Rehearsal”
setting eliminates the rehearsal of historical prototypes in
lifelong learning. The ”w/o Parameter Tying” setting removes
the tying regularization terms in the loss functions. Moreover,
we also verify the distributed knowledge sharing by removing
the parameter server modules. Specifically, the ”w/o Spatial
Model Aggregation” setting removes the model aggregation
on the spatial-relevant edges. The ”w/o Temporal Model
Aggregation” setting omits the historical task features when
calculating task distances in Eq. (5). The ”w/o S-T Integration”
setting removes both spatial- and temporal- knowledge sharing
among edge clients. Table IV demonstrates that mAP and
Rank-1 accuracy will decrease significantly without these
modules, which also verifies that these modules can contribute
to our method’s performance improvements. We then conduct
detailed ablation studies to verify the effectiveness of each
design of our FedSTIL.
TABLE V
T HE COMPARISON OF CATASTROPHIC FORGETTING WITH DIFFERENT
MEMORY SIZES FOR PROTOTYPES REHEARSAL .
Variant

Memory

mAP-F (↓)

R1-F (↓)

R5-F (↓)

w/o PR

NaN

5.40

5.71

4.26

415MB
505MB
572MB
662MB
719MB
783MB

4.87
4.20
4.02
4.14
3.78
3.57

5.25
4.59
4.36
4.48
4.03
3.81

3.51
3.19
2.76
2.92
2.83
2.75

+ PR: 10K
+ PR: 12K
+ PR: 14K
+ PR: 16K
+ PR: 18K
+ PR: 20K

Influence of Memory Size on Forgetting. We analyze
the catastrophic forgetting of our method as Eq. (8) under
different memory sizes for prototype rehearsal. As shown in
Table V, the Rank-1 Forgetting (R1-F), Rank-5 Forgetting
(R5-F), and mAP Forgetting (mAP-F) keep decreasing as
the memory size increases. With a larger memory size for
storing prototypes, the knowledge from more historical tasks
can be saved for future rehearsal. Therefore, the rehearsal
with more historical prototypes can effectively alleviate the
forgetting of the previous knowledge. In addition, the Rank1 Forgetting with the memory size 20,000 (20K) is nearly
2% lower than without prototype rehearsal, which is within
the acceptable range in the real-life person ReID scenario.
Additionally, we also observe that forgetting decreases slowly
or even rebounds (e.g., PR: 16K) as we give more memory
size. The most probable reason is that some stored prototypes
may be redundant or even noisy with larger memory space.
In other words, although we store more prototypes, we can
hardly gain equivalent knowledge for future rehearsal. Hence,
one needs to carefully pick up the PR threshold to balance
the tradeoff between memory usage and knowledge forgetting.
Nevertheless, these results can still indicate that our prototype
rehearsal effectively alleviate the catastrophic forgetting and
retain the knowledge from the previous tasks during federated
lifelong learning.

Loss

11

Task-1

6
5
4
3
2

Task-2

FedSTIL w/o PT
FedSTIL

FedSTIL w/o PT
FedSTIL

4
3
2

0

20

Training Epochs

40

60

Task-3

3.5
3.0
2.5
2.0
1.5

FedSTIL w/o PT
FedSTIL

Task-4

3.0

FedSTIL w/o PT
FedSTIL

2.5

Task-5

2.5

FedSTIL w/o PT
FedSTIL

2.0

2.0
1.5

1.5

80

120

Training Epochs

140

160

Training Epochs

180

220

Training Epochs

240

Training Epochs

Fig. 12. The comparison of the averaged loss for different tasks w/ and w/o parameter tying. Parameter Tying can achieve convergence faster under
continuously training for sequential tasks.

TABLE VII
P ERFORMANCES UNDER DIFFERENT DISTANCE METRICS .

TABLE VI
P ERFORMANCES WITH DIFFERENT BACKBONES .
R5

Storage

TC

Distance

mAP (%)

R1 (%)

R3 (%)

R5 (%)

FedCurv
CFeD
FedWeIT
FedSTIL

58.26
60.09
64.06
68.16

55.42
57.71
61.55
66.05

67.75
68.02
71.38
74.05

631MB
378MB
993MB
825MB

36.8GB
5.6GB
10.8GB
5.6GB

Cosine
Euclidean
KL

66.92
67.03
68.16

65.13
65.27
66.05

70.92
70.88
72.03

72.92
72.93
74.05

FedCurv
CFeD
FedWeIT
FedSTIL

62.35
63.73
65.92
70.55

58.72
59.31
62.04
68.30

70.03
72.14
73.58
75.14

951MB
508MB
1.42GB
1.62GB

68.5GB
14.1GB
18.1GB
14.1GB

FedCurv
CFeD
FedWeIT
FedSTIL

65.71
65.58
66.52
71.31

62.30
62.42
64.35
69.12

72.95
72.08
74.70
75.03

1.14GB
703MB
1.90GB
1.92GB

92.2GB
15.8GB
22.5GB
15.8GB

40th round

60th round

E5

E5

E5

E4

E5

E4

E3

E4

E3

E3

E4

0.5
0.4
0.3
0.2
0.1
E2

E3

E5

E2

E4

E2

E3

E2

E2

E1

E1

E1

E1

20th round
E1

E5

Performances with Different Backbones. We analyze the
influences of choosing different network backbones on the
accuracy, storage cost, and communication cost. Specifically,
we utilize ResNet18, ResNet50, and Swin-Transformer (SwinT) [45] as backbones for training. Considering the limited
computing capacity of edge devices, only the last block of
these backbones are trainable during learning, and the rest
blocks are fixed with the pre-trained weights. As shown in
Table VI, our FedSTIL can outperform the other baselines on
accuracy (i.e., mAP, Rank-1, Rank-5), storage, and total communication cost (TC) with different backbones. The storage
size of FedSTIL is comparable with FedWeIT [34] and larger
than FedCurv [31]. The communication cost of FedSTIL is
lower than other baselines, while the accuracy (e.g., mAP,
Rank-1, and Rank-5) of FedSTIL is the highest with different backbones. These results verify the compatibility of our
method with different backbones. Therefore, one can choose
suitable backbones based on the edge computing capacity.
Error Loss without Parameter Tying. To further investigate the effectiveness of parameter tying, we then analyze
the error loss of different tasks during training. As illustrated
in Fig. 12, we compare the error loss by eliminating the
parameter tying from Task-1 to Task-5 during different training
epochs. Our FedSTIL with the parameter tying can achieve
lower loss and converge much faster than eliminating the parameter tying. The improvement is mainly because parameter
tying enables on-edge models to continuously fit new tasks
with minimal parameter changes and forgetting of previous
knowledge learnt in the past. Thus, the design of tying the

spatial-temporal correlated edge models for jointly optimizing
can achieve better convergence and generalization.
Performances under Different Distance Metrics. Considering the characteristics of the task features, we adopt
Kullback-Leibler (KL) Divergence to compute the spatialtemporal correlation ratio when aggregating the personalized
sharing knowledge. We also evaluate the retrieval accuracy of
our method under some other distance metrics, namely, Cosine
Distance and Euclidean Distance. As shown in Table VII,
KL can achieve higher accuracy than other metrics. Different
from Cosine and Euclidean, KL can effectively measure the
information difference among the different distributions of task
features. Hence, KL can help the parameter server precisely
measure the edge similarities to aggregate the spatial-temporal
knowledge with low noise and less interference.

E4

Swin-T

R1

E3

ResNet50

mAP

E2

ResNet18

Method

E1

Backbone

Fig. 13. The visualization of the correlation weights at 20th , 40th and 60th
rounds. X-axis and Y-axis show the edge IDs.

Edge Correlations at Different Rounds. To better understand the effectiveness of our spatial-temporal knowledge
integration, we visualize the edge task correlation weights
(t)
Wij calculated by our parameter server for all edges at 20th ,
th
40 and 60th round as shown in Fig. 13. Specifically, we predefine the ground-truth correlation groups that the 20th round
are 1-3, 4-5; the 40th round are 1-2, 3-5; the 60th round are
1-2, 3, 4-5. As shown in Fig. 13, we can observe that the pairwise correlation weights with light color almost corresponds to
the ground-truth, which reflect the effectiveness of our spatialtemporal knowledge integration.


codex
第四篇是视频可见红外，我会看它如何把“跨模态差异”从常规外观差异重写成“低频相似、高频身份细节缺失或错配”的问题。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

1

Spatial-Temporal High-Frequency Learning for
Video-based Visible-Infrared Person
Re-Identification
Sichen Tao, Shuang Li, Jun Ye, Neng Dong, Fan Li, and Huafeng Li

I. I NTRODUCTION

P

ERSON Re-identification (ReID) involves matching the
identities of pedestrians across different camera views. It
has become a prominent research area due to its substantial
This work was supported by the National Natural Science Foundation of
China (Nos. 62362045, 61966021, 62276120), the Basic Research Project
of Yunnan Province (No. 202401AT070412), and the Yunnan Fundamental
Research Project (No. 202301AV070004). (Sichen Tao and Shuang Li contributed equally to this work.) (Corresponding author: Fan Li)
Sichen Tao, Fan Li, and Huafeng Li are with the Faculty of
Information Engineering and Automation, Kunming University
of Science and Technology, Kunming 650500, China (e-mail:
dorayaki95720@gmail.com;lifan198686@163.com;lhfchina99@kust.edu.cn).
Shuang Li is with the Chongqing University of Posts and Telecommunications, Chongqing 400065, China (e-mail: shuangli936@gmail.com)
Jun Ye is with the School of Information and Control Engineering,
China University of Mining and Technology, Xuzhou 221116, China (email:tb22060028a41@cumt.edu.cn)
Neng Dong is with the School of Computer Science and Engineering,
Nanjing University of Science and Technology, Nanjing 210094, China (email: neng.dong@njust.edu.cn).

Spatial-Temporal Filtering

}

Edge Detection Algorithm

Frame-level Intermediate Modality
(a) Existing Methods

Shallow
Embedding

Shallow
Embedding

Sequence-level Intermediate Modality
(b) Our Methods

Feature
Interaction

low-level
features

}

Generative
Model

}

Index Terms—Video-Based Visible-Infrared Person ReIdentification, Spatial-Temporal High-Frequency Information,
Sequence-Level Intermediate Modality

}

Abstract—Video-based
Visible-Infrared
Person
ReIdentification (VVI-ReID) aims to learn consistent person
feature representations across video sequences in different
modalities. Existing methods that use an intermediate modality
to bridge the gap between visible (RGB) and infrared (IR)
sequences tend to be limited by high construction costs, loss of
high-frequency details, and lack of temporal cues. Moreover, they
typically focus on refining global representation using high-level
features, neglecting the enhancement of local details through
low-level features. To address these challenges, we propose
the novel Spatial-Temporal High-Frequency Learning (STHF)
framework, which constructs an appropriate intermediate
modality for the VVI-ReID task and alleviates the modality
gap via hierarchical feature enhancement. Specifically, we
introduce the Spatial-Temporal High-Pass Filter (ST-HPF),
which filters out spatial-temporal Low-Frequency Components
(LFC), preserving high-frequency details to construct an
intermediate modality at the sequence level. We then enhance
the local details with low-level features through the Shallow
Detail Compensation (SDC) module, which reduces local
noise interference. Finally, the Deep Semantic Refinement
(DSR) module refines the global representation by modeling
spatial-temporal high-frequency semantic associations using
high-level features. Extensive experiments demonstrate that our
method significantly outperforms state-of-the-art approaches on
the publicly available HITSZ-VCM and BUPTCampus datasets.
The code is available at https://github.com/TSC95720/STHF.

Deep
Embedding

Detail
Enhancement

Feature
Interaction

high-level
features

Semantic
Refinement

Deep
Embedding

(c) Our Two-Branches Framework

Fig. 1. Comparison with existing intermediate modality-based methods and
the diagram of our method. (a) Existing VVI-ReID methods involve high costs
in constructing intermediate modality (i.e., Fake IR [8]), while also suffering
from the loss of high-frequency detailed information in the generated modality
(i.e., Anaglyph [9]). Furthermore, these methods rely solely on singleframe images when constructing intermediate modalities, neglecting temporal
modeling. (b) Our method leverages 3D FFT to extract discriminative spatialtemporal high-frequency details, constructing a sequence-level intermediate
modality. (c) Our method utilizes the intermediate modality to enhance the
local details of the original modality at shallow layers while guiding the
extraction of semantic information at deep layers.

potential in intelligent security and video surveillance [1]–
[6]. Although Person ReID has made considerable progress,
it faces inherent limitations. Primarily, performance degrades
significantly in challenging environments, such as low-light
or night-time conditions, due to compromised image quality.
Moreover, image-based ReID relying solely on single-frame
images makes it unsuitable for real-world applications. Consequently, Video-based Visible-Infrared Person Re-Identification
(VVI-ReID) [7], [8] has emerged as a promising solution,
aiming to match pedestrian sequences captured by night-time
infrared (IR) cameras with those captured by day-time visible
(RGB) cameras.
The core task of VVI-ReID is to bridge the modality gap
between IR and RGB pedestrian sequences. Most existing
methods [7], [10], [11] attempt to mitigate the modality

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

2D Spatial
HPF

(a) Remove Spatial LFC

1D Temporal
HPF

(b) Remove Temporal LFC

Fig. 2. The influence of removing the Low-Frequency Components (LFC)
in video pedestrian sequences. Where the HPF denotes the ideal high-pass
filter. (a) Filtering spatial LFC can reduce spatial redundant information such
as background, while retaining fine-grained structural details of pedestrians.
(b) Removing temporal LFC can filter out static temporal information while
capturing the motion patterns of pedestrians.

discrepancy by learning shared feature embeddings. However,
due to fundamental differences in the imaging mechanisms
of the RGB and IR sequences, substantial variations in color,
texture, and contrast persist, making direct feature alignment
ineffective and susceptible to identity information loss. To address this issue, intermediate modality-based methods [8], [9]
generate an intermediary modality to bridge the gap between
RGB and IR sequences, enhancing feature alignment while
preserving identity information. Despite their promising performance, these methods still face notable limitations. Specifically, as shown in Fig. 1(a), the construction of intermediate
modalities is inappropriate due to the following reasons: 1)
High construction cost: generation-based method [8] demand
substantial computational resources and are inherently prone
to mode collapse [12], [13]; 2) Loss of high-frequency
detail: edge detection-based method [9] typically focus on
capturing abrupt intensity changes and may smooth out subtle
variations, leading to the loss of high-frequency detail; 3) Lack
of temporal cues: these methods relying on single-frame
construct intermediate modalities, neglecting that the temporal
domain also contains important discriminative information,
such as temporal high-frequency information. In addition, the
utilization of intermediate modalities is insufficient: Deep
feature learning follows a low-level to high-level paradigm, as
shown in Fig. 1(c), yet previous methods [8], [9] typically use
the high-level semantic features of the intermediate modality to
refine the global representation while neglecting the low-level
features that are crucial for capturing fine-grained local details.
These factors limit intermediate modality-based methods from
effectively alleviating the modality discrepancy and learning
discriminative feature representations.
Recently, transforming signals into the frequency domain
using the Fast Fourier Transform (FFT) has been widely
adopted in deep learning [14]–[17]. Compared with the spatial
domain, the discriminability of features is physically defined

2

in the frequency domain. As illustrated in Fig. 2, by removing
the Low-Frequency Components (LFC) from video pedestrian
sequences, the spatial High-Frequency Components (HFC)
preserve fine-grained structural details, while temporal HFC
capture motion variations, both of which provide distinct
modality-invariant discriminative cues. Furthermore, FFT has
a low computational complexity of O(N log N), making it
significantly more efficient in processing large-scale video
data compared to traditional spatial domain methods. Thus,
as shown in Fig. 1(b), learning spatial-temporal discriminative
information from a frequency-domain perspective is crucial
for the VVI-ReID task.
Based on the above analysis, in this paper, we propose
a Spatial-Temporal High-Frequency Learning (STHF) framework for the VVI-ReID task. The proposed STHF aims
to mitigate the modality discrepancy while learning spatialtemporal high-frequency information from low-level to highlevel features. To achieve this, we propose three key modules:
Spatial-Temporal High-Pass Filter (ST-HPF), Shallow Detail
Compensation (SDC) module, and Deep Semantic Refinement
(DSR) module. The proposed ST-HPF constructs a novel
sequence-level intermediate modality based on a 3D Fast
Fourier Transform (FFT) by explicitly leveraging spatialtemporal cues to enhance identity discriminability. Based on
this, the proposed SDC module enhances local structural
details with the low-level features by accurately modeling
spatial correspondence between the intermediate modality and
the original modality, while the proposed DSR module refines
global representations with the high-level features by establishing spatial-temporal semantic associations. To ensure effective
information interaction, we first eliminate the style information
of the original modality before applying the SDC and DSR
modules. By learning spatial-temporal high-frequency information, STHF effectively mitigates the modality discrepancy
and extracts discriminative spatial-temporal representations.
Our main contributions can be summarized as follows:
We propose a novel Spatial-Temporal High-Frequency
Learning (STHF) framework to exploit the potential
spatial-temporal high-frequency information for the VVIReID task.
• We propose a Spatial-Temporal High-Pass Filter (STHPF) based on 3D FFT to construct a novel sequencelevel intermediate modality, which jointly addresses
modality alignment and spatial-temporal feature learning.
• We propose a Shallow Detail Compensation (SDC) module to enhance local details with low-level features, and a
Deep Semantic Refinement (DSR) module to refine global
representations with high-level features.
• Experiments on HITSZ-VCM and BUPTCampus demonstrate that our method significantly outperforms stateof-the-art approaches, validating the effectiveness of the
proposed ST-HPF, SDC, and DSR.
•

The remainder of this paper is organized as follows. Section
II reviews related work; Section III presents the proposed
method; Section IV analyzes the experimental results; Finally,
section V concludes the paper.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

II. R ELATED W ORK
A. Visible-Infrared Person Re-Identification
Visible-Infrared Person Re-Identification (VI-ReID) is a
cross-modality person retrieval task that aims to match individuals across cameras with different modalities. To alleviate
the discrepancy between RGB and IR images, numerous
works [18]–[23] have been proposed and have achieved remarkable performance. The current mainstream methods can
be categorized into three types: representation learning-based
methods [24]–[28], metric learning-based methods [29]–[32],
and generation-based methods [33]–[37].
Representation learning-based methods primarily focus
on extracting shared and discriminative features across the
two modalities by designing and modeling appropriate network
architectures. IDKL [24] leverages the discriminative knowledge embedded in modality-specific features to enhance the
discriminability of modality-shared features. MSCMNet [25]
proposes to fuse features at different scales and explores the
semantic correlation of fusion. DMANet [26] develops an
effective multi-granularity features mutual learning module to
eliminate the modality discrepancy. DEEN [27] learns informative feature representations by generating diverse embeddings while reducing the modality discrepancy between RGB
and IR images. DMA [28] proposes to perform compensation
for the information asymmetry in the HSV color space.
Metric learning-based methods aim to reduce the distance
between samples of the same identity across different modalities by designing appropriate feature metrics or loss functions.
HHGF [29] proposes the CMCC loss function to mine the invariance of global features by measuring mutual information in
images from two modalities. SDCL [30] effectively mitigates
the cross-modality discrepancy through the collaboration of
shallow and deep features. HOS-Net [31] proposes a modalityrange identity-center contrastive loss to reduce the distances
between the RGB, IR, and intermediate features. CPM [32]
introduces the closest permutation distance that is invariant to
changes in the order of the group members to measure the
similarity between two sets of features.
Generation-based methods generally transform crossmodality tasks into single-modality tasks through modality
conversion or seek an intermediate modality between the
two modalities to alleviate the modality discrepancy. GCIFS [33] generates high-quality cross-modality pairs of images
and fuses the information of the two modalities. PCMC [34]
splits the two modalities of the same person into patches and
concatenates them into a new modality image, effectively alleviating the problem of modality imbalance. HAT [35] proposes
an auxiliary grayscale modality generated from homogeneous
RGB, which preserves the structural information of visible
images while approximating the image style of the infrared
modality. XIV [36] employs a lightweight network trained in
a self-supervised manner to generate an X modality.
Although the above methods have achieved notable success,
the image-based VI-ReID is suboptimal for video sequence
retrieval due to its limited ability to capture temporal features,
resulting in inherent information loss.

3

B. Video-based Visible-Infrared Person Re-Identification
Compared to single-frame images, video sequences provide
richer spatial content and implicit temporal cues, which have
attracted increasing attention to VVI-ReID in recent years.
Inspired by previous works in video action recognition [38]–
[40] and video-based person re-identification [41]–[49], existing VVI-ReID methods often introduce additional networks
such as Recurrent Neural Networks (RNNs) [7], [9], Graph
Neural Networks (GNNs) [10], and Transformers [50], [51] to
learn discriminative spatial-temporal representations and thus
alleviate the modality discrepancy between RGB and IR video
sequences.
RNNs-based methods leverage recurrent structures to
model sequential dependencies. For instance, MITML [7]
first introduces the VVI-ReID task, contributes the HITZSVCM dataset, and employs an LSTM-based [52] temporal
memory module to aggregate frame-level features. IBAN [9]
proposes a bidirectional LSTM to integrate temporal features across frames and leverages the Anaglyph intermediate
modality to alleviate the modality discrepancy. In contrast,
GNNs-based methods, such as SAADG [10] formulates
modality discrepancies as style attacks and applied a Graph
Neural Network [53] to extract robust cross-modality representations by modeling intra- and inter-modality relations.
Recently, Transformers-based methods have demonstrated
great potential in VVI-ReID. CST [50] proposes a crossmodality spatial-temporal Transformer [54] that encapsulates
local pedestrian information into 3D tubes and facilitates
inter-frame interactions via message tokens. STIMM [51]
employs a temporal Transformer to capture comprehensive
motion patterns, thereby enhancing feature discriminability
across frames. Other methods, such as AuxNet [8], proposes
a temporal k-reciprocal re-ranking strategy to enhance feature
matching over time.
These methods have achieved significant success, fully
demonstrating the feasibility and effectiveness of the VVIReID task. However, they primarily focus on mitigating the
modality discrepancy and extracting spatial-temporal information in the spatial domain, while overlooking potential
solutions in the frequency domain.
C. Fourier Transform
The Fourier Transform has been increasingly applied in
deep learning for its ability to model global dependencies
in the frequency domain. Recent studies have further demonstrated its effectiveness across a range of tasks [14]–[16], [55].
For example, in medical image segmentation, FreMIM [14]
leverages Fast Fourier Transform to replace self-attention,
enabling efficient global information modeling. In human pose
estimation, FTCM [15] models the frequency and temporal
interactions between poses through separate feature mixing operations. In the area of low-light image enhancement, FourLLIE [56] explores the positive correlation between amplitude
magnitude and brightness magnitude. For high-quality image
deblurring, FFTformer [57] develops an efficient frequencybased self-attention solver, which reduces spatial and temporal complexity while improving efficiency and effectiveness.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

Conv

Original data flow

DSR
Module

Conv

3D Binary
HPF

Block 1

Block 2-3

int
Tri

int
ID

L

Conv

FC

BN

GAP

Conv

L

CA

IN

mc

KC

m1−c

f

h

VC

Mc

f cs

QS
KS
VS

F

...

Fp

Ms

F

h

...

VST
KST

Feature
Vector

BN

Batch Normalization

FC

Fully Connected Layer

MP

Modality Purifier

CA

Channel Attention

IN

Instance Normalization

F st
Softmax

f

fc

QC

Softmax

fp

Softmax

Modality Purifier (MP)

Resnet
Layer

(b) Deep Semantic Refinement (DSR) Module
MP

(a) Shallow Detail Compensation (SDC) Module

Common data flow

LTri

Block 4

3D iFFT

High-Frequency data flow

LID

FC

SDC
Module

3D FFT

BN

ST-HPF

GAP

Input sequences

4

M st

QST

Matrix-wise
Multiplication

2D Conv

Element-wise
Addition

2D Conv

Element-wise
Multiplication

3D Conv

Fig. 3. The framework of STHF, consists of a Spatial-Temporal High-Pass Filter (ST-HPF), a Shallow Detail Compensation (SDC) module, and a Deep
Semantic Refinement (DSR) module. The ST-HPF first constructs a sequence-level intermediate modality through spatial-temporal high-pass filtering. Then
SDC compensates the original modality with high-frequency spatial information in the shallow layers of ResNet, while DSR guides the original modality to
extract spatial-temporal high-frequency information in the deep layers. The two information interaction modules operate under modality-irrelevant conditions
by the modality purifier. We perform SDC operations in blocks 1 – 2 of ResNet and DSR operations in blocks 3 – 4. For simplicity, only block 1 and block
3 are depicted in the figure to exemplify the integration of SDC and DSR modules.

In face super-resolution, SFMNet [58] designs a spatialfrequency mutual learning network, where the frequency domain branch captures global dependency and the spatial domain branch extracts local dependency. The Fourier Transform
has also been widely adopted in person ReID. For instance,
FDMNet [59] learns modality-invariant features by enhancing modality-invariant amplitude components and suppressing
modality-specific components. FDNM [60] improves the discriminability of phase components by leveraging amplitude
guidance in the frequency domain, effectively learning robust
feature representations. FIDN [61] disentangles and utilizes
low-frequency and high-frequency spatial-temporal to capture
comprehensive and discriminative video representations.
Inspired by the success of these works, we propose a spatialtemporal high-pass filtering method for VVI-ReID to exploit
the potential spatial-temporal high-frequency information in
pedestrian sequences.

Fourier Transform (FFT). Specifically, Given a 1D signal x[n]
of length N, then the 1D FFT is defined as:
N−1

2π

X[k] = ∑ x[n]e− j( N )kn ,

(1)

n=0

where j is the imaginary unit. X[k] denotes the k-th frequency
component. Since the FFT is a linear transformation, the
original signal can be reconstructed using the inverse Fast
Fourier Transform (iFFT):
x[n] =

2π
1 N−1
X[k]e j( N )kn .
∑
N k=0

(2)

The 1D-FFT can be naturally extended to its multidimensional FFT by sequentially applying 1D FFT along each
dimension, which can be defined as:
N−1

n

Xk = ∑ xn e− j2πk( N ) ,

(3)

n=0

III. P ROPOSED METHOD
A. Basics of Fast Fourier Transform
The core task of VVI-ReID is to bridge the modality
gap between RGB and IR pedestrian sequences. Previous
intermediate modality-based methods [8], [9] focused mainly
on constructing intermediate modalities in the spatial domain
to mitigate the modality discrepancy, while neglecting the
discriminative spatial-temporal high-frequency information,
which is crucial for the VVI-ReID task. Therefore, before
introducing STHF, we first review the basics of the Fast

where k = (k1 , k2 , . . . , kd ) and n = (n1 , n2 , . . . , nd ) are the
frequency and spatial indices in d dimensions, respectively,
and N = (N1 , N2 , . . . , Nd ) denotes the size of each dimension.
For conciseness, the expression of the d-dimensional iFFT,
which can be directly derived from Eq. (2), is not explicitly
presented.
In the frequency domain, both spatial and temporal HFC
contain unique modality-invariant discriminative information
(e.g., edges, motion artifacts). Thus, by filtering out the spatialtemporal LFC from the RGB and IR pedestrian sequences,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

ST-HPF

1
1

1
1

1
11

1

1
1

I

1
1

3D iFFT

X̂
Frequency Domain

X

3D Binary
High-Pass Filter

5

Algorithm 1: Process of proposed ST-HPF
Input: Pedestrian video sequence V of the original
s ; temporal
modality; spatial cutoff frequency fco
t
cutoff frequency fco .
Output: Filtered sequence I with enhanced
spatial-temporal high-frequency components.
Step.1: # 3D Fast Fourier Transform;
X = FFT (V ), X = FFT Shi f t(X);
Step.2: # Generate 3D Binary High-Pass Filter;
F = 1T ×H×W ;
tcrop = T2 , hcrop = H2 , wcrop = W2 ;
t ) : (t
t
trange = (tcrop − fco
crop + f co );
s ) : (h
s
hrange = (hcrop − fco
crop + f co );
s
s );
wrange = (wcrop − fco ) : (wcrop + fco

3D FFT

V

Fig. 4. Spatial-Temporal High-Pass Filter (ST-HPF). The ST-HPF first maps
the original pedestrian sequence from the spatial domain to the frequency
domain using 3D FFT. Then, a 3D ideal high-pass filter is applied to remove
the LFC from the sequence. Finally, the sequence spectrum is mapped back to
the spatial domain by 3D iFFT, obtaining a spatial-temporal high-frequency
pedestrian sequence. Figures on the right are an example of applying ST-HPF.
In the frequency domain, the spatial-temporal LFC (small white regions inside
yellow boxes) are removed.

identity-related features that are robust to modality variations
can be effectively extracted, enabling the construction of a
suitable intermediate modality for the VVI-ReID task. We will
elaborate on how to filter out low-frequency information while
preserving useful spatial-temporal high-frequency information
in ST-HPF.
B. Overview
As shown in Fig. 3, the proposed STHF mainly includes
a Spatial-Temporal High-Pass Filter (ST-HPF), a Shallow
Detail Compensation (SDC) module, and a Deep Semantic
Refinement (DSR) module. The STHF first utilizes the STHPF to generate spatial-temporal high-frequency pedestrian
sequences as an intermediate modality. Based on this, the SDC
module enhances local details of low-level features, and the
DSR module refines the global representation with high-level
features.
C. Spatial-Temporal High-Pass Filter
To remove the redundant LFC across both spatial and
temporal dimensions, thereby constructing a suitable intermediate modality for the VVI-ReID task, inspired by [62], we
introduce a Spatial-Temporal High-Pass Filter (ST-HPF) based
on 3D-FFT with the binary filtering strategy. Since the video
pedestrian sequence V has three dimensions T (temporal), H
(height), and W (width), the filter is also 3D with three cutoff
frequencies. In the centered 3D spectrum, low spatial frequencies lie near (kh , kw ) = (0, 0), and low temporal frequencies are
near kt = 0, as demonstrated in [63], [64]. Following this prior,
we preset the cutoff frequencies according the sequence size.
s handles both H and
Specifically, a spatial cutoff frequency fco

F[trange , hrange , wrange ] = 0;
Step.3: # Spatial-Temporal High-Pass Filtering;
X̂ = F ⊙ X;
Step.4: # 3D inverse Fast Fourier Transform;
X̂ = iFFT Shi f t(X̂), I = iFFT (X̂);

t handles
W dimension, while a temporal cutoff frequency fco
T dimension.

To achieve this, as shown in Fig. 4, we first perform the
frequency domain projection to map the pedestrian sequence
to the frequency domain. Then, we perform the high-pass
filtering in the frequency domain to remove redundant LFC
of the pedestrian sequence. Finally, we conduct the spatial domain reconstruction to convert the spectrum back to the spatial
domain, obtaining the high-frequency pedestrian sequence.
Frequency Domain Projection. We first apply 3D FFT to
the pedestrian sequence V , transforming it into a spectrum.
Afterward, To facilitate spectrum analysis and filtering operations, we center the spectrum to ensure that the LFC is at the
center. The spectrum X is obtained as follows:
X = FFT Shi f t (F(V )),

(4)

where F (•) represents the 3D FFT, whose mathematical definition can be directly derived from Eq. (3), and FFT Shi f t (•)
denotes the operation that centers the frequency spectrum by
shifting the LFC to the center.
High-Pass Filtering. In the frequency domain, we introduce a 3D high-pass filter F, which removes redundant LFC
across both spatial and temporal dimensions through elementwise multiplication with the input sequence X. Therefore, the
resulting filtered spectrum X̂ can be obtained as follows:
X̂ = F ⊙ X,

(5)

where F is a binary matrix where 1 represents HFC and 0
represents LFC, constructed by retaining frequency components whose distance from the spectrum center exceeds the
predefined cutoff values while suppressing those below cutoff

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

values. The detailed construction is as follows:

t ,
1, if |t − tcrop | > fco




s ,
|h − hcrop | > fco
F(t, h, w) =
,
s ,

|w − wcrop | > fco



0, otherwise.

(6)

t and f s denote the cutoff frequencies in temporal
where fco
co
and spatial dimensions, respectively. Meanwhile, hcrop = H2 ,
wcrop = W2 , and tcrop = T2 are the center coordinates used
to determine the center position of the frequency spectrum
region.
Spatial Domain Reconstruction. After the spatial-temporal
high-pass filtering in the frequency domain, we restore the
center position of the filtered spectrum and then map it back
to the spatial domain to obtain the high-frequency pedestrian
sequence I by iFFT. Which can be described as:

(7)
I = F −1 iFFT Shi f t(X̂) ,

where F −1 (•) denotes the 3D iFFT, whose mathematical definition can be derived from Eq. (2) and Eq. (3). iFFT Shi f t (•)
denotes the operation that reverses the frequency spectrum
shift. Algorithm 1 describes the procedure of ST-HPF in detail.
As shown in Fig. 4, by performing the above operations,
spatial-temporal low-frequency information in the pedestrian
sequence is suppressed, thereby enhancing both structural and
motion-based identity cues.
D. Shallow Detail Compensation (SDC) Module
Low-level features typically contain rich details by preserving textures, edges, and local patterns, but are susceptible to
background interference and modality-specific noise. Therefore, we propose a Shallow Detail Compensation (SDC) module that aims to enhance the high-frequency detail of original
modality with low-level features. The structure of the proposed
SDC is shown in Fig. 3(a), considering the interference of
style information in the original modality, we first perform
modality purify on the RGB and IR images. Afterward, we
sequentially perform fine-grained high-frequency channel and
spatial information compensation to enhance the local details
of original modality.
Modality Purify. To prevent style information from interfering with interaction, inspired by [23], we introduce a
Modality Purifier (MP) before performing SCD. It eliminates
style information through an Instance Normalization (IN) layer
while preserving identity-related information using channel
attention. Specifically, given a low-level feature maps f ∈
RC×H×W , through MP, we obtain the purified feature maps
f p:
f p = mc ⊙ f + (1 − mc ) ⊙ IN ( f ),
(8)
where mc represents the channel attention mask and IN (•)
represents the IN layer. we use SE-Net [65] to generate the
channel mask mc :
mc = σ (W2 δ (W1 g ( f ))),

(9)

where the g (•) represents global average pooling, W1 and
W2 denote the learnable parameters of two fully connected

6

(FC) layers, δ (•) and σ (•) represent the ReLU and sigmoid
activation functions, respectively.
Channel Information Compensation. To enhance the
channel information, we extract the local detail affinity between the high-frequency feature maps f h ∈ RC×H×W and
the purified feature maps f p along the channel dimensions.
Specifically, we first calculate the channel-wise similarity
′
′
matrix M c ∈ RC ×C by matrix multiplication followed by
softmax:


M c = Fso f tmax ψq1 ( f p ) × ψk1 ( f h ) ,
(10)
where ψk1 ( f h ) and ψq1 ( f p ) represent the key and query embeddings to capture channel relationships, which are generated
through 2D 1 × 1 convolutional layers. Then, we combine the
value embedding ψv1 ( f h ) with the M c through a weighted sum
and add it to the f p , completing the information compensation
to obtain the channel-compensated feature maps f c :


f c = f p + ω c ψv1 ( f h ) × M c ,
(11)
where ω c is the 2D 1 × 1 convolutional layer used to restore
the feature dimensions.
Spatial Information Compensation. Furthermore, to enhance the spatial information, we compute the spatial-wise
similarity matrix M s ∈ RS×S between f c and f h using a similar
approach:


M s = Fso f tmax ψq2 ( f c ) × ψk2 ( f h ) ,
(12)
where ψq2 ( f c ), ψk2 ( f h ) represent the query and key embeddings
to capture spatial relationships. Then, we perform spatial information compensation, which is similar to channel information
compensation, to obtain the final compensated feature maps
f cs :


f cs = f c + ω s ψv2 ( f h ) × M s ,
(13)
where ω s is used to restore the feature dimensions and ψv1 ( f h )
represents the value embedding of the f h . Through channel
and spatial high-frequency information compensation, SDC
effectively integrates the rich high-frequency local details into
the original modality.
E. Deep Semantic Refinement (DSR) Module
High-level features are essential for abstract semantic understanding and global pattern recognition. To learn discriminative spatial-temporal high-frequency representations, we
propose a Deep Semantic Refinement (DSR) module that
aims to refine the global representation by modeling spatialtemporal high-frequency semantic associations using highlevel features. The structure of the proposed DSR is shown
in Fig. 3(b). Based on the previous analysis, we first perform
modality purify as well. Then, we perform spatial-temporal
refinement to refine the global representation.
Modality Purify. Similar to SDC, we first utilize a Modality
Purifier to eliminate identity-unrelated style information from
the original modality sequence feature maps F ∈ RT ×C×H×W ,
and the sequence feature maps after purification are denoted
as F p :
F p = MP (F),
(14)

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

7

input images
2D DHWT

}
}
3D FFT

Binary
+ 3DFilter

}
}

Anaglyph
Operator

xLL
LD

(a) Existing method

motion-based cues

(b) Our method

Fig. 5. Comparison of intermediate modality construction methods. (a)
Existing VVI-ReID method [9] constructs the intermediate modality based on
single-frame images and suffers from the loss of high-frequency detail information. (b) Our method constructs the intermediate modality at the sequence
level, enhancing spatial-temporal high-frequency discriminative information.

where the operations of MP (•) follow the same principles as
described in Eq. (8).
Spatial-Temporal Refinement. To establish global semantic associations, we extract the spatial-temporal affinity
between the high-frequency sequence feature maps F h ∈
RT ×C×H×W and F p . We first calculate the spatial-temporal
similarity matrix M st ∈ RST ×ST :


M st = Fso f tmax ϕq (F h ) × ϕk (F p ) ,
(15)
where ϕq (F h ) and ϕk (F p ) represent the key and query embeddings to mine spatial-temporal associations, which are generated through 3D 1 × 1 convolutional layers. Next, we combine
the value embedding ϕv (F p ) with the M st through a weighted
sum and add it to the F p , completing the information guidance
to obtain the spatial-temporal-guided sequence feature maps
F st :

F st = F p + ω st ϕv (F p ) × M st ,
(16)
where ω st is the 3D 1×1 convolutional layer use to restore the
feature dimensions. Through spatial-temporal high-frequency
information refinement, DSR guides the original modality in
effectively extracting spatial-temporal information.
F. Discussion
To further highlight the uniqueness and novelty of our
approach, we provide the following discussions focusing on
three key aspects: intermediate modality design, hierarchical
feature learning, and frequency-domain learning strategies.
Intermediate Modality Construction. Rather than constructing image-level intermediate modalities for the VVIReID task, STHF incorporates temporal modeling into the
construction of the intermediate modality, further enhancing
identity discriminability. As shown in Fig. 5(a), although the
Anaglyph data [9] alleviate modality discrepancy to a large extent, they lose critical discriminative information due to gener-

input sequences

2D FFT

3D FFT

P

xHL
LFC
Drop

2D iDHWT
Encoder

(a)

high-frequency detail loss

input feature maps

A

Conv

Conv

Conv

Conv

P

A

2D iFFT

(b)

3D Binary
HPF

3D iFFT
Encoder

(c)

Fig. 6. Different strategies in frequency-domain learning. (a) PHA [66] first
decomposes images into LFC and HFC using 2D Discrete Haar Wavelet
Transform (DHWT), then drops the LFC via a Low-Frequency Drop (LD)
module, and finally enhances spatial high-frequency information through
representation learning. (b) FRB [58] applies 2D FFT to decompose features into amplitude and phase components, and captures image-level global
dependencies via convolutional operations. (c) Our method applies 3D FFT
to the input sequences and removes spatial-temporal LFC using a 3D binary
high-pass filter, followed by representation learning to extract spatial-temporal
high-frequency information.

ating the intermediate modality from a single-frame image, ignoring the inherent spatial-temporal correlations within video
sequences. In contrast, the proposed ST-HPF constructs a
sequence-level intermediate modality by removing the spatialtemporal LFC from IR and RGB pedestrian sequences, which
not only reduces modality discrepancy but also enhances highfrequency details in both temporal and spatial dimensions,
as illustrated in Fig. 5(b). Therefore, constructing a highly
discriminative sequence-level intermediate modality is crucial
for the VVI-ReID task.
Hierarchical Representation Enhancement. STHF introduces two distinct feature interaction modules to enhance
representations from both shallow and deep embedding space.
First, the Shallow Detail Compensation (SDC) module leverages low-level features through 2D CNNs at the frame level
to compensate for local detail loss. Then, the Deep Semantic
Refinement (DSR) module employs high-level features via
3D CNNs at the sequence level to refine global semantic
representations. Through the combined effect of SDC and
DSR, STHF learns more comprehensive and discriminative
representations.
Spatial-Temporal Frequency Modeling. STHF introduces
a frequency-domain strategy well-suited to the VVI-ReID task.
As shown in Fig. 6, compared with capturing spatial highfrequency cues via coarse frequency separation through 2D
Discrete Haar Wavelet Transform (DHWT) [66] and modeling
global frequency information in the spatial domain using 2D
FFT [58], STHF employs 3D FFT combined with a 3D binary
filter to extract spatial-temporal high-frequency information,
thus preserving the complete frequency spectrum and further
enhancing identity discriminability.
G. Optimization
We combine cross-entropy loss and triplet loss to jointly
optimize our network for learning discriminative and modality-

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

8

TABLE I
C OMPARE WITH THE STATE - OF - THE - ART PERSON R E -ID METHODS ON HITSZ-VCM. “R@1”, “R@5” AND “R@10” DENOTE R ANK -1, R ANK -5, AND
R ANK -10, RESPECTIVELY. T HESE RESULTS ARE COPIED FROM THEIR PAPERS .

Methods

Reference

DDAG [67]
Lba [68]
MPANet [23]
VSD [69]
CAJ [70]
SEFL [71]
MITML [7]
IBAN [9]
SADSTRM [72]
SAADG [10]
CST [50]
AuxNet [8]
our

ECCV’20
ICCV’21
CVPR’21
CVPR’21
ICCV’21
CVPR’23
CVPR’22
TCSVT’23
Arxiv’23
ACM MM’23
TMM’24
TIFS’24
-

Type
Image
Image
Image
Image
Image
Image
Video
Video
Video
Video
Video
Video
Video

Infrared to Visible

Seq Len
6
6
6
6
6
6
6
6
6
6
6
6
6

Visible to Infrared

R@1

R@5

R@10

mAP

R@1

R@5

R@10

mAP

54.6
46.4
46.5
54.5
56.6
67.7
63.7
65.0
65.3
69.2
::::
69.4
51.1
70.4

69.8
65.3
63.1
70.0
73.5
80.3
76.9
78.3
77.9
80.6
::::
81.1
81.6

76.0
72.2
70.5
76.3
79.5
84.7
81.7
83.0
82.7
85.0
::::
85.8
86.2

39.3
30.7
35.3
41.2
41.5
52.3
45.3
48.8
49.5
53.8
::::
51.2
46.0
56.2

59.0
49.3
50.3
57.5
60.1
70.2
64.5
69.6
67.7
73.1
::::
72.6
54.6
73.5

74.6
69.3
67.3
73.7
74.6
82.2
79.0
81.5
80.7
83.5
::::
83.4
83.7

79.5
75.9
73.6
79.4
79.9
86.1
83.0
85.4
85.1
86.9
::::
86.7
87.0

41.5
32.4
37.8
43.5
42.8
52.5
47.7
51.0
51.8
56.1
::::
53.0
48.7
58.6

TABLE II
C OMPARE WITH THE STATE - OF - THE - ART PERSON R E -ID METHODS ON BUPTC AMPUS . “R@1”, “R@5” AND “R@10” DENOTE R ANK -1, R ANK -5 AND
R ANK -10, RESPECTIVELY. T HESE RESULTS ARE COPIED FROM THEIR PAPERS . AUX N ET∗ INDICATES THAT WE DID NOT USE THE RE - RANKING
ALGORITHM IN THIS METHOD .

Methods

Reference

Type

Seq Len

AlignGAN [73]
DDAG [67]
LbA [68]
CAJ [70]
AGW [1]
MMN [74]
DART [75]
DEEN [27]
MITML [7]
AuxNet∗ [8]
our

ICCV’19
ECCV’20
ICCV’21
ICCV’21
TPAMI’21
CVPR’21
CVPR’22
CVPR’23
CVPR’22
TIFS’24
-

Image
Image
Image
Image
Image
Image
Image
Image
Video
Video
Video

10
10
10
10
10
10
10
10
6
6
6

invariant features. Therefore, the total loss of the proposed
STHF is formulated as:
int
int
,
Ltotal = LID + LTri + λ1 LID
+ λ2 LTri

(17)

where λ1 and λ2 are the hyper-parameters, LID is the crossentropy loss and LTri is the triplet loss. During training, LID
int and Lint
and LTri work on the RGB and IR data, while LID
Tri
work on high-frequency data.
IV. E XPERIMENTS
A. Datasets and Evaluation Metrics
HITSZ-VCM [7] comprises 251452 RGB images and 211807
IR images from 927 identities captured across 12 cameras.
Each pedestrian track is composed of 24 sequential frames,
and all images are categorized into 11,785 RGB tracks and
10,078 IR tracks.
BUPTCampus [8] comprises 1869066 images of 3080 different identities, all captured from 6 camera perspectives. Each
track of pedestrians consists of 111 sequential frames, and all
images are categorized into 16826 tracks.

Infrared to Visible

Visible to Infrared

R@1

R@5

R@10

mAP

R@1

R@5

R@10

mAP

28.0
40.4
32.1
40.5
36.4
40.9
52.4
53.7
49.1
58.4
::::
66.7

49.1
61.4
54.9
66.8
60.1
67.2
70.5
74.8
67.9
76.1
::::
81.8

57.7
69.8
65.1
73.3
67.2
74.4
77.8
80.7
75.4
84.1
::::
87.0

30.3
40.4
32.9
41.5
37.4
41.7
49.1
50.4
47.5
55.5
::::
62.1

35.4
46.3
39.1
45.0
43.7
43.7
53.3
49.8
50.2
58.4
::::
59.8

53.9
68.2
58.7
70.0
64.4
65.2
75.2
71.6
68.3
76.4
::::
76.8

61.3
74.4
66.5
77.0
73.2
73.5
81.7
81.0
75.7
81.1
::::
82.8

35.1
43.1
37.1
43.6
41.1
42.8
50.5
48.6
46.3
53.5
::::
54.6

Evaluation Metrics. We use Cumulative Matching Characteristics (CMC) and Mean Average Precision (mAP) as metrics
to evaluate the cross-modality retrieval performance of the
proposed STHF.
B. Implementation Details
All our experiments are implemented on a single RTX8000
GPU. The proposed STHF employs the ResNet50 as the backbone network, which is initialized with pre-trained weights on
ImageNet. All images of the input sequence were resized to
288×144. Furthermore, we use random cropping, random flipping [76], and Channel-Augmented Joint Learning (CAJ) [70]
to enhance the input sequence. We trained our network with
an initial learning rate set to 3.5e-4 and adopted the Adam
optimizer with the cosine learning rate decay strategy [77] to
adjust the learning rate dynamically. Each training is run over
for 200 epochs, and 150 epochs for ablation studies. We set
the sequence length to 6 and the batch size to 16 (each minibatch includes 4 IDs and each ID includes 4 video clips of 2
t and
modalities), respectively. For the ST-HPF, we set the fco
s
fco to 2 and 10, respectively.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


codex
第四篇输出较长，中间把方法开头截掉了一部分。我会补齐第 4 到第 7 页，尤其是总体框架和 ST-HPF 的频域构造，因为这里是它从观察到机制的关键链条。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 4 -l 7 'Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

Conv

Original data flow

DSR
Module

Conv

3D Binary
HPF

Block 1

Block 2-3

int
Tri

int
ID

L

Conv

FC

BN

GAP

Conv

L

CA

IN

mc

KC

m1−c

f

h

VC

Mc

f cs

QS
KS
VS

F

...

Fp

Ms

F

h

...

VST
KST

Feature
Vector

BN

Batch Normalization

FC

Fully Connected Layer

MP

Modality Purifier

CA

Channel Attention

IN

Instance Normalization

F st
Softmax

f

fc

QC

Softmax

fp

Softmax

Modality Purifier (MP)

Resnet
Layer

(b) Deep Semantic Refinement (DSR) Module
MP

(a) Shallow Detail Compensation (SDC) Module

Common data flow

LTri

Block 4

3D iFFT

High-Frequency data flow

LID

FC

SDC
Module

3D FFT

BN

ST-HPF

GAP

Input sequences

4

M st

QST

Matrix-wise
Multiplication

2D Conv

Element-wise
Addition

2D Conv

Element-wise
Multiplication

3D Conv

Fig. 3. The framework of STHF, consists of a Spatial-Temporal High-Pass Filter (ST-HPF), a Shallow Detail Compensation (SDC) module, and a Deep
Semantic Refinement (DSR) module. The ST-HPF first constructs a sequence-level intermediate modality through spatial-temporal high-pass filtering. Then
SDC compensates the original modality with high-frequency spatial information in the shallow layers of ResNet, while DSR guides the original modality to
extract spatial-temporal high-frequency information in the deep layers. The two information interaction modules operate under modality-irrelevant conditions
by the modality purifier. We perform SDC operations in blocks 1 – 2 of ResNet and DSR operations in blocks 3 – 4. For simplicity, only block 1 and block
3 are depicted in the figure to exemplify the integration of SDC and DSR modules.

In face super-resolution, SFMNet [58] designs a spatialfrequency mutual learning network, where the frequency domain branch captures global dependency and the spatial domain branch extracts local dependency. The Fourier Transform
has also been widely adopted in person ReID. For instance,
FDMNet [59] learns modality-invariant features by enhancing modality-invariant amplitude components and suppressing
modality-specific components. FDNM [60] improves the discriminability of phase components by leveraging amplitude
guidance in the frequency domain, effectively learning robust
feature representations. FIDN [61] disentangles and utilizes
low-frequency and high-frequency spatial-temporal to capture
comprehensive and discriminative video representations.
Inspired by the success of these works, we propose a spatialtemporal high-pass filtering method for VVI-ReID to exploit
the potential spatial-temporal high-frequency information in
pedestrian sequences.

Fourier Transform (FFT). Specifically, Given a 1D signal x[n]
of length N, then the 1D FFT is defined as:
N−1

2π

X[k] = ∑ x[n]e− j( N )kn ,

(1)

n=0

where j is the imaginary unit. X[k] denotes the k-th frequency
component. Since the FFT is a linear transformation, the
original signal can be reconstructed using the inverse Fast
Fourier Transform (iFFT):
x[n] =

2π
1 N−1
X[k]e j( N )kn .
∑
N k=0

(2)

The 1D-FFT can be naturally extended to its multidimensional FFT by sequentially applying 1D FFT along each
dimension, which can be defined as:
N−1

n

Xk = ∑ xn e− j2πk( N ) ,

(3)

n=0

III. P ROPOSED METHOD
A. Basics of Fast Fourier Transform
The core task of VVI-ReID is to bridge the modality
gap between RGB and IR pedestrian sequences. Previous
intermediate modality-based methods [8], [9] focused mainly
on constructing intermediate modalities in the spatial domain
to mitigate the modality discrepancy, while neglecting the
discriminative spatial-temporal high-frequency information,
which is crucial for the VVI-ReID task. Therefore, before
introducing STHF, we first review the basics of the Fast

where k = (k1 , k2 , . . . , kd ) and n = (n1 , n2 , . . . , nd ) are the
frequency and spatial indices in d dimensions, respectively,
and N = (N1 , N2 , . . . , Nd ) denotes the size of each dimension.
For conciseness, the expression of the d-dimensional iFFT,
which can be directly derived from Eq. (2), is not explicitly
presented.
In the frequency domain, both spatial and temporal HFC
contain unique modality-invariant discriminative information
(e.g., edges, motion artifacts). Thus, by filtering out the spatialtemporal LFC from the RGB and IR pedestrian sequences,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

ST-HPF

1
1

1
1

1
11

1

1
1

I

1
1

3D iFFT

X̂
Frequency Domain

X

3D Binary
High-Pass Filter

5

Algorithm 1: Process of proposed ST-HPF
Input: Pedestrian video sequence V of the original
s ; temporal
modality; spatial cutoff frequency fco
t
cutoff frequency fco .
Output: Filtered sequence I with enhanced
spatial-temporal high-frequency components.
Step.1: # 3D Fast Fourier Transform;
X = FFT (V ), X = FFT Shi f t(X);
Step.2: # Generate 3D Binary High-Pass Filter;
F = 1T ×H×W ;
tcrop = T2 , hcrop = H2 , wcrop = W2 ;
t ) : (t
t
trange = (tcrop − fco
crop + f co );
s ) : (h
s
hrange = (hcrop − fco
crop + f co );
s
s );
wrange = (wcrop − fco ) : (wcrop + fco

3D FFT

V

Fig. 4. Spatial-Temporal High-Pass Filter (ST-HPF). The ST-HPF first maps
the original pedestrian sequence from the spatial domain to the frequency
domain using 3D FFT. Then, a 3D ideal high-pass filter is applied to remove
the LFC from the sequence. Finally, the sequence spectrum is mapped back to
the spatial domain by 3D iFFT, obtaining a spatial-temporal high-frequency
pedestrian sequence. Figures on the right are an example of applying ST-HPF.
In the frequency domain, the spatial-temporal LFC (small white regions inside
yellow boxes) are removed.

identity-related features that are robust to modality variations
can be effectively extracted, enabling the construction of a
suitable intermediate modality for the VVI-ReID task. We will
elaborate on how to filter out low-frequency information while
preserving useful spatial-temporal high-frequency information
in ST-HPF.
B. Overview
As shown in Fig. 3, the proposed STHF mainly includes
a Spatial-Temporal High-Pass Filter (ST-HPF), a Shallow
Detail Compensation (SDC) module, and a Deep Semantic
Refinement (DSR) module. The STHF first utilizes the STHPF to generate spatial-temporal high-frequency pedestrian
sequences as an intermediate modality. Based on this, the SDC
module enhances local details of low-level features, and the
DSR module refines the global representation with high-level
features.
C. Spatial-Temporal High-Pass Filter
To remove the redundant LFC across both spatial and
temporal dimensions, thereby constructing a suitable intermediate modality for the VVI-ReID task, inspired by [62], we
introduce a Spatial-Temporal High-Pass Filter (ST-HPF) based
on 3D-FFT with the binary filtering strategy. Since the video
pedestrian sequence V has three dimensions T (temporal), H
(height), and W (width), the filter is also 3D with three cutoff
frequencies. In the centered 3D spectrum, low spatial frequencies lie near (kh , kw ) = (0, 0), and low temporal frequencies are
near kt = 0, as demonstrated in [63], [64]. Following this prior,
we preset the cutoff frequencies according the sequence size.
s handles both H and
Specifically, a spatial cutoff frequency fco

F[trange , hrange , wrange ] = 0;
Step.3: # Spatial-Temporal High-Pass Filtering;
X̂ = F ⊙ X;
Step.4: # 3D inverse Fast Fourier Transform;
X̂ = iFFT Shi f t(X̂), I = iFFT (X̂);

t handles
W dimension, while a temporal cutoff frequency fco
T dimension.

To achieve this, as shown in Fig. 4, we first perform the
frequency domain projection to map the pedestrian sequence
to the frequency domain. Then, we perform the high-pass
filtering in the frequency domain to remove redundant LFC
of the pedestrian sequence. Finally, we conduct the spatial domain reconstruction to convert the spectrum back to the spatial
domain, obtaining the high-frequency pedestrian sequence.
Frequency Domain Projection. We first apply 3D FFT to
the pedestrian sequence V , transforming it into a spectrum.
Afterward, To facilitate spectrum analysis and filtering operations, we center the spectrum to ensure that the LFC is at the
center. The spectrum X is obtained as follows:
X = FFT Shi f t (F(V )),

(4)

where F (•) represents the 3D FFT, whose mathematical definition can be directly derived from Eq. (3), and FFT Shi f t (•)
denotes the operation that centers the frequency spectrum by
shifting the LFC to the center.
High-Pass Filtering. In the frequency domain, we introduce a 3D high-pass filter F, which removes redundant LFC
across both spatial and temporal dimensions through elementwise multiplication with the input sequence X. Therefore, the
resulting filtered spectrum X̂ can be obtained as follows:
X̂ = F ⊙ X,

(5)

where F is a binary matrix where 1 represents HFC and 0
represents LFC, constructed by retaining frequency components whose distance from the spectrum center exceeds the
predefined cutoff values while suppressing those below cutoff

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

values. The detailed construction is as follows:

t ,
1, if |t − tcrop | > fco




s ,
|h − hcrop | > fco
F(t, h, w) =
,
s ,

|w − wcrop | > fco



0, otherwise.

(6)

t and f s denote the cutoff frequencies in temporal
where fco
co
and spatial dimensions, respectively. Meanwhile, hcrop = H2 ,
wcrop = W2 , and tcrop = T2 are the center coordinates used
to determine the center position of the frequency spectrum
region.
Spatial Domain Reconstruction. After the spatial-temporal
high-pass filtering in the frequency domain, we restore the
center position of the filtered spectrum and then map it back
to the spatial domain to obtain the high-frequency pedestrian
sequence I by iFFT. Which can be described as:

(7)
I = F −1 iFFT Shi f t(X̂) ,

where F −1 (•) denotes the 3D iFFT, whose mathematical definition can be derived from Eq. (2) and Eq. (3). iFFT Shi f t (•)
denotes the operation that reverses the frequency spectrum
shift. Algorithm 1 describes the procedure of ST-HPF in detail.
As shown in Fig. 4, by performing the above operations,
spatial-temporal low-frequency information in the pedestrian
sequence is suppressed, thereby enhancing both structural and
motion-based identity cues.
D. Shallow Detail Compensation (SDC) Module
Low-level features typically contain rich details by preserving textures, edges, and local patterns, but are susceptible to
background interference and modality-specific noise. Therefore, we propose a Shallow Detail Compensation (SDC) module that aims to enhance the high-frequency detail of original
modality with low-level features. The structure of the proposed
SDC is shown in Fig. 3(a), considering the interference of
style information in the original modality, we first perform
modality purify on the RGB and IR images. Afterward, we
sequentially perform fine-grained high-frequency channel and
spatial information compensation to enhance the local details
of original modality.
Modality Purify. To prevent style information from interfering with interaction, inspired by [23], we introduce a
Modality Purifier (MP) before performing SCD. It eliminates
style information through an Instance Normalization (IN) layer
while preserving identity-related information using channel
attention. Specifically, given a low-level feature maps f ∈
RC×H×W , through MP, we obtain the purified feature maps
f p:
f p = mc ⊙ f + (1 − mc ) ⊙ IN ( f ),
(8)
where mc represents the channel attention mask and IN (•)
represents the IN layer. we use SE-Net [65] to generate the
channel mask mc :
mc = σ (W2 δ (W1 g ( f ))),

(9)

where the g (•) represents global average pooling, W1 and
W2 denote the learnable parameters of two fully connected

6

(FC) layers, δ (•) and σ (•) represent the ReLU and sigmoid
activation functions, respectively.
Channel Information Compensation. To enhance the
channel information, we extract the local detail affinity between the high-frequency feature maps f h ∈ RC×H×W and
the purified feature maps f p along the channel dimensions.
Specifically, we first calculate the channel-wise similarity
′
′
matrix M c ∈ RC ×C by matrix multiplication followed by
softmax:


M c = Fso f tmax ψq1 ( f p ) × ψk1 ( f h ) ,
(10)
where ψk1 ( f h ) and ψq1 ( f p ) represent the key and query embeddings to capture channel relationships, which are generated
through 2D 1 × 1 convolutional layers. Then, we combine the
value embedding ψv1 ( f h ) with the M c through a weighted sum
and add it to the f p , completing the information compensation
to obtain the channel-compensated feature maps f c :


f c = f p + ω c ψv1 ( f h ) × M c ,
(11)
where ω c is the 2D 1 × 1 convolutional layer used to restore
the feature dimensions.
Spatial Information Compensation. Furthermore, to enhance the spatial information, we compute the spatial-wise
similarity matrix M s ∈ RS×S between f c and f h using a similar
approach:


M s = Fso f tmax ψq2 ( f c ) × ψk2 ( f h ) ,
(12)
where ψq2 ( f c ), ψk2 ( f h ) represent the query and key embeddings
to capture spatial relationships. Then, we perform spatial information compensation, which is similar to channel information
compensation, to obtain the final compensated feature maps
f cs :


f cs = f c + ω s ψv2 ( f h ) × M s ,
(13)
where ω s is used to restore the feature dimensions and ψv1 ( f h )
represents the value embedding of the f h . Through channel
and spatial high-frequency information compensation, SDC
effectively integrates the rich high-frequency local details into
the original modality.
E. Deep Semantic Refinement (DSR) Module
High-level features are essential for abstract semantic understanding and global pattern recognition. To learn discriminative spatial-temporal high-frequency representations, we
propose a Deep Semantic Refinement (DSR) module that
aims to refine the global representation by modeling spatialtemporal high-frequency semantic associations using highlevel features. The structure of the proposed DSR is shown
in Fig. 3(b). Based on the previous analysis, we first perform
modality purify as well. Then, we perform spatial-temporal
refinement to refine the global representation.
Modality Purify. Similar to SDC, we first utilize a Modality
Purifier to eliminate identity-unrelated style information from
the original modality sequence feature maps F ∈ RT ×C×H×W ,
and the sequence feature maps after purification are denoted
as F p :
F p = MP (F),
(14)

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

7

input images
2D DHWT

}
}
3D FFT

Binary
+ 3DFilter

}
}

Anaglyph
Operator

xLL
LD

(a) Existing method

motion-based cues

(b) Our method

Fig. 5. Comparison of intermediate modality construction methods. (a)
Existing VVI-ReID method [9] constructs the intermediate modality based on
single-frame images and suffers from the loss of high-frequency detail information. (b) Our method constructs the intermediate modality at the sequence
level, enhancing spatial-temporal high-frequency discriminative information.

where the operations of MP (•) follow the same principles as
described in Eq. (8).
Spatial-Temporal Refinement. To establish global semantic associations, we extract the spatial-temporal affinity
between the high-frequency sequence feature maps F h ∈
RT ×C×H×W and F p . We first calculate the spatial-temporal
similarity matrix M st ∈ RST ×ST :


M st = Fso f tmax ϕq (F h ) × ϕk (F p ) ,
(15)
where ϕq (F h ) and ϕk (F p ) represent the key and query embeddings to mine spatial-temporal associations, which are generated through 3D 1 × 1 convolutional layers. Next, we combine
the value embedding ϕv (F p ) with the M st through a weighted
sum and add it to the F p , completing the information guidance
to obtain the spatial-temporal-guided sequence feature maps
F st :

F st = F p + ω st ϕv (F p ) × M st ,
(16)
where ω st is the 3D 1×1 convolutional layer use to restore the
feature dimensions. Through spatial-temporal high-frequency
information refinement, DSR guides the original modality in
effectively extracting spatial-temporal information.
F. Discussion
To further highlight the uniqueness and novelty of our
approach, we provide the following discussions focusing on
three key aspects: intermediate modality design, hierarchical
feature learning, and frequency-domain learning strategies.
Intermediate Modality Construction. Rather than constructing image-level intermediate modalities for the VVIReID task, STHF incorporates temporal modeling into the
construction of the intermediate modality, further enhancing
identity discriminability. As shown in Fig. 5(a), although the
Anaglyph data [9] alleviate modality discrepancy to a large extent, they lose critical discriminative information due to gener-

input sequences

2D FFT

3D FFT

P

xHL
LFC
Drop

2D iDHWT
Encoder

(a)

high-frequency detail loss

input feature maps

A

Conv

Conv

Conv

Conv

P

A

2D iFFT

(b)

3D Binary
HPF

3D iFFT
Encoder

(c)

Fig. 6. Different strategies in frequency-domain learning. (a) PHA [66] first
decomposes images into LFC and HFC using 2D Discrete Haar Wavelet
Transform (DHWT), then drops the LFC via a Low-Frequency Drop (LD)
module, and finally enhances spatial high-frequency information through
representation learning. (b) FRB [58] applies 2D FFT to decompose features into amplitude and phase components, and captures image-level global
dependencies via convolutional operations. (c) Our method applies 3D FFT
to the input sequences and removes spatial-temporal LFC using a 3D binary
high-pass filter, followed by representation learning to extract spatial-temporal
high-frequency information.

ating the intermediate modality from a single-frame image, ignoring the inherent spatial-temporal correlations within video
sequences. In contrast, the proposed ST-HPF constructs a
sequence-level intermediate modality by removing the spatialtemporal LFC from IR and RGB pedestrian sequences, which
not only reduces modality discrepancy but also enhances highfrequency details in both temporal and spatial dimensions,
as illustrated in Fig. 5(b). Therefore, constructing a highly
discriminative sequence-level intermediate modality is crucial
for the VVI-ReID task.
Hierarchical Representation Enhancement. STHF introduces two distinct feature interaction modules to enhance
representations from both shallow and deep embedding space.
First, the Shallow Detail Compensation (SDC) module leverages low-level features through 2D CNNs at the frame level
to compensate for local detail loss. Then, the Deep Semantic
Refinement (DSR) module employs high-level features via
3D CNNs at the sequence level to refine global semantic
representations. Through the combined effect of SDC and
DSR, STHF learns more comprehensive and discriminative
representations.
Spatial-Temporal Frequency Modeling. STHF introduces
a frequency-domain strategy well-suited to the VVI-ReID task.
As shown in Fig. 6, compared with capturing spatial highfrequency cues via coarse frequency separation through 2D
Discrete Haar Wavelet Transform (DHWT) [66] and modeling
global frequency information in the spatial domain using 2D
FFT [58], STHF employs 3D FFT combined with a 3D binary
filter to extract spatial-temporal high-frequency information,
thus preserving the complete frequency spectrum and further
enhancing identity discriminability.
G. Optimization
We combine cross-entropy loss and triplet loss to jointly
optimize our network for learning discriminative and modality-

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


codex
第四篇的方法闭环不是靠“频域听起来新”，而是要证明三件事：3D 高通比 2D 或小波更合适，浅层和深层的用法不同，空间和时间高频都要有。我现在读对应消融。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf' - | rg -n -C 4 'Ablation|ablation|Effectiveness|ST-HPF|SDC|DSR|frequency|high-frequency|Visualization|visual|Table III|Table IV|Table V|Table VI|Fig. 7|Fig. 8|Fig. 9|2D|3D'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
74-feature representations across video sequences in different
75-modalities. Existing methods that use an intermediate modality
76-to bridge the gap between visible (RGB) and infrared (IR)
77-sequences tend to be limited by high construction costs, loss of
78:high-frequency details, and lack of temporal cues. Moreover, they
79-typically focus on refining global representation using high-level
80-features, neglecting the enhancement of local details through
81-low-level features. To address these challenges, we propose
82-the novel Spatial-Temporal High-Frequency Learning (STHF)
83-framework, which constructs an appropriate intermediate
84-modality for the VVI-ReID task and alleviates the modality
85-gap via hierarchical feature enhancement. Specifically, we
86:introduce the Spatial-Temporal High-Pass Filter (ST-HPF),
87-which filters out spatial-temporal Low-Frequency Components
88:(LFC), preserving high-frequency details to construct an
89-intermediate modality at the sequence level. We then enhance
90-the local details with low-level features through the Shallow
91:Detail Compensation (SDC) module, which reduces local
92-noise interference. Finally, the Deep Semantic Refinement
93:(DSR) module refines the global representation by modeling
94:spatial-temporal high-frequency semantic associations using
95-high-level features. Extensive experiments demonstrate that our
96-method significantly outperforms state-of-the-art approaches on
97-the publicly available HITSZ-VCM and BUPTCampus datasets.
98-The code is available at https://github.com/TSC95720/STHF.
--
119-
120-Fig. 1. Comparison with existing intermediate modality-based methods and
121-the diagram of our method. (a) Existing VVI-ReID methods involve high costs
122-in constructing intermediate modality (i.e., Fake IR [8]), while also suffering
123:from the loss of high-frequency detailed information in the generated modality
124-(i.e., Anaglyph [9]). Furthermore, these methods rely solely on singleframe images when constructing intermediate modalities, neglecting temporal
125:modeling. (b) Our method leverages 3D FFT to extract discriminative spatialtemporal high-frequency details, constructing a sequence-level intermediate
126-modality. (c) Our method utilizes the intermediate modality to enhance the
127-local details of the original modality at shallow layers while guiding the
128-extraction of semantic information at deep layers.
129-
--
150-content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874
151-
152-IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
153-
154:2D Spatial
155-HPF
156-
157-(a) Remove Spatial LFC
158-
--
178-preserving identity information. Despite their promising performance, these methods still face notable limitations. Specifically, as shown in Fig. 1(a), the construction of intermediate
179-modalities is inappropriate due to the following reasons: 1)
180-High construction cost: generation-based method [8] demand
181-substantial computational resources and are inherently prone
182:to mode collapse [12], [13]; 2) Loss of high-frequency
183-detail: edge detection-based method [9] typically focus on
184-capturing abrupt intensity changes and may smooth out subtle
185:variations, leading to the loss of high-frequency detail; 3) Lack
186-of temporal cues: these methods relying on single-frame
187-construct intermediate modalities, neglecting that the temporal
188-domain also contains important discriminative information,
189:such as temporal high-frequency information. In addition, the
190-utilization of intermediate modalities is insufficient: Deep
191-feature learning follows a low-level to high-level paradigm, as
192-shown in Fig. 1(c), yet previous methods [8], [9] typically use
193-the high-level semantic features of the intermediate modality to
--
195-features that are crucial for capturing fine-grained local details.
196-These factors limit intermediate modality-based methods from
197-effectively alleviating the modality discrepancy and learning
198-discriminative feature representations.
199:Recently, transforming signals into the frequency domain
200-using the Fast Fourier Transform (FFT) has been widely
201-adopted in deep learning [14]–[17]. Compared with the spatial
202-domain, the discriminability of features is physically defined
203-
204-2
205-
206:in the frequency domain. As illustrated in Fig. 2, by removing
207-the Low-Frequency Components (LFC) from video pedestrian
208-sequences, the spatial High-Frequency Components (HFC)
209-preserve fine-grained structural details, while temporal HFC
210-capture motion variations, both of which provide distinct
--
212-a low computational complexity of O(N log N), making it
213-significantly more efficient in processing large-scale video
214-data compared to traditional spatial domain methods. Thus,
215-as shown in Fig. 1(b), learning spatial-temporal discriminative
216:information from a frequency-domain perspective is crucial
217-for the VVI-ReID task.
218-Based on the above analysis, in this paper, we propose
219-a Spatial-Temporal High-Frequency Learning (STHF) framework for the VVI-ReID task. The proposed STHF aims
220:to mitigate the modality discrepancy while learning spatialtemporal high-frequency information from low-level to highlevel features. To achieve this, we propose three key modules:
221:Spatial-Temporal High-Pass Filter (ST-HPF), Shallow Detail
222:Compensation (SDC) module, and Deep Semantic Refinement
223:(DSR) module. The proposed ST-HPF constructs a novel
224:sequence-level intermediate modality based on a 3D Fast
225-Fourier Transform (FFT) by explicitly leveraging spatialtemporal cues to enhance identity discriminability. Based on
226:this, the proposed SDC module enhances local structural
227-details with the low-level features by accurately modeling
228-spatial correspondence between the intermediate modality and
229:the original modality, while the proposed DSR module refines
230-global representations with the high-level features by establishing spatial-temporal semantic associations. To ensure effective
231-information interaction, we first eliminate the style information
232:of the original modality before applying the SDC and DSR
233:modules. By learning spatial-temporal high-frequency information, STHF effectively mitigates the modality discrepancy
234-and extracts discriminative spatial-temporal representations.
235-Our main contributions can be summarized as follows:
236-We propose a novel Spatial-Temporal High-Frequency
237-Learning (STHF) framework to exploit the potential
238:spatial-temporal high-frequency information for the VVIReID task.
239:• We propose a Spatial-Temporal High-Pass Filter (STHPF) based on 3D FFT to construct a novel sequencelevel intermediate modality, which jointly addresses
240-modality alignment and spatial-temporal feature learning.
241:• We propose a Shallow Detail Compensation (SDC) module to enhance local details with low-level features, and a
242:Deep Semantic Refinement (DSR) module to refine global
243-representations with high-level features.
244-• Experiments on HITSZ-VCM and BUPTCampus demonstrate that our method significantly outperforms stateof-the-art approaches, validating the effectiveness of the
245:proposed ST-HPF, SDC, and DSR.
246-•
247-
248-The remainder of this paper is organized as follows. Section
249-II reviews related work; Section III presents the proposed
--
281-for the information asymmetry in the HSV color space.
282-Metric learning-based methods aim to reduce the distance
283-between samples of the same identity across different modalities by designing appropriate feature metrics or loss functions.
284-HHGF [29] proposes the CMCC loss function to mine the invariance of global features by measuring mutual information in
285:images from two modalities. SDCL [30] effectively mitigates
286-the cross-modality discrepancy through the collaboration of
287-shallow and deep features. HOS-Net [31] proposes a modalityrange identity-center contrastive loss to reduce the distances
288-between the RGB, IR, and intermediate features. CPM [32]
289-introduces the closest permutation distance that is invariant to
--
328-modality discrepancies as style attacks and applied a Graph
329-Neural Network [53] to extract robust cross-modality representations by modeling intra- and inter-modality relations.
330-Recently, Transformers-based methods have demonstrated
331-great potential in VVI-ReID. CST [50] proposes a crossmodality spatial-temporal Transformer [54] that encapsulates
332:local pedestrian information into 3D tubes and facilitates
333-inter-frame interactions via message tokens. STIMM [51]
334-employs a temporal Transformer to capture comprehensive
335-motion patterns, thereby enhancing feature discriminability
336-across frames. Other methods, such as AuxNet [8], proposes
--
338-matching over time.
339-These methods have achieved significant success, fully
340-demonstrating the feasibility and effectiveness of the VVIReID task. However, they primarily focus on mitigating the
341-modality discrepancy and extracting spatial-temporal information in the spatial domain, while overlooking potential
342:solutions in the frequency domain.
343-C. Fourier Transform
344-The Fourier Transform has been increasingly applied in
345-deep learning for its ability to model global dependencies
346:in the frequency domain. Recent studies have further demonstrated its effectiveness across a range of tasks [14]–[16], [55].
347-For example, in medical image segmentation, FreMIM [14]
348-leverages Fast Fourier Transform to replace self-attention,
349-enabling efficient global information modeling. In human pose
350:estimation, FTCM [15] models the frequency and temporal
351-interactions between poses through separate feature mixing operations. In the area of low-light image enhancement, FourLLIE [56] explores the positive correlation between amplitude
352-magnitude and brightness magnitude. For high-quality image
353:deblurring, FFTformer [57] develops an efficient frequencybased self-attention solver, which reduces spatial and temporal complexity while improving efficiency and effectiveness.
354-
355-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
356-© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
357-but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
--
364-Conv
365-
366-Original data flow
367-
368:DSR
369-Module
370-
371-Conv
372-
373:3D Binary
374-HPF
375-
376-Block 1
377-
--
480-
481-Resnet
482-Layer
483-
484:(b) Deep Semantic Refinement (DSR) Module
485-MP
486-
487:(a) Shallow Detail Compensation (SDC) Module
488-
489-Common data flow
490-
491-LTri
492-
493-Block 4
494-
495:3D iFFT
496-
497-High-Frequency data flow
498-
499-LID
500-
501-FC
502-
503:SDC
504-Module
505-
506:3D FFT
507-
508-BN
509-
510:ST-HPF
511-
512-GAP
513-
514-Input sequences
--
521-
522-Matrix-wise
523-Multiplication
524-
525:2D Conv
526-
527-Element-wise
528-Addition
529-
530:2D Conv
531-
532-Element-wise
533-Multiplication
534-
535:3D Conv
536-
537:Fig. 3. The framework of STHF, consists of a Spatial-Temporal High-Pass Filter (ST-HPF), a Shallow Detail Compensation (SDC) module, and a Deep
538:Semantic Refinement (DSR) module. The ST-HPF first constructs a sequence-level intermediate modality through spatial-temporal high-pass filtering. Then
539:SDC compensates the original modality with high-frequency spatial information in the shallow layers of ResNet, while DSR guides the original modality to
540:extract spatial-temporal high-frequency information in the deep layers. The two information interaction modules operate under modality-irrelevant conditions
541:by the modality purifier. We perform SDC operations in blocks 1 – 2 of ResNet and DSR operations in blocks 3 – 4. For simplicity, only block 1 and block
542:3 are depicted in the figure to exemplify the integration of SDC and DSR modules.
543-
544:In face super-resolution, SFMNet [58] designs a spatialfrequency mutual learning network, where the frequency domain branch captures global dependency and the spatial domain branch extracts local dependency. The Fourier Transform
545-has also been widely adopted in person ReID. For instance,
546-FDMNet [59] learns modality-invariant features by enhancing modality-invariant amplitude components and suppressing
547-modality-specific components. FDNM [60] improves the discriminability of phase components by leveraging amplitude
548:guidance in the frequency domain, effectively learning robust
549-feature representations. FIDN [61] disentangles and utilizes
550:low-frequency and high-frequency spatial-temporal to capture
551-comprehensive and discriminative video representations.
552-Inspired by the success of these works, we propose a spatialtemporal high-pass filtering method for VVI-ReID to exploit
553:the potential spatial-temporal high-frequency information in
554-pedestrian sequences.
555-
556-Fourier Transform (FFT). Specifically, Given a 1D signal x[n]
557-of length N, then the 1D FFT is defined as:
--
564-(1)
565-
566-n=0
567-
568:where j is the imaginary unit. X[k] denotes the k-th frequency
569-component. Since the FFT is a linear transformation, the
570-original signal can be reconstructed using the inverse Fast
571-Fourier Transform (iFFT):
572-x[n] =
--
597-gap between RGB and IR pedestrian sequences. Previous
598-intermediate modality-based methods [8], [9] focused mainly
599-on constructing intermediate modalities in the spatial domain
600-to mitigate the modality discrepancy, while neglecting the
601:discriminative spatial-temporal high-frequency information,
602-which is crucial for the VVI-ReID task. Therefore, before
603-introducing STHF, we first review the basics of the Fast
604-
605-where k = (k1 , k2 , . . . , kd ) and n = (n1 , n2 , . . . , nd ) are the
606:frequency and spatial indices in d dimensions, respectively,
607-and N = (N1 , N2 , . . . , Nd ) denotes the size of each dimension.
608-For conciseness, the expression of the d-dimensional iFFT,
609-which can be directly derived from Eq. (2), is not explicitly
610-presented.
611:In the frequency domain, both spatial and temporal HFC
612-contain unique modality-invariant discriminative information
613-(e.g., edges, motion artifacts). Thus, by filtering out the spatialtemporal LFC from the RGB and IR pedestrian sequences,
614-
615-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
--
620-content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874
621-
622-IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY
623-
624:ST-HPF
625-
626-1
627-1
628-
--
641-
642-1
643-1
644-
645:3D iFFT
646-
647-X̂
648-Frequency Domain
649-
650-X
651-
652:3D Binary
653-High-Pass Filter
654-
655-5
656-
657:Algorithm 1: Process of proposed ST-HPF
658-Input: Pedestrian video sequence V of the original
659-s ; temporal
660:modality; spatial cutoff frequency fco
661-t
662:cutoff frequency fco .
663-Output: Filtered sequence I with enhanced
664:spatial-temporal high-frequency components.
665:Step.1: # 3D Fast Fourier Transform;
666-X = FFT (V ), X = FFT Shi f t(X);
667:Step.2: # Generate 3D Binary High-Pass Filter;
668-F = 1T ×H×W ;
669-tcrop = T2 , hcrop = H2 , wcrop = W2 ;
670-t ) : (t
671-t
--
678-s
679-s );
680-wrange = (wcrop − fco ) : (wcrop + fco
681-
682:3D FFT
683-
684-V
685-
686:Fig. 4. Spatial-Temporal High-Pass Filter (ST-HPF). The ST-HPF first maps
687:the original pedestrian sequence from the spatial domain to the frequency
688:domain using 3D FFT. Then, a 3D ideal high-pass filter is applied to remove
689-the LFC from the sequence. Finally, the sequence spectrum is mapped back to
690:the spatial domain by 3D iFFT, obtaining a spatial-temporal high-frequency
691:pedestrian sequence. Figures on the right are an example of applying ST-HPF.
692:In the frequency domain, the spatial-temporal LFC (small white regions inside
693-yellow boxes) are removed.
694-
695-identity-related features that are robust to modality variations
696-can be effectively extracted, enabling the construction of a
697-suitable intermediate modality for the VVI-ReID task. We will
698:elaborate on how to filter out low-frequency information while
699:preserving useful spatial-temporal high-frequency information
700:in ST-HPF.
701-B. Overview
702-As shown in Fig. 3, the proposed STHF mainly includes
703:a Spatial-Temporal High-Pass Filter (ST-HPF), a Shallow
704:Detail Compensation (SDC) module, and a Deep Semantic
705:Refinement (DSR) module. The STHF first utilizes the STHPF to generate spatial-temporal high-frequency pedestrian
706:sequences as an intermediate modality. Based on this, the SDC
707-module enhances local details of low-level features, and the
708:DSR module refines the global representation with high-level
709-features.
710-C. Spatial-Temporal High-Pass Filter
711-To remove the redundant LFC across both spatial and
712-temporal dimensions, thereby constructing a suitable intermediate modality for the VVI-ReID task, inspired by [62], we
713:introduce a Spatial-Temporal High-Pass Filter (ST-HPF) based
714:on 3D-FFT with the binary filtering strategy. Since the video
715-pedestrian sequence V has three dimensions T (temporal), H
716:(height), and W (width), the filter is also 3D with three cutoff
717:frequencies. In the centered 3D spectrum, low spatial frequencies lie near (kh , kw ) = (0, 0), and low temporal frequencies are
718-near kt = 0, as demonstrated in [63], [64]. Following this prior,
719-we preset the cutoff frequencies according the sequence size.
720-s handles both H and
721:Specifically, a spatial cutoff frequency fco
722-
723-F[trange , hrange , wrange ] = 0;
724-Step.3: # Spatial-Temporal High-Pass Filtering;
725-X̂ = F ⊙ X;
726:Step.4: # 3D inverse Fast Fourier Transform;
727-X̂ = iFFT Shi f t(X̂), I = iFFT (X̂);
728-
729-t handles
730:W dimension, while a temporal cutoff frequency fco
731-T dimension.
732-
733-To achieve this, as shown in Fig. 4, we first perform the
734:frequency domain projection to map the pedestrian sequence
735:to the frequency domain. Then, we perform the high-pass
736:filtering in the frequency domain to remove redundant LFC
737-of the pedestrian sequence. Finally, we conduct the spatial domain reconstruction to convert the spectrum back to the spatial
738:domain, obtaining the high-frequency pedestrian sequence.
739:Frequency Domain Projection. We first apply 3D FFT to
740-the pedestrian sequence V , transforming it into a spectrum.
741-Afterward, To facilitate spectrum analysis and filtering operations, we center the spectrum to ensure that the LFC is at the
742-center. The spectrum X is obtained as follows:
743-X = FFT Shi f t (F(V )),
744-
745-(4)
746-
747:where F (•) represents the 3D FFT, whose mathematical definition can be directly derived from Eq. (3), and FFT Shi f t (•)
748:denotes the operation that centers the frequency spectrum by
749-shifting the LFC to the center.
750:High-Pass Filtering. In the frequency domain, we introduce a 3D high-pass filter F, which removes redundant LFC
751-across both spatial and temporal dimensions through elementwise multiplication with the input sequence X. Therefore, the
752-resulting filtered spectrum X̂ can be obtained as follows:
753-X̂ = F ⊙ X,
754-
755-(5)
756-
757-where F is a binary matrix where 1 represents HFC and 0
758:represents LFC, constructed by retaining frequency components whose distance from the spectrum center exceeds the
759-predefined cutoff values while suppressing those below cutoff
760-
761-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
762-© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
--
793-where fco
794-co
795-and spatial dimensions, respectively. Meanwhile, hcrop = H2 ,
796-wcrop = W2 , and tcrop = T2 are the center coordinates used
797:to determine the center position of the frequency spectrum
798-region.
799-Spatial Domain Reconstruction. After the spatial-temporal
800:high-pass filtering in the frequency domain, we restore the
801-center position of the filtered spectrum and then map it back
802:to the spatial domain to obtain the high-frequency pedestrian
803-sequence I by iFFT. Which can be described as:
804-
805-(7)
806-I = F −1 iFFT Shi f t(X̂) ,
807-
808:where F −1 (•) denotes the 3D iFFT, whose mathematical definition can be derived from Eq. (2) and Eq. (3). iFFT Shi f t (•)
809:denotes the operation that reverses the frequency spectrum
810:shift. Algorithm 1 describes the procedure of ST-HPF in detail.
811-As shown in Fig. 4, by performing the above operations,
812:spatial-temporal low-frequency information in the pedestrian
813-sequence is suppressed, thereby enhancing both structural and
814-motion-based identity cues.
815:D. Shallow Detail Compensation (SDC) Module
816-Low-level features typically contain rich details by preserving textures, edges, and local patterns, but are susceptible to
817:background interference and modality-specific noise. Therefore, we propose a Shallow Detail Compensation (SDC) module that aims to enhance the high-frequency detail of original
818-modality with low-level features. The structure of the proposed
819:SDC is shown in Fig. 3(a), considering the interference of
820-style information in the original modality, we first perform
821-modality purify on the RGB and IR images. Afterward, we
822:sequentially perform fine-grained high-frequency channel and
823-spatial information compensation to enhance the local details
824-of original modality.
825-Modality Purify. To prevent style information from interfering with interaction, inspired by [23], we introduce a
826-Modality Purifier (MP) before performing SCD. It eliminates
--
845-
846-(FC) layers, δ (•) and σ (•) represent the ReLU and sigmoid
847-activation functions, respectively.
848-Channel Information Compensation. To enhance the
849:channel information, we extract the local detail affinity between the high-frequency feature maps f h ∈ RC×H×W and
850-the purified feature maps f p along the channel dimensions.
851-Specifically, we first calculate the channel-wise similarity
852-′
853-′
--
857-
858-M c = Fso f tmax ψq1 ( f p ) × ψk1 ( f h ) ,
859-(10)
860-where ψk1 ( f h ) and ψq1 ( f p ) represent the key and query embeddings to capture channel relationships, which are generated
861:through 2D 1 × 1 convolutional layers. Then, we combine the
862-value embedding ψv1 ( f h ) with the M c through a weighted sum
863-and add it to the f p , completing the information compensation
864-to obtain the channel-compensated feature maps f c :
865-
866-
867-f c = f p + ω c ψv1 ( f h ) × M c ,
868-(11)
869:where ω c is the 2D 1 × 1 convolutional layer used to restore
870-the feature dimensions.
871-Spatial Information Compensation. Furthermore, to enhance the spatial information, we compute the spatial-wise
872-similarity matrix M s ∈ RS×S between f c and f h using a similar
873-approach:
--
884-f cs = f c + ω s ψv2 ( f h ) × M s ,
885-(13)
886-where ω s is used to restore the feature dimensions and ψv1 ( f h )
887-represents the value embedding of the f h . Through channel
888:and spatial high-frequency information compensation, SDC
889:effectively integrates the rich high-frequency local details into
890-the original modality.
891:E. Deep Semantic Refinement (DSR) Module
892:High-level features are essential for abstract semantic understanding and global pattern recognition. To learn discriminative spatial-temporal high-frequency representations, we
893:propose a Deep Semantic Refinement (DSR) module that
894:aims to refine the global representation by modeling spatialtemporal high-frequency semantic associations using highlevel features. The structure of the proposed DSR is shown
895-in Fig. 3(b). Based on the previous analysis, we first perform
896-modality purify as well. Then, we perform spatial-temporal
897-refinement to refine the global representation.
898:Modality Purify. Similar to SDC, we first utilize a Modality
899-Purifier to eliminate identity-unrelated style information from
900-the original modality sequence feature maps F ∈ RT ×C×H×W ,
901-and the sequence feature maps after purification are denoted
902-as F p :
--
914-
915-7
916-
917-input images
918:2D DHWT
919-
920-}
921-}
922:3D FFT
923-
924-Binary
925:+ 3DFilter
926-
927-}
928-}
929-
--
940-(b) Our method
941-
942-Fig. 5. Comparison of intermediate modality construction methods. (a)
943-Existing VVI-ReID method [9] constructs the intermediate modality based on
944:single-frame images and suffers from the loss of high-frequency detail information. (b) Our method constructs the intermediate modality at the sequence
945:level, enhancing spatial-temporal high-frequency discriminative information.
946-
947-where the operations of MP (•) follow the same principles as
948-described in Eq. (8).
949-Spatial-Temporal Refinement. To establish global semantic associations, we extract the spatial-temporal affinity
950:between the high-frequency sequence feature maps F h ∈
951-RT ×C×H×W and F p . We first calculate the spatial-temporal
952-similarity matrix M st ∈ RST ×ST :
953-
954-
955-M st = Fso f tmax ϕq (F h ) × ϕk (F p ) ,
956-(15)
957:where ϕq (F h ) and ϕk (F p ) represent the key and query embeddings to mine spatial-temporal associations, which are generated through 3D 1 × 1 convolutional layers. Next, we combine
958-the value embedding ϕv (F p ) with the M st through a weighted
959-sum and add it to the F p , completing the information guidance
960-to obtain the spatial-temporal-guided sequence feature maps
961-F st :
962-
963-F st = F p + ω st ϕv (F p ) × M st ,
964-(16)
965:where ω st is the 3D 1×1 convolutional layer use to restore the
966:feature dimensions. Through spatial-temporal high-frequency
967:information refinement, DSR guides the original modality in
968-effectively extracting spatial-temporal information.
969-F. Discussion
970-To further highlight the uniqueness and novelty of our
971-approach, we provide the following discussions focusing on
972-three key aspects: intermediate modality design, hierarchical
973:feature learning, and frequency-domain learning strategies.
974-Intermediate Modality Construction. Rather than constructing image-level intermediate modalities for the VVIReID task, STHF incorporates temporal modeling into the
975-construction of the intermediate modality, further enhancing
976-identity discriminability. As shown in Fig. 5(a), although the
977-Anaglyph data [9] alleviate modality discrepancy to a large extent, they lose critical discriminative information due to gener-
978-
979-input sequences
980-
981:2D FFT
982-
983:3D FFT
984-
985-P
986-
987-xHL
988-LFC
989-Drop
990-
991:2D iDHWT
992-Encoder
993-
994-(a)
995-
996:high-frequency detail loss
997-
998-input feature maps
999-
1000-A
--
1010-P
1011-
1012-A
1013-
1014:2D iFFT
1015-
1016-(b)
1017-
1018:3D Binary
1019-HPF
1020-
1021:3D iFFT
1022-Encoder
1023-
1024-(c)
1025-
1026:Fig. 6. Different strategies in frequency-domain learning. (a) PHA [66] first
1027:decomposes images into LFC and HFC using 2D Discrete Haar Wavelet
1028-Transform (DHWT), then drops the LFC via a Low-Frequency Drop (LD)
1029:module, and finally enhances spatial high-frequency information through
1030:representation learning. (b) FRB [58] applies 2D FFT to decompose features into amplitude and phase components, and captures image-level global
1031:dependencies via convolutional operations. (c) Our method applies 3D FFT
1032:to the input sequences and removes spatial-temporal LFC using a 3D binary
1033-high-pass filter, followed by representation learning to extract spatial-temporal
1034:high-frequency information.
1035-
1036-ating the intermediate modality from a single-frame image, ignoring the inherent spatial-temporal correlations within video
1037:sequences. In contrast, the proposed ST-HPF constructs a
1038-sequence-level intermediate modality by removing the spatialtemporal LFC from IR and RGB pedestrian sequences, which
1039:not only reduces modality discrepancy but also enhances highfrequency details in both temporal and spatial dimensions,
1040-as illustrated in Fig. 5(b). Therefore, constructing a highly
1041-discriminative sequence-level intermediate modality is crucial
1042-for the VVI-ReID task.
1043-Hierarchical Representation Enhancement. STHF introduces two distinct feature interaction modules to enhance
1044-representations from both shallow and deep embedding space.
1045:First, the Shallow Detail Compensation (SDC) module leverages low-level features through 2D CNNs at the frame level
1046-to compensate for local detail loss. Then, the Deep Semantic
1047:Refinement (DSR) module employs high-level features via
1048:3D CNNs at the sequence level to refine global semantic
1049:representations. Through the combined effect of SDC and
1050:DSR, STHF learns more comprehensive and discriminative
1051-representations.
1052-Spatial-Temporal Frequency Modeling. STHF introduces
1053:a frequency-domain strategy well-suited to the VVI-ReID task.
1054:As shown in Fig. 6, compared with capturing spatial highfrequency cues via coarse frequency separation through 2D
1055-Discrete Haar Wavelet Transform (DHWT) [66] and modeling
1056:global frequency information in the spatial domain using 2D
1057:FFT [58], STHF employs 3D FFT combined with a 3D binary
1058:filter to extract spatial-temporal high-frequency information,
1059:thus preserving the complete frequency spectrum and further
1060-enhancing identity discriminability.
1061-G. Optimization
1062-We combine cross-entropy loss and triplet loss to jointly
1063-optimize our network for learning discriminative and modality-
--
1349-where λ1 and λ2 are the hyper-parameters, LID is the crossentropy loss and LTri is the triplet loss. During training, LID
1350-int and Lint
1351-and LTri work on the RGB and IR data, while LID
1352-Tri
1353:work on high-frequency data.
1354-IV. E XPERIMENTS
1355-A. Datasets and Evaluation Metrics
1356-HITSZ-VCM [7] comprises 251452 RGB images and 211807
1357-IR images from 927 identities captured across 12 cameras.
--
1497-to enhance the input sequence. We trained our network with
1498-an initial learning rate set to 3.5e-4 and adopted the Adam
1499-optimizer with the cosine learning rate decay strategy [77] to
1500-adjust the learning rate dynamically. Each training is run over
1501:for 200 epochs, and 150 epochs for ablation studies. We set
1502-the sequence length to 6 and the batch size to 16 (each minibatch includes 4 IDs and each ID includes 4 video clips of 2
1503-t and
1504:modalities), respectively. For the ST-HPF, we set the fco
1505-s
1506-fco to 2 and 10, respectively.
1507-
1508-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
--
1516-
1517-9
1518-
1519-TABLE III
1520:A BLATION STUDY OF THE COMPONENTS IN STHF ON HITSZ-VCM. “B”: BASELINE . “SDC”: S HALLOW D ETAIL C OMPENSATION . “DSR”: D EEP
1521-S EMANTIC R EFINEMENT.
1522-
1523-Component
1524-
--
1527-Visible to Infrared
1528-
1529-B
1530-
1531:SDC
1532-
1533:DSR
1534-
1535-R@1
1536-
1537-mAP
--
1575-54.5(+6.4)
1576-57.4(+9.3)
1577-
1578-TABLE IV
1579:A BLATION S TUDY ON THE MP M ODULE WITH SDC AND DSR. “B”: BASELINE . “SDC”: S HALLOW D ETAIL C OMPENSATION . “DSR”: D EEP S EMANTIC
1580-R EFINEMENT. “MP”: M ODALITY P URIFY.
1581-
1582-Component
1583-
1584-Infrared to Visible
1585-
1586-Visible to Infrared
1587-
1588:B+SDC
1589-
1590:B+DSR
1591-
1592-MP
1593-
1594-R@1
--
1665-intermediate modality is more suitable for the VVI-ReID task
1666-and has been effectively exploited.
1667-
1668-TABLE V
1669:T HE INFLUENCE OF WHICH BLOCK OF R ES N ET-50 TO PLUG THE SDC
1670:AND DSR MODULES . O N INFRARED TO VISIBLE RETRIEVAL .
1671-
1672:SDC
1673-
1674-Resnet Layer
1675-Block 0
1676-Block 1
1677-Block 2
1678-Block 3
1679-Block 4
1680-
1681:DSR
1682-
1683-R@1
1684-
1685-mAP
--
1714-
1715-TABLE VI
1716-A BLATION S TUDY ON THE I NFLUENCE OF F ILTERING D IMENSION .
1717-S-HPF: S PATIAL H IGH -PASS F ILTER ; T-HPF: T EMPORAL H IGH -PASS
1718:F ILTER ; ST-HPF: S PATIAL -T EMPORAL H IGH -PASS F ILTER .
1719-
1720-Filter
1721-S-HPF
1722-T-HPF
1723:ST-HPF
1724-
1725-Infrared to Visible
1726-
1727-Visible to Infrared
--
1759-Rank-1 in the Visible-to-Infrared setting. Specifically, mAP
1760-is improved by 6.6%, while Rank-1 is increased by 8.3%,
1761-respectively. These results demonstrate the superiority of our
1762-intermediate modality compared to Fake IR [8] due to its
1763:ability to capture spatial-temporal high-frequency information.
1764-
1765-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
1766-© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
1767-but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
--
1773-
1774-10
1775-
1776-TABLE VII
1777:C OMPARISON WITH FREQUENCY- DOMAIN METHODS . “DSR”: D EEP S EMANTIC R EFINEMENT. “PHA”: PATCH WISE H IGH - FREQUENCY AUGMENTATION .
1778:“FRB”: F REQUENCY B LOCK . “ST-HPF”: S PATIAL -T EMPORAL H IGH -PASS F ILTER .
1779-
1780-Infrared to Visible
1781-
1782-Visible to Infrared
--
1810-54.5(+6.4)
1811-
1812-Settings
1813-Baseline (B)
1814:B+DSR+PHA
1815:B+DSR+FRB
1816:B+DSR+ST-HPF
1817-
1818-Infrared to Visible
1819-66.4
1820-
--
1856-PHA [66]
1857-SFMNet [58]
1858-STHF (ours)
1859-
1860:2D DHWT
1861:2D FFT
1862:3D FFT
1863-
1864-65.3
1865-
1866-71.5
--
1878-
1879-(c) Spatial-Temporal HPF
1880-
1881-s and f t .
1882:Fig. 7. Effect of cutoff frequencies fco
1883-co
1884-
1885:Fig. 8. Effect of the hyper-parameters λ1 and λ2 .
1886-
1887:D. Ablation Study
1888:To verify the effectiveness of each component in the proposed STHF, we conducted ablation studies on the HITSZVCM dataset. The experimental results are shown in Table III.
1889-The baseline method employs a two-stream ResNet50 network
1890-optimized using the identity loss (LID ) and the triplet loss
1891:(LTri ). STHF improves the baseline by an SDC module for
1892:detail compensation with high-frequency low-level features
1893:and a DSR module for semantic refinement with high-level
1894-features.
1895:Effectiveness of SDC and DSR. To verify the effectiveness
1896:of SDC and DSR modules, we integrated them into the
1897:baseline sequentially. As shown in Table III, in the Infraredto-Visible evaluation, the Rank-1 accuracy improved from
1898-64.3% to 65.1% (+0.8%), and the mAP increased from 47.0%
1899:to 49.3% (+2.3%) due to the effect of SDC, while DSR
1900-improved the Rank-1 accuracy by 3.8% and the mAP by
1901:5.7%, respectively. Similarly, in the Visible-to-Infrared evaluation, both SDC and DSR improved the Rank-1 accuracy
1902-and mAP to varying degrees. These results demonstrate that
1903:the SDC and DSR modules effectively enhance local details
1904:and refine global semantics. It can be observed that the DSR
1905-module contributes the most to performance improvement,
1906:which demonstrates the superiority of spatial-temporal highfrequency information in semantic information mining. Moreover, when both the SCD and DSR modules are integrated into
1907-the network, the Rank-1 accuracy increases by 6.3% and the
1908-
1909-mAP improves by 8.2% in the Infrared-to-Visible evaluation,
1910-while the Rank-1 accuracy and mAP increase by 5.6% and
1911-9.3%, respectively, in the Visible-to-Infrared evaluation. This
1912:demonstrates that the combined effect of SCD and DSR
1913-enables the model to learn comprehensive and discriminative
1914-feature representations.
1915:Effectiveness of MP on Feature Interaction. To verify that
1916-the MP module preliminarily alleviates modality discrepancies
1917-and thereby facilitates effective feature interaction within the
1918:SDC and DSR modules, we conducted additional ablation
1919-experiments by removing MP from both components. As
1920:shown in Table IV, in the Infrared-to-Visible evaluation, introducing the MP module into the SDC leads to an improvement
1921-in Rank-1 accuracy from 64.9% to 65.1% (+0.2%) and an
1922-increase in mAP from 47.9% to 49.3% (+1.4%). Similarly,
1923:adding MP to the DSR module improves Rank-1 accuracy
1924-from 66.6% to 68.1% (+1.5%) and mAP from 50.9% to 52.7%
1925-(+1.8%). In the Visible-to-Infrared setting, the MP module
1926:similarly enhances both SDC and DSR performance in terms
1927-of Rank-1 and mAP, with improvements of 1.5%/1.3% for
1928:SDC and 1.6%/1.7% for DSR, respectively. These results confirm that eliminating style information via MP promotes more
1929-effective feature interaction and contributes to performance
1930-improvement.
1931:Impact of Insertion Depth for SDC and DSR. To verify
1932:SDC targets low-level detail enhancement, while DSR focuses
1933-
1934-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
1935-© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
1936-but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
--
1973-Retrieval results from Rank@1 to Rank@10
1974-
1975-Ours
1976-
1977:Fig. 9. Visualization of pedestrian search results, where B/L represents the baseline method.
1978-
1979-IR
1980-
1981-1
--
1995-Fig. 10. Comparison of Cross-Modality Feature Distance Distributions
1996-between the baseline method and STHF.
1997-
1998-on high-level semantic refinement, we examined the impact
1999:of integrating the SDC and DSR modules at different blocks
2000-of ResNet-50. Following common practice in representation
2001-learning [78], we consider blocks 1–2 of ResNet-50 as shallow
2002-layers that capture low-level textures and edge details, while
2003:blocks 3–4 are regarded as deep layers responsible for semantic abstraction [79]. As shown in Table V, SDC performs better
2004:when integrated into block 1 of the ResNet layers, while DSR
2005-performs better when integrated into block 3. Specifically,
2006:when applied at block 1 of the ResNet layers, in the Infraredto-Visible evaluation, the SDC module outperforms DSR by
2007-0.2% in Rank-1 accuracy and 0.3% in mAP. In contrast, at
2008:block 3 of the ResNet layers, the DSR module surpasses
2009:SDC by 3.4% in Rank-1 accuracy and 2.9% in mAP. These
2010-results demonstrate that low-level features contain detailed information suitable for information compensation, while highlevel features extract semantic category information suitable
2011-for information guidance, aligning with our analysis in the
2012-previous sections.
2013-Influence of Filtering Strategy. To verify the effectiveness
2014-of spatial-temporal high-pass filtering in the VVI-ReID task,
2015:we conducted ablation experiments by constructing intermediate modalities using different filtering strategies. The cutoff
2016-s = 10 for spatial filtering and f t = 2
2017-frequencies were set to fco
2018-co
2019:for temporal filtering. As shown in Table VI, in the Infraredto-Visible setting, ST-HPF outperforms S-HPF and T-HPF by
2020-3.1% and 3.6% in Rank-1 accuracy, and by 3.1% and 3.3% in
2021-mAP, respectively. A similar trend is observed in the Visible-
2022-
2023-Fig. 11. Comparison of CAM between the baseline method and STHF.
2024-
2025:to-Infrared setting, where ST-HPF achieves improvements of
2026-0.9% and 1.5% in Rank-1, and 2.7% and 2.8% in mAP over
2027-S-HPF and T-HPF, respectively. These results confirm that
2028:spatial-temporal filtering captures complementary frequency
2029-cues, enabling more discriminative intermediate representations than single-dimensional approaches.
2030:Comparison of different frequency-domain methods. To
2031:evaluate the impact of different frequency learning strategies, we replaced ST-HPF with Patch wise High-frequency
2032-Augmentation (PHA) [66] and Frequency Block (FRB) [58]
2033:while leveraging the DSR module to learn frequency-domain
2034:information. Specifically, we directly replace the original STHPF with PHA, using the HFC-enhanced sequence as the intermediate modality. As for FRB, we utilize global frequencydomain features to replace the intermediate modality representations, enabling spatial-frequency mutual learning. As shown
2035:in Table VII, FRB achieves better average performance than
2036:PHA, indicating that FFT-based frequency modeling provides
2037:a more complete and expressive frequency representation than
2038:DHWT. Furthermore, ST-HPF integrates 3D FFT to jointly
2039-model HFC in both spatial and temporal dimensions, achieving
2040-the best results across all metrics. These results demonstrate
2041:the effectiveness of spatial-temporal high-frequency modeling
2042-in enhancing feature representation and alleviating the modality discrepancy in the VVI-ReID task.
2043-
2044-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
2045-© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
--
2069-the preservation of spatial and temporal details in intermediate
2070-modalities. According to Section III-C, the valid ranges of
2071-these frequencies are defined as follows: the temporal cutoff
2072-t varies within [1, T /2), and the spatial cutoff
2073:frequency fco
2074-s
2075:frequency fco lies in the range [1, min(H/2,W /2)]. To analyze
2076-their impact, we evaluate the performance of STHF under three
2077:different configurations, as illustrated in Fig. 7.
2078-s in S-HPF. We first investigate how differEffect of fco
2079-ent spatial cutoff frequencies affect performance by varying
2080:s ∈ {5, 10, 15, 20}. As shown in Fig. 7(a), the best result
2081-fco
2082-s = 10, indicating that moderate spatial
2083-is obtained when fco
2084-filtering effectively preserves important details while reducing
--
2087-Effect of fco
2088-t ∈ {1, 2}. As shown in
2089-temporal cutoff frequencies, with fco
2090-t = 2,
2091:Fig. 7(b), the model achieves peak performance when fco
2092-suggesting that slightly stronger temporal filtering improves
2093-dynamic feature modeling.
2094:s and f t in ST-HPF. Finally, we explore
2095-Joint effect of fco
2096-co
2097-joint spatial-temporal configurations, starting from the bests = 10, f t = 2), and testing variperforming combination ( fco
2098-co
2099:ants with stronger or weaker filtering. As shown in Fig. 7(c),
2100-s = 15, f t = 2) results in
2101-the strongest filtering setting ( fco
2102-co
2103-significant performance degradation, while the original setting
2104-maintains the highest accuracy. These results demonstrate
2105-that weak filtering may provide limited discriminative cues,
2106-while excessive filtering could suppress essential structural
2107-semantics. Based on these observations, we empirically set
2108:s = 10 and f t = 2 for ST-HPF to ensure balanced spatialfco
2109-co
2110-temporal HFC preservation.
2111-The hyperparameter λ1 controls the contribution of the
2112-cross-entropy loss from the intermediate modality, which
2113-promotes identity classification by supervising the predicted
2114:probability distributions. As shown in Fig. 8, we evaluate the
2115-performance of STHF across different values of λ1 within the
2116-range [0.2, 2]. According to the results, we empirically set
2117-λ1 = 1 for optimal performance.
2118-The hyperparameter λ2 balances the influence of the
2119-triplet loss from the intermediate modality. This loss optimizes
2120-the feature space through metric learning to enhance feature
2121:discriminability. As shown in Fig. 8, we report the performance of STHF under different parameter configurations of
2122-λ2 , within the same range as λ1 . Based on the experiment
2123-results, we fix λ2 = 1.
2124:F. Visualization
2125-To demonstrate the effectiveness of the proposed STHF, we
2126:visualized and compared the retrieval results, cross-modality
2127-
2128-12
2129-
2130-feature distance distributions, and feature focus region between
2131-the baseline method and STHF.
2132:Retrieval Results. We visualized the ReID results of the
2133-baseline method and STHF on the Infrared-to-Visible and
2134:Visible-to-Infrared pedestrian retrieval tasks using the HITSZVCM dataset. As shown in Fig. 9, given a query pedestrian
2135-sequence, we retrieved the top 10 most similar pedestrian
2136-sequences, where the green boxes indicate the correct retrieved
2137:results and the red boxes denote incorrect ones. The visualization results show that the proposed STHF outperforms the
2138-baseline method in complex scenarios. For example, in the
2139-Infrared-to-Visible retrieval, Query 1 has low visibility, and
2140-Query 2 is affected by occlusion in the Visible-to-Infrared
2141-retrieval, the performance of the baseline method is impacted
2142-by these factors, whereas STHF exhibits stronger robustness.
2143:Feature Distance Distributions. We visualized the feature
2144-distance distributions of the intra-class and inter-class samples
2145-between the baseline method and STHF as shown in Fig. 10(a)
2146-and Fig. 10(b). It can be distinctly observed that the mean distance between the intra-class and inter-class of STHF is larger
2147-than that of the baseline method (i.e., δ2 > δ1 ), indicating
2148:that spatial-temporal high-frequency information effectively
2149-improves the discriminability between different classes.
2150-Feature Focus Region. We compared class activation maps
2151-(CAMs) [80] between the baseline method and STHF. As
2152-shown in Fig. 11, the proposed STHF effectively mitigates
--
2157-more susceptible to background interference.
2158-
2159-V. C ONCLUSION
2160-In this paper, we propose a novel framework named STHF,
2161:which extracts discriminative spatial-temporal cues while mitigating modality differences from a frequency domain perspective. Specifically, we propose a novel sequence-level intermediate modality based on 3D FFT: Spatial-Temporal HighFrequency Information, which enhances structural and motionbased identity cues. Furthermore, we design two feature
2162:interaction modules: a Shallow Detail Compensation (SDC)
2163-module in shallow layers to enhance local details, and a
2164:Deep Semantic Refinement (DSR) module in deep layers
2165-to refine global representation. Experimental results on the
2166-public HITSZ-VCM and BUPTCampus datasets demonstrate
2167-the effectiveness and superiority of the proposed STHF.
2168:Moreover, although Spatial-Temporal High-Frequency Information mitigates modality discrepancy, a fixed cutoff frequency introduces redundancy for some identities and overfiltering for others, reducing discriminability. Besides, STHF
2169-incurs high training costs due to an additional network to
2170:learn high-frequency information. In future work, we aim to
2171-develop a method that can adaptively select the appropriate
2172:cutoff frequency for each identity to achieve better filtering
2173-performance.
2174-
2175-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
2176-© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
--
2231-collaborative module for efficient 3d human pose estimation in video,”
2232-IEEE Transactions on Circuits and Systems for Video Technology,
2233-vol. 34, no. 2, pp. 911–923, 2023.
2234-[16] M. Zhou, J. Huang, K. Yan, H. Yu, X. Fu, A. Liu, X. Wei, and F. Zhao,
2235:“Spatial-frequency domain information integration for pan-sharpening,”
2236-in European conference on computer vision. Springer, 2022, pp. 274–
2237-291.
2238-[17] J. Zhang, K. Cao, K. Yan, Y. Lin, X. He, Y. Wang, R. Li, C. Xie,
2239-J. Zhang, and M. Zhou, “Frequency decoupled domain-irrelevant feature learning for pan-sharpening,” IEEE Transactions on Circuits and
--
2368-[55] Z. Qin, P. Zhang, F. Wu, and X. Li, “Fcanet: Frequency channel attention
2369-networks,” in Proceedings of the IEEE/CVF international conference on
2370-computer vision, 2021, pp. 783–792.
2371-[56] C. Wang, H. Wu, and Z. Jin, “Fourllie: Boosting low-light image
2372:enhancement by fourier frequency information,” in Proceedings of the
2373-31st ACM International Conference on Multimedia, 2023, pp. 7459–
2374-7469.
2375:[57] L. Kong, J. Dong, J. Ge, M. Li, and J. Pan, “Efficient frequency domainbased transformers for high-quality image deblurring,” in Proceedings of
2376-the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2377-2023, pp. 5886–5895.
2378:[58] C. Wang, J. Jiang, Z. Zhong, and X. Liu, “Spatial-frequency mutual
2379-learning for face super-resolution,” in Proceedings of the IEEE/CVF
2380-Conference on Computer Vision and Pattern Recognition, 2023, pp.
2381-22 356–22 366.
2382-[59] Y. Li, T. Zhang, and Y. Zhang, “Frequency domain modality-invariant
--
2391-[62] J. Kim, T. Kim, M. Shim, D. Han, D. Wee, and J. Kim, “Frequency
2392-selective augmentation for video representation learning,” in Proceedings
2393-of the AAAI Conference on Artificial Intelligence, vol. 37, no. 1, 2023,
2394-pp. 1124–1132.
2395:[63] H. Wang, X. Wu, Z. Huang, and E. P. Xing, “High-frequency component
2396-helps explain the generalization of convolutional neural networks,” in
2397-Advances in Neural Information Processing Systems, 2020, pp. 8684–
2398-8694.
2399-
--
2405-14 383–14 392.
2406-[65] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in
2407-Proceedings of the IEEE conference on computer vision and pattern
2408-recognition, 2018, pp. 7132–7141.
2409:[66] G. Zhang, Y. Zhang, T. Zhang, B. Li, and S. Pu, “Pha: Patch-wise highfrequency augmentation for transformer-based person re-identification,”
2410-in Proceedings of the IEEE/CVF conference on computer vision and
2411-pattern recognition, 2023, pp. 14 133–14 142.
2412-[67] M. Ye, J. Shen, D. J. Crandall, L. Shao, and J. Luo, “Dynamic
2413-dual-attentive aggregation learning for visible-infrared person reidentification,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVII 16.
--
2479-Kunming University of Science and Technology,
2480-Kunming, China, in 2022. He is currently pursuing
2481-the Ph.D. degree with the School of Information and
2482-Control Engineering, China University of Mining
2483:and Technology. His current research interests include human pose estimation, 3D computer vision.
2484-
2485-Neng Dong is currently a Ph.D. candidate with
2486-the School of Computer Science and Engineering
2487-at Nanjing University of Science and Technology.

codex
这篇的证据闭环很完整：它有组件消融、模块插入深度、只做空间高通或只做时间高通、和已有频域方法替换对照，还有距离分布和注意图。为了避免数值遗漏，我把消融表页完整读出来。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 11 'Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

9

TABLE III
A BLATION STUDY OF THE COMPONENTS IN STHF ON HITSZ-VCM. “B”: BASELINE . “SDC”: S HALLOW D ETAIL C OMPENSATION . “DSR”: D EEP
S EMANTIC R EFINEMENT.

Component

Infrared to Visible

Visible to Infrared

B

SDC

DSR

R@1

mAP

R@1

mAP

✓
✓
✓
✓

✗
✓
✗
✓

✗
✗
✓
✓

64.3
65.1(+0.8)
68.1(+3.8)
70.6(+6.3)

47.0
49.3(+2.3)
52.7(+5.7)
55.2(+8.2)

67.2
69.3(+2.1)
71.9(+4.7)
72.8(+5.6)

48.1
50.9(+2.8)
54.5(+6.4)
57.4(+9.3)

TABLE IV
A BLATION S TUDY ON THE MP M ODULE WITH SDC AND DSR. “B”: BASELINE . “SDC”: S HALLOW D ETAIL C OMPENSATION . “DSR”: D EEP S EMANTIC
R EFINEMENT. “MP”: M ODALITY P URIFY.

Component

Infrared to Visible

Visible to Infrared

B+SDC

B+DSR

MP

R@1

mAP

R@1

mAP

✓
✗
✓
✗

✗
✓
✗
✓

✗
✗
✓
✓

64.9
66.6
65.1(+0.2)
68.1(+1.5)

47.9
50.9
49.3(+1.4)
52.7(+1.8)

67.8
70.3
69.3(+1.5)
71.9(+1.6)

49.6
52.8
50.9(+1.3)
54.5(+1.7)

C. Comparison with State-of-the-Art Methods
To verify the performance of STHF, we compared it
with recent state-of-the-art methods on the HITSZ-VCM
and BUPTCampus datasets, including both image-based and
video-based VI-ReID approaches. Note that we did not introduce additional networks to handle the temporal information of the sequence, ensuring a fair comparison with
image-based cross-modality ReID methods. Specifically, the
methods we compared include: Lba [68], MPANet [23],
VSD [69], CAJ [70], SEFL [71], MITML [7], IBAN [9],
SADSTRM [72], SAADG [10], CST [50], AlignGAN [73],
DDAG [67], AGW [1], MMN [74], DEEN [27], DART [75],
MITML [7], AuxNet [8].
Evaluation on HITSZ-VCM Dataset. We first evaluated
the performance of the proposed STHF on the HITSZ-VCM
benchmark. The comparison results are shown in Table I,
we can observe that the performance of MITML, IBAN,
SADSTRM, SAADG, CST, and STHF significantly surpasses
that of the VI-ReID method, demonstrating both the feasibility of the VVI-ReID task and the effectiveness of these
methods. Furthermore, STHF significantly improves performance compared to IBAN and CST. Specifically, in terms of
mAP, STHF achieves a 7.6% improvement in the Visible-toInfrared setting and 7.4% in the Infrared-to-Visible setting
over IBAN, while outperforming CST by 5.6% and 5.0%
in the respective settings. In terms of Rank-1, STHF shows
a 3.9% improvement in the Visible-to-Infrared setting and
a 5.4% improvement in the Infrared-to-Visible setting over
IBAN. Compared to CST, it achieves improvements of 1.0%
and 0.9% in the same settings. These results further validate
the effectiveness of STHF. In addition, it can be observed that
STHF improves the performance by a large margin compared
to IBAN, which demonstrates that our proposed sequence-level
intermediate modality is more suitable for the VVI-ReID task
and has been effectively exploited.

TABLE V
T HE INFLUENCE OF WHICH BLOCK OF R ES N ET-50 TO PLUG THE SDC
AND DSR MODULES . O N INFRARED TO VISIBLE RETRIEVAL .

SDC

Resnet Layer
Block 0
Block 1
Block 2
Block 3
Block 4

DSR

R@1

mAP

R@1

mAP

63.6
65.8
66.0
65.1
66.1

46.8
48.9
49.2
49.1
50.1

62.2
65.5
66.9
68.5
66.1

45.6
48.7
50.3
52.0
50.3

TABLE VI
A BLATION S TUDY ON THE I NFLUENCE OF F ILTERING D IMENSION .
S-HPF: S PATIAL H IGH -PASS F ILTER ; T-HPF: T EMPORAL H IGH -PASS
F ILTER ; ST-HPF: S PATIAL -T EMPORAL H IGH -PASS F ILTER .

Filter
S-HPF
T-HPF
ST-HPF

Infrared to Visible

Visible to Infrared

R@1

mAP

R@1

mAP

67.5
67
70.6

52.1
51.9
55.2

71.9
71.3
72.8

54.7
54.6
57.4

Evaluation on BUPTCampus Dataset. To verify the generalizability of the proposed STHF, we also evaluated its performance on the BUPTCampus benchmark. As shown in Table II,
our method outperforms current state-of-the-art approaches
with a sequence length of 6. Note that when comparing with
AuxNet, for a fair comparison, we evaluate the version of
AuxNet without using the re-ranking algorithm. Furthermore,
compared to AuxNet, STHF significantly improves mAP and
Rank-1 in the Visible-to-Infrared setting. Specifically, mAP
is improved by 6.6%, while Rank-1 is increased by 8.3%,
respectively. These results demonstrate the superiority of our
intermediate modality compared to Fake IR [8] due to its
ability to capture spatial-temporal high-frequency information.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

10

TABLE VII
C OMPARISON WITH FREQUENCY- DOMAIN METHODS . “DSR”: D EEP S EMANTIC R EFINEMENT. “PHA”: PATCH WISE H IGH - FREQUENCY AUGMENTATION .
“FRB”: F REQUENCY B LOCK . “ST-HPF”: S PATIAL -T EMPORAL H IGH -PASS F ILTER .

Infrared to Visible

Visible to Infrared

R@1

mAP

R@1

mAP

64.3
66.6(+2.3)
67.2(+2.9)
68.1(+3.8)

47.0
50.6(+3.6)
51.7(+4.7)
52.7(+5.7)

67.2
70.3(+3.1)
69.9(+2.7)
71.9(+4.7)

48.1
52.4(+4.3)
53.1(+5.0)
54.5(+6.4)

Settings
Baseline (B)
B+DSR+PHA
B+DSR+FRB
B+DSR+ST-HPF

Infrared to Visible
66.4

71.2

67.5

71.9

66.7

70.9

(a) Spatial HPF

67

70.6
69

67

69.7

69.1

68.1

72

72.1

71.6

Methods

Principle

PHA [66]
SFMNet [58]
STHF (ours)

2D DHWT
2D FFT
3D FFT

65.3

71.5

Visible to Infrared
70.1

71.3

(b) Temporal HPF

72.8

72.5

(c) Spatial-Temporal HPF

s and f t .
Fig. 7. Effect of cutoff frequencies fco
co

Fig. 8. Effect of the hyper-parameters λ1 and λ2 .

D. Ablation Study
To verify the effectiveness of each component in the proposed STHF, we conducted ablation studies on the HITSZVCM dataset. The experimental results are shown in Table III.
The baseline method employs a two-stream ResNet50 network
optimized using the identity loss (LID ) and the triplet loss
(LTri ). STHF improves the baseline by an SDC module for
detail compensation with high-frequency low-level features
and a DSR module for semantic refinement with high-level
features.
Effectiveness of SDC and DSR. To verify the effectiveness
of SDC and DSR modules, we integrated them into the
baseline sequentially. As shown in Table III, in the Infraredto-Visible evaluation, the Rank-1 accuracy improved from
64.3% to 65.1% (+0.8%), and the mAP increased from 47.0%
to 49.3% (+2.3%) due to the effect of SDC, while DSR
improved the Rank-1 accuracy by 3.8% and the mAP by
5.7%, respectively. Similarly, in the Visible-to-Infrared evaluation, both SDC and DSR improved the Rank-1 accuracy
and mAP to varying degrees. These results demonstrate that
the SDC and DSR modules effectively enhance local details
and refine global semantics. It can be observed that the DSR
module contributes the most to performance improvement,
which demonstrates the superiority of spatial-temporal highfrequency information in semantic information mining. Moreover, when both the SCD and DSR modules are integrated into
the network, the Rank-1 accuracy increases by 6.3% and the

mAP improves by 8.2% in the Infrared-to-Visible evaluation,
while the Rank-1 accuracy and mAP increase by 5.6% and
9.3%, respectively, in the Visible-to-Infrared evaluation. This
demonstrates that the combined effect of SCD and DSR
enables the model to learn comprehensive and discriminative
feature representations.
Effectiveness of MP on Feature Interaction. To verify that
the MP module preliminarily alleviates modality discrepancies
and thereby facilitates effective feature interaction within the
SDC and DSR modules, we conducted additional ablation
experiments by removing MP from both components. As
shown in Table IV, in the Infrared-to-Visible evaluation, introducing the MP module into the SDC leads to an improvement
in Rank-1 accuracy from 64.9% to 65.1% (+0.2%) and an
increase in mAP from 47.9% to 49.3% (+1.4%). Similarly,
adding MP to the DSR module improves Rank-1 accuracy
from 66.6% to 68.1% (+1.5%) and mAP from 50.9% to 52.7%
(+1.8%). In the Visible-to-Infrared setting, the MP module
similarly enhances both SDC and DSR performance in terms
of Rank-1 and mAP, with improvements of 1.5%/1.3% for
SDC and 1.6%/1.7% for DSR, respectively. These results confirm that eliminating style information via MP promotes more
effective feature interaction and contributes to performance
improvement.
Impact of Insertion Depth for SDC and DSR. To verify
SDC targets low-level detail enhancement, while DSR focuses

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

This article has been accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology. This is the author's version which has not been fully edited and
content may change prior to final publication. Citation information: DOI 10.1109/TCSVT.2026.3670874

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY

11

Infrared to Visible

Visible to Infrared

1

1

2

2

B/L

Retrieval results from Rank@1 to Rank@10

Query

Query

1

1

2

2

Retrieval results from Rank@1 to Rank@10

Ours

Fig. 9. Visualization of pedestrian search results, where B/L represents the baseline method.

IR

1

2

(a)

(b)

RGB

B/L

Ours

Fig. 10. Comparison of Cross-Modality Feature Distance Distributions
between the baseline method and STHF.

on high-level semantic refinement, we examined the impact
of integrating the SDC and DSR modules at different blocks
of ResNet-50. Following common practice in representation
learning [78], we consider blocks 1–2 of ResNet-50 as shallow
layers that capture low-level textures and edge details, while
blocks 3–4 are regarded as deep layers responsible for semantic abstraction [79]. As shown in Table V, SDC performs better
when integrated into block 1 of the ResNet layers, while DSR
performs better when integrated into block 3. Specifically,
when applied at block 1 of the ResNet layers, in the Infraredto-Visible evaluation, the SDC module outperforms DSR by
0.2% in Rank-1 accuracy and 0.3% in mAP. In contrast, at
block 3 of the ResNet layers, the DSR module surpasses
SDC by 3.4% in Rank-1 accuracy and 2.9% in mAP. These
results demonstrate that low-level features contain detailed information suitable for information compensation, while highlevel features extract semantic category information suitable
for information guidance, aligning with our analysis in the
previous sections.
Influence of Filtering Strategy. To verify the effectiveness
of spatial-temporal high-pass filtering in the VVI-ReID task,
we conducted ablation experiments by constructing intermediate modalities using different filtering strategies. The cutoff
s = 10 for spatial filtering and f t = 2
frequencies were set to fco
co
for temporal filtering. As shown in Table VI, in the Infraredto-Visible setting, ST-HPF outperforms S-HPF and T-HPF by
3.1% and 3.6% in Rank-1 accuracy, and by 3.1% and 3.3% in
mAP, respectively. A similar trend is observed in the Visible-

Fig. 11. Comparison of CAM between the baseline method and STHF.

to-Infrared setting, where ST-HPF achieves improvements of
0.9% and 1.5% in Rank-1, and 2.7% and 2.8% in mAP over
S-HPF and T-HPF, respectively. These results confirm that
spatial-temporal filtering captures complementary frequency
cues, enabling more discriminative intermediate representations than single-dimensional approaches.
Comparison of different frequency-domain methods. To
evaluate the impact of different frequency learning strategies, we replaced ST-HPF with Patch wise High-frequency
Augmentation (PHA) [66] and Frequency Block (FRB) [58]
while leveraging the DSR module to learn frequency-domain
information. Specifically, we directly replace the original STHPF with PHA, using the HFC-enhanced sequence as the intermediate modality. As for FRB, we utilize global frequencydomain features to replace the intermediate modality representations, enabling spatial-frequency mutual learning. As shown
in Table VII, FRB achieves better average performance than
PHA, indicating that FFT-based frequency modeling provides
a more complete and expressive frequency representation than
DHWT. Furthermore, ST-HPF integrates 3D FFT to jointly
model HFC in both spatial and temporal dimensions, achieving
the best results across all metrics. These results demonstrate
the effectiveness of spatial-temporal high-frequency modeling
in enhancing feature representation and alleviating the modality discrepancy in the VVI-ReID task.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:08:55 UTC from IEEE Xplore. Restrictions apply.
© 2026 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.


codex
第五篇是视频 ReID 的特征级增广。我会重点看它是不是把“视频数据不足和遮挡”重定义成“时空特征扰动下的鲁棒表征学习”，以及增广是在输入、帧、token 还是特征层完成。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 168 (2025) 111813

Contents lists available at ScienceDirect

Pattern Recognition
journal homepage: www.elsevier.com/locate/pr

Spatio-temporal Feature-level Augmentation Vision Transformer for
video-based person re-identification
Minjung Kim a,b , MyeongAh Cho c , Heansung Lee a,d , Sangyoun Lee a

,∗

a

School of Electrical and Electronic Engineering, Yonsei University, Seoul, South Korea
LG Electronics Inc., Seoul, South Korea
c
Department of Software Convergence, KyungHee University, Yongin, South Korea
d
Samsung Electronics Co., Ltd., Suwon, South Korea
b

ARTICLE

INFO

Keywords:
Feature level augmentation
Vision Transformer
Video-based person re-identification

ABSTRACT
Video-based person re-identification (ReID) aims to match an individual across multiple videos, thus addressing
critical aspects of security applications of computer vision. While previous transformer-based approaches have
used various means to enhance performance, the growing complexities in network design have posed challenges
in meeting the practical requirements of intelligent surveillance systems. To improve network efficiency,
we introduce a Feature-level Augmentation Vision Transformer (FAViT), which reinterprets the attributes of
video ReID. We leverage the property of maintaining identity even when backgrounds change or multiple
persons appear in video frames. First, we introduce Token Representation Learning to distinguish foreground
from background. We also employ spatio-temporal feature-level augmentation, along with conducting Altered
Background ID classification and Anomaly Frame Detection, to strengthen the representation capacity of the
transformer. Extensive experiments validate the effectiveness of FAViT with the least computational overhead
among transformer-based models across five benchmarks. We substantiate our model’s generalization ability
through analyses.

1. Introduction
Person re-identification (ReID) is the task of identifying and matching particular individual across multiple videos captured from distinct
camera viewpoints. Ongoing research addresses challenges like camera
viewpoints [1,2], cross modality [3–5], occlusions [6–8], and language
descriptions [9,10] to achieve high accuracy. The recent success of
transformers in the field of computer vision has led to their adoption in
ReID [11–14], with various models proposed alongside state-of-the-art
(SOTA) solutions. However, the increasing demand for networks with
high accuracy and low computational overhead in intelligent surveillance systems has necessitated research into efficient methodologies
that can harness the advantages of transformers in video ReID.
The significance of possessing strong generalization capabilities is
underscored in ReID, where there is no overlap between training and
testing IDs. Examining Table 1, it is evident that CNN-based models
have smaller sizes, but their performance and generalization capabilities are lower than transformer-based models. DSANet [16], in order to
disentangle camera information, inherently divides its model structure
into branches, posing limitations in learning relationships with the

Table 1
Comparison with state-of-the art methods on LSVID and LSVID-to-MARS.
Method

LSVID

Param.

LSVID → MARS

R-1

mAP

R-1

mAP

BiCNet [15]
DSANet [16]

29.2 M
30.8 M

84.6
85.1

75.1
75.5

49.3
51.2

29.1
31.7

ViT-base [17]
CAViT [18]

85.8 M
218.8 M

85.3
89.2

76.4
79.2

68.7
70.1

50.2
53.1

Ours

78.7 M

89.3

78.7

71.1

53.6

foreground. This constraint is apparent in cross-dataset performance
results, and furthermore, the performance of CNN-based models with
auxiliary tasks does not translate into an enhancement in generalization
ability.
In contrast, the transformer-based models in Table 1 demonstrate
higher performance. CAViT [18], leveraging ViT [17] as a baseline, proposes a method for aligning spatial semantics through spatio-temporal
interaction between adjacent frames, yielding commendable performance. However, it requires a considerable number of parameters to

∗ Corresponding author.

E-mail address: syleee@yonsei.ac.kr (S. Lee).
https://doi.org/10.1016/j.patcog.2025.111813
Received 27 March 2024; Received in revised form 27 November 2024; Accepted 5 May 2025
Available online 24 May 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Pattern Recognition 168 (2025) 111813

M. Kim et al.

Fig. 1. Our proposed method. (a) A background token (BG) is added and separated from the class token (CLS), then reordered to generate spatial feature-level augmented samples
with new backgrounds. (b) Frames of different individuals are inserted to create temporally inconsistent feature-level augmented samples for Anomaly Frame Detection. During
the testing phase, the ability to detect low-consistency frames for the target individual is converted into a consistency score used in the refinement process.

perform patch embeddings of diverse sizes in order to extract temporal clues through attention mechanisms across previous and current frames. Transformers exhibit lower bias towards local textures,
allowing them to focus more on shapes, leading to higher generalization capabilities compared to CNNs. Leveraging self-attention layers
facilitates dense interactions across the entire image, creating a dynamically receptive field that remains robust even in the presence of
occlusions [19]. Furthermore, transformers can perform representative
feature extraction through extensive dataset coverage or task-specific
self-supervised learning. We believe that by further enhancing the
strengths of transformer models, the need for excessive design complexities in modules and structures for video ReID can be eliminated. This
has motivated us to explore feature-level augmentation.
In this paper, we propose a method for understanding the characteristics of video ReID by employing a transformer-based model to
perform sub-tasks generated through feature-level augmentation. We
introduce a new learnable token for embedding background information, endowing the network with the ability to differentiate between
foreground and background. Then, we extract class and background
features with token-attended patches through Token Representation
Learning (TRL). As depicted in Fig. 1, we leverage the property that
a person’s ID in the video remains constant even when the background
changes (Fig. 1-(a)) and that other people occasionally appear in video
(Fig. 1-(b)). We conduct Altered Background ID classification (ABIDC)
for new samples using diverse combinations of class and background
tokens through Spatial Feature Augmentation (SFA). The training strategy that performs Anomaly Frame Detection (AFD) on samples newly
generated through Temporal Feature Augmentation (TFA) enhances
the network’s ability to identify inconsistent frames within the video.
These capabilities can be translated into frame-level scores to improve
the final representation of the video. Our method maintains fewer
parameters owing to the removal of all classifiers involved in subtasks during inference. In conclusion, our proposed model not only
outperforms SOTA models using approximately 65% fewer parameters
than traditional transformer-based models but also exhibits enhanced
generalization ability.
Our main contributions are summarized as follows:

scope of the transformer model, boosting representation capacity
efficiently.
• Extensive experiments and analyses demonstrate the effectiveness
of our proposed approach on five video ReID benchmarks and validate its generalization ability through cross-dataset evaluation.
2. Related work
2.1. Video-based person re-identification
In video ReID, extracting a consistent representation of a person
from a sequence of consecutive frames is crucial.
CNN-based Video ReID. Some studies [20,21] have utilized 3D convolutions for simultaneous feature extraction and temporal modeling.
Other approaches [22–24] focus on extracting discriminative features
to differentiate individuals with similar appearances. Wang et al. [25]
used hierarchical temporal embedding and a pyramid structure for
frame-level feature aggregation. Graph neural networks address structural relationship issues and contextual interactions [6,7]. Certain
methods [15,26,27] extract complementary features from temporal
relations. Additionally, leveraging pose information [28,29], frequency
domain projection [30,31], and motion information [32] enhances
feature extraction and reduces information loss. While CNN-based
ReID models have fewer parameters, they typically perform worse
than transformer-based models due to inductive biases, limiting their
generalization capability.
Transformer-based Video ReID. Recently, transformer-based models
have become the standard in video ReID for extracting multi-scale
features and modeling temporal relations between frames. Zhang et al.
[33] applied multi-direction division strategies to patch embeddings,
rearranging patch features with diverse scales. CAViT [18] used three
different patch sizes and a combination of self- and cross-attention for
temporal modeling. STMN [34] simultaneously learned temporal and
spatial features, but increased parameters significantly by adding 16 additional blocks to the baseline. Several approaches have combined CNN
and transformer architectures to model inter-frame attention [35] or to
capture local feature relationships in the spatial–temporal domain [36].
In contrast to these methods, which increase parameters for spatiotemporal modeling, we propose an efficient training methodology using
feature-level augmentation, leveraging the characteristics specific to
video ReID.

• We propose a new background token and TRL that differentiate
between foreground and background.
• We propose a spatio-temporal feature-level augmentation (SFA,
TFA) and two sub-tasks (ABIDC, AFD) widening the learning
2

Pattern Recognition 168 (2025) 111813

M. Kim et al.

of 𝑃 × 𝑃 resolution, which are then flattened into 𝐷 dimensions
2
through linear projection 𝐄 ∈ R(𝑝 ⋅𝐶)×𝐷 . The input sequence consists
of combining the class token 𝜙 𝑐𝑙𝑠 , which represents the identity of
individuals in the video, with the patch embeddings and then adding
position embedding 𝜙 𝑝 to incorporate spatial information. The typical
input sequence 𝐳𝑘 ∈ R(𝑁+1)×𝐷 for the transformer layer can be expressed
as follows:

2.2. Feature-level augmentation
Various augmentation techniques have been primarily applied at
the image level to enhance deep-learning model performance. While
these methods increase the quantity of data by cropping, attaching,
and mixing images within the existing dataset, they have limitations in
terms of diversity. To address this, feature-level augmentation methods
have been proposed in tasks like image classification and semantic
segmentation, promoting various combinations of features. Some methods have been proposed to extract semantic information, generating
new meaningful features [37,38]. Additionally, techniques that create
bias-conflicting samples or use random Gaussian noise have improved
network generalization [39,40]. Temporal feature-level augmentation,
such as shuffle and reordering methods, in video understanding typically focuses on capturing motion by tracking action changes [41–
43]. However, video ReID prioritizes addressing the occlusions caused
by different objects appearing over time [44], rather than merely
capturing motion information. Thus, we propose feature-level augmentation methods specifically designed for video ReID, aiming to increase
training volume and detect the temporal context inconsistencies.

𝐳𝑘 = [𝜙 𝑘𝑐𝑙𝑠 ; 𝐱𝑘1 𝐄; 𝐱𝑘2 𝐄, … ; 𝐱𝑘𝑁 𝐄] + 𝜙 𝑝 ,

(1)

𝐙 = [𝐳1 ; 𝐳2 ; ⋯ ; 𝐳𝑇 ] ∈ R𝑇 ×(𝑁+1)×𝐷 ,

(2)

where 𝑘 signifies the 𝑘th frame. The final representation is obtained
by averaging the class tokens 𝛷 𝑐𝑙𝑠 produced for each frame. For the
identification loss, Circle Loss [50] is employed as follows: 𝑖𝑑 =
𝑐𝑖𝑟𝑐𝑙𝑒 (𝛷̄ 𝑐𝑙𝑠 ). Circle loss aims to maximize the intra-class similarity (𝑠𝑝 )
and minimize the inter-class similarity (𝑠𝑛 ). The circle loss formula is
given by:
[
]
𝐾
∑ 𝐿
∑
𝑐𝑖𝑟𝑐𝑙𝑒 = 𝑙𝑜𝑔 1 +
𝑒𝑥𝑝(𝛾(𝑠𝑗𝑛 − 𝑠𝑖𝑝 + 𝛥)) .
(3)
𝑖=1 𝑗=1

𝐾 intra-class similarity scores are denoted as 𝑠𝑖𝑝 for 𝑖 = 1, 2, … , 𝐾,
and 𝐿 inter-class similarity scores are denoted as 𝑠𝑗𝑛 for 𝑗 = 1, 2, … , 𝐿.
Additionally, 𝛾 represents the scale factor and 𝛥 denotes the margin.

Augmentation in ReID. Research on data augmentation techniques in
ReID has been ongoing to expand the scope of training and handle diverse cases. Zhang et al. [45] addressed overfitting by pretraining with
additional datasets, increasing the learning capacity of the transformer
model. To create various artificial occlusion scenarios, ADP [46] cut
and pasted background patches at the image level to extract features
robust to occlusion. Similarly, COAT [47] enhanced features against
occlusion by merging object features obtained from detection results
with the target person’s features. In contrast, we propose a training
strategy for video ReID that generates examples in various spatial and
temporal domains at the feature level, which are not achievable at
the image level. Our method is efficient as it does not require image
preprocessing or detection models. To the best of our knowledge, we
are the first to apply feature-level augmentation simultaneously in both
spatial and temporal domains for video ReID.

3.2. Overview
A concise overview of FAViT is shown in Fig. 2. Our model consists of three main processes: Token Representation Learning (TRL),
spatio-temporal feature-level augmentation, and individual sub-tasks.
In TRL, learnable tokens are used to embed foreground and background
information separately, generating token-attended patches. Spatial Feature Augmentation (SFA) is employed to create ID samples with new
backgrounds through random combinations of foregrounds and backgrounds. FAViT then performs the ID classification sub-task. For videos
comprised of features from multiple frames, feature augmentation at
the temporal level is conducted by placing features from different
individuals at random positions, followed by Anomaly Frame Detection
(AFD). Finally, the logits used in AFD are used to generate framelevel scores, refining the ultimate video embedding. Spatio-temporal
feature-level augmentation and its sub-tasks are executed only during
the training phase, while in the testing phase, the input passes through
a pure-transformer block after removing the sub-task classifiers.

2.3. Camera information in ReID
ReID aims to identify the same individual appearing in different
cameras, with the camera ID of the captured images always provided. Given this label without additional labeling, various methodologies in ReID leverage camera ID information. He et al. [48] used
position embedding as a memory bank to store camera viewpoint
information, but this approach requires a camera ID label during inference and is inapplicable to datasets with varying camera numbers.
Lei et al. [49] disentangled foreground and background by camera
information and attention mechanism for domain adaptation. However,
two-stream ResNet50 for feature disentangling makes it less efficient.
Kim et al. [16] disentangled camera information in repeating frames
of video ReID, enhancing performance with auxiliary tasks but limiting
itself by not utilizing temporal information in videos. In contrast, our
newly introduced background token draws inspiration from the class
token in the transformer structure, enabling it to learn foreground and
background separation without additional blocks. Furthermore, utilizing the separated embeddings in spatio-temporal feature augmentation
improves the networks’ performance and generalization capacity.

3.3. Token representation learning
Background Token. To enable the network to distinguish between
foreground and background, we embed a class token 𝜙 𝑘𝑐𝑙𝑠 ∈ R1×𝐷
representing the foreground and a background token 𝜙 𝑘𝑏𝑔 ∈ R1×𝐷 representing the rest. To mine background information in patch embedding,
FAViT performs camera ID classification with 𝑐𝑎𝑚 = 𝑐𝑒 (𝜙̄ 𝑏𝑔 ), similar
to what DSANet [16] uses. While DSANet, a CNN-based model, adds a
branch to disentangle camera information, we leverage the structural
attributes of the transformer, allowing FAViT to embed the desired
features with minimal parameters (a learnable token). Furthermore,
the information mining from class token and background token in the
same self-attention block may be complementary because each obtains
information tailored to its purpose. Input sequences of the transformer
layer are prepared by prepending class token 𝜙 𝑘𝑐𝑙𝑠 and background
token 𝜙 𝑘𝑏𝑔 to the patch embeddings.

3. Proposed methods
3.1. Preliminary

𝜙𝑘𝑐𝑙𝑠 ; 𝜙𝑘𝑏𝑔 ; 𝐱𝑘1 𝐄; 𝐱𝑘2 𝐄, … ; 𝑥𝑁
𝜙
𝐳𝑘 = [𝜙
𝑘 𝐄] + 𝑝
𝑇 ×(𝑁+2)×𝐷

] ∈ R𝑇 ×𝐻×𝑊 ×3 ,

𝐙 = [𝐳1 ; 𝐳2 ; ⋯ ; 𝐳𝑇 ] ∈ R

In video ReID, the input is a video 𝑣 = [𝑥1 , 𝑥2 , … , 𝑥𝑇
where 𝑇 is the number of frames in the input video and 𝐻 and 𝑊
represent height and width, respectively. Following He et al. [48], each
frame 𝑥𝑘 is divided into 𝑁 overlapping image patches [𝐱𝑘1 , 𝐱𝑘2 , … , 𝐱𝑘𝑁 ]

(4)
(5)

Token-Attended Patches. The input sequence 𝐙 is fed into a transformer layer composed of a multi-head self-attention (MSA) block and
3

Pattern Recognition 168 (2025) 111813

M. Kim et al.

Fig. 2. Overall framework of FAViT. In the training phase, video frames serve as inputs, which are divided into overlapping image patches. After undergoing linear projection, patch
embeddings are created. 𝛷 𝑐𝑙𝑠 and 𝛷 𝑏𝑔 are prepended to the patch embeddings, forming the input for the transformer. (a) Token Representation Learning involves passing through
transformer blocks, generating an attention map 𝐀𝐭𝐭𝐧 used to embed background and foreground information into token-attended patches. (b) Spatial feature augmentation and
Altered Background ID Classification entail creating samples with a new background by adding randomly rearranged 𝛷 𝑏𝑔 to the existing 𝛷 𝑐𝑙𝑠 , performing an ID classification
sub-task. (c) Temporal feature augmentation and Anomaly Frame Detection involve introducing frames of different individuals at random temporal positions within the video
feature, performing a sub-task to locate these frames. In the testing phase, only 𝛷 𝑓 𝑖𝑛𝑎𝑙 is used as the final representation. Light blue text indicates loss functions, while dark blue
text indicates feature dimensions. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)

a feed-forward network. For the 𝑚th multi-head self-attention block,
the 𝑞𝑢𝑒𝑟𝑦, 𝑘𝑒𝑦, and 𝑣𝑎𝑙𝑢𝑒 are all 𝐙𝑚−1 , and the attention map for each
head is obtained by matrix multiplication of 𝑞𝑢𝑒𝑟𝑦 and 𝑘𝑒𝑦.
𝐙′𝑚 = 𝑀𝑆𝐴(𝐿𝑁(𝐙𝑚−1 )) + 𝐙𝑚−1

(6)

𝐙𝑚 = 𝑀𝐿𝑃 (𝐿𝑁(𝐙′𝑚 )) + 𝐙′𝑚 ,

(7)

𝑚 = 1, … , 𝑀

The loss function used for TRL is as follows:
𝑡𝑟𝑙 = 𝑖𝑑 + 𝜆𝑐𝑎𝑚 𝑐𝑎𝑚 + 𝑠𝑒𝑝 .

Simultaneously employing 𝑐𝑎𝑚 and 𝑠𝑒𝑝 encourages the model to prioritize global information over specific parts, thereby fostering precise
and complementary embeddings for foreground and background.

Taking the average of these attention maps for each head and summing
them for all 𝑀 blocks yields 𝐀𝐭𝐭𝐧 ∈ R(𝑁+2)×(𝑁+2) . Integrating the attention maps from all layers is to capture receptive field information of
various sizes while simultaneously reducing computational complexity.
The first row of 𝐀𝐭𝐭𝐧 represents the scores of all patches for the class
token, while the second row represents the scores for the background
token.
𝐀𝐭𝐭𝐧𝑐𝑙𝑠 = 𝐀𝐭𝐭𝐧[0, 2 ∶ 𝑁 + 2]

(8)

𝐀𝐭𝐭𝐧𝑏𝑔 = 𝐀𝐭𝐭𝐧[1, 2 ∶ 𝑁 + 2]

(9)

3.4. Spatio-temporal feature augmentation
We introduce feature-level augmentation that simulates new scenarios to make the final representation suitable for video ReID. FAViT
generates samples for situations involving background changes and the
emergence of other individuals without requiring additional labeling.
By performing sub-tasks for each scenario, it learns to focus on the
foreground across diverse backgrounds and quantitatively assess frame
reliability. The mini-batch size is denoted as 𝐵 = 𝑁𝑖𝑛𝑠𝑡 × 𝑐, consisting
of a number 𝑁𝑖𝑛𝑠𝑡 of individuals, each with 𝑐 tracklets. We represent
𝑇 class tokens as 𝛷 𝑐𝑙𝑠 = [𝜙 1𝑐𝑙𝑠 , 𝜙 2𝑐𝑙𝑠 , … , 𝜙 𝑇𝑐𝑙𝑠 ] ∈ R𝑇 ×𝐷 and 𝑇 background
tokens as 𝛷 𝑏𝑔 = [𝜙 1𝑏𝑔 , 𝜙 2𝑏𝑔 , … , 𝜙 𝑇𝑏𝑔 ] ∈ R𝑇 ×𝐷 .

Each token-attended patch is expressed as follows:
𝐅𝑐𝑙𝑠 = 𝐀𝐭𝐭𝐧𝑐𝑙𝑠 ⋅ 𝐙𝑀 [2 ∶ 𝑁 + 2],

(10)

𝐅𝑏𝑔 = 𝐀𝐭𝐭𝐧𝑏𝑔 ⋅ 𝐙𝑀 [2 ∶ 𝑁 + 2].

(11)

Spatial Feature Augmentation. Based on the fact that the person’s ID
in the video remains unchanged even when the background changes
and the need to identify the same individual when the background
alters, we can apply feature-level augmentation at the spatial level. 𝛷 𝑐𝑙𝑠
and 𝛷 𝑏𝑔 utilized in TRL are applied. FAViT randomly selects B indices,
rearranging the order of 𝛷 𝑏𝑔 according to the sampled indices. Reordered 𝛷 𝑏𝑔 combines it with 𝛷 𝑐𝑙𝑠 to create a new representation 𝛷 𝑠𝑝𝑎
with a different background, expressed as 𝛷 𝑠𝑝𝑎 = 𝛷 𝑐𝑙𝑠 + 𝑟𝑒𝑜𝑟𝑑𝑒𝑟(𝛷 𝑏𝑔 ).

The rationale for utilizing 𝐙𝑀 lies in leveraging the comprehensive
global semantic and spatial information found in the features of the
last layer [51]. This approach facilitates distinct spatial differentiations between foreground and background without focusing on partial
details.
Using separation loss [16,52], commonly employed for diverse or
mutually independent feature extraction, the network is trained to
reduce the similarity between the class token-attended patch embedding vector 𝐟𝑐𝑙𝑠 = 𝑔𝑎𝑝(𝐅𝑐𝑙𝑠 ) and the background token-attended patch
embedding vector 𝐟𝑏𝑔 = 𝑔𝑎𝑝(𝐅𝑏𝑔 ). ‘gap’ denotes global average pooling,
reducing patch spatial dimensions to compute Eq. (12).
𝑠𝑒𝑝 = 𝑚𝑎𝑥(

𝐟𝑐𝑙𝑠 ⋅ 𝐟𝑏𝑔
, 0)
‖
‖𝐟𝑐𝑙𝑠 ‖ ‖
𝐟𝑏𝑔 ‖
‖ ‖2 ‖
‖ ‖2

(13)

Altered Background ID Classification. FAViT performs the ID Classification sub-task using the representation generated through SFA. This
ensures that FAViT can correctly identify IDs even when presented
with a novel background or in the presence of background cluttering.
The loss function employed for this purpose is denoted as 𝑠𝑝𝑎 =
𝑐𝑖𝑟𝑐𝑙𝑒 (𝛷̄ 𝑠𝑝𝑎 ).
Temporal Feature Augmentation. In Video ReID, it is crucial to
minimize the reliability of frames with low accuracy concerning the

(12)

4

Pattern Recognition 168 (2025) 111813

M. Kim et al.

target identity. To achieve this, FAViT must effectively identify these
inaccurate frames. However, image-level datasets typically do not provide frame-specific accuracy labels. To address this challenge, TFA
introduces frames from different identities, thereby creating samples
that disrupt consistency and facilitating the detection of unreliable
frames. FAViT randomly selects one index from the numbers up to 𝑇 ,
then chooses the class token of the frame corresponding to the random
index in each mini-batch and shifts it by 𝑐. This process creates a
representation 𝛷 𝑡𝑒𝑚𝑝 ∈ R𝑇 ×𝐷 with a different individual frame.

Algorithm 1 Feature-level augmentation vision transformer (FAViT)
Input: Video 𝑣 = [𝑥1 , 𝑥2 , ⋯ , 𝑥𝑇 ] ∈ R𝑇 ×𝐻×𝑊 ×3
Output: The final representation 𝛷̄ 𝑓 𝑖𝑛𝑎𝑙 ∈ R𝐶
1: Input 𝑣 into linear projections
⊳ Get patch embeddings
2: Z ← Prepend 𝜙 𝑐𝑙𝑠 and 𝜙 𝑏𝑔 to patch embeddings
⊳ Add BG token
3: for 𝑚 ← 1 to 𝑀 do
4:
Z𝑚 ← Input Z𝑚−1 into transformer layer
5: end for
6: F𝑐𝑙𝑠 , F𝑏𝑔 ← Calculate the scores Attn and get token-attended patches
7: if Training phase then
8:
𝛷 𝑠𝑝𝑎 ← Combine reordered 𝛷 𝑏𝑔 with 𝛷 𝑐𝑙𝑠
⊳ Implement SFA
𝛷 𝑡𝑒𝑚𝑝 ← Select frame randomly from other individuals
9:
⊳ Implement
TFA
10: else
11:
𝛷 𝑡𝑒𝑚𝑝 ← 𝛷 𝑐𝑙𝑠
12: end if
13: 𝜙 𝑙𝑜𝑔𝑖𝑡 ← Predict the index of anomaly frames
⊳ Implement AFD
14: 𝛷 𝑓 𝑖𝑛𝑎𝑙 ← Calculate frame-level score S and refine 𝛷 𝑐𝑙𝑠
⊳ Implement RL
15: return Average 𝛷 𝑓 𝑖𝑛𝑎𝑙 along the temporal axis

Anomaly Frame Detection. FAViT performs the AFD sub-task using
the representation 𝛷 𝑡𝑒𝑚𝑝 generated through TFA. This process enables
FAViT to identify frames corresponding to different individuals and
further transform these into frame-level scores that refine the final
representation. The relationships between class tokens for each frame
obtained through temporal feature augmentation are modeled through
a single transformer layer.
𝛷 ′𝑡𝑒𝑚𝑝 = 𝑀𝑆𝐴(𝐿𝑁(𝛷 𝑡𝑒𝑚𝑝 )) + 𝛷 𝑡𝑒𝑚𝑝

(14)

′
′
𝛷 ′′
𝑡𝑒𝑚𝑝 = 𝑀𝐿𝑃 (𝐿𝑁(𝛷 𝑡𝑒𝑚𝑝 )) + 𝛷 𝑡𝑒𝑚𝑝

(15)

A logit 𝜙 𝑙𝑜𝑔𝑖𝑡 ∈ R𝑇 ×1 is generated by predicting the index of anomaly
frames using tokens that model the temporal relations between frames.
𝜙 𝑙𝑜𝑔𝑖𝑡 = 𝑟𝑒𝑙𝑢(𝐶𝑜𝑛𝑣(𝛷 ′′
𝑡𝑒𝑚𝑝 ))

It presents challenges due to a considerable amount of occlusion and
low-quality frames. Given that there is only one correct match in the
gallery set, mAP measurement is not applicable.

(16)

MARS-DL [54] is a refined dataset derived from MARS through redetection, reducing frames where the background occupies a substantial portion and removing individuals prone to ID switches from frames.

Specifically, ‘𝐶𝑜𝑛𝑣’ denotes a convolution layer that transforms 𝛷 ′′
𝑡𝑒𝑚𝑝
into 𝜙 𝑙𝑜𝑔𝑖𝑡 . The loss function used in AFD is the cross-entropy loss,
denoted as 𝑡𝑒𝑚𝑝 = 𝑐𝑒 (𝜙 𝑙𝑜𝑔𝑖𝑡 ). During the testing phase, since TFA is
not applied, 𝛷 𝑡𝑒𝑚𝑝 is replaced by the original 𝛷 𝑐𝑙𝑠 . FAViT then generates
logits with decreased consistency due to real-world temporal occlusions
by analyzing the relations among frame features.

MEVID [55] is the latest video ReID dataset, showcasing 158 individuals across 8092 sequences captured by 33 cameras. Unlike traditional
video ReID datasets, individuals in MEVID wear multiple outfits, and
there is no overlap between the camera IDs in the training and testing
sets.

Refinement Learning. Unlike previous methods [42,43] that conclude
with representation learning, FAViT leverages the capabilities learned
through TFA and AFD to score the consistency of frames within a
sequence. The logit is processed into a frame-level score 𝐒 ∈ R𝑇 ×1 ,
which is used to refine the video representation. 𝐒 tends to have lower
values for frames with inconsistent semantics and higher values for
frames with consistent information.
𝐒 = 1 − 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥(𝜙 𝑙𝑜𝑔𝑖𝑡𝑠 )

MARS [56] is a large video ReID dataset that comprises over 20,000
sequences of 1261 individuals, along with 3248 distractors. The videos
were captured using 6 cameras, and the bounding boxes for the individuals were detected using the DPM detector and the GMMCP
tracker.
LSVID [57] is the largest video ReID dataset, featuring 3,772 individuals and 14,943 sequences, which were recorded using 15 cameras.

(17)

Therefore, the frame-level score is manipulated as follows to refine the
video representation:
𝛷 𝑓 𝑖𝑛𝑎𝑙 = 𝐒 ⋅ (𝛷 𝑐𝑙𝑠 ).

Evaluation Metrics We utilize Cumulative Matching Characteristic
(CMC) and mean Average Precision (mAP).

(18)
4.2. Implementation details

𝛷 𝑓 𝑖𝑛𝑎𝑙 is averaged along the temporal axis, and FAViT is trained with
𝑟𝑒𝑓 𝑖𝑛𝑒 = 𝑐𝑖𝑟𝑐𝑙𝑒 (𝛷̄ 𝑓 𝑖𝑛𝑎𝑙 ).

4. Experiments

We adopt the encoder of ViT [17] pre-trained with ImageNet as our
feature extractor. Our network includes 11 out of the total 12 blocks
in ViT base, excluding the final block. To form our training batches,
we randomly choose 4 identities and sample 4 sequences for each
identity, with a batch size of 16. Furthermore, we train the model by
employing the restricted-begin-crop strategy, which involves selecting
8 frames from all the video sequences. We follow the training settings
of CAViT [18]. During testing, features are extracted for all frames
in the sequence, and the final feature is obtained through temporal
average pooling. The distance between the extracted query vector and
gallery vector is calculated using the cosine similarity operation. FAViT
is trained for 30 epochs using a single RTX Titan, which requires 3 h
for the MARS and 6 h for the LSVID. When testing on all frames, the
evaluation process takes 50 min for the MARS and 3 h for the LSVID.

4.1. Datasets and evaluation metrics

4.3. Comparison with state-of-the-art methods

iLIDS-VID [53] is a relatively small video ReID dataset comprising 300
individuals and 600 sequences, captured by two cameras at an airport.

Benchmarks. We categorize SOTA methods on five video ReID benchmarks into CNN-based and transformer-based approaches in Table 2.

3.5. Training and testing phases
During training, the objective function of FAViT is defined as:
𝑡𝑜𝑡𝑎𝑙 = 𝑡𝑟𝑙 + 𝜆𝑠 𝑠𝑝𝑎 + 𝜆𝑡 𝑡𝑒𝑚𝑝 + 𝜆𝑟 𝑟𝑒𝑓 𝑖𝑛𝑒 ,

(19)

where 𝜆𝑠 , 𝜆𝑡 , and 𝜆𝑟 are the weights for the joint learning. In the
testing phase, the classifiers that perform the sub-tasks are removed,
and the final representation is obtained by averaging 𝛷 𝑓 𝑖𝑛𝑎𝑙 along the
temporal axis. The following pseudo-code outlines FAViT, summarizing
the workflow and detailing key operations in Algorithm 1.

5

Pattern Recognition 168 (2025) 111813

M. Kim et al.

Table 2
Comparison with state-of-the-art methods on iLIDS-VID, MARS-DL, MEVID, MARS, and LSVID video ReID datasets. Performance indicated by† refers to results reimplemented using
publicly available code for models not originally included in the paper.
Models

Pub.

Param.

iLIDS-VID [53]

MARS-DL [54]

MEVID [55]

MARS [56]

LSVID [57]

R-1

R-5

R-1

mAP

R-1

mAP

R-1

mAP

R-1

mAP

P3D [58]
GLTR [57]
DAM [1]
AP3D [20]
TCLNet [26]
AGRL [59]
PA-HVPReid [6]
DCT-IEN [30]
CTL [28]
BiCnet-TKS [15]
PSTA [25]
SGWCNN [7]
SINet [27]
MFA [32]
DSANet [16]
LGCN+TAM [60]
FIDN [61]
ST-DHGNN [31]

ICCV 17
ICCV 19
PR 19
ECCV 20
ECCV 20
TIP 20
PR 21
ACMMM 21
CVPR 21
CVPR 21
ICCV 21
PR 22
CVPR 22
TIP 22
WACV 23
TCSVT 23
TIP 23
TIP 23

–
–
–
31.6 M
29.9 M
–
–
23.6 M
–
29.2 M
35.5 M
39.1 M
–
24.9 M
30.8 M
–
–
122.5 M

–
86.0
77.3
88.7
86.6
84.5
87.9
89.3
89.7
–
91.5
87.8
92.5
93.3
88.7†
90.2
91.3
92.0

–
98.0
94.0
–
–
96.7
93.6
98.0
97.0
–
98.1
96.0
–
98.7
97.3†
98.5
98.0
98.7

91.0
–
–
91.3
91.0
–
–
–
–
–
–
–
–
93.0
94.9†
–
–
–

85.0
–
–
86.5
85.4
–
–
–
–
–
–
–
–
87.2
90.1†
–
–
–

–
–
–
39.6
48.1
48.4
–
–
–
19.0
46.2
–
–
–
43.4†
–
–
–

–
–
–
15.9
23.0
19.1
–
–
–
6.3
21.2
–
–
–
21.6†
–
–
–

88.9
87.0
74.7
90.1
89.8
89.5
89.9
89.9
91.4
90.2
91.5
90.0
91.0
90.4
91.1
91.1
91.5
92.5

83.2
78.5
57.7
85.1
85.1
81.9
79.6
84.4
86.7
86.0
85.8
85.7
86.2
85.0
86.6
86.5
86.8
86.5

53.4
63.0
–
–
81.0
–
–
–
–
84.6
–
–
87.4
88.2
85.1
–
–
–

3.05
44.3
–
–
67.2
–
–
–
–
75.1
–
–
79.6
78.9
75.5
–
–
–

ViT-base [17]
Token Shift [62]
PiT [33]
CAViT [18]
SITN [34]
HASI [63]
DCCT [35]

ICLR 21
MM 21
TII 22
ECCV 22
IoTJ 23
TCSVT 23
TNNLS 23

85.8 M
85.9 M
108.3 M
218.8 M
85.8 M ↑
156.0 M
222.6 M

90.2
86.0
92.1
93.3
94.0
93.3
91.7

93.7
98.0
98.9
98.0
100.0
99.6
98.6

94.6
94.9
95.1†
95.6
–
–
–

89.4
90.1
90.4†
90.5
–
–
–

55.7†
–
34.2
–
–
–
–

31.2†
–
13.6
–
–
–
–

89.7
90.2
90.2
90.8
89.2
91.4
92.3

86.4
86.6
86.8
87.2
85.1
87.5
87.5

85.3
80.4
–
89.2
–
–
–

76.4
68.7
–
79.2
–
–
–

FAViT

Ours

78.7 M

95.3

100.0

96.2

90.9

59.2

33.9

90.4

87.4

89.3

78.7

Table 3
Comparison with state-of-the-art method [18] on cross-dataset evaluation.
Methods
CAViT [18]
FAViT (Ours)

MARS-DL → iLIDS

Param.
218.8 M
78.7 M

MARS-DL → MARS

MARS-DL → LSVID

LSVID → MARS

R-1

R-5

mAP

R-1

mAP

R-1

mAP

R-1

40.7
50.0

64.0
65.33

77.7
82.7

85.2
88.2

21.0
22.0

32.2
33.5

53.1
53.6

70.1
71.1

Despite considerable variations in model size, CNN-based SOTA models
demonstrate performance on par with ViT. Transformer captures relationships between different parts of an image, independent of spatial
distance, thereby enhancing its ability to understand the interdependencies among the various components that constitute a person. First,
when comparing FAViT to CNN-based methods, although our model has
a larger network size, it achieves high performance. Furthermore, as evident in Table 1, it exhibits superior generalizable ability compared to
CNN-based models. The reason for this lies in FAViT’s TRL, which prioritizes the foreground, thus reducing the domain gap between datasets.
When comparing FAViT to other transformer models, a notable aspect
is its competitive performance across the five benchmarks, while having
the smallest model size. While PiT [33] and CAViT [18] have succeeded
in extracting rich representations of various sizes, the model sizes have
increased by up to approximately 2.5 times compared to the baseline
ViT. The compact architecture of our model leads to reduced training
and inference times, thereby improving its applicability in practical
scenarios. In contrast, the relatively lower performance of DCCT [35]
on the iLIDS-VID dataset, which is characterized by limited data, highlights its potential challenges in generalizing to unseen samples. In
contrast, FAViT considers the learning capacity proportional to the
amount of data in the transformer. Instead of designing additional
modules, we perform ABIDC and AFD tailored to the video ReID,
thereby expanding the representative capacity.

on multiple datasets, demonstrated a noticeable improvement in accuracy. In other words, when learning from a cleaner dataset where
challenging situations with occlusion are less frequent, our spatiotemporal augmentation during sub-tasks (ABIDC, AFD) enhances the
network’s robustness to general re-identification. FAViT enhances generalization solely through the training approach, offering a significant
practical advantage.

Cross-dataset Evaluation. We quantitatively compare the generalization ability of our proposed method with CAViT [18] in Table 3 and
with CNN-based methods in Table 1. We reproduced the cross-dataset
evaluation results using the pre-trained model weights and structure
provided by the authors of CAViT. FAViT, trained on MARS-DL, which
involves re-detection of MARS with reduced ID mismatching, and tested

Performance Analysis of Each Component. Table 5 (a) summarizes
the results of the ablation study for the components of FAViT on iLIDSVID and MARS-DL. BG denotes TRL with the addition of a background
token; Spa. represents SFA and ABIDC; Temp. denotes TFA and AFD;
and Refine. stands for RL. Employing BG led to a 1.4%p increase in
rank-1. When applying SFA and TFA separately, iLIDS-VID exhibits

Model Efficiency We report the model size and computational complexity required for training various transformer-based methods in Table 4 (a). The computational complexity, represented in giga floatingpoint operations (GFLOPs), was measured based on the input size
(1 × 8 × 3 × 256 × 128). In Table 4 (b), it is evident that the model’s
parameter count increases during the training phase due to the addition
of classifier head modules for sub-tasks. However, the GFLOPs remain
largely unaffected, as the augmentations mainly involve reordering
operations, softmax functions, and arithmetic calculations. Specifically,
adding the background token increases the parameter count by only
0.01 M. FAViT’s compact size and low computational demands result from effective feature-level augmentation, which enhances the
representational power of the final embeddings.
4.4. Ablation study

6

Pattern Recognition 168 (2025) 111813

M. Kim et al.
Table 4
Number of parameters and computational complexity.
(a) Transformer-based Methods

(b) The components of FAViT

Model

Param.

GFLOPs

Model

Train Param.

GFLOPs

Test Param.

ViT-base [17]
Token Shift [62]
PiT [33]
CAViT [18]

85.8 M
85.9 M
108.3 M
218.8 M

289.1
289.1
361.4
518.5

FAViT (Ours)

78.7 M

242.5

Baseline
+ BG
+ Spa.
+ Temp.
+ Refine.

71.75 M
71.76 M
71.87 M
78.95 M
79.07 M

241.24
242.38
242.38
242.49
242.49

71.63 M
71.64 M
71.64 M
78.72 M
78.72 M

Table 5
Ablation study of the various components in FAViT (a) and Cross-dataset evaluation on MARS-DL-to-iLIDS-VID (b).
(a) Ablation study
BG

✓
✓
✓
✓
✓

Spa.

(b) Cross-dataset evaluation

Temp.

Refine.

✓
✓
✓

✓
✓
✓

✓

iLIDS-VID

MARS-DL

R-1

R-5

R-1

mAP

91.3
92.7
94.7
94.7
94.7
95.3

98.7
98.0
99.3
98.7
99.3
100.0

94.8
95.8
96.0
96.1
96.1
96.2

90.3
90.8
90.9
90.6
90.8
90.9

BG

✓
✓
✓
✓
✓

Spa.

Temp.

Refine.

MARS-DL → iLIDS-VID
R-1
45.3
46.0
46.7
49.3
48.7
50.0

✓
✓
✓
✓

✓
✓

✓

Table 6
Ablation study based on the feature selection in the loss function (a) and Performance on iLIDS-VID with varying loss function weights (b).
(b) Different weights in the loss function
R-1

R-5

0.1
0.5
1
2

92.7
90.7
86.0
84.0

98.0
93.5
96.7
97.3

0
0.1
0.5
1

91.3
90.7
92.0
92.7

98.0
98.7
98.0
98.0

(a) Feature selection in 𝑠𝑝𝑎 , 𝑠𝑒𝑝
Feature Selection

𝑠𝑝𝑎 (𝑃 𝑎𝑡𝑐ℎ) + 𝑠𝑒𝑝 (𝑇 𝑜𝑘𝑒𝑛)
𝑠𝑝𝑎 (𝑃 𝑎𝑡𝑐ℎ) + 𝑠𝑒𝑝 (𝑃 𝑎𝑡𝑐ℎ)
𝑠𝑝𝑎 (𝑇 𝑜𝑘𝑒𝑛) + 𝑠𝑒𝑝 (𝑃 𝑎𝑡𝑐ℎ)
𝑠𝑝𝑎 (𝑇 𝑜𝑘𝑒𝑛) + 𝑠𝑒𝑝 (𝑇 𝑜𝑘𝑒𝑛)

MARS

LSVID

mAP

R-1

mAP

R-1

87.5
87.5
87.7
87.4

91.3
91.1
91.2
91.2

77.0
77.0
77.4
77.1

87.7
87.8
88.1
87.8

𝜆𝑐

𝜆𝑠𝑒𝑝

a 2.0%p increase in rank-1. iLIDS-VID, acquired at an airport and
featuring frequent occlusions, is a challenging dataset, and despite having the smallest training dataset, FAViT’s feature-level augmentation
enables additional learning, contributing to performance enhancement.
Employing BG on MARS-DL, the model shows improved capabilities
in distinguishing foreground and background, resulting in a 1.0%p
increase in rank-1. Despite the sparsity of challenging samples in the
refined MARS-DL compared to MARS, the application of feature-level
augmentation resulted in a 0.3%p improvement in rank-1. Particularly
evident in Fig. 5, the substantial accuracy difference between the
baseline and challenging samples highlights the effectiveness of feature
augmentation and sub-task. The comparison between the fifth and sixth
rows indicates that utilizing the consistency score for refining the final
video embedding vector yields optimal performance. The effectiveness
of feature-level augmentation may vary depending on dataset characteristics. FAViT demonstrates its capability to extend the learning space
in small datasets and enhance accuracy on challenging samples in clean
datasets.

𝜆𝑠

𝜆𝑡

R-1

R-5

0
0.1
0.5
1

92.7
94.7
95.3
95.3

98.0
99.3
98.7
99.3

0
0.1
0.5
1

92.7
94.7
92.7
91.3

98.0
98.7
97.3
96.7

Fig. 3. Visualization of class token and background token on MARS. Warmer colors
represent heightened activation. (For interpretation of the references to color in this
figure legend, the reader is referred to the web version of this article.)

Cross-dataset Evaluation. In practical scenarios, maintaining robust
generalization performance of ReID networks across diverse datasets is
crucial. Table 5 (b) shows that FAViT, trained on MARS-DL and tested
on iLIDS-VID, improves rank-1 by 0.7%p with BG incorporation, highlighting its ability to distinguish between foreground and background
and mitigate domain gaps. FAViT’s spatio-temporal feature augmentation further enhances performance, achieving up to a 2.0%p rank-1
improvement, demonstrating effective learning from challenging scenarios. However, applying both Temp. and Spa. simultaneously led to
a 0.6%p performance drop due to domain discrepancies, as MARS-DL’s
diverse backgrounds were less effective on iLIDS-VID’s limited backgrounds. This aligns with the minimal improvement when using Spa.
alone. Ultimately, RL refines these capabilities, resulting in optimal

performance. FAViT’s remarkable generalization ability underscores its
potential to leverage ReID task characteristics for sub-tasks, enabling
suitable final representations even with differing training and testing
datasets.
Study on Feature Selection. Based on the experimental results in
Table 6 (a), we select features to be used in the loss function (𝑠𝑝𝑎 , 𝑠𝑒𝑝 ).
This experiment evaluates performance with 8 frames. There are two
types of features to which SFA and Separation loss (𝑠𝑒𝑝 ) can be
applied: tokens and patches. After experimenting with all four cases, we
find that applying SFA to tokens and applying 𝑠𝑒𝑝 to Token-Attended
7

Pattern Recognition 168 (2025) 111813

M. Kim et al.

Fig. 4. Visualization of consistency score on iLIDS-VID (a, b) and MEVID (c, d).

Patches yielded the highest performance. The rationale behind this
choice lies in the fact that ultimately passing through the classifier
involves tokens, aligning to augment between tokens in ABIDC. Furthermore, applying 𝑠𝑒𝑝 to Token-Attended Patches allows for the
consideration of spatial information, enabling differentiation between
the foreground and background for each patch.
Effects of Different Hyper-Parameters. Table 6 (b) presents the performance variations of FAViT based on the weights of the loss functions
(𝑐𝑎𝑚 , 𝑠𝑒𝑝 , 𝑠𝑝𝑎 , 𝑡𝑒𝑚𝑝 ) aiding its training. 𝑐𝑎𝑚 , employed for background token representation learning, benefits from a smaller weight
value because of its role in sub-task execution. The inclusion of 𝑠𝑒𝑝 ,
guiding towards reduced similarity with the foreground, enhances accuracy. We observe that a higher weight 𝜆𝑠 for 𝑠𝑝𝑎 , involved in ABIDC,
and a relatively smaller 𝜆𝑡 for 𝑡𝑒𝑚𝑝 , associated with AFD, lead to higher
accuracy.
4.5. Qualitative analysis
Token Visualization. We employ roll-out method [51], a commonly
used visualization tool for ViT, to visualize the tokens in Fig. 3. To
clearly depict the information between the foreground and background
tokens, we normalize the visualization scores relative to each other.
In the first row, where the background predominantly occupies the
scene, FAViT’s class token is observed to focus on the foreground
information. In the second and third rows, it is evident that FAViT
achieves consistent feature extraction regardless of the extent to which
the target’s position changes over time.

Fig. 5. Retrieval results of baseline and FAViT on MARS-DL. The accurate matches are
outlined in green, while the inaccurate ones are highlighted in red. (For interpretation
of the references to color in this figure legend, the reader is referred to the web version
of this article.)

variations in individual appearance due to different camera views or
changes in human pose lead to lower consistency scores.

Consistency Score Visualization. We visualize the consistency score
used in RL for the iLIDS-VID and MEVID datasets in Fig. 4. In cases
where obstacles appear in the middle, as shown in the first row, the
consistency score tends to be lower compared to Temporal Average
Pooling, which assigns a uniform score. Additionally, in the second row,

Results of Retrieval. Fig. 5 depicts the retrieval results from rank-1
to rank-3 on MARS-DL. MARS-DL underwent a process of re-detection
of the MARS, reducing videos with severe scale variations or ID mismatching. However, for a small subset of challenging videos in the
8


codex
第五篇第 1 到第 8 页已经包含方法和消融，但方法中段有截断。我会补齐 TRL、SFA、TFA、AFD 的公式段，因为这里决定它是不是“自然推出”的机制。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 3 -l 5 'Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 168 (2025) 111813

M. Kim et al.

of 𝑃 × 𝑃 resolution, which are then flattened into 𝐷 dimensions
2
through linear projection 𝐄 ∈ R(𝑝 ⋅𝐶)×𝐷 . The input sequence consists
of combining the class token 𝜙 𝑐𝑙𝑠 , which represents the identity of
individuals in the video, with the patch embeddings and then adding
position embedding 𝜙 𝑝 to incorporate spatial information. The typical
input sequence 𝐳𝑘 ∈ R(𝑁+1)×𝐷 for the transformer layer can be expressed
as follows:

2.2. Feature-level augmentation
Various augmentation techniques have been primarily applied at
the image level to enhance deep-learning model performance. While
these methods increase the quantity of data by cropping, attaching,
and mixing images within the existing dataset, they have limitations in
terms of diversity. To address this, feature-level augmentation methods
have been proposed in tasks like image classification and semantic
segmentation, promoting various combinations of features. Some methods have been proposed to extract semantic information, generating
new meaningful features [37,38]. Additionally, techniques that create
bias-conflicting samples or use random Gaussian noise have improved
network generalization [39,40]. Temporal feature-level augmentation,
such as shuffle and reordering methods, in video understanding typically focuses on capturing motion by tracking action changes [41–
43]. However, video ReID prioritizes addressing the occlusions caused
by different objects appearing over time [44], rather than merely
capturing motion information. Thus, we propose feature-level augmentation methods specifically designed for video ReID, aiming to increase
training volume and detect the temporal context inconsistencies.

𝐳𝑘 = [𝜙 𝑘𝑐𝑙𝑠 ; 𝐱𝑘1 𝐄; 𝐱𝑘2 𝐄, … ; 𝐱𝑘𝑁 𝐄] + 𝜙 𝑝 ,

(1)

𝐙 = [𝐳1 ; 𝐳2 ; ⋯ ; 𝐳𝑇 ] ∈ R𝑇 ×(𝑁+1)×𝐷 ,

(2)

where 𝑘 signifies the 𝑘th frame. The final representation is obtained
by averaging the class tokens 𝛷 𝑐𝑙𝑠 produced for each frame. For the
identification loss, Circle Loss [50] is employed as follows: 𝑖𝑑 =
𝑐𝑖𝑟𝑐𝑙𝑒 (𝛷̄ 𝑐𝑙𝑠 ). Circle loss aims to maximize the intra-class similarity (𝑠𝑝 )
and minimize the inter-class similarity (𝑠𝑛 ). The circle loss formula is
given by:
[
]
𝐾
∑ 𝐿
∑
𝑐𝑖𝑟𝑐𝑙𝑒 = 𝑙𝑜𝑔 1 +
𝑒𝑥𝑝(𝛾(𝑠𝑗𝑛 − 𝑠𝑖𝑝 + 𝛥)) .
(3)
𝑖=1 𝑗=1

𝐾 intra-class similarity scores are denoted as 𝑠𝑖𝑝 for 𝑖 = 1, 2, … , 𝐾,
and 𝐿 inter-class similarity scores are denoted as 𝑠𝑗𝑛 for 𝑗 = 1, 2, … , 𝐿.
Additionally, 𝛾 represents the scale factor and 𝛥 denotes the margin.

Augmentation in ReID. Research on data augmentation techniques in
ReID has been ongoing to expand the scope of training and handle diverse cases. Zhang et al. [45] addressed overfitting by pretraining with
additional datasets, increasing the learning capacity of the transformer
model. To create various artificial occlusion scenarios, ADP [46] cut
and pasted background patches at the image level to extract features
robust to occlusion. Similarly, COAT [47] enhanced features against
occlusion by merging object features obtained from detection results
with the target person’s features. In contrast, we propose a training
strategy for video ReID that generates examples in various spatial and
temporal domains at the feature level, which are not achievable at
the image level. Our method is efficient as it does not require image
preprocessing or detection models. To the best of our knowledge, we
are the first to apply feature-level augmentation simultaneously in both
spatial and temporal domains for video ReID.

3.2. Overview
A concise overview of FAViT is shown in Fig. 2. Our model consists of three main processes: Token Representation Learning (TRL),
spatio-temporal feature-level augmentation, and individual sub-tasks.
In TRL, learnable tokens are used to embed foreground and background
information separately, generating token-attended patches. Spatial Feature Augmentation (SFA) is employed to create ID samples with new
backgrounds through random combinations of foregrounds and backgrounds. FAViT then performs the ID classification sub-task. For videos
comprised of features from multiple frames, feature augmentation at
the temporal level is conducted by placing features from different
individuals at random positions, followed by Anomaly Frame Detection
(AFD). Finally, the logits used in AFD are used to generate framelevel scores, refining the ultimate video embedding. Spatio-temporal
feature-level augmentation and its sub-tasks are executed only during
the training phase, while in the testing phase, the input passes through
a pure-transformer block after removing the sub-task classifiers.

2.3. Camera information in ReID
ReID aims to identify the same individual appearing in different
cameras, with the camera ID of the captured images always provided. Given this label without additional labeling, various methodologies in ReID leverage camera ID information. He et al. [48] used
position embedding as a memory bank to store camera viewpoint
information, but this approach requires a camera ID label during inference and is inapplicable to datasets with varying camera numbers.
Lei et al. [49] disentangled foreground and background by camera
information and attention mechanism for domain adaptation. However,
two-stream ResNet50 for feature disentangling makes it less efficient.
Kim et al. [16] disentangled camera information in repeating frames
of video ReID, enhancing performance with auxiliary tasks but limiting
itself by not utilizing temporal information in videos. In contrast, our
newly introduced background token draws inspiration from the class
token in the transformer structure, enabling it to learn foreground and
background separation without additional blocks. Furthermore, utilizing the separated embeddings in spatio-temporal feature augmentation
improves the networks’ performance and generalization capacity.

3.3. Token representation learning
Background Token. To enable the network to distinguish between
foreground and background, we embed a class token 𝜙 𝑘𝑐𝑙𝑠 ∈ R1×𝐷
representing the foreground and a background token 𝜙 𝑘𝑏𝑔 ∈ R1×𝐷 representing the rest. To mine background information in patch embedding,
FAViT performs camera ID classification with 𝑐𝑎𝑚 = 𝑐𝑒 (𝜙̄ 𝑏𝑔 ), similar
to what DSANet [16] uses. While DSANet, a CNN-based model, adds a
branch to disentangle camera information, we leverage the structural
attributes of the transformer, allowing FAViT to embed the desired
features with minimal parameters (a learnable token). Furthermore,
the information mining from class token and background token in the
same self-attention block may be complementary because each obtains
information tailored to its purpose. Input sequences of the transformer
layer are prepared by prepending class token 𝜙 𝑘𝑐𝑙𝑠 and background
token 𝜙 𝑘𝑏𝑔 to the patch embeddings.

3. Proposed methods
3.1. Preliminary

𝜙𝑘𝑐𝑙𝑠 ; 𝜙𝑘𝑏𝑔 ; 𝐱𝑘1 𝐄; 𝐱𝑘2 𝐄, … ; 𝑥𝑁
𝜙
𝐳𝑘 = [𝜙
𝑘 𝐄] + 𝑝
𝑇 ×(𝑁+2)×𝐷

] ∈ R𝑇 ×𝐻×𝑊 ×3 ,

𝐙 = [𝐳1 ; 𝐳2 ; ⋯ ; 𝐳𝑇 ] ∈ R

In video ReID, the input is a video 𝑣 = [𝑥1 , 𝑥2 , … , 𝑥𝑇
where 𝑇 is the number of frames in the input video and 𝐻 and 𝑊
represent height and width, respectively. Following He et al. [48], each
frame 𝑥𝑘 is divided into 𝑁 overlapping image patches [𝐱𝑘1 , 𝐱𝑘2 , … , 𝐱𝑘𝑁 ]

(4)
(5)

Token-Attended Patches. The input sequence 𝐙 is fed into a transformer layer composed of a multi-head self-attention (MSA) block and
3

Pattern Recognition 168 (2025) 111813

M. Kim et al.

Fig. 2. Overall framework of FAViT. In the training phase, video frames serve as inputs, which are divided into overlapping image patches. After undergoing linear projection, patch
embeddings are created. 𝛷 𝑐𝑙𝑠 and 𝛷 𝑏𝑔 are prepended to the patch embeddings, forming the input for the transformer. (a) Token Representation Learning involves passing through
transformer blocks, generating an attention map 𝐀𝐭𝐭𝐧 used to embed background and foreground information into token-attended patches. (b) Spatial feature augmentation and
Altered Background ID Classification entail creating samples with a new background by adding randomly rearranged 𝛷 𝑏𝑔 to the existing 𝛷 𝑐𝑙𝑠 , performing an ID classification
sub-task. (c) Temporal feature augmentation and Anomaly Frame Detection involve introducing frames of different individuals at random temporal positions within the video
feature, performing a sub-task to locate these frames. In the testing phase, only 𝛷 𝑓 𝑖𝑛𝑎𝑙 is used as the final representation. Light blue text indicates loss functions, while dark blue
text indicates feature dimensions. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)

a feed-forward network. For the 𝑚th multi-head self-attention block,
the 𝑞𝑢𝑒𝑟𝑦, 𝑘𝑒𝑦, and 𝑣𝑎𝑙𝑢𝑒 are all 𝐙𝑚−1 , and the attention map for each
head is obtained by matrix multiplication of 𝑞𝑢𝑒𝑟𝑦 and 𝑘𝑒𝑦.
𝐙′𝑚 = 𝑀𝑆𝐴(𝐿𝑁(𝐙𝑚−1 )) + 𝐙𝑚−1

(6)

𝐙𝑚 = 𝑀𝐿𝑃 (𝐿𝑁(𝐙′𝑚 )) + 𝐙′𝑚 ,

(7)

𝑚 = 1, … , 𝑀

The loss function used for TRL is as follows:
𝑡𝑟𝑙 = 𝑖𝑑 + 𝜆𝑐𝑎𝑚 𝑐𝑎𝑚 + 𝑠𝑒𝑝 .

Simultaneously employing 𝑐𝑎𝑚 and 𝑠𝑒𝑝 encourages the model to prioritize global information over specific parts, thereby fostering precise
and complementary embeddings for foreground and background.

Taking the average of these attention maps for each head and summing
them for all 𝑀 blocks yields 𝐀𝐭𝐭𝐧 ∈ R(𝑁+2)×(𝑁+2) . Integrating the attention maps from all layers is to capture receptive field information of
various sizes while simultaneously reducing computational complexity.
The first row of 𝐀𝐭𝐭𝐧 represents the scores of all patches for the class
token, while the second row represents the scores for the background
token.
𝐀𝐭𝐭𝐧𝑐𝑙𝑠 = 𝐀𝐭𝐭𝐧[0, 2 ∶ 𝑁 + 2]

(8)

𝐀𝐭𝐭𝐧𝑏𝑔 = 𝐀𝐭𝐭𝐧[1, 2 ∶ 𝑁 + 2]

(9)

3.4. Spatio-temporal feature augmentation
We introduce feature-level augmentation that simulates new scenarios to make the final representation suitable for video ReID. FAViT
generates samples for situations involving background changes and the
emergence of other individuals without requiring additional labeling.
By performing sub-tasks for each scenario, it learns to focus on the
foreground across diverse backgrounds and quantitatively assess frame
reliability. The mini-batch size is denoted as 𝐵 = 𝑁𝑖𝑛𝑠𝑡 × 𝑐, consisting
of a number 𝑁𝑖𝑛𝑠𝑡 of individuals, each with 𝑐 tracklets. We represent
𝑇 class tokens as 𝛷 𝑐𝑙𝑠 = [𝜙 1𝑐𝑙𝑠 , 𝜙 2𝑐𝑙𝑠 , … , 𝜙 𝑇𝑐𝑙𝑠 ] ∈ R𝑇 ×𝐷 and 𝑇 background
tokens as 𝛷 𝑏𝑔 = [𝜙 1𝑏𝑔 , 𝜙 2𝑏𝑔 , … , 𝜙 𝑇𝑏𝑔 ] ∈ R𝑇 ×𝐷 .

Each token-attended patch is expressed as follows:
𝐅𝑐𝑙𝑠 = 𝐀𝐭𝐭𝐧𝑐𝑙𝑠 ⋅ 𝐙𝑀 [2 ∶ 𝑁 + 2],

(10)

𝐅𝑏𝑔 = 𝐀𝐭𝐭𝐧𝑏𝑔 ⋅ 𝐙𝑀 [2 ∶ 𝑁 + 2].

(11)

Spatial Feature Augmentation. Based on the fact that the person’s ID
in the video remains unchanged even when the background changes
and the need to identify the same individual when the background
alters, we can apply feature-level augmentation at the spatial level. 𝛷 𝑐𝑙𝑠
and 𝛷 𝑏𝑔 utilized in TRL are applied. FAViT randomly selects B indices,
rearranging the order of 𝛷 𝑏𝑔 according to the sampled indices. Reordered 𝛷 𝑏𝑔 combines it with 𝛷 𝑐𝑙𝑠 to create a new representation 𝛷 𝑠𝑝𝑎
with a different background, expressed as 𝛷 𝑠𝑝𝑎 = 𝛷 𝑐𝑙𝑠 + 𝑟𝑒𝑜𝑟𝑑𝑒𝑟(𝛷 𝑏𝑔 ).

The rationale for utilizing 𝐙𝑀 lies in leveraging the comprehensive
global semantic and spatial information found in the features of the
last layer [51]. This approach facilitates distinct spatial differentiations between foreground and background without focusing on partial
details.
Using separation loss [16,52], commonly employed for diverse or
mutually independent feature extraction, the network is trained to
reduce the similarity between the class token-attended patch embedding vector 𝐟𝑐𝑙𝑠 = 𝑔𝑎𝑝(𝐅𝑐𝑙𝑠 ) and the background token-attended patch
embedding vector 𝐟𝑏𝑔 = 𝑔𝑎𝑝(𝐅𝑏𝑔 ). ‘gap’ denotes global average pooling,
reducing patch spatial dimensions to compute Eq. (12).
𝑠𝑒𝑝 = 𝑚𝑎𝑥(

𝐟𝑐𝑙𝑠 ⋅ 𝐟𝑏𝑔
, 0)
‖
‖𝐟𝑐𝑙𝑠 ‖ ‖
𝐟𝑏𝑔 ‖
‖ ‖2 ‖
‖ ‖2

(13)

Altered Background ID Classification. FAViT performs the ID Classification sub-task using the representation generated through SFA. This
ensures that FAViT can correctly identify IDs even when presented
with a novel background or in the presence of background cluttering.
The loss function employed for this purpose is denoted as 𝑠𝑝𝑎 =
𝑐𝑖𝑟𝑐𝑙𝑒 (𝛷̄ 𝑠𝑝𝑎 ).
Temporal Feature Augmentation. In Video ReID, it is crucial to
minimize the reliability of frames with low accuracy concerning the

(12)

4

Pattern Recognition 168 (2025) 111813

M. Kim et al.

target identity. To achieve this, FAViT must effectively identify these
inaccurate frames. However, image-level datasets typically do not provide frame-specific accuracy labels. To address this challenge, TFA
introduces frames from different identities, thereby creating samples
that disrupt consistency and facilitating the detection of unreliable
frames. FAViT randomly selects one index from the numbers up to 𝑇 ,
then chooses the class token of the frame corresponding to the random
index in each mini-batch and shifts it by 𝑐. This process creates a
representation 𝛷 𝑡𝑒𝑚𝑝 ∈ R𝑇 ×𝐷 with a different individual frame.

Algorithm 1 Feature-level augmentation vision transformer (FAViT)
Input: Video 𝑣 = [𝑥1 , 𝑥2 , ⋯ , 𝑥𝑇 ] ∈ R𝑇 ×𝐻×𝑊 ×3
Output: The final representation 𝛷̄ 𝑓 𝑖𝑛𝑎𝑙 ∈ R𝐶
1: Input 𝑣 into linear projections
⊳ Get patch embeddings
2: Z ← Prepend 𝜙 𝑐𝑙𝑠 and 𝜙 𝑏𝑔 to patch embeddings
⊳ Add BG token
3: for 𝑚 ← 1 to 𝑀 do
4:
Z𝑚 ← Input Z𝑚−1 into transformer layer
5: end for
6: F𝑐𝑙𝑠 , F𝑏𝑔 ← Calculate the scores Attn and get token-attended patches
7: if Training phase then
8:
𝛷 𝑠𝑝𝑎 ← Combine reordered 𝛷 𝑏𝑔 with 𝛷 𝑐𝑙𝑠
⊳ Implement SFA
𝛷 𝑡𝑒𝑚𝑝 ← Select frame randomly from other individuals
9:
⊳ Implement
TFA
10: else
11:
𝛷 𝑡𝑒𝑚𝑝 ← 𝛷 𝑐𝑙𝑠
12: end if
13: 𝜙 𝑙𝑜𝑔𝑖𝑡 ← Predict the index of anomaly frames
⊳ Implement AFD
14: 𝛷 𝑓 𝑖𝑛𝑎𝑙 ← Calculate frame-level score S and refine 𝛷 𝑐𝑙𝑠
⊳ Implement RL
15: return Average 𝛷 𝑓 𝑖𝑛𝑎𝑙 along the temporal axis

Anomaly Frame Detection. FAViT performs the AFD sub-task using
the representation 𝛷 𝑡𝑒𝑚𝑝 generated through TFA. This process enables
FAViT to identify frames corresponding to different individuals and
further transform these into frame-level scores that refine the final
representation. The relationships between class tokens for each frame
obtained through temporal feature augmentation are modeled through
a single transformer layer.
𝛷 ′𝑡𝑒𝑚𝑝 = 𝑀𝑆𝐴(𝐿𝑁(𝛷 𝑡𝑒𝑚𝑝 )) + 𝛷 𝑡𝑒𝑚𝑝

(14)

′
′
𝛷 ′′
𝑡𝑒𝑚𝑝 = 𝑀𝐿𝑃 (𝐿𝑁(𝛷 𝑡𝑒𝑚𝑝 )) + 𝛷 𝑡𝑒𝑚𝑝

(15)

A logit 𝜙 𝑙𝑜𝑔𝑖𝑡 ∈ R𝑇 ×1 is generated by predicting the index of anomaly
frames using tokens that model the temporal relations between frames.
𝜙 𝑙𝑜𝑔𝑖𝑡 = 𝑟𝑒𝑙𝑢(𝐶𝑜𝑛𝑣(𝛷 ′′
𝑡𝑒𝑚𝑝 ))

It presents challenges due to a considerable amount of occlusion and
low-quality frames. Given that there is only one correct match in the
gallery set, mAP measurement is not applicable.

(16)

MARS-DL [54] is a refined dataset derived from MARS through redetection, reducing frames where the background occupies a substantial portion and removing individuals prone to ID switches from frames.

Specifically, ‘𝐶𝑜𝑛𝑣’ denotes a convolution layer that transforms 𝛷 ′′
𝑡𝑒𝑚𝑝
into 𝜙 𝑙𝑜𝑔𝑖𝑡 . The loss function used in AFD is the cross-entropy loss,
denoted as 𝑡𝑒𝑚𝑝 = 𝑐𝑒 (𝜙 𝑙𝑜𝑔𝑖𝑡 ). During the testing phase, since TFA is
not applied, 𝛷 𝑡𝑒𝑚𝑝 is replaced by the original 𝛷 𝑐𝑙𝑠 . FAViT then generates
logits with decreased consistency due to real-world temporal occlusions
by analyzing the relations among frame features.

MEVID [55] is the latest video ReID dataset, showcasing 158 individuals across 8092 sequences captured by 33 cameras. Unlike traditional
video ReID datasets, individuals in MEVID wear multiple outfits, and
there is no overlap between the camera IDs in the training and testing
sets.

Refinement Learning. Unlike previous methods [42,43] that conclude
with representation learning, FAViT leverages the capabilities learned
through TFA and AFD to score the consistency of frames within a
sequence. The logit is processed into a frame-level score 𝐒 ∈ R𝑇 ×1 ,
which is used to refine the video representation. 𝐒 tends to have lower
values for frames with inconsistent semantics and higher values for
frames with consistent information.
𝐒 = 1 − 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥(𝜙 𝑙𝑜𝑔𝑖𝑡𝑠 )

MARS [56] is a large video ReID dataset that comprises over 20,000
sequences of 1261 individuals, along with 3248 distractors. The videos
were captured using 6 cameras, and the bounding boxes for the individuals were detected using the DPM detector and the GMMCP
tracker.
LSVID [57] is the largest video ReID dataset, featuring 3,772 individuals and 14,943 sequences, which were recorded using 15 cameras.

(17)

Therefore, the frame-level score is manipulated as follows to refine the
video representation:
𝛷 𝑓 𝑖𝑛𝑎𝑙 = 𝐒 ⋅ (𝛷 𝑐𝑙𝑠 ).

Evaluation Metrics We utilize Cumulative Matching Characteristic
(CMC) and mean Average Precision (mAP).

(18)
4.2. Implementation details

𝛷 𝑓 𝑖𝑛𝑎𝑙 is averaged along the temporal axis, and FAViT is trained with
𝑟𝑒𝑓 𝑖𝑛𝑒 = 𝑐𝑖𝑟𝑐𝑙𝑒 (𝛷̄ 𝑓 𝑖𝑛𝑎𝑙 ).

4. Experiments

We adopt the encoder of ViT [17] pre-trained with ImageNet as our
feature extractor. Our network includes 11 out of the total 12 blocks
in ViT base, excluding the final block. To form our training batches,
we randomly choose 4 identities and sample 4 sequences for each
identity, with a batch size of 16. Furthermore, we train the model by
employing the restricted-begin-crop strategy, which involves selecting
8 frames from all the video sequences. We follow the training settings
of CAViT [18]. During testing, features are extracted for all frames
in the sequence, and the final feature is obtained through temporal
average pooling. The distance between the extracted query vector and
gallery vector is calculated using the cosine similarity operation. FAViT
is trained for 30 epochs using a single RTX Titan, which requires 3 h
for the MARS and 6 h for the LSVID. When testing on all frames, the
evaluation process takes 50 min for the MARS and 3 h for the LSVID.

4.1. Datasets and evaluation metrics

4.3. Comparison with state-of-the-art methods

iLIDS-VID [53] is a relatively small video ReID dataset comprising 300
individuals and 600 sequences, captured by two cameras at an airport.

Benchmarks. We categorize SOTA methods on five video ReID benchmarks into CNN-based and transformer-based approaches in Table 2.

3.5. Training and testing phases
During training, the objective function of FAViT is defined as:
𝑡𝑜𝑡𝑎𝑙 = 𝑡𝑟𝑙 + 𝜆𝑠 𝑠𝑝𝑎 + 𝜆𝑡 𝑡𝑒𝑚𝑝 + 𝜆𝑟 𝑟𝑒𝑓 𝑖𝑛𝑒 ,

(19)

where 𝜆𝑠 , 𝜆𝑡 , and 𝜆𝑟 are the weights for the joint learning. In the
testing phase, the classifiers that perform the sub-tasks are removed,
and the final representation is obtained by averaging 𝛷 𝑓 𝑖𝑛𝑎𝑙 along the
temporal axis. The following pseudo-code outlines FAViT, summarizing
the workflow and detailing key operations in Algorithm 1.

5


codex
第六篇是文本检索，我会看它如何把文本描述的“多义性”和图文的“公共语义”构造成一个方法主线，而不是普通的细粒度对齐模块堆叠。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

1823

SPCL: Semantic Polymorphism and Commonality
Learning for Text-Based Person Retrieval
Jiayi Li , Jun Kong , Member, IEEE, Yunde Zhang , Ming Lu , and Min Jiang , Member, IEEE

Abstract—Text-Based Person Retrieval (TBPR) refers to identifying a specific target pedestrian image based on natural language
descriptions. Most previous methods rely on one-to-one alignment
between paired text-image data, ignoring the polymorphic nature
of visual and linguistic information. Moreover, constrained by
ID, earlier methods have shown limited exploration of intraindividual and inter-individual relations. This limitation confines
them to exploring characteristics within individuals, making it
challenging to uncover commonalities and invariants that extend
across IDs (e.g., attributes). Recently, due to the lack of accurate
annotations, exploring attribute-based cross-modal interactions
and alignments has become a significant challenge in TBPR.
To address these issues, we propose a Semantic Polymorphism
and Commonality Learning (SPCL) framework. First, we present
Relation-Sensitive Semantic Polymorphism Alignment (RSSPA)
and ID-Based Semantic Polymorphism Alignment (IBSPA) to
explore ID-limited Feature Redistribution. Second, we transcend
the constraints of ID, leveraging ID-Free Attribute Alignment
(IFAA) from a macro perspective to explore commonalities and
invariants based on attribute features. Finally, from a micro
perspective, we design Attribute Prior Fusion Reconstruction
(APFR) to optimize the attention of our model, exploring the
positive impact of attribute priors on cross-modal interaction.
Experiments on CUHK-PEDES, ICFG-PEDES and RSTPReid
show that our method achieves state-of-the-art performance on
Rank-1, mAP and mINP.
Index Terms—Text-based person retrieval, semantic polymorphism, semantic commonality, ID-free attribute alignment.

I. I NTRODUCTION
EXT-BASED person retrieval (TBPR) is defined as the
process of employing natural language to retrieve a
specific target person in an extensive gallery of candidate
images [1], [2], [3], [4]. Compared with traditional unimodal
person retrieval methods [5], [6], TBPR offers the advantage
of using text queries to facilitate a more flexible and easily
accessible retrieval process by describing the attributes of
the target person. Unlike general multi-category text-to-image

T

Received 24 October 2024; revised 11 February 2025 and 1 March 2025;
accepted 16 August 2025. Date of publication 20 August 2025; date of
current version 5 February 2026. This work was supported in part by the
National Natural Science Foundation of China under Grant 62371208 and
Grant 62371209 and in part by the Postgraduate Research and Practice
Innovation Program of Jiangsu Province (the Fundamental Research Funds
for the Central Universities) under Grant KYCX24 2643. This article was
recommended by Associate Editor Q. Ye. (Corresponding author: Min Jiang.)
Jiayi Li, Ming Lu, and Min Jiang are with the Engineering Research Center
of Intelligent Technology for Healthcare, Ministry of Education, Jiangnan
University, Wuxi 214122, China (e-mail: minjiang@jiangnan.edu.cn).
Jun Kong and Yunde Zhang are with the Key Laboratory of Advanced Process Control for Light Industry (Ministry of Education), Jiangnan University,
Wuxi 214122, China.
Digital Object Identifier 10.1109/TCSVT.2025.3601071

cross-modal retrieval tasks [7], [8], TBPR focuses exclusively
on the category of pedestrian, which necessitates processing
finer-grained details and demands stricter expressions of crossmodal consistency.
In TBPR, visual appearance is subject to varying degrees
of discrepancy due to changes in lighting conditions and
viewing angles. Similarly, textual expressions exhibit differences due to variations in sentence structure, lexical choices,
and descriptions of object categories. This variation in the
representation of the same concept reflects a core element
of TBPR, namely, semantic polymorphism. Furthermore, the
existence of similar semantic meanings across individuals
under different IDs reflects another core element of TBPR,
namely, semantic commonality. TBPR primarily addressing
the inherent challenge posed by modality heterogeneity. However, the presence of semantic polymorphism and commonality
further increases the difficulty of cross-modal alignment.
As in Fig. 1a, the two pedestrians belong to the same
identity, with one facing front and the other facing back.
Despite similar visual and textual expressions, subtle differences exist between these two images and text segments. For
instance, the left image shows a white shirt with patterns
without visible shoes, while the right image, taken from
behind, displays the shoes but not the shirt. These differences
are also evident in the corresponding texts, which are similar yet complementary, illustrating semantic polymorphism.
Cross-modal semantics with one-to-one complete matching
can be regarded as symmetric relations. We coin the term
strong correspondence for the symmetric relations that can
be established between matched text-image pairs of the same
entity. Correspondingly, we term the phenomenon of similar yet complementary cross-modal visual/textual semantic
descriptions as weak correspondence. To achieve cross-modal
alignment, earlier approaches [9], [10], [11] primarily focused
on one-to-one alignment, ignoring the tight relations among
images or texts, for the identical pedestrian, as well as the
interaction of text and images. Based on this, we explore the
semantic polymorphism and relation sensitivity of cross-modal
combinations that are similar yet complementary. Furthermore,
to deeply delve into semantic polymorphism and achieve
a more regular feature distribution, we explore the semantic polymorphism in each modality to assist cross-modal
matching.
In recent years, an increasing amount of research has
been devoted to exploring the critical role of attributes in
TBPR tasks [4], [12], [13], [14], [15]. For unimodal person

1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

1824

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

Fig. 1. Multiple Relational Mapping (a) Strong and Weak Correspondences in
TBPR. Due to incomplete descriptions or variations in viewpoints, the images
and texts collected from the same person exhibit subtle variance. For example,
the left image shows strong alignment with the top text descriptions, such as
clothing type, clothing color, and hair style, while shows weak correspondence
with the bottom one. (b) Inter-ID Attribute Correspondences. Partial matches
between textual and image attributes can also occur across different IDs.
(c) Different Cross-Modal Attribute Alignment Methods.

re-identification tasks, inevitable appearance disparities
induced by variations in illumination, viewpoint, and other
factors cause confusion in feature matching. Attributes can
serve as higher-level discriminative traits, facilitating the
filtering of visual appearance discrepancies in person retrieval
tasks. For TBPR, the rich multi-granularity attributes in
textual datasets provide valuable prior information, effectively
helping to mitigate the impact of visual noise. Moreover, as
shown in Fig. 1b, commonalities can be found across different
IDs for multi-granularity attribute priors, a phenomenon we
term attribute commonality. Most previous attribute-based
methods [4], [16], [17] were limited by ID, focusing on the
relation between paired text-image sets within the same ID.
This design will inevitably force the separation of visual
representations and their related textual attributes, especially
when they share the same attributes but belong to different
IDs. This can blur the model’s understanding of attributes,
which we consider unfriendly to the model. Based on this,
we investigate the commonalities between textual attributes
and images in a ID-free manner. We disregard ID limitations
and explore the correlations between textual attributes and
different images from a macro perspective, even across

different IDs. The macro perspective refers to an examination
of the overall correspondence between global-level textual
attributes and images within the current batch.
Additionally, most previous approaches primarily utilized
loss functions to explicitly perform cross-modal attribute alignment [14], [18], [19]. As shown in Fig. 1c, these methods
typically either select portions of text (Fig. 1c (i)) or segment
the image (Fig. 1c (ii)) to extract attribute information, aiming
to bridge visual and textual representations. However, such
approaches often lack a deep exploration of multi-granularity
attribute priors in cross-modal interactions. Based on this,
we further establish the relationship between overall visual
elements and discrete textual attributes at a micro level. We
explore the positive impact of text-based multi-granularity
attribute priors on guiding a more refined intra-individual
cross-modal fusion reconstruction (Fig. 1c (iii)). This exploration of micro level is dedicated to identifying fine-grained
characteristics that reflect the intricate relationships within
paired text-image instances.
Building upon prior research, in this paper, we propose
Semantic Polymorphism and Commonality Learning (SPCL)
based on the investigation of semantic polymorphism and
attributes. Firstly, we propose Relation-Sensitive Semantic
Polymorphism Alignment (RSSPA), which delves into the
semantic polymorphism across modalities. Specifically, under
the constraint of IDs, we align features between different modalities with the same ID adaptively according to
their similarity. Concurrently, we propose ID-Based Semantic
Polymorphism Alignment (IBSPA) to investigate intra-modal
semantic polymorphism. This dual strategy, integrating RSSPA
and IBSPA, aims to capture ID-limited semantic distribution
and construct diverse relational polygons in and across modalities. Second, we design ID-Free Attribute Alignment (IFAA)
to explore the commonalities between textual attributes and
images from a macro perspective. IFAA aligns with human
natural cognitive patterns by focusing solely on the correspondence between textual attributes and images, unconstrained
by ID limitations. It adaptively establishes alignment relationships between various attributes and different images. Finally,
we propose Attribute Prior Fusion Reconstruction (APFR)
to filter noise and enhance salient information for crossmodal matching from a micro perspective. APFR extracts
multi-granularity attribute priors from sentences to guide intraindividual cross-modal fusion and attribute reconstruction. By
implicitly achieving cross-modal alignment, APFR enhances
the integrated understanding of attributes. IFAA and APFR
respectively investigate the commonalities and saliencies of
attributes in TBPR, enabling our model to focus more on
the discriminative attribute components during cross-modal
alignment.
The main contributions are summarized as follows:
• We propose RSSPA and IBSPA based on ID constraints,
constructing diverse relational polygons in and across
modalities to address the challenges of semantic polymorphism.
• We present IFAA, disregarding ID constraints, to establish
a more comprehensive attribute-image relation system at
a macro level, exploring their commonalities.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

LI et al.: SPCL: SEMANTIC POLYMORPHISM AND COMMONALITY LEARNING FOR TBPR

• We design APFR to further guide intra-individual crossmodal fusion and attribute reconstruction at a micro level,
purifying the critical information in each modality.
II. R ELATED W ORK
A. Text-based Person Retrieval
Text-Based Person Retrieval (TBPR) is a novel task that
integrates cross-modal matching techniques into the traditional
person re-identification field, aiming to retrieve pedestrians
based on natural language descriptions. This concept was
first introduced by [20] and published the pioneering dataset
CUHK-PEDES. Early methods primarily used dual-stream
networks to extract global features, exploring one-to-one alignment relations between individuals [21], [22], [23], [24], [25].
These methods typically employed VGG [26] or ResNet [27]
as image encoders and LSTM [28] or BERT [29] as text
encoders, aligning cross-modal features at the end of the
network using a loss function. Moreover, some approaches
began to investigate the intrinsic relations in individuals [14],
[30], [31], [32], [33], introducing local-level alignment on the
basis of global-level text-image alignment. References [14],
[31], [32], [34], and [35] explicitly aligned image regions
with text fields, while [36], [37], [38], [39], [40] employed
attention mechanisms to implicitly align cross-modal semantic
information. In recent years, with the rise of large-scale
Vision-Language Pretraining (VLP) models [41], [42], an
increasing number of studies have adopted VLP with finetuning to enhance the underlying alignment [43], [44], [45],
[46]. However, these studies primarily focused on performing
one-to-one text-image alignment under ID constraints to study
cross-modal matching tasks, neglecting the semantic polymorphism present in real-world.
Unlike previous works, we investigate intra-ID semantic
polymorphism by exploring the differential strength of associations among individuals, emphasizing relation sensitivity.
Furthermore, we leverage the attribute commonality to explore
semantic associations between different IDs, reinforcing discriminative semantic information in both visual and textual
modalities.
B. Attribute-based Representation Learning
Attribute-based Representation Learning (ABPL) focuses
on leveraging attribute information to learn more meaningful
and discriminative feature representations. In person reidentification, early works either manually annotated attributes
or utilized pre-trained classifiers to extract attribute features
for cross-modal matching [47], [48], [49]. With the evolution of TBPR tasks, textual datasets provide increasingly
rich and complete descriptions. This advancement allows
for the extraction of attributes from both visual and textual descriptions, facilitating cross-modal matching. Reference
[14] attemptes to automatically achieve local-level matching
between text and images by horizontally segmenting images
and employing an attention mechanism. Moreover, some studies leverage external toolkits [50], [51] to extract attribute
words or phrases [18], [19], [52], which are then matched with
visual information in a cross-modal manner. References [16]

1825

and [17] employ prompt learning to flexibly extract understandable attribute information for cross-modal alignment.
Reference [4] establishes an attribute vocabulary and conducts
research based on the frequency of attribute occurrences in the
dataset, further addressing the long tail effect of attributes.
However, previous attribute-based works were limited to
paired text-image instances, leading to conflicting when the
same attribute appears in different IDs. Additionally, these
methods focus on directly aligning attribute information using
loss functions at the end, lacking in-depth exploration of multigranularity attribute priors in cross-modal interactions.
In this paper, we establish connections between images of
different IDs and various textual attributes, aiming to explore
the macro-level commonalities between ID-free attributeimage pairs. Furthermore, we further explore the deep
perception of attributes in cross-modal interactions in individuals, guiding our model at a micro level to focus on more
discriminative multi-granularity attribute priors.
III. M ETHOD
In this section, we present the proposed Semantic Polymorphism and Commonality Learning (SPCL) framework.
An overview of SPCL is illustrated in Fig 2. It extracts
features using a dual-stream backbone network. Our input
includes original caption-image pairs and multi-granularity
attribute priors extracted from captions. SPCL is optimized
through three branches: ID-limited Feature Redistribution
(IIFR), ID-Free Attribute Alignment (IFAA), and Attribute
Prior Fusion Reconstruction (APFR). Notably, IIFR comprises
two components: Relation-Sensitive Semantic Polymorphism
Alignment (RSSPA) and ID-Based Semantic Polymorphism
Alignment (IBSPA). Ultimately, our model evaluates the
similarity between target text and candidate images. The
subsequent modules will be discussed in detail in dedicated
subsections.
A. Visual and Textual Feature Extraction
Inspired by recent advancements in Vision-language Pretraining (VLP) models, we leverage CLIP to initialize our
model to acquire enhanced cross-modal prior knowledge.
1) Image Feature Extraction: We use CLIP-ViT as the
image encoder. First, input images are uniformly resized to
dimension H × W, followed by dividing into patches of size
H × W/p2 , where p is the patch size. Next, these patches are
then linearly projected into 1-dimensional tokens. Positional
encoding is applied to these tokens, accompanied by the
addition of a [CLS] token to form the sequence {Icls , I1 , . . . , In }.
Finally, this sequence is fed into the image encoder to generate
visual representations, with Icls being the global representation.
2) Text Feature Extraction: We adopt the CLIPTransformer as the text encoder, comprised of L Transformer
[53] blocks. First, we utilize BPE [54] to convert input text
into a sequence of tokens. The tokens are then prefixed
with [SOT] and suffixed with [EOT] to denote the start
and end, respectively. Through truncation or padding with
zeros, this token sequence is adjusted to a fixed length
of 77, forming the sequence {T sot , T 1 , . . . , T eot }. Next, for

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

1826

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

Fig. 2. SPCL consists of a text encoder, an image encoder, and three main branches. The ID-limited Feature Redistribution branch achieves ID-limited alignment
through two components: Relation-Sensitive Semantic Polymorphism Alignment (RSSPA) and ID-Based Semantic Polymorphism Alignment (IBSPA). The
ID-Free Attribute Alignment (IFAA) branch explores inter-ID attribute correspondences. The Attribute Prior Fusion Reconstruction (APFR) branch facilitates
attribute-guided cross-modal interaction. We perform end-to-end training, and during inference, we use only the two encoders to encode features and compute
global similarity.

captions, we apply a mask operation, masking tokens with a
15% probability for text augmentation. For multi-granularity
attribute priors, we retain their positions in the original
sentence and set the rest of the positions to zero without
masking, resulting in a sequence {P sot , P1 , . . . , Peot }. Finally,
these modified sequences is inputted into the text encoder
to obtain text representations and multi-granularity attribute
priors representations. Here, T eot represents the global text
representation and Peot represents the global multi-granularity
attribute priors representation.
B. ID-limited Feature Redistribution
In IIFR, we leverage on the inherent semantic polymorphism of images and texts, exploring Relation-Sensitive
Semantic Polymorphism Alignment (RSSPA) across modalities while concurrently employing ID-Based Semantic Polymorphism Alignment (IBSPA) in each modality. Ultimately,
our goal is to establish a diverse relational polygon for more
convincing alignment, thereby enhancing the credibility and
effectiveness of cross-modal understanding.
1) Relation-Sensitive Semantic Polymorphism Alignment:
Consistent with trainditional approaches [20], [36], we investigate the inter-modal alignment under the constraint of identity.
Unlike conventional past efforts that primarily focus on aligning one-to-one paired text-image instances (Fig. 3a), we
explore a more diverse network of relations. We adaptively
explore Relation-Sensitive Semantic Polymorphism Alignment, which takes into account the varying strengths of
associations ranging from strong to weak between modalities
under ID constraints. Specifically, we establish strong and

Fig. 3. Exploration of Alignment Manners in TBPR. In the feature representation, different colors denote distinct IDs. For the aligned connection lines,
solid lines represent strong links, while dashed lines indicate weak links, with
different colors of dashed lines representing varying strengths of weak links.

weak correspondence in IDs based on text-image similarity
(Fig. 3b). Directly paired text-image instances maintain a
strong relationship, while unpaired instances in the same ID
establish weaker relationships based on cross-modal similarity
to explore Relation-Sensitive Semantic Polymorphism Alignment.
In this module, we primarily use contrastive loss to bring
various cross-modal relations closer, complemented by KL

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

LI et al.: SPCL: SEMANTIC POLYMORPHISM AND COMMONALITY LEARNING FOR TBPR

1827

divergence to balance the distributions. Specifically, we compute the cosine similarity between modalities based on the
textual features extracted from the text encoder and image
features derived from the image encoder:
>

i
i
S (Icls
, T eot
)=

i
I i cls · T eot
,
i
i
||Icls
||2 × ||T eot
||2

(1)

where i is the index and i ∈ {1, . . . , N}. Next, we calculate the
strength of relationships in an ID based on similarity scores.
The intra-ID similarity score between text and image features
is computed as follows:
v
u
i
u S (I i , T eot
)
i2t
,
(2)
S score (i) = t Pm cls j
j
j=1 S (Icls , T eot )
where S i2t
score (i) denotes the sensitivity score, which quantifies
the degree of responsiveness to variations in similarity. m
represents the count of individuals in each ID group in a given
batch. To ensure that paired text-image instances have a strong
correspondence, we enhance their alignment by adding the
identity matrix to the sensitivity coefficient matrix. Then, we
constrain the values of the resultant matrix to be no greater
than one, keeping the sensitivity coefficients in the range
[0, 1]. This can be expressed as follows:
i2t
S i2t
sen = min(S score + E, 1),

(3)

where E is the identity matrix and S i2t
sen is the sensitivity
coefficient.
Next, we extend the one-to-one cross-modal alignment to a
relation-sensitive semantic polymorphism alignment between
each image and all texts under the same ID. The description
is as follows:
g
x
X
exp(S (Icls
)/τ)
, T eot
−S i2t
sen (i)
L (i) =
log PN
.
a
xi
G(xi )
i
a=1 exp(S (Icls , T eot )/τ)
i

i2t

(4)

g∈G(x )

where τ is the temperature hyperparameter. N is the total
number of texts in a given batch. G(xi ) = {g ∈ 1 . . . C : xg = xi }
represents the collection of all text-related features that match
the visual features in the same ID group for the present batch.
Similarly, we compute the cross-modal alignment loss for
each text item in relation to all images sharing the same ID
in the current batch:
Lt2i (i) =

xi
h
X
exp(S (T cls
, Ieot
)/τ)
−S t2i
sen (i)
log
,
PN
i
i
x
a
H(x )
i
a=1 exp(S (T cls , Ieot )/τ)

N

N

i=1 j=1

(6)

(7)
(8)

where δ represents a small positive value to ensure the
denominator is not zero. The loss function after incorporating
the regularization term can be expressed as:
i2t
i2t
Li2t
RS ensi = L + αLre ,

(9)

t2i
t2i
Lt2i
RS ensi = L + αLre ,

(10)

where Li2t and Lt2i denote the mean of Li2t and Lt2i , respectively.
The final expression of the loss calculation formula for
RSSPA can be articulated as follows:
LRS ensi =

1 i2t
(L
+ Lt2i
RS ensi ).
2 RS ensi

(11)

2) ID-Based Semantic Polymorphism Alignment: Similarly,
we achieve ID-Based Semantic Polymorphism Alignment
(IBSPA) in each modality. IBSPA narrows the feature representation gap among individuals sharing the same ID, with
the objective of exploring the maximal similarity amongst
same-ID entities in each modality. Through the synergistic
interplay of inter-modal and intra-modal alignment operations,
we achieve diverse correspondence based on relations (Fig. 4).
This relation establishment facilitates a better understanding
of various semantic dimensions of the same person, thereby
enhancing the model’s analysis and comprehension of semantic polymorphism.
In this module, we compute the intra-modal fundamental
loss, with the objective of aligning the text/image feature
representations of the same ID in each modality for a given
batch:
i

Li2i (i) =

where H(xi ) = {h ∈ 1 . . . C : xh = xi } represents the collection
of all image-related features that match the textual features in
the same ID group for the present batch.
To balance the distance distribution between text and image
according to the ID distribution, we use Kullback-Leibler
(KL) divergence as a regularization term to adjust the distance
distribution. The regularization term is calculated as follows:
1 X X i2t
Ai2t (i)
A (i) log
,
N
B(i) + δ

j
i
exp(S (Icls
, T eot
))
Ai2t (i) = PN
, j ∈ {1, . . . , N},
i
a
a=1 exp(S (Vcls , T eot ))
l(i)
B(i) = PN
,
j=1 l( j)

(5)

h∈H(x )

Li2t
re (i) =

Fig. 4. Based on inter-modal relation-sensitive correspondences and intramodal feature aggregation, we achieve comprehensive relation-sensitive
semantic polymorphism alignment.

y
r
exp(S (Icls
, Icls
)/τ)
−1 X
log
,
PN
i
i
y
a
R(y )
a=1 exp(S (Icls , Icls )/τ)
r∈R(yi )

(12)

where R(yi ) = {r ∈ 1 . . . C : yr = yi } denotes the image or text
features that share the same ID in each modality.
Next, we employ Kullback-Leibler (KL) divergence as a
regularization term within each modality to balance the feature
distribution. The loss within each modality is collaboratively
computed as:
i2i
i2i
Li2i
IDBased = L + βLre ,

(13)

t2t
t2t
Lt2t
IDBased = L + βLre ,

(14)

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

1828

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

toolkit [50] to extract k sets of noun phrases or k sets of nouns
and adjectives from each caption. Subsequently, we encode
the extracted multi-granularity attribute priors in the same
manner as the original captions. The encoded multi-granularity
attribute priors features can be denoted as {P sot , P1 , . . . , Peot }k1 .
Here, Peot represents the global feature representation at
the multi-granularity attribute level. To achieve attribute-level
alignment, we first calculate the alignment loss between an
image and all texts in a batch, denoted as:
j
i
exp(S (Icls
, Peot
)/τ)
,
Li2p
(i)
=
−
log
PN
IDFree
i
a
a=1 exp(S (Icls , Peot )/τ)
j ∈ {1, . . . , N}.

Fig. 5. ID-Free Attribute Alignment matrix details. In the matrix, darker
colors represent higher matching similarity.

where Li2i and Lt2t denote the mean of Li2i and Lt2t , respect2t
tively. Li2i
re and Lre are the KL regularization term for each
modality.
The final expression of the loss calculation formula for
IBSPA can be articulated as follows:
t2t
LIDBased = Li2i
IDBased + LIDBased .

(15)

C. ID-Free Attribute Alignment
Through the collaborative efforts of RSSPA and IBSPA,
our model has been trained to establish diverse relations
among individuals based on their IDs. In the ID-Free Attribute
Alignment (IFAA) module, we transcend the limitations of
IDs to adaptively explore attribute alignment across different
IDs. We have observed that descriptions involving multigranularity attribute priors, such as blue backpack, appear in
different forms across images with various IDs. Specifically,
as in shown Fig. 5, these priors describing multi-granularity
attributs can be observed in images in three distinct ways: a)
Full match. The object in the image completely aligns with
the description, such as seeing a backpack that is confirmed
to be blue. b) Partial match. Only part of the description
is matched. For example, the image may show a backpack
that is not blue, or it may show the color blue without
mentioning a backpack. c) No match. Even though the image
and description belong to the same ID, the pedestrian in
the image does not exhibit the described multi-granularity
attribut priors at all. From these observations, we can infer that
the expression of attribute features is not strictly constrained
by specific IDs but instead exhibits a cross-ID commonality.
To investigate attribute commonality in TBPR, we propose
the IFAA module. This module aims to facilitate attribute
alignment and understanding at a macro level. The macro level
is dedicated to exploring the correspondence between all crossmodal textual and image attributes from the perspective of the
current batch. It transcends ID boundaries. This enhancement
allows the model to generalize and discriminate based on
shared attribute information across different identities.
Before performing the macro-level attribute alignment,
we leveraged the richness of our text dataset to extract
attribute representations at various granularities, forming
multi-granularity attribute priors. First, we utilize a language

(16)

Next, we align the textual multi-granularity attribute priors
with images using the relation-sensitive strength. This involves
optimizing the contrastive loss based on the similarity. We
calculate the sensitivity coefficient between textual multigranularity rattribute prior features and image features as
S i2p
score (i) =
as:

S (I i ,Pieot )
Pm cls j
. Therefore, Eq. 16 can be optimized
j
j=1 S (Icls ,Peot )

j
i
exp(S (Icls
, Peot
)/τ)
i2p
Li2p
(i)
=
−S
(i)
log
.
P
score
IDFree
N
i
a
a=1 exp(S (Icls , Peot )/τ)

(17)

The final expression of the loss calculation formula for
IFAA can be articulated as follows:
1
p2i
+ LIDFree
),
(18)
LIDFree = (Li2p
2 IDFree
p2i
i2p
where Li2p
IDFree and LIDFree denote the mean of LIDFree and
p2i
LIDFree , respectively.

D. Attribute Prior Fusion Reconstruction
Through the previous modules, we have achieved macrolevel exploration. To further investigate the fine-grained
hierarchical attribute characteristics at a micro level, we propose the Attribute Prior Fusion Reconstruction (APFR). The
micro level involves a more fine-grained exploration within
individual instances, focusing on the detailed cross-modal
interactions between paired text and image features. In APFR,
we study the positive impact of multi-granularity attribute
priors to guide intra-individual cross-modal interactions. This
enables the model to concentrate on enhancing its attention
towards multi-granularity attribute priors embedded in both
visual and textual modalities, thereby filtering visual noise
and improving its discriminative capabilities. We first fuse
the visual and textual modality information in APFR. Next,
we extract multi-granularity attribute priors from the fused
features. Through the synergistic integration of the IFAA and
APFR modules, we fortify our model’s ability to discern
attributes across visual and textual domains.
As shown in Fig. 2, the process comprises three main
steps: (1) cross-attention, where text features are used as
Queries and image features as Keys and Values; (2) a stack of
4-layer Transformer blocks, each incorporating Self-Attention
and Feed-Forward Networks (Multi-Layer Perception, MLP);
and (3) an inference component constructed with another MLP

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

LI et al.: SPCL: SEMANTIC POLYMORPHISM AND COMMONALITY LEARNING FOR TBPR

for higher-level reasoning. For cross-modal fusion, we treat
the text as the query, denoted as qi , and the image as the key
and value, denoted as ki and vi respectively. All these features
are subjected to normalization beforehand. Subsequently, they
are fed into the Multi-head Cross Attention (MCA) layer. The
formulation is as follows:
f i = MCA(qi , ki , vi ),

(19)

where d denotes the feature dimension, i represents the index,
i ∈ {1 . . . N}. Next, we input f i into 4-layer Transformer
blocks for deep cross-modal information fusion. The fused
cross-modal features are then processed through a MultiLayer Perception (MLP) for inference. This process can be
represented as follows:
pi = MLP(T rans f ormer( f i )).

(20)

Finally, we align the inference results with the original
multi-granularity attribute priors, constraining the inference
results and enhancing the capability of the encoders to capture
multi-granularity attributes. For the processing of multigranularity attribute priors, we retain their original positions
in the sentence and encode them into token sequences using
BPE [54]. For each sentence, we extract k sets of semantic
priors, with the kth semantic prior of the ith sentence denoted
as zik . The loss is represented as follows:
X X X ij
1
zk
LPRecon = −
|C||D||K|
i∈|C| j∈|D| j∈|K|

exp(pi j )

× log P|V|

c=1 exp(p

ic )

,

(21)

where |C| is the set of tokens for the textual multi-granularity
attribute priors and |D| is the vocabulary size. |K| represents
the total number of semantic priors extracted for each sentence. We aim to narrow the gap between the fused feature
representations and the textual multi-granularity attribute prior
representations using Eq. 21. This emphasizes the positive
impact of multi-granularity attribute priors on cross-modal
interactions.
1) Synthesis: Through the optimization facilitated by four
distinct modules, our ultimate framework is characterized by:
L = λ1 LRS ensi + λ2 LIDBased + λ3 LIDFree + λ4 LPRecon ,

(22)

where λ1 , λ2 , λ3 and λ4 refer to the weights of the RSSPA,
IBSPA, IFAA and APFR components, respectively.
IV. E XPERIMENTS
A. Experimental Settings
1) Datasets: We evaluate our approach on three prominent
datasets in the TBPR domain: CUHK-PEDES [20], ICFGPEDES [36] and RSTPReid [35]. CUHK-PEDES dataset,
derived from five reidentification datasets, encompasses 13,003
unique identities, 40,206 images, each annotated with two
captions, amassing to 80,412 natural language descriptions.
The training set includes 11,003 identities, 34,054 images,
and 68,108 captions. Validation and testing sets hold 3,078
and 3,074 images respectively, paired with 6,158 and 6,156

1829

descriptions, all featuring 1,000 distinct identities. ICFGPEDES dataset, sourced primarily from MSMT17 [60],
presents a larger image pool and more elaborate captions,
featuring 54,522 images linked to 4,102 identifiers, each with
a singular description. It comprises 34,674 image-caption
pairs for training and 19,848 for testing, with the training
set encompassing 3,102 unique identities and the test set
featuring 1,000 distinct identities. RSTPReid dataset covering
4,101 identities with five images per identity,totaling 20,505
images. The images in RSTPReid exhibit greater variation in
shape and deformation. Each image is paired with two text
descriptions. The training set includes 3,701 identities, while
both the validation and test sets each contain 200 identities.
2) Evaluation Metrics: On both benchmarks, we evaluate
our method using three metrics: Rank-k (k=1, 5, 10), mean
Average Precision (mAP), and mean Inverse Negative Penalty
(mINP) [62]. Rank-k is a commonly used metric for evaluating
the performance of information retrieval systems, focusing on
the accuracy of the top-k positions in the retrieval results.
Mean Average Precision (mAP) considers both the precision
and recall of the retrieval results. It is obtained by calculating
the average precision for each query, which is the proportion
of relevant items retrieved among all retrieved items. Unlike
mAP, which mainly focuses on the ranking positions of correct
predictions, mINP emphasizes the localization of the most
challenging matches. It reflects the model’s ability to handle
the most difficult cases in retrieval by considering the position
of the last correct match.
3) Implementation Details: Our backbone primarily consists of two encoders: a visual encoder (CLIP-ViT-B/16 or
CLIP-ViT-L/14) and a text encoder (CLIP-Transformer). Additionally, we initialize the cross-modal interaction network
randomly, setting the dimensions to 512 and the number
of heads to 8. Our experiments are performed on a single
NVIDIA A800 GPU, running for 60 epochs on CUHK-PEDES
and ICFG-PEDES, and 20 epochs on RSTPReid, with a minibatch size of 64. During training, we resize every image input
to 384 × 128 and apply random horizontal flipping, random
erasing, and random cropping with consistent padding for
image augmentation. For text augmentation, we mask the text
input tokens with a probability of 15 %. We use the Adam [63]
optimizer. We adopt a five-epoch warm-up phase to increase
the learning rate linearly from 1×10−6 to 1×10−5 , followed by
a cosine learning rate schedule for decay. We utilize random
and identity sampling with a ratio of 2:1. The temperature
parameter τ is set to 0.02. The regularization coefficient α is
set to 0.1, and β is set to 1. The weights of the components are
set as follows: λ1 for RSSPA to 30, λ2 for IBSPA to 30, λ3 for
IFAA to 1, and λ4 for APFR to 1. Note that the above settings
are primarily based on the ViT-B/16 configuration. When the
image encoder is switched to ViT-L/14, the setting of λ3 for
IFAA is adjusted to 1.8 on CUHK-PEDES and ICFG-PEDES,
and λ4 for APFR is adjusted to 1.2 on RSTPReid.
B. Comparison with State-of-the-Art Methods
We compared our method with state-of-the-art methods on
CUHK-PEDES, ICFG-PEDES, and RSTPReid benchmarks
and evaluated the impact of different scales of CLIP-ViT on

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

1830

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

TABLE I
C OMPARISON W ITH SOTA M ETHODS ON THE CUHK-PEDES DATASET. RN R EFERS TO U SING R ES N ET AS THE F EATURE E NCODER . X FORMER R EFERS
TO U SING T RANSFORMER AS THE F EATURE E NCODER . I N THE TABLE , THE H IGHEST VALUES A RE I NDICATED IN B OLD , W HILE THE
S ECOND -H IGHEST VALUES A RE U NDERLINED . PARAM S HOWS THE N UMBER OF PARAMETERS FOR E ACH M ETHOD D URING T RAINING

model performance. In our comparison, the baseline is established by fine-tuning CLIP using its original loss function.
1) Comparison on CUHK-PEDES: As illustrated in
Table I, we compare our method with state-of-the-art
approaches on CUHK-PEDES. Our model demonstrates
exceptional performance with a 12-layer ViT, achieving stateof-the-art results in Rank-1 accuracy, mAP, and mINP, with a
particularly significant improvement in mAP compared to the
second-best method. When extending the ViT to 24 layers,
our model exhibits substantial improvements across all metrics. Unlike previous methods that exhibited a large disparity
between Rank-1 accuracy and mAP, our model balances these
metrics, indicating that SPCL tends to capture more stable
features. Additionally, the considerable enhancement in mINP
demonstrates our model’s significant advantage in identifying
challenging samples.
Compared to other methods, our model achieves superior
performance with the same number of trainable parameters
when using CLIP-ViT-B/16. Our model even outperforms
IRRA, which has a larger number of trainable parameters.
In our method, the adoption of a larger model leads to
a significant performance improvement. This indicates that
large-scale trainable deep networks hold strong potential for
fine-grained mining in TBPR.
2) Comparison on ICFG-PEDES: As shown in Table II,
we validate the effectiveness of our method on ICFG-PEDES.
We compare various scales of ViT as the visual encoder.
Even with a smaller-scale ViT, our model achieves state-ofthe-art results in Rank-1 accuracy, mAP, and mINP. When
using a larger-scale ViT, our model continues to exhibit

TABLE II
C OMPARISON W ITH SOTA M ETHODS ON THE ICFG-PEDES DATASET

significant improvements in Rank-1 accuracy, mAP, and mINP.
The ICFG-PEDES dataset poses a greater challenge than
the CUHK-PEDES dataset due to factors such as lighting
conditions, viewing angles, and textual complexity. Despite
these increased difficulties, our model still achieves substantial
improvements in mAP and mINP, demonstrating its stability
in exploring cross-modal semantic representations.
3) Comparison on RSTPReid: As shown in Table III, we
validate the competitiveness of our method on RSTPReid.
It can be observed that our method demonstrates substantial
superiority. Compared to the baseline, our model achieves
comprehensive improvements. When using the small-scale ViT

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.


codex
第六篇的核心术语很密：强对应、弱对应、语义多态、属性共性、宏观和微观。刚才第 1 到第 8 页覆盖了方法框架，但中间 IIFR 段有截断，我会补读方法页来保证四个模块关系讲准。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 4 -l 7 'SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
1826

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

Fig. 2. SPCL consists of a text encoder, an image encoder, and three main branches. The ID-limited Feature Redistribution branch achieves ID-limited alignment
through two components: Relation-Sensitive Semantic Polymorphism Alignment (RSSPA) and ID-Based Semantic Polymorphism Alignment (IBSPA). The
ID-Free Attribute Alignment (IFAA) branch explores inter-ID attribute correspondences. The Attribute Prior Fusion Reconstruction (APFR) branch facilitates
attribute-guided cross-modal interaction. We perform end-to-end training, and during inference, we use only the two encoders to encode features and compute
global similarity.

captions, we apply a mask operation, masking tokens with a
15% probability for text augmentation. For multi-granularity
attribute priors, we retain their positions in the original
sentence and set the rest of the positions to zero without
masking, resulting in a sequence {P sot , P1 , . . . , Peot }. Finally,
these modified sequences is inputted into the text encoder
to obtain text representations and multi-granularity attribute
priors representations. Here, T eot represents the global text
representation and Peot represents the global multi-granularity
attribute priors representation.
B. ID-limited Feature Redistribution
In IIFR, we leverage on the inherent semantic polymorphism of images and texts, exploring Relation-Sensitive
Semantic Polymorphism Alignment (RSSPA) across modalities while concurrently employing ID-Based Semantic Polymorphism Alignment (IBSPA) in each modality. Ultimately,
our goal is to establish a diverse relational polygon for more
convincing alignment, thereby enhancing the credibility and
effectiveness of cross-modal understanding.
1) Relation-Sensitive Semantic Polymorphism Alignment:
Consistent with trainditional approaches [20], [36], we investigate the inter-modal alignment under the constraint of identity.
Unlike conventional past efforts that primarily focus on aligning one-to-one paired text-image instances (Fig. 3a), we
explore a more diverse network of relations. We adaptively
explore Relation-Sensitive Semantic Polymorphism Alignment, which takes into account the varying strengths of
associations ranging from strong to weak between modalities
under ID constraints. Specifically, we establish strong and

Fig. 3. Exploration of Alignment Manners in TBPR. In the feature representation, different colors denote distinct IDs. For the aligned connection lines,
solid lines represent strong links, while dashed lines indicate weak links, with
different colors of dashed lines representing varying strengths of weak links.

weak correspondence in IDs based on text-image similarity
(Fig. 3b). Directly paired text-image instances maintain a
strong relationship, while unpaired instances in the same ID
establish weaker relationships based on cross-modal similarity
to explore Relation-Sensitive Semantic Polymorphism Alignment.
In this module, we primarily use contrastive loss to bring
various cross-modal relations closer, complemented by KL

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

LI et al.: SPCL: SEMANTIC POLYMORPHISM AND COMMONALITY LEARNING FOR TBPR

1827

divergence to balance the distributions. Specifically, we compute the cosine similarity between modalities based on the
textual features extracted from the text encoder and image
features derived from the image encoder:
>

i
i
S (Icls
, T eot
)=

i
I i cls · T eot
,
i
i
||Icls
||2 × ||T eot
||2

(1)

where i is the index and i ∈ {1, . . . , N}. Next, we calculate the
strength of relationships in an ID based on similarity scores.
The intra-ID similarity score between text and image features
is computed as follows:
v
u
i
u S (I i , T eot
)
i2t
,
(2)
S score (i) = t Pm cls j
j
j=1 S (Icls , T eot )
where S i2t
score (i) denotes the sensitivity score, which quantifies
the degree of responsiveness to variations in similarity. m
represents the count of individuals in each ID group in a given
batch. To ensure that paired text-image instances have a strong
correspondence, we enhance their alignment by adding the
identity matrix to the sensitivity coefficient matrix. Then, we
constrain the values of the resultant matrix to be no greater
than one, keeping the sensitivity coefficients in the range
[0, 1]. This can be expressed as follows:
i2t
S i2t
sen = min(S score + E, 1),

(3)

where E is the identity matrix and S i2t
sen is the sensitivity
coefficient.
Next, we extend the one-to-one cross-modal alignment to a
relation-sensitive semantic polymorphism alignment between
each image and all texts under the same ID. The description
is as follows:
g
x
X
exp(S (Icls
)/τ)
, T eot
−S i2t
sen (i)
L (i) =
log PN
.
a
xi
G(xi )
i
a=1 exp(S (Icls , T eot )/τ)
i

i2t

(4)

g∈G(x )

where τ is the temperature hyperparameter. N is the total
number of texts in a given batch. G(xi ) = {g ∈ 1 . . . C : xg = xi }
represents the collection of all text-related features that match
the visual features in the same ID group for the present batch.
Similarly, we compute the cross-modal alignment loss for
each text item in relation to all images sharing the same ID
in the current batch:
Lt2i (i) =

xi
h
X
exp(S (T cls
, Ieot
)/τ)
−S t2i
sen (i)
log
,
PN
i
i
x
a
H(x )
i
a=1 exp(S (T cls , Ieot )/τ)

N

N

i=1 j=1

(6)

(7)
(8)

where δ represents a small positive value to ensure the
denominator is not zero. The loss function after incorporating
the regularization term can be expressed as:
i2t
i2t
Li2t
RS ensi = L + αLre ,

(9)

t2i
t2i
Lt2i
RS ensi = L + αLre ,

(10)

where Li2t and Lt2i denote the mean of Li2t and Lt2i , respectively.
The final expression of the loss calculation formula for
RSSPA can be articulated as follows:
LRS ensi =

1 i2t
(L
+ Lt2i
RS ensi ).
2 RS ensi

(11)

2) ID-Based Semantic Polymorphism Alignment: Similarly,
we achieve ID-Based Semantic Polymorphism Alignment
(IBSPA) in each modality. IBSPA narrows the feature representation gap among individuals sharing the same ID, with
the objective of exploring the maximal similarity amongst
same-ID entities in each modality. Through the synergistic
interplay of inter-modal and intra-modal alignment operations,
we achieve diverse correspondence based on relations (Fig. 4).
This relation establishment facilitates a better understanding
of various semantic dimensions of the same person, thereby
enhancing the model’s analysis and comprehension of semantic polymorphism.
In this module, we compute the intra-modal fundamental
loss, with the objective of aligning the text/image feature
representations of the same ID in each modality for a given
batch:
i

Li2i (i) =

where H(xi ) = {h ∈ 1 . . . C : xh = xi } represents the collection
of all image-related features that match the textual features in
the same ID group for the present batch.
To balance the distance distribution between text and image
according to the ID distribution, we use Kullback-Leibler
(KL) divergence as a regularization term to adjust the distance
distribution. The regularization term is calculated as follows:
1 X X i2t
Ai2t (i)
A (i) log
,
N
B(i) + δ

j
i
exp(S (Icls
, T eot
))
Ai2t (i) = PN
, j ∈ {1, . . . , N},
i
a
a=1 exp(S (Vcls , T eot ))
l(i)
B(i) = PN
,
j=1 l( j)

(5)

h∈H(x )

Li2t
re (i) =

Fig. 4. Based on inter-modal relation-sensitive correspondences and intramodal feature aggregation, we achieve comprehensive relation-sensitive
semantic polymorphism alignment.

y
r
exp(S (Icls
, Icls
)/τ)
−1 X
log
,
PN
i
i
y
a
R(y )
a=1 exp(S (Icls , Icls )/τ)
r∈R(yi )

(12)

where R(yi ) = {r ∈ 1 . . . C : yr = yi } denotes the image or text
features that share the same ID in each modality.
Next, we employ Kullback-Leibler (KL) divergence as a
regularization term within each modality to balance the feature
distribution. The loss within each modality is collaboratively
computed as:
i2i
i2i
Li2i
IDBased = L + βLre ,

(13)

t2t
t2t
Lt2t
IDBased = L + βLre ,

(14)

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

1828

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

toolkit [50] to extract k sets of noun phrases or k sets of nouns
and adjectives from each caption. Subsequently, we encode
the extracted multi-granularity attribute priors in the same
manner as the original captions. The encoded multi-granularity
attribute priors features can be denoted as {P sot , P1 , . . . , Peot }k1 .
Here, Peot represents the global feature representation at
the multi-granularity attribute level. To achieve attribute-level
alignment, we first calculate the alignment loss between an
image and all texts in a batch, denoted as:
j
i
exp(S (Icls
, Peot
)/τ)
,
Li2p
(i)
=
−
log
PN
IDFree
i
a
a=1 exp(S (Icls , Peot )/τ)
j ∈ {1, . . . , N}.

Fig. 5. ID-Free Attribute Alignment matrix details. In the matrix, darker
colors represent higher matching similarity.

where Li2i and Lt2t denote the mean of Li2i and Lt2t , respect2t
tively. Li2i
re and Lre are the KL regularization term for each
modality.
The final expression of the loss calculation formula for
IBSPA can be articulated as follows:
t2t
LIDBased = Li2i
IDBased + LIDBased .

(15)

C. ID-Free Attribute Alignment
Through the collaborative efforts of RSSPA and IBSPA,
our model has been trained to establish diverse relations
among individuals based on their IDs. In the ID-Free Attribute
Alignment (IFAA) module, we transcend the limitations of
IDs to adaptively explore attribute alignment across different
IDs. We have observed that descriptions involving multigranularity attribute priors, such as blue backpack, appear in
different forms across images with various IDs. Specifically,
as in shown Fig. 5, these priors describing multi-granularity
attributs can be observed in images in three distinct ways: a)
Full match. The object in the image completely aligns with
the description, such as seeing a backpack that is confirmed
to be blue. b) Partial match. Only part of the description
is matched. For example, the image may show a backpack
that is not blue, or it may show the color blue without
mentioning a backpack. c) No match. Even though the image
and description belong to the same ID, the pedestrian in
the image does not exhibit the described multi-granularity
attribut priors at all. From these observations, we can infer that
the expression of attribute features is not strictly constrained
by specific IDs but instead exhibits a cross-ID commonality.
To investigate attribute commonality in TBPR, we propose
the IFAA module. This module aims to facilitate attribute
alignment and understanding at a macro level. The macro level
is dedicated to exploring the correspondence between all crossmodal textual and image attributes from the perspective of the
current batch. It transcends ID boundaries. This enhancement
allows the model to generalize and discriminate based on
shared attribute information across different identities.
Before performing the macro-level attribute alignment,
we leveraged the richness of our text dataset to extract
attribute representations at various granularities, forming
multi-granularity attribute priors. First, we utilize a language

(16)

Next, we align the textual multi-granularity attribute priors
with images using the relation-sensitive strength. This involves
optimizing the contrastive loss based on the similarity. We
calculate the sensitivity coefficient between textual multigranularity rattribute prior features and image features as
S i2p
score (i) =
as:

S (I i ,Pieot )
Pm cls j
. Therefore, Eq. 16 can be optimized
j
j=1 S (Icls ,Peot )

j
i
exp(S (Icls
, Peot
)/τ)
i2p
Li2p
(i)
=
−S
(i)
log
.
P
score
IDFree
N
i
a
a=1 exp(S (Icls , Peot )/τ)

(17)

The final expression of the loss calculation formula for
IFAA can be articulated as follows:
1
p2i
+ LIDFree
),
(18)
LIDFree = (Li2p
2 IDFree
p2i
i2p
where Li2p
IDFree and LIDFree denote the mean of LIDFree and
p2i
LIDFree , respectively.

D. Attribute Prior Fusion Reconstruction
Through the previous modules, we have achieved macrolevel exploration. To further investigate the fine-grained
hierarchical attribute characteristics at a micro level, we propose the Attribute Prior Fusion Reconstruction (APFR). The
micro level involves a more fine-grained exploration within
individual instances, focusing on the detailed cross-modal
interactions between paired text and image features. In APFR,
we study the positive impact of multi-granularity attribute
priors to guide intra-individual cross-modal interactions. This
enables the model to concentrate on enhancing its attention
towards multi-granularity attribute priors embedded in both
visual and textual modalities, thereby filtering visual noise
and improving its discriminative capabilities. We first fuse
the visual and textual modality information in APFR. Next,
we extract multi-granularity attribute priors from the fused
features. Through the synergistic integration of the IFAA and
APFR modules, we fortify our model’s ability to discern
attributes across visual and textual domains.
As shown in Fig. 2, the process comprises three main
steps: (1) cross-attention, where text features are used as
Queries and image features as Keys and Values; (2) a stack of
4-layer Transformer blocks, each incorporating Self-Attention
and Feed-Forward Networks (Multi-Layer Perception, MLP);
and (3) an inference component constructed with another MLP

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

LI et al.: SPCL: SEMANTIC POLYMORPHISM AND COMMONALITY LEARNING FOR TBPR

for higher-level reasoning. For cross-modal fusion, we treat
the text as the query, denoted as qi , and the image as the key
and value, denoted as ki and vi respectively. All these features
are subjected to normalization beforehand. Subsequently, they
are fed into the Multi-head Cross Attention (MCA) layer. The
formulation is as follows:
f i = MCA(qi , ki , vi ),

(19)

where d denotes the feature dimension, i represents the index,
i ∈ {1 . . . N}. Next, we input f i into 4-layer Transformer
blocks for deep cross-modal information fusion. The fused
cross-modal features are then processed through a MultiLayer Perception (MLP) for inference. This process can be
represented as follows:
pi = MLP(T rans f ormer( f i )).

(20)

Finally, we align the inference results with the original
multi-granularity attribute priors, constraining the inference
results and enhancing the capability of the encoders to capture
multi-granularity attributes. For the processing of multigranularity attribute priors, we retain their original positions
in the sentence and encode them into token sequences using
BPE [54]. For each sentence, we extract k sets of semantic
priors, with the kth semantic prior of the ith sentence denoted
as zik . The loss is represented as follows:
X X X ij
1
zk
LPRecon = −
|C||D||K|
i∈|C| j∈|D| j∈|K|

exp(pi j )

× log P|V|

c=1 exp(p

ic )

,

(21)

where |C| is the set of tokens for the textual multi-granularity
attribute priors and |D| is the vocabulary size. |K| represents
the total number of semantic priors extracted for each sentence. We aim to narrow the gap between the fused feature
representations and the textual multi-granularity attribute prior
representations using Eq. 21. This emphasizes the positive
impact of multi-granularity attribute priors on cross-modal
interactions.
1) Synthesis: Through the optimization facilitated by four
distinct modules, our ultimate framework is characterized by:
L = λ1 LRS ensi + λ2 LIDBased + λ3 LIDFree + λ4 LPRecon ,

(22)

where λ1 , λ2 , λ3 and λ4 refer to the weights of the RSSPA,
IBSPA, IFAA and APFR components, respectively.
IV. E XPERIMENTS
A. Experimental Settings
1) Datasets: We evaluate our approach on three prominent
datasets in the TBPR domain: CUHK-PEDES [20], ICFGPEDES [36] and RSTPReid [35]. CUHK-PEDES dataset,
derived from five reidentification datasets, encompasses 13,003
unique identities, 40,206 images, each annotated with two
captions, amassing to 80,412 natural language descriptions.
The training set includes 11,003 identities, 34,054 images,
and 68,108 captions. Validation and testing sets hold 3,078
and 3,074 images respectively, paired with 6,158 and 6,156

1829

descriptions, all featuring 1,000 distinct identities. ICFGPEDES dataset, sourced primarily from MSMT17 [60],
presents a larger image pool and more elaborate captions,
featuring 54,522 images linked to 4,102 identifiers, each with
a singular description. It comprises 34,674 image-caption
pairs for training and 19,848 for testing, with the training
set encompassing 3,102 unique identities and the test set
featuring 1,000 distinct identities. RSTPReid dataset covering
4,101 identities with five images per identity,totaling 20,505
images. The images in RSTPReid exhibit greater variation in
shape and deformation. Each image is paired with two text
descriptions. The training set includes 3,701 identities, while
both the validation and test sets each contain 200 identities.
2) Evaluation Metrics: On both benchmarks, we evaluate
our method using three metrics: Rank-k (k=1, 5, 10), mean
Average Precision (mAP), and mean Inverse Negative Penalty
(mINP) [62]. Rank-k is a commonly used metric for evaluating
the performance of information retrieval systems, focusing on
the accuracy of the top-k positions in the retrieval results.
Mean Average Precision (mAP) considers both the precision
and recall of the retrieval results. It is obtained by calculating
the average precision for each query, which is the proportion
of relevant items retrieved among all retrieved items. Unlike
mAP, which mainly focuses on the ranking positions of correct
predictions, mINP emphasizes the localization of the most
challenging matches. It reflects the model’s ability to handle
the most difficult cases in retrieval by considering the position
of the last correct match.
3) Implementation Details: Our backbone primarily consists of two encoders: a visual encoder (CLIP-ViT-B/16 or
CLIP-ViT-L/14) and a text encoder (CLIP-Transformer). Additionally, we initialize the cross-modal interaction network
randomly, setting the dimensions to 512 and the number
of heads to 8. Our experiments are performed on a single
NVIDIA A800 GPU, running for 60 epochs on CUHK-PEDES
and ICFG-PEDES, and 20 epochs on RSTPReid, with a minibatch size of 64. During training, we resize every image input
to 384 × 128 and apply random horizontal flipping, random
erasing, and random cropping with consistent padding for
image augmentation. For text augmentation, we mask the text
input tokens with a probability of 15 %. We use the Adam [63]
optimizer. We adopt a five-epoch warm-up phase to increase
the learning rate linearly from 1×10−6 to 1×10−5 , followed by
a cosine learning rate schedule for decay. We utilize random
and identity sampling with a ratio of 2:1. The temperature
parameter τ is set to 0.02. The regularization coefficient α is
set to 0.1, and β is set to 1. The weights of the components are
set as follows: λ1 for RSSPA to 30, λ2 for IBSPA to 30, λ3 for
IFAA to 1, and λ4 for APFR to 1. Note that the above settings
are primarily based on the ViT-B/16 configuration. When the
image encoder is switched to ViT-L/14, the setting of λ3 for
IFAA is adjusted to 1.8 on CUHK-PEDES and ICFG-PEDES,
and λ4 for APFR is adjusted to 1.2 on RSTPReid.
B. Comparison with State-of-the-Art Methods
We compared our method with state-of-the-art methods on
CUHK-PEDES, ICFG-PEDES, and RSTPReid benchmarks
and evaluated the impact of different scales of CLIP-ViT on

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.


codex
第六篇的机制是“两条线”：同一身份内承认多种表达，不同身份间承认共享属性。证据要看它是否分别消融 RSSPA、IBSPA、IFAA、APFR，以及有没有相似度矩阵或注意可视化支撑这些概念。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext 'SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf' - | rg -n -C 4 'Ablation|ablation|Effectiveness|RSSPA|IBSPA|IFAA|APFR|semantic polymorphism|commonality|Visualization|visual|Table IV|Table V|Table VI|Fig. 6|Fig. 7|Fig. 8|matrix|attribute'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
8-
9-Abstract—Text-Based Person Retrieval (TBPR) refers to identifying a specific target pedestrian image based on natural language
10-descriptions. Most previous methods rely on one-to-one alignment
11-between paired text-image data, ignoring the polymorphic nature
12:of visual and linguistic information. Moreover, constrained by
13-ID, earlier methods have shown limited exploration of intraindividual and inter-individual relations. This limitation confines
14-them to exploring characteristics within individuals, making it
15-challenging to uncover commonalities and invariants that extend
16:across IDs (e.g., attributes). Recently, due to the lack of accurate
17:annotations, exploring attribute-based cross-modal interactions
18-and alignments has become a significant challenge in TBPR.
19-To address these issues, we propose a Semantic Polymorphism
20-and Commonality Learning (SPCL) framework. First, we present
21:Relation-Sensitive Semantic Polymorphism Alignment (RSSPA)
22:and ID-Based Semantic Polymorphism Alignment (IBSPA) to
23-explore ID-limited Feature Redistribution. Second, we transcend
24-the constraints of ID, leveraging ID-Free Attribute Alignment
25:(IFAA) from a macro perspective to explore commonalities and
26:invariants based on attribute features. Finally, from a micro
27-perspective, we design Attribute Prior Fusion Reconstruction
28:(APFR) to optimize the attention of our model, exploring the
29:positive impact of attribute priors on cross-modal interaction.
30-Experiments on CUHK-PEDES, ICFG-PEDES and RSTPReid
31-show that our method achieves state-of-the-art performance on
32-Rank-1, mAP and mINP.
33:Index Terms—Text-based person retrieval, semantic polymorphism, semantic commonality, ID-free attribute alignment.
34-
35-I. I NTRODUCTION
36-EXT-BASED person retrieval (TBPR) is defined as the
37-process of employing natural language to retrieve a
38-specific target person in an extensive gallery of candidate
39-images [1], [2], [3], [4]. Compared with traditional unimodal
40-person retrieval methods [5], [6], TBPR offers the advantage
41-of using text queries to facilitate a more flexible and easily
42:accessible retrieval process by describing the attributes of
43-the target person. Unlike general multi-category text-to-image
44-
45-T
46-
--
61-
62-cross-modal retrieval tasks [7], [8], TBPR focuses exclusively
63-on the category of pedestrian, which necessitates processing
64-finer-grained details and demands stricter expressions of crossmodal consistency.
65:In TBPR, visual appearance is subject to varying degrees
66-of discrepancy due to changes in lighting conditions and
67-viewing angles. Similarly, textual expressions exhibit differences due to variations in sentence structure, lexical choices,
68-and descriptions of object categories. This variation in the
69-representation of the same concept reflects a core element
70:of TBPR, namely, semantic polymorphism. Furthermore, the
71-existence of similar semantic meanings across individuals
72-under different IDs reflects another core element of TBPR,
73:namely, semantic commonality. TBPR primarily addressing
74:the inherent challenge posed by modality heterogeneity. However, the presence of semantic polymorphism and commonality
75-further increases the difficulty of cross-modal alignment.
76-As in Fig. 1a, the two pedestrians belong to the same
77-identity, with one facing front and the other facing back.
78:Despite similar visual and textual expressions, subtle differences exist between these two images and text segments. For
79-instance, the left image shows a white shirt with patterns
80-without visible shoes, while the right image, taken from
81-behind, displays the shoes but not the shirt. These differences
82:are also evident in the corresponding texts, which are similar yet complementary, illustrating semantic polymorphism.
83-Cross-modal semantics with one-to-one complete matching
84-can be regarded as symmetric relations. We coin the term
85-strong correspondence for the symmetric relations that can
86-be established between matched text-image pairs of the same
87:entity. Correspondingly, we term the phenomenon of similar yet complementary cross-modal visual/textual semantic
88-descriptions as weak correspondence. To achieve cross-modal
89-alignment, earlier approaches [9], [10], [11] primarily focused
90-on one-to-one alignment, ignoring the tight relations among
91-images or texts, for the identical pedestrian, as well as the
92-interaction of text and images. Based on this, we explore the
93:semantic polymorphism and relation sensitivity of cross-modal
94-combinations that are similar yet complementary. Furthermore,
95:to deeply delve into semantic polymorphism and achieve
96:a more regular feature distribution, we explore the semantic polymorphism in each modality to assist cross-modal
97-matching.
98-In recent years, an increasing amount of research has
99:been devoted to exploring the critical role of attributes in
100-TBPR tasks [4], [12], [13], [14], [15]. For unimodal person
101-
102-1051-8215 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and
103-similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
--
113-and texts collected from the same person exhibit subtle variance. For example,
114-the left image shows strong alignment with the top text descriptions, such as
115-clothing type, clothing color, and hair style, while shows weak correspondence
116-with the bottom one. (b) Inter-ID Attribute Correspondences. Partial matches
117:between textual and image attributes can also occur across different IDs.
118-(c) Different Cross-Modal Attribute Alignment Methods.
119-
120-re-identification tasks, inevitable appearance disparities
121-induced by variations in illumination, viewpoint, and other
122-factors cause confusion in feature matching. Attributes can
123-serve as higher-level discriminative traits, facilitating the
124:filtering of visual appearance discrepancies in person retrieval
125:tasks. For TBPR, the rich multi-granularity attributes in
126-textual datasets provide valuable prior information, effectively
127:helping to mitigate the impact of visual noise. Moreover, as
128-shown in Fig. 1b, commonalities can be found across different
129:IDs for multi-granularity attribute priors, a phenomenon we
130:term attribute commonality. Most previous attribute-based
131-methods [4], [16], [17] were limited by ID, focusing on the
132-relation between paired text-image sets within the same ID.
133:This design will inevitably force the separation of visual
134:representations and their related textual attributes, especially
135:when they share the same attributes but belong to different
136:IDs. This can blur the model’s understanding of attributes,
137-which we consider unfriendly to the model. Based on this,
138:we investigate the commonalities between textual attributes
139-and images in a ID-free manner. We disregard ID limitations
140:and explore the correlations between textual attributes and
141-different images from a macro perspective, even across
142-
143-different IDs. The macro perspective refers to an examination
144-of the overall correspondence between global-level textual
145:attributes and images within the current batch.
146-Additionally, most previous approaches primarily utilized
147:loss functions to explicitly perform cross-modal attribute alignment [14], [18], [19]. As shown in Fig. 1c, these methods
148-typically either select portions of text (Fig. 1c (i)) or segment
149:the image (Fig. 1c (ii)) to extract attribute information, aiming
150:to bridge visual and textual representations. However, such
151-approaches often lack a deep exploration of multi-granularity
152:attribute priors in cross-modal interactions. Based on this,
153:we further establish the relationship between overall visual
154:elements and discrete textual attributes at a micro level. We
155-explore the positive impact of text-based multi-granularity
156:attribute priors on guiding a more refined intra-individual
157-cross-modal fusion reconstruction (Fig. 1c (iii)). This exploration of micro level is dedicated to identifying fine-grained
158-characteristics that reflect the intricate relationships within
159-paired text-image instances.
160-Building upon prior research, in this paper, we propose
161-Semantic Polymorphism and Commonality Learning (SPCL)
162:based on the investigation of semantic polymorphism and
163:attributes. Firstly, we propose Relation-Sensitive Semantic
164:Polymorphism Alignment (RSSPA), which delves into the
165:semantic polymorphism across modalities. Specifically, under
166-the constraint of IDs, we align features between different modalities with the same ID adaptively according to
167-their similarity. Concurrently, we propose ID-Based Semantic
168:Polymorphism Alignment (IBSPA) to investigate intra-modal
169:semantic polymorphism. This dual strategy, integrating RSSPA
170:and IBSPA, aims to capture ID-limited semantic distribution
171:and construct diverse relational polygons in and across modalities. Second, we design ID-Free Attribute Alignment (IFAA)
172:to explore the commonalities between textual attributes and
173:images from a macro perspective. IFAA aligns with human
174:natural cognitive patterns by focusing solely on the correspondence between textual attributes and images, unconstrained
175:by ID limitations. It adaptively establishes alignment relationships between various attributes and different images. Finally,
176:we propose Attribute Prior Fusion Reconstruction (APFR)
177:to filter noise and enhance salient information for crossmodal matching from a micro perspective. APFR extracts
178:multi-granularity attribute priors from sentences to guide intraindividual cross-modal fusion and attribute reconstruction. By
179:implicitly achieving cross-modal alignment, APFR enhances
180:the integrated understanding of attributes. IFAA and APFR
181-respectively investigate the commonalities and saliencies of
182:attributes in TBPR, enabling our model to focus more on
183:the discriminative attribute components during cross-modal
184-alignment.
185-The main contributions are summarized as follows:
186:• We propose RSSPA and IBSPA based on ID constraints,
187-constructing diverse relational polygons in and across
188:modalities to address the challenges of semantic polymorphism.
189:• We present IFAA, disregarding ID constraints, to establish
190:a more comprehensive attribute-image relation system at
191-a macro level, exploring their commonalities.
192-
193-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.
194-
195-LI et al.: SPCL: SEMANTIC POLYMORPHISM AND COMMONALITY LEARNING FOR TBPR
196-
197:• We design APFR to further guide intra-individual crossmodal fusion and attribute reconstruction at a micro level,
198-purifying the critical information in each modality.
199-II. R ELATED W ORK
200-A. Text-based Person Retrieval
201-Text-Based Person Retrieval (TBPR) is a novel task that
--
219-Vision-Language Pretraining (VLP) models [41], [42], an
220-increasing number of studies have adopted VLP with finetuning to enhance the underlying alignment [43], [44], [45],
221-[46]. However, these studies primarily focused on performing
222-one-to-one text-image alignment under ID constraints to study
223:cross-modal matching tasks, neglecting the semantic polymorphism present in real-world.
224-Unlike previous works, we investigate intra-ID semantic
225-polymorphism by exploring the differential strength of associations among individuals, emphasizing relation sensitivity.
226:Furthermore, we leverage the attribute commonality to explore
227:semantic associations between different IDs, reinforcing discriminative semantic information in both visual and textual
228-modalities.
229-B. Attribute-based Representation Learning
230-Attribute-based Representation Learning (ABPL) focuses
231:on leveraging attribute information to learn more meaningful
232:and discriminative feature representations. In person reidentification, early works either manually annotated attributes
233:or utilized pre-trained classifiers to extract attribute features
234-for cross-modal matching [47], [48], [49]. With the evolution of TBPR tasks, textual datasets provide increasingly
235-rich and complete descriptions. This advancement allows
236:for the extraction of attributes from both visual and textual descriptions, facilitating cross-modal matching. Reference
237-[14] attemptes to automatically achieve local-level matching
238-between text and images by horizontally segmenting images
239:and employing an attention mechanism. Moreover, some studies leverage external toolkits [50], [51] to extract attribute
240-words or phrases [18], [19], [52], which are then matched with
241:visual information in a cross-modal manner. References [16]
242-
243-1825
244-
245:and [17] employ prompt learning to flexibly extract understandable attribute information for cross-modal alignment.
246:Reference [4] establishes an attribute vocabulary and conducts
247:research based on the frequency of attribute occurrences in the
248:dataset, further addressing the long tail effect of attributes.
249:However, previous attribute-based works were limited to
250-paired text-image instances, leading to conflicting when the
251:same attribute appears in different IDs. Additionally, these
252:methods focus on directly aligning attribute information using
253:loss functions at the end, lacking in-depth exploration of multigranularity attribute priors in cross-modal interactions.
254-In this paper, we establish connections between images of
255:different IDs and various textual attributes, aiming to explore
256:the macro-level commonalities between ID-free attributeimage pairs. Furthermore, we further explore the deep
257:perception of attributes in cross-modal interactions in individuals, guiding our model at a micro level to focus on more
258:discriminative multi-granularity attribute priors.
259-III. M ETHOD
260-In this section, we present the proposed Semantic Polymorphism and Commonality Learning (SPCL) framework.
261-An overview of SPCL is illustrated in Fig 2. It extracts
262-features using a dual-stream backbone network. Our input
263-includes original caption-image pairs and multi-granularity
264:attribute priors extracted from captions. SPCL is optimized
265-through three branches: ID-limited Feature Redistribution
266:(IIFR), ID-Free Attribute Alignment (IFAA), and Attribute
267:Prior Fusion Reconstruction (APFR). Notably, IIFR comprises
268-two components: Relation-Sensitive Semantic Polymorphism
269:Alignment (RSSPA) and ID-Based Semantic Polymorphism
270:Alignment (IBSPA). Ultimately, our model evaluates the
271-similarity between target text and candidate images. The
272-subsequent modules will be discussed in detail in dedicated
273-subsections.
274-A. Visual and Textual Feature Extraction
--
281-then linearly projected into 1-dimensional tokens. Positional
282-encoding is applied to these tokens, accompanied by the
283-addition of a [CLS] token to form the sequence {Icls , I1 , . . . , In }.
284-Finally, this sequence is fed into the image encoder to generate
285:visual representations, with Icls being the global representation.
286-2) Text Feature Extraction: We adopt the CLIPTransformer as the text encoder, comprised of L Transformer
287-[53] blocks. First, we utilize BPE [54] to convert input text
288-into a sequence of tokens. The tokens are then prefixed
289-with [SOT] and suffixed with [EOT] to denote the start
--
297-
298-IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
299-
300-Fig. 2. SPCL consists of a text encoder, an image encoder, and three main branches. The ID-limited Feature Redistribution branch achieves ID-limited alignment
301:through two components: Relation-Sensitive Semantic Polymorphism Alignment (RSSPA) and ID-Based Semantic Polymorphism Alignment (IBSPA). The
302:ID-Free Attribute Alignment (IFAA) branch explores inter-ID attribute correspondences. The Attribute Prior Fusion Reconstruction (APFR) branch facilitates
303:attribute-guided cross-modal interaction. We perform end-to-end training, and during inference, we use only the two encoders to encode features and compute
304-global similarity.
305-
306-captions, we apply a mask operation, masking tokens with a
307-15% probability for text augmentation. For multi-granularity
308:attribute priors, we retain their positions in the original
309-sentence and set the rest of the positions to zero without
310-masking, resulting in a sequence {P sot , P1 , . . . , Peot }. Finally,
311-these modified sequences is inputted into the text encoder
312:to obtain text representations and multi-granularity attribute
313-priors representations. Here, T eot represents the global text
314-representation and Peot represents the global multi-granularity
315:attribute priors representation.
316-B. ID-limited Feature Redistribution
317:In IIFR, we leverage on the inherent semantic polymorphism of images and texts, exploring Relation-Sensitive
318:Semantic Polymorphism Alignment (RSSPA) across modalities while concurrently employing ID-Based Semantic Polymorphism Alignment (IBSPA) in each modality. Ultimately,
319-our goal is to establish a diverse relational polygon for more
320-convincing alignment, thereby enhancing the credibility and
321-effectiveness of cross-modal understanding.
322-1) Relation-Sensitive Semantic Polymorphism Alignment:
--
387-the degree of responsiveness to variations in similarity. m
388-represents the count of individuals in each ID group in a given
389-batch. To ensure that paired text-image instances have a strong
390-correspondence, we enhance their alignment by adding the
391:identity matrix to the sensitivity coefficient matrix. Then, we
392:constrain the values of the resultant matrix to be no greater
393-than one, keeping the sensitivity coefficients in the range
394-[0, 1]. This can be expressed as follows:
395-i2t
396-S i2t
397-sen = min(S score + E, 1),
398-
399-(3)
400-
401:where E is the identity matrix and S i2t
402-sen is the sensitivity
403-coefficient.
404-Next, we extend the one-to-one cross-modal alignment to a
405:relation-sensitive semantic polymorphism alignment between
406-each image and all texts under the same ID. The description
407-is as follows:
408-g
409-x
--
431-
432-where τ is the temperature hyperparameter. N is the total
433-number of texts in a given batch. G(xi ) = {g ∈ 1 . . . C : xg = xi }
434-represents the collection of all text-related features that match
435:the visual features in the same ID group for the present batch.
436-Similarly, we compute the cross-modal alignment loss for
437-each text item in relation to all images sharing the same ID
438-in the current batch:
439-Lt2i (i) =
--
486-(10)
487-
488-where Li2t and Lt2i denote the mean of Li2t and Lt2i , respectively.
489-The final expression of the loss calculation formula for
490:RSSPA can be articulated as follows:
491-LRS ensi =
492-
493-1 i2t
494-(L
--
499-(11)
500-
501-2) ID-Based Semantic Polymorphism Alignment: Similarly,
502-we achieve ID-Based Semantic Polymorphism Alignment
503:(IBSPA) in each modality. IBSPA narrows the feature representation gap among individuals sharing the same ID, with
504-the objective of exploring the maximal similarity amongst
505-same-ID entities in each modality. Through the synergistic
506-interplay of inter-modal and intra-modal alignment operations,
507-we achieve diverse correspondence based on relations (Fig. 4).
508-This relation establishment facilitates a better understanding
509-of various semantic dimensions of the same person, thereby
510:enhancing the model’s analysis and comprehension of semantic polymorphism.
511-In this module, we compute the intra-modal fundamental
512-loss, with the objective of aligning the text/image feature
513-representations of the same ID in each modality for a given
514-batch:
--
552-Li2t
553-re (i) =
554-
555-Fig. 4. Based on inter-modal relation-sensitive correspondences and intramodal feature aggregation, we achieve comprehensive relation-sensitive
556:semantic polymorphism alignment.
557-
558-y
559-r
560-exp(S (Icls
--
601-IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
602-
603-toolkit [50] to extract k sets of noun phrases or k sets of nouns
604-and adjectives from each caption. Subsequently, we encode
605:the extracted multi-granularity attribute priors in the same
606-manner as the original captions. The encoded multi-granularity
607:attribute priors features can be denoted as {P sot , P1 , . . . , Peot }k1 .
608-Here, Peot represents the global feature representation at
609:the multi-granularity attribute level. To achieve attribute-level
610-alignment, we first calculate the alignment loss between an
611-image and all texts in a batch, denoted as:
612-j
613-i
--
626-a
627-a=1 exp(S (Icls , Peot )/τ)
628-j ∈ {1, . . . , N}.
629-
630:Fig. 5. ID-Free Attribute Alignment matrix details. In the matrix, darker
631-colors represent higher matching similarity.
632-
633-where Li2i and Lt2t denote the mean of Li2i and Lt2t , respect2t
634-tively. Li2i
635-re and Lre are the KL regularization term for each
636-modality.
637-The final expression of the loss calculation formula for
638:IBSPA can be articulated as follows:
639-t2t
640-LIDBased = Li2i
641-IDBased + LIDBased .
642-
643-(15)
644-
645-C. ID-Free Attribute Alignment
646:Through the collaborative efforts of RSSPA and IBSPA,
647-our model has been trained to establish diverse relations
648-among individuals based on their IDs. In the ID-Free Attribute
649:Alignment (IFAA) module, we transcend the limitations of
650:IDs to adaptively explore attribute alignment across different
651:IDs. We have observed that descriptions involving multigranularity attribute priors, such as blue backpack, appear in
652-different forms across images with various IDs. Specifically,
653-as in shown Fig. 5, these priors describing multi-granularity
654-attributs can be observed in images in three distinct ways: a)
655-Full match. The object in the image completely aligns with
--
660-mentioning a backpack. c) No match. Even though the image
661-and description belong to the same ID, the pedestrian in
662-the image does not exhibit the described multi-granularity
663-attribut priors at all. From these observations, we can infer that
664:the expression of attribute features is not strictly constrained
665:by specific IDs but instead exhibits a cross-ID commonality.
666:To investigate attribute commonality in TBPR, we propose
667:the IFAA module. This module aims to facilitate attribute
668-alignment and understanding at a macro level. The macro level
669:is dedicated to exploring the correspondence between all crossmodal textual and image attributes from the perspective of the
670-current batch. It transcends ID boundaries. This enhancement
671-allows the model to generalize and discriminate based on
672:shared attribute information across different identities.
673:Before performing the macro-level attribute alignment,
674-we leveraged the richness of our text dataset to extract
675:attribute representations at various granularities, forming
676:multi-granularity attribute priors. First, we utilize a language
677-
678-(16)
679-
680:Next, we align the textual multi-granularity attribute priors
681-with images using the relation-sensitive strength. This involves
682-optimizing the contrastive loss based on the similarity. We
683:calculate the sensitivity coefficient between textual multigranularity rattribute prior features and image features as
684-S i2p
685-score (i) =
686-as:
687-
--
714-
715-(17)
716-
717-The final expression of the loss calculation formula for
718:IFAA can be articulated as follows:
719-1
720-p2i
721-+ LIDFree
722-),
--
731-LIDFree , respectively.
732-
733-D. Attribute Prior Fusion Reconstruction
734-Through the previous modules, we have achieved macrolevel exploration. To further investigate the fine-grained
735:hierarchical attribute characteristics at a micro level, we propose the Attribute Prior Fusion Reconstruction (APFR). The
736-micro level involves a more fine-grained exploration within
737-individual instances, focusing on the detailed cross-modal
738:interactions between paired text and image features. In APFR,
739:we study the positive impact of multi-granularity attribute
740-priors to guide intra-individual cross-modal interactions. This
741-enables the model to concentrate on enhancing its attention
742:towards multi-granularity attribute priors embedded in both
743:visual and textual modalities, thereby filtering visual noise
744-and improving its discriminative capabilities. We first fuse
745:the visual and textual modality information in APFR. Next,
746:we extract multi-granularity attribute priors from the fused
747:features. Through the synergistic integration of the IFAA and
748:APFR modules, we fortify our model’s ability to discern
749:attributes across visual and textual domains.
750-As shown in Fig. 2, the process comprises three main
751-steps: (1) cross-attention, where text features are used as
752-Queries and image features as Keys and Values; (2) a stack of
753-4-layer Transformer blocks, each incorporating Self-Attention
--
777-
778-(20)
779-
780-Finally, we align the inference results with the original
781:multi-granularity attribute priors, constraining the inference
782-results and enhancing the capability of the encoders to capture
783:multi-granularity attributes. For the processing of multigranularity attribute priors, we retain their original positions
784-in the sentence and encode them into token sequences using
785-BPE [54]. For each sentence, we extract k sets of semantic
786-priors, with the kth semantic prior of the ith sentence denoted
787-as zik . The loss is represented as follows:
--
804-
805-(21)
806-
807-where |C| is the set of tokens for the textual multi-granularity
808:attribute priors and |D| is the vocabulary size. |K| represents
809-the total number of semantic priors extracted for each sentence. We aim to narrow the gap between the fused feature
810:representations and the textual multi-granularity attribute prior
811-representations using Eq. 21. This emphasizes the positive
812:impact of multi-granularity attribute priors on cross-modal
813-interactions.
814-1) Synthesis: Through the optimization facilitated by four
815-distinct modules, our ultimate framework is characterized by:
816-L = λ1 LRS ensi + λ2 LIDBased + λ3 LIDFree + λ4 LPRecon ,
817-
818-(22)
819-
820:where λ1 , λ2 , λ3 and λ4 refer to the weights of the RSSPA,
821:IBSPA, IFAA and APFR components, respectively.
822-IV. E XPERIMENTS
823-A. Experimental Settings
824-1) Datasets: We evaluate our approach on three prominent
825-datasets in the TBPR domain: CUHK-PEDES [20], ICFGPEDES [36] and RSTPReid [35]. CUHK-PEDES dataset,
--
858-predictions, mINP emphasizes the localization of the most
859-challenging matches. It reflects the model’s ability to handle
860-the most difficult cases in retrieval by considering the position
861-of the last correct match.
862:3) Implementation Details: Our backbone primarily consists of two encoders: a visual encoder (CLIP-ViT-B/16 or
863-CLIP-ViT-L/14) and a text encoder (CLIP-Transformer). Additionally, we initialize the cross-modal interaction network
864-randomly, setting the dimensions to 512 and the number
865-of heads to 8. Our experiments are performed on a single
866-NVIDIA A800 GPU, running for 60 epochs on CUHK-PEDES
--
874-a cosine learning rate schedule for decay. We utilize random
875-and identity sampling with a ratio of 2:1. The temperature
876-parameter τ is set to 0.02. The regularization coefficient α is
877-set to 0.1, and β is set to 1. The weights of the components are
878:set as follows: λ1 for RSSPA to 30, λ2 for IBSPA to 30, λ3 for
879:IFAA to 1, and λ4 for APFR to 1. Note that the above settings
880-are primarily based on the ViT-B/16 configuration. When the
881-image encoder is switched to ViT-L/14, the setting of λ3 for
882:IFAA is adjusted to 1.8 on CUHK-PEDES and ICFG-PEDES,
883:and λ4 for APFR is adjusted to 1.2 on RSTPReid.
884-B. Comparison with State-of-the-Art Methods
885-We compared our method with state-of-the-art methods on
886-CUHK-PEDES, ICFG-PEDES, and RSTPReid benchmarks
887-and evaluated the impact of different scales of CLIP-ViT on
--
919-large-scale trainable deep networks hold strong potential for
920-fine-grained mining in TBPR.
921-2) Comparison on ICFG-PEDES: As shown in Table II,
922-we validate the effectiveness of our method on ICFG-PEDES.
923:We compare various scales of ViT as the visual encoder.
924-Even with a smaller-scale ViT, our model achieves state-ofthe-art results in Rank-1 accuracy, mAP, and mINP. When
925-using a larger-scale ViT, our model continues to exhibit
926-
927-TABLE II
--
954-while Rank-5 and Rank-10 also show competitive outcomes.
955-The data in the RSTPReid dataset inherently contains significant noise, which limits the semantic mining capability of our
956-method. A detailed discussion of this issue will be provided
957-in subsection G.
958:C. Ablation Study
959:In this section, we analyze the effectiveness of RSSPA,
960:IBSPA, IFAA, and APFR. We use the CLIP model fine-tuned
961-with cross-entropy loss on each dataset as our baseline. In our
962:setup, RSSPA replaces the cross-entropy loss to validate the
963-positive impact of relation sensitivity in practical applications.
964:RSSPA is designed to enhance comprehensive alignment
965-by employing relation sensitivity. The effectiveness of the
966:proposed RSSPA is well demonstrated through experiments
967:No.0 vs No.2 and No.1 vs No.3, as shown in Table IV. On
968:the CUHK-PEDES dataset, compared to the baseline, RSSPA
969-increases Rank-1 accuracy by over 3%, mAP by approximately 7%, and mINP by about 11%, which are substantial
970-performance improvements for both scales of vision-language
971-models. Similarly, significant improvements are observed on
972-the ICFG-PEDES dataset. To further validate our loss function,
973-we compare its performance when used as a single mapping
974:with our relation-sensitive loss. As shown in Table V, the proposed relation-sensitive approach improves Rank-1 accuracy,
975-mAP, and mINP by 2.76%, 6.16%, and 10.28%, respectively,
976-compared to traditional single mapping alignment. These
977-results demonstrate the high value of our research based on
978-relation sensitivity.
979:IBSPA is designed as an intra-modal feature constraint
980:module to assist cross-modal RSSPA. As shown in Table IV,
981-experiments No.0 vs No.1 demonstrate the effectiveness of
982:IBSPA on the CUHK-PEDES dataset, with Rank-1 accuracy
983-improvements of 2.55% and 2.03%, mAP improvements of
984-3.48% and 3.37%, and mINP improvements of 5.11% and
985-5.40% for two different scales of vision-language models,
986-respectively. On the ICFG-PEDES dataset, these enhancements are even more pronounced, with Rank-1 accuracy
--
988-
989-1831
990-
991-by 3.22% and 4.41%. Experiments No.2 vs No.3 further
992:validates the positive impact of IBSPA when combined with
993:RSSPA, achieving ID-limited feature redistribution.
994:The RSSPA module is the fundamental alignment module of
995:our approach, combining with IBSPA to form a traditional IDlimited cross-modal semantic alignment. To further investigate
996:the overall effect of RSSPA + IBSPA, we replaced them
997-with the SDM (Similarity Distribution Matching) [45] loss
998-and CMPM (Cross-Modal Projection Matching) [23] loss
999-for achieving cross-modal explicit alignment. The results are
1000:shown in Table VI. Compared to the model performance
1001-using the SDM-based approach or CMPM-based approach,
1002:the combination of RSSPA + IBSPA achieved comprehensive
1003-advantages. This further demonstrates the positive impact of
1004-our relation-sensitive strong-weak alignment on effectively
1005-addressing the cross-modal matching ambiguity caused by
1006-semantic differences among different instances under the same
1007-ID. Specifically, it improves the accuracy and reliability of
1008-matching instances across modalities.
1009:IFAA is designed to explore attribute alignment beyond ID
1010:constraints. The effectiveness of IFAA is validated through
1011:experiments No.3 vs No.5 in Table IV. Using a small-scale
1012:vision-language model on the CUHK-PEDES dataset, IFAA
1013-improves Rank-1 accuracy, mAP, and mINP by 0.39%, 0.51%,
1014-and 0.83%, respectively, compared to the ID-limited constraint approach. When using a large-scale vision-language
1015-model, our Rank-1 accuracy on the CUHK-PEDES dataset
1016-reaches 76.93%. These results indicate that larger-scale visionlanguage models can indeed have a positive impact on
1017-accuracy.
1018:APFR is designed to deeply explore the impact of implicit
1019:multi-granularity attribute priors alignment on cross-modal
1020:interaction. Experiments No.3 vs No.4 in Table IV show that
1021:on the CUHK-PEDES dataset, APFR significantly improves
1022-mAP and mINP accuracy, with nearly 0.5% improvement in
1023-mAP and close to 1% in mINP for both scales of visionlanguage models. Similar comprehensive improvements are
1024:observed on the ICFG-PEDES dataset. No. 6 in Table IV
1025-validates the combined effectiveness of all components. Interestingly, while our method performs well with small-scale
1026-vision-language models, the cumulative effect of all components shows a slight decrease when switching to large-scale
1027-vision-language models. We analyze that this is due to the
1028-large-scale models learning more redundant information due
1029:to their complexity, causing a conflict between IFAA’s explicit
1030:attribute alignment and APFR’s implicit attribute alignment,
1031-increasing optimization difficulty.
1032-D. Evaluation of Attribute
1033:In this section, we discuss the relevant settings for attribute
1034-alignment during our experiments. These experiments are
1035-based on small-scale vision-language model.
1036-We evaluated the impact of extracting different sets of
1037:multi-granularity attribute priors on model performance. As
1038:shown in Table VII, the model performance initially improves
1039:as the number of multi-granularity attribute priors extracted
1040-from each caption increases. However, when the number
1041:of multi-granularity attribute priors exceeds four, the model
1042-
1043-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.
1044-
1045-1832
--
1055-T RAINDITIONAL S INGLE A LIGNMENT. RSA M EANS
1056-R ELATION -S ENSITIVE A LIGNMENT
1057-
1058-TABLE VI
1059:E VALUATION OF RSSPA+IBSPA FOR C ROSS -M ODAL A LIGNMENT.
1060:* D ENOTES THE A BBREVIATIONS FOR IFAA AND APFR
1061-
1062-TABLE VII
1063-E VALUATION OF D IFFERENT S ETS OF M ULTI -G RANULARITY
1064-ATTRIBUTE P RIORS . MGAP R EFERS TO THE M ULTI -G RANULARITY
1065-ATTRIBUTE P RIORS
1066-
1067:Fig. 6. A demonstration of the top-10 retrieval results for SPCL (the first
1068-row) and Baseline (the second row) on CUHK-PEDES. Correct retrievals are
1069-indicated with a red rectangle, while incorrect ones are marked with a green
1070-rectangle.
1071-
--
1078-I NTERACTION N ETWORK
1079-
1080-performance declines. We analyze that this is because the
1081-model needs to focus on the most important information in the
1082:sentence, and extracting too many multi-granularity attribute
1083-priors can distract the model’s attention, leading to a negative
1084-effect.
1085-We also validated the impact of using varying numbers
1086:of Transformer blocks in the cross-modal fusion of APFR.
1087:As shown in Table VIII, model performance fluctuates with
1088-
1089-different numbers of layers. Ultimately, we selected the optimal configuration of four Transformer blocks.
1090:Additionally, we compared the use of word level attributes
1091:with multi-granularity attribute priors. As shown in Table IX,
1092:multi-granularity attribute priors provide more accurate critical
1093-information to the model, resulting in better performance. We
1094-analyze that this is because training the model with word
1095:level attributes disperses its attention, disrupts the overall
1096:alignment of visual representations, and lacks the ability to
1097:combine multi-granularity attributes. This causes the model to
1098-lack discriminative judgment on specific parts when analyzing
1099:sentence structures or visual representations. When using text
1100-to search for a target, if a non-target image shares word level
1101:attributes with the target, it can lead to confusion. Experiments
1102-
1103-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.
1104-
1105-LI et al.: SPCL: SEMANTIC POLYMORPHISM AND COMMONALITY LEARNING FOR TBPR
1106-
1107-1833
1108-
1109:Fig. 7. Visualization of comprehensive semantic attention on the baseline model and SPCL. The brighter areas show higher levels of attention.
1110-
1111-TABLE X
1112:E VALUATION OF H YPERPARAMETER λ1 FOR RSSPA
1113-
1114-TABLE XII
1115:E VALUATION OF H YPERPARAMETER λ3 FOR IFAA
1116-
1117-TABLE XI
1118:E VALUATION OF H YPERPARAMETER λ2 FOR IBSPA
1119-
1120-TABLE XIII
1121:E VALUATION OF H YPERPARAMETER λ4 FOR APFR
1122-
1123:show that training the model with multi-granularity attribute
1124-priors significantly alleviates this issue.
1125-
1126-fluctuates with increasing λ4 . After comprehensive consideration, we set λ4 to 1.
1127-
--
1137-this, we fixed λ2 at 30.
1138-λ3 for LIDFree . LIDFree is used to achieve ID-free alignment.
1139-As shown in Table XII, with λ3 ranging from 0.25 to 1.5, the
1140-best performance was observed when λ3 was set to 1.
1141:λ4 for LPRecon . LPRecon is used for implicit attribute alignment. As shown in Table XIII, the model’s performance
1142-
1143-F. Qualitative Results
1144:Fig. 6 lists the top-10 retrieval results of the Baseline and
1145-SPCL. We present the similarity calculation results, marking
1146-correct matches with red rectangles and incorrect matches with
1147-green rectangles. In each ID, compared to the Baseline in the
1148-second row, the first row shows a significant improvement
1149-achieved by our method. It is worth noting that even the
1150-mismatched images have a high degree of correspondence with
1151-the query text. The essence of Text-Based Person Retrieval
1152-is to use query text to find pedestrians that match specific
1153:descriptions. When pedestrian attributes are almost identical,
1154-with only subtle ID-specific differences, such mismatches are
1155-understandable. We will further investigate these subtle IDspecific differences in pedestrian characteristics in future work.
1156:Fig. 7 demonstrates our method’s excellent semantic alignment
1157:capabilities in both overall and partial attribute recognition.
1158-
1159-Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.
1160-
1161-1834
1162-
1163-IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026
1164-
1165:constraints, broadening the role of attributes. Finally, from
1166:a micro perspective of attribute-image interaction, we purify
1167-critical information through text-image semantic interactions
1168-within individuals. By constraining and exploring the semantic
1169:space from multiple angles, we aim to gain a deeper understanding of semantic polymorphism and commonality.
1170-R EFERENCES
1171-[1]
1172-
1173:Fig. 8. Selection of Images from the RSTPReid Dataset.
1174-
1175:When the model focuses on individual attributes, it accurately
1176-identifies specific objects. Moreover, when considering the
1177-entire sentence, our model captures more relevant and effective
1178-information compared to the Baseline.
1179-G. Potential Improvement
1180-As shown in Table III, although our method achieves
1181-superior results, its performance on the RSTPReid dataset is
1182-not as strong as that on the CUHK-PEDES and ICFG-PEDES
1183:datasets. We attribute this to the significant chaos inherent in
1184:visual data, which leads to large semantic differences among
1185:positive samples in the training data. As shown in Fig. 8, the
1186-large variations in lighting and angle differences in RSTPReid
1187:lead to significant visual semantic discrepancies. Even human
1188-vision cannot accurately match each image to the correct ID.
1189-Especially, images from ID 712 and ID 783 only depict small
1190-portions of the human body, missing critical information. In
1191-such cases, the relationship mining of our proposed method
--
1193-fully demonstrated. To address the limitations caused by the
1194-difficulty in capturing key information for semantic mining,
1195-the ID-free approach could be a potentially effective direction
1196-for further exploration. Furthermore, leveraging generative
1197:models, e.g. T2I-ADAPTER [64], to infer and complete visual
1198-semantics could also provide a promising solution. We will
1199-continue to explore these strategies in our future work.
1200-V. C ONCLUSION
1201-In this paper, we design the Semantic Polymorphism and
1202-Commonality Learning framework to comprehensively explore
1203-cross-modal representation and alignment. First, from a macro
1204:perspective of text-image alignment, we explored the sensitivity relations based on semantic polymorphism under ID
1205-constraints to better constrain the semantic representation
1206:space. Second, from the macro attribute-image alignment
1207:perspective, we investigated the commonalities of multigranularity attribute priors in text and images without ID
1208-
1209-C. Chen, M. Ye, and D. Jiang, “Towards modality-agnostic person reidentification with descriptive query,” in Proc. IEEE/CVF Conf. Comput.
1210-Vis. Pattern Recognit. (CVPR), Jun. 2023, pp. 15128–15137.
1211-[2] H. Li, J. Xiao, M. Sun, E. G. Lim, and Y. Zhao, “Transformerbased language-person search with multiple region slicing,” IEEE Trans.
--
1214-transformer learning with proximity data generation for text-based
1215-person search,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 8,
1216-pp. 7005–7016, Aug. 2024.
1217-[4] K. Niu, L. Huang, Y. Long, Y. Huang, L. Wang, and Y. Zhang,
1218:“Comprehensive attribute prediction learning for person search by
1219-language,” IEEE Trans. Image Process., vol. 33, pp. 1990–2003, 2024.
1220-[5] J. Kong, X. Tao, M. Jiang, and T. Liu, “Weakly supervised distribution
1221-discrepancy minimization learning with state information for person reidentification,” IEEE Trans. Multimedia, vol. 25, pp. 1903–1915, 2023.
1222-[6] C. Zhao et al., “Context-aware feature learning for noise robust person
--
1235-Artif. Intell., Aug. 2021, pp. 1068–1074.
1236-[11] Z. Wang et al., “SUM: Serialized updating and matching for textbased person retrieval,” Knowl.-Based Syst., vol. 248, Jul. 2022, Art.
1237-no. 108891.
1238-[12] K. Niu, L. Huang, Y. Huang, P. Wang, L. Wang, and Y. Zhang,
1239:“Cross-modal co-occurrence attributes alignments for person search
1240-by language,” in Proc. 30th ACM Int. Conf. Multimedia, Oct. 2022,
1241-pp. 4426–4434.
1242-[13] C. Wang, Z. Luo, Y. Lin, and S. Li, “Improving embedding learning
1243:by virtual attribute decoupling for text-based person search,” Neural
1244-Comput. Appl., vol. 34, no. 7, pp. 5625–5647, Apr. 2022.
1245-[14] Y. Chen, G. Zhang, Y. Lu, Z. Wang, and Y. Zheng, “TIPCB: A simple but
1246-effective part-based convolutional baseline for text-based person search,”
1247-Neurocomputing, vol. 494, pp. 171–181, Jul. 2022.
1248-[15] L. Gao, K. Niu, Z. Ma, B. Jiao, T. Tan, and P. Wang, “Text-guided
1249:visual feature refinement for text-based person search,” in Proc. Int.
1250-Conf. Multimedia Retr., Aug. 2021, pp. 118–126.
1251-[16] S. Yang, Y. Zhou, Z. Zheng, Y. Wang, L. Zhu, and Y. Wu, “Towards
1252:unified text-based person retrieval: A large-scale multi-attribute and
1253-language search benchmark,” in Proc. 31st ACM Int. Conf. Multimedia,
1254-Oct. 2023, pp. 4492–4501.
1255-[17] G. Wang, F. Yu, J. Li, Q. Jia, and S. Ding, “Exploiting the textual potential from vision-language pre-training for text-based person search,”
1256-2023, arXiv:2303.04497.
--
1268-
1269-[21] T. Chen, C. Xu, and J. Luo, “Improving text-based person search by
1270-spatial matching and adaptive threshold,” in Proc. IEEE Winter Conf.
1271-Appl. Comput. Vis. (WACV), Mar. 2018, pp. 1879–1887.
1272:[22] S. Li, T. Xiao, H. Li, W. Yang, and X. Wang, “Identity-aware textualvisual matching with latent co-attention,” in Proc. IEEE Int. Conf.
1273-Comput. Vis. (ICCV), Oct. 2017, pp. 1908–1917.
1274-[23] Y. Zhang and H. Lu, “Deep cross-modal projection learning for imagetext matching,” in Proc. Eur. Conf. Comp. Vis., 2018, pp. 686–701.
1275-[24] Y. Chen, R. Huang, H. Chang, C. Tan, T. Xue, and B. Ma, “Cross-modal
1276-knowledge adaptation for language-based person search,” IEEE Trans.
--
1288-[29] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training
1289-of deep bidirectional transformers for language understanding,” 2018,
1290-arXiv:1810.04805.
1291-[30] L. Gao, K. Niu, B. Jiao, P. Wang, and Y. Zhang, “Addressing information
1292:inequality for text-based person search via pedestrian-centric visual
1293-denoising and bias-aware alignments,” IEEE Trans. Circuits Syst. Video
1294-Technol., vol. 33, no. 12, pp. 7884–7899, Dec. 2023.
1295-[31] Z. Wang et al., “CAIBC: Capturing all-round information beyond
1296-color for text-based person retrieval,” in Proc. 30th ACM Int. Conf.
1297-Multimedia, Oct. 2022, pp. 5314–5322.
1298-[32] Y. Wu, Z. Yan, X. Han, G. Li, C. Zou, and S. Cui, “LapsCore: Languageguided person search via color reasoning,” in Proc. IEEE/CVF Int. Conf.
1299-Comput. Vis. (ICCV), Oct. 2021, pp. 1604–1613.
1300:[33] Z. Wei, Z. Zhang, P. Wu, J. Wang, P. Wang, and Y. Zhang, “Finegranularity alignment for text-based person retrieval via semanticscentric visual division,” IEEE Trans. Circuits Syst. Video Technol.,
1301-vol. 34, no. 9, pp. 8242–8252, Sep. 2024.
1302-[34] Z. Wang, Z. Fang, J. Wang, and Y. Yang, “ViTAA: Visual-textual
1303:attributes alignment in person search by natural language,” in Proc. Eur.
1304-Conf. Comp. Vis., 2020, pp. 402–420.
1305-[35] A. Zhu et al., “DSSL: Deep surroundings-person separation learning for
1306-text-based person retrieval,” in Proc. 29th ACM Int. Conf. Multimedia,
1307-Oct. 2021, pp. 209–217.
--
1320-AAAI Conf. Artif. Intell., 2022, pp. 4477–4485.
1321-[40] S. Li, M. Cao, and M. Zhang, “Learning semantic-aligned feature
1322-representation for text-based person search,” in Proc. IEEE Int. Conf.
1323-Acoust., Speech Signal Process. (ICASSP), May 2022, pp. 2724–2728.
1324:[41] A. Radford et al., “Learning transferable visual models from natural
1325-language supervision,” in Proc. Int. Conf. Mach. Learn., vol. 139, 2021,
1326-pp. 8748–8763.
1327:[42] C. Jia et al., “Scaling up visual and vision-language representation
1328-learning with noisy text supervision,” in Proc. 38th Int. Conf. Mach.
1329-Learn., vol. 139, pp. 4904–4916.
1330-[43] X. Han, S. He, L. Zhang, and T. Xiang, “Text-based person search with
1331-limited data,” 2021, arXiv:2110.10807.
--
1344-[47] D. A. Vaquero, R. S. Feris, D. Tran, L. Brown, A. Hampapur, and
1345-M. Turk, “Attribute-based people search in surveillance environments,”
1346-in Proc. Workshop Appl. Comput. Vis. (WACV), Dec. 2009, pp. 1–8.
1347-[48] A. Schumann and R. Stiefelhagen, “Person re-identification by deep
1348:learning attribute-complementary information,” in Proc. IEEE Conf.
1349-Comput. Vis. Pattern Recognit. Workshops (CVPRW), Jul. 2017,
1350-pp. 20–28.
1351-[49] S. Aggarwal, R. V. Babu, and A. Chakraborty, “Text-based person search
1352:via attribute-aided matching,” in Proc. IEEE Winter Conf. Appl. Comput.
1353-Vis. (WACV), Mar. 2020, pp. 2617–2625.
1354-[50] S. Bird, “NLTK: The natural language toolkit,” in Proc. COLING/ACL
1355-Interact. Presentation Sessions, 2006, pp. 69–72.
1356-[51] C. Manning, M. Surdeanu, J. Bauer, J. Finkel, S. Bethard, and

codex
SPCL 的消融指出 RSSPA 的贡献最大，IFAA 和 APFR 是进一步的属性线索。这个结构对我们很有用，因为它把“新视角”放在主损失上，把属性机制作为支撑而不是喧宾夺主。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 9 -l 12 'SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
LI et al.: SPCL: SEMANTIC POLYMORPHISM AND COMMONALITY LEARNING FOR TBPR

TABLE III
C OMPARISON W ITH SOTA M ETHODS ON THE RSTPR EID DATASET

for feature extraction, our approach yields competitive results,
particularly with a significant improvement in mINP. When
using the large-scale ViT for feature extraction, our model
achieves state-of-the-art results in Rank-1, mAP, and mINP,
while Rank-5 and Rank-10 also show competitive outcomes.
The data in the RSTPReid dataset inherently contains significant noise, which limits the semantic mining capability of our
method. A detailed discussion of this issue will be provided
in subsection G.
C. Ablation Study
In this section, we analyze the effectiveness of RSSPA,
IBSPA, IFAA, and APFR. We use the CLIP model fine-tuned
with cross-entropy loss on each dataset as our baseline. In our
setup, RSSPA replaces the cross-entropy loss to validate the
positive impact of relation sensitivity in practical applications.
RSSPA is designed to enhance comprehensive alignment
by employing relation sensitivity. The effectiveness of the
proposed RSSPA is well demonstrated through experiments
No.0 vs No.2 and No.1 vs No.3, as shown in Table IV. On
the CUHK-PEDES dataset, compared to the baseline, RSSPA
increases Rank-1 accuracy by over 3%, mAP by approximately 7%, and mINP by about 11%, which are substantial
performance improvements for both scales of vision-language
models. Similarly, significant improvements are observed on
the ICFG-PEDES dataset. To further validate our loss function,
we compare its performance when used as a single mapping
with our relation-sensitive loss. As shown in Table V, the proposed relation-sensitive approach improves Rank-1 accuracy,
mAP, and mINP by 2.76%, 6.16%, and 10.28%, respectively,
compared to traditional single mapping alignment. These
results demonstrate the high value of our research based on
relation sensitivity.
IBSPA is designed as an intra-modal feature constraint
module to assist cross-modal RSSPA. As shown in Table IV,
experiments No.0 vs No.1 demonstrate the effectiveness of
IBSPA on the CUHK-PEDES dataset, with Rank-1 accuracy
improvements of 2.55% and 2.03%, mAP improvements of
3.48% and 3.37%, and mINP improvements of 5.11% and
5.40% for two different scales of vision-language models,
respectively. On the ICFG-PEDES dataset, these enhancements are even more pronounced, with Rank-1 accuracy
increasing by over 4%, mAP by approximately 7%, and mINP

1831

by 3.22% and 4.41%. Experiments No.2 vs No.3 further
validates the positive impact of IBSPA when combined with
RSSPA, achieving ID-limited feature redistribution.
The RSSPA module is the fundamental alignment module of
our approach, combining with IBSPA to form a traditional IDlimited cross-modal semantic alignment. To further investigate
the overall effect of RSSPA + IBSPA, we replaced them
with the SDM (Similarity Distribution Matching) [45] loss
and CMPM (Cross-Modal Projection Matching) [23] loss
for achieving cross-modal explicit alignment. The results are
shown in Table VI. Compared to the model performance
using the SDM-based approach or CMPM-based approach,
the combination of RSSPA + IBSPA achieved comprehensive
advantages. This further demonstrates the positive impact of
our relation-sensitive strong-weak alignment on effectively
addressing the cross-modal matching ambiguity caused by
semantic differences among different instances under the same
ID. Specifically, it improves the accuracy and reliability of
matching instances across modalities.
IFAA is designed to explore attribute alignment beyond ID
constraints. The effectiveness of IFAA is validated through
experiments No.3 vs No.5 in Table IV. Using a small-scale
vision-language model on the CUHK-PEDES dataset, IFAA
improves Rank-1 accuracy, mAP, and mINP by 0.39%, 0.51%,
and 0.83%, respectively, compared to the ID-limited constraint approach. When using a large-scale vision-language
model, our Rank-1 accuracy on the CUHK-PEDES dataset
reaches 76.93%. These results indicate that larger-scale visionlanguage models can indeed have a positive impact on
accuracy.
APFR is designed to deeply explore the impact of implicit
multi-granularity attribute priors alignment on cross-modal
interaction. Experiments No.3 vs No.4 in Table IV show that
on the CUHK-PEDES dataset, APFR significantly improves
mAP and mINP accuracy, with nearly 0.5% improvement in
mAP and close to 1% in mINP for both scales of visionlanguage models. Similar comprehensive improvements are
observed on the ICFG-PEDES dataset. No. 6 in Table IV
validates the combined effectiveness of all components. Interestingly, while our method performs well with small-scale
vision-language models, the cumulative effect of all components shows a slight decrease when switching to large-scale
vision-language models. We analyze that this is due to the
large-scale models learning more redundant information due
to their complexity, causing a conflict between IFAA’s explicit
attribute alignment and APFR’s implicit attribute alignment,
increasing optimization difficulty.
D. Evaluation of Attribute
In this section, we discuss the relevant settings for attribute
alignment during our experiments. These experiments are
based on small-scale vision-language model.
We evaluated the impact of extracting different sets of
multi-granularity attribute priors on model performance. As
shown in Table VII, the model performance initially improves
as the number of multi-granularity attribute priors extracted
from each caption increases. However, when the number
of multi-granularity attribute priors exceeds four, the model

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

1832

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

TABLE IV
A BLATION S TUDIES ON SPCL K EY C OMPONENTS ON CUHK-PEDES AND ICFG-PEDES

TABLE V
S INGLE M APPING V S . R ELATION -S ENSITIVE C ROSS -M ODAL
P OLYMORPHIC M APPING ON CUHK-PEDES. TSA M EANS
T RAINDITIONAL S INGLE A LIGNMENT. RSA M EANS
R ELATION -S ENSITIVE A LIGNMENT

TABLE VI
E VALUATION OF RSSPA+IBSPA FOR C ROSS -M ODAL A LIGNMENT.
* D ENOTES THE A BBREVIATIONS FOR IFAA AND APFR

TABLE VII
E VALUATION OF D IFFERENT S ETS OF M ULTI -G RANULARITY
ATTRIBUTE P RIORS . MGAP R EFERS TO THE M ULTI -G RANULARITY
ATTRIBUTE P RIORS

Fig. 6. A demonstration of the top-10 retrieval results for SPCL (the first
row) and Baseline (the second row) on CUHK-PEDES. Correct retrievals are
indicated with a red rectangle, while incorrect ones are marked with a green
rectangle.

TABLE IX
E VALUATION OF W ORD L EVEL AND MGAP G UIDANCE FOR
C ROSS -M ODAL F USION AND R ECONSTRUCTION . MGAP I S
THE M ULTI -G RANULARITY ATTRIBUTE P RIORS
TABLE VIII
E VALUATION OF D IFFERENT L AYERS FOR C ROSS -M ODAL
I NTERACTION N ETWORK

performance declines. We analyze that this is because the
model needs to focus on the most important information in the
sentence, and extracting too many multi-granularity attribute
priors can distract the model’s attention, leading to a negative
effect.
We also validated the impact of using varying numbers
of Transformer blocks in the cross-modal fusion of APFR.
As shown in Table VIII, model performance fluctuates with

different numbers of layers. Ultimately, we selected the optimal configuration of four Transformer blocks.
Additionally, we compared the use of word level attributes
with multi-granularity attribute priors. As shown in Table IX,
multi-granularity attribute priors provide more accurate critical
information to the model, resulting in better performance. We
analyze that this is because training the model with word
level attributes disperses its attention, disrupts the overall
alignment of visual representations, and lacks the ability to
combine multi-granularity attributes. This causes the model to
lack discriminative judgment on specific parts when analyzing
sentence structures or visual representations. When using text
to search for a target, if a non-target image shares word level
attributes with the target, it can lead to confusion. Experiments

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

LI et al.: SPCL: SEMANTIC POLYMORPHISM AND COMMONALITY LEARNING FOR TBPR

1833

Fig. 7. Visualization of comprehensive semantic attention on the baseline model and SPCL. The brighter areas show higher levels of attention.

TABLE X
E VALUATION OF H YPERPARAMETER λ1 FOR RSSPA

TABLE XII
E VALUATION OF H YPERPARAMETER λ3 FOR IFAA

TABLE XI
E VALUATION OF H YPERPARAMETER λ2 FOR IBSPA

TABLE XIII
E VALUATION OF H YPERPARAMETER λ4 FOR APFR

show that training the model with multi-granularity attribute
priors significantly alleviates this issue.

fluctuates with increasing λ4 . After comprehensive consideration, we set λ4 to 1.

E. Parameter Tuning
In this subsection, we evaluated the hyperparameter settings
for each component. These experiments are based on smallscale vision-language model.
λ1 for LRS ensi . LRS ensi is set as the foundational loss for our
model. As shown in Table X, we evaluated λ1 values from 10
to 40. Ultimately, λ1 was set to 30.
λ2 for LIDBased . LIDBased assists LRS ensi in promoting globallevel ID-limit alignment. As shown in Table XI, changes in
λ2 significantly affect the results. Model performance initially
improves with increasing λ2 but then declines. Considering
this, we fixed λ2 at 30.
λ3 for LIDFree . LIDFree is used to achieve ID-free alignment.
As shown in Table XII, with λ3 ranging from 0.25 to 1.5, the
best performance was observed when λ3 was set to 1.
λ4 for LPRecon . LPRecon is used for implicit attribute alignment. As shown in Table XIII, the model’s performance

F. Qualitative Results
Fig. 6 lists the top-10 retrieval results of the Baseline and
SPCL. We present the similarity calculation results, marking
correct matches with red rectangles and incorrect matches with
green rectangles. In each ID, compared to the Baseline in the
second row, the first row shows a significant improvement
achieved by our method. It is worth noting that even the
mismatched images have a high degree of correspondence with
the query text. The essence of Text-Based Person Retrieval
is to use query text to find pedestrians that match specific
descriptions. When pedestrian attributes are almost identical,
with only subtle ID-specific differences, such mismatches are
understandable. We will further investigate these subtle IDspecific differences in pedestrian characteristics in future work.
Fig. 7 demonstrates our method’s excellent semantic alignment
capabilities in both overall and partial attribute recognition.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

1834

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026

constraints, broadening the role of attributes. Finally, from
a micro perspective of attribute-image interaction, we purify
critical information through text-image semantic interactions
within individuals. By constraining and exploring the semantic
space from multiple angles, we aim to gain a deeper understanding of semantic polymorphism and commonality.
R EFERENCES
[1]

Fig. 8. Selection of Images from the RSTPReid Dataset.

When the model focuses on individual attributes, it accurately
identifies specific objects. Moreover, when considering the
entire sentence, our model captures more relevant and effective
information compared to the Baseline.
G. Potential Improvement
As shown in Table III, although our method achieves
superior results, its performance on the RSTPReid dataset is
not as strong as that on the CUHK-PEDES and ICFG-PEDES
datasets. We attribute this to the significant chaos inherent in
visual data, which leads to large semantic differences among
positive samples in the training data. As shown in Fig. 8, the
large variations in lighting and angle differences in RSTPReid
lead to significant visual semantic discrepancies. Even human
vision cannot accurately match each image to the correct ID.
Especially, images from ID 712 and ID 783 only depict small
portions of the human body, missing critical information. In
such cases, the relationship mining of our proposed method
is weakened, and the advantages of the method cannot be
fully demonstrated. To address the limitations caused by the
difficulty in capturing key information for semantic mining,
the ID-free approach could be a potentially effective direction
for further exploration. Furthermore, leveraging generative
models, e.g. T2I-ADAPTER [64], to infer and complete visual
semantics could also provide a promising solution. We will
continue to explore these strategies in our future work.
V. C ONCLUSION
In this paper, we design the Semantic Polymorphism and
Commonality Learning framework to comprehensively explore
cross-modal representation and alignment. First, from a macro
perspective of text-image alignment, we explored the sensitivity relations based on semantic polymorphism under ID
constraints to better constrain the semantic representation
space. Second, from the macro attribute-image alignment
perspective, we investigated the commonalities of multigranularity attribute priors in text and images without ID

C. Chen, M. Ye, and D. Jiang, “Towards modality-agnostic person reidentification with descriptive query,” in Proc. IEEE/CVF Conf. Comput.
Vis. Pattern Recognit. (CVPR), Jun. 2023, pp. 15128–15137.
[2] H. Li, J. Xiao, M. Sun, E. G. Lim, and Y. Zhao, “Transformerbased language-person search with multiple region slicing,” IEEE Trans.
Circuits Syst. Video Technol., vol. 32, no. 3, pp. 1624–1633, Mar. 2022.
[3] H. Wu, W. Chen, Z. Liu, T. Chen, Z. Chen, and L. Lin, “Contrastive
transformer learning with proximity data generation for text-based
person search,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 8,
pp. 7005–7016, Aug. 2024.
[4] K. Niu, L. Huang, Y. Long, Y. Huang, L. Wang, and Y. Zhang,
“Comprehensive attribute prediction learning for person search by
language,” IEEE Trans. Image Process., vol. 33, pp. 1990–2003, 2024.
[5] J. Kong, X. Tao, M. Jiang, and T. Liu, “Weakly supervised distribution
discrepancy minimization learning with state information for person reidentification,” IEEE Trans. Multimedia, vol. 25, pp. 1903–1915, 2023.
[6] C. Zhao et al., “Context-aware feature learning for noise robust person
search,” IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 10,
pp. 7047–7060, Oct. 2022.
[7] Z. Wang et al., “CRIS: CLIP-driven referring image segmentation,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun.
2022, pp. 11676–11685.
[8] J. Feng et al., “Towards bridged vision and language: Learning crossmodal knowledge representation for relation extraction,” IEEE Trans.
Circuits Syst. Video Technol., vol. 34, no. 1, pp. 561–575, Jan. 2024.
[9] Z. Zheng, L. Zheng, M. Garrett, Y. Yang, M. Xu, and Y.-D. Shen, “Dualpath convolutional image-text embeddings with instance loss,” ACM
Trans. Multimedia Comput., Commun., Appl., vol. 16, no. 2, pp. 1–23,
May 2020.
[10] C. Wang, Z. Luo, Y. Lin, and S. Li, “Text-based person search via
multi-granularity embedding learning,” in Proc. 13th Int. Joint Conf.
Artif. Intell., Aug. 2021, pp. 1068–1074.
[11] Z. Wang et al., “SUM: Serialized updating and matching for textbased person retrieval,” Knowl.-Based Syst., vol. 248, Jul. 2022, Art.
no. 108891.
[12] K. Niu, L. Huang, Y. Huang, P. Wang, L. Wang, and Y. Zhang,
“Cross-modal co-occurrence attributes alignments for person search
by language,” in Proc. 30th ACM Int. Conf. Multimedia, Oct. 2022,
pp. 4426–4434.
[13] C. Wang, Z. Luo, Y. Lin, and S. Li, “Improving embedding learning
by virtual attribute decoupling for text-based person search,” Neural
Comput. Appl., vol. 34, no. 7, pp. 5625–5647, Apr. 2022.
[14] Y. Chen, G. Zhang, Y. Lu, Z. Wang, and Y. Zheng, “TIPCB: A simple but
effective part-based convolutional baseline for text-based person search,”
Neurocomputing, vol. 494, pp. 171–181, Jul. 2022.
[15] L. Gao, K. Niu, Z. Ma, B. Jiao, T. Tan, and P. Wang, “Text-guided
visual feature refinement for text-based person search,” in Proc. Int.
Conf. Multimedia Retr., Aug. 2021, pp. 118–126.
[16] S. Yang, Y. Zhou, Z. Zheng, Y. Wang, L. Zhu, and Y. Wu, “Towards
unified text-based person retrieval: A large-scale multi-attribute and
language search benchmark,” in Proc. 31st ACM Int. Conf. Multimedia,
Oct. 2023, pp. 4492–4501.
[17] G. Wang, F. Yu, J. Li, Q. Jia, and S. Ding, “Exploiting the textual potential from vision-language pre-training for text-based person search,”
2023, arXiv:2303.04497.
[18] X. Shu et al., “See finer, see more: Implicit modality alignment for textbased person retrieval,” in Proc. Eur. Conf. Comput. Vis. Workshops
(ECCVW), 2022, pp. 624–641.
[19] Z. Ji, J. Hu, D. Liu, L. Y. Wu, and Y. Zhao, “Asymmetric crossscale alignment for text-based person search,” IEEE Trans. Multimedia,
vol. 25, pp. 7699–7709, 2022.
[20] S. Li, T. Xiao, H. Li, B. Zhou, D. Yue, and X. Wang, “Person search
with natural language description,” in Proc. IEEE Conf. Comput. Vis.
Pattern Recognit. (CVPR), Jul. 2017, pp. 5187–5196.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.


codex
PDF 文本抽取把 SPCL 的表格数字压成了图片式排版，普通模式没有全读出来。我会用 `-layout` 再抽一次表格页，尽量保留列结构。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -layout -f 9 -l 11 'SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
LI et al.: SPCL: SEMANTIC POLYMORPHISM AND COMMONALITY LEARNING FOR TBPR                                                                           1831



                         TABLE III                                            by 3.22% and 4.41%. Experiments No.2 vs No.3 further
  C OMPARISON W ITH SOTA M ETHODS ON THE RSTPR EID DATASET                    validates the positive impact of IBSPA when combined with
                                                                              RSSPA, achieving ID-limited feature redistribution.
                                                                                 The RSSPA module is the fundamental alignment module of
                                                                              our approach, combining with IBSPA to form a traditional ID-
                                                                              limited cross-modal semantic alignment. To further investigate
                                                                              the overall effect of RSSPA + IBSPA, we replaced them
                                                                              with the SDM (Similarity Distribution Matching) [45] loss
                                                                              and CMPM (Cross-Modal Projection Matching) [23] loss
                                                                              for achieving cross-modal explicit alignment. The results are
                                                                              shown in Table VI. Compared to the model performance
                                                                              using the SDM-based approach or CMPM-based approach,
                                                                              the combination of RSSPA + IBSPA achieved comprehensive
                                                                              advantages. This further demonstrates the positive impact of
for feature extraction, our approach yields competitive results,              our relation-sensitive strong-weak alignment on effectively
particularly with a significant improvement in mINP. When                     addressing the cross-modal matching ambiguity caused by
using the large-scale ViT for feature extraction, our model                   semantic differences among different instances under the same
achieves state-of-the-art results in Rank-1, mAP, and mINP,                   ID. Specifically, it improves the accuracy and reliability of
while Rank-5 and Rank-10 also show competitive outcomes.                      matching instances across modalities.
The data in the RSTPReid dataset inherently contains signifi-                    IFAA is designed to explore attribute alignment beyond ID
cant noise, which limits the semantic mining capability of our                constraints. The effectiveness of IFAA is validated through
method. A detailed discussion of this issue will be provided                  experiments No.3 vs No.5 in Table IV. Using a small-scale
in subsection G.                                                              vision-language model on the CUHK-PEDES dataset, IFAA
                                                                              improves Rank-1 accuracy, mAP, and mINP by 0.39%, 0.51%,
                                                                              and 0.83%, respectively, compared to the ID-limited con-
C. Ablation Study                                                             straint approach. When using a large-scale vision-language
   In this section, we analyze the effectiveness of RSSPA,                    model, our Rank-1 accuracy on the CUHK-PEDES dataset
IBSPA, IFAA, and APFR. We use the CLIP model fine-tuned                       reaches 76.93%. These results indicate that larger-scale vision-
with cross-entropy loss on each dataset as our baseline. In our               language models can indeed have a positive impact on
setup, RSSPA replaces the cross-entropy loss to validate the                  accuracy.
positive impact of relation sensitivity in practical applications.               APFR is designed to deeply explore the impact of implicit
   RSSPA is designed to enhance comprehensive alignment                       multi-granularity attribute priors alignment on cross-modal
by employing relation sensitivity. The effectiveness of the                   interaction. Experiments No.3 vs No.4 in Table IV show that
proposed RSSPA is well demonstrated through experiments                       on the CUHK-PEDES dataset, APFR significantly improves
No.0 vs No.2 and No.1 vs No.3, as shown in Table IV. On                       mAP and mINP accuracy, with nearly 0.5% improvement in
the CUHK-PEDES dataset, compared to the baseline, RSSPA                       mAP and close to 1% in mINP for both scales of vision-
increases Rank-1 accuracy by over 3%, mAP by approxi-                         language models. Similar comprehensive improvements are
mately 7%, and mINP by about 11%, which are substantial                       observed on the ICFG-PEDES dataset. No. 6 in Table IV
performance improvements for both scales of vision-language                   validates the combined effectiveness of all components. Inter-
models. Similarly, significant improvements are observed on                   estingly, while our method performs well with small-scale
the ICFG-PEDES dataset. To further validate our loss function,                vision-language models, the cumulative effect of all compo-
we compare its performance when used as a single mapping                      nents shows a slight decrease when switching to large-scale
with our relation-sensitive loss. As shown in Table V, the pro-               vision-language models. We analyze that this is due to the
posed relation-sensitive approach improves Rank-1 accuracy,                   large-scale models learning more redundant information due
mAP, and mINP by 2.76%, 6.16%, and 10.28%, respectively,                      to their complexity, causing a conflict between IFAA’s explicit
compared to traditional single mapping alignment. These                       attribute alignment and APFR’s implicit attribute alignment,
results demonstrate the high value of our research based on                   increasing optimization difficulty.
relation sensitivity.
   IBSPA is designed as an intra-modal feature constraint                     D. Evaluation of Attribute
module to assist cross-modal RSSPA. As shown in Table IV,                        In this section, we discuss the relevant settings for attribute
experiments No.0 vs No.1 demonstrate the effectiveness of                     alignment during our experiments. These experiments are
IBSPA on the CUHK-PEDES dataset, with Rank-1 accuracy                         based on small-scale vision-language model.
improvements of 2.55% and 2.03%, mAP improvements of                             We evaluated the impact of extracting different sets of
3.48% and 3.37%, and mINP improvements of 5.11% and                           multi-granularity attribute priors on model performance. As
5.40% for two different scales of vision-language models,                     shown in Table VII, the model performance initially improves
respectively. On the ICFG-PEDES dataset, these enhance-                       as the number of multi-granularity attribute priors extracted
ments are even more pronounced, with Rank-1 accuracy                          from each caption increases. However, when the number
increasing by over 4%, mAP by approximately 7%, and mINP                      of multi-granularity attribute priors exceeds four, the model

        Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.
1832                                        IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 2, FEBRUARY 2026



                                                                       TABLE IV
                              A BLATION S TUDIES ON SPCL K EY C OMPONENTS ON CUHK-PEDES AND ICFG-PEDES




                                TABLE V
       S INGLE M APPING V S . R ELATION -S ENSITIVE C ROSS -M ODAL
        P OLYMORPHIC M APPING ON CUHK-PEDES. TSA M EANS
            T RAINDITIONAL S INGLE A LIGNMENT. RSA M EANS
                   R ELATION -S ENSITIVE A LIGNMENT




                               TABLE VI
   E VALUATION OF RSSPA+IBSPA FOR C ROSS -M ODAL A LIGNMENT.
        * D ENOTES THE A BBREVIATIONS FOR IFAA AND APFR




                               TABLE VII
     E VALUATION OF D IFFERENT S ETS OF M ULTI -G RANULARITY
  ATTRIBUTE P RIORS . MGAP R EFERS TO THE M ULTI -G RANULARITY
                        ATTRIBUTE P RIORS                                      Fig. 6. A demonstration of the top-10 retrieval results for SPCL (the first
                                                                               row) and Baseline (the second row) on CUHK-PEDES. Correct retrievals are
                                                                               indicated with a red rectangle, while incorrect ones are marked with a green
                                                                               rectangle.


                                                                                                             TABLE IX
                                                                                      E VALUATION OF W ORD L EVEL AND MGAP G UIDANCE FOR
                                                                                      C ROSS -M ODAL F USION AND R ECONSTRUCTION . MGAP I S
                                                                                           THE M ULTI -G RANULARITY ATTRIBUTE P RIORS

                               TABLE VIII
        E VALUATION OF D IFFERENT L AYERS FOR C ROSS -M ODAL
                      I NTERACTION N ETWORK




                                                                               different numbers of layers. Ultimately, we selected the opti-
                                                                               mal configuration of four Transformer blocks.
                                                                                  Additionally, we compared the use of word level attributes
                                                                               with multi-granularity attribute priors. As shown in Table IX,
                                                                               multi-granularity attribute priors provide more accurate critical
                                                                               information to the model, resulting in better performance. We
performance declines. We analyze that this is because the                      analyze that this is because training the model with word
model needs to focus on the most important information in the                  level attributes disperses its attention, disrupts the overall
sentence, and extracting too many multi-granularity attribute                  alignment of visual representations, and lacks the ability to
priors can distract the model’s attention, leading to a negative               combine multi-granularity attributes. This causes the model to
effect.                                                                        lack discriminative judgment on specific parts when analyzing
   We also validated the impact of using varying numbers                       sentence structures or visual representations. When using text
of Transformer blocks in the cross-modal fusion of APFR.                       to search for a target, if a non-target image shares word level
As shown in Table VIII, model performance fluctuates with                      attributes with the target, it can lead to confusion. Experiments

         Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.
LI et al.: SPCL: SEMANTIC POLYMORPHISM AND COMMONALITY LEARNING FOR TBPR                                                                            1833




Fig. 7. Visualization of comprehensive semantic attention on the baseline model and SPCL. The brighter areas show higher levels of attention.



                              TABLE X                                                                       TABLE XII
           E VALUATION OF H YPERPARAMETER λ1 FOR RSSPA                                     E VALUATION OF H YPERPARAMETER λ3 FOR IFAA




                             TABLE XI                                                                       TABLE XIII
           E VALUATION OF H YPERPARAMETER λ2 FOR IBSPA                                     E VALUATION OF H YPERPARAMETER λ4 FOR APFR




show that training the model with multi-granularity attribute
                                                                               fluctuates with increasing λ4 . After comprehensive considera-
priors significantly alleviates this issue.
                                                                               tion, we set λ4 to 1.
E. Parameter Tuning
   In this subsection, we evaluated the hyperparameter settings                F. Qualitative Results
for each component. These experiments are based on small-                         Fig. 6 lists the top-10 retrieval results of the Baseline and
scale vision-language model.                                                   SPCL. We present the similarity calculation results, marking
   λ1 for LRS ensi . LRS ensi is set as the foundational loss for our          correct matches with red rectangles and incorrect matches with
model. As shown in Table X, we evaluated λ1 values from 10                     green rectangles. In each ID, compared to the Baseline in the
to 40. Ultimately, λ1 was set to 30.                                           second row, the first row shows a significant improvement
   λ2 for LIDBased . LIDBased assists LRS ensi in promoting global-            achieved by our method. It is worth noting that even the
level ID-limit alignment. As shown in Table XI, changes in                     mismatched images have a high degree of correspondence with
λ2 significantly affect the results. Model performance initially               the query text. The essence of Text-Based Person Retrieval
improves with increasing λ2 but then declines. Considering                     is to use query text to find pedestrians that match specific
this, we fixed λ2 at 30.                                                       descriptions. When pedestrian attributes are almost identical,
   λ3 for LIDFree . LIDFree is used to achieve ID-free alignment.              with only subtle ID-specific differences, such mismatches are
As shown in Table XII, with λ3 ranging from 0.25 to 1.5, the                   understandable. We will further investigate these subtle ID-
best performance was observed when λ3 was set to 1.                            specific differences in pedestrian characteristics in future work.
   λ4 for LPRecon . LPRecon is used for implicit attribute align-              Fig. 7 demonstrates our method’s excellent semantic alignment
ment. As shown in Table XIII, the model’s performance                          capabilities in both overall and partial attribute recognition.

         Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:01:40 UTC from IEEE Xplore. Restrictions apply.

web search: 
web search: "Shape-centered Representation Learning" visible-infrared person re-identification ScRL
web search: 
web search: "Spatio-temporal Feature-level Augmentation Vision Transformer" video-based person re-identification
web search: 
web search: "Feature-level Augmentation Vision Transformer" "video-based person"
web search: 
web search: 10.1016/j.patcog.2025.111813
web search: 
web search: https://doi.org/10.1016/j.patcog.2025.111813
web search: 
web search: "10.1016/j.patcog.2025.111813" "Pattern Recognition"
web search: 
web search: "Spatial Temporal High Frequency Learning" "Visible Infrared" "Person"
web search: 
web search: "Weakly Supervised Text-Based Person Re-Identification" CMMT
codex
下面按“怎么把观察构造成创新”拆，不按模块复述。

**1. Shape-centered representation learning for visible-infrared person re-identification**

1. 触发观察：VI-ReID 过去主要学外观共性，但外观正是最受可见光和红外差异影响的东西。作者抓住一个具体现象：人体形状天然跨模态稳定，也有身份信息，但红外图像经人体解析得到的形状会在手臂、腿部等位置出错。
2. 重定义动作：他们没有接受 SEFL 那种“形状不可靠所以丢掉”的叙事，而是改成“形状不是噪声，红外形状估计错误才是噪声”。关键词是 shape-centered、Infrared Shape Restoration、shape-related appearance。
3. 机制怎么长出来：机制几乎顺着问题出来。红外形状错，所以做 ISR 从红外外观特征里恢复形状。推理阶段不能依赖解析网络，所以做 SFP 把形状流能力蒸馏到外观流。外观里有背景和模态噪声，所以用 AFE 让形状去筛外观。
4. 证据闭环：关键不是总表涨点，而是分层消融。SFP、ISR、AFE 逐步把 SYSU all-search 从 70.2/66.9 提到 76.1/72.6。ISR 里空间注意力优于直接相加、只用红外外观、只用形状查询和通道注意力。AFE 里两阶段注意力优于直接拼接和通道注意力。形状单独、外观单独、二者结合的对照也支撑“互补”。
5. reviewer 为什么买账：这篇卖的是视角，不是注意力模块本身。真正的新意是把“形状不可靠”反转成“形状是跨模态锚点，但要先修红外形状，并用形状约束外观”。

**2. Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification**

1. 触发观察：弱监督文本 ReID 只有图文配对，没有身份标签。传统对比学习把非配对样本全当负样本，但实际存在同身份假负样本和局部相似负样本。
2. 重定义动作：作者把问题从“缺身份标签”改写成“严格一对一图文约束本身有害”。关键词是 false negatives、partially similar negatives、many-to-many relationship、similarity regulation、calibration alignment。
3. 机制怎么长出来：SR 用文本语义相似度生成软目标，但把对角线置零，避免正样本支配软目标，再让软目标调节硬目标。CA 则把伪标签分配建模为最优传输，用“类别多、每类样本少、近似均匀”的 ReID 数据先验约束伪标签。
4. 证据闭环：最关键是 SR 目标对照。只用软目标会崩到 R1 34.19，只用硬目标 72.66，普通软硬结合 73.08，作者的 SR 到 73.23。CA 在组件消融中继续提升 R1 和 mAP。注意力可视化显示模型能对齐 glasses、shirt with dots 等细粒度词。
5. reviewer 为什么买账：这篇卖的是问题口径。弱监督不是简单“无标签版监督学习”，而是训练目标里有系统性假负样本。机制不复杂，但和这个口径绑定得很紧。

**3. Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges**

1. 触发观察：真实部署里 ReID 同时遇到两个问题：摄像头环境随时间漂移，原始行人数据又不能集中上传。已有终身 ReID 解决时间漂移但依赖集中训练，已有联邦 ReID 解决隐私但没有持续学习。
2. 重定义动作：作者把问题定义成“边缘端的联邦终身 ReID”，再进一步说不同位置和时间学到的知识有空间时间相关性。关键词是 federated lifelong person ReID、spatial-temporal knowledge integration、task-relevant knowledge。
3. 机制怎么长出来：边缘端只训练自适应层，用原型回放抗遗忘。服务器不平均聚合，而是用任务原型均值计算客户端之间的任务相似度，并对历史任务加时间衰减，得到个性化聚合权重。自适应层把本地知识和服务器发来的空间时间知识组合起来。
4. 证据闭环：最有力的是服务器端消融。去掉空间聚合掉 9.44 mAP，去掉时间聚合掉 7.12 mAP，去掉空间时间联合掉 13.90 mAP。去掉原型回放掉 7.43 mAP。相关性热图还能对应预设的真实边缘相关组。
5. reviewer 为什么买账：这篇卖的是新设定加系统闭环。它不是单纯套联邦或终身，而是把真实部署中的“空间分布、时间漂移、隐私限制、边缘算力”合成一个新问题。

**4. Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification**

1. 触发观察：VVI-ReID 里已有中间模态方法有三类缺陷：生成假红外成本高，边缘类中间模态丢高频细节，单帧构造忽略时间信息。作者又观察到高频保留边缘、结构和运动变化，低频更多是背景、风格和冗余。
2. 重定义动作：他们把“构造中间模态”重定义为“构造序列级空间时间高频中间模态”。关键词是 spatial-temporal high-frequency、sequence-level intermediate modality、low-frequency components、high-pass filter。
3. 机制怎么长出来：ST-HPF 用 3D FFT 和二值高通滤波去掉空间时间低频。浅层有纹理和边缘，所以 SDC 用高频特征补偿局部细节。深层有语义和全局时序，所以 DSR 用高频序列特征引导语义精炼。MP 先去风格，避免跨模态噪声干扰交互。
4. 证据闭环：消融很对题。SDC、DSR、二者结合分别提升，DSR 贡献更大。插入深度实验显示 SDC 适合浅层，DSR 适合深层。只做空间高通或只做时间高通都不如 ST-HPF。替换为 PHA 和 FRB 也不如 3D ST-HPF。距离分布和 CAM 进一步说明类间间隔更大、关注区域更准。
5. reviewer 为什么买账：这篇同时卖视角和机制。视角是“跨模态视频 ReID 可以从频域高频重写中间模态”，机制是 3D 频域构造加分层特征利用。

**5. Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification**

1. 触发观察：视频 ReID transformer 准确率高，但很多方法靠复杂结构和大量参数。作者抓住两个视频 ReID 特性：身份不应随背景改变，视频里可能混入遮挡或其他人的异常帧。
2. 重定义动作：他们没有继续堆时序模块，而是把问题重写成“用特征级增广教会 ViT 视频 ReID 的不变性和异常帧可靠性”。关键词是 Feature-level Augmentation、Background Token、Altered Background ID Classification、Anomaly Frame Detection。
3. 机制怎么长出来：先加 background token，把前景身份和背景分开。SFA 随机替换背景 token，要求身份分类不变。TFA 插入其他身份的帧特征，训练 AFD 找异常帧。测试时不做增广，把 AFD 学到的能力转成 frame consistency score 来加权视频表征。
4. 证据闭环：组件消融显示 BG、空间增广、时间增广、refine 都有贡献。iLIDS-VID 从 91.3 R1 到 95.3，MARS-DL 也稳定提升。跨数据集 MARS-DL 到 iLIDS 从 45.3 到 50.0，说明不是只记住训练集。可视化显示 class token 看人，background token 看背景，consistency score 会压低遮挡和姿态突变帧。
5. reviewer 为什么买账：这篇卖的是机制设计的简洁性。它把 transformer token 结构和视频 ReID 的真实扰动结合起来，训练期做复杂事，测试期保留轻量模型。

**6. SPCL: Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval**

1. 触发观察：TBPR 里同一身份的不同图像和文本并不是完全等价，而是相似但互补。不同身份之间又可能共享属性，比如 blue backpack。传统一对一图文对齐和 ID 内属性对齐都会误处理这些关系。
2. 重定义动作：作者把 TBPR 重写成两个语义现象：同一身份内有 semantic polymorphism，不同身份间有 semantic commonality。关键词是 strong correspondence、weak correspondence、relation-sensitive alignment、ID-free attribute alignment、macro 和 micro。
3. 机制怎么长出来：RSSPA 在同一 ID 内按相似度建立强弱图文对应，而不是只拉近配对样本。IBSPA 在单模态内聚合同 ID 多种表达。IFAA 去掉 ID 限制，让属性和全批次图像建立关系。APFR 用属性先验指导图文融合并重构属性，强化微观属性感知。
4. 证据闭环：RSSPA 是主证据，CUHK 上 Rank-1 提升超过 3%，mAP 约 7%，mINP 约 11%。关系敏感对齐比传统单映射多 2.76 Rank-1、6.16 mAP、10.28 mINP。IBSPA、IFAA、APFR 继续带来增益。作者还比较了 RSSPA+IBSPA 和 SDM/CMPM，属性实验显示多粒度属性优于词级属性，属性数量过多反而伤害性能。
5. reviewer 为什么买账：这篇主要卖视角。它把“一个身份多描述”和“跨身份共享属性”命名成可优化的语义结构，RSSPA 是承重机制，属性模块是支撑这个视角的扩展。

**共同套路**

这批论文共同的做法是：先找到一个被旧训练目标错误处理的关系，比如一对一、全负样本、平均聚合、单帧中间模态、输入级增广，然后给它重新命名成更具体的问题。机制通常不是凭空发明，而是把新问题拆成两到三个障碍，每个障碍对应一个模块，最后用“替代机制消融”证明自己的重定义比普通涨点更可信。  

我主要依据本地 PDF 全文方法和消融页阅读，另外核对了 ScRL 和 FedSTIL 的 arXiv 页面，以及弱监督文本 ReID 相关背景 CPCL 页面：ScRL arXiv([arxiv.org](https://arxiv.org/abs/2310.17952?utm_source=openai))、FedSTIL arXiv([arxiv.org](https://arxiv.org/abs/2207.11759?utm_source=openai))、CPCL arXiv([arxiv.org](https://arxiv.org/abs/2401.10011?utm_source=openai))。
tokens used
297,745
下面按“怎么把观察构造成创新”拆，不按模块复述。

**1. Shape-centered representation learning for visible-infrared person re-identification**

1. 触发观察：VI-ReID 过去主要学外观共性，但外观正是最受可见光和红外差异影响的东西。作者抓住一个具体现象：人体形状天然跨模态稳定，也有身份信息，但红外图像经人体解析得到的形状会在手臂、腿部等位置出错。
2. 重定义动作：他们没有接受 SEFL 那种“形状不可靠所以丢掉”的叙事，而是改成“形状不是噪声，红外形状估计错误才是噪声”。关键词是 shape-centered、Infrared Shape Restoration、shape-related appearance。
3. 机制怎么长出来：机制几乎顺着问题出来。红外形状错，所以做 ISR 从红外外观特征里恢复形状。推理阶段不能依赖解析网络，所以做 SFP 把形状流能力蒸馏到外观流。外观里有背景和模态噪声，所以用 AFE 让形状去筛外观。
4. 证据闭环：关键不是总表涨点，而是分层消融。SFP、ISR、AFE 逐步把 SYSU all-search 从 70.2/66.9 提到 76.1/72.6。ISR 里空间注意力优于直接相加、只用红外外观、只用形状查询和通道注意力。AFE 里两阶段注意力优于直接拼接和通道注意力。形状单独、外观单独、二者结合的对照也支撑“互补”。
5. reviewer 为什么买账：这篇卖的是视角，不是注意力模块本身。真正的新意是把“形状不可靠”反转成“形状是跨模态锚点，但要先修红外形状，并用形状约束外观”。

**2. Similarity Regulation and Calibration Alignment for Weakly Supervised Text-Based Person Re-Identification**

1. 触发观察：弱监督文本 ReID 只有图文配对，没有身份标签。传统对比学习把非配对样本全当负样本，但实际存在同身份假负样本和局部相似负样本。
2. 重定义动作：作者把问题从“缺身份标签”改写成“严格一对一图文约束本身有害”。关键词是 false negatives、partially similar negatives、many-to-many relationship、similarity regulation、calibration alignment。
3. 机制怎么长出来：SR 用文本语义相似度生成软目标，但把对角线置零，避免正样本支配软目标，再让软目标调节硬目标。CA 则把伪标签分配建模为最优传输，用“类别多、每类样本少、近似均匀”的 ReID 数据先验约束伪标签。
4. 证据闭环：最关键是 SR 目标对照。只用软目标会崩到 R1 34.19，只用硬目标 72.66，普通软硬结合 73.08，作者的 SR 到 73.23。CA 在组件消融中继续提升 R1 和 mAP。注意力可视化显示模型能对齐 glasses、shirt with dots 等细粒度词。
5. reviewer 为什么买账：这篇卖的是问题口径。弱监督不是简单“无标签版监督学习”，而是训练目标里有系统性假负样本。机制不复杂，但和这个口径绑定得很紧。

**3. Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges**

1. 触发观察：真实部署里 ReID 同时遇到两个问题：摄像头环境随时间漂移，原始行人数据又不能集中上传。已有终身 ReID 解决时间漂移但依赖集中训练，已有联邦 ReID 解决隐私但没有持续学习。
2. 重定义动作：作者把问题定义成“边缘端的联邦终身 ReID”，再进一步说不同位置和时间学到的知识有空间时间相关性。关键词是 federated lifelong person ReID、spatial-temporal knowledge integration、task-relevant knowledge。
3. 机制怎么长出来：边缘端只训练自适应层，用原型回放抗遗忘。服务器不平均聚合，而是用任务原型均值计算客户端之间的任务相似度，并对历史任务加时间衰减，得到个性化聚合权重。自适应层把本地知识和服务器发来的空间时间知识组合起来。
4. 证据闭环：最有力的是服务器端消融。去掉空间聚合掉 9.44 mAP，去掉时间聚合掉 7.12 mAP，去掉空间时间联合掉 13.90 mAP。去掉原型回放掉 7.43 mAP。相关性热图还能对应预设的真实边缘相关组。
5. reviewer 为什么买账：这篇卖的是新设定加系统闭环。它不是单纯套联邦或终身，而是把真实部署中的“空间分布、时间漂移、隐私限制、边缘算力”合成一个新问题。

**4. Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification**

1. 触发观察：VVI-ReID 里已有中间模态方法有三类缺陷：生成假红外成本高，边缘类中间模态丢高频细节，单帧构造忽略时间信息。作者又观察到高频保留边缘、结构和运动变化，低频更多是背景、风格和冗余。
2. 重定义动作：他们把“构造中间模态”重定义为“构造序列级空间时间高频中间模态”。关键词是 spatial-temporal high-frequency、sequence-level intermediate modality、low-frequency components、high-pass filter。
3. 机制怎么长出来：ST-HPF 用 3D FFT 和二值高通滤波去掉空间时间低频。浅层有纹理和边缘，所以 SDC 用高频特征补偿局部细节。深层有语义和全局时序，所以 DSR 用高频序列特征引导语义精炼。MP 先去风格，避免跨模态噪声干扰交互。
4. 证据闭环：消融很对题。SDC、DSR、二者结合分别提升，DSR 贡献更大。插入深度实验显示 SDC 适合浅层，DSR 适合深层。只做空间高通或只做时间高通都不如 ST-HPF。替换为 PHA 和 FRB 也不如 3D ST-HPF。距离分布和 CAM 进一步说明类间间隔更大、关注区域更准。
5. reviewer 为什么买账：这篇同时卖视角和机制。视角是“跨模态视频 ReID 可以从频域高频重写中间模态”，机制是 3D 频域构造加分层特征利用。

**5. Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification**

1. 触发观察：视频 ReID transformer 准确率高，但很多方法靠复杂结构和大量参数。作者抓住两个视频 ReID 特性：身份不应随背景改变，视频里可能混入遮挡或其他人的异常帧。
2. 重定义动作：他们没有继续堆时序模块，而是把问题重写成“用特征级增广教会 ViT 视频 ReID 的不变性和异常帧可靠性”。关键词是 Feature-level Augmentation、Background Token、Altered Background ID Classification、Anomaly Frame Detection。
3. 机制怎么长出来：先加 background token，把前景身份和背景分开。SFA 随机替换背景 token，要求身份分类不变。TFA 插入其他身份的帧特征，训练 AFD 找异常帧。测试时不做增广，把 AFD 学到的能力转成 frame consistency score 来加权视频表征。
4. 证据闭环：组件消融显示 BG、空间增广、时间增广、refine 都有贡献。iLIDS-VID 从 91.3 R1 到 95.3，MARS-DL 也稳定提升。跨数据集 MARS-DL 到 iLIDS 从 45.3 到 50.0，说明不是只记住训练集。可视化显示 class token 看人，background token 看背景，consistency score 会压低遮挡和姿态突变帧。
5. reviewer 为什么买账：这篇卖的是机制设计的简洁性。它把 transformer token 结构和视频 ReID 的真实扰动结合起来，训练期做复杂事，测试期保留轻量模型。

**6. SPCL: Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval**

1. 触发观察：TBPR 里同一身份的不同图像和文本并不是完全等价，而是相似但互补。不同身份之间又可能共享属性，比如 blue backpack。传统一对一图文对齐和 ID 内属性对齐都会误处理这些关系。
2. 重定义动作：作者把 TBPR 重写成两个语义现象：同一身份内有 semantic polymorphism，不同身份间有 semantic commonality。关键词是 strong correspondence、weak correspondence、relation-sensitive alignment、ID-free attribute alignment、macro 和 micro。
3. 机制怎么长出来：RSSPA 在同一 ID 内按相似度建立强弱图文对应，而不是只拉近配对样本。IBSPA 在单模态内聚合同 ID 多种表达。IFAA 去掉 ID 限制，让属性和全批次图像建立关系。APFR 用属性先验指导图文融合并重构属性，强化微观属性感知。
4. 证据闭环：RSSPA 是主证据，CUHK 上 Rank-1 提升超过 3%，mAP 约 7%，mINP 约 11%。关系敏感对齐比传统单映射多 2.76 Rank-1、6.16 mAP、10.28 mINP。IBSPA、IFAA、APFR 继续带来增益。作者还比较了 RSSPA+IBSPA 和 SDM/CMPM，属性实验显示多粒度属性优于词级属性，属性数量过多反而伤害性能。
5. reviewer 为什么买账：这篇主要卖视角。它把“一个身份多描述”和“跨身份共享属性”命名成可优化的语义结构，RSSPA 是承重机制，属性模块是支撑这个视角的扩展。

**共同套路**

这批论文共同的做法是：先找到一个被旧训练目标错误处理的关系，比如一对一、全负样本、平均聚合、单帧中间模态、输入级增广，然后给它重新命名成更具体的问题。机制通常不是凭空发明，而是把新问题拆成两到三个障碍，每个障碍对应一个模块，最后用“替代机制消融”证明自己的重定义比普通涨点更可信。  

我主要依据本地 PDF 全文方法和消融页阅读，另外核对了 ScRL 和 FedSTIL 的 arXiv 页面，以及弱监督文本 ReID 相关背景 CPCL 页面：ScRL arXiv([arxiv.org](https://arxiv.org/abs/2310.17952?utm_source=openai))、FedSTIL arXiv([arxiv.org](https://arxiv.org/abs/2207.11759?utm_source=openai))、CPCL arXiv([arxiv.org](https://arxiv.org/abs/2401.10011?utm_source=openai))。
