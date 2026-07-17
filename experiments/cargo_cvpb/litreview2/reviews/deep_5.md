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
session id: 019ef835-6984-7aa0-8c3d-8f29983e257a
--------
user
你是 ReID 创新方法论拆解员 5。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(5)==
CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf
CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf
CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf
CLIP-driven fine-grained mining for text-based person search.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会先按项目规则确认当前目标和工作约束，再用 PDF 文本工具逐篇读方法部分。这里列了 6 篇而不是 5 篇，我会按文件清单全部拆解。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pwd && rg --files -g 'GOAL.md' -g 'CLAUDE.md' -g 'AGENTS.md' -g 'STATUS.md' -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
/Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf
3D-Aided Pedestrian Representation Learning for Video-Based Person Re-Identification.pdf
Global aggregated gradient-guided adversarial attacks for person re-identification.pdf
Dual-level modality debiasing learning for unsupervised visible-infrared person re-identification.pdf
Probabilistic Distribution Alignment for Text-Based Person Retrieval.pdf
Towards Anytime Retrieval - A Benchmark for Anytime Person Re-Identification.pdf
Camera-Proxy Enhanced Identity-Recalibration Learning for Unsupervised Visible-Infrared Person Re-Identification.pdf
Adversarial flow-based generative models for visible-to-Infrared person re-Identification.pdf
Part-Based Feature Complementary Denoising for Unsupervised Person Re-Identification.pdf
Hierarchical fusion and local-aware transformer for occluded person re-identification.pdf
Optimal Proxy Mining Contrastive Network for Unsupervised Person Re-Identification.pdf
Causal Clothes-Invariant Feature Learning for Cloth-Changing Person Re-ID.pdf
TP-LReID - Lifelong person re-identification using text prompts.pdf
Dual-Modality-Shared Learning and Label Refinement for Unsupervised Visible-Infrared Person ReID.pdf
InfinitePerson - Innovating Synthetic Data Creation for Generalization Person Re-Identification.pdf
Adaptive Occlusion-Aware Network for Occluded Person Re-Identification.pdf
MoDA - Mixture of Domain Adapters for Parameter-efficient Generalizable Person Re-identification.pdf
HPRNet - Human Parsing Reconstruction With Non-Local Multi-Scale Perception Network for Cloth-Changing Person Re-Identification.pdf
CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf
Spatial-Temporal Federated Learning for Lifelong Person Re-Identification on Distributed Edges.pdf
Mix-Modality Person Re-Identification - A New and Practical Paradigm [2025 ACM TOMM acm_browser_subscription].pdf
Cross-domain person re-identification via learning Heterogeneous Pseudo Labels.pdf
GAReID - Grouped and Attentive High-Order Representation Learning for Person Re-Identification.pdf
Beyond geometry - The power of texture in interpretable 3D person ReID.pdf
Lifelong person re-identification via dynamically knowledge adaptation and retention.pdf
CSGN - CLIP-driven semantic guidance network for Clothes-Changing Person Re-Identification.pdf
Hierarchical knowledge-guided reasoning for text-based person re-identification.pdf
Rethinking Joint Optimization in Feature Compression - Insights from Person Re-Identification.pdf
Hierarchical Proxy Learning for Cloth-Changing Person Re-Identification.pdf
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

codex
已确认要用 PDF 技能的文本抽取路径，不过这里不需要改文件或渲染版面。下一步我会尽量读取项目规则文件；如果沙箱只允许当前论文库目录，我会说明限制，然后继续按你给出的规则处理 PDF。
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
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/Desktop/reid-paper/AGENTS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

codex
项目目标仍是文献差距分析和候选方向形成，当前任务正好属于这一步。现在开始逐篇抽取前 8 页文本，我会重点抓方法章节里的问题定义、机制链条和证据设计，而不是复述网络结构。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
CCUP: A Controllable Synthetic Data Generation
Pipeline for Pretraining Cloth-Changing Person
Re-Identification Models

arXiv:2410.13567v3 [cs.CV] 30 Mar 2025

Yujian Zhao1 , Chengru Wu2 , Yinong Xu2 , Xuanzheng Du2 , Ruiyu Li2 , Guanglin Niu1*
1
School of Artificial Intelligence, Beihang University
2
Shen Yuan Honors College, Beihang University
{yjzhao1019, chengru wu, yinong xu, xuanzheng du, ruiyu li, beihangngl}@buaa.edu.cn

Abstract—Due to the high cost of constructing Cloth-changing
person reidentification (CC-ReID) data, the existing data-driven
models are hard to train efficiently on limited data, which
causes the issue of overfitting. To address this challenge, we
propose a low-cost and efficient pipeline specific to CC-ReID
tasks for generating controllable and high-quality synthetic data
simulating the surveillance scenarios. Particularly, we construct
a new self-annotated CC-ReID dataset named Cloth-Changing
Unreal Person (CCUP), containing 6,000 IDs, 1,179,976 images,
100 cameras, and 26.5 outfits per individual. Based on this largescale dataset, we introduce an effective and scalable pretrainfinetune framework for enhancing the generalization of the
traditional CC-ReID models. The extensive experimental results
demonstrate that our framework could improve the original
models such as two typical models TransReID and FIRe2 after
pretraining on CCUP and finetuning on a benchmark, and
outperform other state-of-the-art models. The dataset is available
at: https://github.com/yjzhao1019/CCUP.
Index Terms—Cloth-changing Person Re-identification, Lowcost Synthetic Dataset, Pretrain-finetune Framework

I. I NTRODUCTION
Person re-identification (ReID) aims to identify gallery
images containing persons with the same identity as the query
image in a cross-camera scenario. Furthermore, cloth-changing
person re-identification (CC-ReID) is a more challenging task
to identify the same person but with different clothes in realworld scenarios at large spatial and temporal scales.
In recent years, deep learning-based models [13]–[15] have
been widely used to learn the discriminative features of person
images for ReID and its extended task CC-ReID [16] .
However, there are two main challenges for the CC-ReID task.
Challenge 1: the high cost of sampling and labeling
real CC-ReID images limits the size of existing datasets,
causing low performance of training models due to the
lack of sufficient ground truth for supervision. Building
a ReID dataset requires a complex environmental setup of
places, devices and pedestrians, as well as manual labeling
without violating privacy (DukeMTMC-ReID has been retracted due to privacy concerns). In particular, the complexity
and costs of generating a CC-ReID dataset further increase
* Correspinding author. This work was supported by the National Natural
Science Foundation of China (No. 62376016).

significantly since it is difficult to capture images of the same
person wearing various outfits on a large spatial and temporal
scale. In contrast, synthetic datasets are emerging to reduce
costs and address privacy concerns. As shown in Tab. I, we
provide the statistic of identities (#IDs), images (#Images),
cameras (#Cam) and average outfits per identity (#avgClo)
of some typical CC-ReID and synthetic datasets. We could
observe that the whole size and especially #avgClo of all
the previous CC-ReID benchmark datasets such as PRCC,
LTCC and VC-Clothes are obviously limited, and Celeb-reID
and LaST are even created from celebrity street photography
and movies rather than real surveillance scenes. Besides, few
existing commonly-used synthetic datasets are not designed
for cloth-changing scenarios and therefore lack rich clothchanging ground truth. To address these issues, we propose
a controllable and low-cost pipeline for generating largescale synthetic data more suitable for CC-ReID tasks.
Challenge 2: cloth-irrelevant features are hard to be
extracted via the existing models straightly trained on
a limited CC-ReID dataset. Specific to the CC-ReID task,
the most pivotal purpose is to extract discriminative clothirrelevant features. Therefore, CAL [17] is proposed to extract
cloth-irrelevant features from original RGB images by penalizing the predictive power of the ReID model. AIM [18] is
proposed to analyze the impact of clothing on model inference
and eliminate clothing bias during training . Besides, various
auxiliary information such as gait [19], skeleton [20], and 3D
shape [21] could be exploited for supplementing more clothirrelevant features. However, all the previous CC-ReID models
suffer from extremely scarce training data, limiting their
performance specifically some advanced visual transformerbased models. To address this challenge, we employ a
scalable pretrain-finetune framework leveraging our largescale synthetic dataset to enhance the model performance
of CC-ReID.
Overall, the contributions of our work are three-fold:
• We construct a high-quality synthetic CC-ReID dataset
named CCUP with our low-cost and controllable data
generation pipeline, which is the first large-scale (over
1,000,000 images) dataset for the CC-ReID task.
• We exploit a scalable pretrain-finetune framework, which

TABLE I
S TATISTIC OF CC-R E ID AND SYNTHETIC DATASETS . H YPHENS REPRESENT THE NUMBER OF OUTFITS IS NOT PROVIDED .
Characteristic

#avgClo

3
12
17
15
29

Surveillance
Simulate
✓
✗
✗
✓
✓
✗
✓
✓

273,456
1,801,816
1,256,381
887,766

6
19
34
24

✗
✓
✓
✓

-

19.060
1,179,976

4
100

✓
✓

2.07
26.5

Dataset

#IDs

#Images

#Cam

PRCC [1]
Celeb-reID [2]
Celeb-reID-light [2]
LTCC [3]
DeepChange [4]
LaST [5]
NKUP [6]
NKUP+ [7]

221
1,052
590
152
1,124
10,862
107
361

33,698
34,186
10,842
17,138
178,407
228,000
9,738
40,217

Synthetic for ReID

PersonX [8]
RandPerson [9]
UnrealPerson [10]
ClonedPerson [11]

1,266
8,000
6,799
5,621

Synthetic for CC-ReID

VC-Clothes [12]
CCUP (ours)

512
6000

Real for CC-ReID

could improve the performance of CC-ReID via finetuning the same model pretrained on our large-scale
synthetic dataset CCUP.
• The extensive experimental results on multiple benchmark datasets including LTCC, VC-Clothes and NKUP
illustrate that our framework outperforms other state-ofthe-art baseline models significantly and consistently.
II. R ELATED W ORK
A. Cloth-changing person re-identification
Traditional ReID studies are highly dependent on clothing
appearance, which is not available to address unstrained clothing changes in real scenarios. Thus, there are an increasing
number of researches related to Cloth-Changing person ReIdentification (CC-ReID). Researchers first recognize the importance of data for the task and therefore construct many
benchmark datasets [1]–[7], [12]. Then, many innovative approaches have been proposed, the core idea of which is to
focus on cloth-irrelevant features. TransReID [16] propose
a pure transformer-based object ReID framework containing
novel modules such as jigsaw patch module and side information embeddings. FSAM [22] propose a two-stream framework
that learns discriminative body shape knowledge and transfers
it to complement the cloth-unrelated knowledge. Pos-Net [23]
reinforces the feature learning process by designing powerful complementary data augmentation strategies. IGCL [24]
proposed a novel framework where the human semantics are
leveraged and the identity is unchangeable to guide collaborative learning. IRM [25] propose a new instruct-ReID task
and a large-scale OmniReID benchmark as well as adaptive
triplet loss. Pixel sampling [26] propose a semantic-guided
approach that forces the model to automatically learn clothirrelevant signals by randomly changing clothes pixels. ISGAN [27] disentangles identity-related and unrelated features
from person images through an identity-shuffling technique
that exploits identification labels. FIRe2 [28] designs a finegrained feature mining module and presents a fine-grained

2.00
3.14
2.62
-

attribute recomposition module by recomposing image features
with different attributes. IFD [29] proposes an Identity-aware
Feature Decoupling learning framework to mine identityrelated features. However, these models have never been able
to identify cloth-changing person well due to the limited data.
B. Dataset synthesis
Techniques for dataset synthesis in CC-ReID domain are
mainly categorized into traditional graphics methods and deep
learning methods. Traditional graphics methods first generate
3D meshes, which are then imported into a physics engine to configure animations, add scenes, and simulate real
surveillance [8]–[12]. However, these methods still have some
shortcomings due to the inability of existing software to
generate high-resolution meshes on large scales. Deep learning
methods commonly use GAN [30] for continuous iteration to
synthesize dataset. CCPG [31] proposes a GAN based model
for clothing and pose transfer across identities to augment
images of more clothing variations. AFD-Net [32] proposes
a novel framework containing intra-class reconstruction and
inter-class adversary to disentangle the identity-related and
identity-unrelated features. What’s more, although diffusion
model [33] has received a lot of attention in the field of image
synthesis in recent years, there is no more mature work for
CC-ReID.
III. M ETHODOLOGY
A. Dataset generation
Considering the numerous advantages of synthetic data,
such as good controllability, a high degree of automation,
significantly lower costs compared to capturing real-world
data, and the excellent ability to simulate various real-world
environments, we prioritize synthetic data generation for CCReID tasks. Accordingly, we propose a low-cost, high-quality,
large-scale, and controllable data generation pipeline along
with a new dataset CCUP for pretraining CC-ReID models.
Specifically, the CCUP generation process consists of three
main procedures: (1) generating skeletal meshes of realistic

Fig. 1. Pipeline of our work. We first generate the skeletal mesh of person and provide a large number of clothing textures for cloth-changing. Then the
skeletal mesh is imported into the three scenarios to simulate surveillance and the person in the surveillance frames is detected. Finally, we construct a
large-scale dataset called CCUP containing 6,000 IDs, 100 cameras, and 1,1179,976 images. We pretrain the TransReID and FIRe2 two baseline models with
CCUP and finetune on PRCC, VC-Clothes, and NKUP benchmarks.

human characters, (2) simulating surveillance in diverse scenarios, and (3) producing self-labeled detection results for
cloth-changing pedestrian, as illustrated in Fig 1.
1) Generate person skeletal mesh: The skeletal mesh is the
basis for building synthetic data, which contains mesh data
and skeletal data. Mesh data stores vertex positions, normal
vectors and texture coordinates, etc. Skeletal data represents
the skeletal node hierarchy. Specifically, we construct the
person skeletal mesh using MakeHuman software, an opensource 3D human modeling software that helps users create
high-quality, vividly realistic human body models.
We modify Makehuman’s AssetDownloader plugin and employ it to collect assets from the MakeHuman community that
can be used to build 3D person models such as skin, hair,
clothes, etc. Besides, we modify MakeHuman’s MassProduce
plugin to create 6000 naked person skeletal meshes with
different physiological features. In this way, unique combinations of physiological parameters determine unique person
IDs, so we consider each naked person skeletal mesh to be
an ID and we get the set of IDs for the dataset ID =
{id1 , id2 , ..., idn }, where n is the number of identities in the
dataset. Benefiting from our modification of the AssetDownloader plugin, we collecte almost 3,000 clothing asset and
get different ensemble of outfits Clo = {clo1 , clo2 , ..., clom },
where m denotes the number of ensemble of outfits. In
turn, we can construct clothed skeleton meshes CSM =
{csm11 , csm12 , ..., csm1t1 , ..., csmn1 , csmn2 , ..., csmntn }:
csmij = {DressU p(idi , cloj ) | 1 ≤ i ≤ n, 1 ≤ j ≤ ti }, (1)
where ti (ti ≤ m) is the number of clothes for person idi and
DressU p(idi , cloj ) denotes wearing clothes cloj on person
idi . Inspired by Unrealperson [10] , we create more than
10,000 clothes textures, constructing the set of textures for the
https://github.com/makehumancommunity

dataset T = {t1 , t2 , ..., tr } for subsequent clothing changes,
where r denotes the number of textures of the dataset.
2) Simulate surveillance in scenarios: Unreal Engine is a
game engine that offers a wide range of rendering functions
and is popular in many applications such as game and movie
development. To generate the CC-ReID data, we employ
Unreal Engine (version 5.3.2) to simulate real surveillance
scenarios. We configure animations for the skeletal meshes
generated in section III-A1 so that they can walk around
the simulation scenarios. Then we replace the texture tk of
clothed skeleton meshed csm when they pass by different
cameras to simulate the more diverse cloth-changing, denoted
as RT (csm, t). Then, we choose three scenarios for simulation
in the epic games marketplace: an European alleyway, an office
building, and a park with 50, 25, and 25 cameras, respectively.
These set of three scenarios are denoted as S = {s1 , s2 , s3 }.
Particularly, we design travel routes of persons in three
scenarios and place cameras along the routes with diverse
viewpoints. Benefiting from a well-designed detection strategy,
the video of the person could be automatically captured if
this person passes under the camera. Then, the set of original
frames with automatically labeled person IDs, camera IDs and
cloth IDs are generated, denoted as OF = {of1 , ..., ofp }:
of = Sim(RT (csm, t), s), csm ∈ CSM, t ∈ T, s ∈ S, (2)
where Sim(RT (csm, t), s) denotes simulating the real
surveillance in scenarios s and obtaining the original frame for
RT (csm, t) of surveillance videos. To guarantee the quality
of labeled data, each image contains only one pedestrian by
adapting the starting time and the speed of this person.
3) Detect and label the bounding boxes: Based on the
surveillance video frames OF , we employ the advanced
https://www.unrealengine.com/

RTMdet [34] model to detect pedestrians and generate their
corresponding bounding boxes:
(
(x, y, h, w) if a person is in of,
Det(of ) =
(3)
0
if no person is in of,
where Det(of ) denotes detecting the person bounding boxes
of frames and return the coordinate information if a person
is in frames. As explained in section III-A2, each frame
contains only one individual, ensuring that the number of
bounding boxes per frame is either 0 or 1. Therefore, the
frame’s label corresponds directly to the label of the bounding box, which significantly simplifies the dataset labeling
process. Finally we get our dataset named CCUP D =
{(I1 , l1 ), (I2 , l2 ), ..., (IN , lN )}:
I = of [x : x + w, y : y + h], if Det(of ) ̸= 0,

(4)

where I denotes image of D and l is the ground truth label
of the identity.
In our pipeline, we not only change outfits at the mesh
body level but also apply diverse texture replacements for the
clothes, greatly enhancing the variety of outfits. Besides, our
approach is highly extensible, requiring only minor code modifications to generate entirely new datasets tailored to different
tasks. The cost of dataset generation is remarkably low due
to the high level of automation in the procedure of generating
video frames and self-labeling, allowing produce datasets with
hundreds of thousands of images in just a few days. Ultimately,
we generate a large-scale CC-ReID dataset CCUP, which
includes 6,000 individuals, 100 cameras, 1,179,976 images,
and an average of 26.5 outfits per individual.
B. Pretrain and finetune
Pretraining and finetuning are originated in the field of nature language processing, and have gradually been introduced
in the field of computer vision, but are rarely used in CCReID due to limitation of training data. After obtaining a large
amount of CC-ReID data, we introduce the pretrain-finetune
framework for CC-ReID tasks.
The model parameters θ is first pretrained on a large dataset
Dpre = {(x1 , y1 ), ..., (xp , yp )}, which helps the model capture
generalizable features from a broad distribution of data:
θ

t+1

t

= θ − η∇L(fθt (xi ), yi ),

(5)

where fθt (xi ) is model’s output for input xi , L is the loss
function such as cross-entropy or triplet and η represents learning rate. After pretraining, we finetune the model on the downstream benchmark dataset Df ine = {(x˜1 , y˜1 ), ..., (x˜q , y˜q )},
allows the model to adapt these learned features to the specific
task and obtain the best performance parameters θ∗ :
θ∗ = arg max R(fθ (x̃), ỹ), (x̃, ỹ) ∈ Df ine test ,
θ

(6)

where R denotes Rank-1 evaluation and Df ine test is the test
set of Df ine .

IV. E XPERIMENT
We pretrain two models on our CCUP and finetune in
the downstream benchmarks to extract more robust clothirrelevant features. Furthermore, we demonstrate the superiority of our proposed dataset by comparing the performance of
the model pretrained on our CCUP and other synthetic datasets
as well as several state-of-the-art baselines.
A. Dataset and evaluation metrics
For a comprehensive comparison, we select some typical
synthetic datasets for pretraining, including UnrealPerson [10],
PersonX [8], and ClonedPerson [11]. Besides, PRCC [1], VCClothes [12], and NKUP [6] are chosen as the downstream
benchmark datasets for finetuning and obtaining the evaluation
results. It is noteworthy that only clothes changing ground
truth samples are used in PRCC and VC-Clothes datasets
while both clothes-changing and clothes-consistent ground
truth samples are used in NKUP. The detailed statistics and
comparison of these datasets are shown in Tab. I. Subsequently, two frequently-used metrics Rank-1 and mAP are
employed to evaluate each model.
B. Implementation details
We select two representative baselines: a general ReID
model TransReID [16] with the backbone ViT [15] and a
CC-ReID model FIRe2 [28] with the backbone ResNet50
pretrained on ImageNet [35] as our pretrained models. Besides, the other 11 typical CC-ReID models are utilized for
comparison. Particularly, we fix some parameters the same
as other baselines for fair comparison. For TransReID, input
images are resized to 256 × 128, the patch size is set to 16,
and the stride size is 12. The SGD optimizer is employed with
the weight decay of 1×10−4 and the learning rate is initialized
as 4×10−3 . For FIRe2 , Input images are resized to 384 × 192
and the batch size is 32. The Adam optimizer is employed with
the weight decay of 5×10−4 and the learning rate is initialized
as 3.5 × 10−4 . We pretrain the two pretrained models for 10
epochs on UnrealPerson, Personx, ClonedPerson, and CCUP,
respectively, and finetune these models on the benchmark
datasets for 80 epochs.
C. Comparison with state-of-the-art methods
We compare our model with some state-of-the-art methods
in Tab. II, where CAL, AIM, and CCFA uses person clothlabels in training process. We observe that both the pretrained
models TransReID and FIRe2 empowered by our CCUP illustrate the superior performance than the other models. Specifically, TransReID+CCUP achieves 59.0% mAP on PRCC and
28.8% rank-1 on NKUP, and FIRe2 achieves 64.7% rank1 on PRCC, and 85.1% rank-1 and 85.0% mAP on VCClothes. The best performance of mAP on NKUP is obtained
by TransReID pretrained by UnrealPerson, showing the effectiveness and scalability of our pretrain-fine framework. In
particular, FIRe2 without pretraining outperforms TransReID,
while the performance of TransReID pretrained on our CCPU
dataset can be significantly improved. The experimental results

TABLE II
C OMPARISON OF STATE - OF - THE - ART METHODS ON CC-R E ID BENCHMARKS . B OLDED VALUES REPRESENT THE BEST VALUES OBTAINED ON THE SAME
BASELINE . U NDERLINE VALUES INDICATE THE OVERALL BEST PERFORMANCE IN EACH COLUMN .

Methods

Backbone

Pretrain dataset

PCB (ECCV18) [36]
MGN (MM18) [37]
FSAM (CVPR21) [22]
3DSL (CVPR21) [38]
LSD (IVC21) [39]
CAL (CVPR22) [17]
GI-ReID (CVPR22) [40]
AD-ViT (AVSS22) [41]
AIM (CVPR23) [18]
CCFA (CVPR23) [42]
IRM (CVPR24) [25]
TransReID (ICCV21) [16]
TransReID (ICCV21) [16]
TransReID (ICCV21) [16]
TransReID (ICCV21) [16]
TransReID (ICCV21) [16]
FIRe2 (TIFS24) [28]
FIRe2 (TIFS24) [28]
FIRe2 (TIFS24) [28]
FIRe2 (TIFS24) [28]
FIRe2 (TIFS24) [28]

ResNet50
ResNet50
ResNet50
ResNet50
ResNet50
ResNet50
ResNet50
ViT
ResNet50
ResNet50
ViT
ViT
ViT
ViT
ViT
ViT
ResNet50
ResNet50
ResNet50
ResNet50
ResNet50

UnrealPerson
PersonX
ClonedPerson
CCUP (ours)
UnrealPerson
PersonX
ClonedPerson
CCUP (ours)

demonstrate that our proposed dataset generation scheme and
the pretrain-finetune framework facilitate both the general and
CC-ReID models to extract more robust and distinguishing
identity-relevant features on CC-ReID tasks.
D. Impact of different pretraining dataset
From the results specific to different synthetic datasets
for pretraining as shown in Tab. II, it is obviously that the
performance of TransReID and FIRe2 could be consistently
and significantly improved after pretraining on all the datasets.
Besides, the model pretrained by CCUP outperforms that
pretrained on the other synthetic datasets. Specifically, we
exhibit the absolute performance and (improvement over the
model without pretraining) of our model. TransReID pretrained by CCUP achieves the best performance with 58.9%
(+13.0%) rank-1 and 59.0% (+10.8%) mAP on PRCC, 83.3%
(+10.2%) rank-1 and 83.1% (+8.8%) mAP on VC-Clothes,
as well as 28.8% (+7.6%) rank-1 on NKUP. Furthermore,
FIRe2 pretrained by CCUP obtains the best preformance on
PRCC and NKUP, achieving 64.7% (+5.6%) rank-1, 57.7
(+7.2%) mAP on PRCC, 26.4% (+4.0%) rank-1 and 17.8
(+1.8%) mAP on NKUP. These results illustrate that the
powerful person clothes-changing number of CCUP benefits
to performance improvements of Re-ID models.
E. Visualization
To visually compare the regions that a model fucuses on
before and after pre-training, we visualize the inference results
of FIRe2 on PRCC as shown in Fig. 2 following [43]. We

PRCC (CC)
rank-1
mAP
41.8
31.7
25.9
54.5
51.3
37.2
47.6
55.2
55.8
37.6
57.9
58.3
61.2
58.4
54.2
52.3
45.9
48.2
51.1
50.4
46.8
47.5
47.7
48.9
58.9
59.0
59.1
50.5
62.5
57.4
60.5
56.0
55.9
56.4
64.7
57.9

Datasets
VC-Clothes (CC)
rank-1
mAP
62.0
62.3
78.6
78.9
79.9
81.2
81.4
81.7
63.7
59.0
74.1
73.7
73.1
74.3
72.2
74.4
72.0
73.8
73.6
71.8
83.3
83.1
78.0
78.9
84.0
84.8
76.1
77.2
85.1
85.0
81.6
81.3

NKUP
rank-1
mAP
16.9
12.4
18.8
15.0
16.4
10.2
27.0
18.9
21.2
14.8
26.7
20.3
25.8
19.3
27.9
19.7
28.8
19.4
22.4
16.0
24.2
16.4
23.0
15.8
21.8
16.0
26.4
17.8

observe that the model is improved via pretrained on CCUP
in the following two aspects: (1) The model puts more focus
on the human body, rather than the background, as illustrated
by the three sets of images in the first row of Fig. 2. (2) The
model pays more attention to the face, neck, shoulders, wrists,
and shoes (Shoes change less frequently than other clothes )
of a person, which are more conducive to extracting clothirrelevant features.

Fig. 2. Visualization of FIRe2 evaluated on PRCC. (a) Original iamges. (b)
Results without pretraining. (c) Results with pretrain-finetune.

V. C ONCLUSIONS
In this paper, we propose a controllable synthetic dataset
generation pipeline and construct a CC-ReID dataset named
CCUP containing 6000 IDs, 100 cameras, 1,179,976 images,

and 26.5 outfits per individual. To overcome the difficulty
of extracting robust and accurate cloth-irrelevant features, we
introduce the pretrain-finetune framework from NLP. After
pretraining both FIRe2 and TransReID baselines, our model
outperforms other state-of-the-art methods on three benchmarks: PRCC, VC-Clothes, and NKUP. Furthermore, we conduct comparison experiments using different synthetic datasets
as pretrain datasets to demonstrate the superiority of CCUP.
R EFERENCES
[1] Qize Yang, Ancong Wu, and Wei-Shi Zheng, “Person re-identification
by contour sketch under moderate clothing change,” TPAMI, vol. 43,
no. 6, pp. 2029–2046, 2019.
[2] Yan Huang, Qiang Wu, Jingsong Xu, and Yi Zhong, “Celebrities-reid:
A benchmark for clothes variation in long-term person re-identification,”
in IJCNN. IEEE, 2019, pp. 1–8.
[3] Xuelin Qian, Wenxuan Wang, Li Zhang, Fangrui Zhu, Yanwei Fu, Tao
Xiang, Yu-Gang Jiang, and Xiangyang Xue, “Long-term cloth-changing
person re-identification,” in ACCV, 2020.
[4] Peng Xu and Xiatian Zhu, “Deepchange: A long-term person reidentification benchmark with clothes change,” in ICCV, 2023, pp.
11196–11205.
[5] Xiujun Shu, Xiao Wang, Xianghao Zang, Shiliang Zhang, Yuanqi
Chen, Ge Li, and Qi Tian, “Large-scale spatio-temporal person reidentification: Algorithm and benchmark,” TCSVT, pp. 4390–4403,
2022.
[6] Kai Wang, Zhi Ma, Shiyan Chen, Jinni Yang, Keke Zhou, and Tao
Li, “A benchmark for clothes variation in person re-identification,”
International Journal of Intelligent Systems, vol. 35, no. 12, pp. 1881–
1898, 2020.
[7] Mengmeng Liu, Zhi Ma, Tao Li, Yanfeng Jiang, and Kai Wang,
“Long-term person re-identification with dramatic appearance change:
Algorithm and benchmark,” in ACM MM, 2022, pp. 6406–6415.
[8] Xiaoxiao Sun and Liang Zheng, “Dissecting person re-identification
from the viewpoint of viewpoint,” in CVPR, 2019.
[9] Yanan Wang, Shengcai Liao, and Ling Shao, “Surpassing real-world
source training data: Random 3d characters for generalizable person reidentification,” in ACM MM, 2020, pp. 3422–3430.
[10] Tianyu Zhang, Lingxi Xie, Longhui Wei, Zijie Zhuang, Yongfei Zhang,
Bo Li, and Qi Tian, “Unrealperson: An adaptive pipeline towards
costless person re-identification,” in CVPR, 2021, pp. 11506–11515.
[11] Yanan Wang, Xuezhi Liang, and Shengcai Liao, “Cloning outfits
from real-world images to 3d characters for generalizable person reidentification,” in CVPR, 2022, pp. 4900–4909.
[12] Fangbin Wan, Yang Wu, Xuelin Qian, Yixiong Chen, and Yanwei Fu,
“When person re-identification meets changing clothes,” in CVPR
workshops, 2020, pp. 830–831.
[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep
residual learning for image recognition,” in CVPR, 2016, pp. 770–778.
[14] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin, “Attention
is all you need,” NIPS, vol. 30, 2017.
[15] Alexey Dosovitskiy, “An image is worth 16x16 words: Transformers for
image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.
[16] Shuting He, Hao Luo, Pichao Wang, Fan Wang, Hao Li, and Wei Jiang,
“Transreid: Transformer-based object re-identification,” in ICCV, 2021,
pp. 15013–15022.
[17] Xinqian Gu, Hong Chang, Bingpeng Ma, Shutao Bai, Shiguang Shan,
and Xilin Chen, “Clothes-changing person re-identification with rgb
modality only,” in CVPR, 2022, pp. 1060–1069.
[18] Zhengwei Yang, Meng Lin, Xian Zhong, Yu Wu, and Zheng Wang,
“Good is bad: Causality inspired cloth-debiasing for cloth-changing
person re-identification,” in CVPR, 2023, pp. 1472–1481.
[19] Chao Fan, Yunjie Peng, Chunshui Cao, Xu Liu, Saihui Hou, Jiannan
Chi, Yongzhen Huang, Qing Li, and Zhiqiang He, “Gaitpart: Temporal
part-based model for gait recognition,” in CVPR, 2020, pp. 14225–
14233.
[20] Haocong Rao and Chunyan Miao, “Transg: transformer-based skeleton
graph prototype contrastive learning with structure-trajectory prompted
reconstruction for person re-identification,” in CVPR, 2023, pp. 22118–
22128.

[21] Feng Liu, Minchul Kim, ZiAng Gu, Anil Jain, and Xiaoming Liu,
“Learning clothing and pose invariant 3d shape representation for longterm person re-identification,” in CVPR, 2023, pp. 19617–19626.
[22] Peixian Hong, Tao Wu, Ancong Wu, Xintong Han, and Wei-Shi Zheng,
“Fine-grained shape-appearance mutual learning for cloth-changing person re-identification,” in CVPR, 2021, pp. 10513–10522.
[23] Xuemei Jia, Xian Zhong, Mang Ye, Wenxuan Liu, and Wenxin
Huang, “Complementary data augmentation for cloth-changing person
re-identification,” TIP, vol. 31, pp. 4227–4239, 2022.
[24] Zan Gao, Shengxun Wei, Weili Guan, Lei Zhu, Meng Wang, and
Shengyong Chen, “Identity-guided collaborative learning for clothchanging person reidentification,” TPAMI, 2023.
[25] Weizhen He, Yiheng Deng, Shixiang Tang, Qihao Chen, Qingsong
Xie, Yizhou Wang, Lei Bai, Feng Zhu, Rui Zhao, Wanli Ouyang,
et al., “Instruct-reid: A multi-purpose person re-identification task with
instructions,” in CVPR, 2024, pp. 17521–17531.
[26] Xiujun Shu, Ge Li, Xiao Wang, Weijian Ruan, and Qi Tian, “Semanticguided pixel sampling for cloth-changing person re-identification,” IEEE
Signal Processing Letters, vol. 28, pp. 1365–1369, 2021.
[27] Chanho Eom, Wonkyung Lee, Geon Lee, and Bumsub Ham, “Disentangled representations for short-term and long-term person reidentification,” TPAMI, vol. 44, no. 12, pp. 8975–8991, 2021.
[28] Qizao Wang, Xuelin Qian, Bin Li, Xiangyang Xue, and Yanwei Fu,
“Exploring fine-grained representation and recomposition for clothchanging person re-identification,” TIFS, 2024.
[29] Haoxuan Xu, Bo Li, and Guanglin Niu, “Identity-aware feature decoupling learning for clothing-change person re-identification,” in ICASSP,
2025, pp. 1–5.
[30] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David
Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio,
“Generative adversarial networks,” Communications of the ACM, vol.
63, no. 11, pp. 139–144, 2020.
[31] Vuong D Nguyen, Pranav Mantini, and Shishir K Shah, “Contrastive clothing and pose generation for cloth-changing person reidentification,” in CVPR, 2024, pp. 7541–7549.
[32] Wanlu Xu, Hong Liu, Wei Shi, Ziling Miao, Zhisheng Lu, and Feihu
Chen, “Adversarial feature disentanglement for long-term person reidentification.,” in IJCAI, 2021, pp. 1201–1207.
[33] Jonathan Ho, Ajay Jain, and Pieter Abbeel, “Denoising diffusion
probabilistic models,” NIPS, vol. 33, pp. 6840–6851, 2020.
[34] Chengqi Lyu, Wenwei Zhang, Haian Huang, Yue Zhou, Yudong Wang,
Yanyi Liu, Shilong Zhang, and Kai Chen, “Rtmdet: An empirical study of designing real-time object detectors,” arXiv preprint
arXiv:2212.07784, 2022.
[35] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei,
“Imagenet: A large-scale hierarchical image database,” in CVPR. Ieee,
2009, pp. 248–255.
[36] Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, and Shengjin Wang, “Beyond
part models: Person retrieval with refined part pooling (and a strong
convolutional baseline),” in ECCV, 2018, pp. 480–496.
[37] Guanshuo Wang, Yufeng Yuan, Xiong Chen, Jiwei Li, and Xi Zhou,
“Learning discriminative features with multiple granularities for person
re-identification,” in ACM MM, 2018, pp. 274–282.
[38] Jiaxing Chen, Xinyang Jiang, Fudong Wang, Jun Zhang, Feng Zheng,
Xing Sun, and Wei-Shi Zheng, “Learning 3d shape feature for textureinsensitive person re-identification,” in CVPR, 2021, pp. 8146–8155.
[39] Ehsan Yaghoubi, Diana Borza, Bruno Degardin, and Hugo Proença,
“You look so different! haven’t i seen you a long time ago?,” Image
and Vision Computing, vol. 115, pp. 104288, 2021.
[40] Xin Jin, Tianyu He, Kecheng Zheng, Zhiheng Yin, Xu Shen, Zhen
Huang, Ruoyu Feng, Jianqiang Huang, Zhibo Chen, and Xian-Sheng
Hua, “Cloth-changing person re-identification from a single image with
gait prediction and regularization,” in CVPR, 2022, pp. 14278–14287.
[41] Kyung Won Lee, Bhavin Jawade, Deen Mohan, Srirangaraj Setlur, and
Venu Govindaraju, “Attribute de-biased vision transformer (ad-vit) for
long-term person re-identification,” in AVSS. IEEE, 2022, pp. 1–8.
[42] Ke Han, Shaogang Gong, Yan Huang, Liang Wang, and Tieniu Tan,
“Clothing-change feature augmentation for person re-identification,” in
CVPR, 2023, pp. 22066–22075.
[43] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra, “Grad-cam: Visual
explanations from deep networks via gradient-based localization,” in
ICCV, 2017, pp. 618–626.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
CCFL: Customized Client Federated Learning for
Unsupervised Person Re-identification
YI ZHENG, School of Information and Electronics Engineering, Jiangsu Vocational Institute of
Architectural Technology, Xuzhou, China
YONG ZHOU, School of Computer Science and Technology, China University of Mining and Technology,
Xuzhou, China
FAYAO LIU, Agency for Science, Technology and Research, Singapore, Singapore
JIAQI ZHAO, HANCHENG ZHU, and WENLIANG DU, School of Computer Science and
Technology, China University of Mining and Technology, Xuzhou, China
Federated learning-based person re-identification (Re-ID) aims to address the issue of data silos in surveillance systems caused by increasingly stringent regulations on sensitive data. However, due to differences
in data collection locations, times, and scales, severe non-independent and identically distributed (non-IID)
characteristics exist across different Re-ID datasets. Existing federated learning-based Re-ID methods often
adopt a unified model structure, which prevents the model from adapting well to diverse data environments,
thereby significantly degrading the overall Re-ID performance. To address the challenges of training neural
networks on non-IID data across different datasets, we propose a customizable federated learning framework.
First, customizable clients allow each organization to freely select suitable neural network training methods
and model architectures based on local data scales and prior knowledge, thus improving training outcomes.
Second, since traditional federated learning frameworks cannot achieve knowledge fusion through parameter
exchange between models with different architectures, we introduce an independent model, referred to as
the interaction model, specifically designed for knowledge exchange among clients. The interaction model
learns parameters (knowledge) from local models on each client through distillation learning. Subsequently,
the interaction model is uploaded to the server, where it undergoes parameter fusion (knowledge exchange)
with interaction models from other clients. Finally, the interaction model, enriched with knowledge from
other clients, guides local model training through knowledge distillation. It is worth noting that selecting
a lightweight interaction model, while potentially impacting Re-ID performance, can significantly reduce
communication costs between the server and clients.
This work was done by Y. Zheng while visiting Institute for Infocomm Research, A*STAR, Singapore.
This work was supported by the National Natural Science Foundation of China (Grant No. 62272461), and by the China
Scholarship Council (Grant No. 202206420034) which awarded Y. Zheng a scholarship for 1 year of study abroad at the
Agency for Science, Technology and Research.
Authors’ Contact Information: Yi Zheng, School of Information and Electronics Engineering, Jiangsu Vocational Institute
of Architectural Technology, Xuzhou, China; e-mail: yizheng@jsjzi.edu.cn; Yong Zhou (corresponding author), School of
Computer Science and Technology, China University of Mining and Technology, Xuzhou, China; e-mail: yzhou@cumt.edu.cn;
Fayao Liu, Agency for Science, Technology and Research, Singapore, Singapore; e-mail: Liu_Fayao@i2r.a-star.edu.sg; Jiaqi
Zhao, School of Computer Science and Technology, China University of Mining and Technology, Xuzhou, China; e-mail:
jiaqizhao@cumt.edu.cn; Hancheng Zhu, School of Computer Science and Technology, China University of Mining and
Technology, Xuzhou, China; e-mail: zhuhancheng@cumt.edu.cn; Wenliang Du, School of Computer Science and Technology,
China University of Mining and Technology, Xuzhou, China; e-mail: wldu@cumt.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM 1551-6865/2025/8-ART225
https://doi.org/10.1145/3735134
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

225:2

Y. Zheng et al.

CCS Concepts: • Computing methodologies → Visual content-based indexing and retrieval; Object
identification; Cooperation and coordination;
Additional Key Words and Phrases: person re-identification, federated learning, non-IID data, knowledge
transfer, knowledge distillation
ACM Reference format:
Yi Zheng, Yong Zhou, Fayao Liu, Jiaqi Zhao, Hancheng Zhu, and Wenliang Du. 2025. CCFL: Customized Client
Federated Learning for Unsupervised Person Re-identification. ACM Trans. Multimedia Comput. Commun.
Appl. 21, 8, Article 225 (August 2025), 21 pages.
https://doi.org/10.1145/3735134

1

Introduction

Person Re-identification (Re-ID) targets the identification of specific pedestrians in extensive
surveillance data [54]. Since Re-ID data is derived from video surveillance, it involves personal and
organizational privacy information (e.g., clothing, appearance, travel patterns), which imposes strict
requirements on data security. Moreover, due to variations in shooting times, locations, devices, and
angles, different Re-ID datasets exhibit significant disparities, leading to the Non-independent and
Identically Distributed (non-IID) problem, which is prevalent in cross-domain Re-ID research
[28, 37, 39].
In recent years, academic research has increasingly focused on minimizing the overexposure of
person Re-ID data due to privacy concerns. Many Re-ID datasets now require researchers to declare
their academic research purposes and sign relevant usage agreements. Simultaneously, researchers
have begun to shift away from supervised learning paradigms, exploring semi-supervised and
unsupervised learning approaches for Re-ID. However, methods based on semi-supervised learning
paradigms (e.g., domain adaptation and domain generalization [6, 33, 34]) remain inherently
dependent on labeled data (annotated source domain data). Moreover, due to the non-IID nature of
Re-ID datasets, unsupervised learning methods [52, 53] often exhibit inconsistent performance,
as they lack tailored training strategies and model architectures for different datasets. To achieve
higher performance in Re-ID models, researchers often adopt centralized training on multiple
Re-ID datasets to increase data scale and learn more generalizable features. However, as mentioned
earlier, due to the security requirements of Re-ID data, collecting sensitive surveillance information
from multiple organizations is generally unacceptable. In this context, the introduction of federated
learning [32] provides a solution to cross-domain learning in Re-ID data.
The core of federated learning algorithms is to enable independent training of model parameters
across multiple clients with private data sources. After a certain number of training iterations, each
client uploads its locally trained model parameters to a centralized server, instead of sharing raw
data across the network. The server then aggregates the parameters from multiple models using
model aggregation methods to obtain a fused model. Subsequently, the fused model parameters
are distributed to each client for the next round of training. This distributed learning approach
allows multiple clients to collaboratively train a shared network model using different data, while
ensuring that the original data remain stored locally, thus safeguarding the security of local data.
Currently, federated learning algorithms have been widely applied in handling data with higher
security sensitivity, such as financial and medical data [1, 14, 46].
In response to the growing concerns about data security in person Re-ID, researchers have
explored the potential of utilizing federated learning frameworks for Re-ID tasks [45, 49, 50].
As an experimental study, Zhuang et al. [56] conducted supervised training across nine largescale datasets to develop a robust Re-ID model. However, supervised training strategies rely
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

Customized Client Federated Learning for Unsupervised Person ReID

225:3

heavily on large amounts of labeled data, and the associated labeling costs limit their applicability.
Moreover, the labeling process itself could be perceived as a threat to the security of Re-ID data.
To address the limitations of manual data handling, they further proposed a federated Re-ID
method based on unsupervised learning, called FedUReID [55], which reduces dependency on data
labels. Nevertheless, these methods did not consider the statistical heterogeneity between different
datasets, which ultimately resulted in lower Re-ID accuracy.
Moreover, although some studies have attempted to mitigate the non-IID problem by designing
model architectures, parameters, and functional modules specifically tailored for handling heterogeneous data [38], these approaches still rely on a unified training strategy. First, according to the
performance summary table of unsupervised person Re-ID methods on various datasets, collected
and organized by the web site “Paper with Code,” different person Re-ID datasets often achieve
optimal performance with different methods. Second, different clients may possess varying levels of
computational resources, and large-scale computing devices, such as those used in TransReID-SSL
[30], are not commonly available.
To address the aforementioned issues, we propose a customizable federated unsupervised person
Re-ID framework. In this work, we utilize two unsupervised Re-ID training strategies based on
the DBSCAN clustering algorithm and the hierarchical clustering algorithm. Additionally, when
training large neural network models on small-scale datasets, these models, due to their large
number of parameters and finer feature descriptions, are more prone to overfitting. Therefore,
based on the number of training samples, we employ two representative models, ResNet-50 and
ResNet-34. It is worth noting that among existing clustering algorithms, DBSCAN and hierarchical
clustering are the most widely used and facilitate finding comparative methods, while requiring
only minimal prior data parameters. ResNet-50 and ResNet-34 are merely examples of complex and
simple models used to demonstrate the compatibility and scalability of the proposed method; the
choice of models is not limited to these in practical applications.
Due to the use of different model architectures, the model parameter averaging method commonly
used in federated learning, such as FedAvg [32], cannot be directly applied to the framework
proposed in this article. Therefore, we introduce an additional network model, referred to as the
interactive model, specifically designed to facilitate cross-client knowledge transfer. Specifically,
the interactive model is trained under the guidance of local models from each client to acquire
network parameters that closely approximate the performance of the local models. The interactive
model then conducts cross-client knowledge exchange using model parameter averaging methods.
Finally, the updated parameters from the interactive model are used to guide the training of the
local models, enabling them to learn more generalized and robust pedestrian features. Additionally,
if a slight sacrifice in Re-ID performance is acceptable, selecting a smaller-scale interactive model
can reduce the amount of data transmitted during cross-client communication, thereby improving
interaction efficiency.
In summary, the main contributions of this article are as follows:
— We propose a novel customizable client federated learning method that allows each client to
adopt independent training strategies and model structures.
— We introduce an innovative client information exchange method, where knowledge exchange
between neural network models with different structures is achieved through an additional
interactive model.
— By accepting a tradeoff in Re-ID performance, selecting a smaller model as the interactive
model reduces communication costs between clients and the server.
The remainder of this article is organized as follows. Section 2 provides a comprehensive review
of related work on unsupervised person Re-ID and federated learning. In Section 3, we analyze the
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

225:4

Y. Zheng et al.

experimental settings, rationale, and necessity for different configurations in federated learning
paradigms for person Re-ID tasks compared to traditional visual image tasks, and we outline the
proposed Customized Client Federated Learning (CCFL) framework. Then, in Section 4, we
present extensive experimental results that demonstrate the effectiveness of CCFL and analyze its
advantages in detail. Finally, Section 5 concludes the work of this article.
2

Related Works

The emergence of federated learning aims to address the problem of data silos caused by concerns
over data privacy and security. In situations where large-scale data sharing is difficult, federated
learning facilitates abstract knowledge exchange among different learning participants via model
parameter communication, without direct data exchange. This approach helps with local data
management and reduces the risk of data breaches. As a result, federated learning was initially
applied in fields that highly value data privacy or involve high-value data, such as the healthcare
and financial sectors [1, 14, 46].
2.1

Federated Learning Methods Based on General Clients

A key challenge in federated learning is the heterogeneity of source data caused by differences in
data sources and sampling methods. To date, numerous studies have been conducted to address the
issue of data heterogeneity. Li et al. [21] improved stability and ensured convergence on non-IID
data by adding a proximal term to the objective function. Tang et al. [41] proposed a Virtual
Heterogeneous Learning method, which shares a virtual homogeneous dataset unrelated to private
data among all clients to correct the heterogeneity of private data on each client. It also mitigates
distribution drift through feature calibration, thereby ensuring generalized performance. Dinh et al.
[9] introduced a Federated Learning algorithm based on Unsupervised Auxiliary Tasks, which can
effectively handle differences in data distribution and label spaces among clients without any prior
knowledge or assumptions. Li et al. [24] proposed a Federated Learning algorithm based on Local
Batch Normalization, which uses local batch normalization to alleviate feature shift and improve
model convergence speed during training. Luo et al. [29] balanced data quality and distribution by
manually filtering local samples. Qu et al. [35] aimed to find a universal model structure that could
handle heterogeneous data. Ma et al. [31] implemented a layered aggregation structure that achieves
layer-wise heterogeneity by scoring different neural network layers. Fang and Ye [11] utilized
additional public datasets and knowledge distillation to extract knowledge from heterogeneous
models.
2.2

Federated Learning Methods Based on Heterogeneous Clients

The aforementioned studies aim to improve effective training within the federated learning framework by addressing data distribution imbalances among different clients. In addition, some researchers have designed heterogeneous clients to adapt to varying data distributions. In these
works, the knowledge exchange methods for heterogeneous clients are relatively flexible and
varied.
He et al. [16] proposed a variant based on the Alternating Minimization method, which achieves
knowledge exchange between heterogeneous clients through distillation learning, combining features extracted from clients and classification soft labels from both the clients and the server.
Li and Wang [19] additionally used a large public dataset, integrating classification scores obtained
from heterogeneous clients on the public dataset as public knowledge on the server, which was
then distilled to all clients. Lin et al. [25] used an unlabeled dataset on the server to aggregate
models from each client and distilled a generalized model distributed to each client. Bistritz et al.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

Customized Client Federated Learning for Unsupervised Person ReID

225:5

[3] designed a distributed distillation method that eliminates the need for a central server, achieving joint optimization among clients through the transmission of soft labels between individual
clients.
However, for the task of person Re-ID, using additional datasets for feature distillation is impractical. First, if existing person Re-ID datasets are used, the existing issues of privacy risks, insufficient
class-balanced data samples, and imbalanced inter-class sample distribution render these datasets
unsuitable as public datasets. If a new person Re-ID dataset is created, the manual processing of
the original data and the high labor costs pose new limitations. If non-person Re-ID data is used,
the non-IID problem between pedestrian data and natural data must be pre-studied. Therefore,
we extract knowledge from heterogeneous data using different model architectures and training
strategies to address the non-IID problem between different person Re-ID datasets. It then designs
a universal interactive model to distill knowledge from local models, using the exchanged global
knowledge to guide the training of local models, thereby solving the knowledge-exchange problem
between heterogeneous clients.
3

Proposed Method

In the introduction of related works in Section 2.2, much of the prior research is based on assumptions regarding the potential challenges federated learning might encounter. Based on these
assumptions, researchers split large datasets into subsets exhibiting non-IID characteristics and
designed corresponding solutions. However, the task of person Re-ID has a unique and realistic
application background, where the distributional differences in Re-ID data are more complex than
those in most artificially designed constrained scenarios.
Firstly, due to the complexity and sensitivity of surveillance data, manual data annotation is an
unsustainable task. Faced with a vast amount of raw data, it is clearly impractical to adjust the
data distribution for generalized clients. Similarly, the proposal by Li and Wang [19] for building
large-scale public datasets is also infeasible.
Secondly, He et al. [16] and Bistritz et al. [3] consider clients as edge devices, where only the server
possesses sufficient computational capacity (a similar setting was used in the work by Lin et al.
[25]). However, in person Re-ID scenarios, clients are typically institutions or organizations with a
certain level of data processing capabilities, such as schools, banks, stations, airports, and so on.
The server, on the other hand, is usually a governmental department responsible for overseeing
the entire area or a trusted third-party organization. Therefore, in some cases, the server may lack
sufficient computational resources to handle person Re-ID data from multiple scenarios and may
be more vulnerable to data attacks.
Thus, we hypothesize that the server should be considered a coordinating entity with limited
computational resources, whose role is merely to control the overall parameters of the federated
learning process (e.g., the number of iterations, number of clients) and aggregate the interactive
model parameters uploaded by each client, without holding any person Re-ID data or engaging in
feature learning. Each client should also minimize manual preprocessing of raw data, refrain from
using data labels, and adopt unsupervised methods to train the neural network parameters.
3.1

Overall Workflow

In deep learning model training, different datasets typically require distinct training strategies,
training parameters, and model architectures. We propose a novel CCFL framework. Examining
the development of deep learning-based person Re-ID methods, it is evident that the performance
improvement from adjusting intricate training parameters for different datasets is often surpassed
by the introduction of new training strategies and model structures (e.g., hierarchical clusteringbased methods [26] generally perform worse than DBSCAN-based methods [8], and traditional
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

225:6

Y. Zheng et al.

Fig. 1. The main workflow of proposed CCFL.

convolutional neural network methods [52] underperform compared to Transformer-based methods
[30]). Additionally, we assume that in practical scenarios, the data distributions and the number of
identities in person Re-ID datasets from different sources are unknown. Therefore, based on the
number of training samples and leveraging existing deep learning research experience, we make an
initial determination of suitable training strategies and model structures. To facilitate knowledge
exchange between different clients, we design a server-client knowledge distillation approach to
enable information exchange between models with varying architectures.
To facilitate knowledge exchange between different clients, we designed a server-client mutual
knowledge distillation method, which enables information transfer between models of different
structures. As shown in Figure 1, the CCFL framework consists of a server and multiple clients,
each of which holds a person Re-ID dataset as local data.
The entire federated learning framework starts with the server. In the initial iteration, the
server creates the necessary interactive neural network model structure and distributes it to the
participating clients. Subsequently, each client selects an appropriate local training strategy and
model architecture for deep learning training based on the scale of its local dataset. After a certain
number of iterations, knowledge from the local model is transferred to the interactive model via
knowledge distillation and then uploaded to the server. Starting from the second iteration, the
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

Customized Client Federated Learning for Unsupervised Person ReID

225:7

Algorithm 1: CCFL

server, upon receiving all client models, performs a weighted fusion of the parameters from the
interactive models, using the local data scale reported by the clients as the weighting factor.
The parameter fusion process for the interactive model is shown in Equation (1):
𝜃𝑖 ←

𝐾
Õ
𝑘=1

𝑁𝑘
Í𝐾

𝜃 𝑖𝑘 .

(1)

𝑛=1 𝑁𝑛

In this context, 𝜃 𝑖𝑘 represents the interactive model parameters uploaded by each dataset,
𝑁𝑛 (𝑛 = 1, 2, . . . , 𝐾), denotes the sample size of each of the 𝐾 datasets, and 𝜃 𝑖 refers to the fused
interactive model parameters after aggregation.
The fused interactive model parameters are redistributed to each client. Upon receiving the updated interactive model parameters, clients use them to guide local model training through knowledge distillation. After training, the local model serves as the teacher model, passing knowledge
back to the interactive model through knowledge distillation, and then uploading the parameters
to the server. Algorithm 1 details the overall workflow. Specifically, our approach follows the basic
paradigm of federated learning, assuming there are 𝐾 clients, each with an independent dataset
𝑋𝑘 = {𝑥 1, 𝑥 2, 𝑥 3, . . . , 𝑥𝑛𝑘}, where 𝑥𝑛𝑘 represents the sample size of the local dataset on the 𝑘th client.
Our goal is for 𝐾 clients to learn robust model parameters 𝜙𝑘 (·, 𝜃𝑏𝑒𝑠𝑡 ) without accessing data from
other clients.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

225:8

Y. Zheng et al.

At each client, the local model 𝜃 𝑘 is used to learn the hidden feature representation knowledge
from the local data, while the interactive model 𝜃 𝑖𝑘 , defined and distributed by the server, is
employed to transfer local knowledge to other clients and receive knowledge to guide the training
of the local model 𝜃 𝑘 . In practice, at the initialization stage, ImageNet pre-trained parameters are
uniformly used as the initial parameters for both the interactive and local models. However, in
the first round of training, the interactive model’s guidance for the local model training is either
meaningless or harmful. Thus, during the first round, the local model is trained directly on local data
without guidance from the interactive model. Ultimately, under the server’s control, the federated
learning process terminates after a pre-determined number of training rounds, and each client
outputs the optimal neural network model parameters achieved during training.
3.2

Unsupervised Learning Strategy Based on DBSCAN Clustering

In recent years, researchers have primarily employed three unsupervised clustering methods
in person Re-ID studies: k-means clustering [10], hierarchical clustering [8, 26], and DBSCAN
clustering [5, 18, 30]. Among these methods, k-means clustering typically requires estimating the
number of classes (i.e., the number of person identities) in the data and relies on domain adaptation
strategies to achieve a more accurate representation of pedestrian image samples. Due to these
limitations, k-means clustering is not suitable for the experimental setting described in the opening
paragraph of this section, which aims to minimize preprocessing and manual intervention on the
raw data.
In contrast, hierarchical clustering and DBSCAN clustering have been widely applied to person
Re-ID tasks, with substantial research supporting their effectiveness. In the DBSCAN algorithm,
hyperparameter selection can be guided by insights from existing studies without the need for
extensive statistical analysis of the dataset. On the other hand, hierarchical clustering does not
require the setting of specific hyperparameters.
In this section, our goal is to divide the dataset 𝑋𝑘 containing 𝑁𝑘 samples into 𝑀𝑘 clusters and
assign pseudo-labels to the unlabeled samples based on their cluster membership. By calculating
the centroid representation of each cluster, we obtain a set of cluster centroid features 𝐶, which
serve as classification weights for unsupervised neural network parameter learning.
To reduce the demand for training resources, we select ClusterNCE loss as the objective function
for this training strategy, in accordance with the realistic conditions assumed earlier. Specifically,
the ClusterNCE loss is represented as follows:

exp 𝑢 · 𝐶𝑖+ /𝜏
L𝐶 −𝑁𝐶𝐸 = − log Í𝑀
,
(2)
𝑘
𝑖=1 exp (𝑢 · 𝐶𝑖 /𝜏)
where 𝑢 represents the sample features, 𝐶𝑖+ denotes the positive cluster features, and 𝜏 is the
temperature parameter that controls the distance scaling.
In the proposed federated learning method, excluding the initial steps, the detailed training process of the unsupervised person Re-ID client based on DBSCAN clustering is illustrated in Figure 2.
To better leverage the knowledge from other datasets, this section utilizes features extracted by the
interactive model for clustering and generating pseudo-labels.
Specifically, based on general intuition, the interactive model, having received knowledge from
multiple clients, is more robust than the local model at the same training iteration. Therefore, this
section uses the interactive model to extract unlabeled features from the local data, applying the
DBSCAN clustering algorithm to generate pseudo-labels and cluster centroid vectors, which serve
as the basis for the ClusterNCE loss computation. In the implementation, the cluster centroid vectors
are normalized to avoid the impact of vector magnitude on loss calculation. Based on the analysis
by Zheng et al. [52], to ensure consistency in similarity measurement, cosine similarity loss is
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.


codex
前两篇已经读到方法主线。CCFL 的方法章节在第 8 页后还有细节，后面我会补读续页，避免只按截断文本判断。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'CFPER - Coarse-to-Fine Part-Experts Retrieval for Efficient Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
2025 IEEE International Conference on Multimedia and Expo (ICME) | 979-8-3315-9495-4/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICME59968.2025.11209843

CFPER: Coarse-to-Fine Part-Experts Retrieval for
Efficient Person Re-identification
1st Shiyu Wang

2nd Mingming Lu⋆

School of Computer Science and Engineering
Central South University
ChangSha, China
224711075@csu.edu.cn

School of Computer Science and Engineering
Central South University
ChangSha, China
mingminglu@csu.edu.cn

Abstract—Most existing person re-identification (ReID) methods focus on improving retrieval accuracy by refining features,
which fails to balance accuracy with inference efficiency. We
observe that query difficulty varies: global features suffice for
simple cases, while fine-grained part features are required for
challenging cases, such as occlusion. However, current methods
typically use the same feature extraction network for all queries,
which may limit accuracy on difficult queries or waste computational resources on easier ones. To address this, we propose
a two-stage Coarse-to-Fine Dynamic Retrieval mechanism that
adaptively allocates resources based on query difficulty. For
“easy” queries, only global features are used in the coarse
stage, and inference terminates early. For “hard” queries, part
features are extracted in the fine stage for detailed matching.
To further reduce computational costs, we introduce Mixture
of Experts for part feature extraction, where a router assigns
patches to part experts using topology annotations, and only activates body-relevant experts, enabling accurate part identification
with significant computation reduction. Extensive experiments
demonstrate that our method achieves competitive performance
while significantly reducing computational costs compared to
state-of-the-art methods.
Index Terms—Transformer-based Person Re-identification,
Computational Efficiency, Early-Exit, Mixture of Experts.

I. I NTRODUCTION
Person re-identification (ReID) aims to retrieve a specific
person from a large database of person images captured by
diverse non-overlapping cameras, which is widely applied in
many domains, such as criminal investigation and smart city
[1]. Most existing person ReID methods primarily focus on
improving retrieval accuracy, often overlooking the importance
of computational efficiency. Specifically, these methods [1]–
[7] process all queries with the same network and use uniform
features for retrieval, without accounting for the varying
difficulty of queries, which may limit accuracy on challenging
queries or consume needless computational resources on easier
ones. In real-world applications, the retrieval difficulty of different queries varies: some query images exhibit distinct discriminative features, and global features alone are sufficient for
accurate retrieval. For these “easy” queries, excessive reliance
on fine-grained part features may lead to false matches, as
different pedestrians may share similar body part appearances.
In contrast, “hard” queries, such as those affected by occlusion,
pose variation, or subtle inter-class variation, require finegrained part features to support more detailed matching.

TABLE I
P ERFORMANCE AND FLOP S OF T RANS R E ID WITH D IFFERENT
R ETRIEVAL F EATURES DURING I NFERENCE . “G” D ENOTES THE C ASE
USING ONLY THE G LOBAL F EATURE . “G+P” D ENOTES THE C ASE USING
THE C ONCATENATED F EATURE OF G LOBAL F EATURE AND F OUR PART
F EATURES .

Dataset
Feature
mAP (%)
Rank-1 (%)
FLOPs (G)

Market-1501
G
G+P
87.1
88.2
94.6
95.0
11.35
12.29

DukeMTMC
G
G+P
79.6
80.6
89.0
89.6
11.35
12.29

Occluded-Duke
G
G+P
53.8
55.7
61.6
64.2
11.35
12.29

To verify this, we conducted experiments on three datasets
with increasing retrieval difficulty: Market-1501 (holistic) [8],
DukeMTMC (holistic) [9], and Occluded-DukeMTMC (occluded) [2]. We tested two TransReID [4] configurations: (1)
global features, and (2) concatenated global and four part
features, and report mAP, Rank-1 and FLOPs in Table I. The
results show that introducing part features increases computational cost by 1.1× in terms of FLOPs, but provides only minor
improvements in mAP (1.1% and 1.0%) and Rank-1 (0.4% and
0.6%) on two holistic datasets, while resulting in significant
improvements (1.9% mAP, 3.1% Rank-1) on the more challenging occluded dataset. This indicates that global features
are generally sufficient to support accurate matching for “easy”
queries with distinctive visual features, while fine-grained part
features are more beneficial for challenging queries. Inspired
by the above observations, we propose a two-stage Coarse-toFine Dynamic Retrieval (CFDR) mechanism that adaptively
allocates computational resources based on the difficulty of
queries to balance performance and efficiency. Specifically, an
early-exit threshold is computed to evaluate query difficulty.
For “easy” queries, only global features are used for fast
retrieval in the coarse stage, and inference terminates early to
avoid unnecessary computational resource consumption. Only
“hard” queries proceed to the fine stage, where fine-grained
part features are extracted for more precise matching.
For part feature extraction in the fine inference stage of
CFDR, existing methods face several limitations. Some methods [5] rely on external models (e.g., pose estimation and
human parsing) to extract part features, introducing additional
computational costs. Other methods [4], [10] divide spatially
adjacent patches or pixels into fixed-size groups to learn

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.

Fig. 1. The framework of CFPER consists of two stages: coarse inference stage and fine inference stage. In the coarse stage, a global feature G and N patch
embeddings Z are extracted. An early-exit decision is made based on the threshold ⌈N α⌉: if met, the query is considered “easy”, only global feature is used
for retrieval and inference terminates. Otherwise, the query is considered “hard” and proceeds to the fine stage to extract part features using the Part-aware
Mixture of Experts (PMoE). PMoE includes a Patch-to-Part Router (PPR) and M Part Experts (PE). Each patch is routed to the corresponding body part via
PPR, and the fine-grained part feature ei of each body part is learned by the dedicated PE. Here, PWAP refers to Probability Weighted Average Pooling.

part features but ignore human topology priors, leading to
inaccurate part detection. Therefore, to further reduce computational costs and improve part identification accuracy, we
introduce Mixture of Experts and human topology priors to
propose a Part-aware Mixture of Experts (PMoE). The PMoE
consists of a Patch-to-Part Router (PPR) and a set of Part
Experts (PE). The PPR routes patches to corresponding body
parts based on identity labels and human topology annotations, enabling accurate part identification without introducing
additional computational costs. Each body part is assigned
to a dedicated Part Expert for fine-grained feature learning.
And during inference, PMoE only activates body-relevant
experts based on binary visibility routing weights to reduce
computational costs without hurting the performance.
Finally, we combine CFDR and PMoE to construct our
Coarse-to-Fine Part-Experts Retrieval model (CFPER).
The main contributions of our work can be summarized as
follows:
• We propose a novel Coarse-to-Fine dynamic retrieval
mechanism that balances computational efficiency and retrieval accuracy by adaptively allocating resources based
on the query difficulty.
• We innovatively apply MoE in part feature extraction,
which uses human topology annotations to guide the
router for accurate part identification without extra computational costs, and only body-relevant experts are ac-

tivated during inference, significantly reducing inference
costs without sacrificing performance.
• The proposed CFPER achieves competitive performance
compared to SOTA methods on both holistic and occluded person ReID datasets, while also improving inference efficiency.
II. M ETHODOLOGY
In this section, we introduce the proposed Coarse-to-Fine
Part-Experts Retrieval (CFPER) in detail. An overview of
CFPER is shown in Fig. 1.
A. Coarse Inference Stage
Coarse Feature Extractor. We use the pre-trained Vision
Transformer (ViT) [11] as our feature extractor. Given a person
image X, the output of the encoder can be divided into two
parts: a global feature G ∈ R1×D and N patch embeddings
Z = [z1 , . . . , zN ] ∈ RN ×D .
Early-exit strategy. In the coarse inference stage, we
introduce an early-exit strategy to balance performance and
efficiency. Traditional early-exit strategies [12] are typically
used in classification tasks, relying on classifier scores. However, in person ReID, the training and testing identities are
inconsistent and the number of testing identities is unknown,
making the classifier score-based exit metric unsuitable. To
address this, we propose an early-exit strategy for person

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.

Fig. 2. Illustration of the early-exit strategy based on the first-order differences
of similarity scores.

ReID, which determines whether to exit early by evaluating
the query difficulty based on the first-order differences of
similarity scores.
To assess query difficulty, our intuition is to count the
number of body-related patches. As shown in Fig. 2, we
first compute the cosine similarity between global feature
G and N patch embeddings Z, obtaining similarity scores
S = [s1 , . . . , sN ]. The scores S are sorted in descending
D
order to obtain S D = [sD
1 , . . . , sN ]. We then compute the
first-order differences D = [D1 , . . . , DN −1 ], where Di =
D
sD
i − si+1 . We assume that body and background/occlusion
features are separate classes in the feature space, so there
should be a noticeable feature transition when a body feature
shifts to a background/occlusion feature. Therefore, we select
the maximum value in D and use its corresponding index
η = arg maxi (D) as the split point between the body and
background/occlusion regions. Patches with the top-η similarity scores are considered as body regions, while the rest
are considered as background/occlusion regions. Finally we
introduce an early-exit threshold α ∈ [0, 1]. If η ≥ ⌈N α⌉, the
query is considered “easy” with sufficient visible body regions
for high-confidence retrieval. In this case, only global features
G are used for retrieval, and the whole inference terminates.
Otherwise, the query is considered “hard” and proceeds to the
fine inference stage to extract fine-grained part features.
Supervision Loss of Coarse Inference Stage. We adopt
cross-entropy loss as ID loss to supervise the learning of
encoder:
LCOARSE = LID (G),
(1)
B. Fine Inference Stage
In the fine inference stage, we employ PMoE to extract finegrained part features for “hard” queries to support detailed
retrieval. PMoE consists of a Patch-to-Part Router and a set
of Part Experts.
1) Patch-to-Part Router. The PPR module learns to route
patches embeddings Z to M+1 parts C = [c0 , c1 , . . . , cM ],
where c0 represents the background, and [c1 , . . . , cM ] corresponds to M body parts. PPR adopts a standard MoE router
structure, consisting of a fully-connected layer with parameters
θ ∈ R(M +1)×D followed by a softmax layer, which produces
the probabilities P ∈ RN ×(M +1) of each patch belonging to
the background and M body parts:
P = {pi ∈ RN | i = 0, 1, ..., M } = Sof tmax(Z · θT ), (2)

where pi represents the probabilities of patches embeddings
Z belonging to part ci .
Next, the router needs to assign patches to each expert based
on P. Traditional MoE patch-level routers [13] commonly
use the Top-k mechanism, where each expert is assigned k
patches (k ≪ N ). However, this fixed Top-k mechanism
is unsuitable for person ReID due to the varying sizes of
body parts. If k is too small, larger body regions (e.g., the
torso) may not be fully covered, leading to information loss.
Conversely, if k is too large, smaller body parts (e.g., the
head) may be polluted by surrounding background/occlusion
patches, introducing noise. To address this, our PPR employs
a soft routing mechanism. First, we aggregate M body parts
probabilities [p1 , . . . , pM ] to compute a foreground probability
pf ∈ RN : pjf = sum(pj1 , . . . , pjM ), where pji represents the
probability of patch embedding zj belonging to body part
ci . We then perform Probability Weighted Average Pooling
(PWAP) on patch embeddings Z to obtain a foreground
feature rf , a background feature r0 and M body part features
[r1 , . . . , rM ]:
PN
j
j=1 zj · pi
(3)
ri = PN j , ∀i ∈ {f, 0, 1, ..., M }
j=1 pi
where zj denotes the j-th patch embedding from the coarse
inference stage.
Supervision Loss of PPR. We introduce human topology
priors in the form of coarse body part labels. For each patch
j ∈ RN , its part label yj ∈ {0, 1, ..., M }. The cross-entropy
loss Lh with label smoothing is computed as:
Lh = −

M
X N
X

qi · log(pji ),

i=0 j=1

with qi =

(
1 − B−1
B ε
ε
B

(4)
if yj = i
otherwise

where B is batch size, ε is label smoothing regularization rate.
Additionally, we propose a Push Loss Lpush to separate the
body regions from background/occlusion noise:
E
D
t
t
B
r
,
r
X
0 f
1
,
(5)
Lpush =
B t=1 ||r0t || · ||rft ||
t
where r0|f
is the background/foreground feature of the t-th
image in a batch.
Finally, we follow the Global-identity Local-triplet (GiLt)
[7] strategy to supervise the training of PPR:

LP P R = λh Lh + Lpush + Lce (rf ) + Lce (rc )
+ Lpart
tri (r1 , . . . , rM ),

(6)

where rc = Concat(r1 , . . . , rM ), Lce is cross-entrophy loss,
Lpart
tri is part-averaged triplet loss [7].
2) Part Experts. Although Transformer excels at capturing
global information, it is less effective at capturing fine-grained
image details. To compensate for this limitation, we assign a

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.

III. E XPERIMENTS
A. Datasets and Evaluation Metrics

Fig. 3. The structure of Part Expert Module.

dedicated part-expert module to each body part, enabling deep
learning of its distinctive discriminative features. Futhermore,
we use pedestrian foreground features rf as coarse-grained
cues to provide contextual support for part feature learning.
The structure of PE is shown in Fig. 3. We first implement
1D convolution on body part features ri to obtain r̃i . Then
we apply the multi-head cross-attention (MHCA) mechanism,
where the query matrix Qi is derived from r̃i , and the key
matrix K and value matrix V are derived from the foreground
features rf . The MHCA is computed as:
MHCA(Qi , K, V ) = Concat(head1 , . . . , headh )W O , (7)
headl = Attention(Qli , K l , V l ),

p 
Attention(Q, K, V ) = Sof tmax QK T / dk V ,

(8)
(9)

where headl is the l-th head output, W O is the output transformation matrix for integrating multi-head outputs. Finally,
following layer normalization and two fully-connected layers,
we can obtain the new part feature ei of body part ci .
Additionally, to further reduce inference computational
costs and address occlusion effectively, PE is activated
based on body part visibility during inference, as shown in
Fig. 1. We generate binary visibility routing weights W =
[w1 , . . . , wM ] ∈ RM from the probabilities P. For body part
ci , if at least one patch in pi has a probability greater than
the threshold λ, wi = 1, the corresponding PE is activated;
otherwise, wi = 0, PE remains inactive:
(
1, if max(pni ) > λ
n
(10)
wi =
0, otherwise.
Supervision Loss of PE. We adopt ID loss and part-average
triplet loss to supervise PE:
LP E =

M
X

LID (ei ) + Lpart
tri (e1 , . . . , eM ),

(11)

i=1

C. Loss Functions
In the training stage, we set the early-exit threshold α = 1,
ensuring all images pass through both coarse and fine inference
stages for joint training. In the fine inference stage, all PE
are activated to ensure comprehensive training of each expert
module. The total training loss L is calculated as:
L = LCOARSE + LF IN E
LF IN E = LP P R + LP E

(12)

We evaluate our method on three ReID benchmarks, including one occluded and two holistic person ReID benchmarks.
Occluded-Duke [2] consists of 15,618 training images
from 702 identities, 2,210 occluded query images from 519
identities, and 17,661 gallery images from 1,110 identities.
Market-1501 [8] contains 36,036 images from 1,501 identities captured from 6 camera viewpoints.
DukeMTMC-ReID [9] comprises 36,411 images from
1,404 identities captured from 8 camera viewpoints.
Evaluation Metrics. We adopt the Cumulative Matching
Characteristic (CMC) curve and mean Average Precision
(mAP) to evaluate the performance of ReID methods. To access the efficiency of our CFPER, we report FLOPs calculated
with the fvcore toolkit.
B. Implementation Details
We adopt the ViT-Base [11] as the backbone. Both training
and testing images are resized to 256×128. The training images
are augmented with random cropping, padding, and random
erasing. The batch size is set to 64 with 4 images per ID.
The hidden dimension D is set to 768. The SGD optimizer
is employed with a momentum of 0.9 and a weight decay
of 1e-4. The learning rate is initialized at 0.008 with cosine
learning rate decay. The number of body parts M is set to
13. The threshold λ is empirically set to 0.4. We train our
model for 300 epochs. For a fair comparison, no re-ranking
techniques are used during inference.
C. Analysis of Trade-Off between Performance and Efficiency
The early-exit threshold α is an important factor that
balances model performance and inference efficiency in our
methods. The CFDR uses the early-exit threshold α to assess
the query difficulty and control the number of queries proceed
to the fine inference stage. A larger α imposes stricter criteria
for identifying queries as “easy”, leading to more queries
entering the fine stage. We conduct experiments with different
α on Occluded-Duke, using Rank-1 and mAP as performance
metrics and FLOPs in the fine inference stage as efficiency
metric. The results are shown in Table II. As α decreases,
FLOPs decrease due to fewer queries entering the fine stage,
demonstrating significant computational savings. Rank-1 accuracy initially increases, indicating that global features can
prevent mismatches since not all part features are discriminative, but then decreases as more “hard” queries requiring
fine-grained part features are missed. The decrease in mAP
shows that fine-grained part features are crucial for detailed
matching. The above results indicate that our CFDR enables
adaptive allocation of computational resources between “easy”
and “hard” samples, reducing unnecessary computation while
maintaining performance.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.

TABLE II
T HE I MPACT OF E ARLY-E XIT T HRESHOLD α ON THE T RADE - OFF
BETWEEN P ERFORMANCE AND E FFICIENCY OF CFPER.
α
1
0.7
0.6
0.5

Rank-1 (%)
67.8
68.5
68.5
65.9

mAP (%)
58.6
58.4
58.4
57.5

FLOPs (G)
179.05
153.93 (↓14%)
121.53 (↓32%)
102.49 (↓42%)

TABLE III
C OMPARISON WITH SOTA M ETHODS ON O CCLUDED (O CCLUDED -D UKE )
AND H OLISTIC (M ARKET-1501 AND D UKE MTMC) R E ID DATASETS .
T HE S YMBOL † D ENOTES THE M ETHODS INTRODUCING E XTERNAL
M ODELS AND ∗ D ENOTES H YBRID C NN -T RANSFORMER METHODS .
Backbone

CNN

Transformer

Methods
PGFA† (ICCV19) [2]
HOReID† (CVPR20) [3]
IGOAS (TIP21) [14]
BPBReID (WACV23) [7]
RTGAT (TIP23) [15]
GPEOG† (ICME23) [16]
PAT∗ (CVPR21) [17]
TransReID (ICCV21) [4]
FED (CVPR22) [10]
PFD† (AAAI22) [5]
DRL-Net∗ (TMM23) [6]
SCAT (TII23) [18]
SPT (AAAI24) [19]
CFPER (α = 1)
CFPER (α = 0.6)

Occluded-Duke
Rank-1
mAP
51.4
37.3
55.1
43.8
60.1
49.4
66.7
54.1
61.0
50.1
64.1
51.2
64.5
53.6
64.2
55.7
68.1
56.4
67.7
60.1
65.0
50.8
62.8
54.9
68.6
57.4
67.8
58.6
68.5
58.4

Market-1501
Rank-1
mAP
91.2
76.8
94.2
84.9
93.4
84.1
95.1
87.0
93.3
85.1
94.8
87.5
95.4
88.0
95.0
88.2
95.0
86.3
95.5
89.6
94.7
86.9
95.1
88.0
94.5
86.2
95.1
88.7
95.6
88.4

DukeMTMC
Rank-1 mAP
82.6
65.5
86.9
75.6
86.9
75.1
89.6
78.3
88.0
76.9
87.5
75.5
88.8
78.2
89.6
80.6
89.4
78.0
90.6
82.2
88.1
76.6
89.3
79.8
89.4
79.1
90.3
80.7
90.5
80.2

TABLE IV
A BLATION S TUDY ON O CCLUDED -D UKE .
Index
1
2
3
4

PPR
✓
✓
✓

PE
✓
✓

CFDR
✓

Rank-1
60.5
63
67.8
68.5

mAP
53.1
55.4
58.6
58.4

D. Comparison with the State-of-the-Art Methods
We compare our method with SOTA methods on three
benchmarks, covering both occluded and holistic person ReID
scenarios. The comparison includes two kinds of methods:
CNN-based and Transformer-based. Notably, PGFA, HOReID,
GPEOG and PFD incorporate external models for part feature
extraction, while PAT and DRL-Net integrate CNNs within
their Transformer architectures. The comparison results are
shown in Table III. On the challenging Occluded-Duke dataset,
our CFPER (α = 0.6) achieves comparable performance with
a Rank-1 accuracy of 68.5% and mAP of 58.4%, outperforming most of the compared methods on both metrics without
relying on external models. This demonstrates its robustness
in handling occluded scenarios. On the holistic ReID datasets,
CFPER (α = 0.6) achieves the highest Rank-1 accuracy
on Market-1501 and the second-highest Rank-1 accuracy on
DukeMTMC, falling behind PFD by only 0.1%. Additionally,
CFPER achieves competitive mAP on both datasets. These
results validate CFPER’s ability to adaptively handle queries
with varying difficulty, ensuring both efficiency and accuracy.
E. Ablation Study
In this section, we conduct ablation studies on OccludedDuke to analyze the effectiveness of components of CFPER.

TABLE V
C OMPARISON OF C OMPUTATION B ETWEEN PPR AND EXTERNAL MODELS .
Model
PPR(ours)
OpenPose
HRNet32

FLOPs (G)
0.0016
37.80
17.86

Parameters (M )
0.0123
25.94
41.23

Effectiveness of proposed Components. The results are
shown in Table IV. Index-1 represents the baseline model,
which is a standard ViT. Index-2 shows that PPR provides
+2.5% Rank-1 accuracy and +2.3% mAP improvements,
demonstrating the effectiveness of combining human topology priors and identity labels for part localization. Index-3
shows that PE further improve performance by +4.8% Rank-1
accuracy and +3.2% mAP, indicating the benefit of specialized
networks for fine-grained feature learning. Comparing Index-3
and Index-4, the CFDR mechanism optimizes computational
efficiency while maintaining performance.
Comparison of Computation between PPR and external
models. We compare the FLOPs and model parameters of our
PPR with external models commonly used in person ReID for
part feature extraction, including HRNet [20] and OpenPose
[21]. As shown in Table V, PPR significantly outperforms both
OpenPose and HRNet32 in computational efficiency, with only
0.0016G FLOPs, dramatically lower than OpenPose (37.80G)
and HRNet32 (17.86G), demonstrating PPR’s substantial advantage in reducing computational cost. Additionally, PPR has
just 0.0123M parameters, greatly reducing model complexity
compared to OpenPose (25.94M) and HRNet32 (41.23M).
These results demonstrate that our PPR can effectively identify
key body parts using a simple network without adding extra
computation, highlighting its efficiency and practicality.
F. Visualization
In this section, we perform visualizations on Occluded-Duke
dataset to demonstrate the effectiveness of our CFPER.
Visualization of the “Easy” and “Hard” query. Fig. 4
shows “easy” and “hard” queries identified by CFPER, validating the soundness of our early-exit strategy in the CFDR
mechanism. “Easy” queries retain visible discriminative features despite partial occlusion, while “hard” queries involve
severe occlusion and fewer discriminative features, making it
difficult to distinguish the target pedestrian from others with
similar appearance. Fig. 5 shows the Top-10 retrieval results
for “easy” and “hard” queries in the coarse and fine inference
stages. For “easy” queries, global features alone are sufficient
for accurate matching in the coarse stage. In contrast, “hard”
queries benefit from fine-grained part features in the fine stage,
improving retrieval accuracy by compensating for occlusion
noise in global features.
Visualization of Body Part Attention Maps for PPR.
Fig. 6 shows the body part localization results of PPR.
Our method accurately identifies body parts and effectively
addresses occlusions and background noise. Furthermore, the
Part Experts for occluded parts remain inactive, allowing the
model to focus on visible discriminative features while further
reducing computational costs during inference.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.

Fig. 4. Visualization of the “easy” and “hard” query images.

Fig. 5. Top-10 Retrieval results of “easy” and “hard” queries. Green/red
borders illustrate correct/false matches. For each query, the first row displays
coarse inference results, the second row shows fine inference results.

Fig. 6. Visualization of body part attention maps for PPR. The second column
denotes foreground attention maps. Green/red border indicates active/inactive
PE during inference.

IV. C ONCLUSION
In this paper, we propose a novel model CFPER to balance
inference efficiency and retrieval accuracy for person ReID.
By dynamically adjusting the retrieval process based on query
difficulty and tailoring the learning of visible body part representations, CFPER achieves competitive performance while
significantly reducing computational costs. Extensive experiments demonstrate the effectiveness of CFPER.
R EFERENCES
[1] Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, and Shengjin Wang, “Beyond
part models: Person retrieval with refined part pooling (and a strong
convolutional baseline),” in Proceedings of the European conference on
computer vision (ECCV), 2018, pp. 480–496.
[2] Jiaxu Miao, Yu Wu, Ping Liu, Yuhang Ding, and Yi Yang, “Pose-guided
feature alignment for occluded person re-identification,” in Proceedings
of the IEEE/CVF international conference on computer vision, 2019,
pp. 542–551.

[3] Guan’an Wang, Shuo Yang, Huanyu Liu, Zhicheng Wang, Yang Yang,
Shuliang Wang, Gang Yu, Erjin Zhou, and Jian Sun, “High-order
information matters: Learning relation and topology for occluded person
re-identification,” in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 2020, pp. 6449–6458.
[4] Shuting He, Hao Luo, Pichao Wang, Fan Wang, Hao Li, and Wei Jiang,
“Transreid: Transformer-based object re-identification,” in Proceedings
of the IEEE/CVF international conference on computer vision, 2021,
pp. 15013–15022.
[5] Tao Wang, Hong Liu, Pinhao Song, Tianyu Guo, and Wei Shi, “Poseguided feature disentangling for occluded person re-identification based
on transformer,” in Proceedings of the AAAI conference on artificial
intelligence, 2022, vol. 36, pp. 2540–2549.
[6] Mengxi Jia, Xinhua Cheng, Shijian Lu, and Jian Zhang, “Learning disentangled representation implicitly via transformer for occluded person
re-identification,” IEEE Transactions on Multimedia, vol. 25, pp. 1294–
1305, 2022.
[7] Vladimir Somers, Christophe De Vleeschouwer, and Alexandre Alahi,
“Body part-based representation learning for occluded person reidentification,” in Proceedings of the IEEE/CVF winter conference on
applications of computer vision, 2023, pp. 1613–1623.
[8] Liang Zheng, Liyue Shen, Lu Tian, Shengjin Wang, Jingdong Wang,
and Qi Tian, “Scalable person re-identification: A benchmark,” in
Proceedings of the IEEE international conference on computer vision,
2015, pp. 1116–1124.
[9] Zhedong Zheng, Liang Zheng, and Yi Yang, “Unlabeled samples
generated by gan improve the person re-identification baseline in vitro,”
in Proceedings of the IEEE international conference on computer vision,
2017, pp. 3754–3762.
[10] Zhikang Wang, Feng Zhu, Shixiang Tang, Rui Zhao, Lihuo He, and
Jiangning Song, “Feature erasing and diffusion network for occluded
person re-identification,” in Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, 2022, pp. 4754–4763.
[11] Alexey Dosovitskiy, “An image is worth 16x16 words: Transformers for
image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.
[12] Yizeng Han, Gao Huang, Shiji Song, Le Yang, Honghui Wang, and
Yulin Wang, “Dynamic neural networks: A survey,” IEEE Transactions
on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 7436–
7456, 2021.
[13] Mohammed Nowaz Rabbani Chowdhury, Shuai Zhang, et al., “Patchlevel routing in mixture-of-experts is provably sample-efficient for
convolutional neural networks,” in International Conference on Machine
Learning. PMLR, 2023, pp. 6074–6114.
[14] Cairong Zhao, Xinbi Lv, et al., “Incremental generative occlusion
adversarial suppression network for person reid,” IEEE Transactions
on Image Processing, vol. 30, pp. 4212–4224, 2021.
[15] Meiyan Huang, Chunping Hou, Qingyuan Yang, and Zhipeng Wang,
“Reasoning and tuning: Graph attention network for occluded person
re-identification,” IEEE Transactions on Image Processing, vol. 32, pp.
1568–1582, 2023.
[16] Zhihao Li, Huaxiang Zhang, Lei Zhu, Jiande Sun, and Li Liu, “Effective
occlusion suppression network via grouped pose estimation for occluded
person re-identification,” in 2023 IEEE International Conference on
Multimedia and Expo (ICME). IEEE, 2023, pp. 2645–2650.
[17] Yulin Li, Jianfeng He, et al., “Diverse part discovery: Occluded person
re-identification with part-aware transformer,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2021,
pp. 2898–2907.
[18] Huijie Fan, Xiaotong Wang, Qiang Wang, Shengpeng Fu, and Yandong
Tang, “Skip connection aggregation transformer for occluded person
reidentification,” IEEE Transactions on Industrial Informatics, vol. 20,
no. 1, pp. 442–451, 2023.
[19] Lei Tan, Jiaer Xia, Wenfeng Liu, Pingyang Dai, Yongjian Wu, and
Liujuan Cao, “Occluded person re-identification via saliency-guided
patch transfer,” in Proceedings of the AAAI Conference on Artificial
Intelligence, 2024, vol. 38, pp. 5070–5078.
[20] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang, “Deep high-resolution
representation learning for human pose estimation,” in Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
2019, pp. 5693–5703.
[21] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh, “Realtime
multi-person 2d pose estimation using part affinity fields,” in Proceedings of the IEEE conference on computer vision and pattern recognition,
2017, pp. 7291–7299.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:51:31 UTC from IEEE Xplore. Restrictions apply.


hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Computer Vision and Image Understanding 262 (2025) 104552

Contents lists available at ScienceDirect

Computer Vision and Image Understanding
journal homepage: www.elsevier.com/locate/cviu

Channel-aware feature mining network for Visible–Infrared Person
Re-identification
Pengxia Li, Zhonghao Du, Linhui Zhang, Yanyi Lv, Yujie Liu ∗
Qingdao Institute of Software, College of Computer Science and Technology, China University of Petroleum (East China), Shandong Key Laboratory of Intelligent
Oil and Gas Industrial Software, China

ARTICLE

INFO

Communicated by Shiliang Zhang
Keywords:
Visible Infrared Person Re-identification
Feature learning
Channel-level processing

ABSTRACT
Visible–Infrared Person Re-identification (VI-ReID) aims to match the identities of pedestrians captured
by non-overlapping cameras in both visible and infrared modalities. The key to overcoming the VI-ReID
challenge lies in extracting diverse modality-shared features. Current methods mainly focus on channellevel operations during data preprocessing, with the aim of expanding the dataset. However, these methods
often overlook the complex relationships among channel features, leading to insufficient utilization of unique
information in each channel. To address this issue, we propose the Channel-Aware Feature Mining Network
(CAFMNet) to improve VI-ReID effectiveness. Specifically, we design three core modules: a Channel-Level
Feature Optimization (CLFO) module, which captures channel-level key features for identity recognition and
directly extracts identity-relevant information at the channel level; a Channel-Level Feature Refinement (CLFR)
module, which enhances channel-level features while retaining useful information—addressing the irrelevant
content in initially extracted features; a Multi-Dimensional Feature Optimization (MDFO) module, which
comprehensively processes multi-dimensional feature information to enhance the model’s ability to understand
and describe input data. Extensive experiments on the SYSU-MM01 and LLCM datasets demonstrate that
our CAFMNet outperforms existing approaches in terms of VI-ReID effectiveness. The code is available at
https://github.com/cobeibei/CAFMNet-1.

1. Introduction
Person re-identification (Re-ID) is a core technology in intelligent
video surveillance systems, as it enables the retrieval of images or
video sequences of specific pedestrians from multi-view data captured
by non-overlapping cameras. In practical urban security scenarios, the
demand for 24-hour continuous surveillance is increasing. However,
traditional Re-ID methods relying solely on RGB images suffer from
significant performance degradation in low-light, nighttime, or backlit
environments. Infrared (IR) cameras can effectively overcome lighting
limitations by capturing thermal radiation from objects, which has
attracted widespread attention among researchers to visible–infrared
person re-identification (VI-ReID).
The core goal of VI-ReID is to associate RGB and IR images of
the same pedestrian in camera data. Nevertheless, the inherent differences between the RGB and IR imaging mechanisms pose enormous
challenges to this task: RGB images contain rich color and texture information, but are limited in use under poor lighting conditions; IR images
reflect thermal radiation information, but have weak texture details and
lack color cues. These differences result in a prominent ‘‘modality gap’’:

Features extracted from the two modalities often exhibit drastically
different statistical distributions, making direct cross-modal matching
extremely difficult. Therefore, the core challenge of VI-ReID lies in
extracting modality-shared, identity-discriminative features that are
robust to cross-modal differences.
Recent advances in deep learning have demonstrated that finegrained feature extraction — particularly channel-level feature extraction — is crucial for enhancing the performance of Re-ID models. In
VI-ReID scenarios, channel-level features play a key role in capturing
identity-discriminative patterns that are invariant across modalities.
For instance, specific channels of RGB images may emphasize clothing
textures, while the single channel of IR images tends to focus on
presenting pedestrian contours. Despite this understanding, existing VIReID methods still have obvious shortcomings in effectively utilizing
channel-level features, and this issue has become a bottleneck hindering
further performance improvement.
Existing VI-ReID methods primarily address modality differences
through two strategies: auxiliary information-based methods and feature learning-based methods. Regrettably, neither strategy fully taps

∗ Corresponding author.

E-mail address: liuyujie@upc.edu.cn (Y. Liu).
https://doi.org/10.1016/j.cviu.2025.104552
Received 17 June 2025; Received in revised form 16 September 2025; Accepted 19 October 2025
Available online 24 October 2025
1077-3142/© 2025 Elsevier Inc. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

Feature Mining Network (CAFMNet). CAFMNet consists of three core
modules: the Channel-Level Feature Optimization (CLFO) module, the
Channel-Level Feature Refinement (CLFR) module, and the MultiDimensional Feature Optimization (MDFO) module. These modules
work collaboratively to overcome the limitations of existing methods. The CLFO module directly extracts useful feature information
for identity recognition at the channel level. Considering that the
initially extracted features may contain some irrelevant information,
we design a Channel-Level Feature Refinement (CLFR) module to
further enhance channel-level features, thereby achieving the goal of
strengthening these features while suppressing irrelevant information.
Finally, to enhance the model’s ability to understand and represent input data, we use the Multi-Dimensional Feature Optimization (MDFO)
module to comprehensively process feature information from multiple dimensions, which improves feature quality and strengthens key
features.
Our primary contributions are summarized as follows:
⋅ We propose a novel deep learning framework, the Channel-Aware
Feature Mining Network (CAFMNet), which focuses on accurately capturing the channel-level features most critical to identification for
efficient visible–infrared person re-identification (VI-ReID).
⋅ We propose a novel Channel-Level Feature Optimization (CLFO)
module that extracts identification-relevant features at the channel
level. Additionally, we introduce a Channel-Level Feature Refinement
(CLFR) module that captures critical discriminative information while
suppressing irrelevant data, thereby enhancing channel-level features.
⋅ We propose the Multi-Dimensional Feature Optimization (MDFO)
module to process feature information from various dimensions, enhancing critical features and improving the model’s understanding of
input data.
⋅ Experimental results on the SYSU-MM01 and LLCM datasets show
that our approach achieves a new state-of-the-art.

Fig. 1. The different channel activations of the feature maps are shown, with
each sub-map corresponding to a different channel. We randomly selected
a pedestrian image from the SYSU-MM01 dataset to visualize the different
channel activations of its original feature map.

into the potential of channel-level features. Auxiliary information-based
methods attempt to build a bridge between RGB and IR modalities by
introducing additional auxiliary data. Although they have contributed
to the field, they have inherent flaws: generating auxiliary information
typically requires extra processing steps, which increases computational
burden; some auxiliary data may also need manual annotation, further
raising data costs. More critically, these methods do not directly model
the channel-level features of raw RGB/IR data; instead, they rely on
auxiliary modalities to indirectly alleviate cross-modal differences, thus
missing the opportunity to capture unique discriminative cues in each
channel of the target modalities.
In contrast, feature learning-based methods aim to extract meaningful feature representations directly from raw multi-modal data without
relying on external auxiliary information. Their core goal is to reduce
cross-modal differences through techniques such as pixel-level feature
alignment and mapping multi-modal features to a shared feature space.
Among them, pixel-level feature alignment methods are extremely
sensitive to image noise and subtle color changes, which may damage
channel-level semantic information; methods that map multi-modal
features to a shared feature space ignore the ‘‘channel imbalance’’
phenomenon—different channels within a single modality (e.g., the
R, G, B channels of RGB images) and across modalities (e.g., RGB
channels vs. IR channels) contribute differently to identity recognition. Additionally, some feature learning-based methods only perform
channel-level operations during the data preprocessing stage, with the
sole purpose of expanding dataset scale. This approach overlooks the
valuable discriminative information inherent in each channel.
Through detailed feature map visualization (as shown in Fig. 1), it
can be observed that some channels of RGB/IR images contain unique
and information-rich identity cues (e.g., some channels highlight the
clothing texture or thermal contour of pedestrians); in contrast, other
channels contribute little to recognition and may even introduce noise.
This channel imbalance phenomenon, combined with the failure of
existing methods to explicitly model and refine channel-level features,
results in poor feature learning performance in VI-ReID tasks and limits
model performance. To address these urgent issues awaiting breakthroughs, we propose a novel Channel-Aware Feature Mining Network
(CAFMNet), which is specifically designed to extract and optimize
channel-level features crucial for VI-ReID.
In this paper, to tackle this challenge, we propose a new method for
extracting complex channel-level features, called the Channel-Aware

2. Related work
Visible–infrared person re-identification (VI-ReID) addresses modality discrepancies through two methods: using auxiliary information and
feature learning.
2.1. Using auxiliary information
In Visible–Infrared Person Re-identification (VI-ReID), the method
based on auxiliary information aims to effectively alleviate cross-modal
differences by introducing additional auxiliary modalities, which play
a crucial role as a bridge connecting the visible and infrared modalities. Common auxiliary modalities in this task include other generated
images, grayscale-transformed images, contour information, skeleton
information, and textual descriptions.
Align-GAN (Wang et al., 2019) uses GANs to convert RGB images to infrared, while AGPI2 (Alehdaghi et al., 2023) employs GANs
to create virtual intermediate images for improving RGB-infrared reidentification. Due to the absence of cross-modal pairs, these methods
may generate noisy images. CycleGAN (Zhu et al., 2017), which enables unsupervised image transformation without paired data, has been
applied in VI-ReID: for example, JSIA-ReID (Yang et al., 2020) uses
CycleGAN to generate cross-modal paired images from unpaired sets,
and another study (Xia et al., 2021) introduces a CycleGAN-based
IMT network to synthesize cross-modal images and expand the dataset.
However, CycleGAN-generated images may still have flaws such as
missing details, inaccurate colors, or unnatural textures, which affect
model performance.
In addition to GAN-generated images, other auxiliary modalities are
also widely used in VI-ReID: PMT (Lu et al., 2023), WF-CAMReVi (Sarker
and Zhao, 2024), and HAT (Ye et al., 2020b) utilize grayscale images to
help models learn modality-invariant features and minimize differences
between RGB and infrared images; SPOT (Chen et al., 2022) enhances
2

P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

recognition through pose structure information; STAR (Jiang et al.,
2024) improves video-based VI-ReID accuracy by leveraging skeleton
data; CGMMNet (Xu and Zhao, 2024) addresses color discrepancies and
blurred boundaries using intermediate modality images and contour
maps; lastly, YYDS (Du et al., 2024) boosts re-identification performance by incorporating rough textual descriptions to fill in missing
color information in infrared images. Overall, using generated images as auxiliary information typically involves Generative Adversarial
Networks (GANs) for modality transformations (e.g., converting RGB
images to infrared images).
The introduction of auxiliary information can provide additional
contextual semantic information to compensate for modality differences, which indeed contributes significantly to improving model performance. However, this approach has inherent limitations: the generation of auxiliary information requires additional image processing
or natural language processing steps, which increases computational
burden; some auxiliary information also needs manual annotation,
raising data costs; furthermore, auxiliary information may introduce
redundant information or noise. In addition, modality differences between auxiliary information and target modalities lead to consistency
issues, which require additional alignment and fusion strategies.

channels of an RGB image with a single channel" as auxiliary modalities
to reduce modality differences. However, both approaches only use
channel operations for data expansion or modality adaptation, without
deeply exploring the identity-discriminative information contained in
individual channels. CHCR (Pang et al., 2023) provides a new perspective for channel-level processing: its inter-channel pseudo-label
refinement method, based on the principle that the three RGB channels
of the same sample correspond to the same identity, performs crossmodal clustering on each of the three channels with the infrared
modality separately. It evaluates consistency using the Intersection
over Union (IoU) to eliminate unreliable pseudo-labels, which not
only mitigates the information loss caused by traditional single-channel
conversion but also verifies the performance gain brought by channellevel features. Our visualization experiments (as shown in Fig. 1)
also confirm that some channels indeed contain highly discriminative
features crucial for identity recognition. Therefore, more efforts should
be devoted to exploring and utilizing channel-level features to enhance
the model’s representational capability and cross-modal recognition
accuracy.

2.2. Feature learning

The network (as shown in Fig. 2) adopts a dual-stream ResNet50 (He et al., 2016; Ye et al., 2020a) architecture to separately process
features from RGB and infrared (IR) images, which effectively addresses
the matching challenges in visible–infrared person re-identification (VIReID) caused by modality differences. First, the input visible–infrared
(VIS-IR) features pass through the Channel-Level Feature Optimization (CLFO) module, which directly extracts channel-level key features
closely related to identity recognition. To further improve the quality
of these channel features, we design the Channel-Level Feature Refinement (CLFR) module to suppress redundant or irrelevant information
and enhance discriminative features, thereby improving the accuracy
and robustness of feature representation.
On this basis, to enhance the model’s ability to understand and
describe input data, we introduce the Multi-Dimensional Feature Optimization (MDFO) module, which further explores and integrates feature information across multiple dimensions and layers. Through the
sequential processing of these three modules, the network can extract
richer and highly relevant key features from the original multi-modal
data, significantly strengthening the model’s discriminative ability.

3. Methodology

Feature learning methods aim to extract and learn meaningful feature representations directly from raw multi-modal data, rather than
relying on image transformations or additional auxiliary information.
Their core objective is to reduce discrepancies between different modalities through specific techniques — such as aligning features at the
pixel level or mapping multi-modal features directly into a shared
feature space — thereby improving the model’s generalization ability
and recognition accuracy. This approach emphasizes enhancing the
model’s understanding and processing capabilities for multi-source data
without introducing external information.
Pixel-level feature alignment methods operate directly on each pixel
in the image. For example, SAAI (Fang et al., 2023) achieves aggregation of potential semantic partial features by calculating the similarity
between pixel-level features and learnable prototypes; DCLNet (Sun
et al., 2022) proposes a dense contrastive learning network to perform
pixel-to-pixel dense alignment; CSL (Nie et al., 2024) designs a pixellevel color transformation module to learn the relationships between
different color channels. However, since these methods operate directly
at the pixel level, they are highly sensitive to image noise or subtle color
variations—this can significantly affect the model’s feature extraction
and recognition performance.
Another category of methods aims to project multi-modal features
into a shared feature space to learn a unified cross-modal representation. For example, MAUM (Liu et al., 2022) designs a one-way metric
learning approach that enhances memory capability by learning crossmodal metrics in two directions; RFM (Tan et al., 2023) introduces
a cross-modal center loss at the feature level to explore more compact intra-class distributions and employs a modality-aware spatial
attention module to better exploit texture regions. However, due to
significant differences between RGB and infrared images in information
capacity, representation, sharpness, and lighting conditions, simply
mapping them into the same feature space is insufficient to fully
eliminate modality gaps. This may also lead to the loss of important modality-specific information, which negatively impacts overall
recognition performance.
In addition, channel-level processing deserves attention in visible–
infrared person re-identification (VI-ReID), yet its significant value
remains underutilized. Current works mostly focus on preprocessing:
for example, Yang et al. (2022b), Wu and Ye (2023), Teng et al.
(2024), Dai et al. (2024), Zhang et al. (2024) generate color-invariant
images through random channel enhancement to expand the dataset;
CAJ (Ye et al., 2021a) uses images generated by "replacing the three

3.1. Channel-level feature optimization module
In the visible–infrared person re-identification (VI-ReID) task, RGB
and infrared (IR) images exhibit significant differences in channel
distribution due to their distinct imaging principles. For example, IR
images typically contain only a single thermal channel, while RGB images have three color channels. This modality asymmetry means certain
channels may carry stronger identity-discriminative information, while
others could be redundant or noisy.
To address this issue, we propose the Channel-Level Feature Optimization (CLFO) module. Its goal is to enhance the model’s ability
to extract discriminative identity features across modalities through
multi-level channel modeling and dynamic feature refinement. Unlike
traditional attention mechanisms (e.g., SE, CBAM) that focus solely on
channel importance estimation, CLFO integrates depthwise separable
convolution, group normalization, and a learnable residual connection
into a unified framework. These components work together to achieve
fine-grained channel-level feature modeling at the early stage of feature extraction, effectively mitigating the impact of channel imbalance
between RGB and IR images.
This design allows CLFO to not only adaptively highlight informative channels but also maintain computational efficiency and training
stability—key requirements for VI-ReID tasks. In what follows, we
3

P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

Fig. 2. The architecture of the proposed Channel-Aware Feature Mining Network (CAFMNet) and the Multi-Dimensional Feature Optimization (MDFO) module
is described in this section. Some other module architectures are shown in Fig. 3.

independently for each input channel (groups = 𝐶in ) to avoid channel
information mixing; (2) Pointwise convolution: Employs a 1 × 1 kernel to
fuse cross-channel information, ensuring the output channel dimension
𝐶out matches 𝐶in for subsequent residual connection.
After DSConv1 , group normalization (GN, denoted as GN1 , with
8 groups to stabilize training across modalities) and ReLU activation
are applied. The resulting intermediate feature tensor 𝐴1 serves as a
core link in the CLFO module: it locates at the frontend of the CLFO
feature processing pipeline, bridging the raw backbone features (𝑋)
and subsequent refined operations (e.g., secondary depthwise separable
convolution, SEBlock).
𝐴1 retains the spatial resolution of 𝑋 (due to stride=1 and padding=1)
while enhancing local discriminative patterns (e.g., pedestrian contours, clothing textures) that are invariant to RGB-IR modality gaps.
Its mathematical definition is:
(
(
))
𝐴1 = ReLU GN1 DSConv1 (𝑋)
(1)
where 𝐴1 ∈ R𝐵×𝐶out ×𝐻×𝑊 (with 𝐶out = 𝐶in ) to ensure dimension
consistency for subsequent feature fusion.
To recalibrate channel-wise importance and suppress modalityspecific noise (e.g., thermal artifacts in IR images), the feature tensor 𝐴1 is further processed by a second depthwise separable convolution (DSConv2 ) and group normalization (GN2 ), followed by a
Squeeze-and-Excitation (SE) Block:
(
(
))
𝑍2′′ = SE GN2 DSConv2 (𝐴1 )
(2)

Fig. 3. The specific design of the (Fig. 3 left (a)) channel-level feature
optimization module as well as the (Fig. 3 right (b)) channel-level feature
refinement module.

where the SE block implements global channel attention via adaptive
average pooling (to 1 × 1 spatial resolution) and two fully connected
layers (reduction ratio=16, consistent with the module’s hyperparameters).
Finally, a learnable residual connection fuses 𝑍2′′ with the transformed input feature Res(𝑋) (via 1 × 1 convolution and GN), and ReLU
activation yields the CLFO module output 𝑂:
(
)
𝑂 = ReLU 𝛼 ⋅ 𝑍2′′ + (1 − 𝛼) ⋅ Res(𝑋)
(3)

describe the detailed architecture and mathematical formulation of the
CLFO module.
Let the input feature tensor of the Channel-Level Feature Optimization (CLFO) module be 𝑋 ∈ R𝐵×𝐶in ×𝐻×𝑊 , where: 𝐵 denotes the batch
size of input multi-modal (RGB/IR) features; 𝐶in represents the number
of input channels, consistent with the output channel dimension of
the dual-stream ResNet-50 backbone (adopted in the CAFMNet architecture); 𝐻 × 𝑊 denotes the spatial resolution of the feature maps
(e.g., 56 × 56 for intermediate features in ResNet-50).
To extract modality-invariant features with low computational cost,
we first apply a depthwise separable convolution (DSConv1 ) operation,
which consists of two sequential steps: (1) Depthwise convolution: Uses
a 3 × 3 kernel, stride=1, and padding=1, performing spatial filtering

Here, 𝛼 ∈ [0, 1] is a learnable parameter (initialized to 0.5) that
dynamically balances refined features and raw input information, enhancing the robustness of cross-modal feature representation.
Unlike traditional attention modules that treat all channels equally,
CLFO explicitly models the interaction between modality-specific characteristics and channel-wise importance. By performing channel-aware
4

P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

feature optimization at an early stage of feature extraction, CLFO effectively mitigates the impact of modality-specific noise and imbalanced
channel information, thereby enhancing the consistency of cross-modal
feature representations—crucial for accurate matching in VI-ReID.

original 𝑂𝑐𝑎 to the reconstructed feature, generating the local–global
fused feature map 𝑂𝑙𝑔 ∈ R𝐵×𝐶×𝐻×𝑊 :
(
)
𝑂𝑙𝑔 = 𝑊 𝑓norm ⋅ 𝑔(𝑂𝑐𝑎 ) + 𝑂𝑐𝑎
(5)
where + denotes element-wise addition (residual connection), ensuring
original local features are preserved while integrating global dependencies.
After local–global information fusion, a depthwise separable convolution block is employed to efficiently refine both spatial and channel
depth
features. It first applies a 3 × 3 depthwise convolution (Conv3×3 ,
groups = 𝐶) to 𝑂𝑙𝑔 , extracting spatial patterns without mixing channel
point
information. A 1 × 1 pointwise convolution (Conv1×1 ) then fuses crosschannel information and restores the channel dimension to 𝐶, resulting
in the spatial-refined feature map 𝑂𝑑𝑠 ∈ R𝐵×𝐶×𝐻×𝑊 . This lightweight
operation balances computational cost and feature expressiveness, with
the formulation:
(
)
point
depth
𝑂𝑑𝑠 = Conv1×1 Conv3×3 (𝑂𝑙𝑔 )
(6)

3.2. Channel-level feature refinement module
Although we use the Channel-Level Feature Optimization (CLFO)
module to directly mine identity-relevant information at the channel
level, the CLFO module only performs coarse-grained extraction of
channel-level features, which may retain irrelevant noise (e.g., background clutter or modality-specific artifacts). To further enhance the
discriminative capability of channel-wise features in complex crossmodal scenarios — such as visible–infrared person re-identification (VIReID), where RGB and infrared (IR) images exhibit significant differences in channel distribution — we design an additional Channel-Level
Feature Refinement (CLFR) module.
The CLFR module is specifically designed to suppress modalityspecific noise (e.g., thermal artifacts in IR images or color distortion in
low-light RGB images) while preserving identity-discriminative features
(e.g., pedestrian contours, clothing textures) across both RGB and IR
modalities. It achieves this through a multi-stage refinement process
involving enhanced channel attention, non-local feature fusion (for
local–global interaction), depthwise separable convolution, and spatial attention with residual learning—each stage explicitly addressing
challenges posed by modality asymmetry (e.g., single-channel IR vs.
three-channel RGB) and background clutter in ReID scenarios. Let the
input feature map of the CLFR module be 𝑂 ∈ R𝐵×𝐶×𝐻×𝑊 , where: 𝐵
denotes the batch size (number of sample pairs in one training batch);
𝐶 represents the number of feature channels, consistent with the output
channel dimension of the preceding CLFO module; 𝐻 × 𝑊 denotes the
spatial resolution of the feature map (height × width, e.g., 28 × 28 for
intermediate features in the dual-stream ResNet-50 backbone).
The CLFR module first applies an enhanced channel attention mechanism to aggregate global channel information and suppress redundant
channels. It uses adaptive average pooling (AvgPool1×1 ) to process 𝑂,
compressing the spatial dimensions of each channel into a 1×1 scalar to
capture global statistical information. Two successive 1×1 convolutions
(denoted Conv1×1 ) — with the first reducing the channel dimension
to 𝐶∕reduction (reduction ratio = 32) and ReLU activation, and the
second restoring the channel dimension to 𝐶 with Sigmoid activation
— generate channel attention weights 𝐶𝐴 ∈ R𝐵×𝐶×1×1 . Element-wise
multiplication of 𝐶𝐴 and 𝑂 yields the channel-refined feature map
𝑂𝑐𝑎 ∈ R𝐵×𝐶×𝐻×𝑊 , whose mathematical formulation is:
(
(
(
𝑂𝑐𝑎 = 𝑂 ⊙ 𝜎 Conv1×1 ReLU Conv1×1 (
(4)
))))
AvgPool1×1 (𝑂)

Next, to suppress background clutter (e.g., walls, vehicles) and
focus on identity regions, the CLFR module applies a spatial attention mechanism. It first performs max pooling (MaxPool) and average
pooling (AvgPool) along the channel dimension of 𝑂𝑑𝑠 , resulting in
MaxPool(𝑂𝑑𝑠 ) ∈ R𝐵×1×𝐻×𝑊 (extracting salient spatial regions) and
AvgPool(𝑂𝑑𝑠 ) ∈ R𝐵×1×𝐻×𝑊 (capturing global spatial context). These
two pooled features are concatenated to form sa_input ∈ R𝐵×2×𝐻×𝑊 ,
and a 7 × 7 convolution (Conv7×7 ) — followed by Sigmoid activation
— generates spatial attention weights 𝑆𝐴 ∈ R𝐵×1×𝐻×𝑊 . Element-wise
multiplication of 𝑆𝐴 and 𝑂𝑑𝑠 enhances identity-relevant spatial regions,
yielding the spatial-refined feature map 𝑂𝑠𝑎 ∈ R𝐵×𝐶×𝐻×𝑊 :
(
(
(
𝑂𝑠𝑎 = 𝑂𝑑𝑠 ⊙ 𝜎 Conv7×7 concatenate MaxPool(𝑂𝑑𝑠 ),
(7)
)))
AvgPool(𝑂𝑑𝑠 )
where concatenate(⋅, ⋅) denotes channel-wise concatenation.
Finally, to prevent overfitting (critical for limited VI-ReID datasets),
𝑂𝑠𝑎 undergoes Dropout regularization (Dropout, probability = 0.1),
randomly setting feature elements to 0 to enhance model robustness.
A residual connection adds the dropout-processed feature Dropout(𝑂𝑠𝑎 )
to the original input 𝑂 of the CLFR module, preserving original feature
information and integrating multi-stage refined features to generate the
final output feature map 𝑂′ ∈ R𝐵×𝐶×𝐻×𝑊 :
𝑂′ = Dropout(𝑂𝑠𝑎 ) + 𝑂

(8)

To summarize the entire computational process of the CLFR module,
we define stage-specific functions: 𝐹𝑐𝑎 (⋅) (enhanced channel attention,
outputting 𝑂𝑐𝑎 from 𝑂), 𝐹𝑛𝑙 (⋅) (non-local feature fusion, outputting
𝑂𝑙𝑔 from 𝑂𝑐𝑎 ), 𝐹𝑑𝑠 (⋅) (depthwise separable convolution, outputting 𝑂𝑑𝑠
from 𝑂𝑙𝑔 ), 𝐹𝑠𝑎 (⋅) (spatial attention, outputting 𝑂𝑠𝑎 from 𝑂𝑑𝑠 ), and
𝐹drop (⋅) (Dropout regularization, outputting Dropout(𝑂𝑠𝑎 ) from 𝑂𝑠𝑎 ).
The entire process can be expressed as a composite function:
(
(
( (
))))
𝑂′ = 𝐹drop 𝐹sa 𝐹ds 𝐹nl 𝐹𝑐𝑎 (𝑂)
+𝑂
(9)

where ⊙ denotes element-wise multiplication (broadcasting 𝐶𝐴 to
match the spatial dimensions of 𝑂), 𝜎(⋅) denotes the Sigmoid function,
and ReLU(⋅) denotes the ReLU activation function.
Subsequently, to capture long-range semantic relationships (e.g., correlations between a pedestrian’s head and legs) that local receptive
fields fail to cover, the CLFR module introduces a Non-Local Block for
local–global feature fusion. This block first uses three 1×1 convolutions
(denoted 𝜃(⋅), 𝜙(⋅), 𝑔(⋅)) to transform 𝑂𝑐𝑎 into intermediate features:
𝜃(𝑂𝑐𝑎 ) ∈ R𝐵×𝐶inter ×𝐻×𝑊 and 𝜙(𝑂𝑐𝑎 ) ∈ R𝐵×𝐶inter ×𝐻×𝑊 (for similarity
calculation, with 𝐶inter = 𝐶∕2 to balance performance and efficiency)
and 𝑔(𝑂𝑐𝑎 ) ∈ R𝐵×𝐶inter ×𝐻×𝑊 (for feature aggregation). The spatial
dimensions of 𝜃(𝑂𝑐𝑎 ) and 𝜙(𝑂𝑐𝑎 ) are flattened to 𝑁 = 𝐻 × 𝑊 , and
the similarity matrix 𝑓 ∈ R𝐵×𝑁×𝑁 is computed as 𝑓 = 𝜃(𝑂𝑐𝑎 )𝑇 ⋅ 𝜙(𝑂𝑐𝑎 ).
Softmax normalization of 𝑓 (along the last dimension) yields 𝑓norm , and
global feature aggregation is implemented by multiplying 𝑓norm with
the flattened 𝑔(𝑂𝑐𝑎 ). After reshaping to restore spatial dimensions, a
1 × 1 convolution (𝑊 (⋅)) — followed by batch normalization (BN) —
restores the channel dimension to 𝐶. A residual connection adds the

Through this series of operations, the CLFR module accurately
retains identity-relevant information and suppresses irrelevant or noisy
features introduced by modality discrepancies. By explicitly modeling
both channel and spatial importance in a cascaded manner, CLFR significantly improves the quality and effectiveness of features, providing
more discriminative representations for subsequent VI-ReID tasks.
3.3. Multi-dimensional feature optimization module
In DEEN (Zhang and Wang, 2023), a Multi-stage Feature Aggregation (MFA) block was proposed to extract channel-wise and spatialwise feature representations from multi-level features. However, the
MFA block primarily focuses on aggregating features across stages,
with limited exploration of cross-dimensional feature interactions. In
5

P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

visible–infrared person re-identification (VI-ReID) — where significant modality discrepancies exist between RGB and infrared images
— such single-path aggregation strategies often fail to fully exploit
the discriminative information embedded in both channel and spatial
dimensions.

3.4. Multi-loss optimization
To guide the training process effectively, we adopt a multi-loss
learning strategy that combines four complementary loss functions:
the cross-perspective mutual learning loss 𝐿𝑐𝑝𝑚 and the orthogonal
constraint loss𝐿𝑜𝑟𝑡 from DEEN (Zhang and Wang, 2023), along with the
widely used cross-entropy loss (Luo et al., 2019) and triplet loss (Hermans et al., 2017). These losses capture different aspects of the learning
objective, including feature discrimination, modality alignment, and
identity classification accuracy. The network is trained in an end-to-end
manner by minimizing the weighted sum of these four loss components.
Specifically, we set the balancing coefficients 𝜆1 and 𝜆2 to 0.8 and 0.01,
respectively. The total loss function is formulated as:

Therefore, we propose a Multi-Dimensional Feature Optimization
(MDFO) block, which goes beyond conventional feature aggregation
by explicitly modeling the complex relationships among features across
multiple dimensions to address this limitation. Unlike traditional modules that process features independently within individual dimensions,
the MDFO block performs cross-dimensional collaborative optimization, facilitating more comprehensive feature learning and refinement.
This design is especially advantageous for VI-ReID, where achieving
robust and discriminative feature representations is critical for bridging
modality gaps and capturing identity-specific patterns.

𝑡𝑜𝑡𝑎𝑙 = 𝑐𝑒 + 𝑡𝑟𝑖 + 𝜆1 𝑐𝑝𝑚 + 𝜆2 𝑜𝑟𝑡 ,

This carefully designed loss combination ensures that each individual component contributes meaningfully to the overall optimization,
without dominating or being overshadowed by others during training.

Specifically, at each stage of the backbone network, we consider
two types of input features: a low-level feature map 𝑥𝑙 and a high-level
feature map 𝑥ℎ . To better model global dependencies, we introduce two
non-local blocks into the architecture. These blocks allow the model to
capture long-range correlations across spatial positions and channels—
information critical for identifying discriminative body parts under
varying imaging conditions.

4. Experiments
4.1. Datasets and evaluation metrics
Datasets. The proposed CAFMNet is evaluated on two challenging
large-scale VI-ReID datasets. SYSU-MM01(Wu et al., 2017), which includes 491 person-ID images from four RGB and two NIR cameras.
The LLCM (Zhang and Wang, 2023) is the largest VI-ReID dataset
with 1,064 person-ID images from nine low-light cameras. It presents
challenges such as illumination variations, motion blur, pose changes,
camera view changes, occlusion, and low resolution. The evaluation
modes for LLCM include VIS-to-IR and IR-to-VIS.
Evaluation Metrics. We adopt widely used metrics in person reidentification tasks, including Rank-k accuracy and mean Average Precision (mAP). All results are reported as averages over 10 independent
experimental runs to ensure statistical reliability.

We first apply three 1 × 1 convolutional layers to transform 𝜓𝑞1 ,
1
𝜓𝑘 , and 𝜓𝑣1 into compact feature representations: 𝜓𝑞1 (𝑓ℎ ), 𝜓𝑘1 (𝑓𝑙 ), and
𝜓𝑣1 (𝑓𝑙 ). This transformation not only reduces computational cost but
also preserves essential semantic information:
(
)
𝑀 𝐶 = 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥 𝜓𝑞1 (𝑓ℎ ) × 𝜓𝑘1 (𝑓𝑙 ) .
(10)
This channel-level similarity matrix 𝑀 𝐶 captures the relative importance of different channels through softmax normalization, offering
a more accurate reflection of inter-channel relationships compared to
traditional methods. Based on 𝑀 𝐶 , we then perform channel-level
feature aggregation:
(
)
𝑓ℎ𝑐 = 𝜔𝑠 𝜓𝑣1 (𝑓𝑙 ) × 𝑀 𝐶 + 𝑓ℎ .
(11)

4.2. Implementation details
All models and experiments are implemented using the PyTorch
framework on a single Tesla P100 GPU. We employ a ResNet-50 backbone pretrained on ImageNet for feature extraction. The learning rate
is initialized with a warm-up strategy: it starts at 0.01 and gradually
increases to 0.1 within the first 10 epochs. It is then decayed to 0.01
at epoch 20, further reduced to 0.001 at epoch 60, and finally set to
0.0001 at epoch 120, remaining constant until the final training epoch
(epoch 130). Input images are resized to a fixed size of 3 × 384 × 144.
During training, we sample 6 identities per mini-batch, each contributing 4 visible (VIS) and 4 infrared (IR) images. In the testing phase, only
modality-shared features are used for performance evaluation.
To enhance generalization, we apply common data augmentation
techniques during training, including random horizontal flipping and
random erasing (Zhong et al., 2020).

By fusing low-level details with high-level semantics in a weighted
manner, this operation enhances both the richness and discriminability
of the resulting features.
Next, we perform a similar aggregation in the spatial domain using
𝑓ℎ𝑐 and the original low-level feature map 𝑓𝑙 :
(
)
𝑓ℎ𝑠 = 𝜔𝑠 𝜓𝑣2 (𝑓𝑙 ) × 𝑀 𝑠 + 𝑓ℎ𝑐 ,
where 𝑀 𝑠

denotes the spatial similarity matrix, and 𝜔𝑠

(14)

(12)
and 𝜓𝑣2

are
implemented via 1 × 1 convolutions. Finally, the refined feature 𝑓ℎ𝑠 is
further processed through a dual-attention mechanism that adaptively
recalibrates feature responses in both channel and spatial dimensions:
(
)
𝑍 = SA CA(𝑓ℎ𝑠 ) ,
(13)

4.3. Comparison with state-of-the-art methods

where CA and SA denote the channel attention and spatial attention
modules, respectively. The channel attention module uses global average and max pooling followed by two 1 × 1 convolutions, while spatial
attention computes feature maps based on max and average values
across channels.

We compare the proposed CAFMNet model with recent state-ofthe-art VI-ReID methods that have been evaluated on public VI-ReID
datasets, including SYSU-MM01 and LLCM.
Comparison on the SYSU-MM01 dataset. Comparison on the SYSUMM01 dataset. As shown in Table 1, the results on the SYSU-MM01
dataset demonstrate that the proposed CAFMNet achieves the best
performance among all compared methods. Specifically, under the allsearch mode in SYSU-MM01, CAFMNet achieves a Rank-1 accuracy
of 77.49% and an mAP of 74.19%. Under the indoor-search mode, it
achieves a Rank-1 accuracy of 84.95% and an mAP of 87.09%. These
results clearly validate the effectiveness of the proposed approach.

The key innovation of this attention mechanism lies in its ability
to dynamically emphasize important feature regions—especially those
most relevant for identity discrimination under cross-modal settings.
By integrating these operations into a unified framework, the MDFO
block enables deep feature optimization across multiple dimensions,
effectively enhancing the representational power of the model for
challenging VI-ReID tasks.
6

P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

Table 1
Comparisons between the proposed CAFMNet and some state-of-the-art methods on the SYSU-MM01 dataset.
Method

Venue

Table 3
The influence of each component on the performance of the proposed CAFMNet.

All-search

Indoor-search

Settings (Epoch=150)

SYSU-MM01

R=1

mAP

R=1

mAP

CLFO

CLFR

MDFO

𝑅=1

mAP

FLOPs (G)

Params (M)

×
✓
✓
✓
✓

×
×
✓
×
✓

×
×
×
✓
✓

69.85
72.11
72.88
75.87
77.49

65.55
69.19
70.04
72.29
74.19

33.085
34.453
39.943
41.080
43.833

86.019
89.322
91.384
92.402
94.526

AlignGAN (Wang et al., 2019)
JSIA-ReID (Yang et al., 2020)
DDAG (Ye et al., 2020a)
AGW (Ye et al., 2021b)
MMN (Zhang et al., 2021)
CAJ (Ye et al., 2021a)
CMT (Jiang et al., 2022)
PMT (Lu et al., 2023)
DEEN (Zhang and Wang, 2023)
SGIEL (Feng et al., 2023)
CAL (Wu et al., 2023)
CSL (Sun et al., 2022)
DEN (Kim et al., 2024)
HOS-Net (Qiu et al., 2024)
DNS (Jiang et al., 2025)

ICCV 19
AAAI 20
ECCV 20
TPAMI 21
ACM MM 21
ICCV 21
ECCV 22
AAAI 23
CVPR 23
CVPR 23
ICCV 23
ICME 24
WACV 24
AAAI 24
ECCV 24

42.40
43.40
54.75
47.50
70.60
69.88
71.88
67.53
74.70
75.18
74.66
73.50
76.36
75.60
77.27

40.70
38.00
53.02
47.95
66.90
66.89
68.57
64.98
71.80
70.12
71.73
72.40
71.30
74.20
74.35

45.90
46.80
61.02
54.17
76.20
76.26
76.90
71.66
80.30
78.40
79.69
78.50
83.26
84.20
84.21

54.30
54.70
67.98
62.97
79.60
80.37
79.91
76.52
83.30
81.20
83.68
76.90
84.65
86.70
86.83

CAFMNet (ours)

–

77.49

74.19

84.95

87.09

filtering and highlighting discriminative channel features. Notably, this
improvement is achieved with only a modest increase in computational
cost (FLOPs: +1.368G, +4.13%; Params: +3.303M, +3.84%). These
results suggest that the performance gain stems primarily from the
unique design of CLFO — its ability to adaptively refine channel-wise
feature responses — rather than simply increasing model capacity.
To address the limitations of CLFO’s coarse-grained extraction, the
CLFR module is introduced to further optimize the quality of channellevel features. By incorporating attention mechanisms and multi-stage
refinement strategies, CLFR suppresses irrelevant noise while enhancing key discriminative signals. Adding CLFR after CLFO leads to an
additional gain of +0.77% in Rank-1 and +0.85% in mAP, despite a relatively larger increase in computational complexity (FLOPs: +6.858G,
+20.73%; Params: +5.365M, +6.24%). The performance improvements
justify the necessity of such refined feature processing, indicating that
the benefits are derived from CLFR’s architectural innovations rather
than merely adding parameters.
The MDFO module performs deep multi-dimensional feature processing and complements CLFO by capturing complex inter-feature
relationships across spatial and channel dimensions. When both CLFO
and MDFO are activated, the model achieves 75.87% Rank-1 accuracy
and 72.29% mAP—an impressive improvement over CLFO alone. This
comes with a moderate increase in FLOPs and parameters (FLOPs:
+7.995G, +24.17%; Params: +6.383M, +7.42%), which is well justified
by the significant gains in performance. The effectiveness of MDFO lies
in its non-local and dual-attention structures, which enable the model
to better understand and represent multi-dimensional dependencies.
When the three modules work together synergistically, the full
CAFMNet model achieves optimal performance: 77.49% Rank-1 accuracy and 74.19% mAP. In this configuration, CLFO provides the
initially screened channel features for subsequent modules, CLFR further improves feature quality by suppressing noise and enhancing
discriminative signals, MDFO integrates and strengthens the features
from a macroscopic multi-dimensional perspective. This hierarchical
and collaborative architecture enables the model to more comprehensively and accurately extract and utilize pedestrian identity features,
enabling effective recognition of pedestrians in different scenarios.
Importantly, the ablation study not only validates the effectiveness
of each module but also addresses potential concerns about whether
performance improvements stem from architectural innovation or simply from increased model complexity. As demonstrated in Table 3,
while each module does introduce additional parameters and FLOPs,
the relative gains in performance far exceed what would be expected
from mere increases in capacity. Therefore, the improvements can be
confidently attributed to the thoughtful design of each module and their
cooperative interactions.
In summary, these ablation study results fully verify the rationality
and effectiveness of the CAFMNet network structure design and clearly
demonstrate the important role of each module in improving overall
model performance.
Impact of Each Component in CLFO Module. To systematically
evaluate the effectiveness of each component in the Channel-Level
Feature Optimization (CLFO) module, we conducted a comprehensive ablation study by enabling or disabling individual components

Table 2
Comparisons between the proposed CAFMNet and some state-of-the-art methods on the LLCM dataset.
Method

Venue

LLCM
IR-to-VIS

VIS-to-IR

𝑅=1

mAP

𝑅=1

mAP

DDAG (Ye et al., 2020a)
CMAliqn (Park et al., 2021)
AGW (Ye et al., 2021b)
MMN (Zhang et al., 2021)
CAJ (Ye et al., 2021a)
DART (Yang et al., 2022a)
DEEN (Zhang and Wang, 2023)

ECCV 20
ICCV 21
TPAMI 21
ACM MM 21
ICCV 21
CVPR 22
CVPR23

42.36
42.76
49.13
50.14
48.80
52.97
55.52

48.97
50.95
55.80
56.66
56.60
59.28
62.07

51.42
54.78
63.72
63.97
56.50
65.33
69.21

38.77
40.81
47.21
48.47
47.71
51.13
55.52

CAFMNet (ours)

–

57.58

64.10

69.64

57.16

Complexity

Moreover, the experimental results show that CAFMNet can effectively
reduce the modality discrepancy between RGB and IR images by directly mining channel-level feature information that is beneficial for
person re-identification.
Comparison on LLCM dataset. We verified the effectiveness of the
model on the largest available VI-ReID dataset (the LLCM dataset)
using the updated test patterns, and the results are shown in Table 2.
As shown, CAFMNet achieves strong performance in both infrared-tovisible (IR-to-VIS) and visible-to-infrared (VIS-to-IR) settings. Specifically, in the VIS-to-IR setting, CAFMNet achieves a Rank-1 accuracy
of 69.64% and an mAP of 57.16%. In the more challenging IR-toVIS setting, CAFMNet still demonstrates competitive performance, with
a Rank-1 accuracy of 57.58% and an mAP of 64.10%. These results
fully demonstrate that CAFMNet exhibits strong robustness across different datasets and query patterns, and is capable of effectively addressing complex real-world scenarios in visible–infrared person reidentification tasks.
4.4. Ablation studies
Effectiveness of Each Component. To systematically evaluate the
individual contributions of each module in CAFMNet, we conducted
ablation studies on the SYSU-MM01 dataset. Experimental conditions
were strictly controlled to remain consistent across all trials, with only
specific modules (CLFO, CLFR, MDFO) enabled or disabled to isolate
their effects on model performance.
The CLFO module is specifically designed to extract identificationrelevant features at the channel level. As shown in Table 3, enabling
CLFO alone improved Rank-1 accuracy from 69.85% to 72.11%, and
mAP from 65.55% to 69.19%. This demonstrates its effectiveness in
7

P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

Table 4
Impact of each component in CLFO module.

Table 5
Impact of each component in CLFR module.

Settings (Epoch=150)

SYSU-MM01

SE GN DSC RC L-P F-P

Rank-1 (%) mAP (%) FLOPs (G) Params (M)

CA

LGF

DSC

SA

Rank-1 (%)

mAP (%)

FLOPs (G)

Params (M)

×
×
✓
✓
✓
✓
✓

69.85
71.51
71.76
72.88
68.78
71.20
72.11

×
×
✓
✓
✓
✓

×
✓
×
✓
✓
✓

×
✓
✓
×
✓
✓

×
✓
✓
✓
×
✓

72.11
72.33
72.45
64.40
72.60
72.88

69.19
69.45
69.56
62.57
69.84
70.04

34.453
39.938
38.559
50.800
39.940
39.943

89.322
91.314
90.317
99.759
91.381
91.384

×
✓
×
✓
✓
✓
✓

×
✓
✓
×
✓
✓
✓

×
✓
✓
✓
×
✓
✓

×
✓
✓
✓
✓
×
✓

×
×
×
×
×
✓
×

Complexity

65.55
68.71
68.91
70.05
65.67
68.62
69.19

33.085
34.450
34.452
41.693
34.000
34.454
34.453

Components

86.019
89.191
89.316
106.082
88.270
89.320
89.322

SYSU-MM01

Complexity

the Channel-Level Feature Refinement (CLFR) module, we executed
a meticulous series of ablation experiments, selectively toggling the
presence of each element. Conducted on the SYSU-MM01 dataset,
these trials aimed to dissect how enhanced channel attention (CA),
local–global feature fusion (LGF), depthwise separable convolution
(DSC), and spatial attention (SA) impact visible–infrared person reidentification performance.
As shown in Table 5, the baseline model achieves a performance
benchmark of 72.11% Rank-1 accuracy and 69.19% mAP. Introducing
LGF, DSC, and SA without CA elevates the Rank-1 accuracy to 72.33%
and mAP to 69.45%, highlighting DSC’s critical role in lightweight
design and the synergistic efficiency of LGF and SA in integrating
spatial and hierarchical information. Through hierarchical operations
that decompose spatial and channel processing, DSC efficiently extracts compact cross-modal features while controlling computational
overhead (FLOPs increased by only 16%, from 34.453G to 39.938G).
Meanwhile, LGF combines local details with global context, and SA
enhances focus on discriminative regions, collectively improving the
richness and precision of feature representation.
Incorporating the CA module brings an additional 0.12% boost
in Rank-1 accuracy and 0.11% in mAP. By analyzing global feature
statistics, CA selectively amplifies identity-bearing channels (such as
pedestrian contours and clothing textures) while suppressing modalityinduced noise (such as thermal artifacts in infrared images), further
optimizing channel-level feature representations and aligning with the
design objectives of CLFR. Experiments show that replacing DSC with
standard convolutions slightly improves performance to 72.88% Rank-1
and 70.05% mAP but leads to a 47% surge in FLOPs and a 12% increase
in parameters, underscoring DSC’s irreplaceable role in balancing computational efficiency and performance.
After removing SA, the Rank-1 accuracy drops to 72.60% and the
mAP drops to 69.84%. Although there is still a certain improvement
compared to the baseline model, compared with the complete CLFR
module (CA + LGF + DSC + SA) configuration, the Rank-1 accuracy
decreases by 0.28% and the mAP decreases by 0.20%. This change
indicates that SA plays an important role in guiding the model to focus
on significant spatial regions in complex backgrounds. It suppresses
background interference through a spatial attention mechanism to
ensure that the model maintains high attention to the pedestrian main
body region. In complex scenarios such as occlusion, SA can assist
the model in more accurately focusing on key pedestrian features,
and its absence will weaken the model’s discriminative ability for
spatial features, thus verifying SA’s important supplementary role in
further improving model performance and optimizing spatial feature
representation.
The complete CLFR module (CA + LGF + DSC + SA) achieves
an optimal balance between performance and efficiency, attaining a
Rank-1 accuracy of 72.88% and an mAP of 70.04% with a moderate
increase in computational resources (16% higher FLOPs and 2% more
parameters). Each component has a distinct role yet complements one
another synergistically: DSC and LGF establish a lightweight framework
for feature extraction and multi-scale fusion, while CA and SA optimize
feature quality from the channel and spatial dimensions, respectively,
suppressing modality differences and background noise.

in each experiment. To comprehensively evaluate the effectiveness of
each component in the CLFO module for visible–infrared person reidentification, a series of ablation experiments were conducted on the
SYSU-MM01 dataset. The baseline model omits all CLFO components,
while variants sequentially incorporate Depthwise Separable Convolution (DSC), Group Normalization (GN), Squeeze-and-Excitation (SE)
block, Residual Connection (RC), and the learnable fusion parameter
(L-P). Here, the fixed-parameter fusion (F-P) represents a configuration where the fusion weight remains constant, contrasting with the
adaptive L-P, to highlight the impact of dynamic feature adjustment.
As illustrated in Table 4, the baseline model achieves a Rank-1
accuracy of 69.85% and mAP of 65.55%. Incorporating DSC, GN, RC,
and L-P (without SE) improves Rank-1 to 71.51% and mAP to 68.71%,
validating DSC’s efficiency in extracting modality-invariant features
with low computational cost and GN’s role in stabilizing feature distribution across diverse input modalities. Specifically, DSC’s two-step
operation — depthwise convolution for spatial filtering and pointwise convolution for cross-channel fusion — enables fine-grained feature extraction while reducing FLOPs. Meanwhile, GN normalizes features within groups, mitigating internal covariate shift and enhancing
generalization.
Adding the SE block further boosts Rank-1 by 0.25% and mAP by
0.2%. By recalibrating channel-wise importance through global feature
statistics, SE suppresses modality-specific noise (e.g., thermal artifacts
in infrared images) and emphasizes identity-discriminative channels,
aligning with CLFO’s goal of channel-level feature optimization. Removing DSC in favor of standard convolutions increases Rank-1 to
72.88% and mAP to 70.05%, but at the cost of a 26% increase in FLOPs
and 23% in parameters, indicating DSC’s pivotal role in balancing
performance and efficiency.
The ablation of RC results in a significant performance drop (Rank1: 68.78%, mAP: 65.67%), confirming its necessity in maintaining
gradient flow and preserving input integrity. RC’s learnable residual
connection dynamically balances refined features and raw inputs, ensuring stability during optimization. Comparing the fusion parameter
configurations, L-P outperforms F-P by 0.91% in Rank-1 and 0.57%
in mAP. The learnable nature of L-P allows it to adaptively adjust the
fusion weight based on input characteristics, while the fixed F-P lacks
the flexibility to capture modality-specific variations, highlighting the
superiority of dynamic feature combinations.
The complete CLFO module (SE + GND + DSC + RC + L-P) achieves
an optimal balance, reaching 72.11% Rank-1 and 69.19% mAP with a
minimal increase in computational cost (4% FLOPs, 4% parameters).
These results collectively demonstrate that each component serves a
distinct yet complementary function: DSC and GN enable efficient
feature extraction and stable training, SE refines channel-wise discriminability, RC maintains information integrity, and L-P adapts feature
fusion for cross-modal compatibility.
This carefully designed modular architecture enables CLFO to efficiently mine discriminative feature representations at the channel level,
making it a robust solution for addressing cross-modality challenges in
visible–infrared person re-identification (VI-ReID) tasks.
Impact of Each Component in CLFR Module. To thoroughly assess the individual and collective contributions of components within
8


codex
CFPER 的完整方法和消融已经读完，它的关键不是“部件专家”本身，而是先用难度分流把效率问题改写成检索流程问题。Channel-aware 这篇文本较长，中间被输出截断，我会再针对第 3 节补读。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
4100

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

CLIP-Based Camera-Agnostic Feature Learning for
Intra-Camera Supervised Person Re-Identification
Xuan Tan , Xun Gong , Member, IEEE, and Yang Xiang
Abstract— Contrastive Language-Image Pre-Training (CLIP)
model excels in traditional person re-identification (ReID) tasks
due to its inherent advantage in generating textual descriptions for pedestrian images. However, applying CLIP directly
to intra-camera supervised person re-identification (ICS ReID)
presents challenges. ICS ReID requires independent identity
labeling within each camera, without associations across cameras. This limits the effectiveness of text-based enhancements.
To address this, we propose a novel framework called CLIP-based
Camera-Agnostic Feature Learning (CCAFL) for ICS ReID.
Accordingly, two custom modules are designed to guide the model
to actively learn camera-agnostic pedestrian features: IntraCamera Discriminative Learning (ICDL) and Inter-Camera
Adversarial Learning (ICAL). Specifically, we first establish
learnable textual prompts for intra-camera pedestrian images
to obtain crucial semantic supervision signals for subsequent
intra- and inter-camera learning. Then, we design ICDL to
increase inter-class variation by considering the hard positive and hard negative samples within each camera, thereby
learning intra-camera finer-grained pedestrian features. Additionally, we propose ICAL to reduce inter-camera pedestrian
feature discrepancies by penalizing the model’s ability to predict the camera from which a pedestrian image originates,
thus enhancing the model’s capability to recognize pedestrians from different viewpoints. Extensive experiments on
popular ReID datasets demonstrate the effectiveness of our
approach. Especially, on the challenging MSMT17 dataset,
we arrive at 58.9% in terms of mAP accuracy, surpassing
state-of-the-art methods by 7.6%. Code is available at https://
gitee.com/swjtugx/classmate/tree/master/OurGroup/CCAFL.
Index Terms— Person re-identification, CLIP, intra-camera
supervision, camera-based adversarial loss.

I. I NTRODUCTION

P

ERSON re-identification (Re-ID) involves identifying the
same individual across different camera views. It has

Received 16 September 2024; revised 12 November 2024 and 2 December
2024; accepted 22 December 2024. Date of publication 24 December 2024;
date of current version 7 May 2025. This work was supported in part by the
National Natural Science Foundation of China under Grant 62376231; in part
by Sichuan Science and Technology Program under Grant 24NSFSC1070;
in part by the Science and Technology Research and Development Program
of China National Railway Group Company Ltd., under Grant K2023T003;
and in part by Tangshan Basic Research Science and Technology Program
under Grant 23130230E. This article was recommended by Associate Editor
Y. M. Ro. (Corresponding author: Xun Gong.)
Xuan Tan and Yang Xiang are with Tangshan Research Institute,
Southwest Jiaotong University, Tangshan 063000, China (e-mail: trangle@
my.swjtu.edu.cn; xiangyang@my.swjtu.edu.cn).
Xun Gong is with the School of Computing and Artificial Intelligence and
the Manufacturing Industry Chains Collaboration and Information Support
Technology Key Laboratory of Sichuan Province, Southwest Jiaotong University, Chengdu, Sichuan 610031, China, and also with the Engineering
Research Center of Sustainable Urban Intelligent Transportation, Ministry of
Education, Chengdu, Sichuan 610031, China (e-mail: xgong@swjtu.edu.cn).
Digital Object Identifier 10.1109/TCSVT.2024.3522178

attracted significant attention because of its applications in
person tracking, security systems, and traffic monitoring.
Current research primarily focuses on two directions: fully
supervised [1], [2], [3], [4], [5], [6], [7] and unsupervised [8],
[9], [10]. With the advent of deep learning technologies,
fully supervised person ReID has seen significant performance improvements. However, the considerable annotation
cost associated with the increasing number of cameras and IDs
in real-world scenarios poses a significant challenge for the
practical deployment of ReID systems. Conversely, unsupervised person ReID does not require any label information but
tends to underperform in complex scenarios involving multiple
IDs. In recent years, to combine the advantages and mitigate
the drawbacks of supervised and unsupervised methods, the
Intra-camera supervision (ICS) approach has been proposed.
This approach assumes individual labeling of IDs within
each camera without establishing cross-camera identity links.
As a result, ICS supervision significantly reduces annotation
costs compared to full supervision while still maintaining
identification accuracy. Therefore, it is considered a more
practical setup for ReID scenarios.
However, the lack of cross-camera annotation information
poses a significant challenge for effectively learning pedestrian
features in ICS ReID. Specifically, the number of annotated
pedestrian training samples within each camera is significantly
lower than in fully supervised cross-camera person ReID
tasks. Additionally, due to factors such as varying viewpoints,
occlusion, and background noise, the absence of inter-camera
labels makes it difficult for models to learn variations in
pedestrian appearance across different views, as illustrated in
Fig. 1. Therefore, effectively utilizing intra-camera supervised
information to learn associations between cross-camera IDs is
crucial for addressing ICS ReID tasks.
In ICS ReID, a common approach is dividing model learning into two stages: intra-camera and inter-camera learning.
For instance, the multi-label learning strategy MATE [11]
constructs a Softmax parameter classifier for each camera to
classify pedestrians while associating cross-camera identity
labels. However, the variation in the number of pedestrian
samples within each camera may lead to suboptimal performance. To address this issue, Precise-ICS [12] constructs a
non-parametric classifier [13] for each camera and continues
to train the model by assigning pseudo-labels to highly similar cross-camera images. Despite these advancements, such
methods fail to fully utilize intra-camera sample annotation
information. Pseudo-labels obtained through simple clustering
of different camera angles are often inaccurate and lack flexibility. Notably, the recent large-scale vision-language model

1051-8215 © 2024 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence
and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.
Authorized licensed use limited to:
Downloaded on June 09,2026 atfor
08:58:38
from IEEE Xplore. Restrictions apply.
SeeTIANJIN UNIVERSITY.
https://www.ieee.org/publications/rights/index.html
more UTC
information.

TAN et al.: CLIP-BASED CAMERA-AGNOSTIC FEATURE LEARNING FOR ICS ReID

Fig. 1. Illustration of label settings under different person Re-ID data configurations. The light-blue areas represent the intra-camera and cross-camera
feature spaces, with different shapes corresponding to different identities.
(a) Conventional fully supervised training data requires unified identity
annotation across all cameras. (b) Intra-camera supervised (ICS) training
data only requires independent identity annotation within each camera view,
utilizing separate class spaces. In ICS ReID data, superscripts of identity labels
indicate camera view labels.

CLIP [14] has demonstrated inherent advantages in generating
image textual descriptions. We can leverage CLIP to describe
unseen pedestrian features, thus generating general descriptions of new pedestrian images without additional annotation
data.
However, this approach faces certain limitations in the
task of cross-camera person re-identification (ICS ReID).
Specifically, since only intra-camera identity labels are available, we initially generate implicit textual features based
on these labels to represent pedestrians. During intra-camera
learning, these textual features serve as effective supervisory
signals within each camera. However, in the cross-camera
learning phase, we rely on clustering algorithms to establish
cross-camera identity associations, generating corresponding
textual features for each pedestrian across cameras. While
these features provide additional semantic supervision for
cross-camera learning, pedestrian images are often affected by
factors such as background and viewpoint variations, which
compromise the stability of cross-camera textual features—
especially in complex environments. This noisy, cross-camera
textual information can introduce biases in model learning,
thereby limiting the full potential of CLIP-based methods in
semi-supervised person re-identification.
To further optimize cross-camera textual features, we propose leveraging intra-camera labeled image data and
camera-specific labels to improve the quality of crosscamera associations. To this end, we introduce a CLIP-based
Camera-Agnostic Feature Learning (CCAFL) method, which
progressively enhances feature discriminability through a
three-stage learning process. As illustrated in Fig. 2, our

4101

Fig. 2. The diagram illustrates our proposed approach, which leverages
CLIP and prompt learning to generate textual descriptions for person images
within each camera. Based on this, we combine the textual information with
intra-camera and inter-camera learning, enabling the model to focus better on
discriminative features.

approach’s components work together to address the challenges inherent in ICS ReID tasks. Initially, we employ prompt
learning to generate textual descriptions for each labeled
pedestrian within each camera, thus providing additional
supervisory signals for subsequent intra- and inter-camera
learning stages. By integrating our proposed Intra-Camera
Discriminative Learning (ICDL) and Inter-Camera Adversarial Learning (ICAL) modules, these textual descriptions
enable the model to more accurately capture the representative
features of the same pedestrian across varying viewpoints,
effectively mitigating the impact of noisy pseudo-labels and
enhancing the supervision efficacy of textual prompt learning
in cross-camera person re-identification.
In the intra-camera learning stage, to further enhance the
discriminative power of pedestrian features within each camera, we construct an independent hybrid feature memory
bank for each camera using annotated intra-camera IDs. This
memory bank stores the average features and all instance
features for each ID. Subsequently, we apply Intra-Camera
Discriminative Learning (ICDL), which considers hard-toclassify positive and negative samples within the same camera
while leveraging textual features obtained in the first stage to
further reduce intra-camera intra-class variance and inter-class
similarity.
In the inter-camera learning stage, we first use a
cross-camera association algorithm to link cross-camera
pedestrian IDs to improve the accuracy of cross-camera
pedestrian ID associations and enhance the model’s ability to
recognize pedestrians from different viewpoints. Using these
associated IDs, we construct a cross-camera feature memory
bank that stores their central features for contrastive learning. Then, to better reduce the data distribution differences
between different camera views, we propose an Inter-Camera

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.

4102

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

Adversarial Learning (ICAL) method. Specifically, we add
a multi-camera classifier after the backbone network of the
re-identification model and define ICAL as a multi-positive
class classification loss. During training, by minimizing
ICAL, we force the backbone network of the re-identification
model to learn camera-invariant features by penalizing the
model’s ability to predict the corresponding camera for the
same identity. Backpropagation enables the feature maps
to capture camera-invariant features. Finally, utilizing the
intra-camera supervisory semantic information obtained previously, we assign a textual description to each cluster based on
the generated pseudo-labels. These textual descriptions effectively summarize individual images and serve as additional
supervisory information for inter-camera learning.
Our main contributions can be summarized as follows:
• We propose a simple yet effective three-stage training
strategy, called CCAFL, that integrates CLIP-generated
textual information into the novel semi-supervised ICS
ReID task for subsequent learning.
• Two critical modules: Intra-Camera Discriminative
Learning (ICDL) and Inter-Camera Adversarial Learning
(ICAL) are introduced to compel the model to learn
camera-agnostic features. ICDL aims to extract intracamera fine-grained pedestrian features, while ICAL
reduces the inter-camera discrepancies in pedestrian feature distribution. These modules collectively enhance the
accuracy of cross-camera pedestrian identity recognition.
• Extensive experiments conducted on three popular person
re-identification benchmarks, Market-1501, DukeMTMCReID and MSMT17, demonstrate that our method
significantly outperforms the current state-of-the-art ICS
methods. Our performance even exceeds that of fully
supervised methods.
II. R ELATED W ORK
A. Intra-Camera Supervised Person ReID
With the increasing number of cameras and persons in realworld scenarios, the task of annotating a large-scale ID dataset
becomes prohibitively costly. To address this issue, a setup
known as Intra-camera supervision (ICS) ReID has been proposed, where annotations are performed independently across
various cameras, with labels only available for persons within
the same camera view. Previous methods approached the
ICS ReID problem from two angles: intra-camera supervised
learning and inter-camera ID association learning. In intracamera learning, PCSL [15] and ACAN [16] employ a direct
triplet loss [17] to train models, while MATE [11] constructs
a multi-branch classifier for each camera. However, when the
distribution of intra-camera ID samples is unbalanced and
scant, it can result in biased learning. In contrast, PreciseICS [12] uses a non-parametric classifier and undertakes joint
learning, but insufficient intra-camera learning can severely
impair the model when persons with high intra-camera similarity are present. For inter-camera learning, Precise-ICS
supervises learning through pseudo-labeling based on the similarity of person features across cameras. CMT [18] combines
contrastive learning with the Mean Teacher [19] paradigm to

construct a semi-supervised learning framework. However, the
methods above overlook the labeled instance features within
the same camera, leading to insufficient intra-camera learning
and consequently failing to effectively distinguish pedestrian
features within the same camera. PIRID [20] and DCL [21]
also leverage contrastive learning to learn pedestrian features.
However, as the number of pedestrians within each camera
increases, relying solely on camera-specific mean features
fails to capture sufficient discriminative characteristics, which
ultimately affects model accuracy. In contrast, we propose
a within-camera discriminative learning approach that combines mean features with instance features, enabling more
comprehensive learning of fine-grained pedestrian features
within each camera and thereby improving model performance. Additionally, these methods do not adequately consider
the disparities in data distribution across different cameras,
failing to fully capture the invariant features of pedestrians
across cameras.
Differently, we design Intra-Camera Discriminative Loss
(ICDL) and Inter-Camera Adversarial Loss (ICAL) methods
to effectively enhance the model’s ability to distinguish pedestrian ID features within and across cameras. Additionally,
we incorporate high-level semantic features generated by CLIP
for each person within a camera to further boost the model’s
performance.
B. Unsupervised Person ReID
In recent years, unsupervised person ReID [8], [9], [10],
[22], [23], [24], [25] tasks have attracted wide attention.
These tasks are primarily categorized based on whether additional related data are employed, encapsulating unsupervised
domain-adaptive (UDA) ReID and purely unsupervised learning (USL) ReID. The latter, pure unsupervised ReID, presents
greater challenges due to its independence from any external
data. However, with the successful application of contrastive
learning in the unsupervised domain, the performance of
USL ReID has significantly increased - notable methods
include SPCL [8]’s self-paced contrastive learning procedure
that builds a mixed memory bank, fully exploiting all available data. CAP [22] technique divides clusters into multiple
camera-perception proxies based on the camera ID to alleviate
discrepancies in ID features generated by camera perspective
alterations. ClusterContrast [24] directly establishes a simple
yet effective cluster-level memory bank, achieving decoupling
between feature update rates and the number of images.
RTMem [26] employs a real-time memory update strategy,
updating cluster centroids by randomly sampling current
mini-batch instance features without the need for momentum.
In contrast, LP [27] considers two types of additional features
from different local views and leverages the knowledge of an
offline teacher model to optimize the model. In this study, our
work is grounded in the framework of intra- and inter-camera
contrastive learning, which is a widely used and effective
representation learning method for unsupervised person ReID.
C. Vision-Language Models
Large-scale pre-trained vision-language models, integrating
copious amounts of textual and visual data, have proven

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.

TAN et al.: CLIP-BASED CAMERA-AGNOSTIC FEATURE LEARNING FOR ICS ReID

their efficacy across various domains. For example, Contrastive Language-Image Pretraining (CLIP) [14] model, which
employs the InfoNCE loss [28] function to jointly train
text and image encoders, resulting in significant performance
improvements in numerous downstream tasks. Additionally,
to further tap into CLIP’s potential, CoOp [29] introduces
prompt learning, aiming to uncover the implicit textual cues
within images, effectively migrating CLIP to a broader range
of downstream tasks. Within the realm of ReID, CLIP has been
extensively applied. For instance, CLIP-ReID [30], by aligning
image and textual information within a singular embedding
space, reinforces the connection between image features and
related textual descriptions. CCLNet [31] establishes learnable
cluster-aware prompts for person images and generates text
descriptions to assist subsequent unsupervised visible-infrared
person re-identification training.
However, the immense potential of CLIP in facilitating
semi-supervised person ReID learning has yet to be explored.
In this paper, we fully integrate CLIP with the ICS ReID task
to construct a CCAFL framework, offering new insights for
semi-supervised ReID.
D. Adversarial Learning
The application of adversarial learning in person
re-identification can be traced back to the use of Generative
Adversarial Networks (GANs) [32] to generate realistic
person images. For example, Jiang et al. [33] proposed a
GAN-based method that performs selective sampling of
generated data to bridge the gap between domains and
enrich the feature space. In recent years, the application of
adversarial learning has extended beyond image generation
and has been widely applied to various aspects of person
re-identification. For instance, in unsupervised domain
adaptation for person re-identification, CAWCL [34] employs
a Gradient Reversal Layer (GRL) [35] to align the distribution
of each camera. However, using traditional domain adversarial
learning to eliminate camera styles can negatively impact the
model’s ability to recognize pedestrians. In clothing change
person re-identification, CAL [36] proposed a clothing-based
adversarial loss to decouple clothing-independent features.
In contrast to these methods, we propose an inter-camera
adversarial loss that penalizes the model’s ability to predict
the same identity under different cameras, thereby enabling
the model to extract inter-camera agnostic features.
E. Semi-Supervised Visible-Infrared Person ReID
Visible-infrared person re-identification (VI-ReID) aims to
match individuals captured in one modality with their counterparts in another. However, the development of existing
VI-ReID methods remains limited due to the lack of annotated infrared data, which complicates the training process.
To address the challenges of large-scale cross-modality data
annotation, several semi-supervised VI-ReID methods [37],
[38], [39] have been proposed. These approaches leverage
both labeled and unlabeled data to learn modality-invariant
and identity-discriminative features. For example, DPIS [37]
introduces a dual pseudo-label interactive self-training method

4103

that integrates pseudo-labels generated by different models into a hybrid pseudo-label, effectively mitigating noise
issues. DMA [38] proposes a dual modality-aware alignment
model that preserves discriminative identity information while
suppressing misleading information. MUN [39] employs a
cross-modality learner and an intra-modality learner to generate robust auxiliary modalities, addressing both modality
discrepancies and intra-class variations effectively.
Currently, mainstream semi-supervised VI-ReID methods
primarily focus on two scenarios: one where only visible light
images are annotated, and another where partial annotations
are provided across both modalities. These methods mainly
concentrate on addressing the issue of cross-modality label
alignment. However, unlike these semi-supervised VI-ReID
scenarios, intra-camera supervised person re-identification
represents a unique setup, where only labels within each
camera are considered, and cross-camera label associations are
ignored. Although our method is currently applied primarily
to intra-camera supervised ReID scenarios, the intra-camera
discriminative contrastive loss module and cross-camera
adversarial learning module in our approach—which leverage
partial labels to learn discriminative features—can also be
effectively applied to semi-supervised visible-infrared ReID.
This approach is particularly suited to scenarios where visible
and infrared cameras have independently annotated identity
labels. One of the main changes is how to conduct robust
feature learning. Therefore, our method provides new insights
and challenges for exploring broader semi-supervised VI-ReID
scenarios.
III. M ETHODOLOGY
A. Overview
Based on the ICS ReID problem, the training dataset
only contains intra-camera IDs and lacks inter-camera IDs.
Therefore, in this case, a dataset consisting of C cameras can
be represented as D = {D1 , D2 , . . . , DC }. Specifically, the
images of persons from the c-th camera can be represented
as Dc = {(xi , y j , c)}, where xi indicates the i-th person
image under this camera, y j (0 ≤ j < Nc ) represents the
corresponding label, and Nc denotes the number of person
IDs under this camera. For instance, in the Market-1501 [40]
dataset, which contains six cameras, there are 751 pedestrian IDs under supervised conditions, meaning these IDs
are associated across different cameras. However, in the ICS
setting, the IDs for each camera are independently annotated,
with the number of pedestrian IDs for each camera being
D = {652, 541, 694, 241, 576, 558}, resulting in a total of
3,262 global IDs. Although multiple cameras may capture the
same pedestrian, their global IDs remain distinct. Therefore,
each person sample in the training set carries three labels:
the intra-camera ID, the camera label, and the global ID.
Moreover, since the same pedestrian might be captured by
multiple cameras, they have different IDs assigned to them
depending on which camera they were captured by. Thus,
our primary objective is to learn feature representation for
individuals across different cameras.
Recently, the CLIP model, trained on large-scale datasets,
has demonstrated remarkable proficiency in matching

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.

4104

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

Fig. 3. The framework of our CCAFL. Left: Through prompt learning paradigms, we generate text descriptions corresponding to the labels of each person’s
image within a camera. This provides semantic supervision information for subsequent intra-camera and inter-camera learning. Upper: In the intra-camera
learning phase, we construct a hybrid memory for each camera, storing both the central features and instance features of pedestrians. By employing an
intra-camera discriminative loss, we enhance the discriminability of pedestrian features within the same camera. Lower: In the inter-camera learning phase,
we obtain cross-camera association IDs through a cross-camera association step. We then build a memory that stores prototype features of associated
pedestrians, aiding the model in learning pedestrian features across different cameras. Additionally, we introduce a global ID classifier and incorporate
inter-camera adversarial learning to mitigate the impact of camera discrepancies.

image-text descriptions. Its image encoder captures complex
and rich visual features, while the text encoder provides
enhanced semantic information. Building upon this, we have
developed a learning framework that integrates CLIP with
ICS-ReID to discern intra- and inter-camera ID identities,
as illustrated in Fig. 3. The framework consists of three
training steps: intra-camera pre-defined prompt learning, intracamera learning, and inter-camera learning. Through this
three-stage training process, the model can deeply explore
pedestrian information from different camera angles and effectively guide the establishment of more accurate cross-camera
ID associations.
B. Intra-Camera Pred-Defined Labels Prompt Learning
The current research on ICS ReID commonly adopts a
two-stage learning approach, namely intra- and inter-camera
stages. In the inter-camera learning phase, pseudo-labels
are often assigned to IDs from different camera views
using similarity-matching. However, due to variations in perspective, these pseudo-labels tend to be inaccurate, which
can hinder the learning process. To address this issue,
we integrate text encoder and prompt learning mechanisms
from the CLIP framework to generate descriptive textual
prompts corresponding to individual identities. This incorporation provides valuable semantic constraints for subsequent
inter-camera learning and serves as an adjunct in rectifying pseudo-labels, thereby enhancing the overall recognition
performance.

More specifically, we first define a textual prompt based
on the predefined labels within each camera, described as “a
photo of [X ]1 [X ]2 . . . [X ] M person,” where M represents the
number of learnable text tokens. Subsequently, we input the
implicit textual prompts and ID images into the CLIP model
to optimize the text tokens [X ]. Through this method, we can
obtain textual representations associated with pedestrian IDs
within each camera. It is important to note that, in this training
phase, we freeze CLIP model’s image and text encoder while
utilizing image-to-text and text-to-image losses to learn the
text tokens:
 
 
C
exp s f iv , f pt /τ
X
1 X
log P B
Li2t = −
  , (1)
t
v
|Pi |
k=1 exp s f i , f k /τ
p∈Pi
c=1
 
 
C
exp s f pv , f it /τ
X
1 X
Lt2i = −
log P B
  , (2)
t
v
|Pi |
k=1 exp s f k , f i /τ
p∈Pi
c=1
where Pi = { p|y p = yi , p ∈ {1, 2, . . . , B}} represents the
index set of positive image samples, s(, ) represents the cosine
similarity between text features and image features, C is the
number of cameras, B is the batch size, and τ denotes the
temperature factor. Ultimately, the loss of intra-camera predefined label prompt learning is:
L pr ompt = Li2t + Lt2i .

(3)

By minimizing L pr ompt , we can learn the corresponding
implicit textual descriptions for the IDs within each camera.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.

TAN et al.: CLIP-BASED CAMERA-AGNOSTIC FEATURE LEARNING FOR ICS ReID

4105

In the subsequent stages, we will utilize these textual descriptions to provide stronger semantic supervision for the model,
thereby enhancing its generalization capability.

C. Intra-Camera Discriminative Learning
The primary challenge of ICS ReID lies in annotating
IDs within each camera’s view and establishing cross-camera
ID associations through the analysis of inter-camera characteristics. Intra-camera learning, therefore, can be considered
a fully supervised problem within a multi-task framework.
However, the distribution of sample numbers for each ID
within a camera is uneven, with most IDs having only a
limited set of training samples. This imbalance can lead to
model bias toward learning prominent camera-style features,
rather than ID features. Additionally, the lack of cross-camera
ID information highlights the significance of intra-camera
learning as a preparatory phase for subsequent cross-camera
correlation.
Considering these factors, we focus on the centroid features
of each pedestrian within each camera, as well as the hard
positive and hard negative samples for that pedestrian within
the camera, as shown in Fig. 4. This compels the model
to learn more accurate intra-camera pedestrian features. This
approach also emphasizes the differences between pedestrian
IDs within the camera view, providing more reliable features
for subsequent inter-camera association steps.
1) Intra-Camera Hybrid Memory Banks Initialization:
Firstly, we initialize the intra-camera hybrid memory bank
using an image feature encoder. All image features are
assigned to the corresponding camera memory based on different camera IDs. Then, for each image under each camera’s
pre-defined labels, the centroid features of each pedestrian ID
within the camera are stored in the intra-camera centroid memory through averaging, to learn the pedestrian features within
each camera. Additionally, the instance features corresponding
to each pedestrian ID within each camera are stored in the
intra-camera instance memory to learn more discriminative
information. The mean feature of the predefined labels of
pedestrians within a camera is calculated as follows:
µic =

1 X
f (x),
|Nci |
i

(4)

x∈Nc

where µic represents the mean feature of pedestrian ID i
within camera c, Nci denotes the set of all images belonging
to pedestrian ID i within camera c, and f (x) represents
the features of image x after being processed by the image
encoder. Therefore, the memory bank Mcintra is initialized with
the mean features of the pedestrian IDs, while Miintra is initialized with the instance sample features corresponding to those
pedestrian IDs. For intra-camera instance memory, we employ
a real-time instance feature memory update strategy. In each
iteration, we directly replace Miintra in the memory with the
current mini-batch instantaneous feature f x :
Miintra

 
y ← f (x).

(5)

Fig. 4. Illustration of Lintra1 and Lintra2 . The same color indicates that
all samples originate from the same camera, while different shapes represent
different pedestrian IDs within the camera.

2) Optimization: In each training iteration, the features
stored in the aforementioned intra-camera hybrid memory
bank are updated using different strategies.
For the memory bank Mcintra :
µic ← αµic + (1 − α)e
µic ,

(6)

where µ
eic denotes the average features of the camera c insider
ID i in each batch, α is the momentum updating factor.
This update mechanism ensures that the features in the memory bank consistently reflect the latest training information,
thereby enhancing the accuracy and stability of the model
in learning intra-camera pedestrian features, as shown in
Fig. 4 (a). To this end, given a query image feature f (x),
we propose an intra-camera centroid contrastive loss function,
which is formulated as:

C
X
exp s( f (xi ), µ+ )/τ
log P K
Lintra1 = −
 , (7)
c
i
j=1 exp s( f (x i ), µc )/τ
c=1
where µ represents the centroid feature for each ID in the
c-th intra-camera memory, K c represents the total number of
pedestrian IDs in the camera, and C is the number of cameras.
Through the above loss, we can effectively bring the sample
closer to the centroid feature of its corresponding ID while
pushing it away from other ID centroid features within the
same camera.
However, when faced with challenging samples within the
camera, such as similar apparel or shared backgrounds, this
approach may result in poor classification of IDs within the
camera. Moreover, as the dataset expands and the number
of individuals per camera increases, the model’s recognition
performance may be adversely affected. Therefore, we further
enhance inter-class separability and intra-class compactness
by merging all instance features under each ID, as shown in
Fig. 4 (b). Specifically, for a query image xi , we examine
the relationship between the hardest positive sample and the
hardest negative sample from other IDs stored in the memory.
By calculating loss across different cameras, we reduce the
distance between samples and their centroids as well as
relevant hard positive samples while increasing distances from
other hard negative samples.
Lintra2 = −

C
X

exp(s( f (x), m +
har d )/τ )
log P K
,
j
c
c=1
j=1 exp(s( f (x), m har d )/τ )

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.

(8)

4106

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

where m +
har d represents the hardest positive sample characteristics retained in memory Miintra , it demonstrates the cosine
similarity is the lowest compared to all instance features under
j
this pedestrian ID. Conversely, m har d is the hardest negative
sample feature which shows the highest cosine similarity when
compared to all other instance features from different IDs
under the camera.
3) Intra-Camera Image-Text Alignment: During the second
stage of intra-camera learning, we freeze the text encoder and
only train the image encoder. Specifically, for each person ID
under each camera, we obtain the corresponding text features
by inputting prompts into the text encoder, derived from the
first stage. Meanwhile, we input the image x into the image
encoder to obtain the raw features f v (x). Subsequently, we use
the loss Li2tce to constrain the image features f v (x) to be
close to the corresponding text features f t (y), while being
distant from the text features of other identities:
Lintra
i2tce =

Kc
C X
X

exp(s( f v (x), f t (yi )))
, (9)
−qz log P K
c
v
t
z=1 exp(s( f (x), f (yz )))
c=1 i=1

where qz is a smoothed ID label. Notably, our loss function
is computed for features within each camera independently,
ensuring that these calculations are performed separately for
each camera.
4) The Loss for Intra-Camera Learning: In summary,
the proposed intra-camera loss Lintra is composed of the
intra-camera discriminative loss and the intra-camera imagetext alignment loss, defined as follows:
L I C DL = λLintra1 + (1 − λ)Lintra2 ,

IDs i and j. The edge e(i, j) is defined as follows:


 1, dist (i, j) < T ∧ c(i) ̸ = c( j)
e(i, j)=
∧ i ∈ N1 ( j, c(i)) ∧ j ∈ N1 (i, c( j));


0, other wise.

where dist (i, j) represents the distance between the centroid
features of the i-th and j-th ID IDs, c(i) indicates the camera
to which the IDs belong, and T is the distance threshold.
N1 ( j, c(i)) designates the nearest neighbor of the j-th ID with
the i-th ID under the c(i) camera. Using the conditions defined
above, a sparsely connected graph is constructed. Then, based
on similarity, we identify all connected components and assign
inter-camera pseudo-labels to IDs.
2) Inter-Camera Memory Banks Initialization: Based on the
successful application of contrastive learning in the field of
person re-identification [10], [24], [41], we employ a prototypical contrastive learning paradigm for inter-camera learning.
First, upon completion of intra-camera learning, we generate
pseudo-labels for IDs using the aforementioned inter-camera
association algorithm. Next, we compute the mean features
of these samples based on their corresponding pseudo-labels
and directly initialize the inter-camera memory bank. This
approach provides a stable starting point for prototypical
contrastive learning. Consequently, our inter-camera memory
stores the mean features of the associated IDs across different
cameras, facilitating the learning of a person’s appearance
characteristics under varying camera conditions. The memory
features are updated using online batch features in a moving
average manner, as described by the following formula:

(10)

M[y] ← α M[y] + (1 − α) f x .

where λ is the balancing factor between the two losses.
Lintra = L I C DL + Lintra
i2tce .

(11)

D. Inter-Camera Learning
Through the aforementioned intra-camera learning, our
model effectively identifies each person within the camera’s
view. However, the abundant learning information of IDs
across cameras has yet to be fully utilized. Therefore, in the
inter-camera learning process, we have devised an alternating
strategy consisting of cross-camera ID association steps and
inter-camera contrastive learning steps to facilitate the model’s
acquisition of view-invariant ID features.
1) Inter-Camera Association: The ICS ReID approach differs from fully unsupervised ReID, which relies on clustering
algorithms to obtain pseudo-labels directly. In the case of
intra-camera IDs, assigning the same pseudo-labels is not
feasible. Therefore, we employ an ID association algorithm
based on connected components proposed in [12]. Specifically,
we impose two constraints on the clustering process: 1) under
the in-camera supervised condition, positive matches among
IDs within each camera should not exist, and 2) a maximum of
one positive match is allowed per camera. We then constructed
an undirected graph G = ⟨V, E⟩ for associations, where
the vertex set V represents the accumulated IDs across all
cameras, and the edge set E represents a positive pair between

(12)

(13)

3) Optimization: To further learn the prototype features
of IDs under different cameras, the inter-camera prototypical
contrastive loss is defined as follows:
exp(s( f (x), M[y])/τ )
,
L I PC L = − log P Z
i=1 exp(s( f (x), M[ j])/τ )

(14)

where Z represents the number of IDs associated in each epoch
of inter-camera correlation.
4) Inter-Camera Image-Text Alignment: Considering the
significant variations in illumination and background, IDs
across cameras often exhibit notable feature differences,
leading to noisy inter-camera association labels. Hence,
we combine the text description learned in the first stage with
the inter-camera prototypical contrastive learning, leveraging
additional semantic supervision information to assist the model
in improving the accuracy of inter-camera ID correlation and
learning the prototype features of IDs across cameras. Specifically, we define an image-to-text contrastive loss function:
Linter
i2tce =

exp(s( f v (x), f t (yi )))
,
−qz log P Z
v (x), f t (y )))
exp(s(
f
z
z=1
i=1

Z
X

(15)

5) The Loss for Inter-Camera Learning: The total loss in
inter-camera learning can be summarized as follows:
Linter = L I PC L + Linter
i2tce .

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.

(16)

TAN et al.: CLIP-BASED CAMERA-AGNOSTIC FEATURE LEARNING FOR ICS ReID

Fig. 5.
The probability distributions of intra-camera samples processed
by the global classifier in Market-1501 dataset are as follows: The left
figure illustrates that, after a certain number of training epochs, the samples
with true intra-camera labels exhibit a distinct probability distribution with
a sharp peak, indicating that the classifier effectively distinguishes different
individuals across different cameras. The right figure shows that, after initiating inter-camera adversarial learning, inter-camera association labels are
obtained through an inter-camera association algorithm. Samples with the
same pseudo-label across different cameras are treated as positive examples,
which enhances the probability distribution of the same person across cameras
in the global classifier, resulting in multiple peaks.

4107

predicting which specific camera a pedestrian image feature
originates from.
2) Learning Inter-Camera Agnostic Features: In the second
step, we fix the parameters of the global ID classifier and compel the network to learn camera-irrelevant features. To achieve
this, we penalize the model’s prediction capability regarding
global IDs. Specifically, while the classifier mentioned above
can distinguish different pedestrians across cameras using
global IDs, our goal is to train the global ID classifier to
not distinguish the same identity across different cameras.
Therefore, we introduce an Inter-Camera Adversarial Loss
(ICAL), a multi-positive class classification loss where all
categories belonging to the same identity but different cameras
are considered positive classes, as shown in Fig. 5. Notably, the
same pedestrians across different cameras are identified using
pseudo-labels obtained via the above inter-camera association
algorithm. The ICAL is formulated as follows:
L I C AL = −

NG
N X
X

q(g)

i=1 g=1






log 



exp f (xi ) · ϕg /τ
 P

,
exp f (xi )· ϕg /τ +
exp f (xi ) · ϕ j /τ 


E. Inter-Camera Adversarial Learning
Our model can recognize pedestrian identities across different cameras through the previously described intra- and
inter-camera contrastive learning. However, during intercamera learning, the variance in pedestrian feature distributions between cameras introduces label noise, and the
value of predefined intra-camera label information is not
fully exploited. To address this issue, we propose InterCamera Adversarial Loss (ICAL). ICAL penalizes the model’s
prediction capabilities across different cameras, forcing the
backbone network to extract inter-camera agnostic features.
To achieve this, we introduce a new global ID classifier
based on camera data, appended to the network, as shown
in Fig. 3. Each training iteration consists of two optimization
steps:
1) Training the Inter-Camera Global ID Classifier: First,
we establish a classifier C C (·), where each class corresponds to a global ID. We optimize this global ID classifier
by minimizing the classification loss LG I D , defined as the
cross-entropy loss between the predicted pedestrian C C ( f (xi ))
and the global label yiG . We perform L2-normalization on the
model’s output features f (xi ) and denote the L2-normalized
weights of the j-th global ID classifier as ϕ j . We detach these
weights before inputting them into the global ID classifier
to ensure that the classifier’s training does not influence the
model itself. Consequently, LG I D can be expressed as:


N
exp f (xi ) · ϕ y G /τ
X
i
LG I D = −
log
,
(17)
N
G

P
i=1
exp f (xi ) · ϕ j /τ
j=1

where N is the batch size, N G is the number of global ID
classes in the training set, and τ is a temperature parameter.
By using global IDs as labels during training, our classifier can
distinguish pedestrians across different cameras, effectively

j∈G i−

(18)
where G i− represents the set of global IDs of the negative
centroids of the query. q(g) is the cross-entropy loss weight
for the g-th global ID category.
Given that the same pedestrians across different cameras are
identified using pseudo-labels obtained via the inter-camera
association algorithm, which contains noise compared to
pre-defined intra-camera labels, to enhance the model’s intercamera pedestrian recognition capability without significantly
compromising intra-camera accuracy, we define q(g) as:

ϵ

1 − ϵ + , if g = yiG


G
ϵ
q(g) =
(19)
,
if g ̸ = yiG and g ∈ G i+

G


0,
if g ∈ G i−
where G i+ represents the set of global IDs of the positive
centroids of the query, G is the number of elements in G i+ , and
G i− represents the set of global IDs of the negative centroids
of the query. ϵ is a hyperparameter with a range of 0 < ϵ ≤ 1.
Importantly, our goal is to optimize both ICAL and the
inter-camera loss concurrently. Linter and L I C AL are correlated in learning camera-irrelevant features. When using Linter
alone, the model tends to learn simple samples (highly similar
pedestrian features across different cameras) in the early
epochs of optimization and gradually distinguishes harder
samples (pedestrian features with low similarity due to factors
like pose, lighting, and background changes). L I C AL aims to
narrow the feature gap for the same identity across different
cameras with pseudo-labels, which is similar to the goal of
Linter . To avoid local optima caused by directly minimizing
L I C AL from the beginning, we execute inter-camera adversarial learning after a certain number of epochs. Consequently,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'CLIP-driven fine-grained mining for text-based person search.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 235ms:
oder (MED) with both understanding and
generation capabilities for deeper interaction between image and
text modalities.

𝐿𝑠𝑑𝑚 =

𝐵 𝐵
𝐵 ∑
𝐵
𝑝𝐼2𝑇
𝑝𝑇𝑖,𝑗2𝐼
∑
𝑖,𝑗
1 ∑ ∑ 𝐼2𝑇
(
𝑝𝑖,𝑗 log(
)+
𝑝𝑇𝑖,𝑗2𝐼 log(
)),
𝐵 𝑖=1 𝑗=1
𝑦𝑖,𝑗 + 𝜖
𝑦𝑖,𝑗 + 𝜖
𝑖=1 𝑗=1

(1)

𝐼2𝑇 ∕𝑇 2𝐼

where 𝑝𝑖,𝑗
represents the proportion of the image-to-text or textto-image cosine similarity score in a mini-batch. 𝜖 is a small number
to avoid numerical problems, 𝐵 denotes the batch size and 𝑦𝑖,𝑗 is the
ground truth probability.
3.2. Local image embedding extraction
To explicitly leverage the fine-grained image information, most previous CNN-based works for text-based person search typically employ
hard horizontal slicing (Sun et al., 2018) to extract local visual features.
However, due to the characteristic of self-attention, using a Transformer as visual encoder inevitably integrates information from the
entire image. Thus, directly applying horizontal slicing to the output of
Vision Transformers (ViT) is suboptimal. According to the analysis of
attention distance in Ghiasi et al. (2022), certain attention heads in the
lower layers of ViT exhibit small attention distance, indicating some
degree of local attention. We suggest that in transformer-based networks, more consideration should be given to the impact of attention
mechanism on local feature extraction.
The masked attention was initially proposed by Veličković et al.
(2017) and became widely known when the Mask2Former model
(Cheng et al., 2021) adapted it as a constrained cross-attention module.
Inspired by advances in image segmentation (Jiao et al., 2023; Xu
et al., 2023; Cheng et al., 2021), we propose a novel Attention Biasbased Forward process (ABF) as illustrated in Fig. 3(a). Similar to
Mask2Former, ABF also extracts local features by modulating the attention matrix to limit attention within specific regions. The difference
is that Mask2Former introduces masked attention to alleviate the
slow convergence of query features caused by global context in the
cross-attention layer. It enhances the sensitivity of query features to
foreground information, aiming to extract region proposals of specific
types from an image. Whereas, our proposed ABF leverages global
contextual information to mine additional semantic clues within the
specified regions through unidirectional information transfer of local
patch sets.
Specifically, ABF does not modify the forward process of the first
𝐿 layers in the visual encoder, allowing the [CLS] token to capture

Some recent re-ID works, such as Yang et al. (2023), Zuo et al.
(2023), Jin et al. (2025) and Shu et al. (2021), attempt to put the spotlight on pre-training a model from scratch. These approaches capture
more fine-grained associations by constructing large-scale pedestrian
datasets and employing pre-training tasks related to alignment targets.
Given that pretraining a model from scratch is too expensive, we adopt
the CLIP model to initialize the encoders and fine-tune it entirely on
the TBPS task.
3. Method
In this section, we present our proposed CDFM framework. The
overview of CDFM is illustrated in Fig. 2, and the details are discussed
in the following subsections.
3.1. Revisiting CLIP’s dual-encoder and global alignment
With the advancement of VLP models, recent studies (Yan et al.,
2022; Jiang and Ye, 2023) have attempted to transfer the knowledge
of CLIP to text-based person search. We initialize the CDFM with the
full CLIP image and text encoders to leverage its powerful cross-modal
alignment capability.
Image Encoder. Given an input image 𝐼 ∈ 𝑅𝐻×𝑊 ×3 , we first divide
it into 𝑁 = 𝐻 × 𝑊 ∕𝑃 2 non-overlapping patches, where 𝑃 denotes the
size of each patch. These patches are then flattened and prepended
with a learnable [CLS] token to form an input sequence. We adopt
a 12-layer Vision Encoder to model correlations among the patches.
3

X. Lin, X. Geng, S. Wu et al.

Computer Vision and Image Understanding 267 (2026) 104741

Fig. 2. Overview of the proposed CDFM framework. We embed image and text into the latent space using a dual-stream feature extraction backbone and
optimize them with SDM Loss and ID Loss. The attention Bias is added to the last 12 − 𝐿 layers of the backbone for extracting local image embeddings. A
multimodal decoder is designed to extract fine-grained text embeddings by applying a cross-attention mechanism. 𝐸𝐷𝐿 and 𝐶𝑆𝐿𝑀𝑂𝐷 are proposed to ensure the
robustness of fine-grained embeddings. In the inference phase, the part surrounded by the black dashed line is removed.

passing through the softmax activation function. Fig. 3(b) represents
the visualization of matrix 𝑀𝐵 . The upper 𝐾 rows of the matrix
𝐿 . We set the value corresponding
represent the update process of 𝐹𝑙𝑜𝑐𝑎𝑙
𝐿
to 𝑓𝑐𝑙𝑠 to zero, allowing each group to still receive global information,
making local features more robust. The upper 𝐾 rows can be formulated
as:
{
0, if 𝑚𝑖,𝑗 ∈ 𝑖th 𝑔𝑟𝑜𝑢𝑝
𝑚𝑖,𝑗 =
, 𝑖 ∈ [0, 𝐾 − 1], 𝑗 ∈ [0, 𝑁 + 𝐾],
(2)
−∞, if 𝑚𝑖,𝑗 ∉ 𝑖th 𝑔𝑟𝑜𝑢𝑝
The bottom (𝑁 +1)×(𝑁 +1+𝐾) elements are the numerical setting of
𝐹𝑣𝐿 . It can be seen that with the assistance of matrix, the 𝐹𝑣𝐿 (latter part
∗
of 𝐹𝑣𝐿 ) is updated with the original vision encoder forward process,
ensuring the model’s global context extraction capability is preserved.
Under the setting of matrix, the forward process of the subsequent 12−𝐿
layers is as follows:
( (
)
) (
)𝑇
 𝐹𝑣𝐿∗  𝐹𝑣𝐿∗
( 𝐿∗ )
𝐹𝑣(𝐿+1) = Sof tmax
+
𝑀
+ 𝐹𝑣𝐿∗ ,
(3)
√
𝐵  𝐹𝑣
𝑐

Fig. 3. Overview of the proposed ABF. (a) The proposed Attention Biasbased Forward process (ABF), (b) Visualization of the attention bias matrix
𝑀𝐵 , where white squares indicate a value of 0 and black squares indicate a
large negative value.

rich contextual information during early feature extraction. We denote
𝐿 ,𝐹𝐿
𝐿
the output of 𝐿th layer as 𝐹𝑣𝐿 = {𝑓𝑐𝑙𝑠
} ∈ 𝑅(𝑁+1)×𝐶 , where 𝑓𝑐𝑙𝑠
𝑃 𝑎𝑡𝑐ℎ
represents the embedded visual [CLS] token and 𝐹𝑃𝐿𝑎𝑡𝑐ℎ represents the
patch tokens. In order to mine fine-grained information in transformer
architecture, we divide patch tokens into 𝐾 groups and prepend a
specific local [CLS] token to the beginning of each group. As mentioned
before, the embedded visual [CLS] token should have a comprehensive
understanding of the context within the entire image. We repeat it 𝐾
𝐿
times, denote as 𝐹𝑙𝑜𝑐𝑎𝑙
and concatenate them to the beginning of 𝐹𝑣𝐿 as
local [CLS] tokens. Therefore, the input of subsequent 12-L layers will
∗
𝐿 , 𝐹 𝐿 }.
be 𝐹𝑣𝐿 = {𝐹𝑙𝑜𝑐𝑎𝑙
𝑣
Considering the standard multi-head attention treats each token
equally, we introduce an attention bias matrix 𝑀𝐵 = {𝑚𝑖,𝑗 } ∈
𝑅(𝑁+1+𝐾)×(𝑁+1+𝐾) into the self-attention computation to ensure that
each local [CLS] token only focuses on its respective local group.
Numerically, when an element of the matrix is set to a very large
negative value, the attention weight at that position will be zero after

where (⋅), (⋅) and (⋅) denote the query, key and value transformation respectively.
The output of vision encoder is then projected to the image–text
joint latent embedding space via a learnable image projection, resulting
in the local image embeddings 𝐹𝑣𝑙𝑜𝑐𝑎𝑙 .
3.3. Fine-grained text embedding extraction
3.3.1. Fine-grained embedding learning
Existing methods for parsing input text into fine-grained phrases
typically rely on external tools such as natural language toolkit, which
use lexical properties for text analysis. However, the accuracy of these
methods is highly dependent on the quality of the external tools. To
address this limitation, we propose a Fine-grained Embedding Learning
(FEL) module for extracting local representations based on semantic
similarity, eliminating the need for external tools.
4

X. Lin, X. Geng, S. Wu et al.

Computer Vision and Image Understanding 267 (2026) 104741

fine-grained image embeddings (𝐹𝑣𝑓 𝑖𝑛𝑒 ). This modality-sharing facilitates modal interaction and helps bridge the modality gap between
images and text. As mentioned before, local image embeddings are
obtained through ABF guidance during global feature extraction. Compared to the embeddings acquired by the newly introduced FEL, those
embeddings are more robust and consistent with the global image
semantics. By treating the local image embeddings as supervision
(without gradient backpropagation), we optimize the cosine similarity and Euclidean distance between 𝐹𝑣𝑓 𝑖𝑛𝑒 and 𝐹𝑣𝑙𝑜𝑐𝑎𝑙 , enhancing the
compatibility between pre-training model and FEL module.
For each image–text pair in the mini-batch of size N, the matching
probability between 𝐹𝑣𝑓 𝑖𝑛𝑒 and 𝐹𝑣𝑙𝑜𝑐𝑎𝑙 can be computed using the softmax
function:
exp(𝑠𝑖𝑚(𝑓𝑣𝑓𝑖 𝑖𝑛𝑒 , 𝑓𝑣𝑙𝑜𝑐𝑎𝑙 )∕𝜏)
𝑗
𝑝𝑖2𝑖
=
,
∑𝑁
𝑘𝑖,𝑗
𝑓 𝑖𝑛𝑒
exp(𝑠𝑖𝑚(𝑓
,
𝑓𝑣𝑙𝑜𝑐𝑎𝑙 )∕𝜏)
𝑡
𝑘=1
𝑖

Fig. 4. Details of the Multimodal Decoder. A five-layer decoder extracts
fine-grained embeddings via cross-attention. At every attention layer, the
entire original learnable tokens are re-added to the query tokens.

(4)

𝑘

where sim(𝐮, 𝐯) denotes the cosine similarity between 𝐮 and 𝐯, 𝜏 is the
temperature hyperparameter. The local matching probability 𝑝𝑖2𝑖
can
𝑘𝑖,𝑗
be viewed as the proportion of cosine similarity score between the finegrained image and the local image to the sum of all candidate pairs. We
employ the cross-entropy loss to optimize cosine similarity to associate
the embeddings across different modalities. Then, the Cosine Similarity
Loss (CSL) can be calculated by:

Unlike ALBEF or BLIP, which inject visual information into text
embeddings by inserting a cross-attention layer into traditional transformer block to obtain multimodal features, the FEL module introduces
a set of learnable tokens and a multimodal decoder. It aggregates texts
of variable lengths into semantically similar learnable tokens through
the cross-attention mechanism. Inspired by Kirillov et al. (2023), we design a variant of its mask decoder to serve as our multimodal decoder.
It consists of 𝐷 layers, as shown in Fig. 4. Each decoder layer performs
three steps: (1) cross-attention from learnable tokens (as queries) to
embedded word tokens, (2) self-attention on learnable tokens, (3) a
point-wise MLP updates learnable tokens. Each self/cross-attention and
MLP includes residual connection and layer normalization. The next
decoder layer processes the original embedded word tokens and the
updated learnable tokens from the previous layer. To enhance the
dependency of the decoder output on the original learnable tokens, the
entire original learnable tokens are re-added to the updated ones as
positional embeddings, whenever they participate in an attention layer.
Finally, we utilize a Squeeze-and-Excitation (Hu et al., 2018; Zhou
et al., 2025) layer, which contains two linear layers, a ReLU function
and a sigmoid function, to eliminate the distraction of unrelated features, referring to the output of this SE layer as the fine-grained text
embeddings 𝐹𝑡𝑓 𝑖𝑛𝑒 .

1 ∑ 𝑖2𝑖
𝑃 )),
𝐾 𝑘=1 𝑘
𝐾

𝐿𝐶𝑆𝐿 = −𝐾𝑌 𝑖2𝑖 log(Sof tmax(

(5)

where 𝐾 represents the number of local groups, 𝑃𝑘𝑖2𝑖 denotes the 𝐾th
fine-grained image-to-image similarity, and 𝑌 𝑖2𝑖 represents the ground
truth probability.
The Euclidean Distance Loss (EDL) aims to transfer the Euclidean
knowledge of the CLIP model to the FEL module, reducing the discrepancy between fine-grained text embeddings and the joint latent
embedding space by aligning the fine-grained image embeddings with
local image embeddings. We use mean squared error loss to optimize
the magnitude relationship between them:
2

𝐿𝐸𝐷𝐿 = ∥ 𝑓𝑣𝑙𝑜𝑐𝑎𝑙 − 𝑓𝑣𝑓 𝑖𝑛𝑒 ∥2 ,

(6)

The TES effectively bridges the modal gap between images and
text under the supervision of visual modality. CSL ensures semantic
alignment by optimizing the cosine similarity between fine-grained
image embeddings and local image embeddings. Meanwhile, EDL plays
a role in stabilizing the training process. Since image embeddings serve
as a semantic bridge, if the Euclidean distance is not optimized, the
target text embeddings extracted by the modality-shared FEL will be
distorted magnitude. This will make them incompatible with CLIP’s
joint embedding space, and result in a decline in the overall model
performance.
With the supervision of TES, knowledge from the CLIP space has
been successfully transferred to FEL. To mine fine-grained discrepancies, we introduce a Cross-Modal Alignment (CMA) loss, which employs the cross-entropy loss to optimize the cosine similarity between
fine-grained text embeddings and local image embeddings by:

3.3.2. Text extraction strategy
An intuitive approach is to directly align 𝐹𝑣𝑙𝑜𝑐𝑎𝑙 and 𝐹𝑡𝑓 𝑖𝑛𝑒 . However,
experimental results demonstrate that the performance is suboptimal.
We argue that while the local image embeddings are acquired from
an instance-level pre-trained model guided by ABF, the fine-grained
text embeddings are obtained by a newly introduced fine-grained embedding learning module. A simple learning strategy is insufficient
to enable the randomly initialized multimodal decoder to adapt to
the pre-training model during fine-tuning, thereby hindering its capacity to learn modality-invariant fine-grained representations. Notably, the FEL module is not an independent structure trained from
scratch, but further mines fine-grained feature information based on
the pre-trained CLIP model. The image–text joint latent space of CLIP,
obtained through large-scale image–text pair pre-training, exhibits excellent cross-modal semantic consistency. This property ensures that
if the learnable tokens can efficiently extract fine-grained information
from images, they should be equally capable of extracting corresponding fine-grained features from text. Therefore, we propose a new Text
Extraction Strategy (TES), which uses visual modality as a bridge to
enhance the semantic robustness of fine-grained text embeddings.
Specifically, we set the FEL as modality-shared, leveraging Modalityshared Learnable Tokens (MLT) and multimodal decoder to extract

1 ∑ 𝑡2𝑖
𝑃 )),
𝐾 𝑘=1 𝑘
𝐾

𝐿𝐶𝑀𝐴 = −𝐾𝑌 𝑡2𝑖 log(Sof tmax(

(7)

where 𝑃𝑘𝑡2𝑖 represents the matching probability of fine-grained text–
image pairs computed by Eq. (4), and 𝑌 𝑡2𝑖 represents the ground truth
probability.
3.4. Optimization improvements
To reduce the impact of multimodal decoder on the aligned pretraining image–text joint latent embedding space and enhance training
stability, we draw inspiration from J. Li et al. (2021), Carion et al.
5

X. Lin, X. Geng, S. Wu et al.

Computer Vision and Image Understanding 267 (2026) 104741

(2020) and Cheng et al. (2024) by initializing all MLT to zero and
introducing momentum distillation.
The MLT are designed to extract fine-grained features, but random
initialization may cause them to focus on noise and ignore effective
information. Given the unknown importance of patch/word tokens in
each local group for constructing fine-grained semantic embeddings,
we initialize MLT to zero, so the extracted features can be considered
average pooling feature of patch/word tokens.
Momentum Distillation (MoD), proposed by J. Li et al. (2021),
aims to improve the effectiveness of learning from noisy data. It is
reasonable to treat average pooling features as noisy data and introduce
momentum distillation to enhance the learning of MLT and the decoder.
The momentum MLT are a slow-moving average of the online MLT, capable of learning more stable information. Specifically, for CSL, we first
input momentum MLT and patch tokens into the multimodal decoder
to get momentum fine-grained image embeddings. After calculating
the similarity between momentum fine-grained image embeddings and
local image embeddings and computing the soft pseudo-target (𝑞 𝑖2𝑖 )
according to Eq. (4), the ground truth probability 𝑌 𝑖2𝑖 in Eq. (5) is
replaced with 𝑄𝑖2𝑖 . The momentum CSL (𝐶𝑆𝐿𝑀𝑜𝐷 ) loss is then defined
as:
1 ∑ 𝑖2𝑖
𝑃 )),
𝐾 𝑘=1 𝑘

annotated with 2 text descriptions. The dataset is split into training set,
test set, and validation set, containing 3701, 200, and 200 identities,
respectively.
We adopt the popular Rank-k (k = 1, 5, 10) as the evaluation metric
and report mean Average Precision (mAP) and mean Inverse Negative
Penalty (mINP) for comprehensive evaluation. Higher Rank-k, mAP and
mINP scores indicate better performance of the proposed method.
4.2. Implementation details
All experiments are conducted on a single RTX3090 24 GB GPU.
The visual encoder, i.e., CLIP-ViT-B/16, and text encoder, i.e., CLIP
text Transformer, are initialized using the pre-trained parameters of the
full CLIP model, while the multimodal decoder is initialized randomly.
All input images are resized to 384 × 128, and enhanced with random
horizontal flipping, random cropping with padding, and random erasing. For text input, we employ random masking and set the maximum
sequence length to 77. The dimension of the shared image–text joint
latent embedding space is set to 512. The multimodal decoder consists
of 5 layers, with the hidden size and number of heads set to 512
and 8, for each layer. Following IRRA, our model is trained using the
Adam optimizer for 60 epochs with a batch size of 64. The learning
rate strategy involves linear warmup and cosine learning rate decay.
The initial learning rates of backbone and proposed modules are set
to 1 × 10−5 and 1 × 10−3 , respectively. The temperature 𝜏 is set to 1
for CUHK-PEDES, 0.02 for ICFG-PEDES, and 0.1 for RSTPReid. The
momentum parameter for updating momentum MLT is set as 0.5, and
the momentum 𝛼 is set to 0.4. The layer for applying Attention Bias is
set to 8, and the topK is set to 32 during inference.

𝐾

𝐿𝐶𝑆𝐿𝑀𝑜𝐷 = 𝛼𝐿𝐶𝑆𝐿 − (1 − 𝛼)𝐾𝑄𝑖2𝑖 log(Sof tmax(

(8)

Similarly, the 𝐶𝑀𝐴𝑀𝑜𝐷 loss is obtained by replacing 𝑌 𝑡2𝑖 with 𝑄𝑡2𝑖
in Eq. (7), and is formulated as:
1 ∑ 𝑡2𝑖
𝑃 )),
𝐾 𝑘=1 𝑘
𝐾

𝐿𝐶𝑀𝐴𝐿𝑀𝑜𝐷 = 𝛼𝐿𝐶𝑀𝐴 − (1 − 𝛼)𝐾𝑄𝑡2𝑖 log(Sof tmax(

(9)

Since momentum distillation is solely applied to MLT, it will not
incur substantial additional memory usage.

4.3. Ablation study
To fully demonstrate the effectiveness of proposed modules in our
CDFM, we conduct comprehensive ablation studies on three public
benchmarks. We adopt a dual encoder (i.e., CLIP-ViT-B/16) fine-tuned
with global features alignment mentioned in Section 3.1 as Global
Baseline (No. 0). The experimental results are shown in Table 1.
Introduction of fine-grained embeddings. To quantify the impact
of adding fine-grained modules to the globally aligned model, we introduce a commonly used modality-shared extraction method (No. 1) as
our Baseline. The modality-shared method inputs MLT and embedded
patch/word tokens into the multimodal decoder separately, using the
cross-attention mechanism to get fine-grained embeddings. Compared
to the Global Baseline, the method results in a 1.7%, 1.15%, and
2.65% Rank-1 accuracy drop on the three datasets, respectively. By
introducing ABF into baseline, the method also results in a Rank-1
accuracy drop on the three datasets. We believe the main reason is
that the CLIP model is pre-trained with instance-level alignment on
large-scale image–text pairs, and non-robust local features introduce
irrelevant local noise to the backbone network, impairing the model’s
extraction capability.
Components analysis. To address the issue of feature extraction
capability degradation caused by the arbitrary introduction of finegrained information, we propose Text Extraction Strategy (TES) and
introduce Momentum Distillation (MoD) to enhance the modality robustness of the extracted fine-grained embeddings. When adding the
proposed MoD and ABF into our baseline (No. 3), we observe a performance gain of 1.57%, 0.09%, and 1.55% in terms of Rank-1 accuracy
on the three datasets, respectively. This result supports our statement in Section 3.4 that the fine-grained embeddings obtained by
initializing MLT with zeros, which are equivalent to average pooling
embeddings, are noisy. The introduction of momentum MLT forces the
model to learn the cross-modal similarity relationship from different
perspectives within the same batch via the generated pseudo-targets,
effectively reducing noise interference in the network. By introducing
TES as auxiliary supervision (No. 4) to No. 2, it achieves performance

3.5. Training and inference
The overall optimization objective for training is defined as:
𝐿 = 𝐿𝑖𝑑 + 𝐿𝑠𝑑𝑚 + 𝐿𝐶𝑆𝐿𝑀𝑜𝐷 + 𝐿𝐶𝑀𝐴𝑀𝑜𝐷 + 𝐿𝐸𝐷𝐿 ,

(10)

During inference, for a text query, we first compute the global similarity score 𝑆𝑔𝑙𝑜𝑏𝑎𝑙 between it and all image candidates. After selecting
the top-k candidates, we calculate their fine-grained score 𝑆𝑓 𝑖𝑛𝑒 for
re-ranking. The final similarity for text–image pairs is computed as
𝑆𝑔𝑙𝑜𝑏𝑎𝑙 + 𝑆𝑓 𝑖𝑛𝑒 . Since k can be set to a very small number, our inference
speed is much faster than methods that require dynamically computing
local feature similarity for all image–text pairs.
4. Experiments
4.1. Datasets and metrics
We evaluated our method on three publicly available and challenging TBPS datasets.
CUHK-PEDES (Li et al., 2017) is the first dataset collected for
text-based person search. It contains 40,206 images and 80,412 text
descriptions of 13,003 pedestrians. Following the official protocol, the
dataset is split into training, validation and test sets. The training set
includes 34,054 images and 68,108 text descriptions for 11,003 identities. The validation set includes 3078 images and 6158 text descriptions
for 1000 pedestrians. The test set includes 3074 images and 6156 text
descriptions for 1000 identities.
ICFG-PEDES (Ding et al., 2021) includes 54,522 images of 4102
identities collected from MSMT17. It is split into training and test
set, containing 34,674 image–text pairs for 3102 identities and 19,848
image–text pairs for 1000 identities, respectively. The text description
of this dataset is more fine-grained.
RSTPReid (Zhu et al., 2021) is constructed based on MSMT17. It
includes 20,505 images of 4101 pedestrians from 15 cameras, each
6

X. Lin, X. Geng, S. Wu et al.

Computer Vision and Image Understanding 267 (2026) 104741

Table 1
Ablation study on each component of CDFM on CUHK-PEDES, ICFG-PEDES and RSTPReid. ABF denotes the proposed Attention Bias-based visual forward process,
MoD refers to introducing momentum MLT to generate pseudo-targets for 𝐿𝐶𝑆𝐿𝑀𝑜𝐷 and 𝐿𝐶𝑀𝐴𝑀𝑜𝐷 , TES indicates the process of using local visual embeddings 𝐹𝑣𝑙𝑜𝑐𝑎𝑙
as supervised signals to promote the learning of MLT and the multimodal decoder.
No.

Methods

Components
ABF

TES

0
1
2
3
4

Global Baseline
Baseline
+ABF
+ABF+MoD
+ABF+TES

✓
✓
✓

✓

5
6

CDFM-G
CDFM-L

✓
✓

✓
✓

CUHK-PEDES
MoD

✓
✓
✓

ICFG-PEDES

RSTPReid

Rank-1

Rank-5

Rank-10

Rank-1

Rank-5

Rank-10

Rank-1

Rank-5

Rank-10

73.16
71.46
71.67
73.03
73.73

89.67
88.08
87.93
88.92
89.51

93.55
92.95
93.32
93.73
93.58

63.34
62.19
62.02
62.21
64.23

80.06
79.30
78.95
79.18
80.50

85.63
85.00
85.06
85.02
86.08

60.70
58.05
57.45
59.60
61.85

82.10
80.95
79.50
78.45
83.00

88.35
87.60
87.20
85.95
88.90

73.67
74.68

89.21
89.54

93.58
93.62

63.22
64.56

80.07
80.41

85.49
85.86

60.80
61.85

81.80
82.05

87.25
87.90

Table 2
Comparison about the size of fine-tuning parameters and training efficiency on CUHK-PEDES.
Methods

Rank-1

Trainable Parameters

Training Time per Batch

Training Throughput

Global Baseline
Baseline
+ABF
+ABF+MoD
+ABF+TES
CDFM

73.16
71.46
71.67
73.03
73.73
74.68

155 M
177 M
177 M
177 M
177 M
177 M

0.302 s
0.341 s
0.324 s
0.339 s
0.356 s
0.379 s

212.4 Samples/s
187.7 Samples/s
195.3 Samples/s
188.9 Samples/s
177.6 Samples/s
169.1 Samples/s

improvements of 2.27%, 2.04%, and 3.8% in Rank-1 accuracy over
Baseline on the three datasets, respectively. These results indicate
that TES successfully transfers the cosine relationship knowledge and
Euclidean knowledge of the latent space from the pre-training CLIP
to FEL. With TES, the fine-grained text embeddings extracted by FEL
are no longer solely reliant on text modalities but are aligned with
corresponding local visual representations through parameter sharing
and cross-modal supervision. Furthermore, we explore the effect of the
proposed modules on global alignment (No. 5). The results show that
with the assistance of ABF, TES, and MoD, the impact of local noise
and random initialization of modules on the robustness of fine-grained
embeddings is attenuated, and the global inference performance of
CDFM is fairly close to the Global Baseline. With the introduction
of local inference, our CDFM achieves a significant improvement of
3.22%, 2.37%, and 3.8% in Rank-1 accuracy over the baseline on the
three datasets, respectively.

4.3.2. Ablation study of ABF
We propose ABF to incorporate Attention Bias into the forward
process of the vision encoder for extracting local visual features without
compromising the global modeling capability of vision encoder. To verify the effectiveness of ABF, we compared the widely used TBPS image
partition method, Part-based Convolutional Baseline (PCB), applied to
different layers with our proposed ABF on CUHK-PEDES and ICFGPEDES. For a fair comparison, all methods were performed with the
same experimental configuration as CDFM apart from the local image
feature extraction design.
As shown in Table 3, to demonstrate the effectiveness of ABF, we
compared it with the vanilla PCB (No. 0 vs. No. 4). The vanilla PCB
slices the output image embeddings into equal-sized regions, and each
region is fed into an average pooling layer to obtain fine-grained image
embeddings, which are then aligned with the text. In terms of Rank1 accuracy, the performance of the vanilla PCB drops by 1.32% and
1.75% on CUHK-PEDES and ICFG-PEDES, respectively. Next, to further
validate ABF, we apply average pooling to the patch embeddings of
the 8th layer into 𝐾 groups, using the average pooling embeddings as
local [CLS] tokens (No. 1). The results show that replacing the vanilla
PCB with the 8th layer pooling method (No. 0 vs. No. 1) improves
Rank-1 accuracy by 0.83% and 1.13%, respectively. This indicates
that our proposed attention bias-based local feature extraction method
is more suitable than the traditional PCB method when utilizing the
transformer-based vision encoder as the visual backbone. Meanwhile,
we attempt to use maximum pooling to replace average pooling in PCB
(No. 2–3) to explore the effect of different local CLS settings on the
results. The experiments show that applying maximum pooling on the
8th layer on the CUHK dataset gives better results than applying it on
the last layer, while the ICFG dataset is not sensitive to the application
layer of maximum pooling. A comparison of experiments No. 1, No. 3,
and No. 4 demonstrates the efficacy of using copies of the [CLS] token
as local [CLS] tokens. The above results demonstrate that the [CLS]
token in the 8th layer already captures a comprehensive representation
of the whole image and can aggregate valid information within the
local token groups to facilitate cross-modal matching.

4.3.1. Analysis of complexity and performance
Table 2 presents the impact of different components on model performance, complexity, and training efficiency. Comparing the Global
Baseline with the Baseline reveals that the parameter-shared finegrained feature extraction strategy (Baseline), despite increasing the
parameter count to 177 M, leads to a degradation in Rank-1 accuracy
(73.16% to 71.46%). This suggests that merely increasing model capacity does not guarantee performance improvements. Incorporating
the ABF module into the Baseline yields slight improvements in both
accuracy and training throughput, validating its effectiveness. Further
combining the MoD module restores accuracy to a level comparable
to the Global Baseline (73.03%). The integration of the TES module
further improves accuracy to 73.73%; however, it introduces additional computational latency, resulting in a notable decrease in training
throughput.
By leveraging the effects of these modules, CDFM achieves the best
Rank-1 accuracy of 74.68%, outperforming the Baseline by 3.22%.
Although the training throughput decreases from 187.7 samples/s to
169.1 samples/s due to increased computational complexity, the number of trainable parameters remains constant at 177 M. This demonstrates that CDFM significantly enhances feature representation capabilities with only marginal computational overhead and no additional
storage burden.

4.3.3. Ablation study of TES
To evaluate the respective effects of the loss functions included in
TES, we retain all modules of CDFM and conduct ablation experiments
on the loss functions in TES, as shown in Table 4.
In the baseline configuration (No. 0), we directly align local visual embeddings with fine-grained text embeddings. Introducing the
7

X. Lin, X. Geng, S. Wu et al.

Computer Vision and Image Understanding 267 (2026) 104741

Fig. 5. Effects of three hyper-parameters, Attention Bias applied layer L, number of the part groups K and number of Decoder Layers D on CUHK-PEDES and
ICFG-PEDES datasets in terms of Rank-1 accuracy.
Table 3
Ablation study on different setting of ABF on CUHK-PEDES and ICFG-PEDES.
No.

0
1
2
3
4

Methods

CUHK-PEDES

PCB in 12th layer
PCB in 8th layer
MaxPooling in 12th layer
MaxPooling in 8th layer
CDFM

ICFG-PEDES

Rank-1

Rank-5

Rank-10

Rank-1

Rank-5

Rank-10

73.36
74.19
71.80
73.46
74.68

88.20
89.41
89.00
89.07
89.54

93.55
93.75
93.32
93.37
93.62

62.81
63.94
63.08
62.89
64.56

79.59
80.73
79.70
79.86
80.41

85.35
86.16
85.24
85.29
85.86

Table 4
Ablation study on different setting of TES on CUHK-PEDES and ICFG-PEDES.
No.

Methods

Components
𝐿𝐸𝐷𝐿

0
1
2
3

Baseline
+𝐿𝐸𝐷𝐿
+𝐿𝐶𝑆𝐿𝑀𝑜𝐷
CDFM

CUHK-PEDES
Rank-1

Rank-5

Rank-10

Rank-1

Rank-5

Rank-10

✓
✓

71.67
74.20
74.20
74.68

87.93
89.41
89.08
89.54

93.32
93.73
93.62
93.62

62.02
64.06
62.62
64.56

78.95
80.10
79.79
80.41

85.06
85.88
85.12
85.86

✓
✓

ICFG-PEDES

𝐿𝐶𝑆𝐿𝑀𝑜𝐷

Euclidean Distance Loss (𝐿𝐸𝐷𝐿 ) to constrain the learning of MLT and
decoders (No. 1) results in a Rank-1 accuracy improvement of 2.53%
on CUHK-PEDES and 2.49% on ICFG-PEDES. This demonstrates that
transferring the Euclidean knowledge of the pre-training CLIP model
to the FEL module is effective for modal embedding learning. Next, we
introduce the momentum Cosine Similarity Loss (𝐿𝐶𝑆𝐿𝑀𝑜𝐷 ) to constrain
cosine similarity relationships (No. 2). This approach improves Rank1 accuracy by 2.53% on CUHK-PEDES and 0.60% on ICFG-PEDES.
The results indicate that under the supervision of 𝐿𝐶𝑆𝐿𝑀𝑜𝐷 , FEL can
effectively learn embeddings with cosine discriminability from image
modalities. Finally, by applying 𝐿𝐶𝑆𝐿𝑀𝑜𝐷 and 𝐿𝐸𝐷𝐿 to the baseline,
Rank-1 accuracy improved significantly by 3.01% on CUHK-PEDES and
2.54% on ICFG-PEDES. This suggests that using the embeddings of the
original pre-training model as a supervision signal to assist the training
of MLT and the multimodal decoder makes the fine-grained embeddings
extracted by FEL more compatible with the pre-training space.

layers (D) on Rank-1 accuracy. As shown in Fig. 5(a), Rank-1 accuracy
fluctuates non-linearly when D ranges from 2 to 4. It peaks at 𝐷 = 5 on
both datasets and then tends to flatten. This behavior can be explained
by the following: when the number of layers is low, the decoder lacks
the capacity to model complex relationships, resulting in less robust
fine-grained embeddings. When the number of layers reaches 5, the
decoder achieves an optimal balance, yielding well-represented finegrained embeddings. However, beyond this point, additional layers lead
to overfitting, negatively affecting performance.
Evaluation of the number of part groups. The number of part
groups (K) influences the degree of fine-grained semantics. We conduct
experiments to examine the impact of different values of K on Rank1 accuracy. As shown in Fig. 5(b), the performance improves as K
increases, peaking at 𝐾 = 3 on CUHK-PEDES and at 𝐾 = 4 on ICFGPEDES. To maintain consistent settings on all datasets, we select 𝐾 = 4,
achieving a balance between accuracy and efficiency
Evaluation of the layer for applying attention bias matrix. We
also examine the effect of applying attention bias at different layers
on Rank-1 accuracy. As shown in Fig. 5(c), the accuracy trend is
similar across both datasets, with the exception of a dip at 𝐿 = 7,
and peaks at 𝐿 = 8. This finding is consistent with the explanation
in Section 3.4 and Ref. (Ghiasi et al., 2022), where it is noted that
Vision Transformers (ViT) exhibit localized attention in the lower
layers. When 𝐿 is set too low, the [CLS] token primarily captures color
and texture information without capturing comprehensive semantic
information, leading to ineffective learning in the divided token groups.
Conversely, when 𝐿 is set too high, the patch tokens have already
interacted extensively, losing their local characteristics and resulting in
suboptimal performance. Based on these observations, we select 𝐿 = 8
as the optimal layer for applying Attention Bias.

4.3.4. Analysis of the multimodal decoder
To demonstrate the advantages of our proposed Multimodal Decoder architecture, we compare it with traditional multimodal interaction architecture in Table 5 under our proposed CDFM setting. The
operation of re-adding the original MLT (as a residual connection)
brings performance gains on the Rank-1 metrics, regardless of whether
the local inference protocol is used. The residual connection of the
original MLT is an efficient operation to fuse multimodal features, as it
enhancing the connection between the MLT and both modalities.
4.4. Evaluation of the hyper-parameter
Evaluation of the number of decoder layers. We conduct several
experiments to investigate the effect of varying numbers of decoder
8


codex
还有三处需要补齐：CCFL 的后半段方法、Channel-aware 的三模块细节、两篇 CLIP 文本被截断的方法中段。补齐后我再做方法论拆解，避免把摘要里的贡献当成真正的创新链条。
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
/bin/zsh -lc "pdftotext -f 8 -l 14 'CCFL - Customized Client Federated Learning for Unsupervised Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
225:8

Y. Zheng et al.

At each client, the local model 𝜃 𝑘 is used to learn the hidden feature representation knowledge
from the local data, while the interactive model 𝜃 𝑖𝑘 , defined and distributed by the server, is
employed to transfer local knowledge to other clients and receive knowledge to guide the training
of the local model 𝜃 𝑘 . In practice, at the initialization stage, ImageNet pre-trained parameters are
uniformly used as the initial parameters for both the interactive and local models. However, in
the first round of training, the interactive model’s guidance for the local model training is either
meaningless or harmful. Thus, during the first round, the local model is trained directly on local data
without guidance from the interactive model. Ultimately, under the server’s control, the federated
learning process terminates after a pre-determined number of training rounds, and each client
outputs the optimal neural network model parameters achieved during training.
3.2

Unsupervised Learning Strategy Based on DBSCAN Clustering

In recent years, researchers have primarily employed three unsupervised clustering methods
in person Re-ID studies: k-means clustering [10], hierarchical clustering [8, 26], and DBSCAN
clustering [5, 18, 30]. Among these methods, k-means clustering typically requires estimating the
number of classes (i.e., the number of person identities) in the data and relies on domain adaptation
strategies to achieve a more accurate representation of pedestrian image samples. Due to these
limitations, k-means clustering is not suitable for the experimental setting described in the opening
paragraph of this section, which aims to minimize preprocessing and manual intervention on the
raw data.
In contrast, hierarchical clustering and DBSCAN clustering have been widely applied to person
Re-ID tasks, with substantial research supporting their effectiveness. In the DBSCAN algorithm,
hyperparameter selection can be guided by insights from existing studies without the need for
extensive statistical analysis of the dataset. On the other hand, hierarchical clustering does not
require the setting of specific hyperparameters.
In this section, our goal is to divide the dataset 𝑋𝑘 containing 𝑁𝑘 samples into 𝑀𝑘 clusters and
assign pseudo-labels to the unlabeled samples based on their cluster membership. By calculating
the centroid representation of each cluster, we obtain a set of cluster centroid features 𝐶, which
serve as classification weights for unsupervised neural network parameter learning.
To reduce the demand for training resources, we select ClusterNCE loss as the objective function
for this training strategy, in accordance with the realistic conditions assumed earlier. Specifically,
the ClusterNCE loss is represented as follows:

exp 𝑢 · 𝐶𝑖+ /𝜏
L𝐶 −𝑁𝐶𝐸 = − log Í𝑀
,
(2)
𝑘
𝑖=1 exp (𝑢 · 𝐶𝑖 /𝜏)
where 𝑢 represents the sample features, 𝐶𝑖+ denotes the positive cluster features, and 𝜏 is the
temperature parameter that controls the distance scaling.
In the proposed federated learning method, excluding the initial steps, the detailed training process of the unsupervised person Re-ID client based on DBSCAN clustering is illustrated in Figure 2.
To better leverage the knowledge from other datasets, this section utilizes features extracted by the
interactive model for clustering and generating pseudo-labels.
Specifically, based on general intuition, the interactive model, having received knowledge from
multiple clients, is more robust than the local model at the same training iteration. Therefore, this
section uses the interactive model to extract unlabeled features from the local data, applying the
DBSCAN clustering algorithm to generate pseudo-labels and cluster centroid vectors, which serve
as the basis for the ClusterNCE loss computation. In the implementation, the cluster centroid vectors
are normalized to avoid the impact of vector magnitude on loss calculation. Based on the analysis
by Zheng et al. [52], to ensure consistency in similarity measurement, cosine similarity loss is
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

Customized Client Federated Learning for Unsupervised Person ReID

225:9

Fig. 2. The person Re-ID training method based on the DBSCAN clustering algorithm generates new pseudolabels before each training round.

employed during feature knowledge distillation, as shown in Equation (3). This ensures uniformity
in calculating the difference between the similarity of features extracted by the local model 𝜃 𝑘 from
sample 𝑥𝑛 and features 𝑓𝑘𝑛 extracted by the interactive model 𝑓𝑖𝑛 , and the classification similarity
in ClusterNCE:
!
𝑁
𝑓𝑘𝑛 · 𝑓𝑖𝑛𝑇
1 Õ
L𝑐𝑜𝑠 =
1−
.
(3)
k𝑓𝑘𝑛 k · k 𝑓𝑖𝑛 k
𝑁 𝑛=1
Based on the description above, the DBSCAN algorithm-based training strategy designed for
customizable clients is detailed in Algorithm 2. In the first round of training, since both the
interactive model and local model use pre-trained parameters from the ImageNet dataset, the local
model training process is not guided by the interactive model. Instead, the local model is trained
directly using the generated pseudo-labels and local data.
3.3

Unsupervised Learning Strategy Based on Hierarchical Clustering

Although the DBSCAN algorithm is a powerful and flexible clustering algorithm, its performance
on some small-scale person Re-ID datasets is not particularly strong. In certain small datasets,
an identity may only have two image samples, requiring the 𝑚𝑖𝑛_𝑝𝑡𝑠 parameter to be set to 2 for
clustering with DBSCAN. However, this parameter setting can negatively affect Re-ID performance
on larger datasets. Therefore, we propose using different training strategies tailored to different
Re-ID datasets.
Unlike the k-means algorithm, hierarchical clustering does not require manually estimating
the number of identity classes in the dataset. Compared to DBSCAN, hierarchical clustering does
not involve parameters like 𝑒𝑝𝑠 and 𝑚𝑖𝑛_𝑝𝑡𝑠, which significantly impact clustering results. In
hierarchical clustering, each sample in dataset D is initially treated as an independent cluster, i.e,
𝑋𝑘 = {𝑐 1, 𝑐 2, 𝑐 3, ..., 𝑐𝑛𝑘 },

(4)

where 𝑛𝑘 represents the total number of samples in dataset 𝑋𝑘 . During the iterative process,
assuming that one sample is merged per iteration, the pair of samples 𝑐 𝑎 and 𝑐𝑏 with the highest
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

225:10

Y. Zheng et al.

Algorithm 2: DBSCAN Clustering-Based Strategy

similarity score is merged into the same cluster based on their feature similarity. At this point, the
representation of dataset 𝑋𝑘 is updated to:
𝑐 𝑎𝑏 = {𝑐 𝑎 , 𝑐𝑏 }

𝑋𝑘 = 𝑐 1, 𝑐 2, ..., 𝑐 𝑎−1, 𝑐 𝑎𝑏 , 𝑐 𝑎+1, ..., 𝑐𝑏 −1, 𝑐𝑏+1, ..., 𝑐𝑛𝑘 .

(5)

This merging step is repeated until the total number of clusters in the dataset 𝑋𝑘 is reduced to
the pre-defined number 𝑚, forming the final cluster partitioning representation:
𝑋𝑘 = {𝑐 1, 𝑐 2, 𝑐 3, ..., 𝑐𝑚 }, 𝑚 < 𝑛𝑘 .

(6)

During training, a centroid lookup table 𝑉 is constructed using the extracted sample features and
cluster partition results. Then, based on the sample features 𝑣 and pseudo-labels ˆ
𝑦, the probability
of sample 𝑥 belonging to cluster 𝑦ˆ is calculated, and the repulsion loss is formulated as shown in
Equation (7):
L𝐸𝑋 = − log

𝑒
Í𝑚



𝑉𝑦𝑇ˆ 𝑣/𝜏

𝑗=0 𝑒


.
𝑉 𝑗𝑇 𝑣/𝜏

(7)

Similar to clients using the DBSCAN algorithm, the local model is trained directly in the initial
rounds. In all subsequent rounds, the interactive model first guides the training of the local model,
and then, through Equation (3), the knowledge learned by the local model is transferred back to
the interactive model. However, unlike the clients using DBSCAN clustering, as shown in Figure 3,
clients using hierarchical clustering update pseudo-labels only after the training is completed. This
means that the training in a given round is determined by the features extracted using the neural
network parameters from the previous round.
The rationale behind this design is the smaller dataset size. As described earlier, the fusion of
interactive models relies on the number of samples in the dataset as a weighting factor, meaning
that small-scale datasets contribute less to the interactive model parameters. In this case, it is
equivalent to using parameters trained on other datasets for local fine-tuning of Re-ID. Since this
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

Customized Client Federated Learning for Unsupervised Person ReID

225:11

Fig. 3. The person Re-ID training method based on the hierarchical clustering algorithm trains the model
based on the pseudo-labels from the previous round and updates the pseudo-labels using the similarity
calculated from the features extracted by the trained model. At initialization, each sample is assigned an
independent and distinct pseudo-label.

section does not address non-IID data handling measures for such fine-tuning, this may introduce
significant domain bias during training.
Additionally, when the number of clusters to be merged per iteration in hierarchical clustering
is set to 𝑚(𝑚 > 0), after each iteration, the number of clusters in the dataset 𝑋𝑘 = {C1, C2, . . . , C𝑐 },
represented by clusters, will decrease by 𝑚, i.e., 𝑋𝑘 = {C1, C2, . . . , C𝑐 −𝑚 }. When the remaining
number of clusters drops below 𝑚, the remaining clusters are insufficient to support the next
iteration, potentially leading to clustering errors.
To ensure that clients using the hierarchical clustering algorithm can participate in long-term
training alongside clients using the DBSCAN algorithm, this section introduces a label resetting
step. When the remaining number of clusters is insufficient for the next clustering round, all
pseudo-labels are reset, and the clustering process is restarted. This method not only allows the
hierarchical clustering algorithm to continue training indefinitely but also leverages more accurate
feature representations extracted from a more fully trained neural network. This optimization of
the clustering results helps correct errors from previous clustering iterations and further improves
the training of model parameters, enhancing Re-ID performance.
In summary, the proposed client training process based on hierarchical clustering is outlined in
Algorithm 3.
3.4

Model Selection and Aggregation Strategy

To capture subtle differences and features in the data, neural networks utilize more parameters to
fit the data distribution. However, when the training samples in a dataset are insufficient or the data
distribution is imbalanced, a complex model structure combined with multiple training rounds may
lead to excessive overfitting of the neural network model to the training set data distribution. On
the other hand, simpler model structures or fewer parameters may result in the neural network’s
inability to adequately fit the data distribution, failing to capture the complexity and diversity of
features in large-scale datasets.
Thus, the traditional training approach where all clients share the same model structure within
a federated learning framework may not yield good performance on person Re-ID data. So, we
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

225:12

Y. Zheng et al.

Algorithm 3: Hierarchical Clustering-Based Strategy

introduce the design of the interactive model, allowing each client to select different model structures
that better align with their local data distribution. The number of data samples each client possesses
serves not only as a basis for selecting training strategies but also as a criterion for choosing
the neural network model structure. In this context, ResNet-50 and ResNet-34 are selected as
representatives of different model structures for training across various clients.
At the server, the model parameters from each client are weighted based on the amount of data
they hold, and the parameters of the different interactive models are fused. The fusion process can
be expressed by Equation (8):
𝜃𝑖 =

𝐾
Õ
𝑛

𝑘

𝜃 𝑖𝑘 ,

(8)

𝑁
𝑘=1

where 𝜃 𝑖𝑘 represents the parameters of the interactive model uploaded by the 𝑘th client in the
current round, 𝑁 = 𝑛 1 +𝑛 2 + . . . +𝑛𝑘 is the total number of data samples across all clients, 𝑛𝑘 denotes
the number of data samples held by the 𝑘th client, and 𝜃 𝑖 represents the updated parameters of the
interactive model for distribution in the next round.
4

Experiments

This section conducts simulated experiments on eight commonly used person Re-ID datasets to
evaluate the proposed federated learning algorithm. The datasets include DukeMTMC-ReID [36],
Market-1501 [51], CUHK03 [23], PRID2011 [17], CUHK01 [22], VIPeR [13], 3DPeS [2], and iLIDS
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

Customized Client Federated Learning for Unsupervised Person ReID

225:13

[43]. These datasets vary significantly in terms of shooting locations, equipment, sampling methods,
and sample scales. This diversity makes them suitable for simulating the non-IID phenomenon
commonly encountered in real-world person Re-ID tasks. In this context, each dataset is exclusively
assigned to a specific client, and the server does not have access to these datasets.
4.1

Experimental Settings

All our experiments were conducted on a graphics workstation equipped with an AMD Ryzen 9
5950X CPU, 128 GB of RAM, and an NVIDIA RTX A5000 GPU. The following hyperparameters were
utilized in this section to evaluate the model’s person Re-ID accuracy. The server was configured
to conduct a total of 300 rounds of interactive training to simulate a continuous training process.
For clients employing the DBSCAN clustering strategy, the batch size was set to 64, with each
batch containing 8 pseudo-identity labels, each of which included 8 pedestrian data samples.
Each training round consisted of one iteration, with the learning rate set at a specified value.
In Equation (2), the temperature parameter (𝜏) was set to 0.05.
As discussed in Section 3.2, optimal experimental configurations for the DBSCAN algorithm
were collected from various papers, including those by Luo et al. [30], Cho et al. [5], Hu et al. [18],
Dai et al. [7], Chen et al. [4], Li et al. [20], and Han et al. [15]. Specifically, the value of 𝑒𝑝𝑠 used in
the DBSCAN algorithm was set to 0.6, and 𝑚𝑖𝑛𝑝𝑡𝑠 was set to 4. For clients using the hierarchical
clustering strategy, the batch size was set to 16, with an initial training iteration count (𝑖𝑛𝑖𝑡 𝐸 ) of 20
and a normal training iteration count set to 2. The learning rate was also set to 0.05. In hierarchical
clustering, the imbalance clustering penalty parameter (𝑃) was set to 0.003. The dimensions of the
features extracted by the interactive model and the two local models were uniformly set to 2,048 to
facilitate knowledge transfer through knowledge distillation.
Regarding the training strategy and model architecture selection parameters, the following
settings were employed: Clients with more than 2,000 data samples used the DBSCAN clusteringbased training strategy, while those with fewer than 2,000 samples utilized the hierarchical clustering
strategy. For clients with more than 10,000 samples, ResNet-50 was employed as the local model,
whereas the remaining clients adopted the ResNet-34 model structure. Specifically, the DukeMTMCReID and Market-1501 datasets were trained using DBSCAN clustering with the ResNet-50 model,
while CUHK03 and PRID datasets utilized DBSCAN clustering with the ResNet-34 model. The
CUHK01, VIPeR, 3DPeS, and iLIDS datasets were trained using ResNet-34 with a hierarchical
clustering approach. To explore the optimal performance of the proposed method, the interactive
model was selected to use the ResNet-50 model structure.
4.2

Performance Comparison

In the unsupervised pedestrian Re-ID experiments shown in Table 1, this section first validates
the independent performance of CCFL (Collaborative Clustering Federated Learning) when only
the Market-1501 or DukeMTMC-ReID dataset is used. The training strategy for the client models
follows the SpCL [12] framework. To showcase the performance in a federated learning framework,
even without the presence of other clients, knowledge is transferred between the interactive model
and the local model via mutual distillation. Compared to SpCL, CCFL introduces some losses in
Rank-1 accuracy but maintains consistency in Mean Average Precision (mAP). This is because,
starting from the second round, the local model’s training is guided by the parameters from the
previous iteration, without additional knowledge from other datasets, leading to delayed parameter
updates, which affect model accuracy.
In federated pedestrian Re-ID methods, CCFL significantly outperforms existing approaches in
both Rank-1 accuracy and mAP. Specifically, compared to the FedUReID [55] method, where all
clients use hierarchical clustering, CCFL achieves a multiple-fold improvement in mAP on both
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.

225:14

Y. Zheng et al.
Table 1. Comparison of Results on Market-1501 and DukeMTMC-ReID Datasets with Other
State-of-the-Art Methods

Market-1501
Rank@1 Rank@5 Rank@10
BUC [26]
61.9
73.5
78.2
SSL [27]
71.7
83.8
87.4
HCT [48]
80.0
91.6
95.2
MMCL [42]
80.3
89.4
92.3
Standalone FUL
SpCL [12]
87.7
95.2
96.9
IICS [47]
89.5
95.2
97.0
MGCE-HCL [40]
92.1
CCFL (Ours)
80.3
90.7
93.0
FedReID [56]
Supervised FL
82.0
92.4
95.0
FedUReID [55]
65.2
77.8
82.2
FedUCC [44]
Unsupervised FL
86.5
94.5
96.7
CCFL (Ours)
83.6
94.6
96.1
Methods

Types

DukeMTMC-ReID
mAP Rank@1 Rank@5 Rank@10
29.6
40.4
52.5
58.2
37.8
52.5
63.5
68.9
56.4
69.6
83.4
87.4
45.5
65.2
75.9
80.0
72.6
81.2
90.3
92.2
72.9
80.0
89.0
91.6
79.6
82.5
73.7
76.9
83.2
86.2
58.8
74.2
85.0
88.4
34.2
51.0
62.4
67.6
65.5
78.8
87.2
89.8
80.7
81.5
87.2
89.3

mAP
22.1
28.6
50.7
40.2
65.3
64.4
67.5
68.5
55.6
29.5
60.5
74.2

mAP, Mean Average Precision.

Table 2. Rank@1 Results on Eight Datasets (%)
Dataset
Duke Market CUHK03 PRID CUHK01 VIPeR 3DPeS iLIDS Avg
FedUReID [55]
51.00 65.20
8.90
38.00
43.60
26.60 65.50 73.50 46.54
Simplification 52.78 58.70
8.14
36.00
37.65
25.00 63.82 72.45 44.32
FedUReID
Long-term
49.37 54.42
8.36
39.00
36.01
20.25 63.82 74.49 43.22
FedUCC [44]
78.80 86.50
9.60
58.90
78.30
31.30 68.90 74.70 60.88
Multi-strategy 81.55 81.56
33.43
48.00
70.88
45.57 77.64 77.55 64.52
CCFL
Multi-model 81.51 83.61
36.79
52.00
75.21
48.42 81.30 83.67 67.81

Table 3. mAP Results on Eight Datasets (%)
Dataset
Duke Market CUHK03 PRID CUHK01 VIPeR 3DPeS iLIDS Avg
FedUReID*
30.45 29.53
7.67
43.16
34.01
24.94 47.03 64.26 35.13
Simplification 30.85 31.89
7.25
42.40
36.22
29.05 50.66 62.63 36.37
FedUReID
Long-term
29.22 26.18
7.98
43.40
34.43
23.98 50.30 64.26 34.97
FedUCC [44]
60.50 65.50
9.70
63.10
75.30
36.70 50.90 59.70 52.68
Multi-strategy 74.57 79.38
35.88
53.03
67.88
50.00 66.55 73.48 62.60
CCFL
Multi-model 74.23 80.72
38.17
57.64
71.21
51.78 70.78 76.24 65.10

the Market-1501 and DukeMTMC-ReID datasets. Furthermore, compared to FedUCC [44], which
only uses DBSCAN clustering, CCFL shows around a 15% improvement in mAP on both datasets.
Additionally, CCFL surpasses the supervised federated learning method FedReID [56] in all metrics.
Tables 2 and 3 show the Rank-1 and mAP results for FedUReID, FedUCC, and CCFL (ours)
across eight pedestrian Re-ID datasets. To make the tables more concise, some dataset names are
abbreviated: DukeMTMC-reID (Duke), Market-1501 (Market), PRID2011 (PRID), and iLIDS-VID
(iLIDS). Since FedUReID did not report mAP results, we reproduced the method and used the mAP
corresponding to the Rank-1 accuracy closest to the results reported by Zhuang et al. [55] as an
approximation (marked with * in Table 3). Compared to hierarchical clustering-based FedUReID,
our method achieves significant improvements across all datasets, with an average Rank-1 accuracy
increase of over 20% and an mAP increase of about 30%.
ACM Trans. Multimedia Comput. Commun. Appl., Vol. 21, No. 8, Article 225. Publication date: August 2025.


exec
/bin/zsh -lc "pdftotext -f 3 -l 6 'Channel-aware feature mining network for Visible-Infrared Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

recognition through pose structure information; STAR (Jiang et al.,
2024) improves video-based VI-ReID accuracy by leveraging skeleton
data; CGMMNet (Xu and Zhao, 2024) addresses color discrepancies and
blurred boundaries using intermediate modality images and contour
maps; lastly, YYDS (Du et al., 2024) boosts re-identification performance by incorporating rough textual descriptions to fill in missing
color information in infrared images. Overall, using generated images as auxiliary information typically involves Generative Adversarial
Networks (GANs) for modality transformations (e.g., converting RGB
images to infrared images).
The introduction of auxiliary information can provide additional
contextual semantic information to compensate for modality differences, which indeed contributes significantly to improving model performance. However, this approach has inherent limitations: the generation of auxiliary information requires additional image processing
or natural language processing steps, which increases computational
burden; some auxiliary information also needs manual annotation,
raising data costs; furthermore, auxiliary information may introduce
redundant information or noise. In addition, modality differences between auxiliary information and target modalities lead to consistency
issues, which require additional alignment and fusion strategies.

channels of an RGB image with a single channel" as auxiliary modalities
to reduce modality differences. However, both approaches only use
channel operations for data expansion or modality adaptation, without
deeply exploring the identity-discriminative information contained in
individual channels. CHCR (Pang et al., 2023) provides a new perspective for channel-level processing: its inter-channel pseudo-label
refinement method, based on the principle that the three RGB channels
of the same sample correspond to the same identity, performs crossmodal clustering on each of the three channels with the infrared
modality separately. It evaluates consistency using the Intersection
over Union (IoU) to eliminate unreliable pseudo-labels, which not
only mitigates the information loss caused by traditional single-channel
conversion but also verifies the performance gain brought by channellevel features. Our visualization experiments (as shown in Fig. 1)
also confirm that some channels indeed contain highly discriminative
features crucial for identity recognition. Therefore, more efforts should
be devoted to exploring and utilizing channel-level features to enhance
the model’s representational capability and cross-modal recognition
accuracy.

2.2. Feature learning

The network (as shown in Fig. 2) adopts a dual-stream ResNet50 (He et al., 2016; Ye et al., 2020a) architecture to separately process
features from RGB and infrared (IR) images, which effectively addresses
the matching challenges in visible–infrared person re-identification (VIReID) caused by modality differences. First, the input visible–infrared
(VIS-IR) features pass through the Channel-Level Feature Optimization (CLFO) module, which directly extracts channel-level key features
closely related to identity recognition. To further improve the quality
of these channel features, we design the Channel-Level Feature Refinement (CLFR) module to suppress redundant or irrelevant information
and enhance discriminative features, thereby improving the accuracy
and robustness of feature representation.
On this basis, to enhance the model’s ability to understand and
describe input data, we introduce the Multi-Dimensional Feature Optimization (MDFO) module, which further explores and integrates feature information across multiple dimensions and layers. Through the
sequential processing of these three modules, the network can extract
richer and highly relevant key features from the original multi-modal
data, significantly strengthening the model’s discriminative ability.

3. Methodology

Feature learning methods aim to extract and learn meaningful feature representations directly from raw multi-modal data, rather than
relying on image transformations or additional auxiliary information.
Their core objective is to reduce discrepancies between different modalities through specific techniques — such as aligning features at the
pixel level or mapping multi-modal features directly into a shared
feature space — thereby improving the model’s generalization ability
and recognition accuracy. This approach emphasizes enhancing the
model’s understanding and processing capabilities for multi-source data
without introducing external information.
Pixel-level feature alignment methods operate directly on each pixel
in the image. For example, SAAI (Fang et al., 2023) achieves aggregation of potential semantic partial features by calculating the similarity
between pixel-level features and learnable prototypes; DCLNet (Sun
et al., 2022) proposes a dense contrastive learning network to perform
pixel-to-pixel dense alignment; CSL (Nie et al., 2024) designs a pixellevel color transformation module to learn the relationships between
different color channels. However, since these methods operate directly
at the pixel level, they are highly sensitive to image noise or subtle color
variations—this can significantly affect the model’s feature extraction
and recognition performance.
Another category of methods aims to project multi-modal features
into a shared feature space to learn a unified cross-modal representation. For example, MAUM (Liu et al., 2022) designs a one-way metric
learning approach that enhances memory capability by learning crossmodal metrics in two directions; RFM (Tan et al., 2023) introduces
a cross-modal center loss at the feature level to explore more compact intra-class distributions and employs a modality-aware spatial
attention module to better exploit texture regions. However, due to
significant differences between RGB and infrared images in information
capacity, representation, sharpness, and lighting conditions, simply
mapping them into the same feature space is insufficient to fully
eliminate modality gaps. This may also lead to the loss of important modality-specific information, which negatively impacts overall
recognition performance.
In addition, channel-level processing deserves attention in visible–
infrared person re-identification (VI-ReID), yet its significant value
remains underutilized. Current works mostly focus on preprocessing:
for example, Yang et al. (2022b), Wu and Ye (2023), Teng et al.
(2024), Dai et al. (2024), Zhang et al. (2024) generate color-invariant
images through random channel enhancement to expand the dataset;
CAJ (Ye et al., 2021a) uses images generated by "replacing the three

3.1. Channel-level feature optimization module
In the visible–infrared person re-identification (VI-ReID) task, RGB
and infrared (IR) images exhibit significant differences in channel
distribution due to their distinct imaging principles. For example, IR
images typically contain only a single thermal channel, while RGB images have three color channels. This modality asymmetry means certain
channels may carry stronger identity-discriminative information, while
others could be redundant or noisy.
To address this issue, we propose the Channel-Level Feature Optimization (CLFO) module. Its goal is to enhance the model’s ability
to extract discriminative identity features across modalities through
multi-level channel modeling and dynamic feature refinement. Unlike
traditional attention mechanisms (e.g., SE, CBAM) that focus solely on
channel importance estimation, CLFO integrates depthwise separable
convolution, group normalization, and a learnable residual connection
into a unified framework. These components work together to achieve
fine-grained channel-level feature modeling at the early stage of feature extraction, effectively mitigating the impact of channel imbalance
between RGB and IR images.
This design allows CLFO to not only adaptively highlight informative channels but also maintain computational efficiency and training
stability—key requirements for VI-ReID tasks. In what follows, we
3

P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

Fig. 2. The architecture of the proposed Channel-Aware Feature Mining Network (CAFMNet) and the Multi-Dimensional Feature Optimization (MDFO) module
is described in this section. Some other module architectures are shown in Fig. 3.

independently for each input channel (groups = 𝐶in ) to avoid channel
information mixing; (2) Pointwise convolution: Employs a 1 × 1 kernel to
fuse cross-channel information, ensuring the output channel dimension
𝐶out matches 𝐶in for subsequent residual connection.
After DSConv1 , group normalization (GN, denoted as GN1 , with
8 groups to stabilize training across modalities) and ReLU activation
are applied. The resulting intermediate feature tensor 𝐴1 serves as a
core link in the CLFO module: it locates at the frontend of the CLFO
feature processing pipeline, bridging the raw backbone features (𝑋)
and subsequent refined operations (e.g., secondary depthwise separable
convolution, SEBlock).
𝐴1 retains the spatial resolution of 𝑋 (due to stride=1 and padding=1)
while enhancing local discriminative patterns (e.g., pedestrian contours, clothing textures) that are invariant to RGB-IR modality gaps.
Its mathematical definition is:
(
(
))
𝐴1 = ReLU GN1 DSConv1 (𝑋)
(1)
where 𝐴1 ∈ R𝐵×𝐶out ×𝐻×𝑊 (with 𝐶out = 𝐶in ) to ensure dimension
consistency for subsequent feature fusion.
To recalibrate channel-wise importance and suppress modalityspecific noise (e.g., thermal artifacts in IR images), the feature tensor 𝐴1 is further processed by a second depthwise separable convolution (DSConv2 ) and group normalization (GN2 ), followed by a
Squeeze-and-Excitation (SE) Block:
(
(
))
𝑍2′′ = SE GN2 DSConv2 (𝐴1 )
(2)

Fig. 3. The specific design of the (Fig. 3 left (a)) channel-level feature
optimization module as well as the (Fig. 3 right (b)) channel-level feature
refinement module.

where the SE block implements global channel attention via adaptive
average pooling (to 1 × 1 spatial resolution) and two fully connected
layers (reduction ratio=16, consistent with the module’s hyperparameters).
Finally, a learnable residual connection fuses 𝑍2′′ with the transformed input feature Res(𝑋) (via 1 × 1 convolution and GN), and ReLU
activation yields the CLFO module output 𝑂:
(
)
𝑂 = ReLU 𝛼 ⋅ 𝑍2′′ + (1 − 𝛼) ⋅ Res(𝑋)
(3)

describe the detailed architecture and mathematical formulation of the
CLFO module.
Let the input feature tensor of the Channel-Level Feature Optimization (CLFO) module be 𝑋 ∈ R𝐵×𝐶in ×𝐻×𝑊 , where: 𝐵 denotes the batch
size of input multi-modal (RGB/IR) features; 𝐶in represents the number
of input channels, consistent with the output channel dimension of
the dual-stream ResNet-50 backbone (adopted in the CAFMNet architecture); 𝐻 × 𝑊 denotes the spatial resolution of the feature maps
(e.g., 56 × 56 for intermediate features in ResNet-50).
To extract modality-invariant features with low computational cost,
we first apply a depthwise separable convolution (DSConv1 ) operation,
which consists of two sequential steps: (1) Depthwise convolution: Uses
a 3 × 3 kernel, stride=1, and padding=1, performing spatial filtering

Here, 𝛼 ∈ [0, 1] is a learnable parameter (initialized to 0.5) that
dynamically balances refined features and raw input information, enhancing the robustness of cross-modal feature representation.
Unlike traditional attention modules that treat all channels equally,
CLFO explicitly models the interaction between modality-specific characteristics and channel-wise importance. By performing channel-aware
4

P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

feature optimization at an early stage of feature extraction, CLFO effectively mitigates the impact of modality-specific noise and imbalanced
channel information, thereby enhancing the consistency of cross-modal
feature representations—crucial for accurate matching in VI-ReID.

original 𝑂𝑐𝑎 to the reconstructed feature, generating the local–global
fused feature map 𝑂𝑙𝑔 ∈ R𝐵×𝐶×𝐻×𝑊 :
(
)
𝑂𝑙𝑔 = 𝑊 𝑓norm ⋅ 𝑔(𝑂𝑐𝑎 ) + 𝑂𝑐𝑎
(5)
where + denotes element-wise addition (residual connection), ensuring
original local features are preserved while integrating global dependencies.
After local–global information fusion, a depthwise separable convolution block is employed to efficiently refine both spatial and channel
depth
features. It first applies a 3 × 3 depthwise convolution (Conv3×3 ,
groups = 𝐶) to 𝑂𝑙𝑔 , extracting spatial patterns without mixing channel
point
information. A 1 × 1 pointwise convolution (Conv1×1 ) then fuses crosschannel information and restores the channel dimension to 𝐶, resulting
in the spatial-refined feature map 𝑂𝑑𝑠 ∈ R𝐵×𝐶×𝐻×𝑊 . This lightweight
operation balances computational cost and feature expressiveness, with
the formulation:
(
)
point
depth
𝑂𝑑𝑠 = Conv1×1 Conv3×3 (𝑂𝑙𝑔 )
(6)

3.2. Channel-level feature refinement module
Although we use the Channel-Level Feature Optimization (CLFO)
module to directly mine identity-relevant information at the channel
level, the CLFO module only performs coarse-grained extraction of
channel-level features, which may retain irrelevant noise (e.g., background clutter or modality-specific artifacts). To further enhance the
discriminative capability of channel-wise features in complex crossmodal scenarios — such as visible–infrared person re-identification (VIReID), where RGB and infrared (IR) images exhibit significant differences in channel distribution — we design an additional Channel-Level
Feature Refinement (CLFR) module.
The CLFR module is specifically designed to suppress modalityspecific noise (e.g., thermal artifacts in IR images or color distortion in
low-light RGB images) while preserving identity-discriminative features
(e.g., pedestrian contours, clothing textures) across both RGB and IR
modalities. It achieves this through a multi-stage refinement process
involving enhanced channel attention, non-local feature fusion (for
local–global interaction), depthwise separable convolution, and spatial attention with residual learning—each stage explicitly addressing
challenges posed by modality asymmetry (e.g., single-channel IR vs.
three-channel RGB) and background clutter in ReID scenarios. Let the
input feature map of the CLFR module be 𝑂 ∈ R𝐵×𝐶×𝐻×𝑊 , where: 𝐵
denotes the batch size (number of sample pairs in one training batch);
𝐶 represents the number of feature channels, consistent with the output
channel dimension of the preceding CLFO module; 𝐻 × 𝑊 denotes the
spatial resolution of the feature map (height × width, e.g., 28 × 28 for
intermediate features in the dual-stream ResNet-50 backbone).
The CLFR module first applies an enhanced channel attention mechanism to aggregate global channel information and suppress redundant
channels. It uses adaptive average pooling (AvgPool1×1 ) to process 𝑂,
compressing the spatial dimensions of each channel into a 1×1 scalar to
capture global statistical information. Two successive 1×1 convolutions
(denoted Conv1×1 ) — with the first reducing the channel dimension
to 𝐶∕reduction (reduction ratio = 32) and ReLU activation, and the
second restoring the channel dimension to 𝐶 with Sigmoid activation
— generate channel attention weights 𝐶𝐴 ∈ R𝐵×𝐶×1×1 . Element-wise
multiplication of 𝐶𝐴 and 𝑂 yields the channel-refined feature map
𝑂𝑐𝑎 ∈ R𝐵×𝐶×𝐻×𝑊 , whose mathematical formulation is:
(
(
(
𝑂𝑐𝑎 = 𝑂 ⊙ 𝜎 Conv1×1 ReLU Conv1×1 (
(4)
))))
AvgPool1×1 (𝑂)

Next, to suppress background clutter (e.g., walls, vehicles) and
focus on identity regions, the CLFR module applies a spatial attention mechanism. It first performs max pooling (MaxPool) and average
pooling (AvgPool) along the channel dimension of 𝑂𝑑𝑠 , resulting in
MaxPool(𝑂𝑑𝑠 ) ∈ R𝐵×1×𝐻×𝑊 (extracting salient spatial regions) and
AvgPool(𝑂𝑑𝑠 ) ∈ R𝐵×1×𝐻×𝑊 (capturing global spatial context). These
two pooled features are concatenated to form sa_input ∈ R𝐵×2×𝐻×𝑊 ,
and a 7 × 7 convolution (Conv7×7 ) — followed by Sigmoid activation
— generates spatial attention weights 𝑆𝐴 ∈ R𝐵×1×𝐻×𝑊 . Element-wise
multiplication of 𝑆𝐴 and 𝑂𝑑𝑠 enhances identity-relevant spatial regions,
yielding the spatial-refined feature map 𝑂𝑠𝑎 ∈ R𝐵×𝐶×𝐻×𝑊 :
(
(
(
𝑂𝑠𝑎 = 𝑂𝑑𝑠 ⊙ 𝜎 Conv7×7 concatenate MaxPool(𝑂𝑑𝑠 ),
(7)
)))
AvgPool(𝑂𝑑𝑠 )
where concatenate(⋅, ⋅) denotes channel-wise concatenation.
Finally, to prevent overfitting (critical for limited VI-ReID datasets),
𝑂𝑠𝑎 undergoes Dropout regularization (Dropout, probability = 0.1),
randomly setting feature elements to 0 to enhance model robustness.
A residual connection adds the dropout-processed feature Dropout(𝑂𝑠𝑎 )
to the original input 𝑂 of the CLFR module, preserving original feature
information and integrating multi-stage refined features to generate the
final output feature map 𝑂′ ∈ R𝐵×𝐶×𝐻×𝑊 :
𝑂′ = Dropout(𝑂𝑠𝑎 ) + 𝑂

(8)

To summarize the entire computational process of the CLFR module,
we define stage-specific functions: 𝐹𝑐𝑎 (⋅) (enhanced channel attention,
outputting 𝑂𝑐𝑎 from 𝑂), 𝐹𝑛𝑙 (⋅) (non-local feature fusion, outputting
𝑂𝑙𝑔 from 𝑂𝑐𝑎 ), 𝐹𝑑𝑠 (⋅) (depthwise separable convolution, outputting 𝑂𝑑𝑠
from 𝑂𝑙𝑔 ), 𝐹𝑠𝑎 (⋅) (spatial attention, outputting 𝑂𝑠𝑎 from 𝑂𝑑𝑠 ), and
𝐹drop (⋅) (Dropout regularization, outputting Dropout(𝑂𝑠𝑎 ) from 𝑂𝑠𝑎 ).
The entire process can be expressed as a composite function:
(
(
( (
))))
𝑂′ = 𝐹drop 𝐹sa 𝐹ds 𝐹nl 𝐹𝑐𝑎 (𝑂)
+𝑂
(9)

where ⊙ denotes element-wise multiplication (broadcasting 𝐶𝐴 to
match the spatial dimensions of 𝑂), 𝜎(⋅) denotes the Sigmoid function,
and ReLU(⋅) denotes the ReLU activation function.
Subsequently, to capture long-range semantic relationships (e.g., correlations between a pedestrian’s head and legs) that local receptive
fields fail to cover, the CLFR module introduces a Non-Local Block for
local–global feature fusion. This block first uses three 1×1 convolutions
(denoted 𝜃(⋅), 𝜙(⋅), 𝑔(⋅)) to transform 𝑂𝑐𝑎 into intermediate features:
𝜃(𝑂𝑐𝑎 ) ∈ R𝐵×𝐶inter ×𝐻×𝑊 and 𝜙(𝑂𝑐𝑎 ) ∈ R𝐵×𝐶inter ×𝐻×𝑊 (for similarity
calculation, with 𝐶inter = 𝐶∕2 to balance performance and efficiency)
and 𝑔(𝑂𝑐𝑎 ) ∈ R𝐵×𝐶inter ×𝐻×𝑊 (for feature aggregation). The spatial
dimensions of 𝜃(𝑂𝑐𝑎 ) and 𝜙(𝑂𝑐𝑎 ) are flattened to 𝑁 = 𝐻 × 𝑊 , and
the similarity matrix 𝑓 ∈ R𝐵×𝑁×𝑁 is computed as 𝑓 = 𝜃(𝑂𝑐𝑎 )𝑇 ⋅ 𝜙(𝑂𝑐𝑎 ).
Softmax normalization of 𝑓 (along the last dimension) yields 𝑓norm , and
global feature aggregation is implemented by multiplying 𝑓norm with
the flattened 𝑔(𝑂𝑐𝑎 ). After reshaping to restore spatial dimensions, a
1 × 1 convolution (𝑊 (⋅)) — followed by batch normalization (BN) —
restores the channel dimension to 𝐶. A residual connection adds the

Through this series of operations, the CLFR module accurately
retains identity-relevant information and suppresses irrelevant or noisy
features introduced by modality discrepancies. By explicitly modeling
both channel and spatial importance in a cascaded manner, CLFR significantly improves the quality and effectiveness of features, providing
more discriminative representations for subsequent VI-ReID tasks.
3.3. Multi-dimensional feature optimization module
In DEEN (Zhang and Wang, 2023), a Multi-stage Feature Aggregation (MFA) block was proposed to extract channel-wise and spatialwise feature representations from multi-level features. However, the
MFA block primarily focuses on aggregating features across stages,
with limited exploration of cross-dimensional feature interactions. In
5

P. Li, Z. Du, L. Zhang et al.

Computer Vision and Image Understanding 262 (2025) 104552

visible–infrared person re-identification (VI-ReID) — where significant modality discrepancies exist between RGB and infrared images
— such single-path aggregation strategies often fail to fully exploit
the discriminative information embedded in both channel and spatial
dimensions.

3.4. Multi-loss optimization
To guide the training process effectively, we adopt a multi-loss
learning strategy that combines four complementary loss functions:
the cross-perspective mutual learning loss 𝐿𝑐𝑝𝑚 and the orthogonal
constraint loss𝐿𝑜𝑟𝑡 from DEEN (Zhang and Wang, 2023), along with the
widely used cross-entropy loss (Luo et al., 2019) and triplet loss (Hermans et al., 2017). These losses capture different aspects of the learning
objective, including feature discrimination, modality alignment, and
identity classification accuracy. The network is trained in an end-to-end
manner by minimizing the weighted sum of these four loss components.
Specifically, we set the balancing coefficients 𝜆1 and 𝜆2 to 0.8 and 0.01,
respectively. The total loss function is formulated as:

Therefore, we propose a Multi-Dimensional Feature Optimization
(MDFO) block, which goes beyond conventional feature aggregation
by explicitly modeling the complex relationships among features across
multiple dimensions to address this limitation. Unlike traditional modules that process features independently within individual dimensions,
the MDFO block performs cross-dimensional collaborative optimization, facilitating more comprehensive feature learning and refinement.
This design is especially advantageous for VI-ReID, where achieving
robust and discriminative feature representations is critical for bridging
modality gaps and capturing identity-specific patterns.

𝑡𝑜𝑡𝑎𝑙 = 𝑐𝑒 + 𝑡𝑟𝑖 + 𝜆1 𝑐𝑝𝑚 + 𝜆2 𝑜𝑟𝑡 ,

This carefully designed loss combination ensures that each individual component contributes meaningfully to the overall optimization,
without dominating or being overshadowed by others during training.

Specifically, at each stage of the backbone network, we consider
two types of input features: a low-level feature map 𝑥𝑙 and a high-level
feature map 𝑥ℎ . To better model global dependencies, we introduce two
non-local blocks into the architecture. These blocks allow the model to
capture long-range correlations across spatial positions and channels—
information critical for identifying discriminative body parts under
varying imaging conditions.

4. Experiments
4.1. Datasets and evaluation metrics
Datasets. The proposed CAFMNet is evaluated on two challenging
large-scale VI-ReID datasets. SYSU-MM01(Wu et al., 2017), which includes 491 person-ID images from four RGB and two NIR cameras.
The LLCM (Zhang and Wang, 2023) is the largest VI-ReID dataset
with 1,064 person-ID images from nine low-light cameras. It presents
challenges such as illumination variations, motion blur, pose changes,
camera view changes, occlusion, and low resolution. The evaluation
modes for LLCM include VIS-to-IR and IR-to-VIS.
Evaluation Metrics. We adopt widely used metrics in person reidentification tasks, including Rank-k accuracy and mean Average Precision (mAP). All results are reported as averages over 10 independent
experimental runs to ensure statistical reliability.

We first apply three 1 × 1 convolutional layers to transform 𝜓𝑞1 ,
1
𝜓𝑘 , and 𝜓𝑣1 into compact feature representations: 𝜓𝑞1 (𝑓ℎ ), 𝜓𝑘1 (𝑓𝑙 ), and
𝜓𝑣1 (𝑓𝑙 ). This transformation not only reduces computational cost but
also preserves essential semantic information:
(
)
𝑀 𝐶 = 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥 𝜓𝑞1 (𝑓ℎ ) × 𝜓𝑘1 (𝑓𝑙 ) .
(10)
This channel-level similarity matrix 𝑀 𝐶 captures the relative importance of different channels through softmax normalization, offering
a more accurate reflection of inter-channel relationships compared to
traditional methods. Based on 𝑀 𝐶 , we then perform channel-level
feature aggregation:
(
)
𝑓ℎ𝑐 = 𝜔𝑠 𝜓𝑣1 (𝑓𝑙 ) × 𝑀 𝐶 + 𝑓ℎ .
(11)

4.2. Implementation details
All models and experiments are implemented using the PyTorch
framework on a single Tesla P100 GPU. We employ a ResNet-50 backbone pretrained on ImageNet for feature extraction. The learning rate
is initialized with a warm-up strategy: it starts at 0.01 and gradually
increases to 0.1 within the first 10 epochs. It is then decayed to 0.01
at epoch 20, further reduced to 0.001 at epoch 60, and finally set to
0.0001 at epoch 120, remaining constant until the final training epoch
(epoch 130). Input images are resized to a fixed size of 3 × 384 × 144.
During training, we sample 6 identities per mini-batch, each contributing 4 visible (VIS) and 4 infrared (IR) images. In the testing phase, only
modality-shared features are used for performance evaluation.
To enhance generalization, we apply common data augmentation
techniques during training, including random horizontal flipping and
random erasing (Zhong et al., 2020).

By fusing low-level details with high-level semantics in a weighted
manner, this operation enhances both the richness and discriminability
of the resulting features.
Next, we perform a similar aggregation in the spatial domain using
𝑓ℎ𝑐 and the original low-level feature map 𝑓𝑙 :
(
)
𝑓ℎ𝑠 = 𝜔𝑠 𝜓𝑣2 (𝑓𝑙 ) × 𝑀 𝑠 + 𝑓ℎ𝑐 ,
where 𝑀 𝑠

denotes the spatial similarity matrix, and 𝜔𝑠

(14)

(12)
and 𝜓𝑣2

are
implemented via 1 × 1 convolutions. Finally, the refined feature 𝑓ℎ𝑠 is
further processed through a dual-attention mechanism that adaptively
recalibrates feature responses in both channel and spatial dimensions:
(
)
𝑍 = SA CA(𝑓ℎ𝑠 ) ,
(13)

4.3. Comparison with state-of-the-art methods

where CA and SA denote the channel attention and spatial attention
modules, respectively. The channel attention module uses global average and max pooling followed by two 1 × 1 convolutions, while spatial
attention computes feature maps based on max and average values
across channels.

We compare the proposed CAFMNet model with recent state-ofthe-art VI-ReID methods that have been evaluated on public VI-ReID
datasets, including SYSU-MM01 and LLCM.
Comparison on the SYSU-MM01 dataset. Comparison on the SYSUMM01 dataset. As shown in Table 1, the results on the SYSU-MM01
dataset demonstrate that the proposed CAFMNet achieves the best
performance among all compared methods. Specifically, under the allsearch mode in SYSU-MM01, CAFMNet achieves a Rank-1 accuracy
of 77.49% and an mAP of 74.19%. Under the indoor-search mode, it
achieves a Rank-1 accuracy of 84.95% and an mAP of 87.09%. These
results clearly validate the effectiveness of the proposed approach.

The key innovation of this attention mechanism lies in its ability
to dynamically emphasize important feature regions—especially those
most relevant for identity discrimination under cross-modal settings.
By integrating these operations into a unified framework, the MDFO
block enables deep feature optimization across multiple dimensions,
effectively enhancing the representational power of the model for
challenging VI-ReID tasks.
6


exec
/bin/zsh -lc "pdftotext -f 4 -l 9 'CLIP-Based Camera-Agnostic Feature Learning for Intra-Camera Supervised Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
TAN et al.: CLIP-BASED CAMERA-AGNOSTIC FEATURE LEARNING FOR ICS ReID

their efficacy across various domains. For example, Contrastive Language-Image Pretraining (CLIP) [14] model, which
employs the InfoNCE loss [28] function to jointly train
text and image encoders, resulting in significant performance
improvements in numerous downstream tasks. Additionally,
to further tap into CLIP’s potential, CoOp [29] introduces
prompt learning, aiming to uncover the implicit textual cues
within images, effectively migrating CLIP to a broader range
of downstream tasks. Within the realm of ReID, CLIP has been
extensively applied. For instance, CLIP-ReID [30], by aligning
image and textual information within a singular embedding
space, reinforces the connection between image features and
related textual descriptions. CCLNet [31] establishes learnable
cluster-aware prompts for person images and generates text
descriptions to assist subsequent unsupervised visible-infrared
person re-identification training.
However, the immense potential of CLIP in facilitating
semi-supervised person ReID learning has yet to be explored.
In this paper, we fully integrate CLIP with the ICS ReID task
to construct a CCAFL framework, offering new insights for
semi-supervised ReID.
D. Adversarial Learning
The application of adversarial learning in person
re-identification can be traced back to the use of Generative
Adversarial Networks (GANs) [32] to generate realistic
person images. For example, Jiang et al. [33] proposed a
GAN-based method that performs selective sampling of
generated data to bridge the gap between domains and
enrich the feature space. In recent years, the application of
adversarial learning has extended beyond image generation
and has been widely applied to various aspects of person
re-identification. For instance, in unsupervised domain
adaptation for person re-identification, CAWCL [34] employs
a Gradient Reversal Layer (GRL) [35] to align the distribution
of each camera. However, using traditional domain adversarial
learning to eliminate camera styles can negatively impact the
model’s ability to recognize pedestrians. In clothing change
person re-identification, CAL [36] proposed a clothing-based
adversarial loss to decouple clothing-independent features.
In contrast to these methods, we propose an inter-camera
adversarial loss that penalizes the model’s ability to predict
the same identity under different cameras, thereby enabling
the model to extract inter-camera agnostic features.
E. Semi-Supervised Visible-Infrared Person ReID
Visible-infrared person re-identification (VI-ReID) aims to
match individuals captured in one modality with their counterparts in another. However, the development of existing
VI-ReID methods remains limited due to the lack of annotated infrared data, which complicates the training process.
To address the challenges of large-scale cross-modality data
annotation, several semi-supervised VI-ReID methods [37],
[38], [39] have been proposed. These approaches leverage
both labeled and unlabeled data to learn modality-invariant
and identity-discriminative features. For example, DPIS [37]
introduces a dual pseudo-label interactive self-training method

4103

that integrates pseudo-labels generated by different models into a hybrid pseudo-label, effectively mitigating noise
issues. DMA [38] proposes a dual modality-aware alignment
model that preserves discriminative identity information while
suppressing misleading information. MUN [39] employs a
cross-modality learner and an intra-modality learner to generate robust auxiliary modalities, addressing both modality
discrepancies and intra-class variations effectively.
Currently, mainstream semi-supervised VI-ReID methods
primarily focus on two scenarios: one where only visible light
images are annotated, and another where partial annotations
are provided across both modalities. These methods mainly
concentrate on addressing the issue of cross-modality label
alignment. However, unlike these semi-supervised VI-ReID
scenarios, intra-camera supervised person re-identification
represents a unique setup, where only labels within each
camera are considered, and cross-camera label associations are
ignored. Although our method is currently applied primarily
to intra-camera supervised ReID scenarios, the intra-camera
discriminative contrastive loss module and cross-camera
adversarial learning module in our approach—which leverage
partial labels to learn discriminative features—can also be
effectively applied to semi-supervised visible-infrared ReID.
This approach is particularly suited to scenarios where visible
and infrared cameras have independently annotated identity
labels. One of the main changes is how to conduct robust
feature learning. Therefore, our method provides new insights
and challenges for exploring broader semi-supervised VI-ReID
scenarios.
III. M ETHODOLOGY
A. Overview
Based on the ICS ReID problem, the training dataset
only contains intra-camera IDs and lacks inter-camera IDs.
Therefore, in this case, a dataset consisting of C cameras can
be represented as D = {D1 , D2 , . . . , DC }. Specifically, the
images of persons from the c-th camera can be represented
as Dc = {(xi , y j , c)}, where xi indicates the i-th person
image under this camera, y j (0 ≤ j < Nc ) represents the
corresponding label, and Nc denotes the number of person
IDs under this camera. For instance, in the Market-1501 [40]
dataset, which contains six cameras, there are 751 pedestrian IDs under supervised conditions, meaning these IDs
are associated across different cameras. However, in the ICS
setting, the IDs for each camera are independently annotated,
with the number of pedestrian IDs for each camera being
D = {652, 541, 694, 241, 576, 558}, resulting in a total of
3,262 global IDs. Although multiple cameras may capture the
same pedestrian, their global IDs remain distinct. Therefore,
each person sample in the training set carries three labels:
the intra-camera ID, the camera label, and the global ID.
Moreover, since the same pedestrian might be captured by
multiple cameras, they have different IDs assigned to them
depending on which camera they were captured by. Thus,
our primary objective is to learn feature representation for
individuals across different cameras.
Recently, the CLIP model, trained on large-scale datasets,
has demonstrated remarkable proficiency in matching

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.

4104

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

Fig. 3. The framework of our CCAFL. Left: Through prompt learning paradigms, we generate text descriptions corresponding to the labels of each person’s
image within a camera. This provides semantic supervision information for subsequent intra-camera and inter-camera learning. Upper: In the intra-camera
learning phase, we construct a hybrid memory for each camera, storing both the central features and instance features of pedestrians. By employing an
intra-camera discriminative loss, we enhance the discriminability of pedestrian features within the same camera. Lower: In the inter-camera learning phase,
we obtain cross-camera association IDs through a cross-camera association step. We then build a memory that stores prototype features of associated
pedestrians, aiding the model in learning pedestrian features across different cameras. Additionally, we introduce a global ID classifier and incorporate
inter-camera adversarial learning to mitigate the impact of camera discrepancies.

image-text descriptions. Its image encoder captures complex
and rich visual features, while the text encoder provides
enhanced semantic information. Building upon this, we have
developed a learning framework that integrates CLIP with
ICS-ReID to discern intra- and inter-camera ID identities,
as illustrated in Fig. 3. The framework consists of three
training steps: intra-camera pre-defined prompt learning, intracamera learning, and inter-camera learning. Through this
three-stage training process, the model can deeply explore
pedestrian information from different camera angles and effectively guide the establishment of more accurate cross-camera
ID associations.
B. Intra-Camera Pred-Defined Labels Prompt Learning
The current research on ICS ReID commonly adopts a
two-stage learning approach, namely intra- and inter-camera
stages. In the inter-camera learning phase, pseudo-labels
are often assigned to IDs from different camera views
using similarity-matching. However, due to variations in perspective, these pseudo-labels tend to be inaccurate, which
can hinder the learning process. To address this issue,
we integrate text encoder and prompt learning mechanisms
from the CLIP framework to generate descriptive textual
prompts corresponding to individual identities. This incorporation provides valuable semantic constraints for subsequent
inter-camera learning and serves as an adjunct in rectifying pseudo-labels, thereby enhancing the overall recognition
performance.

More specifically, we first define a textual prompt based
on the predefined labels within each camera, described as “a
photo of [X ]1 [X ]2 . . . [X ] M person,” where M represents the
number of learnable text tokens. Subsequently, we input the
implicit textual prompts and ID images into the CLIP model
to optimize the text tokens [X ]. Through this method, we can
obtain textual representations associated with pedestrian IDs
within each camera. It is important to note that, in this training
phase, we freeze CLIP model’s image and text encoder while
utilizing image-to-text and text-to-image losses to learn the
text tokens:
 
 
C
exp s f iv , f pt /τ
X
1 X
log P B
Li2t = −
  , (1)
t
v
|Pi |
k=1 exp s f i , f k /τ
p∈Pi
c=1
 
 
C
exp s f pv , f it /τ
X
1 X
Lt2i = −
log P B
  , (2)
t
v
|Pi |
k=1 exp s f k , f i /τ
p∈Pi
c=1
where Pi = { p|y p = yi , p ∈ {1, 2, . . . , B}} represents the
index set of positive image samples, s(, ) represents the cosine
similarity between text features and image features, C is the
number of cameras, B is the batch size, and τ denotes the
temperature factor. Ultimately, the loss of intra-camera predefined label prompt learning is:
L pr ompt = Li2t + Lt2i .

(3)

By minimizing L pr ompt , we can learn the corresponding
implicit textual descriptions for the IDs within each camera.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.

TAN et al.: CLIP-BASED CAMERA-AGNOSTIC FEATURE LEARNING FOR ICS ReID

4105

In the subsequent stages, we will utilize these textual descriptions to provide stronger semantic supervision for the model,
thereby enhancing its generalization capability.

C. Intra-Camera Discriminative Learning
The primary challenge of ICS ReID lies in annotating
IDs within each camera’s view and establishing cross-camera
ID associations through the analysis of inter-camera characteristics. Intra-camera learning, therefore, can be considered
a fully supervised problem within a multi-task framework.
However, the distribution of sample numbers for each ID
within a camera is uneven, with most IDs having only a
limited set of training samples. This imbalance can lead to
model bias toward learning prominent camera-style features,
rather than ID features. Additionally, the lack of cross-camera
ID information highlights the significance of intra-camera
learning as a preparatory phase for subsequent cross-camera
correlation.
Considering these factors, we focus on the centroid features
of each pedestrian within each camera, as well as the hard
positive and hard negative samples for that pedestrian within
the camera, as shown in Fig. 4. This compels the model
to learn more accurate intra-camera pedestrian features. This
approach also emphasizes the differences between pedestrian
IDs within the camera view, providing more reliable features
for subsequent inter-camera association steps.
1) Intra-Camera Hybrid Memory Banks Initialization:
Firstly, we initialize the intra-camera hybrid memory bank
using an image feature encoder. All image features are
assigned to the corresponding camera memory based on different camera IDs. Then, for each image under each camera’s
pre-defined labels, the centroid features of each pedestrian ID
within the camera are stored in the intra-camera centroid memory through averaging, to learn the pedestrian features within
each camera. Additionally, the instance features corresponding
to each pedestrian ID within each camera are stored in the
intra-camera instance memory to learn more discriminative
information. The mean feature of the predefined labels of
pedestrians within a camera is calculated as follows:
µic =

1 X
f (x),
|Nci |
i

(4)

x∈Nc

where µic represents the mean feature of pedestrian ID i
within camera c, Nci denotes the set of all images belonging
to pedestrian ID i within camera c, and f (x) represents
the features of image x after being processed by the image
encoder. Therefore, the memory bank Mcintra is initialized with
the mean features of the pedestrian IDs, while Miintra is initialized with the instance sample features corresponding to those
pedestrian IDs. For intra-camera instance memory, we employ
a real-time instance feature memory update strategy. In each
iteration, we directly replace Miintra in the memory with the
current mini-batch instantaneous feature f x :
Miintra

 
y ← f (x).

(5)

Fig. 4. Illustration of Lintra1 and Lintra2 . The same color indicates that
all samples originate from the same camera, while different shapes represent
different pedestrian IDs within the camera.

2) Optimization: In each training iteration, the features
stored in the aforementioned intra-camera hybrid memory
bank are updated using different strategies.
For the memory bank Mcintra :
µic ← αµic + (1 − α)e
µic ,

(6)

where µ
eic denotes the average features of the camera c insider
ID i in each batch, α is the momentum updating factor.
This update mechanism ensures that the features in the memory bank consistently reflect the latest training information,
thereby enhancing the accuracy and stability of the model
in learning intra-camera pedestrian features, as shown in
Fig. 4 (a). To this end, given a query image feature f (x),
we propose an intra-camera centroid contrastive loss function,
which is formulated as:

C
X
exp s( f (xi ), µ+ )/τ
log P K
Lintra1 = −
 , (7)
c
i
j=1 exp s( f (x i ), µc )/τ
c=1
where µ represents the centroid feature for each ID in the
c-th intra-camera memory, K c represents the total number of
pedestrian IDs in the camera, and C is the number of cameras.
Through the above loss, we can effectively bring the sample
closer to the centroid feature of its corresponding ID while
pushing it away from other ID centroid features within the
same camera.
However, when faced with challenging samples within the
camera, such as similar apparel or shared backgrounds, this
approach may result in poor classification of IDs within the
camera. Moreover, as the dataset expands and the number
of individuals per camera increases, the model’s recognition
performance may be adversely affected. Therefore, we further
enhance inter-class separability and intra-class compactness
by merging all instance features under each ID, as shown in
Fig. 4 (b). Specifically, for a query image xi , we examine
the relationship between the hardest positive sample and the
hardest negative sample from other IDs stored in the memory.
By calculating loss across different cameras, we reduce the
distance between samples and their centroids as well as
relevant hard positive samples while increasing distances from
other hard negative samples.
Lintra2 = −

C
X

exp(s( f (x), m +
har d )/τ )
log P K
,
j
c
c=1
j=1 exp(s( f (x), m har d )/τ )

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.

(8)

4106

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

where m +
har d represents the hardest positive sample characteristics retained in memory Miintra , it demonstrates the cosine
similarity is the lowest compared to all instance features under
j
this pedestrian ID. Conversely, m har d is the hardest negative
sample feature which shows the highest cosine similarity when
compared to all other instance features from different IDs
under the camera.
3) Intra-Camera Image-Text Alignment: During the second
stage of intra-camera learning, we freeze the text encoder and
only train the image encoder. Specifically, for each person ID
under each camera, we obtain the corresponding text features
by inputting prompts into the text encoder, derived from the
first stage. Meanwhile, we input the image x into the image
encoder to obtain the raw features f v (x). Subsequently, we use
the loss Li2tce to constrain the image features f v (x) to be
close to the corresponding text features f t (y), while being
distant from the text features of other identities:
Lintra
i2tce =

Kc
C X
X

exp(s( f v (x), f t (yi )))
, (9)
−qz log P K
c
v
t
z=1 exp(s( f (x), f (yz )))
c=1 i=1

where qz is a smoothed ID label. Notably, our loss function
is computed for features within each camera independently,
ensuring that these calculations are performed separately for
each camera.
4) The Loss for Intra-Camera Learning: In summary,
the proposed intra-camera loss Lintra is composed of the
intra-camera discriminative loss and the intra-camera imagetext alignment loss, defined as follows:
L I C DL = λLintra1 + (1 − λ)Lintra2 ,

IDs i and j. The edge e(i, j) is defined as follows:


 1, dist (i, j) < T ∧ c(i) ̸ = c( j)
e(i, j)=
∧ i ∈ N1 ( j, c(i)) ∧ j ∈ N1 (i, c( j));


0, other wise.

where dist (i, j) represents the distance between the centroid
features of the i-th and j-th ID IDs, c(i) indicates the camera
to which the IDs belong, and T is the distance threshold.
N1 ( j, c(i)) designates the nearest neighbor of the j-th ID with
the i-th ID under the c(i) camera. Using the conditions defined
above, a sparsely connected graph is constructed. Then, based
on similarity, we identify all connected components and assign
inter-camera pseudo-labels to IDs.
2) Inter-Camera Memory Banks Initialization: Based on the
successful application of contrastive learning in the field of
person re-identification [10], [24], [41], we employ a prototypical contrastive learning paradigm for inter-camera learning.
First, upon completion of intra-camera learning, we generate
pseudo-labels for IDs using the aforementioned inter-camera
association algorithm. Next, we compute the mean features
of these samples based on their corresponding pseudo-labels
and directly initialize the inter-camera memory bank. This
approach provides a stable starting point for prototypical
contrastive learning. Consequently, our inter-camera memory
stores the mean features of the associated IDs across different
cameras, facilitating the learning of a person’s appearance
characteristics under varying camera conditions. The memory
features are updated using online batch features in a moving
average manner, as described by the following formula:

(10)

M[y] ← α M[y] + (1 − α) f x .

where λ is the balancing factor between the two losses.
Lintra = L I C DL + Lintra
i2tce .

(11)

D. Inter-Camera Learning
Through the aforementioned intra-camera learning, our
model effectively identifies each person within the camera’s
view. However, the abundant learning information of IDs
across cameras has yet to be fully utilized. Therefore, in the
inter-camera learning process, we have devised an alternating
strategy consisting of cross-camera ID association steps and
inter-camera contrastive learning steps to facilitate the model’s
acquisition of view-invariant ID features.
1) Inter-Camera Association: The ICS ReID approach differs from fully unsupervised ReID, which relies on clustering
algorithms to obtain pseudo-labels directly. In the case of
intra-camera IDs, assigning the same pseudo-labels is not
feasible. Therefore, we employ an ID association algorithm
based on connected components proposed in [12]. Specifically,
we impose two constraints on the clustering process: 1) under
the in-camera supervised condition, positive matches among
IDs within each camera should not exist, and 2) a maximum of
one positive match is allowed per camera. We then constructed
an undirected graph G = ⟨V, E⟩ for associations, where
the vertex set V represents the accumulated IDs across all
cameras, and the edge set E represents a positive pair between

(12)

(13)

3) Optimization: To further learn the prototype features
of IDs under different cameras, the inter-camera prototypical
contrastive loss is defined as follows:
exp(s( f (x), M[y])/τ )
,
L I PC L = − log P Z
i=1 exp(s( f (x), M[ j])/τ )

(14)

where Z represents the number of IDs associated in each epoch
of inter-camera correlation.
4) Inter-Camera Image-Text Alignment: Considering the
significant variations in illumination and background, IDs
across cameras often exhibit notable feature differences,
leading to noisy inter-camera association labels. Hence,
we combine the text description learned in the first stage with
the inter-camera prototypical contrastive learning, leveraging
additional semantic supervision information to assist the model
in improving the accuracy of inter-camera ID correlation and
learning the prototype features of IDs across cameras. Specifically, we define an image-to-text contrastive loss function:
Linter
i2tce =

exp(s( f v (x), f t (yi )))
,
−qz log P Z
v (x), f t (y )))
exp(s(
f
z
z=1
i=1

Z
X

(15)

5) The Loss for Inter-Camera Learning: The total loss in
inter-camera learning can be summarized as follows:
Linter = L I PC L + Linter
i2tce .

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.

(16)

TAN et al.: CLIP-BASED CAMERA-AGNOSTIC FEATURE LEARNING FOR ICS ReID

Fig. 5.
The probability distributions of intra-camera samples processed
by the global classifier in Market-1501 dataset are as follows: The left
figure illustrates that, after a certain number of training epochs, the samples
with true intra-camera labels exhibit a distinct probability distribution with
a sharp peak, indicating that the classifier effectively distinguishes different
individuals across different cameras. The right figure shows that, after initiating inter-camera adversarial learning, inter-camera association labels are
obtained through an inter-camera association algorithm. Samples with the
same pseudo-label across different cameras are treated as positive examples,
which enhances the probability distribution of the same person across cameras
in the global classifier, resulting in multiple peaks.

4107

predicting which specific camera a pedestrian image feature
originates from.
2) Learning Inter-Camera Agnostic Features: In the second
step, we fix the parameters of the global ID classifier and compel the network to learn camera-irrelevant features. To achieve
this, we penalize the model’s prediction capability regarding
global IDs. Specifically, while the classifier mentioned above
can distinguish different pedestrians across cameras using
global IDs, our goal is to train the global ID classifier to
not distinguish the same identity across different cameras.
Therefore, we introduce an Inter-Camera Adversarial Loss
(ICAL), a multi-positive class classification loss where all
categories belonging to the same identity but different cameras
are considered positive classes, as shown in Fig. 5. Notably, the
same pedestrians across different cameras are identified using
pseudo-labels obtained via the above inter-camera association
algorithm. The ICAL is formulated as follows:
L I C AL = −

NG
N X
X

q(g)

i=1 g=1






log 



exp f (xi ) · ϕg /τ
 P

,
exp f (xi )· ϕg /τ +
exp f (xi ) · ϕ j /τ 


E. Inter-Camera Adversarial Learning
Our model can recognize pedestrian identities across different cameras through the previously described intra- and
inter-camera contrastive learning. However, during intercamera learning, the variance in pedestrian feature distributions between cameras introduces label noise, and the
value of predefined intra-camera label information is not
fully exploited. To address this issue, we propose InterCamera Adversarial Loss (ICAL). ICAL penalizes the model’s
prediction capabilities across different cameras, forcing the
backbone network to extract inter-camera agnostic features.
To achieve this, we introduce a new global ID classifier
based on camera data, appended to the network, as shown
in Fig. 3. Each training iteration consists of two optimization
steps:
1) Training the Inter-Camera Global ID Classifier: First,
we establish a classifier C C (·), where each class corresponds to a global ID. We optimize this global ID classifier
by minimizing the classification loss LG I D , defined as the
cross-entropy loss between the predicted pedestrian C C ( f (xi ))
and the global label yiG . We perform L2-normalization on the
model’s output features f (xi ) and denote the L2-normalized
weights of the j-th global ID classifier as ϕ j . We detach these
weights before inputting them into the global ID classifier
to ensure that the classifier’s training does not influence the
model itself. Consequently, LG I D can be expressed as:


N
exp f (xi ) · ϕ y G /τ
X
i
LG I D = −
log
,
(17)
N
G

P
i=1
exp f (xi ) · ϕ j /τ
j=1

where N is the batch size, N G is the number of global ID
classes in the training set, and τ is a temperature parameter.
By using global IDs as labels during training, our classifier can
distinguish pedestrians across different cameras, effectively

j∈G i−

(18)
where G i− represents the set of global IDs of the negative
centroids of the query. q(g) is the cross-entropy loss weight
for the g-th global ID category.
Given that the same pedestrians across different cameras are
identified using pseudo-labels obtained via the inter-camera
association algorithm, which contains noise compared to
pre-defined intra-camera labels, to enhance the model’s intercamera pedestrian recognition capability without significantly
compromising intra-camera accuracy, we define q(g) as:

ϵ

1 − ϵ + , if g = yiG


G
ϵ
q(g) =
(19)
,
if g ̸ = yiG and g ∈ G i+

G


0,
if g ∈ G i−
where G i+ represents the set of global IDs of the positive
centroids of the query, G is the number of elements in G i+ , and
G i− represents the set of global IDs of the negative centroids
of the query. ϵ is a hyperparameter with a range of 0 < ϵ ≤ 1.
Importantly, our goal is to optimize both ICAL and the
inter-camera loss concurrently. Linter and L I C AL are correlated in learning camera-irrelevant features. When using Linter
alone, the model tends to learn simple samples (highly similar
pedestrian features across different cameras) in the early
epochs of optimization and gradually distinguishes harder
samples (pedestrian features with low similarity due to factors
like pose, lighting, and background changes). L I C AL aims to
narrow the feature gap for the same identity across different
cameras with pseudo-labels, which is similar to the goal of
Linter . To avoid local optima caused by directly minimizing
L I C AL from the beginning, we execute inter-camera adversarial learning after a certain number of epochs. Consequently,

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.

4108

IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 35, NO. 5, MAY 2025

TABLE I
D ETAILS OF E ACH DATASET. I N T RAINING S ET, #Camera, #I D, AND #I mage A RE THE N UMBER OF C AMERAS , ID S , AND I MAGES , R ESPECTIVELY.
U NDER THE ICS S ETTING , #I D I C S R EPRESENTS THE N UMBER OF ACCUMULATED ID S . I N T ESTING S ET, THE N UMBER
OF ID S AND I MAGES IN G ALLERY AND Q UERY S ETS A RE A LSO L ISTED

our model improves inter-camera pedestrian recognition capability and effectively learns camera-irrelevant features without
significantly compromising intra-camera recognition accuracy.
F. A Summary of the Objective Function
According to the above description, we name the CLIPbased camera-agnostic feature learning framework CCAFL.
The overall loss function of CCAFL is:


if epoch ≤ E intra ,
 Lintra + LG I D ,
LCC AF L = Linter + LG I D ,
if E intra < epoch,


Linter + LG I D + L I C AL , if E adv ≤ epoch.
(20)
where E intra and E adv denote the number of epochs for
intra-camera and inter-camera adversarial learning, respectively. Through the proposed learning process, the CCAFL
algorithm not only effectively utilizes the textual information generated by CLIP to provide semantic supervision
for subsequent learning but also leverages pre-defined labels
within each camera for supervised intra-camera learning and
adversarial inter-camera learning. This approach enables the
model to learn camera-agnostic features, thereby enhancing the
quality of inter-camera clustering. The entire training details
of our CCAFL are provided in Algorithm 1.
IV. E XPERIMENTS
A. Datasets and Evaluation Metrics
Our method is validated on three large-scale person re-identification (ReID) datasets: Market-1501 [40],
DukeMTMC-ReID [42], and MSMT17 [43]. Following the
ICS setting, we re-annotate individuals within each camera
in the training set and add accumulation labels. Table I
summarizes the specifics of the datasets under the ICS setup,
including the number of cameras, IDs, and images in the
training, gallery, and query sets. Additionally, we provide
the accumulated total identity number under intra-camera
supervision (#I D I C S ).
In terms of evaluation metrics, we adopt cumulative matching characteristics (CMC) [44] including Rank-1, Rank-5, and
Rank-10 as well as mean Average Precision (mAP).
B. Implementation Details
We adopt the ResNet50 model pre-trained on CLIP as our
feature extractor. All input images are resized to 256×128
and subjected to data augmentation techniques such as random

Algorithm 1 CLIP-Based Camera-Agnostic Feature
Learning
Data: Intra-camera person label training dataset D;
Input: Backbone image encoder Fv and text encoder Ft
initialized from CLIP, the epoch number
num_epochs, the train batches num_batches;
Output: Trained model Fv and Ft ;
// intra-camera prompt learning period
for n in [1, init_epochs] do
Optimize [X ]1 [X ]2 . . . [X ] M with Eq. (3);
end
// intra-camera training period
for n in [1, intra_epochs] do
Extract features from D by encoder Fv ;
Calculate the centroid features for pedestrian IDs per
camera.
Initialize every camera memory Mc with Eq. (4);
for iter in [1, num_batches] do
Sample P × K query images from D;
Compute loss with Eq. (11) and Eq. (17);
Updating intra-camera memory banks by with
Eq. (5), Eq. (6);
end
end
// inter-camera training period
for n in [intra_epochs + 1, nums_epochs] do
Extract features from D by encoder Fv ;
Cluster the features to generate K clusters for
inter-camera K clusters.
Initialize inter-camera memory Mg with Eq. (4);
for iter in [1, num_batches] do
Sample P × K query images from D;
Compute loss with Eq. (16), Eq. (17) and
Eq. (18);
Updating inter-camera memory bank by with
Eq. (13);
end
end

flipping, cropping, and erasing [45]. During the prompt learning phase, we learn tokens represented as [X ]1 [X ]2 . . . [X ] M .
We utilize the Adam optimizer [46] with a learning rate of
0.00035, which is adjusted using a cosine annealing policy.
Our training batch size is 64, and the training process lasts
for 60 epochs. For the subsequent training phase, we set the
batch size to 128 and employ the PK sample [17]. Within
each mini-batch, we sample images based on the labels of
each camera. We randomly select 16 IDs, with each ID having

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:58:38 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 3 -l 6 'CLIP-driven fine-grained mining for text-based person search.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
X. Lin, X. Geng, S. Wu et al.

Computer Vision and Image Understanding 267 (2026) 104741

et al. (2024) proposed a CLIP-guided fusion framework for pedestrian attribute recognition, exploring CLIP’s capability in understanding fine-grained information such as attribute labels. Jiang and Ye
(2023) directly fine-tuned the CLIP model, and the remarkable results demonstrated that the visual-language pretraining model has high
compatibility with TBPS.

The output of vision encoder is projected into the image–text joint
latent embedding space using a learnable image projection, denoted
as 𝐹𝑣 = {𝑓𝑣𝑐𝑙𝑠 , 𝑓𝑣1 , … , 𝑓𝑣𝑁 } ∈ 𝑅(𝑁+1)×𝐶 , where 𝑓𝑣𝑐𝑙𝑠 represents the global
embeddings of the input image.
Text Encoder. For text description, the input sentence 𝑇 is first
tokenized via a lower-cased Byte Pair Encoding (BPE) with a vocabulary size of 49,152, and then processed by a Transformer with masked
self-attention modified by Radford et al. (2021). Since the text description is of variable length, we prepend and postpend learnable [SOS]
and [EOS] tokens to the sequence, setting the maximum sequence
length to 77. Similar to image branch, a learnable text projection is
used to project text features into the latent space, denoted as 𝐹𝑡 =
{𝑓𝑡SOS , 𝑓𝑡1 , … , 𝑓𝑡𝑀 , 𝑓𝑡EOS } ∈ 𝑅(𝑀+2)×𝐶 , where 𝑓𝑡EOS serves as the global
text embedding, and {𝑓𝑡𝑖 }𝑀
are word-level embeddings.
𝑖=1
Global alignment. Our proposed CDFM mines fine-grained local
relations under the alignment of global image–text embeddings. We
optimize the global embedding via common ID Loss (Zheng et al., 2020)
and SDM Loss (Jiang and Ye, 2023). ID loss is a cross-entropy loss
that promotes the transfer of CLIP knowledge in TBPS tasks by treating
text-based person search problem as classification problem through a
modality-shared classification head. SDM loss utilizes KL divergence
to measure the difference between the cross-modal cosine similarity
matrix and the ground truth matrix in a mini-batch. This loss function
enlarges the similarity of matched pairs and reduces the similarity of
mismatched pairs. The SDM loss is formulated as:

2.2. Vision-language pre-training
Inspired by the success of unimodal pretraining models in Transformer-based language pretraining, such as BERT (Devlin et al.,
2018) and Vision Transformer (Dosovitskiy et al., 2020), many works
have attempted to exploit large-scale image–text datasets for pre-training to enhance the relevance of image–text modalities. Pre-training
and fine-tuning have emerged as a mainstream paradigm for learning
multimodal representations.
Existing VLP work can be categorized into single-stream, dualstream, and mixed architectures based on the backbone architecture:
• Single-stream framework: This consists of a single shared Transformer encoder (Shu et al., 2022; H. Li et al., 2021) where
images and texts are concatenated and fed into the encoder to
extract representations. This reduces the number of parameters
but introduces a large number of computations.
• Dual-stream framework: Although this framework may lack the
ability to model complex interactions across modalities, its independent encoders demonstrate remarkable performance in image–
text retrieval tasks. For example, CLIP utilizes 400 million image–
text pairs trained with cross-modal contrast loss, enabling its
visual encoder to contain textual semantic information.
• Mixed architecture: This approach first extracts and aligns the
representations of respective modalities with a dual-stream architecture, then feeds the image and text features into a multimodal
encoder for fusion. ALBEF (J. Li et al., 2021) employs this architecture, and BLIP (Li et al., 2022) proposes a multimodal
Mixture of Encoder–Decoder (MED) with both understanding and
generation capabilities for deeper interaction between image and
text modalities.

𝐿𝑠𝑑𝑚 =

𝐵 𝐵
𝐵 ∑
𝐵
𝑝𝐼2𝑇
𝑝𝑇𝑖,𝑗2𝐼
∑
𝑖,𝑗
1 ∑ ∑ 𝐼2𝑇
(
𝑝𝑖,𝑗 log(
)+
𝑝𝑇𝑖,𝑗2𝐼 log(
)),
𝐵 𝑖=1 𝑗=1
𝑦𝑖,𝑗 + 𝜖
𝑦𝑖,𝑗 + 𝜖
𝑖=1 𝑗=1

(1)

𝐼2𝑇 ∕𝑇 2𝐼

where 𝑝𝑖,𝑗
represents the proportion of the image-to-text or textto-image cosine similarity score in a mini-batch. 𝜖 is a small number
to avoid numerical problems, 𝐵 denotes the batch size and 𝑦𝑖,𝑗 is the
ground truth probability.
3.2. Local image embedding extraction
To explicitly leverage the fine-grained image information, most previous CNN-based works for text-based person search typically employ
hard horizontal slicing (Sun et al., 2018) to extract local visual features.
However, due to the characteristic of self-attention, using a Transformer as visual encoder inevitably integrates information from the
entire image. Thus, directly applying horizontal slicing to the output of
Vision Transformers (ViT) is suboptimal. According to the analysis of
attention distance in Ghiasi et al. (2022), certain attention heads in the
lower layers of ViT exhibit small attention distance, indicating some
degree of local attention. We suggest that in transformer-based networks, more consideration should be given to the impact of attention
mechanism on local feature extraction.
The masked attention was initially proposed by Veličković et al.
(2017) and became widely known when the Mask2Former model
(Cheng et al., 2021) adapted it as a constrained cross-attention module.
Inspired by advances in image segmentation (Jiao et al., 2023; Xu
et al., 2023; Cheng et al., 2021), we propose a novel Attention Biasbased Forward process (ABF) as illustrated in Fig. 3(a). Similar to
Mask2Former, ABF also extracts local features by modulating the attention matrix to limit attention within specific regions. The difference
is that Mask2Former introduces masked attention to alleviate the
slow convergence of query features caused by global context in the
cross-attention layer. It enhances the sensitivity of query features to
foreground information, aiming to extract region proposals of specific
types from an image. Whereas, our proposed ABF leverages global
contextual information to mine additional semantic clues within the
specified regions through unidirectional information transfer of local
patch sets.
Specifically, ABF does not modify the forward process of the first
𝐿 layers in the visual encoder, allowing the [CLS] token to capture

Some recent re-ID works, such as Yang et al. (2023), Zuo et al.
(2023), Jin et al. (2025) and Shu et al. (2021), attempt to put the spotlight on pre-training a model from scratch. These approaches capture
more fine-grained associations by constructing large-scale pedestrian
datasets and employing pre-training tasks related to alignment targets.
Given that pretraining a model from scratch is too expensive, we adopt
the CLIP model to initialize the encoders and fine-tune it entirely on
the TBPS task.
3. Method
In this section, we present our proposed CDFM framework. The
overview of CDFM is illustrated in Fig. 2, and the details are discussed
in the following subsections.
3.1. Revisiting CLIP’s dual-encoder and global alignment
With the advancement of VLP models, recent studies (Yan et al.,
2022; Jiang and Ye, 2023) have attempted to transfer the knowledge
of CLIP to text-based person search. We initialize the CDFM with the
full CLIP image and text encoders to leverage its powerful cross-modal
alignment capability.
Image Encoder. Given an input image 𝐼 ∈ 𝑅𝐻×𝑊 ×3 , we first divide
it into 𝑁 = 𝐻 × 𝑊 ∕𝑃 2 non-overlapping patches, where 𝑃 denotes the
size of each patch. These patches are then flattened and prepended
with a learnable [CLS] token to form an input sequence. We adopt
a 12-layer Vision Encoder to model correlations among the patches.
3

X. Lin, X. Geng, S. Wu et al.

Computer Vision and Image Understanding 267 (2026) 104741

Fig. 2. Overview of the proposed CDFM framework. We embed image and text into the latent space using a dual-stream feature extraction backbone and
optimize them with SDM Loss and ID Loss. The attention Bias is added to the last 12 − 𝐿 layers of the backbone for extracting local image embeddings. A
multimodal decoder is designed to extract fine-grained text embeddings by applying a cross-attention mechanism. 𝐸𝐷𝐿 and 𝐶𝑆𝐿𝑀𝑂𝐷 are proposed to ensure the
robustness of fine-grained embeddings. In the inference phase, the part surrounded by the black dashed line is removed.

passing through the softmax activation function. Fig. 3(b) represents
the visualization of matrix 𝑀𝐵 . The upper 𝐾 rows of the matrix
𝐿 . We set the value corresponding
represent the update process of 𝐹𝑙𝑜𝑐𝑎𝑙
𝐿
to 𝑓𝑐𝑙𝑠 to zero, allowing each group to still receive global information,
making local features more robust. The upper 𝐾 rows can be formulated
as:
{
0, if 𝑚𝑖,𝑗 ∈ 𝑖th 𝑔𝑟𝑜𝑢𝑝
𝑚𝑖,𝑗 =
, 𝑖 ∈ [0, 𝐾 − 1], 𝑗 ∈ [0, 𝑁 + 𝐾],
(2)
−∞, if 𝑚𝑖,𝑗 ∉ 𝑖th 𝑔𝑟𝑜𝑢𝑝
The bottom (𝑁 +1)×(𝑁 +1+𝐾) elements are the numerical setting of
𝐹𝑣𝐿 . It can be seen that with the assistance of matrix, the 𝐹𝑣𝐿 (latter part
∗
of 𝐹𝑣𝐿 ) is updated with the original vision encoder forward process,
ensuring the model’s global context extraction capability is preserved.
Under the setting of matrix, the forward process of the subsequent 12−𝐿
layers is as follows:
( (
)
) (
)𝑇
 𝐹𝑣𝐿∗  𝐹𝑣𝐿∗
( 𝐿∗ )
𝐹𝑣(𝐿+1) = Sof tmax
+
𝑀
+ 𝐹𝑣𝐿∗ ,
(3)
√
𝐵  𝐹𝑣
𝑐

Fig. 3. Overview of the proposed ABF. (a) The proposed Attention Biasbased Forward process (ABF), (b) Visualization of the attention bias matrix
𝑀𝐵 , where white squares indicate a value of 0 and black squares indicate a
large negative value.

rich contextual information during early feature extraction. We denote
𝐿 ,𝐹𝐿
𝐿
the output of 𝐿th layer as 𝐹𝑣𝐿 = {𝑓𝑐𝑙𝑠
} ∈ 𝑅(𝑁+1)×𝐶 , where 𝑓𝑐𝑙𝑠
𝑃 𝑎𝑡𝑐ℎ
represents the embedded visual [CLS] token and 𝐹𝑃𝐿𝑎𝑡𝑐ℎ represents the
patch tokens. In order to mine fine-grained information in transformer
architecture, we divide patch tokens into 𝐾 groups and prepend a
specific local [CLS] token to the beginning of each group. As mentioned
before, the embedded visual [CLS] token should have a comprehensive
understanding of the context within the entire image. We repeat it 𝐾
𝐿
times, denote as 𝐹𝑙𝑜𝑐𝑎𝑙
and concatenate them to the beginning of 𝐹𝑣𝐿 as
local [CLS] tokens. Therefore, the input of subsequent 12-L layers will
∗
𝐿 , 𝐹 𝐿 }.
be 𝐹𝑣𝐿 = {𝐹𝑙𝑜𝑐𝑎𝑙
𝑣
Considering the standard multi-head attention treats each token
equally, we introduce an attention bias matrix 𝑀𝐵 = {𝑚𝑖,𝑗 } ∈
𝑅(𝑁+1+𝐾)×(𝑁+1+𝐾) into the self-attention computation to ensure that
each local [CLS] token only focuses on its respective local group.
Numerically, when an element of the matrix is set to a very large
negative value, the attention weight at that position will be zero after

where (⋅), (⋅) and (⋅) denote the query, key and value transformation respectively.
The output of vision encoder is then projected to the image–text
joint latent embedding space via a learnable image projection, resulting
in the local image embeddings 𝐹𝑣𝑙𝑜𝑐𝑎𝑙 .
3.3. Fine-grained text embedding extraction
3.3.1. Fine-grained embedding learning
Existing methods for parsing input text into fine-grained phrases
typically rely on external tools such as natural language toolkit, which
use lexical properties for text analysis. However, the accuracy of these
methods is highly dependent on the quality of the external tools. To
address this limitation, we propose a Fine-grained Embedding Learning
(FEL) module for extracting local representations based on semantic
similarity, eliminating the need for external tools.
4

X. Lin, X. Geng, S. Wu et al.

Computer Vision and Image Understanding 267 (2026) 104741

fine-grained image embeddings (𝐹𝑣𝑓 𝑖𝑛𝑒 ). This modality-sharing facilitates modal interaction and helps bridge the modality gap between
images and text. As mentioned before, local image embeddings are
obtained through ABF guidance during global feature extraction. Compared to the embeddings acquired by the newly introduced FEL, those
embeddings are more robust and consistent with the global image
semantics. By treating the local image embeddings as supervision
(without gradient backpropagation), we optimize the cosine similarity and Euclidean distance between 𝐹𝑣𝑓 𝑖𝑛𝑒 and 𝐹𝑣𝑙𝑜𝑐𝑎𝑙 , enhancing the
compatibility between pre-training model and FEL module.
For each image–text pair in the mini-batch of size N, the matching
probability between 𝐹𝑣𝑓 𝑖𝑛𝑒 and 𝐹𝑣𝑙𝑜𝑐𝑎𝑙 can be computed using the softmax
function:
exp(𝑠𝑖𝑚(𝑓𝑣𝑓𝑖 𝑖𝑛𝑒 , 𝑓𝑣𝑙𝑜𝑐𝑎𝑙 )∕𝜏)
𝑗
𝑝𝑖2𝑖
=
,
∑𝑁
𝑘𝑖,𝑗
𝑓 𝑖𝑛𝑒
exp(𝑠𝑖𝑚(𝑓
,
𝑓𝑣𝑙𝑜𝑐𝑎𝑙 )∕𝜏)
𝑡
𝑘=1
𝑖

Fig. 4. Details of the Multimodal Decoder. A five-layer decoder extracts
fine-grained embeddings via cross-attention. At every attention layer, the
entire original learnable tokens are re-added to the query tokens.

(4)

𝑘

where sim(𝐮, 𝐯) denotes the cosine similarity between 𝐮 and 𝐯, 𝜏 is the
temperature hyperparameter. The local matching probability 𝑝𝑖2𝑖
can
𝑘𝑖,𝑗
be viewed as the proportion of cosine similarity score between the finegrained image and the local image to the sum of all candidate pairs. We
employ the cross-entropy loss to optimize cosine similarity to associate
the embeddings across different modalities. Then, the Cosine Similarity
Loss (CSL) can be calculated by:

Unlike ALBEF or BLIP, which inject visual information into text
embeddings by inserting a cross-attention layer into traditional transformer block to obtain multimodal features, the FEL module introduces
a set of learnable tokens and a multimodal decoder. It aggregates texts
of variable lengths into semantically similar learnable tokens through
the cross-attention mechanism. Inspired by Kirillov et al. (2023), we design a variant of its mask decoder to serve as our multimodal decoder.
It consists of 𝐷 layers, as shown in Fig. 4. Each decoder layer performs
three steps: (1) cross-attention from learnable tokens (as queries) to
embedded word tokens, (2) self-attention on learnable tokens, (3) a
point-wise MLP updates learnable tokens. Each self/cross-attention and
MLP includes residual connection and layer normalization. The next
decoder layer processes the original embedded word tokens and the
updated learnable tokens from the previous layer. To enhance the
dependency of the decoder output on the original learnable tokens, the
entire original learnable tokens are re-added to the updated ones as
positional embeddings, whenever they participate in an attention layer.
Finally, we utilize a Squeeze-and-Excitation (Hu et al., 2018; Zhou
et al., 2025) layer, which contains two linear layers, a ReLU function
and a sigmoid function, to eliminate the distraction of unrelated features, referring to the output of this SE layer as the fine-grained text
embeddings 𝐹𝑡𝑓 𝑖𝑛𝑒 .

1 ∑ 𝑖2𝑖
𝑃 )),
𝐾 𝑘=1 𝑘
𝐾

𝐿𝐶𝑆𝐿 = −𝐾𝑌 𝑖2𝑖 log(Sof tmax(

(5)

where 𝐾 represents the number of local groups, 𝑃𝑘𝑖2𝑖 denotes the 𝐾th
fine-grained image-to-image similarity, and 𝑌 𝑖2𝑖 represents the ground
truth probability.
The Euclidean Distance Loss (EDL) aims to transfer the Euclidean
knowledge of the CLIP model to the FEL module, reducing the discrepancy between fine-grained text embeddings and the joint latent
embedding space by aligning the fine-grained image embeddings with
local image embeddings. We use mean squared error loss to optimize
the magnitude relationship between them:
2

𝐿𝐸𝐷𝐿 = ∥ 𝑓𝑣𝑙𝑜𝑐𝑎𝑙 − 𝑓𝑣𝑓 𝑖𝑛𝑒 ∥2 ,

(6)

The TES effectively bridges the modal gap between images and
text under the supervision of visual modality. CSL ensures semantic
alignment by optimizing the cosine similarity between fine-grained
image embeddings and local image embeddings. Meanwhile, EDL plays
a role in stabilizing the training process. Since image embeddings serve
as a semantic bridge, if the Euclidean distance is not optimized, the
target text embeddings extracted by the modality-shared FEL will be
distorted magnitude. This will make them incompatible with CLIP’s
joint embedding space, and result in a decline in the overall model
performance.
With the supervision of TES, knowledge from the CLIP space has
been successfully transferred to FEL. To mine fine-grained discrepancies, we introduce a Cross-Modal Alignment (CMA) loss, which employs the cross-entropy loss to optimize the cosine similarity between
fine-grained text embeddings and local image embeddings by:

3.3.2. Text extraction strategy
An intuitive approach is to directly align 𝐹𝑣𝑙𝑜𝑐𝑎𝑙 and 𝐹𝑡𝑓 𝑖𝑛𝑒 . However,
experimental results demonstrate that the performance is suboptimal.
We argue that while the local image embeddings are acquired from
an instance-level pre-trained model guided by ABF, the fine-grained
text embeddings are obtained by a newly introduced fine-grained embedding learning module. A simple learning strategy is insufficient
to enable the randomly initialized multimodal decoder to adapt to
the pre-training model during fine-tuning, thereby hindering its capacity to learn modality-invariant fine-grained representations. Notably, the FEL module is not an independent structure trained from
scratch, but further mines fine-grained feature information based on
the pre-trained CLIP model. The image–text joint latent space of CLIP,
obtained through large-scale image–text pair pre-training, exhibits excellent cross-modal semantic consistency. This property ensures that
if the learnable tokens can efficiently extract fine-grained information
from images, they should be equally capable of extracting corresponding fine-grained features from text. Therefore, we propose a new Text
Extraction Strategy (TES), which uses visual modality as a bridge to
enhance the semantic robustness of fine-grained text embeddings.
Specifically, we set the FEL as modality-shared, leveraging Modalityshared Learnable Tokens (MLT) and multimodal decoder to extract

1 ∑ 𝑡2𝑖
𝑃 )),
𝐾 𝑘=1 𝑘
𝐾

𝐿𝐶𝑀𝐴 = −𝐾𝑌 𝑡2𝑖 log(Sof tmax(

(7)

where 𝑃𝑘𝑡2𝑖 represents the matching probability of fine-grained text–
image pairs computed by Eq. (4), and 𝑌 𝑡2𝑖 represents the ground truth
probability.
3.4. Optimization improvements
To reduce the impact of multimodal decoder on the aligned pretraining image–text joint latent embedding space and enhance training
stability, we draw inspiration from J. Li et al. (2021), Carion et al.
5

X. Lin, X. Geng, S. Wu et al.

Computer Vision and Image Understanding 267 (2026) 104741

(2020) and Cheng et al. (2024) by initializing all MLT to zero and
introducing momentum distillation.
The MLT are designed to extract fine-grained features, but random
initialization may cause them to focus on noise and ignore effective
information. Given the unknown importance of patch/word tokens in
each local group for constructing fine-grained semantic embeddings,
we initialize MLT to zero, so the extracted features can be considered
average pooling feature of patch/word tokens.
Momentum Distillation (MoD), proposed by J. Li et al. (2021),
aims to improve the effectiveness of learning from noisy data. It is
reasonable to treat average pooling features as noisy data and introduce
momentum distillation to enhance the learning of MLT and the decoder.
The momentum MLT are a slow-moving average of the online MLT, capable of learning more stable information. Specifically, for CSL, we first
input momentum MLT and patch tokens into the multimodal decoder
to get momentum fine-grained image embeddings. After calculating
the similarity between momentum fine-grained image embeddings and
local image embeddings and computing the soft pseudo-target (𝑞 𝑖2𝑖 )
according to Eq. (4), the ground truth probability 𝑌 𝑖2𝑖 in Eq. (5) is
replaced with 𝑄𝑖2𝑖 . The momentum CSL (𝐶𝑆𝐿𝑀𝑜𝐷 ) loss is then defined
as:
1 ∑ 𝑖2𝑖
𝑃 )),
𝐾 𝑘=1 𝑘

annotated with 2 text descriptions. The dataset is split into training set,
test set, and validation set, containing 3701, 200, and 200 identities,
respectively.
We adopt the popular Rank-k (k = 1, 5, 10) as the evaluation metric
and report mean Average Precision (mAP) and mean Inverse Negative
Penalty (mINP) for comprehensive evaluation. Higher Rank-k, mAP and
mINP scores indicate better performance of the proposed method.
4.2. Implementation details
All experiments are conducted on a single RTX3090 24 GB GPU.
The visual encoder, i.e., CLIP-ViT-B/16, and text encoder, i.e., CLIP
text Transformer, are initialized using the pre-trained parameters of the
full CLIP model, while the multimodal decoder is initialized randomly.
All input images are resized to 384 × 128, and enhanced with random
horizontal flipping, random cropping with padding, and random erasing. For text input, we employ random masking and set the maximum
sequence length to 77. The dimension of the shared image–text joint
latent embedding space is set to 512. The multimodal decoder consists
of 5 layers, with the hidden size and number of heads set to 512
and 8, for each layer. Following IRRA, our model is trained using the
Adam optimizer for 60 epochs with a batch size of 64. The learning
rate strategy involves linear warmup and cosine learning rate decay.
The initial learning rates of backbone and proposed modules are set
to 1 × 10−5 and 1 × 10−3 , respectively. The temperature 𝜏 is set to 1
for CUHK-PEDES, 0.02 for ICFG-PEDES, and 0.1 for RSTPReid. The
momentum parameter for updating momentum MLT is set as 0.5, and
the momentum 𝛼 is set to 0.4. The layer for applying Attention Bias is
set to 8, and the topK is set to 32 during inference.

𝐾

𝐿𝐶𝑆𝐿𝑀𝑜𝐷 = 𝛼𝐿𝐶𝑆𝐿 − (1 − 𝛼)𝐾𝑄𝑖2𝑖 log(Sof tmax(

(8)

Similarly, the 𝐶𝑀𝐴𝑀𝑜𝐷 loss is obtained by replacing 𝑌 𝑡2𝑖 with 𝑄𝑡2𝑖
in Eq. (7), and is formulated as:
1 ∑ 𝑡2𝑖
𝑃 )),
𝐾 𝑘=1 𝑘
𝐾

𝐿𝐶𝑀𝐴𝐿𝑀𝑜𝐷 = 𝛼𝐿𝐶𝑀𝐴 − (1 − 𝛼)𝐾𝑄𝑡2𝑖 log(Sof tmax(

(9)

Since momentum distillation is solely applied to MLT, it will not
incur substantial additional memory usage.

4.3. Ablation study
To fully demonstrate the effectiveness of proposed modules in our
CDFM, we conduct comprehensive ablation studies on three public
benchmarks. We adopt a dual encoder (i.e., CLIP-ViT-B/16) fine-tuned
with global features alignment mentioned in Section 3.1 as Global
Baseline (No. 0). The experimental results are shown in Table 1.
Introduction of fine-grained embeddings. To quantify the impact
of adding fine-grained modules to the globally aligned model, we introduce a commonly used modality-shared extraction method (No. 1) as
our Baseline. The modality-shared method inputs MLT and embedded
patch/word tokens into the multimodal decoder separately, using the
cross-attention mechanism to get fine-grained embeddings. Compared
to the Global Baseline, the method results in a 1.7%, 1.15%, and
2.65% Rank-1 accuracy drop on the three datasets, respectively. By
introducing ABF into baseline, the method also results in a Rank-1
accuracy drop on the three datasets. We believe the main reason is
that the CLIP model is pre-trained with instance-level alignment on
large-scale image–text pairs, and non-robust local features introduce
irrelevant local noise to the backbone network, impairing the model’s
extraction capability.
Components analysis. To address the issue of feature extraction
capability degradation caused by the arbitrary introduction of finegrained information, we propose Text Extraction Strategy (TES) and
introduce Momentum Distillation (MoD) to enhance the modality robustness of the extracted fine-grained embeddings. When adding the
proposed MoD and ABF into our baseline (No. 3), we observe a performance gain of 1.57%, 0.09%, and 1.55% in terms of Rank-1 accuracy
on the three datasets, respectively. This result supports our statement in Section 3.4 that the fine-grained embeddings obtained by
initializing MLT with zeros, which are equivalent to average pooling
embeddings, are noisy. The introduction of momentum MLT forces the
model to learn the cross-modal similarity relationship from different
perspectives within the same batch via the generated pseudo-targets,
effectively reducing noise interference in the network. By introducing
TES as auxiliary supervision (No. 4) to No. 2, it achieves performance

3.5. Training and inference
The overall optimization objective for training is defined as:
𝐿 = 𝐿𝑖𝑑 + 𝐿𝑠𝑑𝑚 + 𝐿𝐶𝑆𝐿𝑀𝑜𝐷 + 𝐿𝐶𝑀𝐴𝑀𝑜𝐷 + 𝐿𝐸𝐷𝐿 ,

(10)

During inference, for a text query, we first compute the global similarity score 𝑆𝑔𝑙𝑜𝑏𝑎𝑙 between it and all image candidates. After selecting
the top-k candidates, we calculate their fine-grained score 𝑆𝑓 𝑖𝑛𝑒 for
re-ranking. The final similarity for text–image pairs is computed as
𝑆𝑔𝑙𝑜𝑏𝑎𝑙 + 𝑆𝑓 𝑖𝑛𝑒 . Since k can be set to a very small number, our inference
speed is much faster than methods that require dynamically computing
local feature similarity for all image–text pairs.
4. Experiments
4.1. Datasets and metrics
We evaluated our method on three publicly available and challenging TBPS datasets.
CUHK-PEDES (Li et al., 2017) is the first dataset collected for
text-based person search. It contains 40,206 images and 80,412 text
descriptions of 13,003 pedestrians. Following the official protocol, the
dataset is split into training, validation and test sets. The training set
includes 34,054 images and 68,108 text descriptions for 11,003 identities. The validation set includes 3078 images and 6158 text descriptions
for 1000 pedestrians. The test set includes 3074 images and 6156 text
descriptions for 1000 identities.
ICFG-PEDES (Ding et al., 2021) includes 54,522 images of 4102
identities collected from MSMT17. It is split into training and test
set, containing 34,674 image–text pairs for 3102 identities and 19,848
image–text pairs for 1000 identities, respectively. The text description
of this dataset is more fine-grained.
RSTPReid (Zhu et al., 2021) is constructed based on MSMT17. It
includes 20,505 images of 4101 pedestrians from 15 cameras, each
6


web search: 
web search: FedUCC FedUReID heterogeneous client federated learning person re-identification knowledge distillation
web search: 
web search: "CFPER" "Coarse-to-Fine Part-Experts Retrieval"
web search: 
web search: "Coarse-to-Fine Part-Experts Retrieval" "person re-identification"
web search: 
web search: "CDFM" "Text Extraction Strategy" "text-based person search"
codex
你列的是 6 篇，不是 5 篇。我按 6 篇都拆了。

**CCFL**
1. 触发观察：作者抓住的不是“联邦 ReID 精度不够”，而是 ReID 的联邦场景天然比普通联邦学习更异质。不同机构的数据来自不同地点、时间、设备和规模，数据非独立同分布很重，而且客户端算力也不同。统一模型、统一训练策略在这种场景下反而不合理。

2. 重定义动作：他们把问题从“怎么在隐私约束下聚合多个 ReID 数据集”改写成“不同客户端应该被允许使用不同训练策略和不同模型结构”。关键词是 customized client、heterogeneous model、interaction model、non-IID、limited server。这个重定义很重要，因为它把 FedAvg 失效从技术限制变成了问题本身。

3. 机制怎么长出来：既然客户端可定制，就不能直接平均本地模型参数，所以自然需要一个结构统一的交互模型作为知识交换媒介。大数据客户端用 DBSCAN 和较大模型，小数据客户端用层次聚类和较小模型，本地模型把知识蒸馏给交互模型，交互模型在服务器聚合后再反向指导本地模型。这个机制和问题定义绑定比较紧，交互模型是异构客户端设定下的自然产物。

4. 证据闭环：关键不是只报 mAP，而是比较统一策略、统一模型、多策略、多模型的差别。它用 8 个 ReID 数据集模拟真实客户端异质性，并和 FedUReID、FedUCC 等联邦 ReID 方法比较。真正支撑重定义的是“多策略、多模型版本优于简化联邦版本”，说明定制客户端确实不是装饰。

5. 审稿人为什么买账：这篇卖的主要是视角，其次才是蒸馏机制。它的真实新意来自“联邦 ReID 不应该默认一个全局同构模型”这个部署假设，而不是某个 ReID 特征提取模块。对我们有用的点是，先把现实约束讲成旧范式不成立，再让机制变成补救这个范式缺口的必然选择。

**CCUP**
1. 触发观察：作者注意到换衣 ReID 的核心瓶颈不是模型不够复杂，而是真实数据太少，尤其是同一个身份在大量服装变化下的标注样本太少。PRCC、LTCC 等数据集规模和每人服装数都有限，模型很容易过拟合衣服。

2. 重定义动作：他们把“如何设计更好的换衣 ReID 模型”改写成“如何构造足够覆盖身份、服装、相机变化的预训练分布”。关键词是 controllable synthetic data、low-cost、self-annotated、outfits per identity、pretrain-finetune、cloth-irrelevant features。

3. 机制怎么长出来：如果缺的是同一身份多服装数据，那机制自然就是用 3D 人体、服装资产、纹理替换和虚拟相机生成可控数据。裸人体网格定义身份，服装和纹理定义换衣变化，Unreal Engine 模拟监控场景，检测器自动裁框和生成标签。后面的预训练加微调只是让这个大规模合成分布进入普通模型。

4. 证据闭环：最关键的证据是同一模型在 CCUP、UnrealPerson、PersonX、ClonedPerson 等不同合成数据上预训练后的对比。如果只是“多一点数据有用”，通用合成数据也应该同样有效；如果 CCUP 更好，才说明“换衣预训练分布”这个重定义成立。Grad-CAM 可视化进一步说明预训练后模型更关注脸、脖子、肩、手腕、鞋等相对少变的区域，而不是背景或衣服。

5. 审稿人为什么买账：这篇卖的是数据和任务分布视角，不是模型机制。新意来自“换衣 ReID 缺的不是又一个去衣服损失，而是可控的大规模换衣监督”。对我们有用的套路是，发现任务缺少某个关键变化轴，然后自己构造这个轴，并用跨数据集预训练收益证明它不是普通增广。

**CFPER**
1. 触发观察：作者先做了一个很实用的观察，global feature 加 part feature 在 Market、Duke 这种整体行人数据上只带来很小收益，却增加计算量；但在 Occluded-Duke 上收益明显更大。也就是说，查询样本难度不同，统一走细粒度匹配会浪费简单样本，也会让困难样本得不到足够处理。

2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。

3. 机制怎么长出来：先用 ViT 得到全局特征和 patch 特征，用全局特征与 patch 特征的相似度排序，再用一阶差分估计可见人体区域比例。可见人体足够多就判为简单样本，只用全局特征提前退出；否则进入细阶段，用拓扑监督的 patch-to-part router 和 part experts 提取细粒度部件特征。这个机制和重定义绑定很紧，难度分流直接决定是否启用细粒度计算。

4. 证据闭环：最关键的是三类证据连起来了。第一，global 和 global+part 在不同难度数据集上的收益差异证明观察成立。第二，early-exit 阈值的表格展示了 mAP、rank1 和 FLOPs 的权衡，证明它真在做资源分配。第三，easy/hard 查询可视化和 Top-10 检索结果说明简单样本用全局足够，困难样本确实从部件阶段受益。

5. 审稿人为什么买账：这篇卖的是视角，部件专家只是服务于视角。它把效率从附属指标变成方法的核心问题，这比单纯说“我又做了一个 part module”更容易成立。对我们有用的是，找一个主流评价以外但真实存在的轴，比如计算、更新、标注、部署，再用一个很小的观察表把问题立住。

**Channel-aware feature mining network**
1. 触发观察：作者的观察是 VI-ReID 中 RGB 和红外不仅有整体模态差异，还有通道层面的不平衡。有些通道包含衣服纹理、轮廓、热分布等身份线索，有些通道贡献很弱甚至引入噪声。现有方法多把通道操作当预处理或数据增强，没有显式挖掘通道关系。

2. 重定义动作：它把 VI-ReID 的模态差距问题改写成“通道级身份信息没有被充分建模”。关键词是 channel-aware、channel-level feature optimization、channel-level feature refinement、channel imbalance、identity-relevant channels。

3. 机制怎么长出来：机制是三段式。CLFO 在早期用深度可分离卷积、归一化、SE 和可学习残差做通道筛选；CLFR 再用通道注意力、非局部融合、空间注意力抑制噪声；MDFO 在高低层特征之间做通道和空间维度的联合优化。这个机制和重定义的绑定中等偏弱，因为一旦说“通道重要”，很多通用注意力堆叠都可以被解释成通道挖掘。

4. 证据闭环：它的闭环主要靠通道激活可视化和模块消融。可视化说明不同通道确实响应不同身份区域；CLFO、CLFR、MDFO 逐步加入带来提升；子模块消融进一步说明各部件有增益。不过它更像“模块有效性证明”，对“通道重定义一定正确”的因果证明不够强，例如缺少更硬的参数匹配注意力对照或随机通道扰动对照。

5. 审稿人为什么买账：这篇主要卖机制和结果，视角相对普通。它的真实新意是把 VI-ReID 里已有的通道增强思路推进到特征学习内部，并用较完整实验支撑 SOTA。对我们来说，这是一个警示样本：如果观察不够尖，机制就容易变成通用模块堆叠，投稿时会更依赖结果强度。

**CLIP-Based Camera-Agnostic Feature Learning**
1. 触发观察：ICS ReID 只在每个相机内部标身份，不给跨相机身份对应。已有方法先做相机内学习，再用相似度或聚类做跨相机关联，但伪标签容易被视角、背景、光照和相机风格污染。CLIP 可以给语义监督，但直接用于跨相机阶段也会受噪声影响。

2. 重定义动作：它把 ICS ReID 从“低标注 ReID 的伪标签问题”改写成“如何利用相机内标签学习 camera-agnostic 特征”。关键词是 camera-agnostic feature learning、intra-camera discriminative learning、inter-camera adversarial learning、prompt learning、semantic supervision。

3. 机制怎么长出来：第一阶段用 CLIP 和可学习 prompt 给每个相机内身份生成隐式文本描述。第二阶段用相机内混合记忆库存身份中心特征和实例特征，同时拉近同身份中心和困难正样本，推开困难负样本。第三阶段做跨相机关联和原型对比，再用 ICAL 惩罚模型区分同一伪身份在不同相机下的全局 ID。机制和重定义绑定很紧，因为每一步都在回答“相机内标签能提供什么，跨相机噪声怎么压”。

4. 证据闭环：最能支撑重定义的不是最终 mAP，而是 ICDL、ICAL、CLIP prompt 的消融，以及全局分类器概率分布可视化。图里从单峰变多峰的现象说明，同一伪身份在不同相机下被拉到一起，模型的相机可分性被削弱。若伪标签质量或跨相机关联准确率也提升，闭环会更完整。

5. 审稿人为什么买账：这篇卖的是视角加协议适配。它不是简单说“CLIP 用在 ReID”，而是把 CLIP 放进 ICS 的特殊标签结构里，用剩余标签解决缺失标签带来的跨相机偏差。对我们有用的套路是，先精确定义“哪些监督缺失，哪些监督还在”，然后让机制围绕剩余监督自然展开。

**CLIP-driven fine-grained mining**
1. 触发观察：作者注意到 CLIP 在文本行人检索里全局对齐很强，但难以区分细粒度人物差异。更关键的是，简单加局部对齐并不一定有效，甚至会伤害 CLIP，因为 CLIP 预训练是实例级图文对齐，随机初始化的局部解码器和粗暴切块会破坏预训练空间。

2. 重定义动作：它把问题从“给 CLIP 加 fine-grained alignment”改写成“在不引入外部工具、不破坏 CLIP 预训练空间的前提下挖细粒度关系”。关键词是 pre-training space compatibility、explicit fine-grained mining、attention bias forward、text extraction strategy、visual bridge、momentum distillation。

3. 机制怎么长出来：ABF 不直接切最后层特征，而是在 ViT 中后层复制 CLS 作为局部 token，并用 attention bias 限制每个局部 token 只关注对应 patch 组，同时保留全局路径。FEL 用共享可学习 token 和多模态解码器从文本中聚合细粒度语义。TES 用局部视觉表示监督 FEL，把 CLIP 空间里的余弦关系和欧氏距离关系转移给细粒度文本表示。MoD 和零初始化用于避免随机局部 token 一开始学噪声。这个机制和重定义绑定非常紧。

4. 证据闭环：这篇的消融很有说服力。普通细粒度 baseline 比 global baseline 还差，证明“直接加局部会伤”这个观察成立；加入 ABF 仍不够，加入 TES 和 MoD 后才稳定提升，说明关键不是局部本身，而是和 CLIP 预训练空间兼容。ABF 对 PCB 的对比也证明 ViT 不能照搬 CNN 的水平切块。

5. 审稿人为什么买账：这篇卖的是诊断视角。它不是声称“我有一个更强 local module”，而是先证明常识动作会失败，再提出兼容预训练空间的约束和机制。对我们最有启发的是，好的创新可以来自“旧办法为什么在新基础模型上不再成立”的反直觉证据。

**共同套路**
这批论文的共同写法是，先找一个具体、可测的失败现象，再把它上升成旧范式的盲点：统一客户端、有限换衣数据、统一查询预算、忽略通道、相机内标签孤岛、粗暴局部 CLIP。真正强的几篇不是模块复杂，而是机制几乎从重定义里自然长出来，并且用消融证明“这个新问题定义是对的”，而不是只证明“加了模块涨点”。

补查来源主要用于核对先例背景：FedUCC/FedReID 相关联邦 ReID 先例见 [Robust Knowledge Adaptation for Federated Unsupervised Person ReID](https://arxiv.org/abs/2301.07320) 和 [FedReID benchmark](https://arxiv.org/abs/2008.11560)；CCUP 与 CCAFL 条目见 [CCUP arXiv](https://arxiv.org/abs/2410.13567) 和 [CCAFL arXiv](https://arxiv.org/abs/2409.19563)。
tokens used
145,807
你列的是 6 篇，不是 5 篇。我按 6 篇都拆了。

**CCFL**
1. 触发观察：作者抓住的不是“联邦 ReID 精度不够”，而是 ReID 的联邦场景天然比普通联邦学习更异质。不同机构的数据来自不同地点、时间、设备和规模，数据非独立同分布很重，而且客户端算力也不同。统一模型、统一训练策略在这种场景下反而不合理。

2. 重定义动作：他们把问题从“怎么在隐私约束下聚合多个 ReID 数据集”改写成“不同客户端应该被允许使用不同训练策略和不同模型结构”。关键词是 customized client、heterogeneous model、interaction model、non-IID、limited server。这个重定义很重要，因为它把 FedAvg 失效从技术限制变成了问题本身。

3. 机制怎么长出来：既然客户端可定制，就不能直接平均本地模型参数，所以自然需要一个结构统一的交互模型作为知识交换媒介。大数据客户端用 DBSCAN 和较大模型，小数据客户端用层次聚类和较小模型，本地模型把知识蒸馏给交互模型，交互模型在服务器聚合后再反向指导本地模型。这个机制和问题定义绑定比较紧，交互模型是异构客户端设定下的自然产物。

4. 证据闭环：关键不是只报 mAP，而是比较统一策略、统一模型、多策略、多模型的差别。它用 8 个 ReID 数据集模拟真实客户端异质性，并和 FedUReID、FedUCC 等联邦 ReID 方法比较。真正支撑重定义的是“多策略、多模型版本优于简化联邦版本”，说明定制客户端确实不是装饰。

5. 审稿人为什么买账：这篇卖的主要是视角，其次才是蒸馏机制。它的真实新意来自“联邦 ReID 不应该默认一个全局同构模型”这个部署假设，而不是某个 ReID 特征提取模块。对我们有用的点是，先把现实约束讲成旧范式不成立，再让机制变成补救这个范式缺口的必然选择。

**CCUP**
1. 触发观察：作者注意到换衣 ReID 的核心瓶颈不是模型不够复杂，而是真实数据太少，尤其是同一个身份在大量服装变化下的标注样本太少。PRCC、LTCC 等数据集规模和每人服装数都有限，模型很容易过拟合衣服。

2. 重定义动作：他们把“如何设计更好的换衣 ReID 模型”改写成“如何构造足够覆盖身份、服装、相机变化的预训练分布”。关键词是 controllable synthetic data、low-cost、self-annotated、outfits per identity、pretrain-finetune、cloth-irrelevant features。

3. 机制怎么长出来：如果缺的是同一身份多服装数据，那机制自然就是用 3D 人体、服装资产、纹理替换和虚拟相机生成可控数据。裸人体网格定义身份，服装和纹理定义换衣变化，Unreal Engine 模拟监控场景，检测器自动裁框和生成标签。后面的预训练加微调只是让这个大规模合成分布进入普通模型。

4. 证据闭环：最关键的证据是同一模型在 CCUP、UnrealPerson、PersonX、ClonedPerson 等不同合成数据上预训练后的对比。如果只是“多一点数据有用”，通用合成数据也应该同样有效；如果 CCUP 更好，才说明“换衣预训练分布”这个重定义成立。Grad-CAM 可视化进一步说明预训练后模型更关注脸、脖子、肩、手腕、鞋等相对少变的区域，而不是背景或衣服。

5. 审稿人为什么买账：这篇卖的是数据和任务分布视角，不是模型机制。新意来自“换衣 ReID 缺的不是又一个去衣服损失，而是可控的大规模换衣监督”。对我们有用的套路是，发现任务缺少某个关键变化轴，然后自己构造这个轴，并用跨数据集预训练收益证明它不是普通增广。

**CFPER**
1. 触发观察：作者先做了一个很实用的观察，global feature 加 part feature 在 Market、Duke 这种整体行人数据上只带来很小收益，却增加计算量；但在 Occluded-Duke 上收益明显更大。也就是说，查询样本难度不同，统一走细粒度匹配会浪费简单样本，也会让困难样本得不到足够处理。

2. 重定义动作：它把 ReID 从“所有查询都用同一个特征管线”改写成“检索应该根据查询难度动态分配计算资源”。关键词是 coarse-to-fine retrieval、query difficulty、early exit、easy query、hard query、adaptive resource allocation。

3. 机制怎么长出来：先用 ViT 得到全局特征和 patch 特征，用全局特征与 patch 特征的相似度排序，再用一阶差分估计可见人体区域比例。可见人体足够多就判为简单样本，只用全局特征提前退出；否则进入细阶段，用拓扑监督的 patch-to-part router 和 part experts 提取细粒度部件特征。这个机制和重定义绑定很紧，难度分流直接决定是否启用细粒度计算。

4. 证据闭环：最关键的是三类证据连起来了。第一，global 和 global+part 在不同难度数据集上的收益差异证明观察成立。第二，early-exit 阈值的表格展示了 mAP、rank1 和 FLOPs 的权衡，证明它真在做资源分配。第三，easy/hard 查询可视化和 Top-10 检索结果说明简单样本用全局足够，困难样本确实从部件阶段受益。

5. 审稿人为什么买账：这篇卖的是视角，部件专家只是服务于视角。它把效率从附属指标变成方法的核心问题，这比单纯说“我又做了一个 part module”更容易成立。对我们有用的是，找一个主流评价以外但真实存在的轴，比如计算、更新、标注、部署，再用一个很小的观察表把问题立住。

**Channel-aware feature mining network**
1. 触发观察：作者的观察是 VI-ReID 中 RGB 和红外不仅有整体模态差异，还有通道层面的不平衡。有些通道包含衣服纹理、轮廓、热分布等身份线索，有些通道贡献很弱甚至引入噪声。现有方法多把通道操作当预处理或数据增强，没有显式挖掘通道关系。

2. 重定义动作：它把 VI-ReID 的模态差距问题改写成“通道级身份信息没有被充分建模”。关键词是 channel-aware、channel-level feature optimization、channel-level feature refinement、channel imbalance、identity-relevant channels。

3. 机制怎么长出来：机制是三段式。CLFO 在早期用深度可分离卷积、归一化、SE 和可学习残差做通道筛选；CLFR 再用通道注意力、非局部融合、空间注意力抑制噪声；MDFO 在高低层特征之间做通道和空间维度的联合优化。这个机制和重定义的绑定中等偏弱，因为一旦说“通道重要”，很多通用注意力堆叠都可以被解释成通道挖掘。

4. 证据闭环：它的闭环主要靠通道激活可视化和模块消融。可视化说明不同通道确实响应不同身份区域；CLFO、CLFR、MDFO 逐步加入带来提升；子模块消融进一步说明各部件有增益。不过它更像“模块有效性证明”，对“通道重定义一定正确”的因果证明不够强，例如缺少更硬的参数匹配注意力对照或随机通道扰动对照。

5. 审稿人为什么买账：这篇主要卖机制和结果，视角相对普通。它的真实新意是把 VI-ReID 里已有的通道增强思路推进到特征学习内部，并用较完整实验支撑 SOTA。对我们来说，这是一个警示样本：如果观察不够尖，机制就容易变成通用模块堆叠，投稿时会更依赖结果强度。

**CLIP-Based Camera-Agnostic Feature Learning**
1. 触发观察：ICS ReID 只在每个相机内部标身份，不给跨相机身份对应。已有方法先做相机内学习，再用相似度或聚类做跨相机关联，但伪标签容易被视角、背景、光照和相机风格污染。CLIP 可以给语义监督，但直接用于跨相机阶段也会受噪声影响。

2. 重定义动作：它把 ICS ReID 从“低标注 ReID 的伪标签问题”改写成“如何利用相机内标签学习 camera-agnostic 特征”。关键词是 camera-agnostic feature learning、intra-camera discriminative learning、inter-camera adversarial learning、prompt learning、semantic supervision。

3. 机制怎么长出来：第一阶段用 CLIP 和可学习 prompt 给每个相机内身份生成隐式文本描述。第二阶段用相机内混合记忆库存身份中心特征和实例特征，同时拉近同身份中心和困难正样本，推开困难负样本。第三阶段做跨相机关联和原型对比，再用 ICAL 惩罚模型区分同一伪身份在不同相机下的全局 ID。机制和重定义绑定很紧，因为每一步都在回答“相机内标签能提供什么，跨相机噪声怎么压”。

4. 证据闭环：最能支撑重定义的不是最终 mAP，而是 ICDL、ICAL、CLIP prompt 的消融，以及全局分类器概率分布可视化。图里从单峰变多峰的现象说明，同一伪身份在不同相机下被拉到一起，模型的相机可分性被削弱。若伪标签质量或跨相机关联准确率也提升，闭环会更完整。

5. 审稿人为什么买账：这篇卖的是视角加协议适配。它不是简单说“CLIP 用在 ReID”，而是把 CLIP 放进 ICS 的特殊标签结构里，用剩余标签解决缺失标签带来的跨相机偏差。对我们有用的套路是，先精确定义“哪些监督缺失，哪些监督还在”，然后让机制围绕剩余监督自然展开。

**CLIP-driven fine-grained mining**
1. 触发观察：作者注意到 CLIP 在文本行人检索里全局对齐很强，但难以区分细粒度人物差异。更关键的是，简单加局部对齐并不一定有效，甚至会伤害 CLIP，因为 CLIP 预训练是实例级图文对齐，随机初始化的局部解码器和粗暴切块会破坏预训练空间。

2. 重定义动作：它把问题从“给 CLIP 加 fine-grained alignment”改写成“在不引入外部工具、不破坏 CLIP 预训练空间的前提下挖细粒度关系”。关键词是 pre-training space compatibility、explicit fine-grained mining、attention bias forward、text extraction strategy、visual bridge、momentum distillation。

3. 机制怎么长出来：ABF 不直接切最后层特征，而是在 ViT 中后层复制 CLS 作为局部 token，并用 attention bias 限制每个局部 token 只关注对应 patch 组，同时保留全局路径。FEL 用共享可学习 token 和多模态解码器从文本中聚合细粒度语义。TES 用局部视觉表示监督 FEL，把 CLIP 空间里的余弦关系和欧氏距离关系转移给细粒度文本表示。MoD 和零初始化用于避免随机局部 token 一开始学噪声。这个机制和重定义绑定非常紧。

4. 证据闭环：这篇的消融很有说服力。普通细粒度 baseline 比 global baseline 还差，证明“直接加局部会伤”这个观察成立；加入 ABF 仍不够，加入 TES 和 MoD 后才稳定提升，说明关键不是局部本身，而是和 CLIP 预训练空间兼容。ABF 对 PCB 的对比也证明 ViT 不能照搬 CNN 的水平切块。

5. 审稿人为什么买账：这篇卖的是诊断视角。它不是声称“我有一个更强 local module”，而是先证明常识动作会失败，再提出兼容预训练空间的约束和机制。对我们最有启发的是，好的创新可以来自“旧办法为什么在新基础模型上不再成立”的反直觉证据。

**共同套路**
这批论文的共同写法是，先找一个具体、可测的失败现象，再把它上升成旧范式的盲点：统一客户端、有限换衣数据、统一查询预算、忽略通道、相机内标签孤岛、粗暴局部 CLIP。真正强的几篇不是模块复杂，而是机制几乎从重定义里自然长出来，并且用消融证明“这个新问题定义是对的”，而不是只证明“加了模块涨点”。

补查来源主要用于核对先例背景：FedUCC/FedReID 相关联邦 ReID 先例见 [Robust Knowledge Adaptation for Federated Unsupervised Person ReID](https://arxiv.org/abs/2301.07320) 和 [FedReID benchmark](https://arxiv.org/abs/2008.11560)；CCUP 与 CCAFL 条目见 [CCUP arXiv](https://arxiv.org/abs/2410.13567) 和 [CCAFL arXiv](https://arxiv.org/abs/2409.19563)。
