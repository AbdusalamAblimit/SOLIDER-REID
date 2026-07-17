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
session id: 019ef835-7164-7d60-8c1b-d9a98aa6d2ce
--------
user
你是 ReID 创新方法论拆解员 6。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(6)==
CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf
CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf
Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf
ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会按你的要求先从本地 PDF 抽方法部分，不只看摘要；需要判断先例或新意边界时再查网页。因为当前目录就是论文库，我先确认文件和可用工具，再逐篇读方法。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pwd && rg --files -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
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
/bin/zsh -lc 'for f in GOAL.md CLAUDE.md experiments/STATUS.md; do [ -r "../../../$f" ] && printf '"'%s\\n' \"../../../"'$f"; [ -r "../../../../$f" ] && printf '"'%s\\n' \"../../../../"'$f"; done' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 exited 1 in 0ms:
../../../GOAL.md
../../../CLAUDE.md
../../../experiments/STATUS.md

codex
我已经确认会使用 PDF 技能的本地抽取流程，也确认这 6 篇都在当前目录。项目规则要求接手先看目标和状态文件；我先快速读这些规则文件，然后开始抽每篇方法段。
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
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "sed -n '1,240p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "sed -n '1,240p' ../../../experiments/STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "[ -r ../../../AGENTS.md ] && sed -n '1,220p' ../../../AGENTS.md || true" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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


任务是方法论拆解，不改文件。先按用户指定的 pdftotext 口径读方法，再写中文分析。
  → 抽取 6 篇 PDF 的前 8 页文本并定位方法段。
  • 逐篇归纳触发观察、重定义、机制、证据闭环和 novelty 来源。
  • 必要时检索先例，补充共同创新套路总结。
codex
我现在会先用 `pdftotext -f 1 -l 8` 找每篇的章节边界。这个步骤只定位方法段，避免把引言里的卖点当成方法本身。
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
/bin/zsh -lc "pdfinfo 'Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf' | sed -n '1,80p' && pdftotext -f 1 -l 8 'Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf' - | nl -ba | rg -i \"method|approach|framework|overview|proposed|stable|diffusion|cloth|debias|experiment|ablation|analysis|3\\.|iii|iv|architecture\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification
Subject:         ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP);2025; ; ;10.1109/ICASSP49660.2025.10890718
Creator:         Certified by IEEE PDFExpress at January 15, 2025 11:05:17
Producer:        pdfTeX-1.40.26; modified using iText® Core 7.2.4 (AGPL version) ©2000-2022 iText Group NV
CreationDate:    Wed Jan 15 18:56:25 2025 CST
ModDate:         Sun Feb 23 01:23:40 2025 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           5
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       962278 bytes
Optimized:       no
PDF version:     1.5
     3	Cloth-debiasing with Stable Diffusion in
     4	Cloth-changing Person Re-identification
    10	Beijing University Of Posts and Telecommunications
    13	Beijing University Of Posts and Telecommunications
    15	Abstract—In the current study of cloth-changing person reidentification (CC-ReID), the misidentification rate is significantly
    16	high for different individuals wearing similar attire, due to biases
    17	in clothing features. The generation model can unify the clothing
    18	feature space, minimizing interference from clothing color and
    19	type, and enabling model to concentrate on extracting clothingirrelevant features. However, the current use of Generative
    20	Adversarial Networks (GANs) for changing clothes in CC-ReID
    22	generated images and the original images, resulting in unstable
    23	outcomes when changing the same clothing for pedestrians with
    24	varying postures and clothing types. Consequently, we generate
    25	cloth-changing pedestrian images with consistent clothing based
    26	on a stable diffusion model controlled by body keypoint information, ensuring the images conform to the geometric structure
    29	dataset with consistent clothing styles. Meanwhile, we improve
    32	to distinguish between pedestrians wearing similar clothing. Extensive experiments demonstrate that our approach outperforms
    33	previous methods, achieving a 6% increase in Rank-1 and a 4.4%
    35	Index Terms—cloth-changing person re-identification, stable
    36	diffusion
    39	Person re-identification (ReID) aims to identify individuals in surveillance videos across various locations and time
    41	clothing remains unchanged over short duration. However, the
    42	challenge of changes in clothing not only exists in identifying a
    45	their clothes to avoid being identified and tracked. Due to its
    46	crucial role in intelligent monitoring systems, cloth-changing
    48	Humans can recognize acquaintances, even if those acquaintances are wearing clothes that they have never seen before,
    49	because human brains can decouple and make use of clothirrelevant features, such as body shape and gaits. Similarly in
    50	CC-ReID, if the data is sufficiently complex, the data-driven
    52	cloth-irrelevant and discriminative features. However, humans
    53	tend to identify strangers based on their different clothes. It
    57	diversity leads to classification bias during training. The model
    60	high scores to images of people with similar clothing.
    61	To mitigate this bias, the generative model can be used
    62	to unify the clothing feature space, and change the clothing
    64	similar clothing data of pedestrians for training can reduce
    65	the interference of clothing-related features and enhance the
    67	clothing. The most current cloth-changing method in CCReID involves GAN models [7]–[9]. In GANs, the inadequate
    70	in high-dimensional clothing features. If there is a complex
    72	changing a pedestrian’s clothing type or altering the attire
    74	clothing data, GANs tend to retain most of the original features
    75	of pedestrian clothing and may generate low-quality images
    78	To address the limitations of using GANs for clothingchange in CC-ReID, referring to the application of diffusion
    79	model in virtual try-on field [10], we propose adopting a generation scheme based on the stable diffusion model to ensure
    80	the quality of the generated clothing-change data. By gradually
    81	varying during the generation process, the diffusion model [11]
    84	masks of pedestrians as masks for the inpainting method in
    85	stable diffusion, ensuring the non-clothing parts of the human
    87	pedestrians are used to control the generated clothing-change
    90	clothing. To enhance the model’s ability to distinguish pedestrians wearing similar clothing, we introduce centroid loss
    96	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:59 UTC from IEEE Xplore. Restrictions apply.
    98	Fig. 1. Overall architecture of the proposed method. On the left, auxiliary information is generated to assist with the cloth-changing process. In the center,
    99	the generate-filter clothing-change module(GFCC) produces pedestrian images with consistent clothing. On the right is the structure of our CCAL model that
   105	cloth-changing scheme based on the stable diffusion model is
   106	employed to generate consistent clothing for pedestrians. The
   107	masks for the stable diffusion inpainting method are derived
   109	of human features, excluding clothing. To ensure that the
   114	construct high-quality consistent clothing CC-ReID datasets.
   115	(3) The consistent clothing CC-ReID datasets are used to train
   117	loss. Experimental results demonstrate that our approach outperforms previous methods by a significant margin.
   123	segmentation masks that represent body parts and clothing
   127	masks, such as ’Upper-clothes’, ’Left-leg’, ’Right-leg’, ’Leftarm’, ’Right-arm’, ’Bag’ and so on. Additionally, we utilize
   133	The generating phase of the generate-filter clothing-change
   134	module(GFCC) adopts LaDI-VTON method [10], as illustrated in Fig.1. The expanded Stable Diffusion inpainting
   135	pipeline of LaDI-VTON is used to change clothing for pedestrian images in CC-ReID.
   144	map p ∈ R18×h×w and the encoded clothing E(Ĉ) ∈ R4×h×w .
   155	original inpainting mask M is derived from the segmentation
   157	M . To ensure sufficient coverage of the intended clothing
   158	regions, the method proposed in prior work [16] is applied.
   160	pipeline, including the textual prompt describing the clothing
   161	[10], the pose map P , and the clothing Ĉ which is warped
   162	according to the pedestrian body shape. The warped clothing
   165	module determines the correlation between the clothing C and
   166	a clothing-irrelevant pedestrian representation, which includes
   167	Iˆ and P , generating parameters θ. A thin-plate spline transformation [19] generates the coarse warped clothing C ′ from the
   168	clothing C by C ′ = T P Sθ (C). The U-Net refines the warped
   170	clothing by Ĉ = U net(C ′ , P, I).
   172	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:59 UTC from IEEE Xplore. Restrictions apply.
   175	To address the issue of low-quality generated cloth-changing
   183	2) FID Threshold Filter: We divide the generated images
   195	samples are divided into two groups: PA , which includes
   201	each ID i has N samples. f (n) and Xn respectively represent
   204	The centroids for both clusters of ID i are derived as follows:
   239	Fig. 2. The examples show the effects of changing the clothing for pedestrians
   240	with different postures and different original clothing types.
   243	same clothes setting and clothing-change setting. For LTCC,
   247	of each pedestrian and change their clothing to a set of long
   250	that are typically visible despite clothing, such as the head and
   251	neck, while retaining human contours. We use stable diffusion
   254	better detail reconstruction in inpainting task. Example clothchanging images are shown in Fig. 2. After filtering, we add
   257	We train our model on our consistent clothing datasets. Following the CAL [22], ResNet50 [26] pre-trained on ImageNet
   263	and 0.1, respectively. We add Side Information Embeddings
   271	and divided by 10 after every 20 epochs.
   272	C. Comparison with state-of-the-art methods
   273	We compare our method with several existing ReID methods on LTCC and PRCC datasets. For the PRCC dataset
   279	III. E XPERIMENT
   281	We evaluate our approach on two widely used datasets
   283	33698 images of 221 individuals captured by 3 cameras.
   284	LTCC contains 17,138 images of 152 individuals taken by 12
   287	C OMPARISON WITH SOTA METHODS ON PRCC DATASET (%).
   288	Method
   311	Cloth-Changing
   328	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:59 UTC from IEEE Xplore. Restrictions apply.
   331	C OMPARISON WITH SOTA METHODS ON LTCC DATASET (%).
   332	Method
   343	63.7
   345	73.2
   357	Cloth-Changing
   373	as shown in TABLE I, in the standard setting, our method
   374	achieves the same metrics as the baseline method and maintains a high level of performance. In the cloth-changing scenario, our method outperforms the baseline by 6% and 4.4%
   375	in Rank-1 and mAP, respectively. And it also outperforms the
   376	previous state-of-the-art method by 3.3% and 1.9% on Rank-1
   378	method improves the Rank-1 and mAP by 1.5% and 0.2%,
   379	respectively, compared to the baseline method in the clothchanging setting. In the standard setting, Rank-1 and mAP
   380	improve by 0.6% and 0.3%. This demonstrates that our method
   381	for enhancing cloth-changing data has effectively mitigated
   382	biases related to clothing features, resulting in a noticeable
   384	D. Ablation Study
   385	We conduct comprehensive experiments on the PRCC
   386	dataset to validate the effectiveness of our GFCC module and
   387	CCAL module. As shown in TABLE III, the baseline CAL
   390	rank-1 and mAP increasing by 0.6% and 0.7%, respectively,
   391	compared to the baseline in the cloth-changing setting. Upon
   394	Fig. 4. Examples of improved misidentification situations of Fig. 3.
   398	maps of the baseline method and our method, respectively.
   400	method (GFCC+CCAL) is elevated to 61.2% and 60.2% on
   402	demonstrating that our scheme effectively mitigates biases of
   403	clothing features.
   404	E. Qualitative Results
   406	method, we randomly select images of pedestrians to display
   409	of pedestrians with similar clothing. After implementing our
   410	method, as depicted in Fig. 4, the misidentification of these
   411	pedestrians with similar clothing is solved to a large extent.
   413	baseline method and our method. The ID features of the
   414	baseline method are misled by clothing bias. The clothing
   416	easily recognizable cloth-relevant features. While our method
   417	effectively mitigates the impact of clothing bias by emphasizing more clothing-irrelevant information and redirecting
   418	the model’s attention towards the non-clothing areas such
   419	as the face and neck. Through introducing similar clothing
   421	method become more concentrated on the areas that influence
   423	IV. C ONCLUSION
   425	Fig. 3. Examples of identifying pedestrians by the CAL model. The first
   429	TABLE III
   430	A BLATION EXPERIMENTS ON PRCC DATASET (%).
   431	Method
   445	Cloth-Changing
   454	In this work, we utilize a scheme based on stable diffusion
   455	model to generate pedestrian images with consistent clothing.
   456	The generating phase effectively controls the geometry and
   457	details of generated cloth-changing images by body keypoints.
   458	The consistent clothing CC-ReID data is filtered and used
   460	centroid loss. Our method mitigates clothing feature biases,
   462	pedestrians with similar clothing, which makes the CC-ReID
   463	more robust. Extensive experiments demonstrate the superiority of our method.
   465	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:59 UTC from IEEE Xplore. Restrictions apply.
   472	and X. Xie, “Learning resolution-adaptive representations for crossresolution person re-identification,” IEEE Transactions on Image Processing, 2023.
   476	Vision (ICCV), pp. 15847–15858, October 2023.
   482	(CVPR), pp. 22752–22761, June 2023.
   483	[6] K. Ren and L. Zhang, “Implicit discriminative knowledge learning
   496	“A cost-efficient approach for creating virtual fitting room using generative adversarial networks (gans),” International Journal of Advanced
   498	[10] D. Morelli, A. Baldrati, G. Cartella, M. Cornia, M. Bertini, and R. Cucchiara, “Ladi-vton: latent diffusion textual-inversion enhanced virtual
   500	Multimedia, pp. 8580–8589, 2023.
   501	[11] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,”
   505	[12] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image
   511	[14] P. Li, Y. Xu, Y. Wei, and Y. Yang, “Self-correction for human parsing,” IEEE Transactions on Pattern Analysis and Machine Intelligence,
   525	computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18,
   527	[19] I. Rocco, R. Arandjelovic, and J. Sivic, “Convolutional neural network
   528	architecture for geometric matching,” in Proceedings of the IEEE
   533	data with diffusion models for person re-identification pre-training,”
   534	arXiv preprint arXiv:2406.06045, 2024.
   539	[22] X. Gu, H. Chang, B. Ma, S. Bai, S. Shan, and X. Chen, “Clotheschanging person re-identification with rgb modality only,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
   542	sketch under moderate clothing change,” IEEE Transactions on Pattern
   543	Analysis and Machine Intelligence, vol. 43, no. 6, pp. 2029–2046, 2021.
   545	and X. Xue, “Long-term cloth-changing person re-identification,” in
   548	[25] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings
   564	[30] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”
   569	[32] Y. Huang, Q. Wu, J. Xu, Y. Zhong, and Z. Zhang, “Clothing status
   574	inspired cloth-debiasing for cloth-changing person re-identification,” in
   576	Pattern Recognition (CVPR), pp. 1472–1481, June 2023.
   577	[34] F. Liu, M. Ye, and B. Du, “Dual level adaptive weighting for clothchanging person re-identification,” IEEE Transactions on Image Processing, 2023.
   582	[36] P. Hong, T. Wu, A. Wu, X. Han, and W.-S. Zheng, “Fine-grained shapeappearance mutual learning for cloth-changing person re-identification,”
   586	by competition: Auxiliary-free cloth-changing person re-identification,”
   587	IEEE Transactions on Image Processing, vol. 32, pp. 2985–2999, 2023.
   589	unified network for cloth-changing person re-identification,” IEEE
   590	Transactions on Image Processing, vol. 32, pp. 4555–4566, 2023.
   592	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:59 UTC from IEEE Xplore. Restrictions apply.

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
exec
/bin/zsh -lc "pdfinfo 'CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf' | sed -n '1,80p' && pdftotext -f 1 -l 8 'CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf' - | nl -ba | rg -i \"method|approach|framework|overview|proposed|graph|memory|attention|experiment|ablation|analysis|3\\.|iii|iv|architecture\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           CMAG: Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification
Subject:         IEEE Transactions on Circuits and Systems for Video Technology;2026;36;1;10.1109/TCSVT.2025.3595846
Creator:         LaTeX with hyperref package
Producer:        pdfTeX-1.40.18; modified using iText® Core 7.2.4 (AGPL version) ©2000-2022 iText Group NV
CreationDate:    Mon Jan 19 16:19:37 2026 CST
ModDate:         Tue Jan 27 03:24:02 2026 CST
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
File size:       6575728 bytes
Optimized:       no
PDF version:     1.5
     5	CMAG: Cross-Modal Attention and
     6	Graph-Enhanced Memory for Unsupervised
    14	reidentification (USL-VI-ReID) has garnered widespread attention
    18	bias. This paper proposes the CMAG (Cross-Modal Attention
    19	and Graph-enhanced Memory) framework, which innovatively
    20	combines circular topology structure with cross-modal attention
    24	paths in feature space, effectively addressing the pseudo-label
    25	noise problem; (2) designing a cross-modal attention mechanism
    28	discrepancy issue; (3) constructing a graph-structured memory
    29	enhancement module with adaptive graph construction and
    33	Extensive experiments on SYSU-MM01 and RegDB datasets
    34	demonstrate the effectiveness of CMAG, achieving approximately
    35	3.5% improvement in Rank-1 accuracy and 2.8% in mAP
    36	on average compared to state-of-the-art methods, validating
    37	our approach’s advantages in addressing key challenges in
    41	Received 23 April 2025; revised 4 July 2025; accepted 31 July 2025.
    52	Ministry of Education, Guangxi Normal University, Guilin 541004,
    58	Normal University, Guilin 541004, China (e-mail: clzhang@gxnu.edu.cn;
    61	University of Science and Technology, Liuzhou 545006, China (e-mail:
    65	Index Terms—Unsupervised cross-modal re-identification, circular topology structure, graph-structured memory, cross-modal
    66	attention, vision transformer.
    72	N RECENT years, visible-infrared cross-modal person reidentification (VI-ReID) has received widespread attention
    79	methods have made significant progress in this field [12], [13],
    84	that fundamentally limit existing approaches: modality
    87	limitation of current methods stems from a fundamental
    94	Existing methods fail to effectively address these challenges
    95	due to specific technical limitations. For modality discrepancy, CNN-based approaches like DDAG [14] and Hi-CMD
    96	[15] rely on simple projections without fine-grained attention
    98	cross-modal alignment [16], [17]. For batch training limitations, current methods use either Cluster Memory (losing
    99	instance details) or Instance Memory (ignoring structured
   100	relationships) without effective global transfer mechanisms,
   102	pseudo-label noise, clustering methods like PGM [21] and
   104	[23], [24]. For camera view bias, methods like OTLA [25]
   109	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   116	observe from a color perspective). Core challenges in cross-modal person reidentification. (a) Modality difference and identity relationship representation:
   123	Conventional approaches assign pseudo-labels by selecting the most similar
   127	prototypes derived from clustering.(d) Camera view bias: Same person appears
   129	same identity from front, right side, back, and left side views respectively
   140	learning strategies [29], and self-supervised approaches [30],
   144	we propose the CMAG (Cross-Modal Attention and Graphenhanced Memory) framework with four targeted innovations:
   145	Dynamic Cross-modal Attention (addressing modality discrepancy), Graph-Structured Memory Enhancement (overcoming
   147	pseudo-label noise), and Camera-Aware consistency constraints (mitigating camera bias). Unlike existing approaches
   149	unified solution through novel integration of algebraic topology theory, Vision Transformer-specific attention mechanisms,
   152	lies in examining feature space from a topological perspective, introducing circular topology structure (CATS) theory
   157	pose variations. We also designed a cross-modal attention
   158	mechanism specifically for unsupervised visible-infrared person re-identification, called Dynamic Cross-modal Attention
   160	an innovative residual design that dynamically adjusts fusion
   163	The Graph-Structured Memory Enhancement Module
   164	(GSMEM) breaks through batch training limitations by maintaining a global feature memory bank and constructing
   165	cross-batch sample relationship graphs. Unlike existing methods relying solely on Cluster or Instance Memory, GSMEM
   166	organically integrates memory networks with graph neural networks. We also propose Camera-Aware consistency constraint
   172	• Circular topology structure (CATS) introducing algebraic topology theory for cross-modal pseudo-label validation, effectively addressing pseudo-label noise through
   173	transitivity verification.
   174	• Vision Transformer-specific cross-modal attention
   177	• Graph-structured memory enhancement (GSMEM)
   178	integrating memory networks with graph neural networks,
   181	reducing camera background bias through optimized clustering and adaptive filtering.
   185	to learn discriminative features without labels through
   186	pseudo-label generation and feature enhancement approaches.
   188	pseudo-supervised signals [16], [31], while feature enhancement methods employ contrastive learning with memory
   189	mechanisms. However, these single-modal approaches cannot
   194	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   196	ZHANG et al.: CMAG: CROSS-MODAL ATTENTION AND GRAPH-ENHANCED MEMORY FOR USL-VI-ReID
   199	dual contrastive learning (ADCA), Wang et al. [25] optimal
   200	transport (OTLA), and Wu and Ye [21] graph matching
   201	(PGM). The most recent works employ token-level attention mechanisms [27] and dual-stream contrastive learning
   202	approaches [34].
   205	employs feature alignment [12] or generative approaches.
   206	Ye et al. [35] proposed dual-stream architectures, while Wu
   207	et al. [13] designed mutual learning frameworks. These
   208	methods require extensive labeled data, limiting practical
   210	Key limitations include: most methods rely on CNN architectures without leveraging Transformer advantages, and lack
   211	effective global feature transfer mechanisms, resulting in significant cross-modal pseudo-label noise. While comprehensive
   212	approaches like GUR [26] address hierarchical discrepancy,
   214	attention, graph-enhanced memory, and topological structure
   216	C. Graph Neural Networks in Feature Learning
   217	Graph Neural Networks have been applied to person reidentification for structured modeling [36]. Memory-enhanced
   218	approaches like Li et al. [37] introduced global feature memory banks, while recent works have explored cross-modal
   219	graph applications. However, circular graph structures—paths
   225	and Zhao [40] proposed cross-modal attention Transformers,
   226	while Yang et al.’s SDCL [41] employed shallow-deep collaboration. However, existing methods typically adopt simple
   228	relationships. Our research focuses on ViT-specific crossmodal attention design through class token interaction and
   229	residual fusion, operating at the attention mechanism level
   234	A. Problem Definition and Framework Overview
   235	1) Two-Stage Learning Strategy: Our CMAG framework
   236	adopts a progressive two-stage learning approach to address
   238	extraction through a shared ViT backbone with modalityspecific clustering, building discriminative representations for
   239	both RGB and IR modalities. Stage 2 (100 epochs) activates our novel cross-modal components (CATS, DCAM-ViT,
   251	represent the i-th visible and infrared images respectively, Nrgb
   254	objective is to learn a unified feature space under unsupervised
   256	The CMAG framework employs a multi-task learning
   260	DCAM-ViT, graph-structured memory enhancement loss from
   262	These loss functions are jointly optimized with adaptive weight
   264	process. This multi-stage collaborative optimization allows
   265	the framework to efficiently handle modality differences, feature inconsistency, and camera view bias under unsupervised
   267	2) Framework Visualization Details: The visual elements
   269	CMAG. In the clustering phase (a), blue and orange bars represent RGB and IR features respectively, with different colored
   274	these paths. The memory module (c) shows both the temporal
   276	position) and adaptive graph construction (right, where node
   277	size reflects local density). Throughout the framework, blue
   282	III. P ROPOSED M ETHOD
   285	noise issues, and camera view bias. This section introduces our proposed CMAG (Cross-Modal Attention and
   286	Graph-enhanced Memory) framework, which addresses these
   287	challenges through four innovative components.
   291	network. Inspired by recent multi-token approaches [27],
   300	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   308	Fig. 2. Core module structure of the CMAG Framework with two-stage learning strategy (best viewed in color). The framework employs a two-stage
   309	learning approach: Stage 1 focuses on robust feature representation learning using a shared ViT backbone with modality-specific clustering, enabling effective
   310	feature extraction for both RGB and IR modalities (a). In Stage 2, the framework activates advanced cross-modal modules: (b) Cycle-Aware Topological
   311	Structure (CATS): Activated only in Stage 2 for circular path-based pseudo-label verification. Left shows feature distribution before clustering; right shows
   312	graph structure with circular paths. The formula C p = (A p ) A enables detection of circular paths like A−→B−→C−→D−→A. (c) Graph-Structured Memory
   313	Enhancement Module (GSMEM): Activated in Stage 2 for global cross-batch feature propagation. Left shows queue memory update mechanism maintaining
   314	a global feature memory bank; right demonstrates adaptive graph construction based on local density. Throughout both stages, clustering evolves from
   321	individual class tokens are obtained as Zi = Zmodality [i − 1] for
   330	extraction networks for visible and infrared modalities respectively. Each modality representation contains K class token
   335	multi-token framework of TokenMatcher [27], our ViT
   340	the motivation, theoretical foundation, and implementation of
   341	this approach.
   342	1) Problem Analysis and Method Motivation: A core challenge in unsupervised cross-modal learning is constructing
   344	Existing methods mainly focus on direct similarity relationships, neglecting higher-order structures in feature space,
   350	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   352	ZHANG et al.: CMAG: CROSS-MODAL ATTENTION AND GRAPH-ENHANCED MEMORY FOR USL-VI-ReID
   354	While existing methods address pseudo-label noise
   360	through intermediate poses, addressing the feature discontinuity problem that traditional methods cannot effectively solve.
   362	structures [42], [43] have special properties in graph theory
   363	[44], [45] and topology. From a topological perspective, a
   364	circular structure is characterized by non-trivial first-order
   366	space. Formally, in a graph G=(V,E), if nodes {v1 , v2 , . . . , vn }
   369	In the feature space of person re-identification, non-trivial
   372	a probability perspective [46], if random walks on the feature
   373	graph can form closed paths, the nodes on such paths are likely
   377	this long-range dependency relationship can effectively resist
   396	an adaptive k-nearest neighbor approach with label-aware
   425	Algorithm 1 CATS-Based Circular Graph Construction
   498	methods. When the direct similarity between two nodes (such
   499	as nodes A and D) is relatively low, traditional methods
   501	circular path A−→B−→C−→D−→A, our method can verify
   506	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   532	information, enhancing the diversity and robustness of feature
   546	represents the contrastive
   554	joint circular graph for structured information interaction
   572	D. Dynamic Cross-Modal Attention Mechanism for ViT
   573	1) Modality Difference Analysis and Challenges: Crossmodal scenarios face inherent modality differences that cause
   574	inconsistent feature distributions, requiring sophisticated attention mechanisms beyond traditional CNN-based approaches.
   575	Existing CNN-based cross-modal alignment methods employ
   577	utilize Transformer’s self-attention capabilities [47] and struggling to capture local correspondences and complex non-linear
   579	2) Cross-Modal Attention Design: Token-wise Crossmodal Attention: Building upon the multi-token architecture
   581	the individual class token level, where each token Zsource
   584	RB×d from the source modality interacts with the complete target modality representation Ztarget ∈ RB×K×d through attention
   589	Addressing these challenges, we design the first crossmodal attention mechanism specifically tailored for Vision
   590	Transformer architecture. This mechanism leverages ViT’s
   593	alignment through fine-grained attention computation. The
   595	In our cross-modal attention mechanism, individual class
   598	representation through the following attention computation:
   609	modality feature matrices respectively, B is the batch size,
   612	query, key, and value transformations respectively. The feature
   614	Second, we introduce a batch-wise dynamic attention
   632	From an information theory perspective [48], crossmodal learning must maximize mutual information between
   633	modalities while preserving modality-specific discriminative
   635	adaptive feature fusion network with a residual structure to
   636	balance these objectives:
   647	The key advantages of our cross-modal attention mechanism
   649	attention for flexible modality alignment, and adopting a
   652	E. Graph-Structured Memory Enhancement Module
   655	cross-modal learning through: (1) providing more negative
   656	samples for contrastive optimization [49], (2) reducing gradient variance (Var[∇Lbatch ] = σ2 /B) for stable convergence
   658	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   660	ZHANG et al.: CMAG: CROSS-MODAL ATTENTION AND GRAPH-ENHANCED MEMORY FOR USL-VI-ReID
   665	batch limitations through global memory banks and crossbatch relationship graphs.
   666	1) Global Memory Bank Design: Our GSMEM implements
   667	a dynamic queue memory update mechanism to maintain a
   668	global feature memory bank:
   679	where Mt ∈ R M×d is the memory bank at time step t, M is
   680	the memory bank size, f j ∈ Rd is the feature vector to be
   681	stored, ptr ∈ {0, 1, . . . , M − 1} is the current memory pointer,
   683	Additionally, the memory bank design maintains balance
   686	2) Adaptive Graph Construction and Feature
   688	a) Memory graph construction details: Our GSMEM
   689	constructs normalized k-NN graphs to facilitate cross-batch
   692	normalized k-NN graphs for stable feature enhancement.
   697	Z = Wnode · [fcurrent ; Mmemory ]
   701	where fcurrent represents current batch features, Mmemory represents memory bank features, and Wnode ∈ Rd×d is a learnable
   703	c) k-NN graph with degree normalization: Following the
   704	same k-NN approach as CATS:
   726	Based on the memory bank, we propose an adaptive
   727	k-nearest neighbor graph construction algorithm to capture
   736	projection matrix, and S is the similarity matrix. The adaptive
   744	d) Graph construction implementation: The similarity
   748	the concatenated current batch features f and memory bank
   751	neighbors, where ki is adaptively determined by:
   758	connectivity), base k = 10 (base neighbor count), and β =
   759	0.5 (regulation parameter). This adaptive mechanism reduces
   761	maintaining sufficient connectivity in sparse areas.
   762	As shown in Figure 2(c), our adaptive algorithm adjusts the
   767	After graph construction, we implement a residual graph
   791	dividing same identities across cameras or clustering different
   793	Existing methods either ignore camera information or simply incorporate camera ID as an auxiliary feature, failing to
   794	utilize intra-camera sample structure consistency to effectively
   797	camera-aware clustering method that utilizes camera-specific
   807	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   818	view changes. The choice of ε = 0.6 is based on extensive empirical validation across different camera views and
   822	3) Adaptive Consistency Constraint Strategy: Our key
   823	innovation lies in the adaptive post-processing strategy that
   825	variation approaches.
   843	labels, and c j is the camera ID. We propose an adaptive
   860	respectively.
   865	through adaptive probability decay, allowing the model to
   868	IV. E XPERIMENTS AND A NALYSIS
   871	standard protocol [12], we divide the dataset into training and testing sets of 206 identities each. Evaluation is
   875	adopt Cumulative Matching Characteristic (CMC) curve and
   878	top-k candidates, while mAP provides a comprehensive
   879	retrieval assessment. All experiments are repeated 10 times
   881	Training follows the two-stage protocol described in Section III-A over 150 epochs total.
   882	Our CMAG framework is implemented in PyTorch on an
   889	To evaluate CMAG’s effectiveness, we conduct comprehensive comparisons with state-of-the-art methods on
   892	person re-identification methods.
   893	1) Comparison With USVI-ReID Methods: Among directly
   894	comparable unsupervised methods, CMAG achieves leading
   897	the latest 2025 top-tier conference and journal methods. On
   899	mAP) significantly outperforms recent state-of-the-art methods
   900	including the latest AAAI’25 methods TokenMatcher (65.0%,
   902	top-tier methods such as TCSVT’25 APPD (64.7%, 60.5%),
   904	CMAG achieves 73.3% Rank-1 and 77.7% mAP, substantially outperforming the latest methods TokenMatcher (69.0%,
   905	74.9%) and N-ULC (67.0%, 73.08%). On RegDB dataset,
   906	CMAG demonstrates exceptional performance in both evaluation directions, achieving 94.7%/86.4% and 93.4%/85.6%,
   907	consistently outperforming the latest AAAI’25 method TokenMatcher (93.0%/86.3% and 91.8%/85.2%). These results
   908	validate the significant advantages of our approach over the
   918	C. Ablation Studies
   919	To verify the effectiveness of each component in the CMAG
   920	framework and their synergistic effects, we designed a series
   921	of ablation experiments on SYSU-MM01 and RegDB datasets,
   924	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.

exec
/bin/zsh -lc "pdfinfo 'CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf' | sed -n '1,80p' && pdftotext -f 1 -l 8 'CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf' - | nl -ba | rg -i \"method|approach|framework|overview|proposed|spiral|centering|experiment|ablation|analysis|3\\.|iii|iv|architecture\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdfinfo 'CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf' | sed -n '1,80p' && pdftotext -f 1 -l 8 'CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf' - | nl -ba | rg -i \"method|approach|framework|overview|proposed|camera|noise|suppression|experiment|ablation|analysis|3\\.|iii|iv|architecture\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           CLIP -powered modality centering with spiral training for visible-infrared person re-identification
Subject:         Pattern Recognition, 177 (2026) 113333. doi:10.1016/j.patcog.2026.113333
Keywords:        Person re-identification,Visible-infrared,Modality alignment,CLIP
Author:          Jianghao Xiong
Creator:         Elsevier
Producer:        Acrobat Distiller 8.1.0 (Windows)
CreationDate:    Thu Apr 30 14:01:25 2026 CST
ModDate:         Thu Apr 30 17:19:27 2026 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          yes
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           12
Encrypted:       no
Page size:       595.276 x 793.701 pts
Page rot:        0
File size:       4418653 bytes
Optimized:       no
PDF version:     1.7
     8	CLIP -powered modality centering with spiral training for visible-infrared
    21	School of Computer Science and Engineering, Sun Yat-Sen University, Guangzhou, 510006, China
    26	Guangdong Province Key Laboratory of Information Security Technology, Sun Yat-Sen University, Guangzhou, 510006, China
    28	d Key Laboratory of Machine Intelligence and Advanced Computing, Ministry of Education, Sun Yat-Sen University, Guangzhou, 510006, China
    45	diﬀerent modalities, speciﬁcally visible and infrared in this context. Given CLIP’s powerful cross-modal learning
    46	capabilities, we explore its potential to bridge these modality gaps. This paper introduces the Modality Centering
    47	with Spiral Training Network (MCST). We enhance text prompts by employing separable descriptions to independently capture personal and modality-speciﬁc information, thus disentangling identity-speciﬁc features from
    50	we propose a text-text centering loss to minimize distance between visible and infrared text representations, and
    51	an image-text centering loss to reduce discrepancies between image and text features. In addition, we introduce a
    52	novel spiral training strategy, which alternates the training of the text prompt and image encoder, ensuring consistency and improving the alignment of text and image features. Furthermore, we introduce CMG-P, a new visibleinfrared ReID dataset that includes challenging scenarios such as clothing changes and occlusions, oﬀering a more
    53	realistic evaluation benchmark. Extensive experiments demonstrate that our approach achieves state-of-the-art
    57	Visible-infrared person re-identiﬁcation (V-I ReID) focuses on identifying individuals across non-overlapping camera views that operate in
    59	modalities, V-I ReID facilitates reliable identity recognition under diverse lighting conditions, ranging from daylight to nighttime. This technology is particularly valuable in applications such as security surveillance, public safety, and intelligent transportation systems, where robust
    65	Traditional V-I ReID methods focus on extracting image features from
    72	These textual descriptions then guide the model to extract discriminative image features. This approach does not require manual annotation
    73	of text and simplify the training process. Instead, it harnesses the interpretive power of large models, oﬀering greater ﬂexibility and potential.
    74	Despite the remarkable success of CLIP-based ReID methods [6,7] in
    76	frameworks that only consider two visual modalities, CLIP-based methods introduce an additional semantic modality through text prompts,
    85	Received 9 January 2025; Received in revised form 13 February 2026; Accepted 16 February 2026
    93	cross-modal alignment, and naive pairwise alignment strategies are insuﬃcient to achieve identity-preserving modality-invariant representations.
    94	Furthermore, existing CLIP-based V-I ReID methods typically treat
    97	training paradigm, text prompts are derived from initial image representations and remain ﬁxed during subsequent training, causing inconsistency between textual and visual embeddings as the image features
    99	These challenges indicate that existing CLIP-based V-I ReID methods lack a principled mechanism to jointly model heterogeneous multimodal feature spaces and dynamically maintain semantic consistency
   100	during training. Therefore, a uniﬁed framework that can simultaneously
   103	Motivated by this insight, we propose a novel Modality Centering
   104	with Spiral Training Network (MCST) for V-I ReID, as illustrated in
   105	Fig. 1. The MCST framework comprises a Modality Centering (MC)
   106	strategy to mitigate modality discrepancies and a Spiral Training (ST)
   118	ReID datasets, CMG-P oﬀers two key advantages. The ﬁrst is its realworld diversity. The images are sourced from actual pedestrians and
   121	contains 36,031 visible images and 36,144 infrared images of 1011 individuals, which is one of the most extensive datasets in the ﬁeld. Further
   126	We propose a novel Modality Centering (MC) framework for VI ReID, which jointly aligns four heterogeneous feature spaces
   127	(visible/infrared images and texts) toward a shared identitydiscriminative embedding space, addressing both cross-modal and
   129	• We introduce a Spiral Training (ST) scheme that dynamically updates text prompts and image encoders, overcoming the semantic
   133	Fig. 1. Comparison of CLIP-ReID and our method. (a) CLIP-ReID uses a two-stage approach, ﬁrst optimizing the implicit text prompt and then training the image
   134	encoder. (b) Our method for VI-ReID integrates text prompts to create and align four distinct feature spaces, i.e visible image, infrared image, visible text, and infrared
   154	to guide image feature alignment, mainly for image-image ReID. CLIPReID [4] is a pioneering method that utilizes CLIP’s vision-language
   155	capabilities to learn discriminative image representations without explicit text labels, aligning visual features with learnable text prompts.
   157	for video-based ReID, enabling eﬀective cross-view and temporal feature alignment without relying on textual descriptions. PromptSG [19]
   158	is an end-to-end method that generates text prompts from image features using an inversion network and employs a language-guided crossattention module to perform text-image semantic guidance. The use of
   160	a one-stage method that enhances modality alignment by generating
   166	CSDN employ an entangled approach to integrate modality information into the ﬁnal representation-potentially compromising discriminative power-our method adopts a disentangled strategy. By separating
   169	for accurate person retrieval. Furthermore, our multi-stage framework
   170	dynamically adapts to diverse datasets while preserving tight alignment
   176	methods can generally be divided into feature alignment [8–10] and
   177	modality conversion [11–13] approaches. Feature alignment aims to
   180	capture rich cross-modal discriminative information. IDKL [10] aligns
   181	multi-modal features in a shared space through implicit discriminative knowledge learning. In contrast, modality conversion seeks to reduce cross-modal diﬀerences by transforming images from one modality
   182	into the other or a third modality. RBDF [11] employs two generative
   185	samples by mixing part descriptors and learns to identify discriminative body parts across modalities. CAJ+ [13] augments spectral channels and eﬀectively integrate complementary information from visible
   186	and infrared modalities. Both approaches aim to overcome modality
   187	discrepancies and extract modality-invariant human semantics. However, feature alignment often struggles to selectively preserve identitydiscriminative semantics while reducing modality diﬀerences. Meanwhile, modality conversion is non-directional and tends to introduce
   189	impact model training. Our method combines these two approaches, utilizing CLIP’s ability to understand images to generate text modality related to the image semantics. The text modality serve as a bridge to help
   191	In addition to supervised approaches, unsupervised V-I ReID has
   194	augmented dual-contrastive aggregation framework that maintains
   196	perspective, CHCR [15] adopts a hierarchical clustering and reﬁnement
   197	strategy to progressively establish reliable pseudo labels across modalities under large modality discrepancies. Although these works focus on
   198	label-free training, the CLIP-guided text modality in our method provides auxiliary cross-modal alignment signals and can be naturally integrated with pseudo-label or clustering-based pipelines, making our
   199	framework compatible with future label-eﬃcient or semi-supervised extensions.
   201	2.3. Vision-language models
   203	and language embeddings using large-scale image-text datasets, eﬀectively bridging the gap between visual and textual modalities to solve a
   205	It uses contrastive learning to map images and text into a shared semantic space, allowing zero-shot reasoning across diverse vision-language
   208	3. Preliminary
   209	3.1. Contrastive language-image pre-training (CLIP)
   210	CLIP is trained on a large-scale dataset of diverse image-text pairs,
   215	typically utilizes a Vision Transformer [28] or ResNet [29] architecture. Given an input image 𝑥 ∈ ℝ𝐻×𝑊 ×𝐶 , its feature representation is
   225	the challenges of diverse ReID tasks. Existing CLIP-based ReID methods can be divided into explicit text descriptions [16,17] and implicit
   226	text prompts [4,18,19] methods. The former involves obtaining text descriptions for each image through manual annotation or text generation
   243	4. Methodology
   244	In this section, we present the Modality Centering with Spiral Training Network (MCST). MCST alternates between two types of text descriptions during the learning process of text tokens, establishing a connection between visible-text and infrared-text to impose a constraint.
   247	Additionally, a spiral training approach is proposed to maintain the connection between the text embedding and the image embedding, facilitating better alignment.
   249	Fig. 2 illustrates the framework of the text token training stage. We
   251	decoupler. Let 𝑉𝑥 = {𝑥𝑣 } and 𝑈𝑥 = {𝑥𝑢 } represent the visible and infrared image set in a batch, respectively, where each set has 𝑁 samples.
   259	image-to-text contrastive loss is calculated as:
   291	and the text-to-image contrastive loss is calculated as:
   293	Fig. 2. Overview of our proposed CLIP-powered Modality Centering with Spiral
   294	Training Network (MCST). (a) Framework of the text token training stage: The
   295	separable text description decouples identity-aware and modality-aware information, while identity-aware text prompts are centered to preserve modalityinvariant information. (b) Framework of the image encoder training stage:
   347	where 𝑄(𝑦𝑡𝑎 ) is the positive image set related to text 𝑡𝑎 .
   348	Eqs. (2) and (3) connect the image and text representations for the visible and infrared modalities, respectively. However, what is required is
   352	denote the partial text description for person, we propose the visibleinfrared text centering loss 𝑣𝑡2𝑖𝑡 as follows:
   366	optimized over the 𝑁 × 𝑁 similarity scores for a contrastive objective.
   386	and the infrared-visible text centering loss 𝑖𝑡2𝑣𝑡 is:
   398	3.2. CLIP-ReID
   419	where 𝛼 is the margin. The text-to-text centering loss is then given by:
   433	 (⋅) are frozen in the text token training stage, and the overall objective
   446	Fig. 2 illustrates the framework of the image encoder training stage.
   449	leave out modality-speciﬁc information and propose the visible imagetext centering loss 𝑣𝑖2𝑡 as follows:
   487	and the infrared image-text centering loss 𝑖𝑖2𝑡 as follows:
   517	Fig. 3. Our proposed method establishes pairwise connections between the four
   527	centering loss is then given by:
   534	which have been proven eﬀective in ReID. The formula of ID loss is as
   624	other. When using the training approach of CLIP-ReID, the generated
   629	Therefore, we propose a Spiral Training (ST) strategy that alternately
   654	where (𝑥𝑗 , 𝑥𝑘 , 𝑥𝑙 ) denotes a triplet within each training batch for a given
   657	positive set, and 𝑆𝑗𝑛 represents the negative set. 𝑑𝑗𝑘
   659	pairwise Euclidean distances between the positive and negative sample
   660	pairs, respectively.
   665	5. Experiments
   668	4.3. Spiral training strategy
   670	Existing V-I ReID datasets are often limited in scale and diversity
   671	due to the challenges of capturing the same individuals across both day
   677	RegDB [31], SYSU-MM01 [32], and LLCM [33]. For all datasets, we utilize Cumulative Matching Characteristics (CMC) curves, mean Average
   678	Precision (mAP) and mean Inverse Negative Penalty (mINP) [30]as the
   688	dataset, with experiments conducted in the Infrared-Visible mode.
   718	encoder (⋅) need to be trained. The overall objective function in the
   730	Fig. 4. The Spiral Training (ST) strategy alternates between training the text prompt and the image encoder, allowing the parameters at each stage to be adjusted
   734	Method
   798	83.4
   810	43.3
   837	83.5
   840	83.9
   852	43.8
   870	83.3
   882	83.2
   895	Method
   975	73.0
   997	83.5
  1024	83.2
  1029	83.4
  1033	83.3
  1034	83.0
  1096	73.9
  1109	43.7
  1121	93.9
  1129	CMG-P is derived from the CM-Group dataset [34], with modiﬁcations to enhance its utility for visible-infrared person re-identiﬁcation
  1130	research. In CMG-P, each individual’s image is cropped from the original frames using bounding boxes provided in CM-Group. The dataset
  1131	captures 1011 individuals, resulting in a total of 72,175 images: 36,031
  1133	CMG-P utilizes six surveillance cameras strategically placed in diverse environments. Cameras 1–3 are RGB cameras, located on a road,
  1134	in a courtyard, and on a staircase, respectively. Cameras 4–6 are infrared
  1136	cameras, positioned on a sidewalk, in a hallway, and on a terrace, respectively. The raw video footage was recorded from April to September,
  1151	Method
  1206	93.3
  1235	83.4
  1249	63.2
  1296	83.3
  1336	93.8
  1349	Method
  1392	43.6
  1416	63.2
  1417	63.5
  1431	63.7
  1440	93.6
  1449	63.2
  1467	83.4
  1486	Analysis of the eﬀectiveness of separable text description (STD) based on CMC (%) and mAP (%) performance on the
  1487	CMG-P dataset, where ‘B’ represents the baseline method, ‘MCL’ refers to modality centering losses, and ‘ST’ denotes
  1488	spiral training.
  1493	Method
  1522	63.7
  1537	63.4
  1541	83.0
  1607	63.9
  1611	83.0
  1631	83.3
  1646	83.2
  1658	Analysis of the eﬀectiveness of modality centering losses based on CMC (%) and
  1660	centering loss and ‘I2T’ denotes image-to-text centering loss.
  1661	Method
  1721	83.0
  1743	83.3
  1753	83.2
  1763	5.1.3. SYSU-MM01
  1797	We implement our method using PyTorch and train it on an NVIDIA
  1806	Since no theoretical framework exists to guide the selection of optimal training sequences for diverse datasets, we determine these sequences empirically. For CMG-P, we use the training stage sequence
  1838	Analysis of the eﬀectiveness of the spiral training (ST) strategy based on CMC
  1840	baseline method, ‘STD’ denotes separable text description, and‘MCL’ refers to
  1841	modality centering losses.
  1842	Method
  1864	63.7 81.5 87.6
  1873	63.4 81.8 87.5
  1874	63.9 82.3 87.8
  1896	64.1 83.0 87.9
  1897	64.6 83.0 88.0
  1934	66.4 83.3 88.6
  1943	65.5 83.2 88.5
  1952	Analysis of the ﬁrst-stage selection based on CMC (%) and mAP (%) performance
  2094	5.3. Comparison with state-of-the-art methods
  2096	ReID approaches, and the results on the CMG-P dataset are shown in
  2097	Table 2. Note that none of the methods in this experiment use reranking. Our proposed MCST outperforms existing SOTA methods,
  2101	metric because its AIM module increases the similarity between positive
  2103	improving the average recognition rate. As shown in Table 3, MCST outperforms SOTA methods on the SYSU-MM01 dataset. With re-ranking,
  2106	approaches. Notably, it improves Rank-1 accuracy by 9.1% and mAP
  2113	with a total of 46,767 images. The dataset is divided into a training set

 succeeded in 0ms:
Title:           CLNS: Camera-aware label noise suppression for unsupervised visible–infrared person re-identification
Subject:         Pattern Recognition, 179 (2026) 113873. doi:10.1016/j.patcog.2026.113873
Keywords:        Unsupervised person re-identification,Cross-modality,Contrastive learning,Noisy labels
Author:          Sicheng Zhao
Creator:         Elsevier
Producer:        Acrobat Distiller 8.1.0 (Windows)
CreationDate:    Sat Apr 25 21:48:26 2026 CST
ModDate:         Tue May  5 21:26:52 2026 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          yes
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           12
Encrypted:       no
Page size:       595.276 x 793.701 pts
Page rot:        0
File size:       3451491 bytes
Optimized:       no
PDF version:     1.7
     8	CLNS: Camera-aware label noise suppression for unsupervised
    17	The MOE Key Laboratory of ICSP, IMIS Laboratory of Anhui, Anhui Provincial Key Laboratory of Multimodal Cognitive Computation, Zenmorn-AHU AI Joint
    18	Laboratory, School of Computer Science and Technology, Anhui University, Hefei, China
    19	b The school of Artificial Intelligence, Anhui University, Hefei, China
    20	c The School of Data Science (SDS), Chinese University of Hong Kong, Shenzhen 518172, China
    29	Contrastive learning
    33	Unsupervised visible–infrared person re-identification (US-VI-ReID) retrieves pedestrian images across modalities without manual annotations. To address camera-specific biases that fragment identities and amplify label
    34	noise, we propose the Camera-aware Label Noise Suppression (CLNS) framework, a coarse-to-fine pipeline
    35	that progressively purifies noise. Specifically, the Camera-aware Prototype Calibration (CPC) module exploits
    36	cross-camera consistency to rectify structural errors and construct reliable prototypes. Building on this,
    38	level, Neighbor-guided Camera-domain Learning (NCL) densifies feature distributions using soft supervision,
    39	while a Noise-aware Memory Updating (NMU) strategy prevents error accumulation. On the SYSU-MM01,
    41	54.6% (57.8%), respectively, significantly surpassing state-of-the-art methods. The code will be released at
    46	in recent work [1], matches pedestrian images across visible and infrared modalities, playing a pivotal role in intelligent surveillance.
    47	While supervised approaches have achieved remarkable progress, they
    48	rely heavily on large-scale, labor-intensive annotations [2–6]. Consequently, Unsupervised VI-ReID (US-VI-ReID) has garnered increasing
    51	discriminative representations without manual labels.
    52	A central challenge in US-VI-ReID lies in generating reliable pseudolabels. Mainstream approaches typically employ clustering algorithms
    53	(e.g., DBSCAN) to generate pseudo-labels from global feature similarities, as seen in PGM [9]. While effective in general scenarios, these
    54	methods often overlook a critical source of error: camera-specific bias.
    56	same camera often exhibit higher visual similarity than those from
    57	different cameras due to background and viewpoint consistencies. To
    58	quantitatively illustrate the severity of this issue, we conducted a
    60	preliminary analysis on the SYSU-MM01 dataset using the baseline
    61	method. Specifically, the average cosine distance of inter-camera positive pairs (images of the same identity from different cameras) is up to
    62	1.0615, which is significantly higher than that of intra-camera positive
    63	pairs (0.7795). This camera-induced discrepancy misleads clustering
    64	algorithms into splitting a single identity into multiple camera-specific
    66	noise not only corrupts the memory bank but also severs cross-modality
    67	correspondences, severely undermining training effectiveness.
    69	proposed, they mostly operate on the final clustering results without
    70	explicitly modeling or suppressing the underlying camera bias. Lacking this constraint, they are easily misled by the high intra-camera
    73	propose Camera-aware Label Noise Suppression (CLNS), a synergistic
    74	framework designed to progressively purify camera-induced noise from
    75	a coarse-to-fine perspective. Our method operates through a coherent
    82	Received 24 December 2025; Received in revised form 17 April 2026; Accepted 27 April 2026
    90	Fig. 1. Impact of camera-induced label noise. (a) Intra-camera visual similarity causes identity fragmentation, leading to (b) unreliable prototypes and (c) incorrect
    91	cross-modality alignment. In contrast, (a’–c’) CLNS calibrates camera biases to construct robust prototypes, establishing accurate correspondences.
    93	pipeline. Specifically, we first introduce a Camera-aware Prototype Calibration (CPC) module. By exploiting cross-camera neighborhood consistency, CPC filters out unreliable samples to rectify structural errors
    96	to refine feature distributions at the instance level, the Neighborguided Camera-domain Learning (NCL) module utilizes confidenceaware soft labels to suppress residual distribution noise. Finally, a
    97	Noise-aware Memory Updating (NMU) strategy adaptively re-weights
   102	modality gap. Ye et al. [15] proposed a channel-level perturbation strategy (CA) to improve robustness against color variations, while Chen
   106	scalability of S-VI-ReID in real-world deployments, motivating the shift
   109	Unsupervised VI-ReID (US-VI-ReID) [9] aims to learn discriminative representations without identity labels. The mainstream paradigm
   112	introduced a multi-memory matching (MMM) framework, which leverages multiple memory banks to capture diverse feature representations. Yang et al. [18] proposed SDCL, a shallow-to-deep collaborative
   113	learning approach that progressively mines fine-grained relations from
   118	this idea with DOTLA, enforcing dual-level transport constraints. Furthermore, graph-based approaches have shown promise in modeling
   119	intrinsic relationships. Wu et al. [9] proposed PGM, utilizing graph
   123	However, these methods predominantly rely on global feature clustering, which is inherently sensitive to intra-class variance. Crucially,
   124	they often overlook camera-specific bias, a primary factor that fragments identities into disjoint clusters. Without explicitly modeling this
   128	• We propose the CLNS framework to address the camera-specific
   129	bias in US-VI-ReID. By progressively purifying noise from coarselevel prototypes to fine-grained instances, CLNS ensures robust
   132	gatekeepers. CPC rectifies pseudo-label errors via cross-camera
   135	noise. NCL compacts feature distributions using soft supervision,
   137	Extensive experiments on SYSU-MM01, RegDB, and LLCM datasets
   138	demonstrate that CLNS achieves state-of-the-art performance, validating the effectiveness of suppressing camera-aware noise.
   141	Supervised Visible–Infrared Person Re-identification (S-VI-ReID) relies on cross-modality annotations to learn shared feature spaces. Existing approaches typically tackle the modality discrepancy through two
   143	Feature alignment methods aim to bridge the modality gap by projecting features into a common subspace. For instance, Wang et al. [12]
   144	proposed the D2RL framework, which optimizes feature embeddings
   147	cross-modality metrics, effectively aligning distributions across different spectrums. More recently, Fang et al. [14] focused on fine-grained
   151	2.3. Learning with noisy labels
   152	Since unsupervised clustering inevitably introduces noise, mitigating noisy pseudo-labels is a central challenge. Existing solutions generally fall into two categories: label correction and noise-tolerant loss
   154	Label correction methods aim to refine noisy targets using model
   155	predictions or structural constraints. For example, Yin et al. [22] proposed RPNR, which exploits neighborhood consistency to select highconfidence samples for reliable cluster reconstruction. In their subsequent work, Yin et al. [23] introduced APPD, further enhancing
   162	Fig. 2. The proposed CLNS framework. It consists of five key components: (a) shows the initial clustering; (b) CPC calibrates camera-biased prototypes; (c) OTPM
   168	label purification by adaptively adjusting the pseudo-label distribution.
   170	to identify and eliminate unreliable labels based on cross-channel consensus. On the other hand, noise-tolerant loss functions aim to downweight the contribution of unreliable samples during optimization.
   174	Despite these advances, most existing works treat label noise as
   176	ignore the systematic camera-domain gap as a structural noise source.
   177	Unlike random noise, camera-induced noise is structured and consistent, which requires specific handling. In contrast to previous approaches, our CLNS framework explicitly models and suppresses this
   178	camera-induced noise, providing a more fundamental solution to label
   189	the iteration index. The training objective consists of two components.
   190	Intra-modality contrastive learning. For a query 𝑞 𝑡 , the InfoNCE loss pulls
   202	Inter-modality contrastive learning. In the second stage, cross-modality
   216	3. Method
   217	3.1. Preliminaries and baseline
   234	where 𝑆 and 𝐿 are the number of clusters. To explicitly model cameraspecific biases, we partition each cluster into finer camera domains
   235	based on camera IDs. Let 𝑣 = {𝜙𝑣1 , … , 𝜙𝑣𝑆×𝐾 } and 𝑟 = {𝜙𝑟1 , … , 𝜙𝑟𝐿×𝐾 }
   238	represent the sets of camera-domain centroids, computed by averaging
   239	features within each camera-specific subset of a cluster, where 𝐾𝑣 and
   240	𝐾𝑟 denote the number of cameras in each modality.
   251	Our proposed modules are integrated into the second stage to robustify this learning process. The overall framework of CLNS is illustrated
   253	3.2. Camera-aware prototype calibration
   254	Standard clustering often yields noisy pseudo-labels. While neighborhood consistency strategies like RPNR [22] effectively filter random
   255	noise, they fail in US-VI-ReID contexts. High visual similarity within
   256	the same camera view (due to background and lighting) creates false
   257	neighborhoods, causing standard methods to reinforce camera-specific
   259	Camera-aware Prototype Calibration (CPC) module, which exploits
   260	cross-camera consistency to filter unreliable samples and construct
   261	robust, camera-invariant prototypes.
   273	intra-camera bias by masking distances between samples from the same
   274	camera:
   284	3.4. Neighbor-guided camera-domain learning
   285	While OTPM establishes reliable correspondences, residual camera
   286	noise can still destabilize feature learning. To address this, we propose
   287	the Neighbor-guided Camera-domain Learning (NCL) module. Unlike
   289	camera-domain centroids as fine-grained proxies to explicitly capture
   290	and adapt to camera-specific variations.
   291	Given the training features {𝑈𝑣 , 𝑈𝑟 } and the camera-domain centroid sets 𝑣 and 𝑟 , we first initialize the centroid for the 𝑙th visible
   292	camera domain 𝑐𝑙𝑣 as:
   299	where 𝑐𝑎𝑚𝑣𝑖 denotes the camera ID. Based on ̃
   301	of 𝐾1 nearest cross-camera neighbors 𝑖𝑣 . Sample 𝑖’s reliability is
   312	the same identity with neighbors from different views, providing a robust signal against camera-specific outliers. Consequently, we construct
   325	to the 𝑙′ -th visible camera domain, let its matched infrared camera
   329	visible camera domain:
   345	camera. To ensure compatibility with such single-camera identities,
   352	Similarly, the inter-modality correlation with any 𝑙th infrared camera
   366	neighborhood and the target camera domain, serving as a reliable soft
   368	To mitigate label noise, we fuse the hard one-hot pseudo-label with
   375	3.3. Optimal transport prototype matching
   376	While CPC ensures intra-modality purity, the semantic correspondence between visible (𝑃 𝑣 ) and infrared (𝑃 𝑟 ) prototypes remains unknown due to independent clustering. Existing methods [19] typically
   379	structure caused by camera variations, leading to suboptimal matching.
   381	establishing correspondences at both the cluster level and the cameradomain level (), OTPM ensures alignment in both abstract identity
   384	camera-domain centroid sets 𝑣 , 𝑟 (as defined in Section 3.1). Since
   400	vector for the assigned domain 𝑙′ , and matched domain 𝑀, respectively.
   499	From these optimal plans, we derive probabilistic mappings to guide
   504	camera-domain centroids 𝜙𝑣→𝑟 and 𝜙𝑟→𝑣 from 𝑄∗𝑑𝑜𝑚 . This dual-matching
   506	identity consistency, while camera-domain correspondences capture
   518	is derived from the consistency score of its assigned domain:
   562	where 𝑤 controls the sensitivity. Samples with consistent neighborhoods receive higher weights, while outliers are down-weighted. The
   569	4. Experiments
   571	final NCL objective aggregates the weighted losses over all samples in
   583	We evaluate our method on three public benchmarks: SYSU-MM01
   585	SYSU-MM01 is a large-scale dataset captured by 6 cameras in indoor and outdoor environments. It contains 22,258 visible and 11,909
   590	infrared images. The dataset is randomly divided into two halves (206
   595	using a 9-camera network. It comprises 46,767 annotated images of
   599	Evaluation metrics. We employ Cumulative Matching Characteristics (CMC), mean Average Precision (mAP), and mean Inverse Negative
   607	proposed framework is 𝑡𝑜𝑡𝑎𝑙 = 𝑏𝑎𝑠𝑒 + 𝑛𝑐𝑙 . By operating at the
   608	camera-domain level, NCL effectively densifies feature distributions
   609	and mitigates residual noise, complementing the structural alignment
   611	3.5. Noise-aware memory updating
   614	from the true distribution. To counteract this, we propose the Noiseaware Memory Updating (NMU) strategy, which adaptively re-weights
   616	Consider a camera domain with centroid 𝝓𝑘 and a batch of assigned
   617	samples 𝑘 = {𝐟𝑖 }𝑏𝑖=1 . We first quantify the ‘‘noise probability’’ of each
   646	To ensure a fair comparison and isolate the effectiveness of our
   647	proposed modules, we construct our framework based on the two-stage
   650	of CEIL with our proposed CPC, OTPM, NCL, and NMU modules.
   658	epochs, linearly increasing the learning rate from 3.5×10−6 to 3.5×10−4 .
   659	Subsequently, the learning rate is decayed to 3.5 × 10−5 at epoch 20 and
   660	further to 3.5 × 10−6 at epoch 50.
   664	to 0.6 and the minimum number of samples set to 4. For the contrastive
   666	Our proposed modules are formally integrated into the training
   667	process during the second stage. All experiments are conducted on a
   675	Finally, the centroid is updated via 𝝓𝑘 ← 𝛽𝝓𝑘 + (1 − 𝛽)𝐟̄𝑘 . By filtering noise contributions, NMU maintains a pure representation of the
   676	camera domain throughout training.
   678	true label noise from informative ‘‘hard positives’’ solely via angular deviations. To handle this ambiguity, NMU eschews hard filtering in favor of a Softmax-based soft-weighting mechanism. This robust design assigns ambiguous samples diminished yet strictly nonzero weights: it effectively suppresses potential noise to prevent centroid drift while retaining hard positives to support the learning of
   693	Construct camera-domain centroid sets 𝑣 and 𝑟 based on refined
   696	Initialize memory banks and camera-domain prototypes for both
   700	cluster and camera-domain levels via OTPM (Eq. (9))
   710	Update camera-domain centroids in the memory bank via NMU
   717	4.3. Comparison with state-of-the-art methods
   719	methods: Supervised VI-ReID (S-VI-ReID), Semi-supervised VI-ReID (SSVI-ReID), and Unsupervised VI-ReID (US-VI-ReID). The quantitative
   722	Comparison with S-VI-ReID methods. Despite being fully unsupervised, CLNS competes with supervised counterparts that rely on
   730	Performance comparison with state-of-the-art methods on SYSU-MM01 and RegDB.
   745	Method
   819	73.81
   822	73.0
   825	73.0
   829	53.61
   830	53.26
   846	83.20
   882	83.60
   914	73.6
   934	73.78
   980	53.12
   990	63.0
   998	63.83
  1013	43.9
  1027	83.7
  1035	83.5
  1086	43.9
  1093	63.24
  1095	63.68
  1111	43.7
  1117	53.47
  1123	73.28
  1149	73.5
  1155	23.81
  1176	83.7
  1178	83.8
  1213	63.81
  1217	83.2
  1271	expensive manual annotations. On SYSU-MM01, CLNS achieves results
  1272	comparable to representative supervised methods such as AGW, NFS,
  1274	surpasses most supervised methods, highlighting its ability to learn
  1275	discriminative features from unlabeled data. Furthermore, on the LLCM
  1277	performs on par with strong supervised approaches like AGW and CA.
  1278	This demonstrates that our camera-aware strategy mitigates domain
  1279	noise, enabling the model to adapt to difficult environments without
  1281	Comparison with SS-VI-ReID methods. Semi-supervised methods
  1284	methods on RegDB and LLCM without using any identity labels, showcasing the superior efficiency of our noise suppression framework. On
  1285	the large-scale SYSU-MM01 dataset, CLNS also surpasses representative methods such as OTLA, DPIS and CGSFL. This confirms that by
  1286	explicitly modeling camera bias, unsupervised learning can achieve
  1288	from label noise in the unlabeled subset.
  1289	Comparison with US-VI-ReID methods. Comparisons with existing unsupervised methods confirm that CLNS sets a new state-of-the-art
  1293	substantial margins of 1.30%, 1.96%, and 2.55%, respectively. Similar
  1295	the RegDB dataset, our method achieves saturating performance with
  1296	94.79% and 95.33% Rank-1 for VIS-to-IR and IR-to-VIS modes, respectively, averaging over 1% improvement across all metrics. Furthermore,
  1301	Performance comparison with state-of-the-art methods on LLCM.
  1302	Methods
  1341	63.7
  1346	63.2
  1351	63.6
  1367	43.6
  1382	63.2
  1383	63.5
  1385	63.5
  1412	43.8
  1439	23.5
  1451	33.3
  1470	43.4
  1533	An effectiveness analysis of different components is conducted on the SYSU-MM01 dataset.
  1633	63.25
  1634	63.91
  1652	63.73
  1682	73.45
  1707	73.38
  1710	73.80
  1711	73.70
  1728	Fig. 3. Training dynamics on SYSU-MM01. Evolution of pseudo-label quality (ARI) and camera bias (Inter-camera feature distance) during training.
  1730	Comparative analysis of cross-modality alignment strategies on the SYSUMM01 dataset. We evaluate the impact of the matching algorithm (PGM vs.
  1735	not merely overfitting to specific domains but provides a generalized solution for alleviating camera-induced label noise and enhancing
  1738	Methods
  1740	4.4. Ablation study
  1748	We conduct ablation studies on the SYSU-MM01 dataset to evaluate
  1750	Table 3, where ‘‘Baseline’’ denotes the model without our proposed
  1752	overhead compared to the Baseline. Notably, all proposed auxiliary
  1753	modules are exclusively active during training to regularize the feature
  1755	retrieval. Consequently, CLNS introduces strictly zero additional computational overhead during testing. Given the substantial performance
  1758	Effectiveness of the CPC module. CPC acts as a structural gatekeeper to mitigate camera-induced label noise. Quantitative comparisons in Table 3 (Index 3 vs. 4; Index 8 vs. 13) show consistent
  1760	tracks pseudo-label quality (ARI) and camera bias (mean inter-camera
  1761	feature distance). We observe a distinct inverse correlation: the significant reduction in camera bias directly parallels the rise in ARI
  1762	across both modalities. This confirms that CPC effectively rectifies
  1763	distribution shifts caused by camera views, preventing the model from
  1794	33.97
  1801	73.45
  1813	73.80
  1814	73.7
  1815	73.38
  1819	quality with minimal camera discrepancy. This conversely validates
  1821	where camera-induced fragmentation is severe (see Fig. 4).
  1822	Effectiveness of the OTPM module. The OTPM module is pivotal
  1843	Methods
  1849	Ablation study on the distinct roles of CPC (Label Quality) and NCL (Feature
  1854	Method variant
  1870	joint optimization effectively handles both identity discrimination and
  1872	Effectiveness of the NCL module. While CPC corrects structural
  1878	Effectiveness of the NMU module. NMU serves as a critical stabilizer against memory bank drift. Removing NMU (Index 10 vs. 13)
  1917	4.6. Parameter analysis
  1918	We systematically evaluate the sensitivity of CLNS to five key
  1924	For 𝐾1 , a moderate value balances the receptive field: a larger 𝐾1 incorporates diverse cross-camera views to enrich structural information,
  1925	while an excessive value introduces irrelevant noise. Regarding 𝜌, its
  1927	requires a higher threshold (𝜌 = 0.85) to enforce strict noise filtering
  1928	and ensure prototype purity. In contrast, the relatively cleaner SYSUMM01 benefits from a lower threshold (𝜌 = 0.8) to exploit informative
  1929	hard samples and maintain feature diversity.
  1935	smaller (RegDB) or noisier (LLCM) datasets prefer smaller neighborhoods (𝐾2 = 40 and 15, respectively) to prevent the aggregation of
  1938	soft neighborhood-guided labels. Experiments show that 𝑢 = 0.85 works
  1941	4.5. Module interaction and synergy analysis
  1942	Beyond verifying individual contributions in Section 4.4, we investigate the internal synergy among modules. Specifically, we analyze the
  1947	derived from cluster prototypes. If prototypes are corrupted by cameraspecific biases, the resulting transport plan is inevitably suboptimal.
  1951	accuracy caps at 0.748 due to camera-distorted visual prototypes.
  1956	both modules utilize neighborhood information, they target noise at

hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdfinfo 'ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf' | sed -n '1,80p' && pdftotext -f 1 -l 8 'ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf' - | nl -ba | rg -i \"method|approach|framework|overview|proposed|color|sketch|texture|experiment|ablation|analysis|3\\.|iii|iv|architecture\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdfinfo 'Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf' | sed -n '1,80p' && pdftotext -f 1 -l 8 'Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf' - | nl -ba | rg -i \"method|approach|framework|overview|proposed|generation|coarse|fine|vehicle|experiment|ablation|analysis|3\\.|iii|iv|architecture\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Title:           Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data
Subject:         2025 IEEE International Conference on Robotics and Automation (ICRA);2025; ; ;10.1109/ICRA55743.2025.11127895
Creator:         The PaperPlaza Conference/Journal Management System
Producer:        PDFlib+PDI 8.0.1p8 (Perl 5.10.0/Linux-x86_64); modified using iTextSharp 5.4.1 ©2000-2012 1T3XT BVBA (AGPL-version); modified using iText® Core 7.2.4 (AGPL version) ©2000-2022 iText Group NV
CreationDate:    Tue Jun  3 03:25:15 2025 CST
ModDate:         Sun Aug 24 11:26:37 2025 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           8
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
File size:       4224547 bytes
Optimized:       no
PDF version:     1.5
     4	Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle
     7	2025 IEEE International Conference on Robotics and Automation (ICRA) | 979-8-3315-4139-2/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICRA55743.2025.11127895
    10	Abstract— Due to the critical issues of privacy and partial
    12	in vehicle recognition systems. Consequently, researchers have
    13	increasingly turned towards vehicle re-identification (reID)
    17	To address this challenge, this paper introduces a coarse-to-fine
    18	generation pipeline designed to synthesize high-fidelity vehicle
    19	data, thereby facilitating subsequent vehicle representation
    20	learning. Specifically, the proposed approach consists of three
    21	stages: Prompt Processing, Diffusion Fine-tuning, and Semantic Filtering. First, we collect detailed prompts from vehicle
    22	websites and companies with fine-grained vehicle prototype
    24	automotive prototypes to fine-tune diffusion models. Finally,
    27	we validate the effectiveness using vanilla models. Extensive experimental evaluations demonstrate that our approach achieves
    28	competitive accuracy on public benchmarks such as VeRi-776,
    29	VehicleID and CityFlowV2, and is compatible with various
    30	model architectures.
    33	Vehicle re-identification (reID) aims to match images of
    34	the same vehicle across multiple cameras, which is crucial
    35	for the deployment of autonomous vehicles [1] and intelligent traffic systems [2]. Given the minor intra-class differences between car models, vehicle reID is typically treated as
    36	a fine-grained representation learning task [3], [4]. However,
    37	privacy concerns [5] and annotation difficulties in multisensor systems [6], [7] result in a scarcity of realistic training
    39	has focused on generating synthetic data for vehicle reID.
    43	Existing efforts on vehicle reID data generation can be
    44	divided into two directions: 1) Graphics-engine-based methods, such as PAMTRI [8] and VehicleX [9]. They employ 3D
    46	University of Singapore, Singapore 117417 e0792447@u.nus.edu,
    50	University, China 215163 weiji@nju.edu.cn
    51	3 Zhedong Zheng is with the FST and ICI, University of Macau, China
    53	This work is supported by the University of Macau Start-up Research Grant SRG2024-00002-FST and Multi-Year Research Grant MYRGGRG2024-00077-FST-UMDF
    57	Fig. 1: We compare our Vehicle-Diff dataset to existing
    59	are based on 3D engines (PAMTRI [8] and VehicleX [9]),
    60	while PTGAN [11] and VehicleGAN [10] adopt the datadriven structure, i.e., Generative Adversarial Networks [12].
    61	We could observe that the proposed method is with a closer
    62	visual appearance compared to the real dataset, i.e., VeRi776. Besides, the generated images by the proposed method
    64	knowledge to guide generation.
    65	CAD models to generate vehicle images. While these methods have made significant strides, they still face challenges.
    67	vehicle images and actual real-world images. Additionally,
    68	the process of generating the VehicleX dataset relies heavily
    69	on a large amount of labeled vehicle re-identification data,
    70	which is costly and raises privacy concerns. Similarly, synthetic data from PAMTRI needs to be combined with fully
    71	labeled re-identification datasets. 2) Data-driven methods,
    72	such as generative adversarial networks (GANs) [12]. For
    73	instance, PTGAN [11] and VehicleGAN [10] explore GANs
    74	to synthesize novel vehicle views. Although these methods
    75	generate vehicle images with relatively good visual quality,
    77	fine-grained attributes of the same vehicle are often inconsistent, compromising the training process of the vehicle reID.
    79	Vehicle-Diff, a new pipeline designed to synthesize large-
    83	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.
    85	scale training data for vehicle re-identification, facilitating
    89	prompt for vehicles with a focus on the vehicle attribute. To
    91	we employ carefully crafted prompts. Then, we fine-tune
    93	during the generation stage. It enables the diffusion model
    94	to adapt to the target vehicle domain at both the content
    99	reducing labeling costs and privacy concerns. As shown in
   100	Fig. 1, the generated vehicle images are much closer to the
   101	real-world data. Finally, we construct a new labeled vehicle re-identification dataset, called Vehicle-Diff, comprising
   102	149,472 images of 4,940 distinct vehicles. The efficacy of
   103	Vehicle-Diff is substantiated through comparative evaluations
   104	with synthetic datasets produced by existing approaches. In
   106	• A new coarse-to-fine cross-modality generation pipeline
   108	vehicle re-identification dataset tailored to a downstream
   112	generation with attributes for vehicle re-identification.
   113	• Extensive experiments have validated that our pipeline
   116	The proposed method has achieved competitive performance, e.g., 83.79% mAP on the VeRi-776 dataset.
   118	Vehicle Re-Identification. Vehicle re-identification (reID)
   119	involves retrieving vehicles of interest from a database of
   122	supervised learning. However, this approach faces challenges
   123	such as high annotation costs and privacy concerns when
   126	reduce annotation requirements. Despite these efforts, substantial real data is still needed for general vehicle reID
   129	approach that significantly reduces the need for both real data
   130	and annotations, addressing these limitations effectively.
   131	Synthetic Datasets for Vehicle Re-Identification Task.
   132	Synthetic data are increasingly used to address privacy concerns and high annotation costs in creating re-identification
   135	vehicles, but these assets suffer from the intrinsic domain gap
   138	create. VehicleGAN [10] and PTGAN [11] deploy GANs for
   139	data augmentation, with VehicleGAN focusing on AutoReconstruction and pose consistency, and PTGAN generating
   140	novel vehicle views based on given poses. However, these
   141	methods still require large labeled datasets for effective
   144	approach reduces the need for both real data and annotations,
   145	addressing these limitations effectively.
   147	[27] have recently emerged as promising generative models,
   148	particularly for text-to-image generation, where they can
   152	methods like [31], [32], [33] have utilized diffusion models,
   153	e.g., GLIDE [34], to generate synthetic data for image classification. Despite their impressive visual outcomes and applications, the potential of text-to-image diffusion models for
   154	vehicle re-identification remains underexplored. In this paper,
   157	vehicle re-identification performance.
   158	III. METHOD
   159	An overview of Vehicle-Diff is provided in Fig. 2. VehicleDiff generates high-fidelity data in a coarse-to-fine manner
   161	(1) prompt processing, (2) diffusion fine-tuning, and (3)
   162	semantic filtering. First, the prompt processing stage (§IIIA) constructs a prompt library and specifies vehicle attributes
   163	such as models and colors for image generation. Next, during
   164	the diffusion fine-tuning stage (§III-B), Vehicle-Diff finetunes the diffusion model using unlabeled vehicle images,
   165	improving its adaptation to vehicle image generation. Finally,
   166	in the semantic filtering stage (§III-C), Vehicle-Diff generates
   167	vehicle images with different IDs using the prompt library
   168	and fine-tuned model, followed by filtering these images
   171	The prompt processing stage aims to construct discriminative vehicle attribute prompts to guide image generation, thus
   172	enhancing inter-class consistency and intra-class diversity.
   173	We first filter the noisy online information to collect vehicle
   185	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.
   195	Chevrolet Impala vehicle
   203	Vehicle model
   207	Chevrolet Impala vehicle
   212	Vehicle model
   231	a black car driving
   251	Fine-Tuned
   265	Stage2: Diffusion Fine-Tuning
   383	Fig. 2: An overview of our coarse-to-fine cross-modality pipeline Vehicle-Diff. It has three stages: Prompt Processing,
   384	Diffusion Fine-tuning, and Semantic Filtering. (1) We first scrape and filter vehicle model information from online vehicle
   385	websites. Given the diffusion model, we then select the prompt template according to the visual quality. (2) In the second
   386	stage, we leverage the off-the-shelf image captioner to generate the pseudo caption. It is worth noting that the proposed
   387	pipeline only requests a few unlabeled real images from the downstream dataset. After the data preparation, we fine-tune
   388	the diffusion model via Mean Squared Error (MSE) loss. (3) In the third stage, using the refined prompts, we choose the
   389	most effective diffusion model by comparing visual quality, such as consistency. Then, we create synthetic data for the
   390	vehicle re-identification task. We use the cross-modality model to filter out semantically misaligned data. Finally, we feed
   393	[production year] [brand] [car model] [body style] driving
   396	B. Diffusion Fine-tuning
   397	Vehicle-Diff leverages a text-to-image diffusion model
   398	to generate vehicle images according to prompts. However, a pre-trained diffusion model still struggles to adapt
   399	well to the real-world vehicle images, resulting in a domain gap between synthesized images and those in vehicle
   400	reID datasets. Therefore, we further fine-tune the diffusion
   402	its generation capability. As shown in Fig. 2 (Stage 2), we
   403	illustrate the step-by-step fine-tuning stage from the data
   406	text prompts for unlabeled vehicle images, and then employ
   408	the generated image-text pairs to fine-tune the text-to-image
   412	final visual style, while maintaining the generative capability.
   413	The optimization objective is the mean squared error (MSE)
   414	loss. It is worth noting that, our Vehicle-Diff could be trained
   415	with only a few (1%) unlabeled images of the vehicle
   416	dataset for fine-tuning, i.e., 378 images for VeRi-776 and
   417	527 images for CityFlowV2, while previous methods either
   418	require large-scale datasets (GAN-based methods [10], [11])
   419	or rely on labeled images (graphics-engine-based methods
   420	[8], [9]). Moreover, different from these methods, VehicleDiff harnesses the generative power of diffusion models,
   422	Fig. 1. Similarly, we fine-tune multiple candidate diffusion
   426	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.
   430	We first sample approximately 10 prompts from the optimized prompt library to evaluate and select the optimal finetuned diffusion model. With a similar idea to our prompt
   431	template design, the selection of the fine-tuned model is
   432	informed by a qualitative assessment of the images generated
   433	by each candidate model. Fig. 2 (Stage 3) provides illustrative examples of fine-tuned models evaluated alongside the
   435	we opt for the fine-tuned diffusion model that maintains the
   437	prompts into the optimal fine-tuned diffusion model, which
   438	generates synthetic images automatically. Because of the limitations of text-to-image generation models in producing finegrained and controllable outputs, directly using generated
   439	images is insufficient for training vehicle re-identification
   442	portions of the images that include the high-quality vehicle.
   444	those with multiple vehicles, fragmented vehicles, or no
   445	vehicle at all. To tackle this issue, we utilize the YOLOv5x6
   447	images, for vehicle detection and cropping. The model is
   448	configured to detect only vehicle categories, with a single
   450	vehicle. We retain images with high-confidence detections
   451	and discard vehicles smaller than or equal to 250 pixels
   452	in height or width. After cropping, we have the vehicle in
   454	images with semantic misalignment, such as vehicles with
   460	vehicle,” where the color term is dynamically substituted
   461	from a predefined color list, such as “red,” “yellow,” “green,”
   485	PKU-Vehicle [51]
   489	VehicleID [54]
   490	VehicleReID [55]
   496	VehicleX [9]
   497	Vehicle-Diff
   564	synthetic vehicle re-ID datasets in terms of the number of
   565	vehicle IDs, images, and viewpoints, and the availability of
   566	attributes. † : Number of images in their code. ‡ : Given more
   573	to classify different vehicles, and the Lcircle is to optimize
   574	the representation space by pulling closer positive images,
   575	while pushing away the negative samples. We apply the same
   579	IV. E XPERIMENT
   581	Synthetic data generation. The Diffusion Fine-tuning process uses the Adam optimizer [49], with a learning rate of
   586	set to 1024 × 1024. The vehicle detection threshold is set to
   587	0.65. Our generation pipeline, Vehicle-Diff, yields 149,472
   588	images of 4,940 vehicles on VeRi-776.
   594	our Vehicle-Diff and other existing vehicle re-ID datasets. We
   597	IDs compared with VehicleX [9]. It is worth noting that our
   598	proposed Vehicle-Diff could further generate more images,
   599	if more text prompts are provided. In Tab. II, Tab. III
   600	and Tab. IV, we compare our proposed Vehicle-Diff with
   601	existing vehicle re-ID methods on three real-world datasets,
   602	i.e., VeRi-776 [56], VehicleID [54] and CityFlowV2 [58],
   603	respectively. For a fair comparison, we follow the setting in
   606	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.
   608	Fine-grained differences between
   611	Fine-grained differences between body
   615	Fine-grained differences between
   621	Widebodydriving down the road
   627	Hatchback driving down the road
   631	SUV driving down the road
   635	Fig. 3: Our pipeline reflects the fine-grained discrepancy between two appearance-similar vehicles, e.g., front grilles, rear
   636	lights, and body types, while we also depict reasonable intra-class variations of the same vehicle, such as vehicle pose.
   637	Method
   638	VehicleX [9]
   639	Vehicle-Diff
   640	VehicleX [9]
   641	Vehicle-Diff
   645	VehicleX [9]
   646	Vehicle-Diff
   649	VehicleGAN [10]
   652	VehicleX (PCB) [9]
   653	Vehicle-Diff (PCB)
   662	VehicleX [9]
   663	Vehicle-Diff
   664	VehicleX [9]
   665	Vehicle-Diff
   748	93.44
   751	93.30
   752	93.60
   817	83.30
   821	83.79
   823	Method
   831	VehicleX [9]
   832	Vehicle-Diff
   833	VehicleX [9]
   834	Vehicle-Diff
   844	VehicleX [9]
   846	VehicleGAN [10]
   847	Vehicle-Diff
   850	VehicleX [9]
   851	Vehicle-Diff
   852	VehicleX [9]
   853	Vehicle-Diff
   927	83.50
   928	83.87
   930	83.60
   933	83.75
   943	93.82
   966	53.87
   968	83.10
  1008	63.46
  1011	83.60
  1013	83.00
  1020	93.20
  1025	93.40
  1026	93.92
  1031	43.89
  1036	83.88
  1037	83.80
  1042	83.03
  1043	83.47
  1046	23.46
  1051	63.54
  1111	TABLE III: Comparisons with the state-of-the-art methods
  1112	on VehicleID [54].
  1113	Method
  1117	VehicleX [9]
  1118	Vehicle-Diff
  1131	33.09
  1141	83.61
  1143	TABLE IV: Comparisons with competitive VehicleX [9] on
  1146	TABLE II: Comparisons with the state-of-the-art methods on
  1148	respectively. “B” indicates that each training batch selects
  1150	in § III-D), whereas “D” indicates that synthetic and real
  1155	Vehicle-Diff enables to achieve competitive vehicle re-ID accuracy on VeRi-776. This indicates that our proposed coarseto-fine generation pipeline adapts well to vehicle re-ID,
  1157	through our generative diffusion model is fine-tuned only
  1159	reID model is trained solely on synthetic data, our approach
  1160	improves mAP by 0.92% compared with VehicleX on VeRi776. When the reID backbone is switched to SwinV2-Base,
  1164	performance. In particular, our approach achieves 0.94%
  1165	and 3.57% improvements in mAP compared with VehicleX
  1166	and PAMTRI, respectively, when jointly trained with the
  1168	SwinV2-Base reID backbone, our method shows a consistent
  1170	improvement. In VeRi-776 dataset, Vehicle-Diff ourperforms
  1171	VehicleX by 0.62% on mAP when using random combination strategy (“D” in Tab. II) and 2.4% on mAP when using
  1173	Notably, Vehicle-Diff achieves a 0.6% increase in Rank1 accuracy over VehicleX, whose Rank-1 is already high
  1174	at 97.08%. This increase is non-trivial. Besides, compared
  1175	with other state-of-the-art methods, Vehicle-Diff also shows
  1176	competitive performances. Our Vehicle-Diff method achieves
  1177	97.68% Rank-1 and 83.79% mAP, which surpasses CLIPReID [64] of 97.40% Rank-1 and 83.30% mAP. Similarly, for
  1178	the VehicleID dataset, Vehicle-Diff shows competitive performance in Tab. III. In CityFlowV2, Vehicle-Diff outperforms
  1179	VehicleX by 4.17% on Rank-1 and 4.26% on Rank-5 (see
  1180	the left section of Tab. IV). We further conduct experiments
  1181	to evaluate the generalization capability of Vehicle-Diff. As
  1182	shown in the right section of Tab. IV, we apply the reID
  1185	from CityFlowV2 were used for fine-tuning the generative
  1186	model, without any label information. Despite this, VehicleDiff consistently outperforms VehicleX.
  1188	through both quantitative and qualitative evaluation. For the
  1189	quantitative assessment, we utilize the Frechet Inception
  1194	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.
  1232	VehicleX
  1237	Vehicle-Diff
  1264	Fig. 4: Qualitative retrieval results. Here we compare our
  1265	method with both our baseline and VehicleX. The ranking
  1268	Method
  1269	VehicleGAN [10]
  1271	VehicleX
  1272	Vehicle-Diff
  1277	233.0
  1312	33.19
  1324	TABLE VI: Ablation study on components, i.e., diffusion
  1325	fine-tuning (DFT) and semantic filtering (SF).
  1328	TABLE V: Quantitative comparisons on generated data quality. For a fair comparison, both Vehicle-Diff and VehicleX
  1332	datasets to train VehicleX and generate sample images. As
  1333	shown in Tab. V, Vehicle-Diff achieves a lower FID score
  1334	compared to all other generative methods. For qualitative
  1335	comparison, we visualize the sample outputs of competitive
  1336	generative methods in Fig. 1. The images in the first row
  1338	remaining five rows are from different synthetic data pipeline
  1340	Vehicle-Diff produces images that are visually closer to the
  1341	real-world dataset while keeping the fine-grained texture.
  1342	C. Ablation Studies and Further Discussion
  1343	Effectiveness of the coarse-to-fine strategy. Here, we
  1344	evaluate the effectiveness of each component in our coarseto-fine generation pipeline. Although the filtering process has
  1346	Distance (FID) change after fine-tuning is negligible, the
  1351	Effectiveness of the balanced sampling strategy. Previous
  1352	methods, such as VehicleX and PAMTRI, typically conduct
  1358	improves model learning on both VehicleX and Vehicle-Diff
  1361	yields a +1.06% boost in mAP for VehicleX and +2.81%
  1362	boost in mAP for Vehicle-Diff.
  1386	TABLE VII: Ablation study on the number of synthetic
  1389	the qualitative image retrieval comparison on VeRi-776. Our
  1390	method has successfully recalled the target vehicle in the
  1392	on real data or VehicleX. It is because that our Vehicle-Diff
  1393	contains a large number of vehicle images with fine-grained
  1395	facilitating the discriminative feature learning (see Fig. 3).
  1396	Therefore, the model trained on our Vehicle-Diff is able to
  1397	handle challenging matches with fine-grained differences and
  1399	Limited real data? To evaluate the effectiveness of VehicleDiff under limited real data conditions, we systematically
  1406	synthetic data for vehicle re-identification (reID). We introduce Vehicle-Diff, a novel coarse-to-fine cross-modality
  1407	generation pipeline that creates a synthetic reID dataset using
  1409	tailored to specific downstream tasks. Extensive experiments
  1411	synthetic and real-world data, thereby enhancing reID performance. Specifically, our method achieves a competitive
  1412	83.79% mAP on VeRi-776. Furthermore, we analyze the
  1415	future, we plan to integrate 3D-aware framework [75] into
  1419	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.
  1425	[2] X. Liu, W. Liu, H. Ma, and H. Fu, “Large-scale vehicle reidentification in urban surveillance videos,” in 2016 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2016,
  1427	[3] P. Shyam, K.-J. Yoon, and K.-S. Kim, “Adversarially-trained hierarchical feature extractor for vehicle re-identification,” in 2021 IEEE
  1431	for unsupervised vehicle re-identification,” IEEE Transactions on
  1433	2023.
  1434	[5] A. Khurshudov, “The smart city conundrum: technology, privacy, and
  1445	vehicle re-identification using highly randomized synthetic data,” in
  1449	content consistent vehicle datasets with attribute descent,” in Computer
  1452	[10] B. Li, P. Liu, L. Fu, J. Li, J. Fang, Z. Xu, and H. Yu, “Vehiclegan:
  1453	Pair-flexible pose guided image synthesis for vehicle re-identification,”
  1454	arXiv:2311.16278, 2023.
  1455	[11] C.-S. Hu, S.-W. Tseng, X.-Y. Fan, and C.-K. Chiang, “Vehicle view
  1456	synthesis by generative adversarial network,” in ICASSP 2023-2023
  1460	S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” Communications of the ACM, vol. 63, no. 11, pp. 139–144,
  1463	Y. You, and J. Zhao, “Msinet: Twins contrastive search of multiscale interaction for object reid,” in Proceedings of the IEEE/CVF
  1465	19 243–19 253.
  1471	deep feature fusion for vehicle re-identification,” in ICASSP 20202020 IEEE International Conference on Acoustics, Speech and Signal
  1473	[16] Z. Zheng, T. Ruan, Y. Wei, Y. Yang, and T. Mei, “Vehiclenet:
  1474	Learning robust visual representation for vehicle re-identification,”
  1476	[17] Y. Xu, N. Jiang, L. Zhang, Z. Zhou, and W. Wu, “Multi-scale vehicle
  1481	[18] J. Yu, J. Kim, M. Kim, and H. Oh, “Camera-tracklet-aware contrastive
  1482	learning for unsupervised vehicle re-identification,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022,
  1484	[19] B. Jiao, L. Yang, L. Gao, P. Wang, S. Zhang, and Y. Zhang, “Vehicle
  1485	re-identification in aerial images and videos: Dataset and approach,”
  1490	transformer network for vehicle re-identification,” IEEE Transactions
  1503	and G. Ding, “Tagperson: A target-aware generation pipeline for
  1507	is more: Learning from synthetic data with fine-grained attributes for
  1508	person re-identification,” ACM Transactions on Multimedia Computing, Communications and Applications, vol. 19, no. 5s, pp. 1–20, 2023.
  1522	for high-resolution image synthesis,” arXiv:2307.01952, 2023.
  1526	synthetic data from generative models ready for image recognition?” in
  1528	2023.
  1530	arXiv:2304.08466, 2023.
  1532	for video-based geo-localization,” arXiv preprint arXiv:2411.13610,
  1535	image generation and editing with text-guided diffusion models,”
  1536	arXiv:2112.10741, 2021.
  1537	[35] Z. Zheng, L. Zheng, and Y. Yang, “A discriminatively learned cnn
  1545	“Circle loss: A unified perspective of pair similarity optimization,” in
  1554	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.
  1556	image generation,” in Proceedings of the IEEE/CVF Conference on
  1559	Low, “Prompt optimization with human feedback,” arXiv preprint
  1560	arXiv:2405.17346, 2024.
  1570	arXiv:2106.09685, 2021.
  1573	“Yolov10 to its genesis: A decadal and comprehensive review of the
  1574	you only look once series,” arXiv:2406.19407, 2024.
  1578	Conference on Machine Learning. PMLR, 2021, pp. 8748–8763.
  1588	regularization in vivo,” IJCAI, 2020.
  1589	[49] D. Kinga, J. B. Adam, et al., “A method for stochastic optimization,” in
  1592	[50] J. Krause, M. Stark, J. Deng, and L. Fei-Fei, “3d object representations for fine-grained categorization,” in Proceedings of the IEEE
  1595	[51] Y. Bai, Y. Lou, F. Gao, S. Wang, Y. Wu, and L.-Y. Duan, “Groupsensitive triplet embedding for vehicle reidentification,” IEEE Transactions on Multimedia, vol. 20, no. 9, pp. 2385–2399, 2018.
  1597	dataset for fine-grained categorization and verification,” in Proceedings
  1602	vehicles,” in Proceedings of the IEEE International Conference on
  1604	[54] H. Liu, Y. Tian, Y. Yang, L. Pang, and T. Huang, “Deep relative
  1605	distance learning: Tell the difference between similar vehicles,” in
  1608	[55] D. Zapletal and A. Herout, “Vehicle re-identification for automatic
  1612	[56] X. Liu, W. Liu, T. Mei, and H. Ma, “A deep learning-based approach
  1613	to progressive vehicle re-identification for urban surveillance,” in Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The
  1618	benchmark for multi-target multi-camera vehicle tracking and reidentification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 8797–8806.
  1623	Pattern Recognition (CVPR) Workshops, June 2021, pp. 4263–4273.
  1625	[59] A. Kanaci, X. Zhu, and S. Gong, “Vehicle re-identification in context,”
  1628	[60] R. Chu, Y. Sun, Y. Li, Z. Liu, C. Zhang, and Y. Wei, “Vehicle reidentification with viewpoint-aware metric learning,” in Proceedings
  1632	and R. Chellappa, “A dual-path model with adaptive attention for vehicle re-identification,” in Proceedings of the IEEE/CVF International
  1639	learning for fine-grained visual categorization and re-identification,” in
  1645	no. 1, 2023, pp. 1405–1413.
  1646	[65] H. Zhu, W. Ke, D. Li, J. Liu, L. Tian, and Y. Shan, “Dual crossattention learning for fine-grained visual categorization and object
  1649	[66] F. Shen, Y. Xie, J. Zhu, X. Zhu, and H. Zeng, “Git: Graph interactive
  1650	transformer for vehicle re-identification,” IEEE Transactions on Image
  1651	Processing, vol. 32, pp. 1039–1051, 2023.
  1652	[67] J. Li and X. Gong, “Prototypical contrastive learning-based clip finetuning for object re-identification,” arXiv preprint arXiv:2310.17218,
  1653	2023.
  1655	deep model for vehicle re-identification,” in 2018 IEEE International
  1658	convolutional nets,” arXiv preprint arXiv:1405.3531, 2014.
  1660	vehicle re-identification,” in Proceedings of the IEEE/CVF Conference
  1663	in the details: Self-supervised attention for vehicle re-identification,” in
  1665	UK, August 23–28, 2020, Proceedings, Part XIV 16. Springer, 2020,
  1668	for vehicle re-identification based on common field of view,” in
  1685	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.

 succeeded in 148ms:
   188	et al., 2025) eﬀectively mitigates cross-modal semantic noise through
   190	their strong performance in bi-modal retrieval, these methods heavily
   194	Sketch-RGB person ReID. Compared with text-image tasks, sketchimage retrieval focuses more on visual attribute discrepancies, such as
   195	the absence of color and texture details in sketch. Existing methods are
   196	divided into direct cross-modal feature alignment methods (Gui et al.,
   197	2020; Zhu et al., 2022) and auxiliary modality generation-based methods (Chen et al., 2022). The former emphasize domain-invariant feature learning, for instance, Pang et al. (2018a) introduced an adversarial learning model to reduce the domain gap between sketch and RGB,
   198	while the MCGI framework (Zhang et al., 2025b) proposed modality collaboration and granularity interaction mechanisms to fuse complementary features. StyleGAN-based approaches (Koley et al., 2024a) further
   199	leveraged sketch abstraction levels combined with diﬀerentiable ranking losses to improve cross-style alignment stability. The latter introduce
   202	sketch and RGB to mitigate the modality gap. Later, Liu et al. (2024a)
   203	proposed an auxiliary modality that is robust to background clutter and
   204	style variation, which developed a multi-modal collaborative learning
   205	scheme to align feature relationships and distributions. Despite their effectiveness in aligning visual modalities, most existing methods remain
   206	limited to low-level visual feature compensation and often fail to capture semantic-level discrepancies. Moreover, existing generative methods typically produce black-and-white sketch as auxiliary modalities,
   207	neglecting color distributions and texture details, which restricts their
   209	Tri-modality person ReID. This task aims to fully leverage the advantages of diﬀerent modalities. Zhai et al. (2022) ﬁrst proposed a symmetric disentangling scheme to promote adversarial alignment of descriptive features from sketch and text with RGB image features, which
   210	improved the model’s semantic understanding. Chen et al. (2023a) presented a uniﬁed person ReID framework for cross-modal and multimodal tasks, introducing sketch and text as descriptive queries to
   212	methods primarily focus on mapping global information extracted from
   214	alignment. In contrast, we propose a ColorSketchNet framework to alleviate the unfair modality attributes across the modalities.
   218	We are the ﬁrst to introduce an auxiliary color sketch modality into
   220	the inherent attribute diﬀerences between text, sketch, and RGB, enabling uniﬁed representation learning across heterogeneous modalities that existing paired modality or uniﬁed frameworks cannot
   222	• We design a Color Sketch Generator (CSG) that not only produces
   223	vivid and texture-rich color sketch, but also incorporates a dynamic
   225	limitation of prior ReID methods that often fail under complex lighting conditions.
   227	(ACRM) that leverages the auxiliary color sketch as a bridge to build
   230	• Extensive experiments on three challenging trimodal benchmark datasets (Tri-ICFG-PEDES, Tri-RSTPReid, and Tri-PKU-Sketch)
   231	demonstrate that our ColorSketchNet delivers robust and superior
   233	Furthermore, additional evaluation on the SketchyCOCO dataset
   238	Cross-modal ReID tasks can be divided into three categories: TextRGB person ReID (Xia et al., 2025), Sketch-RGB person ReID (Zhang et
   242	textual descriptions and visual images, as well as the diﬃculty of modeling ﬁne-grained attributes such as clothing colors and patterns. In recent years, contrastive learning has gradually become the mainstream
   243	framework for cross-modal alignment. Within this framework, existing
   244	studies can be broadly divided into global feature modeling and local
   245	feature modeling. The former methods (Cheng et al., 2024; Li et al.,
   251	compensate for missing information in speciﬁc modalities (such as images, text, sketch, etc.) to alleviate the modality attributes imbalance.
   253	Yu et al., 2024) have been proposed to generate auxiliary modalities to
   255	In addition, Zhang et al. (2024) proposed a MSALNet to learn information compensation and fusion between visible light and infrared features by generating auxiliary branches, which utilized multi-stage auxiliary learning strategy to suppress interference information and improve
   262	Fig. 2. The overall architecture of the ColorSketchNet framework, which includes color sketch generator (CSG), attribute compensation and reﬁned module (ACRM)
   263	and uniﬁed collaborative alignment learning (UCA). During the inference, we do not utilize the color sketch generator, and only use photo, sketch and text modalities
   267	proposed a PMT framework to ﬁlter modality-speciﬁc information in
   270	presented a GECNet to narrow the modality gap between infrared images and visible images, which colorized single-channel infrared images
   272	Inspired by the above works, an auxiliary color sketch modality is
   273	developed to eﬀectively overcome the negative impact of the inherent
   274	diﬀerences (e.g., color and spectrum) between text, sketch and photo
   278	resulting in incomplete auxiliary modality information and poor crossmodal adaptability. In addition, these methods suﬀer from the problem
   280	text cannot provide the texture details of a photo, while sketch lacks
   281	color information. Unlike the SOTAs that generate black and white auxiliary sketch, we introduce an auxiliary color sketch modality as a bridge
   282	to compensate for the missing attributes (i.e., color, texture and structure) in text and sketch modalities, which consists of dynamic lighting
   283	modiﬁer and color sketch auxiliary generator, laying the foundation for
   285	Dynamic Lighting Modiﬁer. The core of the dynamic light modiﬁer is to adaptively modify the lighting based on the average brightness
   286	of the RGB photo. Before generating an auxiliary color sketch modality, we propose a dynamic lighting modiﬁer to avoid loss of details due
   287	to dark or overexposed environments, which can adaptively adjust the
   290	Speciﬁcally, given the RGB photo sets 𝐗𝑝 = {x𝑝 |x𝑝 ∈ ℝ𝐶×𝐻×𝑊 },
   292	respectively. To compute the average brightness of a photo x𝑝 , we
   293	ﬁrst transform a color photo x𝑝 into a single channel grayscale photo
   297	3. Methodology
   298	In this section, we will introduce the details of the proposed ColorSketchNet. Firstly, we propose a color sketch generator (CSG) in Section 3.1, which consists of dynamic lighting modiﬁer and color sketch
   299	auxiliary generator, which generate color-rich and texture-detailed
   300	sketch and cope with complex lighting conditions. Secondly, to compensate for the missing attributes in sketch and text, and suppress the
   302	attribute space that addresses the task imbalance problem in Section 3.2.
   303	Finally, an uniﬁed collaborative alignment (UCA) scheme is developed
   304	to adjust the latent distributions of the four modalities in Section 3.3.
   305	The overview of ColorSketchNet is shown in Fig. 2.
   312	photo, respectively.
   316	3.1. Color sketch generator
   322	the key factors of color attributes for generating black and white sketch,
   342	where lower values indicate darker images prone to losing important details, while higher values report brighter images that suﬀer from overexposure. Therefore,based on the dataset distribution and threshold analysis in Section 4.7, we set a threshold 𝑇 = 128 to distinguish darker images from brighter ones.
   348	obtained, denoted as the bright photo x𝑙 ∈ ℝ𝐶×𝐻×𝑊 , which can improve the quality of auxiliary color sketch modality for the input of color
   349	sketch auxiliary generator.
   350	Color Sketch Auxiliary Generator. We propose a color sketch auxiliary generator to generate sketch that retains the sketch style and
   351	identity-related color information for attribute compensation.
   355	of x𝑙 , generating a grayscale sketch image x𝑑 .
   356	Further, we fuse the structural information of x𝑑 with the color information of x𝑙 in the YUV color space. To make this mapping explicit,
   372	3.2. Attribute compensation and reﬁned module
   373	To ensure attribute fairness between diﬀerent modalities, we propose a attribute compensation and reﬁned module (ACRM) to adaptively
   375	for missing color distribution and edge features from the auxiliary color
   376	sketch; in the Sketch-RGB task, the color and texture details of the auxiliary color sketch can be utilized to compensate for the blalck and white
   377	sketch; in the Text and Sketch-RGB task, the color sketch can make up
   380	by the auxiliary color sketch modality and enhance the coherence and
   383	Sketch-RGB task. The principles of these two tasks are the same. For clarity, we take the Sketch-RGB task as an example to explain the retrieval
   387	the sketch features f𝑠 , which is deﬁned by:
   397	However, sketch inherently lack color and texture attributes. To
   398	overcome this issue, we compute the dissimilarity between sketch and
   399	color sketch, denoted as 1 − 𝛼𝑎→𝑠 . Further, the missing information ḟ 𝑠
   403	Speciﬁcally, the contours of the grayscale sketch image are mapped
   408	of the sketch modality is formulated as:
   417	where V 𝑎 is the mapping of the color sketch feature f𝑎 ; F (⋅) consists of
   428	to enhance the black and white sketch features.
   429	It is worth noting that the structure of sketch features will be disrupted if the missing features are directly added to the black and white
   430	sketch features. To address this, we introduce a weight factor g and
   431	a normalization term b. Speciﬁcally, g serves as a channel-wise gating mechanism, adaptively regulating the contribution of compensation features so that channels correlated with the missing attributes
   432	(e.g., color or texture) are enhanced while irrelevant channels are suppressed. Meanwhile, b acts as a residual normalization term: the constant 1 guarantees preservation of original sketch features, while the additional adaptive bias ensures numerical stability and balanced feature
   433	magnitudes across channels. This design enables ACRM to selectively
   445	Finally, the auxiliary color sketch x𝑎 is generated by combining these
   454	We choose YUV over HSV because YUV explicitly decouples luminance Y from chrominance U, V, enabling a clean assignment of structural contours to the Y channel and color information to the U and V
   459	that can introduce instability near boundaries. Consequently, YUV provides a more robust and linear basis for integrating contour and color,
   462	Prativadibhayankaram et al., 2024).
   463	Subsequently, the global features of photo x𝑝 , sketch x𝑠 , auxiliary
   464	color sketch x𝑎 and text x𝑡 are fed into the four-stream CLIP network to
   465	obtain the corresponding features f𝑝 , f𝑠 , f𝑎 and f𝑡 , respectively.
   466	To make the generated auxiliary color sketch consistent with identity information of the original photo, an identity preservation learning
   483	where 𝜙(⋅) is the sigmoid activation function; F (⋅) denotes a series of
   485	After that, the features of each channel in sketch feature f𝑠 are normalized to 𝜎𝑠 and then combined with g and b. So, the compensated
   491	However, the generated auxiliary color sketch modality may introduce elements that are inconsistent with the photo modality, such as
   492	unnatural textures, color distortions or geometric deformations. These
   502	the i-th identity; 𝑓𝑎𝑖 indicates the output of auxiliary color sketch features
   510	To be speciﬁc, we compute the similarity 𝛼𝑝→𝑠 between the photo features f𝑝 and the compensated sketch features f̃𝑠 by referring to Eq. (9).
   513	is subtracted from the compensated sketch features f̃𝑠 . The ﬁnal compensated sketch features f̂𝑠 can be obtained by:
   518	This optimization process can adaptively adjust the importance of
   519	the sketch features and enhance the robustness of the model, which suppresses noise from the auxiliary color sketch modality.
   520	Similar to above-mentioned analysis, for Text-RGB task, the ﬁnal
   526	Tri-modality task. This task includes text, sketch and RGB modalities, where text and sketch are utilized together as query modalities
   527	to retrieve RGB images. The proposed ACRM can be extended to trimodality retrieval scenario.
   528	Sketch and text have the advantage of complementing each other.
   529	Sketch can directly represent geometric information, such as contours
   531	the high-level semantic information for sketch. Therefore, compared to
   533	richer and more comprehensive pedestrian description. Nevertheless,
   534	there are redundant features between text and sketch, and they lack important visual attributes (such as color distribution) which cannot be
   535	described by text and sketch. Fortunately, the proposed ACRM can still
   537	color sketch modality, alleviating the negative impact of redundant features in both the text and sketch modalities.
   538	Speciﬁcally, text feature and sketch feature are fused by using a simple summation operation, denoted as the fused features f𝑓 . Similarly
   539	to obtaining f̂𝑠 and f̂𝑡 , the ﬁnal compensated features f̂𝑓 that simultaneously compensate for text and sketch features can be obtained by
   560	where f𝑝 𝑖 and f̂𝑚𝑖 denote the photo features and the compensated features from modality 𝑚 of the 𝑖-th identity, respectively. 𝐶𝑚 is the classiﬁer corresponding to the m-th modality; y𝑖 is the i-th identity label; 𝑁
   563	structural alignment across modalities, which takes the auxiliary color
   564	sketch as a semantic anchor. The auxiliary color sketch provides rich
   565	structural and color-invariant information, making it a suitable bridge
   597	where f̂𝑚𝑖 denotes the i-th identity features of f̂𝑚 ; f𝑎 denotes the auxiliary color sketch feature for the j-th identity; || ⋅ ||2 represents the Euclidean 2-norm. This loss function simultaneously minimizes the distance between the sketch anchor and both the photo and other modalities’ features, thereby encouraging tighter inter-modality alignment for
   603	cluster around their class center. Unlike unsupervised clustering methods (Yang et al., 2024) that infer cluster assignments without labels, our
   619	To summarize, the overall objective function loss of our method can
   626	In general, the proposed ACRM is able to be utilized to compensate
   627	for missing features in both paired modality retrieval task (i.e., TextRGB, Sketch-RGB) and Tri-modality retrieval task (i.e., Text and SketchRGB), verifying its compatibility and generalization 𝑚𝑚 .
   633	During inference, the color sketch generator (CSG), the attribute
   634	compensation and reﬁnement module (ACRM), and all auxiliary supervision branches are removed. These components are used exclusively
   635	during training to improve representation learning and do not participate in the forward pass at test time. The ﬁnal feature extractor is selfcontained and operates directly on raw sketch or text queries to produce
   639	3.3. Uniﬁed collaborative alignment learning scheme
   640	Our framework designs a uniﬁed collaborative alignment learning
   641	scheme that systematically coordinates cross-modal discrimination, semantic structure alignment, and intra-modal compactness. This collaborative design builds a modality-invariant feature space while preserving
   642	discriminative identity features. The scheme integrates three complementary loss functions: identity matching loss 𝑖𝑑𝑚 , cross-modal structure regularization loss 𝑐𝑚 , and intra-modal class constraint loss 𝑖𝑐𝑐 .
   646	4. Experiments
   647	4.1. Experiments settings
   648	Datasets. The experiments are performed on three multi-modal
   650	2021a), and Tri-PKU-Sketch (Pang et al., 2018b). For fair comparisons,
   651	we follow the previous works to divide the training and testing datasets.
   667	#Sketch
   674	SketchyCOO
   696	𝛼𝑎→𝑠 s exploited to extract attributes (e.g., color or texture) that
   697	are missing in the primary modality, and these complementary features are selectively integrated via channel-wise gating and residual bias. In the RF stage, the dissimilarity score 1 − 𝛼𝑝→𝑠 is further
   700	As shown in Table 3, ablation studies show that using AC alone
   702	the complete ACRM (AC+RF) achieves the best performance. Specifically, on the Tri-RSTPReid and Tri-PKU-Sketch datasets, Rank-1 accuracies reach 58.72 % and 73.00 % for the Text-RGB task, 65.50 %
   703	and 88.50 % for the Sketch-RGB task, and 72.54 % and 90.60 % for
   704	the Text+Sketch-RGB task, respectively. These results verify that
   705	the dissimilarity-driven dual design of ACRM is both eﬀective and
   707	Eﬀect of Uniﬁed Collaborative Alignment Learning Scheme.
   708	To further verify the eﬀectiveness of our proposed Uniﬁed Collaborative Alignment (UCA) scheme, we conduct comprehensive ablation
   709	studies by progressively incorporating each loss component into the
   710	overall objective and evaluating performance across the three sub-tasks.
   711	As shown in Table 3, introducing the cross-modal structure regularization loss 𝑐𝑚 in the Sketch-RGB task, the Rank-1 accuracy increases
   712	by 2.23 %, 1.30 % on the Tri-RSTPReid and Tri-PKU-Sketch datasets.
   714	shows improvements of 60.95 % and 77.80 %, respectively, demonstrating that the auxiliary color sketch serves as an eﬀective semantic anchor to facilitate feature alignment across modalities. Finally, after incorporating the intra-class compactness loss 𝑖𝑐𝑐 , the model achieves a
   715	Rank-1 accuracy of 78.00 %, 93.50 % under the “Text + Sketch" query
   716	setting. This indicates that 𝑖𝑐𝑐 eﬀectively enhances the robustness
   720	added the sketchyCOCO (Gao et al., 2020) dataset to verify the adaptability of the model.
   723	Sketch to RGB scenarios. We utilize Cumulative Matching Characteristics (CMC), mean Average Precision (mAP), and mean Inverse Negative
   725	comparisons, all the evaluation procedures are consistent with the comparative state of the arts. The performance is obtained from the average
   726	experimental results of 10 randomly divided training sets and testing
   728	Implementation Details. We implement all the experiments in the
   729	PyTorch framework with an NVIDIA A100 GPU. To ensure the repeatability and validate the universality of our proposed methods, we utilize
   737	𝜆3 = 0.04 are decided by ablations on settings. During evaluation, the
   738	Color Sketch Generator (CSG) and the Attribute Compensation and Reﬁnement Module (ACRM) are not involved. The model extracts features
   739	from the queries (text or sketch) and gallery images (RGB) solely using
   743	4.3. Comparison with state-of-the-art methods
   744	In this section, our method is compared with existing text-based
   745	state-of-the-art approaches, as presented in Tables 4 and 5. For fair
   747	datasets. In Table 4, the proposed approach outperforms the multi-task
   749	3.10 % in Rank-1 and Rank-5 accuracy, respectively. Likewise, Table 4
   752	Nevertheless, in the single Text-RGB retrieval task, our method still
   756	to the inherent optimization trade-oﬀs introduced by multi-task learning. While ColorSketchNet is required to learn generalized and robust
   757	shared representations across diverse modalities(i.e., text, sketch, and
   759	matching, such as diverse style modeling, uncertainty-aware feature
   760	learning, or graph-based relational reasoning, to achieve superior alignment. These models often demonstrate stronger discriminative power
   762	Therefore, we plan to explore more adaptive and task-aware feature
   764	In addition, the results on the Tri-PKU-Sketch dataset are also evaluated in Table 6. In this dataset, the image samples are extremely clear
   765	and include professional hand-drawn sketch of pedestrians. (Zhai et al.,
   767	4.2. Ablation study
   768	We perform ablation studies on the baselines 𝑏𝑎𝑠𝑒 to analyze the effectiveness of each component on the Tri-RSTPReid and Tri-PKU-Sketch
   769	datasets. Meanwhile, reasonable settings of each proposed component
   771	Eﬀect of Color Sketch Auxiliary Generator. auxiliary color sketch
   772	modality plays an important role in bridging the visual gap between sketch, text, and photo modality. Existing methods utilize generators (Xiang et al., 2022) to generate black-and-white sketch as
   774	lacks color information. As shown in Table 3, compared with the
   775	black and white sketch auxiliary modality (𝐴𝑢𝑥𝑏𝑙𝑎𝑐𝑘 ), the auxiliary
   776	color sketch modality shows signiﬁcant improvement in Text-based
   777	queries, with Rank-1 accuracy gains of 1.54 % and 3.80 % on the TriRSTPReid and Tri-PKU-Sketch datasets, respectively. Under sketchbased queries, the auxiliary color sketch modality achieves Rank-1 accuracies of 70.24 % and 91.00 % on Tri-RSTPReid and Tri-PKU-Sketch,
   778	respectively. When using both text and sketch as the query conditions, the auxiliary color sketch modality outperforms its black-white
   779	counterpart by 0.95 % and 3.30 % in Rank-1 accuracy on two datasets,
   780	respectively.
   789	Ablation study about each component on three multi-modal datasets. Rank (R) at k accuracy (%), mAP (%), and mINP (%) are reported.
   864	Sketch-RGB
   867	Text+Sketch-RGB
   881	Tri-PKU-Sketch
   932	73.00
   949	73.40
   968	53.60
   971	53.34
   978	33.78
   993	93.64
  1001	93.64
  1016	53.52
  1021	63.85
  1025	33.45
  1037	93.50
  1043	93.50
  1051	93.50
  1096	Comparison with the state-of-the-arts methods on Tri-RSTPReid dataset.
  1099	Methods
  1110	IVT (Shu et al., 2022)
  1132	63.40
  1133	63.67
  1141	83.30
  1145	63.19
  1149	83.45
  1170	method introduces only a relatively small increase in parameters and
  1172	the attention mechanisms and MLP structures employed in the attribution compensation and reﬁnement modules. Given that the baseline
  1174	incremental overhead is negligible. Meanwhile, our method brings signiﬁcant improvements in cross-modal alignment, making this trade-oﬀ
  1176	training time on the PKU dataset. It is also worth noting that the Dynamic Lighting Modiﬁer and the Color Sketch Auxiliary Generator function as data preprocessing components before training and inference,
  1182	Comparison with the state-of-the-arts methods on Tri-ICFG-PEDES
  1185	Methods
  1198	IVT (Shu et al., 2022)
  1210	Arxiv23
  1231	73.60
  1236	83.06
  1267	ColourSketchNet, we transferred the model from person re-ID to scenelevel freehand sketch retrieval and conducted multi-task retrieval experiments on the SketchyCOCO (Gao et al., 2020) dataset. SketchyCOCO
  1268	(Gao et al., 2020) contains 14,081 sketch-text-image triplets, where the
  1269	images and text captions are from MS-COCO and the sketch are scenelevel freehand line drawings. Because ColourSketchNet’s color sketch
  1271	this. As shown in Table 8, ColourSketchNet achieves consistent performance advantages on all three retrieval tasks, demonstrating that the
  1272	method can eﬀectively generalize to scene-level freehand sketch and
  1277	Compared with existing sketch-based person ReID methods, our method
  1278	can surpass most learning frameworks on the Sketch-RGB task.
  1280	Visualization of color sketch eﬀect. We visualize the learned feature representations using T-SNE (Maaten & Hinton, 2008). As shown in
  1281	Fig. 3, it can be observed that the auxiliary color sketch can eﬀectively
  1282	bridge the sketch and photo modalities, reducing the gap caused by
  1286	To evaluate the eﬃciency of our method, we compare its training

codex
章节边界已经出来了：这 6 篇都在前 8 页包含完整方法或主要方法，少数因为双栏抽取导致“实验”标题和方法小节顺序交错。我接下来直接抽方法正文范围，重点看作者怎么从观察推出机制。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf' - | nl -ba | sed -n '118,257p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   118	II. M ETHODOLOGY
   119	A. Auxiliary information generation
   120	The body segmentation map and pose keypoints of pedestrians are required to assist the generating phase, as illustrated in
   121	the auxiliary information generation module of Fig.1. We use
   122	self-correction human parsing(SCHP) model [14] to generate
   123	segmentation masks that represent body parts and clothing
   124	items. These segmentation maps serve as masks for the
   125	inpainting process in II-B. We utilize the pre-trained SCHP
   126	model on the ATR dataset to extract human parsing representations. The model includes 18 labels for body segmentation
   127	masks, such as ’Upper-clothes’, ’Left-leg’, ’Right-leg’, ’Leftarm’, ’Right-arm’, ’Bag’ and so on. Additionally, we utilize
   128	OpenPose model [15] to extract 18 pose keypoints of each
   129	pedestrian in the images. The keypoints is used in II-B to
   130	control the generated images with a natural pose.
   131	
   132	B. Generating phase
   133	The generating phase of the generate-filter clothing-change
   134	module(GFCC) adopts LaDI-VTON method [10], as illustrated in Fig.1. The expanded Stable Diffusion inpainting
   135	pipeline of LaDI-VTON is used to change clothing for pedestrian images in CC-ReID.
   136	To be specific, the original spatial input γ of is formed by
   137	1×h×w
   138	concatenating a binary inpainting mask m ∈ {0, 1}
   139	, the
   140	masked image E( ˆ
   141	I) which is encoded into a latent representation, and the denoising network input zt along the channel
   142	dimension. Additionally, the spatial input γ ∈ R9×h×w is
   143	expanded by appending two other components:the resized pose
   144	map p ∈ R18×h×w and the encoded clothing E(Ĉ) ∈ R4×h×w .
   145	The final spatial input of the inpainting denoising network is:
   146	ˆ p; E(Ĉ)] ∈ R(9+18+4)×h×w
   147	γ = [zt ; m; E(I);
   148	
   149	(1)
   150	
   151	Iˆ represents the pedestrian image I masked by a mask
   152	1×H×W
   153	M ∈ {0, 1}
   154	. The mask m is resized from M . The
   155	original inpainting mask M is derived from the segmentation
   156	map of pedestrian. The inpainting area is specified by the mask
   157	M . To ensure sufficient coverage of the intended clothing
   158	regions, the method proposed in prior work [16] is applied.
   159	Additional inputs are utilized to condition the inpainting
   160	pipeline, including the textual prompt describing the clothing
   161	[10], the pose map P , and the clothing Ĉ which is warped
   162	according to the pedestrian body shape. The warped clothing
   163	Ĉ is generated through a geometric matching module [17]
   164	and a U-Net refinement model [18]. The geometric matching
   165	module determines the correlation between the clothing C and
   166	a clothing-irrelevant pedestrian representation, which includes
   167	Iˆ and P , generating parameters θ. A thin-plate spline transformation [19] generates the coarse warped clothing C ′ from the
   168	clothing C by C ′ = T P Sθ (C). The U-Net refines the warped
   169	ˆ
   170	clothing by Ĉ = U net(C ′ , P, I).
   171	
   172	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:53:59 UTC from IEEE Xplore. Restrictions apply.
   173	
   174	C. Filtering phase
   175	To address the issue of low-quality generated cloth-changing
   176	pedestrian images, which may impact ReID performance, we
   177	use two threshold filters, shown in Fig.1.
   178	1) Similarity Threshold Filter: This filter compares the
   179	ReID features of generated images with those of original
   180	images, filtering based on similarity scores [20]. Generated
   181	images with higher scores are retained, while those with lower
   182	scores are discarded.
   183	2) FID Threshold Filter: We divide the generated images
   184	into small batches and compute the FID [21] value for each
   185	batch compared to the real image set. If a batch’s FID value
   186	exceeds the set threshold, it is discarded. To enhance dataset
   187	quality, we refine the process by regrouping remaining images
   188	into new batches and recalculating FID values.
   189	Additionally, low-quality images are removed through manual inspection to ensure the final dataset meets high standards.
   190	D. Centroid loss for CC-ReID
   191	Our CCAL model utilizes the CAL model [22] as the
   192	baseline, and introduces the centroid loss in prior work [13]
   193	for training. Specifically, the L2 loss is computed using the
   194	centroids. For each pedestrian identity i in the batch, the
   195	samples are divided into two groups: PA , which includes
   196	all samples of ID i, and PB , which comprises the samples
   197	which are not belonging to ID i. The centroid embedding of
   198	both groups is computed. The loss function is defined as the
   199	euclidean distance between the centroids of the two groups.
   200	This loss for each ID in the batch is summed. Within a batch,
   201	each ID i has N samples. f (n) and Xn respectively represent
   202	the ID of sample n and its feature embedding. PA for ID i
   203	consists of N samples, and PB of this ID contains M samples.
   204	The centroids for both clusters of ID i are derived as follows:
   205	X
   206	1
   207	·
   208	(Xn )
   209	(2)
   210	PA (i) =
   211	N
   212	n,f (n)=i
   213	
   214	PB (i) =
   215	
   216	1
   217	·
   218	M
   219	
   220	X
   221	
   222	(Xn )
   223	
   224	(3)
   225	
   226	n,f (n)̸=i
   227	
   228	The centroid loss for id i is defined as:
   229	LP (i) = ||PA (i) − PB (i)||2
   230	For a batch, the centroid loss LP is calculated as
   231	The final loss function is as follows:
   232	L = αLcal + βLP
   233	
   234	(4)
   235	P
   236	
   237	i LP (i).
   238	
   239	Fig. 2. The examples show the effects of changing the clothing for pedestrians
   240	with different postures and different original clothing types.
   241	
   242	cameras. We report the recognition accuracy for PRCC in both
   243	same clothes setting and clothing-change setting. For LTCC,
   244	we provide the accuracy for general ReID and CC-ReID.
   245	B. Implementation details
   246	In the generating phase, we randomly select several images
   247	of each pedestrian and change their clothing to a set of long
   248	sleeves and pants, which cover most body areas (e.g., arms
   249	and legs) and allow the model to focus on pedestrian features
   250	that are typically visible despite clothing, such as the head and
   251	neck, while retaining human contours. We use stable diffusion
   252	inpainting pre-trained model [25] with 20 inference steps and
   253	3 batch size. The EMASC module [10] is used to enable
   254	better detail reconstruction in inpainting task. Example clothchanging images are shown in Fig. 2. After filtering, we add
   255	4556 generated pedestrian images to the PRCC dataset and
   256	1310 images to the LTCC dataset.
   257	We train our model on our consistent clothing datasets. Following the CAL [22], ResNet50 [26] pre-trained on ImageNet

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf' - | nl -ba | sed -n '243,730p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   243	4. Methodology
   244	In this section, we present the Modality Centering with Spiral Training Network (MCST). MCST alternates between two types of text descriptions during the learning process of text tokens, establishing a connection between visible-text and infrared-text to impose a constraint.
   245	During the training of the image encoder, all image-text relationships
   246	are constrained to promote the centralization of the multi-modal space.
   247	Additionally, a spiral training approach is proposed to maintain the connection between the text embedding and the image embedding, facilitating better alignment.
   248	4.1. Text token training stage
   249	Fig. 2 illustrates the framework of the text token training stage. We
   250	ﬁrst introduce the separable text description served as the modality
   251	decoupler. Let 𝑉𝑥 = {𝑥𝑣 } and 𝑈𝑥 = {𝑥𝑢 } represent the visible and infrared image set in a batch, respectively, where each set has 𝑁 samples.
   252	Since the images themselves inherently contain modality-speciﬁc information, the descriptions generated by CLIP for these images also carry
   253	modality-speciﬁc details. We design the full text description as “A photo
   254	of a [𝑋]1 [𝑋]2 [𝑋]3 … [𝑋]𝑀 person from [𝑍]1 [𝑍]2 [𝑍]3 … [𝑍]𝐿 modality”
   255	, where each [𝑋]𝑚 (𝑚 ∈ {1, 2, … 𝑀}) represents an ID-speciﬁc learnable
   256	text token to describe person, and each [𝑍]𝑙 (𝑙 ∈ {1, 2, … 𝐿}) represents
   257	an ID-speciﬁc learnable text token to describe modality. Each ID has a
   258	visible text description 𝑡𝑣 and an infrared text description 𝑡𝑢 , then the
   259	image-to-text contrastive loss is calculated as:
   260	𝑖2𝑡 = −
   261	
   262	𝑁
   263	𝑁
   264	exp(sim(𝑓𝑥𝑎𝑣 , 𝑓𝑡𝑎𝑣 )∕𝜏)
   265	exp(sim(𝑓𝑥𝑢𝑎 , 𝑓𝑡𝑢𝑎 )∕𝜏)
   266	1 ∑
   267	1 ∑
   268	log ∑𝑁
   269	−
   270	log ∑𝑁
   271	,
   272	𝑁 𝑎=1
   273	exp(sim(𝑓 𝑣 , 𝑓 𝑣 )∕𝜏) 𝑁 𝑎=1
   274	exp(sim(𝑓 𝑢 , 𝑓 𝑢 )∕𝜏)
   275	𝑘=1
   276	
   277	𝑥𝑎
   278	
   279	𝑡
   280	𝑘
   281	
   282	𝑘=1
   283	
   284	𝑥𝑎
   285	
   286	𝑡
   287	𝑘
   288	
   289	(2)
   290	
   291	and the text-to-image contrastive loss is calculated as:
   292	
   293	Fig. 2. Overview of our proposed CLIP-powered Modality Centering with Spiral
   294	Training Network (MCST). (a) Framework of the text token training stage: The
   295	separable text description decouples identity-aware and modality-aware information, while identity-aware text prompts are centered to preserve modalityinvariant information. (b) Framework of the image encoder training stage:
   296	Identity-aware text prompts guide the encoder to extract identity-aware image
   297	features, and image and text features from both visible and infrared modalities
   298	are centered. Both stages can be trained alternately.
   299	
   300	𝑡2𝑖 = −
   301	
   302	𝑁
   303	exp(sim(𝑓𝑥𝑣 , 𝑓𝑡𝑣𝑎 )∕𝜏)
   304	∑
   305	𝑗
   306	1 ∑
   307	1
   308	log ∑𝑁
   309	|
   310	|
   311	𝑁 𝑎=1 |𝑄(𝑦 𝑣 )| 𝑣
   312	exp(sim(𝑓
   313	)∕𝜏)
   314	𝑥𝑣 , 𝑓 𝑡 𝑣
   315	𝑘=1
   316	𝑡𝑎 | 𝑥𝑗 ∈𝑄(𝑦𝑡𝑣 )
   317	𝑎
   318	𝑘
   319	|
   320	𝑎
   321	
   322	−
   323	
   324	𝑁
   325	exp(sim(𝑓𝑥𝑢 , 𝑓𝑡𝑢𝑎 )∕𝜏)
   326	∑
   327	𝑗
   328	1 ∑
   329	1
   330	log ∑𝑁
   331	,
   332	𝑁 𝑎=1 ||𝑄(𝑦 𝑢 )|| 𝑢
   333	exp(sim(𝑓𝑥𝑢 , 𝑓𝑡𝑢𝑎 )∕𝜏)
   334	𝑥
   335	∈𝑄(𝑦
   336	)
   337	𝑢
   338	𝑘=1
   339	𝑡
   340	𝑡𝑎
   341	𝑎 | 𝑗
   342	𝑘
   343	|
   344	
   345	(3)
   346	
   347	where 𝑄(𝑦𝑡𝑎 ) is the positive image set related to text 𝑡𝑎 .
   348	Eqs. (2) and (3) connect the image and text representations for the visible and infrared modalities, respectively. However, what is required is
   349	person-speciﬁc information that is independent of modality. To achieve
   350	this, we use a partial text description, “A photo of a [𝑋]1 [𝑋]2 [𝑋]3 … [𝑋]𝑀
   351	person” to bridge the visible and infrared text representations. Let 𝑝
   352	denote the partial text description for person, we propose the visibleinfrared text centering loss 𝑣𝑡2𝑖𝑡 as follows:
   353	1 ∑‖
   354	1
   355	‖
   356	‖𝑓 𝑣 − 𝑓𝑝𝑢 ‖ +
   357	𝑘 ‖2
   358	𝑁 𝑘=1 ‖ 𝑝𝑘
   359	2𝑁(𝑁 − 1)
   360	𝑁
   361	
   362	𝑣𝑡2𝑖𝑡 =
   363	
   364	where 𝑦𝑡 is the label of text 𝑡, sim is cosine similarity function, and 𝜏
   365	is the temperature parameter. A symmetric cross entropy loss is then
   366	optimized over the 𝑁 × 𝑁 similarity scores for a contrastive objective.
   367	
   368	𝑁
   369	∑
   370	𝑗,𝑘=1
   371	𝑗≠𝑘,𝑚∈{𝑢,𝑣}
   372	
   373	|
   374	‖
   375	‖ |
   376	|𝛼 − ‖𝑓 𝑣 − 𝑓 𝑚 ‖ | ,
   377	𝑝 ‖ |
   378	|
   379	‖ 𝑝𝑗
   380	𝑘 ‖2 |+
   381	|
   382	‖
   383	
   384	(4)
   385	
   386	and the infrared-visible text centering loss 𝑖𝑡2𝑣𝑡 is:
   387	1 ∑‖
   388	1
   389	‖
   390	‖𝑓 𝑢 − 𝑓𝑝𝑣 ‖ +
   391	𝑘 ‖2
   392	𝑁 𝑘=1 ‖ 𝑝𝑘
   393	2𝑁(𝑁 − 1)
   394	𝑁
   395	
   396	𝑖𝑡2𝑣𝑡 =
   397	
   398	3.2. CLIP-ReID
   399	
   400	𝑁
   401	∑
   402	
   403	|
   404	‖
   405	‖ |
   406	|𝛼 − ‖𝑓 𝑢 − 𝑓 𝑚 ‖ | ,
   407	𝑝 ‖ |
   408	|
   409	‖ 𝑝𝑗
   410	𝑘 ‖2 |+
   411	‖
   412	
   413	|
   414	𝑗,𝑘=1
   415	𝑗≠𝑘,𝑚∈{𝑢,𝑣}
   416	
   417	(5)
   418	
   419	where 𝛼 is the margin. The text-to-text centering loss is then given by:
   420	
   421	CLIP-ReID integrates CLIP for person ReID without requiring text annotations. In the ﬁrst stage, CLIP-ReID freezes the parameters of both the
   422	image and text encoders, pre-training a set of learnable text tokens using
   423	CoOp. The model constructs text descriptions in the ﬁxed form “A photo
   424	of a [𝑋]1 [𝑋]2 [𝑋]3 … [𝑋]𝑀 person”, where each [𝑋]𝑚 (𝑚 ∈ {1, 2, … 𝑀})
   425	represents an ID-speciﬁc learnable text token. In the second stage, only
   426	the parameters of the image encoder are updated. The text prompts
   427	
   428	𝑡2𝑡𝑐 = 𝑣𝑡2𝑖𝑡 + 𝑖𝑡2𝑣𝑡 .
   429	
   430	(6)
   431	
   432	Consequently, the parameters of image encoder (⋅) and text encoder
   433	 (⋅) are frozen in the text token training stage, and the overall objective
   434	
   435	function is formulated as follows:
   436	𝑡𝑒𝑥𝑡 = 𝑖2𝑡 + 𝑡2𝑖 + 𝜆1 𝑡2𝑡𝑐 .
   437	4
   438	
   439	(7)
   440	
   441	Pattern Recognition 177 (2026) 113333
   442	
   443	J. Xiong et al.
   444	
   445	4.2. Image encoder training stage
   446	Fig. 2 illustrates the framework of the image encoder training stage.
   447	The text prompts obtained through optimization can be used in this stage
   448	to guide the training of the image encoder. To fully leverage the semantic information in the text prompts, we use partial text description 𝑝 to
   449	leave out modality-speciﬁc information and propose the visible imagetext centering loss 𝑣𝑖2𝑡 as follows:
   450	|
   451	𝑁 |
   452	‖
   453	‖ |
   454	1 ∑ ||
   455	‖
   456	‖
   457	‖
   458	‖
   459	‖𝑓 𝑣 − 𝑓 𝑚 ‖ || ,
   460	min
   461	|𝛽 + ‖𝑓𝑥𝑣 − 𝑓𝑝𝑣 ‖ + ‖𝑓𝑥𝑣 − 𝑓𝑝𝑢 ‖ −
   462	𝑥
   463	𝑝 ‖ |
   464	‖
   465	|
   466	‖
   467	‖
   468	‖
   469	‖
   470	𝑗 ‖2
   471	𝑚∈{𝑢,𝑣}
   472	𝑘
   473	𝑘 2
   474	𝑘
   475	𝑘 2
   476	𝑁 𝑘=1 |
   477	‖ 𝑘
   478	|
   479	𝑗∈{1,2…𝑁},𝑗≠𝑘
   480	|
   481	|+
   482	
   483	𝑣𝑖2𝑡 =
   484	
   485	(8)
   486	
   487	and the infrared image-text centering loss 𝑖𝑖2𝑡 as follows:
   488	𝑖𝑖2𝑡 =
   489	
   490	|
   491	𝑁 |
   492	‖
   493	‖ |
   494	1 ∑ ||
   495	‖
   496	‖
   497	‖
   498	‖
   499	‖𝑓 𝑢 − 𝑓 𝑚 ‖ || ,
   500	min
   501	|𝛽 + ‖𝑓𝑥𝑢 − 𝑓𝑝𝑢 ‖ + ‖𝑓𝑥𝑢 − 𝑓𝑝𝑣 ‖ −
   502	𝑝 ‖ |
   503	‖ 𝑥𝑘
   504	‖ 𝑘
   505	‖ 𝑘
   506	𝑗 ‖2
   507	𝑚∈{𝑢,𝑣}
   508	𝑘 ‖2
   509	𝑘 ‖2
   510	𝑁 𝑘=1 ||
   511	‖
   512	|
   513	𝑗∈{1,2…𝑁},𝑗≠𝑘
   514	|
   515	|+
   516	
   517	Fig. 3. Our proposed method establishes pairwise connections between the four
   518	feature spaces and brings them together into a shared space.
   519	
   520	(9)
   521	
   522	Table 1
   523	Comparisons with three existing V-I ReID datasets.
   524	
   525	where 𝛽 is the margin. Eqs. (8) and (9) aim to reduce the distance between the image and all text features of the same ID, while increasing
   526	the distance between text features of diﬀerent IDs. The image-to-text
   527	centering loss is then given by:
   528	𝑖2𝑡𝑐 = 𝑣𝑖2𝑡 + 𝑖𝑖2𝑡 .
   529	
   530	(10)
   531	
   532	To reduce the feature distance between visible and infrared images,
   533	we use the ID loss and Weighted Regularization Triplet (WRT) loss [30],
   534	which have been proven eﬀective in ReID. The formula of ID loss is as
   535	follows:
   536	𝑖𝑑 = −
   537	
   538	2𝑁
   539	(
   540	)
   541	1 ∑
   542	𝑞 log (𝑓𝑥𝑘 ) ,
   543	2𝑁 𝑘=1 𝑘
   544	
   545	(11)
   546	
   547	2𝑁
   548	(
   549	(∑
   550	))
   551	∑
   552	1 ∑
   553	𝑝 𝑝
   554	𝑛 𝑛
   555	log 1 + exp
   556	,
   557	𝑗𝑘 𝑤𝑗𝑘 𝑑𝑗𝑘 −
   558	𝑗𝑙 𝑤𝑗𝑙 𝑑𝑗𝑙
   559	2𝑁 𝑗=1
   560	
   561	( )
   562	(
   563	)
   564	𝑝
   565	exp 𝑑𝑗𝑘
   566	exp −𝑑𝑗𝑙𝑛
   567	𝑤𝑝𝑗𝑘 = ∑
   568	( ) , 𝑤𝑛𝑗𝑙 = ∑
   569	(
   570	),
   571	𝑝
   572	𝑛
   573	𝑝
   574	𝑝
   575	𝑑 𝑛 ∈𝑆 𝑛 exp −𝑑𝑗𝑙
   576	𝑑 ∈𝑆 exp 𝑑𝑗𝑘
   577	𝑗𝑘
   578	
   579	𝑗𝑙
   580	
   581	𝑗
   582	
   583	#IDs
   584	
   585	#Images
   586	
   587	#V/I Cams
   588	
   589	Occlusion
   590	
   591	Clothes Change
   592	
   593	RegDB [31]
   594	SYSU-MM01 [32]
   595	LLCM [33]
   596	CMG-P (Ours)
   597	
   598	412
   599	491
   600	1064
   601	1011
   602	
   603	8240
   604	38,271
   605	46,767
   606	72,175
   607	
   608	1/1
   609	4/2
   610	9/9
   611	3/3
   612	
   613	×
   614	×
   615	×
   616	✓
   617	
   618	×
   619	×
   620	×
   621	✓
   622	
   623	to the large appearance diﬀerences between visible and infrared images, the features of the two modalities are initially distant from each
   624	other. When using the training approach of CLIP-ReID, the generated
   625	text embeddings are close to the corresponding original image features,
   626	but there will still be a signiﬁcant gap. If the ﬁxed text prompts are used
   627	to guide the image encoder, the distance between the text embeddings
   628	remains unchanged, which could hinder the process of bringing the image features of the two modalities closer together. Fig. 3
   629	Therefore, we propose a Spiral Training (ST) strategy that alternately
   630	trains text prompts and the image encoder as shown in Fig. 4. The text token training stage is deﬁned as 𝑠𝑡𝑡 , and the image encoder training stage
   631	{
   632	}
   633	as 𝑠𝑖𝑒 . ST can be represented as a training stage sequence 𝑠1 , 𝑠2 , … , 𝑠𝐺 ,
   634	{
   635	}
   636	where 𝑠𝑔 ∈ 𝑠𝑡𝑡 , 𝑠𝑖𝑒 , 𝑔 ∈ {1, 2, … 𝐺}, and adjacent training stages are diﬀerent. ST allows ﬂexible adjustment of the number of training iterations
   637	for 𝑠𝑡𝑡 and 𝑠𝑖𝑒 , as well as the training parameters for each stage, based
   638	on the task and dataset, to better maintain the similarity between the
   639	image embedding and its corresponding text embedding.
   640	
   641	where 𝑞𝑘 is the one-hot identity vector of image 𝑥𝑘 which can be either
   642	visible or infrared,  is the shared identity classiﬁer for both visible and
   643	infrared images. The formula of WRT loss is as follows:
   644	𝑤𝑟𝑡 =
   645	
   646	Dataset
   647	
   648	(12)
   649	
   650	(13)
   651	
   652	𝑗
   653	
   654	where (𝑥𝑗 , 𝑥𝑘 , 𝑥𝑙 ) denotes a triplet within each training batch for a given
   655	anchor sample 𝑥𝑖. For the anchor 𝑥𝑗 , 𝑆𝑗𝑝 represents the corresponding
   656	𝑝
   657	positive set, and 𝑆𝑗𝑛 represents the negative set. 𝑑𝑗𝑘
   658	and 𝑑𝑗𝑙𝑛 denote the
   659	pairwise Euclidean distances between the positive and negative sample
   660	pairs, respectively.
   661	To enhance the feature similarity between the image and the text
   662	of the same modality, we use the image-to-text cross-entropy in each
   663	modality as follows:
   664	
   665	5. Experiments
   666	5.1. Datasets
   667	
   668	4.3. Spiral training strategy
   669	
   670	Existing V-I ReID datasets are often limited in scale and diversity
   671	due to the challenges of capturing the same individuals across both day
   672	and night conditions. These datasets are typically collected in controlled
   673	environments, lacking real-world complexities such as occlusion, clothing changes, and other appearance variations. To address these challenges, we have developed a new V-I ReID dataset, CMG-P, which incorporates more complex scenarios. As illustrated in Table 1, CMG-P
   674	introduces challenging real-world conditions, including occlusion and
   675	clothing changes, while oﬀering a larger scale with 72,175 images. This
   676	is a signiﬁcant increase compared to existing V-I ReID datasets such as
   677	RegDB [31], SYSU-MM01 [32], and LLCM [33]. For all datasets, we utilize Cumulative Matching Characteristics (CMC) curves, mean Average
   678	Precision (mAP) and mean Inverse Negative Penalty (mINP) [30]as the
   679	evaluation metrics.
   680	
   681	In CLIP-ReID, the image-text-image training paradigm, where untrained image features are used to generate text prompts and then the
   682	ﬁxed text prompts are used to train image features, heavily relies on the
   683	similarity between text and image embeddings. For the same ID, due
   684	
   685	5.1.1. CMG-P
   686	5.1.2. Eﬀect of diﬀerent training sequences
   687	We analyze the impact of diﬀerent training sequences on the CMG-P
   688	dataset, with experiments conducted in the Infrared-Visible mode.
   689	
   690	𝑖2𝑡𝑐𝑒𝑣 = −
   691	
   692	1 ∑∑
   693	𝑟 log𝑃 (𝑦𝑡𝑣 |𝑥𝑣𝑗 ),
   694	𝑘
   695	𝑁 𝑗=1 𝑘=1 𝑗𝑘
   696	
   697	(14)
   698	
   699	𝑖2𝑡𝑐𝑒𝑖 = −
   700	
   701	1 ∑∑
   702	𝑟 log𝑃 (𝑦𝑡𝑢 |𝑥𝑢𝑗 ),
   703	𝑘
   704	𝑁 𝑗=1 𝑘=1 𝑗𝑘
   705	
   706	(15)
   707	
   708	𝑁
   709	
   710	𝑁
   711	
   712	𝑁
   713	
   714	𝑁
   715	
   716	where 𝑟𝑗𝑘 = 1 if 𝑥𝑗 and 𝑡𝑘 belong to the same ID, and 0 otherwise.
   717	Based on the above loss functions, only the parameters of the image
   718	encoder (⋅) need to be trained. The overall objective function in the
   719	image encoder training stage is formulated as follows:
   720	𝑖𝑚𝑎𝑔𝑒 = 𝑖𝑑 + 𝜆2 𝑤𝑟𝑡 + 𝜆3 𝑖2𝑡𝑐𝑒𝑣 + 𝜆4 𝑖2𝑡𝑐𝑒𝑖 + 𝜆5 𝑖2𝑡𝑐 .
   721	
   722	(16)
   723	
   724	5
   725	
   726	Pattern Recognition 177 (2026) 113333
   727	
   728	J. Xiong et al.
   729	
   730	Fig. 4. The Spiral Training (ST) strategy alternates between training the text prompt and the image encoder, allowing the parameters at each stage to be adjusted

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf' - | nl -ba | sed -n '216,710p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   216	3. Method
   217	3.1. Preliminaries and baseline
   218	
   219	(4)
   220	
   221	The total baseline loss is:
   222	𝑏𝑎𝑠𝑒 = 𝑣intra + 𝑟intra + 𝑣inter + 𝑟inter .
   223	
   224	Preliminaries. Let 𝑋 = {𝑋𝑣 , 𝑋𝑟 } denote the unlabeled datasets
   225	from visible and infrared modalities, where 𝑋𝑣 = {𝑥𝑣1 , 𝑥𝑣2 , … , 𝑥𝑣𝑁 } and
   226	𝑋𝑟 = {𝑥𝑟1 , 𝑥𝑟2 , … , 𝑥𝑟𝑀 }. We utilize a two-stream encoder with shared
   227	parameters 𝜃 and modality-specific heads to extract features, denoted
   228	𝑣 } and
   229	𝑟 }. To generate
   230	as 𝐹𝑣 = {𝑓1𝑣 , 𝑓2𝑣 , … , 𝑓𝑁
   231	𝐹𝑟 = {𝑓1𝑟 , 𝑓2𝑟 , … , 𝑓𝑀
   232	initial pseudo-labels, DBSCAN is applied independently on 𝐹𝑣 and 𝐹𝑟 ,
   233	yielding cluster sets 𝑣 = {𝐶1𝑣 , 𝐶2𝑣 , … , 𝐶𝑆𝑣 } and 𝑟 = {𝐶1𝑟 , 𝐶2𝑟 , … , 𝐶𝐿𝑟 },
   234	where 𝑆 and 𝐿 are the number of clusters. To explicitly model cameraspecific biases, we partition each cluster into finer camera domains
   235	based on camera IDs. Let 𝑣 = {𝜙𝑣1 , … , 𝜙𝑣𝑆×𝐾 } and 𝑟 = {𝜙𝑟1 , … , 𝜙𝑟𝐿×𝐾 }
   236	𝑣
   237	𝑟
   238	represent the sets of camera-domain centroids, computed by averaging
   239	features within each camera-specific subset of a cluster, where 𝐾𝑣 and
   240	𝐾𝑟 denote the number of cameras in each modality.
   241	Baseline. We adopt CEIL [27] as our baseline. In the first stage,
   242	CEIL generates initial pseudo-labels via independent clustering to initialize cluster prototypes:
   243	1 ∑ 𝑡
   244	𝑚𝑡𝑞 = 𝑡
   245	𝑓 and 𝑡 ∈ {𝑣, 𝑟},
   246	(1)
   247	|𝐶𝑖 | 𝑠∈𝐶 𝑡 𝑠
   248	
   249	(5)
   250	
   251	Our proposed modules are integrated into the second stage to robustify this learning process. The overall framework of CLNS is illustrated
   252	in Fig. 2.
   253	3.2. Camera-aware prototype calibration
   254	Standard clustering often yields noisy pseudo-labels. While neighborhood consistency strategies like RPNR [22] effectively filter random
   255	noise, they fail in US-VI-ReID contexts. High visual similarity within
   256	the same camera view (due to background and lighting) creates false
   257	neighborhoods, causing standard methods to reinforce camera-specific
   258	errors rather than correct them. To address this, we introduce the
   259	Camera-aware Prototype Calibration (CPC) module, which exploits
   260	cross-camera consistency to filter unreliable samples and construct
   261	robust, camera-invariant prototypes.
   262	Taking the visible modality as an example, let 𝑣 = {𝐶1𝑣 , … , 𝐶𝑆𝑣 }
   263	denote the initial clusters. For a sample 𝑥𝑣𝑖 with feature 𝐟𝑖𝑣 , we suppress
   264	
   265	𝑖
   266	
   267	3
   268	
   269	Pattern Recognition 179 (2026) 113873
   270	
   271	S. Zhao et al.
   272	
   273	intra-camera bias by masking distances between samples from the same
   274	camera:
   275	{
   276	‖𝐟𝑖𝑣 − 𝐟𝑗𝑣 ‖2
   277	if 𝑐𝑎𝑚𝑣𝑖 ≠ 𝑐𝑎𝑚𝑣𝑗 ,
   278	̃
   279	𝐷𝑖𝑗 =
   280	(6)
   281	+∞
   282	otherwise,
   283	
   284	3.4. Neighbor-guided camera-domain learning
   285	While OTPM establishes reliable correspondences, residual camera
   286	noise can still destabilize feature learning. To address this, we propose
   287	the Neighbor-guided Camera-domain Learning (NCL) module. Unlike
   288	prior works [28] that operate at the coarse cluster level, NCL utilizes
   289	camera-domain centroids as fine-grained proxies to explicitly capture
   290	and adapt to camera-specific variations.
   291	Given the training features {𝑈𝑣 , 𝑈𝑟 } and the camera-domain centroid sets 𝑣 and 𝑟 , we first initialize the centroid for the 𝑙th visible
   292	camera domain 𝑐𝑙𝑣 as:
   293	1 ∑ 𝑣
   294	𝜙𝑣𝑙 = 𝑣
   295	𝐟 ,
   296	(10)
   297	|𝑐𝑙 | 𝑖∈𝑐 𝑣 𝑖
   298	
   299	where 𝑐𝑎𝑚𝑣𝑖 denotes the camera ID. Based on ̃
   300	𝐷, we retrieve the set
   301	of 𝐾1 nearest cross-camera neighbors 𝑖𝑣 . Sample 𝑖’s reliability is
   302	measured by its local label consistency:
   303	1 ∑
   304	𝑠𝑣𝑖 =
   305	I(𝑦𝑣𝑗 = 𝑦𝑣𝑖 ),
   306	(7)
   307	𝐾1
   308	𝑣
   309	𝑗∈𝑖
   310	
   311	where I(⋅) is the indicator function. A high 𝑠𝑣𝑖 implies the sample shares
   312	the same identity with neighbors from different views, providing a robust signal against camera-specific outliers. Consequently, we construct
   313	a reliable sample set 𝑣𝑐 = {𝑖 ∣ 𝑦𝑣𝑖 = 𝑐, 𝑠𝑣𝑖 ≥ 𝜌} for each cluster using a
   314	threshold 𝜌. The calibrated prototype 𝐩𝑣𝑐 is computed by aggregating
   315	only these reliable samples:
   316	1 ∑ 𝑣
   317	𝐩𝑣𝑐 = 𝑣
   318	𝐟 .
   319	(8)
   320	|𝑐 | 𝑖∈𝑣 𝑖
   321	
   322	𝑙
   323	
   324	where 𝐟𝑖𝑣 is the feature of sample 𝑥𝑣𝑖 . For a query sample 𝑞𝑖𝑣 assigned
   325	to the 𝑙′ -th visible camera domain, let its matched infrared camera
   326	domain (identified by OTPM) be indexed by 𝑀. We retrieve its 𝐾2
   327	nearest neighbors from 𝑈𝑣 , denoted as  (𝑞𝑖𝑣 , 𝑈𝑣 , 𝐾2 ). Based on this
   328	neighborhood, we estimate the intra-modality correlation with any 𝑙th
   329	visible camera domain:
   330	[
   331	]
   332	| (𝑞𝑖𝑣 , 𝑈𝑣 , 𝐾2 ) ∩ 𝑐𝑙𝑣 |
   333	𝑃𝑞intra
   334	=
   335	.
   336	(11)
   337	𝑣
   338	𝑖
   339	| (𝑞𝑖𝑣 , 𝑈𝑣 , 𝐾2 ) ∪ 𝑐𝑙𝑣 |
   340	𝑙
   341	
   342	𝑐
   343	
   344	It is worth noting that some identities may only appear in a single
   345	camera. To ensure compatibility with such single-camera identities,
   346	we employ a straightforward fallback mechanism: if the reliable set
   347	𝑣𝑐 becomes empty after filtering, we simply retain the original uncalibrated cluster centroid. Finally, we rectify the initial noisy labels
   348	by reassigning every sample to its nearest calibrated prototype: 𝑦̂𝑣𝑖 =
   349	arg max𝑐 cos(𝐟𝑖𝑣 , 𝐩𝑣𝑐 ). This calibration yields a robust prototype set 𝑃 𝑣 =
   350	{𝐩𝑣1 , … , 𝐩𝑣𝑆 }, providing a clean foundation for cross-modality alignment.
   351	
   352	Similarly, the inter-modality correlation with any 𝑙th infrared camera
   353	domain is computed using neighbors from 𝑈𝑟 :
   354	[
   355	]
   356	| (𝑞𝑖𝑣 , 𝑈𝑟 , 𝐾2 ) ∩ 𝑐𝑙𝑟 |
   357	𝑃𝑞inter
   358	=
   359	.
   360	𝑣
   361	𝑖
   362	| (𝑞𝑖𝑣 , 𝑈𝑟 , 𝐾2 ) ∪ 𝑐𝑙𝑟 |
   363	𝑙
   364	
   365	A higher value of 𝑃 indicates greater consistency between the sample’s
   366	neighborhood and the target camera domain, serving as a reliable soft
   367	supervision signal.
   368	To mitigate label noise, we fuse the hard one-hot pseudo-label with
   369	this neighbor-based soft distribution. Let 𝐈intra
   370	and 𝐈inter
   371	be the one-hot
   372	𝑞𝑣
   373	𝑞𝑣
   374	
   375	3.3. Optimal transport prototype matching
   376	While CPC ensures intra-modality purity, the semantic correspondence between visible (𝑃 𝑣 ) and infrared (𝑃 𝑟 ) prototypes remains unknown due to independent clustering. Existing methods [19] typically
   377	employ Optimal Transport (OT) for global cluster alignment. However,
   378	relying solely on coarse cluster-level alignment overlooks the internal
   379	structure caused by camera variations, leading to suboptimal matching.
   380	To bridge this gap, we propose the Optimal Transport Prototype Matching (OTPM) module, implementing a dual-level alignment strategy. By
   381	establishing correspondences at both the cluster level and the cameradomain level (), OTPM ensures alignment in both abstract identity
   382	space and fine-grained structural space.
   383	Formally, we define the cluster prototype sets 𝑃 𝑣 , 𝑃 𝑟 and the
   384	camera-domain centroid sets 𝑣 , 𝑟 (as defined in Section 3.1). Since
   385	the cardinalities often differ across modalities (e.g., 𝑆 ≠ 𝐿), we model
   386	the alignment as an OT problem. For cluster-level matching, we seek a
   387	transport plan 𝑄 ∈ R𝑆×𝐿 by minimizing the transport cost:
   388	min⟨𝑄, 𝐶⟩ + 𝜆1 (𝑄),
   389	𝑄{
   390	𝑄𝟏 = 𝟏 ⋅ 𝑆1 ,
   391	s.t.
   392	𝑄𝑇 𝟏 = 𝟏 ⋅ 𝐿1 ,
   393	
   394	(12)
   395	
   396	𝑖
   397	
   398	𝑖
   399	
   400	vector for the assigned domain 𝑙′ , and matched domain 𝑀, respectively.
   401	The refined soft targets are defined as:
   402	𝐈̃intra
   403	= 𝜇𝐈intra
   404	+ (1 − 𝜇)𝑃̃𝑞intra
   405	,
   406	𝑣
   407	𝑞𝑣
   408	𝑞𝑣
   409	
   410	(13)
   411	
   412	𝐈̃inter
   413	= 𝜇𝐈inter
   414	+ (1 − 𝜇)𝑃̃𝑞inter
   415	,
   416	𝑣
   417	𝑞𝑣
   418	𝑞𝑣
   419	
   420	(14)
   421	
   422	𝑖
   423	
   424	𝑖
   425	
   426	𝑖
   427	
   428	𝑖
   429	
   430	𝑖
   431	
   432	𝑖
   433	
   434	where 𝑃̃ is the 𝓁1 -normalized correlation vector and 𝜇 is a balancing
   435	factor. With these targets, we define smoothed cross-entropy losses for
   436	intra- and inter-modality learning:
   437	⎛
   438	
   439	⎞
   440	( 𝑣 𝑣 ) ⎟
   441	𝑞𝑖 ⋅𝝓𝑘 ∕𝜏
   442	⎟
   443	⎟,
   444	(
   445	)⎟
   446	𝑣 𝑣
   447	exp 𝑞𝑖 ⋅𝝓𝑗 ∕𝜏 ⎟
   448	⎜
   449	⎝ 𝑗=1
   450	⎠
   451	
   452	⎜
   453	𝑣
   454	𝑆×𝑐𝑎𝑚
   455	∑
   456	⎜ exp
   457	𝑣𝑖𝑛𝑡𝑟𝑎=−
   458	𝐈̃intra
   459	(𝑘)⋅log
   460	⎜𝑆×𝑐𝑎𝑚𝑣
   461	𝑞𝑖𝑣
   462	⎜ ∑
   463	𝑘=1
   464	
   465	⎞
   466	( 𝑣 𝑣→𝑟 ) ⎟
   467	𝑞𝑖 ⋅𝝓𝑘 ∕𝜏 ⎟
   468	⎟,
   469	(
   470	)⎟
   471	𝑣 𝑣→𝑟
   472	exp 𝑞𝑖 ⋅𝝓𝑗 ∕𝜏 ⎟
   473	⎜
   474	⎝ 𝑗=1
   475	⎠
   476	
   477	(15)
   478	
   479	⎛
   480	
   481	⎜
   482	𝑟
   483	𝐿×𝑐𝑎𝑚
   484	∑
   485	⎜ exp
   486	𝑣𝑖𝑛𝑡𝑒𝑟=− 𝐈̃inter
   487	(𝑘)⋅log
   488	⎜𝐿×𝑐𝑎𝑚𝑟
   489	𝑞𝑖𝑣
   490	⎜ ∑
   491	𝑘=1
   492	
   493	(9)
   494	
   495	where 𝐶𝑖𝑗 = 1 − cos(𝐩𝑣𝑖 , 𝐩𝑟𝑗 ) is the cost matrix, and (𝑄) is the entropic
   496	regularization term. This optimization is efficiently solved via the
   497	Sinkhorn–Knopp algorithm, yielding the optimal plan 𝑄∗ . Simultaneously, an analogous optimization is performed on 𝑣 and 𝑟 to obtain
   498	the domain-level plan 𝑄∗𝑑𝑜𝑚 .
   499	From these optimal plans, we derive probabilistic mappings to guide
   500	subsequent learning. The cluster-level correspondences are defined as
   501	𝑦𝑣→𝑟
   502	= arg max𝑗 𝑄∗𝑖𝑗 (and vice versa). Similarly, we obtain the matched
   503	𝑖
   504	camera-domain centroids 𝜙𝑣→𝑟 and 𝜙𝑟→𝑣 from 𝑄∗𝑑𝑜𝑚 . This dual-matching
   505	mechanism bridges the modality gap: cluster correspondences ensure
   506	identity consistency, while camera-domain correspondences capture
   507	structural alignment, providing precise supervision for instance-level
   508	refinement.
   509	
   510	(16)
   511	
   512	where 𝜙𝑣→𝑟
   513	represents the mapped infrared centroid obtained from
   514	𝑘
   515	OTPM, ensuring semantic alignment.
   516	Furthermore, to suppress unreliable samples, we introduce a
   517	confidence-aware weighting mechanism. The confidence weight for 𝑞𝑖𝑣
   518	is derived from the consistency score of its assigned domain:
   519	(
   520	(
   521	)2 )
   522	intra
   523	𝜔intra
   524	=
   525	exp
   526	−𝑤
   527	1
   528	−
   529	[𝑃
   530	]
   531	,
   532	(17)
   533	′
   534	𝑣
   535	𝑣
   536	𝑙
   537	𝑞
   538	𝑞
   539	𝑖
   540	
   541	(
   542	
   543	𝑖
   544	
   545	(
   546	
   547	𝜔inter
   548	= exp −𝑤 1 − [𝑃𝑞inter
   549	]𝑀
   550	𝑣
   551	𝑞𝑣
   552	𝑖
   553	
   554	)2 )
   555	
   556	,
   557	
   558	(18)
   559	
   560	𝑖
   561	
   562	where 𝑤 controls the sensitivity. Samples with consistent neighborhoods receive higher weights, while outliers are down-weighted. The
   563	4
   564	
   565	Pattern Recognition 179 (2026) 113873
   566	
   567	S. Zhao et al.
   568	
   569	4. Experiments
   570	
   571	final NCL objective aggregates the weighted losses over all samples in
   572	a batch 𝐵𝑣 :
   573	1 ∑ intra 𝑣
   574	𝑣𝑛𝑐𝑙 =
   575	(𝜔
   576	𝑖𝑛𝑡𝑟𝑎 + 𝜔inter
   577	𝑣𝑖𝑛𝑡𝑒𝑟 ).
   578	(19)
   579	𝑞
   580	|𝐵𝑣 | 𝑞∈𝐵 𝑞
   581	
   582	4.1. Datasets and metrics
   583	We evaluate our method on three public benchmarks: SYSU-MM01
   584	[29], RegDB [30] and LLCM [31].
   585	SYSU-MM01 is a large-scale dataset captured by 6 cameras in indoor and outdoor environments. It contains 22,258 visible and 11,909
   586	infrared training images of 395 identities, and 96 testing identities. We
   587	report average performance over 10 random splits for both all-search
   588	and indoor-search modes.
   589	RegDB consists of 412 identities, each having 10 visible and 10
   590	infrared images. The dataset is randomly divided into two halves (206
   591	identities each) for training and testing. We report the average results of
   592	10 trials for both Visible-to-Infrared (VIS-to-IR) and Infrared-to-Visible
   593	(IR-to-VIS) modes.
   594	LLCM is a challenging dataset collected in low-light environments
   595	using a 9-camera network. It comprises 46,767 annotated images of
   596	1064 identities, split into 713 identities for training and 351 for testing.
   597	Evaluation follows the RegDB protocol, assessing both VIS-to-IR and
   598	IR-to-VIS retrieval modes.
   599	Evaluation metrics. We employ Cumulative Matching Characteristics (CMC), mean Average Precision (mAP), and mean Inverse Negative
   600	Penalty (mINP) [32] as evaluation metrics, mINP measures the cost of
   601	retrieving the hardest correct match, providing a robust assessment of
   602	the model’s ability to handle difficult samples.
   603	
   604	𝑣
   605	
   606	The total NCL loss is 𝑛𝑐𝑙 = 𝑣𝑛𝑐𝑙 + 𝑟𝑛𝑐𝑙 , and the total loss for the
   607	proposed framework is 𝑡𝑜𝑡𝑎𝑙 = 𝑏𝑎𝑠𝑒 + 𝑛𝑐𝑙 . By operating at the
   608	camera-domain level, NCL effectively densifies feature distributions
   609	and mitigates residual noise, complementing the structural alignment
   610	provided by CPC and OTPM.
   611	3.5. Noise-aware memory updating
   612	Conventional momentum updates treat all samples equally, making
   613	the memory bank vulnerable to outliers that drift the centroids away
   614	from the true distribution. To counteract this, we propose the Noiseaware Memory Updating (NMU) strategy, which adaptively re-weights
   615	sample contributions based on their reliability.
   616	Consider a camera domain with centroid 𝝓𝑘 and a batch of assigned
   617	samples 𝑘 = {𝐟𝑖 }𝑏𝑖=1 . We first quantify the ‘‘noise probability’’ of each
   618	sample by its normalized angular deviation from the centroid:
   619	𝑑𝑖 = 1 − cos(𝐟𝑖 , 𝝓𝑘 ),
   620	
   621	𝑝𝑖 =
   622	
   623	𝑑𝑖
   624	.
   625	max𝑗∈𝑘 𝑑𝑗 + 𝜖
   626	
   627	(20)
   628	
   629	where 𝜖 is a small constant for numerical stability. A larger 𝑝𝑖 indicates
   630	the sample is likely an outlier. We then compute a reliability weight 𝑤𝑖
   631	using a Softmax function over −𝑝𝑖 , ensuring clean samples dominate
   632	the update:
   633	𝑤𝑖 = ∑
   634	
   635	exp(−𝑝𝑖 )
   636	,
   637	𝑗∈𝑘 exp(−𝑝𝑗 )
   638	
   639	𝐟̄𝑘 =
   640	
   641	∑
   642	
   643	𝑤𝑖 𝐟𝑖 .
   644	
   645	4.2. Implementation details
   646	To ensure a fair comparison and isolate the effectiveness of our
   647	proposed modules, we construct our framework based on the two-stage
   648	training paradigm of CEIL [27]. Specifically, we retain the first-stage
   649	clustering and initialization pipeline but replace the core components
   650	of CEIL with our proposed CPC, OTPM, NCL, and NMU modules.
   651	We employ a dual-stream ResNet-50 pretrained on ImageNet as the
   652	feature extractor. All input images are resized to 288 × 144. Data augmentation includes horizontal flipping, random erasing, and channel
   653	augmentation. The training settings are tailored to the scale of each
   654	dataset. For the large-scale SYSU-MM01 and LLCM datasets, each minibatch contains 8 identities, with 10 visible and 10 infrared images per
   655	identity. For the smaller RegDB dataset, we sample 4 identities with
   656	5 images per modality. The model is trained for a total of 90 epochs
   657	using the Adam optimizer. We adopt a warm-up strategy for the first 10
   658	epochs, linearly increasing the learning rate from 3.5×10−6 to 3.5×10−4 .
   659	Subsequently, the learning rate is decayed to 3.5 × 10−5 at epoch 20 and
   660	further to 3.5 × 10−6 at epoch 50.
   661	In the pseudo-label generation stage, we utilize the Jaccard distance
   662	based on k-reciprocal encoding to compute feature similarities. The
   663	DBSCAN algorithm is applied with a maximum distance threshold set
   664	to 0.6 and the minimum number of samples set to 4. For the contrastive
   665	learning, the temperature parameter 𝜏 is empirically set to 0.05.
   666	Our proposed modules are formally integrated into the training
   667	process during the second stage. All experiments are conducted on a
   668	single NVIDIA RTX 3090 GPU with PyTorch, equipped with an Intel
   669	Core i7-12700 CPU and 64 GB RAM.
   670	
   671	(21)
   672	
   673	𝑖∈𝑘
   674	
   675	Finally, the centroid is updated via 𝝓𝑘 ← 𝛽𝝓𝑘 + (1 − 𝛽)𝐟̄𝑘 . By filtering noise contributions, NMU maintains a pure representation of the
   676	camera domain throughout training.
   677	In unsupervised settings, it is inherently difficult to distinguish
   678	true label noise from informative ‘‘hard positives’’ solely via angular deviations. To handle this ambiguity, NMU eschews hard filtering in favor of a Softmax-based soft-weighting mechanism. This robust design assigns ambiguous samples diminished yet strictly nonzero weights: it effectively suppresses potential noise to prevent centroid drift while retaining hard positives to support the learning of
   679	pose-invariant representations.
   680	Algorithm 1 Training process for CLNS
   681	Require: Unlabeled images 𝑣 = {𝑥𝑣1 , … , 𝑥𝑣𝑁 } and 𝑟 ; feature extractor 𝑓𝜃
   682	Ensure: Trained feature extractor 𝑓𝜃
   683	1: Initialize: Maximal epoch 𝐸; Maximal iteration 𝑀𝑎𝑥𝐼𝑡𝑒𝑟
   684	2: for 𝑒𝑝𝑜𝑐ℎ = 1 to 𝐸 do
   685	3:
   686	Extract features 𝐹𝑣 and 𝐹𝑟 from 𝑣 and 𝑟 via 𝑓𝜃
   687	4:
   688	Generate initial pseudo-labels via DBSCAN clustering
   689	5:
   690	Refine pseudo-labels and construct calibrated prototypes 𝑃 𝑣 , 𝑃 𝑟 via CPC
   691	(Eq. (6)–(8))
   692	6:
   693	Construct camera-domain centroid sets 𝑣 and 𝑟 based on refined
   694	labels
   695	7:
   696	Initialize memory banks and camera-domain prototypes for both
   697	modalities
   698	8:
   699	Establish cross-modality correspondences 𝑦𝑣→𝑟 , 𝑦𝑟→𝑣 , 𝜙𝑣→𝑟 , 𝜙𝑟→𝑣 at both
   700	cluster and camera-domain levels via OTPM (Eq. (9))
   701	9:
   702	for 𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛 = 1 to 𝑀𝑎𝑥𝐼𝑡𝑒𝑟 do
   703	10:
   704	Sample mini-batch  = {𝑣 , 𝑟 }
   705	11:
   706	Calculate overall loss 𝑡𝑜𝑡𝑎𝑙 = 𝑏𝑎𝑠𝑒 + 𝑛𝑐𝑙 via NCL (Eq. (10)–(19))
   707	12:
   708	Update parameters 𝜃 via back-propagation
   709	13:
   710	Update camera-domain centroids in the memory bank via NMU

codex
前三篇的方法主体已经读完。到这里能看到一个明显模式：他们都不是先说“我加了模块”，而是先把错误来源命名成结构性问题，例如提示漂移、相机诱导噪声、衣服偏置，再让机制按这个错误来源分层展开。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '234,865p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   234	A. Problem Definition and Framework Overview
   235	1) Two-Stage Learning Strategy: Our CMAG framework
   236	adopts a progressive two-stage learning approach to address
   237	the complexity of unsupervised cross-modal person reidentification. Stage 1 (50 epochs) establishes robust feature
   238	extraction through a shared ViT backbone with modalityspecific clustering, building discriminative representations for
   239	both RGB and IR modalities. Stage 2 (100 epochs) activates our novel cross-modal components (CATS, DCAM-ViT,
   240	GSMEM) for advanced modality alignment and unified feature
   241	space learning.
   242	In the USL-VI-ReID task, we have two unlabeled datasets:
   243	Nrgb
   244	visible light dataset XRGB = {xirgb }i=1
   245	and infrared dataset
   246	rgb
   247	ir Nir
   248	H×W×3
   249	XIR = {xi }i=1 , where xi ∈ R
   250	and xiir ∈ RH×W×1
   251	represent the i-th visible and infrared images respectively, Nrgb
   252	and Nir denote the number of visible and infrared training
   253	samples, and H, W are the image height and width. The
   254	objective is to learn a unified feature space under unsupervised
   255	conditions.
   256	The CMAG framework employs a multi-task learning
   257	paradigm that integrates loss functions from all modules.
   258	Each component contributes its specialized loss terms: circular
   259	consistency loss from CATS, cross-modal alignment loss from
   260	DCAM-ViT, graph-structured memory enhancement loss from
   261	GSMEM, and camera-aware consistency loss from CARC.
   262	These loss functions are jointly optimized with adaptive weight
   263	adjustment to balance modality alignment, feature enhancement, and pseudo-label optimization throughout the training
   264	process. This multi-stage collaborative optimization allows
   265	the framework to efficiently handle modality differences, feature inconsistency, and camera view bias under unsupervised
   266	conditions.
   267	2) Framework Visualization Details: The visual elements
   268	in Figure 2 are designed to illustrate the key mechanisms of
   269	CMAG. In the clustering phase (a), blue and orange bars represent RGB and IR features respectively, with different colored
   270	stars and circles in the clustering results denoting different
   271	identity clusters. The cycle-aware structure (b) visualizes how
   272	circular paths (thick green edges) connect features to validate
   273	identity consistency, with the formula C p = (A p ) A computing
   274	these paths. The memory module (c) shows both the temporal
   275	queue update mechanism (left, with ‘ptr’ indicating current
   276	position) and adaptive graph construction (right, where node
   277	size reflects local density). Throughout the framework, blue
   278	consistently represents RGB modality while orange represents
   279	IR modality, facilitating cross-modal understanding.
   280	B. Basic Feature Learning Network
   281	
   282	III. P ROPOSED M ETHOD
   283	Unsupervised visible-infrared person re-identification (USLVI-ReID) faces four key challenges: feature inconsistency
   284	caused by modality differences, limited global information access due to batch training constraints, pseudo-label
   285	noise issues, and camera view bias. This section introduces our proposed CMAG (Cross-Modal Attention and
   286	Graph-enhanced Memory) framework, which addresses these
   287	challenges through four innovative components.
   288	
   289	To capture rich semantic information, we adopt a Vision
   290	Transformer (ViT)-based multi-class token feature extraction
   291	network. Inspired by recent multi-token approaches [27],
   292	we employ K = 4 class tokens, significantly enhancing
   293	the model’s ability to express multi-granularity features.
   294	The feature extraction process is represented as shown in
   295	Equation (1):
   296	Zrgb = frgb (xrgb ) ∈ RK×d ,
   297	
   298	Zir = fir (xir ) ∈ RK×d
   299	
   300	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   301	
   302	(1)
   303	
   304	208
   305	
   306	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   307	
   308	Fig. 2. Core module structure of the CMAG Framework with two-stage learning strategy (best viewed in color). The framework employs a two-stage
   309	learning approach: Stage 1 focuses on robust feature representation learning using a shared ViT backbone with modality-specific clustering, enabling effective
   310	feature extraction for both RGB and IR modalities (a). In Stage 2, the framework activates advanced cross-modal modules: (b) Cycle-Aware Topological
   311	Structure (CATS): Activated only in Stage 2 for circular path-based pseudo-label verification. Left shows feature distribution before clustering; right shows
   312	graph structure with circular paths. The formula C p = (A p ) A enables detection of circular paths like A−→B−→C−→D−→A. (c) Graph-Structured Memory
   313	Enhancement Module (GSMEM): Activated in Stage 2 for global cross-batch feature propagation. Left shows queue memory update mechanism maintaining
   314	a global feature memory bank; right demonstrates adaptive graph construction based on local density. Throughout both stages, clustering evolves from
   315	modality-specific operations to unified cross-modal correspondence learning.
   316	
   317	TABLE I
   318	K EY N OTATION IN CMAG F RAMEWORK
   319	
   320	backbone extracts K = 4 class tokens for each modality, where
   321	individual class tokens are obtained as Zi = Zmodality [i − 1] for
   322	i ∈ {1, 2, 3, 4}, with each Zi ∈ Rd capturing distinct semantic
   323	aspects. The final feature representation is obtained by processing each token through dedicated bottleneck layers and
   324	concatenating: F = [Bottleneck1 (Z1 ); . . . ; Bottleneck4 (Z4 )] ∈
   325	R4d×1 .
   326	C. Cycle-Aware Topological Structure
   327	
   328	where all variables are defined in Table I. frgb : RH×W×3 →
   329	RK×d and fir : RH×W×1 → RK×d are ViT-based feature
   330	extraction networks for visible and infrared modalities respectively. Each modality representation contains K class token
   331	features. The final global feature is obtained through vertical
   332	concatenation: F = [Z1 ; Z2 ; . . . ; ZK ] ∈ RKd×1 , where [·; ·]
   333	represents vertical concatenation operation.
   334	1) Multi-Class Token Processing Details: Following the
   335	multi-token framework of TokenMatcher [27], our ViT
   336	
   337	The Cycle-Aware Topological Structure (CATS) module
   338	explores circular structures in feature space to provide a
   339	verification mechanism for pseudo-labels. This section details
   340	the motivation, theoretical foundation, and implementation of
   341	this approach.
   342	1) Problem Analysis and Method Motivation: A core challenge in unsupervised cross-modal learning is constructing
   343	reliable feature space structure without label supervision.
   344	Existing methods mainly focus on direct similarity relationships, neglecting higher-order structures in feature space,
   345	especially the rich semantic information carried by circular
   346	paths. This limitation leads to unstable clustering results and
   347	significant pseudo-label noise in scenarios with substantial
   348	modality differences.
   349	
   350	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   351	
   352	ZHANG et al.: CMAG: CROSS-MODAL ATTENTION AND GRAPH-ENHANCED MEMORY FOR USL-VI-ReID
   353	
   354	While existing methods address pseudo-label noise
   355	through various strategies, they lack mathematical
   356	verification for global identity consistency across crossmodal scenarios.
   357	Samples forming closed-loop connections in feature space
   358	likely belong to the same identity, providing natural verification for pseudo-labels. For pedestrian images with pose
   359	variations, circular structures connect front and back views
   360	through intermediate poses, addressing the feature discontinuity problem that traditional methods cannot effectively solve.
   361	2) Theoretical Foundation of Circular Structure: Circular
   362	structures [42], [43] have special properties in graph theory
   363	[44], [45] and topology. From a topological perspective, a
   364	circular structure is characterized by non-trivial first-order
   365	homology group H1 (G), representing “holes” in the feature
   366	space. Formally, in a graph G=(V,E), if nodes {v1 , v2 , . . . , vn }
   367	form a cycle, then for any node i ∈ {1, 2, . . . , n−1}, there exists
   368	an edge (vi , vi+1 ) and an edge (vn , v1 ), creating a closed path.
   369	In the feature space of person re-identification, non-trivial
   370	elements of the first-order homology group represent sets of
   371	samples that are highly correlated and form closed loops. From
   372	a probability perspective [46], if random walks on the feature
   373	graph can form closed paths, the nodes on such paths are likely
   374	to belong to the same identity category.
   375	Unlike traditional pairwise or triplet constraints, circular
   376	structures provide a global consistency verification mechanism. If samples A−→B−→C−→. . . −→Z−→A form a cycle,
   377	this long-range dependency relationship can effectively resist
   378	the influence of local noise and outliers, particularly important
   379	in cross-modal scenarios where direct similarity measures are
   380	often unreliable.
   381	3) Construction and Path Detection: We propose an efficient circular path detection algorithm that identifies circular
   382	structures in feature space by combining adjacency matrix
   383	power operations with element-wise products:
   384	p
   385	P(p)
   386	path = A ,
   387	
   388	Pcircular = A p
   389	
   390	A
   391	
   392	(2)
   393	
   394	4) Adjacency Matrix Construction Details: The binary
   395	adjacency matrix A ∈ {0, 1}N×N is constructed using
   396	an adaptive k-nearest neighbor approach with label-aware
   397	enhancement:
   398	Step 1 - k-NN Construction: For normalized features
   399	F ∈ RN×d :
   400	similarity = FFT
   401	indicesi = TopK(similarity[i], k + 1)[1 :]
   402	
   403	(3)
   404	(4)
   405	
   406	where k = 10 represents the number of nearest neighbors,
   407	excluding self-connections.
   408	Step 2 - Initial Adjacency Construction:
   409	(
   410	1, if j ∈ indicesi
   411	Ainit
   412	(5)
   413	ij =
   414	0, otherwise
   415	Step 3 - Label-aware Enhancement (when available):
   416	When pseudo-labels are available during training:
   417	(
   418	1, if labeli = label j
   419	Mi j =
   420	(6)
   421	0, otherwise
   422	
   423	209
   424	
   425	Algorithm 1 CATS-Based Circular Graph Construction
   426	Require: Features F ∈ RN×d (normalized), k = 10,
   427	pseudo-labels L (optional)
   428	Ensure: Circular adjacency matrix A ∈ RN×N
   429	1: Step 1: k-NN Construction
   430	2: Compute similarity matrix: S ← FFT
   431	3: Initialize A ← 0N×N
   432	4: Step 2: Initial Adjacency Construction
   433	5: for i = 0 to N − 1 do
   434	6:
   435	indicesi ← TopK(S[i, :], k + 1)[1 :] Exclude self
   436	7:
   437	for j ∈ indicesi do
   438	8:
   439	A[i, j] ← 1
   440	9:
   441	end for
   442	10: end for
   443	11: Step 3: Label-aware Enhancement (if available)
   444	12: if pseudo-labels L are provided then
   445	13:
   446	for i = 0 to N − 1 do
   447	14:
   448	for j = 0 to N − 1 do
   449	15:
   450	if L[i] , L[ j] then
   451	16:
   452	A[i, j] ← 0 Remove cross-label edges
   453	17:
   454	end if
   455	18:
   456	end for
   457	19:
   458	end for
   459	20: end if
   460	21: Step 4: Circular Connection Enhancement
   461	22: for i = 0 to N − 1 do
   462	23:
   463	neighbors ← { j|A[i, j] = 1}
   464	24:
   465	if |neighbors| ≥ 2 then
   466	25:
   467	Sort neighbors by similarity: neighborssorted
   468	26:
   469	A[neighborssorted [−1], neighborssorted [0]] ← 1
   470	27:
   471	end if
   472	28: end for
   473	29: return A Return binary adjacency matrix
   474	
   475	Aenhanced
   476	= Ainit
   477	ij
   478	i j · Mi j
   479	
   480	(7)
   481	
   482	Step 4 - Circular Connection Enhancement: For each
   483	node with at least 2 neighbors, add circular connections to
   484	ensure cycle formation:
   485	A[neighbors[−1], neighbors[0]] = 1.0
   486	
   487	(8)
   488	
   489	where A p represents the p-th power of matrix A, denotes
   490	element-wise multiplication, and N is the total number of
   491	samples in the current batch. P(p)
   492	path [i, j] indicates the number of
   493	paths of length p from node i to node j, while Pcircular [i, j] > 0
   494	signifies the existence of circular paths connecting nodes i
   495	and j.
   496	As shown in Figure 2(b), circular topological structures
   497	can discover identity associations overlooked by traditional
   498	methods. When the direct similarity between two nodes (such
   499	as nodes A and D) is relatively low, traditional methods
   500	often fail to establish associations. However, by detecting the
   501	circular path A−→B−→C−→D−→A, our method can verify
   502	that these nodes likely belong to the same identity. This
   503	mechanism performs well on datasets sampled from video
   504	sequences, as they naturally contain continuous pose transformation information.
   505	
   506	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   507	
   508	210
   509	
   510	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   511	
   512	5) Circular Feature Propagation Mechanism: Based on
   513	the detected circular paths, we design a dual-channel circular information propagation mechanism, differentiating
   514	information flow between regular connections and circular
   515	connections:
   516	Mregular = Wregular · (AX)
   517	
   518	(9)
   519	
   520	Mcircular = Wcircular · (Pcircular X)
   521	
   522	(10)
   523	
   524	Xenhanced = LayerNorm(Wtrans f orm [Mregular , Mcircular ])
   525	
   526	(11)
   527	
   528	where Wregular , Wcircular , and Wtrans f orm are learnable weight
   529	matrices, X is the node feature matrix, and [·, ·] represents
   530	the feature concatenation operation. This design enables the
   531	model to differentiate and integrate different types of structural
   532	information, enhancing the diversity and robustness of feature
   533	representation.
   534	To strengthen the consistency of node features on circular
   535	paths, we propose a circular consistency loss:
   536	X
   537	1
   538	`(i, j)
   539	(12)
   540	Lcircular = −
   541	|Pcircular |
   542	(i, j)∈Pcircular
   543	
   544	exp(sim( fi , f j )/τ)
   545	where `(i, j) = log PN exp(sim(
   546	represents the contrastive
   547	fi , fk )/τ)
   548	k=1
   549	
   550	loss between each pair of nodes.
   551	Where Pcircular is the set of node pairs on circular paths,
   552	sim(·, ·) is the cosine similarity function, and τ is a temperature
   553	parameter (set to 0.07). We also constructed a cross-modal
   554	joint circular graph for structured information interaction
   555	between different modalities:
   556	Z joint = EncoderNetwork([ frgb ; fir ; M])
   557	
   558	(13)
   559	
   560	Across
   561	circular = ConstructCircular(Z joint , Y joint )
   562	
   563	(14)
   564	
   565	Zenhanced = CircularGCN(Z joint , Across
   566	circular )
   567	
   568	(15)
   569	
   570	This mechanism achieves modality alignment at the structural level of feature space, improving cross-modal matching
   571	performance.
   572	D. Dynamic Cross-Modal Attention Mechanism for ViT
   573	1) Modality Difference Analysis and Challenges: Crossmodal scenarios face inherent modality differences that cause
   574	inconsistent feature distributions, requiring sophisticated attention mechanisms beyond traditional CNN-based approaches.
   575	Existing CNN-based cross-modal alignment methods employ
   576	simple feature projections or adversarial learning, failing to
   577	utilize Transformer’s self-attention capabilities [47] and struggling to capture local correspondences and complex non-linear
   578	relationships between modalities [48].
   579	2) Cross-Modal Attention Design: Token-wise Crossmodal Attention: Building upon the multi-token architecture
   580	inspired by TokenMatcher [27], our DCAM-ViT operates at
   581	the individual class token level, where each token Zsource
   582	∈
   583	i
   584	RB×d from the source modality interacts with the complete target modality representation Ztarget ∈ RB×K×d through attention
   585	mechanism: EnhancedZi = DCAM(Zsource
   586	, Ztarget ), enabling
   587	i
   588	fine-grained cross-modal feature alignment.
   589	Addressing these challenges, we design the first crossmodal attention mechanism specifically tailored for Vision
   590	Transformer architecture. This mechanism leverages ViT’s
   591	
   592	class token characteristics, achieving more precise modality
   593	alignment through fine-grained attention computation. The
   594	implementation includes three key steps:
   595	In our cross-modal attention mechanism, individual class
   596	tokens Zi ∈ Rd×1 are extracted from the source modality
   597	representation and interact with the complete target modality
   598	representation through the following attention computation:
   599	First, we construct a low-dimensional interaction space
   600	through asymmetric feature projection, reducing computational complexity while enhancing modality interaction
   601	flexibility:
   602	q = Wq Zi ∈ Rd/8×1 ,
   603	k = Wk Ztarget ∈ RK×d/8 ,
   604	v = Wv Ztarget ∈ RK×d/8
   605	
   606	(16)
   607	
   608	where f source , ftarget ∈ RB×d represent the source and target
   609	modality feature matrices respectively, B is the batch size,
   610	d
   611	and Wq , Wk , Wv ∈ Rd× 8 are learnable projection matrices for
   612	query, key, and value transformations respectively. The feature
   613	dimension d is defined in Table I.
   614	Second, we introduce a batch-wise dynamic attention
   615	computation mechanism for fine-grained cross-modal feature
   616	selection:
   617	p
   618	αi = softmax(qTi k/ d/8) ∈ RK
   619	(17)
   620	vweighted = α · v ∈ RB×d
   621	
   622	(18)
   623	
   624	d
   625	
   626	where qi ∈ R 8 is the i-th row of q (i.e., query vector for the
   627	d
   628	i-th sample),
   629	kT ∈ R 8 ×B is the transpose of the key matrix,
   630	√
   631	and d/8 is the scaling factor for numerical stability.
   632	From an information theory perspective [48], crossmodal learning must maximize mutual information between
   633	modalities while preserving modality-specific discriminative
   634	information. After obtaining weighted features, we design an
   635	adaptive feature fusion network with a residual structure to
   636	balance these objectives:
   637	fused = concat[ f source , vweighted ] ∈ RB×2d
   638	fenhanced = f source + γ · MLP( fused ) ∈ R
   639	
   640	B×d
   641	
   642	(19)
   643	(20)
   644	
   645	where γ is a learnable balance parameter, and MLP is a multilayer perceptron. The residual connections preserve original
   646	source modality information, avoiding the loss of modalityspecific features while enhancing shared information.
   647	The key advantages of our cross-modal attention mechanism
   648	include: utilizing ViT’s multi-class token structure for finegrained feature interaction, implementing batch-wise dynamic
   649	attention for flexible modality alignment, and adopting a
   650	residual structure to balance modality-specific and shared
   651	information.
   652	E. Graph-Structured Memory Enhancement Module
   653	While CATS provides structured feature representations,
   654	batch training fundamentally limits access to global data distribution. Theoretically, larger batch sizes benefit unsupervised
   655	cross-modal learning through: (1) providing more negative
   656	samples for contrastive optimization [49], (2) reducing gradient variance (Var[∇Lbatch ] = σ2 /B) for stable convergence
   657	
   658	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   659	
   660	ZHANG et al.: CMAG: CROSS-MODAL ATTENTION AND GRAPH-ENHANCED MEMORY FOR USL-VI-ReID
   661	
   662	[50], and (3) better approximating true data distribution for
   663	improved clustering [51].
   664	Based on these insights, we propose GSMEM to address
   665	batch limitations through global memory banks and crossbatch relationship graphs.
   666	1) Global Memory Bank Design: Our GSMEM implements
   667	a dynamic queue memory update mechanism to maintain a
   668	global feature memory bank:
   669	Mt [i] = f j ,
   670	
   671	i = (ptr + i0 )
   672	
   673	mod
   674	
   675	M
   676	
   677	(21)
   678	
   679	where Mt ∈ R M×d is the memory bank at time step t, M is
   680	the memory bank size, f j ∈ Rd is the feature vector to be
   681	stored, ptr ∈ {0, 1, . . . , M − 1} is the current memory pointer,
   682	and i0 ∈ {0, 1, . . . , B − 1} is the index within the current batch.
   683	Additionally, the memory bank design maintains balance
   684	between visible and infrared samples through stratified sampling, ensuring the ratio of modality samples is close to 1:1
   685	to reduce modality bias.
   686	2) Adaptive Graph Construction and Feature
   687	Propagation:
   688	a) Memory graph construction details: Our GSMEM
   689	constructs normalized k-NN graphs to facilitate cross-batch
   690	feature propagation. Unlike CATS which uses label-aware
   691	k-NN for pseudo-label validation, GSMEM employs degree
   692	normalized k-NN graphs for stable feature enhancement.
   693	b) Unified k-NN construction: The similarity matrix S is
   694	computed using normalized features:
   695	S = ZZT ,
   696	
   697	Z = Wnode · [fcurrent ; Mmemory ]
   698	
   699	(22)
   700	
   701	where fcurrent represents current batch features, Mmemory represents memory bank features, and Wnode ∈ Rd×d is a learnable
   702	transformation matrix.
   703	c) k-NN graph with degree normalization: Following the
   704	same k-NN approach as CATS:
   705	indicesi = TopK(S[i], k + 1)[1 :] // k=10, exclude self
   706	Araw
   707	i j = 1 if j ∈ indicesi ,
   708	raw
   709	A=A
   710	
   711	else 0
   712	
   713	+ I // add self-loops
   714	
   715	A = 0.5 × (A + AT ) // symmetrization
   716	Anorm = D−1/2 AD−1/2
   717	
   718	// degree normalization
   719	
   720	(23)
   721	
   722	where D is the degree matrix. The key difference from CATS
   723	is the addition of degree normalization for stable cross-batch
   724	feature propagation, while CATS focuses on circular path
   725	detection without normalization.
   726	Based on the memory bank, we propose an adaptive
   727	k-nearest neighbor graph construction algorithm to capture
   728	complex relationships between samples:
   729	S = ZZ T ,
   730	
   731	Z = Wnode · [ f ; M]
   732	
   733	(24)
   734	
   735	where f is the current batch features, Wnode is the node feature
   736	projection matrix, and S is the similarity matrix. The adaptive
   737	number of neighbors ki is dynamically determined by:
   738	ki = max(min k, basek − β · density(i))
   739	
   740	(25)
   741	
   742	211
   743	
   744	d) Graph construction implementation: The similarity
   745	matrix S is computed using normalized features, where each
   746	element S i j represents the cosine similarity between samples
   747	i and j. The node features Z are obtained by projecting
   748	the concatenated current batch features f and memory bank
   749	features M through a learnable transformation Wnode ∈ Rd×d .
   750	For each node i, we select the top ki most similar nodes as
   751	neighbors, where ki is adaptively determined by:
   752	(26)
   753	ki = max(([a − z]+) k, base k − β · density(i))
   754	P
   755	where density(i) = R1 j∈NR (i) 1 measures the local density
   756	within radius R = 0.3 around node i. The empirically
   757	determined parameters are: ([a − z]+) k = 3 (minimum
   758	connectivity), base k = 10 (base neighbor count), and β =
   759	0.5 (regulation parameter). This adaptive mechanism reduces
   760	connections in dense feature regions to avoid noise while
   761	maintaining sufficient connectivity in sparse areas.
   762	As shown in Figure 2(c), our adaptive algorithm adjusts the
   763	k-nearest neighbor parameter based on local feature density.
   764	In dense areas (red box), the algorithm reduces connections
   765	to avoid noise, while in sparse areas, it maintains sufficient
   766	connections for information flow.
   767	After graph construction, we implement a residual graph
   768	convolutional network for feature propagation:
   769	
   770	 1
   771	1
   772	(27)
   773	Z (l+1) = σ D− 2 ÃD− 2 Z (l) W (l) + Z (l)
   774	(L)
   775	fenhanced = f + MLP(Z1:2
   776	)
   777	
   778	(28)
   779	
   780	where Ã = A + I is the adjacency matrix with self-loops, D is
   781	the degree matrix, and W (l) is the learnable weight at layer
   782	l. The residual connections alleviate over-smoothing while
   783	the multi-layer structure enhances long-distance information
   784	transfer.
   785	F. Camera-Aware Consistency Constraint
   786	1) Camera View Bias Problem: Camera view differences
   787	present a significant challenge in person re-identification.
   788	Images of the same identity from different cameras show variations in viewpoint, lighting, and background, while different
   789	identities from the same camera may appear similar due to
   790	shared backgrounds [52]. This bias often leads to incorrectly
   791	dividing same identities across cameras or clustering different
   792	identities within the same camera [53].
   793	Existing methods either ignore camera information or simply incorporate camera ID as an auxiliary feature, failing to
   794	utilize intra-camera sample structure consistency to effectively
   795	reduce background bias.
   796	2) Camera-Aware Clustering Strategy: We propose a
   797	camera-aware clustering method that utilizes camera-specific
   798	information to improve pseudo-label quality. DBSCAN clustering is performed independently on samples under each
   799	camera:
   800	Ccam = DBSCAN(Xcam , ε = 0.6, min samples = 4)
   801	
   802	(29)
   803	
   804	where Xcam is the sample set under camera cam, ε = 0.6 is
   805	the neighborhood radius parameter, and min samples = 4
   806	
   807	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   808	
   809	212
   810	
   811	IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. 36, NO. 1, JANUARY 2026
   812	
   813	defines the minimum number of samples required to form a
   814	cluster.
   815	Camera-specific clustering utilizes the consistency of
   816	images under the same camera, making it easier to identify true
   817	identity relationships without interference from cross-camera
   818	view changes. The choice of ε = 0.6 is based on extensive empirical validation across different camera views and
   819	datasets, providing optimal balance between cluster cohesion
   820	and separation in cross-modal scenarios while maintaining
   821	computational efficiency.
   822	3) Adaptive Consistency Constraint Strategy: Our key
   823	innovation lies in the adaptive post-processing strategy that
   824	intelligently combines global feature similarity with cameraspecific clustering results, rather than relying on parameter
   825	variation approaches.
   826	Based on global feature similarity and camera-specific clustering results, we construct two types of neighborhoods: global
   827	neighborhoods (Ng ) and camera-internal local neighborhoods
   828	(Nl ):
   829	Ng (i) = { j|sim( fi , f j ) > k · max(sim( fi , ∗))}
   830	
   831	(30)
   832	
   833	Nl (i) = { j|vcam
   834	= vcam
   835	j
   836	i , c j = ci }
   837	
   838	(31)
   839	
   840	where sim(·, ·) is the cosine similarity function, k is a threshold
   841	factor (set to 0.9), vcam
   842	represents camera-specific pseudoj
   843	labels, and c j is the camera ID. We propose an adaptive
   844	consistency constraint strategy that dynamically filters conflicting samples:
   845	8
   846	ˆ
   847	if c j , ci
   848	<Ng (i),
   849	N f inal (i) = Ng (i) ∩ Nl (i), if c j = ci , Ng ∩ Nl , ∅ (32)
   850	ˆ
   851	:
   852	Ng (i) \ S ,
   853	otherwise
   854	Here, S represents a set of potentially conflicting samples,
   855	determined by a probability-based filtering mechanism. Each
   856	potential conflict is retained with probability Pdecay (e) =
   857	p0 · (1 − e/Emax ), which is a decay probability based
   858	on training epoch e. Parameters p0 and Emax represent
   859	the initial probability (0.8) and maximum training epochs,
   860	respectively.
   861	This strategy maintains openness toward cross-camera relationships while applying strict verification within the same
   862	camera. For cross-camera relations (c j , ci ), we fully trust
   863	global similarity. For intra-camera samples with consistent
   864	clustering results, we require both global and local neighborhood agreement. In other cases, we filter potential conflicts
   865	through adaptive probability decay, allowing the model to

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf' - | nl -ba | sed -n '158,579p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   158	III. METHOD
   159	An overview of Vehicle-Diff is provided in Fig. 2. VehicleDiff generates high-fidelity data in a coarse-to-fine manner
   160	to enhance reID network training, comprising three stages:
   161	(1) prompt processing, (2) diffusion fine-tuning, and (3)
   162	semantic filtering. First, the prompt processing stage (§IIIA) constructs a prompt library and specifies vehicle attributes
   163	such as models and colors for image generation. Next, during
   164	the diffusion fine-tuning stage (§III-B), Vehicle-Diff finetunes the diffusion model using unlabeled vehicle images,
   165	improving its adaptation to vehicle image generation. Finally,
   166	in the semantic filtering stage (§III-C), Vehicle-Diff generates
   167	vehicle images with different IDs using the prompt library
   168	and fine-tuned model, followed by filtering these images
   169	through off-the-shelf detection and cross-modality alignment.
   170	A. Prompt Processing
   171	The prompt processing stage aims to construct discriminative vehicle attribute prompts to guide image generation, thus
   172	enhancing inter-class consistency and intra-class diversity.
   173	We first filter the noisy online information to collect vehicle
   174	attributes, i.e., brand, production year, and body style, for
   175	different car models from an online car information website 1 . It is worth noting that color is an important attribute,
   176	and we will use it again in the third stage for semantic
   177	filtering. Moreover, inspired by alternating optimization [38]
   178	and human-diffusion interaction [39], [40], [41], we also
   179	develop a prompt template to improve the quality of the
   180	generated images. Specifically, we adjusted one component
   181	of the prompt template based on feedback from the diffusion
   182	1 https://www.autoevolution.com/
   183	
   184	7320
   185	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.
   186	
   187	Select
   188	attribute
   189	keywords
   190	
   191	Diffusion
   192	Model
   193	
   194	Prompt: a green 2013
   195	Chevrolet Impala vehicle
   196	
   197	Diffusion
   198	Model
   199	
   200	poor cross-image consistency
   201	
   202	Filtering
   203	Vehicle model
   204	information
   205	
   206	Prompt: a green
   207	Chevrolet Impala vehicle
   208	
   209	Prompt Template Design:
   210	Iterate till the best prompt template is found
   211	
   212	Vehicle model
   213	attribute library
   214	
   215	good cross-image consistency
   216	
   217	Optimized
   218	prompt library
   219	
   220	Text
   221	embeddings
   222	
   223	Gradient update
   224	
   225	1% Unlabeled real images
   226	Text
   227	Encoder
   228	
   229	Captioned 1% Text-image Pairs
   230	
   231	a black car driving
   232	down the road
   233	
   234	Image
   235	Captioner
   236	
   237	LoRA
   238	Layers
   239	
   240	Diffusion Model
   241	
   242	……
   243	
   244	MSE
   245	Loss
   246	
   247	Apply noise × n
   248	Diffusion Model Selection
   249	
   250	Optimal
   251	Fine-Tuned
   252	Diffusion
   253	Model
   254	
   255	Noisy adapted images
   256	
   257	Correct Front Grills
   258	! $
   259	× ×48
   260	4 4
   261	
   262	Additional Weights
   263	Trainable Parameters
   264	
   265	Stage2: Diffusion Fine-Tuning
   266	Stage3: Semantic Filtering
   267	
   268	H×W×3
   269	Images
   270	
   271	Linear Embedding
   272	
   273	Stage1: Prompt Processing
   274	
   275	4
   276	
   277	×
   278	
   279	$
   280	4
   281	
   282	Stage 2
   283	
   284	Swin
   285	Transformer
   286	Block
   287	
   288	×2
   289	
   290	! $
   291	×
   292	×4&
   293	16 16
   294	
   295	! $
   296	× ×2 &
   297	8 8
   298	
   299	×&
   300	
   301	Stage 1
   302	
   303	Patch Partition
   304	
   305	Frozen Parameters
   306	
   307	!
   308	
   309	Filtered adapted images
   310	
   311	Swin
   312	Transformer
   313	Block
   314	
   315	×2
   316	
   317	! $
   318	×
   319	×8&
   320	32 32
   321	
   322	Stage 3
   323	
   324	Swin
   325	Transformer
   326	Block
   327	
   328	×6
   329	
   330	reID model
   331	
   332	Stage 4
   333	
   334	Swin
   335	Transformer
   336	Block
   337	
   338	×2
   339	
   340	Pri
   341	ma
   342	ry
   343	Cla
   344	ssif
   345	ier
   346	
   347	……
   348	
   349	Semantic
   350	Filtering
   351	
   352	ier
   353	
   354	Optimized
   355	prompt library
   356	
   357	Reference
   358	prompt
   359	
   360	Patch Merging
   361	
   362	Diffusion Text
   363	Encoder Frozen
   364	
   365	Au
   366	xilia
   367	ry C
   368	las
   369	sif
   370	
   371	Wrong Front Grills
   372	
   373	Patch Merging
   374	
   375	Reference
   376	prompt
   377	
   378	Patch Merging
   379	
   380	Diffusion Text
   381	Encoder Tuned
   382	
   383	Fig. 2: An overview of our coarse-to-fine cross-modality pipeline Vehicle-Diff. It has three stages: Prompt Processing,
   384	Diffusion Fine-tuning, and Semantic Filtering. (1) We first scrape and filter vehicle model information from online vehicle
   385	websites. Given the diffusion model, we then select the prompt template according to the visual quality. (2) In the second
   386	stage, we leverage the off-the-shelf image captioner to generate the pseudo caption. It is worth noting that the proposed
   387	pipeline only requests a few unlabeled real images from the downstream dataset. After the data preparation, we fine-tune
   388	the diffusion model via Mean Squared Error (MSE) loss. (3) In the third stage, using the refined prompts, we choose the
   389	most effective diffusion model by comparing visual quality, such as consistency. Then, we create synthetic data for the
   390	vehicle re-identification task. We use the cross-modality model to filter out semantically misaligned data. Finally, we feed
   391	the high-fidelity data to train the reID model via cross-entropy loss [35], [36] and circle loss [37].
   392	model. The final prompt template is designed as “a [color]
   393	[production year] [brand] [car model] [body style] driving
   394	down the road.” In the bottom of Fig. 1, we show several
   395	examples of the prompt template and the resulting images.
   396	B. Diffusion Fine-tuning
   397	Vehicle-Diff leverages a text-to-image diffusion model
   398	to generate vehicle images according to prompts. However, a pre-trained diffusion model still struggles to adapt
   399	well to the real-world vehicle images, resulting in a domain gap between synthesized images and those in vehicle
   400	reID datasets. Therefore, we further fine-tune the diffusion
   401	model to mitigate the domain discrepancy while retaining
   402	its generation capability. As shown in Fig. 2 (Stage 2), we
   403	illustrate the step-by-step fine-tuning stage from the data
   404	preparation to the model optimization. To be specific, we
   405	first deploy an image captioner, i.e., BLIP-2 [42], to predict
   406	text prompts for unlabeled vehicle images, and then employ
   407	
   408	the generated image-text pairs to fine-tune the text-to-image
   409	diffusion model. We incorporate additional weights [43] in
   410	the decoder part, while keeping the pre-trained weights
   411	unchanged. Therefore, the additional weights could adapt the
   412	final visual style, while maintaining the generative capability.
   413	The optimization objective is the mean squared error (MSE)
   414	loss. It is worth noting that, our Vehicle-Diff could be trained
   415	with only a few (1%) unlabeled images of the vehicle
   416	dataset for fine-tuning, i.e., 378 images for VeRi-776 and
   417	527 images for CityFlowV2, while previous methods either
   418	require large-scale datasets (GAN-based methods [10], [11])
   419	or rely on labeled images (graphics-engine-based methods
   420	[8], [9]). Moreover, different from these methods, VehicleDiff harnesses the generative power of diffusion models,
   421	enabling to generate more realistic images, as shown in
   422	Fig. 1. Similarly, we fine-tune multiple candidate diffusion
   423	models in preparation for the next stage, which involves
   424	
   425	7321
   426	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.
   427	
   428	selecting the optimal diffusion model.
   429	C. Semantic Filtering
   430	We first sample approximately 10 prompts from the optimized prompt library to evaluate and select the optimal finetuned diffusion model. With a similar idea to our prompt
   431	template design, the selection of the fine-tuned model is
   432	informed by a qualitative assessment of the images generated
   433	by each candidate model. Fig. 2 (Stage 3) provides illustrative examples of fine-tuned models evaluated alongside the
   434	corresponding generated imagery. Through this evaluation,
   435	we opt for the fine-tuned diffusion model that maintains the
   436	text encoder in a frozen state. We then feed our designed
   437	prompts into the optimal fine-tuned diffusion model, which
   438	generates synthetic images automatically. Because of the limitations of text-to-image generation models in producing finegrained and controllable outputs, directly using generated
   439	images is insufficient for training vehicle re-identification
   440	networks due to the following two major challenges, i.e.,
   441	multiple objects and semantic misalignment. We only need
   442	portions of the images that include the high-quality vehicle.
   443	Diffusion models can generate low-quality images, such as
   444	those with multiple vehicles, fragmented vehicles, or no
   445	vehicle at all. To tackle this issue, we utilize the YOLOv5x6
   446	detection model [44], trained on high-resolution 1280×1280
   447	images, for vehicle detection and cropping. The model is
   448	configured to detect only vehicle categories, with a single
   449	bounding box per image prioritizing the most prominent
   450	vehicle. We retain images with high-confidence detections
   451	and discard vehicles smaller than or equal to 250 pixels
   452	in height or width. After cropping, we have the vehicle in
   453	the center of the image, and we further screen out noisy
   454	images with semantic misalignment, such as vehicles with
   455	incorrect colors. In particular, we employ a cross-modal
   456	vision-language model, i.e., CLIP [45], to extract the feature
   457	for both text and image modalities. We then remove semantic
   458	misaligned images that match wrong colors. Specifically,
   459	the test prompts are constructed as phrases, e.g., “a red
   460	vehicle,” where the color term is dynamically substituted
   461	from a predefined color list, such as “red,” “yellow,” “green,”
   462	“white,” and “black.” The cosine similarity between image
   463	and test text in the feature level is:
   464	f I · f Tk
   465	simk =
   466	.
   467	(1)
   468	∥fI ∥∥fTk ∥
   469	The predicted color k̂ is identified as: k̂ = arg maxk (simk ).
   470	We then compare the predicted color to the expected color,
   471	which is specified within the prompt used to generate the
   472	image. If the predicted color matches the expected color, the
   473	image is preserved; otherwise, it is discarded.
   474	D. ReID Learning
   475	In this paper, we do not pursue the network structure,
   476	but focus on the data aspect. Our generated data is compatible with different networks, and we are free to the reID
   477	model selection. Here, we take the typical transformer, SwinV2 [46], as an example (please see the bottom of Fig. 2).
   478	
   479	Real
   480	
   481	Synthetic
   482	
   483	Dataset
   484	StanfordCars [50]
   485	PKU-Vehicle [51]
   486	CompCar [52]
   487	PKU-VD1 [53]
   488	PKU-VD2 [53]
   489	VehicleID [54]
   490	VehicleReID [55]
   491	VeRi-776 [56]
   492	CityFlow [57]
   493	CityFlowV2 [58]
   494	VRIC [59]
   495	PAMTRI [8]
   496	VehicleX [9]
   497	Vehicle-Diff
   498	
   499	#IDs
   500	196
   501	N/A
   502	4,701
   503	1,232
   504	1,112
   505	26,328
   506	N/A
   507	776
   508	666
   509	440
   510	5,622
   511	402
   512	1,362
   513	4,896
   514	
   515	#Img
   516	16,185
   517	10,000,000
   518	136,726
   519	1,097,649
   520	807,260
   521	222,629
   522	47,123
   523	49,357
   524	56,277
   525	52,717
   526	60,430
   527	41,000
   528	75,516†
   529	149,472‡
   530	
   531	#Cam
   532	N/A
   533	N/A
   534	N/A
   535	1
   536	1
   537	2
   538	2
   539	20
   540	40
   541	46
   542	60
   543	Varied
   544	Varied
   545	Varied
   546	
   547	Attr
   548	✓
   549	✗
   550	✗
   551	✓
   552	✓
   553	✗
   554	✗
   555	✓
   556	✗
   557	✗
   558	✗
   559	✓
   560	✓
   561	✓
   562	
   563	TABLE I: Statistic comparisons with public real-world and
   564	synthetic vehicle re-ID datasets in terms of the number of
   565	vehicle IDs, images, and viewpoints, and the availability of
   566	attributes. † : Number of images in their code. ‡ : Given more
   567	text prompts, we could generate more images.
   568	We follow the existing works [47], [48] to add an auxiliary
   569	classifier to facilitate the backward gradients, especially for
   570	the large-scale dataset. To optimize the network, we adopt
   571	the classification loss [35], [36] and the circle loss [37] as
   572	Ltotal = Lce + Lcircle , where Lce is the cross-entropy loss
   573	to classify different vehicles, and the Lcircle is to optimize
   574	the representation space by pulling closer positive images,
   575	while pushing away the negative samples. We apply the same
   576	loss terms to both the primary and auxiliary classifiers. It is
   577	worth noting that our synthetic data can be combined with
   578	real-world data to improve performance even further.
   579	IV. E XPERIMENT

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf' - | nl -ba | sed -n '297,642p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   297	3. Methodology
   298	In this section, we will introduce the details of the proposed ColorSketchNet. Firstly, we propose a color sketch generator (CSG) in Section 3.1, which consists of dynamic lighting modiﬁer and color sketch
   299	auxiliary generator, which generate color-rich and texture-detailed
   300	sketch and cope with complex lighting conditions. Secondly, to compensate for the missing attributes in sketch and text, and suppress the
   301	noise interference of modality-irrelevant information, we propose an attribute compensation and reﬁnement module (ACRM) to establish a fair
   302	attribute space that addresses the task imbalance problem in Section 3.2.
   303	Finally, an uniﬁed collaborative alignment (UCA) scheme is developed
   304	to adjust the latent distributions of the four modalities in Section 3.3.
   305	The overview of ColorSketchNet is shown in Fig. 2.
   306	
   307	x𝑝𝑔 = 0.299𝑅 + 0.587𝐺 + 0.114𝐵,
   308	
   309	(1)
   310	
   311	where 𝑅, 𝐺 and 𝐵 denote the red, green and blue channels of the RGB
   312	photo, respectively.
   313	After that, the average value of all pixels in the grayscale photo x𝑝𝑔
   314	is obtained by:
   315	
   316	3.1. Color sketch generator
   317	
   318	̄
   319	𝐿(x𝑝𝑔 ) =
   320	
   321	Existing studies (Chen et al., 2022; Liu et al., 2024a) often ignore
   322	the key factors of color attributes for generating black and white sketch,
   323	4
   324	
   325	𝐻 ∑
   326	𝑊
   327	∑
   328	1
   329	x𝑝 (ℎ, 𝑤),
   330	𝐻 × 𝑊 ℎ=1 𝑤=1 𝑔
   331	
   332	(2)
   333	
   334	Neural Networks 196 (2026) 108374
   335	
   336	M. Liu et al.
   337	
   338	where ̄
   339	𝐿(x𝑝𝑔 ) denotes the average brightness of x𝑝𝑔 ; x𝑝𝑔 (ℎ, 𝑤) represents
   340	the grayscale value at coordinates (ℎ, 𝑤). The range of ̄
   341	𝐿(x𝑝𝑔 ) is [0, 255],
   342	where lower values indicate darker images prone to losing important details, while higher values report brighter images that suﬀer from overexposure. Therefore,based on the dataset distribution and threshold analysis in Section 4.7, we set a threshold 𝑇 = 128 to distinguish darker images from brighter ones.
   343	Further, we exploit EnlightenGAN (Jiang et al., 2021) to enhance illumination and visibility in darker regions when ̄
   344	𝐿(x𝑝𝑔 ) < 𝑇 . Otherwise,
   345	we apply CLAHE (Reza, 2004) to alleviate brightness and overexposure
   346	̄ 𝑝𝑔 ) ≥ 𝑇 . Finally, the output of the dynamic light modiﬁer is
   347	when 𝐿(x
   348	obtained, denoted as the bright photo x𝑙 ∈ ℝ𝐶×𝐻×𝑊 , which can improve the quality of auxiliary color sketch modality for the input of color
   349	sketch auxiliary generator.
   350	Color Sketch Auxiliary Generator. We propose a color sketch auxiliary generator to generate sketch that retains the sketch style and
   351	identity-related color information for attribute compensation.
   352	Speciﬁcally, we ﬁrst utilize a 1 × 1 convolutional layer to highlight
   353	the structural information of the bright photo. Then, a Diﬀerence of
   354	Gaussian (DoG) ﬁlter is utilized to extract contour and edge information
   355	of x𝑙 , generating a grayscale sketch image x𝑑 .
   356	Further, we fuse the structural information of x𝑑 with the color information of x𝑙 in the YUV color space. To make this mapping explicit,
   357	we adopt the standard linear RGB to YUV transformation:
   358	⎡ 𝑌 ⎤ ⎡ 0.299
   359	⎢𝑈 ⎥ = ⎢-0.147
   360	⎢ ⎥ ⎢
   361	⎣𝑉 ⎦ ⎣ 0.615
   362	
   363	0.587
   364	-0.289
   365	-0.515
   366	
   367	0.114 ⎤⎡𝑅⎤
   368	0.436 ⎥⎢𝐺⎥.
   369	⎥⎢ ⎥
   370	-0.100⎦⎣𝐵 ⎦
   371	
   372	3.2. Attribute compensation and reﬁned module
   373	To ensure attribute fairness between diﬀerent modalities, we propose a attribute compensation and reﬁned module (ACRM) to adaptively
   374	compensate for the diﬀerences between modalities, which doesn’t require designing speciﬁc feature compensation branches for each modality. For example, in the Text-RGB task, the text can be compensated
   375	for missing color distribution and edge features from the auxiliary color
   376	sketch; in the Sketch-RGB task, the color and texture details of the auxiliary color sketch can be utilized to compensate for the blalck and white
   377	sketch; in the Text and Sketch-RGB task, the color sketch can make up
   378	for the missing speciﬁc attributes.
   379	In this work, our ACRM module also adds a modality-invariant ﬁltering strategy to help weaken the impact of potential noise introduced
   380	by the auxiliary color sketch modality and enhance the coherence and
   381	robustness of cross-modal representations.
   382	Paired modality task. Paired modality refers to Text-RGB task and
   383	Sketch-RGB task. The principles of these two tasks are the same. For clarity, we take the Sketch-RGB task as an example to explain the retrieval
   384	process.
   385	Speciﬁcally, we ﬁrst compute the attention distribution of the two
   386	modalities to obtain the similarity between the auxiliary features f𝑎 and
   387	the sketch features f𝑠 , which is deﬁned by:
   388	Q𝑎 K 𝑠
   389	𝛼𝑎→𝑠 = 𝑠𝑜𝑓 𝑡𝑚𝑎𝑥( √
   390	),
   391	𝑑𝑘
   392	
   393	(9)
   394	
   395	where Q𝑎 and K 𝑠 denote linear maps of f𝑎 and f𝑠 ; 𝑑𝑘 represents the
   396	channel dimension of K 𝑠 ; 𝛼𝑎→𝑠 is the similarity between f𝑎 and f𝑠 .
   397	However, sketch inherently lack color and texture attributes. To
   398	overcome this issue, we compute the dissimilarity between sketch and
   399	color sketch, denoted as 1 − 𝛼𝑎→𝑠 . Further, the missing information ḟ 𝑠
   400	
   401	(3)
   402	
   403	Speciﬁcally, the contours of the grayscale sketch image are mapped
   404	to the luminance channel 𝑌 , while the chrominance channels 𝑈 and 𝑉
   405	are encoded from x𝑙 . Accordingly, the channel-wise transformations can
   406	be written as:
   407	
   408	of the sketch modality is formulated as:
   409	ḟ 𝑠 = F ((1 − 𝛼𝑎→𝑠 ) × V 𝑎 ),
   410	
   411	(10)
   412	
   413	𝑌 (x ) = 0.299𝑅(x ) + 0.587𝐺(x ) + 0.114𝐵(x ),
   414	
   415	(4)
   416	
   417	where V 𝑎 is the mapping of the color sketch feature f𝑎 ; F (⋅) consists of
   418	
   419	𝑈 (x𝑙 ) = −0.147𝑅(x𝑙 ) − 0.289𝐺(x𝑙 ) + 0.436𝐵(x𝑙 ),
   420	
   421	(5)
   422	
   423	𝑉 (x𝑙 ) = 0.615𝑅(x𝑙 ) − 0.515𝐺(x𝑙 ) − 0.100𝐵(x𝑙 ).
   424	
   425	(6)
   426	
   427	linear layers that transform the missing features into the feature space
   428	to enhance the black and white sketch features.
   429	It is worth noting that the structure of sketch features will be disrupted if the missing features are directly added to the black and white
   430	sketch features. To address this, we introduce a weight factor g and
   431	a normalization term b. Speciﬁcally, g serves as a channel-wise gating mechanism, adaptively regulating the contribution of compensation features so that channels correlated with the missing attributes
   432	(e.g., color or texture) are enhanced while irrelevant channels are suppressed. Meanwhile, b acts as a residual normalization term: the constant 1 guarantees preservation of original sketch features, while the additional adaptive bias ensures numerical stability and balanced feature
   433	magnitudes across channels. This design enables ACRM to selectively
   434	incorporate complementary information while maintaining robustness
   435	across diﬀerent modality pairs. g and b are written by:
   436	
   437	𝑑
   438	
   439	𝑑
   440	
   441	𝑑
   442	
   443	𝑑
   444	
   445	Finally, the auxiliary color sketch x𝑎 is generated by combining these
   446	channels:
   447	
   448	(
   449	)
   450	x𝑎 = 𝐶𝑜𝑚𝑏𝑖𝑛𝑒𝑟 𝑌 (x𝑑 ), 𝑈 (x𝑙 ), 𝑉 (x𝑙 ) .
   451	
   452	(7)
   453	
   454	We choose YUV over HSV because YUV explicitly decouples luminance Y from chrominance U, V, enabling a clean assignment of structural contours to the Y channel and color information to the U and V
   455	channels. This separation stabilizes fusion, maintaining edge sharpness
   456	and global brightness consistency. By comparison, HSV is less reliable
   457	for our purpose: the value channel is the per-pixel maximum of RGB and
   458	thus prone to outliers, while hue is an angular, discontinuous quantity
   459	that can introduce instability near boundaries. Consequently, YUV provides a more robust and linear basis for integrating contour and color,
   460	consistent with long-standing video coding practice (Charles & et al,
   461	2003) and corroborated by recent vision studies (Brateanu et al., 2025;
   462	Prativadibhayankaram et al., 2024).
   463	Subsequently, the global features of photo x𝑝 , sketch x𝑠 , auxiliary
   464	color sketch x𝑎 and text x𝑡 are fed into the four-stream CLIP network to
   465	obtain the corresponding features f𝑝 , f𝑠 , f𝑎 and f𝑡 , respectively.
   466	To make the generated auxiliary color sketch consistent with identity information of the original photo, an identity preservation learning
   467	strategy 𝑖𝑝 is introduced as follows:
   468	𝑖𝑝 = −
   469	
   470	𝑐
   471	∑
   472	
   473	𝑦𝑖 𝑙𝑜𝑔(f𝑎𝑖 ),
   474	
   475	g = 𝜙(F (ḟ 𝑠 )),
   476	
   477	(11)
   478	
   479	b = 1 + 𝜙(F (ḟ 𝑠 )),
   480	
   481	(12)
   482	
   483	where 𝜙(⋅) is the sigmoid activation function; F (⋅) denotes a series of
   484	linear layers.
   485	After that, the features of each channel in sketch feature f𝑠 are normalized to 𝜎𝑠 and then combined with g and b. So, the compensated
   486	feature f̃𝑠 is obtained by Eq. (13).
   487	f̃𝑠 = 𝜎𝑠 ⊙ g + b.
   488	
   489	(13)
   490	
   491	However, the generated auxiliary color sketch modality may introduce elements that are inconsistent with the photo modality, such as
   492	unnatural textures, color distortions or geometric deformations. These
   493	noises can aﬀect the ability of the model to extract shared features. To
   494	address this issue, we further optimize compensated features f̃𝑠 to reduce the impact of modality-irrelevant information and focus on the
   495	shared information.
   496	
   497	(8)
   498	
   499	𝑖=0
   500	
   501	where 𝑐 denotes the number of identity; 𝑦𝑖 is the ground-truth label for
   502	the i-th identity; 𝑓𝑎𝑖 indicates the output of auxiliary color sketch features
   503	f𝑎 after pooling Layer, batch normalization and classiﬁcation layer.
   504	5
   505	
   506	Neural Networks 196 (2026) 108374
   507	
   508	M. Liu et al.
   509	
   510	To be speciﬁc, we compute the similarity 𝛼𝑝→𝑠 between the photo features f𝑝 and the compensated sketch features f̃𝑠 by referring to Eq. (9).
   511	The dissimilarity score 1 − 𝛼𝑝→𝑠 can reﬂect the degree of modality irrelevance. Further, referring to Eq. (10), the redundant noise information
   512	is extracted by multiplying V 𝑝 by 1 − 𝛼𝑝→𝑠 . In addition, redundant noise
   513	is subtracted from the compensated sketch features f̃𝑠 . The ﬁnal compensated sketch features f̂𝑠 can be obtained by:
   514	f̂𝑠 = f̃𝑠 − F ((1 − 𝛼𝑝→𝑠 ) × V 𝑝 ).
   515	
   516	identities. It is deﬁned as:
   517	
   518	This optimization process can adaptively adjust the importance of
   519	the sketch features and enhance the robustness of the model, which suppresses noise from the auxiliary color sketch modality.
   520	Similar to above-mentioned analysis, for Text-RGB task, the ﬁnal
   521	compensated text features f̂𝑡 can be also obtained by:
   522	(15)
   523	
   524	𝑐𝑚 = −
   525	
   526	Tri-modality task. This task includes text, sketch and RGB modalities, where text and sketch are utilized together as query modalities
   527	to retrieve RGB images. The proposed ACRM can be extended to trimodality retrieval scenario.
   528	Sketch and text have the advantage of complementing each other.
   529	Sketch can directly represent geometric information, such as contours
   530	and shapes, which is diﬃcult for text to describe, while text can provide
   531	the high-level semantic information for sketch. Therefore, compared to
   532	the paired modality retrieval task, tri-modality retrieval task oﬀers a
   533	richer and more comprehensive pedestrian description. Nevertheless,
   534	there are redundant features between text and sketch, and they lack important visual attributes (such as color distribution) which cannot be
   535	described by text and sketch. Fortunately, the proposed ACRM can still
   536	eﬃciently compensate for missing features with the help of auxiliary
   537	color sketch modality, alleviating the negative impact of redundant features in both the text and sketch modalities.
   538	Speciﬁcally, text feature and sketch feature are fused by using a simple summation operation, denoted as the fused features f𝑓 . Similarly
   539	to obtaining f̂𝑠 and f̂𝑡 , the ﬁnal compensated features f̂𝑓 that simultaneously compensate for text and sketch features can be obtained by
   540	referring to Eq. (9) to Eq. (14).
   541	In addition, we develop a content consistency loss 𝑐𝑐𝑙 to narrow the
   542	distance between the ﬁnal compensated features f̂𝑚 (𝑚 = 𝑠, 𝑡, 𝑓 ) and the
   543	photo features f𝑝 , which is deﬁned as:
   544	1 ∑𝛽
   545	𝑐𝑐𝑙 =
   546	(
   547	𝑁 𝑖=1 2
   548	𝑁
   549	
   550	√
   551	1+(
   552	
   553	f̂𝑚 − f𝑝
   554	𝛽
   555	
   556	)2 − 1), 𝑚 ∈ {𝑠, 𝑡, 𝑓 },
   557	
   558	(17)
   559	
   560	where f𝑝 𝑖 and f̂𝑚𝑖 denote the photo features and the compensated features from modality 𝑚 of the 𝑖-th identity, respectively. 𝐶𝑚 is the classiﬁer corresponding to the m-th modality; y𝑖 is the i-th identity label; 𝑁
   561	indicates the number of batches.
   562	Further, we propose a tri-directional constraint loss 𝑐𝑚 to enhance
   563	structural alignment across modalities, which takes the auxiliary color
   564	sketch as a semantic anchor. The auxiliary color sketch provides rich
   565	structural and color-invariant information, making it a suitable bridge
   566	to align features of diﬀerent modalities from the same identity. Inspired
   567	by Wei et al. (2021), it is formulated as:
   568	
   569	(14)
   570	
   571	f̂𝑡 = f̃𝑡 − F ((1 − 𝛼𝑝→𝑡 ) × V 𝑝 ).
   572	
   573	1 ∑
   574	y 𝑙𝑜𝑔(𝐶𝑚 (f𝑝 𝑖 , f̂𝑚𝑖 )), 𝑚 ∈ {𝑠, 𝑡, 𝑓 },
   575	𝑁 𝑖=1 𝑖
   576	𝑁
   577	
   578	𝑖𝑑𝑚 = −
   579	
   580	𝑁
   581	𝑒𝑥𝑝(||f̂𝑚𝑖 − f𝑎 𝑗 ||2 + ||f𝑝 𝑖 − f𝑎 𝑗 ||2 )
   582	∑
   583	𝑙𝑜𝑔 ∑𝑁
   584	,
   585	∑
   586	𝑗
   587	𝑖
   588	( 𝑗=1 𝑒𝑥𝑝(||f̂𝑚𝑖 − f𝑎 𝑗 ||2 )) ⋅ ( 𝑁
   589	𝑖=1
   590	𝑗=1 𝑒𝑥𝑝(||f𝑝 − f𝑎 ||2 ))
   591	
   592	𝑚 ∈ {𝑠, 𝑡, 𝑓 },
   593	
   594	(18)
   595	𝑗
   596	
   597	where f̂𝑚𝑖 denotes the i-th identity features of f̂𝑚 ; f𝑎 denotes the auxiliary color sketch feature for the j-th identity; || ⋅ ||2 represents the Euclidean 2-norm. This loss function simultaneously minimizes the distance between the sketch anchor and both the photo and other modalities’ features, thereby encouraging tighter inter-modality alignment for
   598	the same identity.
   599	Nevertheless, despite the remarkable progress in cross-modal representation learning, intra-modality feature distributions remain dispersed
   600	due to variations in viewpoint, illumination, and partial occlusion. This
   601	dispersion compromises the consistency and discriminability of features
   602	within the same modality. To alleviate this issue, we introduce an intraclass compactness loss 𝑖𝑐𝑐 , which is inspired by the intuition of clustering: samples belonging to the same identity should form a compact
   603	cluster around their class center. Unlike unsupervised clustering methods (Yang et al., 2024) that infer cluster assignments without labels, our
   604	formulation leverages identity annotations to explicitly compute class
   605	centers and penalize intra-class variance. The loss is deﬁned as:
   606	𝑖𝑐𝑐 =
   607	
   608	𝑁𝑐
   609	𝐶
   610	1 ∑ 1 ∑ 𝑖
   611	‖𝐟 − 𝜇𝑐 ‖2
   612	𝐶 𝑐=1 𝑁𝑐 𝑖=1 𝑚
   613	
   614	(19)
   615	
   616	where 𝜇𝑐 denotes the mean feature vector of class c in modality m, 𝑓𝑚𝑖
   617	represents the feature of the i-th sample in the m-th modality; 𝐶 is the
   618	total number of classes; 𝑁𝑐 is the number of samples in the c-th class.
   619	To summarize, the overall objective function loss of our method can
   620	be summarized as:
   621	
   622	(16)
   623	
   624	where 𝛽 is used to adjust the scale of the smoothing loss function; 𝑁 is
   625	the total number of samples.
   626	In general, the proposed ACRM is able to be utilized to compensate
   627	for missing features in both paired modality retrieval task (i.e., TextRGB, Sketch-RGB) and Tri-modality retrieval task (i.e., Text and SketchRGB), verifying its compatibility and generalization 𝑚𝑚 .
   628	
   629	𝑈 𝐶𝐴 = 𝑖𝑝 + 𝑐𝑐𝑙 + 𝜆1 𝑐𝑚 + 𝜆2 𝑖𝑑𝑚 + 𝜆3 𝑖𝑐𝑐 ,
   630	
   631	(20)
   632	
   633	During inference, the color sketch generator (CSG), the attribute
   634	compensation and reﬁnement module (ACRM), and all auxiliary supervision branches are removed. These components are used exclusively
   635	during training to improve representation learning and do not participate in the forward pass at test time. The ﬁnal feature extractor is selfcontained and operates directly on raw sketch or text queries to produce
   636	embeddings, which are then compared via similarity computation in the
   637	shared embedding space. This design ensures a fully feed-forward, eﬃcient, and deployment-friendly retrieval pipeline.
   638	
   639	3.3. Uniﬁed collaborative alignment learning scheme
   640	Our framework designs a uniﬁed collaborative alignment learning
   641	scheme that systematically coordinates cross-modal discrimination, semantic structure alignment, and intra-modal compactness. This collaborative design builds a modality-invariant feature space while preserving
   642	discriminative identity features. The scheme integrates three complementary loss functions: identity matching loss 𝑖𝑑𝑚 , cross-modal structure regularization loss 𝑐𝑚 , and intra-modal class constraint loss 𝑖𝑐𝑐 .

codex
六篇的方法正文已经读完。最后还要看每篇用什么消融和可视化把“问题重定义”闭环，而不是只看最终表格涨点；我会抽相关实验分析段。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'CLIP-powered modality centering with spiral training for visible-infrared person re-identification.pdf' - | nl -ba | sed -n '1460,1965p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1460	
  1461	MCST
  1462	
  1463	−
  1464	
  1465	55.2
  1466	
  1467	83.4
  1468	
  1469	89.4
  1470	
  1471	62.5
  1472	
  1473	58.3
  1474	
  1475	62.3
  1476	
  1477	90.6
  1478	
  1479	95.0
  1480	
  1481	68.3
  1482	
  1483	57.1
  1484	
  1485	Table 6
  1486	Analysis of the eﬀectiveness of separable text description (STD) based on CMC (%) and mAP (%) performance on the
  1487	CMG-P dataset, where ‘B’ represents the baseline method, ‘MCL’ refers to modality centering losses, and ‘ST’ denotes
  1488	spiral training.
  1489	Infrared-Visible
  1490	
  1491	Visible-Infrared
  1492	
  1493	Method
  1494	
  1495	STD
  1496	
  1497	R-1
  1498	
  1499	R-5
  1500	
  1501	R-10
  1502	
  1503	R-20
  1504	
  1505	mAP
  1506	
  1507	R-1
  1508	
  1509	R-5
  1510	
  1511	R-10
  1512	
  1513	R-20
  1514	
  1515	mAP
  1516	
  1517	B
  1518	
  1519	×
  1520	✓
  1521	
  1522	63.7
  1523	64.7
  1524	
  1525	81.5
  1526	82.2
  1527	
  1528	87.6
  1529	87.6
  1530	
  1531	92.0
  1532	91.6
  1533	
  1534	51.1
  1535	51.2
  1536	
  1537	63.4
  1538	64.1
  1539	
  1540	81.8
  1541	83.0
  1542	
  1543	87.5
  1544	87.9
  1545	
  1546	91.7
  1547	92.0
  1548	
  1549	51.5
  1550	51.4
  1551	
  1552	B+MCL
  1553	
  1554	×
  1555	✓
  1556	
  1557	65.3
  1558	64.8
  1559	
  1560	82.4
  1561	81.8
  1562	
  1563	87.7
  1564	87.6
  1565	
  1566	92.2
  1567	92.2
  1568	
  1569	52.0
  1570	51.0
  1571	
  1572	64.4
  1573	64.3
  1574	
  1575	82.3
  1576	82.0
  1577	
  1578	87.9
  1579	87.1
  1580	
  1581	92.1
  1582	91.6
  1583	
  1584	52.3
  1585	49.8
  1586	
  1587	B+ST
  1588	
  1589	×
  1590	✓
  1591	
  1592	64.3
  1593	64.5
  1594	
  1595	82.0
  1596	82.1
  1597	
  1598	87.4
  1599	88.2
  1600	
  1601	91.8
  1602	92.5
  1603	
  1604	51.8
  1605	52.1
  1606	
  1607	63.9
  1608	64.6
  1609	
  1610	82.3
  1611	83.0
  1612	
  1613	87.8
  1614	88.0
  1615	
  1616	92.2
  1617	91.8
  1618	
  1619	52.3
  1620	52.4
  1621	
  1622	B+MCL+ST
  1623	
  1624	×
  1625	✓
  1626	
  1627	64.8
  1628	66.4
  1629	
  1630	82.2
  1631	83.3
  1632	
  1633	87.9
  1634	88.6
  1635	
  1636	92.6
  1637	92.8
  1638	
  1639	52.4
  1640	52.2
  1641	
  1642	64.9
  1643	65.5
  1644	
  1645	82.6
  1646	83.2
  1647	
  1648	87.9
  1649	88.5
  1650	
  1651	91.9
  1652	92.3
  1653	
  1654	52.7
  1655	52.4
  1656	
  1657	Table 7
  1658	Analysis of the eﬀectiveness of modality centering losses based on CMC (%) and
  1659	mAP (%) performance on the CMG-P dataset, where ‘T2T’ denotes text-to-text
  1660	centering loss and ‘I2T’ denotes image-to-text centering loss.
  1661	Method
  1662	
  1663	Infrared-Visible
  1664	
  1665	T2T
  1666	
  1667	I2T
  1668	
  1669	R-1
  1670	
  1671	R-5
  1672	
  1673	R-10
  1674	
  1675	R-20
  1676	
  1677	mAP
  1678	
  1679	R-1
  1680	
  1681	R-5
  1682	
  1683	R-10
  1684	
  1685	R-20
  1686	
  1687	mAP
  1688	
  1689	×
  1690	✓
  1691	×
  1692	
  1693	×
  1694	×
  1695	✓
  1696	
  1697	64.5
  1698	66.0
  1699	66.6
  1700	
  1701	82.1
  1702	82.8
  1703	82.6
  1704	
  1705	88.2
  1706	88.4
  1707	87.9
  1708	
  1709	92.5
  1710	92.8
  1711	92.3
  1712	
  1713	52.1
  1714	52.3
  1715	50.6
  1716	
  1717	64.6
  1718	64.5
  1719	64.8
  1720	
  1721	83.0
  1722	82.7
  1723	82.1
  1724	
  1725	88.0
  1726	88.4
  1727	87.3
  1728	
  1729	91.8
  1730	92.2
  1731	91.7
  1732	
  1733	52.4
  1734	52.7
  1735	50.7
  1736	
  1737	✓
  1738	
  1739	✓
  1740	
  1741	66.4
  1742	
  1743	83.3
  1744	
  1745	88.6
  1746	
  1747	92.8
  1748	
  1749	52.2
  1750	
  1751	65.5
  1752	
  1753	83.2
  1754	
  1755	88.5
  1756	
  1757	92.3
  1758	
  1759	52.4
  1760	
  1761	randomly selected to form the gallery set for evaluation. This process is
  1762	repeated 10 times, and the average performance is reported.
  1763	5.1.3. SYSU-MM01
  1764	SYSU-MM01 is captured using 4 visible and 2 infrared cameras. The
  1765	training set consists of 22,258 visible and 11,909 near-infrared images
  1766	from 395 identities. The query and gallery sets include 3803 infrared
  1767	images and 301 (3,010) randomly selected visible images from 96 identities for single-shot (multi-shot) evaluation. The dataset supports two
  1768	testing modes: all-search, where the gallery includes images from all visible cameras, and indoor-search, where the gallery only contains images
  1769	from indoor visible cameras.
  1770	
  1771	Visible-Infrared
  1772	
  1773	clothing/style changes, and scale variations, making it particularly valuable for advancing V-I ReID research in real-world contexts.
  1774	Regarding the evaluation protocol for CMG-P, the training set consists of 26,979 visible and 27,058 near-infrared images from 758 identities, while the test set includes 9052 visible and 9086 near-infrared
  1775	images from 253 identities. Both V-I (Visible-to-Infrared) and I-V
  1776	(Infrared-to-Visible) modes are used to assess the performance of the V-I
  1777	ReID models. During testing, for each camera, one image per identity is
  1778	
  1779	5.1.4. RegDB
  1780	RegDB is captured using one visible and one thermal camera, and
  1781	consists of 8240 images from 412 identities, with each identity having
  1782	10 images from the visible camera and 10 from the thermal camera.
  1783	The dataset is randomly split into two halves, with 206 identities used
  1784	for training and the remaining 206 identities for testing. RegDB also
  1785	supports two test modes: the V-I mode, which retrieves infrared images
  1786	7
  1787	
  1788	Pattern Recognition 177 (2026) 113333
  1789	
  1790	J. Xiong et al.
  1791	
  1792	of 713 identities with 16,946 visible and 13,975 infrared images, and
  1793	a testing set of 351 identities with 8680 visible-light and 7166 infrared
  1794	images. For evaluation purposes, LLCM supports both visible-to-infrared
  1795	and infrared-to-visible testing modes.
  1796	5.2. Implementation details
  1797	We implement our method using PyTorch and train it on an NVIDIA
  1798	GTX3090 GPU. A two-stream ResNet-50 [29] pre-trained on ImageNet
  1799	[36] is used as the image encoder. The ﬁrst three stages of ResNet-50
  1800	in the two branches have separate weights, followed by two stages that
  1801	share weights. The batch size is set to 64, with 8 identities randomly selected per batch, and 4 images per identity for each modality. Each image is resized to 256×128. During the text token training stage, we use
  1802	the original images without data augmentation. The hyper-parameters
  1803	𝑀 and 𝐿 are set to 4. For the image encoder training, data augmentation
  1804	techniques such as random cropping with zero-padding, random horizontal ﬂipping, random channel exchange, and random channel erasing
  1805	[13] are applied.
  1806	Since no theoretical framework exists to guide the selection of optimal training sequences for diverse datasets, we determine these sequences empirically. For CMG-P, we use the training stage sequence
  1807	{ 1 1 2 2 3}
  1808	𝑠𝑖𝑒 , 𝑠𝑡𝑡 , 𝑠𝑖𝑒 , 𝑠𝑡𝑡 , 𝑠𝑖𝑒 . The conﬁgurations of 𝑠1𝑡𝑡 and 𝑠2𝑡𝑡 are identical, and both
  1809	are optimized using the Adam optimizer with a weight decay of 1 × 10−4 .
  1810	The initial learning rate is set to 3 × 10−3 , and it is decayed following a
  1811	cosine schedule. The training duration for each stage is 10 epochs. The
  1812	hyper-parameter 𝜆1 is set to 0.05, and the margin 𝛼 is 0.8. All image
  1813	encoders are optimized using the Adam optimizer with a weight decay
  1814	of 5 × 10−4 , and the initial learning rate is set to 3 × 10−4 . The margin 𝛽
  1815	is 0.7. For 𝑠1𝑖𝑒 , the training epoch is 10, and the hyper-parameter list for
  1816	𝜆2 − 𝜆5 is [0.15, 0, 0, 0]. For 𝑠2𝑖𝑒 and 𝑠3𝑖𝑒 , the hyper-parameter list for 𝜆2 − 𝜆5 is
  1817	[0.15, 0.05, 0.1, 0.1]. The former is trained for 40 epochs, while the latter is
  1818	trained for 100 epochs, with learning rate decays of 0.1 at the 40th and
  1819	70th epochs.
  1820	For SYSU-MM01 and LLCM, we use the training stage sequence
  1821	{ 1 1 2} 1
  1822	𝑠𝑖𝑒 , 𝑠𝑡𝑡 , 𝑠𝑖𝑒 . 𝑠𝑡𝑡 is optimized using the Adam optimizer with a weight decay of 1 × 10−4 . The initial learning rate is set to 3 × 10−4 , and it is decayed following a cosine schedule. The training duration is 20 epochs.
  1823	The hyper-parameter 𝜆1 is set to 0.02, and the margin 𝛼 is 0.8. All image encoders are optimized using the Adam optimizer with a weight
  1824	decay of 5 × 10−4 , and the initial learning rate is set to 3 × 10−4 . The margin 𝛽 is 0.7. For 𝑠1𝑖𝑒 , the training epoch is 10, and the hyper-parameter
  1825	list for 𝜆2 − 𝜆5 is [0.15, 0, 0, 0]. For 𝑠2𝑖𝑒 , the hyper-parameter list for 𝜆2 − 𝜆5 is
  1826	[0.15, 0.05, 0.1, 0.01]. 𝑠2𝑖𝑒 is trained for 120 epochs, with learning rate decays
  1827	of 0.1 at the 40th and 70th epochs.
  1828	{
  1829	}
  1830	For RegDB, we use the training stage sequence 𝑠1𝑖𝑒 , 𝑠1𝑡𝑡 , 𝑠2𝑖𝑒 , 𝑠2𝑡𝑡 , 𝑠3𝑖𝑒 . The
  1831	−4
  1832	initial learning rate of all stages is set to 3 × 10 , and the training duration for 𝑠1𝑡𝑡 and 𝑠2𝑡𝑡 is 20 epochs each. The hyper-parameter 𝜆1 is set to
  1833	0.01. All other conﬁgurations are the same as those used for CMG-P.
  1834	
  1835	Fig. 5. Image examples from CMG-P, SYSU-MM01, RegDB and LLCM are provided to highlight the diﬀerences in dataset characteristics. CMG-P features
  1836	richer and more realistic scenes, including variations in clothing and occlusions.
  1837	Table 8
  1838	Analysis of the eﬀectiveness of the spiral training (ST) strategy based on CMC
  1839	(%) and mAP (%) performance on the CMG-P dataset, where ‘B’ represents the
  1840	baseline method, ‘STD’ denotes separable text description, and‘MCL’ refers to
  1841	modality centering losses.
  1842	Method
  1843	
  1844	ST
  1845	
  1846	Infrared-Visible
  1847	R-1
  1848	
  1849	R-5
  1850	
  1851	Visible-Infrared
  1852	
  1853	R-10 R-20 mAP R-1
  1854	
  1855	R-5
  1856	
  1857	R-10 R-20 mAP
  1858	
  1859	B
  1860	
  1861	×
  1862	✓
  1863	
  1864	63.7 81.5 87.6
  1865	65.4 82.0 87.9
  1866	
  1867	92.0
  1868	92.1
  1869	
  1870	51.1
  1871	52.1
  1872	
  1873	63.4 81.8 87.5
  1874	63.9 82.3 87.8
  1875	
  1876	91.7
  1877	92.2
  1878	
  1879	51.5
  1880	52.3
  1881	
  1882	B+STD
  1883	
  1884	×
  1885	✓
  1886	
  1887	64.7 82.2 87.6
  1888	64.5 82.2 88.2
  1889	
  1890	91.6
  1891	92.5
  1892	
  1893	51.2
  1894	52.2
  1895	
  1896	64.1 83.0 87.9
  1897	64.6 83.0 88.0
  1898	
  1899	92.0
  1900	91.8
  1901	
  1902	51.4
  1903	52.4
  1904	
  1905	B+MCL
  1906	
  1907	×
  1908	✓
  1909	
  1910	65.3 82.3 87.7
  1911	64.8 82.2 87.9
  1912	
  1913	92.2
  1914	92.6
  1915	
  1916	52.0
  1917	52.4
  1918	
  1919	64.4 82.3 87.9
  1920	64.9 82.6 87.9
  1921	
  1922	92.1
  1923	91.9
  1924	
  1925	52.3
  1926	52.7
  1927	
  1928	B+STD+MCL
  1929	
  1930	×
  1931	✓
  1932	
  1933	64.8 81.8 87.6
  1934	66.4 83.3 88.6
  1935	
  1936	92.2
  1937	92.8
  1938	
  1939	51.0
  1940	52.2
  1941	
  1942	64.3 82.0 87.1
  1943	65.5 83.2 88.5
  1944	
  1945	91.6
  1946	92.3
  1947	
  1948	49.8
  1949	52.4
  1950	
  1951	Table 9
  1952	Analysis of the ﬁrst-stage selection based on CMC (%) and mAP (%) performance
  1953	on the CMG-P dataset.
  1954	Epoch of Stage
  1955	
  1956	Infrared-Visible
  1957	
  1958	Epoch of Stage
  1959	
  1960	Infrared-Visible
  1961	
  1962	𝑠𝑖𝑒
  1963	
  1964	𝑠𝑡𝑡
  1965	

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf' - | nl -ba | sed -n '918,1160p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   918	C. Ablation Studies
   919	To verify the effectiveness of each component in the CMAG
   920	framework and their synergistic effects, we designed a series
   921	of ablation experiments on SYSU-MM01 and RegDB datasets,
   922	evaluating the contributions of four core components: CATS
   923	
   924	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 09:06:18 UTC from IEEE Xplore. Restrictions apply.
   925	
   926	
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf' - | nl -ba | sed -n '1530,1965p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1530	S. Zhao et al.
  1531	
  1532	Table 3
  1533	An effectiveness analysis of different components is conducted on the SYSU-MM01 dataset.
  1534	Index
  1535	
  1536	Components
  1537	
  1538	SYSU-MM01
  1539	All search
  1540	
  1541	Baseline
  1542	1
  1543	2
  1544	3
  1545	4
  1546	5
  1547	6
  1548	7
  1549	8
  1550	9
  1551	10
  1552	11
  1553	12
  1554	13
  1555	
  1556	✓
  1557	✓
  1558	✓
  1559	✓
  1560	✓
  1561	✓
  1562	✓
  1563	✓
  1564	✓
  1565	✓
  1566	✓
  1567	✓
  1568	✓
  1569	
  1570	CPC
  1571	
  1572	NCL
  1573	
  1574	OTPM
  1575	
  1576	NMU
  1577	
  1578	✓
  1579	✓
  1580	
  1581	✓
  1582	✓
  1583	✓
  1584	
  1585	✓
  1586	
  1587	✓
  1588	✓
  1589	✓
  1590	✓
  1591	
  1592	✓
  1593	✓
  1594	✓
  1595	✓
  1596	✓
  1597	
  1598	✓
  1599	✓
  1600	✓
  1601	✓
  1602	
  1603	✓
  1604	✓
  1605	
  1606	✓
  1607	✓
  1608	✓
  1609	
  1610	✓
  1611	✓
  1612	✓
  1613	
  1614	Indoor search
  1615	
  1616	Times (s)
  1617	
  1618	R1
  1619	
  1620	mAP
  1621	
  1622	mINP
  1623	
  1624	R1
  1625	
  1626	mAP
  1627	
  1628	mINP
  1629	
  1630	61.54
  1631	62.34
  1632	62.46
  1633	63.25
  1634	63.91
  1635	64.85
  1636	65.65
  1637	67.85
  1638	67.73
  1639	65.33
  1640	68.43
  1641	68.11
  1642	69.49
  1643	
  1644	59.33
  1645	60.06
  1646	60.18
  1647	60.65
  1648	61.00
  1649	61.81
  1650	62.52
  1651	64.22
  1652	63.73
  1653	62.09
  1654	64.52
  1655	64.58
  1656	65.64
  1657	
  1658	45.54
  1659	46.19
  1660	46.41
  1661	46.37
  1662	46.78
  1663	47.48
  1664	48.13
  1665	49.68
  1666	48.85
  1667	47.69
  1668	49.85
  1669	50.25
  1670	51.20
  1671	
  1672	68.29
  1673	68.99
  1674	69.12
  1675	69.83
  1676	69.26
  1677	69.98
  1678	70.66
  1679	72.62
  1680	72.18
  1681	70.52
  1682	73.45
  1683	72.73
  1684	74.67
  1685	
  1686	74.03
  1687	74.47
  1688	74.50
  1689	75.10
  1690	74.48
  1691	74.86
  1692	75.74
  1693	77.09
  1694	76.63
  1695	75.74
  1696	77.64
  1697	77.40
  1698	78.95
  1699	
  1700	70.44
  1701	70.93
  1702	70.91
  1703	71.38
  1704	70.85
  1705	71.05
  1706	72.01
  1707	73.38
  1708	72.80
  1709	71.84
  1710	73.80
  1711	73.70
  1712	75.31
  1713	
  1714	7.016 × 103
  1715	7.089 × 103
  1716	7.242 × 103
  1717	7.266 × 103
  1718	7.063 × 103
  1719	7.348 × 103
  1720	7.243 × 103
  1721	7.007 × 103
  1722	6.986 × 103
  1723	7.262 × 103
  1724	7.228 × 103
  1725	7.137 × 103
  1726	7.232 × 103
  1727	
  1728	Fig. 3. Training dynamics on SYSU-MM01. Evolution of pseudo-label quality (ARI) and camera bias (Inter-camera feature distance) during training.
  1729	Table 4
  1730	Comparative analysis of cross-modality alignment strategies on the SYSUMM01 dataset. We evaluate the impact of the matching algorithm (PGM vs.
  1731	OTPM) and the alignment granularity (Prototype-only vs. Centroid-only).
  1732	
  1733	Rank-1, 2.69% in mAP, and 2.85% in mINP compared to the baseline.
  1734	These consistent gains across varied datasets validate that CLNS is
  1735	not merely overfitting to specific domains but provides a generalized solution for alleviating camera-induced label noise and enhancing
  1736	cross-modality matching.
  1737	
  1738	Methods
  1739	
  1740	4.4. Ablation study
  1741	
  1742	Stage1
  1743	CLNS w/ PGM
  1744	Only prototype
  1745	Only centroid
  1746	CLNS (Full)
  1747	
  1748	We conduct ablation studies on the SYSU-MM01 dataset to evaluate
  1749	the contribution of each component. The results are summarized in
  1750	Table 3, where ‘‘Baseline’’ denotes the model without our proposed
  1751	modules. Additionally, we assess computational efficiency. Notably, despite integrating four modules, CLNS incurs only marginal training time
  1752	overhead compared to the Baseline. Notably, all proposed auxiliary
  1753	modules are exclusively active during training to regularize the feature
  1754	space. At inference, only the bare ResNet-50 backbone is deployed for
  1755	retrieval. Consequently, CLNS introduces strictly zero additional computational overhead during testing. Given the substantial performance
  1756	leap, this confirms that CLNS achieves an optimal accuracy–efficiency
  1757	trade-off.
  1758	Effectiveness of the CPC module. CPC acts as a structural gatekeeper to mitigate camera-induced label noise. Quantitative comparisons in Table 3 (Index 3 vs. 4; Index 8 vs. 13) show consistent
  1759	performance gains upon integrating CPC. To visualize its impact, Fig. 3
  1760	tracks pseudo-label quality (ARI) and camera bias (mean inter-camera
  1761	feature distance). We observe a distinct inverse correlation: the significant reduction in camera bias directly parallels the rise in ARI
  1762	across both modalities. This confirms that CPC effectively rectifies
  1763	distribution shifts caused by camera views, preventing the model from
  1764	overfitting to specific domains. Notably, the improvement on RegDB
  1765	
  1766	All search
  1767	
  1768	Indoor
  1769	
  1770	R1
  1771	
  1772	mAP
  1773	
  1774	mINP
  1775	
  1776	R1
  1777	
  1778	mAP
  1779	
  1780	mINP
  1781	
  1782	51.9
  1783	68.43
  1784	68.11
  1785	68.04
  1786	69.49
  1787	
  1788	49.91
  1789	64.52
  1790	64.58
  1791	64.32
  1792	65.64
  1793	
  1794	33.97
  1795	49.85
  1796	50.25
  1797	49.88
  1798	51.2
  1799	
  1800	59.44
  1801	73.45
  1802	72.73
  1803	72.5
  1804	74.67
  1805	
  1806	65.41
  1807	77.64
  1808	77.4
  1809	77.12
  1810	78.95
  1811	
  1812	60.84
  1813	73.80
  1814	73.7
  1815	73.38
  1816	75.31
  1817	
  1818	is marginal because this dataset inherently possesses high clustering
  1819	quality with minimal camera discrepancy. This conversely validates
  1820	that CPC is critical for challenging scenarios (SYSU-MM01 and LLCM)
  1821	where camera-induced fragmentation is severe (see Fig. 4).
  1822	Effectiveness of the OTPM module. The OTPM module is pivotal
  1823	for bridging the semantic gap. We first evaluate the superiority of
  1824	the matching mechanism. Unlike PGM [9], which relies on greedy
  1825	bipartite matching prone to local optima, OTPM formulates alignment
  1826	as a global optimal transport problem. As shown in Table 4, replacing
  1827	OTPM with PGM (‘‘CLNS w/ PGM’’) results in a notable performance
  1828	drop (e.g., −1.06% Rank-1 and −1.12% mAP). This empirical evidence
  1829	confirms that the holistic matching plan of OTPM establishes more
  1830	robust and principled cross-modality associations. Furthermore, Table 4 decomposes the efficacy of OTPM’s dual-level strategy. ‘‘OTPM
  1831	(Prototype-only)’’ ensures broad identity consistency, while ‘‘OTPM
  1832	(Centroid-only)’’ explicitly bridges fine-grained structural gaps. The
  1833	superior performance of the full module indicates strong synergy: their
  1834	7
  1835	
  1836	Pattern Recognition 179 (2026) 113873
  1837	
  1838	S. Zhao et al.
  1839	
  1840	Fig. 4. Evolution of cross-modality alignment accuracy (ARI) during training on SYSU-MM01 dataset.
  1841	Table 5
  1842	Impact of CPC on the cross-modality matching accuracy of OTPM on SYSUMM01 dataset.
  1843	Methods
  1844	
  1845	w/o CPC
  1846	w/ CPC
  1847	
  1848	Table 6
  1849	Ablation study on the distinct roles of CPC (Label Quality) and NCL (Feature
  1850	Compactness) on SYSU-MM01 dataset.
  1851	
  1852	ARI
  1853	
  1854	Method variant
  1855	
  1856	VIS to IR
  1857	
  1858	IR to VIS
  1859	
  1860	0.748
  1861	0.761
  1862	
  1863	0.798
  1864	0.799
  1865	
  1866	CLNS w/o CPC
  1867	CLNS w/o NCL
  1868	CLNS (Full)
  1869	
  1870	joint optimization effectively handles both identity discrimination and
  1871	domain adaptation.
  1872	Effectiveness of the NCL module. While CPC corrects structural
  1873	errors, NCL addresses the loose intra-class distributions caused by
  1874	residual variance. Comparing Index 1 vs. 3 and Index 2 vs. 4 in
  1875	Table 3 reveals that NCL yields consistent improvements. This validates a coarse-to-fine synergy: CPC rectifies label orientation, while
  1876	NCL compacts the distribution. Without NCL, the learned representations, although structurally correct, lack the compactness required for
  1877	fine-grained matching.
  1878	Effectiveness of the NMU module. NMU serves as a critical stabilizer against memory bank drift. Removing NMU (Index 10 vs. 13)
  1879	causes a significant performance drop, indicating that standard momentum updates are highly vulnerable to accumulated outliers. By coupling
  1880	update rates with sample reliability, NMU maintains memory bank
  1881	purity throughout training. The robust performance in Index 11 and
  1882	12 further confirms NMU as a fundamental quality control mechanism,
  1883	safeguarding representation learning regardless of the specific module
  1884	combination.
  1885	
  1886	Label quality (ARI) ↑
  1887	
  1888	Feature compactness
  1889	
  1890	Visible
  1891	
  1892	Infrared
  1893	
  1894	Intra-class Dist ↓
  1895	
  1896	0.679
  1897	0.718
  1898	0.721
  1899	
  1900	0.896
  1901	0.903
  1902	0.907
  1903	
  1904	0.927
  1905	0.934
  1906	0.921
  1907	
  1908	distinct granularities. We analyze their roles using ARI (structural
  1909	correctness) and Intra-class Distance (feature compactness), as detailed
  1910	in Table 6. Comparing ‘‘CLNS w/o NCL’’ with the full model, removing
  1911	NCL increases intra-class distance (0.921 → 0.934). This implies that
  1912	while CPC ensures structural correctness, NCL is crucial for compacting
  1913	distributions via soft regularization. Conversely, removing CPC (‘‘CLNS
  1914	w/o CPC’’) causes a sharp drop in Visible ARI (0.721 → 0.679). Although NCL attempts to maintain compactness (0.927), performance is
  1915	bottlenecked by structural pseudo-label errors. The full model achieves
  1916	optimal metrics on both fronts, validating a coarse-to-fine synergy.
  1917	4.6. Parameter analysis
  1918	We systematically evaluate the sensitivity of CLNS to five key
  1919	hyperparameters using a control variable strategy. The results for CPC
  1920	(𝐾1 , 𝜌), NCL (𝐾2 , 𝑢, 𝑤) and OTPM (𝜆) modules are analyzed below.
  1921	The hyperparameter of CPC. Fig. 5 illustrates the impact of the
  1922	neighbor number 𝐾1 and confidence threshold 𝜌. Optimal settings are
  1923	(𝐾1 = 50, 𝜌 = 0.8) for SYSU-MM01 and (𝐾1 = 45, 𝜌 = 0.85) for LLCM.
  1924	For 𝐾1 , a moderate value balances the receptive field: a larger 𝐾1 incorporates diverse cross-camera views to enrich structural information,
  1925	while an excessive value introduces irrelevant noise. Regarding 𝜌, its
  1926	choice correlates with dataset difficulty. The challenging LLCM dataset
  1927	requires a higher threshold (𝜌 = 0.85) to enforce strict noise filtering
  1928	and ensure prototype purity. In contrast, the relatively cleaner SYSUMM01 benefits from a lower threshold (𝜌 = 0.8) to exploit informative
  1929	hard samples and maintain feature diversity.
  1930	The hyperparameter of NCL. Next, we investigate the NCL module
  1931	parameters (𝐾2 , 𝑢, 𝑤), as illustrated in Fig. 6. First, for the neighborhood size 𝐾2 , there is a strong correlation between the optimal
  1932	value and the dataset scale/density. For the large-scale SYSU-MM01,
  1933	performance peaks at 𝐾2 = 50, indicating that a broader context is
  1934	required to capture sufficient local manifold structures. Conversely,
  1935	smaller (RegDB) or noisier (LLCM) datasets prefer smaller neighborhoods (𝐾2 = 40 and 15, respectively) to prevent the aggregation of
  1936	outliers and misleading information from distant samples. Second, the
  1937	fusion ratio 𝑢 balances the contribution between hard pseudo-labels and
  1938	soft neighborhood-guided labels. Experiments show that 𝑢 = 0.85 works
  1939	best for SYSU-MM01 and RegDB, while LLCM favors a higher value
  1940	
  1941	4.5. Module interaction and synergy analysis
  1942	Beyond verifying individual contributions in Section 4.4, we investigate the internal synergy among modules. Specifically, we analyze the
  1943	cascade dependency between CPC and OTPM, and the coarse-to-fine
  1944	complementarity between CPC and NCL.
  1945	CPC as a Prerequisite for OTPM. The reliability of the optimal
  1946	transport plan in OTPM hinges on the accuracy of the cost matrix
  1947	derived from cluster prototypes. If prototypes are corrupted by cameraspecific biases, the resulting transport plan is inevitably suboptimal.
  1948	Thus, CPC acts as a critical prerequisite, purifying prototypes to ensure
  1949	accurate cost computation. To validate this, we evaluate the crossmodality matching accuracy (ARI) of OTPM with and without CPC
  1950	calibration. As shown in Table 5, without CPC, the VIS-to-IR matching
  1951	accuracy caps at 0.748 due to camera-distorted visual prototypes.
  1952	In contrast, integrating CPC boosts accuracy to 0.761. This confirms
  1953	that CPC-calibrated prototypes are essential for OTPM to establish
  1954	matching.
  1955	Coarse-to-Fine Complementarity between CPC and NCL. While
  1956	both modules utilize neighborhood information, they target noise at
  1957	8
  1958	
  1959	
codex
CLNS 的证据闭环很清楚：它不是只报总表，而是用 ARI、相机偏差、跨模态匹配准确率和类内距离去对应“结构噪声”和“残余分布噪声”。CMAG 的消融刚好跨到第 9 页以后，我会多读后续页的消融段，避免漏掉证据部分。
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
/bin/zsh -lc "pdftotext -f 9 -l 12 'CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf' - | nl -ba | rg -i \"ablation|CATS|DCAM|GSMEM|CARC|visual|t-sne|attention|memory|circular|module|Table|Fig\\.|without|w/o|effect|contribution|component|analysis|RegDB|SYSU\"" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
     1	ZHANG et al.: CMAG: CROSS-MODAL ATTENTION AND GRAPH-ENHANCED MEMORY FOR USL-VI-ReID
     5	TABLE II
     6	C OMPARISON W ITH S TATE - OF - THE -A RT M ETHODS ON SYSU-MM01 AND R EG DB DATASETS
     8	TABLE III
     9	A BLATION S TUDIES OF D IFFERENT C OMPONENTS ON THE SYSU-MM01 DATASET
    11	for addressing pseudo-label noise, DCAM-ViT for solving
    12	modality discrepancy, GSMEM for overcoming batch training limitations, and CARC for reducing camera background
    14	1) Analysis of Core Component Contributions: To thoroughly assess the effectiveness of each component, we
    15	conducted detailed ablation experiments in both all-search and
    16	indoor-search scenarios on SYSU-MM01. Table III presents
    17	the performance results of different component combinations.
    23	b) Effectiveness of vision transformer backbone: After
    25	to 58.36% Rank-1 and 52.93%mAP in all-search mode, indicating that the ViT architecture effectively captures long-range
    27	c) Effectiveness of DCAM-ViT module: Adding the
    28	cross-modal attention mechanism (Row 4) improved performance to 62.54% Rank-1 and 56.98%mAP (all-search). When
    29	used with CATS and CARC (Row 9), performance further
    37	TABLE IV
    41	Fig. 4.
    47	Fig. 5. Comparison of clustering quality among different methods based
    54	Fig. 3. Impact of hyperparameters p and M on model performance
    55	in SYSU-MM01 dataset. (a) Effect of circular path length p: optimal
    56	performance is achieved at p=3, balancing the ability to capture higherorder relationships without introducing noise. (b) Effect of memory bank
    59	selecting circular structure parameters in cross-modal person re-identification.
    61	improved to 63.18% Rank-1 and 59.91%mAP, demonstrating synergistic effects between components. This validates
    62	that our ViT-specific cross-modal attention mechanism effectively balances modality-specific information with shared
    64	d) Effectiveness of CATS module: CATS (comparing
    66	64.77% Rank-1 and 59.18%mAP (all-search), demonstrating the unique value of circular topological structure in
    70	e) Effectiveness of CARC module: Adding CARC (comparing Row 7 with Row 6) improved performance to 66.39%
    72	Fig. 6. Comparison of top-10 retrieval results between Baseline, partial
    73	modules, and our proposed method(recommended to observe from a color
    76	matching visually similar but identity-different candidates. The partial module
    77	combination (Baseline+ViT+DCAM-ViT+CARC) improves performance but
    79	successfully builds associative paths between different poses through circular
    80	topological structures, effectively connecting front, side, and back views of
    88	ZHANG et al.: CMAG: CROSS-MODAL ATTENTION AND GRAPH-ENHANCED MEMORY FOR USL-VI-ReID
    92	Fig. 7. Progressive component analysis of CMAG framework. The t-SNE visualization (first row) and distance distribution (second row) of 8 randomly
    93	selected identities across different component configurations. In t-SNE visualization, colors indicate identities, circles represent RGB modality, and triangles
    96	f) Effectiveness of GSMEM module: The addition of
    97	GSMEM alone improved performance by approximately 3
    99	complete model with all components (Row 10) achieved the
   100	best performance: 68.52% Rank-1 and 64.0%mAP, demonstrating the important role of global feature memory banks and
   103	2) Comparative Analysis of Graph-Structured Memory
   104	Enhancement: To evaluate different memory mechanisms,
   108	As shown in Table IV, compared to the baseline without
   109	memory, traditional Instance Memory+Cluster Memory shows
   111	GSMEM further outperforms traditional methods, improving
   113	(20.0%). These results validate the advantages of graphstructured memory in maintaining global feature consistency
   116	D. Further Analysis
   117	1) Hyper-Parameter Analysis: Figure 3 shows the impact
   125	that memory bank size M affects model stability, with optimal
   131	C3’s moderate min k value effectively controls connections
   136	E. Visual Comparison
   142	improvement demonstrates CMAG’s ability to generate highquality pseudo-labels through circular topological structures
   143	and graph-structured memory.
   144	Figure 6 presents a visual comparison of retrieval
   145	results by different methods on SYSU-MM01. The baseline method struggles with pose variation issues, especially
   148	red boxes). Although the baseline+ViT+DCAM-ViT+CARC
   149	method improves performance, it still struggles with significant pose differences due to lack of circular structure.
   154	Figure 7 provides progressive visualization analysis of how
   155	each CMAG component enhances feature quality using t-SNE
   159	discrimination. In the t-SNE feature space, RGB (circles)
   170	better separation with each added component. Specifically,
   171	DCAM-ViT enhances cross-modal alignment, CATS improves
   173	shift of intra-class distances, and GSMEM facilitates moderate
   182	attention mechanisms, and global memory enhancement. Our
   184	standard benchmarks, demonstrating the effectiveness of combining algebraic topology theory with modern deep learning
   263	attention via target-aware alignment for visible-infrared person reidentification,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct.
   265	[18] J. Shi et al., “Multi-memory matching for unsupervised visible-infrared

exec
/bin/zsh -lc "pdftotext -f 1 -l 5 'Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf' - | nl -ba | sed -n '372,423p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
   372	
   373	as shown in TABLE I, in the standard setting, our method
   374	achieves the same metrics as the baseline method and maintains a high level of performance. In the cloth-changing scenario, our method outperforms the baseline by 6% and 4.4%
   375	in Rank-1 and mAP, respectively. And it also outperforms the
   376	previous state-of-the-art method by 3.3% and 1.9% on Rank-1
   377	and mAP. For the LTCC dataset, presented in TABLE II, our
   378	method improves the Rank-1 and mAP by 1.5% and 0.2%,
   379	respectively, compared to the baseline method in the clothchanging setting. In the standard setting, Rank-1 and mAP
   380	improve by 0.6% and 0.3%. This demonstrates that our method
   381	for enhancing cloth-changing data has effectively mitigated
   382	biases related to clothing features, resulting in a noticeable
   383	improvement across various datasets.
   384	D. Ablation Study
   385	We conduct comprehensive experiments on the PRCC
   386	dataset to validate the effectiveness of our GFCC module and
   387	CCAL module. As shown in TABLE III, the baseline CAL
   388	model achieves 55.2% on Rank-1 and 55.8% on mAP. When
   389	using the CCAL model, the performance improves, with the
   390	rank-1 and mAP increasing by 0.6% and 0.7%, respectively,
   391	compared to the baseline in the cloth-changing setting. Upon
   392	applying the GFCC module, the performance of our complete
   393	
   394	Fig. 4. Examples of improved misidentification situations of Fig. 3.
   395	
   396	Fig. 5. The visualization of feature maps. In each triplet, the first column
   397	shows the original image. The second and third columns display the feature
   398	maps of the baseline method and our method, respectively.
   399	
   400	method (GFCC+CCAL) is elevated to 61.2% and 60.2% on
   401	Rank-1 and mAP. This represents a significant improvement,
   402	demonstrating that our scheme effectively mitigates biases of
   403	clothing features.
   404	E. Qualitative Results
   405	To notice the differences between the CAL model and our
   406	method, we randomly select images of pedestrians to display
   407	their retrieval rank results. The results in Fig. 3 indicate
   408	that the CAL model has a high misidentification probability
   409	of pedestrians with similar clothing. After implementing our
   410	method, as depicted in Fig. 4, the misidentification of these
   411	pedestrians with similar clothing is solved to a large extent.
   412	Fig. 5 illustrates the pedestrian heatmaps generated by the
   413	baseline method and our method. The ID features of the
   414	baseline method are misled by clothing bias. The clothing
   415	features of pedestrian’s shoulder and chest being highlighted as
   416	easily recognizable cloth-relevant features. While our method
   417	effectively mitigates the impact of clothing bias by emphasizing more clothing-irrelevant information and redirecting
   418	the model’s attention towards the non-clothing areas such
   419	as the face and neck. Through introducing similar clothing
   420	pedestrian data, the identification features extracted by our
   421	method become more concentrated on the areas that influence
   422	the learning of intrinsic ID information.
   423	IV. C ONCLUSION

exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Coarse-to-Fine Cross-Modality Generation for Enhancing Vehicle Re-Identification with High-Fidelity Synthetic Data.pdf' - | nl -ba | sed -n '1188,1399p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
  1188	through both quantitative and qualitative evaluation. For the
  1189	quantitative assessment, we utilize the Frechet Inception
  1190	Distance (FID) [74], a widely recognized evaluation metric.
  1191	Unfortunately, since the PAMTRI dataset is not publicly
  1192	
  1193	7323
  1194	Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:37 UTC from IEEE Xplore. Restrictions apply.
  1195	
  1196	Query ID
  1197	
  1198	Model
  1199	
  1200	Query
  1201	
  1202	Rank-1
  1203	
  1204	Rank-2
  1205	
  1206	Rank-3
  1207	
  1208	Rank-4
  1209	
  1210	Rank-5
  1211	
  1212	70
  1213	60
  1214	
  1215	Baseline
  1216	
  1217	50
  1218	
  1219	mAP
  1220	
  1221	ID 1130
  1222	
  1223	42.83
  1224	
  1225	40
  1226	
  1227	+10.64
  1228	32.19
  1229	
  1230	30
  1231	
  1232	VehicleX
  1233	
  1234	20
  1235	10
  1236	
  1237	Vehicle-Diff
  1238	
  1239	71.50
  1240	+4.96
  1241	66.54
  1242	
  1243	66.56
  1244	+5.05
  1245	61.51
  1246	
  1247	20.65
  1248	
  1249	mAP with real data
  1250	mAP with real and synthetic data
  1251	
  1252	+12.52
  1253	8.13
  1254	1
  1255	
  1256	10
  1257	
  1258	50
  1259	
  1260	100
  1261	
  1262	Real Data Usage in VeRi-776 (Ratio)
  1263	
  1264	Fig. 4: Qualitative retrieval results. Here we compare our
  1265	method with both our baseline and VehicleX. The ranking
  1266	list is presented in descending order from left to right based
  1267	on the similarity score. The images in red boxes are falsematched, whereas the green ones are true-matched.
  1268	Method
  1269	VehicleGAN [10]
  1270	PTGAN [11]
  1271	VehicleX
  1272	Vehicle-Diff
  1273	
  1274	FID↓
  1275	VeRi-776
  1276	CityFlowV2
  1277	233.0
  1278	231.1
  1279	88.20
  1280	77.87
  1281	44.84
  1282	54.84
  1283	
  1284	Fig. 5: Synthetic data provides notable mAP improvements,
  1285	especially when the amount of real training data is small.
  1286	Components
  1287	DFT
  1288	SF
  1289	✓
  1290	✓
  1291	
  1292	✓
  1293	
  1294	#IDs
  1295	
  1296	#Imgs
  1297	
  1298	Rank-1
  1299	
  1300	mAP
  1301	
  1302	FID
  1303	
  1304	5,305
  1305	4,940
  1306	4,896
  1307	
  1308	191,720
  1309	160,758
  1310	149,472
  1311	
  1312	33.19
  1313	58.34
  1314	58.76
  1315	
  1316	8.26
  1317	22.00
  1318	22.33
  1319	
  1320	126.24
  1321	44.35
  1322	44.78
  1323	
  1324	TABLE VI: Ablation study on components, i.e., diffusion
  1325	fine-tuning (DFT) and semantic filtering (SF).
  1326	Baseline
  1327	
  1328	TABLE V: Quantitative comparisons on generated data quality. For a fair comparison, both Vehicle-Diff and VehicleX
  1329	are trained on 1% images of VeRi-776.
  1330	available, we are unable to calculate its FID score. To ensure
  1331	a fair comparison, we randomly selected 1% of the training
  1332	datasets to train VehicleX and generate sample images. As
  1333	shown in Tab. V, Vehicle-Diff achieves a lower FID score
  1334	compared to all other generative methods. For qualitative
  1335	comparison, we visualize the sample outputs of competitive
  1336	generative methods in Fig. 1. The images in the first row
  1337	are from the real-world dataset, while the images in the
  1338	remaining five rows are from different synthetic data pipeline
  1339	based on both 3D engines and GAN. We could observe that
  1340	Vehicle-Diff produces images that are visually closer to the
  1341	real-world dataset while keeping the fine-grained texture.
  1342	C. Ablation Studies and Further Discussion
  1343	Effectiveness of the coarse-to-fine strategy. Here, we
  1344	evaluate the effectiveness of each component in our coarseto-fine generation pipeline. Although the filtering process has
  1345	minimal impact on visual quality and the Fréchet Inception
  1346	Distance (FID) change after fine-tuning is negligible, the
  1347	reID model performance shows consistent improvement (see
  1348	Tab. VI). Tab. VI validates that quality matters more than
  1349	quantity, and Tab. VII shows that more high-quality data
  1350	leads to better results.
  1351	Effectiveness of the balanced sampling strategy. Previous
  1352	methods, such as VehicleX and PAMTRI, typically conduct
  1353	random sampling on mixed real and synthetic data to train
  1354	the model. As a by-product of our pipeline, we introduce
  1355	a balanced sampling strategy. We merge two mini-batch
  1356	samples from real and synthetic datasets as a new mini-batch
  1357	for training. We find that our balanced sampling strategy
  1358	improves model learning on both VehicleX and Vehicle-Diff
  1359	data. As shown in the last four rows of Tab. II, compared to
  1360	the vanilla sampling strategy, our balanced sampling strategy
  1361	yields a +1.06% boost in mAP for VehicleX and +2.81%
  1362	boost in mAP for Vehicle-Diff.
  1363	
  1364	IDE
  1365	
  1366	#IDs
  1367	4,894
  1368	4,896
  1369	
  1370	#imgs
  1371	45,338
  1372	149,472
  1373	
  1374	Rank-1
  1375	57.87
  1376	58.76
  1377	
  1378	Rank-5
  1379	74.97
  1380	74.43
  1381	
  1382	mAP
  1383	22.21
  1384	22.33
  1385	
  1386	TABLE VII: Ablation study on the number of synthetic
  1387	images for training the reID model on the IDE baseline.
  1388	Retrieval visualization. As shown in Fig. 4, we conduct
  1389	the qualitative image retrieval comparison on VeRi-776. Our
  1390	method has successfully recalled the target vehicle in the
  1391	top-5 of the ranking list, surpassing the same model trained
  1392	on real data or VehicleX. It is because that our Vehicle-Diff
  1393	contains a large number of vehicle images with fine-grained
  1394	attributes and intra-class variances such as camera angle,
  1395	facilitating the discriminative feature learning (see Fig. 3).
  1396	Therefore, the model trained on our Vehicle-Diff is able to
  1397	handle challenging matches with fine-grained differences and
  1398	significant camera angle variations.
  1399	Limited real data? To evaluate the effectiveness of VehicleDiff under limited real data conditions, we systematically

hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf' - | nl -ba | sed -n '696,790p' && pdftotext -f 1 -l 10 'ColorSketchNet - Unifying color, sketch and texture for modality-agnostic multi-modal person re-identification.pdf' - | nl -ba | sed -n '1276,1288p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 124ms:
  1276	2022) expands this dataset to three modalities through image captions.
  1277	Compared with existing sketch-based person ReID methods, our method
  1278	can surpass most learning frameworks on the Sketch-RGB task.
  1279	
  1280	Visualization of color sketch eﬀect. We visualize the learned feature representations using T-SNE (Maaten & Hinton, 2008). As shown in
  1281	Fig. 3, it can be observed that the auxiliary color sketch can eﬀectively
  1282	bridge the sketch and photo modalities, reducing the gap caused by
  1283	modality gaps.Existing multimodal ReID may retrieve multiple pedestrians with similar appearances but diﬀerent identities due to diﬀerences
  1284	
  1285	4.4. On the model complexity
  1286	To evaluate the eﬃciency of our method, we compare its training
  1287	overhead and parameter scale in Table 7. As shown, under the same
  1288	8

codex
证据段也读完了，足够支撑逐篇拆解。后面我会把重点放在“观察如何被包装成新问题，以及机制和证据是否真的绑定”，不会把每篇变成模块清单。
方法正文和关键消融段已读完。CMAG 的消融跨到第 9 页后，我补读了后续消融段。
  ✓ 抽取 6 篇 PDF 的前 8 页文本并定位方法段。
  → 逐篇归纳触发观察、重定义、机制、证据闭环和 novelty 来源。
  • 必要时检索先例，补充共同创新套路总结。
web search: 
web search: "CLIP-powered modality centering with spiral training" visible infrared person re-identification
web search: 
web search: "Coarse-to-Fine Cross-Modality Generation" "Vehicle-Diff" vehicle re-identification
web search: 
web search: Vehicle-Diff synthetic data vehicle re-identification diffusion ICRA 2025
web search: 
web search: "UNIReID" unified person re-identification text sketch RGB
已完成逐篇方法论拆解。联网只做了轻量先例边界核对，主体判断来自本地 PDF 方法和消融段。
  ✓ 抽取 6 篇 PDF 的前 8 页文本并定位方法段。
  ✓ 逐篇归纳触发观察、重定义、机制、证据闭环和 novelty 来源。
  ✓ 必要时检索先例，补充共同创新套路总结。
codex
我按 `pdftotext -f 1 -l 8` 读了方法段。CMAG 的消融跨到第 9 页以后，我补读了后续消融段。下面只拆“怎么把观察构造成创新”，不抄模块。

**1. CLIP-powered modality centering with spiral training**

1. 触发观察：作者抓住的是 CLIP-ReID 两阶段范式在 V-I ReID 里会失效。文本提示先由初始图像特征学出来，之后固定文本提示去训练图像编码器；但可见光和红外本来距离很远，图像特征训练过程中不断移动，文本提示不再同步，语义指导会变成陈旧锚点。

2. 重定义动作：他们没有把问题讲成“再做一个跨模态对齐损失”，而是讲成“四个异质空间没有被同时居中”：可见图像、红外图像、可见文本、红外文本。关键词是 `separable text description`、`modality centering`、`identity-aware`、`modality-aware`、`semantic consistency`。

3. 机制怎么长出来：既然文本里也混了身份信息和模态信息，就把提示拆成身份 token 和模态 token，再用不含模态描述的 partial prompt 做 visible-text 和 infrared-text 居中。既然固定提示会过期，就交替训练文本提示和图像编码器，形成 spiral training。这个机制和重定义绑定较紧，但训练序列靠经验选，稍微削弱了“自然推出”的纯度。

4. 证据闭环：关键不是最终 SOTA，而是 STD、MCL、ST 的组合消融。表 6 到表 8 分别证明可分离文本、模态居中、螺旋训练各有作用；表 7 拆 text-to-text 和 image-to-text centering，说明“居中”不是空话。遗憾是没有更直接画出四个空间距离随训练收敛的轨迹，所以闭环是中等强度。

5. reviewer 为什么买账：这篇卖的是视角多于单个损失。它把 CLIP 在 V-I ReID 里的问题从“加文本辅助”改成“文本、图像、模态、身份四个空间共同演化”。novelty 真正来源是 CLIP-ReID 范式迁移到 V-I 后的失配诊断。

**2. CLNS**

1. 触发观察：作者给了一个很具体的数值观察：SYSU-MM01 上，同一身份跨相机正样本平均距离高达 1.0615，而同一身份同相机正样本只有 0.7795。也就是说，聚类更容易按相机分裂身份，而不是按真实身份聚合。

2. 重定义动作：他们把“伪标签有噪声”重定义成“相机诱导的结构性噪声”。这比随机噪声更好讲，因为它有稳定来源：背景、视角、光照和相机域。关键词是 `camera-induced label noise`、`camera-domain centroid`、`coarse-to-fine purification`、`structural gatekeeper`。

3. 机制怎么长出来：如果噪声来自相机，就不能只在全局聚类结果上修补。CPC 先屏蔽同相机近邻，用跨相机一致性校准原型；OTPM 再同时匹配身份原型和相机域中心；NCL 用相机域中心做软监督压实分布；NMU 按可靠性更新 memory，防止离群点污染中心。机制和问题定义绑定很紧。

4. 证据闭环：它做得比较规范。除了组件消融，还用 ARI、inter-camera distance、跨模态匹配准确率和类内距离来证明“相机噪声真的被压下去了”。例如 CPC 后 ARI 提升，PGM 替换 OTPM 会掉点，去掉 NCL 后类内距离变大，去掉 NMU 后 memory 漂移带来性能下降。这些指标正好对应它重定义的噪声来源。

5. reviewer 为什么买账：这篇卖的是视角加证据。模块本身不算激进，但它把无监督 V-I 的伪标签错误讲成可测量、可分层处理的相机结构噪声，所以比“再加一个伪标签清洗模块”更容易成立。

**3. Cloth-debiasing with Stable Diffusion**

1. 触发观察：作者看到的是换衣 ReID 里模型会误认“不同人穿相似衣服”。热图也显示基线更关注肩部、胸口等衣服区域，而不是脸、颈部、轮廓等更稳定线索。另一个观察是 GAN 换衣质量差，容易保留原衣服或姿态几何不稳定。

2. 重定义动作：它把换衣 ReID 从“需要更多换衣样本”改写成“训练集中衣服是分类捷径”。所以目标不是随机换衣，而是“统一衣服特征空间”，让衣服失去区分身份的能力。关键词是 `cloth-debiasing`、`consistent clothing`、`clothing feature bias`、`clothing-irrelevant features`。

3. 机制怎么长出来：既然要让衣服不能当捷径，就用 Stable Diffusion inpainting 生成同风格长袖长裤，尽量覆盖手臂和腿部，迫使模型看头颈、轮廓等非衣服信息。SCHP 提供衣服 mask，OpenPose 控制姿态，LaDI-VTON 保证换衣几何。再用 ReID 相似度、FID 和人工过滤去掉低质图，最后配合 centroid loss 训练。

4. 证据闭环：PRCC 上 baseline 是 55.2/55.8，单加 CCAL 只涨 0.6/0.7，GFCC+CCAL 到 61.2/60.2，说明主要贡献来自“相似衣服数据”这个去偏干预。更关键的是 Fig. 3 到 Fig. 5：相似衣服误检减少，热图从衣服区域转向脸和颈部。这比单纯涨点更能支撑“衣服捷径被削弱”。

5. reviewer 为什么买账：这篇卖的是数据干预视角，不是网络结构。novelty 在于把生成模型用成反捷径工具：不是扩充外观多样性，而是故意制造衣服相似性，让身份学习从衣服上移开。

**4. CMAG**

1. 触发观察：作者把 US-VI ReID 的失败拆成四个具体来源：模态差异、批训练看不到全局关系、伪标签噪声、相机偏置。其中最有辨识度的观察是：直接相似度会漏掉同一身份的远距离姿态变化，但特征图中的闭环路径可能把正面、侧面、背面串起来。

2. 重定义动作：它把伪标签验证从“点对点相似度”重定义为“高阶图结构一致性”。关键词是 `circular topology structure`、`cycle-aware`、`graph-enhanced memory`、`global feature memory`、`camera-aware consistency`。

3. 机制怎么长出来：CATS 用 kNN 图和矩阵幂找闭环路径，并用 circular consistency loss 约束闭环节点；DCAM-ViT 在多 class token 上做跨模态注意力；GSMEM 用队列 memory 和自适应 kNN 图做跨 batch 传播；CARC 用相机内聚类和全局相似度联合过滤冲突样本。CATS 和“拓扑闭环”绑定较紧，其余模块更像是围绕四个失败来源各补一块。

4. 证据闭环：消融显示 DCAM、CATS、GSMEM、CARC 逐步涨，完整模型在 SYSU-MM01 到 68.52 Rank-1 和 64.0 mAP。更贴近重定义的是三类图：超参 p 证明闭环长度有效，检索可视化说明 partial modules 仍处理不好大姿态差，t-SNE 和距离分布显示跨模态聚合和类间分离逐步改善。但如果严审，“拓扑”本身是否比普通图传播更关键，还需要更强的 cycle-only 对照。

5. reviewer 为什么买账：这篇卖的是机制包装加大组合。它的新意来源是把图记忆、跨模态注意力、相机约束放进“闭环拓扑验证”的叙事里。优点是故事大，风险是模块堆叠感重。

**5. Coarse-to-Fine Cross-Modality Generation**

1. 触发观察：车辆 ReID 真实训练数据受隐私和标注成本限制，3D 引擎数据有域差，GAN 数据细粒度属性不稳定。车辆类内差异很小，前格栅、尾灯、车身类型、颜色错一点，合成数据就会误导训练。

2. 重定义动作：它明确说自己“不追求网络结构”，把问题重定义成“面向 ReID 的高保真合成数据生产”。关键词是 `coarse-to-fine generation`、`prompt processing`、`diffusion fine-tuning`、`semantic filtering`、`high-fidelity synthetic data`。

3. 机制怎么长出来：如果问题是属性一致性和真实域风格，就先建车辆属性 prompt 库；如果通用 diffusion 不贴近目标域，就用 1% 无标签真实图经 BLIP-2 生成 caption，再用 LoRA 方式微调 diffusion；如果生成图会多车、碎车、颜色错，就用 YOLO 裁车、CLIP 做颜色语义过滤；最后用 CE 和 circle loss 训练任意 ReID 模型。

4. 证据闭环：它没有只报 ReID 精度，而是同时报 FID、生成组件消融、数据量消融、有限真实数据曲线和检索可视化。关键结论是“质量比数量重要”：语义过滤后数量减少，但 ReID 更好；真实数据越少，合成数据收益越明显。这正好证明它定义的数据瓶颈是成立的。

5. reviewer 为什么买账：这篇卖的是数据生成 pipeline，不是 ReID backbone。novelty 在于把扩散模型从“好看图像生成”转成“细粒度属性受控、少量无标签目标域适配、自动语义过滤”的 ReID 数据工厂。

**6. ColorSketchNet**

1. 触发观察：现实里嫌疑人检索可能同时有文字描述和素描，但现有方法多按 Text-RGB、Sketch-RGB 成对对齐。作者进一步统计属性覆盖：RGB 信息最全，text 有颜色和粗语义但缺轮廓，sketch 有轮廓和姿态但缺颜色和纹理。直接投到共同空间会造成语义歧义。

2. 重定义动作：它把跨模态 ReID 从“缩小模态差距”重定义为“建立公平属性空间”。也就是说，不同模态不是噪声不同，而是属性缺失不同。关键词是 `attribute fairness`、`auxiliary color sketch`、`attribute compensation`、`modality-agnostic`、`unified collaborative alignment`。

3. 机制怎么长出来：既然 text 和 sketch 都缺属性，就生成一个同时含轮廓、颜色、纹理的 color sketch 作为桥。CSG 先做动态光照修正，再用 DoG 抽轮廓，并在 YUV 空间把轮廓放到亮度通道、颜色放到色度通道。ACRM 用 dissimilarity 找缺失信息并补偿，再用 photo-query dissimilarity 去掉辅助模态引入的噪声。UCA 再用身份保持、内容一致、三向结构约束和类内紧致损失统一对齐。

4. 证据闭环：它的消融正好围绕“属性公平”展开。彩色 sketch 相比黑白 sketch 在 text、sketch、text+sketch 三种任务都涨；AC 单独补属性有效，RF 去噪后继续涨；UCA 的各个损失逐步加入也有增益；t-SNE 显示 color sketch 把 sketch 和 photo 拉近。这些证据不是只证明模块有效，而是在证明“补属性再对齐”比直接对齐更合理。

5. reviewer 为什么买账：这篇卖的是视角。它把统一多模态检索讲成“缺失属性补偿问题”，而不是多个任务共享 backbone。novelty 真正来源是 auxiliary color sketch 这个桥接媒介，以及“先补偿再对齐”的逻辑。

这批论文共同套路很清楚：先抓一个具体失败，不说“模型不够强”，而是把失败命名成结构性变量，例如提示漂移、相机诱导噪声、衣服捷径、闭环关系缺失、合成数据语义不稳、属性空间不公平。然后机制只做一件事：制造一个能直接干预这个变量的桥、中心、过滤器或生成过程。最后证据不能只放 mAP，要有对应诊断量或可视化，证明那个被重新定义的问题确实被改变了。

联网只做轻量先例边界核对，主体判断来自本地 PDF。相关背景包括 [CLIP-ReID](https://arxiv.org/abs/2211.13977)、[LaDI-VTON](https://arxiv.org/abs/2305.13501)、[UNIReID](https://arxiv.org/abs/2307.10348) 和 [Vehicle-Diff](https://arxiv.org/abs/2404.06468)。
tokens used
325,689
我按 `pdftotext -f 1 -l 8` 读了方法段。CMAG 的消融跨到第 9 页以后，我补读了后续消融段。下面只拆“怎么把观察构造成创新”，不抄模块。

**1. CLIP-powered modality centering with spiral training**

1. 触发观察：作者抓住的是 CLIP-ReID 两阶段范式在 V-I ReID 里会失效。文本提示先由初始图像特征学出来，之后固定文本提示去训练图像编码器；但可见光和红外本来距离很远，图像特征训练过程中不断移动，文本提示不再同步，语义指导会变成陈旧锚点。

2. 重定义动作：他们没有把问题讲成“再做一个跨模态对齐损失”，而是讲成“四个异质空间没有被同时居中”：可见图像、红外图像、可见文本、红外文本。关键词是 `separable text description`、`modality centering`、`identity-aware`、`modality-aware`、`semantic consistency`。

3. 机制怎么长出来：既然文本里也混了身份信息和模态信息，就把提示拆成身份 token 和模态 token，再用不含模态描述的 partial prompt 做 visible-text 和 infrared-text 居中。既然固定提示会过期，就交替训练文本提示和图像编码器，形成 spiral training。这个机制和重定义绑定较紧，但训练序列靠经验选，稍微削弱了“自然推出”的纯度。

4. 证据闭环：关键不是最终 SOTA，而是 STD、MCL、ST 的组合消融。表 6 到表 8 分别证明可分离文本、模态居中、螺旋训练各有作用；表 7 拆 text-to-text 和 image-to-text centering，说明“居中”不是空话。遗憾是没有更直接画出四个空间距离随训练收敛的轨迹，所以闭环是中等强度。

5. reviewer 为什么买账：这篇卖的是视角多于单个损失。它把 CLIP 在 V-I ReID 里的问题从“加文本辅助”改成“文本、图像、模态、身份四个空间共同演化”。novelty 真正来源是 CLIP-ReID 范式迁移到 V-I 后的失配诊断。

**2. CLNS**

1. 触发观察：作者给了一个很具体的数值观察：SYSU-MM01 上，同一身份跨相机正样本平均距离高达 1.0615，而同一身份同相机正样本只有 0.7795。也就是说，聚类更容易按相机分裂身份，而不是按真实身份聚合。

2. 重定义动作：他们把“伪标签有噪声”重定义成“相机诱导的结构性噪声”。这比随机噪声更好讲，因为它有稳定来源：背景、视角、光照和相机域。关键词是 `camera-induced label noise`、`camera-domain centroid`、`coarse-to-fine purification`、`structural gatekeeper`。

3. 机制怎么长出来：如果噪声来自相机，就不能只在全局聚类结果上修补。CPC 先屏蔽同相机近邻，用跨相机一致性校准原型；OTPM 再同时匹配身份原型和相机域中心；NCL 用相机域中心做软监督压实分布；NMU 按可靠性更新 memory，防止离群点污染中心。机制和问题定义绑定很紧。

4. 证据闭环：它做得比较规范。除了组件消融，还用 ARI、inter-camera distance、跨模态匹配准确率和类内距离来证明“相机噪声真的被压下去了”。例如 CPC 后 ARI 提升，PGM 替换 OTPM 会掉点，去掉 NCL 后类内距离变大，去掉 NMU 后 memory 漂移带来性能下降。这些指标正好对应它重定义的噪声来源。

5. reviewer 为什么买账：这篇卖的是视角加证据。模块本身不算激进，但它把无监督 V-I 的伪标签错误讲成可测量、可分层处理的相机结构噪声，所以比“再加一个伪标签清洗模块”更容易成立。

**3. Cloth-debiasing with Stable Diffusion**

1. 触发观察：作者看到的是换衣 ReID 里模型会误认“不同人穿相似衣服”。热图也显示基线更关注肩部、胸口等衣服区域，而不是脸、颈部、轮廓等更稳定线索。另一个观察是 GAN 换衣质量差，容易保留原衣服或姿态几何不稳定。

2. 重定义动作：它把换衣 ReID 从“需要更多换衣样本”改写成“训练集中衣服是分类捷径”。所以目标不是随机换衣，而是“统一衣服特征空间”，让衣服失去区分身份的能力。关键词是 `cloth-debiasing`、`consistent clothing`、`clothing feature bias`、`clothing-irrelevant features`。

3. 机制怎么长出来：既然要让衣服不能当捷径，就用 Stable Diffusion inpainting 生成同风格长袖长裤，尽量覆盖手臂和腿部，迫使模型看头颈、轮廓等非衣服信息。SCHP 提供衣服 mask，OpenPose 控制姿态，LaDI-VTON 保证换衣几何。再用 ReID 相似度、FID 和人工过滤去掉低质图，最后配合 centroid loss 训练。

4. 证据闭环：PRCC 上 baseline 是 55.2/55.8，单加 CCAL 只涨 0.6/0.7，GFCC+CCAL 到 61.2/60.2，说明主要贡献来自“相似衣服数据”这个去偏干预。更关键的是 Fig. 3 到 Fig. 5：相似衣服误检减少，热图从衣服区域转向脸和颈部。这比单纯涨点更能支撑“衣服捷径被削弱”。

5. reviewer 为什么买账：这篇卖的是数据干预视角，不是网络结构。novelty 在于把生成模型用成反捷径工具：不是扩充外观多样性，而是故意制造衣服相似性，让身份学习从衣服上移开。

**4. CMAG**

1. 触发观察：作者把 US-VI ReID 的失败拆成四个具体来源：模态差异、批训练看不到全局关系、伪标签噪声、相机偏置。其中最有辨识度的观察是：直接相似度会漏掉同一身份的远距离姿态变化，但特征图中的闭环路径可能把正面、侧面、背面串起来。

2. 重定义动作：它把伪标签验证从“点对点相似度”重定义为“高阶图结构一致性”。关键词是 `circular topology structure`、`cycle-aware`、`graph-enhanced memory`、`global feature memory`、`camera-aware consistency`。

3. 机制怎么长出来：CATS 用 kNN 图和矩阵幂找闭环路径，并用 circular consistency loss 约束闭环节点；DCAM-ViT 在多 class token 上做跨模态注意力；GSMEM 用队列 memory 和自适应 kNN 图做跨 batch 传播；CARC 用相机内聚类和全局相似度联合过滤冲突样本。CATS 和“拓扑闭环”绑定较紧，其余模块更像是围绕四个失败来源各补一块。

4. 证据闭环：消融显示 DCAM、CATS、GSMEM、CARC 逐步涨，完整模型在 SYSU-MM01 到 68.52 Rank-1 和 64.0 mAP。更贴近重定义的是三类图：超参 p 证明闭环长度有效，检索可视化说明 partial modules 仍处理不好大姿态差，t-SNE 和距离分布显示跨模态聚合和类间分离逐步改善。但如果严审，“拓扑”本身是否比普通图传播更关键，还需要更强的 cycle-only 对照。

5. reviewer 为什么买账：这篇卖的是机制包装加大组合。它的新意来源是把图记忆、跨模态注意力、相机约束放进“闭环拓扑验证”的叙事里。优点是故事大，风险是模块堆叠感重。

**5. Coarse-to-Fine Cross-Modality Generation**

1. 触发观察：车辆 ReID 真实训练数据受隐私和标注成本限制，3D 引擎数据有域差，GAN 数据细粒度属性不稳定。车辆类内差异很小，前格栅、尾灯、车身类型、颜色错一点，合成数据就会误导训练。

2. 重定义动作：它明确说自己“不追求网络结构”，把问题重定义成“面向 ReID 的高保真合成数据生产”。关键词是 `coarse-to-fine generation`、`prompt processing`、`diffusion fine-tuning`、`semantic filtering`、`high-fidelity synthetic data`。

3. 机制怎么长出来：如果问题是属性一致性和真实域风格，就先建车辆属性 prompt 库；如果通用 diffusion 不贴近目标域，就用 1% 无标签真实图经 BLIP-2 生成 caption，再用 LoRA 方式微调 diffusion；如果生成图会多车、碎车、颜色错，就用 YOLO 裁车、CLIP 做颜色语义过滤；最后用 CE 和 circle loss 训练任意 ReID 模型。

4. 证据闭环：它没有只报 ReID 精度，而是同时报 FID、生成组件消融、数据量消融、有限真实数据曲线和检索可视化。关键结论是“质量比数量重要”：语义过滤后数量减少，但 ReID 更好；真实数据越少，合成数据收益越明显。这正好证明它定义的数据瓶颈是成立的。

5. reviewer 为什么买账：这篇卖的是数据生成 pipeline，不是 ReID backbone。novelty 在于把扩散模型从“好看图像生成”转成“细粒度属性受控、少量无标签目标域适配、自动语义过滤”的 ReID 数据工厂。

**6. ColorSketchNet**

1. 触发观察：现实里嫌疑人检索可能同时有文字描述和素描，但现有方法多按 Text-RGB、Sketch-RGB 成对对齐。作者进一步统计属性覆盖：RGB 信息最全，text 有颜色和粗语义但缺轮廓，sketch 有轮廓和姿态但缺颜色和纹理。直接投到共同空间会造成语义歧义。

2. 重定义动作：它把跨模态 ReID 从“缩小模态差距”重定义为“建立公平属性空间”。也就是说，不同模态不是噪声不同，而是属性缺失不同。关键词是 `attribute fairness`、`auxiliary color sketch`、`attribute compensation`、`modality-agnostic`、`unified collaborative alignment`。

3. 机制怎么长出来：既然 text 和 sketch 都缺属性，就生成一个同时含轮廓、颜色、纹理的 color sketch 作为桥。CSG 先做动态光照修正，再用 DoG 抽轮廓，并在 YUV 空间把轮廓放到亮度通道、颜色放到色度通道。ACRM 用 dissimilarity 找缺失信息并补偿，再用 photo-query dissimilarity 去掉辅助模态引入的噪声。UCA 再用身份保持、内容一致、三向结构约束和类内紧致损失统一对齐。

4. 证据闭环：它的消融正好围绕“属性公平”展开。彩色 sketch 相比黑白 sketch 在 text、sketch、text+sketch 三种任务都涨；AC 单独补属性有效，RF 去噪后继续涨；UCA 的各个损失逐步加入也有增益；t-SNE 显示 color sketch 把 sketch 和 photo 拉近。这些证据不是只证明模块有效，而是在证明“补属性再对齐”比直接对齐更合理。

5. reviewer 为什么买账：这篇卖的是视角。它把统一多模态检索讲成“缺失属性补偿问题”，而不是多个任务共享 backbone。novelty 真正来源是 auxiliary color sketch 这个桥接媒介，以及“先补偿再对齐”的逻辑。

这批论文共同套路很清楚：先抓一个具体失败，不说“模型不够强”，而是把失败命名成结构性变量，例如提示漂移、相机诱导噪声、衣服捷径、闭环关系缺失、合成数据语义不稳、属性空间不公平。然后机制只做一件事：制造一个能直接干预这个变量的桥、中心、过滤器或生成过程。最后证据不能只放 mAP，要有对应诊断量或可视化，证明那个被重新定义的问题确实被改变了。

联网只做轻量先例边界核对，主体判断来自本地 PDF。相关背景包括 [CLIP-ReID](https://arxiv.org/abs/2211.13977)、[LaDI-VTON](https://arxiv.org/abs/2305.13501)、[UNIReID](https://arxiv.org/abs/2307.10348) 和 [Vehicle-Diff](https://arxiv.org/abs/2404.06468)。
