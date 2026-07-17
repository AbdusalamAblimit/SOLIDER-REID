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
session id: 019ef835-d80c-71e1-94d1-52dd737219e5
--------
user
你是 ReID 创新方法论拆解员 19。当前目录是论文库,下面这批 PDF 在本目录。用 `pdftotext -f 1 -l 8 '文件名' -` 读每篇的**完整方法部分**(不只摘要,要 method/approach)。你开了联网搜索,需要查背景/先例/这个 idea 是否新就搜。

== 目标 ==
我们要发 B 类 ReID 方法稿。我**不要你帮我抄模块**,我要学**人家怎么把一个观察构造成能发的创新**——方法论,不是零件。

== 每篇拆解(这才是我要学的,5 点逐篇写全)==
1. **触发观察**: 作者先注意到什么现象/baseline 失败/反直觉结果?(很多创新始于一个具体观察)
2. **重定义动作**: 怎么把这个观察上升成'大家以为 X,其实是 Y'的新问题?用了哪些关键词把旧问题讲成新问题?
3. **机制怎么长出来**: 重定义之后,机制是不是几乎'自然推出'?机制和重定义的逻辑绑定有多紧?
4. **证据闭环**: 用哪个关键消融/可视化证明'重定义是对的'(而不只是机制涨点)?
5. **reviewer 为什么买账**: 这篇卖的到底是机制还是视角?novelty 的真正来源?

== 这批论文(19)==
Multi Queue for Unsupervised Person Re-identification.pdf
Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
Multi-feature balanced network for clothes-changing person re-identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf

中文,务实,**完整**。每篇都要拆,别跳别省。最后用 2-3 句总结这批论文共同的'创新构造套路'。
codex
我会按你的要求把重点放在“观察如何被包装成问题、问题如何推出机制、证据如何闭环”，不是复述模块结构。先确认本目录文件并读取 PDF 处理规则，然后逐篇抽取方法部分。
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
/bin/zsh -lc 'pwd && ls' in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
/bin/zsh -lc "rg --files -g '*.pdf'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
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
Privacy preserving person re-identification via anonymizing diffusion model.pdf
RMPSNet - Occluded person re-identification via regional masking and prompt-distribution synergy.pdf
Dynamic Token Selective Transformer for Aerial-Ground Person Re-Identification.pdf
UAV-based person re-identification - A survey of UAV datasets, approaches, and challenges.pdf
DiverseReID - Towards generalizable person re-identification via Dynamic Style Hallucination and decoupled domain experts.pdf
Bridging the gap - Learning adaptive knowledge transition for lifelong person re-identification.pdf
MSP-ReID - Hairstyle-Robust Cloth-Changing Person Re-Identification.pdf
Semantic Alignment and Hard Sample Retraining for Visible-Infrared Person Re-Identification.pdf
Text-Guided Cross-Modal Alignment with Attribute and Contour Prototypes for Visible-Infrared Person Re-Identification.pdf
ESTI - An Efficient Spatial-Temporal Interaction Network For Video-Based Person Re-Identification.pdf
CLIP-driven fine-grained mining for text-based person search.pdf
Adaptive Pseudo-Label Purification and Debiasing for Unsupervised Visible-Infrared Person Re-Identification.pdf
'Knowledge and experience' for visible-infrared person re-identification.pdf
Attribute Conditional Diffusion-Augmented Person Re-Identification.pdf
Enhancing Visible-Infrared Person Re-Identification With Modality- and Instance-Aware Adaptation Learning.pdf
Spatio-temporal Feature-level Augmentation Vision Transformer for video-based person re-identification.pdf
Dynamic adaptive multi-view contrastive learning for unsupervised person re-identification.pdf
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf
Cross-modality average precision optimization for visible thermal person re-identification.pdf
When Large Vision-Language Models Meet Person Re-Identification.pdf
DIRL - Learning Discriminative ID-Related Representations for Video Visible-Infrared Person ReID.pdf
CMAG - Cross-Modal Attention and Graph-Enhanced Memory for Unsupervised Visible-Infrared Person Re-Identification.pdf
SPCL - Semantic Polymorphism and Commonality Learning for Text-Based Person Retrieval.pdf
Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf
CCUP - A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models.pdf
A Semantic-guided occlusion simulation based local feature semantic expansion network for person re-identification.pdf
Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification.pdf
Multi-Model Synergy Perception for Open-World Person Re-Identification.pdf
Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification.pdf
Cloth-debiasing with Stable Diffusion in Cloth-changing Person Re-identification.pdf
Occluded person Re-Identification with noise injection.pdf
Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification.pdf
Robust mixed-degradation person Re-identification via structural consistency distillation.pdf
Memory-augmented shuffled meta learning for visible-infrared person re-identification.pdf
Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf
Text-to-image Person Search based on Semantic Reorganization.pdf
See what you seek - Semantic contextual integration for cloth-changing person re-identification.pdf
Instant pose extraction based on mask transformer for occluded person re-identification.pdf
FLAG - A Framework With Explicit Learning Based on Appearance and Gait for Video-Based Clothes-Changing Person Re-Identification.pdf
Cross-Modal Full-Mode Fine-Grained Alignment for Text-to-Image Person Retrieval [2026 ACM TOMM arXiv].pdf
CLNS - Camera-aware label noise suppression for unsupervised visible-infrared person re-identification.pdf

codex
本轮我使用 PDF 技能，只做阅读和方法论拆解，不改任何文件。目录里目标 6 篇都存在；我再按项目规则尝试读取上层目标和状态文件，如果沙箱不允许访问，就继续基于你在消息中给出的规则工作。
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
/bin/zsh -lc "sed -n '1,200p' ../../../GOAL.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "sed -n '1,160p' ../../STATUS.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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

exec
/bin/zsh -lc "sed -n '1,180p' ../../../CLAUDE.md" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
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
我已经读到当前目标还是文献调研和候选方向生成，所以这次拆论文是直接服务于“怎么把观察变成可投稿方法”。下面开始逐篇用 `pdftotext -f 1 -l 8` 抽取前 8 页方法正文，之后如果某篇方法没覆盖完整，我会再补读后续页。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi Queue for Unsupervised Person Re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ECAI 2025
I. Lynce et al. (Eds.)
© 2025 The Authors.
This article is published online with Open Access by IOS Press and distributed under the terms
of the Creative Commons Attribution Non-Commercial License 4.0 (CC BY-NC 4.0).
doi:10.3233/FAIA250870

707

Multi-Branch Clothes-Agnostic Feature Learning
for Cloth-Changing Person Re-Identification
Shijuan Huanga, * and Hefei Lingb, **
a Huazhong University of Science and Technology
b Huazhong University of Science and Technology

ORCID (Shijuan Huang): https://orcid.org/0009-0000-2177-5110, ORCID (Hefei Ling):
https://orcid.org/0000-0001-6797-7412
Abstract. Person Re-Identiﬁcation (Re-ID) is crucial for video
surveillance and multi-camera tracking, yet traditional methods
struggle with clothing changes that undermine their reliability. This
paper introduces a novel multi-branch clothes-agnostic feature learning framework to address cloth-changing person re-identiﬁcation
(CC-ReID), which comprises two key modules: Multi-grained
Clothes Caption Generation (MCG), and Multi-Branch ClothesAgnostic Feature Extraction (MAE). MCG leverages Large VisionLanguage Models to generate diverse coarse-to-ﬁne clothing descriptions, reducing the impact of clothing on feature extraction. MAE
employs a dual-branch architecture combining Semantic-Guided
Feature Extraction (SGE) and Parsing Image Feature Extraction
(PIE) to focus on identity-related features while minimizing dependence on clothing characteristics. Extensive experiments on benchmark datasets demonstrate that our method achieves state-of-the-art
performance for CC-ReID tasks, showcasing our method’s effectiveness in real-world applications.

1

Introduction

Person Re-identiﬁcation (Re-ID) [23], a cornerstone technology for
multi-camera surveillance systems and cross-view tracking, focuses
on robustly associating individuals across non-overlapping camera
networks. Conventional Re-ID methodologies predominantly rely on
appearance-based features—such as clothing color distributions, texture patterns, and stylistic attributes—operating under the assumption that subjects retain consistent attire within spatially or temporally constrained environments. While effective in controlled scenarios, these approaches face critical limitations in practical deployments: pedestrians frequently alter clothing across extended time
spans or diverse geographical contexts, rendering appearance-centric
features unstable and severely compromising the practical utility of
traditional models. Consequently, cloth-changing Re-ID (CC-ReID)
has become a critical research area, focusing on accurately identifying individuals despite changes in clothing.
To address the challenge of clothing variations, researchers have
pursued two primary strategies: suppressing clothing interference
and extracting clothes-agnostic features to improve recognition accuracy and robustness. The ﬁrst line of work employs feature disentanglement frameworks to isolate apparel-related attributes from
∗ Corresponding Author. Email: shijuan_huang@hust.edu.cn
∗∗ Corresponding Author. Email: lhefei@hust.edu.cn

identity-discriminative representations. For instance, CAL [10] introduces a clothes-agnostic learning paradigm that penalizes clothing
classiﬁcation accuracy through adversarial training, thereby forcing
the model to discard transient apparel cues while preserving stable
identity semantics. The second strategy leverages multi-modal biometric features insensitive to clothing changes. GI-ReID [17] use gait
features, 3DSL [2] reconstructs 3D shape features, and FSAM [13]
and PRCC [35] utilize silhouette information. However, these methods are complex and often eliminate useful color information. CCAF
[19] uses textual descriptions and pre-trained models like CLIP [26]
to extract clothes-agnostic features, but the semantic gap between
text and image introduces noise.
Text descriptions can capture ﬁne-grained features from images,
making them valuable for CC-ReID. However, using CLIP introduces a semantic gap between text and image, reducing recognition
effectiveness. Furthermore, these methods focus on non-clothing descriptions such as body shape and posture, but these descriptions are
often imprecise, which can weaken model performance. To address
this, we leverage generated clothing descriptions through an attention mechanism and use image parsing techniques to obtain clothingirrelevant features. This approach helps the model learn more robust
identity features while minimizing dependence on clothing-speciﬁc
characteristics.
First, we use various Large Vision-Language Models (LVLMs)
to generate clothing descriptions with different granularities, from
coarse to ﬁne. To ensure accuracy, we apply human parsing algorithms [18] to isolate clothing regions in images, allowing LVLMs
to focus speciﬁcally on these areas. Given that generated descriptions may contain noise, we ﬁlter them using cosine similarity. This
process yields multi-granular, precise clothing descriptions for each
image, which are crucial for reducing the inﬂuence of clothing on
feature extraction.
Next, we adopt a Multi-Branch Clothes-Agnostic Feature Extraction approach to minimize dependence on clothing features and focus
on identity-related features. This approach contains two main components: Semantic-Guided Feature Extraction (SGE) and Parsing Image Feature Extraction (PIE). In SGE, we input generated clothing
captions into a text encoder to extract textual features, then apply an
attention mechanism to reduce the inﬂuence of clothing characteristics on image features. In PIE, we use parsing algorithms to identify
and mask clothing regions, creating additional training samples for
the image encoder. This multi-branch design encourages the model

708

S. Huang and H. Ling / Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification

to focus on persistent features like body structure and posture, rather
than speciﬁc clothing colors and styles.
Our contributions are summarized as follows:
• We use LVLMs to generate precise coarse-to-ﬁne captions for the
clothes of each image, mitigating the impact of clothing changes
on recognition.
• We employ a multi-branch architecture: semantic-guided feature
extraction and parsing image feature extraction, diminishing the
inﬂuence of clothing in feature extraction.
• Extensive experiments on two widely used CC-ReID datasets
demonstrate that our method achieves state-of-the-art performance.

2
2.1

Related work
Person Re-identiﬁcation

Person Re-identiﬁcation (Re-ID) is a fundamental computer vision
task focused on retrieving target individuals across distributed camera networks with non-overlapping ﬁelds of view. Early works [8]
establish early methodologies for viewpoint-invariant feature extraction, emphasizing handcrafted descriptors robust to pose variations and illumination changes. The ﬁeld underwent a paradigm shift
with the integration of deep neural networks, which enabled end-toend learning of discriminative representations. A milestone work by
Zheng et al. demonstrates this potential through a dedicated CNN
architecture optimized for Re-ID tasks.
Subsequently, researchers turn their attention to addressing challenges like domain adaptation and cross-view matching. Wei et al.
[32] pioneer GAN-based frameworks to align feature distributions
between source and target domains, effectively mitigating performance degradation in unseen environments. Concurrently, Luo et al.
[22] tackle fragmented label spaces across domains through a novel
cross-domain mixup strategy, which interpolates both features and
labels to synthesize intermediate samples, thereby enhancing model
robustness to domain shifts.

fusion mechanisms that balance semantic guidance with visual speciﬁcity in CC-ReID systems.

2.3

In recent years, LVLMs have signiﬁcantly improved multimodal data
processing by leveraging the capabilities of Large Language Models,
enabling efﬁcient image-to-text conversion and generation. QwenVL [1] excels in object detection and localization, providing robust cross-language visual understanding. InternVL2.5 [4] optimizes
cross-modal interactions, making it suitable for advanced applications. MiniGPT-4 [42] aligns a pre-trained vision encoder with LLMs
via ﬁne-tuning a single linear layer, achieving efﬁcient multimodal
understanding. MiniGPT-5 [41] combines Stable Diffusion mechanism with LLMs through special visual tokens known as "generative vokens", introducing a proﬁcient multi-modal generation. BLIP3
[34] introduces a 3-stage training paradigm and uses next token prediction to unify the training objective, enhancing both training efﬁciency and model performance. LLaVA [21] enhances performance
in complex tasks through instruction tuning with an open-source
dataset.

3

Cloth-Changing Person Re-identiﬁcation

The core challenge in CC-ReID lies in extracting identity-consistent
features resilient to apparel variations. Some methods employ adversarial learning to decouple clothing attributes from identity semantics. For instance, CAL [10] employs an adversarial loss function
to penalize clothing classiﬁcation, extracting clothing-irrelevant features. Other studies exploit complementary biometric modalities inherently invariant to clothing changes. Gait-based approaches such
as GI-ReID [17] analyze temporal motion dynamics in walking sequences, while MBUNet [38] fuses posture estimation with spatial
attention to capture anatomical invariants. Contour-driven methods
like FSAM [13] employ semantic parsing networks to align silhouette boundaries, and 3DSL [2] reconstructs 3D mesh representations to model body shape topology. However, these techniques often
demand complex preprocessing pipelines (e.g., pose estimation or
depth sensing) and risk discarding discriminative color/texture features crucial for short-term identiﬁcation. To mitigate these limitations, hybrid frameworks like CCAF [19] integrate CLIP’s visionlanguage capabilities [26] with textual prompts to guide clothesagnostic feature learning. Nevertheless, the inherent semantic gap
between coarse textual descriptions and ﬁne-grained visual patterns
introduces alignment noise. This underscores the need for adaptive

Method

In this section, we introduce our proposed method, which consists of
two modules: the Multi-grained Clothes Caption Generation (MCG)
and the Multi-Branch Clothes-Agnostic Feature Extraction (MAE).
Figure 1 provides an overview of the framework. The MCG module
generates coarse-to-ﬁne clothes captions to guide the MAE module
in suppressing clothing-related features. The MAE module employs
a multi-branch architecture to extract clothes-agnostic features.

3.1
2.2

Large Vision-Language Models

Multi-grained Clothes Caption Generation

Previous studies attempt to extract clothes-agnostic features by introducing text descriptions, focusing on inherent characteristics like
body shape while avoiding clothing details. However, these descriptions often lack speciﬁcity and unique identifying features. To address this, we leverage LVLMs to generate detailed clothing descriptions from images. We then use a text encoder to extract clothing features from these descriptions and apply an attention mechanism to reduce the inﬂuence of clothing characteristics on image features. This
approach enables a more accurate and robust extraction of clothesagnostic features.
We use human parsing algorithms to isolate clothing areas in images and direct the LVLMs to focus on these areas. To enhance description diversity, we employ multiple open-source LVLMs, including Qwen2-VL, BLIP3, MiniGPT-4, and InternVL2.5.
To generate descriptions with varying levels of detail, we use
prompts ranging from broad to speciﬁc. A broad prompt like "Please
brieﬂy summarize the person’s clothing in one sentence." leads
LVLMs to produce succinct summaries, highlighting the most prominent clothing features while omitting ﬁner details. In contrast, a detailed prompt such as "Provide a detailed description of the outﬁt,
specifying its color, type, style, and material" encourages rich and
comprehensive clothing descriptions.

S. Huang and H. Ling / Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification

Figure 1.

709

Overview of our proposed framework, which consists of two modules: the Multi-grained Clothes Caption Generation (MCG) module and the
Multi-Branch Clothes-Agnostic Feature Extraction (MAE) module.

description. Thus, we select the two descriptions with the highest
scores for subsequent model training.

Example Clothes Caption with VLM
Broad Clothes Description
Input: Please brieﬂy summarize the person’s clothing in one
sentence.

3.2

LVLM: A white short-sleeved T-shirt paired with black
cropped pants and red sneakers.

Multi-Branch Clothes-Agnostic Feature
Extraction

Detailed Clothes Description
Input: Provide a detailed description of the outﬁt, specifying
its color, type, style, and material.
LVLM: A white cotton short-sleeve crewneck T-shirt
(minimalist design) paired with black mid-calf cotton trousers
and red mesh-paneled sneakers with white rubber soles,
creating a casual monochromatic contrast.
Using four LVLMs and two prompts, we generate eight distinct
clothing descriptions for each image. However, due to the inherent unpredictability of LVLMs, these captions may contain mismatches or inaccuracies. To select clean clothing descriptions, we
input the descriptions and their corresponding images into the pretrained CLIP model’s text and image encoders, generating text features Ft and image features Fv . We calculate the cosine similarity to
obtain a matching score s as follows:
s = sim(Fv , Ft ) =

F v · Ft
.
Fv  × Ft 

(1)

The higher its matching score with the image, the more accurate the

We employ a multi-branch architecture comprising two components:
Semantic-Guided Feature Extraction (SGE) and Parsing Image Feature Extraction (PIE).
Semantic-Guided Feature Extraction. Given a person image
I ∈ RW ×H×C , the image is divided into non-overlapping patches,
with the number of patches N = H × W/P 2 , where P is the
patch size. A trainable linear projection layer maps these patches
to one-dimensional vectors {fiv }N
i=1 . A learnable [CLS] vector is
prepended to this sequence. The sequence is then processed by
the EVA-02 image encoder [7] to extract visual features Fv =
v
v
v
{fcls
, f1v , . . . , fN
}, where fcls
represents the global feature and
v
v
{f1 , . . . , fN } represent local features.
For each person image, we obtain two clothing descriptions from
the MCG module. Each description is tokenized using lower-cased
byte pair encoding (BPE) with [SOS] and [EOS] tokens added at
the beginning and end, respectively. These tokenized sequences are
processed through CLIP’s text encoder to generate textual features.
The text features from the two descriptions are averaged to produce
t
t
t
the ﬁnal text feature Ft = {fsos
, f1t , . . . , fM
, feos
} for the image.
First, we use the LayerNorm to normalize the visual features Fv

and the text features Ft , resulting in the normalized features Fv and

S. Huang and H. Ling / Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification

710


3.3

Ft :


Fv = LayerN orm(Fv ),

(2)



Ft = LayerN orm(Ft ).

In our experiments, we use cross-entropy classiﬁcation loss (without label smoothing) and triplet loss. The cross-entropy loss Lid is
deﬁned as:







(3)

The Multi-Head Attention mechanism enables the model to focus on multiple parts of the input data simultaneously, enhancing
its ability to capture complex relationships. Speciﬁcally, given query
Q, key K, and value V matrices, Multi-Head Attention computes a
weighted sum of the values, with weights determined by the compatibility between queries and keys. This process is applied multiple
times in parallel using different learned projections of Q, K, and V .
Mathematically:
M ultiHead(Q, K, V ) = Concat(head1 , . . . , headn )W O ,

(4)

where each headi is computed as:
headi = Attention(QWiQ , KWiK , V WiV ),

(5)

and the scaled dot-product attention is:
QK T
Attention(Q, K, V ) = sof tmax( √ )V.
dk
O

Here, W
are projection matrices for the i-th head; dk is the dimensionality of
the key vectors.
Subsequently, to suppress clothing-related features, we subtract
vm
vm
Fvc from Fv , yielding Fvm = {fcls
, f1vm , . . . , fN
}:
Fvm = Fv − Fvc .

(7)

This subtraction aims to eliminate clothing-speciﬁc characteristics,
isolating stable and person-speciﬁc visual features.
To capture ﬁne-grained information, we utilize the local features
vm
{f1vm , . . . , fN
} of Fvm . These local features undergo average
pooling to aggregate detailed information into a single feature vector,
which is then normalized for consistency:
vm
f = LayerN orm(mean-pooling({f1vm , . . . , fN
})).

Lid = −

(8)

The feature f is passed through an identity classiﬁer to generate
the predicted person ID R.
Parsing Image Feature Extraction (PIE). As another critical
branch of our model, for a given person image I, we utilize human parsing algorithms to generate a parsing image Ip with occluded
clothes. The human parsing algorithm accurately identiﬁes clothing
locations, and we mask the person’s clothing to further minimize the
impact of clothing changes. Following similar processing steps: the
image is segmented into N patches and ﬂattened, a learnable [CLS]
vector is prepended before input into EVA-02 image encoder, exp
p
tracting a set of image features represented as {fcls
, f1p , . . . , fN
}.
p
p
By average pooling and normalizing {f1 , . . . , fN }, the feature f p
is obtained. Subsequently, f p passes through an identity classiﬁer to
generate the predicted person ID Rp .
The features f , R, f p , and Rp undergo relevant loss computations
and model training processes. By integrating these two branches, our
model achieves a more stable and accurate feature representation.

C

(9)

where B is the batch size, i is the instance index, yi,c is the one hot
label of class c, ŷi,c is the softmax probability score for class c.
The triplet loss function Ltri is deﬁned as follows:
Ltri (IiA , IiP , IiN ) = max{0, M + D (IiA , IiP ) − D (IiA , IiN )}, (10)
where D (·) denotes the squared Euclidean distance in the embedding space, and M is a tunable margin hyperparameter that governs
the separation between positive and negative sample pairs. IiA , IiP ,
and IiN represent the anchor image, positive image, and negative
image, respectively, minimizing intra-class distances while maximizing inter-class separation. The model learns to minimize the distance
between images of the same class (e.g., anchor and positive) while
maximizing the distance between images of different classes (e.g.,
anchor and negative).
In the MAE module, both branches adopt two core loss functions
to guide the training processes. Speciﬁcally, the overall loss function
is deﬁned as follows:

(6)

is the ﬁnal output projection matrix; WiQ , WiK , and WiV

1 
yi,c log (ŷi,c ) ,
B i=1 c=1
B

Then, we apply the Multi-Head Attention mechanism to obtain
image features highly correlated with clothing, denoted as Fvc :
Fvc = M ultiHead(Fv , Ft , Ft ).

Optimization

L = λ1 Lid + λ2 Ltri ,

(11)

where λ1 and λ2 are constant coefﬁcients to control the contribution
of each loss function. Without loss of generality, we set them to 1 in
all of our experiments.

4
4.1

Experiments
Datasets and Settings

PRCC [35] contains 33,698 images of 221 unique identities captured
across three cameras. There are about 50 images of each person under each camera view. While clothing appearance remains consistent
across cameras A and B, pedestrians wear different garments in camera C.
LTCC [25] includes 17,138 images of 152 distinct identities, captured from 12 different cameras. The dataset encompasses a total of
478 unique clothing sets. It is divided into two subsets: one with
changing clothes, including 91 people wearing 415 different outﬁts
across 14,756 images, and another with consistent clothes, comprising 61 individuals and 2,382 images. On average, each person has
ﬁve clothing changes.
VC-Clothes [29] The VC-Clothes dataset is a virtual dataset generated from the game GTA-V, containing 512 identities and a total
of 19,060 images. It is captured by four cameras, with an average
of 9 images per identity. The dataset is split into a training set consisting of 256 identities and a test set containing the remaining 256
identities.
DeepChange [33] is a large-scale outdoor CC-ReID dataset collected over two years, consisting of more than 170,000 images featuring 1,121 distinct individuals captured by 17 surveillance cameras.
The dataset is divided into three subsets: 450 identities for training,
150 for validation, and 521 for testing.
Celeb-reID-light [15], a subset of Celeb-reID, comprises 10,842
images of 9,021 individuals collected from diverse sources. This

S. Huang and H. Ling / Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification

Table 1.

Performance comparisons with state-of-the-art methods on PRCC and LTCC datasets.
PRCC

Methods

Venue

HACNN [20]
PCB [27]
IANet [14]
FSAM [13]
GI-ReID [17]
3DSL [17]
CAL [10]
ACID [37]
AIM [36]
DCR-ReID [5]
CCFA [11]
MBUNet [38]
MADE [24]
CCAF [19]
CDM-GCA [6]
FRD-ReID [3]
PGDS [28]
F IRe2 [30]
CCUP [40]
CSSC [31]
Ours

CVPR18
ECCV18
CVPR19
CVPR21
CVPR22
CVPR22
CVPR22
TIP23
CVPR23
TCSVT23
CVPR23
TIP23
TMM25
arXiv24
ICME24
ICIC24
AVSS24
TIFS24
ICME25
ICASSP25
-

Table 2.

Same-clothes
Rank-1
mAP
82.5
84.8
99.8
97.0
99.4
98.3
98.8
80.0
100.0
99.8
99.1
99.0
100.0
99.9
100.0
99.7
99.6
98.7
100.0
99.6
100.0
98.9
99.9
98.4
99.3
94.3
100.0
99.9
100.0
99.5
100.0
99.1
100.0
99.0

LTCC
Cloth-changing
Rank-1
mAP
21.8
23.2
41.8
38.7
46.3
45.9
54.5
33.3
37.5
51.3
55.2
55.8
55.4
66.1
57.9
58.3
57.2
57.4
61.2
58.4
68.7
65.2
67.5
64.2
70.4
63.7
64.8
61.3
65.5
63.3
65.0
63.1
64.7
57.9
65.5
63.0
71.6
67.6

Performance comparisons on DeepChange(%).

Methods

Venue/Year

RGA-SC [39]
TransreID [12]
CAL [9]
MCSC [16]
CCAF [19]
Ours

CVPR 20
ICCV 21
CVPR 22
IEEE TIP 24
arXiv 24
-

DeepChange
Rank-1
mAP
28.9
8.6
35.9
14.4
54.0
19.0
56.9
21.5
59.6
21.0
69.7
29.5

Table 3. Comparison of our method with baseline across different datasets.
The LTCC, PRCC, and VC-Clothes are all evaluated under the
cloth-changing setting.
Dataset
PRCC
LTCC
VC-Clothes
DeepChange
Celeb-reID-light

711

Baseline
Rank-1
mAP
61.4
60.2
42.9
21.5
84.5
84.7
62.5
23.5
74.6
54.1

Ours
Rank-1
mAP
71.6
67.6
52.3
29.9
85.7
85.8
69.7
29.5
75.6
55.8

Cloth-changing
Rank-1
mAP
21.6
9.3
23.5
10.0
25.0
12.6
38.5
16.2
23.7
10.4
31.2
14.8
40.1
18.0
29.1
14.5
40.6
19.1
41.1
20.4
45.3
22.1
40.3
15.0
46.9
25.0
42.9
20.1
50.9
29.8
49.1
26.7
44.6
19.1
43.6
18.6
52.3
29.9

Implementation details. We adopt the EVA02-CLIP-L model [7]
with a patch size of 14 as our image encoder, while leveraging the
CLIP text encoder to generate textual embeddings for clothing attributes. To address viewpoint variations, we integrate camera metadata as camera-aware positional encoding following prior work [24],
where camera identiﬁers are embedded as side information (SIE)
during feature projection. All input images are resized to 224×224
resolution, and training is conducted with a batch size of 8 across
all datasets. The optimization process employs the SGD optimizer
with an initial learning rate of 2 × 10−5 , which is decayed by a factor of 10 at the 20th and 40th epochs within the 60-epoch training
schedule. All experiments are performed on a single NVIDIA RTX
4090 GPU for both training and evaluation phases. The model architecture contains 606.94 million trainable parameters. To optimize
computational efﬁciency, clothing captions are generated ofﬂine using Large Vision-Language Models, while human parsing images are
precomputed before training, eliminating any runtime overhead during model execution.

4.2
dataset is derived from candid street photography of celebrities, with
its images exhibiting signiﬁcant pose variations.
Evaluation Metrics. We use Cumulative Matching Characteristics (CMC) and mean Average Precision (mAP) as our metrics.
Rank-1 in CMC measures the probability of correctly identifying the
target ID within the Rank-1 results, and mAP quantiﬁes overall retrieval precision by averaging precision across recall thresholds.
Three evaluation settings. Same-Clothes Setting exclusively utilizes clothing-consistent samples by excluding all same-identity
same-camera instances, simulating short-term re-identiﬁcation scenarios where apparel remains unchanged. Cloth-Changing Setting
isolates clothing-variant scenarios by excluding same-identity, samecamera, and same-clothing samples, speciﬁcally evaluating robustness to apparel modiﬁcations. Standard Setting adopts real-world
operational conditions by incorporating both clothing-consistent
and clothing-variant samples while excluding same-identity samecamera pairs, providing comprehensive cross-camera retrieval assessment.

Standard
Rank-1
mAP
60.2
26.7
65.1
30.6
63.7
31.0
73.2
35.4
63.2
29.4
74.2
40.8
65.1
30.6
76.3
41.1
76.1
42.3
75.8
42.5
67.6
34.8
82.2
49.3
75.3
41.3
77.9
45.1
77.5
43.0
75.9
39.9
78.1
40.2
85.2
51.3

Comparison with state-of-the-art methods

In this section, we compare our method with other state-of-the-art
approaches on the PRCC and LTCC datasets. The comparison results
are summarized in Table 1.
For the PRCC dataset, our method demonstrates exceptional performance in the same-clothes setting, achieving a perfect Rank-1 of
100.0% and an mAP of 99.0%. In the cloth-changing (CC) setting,
our method attains a Rank-1 of 71.6% and an mAP of 67.6%, surpassing the next best method (CCAF) by 1.2% in Rank-1 and 3.9%
in mAP.
On the LTCC dataset, our method achieves a Rank-1 of 85.2% and
an mAP of 51.3% under standard setting. In the more challenging CC
setting, our method attains a Rank-1 of 52.3% and an mAP of 29.9%,
outperforming the second-best competitor (FRD-ReID) by 1.4% in
Rank-1 accuracy.
To further evaluate in complex scenarios, we evaluate our method
on the large-scale real-world DeepChange. As shown in Table 2, our
method reaches 69.7% Rank-1 accuracy and 29.5% mAP, surpassing

S. Huang and H. Ling / Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification

712

Table 4.

Ablation studies of each component on PRCC and LTCC datasets, “MC” denotes multi-grained clothes caption.
PRCC

Components
Baseline






SGE




MC



PIE



Same-clothes
R-1
mAP
100.0
99.7
100.0
99.4
100.0
99.1
99.9
98.0
100.0
99.0

Table 5. Comparison of global and local features in images. “SC”:
Same-clothes, “CC”: Cloth-changing, “ST”:Standard.
PRCC
Feature
Global
Local

SC
R-1
100.0
100.0

mAP
99.0
99.0

CC
R-1 mAP
68.8 67.5
71.6 67.6

LTCC
ST
CC
R-1 mAP R-1 mAP
78.7 43.3 46.4 23.5
85.2 51.3 52.3 29.9

the second-best method by +10.1% and +8.0%, conﬁrming strong
real-world applicability.

4.3

Comparison with the baseline method

We conduct comprehensive comparisons against a baseline using
the same backbone architecture and loss function. We evaluate performance across ﬁve challenging benchmarks: LTCC, PRCC, VCClothes under cloth-changing setting, DeepChange, and Celeb-reIDlight in Table 3.
On the PRCC dataset, our method achieves a Rank-1 accuracy
improvement from 61.4% to 71.6% and an increase in mAP from
60.2% to 67.6%. The LTCC benchmark sees even more pronounced
advancements, with Rank-1 accuracy surging by 9.4% (52.3% vs.
42.9%) and mAP rising 8.4% (29.9% vs. 21.5%). Similarly, for
the VC-Clothes dataset, our approach raises Rank-1 accuracy from
84.5% to 85.7% and mAP from 84.7% to 85.8%.
The most striking gains emerge on DeepChange, where Rank-1
accuracy climbs 7.2% from 62.5% to 69.7%, alongside a 6.0% mAP
boost from 23.5% to 29.5%. Similarly, Celeb-reID-light beneﬁts substantially from our framework, attaining excellent performance with
75.6% Rank-1 and 55.8% mAP.
These consistent advancements across diverse benchmarks—spanning controlled laboratory environments (PRCC)
to real-world surveillance scenarios (DeepChange)—validate our
method’s robustness in addressing the core challenges of CC-ReID,
particularly in disentangling appearance-irrelevant features while
preserving identity-discriminative cues.

4.4

Ablations and analysis

To assess the effectiveness of each component, we conduct ablation
experiments on PRCC and LTCC datasets, as shown in Table 4. We
use the EVA-02 model trained with cross-entropy loss and triplet loss
as the baseline method.
The effectiveness of each proposed module. By integrating the
SGE module using single captions from Qwen2-VL, we observe performance improvements over the baseline method: on the PRCC
dataset, Rank-1 for CC increases by 2.0% (from 61.4% to 63.4%).
On the LTCC dataset, Rank-1 for CC improves by 2.0% (from 42.9%
to 44.9%), and mAP increases from 21.5% to 22.7%.
When employing multi-grained captions from diverse LVLMs,
performance is further enhanced: on the PRCC dataset, Rank-1 for

LTCC
Cloth-changing
R-1
mAP
61.4
60.2
63.4
60.9
65.9
61.7
66.1
63.4
71.6
67.6

Table 6.

Standard
R-1
mAP
75.9
40.1
78.1
43.2
83.2
48.4
82.8
47.0
85.2
51.3

Cloth-changing
R-1
mAP
42.9
21.5
44.9
22.7
46.9
24.2
45.4
22.3
52.3
29.9

Performance comparison with different numbers of captions on
the PRCC dataset under CC setting.

Number of Captions
Rank-1 (%)

One
63.4

Two
65.9

Three
64.2

Four
63.3

CC increases to 65.9%, and mAP improves to 61.7%. On the LTCC
dataset, Rank-1 for CC improves to 46.9%, and mAP increases to
24.2%.
Incorporating the PIE module into the baseline also yields signiﬁcant gains: on the PRCC dataset, Rank-1 for CC reaches 66.1%, and
mAP improves to 63.4%; on the LTCC dataset, Rank-1 for CC is
45.4%, and mAP is 22.3%.
Combining all three components achieves the best overall performance: on the PRCC dataset, Rank-1 accuracy for CC reaches
71.6%, and mAP climbs to 67.6%; on the LTCC dataset, Rank-1 accuracy for CC hits 52.3%, and mAP reaches 29.9%. These results
demonstrate the powerful synergistic effect of combining these modules, effectively addressing clothing changes and boosting overall
performance.
Comparison of global and local features in images. We perform a systematic evaluation of global and local feature paradigms
on the PRCC and LTCC datasets, as summarized in Table 5. The
ﬁndings reveal that local features consistently outperform global features across various metrics and scenarios. Speciﬁcally, on the PRCC
dataset, in CC scenarios, local features signiﬁcantly surpass global
features with a Rank-1 accuracy of 71.6% compared to 68.8%. On
the LTCC dataset, the superiority of local features is even more pronounced. For SC scenarios, local features yield 85.2% Rank-1 and
51.3% mAP, whereas global features only reach 78.7% and 43.3, respectively. On CC conditions, local features again show a marked
advantage with a Rank-1 accuracy of 52.3%, compared to 46.4% for
global features. And local features attain an mAP of 29.9% while
global features lag behind at 23.5These results highlight the critical
role of local features in capturing ﬁne-grained details, especially in
challenging cloth-changing scenarios.
Comparison with different numbers of captions. Table 6 compares performance with varying caption numbers on PRCC CC settings. Optimal results (65.9% Rank-1) are achieved using two captions, outperforming single-caption (63.4%) and multi-caption conﬁgurations (≤ 64.2%). This demonstrates that two captions optimally balance diversity and relevance: they mitigate single-model
bias through complementary descriptions while avoiding noise from
lower-quality additional captions. Single descriptions lack robustness to LVLM limitations, while three or more introduce conﬂicting semantics that degrade feature learning. Our CLIP-ﬁltered dualcaption strategy maximizes information gain while minimizing inconsistency. Our selection strategy thus achieves the critical equilibrium between information diversity and signal purity.
Visualization of feature maps. To quantitatively contrast the feature representation patterns between the baseline and our approach,
we visualize feature heatmaps on the PRCC and LTCC datasets

S. Huang and H. Ling / Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification

713

Figure 2. Visualization of feature maps on PRCC and LTCC. The ﬁrst
column shows the original image, while the second and third columns
display feature maps of the baseline and our method.
Table 7.
Method
GI-ReID
AIM
MADE
Ours

Runtime efﬁciency comparison of different methods during
inference.
Processing Time(ms)
27.13
29.13
33.99
14.12

Rank-1(%)
33.3
57.9
67.5
71.3

mAP(%)
37.5
58.3
64.2
67.6

in Figure 2. The baseline exhibits diffuse attention patterns, disproportionately emphasizing transient attributes such as clothing
textures and environmental artifacts, which are inherently unstable
across clothing-change scenarios. Conversely, our method demonstrates spatially concentrated activations targeting persistent biometric identiﬁers—including facial contours, gait postures, and accessory geometries—while actively suppressing apparel-related regions.
These observations validate our dual strategy of semantic-guided feature extraction and parsing image Feature extraction, which synergistically disentangle transient and persistent identity factors for reliable
cloth-changing retrieval.
Runtime efﬁciency comparison of different methods. We evaluate computational efﬁciency by measuring the processing time per
image in Table 7. Our method uses only the original RGB image
during inference, achieving 14.12 ms per image, signiﬁcantly outperforming GI-ReID, AIM, and MADE. Notably, conventional approaches consistently reveal an inherent speed-accuracy compromise—illustrated by MADE attaining 67.5% Rank-1 accuracy at
33.99 ms per-image latency. Our methodology fundamentally breaks
this trade-off paradigm: achieving state-of-the-art recognition performance while slashing processing time by 58.5% relative to MADE.
This dual optimization demonstrates unprecedented computational
efﬁciency without sacriﬁcing retrieval precision, establishing exceptional practicality for real-world implementation.
Retrieval result visualization. Figure 3 illustrates the top-10 retrieval results for query images selected from the PRCC dataset.
Our model demonstrates robust recognition capabilities by accurately matching identities across drastic apparel changes and viewpoint discrepancies, leveraging multi-branch clothes-agnostic feature
extraction to prioritize stable biometric cues such as facial structure
and gait dynamics. In contrast, the baseline method exhibits critical
limitations: it frequently misidentiﬁes subjects with visually similar
clothing styles or hairstyles, even when their body proportions differ

Figure 3. Top-10 retrieval results on the PRCC dataset using the baseline
and our approach. Each row presents the ranking for a query image
(leftmost), followed by Rank-1 to Rank-10 retrieved images from left to
right. The upper row denotes the baseline, and the lower row shows our
results, with true matches in green and false ones in red.

markedly. These results collectively highlight the superiority of our
method and validate its effectiveness in capturing clothes-agnostic
features for accurate CC-ReID.

5

Conclusion

In this paper, we propose a novel framework for CC-ReID that includes two key modules: Multi-grained Clothes Caption Generation (MCG) and Multi-Branch Clothes-Agnostic Feature Extraction
(MAE). MCG uses LVLMs to generate diverse clothing descriptions, reducing the inﬂuence of clothing changes. MAE combines
semantic-guided and parsing image feature extraction to focus on
identity-related features. Our approach achieves state-of-the-art performance, showing its effectiveness in handling real-world clothing
variations.

Acknowledgements
This work was supported in part by the Natural Science Foundation
of China under Grant 62372203 and 62302186, in part by the Major Scientiﬁc and Technological Project of Shenzhen (202316021),
in part by the National key research and development program of
China(2022YFB2601802), in part by the Major Scientiﬁc and Technological Project of Hubei Province (2022BAA046, 2022BAA042).

References
[1] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou,
and J. Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint
arXiv:2308.12966, 1(2):3, 2023.
[2] J. Chen, X. Jiang, F. Wang, J. Zhang, F. Zheng, X. Sun, and W.-S.
Zheng. Learning 3d shape feature for texture-insensitive person reidentiﬁcation. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 8146–8155, 2021.
[3] Z. Chen, Y. Ge, and Q. Yue. Features reconstruction disentanglement
cloth-changing person re-identiﬁcation. In International Conference
on Intelligent Computing, 2024. URL https://api.semanticscholar.org/
CorpusID:271212520.

714

S. Huang and H. Ling / Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification

[4] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong,
Q. Zhang, X. Zhu, L. Lu, et al. Internvl: Scaling up vision foundation
models and aligning for generic visual-linguistic tasks. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.
[5] Z. Cui, J. Zhou, Y. Peng, S. Zhang, and Y. Wang. Dcr-reid: Deep
component reconstruction for cloth-changing person re-identiﬁcation.
IEEE Transactions on Circuits and Systems for Video Technology,
33:4415–4428, 2023. URL https://api.semanticscholar.org/CorpusID:
256595882.
[6] Y. Ding, R. Mao, H. Zhu, A. Wang, and L. Zhang. Discriminative pedestrian features and gated channel attention for clothes-changing person
re-identiﬁcation. 2024 IEEE International Conference on Multimedia
and Expo (ICME), pages 1–6, 2024. URL https://api.semanticscholar.
org/CorpusID:273008610.
[7] Y. Fang, Q. Sun, X. Wang, T. Huang, X. Wang, and Y. Cao. Eva-02:
A visual representation for neon genesis. Image and Vision Computing,
149:105171, 2024.
[8] D. Gray and H. Tao. Viewpoint invariant pedestrian recognition with an
ensemble of localized features. In Computer Vision–ECCV 2008: 10th
European Conference on Computer Vision, Marseille, France, October
12-18, 2008, Proceedings, Part I 10, pages 262–275. Springer, 2008.
[9] X. Gu, H. Chang, B. Ma, S. Bai, S. Shan, and X. Chen. Clotheschanging person re-identiﬁcation with rgb modality only. 2022
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), pages 1050–1059, 2022. URL https://api.semanticscholar.org/
CorpusID:248177891.
[10] X. Gu, H. Chang, B. Ma, S. Bai, S. Shan, and X. Chen. Clotheschanging person re-identiﬁcation with rgb modality only. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1060–1069, 2022.
[11] K. Han, S. Gong, Y. Huang, L. Wang, and T.-P. Tan. Clothingchange feature augmentation for person re-identiﬁcation. URL https:
//api.semanticscholar.org/CorpusID:260068421.
[12] S. He, H. Luo, P. Wang, F. Wang, H. Li, and W. Jiang. Transreid: Transformer-based object re-identiﬁcation. In Proceedings of the
IEEE/CVF international conference on computer vision, pages 15013–
15022, 2021.
[13] P. Hong, T. Wu, A. Wu, X. Han, and W.-S. Zheng. Fine-grained shapeappearance mutual learning for cloth-changing person re-identiﬁcation.
In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 10513–10522, 2021.
[14] R. Hou, B. Ma, H. Chang, X. Gu, S. Shan, and X. Chen. Interaction-andaggregation network for person re-identiﬁcation. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages
9309–9318, 2019.
URL https://api.semanticscholar.org/CorpusID:
197935526.
[15] Y. Huang, Q. Wu, J. Xu, and Y. Zhong. Celebrities-reid: A benchmark
for clothes variation in long-term person re-identiﬁcation. In 2019 International Joint Conference on Neural Networks (IJCNN), pages 1–8,
2019. doi: 10.1109/IJCNN.2019.8851957.
[16] Y. Huang, Q. Wu, Z. Zhang, C. Shan, Y. Zhong, and L. Wang. Meta
clothing status calibration for long-term person re-identiﬁcation. IEEE
Transactions on Image Processing, 2024.
[17] X. Jin, T. He, K. Zheng, Z. Yin, X. Shen, Z. Huang, R. Feng, J. Huang,
Z. Chen, and X.-S. Hua. Cloth-changing person re-identiﬁcation from a
single image with gait prediction and regularization. In Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
pages 14278–14287, 2022.
[18] P. Li, Y. Xu, Y. Wei, and Y. Yang. Self-correction for human parsing.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 44
(6):3260–3271, 2020.
[19] S. Li, J. Leng, G. Li, J. Gan, H. Chen, and X. Gao. Clip-driven clothagnostic feature learning for cloth-changing person re-identiﬁcation.
ArXiv, abs/2406.09198, 2024. URL https://api.semanticscholar.org/
CorpusID:270440427.
[20] W. Li, X. Zhu, and S. Gong. Harmonious attention network for person re-identiﬁcation. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2285–2294, 2018. URL https:
//api.semanticscholar.org/CorpusID:3458516.
[21] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. Advances
in neural information processing systems, 36, 2024.
[22] C. Luo, C. Song, and Z. Zhang. Learning to adapt across dual discrepancy for cross-domain person re-identiﬁcation. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 45(2):1963–1980, 2022.
[23] H. Luo, Y. Gu, X. Liao, S. Lai, and W. Jiang. Bag of tricks and a
strong baseline for deep person re-identiﬁcation. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops,

June 2019.
[24] C. Peng, B. Wang, D. Liu, N. Wang, R. Hu, and X. Gao.
Masked attribute description embedding for cloth-changing person reidentiﬁcation. IEEE Transactions on Multimedia, 27:1475–1485, 2025.
doi: 10.1109/TMM.2024.3521730.
[25] X. Qian, W. Wang, L. Zhang, F. Zhu, Y. Fu, T. Xiang, Y.-G. Jiang,
and X. Xue.
Long-term cloth-changing person re-identiﬁcation.
In Asian Conference on Computer Vision, 2020. URL https://api.
semanticscholar.org/CorpusID:218889592.
[26] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable
visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
[27] Y. Sun, L. Zheng, Y. Yang, Q. Tian, and S. Wang. Beyond part models: Person retrieval with reﬁned part pooling. In European Conference on Computer Vision, 2017. URL https://api.semanticscholar.org/
CorpusID:10013306.
[28] Q.-H. Trinh, N.-T. Bui, P.-T. V. Thi, H.-D. Nguyen, D. Jha, U. Bagci,
and M. Tran. Pgds: Pose-guidance deep supervision for mitigating clothes-changing in person re-identiﬁcation. 2024 IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS), pages 1–8, 2023. URL https://api.semanticscholar.org/
CorpusID:266163303.
[29] F. Wan, Y. Wu, X. Qian, and Y. Fu. When person re-identiﬁcation
meets changing clothes. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 3620–3628,
2020. URL https://api.semanticscholar.org/CorpusID:212633939.
[30] Q. Wang, X. Qian, B. Li, X. Xue, and Y. Fu. Exploring ﬁnegrained representation and recomposition for cloth-changing person reidentiﬁcation. IEEE Transactions on Information Forensics and Security, 19:6280–6292, 2024.
[31] Q. Wang, X. Qian, B. Li, L. Chen, Y. Fu, and X. Xue. Content and salient semantics collaboration for cloth-changing person reidentiﬁcation. In ICASSP 2025-2025 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE,
2025.
[32] L. Wei, S. Zhang, W. Gao, and Q. Tian. Person transfer gan to bridge
domain gap for person re-identiﬁcation. In Proceedings of the IEEE
conference on computer vision and pattern recognition, pages 79–88,
2018.
[33] P. Xu and X. Zhu. Deepchange: A long-term person re-identiﬁcation
benchmark with clothes change. In Proceedings of the IEEE international conference on computer vision (ICCV), 2023.
[34] L. Xue, M. Shu, A. Awadalla, J. Wang, A. Yan, S. Purushwalkam,
H. Zhou, V. Prabhu, Y. Dai, M. S. Ryoo, et al. xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872,
2024.
[35] Q. Yang, A. Wu, and W.-S. Zheng. Person re-identiﬁcation by contour
sketch under moderate clothing change. IEEE transactions on pattern
analysis and machine intelligence, 43(6):2029–2046, 2019.
[36] Z. Yang, M. Lin, X. Zhong, Y. Wu, and Z. Wang. Good is bad: Causality inspired cloth-debiasing for cloth-changing person re-identiﬁcation.
2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1472–1481, 2023. URL https://api.semanticscholar.
org/CorpusID:260844125.
[37] Z. Yang, X. Zhong, Z. Zhong, H. Liu, Z. Wang, and S. Satoh. Win-win
by competition: Auxiliary-free cloth-changing person re-identiﬁcation.
IEEE Transactions on Image Processing, 32:2985–2999, 2023. URL
https://api.semanticscholar.org/CorpusID:258843841.
[38] G. Zhang, J. Liu, Y. Chen, Y. Zheng, and H. Zhang. Multi-biometric
uniﬁed network for cloth-changing person re-identiﬁcation. 2022 IEEE
International Conference on Multimedia and Expo (ICME), pages 01–
06, 2022. URL https://api.semanticscholar.org/CorpusID:251848165.
[39] Z. Zhang, C. Lan, W. Zeng, X. Jin, and Z. Chen. Relation-aware global
attention for person re-identiﬁcation. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 3186–3195,
2020.
[40] Y. Zhao, C. Wu, Y. Xu, X. Du, R. Li, and G. Niu. Ccup: A controllable
synthetic data generation pipeline for pretraining cloth-changing person
re-identiﬁcation models. ArXiv, abs/2410.13567, 2024. URL https:
//api.semanticscholar.org/CorpusID:273404183.
[41] K. Zheng, X. He, and X. E. Wang. Minigpt-5: Interleaved vision-andlanguage generation via generative vokens, 2023.
[42] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language
models. arXiv preprint arXiv:2304.10592, 2023.


 succeeded in 0ms:
ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 979-8-3503-6874-1/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICASSP49660.2025.10889622

Multi Queue for Unsupervised Person
Re-identification
Zhenyuan Lin †

Shengyong Xie †

Danhua Liu *

Guangzhou Institute of Technology School of Computer Science and information Security School of Artificial Intelligence
Xidian University
Guilin University Of Electronic Technology
Xidian University
Xian, China
Guilin, China
Guangzhou, China
dhliu@xidian.edu.cn
2416057194@qq.com
linzhenyuan@stu.xidian.edu.cn

Weikun Li

Ang Gao

Yubo Dong

School of Computer Science and information Security
Guilin University Of Electronic Technology
Guilin, China
liweikun1105 @163.com

School of Artificial Intelligence
Xidian University
Xian, China
anggao@stu.xidian.edu.cn

School of Artificial Intelligence
Xidian University
Xian, China
ybdong@stu.xidian.edu.cn

Abstract—Recently, cluster-based methods have achieved significant success in unsupervised re-ID tasks. The hierarchical
clustering algorithm, exemplified by SpCL, has been widely
adopted in unsupervised cross-domain adaptation and unsupervised learning. The momentum-based feature update mechanism
in SpCL has been integrated into various algorithms, achieving
notable results in subsequent studies. In this paper, we propose
a multi-queue feature updating algorithm that stores feature
vectors corresponding to person IDs in multiple queues. Random
sampling is then applied to construct the negative sample matrix
for contrastive loss, addressing the limitations of momentumbased updating methods. Additionally, we replace the static
temperature coefficient in contrastive loss with a trainable temperature coefficient, enabling the model to automatically balance
sensitivity between easy and hard samples. The code is available
at https://github.com/bmfarer/multi-queue.git.
Index Terms—unsupervised re-ID tasks; hierarchical clustering algorithm;updating feature vectors; multi queue features
updating algorithm;

I. I NTRODUCTION
The unsupervised person re-identification (re-ID) task is
particularly challenging due to the need to distinguish finegrained visual differences between individuals [1] and manage long-tailed data distributions [2], where certain identities
dominate the dataset. These challenges hinder the direct application of unsupervised classification algorithms, requiring
the development of specialized techniques to learn effective
representations without labels .
Current approaches for training on unlabeled data can be
broadly categorized into two main strategies: unsupervised
cross-domain adaptation (UDA) and unsupervised learning
(USL). UDA transfers knowledge from a labeled source domain to an unlabeled target domain, leveraging existing labeled
data to enhance performance in the target domain. In contrast,
USL methods aim to extract useful information directly from
the unlabeled data of the target domain, building effective
† : Equal contribution, * : Corresponding author
This work was supported by the Natural Science Foundation (NSF) of
China (No. 62476206, 62293483, 62101398), the National Key R&D Program
of China (2022YFB2902900)and the project of Pazhou Lab (Huangpu) (No.
2022K0904).

models without relying on external supervision or labeled
examples.
General UDA methods can be classified into generative style
transfer algorithms [3] and pseudo-label generation algorithms.
The latter includes mutual learning algorithms, which use
siamese networks to construct teacher-student models [4], [5],
K-nearest neighbor sampling algorithms [6], [7], and clusterbased algorithms [8]–[10].
Compared to UDA, USL is more challenging to train directly on unlabeled data due to the lack of pretraining support
from source domain data. MMCL [7] generates pseudo labels
through clustering and stores historical features in a memory
bank for learning. PIL [11] also generates pseudo-labels, but
uses linear interpolation to place these labels at the centers of
two clusters, guiding the learning process.
Cluster-based contrastive learning algorithms are widely
employed in both UDA [9], [12], [13] and USL [9], [14]. These
methods typically utilize InfoNCE [15] as the foundational
loss function and incorporate a memory bank [16] to store
negative sample features. This memory bank is often used to
refine pseudo labels in various works.
Therefore, this paper makes the following improvements to
memory bank and momentum update (Fig.1):
• Multi-queue is applied to store and manage historical
features.
• The features updating method of momentum updating is
removed and the operations of queue are used to update
features of the memory bank;
• Replace the static temperature coefficient InfoNCE [15]
with trainable temperature coefficient, so that the algorithm can automatically balance the attention of difficult
samples in the training process.
II. R ELATE W ORK
Self-supervised learning. Most self-supervised learning
methods primarily use contrastive loss [15], [19] as the core
loss function. These include siamese networks for constructing dissimilarities, step-by-step model representation learning
during asymmetric training [20], [21], clustering algorithms

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:42 UTC from IEEE Xplore. Restrictions apply.

Features
Enqueue

Cluster
Algorithm

F(θ)

Encoder
Unlabeled
Dataset

Pseudo Labels
and features

Features

Multi Queue
Memory Bank

Stage 1：Cluster Features

Pseudo Labels
Enqueue

Resample

Contrastive
Loss

F(θ)

Sample
features

Backward
Encoder
Pseudo Labels
Dataset

Features

Backward

Stage 2：Train model with pseudo labels

Fig. 1. Architecture of the multi-queue algorithm. The proposed algorithm is divided into two stages. In the first stage, pseudo labels are generated by
clustering algorithm, which are placed in the memory bank and the corresponding pseudo-labels are affixed to the image. In the second stage, based on the
pseudo labels id, the classification sampling is performed and a feature is taken from each id queue in the memory bank to form a negative sample feature
matrix K, which is computed and compared with the feature Q predicted by the current round and backpropagated. After that, Q is put into the corresponding
id queue according to the pseudo labels.

for generating pseudo labels [22], memory banks for storing
negative sample features [16], and momentum updates for
refreshing frozen encoders with large negative sample dictionaries to aid encoder learning [23]. However, due to the specific characteristics of person re-ID datasets, self-supervised
learning methods tend to experience model collapse [26], [36]
when directly applied to unsupervised re-ID. Therefore, these
methods are not directly applicable to the tasks addressed in
this paper.
Unsupervised re-ID. BUC [37] introduces a bottom-up
hierarchical clustering approach that progressively merges
smaller clusters into larger, more coherent ones, thereby
enhancing identity representation through improved cluster
quality. CycAs [38] and TSSL [39] improve pseudo label
accuracy by integrating continuous tracking predictions over
time, which enhances the consistency and reliability of identity
assignment. CCE [6] employs a soft similarity mechanism
to learn robust image representations across different camera views, facilitating better cross-camera identification and
minimizing discrepancies between camera-specific features.
Cluster-based contrastive algorithms [9], [12]–[14] modify
the InfoNCE [15] loss function by incorporating a memory
bank to store negative samples, updating these samples using
a momentum-based approach, and applying clustering algorithms to generate accurate pseudo labels.
III. A NALYSIS AND M ETHODS
A. Preliminary
Suppose that for a single sample, the predicted vector is
q and the sequence of negative sample matrices drawn from
the memory bank is K = {k1 , k2 , ..., kc }, in which c is the
number of pseudo label classes generated by the clustering.
Denote k+ as the positive feature vector corresponding to q
taken from the memory bank, then InfoNCE is as in Eq.1:

features

𝐿𝑚𝑎𝑥

Multi queue memory bank

Fig. 2. Methods for initializing and updating a multi queue. After the features
are obtained, they are separated according to the pseudo labels and enqueue
to cover the old features.

exp(q·k+ /τ )
Lq = −log PK
i=0 exp(q·ki /τ )

(1)

B. Multi Queue
At the beginning of each epoch, the extracted features
are first enqueue according to the pseudo labels. Denote the
maximum length of the queue to Lmax and the current length
of the queue L. If L > Lmax , discard redundant features; Els
enqueue the features. Since the sampling method employed
in this algorithm is the id-resampling method [14], this is to
ensure the basic stability of the queue length so that historical
features can be dequeueed in a timely manner and avoid them
from being resampled, which can reduce the overfitting of the
model.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:42 UTC from IEEE Xplore. Restrictions apply.

C. Trainable Temperature InfoNCE
Based on InfoNCE [15], since the multi queues method for
constructing memory bank is available in this algorithm. If
only update the parameters of the model with features from
the memory bank, the model will be prone to overfitting,
especially on low difficulty datasets. Therefore, in this paper,
we propose to use variable temperature to balance the sample
difficulty and suppress model overfitting. Following Eq.1, the
basic skeleton of the contrastive loss function for variable
temperature is shown in Eq.1. But the τ is a variable parameter.
In the experiments of our work, Adam is used to update the
parameters, so the updated formula for the τ is given as Eq.2:
Fig. 3. Ablation of Queue Length Maximum.

λ
m̂t
vˆt + ϵ

(2)

m̂t =

mt
1 − β1t

(3)

vˆt =

vt
1 − β2t

(4)

τ ←τ−√

λ = lα

100 and 400, respectively. Adam is the optimizer with a weight
decay factor of 5e-4. Step learning rate decay is adjusted every
20 epochs with a decay rate of 0.2. The number of re-samples
per ID and the length and number of re-samples for a single
queue are set to 16. The hyperparameters of DBSCAN and
Infomap are the same as those of Cluster Contrast [14].

(5)

B. Comparison with Other USL Methods.
l is the total learning rate and α is a multiple of the learning
rate, which is used to automatically adjust learning rates of
τ . The reason why λ = l is not set directly is that, if λ =
l, then the gradient update of the model to the temperature
will generate redundant label noise due to the excessive speed,
which will affect the optimization of the model. Meanwhile,
making λ ̸= l properly keeps the learning rate of model and τ
out of sync so that the model can find more possible solutions.

The results of the horizontal comparison are given in Tab.II.
From the experimental results, Market1501 [33],
DukeMTMC-reID [35] and PersonX [36] have achieved
better results than other algorithms, and the mAP is 0.2%,
0.4% and 1.4% higher than the highest algorithm in the
corresponding dataset, respectively. On MSMT17 [34],
however, the effect is slightly lower than that of the highest
algorithm, named Cluster Contrast [14], at -0.5%.

IV. E XPERIMENT
C. Ablation Experiments
TABLE I
S TATISTICS OF DATASETS USED IN THE EXPERIMENTAL SECTION .
Dataset Object
MSMT17
PersonX
Market-1501
DukeMTMC-reID

train IDs
1,041
410
751
702

train images
62,621
9,840
12,936
16,522

test IDs
3,060
856
750
702

query images
11,659
5,136
3,368
2,228

total images
126,441
45,792
32,668
36,441

cameras
15
6
6
8

To verify the robustness of the proposed algorithm, we
evaluate our mothod on Market1501 [33], MSMT17 [34],
DukeMTMC [35] and Personx [36], whose details are given
in Tab.I. The test setup and environment are two GTX1080s
with CUDA 11.3.
A. Implementation Details.
In this paper, Resnet50 [37] pre-trained by Imagenet [38] is
applied as the experimental model, modules from first to four
adopted, the Average Pooling layer between each two adjacent
modules removed, and the GAP (Global Average Pooling) [14]
adopted at the tail of the model. DBSCAN [32] and Infomap
[26] are used to test the effect of the clustering algorithm.
The image input size for this experiment is 256 × 128. The
employed data augmentation are random horizontal flipping,
10-pixel padding, and random erasing [39].The batch size,
learning rate, epochs and iterations are set to 128, 1.75e-4,

The baseline implementation details as shown in Sec.IV-A.
While different from the Sec.IV-B, the test setup and environment are four Titan-Xps with CUDA 10.2.
1) Queue Length: The MSMT17 dataset was utilized for
the ablation experiment conducted on the cohort. Initially,
the maximum queue lengths were established at 4, 8, 12,
16, 20, 24, and 28, respectively. Analysis of Figure 3 reveals that within the range of 4 ≤ Lmax ≤ 20, the mAP
values consistently increase; conversely, within the range of
20 ≤ Lmax ≤ 28, the mAP values initially rise before
subsequently declining. It is noted that 16 corresponds to the
resampling frequency, indicating that each time pedestrian data
associated with ID B is resampled, the 16 feature vectors of
B within the multi-queue are updated. Consequently, when
Lmax ≤ 16, the queue exclusively contains features derived
from the most recent resampling (new features), whereas when
Lmax > 16, the queue incorporates features from the previous
two resampling events (old features). The data suggest that
retaining a portion of old features within the queue enhances
the model’s representational capacity.
2) Wether apply trainable temperature: Initially, we performed ablation experiments on the datasets Market1501,
DukeMTMC-reID, and MSMT17, both with and without a

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:42 UTC from IEEE Xplore. Restrictions apply.

TABLE II
E XPERIMENTS ON M ARKET 1501, D UKE MTMC- RE ID, MSMT17, P ERSON X DATASETS

Methods
BUC [37]
MMCL [7]
HCT [40]
MMT [4]
SPCL [9]
RLCC [41]
MCRN [12]
Cluster Contrast [14]
Ours/Dbscan
Ours/Infomap
Methods
MMT [4]
SPCL [9]
Cluster Contrast [14]
Ours/Dbscan
Ours/Infomap

Source
None
None
None
MSMT17
None
None
None
None
None
None
Source
Market
None
None
None
None

Market1501
mAP top-1
38.3
66.2
45.5
80.3
56.4
80.0
75.6
89.3
73.1
88.1
77.7
90.8
80.8
92.5
82.6
92.4
82.8
92.8
82.5
92.4
MSMT17
mAP top-1
24.0
50.1
17.2
40.6
35.5
65.9
34.7
64.7
35.0
65.1

top-5
79.6
89.4
91.6
95.8
95.1
96.3
96.7
96.9
97.1

top-10
84.5
92.3
95.2
97.5
97.0
97.5
97.8
98.0
98.2

Source
None
None
None
Market
None
None
None
None
None
None

top-5
63.5
53.6
76.6
75.4
75.7

top-10
69.3
59.2
80.4
79.3
79.6

Source
Market
Market
None
None
None

DukeMTMC-reID
mAP top-1
top-5
27.5
47.4
62.6
51.4
72.4
82.9
50.7
69.6
83.4
65.1
78.9
88.8
65.3
81.2
90.3
69.2
83.2
91.6
69.9
83.5
72.8
84.6
91.3
72.0
84.6
91.9
73.3
85.6
92.4
PersonX
mAP top-1
top-5
78.9
90.6
96.8
78.5
91.1
97.8
86.6
95.3
98.8
87.2
95.5
98.9
88.0
95.5
98.9

top-10
68.4
85.0
87.4
92.5
92.2
93.8
93.6
94.1
94.2
top-10
98.2
99.0
99.5
99.5
99.5

TABLE III
A BLATION STUDY BY OBSERVING WETHER APPLY TRAINABLE
TEMPERATURE . AT THE TEMP, THE ”✓” MEANS APPLY THE TRAINABLE
TEMPERATURE

Cluster
Dbscan
Infomap
Dbscan
Infomap
Cluster
Dbscan
Infomap
Dbscan
Infomap
Cluster
Dbscan
Infomap
Dbscan
Infomap

temp
✓
✓
temp
✓
✓
temp
✓
✓

Market1501
top-1
top-5
92.7
97.1
92.5
96.8
93.3
97.6
92.3
96.8
MSMT17
mAP top-1
top-5
26.1
54.3
64.6
29.8
60
70.6
34.4
64.8
75.2
31.1
61.1
71.9
DukeMTMC-reID
mAP top-1
top-5
73.1
73.1
91.7
72.5
85.5
91.8
72.7
85.1
91.9
74
86
92.1
mAP
82.8
82.4
83.3
81.7

top-10
98
97.9
98.4
97.7
top-10
69.3
74.9
79.1
76.1
top-10
93.5
93.4
93.9
94

trainable temperature coefficient, resulting in the data presented in Tables III . The results indicate that the implementation of a trainable temperature coefficient yields superior
performance compared to its absence, particularly evident in
the more challenging dataset, MSMT17.
Furthermore, as illustrated in Figure 4, the trainable temperature coefficient exhibits a decreasing trend across all datasets
throughout the training process. This trend suggests that the
model increasingly focuses on more challenging samples as

Fig. 4. The graph illustrates variations in temperature over time, with the
horizontal axis representing epochs and the vertical axis denoting temperature.

its representational capabilities enhance. The datasets can
be ranked in terms of difficulty as follows: Market1501 <
DukeMTMC-reID < MSMT17. Notably, the more challenging the dataset, the more rapidly the temperature coefficient
declines, which correlates with an increased model focus
on difficult samples. This adaptive adjustment allows the
model to engage in targeted learning, thereby enhancing its
representational proficiency.
V. C ONCLUSION
In this paper, we propose a multi-queue based USL method,
which achieves better results than previous algorithms on mentioned datasets. In the next paper, we will mention long-tail
classification [24], [43], [45], fine-grained image classification
[46], and greedy algorithms, among others, to address the
problems of inadequate representation, imbalanced data distribution, and immaturity due to difficult sample management.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:42 UTC from IEEE Xplore. Restrictions apply.

R EFERENCES
[1] Bukchin G, Schwartz E, Saenko K, et al. Fine-grained angular contrastive learning with coarse labels[C]//Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 2021: 87308740.
[2] Wang P, Han K, Wei X S, et al. Contrastive learning based hybrid
networks for long-tailed image classification[C]//Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition. 2021:
943-952.
[3] Ge Y, Li Z, Zhao H, et al. Fd-gan: Pose-guided feature distilling gan
for robust person re-identification[J]. Advances in neural information
processing systems, 2018, 31.
[4] Ge Y, Chen D, Li H. Mutual mean-teaching: Pseudo label refinery for
unsupervised domain adaptation on person re-identification[J]. arXiv
preprint arXiv:2001.01526, 2020.
[5] Ge Y, Yu S, Chen D. Improved mutual mean-teaching for unsupervised
domain adaptive re-ID[J]. arXiv preprint arXiv:2008.10313, 2020.
[6] Lin Y, Xie L, Wu Y, et al. Unsupervised person re-identification via
softened similarity learning[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 3390-3399.
[7] Wang D, Zhang S. Unsupervised person re-identification via multi-label
classification[C]//Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition. 2020: 10981-10990.
[8] Zhai Y, Lu S, Ye Q, et al. Ad-cluster: Augmented discriminative
clustering for domain adaptive person re-identification[C]//Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition.
2020: 9021-9030.
[9] Ge Y, Zhu F, Chen D, et al. Self-paced contrastive learning with
hybrid memory for domain adaptive object re-id[J]. Advances in neural
information processing systems, 2020, 33: 11309-11321.
[10] Fu Y, Wei Y, Wang G, et al. Self-similarity grouping: A simple unsupervised cross domain adaptation approach for person reidentification[C]//proceedings of the IEEE/CVF international conference
on computer vision. 2019: 6112-6121.
[11] Zhang X, Li D, Wang Z, et al. Implicit sample extension for unsupervised person re-identification[C]//Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 2022: 73697378.
[12] Wu Y, Huang T, Yao H, et al. Multi-centroid representation network for
domain adaptive person re-id[C]//Proceedings of the AAAI conference
on artificial intelligence. 2022, 36(3): 2750-2758.
[13] Zheng K, Liu W, He L, et al. Group-aware label transfer for domain
adaptive person re-identification[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 5310-5319.
[14] Dai Z, Wang G, Yuan W, et al. Cluster contrast for unsupervised person
re-identification[C]//Proceedings of the Asian conference on computer
vision. 2022: 1142-1160.
[15] Oord A, Li Y, Vinyals O. Representation learning with contrastive
predictive coding[J]. arXiv preprint arXiv:1807.03748, 2018.
[16] Wu Z, Xiong Y, Yu S X, et al. Unsupervised feature learning via
non-parametric instance discrimination[C]//Proceedings of the IEEE
conference on computer vision and pattern recognition. 2018: 37333742.
[17] Baldi P, Sadowski P J. Understanding dropout[J]. Advances in neural
information processing systems, 2013, 26.
[18] Larsson G, Maire M, Shakhnarovich G. Fractalnet: Ultra-deep neural
networks without residuals[J]. arXiv preprint arXiv:1605.07648, 2016.
[19] Gutmann M, Hyvärinen A. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models[C]//Proceedings of the
thirteenth international conference on artificial intelligence and statistics.
JMLR Workshop and Conference Proceedings, 2010: 297-304.
[20] Chen T, Kornblith S, Swersky K, et al. Big self-supervised models
are strong semi-supervised learners[J]. Advances in neural information
processing systems, 2020, 33: 22243-22255.
[21] Chen T, Kornblith S, Norouzi M, et al. A simple framework for contrastive learning of visual representations[C]//International conference
on machine learning. PMLR, 2020: 1597-1607.
[22] Caron M, Bojanowski P, Joulin A, et al. Deep clustering for unsupervised
learning of visual features[C]//Proceedings of the European conference
on computer vision (ECCV). 2018: 132-149.
[23] He K, Fan H, Wu Y, et al. Momentum contrast for unsupervised visual
representation learning[C]//Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition. 2020: 9729-9738.

[24] He K, Chen X, Xie S, et al. Masked autoencoders are scalable vision
learners[C]//Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition. 2022: 16000-16009.
[25] Chen X, He K. Exploring simple siamese representation learning[C]//Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition. 2021: 15750-15758.
[26] Papyan V, Han X Y, Donoho D L. Prevalence of neural collapse during
the terminal phase of deep learning training[J]. Proceedings of the
National Academy of Sciences, 2020, 117(40): 24652-24663.
[27] Graf F, Hofer C, Niethammer M, et al. Dissecting supervised contrastive
learning[C]//International Conference on Machine Learning. PMLR,
2021: 3821-3830.
[28] Lin Y, Dong X, Zheng L, et al. A bottom-up clustering approach
to unsupervised person re-identification[C]//Proceedings of the AAAI
conference on artificial intelligence. 2019, 33(01): 8738-8745.
[29] Wang Z, Zhang J, Zheng L, et al. Cycas: Self-supervised cycle
association for learning re-identifiable descriptions[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August
23–28, 2020, Proceedings, Part XI 16. Springer International Publishing,
2020: 72-88.
[30] Wu G, Zhu X, Gong S. Tracklet self-supervised learning for unsupervised person re-identification[C]//Proceedings of the AAAI Conference
on Artificial Intelligence. 2020, 34(07): 12362-12369.
[31] Rosvall M, Bergstrom C T. Maps of random walks on complex networks
reveal community structure[J]. Proceedings of the national academy of
sciences, 2008, 105(4): 1118-1123.
[32] Ester M, Kriegel H P, Sander J, et al. A density-based algorithm for
discovering clusters in large spatial databases with noise[C]//kdd. 1996,
96(34): 226-231.
[33] Zheng L, Shen L, Tian L, et al. Person re-identification meets image
search[J]. arXiv preprint arXiv:1502.02171, 2015.
[34] Wei L, Zhang S, Gao W, et al. Person transfer gan to bridge domain
gap for person re-identification[C]//Proceedings of the IEEE conference
on computer vision and pattern recognition. 2018: 79-88.
[35] Ristani E, Solera F, Zou R, et al. Performance measures and a data
set for multi-target, multi-camera tracking[C]//European conference on
computer vision. Cham: Springer International Publishing, 2016: 17-35.
[36] Sun X, Zheng L. Dissecting person re-identification from the viewpoint
of viewpoint[C]//Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition. 2019: 608-617.
[37] He K, Zhang X, Ren S, et al. Deep residual learning for image
recognition[C]//Proceedings of the IEEE conference on computer vision
and pattern recognition. 2016: 770-778.
[38] Deng J, Dong W, Socher R, et al. Imagenet: A large-scale hierarchical
image database[C]//2009 IEEE conference on computer vision and
pattern recognition. Ieee, 2009: 248-255.
[39] Zhong Z, Zheng L, Kang G, et al. Random erasing data augmentation[C]//Proceedings of the AAAI conference on artificial intelligence.
2020, 34(07): 13001-13008.
[40] Zeng K, Ning M, Wang Y, et al. Hierarchical clustering with hard-batch
triplet loss for person re-identification[C]//Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 2020: 1365713665.
[41] Zhang X, Ge Y, Qiao Y, et al. Refining pseudo labels with
clustering consensus over generations for unsupervised object reidentification[C]//Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition. 2021: 3436-3445.
[42] Cho Y, Kim W J, Hong S, et al. Part-based pseudo label refinement for
unsupervised person re-identification[C]//Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 2022: 73087318.
[43] Alshammari S, Wang Y X, Ramanan D, et al. Long-tailed recognition
via weight balancing[C]//Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition. 2022: 6897-6907.
[44] Kang B, Xie S, Rohrbach M, et al. Decoupling representation and classifier for long-tailed recognition[J]. arXiv preprint arXiv:1910.09217,
2019.
[45] Zhu J, Wang Z, Chen J, et al. Balanced contrastive learning for longtailed visual recognition[C]//Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. 2022: 6908-6917.
[46] Yang X, Wang Y, Chen K, et al. Fine-grained object classification
via self-supervised pose alignment[C]//Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 2022: 73997408.

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:52:42 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-feature balanced network for clothes-changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 190 (2025) 107617

Contents lists available at ScienceDirect

Neural Networks
journal homepage: www.elsevier.com/locate/neunet

Full Length Article

Multi-feature balanced network for clothes-changing person
re-identificationI
Mengqing Mei a,b , Chun Ye a , Zhiwei Ye a,b
Liye Mei a,b , Ziyuan Chen a
a

,∗, Fangyi Liu c , Mang Ye c , LingYu Yan a,b

,

School of Computer Science, Hubei University of Technology, Wuhan, 430068, China

b Hubei Key Laboratory of Green Intelligent Computing Power Network, Hubei University of Technology, Wuhan, 430068, China
c School of Computer Science, Wuhan University, Wuhan 430072, China

ARTICLE

INFO

Keywords:
Deep learning
Machine learning
Robustness
Cloth-changing person re-identification
Fine-grained

ABSTRACT
Clothes-changing person re-identification (CC-ReID) is pivotal in long-term scenes, especially when involving
significant variations in clothes. The principal challenge of this task lies in the extraction of clothes-irrelevant
features. Currently, most methods alleviate the interference caused by clothing changes through separating
and identifying body regions. However, these approaches could not fully utilize all the useful information in
pedestrian images. In this work, a novel Multi-features Balanced Network (MBNet) is proposed for improving
the robustness of the CC-ReID model by exploiting clothing-unrelated features, consisting of a global branch, a
clothing-unrelated branch, and a mask branch. Specifically, to highlight clothing-unrelated clues, a knowledge
transfer module (KTM) is first designed. Then, the clothing-unrelated branch only receives images that are
unaffected or less affected clothing to resist clothes-changing. Besides, a feature attention module (FAM) is
introduced in this branch, which can suppress background clutter and extract discriminative fine-grained
features. Finally, a cross fusion module (CFM) is used to integrate more contextual information and mine
more clothing-independent and pose features in the mask branch. Three branches are combined to perform
CC-ReID. Extensive experiments on three popular synthetic and realistic datasets show that the superiority of
the proposed approach, achieving a Rank-1/mAP accuracy of 44.6%/22.7%, 58.3%/57.9%, 87.2%/84.0%.

1. Introduction
Person re-identification (ReID) (Ye et al., 2022) is an image retrieval
task aimed at matching target pedestrians across different viewpoints
and multiple cameras to find the movement trajectory of pedestrians,
which has been broadly applied in real-life scenarios, including suspect tracking, searching missing persons, unmanned supermarkets, and
autonomous driving. In recent years, the rise of deep learning has
propelled the rapid development of ReID methods, making it a research
hotspot. Most previous ReID methods mainly match target pedestrians
in a short-term scenario. However, these methods overlook an important issue: the possibility of pedestrians changing their clothing. In a
realistic long-term scenario, pedestrian often changes their clothes at
different times, making the clothing appearance information tends to
be unreliable. Compared to short-term person ReID, clothes-changing
person ReID (CC-ReID) is suitable for more complex scenarios and is

more flexible. Therefore, in the paper, a more realistic and practical
clothes-changing person ReID task is concerned.
Most existing ReID methods (Yang, Liu et al., 2023; Zhong et al.,
2024) have primarily addressed conventional challenges. These challenges include overcoming occlusions, visible infrared scenarios (Qian,
Lin, & Du, 2025), camera view variations, and pose changes. However,
when confronted with scenarios involving clothing changes, traditional
ReID methods (Peng, Tang, Lin, & Zheng, 2024; Si, He, Li, & Gao,
2023; Ye et al., 2022) heavily rely on clothing-related information for
identification, which leads to a sharp decline in model performance
under the circumstance of clothes-changing. In addition, such information may mislead the model when people wear different clothes
or when different people wear similar clothes. Therefore, the clothing
appearance information is invalidated in this case. The main challenge
of cloth-changing ReID is to overcome the variation of clothes and
extract discriminative clothing-irrelevant features.

I This research was supported by National Natural Science Foundation of China (Grant Nos. 62376089, U23A20318, 62302154, 62472149), and Young and
Middle-aged Scientific and Technological Innovation Team Plan in Higher Education Institutions in Hubei Province, China (Grant No. T2023007).
∗ Corresponding author.
E-mail address: hgcsyzw@hbut.edu.cn (Z. Ye).

https://doi.org/10.1016/j.neunet.2025.107617
Received 23 November 2024; Received in revised form 13 February 2025; Accepted 11 May 2025
Available online 29 May 2025
0893-6080/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

Neural Networks 190 (2025) 107617

M. Mei et al.

To handle with the problems above, researchers have gradually
shifted their focus to cloth-changing ReID (Hong, Wu, Wu, Han, &
Zheng, 2021; Qian et al., 2020; Yang, Wu, & Zheng, 2021). The current
CC-ReID methods mainly pursue learning discriminative cloth-invariant
feature representations and could be roughly divided into two categories. The former (Liu, Kim, Gu, Jain, & Liu, 2023; Zhang, Liu, Chen,
Zheng, & Zhang, 2023) is based on additional auxiliary knowledge to
dig out cloth-irrelevant features, such as keypoints, 3D shapes, semantic
masks, and human parsing techniques to resist the interference caused
by clothing changes. For example, DeSKPro (Wu, Liu, Shi, Tang, &
Guo, 2022) used human parsing knowledge to eliminate the distraction
of clothing appearance, and mined identity-sensitive cues from the
recovered facial details. However, these facial features are not clear
enough due to the occlusion of masks or hats in surveillance videos.
Hence, it becomes impractical in real-world applications. The latter (Gu
et al., 2022; Hong et al., 2021; Xu et al., 2021) designs disentanglement strategies to separate clothing-related and clothing-unrelated
features from pedestrian images, in order to mitigate the impact of
clothing changes. It includes using Generative Adversarial Networks
(GAN) (Wang, Zhang, Lu, Fang and Wang, 2022; Yu et al., 2022) or randomly changing the pixels of clothing regions (Shu, Li, Wang, Ruan, &
Tian, 2021) to force the model to learn cloth-irrelevant features which
can reduce the differences in clothing color and texture information in
the image. For example, DG-Net (Zheng et al., 2019) proposed a joint
learning framework that utilizes discriminative and generative modules
separately, learning the appearance and structure codes to generate
high-quality cross-id composed images. However, ensuring the quality
of the generated images is challenging, and there is no guarantee that
the generated images will not damage cloth-irrelevant features.
In person re-identification, particularly in clothes-changing scenarios, relying on only one type of feature often yields poor results.
This is due to an over-reliance on a specific feature, which makes
the model susceptible to environmental changes such as lighting, occlusion, and perspective. For example, PCB (Part-based Convolutional
Baseline) (Sun, Zheng, Yang, Tian, & Wang, 2018) relied on features of
color and texture that showed significant performance degradation in
clothes-changing scenarios due to its sensitivity to clothing variations.
While appearance features provide basic identification information,
they may fail under varying lighting conditions or partial occlusion.
Similarly, the effectiveness of the posture feature is also limited when
the pose of the individual changes. These issues will indicate that
single-feature methods could result in a decrease in model identification accuracy when faced with complex factors such as perspective,
occlusion, and illumination. Zhang et al. (2023) pointed out that utilizing multiple biological features, including the head, shoulders, and
neck, can significantly alleviate the issue of clothing variation. Jin
et al. (2022) introduced gait recognition as auxiliary information by
leveraging person unique gait characteristics to drive the ReID model
to learn cloth-agnostic representations. Liu, Liu et al. (2024) proposed
using pose-guided attention learning to study and extract invariant
features of pedestrians under different conditions. Therefore, it is particularly important to explore how to effectively combine multiple
features to enhance robustness and accuracy of the model. In this
paper, a novel multi-feature balanced network (MBNet) is proposed
that combines multiple features such as head, leg, appearance, posture,
and background to enhance the performance of the clothes-changing
ReID model.
The proposed MBNet utilizes different source images as input, allowing it to learn invariant features under variations of the conditions.
By leveraging the complementary nature of multiple features and balancing them, MBNet can dynamically learn and adapt to different
condition changes. By merging the multimodal features from the input
image, more comprehensive information can be obtained in the same
scenario compared to methods that rely on a single feature. For example, head features can effectively distinguish individuals, while dynamic information about movements provided in pose features can play

an important role when static appearance features fail. In MBNet, different features are adjusted to ensure that the most favorable features
could be selected for identification in complex environments, independent of the failure of individual features. It includes a global branch, a
cloth-unrelated branch, and a mask branch, as shown in Fig. 1. Specifically, MBNet applies a knowledge transfer module in the global branch
to transfer the knowledge learned from the clothing-unrelated branch
to the global branch, which aims to emphasize clothing-unrelated areas.
To further extract fine-grained clothes-irrelevant features, the clothingunrelated branch uses cropped RGB images as input to focus on identity
information. Besides, MBNet adds a feature attention mechanism in
this branch to reduce the local noise and enhance the robustness of
clothes-changing problems. In addition, a background constraint loss
is embedded to suppress the learned background features, focusing on
the foreground features of pedestrians. To resist clothing changes, in the
mask branch, MBNet takes the silhouette mask as input to effectively
learn their body features. However, it overlooks important appearance
clues. To integrate more contextual information, MBNet introduces a
cross fusion module.
In summary, the main contributions of this paper could be outlined
as follows:
• A Multi-feature Balanced Network (MBNet) is proposed to learn
multiple clothing-irrelevant features to overcome the interference caused by cloth-changing. The core idea is to improve the
robustness of the ReID model by balancing multiple identity
features.
• To highlight clothing-unrelated areas, a knowledge transfer module (KTM) is proposed for resisting clothes-changing problems.
• A feature attention module (FAM) is proposed that reduces the
local noise and extracts more purely clothing independent features. Meanwhile, a background constraint loss is embedded to
strengthen identity-related features and resist the interference of
irrelevant features.
• A cross fusion module (CFM) is proposed that utilizes a twoway cross attention mechanism to integrate mask features and
clothing-irrelevant features of pedestrians to cope with situations
where different people wear similar clothing.
• More extensive evaluations of our method are conducted across
cloth-changing scenarios. The experimental results demonstrate
the effectiveness of our proposed framework MBNet performs well
on cloth-changing ReID datasets.
The rest of this paper is organized as follows. Section 2 surveys
recent work related to general person ReID and cloth-changing person
ReID. Section 3 describes the framework of the proposed method in
detail. Section 4 presents experimental results and analysis to show the
superiority of our method. Section 5 concludes the paper.
2. Related work
2.1. General person re-identification
Benefiting from the development of convolutional neural networks
in recent years, ReID methods have achieved considerable success
and widely explored. Existing person ReID mainly addressed some
traditional problems: overcoming occlusions (Li et al., 2021; Wang
et al., 2020), visible infrared scenarios (Ye, Shen, & Shao, 2021; Zhu
et al., 2024), camera view variations (Peng et al., 2024; Su, Zhang,
Xing, Gao, & Tian, 2016), and pose changes (Nguyen, Mantini, &
Shah, 2024). Gao, Wang, Lu, and Liu (2020) used pose-guided to
overcome occlusion and detected the non-occluded body parts and
then matched the correspondents separately, which aimed to exploit
more discriminative local features. Wang et al. (2020) viewed the
learned semantic features of key points in images as nodes of the
graph, and used graph convolutional layers to pass relation information
2

Neural Networks 190 (2025) 107617

M. Mei et al.

Fig. 1. The framework of our proposed Multi-feature Balance Network (MBNet). It consists of three branches:the global branch, the clothing-unrelated branch and the mask branch.
The global branch uses the Knowledge Transfer Module (KTM) which aims to transfer the clothes information from clothing-unrelated branch to this branch. The clothing-unrelated
branch uses the Feature Attention Module (FAM) to further enhance its ability to learn clothing-irrelevant fine-grained features. The mask branch uses the Cross Fusion Module
(CFM) to obtain the fusion feature 𝐹𝑎𝑝 .

between nodes, learning high-order information and topological information for discriminative features and robust alignment. Zhong et al.
(2022) utilized intermediate grayscale images as auxiliary information
to colorize the single-modality infrared images and eliminate crossmodality variations. Su et al. (2016) proposed a three-stage attribute
enhancement ReID method. First, they used a dataset labeled with
attributes for training, then fine-tuned the network using attributes and
identity labels, and finally fused the predicted attributes.
Although these methods have made significant progress, they heavily rely on clothing appearance information, which would become
ineffective or even misleading when faced with changing scenes.

neighbor structure in a two-branches network. Wu et al. (2022) proposed an effective identity-sensitive knowledge propagation framework
to mitigate the resolution degradation issues and learn cloth-irrelevant
features.
The second type is to separate clothing-related and clothingunrelated features from pedestrian images, encouraging the model
to learn clothes-invariant features. For example, Shu et al. (2021)
proposed a method of randomly changing clothing pixels. They used
a pre-trained human parsing model to obtain body parts and randomly
exchanged the pixels of clothes or pants among different pedestrians,
which makes the model less dependent on clothing colors and textures.
Yu et al. (2022) employed GAN to generate multiple images of the same
identity wearing different clothes, simulating a cloth-changing scene.
Hong et al. (2021) proposed a two-stream mutual learning framework
that utilizes shape and appearance features to explore body shape
cues of clothing changes. It adopts a dense interactive mutual learning
strategy, encouraging the appearance stream to learn fine-grained discriminative body shape knowledge from the shape stream. Cui, Zhou,
Peng, Zhang, and Wang (2023) proposed a novel disentanglement
framework that utilizes human component region reconstruction and
a deep assembled module to separate clothing-relevant and clothingirrelevant features. Gu et al. (2022) mined clothing-irrelevant features
from the original RGB images. They collected clothes labels and used
clothes classifiers to learn clothing-relevant features, which forced
the model to learn clothes-irrelevant features through the clothes
adversarial loss function. However, collecting clothes labels is very
time-consuming and not feasible in real life. Zhao et al. (2024) utilized
instance-level features from each identity sample to synthesize novel
and diverse clothing styles to train a robust model. Simultaneously,
enhancing samples through style to construct informative and robust
prototypes.
To sum up, the former methods introduce additional auxiliary information to overcome the clothes-changing, but this often incurs high
computational costs. The latter methods may destroy the most discriminative information from the original RGB image. And it is unclear

2.2. Cloth-changing person re-identification
In recent years, a lot of work (Bansal, Foresti, & Martinel, 2022;
Li, Xu, Wei, & Yang, 2022; Yao, Kusakunniran, Wu, Xu, & Zhang,
2022) has emerged for clothes-changing person ReID. The core idea
of the work is to learn clothes-irrelevant features. The first type of
these methods are to introduce clothing-unrelated auxiliary information for feature mining, including keypoints (Qian et al., 2020; Zhang
et al., 2023), 3D shapes (Chen et al., 2021; Liu et al., 2023), semantic
masks (Li, Xu, Li, & Guo, 2023), and human parsing techniques (Wu
et al., 2022). Zhang et al. (2023) proposed a novel multi-biological
unified network that utilizes multi-biometric features to mitigate the
influence of clothing. It used a key points estimation model and a
graph convolution layer to capture more discriminative pose features,
aiming to enhance the robustness of the cloth-changing ReID model.
Qian et al. (2020) leveraged the human keypoints to encode shape
embedding, distilling the identity-relevant shape feature, which aims to
focus on the body shape information. Liu et al. (2023) used 3D shapes
to reconstruct 3D clothed body shapes, disentangling identity from
non-identity components. It learns discriminative features of naked
body shapes in a joint manner. Li et al. (2023) utilized unsupervised
contrastive learning framework that combines RGB and semantic mask
features to learn cross-clothes invariance from a hierarchically semantic
3

Neural Networks 190 (2025) 107617

M. Mei et al.

which features the model has learned. Therefore, a MBNet is proposed
to balance multiple clothing irrelevant discriminative features, such as
head, legs, appearance, posture, and other information.
3. Methodology
This section presents the details of the proposed MBNet framework
as shown in Fig. 1, which consists of three parts: a global branch, a
clothing-unrelated branch, and a mask branch. Firstly, in the global
branch, complete RGB images are used as input and a knowledge transfer module is applied to learn more clothing-irrelevant fine-grained
features. Secondly, in the clothing-unrelated branch, the parts that are
unaffected by clothing are used as input to enhance the robustness
of clothing change problems. Meanwhile, a feature attention module
is introduced to mitigate the interference caused by the local noise.
Finally, in the mask branch, a cross module is applied to fuse the
learned features, aiming to dig out more contextual information.

Fig. 2. The structure of Knowledge Transfer Module (KTM).

3.1. Global branch

3.2. Clothing-unrelated branch

Existing work (Wang, Qian, Fu and Xue, 2022; Yao et al., 2022)
learns features to resist clothes changes from pedestrian images, simply concatenating the original features and clothes-irrelevant features
without knowledge fusion. However, these methods have two critical issues: the concatenation operation fixes the weights of the two
types of features and cannot dynamically adjust them based on the
samples. Additionally, the spatial alignment of the two features is not
constrained, leading to inefficient fusion. To address these, a Knowledge Transfer Module (KTM) is proposed, inspired by the collaborative optimization idea in knowledge distillation (Zheng et al., 2021):
the clothing-unrelated branch serves as the teacher network, extracting fine-grained discriminative features, while the global branch acts
as the student network, receiving guidance signals from the teacher
through the KTM to dynamically fuse clothing-irrelevant knowledge.
The structure is illustrated in Fig. 2. To highlight the clothing independent regions, knowledge from the global branch is combined with
the clothing-unrelated branch which aim to improve the performance
of clothes-changing ReID. In the clothing-unrelated branch, the framework learns fine-grained discriminative clothing-irrelevant features and
transfers them to the global branch to complement the cloth-unrelated
knowledge.
Specifically, the backbone network is utilized to extract the appearance feature 𝐹𝑔 from the RGB images. Then, the learned feature
𝐹𝑡𝑏 from the clothing-unrelated branch are taken to generate finer
features through the attention module, which mainly concentrates on
the pedestrian area. The formula is as follows in Eq. (1):

To address clothes-changing problems, the features from the
non/less affected body parts of pedestrians are explored. The most
decisive point accurately segments the human bodies into the affected
and non/less parts. In the MBUNet (Zhang et al., 2023), they used a
localization layer based on Spatial Transformer Network (STN) (Xu
et al., 2020), to roughly locate and cut out the partial image, and
then use it as the input. However, it incurred additional computational
costs. In the NPFM (Kweon & Cho, 2023), to concentrate on clothingunrelated identity information, they cropped 12.5% of the top and
bottom regions of the input RGB image, but this method has not been
effectively proven. In the paper, segmentation based on the anatomical
information (Leboeuf, Barre, Aminian, & Sangeux, 2023) from human
bodies has been proven feasible in the gait recognition method (Yao
et al., 2022). Especially, for a person of height H, his/her body can
be separated around some semantic positions, such as knees (0.285H),
pelvis (0.48H), waist (0.535H), and neck (0.87H) (Leboeuf et al., 2023).
It mainly focuses on the parts that less affected by clothes-changing: the
area between the head and shoulders, and the crus part extending from
the knees to the feet.
In the clothing-unrelated branch, the non/less affected body parts
are used as input to enhance robustness for clothes-changing problems.
Moreover, to further dig out fine-grained clothes-irrelevant features,
a feature attention module (FAM) is introduced to reduce the local
noise and suppress the background clutter, which aims to preserve the
pure feature map. The structure of FAM is shown in Fig. 3. Firstly, the
cropped part is inputted to extract the upper body features 𝐹𝑡 and lower
body features 𝐹𝑏 through the backbone network. The feature 𝐹𝑡𝑏 is
obtained by channel-wise concatenating two features, as demonstrated
in Eq. (4):

𝐴 = 𝑆𝑖𝑔𝑚𝑜𝑖𝑑(𝑐𝑜𝑛𝑣(𝑊1 , (Re 𝐿𝑈 (𝑐𝑜𝑛𝑣(𝑊2 , 𝐹𝑡𝑏 )))))

(1)

where 𝐴 represents the attention map, and 𝑊1 and 𝑊2 are the parameters of the first and second convolutional layers respectively. The
clothing-unrelated branch not only provides supplementary features
but also guides the global branch to focus on key regions that are
clothing-irrelevant through attention map 𝐴.
Next, the learned knowledge is transferred to the global branch
for knowledge fusion. With this approach, more robust pedestrian
representations could be obtained. The learned attention map 𝐴 is used
to channel weight to 𝐹𝑔 . And the fused feature 𝐹𝑓 is obtained through
the fast connection architecture. The final 𝐹𝑓 is then input into the
feature learning stage, as shown in Eq. (2):
𝐹𝑓 = 𝐹𝑔 ⊗ 𝐴 + 𝐹𝑔

Secondly, 𝐹𝑡𝑏 passes through a fully connected layer 𝐹1 for dimension reduction, a 𝑅𝑒𝐿𝑈 activation, another fully connected layer 𝐹2
for dimension incrementation, and a 𝑆𝑖𝑔𝑚𝑜𝑖𝑑 activation to generate a
+
clothing-unrelated attention map 𝐹𝑎𝑡𝑡
, as shown in Eq. (5):
+
𝐹𝑎𝑡𝑡
= 𝑆𝑖𝑔(𝐹2 (𝑅(𝐹1 𝐹𝑡𝑏 )))

Then, an inverse attention map
contrastive features in Eq. (6):

(2)

−
+
𝐹𝑎𝑡𝑡
= 1 − 𝐹𝑎𝑡𝑡

where ⊗ denotes the tensor elements multiplication.
Cross-entropy loss and hard sample triplet loss are used to reduce the feature differences of pedestrians with the same identity
in clothing-changing conditions, while reducing intra-class differences
and increasing inter-class differences, as Eq. (3):
𝐿𝑔𝑙𝑜 = 𝐿𝑡𝑟𝑖 + 𝐿𝑐𝑒

(4)

𝐹𝑡𝑏 = 𝐹𝑡 + 𝐹𝑏

(5)
−
𝐹𝑎𝑡𝑡

is generated to attend the
(6)

Consequently, this pair of attention maps are employed to the
feature 𝐹𝑡𝑏 to obtain identity feature 𝐹𝑐 and background feature 𝐹𝑏𝑝 .
The formula are shown in Eqs. (7) and (8):

(3)

+
𝐹𝑐 = 𝐹𝑎𝑡𝑡
⊗ 𝐹𝑡𝑏

4

(7)

Neural Networks 190 (2025) 107617

M. Mei et al.

Fig. 3. The structure of Feature Attention Module (FAM). It aims to obtain more refined clothing-irrelevant feature. It uses the features of upper and lower body to extract the
foreground features of pedestrians. And the background loss function 𝐿𝑏𝑔𝑟 is introduced to push the distance between the foreground features and background features.

−
𝐹𝑏𝑝 = 𝐹𝑎𝑡𝑡
⊗ 𝐹𝑡𝑏

lighting conditions, occlusion, and different pedestrians wearing similar
clothes. Mask features can provide more consistent identity information and increase the differences between classes, which improves the
discrimination ability of the model. As shown in Fig. 5, such coarse estimated human silhouette masks are not accurate enough to differentiate
between cloth-changing pedestrians.
To solve these problems, a cross-fusion module (CFM) is used that
employs two-way cross-attention to learn cross-clothes invariance by
combining both mask features and clothes-irrelevant features in a
two-branches network. The CFM module not only utilizes the complementary relationship between mask features and clothing-irrelevant
features, but also effectively overcomes the limitations of ReID models which rely on the appearance characteristics for a long time. In
this way, deep representations of pedestrians are obtained that could
distinguish the identities.
The feature maps 𝐹𝑚 and 𝐹𝑡𝑏 are extracted using the backbone
network, and each feature map is flattened and input into CFM module.
The details of the CFM module are shown in Fig. 4. First, the tokens of
the mask feature 𝐹𝑚 are projected to two independent matrices 𝐾1 and
𝑉1 as a set of values and keys. The clothing-irrelevant feature tokens
𝐹𝑐 are projected to another separate matrix 𝑄1 , as a set of queries. As
shown in Eq. (11):

(8)

where ⊗ denotes the tensor elements multiplication. However, the
background information will affect the feature 𝐹𝑐 . Suppressing the
background clutter is an effective method (Yang, Liu et al., 2023) for
ReID tasks. Different people have the similar backgrounds, and the
same person can also have different background, which might affect
the performance of model recognition. And, directly using the feature
𝐹𝑐 would limit the ability of model to learn more diverse features.
A background constraint loss 𝐿𝑏𝑔𝑟 is proposed to push the distances
between the background feature 𝐹𝑏𝑝 and global feature 𝐹𝑔 are large,
and bring the distances between the identity feature 𝐹𝑐 and global
feature 𝐹𝑔 small. During the testing stage, only the pure identity feature
mapping 𝐹𝑐 is retained for final pedestrian detection and discard the
feature 𝐹𝑏𝑝 . The formula of 𝐿𝑏𝑔𝑟 is as shown in Eq. (9):
1 ∑ ‖
‖2
‖
‖2
(‖𝐹 − 𝐹𝑐 ‖ + max{(𝑚 − ‖𝐹𝑔 − 𝐹𝑏𝑝 ‖ ), 0})
‖2
‖
‖2
𝑁 𝑛=1 ‖ 𝑔
𝑁

𝐿𝑏𝑔𝑟 =

(9)

where 𝑁 is the batch size. And 𝑚 is a margin parameter which is empirically set to 10 in the experiments. By minimizing the loss, identity
features and global features will get close to each other, while the background features will be away. It can intensify identity features while
reducing the interference of irrelevant features, which enhance the
performance in person ReID tasks. To minimize the distance between
positive sample pairs and maximize the distance between negative sample pairs, which address the clothes-changing problems. Cross-entropy
loss and hard sample triple loss are utilized in Eq. (10):
𝐿𝑐𝑢 = 𝐿𝑡𝑟𝑖 + 𝐿𝑐𝑒 + 𝜆𝐿𝑏𝑔𝑟

𝑄1 = 𝐹𝑐 ; 𝐾1 = 𝑉1 = 𝐹𝑚

(11)

Secondly, the correlation matrix is established using a dot product operation, and then the correlation attention score is obtained
through the softmax function, which represents the weight between
mask features and clothing-irrelevant features. Then the attention score
matrix is multiplied by 𝑉1 to obtain a vector, which is normalized using
the 𝐹𝑐 residual connection of clothing-irrelevant features to prevent
overfitting. The similarity feature 𝐹𝑐𝑚 is obtained as Eq. (12):

(10)

where 𝐿𝑡𝑟𝑖 and 𝐿𝑐𝑒 denote hard triple loss and cross-entropy loss,
respectively. To mitigate the interference of background clutter, a
background constraint loss function 𝐿𝑏𝑔𝑟 is adopted in the clothingunrelated branch, which helps make global and identity features invariant to background clutter, and parameter 𝜆 is set to 0.01 in the
experiment.

𝑄1 𝐾1 𝑇
𝐹𝑐𝑚 = 𝐿𝑁1 (𝑠𝑜𝑓 𝑡 max( √
)𝑉1 + 𝐹𝑐 )
𝑑1

(12)

where 𝐿𝑁1 represents the normalization operation. 𝑄1 and 𝐾1 denote
the query matrix and key matrix, respectively. 𝑑1 represents the scaling
factor of the feature dimension of the query and key. 𝑉1 indicates the
value matrix. Besides, a multi-head cross-attention mechanism with 8
parallel heads is employed, which enables the model to jointly understand the correlation between patch images and pedestrian contour
images from different angles. Where 𝑑1 represents the dimensions of
𝑄1 and 𝐾1 divided by 8, used to scale the attention size. The similarity
feature 𝐹𝑚𝑐 between clothing-irrelevant feature and mask feature is also
similar to the Eq. (12), as follows:

3.3. Mask branch
In the mask branch, the pedestrian silhouette masks are utilized
as input to further assist the network in learning clothing-irrelevant
features. The corresponding pedestrian silhouette masks are generated
through SCHP (Li et al., 2022). Pedestrian silhouette masks contain
rich clothes-invariant features which helps us to re-identification under
the circumstance of clothes-changing, as the body shape does not
change significantly for the same person. However, it ignores some
important appearance features, such as color attributes, facial features,
and so on. The model solely relies on pose information, which may
not be reliable, especially in complex environments, such as varying
5

𝑄2 = 𝐹𝑚 ; 𝐾2 = 𝑉2 = 𝐹𝑐

(13)

𝑄2 𝐾2 𝑇
𝐹𝑚𝑐 = 𝐿𝑁2 (𝑠𝑜𝑓 𝑡 max( √
)𝑉2 + 𝐹𝑚 )
𝑑2

(14)

Neural Networks 190 (2025) 107617

M. Mei et al.

Fig. 4. The detailed structure of the Cross Fusion Module (CFM). It aims to use a two-way cross-attention mechanism to integrate clothing-irrelevant features 𝐹𝑐 and mask features
𝐹𝑚 .

4. Experiments and analysis

where 𝐿𝑁2 represents the normalization operation. 𝑄2 and 𝐾2 denote
the query matrix and key matrix, respectively. 𝑑2 represents the scaling
factor of the feature dimension of the query and key. 𝑉2 indicates the
value matrix. Then channels are added to the two similar features,
utilizing the global average pooling to get a more refined feature 𝑓𝑎𝑝
according to Eq. (15):

4.1. Datasets and settings
The proposed approach is evaluated on three cloth-changing
datasets, i.e. LTCC (Qian et al., 2020), PRCC (Yang et al., 2021) and
VC-Clothes (Wan, Wu, Qian, Chen, & Fu, 2020).
The LTCC (Qian et al., 2020) dataset contains 17 138 images of 152
identities captured by 12 cameras. In the training set with 77 identities,
46 individuals appear in different clothing and the remaining 31 do
not change clothing, while the test set consists of 45 clothing-change
and 30 clothing-consistent identities. On average, each individual has
5 different outfits for LTCC.
The PRCC (Yang et al., 2021) dataset is a large indoor clotheschanging ReID dataset which contains 33 698 images of 221 identities
with three cameras. In cameras A and B, pedestrians of the same
identity wear the same clothes, but appear with different backgrounds.
In the view of camera C, the same identity wears different clothes
compared to the other two views. Each person is limited to two clothing
changes.
The VC-Clothes (Wan et al., 2020) dataset is a virtual dataset from
the game GTA-V in 4 different scenes, which consists of 512 identities
and 19 060 images. In VC-Clothes, 256 identities are used for the
training set and 256 are used for the testing set, with an average of
1 to 3 outfits for each identity.
We follow the original protocol for each dataset. For LTCC, two
test settings are used: the standard setting and the cloth-changing
setting (CC). The standard setting: all gallery samples are used to
calculate accuracy. The cloth-changing setting: only clothes-changing
gallery samples are used to calculate the accuracy. For PRCC and VCClothes, two test settings are introduced: the same-clothing setting (SC)
and the cross-clothing setting (CC). The same-clothing setting: only
clothes-consistent gallery samples are used to calculate accuracy. The
cumulative matching characteristics (CMC) and mean average precision
(mAP) are employed to evaluate the performance of our method.

(15)

𝐹𝑎𝑝 = 𝐺𝐴𝑃 (𝑐𝑜𝑛𝑐𝑎𝑡(𝐹𝑐𝑚 , 𝐹𝑚𝑐 ))

where GAP denotes global average pooling. To further learn the mask
body features beyond the clothing-invariant features, the mask branch
loss function 𝐿𝑚𝑎 is defined as Eq. (16):
(16)

𝐿𝑚𝑎 = 𝐿𝑡𝑟𝑖 + 𝐿𝑐𝑒

where 𝐿𝑡𝑟𝑖 and 𝐿𝑐𝑒 denote hard triple loss and cross-entropy loss,
respectively.
3.4. Total loss
The total loss function used to train the entire network could be
denoted as Eq. (17):
(17)

𝐿𝑡𝑜𝑡𝑎𝑙 = 𝛼𝐿𝑐𝑢 + 𝛾𝐿𝑚𝑎 + 𝐿𝑔𝑙𝑜

where the parameter 𝛼 and parameter 𝛽 are both set to 0.4 in the paper.
(1) Triplet loss: triplet loss with hard example mining is defined as
Eq. (18):
𝐿𝑡𝑟𝑖 =

𝑁
∑

[𝑑max (𝑥𝑖𝑎 , 𝑥𝑖𝑝 ) − 𝑑min (𝑥𝑖𝑎 , 𝑥𝑖𝑛 ) + 𝑚]

+

𝑖=1

(18)

where 𝑁 denotes the number of images in mini-batch, and 𝑑(⋅, ⋅)
represents the commonly used Euclidean distance. 𝑥𝑎 is the anchor
image, and 𝑥𝑝 is a positive example image from the same person as the
anchor image. 𝑥𝑛 is a negative example image with a different identity
from the anchor image. 𝑚 is a margin hyperparameter, representing the
boundary.
(2) Cross-Entropy Loss: cross-entropy loss with label smoothing
regularization (Fu, Zhang, & Wang, 2024) is defined as Eq. (19):
𝑦̂𝑖 = (1 − 𝜖)𝑦𝑖 +

𝜖
𝐾

4.2. Implementation details
The software tools are Pytorch 1.11.0, CUDA 11.3, and Python 3.8.
The hardware device is a GeForce RTX 3090 GPU.
For the backbone network, the CAL (Gu et al., 2022) is used as the
baseline. The input images are padded and resized into 384 × 192 and
the training batch size is set at 32 for each dataset, accommodating 4
pedestrians with 8 images per person. And the testing batch size is set
to 128. These images are further augmented with random horizontal
flipping, cropping, and erasing. The model is trained using the Adam
optimizer for 60 epochs, with a weight decay of 5 × 10-4. For LTCC
datasets, the initial learning rate is set to 1.5 × 10-4 and divided by
10 after 20, 40, and 60 epochs. For PRCC and VC-Clothes datasets, the

(19)

where 𝑦 denotes the ground-truth label, 𝐾 is the total number of categories, and 𝜖 is a smoothing parameter. And 𝑦̂𝑖 indicates the smoothed
label, while 𝑦𝑖 is the 𝑖th element of the true label 𝑦.
𝐿𝑐𝑒 = −

𝐾
∑

𝑦̂𝑖 log(𝑝𝑖 )

(20)

𝑖=1

where log(𝑝𝑖 ) is the predicted probability from the model.
6

Neural Networks 190 (2025) 107617

M. Mei et al.

Table 1
Comparison of Rank-k and mAP Performance with State-Of-The-Art (SOTA) Methods in LTCC and PRCC. The best and second-best results are highlighted in bold and underline.
Method

LTCC

PRCC

General

Cloth-changing

Same-clothes

Cross-clothes

Rank1

mAP

Rank1

mAP

Rank1

mAP

Rank1

mAP

HACNN (Li, Zhu, & Gong, 2018)
PCB (Sun et al., 2018)
MGN (Wang, Yuan, Chen, Li, & Zhou, 2018)
ISP (Zhu, Guo, Liu, Tang, & Wang, 2020)
AGW (Ye et al., 2022)

60.2
65.1
68.4
66.3
61.4

26.7
30.6
34.6
29.6
21.5

21.6
23.5
25.0
27.8
27.0

9.3
10.0
12.6
11.9
8.4

82.5
99.8
98.2
92.8
97.8

84.8
97.0
98.4
–
91.5

21.8
41.8
53.5
36.6
44.7

23.2
38.7
53.3
–
37.1

SPT+ASE (Yang et al., 2021)
GI-ReID (Jin et al., 2022)
CESD (Qian et al., 2020)
3DSL (Chen et al., 2021)
FSAM (Hong et al., 2021)
CAL (Gu et al., 2022)
AIM (Yang, Lin, Zhong, Wu and Wang, 2023)
CCFA (Han, Gong, Huang, Wang, & Tan, 2023)
3DInvarReID (Liu et al., 2023)
DCR-ReID (Cui et al., 2023)
AFL (Liu, Ge, Wang, Hou and Zhao, 2024)
MCSC-CAL (Huang et al., 2024)

–
63.2
71.4
–
73.2
74.2
76.3
75.8
–
76.1
74.4
73.9

–
29.4
34.3
–
35.4
40.8
41.1
42.5
–
42.3
39.1
40.2

–
23.7
26.2
31.2
38.5
40.1
40.6
45.3
40.9
41.1
42.1
42.2

–
10.4
12.4
14.8
16.2
18.0
19.1
22.1
18.9
20.4
18.4
19.4

64.2
80.0
–
–
–
100.0
100.0
99.6
–
100.0
100.0
99.8

–
–
–
–
–
99.8
99.9
98.7
–
99.7
99.7
99.8

34.4
33.3
–
51.3
–
55.2
57.9
61.2
56.5
57.2
57.4
57.8

–
–
–
–
–
55.8
58.3
58.4
57.2
57.4
56.5
57.3

MBNet (Ours)

77.9

43.1

44.6

22.7

99.9

98.7

58.3

57.9

Table 2
Comparison of Rank-k and mAP Performance with State-Of-The-Art (SOTA) Methods in VC-Clothes. The best and second-best results are
highlighted in bold and underline.
Method

VC-Clothes
Same-clothes

Cloth-changing

Rank1

mAP

Rank1

mAP

LOMO+XQDA (Liao, Hu, Zhu, & Li, 2015)
MDLA (Qian, Fu, Jiang, Xiang, & Xue, 2017)
PCB (Sun et al., 2018)
Part-Aligned (Suh, Wang, Tang, Mei, & Lee, 2018)
RGA-SC (Zhang, Lan, Zeng, Jin, & Chen, 2020)
ISP (Zhu et al., 2020)

86.2
94.3
94.7
93.9
95.4
94.5

83.3
93.9
94.3
93.4
94.8
94.7

34.5
59.2
62.0
69.4
71.1
72.0

30.9
60.8
62.2
67.3
67.4
72.1

3DSL (Chen et al., 2021)
GI-ReID (Jin et al., 2022)
FSAM (Hong et al., 2021)
MBUNet (Zhang et al., 2023)
CAL (Gu et al., 2022)

–
–
94.7
95.7
95.1

–
–
94.8
94.2
95.3

79.9
64.5
78.6
82.7
81.4

–
57.8
78.9
70.3
81.7

MBNet (Ours)

94.7

94.7

87.2

84.0

initial learning rate is set to 3.5 × 10-4. Hyper-parameters 𝛼 and 𝛾 are
set to 0.4 and 0.7 for PRCC datasets, and 0.4 and 0.4 for LTCC dataset
and VC-Clothes, respectively.

dataset, by learning clothes-irrelevant from the appearance and shape
streams. CAL (Gu et al., 2022) uses clothing labels to improve accuracy
to 40.1% and 18.0% under cloth-changing conditions. AIM (Yang,
Lin et al., 2023) uses causality to eliminate the influence of clothing
bias on model training, encouraging it to learn more discriminative
ID clues that were not affected by clothing bias, which increased by
0.5% and 1.1% compared to CAL. CCFA (Han et al., 2023) achieves
45.3% of Rank-1 through feature augmentation learning to augment
plausible clothing-change features. In the general setting, our method
shows improvements in both Rank-1 and mAP compared to the current
best-performing CCFA (Han et al., 2023). Compared to the latest MCSCCAL (Huang et al., 2024) method, our method is improved by 2.4% on
Rank-1 in cloth-changing. As it can be seen in Table 1, our method
also achieves a competitive result that achieves 77.9% and 44.6%
respectively on rank-1 and increases by 3.7% and 4.3% compared with
the baseline, indicating that it is able to eliminate the effects of clothing
to a certain extent.
Result on PRCC (Yang et al., 2021). For the PRCC dataset, the
results are reported for the same clothing and cross-clothes settings,
as shown in Table 1. In the same-clothes setting, the traditional ReID
methods, such as PCB (Sun et al., 2018) and AGW (Ye et al., 2022),
achieve 99.8% and 98.2% accuracy on Rank-1, respectively. In the
cross-clothes setting, our approach outperforms the PCB (Sun et al.,
2018) by 16.5% and 19.2% in Rank-1 and mAP, respectively. Comparing the disentanglement methods, our method surpasses the CAL (Gu
et al., 2022) by 3.1% and 2.1% in Rank-1 and mAP, respectively. Our

4.3. Experimental results
The proposed method is evaluated on three cloth-changing datasets,
i.e. LTCC (Qian et al., 2020), PRCC (Yang et al., 2021), and VCCloth (Wan et al., 2020). The proposed method MBNet is compared
with representative cloth-changing person ReID methods based
on hand-crafted features and deep-learning methods, as shown in
Table 1.
Result on LTCC (Qian et al., 2020). The proposed method is compared with related ReID methods on the LTCC datasets are shown in
Table 1. For PCB (Sun et al., 2018) and MGN (Wang et al., 2018),
they achieve great competitive results in the traditional ReID. PCB (Sun
et al., 2018) and MGN (Wang et al., 2018), achieve 65.1% and 68.4%
accuracy on Rank-1, respectively. However, they performed poorly in
the cloth-changing settings because they did not take clothing changes
into account. Compared with the ISP (Zhu et al., 2020), Rank-1 is
improved by 11.6% in general setting. SPT+ASE (Yang et al., 2021)
and CESD (Qian et al., 2020) extract invariant features mainly from
a person pose key points. GI-ReID (Jin et al., 2022) mainly uses gait
features, and it results are not satisfactory because LTCC is static data.
FSAM (Hong et al., 2021) reaches 38.5% and 16.2% on the LTCC
7

Neural Networks 190 (2025) 107617

M. Mei et al.
Table 3
Ablation studies on different branches of our method on the LTCC dataset.
General

Baseline
GL branch
CU branch
MA branch
CU+MA branch
GL+CU+MA(MBNet)

Cloth-changing

Rank1

mAP

Rank1

mAP

74.2
76.1
77.3
76.9
76.7
77.9

40.8
40.3
43.2
40.4
42.3
43.1

40.1
42.1
41.6
41.1
43.1
44.6

18.0
18.7
20.8
19.4
22.2
22.7

Table 4
Ablation studies of three feature fusion strategies on the LTCC dataset.
General

Direct concatenation
Single-direction attention C→M
Single-direction attention M→C
Cross fusion module

Cloth-changing

Rank1

mAP

Rank1

mAP

76.1
76.5
76.1
77.9

42.4
42.1
41.8
43.1

43.5
42.9
42.6
44.6

22.4
21.9
21.6
22.7

method achieves at least 2% increase over baseline in both Rank-1
and mAP evaluations. Compared to the DCR-ReID (Cui et al., 2023)
method, our method improves by 1.1% on Rank-1 in same-clothes.
Compared to the best competitor CCFA (Han et al., 2023), Rank-1 and
mAP differ by 2.9% and 0.5%, respectively. Our approach does not
outperform the CCFA (Han et al., 2023) method on the PRCC dataset
because the dataset includes sketch images containing pedestrian mask
images. However, compared with other cloth-changing ReID methods,
our method also gets strong competitive results, which testifies the
effectiveness of our method.
Result on VC-Clothes (Wan et al., 2020). Also, comprehensive
experiments are executed to evaluate the performance of our proposed
MBNet on the VC-Clothes dataset, and the results are shown in Table 2.
As it could be seen, there is a significant performance gap between our
method and the traditional learning-based methods in cloth-changing.
In the same-clothes setting, RGA-SC (Zhang et al., 2020) and ISP (Zhu
et al., 2020) show great competitiveness, achieving Rank-1 and mAP
of 95.4%, 94.5% and 94.5%, 94.7%, respectively. In the cloth-change
setting, our method achieves 87.2% and 84.0% respectively on Rank-1
and mAP, which increasing by 6.2% and 2.3% compared to the optimal
CAL (Gu et al., 2022). Compared with the second-best MBUNet (Zhang
et al., 2023), our method is improved by 4.5% and 13.7% respectively on Rank-1 and mAP. Compared with the GI-ReID (Jin et al.,
2022) method based on gait, Rank-1 and mAP are improved by 22.7%
and 26.2%. The experiment results show that our method is also
applicable to clothing-consistent scenes and achieves competitive results, indicating that the proposed MBNet method can adequately mine
identity-related information in general person re-identification.

Fig. 5. Illustration of the pedestrian silhouette masks images. It describes the mask
image relative to the original RGB image in there datasets.

supplement the discriminative ability of clothing independent knowledge in the global branch, effectively eliminating the impact of clothing
changes.
Effectiveness of the Clothing-unrelated Branch. The clothingunrelated branch is tested. The results are summarized in Table 3.
CAL (Gu et al., 2022) is utilized as the baseline. Performance boosts
of 3.1%, 2.4%, 1.5%, and 2.8% are observed on two evaluation criteria, respectively, when combining the clothing-unrelated branch. This
suggested that the patch images of pedestrians can be used as input to
effectively learn clothing-irrelevant features (e.g., head, legs, pose, and
carrying object). In this branch, the learned features are more robust
and comprehensive in resisting the influence of clothing scenario. The
experiment results show that the clothing-unrelated branch effectively
learns fine-grained clothing-unrelated features, which helps validate
the performance of the model.
Effectiveness of the Mask Branch. As shown in Table 3, the mask
branch significantly outperforms the baseline. In the cloth-changing
setting, Rank-1 and mAP are increased by 1.0% and 1.4% respectively,
demonstrating the benefits of using pedestrian silhouette masks as
input. As shown in Fig. 5, pedestrian silhouette masks contain rich body
features. To effectively combine mask features and clothes-irrelevant
features, our model utilizes two-way cross-attention, thus significantly
overcoming the limitations of traditional person ReID that relies on
appearance information. In the mask branch, Rank-1 and mAP are increased by 2.5% and 1.3% respectively in the general setting, indicating
that the branch can better activate the potential of mask features in
clothes-changing.
Effectiveness of the Cross Fusion Module. To evaluate the effectiveness of the CFM approach compared to simpler strategies, experiments were conducted on the LTCC dataset, as presented in Table
4. In this context, ‘‘C → M’’ refers to the single attention interaction
from the clothing-unrelated branch to the mask branch, while ‘‘M →
C’’ indicates the single attention interaction from the mask branch to
the clothing-unrelated branch. The experimental results demonstrate

4.4. Ablation study
To gain more insights into our proposed MBNet, the influence is
explored of different branches based on the LTCC (Qian et al., 2020)
dataset. Then, the proposed method is compared with the baseline
model to comprehensively evaluate its effectiveness.
Effectiveness of the Global Branch. In the global branch, a knowledge transfer module (KTM) is introduced to transfer the clothing
invariance knowledge learned from clothing-unrelated branch to global
branch. It increases Rank-1 and mAP to 42.1% and 18.7% respectively,
in a cloth-changing scenario. This suggested that our proposed KTM
is effective compared to the methods that use simple concatenation of
knowledge features. It can highlight the clothes-unrelated areas and
8


codex
前三篇里，无监督多队列的方法部分完整；两篇换衣论文的前 8 页覆盖了方法和主要消融，但其中 MBNet 的分支细节在终端输出中被截断。我会补读它的第 4 到 6 页，同时继续读剩下三篇。
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
/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 979-8-3315-6701-9/26/$31.00 ©2026 IEEE | DOI: 10.1109/ICASSP55912.2026.11464029

MULTI-GRANULARITY ATTRIBUTE PROMPT LEARNING FOR CLOTH-CHANGING
PERSON RE-IDENTIFICATION
Xu Li, Hefei Ling, Shijuan Huang, Chenxing Zhao
Huazhong University of Science and Technology, Wuhan, China
ABSTRACT
Cloth-changing person re-identification (CC-ReID) addresses
the practical need to recognize pedestrians across cameras after clothing changes. Existing methods use biological traits
and clothing labels to extract invariant features but underutilize semantic attributes and require costly manual annotation. To overcome these limitations, we propose a Multigranularity Attribute Prompt Learning (MAPL) method.
MAPL incorporates visual deep prompts in a dual-branch
network (raw and clothing-erased images) to learn clothingirrelevant representations. It generates pseudo clothing labels
through attribute recognition to reduce annotation dependency. Using these pseudo-labels and attribute texts, it enhances identity-related attributes while suppressing clothingrelated ones. This is achieved through feature disentanglement at both the caption and attribute levels. Experiments
show that MAPL achieves state-of-the-art performance,
outperforming the second-best method by 3.2%/13.0% in
Rank-1 and 4.0%/18.3% in mAP on PRCC and LTCC under
clothing-changing settings. The source code is released at
https://github.com/IPMD2025/MAPL.
Index Terms— Cloth-Changing Person ReID, MultiModal Fusion, Attribute, Prompt Learning.
1. INTRODUCTION
Person re-identification (ReID) aims to identify individuals across non-overlapping camera views in surveillance scenarios, playing an essential role in smart city development
and intelligent security systems [1, 2, 3]. Traditional ReID
methods primarily focus on short-term scenarios under the
assumption that the same individual retains the same clothing. However, clothing changes are common in real-world
scenarios and may even be used intentionally to evade surveillance. Most ReID models also over-rely on apparel features
[4]. Therefore, CC-ReID, which operates over longer time
spans, is both more challenging and practically significant.
This work is supported in part by the Natural Science Foundation of
China under Grant 62372203 and 62302186, in part by the Major Scientific and Technological Project of Shenzhen (202316021), in part by the National key research and development program of China(2022YFB2601802),
in part by the Major Scientific and Technological Project of Hubei Province
(2022BAA046, 2022BAA042).

979-8-3315-6701-9/26/$31.00 ©2026 IEEE

Existing methods can be broadly categorized into unimodal and multimodal approaches. Unimodal methods
[5, 6, 7, 8] disentangle clothing-related features solely from
RGB images. Multimodal methods [9, 10, 11, 12, 13] enhance clothing-invariant feature extraction by incorporating
additional cues like sketches, face, gait, clothing-erased images, attributes, and textual prompts. However, they fail to
effectively utilize attributes, particularly neglecting attribute
texts. AD ViT [14] utilizes only clothing-related attributes,
while MADE [15] employs identity-relevant attributes. Both
are limited to one-hot attribute labels and overlook attribute
texts containing richer semantic information. CCAF [16] and
MIPL [17] generate text from images but lack attribute information. Moreover, their reliance on clothing labels increases
manual annotation costs.
In this paper, We propose a dual-branch architecture based
on visual deep prompt tuning [18], which leverages attribute
labels and attribute texts to compel the model to focus on
identity-relevant attributes while suppressing clothing-related
information. First, to enhance generalization and achieve
pixel-level disentanglement, clothing-erased images and visual deep prompt tuning (VLP deep) are introduced as the
baseline. Second, to address the lack of high-level semantic
guidance in pixel-level disentanglement, an attribute textual disentanglement module is implemented by reinforcing
identity-relevant attributes and suppressing clothing-related
attributes. A BNNeck layer [2] is incorporated to separate
image tuning from attribute disentanglement, preventing corruption of the alignment knowledge.
Our main contributions can be summarized as follows:(1)
A visual deep prompt tuning method is introduced, and a
separation mechanism is incorporated to buffer potential misalignment caused by image tuning and attribute disentanglement. (2) A multi-granularity attribute disentanglement module is proposed, which enhances identity-relevant attributes
while suppressing clothing-related information. This compensates for the lack of high-level semantics in pixel-level disentanglement and reduces the reliance on manually annotated
clothing labels. (3) Extensive experimental results demonstrate that the proposed MAPL achieves new state-of-the-art
(SOTA) performance.

10942

ICASSP 2026

Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:09 UTC from IEEE Xplore. Restrictions apply.

Mask images

2. METHOD

Lmap , Lman , Lmsp , Lmsn

...

Transformer

BN

1

0
Sneaker
Attributes

Input

Fvt
CAT

Fv

A pedestrian in a short-sleeved top
A pedestrian in sneakers

Identity- related

Head

Clothing- related

Feature extraction

Lce

Lap
Lan

Fvt

[cls]G token
Prompt token

Lsp

The man under the age of 30
A Pedestrian with short hair, no
attachments, short sleeves,
trousers and sneakers.

Lmce

Identity
classification

BN

Clothing- related
Text Encoder

Jacket

A male pedestrian
A pedestrian under the age of 30

...

0

Head

Ltri

...

where, Mic is the mask matrix, and ⊙ denotes the Hadamard
product. It is worth noting that certain regions corresponding
to attributes such as shoes and hair are not masked. Compared to other clothing attributes, these regions exhibit lower
variability, and directly erasing them could impair ReID performance. However, they may still retain clothing style information (e.g., short sleeves vs. long sleeves, hair). Therefore,
we adopt coarse-grained label-level disentanglement instead
of fine-grained pixel masking in subsequent modules.
Attribute Recognition. AAPAR [20] achieves SOTA
performance on standard PAR datasets including PETA, PA100K, RAPv2, as well as the competition dataset UPAR,
demonstrating strong cross-dataset generalization capability. Therefore, we employ the AAPAR model pre-trained on
PETA to identify attributes. Following the standard configuration, we recognize 7 attribute groups (35 attributes). The
attribute groups consist of age, gender, headwear, upper-body
clothing, lower-body clothing, shoes, and attachments. Age
and gender are considered identity-relevant attributes, while
the others are categorized as clothing-related attributes. The
a
attribute labels are defined as yij
, where i ∈ {1, 2, . . . , N } , j ∈
{1, 2, . . . , 35}.
Pseudo label generation. CC-ReID requires recognizing
different individuals while disregarding clothing variations

Less30

Caption-level

(1)

1

BN

Lmtri

Mutual
Learning

Identity- related

Attribute-level

xsi = xi ⊙ Mic ,

Male

...

Given a training dataset D = (xi , yiid )|N
i=1 , where xi denotes the i-th image, N represents the number of pedestrian
images, and yiid is the corresponding person identity label.
Clothing-erased images generation. The SCHP [19]
model is employed to parse pedestrian images into 20 semantic categories, including background, hair, upper clothing,
etc. Regions corresponding to hair, face, arms, legs, and
shoes remain unmasked. All other regions are defined as
clothing-related areas, and their pixels are set to zero to generate the masked image.

AAPAR

pseudo-label

BN

 24

...

2.2. Auxiliary Information Generation

...

Transformer

Raw images

The proposed MAPL is a dual-branch network consisting of a raw image branch and a masked image branch. ,
As shown in Figure.1, it comprises three main components:
1) Auxiliary information generation; 2) Multimodal feature
extraction module; and 3) A classification module. In the
auxiliary information generation stage, clothing-erased images, one-hot pedestrian attribute labels, pseudo-labels, and
attribute texts are generated. In the feature extraction module, features are extracted separately from the raw images,
masked images, caption-level texts, and attribute-level texts.
The classification module comprises person identity recognition, attribute disentanglement, and branch interaction.

Prompt

SCHP

Transformer

Prompt

2.1. Preliminaries

Attribute Decoupling for
Masked Images

Image Encoder

...

Patch token
CAT

Lsn

Attribute Decoupling
for Raw Images
Classification

Fig. 1. Overview structure of the MAPL framework.
within the same identity. Therefore, identity labels are integrated during pseudo-label generation. They are concatenated
with attribute labels to form attribute-level pseudo-labels, and
with all clothing-related attribute labels to form caption-level
pseudo-labels.
Attribute Text Generation. The attribute labels are converted into attribute texts using hand-crafted templates. For
each image, 5 invariant attribute-level texts Tap are generated (e.g., “A pedestrian under the age of 30”), along with
30 clothing-related attribute-level texts Tan (e.g., “A pedestrian without leather shoes”). Each image is also associated
with one positive caption-level text Tsp that includes age and
gender (e.g., “A man under the age of 30”), and one negative
caption-level text Tsn that describes other clothing attributes
(e.g., “A pedestrian with short hair, no attachments, ...”).
2.3. Multi modal feature extraction
2.4. Overview
Image feature extraction. Visual deep prompts are introduced to enhance generalization capability. Specifically,
learnable random prompt tokens xjprompt are appended to the
input of each Transformer layer:
h
i
xjin = xjcls , xjprompt , xjimg , j ∈ {0, 1, . . . , L − 1} , (2)
where xjprompt ∈ RL×M ×1×C , the notation [, ] indicates the
concatenation operation and M is the length of the prompt
tokens. During the iterative computation of the Transformer
layers, xj+1
prompt replaces the prompt tokens output from the
j-th Transformer layer. The output of the final layer is passed
through layer normalization and a modality-specific projection to obtain the pedestrian image features. Its class token,
used for person classification, is denoted as Fvt , while the
m
features of the masked image are defined as Fvt
. In this
work, a BNNeck layer is used to separate image classification

10943
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:09 UTC from IEEE Xplore. Restrictions apply.

from modal alignment, preventing the disruption of alignment
knowledge during image tuning.
Fv = BN N eck(Fvt )

Table 1. Comparison with SOTA Methods on PRCC, LTCC.
Methods

(3)
AFL [6]
FIRe2 [22]
MADE [15]
CAL [5]
AIM [7]
MCSC [23]
CCAF [16]
DIFIF [24]
ours(MAPL)

Text feature extraction. We employ a frozen CLIP [21]
text encoder to extract textual features at both the caption
level and the attribute level, encompassing both identityrelevant and clothing-relevant attributes, denoted as FT sp ,
FT sn , FT ap , and FT an .

PRCC
CC
SC
R1 mAP
R1
mAP
57.4 56.5
100
99.8
65
63.1
100
99.5
67.5 64.2
100
98.9
55.2 55.8
100
99.8
57.9 58.3
100
99.9
57.8 57.3
99.8
99.8
70.4 63.7
99.9
98.4
71.3 67.0
100
99.5
74.5 71.0
99.17 98.36

LTCC
CC
General
R1 mAP R1
mAP
42.1 18.4 74.4 39.1
44.6 19.1 75.9 39.9
46.9 25.0
82.2 49.3
40.1 18.0 74.2 40.8
40.6 19.1 76.3 41.1
42.2 19.4 73.9 40.2
42.9 20.1 75.3 41.3
50.5 25.1
80.9 46.2
63.5 43.4
83.5 62.73

2.5. Classification Module
Identity classification. To ensure that the pedestrian
features retain identity discriminability, the hard example
mining-based triplet loss and the cross-entropy loss are applied to constrain image features. Lid and Lm
id denote the
identity classification losses for the raw image and the masked
image, respectively.
Attribute decoupling. A multi-granularity attribute disentanglement mechanism is introduced in the multimodal
space to further eliminate clothing-related interference. It
comprises focusing on identity-relevant attributes and suppressing clothing-relevant attributes. 1) Focusing on identityrelevant attributes. Since feature fusion can enhance the representation of consistent positional biases and enable precise
intervention [7], for the original images, we concatenate the
image features with the identity-relevant attribute texts. The
triplet loss based on attribute pseudo-labels is then applied to
enforce focus on the invariant attributes of the pedestrian.
5

Lap =

1X
(Ltri ([Fv , FT ap ] , yipse a ))
5 i=1

(4)

To enrich the granularity of semantic descriptions, captionlevel focusing on identity-relevant attributes is introduced in
the same manner.

Lsp = Ltri [Fv , FT sp ] , yiid
(5)
The sum of these two losses is denoted as Lp . For the
clothing-erased images, considering the minimal residual
clothing information, only label-based constraints are applied.
5
X
(6)
Lm
=
(Ltri (Fv , yipse a ))
ap
i=1

Similarly, the final loss for the masked image branch is
denoted as Lm
2) Suppression of clothing-relevant atp .
tributes. By explicitly suppressing clothing-relevant attributes, clothing-related interference is further eliminated.
The specific calculation method is consistent with that used
for focusing on identity-relevant attributes, but suppression is
achieved through a negative loss based on clothing pseudolabels. The masked branch still employs only one-hot labels

for adversarial training. The final adversarial losses, denoted
as Ln and Lm
n . For example:
Lsn = −Ltri ([Fv , FT sp ] , yipse s ) .

(7)

Branch interaction. Mutual learning [16] is adopted to
ensure semantic consistency between the image features of
the two branches.
P

Lvcon =

1 X
2
∥ci − cm
i ∥ ,
P i=1

(8)

where, ci and cm
i represent the within-batch centroids of the
image features BN (Fv ) from the two branches, respectively.
m
The image features in the multimodal space, Fvt and Fvt
, also
vt
participate in mutual learning, yielding the loss Lcon . The
sum of the losses from the two spaces yields Lcon .
The overall training loss for optimizing the network parameters is defined as follows:
m
m
Ltotal = Lid +Lm
id +α×(Lp +Lp +Ln +Ln )+Lcon , (9)

where α is a hyperparameter, set to 0.1 on the PRCC dataset
and 0.2 on the LTCC dataset. During inference we only adopt
the image-modal space features from the original image
branch.
3. EXPERIMENTS
3.1. Experimental Settings
We conduct experiments on standard CC-ReID datasets,
PRCC [9] and LTCC [4], using Rank-1 accuracy and mean
Average Precision (mAP) for evaluation. Following prior
studies, the experiments include three settings: 1) Clothing
Change (CC): indicating that the same identity wears different clothing in the query and gallery images; 2) Same
Clothing (SC): indicating that the same identity wears identical clothing in the query and gallery images; 3) General:
indicating that the same identity may appear in either the
same or different clothing across query and gallery images.

10944
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:09 UTC from IEEE Xplore. Restrictions apply.

Query

R-1

R-2

R-3

R-4

R-5

Query

R-1

R-2

R-3

R-4

R-5

Table 2. Ablation study result on the PRCC dataset
Component
PRCC
Index B A N A P CAP M LB R-1 mAP
√
√
1
65.9 66.3
√
√
√
2
70.1 69.8
√
√
√
√
3
72.5 71.0
√
√
√
√
4
73.1 71.1
√
√
√
√
√
5
74.5 71.0

(a)Baseline

(a) Effect of M

(b)MAPL

Fig. 3. Visualization of top-5 retrieval results for the baseline
and our model on PRCC. Green borders: correct matches; red
borders: incorrect matches.

(b) Effect of 

Fig. 2. Effect of parameter values.
3.2. Experimental Details
The backbone network adopts the CLIP model (ViTB/16), and all images are uniformly resized to 224×224
resolution with black pixel padding. The batch size is set to
32, consisting of 8 identities with 4 images each. The token
length M is set to 30 on PRCC and 40 on LTCC. The AdamW
optimizer is employed, with an initial value of 3.5e-4, weight
decay of 5e-4, and training iterations of 40 epochs. On the
LTCC dataset, the camera-adaptive inference strategy from
[24] is adopted.
3.3. Comparison with State-of-the-Art Methods
The overall experimental results are presented in Table.1. It can be clearly observed that our proposed method
outperforms all existing approaches. On PRCC under the
clothing-changing setting, our method outperforms DIFIF
[24] by 3.2% in Rank-1 and 4.0% in mAP; on LTCC, it surpasses DIFIF by 13.0% in Rank-1 and 18.3% in mAP. We
achieved the SOTA performance without using the clothing
label.

at the attribute level improves Rank-1/mAP by 4.2%/3.5%,
validating clothing attribute suppression. Attribute-Level
Identity-Relevant Attributes (A P ). Enhancing identityrelevant attributes at the attribute level improves Rank-1/mAP
by 2.4%/1.2%, confirming the benefit of reinforcing age and
gender attributes. Caption-Level Disentanglement (CAP ).
Adding caption-level enhancement and suppression improves
Rank-1 by 2.0%, validating multi-granularity textual disentanglement. Masked Branch Disentanglement (M LB).
Removing attribute disentanglement from the masked branch
reduces Rank-1 by 0.6%, indicating residual clothing interference. Hyperparameter Impact. As shown in Fig.2,
mAP remains stable while Rank-1 peaks at a = 0.1. Performance degrades when prompt length exceeds 30, suggesting
overfitting.
3.5. Visualization of experimental results.
As shown in Fig.3, the retrieval results of the baseline
overly rely on clothing color and style (e.g., the first row focuses on yellow short sleeves, leading to misidentification).
Compared to the baseline, MAPL achieves higher retrieval
accuracy.
4. CONCLUSION

3.4. Ablation Studies
To evaluate each component, we conduct ablation studies
on the PRCC dataset, as shown in Table.2. Baseline Effectiveness. The baseline method (B) utilizes visual prompt tuning within a dual-branch architecture for person recognition.
v
The baseline supervised by Lid , Lm
id and Lcon , shows strong
performance, demonstrating the generalization capability of
visual prompt tuning and the effectiveness of the masked image branch. Attribute-Level Clothing Adversarial Learning (A N ). Adding BNNeck and adversarial suppression

This paper proposes a multi-granularity attribute prompt
learning framework (MAPL) for CC-ReID to learn clothinginvariant representations. Initially, visual prompts are incorporated into a dual-branch network. Then, attribute disentanglement is achieved by suppressing clothing interference
and enhancing identity features. Finally, branch fusion is performed in both the image space and the multimodal space,
separated by a BNNeck layer. Experiments show MAPL significantly improves performance.

10945
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:09 UTC from IEEE Xplore. Restrictions apply.

5. REFERENCES
[1] Alexander Hermans, Lucas Beyer, and Bastian Leibe, “In defense of the triplet loss for person re-identification,” arXiv
preprint arXiv:1703.07737, 2017.
[2] Hao Luo, “Bags of tricks and a strong baseline for deep person
re-identification,” IEEE, 2019.
[3] Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, and Shengjin
Wang, “Beyond part models: Person retrieval with refined part
pooling (and a strong convolutional baseline),” in Proceedings
of the European conference on computer vision (ECCV), 2018,
pp. 480–496.
[4] Xuelin Qian, Wenxuan Wang, Li Zhang, Fangrui Zhu, Yanwei
Fu, Tao Xiang, Yu-Gang Jiang, and Xiangyang Xue, “Longterm cloth-changing person re-identification,” in Proceedings
of the Asian conference on computer vision, 2020.
[5] Xinqian Gu, Hong Chang, Bingpeng Ma, Shutao Bai,
Shiguang Shan, and Xilin Chen, “Clothes-changing person
re-identification with rgb modality only,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 1060–1069.
[6] Yuxuan Liu, Hongwei Ge, Zhen Wang, Yaqing Hou, and
Mingde Zhao, “Clothes-changing person re-identification via
universal framework with association and forgetting learning,”
IEEE Transactions on Multimedia, vol. 26, pp. 4294–4307,
2023.
[7] Zhengwei Yang, Meng Lin, Xian Zhong, Yu Wu, and Zheng
Wang, “Good is bad: Causality inspired cloth-debiasing for
cloth-changing person re-identification,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 1472–1481.
[8] Ke Han, Shaogang Gong, Yan Huang, Liang Wang, and Tieniu Tan, “Clothing-change feature augmentation for person
re-identification,” in Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, 2023, pp. 22066–
22075.
[9] Qize Yang, Ancong Wu, and Wei-Shi Zheng,
“Person
re-identification by contour sketch under moderate clothing
change,” IEEE transactions on pattern analysis and machine
intelligence, vol. 43, no. 6, pp. 2029–2046, 2019.
[10] Peini Guo, Hong Liu, Jianbing Wu, Guoquan Wang, and
Tao Wang, “Semantic-aware consistency network for clothchanging person re-identification,” in Proceedings of the 31st
ACM international conference on multimedia, 2023, pp. 8730–
8739.
[11] Peixian Hong, Tao Wu, Ancong Wu, Xintong Han, and WeiShi Zheng, “Fine-grained shape-appearance mutual learning
for cloth-changing person re-identification,” in Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, 2021, pp. 10513–10522.

[15] Chunlei Peng, Boyu Wang, Decheng Liu, Nannan Wang,
Ruimin Hu, and Xinbo Gao, “Masked attribute description
embedding for cloth-changing person re-identification,” IEEE
Transactions on Multimedia, 2024.
[16] Shuang Li, Jiaxu Leng, Guozhang Li, Ji Gan, Haosheng Chen,
and Xinbo Gao, “Clip-driven cloth-agnostic feature learning
for cloth-changing person re-identification,” 2024.
[17] Shengxun Wei, Zan Gao, Chunjie Ma, Yibo Zhao, Weili Guan,
and Shengyong Chen, “Multiple information prompt learning
for cloth-changing person re-identification,” IEEE Transactions on Image Processing, 2024.
[18] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie,
Serge Belongie, Bharath Hariharan, and Ser-Nam Lim, “Visual
prompt tuning,” in European conference on computer vision.
Springer, 2022, pp. 709–727.
[19] Peike Li, Yunqiu Xu, Yunchao Wei, and Yi Yang, “Selfcorrection for human parsing,” IEEE Transactions on Pattern
Analysis and Machine Intelligence, vol. 44, no. 6, pp. 3260–
3271, 2020.
[20] Xu Li, Hefei Ling, Yuxuan Shi, Zongyi Li, and Ping Li, “Aapar: Clip-based adaptation and alignment for pedestrian attribute recognition,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. PP.
[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh,
Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda
Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in
International conference on machine learning. PmLR, 2021,
pp. 8748–8763.
[22] Qizao Wang, Xuelin Qian, Bin Li, Xiangyang Xue, and Yanwei
Fu, “Exploring fine-grained representation and recomposition
for cloth-changing person re-identification,” IEEE Transactions on Information Forensics and Security, 2024.
[23] Yan Huang, Qiang Wu, Zhang Zhang, Caifeng Shan, Yi Zhong,
and Liang Wang, “Meta clothing status calibration for longterm person re-identification,” IEEE transactions on image
processing, 2024.
[24] Yubo Li, De Cheng, Chaowei Fang, Changzhe Jiao, Nannan Wang, and Xinbo Gao, “Disentangling identity features from interference factors for cloth-changing person reidentification,” in Proceedings of the 32nd ACM International
Conference on Multimedia, 2024, pp. 2252–2261.

[12] Xin Jin, Tianyu He, Kecheng Zheng, Zhiheng Yin, Xu Shen,
Zhen Huang, Ruoyu Feng, Jianqiang Huang, Zhibo Chen, and
Xian-Sheng Hua, “Cloth-changing person re-identification
from a single image with gait prediction and regularization,”
in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 14278–14287.
[13] Fangbin Wan, Yang Wu, Xuelin Qian, Yixiong Chen, and
Yanwei Fu, “When person re-identification meets changing
clothes,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 2020, pp.
830–831.
[14] Kyung Won Lee, Bhavin Jawade, Deen Mohan, Srirangaraj
Setlur, and Venu Govindaraju, “Attribute de-biased vision
transformer (ad-vit) for long-term person re-identification,” in
2022 18th IEEE International Conference on Advanced Video
and Signal Based Surveillance (AVSS). IEEE, 2022, pp. 1–8.

10946
Authorized licensed use limited to: TIANJIN UNIVERSITY. Downloaded on June 09,2026 at 08:54:09 UTC from IEEE Xplore. Restrictions apply.


exec
/bin/zsh -lc "pdftotext -f 4 -l 6 'Multi-feature balanced network for clothes-changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 190 (2025) 107617

M. Mei et al.

which features the model has learned. Therefore, a MBNet is proposed
to balance multiple clothing irrelevant discriminative features, such as
head, legs, appearance, posture, and other information.
3. Methodology
This section presents the details of the proposed MBNet framework
as shown in Fig. 1, which consists of three parts: a global branch, a
clothing-unrelated branch, and a mask branch. Firstly, in the global
branch, complete RGB images are used as input and a knowledge transfer module is applied to learn more clothing-irrelevant fine-grained
features. Secondly, in the clothing-unrelated branch, the parts that are
unaffected by clothing are used as input to enhance the robustness
of clothing change problems. Meanwhile, a feature attention module
is introduced to mitigate the interference caused by the local noise.
Finally, in the mask branch, a cross module is applied to fuse the
learned features, aiming to dig out more contextual information.

Fig. 2. The structure of Knowledge Transfer Module (KTM).

3.1. Global branch

3.2. Clothing-unrelated branch

Existing work (Wang, Qian, Fu and Xue, 2022; Yao et al., 2022)
learns features to resist clothes changes from pedestrian images, simply concatenating the original features and clothes-irrelevant features
without knowledge fusion. However, these methods have two critical issues: the concatenation operation fixes the weights of the two
types of features and cannot dynamically adjust them based on the
samples. Additionally, the spatial alignment of the two features is not
constrained, leading to inefficient fusion. To address these, a Knowledge Transfer Module (KTM) is proposed, inspired by the collaborative optimization idea in knowledge distillation (Zheng et al., 2021):
the clothing-unrelated branch serves as the teacher network, extracting fine-grained discriminative features, while the global branch acts
as the student network, receiving guidance signals from the teacher
through the KTM to dynamically fuse clothing-irrelevant knowledge.
The structure is illustrated in Fig. 2. To highlight the clothing independent regions, knowledge from the global branch is combined with
the clothing-unrelated branch which aim to improve the performance
of clothes-changing ReID. In the clothing-unrelated branch, the framework learns fine-grained discriminative clothing-irrelevant features and
transfers them to the global branch to complement the cloth-unrelated
knowledge.
Specifically, the backbone network is utilized to extract the appearance feature 𝐹𝑔 from the RGB images. Then, the learned feature
𝐹𝑡𝑏 from the clothing-unrelated branch are taken to generate finer
features through the attention module, which mainly concentrates on
the pedestrian area. The formula is as follows in Eq. (1):

To address clothes-changing problems, the features from the
non/less affected body parts of pedestrians are explored. The most
decisive point accurately segments the human bodies into the affected
and non/less parts. In the MBUNet (Zhang et al., 2023), they used a
localization layer based on Spatial Transformer Network (STN) (Xu
et al., 2020), to roughly locate and cut out the partial image, and
then use it as the input. However, it incurred additional computational
costs. In the NPFM (Kweon & Cho, 2023), to concentrate on clothingunrelated identity information, they cropped 12.5% of the top and
bottom regions of the input RGB image, but this method has not been
effectively proven. In the paper, segmentation based on the anatomical
information (Leboeuf, Barre, Aminian, & Sangeux, 2023) from human
bodies has been proven feasible in the gait recognition method (Yao
et al., 2022). Especially, for a person of height H, his/her body can
be separated around some semantic positions, such as knees (0.285H),
pelvis (0.48H), waist (0.535H), and neck (0.87H) (Leboeuf et al., 2023).
It mainly focuses on the parts that less affected by clothes-changing: the
area between the head and shoulders, and the crus part extending from
the knees to the feet.
In the clothing-unrelated branch, the non/less affected body parts
are used as input to enhance robustness for clothes-changing problems.
Moreover, to further dig out fine-grained clothes-irrelevant features,
a feature attention module (FAM) is introduced to reduce the local
noise and suppress the background clutter, which aims to preserve the
pure feature map. The structure of FAM is shown in Fig. 3. Firstly, the
cropped part is inputted to extract the upper body features 𝐹𝑡 and lower
body features 𝐹𝑏 through the backbone network. The feature 𝐹𝑡𝑏 is
obtained by channel-wise concatenating two features, as demonstrated
in Eq. (4):

𝐴 = 𝑆𝑖𝑔𝑚𝑜𝑖𝑑(𝑐𝑜𝑛𝑣(𝑊1 , (Re 𝐿𝑈 (𝑐𝑜𝑛𝑣(𝑊2 , 𝐹𝑡𝑏 )))))

(1)

where 𝐴 represents the attention map, and 𝑊1 and 𝑊2 are the parameters of the first and second convolutional layers respectively. The
clothing-unrelated branch not only provides supplementary features
but also guides the global branch to focus on key regions that are
clothing-irrelevant through attention map 𝐴.
Next, the learned knowledge is transferred to the global branch
for knowledge fusion. With this approach, more robust pedestrian
representations could be obtained. The learned attention map 𝐴 is used
to channel weight to 𝐹𝑔 . And the fused feature 𝐹𝑓 is obtained through
the fast connection architecture. The final 𝐹𝑓 is then input into the
feature learning stage, as shown in Eq. (2):
𝐹𝑓 = 𝐹𝑔 ⊗ 𝐴 + 𝐹𝑔

Secondly, 𝐹𝑡𝑏 passes through a fully connected layer 𝐹1 for dimension reduction, a 𝑅𝑒𝐿𝑈 activation, another fully connected layer 𝐹2
for dimension incrementation, and a 𝑆𝑖𝑔𝑚𝑜𝑖𝑑 activation to generate a
+
clothing-unrelated attention map 𝐹𝑎𝑡𝑡
, as shown in Eq. (5):
+
𝐹𝑎𝑡𝑡
= 𝑆𝑖𝑔(𝐹2 (𝑅(𝐹1 𝐹𝑡𝑏 )))

Then, an inverse attention map
contrastive features in Eq. (6):

(2)

−
+
𝐹𝑎𝑡𝑡
= 1 − 𝐹𝑎𝑡𝑡

where ⊗ denotes the tensor elements multiplication.
Cross-entropy loss and hard sample triplet loss are used to reduce the feature differences of pedestrians with the same identity
in clothing-changing conditions, while reducing intra-class differences
and increasing inter-class differences, as Eq. (3):
𝐿𝑔𝑙𝑜 = 𝐿𝑡𝑟𝑖 + 𝐿𝑐𝑒

(4)

𝐹𝑡𝑏 = 𝐹𝑡 + 𝐹𝑏

(5)
−
𝐹𝑎𝑡𝑡

is generated to attend the
(6)

Consequently, this pair of attention maps are employed to the
feature 𝐹𝑡𝑏 to obtain identity feature 𝐹𝑐 and background feature 𝐹𝑏𝑝 .
The formula are shown in Eqs. (7) and (8):

(3)

+
𝐹𝑐 = 𝐹𝑎𝑡𝑡
⊗ 𝐹𝑡𝑏

4

(7)

Neural Networks 190 (2025) 107617

M. Mei et al.

Fig. 3. The structure of Feature Attention Module (FAM). It aims to obtain more refined clothing-irrelevant feature. It uses the features of upper and lower body to extract the
foreground features of pedestrians. And the background loss function 𝐿𝑏𝑔𝑟 is introduced to push the distance between the foreground features and background features.

−
𝐹𝑏𝑝 = 𝐹𝑎𝑡𝑡
⊗ 𝐹𝑡𝑏

lighting conditions, occlusion, and different pedestrians wearing similar
clothes. Mask features can provide more consistent identity information and increase the differences between classes, which improves the
discrimination ability of the model. As shown in Fig. 5, such coarse estimated human silhouette masks are not accurate enough to differentiate
between cloth-changing pedestrians.
To solve these problems, a cross-fusion module (CFM) is used that
employs two-way cross-attention to learn cross-clothes invariance by
combining both mask features and clothes-irrelevant features in a
two-branches network. The CFM module not only utilizes the complementary relationship between mask features and clothing-irrelevant
features, but also effectively overcomes the limitations of ReID models which rely on the appearance characteristics for a long time. In
this way, deep representations of pedestrians are obtained that could
distinguish the identities.
The feature maps 𝐹𝑚 and 𝐹𝑡𝑏 are extracted using the backbone
network, and each feature map is flattened and input into CFM module.
The details of the CFM module are shown in Fig. 4. First, the tokens of
the mask feature 𝐹𝑚 are projected to two independent matrices 𝐾1 and
𝑉1 as a set of values and keys. The clothing-irrelevant feature tokens
𝐹𝑐 are projected to another separate matrix 𝑄1 , as a set of queries. As
shown in Eq. (11):

(8)

where ⊗ denotes the tensor elements multiplication. However, the
background information will affect the feature 𝐹𝑐 . Suppressing the
background clutter is an effective method (Yang, Liu et al., 2023) for
ReID tasks. Different people have the similar backgrounds, and the
same person can also have different background, which might affect
the performance of model recognition. And, directly using the feature
𝐹𝑐 would limit the ability of model to learn more diverse features.
A background constraint loss 𝐿𝑏𝑔𝑟 is proposed to push the distances
between the background feature 𝐹𝑏𝑝 and global feature 𝐹𝑔 are large,
and bring the distances between the identity feature 𝐹𝑐 and global
feature 𝐹𝑔 small. During the testing stage, only the pure identity feature
mapping 𝐹𝑐 is retained for final pedestrian detection and discard the
feature 𝐹𝑏𝑝 . The formula of 𝐿𝑏𝑔𝑟 is as shown in Eq. (9):
1 ∑ ‖
‖2
‖
‖2
(‖𝐹 − 𝐹𝑐 ‖ + max{(𝑚 − ‖𝐹𝑔 − 𝐹𝑏𝑝 ‖ ), 0})
‖2
‖
‖2
𝑁 𝑛=1 ‖ 𝑔
𝑁

𝐿𝑏𝑔𝑟 =

(9)

where 𝑁 is the batch size. And 𝑚 is a margin parameter which is empirically set to 10 in the experiments. By minimizing the loss, identity
features and global features will get close to each other, while the background features will be away. It can intensify identity features while
reducing the interference of irrelevant features, which enhance the
performance in person ReID tasks. To minimize the distance between
positive sample pairs and maximize the distance between negative sample pairs, which address the clothes-changing problems. Cross-entropy
loss and hard sample triple loss are utilized in Eq. (10):
𝐿𝑐𝑢 = 𝐿𝑡𝑟𝑖 + 𝐿𝑐𝑒 + 𝜆𝐿𝑏𝑔𝑟

𝑄1 = 𝐹𝑐 ; 𝐾1 = 𝑉1 = 𝐹𝑚

(11)

Secondly, the correlation matrix is established using a dot product operation, and then the correlation attention score is obtained
through the softmax function, which represents the weight between
mask features and clothing-irrelevant features. Then the attention score
matrix is multiplied by 𝑉1 to obtain a vector, which is normalized using
the 𝐹𝑐 residual connection of clothing-irrelevant features to prevent
overfitting. The similarity feature 𝐹𝑐𝑚 is obtained as Eq. (12):

(10)

where 𝐿𝑡𝑟𝑖 and 𝐿𝑐𝑒 denote hard triple loss and cross-entropy loss,
respectively. To mitigate the interference of background clutter, a
background constraint loss function 𝐿𝑏𝑔𝑟 is adopted in the clothingunrelated branch, which helps make global and identity features invariant to background clutter, and parameter 𝜆 is set to 0.01 in the
experiment.

𝑄1 𝐾1 𝑇
𝐹𝑐𝑚 = 𝐿𝑁1 (𝑠𝑜𝑓 𝑡 max( √
)𝑉1 + 𝐹𝑐 )
𝑑1

(12)

where 𝐿𝑁1 represents the normalization operation. 𝑄1 and 𝐾1 denote
the query matrix and key matrix, respectively. 𝑑1 represents the scaling
factor of the feature dimension of the query and key. 𝑉1 indicates the
value matrix. Besides, a multi-head cross-attention mechanism with 8
parallel heads is employed, which enables the model to jointly understand the correlation between patch images and pedestrian contour
images from different angles. Where 𝑑1 represents the dimensions of
𝑄1 and 𝐾1 divided by 8, used to scale the attention size. The similarity
feature 𝐹𝑚𝑐 between clothing-irrelevant feature and mask feature is also
similar to the Eq. (12), as follows:

3.3. Mask branch
In the mask branch, the pedestrian silhouette masks are utilized
as input to further assist the network in learning clothing-irrelevant
features. The corresponding pedestrian silhouette masks are generated
through SCHP (Li et al., 2022). Pedestrian silhouette masks contain
rich clothes-invariant features which helps us to re-identification under
the circumstance of clothes-changing, as the body shape does not
change significantly for the same person. However, it ignores some
important appearance features, such as color attributes, facial features,
and so on. The model solely relies on pose information, which may
not be reliable, especially in complex environments, such as varying
5

𝑄2 = 𝐹𝑚 ; 𝐾2 = 𝑉2 = 𝐹𝑐

(13)

𝑄2 𝐾2 𝑇
𝐹𝑚𝑐 = 𝐿𝑁2 (𝑠𝑜𝑓 𝑡 max( √
)𝑉2 + 𝐹𝑚 )
𝑑2

(14)

Neural Networks 190 (2025) 107617

M. Mei et al.

Fig. 4. The detailed structure of the Cross Fusion Module (CFM). It aims to use a two-way cross-attention mechanism to integrate clothing-irrelevant features 𝐹𝑐 and mask features
𝐹𝑚 .

4. Experiments and analysis

where 𝐿𝑁2 represents the normalization operation. 𝑄2 and 𝐾2 denote
the query matrix and key matrix, respectively. 𝑑2 represents the scaling
factor of the feature dimension of the query and key. 𝑉2 indicates the
value matrix. Then channels are added to the two similar features,
utilizing the global average pooling to get a more refined feature 𝑓𝑎𝑝
according to Eq. (15):

4.1. Datasets and settings
The proposed approach is evaluated on three cloth-changing
datasets, i.e. LTCC (Qian et al., 2020), PRCC (Yang et al., 2021) and
VC-Clothes (Wan, Wu, Qian, Chen, & Fu, 2020).
The LTCC (Qian et al., 2020) dataset contains 17 138 images of 152
identities captured by 12 cameras. In the training set with 77 identities,
46 individuals appear in different clothing and the remaining 31 do
not change clothing, while the test set consists of 45 clothing-change
and 30 clothing-consistent identities. On average, each individual has
5 different outfits for LTCC.
The PRCC (Yang et al., 2021) dataset is a large indoor clotheschanging ReID dataset which contains 33 698 images of 221 identities
with three cameras. In cameras A and B, pedestrians of the same
identity wear the same clothes, but appear with different backgrounds.
In the view of camera C, the same identity wears different clothes
compared to the other two views. Each person is limited to two clothing
changes.
The VC-Clothes (Wan et al., 2020) dataset is a virtual dataset from
the game GTA-V in 4 different scenes, which consists of 512 identities
and 19 060 images. In VC-Clothes, 256 identities are used for the
training set and 256 are used for the testing set, with an average of
1 to 3 outfits for each identity.
We follow the original protocol for each dataset. For LTCC, two
test settings are used: the standard setting and the cloth-changing
setting (CC). The standard setting: all gallery samples are used to
calculate accuracy. The cloth-changing setting: only clothes-changing
gallery samples are used to calculate the accuracy. For PRCC and VCClothes, two test settings are introduced: the same-clothing setting (SC)
and the cross-clothing setting (CC). The same-clothing setting: only
clothes-consistent gallery samples are used to calculate accuracy. The
cumulative matching characteristics (CMC) and mean average precision
(mAP) are employed to evaluate the performance of our method.

(15)

𝐹𝑎𝑝 = 𝐺𝐴𝑃 (𝑐𝑜𝑛𝑐𝑎𝑡(𝐹𝑐𝑚 , 𝐹𝑚𝑐 ))

where GAP denotes global average pooling. To further learn the mask
body features beyond the clothing-invariant features, the mask branch
loss function 𝐿𝑚𝑎 is defined as Eq. (16):
(16)

𝐿𝑚𝑎 = 𝐿𝑡𝑟𝑖 + 𝐿𝑐𝑒

where 𝐿𝑡𝑟𝑖 and 𝐿𝑐𝑒 denote hard triple loss and cross-entropy loss,
respectively.
3.4. Total loss
The total loss function used to train the entire network could be
denoted as Eq. (17):
(17)

𝐿𝑡𝑜𝑡𝑎𝑙 = 𝛼𝐿𝑐𝑢 + 𝛾𝐿𝑚𝑎 + 𝐿𝑔𝑙𝑜

where the parameter 𝛼 and parameter 𝛽 are both set to 0.4 in the paper.
(1) Triplet loss: triplet loss with hard example mining is defined as
Eq. (18):
𝐿𝑡𝑟𝑖 =

𝑁
∑

[𝑑max (𝑥𝑖𝑎 , 𝑥𝑖𝑝 ) − 𝑑min (𝑥𝑖𝑎 , 𝑥𝑖𝑛 ) + 𝑚]

+

𝑖=1

(18)

where 𝑁 denotes the number of images in mini-batch, and 𝑑(⋅, ⋅)
represents the commonly used Euclidean distance. 𝑥𝑎 is the anchor
image, and 𝑥𝑝 is a positive example image from the same person as the
anchor image. 𝑥𝑛 is a negative example image with a different identity
from the anchor image. 𝑚 is a margin hyperparameter, representing the
boundary.
(2) Cross-Entropy Loss: cross-entropy loss with label smoothing
regularization (Fu, Zhang, & Wang, 2024) is defined as Eq. (19):
𝑦̂𝑖 = (1 − 𝜖)𝑦𝑖 +

𝜖
𝐾

4.2. Implementation details
The software tools are Pytorch 1.11.0, CUDA 11.3, and Python 3.8.
The hardware device is a GeForce RTX 3090 GPU.
For the backbone network, the CAL (Gu et al., 2022) is used as the
baseline. The input images are padded and resized into 384 × 192 and
the training batch size is set at 32 for each dataset, accommodating 4
pedestrians with 8 images per person. And the testing batch size is set
to 128. These images are further augmented with random horizontal
flipping, cropping, and erasing. The model is trained using the Adam
optimizer for 60 epochs, with a weight decay of 5 × 10-4. For LTCC
datasets, the initial learning rate is set to 1.5 × 10-4 and divided by
10 after 20, 40, and 60 epochs. For PRCC and VC-Clothes datasets, the

(19)

where 𝑦 denotes the ground-truth label, 𝐾 is the total number of categories, and 𝜖 is a smoothing parameter. And 𝑦̂𝑖 indicates the smoothed
label, while 𝑦𝑖 is the 𝑖th element of the true label 𝑦.
𝐿𝑐𝑒 = −

𝐾
∑

𝑦̂𝑖 log(𝑝𝑖 )

(20)

𝑖=1

where log(𝑝𝑖 ) is the predicted probability from the model.
6


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Journal Pre-proof

Multi-Granularity Dynamic Hierarchical Graphs for Video-Based
Person Re-Identification
Wei Zhao, Bingyi Zhou, Yongquan Wang, Yuanpeng Fan,
Guoyou Wang
PII:
DOI:
Reference:

S0893-6080(26)00699-4
https://doi.org/10.1016/j.neunet.2026.109238
NN 109238

To appear in:

Neural Networks

Received date:
Revised date:
Accepted date:

20 June 2025
25 May 2026
6 June 2026

Please cite this article as: Wei Zhao, Bingyi Zhou, Yongquan Wang, Yuanpeng Fan, Guoyou Wang,
Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification, Neural Networks (2025), doi: https://doi.org/10.1016/j.neunet.2026.109238

This is a PDF of an article that has undergone enhancements after acceptance, such as the addition of a cover page and metadata, and formatting for readability. This version will undergo additional copyediting, typesetting and review before it is published in its final form. As such, this version is no longer the Accepted Manuscript, but it is not yet the definitive Version of Record; we are
providing this early version to give early visibility of the article. Please note that Elsevier’s sharing
policy for the Published Journal Article applies to this version, see: https://www.elsevier.com/about/
policies-and-standards/sharing#4-published-journal-article. Please also note that, during the production process, errors may be discovered which could affect the content, and all legal disclaimers that
apply to the journal pertain.
© 2026 Published by Elsevier Ltd.

Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person
Re-Identification
Wei Zhaoa , Bingyi Zhoua , Yongquan Wanga , Yuanpeng Fana , Guoyou Wanga,∗
a School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, 1037 Luoyu

Road, 430074, Wuhan, China

Abstract
Video-based person re-identification (Re-ID) aims to identify video sequences of specific pedestrians captured in a
distributed camera system. The key to tackling this challenging task is to model rich temporal features in video
sequences. However, existing methods usually capture only fixed-length global temporal clues, making it difficult to
eliminate the effect of occlusion and accurately model the motion information of pedestrians. To address this limitation,
we propose a novel graph-based framework, namely dynamic hierarchical graph network(DHGN), to model temporal
features by adaptively capturing multi-granularity temporal clues. Specifically, we adaptively segment the video features
horizontally into several regions and construct graphs for the features at the same level. The features of each frame
at the same level are considered as graph nodes, which are then adaptively connected based on feature similarity.
By dynamically aggregating features from neighboring nodes in the same graph, DHGN is able to adaptively capture
temporal cues from different body parts. Furthermore, in order to obtain more robust matching results, we propose
a similarity weighted inference module (SWIM), which utilizes gallery-gallery similarity to modify the query-gallery
similarity matrix. Extensive experiments on four benchmarks clearly demonstrate the effectiveness of the proposed
method.
Keywords: video-based person re-identification, graph neural network, feature inference
1. Introduction
Person re-identification (Re-ID) aims to match the
same person in different cameras with great potential in
areas such as criminal investigation, forensics, and security. Existing Re-ID methods are mainly divided into two
categories, i.e.,image-based methods [1, 2, 3] and videobased methods [4, 5]. Image-based methods primarily rely
on appearance features, such as the color and texture of
clothing, and do not consider temporal information. In
contrast, video-based methods take tracklets as input for
more robust Re-ID by exploiting richer spatio-temporal
features.
Video-based methods critically focus on how to model
the temporal features of pedestrians. Generally, temporal
clues of different lengths in the video play distinct roles in
identification [6]. Long-term temporal clues can mitigate
the effects of occlusion, while short-term temporal clues
can model detailed motion patterns [7]. Furthermore, because of differing movement patterns and viewpoints, various hierarchical body parts of pedestrians exhibit differences in movement intensity and occlusion. For example,
∗ Corresponding author

Email addresses: wzhao97@hust.edu.cn (Wei Zhao),
zhou_bingyi@hust.edu.cn (Bingyi Zhou),
wangyongquan@hust.edu.cn (Yongquan Wang), ypfan@hust.edu.cn
(Yuanpeng Fan), gywang@mail.hust.edu.cn (Guoyou Wang)
Preprint submitted to Elsevier

as shown in the Figure 1 (a), some body parts (e.g. legs)
varied greatly during the walk, while other body parts (e.g.
torso) remained stable. Concurrently, various body parts
of the individual are intermittently occluded in different
frames, and these occluded parts subsequently reappear
in other frames. Therefore, processing these parts separately and adaptively capturing temporal clues can model
temporal features more efficiently.
However, most of the existing methods [8, 6, 9, 4] can
only model fixed-length global temporal features, thus failing to eliminate the effect of occlusion and accurately extract motion information of body parts. Accordingly, some
methods [7, 10] use uniform horizontal segmentation and
graph neural networks to model fine-grained temporal features. While these methods can adaptively extract temporal clues of different lengths, they cannot dynamically
segment features based on occlusion and body parts. In
addition, these methods have a fixed number of neighboring nodes, which can lead to the introduction of some lowcorrelation features.
Following the above discussion, we propose a novel
graph-based framework for video-based person Re-ID,
named dynamic hierarchical graph network (DHGN). As
shown in Figure 1 (b), DHGN dynamically divides global
features into segments of varying sizes based on visual
content and adaptively models the temporal features of
different body parts. The core component of DHGN is
June 7, 2026

the graph convolutional (GC) layer, which constructs individual graphs for different body parts. Subsequently, the
GC layer adaptively captures temporal clues for each body
part by aggregating features from adjacent nodes within
the same graph.
Specifically, for each frame, the feature map is first
evenly divided into fine-grained horizontal blocks, which
are then clustered based on feature similarity into adaptive
semantic regions of different sizes. These clustered regions
are treated as graph nodes. Graphs are constructed within
the same body-part level, rather than across arbitrary regions, to maintain semantic consistency. Edges are established using a mutual nearest neighbor criterion, so that
only mutually similar nodes are connected. This strategy yields a similarity-adaptive and dynamically updated
graph topology, enabling the GC layer to aggregate reliable
temporal information from semantically corresponding regions across frames.
In addition, due to the differences in pedestrians’ movement patterns and occlusion areas, a fixed number of
division is not appropriate. Thus, we apply a multigranularity segmentation scheme and construct multiple
graphs at each granularity. Finally, we interact information on the graph structure via graph neural networks
(GNNs) to adaptively model the corresponding temporal features. Moreover, most of the approaches use only
query-gallery similarities for matching and ignore gallerygallery relationships. Therefore, we design a similarity
weighted inference module (SWIM) which utilizes the relationship between query-gallery and gallery-gallery to optimize pedestrian matching.
Compared with other methods, DHGN has several advantages: (1) Adaptive capture of temporal clues from
different body parts. DHGN can adaptively aggregate
feature patches of varying sizes based on similarity, thus
avoiding the interference of factors such as occlusion, and
effectively modeling the motion information of pedestrians.
(2) The number of features involved in extracting temporal
information dynamically changes. When some nodes are
excessively different from other nodes, DHGN is able to dynamically adjust the number of neighbours, thus avoiding
information interference. (3) The GC layer can be added
to any convolutional neural network and specific granularity of graphs can be deployed depending on the resolution
of the features. As the depth of the network increases, the
resolution of the features decreases and the semantic information becomes more abstract. Therefore, feature maps
with low resolution are difficult to localize fine-grained targets. Accordingly, DHGN captures finer-grained temporal
clues in high-resolution features and fuses them with the
original features through residual connection. The main
contributions of this paper can be summarized as follows:

Figure 1: The temporal clue capture method. (a) Most methods can
only process Euclidean data, capturing fixed-length global temporal
cues such as long-term (fixed-frame intervals) and short-term (adjacent frames). (b) Our method employs graph neural networks that
can cope with non-Euclidean data and adaptively capture temporal
clues of different lengths for each body part.

mation of each part.
• We propose a similarity weighted inference module
(SWIM) to exploit pedestrian relationships in gallery
for matching.
• We evaluated DHGN and SWIM on four widely used
video re-identification benchmarks. The evaluation
demonstrates that the proposed method significantly
improves the baseline performance and outperforms
other state-of-the-art algorithms.

2. Related work
2.1. Video person Re-ID
Video person Re-ID aims to retrieval the target person
from the video gallery set. Compared with image-based
methods, video-based methods can extract rich spatiotemporal information in sequences, which helps to achieve
more robust Re-ID. Existing video-based methods mainly
use RNN [11, 12], 3D-CNN [13, 14], Transformer [15, 16]
and GNN [10, 17] to model temporal information. RNNbased methods typically aggregate spatial features by feeding CNN features into an RNN [18] or its variant LSTM
[19]. While RNNs are capable of capturing long-term temporal clues, RNN-based methods are difficult to train and
have limited effectiveness in modelling temporal information in Re-ID tasks [8]. 3D-CNN-based methods [6, 4]
mainly extract spatio-temporal information by 3D convolution. By employing multiple 3D convolutions with different temporal kernels, these methods can capture both

• We propose a novel framework, DHGN, for videobased person re-identification. DHGN adaptively
captures temporal clues at different granularities to
cope with occlusions and to model the motion infor2

long-term and short-term temporal clues [9]. However,
these methods are limited by the fixed temporal kernel
and cannot adaptively capture temporal clues of different lengths. Transformer-based approaches have demonstrated promising potential due to their powerful learning
capabilities. For example, Yu et al. [20] proposed TFCLIP, which utilizes a combination of vit [21] and pretrained CLIP to obtain robust video features. While this
method achieves excellent results, it introduces additional
information (i.e., pre-trained clip) and transformer-based
method has large parameters as well as training consumption.

combining the vectors of the top-k results, which are then
used to re-query the gallery. Shen et al. [25] generated
a revised ranking list by using the top-k nearest neighbors from the initial ranking list as new queries. Although
re-query-based methods can extend query representations,
their performance degrades significantly if the initial topk results contain noisy samples. Therefore, Ye et al. [26]
attempted to address the re-ranking problem by utilizing
common nearest neighbors. To address the false matching
problem in the k-nearest neighbors algorithm, Zhong et
al. [27] proposed the k-mutual-neighbor method, which revises the initial ranking list by calculating the Jaccard distance between samples. Recently, Cheb-GR [28] replaced
traditional k-nearest neighbor search with graph convolutions for re-ranking, significantly reducing computational
cost. However, the above method ignores the specific similarity scores, which may limit its effectiveness. In contrast,
our SWIM leverages similarity scores to compute weights,
enabling more comprehensive utilization of information.

2.2. Graph neural networks
In recent years, graph neural networks (GNNs) and
their variants have been successfully applied to video person Re-ID [22, 8, 10, 17]. Existing methods can be broadly
categorized into two categories: skeleton informationbased methods and horizontal segmentation-based methods. Skeletal information-based methods [22, 17] use human keypoints to extract features of body parts for constructing graphs. While such methods can localize body
parts more accurately, they require pose estimation algorithms to acquire the human keypoints, which leads to
additional consumption. Horizontal segmentation based
methods typically use pyramid pooling [23] to uniformly
segment the features extracted from the backbone for constructing the graph. For example, Yan et al [7] used horizontal segmentation to construct graph nodes with multiple granularities and utilized Hypergraph Neural Networks
(HGNN) to build more complex graphs in both spatial
and temporal scales. In [10], Pan et al. segmented image sequences to construct multi-granularity graphs and
proposed a graph pooling network(GPNet) to learn multigranularity graph representations. While these methods
can extract fine-grained pedestrian features, they are unable to adaptively segment features based on occlusion
versus body part, nor can they adaptively capture temporal clues. In addition, although these methods divide
the features at multiple granularities, it is difficult to localize subtle parts in low-resolution features.
Different from the above methods, our approach adopts
a hierarchical adaptive graph construction strategy. Instead of using fixed horizontal strips as nodes, we form
data-dependent part nodes by clustering fine-grained horizontal blocks based on feature similarity. Graph edges
are built within same-part groups using a mutual-nearestneighbor rule, and the graph topology is updated dynamically during training.

3. Dynamic hierarchical graph network
In this section, we present the details of the proposed DHGN, including the framework, graph convolutional (GC) layer, graph fully connected (GFC) layer, and
loss function.
3.1. The Framework
The framework of our method based on ResNet is presented in Figure 2. DHGN mainly consists of two components: the GC layer and the GFC layer. Specifically,
GC layers are inserted into Stage2 and Stage3 of ResNet
through residual connection, where the numbers of clusters are set to p = 4 and p = 2, respectively. The GFC
layer is inserted at the end of ResNet to process the extracted features and adaptively capture global temporal
clues. By combining multi-granularity local features with
global temporal features, DHGN can effectively model
temporal dependencies of different body parts for more
discriminative person identification. The details of the
GC layer and GFC layer are presented in the following
two subsections.
3.2. Graph Convolutional Layer
The objective of the GC layer is to adaptively capture
temporal clues from different body parts in the video. It
primarily consists of four parts: horizontal segmentation,
graph construction, graph propagation and Feature reconstruction.
Horizontal segmentation. As discussed in Section
1, different body parts in a video sequence often exhibit
different motion intensities and occlusion patterns. Modeling these regions separately is therefore beneficial for capturing discriminative temporal cues. To this end, as illustrated in Fig. 3, we adopt a two-stage horizontal partition–clustering strategy to obtain semantically consistent

2.3. Re-ranking for Re-ID
Re-ranking aims to modify the original sequence of
rankings through an automated and unsupervised approach. Currently, various Re-ranking methods have been
proposed. In [24], Chum et al. proposed an average query
expansion method that generates new query vectors by
3

Figure 2: The Overall Framework of DHGN. DHGN is constructed based on ResNet. We insert the GC layer (including horizontal segmentation, graph construction, graph propagation and feature reconstruction) with p=4 and p=2 into the Stage 2 and Stage 3 of ResNet,
respectively. The GFC layer (including graph construction, graph propagation and temporal graph pooling) is inserted at the end of ResNet.
In this figure, features of the same color represent feature maps from the same horizontal level after segmentation. The overall process
includes: (1) Sampling T-frame images and feed them into the ResNet to extract feature maps. (2) Feature maps are input into the GC layer
at stages 2 and 3 of the ResNet to adaptively extract multi-granularity features, which are then reconstructed and passed to the next stage.
(3) The features output by ResNet perform spatial pooling and subsequently feed into the GFC layer to capture global temporal clues. (4)
The final identity representation is obtained through temporal graph pooling.

Graph construction. To capture temporal clues in
a semantically consistent manner, we construct p partconsistent graphs according to the adaptive grouping results from horizontal partition and clustering. Each graph
corresponds to one semantic body-part level rather than
a fixed spatial strip, and models temporal relations only
among nodes belonging to the same part group. We then
employ GCN to model interactions among these part-level
features.
Let Gl = {Glk | 1 ≤ k ≤ p, k ∈ Z} denote the set of p
graphs for a video sequence at layer l. Each graph Glk =
(Vlk , Elk ) contains T nodes, where each node represents a
clustered part-level feature from one frame, and Elk is the
edge set. Since the nodes are obtained via similarity-driven
clustering rather than fixed partition, the graph nodes are
data-dependent and semantically aligned.
Instead of using fixed neighbors or similarity thresholds, we adopt a mutual nearest neighbor strategy to build
adaptive edges. Two nodes are connected only when they
are mutually among each other’s nearest neighbors in the
same-part graph, which helps suppress noisy or unreliable connections caused by occlusion or background interference. As node similarities evolve during training, the
neighbor relationships and graph topology are dynamically
updated, resulting in an adaptive graph structure. Mathematically, the edge set is defined as
n
o
el⟨ij⟩ = ⟨vli → vlj ⟩ | vli ∈ NK1 (vlj ), vlj ∈ NK2 (vli ) , (1)

Figure 3: Adaptive horizontal segmentation. Given sequence features, spatial average pooling is first applied to obtain frame-level
feature maps. Each map is then evenly divided into horizontal blocks,
which are further grouped by feature-similarity–based clustering into
adaptive semantic regions.

body-part regions. Specifically, given the feature tensor
Fl ∈ RC×T ×H×W at layer l (where C, T , H, and W are
its channels, frames, heights, and widths, respectively), we
first use avgpooling on it to get the Fla ∈ RH×W . Then,
we divide it uniformly into β horizontal parts along the
height and classify them into p groups according to the hierarchical clustering. Based on the grouping results, each
frame feature is divided into p parts in height to construct
the part-level feature. The whole video sequence generates Np = T × p part-level features, which we denote as
N
Vl = {vl1 , vl2 , ..., vl p }. Where part-level features located at
the same horizontal level can form p subsets of Vl , which
we denote as {Vlk |1 ≤ k ≤ p, k ∈ Z}.
4

where NK1 (vlj ) denotes the set of K1 nearest neighbors
of vlj within the same graph. Nearest neighbors are determined by cosine similarity between node features after
average pooling.
Graph propagation. After constructing the partconsistent adaptive graphs, we perform graph propagation
to model temporal dynamics for each body part across
frames. The main objective is to extract part-wise temporal features by enabling information interaction among
semantically corresponding regions over time. Since each
graph only contains nodes from the same body-part group,
the propagation process focuses on learning temporal patterns specific to that part rather than mixing heterogeneous regions. As shown in Figure 4, the proposed graph
convolutional network consists of two steps: neighbor information aggregation and node feature update. For a
node vlj , its neighbors are defined as all its connected predecessors in the same-part graph, denoted as N (vlj ). We
first aggregate features from its neighbors to obtain temporal contextual information:
mjl =

P

k k
k wl vl ,

∀vlk ∈ N (vlj ),

(2)

Figure 4: Illustration of feature propagation in the GC layer under
the setting of p = 2. H, W, C, and T represent the feature height,
width, channels, and frames, respectively. Nodes with the same color
belong to the same semantic body-part group. Information is propagated within each same-part graph through similarity-weighted aggregation and node feature updating.

where wlk is the weight of each neighbour, which is calculated as follows:
 

exp D flk , flj
 
 ,
(3)
wlk = P
j
k
k exp D fl , fl

Then, we designed a channel fusion method based on
the attention mechanism. This method can determine the
importance of a node’s neighborhood relative to the node
itself based on video content. For example, when occlusion
occurs, focus on its adjacent nodes to mitigate the negative
impact. The attention module is shown in Figure 5. It first
generates channel weights, which can be expressed as:

where D represents the cosine similarity between features. This similarity-weighted aggregation encourages
nodes to absorb information from temporally and semantically aligned regions, enabling more accurate part-level
temporal modeling.
The node feature is then updated by combining the
original node feature with the aggregated neighbor message:
n
 
 o
′
vlj = Conv1l vlj || Conv2l mjl
n
o
(4)
= sjl || njl ,

Zl1 = Conv3l (GAP (Sl + Nl )) ,
Zl2 = Conv4l (GAP (Sl + Nl )) ,

(5)

where Sl and Nl are the combinations of sl and nl in all
nodes, respectively. GAP denotes the global average pooling along the spatial and temporal dimension. Conv3l and
Conv4l are two convolutional layers used to generate independent channel selection weighing tensors (i.e., Zl1 and
Zl2 ∈ RC×1×1×1 ). The two weight tensors are further normalized as follows:

′

where vlj is the updated node feature of vlj and || denotes
the concat operation. Both Conv1l and Conv2l are learnable
weights of the two 1×1 convolutional layers inserted by the
batch normalization (BN) and ReLU activation functions.
This update preserves the current part representation while incorporating cross-frame context, strengthening part-wise temporal modeling and robustness to pose
variation and occlusion.
Feature reconstruction. After updating the node
features, we employ residual connection to pass them to
the next layer. Therefore, we need reconstruct the updated
nodes to the size of the original features. For all updated
node features, we first patch them together according to
their original positions to obtain the feature map Vl ∈
R2C×T ×H×W .

Wlk =

exp(Zlk )
k ∈ {1, 2} ,
exp(Zl1 ) + exp(Zl2 )

(6)

Wl1 and Wl2 is the weight value of Sl and Nl , respectively. After that, we aggregate Sl and Nl according to
the weights:
Yl = Wl1 ⊙ Sl + Wl2 ⊙ Nl ,

(7)

where ⊙ denotes the element-wise multiplication operation. Yl is the residual term at layer l. The final feature
map Fl ′ of layer l is obtained from Fl ′ = Fl + Yl .
5

′

where vgj ∈ RCf is the updated node feature of vgj .
Both FC1 and FC2 are learnable weights of the two fullyconnected layers inserted by the BN and ReLU activation
functions.
Temporal graph pooling. Existing methods typically use average pooling to derive the average features of
all frames to represent the entire video. However, this approach inevitably introduces some noisy frame features.
Therefore, we propose a simple and effective temporal
graph pooling method to exclude features of interfering
frames. Specifically, our approach employs a mutual nearest neighbor strategy to determine the neighbors of a node,
with the number of neighbors of each node represents the
similarity with other nodes. Thus, we can exclude frames
with too low number of neighbors to avoid interference
when computing the average feature. The final video feature can be expressed as:

Figure 5: Illustration of feature reconstruction in the GC layer under the setting of p = 2. Features with similar components are first
recombined to form a complete representation. Then, the original
features and their aggregated neighborhood features are divided into
two branches. By utilizing the generated attention weights, the features of each branch are adaptively reweighted and fused.

t

V =

(12)

where vgo ′ denotes the node with number of neighbors
larger than 1.

3.3. Graph Fully Connected Layer
After using the backbone with the GC layer to extract
the multi-granular features of each image frame, we then
capture the global temporal clues by employing the graph
fully connected (GFC) layer.
Graph construction. Similarly, we adaptively capture global temporal clues by constructing graph and using
GNN. For the output feature Ff inal ∈ RCf ×T ×Hf ×Wf at
the final layer of the backbone, we first perform average
pooling to obtain the global feature Fglobal ∈ RCf ×T . Subsequently, Fglobal is divided into T features, each of which
represents a node. Finally, we adaptively connect these
nodes based on the similarity between them to construct
the global graph Gglobal = (Vglobal , Eglobal ). Mathematically, the set of edges in the global graph can be denoted
as:

eg⟨ij⟩ = vgi → vgj |vgi ∈ NK1 (vgj ), vgj ∈ NK2 (vgi ) . (8)

3.4. Similarity Weighted Inference Module
Most existing matching methods treat gallery images
as unrelated, ignoring their inherent relationships. To address this, we propose the Similarity Weighted Inference
Module (SWIM), which adjusts the distance by considering the similarity between gallery images. The intuition
behind SWIM is that if a gallery image gi is similar to
a query image qi , then other gallery images similar to gi
should also be similar to qi .
SWIM primarily consists of four steps: initial similarity
calculation, neighbor search, neighbor weight calculation,
and final distance calculation.
Initial similarity calculation. DHGN first converts
all query images q and gallery images g into features Fq and
Fg respectively. Afterwards, we compute the cosine similarity between query-gallery and gallery-gallery to obtain
similarity matrices Dqg ∈ RNq ×Ng and Dgg ∈ RNg ×Ng .
Neighbor search. For a gallery image gi , we need
to find a set of similar images to compute its similarity
matrix. To eliminate interference from noisy images, we
adopted a mutual nearest neighbor strategy. Specifically,
we search for the similarity between gi and its mutual mnearest neighbor according to Dgg . After that, we form
them into a similarity matrix.
o
n
i (2)
i (x)
i
i (1)
, ..., Dgg
,
(13)
Dgg
(Nm ) = Dgg
, Dgg

Graph propagation. Based on Gglobal , we similarly
design a graph neural network to propagate graph information and update node features. We first aggregate all
predecessor node features of vgj to obtain the neighbour
information:
P
mjg = k wgk vgk , ∀vgk ∈ N (vgj ),
(9)

where wgk is the weight of each neighbour, which is calculated as follows:

exp D vgk , vgj
k
 ,
(10)
wg = P
k j
k exp D vg , vg
where D represents the cosine similarity between features.
Then, we update the node feature based on the neighbour
information as well as the original node’s feature, which is
defined as:


′
(11)
vgj = FC1 vgj || FC2 (mjg ) ,

1 X o′
v ,
t o=1 g

i
where Dgg
∈ R1×Ng denotes the similarity matrix for gi
i
with all gallery images. Dgg
(Nm ) ∈ R1×Nx represents the
set of mutual m-nearest similarities of gi .
Neighbor weight calculation. We can use the similarity between gi ’s mutual nearest neighbors and qi to
robustly represent the similarity between qi and gi . The

6

intuitive approach is to use the similarity scores between
gi and its nearest neighbors as weights, calculating the
weighted sum of the similarities between qi and gi ’s nearest
neighbors. However, this method results in higher similarity of images that have more mutual m-nearest neighbors.
i
Therefore, we perform the Softmax function on Dgg
(Nm )
to obtain the D̃gg (Nm ) so that the sum of their elements
is 1. Then, we replace the elements of each column in Dgg
to obtain the weight matrix D̂gg .
(
c,r
D̃gg
if r ∈ Nm (c)
c,r
D̂gg =
(14)
0
otherwise,

4. Experimentation
4.1. Experimental setup
Dataset. We evaluate our proposed model on four
widely used video-based person Re-ID datasets, i.e. MARS
[29], LS-VID [30], iLIDS-VID [31] and PRID-2011 [32].
MARS is currently the most widely used video person
Re-ID dataset, containing 17,503 video tracklets from 1261
identities and 3,248 distractor sequences. Its training set
consists of 625 identities and the testing set consists of 636
identities.
LS-VID is another large public dataset for video person Re-ID. This dataset was captured by 3 indoor cameras
and 12 outdoor cameras and contains 14,943 tracklets from
3,772 identities. Compared to other datasets, LS-VID contains more complex scenes, such as drastic changes in illumination due to indoor/outdoor scenarios.
iLIDS-VID dataset consists of 600 video sequences of
300 different identities, each with a length between 23 and
192 frames. Compared to the other datasets, this dataset
contains more occluded samples.
PRID-2011 dataset consists of 200 identities and 400
tracklets captured by two cameras. The length of each
tracklets is from 5 to 675 frames.
Evaluation metrics. We adopt the mean Average
Precision (mAP) [33] and Cumulative Matching Characteristics (CMC) [34] as the metrics to evaluate our method.
For the MARS and LS-VID datasets, we report both the
CMC and mAP. For iLIDS-VID and PRID-2011, since
they merely contain one correct match in the gallery set,
we only report their CMC values.
Implementation details. In this paper, all experiments are conducted on a RTX 3090 GPU. We extended
the pre-trained ResNet-50 to 3 dimensions (with the temporal kernel size set to 1) as our backbone. For the training
stage, we train the model for 150 epochs and adopt Adam
as our optimizer with the weight decay of 0.0005. The initial learning rate is set to 0.00035 and follows a learning
rate decay strategy. The weights of both loss functions λ1
and λ2 are set to 1. For the input image sequences, we resize them to 256 × 128 and augment them with horizontal
flipping and random erasing. The training batch size is set
to 32 and includes 4 identities per batch. For each original
video, we randomly sample 8 frames with stride 4 to construct a video clip for training. For the testing stage, we
used cosine similarity as a distance metric to match the
final representation.

c,r
where D̂gg
denotes the elements in column c and row r of
the D̂gg . Nm (c) denotes the mutual m-nearest similarities
of gc within the threshold.
Final distance calculation. Finally, we utilize Dqg
and D̂gg to compute the final distance. The modified similarity matrix is calculated as:

D̂qg = Dqg + Dqg × D̂gg .

(15)

D̂qg is the final similarity matrix and × denotes matrix
multiplication.
3.5. Loss Function
Following the previous studies [4] [5], we adopt two
kinds of losses to jointly train our model: triplet loss [8]
and cross-entropy loss. The cross-entropy loss Lce is defined as follows:
N

Lce = −

1 X
yi logpi
N i=1

(16)

where N is the batch size. yi represents ground truth
label and pi denotes the predicted score. The triplet loss
with hard mining Ltri is defined as follows, which consists
of anchor, positive and negative features:

Ltri =

N
X
i=1

max(0, m + max

p=1···N

− min

n=1···N



∥Hia − Hip ∥2

(17)

{∥Hia − Hin ∥2 })

where m is the margin parameter to control the distances.
Hia , Hip and Hin represent anchor, positive and negative
samples respectively. The overall objective function L of
our model is the combination of two losses:
L = λ1 Lide + λ2 Ltri ,

4.2. Comparison with State-of-the-arts
In this section, we compare the proposed method with
other state-of-the-art (SOTA) methods on four videobased person Re-ID benchmarks. The results are shown in
Table 1. On the MARS dataset, the mAP accuracy of our
method is 89.1% and the Rank-1 accuracy is 91.2%. In
terms of mAP metrics, our approach achieves competitive
results. It significantly outperforms other ResNet50-based
methods and is only surpassed by T2MEA [35] among

(18)

where λ1 and λ2 are the weights of two losses.

7


exec
/bin/zsh -lc "pdftotext -f 1 -l 8 'Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 148ms:
se two instances [38]. The final Jaccard distance matrix 𝑊𝑐𝑠𝑡
is obtained by the weighted fusion of matrices 𝑊𝑝𝑐1 , 𝑊𝑝𝑐2 , and 𝑊𝑔𝑙𝑜𝑏𝑎𝑙 :
1−𝜇
(𝑊𝑝𝑐1 + 𝑊𝑝𝑐2 ),
(16)
2
where 𝜇 is the parameter that controls the fusion ratio. Finally, we
perform DBSCAN clustering on the fused Jaccard distance matrix 𝑊𝑐𝑠𝑡 ,
thereby realizing pseudo-label assignment for the model.
𝑊𝑐𝑠𝑡 = 𝜇𝑊𝑔𝑙𝑜𝑏𝑎𝑙 +

𝑄𝑖𝑗 = {𝑞𝑖𝑗 |𝑅𝑖𝑗 ⟩𝜎}.
5

(20)

Pattern Recognition 179 (2026) 113676

Y. Chen et al.

Fig. 3. (a) T-SNE visualization of 20 randomly sampled identities from the test set. (b) Visualization of the intra-class and inter-class distance distribution based
on all test samples.

3.4. Discussion

4. Experiments

Discussion on the redundancy between 𝑐𝑙𝑏 and 𝑐𝑠𝑡 : 𝑐𝑙𝑏 aims to
learn discriminative identity features between samples, while 𝑐𝑠𝑡 is
designed to achieve semantic alignment of multi-granularity features
within a single sample. They form a pair of mutually orthogonal
loss functions, establishing a hierarchical and closed-loop collaborative
optimization mechanism in different optimization dimensions: 𝑐𝑠𝑡 performs semantic calibration of intra-sample features, providing a reliable
feature foundation with no semantic shift and stable identity information for 𝑐𝑙𝑏 . In turn, 𝑐𝑙𝑏 fully exploits the discriminative capability of
multi-granularity features in the high-quality calibrated feature space,
further improving the inter-class discrimination performance of the
model. The two losses are functionally complementary, non-interfering,
and mutually reinforcing, enabling the model to achieve collaborative
optimization in both cross-granularity semantic consistency and intersample identity discrimination. Therefore, no functional redundancy
exists between them.
Discussion on the Robustness of 𝑊𝑐𝑠𝑡 across datasets: The 𝑊𝑐𝑠𝑡
fusion mechanism adopts a weighted fusion strategy of global feature anchoring plus fine-grained feature supplementation. Specifically,
global features serve as the prototype representation of pedestrian
identities and provide a stable core anchor for clustering. Fine-grained
features are derived from the constraint positive feature set 𝑃 𝑗 , which is
dynamically updated during training and real-timely selects the patch
tokens with the highest matching degree to global semantics and partlevel semantics. This enables flexible adaptation to the distribution
characteristics of different datasets, thus endowing 𝑊𝑐𝑠𝑡 with favorable
robustness across various data distributions. Experiments demonstrate
that a fixed fusion ratio already yields excellent performance. Nevertheless, leveraging the dynamic nature of 𝑊𝑐𝑠𝑡 , a dynamically adjusted
fusion ratio may further enhance its adaptability to data distributions, and this direction warrants in-depth verification in subsequent
research.

4.1. Datasets and evaluation protocol
Datasets: We evaluated the proposed method on four widely-used
ReID datasets: Market-1501 [24], MSMT17 [25], PersonX [26], and
Veri-776 [27]. Their detailed statistics are summarized in Table 1.
Evaluation Metrics: The performance of the proposed method is
evaluated using the mean Average Precision (mAP) and the Cumulative
Matching Characteristic (CMC) curve. The CMC accuracy is reported
at Rank-1 (R1), Rank-5 (R5), and Rank-10 (R10). To ensure a fair
evaluation, no post-processing techniques are employed throughout the
experiments. Furthermore, we exclusively utilize global features for
person retrieval. On the one hand, since prevailing methods typically
rely solely on global features during testing, this setup ensures a fair
comparison. On the other hand, the extraction and fusion of multigranularity features significantly degrades inference speed, thereby
hindering the practical deployment efficiency of the model.
4.2. Implementations details and parameter setting
The Transformer backbone adopted in this paper is derived from
that of Transreid-SSL [49], which utilizes the ViT-Small/16 architecture [50]. This architecture consists of 12 Transformer layers, with 6
attention heads per layer and a feature dimension of 384. The backbone
is pre-trained on LUPerson, a large-scale unlabeled dataset [51]. Input
images are resized to 384 × 128, with a patch size of 16 × 16, resulting in 192 patches in total. We employ standard data augmentation
strategies including random horizontal flipping, random cropping, and
random erasing [52]. We adopt the SGD optimizer with a weight decay
of 5 × 10−4 and an initial learning rate of 3.5 × 10−4 , where the learning
rate is decayed by a factor of 0.1 every 20 epochs. For the DBSCAN
clustering algorithm, the maximum neighborhood distance (i.e., the
𝜖 parameter) used to identify two samples as neighboring ones is set
to 0.5, 0.7, 0.4 and 0.5 for the Market-1501, MSMT17, PersonX and
6

Pattern Recognition 179 (2026) 113676

Y. Chen et al.

Table 2
Comparison with SOTA methods on Market-1501 and MSMT17 datasets (best results are boldfaced).
Methods

Reference

SPCL [39]
ICE [31]
CC [10]
ISE [7]
PPLR [21]
LP [15]
RTMem [11]
DCMIP [40]
CC-FCM-ViT [41]
DSGL-ViT [12]
GL-NCPLR [22]
MPRD [42]
CGMAL [43]
PFCD [16]
ACWRCL [44]
CMLR-ViT [23]
MCCAN

NeurIPS’20
ICCV’21
ACCV’22
CVPR’22
CVPR’22
TIP’23
TIP’23
ICCV’23
AAAI’24
TIP’24
TIFS’24
TNNLS’25
PR’25
TCSVT’25
TMM’25
NN’26
Ours

Market1501

MSMT17

mAP

Rank-1

Rank-5

Rank-10

mAP

Rank-1

Rank-5

Rank-10

73.1
79.5
82.6
85.3
84.4
85.8
86.5
86.7
88.6
89.0
87.8
85.8
86.8
87.7
89.6
88.6
90.5

88.1
92.0
93.0
94.3
94.3
94.5
94.3
94,7
94.9
95.1
95.2
94.6
94.9
94.7
95.8
95.0
95.9

95.1
97.0
97.0
98.0
97.8
97.8
97.9
98.0
97.9
97.7
98.1
97.8
98.2
97.9
98.1
98.2
98.1

97.0
98.1
98.1
98.8
98.6
98.7
98.5
98.8
98.8
98.6
98.8
98.7
98.9
98.8
98.8
98.9
99.0

19.1
29.8
33.3
37.0
42.2
39.5
38.5
40.9
49.1
46.4
43.0
39.1
44.1
42.3
55.2
57.0
58.9

42.3
59.0
63.3
67.6
73.3
67.9
63.3
69.3
74.2
71.9
70.8
67.2
69.7
70.6
82.2
79.2
83.7

55.6
71.7
73.7
77.5
83.5
78.0
75.4
79.7
83.5
81.3
81.1
78.3
79.9
81.2
89.0
88.2
90.4

61.2
77.0
77.8
81.0
86.5
81.6
79.6
86.3
86.7
84.0
84.5
82.3
83.2
84.5
91.2
91.0
93.0

Table 3
Comparison with SOTA methods on PersonX (best results are boldfaced).
Method

Reference

mAP

Rank-1

Rank-5

Rank-10

SPCL [39]
CC [10]
AdaMG [45]
DCCT [34]
DHCCN [46]
MPRD [42]
PFCD [16]
ACWRCL [44]
MCCAN

NeurIPS’20
ACCV’22
TCSVT’23
TCSVT’23
TCSVT’24
TNNLS’25
TCSVT’25
TMM’25
Ours

72.3
84.7
87.6
87.6
86.7
86.1
89.3
89.9
90.6

88.1
94.4
95.0
95.0
94.4
95.0
95.3
95.7
95.8

96.6
98.3
98.8
98.7
–
98.6
98.9
98.9
98.9

98.3
99.3
99.5
99.4
–
99.9
99.6
99.6
99.7

4.3. Comparison with state-of-the-art methods
Comprehensive comparative experiments are conducted on four
benchmark datasets, namely Market-1501, MSMT17, PersonX, and
Veri776, where the proposed method is compared against the stateof-the-art (SOTA) approaches in the field of USL ReID. Detailed quantitative results are summarized in Table 2, Table 3, and Table 4.
Specifically, as presented in Table 2, the proposed method achieves
a mean average precision (mAP) of 90.5% and a Rank-1 accuracy of
95.9% on Market-1501, along with a mAP of 58.9% and a Rank-1
accuracy of 83.7% on MSMT17. These results outperform all competing
methods by a significant margin, representing a notable performance
breakthrough. In addition, as illustrated in Tables 3 and 4, the proposed
method attains mAP values of 90.6% and 47.0%, as well as Rank-1
accuracies of 95.8% and 90.9%, on PersonX and Veri776, respectively,
which outperforms the existing approaches. This further validates the
effectiveness and cross-dataset generalization capability of the MCCAN,
demonstrating robust overall competitiveness.
Furthermore, our proposed method outperforms state-of-the-art counterparts that also adopt fine-grained feature learning strategies — including PPLR [21], LP [15], GL-NCPLR [22], PFCD [16], and CMLRViT [23] — across all evaluated datasets. These results fully demonstrate that, compared with fine-grained learning methods without explicit feature alignment constraint mechanisms, our proposed MCCAN
can effectively maintain the semantic consistency across features of
different granularities under pseudo-label supervision, thereby enabling
more robust learning of discriminative fine-grained details.

Table 4
Comparison with SOTA methods on Veri776 (best results are boldfaced).
Method

Reference

mAP

Rank-1

Rank-5

Rank-10

PPLR [21]
RTMem [11]
AdaMG [45]
STDA [47]
DiDAL [48]
KDK-MPL [35]
ACWRCL [44]
DSFNet [36]
MCCAN

CVPR’22
TIP’23
TCSVT’23
TITS’24
TMM’24
TMM’24
TMM’25
InfFus’26
Ours

43.5
44.2
41.0
42.3
43.5
45.8
46.5
45.8
47.0

88.3
85.2
86.2
87.4
89.0
89.6
90.7
89.2
90.9

92.7
89.6
90.6
90.8
93.5
93.3
93.9
94.6
94.9

94.9
92.0
–
93.0
–
94.0
95.0
–
95.7

Veri776 datasets, respectively. These parameter settings are adopted
from the empirical values of existing prevailing methods. Among them,
the MSMT17 dataset involves more cameras and more complex scenes,
leading to a more dispersed distribution of features within the same
identity. Therefore, a larger value of 𝜖 (0.7) is required to effectively
cluster the features of the same identity. In contrast, datasets such
as Market-1501, PersonX, and Veri776 have relatively simple scenes
and compact feature distributions, so smaller values of 𝜖 (0.4, 0.5) are
adopted. The 𝜖 values for Market-1501 and MSMT17 are taken from
RTMem [11], while those for PersonX and Veri776 are adopted from
ACWRCL [44]. The total number of training epochs is set to 60. The
batch size is configured as 32, and the temperature parameter 𝜏 is fixed
at 0.07 based on common empirical practices in this research field.
Specifically, 𝐾 is set to 6 in Eq. (2), 𝑀 is set to 3 in Eq. (4), 𝜆1 is
set to 0.2 in Eq. (9) and 𝜇 is set to 0.8 in Eq. (16). Additionally, 𝜆 (in
Eq. (13)) and 𝜎 (in Eq. (20)) are set to 0.6 and 0.55, respectively. All
the aforementioned hyperparameter settings will be further elaborated
and discussed in the Parameter Analysis section. All experiments are
conducted on two NVIDIA RTX 3090 GPUs.

4.4. Ablation studies and visualization analysis
Ablation Studies: To evaluate the effectiveness of each component
in the MCCAN, we conduct comprehensive ablation experiments on
the Market-1501 and MSMT17 datasets, and the results are shown in
Table 5. Here, 𝑏𝑎𝑠𝑒 refers to the contrastive loss applied only to the
global features using RTMem. By comparing #1 and #2, we observe
that 𝑐𝑙𝑏 improved the mAP by 0.3% and 2.0%, and Rank-1 by 0.2%
and 0.8% on the Market1501 and MSMT17 datasets, respectively. This
indicates that the multi-granularity framework effectively facilitates the
learning of more refined local features and alleviates model feature
bias. Comparing #2, #3 and #4, we find that under the influence of
both 𝑐𝑠𝑡 and 𝑊𝑐𝑠𝑡 , the model achieved improvements of 0.7% and 3.4%
in mAP, and 0.4% and 1.3% in Rank-1 on the Market1501 and MSMT17
datasets, respectively. This demonstrates that 𝑐𝑠𝑡 and 𝑊𝑐𝑠𝑡 effectively
maintain semantic consistency during feature learning by imposing
alignment constraints on features of different granularities. Comparing
#4 and #5, we see that the model achieved improvements of 0.2% and
7

Pattern Recognition 179 (2026) 113676

Y. Chen et al.

Table 5
Ablation study on Market1501 and MSMT17.
Order(#)

Module
𝑏𝑎𝑠𝑒

1
2
3
4
5

✓
✓
✓
✓
✓

Market1501
𝑐𝑙𝑏
✓
✓
✓
✓

𝑐𝑠𝑡

✓
✓
✓

𝑊𝑐𝑠𝑡

✓
✓

MSMT17

CCF

mAP

Rank-1

mAP

Rank-1

✓

89.3
89.6
90.2
90.3
90.5

95.2
95.4
95.7
95.8
95.9

52.2
54.2
56.9
57.6
58.9

81.1
81.9
82.9
83.2
83.7

Fig. 4. Retrieval rank-list visualization: green indicates correct recognition, and red indicates incorrect recognition.

Fig. 5. Attention heatmap visualization: the darker the color, the greater the weight of the corresponding position. In each subfigure, from left to right: original
image, RTMem heatmap, and MCCAN heatmap.

8


codex
剩下三篇也读到了方法主体。为了避免只看方法结构而漏掉“作者怎么证明问题定义”的证据，我再补读几篇后半部分的消融和可视化页，重点看它们是否真的验证了重定义，而不是只报涨点。
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
/bin/zsh -lc "pdfinfo 'Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf' | sed -n '1,80p'" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Creator:         Elsevier
Producer:        Acrobat Distiller 9.2.0 (Windows)
CreationDate:    Mon Jun  8 13:41:42 2026 CST
ModDate:         Mon Jun  8 13:41:42 2026 CST
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           14
Encrypted:       no
Page size:       595 x 842 pts (A4)
Page rot:        0
File size:       2429090 bytes
Optimized:       no
PDF version:     1.7

exec
/bin/zsh -lc "pdftotext -f 8 -l 13 'Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
intuitive approach is to use the similarity scores between
gi and its nearest neighbors as weights, calculating the
weighted sum of the similarities between qi and gi ’s nearest
neighbors. However, this method results in higher similarity of images that have more mutual m-nearest neighbors.
i
Therefore, we perform the Softmax function on Dgg
(Nm )
to obtain the D̃gg (Nm ) so that the sum of their elements
is 1. Then, we replace the elements of each column in Dgg
to obtain the weight matrix D̂gg .
(
c,r
D̃gg
if r ∈ Nm (c)
c,r
D̂gg =
(14)
0
otherwise,

4. Experimentation
4.1. Experimental setup
Dataset. We evaluate our proposed model on four
widely used video-based person Re-ID datasets, i.e. MARS
[29], LS-VID [30], iLIDS-VID [31] and PRID-2011 [32].
MARS is currently the most widely used video person
Re-ID dataset, containing 17,503 video tracklets from 1261
identities and 3,248 distractor sequences. Its training set
consists of 625 identities and the testing set consists of 636
identities.
LS-VID is another large public dataset for video person Re-ID. This dataset was captured by 3 indoor cameras
and 12 outdoor cameras and contains 14,943 tracklets from
3,772 identities. Compared to other datasets, LS-VID contains more complex scenes, such as drastic changes in illumination due to indoor/outdoor scenarios.
iLIDS-VID dataset consists of 600 video sequences of
300 different identities, each with a length between 23 and
192 frames. Compared to the other datasets, this dataset
contains more occluded samples.
PRID-2011 dataset consists of 200 identities and 400
tracklets captured by two cameras. The length of each
tracklets is from 5 to 675 frames.
Evaluation metrics. We adopt the mean Average
Precision (mAP) [33] and Cumulative Matching Characteristics (CMC) [34] as the metrics to evaluate our method.
For the MARS and LS-VID datasets, we report both the
CMC and mAP. For iLIDS-VID and PRID-2011, since
they merely contain one correct match in the gallery set,
we only report their CMC values.
Implementation details. In this paper, all experiments are conducted on a RTX 3090 GPU. We extended
the pre-trained ResNet-50 to 3 dimensions (with the temporal kernel size set to 1) as our backbone. For the training
stage, we train the model for 150 epochs and adopt Adam
as our optimizer with the weight decay of 0.0005. The initial learning rate is set to 0.00035 and follows a learning
rate decay strategy. The weights of both loss functions λ1
and λ2 are set to 1. For the input image sequences, we resize them to 256 × 128 and augment them with horizontal
flipping and random erasing. The training batch size is set
to 32 and includes 4 identities per batch. For each original
video, we randomly sample 8 frames with stride 4 to construct a video clip for training. For the testing stage, we
used cosine similarity as a distance metric to match the
final representation.

c,r
where D̂gg
denotes the elements in column c and row r of
the D̂gg . Nm (c) denotes the mutual m-nearest similarities
of gc within the threshold.
Final distance calculation. Finally, we utilize Dqg
and D̂gg to compute the final distance. The modified similarity matrix is calculated as:

D̂qg = Dqg + Dqg × D̂gg .

(15)

D̂qg is the final similarity matrix and × denotes matrix
multiplication.
3.5. Loss Function
Following the previous studies [4] [5], we adopt two
kinds of losses to jointly train our model: triplet loss [8]
and cross-entropy loss. The cross-entropy loss Lce is defined as follows:
N

Lce = −

1 X
yi logpi
N i=1

(16)

where N is the batch size. yi represents ground truth
label and pi denotes the predicted score. The triplet loss
with hard mining Ltri is defined as follows, which consists
of anchor, positive and negative features:

Ltri =

N
X
i=1

max(0, m + max

p=1···N

− min

n=1···N



∥Hia − Hip ∥2

(17)

{∥Hia − Hin ∥2 })

where m is the margin parameter to control the distances.
Hia , Hip and Hin represent anchor, positive and negative
samples respectively. The overall objective function L of
our model is the combination of two losses:
L = λ1 Lide + λ2 Ltri ,

4.2. Comparison with State-of-the-arts
In this section, we compare the proposed method with
other state-of-the-art (SOTA) methods on four videobased person Re-ID benchmarks. The results are shown in
Table 1. On the MARS dataset, the mAP accuracy of our
method is 89.1% and the Rank-1 accuracy is 91.2%. In
terms of mAP metrics, our approach achieves competitive
results. It significantly outperforms other ResNet50-based
methods and is only surpassed by T2MEA [35] among

(18)

where λ1 and λ2 are the weights of two losses.

7

transformer-based approaches. On the LS-VID dataset,
the mAP and Rank-1 metrics of our method are 83.4%
and 90.2%, respectively, both of which achieve optimal results. On the iLIDS-VID dataset, DHGN achieves a Rank1 accuracy of 94.0%, outperforming other ResNet50-based
methods. It is worth noting that while TCViT [36] and
FAViT [37] achieve superior results on iLIDS-VID, our approach has a significant advantage in terms of the number of parameters. Furthermore, compared to these two
methods, DHGN also achieves superior results on both the
MARS and LSVID datasets. On the PRID-2011 dataset,
our method achieves 97.8% Rank-1 accuracy, which is significantly better than other SOTA methods.
Additionally, SWIM algorithm demonstrates significant performance improvements on the MARS and LSVID datasets, but shows no improvement on the iLIDSVID and PRID-2011 datasets. This is because each query
in the iLIDS-VID and PRID-2011 datasets has only one
correct match in the gallery, the relational modeling of
SWIM becomes less effective in these datasets. Therefore,
the performance improvement of SWIM on these datasets
is relatively limited compared to MARS and LS-VID.

LS-VID datasets. Furthermore, we observed that overly
fine-grained segmentation does not improve model performance. This may be attributed to the excessive segmentation of the image, which has the effect of destroying the
features of the pedestrian’s body parts.
Impact of the number of divisions. Our method
uniformly divides video features into β horizontal segments
and classifies them into p groups based on hierarchical clustering. Table 4 shows the performance of DHGN at different β values. As can be seen, as the number of divisions
increases, the model’s performance increases initially and
then decreases. This may be due to the fact that an excessively low number of divisions fails to cluster appropriate
results, while an excessively high number of divisions increases the difficulty of clustering.
Effect of GC layer insertion position. We compare the results of inserting GC layer (p = 2 and p =
4) into different stages of ResNet in Table 5. It can be
seen that inserting the GC layer at any stage significantly
improves the performance of the model. Moreover, the results of inserting the GC layer in the stage2 and stage3 are
also higher than the results of inserting the GC layer in
the stage1 and stage4 . This may be attributed to features
in stage2 and stage3 containing richer semantic information than stage1 . Compared to stage2 and stage3 , stage4
greatly increases the number of parameters, which makes
training more difficult. In addition, stage4 features are
more abstract and difficult to localize subtle targets, thus
unsuitable for segmenting fine-grained features.
Analysis of the maximum number of neighbors
K1 . Our method employs mutual nearest neighbor strategy to determine the neighbors of each node. Thus, the hyperparameter K1 controls the maximum number of nodes
involved in aggregation. In this section, we verify the effect of different K1 values for the proposed DHGN on the
MARS and LS-VID datasets. The results are shown in
Figure 6. It can be seen that, in the beginning, the performance of the model improves as the number of neighbours increases. However, the performance of the model
degrades when K1 > 3. This is due to the difficulty of
extracting enough temporal features with too few neighbours, while too many neighbours can lead to a failure of
adaptive capture of temporal clues.

4.3. Ablation studies
Model component analysis. In this section, we explore the contributions of the different components of our
model and report the results in Table 2. In this table, the
baseline (Base) represents the backbone trained with cross
entropy and triplet loss. “+GC ” denotes the addition of
graph convolutional layer(without attention) with p = 4
and p = 2 in stages 2 and 3 of the baseline. “+GC+Att
” denotes using the attention mechanism to aggregate the
features of the node and its neighbours when employing
the graph convolutional layer. “+GFC ” denotes the adaptive capture of global temporal clues using a GFC layer after the baseline extraction of sequence features. It can be
observed that when only GC is employed, the parameters
and FLOPs of the model exhibit a notable increase, yet
the performance enhancement is constrained. This phenomenon may be attributed to the dilution of the distinct
roles of nodes and their neighbours. In addition, our proposed GC and GFC layers can be effectively combined to
adaptively capture temporal clues at different hierarchical
levels. It is worth noting that SWIM significantly improves
the mAP metrics on the MARS dataset, while the effect
on the Rank-1 metrics is less pronounced. This may be
due to the reason that some of the images have only one
correct match.
Impact of the granularity. In this section, we incorporate GC layers with different p into the network to
investigate the effectiveness of multi-granularity features.
The results are illustrated in Table 3. In this table, {2, 4}
denotes adding GC layers with p = 2 and p = 4 on the
basis of deploying the GFC module. It can be seen that
using features with multi-granularity can significantly improve the performance of the model. When p = {2, 4}, our
method achieves the best results on both the MARS and

The effect of different K1 values
on MARS

The effect of different K1 values
on LS-VID

Figure 6: Analysis of the maximum number of neighbors K1 .

Analysis of the dynamic neighborhood number
8

Table 1: Performance comparison of our method with the State-of-the-art methods on four benchmark datasets.

Methods

Source

Backbone

Param.

AGRL [38]
MGH [7]
MGRA [39]
CTL [22]
BiCnet-TKS[6]
GRL [40]
SINet [41]
MSTAT [42]
CAVIT [43]
MFA [44]
MMA-GGA [45]
DSANet [46]
GPNet [10]
SDCL [5]
MS-STI[47]
TCViT [36]
FAViT [37]
T2MEA [35]
GAE-Net [48]
DHGN
DHGN+SWIM

TIP 20
CVPR 20
CVPR 20
CVPR 21
CVPR 21
CVPR 21
CVPR 22
TMM 22
ECCV 22
TIP 22
TCSVT 22
WACV 23
NN 23
CVPR 23
TCSVT 24
AAAI 24
PR 25
KBS 25
NN 25
-

ResNet-50
ResNet-50
ResNet-50
ResNet-50
ResNet-50
ResNet-50
ResNet-50
Transformer
Transformer
ResNet-50
ResNet-50
ResNet-50
ResNet-50
ResNet-50
ResNet-50
Transformer
Transformer
Transformer
ResNet-50
ResNet-50
ResNet-50

29.2M
24.9M
48.2M
30.8M
35.8M
26.5M
87.2M
78.7M
89.6M
24.8M
30.8M
30.8M

Methods

Table 2: Component analysis
MARS
FLOPs
mAP Rank-1
24.8M
32.6G
85.5
89.9
26.1M
34.7G
87.0
90.5
26.4M
34.7G
87.2
91.2
29.2M
32.6G
87.2
90.6
30.8M
34.7G
88.1
91.6
30.8M
34.7G
89.1
91.2

Param.

Base
Base+GC
Base+GC+Att
Base+GFC
DHGN
DHGN+SWIM

MARS
mAP Rank-1
81.9
89.5
85.8
90.0
85.9
88.0
86.7
91.4
86.0
90.2
84.8
91.0
86.2
91.0
85.3
91.8
87.2
90.8
85.0
90.4
85.4
91.0
86.6
91.1
86.0
90.5
86.5
91.1
87.2
92.7
87.6
91.7
87.4
90.4
89.9
95.9
87.7
91.1
88.1
91.6
89.1
91.2

LS-VID
mAP Rank-1
75.1
84.6
79.6
87.4
79.2
89.2
78.9
88.2
75.5
85.1
80.7
89.6
83.1
90.1
78.7
89.3
75.1
84.6
79.5
89.6
83.4
90.2

iLIDS-VID
Rank-1
84.5
85.6
88.6
89.7
90.4
92.5
93.3
93.3
93.3
87.7
88.8
92.7
94.3
95.3
92.1
94.0
94.0

PRID-2011
Rank-1
94.6
94.8
95.9
96.2
96.5
95.5
95.5
95.5
96.1
96.5
96.7
97.8
97.8

Table 4: Performance of DHGN under different division numbers.
LS-VID
mAP Rank-1
75.2
84.8
77.4
87.5
78.5
88.7
78.6
88.4
79.5
89.6
83.4
90.2

β
4
6
8
10
12

MARS
mAP Rank-1
87.6
91.5
87.7
91.5
88.1
91.6
87.8
91.5
87.4
91.0

LS-VID
mAP Rank-1
79.0
89.3
79.2
89.4
79.5
89.6
79.0
88.9
78.6
88.5

Table 3: Performance of DHGN with different feature granularities.

p

Param.

FLOPs

{2}
{3}
{4}
{2,4}
{2,6}
{3,5}
{4,8}
{2,4,6}
{2,4,8}
{3,5,7}

30.5M
30.5M
29.5M
30.8M
30.6M
30.8M
29.6M
30.9M
30.9M
30.9M

33.7G
33.7G
33.7G
34.7G
34.8G
34.7G
34.8G
35.8G
35.8G
35.8G

MARS
mAP Rank-1
87.5
90.9
87.3
90.9
87.2
90.8
88.1
91.6
87.8
91.3
87.9
91.3
87.4
91.1
88.0
91.6
87.8
91.2
87.6
91.2

Table 5: Results of inserting GC layers at different stages.

LS-VID
mAP Rank-1
79.0
89.2
78.7
89.0
78.7
88.9
79.5
89.6
79.4
89.0
79.4
89.3
78.9
89.0
79.4
89.3
79.3
89.1
79.1
89.0

Model

Stage

Param.

FLOPs

Base+GFC

stage1,2
stage1,3
stage1,4
stage2,3
stage2,4
stage3,4

29.2M
29.6M
30.6M
34.3M
30.8M
34.5M
35.5M

32.6G
34.8G
34.7G
38.0G
34.7G
38.0G
38.0G

DHGN

MARS
mAP Rank-1
87.2
90.6
87.6
91.1
87.7
91.3
87.5
91.2
88.1
91.6
87.6
91.1
87.7
91.0

LS-VID
mAP Rank-1
78.6
88.4
79.1
89.2
79.0
89.2
78.8
89.3
79.5
89.6
79.4
89.4
79.2
89.4

also degrade the performance of the model. When the K2
is 5, the model achieves optimal results on both datasets.
Analysis of the SWIM. Our SWIM module utilizes
the mutual m-nearest neighbors of the gallery images to
help compute the final distance. To explore the effect of
the number of m, we conducted experiments on the MARS
and LS-VID datasets. The results are shown in Table 6.
It can be seen that as the value of m increases, the mAP
metrics on both datasets also improve. At the same time,
on the MARS dataset, the Rank-1 metric shows a slight
decrease. This may be due to the fact that some images
in the MARS dataset have only one correct match and

K2 . In order to exclude nodes with excessive differences,
we set a parameter K2 to dynamically determine the number of nearest nodes. To explore the impact of different
parameter K2 , we conducted experiments on the MARS
and LS-VID datasets. The results are shown in Figure
7. It can be observed that excluding nodes with excessive
differences can effectively improve the performance of the
model. However, strict exclusion of difference nodes can
9

Additionally, we observed that an excessive bias toward a
particular loss can degrade model performance. The best
results are obtained when λ1 and λ2 are set to 1.

The effect of different thresholds
on MARS

Table 8: Analysis of loss functions with different weights.
MARS
LS-VID
Weight
mAP Rank-1 mAP Rank-1
λ2 = 0
67.3
78.2
31.2
39.6
λ2 = 0.5 87.6
91.0
77.1
86.3
λ1 = 1
λ2 = 1.5 88.1
91.3
78.4
88.6
λ2 = 2
87.7
90.8
77.6
87.2
λ1 = 0.5 88.0
90.9
78.2
88.4
λ1 = 1
88.1
91.6
79.5
89.6
λ2 = 1
λ1 = 1.5 87.9
91.2
77.9
87.5
λ1 = 2
87.4
90.8
76.3
86.2

The effect of different thresholds
on LS-VID

Figure 7: Analysis of the dynamic neighborhood number K2 .

other images with different identities interfere with the
results. Combining the results of the experiments on the
two datasets, we set the value of m to 4.

Analysis of different backbones. The proposed
GC and GFC layers can be inserted into different convolutional neural networks to improve model performance.
To investigate the effectiveness of the proposed method in
different backbone, we also conducted experiments based
on ResNet-101, ResNet-152 and DenseNet-121 [51]. The
results are shown in Table 9. It can be seen that the
proposed method shows excellent adaptability on different backbones. The performance of the model is further
improved when a better performing backbone is used.

Table 6: Analysis of SWIM.

Model

m

DHGN

2
3
4
5
6

DHGN+SWIM

MARS
mAP Rank-1
88.1
91.6
88.3
91.5
88.7
91.3
89.1
91.2
89.2
91.0
89.3
90.6

LS-VID
mAP Rank-1
79.5
89.6
81.7
89.8
82.4
90.0
83.4
90.2
83.2
90.2
82.8
89.9

4.4. Visualization
Node similarity visualization. To better demonstrate the ability of the proposed method to adaptively
capture temporal clues, we visualise the node similarity
scores of a set of images. In Figure 8, the first row represents the node similarity scores for the global features
learned by the GFC layer, and the second and third rows
represent the node similarity scores of the GC layer for
p = 2. It can be observed that when occlusion occurs,
long-term temporal clues can be captured based on the
similarity of global features, thus mitigating the effects of
occlusion. When occlusion is not present, short-term temporal clues can be captured, thus modelling motion information more effectively. Meanwhile, the GFC layer also
wastes features in the unobstructed regions of the image.
In contrast, the GC layer can divide the image into multiple regions, fully exploiting the temporal features of each

Comparison with other re-ranking methods. Table 7 reports the comparison between different re-ranking
strategies on the MARS and LS-VID datasets. Classical
query expansion (QE) and affinity-based methods (AIM)
yield noticeable gains in mAP with only marginal additional inference cost. KR and GCR further enhance retrieval accuracy, especially on LS-VID, at the expense of
significantly increased computational overhead. In contrast, the proposed SWIM re-ranking strategy achieves the
best overall mAP on both datasets while maintaining substantially lower time cost than other methods. These results demonstrate that SWIM provides a better trade-off
between effectiveness and efficiency for large-scale video
re-identification re-ranking.
Table 7: Comparison with other re-ranking methods.

Method

Time

DHGN
+QE [24]
+AIM [49]
+KR [27]
+GCR [50]
+SWIM

0.3s
4.3s
6.2s
15.6s
8.6s
3.8s

MARS
mAP Rank-1
88.1
91.6
88.6
91.3
88.9
91.0
88.7
91.3
89.0
91.5
89.1
91.2

LS-VID
mAP Rank-1
79.5
89.6
82.1
89.9
82.5
90.0
83.2
90.5
83.1
90.2
83.4
90.2

Table 9: Ablation results for different backbone networks.
MARS
LS-VID
Model
mAP Rank-1 mAP Rank-1
Base
83.6
88.7
73.8
84.0
DenseNet-121
DHGN
86.4
90.1
76.3
88.2
DHGN+SWIM 87.6
89.5
79.4
89.3
Base
85.5
89.9
75.2
84.8
ResNet-50
DHGN
88.1
91.6
79.5
89.6
DHGN+SWIM 89.1
91.2
83.4
90.2
Base
86.0
90.1
76.4
85.7
ResNet-101
DHGN
88.1
92.0
79.8
90.0
DHGN+SWIM 89.1
91.5
83.6
90.5
Base
86.2
90.6
77.1
86.2
ResNet-152
DHGN
88.3
92.2
80.6
90.2
DHGN+SWIM 89.2
91.8
84.0
90.8
Backbone

Analysis of loss weights. We jointly optimize our
model using cross-entropy loss and triplet loss, weighted by
λ1 and λ2 respectively. Table 8 presents the performance
variations of DHGN under different loss weight settings.
We observe that Lide and Ltri exhibit good compatibility
and can jointly enhance model performance. When using only a single loss, model performance declines sharply.
10

Figure 8: Node similarity visualization. We visualise the similarity scores of global and 2-granularity node features for a set of images. The
values in the first to eighth rows represent the similarity scores of this frame to T1 to T8 , respectively. Where red scores indicate the top-3
nodes in terms of similarity for that node, blue scores indicate the top-5 nodes in terms of similarity for that node, and - indicates the
neighboring nodes of that node.

part. As can be seen in Figure 8, there is a clear difference between the neighboring nodes in each row. When
the lower body of the pedestrian is occluded, the GC layer
is still able to model the motion information of the upper
body. By integrating the GC and GFC layers, DHGN is
able to efficiently capture multi-granularity temporal clues
for robust re-identification.
Activation maps visualization. In Figure 9, we visualize the channel activation maps of the baseline and
the DHGN. In this figure, the first row displays the original images at different time stamps. The second and third
rows show the channel activation maps for the baseline
and DHGN, respectively. As shown in Figure 9 (a), compared to the baseline, DHGN can accurately focus on target with vigorous motion while avoiding interference from
background information. In Figure 9 (b), we observe that
DHGN is able to mitigate the impact of occlusions and
focus on discriminating regions.
Feature distribution visualization. We utilize tSNE [52] to visualize the distributions of video-level features extracted by the baseline and DHGN. The visualization results are shown in Figure 10. In this figure, we randomly selected 15 pedestrians with different identities from
the MARS dataset. Each point represents a feature extracted from an 8-frame video segment, with different colors denoting different identities. It can be seen that compared to the baseline, DHGN can more effectively reduce
intra-class distances while increasing inter-class distances.
This further demonstrates the advantages of DHGN.

Figure 9: Channel Activation Heatmap for Baseline and DHGN. The
first row shows original video frames, while the second and third
rows present the corresponding activation maps of the Baseline and
DHGN, respectively. Compared with the Baseline, DHGN focuses
more consistently on discriminative body regions and shows stronger
robustness to occlusion across frames.

Baseline

DHGN

Figure 10: Feature distribution of the baseline and DHGN. Each
color denotes a different identity. Compared with the Baseline,
11 DHGN exhibits more compact intra-class clusters and clearer interclass separation, indicating improved feature discriminability.

5. Conclusion

[7] Yichao Yan, Jie Qin, Jiaxin Chen, Li Liu, Fan Zhu, Ying Tai,
and Ling Shao. Learning multi-granular hypergraphs for videobased person re-identification. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages
2899–2908, 2020.
[8] Jinrui Yang, Wei-Shi Zheng, Qize Yang, Ying-Cong Chen,
and Qi Tian. Spatial-temporal graph convolutional network
for video-based person re-identification. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 3289–3299, 2020.
[9] Jianing Li, Shiliang Zhang, and Tiejun Huang. Multi-scale 3d
convolution network for video based person re-identification. In
Proceedings of the AAAI Conference on Artificial Intelligence,
volume 33, pages 8618–8625, 2019.
[10] Honghu Pan, Yongyong Chen, and Zhenyu He.
Multigranularity graph pooling for video-based person reidentification. Neural Networks, 160:22–33, 2023.
[11] Yichao Yan, Bingbing Ni, Zhichao Song, Chao Ma, Yan Yan,
and Xiaokang Yang. Person re-identification via recurrent feature aggregation. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14,
2016, Proceedings, Part VI 14, pages 701–716. Springer, 2016.
[12] Zhen Zhou, Yan Huang, Wei Wang, Liang Wang, and Tieniu
Tan. See the forest for the trees: Joint spatial and temporal recurrent neural networks for video-based person re-identification.
In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 4747–4756, 2017.
[13] Xiaoke Jiang, Yu Qiao, Junjie Yan, Qichen Li, Wanrong Zheng,
and Dapeng Chen. Ssn3d: Self-separated network to align parts
for 3d convolution in video person re-identification. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 1691–1699, 2021.
[14] Jianing Li, Shiliang Zhang, and Tiejun Huang. Multi-scale temporal cues learning for video person re-identification. IEEE
Transactions on Image Processing, 29:4461–4473, 2020.
[15] Tianyu Zhang, Longhui Wei, Lingxi Xie, Zijie Zhuang,
Yongfei Zhang, Bo Li, and Qi Tian. Spatiotemporal transformer for video-based person re-identification. arXiv preprint
arXiv:2103.16469, 2021.
[16] Xuehu Liu, Chenyang Yu, Pingping Zhang, and Huchuan
Lu. Deeply coupled convolution–transformer with spatial–
temporal complementary learning for video-based person reidentification. IEEE Transactions on Neural Networks and
Learning Systems, 2023.
[17] Jiaxuan Lu, Hai Wan, Peiyan Li, Xibin Zhao, Nan Ma, and
Yue Gao. Exploring high-order spatio–temporal correlations
from skeleton for person re-identification. IEEE Transactions
on Image Processing, 32:949–963, 2023.
[18] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and
Yoshua Bengio.
Empirical evaluation of gated recurrent
neural networks on sequence modeling.
arXiv preprint
arXiv:1412.3555, 2014.
[19] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term
memory. Neural computation, 9(8):1735–1780, 1997.
[20] Chenyang Yu, Xuehu Liu, Yingquan Wang, Pingping Zhang,
and Huchuan Lu. Tf-clip: Learning text-free clip for video-based
person re-identification. In Proceedings of the AAAI conference
on artificial intelligence, volume 38, pages 6764–6772, 2024.
[21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk
Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al.
An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
[22] Jiawei Liu, Zheng-Jun Zha, Wei Wu, Kecheng Zheng, and Qibin
Sun. Spatial-temporal correlation and topology learning for person re-identification in videos. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages
4370–4379, 2021.
[23] Yang Fu, Yunchao Wei, Yuqian Zhou, Honghui Shi, Gao Huang,
Xinchao Wang, Zhiqiang Yao, and Thomas Huang. Horizontal
pyramid matching for person re-identification. In Proceedings

In this paper, in order to overcome the shortcomings of
existing methods in temporal feature extraction, we propose DHGN to adaptively capture temporal clues at each
part. We first construct the graph with features from the
same parts and connect the nodes based on feature similarity. Then, we implemented graph convolution for node
feature updating to adaptively capture temporal clues at
each part. In addition, we develop an attention mechanism for aggregating the features of nodes and neighbours to maintain the size of the feature map. Finally,
we capture global temporal clues and concatenate them
with multi-granularity temporal features. Except DHGN,
we also propose SWIM, which utilizes the similarity of images in the gallery to modify the query-gallery distance.
Experimental results on four datasets demonstrate the superiority of our DHGN and SWIM.
Declaration of interests
The authors declare that they have no known competing financial interests or personal relationships that could
have appeared to influence the work reported in this paper.
Acknowledgement
This document is funded by the Few-shot Continual
Learning Techniques for Complex and Variable Objects
6142113-JCKY0001.
References
[1] Zhiqi Pang, Chunyu Wang, Lingling Zhao, Yang Liu, and Gaurav Sharma. Cross-modality hierarchical clustering and refinement for unsupervised visible-infrared person re-identification.
IEEE Transactions on Circuits and Systems for Video Technology, 34(4):2706–2718, 2024.
[2] Yuxuan Liu, Hongwei Ge, Liang Sun, and Yaqing Hou. Complementary attention-driven contrastive learning with hard-sample
exploring for unsupervised domain adaptive person re-id. IEEE
Transactions on Circuits and Systems for Video Technology,
33(1):326–341, 2023.
[3] Zhenyu Cui, Jiahuan Zhou, Yuxin Peng, Shiliang Zhang, and
Yaowei Wang. Dcr-reid: Deep component reconstruction for
cloth-changing person re-identification. IEEE Transactions on
Circuits and Systems for Video Technology, 33(8):4415–4428,
2023.
[4] Abhishek Aich, Meng Zheng, Srikrishna Karanam, Terrence
Chen, Amit K Roy-Chowdhury, and Ziyan Wu.
Spatiotemporal representation factorization for video-based person reidentification. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 152–162, 2021.
[5] Chengzhi Cao, Xueyang Fu, Hongjian Liu, Yukun Huang,
Kunyu Wang, Jiebo Luo, and Zheng-Jun Zha. Event-guided
person re-identification via sparse-dense complementary learning. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 17990–17999, 2023.
[6] Ruibing Hou, Hong Chang, Bingpeng Ma, Rui Huang, and
Shiguang Shan. Bicnet-tks: Learning efficient spatial-temporal
representation for video person re-identification. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 2014–2023, 2021.

12


exec
/bin/zsh -lc "pdftotext -f 8 -l 11 'Multi-feature balanced network for clothes-changing person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Neural Networks 190 (2025) 107617

M. Mei et al.
Table 3
Ablation studies on different branches of our method on the LTCC dataset.
General

Baseline
GL branch
CU branch
MA branch
CU+MA branch
GL+CU+MA(MBNet)

Cloth-changing

Rank1

mAP

Rank1

mAP

74.2
76.1
77.3
76.9
76.7
77.9

40.8
40.3
43.2
40.4
42.3
43.1

40.1
42.1
41.6
41.1
43.1
44.6

18.0
18.7
20.8
19.4
22.2
22.7

Table 4
Ablation studies of three feature fusion strategies on the LTCC dataset.
General

Direct concatenation
Single-direction attention C→M
Single-direction attention M→C
Cross fusion module

Cloth-changing

Rank1

mAP

Rank1

mAP

76.1
76.5
76.1
77.9

42.4
42.1
41.8
43.1

43.5
42.9
42.6
44.6

22.4
21.9
21.6
22.7

method achieves at least 2% increase over baseline in both Rank-1
and mAP evaluations. Compared to the DCR-ReID (Cui et al., 2023)
method, our method improves by 1.1% on Rank-1 in same-clothes.
Compared to the best competitor CCFA (Han et al., 2023), Rank-1 and
mAP differ by 2.9% and 0.5%, respectively. Our approach does not
outperform the CCFA (Han et al., 2023) method on the PRCC dataset
because the dataset includes sketch images containing pedestrian mask
images. However, compared with other cloth-changing ReID methods,
our method also gets strong competitive results, which testifies the
effectiveness of our method.
Result on VC-Clothes (Wan et al., 2020). Also, comprehensive
experiments are executed to evaluate the performance of our proposed
MBNet on the VC-Clothes dataset, and the results are shown in Table 2.
As it could be seen, there is a significant performance gap between our
method and the traditional learning-based methods in cloth-changing.
In the same-clothes setting, RGA-SC (Zhang et al., 2020) and ISP (Zhu
et al., 2020) show great competitiveness, achieving Rank-1 and mAP
of 95.4%, 94.5% and 94.5%, 94.7%, respectively. In the cloth-change
setting, our method achieves 87.2% and 84.0% respectively on Rank-1
and mAP, which increasing by 6.2% and 2.3% compared to the optimal
CAL (Gu et al., 2022). Compared with the second-best MBUNet (Zhang
et al., 2023), our method is improved by 4.5% and 13.7% respectively on Rank-1 and mAP. Compared with the GI-ReID (Jin et al.,
2022) method based on gait, Rank-1 and mAP are improved by 22.7%
and 26.2%. The experiment results show that our method is also
applicable to clothing-consistent scenes and achieves competitive results, indicating that the proposed MBNet method can adequately mine
identity-related information in general person re-identification.

Fig. 5. Illustration of the pedestrian silhouette masks images. It describes the mask
image relative to the original RGB image in there datasets.

supplement the discriminative ability of clothing independent knowledge in the global branch, effectively eliminating the impact of clothing
changes.
Effectiveness of the Clothing-unrelated Branch. The clothingunrelated branch is tested. The results are summarized in Table 3.
CAL (Gu et al., 2022) is utilized as the baseline. Performance boosts
of 3.1%, 2.4%, 1.5%, and 2.8% are observed on two evaluation criteria, respectively, when combining the clothing-unrelated branch. This
suggested that the patch images of pedestrians can be used as input to
effectively learn clothing-irrelevant features (e.g., head, legs, pose, and
carrying object). In this branch, the learned features are more robust
and comprehensive in resisting the influence of clothing scenario. The
experiment results show that the clothing-unrelated branch effectively
learns fine-grained clothing-unrelated features, which helps validate
the performance of the model.
Effectiveness of the Mask Branch. As shown in Table 3, the mask
branch significantly outperforms the baseline. In the cloth-changing
setting, Rank-1 and mAP are increased by 1.0% and 1.4% respectively,
demonstrating the benefits of using pedestrian silhouette masks as
input. As shown in Fig. 5, pedestrian silhouette masks contain rich body
features. To effectively combine mask features and clothes-irrelevant
features, our model utilizes two-way cross-attention, thus significantly
overcoming the limitations of traditional person ReID that relies on
appearance information. In the mask branch, Rank-1 and mAP are increased by 2.5% and 1.3% respectively in the general setting, indicating
that the branch can better activate the potential of mask features in
clothes-changing.
Effectiveness of the Cross Fusion Module. To evaluate the effectiveness of the CFM approach compared to simpler strategies, experiments were conducted on the LTCC dataset, as presented in Table
4. In this context, ‘‘C → M’’ refers to the single attention interaction
from the clothing-unrelated branch to the mask branch, while ‘‘M →
C’’ indicates the single attention interaction from the mask branch to
the clothing-unrelated branch. The experimental results demonstrate

4.4. Ablation study
To gain more insights into our proposed MBNet, the influence is
explored of different branches based on the LTCC (Qian et al., 2020)
dataset. Then, the proposed method is compared with the baseline
model to comprehensively evaluate its effectiveness.
Effectiveness of the Global Branch. In the global branch, a knowledge transfer module (KTM) is introduced to transfer the clothing
invariance knowledge learned from clothing-unrelated branch to global
branch. It increases Rank-1 and mAP to 42.1% and 18.7% respectively,
in a cloth-changing scenario. This suggested that our proposed KTM
is effective compared to the methods that use simple concatenation of
knowledge features. It can highlight the clothes-unrelated areas and
8

Neural Networks 190 (2025) 107617

M. Mei et al.

Fig. 6. The impact of different hyper-parameters 𝛼 and 𝛾 on Rank-1 accuracy in the LTCC dataset.

Fig. 7. The visualization of the retrieval ranking results of baseline and our proposed model on LTCC and VC-Clothes datasets. The first column is the query image. The images
with green box denote the same person and red borders denote the different person.

that the proposed CFM method not only outperforms the two other
fusion strategies but also exhibits greater stability across various scenarios. This suggests that CFM is a more robust solution for effectively
integrating information from different branches.
Effectiveness of the hyper-parameters. Two hyperparameters 𝛼
and 𝛾 are used to balance the weights of the loss function in both the
clothing-unrelated branch and the mask branch, as shown in Eq. (17).
To explore the effect of these hyper-parameters on the model performance, experiments are conducted on the LTCC datasets. In the
experiments, the effect of loss weight parameters are explored in increments of 0.1, ranging from 0.1 to 1. The results are shown in Fig. 6.
When evaluating one of the parameters, the other hyper-parameter is
fixed at its optimal value. It can be observed that the MBNet achieves
the best performance when both 𝛼 and 𝛾 are set to 0.4 on the LTCC
dataset.
Visualization of retrieval results. To give an intuitive evaluation
of our method, the retrieval results of the 10 most similar images are
presented under the cloth-changing settings on LTCC and VC-Clothes
datasets, based on both the baseline network and the proposed MBNet,
as depicted in Fig. 7. The results indicate that the ranking results

given by MBNet outperform those of the baseline model in the clothchanging scenario. Our proposed method is more robust, as it can
effectively capture more identity related features. Our model is able
to effectively resist the interference of clothing changes and return the
correct retrieval images.
5. Conclusion
In this paper, a new Multi-feature Balance Network (MBNet) is
presented that strikes a good balance between clothing-unrelated and
clothing-related cues. Our framework mainly contains three branches.
The global branch aims to learn appearance features from the raw
RGB images. The clothing-unrelated branch receives the patch images
that are unaffected or less affected by the clothing as input, and combines clothes-irrelevant features including the head, leg, shoulder, and
neck, to exploit the robust local features. The mask branch effectively
learns the pose features of pedestrians and employs a cross-attention
mechanism to resist the influence of clothing changes. Further, to
focus on the foreground features of pedestrians, a background constraint loss function is introduced to suppress the learned background
features. In our framework, the three branches together effectively
9

Neural Networks 190 (2025) 107617

M. Mei et al.

enhance the distinguishing ability of the features, significantly improving clothes-changing ReID accuracy. Compared with existing clotheschanging ReID methods, the proposed method combines multiple discriminative features (e.g., head, leg, shoulder, pose, and background).
Balancing and coordinating them with each other to effectively handle
the cloth-changing problem. The experimental results show that the
MBNet is effective and improved performance has been achieved. Our
future work includes applying the framework to cross modal retrieval
tasks, such as infrared-RGB pedestrian re-identification, and further
optimizing model performance to enhance its robustness.

Leboeuf, F., Barre, A., Aminian, K., & Sangeux, M. (2023). On the accuracy of the
conventional gait model: Distinction between marker misplacement and soft tissue
artefact errors. Journal of Biomechanics, 159, Article 111774. http://dx.doi.org/10.
1016/j.jbiomech.2023.111774.
Li, Y., He, J., Zhang, T., Liu, X., Zhang, Y., & Wu, F. (2021). Diverse part discovery:
Occluded person re-identification with part-aware transformer. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition (pp. 2898–2907).
Li, M., Xu, P., Li, C.-G., & Guo, J. (2023). SiCL: Silhouette-driven contrastive learning
for unsupervised person re-identification with clothes change. http://dx.doi.org/10.
48550/arXiv.2305.13600, arXiv preprint arXiv:2305.13600.
Li, P., Xu, Y., Wei, Y., & Yang, Y. (2022). Self-correction for human parsing. IEEE
Transactions on Pattern Analysis and Machine Intelligence, 44(6), 3260–3271. http:
//dx.doi.org/10.1109/TPAMI.2020.3048039.
Li, W., Zhu, X., & Gong, S. (2018). Harmonious attention network for person reidentification. In Proceedings of the IEEE conference on computer vision and pattern
recognition (pp. 2285–2294).
Liao, S., Hu, Y., Zhu, X., & Li, S. Z. (2015). Person re-identification by local maximal
occurrence representation and metric learning. In Proceedings of the IEEE conference
on computer vision and pattern recognition (pp. 2197–2206).
Liu, Y., Ge, H., Wang, Z., Hou, Y., & Zhao, M. (2024). Clothes-changing person reidentification via universal framework with association and forgetting learning.
IEEE Transactions on Multimedia, 26, 4294–4307. http://dx.doi.org/10.1109/TMM.
2023.3321498.
Liu, F., Kim, M., Gu, Z., Jain, A., & Liu, X. (2023). Learning clothing and pose invariant
3d shape representation for long-term person re-identification. In Proceedings of the
IEEE/CVF international conference on computer vision (pp. 19617–19626).
Liu, X., Liu, K., Guo, J., Zhao, P., Quan, Y., & Miao, Q. (2024). Pose-guided
attention learning for cloth-changing person re-identification. IEEE Transactions on
Multimedia, 26, 5490–5498. http://dx.doi.org/10.1109/TMM.2023.3334975.
Nguyen, V. D., Mantini, P., & Shah, S. K. (2024). Contrastive clothing and pose
generation for cloth-changing person re-identification. In 2024 IEEE/CVF conference
on computer vision and pattern recognition workshops (pp. 7541–7549).
Peng, Y.-X., Tang, Y.-M., Lin, K.-Y., & Zheng, W.-S. (2024). Generalized intra-camera supervised person re-identification. IEEE Transactions on Circuits and Systems for Video
Technology, 34(6), 4516–4527. http://dx.doi.org/10.1109/TCSVT.2023.3340346.
Qian, X., Fu, Y., Jiang, Y.-G., Xiang, T., & Xue, X. (2017). Multi-scale deep learning
architectures for person re-identification. In Proceedings of the IEEE international
conference on computer vision (pp. 5399–5408).
Qian, Z., Lin, Y., & Du, B. (2025). Visible–infrared person re-identification via patchmixed cross-modality learning. Pattern Recognition, 157, Article 110873. http://dx.
doi.org/10.1016/j.patcog.2024.110873.
Qian, X., Wang, W., Zhang, L., Zhu, F., Fu, Y., Xiang, T., et al. (2020). Long-term
cloth-changing person re-identification. In Proceedings of the Asian conference on
computer vision.
Shu, X., Li, G., Wang, X., Ruan, W., & Tian, Q. (2021). Semantic-guided pixel
sampling for cloth-changing person re-identification. IEEE Signal Processing Letters,
28, 1365–1369. http://dx.doi.org/10.1109/LSP.2021.3091924.
Si, T., He, F., Li, P., & Gao, X. (2023). Tri-modality consistency optimization
with heterogeneous augmented images for visible-infrared person re-identification.
Neurocomputing, 523, 170–181. http://dx.doi.org/10.1016/j.neucom.2022.12.042.
Su, C., Zhang, S., Xing, J., Gao, W., & Tian, Q. (2016). Deep attributes driven multicamera person re-identification. In Computer vision–ECCV 2016: 14th European
conference, Amsterdam, the Netherlands, October 11-14, 2016, proceedings, part II 14
(pp. 475–491). Springer.
Suh, Y., Wang, J., Tang, S., Mei, T., & Lee, K. M. (2018). Part-aligned bilinear
representations for person re-identification. In Proceedings of the European conference
on computer vision (pp. 402–419).
Sun, Y., Zheng, L., Yang, Y., Tian, Q., & Wang, S. (2018). Beyond part models:
Person retrieval with refined part pooling (and a strong convolutional baseline). In
Proceedings of the European conference on computer vision (pp. 480–496).
Wan, F., Wu, Y., Qian, X., Chen, Y., & Fu, Y. (2020). When person re-identification
meets changing clothes. In Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition workshops (pp. 830–831).
Wang, Q., Qian, X., Fu, Y., & Xue, X. (2022). Co-attention aligned mutual cross-attention
for cloth-changing person re-identification. In Proceedings of the Asian conference on
computer vision (pp. 2270–2288).
Wang, G., Yang, S., Liu, H., Wang, Z., Yang, Y., Wang, S., et al. (2020). Highorder information matters: Learning relation and topology for occluded person
re-identification. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition (pp. 6449–6458).
Wang, G., Yuan, Y., Chen, X., Li, J., & Zhou, X. (2018). Learning discriminative
features with multiple granularities for person re-identification. In Proceedings of
the 26th ACM international conference on multimedia (pp. 274–282). New York, NY,
USA: Association for Computing Machinery, http://dx.doi.org/10.1145/3240508.
3240552.
Wang, L., Zhang, Y., Lu, T., Fang, W., & Wang, Y. (2022). Multi feature fusion
attention learning for clothing-changing person re-identification. IEICE Transactions
on Fundamentals of Electronics, Communications and Computer Sciences, 105(8),
1170–1174. http://dx.doi.org/10.1587/transfun.2021EAL2097.

CRediT authorship contribution statement
Mengqing Mei: Methodology, Funding acquisition, Conceptualization. Chun Ye: Writing – original draft, Visualization, Software,
Methodology. Zhiwei Ye: Writing – review & editing, Supervision,
Methodology, Funding acquisition. Fangyi Liu: Validation, Supervision, Software. Mang Ye: Writing – review & editing, Project administration, Investigation, Data curation. LingYu Yan: Resources, Project
administration, Investigation. Liye Mei: Project administration, Formal
analysis, Data curation. Ziyuan Chen: Validation, Formal analysis,
Data curation.
Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to
influence the work reported in this paper.
Data availability
Data will be made available on request.

References
Bansal, V., Foresti, G. L., & Martinel, N. (2022). Cloth-changing person re-identification
with self-attention. In Proceedings of the IEEE/CVF winter conference on applications
of computer vision (pp. 602–610).
Chen, J., Jiang, X., Wang, F., Zhang, J., Zheng, F., Sun, X., et al. (2021). Learning 3D
shape feature for texture-insensitive person re-identification. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition (pp. 8146–8155).
Cui, Z., Zhou, J., Peng, Y., Zhang, S., & Wang, Y. (2023). DCR-ReID: Deep component
reconstruction for cloth-changing person re-identification. IEEE Transactions on
Circuits and Systems for Video Technology, 33(8), 4415–4428. http://dx.doi.org/10.
1109/TCSVT.2023.3241988.
Fu, H., Zhang, K., & Wang, J. (2024). An adaptive self-correction joint training
framework for person re-identification with noisy labels. Expert Systems with
Applications, 238, Article 121771. http://dx.doi.org/10.1016/j.eswa.2023.121771.
Gao, S., Wang, J., Lu, H., & Liu, Z. (2020). Pose-guided visible part matching for
occluded person reid. In Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition (pp. 11744–11752).
Gu, X., Chang, H., Ma, B., Bai, S., Shan, S., & Chen, X. (2022). Clothes-changing person
re-identification with rgb modality only. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition (pp. 1060–1069).
Han, K., Gong, S., Huang, Y., Wang, L., & Tan, T. (2023). Clothing-change feature
augmentation for person re-identification. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition (pp. 22066–22075).
Hong, P., Wu, T., Wu, A., Han, X., & Zheng, W.-S. (2021). Fine-grained shapeappearance mutual learning for cloth-changing person re-identification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp.
10513–10522).
Huang, Y., Wu, Q., Zhang, Z., Shan, C., Huang, Y., Zhong, Y., et al. (2024). Meta
clothing status calibration for long-term person re-identification. IEEE Transactions
on Image Processing, 33, 2334–2346. http://dx.doi.org/10.1109/TIP.2024.3374634.
Jin, X., He, T., Zheng, K., Yin, Z., Shen, X., Huang, Z., et al. (2022). Cloth-changing
person re-identification from a single image with gait prediction and regularization.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition
(pp. 14278–14287).
Kweon, H.-J., & Cho, D. (2023). Cloth-changing person re-identification with noisy
patch filtering. IEEE Signal Processing Letters, 30, 334–338. http://dx.doi.org/10.
1109/LSP.2023.3262447.
10

Neural Networks 190 (2025) 107617

M. Mei et al.
Wu, J., Liu, H., Shi, W., Tang, H., & Guo, J. (2022). Identity-sensitive knowledge
propagation for cloth-changing person re-identification. In 2022 IEEE international conference on image processing (pp. 1016–1020). http://dx.doi.org/10.1109/
ICIP46576.2022.9897243.
Xu, B., He, L., Liao, X., Liu, W., Sun, Z., & Mei, T. (2020). Black re-id: A head-shoulder
descriptor for the challenging problem of person re-identification. In Proceedings of
the 28th ACM international conference on multimedia (pp. 673–681).
Xu, W., Liu, H., Shi, W., Miao, Z., Lu, Z., & Chen, F. (2021). Adversarial feature
disentanglement for long-term person Re-identification. In IJCAI (pp. 1201–1207).
Yang, Z., Lin, M., Zhong, X., Wu, Y., & Wang, Z. (2023). Good is bad: Causality inspired
cloth-debiasing for cloth-changing person re-identification. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition (pp. 1472–1481).
Yang, S., Liu, W., Yu, Y., Hu, H., Chen, D., & Su, T. (2023). Diverse feature
learning network with attention suppression and part level background suppression
for person re-identification. IEEE Transactions on Circuits and Systems for Video
Technology, 33(1), 283–297. http://dx.doi.org/10.1109/TCSVT.2022.3199394.
Yang, Q., Wu, A., & Zheng, W.-S. (2021). Person Re-identification by contour sketch
under moderate clothing change. IEEE Transactions on Pattern Analysis and Machine
Intelligence, 43(6), 2029–2046. http://dx.doi.org/10.1109/TPAMI.2019.2960509.
Yao, L., Kusakunniran, W., Wu, Q., Xu, J., & Zhang, J. (2022). Collaborative feature
learning for gait recognition under cloth changes. IEEE Transactions on Circuits and
Systems for Video Technology, 32(6), 3615–3629. http://dx.doi.org/10.1109/TCSVT.
2021.3112564.
Ye, M., Shen, J., Lin, G., Xiang, T., Shao, L., & Hoi, S. C. H. (2022). Deep learning
for person re-identification: A survey and outlook. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 44(6), 2872–2893. http://dx.doi.org/10.1109/
TPAMI.2021.3054775.
Ye, M., Shen, J., & Shao, L. (2021). Visible-infrared person re-identification via homogeneous augmented tri-modal learning. IEEE Transactions on Information Forensics
and Security, 16, 728–739. http://dx.doi.org/10.1109/TIFS.2020.3001665.
Yu, Z., Zhao, Y., Hong, B., Jin, Z., Huang, J., Cai, D., et al. (2022). Apparel-invariant
feature learning for person Re-identification. IEEE Transactions on Multimedia, 24,
4482–4492. http://dx.doi.org/10.1109/TMM.2021.3119133.

Zhang, Z., Lan, C., Zeng, W., Jin, X., & Chen, Z. (2020). Relation-aware global attention
for person re-identification. In Proceedings of the ieee/cvf conference on computer
vision and pattern recognition (pp. 3186–3195).
Zhang, G., Liu, J., Chen, Y., Zheng, Y., & Zhang, H. (2023). Multi-biometric unified
network for cloth-changing person re-identification. IEEE Transactions on Image
Processing, 32, 4555–4566. http://dx.doi.org/10.1109/TIP.2023.3279673.
Zhao, Z., Liu, B., Lu, Y., Chu, Q., Yu, N., & Chen, C. W. (2024). Joint
identity-aware mixstyle and graph-enhanced prototype for clothes-changing person
re-identification. IEEE Transactions on Multimedia, 26, 3457–3468. http://dx.doi.
org/10.1109/TMM.2023.3311143.
Zheng, K., Lan, C., Zeng, W., Liu, J., Zhang, Z., & Zha, Z.-J. (2021). Pose-guided
feature learning with knowledge distillation for occluded person re-identification. In
Proceedings of the 29th ACM international conference on multimedia (pp. 4537–4545).
New York, NY, USA: Association for Computing Machinery, http://dx.doi.org/10.
1145/3474085.3475610.
Zheng, Z., Yang, X., Yu, Z., Zheng, L., Yang, Y., & Kautz, J. (2019). Joint discriminative
and generative learning for person re-identification. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition (pp. 2138–2147).
Zhong, X., Han, X., Jia, X., Huang, W., Liu, W., Su, S., et al. (2024). ICLR: Instance
credibility-based label refinement for label noisy person re-identification. Pattern
Recognition, 148, Article 110168. http://dx.doi.org/10.1016/j.patcog.2023.110168.
Zhong, X., Lu, T., Huang, W., Ye, M., Jia, X., & Lin, C.-W. (2022). Grayscale
enhancement colorization network for visible-infrared person re-identification. IEEE
Transactions on Circuits and Systems for Video Technology, 32(3), 1418–1430. http:
//dx.doi.org/10.1109/TCSVT.2021.3072171.
Zhu, K., Guo, H., Liu, Z., Tang, M., & Wang, J. (2020). Identity-guided human semantic
parsing for person re-identification. In Computer vision–ECCV 2020: 16th European
conference, Glasgow, UK, August 23–28, 2020, proceedings, part III 16 (pp. 346–363).
Springer.
Zhu, J., Wu, H., Chen, Y., Xu, H., Fu, Y., Zeng, H., et al. (2024). Cross-modal grouprelation optimization for visible–infrared person re-identification. Neural Networks,
179, Article 106576. http://dx.doi.org/10.1016/j.neunet.2024.106576.

11


exec
/bin/zsh -lc "pdftotext -f 8 -l 11 'Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113676

Y. Chen et al.

Table 5
Ablation study on Market1501 and MSMT17.
Order(#)

Module
𝑏𝑎𝑠𝑒

1
2
3
4
5

✓
✓
✓
✓
✓

Market1501
𝑐𝑙𝑏
✓
✓
✓
✓

𝑐𝑠𝑡

✓
✓
✓

𝑊𝑐𝑠𝑡

✓
✓

MSMT17

CCF

mAP

Rank-1

mAP

Rank-1

✓

89.3
89.6
90.2
90.3
90.5

95.2
95.4
95.7
95.8
95.9

52.2
54.2
56.9
57.6
58.9

81.1
81.9
82.9
83.2
83.7

Fig. 4. Retrieval rank-list visualization: green indicates correct recognition, and red indicates incorrect recognition.

Fig. 5. Attention heatmap visualization: the darker the color, the greater the weight of the corresponding position. In each subfigure, from left to right: original
image, RTMem heatmap, and MCCAN heatmap.

8

Pattern Recognition 179 (2026) 113676

Y. Chen et al.

Fig. 6. Parameter tuning experiments: (a), (b) and (c) are the parameter tuning plots on the Market-1501 dataset, while (d), (e) and (f) are those on the MSMT17
dataset.
Table 6
Comparison of filtering degrees of the CCF module on 𝑓𝑝𝑎𝑡𝑐ℎ and 𝑓𝑝𝑎𝑟𝑡 at
different training stages (average number and proportion of filtered samples
per cluster).
epoch

Market-1501
𝑓𝑝𝑎𝑡𝑐ℎ

5
10
20
30
40
50

10.0
7.8
3.3
1.9
1.5
1.5

MSMT17
𝑓𝑝𝑎𝑟𝑡

76%
49%
19%
11%
8%
8%

The t-SNE feature distribution visualization results in Fig. 3(a)
demonstrate that, compared with the baseline method, the feature
clusters learned by the MCCAN method exhibit superior intra-class
compactness, with clearer separability between feature clusters corresponding to different identities. Furthermore, the distance distribution
visualization results in Fig. 3(b) further verify that the features extracted by the MCCAN method have smaller intra-class distances and
larger inter-class distances. By comparing the peak-to-peak distances
between intra-class and inter-class distances, it can be seen that the
peak-to-peak distance value of the MCCAN method is significantly
larger than that of the baseline method (𝑑2 > 𝑑1 ). These quantitative
results fully corroborate that the MCCAN method possesses stronger
feature extraction capability.
As shown in the retrieval rank-list in Fig. 4, the MCCAN method
significantly outperforms the baseline model RTMem in terms of Top-5
retrieval accuracy. This result fully demonstrates the retrieval superiority of MCCAN and strongly supports the feasibility of its practical
deployment.
The attention heatmap visualization results in Fig. 5 indicate that
the MCCAN method can more accurately focus on key detail regions
related to identity than the baseline model. This advantage stems from
its built-in alignment constraint mechanism—which can effectively
suppress the misalignment issue across multi-granularity features and
guide the model to learn more accurate fine-grained features.

10.7
10.1
6.7
4.0
3.6
3.6

𝑓𝑝𝑎𝑡𝑐ℎ
81%
64%
38%
25%
21%
21%

8.0
6.3
6.0
5.5
5.3
5.3

𝑓𝑝𝑎𝑟𝑡
69%
46%
38%
32%
29%
29%

8.1
8.0
8.3
9.0
9.1
9.1

65%
60%
55%
53%
42%
42%

1.3% in mAP, and 0.1% and 0.5% in Rank-1 on the Market1501 and
MSMT17 datasets, respectively. This indicates that the model, under
the influence of CCF, is able to maintain the semantic consistency in
the memory bank and thus obtain a more robust memory bank.
To improve the interpretability of the CCF module, we calculate
the average number and ratio of filtered fine-grained features per
cluster at different training stages on the Market-1501 and MSMT17
datasets, as shown in Table 6. In the early training stage, the model
has weak discriminability and the fine-grained features contain heavy
noise. Thus, the CCF module filters most noisy features for effective
alignment. As training proceeds and the model converges, the filtering
ratio drops notably. Notably, the filtering ratio of part-level features
𝑓𝑝𝑎𝑟𝑡 is overall higher than that of patch-level features 𝑓𝑝𝑎𝑡𝑐ℎ , mainly
because part-level division easily introduces background and noise.
Besides, MSMT17 involves more cameras and complex scenes, leading
to heavier noise in fine-grained features and stricter filtering by the CCF
module than on Market-1501. Although the CCF module discards some
fine-grained features, it only operates on fine-grained features while
fully preserving global features. Since sample diversity is maintained
by the global features, it does not decrease significantly. Instead, this
module effectively strengthens feature alignment, suppresses the learning of noisy details, and thereby improves the discriminative capacity
of features.
Visualization Analysis: To intuitively verify the effectiveness of the
proposed MCCAN method, four visualization validation experiments
are conducted on the Market-1501 dataset: t-SNE [53] feature distribution visualization (Fig. 3(a)), intra-class and inter-class distance
distribution visualization (Fig. 3(b)), retrieval rank-list visualization
(Fig. 4), and attention heatmap visualization (Fig. 5).

4.5. Parameter analysis
Analysis of parameter 𝜆: Parameter 𝜆 regulates the trade-off between the collaborative loss 𝑐𝑙𝑏 , which is designed for complementary
learning of multi-granularity information, and the constraint loss 𝑐𝑠𝑡 ,
which serves to enforce semantic alignment across different granularities. To evaluate its impact on model performance, we conduct
extensive experiments by varying the value of 𝜆 on two large-scale
benchmarks, i.e., Market-1501 and MSMT17. The corresponding results
are presented in Fig. 6(a) and Fig. 6(d), respectively. Experimental
observations reveal that an excessively small 𝜆 weakens the alignment
constraint, thereby failing to guarantee the reliability of the learned
fine-grained features. In contrast, an overly large 𝜆 overemphasizes
the significance of semantic alignment between different granularities
during training, which renders the model overly conservative and
impairs the learning of discriminative fine-grained details. The model
achieves the best performance when 𝜆 = 6. Based on the comprehensive
experimental results, we set the default value of 𝜆 to 0.6.
9

Pattern Recognition 179 (2026) 113676

Y. Chen et al.

Fig. 7. Parameter tuning experiments: (a) and (b) are the parameter tuning experiments for parameter 𝜆1 on the Market-1501 and MSMT17 datasets, respectively.

Fig. 8. Parameter tuning experiments: (a) and (b) are the parameter tuning experiments for parameter 𝐾 on the Market-1501 and MSMT17 datasets, respectively.
Table 7
Parameter tuning experiments for Parameter 𝑀.

Analysis of parameter 𝜎: Parameter 𝜎 is used to modulate the
proportion of low-confidence fine-grained features filtered out from the
memory bank in the CCF module, and its value directly affects the
quality and diversity of features in the memory bank. The experimental
results presented in Fig. 6(b) and 6(e) show that: when the value of 𝜎 is
excessively small, it is difficult to ensure the semantic consistency of the
fine-grained feature memory bank, thereby leading to unsatisfactory
model performance; when 𝜎 falls within the range of 0.45–0.55, the
model performance tends to be stable; when 𝜎 = 0.55, the model
achieves the optimal performance; in contrast, when 𝜎 exceeds 0.55,
the performance exhibits a significant downward trend. This is because
an overly large 𝜎 excessively filters out the fine-grained features in the
memory bank, resulting in the loss of feature diversity, thereby impairing the model’s ability to learn the fine-grained differences between
different identities. Based on the above experimental results, we set the
default value of 𝜎 to 0.55 in this paper.
Analysis of parameter 𝜇: Parameter 𝜇 is used to modulate the fusion
ratio between semantically consistent fine-grained features and global
features during the clustering process: the larger the value of 𝜇, the
higher the model’s emphasis on global features; the smaller the value
of 𝜇, the more it emphasizes the importance of fine-grained features
and semantic alignment. The experimental results presented in Fig. 6(c)
and 6(f) show that: when 𝜇 = 0.8, the model achieves optimal performance; when 𝜇 < 0.8, the model performance exhibits a continuous
downward trend—this is because excessive emphasis on fine-grained
feature learning and semantic alignment during the clustering process

𝑀

𝑀=2
𝑀=3
𝑀=4

Market-1501

MSMT17

mAP

Rank-1

mAP

Rank-1

90.3
90.5
89.7

95.7
95.9
95.3

58.0
58.9
52.0

83.1
83.7
81.3

introduces a large amount of redundant noise, undermining the accuracy of the clustering results. Conversely, when 𝜇 > 0.8, the model tends
to suffer from a global perspective bias, i.e., over-reliance on global
features while neglecting discriminative fine-grained information, thus
limiting the further improvement of model performance. Based on the
comprehensive above experimental analysis, we set the default value
of 𝜇 to 0.8 in this paper.
Analysis of parameter 𝜆1 : Parameter 𝜆1 controls the balance between local and global features within the multi-granularity collaborative loss. The results of the sensitivity analysis are illustrated in Fig.
7. It can be observed that when 𝜆1 is in the range of 0.1 to 0.4, the
model performance, albeit with some fluctuations, remains superior.
This is attributed to the fact that hierarchical fine-grained features
can effectively complement information from different perspectives,
thereby reducing feature bias. However, when 𝜆1 > 0.4, a significant
performance drop is observed. This is because overemphasizing the
learning of detailed features causes the model to drift away from
10

Pattern Recognition 179 (2026) 113676

Y. Chen et al.

Table 8
Comparison of computational overhead.
Method

Params

FLOPs

Training time (per epoch)

Inference time (per image)

CC [10]
RTMem [11]
PFDS [16]
MCCAN

23.53M
23.52M
23.72M
24.78M

4.09G
4.09G
8.16G
4.13G

166 s
215 s
298 s
367 s

2,94 ms
2.94 ms
3.63 ms
3.03 ms

CRediT authorship contribution statement

global semantics and easily introduces noise. Extensive experiments
demonstrate that the model achieves optimal performance when 𝜆1 =
0.2, and thus, this value is adopted as the default setting.
Analysis of parameter 𝐾 and 𝑀: Parameters 𝐾 and 𝑀 respectively regulate the patch-based fine-grained features and part-based
fine-grained features output by the two branches of the network. Specifically, when the value of 𝐾 increases and the value of 𝑀 decreases, the
fine-grained feature representations output by the model will cover a
broader perspective, but it may also weaken the ability to learn features
of local tiny regions. Conversely, when the value of 𝐾 decreases and the
value of 𝑀 increases, the model’s focus will shift to local tiny regions,
yet this mode is accompanied by the risk of introducing more noise.
Notably, the combination strategy of 𝐾 and 𝑀 plays a crucial role in
the performance of the model. A reasonable parameter combination
can achieve the complementary effect of different feature perspectives,
which not only effectively alleviates the overfitting problem in the
process of fine-grained feature learning from a single perspective, but
also facilitates the MCCAN model in aligning the semantics between
fine-grained features, thereby maintaining the semantic consistency of
the model during the learning process. The experimental results in Fig.
8 and Table 7 consistently demonstrate that the model can achieve the
optimal performance when 𝐾 = 6 and 𝑀 = 3.
We compared MCCAN with several SOTA methods in terms of the
number of parameters (Params), floating point operations (FLOPs),
training time, and inference time under the same experimental setup,
with detailed results shown in Table 8. As can be observed from
the table, the increased computational overhead of MCCAN is mainly
reflected in the number of parameters and training time. This is because the multi-branch and multi-output architecture leads to a larger
number of parameters, while the complex loss computation and feature
processing modules based on multiple memory banks further prolong
the training process. Although MCCAN introduces extra computational
overhead to some extent, it achieves significant performance improvements over these methods. This demonstrates that sacrificing training
efficiency for stronger representation ability is reasonable and effective.

Yanbing Chen: Writing – review & editing, Writing – original draft,
Funding acquisition, Conceptualization. Lingyi Guo: Writing – original
draft, Visualization, Project administration, Methodology, Conceptualization. Zhixin Tie: Writing – review & editing, Conceptualization.
Yinghong Xu: Writing – review & editing, Supervision. Hao Sheng:
Writing – review & editing, Funding acquisition.
Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to
influence the work reported in this paper.
Acknowledgments
This study is partially supported by the National Key R&D Program of China (No. 2022YFC3803600), the National Natural Science
Foundation of China (No. 62372023), and the Open Fund of the State
Key Laboratory of Software Development Environment (No. SKLSDE2023ZX-11). This research was supported by the Research Start-up
Funds of Hangzhou International Innovation Institute of Beihang University under Grant No. 2024KQ012. Thank you for the support from
HAWKEYE Group.
Data availability
Data will be made available on request.

References
[1] Y. Lin, L. Xie, Y. Wu, C. Yan, Q. Tian, Unsupervised person re-identification
via softened similarity learning, in: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2020, pp. 3390–3399.
[2] Y. Zhai, Q. Ye, S. Lu, M. Jia, R. Ji, Y. Tian, Multiple expert brainstorming for
domain adaptive person re-identification, in: Computer Vision–ECCV 2020: 16th
European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII
16, Springer, 2020, pp. 594–611.
[3] S. Xuan, S. Zhang, Intra-inter domain similarity for unsupervised person
re-identification, IEEE Trans. Pattern Anal. Mach. Intell. 46 (3) (2022)
1711–1726.
[4] H. Ji, L. Wang, S. Zhou, W. Tang, N. Zheng, G. Hua, Transfer easy to hard:
Adversarial contrastive feature learning for unsupervised person re-identification,
Pattern Recognit. 145 (2024) 109973.
[5] J. Li, S. Zhang, Joint visual and temporal consistency for unsupervised domain
adaptive person re-identification, in: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXIV 16,
Springer, 2020, pp. 483–499.
[6] Y. Zou, X. Yang, Z. Yu, B.V. Kumar, J. Kautz, Joint disentangling and adaptation
for cross-domain person re-identification, in: Computer Vision–ECCV 2020: 16th
European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II
16, Springer, 2020, pp. 87–104.
[7] X. Zhang, D. Li, Z. Wang, J. Wang, E. Ding, J.Q. Shi, Z. Zhang, J. Wang, Implicit
sample extension for unsupervised person re-identification, in: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp.
7369–7378.
[8] D. Cheng, J. Zhou, N. Wang, X. Gao, Hybrid dynamic contrast and probability
distillation for unsupervised person re-id, IEEE Trans. Image Process. 31 (2022)
3334–3346.
[9] X. Wang, Z. Sun, A. Chehri, G. Jeon, Y. Song, A novel attention-driven framework
for unsupervised pedestrian re-identification with clustering optimization, Pattern
Recognit. 146 (2024) 110045.

5. Conclusion
In this paper, we propose a Multi-Granularity Collaborative Constraint Feature Alignment Network. Specifically, the network first extracts and learns hierarchical multi-granular features, and then completes explicit feature alignment through multi-granularity constraint
loss to ensure the stable maintenance of semantic consistency throughout the learning process. Furthermore, the CCF module leverages constraint relationships in the feature space to filter local features during the memory bank initialization phase, thereby achieving accurate
alignment of multi-granular features within the memory bank. The
cross-granularity alignment concept of MCCAN is applicable to unsupervised person re-identification tasks, and can provide a reference
for other researchers—by focusing on the granularity alignment within
the same sample, the model performance can be further improved. Although MCCAN has achieved expected results in multi-granular feature
alignment tasks, the multi-granular feature extraction hierarchy of the
model still relies on manual presets, which cannot dynamically adjust
the extraction strategy according to the semantic complexity of input
samples, thereby leading to inconsistent alignment progress among
different samples and affecting the overall alignment efficiency. In
future research, we will conduct in-depth exploration on this limitation
and construct a dynamically adaptive multi-granular feature alignment mechanism to further improve the generalization performance of
MCCAN.
11


codex
MCCAN 那篇的核心不是“又做多粒度”，而是把多粒度之间的语义一致性单独抬成优化目标。为了拆清楚这一点，我再补读它方法页中间被截断的部分。
hook: PreToolUse
hook: PreToolUse
hook: PreToolUse Completed
hook: PreToolUse Completed
exec
/bin/zsh -lc "pdftotext -f 3 -l 6 'Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification.pdf' -" in /Users/abdslm/Desktop/reid-paper/experiments/paper_materials/pdfs_by_title
 succeeded in 0ms:
Pattern Recognition 179 (2026) 113676

Y. Chen et al.

Fig. 1. Overview of the MCCAN framework. The upper half illustrates the main framework, which encompasses feature extraction, clustering, and the two loss
functions designed for this framework. The lower half presents the two core components: the multi-granularity memory bank and the CCF module.

representation of pedestrian details. In the field of cross-modal ReID,
CM2 GT [37] further enhanced the robust representation of pedestrian
details via a global-patch-pixel hierarchical fine-grained feature design,
and verified the effectiveness of such a design as well. Furthermore,
existing USL ReID methods mostly focus on feature fusion and the
complementary learning of multi-granularity information but generally
lack explicit alignment mechanisms. This limitation prevents the model
from maintaining stable semantic consistency across features of varying
granularities. To address these issues, MCCAN effectively resolves the
aforementioned problems by introducing a constraint loss and the CCF
module.

To be specific, the 𝐿 − 𝑡ℎ layer of the model is duplicated to construct
a dual-branch output architecture, where each branch extracts finegrained features at distinct hierarchical levels, thereby enhancing the
representational capacity of fine-grained features.
Fig. 1 illustrates our framework and its individual modules. The MCCAN outputs two types of fine-grained features at different hierarchical
levels, which are generated by Branch 1 and Branch 2, respectively.
In particular, Branch 1 calculates the dot product between each patch
token and the class token, and ranks them to obtain a Rank-list. We
select the top K ranks to form the Top-K list 𝐾 :
(
)
𝑑𝑜𝑡𝑖 = 𝑥𝑖 ⊤ 𝑐𝑙𝑠1 ,  = rank {𝑑𝑜𝑡𝑖 }𝑛𝑖=1 ,  = 𝜉(),
(1)

3. Methodology

𝐾 = {𝑑𝑜𝑡𝑖 ∈  ∣ 𝑖 ≤ 𝐾}, 𝐾 = {𝑥𝑖 ∈  ∣ 𝑖 ≤ 𝐾},

3.1. Multi-granularity collaborative constraint feature alignment network
framework

(2)

where 𝑥𝑖 is the 𝑖th patch token output from Branch 1, with a total of
𝑛 patch tokens. 𝑐𝑙𝑠1 is the class token of Branch 1, 𝑟𝑎𝑛𝑘(⋅) represents
the sorting operation in descending order, and 𝜉(⋅) is the mapping from
the dot product list to the feature list. 𝑑𝑜𝑡𝑖 denotes the dot product
result. We apply a softmax function to the features in the obtained TopK list and compute a weighted sum to encourage the model’s contextual
awareness of local details. The final fine-grained features based on
patch tokens are obtained as:

The unlabeled pedestrian dataset is denoted as 𝐷 = {𝑥𝑖 ∣ 𝑖 =
1, 2, … , 𝑁}, where 𝑥𝑖 represents the 𝑖th image and 𝑁 denotes the total
number of samples in the dataset. The goal of USL ReID is to train a
deep neural network  = 𝑓 (𝜃; 𝑥) that maximizes the feature distance
between different individuals and minimizes the feature discrepancy
between the same individual.  = 𝑓 (𝜃; 𝑥) maps an image to a specific
embedding feature 𝒇 ∈ R𝑑 . To mine more refined discriminative
features, existing methods such as PPLR [21] and GL-NCPLR [22]
typically perform horizontal partitioning on feature maps prior to the
network output layer, so as to construct part-based fine-grained feature
representations. However, such a single partitioning strategy still suffers from the limitation of insufficient representation granularity, and
is prone to induce the model to overfit to the fixed perspective of part
division. In contrast, to achieve multi-granularity feature extraction
across hierarchical levels ranging from global to local, we employ a
Transformer network with shared weights across the first 𝐿 − 1 layers.

𝑓patch =

∑
𝑥𝑖 ∈𝐾

∑

exp(𝑑𝑜𝑡𝑖 )
𝑑𝑜𝑡𝑗 ∈𝐾 exp(𝑑𝑜𝑡𝑗 )

𝑥𝑖 .

(3)

Inspired by MGN [18], we perform a reshape operation on the 𝑛
D-dimensional patch tokens from Branch 2, reconstructing them into a
feature map 𝑓2 with the size of 𝐻
×𝑊
×D, where 𝑃 is the patch size. We
𝑃
𝑃
then uniformly divide the reconstructed feature map horizontally into
𝑀 parts and apply GeM pooling to obtain the part-based fine-grained
features:
(
(
))
𝑓𝑝𝑎𝑟𝑡 = 𝐶𝑜𝑛𝑐𝑎𝑡(𝐺𝑒𝑀 𝑆𝑝𝑙𝑖𝑡 𝑓2 , 𝑀 ),
(4)
3

Pattern Recognition 179 (2026) 113676

Y. Chen et al.

where 𝐶𝑜𝑛𝑐𝑎𝑡(⋅) denotes concatenation along the batch dimension and
𝑆𝑝𝑙𝑖𝑡(⋅) represents the operation of division.
Additionally, for the class tokens obtained from the two branches,
we perform an averaging operation to obtain the global feature:

3.2. Multi-granularity constraint loss and constrained clustering
In the previous subsection, the loss function 𝑐𝑙𝑏 has fully exploited
global information as well as various types of complementary detail
information. However, as elaborated in detail in the introduction section, imposing explicit semantic alignment on fine-grained features is
of crucial significance for unsupervised learning tasks. Therefore, to
directly constrain the semantic expressions of fine-grained features and
effectively achieve semantic alignment between local details and global
features, we reshape the features in  (Eq. (1)) based on the 𝑀-split
partition structure derived from Branch 2. This yields a set of feature
maps 𝑓1𝑗 (𝑗 = 1, 2, … , 𝑀), where each patch token is denoted as 𝑥𝑗𝑖 . Each
feature map has a size of 𝑃𝐻𝑀 × 𝑊
× D. Meanwhile, we slice 𝑓𝑝𝑎𝑟𝑡 into
𝑃
𝑗
𝑗
𝑀 segments to obtain 𝑓𝑝𝑎𝑟𝑡
(𝑗 = 1, 2, … , 𝑀). For each 𝑓𝑝𝑎𝑟𝑡
, we compute
𝑗
𝑗
the dot product 𝑑𝑜𝑡𝑖 with the corresponding 𝑥𝑖 , and obtain a re-ranked
Rank-list 𝑋 𝑗 based on the computed dot products:

1
1
𝑐𝑙𝑠 + 𝑐𝑙𝑠 ,
(5)
2 1 2 2
where 𝑐𝑙𝑠1 and 𝑐𝑙𝑠2 denote the class tokens from Branch 1 and Branch
2, respectively. At this point, the MCCAN completes the extraction
of three groups of progressive features 𝑓𝑔𝑙𝑜𝑏𝑎𝑙 , 𝑓𝑝𝑎𝑟𝑡 , and 𝑓𝑝𝑎𝑡𝑐ℎ that
range from coarse to fine. In general, after feature extraction, it is
necessary to perform clustering on the obtained features to generate
pseudo-labels. Given that this study adopts a special processing strategy
in the clustering stage, the relevant details will be elaborated in the
subsequent subsection.
Inspired by our baseline RTMem [11], we simultaneously introduce
an instance-level memory bank and a cluster-level memory bank, which
are dedicated to the calculation of loss function 𝐿I and loss function 𝐿C ,
respectively. Specifically, the computation of loss function 𝐿I relies on
the instance-level memory bank that stores the full-set sample features.
In contrast to the centroid-based cluster-level memory bank such as
CC [10], the calculation of loss function 𝐿C does not adopt the intracluster sample mean as the cluster proxy. Instead, it randomly selects
a single instance sample as the proxy sample for the corresponding
cluster. Such a cluster-level proxy is more consistent with the streaming distribution of DBSCAN clustering. The specific loss calculation
formulas are given as follows:
∑
⊤
𝑠∈𝑆 exp(𝒇 𝑖 ⋅ 𝒎𝑠 ∕𝜏)
I = − log ∑𝑁
,
(6)
⊤
𝑗=1 exp(𝒇 𝑖 ⋅ 𝒎𝑗 ∕𝜏)
𝑓𝑔𝑙𝑜𝑏𝑎𝑙 =

exp(𝒇 ⊤
𝑖 ⋅ C [𝑖]∕𝜏)
C = − log ∑𝐶
,
⊤
𝑗=1 exp(𝒇 𝑖 ⋅ C [𝑗]∕𝜏)

𝑗
𝑗 𝑛
𝑗
̃𝑗
̃𝑗
𝑑𝑜𝑡𝑗𝑖 = 𝑥𝑗⊤
𝑖 𝑓𝑝𝑎𝑟𝑡 , 𝐷 = 𝑟𝑎𝑛𝑘({𝑑𝑜𝑡𝑖 }𝑖=1 ), 𝑋 = 𝜉(𝐷 ),

where 𝑟𝑎𝑛𝑘(⋅) represents the sorting operation in descending order, and
𝜉(⋅) is the mapping from the dot product list to the feature list. The reranked list 𝑋 𝑗 essentially represents the semantic similarity between
the patch tokens of Branch 1 and the part-based fine-grained features of
Branch 2; the higher the ranking, the greater the semantic consistency
between the two types of features. This operation actually establishes
associations between fine-grained features at different levels. To further
bridge the connection between such fine-grained features and global
features, this study needs to introduce the global-feature-based ranked
list  in Eq. (1) for assistance. For all features in 𝑋 𝑗 , we select the topranked feature 𝑝𝑗𝑐1 and the highest-ranked feature in the  list, denoted
as 𝑝𝑗𝑐2 , to form our set of constraint positive features:

(7)

𝑃 𝑗 = {𝑝𝑗𝑐1 = 𝑇 𝑜𝑝(𝑋 𝑗 ), 𝑝𝑗𝑐2 = arg min index (𝑥𝑗𝑖 )},

where 𝐼 and 𝐶 denote the instance-level and cluster-level memory
banks, respectively, and 𝑠 ∈  , where  is the set composed
of all positive samples. 𝑚𝑠 = 𝐼 [𝑠], and 𝑓𝑖 represents the current
instance feature from the mini-batch. 𝜏 is the temperature coefficient
that controls the concentration of distribution, while 𝐶 and 𝑁 represent
the number of clusters and the number of instances, respectively. As the
training process advances and the network parameters are iteratively
optimized, the historical features cached in the memory bank will
gradually exhibit a distribution discrepancy with the new features
output by the current model. To tackle this problem, we update the
instance-level memory bank with the current instance feature 𝑓𝑖 , while
for the cluster-level memory bank, we randomly sample the feature 𝑓̃𝑖
of a random sample within the corresponding cluster from the current
mini-batch for its update [11]. The dynamic updates of the memory
banks are as follows:
𝐼 [𝑖] ← 𝑓𝑖 , 𝐶 [𝑖] ← 𝑓̃𝑖 .

𝑗

where 𝑇 𝑜𝑝(⋅) denotes the operation of selecting the highest-ranked
feature in a list. Essentially, 𝑝𝑗𝑐1 and 𝑝𝑗𝑐2 correspond to the patch tokens
with the highest semantic similarity to the part-based fine-grained
feature and the global feature, respectively. While the patch-level finegrained feature 𝑓patch is directly generated by the weighted fusion
of patch tokens. With the aid of these two types of features, crossgranularity semantic alignment between features of different granularities can be effectively constrained. Taking the part-level feature
𝑗
𝑓𝑝𝑎𝑟𝑡
as a query, we regard the constraint positive features as positive sample features and impose alignment constraints to maintain
semantic consistency among features at different granularity levels.
Fig. 2 clearly illustrates the overall computation pipeline of the multigranularity constraint loss. First, the patch tokens in Branch 1 are
guided for re-ranking by the part-based fine-grained features from
Branch 2 (Eq. (10)), and the top-ranked feature is denoted as 𝑝𝑗𝑐1 .
Then, under the further guidance of the list  obtained from global
feature-based ranking (Eq. (1)), the top-ranked feature is indexed and
denoted as 𝑝𝑗𝑐2 . 𝑝𝑗𝑐1 and 𝑝𝑗𝑐2 jointly serve as positive anchor patch
𝑗
features, which pull the local feature 𝑓𝑝𝑎𝑟𝑡
toward them, thus enforcing
semantic consistency among global, part-level, and patch-level features.
The definition of the proposed multi-granularity constraint loss is given
as follows:
∑
𝑗⊤
𝑗
𝑀
𝑝∈𝑃 𝑗 exp (𝑓𝑝𝑎𝑟𝑡 ⋅ 𝑥𝑝 ∕𝜏)
1 ∑
𝑐𝑠𝑡 =
−𝑙𝑜𝑔 ∑𝑛
.
(12)
𝑗⊤
𝑗
𝑀 𝑗=1
𝑖=1 exp (𝑓𝑝𝑎𝑟𝑡 ⋅ 𝑥𝑖 ∕𝜏)

(8)

global features inherently suffer from viewpoint bias, which causes the
model to tend to capture the most salient cues while ignoring crucial
local details. Different from RTMem, our method extends this loss to
the constructed multi-granularity features 𝑓𝑔𝑙𝑜𝑏𝑎𝑙 , 𝑓𝑝𝑎𝑟𝑡 and 𝑓𝑝𝑎𝑡𝑐ℎ . This
implies that the memory bank is also expanded from the original two
versions targeting only global features to 3 × 2 types of memory banks
based on different granularities, thereby achieving sufficient learning
of pedestrian detailed information. This design not only retains the
advantages of the instance-level and cluster-level losses in the finegrained representation of pedestrian identities, but also enhances the
capability of global features and multi-level fine-grained features in
learning the internal details of individual samples. The definition of the
multi-granularity collaborative loss proposed in this paper, where 𝜆1
is the hyperparameter that controls the contribution weights between
global and local features, is given as follows:
C
C
+ C
𝑔𝑙𝑜𝑏𝑎𝑙 + 𝜆1 (𝑝𝑎𝑡𝑐ℎ + 𝑝𝑎𝑟𝑡 ).

(11)

𝑥𝑖 ∈𝑋 𝑗

In RTMem, I and C are only applied to global features. However,

𝑐𝑙𝑏 = I𝑔𝑙𝑜𝑏𝑎𝑙 + 𝜆1 (I𝑝𝑎𝑡𝑐ℎ + I𝑝𝑎𝑟𝑡 )

(10)

Different from the loss calculation logic of traditional USL ReID methods that relies on inter-sample relationships [8,12], the loss function
𝑐𝑠𝑡 proposed in this paper only focuses on imposing semantic alignment constraints on features of different granularities within the same
sample. This design can prompt the model to achieve precise alignment
between global and local features during the training process, thereby
ensuring that the learned fine-grained features possess a high degree

(9)
4

Pattern Recognition 179 (2026) 113676

Y. Chen et al.

Fig. 2. Computation pipeline of multi-granularity constraint loss.
Table 1
The datasets used for training and testing.

3.3. Center-constrained filtering module

Datasets

Cameras

Training samples

Total samples

Market-1501
MSMT17
PersonX
Veri776

6
15
6
20

12,936
32,621
9840
412

32,668
126,441
45,792
776

Prior to initializing the memory bank, we take the centroid of global
features as the reference point for all fine-grained features, thereby
implementing constrained filtering on the fine-grained features. The
core rationale for this operation is as follows: global features themselves
are an aggregated form of fine-grained features, while the centroid of
global features serves as the prototype representation of pedestrians
within each cluster. On this basis, we regard the centroid of global
features as the center of all fine-grained features in a sample, and the
fine-grained features should ideally be distributed around this center
in the feature space. After passing through two branches, each sample
will output 𝑀 + 1 types of fine-grained features, including 𝑀 features
of 𝑓𝑝𝑎𝑟𝑡 and one feature of 𝑓𝑝𝑎𝑡𝑐ℎ . For the 𝑖th cluster and 𝑗th feature
𝑞𝑖𝑗 in each type of fine-grained feature, we select the global feature
centroids of the same cluster and the nearest neighboring different
cluster, respectively, and define them as the fine-grained positive center
 + and the fine-grained negative center  − .

of semantic consistency. The total loss function adopted in this paper
is defined as the weighted sum of the multi-granularity collaborative
loss and the multi-granularity constraint loss, where 𝜆 is the coefficient
used to balance the contribution weights of these two loss components:
𝑚𝑐𝑐𝑎 = 𝑐𝑙𝑏 + 𝜆𝑐𝑠𝑡 .

(13)

To effectively incorporate local detail cues during the clustering
phase while minimizing noise interference to the greatest extent, we
embed the constraint positive features set into the clustering pipeline.
The rationality of this design lies in the fact that the constraint positive
features set inherently possesses natural fine-grained representation
properties and noise robustness derived from semantic consistency,
thus providing accurate and robust discriminative criteria for clustering. First, we perform fusion on the features within the constraint
positive features set to reduce computational complexity:
𝑝̃𝑐1 =

𝑀
𝑀
∑
∑
𝑝𝑗𝑐1 , 𝑝̃𝑐2 =
𝑝𝑗𝑐2 .
𝑗=1

 + = 𝑖 ,  − = arg min
d(𝑞𝑖𝑗 , 𝑚 ),
+
 ≠𝑚

(17)

where 𝑖 represents the centroid of the global features of the 𝑖th cluster,
and d(⋅) is the function used to compute the cosine distance between
two features. We calculate the distance of each fine-grained feature to
the positive center and to the negative center, respectively:

(14)

𝑎𝑖𝑗 = d(𝑞𝑖𝑗 ,  + ), 𝑏𝑖𝑗 = d(𝑞𝑖𝑗 ,  − ).

𝑗=1

Based on the fused features 𝑝̃𝑐1 , 𝑝̃𝑐2 , and the previously extracted global
feature 𝑓𝑔𝑙𝑜𝑏𝑎𝑙 , we construct the Jaccard distance matrices 𝑊𝑝𝑐1 , 𝑊𝑝𝑐2
and 𝑊𝑔𝑙𝑜𝑏𝑎𝑙 between samples, respectively. The Jaccard distance is
defined as follows:
|(𝑓𝑖 , 𝜅) ∩ (𝑓𝑗 , 𝜅)|
𝑑𝐽 (𝑓𝑖 , 𝑓𝑗 ) = 1 −
,
(15)
|(𝑓𝑖 , 𝜅) ∪ (𝑓𝑗 , 𝜅)|

(18)

The core function of 𝑎𝑖𝑗 and 𝑏𝑖𝑗 is to measure the proximity between individual fine-grained features and their corresponding identity
centers, as well as the separability between these features and the
identity centers of other categories. Notably, since their calculation
relies on the distance metric between global and local features, 𝑎𝑖𝑗 and
𝑏𝑖𝑗 can simultaneously characterize the semantic consistency between
fine-grained features and global features. We construct the constraint
factor 𝑅𝑖𝑗 based on the aforementioned distances, a smaller value of 𝑅𝑖𝑗
indicates that the corresponding fine-grained feature not only deviates
from the feature expression pattern of pedestrians within the sample,
but also deviates further from the global prototype of the identity,
implying that it carries more identity-irrelevant noise information. To
align the semantic spaces of multi-granularity features in the memory
bank and filter out such identity-irrelevant noise, we eliminate all finegrained features whose constraint factor is smaller than the threshold
𝜎, and finally obtain the fine-grained feature set 𝑄𝑖𝑗 for memory bank
construction:
|
|
|𝑎𝑖𝑗 − 𝑏𝑖𝑗 |
| ,
𝑅𝑖𝑗 = |
(19)
max (𝑎𝑖𝑗 , 𝑏𝑖𝑗 )

where (𝑓𝑖 , 𝜅) denotes the 𝜅-reciprocal nearest neighbors [38], defined
{
(
) (
)}
as: (𝑓 , 𝜅) = 𝑔𝑖 ∣ 𝑔𝑖 ∈  (𝑓 , 𝜅) ∧ 𝑓 ∈  (𝑔𝑖 , 𝜅) . Here,  (𝑓 , 𝜅)
represents the 𝜅-nearest neighbors of a probe 𝑓 , and | ⋅ | denotes the size
of the candidate set. Subsequently, we compute the Jaccard distance
𝑑𝐽 (𝑓𝑖 , 𝑓𝑗 ) between instances 𝑓𝑖 and 𝑓𝑗 by comparing their respective 𝜅reciprocal nearest neighbor sets. The core assumption of this calculation
is that: the more overlapping samples exist between the 𝜅-reciprocal
nearest neighbor sets of two instances, the higher the similarity between these two instances [38]. The final Jaccard distance matrix 𝑊𝑐𝑠𝑡
is obtained by the weighted fusion of matrices 𝑊𝑝𝑐1 , 𝑊𝑝𝑐2 , and 𝑊𝑔𝑙𝑜𝑏𝑎𝑙 :
1−𝜇
(𝑊𝑝𝑐1 + 𝑊𝑝𝑐2 ),
(16)
2
where 𝜇 is the parameter that controls the fusion ratio. Finally, we
perform DBSCAN clustering on the fused Jaccard distance matrix 𝑊𝑐𝑠𝑡 ,
thereby realizing pseudo-label assignment for the model.
𝑊𝑐𝑠𝑡 = 𝜇𝑊𝑔𝑙𝑜𝑏𝑎𝑙 +

𝑄𝑖𝑗 = {𝑞𝑖𝑗 |𝑅𝑖𝑗 ⟩𝜎}.
5

(20)

Pattern Recognition 179 (2026) 113676

Y. Chen et al.

Fig. 3. (a) T-SNE visualization of 20 randomly sampled identities from the test set. (b) Visualization of the intra-class and inter-class distance distribution based
on all test samples.

3.4. Discussion

4. Experiments

Discussion on the redundancy between 𝑐𝑙𝑏 and 𝑐𝑠𝑡 : 𝑐𝑙𝑏 aims to
learn discriminative identity features between samples, while 𝑐𝑠𝑡 is
designed to achieve semantic alignment of multi-granularity features
within a single sample. They form a pair of mutually orthogonal
loss functions, establishing a hierarchical and closed-loop collaborative
optimization mechanism in different optimization dimensions: 𝑐𝑠𝑡 performs semantic calibration of intra-sample features, providing a reliable
feature foundation with no semantic shift and stable identity information for 𝑐𝑙𝑏 . In turn, 𝑐𝑙𝑏 fully exploits the discriminative capability of
multi-granularity features in the high-quality calibrated feature space,
further improving the inter-class discrimination performance of the
model. The two losses are functionally complementary, non-interfering,
and mutually reinforcing, enabling the model to achieve collaborative
optimization in both cross-granularity semantic consistency and intersample identity discrimination. Therefore, no functional redundancy
exists between them.
Discussion on the Robustness of 𝑊𝑐𝑠𝑡 across datasets: The 𝑊𝑐𝑠𝑡
fusion mechanism adopts a weighted fusion strategy of global feature anchoring plus fine-grained feature supplementation. Specifically,
global features serve as the prototype representation of pedestrian
identities and provide a stable core anchor for clustering. Fine-grained
features are derived from the constraint positive feature set 𝑃 𝑗 , which is
dynamically updated during training and real-timely selects the patch
tokens with the highest matching degree to global semantics and partlevel semantics. This enables flexible adaptation to the distribution
characteristics of different datasets, thus endowing 𝑊𝑐𝑠𝑡 with favorable
robustness across various data distributions. Experiments demonstrate
that a fixed fusion ratio already yields excellent performance. Nevertheless, leveraging the dynamic nature of 𝑊𝑐𝑠𝑡 , a dynamically adjusted
fusion ratio may further enhance its adaptability to data distributions, and this direction warrants in-depth verification in subsequent
research.

4.1. Datasets and evaluation protocol
Datasets: We evaluated the proposed method on four widely-used
ReID datasets: Market-1501 [24], MSMT17 [25], PersonX [26], and
Veri-776 [27]. Their detailed statistics are summarized in Table 1.
Evaluation Metrics: The performance of the proposed method is
evaluated using the mean Average Precision (mAP) and the Cumulative
Matching Characteristic (CMC) curve. The CMC accuracy is reported
at Rank-1 (R1), Rank-5 (R5), and Rank-10 (R10). To ensure a fair
evaluation, no post-processing techniques are employed throughout the
experiments. Furthermore, we exclusively utilize global features for
person retrieval. On the one hand, since prevailing methods typically
rely solely on global features during testing, this setup ensures a fair
comparison. On the other hand, the extraction and fusion of multigranularity features significantly degrades inference speed, thereby
hindering the practical deployment efficiency of the model.
4.2. Implementations details and parameter setting
The Transformer backbone adopted in this paper is derived from
that of Transreid-SSL [49], which utilizes the ViT-Small/16 architecture [50]. This architecture consists of 12 Transformer layers, with 6
attention heads per layer and a feature dimension of 384. The backbone
is pre-trained on LUPerson, a large-scale unlabeled dataset [51]. Input
images are resized to 384 × 128, with a patch size of 16 × 16, resulting in 192 patches in total. We employ standard data augmentation
strategies including random horizontal flipping, random cropping, and
random erasing [52]. We adopt the SGD optimizer with a weight decay
of 5 × 10−4 and an initial learning rate of 3.5 × 10−4 , where the learning
rate is decayed by a factor of 0.1 every 20 epochs. For the DBSCAN
clustering algorithm, the maximum neighborhood distance (i.e., the
𝜖 parameter) used to identify two samples as neighboring ones is set
to 0.5, 0.7, 0.4 and 0.5 for the Market-1501, MSMT17, PersonX and
6


codex
我按每篇前 8 页方法正文读了，并补读了后半部分的关键消融和可视化页。下面拆的是“创新怎么被构造出来”，不是模块复述。

**1. Multi Queue for Unsupervised Person Re-identification**

1. 触发观察：作者盯上的不是聚类本身，而是无监督 ReID 里 memory bank 和 momentum update 的细节。SpCL、Cluster Contrast 这类方法默认用动量方式更新历史特征，但这会把一个伪身份压成较固定的历史表示，负样本矩阵也缺少足够的类内变化。另一个观察是固定温度系数无法随训练阶段和数据难度调整对难样本的关注。

2. 重定义动作：他们把“memory bank 怎么更新”从实现细节抬成了核心问题。旧说法是无监督 ReID 难在伪标签噪声和聚类质量；他们的新说法是，伪标签生成后，历史特征如何被保存、采样、淘汰，会直接决定对比学习看到的负样本分布。关键词是 multi-queue、historical features、resampling、trainable temperature、hard samples。

3. 机制怎么长出来：机制基本顺着重定义出来。既然问题是单一动量特征不足以表示一个伪身份，那就给每个伪身份维护一个队列，按入队出队保留近期多个历史特征，再随机采样构造负样本矩阵。既然不同数据集和训练阶段难度不同，那就把 InfoNCE 的温度变成可训练参数，让模型自己调节难样本敏感度。

4. 证据闭环：关键证据是队列长度消融和可训练温度消融。队列长度实验说明保留一部分旧特征有利于表示能力，但太长会引入过旧噪声。温度实验显示可训练温度在多个数据集上有效，且温度随训练下降，MSMT17 这类更难数据下降更快，作者用它证明模型确实逐步转向难样本。

5. 审稿人为什么买账：这篇卖的主要是机制，不是很强的新视角。它的真实新意在于把 memory bank 的“更新策略”讲成了负样本分布管理问题。对我们有用的地方是：一个小机制也能发，但前提是你要把它和训练信号分布、难样本日程、过拟合风险绑定起来，而不是只说“我换了一个队列”。

**2. Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification**

1. 触发观察：已有换衣 ReID 往往回避衣服信息，或者用非衣服描述如体型、姿态、轮廓来学习不变特征。但这些非衣服描述经常不精确，CLIP 文本和细粒度图像之间又有语义差距。作者反过来观察到，衣服描述虽然不该成为最终身份特征，但它可以告诉模型“哪些东西应该被压掉”。

2. 重定义动作：旧问题是“如何避免模型依赖衣服”；新问题是“如何准确定位衣服语义并把它从视觉特征中减掉”。这一步很关键，因为作者不是说衣服文本有用来识别人，而是说衣服文本有用来抑制衣服。关键词是 clothing descriptions、coarse-to-fine、semantic-guided、clothes-agnostic、parsing image。

3. 机制怎么长出来：机制和重定义绑定较紧。先用 human parsing 抠出衣服区域，让多个 LVLM 只描述衣服，并用粗粒度和细粒度提示生成多种衣服 caption。再用 CLIP 相似度过滤，保留最可信的两个描述。SGE 分支让图像特征和衣服文本做注意力，然后从视觉特征里减去衣服相关成分。PIE 分支直接遮掉衣服区域训练，让模型被迫使用更稳定的身体、姿态、局部线索。

4. 证据闭环：关键证据不是单纯 SOTA，而是 caption 数量和模块消融。单 caption 有提升，多模型多粒度 caption 更好，但 caption 太多反而引入冲突语义，说明作者讲的“多样性和噪声平衡”成立。特征图显示 baseline 更看衣服和背景，方法更看脸部、姿态、配饰等稳定区域。局部特征优于全局特征也支撑了细粒度抑制衣服的叙事。

5. 审稿人为什么买账：这篇卖的是视角多于模块。真正的新意是把“衣服语义”从要丢弃的干扰项，改造成可用的负向指导信号。这个构造很值得学：不是增加一个 VLM，而是改变 VLM 语义在任务里的角色。

**3. Multi-feature balanced network for clothes-changing person re-identification**

1. 触发观察：作者注意到换衣 ReID 里单一线索都不可靠。RGB 外观会被衣服误导，mask 或 silhouette 丢掉颜色、脸部和配饰等有用线索，身体局部裁剪又会受背景和局部噪声影响。简单拼接多个特征也不够，因为不同特征权重固定，空间语义也未对齐。

2. 重定义动作：旧说法是“找到一种衣服无关特征”；新说法是“不同衣服无关线索各有缺陷，关键是平衡和协同”。关键词是 multi-feature balanced、clothing-unrelated branch、mask branch、knowledge transfer、background constraint、cross fusion。

3. 机制怎么长出来：机制是三分支。global branch 保留完整 RGB 外观，但用 clothing-unrelated branch 的知识通过 KTM 引导它看衣服无关区域。clothing-unrelated branch 按人体比例裁剪头肩、腿部等受衣服影响较小的区域，用 FAM 分出身份前景和背景，并用背景约束让身份特征靠近全局特征、远离背景特征。mask branch 使用 silhouette 学身体形状，再用双向 cross-attention 把 mask 特征和衣服无关局部特征融合。

4. 证据闭环：作者用分支消融证明 global、clothing-unrelated、mask 都有增益，完整三分支最好。还比较了直接拼接、单向注意力和双向 CFM，证明“融合方式”确实比简单堆特征重要。检索可视化展示换衣场景下返回结果更稳定。不过它对“balanced”的证明还不够强，缺少动态权重或各线索可靠性变化的直接证据。

5. 审稿人为什么买账：这篇更偏机制工程。买账点是它把换衣下的多种稳定线索系统化，且每个分支都有消融支撑。对我们来说，它也暴露一个风险：如果主张叫“平衡”，就最好真的证明平衡过程，而不是只证明多个分支相加有效。

**4. Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification**

1. 触发观察：作者看到现有多模态换衣 ReID 没把属性用充分。AD-ViT、MADE 一类用属性标签，但多是一维 one-hot；CCAF、MIPL 一类用文本，但缺少结构化属性信息；很多方法还依赖人工衣服标签。另一个观察是，像素级遮衣服能做低层解耦，但缺少高层语义指导。

2. 重定义动作：旧问题是“属性能不能作为辅助标签”；新问题是“属性文本能不能作为多粒度提示，明确规定哪些语义该保留、哪些该压制”。关键词是 multi-granularity attribute prompt、attribute-level、caption-level、identity-related、clothing-related、pseudo label。

3. 机制怎么长出来：机制很顺。先用 SCHP 生成衣服擦除图，再用 AAPAR 识别 35 类属性。年龄、性别被当作身份相关属性，头饰、上衣、下装、鞋、配饰等被当作衣服相关属性。然后用模板把属性变成文本，在 raw image branch 和 masked image branch 上用 CLIP ViT 加 visual deep prompt 学图像表示。属性解耦部分把图像特征和身份相关文本拼接，用 triplet 促进保留；对衣服相关文本用负向损失或对抗约束进行压制。BNNeck 用来隔开图像调优和文本对齐，避免一个目标破坏另一个目标。

4. 证据闭环：消融很对题。baseline 加 clothing adversarial attribute 后提升，说明压制衣服属性有效；加 identity-relevant attribute 后继续提升，说明保留稳定属性有效；再加 caption-level 约束提升，说明多粒度文本不是摆设；masked branch 的属性解耦去掉后也下降，说明擦衣服后仍有残留衣服语义。检索可视化显示 baseline 受颜色和款式影响更重，MAPL 更能跨衣服找人。

5. 审稿人为什么买账：这篇卖的是视角。它不是“又用了属性”，而是把属性拆成正向身份语义和负向衣服语义，再用文本提示做多粒度解耦。真正新意来自“属性从标签变成 prompt 化的解耦约束”，同时减少人工衣服标签依赖。

**5. Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification**

1. 触发观察：无监督 ReID 已经有很多 global-local、多粒度和细粒度方法，但大多默认模型会自己学到不同粒度之间的语义一致性。作者指出，在伪标签噪声和局部特征噪声共存时，模型可能学到的是虚假的互补信息，导致 global、part、patch 之间语义错位。

2. 重定义动作：旧问题是“局部细节不够”；新问题是“多粒度特征之间缺少显式语义对齐”。这一步很强，因为它把多粒度从“多加几个特征”提升为“不同粒度之间需要保持同一个身份语义”。关键词是 explicit alignment、semantic consistency、multi-granularity collaborative constraint、center-constrained filtering、within-sample alignment。

3. 机制怎么长出来：机制围绕对齐展开。Transformer 输出 global、part、patch 三种粒度。多粒度 collaborative loss 把 RTMem 式 instance memory 和 cluster memory 扩展到三种粒度，让每种粒度都参与判别学习。constraint loss 不看样本间关系，而是在同一样本内部，用 part 特征和 global 排名选出语义一致的 patch token，把 part、patch、global 拉到同一个语义方向。聚类阶段也用语义一致的细粒度特征和 global 特征融合成距离矩阵。CCF 模块在 memory bank 初始化前，用 global cluster centroid 过滤偏离身份中心的局部特征。

4. 证据闭环：这篇证据闭环比较完整。消融从只用 baseline，到加多粒度判别损失、加约束损失、加约束聚类、加 CCF，逐步证明每一步都服务于对齐。CCF 过滤比例随训练下降，说明早期局部噪声多、后期特征变稳定；MSMT17 上过滤更强，也符合复杂数据更噪的说法。t-SNE、类内类间距离、rank-list、attention heatmap 都围绕“语义更一致、细节更准确”展开。

5. 审稿人为什么买账：这篇卖的是视角，而且机制绑定很紧。它不是泛泛说多粒度有效，而是说多粒度在无监督下会错位，所以要在损失、聚类、memory bank 三个阶段都做显式对齐。这个构造对我们最值得学：一个好问题重定义，会自然长出一组同向机制。

**6. Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification**

1. 触发观察：视频 ReID 里，很多方法只建模固定长度的全局时间线索。但人的不同身体部位运动和遮挡规律不同，腿部变化大，躯干更稳定，某些部位会在部分帧被遮挡后又出现。已有水平切分图方法虽然能做局部建模，但切分固定、邻居数量固定，容易连接低相关节点。

2. 重定义动作：旧问题是“怎么建模视频时间特征”；新问题是“怎么按身体部位自适应选择不同长度的时间线索”。关键词是 dynamic hierarchical graph、adaptive multi-granularity temporal clues、part-consistent graph、mutual nearest neighbor、similarity weighted inference。

3. 机制怎么长出来：DHGN 先把每帧特征图水平细分，再按特征相似度聚成自适应语义区域，而不是固定条带。每个语义层级跨帧建图，同一身体部位的节点才互连。边用 mutual nearest neighbor 建，避免低相关节点干扰。GC layer 用相似度加权聚合邻居信息，再用注意力在原节点和邻居信息之间重构残差特征。末端 GFC layer 建全局帧图，用 temporal graph pooling 排除干扰帧。SWIM 再用 gallery-gallery 相似度修正 query-gallery 相似度。

4. 证据闭环：消融非常贴合问题定义。GC、GC+attention、GFC、DHGN、SWIM 分别验证局部动态时间建模、注意力重构、全局时间建模和图库关系推理。粒度数、水平划分数、插入层位置、邻居 K1 和 K2 都直接对应“动态多粒度图是否必要”。节点相似度可视化最关键，它显示遮挡时模型会找长程相似帧，无遮挡时更偏短程运动线索。activation map 和 t-SNE 则证明特征更集中、更可分。

5. 审稿人为什么买账：这篇同时卖视角和机制。核心新意是把视频时间建模从固定全局序列，改成部位级、动态邻居、可变时间范围的图建模。SWIM 更像附加推理增强，真正支撑论文的是 DHGN 对“不同部位需要不同时间线索”的机制化落实。

最后看这批论文，共同套路很明确：作者先不急着造模块，而是把一个已有系统里的次级矛盾抬成主问题，比如 memory bank 更新、多粒度错位、衣服语义怎么用、视频时间线索固定。然后机制必须在多个位置反复服务同一句重定义，最后用消融和可视化证明“这个重定义本身是对的”，而不是只证明最终涨点。
tokens used
145,117
我按每篇前 8 页方法正文读了，并补读了后半部分的关键消融和可视化页。下面拆的是“创新怎么被构造出来”，不是模块复述。

**1. Multi Queue for Unsupervised Person Re-identification**

1. 触发观察：作者盯上的不是聚类本身，而是无监督 ReID 里 memory bank 和 momentum update 的细节。SpCL、Cluster Contrast 这类方法默认用动量方式更新历史特征，但这会把一个伪身份压成较固定的历史表示，负样本矩阵也缺少足够的类内变化。另一个观察是固定温度系数无法随训练阶段和数据难度调整对难样本的关注。

2. 重定义动作：他们把“memory bank 怎么更新”从实现细节抬成了核心问题。旧说法是无监督 ReID 难在伪标签噪声和聚类质量；他们的新说法是，伪标签生成后，历史特征如何被保存、采样、淘汰，会直接决定对比学习看到的负样本分布。关键词是 multi-queue、historical features、resampling、trainable temperature、hard samples。

3. 机制怎么长出来：机制基本顺着重定义出来。既然问题是单一动量特征不足以表示一个伪身份，那就给每个伪身份维护一个队列，按入队出队保留近期多个历史特征，再随机采样构造负样本矩阵。既然不同数据集和训练阶段难度不同，那就把 InfoNCE 的温度变成可训练参数，让模型自己调节难样本敏感度。

4. 证据闭环：关键证据是队列长度消融和可训练温度消融。队列长度实验说明保留一部分旧特征有利于表示能力，但太长会引入过旧噪声。温度实验显示可训练温度在多个数据集上有效，且温度随训练下降，MSMT17 这类更难数据下降更快，作者用它证明模型确实逐步转向难样本。

5. 审稿人为什么买账：这篇卖的主要是机制，不是很强的新视角。它的真实新意在于把 memory bank 的“更新策略”讲成了负样本分布管理问题。对我们有用的地方是：一个小机制也能发，但前提是你要把它和训练信号分布、难样本日程、过拟合风险绑定起来，而不是只说“我换了一个队列”。

**2. Multi-Branch Clothes-Agnostic Feature Learning for Cloth-Changing Person Re-Identification**

1. 触发观察：已有换衣 ReID 往往回避衣服信息，或者用非衣服描述如体型、姿态、轮廓来学习不变特征。但这些非衣服描述经常不精确，CLIP 文本和细粒度图像之间又有语义差距。作者反过来观察到，衣服描述虽然不该成为最终身份特征，但它可以告诉模型“哪些东西应该被压掉”。

2. 重定义动作：旧问题是“如何避免模型依赖衣服”；新问题是“如何准确定位衣服语义并把它从视觉特征中减掉”。这一步很关键，因为作者不是说衣服文本有用来识别人，而是说衣服文本有用来抑制衣服。关键词是 clothing descriptions、coarse-to-fine、semantic-guided、clothes-agnostic、parsing image。

3. 机制怎么长出来：机制和重定义绑定较紧。先用 human parsing 抠出衣服区域，让多个 LVLM 只描述衣服，并用粗粒度和细粒度提示生成多种衣服 caption。再用 CLIP 相似度过滤，保留最可信的两个描述。SGE 分支让图像特征和衣服文本做注意力，然后从视觉特征里减去衣服相关成分。PIE 分支直接遮掉衣服区域训练，让模型被迫使用更稳定的身体、姿态、局部线索。

4. 证据闭环：关键证据不是单纯 SOTA，而是 caption 数量和模块消融。单 caption 有提升，多模型多粒度 caption 更好，但 caption 太多反而引入冲突语义，说明作者讲的“多样性和噪声平衡”成立。特征图显示 baseline 更看衣服和背景，方法更看脸部、姿态、配饰等稳定区域。局部特征优于全局特征也支撑了细粒度抑制衣服的叙事。

5. 审稿人为什么买账：这篇卖的是视角多于模块。真正的新意是把“衣服语义”从要丢弃的干扰项，改造成可用的负向指导信号。这个构造很值得学：不是增加一个 VLM，而是改变 VLM 语义在任务里的角色。

**3. Multi-feature balanced network for clothes-changing person re-identification**

1. 触发观察：作者注意到换衣 ReID 里单一线索都不可靠。RGB 外观会被衣服误导，mask 或 silhouette 丢掉颜色、脸部和配饰等有用线索，身体局部裁剪又会受背景和局部噪声影响。简单拼接多个特征也不够，因为不同特征权重固定，空间语义也未对齐。

2. 重定义动作：旧说法是“找到一种衣服无关特征”；新说法是“不同衣服无关线索各有缺陷，关键是平衡和协同”。关键词是 multi-feature balanced、clothing-unrelated branch、mask branch、knowledge transfer、background constraint、cross fusion。

3. 机制怎么长出来：机制是三分支。global branch 保留完整 RGB 外观，但用 clothing-unrelated branch 的知识通过 KTM 引导它看衣服无关区域。clothing-unrelated branch 按人体比例裁剪头肩、腿部等受衣服影响较小的区域，用 FAM 分出身份前景和背景，并用背景约束让身份特征靠近全局特征、远离背景特征。mask branch 使用 silhouette 学身体形状，再用双向 cross-attention 把 mask 特征和衣服无关局部特征融合。

4. 证据闭环：作者用分支消融证明 global、clothing-unrelated、mask 都有增益，完整三分支最好。还比较了直接拼接、单向注意力和双向 CFM，证明“融合方式”确实比简单堆特征重要。检索可视化展示换衣场景下返回结果更稳定。不过它对“balanced”的证明还不够强，缺少动态权重或各线索可靠性变化的直接证据。

5. 审稿人为什么买账：这篇更偏机制工程。买账点是它把换衣下的多种稳定线索系统化，且每个分支都有消融支撑。对我们来说，它也暴露一个风险：如果主张叫“平衡”，就最好真的证明平衡过程，而不是只证明多个分支相加有效。

**4. Multi-Granularity Attribute Prompt Learning for Cloth-Changing Person Re-Identification**

1. 触发观察：作者看到现有多模态换衣 ReID 没把属性用充分。AD-ViT、MADE 一类用属性标签，但多是一维 one-hot；CCAF、MIPL 一类用文本，但缺少结构化属性信息；很多方法还依赖人工衣服标签。另一个观察是，像素级遮衣服能做低层解耦，但缺少高层语义指导。

2. 重定义动作：旧问题是“属性能不能作为辅助标签”；新问题是“属性文本能不能作为多粒度提示，明确规定哪些语义该保留、哪些该压制”。关键词是 multi-granularity attribute prompt、attribute-level、caption-level、identity-related、clothing-related、pseudo label。

3. 机制怎么长出来：机制很顺。先用 SCHP 生成衣服擦除图，再用 AAPAR 识别 35 类属性。年龄、性别被当作身份相关属性，头饰、上衣、下装、鞋、配饰等被当作衣服相关属性。然后用模板把属性变成文本，在 raw image branch 和 masked image branch 上用 CLIP ViT 加 visual deep prompt 学图像表示。属性解耦部分把图像特征和身份相关文本拼接，用 triplet 促进保留；对衣服相关文本用负向损失或对抗约束进行压制。BNNeck 用来隔开图像调优和文本对齐，避免一个目标破坏另一个目标。

4. 证据闭环：消融很对题。baseline 加 clothing adversarial attribute 后提升，说明压制衣服属性有效；加 identity-relevant attribute 后继续提升，说明保留稳定属性有效；再加 caption-level 约束提升，说明多粒度文本不是摆设；masked branch 的属性解耦去掉后也下降，说明擦衣服后仍有残留衣服语义。检索可视化显示 baseline 受颜色和款式影响更重，MAPL 更能跨衣服找人。

5. 审稿人为什么买账：这篇卖的是视角。它不是“又用了属性”，而是把属性拆成正向身份语义和负向衣服语义，再用文本提示做多粒度解耦。真正新意来自“属性从标签变成 prompt 化的解耦约束”，同时减少人工衣服标签依赖。

**5. Multi-granularity collaborative constraint feature alignment network for unsupervised person re-identification**

1. 触发观察：无监督 ReID 已经有很多 global-local、多粒度和细粒度方法，但大多默认模型会自己学到不同粒度之间的语义一致性。作者指出，在伪标签噪声和局部特征噪声共存时，模型可能学到的是虚假的互补信息，导致 global、part、patch 之间语义错位。

2. 重定义动作：旧问题是“局部细节不够”；新问题是“多粒度特征之间缺少显式语义对齐”。这一步很强，因为它把多粒度从“多加几个特征”提升为“不同粒度之间需要保持同一个身份语义”。关键词是 explicit alignment、semantic consistency、multi-granularity collaborative constraint、center-constrained filtering、within-sample alignment。

3. 机制怎么长出来：机制围绕对齐展开。Transformer 输出 global、part、patch 三种粒度。多粒度 collaborative loss 把 RTMem 式 instance memory 和 cluster memory 扩展到三种粒度，让每种粒度都参与判别学习。constraint loss 不看样本间关系，而是在同一样本内部，用 part 特征和 global 排名选出语义一致的 patch token，把 part、patch、global 拉到同一个语义方向。聚类阶段也用语义一致的细粒度特征和 global 特征融合成距离矩阵。CCF 模块在 memory bank 初始化前，用 global cluster centroid 过滤偏离身份中心的局部特征。

4. 证据闭环：这篇证据闭环比较完整。消融从只用 baseline，到加多粒度判别损失、加约束损失、加约束聚类、加 CCF，逐步证明每一步都服务于对齐。CCF 过滤比例随训练下降，说明早期局部噪声多、后期特征变稳定；MSMT17 上过滤更强，也符合复杂数据更噪的说法。t-SNE、类内类间距离、rank-list、attention heatmap 都围绕“语义更一致、细节更准确”展开。

5. 审稿人为什么买账：这篇卖的是视角，而且机制绑定很紧。它不是泛泛说多粒度有效，而是说多粒度在无监督下会错位，所以要在损失、聚类、memory bank 三个阶段都做显式对齐。这个构造对我们最值得学：一个好问题重定义，会自然长出一组同向机制。

**6. Multi-Granularity Dynamic Hierarchical Graphs for Video-Based Person Re-Identification**

1. 触发观察：视频 ReID 里，很多方法只建模固定长度的全局时间线索。但人的不同身体部位运动和遮挡规律不同，腿部变化大，躯干更稳定，某些部位会在部分帧被遮挡后又出现。已有水平切分图方法虽然能做局部建模，但切分固定、邻居数量固定，容易连接低相关节点。

2. 重定义动作：旧问题是“怎么建模视频时间特征”；新问题是“怎么按身体部位自适应选择不同长度的时间线索”。关键词是 dynamic hierarchical graph、adaptive multi-granularity temporal clues、part-consistent graph、mutual nearest neighbor、similarity weighted inference。

3. 机制怎么长出来：DHGN 先把每帧特征图水平细分，再按特征相似度聚成自适应语义区域，而不是固定条带。每个语义层级跨帧建图，同一身体部位的节点才互连。边用 mutual nearest neighbor 建，避免低相关节点干扰。GC layer 用相似度加权聚合邻居信息，再用注意力在原节点和邻居信息之间重构残差特征。末端 GFC layer 建全局帧图，用 temporal graph pooling 排除干扰帧。SWIM 再用 gallery-gallery 相似度修正 query-gallery 相似度。

4. 证据闭环：消融非常贴合问题定义。GC、GC+attention、GFC、DHGN、SWIM 分别验证局部动态时间建模、注意力重构、全局时间建模和图库关系推理。粒度数、水平划分数、插入层位置、邻居 K1 和 K2 都直接对应“动态多粒度图是否必要”。节点相似度可视化最关键，它显示遮挡时模型会找长程相似帧，无遮挡时更偏短程运动线索。activation map 和 t-SNE 则证明特征更集中、更可分。

5. 审稿人为什么买账：这篇同时卖视角和机制。核心新意是把视频时间建模从固定全局序列，改成部位级、动态邻居、可变时间范围的图建模。SWIM 更像附加推理增强，真正支撑论文的是 DHGN 对“不同部位需要不同时间线索”的机制化落实。

最后看这批论文，共同套路很明确：作者先不急着造模块，而是把一个已有系统里的次级矛盾抬成主问题，比如 memory bank 更新、多粒度错位、衣服语义怎么用、视频时间线索固定。然后机制必须在多个位置反复服务同一句重定义，最后用消融和可视化证明“这个重定义本身是对的”，而不是只证明最终涨点。
